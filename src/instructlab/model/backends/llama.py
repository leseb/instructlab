# SPDX-License-Identifier: Apache-2.0

# Standard
from contextlib import redirect_stderr, redirect_stdout
import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import time

# Third Party
from llama_cpp import llama_chat_format
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import Settings
from uvicorn import Config
import llama_cpp.server.app as llama_app
import uvicorn

# First Party
# Assuming ClientException and other necessary imports are defined elsewhere
from instructlab import client

# Local
from ...client import ClientException, list_models
from ...configuration import get_api_base, get_model_family
from .backends import BackendServer

templates = [
    {
        "model": "merlinite",
        "template": "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n' + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}",
    },
    {
        "model": "mixtral",
        "template": "{{ bos_token }}\n{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '[INST] ' + message['content'] + ' [/INST]' }}\n{% elif message['role'] == 'assistant' %}\n{{ message['content'] + eos_token}}\n{% endif %}\n{% endfor %}",
    },
]


class ServerException(Exception):
    """An exception raised when serving the API."""


class UvicornServer(uvicorn.Server):
    """Override uvicorn.Server to handle SIGINT."""

    def handle_exit(self, sig, frame):
        # type: (int, Optional[FrameType]) -> None
        if not is_temp_server_running() or sig != signal.SIGINT:
            super().handle_exit(sig=sig, frame=frame)


class Server(BackendServer):
    def __init__(
        self,
        logger,
        api_base,
        model_path,
        host,
        port,
        gpu_layers,
        max_ctx_size,
        model_family,
    ):
        # self.logger = logger
        # self.api_base = api_base
        # self.model_path = model_path
        # self.host = host
        # self.port = port
        super().__init__(logger, api_base, model_path, host, port)
        self.gpu_layers = gpu_layers
        self.max_ctx_size = max_ctx_size
        self.model_family = model_family
        self.process = None
        self.queue = None

    def run(self, tls_insecure, tls_client_cert, tls_client_key, tls_client_passwd):
        """Start an OpenAI-compatible server with llama-cpp"""
        host_port = f"{self.host}:{self.port}"
        mpctx = multiprocessing.get_context(None)
        # use a queue to communicate between the main process and the server process
        self.queue = mpctx.Queue()
        # create a temporary, throw-away logger
        server_logger = logging.getLogger(host_port)
        server_logger.setLevel(logging.FATAL)
        self.logger.debug(f"Starting up llama-cpp on {self.api_base}...")
        self.process = mpctx.Process(
            target=server,
            kwargs={
                "logger": server_logger,
                "model_path": self.model_path,
                "gpu_layers": self.gpu_layers,
                "max_ctx_size": self.max_ctx_size,
                "model_family": self.model_family,
                "port": self.port,
                "host": self.host,
                "queue": self.queue,
            },
        )

        self.process.start()

        # in case the server takes some time to fail we wait a bit
        count = 0
        while self.process.is_alive():
            time.sleep(0.1)
            try:
                client.list_models(
                    api_base=self.api_base,
                    tls_insecure=tls_insecure,
                    tls_client_cert=tls_client_cert,
                    tls_client_key=tls_client_key,
                    tls_client_passwd=tls_client_passwd,
                )
                break
            except client.ClientException:
                pass
            if count > 50:
                self.logger.error("failed to reach the API server")
                break
            count += 1

        # if the queue is not empty it means the server failed to start
        if not self.queue.empty():
            # pylint: disable=raise-missing-from
            raise self.queue.get()

    def shutdown(self):
        """Clean up llama-cpp"""
        if self.process and self.queue:
            self.process.terminate()
            self.process.join(timeout=30)
            self.queue.close()
            self.queue.join_thread()


def server(
    logger,
    model_path,
    gpu_layers,
    max_ctx_size,
    model_family,
    threads=None,
    host="localhost",
    port=8000,
    queue=None,
):
    """Start OpenAI-compatible server"""
    settings = Settings(
        host=host,
        port=port,
        model=model_path,
        n_ctx=max_ctx_size,
        n_gpu_layers=gpu_layers,
        verbose=logger.level == logging.DEBUG,
    )
    if threads is not None:
        settings.n_threads = threads
    try:
        app = create_app(settings=settings)

        @app.get("/")
        def read_root():
            return {
                "message": "Hello from InstructLab! Visit us at https://instructlab.ai"
            }
    except ValueError as exc:
        if queue:
            queue.put(exc)
            queue.close()
            queue.join_thread()
            return
        raise ServerException(f"failed creating the server application: {exc}") from exc

    template = ""
    eos_token = "<|endoftext|>"
    bos_token = ""
    for template_dict in templates:
        if template_dict["model"] == get_model_family(model_family, model_path):
            template = template_dict["template"]
            if template_dict["model"] == "mixtral":
                eos_token = "</s>"
                bos_token = "<s>"
    try:
        llama_app._llama_proxy._current_model.chat_handler = (
            llama_chat_format.Jinja2ChatFormatter(
                template=template,
                eos_token=eos_token,
                bos_token=bos_token,
            ).to_chat_handler()
        )
    # pylint: disable=broad-exception-caught
    except Exception as exc:
        if queue:
            queue.put(exc)
            queue.close()
            queue.join_thread()
            return
        raise ServerException(f"failed creating the server application: {exc}") from exc

    logger.info("Starting server process, press CTRL+C to shutdown server...")
    logger.info(
        f"After application startup complete see http://{host}:{port}/docs for API."
    )

    config = Config(
        app,
        host=host,
        port=port,
        log_level=logging.ERROR,
        limit_concurrency=2,  # Make sure we only serve a single client at a time
        timeout_keep_alive=0,  # prevent clients holding connections open (we only have 1)
    )
    s = UvicornServer(config)

    # If this is not the main process, this is the temp server process that ran in the background
    # after `ilab chat` was executed.
    # In this case, we want to redirect stdout to null to avoid cluttering the chat with messages
    # returned by the server.
    if is_temp_server_running():
        # TODO: redirect temp server logs to a file instead of hidding the logs completely
        # Redirect stdout and stderr to null
        with (
            open(os.devnull, "w", encoding="utf-8") as f,
            redirect_stdout(f),
            redirect_stderr(f),
        ):
            s.run()
    else:
        s.run()

    if queue:
        queue.close()
        queue.join_thread()


def is_temp_server_running():
    """Check if the temp server is running."""
    return multiprocessing.current_process().name != "MainProcess"
