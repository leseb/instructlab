# SPDX-License-Identifier: Apache-2.0

# Standard
from time import sleep
import logging
import multiprocessing
import subprocess
import sys

# First Party
from instructlab import client

# Local
from .backends import BackendServer


class Server(BackendServer):
    def __init__(self, logger, api_base, model_path, host, port, backend_args):
        # self.logger = logger
        # self.api_base = api_base
        # self.model_path = model_path
        # self.host = host
        # self.port = port
        super().__init__(logger, api_base, model_path, host, port)
        self.backend_args = backend_args
        self.process = None
        self.timeout = 300

    def run(self, tls_insecure, tls_client_cert, tls_client_key, tls_client_passwd):
        """Start an OpenAI-compatible server with vllm"""
        vllm_cmd = [
            # TODO: should this run in uvicorn?
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--model",
            self.model_path,
        ]
        if self.vllm_args is not None:
            vllm_cmd.extend(self.vllm_args.split())

        self.logger.info(f"vllm serving command is: {vllm_cmd}")
        try:
            self.process = subprocess.Popen(
                args=vllm_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as err:
            self.logger.debug(
                f"Vllm did not start properly. Exited with return code: {err.returncode}"
            )

        count = 0
        self.logger.debug(f"Starting up vllm on {self.api_base}...")
        while count < self.timeout:
            sleep(1)
            try:
                client.list_models(
                    api_base=self.api_base,
                    tls_insecure=tls_insecure,
                    tls_client_cert=tls_client_cert,
                    tls_client_key=tls_client_key,
                    tls_client_passwd=tls_client_passwd,
                )
                self.logger.info(f"model at {self.model_path} served on vllm")
                break
            except client.ClientException:
                count += 1

    def shutdown(self):
        """Shutdown vllm server"""
        self.process.terminate()
