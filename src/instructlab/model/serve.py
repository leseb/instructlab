# SPDX-License-Identifier: Apache-2.0

# Standard
import subprocess
import sys
import time

# Third Party
import click

# First Party
from instructlab import configuration as config
from instructlab import log, utils
from instructlab.model.backends.server import ServerException, VllmServer, server


@click.command()
@click.option(
    "--model-path",
    type=click.Path(),
    default=config.DEFAULT_MODEL_PATH,
    show_default=True,
    help="Path to the model used during generation.",
)
@click.option(
    "--gpu-layers",
    type=click.INT,
    help="The number of layers to put on the GPU. The rest will be on the CPU. Defaults to -1 to move all to GPU.",
)
@click.option("--num-threads", type=click.INT, help="The number of CPU threads to use.")
@click.option(
    "--max-ctx-size",
    type=click.INT,
    help="The context size is the maximum number of tokens considered by the model, for both the prompt and response. Defaults to 4096.",
)
@click.option(
    "--model-family",
    type=str,
    help="Model family is used to specify which chat template to serve with",
)
@click.option(
    "--log-file",
    type=click.Path(),
    help="Log file path to write server logs to.",
)
@click.option(
    "--backend",
    type=click.Choice(["llama", "vllm"]),
    show_default=True,
    help=(
        "The backend to use for serving the model."
        "Automatically detected based on the model file properties."
        "Supported backends are: 'vllm' (on Linux only) and 'llama'."
    ),
)
@click.option(
    "--backend-args",
    type=str,
    help=(
        "Specific arguments to pass into the backend."
        "'--backend' must be specified with this flag."
        "e.g. --backend-args '--dtype auto --enable-lora'"
    ),
)
@click.pass_context
@utils.display_params
def serve(
    ctx,
    model_path,
    gpu_layers,
    num_threads,
    max_ctx_size,
    model_family,
    log_file,
    backend,
    backend_args,
):
    """Start a local server"""
    # pylint: disable=C0415

    # Redirect server stdout and stderr to the logger
    log.stdout_stderr_to_logger(ctx.obj.logger, log_file)
    # First Party
    from instructlab.model.backends import backends

    # If the backend was set using the --backend flag, validate it
    if backend is not None:
        try:
            backends.validate_backend(backend)
        except ValueError as e:
            click.secho(f"Failed to validate backend: {backend}", fg="red")
            raise click.exceptions.Exit(1)
    # Otherwise, determine the backend based on the model file
    # TODO: how to accomodate using the backend from the config file?
    else:
        try:
            backend = backends.determine_backend(model_path)
        except ValueError as e:
            click.secho(f"Failed to determine backend: {e}", fg="red")
            raise click.exceptions.Exit(1)

    ctx.obj.logger.info(f"Serving model '{model_path}' with {backend}")

    if backend == "vllm":
        # TODO: check flags, then serve_config for values of model_path, vllm_args
        # TODO: make sure all options in serve_config are also flags for model serve command
        # TODO: look into using VllmServer and LlamaCppServer classes here

        # Step 2: Instantiate VllmServer
        from instructlab.model.backends import vllm

        vllm_server = vllm.Server(
            logger=ctx.obj.logger,
            api_base="http://localhost:8000",
            model_path="/path/to/model",
            host="localhost",
            port=8000,
            vllm_args="--additional-args if-any",
        )

        # Step 3: Call the run() method with TLS configuration parameters
        vllm_server.run(
            tls_insecure=False,
            tls_client_cert="/path/to/client.crt",
            tls_client_key="/path/to/client.key",
            tls_client_passwd="password",
        )

        # vllm_cmd = [
        #     sys.executable,
        #     "-m",
        #     "vllm.entrypoints.openai.api_server",
        #     "--model",
        #     model_path,
        # ]
        # if backend_args is not None:
        #     vllm_cmd.extend(backend_args.split())

        # try:
        #     # TODO: below command gives no output to stdout, play around with stdout, stderr options for UX
        #     # vllm_process = subprocess.Popen(args=vllm_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #     vllm_process = subprocess.Popen(args=vllm_cmd)
        # except subprocess.CalledProcessError as err:
        #     ctx.obj.logger.debug(
        #         f"Vllm did not start properly. Exited with return code: {err.returncode}"
        #     )

        # TODO: look a into cleaner way to terminate vllm process
        # try:
        #     while True:
        #         time.sleep(1)
        # except KeyboardInterrupt:
        #     ctx.obj.logger.debug(f"Vllm terminated by user")
        #     vllm_process.terminate()
    else:
        ctx.obj.logger.info(
            f"Using model '{model_path}' with {gpu_layers} gpu-layers and {max_ctx_size} max context size."
        )

        # Step 2: Instantiate VllmServer
        from instructlab.model.backends import llama

        llama_server = llama.Server(
            logger=ctx.obj.logger,
            api_base="http://localhost:8000",
            model_path="/path/to/model",
            host="localhost",
            port=8000,
            vllm_args="--additional-args if-any",
        )

        # Step 3: Call the run() method with TLS configuration parameters
        llama_server.run(
            tls_insecure=False,
            tls_client_cert="/path/to/client.crt",
            tls_client_key="/path/to/client.key",
            tls_client_passwd="password",
        )

        # try:
        #     host = ctx.obj.config.serve.host_port.split(":")[0]
        #     port = int(ctx.obj.config.serve.host_port.split(":")[1])
        #     server(
        #         ctx.obj.logger,
        #         model_path,
        #         gpu_layers,
        #         max_ctx_size,
        #         model_family,
        #         num_threads,
        #         host,
        #         port,
        #     )
        # except ServerException as exc:
        #     click.secho(f"Error creating server: {exc}", fg="red")
        #     raise click.exceptions.Exit(1)
