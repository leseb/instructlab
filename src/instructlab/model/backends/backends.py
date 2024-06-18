# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC, abstractmethod
from pathlib import Path
import sys

supported_backends = ["llama", "vllm"]


class BackendServer(ABC):
    """Base class for a serving backend"""

    def __init__(self, logger, api_base, model_path, host, port):
        self.logger = logger
        self.api_base = api_base
        self.model_path = model_path
        self.host = host
        self.port = port

    @abstractmethod
    def run(self, tls_insecure, tls_client_cert, tls_client_key, tls_client_passwd):
        """Run serving backend"""
        pass

    @abstractmethod
    def shutdown(self):
        """Shutdown serving backend"""
        pass


def is_model_gguf(model_path: Path) -> bool:
    """
    Check if the file is a GGUF file.

    Args:
        file_path (Path): The path to the file.

    Returns:
        bool: True if the file is a GGUF file, False otherwise.
    """
    # Third Party
    import gguf

    try:
        gguf.GGUFReader(model_path, "r")
        return True
    except (ValueError, FileNotFoundError):
        return False

    return False


def validate_backend(backend: str) -> None:
    """
    Validate the backend.

    Args:
        backend (str): The backend to validate.

    Raises:
        ValueError: If the backend is not supported.
    """
    # lowercase backend for comparison in case of user input like 'vLLM'
    if backend.lower() not in supported_backends:
        raise ValueError(f"Backend '{backend}' is not supported.")

    # make sure vLLM is only used on Linux
    if sys.platform != "linux" and backend == "vllm":
        raise ValueError("vLLM only supports Linux platform (including WSL)")


def determine_backend(model_path: Path) -> str:
    """
    Determine the backend to use based on the model file properties.

    Args:
        model_path (Path): The path to the model file.

    Returns:
        str: The backend to use.
    """
    # Check if the model is a GGUF file
    try:
        is_gguf = is_model_gguf(model_path)
    except Exception as e:
        raise ValueError(f"Failed to determine whether the model is a GGUF format: {e}")

    if is_gguf:
        return "llama"
    else:
        if sys.platform != "linux":
            raise ValueError(
                f"The model file {model_path} is not a GGUF format so the backend was determined to be vLLM."
                "However vLLM only supports Linux platform (including WSL)"
            )

    return "vllm"
