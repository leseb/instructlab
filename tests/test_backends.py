# Standard
from pathlib import Path

# Third Party
import pytest

# First Party
from instructlab import backends


# Mock function for is_model_gguf
def mock_is_model_gguf_true(*args, **kwargs):
    return True


def mock_is_model_gguf_false(*args, **kwargs):
    return False


@pytest.fixture
def model_path():
    return Path("/path/to/model")


def test_determine_backend_gguf(monkeypatch, model_path):
    # Mock is_model_gguf to return True
    monkeypatch.setattr("instructlab.backends.is_model_gguf", mock_is_model_gguf_true)
    assert (
        backends.determine_backend(model_path) == "llama"
    ), "Should return 'llama' for GGUF files"


def test_determine_backend_non_gguf_linux(monkeypatch, model_path):
    # Mock is_model_gguf to return False and sys.platform to "linux"
    monkeypatch.setattr("instructlab.backends.is_model_gguf", mock_is_model_gguf_false)
    monkeypatch.setattr("sys.platform", "linux")
    assert (
        backends.determine_backend(model_path) == "vllm"
    ), "Should return 'vllm' for non-GGUF files on Linux"


def test_determine_backend_non_gguf_non_linux(monkeypatch, model_path):
    # Mock is_model_gguf to return False and sys.platform to "win32"
    monkeypatch.setattr("instructlab.backends.is_model_gguf", mock_is_model_gguf_false)
    monkeypatch.setattr("sys.platform", "win32")
    with pytest.raises(ValueError) as excinfo:
        backends.determine_backend(model_path)
    assert "not a GGUF format" in str(
        excinfo.value
    ), "Should raise ValueError for non-GGUF files on non-Linux platforms"
