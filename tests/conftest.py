"""Shared pytest config: auto-skip ``@pytest.mark.gpu`` tests when no GPU is present."""
import pytest


def _gpu_available() -> bool:
    try:
        import gpusw

        return gpusw.gpu_available()
    except Exception:
        return False


HAS_GPU = _gpu_available()


def pytest_collection_modifyitems(config, items):
    if HAS_GPU:
        return
    skip = pytest.mark.skip(reason="no CUDA GPU / CuPy available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
