"""Shared pytest config: auto-skip ``@pytest.mark.gpu`` (CUDA) and ``@pytest.mark.metal``
(Apple Silicon) tests when the corresponding GPU/backend is not present."""
import pytest


def _gpu_available() -> bool:
    try:
        import gpusw

        return gpusw.gpu_available()
    except Exception:
        return False


def _metal_available() -> bool:
    try:
        import gpusw

        return gpusw.metal_available()
    except Exception:
        return False


HAS_GPU = _gpu_available()
HAS_METAL = _metal_available()


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="no CUDA GPU / CuPy available")
    skip_metal = pytest.mark.skip(reason="no Apple-Silicon Metal GPU / PyObjC available")
    for item in items:
        if "gpu" in item.keywords and not HAS_GPU:
            item.add_marker(skip_gpu)
        if "metal" in item.keywords and not HAS_METAL:
            item.add_marker(skip_metal)
