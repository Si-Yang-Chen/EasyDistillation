"""Shared pytest configuration for the EasyDistillation test suite."""

import pytest


def _cupy_available():
    """Check if CuPy import succeeds without triggering CUDARuntimeError."""
    try:
        import cupy  # noqa: F401
    except ImportError:
        return False
    # Even if importable, the GPU driver may be absent
    try:
        cupy.cuda.runtime.getDeviceCount()
    except Exception:
        return False
    return True


def _mpi_available():
    """Check if mpi4py + PyQuda are available."""
    try:
        import mpi4py  # noqa: F401
        import pyquda  # noqa: F401
    except ImportError:
        return False
    return True


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: tests requiring GPU (CuPy)",
    )
    config.addinivalue_line(
        "markers",
        "mpi: tests requiring MPI (mpi4py/PyQuda)",
    )
    config.addinivalue_line(
        "markers",
        "slow: tests that take >10 seconds",
    )
    config.addinivalue_line(
        "markers",
        "integration: tests requiring external data files",
    )


def pytest_collection_modifyitems(config, items):
    """Automatically skip GPU/MPI tests when dependencies are unavailable."""
    skip_gpu = pytest.mark.skip(reason="CuPy/GPU not available")
    skip_mpi = pytest.mark.skip(reason="mpi4py/PyQuda not available")

    gpu_available = _cupy_available()
    mpi_available = _mpi_available()

    for item in items:
        if not gpu_available and item.get_closest_marker("gpu"):
            item.add_marker(skip_gpu)
        if not mpi_available and item.get_closest_marker("mpi"):
            item.add_marker(skip_mpi)
