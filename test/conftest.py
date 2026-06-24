"""Pytest configuration for test/ directory."""
import pytest


def _cupy_available():
    try:
        import cupy
    except ImportError:
        return False
    try:
        cupy.cuda.runtime.getDeviceCount()
    except Exception:
        return False
    return True


def _mpi_available():
    try:
        import mpi4py
        import pyquda
    except ImportError:
        return False
    return True


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: tests requiring GPU (CuPy)")
    config.addinivalue_line("markers", "mpi: tests requiring MPI (mpi4py/PyQuda)")
    config.addinivalue_line("markers", "slow: tests that take >10 seconds")
    config.addinivalue_line("markers", "integration: tests requiring external data files")


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="CuPy/GPU not available")
    skip_mpi = pytest.mark.skip(reason="mpi4py/PyQuda not available")
    gpu_available = _cupy_available()
    mpi_available = _mpi_available()
    for item in items:
        if not gpu_available and item.get_closest_marker("gpu"):
            item.add_marker(skip_gpu)
        if not mpi_available and item.get_closest_marker("mpi"):
            item.add_marker(skip_mpi)
