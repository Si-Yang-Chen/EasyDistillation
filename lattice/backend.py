from typing import Literal, List
import logging

logger = logging.getLogger(__name__)

_BACKEND = None
PYQUDA = None


def get_backend():
    global _BACKEND
    if _BACKEND is None:
        set_backend("numpy")
    return _BACKEND


def set_backend(backend: Literal["numpy", "cupy"]):
    global _BACKEND
    if not isinstance(backend, str):
        backend = backend.__name__
    backend = backend.lower()
    assert backend in ["numpy", "cupy"]
    if backend == "numpy":
        import numpy

        _BACKEND = numpy
    elif backend == "cupy":
        import cupy

        _BACKEND = cupy
    # elif backend == "torch":
    #     import torch
    #     torch.set_default_device("cuda")
    #     _BACKEND = torch
    else:
        raise ValueError(R'backend must be "numpy", "cupy" or "torch"')


def check_QUDA(
    grid_size: List[int] = None,
    backend: Literal["cupy", "torch"] = "cupy",
    resource_path: str = None,
):
    global PYQUDA
    if PYQUDA is None:
        try:
            import pyquda

            pyquda.init(grid_size, backend=backend, resource_path=resource_path)
            logger.info("PyQUDA installed in: %s", pyquda.__file__)
            from packaging.version import Version

            if Version(pyquda.__version__) < Version("0.9.0"):
                raise ImportError(
                    f"PyQuda version {pyquda.__version__} < Required 0.9.X"
                )

        except ImportError as e:
            logger.warning("ImportError: %s", e)
        except RuntimeError as e:
            logger.warning("RuntimeError: %s", e)
        else:
            PYQUDA = True
    if PYQUDA is None:
        PYQUDA = False

    return PYQUDA


def log_gpu_memory(tag: str) -> None:
    """
    Log current GPU memory usage (CuPy) or CPU memory usage (NumPy).

    This function attempts to query GPU memory information if using CuPy backend,
    or CPU memory information if using NumPy backend.

    Args:
        tag: A string tag to identify the memory log entry (e.g., "load_data", "compute_overlap")

    Example:
        >>> from lattice.backend import log_gpu_memory
        >>> log_gpu_memory("before_computation")
        [GPU MEM][before_computation] used=2.345GB free=5.678GB total=8.023GB
        # or for NumPy:
        [CPU MEM][before_computation] rss=1.234GB vms=2.345GB percent=12.3%
    """
    try:
        backend = get_backend()
        backend_name = backend.__name__

        if backend_name == "cupy":
            # Query GPU memory for CuPy
            free_bytes, total_bytes = backend.cuda.runtime.memGetInfo()
            used_bytes = total_bytes - free_bytes
            logger.info(
                "[GPU MEM][%s] used=%.3fGB free=%.3fGB total=%.3fGB",
                tag, used_bytes / 1024 ** 3, free_bytes / 1024 ** 3, total_bytes / 1024 ** 3,
            )
        elif backend_name == "numpy":
            # Query CPU memory for NumPy
            try:
                import psutil
                import os

                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                rss_bytes = mem_info.rss  # Resident Set Size (actual physical memory)
                vms_bytes = mem_info.vms  # Virtual Memory Size
                mem_percent = process.memory_percent()

                logger.info(
                    "[CPU MEM][%s] rss=%.3fGB vms=%.3fGB percent=%.1f%%",
                    tag, rss_bytes / 1024 ** 3, vms_bytes / 1024 ** 3, mem_percent,
                )
            except ImportError:
                # psutil not available, try alternative method using sys.getsizeof
                # This is less accurate but doesn't require external dependencies
                import sys
                import gc

                total_size = 0
                for obj in gc.get_objects():
                    try:
                        total_size += sys.getsizeof(obj)
                    except Exception:
                        pass

                logger.info(
                    "[CPU MEM][%s] approximate_size=%.3fGB (note: install psutil for more accurate memory monitoring)",
                    tag, total_size / 1024 ** 3,
                )
    except Exception as err:
        logger.warning("[MEM][%s] query failed: %s", tag, err)
