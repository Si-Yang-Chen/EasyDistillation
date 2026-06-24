# Changelog

All notable changes to EasyDistillation will be documented in this file.

## [1.0] - 2026-06-09

### Project Infrastructure (Phase A)

- **pyproject.toml**: Added build system configuration with setuptools, pytest configuration
  with `pythonpath = ["."]`, and dependency management.
- **.pre-commit-config.yaml**: Added pre-commit hooks with ruff (linting + formatting)
  and isort (import sorting).
- **.github/workflows/ci.yml**: Added CI pipeline for automated testing.

### Code Quality (Phase B)

- Standardized import ordering across the entire codebase.
- Removed dead code, duplicate imports, and commented-out blocks.
- Applied consistent code formatting via ruff.
- Fixed type annotation issues and improved docstrings.

### Test Infrastructure (Phase C)

- **tests/conftest.py**: Added centralized pytest configuration with automatic GPU/MPI
  skip logic via `pytest_collection_modifyitems`.
- **test/conftest.py**: Added same configuration for `test/` integration tests.
- **pytest markers**: Defined `gpu`, `mpi`, `slow`, and `integration` markers.
  - `gpu`: Tests requiring CuPy/GPU (auto-skipped when unavailable).
  - `mpi`: Tests requiring mpi4py/PyQuda (auto-skipped when unavailable).
  - `slow`: Tests that take >10 seconds.
  - `integration`: Tests requiring external data files (auto-skipped in CI).
- **Removed sys.path boilerplate**: Eliminated `sys.path.insert(0, ...)` / `sys.path.append(...)`
  from all test files (~28 files across `tests/` and `test/`), made redundant by
  `pythonpath = ["."]` in pyproject.toml.
- **Removed unused imports**: Cleaned up dangling `import os` and `import sys` after
  sys.path removal.
- **Added pytestmark markers**: All test files now declare their requirements via
  `pytestmark = pytest.mark.{gpu,mpi,integration}`.
- **test/__init__.py**: Created package init for `test/` directory.
- **Fixed GPU module-level collection crashes**: Added pre-flight CuPy device check
  (`cupy.cuda.runtime.getDeviceCount()`) before any lattice imports in 8 script-style
  GPU test files. This prevents `CUDARuntimeError` during pytest collection when
  CuPy is installed but no GPU is available. Previously, `set_backend("cupy")`
  succeeded but subsequent `CurrentElementalGenerator(...)` triggered CUDA kernel
  compilation that crashed. The fix ensures clean `pytest.skip()` at module level
  before any CUDA-dependent code executes.
