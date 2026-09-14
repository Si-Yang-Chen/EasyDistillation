# EasyDistillation

EasyDistillation is a Python framework for lattice QCD distillation calculations — integrating the generation of Laplacian eigenvectors, perambulators, elementals, and automatic quark diagram contraction with multi-backend support (CPU / GPU).

## Features

- **Multi-backend**: NumPy (CPU) or CuPy/PyQuda (GPU) via unified backend interface
- **Lazy loading & memory mapping**: Efficient FileData system for large lattice datasets
- **Full propagator support**: V2V (Perambulator), P2V, V2P, P2P with time-sliced file formats
- **Automatic contraction**: opt_einsum-optimized quark diagram contraction with symbolic SymPy-based simplification
- **Symmetry analysis**: Built-in group theory support for hadron operator construction
- **High-mode projection**: Localized Blending method for point-source sampling
- **MPI-ready**: Distributed parallelism support

## Quick Start

The API is documented through the package exports and the tests under `test/`:

```python
from lattice import Dispatch, set_backend, get_backend
```

## Documentation

The previous documentation set is archived read-only under
[`archive/docs/`](archive/docs/ARCHIVE_NOTE.md). It was written against an earlier
code layout and is **known to be out of date** — see the archive note for specifics.
A replacement set has not been written yet.

The only live documents are this README and [CHANGELOG.md](CHANGELOG.md).

## Requirements

- Python ≥ 3.8
- NumPy, SciPy
- [opt_einsum](https://github.com/dgasmith/opt_einsum)
- [SymPy](https://www.sympy.org/) (for symbolic simplification)

**Optional** (GPU acceleration):
- [CuPy](https://cupy.dev/)
- [PyQuda](https://github.com/IHEP-LQCD/PyQuda) + QUDA

## Running Tests

```bash
pytest test/ -v                 # All tests
pytest test/ -m "not gpu"       # Skip GPU tests
pytest test/test_perambulator.py -v  # Specific test
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
