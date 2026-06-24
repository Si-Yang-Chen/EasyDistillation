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

See [Quick Reference](QUICK_REFERENCE.md) for a getting-started guide, API cheat sheet, and common operations.

## Documentation

| Document | Description |
|----------|-------------|
| [Quick Reference](QUICK_REFERENCE.md) | Fast lookup for common operations |
| [Project Architecture](PROJECT_ARCHITECTURE.md) | Full architecture and development guide |
| [Data Shapes](doc/README.md) | Data type shapes and file naming conventions |
| [Propagator Theory & Usage](doc/propagator_theory_and_usage.md) | Propagator types, memory estimates, loading examples |
| [Distillation Workflow](docs/DISTILLATION_WORKFLOW.md) | Traditional distillation step-by-step workflow |
| [Localized Blending](doc/localized_blending/localized_blending.md) | Theory, implementation, and appendix |
| [FileData Deep Dive](FILEDATA_DETAILED.md) | File I/O architecture and performance analysis |
| [Document Index](DOCUMENTATION_INDEX.md) | Full documentation index by scenario |
| [Vector Current Workflow](WORKFLOW_ANALYSIS.md) | Localized Blending two-point contraction workflow |

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
pytest tests/ -v                 # All tests
pytest tests/ -m "not gpu"       # Skip GPU tests
pytest tests/test_perambulator.py -v  # Specific test
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
