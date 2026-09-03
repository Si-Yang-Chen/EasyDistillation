# EasyDistillation

EasyDistillation is a Python framework for lattice QCD distillation calculations with CPU/GPU backends, perambulators, elementals and quark-diagram contraction.

## Minimal Development Flow

```bash
cd C:/Users/Lenovo/Project/lattice-flow-restart
python -m pytest -p no:cacheprovider -q \
  tests/test_current.py \
  tests/test_current_consumption.py \
  tests/test_temporal_current_elemental.py \
  tests/test_current_v2v_persistence.py \
  tests/test_current_v2v_contraction.py
python -m ruff format --check \
  lattice/current_elemental.py \
  lattice/generator/elemental.py \
  lattice/generator/sparsened_point.py \
  lattice/insertion/current.py \
  lattice/insertion/gauge_link.py \
  lattice/quark_diagram.py \
  tests/test_current_v2v_contraction.py \
  experiments/directed-current-v2v/contract_existing_vsv_pair.py
python -m ruff check --no-cache \
  lattice/current_elemental.py \
  lattice/generator/elemental.py \
  lattice/generator/sparsened_point.py \
  lattice/insertion/current.py \
  lattice/insertion/gauge_link.py \
  lattice/quark_diagram.py \
  tests/test_current_v2v_contraction.py \
  experiments/directed-current-v2v/contract_existing_vsv_pair.py
git diff --check
```

Run the deterministic synthetic precheck when changing current conservation logic:

```bash
PYTHONDONTWRITEBYTECODE=1 python test/current_conservation_cpu_precheck.py
```

Read [`AGENTS.md`](AGENTS.md) before Kunshan work, [`TASKBOARD.md`](TASKBOARD.md) for the current phase, and [`PLAN.md`](PLAN.md) for long-term work. The user handoff and archived deployment rules are indexed in [`DOCUMENTATION_INDEX.md`](DOCUMENTATION_INDEX.md).

## Documentation

- [Current handover](HANDOVER.md)
- [Current taskboard](TASKBOARD.md)
- [Long-term plan](PLAN.md)
- [Restart guide](RESTART.md)
- [Delivery inventory](INVENTORY.md)
- [Current API](docs/current-api.md)
- [Kunshan data map](docs/kunshan-easydistillation-data-map.md)
- [Documentation index](DOCUMENTATION_INDEX.md)

## Requirements

- Python >= 3.9
- NumPy, SciPy, opt_einsum and SymPy
- Optional: CuPy, PyQuda and QUDA for DCU execution

## License

MIT; see [LICENSE](LICENSE).
