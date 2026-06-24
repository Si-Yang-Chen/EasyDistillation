## Why

The current implementation of sparsened point generation is embedded within a standalone script (`1.gen_sparsened_field.py`), making it difficult to reuse, test, and maintain. The perambulator calculation with point sources needs to be properly abstracted and validated with unit tests to ensure correctness and reliability for lattice QCD distillation calculations.

## What Changes

- **Extract sparsened point generation logic** from `1.gen_sparsened_field.py` into a simple function `generate_sparsened_points()` in the generator module
- **Enhance PerambulatorGenerator** documentation for point sources and sinks with clear docstrings
- **Add comprehensive unit tests** for the generation function to validate correctness
- **Update lattice.preset** to support PointSource data loading and manipulation
- **Refactor generator/__init__.py** to export the new function

## Impact

- **Affected specs**: generators (new), perambulator (modified)
- **Affected code**: 
  - `lattice/generator/sparsened_point.py` (new, single function)
  - `lattice/generator/__init__.py` (modified)
  - `lattice/generator/perambulator.py` (review + documentation)
  - `lattice/preset.py` (modified to support PointSource)
  - `tests/test_sparsened_point.py` (new)
  - `tests/test_perambulator_point_sources.py` (new)
- **Breaking changes**: None (additions only)
- **Migration impact**: Script users can import `generate_sparsened_points()` instead of running standalone scripts
