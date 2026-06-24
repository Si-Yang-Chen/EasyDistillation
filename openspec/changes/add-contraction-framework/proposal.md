# Proposal: Add Contraction Framework for Lattice QCD Distillation

## Why

The current contraction calculation in `4.contraction.py` is implemented as a standalone script with embedded logic for specific use cases (meson, current diagrams). This creates several problems:

1. **Not reusable**: Cannot easily use for different hadron types or contraction patterns
2. **Hard to test**: Complex multi-time calculations are difficult to unit test
3. **Unclear flow**: The relationship between perambulators, elementals, and gauge fields is implicit
4. **Performance unknown**: No profiling or optimization data available
5. **Limited flexibility**: Adding new hadron types or operators requires script modification

We need to abstract contraction calculations into a reusable, testable framework that:
- Encapsulates diagram contraction logic (meson, baryon, current, etc.)
- Provides clear I/O specifications
- Supports both CPU and GPU backends
- Enables systematic testing and optimization

## What Changes

- **New module**: `lattice/contraction/` with specialized contraction implementations
  - `lattice/contraction/__init__.py` - Module exports
  - `lattice/contraction/meson.py` - Meson contraction (V→V, P→V, etc.)
  - `lattice/contraction/current.py` - Current insertion contractions
  - `lattice/contraction/diagram.py` - Generic diagram contraction utilities
  
- **Update existing files**:
  - `lattice/quark_diagram.py` - Refactor to use new contraction module (if needed)
  - `4.contraction.py` - Adapt to use contraction framework
  
- **Add comprehensive tests**:
  - `tests/test_contraction_meson.py` - Meson contraction tests
  - `tests/test_contraction_current.py` - Current contraction tests
  - `tests/test_contraction_diagram.py` - Generic diagram tests
  
- **Add documentation**:
  - `openspec/changes/add-contraction-framework/design.md` - Architecture and design decisions
  - Docstrings for all contraction functions
  
## Impact

- **Affected specs**: New `contraction` capability
- **Affected code**:
  - `lattice/contraction/` (new module, ~500-1000 lines)
  - `lattice/quark_diagram.py` (integration, ~100-200 lines)
  - `tests/test_contraction_*.py` (new test files, ~800-1200 lines)
  - `4.contraction.py` (integration, ~50 lines)
  
- **Breaking changes**: None initially (additions only)
- **Performance**: Vectorized contraction may provide 2-5× speedup over naive loops
- **Testing**: Requires reference data (placeholders for cluster testing)

## Acceptance Criteria

1. ✅ Contraction module structure designed and documented
2. ✅ Clear I/O specification for contraction functions
3. ✅ Design decisions documented in `design.md`
4. ⏳ Implementation skeleton created (Phase 1)
5. ⏳ Comprehensive tests with placeholders (Phase 2)
6. ⏳ Integration with existing pipeline (Phase 3)
7. ⏳ Performance validated on cluster (Phase 4)
