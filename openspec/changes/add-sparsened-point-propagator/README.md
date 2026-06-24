# OpenSpec Proposal: Add Sparsened Point Generator & Propagator Enhancements

**Change ID**: `add-sparsened-point-propagator`  
**Status**: Proposal (awaiting review and approval)  
**Created**: October 27, 2025  
**Complexity**: Medium (multi-component refactoring with no breaking changes)

## Overview

This proposal refactors the EasyDistillation distillation pipeline to extract sparsened point generation into a reusable `SparsenedPointGenerator` class and enhances documentation and testing for the `PerambulatorGenerator`'s point source/sink calculations.

## Quick Summary

| Aspect | Detail |
|--------|--------|
| **Primary Goal** | Extract sparsened point generation from script into a reusable, testable function |
| **Secondary Goal** | Validate and document PerambulatorGenerator point source calculations |
| **Files Created** | `proposal.md`, `tasks.md`, `design.md`, 2 spec deltas |
| **Impact** | New function + modified perambulator documentation (no breaking changes) |
| **Effort** | ~5 phases, 15+ tasks |
| **Risk** | Low (additions only; existing code unchanged) |

## Key Deliverables

### 1. generate_sparsened_points Function (`lattice/generator/sparsened_point.py`)
```python
def generate_sparsened_points(
    latt_size: List[int],
    num_points: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate random sparsened points on a lattice.
    
    Args:
        latt_size: [Lx, Ly, Lz, Lt] lattice dimensions
        num_points: Number of spatial points Np per time slice
        seed: Random seed for reproducibility (None = random)
        
    Returns:
        np.ndarray of shape (Np, Lt, 3) with dtype=np.int32
        coords[p, t, :] = [x, y, z] with unique coords per time slice
    """
```

**Input/Output Specification**:
- **Inputs**:
  - `latt_size`: List[int], exactly 4 elements [Lx, Ly, Lz, Lt]
  - `num_points`: int, 0 < Np ≤ Lx×Ly×Lz
  - `seed`: Optional[int], None for random or integer for reproducible sequence

- **Output**:
  - NumPy array, shape (Np, Lt, 3), dtype=np.int32
  - coords[p, t, 0] = x ∈ [0, Lx)
  - coords[p, t, 1] = y ∈ [0, Ly)
  - coords[p, t, 2] = z ∈ [0, Lz)
  - For each fixed t, all Np coordinates are spatially distinct

### 2. PerambulatorGenerator Enhancements
- Review existing point source/sink support in `calc_old()` and `calc_new()` methods
- Add comprehensive docstrings explaining:
  - PSV (point-source to eigenvector) calculation
  - PSP (point-source to point-sink) calculation
  - GPU checkerboard indexing
  - MPI grid awareness
- Document array shapes and GPU memory requirements

### 3. Comprehensive Testing
- **Unit tests**: `tests/test_sparsened_point.py`
  - Point uniqueness per time slice
  - Seed reproducibility
  - Boundary conditions
  - Save/load round-trip
  
- **Integration tests**: `tests/test_perambulator_point_sources.py`
  - PSV calculation validation
  - PSP calculation validation
  - Vectorized vs sequential consistency
  - GPU memory efficiency

## Architecture Decisions

See `design.md` for detailed rationale. Key decisions:

1. **Location**: New class in `lattice/generator/sparsened_point.py` (follows existing patterns)
2. **API**: Stateless generator returning NumPy arrays; caller controls storage
3. **Seeding**: Optional per-call seed for flexibility
4. **PerambulatorGenerator**: No refactoring; documentation + testing validates correctness
5. **Testing**: Unit tests for pure Python; integration tests for GPU code

## Affected Components

```
lattice/
├── generator/
│   ├── __init__.py (modified: add SparsenedPointGenerator export)
│   ├── sparsened_point.py (new)
│   └── perambulator.py (review + enhanced docstrings)
├── preset.py (review: ensure PointSource compatibility)
└── ...

tests/
├── test_sparsened_point.py (new)
├── test_perambulator_point_sources.py (new)
└── ...

examples/
└── gen_sparsened_field_refactored.py (optional: show new usage pattern)
```

## Implementation Phases

1. **Phase 1**: Create SparsenedPointGenerator class + module exports
2. **Phase 2**: Unit tests for SparsenedPointGenerator
3. **Phase 3**: Review & document PerambulatorGenerator
4. **Phase 4**: Integration tests for PerambulatorGenerator
5. **Phase 5**: Update preset.py if needed
6. **Phase 6**: Final validation and documentation

See `tasks.md` for detailed checklist.

## Spec Changes

### New Spec: `generators`
- **Capability**: Sparsened Point Generation
- **Requirement 1**: SparsenedPointGenerator for lattice point sampling
- **Requirement 2**: Generator module exports

### Modified Spec: `perambulator`
- Enhanced point source calculation documentation
- Clarified parameter roles and return values
- GPU memory requirements noted

## Validation

✅ **OpenSpec Validation**: Passed strict checks
```
openspec validate add-sparsened-point-propagator --strict
→ Change 'add-sparsened-point-propagator' is valid
```

## Next Steps

1. **Review**: Team reviews proposal content (this document + linked specs)
2. **Approval**: Stakeholder approval before implementation begins
3. **Implementation**: Follow `tasks.md` checklist in order
4. **Validation**: Run all tests; confirm >80% coverage for new code
5. **Archive**: After completion, archive change in OpenSpec

## Files in This Change

```
openspec/changes/add-sparsened-point-propagator/
├── README.md (this file)
├── proposal.md (why, what, impact)
├── design.md (architectural decisions, risks, migration plan)
├── tasks.md (implementation checklist, 6 phases)
├── PHASE1_IMPLEMENTATION.md (Phase 1 completion record)
├── PHASE3_REVIEW.md (calc_old vs calc_new performance analysis)
├── PHASE3_OPTIMIZATION_PLAN.md (naming & optimization plan)
├── PHASE3_COMPLETE.md (Phase 3 completion record)
└── specs/
    ├── generators/
    │   └── spec.md (ADDED: SparsenedPointGenerator requirements)
    └── perambulator/
        └── spec.md (MODIFIED: Point source documentation)
```

## Backward Compatibility

- ✅ No breaking changes
- ✅ Existing scripts (`1.gen_sparsened_field.py`) continue to work
- ✅ New class is purely additive
- ✅ Optional migration path for users (use new class when ready)

## Questions & Feedback

For questions on this proposal, refer to:
- **Why and What**: `proposal.md`
- **Technical Details**: `design.md`
- **Specifications**: `specs/generators/spec.md`, `specs/perambulator/spec.md`
- **Implementation Plan**: `tasks.md`

---

**Proposal Created By**: AI Assistant  
**Status**: Ready for Review  
**Last Updated**: October 27, 2025
