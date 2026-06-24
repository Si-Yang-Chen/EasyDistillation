# Contraction Framework Proposal - README

**Status**: 📋 **PROPOSAL CREATED** (Awaiting Review)  
**Date**: October 27, 2025  
**Target**: Phase 1 Design & Architecture

---

## Quick Overview

This proposal introduces a **modular, reusable contraction framework** for lattice QCD distillation calculations. Currently, contractions are embedded in a standalone script; this change abstracts them into testable, composable functions.

### Why This Matters

| Problem | Solution |
|---------|----------|
| Not reusable | Separate `lattice/contraction/` module |
| Hard to test | Unit tests with [PLACEHOLDER] reference data |
| Unclear flow | Clear I/O specs and docstrings |
| Performance unknown | Benchmarks & profiling framework |
| Limited flexibility | Support multiple diagram types (meson, current, baryon) |

---

## Proposal Structure

### Files Created

1. **proposal.md** - Executive summary (Why, What Changes, Impact)
2. **tasks.md** - Implementation checklist (5 phases, 50+ tasks)
3. **design.md** - Architecture & design decisions
4. **README.md** - This file

### Key Sections

#### proposal.md
- **Why**: Current limitations and problems
- **What Changes**: New modules, updated files, test files
- **Impact**: Affected code, breaking changes, performance expectations
- **Acceptance Criteria**: 7 clear checkpoints

#### design.md
- **Architecture**: Module structure and components
- **Design Decisions**: 4 key decisions with reasoning
  1. Function-based API (not classes)
  2. Backend abstraction (NumPy/CuPy)
  3. Einsum for contractions
  4. Clear I/O specifications
- **Data Flow**: User input → Framework → Output
- **Performance**: Expected 2-3× speedup through vectorization
- **Testing Strategy**: Unit, integration, performance tests
- **Migration Path**: 3-step rollout plan

#### tasks.md
- **Phase 1**: Design & Architecture (3 sections, 13 tasks)
  - Framework architecture design
  - Create module skeleton
  - Define API contracts
  
- **Phase 2**: Core Implementation (3 sections, 9 tasks)
  - Implement base utilities
  - Implement meson contraction
  - Implement current insertion
  
- **Phase 3**: Testing (3 sections, 9 tasks)
  - Unit tests for meson/current/utilities
  - Numerical validation [PLACEHOLDER]
  - Performance benchmarks [PLACEHOLDER]
  
- **Phase 4**: Integration (3 sections, 6 tasks)
  - Integrate with pipeline
  - Complete documentation
  - Validation testing
  
- **Phase 5**: Optimization & Deployment (2 sections, 5 tasks)
  - Optional performance optimization
  - Final validation

---

## Module Architecture

### Directory Structure

```
lattice/contraction/
├── __init__.py              # Exports: meson, current, diagram
├── diagram.py               # Base utilities (backend abstraction, tensor ops)
├── meson.py                 # Meson contraction (VV, PV, AV)
├── current.py               # Current insertion contractions
└── baryon.py               # [Future Phase 5] Baryon contractions
```

### Core Functions

**Base Utilities** (`diagram.py`):
```python
def contract_tensors(tensors, indices, backend=None) -> Array
def permute_indices(array, old_order, new_order) -> Array
def contiguous_copy(array) -> Array
```

**Meson Contraction** (`meson.py`):
```python
def compute_meson_vv(elemental, perambulator, backend=None) -> Array
def compute_meson_pv(elemental, perambulator, gamma=5, backend=None) -> Array
def compute_diagrams_multitime(time_indices, correlator_times, ...) -> Array
```

**Current Insertion** (`current.py`):
```python
def compute_current_insertion(
    eigenvectors, point_sources, gauge_field,
    current_type="vector", derivative="identity", backend=None
) -> Array
```

---

## Design Highlights

### 1. Function-Based API
- Stateless functions (no shared state)
- Easy to parallelize
- Aligns with NumPy/SciPy conventions
- Independent diagram calculations

### 2. Transparent Backend Support
```python
# Works with NumPy arrays
result = compute_meson_vv(elemental_cpu, perambulator_cpu)

# Works with CuPy arrays (same code)
result = compute_meson_vv(elemental_gpu, perambulator_gpu)

# Explicit backend selection (optional)
result = compute_meson_vv(..., backend="cupy")
```

### 3. Einsum-Based Contractions
```python
# Readable mathematical expression
einsum("tij,tklij->t", elemental, perambulator)
# Automatically optimized by NumPy/CuPy
```

### 4. Clear I/O Specifications
- **Input**: Perambulators `(Lt, Ns, Ns, Ne, Ne)`, Elementals `(Lt, Ne, Ne)`
- **Output**: Correlators `(Lt,)`, always `dtype=complex128`

---

## Performance Expectations

| Metric | Expectation |
|--------|-------------|
| Vectorization Speedup | 2-3× vs Python loops |
| GPU Speedup | 5-10× vs CPU |
| Memory Overhead | ~10% for temp arrays |
| Scaling | Multi-GPU ready (Phase 4) |

---

## Implementation Timeline

### Phase 1: Design (Current)
- ✅ Architecture designed
- ✅ Module structure specified
- ✅ Functions signatures documented
- ⏳ **Next**: Implementation kickoff

### Phase 2-3: Implementation & Testing
- ⏳ Base utilities implementation
- ⏳ Meson/current contractions
- ⏳ Unit tests (with [PLACEHOLDER] reference data)
- ⏳ Performance benchmarks

### Phase 4: Integration
- ⏳ Pipeline integration
- ⏳ Backward compatibility validation
- ⏳ Cluster performance testing

### Phase 5: Optimization (Optional)
- ⏳ Performance profiling
- ⏳ GPU kernel optimization
- ⏳ Scaling analysis

---

## Testing Strategy

### Unit Tests (Phase 3)

**Meson Tests** (`test_contraction_meson.py`):
- VV contraction correctness [PLACEHOLDER]
- PV contraction correctness [PLACEHOLDER]
- Output shapes and dtypes
- Backend agreement (NumPy vs CuPy)

**Current Tests** (`test_contraction_current.py`):
- Vector current correctness [PLACEHOLDER]
- Axial current correctness [PLACEHOLDER]
- Derivative operators
- Point source handling

**Utility Tests** (`test_contraction_diagram.py`):
- Tensor contraction wrapper
- Index permutations
- Backend abstraction
- Performance benchmarks [PLACEHOLDER]

### Integration Tests (Phase 4)
- Pipeline compatibility
- Backward compatibility with `4.contraction.py`
- Numerical stability
- Cluster performance [CLUSTER]

---

## Placeholders for Future Work

The following items require **cluster testing** with actual lattice data:

| Item | Type | Phase |
|------|------|-------|
| Meson VV reference data | Numerical | Phase 3 |
| Meson PV reference data | Numerical | Phase 3 |
| Current insertion reference data | Numerical | Phase 3 |
| Performance benchmarks | Timing | Phase 3 |
| Scaling tests (multi-GPU) | Performance | Phase 4 |

---

## Next Steps

### Approval Phase
1. **Review proposal.md** for scope and impact
2. **Review design.md** for architecture
3. **Review tasks.md** for implementation plan
4. **Approve** or request changes

### Implementation Phase (After Approval)
1. **Phase 1**: Create module skeleton & function signatures
2. **Phase 2**: Implement meson & current contractions
3. **Phase 3**: Write unit tests with placeholders
4. **Phase 4**: Integrate with pipeline, cluster testing
5. **Phase 5** (Optional): Performance optimization

---

## Key Decision Points

### Q1: Function vs Class?
**A**: Functions. Each contraction is independent; classes add unnecessary overhead.

### Q2: Where to run tests?
**A**: CPU (NumPy) for development, cluster (CuPy) for validation and benchmarks.

### Q3: GPU kernels needed?
**A**: Not initially. CuPy einsum is sufficient; can add custom kernels in Phase 5 if profiling shows need.

### Q4: Support baryon?
**A**: Phase 5+ extension. Focus on meson/current first (most common use cases).

---

## Success Criteria

- ✅ Architecture documented and approved
- ⏳ All unit tests passing
- ⏳ Integration with pipeline complete
- ⏳ Performance validated on cluster
- ⏳ Code reviewed and merged

---

## Related Changes

- **Phase 1-2** (Completed): Sparsened point generation framework
- **Phase 3** (Completed): PerambulatorGenerator documentation & optimization
- **Phase 4** (Current): Contraction framework proposal

---

## Questions & Contact

For questions about this proposal:
1. Review **design.md** for technical details
2. Check **tasks.md** for implementation scope
3. Reference **proposal.md** for impact assessment
