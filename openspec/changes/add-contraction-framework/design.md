# Contraction Framework Design Document

## Context

Lattice QCD distillation calculations require efficient computation of correlation functions through tensor contractions. The pipeline flow is:

1. **Eigenvectors** (Laplace basis)
2. **Perambulators** (propagators between eigenvectors/point sources)
3. **Elementals** (gauge field matrix elements)
4. **Contractions** (correlation functions)

Currently, contraction calculations are embedded in `4.contraction.py` with logic tightly coupled to specific diagrams (meson, current). This limits reusability and makes testing difficult.

## Goals

- **Primary**: Create reusable, testable contraction framework for lattice hadron calculations
- **Secondary**: Support multiple diagram types (meson, current, baryon)
- **Tertiary**: Enable performance optimization and vectorization
- **Quaternary**: Provide clear I/O contracts for integration with analysis tools

## Non-Goals

- Rewrite existing diagram computation logic (preserve `quark_diagram.py`)
- Support exotic hadron types beyond meson/baryon
- Implement advanced diagrammatic techniques (GPD, TMDs)
- GPU-specific kernels (use CuPy for portability)

## Architecture

### Module Structure

```
lattice/contraction/
├── __init__.py              # Exports: meson, current, diagram
├── diagram.py               # Base utilities for tensor contractions
├── meson.py                 # Meson contraction (VV, PV, AV, etc.)
├── current.py               # Current insertion contractions
└── baryon.py               # [Future] Baryon contractions
```

### Key Components

#### 1. Base Utilities (`diagram.py`)

**Backend Abstraction**:
```python
def contract_tensors(
    tensors: List[ArrayLike],
    indices: str,  # 'ij,jk->ik' einsum notation
    backend: Optional[str] = None
) -> ArrayLike:
    """General tensor contraction wrapper supporting NumPy/CuPy"""
```

**Index Permutation**:
```python
def permute_indices(
    array: ArrayLike,
    old_order: Tuple[int, ...],
    new_order: Tuple[int, ...]
) -> ArrayLike:
    """Efficiently reorder array dimensions"""
```

**Memory Optimization**:
```python
def contiguous_copy(array: ArrayLike) -> ArrayLike:
    """Ensure contiguous memory layout for GPU efficiency"""
```

#### 2. Meson Contraction (`meson.py`)

**Vector-to-Vector (VV)**:
```python
def compute_meson_vv(
    elemental: np.ndarray,    # shape (Lt, Ne, Ne)
    perambulator: np.ndarray, # shape (Lt, Ns, Ns, Ne, Ne)
    backend: Optional[str] = None
) -> np.ndarray:
    """
    Compute meson 2-point correlator V->V
    
    Returns:
        shape (Lt,), dtype complex128
    """
```

**Pseudoscalar-to-Vector (PV)**:
```python
def compute_meson_pv(
    elemental: np.ndarray,
    perambulator: np.ndarray,
    gamma: int = 5,  # Gamma matrix type
    backend: Optional[str] = None
) -> np.ndarray:
    """
    Compute meson 2-point correlator P->V
    
    Returns:
        shape (Lt,), dtype complex128
    """
```

**Multi-time Diagrams**:
```python
def compute_diagrams_multitime(
    time_indices: List[Tuple[int, int]],
    correlator_times: np.ndarray,
    diagrams: List[Diagram],
    perambulators: List[np.ndarray],
    multitime_shape: bool = False
) -> np.ndarray:
    """
    Compute correlation functions at multiple time separations
    
    Returns:
        shape (len(diagrams), len(correlator_times), Lt), dtype complex128
    """
```

#### 3. Current Insertion (`current.py`)

**Vector Current**:
```python
def compute_current_insertion(
    eigenvectors: np.ndarray,        # shape (Lt, Ne, Lz, Ly, Lx, Nc)
    point_sources: np.ndarray,       # shape (Np, Lt, 3)
    gauge_field: np.ndarray,         # shape (Lt, Lz, Ly, Lx, Nd, Nc, Nc)
    current_type: str = "vector",    # "vector", "axial"
    derivative: str = "identity",    # Derivative operator
    backend: Optional[str] = None
) -> np.ndarray:
    """
    Compute current insertion with gauge field
    
    Returns:
        shape (Np, Lt), dtype complex128
    """
```

### Data Flow

```
User Input:
├── Perambulator arrays (VSV, PSV)
├── Elemental arrays (gauge field matrix elements)
└── Eigenvectors and point source coordinates

↓

Contraction Framework:
├── diagram.py:     Backend abstraction, tensor ops
├── meson.py:       Diagram-specific logic
└── current.py:     Current insertion logic

↓

Output:
├── Correlation functions (shape varies by diagram)
└── Complex128 arrays ready for analysis
```

## Design Decisions

### 1. Function-Based API

**Decision**: Use stateless functions rather than classes

**Reasoning**:
- Each contraction is independent
- No shared state between calls
- Easy to parallelize
- Aligns with NumPy/SciPy conventions

### 2. Backend Abstraction

**Decision**: Single `backend` parameter, auto-select from array type

```python
def _get_backend(array: ArrayLike) -> str:
    """Infer backend from array type (NumPy vs CuPy)"""
    if isinstance(array, np.ndarray):
        return "numpy"
    elif hasattr(array, '__cuda_array_interface__'):
        return "cupy"
    return "numpy"  # Default
```

**Reasoning**:
- Transparent GPU support
- No code changes needed for CPU/GPU switch
- User can override if needed

### 3. Einsum for Tensor Contractions

**Decision**: Use NumPy/CuPy `einsum()` for all contractions

**Example**:
```python
# Meson VV contraction (simplified)
# E[t, ne1, ne2] * P[t, ns1, ns2, ne2, ne3] * ... -> correlator[t]
einsum("tij,tklij->t", elemental, perambulator)
```

**Reasoning**:
- Portable (NumPy/CuPy have identical `einsum`)
- Optimizable by backend (auto-ordering)
- Clear mathematical expression
- Human-readable code

### 4. I/O Specifications

**Input Arrays**:
- Perambulators: `(Lt, Ns, Ns, Ne, Ne)` or `(Lt, Ns, Ns, Np, Ne)`
- Elementals: `(Lt, Ne, Ne)`
- Eigenvectors: `(Lt, Ne, Lz, Ly, Lx, Nc)`
- Point sources: `(Np, Lt, 3)` - coordinates only

**Output Arrays**:
- Correlators: `(Lt,)` for time separations
- Multi-time: `(n_diagrams, n_times, Lt)`
- Always `dtype=complex128`

## Performance Considerations

### Vectorization Strategy

1. **Batch Operations**: Use NumPy broadcasting for multiple diagrams
2. **Memory Layout**: Ensure contiguous arrays for GPU efficiency
3. **GPU Reduction**: Minimize CPU-GPU transfers
4. **Parallelization**: Time slices can be processed in parallel

### Expected Performance

- **2-3× speedup** vs naive Python loops through vectorization
- **5-10× GPU speedup** over CPU for large arrays
- **Memory overhead**: ~10% for temporary arrays

## Testing Strategy

### Unit Tests

1. **Correctness**: Compare with reference results [CLUSTER DATA]
2. **Shapes**: Verify output shapes and dtypes
3. **Backend Agreement**: NumPy and CuPy produce identical results
4. **Edge Cases**: Single point, full lattice, etc.

### Integration Tests

1. **Pipeline Integration**: Works with existing `quark_diagram.py`
2. **Backward Compatibility**: Script `4.contraction.py` still works
3. **Numerical Stability**: No NaN/Inf in results

### Performance Tests

1. **Speedup Benchmarks**: Measure vs naive loops
2. **Scaling**: Multi-GPU scaling analysis [CLUSTER]
3. **Memory Usage**: Peak memory during contraction

## Migration Path

### Step 1: New Module (No Breaking Changes)
- Create `lattice/contraction/` module
- Implement functions alongside existing code
- Both can coexist

### Step 2: Gradual Integration
- Update `4.contraction.py` to use new API
- Keep existing `quark_diagram.py` functional
- Add deprecation warnings (optional)

### Step 3: Full Integration
- All contraction calls through framework
- Optimized GPU kernels
- Production deployment

## Open Questions

1. **GPU Kernels**: Should we write CUDA kernels for specific contractions?
   - **Current Answer**: No, CuPy einsum is sufficient for now
   - **Future**: Can be added if profiling shows need

2. **Baryon Support**: Include baryon contractions in Phase 1?
   - **Current Answer**: No, focus on meson/current first
   - **Future**: Phase 5 extension

3. **Distributed Computing**: MPI support for multi-node?
   - **Current Answer**: Single-node multi-GPU only
   - **Future**: Can layer on top of this framework

## Dependencies

- NumPy: Array operations (CPU)
- CuPy: Array operations (GPU)
- opt_einsum: Optional, for contraction optimization
- PyQUDA: For perambulator data structures

## Success Criteria

- ✅ All unit tests passing
- ✅ Integration with pipeline working
- ✅ Documentation complete
- ✅ Performance validated on cluster
- ✅ Code review approved
