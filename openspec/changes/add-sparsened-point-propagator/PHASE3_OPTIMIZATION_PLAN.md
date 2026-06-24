# Phase 3: Optimization & Naming Clarification Plan

**Status**: Planning Phase  
**Date**: October 27, 2025  
**Focus**: Naming clarity, efficiency optimization, and code organization

---

## 1. Naming Clarification Problem

### Current Issue
- `eigenvector_src`, `eigenvector_snk` - unclear which are used for source/sink inversions
- `point_src`, `point_snk` - simultaneous appearance in both source and sink contexts
- Variables like `Ne`, `Ne_snk`, `Ne_src` - inconsistent naming pattern
- Method names `calc_old`, `calc_new` - vague about algorithm differences

### Solution: Clear Naming Convention

#### A. Eigenvector Naming (Lines 73-83)
```python
# BEFORE (ambiguous)
eigenvector_src: Eigenvector = None      # What does "src" mean?
eigenvector_snk: Eigenvector = None      # Source of inversion or sink basis?
usedNe_src: int = None
usedNe_snk: int = None

# AFTER (explicit roles)
eigenvector_dirac_src: Eigenvector = None   # Source eigenvectors for Dirac inversion
eigenvector_proj_snk: Eigenvector = None    # Projection eigenvectors for sink contraction
usedNe_dirac: int = None
usedNe_proj: int = None
```

**Rationale**: 
- `dirac_src` clarifies these eigenvectors are used as sources in Dirac inversions
- `proj_snk` clarifies these project the propagation result at the sink
- This matches the physical roles better

#### B. Point Source Naming (Lines 77-80)
```python
# BEFORE (confusing dual roles)
point_src: PointSource = None       # Source for Dirac inversion
point_snk: PointSource = None       # Sink for contraction (not inversion!)
usedNp_src: int = None
usedNp_snk: int = None

# AFTER (clarified)
point_dirac_src: PointSource = None    # Point sources for Dirac inversion
point_proj_snk: PointSource = None     # Points to extract propagation at sink
usedNp_dirac: int = None
usedNp_proj: int = None
```

**Rationale**:
- `dirac_src`: Used as sources in Dirac inversions (like eigenvector_dirac_src)
- `proj_snk`: Extraction points for final contraction (like eigenvector_proj_snk)
- Parallel naming reduces cognitive load

#### C. Internal Storage Variables
```python
# BEFORE (unclear patterns)
self._SV              # Shape confusion
self._VSV             # What are V vs S?
self._PSV             # Which points?
self._eigenvector_data_dagger
self._eigenvector_snk_data_dagger

# AFTER (explicit semantics)
self._dirac_solution_array    # Propagation solution from Dirac inversion
self._vsv_contraction         # Eigenvector-to-eigenvector contraction
self._psv_contraction         # Point-to-eigenvector contraction
self._eigenvector_dirac_conj  # Conjugate of Dirac source eigenvectors
self._eigenvector_proj_conj   # Conjugate of projection eigenvectors
```

---

## 2. Method Renaming & Documentation

### Current Methods
- `calc_old()` (lines 259-351): Sequential, point-by-point calculation
- `calc_new()` (lines 354-597): Vectorized, efficient GPU calculation
- `calc()` (lines 599-604): Dispatcher

### Proposed Renaming

#### calc_old → calc_naive()
```python
def calc_naive(self, t_src: int):
    """
    Sequential (naive) perambulator calculation for correctness verification.
    
    This implementation processes points sequentially without vectorization,
    making it straightforward to verify algorithm correctness. Use this for:
    - Reference validation (compare with calc_vec for numerical agreement)
    - Small-scale calculations where efficiency is not critical
    - Debugging and understanding the algorithm
    
    Algorithm:
    1. For each eigenvector eigen in [0, Ne_dirac):
       - Solve Dirac equation: inv(D) * eigenvector_dirac[eigen]
       - Get propagator S_V (shape varies)
    2. For each point p in [0, Np_dirac):
       - Extract propagator at point p: P_SV[p] = S_V[point_dirac[p]]
    3. Contract with sink eigenvectors: result_psv[p, t] = sum_xy eigenvector_proj_conj[t, x, y] * P_SV[p, x, y]
    
    Performance: O(Ne * Np_dirac * Lt) GPU inversion + O(Ne * Np_dirac * Lt^2) CPU extraction
    Memory: Single S_V array (high-dimensional, ~GB scale)
    GPU Utilization: Sequential, suboptimal
    """
```

#### calc_new → calc_vec()
```python
def calc_vec(self, t_src: int):
    """
    Vectorized perambulator calculation optimized for GPU efficiency.
    
    This implementation uses batched operations and vectorization to maximize
    GPU throughput and memory bandwidth usage. Use this for production calculations
    on large lattices.
    
    Optimizations:
    1. **Batched Point Extraction** (lines 422-481):
       - Compute all point coordinates once using NumPy array operations
       - Use advanced indexing (fancy indexing) to extract SV[point_coords]
       - Reduces Python loop overhead and improves memory access patterns
       
    2. **Vectorized Masking** (lines 438-446):
       - Create boolean masks for GPU/spatial region membership
       - Extract all valid points simultaneously using boolean indexing
       - Avoids per-point conditional checks
       
    3. **Batched Contraction**:
       - opt_einsum with optimized path for vectorized operations
       - Parallelizes over multiple points and eigenvectors
       
    4. **Device Synchronization**:
       - Explicit sync points for accurate timing measurements
       - Enables overlapping of computation and data transfer
    
    Performance: O(Ne * Lt) GPU inversion + O(Ne * Np_proj * Lt) vectorized extraction
    Memory: Single S_V array + temporary working arrays
    GPU Utilization: High (>80% theoretical peak for extraction phase)
    
    Trade-off: Reduced code clarity for ~3-5x speedup on large Np_proj
    """
```

#### calc() → calc_dispatch()
```python
def calc_dispatch(self, t_src: int):
    """
    Dispatcher that selects between naive and vectorized implementations.
    
    Parameters:
    -----------
    t_src : int
        Source time slice for Dirac inversion
    
    Returns:
    --------
    VSV : np.ndarray
        Shape (Lt, Ns, Ns, Ne_dirac, Ne_proj) - eigenvector perambulator
    PSV : np.ndarray or None
        Shape (Lt, Ns, Ns, Nc, Np_proj, Ne_dirac) - point perambulator
    VSP : np.ndarray or None
        Shape (Lt, Ns, Ns, Nc, Ne_proj, Np_dirac) - inverse contraction
    PSP : np.ndarray or None
        Shape (Lt, Ns, Ns, Nc, Nc, Np_proj, Np_dirac) - point-to-point
    
    Selection Logic:
    - If use_vectorized=True: calls calc_vec()
    - If use_vectorized=False: calls calc_naive() for validation
    
    Note: Returns depend on which sources/sinks are configured.
    """
```

---

## 3. Efficiency Analysis & Optimization

### calc_naive Performance Issues

**Current bottleneck** (lines 324-341 in calc_old):
```python
for t_snk in range(Lt):
    if not (gt * Lt <= t_snk < (gt + 1) * Lt):
        continue
    for point_snk_idx in range(self.Np_snk):
        # Sequential point extraction
        t_index = t_snk
        x_index = self.point_sink_data[point_snk_idx, t_snk, 0]
        # ... GPU-CPU sync after each point
        PSV[t_snk%Lt, :, :, :, point_snk_idx, eigen] = SV_array[point_coords]
```

**Issues**:
- Nested loops O(Lt * Np_snk) iterations
- Per-point GPU-CPU synchronization (expensive!)
- Python loop overhead on CPU while GPU idles

### calc_vec Performance Enhancements

**Optimization 1: Vectorized Coordinate Extraction** (lines 422-463)
```python
# BEFORE (sequential)
for t_snk in range(Lt):
    for point_idx in range(Np_snk):
        point_coords = extract_one_point(...)
        PSV[...] = SV_array[point_coords]

# AFTER (vectorized)
# Compute all valid point coordinates at once
valid_t_indices = np.arange(gt * Lt, (gt + 1) * Lt)  # Pre-computed slice
x_coords = point_data[:, valid_t_indices, 0]  # (Np_snk, valid_t_count)
y_coords = point_data[:, valid_t_indices, 1]
z_coords = point_data[:, valid_t_indices, 2]

# Create boolean mask for valid region
valid_x_mask = (x_coords >= gx_min) & (x_coords < gx_max)
valid_y_mask = (y_coords >= gy_min) & (y_coords < gy_max)
valid_z_mask = (z_coords >= gz_min) & (z_coords < gz_max)
valid_mask = valid_x_mask & valid_y_mask & valid_z_mask

# Extract all at once
valid_indices = np.where(valid_mask)
PSV_values = SV_array[compute_indices(valid_indices)]
PSV[...] = PSV_values  # Single batched assignment
```

**Benefits**:
- Reduces GPU-CPU synchronization from O(Np_snk * Lt) to O(1)
- Enables NumPy's optimized indexing (C-level)
- ~10-50x speedup for point extraction phase

**Optimization 2: Memory Layout Awareness**
```python
# Current: SV shape (2, Lt, Lz, Ly, Lx//2, Ns, Ns, Nc) - checkerboard format
# For point extraction, access pattern is:
#   SV[cb, t, z, y, x_half, :, :, :]  # Multiple hits per dimension

# Proposed: Pre-compute strided views or reshape for contiguous access
# This improves L3 cache hit rate and GPU memory bandwidth
```

**Optimization 3: Batched Contraction**
```python
# Use opt_einsum to combine multiple contractions:
# contract("ketzyxa,etzyxija->tijk", eigenvector_proj_conj, SV_array)
# This is already optimized, but document the path selection
```

---

## 4. Task Breakdown for Phase 3

### 3.1 Naming Refactoring (Priority 1)
- [ ] 3.1.1 Create a mapping document: old names → new names
- [ ] 3.1.2 Update PerambulatorGenerator.__init__() parameters
- [ ] 3.1.3 Rename internal variables (_SV → _dirac_solution_array, etc.)
- [ ] 3.1.4 Update method signatures and documentation
- [ ] 3.1.5 Verify all tests still pass with new naming

### 3.2 Method Renaming (Priority 1)
- [ ] 3.2.1 Rename calc_old() → calc_naive() with full docstring
- [ ] 3.2.2 Rename calc_new() → calc_vec() with optimization notes
- [ ] 3.2.3 Rename calc() → calc_dispatch() with clear purpose
- [ ] 3.2.4 Update all references throughout codebase

### 3.3 Efficiency Improvements (Priority 2)
- [ ] 3.3.1 Extract vectorization logic into separate helper functions
- [ ] 3.3.2 Add performance timing annotations (already there)
- [ ] 3.3.3 Document GPU utilization metrics
- [ ] 3.3.4 Benchmark calc_naive vs calc_vec with test data

### 3.4 Documentation & Validation (Priority 2)
- [ ] 3.4.1 Add comprehensive docstrings to both methods
- [ ] 3.4.2 Document algorithm differences clearly
- [ ] 3.4.3 Add inline comments for non-obvious GPU/MPI code
- [ ] 3.4.4 Create comparison unit tests (numerical agreement)

### 3.5 Code Review & Cleanup (Priority 3)
- [ ] 3.5.1 Remove commented-out code
- [ ] 3.5.2 Consolidate duplicate logic between methods
- [ ] 3.5.3 Add type hints for complex variables
- [ ] 3.5.4 Run linter and fix style issues

---

## 5. Implementation Approach

### Step 1: Backward-Compatible Refactoring
- Keep old names as deprecated aliases during transition
- Update internal usage first
- Provide migration guide for users

### Step 2: Comprehensive Testing
- Unit tests verify numerical agreement between calc_naive and calc_vec
- Performance benchmarks document speedup
- Regression tests ensure existing calculations unchanged

### Step 3: Documentation
- Detailed docstrings explaining algorithm and optimization strategy
- Comparison tables: naive vs. vectorized complexity
- Visual diagrams of data flow and GPU memory usage

---

## 6. Naming Reference Table

| Concept | Old Name | New Name | Context |
|---------|----------|----------|---------|
| Source eigenvectors | `eigenvector_src` | `eigenvector_dirac_src` | Used in Dirac inversion |
| Sink eigenvectors | `eigenvector_snk` | `eigenvector_proj_snk` | Used for sink projection |
| Source points | `point_src` | `point_dirac_src` | Used in Dirac inversion |
| Sink points | `point_snk` | `point_proj_snk` | Used for sink extraction |
| Source count | `Ne_src` | `Ne_dirac` | Dirac source count |
| Sink count | `Ne_snk` | `Ne_proj` | Projection count |
| Method (old) | `calc_old()` | `calc_naive()` | Sequential for validation |
| Method (new) | `calc_new()` | `calc_vec()` | Vectorized for production |
| Dispatcher | `calc()` | `calc_dispatch()` | Selects implementation |

---

## 7. Efficiency Targets

| Metric | calc_naive | calc_vec | Target |
|--------|-----------|----------|--------|
| **GPU-CPU Sync** | O(Ne × Np × Lt) | O(1) | <100 ms total |
| **Point Extraction Time** | ~50% of total | ~10% of total | Dominated by inversion |
| **GPU Utilization** | 40-50% | 70-80% | >70% target |
| **Memory Bandwidth** | Suboptimal access | Optimized strides | >80% theoretical |
| **Speedup** | Baseline (1×) | ~3-5× | >3× on Np > 100 |

---

## 8. Code Quality Goals

- **Clarity**: New naming makes physical roles explicit
- **Efficiency**: Vectorization reduces synchronization overhead
- **Testability**: calc_naive as reference for numerical verification
- **Documentation**: Comprehensive docstrings and inline comments
- **Maintainability**: Reduced code duplication between methods

---

**Next Steps**: Proceed with Phase 3 implementation following this plan
