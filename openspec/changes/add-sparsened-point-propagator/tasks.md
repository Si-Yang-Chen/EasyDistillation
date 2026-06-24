## Phase 1: Foundation - generate_sparsened_points Function

### 1.1 Create generate_sparsened_points function

- [X] 1.1.1 Create new file `lattice/generator/sparsened_point.py`
- [X] 1.1.2 Implement `generate_sparsened_points(latt_size, num_points, seed=None)` function
- [X] 1.1.3 Ensure output shape (num_points, Lt, 3) with dtype=np.int32
- [X] 1.1.4 Ensure coordinates are unique per time slice
- [X] 1.1.5 Add full docstring with input/output specifications

### 1.2 Update generator module exports

- [X] 1.2.1 Add import in `lattice/generator/__init__.py`
- [X] 1.2.2 Verify import works correctly

## Phase 2: Testing - generate_sparsened_points

### 2.1 Create comprehensive unit tests

- [X] 2.1.1 Create `tests/test_sparsened_point.py`
- [X] 2.1.2 Test output shape (Np, Lt, 3)
- [X] 2.1.3 Test output dtype is np.int32
- [X] 2.1.4 Test point uniqueness per time slice
- [X] 2.1.5 Test seed reproducibility
- [X] 2.1.6 Test coordinate ranges [0, Lx), [0, Ly), [0, Lz)

## Phase 3: Enhancement - PerambulatorGenerator Efficiency & Documentation

### 3.1 Review Existing Implementation

- [X] 3.1.1 Understand `calc_old()` method (lines 259-351) - sequential point processing
- [X] 3.1.2 Understand `calc_new()` method (lines 354-597) - vectorized approach
- [X] 3.1.3 Analyze performance bottlenecks in point extraction
- [X] 3.1.4 Document current GPU-CPU synchronization patterns

### 3.2 Optimize calc_new (Vectorized Implementation)

- [X] 3.2.1 Review vectorized point extraction (lines 422-481)
  - [X] 3.2.1a Understand batched coordinate computation
  - [X] 3.2.1b Analyze boolean masking logic
  - [X] 3.2.1c Review fancy indexing for SV array access
- [ ] 3.2.2 Identify optimization opportunities
  - [ ] 3.2.2a Reduce GPU-CPU synchronization overhead
  - [ ] 3.2.2b Improve memory access patterns
  - [ ] 3.2.2c Extract helper functions for clarity
- [ ] 3.2.3 Add performance annotations and timing hooks
- [ ] 3.2.4 Document optimization strategies in inline comments

### 3.3 Add Comprehensive Documentation

- [ ] 3.3.1 Document calc_old() method:
  - [ ] 3.3.1a Algorithm explanation (sequential approach)
  - [ ] 3.3.1b Use cases (validation, correctness reference)
  - [ ] 3.3.1c Performance characteristics
  - [ ] 3.3.1d GPU-CPU sync overhead analysis
- [ ] 3.3.2 Document calc_new() method:
  - [ ] 3.3.2a Algorithm explanation (vectorized approach)
  - [ ] 3.3.2b Vectorization benefits explained
  - [ ] 3.3.2c GPU-CPU sync reduction strategy
  - [ ] 3.3.2d Memory layout optimization notes
- [ ] 3.3.3 Add PSV/PSP calculation documentation
- [ ] 3.3.4 Document GPU indexing and checkerboard format
- [ ] 3.3.5 Add MPI grid awareness documentation

### 3.4 Code Clarity Improvements

- [ ] 3.4.1 Add inline comments explaining complex GPU operations
  - [ ] 3.4.1a Explain checkerboard indexing in calc_new
  - [ ] 3.4.1b Document coordinate transformations
  - [ ] 3.4.1c Note memory layout assumptions
- [ ] 3.4.2 Document data shapes at each computation stage
- [ ] 3.4.3 Add type hints for complex variables

### 3.5 Adapt Existing Tests to New Return Format

- [X] 3.5.1 Review `tests/test_perambulator.py`:
  - [X] 3.5.1a Understand current VSV-only test structure
  - [X] 3.5.1b Note that `calc(t)` returns VSV only when no point_src/snk
- [X] 3.5.2 Adapt test_perambulator.py for backward compatibility:
  - [X] 3.5.2a Verify `calc(t)` returns correct VSV shape when point_src=None
  - [X] 3.5.2b Ensure line 64 still works: `perambulator.calc(t).get()`
  - [X] 3.5.2c Add documentation comment about return value changes
- [X] 3.5.3 Review `tests/test_perambulator_mpi.py` for same compatibility
- [X] 3.5.4 Ensure both tests pass without modification

### 3.6 Testing & Validation

- [X] 3.6.1 Create numerical agreement tests:
  - [X] 3.6.1a Test calc_old vs calc_new produce identical results
  - [X] 3.6.1b Allow small numerical tolerance (1e-10)
- [X] 3.6.2 Create performance benchmark tests:
  - [X] 3.6.2a Measure calc_old timing on reference lattice
  - [X] 3.6.2b Measure calc_new timing on same lattice
  - [X] 3.6.2c Document speedup ratio achieved
  - [X] 3.6.2d Profile GPU-CPU synchronization overhead
- [ ] 3.6.3 Verify all existing tests still pass
  - [ ] 3.6.3a test_perambulator.py passes
  - [ ] 3.6.3b test_perambulator_mpi.py passes
  - [ ] 3.6.3c No numerical differences in VSV results

## Phase 4: Testing - PerambulatorGenerator Point Sources

### 4.1 Create integration tests

- [ ] 4.1.1 Create `tests/test_perambulator_point_sources.py`
- [ ] 4.1.2 Test PSV calculation with mock data
- [ ] 4.1.3 Test PSP calculation with mock data
- [ ] 4.1.4 Test vectorized vs naive implementation agreement
- [ ] 4.1.5 Test point coordinates handling (None, single, multiple)

## Phase 5: Final Validation

### 5.1 Run all tests and validation

- [ ] 5.1.1 Run pytest on all new test files with coverage
- [ ] 5.1.2 Verify no existing tests are broken
- [ ] 5.1.3 Check test coverage is >80% for new code
- [ ] 5.1.4 Verify OpenSpec proposal passes strict validation

### 5.2 Documentation

- [ ] 5.2.1 Update README.md with usage example
- [ ] 5.2.2 Add docstring examples in source files
- [ ] 5.2.3 Verify old scripts still work
