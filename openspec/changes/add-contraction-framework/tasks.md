# Contraction Framework Implementation Tasks

## Phase 1: Design & Architecture

### 1.1 Framework Architecture Design
- [ ] 1.1.1 Define contraction module structure
  - [ ] 1.1.1a Core utilities for tensor contractions
  - [ ] 1.1.1b Diagram types (meson, current, baryon)
  - [ ] 1.1.1c Backend abstraction (NumPy/CuPy)
- [ ] 1.1.2 Design I/O specifications
  - [ ] 1.1.2a Input arrays (perambulator, elemental, gauge field)
  - [ ] 1.1.2b Output shapes and data types
  - [ ] 1.1.2c Memory layout and efficiency considerations
- [ ] 1.1.3 Document design decisions in `design.md`
  - [ ] 1.1.3a Vectorization strategy
  - [ ] 1.1.3b Parallelization approach
  - [ ] 1.1.3c GPU acceleration considerations

### 1.2 Create Module Skeleton
- [ ] 1.2.1 Create directory structure
  - [ ] 1.2.1a `lattice/contraction/` directory
  - [ ] 1.2.1b `lattice/contraction/__init__.py`
  - [ ] 1.2.1c `lattice/contraction/diagram.py` (base utilities)
  - [ ] 1.2.1d `lattice/contraction/meson.py` (meson diagrams)
  - [ ] 1.2.1e `lattice/contraction/current.py` (current insertions)
- [ ] 1.2.2 Update `lattice/__init__.py` to export contraction module
- [ ] 1.2.3 Add comprehensive module docstrings

### 1.3 Define API Contracts
- [ ] 1.3.1 Specify meson contraction functions
  - [ ] 1.3.1a `compute_meson_vv()` - Vector-to-vector
  - [ ] 1.3.1b `compute_meson_pv()` - Pseudoscalar-to-vector
  - [ ] 1.3.1c Function signatures and I/O
- [ ] 1.3.2 Specify current contraction functions
  - [ ] 1.3.2a `compute_current_insertion()` - Generic current
  - [ ] 1.3.2b Input: eigenvectors, point sources, gauge field
  - [ ] 1.3.2c Output: correlation functions
- [ ] 1.3.3 Specify generic diagram utilities
  - [ ] 1.3.3a `contract_tensors()` - General tensor contraction wrapper
  - [ ] 1.3.3b Backend selection (NumPy/CuPy)
  - [ ] 1.3.3c Memory efficiency mode

## Phase 2: Core Implementation

### 2.1 Implement Base Utilities
- [ ] 2.1.1 Backend abstraction in `diagram.py`
  - [ ] 2.1.1a NumPy-based tensor operations
  - [ ] 2.1.1b CuPy GPU acceleration
  - [ ] 2.1.1c Seamless backend switching
- [ ] 2.1.2 Common tensor operations
  - [ ] 2.1.2a Batched matrix multiplication
  - [ ] 2.1.2b Index permutation utilities
  - [ ] 2.1.2c Memory layout optimization

### 2.2 Implement Meson Contraction
- [ ] 2.2.1 Meson diagrams in `meson.py`
  - [ ] 2.2.1a Vector-to-vector (VV) contraction
  - [ ] 2.2.1b Pseudoscalar-to-vector (PV) contraction
  - [ ] 2.2.1c Axial-vector contractions
- [ ] 2.2.2 Multi-time contraction support
  - [ ] 2.2.2a `compute_diagrams_multitime()` wrapper
  - [ ] 2.2.2b Parallel time slice processing
  - [ ] 2.2.2c Output formatting

### 2.3 Implement Current Contraction
- [ ] 2.3.1 Current insertions in `current.py`
  - [ ] 2.3.1a Vector current (γ_μ)
  - [ ] 2.3.1b Axial current (γ_μγ_5)
  - [ ] 2.3.1c Derivatives of currents
- [ ] 2.3.2 Point source handling
  - [ ] 2.3.2a Contraction with point sources
  - [ ] 2.3.2b Gauge field insertion
  - [ ] 2.3.2c Spin/color averaging

## Phase 3: Testing

### 3.1 Unit Tests - Meson
- [ ] 3.1.1 Create `tests/test_contraction_meson.py`
  - [ ] 3.1.1a Test VV contraction
  - [ ] 3.1.1b Test PV contraction
  - [ ] 3.1.1c Test output shapes and dtypes
- [ ] 3.1.2 Numerical validation
  - [ ] 3.1.2a Reference data [PLACEHOLDER]
  - [ ] 3.1.2b Tolerance settings
  - [ ] 3.1.2c Backend agreement (NumPy vs CuPy)

### 3.2 Unit Tests - Current
- [ ] 3.2.1 Create `tests/test_contraction_current.py`
  - [ ] 3.2.1a Test vector current
  - [ ] 3.2.1b Test axial current
  - [ ] 3.2.1c Test derivative operators
- [ ] 3.2.2 Numerical validation
  - [ ] 3.2.2a Reference data [PLACEHOLDER]
  - [ ] 3.2.2b Consistency checks
  - [ ] 3.2.2c Error handling

### 3.3 Unit Tests - Utilities
- [ ] 3.3.1 Create `tests/test_contraction_diagram.py`
  - [ ] 3.3.1a Test tensor contraction wrapper
  - [ ] 3.3.1b Test index permutations
  - [ ] 3.3.1c Test backend abstraction
- [ ] 3.3.2 Performance benchmarks
  - [ ] 3.3.2a Measure contraction time [PLACEHOLDER]
  - [ ] 3.3.2b Memory usage analysis
  - [ ] 3.3.2c Vectorization efficiency

## Phase 4: Integration

### 4.1 Integrate with Pipeline
- [ ] 4.1.1 Update `lattice/quark_diagram.py`
  - [ ] 4.1.1a Import contraction functions
  - [ ] 4.1.1b Adapt to use new API
  - [ ] 4.1.1c Maintain backward compatibility
- [ ] 4.1.2 Update `4.contraction.py`
  - [ ] 4.1.2a Use contraction framework
  - [ ] 4.1.2b Simplify script logic
  - [ ] 4.1.2c Add configuration options

### 4.2 Documentation
- [ ] 4.2.1 Complete API documentation
  - [ ] 4.2.1a Function docstrings (NumPy style)
  - [ ] 4.2.1b Parameter descriptions
  - [ ] 4.2.1c Return value specifications
- [ ] 4.2.2 Usage examples
  - [ ] 4.2.2a Example: Basic meson contraction
  - [ ] 4.2.2b Example: Current insertion
  - [ ] 4.2.2c Example: Multi-time calculation

### 4.3 Validation
- [ ] 4.3.1 Run integration tests
  - [ ] 4.3.1a Pipeline compatibility tests
  - [ ] 4.3.1b Numerical agreement with reference
  - [ ] 4.3.1c No regressions in existing tests
- [ ] 4.3.2 Performance validation [CLUSTER]
  - [ ] 4.3.2a Run on actual data
  - [ ] 4.3.2b Measure speedup
  - [ ] 4.3.2c Document scaling behavior

## Phase 5: Optimization & Deployment

### 5.1 Performance Optimization [OPTIONAL]
- [ ] 5.1.1 Profile contraction code
  - [ ] 5.1.1a Identify bottlenecks
  - [ ] 5.1.1b Vectorization opportunities
  - [ ] 5.1.1c GPU acceleration targets
- [ ] 5.1.2 Implement optimizations
  - [ ] 5.1.2a Batched operations
  - [ ] 5.1.2b GPU kernel optimization
  - [ ] 5.1.2c Memory pre-allocation
- [ ] 5.1.3 Document improvements
  - [ ] 5.1.3a Speedup metrics
  - [ ] 5.1.3b Scaling analysis
  - [ ] 5.1.3c Resource usage

### 5.2 Final Validation
- [ ] 5.2.1 Comprehensive testing
  - [ ] 5.2.1a All unit tests passing
  - [ ] 5.2.1b Integration tests passing
  - [ ] 5.2.1c Coverage >80% for new code
- [ ] 5.2.2 Documentation complete
  - [ ] 5.2.2a All functions documented
  - [ ] 5.2.2b README with examples
  - [ ] 5.2.2c Design doc comprehensive

## Placeholders for Cluster Testing

The following items require actual reference data from cluster runs:

- [ ] `test_contraction_meson.py::test_vv_contraction` - Reference data [PLACEHOLDER]
- [ ] `test_contraction_current.py::test_current_insertion` - Reference data [PLACEHOLDER]
- [ ] Performance benchmarks - Timing data [PLACEHOLDER]
- [ ] Scaling tests - Multi-GPU performance [PLACEHOLDER]
