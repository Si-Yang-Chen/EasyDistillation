## MODIFIED Requirements

### Requirement: Perambulator generation with point sources
PerambulatorGenerator SHALL compute perambulators between Laplace eigenvectors and point sources/sinks, with proper documentation of the calculation methods.

#### Scenario: Point source initialization
- **WHEN** `PerambulatorGenerator.__init__()` is called with `point_src` and `point_snk` parameters
- **THEN** it SHALL initialize internal arrays for PSV (point-source to eigenvector) and PSP (point-source to point-sink) calculations

#### Scenario: PSV calculation (point-source to eigenvector)
- **WHEN** `calc(t_src)` is called with point sources provided
- **THEN** for each point in `point_src`, it SHALL compute propagation from that point to all sink eigenvectors
- **AND** return PSV array of shape `(Lt, Ns, Ns, Nc, Np_snk, Ne_src)`

#### Scenario: PSP calculation (point-source to point-sink)
- **WHEN** `calc(t_src)` is called with both `point_src` and `point_snk` provided
- **THEN** for each source and sink point pair, it SHALL compute the propagator between them
- **AND** return PSP array of shape `(Lt, Ns, Ns, Nc, Nc, Np_snk, Np_src)`

#### Scenario: Vectorized GPU computation
- **WHEN** `use_vectorized=True` (default) in `__init__()` and GPU backend is available
- **THEN** `calc()` SHALL use the `calc_new()` method for efficient batched GPU operations
- **AND** process all valid points per eigenvector in a single vectorized operation

#### Scenario: Sequential fallback computation
- **WHEN** `use_vectorized=False` in `__init__()`
- **THEN** `calc()` SHALL use the `calc_old()` method for sequential point processing
- **AND** produce results consistent with the vectorized method

#### Scenario: Checkerboard indexing for GPU
- **WHEN** computing PSV or PSP values, coordinates are accessed in the SV/SP arrays
- **THEN** checkerboard indices SHALL be computed correctly: `cb = (t + x + y + z) % 2`
- **AND** x-coordinates SHALL be halved: `x_half = x // 2` for GPU memory layout

#### Scenario: MPI grid awareness
- **WHEN** computing point sources in a distributed GPU environment
- **THEN** each GPU SHALL only process points whose spatial coordinates belong to its `grid_coord` region
- **AND** time slices SHALL be partitioned as `[gx * Lt, (gx + 1) * Lt)`

### Requirement: PerambulatorGenerator point source parameter documentation
The class documentation and method docstrings SHALL clearly explain point source parameters and their effects.

#### Scenario: Parameter documentation completeness
- **WHEN** user reads `PerambulatorGenerator.__init__()` docstring
- **THEN** it SHALL document all point-source-related parameters:
  - `point_src`: Point source data structure
  - `usedNp_src`: Number of point sources to use
  - `point_snk`: Point sink data structure  
  - `usedNp_snk`: Number of point sinks to use

#### Scenario: Return value documentation
- **WHEN** user reads `calc()` method docstring
- **THEN** it SHALL document return values including PSV and PSP arrays
- **AND** array shapes and meanings

#### Scenario: GPU memory requirements note
- **WHEN** user initializes PerambulatorGenerator with point sources
- **THEN** docstring SHALL note GPU memory requirements scale with `Np_src * Np_snk`
