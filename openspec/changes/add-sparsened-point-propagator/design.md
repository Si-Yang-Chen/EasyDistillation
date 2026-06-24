## Context

EasyDistillation implements lattice QCD distillation calculations. The current workflow involves:

1. Computing Laplace eigenvectors (EigenvectorGenerator)
2. Generating random "sparsened" points on the lattice (currently embedded in `1.gen_sparsened_field.py`)
3. Computing perambulators between eigenvectors and point sources (PerambulatorGenerator with partial point source support)
4. Computing gauge field matrix elements (ElementalGenerator)
5. Computing contractions (standalone logic in `4.contraction.py`)

The sparsened point generation is currently script-based without proper abstraction. The PerambulatorGenerator has point source support in the code but lacks testing and clear documentation. This creates maintenance challenges and makes the pipeline harder to use programmatically.

## Goals

- **Primary**: Extract sparsened point generation into a reusable, testable function with clear I/O
- **Secondary**: Validate and document PerambulatorGenerator's point source/sink calculations
- **Tertiary**: Enable pipeline-as-library usage pattern (end users can import and compose functions)

## Non-Goals

- Rewrite PerambulatorGenerator internals (only review and document)
- Optimize GPU memory usage (focus on correctness)
- Support additional lattice types (stick to 4D hypercubic)
- Change the distillation algorithm itself

## Architectural Decisions

### 1. Single Function for Point Generation

**Decision**: Create simple function `generate_sparsened_points(latt_size, num_points, seed)` in `lattice/generator/sparsened_point.py`

**Reasoning**:

- Simplest possible abstraction (no class overhead)
- Stateless and pure (deterministic with seed)
- Easy to test and compose
- Matches NumPy/SciPy convention for generator functions
- Caller controls storage and reproducibility

**Function Signature**:

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
        num_points: Number of spatial points Np to generate per time slice
        seed: Random seed for reproducibility (None = random)
      
    Returns:
        np.ndarray of shape (Np, Lt, 3) with dtype=np.int32
        coords[p, t, :] = [x, y, z] with unique coords per time slice
    """
```

**Alternatives considered**:

- Class-based: Over-engineered for this use case
- Embedded in PerambulatorGenerator: Couples two separate concerns
- Keep in standalone script: Current state; hard to test and reuse

### 2. Input/Output Specification

**Decision**: Clear, explicit shape documentation

**Inputs**:

- `latt_size`: List[int], exactly 4 elements [Lx, Ly, Lz, Lt]
- `num_points`: int, 0 < Np ≤ Lx*Ly*Lz (spatial volume)
- `seed`: Optional[int], None for random or integer for reproducible sequence

**Output**:

- NumPy array, shape (Np, Lt, 3), dtype=np.int32
- coords[p, t, 0] = x, coords[p, t, 1] = y, coords[p, t, 2] = z
- For each fixed t ∈ [0, Lt), all Np coordinates (x,y,z) are spatially distinct

### 3. PerambulatorGenerator Point Source Support

**Decision**: Review existing code (calc_old/calc_new); add documentation; no refactoring

**Reasoning**:

- Point source logic is complex (vectorization, GPU indexing, MPI grids)
- Refactoring risks introducing bugs in GPU code
- Documentation + testing validates correctness without rewrite
- Users can benefit from vectorized calc_new immediately

## Key Abstractions

### Point Generation Function

```python
# Simple, pure function
coords = generate_sparsened_points([24, 24, 24, 72], 216, seed=42)
# coords.shape = (216, 72, 3)
# coords.dtype = np.int32
# coords[p, t, :] = [x, y, z] with 0 <= x,y,z < 24 for all p,t

# User can then save/load as needed
np.save("points.npy", coords)
```

### PerambulatorGenerator.calc() Method Structure

- `load()`: Initializes gauge field, eigenvectors, point sources
- `calc_old()`: Original implementation, sequential point processing
- `calc_new()`: Vectorized implementation for GPU efficiency
- `calc()`: Dispatcher method that chooses old/new based on flag

## Data Flow

```
User Code
  |
  +---> generate_sparsened_points(latt_size, num_points, seed)
  |     -> np.ndarray (shape: Np, Lt, 3, dtype: int32)
  |
  +---> Save or pass to PointSource
  |
  +---> PerambulatorGenerator
  |     -> Use point_src/point_snk
  |     -> Return PSV/PSP arrays
```

## Risks and Mitigations

| Risk                  | Mitigation                                                 |
| --------------------- | ---------------------------------------------------------- |
| Point seed collisions | Document recommendation: use `cfg_id * 1000 + t` pattern |
| GPU indexing bugs     | Add detailed comments in calc_new(); validation tests      |
| Backwards compat      | Keep 1.gen_sparsened_field.py; add deprecation notice      |
| Test coverage gaps    | Require >80% coverage for new code before merge            |

## Migration Plan

1. **Phase 1** (PR 1): Add `generate_sparsened_points()` + unit tests (no API changes)
2. **Phase 2** (PR 2): Review & document PerambulatorGenerator + integration tests
3. **Phase 3** (PR 3): Update 1.gen_sparsened_field.py to use function (backward-compatible)
4. **Phase 4** (PR 4): Archive changes in OpenSpec

Each phase stands alone; no breaking changes introduced.

## Open Questions

- Should PointSource in preset.py support additional data formats (HDF5, PyTorch tensors)?
  - **Answer pending** user review; current implementation uses NumPy arrays
- Should function validate lattice size bounds?
  - **Proposal**: Add optional validation; default False for performance
- Should we support custom distributions beyond uniform random?
  - **Out of scope** for this change; function design allows it as future extension
