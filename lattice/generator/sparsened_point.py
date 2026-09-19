"""
Sparsened point generation for lattice QCD distillation calculations.
"""

from typing import List, Optional
import numpy as np


def generate_sparsened_points(
    latt_size: List[int], num_points: int, seed: Optional[int] = None,
    temporal_policy: str = "independent",
) -> np.ndarray:
    """
    Generate random sparsened points on a lattice.

    Generates a set of random spatial coordinates on a 4D hypercubic lattice,
    with the property that all points are unique within each time slice.
    Useful for distillation method calculations.

    Parameters
    ----------
    latt_size : List[int]
        Lattice dimensions [Lx, Ly, Lz, Lt] where:
        - Lx, Ly, Lz: spatial extents
        - Lt: temporal extent
        Must be exactly 4 elements.

    num_points : int
        Number of spatial points to generate per time slice (Np).
        Must satisfy: 0 < num_points ≤ Lx × Ly × Lz

    temporal_policy : {"independent", "same"}, default="independent"
        Whether each time slice receives an independent spatial point set or
        all time slices reuse the same spatial point set.

    seed : Optional[int], default=None
        Random seed for reproducibility.
        - If None: uses current random state (non-deterministic)
        - If int: sets np.random.seed(seed) for deterministic generation

    Returns
    -------
    np.ndarray
        Array of shape (num_points, Lt, 3) with dtype=np.int32.

        coords[p, t, :] = [x, y, z] represents point p at time slice t where:
        - x ∈ [0, Lx)
        - y ∈ [0, Ly)
        - z ∈ [0, Lz)
        - All Np points are spatially distinct within each time slice t

    Raises
    ------
    ValueError
        If num_points > Lx × Ly × Lz (requested more points than spatial volume)
        If latt_size doesn't have exactly 4 elements

    Examples
    --------
    >>> coords = generate_sparsened_points([24, 24, 24, 72], 216, seed=42)
    >>> coords.shape
    (216, 72, 3)
    >>> coords.dtype
    dtype('int32')
    >>> coords[0, 0, :]  # First point at time 0
    array([x, y, z], dtype=int32)

    # Generate with reproducibility
    >>> coords1 = generate_sparsened_points([24, 24, 24, 72], 216, seed=42)
    >>> coords2 = generate_sparsened_points([24, 24, 24, 72], 216, seed=42)
    >>> np.allclose(coords1, coords2)
    True

    # Save and reuse
    >>> np.save("points.npy", coords)
    >>> loaded = np.load("points.npy")
    """
    # Validate input
    if len(latt_size) != 4:
        raise ValueError(f"latt_size must have 4 elements, got {len(latt_size)}")
    if temporal_policy not in ("independent", "same"):
        raise ValueError(f"temporal_policy must be 'independent' or 'same', got {temporal_policy!r}")

    Lx, Ly, Lz, Lt = latt_size
    spatial_volume = Lx * Ly * Lz

    if num_points > spatial_volume:
        raise ValueError(
            f"num_points ({num_points}) exceeds spatial volume " f"({Lx} × {Ly} × {Lz} = {spatial_volume})"
        )

    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Initialize output array
    coords = np.zeros((num_points, Lt, 3), dtype=np.int32)

    # Generate one spatial set when requested, then reuse it for every time.
    if temporal_policy == "same":
        flat = np.random.choice(spatial_volume, size=num_points, replace=False)
        spatial = np.stack(
            [flat % Lx, (flat // Lx) % Ly, flat // (Lx * Ly)], axis=1
        ).astype(np.int32)
        coords[:] = spatial[:, None, :]
        return coords

    # Generate points independently for each time slice
    for t in range(Lt):
        # Track used coordinates to ensure uniqueness
        used_points = set()
        point_count = 0

        # Generate unique points for this time slice
        while point_count < num_points:
            x = np.random.randint(0, Lx)
            y = np.random.randint(0, Ly)
            z = np.random.randint(0, Lz)
            point_tuple = (x, y, z)

            # Only add if not already used
            if point_tuple not in used_points:
                used_points.add(point_tuple)
                coords[point_count, t, 0] = x
                coords[point_count, t, 1] = y
                coords[point_count, t, 2] = z
                point_count += 1

    return coords
