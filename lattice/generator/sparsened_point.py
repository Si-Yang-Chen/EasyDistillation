"""Generate unique sparse spatial points for each lattice time slice."""

from __future__ import annotations

from numbers import Integral
from typing import List, Optional

import numpy as np


def generate_sparsened_points(latt_size: List[int], num_points: int, seed: Optional[int] = None) -> np.ndarray:
    """Return ``(num_points, Lt, 3)`` unique integer coordinates per time."""
    if len(latt_size) != 4:
        raise ValueError(f"latt_size must have 4 elements, got {len(latt_size)}")
    if any(isinstance(size, bool) or not isinstance(size, Integral) for size in latt_size):
        raise TypeError("latt_size must contain four integers")
    Lx, Ly, Lz, Lt = (int(size) for size in latt_size)
    if min(Lx, Ly, Lz, Lt) <= 0:
        raise ValueError("latt_size extents must be positive")
    if isinstance(num_points, bool) or not isinstance(num_points, Integral):
        raise TypeError("num_points must be an integer")
    num_points = int(num_points)
    spatial_volume = Lx * Ly * Lz
    if num_points > spatial_volume:
        raise ValueError(f"num_points ({num_points}) exceeds spatial volume ({Lx} × {Ly} × {Lz} = {spatial_volume})")
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
        raise TypeError("seed must be an integer or None")

    generator = np.random.default_rng(None if seed is None else int(seed))
    coordinates = np.empty((num_points, Lt, 3), dtype=np.int32)
    for time in range(Lt):
        flat = generator.choice(spatial_volume, size=num_points, replace=False)
        coordinates[:, time, 0] = flat % Lx
        coordinates[:, time, 1] = (flat // Lx) % Ly
        coordinates[:, time, 2] = flat // (Lx * Ly)
    return coordinates


__all__ = ["generate_sparsened_points"]
