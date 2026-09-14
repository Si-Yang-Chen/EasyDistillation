"""
Unit tests for the generate_sparsened_points function.
"""

import pytest
import numpy as np
from lattice.generator import generate_sparsened_points


class TestGenerateSparsenedPoints:
    """Test suite for generate_sparsened_points function."""

    def test_basic_generation(self):
        """Test basic generation with standard lattice size."""
        latt_size = [24, 24, 24, 72]
        num_points = 216

        coords = generate_sparsened_points(latt_size, num_points, seed=42)

        assert coords is not None
        assert isinstance(coords, np.ndarray)

    def test_output_shape(self):
        """Test that output shape is (num_points, Lt, 3)."""
        latt_size = [24, 24, 24, 72]
        num_points = 216

        coords = generate_sparsened_points(latt_size, num_points, seed=42)

        assert coords.shape == (num_points, 72, 3)

    def test_output_dtype(self):
        """Test that output dtype is np.int32."""
        latt_size = [24, 24, 24, 72]
        num_points = 216

        coords = generate_sparsened_points(latt_size, num_points, seed=42)

        assert coords.dtype == np.int32

    def test_coordinate_ranges(self):
        """Test that all coordinates are within valid ranges."""
        latt_size = [16, 12, 8, 10]
        num_points = 50

        coords = generate_sparsened_points(latt_size, num_points, seed=42)

        Lx, Ly, Lz, Lt = latt_size

        # Check x coordinates
        assert np.all(coords[:, :, 0] >= 0)
        assert np.all(coords[:, :, 0] < Lx)

        # Check y coordinates
        assert np.all(coords[:, :, 1] >= 0)
        assert np.all(coords[:, :, 1] < Ly)

        # Check z coordinates
        assert np.all(coords[:, :, 2] >= 0)
        assert np.all(coords[:, :, 2] < Lz)

    def test_uniqueness_per_time_slice(self):
        """Test that all points are unique within each time slice."""
        latt_size = [10, 10, 10, 5]
        num_points = 50

        coords = generate_sparsened_points(latt_size, num_points, seed=42)

        # Check uniqueness for each time slice
        for t in range(5):
            # Convert to tuple for set operations
            points_at_t = [tuple(coords[p, t, :]) for p in range(num_points)]
            unique_points = set(points_at_t)

            # All points should be unique
            assert len(unique_points) == num_points, (
                f"Time slice {t} has {len(unique_points)} unique points, " f"expected {num_points}"
            )

    def test_seed_reproducibility(self):
        """Test that same seed produces identical results."""
        latt_size = [24, 24, 24, 72]
        num_points = 216
        seed = 42

        coords1 = generate_sparsened_points(latt_size, num_points, seed=seed)
        coords2 = generate_sparsened_points(latt_size, num_points, seed=seed)

        assert np.array_equal(coords1, coords2)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        latt_size = [24, 24, 24, 72]
        num_points = 216

        coords1 = generate_sparsened_points(latt_size, num_points, seed=42)
        coords2 = generate_sparsened_points(latt_size, num_points, seed=43)

        # Results should be different (extremely unlikely to be the same)
        assert not np.array_equal(coords1, coords2)

    def test_none_seed_uses_random_state(self):
        """Test that seed=None produces non-deterministic results."""
        latt_size = [24, 24, 24, 72]
        num_points = 216

        coords1 = generate_sparsened_points(latt_size, num_points, seed=None)
        coords2 = generate_sparsened_points(latt_size, num_points, seed=None)

        # Results should be different (with very high probability)
        # We use a relaxed check since random collisions are theoretically possible
        assert not np.array_equal(coords1, coords2)

    def test_small_lattice(self):
        """Test with small lattice."""
        latt_size = [4, 4, 4, 2]
        num_points = 8

        coords = generate_sparsened_points(latt_size, num_points, seed=42)

        assert coords.shape == (num_points, 2, 3)
        assert coords.dtype == np.int32

        # Check uniqueness
        for t in range(2):
            points_at_t = [tuple(coords[p, t, :]) for p in range(num_points)]
            unique_points = set(points_at_t)
            assert len(unique_points) == num_points

    def test_large_lattice(self):
        """Test with large lattice."""
        latt_size = [100, 100, 100, 10]
        num_points = 1000

        coords = generate_sparsened_points(latt_size, num_points, seed=42)

        assert coords.shape == (num_points, 10, 3)
        assert coords.dtype == np.int32

    def test_single_point(self):
        """Test with single point per time slice."""
        latt_size = [24, 24, 24, 72]
        num_points = 1

        coords = generate_sparsened_points(latt_size, num_points, seed=42)

        assert coords.shape == (1, 72, 3)
        assert coords.dtype == np.int32

    def test_max_points(self):
        """Test with maximum possible points (full spatial volume)."""
        Lx, Ly, Lz = 5, 5, 5
        num_points = Lx * Ly * Lz
        latt_size = [Lx, Ly, Lz, 3]

        coords = generate_sparsened_points(latt_size, num_points, seed=42)

        assert coords.shape == (num_points, 3, 3)

        # All spatial volume should be covered at each time slice
        for t in range(3):
            points_at_t = [tuple(coords[p, t, :]) for p in range(num_points)]
            unique_points = set(points_at_t)
            assert len(unique_points) == num_points

    # Error handling tests

    def test_latt_size_wrong_length(self):
        """Test error when latt_size doesn't have 4 elements."""
        with pytest.raises(ValueError, match="latt_size must have 4 elements"):
            generate_sparsened_points([24, 24, 24], 216)

        with pytest.raises(ValueError, match="latt_size must have 4 elements"):
            generate_sparsened_points([24, 24, 24, 72, 1], 216)

    def test_num_points_exceeds_volume(self):
        """Test error when num_points exceeds spatial volume."""
        latt_size = [5, 5, 5, 10]
        num_points = 200  # Exceeds 5*5*5=125

        with pytest.raises(ValueError, match="exceeds spatial volume"):
            generate_sparsened_points(latt_size, num_points)

    def test_num_points_zero(self):
        """Test error when num_points is zero."""
        latt_size = [24, 24, 24, 72]
        num_points = 0

        with pytest.raises(ValueError, match="must be positive"):
            generate_sparsened_points(latt_size, num_points)

    def test_num_points_negative(self):
        """Test error when num_points is negative."""
        latt_size = [24, 24, 24, 72]
        num_points = -10

        with pytest.raises(ValueError, match="must be positive"):
            generate_sparsened_points(latt_size, num_points)

    # Edge case tests

    def test_rectangular_lattice(self):
        """Test with non-cubic spatial lattice."""
        latt_size = [8, 16, 32, 4]
        num_points = 100

        coords = generate_sparsened_points(latt_size, num_points, seed=42)

        assert coords.shape == (num_points, 4, 3)

        # Check ranges
        assert np.all(coords[:, :, 0] < 8)
        assert np.all(coords[:, :, 1] < 16)
        assert np.all(coords[:, :, 2] < 32)

    def test_single_time_slice(self):
        """Test with single time slice."""
        latt_size = [24, 24, 24, 1]
        num_points = 216

        coords = generate_sparsened_points(latt_size, num_points, seed=42)

        assert coords.shape == (216, 1, 3)

        # Verify uniqueness
        points = [tuple(coords[p, 0, :]) for p in range(num_points)]
        unique_points = set(points)
        assert len(unique_points) == num_points

    def test_array_independence(self):
        """Test that returned arrays are independent."""
        latt_size = [24, 24, 24, 72]
        num_points = 216

        coords1 = generate_sparsened_points(latt_size, num_points, seed=42)
        coords2 = generate_sparsened_points(latt_size, num_points, seed=42)

        # Arrays should be equal in value
        assert np.array_equal(coords1, coords2)

        # But should be different objects in memory
        assert coords1 is not coords2


class TestIntegration:
    """Integration tests for generate_sparsened_points."""

    def test_usage_with_numpy_save(self):
        """Test that output can be saved with numpy."""
        import tempfile
        import os

        latt_size = [24, 24, 24, 72]
        num_points = 216

        coords = generate_sparsened_points(latt_size, num_points, seed=42)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as f:
            temp_path = f.name

        try:
            np.save(temp_path, coords)
            coords_loaded = np.load(temp_path)

            # Verify data integrity
            assert np.array_equal(coords, coords_loaded)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_reproducible_workflow(self):
        """Test reproducible workflow with seed."""
        latt_size = [24, 24, 24, 72]
        num_points = 216
        cfg_id = 10000
        t = 5

        # Simulate reproducible generation for config and time slice
        seed = cfg_id * 1000 + t
        coords1 = generate_sparsened_points(latt_size, num_points, seed=seed)

        # Later, regenerate the same set
        coords2 = generate_sparsened_points(latt_size, num_points, seed=seed)

        assert np.array_equal(coords1, coords2)


# Parametrized tests for comprehensive coverage


@pytest.mark.parametrize(
    "latt_size,num_points",
    [
        ([8, 8, 8, 4], 32),
        ([10, 10, 10, 5], 50),
        ([16, 16, 16, 8], 128),
        ([24, 24, 24, 72], 216),
    ],
)
def test_various_lattice_sizes(latt_size, num_points):
    """Test with various lattice sizes."""
    coords = generate_sparsened_points(latt_size, num_points, seed=42)

    Lx, Ly, Lz, Lt = latt_size
    expected_shape = (num_points, Lt, 3)

    assert coords.shape == expected_shape
    assert coords.dtype == np.int32

    # Verify coordinate ranges
    assert np.all(coords >= 0)
    assert np.all(coords[:, :, 0] < Lx)
    assert np.all(coords[:, :, 1] < Ly)
    assert np.all(coords[:, :, 2] < Lz)


@pytest.mark.parametrize("seed", [0, 42, 123, 999, 1000000])
def test_various_seeds(seed):
    """Test with various seed values."""
    latt_size = [24, 24, 24, 72]
    num_points = 216

    coords = generate_sparsened_points(latt_size, num_points, seed=seed)

    assert coords.shape == (num_points, 72, 3)
    assert coords.dtype == np.int32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
