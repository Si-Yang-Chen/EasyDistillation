"""Test the gamma matrix module (lattice.insertion.gamma)."""

import numpy as np
import pytest

from lattice.insertion.gamma import (
    gamma,
    output,
    scheme,
    group,
    parity,
    charge_conjugation,
    hermiticity,
    GammaName,
    gamma_transform,
)


class TestGammaMatrices:
    """Test individual gamma matrices."""

    @pytest.mark.parametrize("n,expected_label", [
        (0, ""),
        (15, "γ5"),
        (7, "γ5γ4"),
        (8, "γ4"),
    ])
    def test_output_labels(self, n, expected_label):
        """output(n) should return known labels for key indices."""
        assert output(n) == expected_label

    def test_gamma_0_is_identity(self):
        """gamma(0) is identity (np.eye(4))."""
        g0 = gamma(0)
        assert g0.shape == (4, 4)
        np.testing.assert_allclose(g0, np.eye(4), atol=1e-10)

    @pytest.mark.parametrize("n", [1, 2, 4, 8, 15])
    def test_gamma_shapes(self, n):
        """All gamma matrices are 4x4."""
        g = gamma(n)
        assert g.shape == (4, 4)

    @pytest.mark.parametrize("n", range(16))
    def test_gamma_is_array(self, n):
        """Every gamma(n) for 0 <= n <= 15 is a 4x4 array."""
        g = gamma(n)
        assert g.shape == (4, 4)

    def test_gamma_5_is_hermitian(self):
        """gamma_5 (n=15) is hermitian: gamma_5^dagger = gamma_5."""
        g5 = gamma(15)
        np.testing.assert_allclose(g5.conj().T, g5, atol=1e-10)

    def test_gamma_5_squares_to_identity(self):
        """gamma_5 @ gamma_5 = I."""
        g5 = gamma(15)
        np.testing.assert_allclose(g5 @ g5, np.eye(4), atol=1e-10)


class TestGammaName:
    """Test GammaName class."""

    def test_gamma_name_has_a0(self):
        assert GammaName.A0 == R"$a_0$"

    def test_gamma_name_has_pi(self):
        assert GammaName.PI == R"$\pi$"

    def test_gamma_name_has_rho(self):
        assert GammaName.RHO == R"$\rho$"

    def test_all_names_are_strings(self):
        """All GammaName attributes should be non-empty strings."""
        for attr in ["A0", "B0", "PI", "PI_2", "RHO", "RHO_2", "A1", "B1"]:
            val = getattr(GammaName, attr)
            assert isinstance(val, str) and len(val) > 0


class TestGammaConventions:
    """Test gamma scheme/group/parity/conjugation functions."""

    @pytest.mark.parametrize("name_key", [
        GammaName.A0,
        GammaName.B0,
        GammaName.PI,
        GammaName.RHO,
        GammaName.A1,
        GammaName.B1,
    ])
    def test_scheme_returns_list_of_ints(self, name_key):
        s = scheme(name_key)
        assert isinstance(s, list)
        assert all(isinstance(x, int) for x in s)

    @pytest.mark.parametrize("name_key", [
        GammaName.A0, GammaName.PI, GammaName.RHO,
    ])
    def test_group_returns_string(self, name_key):
        g = group(name_key)
        assert isinstance(g, str)

    @pytest.mark.parametrize("name_key", [
        GammaName.A0, GammaName.PI, GammaName.RHO,
    ])
    def test_parity_returns_int(self, name_key):
        p = parity(name_key)
        assert p in (1, -1)

    @pytest.mark.parametrize("name_key", [
        GammaName.A0, GammaName.PI, GammaName.RHO,
    ])
    def test_charge_conjugation_returns_int(self, name_key):
        c = charge_conjugation(name_key)
        assert c in (1, -1)

    @pytest.mark.parametrize("name_key", [
        GammaName.A0, GammaName.PI, GammaName.RHO,
    ])
    def test_hermiticity_returns_int(self, name_key):
        h = hermiticity(name_key)
        assert h in (1, -1)


class TestGammaTransform:
    """Test gamma transform dictionary."""

    def test_transform_iden_is_identity(self):
        """Under 'iden' rotation, every gamma index maps to itself."""
        for i in range(16):
            assert gamma_transform("iden", i) == i

    def test_transform_known_keys(self):
        """gamma_transform should work for known rotation keys."""
        for key in ["c4x", "c2x", "c4y", "c4z", "c2z"]:
            val = gamma_transform(key, 0)
            assert isinstance(val, int)

    def test_transform_all_keys_for_idx_0(self):
        """For gamma_0 (identity), transformation should be 0 for all rotations."""
        # gamma_0 is the scalar/identity, it should be invariant
        known_keys = ["iden", "c4x", "c2x", "c4y", "c2y", "c4z", "c2z"]
        for key in known_keys:
            assert gamma_transform(key, 0) == 0
