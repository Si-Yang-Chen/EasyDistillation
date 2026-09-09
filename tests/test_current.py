import numpy as np
import pytest

from lattice.insertion.current import (
    ConservedVectorCurrent, LocalAxialCurrent, LocalVectorCurrent,
    PseudoScalarDensity, lattice_divergence, verify_pcac, verify_wt,
)


def test_compute_elemental_adapts_calc_all():
    class Generator:
        def calc_all(self, t):
            return {"v2v": np.array([t]), "v2p": 2, "p2v": 3, "p2p": 4}
    result = LocalVectorCurrent().compute_elemental(Generator(), t=7)
    assert result["time"] == 7
    np.testing.assert_array_equal(result["elemental"], [7])
    assert len(result["terms"]) == 4


def test_compute_elemental_rejects_missing_calc_all():
    with pytest.raises(TypeError, match="calc_all"):
        LocalVectorCurrent().compute_elemental(object(), t=0)


def test_conserved_current_has_explicit_wilson_links():
    terms = ConservedVectorCurrent(wilson_r=1).terms
    assert len(terms) == 8
    assert [term.link for term in terms[:2]] == ["forward", "backward"]
    assert terms[0].displacement == (1, 0, 0, 0)
    assert terms[1].displacement == (-1, 0, 0, 0)
    assert all(term.wilson_r == 1 for term in terms)


def test_array_level_wt_divergence():
    current = np.zeros((4, 2, 2, 2, 2), dtype=np.complex128)
    assert verify_wt(current=current)["passed"]
    current[0, 0, 0, 0, 0] = 1
    assert not verify_wt(current=current)["passed"]


def test_z_and_pcac_are_not_false_positives():
    assert LocalVectorCurrent().verify(z=1, divergence=np.zeros((2,))).keys() >= {"Z", "WT"}
    with pytest.raises(ValueError, match="renormalization"):
        LocalVectorCurrent().verify(divergence=np.zeros((2,)))
    assert verify_pcac(axial_divergence=np.ones((2,)), pseudoscalar=np.ones((2,)), mass=.5)["passed"]
    with pytest.raises(ValueError, match="mass"):
        verify_pcac(axial_divergence=np.ones((2,)), pseudoscalar=np.ones((2,)))


def test_axial_and_density_require_pcac_inputs():
    with pytest.raises(ValueError, match="axial_divergence"):
        LocalAxialCurrent().verify(z=1)
    with pytest.raises(ValueError, match="pseudoscalar"):
        PseudoScalarDensity().verify(z=1, axial_divergence=np.zeros(2), mass=1)


def test_divergence_supports_nonperiodic_boundary():
    current = np.zeros((4, 2, 1, 1, 1))
    current[0, 1, 0, 0, 0] = 1
    result = lattice_divergence(current, periodic=False)
    assert result[1, 0, 0, 0] == 1
