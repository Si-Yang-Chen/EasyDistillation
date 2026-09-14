from __future__ import annotations

import numpy as np

from lattice import set_backend
from lattice.insertion.gamma import gamma
from lattice.quark_diagram import PropagatorWithCurrent


def test_vsp_psv_gamma5_dagger_swaps_sink_source_spin_axes():
    set_backend("numpy")
    rng = np.random.default_rng(20260902)
    temporal_extent = 2
    spin_count = 4
    eigen_count = 3
    point_count = 2
    color_count = 3
    gamma5 = np.asarray(gamma(15))
    propagator = PropagatorWithCurrent(Lt=temporal_extent)

    psv = rng.normal(
        size=(
            temporal_extent,
            spin_count,
            spin_count,
            point_count,
            color_count,
            eigen_count,
        )
    ) + 1j * rng.normal(
        size=(
            temporal_extent,
            spin_count,
            spin_count,
            point_count,
            color_count,
            eigen_count,
        )
    )
    expected_vsp = np.einsum(
        "ia,tbaxce,bj->tijexc", gamma5, psv.conj(), gamma5
    )
    actual_vsp = propagator._dagger_psv(psv)
    np.testing.assert_allclose(actual_vsp, expected_vsp, atol=1e-12, rtol=1e-12)

    vsp = rng.normal(
        size=(
            temporal_extent,
            spin_count,
            spin_count,
            eigen_count,
            point_count,
            color_count,
        )
    ) + 1j * rng.normal(
        size=(
            temporal_extent,
            spin_count,
            spin_count,
            eigen_count,
            point_count,
            color_count,
        )
    )
    expected_psv = np.einsum(
        "ia,tbaexc,bj->tijxce", gamma5, vsp.conj(), gamma5
    )
    actual_psv = propagator._dagger_vsp(vsp)
    np.testing.assert_allclose(actual_psv, expected_psv, atol=1e-12, rtol=1e-12)

    np.testing.assert_allclose(
        propagator._dagger_vsp(actual_vsp), psv, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        propagator._dagger_psv(actual_psv), vsp, atol=1e-12, rtol=1e-12
    )
