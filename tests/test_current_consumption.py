import numpy as np
import pytest

from lattice import set_backend
from lattice.insertion.current import (
    CURRENT_ASSEMBLER_SCHEMA,
    ConservedVectorCurrent,
    LocalAxialCurrent,
    LocalVectorCurrent,
    PseudoScalarDensity,
    assemble_spin_aware_current,
    consume_spin_aware_current,
    resolve_current_term_spin,
    spin_aware_current_adapter,
)
from lattice.insertion.gamma import gamma


def _raw_value(term, sink_ne, source_ne):
    direction = term["direction"] + 2
    link = {"none": 1, "forward": 2, "backward": 3}[term["link"]]
    values = np.arange(sink_ne * source_ne, dtype=np.float64).reshape(
        sink_ne, source_ne
    )
    return (direction + 1j * link) + (1 + 2j) * values


def _raw_resolver(term, *, endpoints, source_ne, sink_ne):
    return {
        "value": _raw_value(term, sink_ne, source_ne),
        "source_ne": source_ne,
        "sink_ne": sink_ne,
        "provenance": {"bar_time": endpoints["bar_time"]},
    }


def _spin_matrix(term):
    gamma_matrix = np.asarray(gamma(term.gamma_index))
    if term.link == "none":
        return gamma_matrix
    identity = np.eye(4, dtype=gamma_matrix.dtype)
    return (
        term.wilson_r * identity - gamma_matrix
        if term.link == "forward"
        else term.wilson_r * identity + gamma_matrix
    )


@pytest.mark.parametrize(
    "current",
    [
        LocalVectorCurrent(),
        LocalAxialCurrent(),
        PseudoScalarDensity(),
        ConservedVectorCurrent(wilson_r=1.25),
    ],
)
def test_spin_aware_current_consumer_exact_numpy_regression(current):
    set_backend("numpy")
    assembled = assemble_spin_aware_current(
        current,
        _raw_resolver,
        available_source_ne=3,
        available_sink_ne=4,
        used_source_ne=2,
        used_sink_ne=3,
        anchor_time=1,
    )
    expected = sum(
        term.coefficient
        * term.normalization
        * _spin_matrix(term)[:, :, None, None]
        * _raw_value(term.as_dict(), 3, 2)[None, None, :, :]
        for term in current.terms
    )
    np.testing.assert_array_equal(assembled["value"], expected)
    adapter = spin_aware_current_adapter(assembled)
    sink_spin = np.array([1 + 1j, -2j, 0.5, 3], dtype=np.complex128)
    source_spin = np.array([-1j, 2, 1 - 1j, 0.25], dtype=np.complex128)
    actual = consume_spin_aware_current(adapter, sink_spin, source_spin)
    reference = np.einsum("a,abij,b->ij", sink_spin, expected, source_spin)
    assert adapter["schema"] == CURRENT_ASSEMBLER_SCHEMA
    assert adapter["axes"] == ("sink_spin", "source_spin", "sink_ne", "source_ne")
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(actual, reference)


def test_local_vector_axial_and_pseudoscalar_apply_explicit_gamma_once():
    set_backend("numpy")
    for current in (LocalVectorCurrent(), LocalAxialCurrent(), PseudoScalarDensity()):
        for term in current.terms:
            resolved = resolve_current_term_spin(
                term,
                _raw_resolver,
                endpoints={
                    "bar_time": 0,
                    "field_time": 0,
                    "link_origin_time": 0,
                    "temporal_point_split": False,
                    "boundary": "unbounded",
                },
                source_ne=2,
                sink_ne=3,
            )
            raw = _raw_value(term.as_dict(), 3, 2)
            np.testing.assert_allclose(
                resolved["value"],
                np.asarray(gamma(term.gamma_index))[:, :, None, None]
                * raw[None, None, :, :],
            )
    density = PseudoScalarDensity().terms[0]
    assert density.direction == -1
    assert density.spin_structure == "gamma_5"


@pytest.mark.parametrize("mu", [0, 3])
def test_conserved_forward_backward_spin_and_weights_are_applied_once(mu):
    set_backend("numpy")
    current = ConservedVectorCurrent(wilson_r=1.25)
    forward, backward = current.terms[2 * mu : 2 * mu + 2]
    raw = np.array(
        [[1 + 2j, 3 - 4j], [-2 + 0.5j, 5 + 7j], [11 - 3j, -1 + 6j]],
        dtype=np.complex128,
    )

    def resolver(term, *, endpoints, source_ne, sink_ne):
        assert endpoints["temporal_point_split"] is (mu == 3)
        return {"value": raw, "source_ne": source_ne, "sink_ne": sink_ne}

    for term, sign in ((forward, -1), (backward, 1)):
        result = resolve_current_term_spin(
            term,
            resolver,
            endpoints={
                "bar_time": 1,
                "field_time": 2 if mu == 3 else 1,
                "link_origin_time": 1,
                "temporal_point_split": mu == 3,
                "boundary": "unbounded",
            },
            source_ne=2,
            sink_ne=3,
        )
        expected_spin = (
            term.wilson_r * np.eye(4) + sign * np.asarray(gamma(term.gamma_index))
        )
        np.testing.assert_allclose(
            result["value"], expected_spin[:, :, None, None] * raw[None, None, :, :]
        )

    assembled = assemble_spin_aware_current(
        (forward, backward),
        resolver,
        available_source_ne=2,
        available_sink_ne=3,
        used_source_ne=2,
        used_sink_ne=3,
        anchor_time=1,
    )
    direct = (
        -0.5
        * (1.25 * np.eye(4) - np.asarray(gamma(forward.gamma_index)))[:, :, None, None]
        * raw[None, None, :, :]
        + 0.5
        * (1.25 * np.eye(4) + np.asarray(gamma(backward.gamma_index)))[:, :, None, None]
        * raw[None, None, :, :]
    )
    np.testing.assert_allclose(assembled["value"], direct)


@pytest.mark.parametrize(
    "bad_value,error",
    [
        (np.ones((2, 2), dtype=np.float64), TypeError),
        (np.ones((2, 3), dtype=np.complex128), ValueError),
        (np.array([[np.nan + 0j]]), ValueError),
    ],
)
def test_spin_resolver_rejects_bad_raw_dtype_shape_and_ne(bad_value, error):
    term = LocalVectorCurrent().terms[0]

    def resolver(term, *, endpoints, source_ne, sink_ne):
        return {"value": bad_value, "source_ne": source_ne, "sink_ne": sink_ne}

    with pytest.raises(error):
        resolve_current_term_spin(
            term,
            resolver,
            endpoints={
                "bar_time": 0,
                "field_time": 0,
                "link_origin_time": 0,
                "temporal_point_split": False,
                "boundary": "unbounded",
            },
            source_ne=2,
            sink_ne=1,
        )


def test_spin_resolver_rejects_ne_mismatch_and_adapter_bad_spin_axes():
    term = LocalVectorCurrent().terms[0]

    def mismatch(term, *, endpoints, source_ne, sink_ne):
        return {
            "value": np.ones((sink_ne, source_ne), dtype=np.complex128),
            "source_ne": source_ne + 1,
            "sink_ne": sink_ne,
        }

    with pytest.raises(ValueError, match="exactly match"):
        resolve_current_term_spin(
            term,
            mismatch,
            endpoints={
                "bar_time": 0,
                "field_time": 0,
                "link_origin_time": 0,
                "temporal_point_split": False,
                "boundary": "unbounded",
            },
            source_ne=1,
            sink_ne=1,
        )
    with pytest.raises(ValueError, match="spin axes"):
        consume_spin_aware_current(
            {"schema": CURRENT_ASSEMBLER_SCHEMA, "axes": (), "vertex": np.ones((4, 4, 1, 1), complex)},
            np.ones(4, complex),
            np.ones(4, complex),
        )
