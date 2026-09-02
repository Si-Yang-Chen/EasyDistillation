"""Tests for convention-neutral conserved-charge primitives."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from lattice.correlator.conserved_charge import (
    build_declared_ratio,
    project_explicit_v2v_scalar,
)
from lattice.current_elemental import contract_directed_current_v2v
from lattice.insertion.current import ConservedVectorCurrent, build_current_raw_contract


def _definition(identifier="fixture"):
    return {
        "schema": "lattice.current.observable-definition/v1",
        "id": identifier,
        "formula": "fixture explicit formula",
        "approval_sha256": hashlib.sha256(b"approved fixture").hexdigest(),
    }


def test_explicit_dual_projection_has_no_hidden_conjugation():
    value = np.arange(4 * 4 * 2 * 3, dtype=float).reshape(4, 4, 2, 3).astype(complex)
    value += 1j * (value + 1)
    weight = np.full(value.shape, 2 - 3j, dtype=complex)
    value_copy = value.copy()
    weight_copy = weight.copy()
    result = project_explicit_v2v_scalar(value, weight, definition=_definition())
    expected = np.sum(weight * value)
    np.testing.assert_allclose(result["value"], expected)
    assert result["operation"] == "sum(dual_weight * value); no implicit conjugation"
    assert (
        result["definition_identity"]
        == project_explicit_v2v_scalar(value, weight, definition=_definition())["definition_identity"]
    )
    np.testing.assert_array_equal(value, value_copy)
    np.testing.assert_array_equal(weight, weight_copy)


def test_projection_rejects_shape_dtype_and_definition_errors():
    value = np.ones((4, 4, 1, 1), dtype=complex)
    with pytest.raises(ValueError, match="shape"):
        project_explicit_v2v_scalar(value, np.ones((4, 4, 1, 2), complex), definition=_definition())
    with pytest.raises(TypeError, match="complex dtype"):
        project_explicit_v2v_scalar(value.real, value, definition=_definition())
    bad = _definition()
    bad["unknown"] = True
    with pytest.raises(ValueError, match="unknown"):
        project_explicit_v2v_scalar(value, value, definition=bad)


def test_declared_ratio_exact_shapes_and_axis_alignment():
    numerator = np.arange(2 * 3, dtype=float).reshape(2, 3) + 1
    denominator = np.array([2.0, 4.0])
    result = build_declared_ratio(
        numerator,
        denominator,
        denominator_axis=1,
        definition=_definition("ratio"),
    )
    np.testing.assert_allclose(result["value"], numerator / denominator[:, None])
    identical = build_declared_ratio(
        numerator,
        np.full_like(numerator, 2),
        definition=_definition("same-shape"),
    )
    np.testing.assert_allclose(identical["value"], numerator / 2)


def test_declared_ratio_rejects_zero_and_shape_mismatch():
    numerator = np.ones((2, 3))
    with pytest.raises(ZeroDivisionError, match="zero_tolerance"):
        build_declared_ratio(
            numerator,
            np.array([1.0, 0.0]),
            denominator_axis=1,
            definition=_definition(),
        )
    with pytest.raises(ValueError, match="shape"):
        build_declared_ratio(
            numerator,
            np.ones(3),
            denominator_axis=1,
            definition=_definition(),
        )
    with pytest.raises(ValueError, match="shapes must match"):
        build_declared_ratio(
            numerator,
            np.ones(2),
            definition=_definition(),
        )


class _Accessor:
    def __init__(self, blocks):
        self.blocks = blocks
        self.calls = []

    def get(self, source_time, sink_time):
        self.calls.append((source_time, sink_time))
        return self.blocks[(source_time, sink_time)]


def test_projection_wraps_endpoint_aware_temporal_current_without_changing_it():
    raw_values = np.zeros((8, 3, 1, 1, 1), dtype=complex)
    raw_values[6, 2, 0, 0, 0] = 2 + 1j
    raw_values[7, 0, 0, 0, 0] = 3 - 2j
    raw = {"v2v": raw_values}
    contract = build_current_raw_contract(
        raw,
        boundary="periodic",
        available_ne=1,
        used_ne=1,
        momentum_count=1,
    )
    blocks = {}
    for source in range(3):
        for sink in range(3):
            values = np.arange(16, dtype=float).reshape(4, 4, 1, 1)
            blocks[(source, sink)] = (1 + source + 2 * sink + values + 1j).astype(complex)
    incoming = _Accessor(blocks)
    outgoing = _Accessor(blocks)
    assembled = contract_directed_current_v2v(
        ConservedVectorCurrent().terms[6:8],
        raw,
        contract,
        incoming,
        outgoing,
        source_time=0,
        sink_time=1,
        anchor_time=2,
        current_source_ne=1,
        current_sink_ne=1,
    )
    value_copy = assembled["value"].copy()
    weight = np.zeros_like(assembled["value"])
    weight[0, 0, 0, 0] = 1
    scalar = project_explicit_v2v_scalar(assembled["value"], weight, definition=_definition("component-00"))
    np.testing.assert_allclose(scalar["value"], assembled["value"][0, 0, 0, 0])
    assert outgoing.calls == [(0, 1), (2, 1)]
    assert incoming.calls == [(0, 2), (0, 0)]
    np.testing.assert_array_equal(assembled["value"], value_copy)
