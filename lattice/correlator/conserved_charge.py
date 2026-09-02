"""Convention-neutral scalar projection and declared-ratio helpers.

These helpers do not define a hadron operator, spin projector, flavor weight,
charge normalization, or Ward--Takahashi identity. Callers must supply the full
dual tensor and an approved formula definition.
"""

from __future__ import annotations

import hashlib
import json
from numbers import Integral
from typing import Mapping

import numpy as np

SCALAR_PROJECTION_SCHEMA = "lattice.current.explicit-v2v-scalar-projection/v1"
DECLARED_RATIO_SCHEMA = "lattice.current.declared-ratio/v1"


def _complex_array(value, name):
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise TypeError(f"{name} must be array-like") from exc
    if not np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must have a complex dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _real_or_complex_array(value, name):
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise TypeError(f"{name} must be array-like") from exc
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _definition(definition: Mapping) -> tuple[dict, str]:
    if not isinstance(definition, Mapping):
        raise TypeError("definition must be a mapping")
    required = {"schema", "id", "formula", "approval_sha256"}
    if set(definition) != required:
        raise ValueError("definition has missing or unknown fields")
    if definition["schema"] != "lattice.current.observable-definition/v1":
        raise ValueError("definition schema is unsupported")
    for name in ("id", "formula"):
        if not isinstance(definition[name], str) or not definition[name]:
            raise ValueError(f"definition.{name} must be non-empty")
    digest = definition["approval_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest.lower())
    ):
        raise ValueError("definition.approval_sha256 must be a SHA-256")
    normalized = dict(definition)
    normalized["approval_sha256"] = digest.lower()
    identity = hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return normalized, identity


def project_explicit_v2v_scalar(value, dual_weight, *, definition):
    """Apply an explicitly supplied dual tensor to one external V2V matrix.

    ``value`` and ``dual_weight`` must both use axes
    ``(sink_spin, source_spin, sink_ne, source_ne)``. This function performs
    exactly ``sum(dual_weight * value)``: it does not conjugate, transpose, or
    normalize either input.
    """
    value_array = _complex_array(value, "value")
    weight_array = _complex_array(dual_weight, "dual_weight")
    if value_array.ndim != 4 or value_array.shape[:2] != (4, 4):
        raise ValueError("value must have axes (sink_spin, source_spin, sink_ne, source_ne)")
    if weight_array.shape != value_array.shape:
        raise ValueError("dual_weight shape must exactly match value")
    normalized, identity = _definition(definition)
    scalar = np.einsum("abij,abij->", weight_array, value_array, optimize=True)
    return {
        "schema": SCALAR_PROJECTION_SCHEMA,
        "value": scalar,
        "input_axes": (
            "sink_spin",
            "source_spin",
            "sink_ne",
            "source_ne",
        ),
        "operation": "sum(dual_weight * value); no implicit conjugation",
        "definition": normalized,
        "definition_identity": identity,
    }


def build_declared_ratio(
    numerator,
    denominator,
    *,
    definition,
    denominator_axis=None,
    zero_tolerance=0.0,
):
    """Divide arrays only under an explicit approved observable definition.

    ``denominator_axis`` may name one numerator axis along which a lower-rank
    denominator is aligned. With ``None``, numerator and denominator must have
    identical shapes. The function performs no symmetrization, fit, contact
    removal, or plateau selection.
    """
    numerator_array = _real_or_complex_array(numerator, "numerator")
    denominator_array = _real_or_complex_array(denominator, "denominator")
    tolerance = np.asarray(zero_tolerance)
    if (
        tolerance.ndim != 0
        or not np.issubdtype(tolerance.dtype, np.number)
        or np.issubdtype(tolerance.dtype, np.complexfloating)
        or not np.isfinite(tolerance)
        or float(tolerance) < 0
    ):
        raise ValueError("zero_tolerance must be a non-negative finite real scalar")
    if denominator_axis is None:
        if numerator_array.shape != denominator_array.shape:
            raise ValueError("numerator and denominator shapes must match when denominator_axis is None")
        expanded_denominator = denominator_array
    else:
        if isinstance(denominator_axis, bool) or not isinstance(denominator_axis, Integral):
            raise TypeError("denominator_axis must be an integer or None")
        axis = int(denominator_axis)
        if axis < 0:
            axis += numerator_array.ndim
        if not 0 <= axis < numerator_array.ndim:
            raise ValueError("denominator_axis is outside numerator dimensions")
        expected = numerator_array.shape[:axis] + numerator_array.shape[axis + 1 :]
        if denominator_array.shape != expected:
            raise ValueError("denominator shape must equal numerator shape with denominator_axis removed")
        expanded_denominator = np.expand_dims(denominator_array, axis=axis)
    invalid = np.abs(expanded_denominator) <= float(tolerance)
    if np.any(invalid):
        raise ZeroDivisionError("denominator contains values at or below zero_tolerance")
    normalized, identity = _definition(definition)
    ratio = numerator_array / expanded_denominator
    if not np.all(np.isfinite(ratio)):
        raise ValueError("declared ratio contains non-finite values")
    return {
        "schema": DECLARED_RATIO_SCHEMA,
        "value": ratio,
        "operation": "numerator / denominator; no implicit fit or normalization",
        "denominator_axis": denominator_axis,
        "zero_tolerance": float(tolerance),
        "definition": normalized,
        "definition_identity": identity,
    }


__all__ = [
    "SCALAR_PROJECTION_SCHEMA",
    "DECLARED_RATIO_SCHEMA",
    "project_explicit_v2v_scalar",
    "build_declared_ratio",
]
