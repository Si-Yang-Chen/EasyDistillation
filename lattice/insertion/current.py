"""Current and density insertion operators with explicit verification."""
from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
import json
from numbers import Integral
from typing import Mapping, Optional, Sequence

import numpy as np

CURRENT_API_VERSION = "1.2.0"
CURRENT_ELEMENTAL_SCHEMA = "lattice.current.raw-spatial-displacement-basis/v1"
CURRENT_DIRECTED_RAW_SCHEMA = "lattice.current.raw-directed-one-link-basis/v1"
CURRENT_TERM_SCHEMA = "lattice.current.term/v1"
CURRENT_ASSEMBLER_SCHEMA = "lattice.current.assembler/v1"

_VECTOR_GAMMA = (1, 2, 4, 8)
_AXIAL_GAMMA = (14, 13, 11, 7)
_ZERO_OFFSET = (0, 0, 0, 0)
_TERM_FIELD_NAMES = frozenset()


def _current_basis_metadata():
    from .gauge_link import DirectedCurrentBasis

    return DirectedCurrentBasis.metadata()


def _current_raw_fingerprint(contract):
    semantic = {key: value for key, value in contract.items() if key != "cache_identity"}
    return sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _current_raw_contract_keys():
    return {
        "schema", "version", "basis", "boundary", "combined_with_current_terms",
        "supports_temporal_point_split", "term_application", "term_schema",
        "assembler_schema", "channels", "shapes", "dtypes", "axes", "ne",
        "cache_identity",
    }


def _json_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _validate_current_raw_contract_metadata(contract):
    if not isinstance(contract, Mapping):
        raise TypeError("raw current contract must be a mapping")
    if set(contract) != _current_raw_contract_keys():
        raise ValueError("raw current contract has missing or unknown metadata")
    if contract["schema"] != CURRENT_DIRECTED_RAW_SCHEMA or contract["version"] != 1:
        raise ValueError("raw current contract schema/version is unsupported")
    if contract["basis"] != _current_basis_metadata():
        raise ValueError("raw current contract directions do not match the v1 basis")
    if contract["boundary"] not in {"periodic", "open"}:
        raise ValueError("raw current contract boundary is invalid")
    if contract["combined_with_current_terms"] is not False:
        raise ValueError("raw current contract must not combine Current terms")
    if contract["supports_temporal_point_split"] is not True:
        raise ValueError("raw current contract must support temporal point splitting")
    if contract["term_application"] != "external-resolver-required":
        raise ValueError("raw current contract term application is invalid")
    if contract["term_schema"] != CURRENT_TERM_SCHEMA:
        raise ValueError("raw current contract term schema is invalid")
    if contract["assembler_schema"] != CURRENT_ASSEMBLER_SCHEMA:
        raise ValueError("raw current contract assembler schema is invalid")
    if contract["channels"] != ["v2v-one-link"]:
        raise ValueError("raw current contract is v2v-one-link only")
    if contract["axes"] != {
        "v2v": ["direction", "time", "momentum", "sink_ne", "source_ne"]
    }:
        raise ValueError("raw current contract axes are invalid")
    if set(contract["shapes"]) != {"v2v"} or set(contract["dtypes"]) != {"v2v"}:
        raise ValueError("raw current contract channel metadata is invalid")
    shape = contract["shapes"]["v2v"]
    if not isinstance(shape, list) or len(shape) != 5:
        raise ValueError("raw current contract v2v shape is invalid")
    shape = [_json_integer(value, "raw current shape") for value in shape]
    if shape[0] != 8 or any(value < 0 for value in shape):
        raise ValueError("raw current contract v2v shape is invalid")
    try:
        dtype = np.dtype(contract["dtypes"]["v2v"])
    except TypeError as exc:
        raise ValueError("raw current contract dtype is invalid") from exc
    if not np.issubdtype(dtype, np.complexfloating):
        raise ValueError("raw current contract dtype must be complex")
    ne = contract["ne"]
    if not isinstance(ne, Mapping) or set(ne) != {
        "available", "used", "source", "sink", "raw_generator_used_ne_is_symmetric"
    }:
        raise ValueError("raw current contract Ne metadata is invalid")
    available = _json_integer(ne["available"], "available raw Ne")
    used = _json_integer(ne["used"], "used raw Ne")
    if available < 0 or not 0 <= used <= available:
        raise ValueError("raw current contract Ne bounds are invalid")
    if ne["source"] != used or ne["sink"] != used or ne["raw_generator_used_ne_is_symmetric"] is not True:
        raise ValueError("raw current contract must have symmetric raw Ne")
    if shape[3:] != [used, used]:
        raise ValueError("raw current contract shape/Ne metadata disagrees")
    identity = contract["cache_identity"]
    if not isinstance(identity, str) or identity != _current_raw_fingerprint(contract):
        raise ValueError("raw current contract cache identity is stale or tampered")
    return {key: contract[key] for key in contract}


def build_current_raw_contract(raw, *, boundary, available_ne, used_ne, momentum_count):
    """Describe generated v2v-only eight-one-link raw data without writing it."""
    if not isinstance(raw, Mapping) or set(raw) != {"v2v"}:
        raise ValueError("directed current raw data must contain exactly v2v")
    if boundary not in {"periodic", "open"}:
        raise ValueError("raw current boundary must be 'periodic' or 'open'")
    available_ne, used_ne = _requested_ne(available_ne, used_ne, "raw")
    momentum_count = _ne_count(momentum_count, "momentum_count")
    value = raw["v2v"]
    shape = _shape(value)
    if len(shape) != 5 or shape[0] != 8 or shape[2:] != (momentum_count, used_ne, used_ne):
        raise ValueError("directed current v2v shape must be (8, Lt, momentum, sink_ne, source_ne)")
    dtype = np.dtype(value.dtype)
    if not np.issubdtype(dtype, np.complexfloating):
        raise TypeError("directed current v2v dtype must be complex")
    contract = {
        "schema": CURRENT_DIRECTED_RAW_SCHEMA,
        "version": 1,
        "basis": _current_basis_metadata(),
        "boundary": boundary,
        "combined_with_current_terms": False,
        "supports_temporal_point_split": True,
        "term_application": "external-resolver-required",
        "term_schema": CURRENT_TERM_SCHEMA,
        "assembler_schema": CURRENT_ASSEMBLER_SCHEMA,
        "channels": ["v2v-one-link"],
        "shapes": {"v2v": list(shape)},
        "dtypes": {"v2v": dtype.str},
        "axes": {"v2v": ["direction", "time", "momentum", "sink_ne", "source_ne"]},
        "ne": {
            "available": available_ne,
            "used": used_ne,
            "source": used_ne,
            "sink": used_ne,
            "raw_generator_used_ne_is_symmetric": True,
        },
    }
    contract["cache_identity"] = _current_raw_fingerprint(contract)
    return contract


def _legacy_spatial_contract_keys():
    return {
        "schema", "representation", "combined_with_current_terms",
        "supports_temporal_point_split", "term_application", "term_schema",
        "assembler_schema", "ne", "np", "shapes", "axes",
    }


def _finite_complex_array(value, name):
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise TypeError(f"{name} must be array-like") from exc
    if not np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} dtype must be complex")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _legacy_count_metadata(contract, key, required_keys):
    metadata = contract[key]
    if not isinstance(metadata, Mapping) or set(metadata) != required_keys:
        raise ValueError(f"legacy spatial contract {key} metadata is invalid")
    return metadata


def _validate_legacy_p2p(value, shape):
    if not isinstance(value, (list, tuple)) or tuple(shape) != (len(value),):
        raise ValueError("legacy spatial p2p shape does not match sparse entries")
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping) or set(entry) != {"type"} | (
            {"indices", "values"} if entry.get("type") == "sparse" else set()
        ):
            raise ValueError(f"legacy spatial p2p entry {index} is invalid")
        if entry["type"] == "identity":
            continue
        if entry["type"] != "sparse":
            raise ValueError(f"legacy spatial p2p entry {index} type is invalid")
        indices = np.asarray(entry["indices"])
        values = _finite_complex_array(entry["values"], f"legacy spatial p2p values {index}")
        if indices.ndim != 2 or indices.shape[1] != 2 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"legacy spatial p2p indices {index} are invalid")
        if values.shape != (indices.shape[0], 3, 3):
            raise ValueError(f"legacy spatial p2p values {index} shape is invalid")


def validate_legacy_spatial_current_raw(raw, contract):
    """Validate the explicit v1.1/v1.2 spatial `calc_all` raw contract."""
    if not isinstance(raw, Mapping) or set(raw) != {"v2v", "v2p", "p2v", "p2p"}:
        raise ValueError("legacy spatial raw data must contain v2v, v2p, p2v, and p2p")
    if not isinstance(contract, Mapping) or set(contract) != _legacy_spatial_contract_keys():
        raise ValueError("legacy spatial contract has missing or unknown metadata")
    if contract["schema"] != CURRENT_ELEMENTAL_SCHEMA:
        raise ValueError("legacy spatial contract schema is invalid")
    if contract["representation"] != "raw-spatial-displacement-basis":
        raise ValueError("legacy spatial contract representation is invalid")
    if contract["combined_with_current_terms"] is not False:
        raise ValueError("legacy spatial contract must not combine Current terms")
    if contract["supports_temporal_point_split"] is not False:
        raise ValueError("legacy spatial contract must not support temporal point splitting")
    if contract["term_application"] != "external-resolver-required":
        raise ValueError("legacy spatial contract term application is invalid")
    if contract["term_schema"] != CURRENT_TERM_SCHEMA:
        raise ValueError("legacy spatial contract term schema is invalid")
    if contract["assembler_schema"] != CURRENT_ASSEMBLER_SCHEMA:
        raise ValueError("legacy spatial contract assembler schema is invalid")
    if contract["axes"] != {
        "v2v": ("displacement", "momentum", "sink_ne", "source_ne"),
        "v2p": ("displacement", "sink_ne", "point", "color"),
        "p2v": ("displacement", "point", "color", "source_ne"),
        "p2p": "sparse-per-displacement",
    }:
        raise ValueError("legacy spatial contract axes are invalid")
    if not isinstance(contract["shapes"], Mapping) or set(contract["shapes"]) != set(raw):
        raise ValueError("legacy spatial contract shapes are invalid")

    ne = _legacy_count_metadata(
        contract, "ne", {
            "available", "used", "source", "sink", "requested_source",
            "requested_sink", "raw_generator_used_ne_is_symmetric",
        },
    )
    available = _json_integer(ne["available"], "available legacy Ne")
    used = _json_integer(ne["used"], "used legacy Ne")
    requested_source = _json_integer(ne["requested_source"], "requested source Ne")
    requested_sink = _json_integer(ne["requested_sink"], "requested sink Ne")
    if available < 0 or not 0 <= used <= available:
        raise ValueError("legacy spatial contract Ne bounds are invalid")
    if ne["source"] != used or ne["sink"] != used or ne["raw_generator_used_ne_is_symmetric"] is not True:
        raise ValueError("legacy spatial contract must have symmetric raw Ne")
    if requested_source > used or requested_sink > used:
        raise ValueError("legacy spatial requested Ne exceeds raw used Ne")

    np_metadata = _legacy_count_metadata(contract, "np", {"available", "used"})
    available_np = _json_integer(np_metadata["available"], "available legacy Np")
    used_np = _json_integer(np_metadata["used"], "used legacy Np")
    if available_np < 0 or not 0 <= used_np <= available_np:
        raise ValueError("legacy spatial contract Np bounds are invalid")

    arrays = {key: _finite_complex_array(raw[key], f"legacy spatial {key}") for key in ("v2v", "v2p", "p2v")}
    actual_shapes = {key: _shape(raw[key]) for key in raw}
    if contract["shapes"] != actual_shapes:
        raise ValueError("legacy spatial contract shapes do not match raw data")
    if arrays["v2v"].ndim != 4 or arrays["v2v"].shape[-2:] != (used, used):
        raise ValueError("legacy spatial v2v shape is invalid")
    if arrays["v2p"].ndim != 4 or arrays["v2p"].shape[-3:] != (used, used_np, 3):
        raise ValueError("legacy spatial v2p shape is invalid")
    if arrays["p2v"].ndim != 4 or arrays["p2v"].shape[-3:] != (used_np, 3, used):
        raise ValueError("legacy spatial p2v shape is invalid")
    if arrays["v2v"].shape[0] != arrays["v2p"].shape[0] or arrays["v2v"].shape[0] != arrays["p2v"].shape[0]:
        raise ValueError("legacy spatial displacement axes are inconsistent")
    _validate_legacy_p2p(raw["p2p"], actual_shapes["p2p"])
    return {
        "schema": CURRENT_ELEMENTAL_SCHEMA,
        "legacy_spatial_only": True,
        "channels": ("v2v", "v2p", "p2v", "p2p"),
        "contract": dict(contract),
    }


def validate_current_raw_contract(raw, contract, *, require_temporal=False):
    """Validate raw data before consuming it; legacy data can never imply time links."""
    if not isinstance(contract, Mapping):
        raise TypeError("raw current contract must be a mapping")
    schema = contract.get("schema")
    if schema == CURRENT_ELEMENTAL_SCHEMA:
        if require_temporal:
            raise ValueError("legacy spatial raw contract cannot supply temporal current links")
        return validate_legacy_spatial_current_raw(raw, contract)
    if schema != CURRENT_DIRECTED_RAW_SCHEMA:
        raise ValueError("unrecognized raw current schema")
    metadata = _validate_current_raw_contract_metadata(contract)
    expected = build_current_raw_contract(
        raw, boundary=metadata["boundary"],
        available_ne=metadata["ne"]["available"],
        used_ne=metadata["ne"]["used"],
        momentum_count=metadata["shapes"]["v2v"][2],
    )
    if metadata != expected:
        raise ValueError("raw current contract does not match its data")
    values = np.asarray(raw["v2v"])
    if not np.all(np.isfinite(values)):
        raise ValueError("directed current raw v2v must contain only finite values")
    return expected


def current_raw_cache_key(key, contract):
    if not isinstance(key, str) or not key:
        raise ValueError("raw current cache key must be a non-empty string")
    validated = _validate_current_raw_contract_metadata(contract)
    return f"{key}:{validated['cache_identity']}"


def _validated_raw_endpoints(endpoints, term, boundary):
    if not isinstance(endpoints, Mapping) or set(endpoints) != {
        "bar_time", "field_time", "link_origin_time", "temporal_point_split", "boundary"
    }:
        raise ValueError("resolved current endpoints are missing or unknown")
    if endpoints["boundary"] != boundary:
        raise ValueError("resolved endpoint boundary must match the raw current contract")
    if endpoints["temporal_point_split"] is not term.temporal_point_split:
        raise ValueError("resolved endpoint temporal flag does not match the current term")
    for name in ("bar_time", "field_time", "link_origin_time"):
        _integer(endpoints[name], name)
    return endpoints


def resolve_directed_current_raw(raw, contract, term, *, endpoints, source_ne, sink_ne, momentum=0):
    """Resolve raw orientation/anchor only; Current weighting remains in the assembler."""
    current_term = _coerce_term(term)
    temporal = current_term.direction == 3 and current_term.link != "none"
    validated = validate_current_raw_contract(raw, contract, require_temporal=temporal)
    if current_term.link == "none":
        raise ValueError("directed raw resolver requires a forward or backward linked current term")
    endpoints = _validated_raw_endpoints(endpoints, current_term, validated["boundary"])
    source_ne = _ne_count(source_ne, "source_ne")
    sink_ne = _ne_count(sink_ne, "sink_ne")
    used_ne = validated["ne"]["used"]
    if source_ne > used_ne or sink_ne > used_ne:
        raise ValueError("requested source_ne/sink_ne exceeds symmetric raw generator used_ne")
    if not isinstance(momentum, Integral) or isinstance(momentum, (bool, np.bool_)):
        raise TypeError("momentum must be an integer")
    momentum = int(momentum)
    if not 0 <= momentum < validated["shapes"]["v2v"][2]:
        raise ValueError("momentum is outside the raw current momentum axis")
    from .gauge_link import DirectedCurrentBasis

    direction = DirectedCurrentBasis.index_for_term(current_term.direction, current_term.link)
    raw_anchor_time = endpoints["bar_time"]
    extent = validated["shapes"]["v2v"][1]
    if not 0 <= raw_anchor_time < extent:
        raise ValueError("resolved raw-basis bar anchor is outside raw current data")
    value = raw["v2v"][direction, raw_anchor_time, momentum, :sink_ne, :source_ne]
    return {
        "value": value,
        "source_ne": source_ne,
        "sink_ne": sink_ne,
        "provenance": {
            "raw_schema": validated["schema"], "cache_identity": validated["cache_identity"],
            "direction": direction, "time": raw_anchor_time, "momentum": momentum,
            "raw_anchor": "bar_endpoint",
        },
    }


def _finite_numeric_scalar(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite numeric scalar")
    arr = np.asarray(value)
    if arr.ndim != 0 or not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must be a finite numeric scalar")
    try:
        finite = np.isfinite(arr)
    except TypeError as exc:
        raise TypeError(f"{name} must be a finite numeric scalar") from exc
    if not bool(finite):
        raise ValueError(f"{name} must be a finite numeric scalar")
    return value


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _offset(value, name):
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must contain four integers")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must contain four integers") from exc
    if len(values) != 4:
        raise ValueError(f"{name} must contain four integers")
    return tuple(_integer(component, f"{name}[{index}]")
                 for index, component in enumerate(values))


@dataclass(frozen=True)
class CurrentTerm:
    coefficient: complex
    direction: int
    displacement: tuple[int, int, int, int]
    gamma_index: int
    link: str = "none"
    wilson_r: Optional[float] = None
    spin_structure: str = "gamma"
    normalization: complex = 1
    bar_offset: tuple[int, int, int, int] = _ZERO_OFFSET
    field_offset: tuple[int, int, int, int] = _ZERO_OFFSET
    link_origin_offset: tuple[int, int, int, int] = _ZERO_OFFSET
    link_dagger: bool = False
    boundary_policy: str = "caller-supplied"
    temporal_point_split: bool = False

    def __post_init__(self):
        _finite_numeric_scalar(self.coefficient, "coefficient")
        _finite_numeric_scalar(self.normalization, "normalization")
        direction = _integer(self.direction, "direction")
        if not -1 <= direction <= 3:
            raise ValueError("direction must be between -1 and 3")
        gamma_index = _integer(self.gamma_index, "gamma_index")
        if not 0 <= gamma_index <= 15:
            raise ValueError("gamma_index must be between 0 and 15")
        displacement = _offset(self.displacement, "displacement")
        bar_offset = _offset(self.bar_offset, "bar_offset")
        field_offset = _offset(self.field_offset, "field_offset")
        link_origin_offset = _offset(self.link_origin_offset, "link_origin_offset")
        if self.link not in {"none", "forward", "backward"}:
            raise ValueError("link must be one of: none, forward, backward")
        if self.wilson_r is not None:
            _finite_numeric_scalar(self.wilson_r, "wilson_r")
        if not isinstance(self.spin_structure, str) or not self.spin_structure:
            raise TypeError("spin_structure must be a non-empty string")
        if not isinstance(self.link_dagger, (bool, np.bool_)):
            raise TypeError("link_dagger must be a bool")
        if self.boundary_policy != "caller-supplied":
            raise ValueError("boundary_policy must be 'caller-supplied'")
        if not isinstance(self.temporal_point_split, (bool, np.bool_)):
            raise TypeError("temporal_point_split must be a bool")
        has_temporal_endpoint = any(
            offset[3] != 0
            for offset in (bar_offset, field_offset, link_origin_offset)
        )
        if has_temporal_endpoint and direction != 3:
            raise ValueError("temporal endpoint offsets require direction=3")
        expected_temporal = direction == 3 and has_temporal_endpoint
        if bool(self.temporal_point_split) != expected_temporal:
            raise ValueError(
                "temporal_point_split must match direction=3 temporal endpoints"
            )
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "gamma_index", gamma_index)
        object.__setattr__(self, "displacement", displacement)
        object.__setattr__(self, "bar_offset", bar_offset)
        object.__setattr__(self, "field_offset", field_offset)
        object.__setattr__(self, "link_origin_offset", link_origin_offset)
        object.__setattr__(self, "link_dagger", bool(self.link_dagger))
        object.__setattr__(self, "temporal_point_split", bool(self.temporal_point_split))

    def as_dict(self):
        return {
            "schema": CURRENT_TERM_SCHEMA,
            "coefficient": self.coefficient,
            "direction": self.direction,
            "displacement": self.displacement,
            "gamma_index": self.gamma_index,
            "link": self.link,
            "wilson_r": self.wilson_r,
            "spin_structure": self.spin_structure,
            "normalization": self.normalization,
            "bar_offset": self.bar_offset,
            "field_offset": self.field_offset,
            "link_origin_offset": self.link_origin_offset,
            "link_dagger": self.link_dagger,
            "boundary_policy": self.boundary_policy,
            "temporal_point_split": self.temporal_point_split,
        }


_TERM_FIELD_NAMES = frozenset(field.name for field in fields(CurrentTerm))


def _coerce_term(term):
    if isinstance(term, CurrentTerm):
        return term
    if not isinstance(term, Mapping):
        raise TypeError("term must be a CurrentTerm or v1 mapping")
    if "schema" not in term:
        raise ValueError("term mapping must include its v1 schema")
    schema = term["schema"]
    if schema != CURRENT_TERM_SCHEMA:
        raise ValueError(f"term schema must be {CURRENT_TERM_SCHEMA}")
    unknown = set(term).difference(_TERM_FIELD_NAMES | {"schema"})
    if unknown:
        raise ValueError("term contains unknown fields: " + ", ".join(sorted(unknown)))
    try:
        values = {name: term[name] for name in _TERM_FIELD_NAMES if name in term}
        return CurrentTerm(**values)
    except KeyError as exc:
        raise ValueError(f"term is missing required field: {exc.args[0]}") from exc
    except TypeError as exc:
        if "required positional argument" in str(exc):
            raise ValueError("term is missing required fields") from exc
        raise


def _array(value, name):
    if value is None:
        raise ValueError(f"{name} is required for verification")
    try:
        arr = np.asarray(value)
    except Exception as exc:
        raise TypeError(f"{name} must be array-like") from exc
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    try:
        finite = np.isfinite(arr)
    except TypeError as exc:
        raise TypeError(f"{name} must contain numeric values") from exc
    if not np.all(finite):
        raise ValueError(f"{name} must contain only finite values")
    return arr

def _finite_real_scalar(value, name):
    arr = np.asarray(value)
    if (arr.ndim != 0 or not np.issubdtype(arr.dtype, np.number)
            or np.issubdtype(arr.dtype, np.complexfloating)):
        raise ValueError(f"{name} must be a finite real scalar")
    scalar = float(arr)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real scalar")
    return scalar

def _tolerances(atol, rtol):
    atol = _finite_real_scalar(atol, "atol")
    rtol = _finite_real_scalar(rtol, "rtol")
    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative")
    return atol, rtol

def _lattice_spacing(value):
    spacing = _finite_real_scalar(value, "lattice_spacing")
    if spacing <= 0:
        raise ValueError("lattice_spacing must be greater than zero")
    return spacing


def _generator_count(generator, available_name, used_name):
    available = getattr(generator, available_name, None)
    used = getattr(generator, used_name, None)
    if isinstance(available, bool) or not isinstance(available, Integral):
        raise TypeError(f"generator.{available_name} must be an integer")
    if isinstance(used, bool) or not isinstance(used, Integral):
        raise TypeError(f"generator.{used_name} must be an integer")
    available, used = int(available), int(used)
    if available < 0 or not 0 <= used <= available:
        raise ValueError(
            f"generator.{used_name} must satisfy 0 <= {used_name} <= "
            f"{available_name} ({available})"
        )
    return available, used


def _shape(value):
    return tuple(int(size) for size in np.shape(value))


def _ne_count(value, name):
    count = _integer(value, name)
    if count < 0:
        raise ValueError(f"{name} must be non-negative")
    return count


def _requested_ne(available, used, axis):
    available = _ne_count(available, f"available_{axis}_ne")
    if used is None:
        used = available
    used = _ne_count(used, f"used_{axis}_ne")
    if used > available:
        raise ValueError(
            f"used_{axis}_ne must not exceed available_{axis}_ne ({available})"
        )
    return available, used


def _boundary_inputs(anchor_time, temporal_extent, boundary):
    anchor_time = _integer(anchor_time, "anchor_time")
    if boundary not in {"periodic", "open", "unbounded"}:
        raise ValueError("boundary must be one of: periodic, open, unbounded")
    if boundary == "unbounded":
        return anchor_time, temporal_extent
    temporal_extent = _integer(temporal_extent, "temporal_extent")
    if temporal_extent <= 0:
        raise ValueError("temporal_extent must be a positive integer")
    return anchor_time, temporal_extent


def resolve_current_term_endpoints(
    term, *, anchor_time, temporal_extent=None, boundary,
):
    """Resolve a v1 current term's temporal endpoints under caller policy."""
    current_term = _coerce_term(term)
    anchor_time, temporal_extent = _boundary_inputs(
        anchor_time, temporal_extent, boundary,
    )
    raw_times = {
        "bar_time": anchor_time + current_term.bar_offset[3],
        "field_time": anchor_time + current_term.field_offset[3],
        "link_origin_time": anchor_time + current_term.link_origin_offset[3],
    }
    if boundary == "periodic":
        times = {
            name: value % temporal_extent for name, value in raw_times.items()
        }
    elif boundary == "open":
        if any(value < 0 or value >= temporal_extent for value in raw_times.values()):
            raise IndexError("current term endpoint is outside the open temporal boundary")
        times = raw_times
    else:
        times = raw_times
    return {
        **times,
        "temporal_point_split": current_term.temporal_point_split,
        "boundary": boundary,
    }


def _spin_matrix_for_term(term):
    """Return the one explicit 4x4 spin action encoded by a current term."""
    current_term = _coerce_term(term)
    from .gamma import gamma

    gamma_matrix = np.asarray(gamma(current_term.gamma_index))
    if gamma_matrix.shape != (4, 4):
        raise ValueError("current gamma matrix must have shape (4, 4)")
    if not np.issubdtype(gamma_matrix.dtype, np.complexfloating):
        gamma_matrix = gamma_matrix.astype(np.complex128)
    if not np.all(np.isfinite(gamma_matrix)):
        raise ValueError("current gamma matrix must contain only finite values")
    if current_term.link == "none":
        return gamma_matrix
    if current_term.wilson_r is None:
        raise ValueError("linked current term requires wilson_r")
    identity = np.eye(4, dtype=gamma_matrix.dtype)
    if current_term.link == "forward":
        return current_term.wilson_r * identity - gamma_matrix
    return current_term.wilson_r * identity + gamma_matrix


def _validate_spin_raw_result(resolved, *, source_ne, sink_ne):
    """Validate one unweighted, spinless elemental contribution from a callback."""
    if not isinstance(resolved, Mapping):
        raise TypeError("raw elemental resolver result must be a mapping")
    missing = {"value", "source_ne", "sink_ne"}.difference(resolved)
    if missing:
        raise TypeError(
            "raw elemental resolver result is missing: " + ", ".join(sorted(missing))
        )
    resolved_source = _ne_count(resolved["source_ne"], "raw resolver source_ne")
    resolved_sink = _ne_count(resolved["sink_ne"], "raw resolver sink_ne")
    if resolved_source != source_ne or resolved_sink != sink_ne:
        raise ValueError("raw resolver source_ne/sink_ne must exactly match requested values")
    value = _finite_complex_array(resolved["value"], "raw elemental contribution")
    if value.ndim != 2 or value.shape != (sink_ne, source_ne):
        raise ValueError(
            "raw elemental contribution must have shape "
            f"(sink_ne, source_ne) = ({sink_ne}, {source_ne})"
        )
    if value.size == 0:
        raise ValueError("raw elemental contribution must be non-empty")
    return value


def resolve_current_term_spin(
    term, raw_resolver, *, endpoints, source_ne, sink_ne,
):
    """Resolve one term into `(sink_spin, source_spin, sink_ne, source_ne)`.

    ``raw_resolver`` owns raw link/displacement selection and returns a complex
    spinless V2V contribution.  This bridge applies exactly one explicit spin
    matrix; it never applies a term coefficient or normalization.
    """
    if not callable(raw_resolver):
        raise TypeError("raw_resolver must be callable")
    current_term = _coerce_term(term)
    source_ne = _ne_count(source_ne, "source_ne")
    sink_ne = _ne_count(sink_ne, "sink_ne")
    raw_result = raw_resolver(
        current_term.as_dict(), endpoints=endpoints,
        source_ne=source_ne, sink_ne=sink_ne,
    )
    raw_value = _validate_spin_raw_result(
        raw_result, source_ne=source_ne, sink_ne=sink_ne,
    )
    spin_matrix = _spin_matrix_for_term(current_term)
    value = spin_matrix[:, :, None, None] * raw_value[None, None, :, :]
    return {
        "value": value,
        "source_ne": source_ne,
        "sink_ne": sink_ne,
        "provenance": {
            "spin_structure": current_term.spin_structure,
            "spin_axes": ("sink_spin", "source_spin"),
            "raw": raw_result.get("provenance"),
        },
    }


def make_spin_aware_current_resolver(raw_resolver):
    """Adapt a spinless raw elemental callback for ``assemble_current_terms``."""
    if not callable(raw_resolver):
        raise TypeError("raw_resolver must be callable")

    def resolver(term, *, endpoints, source_ne, sink_ne):
        return resolve_current_term_spin(
            term, raw_resolver, endpoints=endpoints,
            source_ne=source_ne, sink_ne=sink_ne,
        )

    return resolver


def assemble_spin_aware_current(current_or_terms, raw_resolver, **kwargs):
    """Assemble a Current's spin-aware V2V vertex without duplicating weights."""
    terms = current_or_terms.terms if isinstance(current_or_terms, Current) else current_or_terms
    return assemble_current_terms(
        terms, make_spin_aware_current_resolver(raw_resolver), **kwargs,
    )


def spin_aware_current_adapter(assembled):
    """Expose assembled V2V data for a legacy-style spin/elemental consumer.

    The legacy ``quark_diagram.Current`` path cannot directly consume this
    object: it requires a time-indexed storage service and separate legacy
    displacement/operator rows.  The returned ``vertex`` is the exact eager
    replacement shape for a minimal consumer.
    """
    if not isinstance(assembled, Mapping):
        raise TypeError("assembled current must be a mapping")
    if assembled.get("schema") != CURRENT_ASSEMBLER_SCHEMA:
        raise ValueError("assembled current has an unsupported schema")
    vertex = _finite_complex_array(assembled.get("value"), "assembled spin-aware vertex")
    if vertex.ndim != 4 or vertex.shape[:2] != (4, 4):
        raise ValueError(
            "assembled spin-aware vertex must have shape "
            "(sink_spin, source_spin, sink_ne, source_ne)"
        )
    ne = assembled.get("ne")
    if not isinstance(ne, Mapping):
        raise TypeError("assembled current is missing Ne provenance")
    try:
        sink_ne = _ne_count(ne["sink"]["used"], "assembled sink Ne")
        source_ne = _ne_count(ne["source"]["used"], "assembled source Ne")
    except (KeyError, TypeError) as exc:
        raise TypeError("assembled current has invalid Ne provenance") from exc
    if vertex.shape[2:] != (sink_ne, source_ne):
        raise ValueError("assembled spin-aware vertex shape disagrees with Ne provenance")
    return {
        "schema": CURRENT_ASSEMBLER_SCHEMA,
        "api_version": CURRENT_API_VERSION,
        "vertex": vertex,
        "axes": ("sink_spin", "source_spin", "sink_ne", "source_ne"),
        "ne": ne,
        "term_count": assembled["term_count"],
        "terms": assembled["terms"],
    }


def legacy_current_vertex_adapter(vertices_by_time):
    """Adapt equal-time assembled V2V vertices to the legacy ``get(t)`` protocol.

    Temporal point-split terms require distinct bar/field propagator endpoints
    and must use :func:`lattice.current_elemental.contract_directed_current_v2v`.
    """
    from lattice.quark_diagram import CurrentVertexAdapter

    return CurrentVertexAdapter(vertices_by_time)



def _spin_weights(value, name):
    weights = _finite_complex_array(value, name)
    if weights.shape != (4,):
        raise ValueError(f"{name} must have shape (4,)")
    return weights


def consume_spin_aware_current(adapter, sink_spin, source_spin):
    """Minimal eager consumer for the adapter; inputs are explicit spin weights."""
    if not isinstance(adapter, Mapping):
        raise TypeError("spin-aware adapter must be a mapping")
    if adapter.get("schema") != CURRENT_ASSEMBLER_SCHEMA:
        raise ValueError("spin-aware adapter has an unsupported schema")
    if adapter.get("axes") != (
        "sink_spin", "source_spin", "sink_ne", "source_ne",
    ):
        raise ValueError("spin axes in spin-aware adapter are invalid")
    vertex = _finite_complex_array(adapter.get("vertex"), "spin-aware adapter vertex")
    if vertex.ndim != 4 or vertex.shape[:2] != (4, 4):
        raise ValueError("spin-aware adapter vertex has invalid spin axes")
    sink_spin = _spin_weights(sink_spin, "sink_spin")
    source_spin = _spin_weights(source_spin, "source_spin")
    return np.einsum("a,abij,b->ij", sink_spin, vertex, source_spin)


def assemble_current_terms(
    terms, resolver, *, available_source_ne, available_sink_ne,
    used_source_ne=None, used_sink_ne=None, anchor_time=0,
    temporal_extent=None, boundary="unbounded",
):
    """Combine externally resolved elemental terms with the unique v1 weighting."""
    if not callable(resolver):
        raise TypeError("resolver must be callable")
    try:
        terms = tuple(terms)
    except TypeError as exc:
        raise TypeError("terms must be an iterable of current terms") from exc
    if not terms:
        raise ValueError("terms must be non-empty")
    source_available, source_used = _requested_ne(
        available_source_ne, used_source_ne, "source",
    )
    sink_available, sink_used = _requested_ne(
        available_sink_ne, used_sink_ne, "sink",
    )
    anchor_time, temporal_extent = _boundary_inputs(
        anchor_time, temporal_extent, boundary,
    )

    assembled = None
    expected_shape = None
    records = []
    for index, raw_term in enumerate(terms):
        current_term = _coerce_term(raw_term)
        term_mapping = current_term.as_dict()
        endpoints = resolve_current_term_endpoints(
            current_term,
            anchor_time=anchor_time,
            temporal_extent=temporal_extent,
            boundary=boundary,
        )
        resolved = resolver(
            term_mapping,
            endpoints=endpoints,
            source_ne=source_used,
            sink_ne=sink_used,
        )
        if not isinstance(resolved, Mapping):
            raise TypeError(f"resolver result for term {index} must be a mapping")
        missing = {"value", "source_ne", "sink_ne"}.difference(resolved)
        if missing:
            raise TypeError(
                f"resolver result for term {index} is missing: "
                + ", ".join(sorted(missing))
            )
        resolved_source = _ne_count(
            resolved["source_ne"], f"resolver source_ne for term {index}",
        )
        resolved_sink = _ne_count(
            resolved["sink_ne"], f"resolver sink_ne for term {index}",
        )
        if resolved_source != source_used or resolved_sink != sink_used:
            raise ValueError(
                "resolver source_ne/sink_ne must exactly match requested values"
            )
        value = _array(resolved["value"], f"resolver value for term {index}")
        if expected_shape is None:
            expected_shape = value.shape
        elif value.shape != expected_shape:
            raise ValueError(
                f"resolver value shape {value.shape} for term {index} does not "
                f"match {expected_shape}"
            )
        weighted = current_term.coefficient * current_term.normalization * value
        assembled = weighted.copy() if assembled is None else assembled + weighted
        records.append({
            "schema": term_mapping["schema"],
            "endpoints": endpoints,
            "resolver_provenance": resolved.get("provenance"),
        })

    return {
        "schema": CURRENT_ASSEMBLER_SCHEMA,
        "api_version": CURRENT_API_VERSION,
        "value": assembled,
        "term_count": len(terms),
        "ne": {
            "source": {"available": source_available, "used": source_used},
            "sink": {"available": sink_available, "used": sink_used},
        },
        "boundary": boundary,
        "anchor_time": anchor_time,
        "terms": tuple(records),
    }


def lattice_divergence(current, *, site_axes: Optional[Sequence[int]] = None,
                       periodic: bool = True, lattice_spacing=1.0):
    """Backward divergence of current with direction axis 0 and four site axes."""
    arr = _array(current, "current")
    spacing = _lattice_spacing(lattice_spacing)
    if arr.ndim < 5 or arr.shape[0] != 4:
        raise ValueError("current must have shape (4, ...) and four site axes")
    site_axes = (1, 2, 3, 4) if site_axes is None else tuple(site_axes)
    if len(site_axes) != 4:
        raise ValueError("site_axes must contain one axis for each direction")
    if any(isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer))
           for axis in site_axes):
        raise ValueError("site_axes must contain integer axes")
    if len(set(site_axes)) != 4:
        raise ValueError("site_axes must contain unique axes")
    out = np.zeros(arr.shape[1:], dtype=np.result_type(arr, np.complex128))
    for mu, full_axis in enumerate(site_axes):
        axis = int(full_axis) - 1
        if axis < 0 or axis >= out.ndim:
            raise ValueError(f"site axis {full_axis} is invalid for current shape {arr.shape}")
        if periodic:
            previous = np.roll(arr[mu], 1, axis=axis)
        else:
            previous = np.zeros_like(arr[mu])
            sl_dst = [slice(None)] * previous.ndim
            sl_src = [slice(None)] * previous.ndim
            sl_dst[axis] = slice(1, None)
            sl_src[axis] = slice(None, -1)
            previous[tuple(sl_dst)] = arr[mu][tuple(sl_src)]
        out = out + arr[mu] - previous
    return out / spacing

def _zero_result(value, condition, atol, rtol):
    arr = _array(value, condition)
    atol, _ = _tolerances(atol, rtol)
    max_abs = float(np.max(np.abs(arr)))
    return {"condition": condition, "passed": bool(np.all(np.abs(arr) <= atol)),
            "max_abs": max_abs, "array": arr}

def verify_wt(*, divergence=None, current=None, site_axes=None,
              periodic: bool = True, lattice_spacing=1.0,
              atol=1e-10, rtol=1e-7):
    if divergence is None:
        if current is None:
            raise ValueError("divergence or current is required for WT verification")
        divergence = lattice_divergence(
            current, site_axes=site_axes, periodic=periodic,
            lattice_spacing=lattice_spacing,
        )
    return _zero_result(divergence, "WT divergence = 0", atol, rtol)

def verify_pcac(*, axial_divergence=None, axial_current=None,
                pseudoscalar=None, mass=None, improvement_residual=0,
                site_axes=None, periodic: bool = True, lattice_spacing=1.0,
                atol=1e-10, rtol=1e-7):
    if axial_divergence is None and axial_current is None:
        raise ValueError("axial_divergence or axial_current is required for PCAC verification")
    if axial_divergence is not None and axial_current is not None:
        raise ValueError("provide exactly one of axial_divergence and axial_current")
    if pseudoscalar is None:
        raise ValueError("pseudoscalar is required for PCAC verification")
    if mass is None:
        raise ValueError("mass is required for PCAC verification")
    if axial_current is not None:
        lhs = lattice_divergence(
            axial_current, site_axes=site_axes, periodic=periodic,
            lattice_spacing=lattice_spacing,
        )
    else:
        lhs = _array(axial_divergence, "axial_divergence")
    density = _array(pseudoscalar, "pseudoscalar")
    if lhs.shape != density.shape:
        raise ValueError(
            f"PCAC arrays have incompatible shapes: {lhs.shape} and {density.shape}"
        )
    mass_arr = _array(mass, "mass")
    if mass_arr.shape not in ((), lhs.shape):
        raise ValueError(
            f"mass has incompatible shape {mass_arr.shape}; expected scalar or {lhs.shape}"
        )
    improvement = _array(improvement_residual, "improvement_residual")
    if improvement.shape == ():
        improvement = np.full(
            lhs.shape, improvement.item(),
            dtype=np.result_type(lhs, density, mass_arr, improvement),
        )
    elif improvement.shape != lhs.shape:
        raise ValueError(
            "improvement_residual has incompatible shape "
            f"{improvement.shape}; expected scalar or {lhs.shape}"
        )
    rhs = 2 * mass_arr * density + improvement
    atol, rtol = _tolerances(atol, rtol)
    delta = lhs - rhs
    max_abs = float(np.max(np.abs(delta)))
    return {
        "condition": "div A = 2 m P + E",
        "passed": bool(np.all(np.abs(delta) <= atol + rtol * np.abs(rhs))),
        "max_abs": max_abs,
        "lhs": lhs,
        "rhs": rhs,
    }

class Current:
    name = "current"
    elemental_key = "v2v"
    verification_checks = frozenset()
    requires_z = False
    requires_wt = False
    requires_pcac = False
    gamma_indices = _VECTOR_GAMMA
    def __init__(self, generator=None, *, z=None, lattice_spacing=1.0, wilson_r=1.0):
        self.generator, self.z = generator, z
        self.lattice_spacing, self.wilson_r = lattice_spacing, wilson_r
    @property
    def terms(self):
        displacement = (0, 0, 0, 0)
        return tuple(CurrentTerm(1, mu, displacement, self.gamma_indices[mu])
                     for mu in range(4))
    def _operator(self):
        return {"api_version": CURRENT_API_VERSION,
                "name": self.name, "kind": self.__class__.__name__,
                "gamma_indices": self.gamma_indices}
    def compute_elemental(
        self, generator=None, t=None, *, used_source_ne=None,
        used_sink_ne=None, **kwargs,
    ):
        gen = generator if generator is not None else self.generator
        if gen is None:
            raise TypeError("compute_elemental requires a CurrentElementalGenerator")
        calc_all = getattr(gen, "calc_all", None)
        if not callable(calc_all):
            raise TypeError("generator must provide callable calc_all(t)")
        if t is None:
            raise TypeError("compute_elemental requires t for generator.calc_all(t)")
        available_ne, used_ne = _generator_count(gen, "Ne", "usedNe")
        requested_source_ne = (
            used_ne if used_source_ne is None
            else _requested_ne(used_ne, used_source_ne, "source")[1]
        )
        requested_sink_ne = (
            used_ne if used_sink_ne is None
            else _requested_ne(used_ne, used_sink_ne, "sink")[1]
        )
        available_np, used_np = _generator_count(gen, "Np", "usedNp")
        results = calc_all(t)
        if not isinstance(results, Mapping):
            raise TypeError("generator.calc_all(t) must return a mapping")
        required_keys = {"v2v", "v2p", "p2v", "p2p"}
        missing = required_keys.difference(results)
        if missing:
            raise TypeError(
                "generator.calc_all(t) is missing elemental keys: "
                + ", ".join(sorted(missing))
            )
        shapes = {key: _shape(results[key]) for key in required_keys}
        if len(shapes["v2v"]) < 2 or shapes["v2v"][-2:] != (used_ne, used_ne):
            raise ValueError("v2v elemental shape does not match generator.usedNe")
        if len(shapes["v2p"]) < 3 or shapes["v2p"][-3:-1] != (used_ne, used_np):
            raise ValueError("v2p elemental shape does not match generator.usedNe/usedNp")
        if len(shapes["p2v"]) < 3 or shapes["p2v"][-3:-1] != (used_np, 3) or shapes["p2v"][-1] != used_ne:
            raise ValueError("p2v elemental shape does not match generator.usedNp/usedNe")
        contract = {
            "schema": CURRENT_ELEMENTAL_SCHEMA,
            "representation": "raw-spatial-displacement-basis",
            "combined_with_current_terms": False,
            "supports_temporal_point_split": False,
            "term_application": "external-resolver-required",
            "term_schema": CURRENT_TERM_SCHEMA,
            "assembler_schema": CURRENT_ASSEMBLER_SCHEMA,
            "ne": {
                "available": available_ne,
                "used": used_ne,
                "source": used_ne,
                "sink": used_ne,
                "requested_source": requested_source_ne,
                "requested_sink": requested_sink_ne,
                "raw_generator_used_ne_is_symmetric": True,
            },
            "np": {"available": available_np, "used": used_np},
            "shapes": shapes,
            "axes": {
                "v2v": ("displacement", "momentum", "sink_ne", "source_ne"),
                "v2p": ("displacement", "sink_ne", "point", "color"),
                "p2v": ("displacement", "point", "color", "source_ne"),
                "p2p": "sparse-per-displacement",
            },
        }
        return {"api_version": CURRENT_API_VERSION,
                "schema": CURRENT_ELEMENTAL_SCHEMA,
                "operator": self._operator(),
                "terms": tuple(term.as_dict() for term in self.terms),
                "elemental": results[self.elemental_key],
                "all_elementals": results, "contract": contract, "time": t}
    @staticmethod
    def verify_z(z=None, *, atol=1e-10, rtol=1e-7):
        if z is None:
            raise ValueError("z (measured renormalization factor) is required for Z=1 verification")
        return _zero_result(_array(z, "z") - 1, "Z = 1", atol, rtol)
    def verify(self, *, z=None, renormalization=None, divergence=None, current=None,
               axial_divergence=None, axial_current=None, pseudoscalar=None,
               mass=None, improvement_residual=0, site_axes=None,
               periodic: bool = True, lattice_spacing=None,
               atol=1e-10, rtol=1e-7, **kwargs):
        measured_z = z if z is not None else renormalization
        if measured_z is None:
            measured_z = self.z
        spacing = self.lattice_spacing if lattice_spacing is None else lattice_spacing
        wt_supplied = divergence is not None or current is not None
        pcac_supplied = any(value is not None for value in
                            (axial_divergence, axial_current, pseudoscalar, mass))
        result = {}
        if "z" in self.verification_checks and (self.requires_z or measured_z is not None):
            result["Z"] = self.verify_z(measured_z, atol=atol, rtol=rtol)
        if "wt" in self.verification_checks and (self.requires_wt or wt_supplied):
            result["WT"] = verify_wt(
                divergence=divergence, current=current, site_axes=site_axes,
                periodic=periodic, lattice_spacing=spacing, atol=atol, rtol=rtol,
            )
        if "pcac" in self.verification_checks and (self.requires_pcac or pcac_supplied):
            result["PCAC"] = verify_pcac(
                axial_divergence=axial_divergence, axial_current=axial_current,
                pseudoscalar=pseudoscalar, mass=mass,
                improvement_residual=improvement_residual, site_axes=site_axes,
                periodic=periodic, lattice_spacing=spacing,
                atol=atol, rtol=rtol,
            )
        return result

class LocalVectorCurrent(Current):
    name = "local_vector"

class LocalAxialCurrent(Current):
    name = "local_axial"
    gamma_indices = _AXIAL_GAMMA
    verification_checks = frozenset({"pcac"})
    requires_pcac = True
    @property
    def terms(self):
        displacement = (0, 0, 0, 0)
        coefficients = (1, -1, 1, -1)
        return tuple(
            CurrentTerm(
                coefficients[mu], mu, displacement, self.gamma_indices[mu],
                spin_structure=f"gamma_{mu}gamma_5",
            )
            for mu in range(4)
        )

class ConservedVectorCurrent(Current):
    name = "conserved_vector"
    verification_checks = frozenset({"z", "wt"})
    requires_z = True
    requires_wt = True
    @property
    def terms(self):
        terms = []
        for mu in range(4):
            fwd = tuple(1 if i == mu else 0 for i in range(4))
            back = tuple(-1 if i == mu else 0 for i in range(4))
            terms.extend((
                CurrentTerm(
                    -0.5, mu, fwd, self.gamma_indices[mu],
                    "forward", self.wilson_r, f"r-gamma_{mu}",
                    bar_offset=_ZERO_OFFSET,
                    field_offset=fwd,
                    link_origin_offset=_ZERO_OFFSET,
                    link_dagger=False,
                    temporal_point_split=mu == 3,
                ),
                CurrentTerm(
                    0.5, mu, back, self.gamma_indices[mu],
                    "backward", self.wilson_r, f"r+gamma_{mu}",
                    bar_offset=fwd,
                    field_offset=_ZERO_OFFSET,
                    link_origin_offset=_ZERO_OFFSET,
                    link_dagger=True,
                    temporal_point_split=mu == 3,
                ),
            ))
        return tuple(terms)
    def _operator(self):
        op = super()._operator()
        op.update({"wilson_r": self.wilson_r, "point_split": True,
                   "link_orientations": ("forward", "backward")})
        return op

class PseudoScalarDensity(Current):
    name = "pseudoscalar_density"
    gamma_indices = (15,)
    verification_checks = frozenset({"pcac"})
    requires_pcac = True
    @property
    def terms(self):
        return (CurrentTerm(
            1, -1, (0, 0, 0, 0), 15, spin_structure="gamma_5",
        ),)

__all__ = [
    "CURRENT_API_VERSION", "CURRENT_ELEMENTAL_SCHEMA", "CURRENT_DIRECTED_RAW_SCHEMA", "CURRENT_TERM_SCHEMA",
    "CURRENT_ASSEMBLER_SCHEMA", "build_current_raw_contract",
    "validate_legacy_spatial_current_raw", "validate_current_raw_contract",
    "current_raw_cache_key", "resolve_directed_current_raw",
    "resolve_current_term_spin", "make_spin_aware_current_resolver",
    "assemble_spin_aware_current", "spin_aware_current_adapter",
    "legacy_current_vertex_adapter", "consume_spin_aware_current", "Current", "CurrentTerm", "LocalVectorCurrent",
    "LocalAxialCurrent", "ConservedVectorCurrent", "PseudoScalarDensity",
    "resolve_current_term_endpoints", "assemble_current_terms",
    "lattice_divergence", "verify_wt", "verify_pcac",
]
