"""Persistent directed-current V2V artifacts and term-wise VSV contraction."""

from __future__ import annotations

from hashlib import sha256
import json
from numbers import Integral
import os
from pathlib import Path
from typing import Mapping
from uuid import uuid4

import numpy as np

from .insertion.current import (
    Current,
    resolve_current_term_endpoints,
    resolve_current_term_spin,
    resolve_directed_current_raw,
    validate_current_raw_contract,
)

CURRENT_V2V_ARTIFACT_SCHEMA = "lattice.current.directed-v2v-artifact/v1"
CURRENT_V2V_CONTRACTION_SCHEMA = "lattice.current.v2v-termwise-contraction/v1"
CURRENT_V2V_PAIR_CONTRACTION_SCHEMA = "lattice.current.v2v-term-pair-contraction/v1"
_ARTIFACT_KEYS = {
    "schema",
    "version",
    "configuration",
    "data",
    "raw_contract",
    "momenta",
    "sources",
    "consumer",
    "artifact_identity",
}


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _manifest_identity(manifest):
    semantic = {key: value for key, value in manifest.items() if key != "artifact_identity"}
    return sha256(_canonical_json(semantic)).hexdigest()


def _sha256_file(path):
    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(path, name):
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{name} source file does not exist: {source}")
    return {"path": source.as_posix(), "sha256": _sha256_file(source)}


def _configuration(value):
    if not isinstance(value, str) or not value:
        raise ValueError("configuration must be a non-empty string")
    return value


def _momenta(values, expected_count):
    try:
        values = list(values)
    except TypeError as exc:
        raise TypeError("momenta must be an iterable of three-integer vectors") from exc
    if len(values) != expected_count:
        raise ValueError("momenta count does not match the raw momentum axis")
    result = []
    for index, momentum in enumerate(values):
        try:
            momentum = list(momentum)
        except TypeError as exc:
            raise TypeError(f"momentum {index} must contain three integers") from exc
        if len(momentum) != 3:
            raise ValueError(f"momentum {index} must contain three integers")
        if any(isinstance(component, bool) or not isinstance(component, Integral) for component in momentum):
            raise TypeError(f"momentum {index} must contain three integers")
        result.append([int(component) for component in momentum])
    return result


def _host_array(value):
    if hasattr(value, "get") and callable(value.get):
        value = value.get()
    return np.ascontiguousarray(np.asarray(value))


def save_directed_current_v2v(
    directory,
    raw,
    contract,
    *,
    configuration,
    momenta,
    gauge_source,
    eigenvector_source,
    overwrite=False,
):
    """Atomically publish one content-addressed directed-current V2V artifact.

    The manifest is the commit point. It binds the NPY bytes, strict raw
    contract, momentum vectors, configuration key, and input file hashes.
    """
    if not isinstance(raw, Mapping) or set(raw) != {"v2v"}:
        raise ValueError("directed current raw data must contain exactly v2v")
    raw = {"v2v": _host_array(raw["v2v"])}
    validated = validate_current_raw_contract(raw, contract, require_temporal=True)
    configuration = _configuration(configuration)
    momenta = _momenta(momenta, validated["shapes"]["v2v"][2])
    sources = {
        "gauge": _source_record(gauge_source, "gauge"),
        "eigenvector": _source_record(eigenvector_source, "eigenvector"),
    }

    destination = Path(directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"current artifact already exists: {manifest_path}")

    temporary_data = destination / f".v2v-{uuid4().hex}.tmp"
    try:
        with temporary_data.open("wb") as output:
            np.save(output, raw["v2v"], allow_pickle=False)
            output.flush()
            os.fsync(output.fileno())
        data_sha256 = _sha256_file(temporary_data)
        data_name = f"v2v-{data_sha256}.npy"
        data_path = destination / data_name
        if data_path.exists():
            if _sha256_file(data_path) != data_sha256:
                raise ValueError("existing content-addressed V2V file is corrupt")
            temporary_data.unlink()
        else:
            os.replace(temporary_data, data_path)
    finally:
        if temporary_data.exists():
            temporary_data.unlink()

    manifest = {
        "schema": CURRENT_V2V_ARTIFACT_SCHEMA,
        "version": 1,
        "configuration": configuration,
        "data": {
            "format": "npy",
            "filename": data_name,
            "sha256": data_sha256,
            "allow_pickle": False,
        },
        "raw_contract": validated,
        "momenta": momenta,
        "sources": sources,
        "consumer": {
            "schema": CURRENT_V2V_CONTRACTION_SCHEMA,
            "raw_axes": [
                "direction",
                "bar_time",
                "momentum",
                "bar_ne",
                "field_ne",
            ],
            "vsv_axes": ["sink_spin", "source_spin", "sink_ne", "source_ne"],
            "term_contraction": "afAi,bfji,bcjC->acAC",
            "requires_preloaded_vsv": True,
        },
    }
    manifest["artifact_identity"] = _manifest_identity(manifest)

    temporary_manifest = destination / f".manifest-{uuid4().hex}.tmp"
    try:
        with temporary_manifest.open("wb") as output:
            output.write(_canonical_json(manifest) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        if overwrite:
            os.replace(temporary_manifest, manifest_path)
        else:
            try:
                os.link(temporary_manifest, manifest_path)
            except FileExistsError as exc:
                raise FileExistsError(f"current artifact already exists: {manifest_path}") from exc
            temporary_manifest.unlink()
    finally:
        if temporary_manifest.exists():
            temporary_manifest.unlink()
    return manifest_path


def _validate_sha256(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _read_manifest(path):
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("current artifact manifest is unreadable") from exc
    if not isinstance(manifest, Mapping) or set(manifest) != _ARTIFACT_KEYS:
        raise ValueError("current artifact manifest has missing or unknown fields")
    if manifest["schema"] != CURRENT_V2V_ARTIFACT_SCHEMA or manifest["version"] != 1:
        raise ValueError("current artifact schema/version is unsupported")
    identity = manifest["artifact_identity"]
    if identity != _manifest_identity(manifest):
        raise ValueError("current artifact identity is stale or tampered")
    return dict(manifest)


def load_directed_current_v2v(
    path,
    *,
    expected_configuration=None,
    expected_gauge_sha256=None,
    expected_eigenvector_sha256=None,
    verify_sources=True,
    mmap_mode="r",
):
    """Load and fully validate a persisted directed-current V2V artifact."""
    if not isinstance(verify_sources, bool):
        raise TypeError("verify_sources must be a bool")
    if mmap_mode not in {None, "r"}:
        raise ValueError("mmap_mode must be None or read-only 'r'")
    path = Path(path).expanduser().resolve()
    manifest_path = path / "manifest.json" if path.is_dir() else path
    manifest_digest = _sha256_file(manifest_path)
    manifest = _read_manifest(manifest_path)
    configuration = _configuration(manifest["configuration"])
    if expected_configuration is not None and configuration != expected_configuration:
        raise ValueError("current artifact configuration does not match the expected key")

    data = manifest["data"]
    if not isinstance(data, Mapping) or set(data) != {
        "format",
        "filename",
        "sha256",
        "allow_pickle",
    }:
        raise ValueError("current artifact data metadata is invalid")
    if data["format"] != "npy" or data["allow_pickle"] is not False:
        raise ValueError("current artifact data format is unsupported")
    data_digest = _validate_sha256(data["sha256"], "current data hash")
    filename = data["filename"]
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("current artifact data filename is unsafe")
    if filename != f"v2v-{data_digest}.npy":
        raise ValueError("current artifact data filename is not content-addressed")
    data_path = manifest_path.parent / filename
    if not data_path.is_file() or _sha256_file(data_path) != data_digest:
        raise ValueError("current artifact data hash does not match the manifest")

    sources = manifest["sources"]
    verified_source_paths = {}
    if not isinstance(sources, Mapping) or set(sources) != {"gauge", "eigenvector"}:
        raise ValueError("current artifact source metadata is invalid")
    for name in ("gauge", "eigenvector"):
        source = sources[name]
        if not isinstance(source, Mapping) or set(source) != {"path", "sha256"}:
            raise ValueError(f"current artifact {name} source metadata is invalid")
        if not isinstance(source["path"], str) or not Path(source["path"]).is_absolute():
            raise ValueError(f"current artifact {name} source path must be absolute")
        source_digest = _validate_sha256(source["sha256"], f"current artifact {name} source hash")
        if verify_sources:
            source_path = Path(source["path"])
            if not source_path.is_file() or _sha256_file(source_path) != source_digest:
                raise ValueError(f"current artifact {name} source file does not match its hash")
            verified_source_paths[name] = source_path
    expected = {
        "gauge": expected_gauge_sha256,
        "eigenvector": expected_eigenvector_sha256,
    }
    for name, digest in expected.items():
        if digest is not None:
            _validate_sha256(digest, f"expected {name} source hash")
            if sources[name]["sha256"] != digest:
                raise ValueError(f"current artifact {name} source hash does not match")

    contract = manifest["raw_contract"]
    values = np.load(data_path, allow_pickle=False, mmap_mode=mmap_mode)
    raw = {"v2v": values}
    validated = validate_current_raw_contract(raw, contract, require_temporal=True)
    _momenta(manifest["momenta"], validated["shapes"]["v2v"][2])
    consumer = manifest["consumer"]
    if consumer != {
        "schema": CURRENT_V2V_CONTRACTION_SCHEMA,
        "raw_axes": [
            "direction",
            "bar_time",
            "momentum",
            "bar_ne",
            "field_ne",
        ],
        "vsv_axes": ["sink_spin", "source_spin", "sink_ne", "source_ne"],
        "term_contraction": "afAi,bfji,bcjC->acAC",
        "requires_preloaded_vsv": True,
    }:
        raise ValueError("current artifact consumer contract is unsupported")
    if _sha256_file(manifest_path) != manifest_digest or _read_manifest(manifest_path) != manifest:
        raise ValueError("current artifact manifest changed while it was being validated")
    if _sha256_file(data_path) != data_digest:
        raise ValueError("current artifact data changed while it was being validated")
    for name, source_path in verified_source_paths.items():
        if _sha256_file(source_path) != sources[name]["sha256"]:
            raise ValueError(f"current artifact {name} source changed while it was being validated")
    return {
        "raw": raw,
        "contract": validated,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "verified_files": {
            "manifest": {"path": manifest_path, "sha256": manifest_digest},
            "data": {"path": data_path, "sha256": data_digest},
            "sources": {
                name: {"path": source_path, "sha256": sources[name]["sha256"]}
                for name, source_path in verified_source_paths.items()
            },
        },
    }


def _time(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _vsv_block(value, name):
    if hasattr(value, "get") and callable(value.get):
        value = value.get()
    try:
        value = np.asarray(value)
    except Exception as exc:
        raise TypeError(f"{name} must be array-like") from exc
    if value.ndim != 4 or value.shape[:2] != (4, 4):
        raise ValueError(f"{name} must have axes (sink_spin, source_spin, sink_ne, source_ne)")
    if not np.issubdtype(value.dtype, np.complexfloating):
        raise TypeError(f"{name} must have a complex dtype")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    return value


def contract_directed_current_v2v(
    current_or_terms,
    raw,
    raw_contract,
    incoming_vsv,
    outgoing_vsv,
    *,
    source_time,
    sink_time,
    anchor_time,
    current_source_ne,
    current_sink_ne,
    momentum=0,
):
    """Contract point-split Current terms with already-loaded VSV accessors.

    For every term this computes ``S(field->sink) J(bar<-field)
    S(source->bar)`` and only then applies the term coefficient and sums. The
    accessors must already be loaded and expose ``get(t_source, t_sink)``;
    this function never calls ``load`` or any propagator generator.
    """
    validated = validate_current_raw_contract(raw, raw_contract, require_temporal=True)
    terms = current_or_terms.terms if isinstance(current_or_terms, Current) else tuple(current_or_terms)
    if not terms:
        raise ValueError("current terms must be non-empty")
    for name, accessor in (("incoming_vsv", incoming_vsv), ("outgoing_vsv", outgoing_vsv)):
        if not callable(getattr(accessor, "get", None)):
            raise TypeError(f"{name} must provide get(t_source, t_sink)")
    source_time = _time(source_time, "source_time")
    sink_time = _time(sink_time, "sink_time")
    anchor_time = _time(anchor_time, "anchor_time")
    extent = validated["shapes"]["v2v"][1]
    for name, time in (("source_time", source_time), ("sink_time", sink_time)):
        if not 0 <= time < extent:
            raise ValueError(f"{name} must be inside the raw temporal extent")
    boundary = validated["boundary"]

    result = None
    expected_shape = None
    records = []
    for index, term in enumerate(terms):
        endpoints = resolve_current_term_endpoints(
            term,
            anchor_time=anchor_time,
            temporal_extent=extent,
            boundary=boundary,
        )

        def raw_resolver(term_mapping, **kwargs):
            return resolve_directed_current_raw(
                raw,
                validated,
                term_mapping,
                momentum=momentum,
                **kwargs,
            )

        resolved = resolve_current_term_spin(
            term,
            raw_resolver,
            endpoints=endpoints,
            source_ne=current_source_ne,
            sink_ne=current_sink_ne,
        )
        vertex = resolved["value"]
        outgoing = _vsv_block(
            outgoing_vsv.get(endpoints["field_time"], sink_time),
            "outgoing VSV block",
        )
        incoming = _vsv_block(
            incoming_vsv.get(source_time, endpoints["bar_time"]),
            "incoming VSV block",
        )
        if outgoing.shape[3] < current_source_ne:
            raise ValueError("outgoing VSV field Ne is smaller than current_source_ne")
        if incoming.shape[2] < current_sink_ne:
            raise ValueError("incoming VSV bar Ne is smaller than current_sink_ne")
        outgoing = outgoing[..., :current_source_ne]
        incoming = incoming[..., :current_sink_ne, :]
        term_value = np.einsum(
            "afAi,bfji,bcjC->acAC",
            outgoing,
            vertex,
            incoming,
            optimize=True,
        )
        if expected_shape is None:
            expected_shape = term_value.shape
        elif term_value.shape != expected_shape:
            raise ValueError("term-wise VSV contractions must have identical external shapes")
        term_mapping = term.as_dict() if hasattr(term, "as_dict") else term
        coefficient = term_mapping["coefficient"]
        normalization = term_mapping.get("normalization", 1)
        weighted = coefficient * normalization * term_value
        result = weighted.copy() if result is None else result + weighted
        records.append(
            {
                "term_index": index,
                "endpoints": endpoints,
                "coefficient": coefficient,
                "normalization": normalization,
                "incoming_vsv_get": [source_time, endpoints["bar_time"]],
                "outgoing_vsv_get": [endpoints["field_time"], sink_time],
                "raw": resolved["provenance"]["raw"],
            }
        )

    return {
        "schema": CURRENT_V2V_CONTRACTION_SCHEMA,
        "value": result,
        "axes": (
            "external_sink_spin",
            "external_source_spin",
            "external_sink_ne",
            "external_source_ne",
        ),
        "source_time": source_time,
        "sink_time": sink_time,
        "anchor_time": anchor_time,
        "boundary": boundary,
        "momentum": int(momentum),
        "terms": tuple(records),
        "raw_cache_identity": validated["cache_identity"],
        "uses_preloaded_vsv_only": True,
    }


def contract_directed_current_pair_v2v(
    first_current_or_terms,
    first_raw,
    first_raw_contract,
    second_current_or_terms,
    second_raw,
    second_raw_contract,
    vsv,
    *,
    first_anchor_time,
    second_anchor_time,
    first_field_ne,
    first_bar_ne,
    second_field_ne,
    second_bar_ne,
    first_momentum=0,
    second_momentum=0,
):
    """Close two point-split Current vertices with preloaded VSV blocks.

    This computes the ordered, unflavored and unsigned connected trace for
    every pair of Current terms.  For first term ``A`` and second term ``B``
    the two propagators are ``S(field_A -> bar_B)`` and
    ``S(field_B -> bar_A)``.  The exact contraction is
    ``bfji,ackl,afki,bcjl->`` before multiplying the two terms' coefficients
    and normalizations.  No Wick sign, flavor factor, volume normalization,
    conjugation, real-part selection, or source averaging is implicit.

    ``vsv`` must already be loaded and expose ``get(t_source, t_sink)``.  This
    function never loads or generates propagators.
    """
    first_validated = validate_current_raw_contract(first_raw, first_raw_contract, require_temporal=True)
    second_validated = validate_current_raw_contract(second_raw, second_raw_contract, require_temporal=True)
    first_shape = first_validated["shapes"]["v2v"]
    second_shape = second_validated["shapes"]["v2v"]
    if first_shape[1] != second_shape[1]:
        raise ValueError("paired Current raw temporal extents must match")
    if first_validated["boundary"] != second_validated["boundary"]:
        raise ValueError("paired Current raw boundaries must match")
    if not callable(getattr(vsv, "get", None)):
        raise TypeError("vsv must provide get(t_source, t_sink)")

    first_terms = (
        first_current_or_terms.terms if isinstance(first_current_or_terms, Current) else tuple(first_current_or_terms)
    )
    second_terms = (
        second_current_or_terms.terms
        if isinstance(second_current_or_terms, Current)
        else tuple(second_current_or_terms)
    )
    if not first_terms or not second_terms:
        raise ValueError("both Current term collections must be non-empty")

    extent = first_shape[1]
    boundary = first_validated["boundary"]
    first_anchor_time = _time(first_anchor_time, "first_anchor_time")
    second_anchor_time = _time(second_anchor_time, "second_anchor_time")
    for name, time in (
        ("first_anchor_time", first_anchor_time),
        ("second_anchor_time", second_anchor_time),
    ):
        if not 0 <= time < extent:
            raise ValueError(f"{name} must be inside the raw temporal extent")
    ne_counts = {
        "first_field_ne": _time(first_field_ne, "first_field_ne"),
        "first_bar_ne": _time(first_bar_ne, "first_bar_ne"),
        "second_field_ne": _time(second_field_ne, "second_field_ne"),
        "second_bar_ne": _time(second_bar_ne, "second_bar_ne"),
    }
    if any(value <= 0 for value in ne_counts.values()):
        raise ValueError("paired Current Ne counts must be positive")

    result = None
    records = []
    for first_index, first_term in enumerate(first_terms):
        first_endpoints = resolve_current_term_endpoints(
            first_term,
            anchor_time=first_anchor_time,
            temporal_extent=extent,
            boundary=boundary,
        )

        def first_raw_resolver(term_mapping, **kwargs):
            return resolve_directed_current_raw(
                first_raw,
                first_validated,
                term_mapping,
                momentum=first_momentum,
                **kwargs,
            )

        first_resolved = resolve_current_term_spin(
            first_term,
            first_raw_resolver,
            endpoints=first_endpoints,
            source_ne=ne_counts["first_field_ne"],
            sink_ne=ne_counts["first_bar_ne"],
        )
        for second_index, second_term in enumerate(second_terms):
            second_endpoints = resolve_current_term_endpoints(
                second_term,
                anchor_time=second_anchor_time,
                temporal_extent=extent,
                boundary=boundary,
            )

            def second_raw_resolver(term_mapping, **kwargs):
                return resolve_directed_current_raw(
                    second_raw,
                    second_validated,
                    term_mapping,
                    momentum=second_momentum,
                    **kwargs,
                )

            second_resolved = resolve_current_term_spin(
                second_term,
                second_raw_resolver,
                endpoints=second_endpoints,
                source_ne=ne_counts["second_field_ne"],
                sink_ne=ne_counts["second_bar_ne"],
            )
            first_to_second = _vsv_block(
                vsv.get(
                    first_endpoints["field_time"],
                    second_endpoints["bar_time"],
                ),
                "first-to-second VSV block",
            )
            second_to_first = _vsv_block(
                vsv.get(
                    second_endpoints["field_time"],
                    first_endpoints["bar_time"],
                ),
                "second-to-first VSV block",
            )
            if (
                first_to_second.shape[2] < ne_counts["second_bar_ne"]
                or first_to_second.shape[3] < ne_counts["first_field_ne"]
            ):
                raise ValueError("first-to-second VSV Ne extents are smaller than requested")
            if (
                second_to_first.shape[2] < ne_counts["first_bar_ne"]
                or second_to_first.shape[3] < ne_counts["second_field_ne"]
            ):
                raise ValueError("second-to-first VSV Ne extents are smaller than requested")
            first_to_second = first_to_second[
                ...,
                : ne_counts["second_bar_ne"],
                : ne_counts["first_field_ne"],
            ]
            second_to_first = second_to_first[
                ...,
                : ne_counts["first_bar_ne"],
                : ne_counts["second_field_ne"],
            ]
            first_vertex = first_resolved["value"]
            second_vertex = second_resolved["value"]
            pair_value = np.einsum(
                "bfji,ackl,afki,bcjl->",
                first_to_second,
                second_to_first,
                first_vertex,
                second_vertex,
                optimize=True,
            )
            first_mapping = first_term.as_dict() if hasattr(first_term, "as_dict") else first_term
            second_mapping = second_term.as_dict() if hasattr(second_term, "as_dict") else second_term
            first_weight = first_mapping["coefficient"] * first_mapping.get("normalization", 1)
            second_weight = second_mapping["coefficient"] * second_mapping.get("normalization", 1)
            weighted = np.asarray(first_weight * second_weight * pair_value)
            result = weighted.copy() if result is None else result + weighted
            records.append(
                {
                    "first_term_index": first_index,
                    "second_term_index": second_index,
                    "first_endpoints": first_endpoints,
                    "second_endpoints": second_endpoints,
                    "first_coefficient": first_mapping["coefficient"],
                    "first_normalization": first_mapping.get("normalization", 1),
                    "second_coefficient": second_mapping["coefficient"],
                    "second_normalization": second_mapping.get("normalization", 1),
                    "first_to_second_vsv_get": [
                        first_endpoints["field_time"],
                        second_endpoints["bar_time"],
                    ],
                    "second_to_first_vsv_get": [
                        second_endpoints["field_time"],
                        first_endpoints["bar_time"],
                    ],
                    "first_raw": first_resolved["provenance"]["raw"],
                    "second_raw": second_resolved["provenance"]["raw"],
                }
            )

    return {
        "schema": CURRENT_V2V_PAIR_CONTRACTION_SCHEMA,
        "value": result,
        "axes": (),
        "operation": (
            "sum_terms bfji,ackl,afki,bcjl->; ordered connected trace; "
            "no implicit Wick sign, flavor factor, normalization, conjugation, "
            "real-part selection, or source averaging"
        ),
        "first_anchor_time": first_anchor_time,
        "second_anchor_time": second_anchor_time,
        "boundary": boundary,
        "momenta": {
            "first": int(first_momentum),
            "second": int(second_momentum),
        },
        "ne": ne_counts,
        "term_pairs": tuple(records),
        "raw_cache_identities": {
            "first": first_validated["cache_identity"],
            "second": second_validated["cache_identity"],
        },
        "uses_preloaded_vsv_only": True,
    }


__all__ = [
    "CURRENT_V2V_ARTIFACT_SCHEMA",
    "CURRENT_V2V_CONTRACTION_SCHEMA",
    "CURRENT_V2V_PAIR_CONTRACTION_SCHEMA",
    "save_directed_current_v2v",
    "load_directed_current_v2v",
    "contract_directed_current_v2v",
    "contract_directed_current_pair_v2v",
]
