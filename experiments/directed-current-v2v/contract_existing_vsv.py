#!/usr/bin/env python3
"""Contract a persisted directed Current artifact with an existing rank-6 VSV.

This is an artifact smoke test, not a Ward--Takahashi or physics validation.
It never generates or loads a propagator through the production generator path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Union
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lattice.current_elemental import (  # noqa: E402
    CURRENT_V2V_CONTRACTION_SCHEMA,
    contract_directed_current_v2v,
    load_directed_current_v2v,
)
from lattice.insertion.current import ConservedVectorCurrent  # noqa: E402

SCHEMA = "lattice.current.existing-vsv-v2v-contraction/v1"
EXECUTION_RECORD_SCHEMA = "lattice.current.kunshan-execution-record/v1"
VSV_TIMESLICE_MANIFEST_SCHEMA = "lattice.current.vsv-timeslice-manifest/v1"
CLASSIFICATION = "artifact-smoke-not-physics-validation"
VSV_AXES = [
    "source_time",
    "second_time",
    "sink_spin",
    "source_spin",
    "sink_ne",
    "source_ne",
]
OUTPUT_AXES = [
    "external_sink_spin",
    "external_source_spin",
    "external_sink_ne",
    "external_source_ne",
]


class SmokeError(ValueError):
    """An input or output violates the existing-VSV smoke contract."""


def _fail(message: str) -> None:
    raise SmokeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        _fail(f"{name} must be a 64-hex SHA-256")
    return value.lower()


def _absolute_file(value: Union[Path, str], name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        _fail(f"{name} must be an absolute path")
    path = path.resolve()
    if not path.is_file():
        _fail(f"{name} does not exist as a regular file: {path}")
    return path


def _absolute_new_directory(value: Union[Path, str]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        _fail("result directory must be absolute")
    path = path.resolve()
    if path.exists():
        _fail(f"refusing to reuse existing output directory: {path}")
    return path


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _fail(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        _fail(f"{name} must be positive")
    return value


def _finite_float(value: Any, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise SmokeError(f"{name} must be a finite number") from exc
    if not np.isfinite(value):
        _fail(f"{name} must be a finite number")
    return value


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _semantic_identity(value: dict[str, Any]) -> str:
    semantic = {key: item for key, item in value.items() if key != "manifest_identity"}
    return hashlib.sha256(_canonical_bytes(semantic).rstrip(b"\n")).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class ExistingVSVNpy:
    """Direct, read-only full-file accessor with a declared second time axis."""

    layout = "full-rank6"

    def __init__(
        self,
        path: Path,
        *,
        sha256: str,
        time_axis: str,
        temporal_extent: int,
    ):
        if time_axis not in {"source-relative", "source-sink"}:
            _fail("VSV time-axis semantics must be source-relative or source-sink")
        try:
            values = np.load(path, allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise SmokeError(f"cannot read VSV NPY: {exc}") from exc
        if values.ndim != 6:
            _fail("VSV NPY must have rank 6")
        if values.shape[:4] != (temporal_extent, temporal_extent, 4, 4):
            _fail("VSV NPY must have shape (Lt, Lt, 4, 4, sink_ne, source_ne) matching Current Lt")
        if values.shape[4] <= 0 or values.shape[5] <= 0:
            _fail("VSV NPY eigenvector extents must be positive")
        if not np.issubdtype(values.dtype, np.complexfloating):
            _fail("VSV NPY must have a complex dtype")
        self.path = path
        self.sha256 = sha256
        self.values = values
        self.shape = tuple(int(size) for size in values.shape)
        self.dtype = values.dtype
        self.time_axis = time_axis
        self.temporal_extent = temporal_extent
        self.accesses: list[dict[str, Any]] = []

    def verify_stable(self) -> None:
        if _sha256(self.path) != self.sha256:
            _fail("VSV file changed while it was being consumed")

    def provenance(self) -> dict[str, Any]:
        return {
            "layout": self.layout,
            "path": self.path.as_posix(),
            "sha256": self.sha256,
            "format": "npy",
            "disk_axes": VSV_AXES,
            "time_axis_semantics": self.time_axis,
            "shape": list(self.shape),
            "dtype": self.dtype.str,
            "allow_pickle": False,
            "mmap_mode": "r",
            "finiteness_validation": "accessed-blocks",
            "accessed_disk_indices": self.accesses,
        }

    def get(self, source_time: int, sink_time: int):
        for value, name in ((source_time, "VSV source time"), (sink_time, "VSV sink time")):
            if isinstance(value, bool) or not isinstance(value, Integral):
                _fail(f"{name} must be an integer")
            if not 0 <= int(value) < self.temporal_extent:
                _fail(f"{name} is outside the temporal extent")
        source_time, sink_time = int(source_time), int(sink_time)
        second_index = (
            (sink_time - source_time) % self.temporal_extent if self.time_axis == "source-relative" else sink_time
        )
        block = np.array(self.values[source_time, second_index], copy=True)
        if not np.all(np.isfinite(block)):
            _fail("accessed VSV block contains non-finite values")
        self.accesses.append(
            {
                "source_time": source_time,
                "sink_time": sink_time,
                "disk_indices": [source_time, second_index],
            }
        )
        return block


def _validate_vsv_header(
    path: Path,
    *,
    expected_sha256: str,
    time_axis: str,
    temporal_extent: int,
) -> tuple[ExistingVSVNpy, str]:
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        _fail("VSV SHA-256 does not match --vsv-sha256")
    return (
        ExistingVSVNpy(
            path,
            sha256=actual_sha256,
            time_axis=time_axis,
            temporal_extent=temporal_extent,
        ),
        actual_sha256,
    )


def _load_json_snapshot(path: Path, name: str) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokeError(f"cannot read {name}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{name} must contain a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _load_json(path: Path, name: str) -> dict[str, Any]:
    value, _digest = _load_json_snapshot(path, name)
    return value


def _timeslice_manifest_identity(value: dict[str, Any]) -> str:
    semantic = {key: item for key, item in value.items() if key != "manifest_identity"}
    return hashlib.sha256(_canonical_bytes(semantic).rstrip(b"\n")).hexdigest()


class ExistingVSVTimeslices:
    """Read-only source-timeslice family with a strict per-file hash manifest."""

    layout = "source-timeslices-rank5"

    def __init__(
        self,
        pattern: str,
        hash_manifest_path: Path,
        *,
        configuration: str,
        time_axis: str,
        temporal_extent: int,
    ):
        if time_axis not in {"source-relative", "source-sink"}:
            _fail("VSV time-axis semantics must be source-relative or source-sink")
        if "{source_time:03d}" not in pattern:
            _fail("VSV timeslice pattern must contain {source_time:03d}")
        sample_path = Path(pattern.format(source_time=0)).expanduser()
        if not sample_path.is_absolute():
            _fail("VSV timeslice pattern must be absolute")
        manifest_path = _absolute_file(hash_manifest_path, "VSV timeslice hash manifest")
        manifest, manifest_sha256 = _load_json_snapshot(manifest_path, "VSV timeslice hash manifest")
        required = {
            "schema",
            "version",
            "configuration",
            "layout",
            "time_axis_semantics",
            "temporal_extent",
            "disk_axes",
            "files",
            "manifest_identity",
        }
        if set(manifest) != required:
            _fail("VSV timeslice hash manifest has missing or unknown fields")
        if manifest["schema"] != VSV_TIMESLICE_MANIFEST_SCHEMA or manifest["version"] != 1:
            _fail("VSV timeslice hash manifest schema/version is unsupported")
        if manifest["manifest_identity"] != _timeslice_manifest_identity(manifest):
            _fail("VSV timeslice hash manifest identity is stale or tampered")
        if manifest["configuration"] != configuration:
            _fail("VSV timeslice manifest configuration does not match")
        if manifest["layout"] != self.layout:
            _fail("VSV timeslice manifest layout is unsupported")
        if manifest["time_axis_semantics"] != time_axis:
            _fail("VSV timeslice manifest time-axis semantics do not match")
        if manifest["temporal_extent"] != temporal_extent:
            _fail("VSV timeslice manifest temporal extent does not match Current")
        if manifest["disk_axes"] != VSV_AXES[1:]:
            _fail("VSV timeslice manifest disk axes are unsupported")
        files = manifest["files"]
        if not isinstance(files, list) or len(files) != temporal_extent:
            _fail("VSV timeslice manifest must list every source time exactly once")
        records: dict[int, dict[str, Any]] = {}
        expected_shape = None
        expected_dtype = None
        for record in files:
            if not isinstance(record, dict) or set(record) != {
                "source_time",
                "path",
                "sha256",
                "shape",
                "dtype",
            }:
                _fail("VSV timeslice file record is invalid")
            source_time = record["source_time"]
            if (
                isinstance(source_time, bool)
                or not isinstance(source_time, Integral)
                or not 0 <= int(source_time) < temporal_extent
                or int(source_time) in records
            ):
                _fail("VSV timeslice source times must be unique and complete")
            source_time = int(source_time)
            expected_path = Path(pattern.format(source_time=source_time)).expanduser().resolve()
            path = _absolute_file(record["path"], "VSV timeslice file")
            if path != expected_path:
                _fail("VSV timeslice manifest path does not match the declared pattern")
            digest = _require_hash(record["sha256"], "VSV timeslice file hash")
            if _sha256(path) != digest:
                _fail(f"VSV timeslice SHA-256 mismatch for source time {source_time}")
            try:
                values = np.load(path, allow_pickle=False, mmap_mode="r")
            except (OSError, ValueError) as exc:
                raise SmokeError(f"cannot read VSV timeslice NPY: {exc}") from exc
            shape = tuple(int(size) for size in values.shape)
            if shape != tuple(record["shape"]):
                _fail("VSV timeslice shape does not match its manifest")
            if values.dtype.str != record["dtype"]:
                _fail("VSV timeslice dtype does not match its manifest")
            if values.ndim != 5 or shape[:3] != (temporal_extent, 4, 4):
                _fail("each VSV timeslice must have shape (Lt, 4, 4, sink_ne, source_ne)")
            if shape[3] <= 0 or shape[4] <= 0:
                _fail("VSV timeslice eigenvector extents must be positive")
            if not np.issubdtype(values.dtype, np.complexfloating):
                _fail("VSV timeslice must have a complex dtype")
            if expected_shape is None:
                expected_shape = shape
                expected_dtype = values.dtype
            elif shape != expected_shape or values.dtype != expected_dtype:
                _fail("all VSV timeslices must share shape and dtype")
            records[source_time] = {
                "path": path,
                "sha256": digest,
                "shape": shape,
                "dtype": values.dtype,
            }
        if set(records) != set(range(temporal_extent)):
            _fail("VSV timeslice source-time coverage is incomplete")
        self.pattern = pattern
        self.hash_manifest_path = manifest_path
        self.hash_manifest_sha256 = manifest_sha256
        self.manifest = manifest
        self.records = records
        self.time_axis = time_axis
        self.temporal_extent = temporal_extent
        self.shape = (temporal_extent,) + expected_shape
        self.dtype = expected_dtype
        self.accesses: list[dict[str, Any]] = []
        self.accessed_source_times: set[int] = set()

    def verify_stable(self) -> None:
        if _sha256(self.hash_manifest_path) != self.hash_manifest_sha256:
            _fail("VSV timeslice hash manifest changed while it was being consumed")
        current = _load_json(self.hash_manifest_path, "VSV timeslice hash manifest")
        if current != self.manifest:
            _fail("VSV timeslice hash manifest changed while it was being consumed")
        for source_time, record in self.records.items():
            if _sha256(record["path"]) != record["sha256"]:
                _fail(f"VSV timeslice file changed while it was being consumed: source time {source_time}")

    def provenance(self) -> dict[str, Any]:
        return {
            "layout": self.layout,
            "pattern": self.pattern,
            "hash_manifest_path": self.hash_manifest_path.as_posix(),
            "hash_manifest_sha256": self.hash_manifest_sha256,
            "hash_manifest_identity": self.manifest["manifest_identity"],
            "format": "npy-timeslices",
            "disk_axes": VSV_AXES,
            "time_axis_semantics": self.time_axis,
            "shape": list(self.shape),
            "dtype": self.dtype.str,
            "allow_pickle": False,
            "mmap_mode": "r",
            "finiteness_validation": "accessed-blocks",
            "accessed_disk_indices": self.accesses,
        }

    def get(self, source_time: int, sink_time: int):
        for value, name in (
            (source_time, "VSV source time"),
            (sink_time, "VSV sink time"),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                _fail(f"{name} must be an integer")
            if not 0 <= int(value) < self.temporal_extent:
                _fail(f"{name} is outside the temporal extent")
        source_time, sink_time = int(source_time), int(sink_time)
        second_index = (
            (sink_time - source_time) % self.temporal_extent if self.time_axis == "source-relative" else sink_time
        )
        record = self.records[source_time]
        self.accessed_source_times.add(source_time)
        values = np.load(record["path"], allow_pickle=False, mmap_mode="r")
        block = np.array(values[second_index], copy=True)
        if not np.all(np.isfinite(block)):
            _fail("accessed VSV block contains non-finite values")
        self.accesses.append(
            {
                "source_time": source_time,
                "sink_time": sink_time,
                "file": record["path"].as_posix(),
                "file_sha256": record["sha256"],
                "disk_indices": [source_time, second_index],
            }
        )
        return block


def _runtime_execution_binding() -> dict[str, Any]:
    cluster = os.environ.get("LATTICE_EXECUTION_CLUSTER")
    git_commit = os.environ.get("LATTICE_SOURCE_GIT_COMMIT")
    git_dirty_text = os.environ.get("LATTICE_SOURCE_GIT_DIRTY")
    resources_text = os.environ.get("LATTICE_EXECUTION_RESOURCES_JSON")
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("LATTICE_EXECUTION_JOB_ID")
    if cluster != "kunshan":
        _fail("trusted launcher must set LATTICE_EXECUTION_CLUSTER=kunshan")
    if not git_commit:
        _fail("trusted launcher must set LATTICE_SOURCE_GIT_COMMIT")
    if git_dirty_text not in {"true", "false"}:
        _fail("trusted launcher must set LATTICE_SOURCE_GIT_DIRTY=true or false")
    if not resources_text:
        _fail("trusted launcher must set LATTICE_EXECUTION_RESOURCES_JSON")
    try:
        resources = json.loads(resources_text)
    except json.JSONDecodeError as exc:
        raise SmokeError("trusted launcher resources JSON is invalid") from exc
    if not isinstance(resources, dict) or not resources:
        _fail("trusted launcher resources must be a non-empty object")
    if not job_id:
        _fail("trusted launcher must supply an execution Job ID")
    return {
        "cluster": cluster,
        "git": {"commit": git_commit, "dirty": git_dirty_text == "true"},
        "resources": resources,
        "slurm_job_id": job_id,
    }


def _validate_execution_record(
    path: Path,
    *,
    configuration: str,
    vsv_time_axis: str,
) -> dict[str, Any]:
    record_path = _absolute_file(path, "execution record")
    record, record_sha256 = _load_json_snapshot(record_path, "execution record")
    required = {
        "schema",
        "version",
        "cluster",
        "configuration",
        "git",
        "resources",
        "slurm_job_id",
        "input_hashes",
        "axis_declarations",
        "record_identity",
    }
    if set(record) != required:
        _fail("execution record has missing or unknown fields")
    if record["schema"] != EXECUTION_RECORD_SCHEMA or record["version"] != 1:
        _fail("execution record schema/version is unsupported")
    semantic = {key: value for key, value in record.items() if key != "record_identity"}
    if record["record_identity"] != hashlib.sha256(_canonical_bytes(semantic).rstrip(b"\n")).hexdigest():
        _fail("execution record identity is stale or tampered")
    if record["cluster"] != "kunshan" or record["configuration"] != configuration:
        _fail("execution record cluster/configuration does not match")
    git = record["git"]
    if not isinstance(git, dict) or set(git) != {"commit", "dirty"}:
        _fail("execution record git state is invalid")
    if not isinstance(git["commit"], str) or not git["commit"] or not isinstance(git["dirty"], bool):
        _fail("execution record git state is invalid")
    if not isinstance(record["resources"], dict) or not record["resources"]:
        _fail("execution record resources must be a non-empty object")
    if not isinstance(record["slurm_job_id"], str) or not record["slurm_job_id"]:
        _fail("execution record Slurm Job ID must be non-empty")
    if not isinstance(record["input_hashes"], dict) or not record["input_hashes"]:
        _fail("execution record input_hashes must be a non-empty object")
    axes = record["axis_declarations"]
    if not isinstance(axes, dict) or axes.get("vsv_time_axis") != vsv_time_axis:
        _fail("execution record VSV axis declaration does not match")
    runtime = _runtime_execution_binding()
    for field in ("cluster", "git", "resources", "slurm_job_id"):
        if record[field] != runtime[field]:
            _fail(f"execution record {field} does not match trusted runtime state")
    return {
        "path": record_path,
        "sha256": record_sha256,
        "record": record,
    }


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    current_path = Path(args.current_artifact).expanduser()
    if not current_path.is_absolute():
        _fail("Current artifact path must be absolute")
    current_path = current_path.resolve()
    if not current_path.exists():
        _fail(f"Current artifact does not exist: {current_path}")
    result_dir = _absolute_new_directory(args.result_dir)
    configuration = args.configuration
    if not isinstance(configuration, str) or not configuration:
        _fail("configuration must be a non-empty string")
    expected_gauge = (
        None if args.expected_gauge_sha256 is None else _require_hash(args.expected_gauge_sha256, "expected gauge hash")
    )
    expected_eigenvector = (
        None
        if args.expected_eigenvector_sha256 is None
        else _require_hash(args.expected_eigenvector_sha256, "expected eigenvector hash")
    )
    current = load_directed_current_v2v(
        current_path,
        expected_configuration=configuration,
        expected_gauge_sha256=expected_gauge,
        expected_eigenvector_sha256=expected_eigenvector,
        verify_sources=not args.no_verify_current_sources,
        mmap_mode=None,
    )
    contract = current["contract"]
    temporal_extent = contract["shapes"]["v2v"][1]
    momentum_count = contract["shapes"]["v2v"][2]
    for value, name in (
        (args.source_time, "source_time"),
        (args.sink_time, "sink_time"),
        (args.current_time, "current_time"),
    ):
        if not 0 <= value < temporal_extent:
            _fail(f"{name} must be inside the Current temporal extent")
    if not 0 <= args.momentum_index < momentum_count:
        _fail("momentum_index is outside the Current momentum axis")
    current_source_ne = _positive_int(args.current_source_ne, "current_source_ne")
    current_sink_ne = _positive_int(args.current_sink_ne, "current_sink_ne")
    raw_used_ne = contract["ne"]["used"]
    if current_source_ne > raw_used_ne or current_sink_ne > raw_used_ne:
        _fail("requested Current Ne exceeds the persisted raw Current Ne")
    wilson_r = _finite_float(args.wilson_r, "wilson_r")
    if args.vsv is not None:
        if args.vsv_sha256 is None:
            _fail("--vsv-sha256 is required with --vsv")
        if args.vsv_timeslice_manifest is not None:
            _fail("--vsv-timeslice-manifest cannot be combined with --vsv")
        vsv_path = _absolute_file(args.vsv, "VSV")
        vsv_sha256 = _require_hash(args.vsv_sha256, "VSV hash")
        accessor, _actual_vsv_sha256 = _validate_vsv_header(
            vsv_path,
            expected_sha256=vsv_sha256,
            time_axis=args.vsv_time_axis,
            temporal_extent=temporal_extent,
        )
    else:
        if args.vsv_sha256 is not None:
            _fail("--vsv-sha256 can only be used with --vsv")
        if args.vsv_timeslice_manifest is None:
            _fail("--vsv-timeslice-manifest is required with --vsv-timeslice-pattern")
        accessor = ExistingVSVTimeslices(
            args.vsv_timeslice_pattern,
            args.vsv_timeslice_manifest,
            configuration=configuration,
            time_axis=args.vsv_time_axis,
            temporal_extent=temporal_extent,
        )
    if current_source_ne > accessor.shape[-1]:
        _fail("current_source_ne exceeds the VSV source Ne")
    if current_sink_ne > accessor.shape[-2]:
        _fail("current_sink_ne exceeds the VSV sink Ne")
    execution_record = None
    if not args.dry_run:
        if args.execution_record is None:
            _fail("--execution-record is required for a non-dry smoke")
        execution_record = _validate_execution_record(
            args.execution_record,
            configuration=configuration,
            vsv_time_axis=args.vsv_time_axis,
        )
        record = execution_record["record"]
        if record["axis_declarations"].get("current_direction") != args.current_direction:
            _fail("execution record current direction does not match")
        if record["axis_declarations"].get("vsv_layout") != accessor.layout:
            _fail("execution record VSV layout does not match")
        input_hashes = record["input_hashes"]
        if input_hashes.get("current_artifact_manifest_sha256") != current["verified_files"]["manifest"]["sha256"]:
            _fail("execution record Current artifact hash does not match")
        if accessor.layout == "full-rank6":
            if input_hashes.get("vsv_sha256") != accessor.sha256:
                _fail("execution record VSV hash does not match")
        elif input_hashes.get("vsv_timeslice_manifest_sha256") != accessor.hash_manifest_sha256:
            _fail("execution record VSV timeslice manifest hash does not match")
    return {
        "accessor": accessor,
        "execution_record": execution_record,
        "current": current,
        "result_dir": result_dir,
        "configuration": configuration,
        "temporal_extent": temporal_extent,
        "current_source_ne": current_source_ne,
        "current_sink_ne": current_sink_ne,
        "wilson_r": wilson_r,
    }


def _current_metadata(prepared: dict[str, Any]) -> dict[str, Any]:
    current = prepared["current"]
    manifest = current["manifest"]
    sources = manifest["sources"]
    direction = prepared["args"].current_direction
    return {
        "kind": "conserved-vector",
        "direction": direction,
        "direction_name": ("x", "y", "z", "t")[direction],
        "wilson_r": prepared["wilson_r"],
        "artifact_manifest_path": current["manifest_path"].as_posix(),
        "artifact_manifest_sha256": current["verified_files"]["manifest"]["sha256"],
        "artifact_identity": manifest["artifact_identity"],
        "raw_cache_identity": current["contract"]["cache_identity"],
        "data_sha256": manifest["data"]["sha256"],
        "gauge_sha256": sources["gauge"]["sha256"],
        "eigenvector_sha256": sources["eigenvector"]["sha256"],
        "sources_verified": not prepared["args"].no_verify_current_sources,
    }


def _publish(prepared: dict[str, Any], contraction: dict[str, Any]) -> dict[str, Any]:
    result_dir = prepared["result_dir"]
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = result_dir.parent / f".{result_dir.name}.partial-{uuid.uuid4().hex}"
    try:
        if result_dir.exists():
            _fail(f"refusing to reuse existing output directory: {result_dir}")
        stage.mkdir()
        temporary_result = stage / "contraction.npy"
        with temporary_result.open("wb") as output:
            np.save(output, np.asarray(contraction["value"]), allow_pickle=False)
            output.flush()
            os.fsync(output.fileno())
        result_sha256 = _sha256(temporary_result)
        result_name = f"contraction-{result_sha256}.npy"
        result_path = stage / result_name
        temporary_result.rename(result_path)

        prepared["accessor"].verify_stable()
        _verify_current_stable(prepared["current"])
        execution = prepared["execution_record"]
        if _sha256(execution["path"]) != execution["sha256"]:
            _fail("execution record changed while it was being consumed")

        args = prepared["args"]
        accessor = prepared["accessor"]
        current_manifest = prepared["current"]["manifest"]
        momentum_vector = current_manifest["momenta"][args.momentum_index]
        result_array = np.asarray(contraction["value"])
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "version": 1,
            "status": "contracted",
            "classification": CLASSIFICATION,
            "configuration": prepared["configuration"],
            "times": {
                "source": args.source_time,
                "sink": args.sink_time,
                "current": args.current_time,
            },
            "boundary": prepared["current"]["contract"]["boundary"],
            "momentum": {"index": args.momentum_index, "vector": momentum_vector},
            "ne": {
                "current_source": prepared["current_source_ne"],
                "current_sink": prepared["current_sink_ne"],
                "external_sink": int(result_array.shape[2]),
                "external_source": int(result_array.shape[3]),
            },
            "current": _current_metadata(prepared),
            "vsv": accessor.provenance(),
            "consumer": {
                "schema": CURRENT_V2V_CONTRACTION_SCHEMA,
                "term_contraction": "afAi,bfji,bcjC->acAC",
                "output_axes": OUTPUT_AXES,
                "uses_preloaded_vsv_only": True,
                "terms": list(contraction["terms"]),
            },
            "result": {
                "filename": result_name,
                "sha256": result_sha256,
                "shape": list(result_array.shape),
                "dtype": result_array.dtype.str,
                "axes": OUTPUT_AXES,
                "finite": bool(np.all(np.isfinite(result_array))),
            },
            "producer": {
                "script_path": Path(__file__).resolve().as_posix(),
                "script_sha256": _sha256(Path(__file__).resolve()),
            },
            "execution": {
                "record_path": prepared["execution_record"]["path"].as_posix(),
                "record_sha256": prepared["execution_record"]["sha256"],
                "record_identity": prepared["execution_record"]["record"]["record_identity"],
                "cluster": "kunshan",
                "git": prepared["execution_record"]["record"]["git"],
                "resources": prepared["execution_record"]["record"]["resources"],
                "slurm_job_id": prepared["execution_record"]["record"]["slurm_job_id"],
            },
        }
        if not manifest["result"]["finite"]:
            _fail("contraction result contains non-finite values")
        manifest["manifest_identity"] = _semantic_identity(manifest)
        _atomic_write(stage / "manifest.json", _canonical_bytes(manifest))
        done = {
            "status": "complete",
            "artifact_sha256": {
                result_name: result_sha256,
                "manifest.json": _sha256(stage / "manifest.json"),
            },
        }
        _atomic_write(stage / "DONE", _canonical_bytes(done))
        if result_dir.exists():
            _fail(f"refusing to reuse existing output directory: {result_dir}")
        os.rename(stage, result_dir)
    except FileExistsError as exc:
        shutil.rmtree(stage, ignore_errors=True)
        raise SmokeError(f"another writer published this result directory: {result_dir}") from exc
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def _verify_current_stable(current: dict[str, Any]) -> None:
    files = current["verified_files"]
    manifest = files["manifest"]
    data = files["data"]
    if _sha256(manifest["path"]) != manifest["sha256"]:
        _fail("Current artifact manifest changed while it was being consumed")
    if _sha256(data["path"]) != data["sha256"]:
        _fail("Current artifact data changed while it was being consumed")
    for name, source in files["sources"].items():
        if _sha256(source["path"]) != source["sha256"]:
            _fail(f"Current artifact {name} source changed while it was being consumed")


def _validate_selected_blocks(prepared: dict[str, Any], terms, args: argparse.Namespace) -> None:
    from lattice.insertion.current import resolve_current_term_endpoints

    extent = prepared["temporal_extent"]
    boundary = prepared["current"]["contract"]["boundary"]
    accessor = prepared["accessor"]
    for term in terms:
        endpoints = resolve_current_term_endpoints(
            term,
            anchor_time=args.current_time,
            temporal_extent=extent,
            boundary=boundary,
        )
        outgoing = np.asarray(accessor.get(endpoints["field_time"], args.sink_time))
        incoming = np.asarray(accessor.get(args.source_time, endpoints["bar_time"]))
        if outgoing.shape[3] < prepared["current_source_ne"]:
            _fail("outgoing VSV field Ne is smaller than current_source_ne")
        if incoming.shape[2] < prepared["current_sink_ne"]:
            _fail("incoming VSV bar Ne is smaller than current_sink_ne")


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepared = _prepare(args)
    prepared["args"] = args
    current = prepared["current"]
    conserved = ConservedVectorCurrent(wilson_r=prepared["wilson_r"])
    direction = args.current_direction
    terms = conserved.terms[2 * direction : 2 * direction + 2]
    if args.dry_run:
        _validate_selected_blocks(prepared, terms, args)
        prepared["accessor"].verify_stable()
        _verify_current_stable(current)
        return {
            "status": "dry-run-valid",
            "classification": CLASSIFICATION,
            "configuration": prepared["configuration"],
            "result_dir": prepared["result_dir"].as_posix(),
            "vsv_shape": list(prepared["accessor"].shape),
            "vsv_layout": prepared["accessor"].layout,
            "vsv_time_axis": args.vsv_time_axis,
            "sources_verified": not args.no_verify_current_sources,
            "validated_vsv_accesses": len(prepared["accessor"].accesses),
        }
    contraction = contract_directed_current_v2v(
        terms,
        current["raw"],
        current["contract"],
        prepared["accessor"],
        prepared["accessor"],
        source_time=args.source_time,
        sink_time=args.sink_time,
        anchor_time=args.current_time,
        current_source_ne=prepared["current_source_ne"],
        current_sink_ne=prepared["current_sink_ne"],
        momentum=args.momentum_index,
    )
    prepared["accessor"].verify_stable()
    _verify_current_stable(current)
    execution = prepared["execution_record"]
    if _sha256(execution["path"]) != execution["sha256"]:
        _fail("execution record changed while it was being consumed")
    return _publish(prepared, contraction)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    vsv_input = parser.add_mutually_exclusive_group(required=True)
    vsv_input.add_argument("--vsv", type=Path, help="absolute full rank-6 VSV NPY")
    vsv_input.add_argument(
        "--vsv-timeslice-pattern",
        help="absolute rank-5 source file pattern containing {source_time:03d}",
    )
    parser.add_argument("--vsv-sha256")
    parser.add_argument("--vsv-timeslice-manifest", type=Path)
    parser.add_argument(
        "--vsv-time-axis",
        choices=("source-relative", "source-sink"),
        required=True,
        help="mandatory interpretation of VSV disk axis 1",
    )
    parser.add_argument("--current-artifact", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--source-time", type=int, required=True)
    parser.add_argument("--sink-time", type=int, required=True)
    parser.add_argument("--current-time", type=int, required=True)
    parser.add_argument("--current-source-ne", type=int, required=True)
    parser.add_argument("--current-sink-ne", type=int, required=True)
    parser.add_argument("--momentum-index", type=int, default=0)
    parser.add_argument("--current", choices=("conserved-vector",), default="conserved-vector")
    parser.add_argument(
        "--current-direction",
        type=int,
        choices=(0, 1, 2, 3),
        required=True,
        help="explicit current component: 0=x, 1=y, 2=z, 3=t",
    )
    parser.add_argument("--wilson-r", type=float, default=1.0)
    parser.add_argument("--expected-gauge-sha256")
    parser.add_argument("--expected-eigenvector-sha256")
    parser.add_argument("--execution-record", type=Path)
    parser.add_argument(
        "--no-verify-current-sources",
        action="store_true",
        help="offline/degraded mode; recorded as sources_verified=false",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate hashes, schemas, axes and bounds without contraction or output",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = run(args)
    except (SmokeError, ValueError, TypeError, IndexError, OSError) as exc:
        print(f"existing-VSV smoke error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
