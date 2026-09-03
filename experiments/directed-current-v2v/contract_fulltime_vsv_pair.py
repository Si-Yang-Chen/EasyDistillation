#!/usr/bin/env python3
"""Produce a full-current-time Wilson J4xJ4 V2V matrix for one configuration.

The result is a raw ordered, connected, unflavored and unsigned matrix with
axes ``(first_current_anchor, second_current_anchor)``.  This driver consumes
an already generated directed-current artifact and an audited full-time VSV
source-time/rank-slab dataset.  It never generates propagators and never adds
Wick signs, flavor factors, normalization, conjugation, real-part selection or
source averaging.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
import uuid

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for search_path in (ROOT, HERE):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import contract_existing_vsv as single  # noqa: E402
import contract_existing_vsv_pair as pair  # noqa: E402
from lattice.current_elemental import (  # noqa: E402
    CURRENT_V2V_PAIR_CONTRACTION_SCHEMA,
    contract_directed_current_pair_v2v,
    load_directed_current_v2v,
)
from lattice.insertion.current import ConservedVectorCurrent  # noqa: E402

SCHEMA = "lattice.current.fulltime-vsv-v2v-pair-matrix/v1"
EXECUTION_RECORD_SCHEMA = "lattice.current.kunshan-execution-record/v1"
DIRECTIONS = ("x", "y", "z", "t")
VSV_LAYOUT = "fulltime-source-time-rank-slab"
REQUIRED_SOURCE_FILES = {
    "experiments/directed-current-v2v/contract_existing_vsv.py",
    "experiments/directed-current-v2v/contract_existing_vsv_pair.py",
    "experiments/directed-current-v2v/contract_fulltime_vsv_pair.py",
    "experiments/directed-current-v2v/prepare_fulltime_execution_record.py",
    "experiments/directed-current-v2v/run_fulltime_vsv_pair_ensemble.py",
    "lattice/__init__.py",
    "lattice/current_elemental.py",
    "lattice/generator/elemental.py",
    "lattice/insertion/__init__.py",
    "lattice/insertion/current.py",
    "lattice/insertion/gauge_link.py",
}


class FullTimeError(ValueError):
    """Raised when a full-time input violates the production contract."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(_canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_identity(value: dict[str, Any]) -> str:
    semantic = {key: item for key, item in value.items() if key != "manifest_identity"}
    return hashlib.sha256(_canonical(semantic)).hexdigest()


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise FullTimeError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FullTimeError(f"manifest is not a JSON object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def _require_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise FullTimeError(f"{name} must be a SHA-256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise FullTimeError(f"{name} must be a SHA-256 hex string") from exc
    return value.lower()


@dataclass
class FullTimeVSVSlabAccessor:
    """Read-only accessor for absolute-time VSV rank slabs.

    A source-time directory stores one source time per file family and splits
    the sink-time axis over the temporal MPI ranks.  Each request reads one
    local row from one mmap and returns a detached finite NumPy block.
    """

    directory: Path
    configuration: str
    verify_hashes: bool = False

    def __post_init__(self) -> None:
        self.directory = self.directory.expanduser().resolve()
        self.manifest_path = self.directory / "manifest.json"
        self.manifest, self.manifest_sha256 = _load_manifest(self.manifest_path)
        required = {
            "version",
            "layout",
            "product",
            "time_convention",
            "global_lattice",
            "grid_size",
            "local_lattice",
            "source_times",
            "tail_shape",
            "dtype",
        }
        missing = sorted(required.difference(self.manifest))
        if missing:
            raise FullTimeError(f"VSV manifest is missing fields: {', '.join(missing)}")
        if self.manifest["version"] != 1 or self.manifest["layout"] != "source-time-rank-slab":
            raise FullTimeError("unsupported full-time VSV manifest layout/version")
        if self.manifest["product"] != "VSV":
            raise FullTimeError("full-time slab directory is not a VSV dataset")
        if self.manifest["time_convention"] != "absolute-global-source-and-sink":
            raise FullTimeError("full-time VSV requires absolute-global-source-and-sink")
        self.temporal_extent = int(self.manifest["global_lattice"][3])
        self.local_time = int(self.manifest["local_lattice"][3])
        self.temporal_ranks = int(self.manifest["grid_size"][3])
        if tuple(self.manifest["grid_size"][:3]) != (1, 1, 1):
            raise FullTimeError("full-time VSV currently requires a time-only MPI grid")
        if self.local_time * self.temporal_ranks != self.temporal_extent:
            raise FullTimeError("VSV local/global temporal extents are inconsistent")
        self.tail_shape = tuple(int(value) for value in self.manifest["tail_shape"])
        if self.tail_shape[:2] != (4, 4) or len(self.tail_shape) != 4:
            raise FullTimeError("VSV tail_shape must be (4, 4, sink_ne, source_ne)")
        self.expected_shape = (self.local_time, *self.tail_shape)
        self.expected_dtype = np.dtype(self.manifest["dtype"])
        self.source_times = tuple(sorted({int(value) for value in self.manifest["source_times"]}))
        if self.source_times != tuple(range(self.temporal_extent)):
            raise FullTimeError("full-time VSV source_times must cover every global time")
        if not self.configuration:
            raise FullTimeError("configuration must be non-empty")
        self._records: dict[tuple[int, int], dict[str, Any]] = {}
        self._accesses: list[dict[str, Any]] = []
        for source_time in self.source_times:
            for rank in range(self.temporal_ranks):
                path = self._path(source_time, rank)
                if not path.is_file():
                    raise FullTimeError(f"missing full-time VSV slab: {path}")
                try:
                    values = np.load(path, mmap_mode="r", allow_pickle=False)
                except (OSError, ValueError) as exc:
                    raise FullTimeError(f"cannot load VSV slab {path}: {exc}") from exc
                if values.shape != self.expected_shape or values.dtype != self.expected_dtype:
                    raise FullTimeError(f"VSV slab shape/dtype mismatch at {path}: {values.shape}, {values.dtype}")
                expected_bytes = int(values.offset) + int(values.nbytes)
                if path.stat().st_size != expected_bytes:
                    raise FullTimeError(f"VSV slab byte size mismatch: {path}")
                record = {
                    "path": path.as_posix(),
                    "bytes": int(path.stat().st_size),
                    "shape": list(values.shape),
                    "dtype": values.dtype.str,
                }
                if self.verify_hashes:
                    record["sha256"] = _sha256(path)
                self._records[(source_time, rank)] = record

    def _path(self, source_time: int, rank: int) -> Path:
        return self.directory / f"{self.configuration}.t{source_time:03d}.rank{rank:04d}.npy"

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.temporal_extent, self.temporal_extent, *self.tail_shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.expected_dtype

    def get(self, source_time: int, sink_time: int) -> np.ndarray:
        if not 0 <= int(source_time) < self.temporal_extent:
            raise FullTimeError(f"VSV source time is outside [0, {self.temporal_extent})")
        if not 0 <= int(sink_time) < self.temporal_extent:
            raise FullTimeError(f"VSV sink time is outside [0, {self.temporal_extent})")
        source_time, sink_time = int(source_time), int(sink_time)
        rank, local_time = divmod(sink_time, self.local_time)
        key = (source_time, rank)
        record = self._records[key]
        path = Path(record["path"])
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        block = np.array(values[local_time], copy=True)
        if not np.all(np.isfinite(block)):
            raise FullTimeError(f"VSV block contains non-finite values: {path}")
        self._accesses.append(
            {
                "source_time": source_time,
                "sink_time": sink_time,
                "rank": rank,
                "local_time": local_time,
                "file": path.as_posix(),
                "file_sha256": record.get("sha256"),
            }
        )
        return block

    def verify_stable(self) -> None:
        current_manifest_sha256 = _sha256(self.manifest_path)
        if current_manifest_sha256 != self.manifest_sha256:
            raise FullTimeError("full-time VSV manifest changed while being consumed")
        for record in self._records.values():
            path = Path(record["path"])
            if not path.is_file() or path.stat().st_size != record["bytes"]:
                raise FullTimeError(f"full-time VSV slab changed while being consumed: {path}")
            if self.verify_hashes and _sha256(path) != record["sha256"]:
                raise FullTimeError(f"full-time VSV slab hash changed while being consumed: {path}")

    def provenance(self) -> dict[str, Any]:
        return {
            "layout": VSV_LAYOUT,
            "directory": self.directory.as_posix(),
            "manifest_path": self.manifest_path.as_posix(),
            "manifest_sha256": self.manifest_sha256,
            "configuration": self.configuration,
            "time_convention": self.manifest["time_convention"],
            "disk_axes": ["t_sink_local", "spin_sink", "spin_source", "sink_ne", "source_ne"],
            "shape": list(self.shape),
            "dtype": self.dtype.str,
            "source_times": list(self.source_times),
            "temporal_ranks": self.temporal_ranks,
            "local_time": self.local_time,
            "finiteness_validation": "every-loaded-block",
            "hash_mode": "sha256" if self.verify_hashes else "stat-only",
            "slabs": [
                {"source_time": source, "rank": rank, **record}
                for (source, rank), record in sorted(self._records.items())
            ],
            "accessed_blocks": self._accesses,
        }


def _current_terms(direction: int, wilson_r: float):
    current = ConservedVectorCurrent(wilson_r=wilson_r)
    return current.terms[2 * direction : 2 * direction + 2]


def _validate_source_manifest(path: Path) -> dict[str, Any]:
    source_manifest = pair._validate_source_manifest(path)
    missing = sorted(REQUIRED_SOURCE_FILES.difference(source_manifest["records"]))
    if missing:
        raise FullTimeError("source manifest lacks full-time dependencies: " + ", ".join(missing))
    return source_manifest


def _load_current(args: argparse.Namespace) -> dict[str, Any]:
    current_path = args.current_artifact.expanduser().resolve()
    if not current_path.is_dir():
        raise FullTimeError(f"Current artifact directory does not exist: {current_path}")
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
        expected_configuration=args.configuration,
        expected_gauge_sha256=expected_gauge,
        expected_eigenvector_sha256=expected_eigenvector,
        verify_sources=not args.no_verify_current_sources,
        mmap_mode=None,
    )
    contract = current["contract"]
    extent = int(contract["shapes"]["v2v"][1])
    if extent != args.temporal_extent:
        raise FullTimeError(f"Current temporal extent {extent} does not match VSV {args.temporal_extent}")
    if int(args.current_ne) > int(contract["ne"]["used"]):
        raise FullTimeError("current-ne exceeds persisted Current Ne")
    return current


def _validate_execution(
    args: argparse.Namespace,
    current: dict[str, Any],
    accessor: FullTimeVSVSlabAccessor,
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    if args.execution_record is None:
        raise FullTimeError("--execution-record is required")
    execution = single._validate_execution_record(
        args.execution_record,
        configuration=args.configuration,
        vsv_time_axis="absolute-global-source-and-sink",
    )
    record = execution["record"]
    axes = record["axis_declarations"]
    if axes.get("current_direction") != args.current_direction:
        raise FullTimeError("execution record current direction does not match")
    if axes.get("wilson_r") != args.wilson_r:
        raise FullTimeError("execution record Wilson r does not match")
    if axes.get("vsv_layout") != VSV_LAYOUT:
        raise FullTimeError("execution record VSV layout does not match full-time slab accessor")
    hashes = record["input_hashes"]
    if hashes.get("current_artifact_manifest_sha256") != current["verified_files"]["manifest"]["sha256"]:
        raise FullTimeError("execution record Current artifact hash does not match")
    if hashes.get("vsv_manifest_sha256") != accessor.manifest_sha256:
        raise FullTimeError("execution record VSV manifest hash does not match")
    if hashes.get("source_manifest_sha256") != source_manifest["sha256"]:
        raise FullTimeError("execution record source manifest hash does not match")
    if record["git"] != source_manifest["git"]:
        raise FullTimeError("execution record Git state does not match source manifest")
    return execution


class _PairBlockCache:
    """Cache the VSV blocks reused by one anchor pair's four term pairs."""

    def __init__(self, accessor: FullTimeVSVSlabAccessor):
        self.accessor = accessor
        self.blocks: dict[tuple[int, int], np.ndarray] = {}

    def get(self, source_time: int, sink_time: int) -> np.ndarray:
        key = (int(source_time), int(sink_time))
        if key not in self.blocks:
            self.blocks[key] = self.accessor.get(*key)
        return self.blocks[key]


def _matrix(
    args: argparse.Namespace,
    current: dict[str, Any],
    accessor: FullTimeVSVSlabAccessor,
    backend: Any,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    terms = _current_terms(args.current_direction, args.wilson_r)
    contract = current["contract"]
    result = np.empty((args.temporal_extent, args.temporal_extent), dtype=np.complex128)
    pair_records: list[dict[str, Any]] = []
    for first_time in range(args.temporal_extent):
        for second_time in range(args.temporal_extent):
            pair_accessor = _PairBlockCache(accessor)
            contraction = contract_directed_current_pair_v2v(
                terms,
                current["raw"],
                contract,
                terms,
                current["raw"],
                contract,
                pair_accessor,
                first_anchor_time=first_time,
                second_anchor_time=second_time,
                first_field_ne=args.current_ne,
                first_bar_ne=args.current_ne,
                second_field_ne=args.current_ne,
                second_bar_ne=args.current_ne,
                first_momentum=args.first_momentum_index,
                second_momentum=args.second_momentum_index,
                array_backend=backend,
            )
            value = contraction["value"]
            if hasattr(value, "get") and callable(value.get):
                value = value.get()
            value = np.asarray(value)
            if value.shape != () or not np.isfinite(value):
                raise FullTimeError(f"non-scalar or non-finite pair at ({first_time}, {second_time})")
            result[first_time, second_time] = value
            pair_records.append(
                {
                    "first_current_anchor": first_time,
                    "second_current_anchor": second_time,
                    "term_pairs": len(contraction["term_pairs"]),
                }
            )
    return result, pair_records


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.current_direction not in range(4):
        raise FullTimeError("current direction must be 0, 1, 2 or 3")
    if args.current_ne <= 0:
        raise FullTimeError("current-ne must be positive")
    if not np.isfinite(args.wilson_r):
        raise FullTimeError("wilson-r must be finite")
    if args.first_momentum_index < 0 or args.second_momentum_index < 0:
        raise FullTimeError("momentum indices must be nonnegative")
    source_manifest = _validate_source_manifest(args.source_manifest)
    accessor = FullTimeVSVSlabAccessor(args.vsv_directory, args.configuration, verify_hashes=args.hash_files)
    args.temporal_extent = accessor.temporal_extent
    current = _load_current(args)
    manifest = _validate_execution(args, current, accessor, source_manifest)
    if args.backend == "numpy":
        backend = np
    else:
        try:
            import cupy
        except ImportError as exc:
            raise FullTimeError("--backend cupy requires CuPy on the compute node") from exc
        backend = cupy
    result, pair_records = _matrix(args, current, accessor, backend)
    accessor.verify_stable()
    single._verify_current_stable(current)
    pair._verify_source_manifest_stable(source_manifest)
    if _sha256(manifest["path"]) != manifest["sha256"]:
        raise FullTimeError("execution record changed while being consumed")

    result_dir = args.result_dir.expanduser().resolve()
    if result_dir.exists():
        raise FullTimeError(f"refusing to reuse output directory: {result_dir}")
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = result_dir.parent / f".{result_dir.name}.partial-{uuid.uuid4().hex}"
    try:
        stage.mkdir()
        temporary_result = stage / "correlator.npy"
        with temporary_result.open("wb") as stream:
            np.save(stream, result, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        result_sha256 = _sha256(temporary_result)
        result_name = f"correlator-{result_sha256}.npy"
        result_path = stage / result_name
        temporary_result.rename(result_path)
        result_manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "version": 1,
            "status": "complete",
            "classification": "real-gauge-artifact-raw-not-physics-validation",
            "configuration": args.configuration,
            "current": {
                "direction": args.current_direction,
                "direction_name": DIRECTIONS[args.current_direction],
                "wilson_r": args.wilson_r,
                "ne": args.current_ne,
                "first_momentum_index": args.first_momentum_index,
                "second_momentum_index": args.second_momentum_index,
                "backend": args.backend,
                "artifact_manifest_path": current["manifest_path"].as_posix(),
                "artifact_manifest_sha256": current["verified_files"]["manifest"]["sha256"],
                "artifact_identity": current["manifest"]["artifact_identity"],
                "raw_cache_identity": current["contract"]["cache_identity"],
                "sources_verified": not args.no_verify_current_sources,
            },
            "times": {
                "axes": ["first_current_anchor", "second_current_anchor"],
                "temporal_extent": args.temporal_extent,
                "full_time": True,
            },
            "vsv": accessor.provenance(),
            "source": {
                "manifest_path": source_manifest["path"].as_posix(),
                "manifest_sha256": source_manifest["sha256"],
                "manifest_identity": source_manifest["identity"],
                "git": source_manifest["git"],
                "file_count": source_manifest["file_count"],
            },
            "consumer": {
                "schema": CURRENT_V2V_PAIR_CONTRACTION_SCHEMA,
                "term_pair_contraction": "bfji,ackl,afki,bcjl->",
                "operation": "ordered connected unflavored unsigned raw trace over four Wilson term pairs",
                "pair_count": len(pair_records),
                "term_pairs_per_anchor_pair": 4,
                "uses_preloaded_vsv_only": True,
                "implicit_physics_factors": [],
            },
            "result": {
                "filename": result_name,
                "sha256": result_sha256,
                "shape": list(result.shape),
                "dtype": result.dtype.str,
                "axes": ["first_current_anchor", "second_current_anchor"],
                "finite": bool(np.all(np.isfinite(result))),
            },
            "execution": {
                "record_path": manifest["path"].as_posix(),
                "record_sha256": manifest["sha256"],
                "record_identity": manifest["record"]["record_identity"],
                "cluster": "kunshan",
                "git": manifest["record"]["git"],
                "resources": manifest["record"]["resources"],
                "slurm_job_id": manifest["record"]["slurm_job_id"],
            },
        }
        result_manifest["manifest_identity"] = _manifest_identity(result_manifest)
        _atomic_json(stage / "manifest.json", result_manifest)
        result_summary = {
            "schema": "lattice.current.fulltime-vsv-v2v-pair-result/v1",
            "status": "complete",
            "classification": result_manifest["classification"],
            "configuration": args.configuration,
            "manifest_identity": result_manifest["manifest_identity"],
            "result": result_manifest["result"],
            "execution": result_manifest["execution"],
        }
        _atomic_json(stage / "result.json", result_summary)
        _atomic_json(
            stage / "DONE",
            {
                "status": "complete",
                "artifact_sha256": {
                    result_name: result_sha256,
                    "manifest.json": _sha256(stage / "manifest.json"),
                    "result.json": _sha256(stage / "result.json"),
                },
            },
        )
        if result_dir.exists():
            raise FullTimeError(f"result directory appeared concurrently: {result_dir}")
        os.rename(stage, result_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return result_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--current-artifact", type=Path, required=True)
    parser.add_argument("--vsv-directory", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--execution-record", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--current-ne", type=int, required=True)
    parser.add_argument("--wilson-r", type=float, default=1.0)
    parser.add_argument("--current-direction", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    parser.add_argument("--first-momentum-index", type=int, default=0)
    parser.add_argument("--second-momentum-index", type=int, default=0)
    parser.add_argument("--expected-gauge-sha256")
    parser.add_argument("--expected-eigenvector-sha256")
    parser.add_argument("--no-verify-current-sources", action="store_true")
    parser.add_argument(
        "--hash-files",
        action="store_true",
        help="hash every full-time VSV slab before and after contraction",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = run(args)
    except (FullTimeError, single.SmokeError, OSError, TypeError, ValueError, IndexError) as exc:
        print(f"full-time VSV pair error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "manifest_identity": result["manifest_identity"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
