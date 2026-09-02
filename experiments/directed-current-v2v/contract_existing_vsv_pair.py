#!/usr/bin/env python3
"""Contract two directed Current insertions with an audited existing VSV.

The output is an ordered, connected, unflavored and unsigned V2V trace.  This
is an artifact smoke test, not a physical current-current correlator result.
"""

from __future__ import annotations

import argparse
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
from lattice.current_elemental import (  # noqa: E402
    CURRENT_V2V_PAIR_CONTRACTION_SCHEMA,
    contract_directed_current_pair_v2v,
    load_directed_current_v2v,
)
from lattice.insertion.current import (  # noqa: E402
    ConservedVectorCurrent,
    resolve_current_term_endpoints,
)

SCHEMA = "lattice.current.existing-vsv-v2v-pair-contraction/v1"
SOURCE_MANIFEST_SCHEMA = "lattice.current.handover-manifest/v1"
CLASSIFICATION = "artifact-smoke-unflavored-unsigned-not-physics-validation"
TERM_PAIR_CONTRACTION = "bfji,ackl,afki,bcjl->"
REQUIRED_SOURCE_FILES = {
    "experiments/directed-current-v2v/contract_existing_vsv.py",
    "experiments/directed-current-v2v/contract_existing_vsv_pair.py",
    "lattice/__init__.py",
    "lattice/current_elemental.py",
    "lattice/generator/elemental.py",
    "lattice/insertion/__init__.py",
    "lattice/insertion/current.py",
    "lattice/insertion/gauge_link.py",
}


def _validate_source_manifest(path: Path) -> dict[str, Any]:
    manifest_path = single._absolute_file(path, "source manifest")
    manifest, manifest_sha256 = single._load_json_snapshot(
        manifest_path,
        "source manifest",
    )
    required = {
        "schema",
        "version",
        "generated_utc",
        "project_root_at_generation",
        "git",
        "files",
        "release",
        "verification",
        "manifest_identity",
    }
    if set(manifest) != required:
        single._fail("source manifest has missing or unknown fields")
    if manifest["schema"] != SOURCE_MANIFEST_SCHEMA or manifest["version"] != 1:
        single._fail("source manifest schema/version is unsupported")
    if manifest["manifest_identity"] != single._semantic_identity(manifest):
        single._fail("source manifest identity is stale or tampered")
    git = manifest["git"]
    if not isinstance(git, dict) or set(git) != {
        "head",
        "branch",
        "dirty",
        "origin_url",
        "accepted_external_refs",
    }:
        single._fail("source manifest Git metadata is invalid")
    if (
        not isinstance(git["head"], str)
        or len(git["head"]) != 40
        or any(character not in "0123456789abcdef" for character in git["head"].lower())
        or not isinstance(git["dirty"], bool)
    ):
        single._fail("source manifest Git identity is invalid")
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        single._fail("source manifest files must be a non-empty list")
    records = {}
    for index, record in enumerate(files):
        if not isinstance(record, dict) or set(record) != {
            "path",
            "sha256",
            "bytes",
            "git_state",
        }:
            single._fail(f"source manifest file record {index} is invalid")
        relative = record["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in records
        ):
            single._fail("source manifest file paths must be unique and relative")
        source = (ROOT / relative).resolve()
        try:
            source.relative_to(ROOT)
        except ValueError:
            single._fail(f"source manifest file escapes source root: {relative}")
        digest = single._require_hash(
            record["sha256"],
            f"source manifest file hash {relative}",
        )
        if (
            not isinstance(record["bytes"], int)
            or isinstance(record["bytes"], bool)
            or record["bytes"] < 0
            or not source.is_file()
            or source.stat().st_size != record["bytes"]
            or single._sha256(source) != digest
        ):
            single._fail(f"source manifest file bytes do not match: {relative}")
        records[relative] = {"path": source, "sha256": digest}
    missing = REQUIRED_SOURCE_FILES.difference(records)
    if missing:
        single._fail("source manifest lacks required pair dependencies: " + ", ".join(sorted(missing)))
    return {
        "path": manifest_path,
        "sha256": manifest_sha256,
        "identity": manifest["manifest_identity"],
        "git": {"commit": git["head"].lower(), "dirty": git["dirty"]},
        "file_count": len(records),
        "records": records,
    }


def _verify_source_manifest_stable(source_manifest: dict[str, Any]) -> None:
    if single._sha256(source_manifest["path"]) != source_manifest["sha256"]:
        single._fail("source manifest changed while it was being consumed")
    for relative, record in source_manifest["records"].items():
        if single._sha256(record["path"]) != record["sha256"]:
            single._fail(f"source manifest file changed while it was being consumed: {relative}")


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    current_path = Path(args.current_artifact).expanduser()
    if not current_path.is_absolute():
        single._fail("Current artifact path must be absolute")
    current_path = current_path.resolve()
    if not current_path.exists():
        single._fail(f"Current artifact does not exist: {current_path}")
    result_dir = single._absolute_new_directory(args.result_dir)
    if not isinstance(args.configuration, str) or not args.configuration:
        single._fail("configuration must be a non-empty string")
    source_manifest = _validate_source_manifest(args.source_manifest)

    expected_gauge = (
        None
        if args.expected_gauge_sha256 is None
        else single._require_hash(args.expected_gauge_sha256, "expected gauge hash")
    )
    expected_eigenvector = (
        None
        if args.expected_eigenvector_sha256 is None
        else single._require_hash(
            args.expected_eigenvector_sha256,
            "expected eigenvector hash",
        )
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
    temporal_extent = contract["shapes"]["v2v"][1]
    momentum_count = contract["shapes"]["v2v"][2]
    for value, name in (
        (args.first_current_time, "first_current_time"),
        (args.second_current_time, "second_current_time"),
    ):
        if not 0 <= value < temporal_extent:
            single._fail(f"{name} must be inside the Current temporal extent")
    for value, name in (
        (args.first_momentum_index, "first_momentum_index"),
        (args.second_momentum_index, "second_momentum_index"),
    ):
        if not 0 <= value < momentum_count:
            single._fail(f"{name} is outside the Current momentum axis")
    current_ne = single._positive_int(args.current_ne, "current_ne")
    if current_ne > contract["ne"]["used"]:
        single._fail("requested Current Ne exceeds the persisted raw Current Ne")
    wilson_r = single._finite_float(args.wilson_r, "wilson_r")

    if args.vsv is not None:
        if args.vsv_sha256 is None:
            single._fail("--vsv-sha256 is required with --vsv")
        if args.vsv_timeslice_manifest is not None:
            single._fail("--vsv-timeslice-manifest cannot be combined with --vsv")
        vsv_path = single._absolute_file(args.vsv, "VSV")
        vsv_sha256 = single._require_hash(args.vsv_sha256, "VSV hash")
        accessor, _actual_vsv_sha256 = single._validate_vsv_header(
            vsv_path,
            expected_sha256=vsv_sha256,
            time_axis=args.vsv_time_axis,
            temporal_extent=temporal_extent,
        )
    else:
        if args.vsv_sha256 is not None:
            single._fail("--vsv-sha256 can only be used with --vsv")
        if args.vsv_timeslice_manifest is None:
            single._fail("--vsv-timeslice-manifest is required with --vsv-timeslice-pattern")
        accessor = single.ExistingVSVTimeslices(
            args.vsv_timeslice_pattern,
            args.vsv_timeslice_manifest,
            configuration=args.configuration,
            time_axis=args.vsv_time_axis,
            temporal_extent=temporal_extent,
        )
    if current_ne > accessor.shape[-1] or current_ne > accessor.shape[-2]:
        single._fail("current_ne exceeds a VSV eigenvector extent")

    execution_record = None
    if not args.dry_run:
        if args.execution_record is None:
            single._fail("--execution-record is required for a non-dry smoke")
        execution_record = single._validate_execution_record(
            args.execution_record,
            configuration=args.configuration,
            vsv_time_axis=args.vsv_time_axis,
        )
        record = execution_record["record"]
        if record["axis_declarations"].get("current_direction") != args.current_direction:
            single._fail("execution record current direction does not match")
        if record["axis_declarations"].get("vsv_layout") != accessor.layout:
            single._fail("execution record VSV layout does not match")
        input_hashes = record["input_hashes"]
        if input_hashes.get("current_artifact_manifest_sha256") != current["verified_files"]["manifest"]["sha256"]:
            single._fail("execution record Current artifact hash does not match")
        if accessor.layout == "full-rank6":
            if input_hashes.get("vsv_sha256") != accessor.sha256:
                single._fail("execution record VSV hash does not match")
        elif input_hashes.get("vsv_timeslice_manifest_sha256") != accessor.hash_manifest_sha256:
            single._fail("execution record VSV timeslice manifest hash does not match")
        if input_hashes.get("source_manifest_sha256") != source_manifest["sha256"]:
            single._fail("execution record source manifest hash does not match")
        if (
            record["git"]["commit"] != source_manifest["git"]["commit"]
            or record["git"]["dirty"] != source_manifest["git"]["dirty"]
        ):
            single._fail("execution record Git state does not match source manifest")

    return {
        "accessor": accessor,
        "configuration": args.configuration,
        "current": current,
        "current_ne": current_ne,
        "execution_record": execution_record,
        "result_dir": result_dir,
        "source_manifest": source_manifest,
        "temporal_extent": temporal_extent,
        "wilson_r": wilson_r,
    }


def _terms(prepared: dict[str, Any], direction: int):
    current = ConservedVectorCurrent(wilson_r=prepared["wilson_r"])
    return current.terms[2 * direction : 2 * direction + 2]


def _validate_pair_blocks(
    prepared: dict[str, Any],
    first_terms,
    second_terms,
    args: argparse.Namespace,
) -> None:
    extent = prepared["temporal_extent"]
    boundary = prepared["current"]["contract"]["boundary"]
    accessor = prepared["accessor"]
    current_ne = prepared["current_ne"]
    for first_term in first_terms:
        first_endpoints = resolve_current_term_endpoints(
            first_term,
            anchor_time=args.first_current_time,
            temporal_extent=extent,
            boundary=boundary,
        )
        for second_term in second_terms:
            second_endpoints = resolve_current_term_endpoints(
                second_term,
                anchor_time=args.second_current_time,
                temporal_extent=extent,
                boundary=boundary,
            )
            for source_time, sink_time in (
                (
                    first_endpoints["field_time"],
                    second_endpoints["bar_time"],
                ),
                (
                    second_endpoints["field_time"],
                    first_endpoints["bar_time"],
                ),
            ):
                block = np.asarray(accessor.get(source_time, sink_time))
                if block.ndim != 4 or block.shape[:2] != (4, 4):
                    single._fail("pair VSV block has invalid spin axes")
                if block.shape[2] < current_ne or block.shape[3] < current_ne:
                    single._fail("pair VSV block Ne extent is smaller than current_ne")
                if not np.issubdtype(block.dtype, np.complexfloating):
                    single._fail("pair VSV block must have a complex dtype")
                if not np.all(np.isfinite(block)):
                    single._fail("pair VSV block contains non-finite values")


def _current_metadata(prepared: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    current = prepared["current"]
    manifest = current["manifest"]
    sources = manifest["sources"]
    return {
        "kind": "conserved-vector-pair-same-artifact",
        "direction": args.current_direction,
        "direction_name": ("x", "y", "z", "t")[args.current_direction],
        "wilson_r": prepared["wilson_r"],
        "artifact_manifest_path": current["manifest_path"].as_posix(),
        "artifact_manifest_sha256": current["verified_files"]["manifest"]["sha256"],
        "artifact_identity": manifest["artifact_identity"],
        "raw_cache_identity": current["contract"]["cache_identity"],
        "data_sha256": manifest["data"]["sha256"],
        "gauge_sha256": sources["gauge"]["sha256"],
        "eigenvector_sha256": sources["eigenvector"]["sha256"],
        "sources_verified": not args.no_verify_current_sources,
    }


def _publish(
    prepared: dict[str, Any],
    contraction: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    result_dir = prepared["result_dir"]
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = result_dir.parent / f".{result_dir.name}.partial-{uuid.uuid4().hex}"
    try:
        if result_dir.exists():
            single._fail(f"refusing to reuse existing output directory: {result_dir}")
        stage.mkdir()
        temporary_result = stage / "contraction.npy"
        with temporary_result.open("wb") as output:
            np.save(output, np.asarray(contraction["value"]), allow_pickle=False)
            output.flush()
            os.fsync(output.fileno())
        result_sha256 = single._sha256(temporary_result)
        result_name = f"contraction-{result_sha256}.npy"
        result_path = stage / result_name
        temporary_result.rename(result_path)

        prepared["accessor"].verify_stable()
        single._verify_current_stable(prepared["current"])
        _verify_source_manifest_stable(prepared["source_manifest"])
        execution = prepared["execution_record"]
        if single._sha256(execution["path"]) != execution["sha256"]:
            single._fail("execution record changed while it was being consumed")

        source_manifest = prepared["source_manifest"]
        current_manifest = prepared["current"]["manifest"]
        first_momentum = current_manifest["momenta"][args.first_momentum_index]
        second_momentum = current_manifest["momenta"][args.second_momentum_index]
        result_array = np.asarray(contraction["value"])
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "version": 1,
            "status": "contracted",
            "classification": CLASSIFICATION,
            "configuration": prepared["configuration"],
            "times": {
                "first_current_anchor": args.first_current_time,
                "second_current_anchor": args.second_current_time,
            },
            "boundary": prepared["current"]["contract"]["boundary"],
            "momenta": {
                "first": {
                    "index": args.first_momentum_index,
                    "vector": first_momentum,
                },
                "second": {
                    "index": args.second_momentum_index,
                    "vector": second_momentum,
                },
            },
            "ne": {"current": prepared["current_ne"]},
            "source": {
                "manifest_path": source_manifest["path"].as_posix(),
                "manifest_sha256": source_manifest["sha256"],
                "manifest_identity": source_manifest["identity"],
                "git": source_manifest["git"],
                "file_count": source_manifest["file_count"],
                "required_pair_files_verified": sorted(REQUIRED_SOURCE_FILES),
            },
            "current_pair": _current_metadata(prepared, args),
            "vsv": prepared["accessor"].provenance(),
            "consumer": {
                "schema": CURRENT_V2V_PAIR_CONTRACTION_SCHEMA,
                "term_pair_contraction": TERM_PAIR_CONTRACTION,
                "output_axes": [],
                "operation": contraction["operation"],
                "uses_preloaded_vsv_only": True,
                "term_pairs": list(contraction["term_pairs"]),
            },
            "result": {
                "filename": result_name,
                "sha256": result_sha256,
                "shape": list(result_array.shape),
                "dtype": result_array.dtype.str,
                "axes": [],
                "finite": bool(np.all(np.isfinite(result_array))),
            },
            "producer": {
                "script_path": Path(__file__).resolve().as_posix(),
                "script_sha256": single._sha256(Path(__file__).resolve()),
            },
            "execution": {
                "record_path": execution["path"].as_posix(),
                "record_sha256": execution["sha256"],
                "record_identity": execution["record"]["record_identity"],
                "cluster": "kunshan",
                "git": execution["record"]["git"],
                "resources": execution["record"]["resources"],
                "slurm_job_id": execution["record"]["slurm_job_id"],
            },
        }
        if result_array.shape != ():
            single._fail("pair contraction result must be scalar")
        if not manifest["result"]["finite"]:
            single._fail("pair contraction result contains non-finite values")
        manifest["manifest_identity"] = single._semantic_identity(manifest)
        single._atomic_write(stage / "manifest.json", single._canonical_bytes(manifest))
        done = {
            "status": "complete",
            "artifact_sha256": {
                result_name: result_sha256,
                "manifest.json": single._sha256(stage / "manifest.json"),
            },
        }
        single._atomic_write(stage / "DONE", single._canonical_bytes(done))
        if result_dir.exists():
            single._fail(f"refusing to reuse existing output directory: {result_dir}")
        os.rename(stage, result_dir)
    except FileExistsError as exc:
        shutil.rmtree(stage, ignore_errors=True)
        raise single.SmokeError(f"another writer published this result directory: {result_dir}") from exc
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepared = _prepare(args)
    first_terms = _terms(prepared, args.current_direction)
    second_terms = _terms(prepared, args.current_direction)
    if args.dry_run:
        _validate_pair_blocks(prepared, first_terms, second_terms, args)
        prepared["accessor"].verify_stable()
        single._verify_current_stable(prepared["current"])
        _verify_source_manifest_stable(prepared["source_manifest"])
        return {
            "status": "dry-run-valid",
            "classification": CLASSIFICATION,
            "configuration": prepared["configuration"],
            "result_dir": prepared["result_dir"].as_posix(),
            "vsv_shape": list(prepared["accessor"].shape),
            "vsv_layout": prepared["accessor"].layout,
            "vsv_time_axis": args.vsv_time_axis,
            "sources_verified": not args.no_verify_current_sources,
            "source_manifest_identity": prepared["source_manifest"]["identity"],
            "source_manifest_files": prepared["source_manifest"]["file_count"],
            "validated_term_pairs": len(first_terms) * len(second_terms),
            "validated_vsv_accesses": len(prepared["accessor"].accesses),
        }

    contraction = contract_directed_current_pair_v2v(
        first_terms,
        prepared["current"]["raw"],
        prepared["current"]["contract"],
        second_terms,
        prepared["current"]["raw"],
        prepared["current"]["contract"],
        prepared["accessor"],
        first_anchor_time=args.first_current_time,
        second_anchor_time=args.second_current_time,
        first_field_ne=prepared["current_ne"],
        first_bar_ne=prepared["current_ne"],
        second_field_ne=prepared["current_ne"],
        second_bar_ne=prepared["current_ne"],
        first_momentum=args.first_momentum_index,
        second_momentum=args.second_momentum_index,
    )
    prepared["accessor"].verify_stable()
    single._verify_current_stable(prepared["current"])
    _verify_source_manifest_stable(prepared["source_manifest"])
    execution = prepared["execution_record"]
    if single._sha256(execution["path"]) != execution["sha256"]:
        single._fail("execution record changed while it was being consumed")
    return _publish(prepared, contraction, args)


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
    )
    parser.add_argument("--current-artifact", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--first-current-time", type=int, required=True)
    parser.add_argument("--second-current-time", type=int, required=True)
    parser.add_argument("--current-ne", type=int, required=True)
    parser.add_argument("--first-momentum-index", type=int, default=0)
    parser.add_argument("--second-momentum-index", type=int, default=0)
    parser.add_argument(
        "--current-direction",
        type=int,
        choices=(0, 1, 2, 3),
        required=True,
    )
    parser.add_argument("--wilson-r", type=float, default=1.0)
    parser.add_argument("--expected-gauge-sha256")
    parser.add_argument("--expected-eigenvector-sha256")
    parser.add_argument("--execution-record", type=Path)
    parser.add_argument("--no-verify-current-sources", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = run(args)
    except (single.SmokeError, ValueError, TypeError, IndexError, OSError) as exc:
        print(f"existing-VSV pair smoke error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
