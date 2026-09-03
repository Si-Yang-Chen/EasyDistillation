#!/usr/bin/env python3
"""Prepare a trusted execution record for a full-time VSV pair job.

This runs inside the submitted job after the real Slurm ID and runtime
resource binding are available. It only writes the execution record; it does
not start a contraction or alter any input dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import contract_existing_vsv as single  # noqa: E402

SCHEMA = "lattice.current.kunshan-execution-record/v1"
VSV_LAYOUT = "fulltime-source-time-rank-slab"


def _sha256(path: Path) -> str:
    return single._sha256(path)


def _record_identity(value: dict) -> str:
    semantic = {key: item for key, item in value.items() if key != "record_identity"}
    return hashlib.sha256(single._canonical_bytes(semantic).rstrip(b"\n")).hexdigest()


def _required_file(path: Path, name: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{name} does not exist as a regular file: {path}")
    return path


def build_record(args: argparse.Namespace) -> dict:
    if not args.configuration:
        raise ValueError("configuration must be non-empty")
    source_manifest = _required_file(args.source_manifest, "source manifest")
    current_manifest = _required_file(args.current_artifact / "manifest.json", "Current artifact manifest")
    vsv_manifest = _required_file(args.vsv_directory / "manifest.json", "VSV manifest")
    runtime = single._runtime_execution_binding()
    record = {
        "schema": SCHEMA,
        "version": 1,
        "cluster": "kunshan",
        "configuration": args.configuration,
        "git": runtime["git"],
        "resources": runtime["resources"],
        "slurm_job_id": runtime["slurm_job_id"],
        "input_hashes": {
            "current_artifact_manifest_sha256": _sha256(current_manifest),
            "vsv_manifest_sha256": _sha256(vsv_manifest),
            "source_manifest_sha256": _sha256(source_manifest),
        },
        "axis_declarations": {
            "current_direction": args.current_direction,
            "wilson_r": args.wilson_r,
            "vsv_layout": VSV_LAYOUT,
            "vsv_time_axis": "absolute-global-source-and-sink",
            "output_axes": ["first_current_anchor", "second_current_anchor"],
        },
    }
    record["record_identity"] = _record_identity(record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--current-artifact", type=Path, required=True)
    parser.add_argument("--vsv-directory", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--current-direction", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--wilson-r", type=float, default=1.0)
    args = parser.parse_args()
    try:
        output = args.output.expanduser().resolve()
        if output.exists():
            raise ValueError(f"refusing to overwrite execution record: {output}")
        record = build_record(args)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.partial")
        temporary.write_bytes(single._canonical_bytes(record))
        os.replace(temporary, output)
    except (OSError, TypeError, ValueError) as exc:
        print(f"full-time execution record error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "complete",
                "record": output.as_posix(),
                "record_identity": record["record_identity"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
