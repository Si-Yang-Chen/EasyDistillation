#!/usr/bin/env python3
"""Run the full-time Wilson J4xJ4 matrix producer for one configuration.

The wrapper is intentionally one-configuration-per-process. An ensemble Slurm
array can invoke it once per configuration, giving each attempt an isolated
result directory and execution record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--current-artifact", type=Path, required=True)
    parser.add_argument("--vsv-directory", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--current-ne", type=int, required=True)
    parser.add_argument("--wilson-r", type=float, default=1.0)
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    parser.add_argument("--current-direction", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--first-momentum-index", type=int, default=0)
    parser.add_argument("--second-momentum-index", type=int, default=0)
    parser.add_argument("--expected-gauge-sha256")
    parser.add_argument("--expected-eigenvector-sha256")
    parser.add_argument("--no-verify-current-sources", action="store_true")
    parser.add_argument("--hash-files", action="store_true")
    args = parser.parse_args()

    result_root = args.result_root.expanduser().resolve()
    record = result_root / args.configuration / "execution-record.json"
    result_dir = result_root / args.configuration / "result"
    record.parent.mkdir(parents=True, exist_ok=True)
    prepare = [
        sys.executable,
        str(HERE / "prepare_fulltime_execution_record.py"),
        "--configuration",
        args.configuration,
        "--current-artifact",
        str(args.current_artifact.expanduser().resolve()),
        "--vsv-directory",
        str(args.vsv_directory.expanduser().resolve()),
        "--source-manifest",
        str(args.source_manifest.expanduser().resolve()),
        "--output",
        str(record),
        "--wilson-r",
        str(args.wilson_r),
        "--current-direction",
        str(args.current_direction),
    ]
    subprocess.run(prepare, check=True)

    command = [
        sys.executable,
        str(HERE / "contract_fulltime_vsv_pair.py"),
        "--configuration",
        args.configuration,
        "--current-artifact",
        str(args.current_artifact.expanduser().resolve()),
        "--vsv-directory",
        str(args.vsv_directory.expanduser().resolve()),
        "--source-manifest",
        str(args.source_manifest.expanduser().resolve()),
        "--execution-record",
        str(record),
        "--backend",
        args.backend,
        "--result-dir",
        str(result_dir),
        "--current-ne",
        str(args.current_ne),
        "--wilson-r",
        str(args.wilson_r),
        "--current-direction",
        str(args.current_direction),
        "--first-momentum-index",
        str(args.first_momentum_index),
        "--second-momentum-index",
        str(args.second_momentum_index),
    ]
    for option, value in (
        ("--expected-gauge-sha256", args.expected_gauge_sha256),
        ("--expected-eigenvector-sha256", args.expected_eigenvector_sha256),
    ):
        if value is not None:
            command.extend((option, value))
    if args.no_verify_current_sources:
        command.append("--no-verify-current-sources")
    if args.hash_files:
        command.append("--hash-files")
    subprocess.run(command, check=True)
    summary = {
        "schema": "lattice.current.fulltime-vsv-v2v-pair-ensemble-item/v1",
        "status": "complete",
        "configuration": args.configuration,
        "result_directory": result_dir.as_posix(),
        "execution_record": record.as_posix(),
    }
    summary_path = result_root / args.configuration / "array-item.json"
    temporary = summary_path.with_name(f".{summary_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, summary_path)
    finally:
        temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
