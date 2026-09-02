#!/usr/bin/env python3
"""Build a strict SHA-256 manifest for a complete rank-5 VSV timeslice family."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

SCHEMA = "lattice.current.vsv-timeslice-manifest/v1"
DISK_AXES = [
    "second_time",
    "sink_spin",
    "source_spin",
    "sink_ne",
    "source_ne",
]


class ManifestError(ValueError):
    pass


def _fail(message: str) -> None:
    raise ManifestError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _identity(value: dict[str, Any]) -> str:
    semantic = {key: item for key, item in value.items() if key != "manifest_identity"}
    return hashlib.sha256(_canonical(semantic)).hexdigest()


def build_manifest(
    pattern: str,
    *,
    configuration: str,
    temporal_extent: int,
    time_axis_semantics: str,
) -> dict[str, Any]:
    if not configuration:
        _fail("configuration must be non-empty")
    if temporal_extent <= 0:
        _fail("temporal_extent must be positive")
    if time_axis_semantics not in {"source-relative", "source-sink"}:
        _fail("time_axis_semantics must be source-relative or source-sink")
    if "{source_time:03d}" not in pattern:
        _fail("pattern must contain {source_time:03d}")
    if not Path(pattern.format(source_time=0)).expanduser().is_absolute():
        _fail("pattern must be absolute")

    files = []
    expected_shape = None
    expected_dtype = None
    for source_time in range(temporal_extent):
        path = Path(pattern.format(source_time=source_time)).expanduser().resolve()
        if not path.is_file():
            _fail(f"source-time file is missing: {path}")
        try:
            values = np.load(path, allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise ManifestError(f"cannot read source-time file {path}: {exc}") from exc
        shape = tuple(int(size) for size in values.shape)
        if values.ndim != 5 or shape[:3] != (temporal_extent, 4, 4):
            _fail("each source-time file must have shape (Lt, 4, 4, sink_ne, source_ne)")
        if shape[3] <= 0 or shape[4] <= 0:
            _fail("VSV timeslice eigenvector extents must be positive")
        if not np.issubdtype(values.dtype, np.complexfloating):
            _fail("VSV timeslice files must have a complex dtype")
        if expected_shape is None:
            expected_shape = shape
            expected_dtype = values.dtype
        elif shape != expected_shape or values.dtype != expected_dtype:
            _fail("all source-time files must share shape and dtype")
        files.append(
            {
                "source_time": source_time,
                "path": path.as_posix(),
                "sha256": _sha256(path),
                "shape": list(shape),
                "dtype": values.dtype.str,
            }
        )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "configuration": configuration,
        "layout": "source-timeslices-rank5",
        "time_axis_semantics": time_axis_semantics,
        "temporal_extent": temporal_extent,
        "disk_axes": DISK_AXES,
        "files": files,
    }
    result["manifest_identity"] = _identity(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--temporal-extent", type=int, required=True)
    parser.add_argument(
        "--time-axis",
        choices=("source-relative", "source-sink"),
        required=True,
    )
    parser.add_argument("--output", type=Path, help="new output JSON; stdout if omitted")
    args = parser.parse_args()
    try:
        manifest = build_manifest(
            args.pattern,
            configuration=args.configuration,
            temporal_extent=args.temporal_extent,
            time_axis_semantics=args.time_axis,
        )
        encoded = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        if args.output is None:
            print(encoded, end="")
        else:
            output = args.output.expanduser()
            if not output.is_absolute():
                _fail("output path must be absolute")
            if output.exists():
                _fail(f"refusing to overwrite output: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(encoded, encoding="utf-8")
    except (ManifestError, OSError, ValueError) as exc:
        print(f"VSV timeslice manifest error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
