#!/usr/bin/env python3
"""Collect per-configuration full-time Wilson J4xJ4 matrices atomically."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid
from typing import Any

import numpy as np

SCHEMA = "lattice.current.fulltime-vsv-v2v-pair-ensemble/v1"
CONFIGURATIONS = ("10000", "13000", "14000", "15000", "16000", "17000", "18000", "19000")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(value: dict[str, Any]) -> str:
    semantic = {key: item for key, item in value.items() if key != "manifest_identity"}
    return hashlib.sha256(_canonical(semantic)).hexdigest()


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {name}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} is not a JSON object: {path}")
    return value


def _load_item(result_root: Path, configuration: str) -> tuple[np.ndarray, dict[str, Any]]:
    item_root = result_root / configuration / "result"
    manifest_path = item_root / "manifest.json"
    result_summary_path = item_root / "result.json"
    done_path = item_root / "DONE"
    manifest = _load_json(manifest_path, "item manifest")
    summary = _load_json(result_summary_path, "item result summary")
    done = _load_json(done_path, "item DONE")
    if manifest.get("schema") != "lattice.current.fulltime-vsv-v2v-pair-matrix/v1":
        raise ValueError(f"unsupported item manifest schema for {configuration}")
    if manifest.get("manifest_identity") != _identity(manifest):
        raise ValueError(f"item manifest identity is stale for {configuration}")
    if summary.get("manifest_identity") != manifest["manifest_identity"]:
        raise ValueError(f"item result summary does not match manifest for {configuration}")
    result_info = manifest.get("result")
    if not isinstance(result_info, dict) or set(result_info) != {
        "filename",
        "sha256",
        "shape",
        "dtype",
        "axes",
        "finite",
    }:
        raise ValueError(f"item result metadata is invalid for {configuration}")
    result_path = item_root / result_info["filename"]
    if not result_path.is_file() or _sha256(result_path) != result_info["sha256"]:
        raise ValueError(f"item result bytes do not match manifest for {configuration}")
    done_hashes = done.get("artifact_sha256")
    if not isinstance(done_hashes, dict) or done_hashes.get(result_info["filename"]) != result_info["sha256"]:
        raise ValueError(f"item DONE does not bind result for {configuration}")
    if done_hashes.get("manifest.json") != _sha256(manifest_path) or done_hashes.get("result.json") != _sha256(
        result_summary_path
    ):
        raise ValueError(f"item DONE does not bind metadata for {configuration}")
    values = np.load(result_path, allow_pickle=False)
    if values.shape != (72, 72) or values.dtype != np.dtype("<c16") or not np.all(np.isfinite(values)):
        raise ValueError(f"item result has invalid shape/dtype/finiteness for {configuration}")
    return np.asarray(values), {
        "configuration": configuration,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": _sha256(manifest_path),
        "manifest_identity": manifest["manifest_identity"],
        "result_path": result_path.as_posix(),
        "result_sha256": result_info["sha256"],
        "source_manifest_sha256": manifest.get("source", {}).get("manifest_sha256"),
        "vsv_manifest_sha256": manifest.get("vsv", {}).get("manifest_sha256"),
    }


def collect(*, result_root: Path, output: Path, configurations: list[str]) -> dict[str, Any]:
    if not configurations:
        raise ValueError("at least one configuration is required")
    if output.exists():
        raise ValueError(f"refusing to reuse output directory: {output}")
    arrays = []
    items = []
    for configuration in configurations:
        values, provenance = _load_item(result_root, configuration)
        arrays.append(values)
        items.append(provenance)
    source_hashes = {item["source_manifest_sha256"] for item in items}
    vsv_hashes = {item["vsv_manifest_sha256"] for item in items}
    if None in source_hashes or None in vsv_hashes:
        raise ValueError("ensemble items must declare source and VSV manifest hashes")
    if len(source_hashes) != 1 or len(vsv_hashes) != 1:
        raise ValueError("ensemble items do not share source/VSV manifest lineage")
    stacked = np.stack(arrays, axis=0).astype(np.complex128, copy=False)
    stage = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"
    try:
        stage.mkdir(parents=True)
        temporary = stage / "correlator.npy"
        with temporary.open("wb") as stream:
            np.save(stream, stacked, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        result_sha256 = _sha256(temporary)
        result_name = f"correlator-{result_sha256}.npy"
        temporary.rename(stage / result_name)
        manifest = {
            "schema": SCHEMA,
            "version": 1,
            "status": "complete",
            "classification": "real-gauge-artifact-raw-not-physics-validation",
            "configurations": configurations,
            "source_manifest_sha256": next(iter(source_hashes)),
            "vsv_manifest_sha256": next(iter(vsv_hashes)),
            "items": items,
            "result": {
                "filename": result_name,
                "sha256": result_sha256,
                "shape": list(stacked.shape),
                "dtype": stacked.dtype.str,
                "axes": ["configuration", "first_current_anchor", "second_current_anchor"],
                "finite": bool(np.all(np.isfinite(stacked))),
            },
            "consumer": {
                "operation": "stack per-configuration ordered connected unflavored unsigned raw matrices",
                "implicit_physics_factors": [],
            },
        }
        manifest["manifest_identity"] = _identity(manifest)
        manifest_path = stage / "manifest.json"
        with manifest_path.open("wb") as stream:
            stream.write(_canonical(manifest) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        result_summary = {
            "schema": "lattice.current.fulltime-vsv-v2v-pair-ensemble-result/v1",
            "status": "complete",
            "classification": manifest["classification"],
            "configurations": configurations,
            "manifest_identity": manifest["manifest_identity"],
            "result": manifest["result"],
        }
        summary_path = stage / "result.json"
        with summary_path.open("wb") as stream:
            stream.write(_canonical(result_summary) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        done_path = stage / "DONE"
        with done_path.open("wb") as stream:
            stream.write(
                _canonical(
                    {
                        "status": "complete",
                        "artifact_sha256": {
                            result_name: result_sha256,
                            "manifest.json": _sha256(manifest_path),
                            "result.json": _sha256(summary_path),
                        },
                    }
                )
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        if output.exists():
            raise ValueError(f"output directory appeared concurrently: {output}")
        os.rename(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configurations", default=",".join(CONFIGURATIONS))
    args = parser.parse_args()
    try:
        configurations = [item.strip() for item in args.configurations.split(",") if item.strip()]
        manifest = collect(
            result_root=args.result_root.expanduser().resolve(),
            output=args.output.expanduser().resolve(),
            configurations=configurations,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"full-time ensemble collection error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps({"status": manifest["status"], "manifest_identity": manifest["manifest_identity"]}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
