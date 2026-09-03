"""CPU tests for full-time ensemble result collection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from collect_fulltime_vsv_pair_ensemble import collect


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _identity(value: dict) -> str:
    semantic = {key: item for key, item in value.items() if key != "manifest_identity"}
    return hashlib.sha256(_canonical(semantic)).hexdigest()


def _write_item(root: Path, configuration: str, *, source_hash: str = "s" * 64) -> None:
    item = root / configuration / "result"
    item.mkdir(parents=True)
    values = np.full((72, 72), int(configuration[-1]), dtype=np.complex128)
    result_path = item / "correlator-item.npy"
    np.save(result_path, values, allow_pickle=False)
    result_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
    result_info = {
        "filename": result_path.name,
        "sha256": result_sha,
        "shape": [72, 72],
        "dtype": "<c16",
        "axes": ["first_current_anchor", "second_current_anchor"],
        "finite": True,
    }
    manifest = {
        "schema": "lattice.current.fulltime-vsv-v2v-pair-matrix/v1",
        "version": 1,
        "status": "complete",
        "classification": "real-gauge-artifact-raw-not-physics-validation",
        "configuration": configuration,
        "source": {"manifest_sha256": source_hash},
        "vsv": {"manifest_sha256": "v" * 64},
        "result": result_info,
    }
    manifest["manifest_identity"] = _identity(manifest)
    manifest_path = item / "manifest.json"
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    summary = {
        "manifest_identity": manifest["manifest_identity"],
        "result": result_info,
    }
    summary_path = item / "result.json"
    summary_path.write_bytes(_canonical(summary) + b"\n")
    done = {
        "status": "complete",
        "artifact_sha256": {
            result_path.name: result_sha,
            "manifest.json": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "result.json": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        },
    }
    (item / "DONE").write_bytes(_canonical(done) + b"\n")


def test_collects_and_binds_all_item_results(tmp_path: Path):
    _write_item(tmp_path, "10000")
    _write_item(tmp_path, "13000")
    output = tmp_path / "ensemble"

    manifest = collect(
        result_root=tmp_path,
        output=output,
        configurations=["10000", "13000"],
    )

    values = np.load(output / manifest["result"]["filename"], allow_pickle=False)
    assert values.shape == (2, 72, 72)
    assert values[0, 0, 0] == 0
    assert values[1, 0, 0] == 0
    assert manifest["result"]["axes"] == [
        "configuration",
        "first_current_anchor",
        "second_current_anchor",
    ]
    assert (output / "result.json").is_file()
    assert (output / "DONE").is_file()


def test_collect_rejects_mixed_source_lineage(tmp_path: Path):
    _write_item(tmp_path, "10000", source_hash="a" * 64)
    _write_item(tmp_path, "13000", source_hash="b" * 64)

    with pytest.raises(ValueError, match="source/VSV manifest lineage"):
        collect(
            result_root=tmp_path,
            output=tmp_path / "ensemble",
            configurations=["10000", "13000"],
        )
