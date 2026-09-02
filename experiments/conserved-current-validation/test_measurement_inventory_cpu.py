"""CPU tests for measurement inventory construction."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "build_measurement_inventory.py"
SPEC = importlib.util.spec_from_file_location("measurement_inventory", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path: Path):
    evidence = tmp_path / "RUN_NOTES.txt"
    evidence.write_text("fixture evidence\n")
    pattern = str((tmp_path / "{configuration}.npy").resolve())
    for index, configuration in enumerate(("10000", "13000")):
        np.save(
            Path(pattern.format(configuration=configuration)),
            np.full((4, 4), index + 1j, dtype=np.complex128),
        )
    return {
        "schema": MODULE.SOURCE_SCHEMA,
        "version": 1,
        "ensemble_label": "fixture",
        "temporal_extent": 4,
        "boundary": "periodic",
        "configuration_ids": ["10000", "13000"],
        "datasets": [
            {
                "id": "candidate-c2",
                "role": "candidate-two-point",
                "file_pattern": pattern,
                "operator_source": "rho",
                "operator_sink": "rho",
                "current_component": None,
                "topology": "connected-v2v",
                "time_axes": ["source_time", "relative_sink_time"],
                "evidence_path": str(evidence.resolve()),
                "notes": "fixture",
            }
        ],
        "notes": "fixture source",
    }


def test_builds_hashed_inventory(tmp_path):
    result = MODULE.build(_source(tmp_path), verify_finite=True)
    assert result["schema"] == MODULE.OUTPUT_SCHEMA
    dataset = result["datasets"][0]
    assert dataset["configuration_ids"] == ["10000", "13000"]
    assert dataset["files"][0]["shape"] == [4, 4]
    assert dataset["files"][0]["dtype"] == "<c16"
    assert dataset["files"][0]["sha256"] == _digest(Path(dataset["files"][0]["path"]))
    assert dataset["evidence"]["sha256"] == _digest(Path(dataset["evidence"]["path"]))


def test_missing_configuration_file_is_rejected(tmp_path):
    source = _source(tmp_path)
    Path(source["datasets"][0]["file_pattern"].format(configuration="13000")).unlink()
    with pytest.raises(MODULE.InventoryError, match="does not exist"):
        MODULE.build(source, verify_finite=True)


def test_nonfinite_data_is_rejected(tmp_path):
    source = _source(tmp_path)
    path = Path(source["datasets"][0]["file_pattern"].format(configuration="10000"))
    values = np.load(path)
    values[0, 0] = np.nan
    np.save(path, values)
    with pytest.raises(MODULE.InventoryError, match="non-finite"):
        MODULE.build(source, verify_finite=True)


def test_shape_mismatch_is_rejected(tmp_path):
    source = _source(tmp_path)
    path = Path(source["datasets"][0]["file_pattern"].format(configuration="13000"))
    np.save(path, np.ones((3, 4), dtype=np.complex128))
    with pytest.raises(MODULE.InventoryError, match="share shape and dtype"):
        MODULE.build(source, verify_finite=False)
