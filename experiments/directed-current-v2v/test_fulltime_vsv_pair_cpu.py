"""CPU tests for the full-time VSV accessor and pair matrix path."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from lattice.insertion.current import build_current_raw_contract


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from contract_fulltime_vsv_pair import (  # noqa: E402
    FullTimeVSVSlabAccessor,
    _matrix,
)


def _write_vsv(root: Path, configuration: str = "cfg-001") -> Path:
    directory = root / "vsv"
    directory.mkdir()
    manifest = {
        "version": 1,
        "layout": "source-time-rank-slab",
        "product": "VSV",
        "time_convention": "absolute-global-source-and-sink",
        "global_lattice": [2, 2, 2, 4],
        "grid_size": [1, 1, 1, 2],
        "local_lattice": [2, 2, 2, 2],
        "source_times": [0, 1, 2, 3],
        "tail_shape": [4, 4, 1, 1],
        "dtype": "<c16",
    }
    (directory / "manifest.json").write_bytes(json.dumps(manifest, sort_keys=True).encode())
    for source_time in range(4):
        for rank in range(2):
            block = np.zeros((2, 4, 4, 1, 1), dtype=np.complex128)
            block[:, :, :, :, :] = source_time * 10 + rank
            np.save(directory / f"{configuration}.t{source_time:03d}.rank{rank:04d}.npy", block, allow_pickle=False)
    return directory


def test_fulltime_accessor_maps_absolute_sink_to_rank_and_local_time(tmp_path: Path):
    directory = _write_vsv(tmp_path)
    accessor = FullTimeVSVSlabAccessor(directory, "cfg-001", verify_hashes=True)
    assert accessor.shape == (4, 4, 4, 4, 1, 1)
    np.testing.assert_allclose(accessor.get(2, 3), 21)
    np.testing.assert_allclose(accessor.get(2, 0), 20)
    assert accessor.provenance()["accessed_blocks"][0]["rank"] == 1
    accessor.verify_stable()


def test_fulltime_matrix_reuses_duplicate_vsv_blocks_per_anchor_pair(tmp_path: Path):
    directory = _write_vsv(tmp_path)
    accessor = FullTimeVSVSlabAccessor(directory, "cfg-001")
    raw = {"v2v": np.ones((8, 4, 1, 1, 1), dtype=np.complex128)}
    contract = build_current_raw_contract(
        raw,
        boundary="periodic",
        available_ne=1,
        used_ne=1,
        momentum_count=1,
    )
    args = SimpleNamespace(
        temporal_extent=4,
        current_direction=3,
        wilson_r=1.0,
        current_ne=1,
        first_momentum_index=0,
        second_momentum_index=0,
    )

    result, records = _matrix(
        args,
        {"raw": raw, "contract": contract},
        accessor,
        np,
    )

    assert result.shape == (4, 4)
    assert np.all(np.isfinite(result))
    assert len(records) == 16
    assert len(accessor.provenance()["accessed_blocks"]) == 104


def test_fulltime_accessor_rejects_incomplete_source_time_set(tmp_path: Path):
    directory = _write_vsv(tmp_path)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_times"] = [0, 2, 3]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    try:
        FullTimeVSVSlabAccessor(directory, "cfg-001")
    except ValueError as error:
        assert "source_times" in str(error)
    else:
        raise AssertionError("incomplete source-time set was accepted")
