"""CPU tests for the full-size localized input coverage audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from audit_fullsize_coverage import (
    CONFIGURATIONS,
    OVERLAP_DIRNAME,
    PSV_DIRNAME,
    PSP_DIRNAME,
    VSV_DIRNAME,
    _canonical_bytes,
    audit,
)


PRODUCTS = {
    "VSV": (VSV_DIRNAME, [4, 4, 2, 2]),
    "PSV": (PSV_DIRNAME, [4, 4, 2, 1, 2]),
    "PSP": (PSP_DIRNAME, [4, 4, 1, 1, 1, 1]),
}


def _write_manifest(directory: Path, product: str, tail_shape: list[int], source_times: list[int]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "layout": "source-time-rank-slab",
        "product": product,
        "time_convention": "absolute-global-source-and-sink",
        "global_lattice": [2, 2, 2, 4],
        "grid_size": [1, 1, 1, 2],
        "local_lattice": [2, 2, 2, 2],
        "source_times": source_times,
        "tail_shape": tail_shape,
        "dtype": "<c16",
        "mass": -0.277,
        "clover": 1.160920226,
        "t_boundary": -1,
        "gauge_prefix": "/gauge/",
        "point_root": "/points/",
        "tol": 1e-12,
        "maxiter": 1000,
        "stout_steps": 20,
        "stout_rho": 0.12,
        "multigrid": [[1, 1, 1, 1]],
    }
    if product == "PSP":
        manifest.pop("tail_shape")
        manifest.update(
            {
                "np_snk": 1,
                "np_src": 1,
                "axis_order": [
                    "t_sink_local",
                    "spin_sink",
                    "spin_source",
                    "point_sink",
                    "color_sink",
                    "point_source",
                    "color_source",
                ],
            }
        )
    (directory / "manifest.json").write_bytes(_canonical_bytes(manifest))


def _make_dataset(root: Path, *, source_times: dict[str, list[int]] | None = None) -> None:
    source_times = source_times or {name: [0, 1, 2, 3] for name in PRODUCTS}
    for product, (dirname, tail_shape) in PRODUCTS.items():
        directory = root / dirname
        _write_manifest(directory, product, tail_shape, source_times[product])
        shape = (2, 4, 4, 1, 3, 1, 3) if product == "PSP" else (2, *tail_shape)
        for configuration in CONFIGURATIONS:
            for source_time in source_times[product]:
                for rank in range(2):
                    np.save(
                        directory / f"{configuration}.t{source_time:03d}.rank{rank:04d}.npy",
                        np.zeros(shape, dtype=np.complex128),
                        allow_pickle=False,
                    )
    overlap = root / OVERLAP_DIRNAME
    overlap.mkdir(parents=True)
    for configuration in CONFIGURATIONS:
        np.save(overlap / f"{configuration}.overlap_matrix.npy", np.zeros((4, 2, 2, 1), dtype=np.complex128))
        points = root / "01.sparsened_field"
        points.mkdir(exist_ok=True)
        np.save(points / f"{configuration}.npy", np.zeros((2, 4, 1), dtype=np.int32))
        eigenvectors = root / "02.laplace_eigs.nev128"
        eigenvectors.mkdir(exist_ok=True)
        np.save(eigenvectors / f"{configuration}.npy", np.zeros((4, 2, 2, 2, 2, 3), dtype=np.complex128))


def test_audit_reports_complete_inputs_and_pair_accesses(tmp_path: Path):
    _make_dataset(tmp_path)
    report = audit(
        data_root=tmp_path,
        output=tmp_path / "audit.json",
        configurations=list(CONFIGURATIONS),
        current_times=[0, 1, 2, 3],
        hash_files=True,
        overlap_point_count=2,
    )

    assert report["verdict"]["all_requested_files_complete"] is True
    assert report["verdict"]["direct_vsv_pair_endpoint_coverage"] is True
    assert report["verdict"]["v2v_pair_ready"] is True
    assert report["pair_vsv_coverage"]["required_source_times"] == [0, 1, 2, 3]
    assert report["products"]["VSV"]["valid_file_count"] == 8 * 4 * 2
    assert json.loads((tmp_path / "audit.json").read_text())["report_identity"] == report["report_identity"]


def test_audit_rejects_missing_vsv_endpoint_without_promoting_psv_or_psp(tmp_path: Path):
    _make_dataset(
        tmp_path,
        source_times={"VSV": [0, 2], "PSV": [0, 1, 2, 3], "PSP": [0, 2]},
    )
    report = audit(
        data_root=tmp_path,
        output=tmp_path / "audit.json",
        configurations=list(CONFIGURATIONS),
        current_times=[0, 1, 2, 3],
        hash_files=False,
        overlap_point_count=2,
    )

    assert report["verdict"]["all_requested_files_complete"] is True
    assert report["verdict"]["direct_vsv_pair_endpoint_coverage"] is False
    assert report["verdict"]["v2v_pair_ready"] is False
    assert report["pair_vsv_coverage"]["missing_direct_source_times"] == [1, 3]
    assert report["verdict"]["reasons"] == [
        "direct VSV source-time coverage is incomplete for the requested dual-current endpoints"
    ]


def test_audit_rejects_incompatible_product_manifests(tmp_path: Path):
    _make_dataset(tmp_path)
    manifest_path = tmp_path / PSV_DIRNAME / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["mass"] = -0.28
    manifest_path.write_bytes(_canonical_bytes(manifest))

    with pytest.raises(ValueError, match="incompatible manifest fields"):
        audit(
            data_root=tmp_path,
            output=tmp_path / "audit.json",
            configurations=[CONFIGURATIONS[0]],
            current_times=[0],
            hash_files=False,
            overlap_point_count=2,
        )
