"""CPU tests for conserved-current measurement readiness auditing."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "audit_measurement_readiness.py"
SPEC = importlib.util.spec_from_file_location("measurement_readiness", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract(tmp_path: Path, *, status="approved", missing_formula=False):
    approval = tmp_path / "approval.md"
    approval.write_text("approved fixture contract\n")
    return {
        "schema": MODULE.CONTRACT_SCHEMA,
        "version": 1,
        "status": status,
        "observable": "connected-charge-normalization",
        "approval": {
            "authority": "fixture-authority" if status == "approved" else "REQUIRES_AUTHORITY",
            "approved_at": "2026-09-01T00:00:00Z" if status == "approved" else "REQUIRES_APPROVAL_TIME",
            "document_path": str(approval.resolve()) if status == "approved" else "REQUIRES_APPROVAL_DOCUMENT",
            "document_sha256": _digest(approval) if status == "approved" else "REQUIRES_APPROVAL_HASH",
        },
        "ensemble": {
            "label": "fixture-ensemble",
            "configuration_ids": [f"cfg-{index:03d}" for index in range(8)],
            "minimum_configurations": 8,
        },
        "temporal_extent": 4,
        "boundary": "periodic",
        "operators": {
            "source": {
                "name": "rho",
                "definition": "gamma_i smeared V2V",
                "momentum": [0, 0, 0],
                "projector": "T1 row average",
            },
            "sink": {
                "name": "rho",
                "definition": "gamma_i smeared V2V",
                "momentum": [0, 0, 0],
                "projector": "T1 row average",
            },
        },
        "current": {
            "implementation": "lattice.insertion.current.ConservedVectorCurrent",
            "component": 3,
            "wilson_r": 1.0,
            "topology": "connected-v2v",
            "flavor_weights": {"u": 0.6666666666666666, "d": -0.3333333333333333},
        },
        "formulas": {
            "c2": "C2(ts,tf)=Tr[P O(tf) S Obar(ts) S]",
            "c3": "REQUIRES_C3_FORMULA" if missing_formula else "C3(ts,t,tf)=Tr[P O(tf) S J4(t) S Obar(ts) S]",
            "ratio": "R=C3/C2",
            "wt_lhs": "unused",
            "wt_rhs": "unused",
            "contact_term": "unused",
        },
        "time_policy": {
            "source_times": [0],
            "sink_times": [2],
            "current_times": [1],
            "contact_times": [0, 2],
            "excluded_boundary_times": [],
            "plateau_times": [1],
        },
        "required_products": ["two-point-denominator", "three-point-numerator"],
        "notes": "fixture",
    }


def _dataset(tmp_path: Path, role: str, identifier: str):
    cfgs = [f"cfg-{index:03d}" for index in range(8)]
    evidence = tmp_path / f"{identifier}.evidence.txt"
    evidence.write_text(f"evidence for {identifier}\n")
    if role == "three-point-numerator":
        shape = (4, 4, 4)
        time_axes = ["source_time", "current_time", "sink_time"]
    else:
        shape = (4, 4)
        time_axes = ["source_time", "sink_time"]
    files = []
    for index, cfg in enumerate(cfgs):
        path = tmp_path / f"{identifier}.{cfg}.npy"
        np.save(path, np.full(shape, index + 1j, dtype=np.complex128))
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        files.append(
            {
                "configuration": cfg,
                "path": str(path.resolve()),
                "sha256": _digest(path),
                "shape": list(array.shape),
                "dtype": array.dtype.str,
            }
        )
    return {
        "id": identifier,
        "role": role,
        "configuration_ids": cfgs,
        "files": files,
        "operator_source": "rho",
        "operator_sink": "rho",
        "current_component": 3 if role == "three-point-numerator" else None,
        "topology": "connected-v2v",
        "time_axes": time_axes,
        "evidence": {"path": str(evidence.resolve()), "sha256": _digest(evidence)},
        "notes": "fixture dataset",
    }


def _inventory(tmp_path: Path, roles):
    return {
        "schema": MODULE.INVENTORY_SCHEMA,
        "version": 1,
        "ensemble_label": "fixture-ensemble",
        "temporal_extent": 4,
        "boundary": "periodic",
        "datasets": [_dataset(tmp_path, role, role) for role in roles],
        "notes": "fixture inventory",
    }


def test_approved_complete_contract_is_ready(tmp_path):
    report = MODULE.audit(
        _contract(tmp_path),
        _inventory(tmp_path, ["two-point-denominator", "three-point-numerator"]),
        verify_files=True,
    )
    assert report["ready"] is True
    assert report["blockers"] == []
    assert all(candidate["usable"] for candidates in report["product_candidates"].values() for candidate in candidates)


def test_draft_missing_formula_and_c3_is_not_ready(tmp_path):
    report = MODULE.audit(
        _contract(tmp_path, status="draft", missing_formula=True),
        _inventory(tmp_path, ["candidate-two-point", "meson-current-two-point"]),
        verify_files=True,
    )
    assert report["ready"] is False
    assert "measurement contract is not approved" in report["blockers"]
    assert "measurement contract lacks formulas.c3" in report["blockers"]
    assert "no usable inventory dataset for required product three-point-numerator" in report["blockers"]
    assert "no usable inventory dataset for required product two-point-denominator" in report["blockers"]


def test_inventory_hash_tampering_is_rejected(tmp_path):
    inventory = _inventory(tmp_path, ["two-point-denominator", "three-point-numerator"])
    Path(inventory["datasets"][0]["files"][0]["path"]).write_bytes(b"tampered")
    with pytest.raises(MODULE.ReadinessError, match="SHA-256 mismatch"):
        MODULE.audit(_contract(tmp_path), inventory, verify_files=True)


def test_unverified_inventory_can_never_be_ready(tmp_path):
    report = MODULE.audit(
        _contract(tmp_path),
        _inventory(tmp_path, ["two-point-denominator", "three-point-numerator"]),
        verify_files=False,
    )
    assert report["files_verified"] is False
    assert report["ready"] is False
    assert "inventory files, headers, hashes, finiteness, and evidence were not verified" in report["blockers"]


def test_invalid_role_axes_and_shape_are_rejected(tmp_path):
    inventory = _inventory(tmp_path, ["two-point-denominator", "three-point-numerator"])
    c3 = inventory["datasets"][1]
    c3["time_axes"] = ["source_time", "sink_time"]
    with pytest.raises(MODULE.ReadinessError, match="time_axes"):
        MODULE.audit(_contract(tmp_path), inventory, verify_files=False)


def test_topology_mismatch_keeps_product_unusable(tmp_path):
    inventory = _inventory(tmp_path, ["two-point-denominator", "three-point-numerator"])
    inventory["datasets"][1]["topology"] = "connected-point"
    report = MODULE.audit(_contract(tmp_path), inventory, verify_files=True)
    assert report["ready"] is False
    candidates = report["product_candidates"]["three-point-numerator"]
    assert candidates[0]["reasons"] == ["topology mismatch"]


def test_historical_localized_topology_is_valid_but_not_connected_v2v(tmp_path):
    inventory = _inventory(tmp_path, ["two-point-denominator", "three-point-numerator"])
    inventory["datasets"][1]["topology"] = "connected-localized-blending"
    report = MODULE.audit(_contract(tmp_path), inventory, verify_files=True)
    assert report["ready"] is False
    candidates = report["product_candidates"]["three-point-numerator"]
    assert candidates[0]["reasons"] == ["topology mismatch"]


def test_uppercase_hashes_are_normalized_for_verification(tmp_path):
    inventory = _inventory(tmp_path, ["two-point-denominator", "three-point-numerator"])
    inventory["datasets"][0]["files"][0]["sha256"] = inventory["datasets"][0]["files"][0]["sha256"].upper()
    report = MODULE.audit(_contract(tmp_path), inventory, verify_files=True)
    assert report["ready"] is True


def test_operator_mismatch_keeps_product_unusable(tmp_path):
    inventory = _inventory(tmp_path, ["two-point-denominator", "three-point-numerator"])
    inventory["datasets"][1]["operator_sink"] = "pion"
    report = MODULE.audit(_contract(tmp_path), inventory, verify_files=False)
    assert report["ready"] is False
    candidates = report["product_candidates"]["three-point-numerator"]
    assert candidates == [
        {
            "dataset_id": "three-point-numerator",
            "usable": False,
            "reasons": ["sink operator mismatch"],
        }
    ]
