"""CPU-only tests for the conserved-current experiment package; fixtures stay in pytest tmp paths."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_conservation import correlated_constant_fit  # noqa: E402

RUNNER = HERE / "run_real_gauge_validation.py"
HASH = "a" * 64


def manifest(*, boundary="periodic", contact=(0,), excluded=(), plateau=(2, 3, 4, 5)):
    return {
        "schema": "lattice.conserved-current-validation.manifest/v1",
        "experiment_id": "cpu-fixture",
        "dataset_label": "synthetic-fixture",
        "synthetic": True,
        "temporal_extent": 8,
        "boundary": boundary,
        "contact_times": list(contact),
        "excluded_boundary_times": list(excluded),
        "plateau_times": list(plateau),
        "current_api": {
            "version": "1.2.0",
            "term_schema": "lattice.current.term/v1",
            "assembler_schema": "lattice.current.assembler/v1",
        },
        "expected_provenance": {"code_sha256": HASH, "api_sha256": HASH},
        "git_head": "fixture-head",
        "git_dirty": True,
        "job": None,
        "pass_gates": {
            "minimum_configurations": 8,
            "wt": {
                "absolute_tolerance": 1e-8,
                "relative_to_statistical_error": 3.0,
                "max_pull": 3.0,
                "p_value_min": 0.01,
            },
            "charge": {
                "absolute_tolerance": 0.01,
                "relative_to_statistical_error": 3.0,
                "max_pull": 3.0,
                "p_value_min": 0.01,
            },
        },
    }


def provenance(*, synthetic=True, boundary="periodic"):
    return {
        "synthetic": synthetic,
        "code_sha256": HASH,
        "api_sha256": HASH,
        "api_version": "1.2.0",
        "term_schema": "lattice.current.term/v1",
        "assembler_schema": "lattice.current.assembler/v1",
        "boundary": boundary,
        "temporal_extent": 8,
        "source_sink_current_time_semantics": "source-first",
        "propagator_artifacts": {
            name: hashlib.sha256(name.encode()).hexdigest()
            for name in (
                "vsv_perambulator_sha256",
                "psv_sha256",
                "overlap_sha256",
                "current_p2v_sha256",
                "current_p2p_sha256",
            )
        },
    }


def fixture(tmp_path, *, failing=False, boundary="periodic", contact=(0,), excluded=(), plateau=(2, 3, 4, 5)):
    rng = np.random.default_rng(737)
    count, extent = 10, 8
    lhs = rng.normal(0.0, 2e-10, (count, extent))
    rhs = np.zeros((count, extent))
    charge = 1.0 + rng.normal(0.0, 8e-4, (count, extent))
    if failing:
        lhs[:, 3] += 0.15
        charge[:, 2:6] += 0.20
    data = tmp_path / "input.npz"
    np.savez(
        data,
        cfg_ids=np.array([f"cfg-{index:03d}" for index in range(count)]),
        lhs=lhs,
        rhs=rhs,
        charge=charge,
        provenance_json=np.array(json.dumps(provenance(boundary=boundary))),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest(boundary=boundary, contact=contact, excluded=excluded, plateau=plateau)), encoding="utf-8"
    )
    return manifest_path, data


def invoke(manifest_path, data, output, *args):
    return subprocess.run(
        [sys.executable, str(RUNNER), str(manifest_path), str(data), "--result-dir", str(output), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_passing_noisy_fixture_writes_machine_artifacts_and_done_hashes(tmp_path):
    manifest_path, data = fixture(tmp_path)
    output = tmp_path / "result"
    completed = invoke(manifest_path, data, output)
    assert completed.returncode == 0, completed.stderr
    result = json.loads((output / "result.json").read_text())
    done = json.loads((output / "DONE").read_text())
    assert result["passed"] is True
    assert result["classification"] == "synthetic"
    assert {"data_table.csv", "wt_diagnostic.svg", "charge_diagnostic.svg", "result.json"} <= set(
        done["artifact_sha256"]
    )
    for name, digest in done["artifact_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    table = (output / "data_table.csv").read_text()
    assert "cfg_id,time,lhs,rhs,residual" in table
    assert "contact" in table
    assert "<svg" in (output / "wt_diagnostic.svg").read_text()


def test_failing_sign_anchor_and_normalization_has_supported_attribution(tmp_path):
    manifest_path, data = fixture(tmp_path, failing=True)
    output = tmp_path / "failed"
    completed = invoke(manifest_path, data, output)
    assert completed.returncode == 1
    result = json.loads((output / "result.json").read_text())
    assert result["passed"] is False
    attribution = result["outcome"]["attribution"]
    assert attribution["links_sign_anchor"]["evidence_status"] == "supported"
    assert attribution["normalization"]["evidence_status"] == "supported"
    assert attribution["temporal_propagator"]["evidence_status"] == "unverified"


def test_open_boundary_and_contact_exclusions_are_recorded(tmp_path):
    manifest_path, data = fixture(tmp_path, boundary="open", contact=(2,), excluded=(0, 1, 7), plateau=(3, 4, 5, 6))
    output = tmp_path / "open"
    completed = invoke(manifest_path, data, output)
    assert completed.returncode == 0, completed.stderr
    table = (output / "data_table.csv").read_text()
    assert "boundary_stencil" in table and "contact" in table


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda archive: archive.update(lhs=np.full((10, 8), np.nan)), "non-finite"),
        (
            lambda archive: archive.update(
                provenance_json=np.array(json.dumps({**provenance(), "api_sha256": "b" * 64}))
            ),
            "does not match",
        ),
    ],
)
def test_malformed_nonfinite_and_provenance_mismatch_rejected(tmp_path, mutator, message):
    manifest_path, data = fixture(tmp_path)
    with np.load(data, allow_pickle=False) as old:
        archive = {key: np.array(old[key]) for key in old.files}
    mutator(archive)
    np.savez(data, **archive)
    completed = invoke(manifest_path, data, tmp_path / "never")
    assert completed.returncode == 2
    assert message in completed.stderr


def test_require_real_gauge_rejects_synthetic_and_refuses_overwrite(tmp_path):
    manifest_path, data = fixture(tmp_path)
    rejected = invoke(manifest_path, data, tmp_path / "rejected", "--require-real-gauge")
    assert rejected.returncode == 2
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel").write_text("keep")
    refused = invoke(manifest_path, data, output)
    assert refused.returncode == 2
    assert (output / "sentinel").read_text() == "keep"


def test_near_singular_covariance_reports_rank_aware_fit():
    values = np.tile(np.array([1.0, 1.0, 1.0, 1.0]), (10, 1))
    fit = correlated_constant_fit(values)
    assert fit["fit_valid"] is False
    assert fit["p_value"] is None
    assert "rank" in fit["p_value_reason"]
