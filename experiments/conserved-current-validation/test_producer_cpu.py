"""CPU-only contract tests for the audited real-observable producer."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
PRODUCER = HERE / "produce_real_observables.py"
ANALYZER = HERE / "run_real_gauge_validation.py"


def digest(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lineage_hash(files: dict[str, str]) -> str:
    if len(files) == 1:
        return next(iter(files.values()))
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def artifact(tmp_path: Path, name: str, array: np.ndarray, *, npz: bool = False) -> dict:
    path = tmp_path / name
    if npz:
        np.savez(path, elementals=array)
        loader = {
            "format": "npz",
            "member": "elementals",
            "ndim": array.ndim,
            "dtype_kind": array.dtype.kind,
            "finite": True,
        }
    else:
        np.save(path, array)
        loader = {"format": "npy", "ndim": array.ndim, "dtype_kind": array.dtype.kind, "finite": True}
    return {"status": "required", "path": str(path), "sha256": digest(path), "loader": loader}


def make_manifest(
    tmp_path: Path, *, synthetic: bool = True, ne_sink: int = 2, vsv_shape=(2, 2, 2, 2, 2)
) -> tuple[Path, dict]:
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(["cfg-001", "cfg-002", "cfg-003", "cfg-004", "cfg-005", "cfg-006", "cfg-007", "cfg-008"]))
    sources = []
    for name in ("producer-source.py", "current-source.py"):
        path = tmp_path / name
        path.write_text(f"# {name}\n")
        sources.append({"path": str(path), "sha256": digest(path)})
    real = np.ones
    artifacts = {}
    artifacts["gauge"] = artifact(tmp_path, "gauge.npy", real((2, 2, 2, 2), dtype=np.complex128))
    artifacts["eigenvector"] = artifact(tmp_path, "eigen.npy", real((2, 2, 2, 2), dtype=np.complex128))
    artifacts["vsv"] = artifact(tmp_path, "vsv.npy", real(vsv_shape, dtype=np.complex128))
    artifacts["psv"] = artifact(tmp_path, "psv.npy", real((2, 2, 2, 2, 2, 2), dtype=np.complex128))
    artifacts["overlap"] = artifact(tmp_path, "overlap.npy", real((2, 2, 2), dtype=np.complex128))
    artifacts["current_p2v"] = artifact(tmp_path, "p2v.npy", real((2, 2, 2, 2, 2), dtype=np.complex128))
    artifacts["current_p2p"] = artifact(tmp_path, "p2p.npz", real((2, 2, 2, 2), dtype=np.complex128), npz=True)
    ne = {"source": {"used": 2, "available": 2}, "sink": {"used": ne_sink, "available": ne_sink}}
    contraction = tmp_path / "contractions.npz"
    artifact_hashes = {name: lineage_hash({spec["path"]: spec["sha256"]}) for name, spec in artifacts.items()}
    contraction_provenance = {
        "schema": "lattice.conserved-current-validation.audited-contractions/v1",
        "source_sink_current_time_semantics": "source-first",
        "boundary": "periodic",
        "temporal_extent": 4,
        "ne": ne,
        "input_artifacts": artifact_hashes,
    }
    ids = np.array([f"cfg-{index:03d}" for index in range(1, 9)])
    rng = np.random.default_rng(2468)
    np.savez(
        contraction,
        cfg_ids=ids,
        wt_lhs=rng.normal(0.0, 1e-10, (8, 4)),
        wt_rhs=np.zeros((8, 4)),
        charge_ratio=1.0 + rng.normal(0.0, 1e-3, (8, 4)),
        contact_term=np.zeros((8, 4)),
        provenance_json=np.array(json.dumps(contraction_provenance)),
    )
    manifest = {
        "schema": "lattice.conserved-current-validation.manifest/v1",
        "experiment_id": "fixture",
        "dataset_label": "fixture",
        "synthetic": synthetic,
        "temporal_extent": 4,
        "boundary": "periodic",
        "contact_times": [0],
        "excluded_boundary_times": [],
        "plateau_times": [1, 2],
        "ne": ne,
        "current_api": {
            "version": "1.2.0",
            "term_schema": "lattice.current.term/v1",
            "assembler_schema": "lattice.current.assembler/v1",
        },
        "expected_provenance": {
            "code_sha256": digest(sources[0]["path"]),
            "api_sha256": digest(sources[1]["path"]),
            "code_source_path": sources[0]["path"],
            "api_source_path": sources[1]["path"],
        },
        "source_files": sources,
        "configuration_list": {"path": str(cfg), "sha256": digest(cfg)},
        "producer": {
            "mode": "audited-contractions-v1",
            "source_sink_current_time_semantics": "source-first",
            "p2p_required": True,
            "audited_contractions_path": str(contraction),
            "audited_contractions_sha256": digest(contraction),
        },
        "artifacts": artifacts,
        "git_head": "fixture",
        "git_dirty": True,
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
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def invoke(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(PRODUCER), *args], text=True, capture_output=True, check=False)


def save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest))


def test_dry_run_validates_absolute_hashed_paths_without_array_loading(tmp_path):
    path, _ = make_manifest(tmp_path)
    completed = invoke(
        str(path), "--result-dir", str((tmp_path / "result").resolve()), "--dry-run", "--synthetic-fixture"
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "dry-run-valid"


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda manifest: manifest["artifacts"]["vsv"].update(path="relative.npy"), "must be absolute"),
        (
            lambda manifest: manifest["artifacts"]["vsv"].update(
                path=f"{manifest['artifacts']['vsv']['path']}.missing"
            ),
            "does not exist",
        ),
        (lambda manifest: manifest["artifacts"]["vsv"].update(sha256="0" * 64), "SHA-256 does not match"),
        (lambda manifest: manifest["ne"]["sink"].update(used=3, available=2), "used <= available"),
        (lambda manifest: manifest["artifacts"]["vsv"]["loader"].update(ndim=4), "loader rank"),
        (lambda manifest: manifest["artifacts"]["vsv"]["loader"].update(dtype_kind="f"), "dtype kind"),
    ],
)
def test_path_hash_ne_rank_and_dtype_contract_failures(tmp_path, mutate, expected):
    path, manifest = make_manifest(tmp_path)
    mutate(manifest)
    save_manifest(path, manifest)
    completed = invoke(str(path), "--result-dir", str((tmp_path / "result").resolve()), "--synthetic-fixture")
    assert completed.returncode == 2
    assert expected in completed.stderr


def test_ne_mismatch_with_audited_contractions_is_rejected(tmp_path):
    path, manifest = make_manifest(tmp_path)
    manifest["ne"]["sink"] = {"used": 1, "available": 1}
    save_manifest(path, manifest)
    completed = invoke(str(path), "--result-dir", str((tmp_path / "result").resolve()), "--synthetic-fixture")
    assert completed.returncode == 2
    assert "Ne mismatch" in completed.stderr

    path, _ = make_manifest(tmp_path)
    result = (tmp_path / "result").resolve()
    completed = invoke(str(path), "--result-dir", str(result), "--synthetic-fixture")
    assert completed.returncode == 0, completed.stderr
    payload = json.loads((result / "producer-result.json").read_text())
    done = json.loads((result / "DONE").read_text())
    assert payload["classification"] == "synthetic"
    assert set(done["artifact_sha256"]) == {"observables.npz", "producer-result.json"}
    for name, value in done["artifact_sha256"].items():
        assert digest(result / name) == value
    with np.load(result / "observables.npz", allow_pickle=False) as archive:
        assert archive["validity_mask"].tolist() == [False, True, True, True]
        assert "contact_term" in archive.files
    refused = invoke(str(path), "--result-dir", str(result), "--synthetic-fixture")
    assert refused.returncode == 2
    assert "refusing to reuse" in refused.stderr


def test_real_manifest_rejects_synthetic_switch(tmp_path):
    path, _ = make_manifest(tmp_path, synthetic=False)
    completed = invoke(str(path), "--result-dir", str((tmp_path / "result").resolve()), "--synthetic-fixture")
    assert completed.returncode == 2
    assert "allowed only" in completed.stderr


def test_raw_artifacts_cannot_be_misrepresented_as_physical_observables(tmp_path):
    path, manifest = make_manifest(tmp_path)
    manifest["producer"]["mode"] = "raw-artifacts-v1"
    save_manifest(path, manifest)
    completed = invoke(str(path), "--result-dir", str((tmp_path / "result").resolve()), "--synthetic-fixture")
    assert completed.returncode == 2
    assert "unsupported-observable-input" in completed.stderr


def test_producer_npz_is_accepted_by_analyzer_synthetic_path(tmp_path):
    path, manifest = make_manifest(tmp_path)
    producer_result = (tmp_path / "producer-result").resolve()
    completed = invoke(str(path), "--result-dir", str(producer_result), "--synthetic-fixture")
    assert completed.returncode == 0, completed.stderr
    analysis_manifest = {
        key: manifest[key]
        for key in (
            "schema",
            "experiment_id",
            "dataset_label",
            "synthetic",
            "temporal_extent",
            "boundary",
            "contact_times",
            "excluded_boundary_times",
            "plateau_times",
            "current_api",
            "expected_provenance",
            "git_head",
            "git_dirty",
            "pass_gates",
        )
    }
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(analysis_manifest))
    analyzed = subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            str(analysis_path),
            str(producer_result / "observables.npz"),
            "--result-dir",
            str((tmp_path / "analysis-result").resolve()),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert analyzed.returncode == 0, analyzed.stderr
    assert json.loads(analyzed.stdout)["classification"] == "synthetic"
