"""CPU subprocess tests for the existing rank-6 VSV smoke CLI."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lattice.current_elemental import (
    contract_directed_current_pair_v2v,
    contract_directed_current_v2v,
    save_directed_current_v2v,
)
from lattice.insertion.current import (
    ConservedVectorCurrent,
    build_current_raw_contract,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPT = HERE / "contract_existing_vsv.py"
PAIR_SCRIPT = HERE / "contract_existing_vsv_pair.py"
MANIFEST_SCRIPT = HERE / "build_vsv_timeslice_manifest.py"
PAIR_REQUIRED_FILES = (
    "experiments/directed-current-v2v/contract_existing_vsv.py",
    "experiments/directed-current-v2v/contract_existing_vsv_pair.py",
    "lattice/__init__.py",
    "lattice/current_elemental.py",
    "lattice/generator/elemental.py",
    "lattice/insertion/__init__.py",
    "lattice/insertion/current.py",
    "lattice/insertion/gauge_link.py",
)
PAIR_FIXTURE_COMMIT = "a" * 40


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "LATTICE_EXECUTION_CLUSTER": "kunshan",
            "LATTICE_SOURCE_GIT_COMMIT": "fixture-head",
            "LATTICE_SOURCE_GIT_DIRTY": "true",
            "LATTICE_EXECUTION_RESOURCES_JSON": json.dumps(
                {"mode": "cpu-artifact-smoke", "nodes": 1, "dcu": 0},
                sort_keys=True,
                separators=(",", ":"),
            ),
            "LATTICE_EXECUTION_JOB_ID": "none-login-readonly",
        }
    )
    return env


def _record_identity(value: dict) -> str:
    semantic = {key: item for key, item in value.items() if key != "record_identity"}
    payload = json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _execution_record(
    path: Path,
    *,
    current_artifact: Path,
    semantics: str,
    layout: str,
    vsv_hash_key: str,
    vsv_hash: str,
    git_commit: str = "fixture-head",
    git_dirty: bool = True,
    source_manifest_sha256: str | None = None,
) -> Path:
    input_hashes = {
        "current_artifact_manifest_sha256": _digest(current_artifact / "manifest.json"),
        vsv_hash_key: vsv_hash,
    }
    if source_manifest_sha256 is not None:
        input_hashes["source_manifest_sha256"] = source_manifest_sha256
    value = {
        "schema": "lattice.current.kunshan-execution-record/v1",
        "version": 1,
        "cluster": "kunshan",
        "configuration": "cfg-001",
        "git": {"commit": git_commit, "dirty": git_dirty},
        "resources": {"mode": "cpu-artifact-smoke", "nodes": 1, "dcu": 0},
        "slurm_job_id": "none-login-readonly",
        "input_hashes": input_hashes,
        "axis_declarations": {
            "vsv_time_axis": semantics,
            "vsv_layout": layout,
            "current_direction": 3,
        },
    }
    value["record_identity"] = _record_identity(value)
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def _source_manifest_identity(value: dict) -> str:
    semantic = {key: item for key, item in value.items() if key != "manifest_identity"}
    payload = json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_source_manifest(
    tmp_path: Path,
    *,
    omit: tuple[str, ...] = (),
    tamper: tuple[str, ...] = (),
) -> Path:
    files = []
    for relative in sorted(set(PAIR_REQUIRED_FILES)):
        if relative in omit:
            continue
        data = (ROOT / relative).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if relative in tamper:
            digest = "0" * 64
        files.append(
            {
                "path": relative,
                "sha256": digest,
                "bytes": len(data),
                "git_state": "M ",
            }
        )
    manifest = {
        "schema": "lattice.current.handover-manifest/v1",
        "version": 1,
        "generated_utc": "2026-09-03T00:00:00+00:00",
        "project_root_at_generation": ROOT.as_posix(),
        "git": {
            "head": PAIR_FIXTURE_COMMIT,
            "branch": "feature/wilson-current-j4",
            "dirty": True,
            "origin_url": None,
            "accepted_external_refs": {},
        },
        "files": files,
        "release": {
            "source_files_git_clean_at_build": False,
            "retention_decision": {
                "path": "docs/data-retention-decision.json",
                "sha256": "0" * 64,
                "status": "pending",
                "resolved": False,
                "restore_check": None,
            },
            "published_git_ref": None,
            "release_prerequisites_ready_at_build": False,
        },
        "verification": {
            "operation": "SHA-256 over exact file bytes",
            "documents_command": "fixture",
            "release_command": "fixture",
        },
    }
    manifest["manifest_identity"] = _source_manifest_identity(manifest)
    path = tmp_path / "source-manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _current_artifact(tmp_path: Path, *, boundary="periodic"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    gauge = tmp_path / "gauge.input"
    eigenvector = tmp_path / "eigenvector.input"
    gauge.write_bytes(b"kunshan-gauge-fixture")
    eigenvector.write_bytes(b"kunshan-eigenvector-fixture")
    values = np.zeros((8, 3, 1, 2, 2), dtype=np.complex128)
    for direction in range(8):
        for time in range(3):
            values[direction, time, 0] = np.array(
                [
                    [direction + time + 1j, 2 - direction * 1j],
                    [3 + time * 1j, direction + 4 - time * 1j],
                ],
                dtype=np.complex128,
            )
    raw = {"v2v": values}
    contract = build_current_raw_contract(
        raw,
        boundary=boundary,
        available_ne=2,
        used_ne=2,
        momentum_count=1,
    )
    artifact = tmp_path / "current-artifact"
    save_directed_current_v2v(
        artifact,
        raw,
        contract,
        configuration="cfg-001",
        momenta=[(0, 0, 0)],
        gauge_source=gauge,
        eigenvector_source=eigenvector,
    )
    return artifact, raw, contract, gauge, eigenvector


def _block(source_time: int, sink_time: int) -> np.ndarray:
    base = 20 * source_time + 5 * sink_time + 1
    real = np.arange(4 * 4 * 3 * 4, dtype=np.float64).reshape(4, 4, 3, 4)
    return (base + real + 1j * (2 * base - real)).astype(np.complex128)


def _vsv(tmp_path: Path, semantics: str) -> tuple[Path, dict[tuple[int, int], np.ndarray]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    physical = {
        (source_time, sink_time): _block(source_time, sink_time) for source_time in range(3) for sink_time in range(3)
    }
    values = np.empty((3, 3, 4, 4, 3, 4), dtype=np.complex128)
    for (source_time, sink_time), block in physical.items():
        second = (sink_time - source_time) % 3 if semantics == "source-relative" else sink_time
        values[source_time, second] = block
    path = tmp_path / f"vsv-{semantics}.npy"
    np.save(path, values, allow_pickle=False)
    return path, physical


def _vsv_timeslices(tmp_path: Path, semantics: str) -> tuple[str, Path, dict[tuple[int, int], np.ndarray]]:
    physical = {
        (source_time, sink_time): _block(source_time, sink_time) for source_time in range(3) for sink_time in range(3)
    }
    directory = tmp_path / f"vsv-timeslices-{semantics}"
    directory.mkdir(parents=True)
    pattern = str((directory / "cfg-001.t{source_time:03d}.npy").resolve())
    for source_time in range(3):
        values = np.empty((3, 4, 4, 3, 4), dtype=np.complex128)
        for sink_time in range(3):
            second = (sink_time - source_time) % 3 if semantics == "source-relative" else sink_time
            values[second] = physical[(source_time, sink_time)]
        np.save(Path(pattern.format(source_time=source_time)), values, allow_pickle=False)
    manifest = (directory / "manifest.json").resolve()
    built = subprocess.run(
        [
            sys.executable,
            str(MANIFEST_SCRIPT),
            "--pattern",
            pattern,
            "--configuration",
            "cfg-001",
            "--temporal-extent",
            "3",
            "--time-axis",
            semantics,
            "--output",
            str(manifest),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    return pattern, manifest, physical


def _invoke_timeslices(
    *,
    pattern: str,
    manifest: Path,
    current_artifact: Path,
    result_dir: Path,
    semantics: str,
):
    execution = _execution_record(
        result_dir.parent / f"{result_dir.name}.execution.json",
        current_artifact=current_artifact,
        semantics=semantics,
        layout="source-timeslices-rank5",
        vsv_hash_key="vsv_timeslice_manifest_sha256",
        vsv_hash=_digest(manifest),
    )
    command = [
        sys.executable,
        str(SCRIPT),
        "--vsv-timeslice-pattern",
        pattern,
        "--vsv-timeslice-manifest",
        str(manifest),
        "--vsv-time-axis",
        semantics,
        "--current-artifact",
        str(current_artifact.resolve()),
        "--result-dir",
        str(result_dir.resolve()),
        "--configuration",
        "cfg-001",
        "--source-time",
        "0",
        "--sink-time",
        "1",
        "--current-time",
        "2",
        "--current-source-ne",
        "1",
        "--current-sink-ne",
        "2",
        "--momentum-index",
        "0",
        "--current-direction",
        "3",
        "--execution-record",
        str(execution),
    ]
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=_runtime_env(),
    )


class _PhysicalAccessor:
    def __init__(self, blocks):
        self.blocks = blocks

    def get(self, source_time, sink_time):
        return self.blocks[(source_time, sink_time)]


def _invoke(
    *,
    vsv: Path,
    current_artifact: Path,
    result_dir: Path,
    semantics: str,
    extra=(),
):
    execution = _execution_record(
        result_dir.parent / f"{result_dir.name}.execution.json",
        current_artifact=current_artifact,
        semantics=semantics,
        layout="full-rank6",
        vsv_hash_key="vsv_sha256",
        vsv_hash=_digest(vsv),
    )
    command = [
        sys.executable,
        str(SCRIPT),
        "--vsv",
        str(vsv.resolve()),
        "--vsv-sha256",
        _digest(vsv),
        "--vsv-time-axis",
        semantics,
        "--current-artifact",
        str(current_artifact.resolve()),
        "--result-dir",
        str(result_dir.resolve()),
        "--configuration",
        "cfg-001",
        "--source-time",
        "0",
        "--sink-time",
        "1",
        "--current-time",
        "2",
        "--current-source-ne",
        "1",
        "--current-sink-ne",
        "2",
        "--momentum-index",
        "0",
        "--current-direction",
        "3",
        "--execution-record",
        str(execution),
        *extra,
    ]
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=_runtime_env(),
    )


@pytest.mark.parametrize("semantics", ["source-relative", "source-sink"])
def test_cli_contracts_rank5_source_timeslice_family(tmp_path, semantics):
    artifact, raw, contract, _, _ = _current_artifact(tmp_path)
    pattern, hash_manifest, physical = _vsv_timeslices(tmp_path, semantics)
    result_dir = tmp_path / f"timeslice-result-{semantics}"
    completed = _invoke_timeslices(
        pattern=pattern,
        manifest=hash_manifest,
        current_artifact=artifact,
        result_dir=result_dir,
        semantics=semantics,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((result_dir / "manifest.json").read_text())
    actual = np.load(result_dir / manifest["result"]["filename"], allow_pickle=False)
    expected = contract_directed_current_v2v(
        ConservedVectorCurrent().terms[6:8],
        raw,
        contract,
        _PhysicalAccessor(physical),
        _PhysicalAccessor(physical),
        source_time=0,
        sink_time=1,
        anchor_time=2,
        current_source_ne=1,
        current_sink_ne=2,
        momentum=0,
    )["value"]
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert manifest["vsv"]["layout"] == "source-timeslices-rank5"
    assert manifest["vsv"]["time_axis_semantics"] == semantics
    assert len(manifest["vsv"]["accessed_disk_indices"]) == 4
    assert all(Path(access["file"]).is_absolute() for access in manifest["vsv"]["accessed_disk_indices"])


def test_timeslice_manifest_rejects_missing_file_and_tampering(tmp_path):
    artifact, _, _, _, _ = _current_artifact(tmp_path)
    pattern, hash_manifest, _ = _vsv_timeslices(tmp_path, "source-relative")
    Path(pattern.format(source_time=2)).unlink()
    missing_result = tmp_path / "timeslice-missing"
    missing = _invoke_timeslices(
        pattern=pattern,
        manifest=hash_manifest,
        current_artifact=artifact,
        result_dir=missing_result,
        semantics="source-relative",
    )
    assert missing.returncode == 2
    assert "does not exist" in missing.stderr
    assert not missing_result.exists()

    pattern, hash_manifest, _ = _vsv_timeslices(tmp_path / "tampered", "source-relative")
    decoded = json.loads(hash_manifest.read_text())
    decoded["files"][0]["sha256"] = "0" * 64
    hash_manifest.write_text(json.dumps(decoded))
    tampered = _invoke_timeslices(
        pattern=pattern,
        manifest=hash_manifest,
        current_artifact=artifact,
        result_dir=tmp_path / "timeslice-tampered",
        semantics="source-relative",
    )
    assert tampered.returncode == 2
    assert "identity" in tampered.stderr


@pytest.mark.parametrize("semantics", ["source-relative", "source-sink"])
def test_cli_contracts_both_declared_time_axes_and_publishes_hashes(tmp_path, semantics):
    artifact, raw, contract, _, _ = _current_artifact(tmp_path)
    vsv, physical = _vsv(tmp_path, semantics)
    result_dir = tmp_path / f"result-{semantics}"
    completed = _invoke(
        vsv=vsv,
        current_artifact=artifact,
        result_dir=result_dir,
        semantics=semantics,
    )
    assert completed.returncode == 0, completed.stderr

    manifest = json.loads((result_dir / "manifest.json").read_text())
    done = json.loads((result_dir / "DONE").read_text())
    result_path = result_dir / manifest["result"]["filename"]
    actual = np.load(result_path, allow_pickle=False)
    expected = contract_directed_current_v2v(
        ConservedVectorCurrent().terms[6:8],
        raw,
        contract,
        _PhysicalAccessor(physical),
        _PhysicalAccessor(physical),
        source_time=0,
        sink_time=1,
        anchor_time=2,
        current_source_ne=1,
        current_sink_ne=2,
        momentum=0,
    )["value"]
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    assert manifest["classification"] == "artifact-smoke-not-physics-validation"
    assert manifest["vsv"]["time_axis_semantics"] == semantics
    assert manifest["vsv"]["disk_axes"] == [
        "source_time",
        "second_time",
        "sink_spin",
        "source_spin",
        "sink_ne",
        "source_ne",
    ]
    assert manifest["consumer"]["term_contraction"] == "afAi,bfji,bcjC->acAC"
    assert manifest["current"]["sources_verified"] is True
    assert manifest["result"]["axes"] == [
        "external_sink_spin",
        "external_source_spin",
        "external_sink_ne",
        "external_source_ne",
    ]
    assert _digest(result_path) == manifest["result"]["sha256"]
    assert _digest(result_path) == done["artifact_sha256"][result_path.name]
    assert _digest(result_dir / "manifest.json") == done["artifact_sha256"]["manifest.json"]
    assert manifest["current"]["direction"] == 3
    assert manifest["current"]["direction_name"] == "t"
    assert len(manifest["vsv"]["accessed_disk_indices"]) == 4


def test_dry_run_writes_nothing_and_records_no_degraded_claim(tmp_path):
    artifact, _, _, gauge, _ = _current_artifact(tmp_path)
    vsv, _ = _vsv(tmp_path, "source-relative")
    gauge.write_bytes(b"source-not-mounted-anymore")
    result_dir = tmp_path / "dry-result"
    completed = _invoke(
        vsv=vsv,
        current_artifact=artifact,
        result_dir=result_dir,
        semantics="source-relative",
        extra=("--dry-run", "--no-verify-current-sources"),
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "dry-run-valid"
    assert payload["sources_verified"] is False
    assert payload["classification"] == "artifact-smoke-not-physics-validation"
    assert payload["validated_vsv_accesses"] == 4
    assert not result_dir.exists()


def test_dry_run_rejects_open_boundary_crossing_before_output(tmp_path):
    artifact, _, _, _, _ = _current_artifact(tmp_path, boundary="open")
    vsv, _ = _vsv(tmp_path, "source-relative")
    result_dir = tmp_path / "open-boundary-dry"
    completed = _invoke(
        vsv=vsv,
        current_artifact=artifact,
        result_dir=result_dir,
        semantics="source-relative",
        extra=("--dry-run",),
    )
    assert completed.returncode == 2
    assert "open temporal boundary" in completed.stderr
    assert not result_dir.exists()


def test_non_dry_smoke_requires_execution_record(tmp_path):
    artifact, _, _, _, _ = _current_artifact(tmp_path)
    vsv, _ = _vsv(tmp_path, "source-relative")
    result_dir = tmp_path / "no-execution-record"
    command = [
        sys.executable,
        str(SCRIPT),
        "--vsv",
        str(vsv.resolve()),
        "--vsv-sha256",
        _digest(vsv),
        "--vsv-time-axis",
        "source-relative",
        "--current-artifact",
        str(artifact.resolve()),
        "--result-dir",
        str(result_dir.resolve()),
        "--configuration",
        "cfg-001",
        "--source-time",
        "0",
        "--sink-time",
        "1",
        "--current-time",
        "2",
        "--current-source-ne",
        "1",
        "--current-sink-ne",
        "2",
        "--current-direction",
        "3",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    assert completed.returncode == 2
    assert "--execution-record" in completed.stderr
    assert not result_dir.exists()


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("bad-hash", "VSV SHA-256"),
        ("rank-five", "rank 6"),
        ("real-dtype", "complex dtype"),
        ("nonfinite", "non-finite"),
    ],
)
def test_cli_rejects_invalid_vsv_without_partial_output(tmp_path, mutation, expected):
    artifact, _, _, _, _ = _current_artifact(tmp_path)
    vsv, _ = _vsv(tmp_path, "source-relative")
    extra = []
    if mutation == "bad-hash":
        extra = ["--vsv-sha256", "0" * 64]
    else:
        values = np.load(vsv, allow_pickle=False)
        if mutation == "rank-five":
            values = values[0]
        elif mutation == "real-dtype":
            values = values.real
        else:
            values[0, 2, 0, 0, 0, 0] = np.nan
        np.save(vsv, values, allow_pickle=False)
    result_dir = tmp_path / "bad-result"
    execution = _execution_record(
        tmp_path / "bad-result.execution.json",
        current_artifact=artifact,
        semantics="source-relative",
        layout="full-rank6",
        vsv_hash_key="vsv_sha256",
        vsv_hash=_digest(vsv),
    )
    command = [
        sys.executable,
        str(SCRIPT),
        "--vsv",
        str(vsv.resolve()),
        "--vsv-sha256",
        _digest(vsv),
        "--vsv-time-axis",
        "source-relative",
        "--current-artifact",
        str(artifact.resolve()),
        "--result-dir",
        str(result_dir.resolve()),
        "--configuration",
        "cfg-001",
        "--source-time",
        "0",
        "--sink-time",
        "1",
        "--current-time",
        "2",
        "--current-source-ne",
        "1",
        "--current-sink-ne",
        "2",
        "--current-direction",
        "3",
        "--execution-record",
        str(execution),
        *extra,
    ]
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=_runtime_env(),
    )
    assert completed.returncode == 2
    assert expected in completed.stderr
    assert not result_dir.exists()
    assert not list(tmp_path.glob(".bad-result.partial-*"))


def test_execution_record_must_match_trusted_runtime(tmp_path):
    artifact, _, _, _, _ = _current_artifact(tmp_path)
    vsv, _ = _vsv(tmp_path, "source-relative")
    result_dir = tmp_path / "runtime-mismatch"
    execution = _execution_record(
        tmp_path / "runtime-mismatch.execution.json",
        current_artifact=artifact,
        semantics="source-relative",
        layout="full-rank6",
        vsv_hash_key="vsv_sha256",
        vsv_hash=_digest(vsv),
    )
    command = [
        sys.executable,
        str(SCRIPT),
        "--vsv",
        str(vsv.resolve()),
        "--vsv-sha256",
        _digest(vsv),
        "--vsv-time-axis",
        "source-relative",
        "--current-artifact",
        str(artifact.resolve()),
        "--result-dir",
        str(result_dir.resolve()),
        "--configuration",
        "cfg-001",
        "--source-time",
        "0",
        "--sink-time",
        "1",
        "--current-time",
        "2",
        "--current-source-ne",
        "1",
        "--current-sink-ne",
        "2",
        "--current-direction",
        "3",
        "--execution-record",
        str(execution),
    ]
    env = _runtime_env()
    env["LATTICE_EXECUTION_JOB_ID"] = "forged-job"
    completed = subprocess.run(command, text=True, capture_output=True, check=False, env=env)
    assert completed.returncode == 2
    assert "slurm_job_id does not match trusted runtime state" in completed.stderr
    assert not result_dir.exists()


def test_stale_legacy_claim_does_not_control_publication(tmp_path):
    artifact, _, _, _, _ = _current_artifact(tmp_path)
    vsv, _ = _vsv(tmp_path, "source-relative")
    result_dir = (tmp_path / "claimed-result").resolve()
    claim = result_dir.parent / f".{result_dir.name}.claim"
    claim.write_text("legacy-stale-claim")
    completed = _invoke(
        vsv=vsv,
        current_artifact=artifact,
        result_dir=result_dir,
        semantics="source-relative",
    )
    assert completed.returncode == 0, completed.stderr
    assert claim.read_text() == "legacy-stale-claim"
    assert result_dir.exists()
    assert not list(tmp_path.glob(".claimed-result.partial-*"))


def test_existing_output_and_missing_time_axis_are_rejected(tmp_path):
    artifact, _, _, _, _ = _current_artifact(tmp_path)
    vsv, _ = _vsv(tmp_path, "source-sink")
    result_dir = tmp_path / "existing"
    result_dir.mkdir()
    completed = _invoke(
        vsv=vsv,
        current_artifact=artifact,
        result_dir=result_dir,
        semantics="source-sink",
    )
    assert completed.returncode == 2
    assert "refusing to reuse" in completed.stderr

    command = [
        sys.executable,
        str(SCRIPT),
        "--vsv",
        str(vsv.resolve()),
        "--vsv-sha256",
        _digest(vsv),
        "--current-artifact",
        str(artifact.resolve()),
        "--result-dir",
        str((tmp_path / "missing-axis").resolve()),
        "--configuration",
        "cfg-001",
        "--source-time",
        "0",
        "--sink-time",
        "1",
        "--current-time",
        "2",
        "--current-source-ne",
        "1",
        "--current-sink-ne",
        "2",
        "--current-direction",
        "3",
    ]
    missing = subprocess.run(command, text=True, capture_output=True, check=False)
    assert missing.returncode == 2
    assert "--vsv-time-axis" in missing.stderr


def _invoke_pair(
    *,
    tmp_path: Path,
    current_artifact: Path,
    result_dir: Path,
    semantics: str,
    vsv: Path | None = None,
    pattern: str | None = None,
    timeslice_manifest: Path | None = None,
    source_manifest: Path | None = None,
    extra=(),
):
    if (vsv is None) == (pattern is None):
        raise AssertionError("pair fixture requires exactly one VSV input")
    if source_manifest is None:
        source_manifest = _write_source_manifest(tmp_path)
    source_sha = _digest(source_manifest)
    if vsv is not None:
        layout = "full-rank6"
        hash_key = "vsv_sha256"
        vsv_hash = _digest(vsv)
        input_args = ["--vsv", str(vsv.resolve()), "--vsv-sha256", vsv_hash]
    else:
        assert timeslice_manifest is not None
        layout = "source-timeslices-rank5"
        hash_key = "vsv_timeslice_manifest_sha256"
        vsv_hash = _digest(timeslice_manifest)
        input_args = [
            "--vsv-timeslice-pattern",
            str(pattern),
            "--vsv-timeslice-manifest",
            str(timeslice_manifest),
        ]
    execution = _execution_record(
        result_dir.parent / f"{result_dir.name}.execution.json",
        current_artifact=current_artifact,
        semantics=semantics,
        layout=layout,
        vsv_hash_key=hash_key,
        vsv_hash=vsv_hash,
        git_commit=PAIR_FIXTURE_COMMIT,
        git_dirty=True,
        source_manifest_sha256=source_sha,
    )
    command = [
        sys.executable,
        str(PAIR_SCRIPT),
        *input_args,
        "--vsv-time-axis",
        semantics,
        "--current-artifact",
        str(current_artifact.resolve()),
        "--source-manifest",
        str(source_manifest.resolve()),
        "--result-dir",
        str(result_dir.resolve()),
        "--configuration",
        "cfg-001",
        "--first-current-time",
        "2",
        "--second-current-time",
        "0",
        "--current-ne",
        "2",
        "--first-momentum-index",
        "0",
        "--second-momentum-index",
        "0",
        "--current-direction",
        "3",
        "--execution-record",
        str(execution),
        *extra,
    ]
    env = _runtime_env()
    env["LATTICE_SOURCE_GIT_COMMIT"] = PAIR_FIXTURE_COMMIT
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


@pytest.mark.parametrize("semantics", ["source-relative", "source-sink"])
def test_pair_cli_contracts_full_vsv_and_publishes_unsigned_scalar(tmp_path, semantics):
    artifact, raw, contract, _, _ = _current_artifact(tmp_path)
    vsv, physical = _vsv(tmp_path, semantics)
    result_dir = tmp_path / f"pair-result-{semantics}"
    completed = _invoke_pair(
        tmp_path=tmp_path,
        current_artifact=artifact,
        result_dir=result_dir,
        semantics=semantics,
        vsv=vsv,
    )
    assert completed.returncode == 0, completed.stderr

    manifest = json.loads((result_dir / "manifest.json").read_text())
    done = json.loads((result_dir / "DONE").read_text())
    result_path = result_dir / manifest["result"]["filename"]
    actual = np.load(result_path, allow_pickle=False)
    expected = contract_directed_current_pair_v2v(
        ConservedVectorCurrent().terms[6:8],
        raw,
        contract,
        ConservedVectorCurrent().terms[6:8],
        raw,
        contract,
        _PhysicalAccessor(physical),
        first_anchor_time=2,
        second_anchor_time=0,
        first_field_ne=2,
        first_bar_ne=2,
        second_field_ne=2,
        second_bar_ne=2,
    )["value"]
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert actual.shape == ()
    assert manifest["classification"] == ("artifact-smoke-unflavored-unsigned-not-physics-validation")
    assert manifest["consumer"]["schema"] == ("lattice.current.v2v-term-pair-contraction/v1")
    assert manifest["consumer"]["term_pair_contraction"] == ("bfji,ackl,afki,bcjl->")
    assert len(manifest["consumer"]["term_pairs"]) == 4
    assert len(manifest["vsv"]["accessed_disk_indices"]) == 8
    assert manifest["result"]["axes"] == []
    assert _digest(result_path) == manifest["result"]["sha256"]
    assert _digest(result_path) == done["artifact_sha256"][result_path.name]
    assert _digest(result_dir / "manifest.json") == done["artifact_sha256"]["manifest.json"]


def test_pair_cli_contracts_rank5_family_and_dry_run_writes_nothing(tmp_path):
    artifact, raw, contract, _, _ = _current_artifact(tmp_path)
    pattern, hash_manifest, physical = _vsv_timeslices(tmp_path, "source-relative")
    result_dir = tmp_path / "pair-timeslices"
    completed = _invoke_pair(
        tmp_path=tmp_path,
        current_artifact=artifact,
        result_dir=result_dir,
        semantics="source-relative",
        pattern=pattern,
        timeslice_manifest=hash_manifest,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((result_dir / "manifest.json").read_text())
    actual = np.load(result_dir / manifest["result"]["filename"], allow_pickle=False)
    expected = contract_directed_current_pair_v2v(
        ConservedVectorCurrent().terms[6:8],
        raw,
        contract,
        ConservedVectorCurrent().terms[6:8],
        raw,
        contract,
        _PhysicalAccessor(physical),
        first_anchor_time=2,
        second_anchor_time=0,
        first_field_ne=2,
        first_bar_ne=2,
        second_field_ne=2,
        second_bar_ne=2,
    )["value"]
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert manifest["vsv"]["layout"] == "source-timeslices-rank5"
    assert len(manifest["vsv"]["accessed_disk_indices"]) == 8

    dry_result = tmp_path / "pair-dry"
    dry = _invoke_pair(
        tmp_path=tmp_path,
        current_artifact=artifact,
        result_dir=dry_result,
        semantics="source-relative",
        pattern=pattern,
        timeslice_manifest=hash_manifest,
        extra=("--dry-run",),
    )
    assert dry.returncode == 0, dry.stderr
    payload = json.loads(dry.stdout)
    assert payload["status"] == "dry-run-valid"
    assert payload["validated_term_pairs"] == 4
    assert payload["validated_vsv_accesses"] == 8
    assert payload["source_manifest_files"] == len(PAIR_REQUIRED_FILES)
    assert not dry_result.exists()


def test_pair_cli_requires_execution_record_and_rejects_open_crossing(tmp_path):
    artifact, _, _, _, _ = _current_artifact(tmp_path)
    vsv, _ = _vsv(tmp_path, "source-relative")
    result_dir = tmp_path / "pair-no-record"
    command = [
        sys.executable,
        str(PAIR_SCRIPT),
        "--vsv",
        str(vsv.resolve()),
        "--vsv-sha256",
        _digest(vsv),
        "--vsv-time-axis",
        "source-relative",
        "--current-artifact",
        str(artifact.resolve()),
        "--source-manifest",
        str(_write_source_manifest(tmp_path).resolve()),
        "--result-dir",
        str(result_dir.resolve()),
        "--configuration",
        "cfg-001",
        "--first-current-time",
        "2",
        "--second-current-time",
        "0",
        "--current-ne",
        "2",
        "--current-direction",
        "3",
    ]
    missing = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=_runtime_env(),
    )
    assert missing.returncode == 2
    assert "--execution-record" in missing.stderr
    assert not result_dir.exists()

    open_artifact, _, _, _, _ = _current_artifact(tmp_path / "open", boundary="open")
    open_vsv, _ = _vsv(tmp_path / "open", "source-relative")
    open_result = tmp_path / "pair-open"
    crossing = _invoke_pair(
        tmp_path=tmp_path,
        current_artifact=open_artifact,
        result_dir=open_result,
        semantics="source-relative",
        vsv=open_vsv,
        extra=("--dry-run",),
    )
    assert crossing.returncode == 2
    assert "open temporal boundary" in crossing.stderr
    assert not open_result.exists()


def test_pair_cli_rejects_tampered_or_incomplete_source_manifest(tmp_path):
    artifact, _, _, _, _ = _current_artifact(tmp_path)
    vsv, _ = _vsv(tmp_path, "source-relative")
    result_dir = tmp_path / "pair-tampered-source"
    tampered = _write_source_manifest(tmp_path, tamper=("lattice/current_elemental.py",))
    completed = _invoke_pair(
        tmp_path=tmp_path,
        current_artifact=artifact,
        result_dir=result_dir,
        semantics="source-relative",
        vsv=vsv,
        source_manifest=tampered,
        extra=("--dry-run",),
    )
    assert completed.returncode == 2
    assert "bytes do not match" in completed.stderr
    assert not result_dir.exists()

    result_dir = tmp_path / "pair-missing-source"
    omitted = _write_source_manifest(
        tmp_path,
        omit=("lattice/insertion/current.py",),
    )
    missing_dependency = _invoke_pair(
        tmp_path=tmp_path,
        current_artifact=artifact,
        result_dir=result_dir,
        semantics="source-relative",
        vsv=vsv,
        source_manifest=omitted,
        extra=("--dry-run",),
    )
    assert missing_dependency.returncode == 2
    assert "lacks required pair dependencies" in missing_dependency.stderr
    assert not result_dir.exists()


def test_pair_cli_rejects_execution_record_source_mismatch(tmp_path):
    artifact, _, _, _, _ = _current_artifact(tmp_path)
    vsv, _ = _vsv(tmp_path, "source-relative")
    source_manifest = _write_source_manifest(tmp_path)
    result_dir = tmp_path / "pair-record-mismatch"
    layout = "full-rank6"
    execution = _execution_record(
        result_dir.parent / f"{result_dir.name}.execution.json",
        current_artifact=artifact,
        semantics="source-relative",
        layout=layout,
        vsv_hash_key="vsv_sha256",
        vsv_hash=_digest(vsv),
        git_commit=PAIR_FIXTURE_COMMIT,
        git_dirty=True,
        source_manifest_sha256="b" * 64,
    )
    command = [
        sys.executable,
        str(PAIR_SCRIPT),
        "--vsv",
        str(vsv.resolve()),
        "--vsv-sha256",
        _digest(vsv),
        "--vsv-time-axis",
        "source-relative",
        "--current-artifact",
        str(artifact.resolve()),
        "--source-manifest",
        str(source_manifest.resolve()),
        "--result-dir",
        str(result_dir.resolve()),
        "--configuration",
        "cfg-001",
        "--first-current-time",
        "2",
        "--second-current-time",
        "0",
        "--current-ne",
        "2",
        "--current-direction",
        "3",
        "--execution-record",
        str(execution),
    ]
    env = _runtime_env()
    env["LATTICE_SOURCE_GIT_COMMIT"] = PAIR_FIXTURE_COMMIT
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 2
    assert "source manifest hash does not match" in completed.stderr
    assert not result_dir.exists()
