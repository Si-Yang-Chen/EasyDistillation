from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "build_handover_manifest.py"


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def _write_retention(root: Path, status: str) -> None:
    resolved = status in {"approved", "not-required"}
    restore_check = None
    if status == "approved":
        evidence = root / "docs" / "restore-check.txt"
        evidence.write_text("restore check passed\n", encoding="utf-8")
        restore_check = {
            "status": "passed",
            "checked_at": "2026-09-03T00:00:00Z",
            "checked_by": "fixture-owner",
            "evidence_path": "docs/restore-check.txt",
            "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        }
    (root / "docs" / "data-retention-decision.json").write_text(
        json.dumps(
            {
                "schema": "lattice.current.data-retention-decision/v1",
                "version": 1,
                "status": status,
                "authority": "fixture-owner" if resolved else None,
                "decided_at": "2026-09-03T00:00:00Z" if resolved else None,
                "secondary_copy": ("/verified/archive" if status == "approved" else None),
                "retention_period": "project lifetime" if resolved else None,
                "restore_owner": "fixture-owner" if resolved else None,
                "restore_check": restore_check,
                "deletion_policy": "do not delete without owner approval",
                "rationale": ("fixture decision is resolved" if resolved else "fixture decision remains pending"),
            }
        ),
        encoding="utf-8",
    )


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    assert _git(root, "init").returncode == 0
    assert _git(root, "config", "user.email", "fixture@example.invalid").returncode == 0
    assert _git(root, "config", "user.name", "Fixture").returncode == 0
    (root / "HANDOVER.md").write_text("# Handover\n", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "runbook.md").write_text("# Runbook\n", encoding="utf-8")
    _write_retention(root, "pending")
    assert _git(root, "add", "HANDOVER.md").returncode == 0
    assert _git(root, "commit", "-m", "fixture").returncode == 0
    return root


def _run(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *(str(value) for value in args)],
        text=True,
        capture_output=True,
        check=False,
    )


def _build(root: Path, output: Path, *extra: object) -> subprocess.CompletedProcess[str]:
    command: list[object] = [
        "build",
        "--root",
        root,
        "--output",
        output,
        "--file",
        "HANDOVER.md",
        "--file",
        "docs/runbook.md",
        "--file",
        "docs/data-retention-decision.json",
    ]
    if (root / "docs" / "restore-check.txt").is_file():
        command.extend(("--file", "docs/restore-check.txt"))
    command.extend(
        (
            "--retention-decision",
            "docs/data-retention-decision.json",
            *extra,
        )
    )
    return _run(*command)


def test_build_and_verify_handover_manifest(tmp_path):
    root = _repo(tmp_path)
    output = (root / "docs" / "handover-manifest.json").resolve()
    accepted = "a" * 40
    completed = _build(
        root,
        output,
        "--accepted-ref",
        f"other-agent={accepted.upper()}",
    )
    assert completed.returncode == 0, completed.stderr

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["schema"] == "lattice.current.handover-manifest/v1"
    assert manifest["git"]["head"] == _git(root, "rev-parse", "HEAD").stdout.strip()
    assert manifest["git"]["dirty"] is True
    assert manifest["git"]["accepted_external_refs"] == {"other-agent": accepted}
    assert [record["path"] for record in manifest["files"]] == [
        "HANDOVER.md",
        "docs/data-retention-decision.json",
        "docs/runbook.md",
    ]
    assert manifest["files"][0]["git_state"] == "clean"
    assert manifest["files"][1]["git_state"] == "??"
    assert manifest["files"][2]["git_state"] == "??"
    assert manifest["release"]["retention_decision"]["resolved"] is False
    assert manifest["release"]["release_prerequisites_ready_at_build"] is False

    verified = _run("verify", output, "--root", root)
    assert verified.returncode == 0, verified.stderr
    payload = json.loads(verified.stdout)
    assert payload["status"] == "documents-verified"
    assert payload["file_count"] == 3
    assert payload["manifest_identity"] == manifest["manifest_identity"]
    assert payload["release_ready"] is False
    assert "large-data retention decision is unresolved" in payload["release_blockers"]

    release = _run(
        "verify",
        output,
        "--root",
        root,
        "--require-release-ready",
    )
    assert release.returncode == 2
    assert "release is not ready" in release.stderr


def test_verify_rejects_tampered_document_and_manifest_identity(tmp_path):
    root = _repo(tmp_path)
    output = (root / "docs" / "handover-manifest.json").resolve()
    assert _build(root, output).returncode == 0

    (root / "HANDOVER.md").write_text("# Changed\n", encoding="utf-8")
    tampered_document = _run("verify", output, "--root", root)
    assert tampered_document.returncode == 2
    assert "bytes do not match" in tampered_document.stderr

    (root / "HANDOVER.md").write_text("# Handover\n", encoding="utf-8")
    manifest = json.loads(output.read_text(encoding="utf-8"))
    manifest["git"]["branch"] = "tampered"
    output.write_text(json.dumps(manifest), encoding="utf-8")
    tampered_manifest = _run("verify", output, "--root", root)
    assert tampered_manifest.returncode == 2
    assert "identity" in tampered_manifest.stderr


@pytest.mark.parametrize(
    "files,expected",
    [
        (["../outside.md"], "stay inside"),
        (["HANDOVER.md", "HANDOVER.md"], "duplicate"),
        (["docs"], "regular file"),
    ],
)
def test_build_rejects_unsafe_or_invalid_file_entries(tmp_path, files, expected):
    root = _repo(tmp_path)
    (tmp_path / "outside.md").write_text("outside\n", encoding="utf-8")
    output = (root / "docs" / "handover-manifest.json").resolve()
    command: list[object] = [
        "build",
        "--root",
        root,
        "--output",
        output,
        "--retention-decision",
        "docs/data-retention-decision.json",
    ]
    for value in files:
        command.extend(("--file", value))
    completed = _run(*command)
    assert completed.returncode == 2
    assert expected in completed.stderr
    assert not output.exists()


def test_build_rejects_absolute_self_and_invalid_ref(tmp_path):
    root = _repo(tmp_path)
    output = (root / "docs" / "handover-manifest.json").resolve()

    absolute = _run(
        "build",
        "--root",
        root,
        "--output",
        output,
        "--file",
        (root / "HANDOVER.md").resolve(),
        "--retention-decision",
        "docs/data-retention-decision.json",
    )
    assert absolute.returncode == 2
    assert "stay inside" in absolute.stderr

    self_include = _run(
        "build",
        "--root",
        root,
        "--output",
        output,
        "--file",
        "docs/handover-manifest.json",
        "--retention-decision",
        "docs/data-retention-decision.json",
    )
    assert self_include.returncode == 2
    assert "own output" in self_include.stderr

    invalid_ref = _build(root, output, "--accepted-ref", "other=not-a-hash")
    assert invalid_ref.returncode == 2
    assert "accepted-ref" in invalid_ref.stderr
    assert not output.exists()


def test_check_files_rejects_directories_and_duplicates(tmp_path):
    root = _repo(tmp_path)
    file_list = root / "docs" / "files.list"
    file_list.write_text("HANDOVER.md\ndocs/runbook.md\n", encoding="utf-8")
    valid = _run(
        "check-files",
        "--root",
        root,
        "--files-from",
        file_list,
    )
    assert valid.returncode == 0, valid.stderr
    assert json.loads(valid.stdout)["file_count"] == 2

    file_list.write_text("HANDOVER.md\nHANDOVER.md\n", encoding="utf-8")
    duplicate = _run(
        "check-files",
        "--root",
        root,
        "--files-from",
        file_list,
    )
    assert duplicate.returncode == 2
    assert "duplicate" in duplicate.stderr

    file_list.write_text("docs\n", encoding="utf-8")
    directory = _run(
        "check-files",
        "--root",
        root,
        "--files-from",
        file_list,
    )
    assert directory.returncode == 2
    assert "regular file" in directory.stderr


def test_check_files_rejects_ignored_file(tmp_path):
    root = _repo(tmp_path)
    (root / ".gitignore").write_text("ignored.md\n", encoding="utf-8")
    (root / "ignored.md").write_text("ignored\n", encoding="utf-8")
    file_list = root / "docs" / "files.list"
    file_list.write_text("ignored.md\n", encoding="utf-8")
    completed = _run(
        "check-files",
        "--root",
        root,
        "--files-from",
        file_list,
    )
    assert completed.returncode == 2
    assert "ignored by Git" in completed.stderr


def test_build_rejects_local_branch_as_published_ref(tmp_path):
    root = _repo(tmp_path)
    _write_retention(root, "approved")
    assert _git(root, "add", "docs").returncode == 0
    assert _git(root, "commit", "-m", "resolve retention").returncode == 0
    output = (root / "docs" / "handover-manifest.json").resolve()
    completed = _build(root, output, "--published-git-ref", "master")
    assert completed.returncode == 2
    assert "remote-tracking ref" in completed.stderr
    assert not output.exists()


def test_release_gate_requires_push_and_passes_from_fresh_clone(tmp_path):
    root = _repo(tmp_path)
    remote = tmp_path / "remote.git"
    assert _git(tmp_path, "init", "--bare", str(remote)).returncode == 0
    assert _git(root, "remote", "add", "origin", str(remote)).returncode == 0
    _write_retention(root, "approved")
    assert _git(root, "add", "HANDOVER.md", "docs").returncode == 0
    assert _git(root, "commit", "-m", "freeze source handover").returncode == 0
    assert _git(root, "push", "-u", "origin", "HEAD:refs/heads/wilson-current").returncode == 0

    file_list = root / "files.list"
    file_list.write_text(
        "HANDOVER.md\ndocs/runbook.md\ndocs/data-retention-decision.json\ndocs/restore-check.txt\n",
        encoding="utf-8",
    )
    output = (root / "docs" / "handover-manifest.json").resolve()
    built = _run(
        "build",
        "--root",
        root,
        "--output",
        output,
        "--files-from",
        file_list,
        "--retention-decision",
        "docs/data-retention-decision.json",
        "--published-git-ref",
        "origin/wilson-current",
    )
    assert built.returncode == 0, built.stderr
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["release"]["release_prerequisites_ready_at_build"] is True

    before_manifest_commit = _run(
        "verify",
        output,
        "--root",
        root,
        "--require-release-ready",
    )
    assert before_manifest_commit.returncode == 2
    assert "manifest is not tracked" in before_manifest_commit.stderr

    assert _git(root, "add", "docs/handover-manifest.json").returncode == 0
    assert _git(root, "commit", "-m", "publish handover manifest").returncode == 0
    assert _git(root, "push", "origin", "HEAD:refs/heads/wilson-current").returncode == 0

    clone = tmp_path / "fresh-clone"
    cloned = subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            "wilson-current",
            str(remote),
            str(clone),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert cloned.returncode == 0, cloned.stderr
    verified = _run(
        "verify",
        clone / "docs" / "handover-manifest.json",
        "--root",
        clone,
        "--require-release-ready",
    )
    assert verified.returncode == 0, verified.stderr
    payload = json.loads(verified.stdout)
    assert payload["release_ready"] is True
    assert payload["release_blockers"] == []


def test_approved_retention_requires_hashed_listed_restore_evidence(tmp_path):
    root = _repo(tmp_path)
    _write_retention(root, "approved")
    decision_path = root / "docs" / "data-retention-decision.json"
    output = (root / "docs" / "handover-manifest.json").resolve()

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["restore_check"] = None
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    missing = _build(root, output)
    assert missing.returncode == 2
    assert "restore_check" in missing.stderr

    _write_retention(root, "approved")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["restore_check"]["evidence_sha256"] = "0" * 64
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    bad_hash = _build(root, output)
    assert bad_hash.returncode == 2
    assert "evidence hash" in bad_hash.stderr

    _write_retention(root, "approved")
    unlisted = _run(
        "build",
        "--root",
        root,
        "--output",
        output,
        "--file",
        "HANDOVER.md",
        "--file",
        "docs/runbook.md",
        "--file",
        "docs/data-retention-decision.json",
        "--retention-decision",
        "docs/data-retention-decision.json",
    )
    assert unlisted.returncode == 2
    assert "restore-check evidence" in unlisted.stderr


def test_not_required_retention_can_resolve_without_restore_evidence(tmp_path):
    root = _repo(tmp_path)
    _write_retention(root, "not-required")
    output = (root / "docs" / "handover-manifest.json").resolve()
    completed = _build(root, output)
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(output.read_text(encoding="utf-8"))
    retention = manifest["release"]["retention_decision"]
    assert retention["status"] == "not-required"
    assert retention["resolved"] is True
    assert retention["restore_check"] is None
    assert manifest["release"]["release_prerequisites_ready_at_build"] is False


def test_not_required_retention_rejects_fake_restore_evidence(tmp_path):
    root = _repo(tmp_path)
    _write_retention(root, "not-required")
    decision_path = root / "docs" / "data-retention-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["restore_check"] = {
        "status": "passed",
        "checked_at": "2026-09-03T00:00:00Z",
        "checked_by": "fixture-owner",
        "evidence_path": "docs/runbook.md",
        "evidence_sha256": hashlib.sha256((root / "docs" / "runbook.md").read_bytes()).hexdigest(),
    }
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    output = (root / "docs" / "handover-manifest.json").resolve()
    completed = _build(root, output)
    assert completed.returncode == 2
    assert "must not declare restore_check" in completed.stderr


def test_build_refuses_overwrite_and_non_root(tmp_path):
    root = _repo(tmp_path)
    output = (root / "docs" / "handover-manifest.json").resolve()
    assert _build(root, output).returncode == 0
    repeated = _build(root, output)
    assert repeated.returncode == 2
    assert "refusing to overwrite" in repeated.stderr

    wrong_root_output = (root / "other.json").resolve()
    wrong_root = _run(
        "build",
        "--root",
        root / "docs",
        "--output",
        wrong_root_output,
        "--file",
        "runbook.md",
        "--retention-decision",
        "runbook.md",
    )
    assert wrong_root.returncode == 2
    assert "worktree root" in wrong_root.stderr
