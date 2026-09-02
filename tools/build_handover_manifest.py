#!/usr/bin/env python3
"""Build or verify a content-addressed handover-document manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from typing import Any

SCHEMA = "lattice.current.handover-manifest/v1"
RETENTION_SCHEMA = "lattice.current.data-retention-decision/v1"
RESOLVED_RETENTION = {"approved", "not-required"}


class ManifestError(ValueError):
    """A handover manifest request or artifact is invalid."""


def _fail(message: str) -> None:
    raise ManifestError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _identity(value: dict[str, Any]) -> str:
    semantic = {key: item for key, item in value.items() if key != "manifest_identity"}
    return hashlib.sha256(_canonical(semantic)).hexdigest()


def _git(root: Path, *arguments: str, required: bool = True) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        if required:
            _fail(f"git {' '.join(arguments)} failed: {(completed.stderr or completed.stdout).strip()}")
        return None
    return completed.stdout.strip()


def _git_commit(root: Path, ref: str, name: str) -> str:
    if not isinstance(ref, str) or not ref:
        _fail(f"{name} must be a non-empty Git ref")
    commit = _git(root, "rev-parse", "--verify", f"{ref}^{{commit}}")
    if not isinstance(commit, str) or len(commit) != 40:
        _fail(f"{name} does not resolve to a Git commit: {ref}")
    return commit.lower()


def _is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", ancestor, descendant],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode not in {0, 1}:
        _fail(f"git merge-base --is-ancestor failed: {(completed.stderr or completed.stdout).strip()}")
    return completed.returncode == 0


def _root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if not root.is_dir():
        _fail(f"project root is not a directory: {root}")
    top = _git(root, "rev-parse", "--show-toplevel")
    if Path(top).resolve() != root:
        _fail("--root must be the Git worktree root")
    return root


def _relative_file(
    root: Path,
    value: str,
    *,
    output: Path | None = None,
) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        _fail("manifest file entries must be non-empty relative paths")
    logical = Path(value)
    if logical.is_absolute() or ".." in logical.parts:
        _fail(f"manifest file entry must stay inside the project root: {value}")
    path = (root / logical).resolve()
    try:
        normalized = path.relative_to(root).as_posix()
    except ValueError:
        _fail(f"manifest file entry escapes the project root: {value}")
    if output is not None and path == output:
        _fail("handover manifest cannot include its own output file")
    if not path.is_file():
        _fail(f"manifest file entry is not a regular file: {value}")
    return normalized, path


def _git_state(root: Path, relative: str) -> str:
    output = _git(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        relative,
    )
    if not output:
        return "clean"
    lines = output.splitlines()
    if len(lines) != 1:
        _fail(f"unexpected Git status for manifest file: {relative}")
    return lines[0][:2]


def _is_ignored(root: Path, relative: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "--quiet", "--", relative],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode not in {0, 1}:
        _fail(f"git check-ignore failed: {(completed.stderr or completed.stdout).strip()}")
    return completed.returncode == 0


def _files_from(path: Path) -> list[str]:
    source = path.expanduser().resolve()
    if not source.is_file():
        _fail(f"--files-from is not a regular file: {source}")
    values = source.read_text(encoding="utf-8").splitlines()
    if not values or any(not value or value != value.strip() for value in values):
        _fail("--files-from must contain one non-empty unpadded path per line")
    return values


def _requested_files(args: argparse.Namespace) -> list[str]:
    values = list(args.files)
    if args.files_from is not None:
        values.extend(_files_from(args.files_from))
    if not values:
        _fail("at least one --file or --files-from entry is required")
    return values


def _accepted_refs(
    root: Path,
    values: list[str],
    git_refs: list[str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, commit = value.partition("=")
        if (
            not separator
            or not name
            or not commit
            or name in result
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit.lower())
        ):
            _fail("--accepted-ref must be a unique NAME=40-hex identity")
        result[name] = commit.lower()
    for ref in git_refs:
        if ref in result:
            _fail(f"duplicate accepted ref name: {ref}")
        result[ref] = _git_commit(root, ref, "accepted Git ref")
    return dict(sorted(result.items()))


def _retention_decision(root: Path, value: Path) -> dict[str, Any]:
    logical, path = _relative_file(root, str(value))
    try:
        decision = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read retention decision: {exc}") from exc
    required = {
        "schema",
        "version",
        "status",
        "authority",
        "decided_at",
        "secondary_copy",
        "retention_period",
        "restore_owner",
        "restore_check",
        "deletion_policy",
        "rationale",
    }
    if not isinstance(decision, dict) or set(decision) != required:
        _fail("retention decision has missing or unknown fields")
    if decision["schema"] != RETENTION_SCHEMA or decision["version"] != 1:
        _fail("retention decision schema/version is unsupported")
    status = decision["status"]
    if status not in {"pending", *RESOLVED_RETENTION}:
        _fail("retention decision status is unsupported")
    if not isinstance(decision["deletion_policy"], str) or not decision["deletion_policy"]:
        _fail("retention decision requires a deletion policy")
    if not isinstance(decision["rationale"], str) or not decision["rationale"]:
        _fail("retention decision requires a rationale")

    restore_metadata = None
    if status in RESOLVED_RETENTION:
        for field in ("authority", "decided_at", "retention_period", "restore_owner"):
            if not isinstance(decision[field], str) or not decision[field]:
                _fail(f"resolved retention decision requires {field}")
        if status == "approved":
            if not isinstance(decision["secondary_copy"], str) or not decision["secondary_copy"]:
                _fail("approved retention decision requires secondary_copy")
            restore_check = decision["restore_check"]
            if not isinstance(restore_check, dict) or set(restore_check) != {
                "status",
                "checked_at",
                "checked_by",
                "evidence_path",
                "evidence_sha256",
            }:
                _fail("approved retention decision requires restore_check evidence")
            if restore_check["status"] != "passed":
                _fail("approved retention restore_check status must be passed")
            for field in ("checked_at", "checked_by"):
                if not isinstance(restore_check[field], str) or not restore_check[field]:
                    _fail(f"approved retention restore_check requires {field}")
            evidence_relative, evidence_path = _relative_file(
                root,
                restore_check["evidence_path"],
            )
            evidence_sha256 = restore_check["evidence_sha256"]
            if (
                not isinstance(evidence_sha256, str)
                or len(evidence_sha256) != 64
                or any(character not in "0123456789abcdef" for character in evidence_sha256.lower())
            ):
                _fail("approved retention restore_check evidence_sha256 is invalid")
            evidence_sha256 = evidence_sha256.lower()
            if _sha256(evidence_path) != evidence_sha256:
                _fail("approved retention restore_check evidence hash does not match")
            restore_metadata = {
                "status": "passed",
                "checked_at": restore_check["checked_at"],
                "checked_by": restore_check["checked_by"],
                "evidence_path": evidence_relative,
                "evidence_sha256": evidence_sha256,
            }
        else:
            if decision["secondary_copy"] is not None:
                _fail("not-required retention decision must not declare secondary_copy")
            if decision["restore_check"] is not None:
                _fail("not-required retention decision must not declare restore_check")
    elif any(
        decision[field] is not None
        for field in (
            "authority",
            "decided_at",
            "secondary_copy",
            "retention_period",
            "restore_owner",
            "restore_check",
        )
    ):
        _fail("pending retention decision must leave undecided fields null")
    return {
        "path": logical,
        "sha256": _sha256(path),
        "status": status,
        "resolved": status in RESOLVED_RETENTION,
        "restore_check": restore_metadata,
    }


def _published_ref(root: Path, ref: str | None, head: str) -> dict[str, Any] | None:
    if ref is None:
        return None
    if not isinstance(ref, str) or "/" not in ref:
        _fail("published Git ref must name a remote-tracking ref such as origin/branch")
    full_ref = f"refs/remotes/{ref}"
    commit = _git_commit(root, full_ref, "published remote-tracking Git ref")
    return {
        "name": ref,
        "observed_commit": commit,
        "contains_recorded_head": _is_ancestor(root, head, commit),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = _root(args.root)
    output = args.output.expanduser()
    if not output.is_absolute():
        _fail("--output must be an absolute path")
    output = output.resolve()
    if output.exists():
        _fail(f"refusing to overwrite handover manifest: {output}")
    try:
        output.relative_to(root)
    except ValueError:
        _fail("--output must be inside the project root")

    records = []
    seen = set()
    for value in _requested_files(args):
        relative, path = _relative_file(root, value, output=output)
        if _is_ignored(root, relative):
            _fail(f"manifest file entry is ignored by Git: {relative}")
        if relative in seen:
            _fail(f"duplicate manifest file entry: {relative}")
        seen.add(relative)
        records.append(
            {
                "path": relative,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "git_state": _git_state(root, relative),
            }
        )
    records.sort(key=lambda record: record["path"])

    retention = _retention_decision(root, args.retention_decision)
    if retention["path"] not in seen:
        _fail("retention decision must also be listed among manifest files")
    for record in records:
        if record["path"] == retention["path"] and record["sha256"] != retention["sha256"]:
            _fail("retention decision hash changed during manifest build")
    restore_check = retention["restore_check"]
    if restore_check is not None:
        if restore_check["evidence_path"] not in seen:
            _fail("restore-check evidence must also be listed among manifest files")
        for record in records:
            if (
                record["path"] == restore_check["evidence_path"]
                and record["sha256"] != restore_check["evidence_sha256"]
            ):
                _fail("restore-check evidence hash changed during manifest build")

    branch = _git(root, "branch", "--show-current") or None
    origin = _git(root, "remote", "get-url", "origin", required=False)
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    head = _git(root, "rev-parse", "HEAD")
    documents_clean = all(record["git_state"] == "clean" for record in records)
    publication = _published_ref(root, args.published_git_ref, head)
    release_ready_at_build = bool(
        documents_clean and retention["resolved"] and publication is not None and publication["contains_recorded_head"]
    )
    manifest = {
        "schema": SCHEMA,
        "version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_root_at_generation": root.as_posix(),
        "git": {
            "head": head,
            "branch": branch,
            "dirty": bool(status),
            "origin_url": origin,
            "accepted_external_refs": _accepted_refs(
                root,
                args.accepted_ref,
                args.accepted_git_ref,
            ),
        },
        "files": records,
        "release": {
            "source_files_git_clean_at_build": documents_clean,
            "retention_decision": retention,
            "published_git_ref": publication,
            "release_prerequisites_ready_at_build": release_ready_at_build,
        },
        "verification": {
            "operation": "SHA-256 over exact file bytes",
            "documents_command": (
                "python tools/build_handover_manifest.py verify docs/handover-manifest.json --root ."
            ),
            "release_command": (
                "python tools/build_handover_manifest.py verify "
                "docs/handover-manifest.json --root . --require-release-ready"
            ),
        },
    }
    manifest["manifest_identity"] = _identity(manifest)
    encoded = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest


def _manifest_path_in_root(root: Path, path: Path) -> str | None:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def _current_release_status(
    root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons = []
    for record in manifest["files"]:
        if _git_state(root, record["path"]) != "clean":
            reasons.append(f"handover file is not Git-clean: {record['path']}")
    manifest_relative = _manifest_path_in_root(root, manifest_path)
    if manifest_relative is None or _git_state(root, manifest_relative) != "clean":
        reasons.append("handover manifest is not tracked and Git-clean")

    release = manifest["release"]
    retention = release["retention_decision"]
    if not retention["resolved"]:
        reasons.append("large-data retention decision is unresolved")
    publication = release["published_git_ref"]
    if publication is None:
        reasons.append("published Git ref is not recorded")
    else:
        observed = _git_commit(
            root,
            f"refs/remotes/{publication['name']}",
            "published remote-tracking Git ref",
        )
        if not _is_ancestor(root, manifest["git"]["head"], observed):
            reasons.append("published Git ref no longer contains recorded source head")
        if manifest_relative is not None:
            manifest_commit = _git(
                root,
                "log",
                "-1",
                "--format=%H",
                "--",
                manifest_relative,
                required=False,
            )
            if not manifest_commit or not _is_ancestor(root, manifest_commit, observed):
                reasons.append("published Git ref does not contain the manifest commit")
    return not reasons, reasons


def verify(args: argparse.Namespace) -> dict[str, Any]:
    root = _root(args.root)
    manifest_path = args.manifest.expanduser().resolve()
    if not manifest_path.is_file():
        _fail(f"handover manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read handover manifest: {exc}") from exc
    required = {
        "schema",
        "version",
        "generated_utc",
        "project_root_at_generation",
        "git",
        "files",
        "release",
        "verification",
        "manifest_identity",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        _fail("handover manifest has missing or unknown fields")
    if manifest["schema"] != SCHEMA or manifest["version"] != 1:
        _fail("handover manifest schema/version is unsupported")
    if manifest["manifest_identity"] != _identity(manifest):
        _fail("handover manifest identity is stale or tampered")
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        _fail("handover manifest files must be a non-empty list")
    seen = set()
    verified = []
    for index, record in enumerate(files):
        if not isinstance(record, dict) or set(record) != {
            "path",
            "sha256",
            "bytes",
            "git_state",
        }:
            _fail(f"handover manifest file record {index} is invalid")
        relative, path = _relative_file(root, record["path"])
        if relative in seen:
            _fail(f"duplicate handover manifest path: {relative}")
        seen.add(relative)
        if (
            not isinstance(record["sha256"], str)
            or len(record["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in record["sha256"])
        ):
            _fail(f"invalid SHA-256 for handover file: {relative}")
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            _fail(f"handover file bytes do not match manifest: {relative}")
        verified.append(relative)

    release = manifest["release"]
    if not isinstance(release, dict) or set(release) != {
        "source_files_git_clean_at_build",
        "retention_decision",
        "published_git_ref",
        "release_prerequisites_ready_at_build",
    }:
        _fail("handover manifest release metadata is invalid")
    retention = release["retention_decision"]
    if not isinstance(retention, dict) or set(retention) != {
        "path",
        "sha256",
        "status",
        "resolved",
        "restore_check",
    }:
        _fail("handover manifest retention metadata is invalid")
    current_retention = _retention_decision(root, Path(retention["path"]))
    if current_retention != retention:
        _fail("retention decision metadata does not match handover manifest")
    restore_check = retention["restore_check"]
    if restore_check is not None:
        if restore_check["evidence_path"] not in seen:
            _fail("handover manifest does not include restore-check evidence")
        evidence_record = next(record for record in files if record["path"] == restore_check["evidence_path"])
        if evidence_record["sha256"] != restore_check["evidence_sha256"]:
            _fail("restore-check evidence hash does not match handover manifest")
    if not isinstance(release["source_files_git_clean_at_build"], bool) or not isinstance(
        release["release_prerequisites_ready_at_build"], bool
    ):
        _fail("handover manifest release readiness flags are invalid")
    publication = release["published_git_ref"]
    if publication is not None:
        if not isinstance(publication, dict) or set(publication) != {
            "name",
            "observed_commit",
            "contains_recorded_head",
        }:
            _fail("handover manifest published Git ref metadata is invalid")
        if (
            not isinstance(publication["name"], str)
            or not isinstance(publication["observed_commit"], str)
            or not isinstance(publication["contains_recorded_head"], bool)
        ):
            _fail("handover manifest published Git ref values are invalid")
    expected_build_ready = bool(
        release["source_files_git_clean_at_build"]
        and retention["resolved"]
        and publication is not None
        and publication["contains_recorded_head"]
    )
    if release["release_prerequisites_ready_at_build"] is not expected_build_ready:
        _fail("handover manifest build-time release metadata is inconsistent")

    release_ready, release_reasons = _current_release_status(
        root,
        manifest_path,
        manifest,
    )
    if args.require_release_ready and not release_ready:
        _fail("handover release is not ready: " + "; ".join(release_reasons))
    return {
        "schema": SCHEMA,
        "status": "documents-verified",
        "manifest": manifest_path.as_posix(),
        "manifest_identity": manifest["manifest_identity"],
        "file_count": len(verified),
        "current_git_head": _git(root, "rev-parse", "HEAD"),
        "recorded_git_head": manifest["git"].get("head"),
        "release_ready": release_ready,
        "release_blockers": release_reasons,
    }


def check_files(args: argparse.Namespace) -> dict[str, Any]:
    root = _root(args.root)
    values = _files_from(args.files_from)
    verified = []
    seen = set()
    for value in values:
        relative, _path = _relative_file(root, value)
        if _is_ignored(root, relative):
            _fail(f"file-list entry is ignored by Git: {relative}")
        if relative in seen:
            _fail(f"duplicate file-list entry: {relative}")
        seen.add(relative)
        verified.append(relative)
    return {
        "status": "file-list-valid",
        "root": root.as_posix(),
        "file_count": len(verified),
        "files": verified,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--root", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--file", dest="files", action="append", default=[])
    build_parser.add_argument("--files-from", type=Path)
    build_parser.add_argument("--retention-decision", type=Path, required=True)
    build_parser.add_argument("--accepted-ref", action="append", default=[])
    build_parser.add_argument("--accepted-git-ref", action="append", default=[])
    build_parser.add_argument("--published-git-ref")
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("manifest", type=Path)
    verify_parser.add_argument("--root", type=Path, required=True)
    verify_parser.add_argument("--require-release-ready", action="store_true")
    check_parser = commands.add_parser("check-files")
    check_parser.add_argument("--root", type=Path, required=True)
    check_parser.add_argument("--files-from", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "build":
            result = build(args)
        elif args.command == "verify":
            result = verify(args)
        else:
            result = check_files(args)
    except (ManifestError, OSError) as exc:
        print(f"handover manifest error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
