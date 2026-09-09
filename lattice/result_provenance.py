from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "localized-blending-result-manifest/v1"
REQUIRED_FIELDS = {
    "schema",
    "logical_test_id",
    "configuration",
    "source_ne",
    "sink_ne",
    "available_ne",
    "attempt",
    "code",
    "inputs",
    "parameters",
    "config_sha256",
    "run_id",
    "result_directory",
    "status",
}
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: Any) -> str:
    slug = _SLUG_RE.sub("-", str(value)).strip("-._")
    if not slug:
        raise ValueError("path component must contain a safe character")
    return slug


def _normalize_ne(value: Any, available_ne: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if isinstance(available_ne, bool) or not isinstance(available_ne, int):
        raise TypeError("available_ne must be an integer")
    if available_ne < 0 or value < 0 or value > available_ne:
        raise ValueError(f"{name}={value} is outside [0, {available_ne}]")
    return value


def current_git_state(repository: str | Path) -> dict[str, Any]:
    repository = Path(repository).resolve()

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repository), *args],
            text=True,
            capture_output=True,
            check=True,
        )
        return completed.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    diff = subprocess.run(
        ["git", "-C", str(repository), "diff", "HEAD", "--binary"],
        capture_output=True,
        check=True,
    ).stdout
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    worktree = hashlib.sha256()
    worktree.update(status.encode())
    worktree.update(diff)
    for relative_path in sorted(untracked):
        path = repository / relative_path
        worktree.update(relative_path.encode())
        if path.is_file():
            worktree.update(bytes.fromhex(sha256_file(path)))
    return {
        "repository": str(repository),
        "commit": commit,
        "dirty": bool(status),
        "status_sha256": sha256_bytes(status.encode()),
        "worktree_sha256": worktree.hexdigest(),
    }


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_manifest(
    *,
    output_root: str | Path,
    logical_test_id: str,
    configuration: str | int,
    source_ne: int,
    sink_ne: int,
    available_ne: int,
    attempt: int,
    code: Mapping[str, Any],
    inputs: Mapping[str, str],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    source_ne = _normalize_ne(source_ne, available_ne, "source_ne")
    sink_ne = _normalize_ne(sink_ne, available_ne, "sink_ne")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("attempt must be a positive integer")
    for name, digest in inputs.items():
        if not isinstance(name, str) or not _HASH_RE.fullmatch(digest):
            raise ValueError("inputs must map string names to lowercase SHA-256 values")
    if not _COMMIT_RE.fullmatch(str(code.get("commit", ""))):
        raise ValueError("code.commit must be a 40- or 64-character hash")
    required_code = {
        "repository",
        "commit",
        "dirty",
        "status_sha256",
        "worktree_sha256",
    }
    if (
        not required_code <= set(code)
        or not _HASH_RE.fullmatch(str(code.get("status_sha256", "")))
        or not _HASH_RE.fullmatch(str(code.get("worktree_sha256", "")))
    ):
        raise ValueError(f"code must contain {sorted(required_code)}")

    config_payload = {
        "logical_test_id": logical_test_id,
        "configuration": str(configuration),
        "source_ne": source_ne,
        "sink_ne": sink_ne,
        "available_ne": available_ne,
        "attempt": attempt,
        "code": dict(code),
        "inputs": dict(sorted(inputs.items())),
        "parameters": parameters,
    }
    config_sha256 = sha256_bytes(canonical_json(config_payload).encode())
    run_id = config_sha256[:16]
    directory = (
        Path(output_root)
        / _slug(logical_test_id)
        / f"cfg-{_slug(configuration)}"
        / f"srcNe-{source_ne}_snkNe-{sink_ne}"
        / f"attempt-{attempt:02d}-{run_id}"
    ).resolve()
    return {
        "schema": SCHEMA,
        **config_payload,
        "config_sha256": config_sha256,
        "run_id": run_id,
        "result_directory": str(directory),
        "status": "prepared",
    }


def prepare_result_directory(manifest: Mapping[str, Any]) -> Path:
    validate_manifest(manifest)
    directory = Path(manifest["result_directory"])
    directory.mkdir(parents=True, exist_ok=False)
    atomic_write_json(directory / "manifest.json", manifest)
    return directory


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_FIELDS - set(manifest)
    if missing:
        raise ValueError(f"manifest missing fields: {sorted(missing)}")
    if manifest["schema"] != SCHEMA:
        raise ValueError(f"unsupported manifest schema: {manifest['schema']!r}")
    source_ne = _normalize_ne(
        manifest["source_ne"], manifest["available_ne"], "source_ne"
    )
    sink_ne = _normalize_ne(
        manifest["sink_ne"], manifest["available_ne"], "sink_ne"
    )
    if source_ne != manifest["source_ne"] or sink_ne != manifest["sink_ne"]:
        raise ValueError("Ne values are not canonical")
    expected = build_manifest(
        output_root=Path(manifest["result_directory"]).parents[3],
        logical_test_id=manifest["logical_test_id"],
        configuration=manifest["configuration"],
        source_ne=source_ne,
        sink_ne=sink_ne,
        available_ne=manifest["available_ne"],
        attempt=manifest["attempt"],
        code=manifest["code"],
        inputs=manifest["inputs"],
        parameters=manifest["parameters"],
    )
    for field in ("config_sha256", "run_id", "result_directory"):
        if manifest[field] != expected[field]:
            raise ValueError(f"manifest {field} does not match its content")
    return dict(manifest)


def load_result_manifest(
    result_directory: str | Path, *, allow_legacy: bool = False
) -> dict[str, Any]:
    result_directory = Path(result_directory).resolve()
    manifest_path = result_directory / "manifest.json"
    if not manifest_path.is_file():
        if not allow_legacy:
            raise ValueError("legacy result has no manifest and is not verified")
        return {
            "schema": "localized-blending-result-manifest/legacy",
            "result_directory": str(result_directory),
            "status": "legacy-unverified",
            "verified": False,
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validated = validate_manifest(manifest)
    if Path(validated["result_directory"]) != result_directory:
        raise ValueError("manifest belongs to a different result directory")
    validated["verified"] = validated["status"] in {"completed", "validated"}
    outputs = validated.get("outputs", {})
    if validated["verified"]:
        if not outputs:
            raise ValueError("completed manifest has no output hashes")
        for relative_path, expected_hash in outputs.items():
            if not _HASH_RE.fullmatch(expected_hash):
                raise ValueError("output hashes must be lowercase SHA-256 values")
            output_path = result_directory / relative_path
            if not output_path.is_file() or sha256_file(output_path) != expected_hash:
                raise ValueError(f"output does not match manifest: {relative_path}")
    return validated


def finalize_result(
    manifest: Mapping[str, Any], outputs: Mapping[str, str | Path]
) -> dict[str, Any]:
    validated = validate_manifest(manifest)
    directory = Path(validated["result_directory"])
    if not (directory / "manifest.json").is_file():
        raise ValueError("result directory has not been prepared")
    output_hashes = {}
    for name, path in outputs.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("output names must be relative to the result directory")
        path = Path(path).resolve()
        expected_path = (directory / name).resolve()
        if path != expected_path or not path.is_file():
            raise ValueError(f"output is outside result directory: {name}")
        output_hashes[name] = sha256_file(path)
    completed = {
        **validated,
        "status": "completed",
        "outputs": dict(sorted(output_hashes.items())),
    }
    atomic_write_json(directory / "manifest.json", completed)
    return completed
