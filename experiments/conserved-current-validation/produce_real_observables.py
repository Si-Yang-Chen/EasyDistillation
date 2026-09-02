#!/usr/bin/env python3
"""Package audited conserved-current contractions; never synthesize WT physics.

Raw VSV/PSV/overlap/Current elementals establish lineage but do not by themselves
specify the hadron two- and three-point contractions needed for WT sides or Q(t).
Accordingly, the only production mode is ``audited-contractions-v1``: it validates
those raw inputs and copies finite, externally contracted observables from an
independently audited NPZ.  Any other requested mode fails with the explicit
``unsupported-observable-input`` contract.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np

SCHEMA = "lattice.conserved-current-validation.manifest/v1"
CONTRACTION_SCHEMA = "lattice.conserved-current-validation.audited-contractions/v1"
ARTIFACT_NAMES = ("gauge", "eigenvector", "vsv", "psv", "overlap", "current_p2v", "current_p2p")
HASH_FIELDS = {
    "vsv": "vsv_perambulator_sha256",
    "psv": "psv_sha256",
    "overlap": "overlap_sha256",
    "current_p2v": "current_p2v_sha256",
    "current_p2p": "current_p2p_sha256",
}


class ProducerError(ValueError):
    """A manifest or audited input does not meet the producer contract."""


def _fail(message: str) -> None:
    raise ProducerError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _require_hash(value: Any, name: str) -> str:
    if not _is_hash(value):
        _fail(f"{name} must be a 64-hex SHA-256")
    return str(value).lower()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{name} must be an object")
    return value


def _absolute_existing(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        _fail(f"{name} must be an absolute path string")
    path = Path(value)
    if not path.is_absolute():
        _fail(f"{name} must be absolute")
    if not path.exists() or not path.is_file():
        _fail(f"{name} does not exist as a regular file: {path}")
    return path


def _artifact_paths(spec: dict[str, Any], name: str) -> list[Path]:
    has_path = "path" in spec
    has_pattern = "pattern" in spec
    if has_path == has_pattern:
        _fail(f"artifacts.{name} needs exactly one of path or pattern")
    if has_path:
        return [_absolute_existing(spec["path"], f"artifacts.{name}.path")]
    pattern = spec["pattern"]
    if not isinstance(pattern, str) or not Path(pattern).is_absolute():
        _fail(f"artifacts.{name}.pattern must be absolute")
    matches = [Path(item) for item in sorted(glob.glob(pattern))]
    if not matches or any(not item.is_file() for item in matches):
        _fail(f"artifacts.{name}.pattern must match one or more regular files")
    return matches


def _artifact_hashes(spec: dict[str, Any], name: str, paths: list[Path]) -> dict[str, str]:
    expected = spec.get("sha256")
    if len(paths) == 1 and isinstance(expected, str):
        wanted = {_require_hash(expected, f"artifacts.{name}.sha256")}
        actual = _sha256(paths[0])
        if actual not in wanted:
            _fail(f"artifacts.{name} SHA-256 does not match manifest")
        return {str(paths[0]): actual}
    if not isinstance(expected, dict):
        _fail(f"artifacts.{name}.sha256 must map every expanded pattern path to its SHA-256")
    verified: dict[str, str] = {}
    for path in paths:
        wanted = _require_hash(expected.get(str(path)), f"artifacts.{name}.sha256[{path}]")
        actual = _sha256(path)
        if actual != wanted:
            _fail(f"artifacts.{name} SHA-256 does not match manifest for {path}")
        verified[str(path)] = actual
    if set(expected) != set(verified):
        _fail(f"artifacts.{name}.sha256 has paths not selected by its pattern")
    return verified


def _validate_loader(spec: dict[str, Any], name: str, paths: list[Path], load_data: bool) -> None:
    loader = _mapping(spec.get("loader"), f"artifacts.{name}.loader")
    fmt = loader.get("format")
    if fmt == "opaque":
        if set(loader) != {"format"}:
            _fail(f"artifacts.{name}.loader opaque format cannot declare array fields")
        return
    if fmt not in {"npy", "npz"}:
        _fail(f"artifacts.{name}.loader.format must be npy, npz, or opaque")
    ndim = loader.get("ndim")
    kind = loader.get("dtype_kind")
    if not isinstance(ndim, int) or isinstance(ndim, bool) or ndim < 1:
        _fail(f"artifacts.{name}.loader.ndim must be a positive integer")
    if kind not in {"b", "i", "u", "f", "c"}:
        _fail(f"artifacts.{name}.loader.dtype_kind must be a NumPy dtype kind")
    if not isinstance(loader.get("finite"), bool):
        _fail(f"artifacts.{name}.loader.finite must be boolean")
    if fmt == "npz" and not isinstance(loader.get("member"), str):
        _fail(f"artifacts.{name}.loader.member is required for npz")
    if not load_data:
        return
    for path in paths:
        try:
            if fmt == "npy":
                array = np.load(path, allow_pickle=False, mmap_mode="r")
            else:
                with np.load(path, allow_pickle=False) as archive:
                    member = loader["member"]
                    if member not in archive.files:
                        _fail(f"artifacts.{name} NPZ lacks loader member {member!r}")
                    array = np.asarray(archive[member])
        except (OSError, ValueError) as exc:
            raise ProducerError(f"artifacts.{name} loader cannot read {path}: {exc}") from exc
        if array.ndim != ndim:
            _fail(f"artifacts.{name} loader rank {array.ndim} does not equal declared {ndim}")
        if array.dtype.kind != kind:
            _fail(f"artifacts.{name} loader dtype kind {array.dtype.kind!r} does not equal declared {kind!r}")
        if loader["finite"] and (array.dtype.kind not in "fci" or not np.all(np.isfinite(array))):
            _fail(f"artifacts.{name} loader requires finite numeric values")


def _read_cfg_ids(path: Path) -> list[str]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        decoded = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if isinstance(decoded, dict):
        decoded = decoded.get("cfg_ids")
    if not isinstance(decoded, list) or not decoded or any(not isinstance(item, (str, int)) for item in decoded):
        _fail("configuration_list must contain a nonempty JSON/list or one configuration ID per line")
    ids = [str(item) for item in decoded]
    if len(ids) != len(set(ids)):
        _fail("configuration_list has duplicate configuration IDs")
    return ids


def _ne(value: Any, name: str) -> dict[str, int]:
    item = _mapping(value, name)
    result: dict[str, int] = {}
    for side in ("source", "sink"):
        side_value = _mapping(item.get(side), f"{name}.{side}")
        used, available = side_value.get("used"), side_value.get("available")
        if (
            any(isinstance(number, bool) or not isinstance(number, int) for number in (used, available))
            or used < 1
            or available < used
        ):
            _fail(f"{name}.{side} needs integers 1 <= used <= available")
        result[f"{side}_used"] = used
        result[f"{side}_available"] = available
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProducerError(f"cannot read manifest JSON: {exc}") from exc
    return _mapping(value, "manifest root")


def validate_manifest(manifest: dict[str, Any], *, load_arrays: bool) -> dict[str, Any]:
    if manifest.get("schema") != SCHEMA:
        _fail("unsupported manifest schema")
    if manifest.get("synthetic") is not False and manifest.get("synthetic") is not True:
        _fail("manifest.synthetic must be boolean")
    producer = _mapping(manifest.get("producer"), "producer")
    if producer.get("mode") != "audited-contractions-v1":
        _fail(
            "unsupported-observable-input: raw VSV/PSV/overlap/P2V inputs do not define WT RHS "
            "or charge; use audited-contractions-v1"
        )
    if producer.get("source_sink_current_time_semantics") != "source-first":
        _fail("producer.source_sink_current_time_semantics must be source-first")
    if manifest.get("boundary") not in {"periodic", "open"}:
        _fail("manifest.boundary must be periodic or open")
    extent = manifest.get("temporal_extent")
    if isinstance(extent, bool) or not isinstance(extent, int) or extent < 2:
        _fail("manifest.temporal_extent must be an integer >= 2")
    for field in ("contact_times", "excluded_boundary_times", "plateau_times"):
        values = manifest.get(field)
        if (
            not isinstance(values, list)
            or (field != "excluded_boundary_times" and not values)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= extent
                for value in values
            )
        ):
            _fail(f"manifest.{field} must contain in-range integer time indices")
    if manifest["boundary"] == "open" and not manifest["excluded_boundary_times"]:
        _fail("open boundary requires explicit excluded_boundary_times")
    _ne(manifest.get("ne"), "ne")
    expected = _mapping(manifest.get("expected_provenance"), "expected_provenance")
    for field, source_field in (("code_sha256", "code_source_path"), ("api_sha256", "api_source_path")):
        declared = _require_hash(expected.get(field), f"expected_provenance.{field}")
        source_path = _absolute_existing(expected.get(source_field), f"expected_provenance.{source_field}")
        if _sha256(source_path) != declared:
            _fail(f"expected_provenance.{field} does not match {source_field}")
    current_api = _mapping(manifest.get("current_api"), "current_api")
    if (current_api.get("version"), current_api.get("term_schema"), current_api.get("assembler_schema")) != (
        "1.2.0",
        "lattice.current.term/v1",
        "lattice.current.assembler/v1",
    ):
        _fail("manifest must declare frozen Current API 1.2.0 and v1 schemas")
    source_files = manifest.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        _fail("source_files must be a nonempty list of hashed absolute source files")
    for index, source in enumerate(source_files):
        source = _mapping(source, f"source_files[{index}]")
        source_path = _absolute_existing(source.get("path"), f"source_files[{index}].path")
        if _sha256(source_path) != _require_hash(source.get("sha256"), f"source_files[{index}].sha256"):
            _fail(f"source_files[{index}] SHA-256 does not match")
    cfg_spec = _mapping(manifest.get("configuration_list"), "configuration_list")
    cfg_path = _absolute_existing(cfg_spec.get("path"), "configuration_list.path")
    if _sha256(cfg_path) != _require_hash(cfg_spec.get("sha256"), "configuration_list.sha256"):
        _fail("configuration_list SHA-256 does not match")
    cfg_ids = _read_cfg_ids(cfg_path)
    artifacts = _mapping(manifest.get("artifacts"), "artifacts")
    verified: dict[str, dict[str, str]] = {}
    for name in ARTIFACT_NAMES:
        spec = _mapping(artifacts.get(name), f"artifacts.{name}")
        if name == "current_p2p" and spec.get("status") == "unused":
            if (
                producer.get("p2p_required") is not False
                or not isinstance(spec.get("unused_reason"), str)
                or not spec["unused_reason"]
            ):
                _fail("current_p2p can be unused only with producer.p2p_required=false and an explicit unused_reason")
            attestation = _absolute_existing(spec.get("attestation_path"), "artifacts.current_p2p.attestation_path")
            if _sha256(attestation) != _require_hash(
                spec.get("attestation_sha256"), "artifacts.current_p2p.attestation_sha256"
            ):
                _fail("current_p2p unused attestation SHA-256 does not match")
            verified[name] = {str(attestation): _sha256(attestation)}
            continue
        if spec.get("status", "required") != "required":
            _fail(f"artifacts.{name}.status must be required")
        paths = _artifact_paths(spec, name)
        verified[name] = _artifact_hashes(spec, name, paths)
        _validate_loader(spec, name, paths, load_arrays)
    if producer.get("p2p_required") is True and artifacts["current_p2p"].get("status") == "unused":
        _fail("selected observable path requires Current P2P but manifest marks it unused")
    contract_path = _absolute_existing(producer.get("audited_contractions_path"), "producer.audited_contractions_path")
    if _sha256(contract_path) != _require_hash(
        producer.get("audited_contractions_sha256"), "producer.audited_contractions_sha256"
    ):
        _fail("audited contraction input SHA-256 does not match manifest")
    return {"cfg_ids": cfg_ids, "artifacts": verified, "contract_path": contract_path, "ne": _ne(manifest["ne"], "ne")}


def _scalar_json(value: np.ndarray, name: str) -> dict[str, Any]:
    if value.shape not in {(), (1,)}:
        _fail(f"{name} must be a scalar or one-element Unicode array")
    try:
        return _mapping(json.loads(str(value.item() if value.shape == () else value[0])), name)
    except json.JSONDecodeError as exc:
        raise ProducerError(f"{name} is not JSON: {exc}") from exc


def _input_array(archive: Any, alternatives: tuple[str, ...], label: str) -> np.ndarray:
    names = [name for name in alternatives if name in archive.files]
    if len(names) != 1:
        _fail(f"audited contraction input requires exactly one of {alternatives} for {label}")
    return np.asarray(archive[names[0]])


def load_audited_contractions(
    manifest: dict[str, Any], checked: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        with np.load(checked["contract_path"], allow_pickle=False) as archive:
            cfg_ids = _input_array(archive, ("cfg_ids",), "configuration IDs")
            lhs = _input_array(archive, ("wt_lhs", "lhs"), "WT left side")
            rhs = _input_array(archive, ("wt_rhs", "rhs"), "WT right side")
            charge = _input_array(archive, ("charge_ratio", "charge"), "charge ratio")
            contact = np.asarray(archive["contact_term"]) if "contact_term" in archive.files else None
            if "provenance_json" not in archive.files:
                _fail("audited contraction input lacks provenance_json")
            provenance = _scalar_json(np.asarray(archive["provenance_json"]), "audited contractions provenance_json")
    except OSError as exc:
        raise ProducerError(f"cannot load audited contraction input: {exc}") from exc
    if cfg_ids.ndim != 1 or cfg_ids.dtype.kind not in "iuUS" or [str(value) for value in cfg_ids] != checked["cfg_ids"]:
        _fail("audited contraction cfg_ids must exactly equal the hashed configuration_list in order")
    extent = manifest["temporal_extent"]
    for name, array in (("wt_lhs", lhs), ("wt_rhs", rhs), ("charge_ratio", charge)):
        if (
            array.ndim != 2
            or array.shape != (cfg_ids.size, extent)
            or array.dtype.kind not in "fiu"
            or not np.all(np.isfinite(array))
        ):
            _fail(f"audited contraction {name} must be a finite real (Ncfg, temporal_extent) array")
    if contact is not None and (
        contact.ndim != 2
        or contact.shape != lhs.shape
        or contact.dtype.kind not in "fiu"
        or not np.all(np.isfinite(contact))
    ):
        _fail("audited contraction contact_term must be finite real (Ncfg, temporal_extent)")
    if provenance.get("schema") != CONTRACTION_SCHEMA:
        _fail("unsupported-observable-input: audited contraction provenance schema is missing or unsupported")
    if provenance.get("source_sink_current_time_semantics") != "source-first":
        _fail("audited contraction provenance must declare source-first time semantics")
    if provenance.get("boundary") != manifest["boundary"] or provenance.get("temporal_extent") != extent:
        _fail("audited contraction boundary or temporal extent does not match manifest")
    if _ne(provenance.get("ne"), "audited contraction provenance.ne") != checked["ne"]:
        _fail("Ne mismatch between manifest and audited contraction input")
    lineage = _mapping(provenance.get("input_artifacts"), "audited contraction provenance.input_artifacts")
    for name in ARTIFACT_NAMES:
        actual = _lineage_hash(checked["artifacts"][name])
        if _require_hash(lineage.get(name), f"audited contraction provenance.input_artifacts.{name}") != actual:
            _fail(f"audited contraction input artifact hash disagrees for {name}")
    arrays = {"cfg_ids": cfg_ids, "lhs": lhs.astype(float), "rhs": rhs.astype(float), "charge": charge.astype(float)}
    if contact is not None:
        arrays["contact_term"] = contact.astype(float)
    return arrays, provenance


def _git_state() -> dict[str, Any]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
        dirty = bool(
            subprocess.run(["git", "status", "--porcelain"], text=True, capture_output=True, check=True).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "dirty": None}
    return {"head": head, "dirty": dirty}


def _lineage_hash(files: dict[str, str]) -> str:
    """Hash one artifact or a canonical expanded-pattern file/hash mapping."""
    if len(files) == 1:
        return next(iter(files.values()))
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def produce(manifest: dict[str, Any], result_dir: Path, *, synthetic_fixture: bool) -> dict[str, Any]:
    if result_dir.exists():
        _fail(f"refusing to reuse existing output directory: {result_dir}")
    if not result_dir.is_absolute():
        _fail("result directory must be absolute")
    if synthetic_fixture != bool(manifest["synthetic"]):
        _fail("--synthetic-fixture is allowed only with manifest.synthetic=true; real manifests must not use it")
    checked = validate_manifest(manifest, load_arrays=True)
    arrays, contraction_provenance = load_audited_contractions(manifest, checked)
    artifact_hashes = {HASH_FIELDS[name]: _lineage_hash(checked["artifacts"][name]) for name in HASH_FIELDS}
    provenance = {
        "synthetic": bool(manifest["synthetic"]),
        "code_sha256": manifest["expected_provenance"]["code_sha256"],
        "api_sha256": manifest["expected_provenance"]["api_sha256"],
        "api_version": manifest["current_api"]["version"],
        "term_schema": manifest["current_api"]["term_schema"],
        "assembler_schema": manifest["current_api"]["assembler_schema"],
        "boundary": manifest["boundary"],
        "temporal_extent": manifest["temporal_extent"],
        "source_sink_current_time_semantics": "source-first",
        "propagator_artifacts": artifact_hashes,
        "producer_mode": "audited-contractions-v1",
        "ne": checked["ne"],
        "p2p_status": manifest["artifacts"]["current_p2p"].get("status", "required"),
        "audited_contraction_sha256": _sha256(checked["contract_path"]),
        "audited_contraction_provenance": contraction_provenance,
        "time_validity": {
            "contact_times": manifest["contact_times"],
            "excluded_boundary_times": manifest["excluded_boundary_times"],
            "plateau_times": manifest["plateau_times"],
        },
    }
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = result_dir.parent / f".{result_dir.name}.partial-{uuid.uuid4().hex}"
    try:
        stage.mkdir()
        npz_path = stage / "observables.npz"
        payload: dict[str, Any] = {
            "cfg_ids": arrays["cfg_ids"],
            "lhs": arrays["lhs"],
            "rhs": arrays["rhs"],
            "charge": arrays["charge"],
            "wt_lhs": arrays["lhs"],
            "wt_rhs": arrays["rhs"],
            "charge_ratio": arrays["charge"],
            "validity_mask": np.array(
                [
                    time not in set(manifest["contact_times"]) | set(manifest["excluded_boundary_times"])
                    for time in range(manifest["temporal_extent"])
                ],
                dtype=bool,
            ),
            "exclusion_reason": np.array(
                [
                    ";".join(
                        label
                        for label, times in (
                            ("contact", manifest["contact_times"]),
                            ("boundary_stencil", manifest["excluded_boundary_times"]),
                        )
                        if time in times
                    )
                    for time in range(manifest["temporal_extent"])
                ],
                dtype="U64",
            ),
            "provenance_json": np.array(json.dumps(provenance, sort_keys=True)),
        }
        if "contact_term" in arrays:
            payload["contact_term"] = arrays["contact_term"]
        np.savez(npz_path, **payload)
        with npz_path.open("r+b") as handle:
            os.fsync(handle.fileno())
        result = {
            "schema": "lattice.conserved-current-validation.producer-result/v1",
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "classification": "synthetic" if manifest["synthetic"] else "real-gauge",
            "producer_mode": "audited-contractions-v1",
            "observable_npz": "observables.npz",
            "observable_npz_sha256": _sha256(npz_path),
            "configuration_ids": [str(item) for item in arrays["cfg_ids"]],
            "temporal_extent": manifest["temporal_extent"],
            "ne": checked["ne"],
            "input_artifacts": checked["artifacts"],
            "audited_contractions": {
                "path": str(checked["contract_path"]),
                "sha256": _sha256(checked["contract_path"]),
            },
            "script_sha256": _sha256(Path(__file__)),
            "source_files": manifest["source_files"],
            "git": _git_state(),
            "current_api": manifest["current_api"],
            "provenance": provenance,
        }
        _atomic_write_json(stage / "producer-result.json", result)
        done = {
            "status": "complete",
            "artifact_sha256": {name: _sha256(stage / name) for name in ("observables.npz", "producer-result.json")},
        }
        _atomic_write_json(stage / "DONE", done)
        os.replace(stage, result_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="absolute audited producer manifest")
    parser.add_argument("--result-dir", type=Path, required=True, help="new absolute result directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate schema, absolute paths, and hashes without loading arrays or writing output",
    )
    parser.add_argument(
        "--synthetic-fixture", action="store_true", help="CPU-test-only; requires manifest.synthetic=true"
    )
    args = parser.parse_args()
    try:
        manifest_path = _absolute_existing(str(args.manifest), "manifest")
        manifest = _load_json(manifest_path)
        if args.dry_run:
            if args.synthetic_fixture != bool(manifest.get("synthetic")):
                _fail(
                    "--synthetic-fixture is allowed only with manifest.synthetic=true; real manifests must not use it"
                )
            checked = validate_manifest(manifest, load_arrays=False)
            print(
                json.dumps(
                    {
                        "status": "dry-run-valid",
                        "mode": manifest["producer"]["mode"],
                        "configurations": len(checked["cfg_ids"]),
                        "result_dir": str(args.result_dir),
                    },
                    sort_keys=True,
                )
            )
            return 0
        result = produce(manifest, args.result_dir, synthetic_fixture=args.synthetic_fixture)
    except ProducerError as exc:
        print(f"producer error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "result_dir": str(args.result_dir),
                "observable_npz": str(args.result_dir / result["observable_npz"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
