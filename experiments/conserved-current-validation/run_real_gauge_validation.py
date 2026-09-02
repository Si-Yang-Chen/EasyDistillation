#!/usr/bin/env python3
"""Analyze real-gauge Wilson conserved-current NPZ data; this program never launches a GPU job."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np

from analyze_conservation import chi2_p_value, correlated_constant_fit, jackknife, jackknife_covariance, jsonable

REQUIRED_ARRAYS = ("cfg_ids", "lhs", "rhs", "charge", "provenance_json")
REQUIRED_ARTIFACTS = ("data_table.csv", "wt_diagnostic.svg", "charge_diagnostic.svg", "result.json")
HASH_KEYS = (
    "vsv_perambulator_sha256",
    "psv_sha256",
    "overlap_sha256",
    "current_p2v_sha256",
    "current_p2p_sha256",
)


class ValidationError(ValueError):
    """An input or manifest violates the audited experiment contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read manifest JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValidationError("manifest root must be a JSON object")
    return decoded


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{name} must be an object")
    return value


def _require_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ValidationError(f"{name} must be a SHA-256 hex string")
    return value.lower()


def _parse_provenance(npz: Any) -> dict[str, Any]:
    value = npz["provenance_json"]
    if value.shape not in {(), (1,)}:
        raise ValidationError("provenance_json must be a scalar or one-element Unicode array")
    raw = str(value.item() if value.shape == () else value[0])
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"provenance_json is not JSON: {exc}") from exc
    return _require_mapping(parsed, "provenance_json")


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != "lattice.conserved-current-validation.manifest/v1":
        raise ValidationError("unsupported manifest schema")
    if manifest.get("boundary") not in {"periodic", "open"}:
        raise ValidationError("manifest boundary must be periodic or open")
    extent = manifest.get("temporal_extent")
    if not isinstance(extent, int) or isinstance(extent, bool) or extent < 2:
        raise ValidationError("manifest temporal_extent must be an integer >= 2")
    contact = manifest.get("contact_times")
    excluded = manifest.get("excluded_boundary_times")
    plateau = manifest.get("plateau_times")
    for name, values in (("contact_times", contact), ("plateau_times", plateau)):
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(item, int) or item < 0 or item >= extent for item in values)
        ):
            raise ValidationError(f"manifest {name} must be a nonempty list of in-range integer times")
    if not isinstance(excluded, list) or any(
        not isinstance(item, int) or item < 0 or item >= extent for item in excluded
    ):
        raise ValidationError("manifest excluded_boundary_times must be a list of in-range integer times")
    if manifest["boundary"] == "open" and not excluded:
        raise ValidationError("open boundary requires explicit excluded_boundary_times; no wrap is assumed")
    gates = _require_mapping(manifest.get("pass_gates"), "pass_gates")
    if not isinstance(gates.get("minimum_configurations"), int) or gates["minimum_configurations"] < 8:
        raise ValidationError("minimum_configurations must be an integer >= 8")
    for name in ("wt", "charge"):
        gate = _require_mapping(gates.get(name), f"pass_gates.{name}")
        for field in ("absolute_tolerance", "relative_to_statistical_error", "max_pull", "p_value_min"):
            value = gate.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValidationError(f"pass_gates.{name}.{field} must be a nonnegative finite number")
    expected = _require_mapping(manifest.get("expected_provenance"), "expected_provenance")
    for field in ("code_sha256", "api_sha256"):
        _require_hash(expected.get(field), f"expected_provenance.{field}")
    current = _require_mapping(manifest.get("current_api"), "current_api")
    if (
        current.get("version") != "1.2.0"
        or current.get("term_schema") != "lattice.current.term/v1"
        or current.get("assembler_schema") != "lattice.current.assembler/v1"
    ):
        raise ValidationError("manifest must declare frozen Current API 1.2.0 and v1 term/assembler schemas")


def load_input(
    manifest: dict[str, Any], path: Path, require_real_gauge: bool
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _validate_manifest(manifest)
    if path.suffix.lower() != ".npz":
        raise ValidationError("only audited NPZ input is supported; HDF5-like inputs must be converted to NPZ")
    try:
        with np.load(path, allow_pickle=False) as npz:
            missing = set(REQUIRED_ARRAYS) - set(npz.files)
            if missing:
                raise ValidationError(f"NPZ missing required arrays: {sorted(missing)}")
            arrays = {name: np.array(npz[name]) for name in ("cfg_ids", "lhs", "rhs", "charge")}
            provenance = _parse_provenance(npz)
    except OSError as exc:
        raise ValidationError(f"cannot load NPZ: {exc}") from exc
    cfg_ids, lhs, rhs, charge = (arrays[name] for name in ("cfg_ids", "lhs", "rhs", "charge"))
    if cfg_ids.ndim != 1 or cfg_ids.size == 0 or cfg_ids.dtype.kind not in "iuUS":
        raise ValidationError("cfg_ids must be a nonempty one-dimensional integer or Unicode array")
    normalized_ids = [str(value) for value in cfg_ids]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValidationError("cfg_ids must be unique")
    extent = manifest["temporal_extent"]
    for name, array in (("lhs", lhs), ("rhs", rhs), ("charge", charge)):
        if array.dtype.kind not in "fiu" or array.ndim != 2 or array.shape != (cfg_ids.size, extent):
            raise ValidationError(f"{name} must be a finite real array with shape (n_configurations, temporal_extent)")
        arrays[name] = array.astype(float, copy=False)
        if not np.all(np.isfinite(arrays[name])):
            raise ValidationError(f"{name} contains non-finite values")
    expected = manifest["expected_provenance"]
    required = (
        "synthetic",
        "code_sha256",
        "api_sha256",
        "api_version",
        "term_schema",
        "assembler_schema",
        "boundary",
        "source_sink_current_time_semantics",
        "temporal_extent",
        "propagator_artifacts",
    )
    missing = [name for name in required if name not in provenance]
    if missing:
        raise ValidationError(f"provenance missing required fields: {missing}")
    if not isinstance(provenance["synthetic"], (bool, np.bool_)):
        raise ValidationError("provenance.synthetic must be boolean")
    if require_real_gauge and bool(provenance["synthetic"]):
        raise ValidationError("--require-real-gauge rejects provenance.synthetic=true")
    for field in ("code_sha256", "api_sha256"):
        actual = _require_hash(provenance[field], f"provenance.{field}")
        if actual != _require_hash(expected[field], f"expected_provenance.{field}"):
            raise ValidationError(f"provenance {field} does not match manifest")
    for field, wanted in (
        ("api_version", manifest["current_api"]["version"]),
        ("term_schema", manifest["current_api"]["term_schema"]),
        ("assembler_schema", manifest["current_api"]["assembler_schema"]),
        ("boundary", manifest["boundary"]),
        ("temporal_extent", extent),
    ):
        if provenance[field] != wanted:
            raise ValidationError(f"provenance {field} does not match manifest")
    if provenance["source_sink_current_time_semantics"] != "source-first":
        raise ValidationError("provenance must explicitly declare source-first source/sink/current time semantics")
    artifacts = _require_mapping(provenance["propagator_artifacts"], "provenance.propagator_artifacts")
    for field in HASH_KEYS:
        _require_hash(artifacts.get(field), f"provenance.propagator_artifacts.{field}")
    return arrays, provenance


def _safe_pull(mean: np.ndarray, error: np.ndarray, reference: float | np.ndarray = 0.0) -> np.ndarray:
    difference = np.asarray(mean, dtype=float) - reference
    return np.where(error > 0, difference / error, np.where(np.abs(difference) == 0, 0.0, np.inf))


def _time_validity(manifest: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    extent = manifest["temporal_extent"]
    reasons = []
    for time in range(extent):
        labels = []
        if time in manifest["contact_times"]:
            labels.append("contact")
        if time in manifest["excluded_boundary_times"]:
            labels.append("boundary_stencil")
        reasons.append(";".join(labels))
    return np.array([not item for item in reasons], dtype=bool), reasons


def _gate_wt(residual: np.ndarray, selected: np.ndarray, gates: dict[str, Any]) -> dict[str, Any]:
    summary = jackknife(residual)
    mean, error = np.asarray(summary["mean"]), np.asarray(summary["error"])
    pull = _safe_pull(mean, error)
    chosen = np.flatnonzero(selected)
    covariance = jackknife_covariance(residual[:, chosen])
    chi2 = float(mean[chosen] @ covariance["inverse"] @ mean[chosen]) if covariance["rank"] else None
    dof = int(covariance["rank"])
    p_value, p_reason = chi2_p_value(chi2, dof) if chi2 is not None else (None, "zero-rank covariance")
    threshold = np.maximum(gates["absolute_tolerance"], gates["relative_to_statistical_error"] * error)
    tolerance_ok = bool(np.all(np.abs(mean[chosen]) <= threshold[chosen]))
    pull_ok = bool(np.all(np.abs(pull[chosen]) <= gates["max_pull"]))
    p_ok = p_value is None or bool(p_value >= gates["p_value_min"])
    return {
        "mean": mean,
        "error": error,
        "pull": pull,
        "threshold": threshold,
        "chi2": chi2,
        "dof": dof,
        "p_value": p_value,
        "p_value_reason": p_reason,
        "covariance_rank": covariance["rank"],
        "covariance_valid": covariance["valid"],
        "passed": tolerance_ok and pull_ok and p_ok,
        "tolerance_passed": tolerance_ok,
        "pull_passed": pull_ok,
        "p_value_passed_or_not_applicable": p_ok,
    }


def _gate_charge(charge: np.ndarray, plateau: np.ndarray, gates: dict[str, Any]) -> dict[str, Any]:
    summary = jackknife(charge)
    mean, error = np.asarray(summary["mean"]), np.asarray(summary["error"])
    pull = _safe_pull(mean, error, 1.0)
    fit = correlated_constant_fit(charge[:, plateau])
    threshold = np.maximum(gates["absolute_tolerance"], gates["relative_to_statistical_error"] * error[plateau])
    tolerance_ok = bool(np.all(np.abs(mean[plateau] - 1.0) <= threshold))
    pull_ok = bool(np.all(np.abs(pull[plateau]) <= gates["max_pull"]))
    p_ok = fit["p_value"] is None or bool(fit["p_value"] >= gates["p_value_min"])
    return {
        "mean": mean,
        "error": error,
        "pull": pull,
        "fit": fit,
        "threshold": threshold,
        "passed": tolerance_ok and pull_ok and p_ok and fit["fit_valid"],
        "tolerance_passed": tolerance_ok,
        "pull_passed": pull_ok,
        "p_value_passed_or_not_applicable": p_ok,
    }


def analyze(
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    provenance: dict[str, Any],
    input_hash: str,
    script_hash: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    valid, reasons = _time_validity(manifest)
    plateau = np.array(sorted(set(manifest["plateau_times"])), dtype=int)
    if np.any(~valid[plateau]):
        raise ValidationError("plateau_times must exclude contact and boundary-stencil times")
    residual = arrays["lhs"] - arrays["rhs"]
    wt = _gate_wt(residual, valid, manifest["pass_gates"]["wt"])
    charge = _gate_charge(arrays["charge"], plateau, manifest["pass_gates"]["charge"])
    config_ok = arrays["cfg_ids"].size >= manifest["pass_gates"]["minimum_configurations"]
    passed = bool(config_ok and wt["passed"] and charge["passed"])
    failures = {"wt": not wt["passed"], "charge": not charge["passed"]}
    attribution = {
        "links_sign_anchor": {
            "evidence_status": "supported" if failures["wt"] else "unverified",
            "evidence": "WT residual gate failed" if failures["wt"] else "no failing WT residual",
        },
        "temporal_propagator": {
            "evidence_status": "unverified",
            "evidence": "requires producer-side VSV/PSV time-slice audit",
        },
        "boundary_contact": {
            "evidence_status": "supported" if any(reasons) and failures["wt"] else "unverified",
            "evidence": "contact/boundary exclusions were declared" if any(reasons) else "no excluded temporal stencil",
        },
        "normalization": {
            "evidence_status": "supported" if failures["charge"] else "unverified",
            "evidence": "Q(t)=1 normalization gate failed"
            if failures["charge"]
            else "charge normalization gate passed",
        },
        "cache_provenance": {
            "evidence_status": "unverified",
            "evidence": "hashes were matched, but external cache lineage is not independently replayed",
        },
        "statistics_discretization": {
            "evidence_status": "supported"
            if (wt["p_value"] is not None and wt["p_value"] < manifest["pass_gates"]["wt"]["p_value_min"])
            or (
                charge["fit"]["p_value"] is not None
                and charge["fit"]["p_value"] < manifest["pass_gates"]["charge"]["p_value_min"]
            )
            else "unverified",
            "evidence": "low correlated-fit p-value",
        },
    }
    rows: list[dict[str, Any]] = []
    plateau_set = set(plateau.tolist())
    for cfg_index, cfg_id in enumerate(arrays["cfg_ids"]):
        for time in range(manifest["temporal_extent"]):
            in_wt = bool(valid[time])
            in_charge = time in plateau_set
            rows.append(
                {
                    "cfg_id": str(cfg_id),
                    "time": time,
                    "lhs": arrays["lhs"][cfg_index, time],
                    "rhs": arrays["rhs"][cfg_index, time],
                    "residual": residual[cfg_index, time],
                    "residual_mean": wt["mean"][time] if in_wt else None,
                    "residual_error": wt["error"][time] if in_wt else None,
                    "residual_pull": wt["pull"][time] if in_wt else None,
                    "charge": arrays["charge"][cfg_index, time],
                    "charge_mean": charge["mean"][time],
                    "charge_error": charge["error"][time],
                    "charge_fit_residual": charge["mean"][time] - charge["fit"]["constant"]
                    if in_charge and charge["fit"]["constant"] is not None
                    else None,
                    "charge_pull": charge["pull"][time] if in_charge else None,
                    "wt_valid": in_wt,
                    "charge_plateau": in_charge,
                    "exclusion_reason": reasons[time],
                }
            )
    result = {
        "schema": "lattice.conserved-current-validation.result/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "synthetic" if provenance["synthetic"] else "real-gauge",
        "passed": passed,
        "input_sha256": input_hash,
        "script_sha256": script_hash,
        "manifest_git": {"head": manifest.get("git_head"), "dirty": manifest.get("git_dirty")},
        "current_api": manifest["current_api"],
        "expected_provenance": manifest["expected_provenance"],
        "input_provenance": provenance,
        "job": manifest.get("job") if isinstance(manifest.get("job"), dict) else None,
        "pass_gates": manifest["pass_gates"],
        "outcome": {
            "minimum_configurations_passed": config_ok,
            "configuration_count": int(arrays["cfg_ids"].size),
            "wt": jsonable(wt),
            "charge": jsonable(charge),
            "attribution": attribution,
            "limitation": (
                "This analysis validates supplied contractions and provenance only; it neither generates "
                "propagators nor establishes final real-gauge physics."
            ),
        },
    }
    return result, rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "cfg_id",
        "time",
        "lhs",
        "rhs",
        "residual",
        "residual_mean",
        "residual_error",
        "residual_pull",
        "charge",
        "charge_mean",
        "charge_error",
        "charge_fit_residual",
        "charge_pull",
        "wt_valid",
        "charge_plateau",
        "exclusion_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _points(values: np.ndarray, width: int, height: int, y_min: float, y_max: float) -> str:
    span = y_max - y_min or 1.0
    points = []
    for index, value in enumerate(values):
        x = 30 + index * (width - 45) / max(values.size - 1, 1)
        y = height - 25 - (float(value) - y_min) / span * (height - 45)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _write_svg(path: Path, title: str, panels: list[tuple[str, np.ndarray]], label: str) -> None:
    width, panel_height = 760, 180
    total_height = panel_height * len(panels) + 40
    svg_open = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{total_height}" viewBox="0 0 {width} {total_height}">'
    )
    lines = [
        svg_open,
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="18" y="22" font-family="sans-serif" font-size="16">{title} - {label}</text>',
    ]
    for index, (name, values) in enumerate(panels):
        y0 = 40 + panel_height * index
        lo, hi = float(np.min(values)), float(np.max(values))
        padding = max((hi - lo) * 0.12, 1e-12)
        lo, hi = lo - padding, hi + padding
        points = _points(np.asarray(values), width, panel_height, lo, hi)
        translated = " ".join(f"{x},{float(y) + y0:.2f}" for x, y in (pair.split(",") for pair in points.split()))
        lines += [
            f'<text x="18" y="{y0 + 16}" font-family="sans-serif" font-size="13">{name}</text>',
            (
                f'<line x1="30" y1="{y0 + panel_height - 25}" x2="{width - 15}" '
                f'y2="{y0 + panel_height - 25}" stroke="#555"/>'
            ),
            f'<polyline fill="none" stroke="#1565c0" stroke-width="2" points="{translated}"/>',
        ]
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _write_results(stage: Path, result: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    _write_csv(stage / "data_table.csv", rows)
    wt = result["outcome"]["wt"]
    charge = result["outcome"]["charge"]
    _write_svg(
        stage / "wt_diagnostic.svg",
        "Wilson backward-divergence WT",
        [("lhs-rhs residual", np.asarray(wt["mean"])), ("pull", np.asarray(wt["pull"]))],
        result["classification"],
    )
    _write_svg(
        stage / "charge_diagnostic.svg",
        "Zero-momentum temporal charge Q(t), reference Q=1",
        [("Q(t)-1", np.asarray(charge["mean"]) - 1.0), ("plateau pull", np.asarray(charge["pull"]))],
        result["classification"],
    )
    result["artifact_sha256"] = {name: _sha256(stage / name) for name in REQUIRED_ARTIFACTS if name != "result.json"}
    encoded = json.dumps(jsonable(result), sort_keys=True, indent=2) + "\n"
    result_path = stage / "result.json"
    result_path.write_text(encoded, encoding="utf-8")
    with result_path.open("r+b") as handle:
        os.fsync(handle.fileno())
    hashes = {name: _sha256(stage / name) for name in REQUIRED_ARTIFACTS}
    done = stage / "DONE"
    done.write_text(
        json.dumps({"status": "complete", "artifact_sha256": hashes}, sort_keys=True) + "\n", encoding="utf-8"
    )
    with done.open("r+b") as handle:
        os.fsync(handle.fileno())


def run(manifest_path: Path, input_path: Path, result_dir: Path, require_real_gauge: bool) -> dict[str, Any]:
    if result_dir.exists():
        raise ValidationError(f"refusing to overwrite existing result directory: {result_dir}")
    manifest = _load_json(manifest_path)
    arrays, provenance = load_input(manifest, input_path, require_real_gauge)
    result, rows = analyze(manifest, arrays, provenance, _sha256(input_path), _sha256(Path(__file__)))
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = result_dir.parent / f".{result_dir.name}.partial-{uuid.uuid4().hex}"
    try:
        stage.mkdir()
        _write_results(stage, result, rows)
        os.replace(stage, result_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("input_npz", type=Path)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--require-real-gauge", action="store_true", help="reject fixtures marked synthetic")
    args = parser.parse_args()
    try:
        result = run(args.manifest, args.input_npz, args.result_dir, args.require_real_gauge)
    except ValidationError as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "result_dir": str(args.result_dir),
                "passed": result["passed"],
                "classification": result["classification"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
