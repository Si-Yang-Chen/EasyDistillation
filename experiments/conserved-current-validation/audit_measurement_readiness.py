#!/usr/bin/env python3
"""Audit whether available artifacts satisfy an approved conserved-current measurement contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

CONTRACT_SCHEMA = "lattice.conserved-current.measurement-contract/v1"
INVENTORY_SCHEMA = "lattice.conserved-current.measurement-inventory/v1"
REPORT_SCHEMA = "lattice.conserved-current.measurement-readiness/v1"
ROLES = {
    "candidate-two-point",
    "two-point-denominator",
    "meson-current-two-point",
    "three-point-numerator",
    "wt-lhs",
    "wt-rhs",
    "contact-term",
}
OBSERVABLE_PRODUCTS = {
    "connected-charge-normalization": {
        "two-point-denominator",
        "three-point-numerator",
    },
    "ward-takahashi": {"wt-lhs", "wt-rhs", "contact-term"},
}
ROLE_LAYOUTS = {
    "candidate-two-point": {
        ("source_time", "sink_time"),
        ("source_time", "relative_sink_time"),
    },
    "two-point-denominator": {
        ("source_time", "sink_time"),
        ("source_time", "relative_sink_time"),
    },
    "meson-current-two-point": {
        ("source_time", "sink_time"),
        ("source_time", "relative_sink_time"),
    },
    "three-point-numerator": {
        ("source_time", "current_time", "sink_time"),
    },
    "wt-lhs": {("current_time",)},
    "wt-rhs": {("current_time",)},
    "contact-term": {("current_time",)},
}


class ReadinessError(ValueError):
    pass


def _fail(message: str) -> None:
    raise ReadinessError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        _fail(f"{name} must be a 64-hex SHA-256")
    return value.lower()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{name} must be an object")
    return value


def _load(path: Path, name: str) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReadinessError(f"cannot read {name}: {exc}") from exc
    return _mapping(value, name), hashlib.sha256(payload).hexdigest()


def _nonempty(value: Any, name: str, blockers: list[str]) -> None:
    if not isinstance(value, str) or not value.strip() or value.startswith("REQUIRES_"):
        blockers.append(f"measurement contract lacks {name}")


def _times(values: Any, name: str, extent: int) -> list[int]:
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= extent for value in values
    ):
        _fail(f"{name} must be a list of in-range integer times")
    if len(values) != len(set(values)):
        _fail(f"{name} contains duplicates")
    return values


def _validate_contract(contract: dict[str, Any]) -> list[str]:
    required = {
        "schema",
        "version",
        "status",
        "observable",
        "approval",
        "ensemble",
        "temporal_extent",
        "boundary",
        "operators",
        "current",
        "formulas",
        "time_policy",
        "required_products",
        "notes",
    }
    if set(contract) != required:
        _fail("measurement contract has missing or unknown fields")
    if contract["schema"] != CONTRACT_SCHEMA or contract["version"] != 1:
        _fail("measurement contract schema/version is unsupported")
    if contract["status"] not in {"draft", "approved"}:
        _fail("measurement contract status must be draft or approved")
    observable = contract["observable"]
    if observable not in OBSERVABLE_PRODUCTS:
        _fail("measurement contract observable is unsupported")
    extent = contract["temporal_extent"]
    if isinstance(extent, bool) or not isinstance(extent, int) or extent < 2:
        _fail("measurement contract temporal_extent must be an integer >= 2")
    if contract["boundary"] not in {"periodic", "open"}:
        _fail("measurement contract boundary is invalid")

    blockers = []
    approval = _mapping(contract["approval"], "approval")
    if set(approval) != {"authority", "approved_at", "document_path", "document_sha256"}:
        _fail("measurement contract approval fields are invalid")
    if contract["status"] != "approved":
        blockers.append("measurement contract is not approved")
    for field in ("authority", "approved_at"):
        _nonempty(approval[field], f"approval.{field}", blockers)
    if contract["status"] == "approved":
        path_value = approval["document_path"]
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            blockers.append("approved contract lacks an absolute approval document")
        else:
            path = Path(path_value)
            if not path.is_file():
                blockers.append("approval document does not exist")
            else:
                wanted = _hash(approval["document_sha256"], "approval.document_sha256")
                if _sha256(path) != wanted:
                    blockers.append("approval document SHA-256 does not match")

    ensemble = _mapping(contract["ensemble"], "ensemble")
    if set(ensemble) != {"label", "configuration_ids", "minimum_configurations"}:
        _fail("measurement contract ensemble fields are invalid")
    configuration_ids = ensemble["configuration_ids"]
    if (
        not isinstance(configuration_ids, list)
        or not configuration_ids
        or any(not isinstance(value, str) or not value for value in configuration_ids)
        or len(configuration_ids) != len(set(configuration_ids))
    ):
        _fail("ensemble.configuration_ids must be unique non-empty strings")
    minimum = ensemble["minimum_configurations"]
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        _fail("ensemble.minimum_configurations must be positive")
    if len(configuration_ids) < minimum:
        blockers.append("contract configuration list is below its minimum")

    operators = _mapping(contract["operators"], "operators")
    if set(operators) != {"source", "sink"}:
        _fail("operators must contain source and sink")
    for side in ("source", "sink"):
        operator = _mapping(operators[side], f"operators.{side}")
        if set(operator) != {"name", "definition", "momentum", "projector"}:
            _fail(f"operators.{side} fields are invalid")
        _nonempty(operator["name"], f"operators.{side}.name", blockers)
        _nonempty(operator["definition"], f"operators.{side}.definition", blockers)
        _nonempty(operator["projector"], f"operators.{side}.projector", blockers)
        momentum = operator["momentum"]
        if (
            not isinstance(momentum, list)
            or len(momentum) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) for value in momentum)
        ):
            _fail(f"operators.{side}.momentum must contain three integers")

    current = _mapping(contract["current"], "current")
    if set(current) != {
        "implementation",
        "component",
        "wilson_r",
        "topology",
        "flavor_weights",
    }:
        _fail("current fields are invalid")
    if current["implementation"] != "lattice.insertion.current.ConservedVectorCurrent":
        blockers.append("current implementation is not the Wilson conserved Current API")
    if current["component"] not in {0, 1, 2, 3}:
        _fail("current.component must be 0, 1, 2, or 3")
    if (
        isinstance(current["wilson_r"], bool)
        or not isinstance(current["wilson_r"], (int, float))
        or not np.isfinite(current["wilson_r"])
    ):
        _fail("current.wilson_r must be finite")
    if current["topology"] not in {"connected-v2v", "connected-point", "disconnected"}:
        _fail("current.topology is invalid")
    flavor_weights = current["flavor_weights"]
    if not isinstance(flavor_weights, dict) or not flavor_weights:
        blockers.append("current flavor/electric-charge weights are not defined")
    elif any(
        not isinstance(name, str)
        or not name
        or isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not np.isfinite(weight)
        for name, weight in flavor_weights.items()
    ):
        _fail("current.flavor_weights must map flavor names to finite numbers")

    formulas = _mapping(contract["formulas"], "formulas")
    if set(formulas) != {"c2", "c3", "ratio", "wt_lhs", "wt_rhs", "contact_term"}:
        _fail("formula fields are invalid")
    needed_formulas = (
        ("c2", "c3", "ratio")
        if observable == "connected-charge-normalization"
        else ("wt_lhs", "wt_rhs", "contact_term")
    )
    for field in needed_formulas:
        _nonempty(formulas[field], f"formulas.{field}", blockers)

    policy = _mapping(contract["time_policy"], "time_policy")
    if set(policy) != {
        "source_times",
        "sink_times",
        "current_times",
        "contact_times",
        "excluded_boundary_times",
        "plateau_times",
    }:
        _fail("time_policy fields are invalid")
    for field in policy:
        values = _times(policy[field], f"time_policy.{field}", extent)
        if field in {"source_times", "sink_times", "current_times", "plateau_times"} and not values:
            blockers.append(f"measurement contract lacks non-empty time_policy.{field}")
    if set(policy["contact_times"]) & set(policy["plateau_times"]):
        blockers.append("plateau times overlap contact times")
    if set(policy["excluded_boundary_times"]) & set(policy["plateau_times"]):
        blockers.append("plateau times overlap boundary exclusions")

    products = contract["required_products"]
    if not isinstance(products, list) or any(value not in ROLES for value in products):
        _fail("required_products contains unsupported roles")
    expected_products = OBSERVABLE_PRODUCTS[observable]
    if set(products) != expected_products:
        blockers.append("required_products do not exactly match the selected observable")
    return blockers


def _validate_inventory(inventory: dict[str, Any], *, verify_files: bool) -> list[dict[str, Any]]:
    required = {
        "schema",
        "version",
        "ensemble_label",
        "temporal_extent",
        "boundary",
        "datasets",
        "notes",
    }
    if set(inventory) != required:
        _fail("measurement inventory has missing or unknown fields")
    if inventory["schema"] != INVENTORY_SCHEMA or inventory["version"] != 1:
        _fail("measurement inventory schema/version is unsupported")
    extent = inventory["temporal_extent"]
    if isinstance(extent, bool) or not isinstance(extent, int) or extent < 2:
        _fail("measurement inventory temporal_extent is invalid")
    if inventory["boundary"] not in {"periodic", "open"}:
        _fail("measurement inventory boundary is invalid")
    datasets = inventory["datasets"]
    if not isinstance(datasets, list):
        _fail("measurement inventory datasets must be a list")
    normalized = []
    identifiers = set()
    for index, raw in enumerate(datasets):
        dataset = _mapping(raw, f"datasets[{index}]")
        required_fields = {
            "id",
            "role",
            "configuration_ids",
            "files",
            "operator_source",
            "operator_sink",
            "current_component",
            "topology",
            "time_axes",
            "evidence",
            "notes",
        }
        if set(dataset) != required_fields:
            _fail(f"datasets[{index}] has missing or unknown fields")
        identifier = dataset["id"]
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            _fail("dataset IDs must be unique non-empty strings")
        identifiers.add(identifier)
        role = dataset["role"]
        if role not in ROLES:
            _fail(f"dataset {identifier} has unsupported role")
        time_axes = dataset["time_axes"]
        if not isinstance(time_axes, list) or tuple(time_axes) not in ROLE_LAYOUTS[role]:
            _fail(f"dataset {identifier} time_axes are invalid for role {role}")
        if dataset["topology"] not in {
            "connected-v2v",
            "connected-point",
            "connected-localized-blending",
            "disconnected",
        }:
            _fail(f"dataset {identifier} topology is invalid")
        current_component = dataset["current_component"]
        if role in {"three-point-numerator", "wt-lhs", "wt-rhs", "contact-term"}:
            if (
                isinstance(current_component, bool)
                or not isinstance(current_component, int)
                or current_component not in {0, 1, 2, 3}
            ):
                _fail(f"dataset {identifier} requires a current component")
        elif current_component is not None:
            _fail(f"dataset {identifier} must not declare a current component")
        cfgs = dataset["configuration_ids"]
        if (
            not isinstance(cfgs, list)
            or any(not isinstance(value, str) or not value for value in cfgs)
            or len(cfgs) != len(set(cfgs))
        ):
            _fail(f"dataset {identifier} configuration_ids are invalid")
        files = dataset["files"]
        if not isinstance(files, list) or len(files) != len(cfgs):
            _fail(f"dataset {identifier} must have one file per configuration")
        by_cfg = {}
        for record in files:
            if not isinstance(record, dict) or set(record) != {
                "configuration",
                "path",
                "sha256",
                "shape",
                "dtype",
            }:
                _fail(f"dataset {identifier} file record is invalid")
            configuration = record["configuration"]
            if configuration in by_cfg or configuration not in cfgs:
                _fail(f"dataset {identifier} file configuration is invalid")
            path = Path(record["path"])
            if not path.is_absolute():
                _fail(f"dataset {identifier} file paths must be absolute")
            wanted_hash = _hash(record["sha256"], f"dataset {identifier} file SHA-256")
            if (
                not isinstance(record["shape"], list)
                or any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in record["shape"])
                or not isinstance(record["dtype"], str)
            ):
                _fail(f"dataset {identifier} file shape/dtype is invalid")
            if len(record["shape"]) != len(time_axes) or any(size != extent for size in record["shape"]):
                _fail(f"dataset {identifier} shape must match its temporal time_axes")
            if verify_files:
                if not path.is_file():
                    _fail(f"dataset {identifier} file does not exist: {path}")
                if _sha256(path) != wanted_hash:
                    _fail(f"dataset {identifier} file SHA-256 mismatch: {path}")
                try:
                    array = np.load(path, allow_pickle=False, mmap_mode="r")
                except (OSError, ValueError) as exc:
                    raise ReadinessError(f"cannot load dataset {identifier} file: {exc}") from exc
                if list(array.shape) != record["shape"] or array.dtype.str != record["dtype"]:
                    _fail(f"dataset {identifier} header differs from inventory")
                if not np.all(np.isfinite(array)):
                    _fail(f"dataset {identifier} contains non-finite values")
            by_cfg[configuration] = record
        if set(by_cfg) != set(cfgs):
            _fail(f"dataset {identifier} file coverage is incomplete")
        evidence = _mapping(dataset["evidence"], f"dataset {identifier} evidence")
        if set(evidence) != {"path", "sha256"}:
            _fail(f"dataset {identifier} evidence fields are invalid")
        evidence_path = Path(evidence["path"])
        if not evidence_path.is_absolute():
            _fail(f"dataset {identifier} evidence path must be absolute")
        evidence_hash = _hash(evidence["sha256"], f"dataset {identifier} evidence hash")
        if verify_files and (not evidence_path.is_file() or _sha256(evidence_path) != evidence_hash):
            _fail(f"dataset {identifier} evidence file/hash is invalid")
        normalized.append(dataset)
    return normalized


def audit(contract: dict[str, Any], inventory: dict[str, Any], *, verify_files: bool) -> dict[str, Any]:
    blockers = _validate_contract(contract)
    datasets = _validate_inventory(inventory, verify_files=verify_files)
    if not verify_files:
        blockers.append("inventory files, headers, hashes, finiteness, and evidence were not verified")
    if contract["ensemble"]["label"] != inventory["ensemble_label"]:
        blockers.append("contract and inventory ensemble labels differ")
    if contract["temporal_extent"] != inventory["temporal_extent"]:
        blockers.append("contract and inventory temporal extents differ")
    if contract["boundary"] != inventory["boundary"]:
        blockers.append("contract and inventory boundaries differ")

    cfgs = set(contract["ensemble"]["configuration_ids"])
    operators = contract["operators"]
    component = contract["current"]["component"]
    products = {}
    for role in sorted(set(contract["required_products"])):
        candidates = []
        for dataset in datasets:
            if dataset["role"] != role:
                continue
            reasons = []
            if not cfgs.issubset(set(dataset["configuration_ids"])):
                reasons.append("configuration coverage incomplete")
            if dataset["operator_source"] != operators["source"]["name"]:
                reasons.append("source operator mismatch")
            if dataset["operator_sink"] != operators["sink"]["name"]:
                reasons.append("sink operator mismatch")
            if dataset["topology"] != contract["current"]["topology"]:
                reasons.append("topology mismatch")
            if (
                role in {"three-point-numerator", "wt-lhs", "wt-rhs", "contact-term"}
                and dataset["current_component"] != component
            ):
                reasons.append("current component mismatch")
            candidates.append({"dataset_id": dataset["id"], "usable": not reasons, "reasons": reasons})
        products[role] = candidates
        if not any(candidate["usable"] for candidate in candidates):
            blockers.append(f"no usable inventory dataset for required product {role}")

    blockers = sorted(set(blockers))
    return {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "ready": not blockers,
        "files_verified": verify_files,
        "observable": contract["observable"],
        "contract_status": contract["status"],
        "ensemble_label": contract["ensemble"]["label"],
        "configuration_ids": contract["ensemble"]["configuration_ids"],
        "required_products": contract["required_products"],
        "product_candidates": products,
        "available_dataset_roles": sorted({dataset["role"] for dataset in datasets}),
        "blockers": blockers,
        "next_action": (
            "produce audited contractions and run validation"
            if not blockers
            else "resolve measurement-definition and missing-product blockers before numerical physics"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--verify-files", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        contract_path = args.contract.expanduser().resolve()
        inventory_path = args.inventory.expanduser().resolve()
        contract, contract_sha = _load(contract_path, "measurement contract")
        inventory, inventory_sha = _load(inventory_path, "measurement inventory")
        result = audit(contract, inventory, verify_files=args.verify_files)
        result["contract"] = {"path": str(contract_path), "sha256": contract_sha}
        result["inventory"] = {"path": str(inventory_path), "sha256": inventory_sha}
        encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
        if args.output is None:
            print(encoded, end="")
        else:
            output = args.output.expanduser()
            if not output.is_absolute():
                _fail("output path must be absolute")
            if output.exists():
                _fail(f"refusing to overwrite output: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(encoded, encoding="utf-8")
    except (ReadinessError, OSError, ValueError) as exc:
        print(f"measurement readiness error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
