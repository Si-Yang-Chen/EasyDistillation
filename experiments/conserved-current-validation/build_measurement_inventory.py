#!/usr/bin/env python3
"""Build a hashed conserved-current measurement inventory from declared NPY datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

SOURCE_SCHEMA = "lattice.conserved-current.measurement-inventory-source/v1"
OUTPUT_SCHEMA = "lattice.conserved-current.measurement-inventory/v1"
ROLES = {
    "candidate-two-point",
    "two-point-denominator",
    "meson-current-two-point",
    "three-point-numerator",
    "wt-lhs",
    "wt-rhs",
    "contact-term",
}


class InventoryError(ValueError):
    pass


def _fail(message: str) -> None:
    raise InventoryError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryError(f"cannot read inventory source: {exc}") from exc
    if not isinstance(value, dict):
        _fail("inventory source must be a JSON object")
    return value


def _absolute_file(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        _fail(f"{name} must be an absolute path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        _fail(f"{name} must be absolute")
    path = path.resolve()
    if not path.is_file():
        _fail(f"{name} does not exist: {path}")
    return path


def _configuration_ids(value: Any) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        _fail("configuration_ids must be unique non-empty strings")
    return value


def build(source: dict[str, Any], *, verify_finite: bool) -> dict[str, Any]:
    required = {
        "schema",
        "version",
        "ensemble_label",
        "temporal_extent",
        "boundary",
        "configuration_ids",
        "datasets",
        "notes",
    }
    if set(source) != required:
        _fail("inventory source has missing or unknown fields")
    if source["schema"] != SOURCE_SCHEMA or source["version"] != 1:
        _fail("inventory source schema/version is unsupported")
    extent = source["temporal_extent"]
    if isinstance(extent, bool) or not isinstance(extent, int) or extent < 2:
        _fail("temporal_extent must be an integer >= 2")
    if source["boundary"] not in {"periodic", "open"}:
        _fail("boundary must be periodic or open")
    cfgs = _configuration_ids(source["configuration_ids"])
    datasets = source["datasets"]
    if not isinstance(datasets, list):
        _fail("datasets must be a list")

    output_datasets = []
    identifiers = set()
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            _fail(f"datasets[{index}] must be an object")
        required_dataset = {
            "id",
            "role",
            "file_pattern",
            "operator_source",
            "operator_sink",
            "current_component",
            "topology",
            "time_axes",
            "evidence_path",
            "notes",
        }
        if set(dataset) != required_dataset:
            _fail(f"datasets[{index}] has missing or unknown fields")
        identifier = dataset["id"]
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            _fail("dataset IDs must be unique non-empty strings")
        identifiers.add(identifier)
        if dataset["role"] not in ROLES:
            _fail(f"dataset {identifier} role is unsupported")
        pattern = dataset["file_pattern"]
        if not isinstance(pattern, str) or "{configuration}" not in pattern:
            _fail(f"dataset {identifier} file_pattern must contain {{configuration}}")
        sample = Path(pattern.format(configuration=cfgs[0])).expanduser()
        if not sample.is_absolute():
            _fail(f"dataset {identifier} file_pattern must be absolute")
        evidence = _absolute_file(dataset["evidence_path"], f"dataset {identifier} evidence")
        records = []
        expected_shape = None
        expected_dtype = None
        for configuration in cfgs:
            path = _absolute_file(
                pattern.format(configuration=configuration),
                f"dataset {identifier} file",
            )
            try:
                array = np.load(path, allow_pickle=False, mmap_mode="r")
            except (OSError, ValueError) as exc:
                raise InventoryError(f"cannot load dataset {identifier}: {exc}") from exc
            if not np.issubdtype(array.dtype, np.number):
                _fail(f"dataset {identifier} must contain numeric arrays")
            if verify_finite and not np.all(np.isfinite(array)):
                _fail(f"dataset {identifier} contains non-finite values")
            shape = tuple(int(size) for size in array.shape)
            if expected_shape is None:
                expected_shape = shape
                expected_dtype = array.dtype
            elif shape != expected_shape or array.dtype != expected_dtype:
                _fail(f"dataset {identifier} files must share shape and dtype")
            records.append(
                {
                    "configuration": configuration,
                    "path": str(path),
                    "sha256": _sha256(path),
                    "shape": list(shape),
                    "dtype": array.dtype.str,
                }
            )
        output_datasets.append(
            {
                "id": identifier,
                "role": dataset["role"],
                "configuration_ids": cfgs,
                "files": records,
                "operator_source": dataset["operator_source"],
                "operator_sink": dataset["operator_sink"],
                "current_component": dataset["current_component"],
                "topology": dataset["topology"],
                "time_axes": dataset["time_axes"],
                "evidence": {"path": str(evidence), "sha256": _sha256(evidence)},
                "notes": dataset["notes"],
            }
        )
    return {
        "schema": OUTPUT_SCHEMA,
        "version": 1,
        "ensemble_label": source["ensemble_label"],
        "temporal_extent": extent,
        "boundary": source["boundary"],
        "datasets": output_datasets,
        "notes": source["notes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-finite", action="store_true")
    args = parser.parse_args()
    try:
        source_path = args.source.expanduser().resolve()
        output = args.output.expanduser()
        if not output.is_absolute():
            _fail("output path must be absolute")
        output = output.resolve()
        if output.exists():
            _fail(f"refusing to overwrite output: {output}")
        result = build(_load(source_path), verify_finite=args.verify_finite)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    except (InventoryError, OSError, ValueError) as exc:
        print(f"measurement inventory error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
