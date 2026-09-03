#!/usr/bin/env python3
"""Audit full-size localized inputs for a directed-current pair contract.

This is a read-only header/coverage audit. It never mutates a source dataset,
loads full arrays into memory, generates a Current artifact, or performs a
physical contraction. A V2V pair requires direct VSV blocks, so complete PSV
and PSP coverage cannot silently compensate for missing VSV source times.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA = "lattice.current.fullsize-coverage-audit/v1"
CONFIGURATIONS = ("10000", "13000", "14000", "15000", "16000", "17000", "18000", "19000")
VSV_DIRNAME = "04.perambulator.localized.nev128_to_nev128.fulltime.src18.np64"
PSV_DIRNAME = "04.perambulator.localized.nev128_to_np64.fulltime.src72"
PSP_DIRNAME = "04.perambulator.localized.np64_to_np64.fulltime.src18"
OVERLAP_DIRNAME = "03.overlap_matrix"


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_int_list(text: str, *, upper: int) -> list[int]:
    values: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first_text, last_text = token.split("-", 1)
            first, last = int(first_text), int(last_text)
            if last < first:
                raise ValueError(f"invalid descending range: {token}")
            values.update(range(first, last + 1))
        else:
            values.add(int(token))
    if not values or any(value < 0 or value >= upper for value in values):
        raise ValueError(f"times must be inside [0, {upper})")
    return sorted(values)


def _relative_path(path: Path) -> str:
    return path.as_posix()


def _manifest(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def _require_fields(value: dict[str, Any], fields: Iterable[str], name: str) -> None:
    missing = sorted(set(fields).difference(value))
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(missing)}")


def _product_manifest(path: Path, product: str) -> tuple[dict[str, Any], str]:
    value, digest = _manifest(path)
    _require_fields(
        value,
        (
            "version",
            "layout",
            "product",
            "time_convention",
            "global_lattice",
            "grid_size",
            "local_lattice",
            "source_times",
            "dtype",
        ),
        f"{product} manifest",
    )
    if value["version"] != 1 or value["layout"] != "source-time-rank-slab":
        raise ValueError(f"unsupported {product} manifest layout/version")
    if value["product"] != product:
        raise ValueError(f"manifest product is not {product}")
    if value["time_convention"] != "absolute-global-source-and-sink":
        raise ValueError(f"{product} time convention is not absolute-global-source-and-sink")
    return value, digest


def _expected_shape(manifest: dict[str, Any]) -> tuple[int, ...]:
    local_time = int(manifest["local_lattice"][3])
    if "tail_shape" in manifest:
        return (local_time, *(int(value) for value in manifest["tail_shape"]))
    axis_order = manifest.get("axis_order")
    if axis_order == [
        "t_sink_local",
        "spin_sink",
        "spin_source",
        "point_sink",
        "color_sink",
        "point_source",
        "color_source",
    ]:
        return (
            local_time,
            4,
            4,
            int(manifest["np_snk"]),
            3,
            int(manifest["np_src"]),
            3,
        )
    raise ValueError("cannot derive expected rank-slab shape")


def _common_manifest_fields(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: manifest.get(key)
        for key in (
            "global_lattice",
            "grid_size",
            "local_lattice",
            "time_convention",
            "mass",
            "clover",
            "t_boundary",
            "gauge_prefix",
            "point_root",
            "tol",
            "maxiter",
            "stout_steps",
            "stout_rho",
            "multigrid",
        )
    }


def _validate_common_manifests(manifests: dict[str, dict[str, Any]]) -> None:
    reference_name, reference = next(iter(manifests.items()))
    expected = _common_manifest_fields(reference)
    for name, manifest in manifests.items():
        actual = _common_manifest_fields(manifest)
        mismatches = [key for key in expected if actual[key] != expected[key]]
        if mismatches:
            raise ValueError(f"incompatible manifest fields for {name} vs {reference_name}: {mismatches}")


@dataclass
class FileCheck:
    path: Path
    expected_shape: tuple[int, ...]
    expected_dtype: np.dtype[Any]
    hash_files: bool

    def run(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "path": _relative_path(self.path),
            "exists": self.path.is_file(),
            "expected_shape": list(self.expected_shape),
            "expected_dtype": self.expected_dtype.str,
        }
        if not self.path.is_file():
            record.update({"valid": False, "reason": "missing"})
            return record
        try:
            values = np.load(self.path, mmap_mode="r", allow_pickle=False)
            actual_shape = tuple(int(size) for size in values.shape)
            actual_dtype = values.dtype
            actual_size = int(self.path.stat().st_size)
            expected_size = int(values.offset) + int(np.prod(self.expected_shape)) * self.expected_dtype.itemsize
            record.update(
                {
                    "shape": list(actual_shape),
                    "dtype": actual_dtype.str,
                    "bytes": actual_size,
                    "expected_file_bytes": expected_size,
                    "valid": (
                        actual_shape == self.expected_shape
                        and actual_dtype == self.expected_dtype
                        and actual_size == expected_size
                    ),
                }
            )
            if not record["valid"]:
                record["reason"] = "shape-dtype-or-byte-size-mismatch"
            if self.hash_files:
                record["sha256"] = _sha256(self.path)
        except (OSError, ValueError, TypeError) as error:
            record.update({"valid": False, "reason": f"load-error: {type(error).__name__}: {error}"})
        return record


def _audit_product(
    *,
    root: Path,
    dirname: str,
    product: str,
    configurations: list[str],
    hash_files: bool,
) -> dict[str, Any]:
    directory = root / dirname
    manifest, manifest_sha256 = _product_manifest(directory / "manifest.json", product)
    global_time = int(manifest["global_lattice"][3])
    grid = tuple(int(value) for value in manifest["grid_size"])
    if grid[:3] != (1, 1, 1):
        raise ValueError(f"{product} requires a time-only grid for this audit")
    temporal_ranks = grid[3]
    source_times = sorted({int(value) for value in manifest["source_times"]})
    expected_shape = _expected_shape(manifest)
    expected_dtype = np.dtype(manifest["dtype"])
    records: list[dict[str, Any]] = []
    for configuration in configurations:
        for source_time in source_times:
            for rank in range(temporal_ranks):
                path = directory / f"{configuration}.t{source_time:03d}.rank{rank:04d}.npy"
                records.append(FileCheck(path, expected_shape, expected_dtype, hash_files).run())
    valid = [record for record in records if record["valid"]]
    present = [record for record in records if record["exists"]]
    return {
        "directory": directory.as_posix(),
        "manifest_path": (directory / "manifest.json").as_posix(),
        "manifest_sha256": manifest_sha256,
        "product": product,
        "global_time": global_time,
        "temporal_ranks": temporal_ranks,
        "source_times": source_times,
        "source_time_count": len(source_times),
        "expected_shape": list(expected_shape),
        "expected_dtype": expected_dtype.str,
        "expected_file_count": len(records),
        "present_file_count": len(present),
        "valid_file_count": len(valid),
        "complete": len(valid) == len(records),
        "bytes_per_rank_slab": int(np.prod(expected_shape)) * expected_dtype.itemsize,
        "bytes_valid": sum(int(record.get("bytes", 0)) for record in valid),
        "files": records,
    }


def _audit_simple_npy(
    *,
    path: Path,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
    hash_files: bool,
) -> dict[str, Any]:
    record = FileCheck(path, expected_shape, np.dtype(expected_dtype), hash_files).run()
    record["expected_path"] = path.as_posix()
    return record


def _pair_accesses(current_times: Iterable[int], temporal_extent: int) -> list[tuple[int, int]]:
    accesses: set[tuple[int, int]] = set()
    for first_time in current_times:
        for second_time in current_times:
            first_endpoints = ((first_time + 1) % temporal_extent, first_time)
            second_endpoints = ((second_time + 1) % temporal_extent, second_time)
            accesses.add((first_endpoints[0], second_endpoints[1]))
            accesses.add((second_endpoints[0], first_endpoints[1]))
    return sorted(accesses)


def _coverage_for_accesses(accesses: list[tuple[int, int]], available_source_times: set[int]) -> dict[str, Any]:
    required_sources = sorted({source for source, _sink in accesses})
    direct = sorted(set(required_sources).intersection(available_source_times))
    missing = sorted(set(required_sources).difference(available_source_times))
    reverse_candidates = sorted(
        {sink for source, sink in accesses if source in missing and sink in available_source_times}
    )
    return {
        "required_access_count": len(accesses),
        "required_source_times": required_sources,
        "direct_source_times": direct,
        "missing_direct_source_times": missing,
        "reverse_candidate_sink_times": reverse_candidates,
        "direct_complete": not missing,
        "accesses": [{"source_time": source, "sink_time": sink} for source, sink in accesses],
    }


def audit(
    *,
    data_root: Path,
    output: Path | None,
    configurations: list[str],
    current_times: list[int],
    hash_files: bool,
    overlap_point_count: int = 216,
) -> dict[str, Any]:
    global_time = 72
    if any(time >= global_time for time in current_times):
        raise ValueError("current time exceeds the default L_t=72 audit domain")
    products = {
        "VSV": _audit_product(
            root=data_root,
            dirname=VSV_DIRNAME,
            product="VSV",
            configurations=configurations,
            hash_files=hash_files,
        ),
        "PSV": _audit_product(
            root=data_root,
            dirname=PSV_DIRNAME,
            product="PSV",
            configurations=configurations,
            hash_files=hash_files,
        ),
        "PSP": _audit_product(
            root=data_root,
            dirname=PSP_DIRNAME,
            product="PSP",
            configurations=configurations,
            hash_files=hash_files,
        ),
    }
    manifests = {name: _manifest(Path(product["manifest_path"]))[0] for name, product in products.items()}
    _validate_common_manifests(manifests)
    actual_global_time = int(manifests["VSV"]["global_lattice"][3])
    if any(time >= actual_global_time for time in current_times):
        raise ValueError("current time exceeds manifest temporal extent")
    vsv_tail = tuple(int(value) for value in products["VSV"]["expected_shape"][1:])
    psv_tail = tuple(int(value) for value in products["PSV"]["expected_shape"][1:])
    if vsv_tail[:2] != (4, 4) or len(vsv_tail) != 4:
        raise ValueError("VSV manifest tail shape must be (4, 4, sink_ne, source_ne)")
    if psv_tail[:2] != (4, 4) or len(psv_tail) != 5:
        raise ValueError("PSV manifest tail shape must be (4, 4, sink_np, color, source_ne)")
    eigenvectors = manifests["VSV"].get("Ne", vsv_tail[2])
    point_count = int(overlap_point_count)
    if point_count <= 0:
        raise ValueError("overlap point count must be positive")
    if int(eigenvectors) != vsv_tail[2] or int(eigenvectors) != psv_tail[4]:
        raise ValueError("VSV/PSV eigenvector extents are incompatible")
    spatial_shape = tuple(int(value) for value in manifests["VSV"]["global_lattice"][:3])
    accesses = _pair_accesses(current_times, actual_global_time)
    vsv_coverage = _coverage_for_accesses(
        accesses,
        set(products["VSV"]["source_times"]),
    )

    simple_inputs: dict[str, Any] = {}
    overlap_dir = data_root / OVERLAP_DIRNAME
    for configuration in configurations:
        simple_inputs[configuration] = {
            "overlap": _audit_simple_npy(
                path=overlap_dir / f"{configuration}.overlap_matrix.npy",
                expected_shape=(actual_global_time, int(eigenvectors), point_count, psv_tail[3]),
                expected_dtype="<c16",
                hash_files=hash_files,
            ),
            "point": _audit_simple_npy(
                path=data_root / "01.sparsened_field" / f"{configuration}.npy",
                expected_shape=(point_count, actual_global_time, psv_tail[3]),
                expected_dtype="<i4",
                hash_files=hash_files,
            ),
            "eigenvector": _audit_simple_npy(
                path=data_root / "02.laplace_eigs.nev128" / f"{configuration}.npy",
                expected_shape=(actual_global_time, int(eigenvectors), *spatial_shape, 3),
                expected_dtype="<c16",
                hash_files=hash_files,
            ),
        }

    per_configuration_bytes = {
        name: int(product["bytes_valid"] // len(configurations)) if len(configurations) else 0
        for name, product in products.items()
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "classification": "read-only-real-input-coverage-and-cost-audit-not-physics-validation",
        "data_root": data_root.as_posix(),
        "configurations": configurations,
        "current_times": current_times,
        "hash_mode": "sha256" if hash_files else "header-and-stat-only",
        "products": products,
        "simple_inputs": simple_inputs,
        "pair_vsv_coverage": vsv_coverage,
        "cost": {
            "per_configuration_valid_bytes": per_configuration_bytes,
            "all_configurations_valid_bytes": {name: int(product["bytes_valid"]) for name, product in products.items()},
            "all_products_valid_bytes": sum(int(product["bytes_valid"]) for product in products.values()),
            "note": (
                "Storage cost only; GPU working-set and contraction runtime require a separate approved resource pilot."
            ),
        },
        "verdict": {
            "all_requested_files_complete": all(product["complete"] for product in products.values())
            and all(record["valid"] for records in simple_inputs.values() for record in records.values()),
            "direct_vsv_pair_endpoint_coverage": vsv_coverage["direct_complete"],
            "v2v_pair_ready": all(product["complete"] for product in products.values())
            and vsv_coverage["direct_complete"],
            "reasons": [],
        },
    }
    if not report["verdict"]["all_requested_files_complete"]:
        report["verdict"]["reasons"].append("one or more requested input files are missing or invalid")
    if not vsv_coverage["direct_complete"]:
        report["verdict"]["reasons"].append(
            "direct VSV source-time coverage is incomplete for the requested dual-current endpoints"
        )
    report["report_identity"] = hashlib.sha256(_canonical_bytes(report)).hexdigest()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.partial")
        temporary.write_bytes(_canonical_bytes(report))
        temporary.replace(output)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configurations", default=",".join(CONFIGURATIONS))
    parser.add_argument(
        "--current-times",
        default="0-71",
        help="comma-separated times and inclusive ranges, for example 0,4,8-12",
    )
    parser.add_argument(
        "--hash-files",
        action="store_true",
        help="compute exact SHA-256 for every audited NPY; this can read hundreds of GB",
    )
    parser.add_argument("--overlap-point-count", type=int, default=216)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        configurations = [item.strip() for item in args.configurations.split(",") if item.strip()]
        if not configurations:
            raise ValueError("at least one configuration is required")
        current_times = _parse_int_list(args.current_times, upper=72)
        report = audit(
            data_root=args.data_root.expanduser().resolve(),
            output=args.output.expanduser().resolve(),
            configurations=configurations,
            current_times=current_times,
            hash_files=args.hash_files,
            overlap_point_count=args.overlap_point_count,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"full-size coverage audit error: {error}")
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
