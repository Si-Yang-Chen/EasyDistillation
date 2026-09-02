#!/usr/bin/env python3
"""Generate one directed-current V2V artifact on a DCU compute node."""

from __future__ import annotations

import argparse
from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lattice import (  # noqa: E402
    CurrentElementalGenerator,
    EigenvectorNpy,
    GaugeFieldIldg,
    PointSourceNpy,
    set_backend,
)
from lattice.current_elemental import save_directed_current_v2v  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(value, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _input_file(prefix: str, configuration: str, suffix: str) -> Path:
    path = Path(f"{prefix}{configuration}{suffix}").expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def generate(args: argparse.Namespace) -> dict:
    generator = None
    result_dir = args.result_dir.expanduser()
    if not result_dir.is_absolute():
        raise ValueError("result directory must be absolute")
    result_dir = result_dir.resolve()
    if result_dir.exists():
        raise FileExistsError(f"refusing to reuse result directory: {result_dir}")
    if args.used_ne <= 0 or args.used_ne > args.available_ne:
        raise ValueError("used-ne must satisfy 0 < used-ne <= available-ne")
    if args.boundary not in {"periodic", "open"}:
        raise ValueError("boundary must be periodic or open")

    gauge_path = _input_file(args.gauge_prefix, args.configuration, args.gauge_suffix)
    eigenvector_path = _input_file(args.eigenvector_prefix, args.configuration, args.eigenvector_suffix)
    point_path = _input_file(args.point_prefix, args.configuration, args.point_suffix)

    set_backend("cupy")
    gauge = GaugeFieldIldg(
        args.gauge_prefix,
        args.gauge_suffix,
        [args.T, args.L, args.L, args.L, 4, 3, 3],
    )
    eigenvector = EigenvectorNpy(
        args.eigenvector_prefix,
        args.eigenvector_suffix,
        [args.T, args.available_ne, args.L, args.L, args.L, 3],
        args.available_ne,
    )
    point = PointSourceNpy(
        args.point_prefix,
        args.point_suffix,
        [args.point_count, args.T, 3],
        args.point_count,
    )
    generator = CurrentElementalGenerator(
        latt_size=[args.L, args.L, args.L, args.T],
        gauge_field=gauge,
        eigenvector=eigenvector,
        point=point,
        num_nabla=0,
        momentum_list=[(0, 0, 0)],
        usedNe=args.used_ne,
        usedNp=1,
    )

    result_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = result_dir.parent / f".{result_dir.name}.partial-{uuid.uuid4().hex}"
    try:
        stage.mkdir()
        generator.load(args.configuration)
        generated = generator.calc_directed_current_raw(args.boundary)
        artifact_dir = stage / "current-artifact"
        manifest_path = save_directed_current_v2v(
            artifact_dir,
            {"v2v": generated["v2v"]},
            generated["contract"],
            configuration=args.configuration,
            momenta=[(0, 0, 0)],
            gauge_source=gauge_path,
            eigenvector_source=eigenvector_path,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = {
            "schema": "lattice.current.directed-v2v-generation-result/v1",
            "status": "complete",
            "passed": True,
            "classification": "kunshan-smoke-artifact-not-physics-validation",
            "configuration": args.configuration,
            "boundary": args.boundary,
            "available_ne": args.available_ne,
            "used_ne": args.used_ne,
            "lattice": [args.L, args.L, args.L, args.T],
            "artifact": {
                "directory": "current-artifact",
                "manifest": "current-artifact/manifest.json",
                "manifest_sha256": _sha256(manifest_path),
                "artifact_identity": manifest["artifact_identity"],
                "data_filename": manifest["data"]["filename"],
                "data_sha256": manifest["data"]["sha256"],
            },
            "inputs": {
                "gauge": {"path": str(gauge_path), "sha256": manifest["sources"]["gauge"]["sha256"]},
                "eigenvector": {
                    "path": str(eigenvector_path),
                    "sha256": manifest["sources"]["eigenvector"]["sha256"],
                },
                "point": {"path": str(point_path), "sha256": _sha256(point_path)},
            },
            "execution": {
                "git_commit": args.git_commit,
                "git_dirty": args.git_dirty,
                "logical_test_id": args.logical_test_id,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "script_path": str(Path(__file__).resolve()),
                "script_sha256": _sha256(Path(__file__).resolve()),
            },
        }
        _atomic_json(stage / "result.json", result)
        done = {
            "status": "complete",
            "artifact_sha256": {
                "result.json": _sha256(stage / "result.json"),
                "current-artifact/manifest.json": _sha256(manifest_path),
                f"current-artifact/{manifest['data']['filename']}": manifest["data"]["sha256"],
            },
        }
        _atomic_json(stage / "DONE", done)
        if result_dir.exists():
            raise FileExistsError(f"result directory appeared concurrently: {result_dir}")
        os.rename(stage, result_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        if generator is not None:
            with suppress(Exception):
                generator.clear_loaded_data()
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--L", type=int, default=24)
    parser.add_argument("--T", type=int, default=72)
    parser.add_argument("--available-ne", type=int, default=128)
    parser.add_argument("--used-ne", type=int, default=1)
    parser.add_argument("--point-count", type=int, default=216)
    parser.add_argument("--boundary", choices=("periodic", "open"), default="periodic")
    parser.add_argument("--gauge-prefix", required=True)
    parser.add_argument("--gauge-suffix", default=".lime")
    parser.add_argument("--eigenvector-prefix", required=True)
    parser.add_argument("--eigenvector-suffix", default=".npy")
    parser.add_argument("--point-prefix", required=True)
    parser.add_argument("--point-suffix", default=".npy")
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--git-dirty", action="store_true")
    parser.add_argument("--logical-test-id", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = generate(args)
    except Exception as exc:
        print(f"directed-current generation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "result_dir": str(args.result_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
