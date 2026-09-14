from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np


MANIFEST_VERSION = 1
DEFAULT_TIME_CONVENTION = "absolute-global-source-and-sink"


def normalized_manifest(manifest: dict) -> dict:
    normalized = dict(manifest)
    if normalized.get("layout") == "source-time-rank-slab":
        normalized.setdefault("time_convention", DEFAULT_TIME_CONVENTION)
    return normalized


def temporal_slab_manifest(
    *,
    product: str,
    global_lattice: Iterable[int],
    grid_size: Iterable[int],
    tail_shape: Iterable[int],
    dtype: str,
    metadata: dict | None = None,
) -> dict:
    global_lattice = [int(value) for value in global_lattice]
    grid_size = [int(value) for value in grid_size]
    manifest = {
        "version": MANIFEST_VERSION,
        "layout": "source-time-rank-slab",
        "product": product.upper(),
        "time_convention": DEFAULT_TIME_CONVENTION,
        "global_lattice": global_lattice,
        "grid_size": grid_size,
        "local_lattice": [
            size // grid for size, grid in zip(global_lattice, grid_size)
        ],
        "tail_shape": [int(value) for value in tail_shape],
        "dtype": np.dtype(dtype).str,
    }
    manifest.update(metadata or {})
    return manifest


def write_manifest(directory: str | Path, manifest: dict) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    if path.exists():
        existing = normalized_manifest(json.loads(path.read_text()))
        if existing != normalized_manifest(manifest):
            raise ValueError(f"Incompatible temporal-slab manifest: {path}")
        return path
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return path


def json_checksum(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_identity(path: str | Path) -> dict:
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def current_elemental_provenance(
    directory: str | Path, configuration: str | int
) -> list[dict]:
    directory = Path(directory)
    configuration = str(configuration)
    paths = set(directory.glob(f"{configuration}_*"))
    p2p_hdf5 = directory / f"{configuration}_p2p.h5"
    if not p2p_hdf5.exists():
        paths.update(directory.glob(f"{configuration}.t*.p2p.npz"))
    return [file_identity(path) for path in sorted(paths)]


def array_checksum(array) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def configuration_metadata_path(directory: str | Path, configuration) -> Path:
    return Path(directory) / f"{configuration}.metadata.json"


def write_configuration_metadata(directory, configuration, metadata):
    path = configuration_metadata_path(directory, configuration)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != metadata:
            raise ValueError(f"Incompatible configuration metadata: {path}")
        return path
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return path


def validate_compatible_datasets(*directories, configuration=None):
    manifests = [
        normalized_manifest(json.loads((Path(directory) / "manifest.json").read_text()))
        for directory in directories
    ]
    common = (
        "global_lattice",
        "grid_size",
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
    for field in common:
        values = [manifest.get(field) for manifest in manifests]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Incompatible dataset field {field}: {values}")
    if configuration is not None:
        metadata = [
            json.loads(configuration_metadata_path(directory, configuration).read_text())
            for directory in directories
        ]
        checksums = [item.get("point_checksum") for item in metadata]
        if any(checksum != checksums[0] for checksum in checksums[1:]):
            raise ValueError(f"Incompatible point checksums: {checksums}")
        for input_name in ("gauge", "points"):
            identities = [item.get("inputs", {}).get(input_name) for item in metadata]
            present = [identity for identity in identities if identity is not None]
            if present and (
                len(present) != len(identities)
                or any(identity != present[0] for identity in present[1:])
            ):
                raise ValueError(
                    f"Incompatible per-configuration {input_name} inputs: {identities}"
                )
        eigenvector_identities = [
            item.get("inputs", {}).get("eigenvectors") for item in metadata
        ]
        present_eigenvectors = [
            identity for identity in eigenvector_identities if identity is not None
        ]
        if any(
            identity != present_eigenvectors[0]
            for identity in present_eigenvectors[1:]
        ):
            raise ValueError(
                "Incompatible per-configuration eigenvector inputs: "
                f"{eigenvector_identities}"
            )
    return manifests


def collective_rank0_value(communicator, rank: int, operation):
    result = None
    if int(rank) == 0:
        try:
            result = (True, operation())
        except Exception as error:
            result = (False, (type(error).__name__, str(error)))
    success, payload = communicator.bcast(result, root=0)
    if not success:
        error_type, message = payload
        raise RuntimeError(f"Rank-0 setup failed ({error_type}): {message}")
    return payload


def collective_rank0_call(communicator, rank: int, operation) -> None:
    collective_rank0_value(communicator, rank, operation)


def all_ranks_complete(communicator, local_complete: bool) -> bool:
    """Return true only when every MPI rank reports its local slab complete."""
    return all(bool(value) for value in communicator.allgather(bool(local_complete)))


def remaining_output_bytes(
    directory,
    manifest,
    configurations,
    source_times,
):
    directory = Path(directory)
    expected_shape = expected_rank_slab_shape(manifest)
    expected_dtype = manifest["dtype"]
    bytes_per_rank_slab = int(np.prod(expected_shape)) * np.dtype(expected_dtype).itemsize
    temporal_ranks = int(manifest["grid_size"][3])
    global_time = int(manifest["global_lattice"][3])
    required_times = sorted({int(value) % global_time for value in source_times})
    missing = 0
    for configuration in configurations:
        for source_time in required_times:
            for rank in range(temporal_ranks):
                path = rank_slab_path(directory, configuration, source_time, rank)
                if not rank_slab_is_complete(path, expected_shape, expected_dtype):
                    missing += 1
    return missing * bytes_per_rank_slab


def dataset_provenance(
    directory: str | Path,
    configuration: str | int,
    source_times,
) -> dict:
    directory = Path(directory).resolve()
    manifest = normalized_manifest(json.loads((directory / "manifest.json").read_text()))
    metadata = json.loads(
        configuration_metadata_path(directory, configuration).read_text()
    )
    temporal_ranks = int(manifest["grid_size"][3])
    global_time = int(manifest["global_lattice"][3])
    required_times = sorted({int(value) % global_time for value in source_times})
    slabs = [
        file_identity(rank_slab_path(directory, configuration, source_time, rank))
        for source_time in required_times
        for rank in range(temporal_ranks)
    ]
    return {
        "directory": str(directory),
        "manifest_checksum": json_checksum(manifest),
        "configuration_metadata_checksum": json_checksum(metadata),
        "slabs": slabs,
    }


def expected_rank_slab_shape(manifest: dict) -> tuple[int, ...]:
    local_time = int(manifest["local_lattice"][3])
    if "tail_shape" in manifest:
        return (local_time, *(int(value) for value in manifest["tail_shape"]))
    if manifest.get("axis_order") == [
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
    raise ValueError("Manifest does not define a supported rank-slab shape")


def missing_rank_slabs(
    directory: str | Path,
    configuration: str | int,
    source_times,
) -> list[Path]:
    directory = Path(directory)
    manifest = normalized_manifest(json.loads((directory / "manifest.json").read_text()))
    grid_size = manifest["grid_size"]
    if int(np.prod(grid_size[:3])) != 1:
        raise NotImplementedError(
            "Source-time completeness checks require a time-only MPI grid"
        )
    temporal_ranks = int(grid_size[3])
    global_time = int(manifest["global_lattice"][3])
    expected_shape = expected_rank_slab_shape(manifest)
    expected_dtype = manifest["dtype"]
    required_times = sorted({int(source_time) % global_time for source_time in source_times})
    return [
        rank_slab_path(directory, configuration, source_time, rank)
        for source_time in required_times
        for rank in range(temporal_ranks)
        if not rank_slab_is_complete(
            rank_slab_path(directory, configuration, source_time, rank),
            expected_shape,
            expected_dtype,
        )
    ]


def require_rank_slabs(
    directory: str | Path,
    configuration: str | int,
    source_times,
    product: str,
) -> None:
    missing = missing_rank_slabs(directory, configuration, source_times)
    if missing:
        preview = ", ".join(str(path.name) for path in missing[:5])
        remainder = len(missing) - min(len(missing), 5)
        suffix = f" (and {remainder} more)" if remainder else ""
        raise FileNotFoundError(
            f"Missing {len(missing)} required {product} source-time slabs: "
            f"{preview}{suffix}"
        )


def rank_slab_is_complete(path, expected_shape, expected_dtype) -> bool:
    path = Path(path)
    try:
        slab = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = tuple(int(value) for value in expected_shape)
        expected_dtype = np.dtype(expected_dtype)
        return (
            slab.shape == expected_shape
            and slab.dtype == expected_dtype
            and path.stat().st_size == int(slab.offset) + int(slab.nbytes)
        )
    except (OSError, ValueError, TypeError):
        return False


def rank_slab_path(
    directory: str | Path, configuration: str | int, source_time: int, rank: int
) -> Path:
    return Path(directory) / (
        f"{configuration}.t{int(source_time):03d}.rank{int(rank):04d}.npy"
    )


def atomic_save_rank_slab(path: str | Path, array) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with open(temporary, "wb") as output:
        np.save(output, np.asarray(array))
    os.replace(temporary, path)
    return path


class TemporalSlabReader:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.manifest = normalized_manifest(
            json.loads((self.directory / "manifest.json").read_text())
        )
        if self.manifest.get("layout") != "source-time-rank-slab":
            raise ValueError(f"Unsupported slab layout in {self.directory}")
        self.product = self.manifest["product"]
        self.global_lattice = tuple(self.manifest["global_lattice"])
        self.grid_size = tuple(self.manifest["grid_size"])
        self.local_lattice = tuple(self.manifest["local_lattice"])
        self.tail_shape = tuple(self.manifest["tail_shape"])
        self._slab_cache_key = None
        self._slab_cache = None

    @property
    def temporal_ranks(self) -> int:
        return self.grid_size[3]

    @property
    def local_time(self) -> int:
        return self.local_lattice[3]

    def _rank_for_temporal_slab(self, temporal_rank: int) -> int:
        if int(np.prod(self.grid_size[:3])) != 1:
            raise NotImplementedError(
                "TemporalSlabReader currently requires a time-only MPI grid"
            )
        return temporal_rank

    def get_absolute(self, configuration, source_time: int, sink_time):
        sink_times = np.asarray(sink_time, dtype=np.int64) % self.global_lattice[3]
        scalar = sink_times.ndim == 0
        pieces = []
        for sink in np.atleast_1d(sink_times):
            temporal_rank = int(sink) // self.local_time
            local_time = int(sink) % self.local_time
            rank = self._rank_for_temporal_slab(temporal_rank)
            path = rank_slab_path(self.directory, configuration, source_time, rank)
            cache_key = (str(configuration), int(source_time), int(rank))
            if self._slab_cache_key != cache_key:
                self._slab_cache = np.load(path, mmap_mode="r")
                self._slab_cache_key = cache_key
            slab = self._slab_cache
            expected = (self.local_time,) + self.tail_shape
            if slab.shape != expected:
                raise ValueError(f"Unexpected slab shape in {path}: {slab.shape} != {expected}")
            pieces.append(np.asarray(slab[local_time]))
        result = np.stack(pieces, axis=0)
        return result[0] if scalar else result


class TemporalSlabData:
    def __init__(self, reader: TemporalSlabReader, configuration):
        self.reader = reader
        self.configuration = configuration
        self.supports_absolute_times = True
        T = reader.global_lattice[3]
        self.shape = (T, T) + reader.tail_shape

    def get_absolute(self, source_time: int, sink_time):
        return self.reader.get_absolute(self.configuration, source_time, sink_time)


class TemporalSlabFile:
    def __init__(self, directory: str | Path):
        self.reader = TemporalSlabReader(directory)
        manifest = self.reader.manifest
        self.Ne = manifest.get("Ne")
        self.Np = manifest.get("Np")
        self.Np_src = manifest.get("Np_src")
        self.Np_snk = manifest.get("Np_snk")

    def load(self, configuration):
        return TemporalSlabData(self.reader, configuration)
