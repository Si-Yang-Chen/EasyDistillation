from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np


from lattice.temporal_slab_io import (
    DEFAULT_TIME_CONVENTION,
    normalized_manifest,
)


MANIFEST_VERSION = 1


def point_propagator_manifest(
    *,
    global_lattice: Iterable[int],
    grid_size: Iterable[int],
    np_src: int,
    np_snk: int,
    dtype: str,
    source_times: Iterable[int] | None = None,
) -> dict:
    global_lattice = [int(value) for value in global_lattice]
    grid_size = [int(value) for value in grid_size]
    return {
        "version": MANIFEST_VERSION,
        "layout": "source-time-rank-slab",
        "product": "PSP",
        "time_convention": DEFAULT_TIME_CONVENTION,
        "global_lattice": global_lattice,
        "grid_size": grid_size,
        "local_lattice": [
            size // grid for size, grid in zip(global_lattice, grid_size)
        ],
        "np_src": int(np_src),
        "np_snk": int(np_snk),
        "source_times": sorted(
            {int(value) for value in source_times or range(global_lattice[3])}
        ),
        "source_set_version": 1,
        "dtype": np.dtype(dtype).str,
        "axis_order": [
            "t_sink_local",
            "spin_sink",
            "spin_source",
            "point_sink",
            "color_sink",
            "point_source",
            "color_source",
        ],
    }


def write_manifest(directory: str | Path, manifest: dict) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    if path.exists():
        existing = normalized_manifest(json.loads(path.read_text()))
        if existing != normalized_manifest(manifest):
            raise ValueError(f"Incompatible point-propagator manifest: {path}")
        return path
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return path


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


class PointPropagatorSlabReader:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.manifest = normalized_manifest(
            json.loads((self.directory / "manifest.json").read_text())
        )
        if self.manifest.get("layout") != "source-time-rank-slab":
            raise ValueError(f"Unsupported PSP layout in {self.directory}")
        self.global_lattice = tuple(self.manifest["global_lattice"])
        self.grid_size = tuple(self.manifest["grid_size"])
        self.local_lattice = tuple(self.manifest["local_lattice"])
        self.np_src = int(self.manifest["np_src"])
        self.np_snk = int(self.manifest["np_snk"])
        self._cache_key = None
        self._cache = None
        self._slab_cache_key = None
        self._slab_cache = None
        self.supports_absolute_times = True

    @property
    def temporal_ranks(self) -> int:
        return self.grid_size[3]

    @property
    def local_time(self) -> int:
        return self.local_lattice[3]

    def _temporal_rank_to_mpi_rank(self, temporal_rank: int) -> int:
        gx, gy, gz, gt = self.grid_size
        spatial_ranks = gx * gy * gz
        if spatial_ranks != 1:
            raise NotImplementedError(
                "PSP slab reader currently requires grid_size[:3] == [1,1,1]"
            )
        return temporal_rank

    def load_source(self, configuration: str | int, source_time: int) -> np.ndarray:
        key = (str(configuration), int(source_time))
        if self._cache_key == key:
            return self._cache
        slabs = []
        for temporal_rank in range(self.temporal_ranks):
            rank = self._temporal_rank_to_mpi_rank(temporal_rank)
            path = rank_slab_path(self.directory, configuration, source_time, rank)
            slab = np.load(path, mmap_mode="r")
            if slab.shape[0] != self.local_time:
                raise ValueError(f"Unexpected local time shape in {path}: {slab.shape}")
            slabs.append(np.asarray(slab))
        self._cache = np.concatenate(slabs, axis=0)
        self._cache_key = key
        return self._cache

    def get(self, configuration: str | int, source_time: int, sink_time):
        sink_times = np.asarray(sink_time, dtype=np.int64) % self.global_lattice[3]
        scalar = sink_times.ndim == 0
        sink_times = np.atleast_1d(sink_times)
        pieces = []
        for sink in sink_times:
            temporal_rank = int(sink) // self.local_time
            local_time = int(sink) % self.local_time
            rank = self._temporal_rank_to_mpi_rank(temporal_rank)
            path = rank_slab_path(self.directory, configuration, source_time, rank)
            cache_key = (str(configuration), int(source_time), int(rank))
            if self._slab_cache_key != cache_key:
                self._slab_cache = np.load(path, mmap_mode="r")
                self._slab_cache_key = cache_key
            pieces.append(np.asarray(self._slab_cache[local_time]))
        result = np.stack(pieces, axis=0)
        return result[0] if scalar else result


class PointPropagatorSlabData:
    def __init__(self, reader: PointPropagatorSlabReader, configuration):
        self.reader = reader
        self.configuration = configuration
        self.supports_absolute_times = True
        T = reader.global_lattice[3]
        self.shape = (
            T,
            T,
            4,
            4,
            reader.np_snk,
            3,
            reader.np_src,
            3,
        )

    def get_absolute(self, source_time: int, sink_time):
        return self.reader.get(self.configuration, source_time, sink_time)


class PointPropagatorSlabFile:
    def __init__(self, directory: str | Path):
        self.reader = PointPropagatorSlabReader(directory)
        self.Np_snk = self.reader.np_snk
        self.Np_src = self.reader.np_src

    def load(self, configuration):
        return PointPropagatorSlabData(self.reader, configuration)
