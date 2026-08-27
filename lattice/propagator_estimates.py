from __future__ import annotations

import math

from .constant import Nc, Ns


def propagator_storage_estimate(
    *,
    global_time: int,
    local_time: int,
    eigenvectors: int,
    source_points: int,
    sink_points: int,
    dtype_bytes: int = 16,
):
    vsv_local = local_time * Ns * Ns * eigenvectors * eigenvectors * dtype_bytes
    psv_local = (
        local_time
        * Ns
        * Ns
        * sink_points
        * Nc
        * eigenvectors
        * dtype_bytes
    )
    psp_local = (
        local_time
        * Ns
        * Ns
        * sink_points
        * Nc
        * source_points
        * Nc
        * dtype_bytes
    )
    temporal_ranks = math.ceil(global_time / local_time)
    return {
        "local_bytes": {"VSV": vsv_local, "PSV": psv_local, "PSP": psp_local},
        "configuration_bytes": {
            "VSV": vsv_local * temporal_ranks * global_time,
            "PSV": psv_local * temporal_ranks * global_time,
            "PSP": psp_local * temporal_ranks * global_time,
        },
        "per_source_global_bytes": {
            "VSV": vsv_local * temporal_ranks,
            "PSV": psv_local * temporal_ranks,
            "PSP": psp_local * temporal_ranks,
        },
        "temporal_ranks": temporal_ranks,
    }


def total_output_bytes(per_source_global_bytes, source_time_count, configuration_count):
    source_times = int(source_time_count)
    configurations = int(configuration_count)
    if source_times < 0 or configurations < 0:
        raise ValueError("source-time and configuration counts must be nonnegative")
    return int(per_source_global_bytes) * source_times * configurations


def solver_buffer_estimate(
    *,
    local_lattice,
    eigenvectors,
    source_points,
    sink_points,
    eigen_rhs_count=1,
    point_rhs_count=1,
    dtype_bytes=16,
):
    Lx, Ly, Lz, Lt = [int(value) for value in local_lattice]
    volume = Lx * Ly * Lz * Lt
    sv = volume * Ns * Ns * Nc * dtype_bytes if eigenvectors else 0
    point_fermion = volume * Ns * Nc * dtype_bytes
    eigen_solver_fields = (
        2 * point_fermion * int(eigen_rhs_count) if eigenvectors else 0
    )
    point_solver_fields = (
        2 * point_fermion * int(point_rhs_count) if source_points else 0
    )
    products = propagator_storage_estimate(
        global_time=Lt,
        local_time=Lt,
        eigenvectors=eigenvectors,
        source_points=source_points,
        sink_points=sink_points,
        dtype_bytes=dtype_bytes,
    )["local_bytes"]
    return {
        "SV": sv,
        "SP": 0,
        "eigen_solver_fields": eigen_solver_fields,
        "point_solver_fields": point_solver_fields,
        "VSV": products["VSV"],
        "PSV": products["PSV"],
        "PSP": products["PSP"],
        "eigen_path_total": sv + eigen_solver_fields + products["VSV"] + products["PSV"],
        "point_path_total": point_solver_fields + products["PSP"],
    }
