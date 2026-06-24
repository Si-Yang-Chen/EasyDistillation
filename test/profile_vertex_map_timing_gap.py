#!/usr/bin/env python
"""
Profile vertex_map / calc_diagram_bind timing gap (wall vs summed sub-timers).

Usage (on machine with elemental data):
  cd EasyDistillation

  # Only build irrep_vertices.pkl (same as gen_correlator_p1_T2pp prepare, slow):
  python test/profile_vertex_map_timing_gap.py --only-build-irrep-pkl

  # Profile with full T2pp vertex list (auto-build pkl if missing):
  python test/profile_vertex_map_timing_gap.py --use-t2pp --cfg 4460

  # Quick test with 10 vertices from pkl:
  python test/profile_vertex_map_timing_gap.py --use-t2pp --cfg 4460 --n-sample 10

  # Full T2pp replica: prepare + one cfg bind, then exit (same code as production):
  python gen_correlator_p1_T2pp.py --cfg 4460 --stop-after-bind

Exits after all diagnostic tests complete.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import defaultdict
from time import perf_counter

sys.path.insert(0, os.path.dirname(__file__))

# Heavy imports deferred until GPU tests (after --skip-gpu check in main).

DEFAULT_IRREP_PKL = os.path.join(os.path.dirname(__file__), "irrep_vertices_t2pp.pkl")

DATA_ROOT = os.environ.get(
    "LQCD_DATA_ROOT",
    "/dg_hpc/LQCD/DATA/clqcd_nf2_clov_L16_T128_b2.0_ml-0.05766_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104",
)
ELEMENTAL_PREFIX = os.path.join(DATA_ROOT, "04.meson.deriv2.mom9.nev120/")
ELEMENTAL_DISP_PREFIX = os.path.join(DATA_ROOT, "04.meson.disp2.mom3.nev120/")


def cuda_synchronize():
    from lattice import get_backend

    backend = get_backend()
    if hasattr(backend, "cuda"):
        backend.cuda.Device().synchronize()


def _new_stats():
    return defaultdict(float)


def _print_stats(title, stats, wall_key="bind_wall"):
    wall = stats[wall_key]
    _COUNT_KEYS = frozenset(
        ("cache_hit", "cache_miss", "load_calls", "n_vertices", "n_diagrams",
         "n_unique_time_vertex_pairs", "n_unique_propagators")
    )
    print(f"\n=== {title} ===")
    print(f"  {wall_key}: {wall:.4f}s")
    skip = {wall_key, "n_vertices"}
    sub_sum = 0.0
    for k in sorted(stats):
        if k in skip:
            continue
        v = stats[k]
        if k in _COUNT_KEYS or (isinstance(v, int) and not isinstance(v, bool)):
            print(f"  {k}: {v}")
            continue
        if isinstance(v, float):
            sub_sum += v
            pct = 100.0 * v / wall if wall > 0 else 0.0
            print(f"  {k}: {v:.4f}s ({pct:.1f}%)")
    gap = wall - sub_sum
    print(f"  sub_timers_sum (time only): {sub_sum:.4f}s")
    print(f"  gap (wall - sub_sum): {gap:.4f}s ({100.0 * gap / wall if wall > 0 else 0:.1f}%)")
    return gap


def test_accounting_mock():
    """Sanity: gap reporting math."""
    print("\n[test 1] accounting mock")
    stats = _new_stats()
    stats["bind_wall"] = 100.0
    stats["meson_load"] = 50.0
    stats["little_group_projection"] = 1.0
    gap = _print_stats("mock", stats)
    assert abs(gap - 49.0) < 1e-6, "mock gap should be 49s"
    print("  PASS")


def build_and_save_irrep_vertices_pkl(pkl_path, force=False):
    """
    Same cfg-independent pipeline as gen_correlator_p1_T2pp before the cfg loop:
      hadrons -> gen_correlator -> operator_transform -> calc_diagram_prepare(vertex_map=None)
    """
    if os.path.exists(pkl_path) and not force:
        print(f"[build] skip: {pkl_path} already exists (use --build-irrep-pkl to overwrite)")
        return

    from lattice import set_backend

    set_backend("cupy")
    from t2pp_skeleton import build_prepared_skeleton_t2pp

    prepare_timing = {}
    prepared = build_prepared_skeleton_t2pp(timing=prepare_timing)
    irrep_vertices = list(prepared.irrep_vertices)
    with open(pkl_path, "wb") as f:
        pickle.dump(irrep_vertices, f)
    meta_path = pkl_path + ".meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "n_irrep_vertices": len(irrep_vertices),
                "n_diagrams": len(prepared.diagram_list),
                "prepare_timing": dict(prepare_timing) if prepare_timing else {},
            },
            f,
        )
    print(f"[build] saved {len(irrep_vertices)} irrep_vertices -> {os.path.abspath(pkl_path)}")
    if prepare_timing:
        print(f"[build] prepare wall keys: {', '.join(sorted(prepare_timing))}")


def resolve_irrep_vertices(args):
    """Load or build irrep_vertices according to CLI flags."""
    if args.use_t2pp or args.irrep_pkl:
        pkl_path = args.irrep_pkl or DEFAULT_IRREP_PKL
        if args.build_irrep_pkl or not os.path.exists(pkl_path):
            build_and_save_irrep_vertices_pkl(pkl_path, force=args.build_irrep_pkl)
        with open(pkl_path, "rb") as f:
            irrep_vertices = pickle.load(f)
        print(f"Loaded {len(irrep_vertices)} vertices from {os.path.abspath(pkl_path)}")
        return irrep_vertices

    irrep_vertices = make_sample_vertices()
    print(f"Using {len(irrep_vertices)} built-in sample vertices (pass --use-t2pp for production vertex list)")
    return irrep_vertices


def make_sample_vertices():
    from lattice.spatial_structure import HadronIrrepRow, Tag

    return [
        HadronIrrepRow("eta_c2(0)", [0, 0, 0], "T_2", 0, -1, Tag(0, 0)),
        HadronIrrepRow("eta_c2(1)", [0, 0, 0], "T_2", 1, -1, Tag(0, 0)),
        HadronIrrepRow("D", [0, 0, 0], "A_1", 0, -1, Tag(0, 0)),
        HadronIrrepRow("D_star", [0, 0, 1], "A_1", 0, None, Tag(0, 0)),
        HadronIrrepRow("D", [0, 0, -1], "A_2", 0, None, Tag(0, 0)),
    ]


def load_elemental_objects():
    from lattice import preset

    elemental = preset.ElementalNpy(
        ELEMENTAL_PREFIX,
        ".stout.n20.f0.12.deriv2.mom9.nev120.npy",
        [13, 123, 128, 120, 120],
        120,
    )
    elemental_displacement = preset.ElementalNpy(
        ELEMENTAL_DISP_PREFIX,
        ".npy",
        [13, 123, 128, 120, 120],
        120,
    )
    return elemental, elemental_displacement


def make_production_vertex_map(cfg, usedNe, elemental, elemental_displacement, prod_detail):
    from lattice.insertion.mom_dict import momDict_mom9, momDict_mom3
    from lattice.insertion import Insertion, Operator, GammaName, DerivativeName
    from lattice.spatial_structure import HadronIrrepRow
    from lattice.quark_diagram import Meson

    meson_cache = {}

    def vertex_map(vertex):
        if not isinstance(vertex, HadronIrrepRow):
            raise ValueError(f"unexpected vertex type: {type(vertex)}")

        cache_key = (
            vertex.hadron_name,
            tuple(vertex.momentum),
            vertex.irrep_name,
            vertex.row_idx,
            vertex.parity,
            vertex.dagger,
            usedNe,
            cfg,
        )
        if cache_key in meson_cache:
            prod_detail["cache_hit"] += 1
            return meson_cache[cache_key]
        prod_detail["cache_miss"] += 1

        mom_dict = momDict_mom9
        if vertex.hadron_name in ("omega", "J/Psi", "D_star"):
            gamma, rest_frame_irrep, derivative = GammaName.RHO, "T_1", DerivativeName.IDEN
        elif vertex.hadron_name == "D":
            gamma, rest_frame_irrep, derivative = GammaName.PI, "A_1", DerivativeName.IDEN
        elif vertex.hadron_name.startswith("chi_c1"):
            gamma, rest_frame_irrep, derivative = GammaName.A1, "T_1", DerivativeName.IDEN
            mom_dict = momDict_mom3
        elif vertex.hadron_name.startswith(("chi_c2", "eta_c2")):
            gamma, rest_frame_irrep, derivative = (
                (GammaName.RHO, "T_2", DerivativeName.NABLA)
                if vertex.hadron_name.startswith("chi_c2")
                else (GammaName.B1, "T_2", DerivativeName.NABLA)
            )
        elif vertex.hadron_name.startswith("eta_c1"):
            gamma, rest_frame_irrep, derivative = GammaName.RHO, "T_1", DerivativeName.B
        else:
            raise ValueError(f"Unknown hadron: {vertex.hadron_name}")

        t0 = perf_counter()
        ins = Insertion(gamma, derivative, rest_frame_irrep, mom_dict).little_group_projection(
            vertex.momentum, vertex.irrep_name
        )
        prod_detail["little_group_projection"] += perf_counter() - t0

        t0 = perf_counter()
        op = Operator(
            vertex.hadron_name,
            [ins[vertex.row_idx](vertex.momentum[0], vertex.momentum[1], vertex.momentum[2])],
            [1],
        )
        if vertex.hadron_name.startswith("chi_c1"):
            for i in range(len(op.parts[1])):
                op.parts[1][i][1] = int(vertex.hadron_name[-2])
            elem = elemental_displacement
        else:
            elem = elemental
        meson = Meson(elem, op, vertex.dagger)
        prod_detail["meson_setup"] += perf_counter() - t0

        t0 = perf_counter()
        meson.load(cfg, usedNe)
        prod_detail["meson_load"] += perf_counter() - t0
        meson_cache[cache_key] = meson
        return meson

    return vertex_map


class InstrumentedMesonLoad:
    """Temporarily patch Meson.load / _make_cache for fine-grained timing."""

    def __init__(self, sync_after_each_load: bool = False, sync_deferred_at_end: bool = False):
        self.sync_after = sync_after_each_load
        self.sync_deferred = sync_deferred_at_end
        self.stats = _new_stats()
        self._Meson = None
        self._orig_load = None
        self._orig_make_cache = None
        self._patched_load = None
        self._patched_make_cache = None

    def __enter__(self):
        from lattice.quark_diagram import Meson

        self._Meson = Meson
        self._orig_load = Meson.load
        self._orig_make_cache = Meson._make_cache
        instr = self

        def timed_make_cache(meson_self):
            from lattice import get_backend
            from lattice.insertion.gamma import gamma
            from opt_einsum import contract

            backend = get_backend()
            t_slice = perf_counter()
            cache = meson_self.cache
            parts = meson_self.operator.parts
            ret_gamma = []
            ret_elemental = []
            for i in range(len(parts) // 2):
                ret_gamma.append(gamma(parts[i * 2]))
                elemental_part = parts[i * 2 + 1]
                for j in range(len(elemental_part)):
                    elemental_coeff, derivative_idx, momentum_idx, profile = elemental_part[j]
                    elemental_coeff = complex(elemental_coeff)
                    deriv_mom_tuple = (derivative_idx, momentum_idx)
                    if deriv_mom_tuple not in cache:
                        cache[deriv_mom_tuple] = meson_self.elemental_data[
                            derivative_idx, momentum_idx, :, : meson_self.usedNe, : meson_self.usedNe
                        ]
                    if j == 0:
                        ret_elemental.append(elemental_coeff * cache[deriv_mom_tuple])
                    else:
                        ret_elemental[-1] += elemental_coeff * cache[deriv_mom_tuple]
            instr.stats["make_cache_slice"] += perf_counter() - t_slice

            t_contract = perf_counter()
            if meson_self.dagger:
                meson_self.cache = (
                    contract(
                        "ik,xlk,lj->xij",
                        gamma(8),
                        backend.asarray(ret_gamma).conj(),
                        gamma(8),
                    ),
                    contract("xtba->xtab", backend.asarray(ret_elemental).conj()),
                )
            else:
                meson_self.cache = (
                    backend.asarray(ret_gamma),
                    backend.asarray(ret_elemental),
                )
            instr.stats["make_cache_contract"] += perf_counter() - t_contract

        def timed_load(meson_self, key, usedNe=None):
            t_outer = perf_counter()
            meson_self.usedNe = usedNe
            if meson_self.key != key:
                t0 = perf_counter()
                meson_self._release_resources()
                instr.stats["release_resources"] += perf_counter() - t0

                meson_self.key = key
                t0 = perf_counter()
                meson_self.elemental_data = meson_self.elemental.load(key)
                instr.stats["elemental_open"] += perf_counter() - t0

                meson_self.cache = {}
                t0 = perf_counter()
                timed_make_cache(meson_self)
                instr.stats["make_cache_total"] += perf_counter() - t0

            if instr.sync_after:
                t0 = perf_counter()
                cuda_synchronize()
                instr.stats["cuda_sync_per_load"] += perf_counter() - t0

            instr.stats["load_wall"] += perf_counter() - t_outer
            instr.stats["load_calls"] += 1

        self._patched_load = timed_load
        self._patched_make_cache = timed_make_cache
        Meson.load = timed_load
        Meson._make_cache = timed_make_cache
        return self

    def __exit__(self, *args):
        if self._Meson is not None:
            self._Meson.load = self._orig_load
            self._Meson._make_cache = self._orig_make_cache


def run_bind_profile(
    irrep_vertices,
    cfg,
    usedNe,
    elemental,
    elemental_displacement,
    title,
    sync_after_each=False,
    sync_at_end=False,
):
    from lattice import get_backend
    from lattice.quark_diagram import calc_diagram_bind, _CalcDiagramPrepared

    prod_detail = defaultdict(float)
    vertex_map = make_production_vertex_map(
        cfg, usedNe, elemental, elemental_displacement, prod_detail
    )

    prepared = _CalcDiagramPrepared(
        expr=None,
        diagram_list=[],
        combined_diagrams=[],
        all_vertices=list(irrep_vertices),
        all_propagators=[],
        all_times=list(range(len(irrep_vertices))),
        irrep_vertices=list(irrep_vertices),
        save_dir=None,
        debug=False,
        backend=get_backend(),
    )

    instr = InstrumentedMesonLoad(sync_after_each_load=sync_after_each)
    with instr:
        t_wall = perf_counter()
        bind_detail = {}
        calc_diagram_bind(prepared, vertex_map, timing=bind_detail)
        if sync_at_end:
            t0 = perf_counter()
            cuda_synchronize()
            instr.stats["cuda_sync_at_end"] = perf_counter() - t0
        instr.stats["bind_wall"] = perf_counter() - t_wall

    instr.stats["n_vertices"] = len(irrep_vertices)
    prod_detail["vertex_map_total"] = bind_detail.get("vertex_map_total", instr.stats["bind_wall"])

    print(f"\n[production timers] {title}")
    _print_stats(f"production ({title})", prod_detail, wall_key="vertex_map_total")

    print(f"\n[instrumented Meson.load] {title}")
    gap_instr = _print_stats(f"instrumented ({title})", instr.stats, wall_key="bind_wall")

    prod_sum = (
        prod_detail["little_group_projection"]
        + prod_detail["meson_setup"]
        + prod_detail["meson_load"]
    )
    prod_gap = prod_detail["vertex_map_total"] - prod_sum
    print(f"\n[gap diagnosis] {title}")
    print(f"  production meson_load sum: {prod_detail['meson_load']:.4f}s")
    print(f"  instrumented load_wall sum: {instr.stats['load_wall']:.4f}s")
    print(f"  production gap (wall - lgp - setup - meson_load): {prod_gap:.4f}s")
    print(f"  instrumented load_wall vs meson_load: {instr.stats['load_wall'] - prod_detail['meson_load']:.4f}s")
    if instr.stats.get("cuda_sync_per_load", 0) > 0:
        print(f"  cuda_sync_per_load (explicit): {instr.stats['cuda_sync_per_load']:.4f}s")
    if instr.stats.get("cuda_sync_at_end", 0) > 0:
        print(f"  cuda_sync_at_end (deferred): {instr.stats['cuda_sync_at_end']:.4f}s")

    ratio = prod_detail["vertex_map_total"] / prod_detail["meson_load"] if prod_detail["meson_load"] > 0 else 0
    print(f"  wall/meson_load ratio: {ratio:.3f} (expect ~2.0 if GPU sync is deferred)")

    return prod_detail, instr.stats, prod_gap


def test_single_meson_repeated(cfg, usedNe, elemental, n_repeat=3):
    from lattice.insertion.mom_dict import momDict_mom9
    from lattice.insertion import Insertion, Operator, GammaName, DerivativeName
    from lattice.spatial_structure import HadronIrrepRow, Tag
    from lattice.quark_diagram import Meson

    print(f"\n[test 4] single meson x{n_repeat} (D at rest)")
    vertex = HadronIrrepRow("D", [0, 0, 0], "A_1", 0, -1, Tag(0, 0))
    ins = Insertion(GammaName.PI, DerivativeName.IDEN, "A_1", momDict_mom9).little_group_projection(
        vertex.momentum, vertex.irrep_name
    )
    op = Operator(
        vertex.hadron_name,
        [ins[vertex.row_idx](0, 0, 0)],
        [1],
    )
    meson = Meson(elemental, op, vertex.dagger)

    for label, sync_after in [("no_sync", False), ("sync_each", True)]:
        instr = InstrumentedMesonLoad(sync_after_each_load=sync_after)
        with instr:
            for _ in range(n_repeat):
                meson.load(cfg, usedNe)
        _print_stats(f"single meson {label}", instr.stats, wall_key="load_wall")


def main():
    parser = argparse.ArgumentParser(description="Profile vertex_map timing gap")
    parser.add_argument("--cfg", type=str, default="4460", help="cfg id for meson.load")
    parser.add_argument("--used-ne", type=int, default=70)
    parser.add_argument("--n-sample", type=int, default=0, help="max vertices to test (0=all, default all)")
    parser.add_argument(
        "--irrep-pkl",
        type=str,
        default=None,
        help="pickle path for irrep_vertices (with --use-t2pp defaults to test/irrep_vertices_t2pp.pkl)",
    )
    parser.add_argument(
        "--use-t2pp",
        action="store_true",
        help="build/load T2pp production vertex list (gen_correlator_p1_T2pp skeleton)",
    )
    parser.add_argument(
        "--build-irrep-pkl",
        action="store_true",
        help="force rebuild irrep pkl before profiling",
    )
    parser.add_argument(
        "--only-build-irrep-pkl",
        action="store_true",
        help="only run calc_diagram_prepare and save pkl, then exit",
    )
    parser.add_argument("--skip-gpu", action="store_true", help="only run mock test")
    args = parser.parse_args()

    test_accounting_mock()

    if args.skip_gpu:
        print("\nAll tests done (--skip-gpu).")
        sys.exit(0)

    if args.only_build_irrep_pkl:
        build_and_save_irrep_vertices_pkl(args.irrep_pkl or DEFAULT_IRREP_PKL, force=True)
        print("\nBuild-only mode done.")
        sys.exit(0)

    from lattice import set_backend

    set_backend("cupy")

    elemental_path = f"{ELEMENTAL_PREFIX}{args.cfg}.stout.n20.f0.12.deriv2.mom9.nev120.npy"
    if not os.path.exists(elemental_path):
        print(f"\nSKIP GPU tests: elemental not found at {elemental_path}")
        print("Set LQCD_DATA_ROOT or run on HPC with data.")
        sys.exit(0)

    elemental, elemental_displacement = load_elemental_objects()

    irrep_vertices = resolve_irrep_vertices(args)

    if args.n_sample > 0:
        irrep_vertices = irrep_vertices[: args.n_sample]
    print(f"Profiling {len(irrep_vertices)} vertices, cfg={args.cfg}")

    # Test 2: production timers (matches gen_correlator_p1_T2pp)
    run_bind_profile(
        irrep_vertices,
        args.cfg,
        args.used_ne,
        elemental,
        elemental_displacement,
        "no explicit cuda sync",
        sync_after_each=False,
        sync_at_end=False,
    )

    # Test 3: explicit sync after each meson.load
    run_bind_profile(
        irrep_vertices,
        args.cfg,
        args.used_ne,
        elemental,
        elemental_displacement,
        "cuda sync after each load",
        sync_after_each=True,
        sync_at_end=False,
    )

    # Test 3b: deferred sync at end of bind only
    run_bind_profile(
        irrep_vertices,
        args.cfg,
        args.used_ne,
        elemental,
        elemental_displacement,
        "cuda sync once at end",
        sync_after_each=False,
        sync_at_end=True,
    )

    test_single_meson_repeated(args.cfg, args.used_ne, elemental)

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("  production gap = vertex_map_total - (lgp + setup + meson_load)")
    print("  Compare calc_diagram_bind wall vs sum(meson_load) in gap diagnosis above.")
    print("  If wall/meson_load ~ 2 at full vertex count, deferred CuPy sync is likely.")
    print("=" * 60)
    print("\nAll tests done.")
    sys.exit(0)


if __name__ == "__main__":
    main()
