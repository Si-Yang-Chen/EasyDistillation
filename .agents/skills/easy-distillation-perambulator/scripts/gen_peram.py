#!/usr/bin/env python3
"""Generate perambulators with EasyDistillation (PyQUDA/QUDA, GPU required).

Covers all three perambulator generators:
  * PerambulatorGenerator            -> VSV / PSV / PSP
  * DensityPerambulatorGenerator     -> VSSV at fixed tau (per-space-point)
  * GeneralizedPerambulatorGenerator -> VSSV for all sink times

Must run with:  CUDA_VISIBLE_DEVICES=<dev> mpirun -np <nranks> python gen_peram.py ...
Single-GPU jobs use -np 1. Multi-rank jobs hold per-rank partial data and need a
gather step (see SKILL.md).

Input contract (all prefix arguments must end with the directory separator):
  gauge       : QIO/LIME file   {gauge_prefix}{cfg}{gauge_suffix}
                shape [Lt, Lz, Ly, Lx, Nd, Nc, Nc] (GaugeFieldIldg)
  eigenvector : .npy            {eig_*_prefix}{cfg}{eig_*_suffix}
                shape [Lt, Ne, Lz, Ly, Lx, Nc] complex (test-file layout)
  point (opt) : .npy            {point_*_prefix}{cfg}{point_*_suffix}
                shape [Np, Lt, 3] integer coords [x, y, z] (PointSourceNpy)

Output contract:
  --mode vsv  : one .npy, shape [Lt, Lt, Ns, Ns, Ne_snk, Ne_src], <c16
                peramb[t_src, t_rel, s_snk, s_src, e_snk, e_src]
                (axis 1 = (t_snk - t_src) mod Lt; matches PerambulatorNpy)
  --mode psv  : one .npy, shape [Lt, Lt, Ns, Ns, Np_snk, Nc, Ne_src], <c16
                (matches PropagatorPSVNpy full shape)
  --mode psp  : one .npy, shape [Lt, Lt, Ns, Ns, Np_snk, Nc, Np_src, Nc], <c16
                (matches PropagatorPSPNpy full shape)
  --mode density     : shape [N_gamma, N_mom, Lz, Ly, Lx, Ns, Ns, Ne_f, Ne_i]
  --mode generalized : shape [N_gamma, N_mom, Lt, Ns, Ns, Ne_f, Ne_i]
"""

import argparse
import os
import sys
import time

import numpy as np


def parse_multigrid(text):
    """'4 4 4 4 4 4 4 4' -> [[4,4,4,4],[4,4,4,4]]; '' -> None."""
    if text is None or text.strip() in ("", "false", "False", "none", "None"):
        return None
    vals = [int(v) for v in text.replace(",", " ").split()]
    if len(vals) % 4 != 0:
        raise ValueError("--multigrid needs a multiple of 4 ints (one [nx,ny,nz,nt] per level)")
    return [vals[i : i + 4] for i in range(0, len(vals), 4)]


def probe_npy_shape(path):
    """Read only the .npy header; returns shape or None if not a .npy file."""
    if not path.endswith(".npy") or not os.path.isfile(path):
        return None
    from numpy.lib import format as npformat

    with open(path, "rb") as f:
        version = npformat.read_magic(f)
        shape, _, _ = npformat._read_array_header(f, version)
    return tuple(int(s) for s in shape)


def build_parser():
    p = argparse.ArgumentParser(
        description="EasyDistillation perambulator generator (GPU/PyQUDA required)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ed-path", default=os.environ.get("ED_PATH", "."),
                   help="directory containing the `lattice` package (repo root)")
    p.add_argument("--generator", choices=["perambulator", "density", "generalized"],
                   default="perambulator")
    p.add_argument("--cfg", required=True, help="config key substituted into file names")
    # lattice / data locations
    p.add_argument("--latt-size", nargs=4, type=int, required=True, metavar=("LX", "LY", "LZ", "LT"))
    p.add_argument("--gauge-prefix", required=True)
    p.add_argument("--gauge-suffix", default=".lime")
    p.add_argument("--eig-src-prefix", default=None,
                   help="omit together with --mode psp for a pure point-source run")
    p.add_argument("--eig-src-suffix", default=".lime.npy")
    p.add_argument("--eig-src-shape", nargs="+", type=int, default=None,
                   help="eigenvector npy shape override; default probed from the file header")
    p.add_argument("--eig-snk-prefix", default=None)
    p.add_argument("--eig-snk-suffix", default=".lime.npy")
    p.add_argument("--eig-snk-shape", nargs="+", type=int, default=None)
    p.add_argument("--point-src-prefix", default=None)
    p.add_argument("--point-src-suffix", default=".npy")
    p.add_argument("--point-snk-prefix", default=None)
    p.add_argument("--point-snk-suffix", default=".npy")
    p.add_argument("--point-src-np", type=int, default=None, help="usedNp_src; default point_src.Np")
    p.add_argument("--point-snk-np", type=int, default=None, help="usedNp_snk; default point_snk.Np")
    # Dirac operator
    p.add_argument("--mass", type=float, required=True)
    p.add_argument("--tol", type=float, required=True)
    p.add_argument("--maxiter", type=int, required=True)
    p.add_argument("--xi-0", type=float, required=True)
    p.add_argument("--nu", type=float, required=True)
    p.add_argument("--clover-coeff-t", type=float, required=True)
    p.add_argument("--clover-coeff-r", type=float, required=True)
    p.add_argument("--t-boundary", type=int, choices=[1, -1], required=True,
                   help="+1 periodic in t, -1 anti-periodic")
    p.add_argument("--multigrid", type=parse_multigrid, default=None,
                   help="e.g. '4 4 4 4 4 4 4 4' = two levels of [4,4,4,4]; omit to disable")
    p.add_argument("--contract-prec", default="<c16", choices=["<c16", "<c8"])
    # eigenvector truncation / alternates
    p.add_argument("--used-ne-src", type=int, default=None)
    p.add_argument("--used-ne-snk", type=int, default=None)
    p.add_argument("--no-same-eigenvector", action="store_true",
                   help="same_eigenvector=False: use --eig-snk-* as a distinct sink basis")
    # solver layout switches
    p.add_argument("--mRHS", action="store_true", help="QUDA multi-RHS (Ns right-hand sides at once)")
    p.add_argument("--no-vectorized", action="store_true",
                   help="use calc_old instead of calc_new (legacy/reference path)")
    p.add_argument("--products", nargs="+", choices=["vsv", "psv"], default=["vsv", "psv"],
                   help="calc_new products (perambulator mode)")
    # stout smearing (QUDA, in-place on the loaded gauge)
    p.add_argument("--stout-nstep", type=int, default=0, help="stout steps; review against ensemble metadata")
    p.add_argument("--stout-rho", type=float, default=0.1, help="stout rho; review against ensemble metadata")
    p.add_argument("--stout-dir-ignore", type=int, default=3)
    # what to compute
    p.add_argument("--t-src", nargs="+", type=int, default=None,
                   help="source times; default all LT (perambulator mode)")
    p.add_argument("--mode", choices=["vsv", "psv", "psp", "density", "generalized"],
                   default="vsv", help="output product")
    p.add_argument("--tau", type=int, default=None, help="fixed sink time for --mode density")
    p.add_argument("--ti", type=int, default=0, help="initial source time (density/generalized)")
    p.add_argument("--tf", type=int, default=None, help="final source time (density/generalized)")
    p.add_argument("--gamma-list", nargs="+", type=int, default=[i for i in range(16)],
                   help="gamma bitmask list for density/generalized (n in [0,15])")
    p.add_argument("--momentum-list", nargs="+", type=int, action="append", default=None,
                   help="repeatable triple, e.g. --momentum-list 0 0 0 --momentum-list 1 0 0")
    # output
    p.add_argument("--out", required=True, help="output .npy path (single file)")
    p.add_argument("--readback-check", action="store_true",
                   help="reload the VSV output with PerambulatorNpy and compare")
    p.add_argument("--quda-verbose", action="store_true")
    return p


def momentum_triples(args):
    if not args.momentum_list:
        return [(0, 0, 0)]
    out = []
    for chunk in args.momentum_list:
        if len(chunk) != 3:
            raise ValueError("each --momentum-list needs exactly 3 ints")
        out.append(tuple(chunk))
    return out


def make_eigenvector_handle(cls, prefix, suffix, shape_override, cfg):
    if prefix is None:
        return None
    probe = probe_npy_shape(f"{prefix}{cfg}{suffix}")
    shape = tuple(shape_override) if shape_override else probe
    if shape is None:
        raise ValueError(f"cannot determine eigenvector shape for {prefix}*{suffix}; "
                         "pass --eig-*-shape")
    Ne = shape[1]
    print(f"eigenvector {prefix}*{suffix}: shape={shape} Ne={Ne}", flush=True)
    return cls(prefix, suffix, list(shape), Ne)


def main():
    args = build_parser().parse_args()
    sys.path.insert(0, os.path.abspath(args.ed_path))

    from lattice import set_backend, check_QUDA  # noqa: E402

    set_backend("cupy")
    if not check_QUDA():
        raise ImportError("PyQUDA unavailable: run under mpirun with CUDA_VISIBLE_DEVICES set")
    from lattice import (
        PerambulatorGenerator,
        DensityPerambulatorGenerator,
        GeneralizedPerambulatorGenerator,
        GaugeFieldIldg,
        EigenvectorNpy,
        PointSourceNpy,
        PerambulatorNpy,
        Nc,
        Ns,
        Nd,
    )  # noqa: E402
    backend_str = "cupy"

    Lx, Ly, Lz, Lt = args.latt_size

    try:
        from pyquda_utils.core import getMPIRank, getMPISize
    except Exception:
        def getMPISize():
            return 1

    gauge_field = GaugeFieldIldg(
        args.gauge_prefix, args.gauge_suffix, [Lt, Lz, Ly, Lx, Nd, Nc, Nc]
    )
    eig_src = make_eigenvector_handle(
        EigenvectorNpy, args.eig_src_prefix, args.eig_src_suffix, args.eig_src_shape, args.cfg
    )
    eig_snk = make_eigenvector_handle(
        EigenvectorNpy, args.eig_snk_prefix, args.eig_snk_suffix, args.eig_snk_shape, args.cfg
    )
    point_src = (
        PointSourceNpy(args.point_src_prefix, args.point_src_suffix)
        if args.point_src_prefix else None
    )
    point_snk = (
        PointSourceNpy(args.point_snk_prefix, args.point_snk_suffix)
        if args.point_snk_prefix else None
    )

    if args.generator != "perambulator" and (args.no_same_eigenvector or args.used_ne_snk):
        print("NOTE: same_eigenvector/eigenvector_snk only exist on PerambulatorGenerator; ignored")

    common = dict(
        latt_size=args.latt_size,
        gauge_field=gauge_field,
        mass=args.mass,
        tol=args.tol,
        maxiter=args.maxiter,
        xi_0=args.xi_0,
        nu=args.nu,
        clover_coeff_t=args.clover_coeff_t,
        clover_coeff_r=args.clover_coeff_r,
        t_boundary=args.t_boundary,
        multigrid=args.multigrid,
    )

    t_start_all = time.perf_counter()
    if args.generator == "perambulator":
        # same_eigenvector=True reuses the source dagger array for the sink;
        # its row count is Ne_src, so usedNe_snk must match or the VSV buffer
        # (sized by the file Ne) is wider than the contraction result.
        used_ne_snk = args.used_ne_snk
        if used_ne_snk is None and not args.no_same_eigenvector and args.used_ne_src is not None:
            used_ne_snk = args.used_ne_src
        gen = PerambulatorGenerator(
            contract_prec=args.contract_prec,
            eigenvector_src=eig_src,
            usedNe_src=args.used_ne_src,
            eigenvector_snk=eig_snk,
            usedNe_snk=used_ne_snk,
            point_src=point_src,
            usedNp_src=args.point_src_np,
            point_snk=point_snk,
            usedNp_snk=args.point_snk_np,
            MRHS=args.mRHS,
            use_vectorized=not args.no_vectorized,
            same_eigenvector=not args.no_same_eigenvector,
            **common,
        )
        if args.quda_verbose:
            from pyquda import enum_quda

            gen.dirac.invert_param.verbosity = enum_quda.QudaVerbosity.QUDA_SUMMARIZE
    elif args.generator == "density":
        gen = DensityPerambulatorGenerator(
            eigenvector=eig_src,
            gamma_list=args.gamma_list,
            momentum_list=momentum_triples(args),
            **common,
        )
    else:
        gen = GeneralizedPerambulatorGenerator(
            eigenvector=eig_src,
            gamma_list=args.gamma_list,
            momentum_list=momentum_triples(args),
            **common,
        )

    gen.load(args.cfg)
    if args.stout_nstep > 0:
        t0 = time.perf_counter()
        gen.stout_smear(args.stout_nstep, args.stout_rho, args.stout_dir_ignore)
        print(f"stout_smear: {time.perf_counter() - t0:.2f} s", flush=True)

    # ---- compute ------------------------------------------------------------
    if args.generator == "perambulator":
        t_src_list = args.t_src or list(range(Lt))
        if args.mode == "vsv":
            products = tuple(
                p.upper() for p in args.products
                if (p.upper() == "VSV" and gen._VSV is not None)
                or (p.upper() == "PSV" and gen._PSV is not None)
            )
            if not products:
                raise ValueError("no VSV/PSV product available with the given inputs")
            out = np.zeros((Lt, Lt, Ns, Ns, gen.Ne_snk, gen.Ne_src), args.contract_prec)
            for t_src in t_src_list:
                t0 = time.perf_counter()
                vsv, _ = gen.calc_new(t_src, products=products)
                if vsv is not None:
                    out[t_src] = np.roll(getattr(vsv, "get", lambda: vsv)(), -t_src, 0)
                m = gen.last_metrics
                print(f"t_src={t_src}: {time.perf_counter() - t0:.2f} s "
                      f"(inv={m.get('inversion_seconds', float('nan')):.2f} s)", flush=True)
        elif args.mode == "psv":
            out = np.zeros((Lt, Lt, Ns, Ns, gen.Np_snk, Nc, gen.Ne_src), args.contract_prec)
            for t_src in t_src_list:
                t0 = time.perf_counter()
                _, psv = gen.calc_new(t_src, products=("PSV",))
                out[t_src] = np.roll(getattr(psv, "get", lambda: psv)(), -t_src, 0)
                print(f"t_src={t_src}: {time.perf_counter() - t0:.2f} s", flush=True)
        elif args.mode == "psp":
            out = np.zeros((Lt, Lt, Ns, Ns, gen.Np_snk, Nc, gen.Np_src, Nc), args.contract_prec)
            for t_src in t_src_list:
                t0 = time.perf_counter()
                psp = gen.calc_point_sources(t_src)
                out[t_src] = np.roll(getattr(psp, "get", lambda: psp)(), -t_src, 0)
                print(f"t_src={t_src}: {time.perf_counter() - t0:.2f} s", flush=True)
        else:
            raise ValueError(f"--mode {args.mode} requires --generator perambulator")
    else:
        if args.tf is None:
            raise ValueError("--tf is required for density/generalized")
        if args.mode == "density":
            if args.tau is None:
                raise ValueError("--tau is required for --mode density")
            t0 = time.perf_counter()
            out = np.asarray(gen.calc(args.ti, args.tf, args.tau))
            print(f"calc(ti={args.ti}, tf={args.tf}, tau={args.tau}): "
                  f"{time.perf_counter() - t0:.2f} s", flush=True)
        elif args.mode == "generalized":
            t0 = time.perf_counter()
            out = np.asarray(gen.calc(args.ti, args.tf))
            print(f"calc(ti={args.ti}, tf={args.tf}): {time.perf_counter() - t0:.2f} s", flush=True)
        else:
            raise ValueError(f"--mode {args.mode} requires --generator perambulator")

    if getMPISize() > 1:
        print("WARNING: MPI size > 1: per-rank partial data saved; "
              "use pyquda_utils.core.gatherLattice(..., reduce_op='sum') for the full "
              "result (see test/test_perambulator_mpi.py)", flush=True)
    np.save(args.out, out)
    print(f"saved {args.out} shape={out.shape} dtype={out.dtype}", flush=True)

    if args.readback_check and args.mode == "vsv":
        directory, fname = os.path.split(os.path.abspath(args.out))
        if fname.startswith(args.cfg):
            suffix, cfg = fname[len(args.cfg):], args.cfg
        else:
            cfg, suffix = os.path.splitext(fname)[0], os.path.splitext(fname)[1]
        ref = PerambulatorNpy(directory + os.sep, suffix, list(out.shape), int(out.shape[-1]))
        data = ref.load(cfg)[:]
        if hasattr(data, "get"):
            data = data.get()
        diff = np.linalg.norm(np.asarray(data) - out)
        print(f"readback check ({cfg}{suffix}): shape={data.shape} dtype={data.dtype} "
              f"|delta|={diff:.3e}", flush=True)

    print(f"total: {time.perf_counter() - t_start_all:.2f} s", flush=True)


if __name__ == "__main__":
    main()
