"""Generate a lattice Laplacian eigensystem with EasyDistillation.

Solves the 3D covariant Laplacian eigensystem for one gauge configuration with
lattice.generator.EigenvectorGenerator and saves the low-mode eigenvectors and
eigenvalues in the on-disk layout expected by the perambulator generators.

Usage:
    # one config, small smoke run
    python gen_eigenvector.py --cfg <cfg> --gauge-dir <gauge-dir> --gauge-prefix <gauge-prefix> \
        --latt-size <Lx>,<Ly>,<Lz>,<Lt> --Ne 8 --tslice-list 0,1 --backend numpy --no-save
    # production
    python gen_eigenvector.py --cfg <cfg> --gauge-dir <gauge-dir> --gauge-prefix <gauge-prefix> \
        --latt-size <Lx>,<Ly>,<Lz>,<Lt> --Ne 120 --tslices 128 --backend numpy \
        --out out/<cfg>.eigenvector.npy --out-eval out/<cfg>.eigenvalue.npy
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a lattice Laplacian eigensystem for one gauge configuration. "
        "Gauge ensemble location and lattice geometry are site/ensemble specific and must be given explicitly."
    )
    p.add_argument("--cfg", required=True, help="config label appended to the gauge prefix (e.g. the configuration number)")
    p.add_argument("--gauge-dir", required=True, help="directory holding the gauge configurations")
    p.add_argument("--gauge-prefix", required=True, help="filename prefix; files are <gauge-dir><gauge-prefix><cfg>.lime")
    p.add_argument("--latt-size", required=True, help="Lx,Ly,Lz,Lt (time axis last)")
    p.add_argument("--Ne", type=int, default=120)
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--nstep", type=int, default=10, help="stout steps; review against ensemble metadata")
    p.add_argument("--rho", type=float, default=0.12, help="stout rho; review against ensemble metadata")
    p.add_argument("--tslices", type=int, default=None, help="number of timeslices to solve (default: all Lt, starting at t=0)")
    p.add_argument("--tslice-list", default=None, help="explicit comma separated timeslices, overrides --tslices")
    p.add_argument("--backend", default="cupy", choices=["numpy", "cupy"])
    p.add_argument("--chebyshev", type=int, default=0, help="polynomial_degree for QUDA Chebyshev acceleration")
    p.add_argument("--project-su3", action="store_true")
    p.add_argument("--no-renorm-phase", action="store_true", help="pass apply_renorm_phase=False to calc")
    p.add_argument("--out", default=None, help="save eigenvectors to this .npy path")
    p.add_argument("--out-eval", default=None, help="save eigenvalues to this .npy path")
    p.add_argument("--no-save", action="store_true", help="write no files even if --out is given")
    p.add_argument("--skip-save-if-exists", action="store_true", help="skip a config whose --out already exists")
    return p.parse_args()


def to_numpy(backend, x):
    """Return a numpy view/copy of x. CuPy exposes asnumpy; NumPy does not."""
    return backend.asnumpy(x) if hasattr(backend, "asnumpy") else np.asarray(x)


def main():
    args = parse_args()

    if args.skip_save_if_exists and args.out and os.path.exists(args.out):
        print(f"[skip] {args.out} already exists; not overwriting")
        return

    from lattice import set_backend, get_backend

    set_backend(args.backend)
    backend = get_backend()

    from lattice import GaugeFieldIldg, EigenvectorGenerator, Nc, Nd

    latt_size = [int(x) for x in args.latt_size.split(",")]
    Lx, Ly, Lz, Lt = latt_size
    if args.tslices is None:
        args.tslices = Lt

    gauge = GaugeFieldIldg(os.path.join(args.gauge_dir, args.gauge_prefix), ".lime", [Lt, Lz, Ly, Lx, Nd, Nc, Nc])
    gen = EigenvectorGenerator(latt_size, gauge, args.Ne, args.tol)

    t0 = time.perf_counter()
    gen.load(args.cfg)
    print(f"[timing] load {args.cfg}: {time.perf_counter() - t0:.2f} s", flush=True)

    t0 = time.perf_counter()
    gen.stout_smear(args.nstep, args.rho)
    print(f"[timing] stout_smear(nstep={args.nstep}, rho={args.rho}): {time.perf_counter() - t0:.2f} s", flush=True)

    if args.project_su3:
        t0 = time.perf_counter()
        gen.project_SU3()
        print(f"[timing] project_SU3(): {time.perf_counter() - t0:.2f} s", flush=True)

    if args.tslice_list:
        tslices = [int(x) for x in args.tslice_list.split(",")]
    else:
        tslices = list(range(args.tslices))
    if not tslices or any(t < 0 or t >= Lt for t in tslices):
        raise ValueError(f"time slices must be within [0, {Lt}); got {tslices}")

    evecs = backend.zeros((Lt, args.Ne, Lz, Ly, Lx, Nc), "<c16")
    evals = backend.zeros((Lt, args.Ne), "<c16")

    renorm = not args.no_renorm_phase
    for t in tslices:
        t0 = time.perf_counter()
        if args.chebyshev > 0:
            lam_cut = float(to_numpy(backend, evals[t, -1].real)) * 1.1 if t > 0 else 0.0
            if lam_cut <= 0.0:
                _, evals[t] = gen.calc(t, renorm, 0, 0.0)
                lam_cut = float(to_numpy(backend, evals[t, -1].real)) * 1.1
            evecs[t], evals[t] = gen.calc(t, renorm, args.chebyshev, lam_cut)
        else:
            evecs[t], evals[t] = gen.calc(t, renorm, 0, 0.0)
        print(f"[timing] calc(t={t}): {time.perf_counter() - t0:.2f} s", flush=True)

    if args.out and not args.no_save:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        backend.save(args.out, evecs.astype("<c16"))
        print(f"[out] saved {args.out} shape={tuple(evecs.shape)} dtype=<c16")
    if args.out_eval and not args.no_save:
        os.makedirs(os.path.dirname(args.out_eval) or ".", exist_ok=True)
        backend.save(args.out_eval, evals.astype("<c16"))
        print(f"[out] saved {args.out_eval} shape={tuple(evals.shape)} dtype=<c16")


if __name__ == "__main__":
    main()
