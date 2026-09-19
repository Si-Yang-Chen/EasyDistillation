#!/usr/bin/env python
"""Generate elemental (distillation elemental) matrix elements with EasyDistillation.

Covers the full branch matrix of both generators:

  * ``ElementalGenerator``   (lattice/generator/elemental.py:38)
      - calc_mode "calc_deriv" (derivative expansion, default) or "calc_disp"
      - num_nabla derivative chain, momentum_list, dilution/is_blending,
        usedNe subsetting, debug
  * ``CurrentElementalGenerator`` (lattice/generator/elemental.py:996)
      - blocks v2v / v2p / p2v / p2p / all (calc_all reuses gauge products)
      - num_nabla displacement chains, momentum_list (v2v only), usedNe/usedNp,
        debug

Outputs follow the on-disk conventions of the preset loaders in
``lattice/preset.py`` so the files can be read back directly:

  elemental (calc_deriv or calc_disp):
      {out_prefix}{key}{out_suffix}   shape (Nblock, num_mom, Lt, usedNe, usedNe) <c16
      Nblock = num_derivative = (3**(num_nabla+1)-1)//2   for calc_deriv
      Nblock = num_disp = list(GaugeLink.nmax_generator(num_nabla))[-1]  for calc_disp
      -> load with ElementalNpy / ElementalBinary (lattice/preset.py:717 / :657)

  current:
      {out_prefix}{key}_v2v.npy    (Lt, num_disp, num_mom, usedNe, usedNe) <c16
                                   -> CurrentElementalV2V (presented as (disp,mom,Lt,Ne,Ne))
      {out_prefix}{key}_v2p.npy    (Lt, num_disp, usedNe, usedNp, Nc) <c16 -> CurrentElementalV2P
      {out_prefix}{key}_p2v.npy    (Lt, num_disp, usedNp, Nc, usedNe) <c16 -> CurrentElementalP2V
      {out_prefix}{key}.t{t:03d}.p2p.npz  per-t sparse dicts -> CurrentElementalP2P

Example (small smoke run; --repo points at your EasyDistillation checkout):
  python gen_elem.py --repo /path/to/repo --generator elemental --latt 4 4 4 8 \
      --gauge-prefix test/ --gauge-suffix .lime --eigen-prefix test/ \
      --eigen-suffix .eigenvector.input.npy --ne 20 --num-nabla 2 \
      --momentum 0,0,0 0,0,1 --timeslices 0 1 --out-prefix out/ --key weak_field
"""
import argparse
import json
import sys
from time import perf_counter


def parse_momentum(s):
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"momentum '{s}' must be px,py,pz")
    return tuple(parts)


def parse_int_list(s):
    return [int(x) for x in s.split(",") if x != ""]


def build_parser():
    p = argparse.ArgumentParser(
        description="Generate elemental / current-elemental data (EasyDistillation).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--repo", default=".", help="EasyDistillation repo root")
    p.add_argument(
        "--backend", choices=["numpy", "cupy"], default="numpy",
        help="compute backend (cupy needs CUDA_VISIBLE_DEVICES set)",
    )
    p.add_argument("--latt", nargs=4, type=int, required=True, metavar=("LX", "LY", "LZ", "LT"),
                   help="lattice size [Lx, Ly, Lz, Lt]")
    p.add_argument("--key", default="10000", help="configuration key appended to prefixes")
    p.add_argument("--generator", choices=["elemental", "current"], default="elemental",
                   help="ElementalGenerator (V^dagger phi V) or CurrentElementalGenerator")

    g = p.add_argument_group("inputs")
    g.add_argument("--gauge-prefix", required=True, help="gauge file prefix")
    g.add_argument("--gauge-suffix", default=".lime", help="gauge file suffix")
    g.add_argument("--eigen-prefix", required=True, help="eigenvector file prefix")
    g.add_argument("--eigen-suffix", default=".npy", help="eigenvector file suffix")
    g.add_argument("--tot-ne", type=int, required=True,
                   help="total number of stored eigenvectors (loader eigenNum)")
    g.add_argument("--point-prefix", default=None,
                   help="point-source prefix (generator=current only)")
    g.add_argument("--point-suffix", default=".npy", help="point-source suffix")

    g = p.add_argument_group("physics / branch selection")
    g.add_argument("--num-nabla", type=int, default=0,
                   help="max derivative/displacement order")
    g.add_argument("--momentum", nargs="+", type=parse_momentum, default=[(0, 0, 0)],
                   metavar="PX,PY,PZ", help="momentum list (elemental calc & current v2v)")
    g.add_argument("--mom-dict", type=int, default=None,
                   help="alternative to --momentum: take momentum_list from "
                        "lattice.insertion.mom_dict.mom_dict_to_list(N)")
    g.add_argument("--used-ne", type=int, default=None,
                   help="number of eigenvectors used (default: all)")
    g.add_argument("--used-np", type=int, default=None,
                   help="number of points used (current only; default: all)")

    g = p.add_argument_group("elemental generator branches")
    g.add_argument("--calc-mode", choices=["calc_deriv", "calc_disp"], default="calc_deriv",
                   help="derivative expansion or GaugeLink displacement encoding")
    g.add_argument("--is-blending", action="store_true",
                   help="build blending stochastic coefficients (requires --dilution-tot)")
    g.add_argument("--dilution-tot", type=parse_int_list, default=None,
                   help="comma list of totNe per noise-vector group")
    g.add_argument("--dilution-used", default=None,
                   help="comma list of usedNe per group, or a single int applied to all")

    g = p.add_argument_group("current generator branches")
    g.add_argument("--calc-blocks", choices=["v2v", "v2p", "p2v", "p2p", "all"],
                   default="all", help="which current elemental blocks to compute/save")
    g.add_argument("--np", type=int, default=None,
                   help="number of sparsened points Np (required for generator=current)")

    g = p.add_argument_group("smearing")
    g.add_argument("--stout-nstep", type=int, default=0,
                   help="stout smearing steps; review against ensemble metadata")
    g.add_argument("--stout-rho", type=float, default=0.12, help="stout rho; review against ensemble metadata")
    g.add_argument("--no-project-su3", action="store_true",
                   help="skip SU(3) projection before smearing")

    g = p.add_argument_group("output")
    g.add_argument("--out-prefix", required=True, help="output file prefix (directory included)")
    g.add_argument("--out-suffix", default=".elemental.npy",
                   help="elemental output suffix (npy mode)")
    g.add_argument("--out-format", choices=["npy", "binary"], default="npy",
                   help="elemental output: .npy (ElementalNpy) or raw <c16 binary (ElementalBinary)")
    g.add_argument("--timeslices", nargs="+", type=int, default=None,
                   help="time slices to compute (default: all)")
    g.add_argument("--debug", action="store_true", help="pass debug=True to the generator")
    g.add_argument("--meta", action="store_true", default=True, help="write a .meta.json sidecar")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    sys.path.insert(0, args.repo)

    from lattice import set_backend, get_backend

    set_backend(args.backend)
    backend = get_backend()

    from lattice import (
        GaugeFieldIldg,
        EigenvectorNpy,
        PointSourceNpy,
        ElementalGenerator,
        CurrentElementalGenerator,
        ElementalNpy,
        Nd,
        Nc,
    )
    from lattice.preset import ElementalBinary
    from lattice.insertion.gauge_link import GaugeLink

    Lx, Ly, Lz, Lt = args.latt
    latt_size = [Lx, Ly, Lz, Lt]
    Ne = args.tot_ne
    usedNe = args.used_ne

    if args.mom_dict is not None:
        from lattice.insertion.mom_dict import mom_dict_to_list

        momentum_list = [tuple(map(int, m.split())) for m in mom_dict_to_list(args.mom_dict)]
        if args.momentum != [(0, 0, 0)]:
            print("WARNING: both --momentum and --mom-dict given; using --mom-dict")
    else:
        momentum_list = list(args.momentum)
    num_mom = len(momentum_list)

    if args.timeslices is None:
        timeslices = list(range(Lt))
    else:
        timeslices = sorted(set(args.timeslices))
        for t in timeslices:
            if not (0 <= t < Lt):
                sys.exit(f"time slice {t} out of range [0, {Lt})")
        if len(timeslices) < Lt:
            print(
                f"WARNING: only timeslices {timeslices} are computed; the output "
                f"file still spans the full Lt={Lt} axis and un-computed slices "
                "are left zero-filled.",
                flush=True,
            )

    gauge_field = GaugeFieldIldg(
        args.gauge_prefix, args.gauge_suffix, [Lt, Lz, Ly, Lx, Nd, Nc, Nc]
    )
    eigenvector = EigenvectorNpy(
        args.eigen_prefix, args.eigen_suffix, [Lt, Ne, Lz, Ly, Lx, Nc], Ne
    )

    dilution = None
    if args.is_blending or args.dilution_tot is not None:
        if args.dilution_tot is None:
            sys.exit("--is-blending requires --dilution-tot")
        tot_list = args.dilution_tot
        if args.dilution_used is None:
            sys.exit("--is-blending requires --dilution-used")
        if "," in str(args.dilution_used):
            used_list = parse_int_list(args.dilution_used)
        else:
            used_list = int(args.dilution_used)  # single int -> replicated in __init__
        dilution = (tot_list, used_list)

    t0 = perf_counter()
    if args.generator == "elemental":
        gen = ElementalGenerator(
            latt_size,
            gauge_field,
            eigenvector,
            num_nabla=args.num_nabla,
            momentum_list=momentum_list,
            dilution=dilution,
            is_blending=args.is_blending or dilution is not None,
            usedNe=usedNe,
            calc_mode=args.calc_mode,
            debug=args.debug,
        )
        num_block = (
            (3 ** (args.num_nabla + 1) - 1) // 2
            if args.calc_mode == "calc_deriv"
            else list(GaugeLink.nmax_generator(args.num_nabla))[-1]
        )
    else:
        if args.point_prefix is None or args.np is None:
            sys.exit("generator=current requires --point-prefix and --np")
        point = PointSourceNpy(args.point_prefix, args.point_suffix, [args.np, Lt, 3], args.np)
        gen = CurrentElementalGenerator(
            latt_size,
            gauge_field,
            eigenvector,
            point,
            num_nabla=args.num_nabla,
            momentum_list=momentum_list,
            usedNe=usedNe,
            usedNp=args.used_np,
            debug=args.debug,
        )
        num_block = list(GaugeLink.nmax_generator(args.num_nabla))[-1]
    usedNe_eff = gen.usedNe
    usedNp_eff = getattr(gen, "usedNp", None)
    print(f"[gen_elem] generator ready in {perf_counter() - t0:.2f}s; "
          f"num_block={num_block} num_mom={num_mom} usedNe={usedNe_eff}", flush=True)

    t0 = perf_counter()
    gen.load(args.key)
    print(f"[gen_elem] loaded gauge+eigen(+point) for key={args.key} "
          f"in {perf_counter() - t0:.2f}s", flush=True)
    if args.stout_nstep > 0:
        t0 = perf_counter()
        if not args.no_project_su3:
            gen.project_SU3()
        gen.stout_smear(args.stout_nstep, args.stout_rho)
        if backend.__name__ == "cupy":
            backend.cuda.get_current_stream().synchronize()
        print(f"[gen_elem] project_SU3 + stout_smear({args.stout_nstep}, {args.stout_rho}) "
              f"in {perf_counter() - t0:.2f}s", flush=True)

    t0 = perf_counter()
    per_t = []
    if args.generator == "elemental":
        out = backend.zeros((Lt, num_block, num_mom, usedNe_eff, usedNe_eff), "<c16")
        for t in timeslices:
            s = perf_counter()
            out[t] = gen.calc(t)
            per_t.append(perf_counter() - s)
        # disk layout: (num_block, num_mom, Lt, Ne, Ne)  [ElementalNpy / ElementalBinary]
        disk = out.transpose(1, 2, 0, 3, 4)
        if args.out_format == "npy":
            path = f"{args.out_prefix}{args.key}{args.out_suffix}"
            backend.save(path, disk)
        else:
            import numpy as _np

            path = f"{args.out_prefix}{args.key}.meson"
            _np.asarray(disk, dtype="<c16").tofile(path)
        print(f"[gen_elem] saved {path} shape={tuple(disk.shape)} dtype={disk.dtype} "
              f"calc+saving {perf_counter() - t0:.1f}s "
              f"(per-t {min(per_t):.2f}-{max(per_t):.2f}s)", flush=True)
        # self-check: read back through the preset loader
        if args.out_format == "npy":
            back = ElementalNpy(
                args.out_prefix, args.out_suffix,
                [num_block, num_mom, Lt, usedNe_eff, usedNe_eff], Ne,
            ).load(args.key)[:]
            print(f"[gen_elem] ElementalNpy readback shape={tuple(backend.asarray(back).shape)}", flush=True)
    else:
        blocks = [args.calc_blocks] if args.calc_blocks != "all" else ["v2v", "v2p", "p2v", "p2p"]
        store = {b: {} for b in blocks}
        for t in timeslices:
            s = perf_counter()
            if args.calc_blocks == "all":
                res = gen.calc_all(t)
                for b in blocks:
                    store[b][t] = res[b]
            else:
                store[args.calc_blocks][t] = getattr(gen, f"calc_{args.calc_blocks}")(t)
            per_t.append(perf_counter() - s)
        if "v2v" in store and store["v2v"]:
            arr = backend.stack([backend.asarray(store["v2v"][t]) for t in timeslices])
            # calc returns (disp, mom, Ne, Ne) per t; disk wants (Lt, disp, mom, Ne, Ne)
            path = f"{args.out_prefix}{args.key}_v2v.npy"
            backend.save(path, arr)
            print(f"[gen_elem] saved {path} shape={tuple(arr.shape)}", flush=True)
        if "v2p" in store and store["v2p"]:
            arr = backend.stack([backend.asarray(store["v2p"][t]) for t in timeslices])
            path = f"{args.out_prefix}{args.key}_v2p.npy"
            backend.save(path, arr)  # (Lt, disp, Ne, Np, Nc)
            print(f"[gen_elem] saved {path} shape={tuple(arr.shape)}", flush=True)
        if "p2v" in store and store["p2v"]:
            arr = backend.stack([backend.asarray(store["p2v"][t]) for t in timeslices])
            path = f"{args.out_prefix}{args.key}_p2v.npy"
            backend.save(path, arr)  # (Lt, disp, Np, Nc, Ne)
            print(f"[gen_elem] saved {path} shape={tuple(arr.shape)}", flush=True)
        if "p2p" in store and store["p2p"]:
            for t in timeslices:
                entries = store["p2p"][t]
                np_savez(
                    f"{args.out_prefix}{args.key}.t{t:03d}.p2p.npz", entries, backend
                )
            print(f"[gen_elem] saved p2p npz for t in {timeslices}", flush=True)
        print(f"[gen_elem] calc+saving {perf_counter() - t0:.1f}s "
              f"(per-t {min(per_t):.3f}-{max(per_t):.3f}s)", flush=True)

    if args.meta:
        meta = {
            "generator": args.generator,
            "calc_mode": args.calc_mode if args.generator == "elemental" else args.calc_blocks,
            "latt_size": latt_size,
            "key": args.key,
            "num_nabla": args.num_nabla,
            "momentum_list": [list(m) for m in momentum_list],
            "num_block": num_block,
            "num_mom": num_mom,
            "tot_ne": Ne,
            "used_ne": usedNe_eff,
            "used_np": usedNp_eff,
            "dilution": [list(dilution[0]), dilution[1]] if dilution else None,
            "stout": {"nstep": args.stout_nstep, "rho": args.stout_rho},
            "timeslices": timeslices,
            "backend": args.backend,
            "gauge_prefix": args.gauge_prefix,
            "eigen_prefix": args.eigen_prefix,
        }
        import numpy as _np

        mpath = f"{args.out_prefix}{args.key}.elemental.meta.json"
        with open(mpath, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[gen_elem] saved {mpath}", flush=True)


def np_savez(path, entries, backend):
    import numpy as np

    to_np = backend.asnumpy if backend.__name__ == "cupy" else (lambda a: np.asarray(a))

    kw = {}
    for i, e in enumerate(entries):
        kw[f"type_{i}"] = np.array(e["type"])
        if e["type"] == "sparse":
            kw[f"indices_{i}"] = to_np(e["indices"]).astype("int32")
            kw[f"values_{i}"] = to_np(e["values"]).astype("<c16")
    np.savez(path, **kw)


if __name__ == "__main__":
    main()
