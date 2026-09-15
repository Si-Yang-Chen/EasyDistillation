#!/usr/bin/env python
"""
gen_corr.py -- generic irrep-projected correlator contraction for EasyDistillation.

Pipeline (per config JSON + CLI overrides):
  1. Build interpolating-operator channels: for each channel, one or two hadrons,
     each defined by (name, flavor, momentum, little-group irrep, row, parity).
  2. Two-hadron channels: little-group projection of the hadron-row product onto the
     target irrep/row (`hadron_little_group_projection`); single-hadron channels use
     the hadron's own irrep rows directly.
  3. Assemble the correlator expression: `Hadron` + `gen_correlator` (quark-line
     contraction into `Diagram` symbols), optional disconnected-diagram removal.
  4. Symmetry averaging: `operator_transform` over a list of Oh group elements.
  5. `calc_diagram_prepare` once (cfg- and time-independent skeleton) with the
     propagator_map (quark-line handles).
  6. Per cfg: load every propagator handle, then `calc_diagram_bind` with a
     vertex_map that builds Meson objects from the HadronIrrepRow skeletons
     (Insertion + little-group projection + Operator / OperatorDisplacement +
     Meson.load).
  7. Per t_src: `calc_diagram_eval` with time_map {t_src_symbol: t_src,
     t_snk_symbol: numpy.arange(Lt)}; average over group elements, roll the sink
     axis and accumulate; save per cfg as complex128 npy + meta json.

Usage:
  python gen_corr.py --config corr.json --cfg 2000 --used-ne 20 --backend numpy
Run `python gen_corr.py --print-template` to dump a fully populated config template.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from time import perf_counter


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
GAMMA_ALIASES = ["A0", "B0", "PI", "PI_2", "RHO", "RHO_2", "A1", "B1"]
DERIV_ALIASES = ["IDEN", "NABLA", "B", "D", "E"]


def _named_enums():
    from lattice.insertion import GammaName, DerivativeName

    gamma = {a: getattr(GammaName, a) for a in GAMMA_ALIASES}
    for a in GAMMA_ALIASES:  # also accept the raw LaTeX values used by the library
        gamma[getattr(GammaName, a)] = getattr(GammaName, a)
    deriv = {a: getattr(DerivativeName, a) for a in DERIV_ALIASES}
    for a in DERIV_ALIASES:
        deriv[getattr(DerivativeName, a)] = getattr(DerivativeName, a)
    return gamma, deriv


def get_mom_dict(spec):
    """spec: 'mom1'|'mom3'|'mom9' or an explicit {index: "px py pz"} dict."""
    from lattice.insertion.mom_dict import momDict_mom1, momDict_mom3, momDict_mom9

    if isinstance(spec, str):
        return {"mom1": momDict_mom1, "mom3": momDict_mom3, "mom9": momDict_mom9}[spec]
    return {int(k): v for k, v in spec.items()}


def parse_flavor(expr_str: str):
    """Parse a flavor expression like '(uu+dd)/sqrt(2)' or 'ud*du' into a sympy
    expression over HadronFlavorStructure objects."""
    import sympy as sp
    from lattice.flavor_structure import HadronFlavorStructure

    tokens = {}

    def repl(m):
        tok = m.group(0)
        if tok not in tokens:
            tokens[tok] = HadronFlavorStructure(tok)
        return f"f{list(tokens).index(tok)}"

    s = re.sub(r"bar\{[a-z]{1,3}\}|sqrt|[udsctb]{2,3}", repl, expr_str)
    env = {name: obj for name, obj in
           zip([f"f{i}" for i in range(len(tokens))], tokens.values())}
    env["sqrt"] = sp.sqrt
    return sp.sympify(s, locals=env)


def make_file_loader(spec):
    """Build a lazy preset loader from a {prefix, suffix, shape, tot_ne, format} spec."""
    from lattice import preset

    kind = spec.get("kind", "elemental")
    if kind == "elemental":
        return preset.ElementalNpy(spec["prefix"], spec["suffix"], spec["shape"], spec["tot_ne"])
    if kind == "perambulator":
        cls = preset.PerambulatorNpy if spec.get("format", "npy") == "npy" else preset.PerambulatorBinary
        return cls(spec["prefix"], spec["suffix"], spec["shape"], spec["tot_ne"])
    raise ValueError(f"unknown loader kind {kind}")


def build_channel_rows(channel):
    """Return the list of little-group-projected row expressions for one channel."""
    from lattice.base_types import Tag
    from lattice.spatial_structure import HadronIrrep, HadronIrrepRow
    from lattice.group_projection import hadron_little_group_projection

    specs = channel["hadrons"]
    hadrons = [
        HadronIrrep(
            h["name"],
            list(h.get("momentum", [0, 0, 0])),
            h["irrep"],
            h.get("parity"),
            Tag(0, 0),
        )
        for h in specs
    ]
    if len(hadrons) == 1:
        rows_spec = channel.get("rows")
        if rows_spec is None:
            rows_spec = list(range(hadrons[0].lenth))
        return [hadrons[0][r] for r in rows_spec]
    target = channel["target"]
    rows = hadron_little_group_projection(
        hadrons,
        target["irrep"],
        target.get("row", 0),
        parity=target.get("parity"),
        single_result=channel.get("single_result", False),
    )
    # Guard against trivial projections: hadron_little_group_projection can
    # return a numeric constant (no HadronIrrepRow factors) for e.g. two
    # REST-frame hadrons, which makes quark_contract build a vertex with zero
    # quark lines. Only expressions that carry one HadronIrrepRow factor per
    # hadron are valid interpolators.
    import sympy as sp

    n_had = len(hadrons)

    def _valid(expr):
        expr = sp.sympify(expr).expand()
        terms = sp.Add.make_args(expr)
        if not terms:
            return False
        for term in terms:
            n = sum(1 for f in sp.Mul.make_args(term)
                    if isinstance(f, HadronIrrepRow))
            if n != n_had:
                return False
        return True

    rows = [r for r in rows if _valid(r)]
    if not rows:
        raise ValueError(
            f"channel projects to zero/trivial rows: {channel}. Two-hadron "
            "channels are only supported with non-zero single-hadron momenta "
            "(little-group projection of moving pairs); at-rest pair "
            "projections reduce to constants in this library version."
        )
    return rows


def build_propagator_map(cfg, lt):
    from lattice.quark_diagram import Propagator, PropagatorLocal

    prop_map = {}
    for flavor, pdef in cfg["perambulators"].items():
        peram = make_file_loader({**pdef, "kind": "perambulator"})
        ptype = cfg.get("propagator_types", {}).get(flavor, "nonlocal")
        if ptype in ("nonlocal", "both"):
            prop_map[f"S^{flavor}"] = Propagator(peram, lt)
        if ptype in ("local", "both"):
            prop_map[Rf"S^{flavor}_\mathrm{{local}}"] = PropagatorLocal(peram, lt)
    return prop_map


def make_vertex_map(cfg_key, used_ne, cfg, elementals, mom_dicts, gamma_enum, deriv_enum):
    """Build the cfg-dependent vertex_map used by calc_diagram_bind."""
    from lattice.insertion import Insertion, Operator, OperatorDisplacement
    from lattice.quark_diagram import Meson

    hadron_defs = cfg["hadron_defs"]
    default_mom = mom_dicts[cfg.get("mom_dict", "mom9")]
    cache = {}

    def vertex_map(vertex):
        key = (
            vertex.hadron_name,
            tuple(vertex.momentum),
            vertex.irrep_name,
            vertex.row_idx,
            vertex.parity,
            vertex.dagger,
            used_ne,
            cfg_key,
        )
        if key in cache:
            return cache[key]
        hdef = hadron_defs[vertex.hadron_name]
        if hdef["gamma"] not in gamma_enum:
            raise ValueError(f"unknown gamma alias {hdef['gamma']} for {vertex.hadron_name}")
        mom_dict = mom_dicts.get(hdef.get("mom_dict"), default_mom)
        ins = Insertion(
            gamma_enum[hdef["gamma"]],
            deriv_enum[hdef["derivative"]],
            hdef["rest_irrep"],
            mom_dict,
        )
        ins = ins.little_group_projection(list(vertex.momentum), vertex.irrep_name)
        row = ins[vertex.row_idx](
            vertex.momentum[0], vertex.momentum[1], vertex.momentum[2]
        )
        if hdef.get("distance") is not None:
            op = OperatorDisplacement(
                vertex.hadron_name, [row], [1.0], [int(hdef["distance"])]
            )
        else:
            op = Operator(vertex.hadron_name, [row], [1.0])
        meson = Meson(elementals[hdef.get("elemental", "default")], op, vertex.dagger)
        meson.load(cfg_key, used_ne)
        cache[key] = meson
        return meson

    return vertex_map


def matrix_to_numeric(obj_mat, lt):
    """Convert an (Nop, Nop) object matrix whose entries are (Lt,) arrays or
    scalars into a complex (Nop, Nop, Lt) backend array."""
    from lattice import get_backend

    backend = get_backend()
    n0, n1 = obj_mat.shape
    out = backend.zeros((n0, n1, lt), dtype=backend.complex128)
    for i in range(n0):
        for j in range(n1):
            v = obj_mat[i, j]
            if v is None:
                continue
            arr = backend.asarray(v)
            if backend.ndim(arr) == 0:
                out[i, j, :] = complex(v)
            else:
                out[i, j, :] = arr
    return out


# --------------------------------------------------------------------------- #
# config template (Python dict; booleans are Python bools, serialized by json)
# --------------------------------------------------------------------------- #
TEMPLATE = {
    "lt": 8,
    "elementals": {
        "default": {
            "kind": "elemental",
            "prefix": "test/",
            "suffix": ".elemental.npy",
            "shape": [13, 6, 8, 20, 20],
            "tot_ne": 20,
        },
        "disp": {
            "kind": "elemental",
            "prefix": "test/",
            "suffix": ".displacement_elemental.npy",
            "shape": [9, 6, 8, 20, 20],
            "tot_ne": 20,
        },
    },
    "perambulators": {
        "q": {
            "kind": "perambulator",
            "format": "npy",
            "prefix": "test/",
            "suffix": ".perambulator.npy",
            "shape": [8, 8, 4, 4, 20, 20],
            "tot_ne": 20,
        }
    },
    "propagator_types": {"q": "both"},
    "mom_dict": "mom1",
    "hadron_defs": {
        "pi": {"gamma": "PI", "derivative": "IDEN", "rest_irrep": "A_1", "elemental": "default"},
        "rho": {"gamma": "RHO", "derivative": "IDEN", "rest_irrep": "T_1", "elemental": "default"},
        "pi_disp": {
            "gamma": "PI",
            "derivative": "IDEN",
            "rest_irrep": "A_1",
            "elemental": "disp",
            "distance": 8,
        },
    },
    "channels": [
        {
            "comment": "single hadron at rest: its own irrep rows are used directly",
            "hadrons": [
                {"name": "pi", "flavor": "ud", "momentum": [0, 0, 0],
                 "irrep": "A_1", "parity": -1}
            ],
            "rows": [0],
        },
        {
            "comment": "single hadron built from a displacement elemental (OperatorDisplacement)",
            "hadrons": [
                {"name": "pi_disp", "flavor": "ud", "momentum": [0, 0, 0],
                 "irrep": "A_1", "parity": -1}
            ],
            "rows": [0],
        },
        {
            "comment": "two hadrons at rest, row product little-group-projected to target E row 0",
            "hadrons": [
                {"name": "rho", "flavor": "ud", "momentum": [0, 0, 0],
                 "irrep": "T_1", "parity": -1},
                {"name": "pi", "flavor": "ud", "momentum": [0, 0, 0],
                 "irrep": "A_1", "parity": -1},
            ],
            "target": {"irrep": "E", "row": 0, "parity": None},
            "flavor": "ud*ud",
        },
    ],
    "group_elements": ["iden", "c4x", "c4y", "c4z"],
    "remove_disconnected": [],
    "times": [0, 1],
    "time_roles": ["src", "snk"],
    "dagger": [False, True],
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Irrep-projected correlator contraction (EasyDistillation)."
    )
    p.add_argument("--config", help="JSON config file (see --print-template)")
    p.add_argument("--print-template", action="store_true",
                   help="print a config template and exit")
    p.add_argument("--backend", choices=["auto", "numpy", "cupy"], default="auto")
    p.add_argument("--cfg", action="append", default=[], help="cfg key (repeatable)")
    p.add_argument("--cfg-list", help="file with one cfg key per line (Dispatch-distributed under MPI)")
    p.add_argument("--dispatch-suffix", default=None, help="Dispatch tmp-file suffix")
    p.add_argument("--shard", type=int, default=0, help="shard index for job arrays")
    p.add_argument("--nshards", type=int, default=1, help="number of shards for job arrays")
    p.add_argument("--used-ne", type=int, default=None,
                   help="number of eigenvectors used (Ne truncation applied to every load)")
    p.add_argument("--t-src", default="all", help="comma list of source times, or 'all'")
    p.add_argument("--out-dir", default="out_corr")
    p.add_argument("--out-prefix", default="corr_")
    p.add_argument("--out-suffix", default=".npy")
    p.add_argument("--group-elements", default=None,
                   help="comma list overriding config group_elements")
    p.add_argument("--remove-disconnected", default=None,
                   help="'||'-separated propagator tags overriding config "
                        "remove_disconnected, e.g. \"S^q_||mathrm{local}\" with the "
                        "backslash dropped; prefer setting it in the JSON config")
    p.add_argument("--dry-run", action="store_true", help="stop after calc_diagram_prepare")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip cfgs whose output file already exists")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.print_template:
        print(json.dumps(TEMPLATE, indent=2))
        return 0
    if not args.config:
        print("error: --config is required (see --print-template)", file=sys.stderr)
        return 2
    with open(args.config) as f:
        cfg = json.load(f)
    if args.group_elements is not None:
        cfg["group_elements"] = [g for g in args.group_elements.split(",") if g]
    if args.remove_disconnected is not None:
        cfg["remove_disconnected"] = [g for g in args.remove_disconnected.split("||") if g]

    from lattice.backend import set_backend, get_backend

    if args.backend == "numpy":
        set_backend("numpy")
    elif args.backend == "cupy":
        set_backend("cupy")
    else:  # auto: probe cupy, fall back to numpy
        try:
            set_backend("cupy")
            import cupy

            if cupy.cuda.runtime.getDeviceCount() == 0:
                raise RuntimeError("no CUDA device")
        except Exception:
            set_backend("numpy")
    backend = get_backend()
    print(f"[gen_corr] backend = {backend.__name__}")

    # lazy imports (after backend selection)
    from lattice.flavor_structure import HadronFlavorStructure  # noqa: F401 (docs)
    from lattice.hadron import Hadron, gen_correlator
    from lattice.group_projection import operator_transform
    from lattice.quark_diagram import (
        calc_diagram_prepare,
        calc_diagram_bind,
        calc_diagram_eval,
        remove_disconneted_diagram,
    )

    lt = int(cfg["lt"])
    gamma_enum, deriv_enum = _named_enums()
    mom_dicts = {k: get_mom_dict(k) for k in ("mom1", "mom3", "mom9")}
    if isinstance(cfg.get("mom_dict"), dict):
        mom_dicts["custom"] = get_mom_dict(cfg["mom_dict"])
        cfg["mom_dict"] = "custom"

    # ---- stages 1-3: channels -> Hadron list -> correlator expression -------- #
    t_sym = perf_counter()
    hadrons_src = []
    for ch in cfg["channels"]:
        rows = build_channel_rows(ch)
        flavor_str = ch.get("flavor")
        if flavor_str is None:
            flavor_str = "*".join(h["flavor"] for h in ch["hadrons"])
        flavor = parse_flavor(flavor_str)
        for row in rows:
            hadrons_src.append(Hadron(row, flavor))
    print(f"[gen_corr] Nop = {len(hadrons_src)} ({perf_counter() - t_sym:.2f}s)")

    times = cfg.get("times", [0, 1])
    roles = cfg.get("time_roles", ["src", "snk"])
    dagger = cfg.get("dagger", [False, True])
    t0 = perf_counter()
    correlator = gen_correlator([hadrons_src, list(hadrons_src)], times, dagger)
    gen_s = perf_counter() - t0
    print(f"[gen_corr] gen_correlator done ({gen_s:.2f}s)")

    if cfg.get("remove_disconnected"):
        t0 = perf_counter()
        correlator = remove_disconneted_diagram(correlator, list(cfg["remove_disconnected"]))
        print(f"[gen_corr] remove_disconnected {cfg['remove_disconnected']} "
              f"({perf_counter() - t0:.2f}s)")

    group_elements = cfg.get("group_elements", ["iden"])
    t0 = perf_counter()
    correlator_list = [operator_transform(correlator, ge) for ge in group_elements]
    sym_s = perf_counter() - t0
    print(f"[gen_corr] operator_transform over {len(group_elements)} group elements ({sym_s:.2f}s)")

    # ---- stage 4: prepare (cfg- and time-independent) ------------------------ #
    elementals = {
        name: make_file_loader({**spec, "kind": "elemental"})
        for name, spec in cfg["elementals"].items()
    }
    propagator_map = build_propagator_map(cfg, lt)

    timing = {}
    t0 = perf_counter()
    prepared = calc_diagram_prepare(
        correlator_list,
        vertex_map=None,
        propagator_map=propagator_map,
        save_dir=None,
        timing=timing,
    )
    prepare_s = perf_counter() - t0
    uncovered = [p for p in prepared.all_propagators
                 if p is not None and not hasattr(p, "get")]
    if uncovered:
        raise RuntimeError(
            f"diagrams contain quark lines not covered by propagator_map: {uncovered}. "
            "Add the missing flavor to 'perambulators' or fix 'propagator_types'."
        )
    print(f"[gen_corr] calc_diagram_prepare: n_diagrams={len(prepared.diagram_list)} "
          f"n_irrep_vertices={len(prepared.irrep_vertices)} ({prepare_s:.2f}s)")
    if args.dry_run:
        print("[gen_corr] dry-run: stopping after prepare")
        return 0

    # ---- cfg list ------------------------------------------------------------ #
    if args.cfg_list:
        from lattice.dispatch import Dispatch

        cfgs = list(Dispatch(args.cfg_list, args.dispatch_suffix))
    else:
        cfgs = list(args.cfg)
    if args.nshards > 1:
        cfgs = [c for i, c in enumerate(cfgs) if i % args.nshards == args.shard]
    if not cfgs:
        print("error: no cfg given (use --cfg or --cfg-list)", file=sys.stderr)
        return 2

    # ---- time mapping -------------------------------------------------------- #
    import numpy

    t_snk = numpy.arange(lt)  # MUST be a numpy array (backend arrays break multitime)
    t_src_list = (list(range(lt)) if args.t_src == "all"
                  else [int(t) for t in args.t_src.split(",")])

    os.makedirs(args.out_dir, exist_ok=True)
    for cfg_key in cfgs:
        out_path = f"{args.out_dir}/{args.out_prefix}{cfg_key}{args.out_suffix}"
        if args.skip_existing and os.path.exists(out_path):
            print(f"[gen_corr] skip existing {out_path}")
            continue
        print(f"[gen_corr] === cfg {cfg_key} ===")

        t_load = perf_counter()
        for prop in propagator_map.values():
            prop.load(cfg_key, args.used_ne)
        vertex_map = make_vertex_map(cfg_key, args.used_ne, cfg, elementals,
                                     mom_dicts, gamma_enum, deriv_enum)
        calc_diagram_bind(prepared, vertex_map)
        bind_s = perf_counter() - t_load
        print(f"[gen_corr] propagator load + bind done ({bind_s:.2f}s)")

        t_eval = perf_counter()
        acc = None
        for t_src in t_src_list:
            time_map = {}
            for tsym, role in zip(times, roles):
                time_map[tsym] = t_src if role == "src" else t_snk
            vals = calc_diagram_eval(prepared, time_map)
            mats = [matrix_to_numeric(m, lt) for m in vals]
            avg = mats[0]
            for m in mats[1:]:
                avg = avg + m
            avg = avg / len(mats)
            avg = backend.roll(avg, -t_src, 2)
            acc = avg if acc is None else acc + avg
        eval_s = perf_counter() - t_eval
        acc = acc / len(t_src_list)
        backend.save(out_path, backend.asarray(acc, dtype=backend.complex128))
        print(f"[gen_corr] saved {out_path} shape={tuple(acc.shape)} dtype=complex128 "
              f"(bind {bind_s:.2f}s, eval {eval_s:.2f}s over {len(t_src_list)} t_src)")

        meta = {
            "config": os.path.abspath(args.config),
            "cfg": cfg_key,
            "used_ne": args.used_ne,
            "n_operators": len(hadrons_src),
            "group_elements": group_elements,
            "remove_disconnected": cfg.get("remove_disconnected", []),
            "times": times,
            "time_roles": roles,
            "dagger": dagger,
            "t_src_list": t_src_list,
            "output": os.path.abspath(out_path),
            "output_shape": list(acc.shape),
            "output_dtype": "complex128",
            "n_diagrams": len(prepared.diagram_list),
            "n_irrep_vertices": len(prepared.irrep_vertices),
            "timing": {
                "gen_correlator_s": gen_s,
                "symmetry_transform_s": sym_s,
                "prepare_s": prepare_s,
                "bind_s": bind_s,
                "eval_total_s": eval_s,
                "eval_per_tsrc_s": eval_s / max(1, len(t_src_list)),
                "prepare_stage_timing": {k: v for k, v in timing.items()},
            },
        }
        with open(f"{out_path}.meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
    print("[gen_corr] all done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
