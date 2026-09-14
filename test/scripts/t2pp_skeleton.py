"""
Build calc_diagram_prepare skeleton for gen_correlator_p1_T2pp (cfg-independent part).
Used by profile_vertex_map_timing_gap.py to generate irrep_vertices.pkl.
"""
from __future__ import annotations

import os
from time import perf_counter

from sympy import S
import sympy as sp

from lattice import preset
from lattice.flavor_structure import HadronFlavorStructure, Tag
from lattice.spatial_structure import HadronIrrep
from lattice.group_projection import hadron_little_group_projection, operator_transform
from lattice.hadron import Hadron, gen_correlator
from lattice.quark_diagram import (
    calc_diagram_prepare,
    remove_disconneted_diagram,
    PropagatorLocal,
    Propagator,
)

GROUP_ELEMENT_LIST_T2PP = ["c4x", "c4y", "c4z"]

DATA_ROOT = os.environ.get(
    "LQCD_DATA_ROOT",
    "/dg_hpc/LQCD/DATA/clqcd_nf2_clov_L16_T128_b2.0_ml-0.05766_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104",
)
LT = 128


def build_propagator_map_t2pp():
    lightperambulator = preset.PerambulatorNpy(
        f"{DATA_ROOT}/03.perambulator.light.single.prec1e-9.nev70/"
        "clqcd_nf2_clov_L16_T128_b2.0_xi5_ml-0.05766_cfg_",
        ".peram.light.nev70.npy",
        [128, 128, 4, 4, 70, 70],
        70,
    )
    charmperambulator = preset.PerambulatorNpy(
        f"{DATA_ROOT}/03.perambulator.charm.single.prec1e-15.nev120/"
        "clqcd_nf2_clov_L16_T128_b2.0_xi5_ml-0.05766_cfg_",
        ".peram.charm.nev120.npy",
        [128, 128, 4, 4, 120, 120],
        120,
    )
    return {
        Rf"S^q_\mathrm{{local}}": PropagatorLocal(lightperambulator, LT),
        "S^c": Propagator(charmperambulator, LT),
        "S^q": Propagator(lightperambulator, LT),
        Rf"S^c_\mathrm{{local}}": PropagatorLocal(charmperambulator, LT),
    }


def build_hadrons_list_t2pp():
    charmed = [HadronFlavorStructure("dc"), HadronFlavorStructure("uc")]
    charmed_bar = [HadronFlavorStructure("cu"), -HadronFlavorStructure("cd")]
    hidden_charm = HadronFlavorStructure("cc")
    iso_scalar = (HadronFlavorStructure("uu") + HadronFlavorStructure("dd")) / sp.sqrt(2)

    DDbar_isoscalar = charmed[0] * charmed_bar[1] - charmed[1] * charmed_bar[0]
    DbarD_isoscalar = -charmed_bar[0] * charmed[1] + charmed_bar[1] * charmed[0]
    I0_Cm = (DDbar_isoscalar - DbarD_isoscalar) * (1 / S(2))
    I0_Cp = (DDbar_isoscalar + DbarD_isoscalar) * (1 / S(2))
    charm_light = hidden_charm * iso_scalar

    chi_c1 = [
        HadronIrrep("eta_c2(0)", [0, 0, 0], "T_2", 1, Tag(0, 0))[0],
        HadronIrrep("eta_c2(1)", [0, 0, 0], "T_2", 1, Tag(0, 0))[0],
        HadronIrrep("eta_c2(2)", [0, 0, 0], "T_2", 1, Tag(0, 0))[0],
    ]

    rows_list = []
    projections = [
        (HadronIrrep("D_star", [0, 0, 0], "T_1", -1, Tag(0, 0)), HadronIrrep("D", [0, 0, 0], "A_1", -1, Tag(0, 0))),
        (HadronIrrep("D_star", [0, 0, 1], "A_1", None, Tag(0, 0)), HadronIrrep("D", [0, 0, -1], "A_2", None, Tag(0, 0))),
        (HadronIrrep("D_star", [0, 0, 1], "E", None, Tag(0, 0)), HadronIrrep("D", [0, 0, -1], "A_2", None, Tag(0, 0))),
        (HadronIrrep("D_star", [0, 1, 1], "A_1", None, Tag(0, 0)), HadronIrrep("D", [0, -1, -1], "A_2", None, Tag(0, 0))),
        (HadronIrrep("D_star", [0, 1, 1], "B_2", None, Tag(0, 0)), HadronIrrep("D", [0, -1, -1], "A_2", None, Tag(0, 0))),
        (HadronIrrep("D_star", [0, 1, 1], "B_1", None, Tag(0, 0)), HadronIrrep("D", [0, -1, -1], "A_2", None, Tag(0, 0))),
        (HadronIrrep("D_star", [1, 1, 1], "A_1", None, Tag(0, 0)), HadronIrrep("D", [1, -1, -1], "A_2", None, Tag(0, 0))),
        (HadronIrrep("D_star", [1, 1, 1], "E", None, Tag(0, 0)), HadronIrrep("D", [1, -1, -1], "A_2", None, Tag(0, 0))),
    ]
    for d_star, d_meson in projections:
        rows_list_tmp = hadron_little_group_projection([d_star, d_meson], "E", 0, parity=None)
        if rows_list_tmp:
            print(d_star, d_meson, len(rows_list_tmp))
            rows_list.extend(rows_list_tmp)

    row_list_Cp = []
    rows_list_jpsi_omega = []

    hadrons_list = (
        [Hadron(row, hidden_charm) for row in chi_c1]
        + [Hadron(row, I0_Cm) for row in rows_list]
        + [Hadron(row, I0_Cp) for row in row_list_Cp]
        + [Hadron(row, charm_light) for row in rows_list_jpsi_omega]
    )
    print(f"[build] nop = {len(hadrons_list)}")
    return hadrons_list


def build_correlator_list_t2pp(hadrons_list, group_element_list=None):
    if group_element_list is None:
        group_element_list = GROUP_ELEMENT_LIST_T2PP
    correlator = gen_correlator([hadrons_list, hadrons_list])
    correlator = remove_disconneted_diagram(correlator, [Rf"S^c_\mathrm{{local}}"])
    t0 = perf_counter()
    correlator_list = [operator_transform(correlator, ge) for ge in group_element_list]
    print(f"[build] operator_transform done: {perf_counter() - t0:.2f}s, n_ge={len(correlator_list)}")
    return correlator_list


def build_prepared_skeleton_t2pp(propegator_map=None, timing=None):
    if propegator_map is None:
        propegator_map = build_propagator_map_t2pp()
    hadrons_list = build_hadrons_list_t2pp()
    correlator_list = build_correlator_list_t2pp(hadrons_list)
    print("[build] calc_diagram_prepare (vertex_map=None, may take long)...")
    t0 = perf_counter()
    prepared = calc_diagram_prepare(
        correlator_list,
        vertex_map=None,
        propagator_map=propegator_map,
        save_dir=None,
        timing=timing,
    )
    print(f"[build] calc_diagram_prepare done: {perf_counter() - t0:.2f}s")
    print(f"[build] n_diagrams={len(prepared.diagram_list)}, n_irrep_vertices={len(prepared.irrep_vertices)}")
    return prepared
