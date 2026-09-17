"""F7.2: the prepared skeleton keeps every (sector x scene) unit.

``calc_diagram_prepare`` rebuilds each expression's diagrams onto a shared vertex
index space.  It used to rebuild one diagram per input and drop the current
vertex expansion on the way, so a graph whose vertices can be point-sampled
reached evaluation as its ``vv`` sector alone -- four sectors in, one out, with no
error.  Splitting the graph and forgetting why it was split is the kind of defect
that looks like a plausible number.

These tests pin the count and the coefficients, and pin that the mixed sectors
reach evaluation rather than merely existing in the expansion.
"""

import numpy as np
import pytest

from lattice import set_backend
from lattice.base_types import Tag
from lattice.flavor_structure import HadronFlavorStructure
from lattice.hadron import Hadron, gen_correlator
from lattice.quark_diagram import (
    Diagram,
    QuarkDiagram,
    calc_diagram_bind,
    calc_diagram_eval,
    calc_diagram_prepare,
)
from lattice.scene_coefficients import scene_coefficients
from lattice.spatial_structure import HadronIrrep

NE, NP = 2, 2
L = 2                      # spatial extent, so M = L**3 = 8


def _time_axis(t_source, t_sink, block):
    """Mimic the real accessors: an array time prepends a leading time axis.

    Either end may be the scanned one, so take whichever is an array; when both are
    scalars the block is returned as is.
    """
    for time in (t_source, t_sink):
        if not isinstance(time, (int, np.integer)):
            return np.broadcast_to(block, (len(time),) + block.shape).copy()
    return block


class _Propagator:
    """Block accessor answering every sector's request with a recognisable value."""

    def __init__(self):
        self.calls = []

    def get(self, t_source, t_sink):
        self.calls.append("VSV")
        block = np.full((4, 4, NE, NE), 2.0, complex)
        return _time_axis(t_source, t_sink, block)

    def get_PSV_highmode(self, t_source, t_sink, **kwargs):
        self.calls.append("PSV")
        block = np.full((4, 4, NP, 3, NE), 3.0, complex)
        return _time_axis(t_source, t_sink, block)

    def get_VSP_highmode(self, t_source, t_sink, **kwargs):
        self.calls.append("VSP")
        block = np.full((4, 4, NE, NP, 3), 5.0, complex)
        return _time_axis(t_source, t_sink, block)

    def get_PSP_highmode(self, t_source, t_sink, *args):
        self.calls.append("PSP")
        block = np.full((4, 4, NP, 3, NP, 3), 7.0, complex)
        return _time_axis(t_source, t_sink, block)


class _Vertex:
    """Vertex handle answering every sector's request."""

    def __init__(self):
        self.calls = []

    def get(self, t):
        self.calls.append("get")
        return _time_axis(t, t, np.full((4, 4, NE, NE), 2.0, complex))

    def get_v2p(self, t):
        self.calls.append("get_v2p")
        return _time_axis(t, t, np.full((4, 4, NE, NP, 3), 3.0, complex))

    def get_p2v(self, t):
        self.calls.append("get_p2v")
        return _time_axis(t, t, np.full((4, 4, NP, 3, NE), 5.0, complex))

    def get_p2p(self, t):
        self.calls.append("get_p2p")
        return _time_axis(t, t, np.full((4, 4, NP, 3, NP, 3), 7.0, complex))


def _marked_pion_diagram():
    """A two-vertex pion diagram with vertex 1 declared current-capable."""
    row = HadronIrrep("pi", [0, 0, 0], "A_1", -1, Tag(0, 0))[0]
    expression = gen_correlator(
        [[Hadron(row, HadronFlavorStructure("ud"))]] * 2, [0, 1], [False, True]
    )
    original = list(expression[0, 0].atoms(Diagram))[0]
    marked = QuarkDiagram(
        original.diagram.adjacency_matrix, vertex_list=[0, 1], L=L, usedNp=NP
    )
    return Diagram(
        marked, original.time_list, original.vertex_list, original.propagator_list
    )


def test_scene_count_is_the_bell_weighted_sector_sum():
    """Four sectors contribute 1 + 1 + 1 + Bell(2) = 5 scene contractions.

    Losing the expansion collapses this to 1: the state list was the only place
    the sectors lived, and rebuilding the graph discarded it.
    """
    set_backend("numpy")
    prepared = calc_diagram_prepare(
        [_marked_pion_diagram()], propagator_map={"S^q": _Propagator()}
    )

    assert len(prepared.diagram_list) == 1, "one distinct Diagram in the expression"
    assert len(prepared.combined_diagrams) == 5, (
        "every (sector x scene) unit must survive prepare"
    )
    assert prepared.scene_ranges == [(0, 5)], "all scenes belong to the one diagram"


def test_scene_coefficients_survive_prepare():
    """The coefficients carried into evaluation are the solved ones, not w(k).

    On this graph the states are vv, vp, pv and pp.  The first three have a single
    point-free or single-point group and use w(1) = M/Np = 8/2 = 4; pp has one group
    with two point ends and uses the r=2 solution [-24, 28], which is the zeta
    inversion of w_E = [4, 28].  The old code would have produced [28, 4] here.
    """
    set_backend("numpy")
    prepared = calc_diagram_prepare(
        [_marked_pion_diagram()], propagator_map={"S^q": _Propagator()}
    )
    expected = scene_coefficients(2, L**3, NP)
    assert [int(value) for value in expected] == [-24, 28]

    coefficients = [int(value) for value in prepared.scene_coefficients]
    assert coefficients == [1, 4, 4, -24, 28]


def test_no_current_diagram_stays_a_single_unit_with_unit_coefficient():
    """F7.1's guard: a plain diagram is one contraction at coefficient 1, so the
    equal-time two-point function cannot move."""
    set_backend("numpy")
    row = HadronIrrep("pi", [0, 0, 0], "A_1", -1, Tag(0, 0))[0]
    expression = gen_correlator(
        [[Hadron(row, HadronFlavorStructure("ud"))]] * 2, [0, 1], [False, True]
    )
    prepared = calc_diagram_prepare(
        [expression], propagator_map={"S^q": _Propagator()}
    )

    assert len(prepared.combined_diagrams) == 1
    assert [float(value) for value in prepared.scene_coefficients] == [1.0]


def test_mixed_sectors_reach_evaluation():
    """The point sectors must actually be contracted, not merely exist.

    The negative control is the accessor log: with the expansion dropped only
    ``VSV`` is ever asked for, which is exactly how the defect hid.  This two-vertex
    topology has one quark line per sector, so the ``pp`` state asks for PSV and VSP
    (one point end each) rather than PSP; PSP needs both ends point-sampled on one
    line, which a single-line loop cannot supply.
    """
    set_backend("numpy")
    propagator = _Propagator()
    prepared = calc_diagram_prepare(
        [_marked_pion_diagram()], propagator_map={"S^q": propagator}
    )

    vertices = {v.hadron_name: _Vertex() for v in prepared.irrep_vertices}
    calc_diagram_bind(prepared, lambda v: vertices[v.hadron_name])
    result = calc_diagram_eval(prepared, {0: 0, 1: np.arange(LT := 2)})

    used = set(propagator.calls)
    assert "VSV" in used, "the low-mode sector must still be contracted"
    assert {"PSV", "VSP"} <= used, (
        f"point sectors never reached the propagator; only {sorted(used)} asked for"
    )
    vertex_calls = {call for vertex in vertices.values() for call in vertex.calls}
    assert {"get", "get_v2p", "get_p2v"} <= vertex_calls

    # `result` mirrors the input expression's structure (one entry per operator
    # combination), with the times axis last; entries are backend arrays or nested
    # objects, so flatten to plain numbers before checking.
    import numpy as _np

    def _flatten(value):
        if isinstance(value, _np.ndarray):
            for item in value.ravel():
                yield from _flatten(item)
        else:
            yield value

    numbers = [complex(item) for item in _flatten(result[0])]
    assert numbers, "evaluation returned nothing"
    assert all(_np.isfinite(item) for item in numbers)


def test_four_sector_sum_differs_from_the_low_mode_sector_alone():
    """F7.2's teeth: the answer must move when the mixed sectors are included.

    If summing every scene happened to equal the vv-only value, the check that the
    sectors survive would be vacuous.
    """
    set_backend("numpy")
    from lattice.quark_diagram import compute_diagrams_multitime

    propagator = _Propagator()
    marked = _marked_pion_diagram()
    vertices = [_Vertex(), _Vertex()]
    every_scene = compute_diagrams_multitime(
        [marked], [0, np.arange(2)], vertices, [None, propagator]
    )

    plain = Diagram(
        QuarkDiagram(marked.diagram.adjacency_matrix, vertex_list=None, L=L, usedNp=NP),
        marked.time_list,
        marked.vertex_list,
        marked.propagator_list,
    )
    low_mode_only = compute_diagrams_multitime(
        [plain], [0, np.arange(2)], [_Vertex(), _Vertex()], [None, _Propagator()]
    )

    assert len(every_scene) == 5 and len(low_mode_only) == 1
    assert not np.isclose(np.sum(every_scene), np.sum(low_mode_only)), (
        "the four-sector sum equals the vv-only value, so the check is vacuous"
    )
