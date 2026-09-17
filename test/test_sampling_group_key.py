"""F5.1/F5.2/F5.4: a sampling group is keyed by (declared id, leg time).

The pool partition is the caller's choice, but which legs can possibly share a
coordinate is not: point data is indexed by time, so the same sample index names a
different coordinate on each slice.  Two point ends on different slices therefore
cannot coincide, and asserting that they might is a silent bias rather than an error.

Keying the group by the leg's own time makes both known designs fall out of one
rule: design A (one pool across slices) and design B (per-slice redraw) differ only
in whether the legs land on the same key.  It also removes the split flag as an
input -- the flag was right for B and silently wrong for A.
"""

import numpy as np
import pytest

from lattice import set_backend
from lattice.insertion.current import ConservedVectorCurrent
from lattice.quark_diagram import QuarkDiagram


def _self_loop(leg_offsets=None):
    """One current-capable vertex whose quark line closes on itself.

    The self-loop keeps the group structure readable: both point ends belong to the
    same vertex, so the only thing that can separate them is their own time.

    ``leg_offsets`` has one entry per ``vertex_list`` element, so this diagram takes
    exactly one entry -- a list of that vertex's offset classes.
    """
    return QuarkDiagram(
        [[1]], vertex_list=[1], L=2, usedNp=2, leg_offsets=leg_offsets
    )


def _both_ends_sampled(diagram):
    for state in diagram.expanded_diagrams:
        vertex_state = state._vertex_state[0]
        if vertex_state["left"] == "p" and vertex_state["right"] == "p":
            return state
    raise AssertionError("no (p, p) state was expanded")


def test_both_legs_on_the_anchor_share_one_group():
    """Equal-time: the two ends can coincide, so their patterns are enumerated."""
    set_backend("numpy")
    state = _both_ends_sampled(_self_loop())
    groups = list(state.sampling_groups)

    assert groups == [(1, 0)], f"expected one shared group, got {groups}"
    assert len(state.sampling_groups[(1, 0)]) == 2, "both ends belong to that group"


def test_legs_on_different_slices_land_in_different_groups():
    """A split vertex must not assert a coincidence between its two ends.

    The negative control is the group count: one group means the enumeration will
    produce a "same coordinate" scene for legs that provably cannot share one.
    """
    set_backend("numpy")
    offsets = [[((0, 1), [0])]]
    state = _both_ends_sampled(_self_loop(leg_offsets=offsets))
    groups = sorted(state.sampling_groups)

    assert len(groups) == 2, (
        f"a split vertex's ends must be in separate groups, got {groups}"
    )
    assert all(len(positions) == 1 for positions in state.sampling_groups.values()), (
        "each group holds exactly one leg when the ends are on different slices"
    )


def test_the_offset_decides_the_key_not_a_flag():
    """F5.4: moving the leg changes the key; toggling nothing else changes nothing.

    This is the structural form of "the split flag is not an input".  The same
    diagram with the same offsets must produce the same keys, and changing only the
    offsets must change them.
    """
    set_backend("numpy")
    equal_time = _both_ends_sampled(_self_loop()).sampling_groups
    same_again = _both_ends_sampled(_self_loop()).sampling_groups
    assert list(equal_time) == list(same_again), "keys must be deterministic"

    split = _both_ends_sampled(
        _self_loop(leg_offsets=[[((0, 1), [0])]])
    ).sampling_groups
    assert list(split) != list(equal_time), "changing the offsets must change the keys"


def test_a_single_offset_class_adds_no_axis():
    """One class means every term sits on one slice: it must behave as before.

    The spatial conserved current's two terms both have offset (0, 0), so a vertex
    carrying it is indistinguishable from an equal-time one.
    """
    set_backend("numpy")
    terms = ConservedVectorCurrent(wilson_r=1.0).terms[0:2]
    offsets = [[((0, 0), [0, 1])]]
    state = _both_ends_sampled(_self_loop(leg_offsets=offsets))
    groups = list(state.sampling_groups)

    assert len(groups) == 1, "a single offset class must not separate the legs"


def test_groups_are_independent_across_vertices():
    """Different declared ids stay independent pools, whatever the times are.

    Independence is what makes the weights multiply rather than needing a joint
    inclusion probability, so it must survive the key change.
    """
    set_backend("numpy")
    diagram = QuarkDiagram([[0, 1], [1, 0]], vertex_list=[1, 2], L=2, usedNp=2)
    for state in diagram.expanded_diagrams:
        vertex_state = state._vertex_state
        if vertex_state[0]["left"] == "p" and vertex_state[1]["right"] == "p":
            ids = {key[0] for key in state.sampling_groups}
            assert len(ids) == 2, (
                f"the two vertices use different declared ids, got {ids}"
            )
            break
    else:
        raise AssertionError("no state sampled both vertices")


def test_scene_count_is_the_sum_over_classes_not_the_product():
    """The cost formula needs the scene count *per class*, summed.

    A vertex carrying all eight conserved-current terms has three classes: (0,0) for
    the six spatial terms (both legs on the anchor, so they share one pool and their
    coincidence patterns are enumerated) and (0,1) / (1,0) for the temporal ones
    (legs split, so each end is its own pool and there is no coincidence to
    enumerate).  The correct skeleton count is 2 + 1 + 1 = 4.  Multiplying the
    largest scene count by the class count would give 6, which is why 02's cost
    formula is stated as a sum over class combinations.
    """
    set_backend("numpy")
    terms = ConservedVectorCurrent(wilson_r=1.0).terms
    offsets = [[((0, 0), list(range(6))), ((0, 1), [6]), ((1, 0), [7])]]
    diagram = QuarkDiagram(
        [[1]], vertex_list=[1], L=2, usedNp=2, leg_offsets=offsets
    )
    state = _both_ends_sampled(diagram)

    assert len(state.scene_diagrams) == 4, (
        "scenes must be counted per offset class and summed; a product would give 6"
    )
