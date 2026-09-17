"""F2.1-F2.3, F7.1: time belongs to the leg, and offset classes bound the cost.

The pipeline's whole claim about temporal currents rests on one modelling choice:
a vertex holds a single anchor time, and each leg carries an integer offset from it.
A temporal conserved current is then not a special case at all -- it is a vertex
whose two legs sit on different slices -- and the number of extra contractions is
the number of distinct offset *pairs*, not the number of terms.

These tests pin that, plus the degeneracy that keeps everything else safe: a vertex
whose terms share one offset pair adds no axis, so a meson and an equal-time current
keep the single-contraction arithmetic they had before.
"""

import numpy as np
import pytest

from lattice import set_backend
from lattice.insertion.current import (
    ConservedVectorCurrent,
    LocalVectorCurrent,
    PseudoScalarDensity,
)
from lattice.quark_diagram import (
    QuarkDiagram,
    compute_diagrams_multitime,
    offset_classes,
    vertex_leg_offsets,
)


class _Terms:
    """A vertex handle exposing only the per-term leg placement protocol."""

    def __init__(self, terms, value=10.0):
        self.terms = tuple(terms)
        self.value = value
        self.calls = []

    @property
    def term_count(self):
        return len(self.terms)

    def term_time_offsets(self, term_index):
        mapping = self.terms[term_index].as_dict()
        return int(mapping["bar_offset"][3]), int(mapping["field_offset"][3])

    def is_point_split(self):
        return any(
            self.term_time_offsets(index)[0] != self.term_time_offsets(index)[1]
            for index in range(self.term_count)
        )

    # Sector blocks, all constant so the arithmetic stays checkable by hand.
    def get(self, t):
        self.calls.append(("get", t))
        return np.full((4, 4, 2, 2), self.value, complex)

    def get_v2p(self, t):
        return np.full((4, 4, 2, 2, 3), self.value, complex)

    def get_p2v(self, t):
        return np.full((4, 4, 2, 3, 2), self.value, complex)

    def get_p2p(self, t):
        return np.full((4, 4, 2, 3, 2, 3), self.value, complex)


class _PlainVertex:
    """A meson: no term protocol at all."""

    def get(self, t):
        return np.full((4, 4, 2, 2), 1.0, complex)


class _Propagator:
    """Records every (source, sink) time it is asked for."""

    def __init__(self, value=2.0):
        self.calls = []
        self.value = value

    def get(self, t_source, t_sink):
        self.calls.append((t_source, t_sink))
        return np.full((4, 4, 2, 2), self.value, complex)

    def get_PSV_highmode(self, *args, **kwargs):
        return np.full((4, 4, 2, 3, 2), self.value, complex)

    def get_VSP_highmode(self, *args, **kwargs):
        return np.full((4, 4, 2, 2, 3), self.value, complex)

    def get_PSP_highmode(self, *args, **kwargs):
        return np.full((4, 4, 2, 3, 2, 3), self.value, complex)


def _loop_diagram():
    """A two-vertex loop with no current marking, so no sector expansion runs."""
    return QuarkDiagram([[0, 1], [1, 0]], vertex_list=None, L=2, usedNp=2)


def test_temporal_conserved_current_has_two_offset_classes():
    """01's cost claim: mu=3 has exactly two offset pairs, (0,+1) and (+1,0)."""
    terms = ConservedVectorCurrent(wilson_r=1.0).terms[6:8]
    classes = offset_classes(_Terms(terms))

    assert [offsets for offsets, _ in classes] == [(0, 1), (1, 0)]
    assert [indices for _, indices in classes] == [[0], [1]]


def test_spatial_terms_collapse_into_one_offset_class():
    """Terms sharing an offset pair must be one block, not one axis each.

    This is what makes "ten point-split currents are still only 1024 contractions"
    true, and it is also why the spatial current can travel the equal-time path.
    """
    terms = ConservedVectorCurrent(wilson_r=1.0).terms[0:2]
    classes = offset_classes(_Terms(terms))
    assert [offsets for offsets, _ in classes] == [(0, 0)]
    assert [indices for _, indices in classes] == [[0, 1]]


@pytest.mark.parametrize(
    "current",
    [LocalVectorCurrent(), PseudoScalarDensity()],
)
def test_equal_time_currents_add_no_axis(current):
    """Every term is on one slice, so equality with the pre-existing path holds."""
    classes = offset_classes(_Terms(current.terms))
    assert len(classes) == 1
    assert classes[0][0] == (0, 0)


def test_a_plain_vertex_is_not_term_wise():
    """The predicate is the presence of the protocol, and a meson has none."""
    assert offset_classes(_PlainVertex()) == [((0, 0), [None])]
    assert vertex_leg_offsets(_PlainVertex(), 0) == (0, 0)


def test_a_splitting_vertex_demands_an_explicit_term_choice():
    """Contracting a leg-splitting vertex as a whole is an error, not a guess.

    Summing its terms at one anchor would silently place both legs on the same
    slice, which is a different operator.
    """
    vertex = _Terms(ConservedVectorCurrent(wilson_r=1.0).terms[6:8])
    with pytest.raises(ValueError, match="explicit term choice"):
        vertex_leg_offsets(vertex, 0, None)


def test_leg_times_follow_the_anchor_and_the_offset():
    """The two legs land on t and t+1 for the temporal current, in both directions."""
    vertex = _Terms(ConservedVectorCurrent(wilson_r=1.0).terms[6:8])
    assert vertex_leg_offsets(vertex, 0, 0) == (0, 1)   # bar at t, field at t+1
    assert vertex_leg_offsets(vertex, 0, 1) == (1, 0)   # bar at t+1, field at t


def test_a_split_vertex_requests_both_endpoint_pairs():
    """Each offset class fetches its own propagator times, so the split is real.

    With one splitting vertex the contraction count doubles and the propagator sees
    both endpoint pairs; a single fetch would mean the offsets were computed and
    then ignored.
    """
    set_backend("numpy")
    vertex = _Terms(ConservedVectorCurrent(wilson_r=1.0).terms[6:8])
    propagator = _Propagator()
    values = compute_diagrams_multitime(
        [_loop_diagram()], [0, 3], [_PlainVertex(), vertex], [None, propagator]
    )

    asked = sorted(propagator.calls)
    assert asked == [(0, 3), (0, 4), (3, 0), (4, 0)], (
        f"expected both endpoint pairs, got {asked}"
    )
    assert np.isfinite(np.asarray(values[0])).all()


def test_an_equal_time_vertex_requests_one_endpoint_pair():
    """The negative control for the test above: no split, no extra fetches."""
    set_backend("numpy")
    vertex = _Terms(ConservedVectorCurrent(wilson_r=1.0).terms[0:2])
    propagator = _Propagator()
    compute_diagrams_multitime(
        [_loop_diagram()], [0, 3], [_PlainVertex(), vertex], [None, propagator]
    )
    assert sorted(propagator.calls) == [(0, 3), (3, 0)]


def test_multiple_split_vertices_multiply_the_offset_classes():
    """F2.3: enumerate every class combination, exactly, with no approximation."""
    set_backend("numpy")
    temporal = ConservedVectorCurrent(wilson_r=1.0).terms[6:8]
    left = _Terms(temporal)
    right = _Terms(temporal)
    propagator = _Propagator()
    compute_diagrams_multitime(
        [_loop_diagram()], [0, 3], [left, right], [None, propagator]
    )
    # Two splitting vertices at two classes each: four combinations, two lines each.
    assert len(propagator.calls) == 8


def test_offset_classes_need_a_term_count_or_terms():
    """A vertex implementing the protocol must say how many terms it has."""

    class _Half:
        def term_time_offsets(self, index):
            return 0, 1

    with pytest.raises(TypeError, match="term_count"):
        offset_classes(_Half())


def test_a_scanned_vertex_cannot_also_split_its_legs():
    """F8.2: the conflict is semantic, and must be reported as such.

    A scanned time has no single anchor, so there is nothing for a per-term leg
    offset to be measured from.  Failing here with a reason is the requirement;
    letting it surface later as an einsum rank error is not.
    """
    set_backend("numpy")
    temporal = ConservedVectorCurrent(wilson_r=1.0).terms[6:8]
    equal_time = ConservedVectorCurrent(wilson_r=1.0).terms[0:2]

    with pytest.raises(ValueError, match="both a multitime vertex and a point-split"):
        compute_diagrams_multitime(
            [_loop_diagram()],
            [0, np.arange(4)],
            [_Terms(equal_time), _Terms(temporal)],
            [None, _Propagator()],
        )
