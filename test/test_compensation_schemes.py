"""F6: the compensation scheme is one explicit right-hand side, chosen globally.

Step 0 showed the scene coefficient solves ``Z^T c = w``, which means the two
compensation schemes differ *only* in ``w`` -- so the fork is a right-hand side, not a
second code path, and switching schemes must not change the scene structure.

Scheme E (expected) divides by the number of k-tuples a draw is expected to contain and
is design-unbiased for a fixed gauge field.  Scheme O (observed) divides by the number
it actually contains; being a ratio it is biased in general, but it does not assume
equal-probability sampling.  Which to use is the caller's bias-versus-variance call.

Everything here is checked through the graph, because that is where the choice is made
and where a wrong denominator would actually be used.
"""

import pytest

from lattice import set_backend
from lattice.quark_diagram import QuarkDiagram
from lattice.scene_coefficients import CompensationScheme, SCHEMES


def _self_loop(**kwargs):
    """One current-capable vertex whose quark line closes on itself.

    Both point ends belong to the same vertex, so the group structure is readable and
    the only variable is the scheme.
    """
    return QuarkDiagram([[1]], vertex_list=[1], L=2, usedNp=2, **kwargs)


def _both_ends_sampled(diagram):
    for state in diagram.expanded_diagrams:
        vertex_state = state._vertex_state[0]
        if vertex_state["left"] == "p" and vertex_state["right"] == "p":
            return state
    raise AssertionError("no (p, p) state was expanded")


def _falling(Np, k):
    """The ordered count of distinct k-tuples, matching the falling factorials."""
    total = 1
    for step in range(k):
        total *= Np - step
    return total


def test_the_two_schemes_are_the_two_named_values():
    assert set(SCHEMES) == {CompensationScheme.EXPECTED, CompensationScheme.OBSERVED}


def test_expected_is_the_default():
    """Existing callers must keep today's arithmetic without naming a scheme."""
    set_backend("numpy")
    implicit = _both_ends_sampled(_self_loop()).scene_weights
    explicit = _both_ends_sampled(
        _self_loop(compensation=CompensationScheme.EXPECTED)
    ).scene_weights
    assert list(implicit) == list(explicit)


def test_observed_matches_expected_when_counts_equal_expectation():
    """With every point distinct inside one slice the two schemes coincide.

    This is the production case, and it is why the acceptance test has to say the two
    schemes *should* agree here: a divergence would mean the count source is wrong, not
    that the scheme took effect.
    """
    set_backend("numpy")

    def counts(positions, block, M, Np):
        return _falling(Np, len(block))

    expected = _both_ends_sampled(_self_loop()).scene_weights
    observed = _both_ends_sampled(
        _self_loop(compensation=CompensationScheme.OBSERVED, count_source=counts)
    ).scene_weights
    assert list(expected) == list(observed)


def test_a_count_below_expectation_moves_the_coefficients():
    """The negative control for the test above: the count source must be read."""
    set_backend("numpy")

    def half_counts(positions, block, M, Np):
        return max(1, _falling(Np, len(block)) // 2)

    expected = _both_ends_sampled(_self_loop()).scene_weights
    observed = _both_ends_sampled(
        _self_loop(compensation=CompensationScheme.OBSERVED, count_source=half_counts)
    ).scene_weights
    assert list(expected) != list(observed), (
        "halving the observed counts must change the coefficients"
    )


def test_zero_count_is_undefined_not_small():
    """F6.4: a term that did not enter the sample has no defined weight."""
    set_backend("numpy")

    def no_counts(positions, block, M, Np):
        return 0

    with pytest.raises(ValueError, match="undefined"):
        _self_loop(compensation=CompensationScheme.OBSERVED, count_source=no_counts)


def test_observed_needs_a_count_source():
    """F6.5: the pipeline cannot invent the count; only the data has it."""
    set_backend("numpy")
    with pytest.raises(ValueError, match="count source"):
        _self_loop(compensation=CompensationScheme.OBSERVED)


def test_an_unknown_scheme_is_refused():
    set_backend("numpy")
    with pytest.raises(ValueError, match="unknown compensation scheme"):
        _self_loop(compensation="something-else")


def test_the_scheme_does_not_change_the_scene_structure():
    """The schemes are a denominator, not an expansion axis.

    Same scenes, same partitions, same constraints -- only the coefficients move.  If
    this failed, the fork would have become structural and the cost formula would need
    a fourth axis.
    """
    set_backend("numpy")

    def half_counts(positions, block, M, Np):
        return max(1, _falling(Np, len(block)) // 2)

    expected = _both_ends_sampled(_self_loop())
    observed = _both_ends_sampled(
        _self_loop(compensation=CompensationScheme.OBSERVED, count_source=half_counts)
    )

    assert len(expected.scene_diagrams) == len(observed.scene_diagrams)
    assert expected.scene_constraints == observed.scene_constraints
    assert list(expected.sampling_groups) == list(observed.sampling_groups)


def test_mixing_schemes_in_one_expression_is_refused():
    """F6.3: one run, one scheme.

    Two diagrams of the same expression could otherwise carry different denominators,
    which would sum terms with different statistical characters into one number that no
    single provenance record could describe.
    """
    set_backend("numpy")
    from lattice.base_types import Tag
    from lattice.flavor_structure import HadronFlavorStructure
    from lattice.hadron import Hadron, gen_correlator
    from lattice.quark_diagram import Diagram, calc_diagram_prepare
    from lattice.spatial_structure import HadronIrrep

    # "cc" produces three diagrams (a connected loop plus the local and non-local
    # disconnected pieces), which is what makes it possible to disagree on scheme.
    row = HadronIrrep("pi", [0, 0, 0], "A_1", -1, Tag(0, 0))[0]
    expression = gen_correlator(
        [[Hadron(row, HadronFlavorStructure("cc"))]] * 2, [0, 1], [False, True]
    )
    diagrams = list(expression[0, 0].atoms(Diagram))

    # Rebuild the two diagram symbols so their inner diagrams disagree on the scheme.
    def with_scheme(diagram, scheme):
        inner = diagram.diagram
        clone = QuarkDiagram(
            inner.adjacency_matrix, vertex_list=inner.vertex_list,
            L=inner.L, usedNp=inner.usedNp, compensation=scheme,
        )
        return Diagram(clone, diagram.time_list, diagram.vertex_list, diagram.propagator_list)

    if len(diagrams) < 2:
        pytest.skip("this expression has a single diagram, so it cannot mix schemes")

    mixed = [
        with_scheme(diagrams[0], CompensationScheme.EXPECTED),
        with_scheme(diagrams[1], CompensationScheme.OBSERVED),
    ] + list(diagrams[2:])

    with pytest.raises(ValueError, match="must share a compensation scheme"):
        calc_diagram_prepare(mixed)
