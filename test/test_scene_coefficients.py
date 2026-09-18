"""F5.6 / F5.7: the scene coefficient is a zeta inversion, not ``w(k)``.

A scene's contraction merges subscripts, so it evaluates a **free sum** over its
blocks: assignments that put two blocks on one sampled coordinate are counted.  The
estimator needs the blocks mutually distinct.  A tuple whose exact coincidence
pattern is ``pi`` is therefore counted by every scene coarser than ``pi``, which fixes
the coefficients by the linear system

    sum over sigma finer than pi of c(sigma) = w(|pi|)      i.e.   Z^T c = w.

These tests check that identity directly.  They deliberately do **not** verify it by
enumerating drawn subsets and averaging: that oracle would be code written alongside
the implementation, so a mistake shared by both would pass, and the defect would
survive to production.  The identity below depends only on the coefficient table and
the refinement order, so it cannot inherit an error from the contraction path.

What the identity does and does not establish: it is exactly the unbiasedness
condition, given the derivation that ``E[Free_sigma]`` sums the finer patterns.  It
does not re-derive that step, and it does not by itself pin the coefficient as the
*unique* solution -- the system is triangular, so uniqueness follows from the
construction, and the tests below check the triangular structure explicitly.
"""

from fractions import Fraction

import pytest

from lattice.scene_coefficients import (
    CompensationScheme,
    SCHEMES,
    blocks_of,
    expected_weight,
    is_finer,
    labelled_set_partitions,
    observed_weight,
    scene_coefficients,
)

FEASIBLE = [(6, 3, 1), (6, 3, 2), (6, 3, 3), (8, 4, 2), (8, 4, 3), (8, 4, 4), (10, 5, 3)]


def _bell(r):
    return len(labelled_set_partitions(r))


def test_labelled_set_partitions_are_bell_numbers():
    """Integer partitions undercount; the labelled enumeration is the Bell number."""
    assert [_bell(r) for r in range(5)] == [1, 1, 2, 5, 15]


def test_the_partial_order_is_a_partial_order():
    """Refinement must be reflexive, antisymmetric and transitive on these blocks.

    The solve below orders by block count to get a triangular system, which is only
    sound if refinement is a partial order and always runs finer-to-coarser.
    """
    for r in range(1, 5):
        blocks = [blocks_of(p) for p in labelled_set_partitions(r)]
        for index, block in enumerate(blocks):
            assert is_finer(block, block), "refinement must be reflexive"
        for i, a in enumerate(blocks):
            for j, b in enumerate(blocks):
                if i != j and is_finer(a, b) and is_finer(b, a):
                    raise AssertionError("refinement must be antisymmetric")
                if is_finer(a, b):
                    assert len(a) >= len(b), (
                        "a finer partition must not have fewer blocks; the solve "
                        "relies on this to be triangular"
                    )


@pytest.mark.parametrize("M, Np, r", FEASIBLE)
def test_coefficients_satisfy_the_unbiasedness_identity(M, Np, r):
    """`sum over scenes finer than pi of c == w(|pi|)`, for every scene.

    This is the unbiasedness condition itself, checked against the coefficient table
    and the refinement order only -- no contraction, no drawn subsets.
    """
    partitions = labelled_set_partitions(r)
    blocks = [blocks_of(partition) for partition in partitions]
    coefficients = scene_coefficients(r, M, Np)

    for index, coarse in enumerate(blocks):
        total = sum(
            coefficients[j] for j, fine in enumerate(blocks) if is_finer(fine, coarse)
        )
        assert total == expected_weight(M, Np, len(coarse)), (
            f"the identity fails for scene {partitions[index]} at r={r}, M={M}, Np={Np}"
        )


@pytest.mark.parametrize("M, Np, r", FEASIBLE)
def test_the_naive_weight_fails_the_same_identity(M, Np, r):
    """The negative control: ``w(k)`` does not satisfy the identity for r >= 2.

    Without this the identity test could pass for a reason unrelated to the
    coefficients -- for instance if every scene had the same k.
    """
    if r == 1:
        pytest.skip("at r=1 the naive weight is the correct coefficient")
    partitions = labelled_set_partitions(r)
    blocks = [blocks_of(partition) for partition in partitions]

    failures = 0
    for coarse in blocks:
        total = sum(
            expected_weight(M, Np, len(fine))
            for fine in blocks
            if is_finer(fine, coarse)
        )
        if total != expected_weight(M, Np, len(coarse)):
            failures += 1
    assert failures > 0, (
        "w(k) unexpectedly satisfies the identity, so the naive weighting would have "
        "no effect and the test above would have no teeth"
    )


def test_r_one_coefficient_is_the_current_weight():
    """F7.1's guard: a single point end keeps today's weight, so the equal-time
    two-point function cannot move."""
    for M, Np in ((13824, 216), (1000, 10), (64, 8)):
        assert scene_coefficients(1, M, Np) == [expected_weight(M, Np, 1)]


def test_coefficients_are_rational_and_exact():
    """Float storage is not an option: the float solve already loses precision.

    The coefficients are rationals on the integer lattice, so they are computed and
    returned as exact ``Fraction`` values.
    """
    assert scene_coefficients(2, 6, 3) == [Fraction(-3), Fraction(5)]
    assert scene_coefficients(3, 6, 3) == [
        Fraction(27), Fraction(-15), Fraction(-15), Fraction(-15), Fraction(20)
    ]
    assert scene_coefficients(2, 8, 4) == [Fraction(-8, 3), Fraction(14, 3)]


def test_not_estimable_raises_rather_than_returning_zero():
    """`k > Np` has zero inclusion probability: the coefficient is undefined, and
    returning 0 would silently drop the term and change the target."""
    with pytest.raises(ValueError, match="not estimable"):
        scene_coefficients(4, 64, 3)
    with pytest.raises(ValueError, match="not estimable"):
        expected_weight(64, 3, 4)


def test_schemes_differ_only_in_their_right_hand_side():
    """Scheme O is scheme E with a different denominator, nothing else.

    Feeding the expected count back in must reproduce scheme E exactly.  This is a
    consistency property between the two schemes, not evidence that either is right.
    """
    r, M, Np = 2, 100, 10
    assert CompensationScheme.EXPECTED in SCHEMES
    assert CompensationScheme.OBSERVED in SCHEMES

    expected = scene_coefficients(r, M, Np)
    # The denominator counts ORDERED tuples of distinct coordinates (falling
    # factorial), matching expected_weight; a binomial coefficient is the unordered
    # count and would not reproduce scheme E.
    rhs = []
    for partition in labelled_set_partitions(r):
        k = len(blocks_of(partition))
        falling = Fraction(1)
        for step in range(k):
            falling *= Np - step
        rhs.append(observed_weight(M, Np, k, falling))
    assert scene_coefficients(r, M, Np, weights=rhs) == expected

    # A count below expectation must move the coefficients.
    perturbed = list(rhs)
    perturbed[0] = perturbed[0] * 2
    assert scene_coefficients(r, M, Np, weights=perturbed) != expected


def test_zero_observed_count_is_undefined_not_small():
    """F6.4: a term that did not enter the sample has no defined weight."""
    with pytest.raises(ValueError, match="undefined"):
        observed_weight(64, 8, 2, 0)
    with pytest.raises(ValueError, match="undefined"):
        observed_weight(64, 8, 2, -1)


def test_observed_weight_matches_the_expected_one_when_counts_agree():
    """The two schemes coincide exactly when the count equals expectation."""
    for M, Np, k in ((100, 10, 1), (100, 10, 2), (13824, 216, 3)):
        falling = Fraction(1)
        for step in range(k):
            falling *= Np - step
        assert observed_weight(M, Np, k, falling) == expected_weight(M, Np, k)
