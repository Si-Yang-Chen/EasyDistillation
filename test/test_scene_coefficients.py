"""F5.6 / F5.7: the scene coefficient is a zeta inversion, not ``w(k)``.

A scene's contraction merges subscripts, so it performs a free sum over its
blocks: assignments that place two blocks on one sampled coordinate are counted.
The estimator needs the blocks mutually distinct.  Averaging the estimator over
every drawn subset must therefore return the all-point sum only if each scene is
weighted by ``c`` from ``Z^T c = w`` -- naive ``w(k)`` over-counts the coincident
assignments and is biased upward.

These tests pin both halves: the coefficients themselves, and the end-to-end
design mean that fails without them.
"""

import itertools
import random
from fractions import Fraction
from math import comb

import pytest

from lattice.scene_coefficients import (
    blocks_of,
    expected_weight,
    is_finer,
    labelled_set_partitions,
    scene_coefficients,
)


def _bell(r):
    return len(labelled_set_partitions(r))


def _naive_weight(M, Np, k):
    """The weight the implementation used before step 0: C(M,k)/C(Np,k)."""
    if k == 0:
        return Fraction(1)
    if k > min(M, Np):
        return None
    return Fraction(comb(M, k), comb(Np, k))


def _free_sum(blocks, value, M, drawn):
    """Sum of ``value`` over all assignments drawn from ``drawn``, merging within blocks.

    This is what a merged-subscript einsum evaluates: block coordinates range
    independently over the drawn index set, so equal assignments are included.
    """
    r = sum(len(block) for block in blocks)
    positions = [0] * r
    for index, block in enumerate(blocks):
        for position in block:
            positions[position] = index
    total = 0
    for assignment in itertools.product(drawn, repeat=len(blocks)):
        total += value(tuple(assignment[positions[i]] for i in range(r)))
    return total


def _injective_sum(blocks, value, M, drawn):
    """Sum of ``value`` over assignments of *distinct* drawn coordinates to blocks."""
    r = sum(len(block) for block in blocks)
    positions = [0] * r
    for index, block in enumerate(blocks):
        for position in block:
            positions[position] = index
    total = 0
    for assignment in itertools.permutations(drawn, len(blocks)):
        total += value(tuple(assignment[positions[i]] for i in range(r)))
    return total


def _design_mean(M, Np, r, seed):
    """Average the estimator over every drawn subset; return (true, naive, fixed)."""
    rng = random.Random(seed)
    F = [rng.random() for _ in range(M**r)]

    def value(indices):
        total = 0
        for index in indices:
            total = total * M + index
        return F[total]

    truth = sum(value(indices) for indices in itertools.product(range(M), repeat=r))

    partitions = labelled_set_partitions(r)
    blocks = [blocks_of(partition) for partition in partitions]
    fixed = scene_coefficients(r, M, Np)

    naive_total = Fraction(0)
    fixed_total = Fraction(0)
    subsets = list(itertools.combinations(range(M), Np))
    for drawn in subsets:
        for index, block in enumerate(blocks):
            k = len(block)
            free = _free_sum(block, value, M, list(drawn))
            naive = _naive_weight(M, Np, k)
            if naive is not None:
                naive_total += naive * free
            fixed_total += fixed[index] * free
    count = len(subsets)
    return (
        truth,
        float(naive_total) / count,
        float(fixed_total) / count,
    )


def test_labelled_set_partitions_are_bell_numbers():
    """Integer partitions undercount; the labelled enumeration is the Bell number."""
    assert [_bell(r) for r in range(5)] == [1, 1, 2, 5, 15]


def test_labelled_partitions_are_all_distinct():
    for r in range(5):
        partitions = labelled_set_partitions(r)
        assert len(set(partitions)) == len(partitions)
        assert all(len(set(partition)) == len(blocks_of(partition)) for partition in partitions)


def test_r_one_coefficient_is_the_current_weight():
    """F7.1's guard: a single point end keeps today's weight, so the equal-time
    two-point function cannot move."""
    for M, Np in ((13824, 216), (1000, 10), (64, 8)):
        assert scene_coefficients(1, M, Np) == [expected_weight(M, Np, 1)]


def test_coefficients_match_the_independently_solved_system():
    """Pin the rational values; they were cross-checked against a least-squares
    solve of the same system at float precision."""
    assert scene_coefficients(2, 6, 3) == [Fraction(-3), Fraction(5)]
    assert scene_coefficients(3, 6, 3) == [
        Fraction(27), Fraction(-15), Fraction(-15), Fraction(-15), Fraction(20)
    ]
    assert scene_coefficients(2, 8, 4) == [Fraction(-8, 3), Fraction(14, 3)]
    assert scene_coefficients(3, 8, 4) == [
        Fraction(16), Fraction(-28, 3), Fraction(-28, 3), Fraction(-28, 3), Fraction(14)
    ]


def test_coefficients_satisfy_the_unbiasedness_identity():
    """`sum over sigma finer than pi of c(sigma) == w(|pi|)` for every scene."""
    for M, Np, r in ((6, 3, 2), (6, 3, 3), (8, 4, 3), (8, 4, 4), (10, 5, 3)):
        partitions = labelled_set_partitions(r)
        blocks = [blocks_of(partition) for partition in partitions]
        coefficients = scene_coefficients(r, M, Np)
        for index, coarse in enumerate(blocks):
            total = sum(
                coefficients[j]
                for j, fine in enumerate(blocks)
                if is_finer(fine, coarse)
            )
            assert total == expected_weight(M, Np, len(coarse)), (
                f"identity fails for scene {partitions[index]} at r={r}"
            )


@pytest.mark.parametrize("M, Np, r", [(6, 3, 2), (6, 3, 3), (8, 4, 2), (8, 4, 3), (8, 4, 4), (10, 5, 3)])
def test_design_mean_requires_the_coefficients(M, Np, r):
    """Averaged over every drawn subset, only the corrected weighting is unbiased.

    This is the acceptance test for the sampling layer, and it must run at
    ``Np < M``: at ``Np == M`` every coefficient is 1 and a missing scene cancels
    against a missing coincidence, so the defect would pass.
    """
    assert r <= Np, "the finest partition needs r distinct coordinates"
    assert Np < M, "the oracle is vacuous at Np == M"
    truth, naive, fixed = _design_mean(M, Np, r, seed=11)

    assert abs(fixed - truth) <= 1e-9 * abs(truth), (
        f"corrected weighting is biased at r={r}: {fixed} vs {truth}"
    )
    if r >= 2:
        assert abs(naive - truth) > 1e-3 * abs(truth), (
            f"the naive w(k) weighting unexpectedly reproduced the all-point sum at "
            f"r={r}; the oracle would have no teeth ({naive} vs {truth})"
        )


def test_not_estimable_raises_rather_than_returning_zero():
    """`k > Np` has zero inclusion probability: the coefficient is undefined, and
    returning 0 would silently drop the term and change the target."""
    with pytest.raises(ValueError, match="not estimable"):
        scene_coefficients(4, 64, 3)
    with pytest.raises(ValueError, match="not estimable"):
        expected_weight(64, 3, 4)


def test_observed_scheme_enters_only_as_the_right_hand_side():
    """Scheme O changes the right-hand side, not the scene structure or the solver.

    The right-hand side is `possible / observed`.  When the observed count equals
    the expected count -- which is exactly the case inside one sampling group on
    the production point set, where all Np points are distinct -- scheme O must
    reproduce scheme E coefficient for coefficient.
    """
    r, M, Np = 2, 100, 10
    expected = scene_coefficients(r, M, Np)
    partitions = labelled_set_partitions(r)
    observed_rhs = []
    for partition in partitions:
        k = len(blocks_of(partition))
        possible = Fraction(comb(M, k))
        # The expected COUNT of k-tuples is C(Np,k) -- not w_E(k), which is the
        # count ratio.  Scheme O divides by the count, so feeding it the expected
        # count must reproduce scheme E exactly.
        observed_count = Fraction(comb(Np, k))
        observed_rhs.append(possible / observed_count)
    assert scene_coefficients(r, M, Np, weights=observed_rhs) == expected

    # A group whose observed count is below expectation must move the coefficients.
    perturbed = list(observed_rhs)
    perturbed[0] = perturbed[0] * 2
    assert scene_coefficients(r, M, Np, weights=perturbed) != expected


def test_coefficients_are_exact_rationals():
    """Float storage is not an option: the solve residual reaches 3e-12 at r=4."""
    coefficients = scene_coefficients(4, 8, 4)
    assert all(isinstance(value, Fraction) for value in coefficients)
    assert any(value.denominator != 1 for value in coefficients)
