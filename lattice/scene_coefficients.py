"""Horvitz-Thompson coefficients for point-coincidence scenes.

A scene's contraction merges subscripts, so it performs a **free sum** over its
blocks -- assignments that put two blocks on the same sampled coordinate are
counted.  The estimator needs the blocks mutually distinct.  So the free sum of a
scene counts every tuple whose exact coincidence pattern is *finer* than that
scene, and the weight that makes the sum unbiased is not the naive
``w(k) = C(M,k)/C(Np,k)``.

Write ``Free_sigma`` for a scene's free sum and ``C_pi`` for the sum over tuples
whose exact pattern is ``pi``.  A tuple with pattern ``pi`` is counted by
``Free_sigma`` exactly when ``sigma`` is coarser than ``pi``, so

    E[Free_sigma] = sum over pi finer than sigma of C_pi / w_E(|pi|)

and therefore

    sum_sigma c(sigma) E[Free_sigma] = sum_pi C_pi / w_E(|pi|)
                                        * sum over sigma finer than pi of c(sigma).

Unbiasedness for every ``pi`` therefore requires

    sum over sigma finer than pi of c(sigma) = w_E(|pi|),

i.e. the zeta inversion ``Z^T c = w`` on the refinement partial order, with
``Z[i][j] = 1`` iff scene ``i`` is finer than scene ``j``.

Two consequences worth stating because they are easy to get wrong:

* At ``r = 1`` the equation collapses to ``c = w_E(1) = M/Np``, which is what the
  old code used, so single-point groups keep their arithmetic exactly.
* ``w(0)`` for an empty group is 1 by definition (one empty tuple, always drawn).

Coefficients are rational and, in the ``w_E`` scheme, often integral.  They are
computed and stored as :class:`fractions.Fraction`: the float residual of the
solve already reaches ``3e-12`` at ``r = 4`` on production-sized ``M``, which is
far too large to bake into a lookup table.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Sequence, Tuple

__all__ = [
    "labelled_set_partitions",
    "blocks_of",
    "is_finer",
    "scene_coefficients",
    "expected_weight",
]


def labelled_set_partitions(r: int) -> List[Tuple[int, ...]]:
    """Every set partition of ``r`` labelled positions, as restricted growth strings.

    Position ``i`` belongs to block ``blocks[i]``, with blocks numbered by first
    appearance so each partition appears exactly once.  The count is the Bell
    number: 1, 2, 5, 15 for ``r = 1..4``.

    This is the enumeration the Horvitz-Thompson sum runs over.  Integer
    partitions collapse relabellings: at ``r = 3`` the integer partition ``[2, 1]``
    stands for the three distinct scenes ``(0,0,1)``, ``(0,1,0)`` and ``(0,1,1)``.
    They share a weight but impose different coincidence patterns, so treating
    them as one term undercounts the sum.
    """
    if r < 0:
        raise ValueError("r must be non-negative")
    partitions: List[Tuple[int, ...]] = []

    def extend(blocks: List[int]) -> None:
        if len(blocks) == r:
            partitions.append(tuple(blocks))
            return
        limit = (max(blocks) + 2) if blocks else 1
        for block in range(limit):
            extend(blocks + [block])

    extend([])
    return partitions


def blocks_of(partition: Sequence[int]) -> List[Tuple[int, ...]]:
    """Group a restricted-growth partition into its blocks, in first-appearance order."""
    grouped: Dict[int, List[int]] = {}
    for position, block in enumerate(partition):
        grouped.setdefault(block, []).append(position)
    return [tuple(positions) for positions in grouped.values()]


def is_finer(fine: Sequence[Sequence[int]], coarse: Sequence[Sequence[int]]) -> bool:
    """Whether every block of ``fine`` sits inside one block of ``coarse``."""
    return all(
        any(set(block) <= set(other) for other in coarse) for block in fine
    )


def expected_weight(M: int, Np: int, k: int) -> Fraction:
    """``w_E(k) = C(M,k)/C(Np,k)``, the expected number of k-tuples a draw contains.

    An empty group needs no coordinate and contributes nothing: there is exactly
    one empty tuple and it is always drawn, so the weight is 1.
    """
    if k == 0:
        return Fraction(1)
    if k > min(M, Np):
        raise ValueError(
            f"a scene needing {k} distinct coordinates is not estimable from a "
            f"sample of {Np} out of {M}: its inclusion probability is zero"
        )
    numerator = Fraction(1)
    denominator = Fraction(1)
    for step in range(k):
        numerator *= M - step
        denominator *= Np - step
    return numerator / denominator


def scene_coefficients(
    r: int,
    M: int,
    Np: int,
    *,
    weights: Sequence[Fraction] = None,
) -> List[Fraction]:
    """Solve ``Z^T c = w`` for one sampling group with ``r`` point ends.

    Args:
        r: number of point ends in the group (its point-coincidence positions).
        M: number of lattice points, ``L**3``.
        Np: number of sampled points.
        weights: optional right-hand side, one entry per labelled set partition in
            :func:`labelled_set_partitions` order.  Defaults to the expected
            weights ``w_E`` (compensation scheme E).  Passing observed-count
            weights selects scheme O; the scene structure and this solver are
            unchanged either way.

    Returns:
        One :class:`fractions.Fraction` coefficient per labelled set partition, in
        the same order.
    """
    partitions = labelled_set_partitions(r)
    blocks = [blocks_of(partition) for partition in partitions]
    count = len(partitions)

    if weights is None:
        rhs = [expected_weight(M, Np, len(block)) for block in blocks]
    else:
        rhs = [Fraction(w) for w in weights]
        if len(rhs) != count:
            raise ValueError(
                f"weights must have one entry per labelled set partition "
                f"({count} for r={r}), got {len(rhs)}"
            )

    if r == 0:
        return [Fraction(1)]

    # Z[i][j] = 1 iff partition i is finer than partition j.  The refinement order
    # is a partial order, so the system is triangular after a topological sort;
    # solve it that way in exact rational arithmetic rather than inverting a
    # matrix of floats.
    zeta = [
        [1 if is_finer(blocks[i], blocks[j]) else 0 for j in range(count)]
        for i in range(count)
    ]

    # Column j of Z^T is row j of Z; solve for c from the finest partitions down.
    # Order by the number of blocks: a finer partition has more blocks, and
    # "finer" always implies "at least as many blocks".
    order = sorted(range(count), key=lambda index: -len(blocks[index]))
    coefficients: List[Fraction] = [Fraction(0)] * count
    for i in order:
        # Z^T c = w at row i reads: sum over j of Z[j][i] c(j) = w(i).
        # Z[j][i] = 1 iff j is finer than i, i.e. c(i) plus the already-solved
        # strictly-finer terms.
        total = Fraction(0)
        for j in range(count):
            if j != i and zeta[j][i]:
                total += coefficients[j]
        coefficients[i] = rhs[i] - total

    return coefficients
