"""F7.3: the block accessors agree on what an array time means.

A scanned time is how every sink-time loop in the driver works, so the accessors
have to answer it the same way: stacked in request order, with a leading time axis,
and identical to calling once per slice.  ``get``, ``get_VSP`` and ``get_PSV`` do
that.  The high-mode blocks did not -- they re-indexed the already-stacked result by
the relative time, which for a full scan silently permuted the answer and for a
shorter array raised out of bounds.

The unsupported direction is now refused instead of answered.  ``get`` and
``get_VSP`` themselves disagree between the array and pointwise calls when the
*source* is scanned, so returning a number there would be unverified rather than
merely unsupported.
"""

import numpy as np
import pytest

from lattice import set_backend
from lattice.propagators import PropagatorWithCurrent

LT, NE, NP = 6, 3, 2


class _Raw:
    """A plain sliceable array standing in for a lazy FileData handle."""

    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.dtype = array.dtype

    def __getitem__(self, key):
        return self.array[key]


class _Loader:
    def __init__(self, array, ne=None, np_=None):
        self.data = _Raw(array)
        self.Ne = ne
        self.Np = np_
        self.cache_version = 1

    def load(self, key):
        return self.data


@pytest.fixture(scope="module")
def blocks():
    """One fixed set of blocks, shared by every object so comparisons are fair."""
    rng = np.random.default_rng(1)
    return {
        "vsv": rng.normal(size=(LT, LT, 4, 4, NE, NE)).astype(np.complex128),
        "vsp": rng.normal(size=(LT, LT, 4, 4, NE, NP, 3)).astype(np.complex128),
        "psv": rng.normal(size=(LT, LT, 4, 4, NP, 3, NE)).astype(np.complex128),
        "psp": rng.normal(size=(LT, LT, 4, 4, NP, 3, NP, 3)).astype(np.complex128),
        "overlap": rng.normal(size=(LT, NE, NP, 3)).astype(np.complex128),
    }


def _ready(blocks):
    """A fresh, loaded handle over the shared blocks.

    Fresh per call on purpose: the tilde caches are keyed by a single anchor time, so
    reusing one object across anchors would compare cached and recomputed values
    rather than two computations of the same thing.
    """
    set_backend("numpy")
    prop = PropagatorWithCurrent(
        vsv=_Loader(blocks["vsv"], NE),
        vsp=_Loader(blocks["vsp"], NE, NP),
        psv=_Loader(blocks["psv"], NE, NP),
        psp=_Loader(blocks["psp"], NE, NP),
        overlap_matrix=_Loader(blocks["overlap"], NE, NP),
        Lt=LT,
    )
    prop.load("cfg", NE, NP)
    return prop


def _pointwise(blocks, name, fixed, scanned):
    """The scanned accessor called once per slice, stacked in request order."""
    values = []
    for value in scanned:
        prop = _ready(blocks)
        values.append(np.asarray(getattr(prop, name)(fixed, int(value))))
    return np.stack(values, axis=0)


def test_scanned_sink_matches_pointwise_for_vsp(blocks):
    """The main scan direction: a fixed source, every sink time."""
    scanned = np.arange(LT)
    for source in (0, 2, LT - 1):
        vector = np.asarray(_ready(blocks).get_VSP_highmode(source, scanned))
        pointwise = _pointwise(blocks, "get_VSP_highmode", source, scanned)
        assert vector.shape[0] == LT
        np.testing.assert_allclose(vector, pointwise, rtol=0, atol=1e-12)


def test_scanned_sink_matches_pointwise_for_psv(blocks):
    scanned = np.arange(LT)
    for source in (0, 2, LT - 1):
        vector = np.asarray(_ready(blocks).get_PSV_highmode(source, scanned))
        pointwise = _pointwise(blocks, "get_PSV_highmode", source, scanned)
        assert vector.shape[0] == LT
        np.testing.assert_allclose(vector, pointwise, rtol=0, atol=1e-12)


def test_a_short_scan_is_not_silently_reindexed(blocks):
    """A partial scan must not be indexed by relative time.

    Before the fix the already-stacked block was indexed by ``(t - t_source) % Lt``,
    which raised for any array shorter than ``Lt`` -- or, worse, returned another
    slice's value when the index happened to be in range.
    """
    prop = _ready(blocks)
    short = np.array([3, 1])
    vector = np.asarray(prop.get_VSP_highmode(0, short))
    assert vector.shape[0] == 2, "the leading axis must follow the request, not Lt"

    pointwise = _pointwise(blocks, "get_VSP_highmode", 0, short)
    np.testing.assert_allclose(vector, pointwise, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", ["get_VSP_highmode", "get_PSV_highmode"])
def test_scanning_the_source_is_refused_not_guessed(blocks, name):
    """F7.7: an unverified direction must say so rather than return a number.

    ``get`` and ``get_VSP`` disagree between their array and pointwise forms in this
    direction, so there is no verified reference to check a result against.
    """
    with pytest.raises(NotImplementedError, match="scanning t_source"):
        getattr(_ready(blocks), name)(np.arange(LT), 2)


@pytest.mark.parametrize("times", [(2, np.arange(LT)), (np.arange(LT), 2)])
def test_array_time_psp_is_refused_not_guessed(blocks, times):
    """PSP's array paths disagree with the pointwise result in both directions.

    Either refusal is acceptable -- scanning the source is rejected for any
    current-insertion block, and scanning the sink is rejected for PSP specifically --
    so the test asserts that a reason is given, not which reason.
    """
    with pytest.raises(NotImplementedError, match="scan|not verified"):
        _ready(blocks).get_PSP_highmode(*times)


def test_scalar_calls_are_unaffected(blocks):
    """The guarded branches must not touch the single-slice path."""
    prop = _ready(blocks)
    frozen = np.asarray(prop.get_PSP_highmode(2, 3))
    assert frozen.shape == (4, 4, NP, 3, NP, 3)
    assert np.isfinite(frozen).all()

    vsp = np.asarray(prop.get_VSP_highmode(2, 3))
    assert vsp.shape == (4, 4, NE, NP, 3)
    assert np.isfinite(vsp).all()
