"""Regression tests for production hooks ported from the experiment branch.

Two hooks are covered:

* ``Propagator._relative_time_index`` guards relative-time perambulator
  caches whose stored temporal extent ``Dt`` may be smaller than ``Lt``.
* ``PropagatorWithCurrent.get_PSP_highmode`` reuses the last identical PSP
  high-mode request through ``_last_psp_highmode``.
"""

import numpy as np
import pytest

from lattice.quark_diagram import Propagator, PropagatorWithCurrent


def _bare_propagator(lt):
    propagator = Propagator.__new__(Propagator)
    propagator.Lt = lt
    return propagator


def test_relative_time_index_allows_dt_within_extent():
    propagator = _bare_propagator(8)
    cache = np.arange(4 * 2, dtype=np.complex128).reshape(4, 2)
    out = propagator._relative_time_index(cache, 0, 3)
    assert out.shape == (2,)
    np.testing.assert_array_equal(out, cache[3])


def test_relative_time_index_rejects_dt_beyond_extent():
    propagator = _bare_propagator(8)
    cache = np.zeros((4, 2), dtype=np.complex128)
    with pytest.raises(IndexError):
        propagator._relative_time_index(cache, 0, 5)


def test_relative_time_index_full_lifetime_is_noop():
    propagator = _bare_propagator(8)
    cache = np.arange(8 * 2, dtype=np.complex128).reshape(8, 2)
    for sink in range(8):
        out = propagator._relative_time_index(cache, 0, sink)
        np.testing.assert_array_equal(out, cache[sink])


def _bare_psp_propagator():
    propagator = PropagatorWithCurrent.__new__(PropagatorWithCurrent)
    propagator.usedNe = 4
    propagator.usedNp = 8
    propagator._last_psp_highmode = None
    return propagator


def test_psp_highmode_reuses_identical_last_request():
    propagator = _bare_psp_propagator()
    calls = {"count": 0}
    sentinel = np.arange(3, dtype=np.complex128)

    def fake_compute(t_source, t_sink, usedNe_sink=None, usedNe_source=None):
        calls["count"] += 1
        return sentinel

    propagator._compute_PSP_highmode = fake_compute

    first = propagator.get_PSP_highmode(1, 2)
    second = propagator.get_PSP_highmode(1, 2)
    assert calls["count"] == 1
    assert first is second

    propagator.get_PSP_highmode(1, 3)
    assert calls["count"] == 2


def test_psp_highmode_cache_cleared_on_release():
    propagator = _bare_psp_propagator()
    calls = {"count": 0}

    def fake_compute(t_source, t_sink, usedNe_sink=None, usedNe_source=None):
        calls["count"] += 1
        return np.zeros(1, dtype=np.complex128)

    propagator._compute_PSP_highmode = fake_compute

    propagator.get_PSP_highmode(1, 2)
    assert propagator._last_psp_highmode is not None
    propagator._release_current_caches()
    assert propagator._last_psp_highmode is None
    propagator.get_PSP_highmode(1, 2)
    assert calls["count"] == 2
