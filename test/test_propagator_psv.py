#!/usr/bin/env python3
"""
Tests for the PSV propagator loaders.

``PropagatorPSVNpy`` (single-file) and ``PropagatorPSVTimeslicesNpy``
(timeslice-separated files) are the only concrete consumers of the lazy
``FileData`` protocol on the PSV side; no other test in ``test/`` exercises
them with real files (``test_propagator_with_current.py`` uses mocks).

Contract under test
-------------------
``load()`` returns a lazy ``FileData`` handle, not an ndarray.  Materialize it
with ``[:]`` for single-file loaders.  For timeslice loaders, index with a
1-tuple ``[(t_src,)]`` and the file *suffix must contain the ``.t???``
placeholder* (see ``lattice/filedata/ndarray.py``), because the loader rewrites
that placeholder to select the per-source-time file.
"""

import os

import numpy as np
import pytest

from lattice import PropagatorPSVNpy, PropagatorPSVTimeslicesNpy

Lt, Ns, Np, Ne = 8, 4, 10, 5
SHAPE = [Lt, Ns, Ns, Np, Ne]


def _complex_random(*shape):
    return np.random.randn(*shape) + 1j * np.random.randn(*shape)


def test_propagator_psv_npy(tmp_path):
    """PropagatorPSVNpy loads a single .npy file via lazy FileData."""
    cfg = "test_1000"
    test_data = _complex_random(Lt, Ns, Ns, Np, Ne)
    np.save(os.path.join(tmp_path, f"{cfg}.psv.npy"), test_data)

    psv = PropagatorPSVNpy(
        prefix=os.path.join(tmp_path, ""),
        suffix=".psv.npy",
        shape=SHAPE,
        Np=Np,
        Ne=Ne,
        dtype="<c16",
    )

    # load() returns a lazy handle; [:] materializes the array.
    handle = psv.load(cfg)
    assert not isinstance(handle, np.ndarray), "load() must stay lazy"

    loaded = handle[:]
    assert loaded.shape == (Lt, Ns, Ns, Np, Ne)
    assert loaded.dtype == np.dtype("<c16")
    np.testing.assert_allclose(loaded, test_data, rtol=0, atol=1e-12)


def test_propagator_psv_npy_reduced_shape(tmp_path):
    """A shape without Dirac indices round-trips too."""
    cfg = "test_1001"
    test_data = _complex_random(Lt, Np, Ne)
    np.save(os.path.join(tmp_path, f"{cfg}.simple.npy"), test_data)

    psv = PropagatorPSVNpy(
        prefix=os.path.join(tmp_path, ""),
        suffix=".simple.npy",
        shape=[Lt, Np, Ne],
        Np=Np,
        Ne=Ne,
    )

    np.testing.assert_allclose(psv.load(cfg)[:], test_data, rtol=0, atol=1e-12)


def test_propagator_psv_timeslices(tmp_path):
    """Timeslice loader rewrites the .t??? placeholder and returns one t_src file."""
    cfg = "test_2000"
    full_data = []
    for t_src in range(Lt):
        t_data = _complex_random(Lt, Ns, Ns, Np, Ne)
        full_data.append(t_data)
        np.save(os.path.join(tmp_path, f"{cfg}.t{t_src:03d}.npy"), t_data)

    # The suffix must carry the ".t???" placeholder; without it the loader looks
    # for a single literal "test_2000.npy" and raises FileNotFoundError.
    psv = PropagatorPSVTimeslicesNpy(
        prefix=os.path.join(tmp_path, ""),
        suffix=".t???.npy",
        shape=SHAPE,
        Np=Np,
        Ne=Ne,
        dtype="<c16",
    )

    handle = psv.load(cfg)
    for t_src in range(Lt):
        # Indexing with a bare int is rejected; a 1-tuple selects the time slice.
        loaded = handle[(t_src,)]
        assert loaded.shape == (Lt, Ns, Ns, Np, Ne)
        np.testing.assert_allclose(loaded, full_data[t_src], rtol=0, atol=1e-12)


def test_propagator_psv_timeslices_requires_placeholder(tmp_path):
    """A suffix without .t??? cannot resolve any timeslice file."""
    cfg = "test_2001"
    np.save(os.path.join(tmp_path, f"{cfg}.t000.npy"), _complex_random(Lt, Ns, Ns, Np, Ne))

    psv = PropagatorPSVTimeslicesNpy(
        prefix=os.path.join(tmp_path, ""),
        suffix=".npy",
        shape=SHAPE,
        Np=Np,
        Ne=Ne,
    )

    with pytest.raises(FileNotFoundError):
        psv.load(cfg)[(0,)]


def test_propagator_psv_timeslices_rejects_bare_int(tmp_path):
    """A bare int key is not a valid timeslice key."""
    cfg = "test_2002"
    np.save(os.path.join(tmp_path, f"{cfg}.t000.npy"), _complex_random(Lt, Ns, Ns, Np, Ne))

    psv = PropagatorPSVTimeslicesNpy(
        prefix=os.path.join(tmp_path, ""),
        suffix=".t???.npy",
        shape=SHAPE,
        Np=Np,
        Ne=Ne,
    )

    with pytest.raises(TypeError):
        psv.load(cfg)[0]
