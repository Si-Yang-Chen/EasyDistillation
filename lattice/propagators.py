"""Propagator, meson and current handle types for quark-diagram contraction.

These classes wrap the lazy ``preset`` data loaders (perambulators, elementals,
overlap matrices) and expose the indexed views that contraction consumes.

They share a validation layer: every ``load()`` derives an identity from the key,
the requested Ne/Np extents, the loader's own signature and the loaded data's
shape/hash. When any of those change, derived caches are released rather than
silently reused. The helpers below infer extents from whatever the loader
declares, because different producers write slightly different shapes.

``PropagatorWithCurrent`` is the largest piece: it composes V2V, V2P, P2V and P2P
data around an inserted current and applies high-mode projection, with separate
extents for source and sink.

This module is an implementation detail of ``lattice.quark_diagram``; import
through that module (or ``lattice``) rather than directly.
"""

import gc
import hashlib
import logging
import os
from typing import Dict

import numpy as np
from opt_einsum import contract

from lattice.constant import Nc
from .preset import (
    Perambulator,
    PropagatorVSP,
    PropagatorPSV,
    PropagatorPSP,
    OverlapMatrix,
    CurrentElementalV2P,
    CurrentElementalP2V,
    CurrentElementalP2P,
)

from .backend import get_backend, log_gpu_memory

logger = logging.getLogger(__name__)


def _normalize_extent(value, available, name):
    if available is None:
        raise ValueError(f"cannot determine available {name}")
    if value is None:
        value = available
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    available = int(available)
    if value < 0 or value > available:
        raise ValueError(f"{name}={value} is outside [0, {available}]")
    return value


def _available_extent(loader, data, attribute, axes):
    declared = getattr(loader, attribute, None)
    if isinstance(declared, bool) or not isinstance(declared, (int, np.integer)):
        declared = None
    shape = getattr(data, "shape", None)
    dimensions = []
    if shape is not None:
        for axis in axes:
            try:
                dimensions.append(int(shape[axis]))
            except (IndexError, TypeError):
                pass
    if declared is not None:
        declared = int(declared)
        if dimensions and any(dimension != declared for dimension in dimensions):
            raise ValueError(
                f"{attribute}={declared} conflicts with data shape {tuple(shape)}"
            )
        return declared
    if not dimensions:
        return None
    if any(dimension != dimensions[0] for dimension in dimensions[1:]):
        raise ValueError(f"inconsistent {attribute} dimensions in shape {tuple(shape)}")
    return dimensions[0]


def _metadata_shape(loader):
    shape = getattr(loader, "shape", None)
    if not isinstance(shape, (tuple, list)):
        shape = getattr(getattr(loader, "elem", None), "shape", None)
    return tuple(shape) if isinstance(shape, (tuple, list)) else ()


def _loader_extents(loader, role):
    if loader is None:
        return None, None
    declared_ne = getattr(loader, "Ne", None)
    declared_np = getattr(loader, "Np", None)
    ne = int(declared_ne) if isinstance(declared_ne, (int, np.integer)) and not isinstance(declared_ne, bool) else None
    np_ = int(declared_np) if isinstance(declared_np, (int, np.integer)) and not isinstance(declared_np, bool) else None
    shape = _metadata_shape(loader)
    axes = {
        "vsv": ((-2, -1), ()),
        "vsp": ((-3,), (-2,)),
        "psv": ((-1,), (-3,)),
        "overlap": ((-3,), (-2,)),
        "psp": ((), (-4, -2)),
    }
    ne_axes, np_axes = axes[role]
    ne_dims = [int(shape[axis]) for axis in ne_axes if len(shape) >= abs(axis)]
    np_dims = [int(shape[axis]) for axis in np_axes if len(shape) >= abs(axis)]
    if ne_dims and any(value != ne_dims[0] for value in ne_dims[1:]):
        raise ValueError(f"{role} has inconsistent Ne dimensions: {shape}")
    if np_dims and any(value != np_dims[0] for value in np_dims[1:]):
        raise ValueError(f"{role} has inconsistent Np dimensions: {shape}")
    inferred_ne = ne_dims[0] if ne_dims else None
    inferred_np = np_dims[0] if np_dims else None
    if ne is not None and inferred_ne is not None and ne != inferred_ne:
        raise ValueError(f"{role} Ne={ne} conflicts with shape {shape}")
    if np_ is not None and inferred_np is not None and np_ != inferred_np:
        raise ValueError(f"{role} Np={np_} conflicts with shape {shape}")
    return ne if ne is not None else inferred_ne, np_ if np_ is not None else inferred_np


def _data_signature(data):
    version = getattr(data, "cache_version", getattr(data, "version", None))
    file = getattr(data, "file", None)
    file_state = None
    if isinstance(file, str):
        try:
            stat = os.stat(file)
            file_state = (file, stat.st_size, stat.st_mtime_ns)
        except OSError:
            file_state = (file, None, None)
    content_hash = None
    if isinstance(data, np.ndarray):
        contiguous = np.ascontiguousarray(data)
        content_hash = hashlib.sha256(contiguous.view(np.uint8)).hexdigest()
    return (
        type(data).__module__,
        type(data).__qualname__,
        tuple(getattr(data, "shape", ()) or ()),
        str(getattr(data, "dtype", None)),
        version,
        file_state,
        content_hash,
    )


def _loader_signature(loader, key):
    elem = getattr(loader, "elem", None)
    shape = tuple(getattr(elem, "shape", ()) or ())
    dtype = getattr(elem, "dtype", None)
    prefix = getattr(loader, "prefix", None)
    suffix = getattr(loader, "suffix", None)
    resolved = None
    file_state = None
    if isinstance(prefix, str) and isinstance(suffix, str) and "?" not in suffix:
        resolved = f"{prefix}{key}{suffix}"
        try:
            stat = os.stat(resolved)
            file_state = (stat.st_size, stat.st_mtime_ns)
        except OSError:
            file_state = None
    return (
        type(loader).__module__,
        type(loader).__qualname__,
        id(loader),
        prefix,
        suffix,
        shape,
        dtype,
        getattr(loader, "Ne", None),
        getattr(loader, "Np", None),
        getattr(loader, "cache_version", getattr(loader, "version", None)),
        resolved,
        file_state,
    )


def _operator_signature(operator):
    return hashlib.sha256(repr(getattr(operator, "parts", None)).encode()).hexdigest()


class Particle:
    pass


class Meson(Particle):
    def __init__(self, elemental, operator, source) -> None:
        self.elemental = elemental
        self.elemental_data = None
        self.key = None
        self.operator = operator
        self.dagger = source
        self.outward = 1
        self.inward = 1
        self.smeared = True
        # cache is defined as a class variable of the Meson class.
        # cache is shared among all instances of Meson.
        backend = get_backend()
        self.cache: Dict[int, backend.ndarray] = {}
        self._cache_identity = None

    def _release_resources(self):
        self.elemental_data = None
        self.cache = {}
        self._cache_identity = None
        gc.collect()
        backend = get_backend()
        if hasattr(backend, "get_default_memory_pool"):
            try:
                backend.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    def release(self):
        self._release_resources()
        self.key = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def __str__(self) -> str:
        str = "### Meson ###\n"
        str += Rf"\n key = {self.key} \n"
        str += self.operator.__str__()
        str += Rf"\n dagger = {self.dagger} \n"
        return str

    def load(self, key, usedNe: int = None):
        elemental_data = self.elemental.load(key)
        available_ne = _available_extent(
            self.elemental, elemental_data, "Ne", (-2, -1)
        )
        normalized_ne = _normalize_extent(usedNe, available_ne, "usedNe")
        identity = (
            key,
            normalized_ne,
            _loader_signature(self.elemental, key),
            _data_signature(elemental_data),
            _operator_signature(self.operator),
            bool(self.dagger),
        )
        if identity != self._cache_identity:
            self._release_resources()
            self.key = key
            self.usedNe = normalized_ne
            self.elemental_data = elemental_data
            self._make_cache()
            self._cache_identity = identity

    def _make_cache(self):
        from lattice.insertion.gamma import gamma

        backend = get_backend()
        cache = self.cache
        parts = self.operator.parts
        ret_gamma = []
        ret_elemental = []
        for i in range(len(parts) // 2):
            ret_gamma.append(gamma(parts[i * 2]))
            elemental_part = parts[i * 2 + 1]
            for j in range(len(elemental_part)):
                elemental_coeff, derivative_idx, momentum_idx, profile = elemental_part[
                    j
                ]
                elemental_coeff = complex(elemental_coeff)
                deriv_mom_tuple = (derivative_idx, momentum_idx)
                if deriv_mom_tuple not in cache:
                    cache[deriv_mom_tuple] = self.elemental_data[
                        derivative_idx, momentum_idx, :, : self.usedNe, : self.usedNe
                    ]
                if j == 0:
                    ret_elemental.append(elemental_coeff * cache[deriv_mom_tuple])
                else:
                    ret_elemental[-1] += elemental_coeff * cache[deriv_mom_tuple]
        if self.dagger:
            self.cache = (
                contract(
                    "ik,xlk,lj->xij",
                    gamma(8),
                    backend.asarray(ret_gamma).conj(),
                    gamma(8),
                ),
                contract("xtba->xtab", backend.asarray(ret_elemental).conj()),
            )
        else:
            self.cache = (
                backend.asarray(ret_gamma),
                backend.asarray(ret_elemental),
            )

    def get(self, t):
        """
        Get V2V vertex (operator matrix element O_{i,j}).

        Corresponds to formula (3.1): low-low term.

        Args:
            t: Time index (int or array)

        Returns:
            [Ns, Ns, Ne, Ne] if t is int
            [t_len, Ns, Ns, Ne, Ne] if t is array
        """
        if isinstance(t, int):
            if self.dagger:
                return contract("xij,xab->ijab", self.cache[0], self.cache[1][:, t])
            else:
                return contract("xij,xab->ijab", self.cache[0], self.cache[1][:, t])
        else:
            if self.dagger:
                return contract("xij,xtab->tijab", self.cache[0], self.cache[1][:, t])
            else:
                return contract("xij,xtab->tijab", self.cache[0], self.cache[1][:, t])


class Current(Meson):
    def __init__(
        self,
        elemental,
        operator,
        source,
        v2p_data: "CurrentElementalV2P" = None,
        p2v_data: "CurrentElementalP2V" = None,
        p2p_data: "CurrentElementalP2P" = None,
        debug: bool = False,
    ) -> None:
        super().__init__(elemental, operator, source)
        self.smeared = False
        self.v2p_data = v2p_data
        self.p2v_data = p2v_data
        self.p2p_data = p2p_data
        self.debug = debug
        # Two-time propagator caches (initialized on demand)
        # VSP caches
        self.vsp_cache = None
        self.vsp_cache_dagger = None
        self.vsp_cached_time = None
        # PSV caches
        self.psv_cache = None
        self.psv_cache_dagger = None
        self.psv_cached_time = None
        # PSP caches
        self.psp_cache = None
        self.psp_cache_dagger = None
        self.psp_cached_time = None
        self.cache_v2p = {}
        self.cache_p2v = {}
        self.cache_p2p = {}
        # Loaded precomputed data (p2p only, since v2p/p2v are pre-loaded in cache)
        self.p2p_loaded = None
        # Displacement reversal mapping cache
        self._disp_reversal_map = None

    def _release_resources(self):
        super()._release_resources()
        self.cache_v2p = {}
        self.cache_p2v = {}
        self.cache_p2p = {}
        self.p2p_loaded = None
        self._disp_reversal_map = None

    def release(self):
        self._release_resources()
        self.key = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def load(self, key, usedNe: int = None, usedNp: int = None):
        elemental_data = self.elemental.load(key)
        available_ne = _available_extent(
            self.elemental, elemental_data, "Ne", (-2, -1)
        )
        if self.p2v_data is None:
            raise ValueError("p2v_data must be provided for Current integration")
        p2v_ne = getattr(self.p2v_data, "Ne", None)
        if p2v_ne is not None and int(p2v_ne) != int(available_ne):
            raise ValueError(
                f"elemental Ne={available_ne} conflicts with p2v Ne={p2v_ne}"
            )
        available_np = getattr(self.p2v_data, "Np", None)
        normalized_ne = _normalize_extent(usedNe, available_ne, "usedNe")
        normalized_np = _normalize_extent(usedNp, available_np, "usedNp")
        identity = (
            key,
            normalized_ne,
            normalized_np,
            _loader_signature(self.elemental, key),
            _loader_signature(self.p2v_data, key),
            _loader_signature(self.p2p_data, key) if self.p2p_data is not None else None,
            _data_signature(elemental_data),
            _operator_signature(self.operator),
            bool(self.dagger),
        )
        if identity != self._cache_identity:
            self._release_resources()
            self.key = key
            self.elemental_data = elemental_data
            self.Lt = self.elemental_data.shape[2]
            self.usedNe = normalized_ne
            self.usedNp = normalized_np
            self._make_cache()
            self._cache_identity = identity

    def _build_displacement_reversal_map(self):
        """
        Build mapping from gaugelink_idx to its reverse displacement gaugelink_idx.

        Uses the symmetry: v2p(disp) ≈ p2v(-disp).transpose(Ne, Np)
        """
        from lattice.insertion import GaugeLink

        if self._disp_reversal_map is not None:
            return

        # Get num_disp from elemental data
        num_disp = self.elemental_data.shape[0]
        self._disp_reversal_map = {}

        for disp_idx in range(num_disp):
            gauge_link = GaugeLink(disp_idx)
            disp = tuple(gauge_link.displacement)
            reverse_disp = tuple(-d for d in disp)

            # Find gaugelink with reverse displacement
            # Use conjugate() method which reverses the gauge_list
            conjugate_link = gauge_link.conjugate()
            reverse_idx = conjugate_link.idx

            self._disp_reversal_map[disp_idx] = reverse_idx

            if self.debug:
                logger.debug(
                    f"Displacement reversal: {disp_idx} (disp={disp}) -> {reverse_idx} (disp={reverse_disp})"
                )

    def _make_cache(self):
        """
        Build cache from precomputed v2p, p2v, p2p data.

        Loads data from files and accumulates with operator coefficients.
        """
        from lattice.insertion.gamma import gamma
        from lattice.insertion import GaugeLink

        backend = get_backend()
        cache_v2v = self.cache
        parts = self.operator.parts

        # Build displacement reversal map for v2p symmetry
        self._build_displacement_reversal_map()

        if self.debug:
            logger.debug(f"\n{'='*80}")
            logger.debug(f"DEBUG: _make_cache() called (from precomputed data)")
            logger.debug(f"DEBUG: usedNe={self.usedNe}, usedNp={self.usedNp}")
            logger.debug(f"{'='*80}\n")

        # Load p2v data for all time slices (needed for v2p symmetry)
        if self.debug:
            logger.debug("DEBUG: Loading p2v data for v2p symmetry calculation...")
        p2v_full = self.p2v_data.load(self.key)[:]  # [Lt, num_disp, Np, Nc, Ne]

        ret_gamma = []
        ret_elemental_v2v = []
        ret_elemental_v2p = []
        ret_elemental_p2v = []
        ret_elemental_p2p = []
        for i in range(len(parts) // 2):
            ret_gamma.append(gamma(parts[i * 2]))
            elemental_part = parts[i * 2 + 1]

            if self.debug:
                logger.debug(f"\nDEBUG: Processing operatorpart {i}, gamma={parts[i * 2]}")
                logger.debug(f"DEBUG: elemental_part has {len(elemental_part)} terms")

            for j in range(len(elemental_part)):
                # parts terms are [coeff, link/derivative idx, momentum, profile]
                elemental_coeff, gaugelink_idx, momentum_idx, _profile = elemental_part[j]
                elemental_coeff = complex(elemental_coeff)
                deriv_mom_tuple_v2v = ("v2v", gaugelink_idx, momentum_idx)

                if self.debug:
                    logger.debug(
                        f"\n  DEBUG: Term {j}: coeff={elemental_coeff}, gaugelink_idx={gaugelink_idx}, momentum_idx={momentum_idx}"
                    )

                # v2v: load from elemental (already exists)
                if deriv_mom_tuple_v2v not in cache_v2v:
                    cache_v2v[deriv_mom_tuple_v2v] = self.elemental_data[
                        gaugelink_idx, momentum_idx, :, : self.usedNe, : self.usedNe
                    ]
                    if self.debug:
                        logger.debug(
                            f"  DEBUG: cache_v2v created, shape={cache_v2v[deriv_mom_tuple_v2v].shape}"
                        )

                # Accumulate v2v results
                if j == 0:
                    ret_elemental_v2v.append(
                        elemental_coeff * cache_v2v[deriv_mom_tuple_v2v]
                    )
                else:
                    ret_elemental_v2v[-1] += (
                        elemental_coeff * cache_v2v[deriv_mom_tuple_v2v]
                    )

                # p2v: load and accumulate from precomputed data (following v2v pattern)
                # Note: momentum_idx is ignored for p2v (momentum-independent)
                p2v_data_slice = p2v_full[
                    :, gaugelink_idx, : self.usedNp, :, : self.usedNe
                ]  # [Lt, Np, Nc, Ne]

                if j == 0:
                    ret_elemental_p2v.append(elemental_coeff * p2v_data_slice)
                else:
                    ret_elemental_p2v[-1] += elemental_coeff * p2v_data_slice

                # v2p: compute from p2v using symmetry v2p(disp) = p2v(-disp).transpose(Ne, Np)
                reverse_gaugelink_idx = self._disp_reversal_map[gaugelink_idx]
                # Get p2v data with reversed displacement and transpose for v2p shape
                v2p_data_slice = p2v_full[
                    :, reverse_gaugelink_idx, : self.usedNp, :, : self.usedNe
                ]  # [Lt, Np, Nc, Ne]
                v2p_data_slice = v2p_data_slice.transpose(
                    0, 3, 1, 2
                )  # [Lt, Ne, Np, Nc]

                if j == 0:
                    ret_elemental_v2p.append(elemental_coeff * v2p_data_slice)
                else:
                    ret_elemental_v2p[-1] += elemental_coeff * v2p_data_slice

                # p2p: store reference for lazy loading (keep as is since it's sparse)
                if j == 0:
                    ret_elemental_p2p.append(
                        [(gaugelink_idx, momentum_idx, elemental_coeff)]
                    )
                else:
                    ret_elemental_p2p[-1].append(
                        (gaugelink_idx, momentum_idx, elemental_coeff)
                    )

        if self.debug:
            logger.debug(f"\n{'='*80}")
            logger.debug(f"DEBUG: _make_cache() completed")
            logger.debug(f"DEBUG: Total cache entries created:")
            logger.debug(f"  - cache_v2v: {len(cache_v2v)} entries")
            logger.debug(f"DEBUG: ret_gamma length: {len(ret_gamma)}")
            logger.debug(f"DEBUG: ret_elemental_v2v length: {len(ret_elemental_v2v)}")
            logger.debug(
                f"DEBUG: ret_elemental_v2p length: {len(ret_elemental_v2p)} (pre-computed from p2v symmetry)"
            )
            logger.debug(
                f"DEBUG: ret_elemental_p2v length: {len(ret_elemental_p2v)} (pre-loaded data)"
            )
            logger.debug(
                f"DEBUG: ret_elemental_p2p length: {len(ret_elemental_p2p)} (lazy load instructions)"
            )
            logger.debug(f"{'='*80}\n")

        # Store as tuples for pre-loaded data
        # ret_elemental_v2p contains pre-computed data from p2v symmetry
        # ret_elemental_p2v contains pre-loaded data
        # ret_elemental_p2p contains lazy load instructions (sparse)
        if self.dagger:
            self.cache = (
                contract(
                    "ik,xlk,lj->xij",
                    gamma(8),
                    backend.asarray(ret_gamma).conj(),
                    gamma(8),
                ),
                contract("xtba->xtab", backend.asarray(ret_elemental_v2v).conj()),
                backend.asarray(
                    ret_elemental_v2p
                ),  # Pre-converted reverse displacement for v2p
                backend.asarray(ret_elemental_p2v),  # Direct instructions for p2v
                ret_elemental_p2p,
            )
        else:
            self.cache = (
                backend.asarray(ret_gamma),
                backend.asarray(ret_elemental_v2v),
                backend.asarray(ret_elemental_v2p),
                backend.asarray(ret_elemental_p2v),
                ret_elemental_p2p,
            )

    def get_v2p(self, t):
        """
        Get V2P vertex, pre-computed from P2V using symmetry v2p(disp) ≈ p2v(-disp).transpose(Ne, Np).

        Data is pre-loaded and processed in _make_cache() following the same pattern as v2v.

        Args:
            t: Time index (int or array)

        Returns:
            [Ns, Ns, Ne, Np, Nc] if t is int
            [t_len, Ns, Ns, Ne, Np, Nc] if t is array
        """
        # cache[2] contains pre-computed v2p data [num_parts, Lt, Ne, Np, Nc]
        if isinstance(t, int):
            if self.dagger:
                return contract("xij,xepa->ijepa", self.cache[0], self.cache[2][:, t])
            else:
                return contract("xij,xepa->ijepa", self.cache[0], self.cache[2][:, t])
        else:
            if self.dagger:
                return contract("xij,xtepa->tijepa", self.cache[0], self.cache[2][:, t])
            else:
                return contract("xij,xtepa->tijepa", self.cache[0], self.cache[2][:, t])

    def get_p2v(self, t):
        """
        Get P2V vertex, pre-loaded and processed in _make_cache() following the same pattern as v2v.

        Args:
            t: Time index (int or array)

        Returns:
            [Ns, Ns, Np, Nc, Ne] if t is int
            [t_len, Ns, Ns, Np, Nc, Ne] if t is array
        """
        # cache[3] contains pre-loaded p2v data [num_parts, Lt, Np, Nc, Ne]
        if isinstance(t, int):
            if self.dagger:
                return contract("xij,xpae->ijpae", self.cache[0], self.cache[3][:, t])
            else:
                return contract("xij,xpae->ijpae", self.cache[0], self.cache[3][:, t])
        else:
            if self.dagger:
                return contract("xij,xtpae->tijpae", self.cache[0], self.cache[3][:, t])
            else:
                return contract("xij,xtpae->tijpae", self.cache[0], self.cache[3][:, t])

    def get_p2p(self, t):
        """
        Get P2P vertex, loading from precomputed sparse file on demand.

        Args:
            t: Time index (int or array)

        Returns:
            [Ns, Ns, Np, Nc, Np, Nc] if t is int
            [t_len, Ns, Ns, Np, Nc, Np, Nc] if t is array
        """
        backend = get_backend()
        Nc = 3

        if isinstance(t, int):
            # Load p2p data for this specific time slice (if not already loaded)
            if self.p2p_loaded is None or t not in self.p2p_loaded:
                # Load p2p sparse data for time slice t
                # Get num_momentum from elemental_data shape: [num_disp, num_momentum, Lt, Ne, Ne]
                num_momentum = (
                    self.elemental_data.shape[1]
                    if self.elemental_data is not None
                    else None
                )
                p2p_sparse_list = self.p2p_data.load(
                    self.key, t, num_momentum=num_momentum
                )

                if self.p2p_loaded is None:
                    self.p2p_loaded = {}

                # Accumulate with operator coefficients and convert to dense
                # cache[4] contains p2p instructions
                num_parts = len(self.cache[4])
                result_parts = []

                for part_idx in range(num_parts):
                    dense_array = backend.zeros(
                        (self.usedNp, Nc, self.usedNp, Nc), dtype="<c16"
                    )

                    for gaugelink_idx, momentum_idx, coeff in self.cache[4][part_idx]:
                        # Get the sparse data for this gaugelink
                        # p2p_sparse_list is organized as [disp_idx * num_momentum + momentum_idx]
                        # Get num_momentum from elemental_data shape: [num_disp, num_momentum, Lt, Ne, Ne]
                        num_momentum = (
                            self.elemental_data.shape[1]
                            if self.elemental_data is not None
                            else 1
                        )
                        idx = gaugelink_idx * num_momentum + momentum_idx
                        sparse_data = p2p_sparse_list[idx]

                        if sparse_data["type"] == "identity":
                            # Identity case: add to diagonal
                            for p in range(self.usedNp):
                                for c in range(Nc):
                                    dense_array[p, c, p, c] += coeff
                        else:
                            # Sparse case: fill from coordinate list
                            indices = sparse_data["indices"]
                            values = sparse_data["values"]

                            for i in range(len(indices)):
                                l, r = indices[i]
                                if l < self.usedNp and r < self.usedNp:
                                    dense_array[l, :, r, :] += coeff * backend.asarray(
                                        values[i]
                                    )

                    result_parts.append(dense_array)

                self.p2p_loaded[t] = backend.asarray(result_parts)

            elemental_stack = self.p2p_loaded[t]

            if self.dagger:
                return contract("xij,xpwqz->ijpwqz", self.cache[0], elemental_stack)
            else:
                return contract("xij,xpwqz->ijpwqz", self.cache[0], elemental_stack)
        else:
            raise NotImplementedError("PSP propagator with array t not yet implemented")


class Propagator:
    def __init__(self, perambulator, Lt) -> None:
        self.perambulator = perambulator
        self.perambulator_data = None
        self.key = None
        self.Lt = Lt
        self.cache = None
        self.cache_dagger = None
        self.cached_time = None
        self._cache_identity = None

    def _release_resources(self):
        self.perambulator_data = None
        self.cache = None
        self.cache_dagger = None
        self.cached_time = None
        self._cache_identity = None
        gc.collect()
        backend = get_backend()
        if hasattr(backend, "get_default_memory_pool"):
            try:
                backend.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    def release(self):
        self._release_resources()
        self.key = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def load(self, key, usedNe: int = None):
        perambulator_data = self.perambulator.load(key)
        available_ne = _available_extent(
            self.perambulator, perambulator_data, "Ne", (-2, -1)
        )
        normalized_ne = _normalize_extent(usedNe, available_ne, "usedNe")
        identity = (
            key,
            normalized_ne,
            _loader_signature(self.perambulator, key),
            _data_signature(perambulator_data),
        )
        if identity != self._cache_identity:
            self._release_resources()
            self.key = key
            self.usedNe = normalized_ne
            self.perambulator_data = perambulator_data
            self._cache_identity = identity

    def _relative_time_index(self, cache, t_from, t_to):
        """Map (t_from -> t_to) onto a relative-time perambulator cache.

        Timeslice files store only delta_t in [0, Dt), where Dt = cache.shape[0]
        (often Dt < Lt). Indexing with (t_to - t_from) % Lt without this check
        either raises or (on some backends) silently wraps, which corrupts
        blending/highmode contractions that request full Lt sinks.
        """
        rel = (t_to - t_from) % self.Lt
        dt_extent = int(cache.shape[0])
        rel_check = rel.get() if hasattr(rel, "get") else rel
        rel_check = np.asarray(rel_check)
        if np.any(rel_check >= dt_extent):
            bad = np.asarray(rel_check)[np.asarray(rel_check) >= dt_extent]
            raise IndexError(
                f"Relative time index {bad.tolist()[:5]}... exceeds perambulator "
                f"Dt={dt_extent} (Lt={self.Lt}). Only delta_t in [0, {dt_extent}) "
                f"are stored; restrict sink times in the contraction driver."
            )
        return cache[rel]

    def get(self, t_source, t_sink):
        from lattice.insertion.gamma import gamma

        if isinstance(t_source, int) and isinstance(t_sink, int):
            if self.cached_time != t_source and self.cached_time != t_sink:
                self.cache = self.perambulator_data[
                    t_source, :, :, :, : self.usedNe, : self.usedNe
                ]
                self.cache_dagger = contract(
                    "ik,tlkba,lj->tijab", gamma(15), self.cache.conj(), gamma(15)
                )
                self.cached_time = t_source
            if self.cached_time == t_source:
                return self._relative_time_index(self.cache, t_source, t_sink)
            else:
                return self._relative_time_index(self.cache_dagger, t_sink, t_source)
        elif isinstance(t_source, int):
            if self.cached_time != t_source:
                self.cache = self.perambulator_data[
                    t_source, :, :, :, : self.usedNe, : self.usedNe
                ]
                self.cache_dagger = contract(
                    "ik,tlkba,lj->tijab", gamma(15), self.cache.conj(), gamma(15)
                )
                self.cached_time = t_source
            return self._relative_time_index(self.cache, t_source, t_sink)
        elif isinstance(t_sink, int):
            if self.cached_time != t_sink:
                self.cache = self.perambulator_data[
                    t_sink, :, :, :, : self.usedNe, : self.usedNe
                ]
                self.cache_dagger = contract(
                    "ik,tlkba,lj->tijab", gamma(15), self.cache.conj(), gamma(15)
                )
                self.cached_time = t_sink
            return self._relative_time_index(self.cache_dagger, t_sink, t_source)
        else:
            raise ValueError("At least t_source or t_sink should be int")


class PropagatorLocal:
    def __init__(self, perambulator, Lt) -> None:
        self.perambulator = perambulator
        self.key = None
        self.Lt = Lt
        self.cache = None
        self._cache_identity = None

    def _release_resources(self):
        self.perambulator_data = None
        self.cache = None
        self._cache_identity = None

    def release(self):
        self._release_resources()
        self.key = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

    def load(self, key, usedNe: int = None):
        perambulator_data = self.perambulator.load(key)
        available_ne = _available_extent(
            self.perambulator, perambulator_data, "Ne", (-2, -1)
        )
        normalized_ne = _normalize_extent(usedNe, available_ne, "usedNe")
        identity = (
            key,
            normalized_ne,
            _loader_signature(self.perambulator, key),
            _data_signature(perambulator_data),
        )
        if identity != self._cache_identity:
            self._release_resources()
            self.key = key
            self.perambulator_data = perambulator_data
            self.usedNe = normalized_ne
            self._make_cache()
            self._cache_identity = identity

    def _make_cache(self):
        self.cache = self.perambulator_data[0, :, :, :, : self.usedNe, : self.usedNe]
        for t_source in range(1, self.Lt):
            self.cache[t_source] = self.perambulator_data[
                t_source, 0, :, :, : self.usedNe, : self.usedNe
            ]

    def get(self, t_source, t_sink):
        if isinstance(t_source, int):
            assert t_source == t_sink, "You cannot use PropagatorLocal here"
        else:
            assert (t_source == t_sink).all(), "You cannot use PropagatorLocal here"
        return self.cache[t_source]


class PropagatorWithCurrent(Propagator):
    """
    Propagator that supports all four types: VSV, VSP, PSV, PSP.

    This class extends Propagator to support additional propagator types.
    When only vsv is provided, it behaves exactly like Propagator.

    Args:
        vsv: VSV propagator (Perambulator), optional but recommended for compatibility
        vsp: VSP propagator (PropagatorVSP), optional
        psv: PSV propagator (PropagatorPSV), optional
        psp: PSP propagator (PropagatorPSP), optional
        eigenvector: Eigenvector object for high mode projection, optional
        point: PointSource object for high mode projection, optional
        Lt: Temporal extent
        debug: Enable debug output
    """

    def __init__(
        self,
        vsv: Perambulator = None,
        vsp: PropagatorVSP = None,
        psv: PropagatorPSV = None,
        psp: PropagatorPSP = None,
        overlap_matrix: "OverlapMatrix" = None,
        Lt: int = None,
        debug: bool = False,
    ):
        # Initialize parent class with vsv as perambulator if provided
        # If vsv is None, we still need to call super().__init__ with a dummy perambulator
        # But we'll handle this case specially
        if vsv is not None:
            super().__init__(vsv, Lt)
        else:
            # Create a dummy perambulator for parent class initialization
            # This allows the class to work even without vsv
            super().__init__(None, Lt)
        self.vsp_propagator = vsp
        self.psv_propagator = psv
        self.psp_propagator = psp
        self.overlap_matrix = overlap_matrix
        self.debug = debug

        # Caches for loaded data (additional to parent's cache)
        self.vsp_data = None
        self.psv_data = None
        self.psp_data = None
        # Overlap matrix data loaded from file
        self.overlap_matrix_data = None
        # Two-time cache structures for VSP/PSV/PSP (mirror Propagator.get)
        # Naming convention: cache name indicates data source (vsp_data/psv_data/psp_data)
        # _dagger suffix indicates gamma(15) transformation applied
        # VSP data caches: vsp_cache is VSP ordering, vsp_dagger is PSV ordering
        self.vsp_cache = None
        self.vsp_dagger = None
        self.vsp_cached_time = None
        # PSV data caches: psv_cache is PSV ordering, psv_dagger is VSP ordering
        self.psv_cache = None
        self.psv_dagger = None
        self.psv_cached_time = None
        # PSP data caches: psp_cache is original, psp_dagger has left/right swapped
        self.psp_cache = None
        self.psp_dagger = None
        self.psp_cached_time = None
        # Cache for tilde_S PSV (high mode projected PSV)
        # Simple cache for single t_source, value shape: [Lt, Ns, Ns, Np, Nc, Ne]
        # Cache for tilde_S PSV (high mode projected PSV)
        # Simple cache for single t_source, value shape: [Lt, Ns, Ns, Np, Nc, Ne]
        self.tilde_S_psv_cache = None
        self.tilde_S_psv_cached_time = None
        # Dagger version: PSV ordering -> VSP ordering (Ne, Np, Nc)
        self.tilde_S_psv_dagger = None

        # Cache for tilde_S VSP (high mode projected VSP)
        # Simple cache for single t_sink, value shape: [Lt, Ns, Ns, Ne, Np, Nc]
        self.tilde_S_vsp_cache = None
        self.tilde_S_vsp_cached_time = None
        # Dagger version: VSP ordering -> PSV ordering (Np, Nc, Ne)
        self.tilde_S_vsp_dagger = None

        # Cache for tilde_S PSP (high mode projected PSP)
        # Simple cache for single t_source, value shape: [Lt, Ns, Ns, Np_snk, Nc, Np_src, Nc]
        self.tilde_S_psp_cache = None
        self.tilde_S_psp_cached_time = None
        self.tilde_S_psp_dagger = None
        # Last identical PSP high-mode request (scene expansion calls
        # get_PSP_highmode once per scene with the same (t_src, t_snk)).
        self._last_psp_highmode = None

    def _release_current_caches(self):
        self.vsp_cache = None
        self.vsp_dagger = None
        self.vsp_cached_time = None
        self.psv_cache = None
        self.psv_dagger = None
        self.psv_cached_time = None
        self.psp_cache = None
        self.psp_dagger = None
        self.psp_cached_time = None
        self.tilde_S_psv_cache = None
        self.tilde_S_psv_cached_time = None
        self.tilde_S_psv_dagger = None
        self.tilde_S_vsp_cache = None
        self.tilde_S_vsp_cached_time = None
        self.tilde_S_vsp_dagger = None
        self.tilde_S_psp_cache = None
        self.tilde_S_psp_cached_time = None
        self.tilde_S_psp_dagger = None
        self._last_psp_highmode = None

    @staticmethod
    def _time_cache_key(time):
        if isinstance(time, (int, np.integer)):
            return ("i", int(time))
        arr = np.asarray(time).reshape(-1)
        return ("a", tuple(int(v) for v in arr))

    def _psp_highmode_cache_key(self, t_source, t_sink, usedNe_sink, usedNe_source):
        return (
            self._time_cache_key(t_source),
            self._time_cache_key(t_sink),
            None if usedNe_sink is None else int(usedNe_sink),
            None if usedNe_source is None else int(usedNe_source),
            None if getattr(self, "usedNe", None) is None else int(self.usedNe),
            None if getattr(self, "usedNp", None) is None else int(self.usedNp),
        )

    def _cached_psp_highmode(self, key):
        last = getattr(self, "_last_psp_highmode", None)
        if last is not None and last[0] == key:
            return last[1]
        return None

    def _store_psp_highmode(self, key, value):
        self._last_psp_highmode = (key, value)
        return value

    def _current_loader_identity(self, key):
        return tuple(
            _loader_signature(loader, key) if loader is not None else None
            for loader in (
                self.perambulator,
                self.vsp_propagator,
                self.psv_propagator,
                self.psp_propagator,
                self.overlap_matrix,
            )
        )

    def _validate_current_extents(self, usedNe, usedNp):
        available_ne = None
        available_np = None
        for loader, role in (
            (self.perambulator, "vsv"),
            (self.vsp_propagator, "vsp"),
            (self.psv_propagator, "psv"),
            (self.psp_propagator, "psp"),
            (self.overlap_matrix, "overlap"),
        ):
            loader_ne, loader_np = _loader_extents(loader, role)
            if loader_ne is not None:
                if available_ne is None:
                    available_ne = loader_ne
                elif available_ne != loader_ne:
                    raise ValueError("propagator inputs have inconsistent Ne")
            if loader_np is not None:
                if available_np is None:
                    available_np = loader_np
                elif available_np != loader_np:
                    raise ValueError("propagator inputs have inconsistent Np")
        return (
            _normalize_extent(usedNe, available_ne, "usedNe"),
            _normalize_extent(usedNp, available_np, "usedNp"),
        )

    def load(self, key, usedNe: int = None, usedNp: int = None):
        """Load data from all available propagators. Slicing is deferred to get-time (like parent)."""
        if self.debug:
            log_gpu_memory(f"PropagatorWithCurrent.load(before, key={key})")
            logger.debug(f"\n{'='*80}")
            logger.debug(f"PropagatorWithCurrent.load() called")
            logger.debug(f"{'='*80}")
            logger.debug(f"  key: {key}")
            logger.debug(f"  usedNe: {usedNe}")
            logger.debug(f"  usedNp: {usedNp}")
            logger.debug(f"  self.key (current): {self.key}")
            logger.debug(f"  Available propagators:")
            logger.debug(f"    VSV (perambulator): {self.perambulator is not None}")
            logger.debug(f"    VSP (vsp_propagator): {self.vsp_propagator is not None}")
            logger.debug(f"    PSV (psv_propagator): {self.psv_propagator is not None}")
            logger.debug(f"    PSP (psp_propagator): {self.psp_propagator is not None}")

        normalized_ne, normalized_np = self._validate_current_extents(usedNe, usedNp)
        identity = (
            key,
            normalized_ne,
            normalized_np,
            self._current_loader_identity(key),
        )
        if identity != self._cache_identity:
            if self.debug:
                logger.debug(f"\n  Cache identity changed, loading data...")
            self._release_resources()
            self._release_current_caches()

            # Load VSV via parent using the normalized Ne.
            if self.perambulator is not None:
                if self.debug:
                    logger.debug(f"  Loading VSV via parent...")
                    log_gpu_memory("load_VSV(before)")
                try:
                    super().load(key, normalized_ne)
                    if self.debug:
                        log_gpu_memory("load_VSV(after)")
                        if (
                            hasattr(self, "perambulator_data")
                            and self.perambulator_data is not None
                        ):
                            logger.debug(
                                f"    VSV loaded successfully, shape: {self.perambulator_data.shape}"
                            )
                        else:
                            logger.debug(f"    VSV loaded, but perambulator_data is None")
                except Exception as e:
                    if self.debug:
                        logger.debug(f"    ERROR loading VSV: {e}")
                    raise
            else:
                if self.debug:
                    logger.debug(f"  Skipping VSV (perambulator is None)")

            if self.vsp_propagator is not None:
                if self.debug:
                    logger.debug(f"  Loading VSP...")
                    log_gpu_memory("load_VSP(before)")
                try:
                    self.vsp_data = self.vsp_propagator.load(key)
                    if self.debug:
                        log_gpu_memory("load_VSP(after)")
                        if self.vsp_data is not None:
                            logger.debug(
                                f"    VSP loaded successfully, shape: {self.vsp_data.shape}"
                            )
                        else:
                            logger.debug(f"    VSP load returned None")
                except Exception as e:
                    if self.debug:
                        logger.debug(f"    ERROR loading VSP: {e}")
                    raise
            else:
                if self.debug:
                    logger.debug(f"  Skipping VSP (vsp_propagator is None)")

            if self.psv_propagator is not None:
                if self.debug:
                    logger.debug(f"  Loading PSV...")
                    log_gpu_memory("load_PSV(before)")
                try:
                    self.psv_data = self.psv_propagator.load(key)
                    if self.debug:
                        log_gpu_memory("load_PSV(after)")
                        if self.psv_data is not None:
                            logger.debug(
                                f"    PSV loaded successfully, shape: {self.psv_data.shape}"
                            )
                        else:
                            logger.debug(f"    PSV load returned None")
                except Exception as e:
                    if self.debug:
                        logger.debug(f"    ERROR loading PSV: {e}")
                    raise
            else:
                if self.debug:
                    logger.debug(f"  Skipping PSV (psv_propagator is None)")

            if self.psp_propagator is not None:
                if self.debug:
                    logger.debug(f"  Loading PSP...")
                    log_gpu_memory("load_PSP(before)")
                try:
                    self.psp_data = self.psp_propagator.load(key)
                    if self.debug:
                        log_gpu_memory("load_PSP(after)")
                        if self.psp_data is not None:
                            logger.debug(
                                f"    PSP loaded successfully, shape: {self.psp_data.shape}"
                            )
                        else:
                            logger.debug(f"    PSP load returned None")
                except Exception as e:
                    if self.debug:
                        logger.debug(f"    ERROR loading PSP: {e}")
                    raise
            else:
                if self.debug:
                    logger.debug(f"  Skipping PSP (psp_propagator is None)")

            # Load overlap matrix data from file
            if self.overlap_matrix is not None:
                if self.debug:
                    logger.debug(f"  Loading overlap matrix from file...")
                    log_gpu_memory("load_overlap_matrix(before)")
                self.overlap_matrix_data = self.overlap_matrix.load(key)[:]
                if self.debug:
                    logger.debug(
                        f"    overlap_matrix_data.shape: {self.overlap_matrix_data.shape}"
                    )
                    log_gpu_memory("load_overlap_matrix(after)")
            else:
                raise ValueError("overlap_matrix must be provided")

            # Clear high mode caches when key changes
            if self.debug:
                log_gpu_memory("clear_caches(before)")
            # tilde_S_psv is simple cache for single t_source
            self.tilde_S_psv_cache = None
            self.tilde_S_psv_cached_time = None
            # tilde_S_vsp is simple cache for single t_sink
            self.tilde_S_vsp_cache = None
            self.tilde_S_vsp_cached_time = None
            # tilde_S_psp is simple cache for single t_source
            self.tilde_S_psp_cache = None
            self.tilde_S_psp_cached_time = None
            if self.debug:
                log_gpu_memory("clear_caches(after)")

            # Update identity only after every input has loaded successfully.
            self.key = key
            self.usedNe = normalized_ne
            self.usedNp = normalized_np
            self._cache_identity = identity

            if self.debug:
                logger.debug(f"\n  Load completed successfully")
                logger.debug(f"  Updated self.key to: {self.key}")
                logger.debug(f"  Updated self.usedNe to: {self.usedNe}")
                logger.debug(f"  Updated self.usedNp to: {self.usedNp}")
                logger.debug(f"{'='*80}\n")
                log_gpu_memory(f"PropagatorWithCurrent.load(after, key={key})")
        else:
            if self.debug:
                logger.debug(f"  Cache identity unchanged, skipping load")
                logger.debug(f"{'='*80}\n")
                log_gpu_memory(f"PropagatorWithCurrent.load(skipped, key={key})")

    def get(self, t_source, t_sink):
        """
        Get VSV propagator (standard perambulator) for given source/sink times.

        Compatible with standard Propagator interface; slicing occurs in parent get().
        Only raises error if data is not available.
        """
        if self.debug:
            logger.debug(f"\nPropagatorWithCurrent.get(VSV) called:")
            logger.debug(f"  t_source: {t_source} (type: {type(t_source).__name__})")
            logger.debug(f"  t_sink: {t_sink} (type: {type(t_sink).__name__})")
            logger.debug(
                f"  perambulator_data available: {self.perambulator_data is not None}"
            )

        if self.perambulator_data is not None:
            # Delegate to parent which handles caching and usedNe slicing
            result = super().get(t_source, t_sink)
            if self.debug:
                logger.debug(f"  Result shape: {result.shape}")
            return result
        else:
            if self.debug:
                logger.debug(f"  ERROR: VSV propagator not available!")
            raise ValueError(
                "VSV propagator not provided but is required for this diagram"
            )

    def _release_resources(self):
        """Release resources for all propagator types."""
        # Release parent resources
        super()._release_resources()
        # Release additional resources
        self.vsp_data = None
        self.psv_data = None
        self.psp_data = None
        self.overlap_matrix_data = None
        self._release_current_caches()

    def _apply_gamma_on_spin(self, array_with_spin_first_two_axes):
        """
        Apply gamma(15) on both spin indices of an array whose first three axes are [Lt, Ns, Ns, ...].
        Conjugates the input as part of dagger operation.
        Returns the transformed array with the same shape and tail axes ordering preserved.
        """
        from lattice.insertion.gamma import gamma

        backend = get_backend()
        g15 = gamma(15)
        arr = array_with_spin_first_two_axes
        Lt_local, Ns_left, Ns_right = (
            int(arr.shape[0]),
            int(arr.shape[1]),
            int(arr.shape[2]),
        )
        tail_shape = tuple(arr.shape[3:])
        # Flatten tail for two-step matmul to avoid verbose einsum strings
        arr_flat = arr.conj().reshape((Lt_local, Ns_left, Ns_right, -1))
        left_applied = contract("ik,tklr->tilr", g15, arr_flat)
        both_applied = contract("tilr,lj->tijr", left_applied, g15)
        return both_applied.reshape((Lt_local, Ns_left, Ns_right) + tail_shape)

    def _dagger_vsp(self, vsp_block):
        """
        Apply dagger (gamma(15) conjugate) to VSP data.

        Input: VSP tail order [..., Ne, Np, Nc]
        Output: PSV tail order [..., Np, Nc, Ne]
        """
        spun = self._apply_gamma_on_spin(vsp_block)  # keeps same shape
        # Swap sink/source spin axes and move tail (Ne, Np, Nc) -> (Np, Nc, Ne).
        return spun.transpose(0, 2, 1, 4, 5, 3)

    def _dagger_psv(self, psv_block):
        """
        Apply dagger (gamma(15) conjugate) to PSV data.

        Input: PSV tail order [..., Np, Nc, Ne]
        Output: VSP tail order [..., Ne, Np, Nc]
        """
        spun = self._apply_gamma_on_spin(psv_block)
        # Swap sink/source spin axes and move tail (Np, Nc, Ne) -> (Ne, Np, Nc).
        return spun.transpose(0, 2, 1, 5, 3, 4)

    def _dagger_psp(self, psp_block):
        """
        Apply dagger (gamma(15) conjugate) to PSP data.

        Input: PSP tail order [..., Np_snk, Nc, Np_src, Nc]
        Output: Swapped ordering [..., Np_src, Nc, Np_snk, Nc]
        """
        spun = self._apply_gamma_on_spin(psp_block)
        # Swap the two point axes (positions 3 and 5 in tail)
        return spun.transpose(0, 1, 2, 5, 4, 3, 6)

    def get_VSP(self, t_source, t_sink, cache=True):
        """
        Get VSP propagator data with two-time interface mirroring Propagator.get.

        Args:
            t_source: Source time (int or array-like)
            t_sink: Sink time (int or array-like)
            cache: Whether to cache internal timeslice array (default True). Set False to avoid caching.

        Returns:
            If both times are int: [Ns, Ns, Ne, Np, Nc] or [Ns, Ns, Np, Nc, Ne] (if dagger->PSV)
            If one time is int and the other is array-like: [t, Ns, Ns, ...] with same tail ordering
        """
        if self.debug:
            logger.debug(f"\nPropagatorWithCurrent.get_VSP() called:")
            logger.debug(f"  t_source: {t_source} (type: {type(t_source).__name__})")
            logger.debug(f"  t_sink: {t_sink} (type: {type(t_sink).__name__})")
            logger.debug(f"  vsp_data available: {self.vsp_data is not None}")
            if self.vsp_data is not None:
                logger.debug(f"  vsp_data.shape: {self.vsp_data.shape}")
            logger.debug(f"  usedNe: {getattr(self, 'usedNe', None)}")
            logger.debug(f"  usedNp: {getattr(self, 'usedNp', None)}")

        # Populate caches based on which time is int (anchor choice same as Propagator.get)
        if isinstance(t_source, int) and isinstance(t_sink, int):
            # Check if we have cached data
            if self.vsp_cached_time == t_source:
                out = self.vsp_cache[(t_sink - t_source) % self.Lt]
                if self.debug:
                    logger.debug(f"    VSP get(two-int, cached): shape={out.shape}")
                return out
            elif self.psv_cached_time == t_sink:
                out = self.psv_dagger[(t_source - t_sink) % self.Lt]
                if self.debug:
                    logger.debug(f"    VSP get(two-int, cached dagger): shape={out.shape}")
                return out
            else:
                # Need to compute new data
                if self.vsp_data is None:
                    if self.debug:
                        logger.debug(f"  ERROR: VSP propagator not available!")
                    raise ValueError(
                        "VSP propagator not provided but is required for this diagram"
                    )

                vsp_cache_local = self.vsp_data[
                    t_source,
                    ...,
                    : self.usedNe if self.usedNe is not None else None,
                    : self.usedNp if self.usedNp is not None else None,
                    :,
                ]
                if self.debug:
                    logger.debug(f"    after slice: {vsp_cache_local.shape}")

                if cache:
                    self.vsp_cache = vsp_cache_local
                    self.vsp_dagger = self._dagger_vsp(self.vsp_cache)
                    self.vsp_cached_time = t_source
                    out = self.vsp_cache[(t_sink - t_source) % self.Lt]
                else:
                    out = vsp_cache_local[(t_sink - t_source) % self.Lt]

                if self.debug:
                    logger.debug(f"    final output: {out.shape}")
                    logger.debug(
                        f"    VSP get(two-int, {'cached' if cache else 'uncached'}): shape={out.shape}"
                    )
                return out
        elif isinstance(t_source, int):
            # Check if we have cached data
            if self.vsp_cached_time == t_source:
                out = self.vsp_cache[(t_sink - t_source) % self.Lt]
                if self.debug:
                    logger.debug(f"    VSP get(tsrc-int, cached): shape={out.shape}")
                return out
            else:
                # Need to compute new data
                if self.vsp_data is None:
                    if self.debug:
                        logger.debug(f"  ERROR: VSP propagator not available!")
                    raise ValueError(
                        "VSP propagator not provided but is required for this diagram"
                    )

                vsp_cache_local = self.vsp_data[
                    t_source,
                    ...,
                    : self.usedNe if self.usedNe is not None else None,
                    : self.usedNp if self.usedNp is not None else None,
                    :,
                ]
                if self.debug:
                    logger.debug(f"    after slice: {vsp_cache_local.shape}")

                if cache:
                    self.vsp_cache = vsp_cache_local
                    self.vsp_dagger = self._dagger_vsp(self.vsp_cache)
                    self.vsp_cached_time = t_source
                    out = self.vsp_cache[(t_sink - t_source) % self.Lt]
                else:
                    out = vsp_cache_local[(t_sink - t_source) % self.Lt]

                if self.debug:
                    logger.debug(f"    final output: {out.shape}")
                    logger.debug(
                        f"    VSP get(tsrc-int, {'cached' if cache else 'uncached'}): shape={out.shape}"
                    )
                return out
        elif isinstance(t_sink, int):
            # Need dagger: check if PSV cache available
            if self.psv_cached_time == t_sink:
                out = self.psv_dagger[(t_source - t_sink) % self.Lt]
                if self.debug:
                    logger.debug(f"    VSP get(tsink-int, cached dagger): shape={out.shape}")
                return out
            else:
                # Need to compute new data
                if self.psv_data is None:
                    if self.debug:
                        logger.debug(
                            f"  ERROR: PSV propagator not available (needed for VSP dagger)!"
                        )
                    raise ValueError(
                        "PSV propagator not provided but is required for this diagram"
                    )
                # Get and slice in one step: [Lt, Ns, Ns, Np, Nc, Ne]
                psv_cache_local = self.psv_data[
                    t_sink,
                    ...,
                    : self.usedNp if self.usedNp is not None else None,
                    :,
                    : self.usedNe if self.usedNe is not None else None,
                ]

                if cache:
                    self.psv_cache = psv_cache_local
                    self.psv_dagger = self._dagger_psv(self.psv_cache)
                    self.psv_cached_time = t_sink
                    out = self.psv_dagger[(t_source - t_sink) % self.Lt]
                else:
                    psv_dagger_local = self._dagger_psv(psv_cache_local)
                    out = psv_dagger_local[(t_source - t_sink) % self.Lt]

                if self.debug:
                    logger.debug(
                        f"    VSP get(tsink-int, {'cached' if cache else 'uncached'} dagger): shape={out.shape}"
                    )
                return out
        else:
            raise ValueError("At least t_source or t_sink should be int")

    def get_PSV(self, t_source, t_sink, cache=True):
        """
        Get PSV (point->eigen) propagator, math: S_{xa,i} = <eta_x,a | S | xi_i>.

        Args:
            t_source: Source time (int or array-like)
            t_sink: Sink time (int or array-like)
            cache: Whether to cache internal timeslice array (default True). Set False to avoid caching.

        Returns:
            If both times are int: [Ns, Ns, Np, Nc, Ne]
            If one time is int and the other is array-like: [t, Ns, Ns, Np, Nc, Ne]
        """
        if self.debug:
            logger.debug(f"\nPropagatorWithCurrent.get_PSV() called:")
            logger.debug(f"  t_source: {t_source} (type: {type(t_source).__name__})")
            logger.debug(f"  t_sink: {t_sink} (type: {type(t_sink).__name__})")
            logger.debug(f"  psv_data available: {self.psv_data is not None}")
            if self.psv_data is not None:
                logger.debug(f"  psv_data.shape: {self.psv_data.shape}")
            logger.debug(f"  usedNe: {getattr(self, 'usedNe', None)}")
            logger.debug(f"  usedNp: {getattr(self, 'usedNp', None)}")

        if isinstance(t_source, int) and isinstance(t_sink, int):
            # Check if we have cached data
            if self.psv_cached_time == t_source:
                out = self.psv_cache[(t_sink - t_source) % self.Lt]
                if self.debug:
                    logger.debug(f"    PSV get(two-int, cached): shape={out.shape}")
                return out
            elif self.vsp_cached_time == t_sink:
                out = self.vsp_dagger[(t_source - t_sink) % self.Lt]
                if self.debug:
                    logger.debug(f"    PSV get(two-int, cached dagger): shape={out.shape}")
                return out
            else:
                # Need to compute new data
                if self.psv_data is None:
                    if self.debug:
                        logger.debug(f"  ERROR: PSV propagator not available!")
                    raise ValueError(
                        "PSV propagator not provided but is required for this diagram"
                    )

                psv_cache_local = self.psv_data[
                    t_source,
                    ...,
                    : self.usedNp if self.usedNp is not None else None,
                    :,
                    : self.usedNe if self.usedNe is not None else None,
                ]
                if self.debug:
                    logger.debug(f"    after slice: {psv_cache_local.shape}")

                if cache:
                    self.psv_cache = psv_cache_local
                    self.psv_dagger = self._dagger_psv(self.psv_cache)
                    self.psv_cached_time = t_source
                    out = self.psv_cache[(t_sink - t_source) % self.Lt]
                else:
                    out = psv_cache_local[(t_sink - t_source) % self.Lt]

                if self.debug:
                    logger.debug(f"    final output: {out.shape}")
                    logger.debug(
                        f"    PSV get(two-int, {'cached' if cache else 'uncached'}): shape={out.shape}"
                    )
                return out
        elif isinstance(t_source, int):
            # Check if we have cached data
            if self.psv_cached_time == t_source:
                out = self.psv_cache[(t_sink - t_source) % self.Lt]
                if self.debug:
                    logger.debug(f"    PSV get(tsrc-int, cached): shape={out.shape}")
                return out
            else:
                # Need to compute new data
                if self.psv_data is None:
                    if self.debug:
                        logger.debug(f"  ERROR: PSV propagator not available!")
                    raise ValueError(
                        "PSV propagator not provided but is required for this diagram"
                    )

                psv_cache_local = self.psv_data[
                    t_source,
                    ...,
                    : self.usedNp if self.usedNp is not None else None,
                    :,
                    : self.usedNe if self.usedNe is not None else None,
                ]
                if self.debug:
                    logger.debug(f"    after slice: {psv_cache_local.shape}")

                if cache:
                    self.psv_cache = psv_cache_local
                    self.psv_dagger = self._dagger_psv(self.psv_cache)
                    self.psv_cached_time = t_source
                    out = self.psv_cache[(t_sink - t_source) % self.Lt]
                else:
                    out = psv_cache_local[(t_sink - t_source) % self.Lt]

                if self.debug:
                    logger.debug(f"    final output: {out.shape}")
                    logger.debug(
                        f"    PSV get(tsrc-int, {'cached' if cache else 'uncached'}): shape={out.shape}"
                    )
                return out
        elif isinstance(t_sink, int):
            # Need dagger: check if VSP cache available
            if self.vsp_cached_time == t_sink:
                out = self.vsp_dagger[(t_source - t_sink) % self.Lt]
                if self.debug:
                    logger.debug(f"    PSV get(tsink-int, cached dagger): shape={out.shape}")
                return out
            else:
                # Need to compute new data
                if self.vsp_data is None:
                    if self.debug:
                        logger.debug(
                            f"  ERROR: VSP propagator not available (needed for PSV dagger)!"
                        )
                    raise ValueError(
                        "VSP propagator not provided but is required for this diagram"
                    )
                # Get and slice in one step: [Lt, Ns, Ns, Ne, Np, Nc]
                vsp_cache_local = self.vsp_data[
                    t_sink,
                    ...,
                    : self.usedNe if self.usedNe is not None else None,
                    : self.usedNp if self.usedNp is not None else None,
                    :,
                ]

                if cache:
                    self.vsp_cache = vsp_cache_local
                    self.vsp_dagger = self._dagger_vsp(self.vsp_cache)
                    self.vsp_cached_time = t_sink
                    out = self.vsp_dagger[(t_source - t_sink) % self.Lt]
                else:
                    vsp_dagger_local = self._dagger_vsp(vsp_cache_local)
                    out = vsp_dagger_local[(t_source - t_sink) % self.Lt]

                if self.debug:
                    logger.debug(
                        f"    PSV get(tsink-int, {'cached' if cache else 'uncached'} dagger): shape={out.shape}"
                    )
                return out
        else:
            raise ValueError("At least t_source or t_sink should be int")

    def get_PSP(self, t_source, t_sink, cache=True):
        """
        Get PSP (point->point) propagator, math: S_{xa,yb} = <eta_x,a | S | eta_y,b>.

        Args:
            t_source: Source time (int or array-like)
            t_sink: Sink time (int or array-like)
            cache: Whether to cache internal timeslice array (default True). Set False to avoid caching.

        Returns:
            If both times are int: [Ns, Ns, Np_snk, Nc, Np_src, Nc]
            If one time is int and the other is array-like: [t, Ns, Ns, Np_snk, Nc, Np_src, Nc]
        """
        if self.debug:
            logger.debug(f"\nPropagatorWithCurrent.get_PSP() called:")
            logger.debug(f"  t_source: {t_source} (type: {type(t_source).__name__})")
            logger.debug(f"  t_sink: {t_sink} (type: {type(t_sink).__name__})")
            logger.debug(f"  psp_data available: {self.psp_data is not None}")
            if self.psp_data is not None:
                logger.debug(f"  psp_data.shape: {self.psp_data.shape}")
            logger.debug(f"  usedNp: {getattr(self, 'usedNp', None)}")

        if self.psp_data is None:
            if self.debug:
                logger.debug(f"  ERROR: PSP propagator not available!")
            raise ValueError(
                "PSP propagator not provided but is required for this diagram"
            )
        if isinstance(t_source, int) and isinstance(t_sink, int):
            # Check if we have cached data
            if self.psp_cached_time == t_source:
                out = self.psp_cache[(t_sink - t_source) % self.Lt]
                if self.debug:
                    logger.debug(f"    PSP get(two-int, cached): shape={out.shape}")
                return out
            elif self.psp_cached_time == t_sink:
                out = self.psp_dagger[(t_source - t_sink) % self.Lt]
                if self.debug:
                    logger.debug(f"    PSP get(two-int, cached dagger): shape={out.shape}")
                return out
            else:
                psp_cache_local = self.psp_data[
                    t_source
                ]  # [Lt, Ns, Ns, Np_snk, Nc, Np_src, Nc]
                if getattr(self, "usedNp", None) is not None:
                    psp_cache_local = psp_cache_local[
                        ..., : self.usedNp, :, : self.usedNp, :
                    ]
                    if self.debug:
                        logger.debug(f"    after slice: {psp_cache_local.shape}")

                if cache:
                    # Save to cache
                    self.psp_cache = psp_cache_local
                    self.psp_dagger = self._dagger_psp(self.psp_cache)
                    self.psp_cached_time = t_source
                    out = self.psp_cache[(t_sink - t_source) % self.Lt]
                else:
                    # Don't save to cache, just compute result
                    out = psp_cache_local[(t_sink - t_source) % self.Lt]

                if self.debug:
                    logger.debug(f"    final output: {out.shape}")
                    logger.debug(
                        f"    PSP get(two-int, {'cached' if cache else 'uncached'}): shape={out.shape}"
                    )
                return out
        elif isinstance(t_source, int):
            # Check if we have cached data
            if self.psp_cached_time == t_source:
                out = self.psp_cache[(t_sink - t_source) % self.Lt]
                if self.debug:
                    logger.debug(f"    PSP get(tsrc-int, cached): shape={out.shape}")
                return out
            else:
                psp_cache_local = self.psp_data[t_source]
                if getattr(self, "usedNp", None) is not None:
                    psp_cache_local = psp_cache_local[
                        ..., : self.usedNp, :, : self.usedNp, :
                    ]
                    if self.debug:
                        logger.debug(f"    after slice: {psp_cache_local.shape}")

                if cache:
                    self.psp_cache = psp_cache_local
                    self.psp_dagger = self._dagger_psp(self.psp_cache)
                    self.psp_cached_time = t_source
                    out = self.psp_cache[(t_sink - t_source) % self.Lt]
                else:
                    out = psp_cache_local[(t_sink - t_source) % self.Lt]

                if self.debug:
                    logger.debug(f"    final output: {out.shape}")
                    logger.debug(
                        f"    PSP get(tsrc-int, {'cached' if cache else 'uncached'}): shape={out.shape}"
                    )
                return out
        elif isinstance(t_sink, int):
            # Check if we have cached data
            if self.psp_cached_time == t_sink:
                out = self.psp_dagger[(t_source - t_sink) % self.Lt]
                if self.debug:
                    logger.debug(f"    PSP get(tsink-int, cached dagger): shape={out.shape}")
                return out
            else:
                # Need to compute new data
                if self.debug:
                    logger.debug(f"  PSP shape process (dagger):")
                    logger.debug(
                        f"    raw psp_data[t_sink={t_sink}]: {self.psp_data[t_sink].shape}"
                    )
                psp_cache_local = self.psp_data[t_sink]
                if getattr(self, "usedNp", None) is not None:
                    psp_cache_local = psp_cache_local[
                        ..., : self.usedNp, :, : self.usedNp, :
                    ]
                    if self.debug:
                        logger.debug(f"    after slice: {psp_cache_local.shape}")
                psp_dagger_local = (
                    self._dagger_psp(psp_cache_local) if not cache else None
                )
                if self.debug and psp_dagger_local is not None:
                    logger.debug(f"    after dagger: {psp_dagger_local.shape}")

                if cache:
                    self.psp_cache = psp_cache_local
                    self.psp_dagger = self._dagger_psp(self.psp_cache)
                    self.psp_cached_time = t_sink
                    out = self.psp_dagger[(t_source - t_sink) % self.Lt]
                else:
                    out = psp_dagger_local[(t_source - t_sink) % self.Lt]

                if self.debug:
                    logger.debug(f"    final output: {out.shape}")
                    logger.debug(
                        f"    PSP get(tsink-int, {'cached' if cache else 'uncached'} dagger): shape={out.shape}"
                    )
                return out
        else:
            raise ValueError("At least t_source or t_sink should be int")

    def get_VSP_highmode(
        self, t_source, t_sink, usedNe_source=None, *, usedNe_sink=None
    ):
        """
        Get VSP high-mode (eigen->point, projected) propagator.

        Math mapping:
          - unprojected: S_{i,xa} = <xi_i | S | eta_x,a>
          - projected:   \tilde{S}_{i,xa} = S_{i,xa} - \sum_j S_{i,j} M_{jx,a}^*

        Only applies projection when usedNe > 0. When usedNe == 0, returns unprojected VSP.

        Args:
            t_source: Source time
            t_sink: Sink time
            usedNe_source: Number of eigenvectors for source (right) end (default: self.usedNe)
                          When != self.usedNe, no caching is performed for highmode result.

        Returns:
            Same shape as get_VSP: [Ns, Ns, Ne, Np, Nc] or [t, Ns, Ns, Ne, Np, Nc]
        """
        # VSP's output eigen index belongs to the sink; its subtraction index belongs to the source.
        usedNe_source = _normalize_extent(
            usedNe_source, self.usedNe, "usedNe_source"
        )
        usedNe_sink = _normalize_extent(
            usedNe_sink, self.usedNe, "usedNe_sink"
        )

        if usedNe_source == 0:
            return self.get_VSP(t_source, t_sink, cache=False)[
                ..., :usedNe_sink, :, :
            ]

        log_gpu_memory(
            f"get_VSP_highmode(before, t_source={t_source}, t_sink={t_sink})"
        )
        if self.debug:
            logger.debug(f"\nget_VSP_highmode() called:")
            logger.debug(f"  t_source: {t_source}, t_sink: {t_sink}")
            logger.debug(f"  usedNe_source: {usedNe_source}, self.usedNe: {self.usedNe}")

        backend = get_backend()

        # Determine caching strategy
        should_cache_highmode = (
            usedNe_source == self.usedNe and usedNe_sink == self.usedNe
        )
        should_cache_unprojected = not should_cache_highmode
        is_single_time = isinstance(t_source, int) and isinstance(t_sink, int)
        if is_single_time:
            should_cache_highmode = False
            should_cache_unprojected = True

        # Check highmode cache (only when should_cache_highmode and is_single_time)
        if should_cache_highmode:
            if is_single_time:
                t_rel = (t_sink - t_source) % self.Lt
                if self.tilde_S_vsp_cached_time == t_source:
                    t_rel = (t_sink - t_source) % self.Lt
                    return self.tilde_S_vsp_cache[t_rel]
                elif self.tilde_S_psv_cached_time == t_sink:
                    t_rel = (t_source - t_sink) % self.Lt
                    return self.tilde_S_psv_dagger[t_rel]
            else:
                if isinstance(t_source, int):
                    t_rel = (t_sink - t_source) % self.Lt
                    if self.tilde_S_vsp_cached_time == t_source:
                        return self.tilde_S_vsp_cache[t_rel]
                else:
                    t_rel = (t_source - t_sink) % self.Lt
                    if self.tilde_S_vsp_cached_time == t_sink:
                        return self.tilde_S_psv_dagger[t_rel]
        # Get original VSP: S_{i,xa}
        # Cache unprojected when usedNe != self.usedNe (because we won't cache highmode)
        # Don't cache when usedNe == self.usedNe (because highmode result will be cached)
        S_vsp = self.get_VSP(t_source, t_sink, cache=should_cache_unprojected)
        S_vsp = S_vsp[..., :usedNe_sink, :, :]

        # Get VSV: S_{i,j}
        S_vsv = self.get(t_source, t_sink)

        # Get overlap matrix M (full) and slice to usedNe_source
        M_full = self.overlap_matrix_data[
            :, : self.usedNe, : self.usedNp, :
        ]  # [Lt, Ne, Np, Nc]
        M = M_full[:, :usedNe_source, :, :]
        M_conj = M.conj()

        if is_single_time:
            M_conj_t = M_conj[t_source]  # [Ne, Np, Nc]

            # Compute: sum_j S_{i,j} M_{jx,a}^*
            # S_vsv: [Ns_snk, Ns_src, Ne_i, Ne_j]
            # M_conj_t: [Ne_j, Np_x, Nc_c] where M_{jx,c}^* = <xi_j| eta_x,c>
            #   Index order: j (Ne, 0th), x (Np, 1st), c (Nc, 2nd)
            # Result: [Ns_snk, Ns_src, Ne_i, Np_x, Nc_c]
            correction = contract(
                "abij,jxc->abixc",
                S_vsv[:, :, :usedNe_sink, :usedNe_source],
                M_conj_t,
            )

            # tilde{S} = S - correction
            tilde_S = S_vsp - correction
            log_gpu_memory(f"get_VSP_highmode(after, single_time)")
            return tilde_S
        else:
            # Multi-time case
            if isinstance(t_source, int):
                t_rel = (backend.asarray(t_sink) - t_source) % self.Lt
                M_conj_t = M_conj[t_source]  # [Ne, Np, Nc]
                correction = contract(
                    "tabij,jxc->tabixc",
                    S_vsv[:, :, :, :usedNe_sink, :usedNe_source],
                    M_conj_t,
                )
                tilde_S = S_vsp - correction
                if should_cache_highmode:
                    if self.debug:
                        logger.debug(f"caching full tilde_S_vsp for t_source={t_source}")
                    self.tilde_S_vsp_cache = tilde_S
                    self.tilde_S_vsp_dagger = self._dagger_vsp(tilde_S)
                    self.tilde_S_vsp_cached_time = t_source
                log_gpu_memory(f"get_VSP_highmode(after, multi_time, t_source=int)")
                return tilde_S[t_rel]
            else:  # t_sink is int
                t_rel = (backend.asarray(t_source) - t_sink) % self.Lt
                M_conj_t = M_conj[t_rel]  # [Lt, Ne, Np, Nc]
                correction = contract(
                    "tabij,tjxc->tabixc",
                    S_vsv[:, :, :, :usedNe_sink, :usedNe_source],
                    M_conj_t,
                )
                tilde_S = S_vsp - correction
                if should_cache_highmode:
                    if self.debug:
                        logger.debug(f"caching full tilde_S_vsp for t_source={t_source}")
                    self.tilde_S_psv_dagger = tilde_S
                    self.tilde_S_psv_cache = self._dagger_vsp(tilde_S)
                    self.tilde_S_psv_cached_time = t_sink
                    return self.tilde_S_psv_dagger
                if self.debug:
                    logger.debug(f"  tilde_S_vsp shape: {tilde_S.shape}")

                log_gpu_memory(f"get_VSP_highmode(after, multi_time, t_sink=int)")
                return tilde_S[t_rel]

    def get_PSV_highmode(
        self, t_source, t_sink, usedNe_sink=None, *, usedNe_source=None
    ):
        """
        Get PSV high-mode (point->eigen, projected) propagator.

        Math mapping:
          - unprojected: S_{xa,i} = <eta_x,a | S | xi_i>
          - projected:   \tilde{S}_{xa,i} = S_{xa,i} - \sum_j M_{xj,a} S_{j,i}

        Only applies projection when usedNe > 0. When usedNe == 0, returns unprojected PSV.
        Caches per t_source only when usedNe_sink == self.usedNe.

        Args:
            t_source: Source time
            t_sink: Sink time
            usedNe_sink: Number of eigenvectors for sink (left) end (default: self.usedNe)
                        When != self.usedNe, no caching is performed for highmode result.

        Returns:
            Same shape as get_PSV: [Ns, Ns, Np, Nc, Ne] or [t, Ns, Ns, Np, Nc, Ne]
        """
        # PSV's output eigen index belongs to the source; its subtraction index belongs to the sink.
        usedNe_sink = _normalize_extent(usedNe_sink, self.usedNe, "usedNe_sink")
        usedNe_source = _normalize_extent(
            usedNe_source, self.usedNe, "usedNe_source"
        )

        if usedNe_sink == 0:
            return self.get_PSV(t_source, t_sink, cache=False)[
                ..., :usedNe_source
            ]

        log_gpu_memory(
            f"get_PSV_highmode(before, t_source={t_source}, t_sink={t_sink})"
        )
        if self.debug:
            logger.debug(f"\nget_PSV_highmode() called:")
            logger.debug(f"  t_source: {t_source}, t_sink: {t_sink}")
            logger.debug(f"  usedNe_sink: {usedNe_sink}, self.usedNe: {self.usedNe}")

        backend = get_backend()

        # Determine caching strategy
        should_cache_highmode = (
            usedNe_sink == self.usedNe and usedNe_source == self.usedNe
        )
        should_cache_unprojected = not should_cache_highmode
        is_single_time = isinstance(t_source, int) and isinstance(t_sink, int)
        if is_single_time:
            should_cache_highmode = False
            should_cache_unprojected = True

        # Check cache only if should_cache_highmode and is_single_time
        if should_cache_highmode:
            if is_single_time:
                if self.tilde_S_psv_cached_time == t_source:
                    t_rel = (t_sink - t_source) % self.Lt
                    if self.debug:
                        logger.debug(f"  Using cached tilde_S_psv")
                    return self.tilde_S_psv_cache[t_rel]
                elif self.tilde_S_vsp_cached_time == t_sink:
                    t_rel = (t_source - t_sink) % self.Lt
                    if self.debug:
                        logger.debug(f"  Using cached tilde_S_vsp_dagger")
                    t_rel = (t_source - t_sink) % self.Lt
                    return self.tilde_S_vsp_dagger[t_rel]
            else:
                if isinstance(t_source, int):
                    t_rel = (t_sink - t_source) % self.Lt
                    if self.tilde_S_psv_cached_time == t_source:
                        if self.debug:
                            logger.debug(f"  Using cached tilde_S_psv")
                        return self.tilde_S_psv_cache[t_rel]
                else:
                    t_rel = (t_source - t_sink) % self.Lt
                    if self.tilde_S_psv_cached_time == t_sink:
                        if self.debug:
                            logger.debug(f"  Using cached tilde_S_vsp_dagger")
                        return self.tilde_S_vsp_dagger[t_rel]

        # Get original PSV: S_{xa,i} (cache unprojected when usedNe != self.usedNe)
        S_psv = self.get_PSV(t_source, t_sink, cache=should_cache_unprojected)
        S_psv = S_psv[..., :usedNe_source]

        # Get VSV: S_{j,i}
        S_vsv = self.get(t_source, t_sink)

        # Get overlap matrix M (full) and slice to usedNe_sink
        M_full = self.overlap_matrix_data[
            :, : self.usedNe, : self.usedNp, :
        ]  # [Lt, Ne, Np, Nc]
        M = M_full[:, :usedNe_sink, :, :]

        if is_single_time:
            M_t = M[t_sink]  # [Ne, Np, Nc]

            # Compute: sum_j M_{xj,c} S_{j,i}
            # Formula 5.2: tilde{S}_{xc,i} = S_{xc,i} - sum_j M_{xj,c} S_{j,i}
            # M_t: [Ne, Np, Nc] where M_t[j, x, c] = M_{xj,c} = <eta_x,c| xi_j>
            #   Index order: j (Ne, 0th), x (Np, 1st), c (Nc, 2nd)
            # S_vsv: [Ns, Ns, Ne, Ne] where S_vsv[s1, s2, j, i] = S_{j,i} = <xi_j| S |xi_i>
            #   j is sink/eigenvector (Ne, 2nd), i is source/eigenvector (Ne, 3rd)
            # S_psv: [Ns, Ns, Np, Nc, Ne] where S_psv[s1, s2, x, c, i] = S_{xc,i} = <eta_x,c| S |xi_i>
            # Contract j: sum_j M_t[j, x, c] * S_vsv[s1, s2, j, i] -> [s1, s2, x, c, i]
            correction = contract(
                "jxc,abji->abxci",
                M_t,
                S_vsv[:, :, :usedNe_sink, :usedNe_source],
            )

            # tilde{S} = S - correction
            tilde_S = S_psv - correction
            log_gpu_memory(f"get_PSV_highmode(after, single_time)")
            return tilde_S
        else:
            # Multi-time case
            if isinstance(t_source, int):
                t_rel = (backend.asarray(t_sink) - t_source) % self.Lt
                M_t = M[t_rel]  # [t, Ne, Np, Nc]

                # M_t: [t, Ne, Np, Nc] where M_t[t, j, x, c] = M_{xj,c} = <eta_x,c| xi_j>
                #   Index order: t (Lt, 0th), j (Ne, 1st), x (Np, 2nd), c (Nc, 3rd)
                # S_vsv: [t, Ns, Ns, Ne, Ne] where S_{j,i} = <xi_j| S |xi_i>
                #   Index order: t, s1, s2, j (sink, 3rd), i (source, 4th)
                # Contract j: sum_j M_t[t, j, x, c] * S_vsv[t, s1, s2, j, i] -> [t, s1, s2, x, c, i]
                correction = contract(
                    "tjxc,tabji->tabxci",
                    M_t,
                    S_vsv[:, :, :, :usedNe_sink, :usedNe_source],
                )
                tilde_S = S_psv - correction
                if should_cache_highmode:
                    if self.debug:
                        logger.debug(f"caching full tilde_S_psv for t_source={t_source}")
                    self.tilde_S_psv_cache = tilde_S
                    self.tilde_S_psv_dagger = self._dagger_psv(tilde_S)
                    self.tilde_S_psv_cached_time = t_source
                log_gpu_memory(f"get_PSV_highmode(after, multi_time, t_source=int)")
                return tilde_S[t_rel]
            else:  # t_sink is int
                t_rel = (backend.asarray(t_source) - t_sink) % self.Lt
                M_t = M[t_sink]  # [t, Ne, Np, Nc]
                # M_t: [t, Ne, Np, Nc] where M_t[t, j, x, c] = M_{xj,c} = <eta_x,c| xi_j>
                #   Index order: t (Lt, 0th), j (Ne, 1st), x (Np, 2nd), c (Nc, 3rd)
                # S_vsv: [t, Ns, Ns, Ne, Ne] where S_{j,i} = <xi_j| S |xi_i>
                #   Index order: t, s1, s2, j (sink, 3rd), i (source, 4th)
                # Contract j: sum_j M_t[t, j, x, c] * S_vsv[t, s1, s2, j, i] -> [t, s1, s2, x, c, i]
                correction = contract(
                    "tjxc,abji->tabxci",
                    M_t,
                    S_vsv[:, :, :, :usedNe_sink, :usedNe_source],
                )
                tilde_S = S_psv - correction
                if should_cache_highmode:
                    if self.debug:
                        logger.debug(f"caching full tilde_S_psv for t_source={t_source}")
                    self.tilde_S_psv_cache = tilde_S
                    self.tilde_S_psv_dagger = self._dagger_psv(tilde_S)
                    self.tilde_S_psv_cached_time = t_source
                log_gpu_memory(f"get_PSV_highmode(after, multi_time, t_sink=int)")
                return tilde_S[t_rel]

    def get_PSP_highmode(self, t_source, t_sink, usedNe_sink=None, usedNe_source=None):
        """Return the PSP high-mode propagator, reusing an identical last request."""
        normalized_sink = _normalize_extent(usedNe_sink, self.usedNe, "usedNe_sink")
        normalized_source = _normalize_extent(
            usedNe_source, self.usedNe, "usedNe_source"
        )
        cache_key = self._psp_highmode_cache_key(
            t_source, t_sink, normalized_sink, normalized_source
        )
        cached = self._cached_psp_highmode(cache_key)
        if cached is not None:
            return cached
        value = self._compute_PSP_highmode(
            t_source, t_sink, normalized_sink, normalized_source
        )
        return self._store_psp_highmode(cache_key, value)

    def _compute_PSP_highmode(self, t_source, t_sink, usedNe_sink=None, usedNe_source=None):
        """
        Get PSP propagator with high mode projection applied.

        New mapping (no conjugates):
          tilde{S}_{xa,yb} = S_{xa,yb}
              - \sum_i M_{xi,a} \, tilde{S}_{i,yb}
              - \sum_j S_{xa,j} \, M_{jy,b}

        Only applies projection when usedNe > 0. When usedNe == 0, returns unprojected PSP.
        No caching when usedNe parameters != self.usedNe.

        Args:
            t_source: Source time
            t_sink: Sink time
            usedNe_sink: Number of eigenvectors for sink (left) end (default: self.usedNe)
            usedNe_source: Number of eigenvectors for source (right) end (default: self.usedNe)

        Returns:
            Array shape: [Ns, Ns, Np_snk, Nc, Np_src, Nc] or [t, Ns, Ns, Np_snk, Nc, Np_src, Nc]
        """
        usedNe_sink = _normalize_extent(
            usedNe_sink, self.usedNe, "usedNe_sink"
        )
        usedNe_source = _normalize_extent(
            usedNe_source, self.usedNe, "usedNe_source"
        )

        # If no eigenvectors, return unprojected
        if (usedNe_sink == 0) and (usedNe_source == 0):
            return self.get_PSP(t_source, t_sink)

        log_gpu_memory(
            f"get_PSP_highmode(before, t_source={t_source}, t_sink={t_sink})"
        )
        if self.debug:
            logger.debug(f"\nget_PSP_highmode() called:")
            logger.debug(f"  t_source: {t_source}, t_sink: {t_sink}")
            logger.debug(
                f"  usedNe_sink: {usedNe_sink}, usedNe_source: {usedNe_source}, self.usedNe: {self.usedNe}"
            )

        backend = get_backend()

        # Determine caching strategy
        should_cache_highmode = (
            usedNe_sink == self.usedNe and usedNe_source == self.usedNe
        )
        # Cache unprojected when any usedNe != self.usedNe (won't cache highmode)
        should_cache_unprojected = (
            usedNe_sink != self.usedNe or usedNe_source != self.usedNe
        )
        is_single_time = isinstance(t_source, int) and isinstance(t_sink, int)
        if is_single_time:
            should_cache_highmode = False
            should_cache_unprojected = True

        # Check cache only when should_cache_highmode
        if usedNe_sink == self.usedNe and usedNe_source == self.usedNe:
            if is_single_time:
                if self.tilde_S_psp_cached_time == t_source:
                    t_rel = (t_sink - t_source) % self.Lt
                    return self.tilde_S_psp_cache[t_rel]
                elif self.tilde_S_psp_cached_time == t_sink:
                    t_rel = (t_source - t_sink) % self.Lt
                    return self.tilde_S_psp_dagger[t_rel]
            else:
                # Multi-time case
                if isinstance(t_source, int):
                    t_rel = (backend.asarray(t_sink) - t_source) % self.Lt
                    if self.tilde_S_psp_cached_time == t_source:
                        if self.debug:
                            logger.debug(f"  Using cached tilde_S_psp")
                        return self.tilde_S_psp_cache[t_rel]
                else:  # t_sink is int
                    t_rel = (backend.asarray(t_source) - t_sink) % self.Lt
                    if self.tilde_S_psp_cached_time == t_sink:
                        if self.debug:
                            logger.debug(f"  Using cached tilde_S_psp (dagger)")
                        return self.tilde_S_psp_dagger[t_rel]

        # Get original PSP: S_{xa,yb}
        S_psp = self.get_PSP(t_source, t_sink, cache=should_cache_unprojected)
        # Slice if needed (PSP doesn't have Ne dimension to slice directly)

        # Get PSV for mixed term 3: S_{xa,j}
        S_psv = self.get_PSV(t_source, t_sink, cache=should_cache_unprojected)
        # Slice to usedNe_source if needed
        if usedNe_source != self.usedNe:
            S_psv = S_psv[..., :usedNe_source]

        # Get tilde VSP for mixed term 2: tilde{S}_{i,yb}
        S_vsp_tilde = self.get_VSP_highmode(
            t_source,
            t_sink,
            usedNe_source=usedNe_source,
            usedNe_sink=usedNe_sink,
        )

        # Get overlap matrix M (full) and slice for sink and source
        M_full = self.overlap_matrix_data[
            :, : self.usedNe, : self.usedNp, :
        ]  # [Lt, Ne, Np, Nc]
        M_sink = M_full[:, :usedNe_sink, :, :] if usedNe_sink != self.usedNe else M_full
        M_source = (
            M_full[:, :usedNe_source, :, :] if usedNe_source != self.usedNe else M_full
        ).conj()

        if is_single_time:
            t_rel = (t_sink - t_source) % self.Lt
            M_sink_t = M_sink[t_sink]  # [Ne_sink, Np, Nc]
            M_source_t = M_source[t_source]  # [Ne_source, Np, Nc]

            # S_psp: [Ns_snk, Ns_src, Np_x, Nc, Np_y, Nc] where indices are [a, b, x, c, y, d]
            #   a, b: spin indices, x: sink point, c: sink color, y: source point, d: source color
            # S_vsp_tilde: [Ns_snk, Ns_src, Ne_i, Np_y, Nc] where indices are [a, b, i, y, d]
            #   d is source color (same as S_psp's source color)
            # S_psv: [Ns_snk, Ns_src, Np_x, Nc, Ne_j] where indices are [a, b, x, c, j]
            #   c is sink color (same as S_psp's sink color)

            # Term 1: S_{xa,yb}
            term1 = S_psp

            # Term 2: - sum_i M_{xi,c} tilde{S}_{i,yc}
            # Only compute if usedNe_sink > 0 (usedNe=0 means no projection)
            if usedNe_sink == 0:
                term2 = 0
            else:
                # M_sink_t: [Ne_i, Np_x, Nc] where M_{xi,c} = <eta_x,c| xi_i>
                #   Index order: i (Ne, 0th), x (Np, 1st), c (Nc, 2nd) - c is sink color
                # S_vsp_tilde: [Ns, Ns, Ne_i, Np_y, Nc] (already uses usedNe_source)
                #   Index order: a (Ns, 0th), b (Ns, 1st), i (Ne, 2nd), y (Np, 3rd), d (Nc, 4th) - d is source color
                # Contract i: sum_i M_sink_t[i, x, c] * S_vsp_tilde[a, b, i, y, d] -> [a, b, x, c, y, d]
                # Note: c is sink color, d is source color (two independent color dimensions)
                term2 = contract("ixc,abiyd->abxcyd", M_sink_t, S_vsp_tilde)

            # Term 3: - sum_j S_{xc,j} M_{jy,d}
            # Formula 5.1: M_{jy,d} = <xi_j|eta_y,d> = M_{yj,d}^*
            # Only compute if usedNe_source > 0
            if usedNe_source == 0:
                term3 = 0
            else:
                # S_psv: [Ns, Ns, Np_x, Nc, Ne_j] where S_{xc,j} = <eta_x,c| S |xi_j>
                #   Index order: a (Ns, 0th), b (Ns, 1st), x (Np, 2nd), c (Nc, 3rd), j (Ne, 4th)
                #   c is sink color, j is source (right/ket, 4th dim)
                # M_source_t: [Ne_j, Np_y, Nc] where M_{yj,d} = <eta_y,d| xi_j>
                #   Index order: j (Ne, 0th), y (Np, 1st), d (Nc, 2nd) - d is source color
                #   M_{jy,d} = M_{yj,d}^*, but we use M_{yj,d} directly since contract handles conjugation
                # Contract j: sum_j S_psv[a, b, x, c, j] * M_source_t[j, y, d] -> [a, b, x, c, y, d]
                # Note: c is sink color, d is source color (two independent color dimensions)
                term3 = contract(
                    "abxcj,jyd->abxcyd", S_psv[:, :, :, :, :usedNe_source], M_source_t
                )

            # Combine: tilde{S} = term1 - term2 - term3
            tilde_S = term1 - term2 - term3
            log_gpu_memory(f"get_PSP_highmode(after, single_time)")
        else:
            # Multi-time case
            term1 = S_psp
            if isinstance(t_source, int):
                t_rel = (backend.asarray(t_sink) - t_source) % self.Lt
                M_sink_t = M_sink[t_rel]  # [Ne_sink, Np, Nc]
                M_source_t = M_source[t_source]  # [Ne_source, Np, Nc]
                if usedNe_sink == 0:
                    term2 = 0
                else:
                    term2 = contract("ixc,tabiyd->tabxcyd", M_sink_t, S_vsp_tilde)
                if usedNe_source == 0:
                    term3 = 0
                else:
                    term3 = contract(
                        "abxcj,jyd->tabxcyd",
                        S_psv[:, :, :, :, :usedNe_source],
                        M_source_t,
                    )
            else:  # t_sink is int
                t_rel = (backend.asarray(t_source) - t_sink) % self.Lt
                M_sink_t = M_sink[t_sink]  # [Ne_sink, Np, Nc]
                M_source_t = M_source[t_rel]  # [Ne_source, Np, Nc]
                if usedNe_sink == 0:
                    term2 = 0
                else:
                    term2 = contract("ixc,abiyd->tabxcyd", M_sink_t, S_vsp_tilde)
                if usedNe_source == 0:
                    term3 = 0
                else:
                    term3 = contract(
                        "abxcj,tjyd->tabxcyd",
                        S_psv[:, :, :, :, :usedNe_source],
                        M_source_t,
                    )
            tilde_S = term1 - term2 - term3
            if should_cache_highmode:
                if isinstance(t_source, int):
                    if self.debug:
                        logger.debug(f"caching full tilde_S_psp for t_source={t_source}")
                    self.tilde_S_psp_cache = tilde_S
                    self.tilde_S_psp_dagger = self._dagger_psp(tilde_S)
                    self.tilde_S_psp_cached_time = t_source
                else:  # t_sink is int
                    if self.debug:
                        logger.debug(f"caching full tilde_S_psp for t_sink={t_sink}")
                    self.tilde_S_psp_cache = self._dagger_psp(tilde_S)
                    self.tilde_S_psp_dagger = tilde_S
                    self.tilde_S_psp_cached_time = t_sink

        log_gpu_memory(f"get_PSP_highmode(after, multi_time)")
        if self.debug:
            logger.debug(f"  tilde_S_psp shape: {tilde_S.shape}")

        return tilde_S
