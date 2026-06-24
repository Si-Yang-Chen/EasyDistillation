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

    def _release_resources(self):
        self.elemental_data = None
        self.cache = {}
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
        self.usedNe = usedNe
        if self.key != key:
            self._release_resources()
            self.key = key
            self.elemental_data = self.elemental.load(key)
            backend = get_backend()
            self.cache: Dict[int, backend.ndarray] = {}
            self._make_cache()

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
        if self.key != key:
            self.key = key
            # Load elemental data (v2v) as before
            self.elemental_data = self.elemental.load(key)
            self.Lt = self.elemental_data.shape[2]

            # NO LONGER LOAD: eigenvector, point, gauge_field
            # These are only needed if computing on-the-fly

        # Set usedNe and usedNp with defaults from data loaders if not provided
        self.usedNe = usedNe if usedNe is not None else self.elemental_data.shape[3]
        # Get usedNp from p2v_data (since v2p now uses p2v via symmetry)
        self.usedNp = (
            usedNp
            if usedNp is not None
            else (self.p2v_data.Np if self.p2v_data is not None else None)
        )

        backend = get_backend()
        # Cache dictionaries for different propagator types
        self.cache: Dict[int, backend.ndarray] = {}
        self.cache_v2p: Dict[int, backend.ndarray] = {}
        self.cache_p2v: Dict[int, backend.ndarray] = {}
        self.cache_p2p: Dict[int, Dict[tuple, backend.ndarray] | None] = {}
        self._make_cache()

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
                elemental_coeff, gaugelink_idx, momentum_idx = elemental_part[j]
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


