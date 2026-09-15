from typing import List, Literal

from opt_einsum import contract

from ..constant import Nc, Ns, Nd
from ..backend import set_backend, get_backend, check_QUDA
from ..preset import GaugeField, Eigenvector, PointSource
import numpy as np


def sampled_point_local_indices(point_data, point_count, latt_info):
    """Return local checkerboard indices for sampled sink points on this rank."""
    Lx, Ly, Lz, Lt = latt_info.size
    gx, gy, gz, gt = latt_info.grid_coord
    global_times = np.arange(gt * Lt, (gt + 1) * Lt, dtype=np.int64)
    coordinates = point_data[:point_count, global_times, :]
    if hasattr(coordinates, "get"):
        coordinates = coordinates.get()
    coordinates = np.asarray(coordinates)

    point_indices, local_times = np.indices((point_count, Lt), dtype=np.int64)
    x_global = coordinates[..., 0]
    y_global = coordinates[..., 1]
    z_global = coordinates[..., 2]
    owned = (
        (gx * Lx <= x_global)
        & (x_global < (gx + 1) * Lx)
        & (gy * Ly <= y_global)
        & (y_global < (gy + 1) * Ly)
        & (gz * Lz <= z_global)
        & (z_global < (gz + 1) * Lz)
    )

    point_indices = point_indices[owned]
    local_times = local_times[owned]
    x_local = x_global[owned] - gx * Lx
    y_local = y_global[owned] - gy * Ly
    z_local = z_global[owned] - gz * Lz
    parity = (local_times + x_local + y_local + z_local) % 2
    return point_indices, local_times, parity, z_local, y_local, x_local // 2


class PerambulatorGenerator:
    """
     Generate perambulators in distillation,
        based on PyQUDA + QUDA for GPU-accelerated computations.

    Parameters:
    -----------
    latt_size : List[int]
        dimensions of the lattice, order as [Lx, Ly, Lz, Lt].
    gauge_field : GaugeField.
    eigenvector : Eigenvector.
    mass : float
        The mass parameter for the Dirac operator.
    tol : float
        The Dirac operator tol.
    maxiter : int
        The maximum number of iterations of solver.
    xi_0 : float, optional
        The anisotropy, defaults to 1.0.
    nu : float, optional.
    clover_coeff_t : float, optional
        The temporal clover coefficient, defaults to 0.0.
    clover_coeff_r : float, optional
        The spatial clover coefficient, defaults to 1.0.
    t_boundary : Literal[1, -1], optional
        The temporal boundary condition, defaults to 1 (periodic).
    multigrid : List[List[int]], optional
        The multigrid levels for the solver, defaults to None.
    contract_prec : str, optional
        The precision for the contraction operations, defaults to '<c16'.
    usedNe : int, optianal
        The used eigenvectors number, defaults to None, usedNe is eigenvector.Ne .
    eigenvector_snk : Eigenvector, optional
        Alternative sink eigenvector for computing perambulators. If provided,
        this will be used instead of the main eigenvector for sink operations.
        Defaults to None, which uses the main eigenvector.
    usedNe_snk : int, optional
        The used eigenvectors number for sink operations. Defaults to None,
        which uses the same value as eigenvector_snk.Ne.
    MRHS : bool, optional:
        Use MRHS methods to solve perambulators, defaults to False.
        This option requires more device memory.

    Notes:
    ------
    - This class requires PyQUDA + QUDA for GPU-accelerated computations.
    """

    def __init__(
        self,
        latt_size: List[int],
        gauge_field: GaugeField,
        mass: float,
        tol: float,
        maxiter: int,
        xi_0: float = 1.0,
        nu: float = 1.0,
        clover_coeff_t: float = 0.0,
        clover_coeff_r: float = 1.0,
        t_boundary: Literal[1, -1] = 1,
        multigrid: List[List[int]] = None,
        contract_prec: str = "<c16",
        eigenvector_src: Eigenvector = None,
        usedNe_src: int = None,
        eigenvector_snk: Eigenvector = None,
        usedNe_snk: int = None,
        point_src: PointSource = None,
        usedNp_src: int = None,
        point_snk: PointSource = None,
        usedNp_snk: int = None,
        MRHS: bool = False,
        use_vectorized: bool = True,  # New parameter to control version
        same_eigenvector: bool = True,
    ) -> None:
        # if not check_QUDA():
        #     raise ImportError("Please install PyQuda to generate the perambulator or check MPI_init again.")
        from pyquda_utils import core

        self.latt_info = core.LatticeInfo(
            latt_size=latt_size, t_boundary=t_boundary, anisotropy=xi_0 / nu
        )
        self.contract_prec = contract_prec
        self.use_vectorized = use_vectorized  # Store the version flag

        backend = get_backend()
        assert (
            backend.__name__ == "cupy"
        ), "PyQuda only support cupy as the ndarray implementation"
        Lx, Ly, Lz, Lt = self.latt_info.size
        if same_eigenvector:
            eigenvector_snk = eigenvector_src
        self.same_eigenvector = same_eigenvector

        if usedNe_src is None:
            if eigenvector_src is None:
                usedNe_src = 0
            else:
                usedNe_src = eigenvector_src.Ne
        elif eigenvector_src is not None and eigenvector_src.Ne != usedNe_src:
            print(
                f"Warning: used Ne = {usedNe_src}, data maximum Ne = {eigenvector_src.Ne}"
            )
        if usedNe_snk is None:
            if eigenvector_snk is None:
                usedNe_snk = 0
            else:
                # same_eigenvector aliases the sink to the source handle, whose
                # .Ne is the on-disk maximum. When the source is truncated, the
                # sink must follow the truncation too, otherwise VSV is
                # allocated with more sink modes than the loaded data provides.
                usedNe_snk = usedNe_src if self.same_eigenvector else eigenvector_snk.Ne
        self.eigenvector_src = eigenvector_src
        self.eigenvector_snk = eigenvector_snk
        self.Ne_src = usedNe_src
        self.Ne_snk = usedNe_snk
        if (
            self.same_eigenvector
            and eigenvector_src is not None
            and self.Ne_snk > self.Ne_src
        ):
            # With same_eigenvector the sink *is* the source basis, so a larger
            # sink truncation can never be satisfied by the loaded data.
            raise ValueError(
                f"usedNe_snk={self.Ne_snk} exceeds usedNe_src={self.Ne_src}, but "
                "same_eigenvector=True shares a single basis; the sink cannot use "
                "more eigenvectors than the source."
            )

        if usedNp_src is None:
            if point_src is None:
                usedNp_src = 0
            else:
                usedNp_src = point_src.Np
        if usedNp_snk is None:
            if point_snk is None:
                usedNp_snk = 0
            else:
                usedNp_snk = point_snk.Np
        self.point_src = point_src
        self.point_snk = point_snk
        self.Np_src = usedNp_src
        self.Np_snk = usedNp_snk

        self.gauge_field = gauge_field
        self.gauge_field_smear = None
        self.gauge_field_new = None
        self.MRHS = MRHS
        self.last_metrics = {}
        self._point_sink_indices = None
        self.dirac = core.getDirac(
            self.latt_info,
            mass,
            tol,
            maxiter,
            xi_0,
            clover_coeff_t,
            clover_coeff_r,
            multigrid,
        )
        if self.Ne_src > 0:
            self._SV = backend.zeros(
                (2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc), self.contract_prec
            )
            if self.Ne_snk > 0:
                self._VSV = backend.zeros(
                    (Lt, Ns, Ns, self.Ne_snk, self.Ne_src), self.contract_prec
                )
            else:
                self._VSV = None
            if self.Np_snk > 0:
                self._PSV = backend.zeros(
                    (Lt, Ns, Ns, self.Np_snk, Nc, self.Ne_src), self.contract_prec
                )
            else:
                self._PSV = None
        else:
            self._SV = None
            self._VSV = None
            self._PSV = None

        if self.Np_src > 0:
            self._SP = None
            if self.Np_snk > 0:
                self._PSP = backend.zeros(
                    (Lt, Ns, Ns, self.Np_snk, Nc, self.Np_src, Nc), self.contract_prec
                )
            else:
                self._PSP = None
        else:
            self._SP = None
            self._PSP = None

    def load(self, key: str):
        import numpy as np
        from pyquda_utils import core
        from pyquda_utils import io

        backend = get_backend()
        Lx, Ly, Lz, Lt = self.latt_info.size
        gx, gy, gz, gt = self.latt_info.grid_coord
        Ne = self.Ne_src
        Ne_snk = self.Ne_snk
        self.gauge_field_smear = io.readQIOGauge(self.gauge_field.load(key).file)
        self.gauge_field_new = True

        eigenvector_data = None
        if self.eigenvector_src is not None:
            eigenvector_data = self.eigenvector_src.load(key)
            eigenvector_src_data_dagger = np.zeros(
                (Ne, Lt, Lz, Ly, Lx, Nc), self.contract_prec
            )
        if self.same_eigenvector:
            eigenvector_snk_data = eigenvector_data
        elif self.eigenvector_snk is not None:
            eigenvector_snk_data = self.eigenvector_snk.load(key)
            eigenvector_snk_data_dagger = np.zeros(
                (Ne_snk, Lt, Lz, Ly, Lx, Nc), self.contract_prec
            )
        else:
            eigenvector_snk_data = None

        # read data into host memory
        # save V^\dag here to save device memory
        set_backend("numpy")
        if self.eigenvector_src is not None:
            for e in range(Ne):
                for t in range(Lt):
                    eigenvector_src_data_dagger[e, t] = eigenvector_data[
                        gt * Lt + t,
                        e,
                        gz * Lz : (gz + 1) * Lz,
                        gy * Ly : (gy + 1) * Ly,
                        gx * Lx : (gx + 1) * Lx,
                    ].conj()
        else:
            eigenvector_src_data_dagger = None

        if self.same_eigenvector:
            # The source buffer is truncated to Ne_src; the sink view must not
            # index past it when usedNe_snk was set larger than usedNe_src.
            # A None source (point-source-only runs) stays None.
            eigenvector_snk_data_dagger = (
                None
                if eigenvector_src_data_dagger is None
                else eigenvector_src_data_dagger[:Ne_snk]
            )
        elif self.eigenvector_snk is not None:
            for e in range(Ne_snk):
                for t in range(Lt):
                    eigenvector_snk_data_dagger[e, t] = eigenvector_snk_data[
                        gt * Lt + t,
                        e,
                        gz * Lz : (gz + 1) * Lz,
                        gy * Ly : (gy + 1) * Ly,
                        gx * Lx : (gx + 1) * Lx,
                    ].conj()
        else:
            eigenvector_snk_data_dagger = None

        set_backend(backend)
        if self.point_src is not None:
            self.point_source_data = self.point_src.load(key)
        if self.point_snk is not None:
            self.point_sink_data = self.point_snk.load(key)
            host_indices = sampled_point_local_indices(
                self.point_sink_data, self.Np_snk, self.latt_info
            )
            self._point_sink_indices = tuple(
                backend.asarray(index) for index in host_indices
            )
        else:
            self._point_sink_indices = None
        # set eigenvector_data_cb2 on device mem
        if self.eigenvector_src is not None:
            self._eigenvector_data_dagger = backend.asarray(
                core.cb2(eigenvector_src_data_dagger, [1, 2, 3, 4])
            )
        else:
            self._eigenvector_data_dagger = None
        if self.same_eigenvector:
            # Same truncation rule as above: never expose more sink rows than
            # the source dagger buffer (built with Ne_src rows) actually holds.
            # A None source (point-source-only runs) stays None.
            self._eigenvector_snk_data_dagger = (
                None
                if self._eigenvector_data_dagger is None
                else self._eigenvector_data_dagger[: self.Ne_snk]
            )
        elif self.eigenvector_snk is not None:
            self._eigenvector_snk_data_dagger = backend.asarray(
                core.cb2(eigenvector_snk_data_dagger, [1, 2, 3, 4])
            )
        else:
            self._eigenvector_snk_data_dagger = None

    def _stout_smear_quda(self, nstep: int, rho: float, dir_ignore: int):
        gauge = self.gauge_field_smear
        if self.gauge_field_smear is None:
            raise ValueError(
                "Gauge not loaded, please use .load() before .stout_smear()."
            )

        gauge.stoutSmear(nstep, rho, dir_ignore)
        self.gauge_field_smear = gauge

    def stout_smear(self, nstep: int, rho: float, dir_ignore: int = 3):
        backend = get_backend()
        if backend.__name__ == "numpy":
            raise NotImplementedError(
                "Ndarray stout smear not implement in PerambulatorGenerator."
            )
        elif backend.__name__ == "cupy":
            # __init__() has check_QUDA() before !
            self._stout_smear_quda(nstep, rho, dir_ignore)

    def calc_old(self, t_src: int):
        """
        Sequential method for perambulator calculation（参考实现/验证用）。

        步骤概览（对每个 eigen ∈ [0, Ne_src)）:
        1) Dirac 反演：以 t_src 处本征向量为源，求解 S_V = inv(D) × eigen
        2) 与汇点本征向量收缩得到 VSV
        3) 逐点抽取得到 PSV（按 checkerboard 索引提取）

        适用场景：正确性验证、小规模问题；生产环境推荐 calc_new。

        Parameters
        ----------
        t_src : int
            Source time slice index (global lattice coordinate)

        Returns
        -------
        VSV : cp.ndarray
            Eigenvector-to-eigenvector propagators (math: S_{i,j})
            Shape: (Lt, Ns, Ns, Ne_snk, Ne_src)
            Dtype: complex128

        PSV : cp.ndarray (optional)
            Point-to-eigenvector propagators (math: S_{xa,i}), if Np_snk > 0
            Shape: (Lt, Ns, Ns, Np_snk, Nc, Ne_src)
            Dtype: complex128

        Notes
        -----
        - This method is not recommended for large-scale production runs
        - Consider using calc_new() for better performance
        - Requires QUDA backend with GPU support
        - Time cost is dominated by GPU-CPU synchronization in point extraction
        """
        import cupy as cp

        backend = get_backend()
        from pyquda_utils.core import LatticeFermion, MultiLatticeFermion, invert

        if self.gauge_field_new:
            self.dirac.loadGauge(self.gauge_field_smear)  # loadGauge after
            self.gauge_field_new = False

        latt_info = self.latt_info
        Lx, Ly, Lz, Lt = latt_info.size
        Vol = Lx * Ly * Lz * Lt
        Ne = self.Ne_src
        Ne_snk = self.Ne_snk
        eigenvector_dagger = self._eigenvector_data_dagger
        if self.eigenvector_snk is not None:
            eigenvector_sink_dagger = self._eigenvector_snk_data_dagger
        else:
            eigenvector_sink_dagger = eigenvector_dagger
        dirac = self.dirac
        gx, gy, gz, gt = self.latt_info.grid_coord

        SV = self._SV
        VSV = self._VSV
        PSV = self._PSV if self.Np_snk > 0 else None

        from time import perf_counter

        for eigen in range(Ne):
            cp.cuda.runtime.deviceSynchronize()
            s = perf_counter()
            if self.MRHS:
                print("Warning: use MRHS.")
                V_MRHS = MultiLatticeFermion(latt_info, Ns)
                for spin in range(Ns):
                    data = V_MRHS[spin].data.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
                    if gt * Lt <= t_src and (gt + 1) * Lt > t_src:
                        data[:, t_src % Lt, :, :, :, spin, :] = backend.asarray(
                            eigenvector_dagger[eigen, :, t_src % Lt, :, :, :, :].conj()
                        )
                SV_MRHS = dirac.invertMultiSrc(V_MRHS)
                for spin in range(Ns):
                    SV.reshape(Vol, Ns, Ns, Nc)[:, :, spin, :] = SV_MRHS[
                        spin
                    ].data.reshape(Vol, Ns, Nc)
            else:
                for spin in range(Ns):
                    V = LatticeFermion(latt_info)  # V.data is double prec.
                    data = V.data.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
                    if gt * Lt <= t_src and (gt + 1) * Lt > t_src:
                        data[:, t_src % Lt, :, :, :, spin, :] = backend.asarray(
                            eigenvector_dagger[eigen, :, t_src % Lt, :, :, :, :].conj()
                        )  # [Ne, etzyx, Nc]
                    SV.reshape(Vol, Ns, Ns, Nc)[:, :, spin, :] = dirac.invert(
                        V
                    ).data.reshape(
                        Vol, Ns, Nc
                    )  # .get()
            cp.cuda.runtime.deviceSynchronize()
            invert_time = perf_counter() - s

            cp.cuda.runtime.deviceSynchronize()
            s = perf_counter()
            SV_array = backend.asarray(SV)  # Extract array conversion outside loops
            VSV[:, :, :, : self.Ne_snk, eigen] = contract(
                "ketzyxa,etzyxija->tijk",
                backend.asarray(eigenvector_sink_dagger),
                SV_array,
                optimize=True,
            )
            # Use broadcasting to eliminate t_snk loop
            if self.Np_snk > 0:
                for t_snk in range(Lt):
                    if not (gt * Lt <= t_snk < (gt + 1) * Lt):
                        continue
                for point_snk_idx in range(self.Np_snk):
                    t_index = t_snk
                    x_index = self.point_sink_data[point_snk_idx, t_snk, 0]
                    y_index = self.point_sink_data[point_snk_idx, t_snk, 1]
                    z_index = self.point_sink_data[point_snk_idx, t_snk, 2]

                    # Check if this point belongs to current GPU based on grid_coord
                    # Each GPU handles a sub-region of the lattice
                    if (
                        gx * (Lx // 2) <= x_index < (gx + 1) * (Lx // 2)
                        and gy * Ly <= y_index < (gy + 1) * Ly
                        and gz * Lz <= z_index < (gz + 1) * Lz
                    ):
                        # This point belongs to current GPU, process it
                        point_coords = (
                            (t_index + x_index + y_index + z_index) % 2,
                            t_index,
                            z_index,
                            y_index,
                            x_index // 2,
                        )
                        PSV[t_snk % Lt, :, :, point_snk_idx, :, eigen] = SV_array[
                            point_coords
                        ]
            cp.cuda.runtime.deviceSynchronize()
            contraction_time = perf_counter() - s

            # print for check device mem
            free, total = cp.cuda.runtime.memGetInfo()
            print(
                f"Ne = {eigen}:  inv t = {invert_time:.4f} sec, contraction t = {contraction_time:.4f} sec, device mem: {(total - free) / 1024**3} GB, free:{free / 1024**3} GB."
            )

        return VSV, PSV

    def calc_new(self, t_src: int, products=("VSV", "PSV")):
        """
        Vectorized method for perambulator calculation（生产环境推荐）。

        步骤概览（对每个 eigen ∈ [0, Ne_src)）:
        1) Dirac 反演：以 t_src 处本征向量为源，求解 S_V
        2) 与汇点本征向量收缩得到 VSV
        3) 向量化掩码与批量抽取得到 PSV（单次批量 GPU 读）

        说明：该方法以向量化和批量内存访问替代逐点循环，显著降低同步与解释器开销；默认用于大规模/性能敏感场景。

        Parameters
        ----------
        t_src : int
            Source time slice index (global lattice coordinate)

        Returns
        -------
        VSV : cp.ndarray
            Eigenvector-to-eigenvector propagators (math: S_{i,j})
            Shape: (Lt, Ns, Ns, Ne_snk, Ne_src)
            Dtype: complex128

        PSV : cp.ndarray (optional)
            Point-to-eigenvector propagators (math: S_{xa,i}), if Np_snk > 0
            Shape: (Lt, Ns, Ns, Np_snk, Nc, Ne_src)
            Dtype: complex128

        Point-source PSP propagation is calculated separately by
        :meth:`calc_point_sources`; VSP is reconstructed from PSV at contraction time.

        Notes
        -----
        - This is the recommended method for all production calculations
        - Provides 3-5× speedup over calc_old for typical problems
        - Requires QUDA backend with GPU support
        - Performance bottleneck is now Dirac inversion, not extraction
        - Further optimization possible with better MPI coordination

        Examples
        --------
        >>> # Production usage (recommended)
        >>> VSV, PSV = perambulator.calc_new(t_src=0)
        >>> print(f"VSV shape: {VSV.shape}")   # (Lt, Ns, Ns, Ne_snk, Ne_src)
        >>> print(f"PSV shape: {PSV.shape}")   # (Lt, Ns, Ns, Np_snk, Nc, Ne_src)

        >>> # Validation: compare with calc_old
        >>> VSV_old, PSV_old = perambulator.calc_old(t_src=0)
        >>> diff_VSV = cp.linalg.norm(VSV - VSV_old)
        >>> assert diff_VSV < 1e-10, "Numerical agreement failed"
        """
        products = frozenset(str(product).upper() for product in products)
        unsupported = products - {"VSV", "PSV"}
        if unsupported or not products:
            raise ValueError(f"Unsupported eigen products: {sorted(unsupported)}")
        if "VSV" in products and self._VSV is None:
            raise ValueError("VSV requested without an eigenvector sink")
        if "PSV" in products and self._PSV is None:
            raise ValueError("PSV requested without a point sink")

        import cupy as cp

        backend = get_backend()
        from pyquda_utils.core import LatticeFermion, MultiLatticeFermion, invert

        if self.gauge_field_new:
            self.dirac.loadGauge(self.gauge_field_smear)  # loadGauge after
            self.gauge_field_new = False

        latt_info = self.latt_info
        Lx, Ly, Lz, Lt = latt_info.size
        Vol = Lx * Ly * Lz * Lt
        Ne_src = self.Ne_src
        Ne_snk = self.Ne_snk
        eigenvector_dagger = self._eigenvector_data_dagger
        if self.eigenvector_snk is not None:
            eigenvector_sink_dagger = self._eigenvector_snk_data_dagger
        else:
            eigenvector_sink_dagger = eigenvector_dagger
        dirac = self.dirac
        gx, gy, gz, gt = self.latt_info.grid_coord

        SV = self._SV
        VSV = self._VSV
        PSV = self._PSV
        from time import perf_counter

        total_started = perf_counter()
        inversion_seconds = 0.0
        vsv_seconds = 0.0
        psv_seconds = 0.0
        peak_device_used_bytes = 0
        for eigen in range(Ne_src):
            cp.cuda.runtime.deviceSynchronize()
            s = perf_counter()
            if self.MRHS:
                print("Warning: use MRHS.")
                V_MRHS = MultiLatticeFermion(latt_info, Ns)
                for spin in range(Ns):
                    data = V_MRHS[spin].data.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
                    if gt * Lt <= t_src and (gt + 1) * Lt > t_src:
                        data[:, t_src % Lt, :, :, :, spin, :] = backend.asarray(
                            eigenvector_dagger[eigen, :, t_src % Lt, :, :, :, :].conj()
                        )
                SV_MRHS = dirac.invertMultiSrc(V_MRHS)
                for spin in range(Ns):
                    SV.reshape(Vol, Ns, Ns, Nc)[:, :, spin, :] = SV_MRHS[
                        spin
                    ].data.reshape(Vol, Ns, Nc)
            else:
                for spin in range(Ns):
                    V = LatticeFermion(latt_info)  # V.data is double prec.
                    data = V.data.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
                    if gt * Lt <= t_src and (gt + 1) * Lt > t_src:
                        data[:, t_src % Lt, :, :, :, spin, :] = backend.asarray(
                            eigenvector_dagger[eigen, :, t_src % Lt, :, :, :, :].conj()
                        )  # [Ne, etzyx, Nc]
                    SV.reshape(Vol, Ns, Ns, Nc)[:, :, spin, :] = dirac.invert(
                        V
                    ).data.reshape(
                        Vol, Ns, Nc
                    )  # .get()
            cp.cuda.runtime.deviceSynchronize()
            invert_time = perf_counter() - s
            inversion_seconds += invert_time

            cp.cuda.runtime.deviceSynchronize()
            s = perf_counter()
            SV_array = backend.asarray(SV)  # Extract array conversion outside loops
            contraction_time_VSV = 0.0
            if "VSV" in products:
                VSV[:, :, :, :, eigen] = contract(
                    "ketzyxa,etzyxija->tijk",
                    backend.asarray(eigenvector_sink_dagger),
                    SV_array,
                    optimize=True,
                )
                cp.cuda.runtime.deviceSynchronize()
                contraction_time_VSV = perf_counter() - s
                vsv_seconds += contraction_time_VSV
            psv_started = perf_counter()
            contraction_time_PSV = 0.0
            if "PSV" in products:
                point_indices, local_times, parity, z_local, y_local, x_half = (
                    self._point_sink_indices
                )
                PSV[
                    local_times,
                    :,
                    :,
                    point_indices,
                    :,
                    eigen,
                ] = SV_array[
                    parity,
                    local_times,
                    z_local,
                    y_local,
                    x_half,
                ]
                cp.cuda.runtime.deviceSynchronize()
                contraction_time_PSV = perf_counter() - psv_started
                psv_seconds += contraction_time_PSV

            # print for check device mem
            free, total = cp.cuda.runtime.memGetInfo()
            peak_device_used_bytes = max(peak_device_used_bytes, int(total - free))
            print(
                f"Ne = {eigen}:  inv t = {invert_time:.4f} sec, contraction t for VSV = {contraction_time_VSV:.4f} sec, contraction t for PSV = {contraction_time_PSV:.4f} sec, device mem: {(total - free) / 1024**3} GB, free:{free / 1024**3} GB."
            )
        free, total = cp.cuda.runtime.memGetInfo()
        self.last_metrics = {
            "product": "_".join(sorted(products)),
            "source_time": int(t_src),
            "eigenvectors": int(Ne_src),
            "inversion_seconds": inversion_seconds,
            "vsv_seconds": vsv_seconds,
            "psv_seconds": psv_seconds,
            "elapsed_seconds": perf_counter() - total_started,
            "device_used_bytes": int(total - free),
            "peak_device_used_bytes": peak_device_used_bytes,
        }
        return (VSV if "VSV" in products else None), (
            PSV if "PSV" in products else None
        )

    def calc_eigen_sources(self, t_src: int, products=("VSV", "PSV")):
        """Calculate selected VSV/PSV products without point-source inversions."""
        return self.calc_new(t_src, products=products)

    def calc_point_sources(self, t_src: int):
        """Calculate PSP at sampled sinks for every global sink time."""
        import cupy as cp
        from pyquda.field import MultiLatticeFermion
        from pyquda_utils import source
        from time import perf_counter

        if self.point_src is None or self.point_snk is None:
            raise ValueError("point_src and point_snk are required for PSP")
        if self._PSP is None:
            raise ValueError("PSP buffer was not allocated")
        if self.gauge_field_new:
            self.dirac.loadGauge(self.gauge_field_smear)
            self.gauge_field_new = False

        backend = get_backend()
        point_indices, local_times, parity, z_local, y_local, x_half = (
            self._point_sink_indices
        )
        PSP = self._PSP
        PSP.fill(0)
        started = perf_counter()

        inversion_seconds = 0.0
        extraction_seconds = 0.0
        peak_device_used_bytes = 0
        for point_src_idx in range(self.Np_src):
            source_coordinates = self.point_source_data[point_src_idx, t_src]
            if hasattr(source_coordinates, "get"):
                source_coordinates = source_coordinates.get()
            source_position = [int(value) for value in source_coordinates] + [
                int(t_src)
            ]
            rhs_count = 12 if self.MRHS else 1
            for rhs_start in range(0, Ns * Nc, rhs_count):
                batch_count = min(rhs_count, Ns * Nc - rhs_start)
                rhs = MultiLatticeFermion(self.latt_info, batch_count)
                for batch_idx in range(batch_count):
                    spin_color = rhs_start + batch_idx
                    rhs[batch_idx] = source.source(
                        self.latt_info,
                        "point",
                        source_position,
                        spin_color // Nc,
                        spin_color % Nc,
                    )
                inversion_started = perf_counter()
                solutions = self.dirac.invertMultiSrcRestart(rhs, 0)
                inversion_seconds += perf_counter() - inversion_started
                extraction_started = perf_counter()
                for batch_idx in range(batch_count):
                    spin_color = rhs_start + batch_idx
                    values = backend.asarray(solutions[batch_idx].data)[
                        parity,
                        local_times,
                        z_local,
                        y_local,
                        x_half,
                    ]
                    PSP[
                        local_times,
                        :,
                        spin_color // Nc,
                        point_indices,
                        :,
                        point_src_idx,
                        spin_color % Nc,
                    ] = values
                extraction_seconds += perf_counter() - extraction_started
            free, total = cp.cuda.runtime.memGetInfo()
            peak_device_used_bytes = max(peak_device_used_bytes, int(total - free))

        cp.cuda.runtime.deviceSynchronize()
        free, total = cp.cuda.runtime.memGetInfo()
        self.last_metrics = {
            "product": "PSP",
            "source_time": int(t_src),
            "source_points": int(self.Np_src),
            "inversion_seconds": inversion_seconds,
            "extraction_seconds": extraction_seconds,
            "elapsed_seconds": perf_counter() - started,
            "device_used_bytes": int(total - free),
            "peak_device_used_bytes": peak_device_used_bytes,
        }
        print(
            f"PSP t_src={t_src}: sources={self.Np_src}, "
            f"elapsed={self.last_metrics['elapsed_seconds']:.4f}s, "
            f"device={(total - free) / 1024**3:.3f}GB"
        )
        return PSP

    def calc(self, t_src: int):
        """Main calc method that chooses between old and new versions"""
        if self.use_vectorized:
            return self.calc_new(t_src)
        else:
            return self.calc_old(t_src)
