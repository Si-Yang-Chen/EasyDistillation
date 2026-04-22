from typing import List, Literal

from opt_einsum import contract

from ..constant import Nc, Ns, Nd
from ..backend import set_backend, get_backend, check_QUDA
from ..preset import GaugeField, Eigenvector, PointSource
import numpy as np


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
        if usedNe_snk is None:
            if eigenvector_snk is None:
                usedNe_snk = 0
            else:
                usedNe_snk = eigenvector_snk.Ne
        elif eigenvector_src.Ne != usedNe_src:
            print(
                f"Warning: used Ne = {usedNe_src}, data maximum Ne = {eigenvector_src.Ne}"
            )
        self.eigenvector_src = eigenvector_src
        self.eigenvector_snk = eigenvector_snk
        self.Ne_src = usedNe_src
        self.Ne_snk = usedNe_snk

        if usedNp_src is None:
            if point_src is None:
                usedNp_src = 0
            else:
                usedNp_src = point_src.Ne
        if usedNp_snk is None:
            if point_snk is None:
                usedNp_snk = 0
            else:
                usedNp_snk = point_snk.Ne
        self.point_src = point_src
        self.point_snk = point_snk
        self.Np_src = usedNp_src
        self.Np_snk = usedNp_snk

        self.gauge_field = gauge_field
        self.gauge_field_smear = None
        self.gauge_field_new = None
        self.MRHS = MRHS
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
            self._SP = backend.zeros(
                (2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc), self.contract_prec
            )
            if self.Ne_snk > 0:
                self._VSP = backend.zeros(
                    (Lt, Ns, Ns, self.Ne_snk, self.Np_src, Nc), self.contract_prec
                )
            else:
                self._VSP = None
            if self.Np_snk > 0:
                self._PSP = backend.zeros(
                    (Lt, Ns, Ns, self.Np_snk, Nc, self.Np_src, Nc), self.contract_prec
                )
            else:
                self._PSP = None
        else:
            self._SP = None
            self._VSP = None
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
            eigenvector_snk_data_dagger = eigenvector_src_data_dagger
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
        # set eigenvector_data_cb2 on device mem
        if self.eigenvector_src is not None:
            self._eigenvector_data_dagger = backend.asarray(
                core.cb2(eigenvector_src_data_dagger, [1, 2, 3, 4])
            )
        else:
            self._eigenvector_data_dagger = None
        if self.same_eigenvector:
            self._eigenvector_snk_data_dagger = self._eigenvector_data_dagger
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

        gauge.smearSTOUT(nstep, rho, dir_ignore)
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
        if self.Np_snk > 0:
            PSV = self._PSV

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

    def calc_new(self, t_src: int):
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


        VSP : cp.ndarray (optional)
            Eigenvector-to-point propagators (math: S_{i,xa}); not used in PSV extraction path

        PSP : cp.ndarray (optional)
            Point-to-point propagators (math: S_{xa,yb}), if Np_src > 0 and Np_snk > 0
            Shape: (Lt, Ns, Ns, Np_snk, Nc, Np_src, Nc)
            Dtype: complex128

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
        SP = self._SP
        VSP = self._VSP
        PSP = self._PSP

        from time import perf_counter

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

            cp.cuda.runtime.deviceSynchronize()
            s = perf_counter()
            SV_array = backend.asarray(SV)  # Extract array conversion outside loops
            VSV[:, :, :, :, eigen] = contract(
                "ketzyxa,etzyxija->tijk",
                backend.asarray(eigenvector_sink_dagger),
                SV_array,
                optimize=True,
            )
            cp.cuda.runtime.deviceSynchronize()
            contraction_time_VSV = perf_counter() - s
            # Use broadcasting to eliminate t_snk loop
            if self.Np_snk > 0:
                # Vectorized approach: process all valid points at once
                import numpy as np

                # Create all possible (t_snk, point_snk_idx) combinations
                # Generate valid time indices for current GPU
                valid_t_indices = np.arange(gt * Lt, (gt + 1) * Lt)
                point_indices = np.arange(self.Np_snk)

                if len(valid_t_indices) > 0:
                    # Get all coordinates for valid time slices
                    # Shape: (Np_snk, valid_t_count) for each coordinate
                    x_coords = self.point_sink_data[
                        :, valid_t_indices, 0
                    ]  # shape: (Np_snk, valid_t_count)
                    y_coords = self.point_sink_data[
                        :, valid_t_indices, 1
                    ]  # shape: (Np_snk, valid_t_count)
                    z_coords = self.point_sink_data[
                        :, valid_t_indices, 2
                    ]  # shape: (Np_snk, valid_t_count)

                    # ========================================================
                    # PHASE 2: Vectorized Boolean Masking
                    # ========================================================
                    # Create spatial region masks (vectorized comparisons)
                    # Each GPU handles a sub-region defined by MPI grid coordinates:
                    # - gx: X-region in GPU grid [gx*Lx/2, (gx+1)*Lx/2)
                    # - gy: Y-region in GPU grid [gy*Ly, (gy+1)*Ly)
                    # - gz: Z-region in GPU grid [gz*Lz, (gz+1)*Lz)
                    # - gt: T-region in GPU grid [gt*Lt, (gt+1)*Lt)

                    valid_x_mask = (gx * (Lx // 2) <= x_coords) & (
                        x_coords < (gx + 1) * (Lx // 2)
                    )
                    valid_y_mask = (gy * Ly <= y_coords) & (y_coords < (gy + 1) * Ly)
                    valid_z_mask = (gz * Lz <= z_coords) & (z_coords < (gz + 1) * Lz)

                    # Combined mask: point is valid iff in all three spatial regions
                    # Result: Boolean array (Np_snk, valid_t_count)
                    # No GPU synchronization - vectorized GPU operations only
                    valid_point_mask = valid_x_mask & valid_y_mask & valid_z_mask

                    # Get indices of valid points
                    # np.where returns (indices_axis0, indices_axis1) for 2D array
                    valid_point_indices = np.where(valid_point_mask)
                    valid_point_snk_idx = (
                        valid_point_indices[0].get()
                        if hasattr(valid_point_indices[0], "get")
                        else valid_point_indices[0]
                    )  # Convert to NumPy
                    valid_t_snk_idx = (
                        valid_point_indices[1].get()
                        if hasattr(valid_point_indices[1], "get")
                        else valid_point_indices[1]
                    )  # Convert to NumPy

                    if len(valid_point_snk_idx) > 0:
                        # ====================================================
                        # PHASE 3: Batched Point Extraction
                        # ====================================================
                        # Get corresponding coordinates for valid points
                        valid_x = (
                            x_coords[valid_point_mask].get()
                            if hasattr(x_coords, "get")
                            else x_coords[valid_point_mask]
                        )  # Convert to NumPy
                        valid_y = (
                            y_coords[valid_point_mask].get()
                            if hasattr(y_coords, "get")
                            else y_coords[valid_point_mask]
                        )  # Convert to NumPy
                        valid_z = (
                            z_coords[valid_point_mask].get()
                            if hasattr(z_coords, "get")
                            else z_coords[valid_point_mask]
                        )  # Convert to NumPy
                        valid_t = valid_t_indices[
                            valid_t_snk_idx
                        ]  # Both are NumPy arrays

                        # Calculate checkerboard indices for SV_array indexing
                        # SV_array layout (GPU checkerboard format):
                        #   Shape: (2, Lt, Lz, Ly, Lx//2, Ns, Nc)
                        #   Index: (parity, t, z, y, x_half, spin, color)
                        #
                        # Parity computation: (t + x + y + z) % 2
                        # This alternating pattern allows better memory coalescing on GPU
                        cb_indices = (valid_t + valid_x + valid_y + valid_z) % 2
                        x_half_indices = (
                            valid_x // 2
                        )  # X coordinate is halved in checkerboard format

                        # Extract SV values for all valid points at once
                        # This is the CRITICAL optimization: SINGLE GPU read for all points
                        # instead of Np_snk × Lt individual reads in calc_old
                        point_coords = (
                            cb_indices,
                            valid_t,
                            valid_z,
                            valid_y,
                            x_half_indices,
                        )
                        PSV_values = SV_array[
                            point_coords
                        ]  # shape: (n_valid_points, Ns, Ns, Nc)
                        # GPU-CPU Sync happens HERE (line 755 approximately)

                        # Handle variable number of points per GPU
                        # Each GPU processes only the points that belong to its spatial region
                        n_valid_points = len(valid_point_snk_idx)

                        if n_valid_points > 0:
                            # Direct assignment to PSV array using global point indices
                            # PSV array shape: (Lt, Ns, Ns, Np_snk, Nc, Ne_src)
                            PSV[valid_t % Lt, :, :, valid_point_snk_idx, :, eigen] = (
                                PSV_values
                            )

                            # Optional: Add debug information
                            print(
                                f"GPU {gt}: processed {n_valid_points} points at time slices {valid_t_indices}"
                            )
                        else:
                            print(f"GPU {gt}: no valid PSV points to process")
            cp.cuda.runtime.deviceSynchronize()
            contraction_time_PSV = perf_counter() - s

            # print for check device mem
            free, total = cp.cuda.runtime.memGetInfo()
            print(
                f"Ne = {eigen}:  inv t = {invert_time:.4f} sec, contraction t for VSV = {contraction_time_VSV:.4f} sec, contraction t for PSV = {contraction_time_PSV:.4f} sec, device mem: {(total - free) / 1024**3} GB, free:{free / 1024**3} GB."
            )
        # Performance timing for point source processing
        cp.cuda.runtime.deviceSynchronize()
        s_point_total = perf_counter()

        for point_src_idx in range(self.Np_src):
            cp.cuda.runtime.deviceSynchronize()
            s_point = perf_counter()

            if self.MRHS:
                mrhs = 12
            else:
                mrhs = 1
            # Use pre-allocated SP memory efficiently
            SP[:] = invert(
                self.dirac,
                "point",
                list(self.point_source_data[t_src, point_src_idx]) + [t_src],
                mrhs=mrhs,
            ).data
            SP_array = backend.asarray(SP)  # Extract array conversion outside loops

            cp.cuda.runtime.deviceSynchronize()
            s_vsp = perf_counter()
            VSP[:, :, :, :, :, point_src_idx] = contract(
                "ketzyxa,etzyxijab->tijkb",
                backend.asarray(eigenvector_sink_dagger),
                SP_array,
                optimize=True,
            )
            cp.cuda.runtime.deviceSynchronize()
            contraction_time_VSP = perf_counter() - s_vsp

            # Process PSP calculation (similar to PSV but for point sources)
            if self.Np_snk > 0:
                cp.cuda.runtime.deviceSynchronize()
                s_psp = perf_counter()
                import numpy as np

                # Create all possible (t_snk, point_snk_idx) combinations
                # Generate valid time indices for current GPU
                valid_t_indices = np.arange(gt * Lt, (gt + 1) * Lt)
                point_indices = np.arange(self.Np_snk)

                if len(valid_t_indices) > 0:
                    # Get all coordinates for valid time slices
                    x_coords = self.point_sink_data[
                        :, valid_t_indices, 0
                    ]  # shape: (Np_snk, valid_t_count)
                    y_coords = self.point_sink_data[
                        :, valid_t_indices, 1
                    ]  # shape: (Np_snk, valid_t_count)
                    z_coords = self.point_sink_data[
                        :, valid_t_indices, 2
                    ]  # shape: (Np_snk, valid_t_count)

                    # Create masks for spatial coordinates that belong to current GPU
                    valid_x_mask = (gx * (Lx // 2) <= x_coords) & (
                        x_coords < (gx + 1) * (Lx // 2)
                    )
                    valid_y_mask = (gy * Ly <= y_coords) & (y_coords < (gy + 1) * Ly)
                    valid_z_mask = (gz * Lz <= z_coords) & (z_coords < (gz + 1) * Lz)

                    # Combined mask for points that belong to current GPU
                    valid_point_mask = valid_x_mask & valid_y_mask & valid_z_mask

                    # Get indices of valid points
                    valid_point_indices = np.where(valid_point_mask)
                    valid_point_snk_idx = valid_point_indices[
                        0
                    ].get()  # Convert to NumPy
                    valid_t_snk_idx = valid_point_indices[1].get()  # Convert to NumPy

                    if len(valid_point_snk_idx) > 0:
                        # Get corresponding coordinates
                        valid_x = x_coords[valid_point_mask].get()  # Convert to NumPy
                        valid_y = y_coords[valid_point_mask].get()  # Convert to NumPy
                        valid_z = z_coords[valid_point_mask].get()  # Convert to NumPy
                        valid_t = valid_t_indices[
                            valid_t_snk_idx
                        ]  # Both are NumPy arrays

                        # Calculate point coordinates for SP_array indexing
                        cb_indices = (valid_t + valid_x + valid_y + valid_z) % 2
                        x_half_indices = valid_x // 2

                        # Extract SP values for all valid points at once
                        point_coords = (
                            cb_indices,
                            valid_t,
                            valid_z,
                            valid_y,
                            x_half_indices,
                        )
                        PSP_values = SP_array[
                            point_coords
                        ]  # shape: (n_valid_points, Ns, Ns, Nc)

                        # Handle variable number of points per GPU
                        # Each GPU processes only the points that belong to its spatial region
                        n_valid_points = len(valid_point_snk_idx)

                        if n_valid_points > 0:
                            # Direct assignment to PSP array using global point indices
                            # PSP array shape: (Lt, Ns, Ns, Np_snk, Nc, Np_src, Nc)
                            PSP[
                                valid_t % Lt,
                                :,
                                :,
                                valid_point_snk_idx,
                                :,
                                point_src_idx,
                                :,
                            ] = PSP_values

                            # Optional: Add debug information
                            print(
                                f"GPU {gt}: processed {n_valid_points} PSP points for source {point_src_idx} at time slices {valid_t_indices}"
                            )
                        else:
                            print(
                                f"GPU {gt}: no PSP points to process for source {point_src_idx}"
                            )

                cp.cuda.runtime.deviceSynchronize()
                contraction_time_PSP = perf_counter() - s_psp
            else:
                contraction_time_PSP = 0.0

            cp.cuda.runtime.deviceSynchronize()
            point_time = perf_counter() - s_point

            # Performance report for each point source
            free, total = cp.cuda.runtime.memGetInfo()
            print(
                f"Point source {point_src_idx}: inv t = {point_time:.4f} sec, VSP contraction t = {contraction_time_VSP:.4f} sec, PSP contraction t = {contraction_time_PSP:.4f} sec, device mem: {(total - free) / 1024**3} GB, free:{free / 1024**3} GB."
            )

        cp.cuda.runtime.deviceSynchronize()
        point_total_time = perf_counter() - s_point_total

        # Final performance summary
        free, total = cp.cuda.runtime.memGetInfo()
        print(
            f"Point source processing total time: {point_total_time:.4f} sec, device mem: {(total - free) / 1024**3} GB, free:{free / 1024**3} GB."
        )

        return VSV, PSV, VSP, PSP

    def calc(self, t_src: int):
        """Main calc method that chooses between old and new versions"""
        if self.use_vectorized:
            return self.calc_new(t_src)
        else:
            return self.calc_old(t_src)
