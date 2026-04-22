from copy import copy
from math import factorial
from typing import List, Tuple, Literal
from time import perf_counter

from opt_einsum import contract

from ..constant import Nc, Nd
from ..backend import get_backend
from ..preset import GaugeField, Eigenvector, PointSource
from ..insertion.phase import MomentumPhase


def comb(n, i):
    return factorial(n) // (factorial(i) * factorial(n - i))


class ElementalGenerator:
    _recombine_benchmark_calls = 0
    _recombine_selected_method = None
    _recombine_elapsed_dense = []
    _recombine_elapsed_sparse = []
    _recombine_validated = False
    _recombine_validate_rtol = 1e-6
    _recombine_validate_atol = 1e-10

    def __init__(
        self,
        latt_size: List[int],
        gauge_field: GaugeField,
        eigenvector: Eigenvector,
        num_nabla: int = 0,
        momentum_list: List[Tuple[int]] = [(0, 0, 0)],
        dilution: Tuple = None,
        is_blending: bool = False,
        usedNe: int = None,
        usedNp: int = None,
        calc_mode: Literal["calc_deriv", "calc_disp"] = "calc_deriv",
        debug: bool = False,
    ) -> None:
        from ..insertion.derivative import derivative
        from ..insertion.gauge_link import GaugeLink

        backend = get_backend()
        Lx, Ly, Lz, Lt = latt_size
        self.debug = debug
        self.kernel = None
        if backend.__name__ == "cupy":
            import os

            with open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "stout_smear.cu"
                )
            ) as f:
                code = f.read()
            self.kernel = backend.RawModule(
                code=code,
                options=("--std=c++11",),
                name_expressions=("stout_smear<double>",),
            ).get_function(
                "stout_smear<double>"
            )  # TODO: More template instance.

        self.latt_size = latt_size
        self.gauge_field = gauge_field
        self.eigenvector = eigenvector
        self.num_nabla = num_nabla
        self.num_derivative = (3 ** (num_nabla + 1) - 1) // 2
        self.num_disp = list(GaugeLink.nmax_generator(num_nabla))[-1]
        self.derivative_list = [derivative(n) for n in range(self.num_derivative)]
        self.num_momentum = len(momentum_list)
        self.momentum_list = momentum_list
        Ne = eigenvector.Ne
        self.Ne = eigenvector.Ne
        self.usedNe = usedNe if usedNe is not None else Ne
        self.usedNp = (
            usedNp  # Not used in ElementalGenerator, but kept for compatibility
        )
        self.calc_mode = calc_mode
        self._U = None
        if calc_mode == "calc_deriv":
            self._V = backend.zeros((self.usedNe, Lz, Ly, Lx, Nc), "<c8")
            self._VPV = backend.zeros(
                (self.num_derivative, self.num_momentum, self.usedNe, self.usedNe),
                "<c16",
            )
        else:
            self._V = None
            self._VPV = None
        if calc_mode == "calc_disp":
            self._disp_buffer = backend.zeros(
                (self.num_disp, self.num_momentum, self.usedNe, self.usedNe), "<c16"
            )
            self._calc_disp_V_buffer = backend.zeros(
                (self.usedNe, Lz, Ly, Lx, Nc), "<c8"
            )
            self._calc_disp_shift_buffer = backend.zeros(
                (self.usedNe, Lz, Ly, Lx, Nc), "<c8"
            )
            self._calc_disp_gauge_buffer = backend.zeros((Lz, Ly, Lx, Nc, Nc), "<c16")
        else:
            self._disp_buffer = None
            self._calc_disp_V_buffer = None
            self._calc_disp_shift_buffer = None
            self._calc_disp_gauge_buffer = None
        self._gauge_field_data = None
        self._eigenvector_data = None
        self._momentum_phase = MomentumPhase(latt_size)

        if is_blending and (dilution is not None):
            print("Using blending stochastic methods.")
            totNe_list = dilution[0]
            if isinstance(dilution[1], int):
                usedNe_list = [dilution[1]] * len(dilution[0])
            else:
                usedNe_list = dilution[1]
            assert len(usedNe_list) == len(totNe_list)
            assert (backend.array(usedNe_list) <= backend.array(totNe_list)).all()
            assert sum(usedNe_list) == Ne
            self.stocastic_coeff = backend.zeros((Ne, Ne))
            start1 = 0
            for nv1 in range(len(usedNe_list)):
                usedNe1 = usedNe_list[nv1]
                totNe1 = totNe_list[nv1]
                start2 = 0
                for nv2 in range(len(usedNe_list)):
                    usedNe2 = usedNe_list[nv2]
                    totNe2 = totNe_list[nv2]
                    if nv1 != nv2:
                        coeff = totNe1 * totNe2 / usedNe1 / usedNe2
                        self.stocastic_coeff[
                            start1 : start1 + usedNe1, start2 : start2 + usedNe2
                        ] = backend.full((usedNe1, usedNe2), coeff)
                    else:
                        coeff1 = totNe1 / usedNe1
                        coeff2 = coeff1 * (totNe1 - 1) / (usedNe1 - 1)
                        matrix = self.stocastic_coeff[
                            start1 : start1 + usedNe1, start2 : start2 + usedNe2
                        ] = backend.full((usedNe1, usedNe2), coeff2)
                        backend.fill_diagonal(matrix, coeff1)
                        self.stocastic_coeff[
                            start1 : start1 + usedNe1, start2 : start2 + usedNe2
                        ] = matrix
                    start2 += usedNe2
                start1 += usedNe1
            print(self.stocastic_coeff)
        elif is_blending:
            # TODO
            raise ValueError("Dilution tuple is not defined.")
        else:
            self.stocastic_coeff = None

    def load(self, key: str):
        self._U = self.gauge_field.load(key)[:].transpose(4, 0, 1, 2, 3, 5, 6)[: Nd - 1]
        self._gauge_field_path = self.gauge_field.load(key).file
        self._eigenvector_data = self.eigenvector.load(key)

    def project_SU3(self):
        backend = get_backend()
        U = self._U
        Uinv = backend.linalg.inv(U)
        while (
            backend.max(backend.abs(U - contract("...ab->...ba", Uinv.conj()))) > 1e-15
            or backend.max(
                backend.abs(contract("...ab,...cb", U, U.conj()) - backend.identity(Nc))
            )
            > 1e-15
        ):
            U = 0.5 * (U + contract("...ab->...ba", Uinv.conj()))
            Uinv = backend.linalg.inv(U)
        self._U = U

    def _stout_smear_ndarray_naive(self, nstep, rho):
        backend = get_backend()
        U = backend.ascontiguousarray(self._U)

        for _ in range(nstep):
            Q = backend.zeros_like(U)
            for mu in range(Nd - 1):
                for nu in range(Nd - 1):
                    if mu != nu:
                        Q[mu] += contract(
                            "...ab,...bc,...dc->...ad",
                            U[nu],
                            backend.roll(U[mu], -1, 3 - nu),
                            backend.roll(U[nu], -1, 3 - mu).conj(),
                        )
                        Q[mu] += contract(
                            "...ba,...bc,...cd->...ad",
                            backend.roll(U[nu], +1, 3 - nu).conj(),
                            backend.roll(U[mu], +1, 3 - nu),
                            backend.roll(backend.roll(U[nu], +1, 3 - nu), -1, 3 - mu),
                        )
            Q = contract("...ab,...cb->...ac", rho * Q, U.conj())
            Q = 0.5j * (contract("...ab->...ba", Q.conj()) - Q)
            Q -= 1 / Nc * contract("...aa,bc->...bc", Q, backend.identity(Nc))
            c0 = contract("...ab,...bc,...ca->...", Q, Q, Q).real / 3
            c1 = contract("...ab,...ba->...", Q, Q).real / 2

            c0_max = 2 * (c1 / 3) ** (3 / 2)
            parity = c0 < 0
            c0 = backend.abs(c0)
            theta = backend.arccos(c0 / c0_max)
            u = (c1 / 3) ** 0.5 * backend.cos(theta / 3)
            w = c1**0.5 * backend.sin(theta / 3)
            u_sq = u**2
            w_sq = w**2
            e_iu = backend.exp(-1j * u)
            e_2iu = backend.exp(2j * u)
            cos_w = backend.cos(w)
            sinc_w = 1 - w_sq / 6 * (1 - w_sq / 20 * (1 - w_sq / 42 * (1 - w_sq / 72)))
            large = backend.abs(w) > 0.05
            w_large = w[large]
            sinc_w[large] = backend.sin(w_large) / w_large
            f_denom = 1 / (9 * u_sq - w_sq)
            f0 = (
                (u_sq - w_sq) * e_2iu
                + e_iu * (8 * u_sq * cos_w + 2j * u * (3 * u_sq + w_sq) * sinc_w)
            ) * f_denom
            f1 = (
                2 * u * e_2iu - e_iu * (2 * u * cos_w - 1j * (3 * u_sq - w_sq) * sinc_w)
            ) * f_denom
            f2 = (e_2iu - e_iu * (cos_w + 3j * u * sinc_w)) * f_denom
            f0[parity] = f0[parity].conj()
            f1[parity] = -f1[parity].conj()
            f2[parity] = f2[parity].conj()

            f0 = contract("...,ab->...ab", f0, backend.identity(Nc))
            f1 = contract("...,...ab->...ab", f1, Q)
            f2 = contract("...,...ab,...bc->...ac", f2, Q, Q)
            U = contract("...ab,...bc->...ac", f0 + f1 + f2, U)
        self._U = U

    def _stout_smear_ndarray(self, nstep, rho):
        backend = get_backend()
        U = backend.ascontiguousarray(self._U)

        for _ in range(nstep):
            Q = backend.zeros_like(U)
            U_dag = U.transpose(0, 1, 2, 3, 4, 6, 5).conj()
            for mu in range(Nd - 1):
                for nu in range(Nd - 1):
                    if mu != nu:
                        Q[mu] += (
                            U[nu]
                            @ backend.roll(U[mu], -1, 3 - nu)
                            @ backend.roll(U_dag[nu], -1, 3 - mu)
                        )
                        Q[mu] += (
                            backend.roll(U_dag[nu], +1, 3 - nu)
                            @ backend.roll(U[mu], +1, 3 - nu)
                            @ backend.roll(backend.roll(U[nu], +1, 3 - nu), -1, 3 - mu)
                        )

            Q = rho * Q @ U_dag
            Q = 0.5j * (Q.transpose(0, 1, 2, 3, 4, 6, 5).conj() - Q)
            contract("...aa->...a", Q)[:] -= (
                1 / Nc * contract("...aa->...", Q)[..., None]
            )
            Q_sq = Q @ Q
            c0 = contract("...aa->...", Q @ Q_sq).real / 3
            c1 = contract("...aa->...", Q_sq).real / 2
            c0_max = 2 * (c1 / 3) ** (3 / 2)
            parity = c0 < 0
            c0 = backend.abs(c0)
            theta = backend.arccos(c0 / c0_max)
            u = (c1 / 3) ** 0.5 * backend.cos(theta / 3)
            w = c1**0.5 * backend.sin(theta / 3)
            u_sq = u**2
            w_sq = w**2
            e_iu_real = backend.cos(u)
            e_iu_imag = backend.sin(u)
            e_2iu_real = backend.cos(2 * u)
            e_2iu_imag = backend.sin(2 * u)
            cos_w = backend.cos(w)
            sinc_w = 1 - w_sq / 6 * (1 - w_sq / 20 * (1 - w_sq / 42 * (1 - w_sq / 72)))
            large = backend.abs(w) > 0.05
            w_large = w[large]
            sinc_w[large] = backend.sin(w_large) / w_large
            f_denom = 1 / (9 * u_sq - w_sq)
            f0_real = (
                (u_sq - w_sq) * e_2iu_real
                + e_iu_real * 8 * u_sq * cos_w
                + e_iu_imag * 2 * u * (3 * u_sq + w_sq) * sinc_w
            ) * f_denom
            f0_imag = (
                (u_sq - w_sq) * e_2iu_imag
                - e_iu_imag * 8 * u_sq * cos_w
                + e_iu_real * 2 * u * (3 * u_sq + w_sq) * sinc_w
            ) * f_denom
            f1_real = (
                2 * u * e_2iu_real
                - e_iu_real * 2 * u * cos_w
                + e_iu_imag * (3 * u_sq - w_sq) * sinc_w
            ) * f_denom
            f1_imag = (
                2 * u * e_2iu_imag
                + e_iu_imag * 2 * u * cos_w
                + e_iu_real * (3 * u_sq - w_sq) * sinc_w
            ) * f_denom
            f2_real = (
                e_2iu_real - e_iu_real * cos_w - e_iu_imag * 3 * u * sinc_w
            ) * f_denom
            f2_imag = (
                e_2iu_imag + e_iu_imag * cos_w - e_iu_real * 3 * u * sinc_w
            ) * f_denom
            f0_imag[parity] *= -1
            f1_real[parity] *= -1
            f2_imag[parity] *= -1

            f = (f2_real + 1j * f2_imag)[..., None, None] * Q_sq
            f += (f1_real + 1j * f1_imag)[..., None, None] * Q
            contract("...aa->...a", f)[:] += (f0_real + 1j * f0_imag)[..., None]
            U = f @ U
        self._U = U

    def _stout_smear_cuda_kernel(self, nstep, rho):
        backend = get_backend()
        Lx, Ly, Lz, Lt = self.latt_size
        U = backend.ascontiguousarray(self._U)

        for _ in range(nstep):
            U_in = U.copy()
            self.kernel(
                (Lx * Ly * Lz, Nd - 1, 1), (Lt, 1, 1), (U, U_in, rho, Lx, Ly, Lz, Lt)
            )

        self._U = U

    def _stout_smear_quda(self, nstep, rho):
        backend = get_backend()
        from pyquda_utils import io

        gauge = io.readQIOGauge(self._gauge_field_path)
        gauge.smearSTOUT(nstep, rho, dir_ignore=3)

        self._U = backend.asarray(gauge.lexico()[: Nd - 1])

    def stout_smear(self, nstep, rho):
        from ..backend import check_QUDA

        backend = get_backend()
        if backend.__name__ == "numpy":
            self._stout_smear_ndarray(nstep, rho)
        elif backend.__name__ == "cupy":
            if self.kernel is not None:
                self._stout_smear_cuda_kernel(nstep, rho)
            elif check_QUDA():
                self._stout_smear_quda(nstep, rho)
            else:
                self._stout_smear_ndarray(nstep, rho)
                # self._stout_smear_ndarray_naive(nstep, rho)

    def _gauge_links_product(self, gauge_list, t=None):
        """
        Compute gauge link product over the entire lattice field.

        Applies gauge links U_N...U_2 U_1 at every spatial point.

        Args:
            gauge_list: List of gauge link directions
            t: Time slice index (int or None for all times)

        Returns:
            Gauge link product field:
                - If t is int: shape [Lz, Ly, Lx, Nc, Nc]
                - If t is None: shape [Lt, Lz, Ly, Lx, Nc, Nc]
        """
        backend = get_backend()
        Lx, Ly, Lz, Lt = self.latt_size

        if t is None:
            # All time slices
            gauge_link_product = None
            for d in gauge_list:
                if d < 3:  # Forward link: U_d @ field(x+1)
                    if gauge_link_product is None:
                        gauge_link_product = self._U[d]  # [Lt, Lz, Ly, Lx, Nc, Nc]
                        self._U_shift = backend.roll(self._U, -1, 4 - d)
                    else:
                        # Contract with rolled field

                        gauge_link_product = contract(
                            "tzyxab,tzyxbc->tzyxac",
                            gauge_link_product,
                            self._U[d],
                        )
                        self._U_shift = backend.roll(self._U_shift, -1, 4 - d)
                else:  # Backward link: roll(U_d^† @ field, +1)
                    if gauge_link_product is None:
                        self._U_shift = backend.roll(self._U, 1, 4 - (d - 3))
                        gauge_link_product = (
                            self._U_shift[d - 3].conj().transpose(0, 1, 2, 3, 5, 4)
                        )
                    else:
                        self._U_shift = backend.roll(self._U_shift, 1, 4 - (d - 3))
                        gauge_link_product = contract(
                            "tzyxab,tzyxbc->tzyxac",
                            gauge_link_product,
                            self._U_shift[d - 3].conj().transpose(0, 1, 2, 3, 5, 4),
                        )

            return gauge_link_product  # [Lt, Lz, Ly, Lx, Nc, Nc]

        else:
            # Single time slice
            gauge_link_product = None

            for d in gauge_list:
                if d < 3:  # Forward link: U_d @ field(x+1)
                    if gauge_link_product is None:
                        gauge_link_product = self._U[d, t]  # [Lz, Ly, Lx, Nc, Nc]
                        self._U_shift = backend.roll(self._U[:, t], -1, 3 - d)
                    else:
                        # Contract with rolled field
                        gauge_link_product = contract(
                            "zyxab,zyxbc->zyxac",
                            gauge_link_product,
                            self._U_shift[d],
                        )
                        self._U_shift = backend.roll(self._U_shift, -1, 3 - d)
                else:  # Backward link: roll(U_d^† @ field, +1)
                    if gauge_link_product is None:
                        self._U_shift = backend.roll(self._U[:, t], 1, 3 - (d - 3))
                        gauge_link_product = (
                            self._U_shift[d - 3].conj().transpose(0, 1, 2, 4, 3)
                        )
                    else:
                        self._U_shift = backend.roll(self._U_shift, 1, 3 - (d - 3))
                        gauge_link_product = contract(
                            "zyxab,zyxbc->zyxac",
                            gauge_link_product,
                            self._U_shift[d - 3].conj().transpose(0, 1, 2, 4, 3),
                        )

            return gauge_link_product  # [Lz, Ly, Lx, Nc, Nc]

    def _apply_gauge_links_to_points(self, point_left, gauge_list, t=None):
        """
        Apply gauge links along a path starting from point positions.

        Vectorized computation for all points, optionally for all time slices.

        Args:
            point_left: Point coordinates, shape [Np, Lt, 3] or [Np, 3] if t is specified
            gauge_list: List of gauge link directions
            t: Time slice index (int or None for all times)

        Returns:
            gauge_link_product: Gauge link matrices at each point
                - If t is int: shape [Np, Nc, Nc]
                - If t is None: shape [Lt, Np, Nc, Nc]
        """
        from copy import deepcopy

        backend = get_backend()
        Lx, Ly, Lz, Lt = self.latt_size
        sizes = [Lx, Ly, Lz]

        # Handle t parameter
        if t is None:
            # Vectorize over all time slices
            # point_left shape: [Np, Lt, 3]
            Np = point_left.shape[0]
            point_shifted = deepcopy(point_left)  # [Np, Lt, 3]

            # Index arrays for vectorized access
            t_idx = backend.arange(Lt)[None, :, None]  # [1, Lt, 1]
            p_idx = backend.arange(Np)[:, None, None]  # [Np, 1, 1]

            gauge_link_product = None

            for d in gauge_list:
                if d < 3:  # Forward link: U_d
                    point_shifted[:, :, d] = (point_shifted[:, :, d] + 1) % sizes[d]
                    U_current = self._U[
                        d,
                        t_idx.squeeze(0).squeeze(-1),
                        backend.asarray(
                            point_shifted[
                                p_idx.squeeze((1, 2)), t_idx.squeeze((0, 2)), 2
                            ],
                            dtype=int,
                        ),
                        backend.asarray(
                            point_shifted[
                                p_idx.squeeze((1, 2)), t_idx.squeeze((0, 2)), 1
                            ],
                            dtype=int,
                        ),
                        backend.asarray(
                            point_shifted[
                                p_idx.squeeze((1, 2)), t_idx.squeeze((0, 2)), 0
                            ],
                            dtype=int,
                        ),
                        :,
                        :,
                    ]  # [Np, Lt, Nc, Nc]

                    if gauge_link_product is None:
                        gauge_link_product = U_current
                    else:
                        gauge_link_product = contract(
                            "ptab,ptbc->ptac", gauge_link_product, U_current
                        )

                else:  # Backward link: U_d^†
                    U_current = (
                        self._U[
                            d - 3,
                            t_idx.squeeze(0).squeeze(-1),
                            backend.asarray(
                                point_shifted[
                                    p_idx.squeeze((1, 2)), t_idx.squeeze((0, 2)), 2
                                ],
                                dtype=int,
                            ),
                            backend.asarray(
                                point_shifted[
                                    p_idx.squeeze((1, 2)), t_idx.squeeze((0, 2)), 1
                                ],
                                dtype=int,
                            ),
                            backend.asarray(
                                point_shifted[
                                    p_idx.squeeze((1, 2)), t_idx.squeeze((0, 2)), 0
                                ],
                                dtype=int,
                            ),
                            :,
                            :,
                        ]
                        .conj()
                        .transpose(0, 1, 3, 2)
                    )  # [Np, Lt, Nc, Nc]

                    if gauge_link_product is None:
                        gauge_link_product = U_current
                    else:
                        gauge_link_product = contract(
                            "ptab,ptbc->ptac", gauge_link_product, U_current
                        )

                    point_shifted[:, :, d - 3] = (
                        point_shifted[:, :, d - 3] - 1
                    ) % sizes[d - 3]

            # Return shape: [Np, Lt, Nc, Nc] -> transpose to [Lt, Np, Nc, Nc]
            if gauge_link_product is not None:
                return gauge_link_product.transpose(1, 0, 2, 3)
            else:
                return None

        else:
            # Single time slice t
            # point_left can be [Np, Lt, 3] or [Np, 3]
            if len(point_left.shape) == 3:
                point_coords = point_left[:, t, :]  # [Np, 3]
            else:
                point_coords = point_left  # Already [Np, 3]

            Np = point_coords.shape[0]
            point_shifted = deepcopy(point_coords)  # [Np, 3]

            # Index array for points
            p_idx = backend.arange(Np)[:, None]  # [Np, 1]

            gauge_link_product = None

            for d in gauge_list:
                if d < 3:  # Forward link: U_d
                    point_shifted[:, d] = (point_shifted[:, d] + 1) % sizes[d]
                    U_current = self._U[
                        d,
                        t,
                        backend.asarray(point_shifted[p_idx.squeeze(1), 2], dtype=int),
                        backend.asarray(point_shifted[p_idx.squeeze(1), 1], dtype=int),
                        backend.asarray(point_shifted[p_idx.squeeze(1), 0], dtype=int),
                        :,
                        :,
                    ]  # [Np, Nc, Nc]

                    if gauge_link_product is None:
                        gauge_link_product = U_current
                    else:
                        gauge_link_product = contract(
                            "pab,pbc->pac", gauge_link_product, U_current
                        )

                else:  # Backward link: U_d^†
                    U_current = (
                        self._U[
                            d - 3,
                            t,
                            backend.asarray(
                                point_shifted[p_idx.squeeze(1), 2], dtype=int
                            ),
                            backend.asarray(
                                point_shifted[p_idx.squeeze(1), 1], dtype=int
                            ),
                            backend.asarray(
                                point_shifted[p_idx.squeeze(1), 0], dtype=int
                            ),
                            :,
                            :,
                        ]
                        .conj()
                        .transpose(0, 2, 1)
                    )  # [Np, Nc, Nc]

                    if gauge_link_product is None:
                        gauge_link_product = U_current
                    else:
                        gauge_link_product = contract(
                            "pab,pbc->pac", gauge_link_product, U_current
                        )

                    point_shifted[:, d - 3] = (point_shifted[:, d - 3] - 1) % sizes[
                        d - 3
                    ]

            return gauge_link_product  # [Np, Nc, Nc]

    def _nD(self, V, U, deriv):
        backend = get_backend()

        for d in deriv:
            Vf = backend.roll(V, -1, 3 - d)
            UVf = contract("zyxab,ezyxb->ezyxa", U[d], Vf)
            UdV = contract("zyxba,ezyxb->ezyxa", U[d].conj(), V)
            UbdVb = backend.roll(UdV, 1, 3 - d)
            V = UVf - UbdVb
        return V

    def _disp(self, V, U, d):
        backend = get_backend()
        if d < 3:
            Vf = backend.roll(V, -1, 3 - d)
            UVf = contract("zyxab,ezyxb->ezyxa", U[d], Vf)
            return UVf
        else:
            UdV = contract("zyxba,ezyxb->ezyxa", U[d - 3].conj(), V)
            UbdVb = backend.roll(UdV, 1, 6 - d)
            return UbdVb

    def calc_deriv(self, t: int):
        if self._V is None or self._VPV is None:
            raise ValueError(
                "ElementalGenerator.calc_deriv requires calc_mode='calc_deriv'."
            )
        eigenvector = self._eigenvector_data
        momentum_phase = self._momentum_phase
        U = self._U
        V = self._V
        VPV = self._VPV

        for e in range(self.usedNe):
            V[e] = eigenvector[t, e]
        for derivative_idx, derivative in enumerate(self.derivative_list):
            VPV[derivative_idx] = 0
            # for num_nabla_right in range(len(derivative) + 1):
            #     coeff = (-1)**num_nabla_right * comb(len(derivative), num_nabla_right)
            #     right = self._nD(V, U[:, t], derivative[:num_nabla_right])
            #     left = self._nD(V, U[:, t], derivative[num_nabla_right:][::-1])
            #     for momentum_idx, momentum in enumerate(self.momentum_list):
            #         VPV[derivative_idx, momentum_idx] += contract(
            #             "zyx,ezyxc,fzyxc->ef", coeff * momentum_phase.get(momentum), left.conj(), right
            #         )
            for pick_nabla in range(2 ** len(derivative)):
                pick_right = []
                pick_left = []
                pick = pick_nabla
                for direction in derivative:
                    if pick & 1:
                        pick_right.append(direction)
                    else:
                        pick_left.append(direction)
                    pick >>= 1
                coeff = (-1) ** len(pick_right)
                right = self._nD(V, U[:, t], pick_right)
                left = self._nD(V, U[:, t], pick_left[::-1])
                for momentum_idx, momentum in enumerate(self.momentum_list):
                    if self.stocastic_coeff is None:
                        VPV[derivative_idx, momentum_idx] += contract(
                            "zyx,ezyxc,fzyxc->ef",
                            coeff * momentum_phase.get(momentum),
                            left.conj(),
                            right,
                        )
                    else:
                        VPV[derivative_idx, momentum_idx] += contract(
                            "zyx,ezyxc,fzyxc,ef->ef",
                            coeff * momentum_phase.get(momentum),
                            left.conj(),
                            right,
                            self.stocastic_coeff[: self.usedNe, : self.usedNe],
                        )
        return VPV

    def calc(self, t: int):
        if self.calc_mode == "calc_deriv":
            return self.calc_deriv(t)
        return self.calc_disp(t)

    def calc_disp(self, t: int):
        from ..insertion.gauge_link import GaugeLink

        backend = get_backend()
        Lx, Ly, Lz, Lt = self.latt_size

        if self.debug:
            print(f"\nCurrentElementalGenerator.calc_v2v() called:")
            print(f"  t = {t}")
            print(f"  self.Ne = {self.Ne}, self.usedNe = {self.usedNe}")
            print(f"  num_derivative = {self.num_derivative}")
        # Initialize or reuse result buffer (fixed in __init__ for calc_disp mode).
        if self._disp_buffer is None:
            result = backend.zeros(
                (self.num_disp, self.num_momentum, self.usedNe, self.usedNe),
                dtype="<c16",
            )
        else:
            result = self._disp_buffer
            result[...] = 0

        # Reuse preallocated buffers for calc_disp path.
        if self._calc_disp_V_buffer is None:
            self._calc_disp_V_buffer = backend.zeros((self.usedNe, Lz, Ly, Lx, Nc), "<c8")
        if self._calc_disp_shift_buffer is None:
            self._calc_disp_shift_buffer = backend.zeros(
                (self.usedNe, Lz, Ly, Lx, Nc), "<c8"
            )
        if self._calc_disp_gauge_buffer is None:
            self._calc_disp_gauge_buffer = backend.zeros((Lz, Ly, Lx, Nc, Nc), "<c16")

        # Get eigenvector data for this time slice
        V = self._calc_disp_V_buffer
        V[...] = self._eigenvector_data[t, : self.usedNe]

        for disp_idx in range(self.num_disp):
            gauge_link = GaugeLink(disp_idx)
            gauge_list = gauge_link.gauge_list
            disp = gauge_link.displacement
            displacement = (
                (disp[0] / 2) * 2j * backend.pi / Lx,
                (disp[1] / 2) * 2j * backend.pi / Ly,
                (disp[2] / 2) * 2j * backend.pi / Lz,
            )
            shift_V = self._calc_disp_shift_buffer
            shift_V[...] = V
            for coord_idx in range(3):
                if disp[coord_idx] != 0:
                    shift_V[...] = backend.roll(shift_V, -disp[coord_idx], 3 - coord_idx)

            if self.debug:
                print(f"\n  Processing disp_idx {disp_idx}:")
                print(f"    displacement: {disp}")
                print(f"    gauge_list: {gauge_list}")
                print(f"    actual displacement: {displacement}")

            # Compute gauge link product on entire field
            gauge_link_product = self._gauge_links_product(gauge_list, t=t)
            if gauge_link_product is not None:
                self._calc_disp_gauge_buffer[...] = gauge_link_product
                gauge_link_product = self._calc_disp_gauge_buffer

            if gauge_link_product is None:
                # No gauge links (identity case)
                if self.debug:
                    print(f"    No gauge links, using identity")

                # Direct contraction without gauge links
                for momentum_idx, momentum in enumerate(self.momentum_list):
                    phase = self._momentum_phase.get(momentum)
                    result_val = contract(
                        "zyx,ezyxc,fzyxc->ef",
                        phase,
                        V.conj(),
                        V,
                    )
                    result[disp_idx, momentum_idx] = result_val

                    if self.debug:
                        print(
                            f"    momentum_idx {momentum_idx}, result[0,0] = {result_val[0,0]}"
                        )
            else:
                # gauge_link_product shape: [Lz, Ly, Lx, Nc, Nc]
                if self.debug:
                    print(f"    gauge_link_product shape: {gauge_link_product.shape}")
                    print(
                        f"    gauge_link_product[0, 0, 0, 0, 0] = {gauge_link_product[0, 0, 0, 0, 0]}"
                    )

                # Contract: <V_left | gauge_link_product | V_right>
                # V_left: [Ne, Lz, Ly, Lx, Nc]
                # gauge_link_product: [Lz, Ly, Lx, Nc, Nc]
                # V_right: [Ne, Lz, Ly, Lx, Nc]
                # Result: [Ne, Ne]
                for momentum_idx, momentum in enumerate(self.momentum_list):
                    phase = self._momentum_phase.get(momentum)
                    disp_phase = backend.exp(
                        sum([momentum[i] * displacement[i] for i in range(3)])
                    )
                    result_val = disp_phase * contract(
                        "zyx,ezyxa,zyxac,fzyxc->ef",
                        phase,
                        V.conj(),
                        gauge_link_product,
                        shift_V,
                    )
                    result[disp_idx, momentum_idx] = result_val

                    if self.debug:
                        print(f"    momentum_idx {momentum_idx}, momentum {momentum}")
                        print(f"    disp_phase: {disp_phase}")
                        print(f"    result[0,0] = {result_val[0,0]}")

        return result

    @classmethod
    def _re_combine_dense(cls, array, insertion_list, axis, out=None):
        backend = get_backend()
        coeff = backend.zeros((len(insertion_list), array.shape[axis]))
        for i in range(len(insertion_list)):
            for j in range(len(insertion_list[i])):
                disp_idx = insertion_list[i][j][1]
                coeff[i, disp_idx] += float(insertion_list[i][j][0])
        contracted = backend.tensordot(coeff, array, axes=([1], [axis]))
        result = backend.moveaxis(contracted, 0, axis)
        if out is not None:
            out[...] = result
            return out
        return result

    @classmethod
    def _re_combine_sparse(cls, array, insertion_list, axis, out=None):
        backend = get_backend()
        axis = axis if axis >= 0 else array.ndim + axis
        num_row = len(insertion_list)
        num_col = array.shape[axis]

        rows = []
        cols = []
        vals = []
        for i in range(num_row):
            for coeff_val, disp_idx in insertion_list[i]:
                rows.append(i)
                cols.append(disp_idx)
                vals.append(float(coeff_val))

        if backend.__name__ == "cupy":
            from cupyx.scipy import sparse as sparse_backend

            rows = backend.asarray(rows, dtype=backend.int32).ravel()
            cols = backend.asarray(cols, dtype=backend.int32).ravel()
            vals = backend.asarray(vals, dtype=array.dtype).ravel()
        else:
            from scipy import sparse as sparse_backend
            import numpy as np

            rows = np.asarray(rows, dtype=np.int32).ravel()
            cols = np.asarray(cols, dtype=np.int32).ravel()
            vals = np.asarray(vals, dtype=array.dtype).ravel()

        coeff = sparse_backend.coo_matrix(
            (vals, (rows, cols)), shape=(num_row, num_col)
        )
        coeff = coeff.tocsr()

        moved = backend.moveaxis(array, axis, 0)
        moved_2d = moved.reshape((num_col, -1))
        contracted_2d = coeff.dot(moved_2d)
        contracted = contracted_2d.reshape((num_row,) + tuple(moved.shape[1:]))
        result = backend.moveaxis(contracted, 0, axis)
        if out is not None:
            out[...] = result
            return out
        return result

    @classmethod
    def re_combine_auto(cls, array, insertion_list, axis, out=None):
        call_idx = cls._recombine_benchmark_calls
        cls._recombine_benchmark_calls += 1

        if not cls._recombine_validated:
            t0 = perf_counter()
            dense_result = cls._re_combine_dense(array, insertion_list, axis)
            dense_elapsed = perf_counter() - t0

            t1 = perf_counter()
            sparse_result = cls._re_combine_sparse(array, insertion_list, axis)
            sparse_elapsed = perf_counter() - t1

            backend = get_backend()
            if hasattr(backend, "allclose"):
                ok = backend.allclose(
                    dense_result,
                    sparse_result,
                    rtol=cls._recombine_validate_rtol,
                    atol=cls._recombine_validate_atol,
                )
                ok = bool(ok)
            else:
                import numpy as np

                ok = np.allclose(
                    dense_result,
                    sparse_result,
                    rtol=cls._recombine_validate_rtol,
                    atol=cls._recombine_validate_atol,
                )

            if not ok:
                max_abs_diff = float(
                    backend.max(backend.abs(dense_result - sparse_result))
                )
                raise ValueError(
                    f"re_combine validation failed: dense vs sparse mismatch, max_abs_diff={max_abs_diff}"
                )

            cls._recombine_validated = True
            cls._recombine_elapsed_dense.append(dense_elapsed)
            cls._recombine_elapsed_sparse.append(sparse_elapsed)
            if out is not None:
                out[...] = dense_result
                return out
            return dense_result

        if cls._recombine_selected_method is None and call_idx < 6:
            method = "dense" if (call_idx % 2 == 0) else "sparse"
        else:
            if cls._recombine_selected_method is None:
                dense_n = min(3, len(cls._recombine_elapsed_dense))
                sparse_n = min(3, len(cls._recombine_elapsed_sparse))
                dense_avg = sum(cls._recombine_elapsed_dense[:dense_n]) / max(
                    dense_n, 1
                )
                sparse_avg = sum(cls._recombine_elapsed_sparse[:sparse_n]) / max(
                    sparse_n, 1
                )
                cls._recombine_selected_method = (
                    "sparse" if sparse_avg < dense_avg else "dense"
                )
            method = cls._recombine_selected_method

        t0 = perf_counter()
        if method == "sparse":
            result = cls._re_combine_sparse(array, insertion_list, axis, out=out)
        else:
            result = cls._re_combine_dense(array, insertion_list, axis, out=out)
        elapsed = perf_counter() - t0

        if call_idx < 6:
            if method == "sparse":
                cls._recombine_elapsed_sparse.append(elapsed)
            else:
                cls._recombine_elapsed_dense.append(elapsed)

            if (
                cls._recombine_selected_method is None
                and len(cls._recombine_elapsed_dense) >= 3
                and len(cls._recombine_elapsed_sparse) >= 3
            ):
                dense_avg = sum(cls._recombine_elapsed_dense[:3]) / 3
                sparse_avg = sum(cls._recombine_elapsed_sparse[:3]) / 3
                cls._recombine_selected_method = (
                    "sparse" if sparse_avg < dense_avg else "dense"
                )

        return result


class CurrentElementalGenerator:
    """
    Generator for current vertex elemental diagrams (v2p, p2v, p2p).

    Computes propagation from/to eigenvectors and points through gauge links.
    Precomputes and stores results for later use in Current class.

    Args:
        latt_size: [Lx, Ly, Lz, Lt] lattice dimensions
        gauge_field: GaugeField data loader
        eigenvector: Eigenvector data loader
        point: PointSource data loader
        num_nabla: Number of covariant derivatives (displacement degree)
        momentum_list: List of momentum tuples [(px, py, pz), ...]
        usedNe: Number of eigenvectors to use (default: all)
        usedNp: Number of points to use (default: all)
        debug: Enable debug output
    """

    _recombine_benchmark_calls = 0
    _recombine_selected_method = None
    _recombine_elapsed_dense = []
    _recombine_elapsed_sparse = []
    _recombine_validated = False
    _recombine_validate_rtol = 1e-6
    _recombine_validate_atol = 1e-10

    def __init__(
        self,
        latt_size: List[int],
        gauge_field: GaugeField,
        eigenvector: Eigenvector,
        point: "PointSource",
        num_nabla: int = 0,
        momentum_list: List[Tuple[int]] = [(0, 0, 0)],
        usedNe: int = None,
        usedNp: int = None,
        debug: bool = False,
    ) -> None:
        from ..insertion.phase import MomentumPhase
        from ..insertion.gauge_link import GaugeLink

        backend = get_backend()
        Lx, Ly, Lz, Lt = latt_size

        # Copy initialization from ElementalGenerator
        self.kernel = None
        if backend.__name__ == "cupy":
            import os

            with open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "stout_smear.cu"
                )
            ) as f:
                code = f.read()
            self.kernel = backend.RawModule(
                code=code,
                options=("--std=c++11",),
                name_expressions=("stout_smear<double>",),
            ).get_function("stout_smear<double>")

        self.latt_size = latt_size
        self.gauge_field = gauge_field
        self.eigenvector = eigenvector
        self.point = point
        self.num_nabla = num_nabla
        self.num_momentum = len(momentum_list)
        self.momentum_list = momentum_list
        self.debug = debug

        # Use GaugeLink to determine number of displacements
        self.num_disp = list(GaugeLink.nmax_generator(num_nabla))[-1]

        Ne = eigenvector.Ne
        Np = point.Np
        self.Ne = Ne
        self.Np = Np
        self.usedNe = usedNe if usedNe is not None else Ne
        self.usedNp = usedNp if usedNp is not None else Np

        # Data storage
        self._U = None
        self._eigenvector_data = None
        self._point_data = None
        self._gauge_field_path = None
        self._momentum_phase = MomentumPhase(latt_size)

    def load(self, key: str):
        """Load gauge field, eigenvector, and point data."""
        self._U = self.gauge_field.load(key)[:].transpose(4, 0, 1, 2, 3, 5, 6)[: Nd - 1]
        self._gauge_field_path = self.gauge_field.load(key).file
        self._eigenvector_data = self.eigenvector.load(key)[:]
        self._point_data = self.point.load(key)[:]
        self.gauge_field.data = None  # Free memory

    def clear_loaded_data(self):
        """Clear loaded data to free GPU memory."""
        # Force garbage collection to free GPU memory
        import gc

        backend = get_backend()

        if backend.__name__ == "cupy":
            # For CuPy, we need to explicitly synchronize and ensure memory is freed
            if self._U is not None:
                del self._U
            if self._eigenvector_data is not None:
                del self._eigenvector_data
            if self._point_data is not None:
                del self._point_data

            # Clear data loaders' cache
            if hasattr(self.gauge_field, "data"):
                self.gauge_field.data = None
            if hasattr(self.eigenvector, "data"):
                self.eigenvector.data = None
            if hasattr(self.point, "data"):
                self.point.data = None

            # Force synchronization and garbage collection
            backend.cuda.device.Device().synchronize()
            gc.collect()

            # Additional memory pool cleanup for CuPy
            try:
                mempool = backend.cuda.memory.MemoryPool()
                mempool.free_all_blocks()
            except:
                pass

        else:
            del self._U
            del self._eigenvector_data
            del self._point_data
            gc.collect()

        self._U = None
        self._eigenvector_data = None
        self._point_data = None
        self._gauge_field_path = None

    def project_SU3(self):
        """Project gauge field to SU3. Copied from ElementalGenerator."""
        backend = get_backend()
        U = self._U
        Uinv = backend.linalg.inv(U)
        while (
            backend.max(backend.abs(U - contract("...ab->...ba", Uinv.conj()))) > 1e-15
            or backend.max(
                backend.abs(contract("...ab,...cb", U, U.conj()) - backend.identity(Nc))
            )
            > 1e-15
        ):
            U = 0.5 * (U + contract("...ab->...ba", Uinv.conj()))
            Uinv = backend.linalg.inv(U)
        self._U = U

    def _stout_smear_ndarray(self, nstep, rho):
        """Stout smearing using ndarray backend. Copied from ElementalGenerator."""
        backend = get_backend()
        U = backend.ascontiguousarray(self._U)

        for _ in range(nstep):
            Q = backend.zeros_like(U)
            U_dag = U.transpose(0, 1, 2, 3, 4, 6, 5).conj()
            for mu in range(Nd - 1):
                for nu in range(Nd - 1):
                    if mu != nu:
                        Q[mu] += (
                            U[nu]
                            @ backend.roll(U[mu], -1, 3 - nu)
                            @ backend.roll(U_dag[nu], -1, 3 - mu)
                        )
                        Q[mu] += (
                            backend.roll(U_dag[nu], +1, 3 - nu)
                            @ backend.roll(U[mu], +1, 3 - nu)
                            @ backend.roll(backend.roll(U[nu], +1, 3 - nu), -1, 3 - mu)
                        )

            Q = rho * Q @ U_dag
            Q = 0.5j * (Q.transpose(0, 1, 2, 3, 4, 6, 5).conj() - Q)
            contract("...aa->...a", Q)[:] -= (
                1 / Nc * contract("...aa->...", Q)[..., None]
            )
            Q_sq = Q @ Q
            c0 = contract("...aa->...", Q @ Q_sq).real / 3
            c1 = contract("...aa->...", Q_sq).real / 2
            c0_max = 2 * (c1 / 3) ** (3 / 2)
            parity = c0 < 0
            c0 = backend.abs(c0)
            theta = backend.arccos(c0 / c0_max)
            u = (c1 / 3) ** 0.5 * backend.cos(theta / 3)
            w = c1**0.5 * backend.sin(theta / 3)
            u_sq = u**2
            w_sq = w**2
            e_iu_real = backend.cos(u)
            e_iu_imag = backend.sin(u)
            e_2iu_real = backend.cos(2 * u)
            e_2iu_imag = backend.sin(2 * u)
            cos_w = backend.cos(w)
            sinc_w = 1 - w_sq / 6 * (1 - w_sq / 20 * (1 - w_sq / 42 * (1 - w_sq / 72)))
            large = backend.abs(w) > 0.05
            w_large = w[large]
            sinc_w[large] = backend.sin(w_large) / w_large
            f_denom = 1 / (9 * u_sq - w_sq)
            f0_real = (
                (u_sq - w_sq) * e_2iu_real
                + e_iu_real * 8 * u_sq * cos_w
                + e_iu_imag * 2 * u * (3 * u_sq + w_sq) * sinc_w
            ) * f_denom
            f0_imag = (
                (u_sq - w_sq) * e_2iu_imag
                - e_iu_imag * 8 * u_sq * cos_w
                + e_iu_real * 2 * u * (3 * u_sq + w_sq) * sinc_w
            ) * f_denom
            f1_real = (
                2 * u * e_2iu_real
                - e_iu_real * 2 * u * cos_w
                + e_iu_imag * (3 * u_sq - w_sq) * sinc_w
            ) * f_denom
            f1_imag = (
                2 * u * e_2iu_imag
                + e_iu_imag * 2 * u * cos_w
                + e_iu_real * (3 * u_sq - w_sq) * sinc_w
            ) * f_denom
            f2_real = (
                e_2iu_real - e_iu_real * cos_w - e_iu_imag * 3 * u * sinc_w
            ) * f_denom
            f2_imag = (
                e_2iu_imag + e_iu_imag * cos_w - e_iu_real * 3 * u * sinc_w
            ) * f_denom
            f0_imag[parity] *= -1
            f1_real[parity] *= -1
            f2_imag[parity] *= -1

            f = (f2_real + 1j * f2_imag)[..., None, None] * Q_sq
            f += (f1_real + 1j * f1_imag)[..., None, None] * Q
            contract("...aa->...a", f)[:] += (f0_real + 1j * f0_imag)[..., None]
            U = f @ U
        self._U = U

    def _stout_smear_cuda_kernel(self, nstep, rho):
        """Stout smearing using CUDA kernel. Copied from ElementalGenerator."""
        backend = get_backend()
        Lx, Ly, Lz, Lt = self.latt_size
        U = backend.ascontiguousarray(self._U)

        for _ in range(nstep):
            U_in = U.copy()
            self.kernel(
                (Lx * Ly * Lz, Nd - 1, 1), (Lt, 1, 1), (U, U_in, rho, Lx, Ly, Lz, Lt)
            )

        self._U = U

    def _stout_smear_quda(self, nstep, rho):
        """Stout smearing using QUDA. Copied from ElementalGenerator."""
        backend = get_backend()
        from pyquda_utils import io

        gauge = io.readQIOGauge(self._gauge_field_path)
        gauge.smearSTOUT(nstep, rho, dir_ignore=3)

        self._U = backend.asarray(gauge.lexico()[: Nd - 1])

    def stout_smear(self, nstep, rho):
        """Apply stout smearing. Copied from ElementalGenerator."""
        from ..backend import check_QUDA

        backend = get_backend()
        if backend.__name__ == "numpy":
            self._stout_smear_ndarray(nstep, rho)
        elif backend.__name__ == "cupy":
            if self.kernel is not None:
                self._stout_smear_cuda_kernel(nstep, rho)
            elif check_QUDA():
                self._stout_smear_quda(nstep, rho)
            else:
                self._stout_smear_ndarray(nstep, rho)

    def _gauge_links_product(self, gauge_list, t=None):
        """
        Compute gauge link product over the entire lattice field.

        Applies gauge links U_N...U_2 U_1 at every spatial point.

        Args:
            gauge_list: List of gauge link directions
            t: Time slice index (int or None for all times)

        Returns:
            Gauge link product field:
                - If t is int: shape [Lz, Ly, Lx, Nc, Nc]
                - If t is None: shape [Lt, Lz, Ly, Lx, Nc, Nc]
        """
        backend = get_backend()
        Lx, Ly, Lz, Lt = self.latt_size

        if t is None:
            # All time slices
            gauge_link_product = None
            for d in gauge_list:
                if d < 3:  # Forward link: U_d @ field(x+1)
                    if gauge_link_product is None:
                        gauge_link_product = self._U[d]  # [Lt, Lz, Ly, Lx, Nc, Nc]
                        self._U_shift = backend.roll(self._U, -1, 4 - d)
                    else:
                        # Contract with rolled field

                        gauge_link_product = contract(
                            "tzyxab,tzyxbc->tzyxac",
                            gauge_link_product,
                            self._U[d],
                        )
                        self._U_shift = backend.roll(self._U_shift, -1, 4 - d)
                else:  # Backward link: roll(U_d^† @ field, +1)
                    if gauge_link_product is None:
                        self._U_shift = backend.roll(self._U, 1, 4 - (d - 3))
                        gauge_link_product = (
                            self._U_shift[d - 3].conj().transpose(0, 1, 2, 3, 5, 4)
                        )
                    else:
                        self._U_shift = backend.roll(self._U_shift, 1, 4 - (d - 3))
                        gauge_link_product = contract(
                            "tzyxab,tzyxbc->tzyxac",
                            gauge_link_product,
                            self._U_shift[d - 3].conj().transpose(0, 1, 2, 3, 5, 4),
                        )

            return gauge_link_product  # [Lt, Lz, Ly, Lx, Nc, Nc]

        else:
            # Single time slice
            gauge_link_product = None

            for d in gauge_list:
                if d < 3:  # Forward link: U_d @ field(x+1)
                    if gauge_link_product is None:
                        gauge_link_product = self._U[d, t]  # [Lz, Ly, Lx, Nc, Nc]
                        self._U_shift = backend.roll(self._U[:, t], -1, 3 - d)
                    else:
                        # Contract with rolled field
                        gauge_link_product = contract(
                            "zyxab,zyxbc->zyxac",
                            gauge_link_product,
                            self._U_shift[d],
                        )
                        self._U_shift = backend.roll(self._U_shift, -1, 3 - d)
                else:  # Backward link: roll(U_d^† @ field, +1)
                    if gauge_link_product is None:
                        self._U_shift = backend.roll(self._U[:, t], 1, 3 - (d - 3))
                        gauge_link_product = (
                            self._U_shift[d - 3].conj().transpose(0, 1, 2, 4, 3)
                        )
                    else:
                        self._U_shift = backend.roll(self._U_shift, 1, 3 - (d - 3))
                        gauge_link_product = contract(
                            "zyxab,zyxbc->zyxac",
                            gauge_link_product,
                            self._U_shift[d - 3].conj().transpose(0, 1, 2, 4, 3),
                        )

            return gauge_link_product  # [Lz, Ly, Lx, Nc, Nc]

    def _apply_gauge_links_to_points(self, point_left, gauge_list, t=None):
        """
        Apply gauge links along a path starting from point positions.

        Vectorized computation for all points, optionally for all time slices.

        Args:
            point_left: Point coordinates, shape [Np, Lt, 3] or [Np, 3] if t is specified
            gauge_list: List of gauge link directions
            t: Time slice index (int or None for all times)

        Returns:
            gauge_link_product: Gauge link matrices at each point
                - If t is int: shape [Np, Nc, Nc]
                - If t is None: shape [Lt, Np, Nc, Nc]
        """
        from copy import deepcopy

        backend = get_backend()
        Lx, Ly, Lz, Lt = self.latt_size
        sizes = [Lx, Ly, Lz]

        # Handle t parameter
        if t is None:
            # Vectorize over all time slices
            # point_left shape: [Np, Lt, 3]
            Np = point_left.shape[0]
            point_shifted = deepcopy(point_left)  # [Np, Lt, 3]

            # Index arrays for vectorized access
            t_idx = backend.arange(Lt)[None, :, None]  # [1, Lt, 1]
            p_idx = backend.arange(Np)[:, None, None]  # [Np, 1, 1]

            gauge_link_product = None

            for d in gauge_list:
                if d < 3:  # Forward link: U_d
                    point_shifted[:, :, d] = (point_shifted[:, :, d] + 1) % sizes[d]
                    U_current = self._U[
                        d,
                        t_idx.squeeze(0).squeeze(-1),
                        backend.asarray(
                            point_shifted[
                                p_idx.squeeze((1, 2)), t_idx.squeeze((0, 2)), 2
                            ],
                            dtype=int,
                        ),
                        backend.asarray(
                            point_shifted[
                                p_idx.squeeze((1, 2)), t_idx.squeeze((0, 2)), 1
                            ],
                            dtype=int,
                        ),
                        backend.asarray(
                            point_shifted[
                                p_idx.squeeze((1, 2)), t_idx.squeeze((0, 2)), 0
                            ],
                            dtype=int,
                        ),
                        :,
                        :,
                    ]  # [Np, Lt, Nc, Nc]

                    if gauge_link_product is None:
                        gauge_link_product = U_current
                    else:
                        gauge_link_product = contract(
                            "ptab,ptbc->ptac", gauge_link_product, U_current
                        )

                else:  # Backward link: U_d^†
                    U_current = (
                        self._U[
                            d - 3,
                            t_idx.squeeze(0).squeeze(-1),
                            backend.asarray(
                                point_shifted[
                                    p_idx.squeeze((1, 2)), t_idx.squeeze((0, 2)), 2
                                ],
                                dtype=int,
                            ),
                            backend.asarray(
                                point_shifted[
                                    p_idx.squeeze((1, 2)), t_idx.squeeze((0, 2)), 1
                                ],
                                dtype=int,
                            ),
                            backend.asarray(
                                point_shifted[
                                    p_idx.squeeze((1, 2)), t_idx.squeeze((0, 2)), 0
                                ],
                                dtype=int,
                            ),
                            :,
                            :,
                        ]
                        .conj()
                        .transpose(0, 1, 3, 2)
                    )  # [Np, Lt, Nc, Nc]

                    if gauge_link_product is None:
                        gauge_link_product = U_current
                    else:
                        gauge_link_product = contract(
                            "ptab,ptbc->ptac", gauge_link_product, U_current
                        )

                    point_shifted[:, :, d - 3] = (
                        point_shifted[:, :, d - 3] - 1
                    ) % sizes[d - 3]

            # Return shape: [Np, Lt, Nc, Nc] -> transpose to [Lt, Np, Nc, Nc]
            if gauge_link_product is not None:
                return gauge_link_product.transpose(1, 0, 2, 3)
            else:
                return None

        else:
            # Single time slice t
            # point_left can be [Np, Lt, 3] or [Np, 3]
            if len(point_left.shape) == 3:
                point_coords = point_left[:, t, :]  # [Np, 3]
            else:
                point_coords = point_left  # Already [Np, 3]

            Np = point_coords.shape[0]
            point_shifted = deepcopy(point_coords)  # [Np, 3]

            # Index array for points
            p_idx = backend.arange(Np)[:, None]  # [Np, 1]

            gauge_link_product = None

            for d in gauge_list:
                if d < 3:  # Forward link: U_d
                    point_shifted[:, d] = (point_shifted[:, d] + 1) % sizes[d]
                    U_current = self._U[
                        d,
                        t,
                        backend.asarray(point_shifted[p_idx.squeeze(1), 2], dtype=int),
                        backend.asarray(point_shifted[p_idx.squeeze(1), 1], dtype=int),
                        backend.asarray(point_shifted[p_idx.squeeze(1), 0], dtype=int),
                        :,
                        :,
                    ]  # [Np, Nc, Nc]

                    if gauge_link_product is None:
                        gauge_link_product = U_current
                    else:
                        gauge_link_product = contract(
                            "pab,pbc->pac", gauge_link_product, U_current
                        )

                else:  # Backward link: U_d^†
                    U_current = (
                        self._U[
                            d - 3,
                            t,
                            backend.asarray(
                                point_shifted[p_idx.squeeze(1), 2], dtype=int
                            ),
                            backend.asarray(
                                point_shifted[p_idx.squeeze(1), 1], dtype=int
                            ),
                            backend.asarray(
                                point_shifted[p_idx.squeeze(1), 0], dtype=int
                            ),
                            :,
                            :,
                        ]
                        .conj()
                        .transpose(0, 2, 1)
                    )  # [Np, Nc, Nc]

                    if gauge_link_product is None:
                        gauge_link_product = U_current
                    else:
                        gauge_link_product = contract(
                            "pab,pbc->pac", gauge_link_product, U_current
                        )

                    point_shifted[:, d - 3] = (point_shifted[:, d - 3] - 1) % sizes[
                        d - 3
                    ]

            return gauge_link_product  # [Np, Nc, Nc]

    def calc_v2p(self, t: int):
        """
        Calculate v2p (eigenvector to point) elemental diagrams.

        Physics: V(x_p - disp) U_N^†...U_2^†U_1^†(x_p - disp -> x_p)

        Args:
            t: Time slice index

        Returns:
            Array shape [num_disp, num_momentum, usedNe, usedNp, Nc]
        """
        from ..insertion.gauge_link import GaugeLink
        from copy import deepcopy

        backend = get_backend()
        Lx, Ly, Lz, Lt = self.latt_size
        sizes = [Lx, Ly, Lz]

        if self.debug:
            print(f"\nCurrentElementalGenerator.calc_v2p() called:")
            print(f"  t = {t}")
            print(f"  self.Ne = {self.Ne}, self.usedNe = {self.usedNe}")
            print(f"  self.Np = {self.Np}, self.usedNp = {self.usedNp}")
            print(f"  num_disp = {self.num_disp}")

        # Initialize result array
        result = backend.zeros(
            (self.num_disp, self.num_momentum, self.usedNe, self.usedNp, Nc),
            dtype="<c16",
        )

        # Setup index arrays for vectorized operations
        e_idx = backend.arange(self.usedNe)[None, :, None]  # [1, usedNe, 1]
        p_idx = backend.arange(self.usedNp)[None, None, :]  # [1, 1, usedNp]

        # Loop over all gauge link displacements
        for disp_idx in range(self.num_disp):
            gauge_link = GaugeLink(disp_idx)
            gauge_list = gauge_link.gauge_list
            disp = gauge_link.displacement

            if self.debug:
                print(f"\n  Processing disp_idx {disp_idx}:")
                print(f"    displacement: {disp}")
                print(f"    gauge_list: {gauge_list}")

            # Apply displacement backwards: x_p -> x_p - disp
            point_shifted = deepcopy(self._point_data[: self.usedNp])
            for coord_idx in range(3):
                if disp[coord_idx] != 0:
                    point_shifted[:, :, coord_idx] = (
                        point_shifted[:, :, coord_idx] - disp[coord_idx]
                    ) % sizes[coord_idx]

            if self.debug:
                print(f"    point_shifted[0, {t}, :] = {point_shifted[0, t, :]}")

            # Extract V from eigenvector at displaced positions
            # V shape: [usedNe, usedNp, Nc]
            V = self._eigenvector_data[
                t,
                e_idx.squeeze(0),
                backend.asarray(point_shifted[p_idx.squeeze((0, 1)), t, 2], dtype=int),
                backend.asarray(point_shifted[p_idx.squeeze((0, 1)), t, 1], dtype=int),
                backend.asarray(point_shifted[p_idx.squeeze((0, 1)), t, 0], dtype=int),
                :,
            ]  # [usedNe, usedNp, Nc]

            if self.debug:
                print(f"    V shape: {V.shape}")
                print(f"    V[0, 0, :] = {V[0, 0, :]}")

            # Apply gauge links using internal method
            gauge_link_product = self._apply_gauge_links_to_points(
                point_shifted, gauge_list, t
            )

            # OLD CODE BELOW SHOULD BE DELETED - keeping temporarily for reference
            if False:  # Disabled old loop code
                if d < 3:  # Forward link: U_d
                    point_shifted[:, :, d] = (point_shifted[:, :, d] + 1) % sizes[d]
                    U_current = self._U[
                        d,
                        t,
                        backend.asarray(
                            point_shifted[p_idx.squeeze((0, 1)), t, 2], dtype=int
                        ),
                        backend.asarray(
                            point_shifted[p_idx.squeeze((0, 1)), t, 1], dtype=int
                        ),
                        backend.asarray(
                            point_shifted[p_idx.squeeze((0, 1)), t, 0], dtype=int
                        ),
                        :,
                        :,
                    ]  # [Np, Nc, Nc]

                    if gauge_link_product is None:
                        gauge_link_product = U_current
                    else:
                        gauge_link_product = contract(
                            "pab,pbc->pac", gauge_link_product, U_current
                        )

                else:  # Backward link: U_d^†
                    U_current = (
                        self._U[
                            d - 3,
                            t,
                            backend.asarray(
                                point_shifted[p_idx.squeeze((0, 1)), t, 2], dtype=int
                            ),
                            backend.asarray(
                                point_shifted[p_idx.squeeze((0, 1)), t, 1], dtype=int
                            ),
                            backend.asarray(
                                point_shifted[p_idx.squeeze((0, 1)), t, 0], dtype=int
                            ),
                            :,
                            :,
                        ]
                        .conj()
                        .transpose(0, 2, 1)
                    )  # [Np, Nc, Nc]

                    if gauge_link_product is None:
                        gauge_link_product = U_current
                    else:
                        gauge_link_product = contract(
                            "pab,pbc->pac", gauge_link_product, U_current
                        )

                    point_shifted[:, :, d - 3] = (
                        point_shifted[:, :, d - 3] - 1
                    ) % sizes[d - 3]
            # END of old disabled code

            # Apply gauge link product to V: U^† @ V
            if gauge_link_product is not None:
                V = contract("pab,epb->epa", gauge_link_product, V)

            # Store result (momentum-independent)
            result[disp_idx] = V

        return result

    def calc_p2v(self, t: int):
        """
        Calculate p2v (point to eigenvector) elemental diagrams.

        Physics: U_N...U_2 U_1(x_p) V(x_p + disp)

        Args:
            t: Time slice index

        Returns:
            Array shape [num_disp, num_momentum, usedNp, usedNe, Nc]
        """
        from ..insertion.gauge_link import GaugeLink
        from copy import deepcopy

        backend = get_backend()
        Lx, Ly, Lz, Lt = self.latt_size
        sizes = [Lx, Ly, Lz]

        if self.debug:
            print(f"\nCurrentElementalGenerator.calc_p2v() called:")
            print(f"  t = {t}")
            print(f"  self.usedNe = {self.usedNe}, self.usedNp = {self.usedNp}")

        result = backend.zeros(
            (self.num_disp, self.num_momentum, self.usedNp, self.usedNe, Nc),
            dtype="<c16",
        )

        e_idx = backend.arange(self.usedNe)[None, :, None]
        p_idx = backend.arange(self.usedNp)[None, None, :]

        for disp_idx in range(self.num_disp):
            gauge_link = GaugeLink(disp_idx)
            gauge_list = gauge_link.gauge_list
            disp = gauge_link.displacement

            if self.debug:
                print(f"\n  Processing disp_idx {disp_idx}:")
                print(f"    displacement: {disp}")
                print(f"    gauge_list: {gauge_list}")

            # Apply displacement forwards: x_p -> x_p + disp
            point_shifted = deepcopy(self._point_data[: self.usedNp])
            for coord_idx in range(3):
                if disp[coord_idx] != 0:
                    point_shifted[:, :, coord_idx] = (
                        point_shifted[:, :, coord_idx] + disp[coord_idx]
                    ) % sizes[coord_idx]

            if self.debug:
                print(f"    point_shifted[0, {t}, :] = {point_shifted[0, t, :]}")

            # Extract V at displaced position
            V = self._eigenvector_data[
                t,
                e_idx.squeeze(0),
                backend.asarray(point_shifted[p_idx.squeeze((0, 1)), t, 2], dtype=int),
                backend.asarray(point_shifted[p_idx.squeeze((0, 1)), t, 1], dtype=int),
                backend.asarray(point_shifted[p_idx.squeeze((0, 1)), t, 0], dtype=int),
                :,
            ]  # [usedNe, usedNp, Nc]

            if self.debug:
                print(f"    V shape: {V.shape}")
                print(f"    V[0, 0, :] = {V[0, 0, :]}")

            gauge_link_product = self._apply_gauge_links_to_points(
                self._point_data[: self.usedNp], gauge_list, t
            )

            if gauge_link_product is not None:
                gauge_link_product = gauge_link_product.transpose(0, 2, 1).conj()

            #     # Apply gauge links in REVERSE order: U_N ... U_2 U_1
            #     gauge_link_product = None
            #     for d in reversed(gauge_list):
            #         if d >= 3:  # Backward link becomes forward in reverse
            #             point_shifted[:, :, d - 3] = (
            #                 point_shifted[:, :, d - 3] + 1
            #             ) % sizes[d - 3]
            #             U_current = self._U[
            #                 d - 3,
            #                 t,
            #                 backend.asarray(
            #                     point_shifted[p_idx.squeeze((0, 1)), t, 2], dtype=int
            #                 ),
            #                 backend.asarray(
            #                     point_shifted[p_idx.squeeze((0, 1)), t, 1], dtype=int
            #                 ),
            #                 backend.asarray(
            #                     point_shifted[p_idx.squeeze((0, 1)), t, 0], dtype=int
            #                 ),
            #                 :,
            #                 :,
            #             ]

            #             if gauge_link_product is None:
            #                 gauge_link_product = U_current
            #             else:
            #                 gauge_link_product = contract(
            #                     "pab,pbc->pac", gauge_link_product, U_current
            #                 )

            #         else:  # Forward link becomes backward in reverse
            #             U_current = (
            #                 self._U[
            #                     d,
            #                     t,
            #                     backend.asarray(
            #                         point_shifted[p_idx.squeeze((0, 1)), t, 2], dtype=int
            #                     ),
            #                     backend.asarray(
            #                         point_shifted[p_idx.squeeze((0, 1)), t, 1], dtype=int
            #                     ),
            #                     backend.asarray(
            #                         point_shifted[p_idx.squeeze((0, 1)), t, 0], dtype=int
            #                     ),
            #                     :,
            #                     :,
            #                 ]
            #                 .conj()
            #                 .transpose(0, 2, 1)
            #             )

            #             if gauge_link_product is None:
            #                 gauge_link_product = U_current
            #             else:
            #                 gauge_link_product = contract(
            #                     "pab,pbc->pac", gauge_link_product, U_current
            #                 )

            #             point_shifted[:, :, d] = (point_shifted[:, :, d] - 1) % sizes[d]

            # Apply gauge link product to V
            if gauge_link_product is not None:
                V = contract("pab,epb->pea", gauge_link_product, V)

            # Store result (momentum-independent)
            result[disp_idx] = V

        return result

    def calc_p2p(self, t: int):
        """
        Calculate p2p (point to point) elemental diagrams (sparse).

        Physics: U_N...U_2 U_1 propagates from left point to right point
        Only non-zero when right = left + displacement

        Note: Result is independent of momentum, so momentum dimension is removed.

        Args:
            t: Time slice index

        Returns:
            List of dicts, one per disp_idx
            Each dict: {'type': 'identity' | 'sparse', 'indices': array, 'values': array}
        """
        from ..insertion.gauge_link import GaugeLink
        from copy import deepcopy

        backend = get_backend()
        Lx, Ly, Lz, Lt = self.latt_size
        sizes = [Lx, Ly, Lz]

        result = []

        for disp_idx in range(self.num_disp):
            gauge_link = GaugeLink(disp_idx)
            gauge_list = gauge_link.gauge_list
            disp = gauge_link.displacement

            # Special case: zero displacement -> identity
            if disp[0] == 0 and disp[1] == 0 and disp[2] == 0:
                result.append({"type": "identity"})
                continue

            # Find valid (l, r) pairs where r = l + disp
            point_left = self._point_data[:, t, :]  # [Np, 3]
            point_right = self._point_data[:, t, :]  # [Np, 3]

            expected_final = deepcopy(point_left)
            for coord_idx in range(3):
                if disp[coord_idx] != 0:
                    expected_final[:, coord_idx] = (
                        expected_final[:, coord_idx] + disp[coord_idx]
                    ) % sizes[coord_idx]

            # Build mask: [Np_left, Np_right]
            match_mask = None
            for coord_idx in range(3):
                left_coord = backend.asarray(expected_final[:, coord_idx], dtype=int)[
                    :, None
                ]
                right_coord = backend.asarray(point_right[:, coord_idx], dtype=int)[
                    None, :
                ]

                if match_mask is None:
                    match_mask = left_coord == right_coord
                else:
                    match_mask = match_mask & (left_coord == right_coord)

            # Extract valid pairs
            valid_indices = backend.where(match_mask)
            valid_l = valid_indices[0]
            valid_r = valid_indices[1]
            num_valid = len(valid_l)

            # Compute gauge links for valid pairs
            indices_list = []
            values_list = []

            for idx in range(num_valid):
                l = int(valid_l[idx])
                r = int(valid_r[idx])

                # Track position starting from left point
                point_shifted = point_left[l, :].copy()
                gauge_link_product = None

                for d in gauge_list:
                    if d < 3:  # Forward link
                        U_current = self._U[
                            d,
                            t,
                            int(point_shifted[2]),
                            int(point_shifted[1]),
                            int(point_shifted[0]),
                            :,
                            :,
                        ]  # [Nc, Nc]

                        if gauge_link_product is None:
                            gauge_link_product = U_current
                        else:
                            gauge_link_product = contract(
                                "ab,bc->ac", gauge_link_product, U_current
                            )

                        point_shifted[d] = (point_shifted[d] + 1) % sizes[d]

                    else:  # Backward link
                        point_shifted[d - 3] = (point_shifted[d - 3] - 1) % sizes[d - 3]

                        U_current = (
                            self._U[
                                d - 3,
                                t,
                                int(point_shifted[2]),
                                int(point_shifted[1]),
                                int(point_shifted[0]),
                                :,
                                :,
                            ]
                            .conj()
                            .T
                        )

                        if gauge_link_product is None:
                            gauge_link_product = U_current
                        else:
                            gauge_link_product = contract(
                                "ab,bc->ac", gauge_link_product, U_current
                            )

                # Store sparse entry
                indices_list.append([l, r])
                values_list.append(
                    gauge_link_product
                    if gauge_link_product is not None
                    else backend.eye(Nc)
                )

            # Convert to arrays
            indices = backend.asarray(indices_list, dtype="int32")  # [N, 2]
            values = backend.asarray(values_list, dtype="<c16")  # [N, 3, 3]

            # Store result (momentum-independent)
            result.append(
                {
                    "type": "sparse",
                    "indices": indices,
                    "values": values,
                }
            )

        return result

    def calc_v2v(self, t: int):
        """
        Calculate v2v (eigenvector to eigenvector) elemental diagrams.

        Uses gauge link product approach: first compute gauge_links_product on the field,
        then contract with eigenvectors. Matches ElementalGenerator.calc_disp functionality.

        Args:
            t: Time slice index

        Returns:
            Array shape [num_disp, num_momentum, usedNe, usedNe]
        """
        from ..insertion.gauge_link import GaugeLink

        backend = get_backend()
        Lx, Ly, Lz, Lt = self.latt_size

        if self.debug:
            print(f"\nCurrentElementalGenerator.calc_v2v() called:")
            print(f"  t = {t}")
            print(f"  self.Ne = {self.Ne}, self.usedNe = {self.usedNe}")
            print(f"  num_disp = {self.num_disp}")

        # Initialize result array
        result = backend.zeros(
            (self.num_disp, self.num_momentum, self.usedNe, self.usedNe), dtype="<c16"
        )

        # Get eigenvector data for this time slice
        V = self._eigenvector_data[t, : self.usedNe]  # [usedNe, Lz, Ly, Lx, Nc]

        if self.debug:
            print(f"  V shape: {V.shape}")
            print(f"  V[0, 0, 0, 0, :] = {V[0, 0, 0, 0, :]}")

        for disp_idx in range(self.num_disp):
            gauge_link = GaugeLink(disp_idx)
            gauge_list = gauge_link.gauge_list
            disp = gauge_link.displacement
            displacement = (
                (disp[0] / 2) * 2j * backend.pi / Lx,
                (disp[1] / 2) * 2j * backend.pi / Ly,
                (disp[2] / 2) * 2j * backend.pi / Lz,
            )
            shift_V = V
            for coord_idx in range(3):
                if disp[coord_idx] != 0:
                    shift_V = backend.roll(shift_V, -disp[coord_idx], 3 - coord_idx)

            if self.debug:
                print(f"\n  Processing disp_idx {disp_idx}:")
                print(f"    displacement: {disp}")
                print(f"    gauge_list: {gauge_list}")
                print(f"    actual displacement: {displacement}")

            # Compute gauge link product on entire field
            gauge_link_product = self._gauge_links_product(gauge_list, t=t)

            if gauge_link_product is None:
                # No gauge links (identity case)
                if self.debug:
                    print(f"    No gauge links, using identity")

                # Direct contraction without gauge links
                for momentum_idx, momentum in enumerate(self.momentum_list):
                    phase = self._momentum_phase.get(momentum)
                    result_val = contract(
                        "zyx,ezyxc,fzyxc->ef",
                        phase,
                        V.conj(),
                        V,
                    )
                    result[disp_idx, momentum_idx] = result_val

                    if self.debug:
                        print(
                            f"    momentum_idx {momentum_idx}, result[0,0] = {result_val[0,0]}"
                        )
            else:
                # gauge_link_product shape: [Lz, Ly, Lx, Nc, Nc]
                if self.debug:
                    print(f"    gauge_link_product shape: {gauge_link_product.shape}")
                    print(
                        f"    gauge_link_product[0, 0, 0, 0, 0] = {gauge_link_product[0, 0, 0, 0, 0]}"
                    )

                # Contract: <V_left | gauge_link_product | V_right>
                # V_left: [Ne, Lz, Ly, Lx, Nc]
                # gauge_link_product: [Lz, Ly, Lx, Nc, Nc]
                # V_right: [Ne, Lz, Ly, Lx, Nc]
                # Result: [Ne, Ne]
                for momentum_idx, momentum in enumerate(self.momentum_list):
                    phase = self._momentum_phase.get(momentum)
                    disp_phase = backend.exp(
                        sum([momentum[i] * displacement[i] for i in range(3)])
                    )
                    result_val = disp_phase * contract(
                        "zyx,ezyxa,zyxac,fzyxc->ef",
                        phase,
                        V.conj(),
                        gauge_link_product,
                        shift_V,
                    )
                    result[disp_idx, momentum_idx] = result_val

                    if self.debug:
                        print(f"    momentum_idx {momentum_idx}, momentum {momentum}")
                        print(f"    disp_phase: {disp_phase}")
                        print(f"    result[0,0] = {result_val[0,0]}")

        return result

    def calc_all(self, t: int):
        """
        Calculate v2v, v2p, p2v, p2p simultaneously, reusing gauge_link_product from v2v.

        This method computes v2v first (keeping the algorithm unchanged), then reuses
        the gauge_link_product computed for v2v to calculate v2p, p2v, and p2p efficiently.

        Args:
            t: Time slice index

        Returns:
            Dictionary with keys:
                - 'v2v': Array shape [num_disp, num_momentum, usedNe, usedNe]
                - 'v2p': Array shape [num_disp, usedNe, usedNp, Nc]
                - 'p2v': Array shape [num_disp, usedNp, usedNe, Nc]
                - 'p2p': List of dicts, one per disp_idx (same format as calc_p2p)
        """
        from ..insertion.gauge_link import GaugeLink
        from copy import deepcopy

        backend = get_backend()
        Lx, Ly, Lz, Lt = self.latt_size
        sizes = [Lx, Ly, Lz]

        if self.debug:
            print(f"\nCurrentElementalGenerator.calc_all() called:")
            print(f"  t = {t}")
            print(f"  self.Ne = {self.Ne}, self.usedNe = {self.usedNe}")
            print(f"  self.Np = {self.Np}, self.usedNp = {self.usedNp}")
            print(f"  num_disp = {self.num_disp}")

        # Initialize result arrays
        result_v2v = backend.zeros(
            (self.num_disp, self.num_momentum, self.usedNe, self.usedNe), dtype="<c16"
        )
        result_v2p = backend.zeros(
            (self.num_disp, self.usedNe, self.usedNp, Nc),
            dtype="<c16",
        )
        result_p2v = backend.zeros(
            (self.num_disp, self.usedNp, Nc, self.usedNe),
            dtype="<c16",
        )
        result_p2p = []

        # Setup index arrays for vectorized operations
        e_idx = backend.arange(self.usedNe)[None, :, None]  # [1, usedNe, 1]
        p_idx = backend.arange(self.usedNp)[None, None, :]  # [1, 1, usedNp]

        # Get eigenvector data for this time slice
        V = self._eigenvector_data[t, : self.usedNe]  # [usedNe, Lz, Ly, Lx, Nc]

        if self.debug:
            print(f"  V shape: {V.shape}")

        # Compute all elemental diagrams in a single loop
        for disp_idx in range(self.num_disp):
            gauge_link = GaugeLink(disp_idx)
            gauge_list = gauge_link.gauge_list
            disp = gauge_link.displacement

            if self.debug:
                print(f"\n  Processing disp_idx {disp_idx}:")
                print(f"    displacement: {disp}")
                print(f"    gauge_list: {gauge_list}")

            # Compute gauge link product once for this displacement
            gauge_link_product = self._gauge_links_product(gauge_list, t=t)

            # === Compute v2v ===
            displacement = (
                (disp[0] / 2) * 2j * backend.pi / Lx,
                (disp[1] / 2) * 2j * backend.pi / Ly,
                (disp[2] / 2) * 2j * backend.pi / Lz,
            )
            shift_V = V
            for coord_idx in range(3):
                if disp[coord_idx] != 0:
                    shift_V = backend.roll(shift_V, -disp[coord_idx], 3 - coord_idx)

            if gauge_link_product is None:
                # No gauge links (identity case)
                for momentum_idx, momentum in enumerate(self.momentum_list):
                    phase = self._momentum_phase.get(momentum)
                    result_val = contract(
                        "zyx,ezyxc,fzyxc->ef",
                        phase,
                        V.conj(),
                        V,
                    )
                    result_v2v[disp_idx, momentum_idx] = result_val
            else:
                # gauge_link_product shape: [Lz, Ly, Lx, Nc, Nc]
                for momentum_idx, momentum in enumerate(self.momentum_list):
                    phase = self._momentum_phase.get(momentum)
                    disp_phase = backend.exp(
                        sum([momentum[i] * displacement[i] for i in range(3)])
                    )
                    result_val = disp_phase * contract(
                        "zyx,ezyxa,zyxac,fzyxc->ef",
                        phase,
                        V.conj(),
                        gauge_link_product,
                        shift_V,
                    )
                    result_v2v[disp_idx, momentum_idx] = result_val

            # === Compute v2p ===
            # Apply displacement backwards: x_p -> x_p - disp
            point_shifted = deepcopy(self._point_data[: self.usedNp])
            for coord_idx in range(3):
                if disp[coord_idx] != 0:
                    point_shifted[:, :, coord_idx] = (
                        point_shifted[:, :, coord_idx] - disp[coord_idx]
                    ) % sizes[coord_idx]

            # Extract V from eigenvector at displaced positions
            V_at_points = self._eigenvector_data[
                t,
                e_idx.squeeze(0),
                backend.asarray(point_shifted[p_idx.squeeze((0, 1)), t, 2], dtype=int),
                backend.asarray(point_shifted[p_idx.squeeze((0, 1)), t, 1], dtype=int),
                backend.asarray(point_shifted[p_idx.squeeze((0, 1)), t, 0], dtype=int),
                :,
            ]  # [usedNe, usedNp, Nc]

            # Extract gauge_link_product at point positions from field gauge_link_product
            if gauge_link_product is not None:
                # Extract at point_shifted positions: [usedNp, Nc, Nc]
                gauge_link_at_points = gauge_link_product[
                    backend.asarray(
                        point_shifted[p_idx.squeeze((0, 1)), t, 2], dtype=int
                    ),
                    backend.asarray(
                        point_shifted[p_idx.squeeze((0, 1)), t, 1], dtype=int
                    ),
                    backend.asarray(
                        point_shifted[p_idx.squeeze((0, 1)), t, 0], dtype=int
                    ),
                    :,
                    :,
                ]  # [usedNp, Nc, Nc]

                # For v2p, we need reverse path: U_N^†...U_2^† U_1^†
                gauge_link_at_points = gauge_link_at_points.transpose(0, 2, 1).conj()

                # Apply gauge link product to V: U^† @ V
                result = contract("pab,epb->epa", gauge_link_at_points, V_at_points)
            else:
                result = V_at_points

            result_v2p[disp_idx] = result

            # === Compute p2v ===
            # Apply displacement forwards: x_p -> x_p + disp
            point_shifted = deepcopy(self._point_data[: self.usedNp])
            for coord_idx in range(3):
                if disp[coord_idx] != 0:
                    point_shifted[:, :, coord_idx] = (
                        point_shifted[:, :, coord_idx] + disp[coord_idx]
                    ) % sizes[coord_idx]

            # Extract V at displaced position
            V_at_points = self._eigenvector_data[
                t,
                e_idx.squeeze(0),
                backend.asarray(point_shifted[p_idx.squeeze((0, 1)), t, 2], dtype=int),
                backend.asarray(point_shifted[p_idx.squeeze((0, 1)), t, 1], dtype=int),
                backend.asarray(point_shifted[p_idx.squeeze((0, 1)), t, 0], dtype=int),
                :,
            ]  # [usedNe, usedNp, Nc]

            # Extract gauge_link_product at original point positions
            if gauge_link_product is not None:
                # Extract at original point positions: [usedNp, Nc, Nc]
                gauge_link_at_points = gauge_link_product[
                    backend.asarray(
                        self._point_data[p_idx.squeeze((0, 1)), t, 2], dtype=int
                    ),
                    backend.asarray(
                        self._point_data[p_idx.squeeze((0, 1)), t, 1], dtype=int
                    ),
                    backend.asarray(
                        self._point_data[p_idx.squeeze((0, 1)), t, 0], dtype=int
                    ),
                    :,
                    :,
                ]  # [usedNp, Nc, Nc]

                # For p2v, we need forward path: U_N...U_2 U_1
                # Apply to V: U @ V
                result = contract("pab,epb->pae", gauge_link_at_points, V_at_points)
            else:
                # Transpose V from [usedNe, usedNp, Nc] to [usedNp, Nc, usedNe]
                result = V_at_points.transpose(1, 2, 0)

            result_p2v[disp_idx] = result

            # === Compute p2p ===

            # Special case: zero displacement -> identity
            if disp[0] == 0 and disp[1] == 0 and disp[2] == 0:
                result_p2p.append({"type": "identity"})
                continue

            # Find valid (l, r) pairs where r = l + disp
            point_left = self._point_data[:, t, :]  # [Np, 3]
            point_right = self._point_data[:, t, :]  # [Np, 3]

            expected_final = deepcopy(point_left)
            for coord_idx in range(3):
                if disp[coord_idx] != 0:
                    expected_final[:, coord_idx] = (
                        expected_final[:, coord_idx] + disp[coord_idx]
                    ) % sizes[coord_idx]

            # Build mask: [Np_left, Np_right]
            match_mask = None
            for coord_idx in range(3):
                left_coord = backend.asarray(expected_final[:, coord_idx], dtype=int)[
                    :, None
                ]
                right_coord = backend.asarray(point_right[:, coord_idx], dtype=int)[
                    None, :
                ]

                if match_mask is None:
                    match_mask = left_coord == right_coord
                else:
                    match_mask = match_mask & (left_coord == right_coord)

            # Extract valid pairs
            valid_indices = backend.where(match_mask)
            valid_l = valid_indices[0]
            valid_r = valid_indices[1]
            num_valid = len(valid_l)

            if num_valid == 0:
                result_p2p.append(
                    {
                        "type": "sparse",
                        "indices": backend.zeros((0, 2), dtype="int32"),
                        "values": backend.zeros((0, Nc, Nc), dtype="<c16"),
                    }
                )
                continue

            # Extract gauge_link_product for valid pairs
            indices_list = []
            values_list = []

            for idx in range(num_valid):
                l = int(valid_l[idx])
                r = int(valid_r[idx])

                # Extract gauge_link_product at point_left[l] position
                if gauge_link_product is not None:
                    z, y, x = (
                        int(point_left[l, 2]),
                        int(point_left[l, 1]),
                        int(point_left[l, 0]),
                    )
                    gauge_link_val = gauge_link_product[z, y, x, :, :]  # [Nc, Nc]
                else:
                    gauge_link_val = backend.eye(Nc)

                indices_list.append([l, r])
                values_list.append(gauge_link_val)

            # Convert to arrays
            indices = backend.asarray(indices_list, dtype="int32")  # [N, 2]
            values = backend.asarray(values_list, dtype="<c16")  # [N, Nc, Nc]

            result_p2p.append(
                {
                    "type": "sparse",
                    "indices": indices,
                    "values": values,
                }
            )

        return {
            "v2v": result_v2v,
            "v2p": result_v2p,
            "p2v": result_p2v,
            "p2p": result_p2p,
        }

    @classmethod
    def _re_combine_dense(cls, array, insertion_list, axis, out=None):
        backend = get_backend()
        coeff = backend.zeros((len(insertion_list), array.shape[axis]))
        for i in range(len(insertion_list)):
            for j in range(len(insertion_list[i])):
                disp_idx = insertion_list[i][j][1]
                coeff[i, disp_idx] += insertion_list[i][j][0]
        contracted = backend.tensordot(coeff, array, axes=([1], [axis]))
        result = backend.moveaxis(contracted, 0, axis)
        if out is not None:
            out[...] = result
            return out
        return result

    @classmethod
    def _re_combine_sparse(cls, array, insertion_list, axis, out=None):
        backend = get_backend()
        axis = axis if axis >= 0 else array.ndim + axis
        num_row = len(insertion_list)
        num_col = array.shape[axis]

        rows = []
        cols = []
        vals = []
        for i in range(num_row):
            for coeff_val, disp_idx in insertion_list[i]:
                rows.append(i)
                cols.append(disp_idx)
                vals.append(coeff_val)

        if backend.__name__ == "cupy":
            from cupyx.scipy import sparse as sparse_backend

            rows = backend.asarray(rows, dtype=backend.int32).ravel()
            cols = backend.asarray(cols, dtype=backend.int32).ravel()
            vals = backend.asarray(vals, dtype=array.dtype).ravel()
        else:
            from scipy import sparse as sparse_backend
            import numpy as np

            rows = np.asarray(rows, dtype=np.int32).ravel()
            cols = np.asarray(cols, dtype=np.int32).ravel()
            vals = np.asarray(vals, dtype=array.dtype).ravel()

        coeff = sparse_backend.coo_matrix(
            (vals, (rows, cols)), shape=(num_row, num_col)
        )
        coeff = coeff.tocsr()

        moved = backend.moveaxis(array, axis, 0)
        moved_2d = moved.reshape((num_col, -1))
        contracted_2d = coeff.dot(moved_2d)
        contracted = contracted_2d.reshape((num_row,) + tuple(moved.shape[1:]))
        result = backend.moveaxis(contracted, 0, axis)
        if out is not None:
            out[...] = result
            return out
        return result

    @classmethod
    def re_combine_auto(cls, array, insertion_list, axis, out=None):
        call_idx = cls._recombine_benchmark_calls
        cls._recombine_benchmark_calls += 1

        if not cls._recombine_validated:
            t0 = perf_counter()
            dense_result = cls._re_combine_dense(array, insertion_list, axis)
            dense_elapsed = perf_counter() - t0

            t1 = perf_counter()
            sparse_result = cls._re_combine_sparse(array, insertion_list, axis)
            sparse_elapsed = perf_counter() - t1

            backend = get_backend()
            if hasattr(backend, "allclose"):
                ok = backend.allclose(
                    dense_result,
                    sparse_result,
                    rtol=cls._recombine_validate_rtol,
                    atol=cls._recombine_validate_atol,
                )
                ok = bool(ok)
            else:
                import numpy as np

                ok = np.allclose(
                    dense_result,
                    sparse_result,
                    rtol=cls._recombine_validate_rtol,
                    atol=cls._recombine_validate_atol,
                )

            if not ok:
                max_abs_diff = float(
                    backend.max(backend.abs(dense_result - sparse_result))
                )
                raise ValueError(
                    f"re_combine validation failed: dense vs sparse mismatch, max_abs_diff={max_abs_diff}"
                )

            cls._recombine_validated = True
            cls._recombine_elapsed_dense.append(dense_elapsed)
            cls._recombine_elapsed_sparse.append(sparse_elapsed)
            if out is not None:
                out[...] = dense_result
                return out
            return dense_result

        if cls._recombine_selected_method is None and call_idx < 6:
            method = "dense" if (call_idx % 2 == 0) else "sparse"
        else:
            if cls._recombine_selected_method is None:
                dense_n = min(3, len(cls._recombine_elapsed_dense))
                sparse_n = min(3, len(cls._recombine_elapsed_sparse))
                dense_avg = sum(cls._recombine_elapsed_dense[:dense_n]) / max(
                    dense_n, 1
                )
                sparse_avg = sum(cls._recombine_elapsed_sparse[:sparse_n]) / max(
                    sparse_n, 1
                )
                cls._recombine_selected_method = (
                    "sparse" if sparse_avg < dense_avg else "dense"
                )
            method = cls._recombine_selected_method

        t0 = perf_counter()
        if method == "sparse":
            result = cls._re_combine_sparse(array, insertion_list, axis, out=out)
        else:
            result = cls._re_combine_dense(array, insertion_list, axis, out=out)
        elapsed = perf_counter() - t0

        if call_idx < 6:
            if method == "sparse":
                cls._recombine_elapsed_sparse.append(elapsed)
            else:
                cls._recombine_elapsed_dense.append(elapsed)

            if (
                cls._recombine_selected_method is None
                and len(cls._recombine_elapsed_dense) >= 3
                and len(cls._recombine_elapsed_sparse) >= 3
            ):
                dense_avg = sum(cls._recombine_elapsed_dense[:3]) / 3
                sparse_avg = sum(cls._recombine_elapsed_sparse[:3]) / 3
                cls._recombine_selected_method = (
                    "sparse" if sparse_avg < dense_avg else "dense"
                )

        return result


def re_combine(array, insertion_list, axis, out=None):
    return ElementalGenerator.re_combine_auto(array, insertion_list, axis, out=out)
