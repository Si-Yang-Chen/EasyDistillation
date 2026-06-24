"""Test meson spectrum calculations with CuPy backend."""

import os
import pytest

pytestmark = pytest.mark.gpu

from lattice import set_backend, get_backend

set_backend("cupy")
backend = get_backend()

momentum_dict = {
    0: "0 0 0", 1: "0 0 1", 2: "0 1 1",
    3: "1 1 1", 4: "0 0 2", 5: "0 1 2", 6: "1 1 2",
}

from lattice.insertion import Insertion, Operator, GammaName, DerivativeName, ProjectionName
from lattice import preset
from lattice.correlator.one_particle import twopoint, twopoint_matrix

test_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def meson_setup():
    """Set up meson operators and perambulator data."""
    latt_size = [4, 4, 4, 8]
    Lx, Ly, Lz, Lt = latt_size
    Ne = 20

    pi_A1 = Insertion(GammaName.PI, DerivativeName.IDEN, ProjectionName.A1, momentum_dict)
    op_pi = Operator("pi", [pi_A1[0](0, 0, 0)], [1])

    b1xnabla_A1 = Insertion(GammaName.B1, DerivativeName.NABLA, ProjectionName.A1, momentum_dict)
    op_pi2 = Operator("pi2", [pi_A1[0](0, 0, 0), b1xnabla_A1[0](0, 0, 0)], [3, 1])

    elemental = preset.ElementalNpy(
        f"{test_dir}/", ".elemental.npy", [13, 6, Lt, Ne, Ne], Ne
    )
    perambulator = preset.PerambulatorNpy(
        f"{test_dir}/", ".perambulator.npy", [Lt, Lt, 4, 4, Ne, Ne], Ne
    )

    cfg = "weak_field"
    e = elemental.load(cfg)
    p = perambulator.load(cfg)

    return {
        "op_pi": op_pi,
        "op_pi2": op_pi2,
        "e": e,
        "p": p,
        "Lt": Lt,
    }


class TestMesonSpectrum:
    """Test meson two-point correlation functions."""

    def test_twopoint_single(self, meson_setup):
        """Compute 2pt correlator for a single operator."""
        twopt = twopoint(
            [meson_setup["op_pi"]],
            meson_setup["e"],
            meson_setup["p"],
            list(range(meson_setup["Lt"])),
            meson_setup["Lt"],
        )
        twopt = twopt.real
        assert twopt.shape[0] == 1  # [Nop, Lt]
        assert twopt.shape[1] == meson_setup["Lt"]

    def test_twopoint_multi_op(self, meson_setup):
        """Compute 2pt for two operators."""
        twopt = twopoint(
            [meson_setup["op_pi"], meson_setup["op_pi2"]],
            meson_setup["e"],
            meson_setup["p"],
            list(range(meson_setup["Lt"])),
            meson_setup["Lt"],
        )
        twopt = twopt.real
        assert twopt.shape[0] == 2  # Two operators
        assert twopt.shape[1] == meson_setup["Lt"]

    def test_twopoint_matrix(self, meson_setup):
        """Compute 2x2 two-point correlation matrix."""
        twopt_matrix = twopoint_matrix(
            [meson_setup["op_pi"], meson_setup["op_pi2"]],
            meson_setup["e"],
            meson_setup["p"],
            list(range(meson_setup["Lt"])),
            meson_setup["Lt"],
        )
        twopt_matrix = twopt_matrix.real
        assert twopt_matrix.shape == (2, 2, meson_setup["Lt"])
