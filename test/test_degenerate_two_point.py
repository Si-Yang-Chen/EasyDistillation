"""F7.1: the equal-time two-point function is the degenerate case of this pipeline.

The design claim is that there is one contraction path, with a meson two-point function
as its degenerate member: no current vertex means one sector, no splitting vertex means
one offset class, and no point end means one scene at coefficient 1.  So the numbers
must not move at all -- not "agree to a tolerance", but be identical to what the same
call produced before the scene machinery existed.

This is the guard that catches a special case introduced for currents.  Any branch that
changes only currents would leave it passing; any branch that leaks into the common path
fails it loudly.

The reference values below were measured on the weak_field dataset before the scene
refactor and recorded to six decimals, so this compares at that precision rather than
bit-for-bit.  Claiming bit-identity would need the pre-change values at full precision,
which was not captured; the degeneracy checks on the skeleton (one scene, unit
coefficient) are the structural half of the same guard.
"""

import numpy as np
import pytest

from lattice import set_backend

pytestmark = pytest.mark.integration

# Measured on test/weak_field data (Lt=8, Ne=20) before the scene refactor.
REFERENCE = np.array(
    [
        36.561016,
        12.408518,
        8.692614,
        6.330026,
        5.720286,
        6.329110,
        8.344356,
        12.999925,
    ]
)


def test_equal_time_pion_two_point_is_unchanged():
    """One scene, unit coefficient, and the same value as before the refactor."""
    set_backend("numpy")

    from lattice import preset
    from lattice.base_types import Tag
    from lattice.flavor_structure import HadronFlavorStructure
    from lattice.group_projection import operator_transform
    from lattice.hadron import Hadron, gen_correlator
    from lattice.insertion import (
        DerivativeName,
        GammaName,
        Insertion,
        Operator,
        ProjectionName,
    )
    from lattice.insertion.mom_dict import momDict_mom1
    from lattice.quark_diagram import (
        Meson,
        Propagator,
        PropagatorLocal,
        calc_diagram_bind,
        calc_diagram_eval,
        calc_diagram_prepare,
    )
    from lattice.spatial_structure import HadronIrrep

    lt, ne = 8, 20
    elemental = preset.ElementalNpy("test/", ".elemental.npy", [13, 6, lt, ne, ne], ne)
    perambulator = preset.PerambulatorNpy(
        "test/", ".perambulator.npy", [lt, lt, 4, 4, ne, ne], ne
    )

    row = HadronIrrep("pi", [0, 0, 0], "A_1", -1, Tag(0, 0))[0]
    expression = gen_correlator(
        [[Hadron(row, HadronFlavorStructure("ud"))]] * 2, [0, 1], [False, True]
    )
    propagator_map = {
        "S^q": Propagator(perambulator, lt),
        r"S^q_\mathrm{local}": PropagatorLocal(perambulator, lt),
    }

    prepared = calc_diagram_prepare(
        [operator_transform(expression, "iden")], propagator_map=propagator_map
    )

    # The degeneracy must be visible in the skeleton, not just in the numbers.
    assert len(prepared.combined_diagrams) == 1, "a meson two-point must be one scene"
    assert [float(value) for value in prepared.scene_coefficients] == [1.0], (
        "a point-free graph must carry unit coefficient"
    )

    insertion = Insertion(
        GammaName.PI, DerivativeName.IDEN, ProjectionName.A1, momDict_mom1
    )

    def vertex_map(vertex):
        meson = Meson(
            elemental,
            Operator(vertex.hadron_name, [insertion[0](0, 0, 0)], [1.0]),
            vertex.dagger,
        )
        meson.load("weak_field", ne)
        return meson

    for handle in propagator_map.values():
        handle.load("weak_field", ne)
    calc_diagram_bind(prepared, vertex_map)

    result = calc_diagram_eval(prepared, {0: 0, 1: np.arange(lt)})
    measured = np.asarray(result[0][0, 0])

    assert measured.shape == (lt,)
    # The reference is only known to six decimals, so the tolerance reflects that and
    # nothing more; the point is that the number did not move, not that it moved a
    # little.
    np.testing.assert_allclose(measured.real, REFERENCE, rtol=0, atol=5e-7)
    np.testing.assert_allclose(measured.imag, np.zeros(lt), rtol=0, atol=1e-12)
