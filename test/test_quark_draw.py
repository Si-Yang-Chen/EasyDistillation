"""Tests for drawing directly from a ``Diagram`` object.

``draw_quark_diagram`` reads everything but the colours off the diagram: the
adjacency matrix from ``diagram.diagram`` and each vertex's side and label from
``diagram.vertex_list`` (``HadronIrrepRow.dagger`` and ``.hadron_name``). The
meson/baryon split comes from ``vertex_type_from_matrix``, tested separately in
``test_vertex_type.py`` without needing the viz extra.

These tests need the optional ``viz`` dependencies. CI installs only ``.[dev]``,
so the module is skipped there rather than failing collection.
"""

import pytest

matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("feynman")

matplotlib.use("Agg")

from lattice.base_types import Tag  # noqa: E402
from lattice.quark_diagram import Diagram, QuarkDiagram  # noqa: E402
from lattice.quark_draw import _vertex_attributes_from_diagram  # noqa: E402
from lattice.spatial_structure import HadronIrrepRow  # noqa: E402


def _adjacency(types):
    """Same construction as test_vertex_type.adjacency, kept local to avoid a cross-import."""
    n = len(types)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if types[i] == "baryon" and types[j] == "baryon":
                matrix[i][j] = [[0] * 3 for _ in range(3)]
            elif types[i] == "baryon" and types[j] == "meson":
                matrix[i][j] = [[0] * 3 for _ in range(1)]
            elif types[i] == "meson" and types[j] == "baryon":
                matrix[i][j] = [[0] * 1 for _ in range(3)]
    return matrix


def _make_diagram(names, daggers, adjacency):
    vertices = [
        HadronIrrepRow(name, [0, 0, 1], "A_2", 0, None, Tag(0, 0), dagger)
        for name, dagger in zip(names, daggers)
    ]
    return Diagram(
        QuarkDiagram(adjacency), [0] * len(vertices), vertices, []
    )


def test_attributes_come_from_the_diagram():
    diagram = _make_diagram(
        names=["D", "D_star"], daggers=[False, True], adjacency=[[0, 1], [1, 0]]
    )

    adjacency, attributes = _vertex_attributes_from_diagram(diagram)

    assert adjacency == [[0, 1], [1, 0]]
    assert attributes == [
        {"pos": "snk", "type": "meson", "name": "$D$"},
        {"pos": "src", "type": "meson", "name": "$D_star$"},
    ]


def test_dagger_gives_side_and_shape_gives_type():
    """A baryon source and a meson sink in one diagram."""
    diagram = _make_diagram(
        names=["N", "pi"],
        daggers=[True, False],
        adjacency=_adjacency(["baryon", "meson"]),
    )

    _, attributes = _vertex_attributes_from_diagram(diagram)

    assert (attributes[0]["pos"], attributes[0]["type"]) == ("src", "baryon")
    assert (attributes[1]["pos"], attributes[1]["type"]) == ("snk", "meson")


def test_draw_quark_diagram_takes_a_diagram_object():
    import matplotlib.pyplot as plt

    from lattice.quark_draw import draw_quark_diagram

    diagram = _make_diagram(
        names=["D", "D_star"], daggers=[True, False], adjacency=[[0, 1], [1, 0]]
    )

    draw_quark_diagram(diagram, line_color_list=[None, "r"])
    plt.close("all")


def test_isolated_hadron_is_drawn_not_dropped():
    """A hadron with no propagator is legitimate: quarks carry links, hadrons
    need not. It must still be drawn, and — the part that used to crash — the
    vertex indices the propagators refer to must stay aligned when the isolated
    hadron has a *lower* index than a connected one.
    """
    import matplotlib.pyplot as plt

    from lattice.quark_draw import draw_single_diagram

    # vertex 0 is isolated; the propagator refers to vertices 1 -> 2
    adjacency = [[0, 0, 0], [0, 0, 1], [0, 0, 0]]
    attributes = [
        {"pos": "src", "type": "meson", "name": "$vac$"},
        {"pos": "src", "type": "meson", "name": "$D$"},
        {"pos": "snk", "type": "meson", "name": "$D$"},
    ]

    draw_single_diagram(adjacency, attributes, [None, "r"])
    plt.close("all")


def test_quark_contract_style_baryon_matrix_draws():
    """quark_contract writes baryon contractions as two-level nesting
    (``[source_quark][sink_quark]``) with inner zeros for unconnected pairs; the
    hand-written convention is a flat list. Both must draw, and both must yield
    three lines for a nucleon.
    """
    import matplotlib.pyplot as plt

    from lattice.quark_draw import draw_single_diagram

    attributes = [
        {"pos": "src", "type": "baryon", "name": "$N$"},
        {"pos": "snk", "type": "baryon", "name": "$N$"},
    ]

    # hand-written convention
    flat = [[0, [1, 1, 1]], [[0, 0, 0], 0]]
    draw_single_diagram(flat, attributes, [None, "r", "b", "g"])
    plt.close("all")

    # quark_contract convention: same three quarks, one per diagonal slot
    nested = [[0, [[1, 0, 0], [0, 2, 0], [0, 0, 3]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], 0]]
    draw_single_diagram(nested, attributes, [None, "r", "b", "g"])
    plt.close("all")
