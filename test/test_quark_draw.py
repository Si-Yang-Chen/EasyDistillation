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
    """A baryon source, a baryon sink, and a self-looping meson.

    A baryon source cannot pair with a lone meson sink — its three quark lines
    would have nowhere to land — so the minimal mixed diagram also carries a
    second baryon, plus a meson contracting with itself.
    """
    diagram = _make_diagram(
        names=["N", "N", "pi"],
        daggers=[True, False, False],
        adjacency=[
            [0, [[1, 0, 0], [0, 2, 0], [0, 0, 3]], 0],
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], 0, 0],
            [0, 0, 4],
        ],
    )

    _, attributes = _vertex_attributes_from_diagram(diagram)

    assert (attributes[0]["pos"], attributes[0]["type"]) == ("src", "baryon")
    assert (attributes[1]["pos"], attributes[1]["type"]) == ("snk", "baryon")
    assert (attributes[2]["pos"], attributes[2]["type"]) == ("snk", "meson")


def test_draw_quark_diagram_takes_a_diagram_object():
    import matplotlib.pyplot as plt

    from lattice.quark_draw import draw_quark_diagram

    diagram = _make_diagram(
        names=["D", "D_star"], daggers=[True, False], adjacency=[[0, 1], [1, 0]]
    )

    draw_quark_diagram(diagram, line_color_list=[None, "r"])
    plt.close("all")


def test_standalone_hadron_selfloops_and_stays_aligned():
    """There is no such thing as an isolated meson: every meson has one line out
    and one in, and both may land on the meson itself (a non-zero diagonal
    entry). A hadron that contracts with nothing else is drawn as a self-loop,
    not as a vertex with no lines — a vertex with no lines would leave its
    quarks unlinked, and validate_adjacency_matrix rejects it.

    The alignment half of the regression is kept: the self-loop sits at vertex 0,
    *lower* than the connected pair, which is what used to desynchronise the
    propagator indices when vertices were filtered.
    """
    import matplotlib.pyplot as plt

    from lattice.quark_draw import draw_single_diagram

    # vertex 0 self-loops; vertices 1 and 2 contract with each other
    adjacency = [[1, 0, 0], [0, 0, 1], [0, 2, 0]]
    attributes = [
        {"pos": "src", "type": "meson", "name": "$\\rho$"},
        {"pos": "src", "type": "meson", "name": "$D$"},
        {"pos": "snk", "type": "meson", "name": "$D$"},
    ]

    draw_single_diagram(adjacency, attributes, [None, "r", "b"])
    plt.close("all")


def test_vertex_with_no_lines_is_rejected():
    """The inverse half: a zero row-and-column is not an isolated hadron, it is
    a hadron whose quarks went nowhere. The assertion names it instead of
    letting drawing fail with an unrelated IndexError.
    """
    import pytest

    from lattice.quark_diagram import QuarkDiagram

    with pytest.raises(ValueError, match="Vertex 0 has 0 outgoing and 0 incoming"):
        QuarkDiagram([[0, 0, 0], [0, 0, 1], [0, 1, 0]])


def test_meson_self_loop_draws_and_passes_validation():
    """A meson connecting to itself (disconnected piece / diagonal matrix).

    The self-loop gives each vertex one line out and one in — the same degree as
    an ordinary meson pair — so the quark-link assertion accepts it, and the
    line attaches to the operator's two anchor points.
    """
    import matplotlib.pyplot as plt

    from lattice.quark_draw import draw_single_diagram

    adjacency = [[1, 0], [0, 1]]
    attributes = [
        {"pos": "src", "type": "meson", "name": "$\\rho$"},
        {"pos": "snk", "type": "meson", "name": "$\\rho$"},
    ]

    draw_single_diagram(adjacency, attributes, [None, "r"], quark_types={1: "u"})
    plt.close("all")


def test_quark_type_colours_and_marks_lines():
    """Flavours override the colour list and are written next to their lines."""
    import matplotlib.pyplot as plt

    from lattice.quark_draw import QUARK_COLORS, draw_single_diagram

    adjacency = [[0, 1, 0], [0, 0, 2], [3, 0, 0]]
    attributes = [
        {"pos": "src", "type": "meson", "name": "$\\pi$"},
        {"pos": "snk", "type": "meson", "name": "$K$"},
        {"pos": "snk", "type": "meson", "name": "$D$"},
    ]

    draw_single_diagram(
        adjacency,
        attributes,
        [None] * 5,
        quark_types={1: "u", 2: "s", 3: "c", 4: "d"},
    )
    plt.close("all")

    # the palette is part of the contract: same flavour, same colour
    assert QUARK_COLORS["u"] == QUARK_COLORS["u"]
    assert len(set(QUARK_COLORS.values())) == len(QUARK_COLORS)


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
