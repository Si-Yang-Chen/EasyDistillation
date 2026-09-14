"""Tests for inferring vertex type (meson vs baryon) from an adjacency matrix.

This is the fragile piece of the drawing path: ``quark_contract`` decides each
``matrix[i][j]`` shape from the baryon numbers of the two vertices, and the
drawing code has to read that shape back. It is pure data logic, so it lives in
``lattice.quark_diagram`` and these tests need no optional dependency — they run
in CI, which is the point: if the construction order below ever changes, a test
fails rather than the drawing silently picking the wrong vertex shape.

``quark_contract`` builds each entry as ``[[0] * N for _ in range(M)]``, i.e.
shape ``(M, N)`` where the *column* dimension ``N`` belongs to vertex ``i`` and
the row dimension ``M`` belongs to vertex ``j``:

    i=baryon, j=baryon -> (3, 3)
    i=baryon, j=meson  -> (1, 3)
    i=meson,  j=baryon -> (3, 1)
    i=meson,  j=meson  -> scalar

So "is baryon" means "this vertex's row has an entry with a three-wide column
dimension". Reading the row dimension instead misclassifies every meson in a
mixed diagram.
"""

import pytest

from lattice.quark_diagram import vertex_type_from_matrix


def adjacency(types):
    """Build an adjacency matrix exactly the way quark_contract does."""
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


@pytest.mark.parametrize(
    "types",
    [
        ["meson", "meson"],
        ["baryon", "baryon"],
        ["baryon", "meson"],
        ["meson", "baryon"],
        ["baryon", "meson", "meson"],
        ["meson", "baryon", "meson"],
        ["meson", "meson", "baryon"],
        ["baryon", "baryon", "meson"],
    ],
)
def test_vertex_type_matches_construction(types):
    matrix = adjacency(types)
    assert [vertex_type_from_matrix(matrix, i) for i in range(len(types))] == types


def test_meson_diagram_is_all_scalars():
    matrix = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
    assert [vertex_type_from_matrix(matrix, i) for i in range(3)] == ["meson"] * 3


def test_baryon_is_read_from_the_column_not_the_row():
    """The regression this guards.

    For i=baryon, j=meson the entry is (1, 3) — one row, three columns. The
    column dimension belongs to the row's vertex, so vertex 0 is the baryon.
    """
    matrix = adjacency(["baryon", "meson"])
    assert len(matrix[0][1]) == 1
    assert len(matrix[0][1][0]) == 3
    assert vertex_type_from_matrix(matrix, 0) == "baryon"
    assert vertex_type_from_matrix(matrix, 1) == "meson"


def test_empty_and_scalar_rows_are_mesons():
    assert vertex_type_from_matrix([[0]], 0) == "meson"
    assert vertex_type_from_matrix([[0, 0], [0, 0]], 1) == "meson"
