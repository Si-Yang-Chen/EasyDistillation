"""F1.1: exceeding the label alphabet must say so, not raise from inside an index.

Each propagator in a connected contraction group consumes two label slots -- ``node``
and ``node + 1``, one per quark-line end -- so the 13-letter alphabet caps a group at
six lines, not thirteen.  A seven-line ring used to fail with a bare
``IndexError: string index out of range`` raised deep inside subscript construction,
which tells you nothing about the cause.
"""

import pytest

from lattice import set_backend
from lattice.quark_diagram import (
    MAX_PROPAGATORS_PER_GROUP,
    QuarkDiagram,
)


def _ring(vertices):
    """A single closed quark loop through ``vertices`` vertices."""
    matrix = [[0] * vertices for _ in range(vertices)]
    for index in range(vertices):
        matrix[index][(index + 1) % vertices] = index + 1
    return matrix


def test_the_cap_is_half_the_alphabet():
    """Two slots per propagator, so 13 letters mean 6 lines."""
    assert MAX_PROPAGATORS_PER_GROUP == 6


def test_at_the_cap_still_works():
    set_backend("numpy")
    diagram = QuarkDiagram(_ring(MAX_PROPAGATORS_PER_GROUP))
    assert diagram.subscripts, "a group at the cap must still contract"


@pytest.mark.parametrize("vertices", [7, 8, 12])
def test_beyond_the_cap_names_the_limit(vertices):
    """The error must state the limit and the cause, and not be an IndexError."""
    set_backend("numpy")
    with pytest.raises(ValueError, match="quark lines, but only 6 can be labelled"):
        QuarkDiagram(_ring(vertices))


def test_the_error_explains_why():
    """A bare limit is not enough; the message has to say what consumes the slots."""
    set_backend("numpy")
    with pytest.raises(ValueError, match="two .*slots"):
        QuarkDiagram(_ring(7))
