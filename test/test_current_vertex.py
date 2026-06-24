"""
Unit tests for QuarkDiagram with current vertex support.

Tests the expansion of diagrams with current vertices into multiple diagrams
with different propagator types (VSV, PSV, VSP, PSP).
"""

import numpy as np
import sys

from lattice.quark_diagram import QuarkDiagram


def test_single_current_vertex():
    """Test diagram expansion with a single current vertex."""
    # Create a simple diagram: two vertices with one being a current
    # Adjacency matrix: 0 -> 1 and 1 -> 0 (bidirectional connection)
    adjacency_matrix = [[0, 1], [1, 0]]
    vertex_list = [0, 1]  # vertex[1] is a current (value != 0)

    # Create diagram with debug output
    # L and usedNp are needed for sampling weight calculation in expand_scenes()
    diagram = QuarkDiagram(
        adjacency_matrix, vertex_list=vertex_list, L=24, usedNp=216, debug=True
    )

    # Check that we have generated multiple expanded diagrams
    print(f"\nGenerated {len(diagram.expanded_diagrams)} expanded diagrams")

    # For a single current vertex appearing as both source and sink,
    # we should get 4 combinations: (left=v, right=v), (left=v, right=p),
    # (left=p, right=v), (left=p, right=p)
    expected_diagrams = 4
    assert (
        len(diagram.expanded_diagrams) == expected_diagrams
    ), f"Expected {expected_diagrams} expanded diagrams, got {len(diagram.expanded_diagrams)}"

    # Each expanded diagram should have 1 contraction group (since the graph is connected)
    print("\nExpanded diagrams details:")
    for i, expanded in enumerate(diagram.expanded_diagrams):
        print(f"\n  Expanded diagram {i}:")
        print(f"    operands: {expanded.operands}")
        print(f"    subscripts: {expanded.subscripts}")
        assert (
            len(expanded.operands) == 1
        ), f"Each expanded diagram should have 1 operand group, got {len(expanded.operands)}"

    print("\n[OK] Single current vertex test passed!")


def test_no_current_vertex():
    """Test that diagrams without current vertices use standard v2v analysis."""
    # Create a simple diagram with no current vertices
    adjacency_matrix = [[0, 1], [1, 0]]
    vertex_list = [0, 0]  # Both vertices are normal (value = 0)

    diagram = QuarkDiagram(adjacency_matrix, vertex_list=vertex_list, debug=True)

    # Should generate only 1 contraction pattern (standard v2v)
    print(f"\nGenerated {len(diagram.operands)} operand groups")
    assert len(diagram.operands) == 1, f"Expected 1 operand group for v2v only, got {len(diagram.operands)}"

    print("\n[OK] No current vertex test passed!")


def test_two_current_vertices():
    """Test diagram expansion with two current vertices."""
    # Create a diagram: three vertices, two are currents
    # 0 -> 1 -> 2 -> 0 (cycle, so all vertices get both left and right ends set)
    adjacency_matrix = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    vertex_list = [0, 1, 2]  # vertex[1] and vertex[2] are currents

    # L and usedNp are needed for sampling weight calculation in expand_scenes()
    diagram = QuarkDiagram(
        adjacency_matrix, vertex_list=vertex_list, L=24, usedNp=216, debug=True
    )

    print(f"\nGenerated {len(diagram.expanded_diagrams)} expanded diagrams")

    # vertex[1] is a current: can be (left=v,right=v), (left=v,right=p), (left=p,right=v), (left=p,right=p) = 4 choices
    # vertex[2] is a current: can be (left=v,right=v), (left=v,right=p), (left=p,right=v), (left=p,right=p) = 4 choices
    # Total combinations: 4^2 = 16
    expected_diagrams = 16
    assert (
        len(diagram.expanded_diagrams) == expected_diagrams
    ), f"Expected {expected_diagrams} expanded diagrams, got {len(diagram.expanded_diagrams)}"

    # Each expanded diagram should have 1 contraction group
    print("\nExpanded diagrams details (showing first 4 and last 4):")
    # Show first 4
    for i in range(min(4, len(diagram.expanded_diagrams))):
        expanded = diagram.expanded_diagrams[i]
        print(f"\n  Expanded diagram {i}:")
        print(f"    operands: {expanded.operands}")
        print(f"    subscripts: {expanded.subscripts}")
        assert (
            len(expanded.operands) == 1
        ), f"Expanded diagram {i} should have 1 operand group, got {len(expanded.operands)}"

    # Show ellipsis if there are more than 8
    if len(diagram.expanded_diagrams) > 8:
        print(f"\n  ... ({len(diagram.expanded_diagrams) - 8} more diagrams) ...")

    # Show last 4
    if len(diagram.expanded_diagrams) > 4:
        for i in range(max(4, len(diagram.expanded_diagrams) - 4), len(diagram.expanded_diagrams)):
            expanded = diagram.expanded_diagrams[i]
            print(f"\n  Expanded diagram {i}:")
            print(f"    operands: {expanded.operands}")
            print(f"    subscripts: {expanded.subscripts}")
            assert (
                len(expanded.operands) == 1
            ), f"Expanded diagram {i} should have 1 operand group, got {len(expanded.operands)}"

    print("\n[OK] Two current vertices test passed!")


def test_diagram_without_vertex_list():
    """Test that diagrams without vertex_list use standard v2v analysis."""
    adjacency_matrix = [[0, 1], [1, 0]]

    diagram = QuarkDiagram(adjacency_matrix, vertex_list=None, debug=True)

    # Should use standard v2v analysis
    print(f"\nGenerated {len(diagram.operands)} operand groups")
    assert len(diagram.operands) == 1, f"Expected 1 operand group for v2v only, got {len(diagram.operands)}"

    print("\n[OK] No vertex list test passed!")


if __name__ == "__main__":
    print("=" * 80)
    print("Testing QuarkDiagram with Current Vertex Support")
    print("=" * 80)

    try:
        test_diagram_without_vertex_list()
        print()
        test_no_current_vertex()
        print()
        test_single_current_vertex()
        print()
        test_two_current_vertices()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
