"""
Test script for current vertex contraction with all four propagator types.

This script can be run on cluster systems and supports debug output via command-line arguments.

Usage:
    python test_current_contraction.py [--debug] [--config CONFIG_KEY]
"""

import argparse

import pytest
import numpy as np

pytestmark = pytest.mark.integration
from lattice.quark_diagram import QuarkDiagram, compute_diagrams_multitime, compute_diagrams


def test_current_contraction_basic(debug=False):
    """Test basic current vertex contraction with different propagator types."""
    print("\n" + "=" * 80)
    print("Test: Basic Current Vertex Contraction")
    print("=" * 80)

    # Create a simple diagram: two vertices with one being a current
    # Adjacency matrix: 0 -> 1 and 1 -> 0 (bidirectional connection)
    adjacency_matrix = [[0, 1], [1, 0]]
    vertex_list = [0, 1]  # vertex[1] is a current (value != 0)

    # L and usedNp are needed for sampling weight calculation in expand_scenes()
    diagram = QuarkDiagram(
        adjacency_matrix, vertex_list=vertex_list, L=24, usedNp=216, debug=debug
    )

    # Check that we have generated expanded diagrams
    if hasattr(diagram, "expanded_diagrams") and diagram.expanded_diagrams:
        print(f"\nGenerated {len(diagram.expanded_diagrams)} expanded diagrams")
        for i, expanded in enumerate(diagram.expanded_diagrams):
            print(f"\n  Expanded diagram {i}:")
            print(f"    operands: {expanded.operands}")
            print(f"    subscripts: {expanded.subscripts}")
            if hasattr(expanded, "propagator_types"):
                print(f"    propagator_types: {expanded.propagator_types}")
    else:
        print("\nNo expanded diagrams found (this is expected if no current vertices)")

    print("\n[OK] Basic current vertex contraction test completed!")
    return True


def test_current_contraction_complex(debug=False):
    """Test a more complex graph with multiple current vertices and verify expansion and metadata."""
    print("\n" + "=" * 80)
    print("Test: Complex Current Vertex Contraction")
    print("=" * 80)

    # Use a branched graph with multiple currents but only 2 propagators per group to avoid label set overflow
    # Topology: 0 -> 1, 0 -> 2
    adjacency_matrix = [
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ]
    # Mark multiple current vertices (2 currents): 0, 1
    vertex_list = [1, 1, 0]

    # L and usedNp are needed for sampling weight calculation in expand_scenes()
    diagram = QuarkDiagram(
        adjacency_matrix, vertex_list=vertex_list, L=24, usedNp=216, debug=debug
    )

    if not (hasattr(diagram, "expanded_diagrams") and diagram.expanded_diagrams):
        print("No expanded diagrams found")
        return False

    expanded_count = len(diagram.expanded_diagrams)
    expected_count = 4 ** sum(1 for v in vertex_list if v != 0)
    print(f"\nGenerated {expanded_count} expanded diagrams (expected {expected_count})")
    if expanded_count != expected_count:
        print(f"[FAIL] Expanded diagram count mismatch: got {expanded_count}, expected {expected_count}")
        return False

    # Verify that all propagator types appear somewhere across expansions
    all_types = set()
    for expanded in diagram.expanded_diagrams:
        if hasattr(expanded, "propagator_types") and expanded.propagator_types:
            for types in expanded.propagator_types:
                all_types.update(types)
    print(f"Propagator types seen: {sorted(all_types)}")
    expected_types = {"VSV", "VSP", "PSV", "PSP"}
    if not all_types.issuperset(expected_types):
        missing = expected_types - all_types
        print(f"[FAIL] Missing propagator types in complex test: {missing}")
        return False

    # List ALL expanded diagrams with full details
    print("\nListing ALL expanded diagrams with full details:")
    for i, expanded in enumerate(diagram.expanded_diagrams):
        print("\n" + "-" * 76)
        print(f"Expanded diagram {i}:")
        # Basic containers
        print(f"  operands: {expanded.operands}")
        print(f"  subscripts: {expanded.subscripts}")
        if hasattr(expanded, "propagator_types"):
            print(f"  propagator_types: {expanded.propagator_types}")
        else:
            print("  propagator_types: [missing]")

        # Sanity and detailed breakdown per contraction group
        if not (hasattr(expanded, "operands") and expanded.operands):
            print("  [FAIL] No operands found for expanded diagram")
            return False
        if not (hasattr(expanded, "propagator_types") and expanded.propagator_types):
            print("  [FAIL] No propagator_types found for expanded diagram")
            return False

        for group_idx, (ops_group, subs) in enumerate(zip(expanded.operands, expanded.subscripts)):
            edges = ops_group[0]
            vertices = ops_group[1]
            num_edges = len(edges)
            num_vertices = len(vertices)
            types = expanded.propagator_types[group_idx]
            subs_parts = subs.split(",")

            print(f"  Group {group_idx}:")
            print(f"    edges ({num_edges}): {edges}")
            print(f"    vertices ({num_vertices}): {vertices}")
            print(f"    types: {types}")
            print(f"    subs ({len(subs_parts)} parts): {subs_parts}")

            # Per-part annotation: map each subscript part to role and length
            for idx_part, part in enumerate(subs_parts):
                role = "prop" if idx_part < num_edges else "vertex"
                extra = ""
                if role == "prop":
                    extra = f" type={types[idx_part]}"
                print(f"      [{idx_part}] {role}{extra}: '{part}' len={len(part)}")

            # Structural checks
            if len(types) != num_edges:
                print(f"  [FAIL] types count {len(types)} != num_edges {num_edges}")
                return False
            if len(subs_parts) != num_edges + num_vertices:
                print(f"  [FAIL] subs parts {len(subs_parts)} != edges+vertices {num_edges + num_vertices}")
                return False

    print("\n[OK] Complex current vertex contraction test completed!")
    return True


def test_propagator_types_storage(debug=False):
    """Test that propagator types are correctly stored in QuarkDiagram."""
    print("\n" + "=" * 80)
    print("Test: Propagator Types Storage")
    print("=" * 80)

    # Create diagram with current vertex
    adjacency_matrix = [[0, 1], [1, 0]]
    vertex_list = [0, 1]  # vertex[1] is a current

    # L and usedNp are needed for sampling weight calculation in expand_scenes()
    diagram = QuarkDiagram(
        adjacency_matrix, vertex_list=vertex_list, L=24, usedNp=216, debug=debug
    )

    if hasattr(diagram, "expanded_diagrams") and diagram.expanded_diagrams:
        for i, expanded in enumerate(diagram.expanded_diagrams):
            if hasattr(expanded, "propagator_types") and expanded.propagator_types:
                print(f"\nExpanded diagram {i} propagator types:")
                for j, types in enumerate(expanded.propagator_types):
                    print(f"  Contraction group {j}: {types}")
            else:
                print(f"\nExpanded diagram {i}: No propagator_types found")
                return False
    else:
        print("No expanded diagrams found")
        return False

    print("\n[OK] Propagator types storage test completed!")
    return True


def test_all_propagator_types(debug=False):
    """Test that all four propagator types (VSV, VSP, PSV, PSP) are generated."""
    print("\n" + "=" * 80)
    print("Test: All Propagator Types")
    print("=" * 80)

    # Create diagram with two current vertices to get all combinations including PSP
    # Three vertices: 0 -> 1 -> 2, where vertex[1] and vertex[2] are both current
    adjacency_matrix = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    vertex_list = [0, 1, 1]  # vertex[1] and vertex[2] are both current

    # L and usedNp are needed for sampling weight calculation in expand_scenes()
    diagram = QuarkDiagram(
        adjacency_matrix, vertex_list=vertex_list, L=24, usedNp=216, debug=debug
    )

    if hasattr(diagram, "expanded_diagrams") and diagram.expanded_diagrams:
        all_types = set()
        for expanded in diagram.expanded_diagrams:
            if hasattr(expanded, "propagator_types") and expanded.propagator_types:
                for types in expanded.propagator_types:
                    all_types.update(types)

        print(f"\nFound propagator types: {sorted(all_types)}")
        expected_types = {"VSV", "VSP", "PSV", "PSP"}
        if all_types.issuperset(expected_types):
            print(f"[OK] All expected types found: {expected_types}")
        else:
            missing = expected_types - all_types
            print(f"[FAIL] Missing types: {missing}")
            if debug:
                print("\nDetailed propagator types per diagram:")
                for i, expanded in enumerate(diagram.expanded_diagrams):
                    if hasattr(expanded, "propagator_types") and expanded.propagator_types:
                        print(f"  Diagram {i}: {expanded.propagator_types}")
            return False
    else:
        print("No expanded diagrams found")
        return False

    print("\n[OK] All propagator types test completed!")
    return True


def test_subscripts_shape_matching(debug=False):
    """Test that subscripts have correct lengths for all propagator types."""
    print("\n" + "=" * 80)
    print("Test: Subscripts Length Matching")
    print("=" * 80)

    # Create diagram with current vertex to get different propagator types
    adjacency_matrix = [[0, 1], [1, 0]]
    vertex_list = [0, 1]  # vertex[1] is current

    # L and usedNp are needed for sampling weight calculation in expand_scenes()
    diagram = QuarkDiagram(
        adjacency_matrix, vertex_list=vertex_list, L=24, usedNp=216, debug=debug
    )

    if not (hasattr(diagram, "expanded_diagrams") and diagram.expanded_diagrams):
        print("No expanded diagrams found")
        return False

    # Expected subscript lengths for different propagator types
    EXPECTED_LENGTHS = {
        "VSV": 4,  # [Ns, Ns, Ne, Ne]
        "VSP": 5,  # [Ns, Ns, Ne, Np, Nc]
        "PSV": 5,  # [Ns, Ns, Np, Ne, Nc]
        "PSP": 6,  # [Ns, Ns, Np, Nc, Np, Nc]
    }

    # Test each expanded diagram
    for i, expanded in enumerate(diagram.expanded_diagrams):
        print(f"\n--- Testing expanded diagram {i} ---")
        if not (hasattr(expanded, "propagator_types") and expanded.propagator_types):
            print("No propagator types found")
            return False

        for contraction_idx, (operands, subscripts) in enumerate(zip(expanded.operands, expanded.subscripts)):
            prop_types = expanded.propagator_types[contraction_idx]
            subscripts_split = subscripts.split(",")
            print(f"  Propagator types: {prop_types}")
            print(f"  Subscripts: {subscripts}")

            # Verify propagator subscript lengths
            all_lengths_correct = True
            for prop_idx, prop_type in enumerate(prop_types):
                subscript = subscripts_split[prop_idx]
                expected_len = EXPECTED_LENGTHS[prop_type]
                actual_len = len(subscript)

                status = "[OK]" if actual_len == expected_len else "[FAIL]"
                print(
                    f"    Propagator {prop_idx} ({prop_type}): subscript='{subscript}' len={actual_len}, expected={expected_len} {status}"
                )

                if actual_len != expected_len:
                    all_lengths_correct = False

            if not all_lengths_correct:
                print("    ERROR: Subscript length mismatch!")
                return False

    print("\n[OK] All propagator subscript lengths match expected values!")
    return True


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test current vertex contraction with all propagator types")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--config", type=str, help="Configuration key (for future use with real data)")
    parser.add_argument(
        "--test",
        type=str,
        choices=["basic", "complex", "types", "alltypes", "shapes", "all"],
        default="all",
        help="Which test to run (default: all)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Current Vertex Contraction Test Suite")
    print("=" * 80)
    print(f"Debug mode: {args.debug}")
    if args.config:
        print(f"Config key: {args.config}")

    results = []

    if args.test in ["basic", "all"]:
        try:
            results.append(("Basic Contraction", test_current_contraction_basic(args.debug)))
        except Exception as e:
            print(f"\n[FAIL] Basic contraction test failed: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()
            results.append(("Basic Contraction", False))

    if args.test in ["complex", "all"]:
        try:
            results.append(("Complex Contraction", test_current_contraction_complex(args.debug)))
        except Exception as e:
            print(f"\n[FAIL] Complex contraction test failed: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()
            results.append(("Complex Contraction", False))

    if args.test in ["types", "all"]:
        try:
            results.append(("Propagator Types Storage", test_propagator_types_storage(args.debug)))
        except Exception as e:
            print(f"\n[FAIL] Propagator types storage test failed: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()
            results.append(("Propagator Types Storage", False))

    if args.test in ["alltypes", "all"]:
        try:
            results.append(("All Propagator Types", test_all_propagator_types(args.debug)))
        except Exception as e:
            print(f"\n[FAIL] All propagator types test failed: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()
            results.append(("All Propagator Types", False))

    if args.test in ["shapes", "all"]:
        try:
            results.append(("Subscripts Shape Matching", test_subscripts_shape_matching(args.debug)))
        except Exception as e:
            print(f"\n[FAIL] Subscripts shape matching test failed: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()
            results.append(("Subscripts Shape Matching", False))

    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    for test_name, passed in results:
        status = "[OK] PASSED" if passed else "[FAIL] FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(passed for _, passed in results)
    print("\n" + "=" * 80)
    if all_passed:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    # sys.exit(main())
    test_current_contraction_complex(debug=True)
