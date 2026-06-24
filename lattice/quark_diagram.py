from copy import copy, deepcopy
import gc
from itertools import product
import logging
import os
from time import perf_counter
from typing import Callable, Dict, List, Union, Any, Tuple, Optional

import numpy as np
from opt_einsum import contract
import sympy as sp
from sympy.core.numbers import Integer
from sympy import Add, Mul, Pow, S, simplify
import hashlib
from math import comb as math_comb

from lattice.constant import Nd, Nc
from .preset import (
    Eigenvector,
    PointSource,
    GaugeField,
    Perambulator,
    PropagatorVSP,
    PropagatorPSV,
    PropagatorPSP,
    OverlapMatrix,
    CurrentElementalV2P,
    CurrentElementalP2V,
    CurrentElementalP2P,
)
from lattice.spatial_structure import HadronIrrepRow
from .base_types import Tag

from .backend import get_backend, log_gpu_memory

logger = logging.getLogger(__name__)

# Fixed label sets for Einstein summation (opt_einsum)
_SUB_VECTOR = "NOPQRSTUVWXYZ"  # Eigenvector/color indices
_SUB_SPIN = "ABCDEFGHIJKLM"  # Spin indices
_SUB_POINT = "nopqrstuvwxyz"  # Point indices
_SUB_COLOR = "abcdefghijklm"  # Color indices


def integer_partitions(n: int) -> List[List[int]]:
    """
    Generate all integer partitions of n.

    Each partition is a list of positive integers that sum to n, sorted in descending order.

    Args:
        n: Positive integer to partition

    Returns:
        List of partitions, e.g., integer_partitions(3) = [[3], [2,1], [1,1,1]]

    Example:
        integer_partitions(4) = [[4], [3,1], [2,2], [2,1,1], [1,1,1,1]]
    """
    if n == 0:
        return [[]]
    if n == 1:
        return [[1]]

    partitions = []
    for i in range(1, n + 1):
        for partition in integer_partitions(n - i):
            if not partition or i >= partition[0]:
                partitions.append([i] + partition)
    return partitions


def calculate_sampling_weight(L: int, usedNp: int, k: int) -> float:
    """
    Calculate sampling weight for k distinct points from a sampling set of size usedNp drawn from L^3 total points.

    Weight formula: C(L^3,k) / C(usedNp,k) = L^3(L^3-1)...(L^3-k+1) / [usedNp(usedNp-1)...(usedNp-k+1)]

    Args:
        L: Spatial lattice size (total points = L^3)
        usedNp: Number of points in sampling set
        k: Number of distinct points needed

    Returns:
        Sampling weight (compensation factor)

    Example:
        calculate_sampling_weight(10, 10, 2) = 1000*999/(10*9) = 110*11 = 1210
    """
    M = L**3  # Total number of lattice points
    N = usedNp  # Number of sampled points

    if k == 0:
        return 1.0
    if k > min(M, N):
        return 0.0

    # Use descending factorial: M(M-1)...(M-k+1) / N(N-1)...(N-k+1)
    numerator = 1.0
    denominator = 1.0
    for i in range(k):
        numerator *= M - i
        denominator *= N - i

    return numerator / denominator


def enumerate_point_scenes(
    r: int, L: int, usedNp: int
) -> List[Tuple[List[int], float]]:
    """
    Enumerate all point coincidence scenes for r point positions in a sampling group.

    For r point positions, generate all possible coincidence patterns (partitions),
    where each pattern specifies which positions share the same point value.

    Args:
        r: Number of point positions
        L: Spatial lattice size (total points = L^3)
        usedNp: Number of points in sampling set

    Returns:
        List of (partition, weight) tuples, where:
        - partition: List describing grouping, e.g., [2, 1] means "2 positions share one point, 1 position has another"
        - weight: Sampling weight for this scene, w(k) = C(L^3,k)/C(usedNp,k) where k = len(partition) = number of distinct points

    Example:
        enumerate_point_scenes(2, 10, 10) returns:
        [([1, 1], 999.0/9),   # 2 points distinct, weight = L^3(L^3-1)/usedNp(usedNp-1)
         ([2], 1000.0/10)]    # 2 points same, weight = L^3/usedNp

        enumerate_point_scenes(3, 10, 10) returns:
        [([1, 1, 1], ...),  # All 3 distinct
         ([2, 1], ...),      # 2 same, 1 different
         ([3], ...)]         # All 3 same
    """
    partitions = integer_partitions(r)
    scenes = []

    for partition in partitions:
        k = len(partition)  # Number of distinct points
        weight = calculate_sampling_weight(L, usedNp, k)
        scenes.append((partition, weight))

    return scenes


def partition_to_constraints(
    partition: List[int], point_positions: List[Tuple[int, str]]
):
    """
    Convert a partition into constraints for unify_vertex_point_color_indices.

    Args:
        partition: Integer partition describing grouping, e.g., [2, 1] means first 2 positions share a point
        point_positions: List of (vertex_idx, "left"|"right") specifying which vertices need point constraints

    Returns:
        Constraints list where constraints[vertex_idx] = (left_point_id, right_point_id)
        - None means no constraint (eigenvector position)
        - Same number means same point

    Example:
        partition = [2, 1]  # First 2 positions same, 3rd different
        point_positions = [(1, "left"), (1, "right"), (2, "left")]
        Returns constraints where vertex 1's left/right use same point_id, vertex 2's left uses different point_id
    """
    # Find max vertex index
    max_vertex_idx = max(pos[0] for pos in point_positions) if point_positions else 0
    constraints = [(None, None)] * (max_vertex_idx + 1)

    # Assign point IDs according to partition
    point_id_counter = 0
    position_idx = 0

    for group_size in partition:
        # All positions in this group get the same point ID
        group_point_id = point_id_counter

        for _ in range(group_size):
            if position_idx < len(point_positions):
                vertex_idx, side = point_positions[position_idx]
                left_id, right_id = constraints[vertex_idx]

                if side == "left":
                    constraints[vertex_idx] = (group_point_id, right_id)
                else:  # side == "right"
                    constraints[vertex_idx] = (left_id, group_point_id)

                position_idx += 1

        point_id_counter += 1

    return constraints


class QuarkDiagramOriginal:
    def __init__(self, adjacency_matrix) -> None:
        self.adjacency_matrix = adjacency_matrix
        self.operands = []
        self.subscripts = []
        self.operands_data = []
        self.analyse()

    def analyse(self) -> None:
        from copy import deepcopy

        adjacency_matrix = deepcopy(self.adjacency_matrix)
        num_vertex = len(adjacency_matrix)
        visited = [False] * num_vertex
        for idx in range(num_vertex):
            if visited[idx]:
                continue
            propagators = []
            visited[idx] = True
            queue = [idx]
            while queue != []:
                i = queue.pop(0)
                for j in range(num_vertex):
                    path = adjacency_matrix[i][j]
                    if path != 0:
                        adjacency_matrix[i][j] = 0
                        if not visited[j]:
                            visited[j] = True
                            queue.append(j)
                        if isinstance(path, int):
                            propagators.append([path, i, j])
                        elif isinstance(path, list):
                            for _path in path:
                                propagators.append([_path, i, j])
                        else:
                            raise ValueError(
                                f"Invalid value {path} in the adjacency matrix"
                            )
            if propagators == []:
                continue
            vertex_operands = []
            vertex_subscripts = []
            propagator_operands = []
            propagator_subscripts = []
            node = 0
            for propagator in propagators:
                propagator_operands.append(propagator)
                propagator_subscripts.append(
                    _SUB_SPIN[node + 1]
                    + _SUB_VECTOR[node + 1]
                    + _SUB_SPIN[node]
                    + _SUB_VECTOR[node]
                )
                if propagator[1] not in vertex_operands:
                    vertex_operands.append(propagator[1])
                    vertex_subscripts.append(_SUB_SPIN[node] + _SUB_VECTOR[node])
                else:
                    i = vertex_operands.index(propagator[1])
                    vertex_subscripts[i] = (
                        _SUB_SPIN[node] + _SUB_VECTOR[node] + vertex_subscripts[i]
                    )
                if propagator[2] not in vertex_operands:
                    vertex_operands.append(propagator[2])
                    vertex_subscripts.append(
                        _SUB_SPIN[node + 1] + _SUB_VECTOR[node + 1]
                    )
                else:
                    i = vertex_operands.index(propagator[2])
                    vertex_subscripts[i] = (
                        vertex_subscripts[i]
                        + _SUB_SPIN[node + 1]
                        + _SUB_VECTOR[node + 1]
                    )
                node += 2
            for key in range(len(propagator_subscripts)):
                propagator_subscripts[key] = (
                    propagator_subscripts[key][0::2] + propagator_subscripts[key][1::2]
                )
            for key in range(len(vertex_subscripts)):
                vertex_subscripts[key] = (
                    vertex_subscripts[key][0::2] + vertex_subscripts[key][1::2]
                )
            self.operands.append([propagator_operands, vertex_operands])
            self.subscripts.append(
                ",".join(propagator_subscripts) + "," + ",".join(vertex_subscripts)
            )


class QuarkDiagram:
    def __init__(
        self,
        adjacency_matrix,
        vertex_list: List[int] = None,
        L: int = None,
        usedNp: int = None,
        debug: bool = False,
    ) -> None:
        """
        Initialize QuarkDiagram.

        Args:
            adjacency_matrix: Graph adjacency matrix representing the diagram
            vertex_list: List of vertex indices
            L: Spatial lattice size (total number of lattice points = L^3)
            usedNp: Number of sampled points (default: usedNp from Current vertex)
            debug: Enable debug output
        """
        self.adjacency_matrix = adjacency_matrix
        self.vertex_list = vertex_list
        self.L = L  # Spatial lattice size
        self.usedNp = usedNp  # Number of sampled points
        self.operands = []
        self.subscripts = []
        self.operands_data = []
        self.propagator_types = []  # Store propagator types for each contraction group
        self.debug = debug

        # Sampling-related fields for point weight calculation
        self.sampling_groups = (
            {}
        )  # Dict[int, List[Tuple[int, str]]], group_id -> [(vertex_idx, "left"|"right"), ...]
        self.scene_weights = []  # List[float], weight for each scene
        self.scene_constraints = (
            []
        )  # List[List[Tuple]], constraints for unify_vertex_point_color_indices

        self.analyse()

    def analyse(self) -> None:
        """
        Analyze the quark diagram and generate contraction patterns.

        If vertex_list is provided and contains current vertices (non-zero values),
        this method will expand the diagram into multiple diagrams with different
        propagator types (VSV, PSV, VSP, PSP).

        Otherwise, it falls back to the standard v2v analysis.
        """

        # Check if any vertex is a current vertex (non-zero value)
        has_current_vertices = (
            any(v != 0 for v in self.vertex_list) if self.vertex_list else False
        )

        if not has_current_vertices:
            # No current vertices, use standard v2v analysis
            self.analyse_v2v()
            return

        # Has current vertices, need to expand the diagram
        # This will generate multiple QuarkDiagram objects stored in self.expanded_diagrams
        self.expanded_diagrams = []
        self.expanded_diagrams_weights = []
        self.expand_with_current()

    def expand_with_current(self) -> List["QuarkDiagram"]:
        """
        Expand the diagram into multiple diagrams with sampling weights.

        Vertices in vertex_list with non-zero values are current vertices.
        For each current vertex, we need to consider it can be either:
        - 'v' (vertex/eigenvector): connects via VSV propagators
        - 'p' (point): connects via PSV/VSP/PSP propagators depending on the other end
        For non-current vertices (vertex_list value = 0), only 'v' state is allowed.

        The method generates all possible combinations and stores them in self.operands
        and self.subscripts as a list of diagrams.

        Sampling Weight Logic:
        After generating all state combinations, this method further expands each diagram
        into multiple "scenes" based on point coincidence patterns. For each sampling group
        (determined by vertex_list index), it enumerates all possible ways the point positions
        can coincide (using integer partitions), calculates the corresponding sampling weight
        w(k) = C(L^3,k)/C(usedNp,k), and creates sub-diagrams with appropriate constraints.

        See localized_blending.md "程序设计：任意点采样的组合权重" for mathematical details.
        """
        if self.debug:
            logger.debug(f"\n{'='*80}")
            logger.debug(f"QuarkDiagram.expand_with_current() - Expanding diagram")
            logger.debug(f"{'='*80}")
            logger.debug(f"Vertex list: {self.vertex_list}")
            logger.debug(f"Adjacency matrix:")
            logger.debug(self.adjacency_matrix)

        from copy import deepcopy

        # Find all edges connected to current vertices
        # Each edge is (propagator_id, source_vertex, sink_vertex)
        edges_to_current = []
        adjacency_matrix = deepcopy(self.adjacency_matrix)
        num_vertex = len(adjacency_matrix)

        for i in range(num_vertex):
            for j in range(num_vertex):
                path = adjacency_matrix[i][j]
                if path != 0:
                    # Check if either end is a current vertex (non-zero in vertex_list)
                    if (self.vertex_list and self.vertex_list[i] != 0) or (
                        self.vertex_list and self.vertex_list[j] != 0
                    ):
                        if isinstance(path, int):
                            edges_to_current.append([path, i, j])
                        elif isinstance(path, list):
                            for _path in path:
                                edges_to_current.append([_path, i, j])

        if self.debug:
            logger.debug(
                f"\nFound {len(edges_to_current)} edges connected to current vertices:"
            )
            for edge in edges_to_current:
                logger.debug(f"  propagator[{edge[0]}]: vertex {edge[1]} -> vertex {edge[2]}")
            logger.debug(
                f"Current vertices (non-zero in vertex_list): {[i for i, v in enumerate(self.vertex_list) if v != 0]}"
            )

        # Generate all possible state combinations
        # Generate state combinations based on vertex_list
        # Non-current vertices (v == 0) only have (v,v) state
        # Current vertices (v != 0) have all 4 combinations: (v,v), (v,p), (p,v), (p,p)
        state_combinations = []

        for v in self.vertex_list:
            if v == 0:
                # Non-current vertex: only (v,v) state
                state_combinations.append([{"left": "v", "right": "v"}])
            else:
                # Current vertex: all 4 state combinations
                state_combinations.append(
                    [
                        {"left": "v", "right": "v"},
                        {"left": "v", "right": "p"},
                        {"left": "p", "right": "v"},
                        {"left": "p", "right": "p"},
                    ]
                )
        # Generate all combinations across current vertices
        from itertools import product as iter_product

        all_combinations = list(iter_product(*state_combinations))

        if self.debug:
            logger.debug(f"\nGenerating {len(all_combinations)} state combinations:")
            for i, vertex_state in enumerate(all_combinations):
                state_desc = ", ".join(
                    [
                        f"vertex[{j}]=(left={vertex_state[j]['left']}, right={vertex_state[j]['right']})"
                        for j in range(len(self.vertex_list))
                    ]
                )
                logger.debug(f"  Combination {i}: {state_desc}")

        # After all state combinations are generated, expand each diagram into scenes based on point coincidences
        if self.debug:
            logger.debug(f"\n{'='*80}")
            logger.debug(f"Expanding diagrams into point coincidence scenes...")
            logger.debug(f"{'='*80}")

        # For each combination, create a new StateExpandedDiagram and expand it into scenes
        for combo_idx, vertex_state in enumerate(all_combinations):

            if self.debug:
                logger.debug(f"\n{'='*60}")
                logger.debug(f"Processing combination {combo_idx}:")
                state_desc = ", ".join(
                    [
                        f"vertex[{j}]=(left={vertex_state[j]['left']}, right={vertex_state[j]['right']})"
                        for j in range(len(self.vertex_list))
                    ]
                )
                logger.debug(f"  State: {state_desc}")

            # Create a new StateExpandedDiagram for this combination
            new_diagram = StateExpandedDiagram(
                adjacency_matrix=self.adjacency_matrix,
                vertex_list=self.vertex_list,
                vertex_state=vertex_state,
                L=self.L,
                usedNp=self.usedNp,
                debug=self.debug,
            )

            # Expand this diagram into scenes (stored in new_diagram.scene_diagrams)
            new_diagram.expand_scenes()
            if self.debug:
                logger.debug(f"  Expanded to {len(new_diagram.scene_diagrams)} scenes")

            # Store the StateExpandedDiagram (which contains scene_diagrams internally)
            self.expanded_diagrams.append(new_diagram)
            self.expanded_diagrams_weights.append(1.0)

        if self.debug:
            logger.debug(
                f"\nTotal StateExpandedDiagram instances: {len(self.expanded_diagrams)}"
            )

    def analyse_v2v(self) -> None:
        if not self.subscripts == []:
            return None
        from copy import deepcopy

        adjacency_matrix = deepcopy(self.adjacency_matrix)
        num_vertex = len(adjacency_matrix)
        visited = [False] * num_vertex
        for idx in range(num_vertex):
            if visited[idx]:
                continue
            propagators = []
            visited[idx] = True
            queue = [idx]
            while queue != []:
                i = queue.pop(0)
                for j in range(num_vertex):
                    path = adjacency_matrix[i][j]
                    if path != 0:
                        adjacency_matrix[i][j] = 0
                        if not visited[j]:
                            visited[j] = True
                            queue.append(j)
                        if isinstance(path, int):
                            propagators.append([path, i, j])
                        elif isinstance(path, list):
                            # Handle arbitrarily nested lists (e.g., 3x3 matrices for baryon contractions)
                            def extract_propagators(nested_path, src, snk):
                                """Recursively extract propagator indices from nested lists."""
                                results = []
                                for item in nested_path:
                                    if isinstance(item, int):
                                        if item != 0:  # Skip zero entries
                                            results.append([item, src, snk])
                                    elif isinstance(item, list):
                                        results.extend(extract_propagators(item, src, snk))
                                return results
                            propagators.extend(extract_propagators(path, i, j))
                        else:
                            raise ValueError(
                                f"Invalid value {path} in the adjacency matrix"
                            )
            if propagators == []:
                continue
            vertex_operands = []
            vertex_subscripts = []
            propagator_operands = []
            propagator_subscripts = []
            node = 0
            for propagator in propagators:
                propagator_operands.append(propagator)
                propagator_subscripts.append(
                    _SUB_SPIN[node + 1]
                    + _SUB_VECTOR[node + 1]
                    + _SUB_SPIN[node]
                    + _SUB_VECTOR[node]
                )
                if propagator[1] not in vertex_operands:
                    vertex_operands.append(propagator[1])
                    vertex_subscripts.append(_SUB_SPIN[node] + _SUB_VECTOR[node])
                else:
                    i = vertex_operands.index(propagator[1])
                    vertex_subscripts[i] = (
                        _SUB_SPIN[node] + _SUB_VECTOR[node] + vertex_subscripts[i]
                    )
                if propagator[2] not in vertex_operands:
                    vertex_operands.append(propagator[2])
                    vertex_subscripts.append(
                        _SUB_SPIN[node + 1] + _SUB_VECTOR[node + 1]
                    )
                else:
                    i = vertex_operands.index(propagator[2])
                    vertex_subscripts[i] = (
                        vertex_subscripts[i]
                        + _SUB_SPIN[node + 1]
                        + _SUB_VECTOR[node + 1]
                    )
                node += 2
            for key in range(len(propagator_subscripts)):
                propagator_subscripts[key] = (
                    propagator_subscripts[key][0::2] + propagator_subscripts[key][1::2]
                )
            for key in range(len(vertex_subscripts)):
                vertex_subscripts[key] = (
                    vertex_subscripts[key][0::2] + vertex_subscripts[key][1::2]
                )

            # Determine vertex types (all V2V for trivial case)
            vertex_types = ["V2V"] * len(vertex_operands)

            # Initialize vertex_point_info for trivial case
            vertex_point_info = {}
            node = 0
            for prop_idx, propagator in enumerate(propagators):
                src, snk = propagator[1], propagator[2]
                if src not in vertex_point_info:
                    vertex_point_info[src] = {"left": {}, "right": {}}
                vertex_point_info[src]["left"]["spin"] = _SUB_SPIN[node]
                vertex_point_info[src]["left"]["eigen"] = _SUB_VECTOR[node]

                if snk not in vertex_point_info:
                    vertex_point_info[snk] = {"left": {}, "right": {}}
                vertex_point_info[snk]["right"]["spin"] = _SUB_SPIN[node + 1]
                vertex_point_info[snk]["right"]["eigen"] = _SUB_VECTOR[node + 1]
                node += 2

            self.operands.append([propagator_operands, vertex_operands])
            self.subscripts.append(
                ",".join(propagator_subscripts) + "," + ",".join(vertex_subscripts)
            )

            # Initialize propagator_types if not exists
            if not hasattr(self, "propagator_types"):
                self.propagator_types = []
            self.propagator_types.append(["VSV"] * len(propagators))

            # Initialize vertex_types if not exists
            if not hasattr(self, "vertex_types"):
                self.vertex_types = []
            self.vertex_types.append(vertex_types)

            # Initialize vertex_point_info if not exists
            if not hasattr(self, "vertex_point_info"):
                self.vertex_point_info = []
            self.vertex_point_info.append(vertex_point_info)


class StateExpandedDiagram(QuarkDiagram):
    """
    Subclass of QuarkDiagram representing diagrams expanded by state combinations.

    These diagrams are generated by expand_with_current() method, where each vertex
    can independently choose a state combination for (left, right) ends based on vertex_list.
    Current vertices (non-zero in vertex_list) have 4 combinations: (v,v), (v,p), (p,v), (p,p).
    Non-current vertices (zero in vertex_list) only have (v,v).

    Attributes:
        _combo: Tuple of dicts, one for each vertex, each with "left"/"right" -> 'v'/'p'
    """

    def __init__(
        self,
        adjacency_matrix,
        vertex_list: List[int] = None,
        vertex_state: tuple = None,
        L: int = None,
        usedNp: int = None,
        debug: bool = False,
    ) -> None:
        """
        Initialize StateExpandedDiagram.

        Args:
            adjacency_matrix: Graph adjacency matrix representing the diagram
            vertex_list: List of vertex indices
            vertex_state: Tuple of dicts, one for each vertex, each with "left"/"right" -> 'v'/'p'
            L: Spatial lattice size (total number of lattice points = L^3)
            usedNp: Number of sampled points (default: usedNp from Current vertex)
            debug: Enable debug output
        """

        # Override base class initialization for StateExpandedDiagram specific fields
        self.adjacency_matrix = adjacency_matrix
        self.vertex_list = vertex_list
        self.L = L
        self.usedNp = usedNp
        self.debug = debug

        self.operands = []
        self.subscripts = []
        self.operands_data = []
        self.propagator_types = []
        self.vertex_types = []
        self.vertex_infos = []

        # Initialize sampling-related fields
        self.sampling_groups = {}
        self.scene_weights = []  # Store weights for each scene diagram
        self.scene_constraints = []  # Store constraints for each scene diagram
        self.scene_diagrams = []  # Store expanded SceneExpandedDiagram objects

        # Store vertex_state
        self._vertex_state = vertex_state
        self._analyse_with_states(vertex_state)

    def _analyse_with_states(self, vertex_state: tuple) -> None:
        """
        Analyze the diagram with specific states for vertices.
        Fill self.operands and self.subscripts for this specific state combination.

        Args:
            vertex_state: Tuple of dicts, one for each vertex, each with "left"/"right" -> 'v'/'p'
        """
        from copy import deepcopy

        adjacency_matrix = deepcopy(self.adjacency_matrix)
        num_vertex = len(adjacency_matrix)
        visited = [False] * num_vertex

        contraction_group = 0
        for idx in range(num_vertex):
            if visited[idx]:
                continue

            if self.debug:
                logger.debug(
                    f"\n  Contraction group {contraction_group}: Starting from vertex {idx}"
                )

            propagators = []
            visited[idx] = True
            queue = [idx]
            while queue != []:
                i = queue.pop(0)
                for j in range(num_vertex):
                    path = adjacency_matrix[i][j]
                    if path != 0:
                        adjacency_matrix[i][j] = 0
                        if not visited[j]:
                            visited[j] = True
                            queue.append(j)
                        if isinstance(path, int):
                            propagators.append([path, i, j])
                        elif isinstance(path, list):
                            for _path in path:
                                propagators.append([_path, i, j])
                        else:
                            raise ValueError(
                                f"Invalid value {path} in the adjacency matrix"
                            )

            if propagators == []:
                if self.debug:
                    logger.debug(f"    No propagators found, skipping")
                continue

            if self.debug:
                logger.debug(f"    Total propagators in this group: {len(propagators)}")

            # Build subscripts based on propagator types
            self._build_subscripts_with_types(propagators, vertex_state)
            contraction_group += 1

        if self.debug:
            logger.debug(f"\n{'='*60}")
            logger.debug(f"Combination analysis completed")
            logger.debug(
                f"Total contraction groups for this combination: {len(self.operands)}"
            )
            logger.debug(f"{'='*60}\n")

    def _build_subscripts_with_types(
        self,
        propagators: List,
        vertex_state: tuple,
    ) -> None:
        """
        Build subscripts for propagators considering their types (VSV/PSV/VSP/PSP).

        Args:
            propagators: List of [propagator_id, source, sink]
            vertex_state: Tuple of dicts, one for each vertex, each with "left"/"right" -> 'v'/'p'
            contraction_group: Index of current contraction group
        """
        # Use global label sets

        vertex_operands = []
        vertex_subscripts = []
        vertex_types = []
        propagator_operands = []
        propagator_subscripts = []
        propagator_types = []
        vertex_infos = []
        for v_id in range(len(vertex_state)):
            vertex_operands.append(v_id)
            if vertex_state[v_id]["left"] == "v" and vertex_state[v_id]["right"] == "v":
                vertex_types.append("V2V")
                vertex_infos.append(
                    {
                        "left": {"spin": None, "eigen": None},
                        "right": {"spin": None, "eigen": None},
                    }
                )
            elif (
                vertex_state[v_id]["left"] == "v" and vertex_state[v_id]["right"] == "p"
            ):
                vertex_types.append("V2P")
                vertex_infos.append(
                    {
                        "left": {"spin": None, "eigen": None},
                        "right": {"spin": None, "point": None, "color": None},
                    }
                )
            elif (
                vertex_state[v_id]["left"] == "p" and vertex_state[v_id]["right"] == "v"
            ):
                vertex_types.append("P2V")
                vertex_infos.append(
                    {
                        "left": {"spin": None, "point": None, "color": None},
                        "right": {"spin": None, "eigen": None},
                    }
                )
            elif (
                vertex_state[v_id]["left"] == "p" and vertex_state[v_id]["right"] == "p"
            ):
                vertex_types.append("P2P")
                vertex_infos.append(
                    {
                        "left": {"spin": None, "point": None, "color": None},
                        "right": {"spin": None, "point": None, "color": None},
                    }
                )

        node = 0  # Track label position, compatible with analyse_v2v

        for prop_idx, propagator in enumerate(propagators):
            prop_id, src, snk = propagator

            # Determine propagator type based on states
            # src (source) corresponds to right end (ket) of propagator
            # snk (sink) corresponds to left end (bra) of propagator
            # Convention: propagator type named as S_{sink_type, source_type}
            src_state = vertex_state[src]["left"]  # right end state
            snk_state = vertex_state[snk]["right"]  # left end state

            # Determine type: S_{sink_state, src_state}
            if snk_state == "v" and src_state == "v":
                prop_type = "VSV"
            elif snk_state == "p" and src_state == "v":
                prop_type = "PSV"
            elif snk_state == "v" and src_state == "p":
                prop_type = "VSP"
            else:  # snk_state == "p" and src_state == "p"
                prop_type = "PSP"

            propagator_types.append(prop_type)
            propagator_operands.append(propagator)

            if self.debug:
                logger.debug(
                    f"    Propagator {prop_idx}: [{prop_id}, {src}, {snk}] type={prop_type}"
                )

            # Build subscript based on type
            if prop_type == "VSV":
                # VSV: use the same logic as analyse_v2v for consistency
                # Initial subscript: sink_spin + sink_eigen + source_spin + source_eigen
                spin_src = _SUB_SPIN[node]
                vector_src = _SUB_VECTOR[node]
                spin_snk = _SUB_SPIN[node + 1]
                vector_snk = _SUB_VECTOR[node + 1]
                prop_subscript = spin_snk + spin_src + vector_snk + vector_src

                # Vertices
                vertex_infos[src]["left"]["spin"] = spin_src
                vertex_infos[src]["left"]["eigen"] = vector_src

                # if src not in vertex_operands:
                #     vertex_operands.append(src)
                #     vertex_subscripts.append(spin_src + vector_src)
                # else:
                #     i = vertex_operands.index(src)
                #     vertex_subscripts[i] = (
                #         spin_src + vector_src + vertex_subscripts[i]
                #     )

                # Always record metadata, even if vertex already in operands
                vertex_infos[snk]["right"]["spin"] = spin_snk
                vertex_infos[snk]["right"]["eigen"] = vector_snk

                # if snk not in vertex_operands:
                #     vertex_operands.append(snk)
                #     vertex_subscripts.append(spin_snk + vector_snk)
                # else:
                #     i = vertex_operands.index(snk)
                #     vertex_subscripts[i] = (
                #         vertex_subscripts[i] + spin_snk + vector_snk
                #     )

                node += 2

            elif prop_type == "VSP":
                # VSP: Initial: sink_spin + sink_eigen + source_spin + source_point + source_color
                # Similar to VSV but source has point+color instead of eigen
                spin_src = _SUB_SPIN[node]
                point_src = _SUB_POINT[node]
                color_src = _SUB_COLOR[node]
                spin_snk = _SUB_SPIN[node + 1]
                vector_snk = _SUB_VECTOR[node + 1]
                prop_subscript = (
                    spin_snk + spin_src + vector_snk + point_src + color_src
                )

                # Source vertex (point)
                # Always record metadata
                vertex_infos[src]["left"]["spin"] = spin_src
                vertex_infos[src]["left"]["point"] = point_src
                vertex_infos[src]["left"]["color"] = color_src

                # if src not in vertex_operands:
                #     vertex_operands.append(src)
                #     vertex_subscripts.append(spin_src + point_src + color_src)
                # else:
                #     i = vertex_operands.index(src)
                #     vertex_subscripts[i] = (
                #         spin_src + point_src + color_src + vertex_subscripts[i]
                #     )

                # Sink vertex (eigenvector)
                # Always record metadata
                vertex_infos[snk]["right"]["spin"] = spin_snk
                vertex_infos[snk]["right"]["eigen"] = vector_snk

                # if snk not in vertex_operands:
                #     vertex_operands.append(snk)
                #     vertex_subscripts.append(spin_snk + vector_snk)
                # else:
                #     i = vertex_operands.index(snk)
                #     vertex_subscripts[i] = (
                #         vertex_subscripts[i] + spin_snk + vector_snk
                #     )

                node += 2

            elif prop_type == "PSV":

                # PSV: Initial: sink_spin + sink_point + sink_color + source_spin + source_eigen
                spin_src = _SUB_SPIN[node]
                vector_src = _SUB_VECTOR[node]
                spin_snk = _SUB_SPIN[node + 1]
                point_snk = _SUB_POINT[node + 1]
                color_snk = _SUB_COLOR[node + 1]
                prop_subscript = (
                    spin_snk + spin_src + point_snk + color_snk + vector_src
                )

                # Source vertex (eigenvector)
                # Always record metadata
                vertex_infos[src]["left"]["spin"] = spin_src
                vertex_infos[src]["left"]["eigen"] = vector_src

                # if src not in vertex_operands:
                #     vertex_operands.append(src)
                #     vertex_subscripts.append(spin_src + vector_src)
                # else:
                #     i = vertex_operands.index(src)
                #     vertex_subscripts[i] = (
                #         spin_src + vector_src + vertex_subscripts[i]
                #     )

                # Sink vertex (point)
                # Always record metadata
                vertex_infos[snk]["right"]["spin"] = spin_snk
                vertex_infos[snk]["right"]["point"] = point_snk
                vertex_infos[snk]["right"]["color"] = color_snk

                # if snk not in vertex_operands:
                #     vertex_operands.append(snk)
                #     vertex_subscripts.append(
                #         spin_snk + point_snk + color_snk
                #     )
                # else:
                #     i = vertex_operands.index(snk)
                #     vertex_subscripts[i] = (
                #         vertex_subscripts[i]
                #         + spin_snk
                #         + point_snk
                #         + color_snk
                #     )

                node += 2

            else:  # PSP
                # PSP: both source and sink are points
                spin_src = _SUB_SPIN[node]
                point_src = _SUB_POINT[node]
                color_src = _SUB_COLOR[node]
                spin_snk = _SUB_SPIN[node + 1]
                point_snk = _SUB_POINT[node + 1]
                color_snk = _SUB_COLOR[node + 1]
                prop_subscript = (
                    spin_snk + spin_src + point_snk + color_snk + point_src + color_src
                )

                # Source vertex (point)
                # Always record metadata
                vertex_infos[src]["left"]["spin"] = spin_src
                vertex_infos[src]["left"]["point"] = point_src
                vertex_infos[src]["left"]["color"] = color_src

                # if src not in vertex_operands:
                #     vertex_operands.append(src)
                #     vertex_subscripts.append(spin_src + point_src + color_src)
                # else:
                #     i = vertex_operands.index(src)
                #     vertex_subscripts[i] = (
                #         spin_src + point_src + color_src + vertex_subscripts[i]
                #     )

                # Sink vertex (point)
                # Always record metadata
                vertex_infos[snk]["right"]["spin"] = spin_snk
                vertex_infos[snk]["right"]["point"] = point_snk
                vertex_infos[snk]["right"]["color"] = color_snk

                # if snk not in vertex_operands:
                #     vertex_operands.append(snk)
                #     vertex_subscripts.append(
                #         spin_snk + point_snk + color_snk
                #     )
                # else:
                #     i = vertex_operands.index(snk)
                #     vertex_subscripts[i] = (
                #         vertex_subscripts[i]
                #         + spin_snk
                #         + point_snk
                #         + color_snk
                #     )

                node += 2

            propagator_subscripts.append(prop_subscript)

            if self.debug:
                logger.debug(f"      Initial subscript: {prop_subscript}")

        for vertex_idx, vertex_info in enumerate(vertex_infos):
            if vertex_types[vertex_idx] == "V2V":
                vertex_subscripts.append(
                    vertex_info["left"]["spin"]
                    + vertex_info["right"]["spin"]
                    + vertex_info["left"]["eigen"]
                    + vertex_info["right"]["eigen"]
                )
            elif vertex_types[vertex_idx] == "V2P":
                vertex_subscripts.append(
                    vertex_info["left"]["spin"]
                    + vertex_info["right"]["spin"]
                    + vertex_info["left"]["eigen"]
                    + vertex_info["right"]["point"]
                    + vertex_info["right"]["color"]
                )
            elif vertex_types[vertex_idx] == "P2V":
                vertex_subscripts.append(
                    vertex_info["left"]["spin"]
                    + vertex_info["right"]["spin"]
                    + vertex_info["left"]["point"]
                    + vertex_info["left"]["color"]
                    + vertex_info["right"]["eigen"]
                )
            else:  # P2P
                vertex_subscripts.append(
                    vertex_info["left"]["spin"]
                    + vertex_info["right"]["spin"]
                    + vertex_info["left"]["point"]
                    + vertex_info["left"]["color"]
                    + vertex_info["right"]["point"]
                    + vertex_info["right"]["color"]
                )

        final_subscript = (
            ",".join(propagator_subscripts) + "," + ",".join(vertex_subscripts)
        )

        if self.debug:
            logger.debug(f"\n    Final subscript: {final_subscript}")
            logger.debug(f"    Propagator operands: {propagator_operands}")
            logger.debug(f"    Vertex operands: {vertex_operands}")
            logger.debug(f"    Propagator types: {propagator_types}")
            logger.debug(f"    Vertex types: {vertex_types}")
            for v_idx, v_id in enumerate(vertex_operands):
                logger.debug(f"      Vertex {v_id}: type={vertex_types[v_idx]}")

        self.operands.append([propagator_operands, vertex_operands])
        self.subscripts.append(final_subscript)
        self.propagator_types.append(propagator_types)
        self.vertex_types.append(vertex_types)
        self.vertex_infos.append(vertex_infos)
        self.operands_data.append(None)  # Placeholder for operands data

    def expand_scenes(self) -> None:
        """
        Expand this state-expanded diagram into point coincidence scenes.

        Enumerate all possible point coincidence patterns for sampling groups and create
        SceneExpandedDiagram objects with corresponding weights and constraints.
        The results are stored in self.scene_diagrams.

        Uses L and usedNp parameters from the instance (set during initialization).
        """
        from itertools import product as iter_product

        # Use L and usedNp from instance (set during initialization)
        M = self.L  # Total number of lattice points (L^3)
        N = self.usedNp  # Number of sampled points

        # Collect sampling groups from state_dict if not already collected
        self._collect_sampling_groups(self._vertex_state)

        # Initialize lists for storing scene diagrams, weights, and constraints
        self.scene_diagrams = []
        self.scene_weights = []
        self.scene_constraints = []

        if not self.sampling_groups:
            # No point sampling needed, create a single SceneExpandedDiagram
            scene_diagram = SceneExpandedDiagram(
                adjacency_matrix=self.adjacency_matrix,
                vertex_list=self.vertex_list,
                operands=self.operands,
                subscripts=self.subscripts,
                operands_data=self.operands_data,
                propagator_types=self.propagator_types,
                vertex_types=self.vertex_types,
                vertex_infos=self.vertex_infos,
                scene_constraints=[],
                L=M,  # M here represents total lattice points (L^3)
                usedNp=N,
                debug=self.debug,
            )
            scene_diagram.unify_vertex_point_color_indices()
            self.scene_diagrams.append(scene_diagram)
            self.scene_weights.append(1.0)
            self.scene_constraints.append([])
            if self.debug:
                logger.debug(f"No sampling groups, keeping as single scene")
            return

        if self.debug:
            logger.debug(f"Generating scenes for {len(self.sampling_groups)} sampling groups")

        # Enumerate scenes for each sampling group
        # For multiple independent groups, we need Cartesian product of their scenes
        group_scenes = {}
        for group_id, positions in self.sampling_groups.items():
            r = len(positions)
            scenes = enumerate_point_scenes(r, M, N)
            group_scenes[group_id] = [
                (partition, weight, positions) for partition, weight in scenes
            ]

            if self.debug:
                logger.debug(f"  Group {group_id}: {r} positions, {len(scenes)} scenes")

        # Generate Cartesian product of scenes across all groups
        group_ids = list(group_scenes.keys())
        scene_combinations = iter_product(*[group_scenes[gid] for gid in group_ids])

        for scene_combo in scene_combinations:
            # Calculate total weight (product of weights from each group)
            total_weight = 1.0
            all_positions = []
            all_partitions = []

            for group_idx, (partition, weight, positions) in enumerate(scene_combo):
                total_weight *= weight
                all_positions.extend(positions)
                all_partitions.append((partition, positions))

            # Generate constraints from partitions
            constraints = self._build_scene_constraints(all_partitions)

            if self.debug:
                logger.debug(
                    f"    Scene: weight={total_weight:.6f}, constraints={constraints}"
                )

            # Create a new SceneExpandedDiagram for this scene combination
            scene_diagram = SceneExpandedDiagram(
                adjacency_matrix=self.adjacency_matrix,
                vertex_list=self.vertex_list,
                operands=self.operands,
                subscripts=self.subscripts,
                operands_data=self.operands_data,
                propagator_types=self.propagator_types,
                vertex_types=self.vertex_types,
                vertex_infos=self.vertex_infos,
                scene_constraints=constraints,
                L=M,  # M here represents total lattice points (L^3)
                usedNp=N,
                debug=self.debug,
            )
            scene_diagram.unify_vertex_point_color_indices()

            self.scene_diagrams.append(scene_diagram)
            self.scene_weights.append(total_weight)
            self.scene_constraints.append(constraints)

        if self.debug:
            logger.debug(f"Generated {len(self.scene_diagrams)} scene diagrams")

    def _collect_sampling_groups(self, vertex_state: tuple) -> None:
        """
        Collect sampling group information from vertex_state.

        Groups vertices by their vertex_list index to determine which points come from the same sampling set.
        Only vertices with 'p' state (point sampling) are recorded.

        Args:
            vertex_state: Tuple of dicts, one for each vertex, each with "left"/"right" -> 'v'/'p'
        """
        # Group vertices by their vertex_list index
        # vertex_list index determines independent sampling groups
        for vertex_idx in range(len(vertex_state)):
            # Get the group_id from vertex_list
            group_id = self.vertex_list[vertex_idx]

            # Check left state
            left_state = vertex_state[vertex_idx]["left"]
            if left_state == "p":
                if group_id not in self.sampling_groups:
                    self.sampling_groups[group_id] = []
                self.sampling_groups[group_id].append((vertex_idx, "left"))

            # Check right state
            right_state = vertex_state[vertex_idx]["right"]
            if right_state == "p":
                if group_id not in self.sampling_groups:
                    self.sampling_groups[group_id] = []
                self.sampling_groups[group_id].append((vertex_idx, "right"))

        if self.debug:
            logger.debug(f"\n  Collected sampling groups:")
            for group_id, positions in self.sampling_groups.items():
                logger.debug(
                    f"    Group {group_id}: {positions} ({len(positions)} point positions)"
                )

    def _build_scene_constraints(
        self, partitions_and_positions: List[Tuple[List[int], List[Tuple[int, str]]]]
    ) -> List[Tuple[Optional[int], Optional[int]]]:
        """
        Build constraints for unify_vertex_point_color_indices from multiple group partitions.

        Args:
            partitions_and_positions: List of (partition, positions) for each sampling group

        Returns:
            Constraints list for all vertices
        """
        # Find max vertex index
        all_positions = []
        for _, positions in partitions_and_positions:
            all_positions.extend(positions)

        if not all_positions:
            return []

        max_vertex_idx = max(pos[0] for pos in all_positions)
        constraints = [(None, None)] * (max_vertex_idx + 1)

        # Process each group's partition separately, using separate point ID ranges
        point_id_offset = 0

        for partition, positions in partitions_and_positions:
            # Convert this group's partition to constraints
            point_id_counter = point_id_offset
            position_idx = 0

            for group_size in partition:
                group_point_id = point_id_counter

                for _ in range(group_size):
                    if position_idx < len(positions):
                        vertex_idx, side = positions[position_idx]
                        left_id, right_id = constraints[vertex_idx]

                        if side == "left":
                            constraints[vertex_idx] = (group_point_id, right_id)
                        else:  # side == "right"
                            constraints[vertex_idx] = (left_id, group_point_id)

                        position_idx += 1

                point_id_counter += 1

            # Update offset for next group
            point_id_offset = point_id_counter

        return constraints


class SceneExpandedDiagram(QuarkDiagram):
    """
    Subclass of QuarkDiagram representing diagrams expanded by point coincidence scenes.

    These diagrams are generated by StateExpandedDiagram.expand_scenes() method, where each
    state-expanded diagram is further expanded into multiple scenes based on point coincidence
    patterns. Each scene has a sampling weight w(k) = C(L^3,k)/C(usedNp,k) and corresponding constraints.
    The weights are stored in the parent StateExpandedDiagram.scene_weights list.

    Attributes:
        scene_constraints: List of constraints for unify_vertex_point_color_indices
    """

    def __init__(
        self,
        adjacency_matrix,
        vertex_list: List[int] = None,
        operands: List = None,
        subscripts: List[str] = None,
        operands_data: List = None,
        propagator_types: List = None,
        vertex_types: List = None,
        vertex_infos: List = None,
        scene_constraints: List = None,
        L: int = None,
        usedNp: int = None,
        debug: bool = False,
    ) -> None:
        """
        Initialize SceneExpandedDiagram.

        Args:
            adjacency_matrix: Graph adjacency matrix representing the diagram
            vertex_list: List of vertex indices
            operands: List of operands for contraction groups
            subscripts: List of subscripts for contraction groups
            operands_data: List of operands data
            propagator_types: List of propagator types for each contraction group
            vertex_types: List of vertex types for each contraction group
            vertex_infos: List of vertex point info for each contraction group
            scene_constraints: Constraints for unify_vertex_point_color_indices
            L: Spatial lattice size (total number of lattice points = L^3)
            usedNp: Number of sampled points (default: usedNp from Current vertex)
            debug: Enable debug output
        """
        self.adjacency_matrix = adjacency_matrix
        self.vertex_list = vertex_list
        self.L = L
        self.usedNp = usedNp
        self.debug = debug

        # Override base class initialization for SceneExpandedDiagram specific fields
        self.operands = operands if operands is not None else []
        self.subscripts = subscripts if subscripts is not None else []
        self.operands_data = operands_data if operands_data is not None else []
        self.propagator_types = propagator_types if propagator_types is not None else []
        self.vertex_types = vertex_types if vertex_types is not None else []
        self.vertex_infos = vertex_infos

        # Initialize sampling-related fields
        self.sampling_groups = {}
        self.scene_constraints = scene_constraints

    def unify_vertex_point_color_indices(self) -> None:
        """
        Unify point and color indices for specified vertices based on constraints.

        Uses pre-recorded metadata from _build_subscripts_with_types for efficient processing.

        For SceneExpandedDiagram instances, uses the instance's own scene_constraints
        which are generated by _build_scene_constraints method from point coincidence scenes.

        The constraints format: List of tuples, e.g. [(0,1),(1,None),(None,2)]
        - constraints[i] represents vertex i
        - First element represents left, second element represents right
        - Same number means those positions should use the same point/color indices
        - None means that side is vector (no point/color to set)

        Example:
            [(0,1),(1,None),(None,2)] means:
            - vertex 0: left=0, right=1
            - vertex 1: left=1, right=None
            - vertex 2: left=None, right=2
            - Number 1 appears in vertex 0's right and vertex 1's left, so they use the same indices

        Note:
            Eigenvector positions (V in V2V, V2P, P2V) must be set to None.
            Setting a number for an eigenvector position will raise ValueError.
        """
        # For SceneExpandedDiagram, use instance's own scene_constraints if available
        if self.scene_constraints is not None:
            constraints = self.scene_constraints
        else:
            return None

        # Step 1: Find positions for each number in constraints and output corresponding vertex_infos
        number_to_positions = {}  # number -> list of (vertex_idx, side)

        for vertex_idx, (left_id, right_id) in enumerate(constraints):
            if left_id is not None:
                if left_id not in number_to_positions:
                    number_to_positions[left_id] = []
                number_to_positions[left_id].append((vertex_idx, "left"))

            if right_id is not None:
                if right_id not in number_to_positions:
                    number_to_positions[right_id] = []
                number_to_positions[right_id].append((vertex_idx, "right"))

        # Output the mapping information
        logger.debug("Constraint number to positions mapping:")
        for number, positions in number_to_positions.items():
            logger.debug(f"Number {number}: {positions}")

        logger.debug("\nCorresponding vertex_infos:")
        for number, positions in number_to_positions.items():
            logger.debug(f"Number {number}:")
            for vertex_idx, side in positions:
                found = False
                # vertex_infos is a List[List[Dict]]: one list per contraction group
                for group_idx, group_vertex_infos in enumerate(self.vertex_infos):
                    if isinstance(group_vertex_infos, list) and vertex_idx < len(group_vertex_infos):
                        vertex_info = group_vertex_infos[vertex_idx]
                        logger.debug(
                            f"  Group {group_idx}, Vertex {vertex_idx}, side '{side}': {vertex_info.get(side, 'N/A')}"
                        )
                        found = True
                    elif not isinstance(group_vertex_infos, list):
                        logger.debug(
                            f"  Group {group_idx}, Vertex {vertex_idx}, side '{side}': unexpected structure"
                        )
                if not found:
                    logger.debug(
                        f"  Vertex {vertex_idx}, side '{side}': not found in any contraction group"
                    )


class Particle:
    pass


class Meson(Particle):
    def __init__(self, elemental, operator, source) -> None:
        self.elemental = elemental
        self.elemental_data = None
        self.key = None
        self.operator = operator
        self.dagger = source
        self.outward = 1
        self.inward = 1
        self.smeared = True
        # cache is defined as a class variable of the Meson class.
        # cache is shared among all instances of Meson.
        backend = get_backend()
        self.cache: Dict[int, backend.ndarray] = {}

    def _release_resources(self):
        self.elemental_data = None
        self.cache = {}
        gc.collect()
        backend = get_backend()
        if hasattr(backend, "get_default_memory_pool"):
            try:
                backend.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    def release(self):
        self._release_resources()
        self.key = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def __str__(self) -> str:
        str = "### Meson ###\n"
        str += Rf"\n key = {self.key} \n"
        str += self.operator.__str__()
        str += Rf"\n dagger = {self.dagger} \n"
        return str

    def load(self, key, usedNe: int = None):
        self.usedNe = usedNe
        if self.key != key:
            self._release_resources()
            self.key = key
            self.elemental_data = self.elemental.load(key)
            backend = get_backend()
            self.cache: Dict[int, backend.ndarray] = {}
            self._make_cache()

    def _make_cache(self):
        from lattice.insertion.gamma import gamma

        backend = get_backend()
        cache = self.cache
        parts = self.operator.parts
        ret_gamma = []
        ret_elemental = []
        for i in range(len(parts) // 2):
            ret_gamma.append(gamma(parts[i * 2]))
            elemental_part = parts[i * 2 + 1]
            for j in range(len(elemental_part)):
                elemental_coeff, derivative_idx, momentum_idx, profile = elemental_part[
                    j
                ]
                elemental_coeff = complex(elemental_coeff)
                deriv_mom_tuple = (derivative_idx, momentum_idx)
                if deriv_mom_tuple not in cache:
                    cache[deriv_mom_tuple] = self.elemental_data[
                        derivative_idx, momentum_idx, :, : self.usedNe, : self.usedNe
                    ]
                if j == 0:
                    ret_elemental.append(elemental_coeff * cache[deriv_mom_tuple])
                else:
                    ret_elemental[-1] += elemental_coeff * cache[deriv_mom_tuple]
        if self.dagger:
            self.cache = (
                contract(
                    "ik,xlk,lj->xij",
                    gamma(8),
                    backend.asarray(ret_gamma).conj(),
                    gamma(8),
                ),
                contract("xtba->xtab", backend.asarray(ret_elemental).conj()),
            )
        else:
            self.cache = (
                backend.asarray(ret_gamma),
                backend.asarray(ret_elemental),
            )

    def get(self, t):
        """
        Get V2V vertex (operator matrix element O_{i,j}).

        Corresponds to formula (3.1): low-low term.

        Args:
            t: Time index (int or array)

        Returns:
            [Ns, Ns, Ne, Ne] if t is int
            [t_len, Ns, Ns, Ne, Ne] if t is array
        """
        if isinstance(t, int):
            if self.dagger:
                return contract("xij,xab->ijab", self.cache[0], self.cache[1][:, t])
            else:
                return contract("xij,xab->ijab", self.cache[0], self.cache[1][:, t])
        else:
            if self.dagger:
                return contract("xij,xtab->tijab", self.cache[0], self.cache[1][:, t])
            else:
                return contract("xij,xtab->tijab", self.cache[0], self.cache[1][:, t])


class Current(Meson):
    def __init__(
        self,
        elemental,
        operator,
        source,
        v2p_data: "CurrentElementalV2P" = None,
        p2v_data: "CurrentElementalP2V" = None,
        p2p_data: "CurrentElementalP2P" = None,
        debug: bool = False,
    ) -> None:
        super().__init__(elemental, operator, source)
        self.smeared = False
        self.v2p_data = v2p_data
        self.p2v_data = p2v_data
        self.p2p_data = p2p_data
        self.debug = debug
        # Two-time propagator caches (initialized on demand)
        # VSP caches
        self.vsp_cache = None
        self.vsp_cache_dagger = None
        self.vsp_cached_time = None
        # PSV caches
        self.psv_cache = None
        self.psv_cache_dagger = None
        self.psv_cached_time = None
        # PSP caches
        self.psp_cache = None
        self.psp_cache_dagger = None
        self.psp_cached_time = None
        self.cache_v2p = {}
        self.cache_p2v = {}
        self.cache_p2p = {}
        # Loaded precomputed data (p2p only, since v2p/p2v are pre-loaded in cache)
        self.p2p_loaded = None
        # Displacement reversal mapping cache
        self._disp_reversal_map = None

    def _release_resources(self):
        super()._release_resources()
        self.cache_v2p = {}
        self.cache_p2v = {}
        self.cache_p2p = {}
        self.p2p_loaded = None

    def release(self):
        self._release_resources()
        self.key = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def load(self, key, usedNe: int = None, usedNp: int = None):
        if self.key != key:
            self.key = key
            # Load elemental data (v2v) as before
            self.elemental_data = self.elemental.load(key)
            self.Lt = self.elemental_data.shape[2]

            # NO LONGER LOAD: eigenvector, point, gauge_field
            # These are only needed if computing on-the-fly

        # Set usedNe and usedNp with defaults from data loaders if not provided
        self.usedNe = usedNe if usedNe is not None else self.elemental_data.shape[3]
        # Get usedNp from p2v_data (since v2p now uses p2v via symmetry)
        self.usedNp = (
            usedNp
            if usedNp is not None
            else (self.p2v_data.Np if self.p2v_data is not None else None)
        )

        backend = get_backend()
        # Cache dictionaries for different propagator types
        self.cache: Dict[int, backend.ndarray] = {}
        self.cache_v2p: Dict[int, backend.ndarray] = {}
        self.cache_p2v: Dict[int, backend.ndarray] = {}
        self.cache_p2p: Dict[int, Dict[tuple, backend.ndarray] | None] = {}
        self._make_cache()

    def _build_displacement_reversal_map(self):
        """
        Build mapping from gaugelink_idx to its reverse displacement gaugelink_idx.

        Uses the symmetry: v2p(disp) ≈ p2v(-disp).transpose(Ne, Np)
        """
        from lattice.insertion import GaugeLink

        if self._disp_reversal_map is not None:
            return

        # Get num_disp from elemental data
        num_disp = self.elemental_data.shape[0]
        self._disp_reversal_map = {}

        for disp_idx in range(num_disp):
            gauge_link = GaugeLink(disp_idx)
            disp = tuple(gauge_link.displacement)
            reverse_disp = tuple(-d for d in disp)

            # Find gaugelink with reverse displacement
            # Use conjugate() method which reverses the gauge_list
            conjugate_link = gauge_link.conjugate()
            reverse_idx = conjugate_link.idx

            self._disp_reversal_map[disp_idx] = reverse_idx

            if self.debug:
                logger.debug(
                    f"Displacement reversal: {disp_idx} (disp={disp}) -> {reverse_idx} (disp={reverse_disp})"
                )

    def _make_cache(self):
        """
        Build cache from precomputed v2p, p2v, p2p data.

        Loads data from files and accumulates with operator coefficients.
        """
        from lattice.insertion.gamma import gamma
        from lattice.insertion import GaugeLink

        backend = get_backend()
        cache_v2v = self.cache
        parts = self.operator.parts

        # Build displacement reversal map for v2p symmetry
        self._build_displacement_reversal_map()

        if self.debug:
            logger.debug(f"\n{'='*80}")
            logger.debug(f"DEBUG: _make_cache() called (from precomputed data)")
            logger.debug(f"DEBUG: usedNe={self.usedNe}, usedNp={self.usedNp}")
            logger.debug(f"{'='*80}\n")

        # Load p2v data for all time slices (needed for v2p symmetry)
        if self.debug:
            logger.debug("DEBUG: Loading p2v data for v2p symmetry calculation...")
        p2v_full = self.p2v_data.load(self.key)[:]  # [Lt, num_disp, Np, Nc, Ne]

        ret_gamma = []
        ret_elemental_v2v = []
        ret_elemental_v2p = []
        ret_elemental_p2v = []
        ret_elemental_p2p = []
        for i in range(len(parts) // 2):
            ret_gamma.append(gamma(parts[i * 2]))
            elemental_part = parts[i * 2 + 1]

            if self.debug:
                logger.debug(f"\nDEBUG: Processing operatorpart {i}, gamma={parts[i * 2]}")
                logger.debug(f"DEBUG: elemental_part has {len(elemental_part)} terms")

            for j in range(len(elemental_part)):
                elemental_coeff, gaugelink_idx, momentum_idx = elemental_part[j]
                elemental_coeff = complex(elemental_coeff)
                deriv_mom_tuple_v2v = ("v2v", gaugelink_idx, momentum_idx)

                if self.debug:
                    logger.debug(
                        f"\n  DEBUG: Term {j}: coeff={elemental_coeff}, gaugelink_idx={gaugelink_idx}, momentum_idx={momentum_idx}"
                    )

                # v2v: load from elemental (already exists)
                if deriv_mom_tuple_v2v not in cache_v2v:
                    cache_v2v[deriv_mom_tuple_v2v] = self.elemental_data[
                        gaugelink_idx, momentum_idx, :, : self.usedNe, : self.usedNe
                    ]
                    if self.debug:
                        logger.debug(
                            f"  DEBUG: cache_v2v created, shape={cache_v2v[deriv_mom_tuple_v2v].shape}"
                        )

                # Accumulate v2v results
                if j == 0:
                    ret_elemental_v2v.append(
                        elemental_coeff * cache_v2v[deriv_mom_tuple_v2v]
                    )
                else:
                    ret_elemental_v2v[-1] += (
                        elemental_coeff * cache_v2v[deriv_mom_tuple_v2v]
                    )

                # p2v: load and accumulate from precomputed data (following v2v pattern)
                # Note: momentum_idx is ignored for p2v (momentum-independent)
                p2v_data_slice = p2v_full[
                    :, gaugelink_idx, : self.usedNp, :, : self.usedNe
                ]  # [Lt, Np, Nc, Ne]

                if j == 0:
                    ret_elemental_p2v.append(elemental_coeff * p2v_data_slice)
                else:
                    ret_elemental_p2v[-1] += elemental_coeff * p2v_data_slice

                # v2p: compute from p2v using symmetry v2p(disp) = p2v(-disp).transpose(Ne, Np)
                reverse_gaugelink_idx = self._disp_reversal_map[gaugelink_idx]
                # Get p2v data with reversed displacement and transpose for v2p shape
                v2p_data_slice = p2v_full[
                    :, reverse_gaugelink_idx, : self.usedNp, :, : self.usedNe
                ]  # [Lt, Np, Nc, Ne]
                v2p_data_slice = v2p_data_slice.transpose(
                    0, 3, 1, 2
                )  # [Lt, Ne, Np, Nc]

                if j == 0:
                    ret_elemental_v2p.append(elemental_coeff * v2p_data_slice)
                else:
                    ret_elemental_v2p[-1] += elemental_coeff * v2p_data_slice

                # p2p: store reference for lazy loading (keep as is since it's sparse)
                if j == 0:
                    ret_elemental_p2p.append(
                        [(gaugelink_idx, momentum_idx, elemental_coeff)]
                    )
                else:
                    ret_elemental_p2p[-1].append(
                        (gaugelink_idx, momentum_idx, elemental_coeff)
                    )

        if self.debug:
            logger.debug(f"\n{'='*80}")
            logger.debug(f"DEBUG: _make_cache() completed")
            logger.debug(f"DEBUG: Total cache entries created:")
            logger.debug(f"  - cache_v2v: {len(cache_v2v)} entries")
            logger.debug(f"DEBUG: ret_gamma length: {len(ret_gamma)}")
            logger.debug(f"DEBUG: ret_elemental_v2v length: {len(ret_elemental_v2v)}")
            logger.debug(
                f"DEBUG: ret_elemental_v2p length: {len(ret_elemental_v2p)} (pre-computed from p2v symmetry)"
            )
            logger.debug(
                f"DEBUG: ret_elemental_p2v length: {len(ret_elemental_p2v)} (pre-loaded data)"
            )
            logger.debug(
                f"DEBUG: ret_elemental_p2p length: {len(ret_elemental_p2p)} (lazy load instructions)"
            )
            logger.debug(f"{'='*80}\n")

        # Store as tuples for pre-loaded data
        # ret_elemental_v2p contains pre-computed data from p2v symmetry
        # ret_elemental_p2v contains pre-loaded data
        # ret_elemental_p2p contains lazy load instructions (sparse)
        if self.dagger:
            self.cache = (
                contract(
                    "ik,xlk,lj->xij",
                    gamma(8),
                    backend.asarray(ret_gamma).conj(),
                    gamma(8),
                ),
                contract("xtba->xtab", backend.asarray(ret_elemental_v2v).conj()),
                backend.asarray(
                    ret_elemental_v2p
                ),  # Pre-converted reverse displacement for v2p
                backend.asarray(ret_elemental_p2v),  # Direct instructions for p2v
                ret_elemental_p2p,
            )
        else:
            self.cache = (
                backend.asarray(ret_gamma),
                backend.asarray(ret_elemental_v2v),
                backend.asarray(ret_elemental_v2p),
                backend.asarray(ret_elemental_p2v),
                ret_elemental_p2p,
            )

    def get_v2p(self, t):
        """
        Get V2P vertex, pre-computed from P2V using symmetry v2p(disp) ≈ p2v(-disp).transpose(Ne, Np).

        Data is pre-loaded and processed in _make_cache() following the same pattern as v2v.

        Args:
            t: Time index (int or array)

        Returns:
            [Ns, Ns, Ne, Np, Nc] if t is int
            [t_len, Ns, Ns, Ne, Np, Nc] if t is array
        """
        # cache[2] contains pre-computed v2p data [num_parts, Lt, Ne, Np, Nc]
        if isinstance(t, int):
            if self.dagger:
                return contract("xij,xepa->ijepa", self.cache[0], self.cache[2][:, t])
            else:
                return contract("xij,xepa->ijepa", self.cache[0], self.cache[2][:, t])
        else:
            if self.dagger:
                return contract("xij,xtepa->tijepa", self.cache[0], self.cache[2][:, t])
            else:
                return contract("xij,xtepa->tijepa", self.cache[0], self.cache[2][:, t])

    def get_p2v(self, t):
        """
        Get P2V vertex, pre-loaded and processed in _make_cache() following the same pattern as v2v.

        Args:
            t: Time index (int or array)

        Returns:
            [Ns, Ns, Np, Nc, Ne] if t is int
            [t_len, Ns, Ns, Np, Nc, Ne] if t is array
        """
        # cache[3] contains pre-loaded p2v data [num_parts, Lt, Np, Nc, Ne]
        if isinstance(t, int):
            if self.dagger:
                return contract("xij,xpae->ijpae", self.cache[0], self.cache[3][:, t])
            else:
                return contract("xij,xpae->ijpae", self.cache[0], self.cache[3][:, t])
        else:
            if self.dagger:
                return contract("xij,xtpae->tijpae", self.cache[0], self.cache[3][:, t])
            else:
                return contract("xij,xtpae->tijpae", self.cache[0], self.cache[3][:, t])

    def get_p2p(self, t):
        """
        Get P2P vertex, loading from precomputed sparse file on demand.

        Args:
            t: Time index (int or array)

        Returns:
            [Ns, Ns, Np, Nc, Np, Nc] if t is int
            [t_len, Ns, Ns, Np, Nc, Np, Nc] if t is array
        """
        backend = get_backend()
        Nc = 3

        if isinstance(t, int):
            # Load p2p data for this specific time slice (if not already loaded)
            if self.p2p_loaded is None or t not in self.p2p_loaded:
                # Load p2p sparse data for time slice t
                # Get num_momentum from elemental_data shape: [num_disp, num_momentum, Lt, Ne, Ne]
                num_momentum = (
                    self.elemental_data.shape[1]
                    if self.elemental_data is not None
                    else None
                )
                p2p_sparse_list = self.p2p_data.load(
                    self.key, t, num_momentum=num_momentum
                )

                if self.p2p_loaded is None:
                    self.p2p_loaded = {}

                # Accumulate with operator coefficients and convert to dense
                # cache[4] contains p2p instructions
                num_parts = len(self.cache[4])
                result_parts = []

                for part_idx in range(num_parts):
                    dense_array = backend.zeros(
                        (self.usedNp, Nc, self.usedNp, Nc), dtype="<c16"
                    )

                    for gaugelink_idx, momentum_idx, coeff in self.cache[4][part_idx]:
                        # Get the sparse data for this gaugelink
                        # p2p_sparse_list is organized as [disp_idx * num_momentum + momentum_idx]
                        # Get num_momentum from elemental_data shape: [num_disp, num_momentum, Lt, Ne, Ne]
                        num_momentum = (
                            self.elemental_data.shape[1]
                            if self.elemental_data is not None
                            else 1
                        )
                        idx = gaugelink_idx * num_momentum + momentum_idx
                        sparse_data = p2p_sparse_list[idx]

                        if sparse_data["type"] == "identity":
                            # Identity case: add to diagonal
                            for p in range(self.usedNp):
                                for c in range(Nc):
                                    dense_array[p, c, p, c] += coeff
                        else:
                            # Sparse case: fill from coordinate list
                            indices = sparse_data["indices"]
                            values = sparse_data["values"]

                            for i in range(len(indices)):
                                l, r = indices[i]
                                if l < self.usedNp and r < self.usedNp:
                                    dense_array[l, :, r, :] += coeff * backend.asarray(
                                        values[i]
                                    )

                    result_parts.append(dense_array)

                self.p2p_loaded[t] = backend.asarray(result_parts)

            elemental_stack = self.p2p_loaded[t]

            if self.dagger:
                return contract("xij,xpwqz->ijpwqz", self.cache[0], elemental_stack)
            else:
                return contract("xij,xpwqz->ijpwqz", self.cache[0], elemental_stack)
        else:
            raise NotImplementedError("PSP propagator with array t not yet implemented")


class Propagator:
    def __init__(self, perambulator, Lt) -> None:
        self.perambulator = perambulator
        self.perambulator_data = None
        self.key = None
        self.Lt = Lt
        self.cache = None
        self.cache_dagger = None
        self.cached_time = None

    def _release_resources(self):
        self.perambulator_data = None
        self.cache = None
        self.cache_dagger = None
        self.cached_time = None
        gc.collect()
        backend = get_backend()
        if hasattr(backend, "get_default_memory_pool"):
            try:
                backend.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    def release(self):
        self._release_resources()
        self.key = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def load(self, key, usedNe: int = None):
        if self.key != key:
            # self._release_resources()
            self.key = key
            self.usedNe = usedNe
            self.perambulator_data = self.perambulator.load(key)

    def get(self, t_source, t_sink):
        from lattice.insertion.gamma import gamma

        if isinstance(t_source, int) and isinstance(t_sink, int):
            if self.cached_time != t_source and self.cached_time != t_sink:
                self.cache = self.perambulator_data[
                    t_source, :, :, :, : self.usedNe, : self.usedNe
                ]
                self.cache_dagger = contract(
                    "ik,tlkba,lj->tijab", gamma(15), self.cache.conj(), gamma(15)
                )
                self.cached_time = t_source
            if self.cached_time == t_source:
                return self.cache[(t_sink - t_source) % self.Lt]
            else:
                return self.cache_dagger[(t_source - t_sink) % self.Lt]
        elif isinstance(t_source, int):
            if self.cached_time != t_source:
                self.cache = self.perambulator_data[
                    t_source, :, :, :, : self.usedNe, : self.usedNe
                ]
                self.cache_dagger = contract(
                    "ik,tlkba,lj->tijab", gamma(15), self.cache.conj(), gamma(15)
                )
                self.cached_time = t_source
            return self.cache[(t_sink - t_source) % self.Lt]
        elif isinstance(t_sink, int):
            if self.cached_time != t_sink:
                self.cache = self.perambulator_data[
                    t_sink, :, :, :, : self.usedNe, : self.usedNe
                ]
                self.cache_dagger = contract(
                    "ik,tlkba,lj->tijab", gamma(15), self.cache.conj(), gamma(15)
                )
                self.cached_time = t_sink
            return self.cache_dagger[(t_source - t_sink) % self.Lt]
        else:
            raise ValueError("At least t_source or t_sink should be int")


class PropagatorLocal:
    def __init__(self, perambulator, Lt) -> None:
        self.perambulator = perambulator
        self.key = None
        self.Lt = Lt
        self.cache = None

    def load(self, key, usedNe: int = None):
        if self.key != key:
            self.key = key
            self.perambulator_data = self.perambulator.load(key)
            self.usedNe = usedNe
            self._make_cache()

    def _make_cache(self):
        self.cache = self.perambulator_data[0, :, :, :, : self.usedNe, : self.usedNe]
        for t_source in range(1, self.Lt):
            self.cache[t_source] = self.perambulator_data[
                t_source, 0, :, :, : self.usedNe, : self.usedNe
            ]

    def get(self, t_source, t_sink):
        if isinstance(t_source, int):
            assert t_source == t_sink, "You cannot use PropagatorLocal here"
        else:
            assert (t_source == t_sink).all(), "You cannot use PropagatorLocal here"
        return self.cache[t_source]


class PropagatorWithCurrent(Propagator):
    """
    Propagator that supports all four types: VSV, VSP, PSV, PSP.

    This class extends Propagator to support additional propagator types.
    When only vsv is provided, it behaves exactly like Propagator.

    Args:
        vsv: VSV propagator (Perambulator), optional but recommended for compatibility
        vsp: VSP propagator (PropagatorVSP), optional
        psv: PSV propagator (PropagatorPSV), optional
        psp: PSP propagator (PropagatorPSP), optional
        eigenvector: Eigenvector object for high mode projection, optional
        point: PointSource object for high mode projection, optional
        Lt: Temporal extent
        debug: Enable debug output
    """

    def __init__(
        self,
        vsv: Perambulator = None,
        vsp: PropagatorVSP = None,
        psv: PropagatorPSV = None,
        psp: PropagatorPSP = None,
        overlap_matrix: "OverlapMatrix" = None,
        Lt: int = None,
        debug: bool = False,
    ):
        # Initialize parent class with vsv as perambulator if provided
        # If vsv is None, we still need to call super().__init__ with a dummy perambulator
        # But we'll handle this case specially
        if vsv is not None:
            super().__init__(vsv, Lt)
        else:
            # Create a dummy perambulator for parent class initialization
            # This allows the class to work even without vsv
            super().__init__(None, Lt)
        self.vsp_propagator = vsp
        self.psv_propagator = psv
        self.psp_propagator = psp
        self.overlap_matrix = overlap_matrix
        self.debug = debug

        # Caches for loaded data (additional to parent's cache)
        self.vsp_data = None
        self.psv_data = None
        self.psp_data = None
        # Overlap matrix data loaded from file
        self.overlap_matrix_data = None
        # Two-time cache structures for VSP/PSV/PSP (mirror Propagator.get)
        # Naming convention: cache name indicates data source (vsp_data/psv_data/psp_data)
        # _dagger suffix indicates gamma(15) transformation applied
        # VSP data caches: vsp_cache is VSP ordering, vsp_dagger is PSV ordering
        self.vsp_cache = None
        self.vsp_dagger = None
        self.vsp_cached_time = None
        # PSV data caches: psv_cache is PSV ordering, psv_dagger is VSP ordering
        self.psv_cache = None
        self.psv_dagger = None
        self.psv_cached_time = None
        # PSP data caches: psp_cache is original, psp_dagger has left/right swapped
        self.psp_cache = None
        self.psp_dagger = None
        self.psp_cached_time = None
        # Cache for tilde_S PSV (high mode projected PSV)
        # Simple cache for single t_source, value shape: [Lt, Ns, Ns, Np, Nc, Ne]
        # Cache for tilde_S PSV (high mode projected PSV)
        # Simple cache for single t_source, value shape: [Lt, Ns, Ns, Np, Nc, Ne]
        self.tilde_S_psv_cache = None
        self.tilde_S_psv_cached_time = None
        # Dagger version: PSV ordering -> VSP ordering (Ne, Np, Nc)
        self.tilde_S_psv_dagger = None

        # Cache for tilde_S VSP (high mode projected VSP)
        # Simple cache for single t_sink, value shape: [Lt, Ns, Ns, Ne, Np, Nc]
        self.tilde_S_vsp_cache = None
        self.tilde_S_vsp_cached_time = None
        # Dagger version: VSP ordering -> PSV ordering (Np, Nc, Ne)
        self.tilde_S_vsp_dagger = None

        # Cache for tilde_S PSP (high mode projected PSP)
        # Simple cache for single t_source, value shape: [Lt, Ns, Ns, Np_snk, Nc, Np_src, Nc]
        self.tilde_S_psp_cache = None
        self.tilde_S_psp_cached_time = None

    def load(self, key, usedNe: int = None, usedNp: int = None):
        """Load data from all available propagators. Slicing is deferred to get-time (like parent)."""
        if self.debug:
            log_gpu_memory(f"PropagatorWithCurrent.load(before, key={key})")
            logger.debug(f"\n{'='*80}")
            logger.debug(f"PropagatorWithCurrent.load() called")
            logger.debug(f"{'='*80}")
            logger.debug(f"  key: {key}")
            logger.debug(f"  usedNe: {usedNe}")
            logger.debug(f"  usedNp: {usedNp}")
            logger.debug(f"  self.key (current): {self.key}")
            logger.debug(f"  Available propagators:")
            logger.debug(f"    VSV (perambulator): {self.perambulator is not None}")
            logger.debug(f"    VSP (vsp_propagator): {self.vsp_propagator is not None}")
            logger.debug(f"    PSV (psv_propagator): {self.psv_propagator is not None}")
            logger.debug(f"    PSP (psp_propagator): {self.psp_propagator is not None}")

        if self.key != key:
            if self.debug:
                logger.debug(f"\n  Key changed, loading new data...")

            # Load VSV via parent. Parent defers slicing to get().
            if self.perambulator is not None:
                if self.debug:
                    logger.debug(f"  Loading VSV via parent...")
                    log_gpu_memory("load_VSV(before)")
                try:
                    super().load(key, usedNe)
                    if self.debug:
                        log_gpu_memory("load_VSV(after)")
                        if (
                            hasattr(self, "perambulator_data")
                            and self.perambulator_data is not None
                        ):
                            logger.debug(
                                f"    VSV loaded successfully, shape: {self.perambulator_data.shape}"
                            )
                        else:
                            logger.debug(f"    VSV loaded, but perambulator_data is None")
                except Exception as e:
                    if self.debug:
                        logger.debug(f"    ERROR loading VSV: {e}")
                    raise
            else:
                if self.debug:
                    logger.debug(f"  Skipping VSV (perambulator is None)")

            if self.vsp_propagator is not None:
                if self.debug:
                    logger.debug(f"  Loading VSP...")
                    log_gpu_memory("load_VSP(before)")
                try:
                    self.vsp_data = self.vsp_propagator.load(key)
                    if self.debug:
                        log_gpu_memory("load_VSP(after)")
                        if self.vsp_data is not None:
                            logger.debug(
                                f"    VSP loaded successfully, shape: {self.vsp_data.shape}"
                            )
                        else:
                            logger.debug(f"    VSP load returned None")
                except Exception as e:
                    if self.debug:
                        logger.debug(f"    ERROR loading VSP: {e}")
                    raise
            else:
                if self.debug:
                    logger.debug(f"  Skipping VSP (vsp_propagator is None)")

            if self.psv_propagator is not None:
                if self.debug:
                    logger.debug(f"  Loading PSV...")
                    log_gpu_memory("load_PSV(before)")
                try:
                    self.psv_data = self.psv_propagator.load(key)
                    if self.debug:
                        log_gpu_memory("load_PSV(after)")
                        if self.psv_data is not None:
                            logger.debug(
                                f"    PSV loaded successfully, shape: {self.psv_data.shape}"
                            )
                        else:
                            logger.debug(f"    PSV load returned None")
                except Exception as e:
                    if self.debug:
                        logger.debug(f"    ERROR loading PSV: {e}")
                    raise
            else:
                if self.debug:
                    logger.debug(f"  Skipping PSV (psv_propagator is None)")

            if self.psp_propagator is not None:
                if self.debug:
                    logger.debug(f"  Loading PSP...")
                    log_gpu_memory("load_PSP(before)")
                try:
                    self.psp_data = self.psp_propagator.load(key)
                    if self.debug:
                        log_gpu_memory("load_PSP(after)")
                        if self.psp_data is not None:
                            logger.debug(
                                f"    PSP loaded successfully, shape: {self.psp_data.shape}"
                            )
                        else:
                            logger.debug(f"    PSP load returned None")
                except Exception as e:
                    if self.debug:
                        logger.debug(f"    ERROR loading PSP: {e}")
                    raise
            else:
                if self.debug:
                    logger.debug(f"  Skipping PSP (psp_propagator is None)")

            # Load overlap matrix data from file
            if self.overlap_matrix is not None:
                if self.debug:
                    logger.debug(f"  Loading overlap matrix from file...")
                    log_gpu_memory("load_overlap_matrix(before)")
                self.overlap_matrix_data = self.overlap_matrix.load(key)[:]
                if self.debug:
                    logger.debug(
                        f"    overlap_matrix_data.shape: {self.overlap_matrix_data.shape}"
                    )
                    log_gpu_memory("load_overlap_matrix(after)")
            else:
                raise ValueError("overlap_matrix must be provided")

            # Clear high mode caches when key changes
            if self.debug:
                log_gpu_memory("clear_caches(before)")
            # tilde_S_psv is simple cache for single t_source
            self.tilde_S_psv_cache = None
            self.tilde_S_psv_cached_time = None
            # tilde_S_vsp is simple cache for single t_sink
            self.tilde_S_vsp_cache = None
            self.tilde_S_vsp_cached_time = None
            # tilde_S_psp is simple cache for single t_source
            self.tilde_S_psp_cache = None
            self.tilde_S_psp_cached_time = None
            if self.debug:
                log_gpu_memory("clear_caches(after)")

            # Update key and usedNe/usedNp
            self.key = key
            self.usedNe = usedNe
            self.usedNp = usedNp

            if self.debug:
                logger.debug(f"\n  Load completed successfully")
                logger.debug(f"  Updated self.key to: {self.key}")
                logger.debug(f"  Updated self.usedNe to: {self.usedNe}")
                logger.debug(f"  Updated self.usedNp to: {self.usedNp}")
                logger.debug(f"{'='*80}\n")
                log_gpu_memory(f"PropagatorWithCurrent.load(after, key={key})")
        else:
            if self.debug:
                logger.debug(f"  Key unchanged, skipping load")
                logger.debug(f"{'='*80}\n")
                log_gpu_memory(f"PropagatorWithCurrent.load(skipped, key={key})")

    def get(self, t_source, t_sink):
        """
        Get VSV propagator (standard perambulator) for given source/sink times.

        Compatible with standard Propagator interface; slicing occurs in parent get().
        Only raises error if data is not available.
        """
        if self.debug:
            logger.debug(f"\nPropagatorWithCurrent.get(VSV) called:")
            logger.debug(f"  t_source: {t_source} (type: {type(t_source).__name__})")
            logger.debug(f"  t_sink: {t_sink} (type: {type(t_sink).__name__})")
            logger.debug(
                f"  perambulator_data available: {self.perambulator_data is not None}"
            )

        if self.perambulator_data is not None:
            # Delegate to parent which handles caching and usedNe slicing
            result = super().get(t_source, t_sink)
            if self.debug:
                logger.debug(f"  Result shape: {result.shape}")
            return result
        else:
            if self.debug:
                logger.debug(f"  ERROR: VSV propagator not available!")
            raise ValueError(
                "VSV propagator not provided but is required for this diagram"
            )

    def _release_resources(self):
        """Release resources for all propagator types."""
        # Release parent resources
        super()._release_resources()
        # Release additional resources
        self.vsp_data = None
        self.psv_data = None
        self.psp_data = None
        self.overlap_matrix_data = None
        # Clear high mode caches
        self.tilde_S_psv_cache = None
        self.tilde_S_psv_cached_time = None
        self.tilde_S_psv_dagger = None
        self.tilde_S_vsp_cache = None
        self.tilde_S_vsp_cached_time = None
        self.tilde_S_vsp_dagger = None
        self.tilde_S_psp_cache = None
        self.tilde_S_psp_cached_time = None

    def _apply_gamma_on_spin(self, array_with_spin_first_two_axes):
        """
        Apply gamma(15) on both spin indices of an array whose first three axes are [Lt, Ns, Ns, ...].
        Conjugates the input as part of dagger operation.
        Returns the transformed array with the same shape and tail axes ordering preserved.
        """
        from lattice.insertion.gamma import gamma

        backend = get_backend()
        g15 = gamma(15)
        arr = array_with_spin_first_two_axes
        Lt_local, Ns_left, Ns_right = (
            int(arr.shape[0]),
            int(arr.shape[1]),
            int(arr.shape[2]),
        )
        tail_shape = tuple(arr.shape[3:])
        # Flatten tail for two-step matmul to avoid verbose einsum strings
        arr_flat = arr.conj().reshape((Lt_local, Ns_left, Ns_right, -1))
        left_applied = contract("ik,tklr->tilr", g15, arr_flat)
        both_applied = contract("tilr,lj->tijr", left_applied, g15)
        return both_applied.reshape((Lt_local, Ns_left, Ns_right) + tail_shape)

    def _dagger_vsp(self, vsp_block):
        """
        Apply dagger (gamma(15) conjugate) to VSP data.

        Input: VSP tail order [..., Ne, Np, Nc]
        Output: PSV tail order [..., Np, Nc, Ne]
        """
        spun = self._apply_gamma_on_spin(vsp_block)  # keeps same shape
        # Move tail axes from (Ne, Np, Nc) -> (Np, Nc, Ne)
        return spun[:, :, :, :, 1, 2, 0] if False else spun.transpose(0, 1, 2, 4, 5, 3)

    def _dagger_psv(self, psv_block):
        """
        Apply dagger (gamma(15) conjugate) to PSV data.

        Input: PSV tail order [..., Np, Nc, Ne]
        Output: VSP tail order [..., Ne, Np, Nc]
        """
        spun = self._apply_gamma_on_spin(psv_block)
        # Move tail axes from (Np, Nc, Ne) -> (Ne, Np, Nc)
        return spun.transpose(0, 1, 2, 5, 3, 4)

    def _dagger_psp(self, psp_block):
        """
        Apply dagger (gamma(15) conjugate) to PSP data.

        Input: PSP tail order [..., Np_snk, Nc, Np_src, Nc]
        Output: Swapped ordering [..., Np_src, Nc, Np_snk, Nc]
        """
        spun = self._apply_gamma_on_spin(psp_block)
        # Swap the two point axes (positions 3 and 5 in tail)
        return spun.transpose(0, 1, 2, 5, 4, 3, 6)

    def get_VSP(self, t_source, t_sink, cache=True):
        """
        Get VSP propagator data with two-time interface mirroring Propagator.get.

        Args:
            t_source: Source time (int or array-like)
            t_sink: Sink time (int or array-like)
            cache: Whether to cache internal timeslice array (default True). Set False to avoid caching.

        Returns:
            If both times are int: [Ns, Ns, Ne, Np, Nc] or [Ns, Ns, Np, Nc, Ne] (if dagger->PSV)
            If one time is int and the other is array-like: [t, Ns, Ns, ...] with same tail ordering
        """
        if self.debug:
            logger.debug(f"\nPropagatorWithCurrent.get_VSP() called:")
            logger.debug(f"  t_source: {t_source} (type: {type(t_source).__name__})")
            logger.debug(f"  t_sink: {t_sink} (type: {type(t_sink).__name__})")
            logger.debug(f"  vsp_data available: {self.vsp_data is not None}")
            if self.vsp_data is not None:
                logger.debug(f"  vsp_data.shape: {self.vsp_data.shape}")
            logger.debug(f"  usedNe: {getattr(self, 'usedNe', None)}")
            logger.debug(f"  usedNp: {getattr(self, 'usedNp', None)}")

        # Populate caches based on which time is int (anchor choice same as Propagator.get)
        if isinstance(t_source, int) and isinstance(t_sink, int):
            # Check if we have cached data
            if self.vsp_cached_time == t_source:
                out = self.vsp_cache[(t_sink - t_source) % self.Lt]
                if self.debug:
                    logger.debug(f"    VSP get(two-int, cached): shape={out.shape}")
                return out
            elif self.psv_cached_time == t_sink:
                out = self.psv_dagger[(t_source - t_sink) % self.Lt]
                if self.debug:
                    logger.debug(f"    VSP get(two-int, cached dagger): shape={out.shape}")
                return out
            else:
                # Need to compute new data
                if self.vsp_data is None:
                    if self.debug:
                        logger.debug(f"  ERROR: VSP propagator not available!")
                    raise ValueError(
                        "VSP propagator not provided but is required for this diagram"
                    )

                vsp_cache_local = self.vsp_data[
                    t_source,
                    ...,
                    : self.usedNe if self.usedNe is not None else None,
                    : self.usedNp if self.usedNp is not None else None,
                    :,
                ]
                if self.debug:
                    logger.debug(f"    after slice: {vsp_cache_local.shape}")

                if cache:
                    self.vsp_cache = vsp_cache_local
                    self.vsp_dagger = self._dagger_vsp(self.vsp_cache)
                    self.vsp_cached_time = t_source
                    out = self.vsp_cache[(t_sink - t_source) % self.Lt]
                else:
                    out = vsp_cache_local[(t_sink - t_source) % self.Lt]

                if self.debug:
                    logger.debug(f"    final output: {out.shape}")
                    logger.debug(
                        f"    VSP get(two-int, {'cached' if cache else 'uncached'}): shape={out.shape}"
                    )
                return out
        elif isinstance(t_source, int):
            # Check if we have cached data
            if self.vsp_cached_time == t_source:
                out = self.vsp_cache[(t_sink - t_source) % self.Lt]
                if self.debug:
                    logger.debug(f"    VSP get(tsrc-int, cached): shape={out.shape}")
                return out
            else:
                # Need to compute new data
                if self.vsp_data is None:
                    if self.debug:
                        logger.debug(f"  ERROR: VSP propagator not available!")
                    raise ValueError(
                        "VSP propagator not provided but is required for this diagram"
                    )

                vsp_cache_local = self.vsp_data[
                    t_source,
                    ...,
                    : self.usedNe if self.usedNe is not None else None,
                    : self.usedNp if self.usedNp is not None else None,
                    :,
                ]
                if self.debug:
                    logger.debug(f"    after slice: {vsp_cache_local.shape}")

                if cache:
                    self.vsp_cache = vsp_cache_local
                    self.vsp_dagger = self._dagger_vsp(self.vsp_cache)
                    self.vsp_cached_time = t_source
                    out = self.vsp_cache[(t_sink - t_source) % self.Lt]
                else:
                    out = vsp_cache_local[(t_sink - t_source) % self.Lt]

                if self.debug:
                    logger.debug(f"    final output: {out.shape}")
                    logger.debug(
                        f"    VSP get(tsrc-int, {'cached' if cache else 'uncached'}): shape={out.shape}"
                    )
                return out
        elif isinstance(t_sink, int):
            # Need dagger: check if PSV cache available
            if self.psv_cached_time == t_sink:
                out = self.psv_dagger[(t_source - t_sink) % self.Lt]
                if self.debug:
                    logger.debug(f"    VSP get(tsink-int, cached dagger): shape={out.shape}")
                return out
            else:
                # Need to compute new data
                if self.psv_data is None:
                    if self.debug:
                        logger.debug(
                            f"  ERROR: PSV propagator not available (needed for VSP dagger)!"
                        )
                    raise ValueError(
                        "PSV propagator not provided but is required for this diagram"
                    )
                # Get and slice in one step: [Lt, Ns, Ns, Np, Nc, Ne]
                psv_cache_local = self.psv_data[
                    t_sink,
                    ...,
                    : self.usedNp if self.usedNp is not None else None,
                    :,
                    : self.usedNe if self.usedNe is not None else None,
                ]

                if cache:
                    self.psv_cache = psv_cache_local
                    self.psv_dagger = self._dagger_psv(self.psv_cache)
                    self.psv_cached_time = t_sink
                    out = self.psv_dagger[(t_source - t_sink) % self.Lt]
                else:
                    psv_dagger_local = self._dagger_psv(psv_cache_local)
                    out = psv_dagger_local[(t_source - t_sink) % self.Lt]

                if self.debug:
                    logger.debug(
                        f"    VSP get(tsink-int, {'cached' if cache else 'uncached'} dagger): shape={out.shape}"
                    )
                return out
        else:
            raise ValueError("At least t_source or t_sink should be int")

    def get_PSV(self, t_source, t_sink, cache=True):
        """
        Get PSV (point->eigen) propagator, math: S_{xa,i} = <eta_x,a | S | xi_i>.

        Args:
            t_source: Source time (int or array-like)
            t_sink: Sink time (int or array-like)
            cache: Whether to cache internal timeslice array (default True). Set False to avoid caching.

        Returns:
            If both times are int: [Ns, Ns, Np, Nc, Ne]
            If one time is int and the other is array-like: [t, Ns, Ns, Np, Nc, Ne]
        """
        if self.debug:
            logger.debug(f"\nPropagatorWithCurrent.get_PSV() called:")
            logger.debug(f"  t_source: {t_source} (type: {type(t_source).__name__})")
            logger.debug(f"  t_sink: {t_sink} (type: {type(t_sink).__name__})")
            logger.debug(f"  psv_data available: {self.psv_data is not None}")
            if self.psv_data is not None:
                logger.debug(f"  psv_data.shape: {self.psv_data.shape}")
            logger.debug(f"  usedNe: {getattr(self, 'usedNe', None)}")
            logger.debug(f"  usedNp: {getattr(self, 'usedNp', None)}")

        if isinstance(t_source, int) and isinstance(t_sink, int):
            # Check if we have cached data
            if self.psv_cached_time == t_source:
                out = self.psv_cache[(t_sink - t_source) % self.Lt]
                if self.debug:
                    logger.debug(f"    PSV get(two-int, cached): shape={out.shape}")
                return out
            elif self.vsp_cached_time == t_sink:
                out = self.vsp_dagger[(t_source - t_sink) % self.Lt]
                if self.debug:
                    logger.debug(f"    PSV get(two-int, cached dagger): shape={out.shape}")
                return out
            else:
                # Need to compute new data
                if self.psv_data is None:
                    if self.debug:
                        logger.debug(f"  ERROR: PSV propagator not available!")
                    raise ValueError(
                        "PSV propagator not provided but is required for this diagram"
                    )

                psv_cache_local = self.psv_data[
                    t_source,
                    ...,
                    : self.usedNp if self.usedNp is not None else None,
                    :,
                    : self.usedNe if self.usedNe is not None else None,
                ]
                if self.debug:
                    logger.debug(f"    after slice: {psv_cache_local.shape}")

                if cache:
                    self.psv_cache = psv_cache_local
                    self.psv_dagger = self._dagger_psv(self.psv_cache)
                    self.psv_cached_time = t_source
                    out = self.psv_cache[(t_sink - t_source) % self.Lt]
                else:
                    out = psv_cache_local[(t_sink - t_source) % self.Lt]

                if self.debug:
                    logger.debug(f"    final output: {out.shape}")
                    logger.debug(
                        f"    PSV get(two-int, {'cached' if cache else 'uncached'}): shape={out.shape}"
                    )
                return out
        elif isinstance(t_source, int):
            # Check if we have cached data
            if self.psv_cached_time == t_source:
                out = self.psv_cache[(t_sink - t_source) % self.Lt]
                if self.debug:
                    logger.debug(f"    PSV get(tsrc-int, cached): shape={out.shape}")
                return out
            else:
                # Need to compute new data
                if self.psv_data is None:
                    if self.debug:
                        logger.debug(f"  ERROR: PSV propagator not available!")
                    raise ValueError(
                        "PSV propagator not provided but is required for this diagram"
                    )

                psv_cache_local = self.psv_data[
                    t_source,
                    ...,
                    : self.usedNp if self.usedNp is not None else None,
                    :,
                    : self.usedNe if self.usedNe is not None else None,
                ]
                if self.debug:
                    logger.debug(f"    after slice: {psv_cache_local.shape}")

                if cache:
                    self.psv_cache = psv_cache_local
                    self.psv_dagger = self._dagger_psv(self.psv_cache)
                    self.psv_cached_time = t_source
                    out = self.psv_cache[(t_sink - t_source) % self.Lt]
                else:
                    out = psv_cache_local[(t_sink - t_source) % self.Lt]

                if self.debug:
                    logger.debug(f"    final output: {out.shape}")
                    logger.debug(
                        f"    PSV get(tsrc-int, {'cached' if cache else 'uncached'}): shape={out.shape}"
                    )
                return out
        elif isinstance(t_sink, int):
            # Need dagger: check if VSP cache available
            if self.vsp_cached_time == t_sink:
                out = self.vsp_dagger[(t_source - t_sink) % self.Lt]
                if self.debug:
                    logger.debug(f"    PSV get(tsink-int, cached dagger): shape={out.shape}")
                return out
            else:
                # Need to compute new data
                if self.vsp_data is None:
                    if self.debug:
                        logger.debug(
                            f"  ERROR: VSP propagator not available (needed for PSV dagger)!"
                        )
                    raise ValueError(
                        "VSP propagator not provided but is required for this diagram"
                    )
                # Get and slice in one step: [Lt, Ns, Ns, Ne, Np, Nc]
                vsp_cache_local = self.vsp_data[
                    t_sink,
                    ...,
                    : self.usedNe if self.usedNe is not None else None,
                    : self.usedNp if self.usedNp is not None else None,
                    :,
                ]

                if cache:
                    self.vsp_cache = vsp_cache_local
                    self.vsp_dagger = self._dagger_vsp(self.vsp_cache)
                    self.vsp_cached_time = t_sink
                    out = self.vsp_dagger[(t_source - t_sink) % self.Lt]
                else:
                    vsp_dagger_local = self._dagger_vsp(vsp_cache_local)
                    out = vsp_dagger_local[(t_source - t_sink) % self.Lt]

                if self.debug:
                    logger.debug(
                        f"    PSV get(tsink-int, {'cached' if cache else 'uncached'} dagger): shape={out.shape}"
                    )
                return out
        else:
            raise ValueError("At least t_source or t_sink should be int")

    def get_PSP(self, t_source, t_sink, cache=True):
        """
        Get PSP (point->point) propagator, math: S_{xa,yb} = <eta_x,a | S | eta_y,b>.

        Args:
            t_source: Source time (int or array-like)
            t_sink: Sink time (int or array-like)
            cache: Whether to cache internal timeslice array (default True). Set False to avoid caching.

        Returns:
            If both times are int: [Ns, Ns, Np_snk, Nc, Np_src, Nc]
            If one time is int and the other is array-like: [t, Ns, Ns, Np_snk, Nc, Np_src, Nc]
        """
        if self.debug:
            logger.debug(f"\nPropagatorWithCurrent.get_PSP() called:")
            logger.debug(f"  t_source: {t_source} (type: {type(t_source).__name__})")
            logger.debug(f"  t_sink: {t_sink} (type: {type(t_sink).__name__})")
            logger.debug(f"  psp_data available: {self.psp_data is not None}")
            if self.psp_data is not None:
                logger.debug(f"  psp_data.shape: {self.psp_data.shape}")
            logger.debug(f"  usedNp: {getattr(self, 'usedNp', None)}")

        if self.psp_data is None:
            if self.debug:
                logger.debug(f"  ERROR: PSP propagator not available!")
            raise ValueError(
                "PSP propagator not provided but is required for this diagram"
            )
        if isinstance(t_source, int) and isinstance(t_sink, int):
            # Check if we have cached data
            if self.psp_cached_time == t_source:
                out = self.psp_cache[(t_sink - t_source) % self.Lt]
                if self.debug:
                    logger.debug(f"    PSP get(two-int, cached): shape={out.shape}")
                return out
            elif self.psp_cached_time == t_sink:
                out = self.psp_dagger[(t_source - t_sink) % self.Lt]
                if self.debug:
                    logger.debug(f"    PSP get(two-int, cached dagger): shape={out.shape}")
                return out
            else:
                psp_cache_local = self.psp_data[
                    t_source
                ]  # [Lt, Ns, Ns, Np_snk, Nc, Np_src, Nc]
                if getattr(self, "usedNp", None) is not None:
                    psp_cache_local = psp_cache_local[
                        ..., : self.usedNp, :, : self.usedNp, :
                    ]
                    if self.debug:
                        logger.debug(f"    after slice: {psp_cache_local.shape}")

                if cache:
                    # Save to cache
                    self.psp_cache = psp_cache_local
                    self.psp_dagger = self._dagger_psp(self.psp_cache)
                    self.psp_cached_time = t_source
                    out = self.psp_cache[(t_sink - t_source) % self.Lt]
                else:
                    # Don't save to cache, just compute result
                    out = psp_cache_local[(t_sink - t_source) % self.Lt]

                if self.debug:
                    logger.debug(f"    final output: {out.shape}")
                    logger.debug(
                        f"    PSP get(two-int, {'cached' if cache else 'uncached'}): shape={out.shape}"
                    )
                return out
        elif isinstance(t_source, int):
            # Check if we have cached data
            if self.psp_cached_time == t_source:
                out = self.psp_cache[(t_sink - t_source) % self.Lt]
                if self.debug:
                    logger.debug(f"    PSP get(tsrc-int, cached): shape={out.shape}")
                return out
            else:
                psp_cache_local = self.psp_data[t_source]
                if getattr(self, "usedNp", None) is not None:
                    psp_cache_local = psp_cache_local[
                        ..., : self.usedNp, :, : self.usedNp, :
                    ]
                    if self.debug:
                        logger.debug(f"    after slice: {psp_cache_local.shape}")

                if cache:
                    self.psp_cache = psp_cache_local
                    self.psp_dagger = self._dagger_psp(self.psp_cache)
                    self.psp_cached_time = t_source
                    out = self.psp_cache[(t_sink - t_source) % self.Lt]
                else:
                    out = psp_cache_local[(t_sink - t_source) % self.Lt]

                if self.debug:
                    logger.debug(f"    final output: {out.shape}")
                    logger.debug(
                        f"    PSP get(tsrc-int, {'cached' if cache else 'uncached'}): shape={out.shape}"
                    )
                return out
        elif isinstance(t_sink, int):
            # Check if we have cached data
            if self.psp_cached_time == t_sink:
                out = self.psp_dagger[(t_source - t_sink) % self.Lt]
                if self.debug:
                    logger.debug(f"    PSP get(tsink-int, cached dagger): shape={out.shape}")
                return out
            else:
                # Need to compute new data
                if self.debug:
                    logger.debug(f"  PSP shape process (dagger):")
                    logger.debug(
                        f"    raw psp_data[t_sink={t_sink}]: {self.psp_data[t_sink].shape}"
                    )
                psp_cache_local = self.psp_data[t_sink]
                if getattr(self, "usedNp", None) is not None:
                    psp_cache_local = psp_cache_local[
                        ..., : self.usedNp, :, : self.usedNp, :
                    ]
                    if self.debug:
                        logger.debug(f"    after slice: {psp_cache_local.shape}")
                psp_dagger_local = (
                    self._dagger_psp(psp_cache_local) if not cache else None
                )
                if self.debug and psp_dagger_local is not None:
                    logger.debug(f"    after dagger: {psp_dagger_local.shape}")

                if cache:
                    self.psp_cache = psp_cache_local
                    self.psp_dagger = self._dagger_psp(self.psp_cache)
                    self.psp_cached_time = t_sink
                    out = self.psp_dagger[(t_source - t_sink) % self.Lt]
                else:
                    out = psp_dagger_local[(t_source - t_sink) % self.Lt]

                if self.debug:
                    logger.debug(f"    final output: {out.shape}")
                    logger.debug(
                        f"    PSP get(tsink-int, {'cached' if cache else 'uncached'} dagger): shape={out.shape}"
                    )
                return out
        else:
            raise ValueError("At least t_source or t_sink should be int")

    def get_VSP_highmode(self, t_source, t_sink, usedNe_source=None):
        """
        Get VSP high-mode (eigen->point, projected) propagator.

        Math mapping:
          - unprojected: S_{i,xa} = <xi_i | S | eta_x,a>
          - projected:   \tilde{S}_{i,xa} = S_{i,xa} - \sum_j S_{i,j} M_{jx,a}^*

        Only applies projection when usedNe > 0. When usedNe == 0, returns unprojected VSP.

        Args:
            t_source: Source time
            t_sink: Sink time
            usedNe_source: Number of eigenvectors for source (right) end (default: self.usedNe)
                          When != self.usedNe, no caching is performed for highmode result.

        Returns:
            Same shape as get_VSP: [Ns, Ns, Ne, Np, Nc] or [t, Ns, Ns, Ne, Np, Nc]
        """
        # Use self.usedNe if not specified
        if usedNe_source is None:
            usedNe_source = self.usedNe

        # If usedNe_source == 0, return unprojected
        if usedNe_source == 0:
            return self.get_VSP(t_source, t_sink)

        log_gpu_memory(
            f"get_VSP_highmode(before, t_source={t_source}, t_sink={t_sink})"
        )
        if self.debug:
            logger.debug(f"\nget_VSP_highmode() called:")
            logger.debug(f"  t_source: {t_source}, t_sink: {t_sink}")
            logger.debug(f"  usedNe_source: {usedNe_source}, self.usedNe: {self.usedNe}")

        backend = get_backend()

        # Determine caching strategy
        should_cache_highmode = usedNe_source == self.usedNe
        should_cache_unprojected = usedNe_source != self.usedNe
        is_single_time = isinstance(t_source, int) and isinstance(t_sink, int)
        if is_single_time:
            should_cache_highmode = False
            should_cache_unprojected = True

        # Check highmode cache (only when should_cache_highmode and is_single_time)
        if usedNe_source == self.usedNe:
            if is_single_time:
                t_rel = (t_sink - t_source) % self.Lt
                if self.tilde_S_vsp_cached_time == t_source:
                    t_rel = (t_sink - t_source) % self.Lt
                    return self.tilde_S_vsp_cache[t_rel]
                elif self.tilde_S_psv_cached_time == t_sink:
                    t_rel = (t_source - t_sink) % self.Lt
                    return self.tilde_S_psv_dagger[t_rel]
            else:
                if isinstance(t_source, int):
                    t_rel = (t_sink - t_source) % self.Lt
                    if self.tilde_S_vsp_cached_time == t_source:
                        return self.tilde_S_vsp_cache[t_rel]
                else:
                    t_rel = (t_source - t_sink) % self.Lt
                    if self.tilde_S_vsp_cached_time == t_sink:
                        return self.tilde_S_psv_dagger[t_rel]
        # Get original VSP: S_{i,xa}
        # Cache unprojected when usedNe != self.usedNe (because we won't cache highmode)
        # Don't cache when usedNe == self.usedNe (because highmode result will be cached)
        S_vsp = self.get_VSP(t_source, t_sink, cache=should_cache_unprojected)
        # Slice to usedNe_source if needed
        if usedNe_source != self.usedNe:
            S_vsp = S_vsp[..., :usedNe_source, :, :]

        # Get VSV: S_{i,j}
        S_vsv = self.get(t_source, t_sink)

        # Get overlap matrix M (full) and slice to usedNe_source
        M_full = self.overlap_matrix_data[
            :, : self.usedNe, : self.usedNp, :
        ]  # [Lt, Ne, Np, Nc]
        M = M_full[:, :usedNe_source, :, :] if usedNe_source != self.usedNe else M_full
        M_conj = M.conj()

        if is_single_time:
            M_conj_t = M_conj[t_source]  # [Ne, Np, Nc]

            # Compute: sum_j S_{i,j} M_{jx,a}^*
            # S_vsv: [Ns_snk, Ns_src, Ne_i, Ne_j]
            # M_conj_t: [Ne_j, Np_x, Nc_c] where M_{jx,c}^* = <xi_j| eta_x,c>
            #   Index order: j (Ne, 0th), x (Np, 1st), c (Nc, 2nd)
            # Result: [Ns_snk, Ns_src, Ne_i, Np_x, Nc_c]
            correction = contract(
                "abij,jxc->abixc", S_vsv[:, :, :usedNe_source, :usedNe_source], M_conj_t
            )

            # tilde{S} = S - correction
            tilde_S = S_vsp - correction
            log_gpu_memory(f"get_VSP_highmode(after, single_time)")
            return tilde_S
        else:
            # Multi-time case
            if isinstance(t_source, int):
                t_rel = (backend.asarray(t_sink) - t_source) % self.Lt
                M_conj_t = M_conj[t_source]  # [Ne, Np, Nc]
                correction = contract(
                    "tabij,jxc->tabixc",
                    S_vsv[:, :, :, :usedNe_source, :usedNe_source],
                    M_conj_t,
                )
                tilde_S = S_vsp - correction
                if should_cache_highmode:
                    if self.debug:
                        logger.debug(f"caching full tilde_S_vsp for t_source={t_source}")
                    self.tilde_S_vsp_cache = tilde_S
                    self.tilde_S_vsp_dagger = self._dagger_vsp(tilde_S)
                    self.tilde_S_vsp_cached_time = t_source
                log_gpu_memory(f"get_VSP_highmode(after, multi_time, t_source=int)")
                return tilde_S[t_rel]
            else:  # t_sink is int
                t_rel = (backend.asarray(t_source) - t_sink) % self.Lt
                M_conj_t = M_conj[t_rel]  # [Lt, Ne, Np, Nc]
                correction = contract(
                    "tabij,tjxc->tabixc",
                    S_vsv[:, :, :, :usedNe_source, :usedNe_source],
                    M_conj_t,
                )
                tilde_S = S_vsp - correction
                if should_cache_highmode:
                    if self.debug:
                        logger.debug(f"caching full tilde_S_vsp for t_source={t_source}")
                    self.tilde_S_psv_dagger = tilde_S
                    self.tilde_S_psv_cache = self._dagger_vsp(tilde_S)
                    self.tilde_S_psv_cached_time = t_sink
                    return self.tilde_S_psv_dagger
                if self.debug:
                    logger.debug(f"  tilde_S_vsp shape: {tilde_S.shape}")

                log_gpu_memory(f"get_VSP_highmode(after, multi_time, t_sink=int)")
                return tilde_S[t_rel]

    def get_PSV_highmode(self, t_source, t_sink, usedNe_sink=None):
        """
        Get PSV high-mode (point->eigen, projected) propagator.

        Math mapping:
          - unprojected: S_{xa,i} = <eta_x,a | S | xi_i>
          - projected:   \tilde{S}_{xa,i} = S_{xa,i} - \sum_j M_{xj,a} S_{j,i}

        Only applies projection when usedNe > 0. When usedNe == 0, returns unprojected PSV.
        Caches per t_source only when usedNe_sink == self.usedNe.

        Args:
            t_source: Source time
            t_sink: Sink time
            usedNe_sink: Number of eigenvectors for sink (left) end (default: self.usedNe)
                        When != self.usedNe, no caching is performed for highmode result.

        Returns:
            Same shape as get_PSV: [Ns, Ns, Np, Nc, Ne] or [t, Ns, Ns, Np, Nc, Ne]
        """
        # Use self.usedNe if not specified
        if usedNe_sink is None:
            usedNe_sink = self.usedNe

        # If usedNe_sink == 0, return unprojected
        if usedNe_sink == 0:
            return self.get_PSV(t_source, t_sink)

        log_gpu_memory(
            f"get_PSV_highmode(before, t_source={t_source}, t_sink={t_sink})"
        )
        if self.debug:
            logger.debug(f"\nget_PSV_highmode() called:")
            logger.debug(f"  t_source: {t_source}, t_sink: {t_sink}")
            logger.debug(f"  usedNe_sink: {usedNe_sink}, self.usedNe: {self.usedNe}")

        backend = get_backend()

        # Determine caching strategy
        # Cache highmode only when usedNe == self.usedNe
        should_cache_highmode = usedNe_sink == self.usedNe
        # Cache unprojected when usedNe != self.usedNe (won't cache highmode)
        should_cache_unprojected = usedNe_sink != self.usedNe
        is_single_time = isinstance(t_source, int) and isinstance(t_sink, int)
        if is_single_time:
            should_cache_highmode = False
            should_cache_unprojected = True

        # Check cache only if should_cache_highmode and is_single_time
        if usedNe_sink == self.usedNe:
            if is_single_time:
                if self.tilde_S_psv_cached_time == t_source:
                    t_rel = (t_sink - t_source) % self.Lt
                    if self.debug:
                        logger.debug(f"  Using cached tilde_S_psv")
                    return self.tilde_S_psv_cache[t_rel]
                elif self.tilde_S_vsp_cached_time == t_sink:
                    t_rel = (t_source - t_sink) % self.Lt
                    if self.debug:
                        logger.debug(f"  Using cached tilde_S_vsp_dagger")
                    t_rel = (t_source - t_sink) % self.Lt
                    return self.tilde_S_vsp_dagger[t_rel]
            else:
                if isinstance(t_source, int):
                    t_rel = (t_sink - t_source) % self.Lt
                    if self.tilde_S_psv_cached_time == t_source:
                        if self.debug:
                            logger.debug(f"  Using cached tilde_S_psv")
                        return self.tilde_S_psv_cache[t_rel]
                else:
                    t_rel = (t_source - t_sink) % self.Lt
                    if self.tilde_S_psv_cached_time == t_sink:
                        if self.debug:
                            logger.debug(f"  Using cached tilde_S_vsp_dagger")
                        return self.tilde_S_vsp_dagger[t_rel]

        # Get original PSV: S_{xa,i} (cache unprojected when usedNe != self.usedNe)
        S_psv = self.get_PSV(t_source, t_sink, cache=should_cache_unprojected)
        # Slice to usedNe_sink if needed
        if usedNe_sink != self.usedNe:
            S_psv = S_psv[..., :usedNe_sink]

        # Get VSV: S_{j,i}
        S_vsv = self.get(t_source, t_sink)

        # Get overlap matrix M (full) and slice to usedNe_sink
        M_full = self.overlap_matrix_data[
            :, : self.usedNe, : self.usedNp, :
        ]  # [Lt, Ne, Np, Nc]
        M = M_full[:, :usedNe_sink, :, :] if usedNe_sink != self.usedNe else M_full

        if is_single_time:
            M_t = M[t_sink]  # [Ne, Np, Nc]

            # Compute: sum_j M_{xj,c} S_{j,i}
            # Formula 5.2: tilde{S}_{xc,i} = S_{xc,i} - sum_j M_{xj,c} S_{j,i}
            # M_t: [Ne, Np, Nc] where M_t[j, x, c] = M_{xj,c} = <eta_x,c| xi_j>
            #   Index order: j (Ne, 0th), x (Np, 1st), c (Nc, 2nd)
            # S_vsv: [Ns, Ns, Ne, Ne] where S_vsv[s1, s2, j, i] = S_{j,i} = <xi_j| S |xi_i>
            #   j is sink/eigenvector (Ne, 2nd), i is source/eigenvector (Ne, 3rd)
            # S_psv: [Ns, Ns, Np, Nc, Ne] where S_psv[s1, s2, x, c, i] = S_{xc,i} = <eta_x,c| S |xi_i>
            # Contract j: sum_j M_t[j, x, c] * S_vsv[s1, s2, j, i] -> [s1, s2, x, c, i]
            correction = contract(
                "jxc,abji->abxci", M_t, S_vsv[:, :, :usedNe_sink, :usedNe_sink]
            )

            # tilde{S} = S - correction
            tilde_S = S_psv - correction
            log_gpu_memory(f"get_PSV_highmode(after, single_time)")
            return tilde_S
        else:
            # Multi-time case
            if isinstance(t_source, int):
                t_rel = (backend.asarray(t_sink) - t_source) % self.Lt
                M_t = M[t_rel]  # [t, Ne, Np, Nc]

                # M_t: [t, Ne, Np, Nc] where M_t[t, j, x, c] = M_{xj,c} = <eta_x,c| xi_j>
                #   Index order: t (Lt, 0th), j (Ne, 1st), x (Np, 2nd), c (Nc, 3rd)
                # S_vsv: [t, Ns, Ns, Ne, Ne] where S_{j,i} = <xi_j| S |xi_i>
                #   Index order: t, s1, s2, j (sink, 3rd), i (source, 4th)
                # Contract j: sum_j M_t[t, j, x, c] * S_vsv[t, s1, s2, j, i] -> [t, s1, s2, x, c, i]
                correction = contract(
                    "tjxc,tabji->tabxci",
                    M_t,
                    S_vsv[:, :, :, :usedNe_sink, :usedNe_sink],
                )
                tilde_S = S_psv - correction
                if should_cache_highmode:
                    if self.debug:
                        logger.debug(f"caching full tilde_S_psv for t_source={t_source}")
                    self.tilde_S_psv_cache = tilde_S
                    self.tilde_S_psv_dagger = self._dagger_psv(tilde_S)
                    self.tilde_S_psv_cached_time = t_source
                log_gpu_memory(f"get_PSV_highmode(after, multi_time, t_source=int)")
                return tilde_S[t_rel]
            else:  # t_sink is int
                t_rel = (backend.asarray(t_source) - t_sink) % self.Lt
                M_t = M[t_sink]  # [t, Ne, Np, Nc]
                # M_t: [t, Ne, Np, Nc] where M_t[t, j, x, c] = M_{xj,c} = <eta_x,c| xi_j>
                #   Index order: t (Lt, 0th), j (Ne, 1st), x (Np, 2nd), c (Nc, 3rd)
                # S_vsv: [t, Ns, Ns, Ne, Ne] where S_{j,i} = <xi_j| S |xi_i>
                #   Index order: t, s1, s2, j (sink, 3rd), i (source, 4th)
                # Contract j: sum_j M_t[t, j, x, c] * S_vsv[t, s1, s2, j, i] -> [t, s1, s2, x, c, i]
                correction = contract(
                    "tjxc,abji->tabxci",
                    M_t,
                    S_vsv[:, :, :, :usedNe_sink, :usedNe_sink],
                )
                tilde_S = S_psv - correction
                if should_cache_highmode:
                    if self.debug:
                        logger.debug(f"caching full tilde_S_psv for t_source={t_source}")
                    self.tilde_S_psv_cache = tilde_S
                    self.tilde_S_psv_dagger = self._dagger_psv(tilde_S)
                    self.tilde_S_psv_cached_time = t_source
                log_gpu_memory(f"get_PSV_highmode(after, multi_time, t_sink=int)")
                return tilde_S[t_rel]

    def get_PSP_highmode(self, t_source, t_sink, usedNe_sink=None, usedNe_source=None):
        """
        Get PSP propagator with high mode projection applied.

        New mapping (no conjugates):
          tilde{S}_{xa,yb} = S_{xa,yb}
              - \sum_i M_{xi,a} \, tilde{S}_{i,yb}
              - \sum_j S_{xa,j} \, M_{jy,b}

        Only applies projection when usedNe > 0. When usedNe == 0, returns unprojected PSP.
        No caching when usedNe parameters != self.usedNe.

        Args:
            t_source: Source time
            t_sink: Sink time
            usedNe_sink: Number of eigenvectors for sink (left) end (default: self.usedNe)
            usedNe_source: Number of eigenvectors for source (right) end (default: self.usedNe)

        Returns:
            Array shape: [Ns, Ns, Np_snk, Nc, Np_src, Nc] or [t, Ns, Ns, Np_snk, Nc, Np_src, Nc]
        """
        # Use self.usedNe if not specified
        if usedNe_sink is None:
            usedNe_sink = self.usedNe
        if usedNe_source is None:
            usedNe_source = self.usedNe

        # If no eigenvectors, return unprojected
        if (usedNe_sink == 0) and (usedNe_source == 0):
            return self.get_PSP(t_source, t_sink)

        log_gpu_memory(
            f"get_PSP_highmode(before, t_source={t_source}, t_sink={t_sink})"
        )
        if self.debug:
            logger.debug(f"\nget_PSP_highmode() called:")
            logger.debug(f"  t_source: {t_source}, t_sink: {t_sink}")
            logger.debug(
                f"  usedNe_sink: {usedNe_sink}, usedNe_source: {usedNe_source}, self.usedNe: {self.usedNe}"
            )

        backend = get_backend()

        # Determine caching strategy
        should_cache_highmode = (
            usedNe_sink == self.usedNe and usedNe_source == self.usedNe
        )
        # Cache unprojected when any usedNe != self.usedNe (won't cache highmode)
        should_cache_unprojected = (
            usedNe_sink != self.usedNe or usedNe_source != self.usedNe
        )
        is_single_time = isinstance(t_source, int) and isinstance(t_sink, int)
        if is_single_time:
            should_cache_highmode = False
            should_cache_unprojected = True

        # Check cache only when should_cache_highmode
        if usedNe_sink == self.usedNe and usedNe_source == self.usedNe:
            if is_single_time:
                if self.tilde_S_psp_cached_time == t_source:
                    t_rel = (t_sink - t_source) % self.Lt
                    return self.tilde_S_psp_cache[t_rel]
                elif self.tilde_S_psp_cached_time == t_sink:
                    t_rel = (t_source - t_sink) % self.Lt
                    return self.tilde_S_psp_dagger[t_rel]
            else:
                # Multi-time case
                if isinstance(t_source, int):
                    t_rel = (backend.asarray(t_sink) - t_source) % self.Lt
                    if self.tilde_S_psp_cached_time == t_source:
                        if self.debug:
                            logger.debug(f"  Using cached tilde_S_psp")
                        return self.tilde_S_psp_cache[t_rel]
                else:  # t_sink is int
                    t_rel = (backend.asarray(t_source) - t_sink) % self.Lt
                    if self.tilde_S_psp_cached_time == t_sink:
                        if self.debug:
                            logger.debug(f"  Using cached tilde_S_psp (dagger)")
                        return self.tilde_S_psp_dagger[t_rel]

        # Get original PSP: S_{xa,yb}
        S_psp = self.get_PSP(t_source, t_sink, cache=should_cache_unprojected)
        # Slice if needed (PSP doesn't have Ne dimension to slice directly)

        # Get PSV for mixed term 3: S_{xa,j}
        S_psv = self.get_PSV(t_source, t_sink, cache=should_cache_unprojected)
        # Slice to usedNe_source if needed
        if usedNe_source != self.usedNe:
            S_psv = S_psv[..., :usedNe_source]

        # Get tilde VSP for mixed term 2: tilde{S}_{i,yb}
        S_vsp_tilde = self.get_VSP_highmode(
            t_source, t_sink, usedNe_source=usedNe_source
        )

        # Get overlap matrix M (full) and slice for sink and source
        M_full = self.overlap_matrix_data[
            :, : self.usedNe, : self.usedNp, :
        ]  # [Lt, Ne, Np, Nc]
        M_sink = M_full[:, :usedNe_sink, :, :] if usedNe_sink != self.usedNe else M_full
        M_source = (
            M_full[:, :usedNe_source, :, :] if usedNe_source != self.usedNe else M_full
        ).conj()

        if is_single_time:
            t_rel = (t_sink - t_source) % self.Lt
            M_sink_t = M_sink[t_sink]  # [Ne_sink, Np, Nc]
            M_source_t = M_source[t_source]  # [Ne_source, Np, Nc]

            # S_psp: [Ns_snk, Ns_src, Np_x, Nc, Np_y, Nc] where indices are [a, b, x, c, y, d]
            #   a, b: spin indices, x: sink point, c: sink color, y: source point, d: source color
            # S_vsp_tilde: [Ns_snk, Ns_src, Ne_i, Np_y, Nc] where indices are [a, b, i, y, d]
            #   d is source color (same as S_psp's source color)
            # S_psv: [Ns_snk, Ns_src, Np_x, Nc, Ne_j] where indices are [a, b, x, c, j]
            #   c is sink color (same as S_psp's sink color)

            # Term 1: S_{xa,yb}
            term1 = S_psp

            # Term 2: - sum_i M_{xi,c} tilde{S}_{i,yc}
            # Only compute if usedNe_sink > 0 (usedNe=0 means no projection)
            if usedNe_sink == 0:
                term2 = 0
            else:
                # M_sink_t: [Ne_i, Np_x, Nc] where M_{xi,c} = <eta_x,c| xi_i>
                #   Index order: i (Ne, 0th), x (Np, 1st), c (Nc, 2nd) - c is sink color
                # S_vsp_tilde: [Ns, Ns, Ne_i, Np_y, Nc] (already uses usedNe_source)
                #   Index order: a (Ns, 0th), b (Ns, 1st), i (Ne, 2nd), y (Np, 3rd), d (Nc, 4th) - d is source color
                # Contract i: sum_i M_sink_t[i, x, c] * S_vsp_tilde[a, b, i, y, d] -> [a, b, x, c, y, d]
                # Note: c is sink color, d is source color (two independent color dimensions)
                term2 = contract("ixc,abiyd->abxcyd", M_sink_t, S_vsp_tilde)

            # Term 3: - sum_j S_{xc,j} M_{jy,d}
            # Formula 5.1: M_{jy,d} = <xi_j|eta_y,d> = M_{yj,d}^*
            # Only compute if usedNe_source > 0
            if usedNe_source == 0:
                term3 = 0
            else:
                # S_psv: [Ns, Ns, Np_x, Nc, Ne_j] where S_{xc,j} = <eta_x,c| S |xi_j>
                #   Index order: a (Ns, 0th), b (Ns, 1st), x (Np, 2nd), c (Nc, 3rd), j (Ne, 4th)
                #   c is sink color, j is source (right/ket, 4th dim)
                # M_source_t: [Ne_j, Np_y, Nc] where M_{yj,d} = <eta_y,d| xi_j>
                #   Index order: j (Ne, 0th), y (Np, 1st), d (Nc, 2nd) - d is source color
                #   M_{jy,d} = M_{yj,d}^*, but we use M_{yj,d} directly since contract handles conjugation
                # Contract j: sum_j S_psv[a, b, x, c, j] * M_source_t[j, y, d] -> [a, b, x, c, y, d]
                # Note: c is sink color, d is source color (two independent color dimensions)
                term3 = contract(
                    "abxcj,jyd->abxcyd", S_psv[:, :, :, :, :usedNe_source], M_source_t
                )

            # Combine: tilde{S} = term1 - term2 - term3
            tilde_S = term1 - term2 - term3
            log_gpu_memory(f"get_PSP_highmode(after, single_time)")
        else:
            # Multi-time case
            term1 = S_psp
            if isinstance(t_source, int):
                t_rel = (backend.asarray(t_sink) - t_source) % self.Lt
                M_sink_t = M_sink[t_rel]  # [Ne_sink, Np, Nc]
                M_source_t = M_source[t_source]  # [Ne_source, Np, Nc]
                if usedNe_sink == 0:
                    term2 = 0
                else:
                    term2 = contract("ixc,tabiyd->tabxcyd", M_sink_t, S_vsp_tilde)
                if usedNe_source == 0:
                    term3 = 0
                else:
                    term3 = contract(
                        "abxcj,jyd->tabxcyd",
                        S_psv[:, :, :, :, :usedNe_source],
                        M_source_t,
                    )
            else:  # t_sink is int
                t_rel = (backend.asarray(t_source) - t_sink) % self.Lt
                M_sink_t = M_sink[t_sink]  # [Ne_sink, Np, Nc]
                M_source_t = M_source[t_rel]  # [Ne_source, Np, Nc]
                if usedNe_sink == 0:
                    term2 = 0
                else:
                    term2 = contract("ixc,abiyd->tabxcyd", M_sink_t, S_vsp_tilde)
                if usedNe_source == 0:
                    term3 = 0
                else:
                    term3 = contract(
                        "abxcj,tjyd->tabxcyd",
                        S_psv[:, :, :, :, :usedNe_source],
                        M_source_t,
                    )
            tilde_S = term1 - term2 - term3
            if should_cache_highmode:
                if isinstance(t_source, int):
                    if self.debug:
                        logger.debug(f"caching full tilde_S_psp for t_source={t_source}")
                    self.tilde_S_psp_cache = tilde_S
                    self.tilde_S_psp_dagger = self._dagger_psp(tilde_S)
                    self.tilde_S_psp_cached_time = t_source
                else:  # t_sink is int
                    if self.debug:
                        logger.debug(f"caching full tilde_S_psp for t_sink={t_sink}")
                    self.tilde_S_psp_cache = self._dagger_psp(tilde_S)
                    self.tilde_S_psp_dagger = tilde_S
                    self.tilde_S_psp_cached_time = t_sink

        log_gpu_memory(f"get_PSP_highmode(after, multi_time)")
        if self.debug:
            logger.debug(f"  tilde_S_psp shape: {tilde_S.shape}")

        return tilde_S


def compute_diagrams_multitime(
    diagrams: List[QuarkDiagram],
    time_list,
    vertex_list: List[Meson],
    propagator_list: List[Propagator],
    multitime_shape: int = False,
    debug: bool = False,
):
    """
    Compute diagram values with automatic sampling weight application.

    This function automatically expands diagrams with current vertices into multiple
    state combinations and point coincidence scenes. Each scene has a sampling weight
    (scene_weight) that compensates for the finite sampling of spatial points.

    The sampling weights are calculated as w(k) = C(L^3,k)/C(usedNp,k) where:
    - L = spatial lattice size (total points = L^3)
    - usedNp = number of sampled points
    - k = number of distinct points in this scene

    After computing each diagram's contraction, the result is multiplied by its
    scene_weight. The final result is the sum of all weighted scene contributions.

    See localized_blending.md for mathematical formulation.
    """
    backend = get_backend()

    # Auto-expand diagrams if they have vertex_list and expanded_diagrams
    diagrams_to_compute = []
    for diagram in diagrams:
        if hasattr(diagram, "expanded_diagrams") and diagram.expanded_diagrams:
            if debug:
                logger.debug(
                    f"Auto-expanding diagram into {len(diagram.expanded_diagrams)} diagrams"
                )
            diagrams_to_compute.extend(diagram.expanded_diagrams)
        else:
            diagrams_to_compute.append(diagram)

    if debug and len(diagrams_to_compute) != len(diagrams):
        logger.debug(
            f"Total diagrams after expansion: {len(diagrams_to_compute)} (from {len(diagrams)} input diagrams)"
        )

    diagram_value = []
    multi_time = None
    for time in time_list:
        if not isinstance(time, (int, np.integer)):
            if multi_time is None:
                multi_time = time
            else:
                if id(multi_time) != id(time):
                    raise NotImplementedError("only support one multitime yet")
    for diagram_idx, diagram in enumerate(diagrams_to_compute):
        if debug:
            logger.debug(f"\n{'='*80}")
            logger.debug(f"Processing diagram {diagram_idx}")
            logger.debug(f"{'='*80}")
        diagram_value.append(1.0)
        for contraction_idx, (operands, subscripts) in enumerate(
            zip(diagram.operands, diagram.subscripts)
        ):
            if debug:
                logger.debug(f"\n  Contraction {contraction_idx}:")
                logger.debug(f"  Original subscripts: {subscripts}")

            have_multitime = False
            subscripts = subscripts.split(",")
            idx = 0
            operands_data = []

            if debug:
                logger.debug(f"  Propagators (operands[0]):")
            for prop_idx, item in enumerate(operands[0]):
                propagator = propagator_list[item[0]]

                # Determine propagator type from diagram
                if hasattr(diagram, "propagator_types") and diagram.propagator_types:
                    prop_type = diagram.propagator_types[contraction_idx][prop_idx]
                    if debug:
                        logger.debug(f"    [{prop_idx}] Propagator type: {prop_type}")
                else:
                    prop_type = "VSV"  # Default for backward compatibility
                    if debug:
                        logger.debug(
                            f"    [{prop_idx}] No propagator type info, defaulting to VSV"
                        )

                # Extract vertex attributes (item[1]=source/right, item[2]=sink/left)
                src_vertex = vertex_list[item[1]]
                snk_vertex = vertex_list[item[2]]

                usedNe_source = getattr(src_vertex, "usedNe", None)
                usedNe_sink = getattr(snk_vertex, "usedNe", None)
                usedNp_source = getattr(src_vertex, "usedNp", None)
                usedNp_sink = getattr(snk_vertex, "usedNp", None)

                if debug:
                    logger.debug(
                        f"      Vertex attributes: source usedNe={usedNe_source}, usedNp={usedNp_source}; sink usedNe={usedNe_sink}, usedNp={usedNp_sink}"
                    )

                # Get propagator data based on type
                try:
                    if prop_type == "VSV":
                        # Standard VSV: use get(t_source, t_sink) -> S_{i,j}
                        prop_data = propagator.get(
                            time_list[item[1]], time_list[item[2]]
                        )
                        # Slice both ends' usedNe
                        if usedNe_sink is not None:
                            prop_data = prop_data[..., :usedNe_sink, :]
                        if usedNe_source is not None:
                            prop_data = prop_data[..., :usedNe_source]
                        if debug:
                            logger.debug(
                                f"      Called propagator.get(t_source={time_list[item[1]]}, t_sink={time_list[item[2]]})"
                            )
                    elif prop_type == "VSP":
                        # VSP: sink=vector, source=point
                        # get_VSP_highmode handles usedNe_source=0 internally
                        prop_data = propagator.get_VSP_highmode(
                            time_list[item[1]], time_list[item[2]], usedNe_source
                        )
                        if debug:
                            logger.debug(
                                f"      Called propagator.get_VSP_highmode with usedNe_source={usedNe_source}"
                            )
                        # Slice sink端 (vector) usedNe
                        if usedNe_sink is not None:
                            prop_data = prop_data[..., :usedNe_sink, :, :]
                        # Slice source端 (point) usedNp
                        if usedNp_source is not None:
                            prop_data = prop_data[..., :usedNp_source, :]
                    elif prop_type == "PSV":
                        # PSV: sink=point, source=vector
                        # get_PSV_highmode handles usedNe_sink=0 internally
                        prop_data = propagator.get_PSV_highmode(
                            time_list[item[1]], time_list[item[2]], usedNe_sink
                        )
                        if debug:
                            logger.debug(
                                f"      Called propagator.get_PSV_highmode with usedNe_sink={usedNe_sink}"
                            )
                        # Slice sink端 (point) usedNp
                        if usedNp_sink is not None:
                            prop_data = prop_data[..., :usedNp_sink, :, :]
                        # Slice source端 (vector) usedNe
                        if usedNe_source is not None:
                            prop_data = prop_data[..., :usedNe_source]
                    elif prop_type == "PSP":
                        # PSP: sink=point, source=point
                        # get_PSP_highmode handles both usedNe=0 cases internally
                        prop_data = propagator.get_PSP_highmode(
                            time_list[item[1]],
                            time_list[item[2]],
                            usedNe_sink,
                            usedNe_source,
                        )
                        if debug:
                            logger.debug(
                                f"      Called propagator.get_PSP_highmode with usedNe_sink={usedNe_sink}, usedNe_source={usedNe_source}"
                            )
                        # Slice sink端 (point) usedNp
                        if usedNp_sink is not None:
                            prop_data = prop_data[..., :usedNp_sink, :, :, :]
                        # Slice source端 (point) usedNp
                        if usedNp_source is not None:
                            prop_data = prop_data[..., :usedNp_source, :]
                    else:
                        raise ValueError(f"Unknown propagator type: {prop_type}")

                    operands_data.append(prop_data)
                    if debug:
                        logger.debug(
                            f"        shape: {prop_data.shape}, dtype: {prop_data.dtype}"
                        )

                    # Handle multitime subscripts
                    if not isinstance(time_list[item[1]], int) or not isinstance(
                        time_list[item[2]], int
                    ):
                        subscripts[idx] = "t" + subscripts[idx]
                        have_multitime = True
                    idx += 1

                except (AttributeError, ValueError) as e:
                    error_msg = (
                        f"Error getting propagator[{item[0]}] type {prop_type}: {e}\n"
                        f"  Available methods: {[m for m in dir(propagator) if not m.startswith('_') and callable(getattr(propagator, m, None))]}\n"
                        f"  Propagator type: {type(propagator).__name__}"
                    )
                    if debug:
                        logger.debug(f"  ERROR: {error_msg}")
                    raise RuntimeError(error_msg) from e

            if debug:
                logger.debug(f"  Vertices (operands[1]):")
            for vertex_idx, item in enumerate(operands[1]):
                vertex = vertex_list[item]

                # Determine vertex type from diagram
                if hasattr(diagram, "vertex_types") and diagram.vertex_types:
                    vertex_type = diagram.vertex_types[contraction_idx][vertex_idx]
                    if debug:
                        logger.debug(f"    [{idx}] Vertex type: {vertex_type}")
                else:
                    vertex_type = "V2V"  # Default for backward compatibility
                    if debug:
                        logger.debug(f"    [{idx}] No vertex type info, defaulting to V2V")

                # Get vertex data based on type
                if vertex_type == "V2V":
                    vertex_data = vertex.get(time_list[item])
                    if debug:
                        logger.debug(f"      Called vertex[{item}].get(t={time_list[item]})")
                elif vertex_type == "V2P":
                    vertex_data = vertex.get_v2p(time_list[item])
                    if debug:
                        logger.debug(
                            f"      Called vertex[{item}].get_v2p(t={time_list[item]})"
                        )
                elif vertex_type == "P2V":
                    vertex_data = vertex.get_p2v(time_list[item])
                    if debug:
                        logger.debug(
                            f"      Called vertex[{item}].get_p2v(t={time_list[item]})"
                        )
                elif vertex_type == "P2P":
                    vertex_data = vertex.get_p2p(time_list[item])
                    if debug:
                        logger.debug(
                            f"      Called vertex[{item}].get_p2p(t={time_list[item]})"
                        )
                else:
                    raise ValueError(f"Unknown vertex type: {vertex_type}")

                operands_data.append(vertex_data)
                if debug:
                    logger.debug(
                        f"        shape: {vertex_data.shape}, dtype: {vertex_data.dtype}"
                    )
                if not isinstance(time_list[item], int):
                    subscripts[idx] = "t" + subscripts[idx]
                    have_multitime = True
                idx += 1

            if not have_multitime:
                if multitime_shape:
                    subscripts.append("t")
                    operands_data.append([1] * len(multi_time))
                    subscripts[-1] = subscripts[-1] + "->t"
            else:
                subscripts[-1] = subscripts[-1] + "->t"

            final_subscripts = ",".join(subscripts)
            if debug:
                logger.debug(f"  Final subscripts: {final_subscripts}")
                logger.debug(f"  Operands summary:")
                for op_idx, op in enumerate(operands_data):
                    logger.debug(
                        f"    operand[{op_idx}]: shape={op.shape if hasattr(op, 'shape') else type(op)}"
                    )
                logger.debug(f"  Attempting contraction...")

            result = contract(final_subscripts, *operands_data)
            diagram_value[-1] = diagram_value[-1] * result

            if debug:
                logger.debug(
                    f"  Contraction successful! Result shape: {result.shape if hasattr(result, 'shape') else type(result)}"
                )

        # Apply scene_weight if this diagram has sampling weight
        if hasattr(diagram, "scene_weights") and diagram.scene_weights:
            scene_weight = diagram.scene_weights[0]
            diagram_value[-1] = diagram_value[-1] * scene_weight
            if debug:
                logger.debug(f"\n  Applied scene_weight: {scene_weight}")
                logger.debug(
                    f"  Final diagram value shape: {diagram_value[-1].shape if hasattr(diagram_value[-1], 'shape') else type(diagram_value[-1])}"
                )

    return backend.asarray(diagram_value)


def compute_diagrams(
    diagrams: List[QuarkDiagram],
    time_list,
    vertex_list,
    propagator_list,
    debug: bool = False,
):
    backend = get_backend()
    diagram_value = []
    for diagram_idx, diagram in enumerate(diagrams):
        if debug:
            logger.debug(f"\n{'='*80}")
            logger.debug(f"Processing diagram {diagram_idx}")
            logger.debug(f"{'='*80}")
        diagram_value.append(1.0)
        for contraction_idx, (operands, subscripts) in enumerate(
            zip(diagram.operands, diagram.subscripts)
        ):
            if debug:
                logger.debug(f"\n  Contraction {contraction_idx}:")
                logger.debug(f"  Subscripts: {subscripts}")

            operands_data = []
            idx = 0

            if debug:
                logger.debug(f"  Propagators (operands[0]):")
            for prop_idx, item in enumerate(operands[0]):
                propagator = propagator_list[item[0]]

                # Determine propagator type from diagram
                if hasattr(diagram, "propagator_types") and diagram.propagator_types:
                    prop_type = diagram.propagator_types[contraction_idx][prop_idx]
                    if debug:
                        logger.debug(f"    [{prop_idx}] Propagator type: {prop_type}")
                else:
                    prop_type = "VSV"  # Default for backward compatibility
                    if debug:
                        logger.debug(
                            f"    [{prop_idx}] No propagator type info, defaulting to VSV"
                        )

                # Get propagator data based on type
                try:
                    if prop_type == "VSV":
                        # Standard VSV: use get(t_source, t_sink)
                        prop_data = propagator.get(
                            time_list[item[1]], time_list[item[2]]
                        )
                        if debug:
                            logger.debug(
                                f"      Called propagator.get(t_source={time_list[item[1]]}, t_sink={time_list[item[2]]})"
                            )
                    elif prop_type == "VSP":
                        # VSP: use get_v2p(t_source, t_sink)
                        prop_data, effective_type = propagator.get_v2p(
                            time_list[item[1]], time_list[item[2]]
                        )
                        prop_type = effective_type
                        if debug:
                            logger.debug(
                                f"      Called propagator.get_v2p(tsrc={time_list[item[1]]}, tsink={time_list[item[2]]}) -> effective_type={effective_type}"
                            )
                    elif prop_type == "PSV":
                        # PSV: use get_p2v(t_source, t_sink)
                        prop_data, effective_type = propagator.get_p2v(
                            time_list[item[1]], time_list[item[2]]
                        )
                        prop_type = effective_type
                        if debug:
                            logger.debug(
                                f"      Called propagator.get_p2v(tsrc={time_list[item[1]]}, tsink={time_list[item[2]]}) -> effective_type={effective_type}"
                            )
                    elif prop_type == "PSP":
                        # PSP: use get_p2p(t_source, t_sink)
                        prop_data, effective_type = propagator.get_p2p(
                            time_list[item[1]], time_list[item[2]]
                        )
                        prop_type = effective_type
                        if debug:
                            logger.debug(
                                f"      Called propagator.get_p2p(tsrc={time_list[item[1]]}, tsink={time_list[item[2]]}) -> effective_type={effective_type}"
                            )
                    else:
                        raise ValueError(f"Unknown propagator type: {prop_type}")

                    operands_data.append(prop_data)
                    if debug:
                        logger.debug(
                            f"        shape: {prop_data.shape}, dtype: {prop_data.dtype}"
                        )
                    idx += 1

                except (AttributeError, ValueError) as e:
                    error_msg = (
                        f"Error getting propagator[{item[0]}] type {prop_type}: {e}\n"
                        f"  Available methods: {[m for m in dir(propagator) if not m.startswith('_') and callable(getattr(propagator, m, None))]}\n"
                        f"  Propagator type: {type(propagator).__name__}"
                    )
                    if debug:
                        logger.debug(f"  ERROR: {error_msg}")
                    raise RuntimeError(error_msg) from e

            if debug:
                logger.debug(f"  Vertices (operands[1]):")
            for item in operands[1]:
                vertex_data = vertex_list[item].get(time_list[item])
                operands_data.append(vertex_data)
                if debug:
                    logger.debug(f"    [{idx}] vertex[{item}].get(t={time_list[item]})")
                    logger.debug(
                        f"        shape: {vertex_data.shape}, dtype: {vertex_data.dtype}"
                    )
                idx += 1

            result = contract(subscripts, *operands_data)
            diagram_value[-1] *= result

            if debug:
                logger.debug(
                    f"  Contraction successful! Result shape: {result.shape if hasattr(result, 'shape') else type(result)}"
                )
    return backend.asarray(diagram_value)


from typing import Union, List, Dict, Tuple, Any
from sympy import S, Add, Expr, Symbol, Mul
import hashlib


class Diagram(Symbol):
    def __new__(
        cls,
        diagram: QuarkDiagram,
        time_list,
        vertex_list,
        propagator_list,
    ) -> None:
        obj = super().__new__(
            cls,
            f"{diagram.adjacency_matrix},{time_list},{vertex_list},{propagator_list}",
        )
        return obj

    def __init__(
        self,
        diagram: QuarkDiagram,
        time_list,
        vertex_list,
        propagator_list,
    ) -> None:
        """
        Initialize a Diagram object.

        Args:
            diagram: The QuarkDiagram object
            time_list: List of time values
            vertex_list: List of vertices
            propagator_list: List of propagators
        """
        self.diagram = diagram
        self.time_list = time_list
        self.vertex_list = vertex_list
        self.propagator_list = propagator_list
        self.value = None
        self.value_pointer = None

    def calc(self):
        if self.value is None:
            self.value = self.__hash__()
            self.value = compute_diagrams_multitime(
                [self.diagram], self.time_list, self.vertex_list, self.propagator_list
            )
        return self.value

    def __str__(self):
        return f"{self.diagram.adjacency_matrix},{self.time_list},{self.vertex_list},{self.propagator_list}"

    def __repr__(self):
        return f"{self.diagram.adjacency_matrix},{self.time_list},{self.vertex_list},{self.propagator_list}"

    def __eq__(self, other):
        if not isinstance(other, Diagram):
            return False
        return (
            self.diagram.adjacency_matrix == other.diagram.adjacency_matrix
            and self.time_list == other.time_list
            and self.vertex_list == other.vertex_list
            and self.propagator_list == other.propagator_list
        )

    def __hash__(self):
        return int(hashlib.sha256(str(self).encode()).hexdigest(), 16) % (2**31)

    def transform(self, group_element, time=None):
        coeff_vertices_list = []
        for i in range(len(self.vertex_list)):
            assert isinstance(self.vertex_list[i], HadronIrrepRow)
            if time is None or self.time_list[i] == time:
                transformed_vertex = self.vertex_list[i].transform(group_element)
            else:
                transformed_vertex = self.vertex_list[i]
            terms = Add.make_args(transformed_vertex)
            coeff_vertices = []
            for term in terms:
                factors = Mul.make_args(term)
                coeff = S(1)
                for factor in factors:
                    if isinstance(factor, HadronIrrepRow):
                        hadron_irrep_row = factor
                    else:
                        coeff *= factor
                coeff_vertices.append((coeff, hadron_irrep_row))
            coeff_vertices_list.append(coeff_vertices)
        vertex_products = list(product(*coeff_vertices_list))
        result = S(0)
        for i in range(len(vertex_products)):
            vertex_product = vertex_products[i]
            diagram_coeff = S(1)
            new_vertex_list = []
            for coeff_vertex in vertex_product:
                diagram_coeff *= coeff_vertex[0]
                new_vertex_list.append(coeff_vertex[1])
            result += diagram_coeff * Diagram(
                self.diagram, self.time_list, new_vertex_list, self.propagator_list
            )
        return result

    def conjugate(self):
        new_vertex_list = []
        for i in range(len(self.vertex_list)):
            assert isinstance(self.vertex_list[i], HadronIrrepRow)
            new_vertex_list.append(self.vertex_list[i].conjugate())

        return Diagram(
            self.diagram, self.time_list, new_vertex_list, self.propagator_list
        )

    def simplify(self):
        """
        Simplify Diagram object, perform the following operations:
        1. Remove redundant vertices (equivalent to remove_redundant functionality)
        2. Sort vertices and propagators (equivalent to sort_vertex_and_propagator functionality)
        3. Split graph into product of different subgraphs

        Integrates the functionality of the original separate remove_redundant and sort_vertex_and_propagator methods,
        and further splits the graph into connected components, ultimately returning the optimized Diagram or Diagram product expression.

        Returns:
            sympy.Expr or Diagram: Simplified Diagram object or expression representing product of different subgraphs
        """
        from sympy import Mul
        from copy import deepcopy

        # Get graph information
        adjacency_matrix = deepcopy(self.diagram.adjacency_matrix)
        num_vertex = len(adjacency_matrix)
        # Record each vertex's connected component
        component_ids = [-1] * num_vertex
        next_component_id = 0

        # Use BFS to find all connected components
        for start_vertex in range(num_vertex):
            # If already assigned connected component, skip
            if component_ids[start_vertex] != -1:
                continue

            # Check if any edge involves this vertex
            has_connection = False
            for i in range(num_vertex):
                if (
                    isinstance(adjacency_matrix[start_vertex][i], np.ndarray)
                    and (adjacency_matrix[start_vertex][i] != 0).any()
                ):
                    has_connection = True
                elif (
                    isinstance(adjacency_matrix[i][start_vertex], np.ndarray)
                    and (adjacency_matrix[i][start_vertex] != 0).any()
                ):
                    has_connection = True
                elif adjacency_matrix[start_vertex][i] != 0:
                    has_connection = True
                elif adjacency_matrix[i][start_vertex] != 0:
                    has_connection = True
                if has_connection:
                    break
            if not has_connection:
                continue
            # Use BFS to find all connected vertices
            component_ids[start_vertex] = next_component_id
            queue = [start_vertex]
            while queue:
                vertex = queue.pop(0)

                # Check all possible connections
                for next_vertex in range(num_vertex):
                    # Check connection from vertex to next_vertex
                    if component_ids[next_vertex] == -1:
                        is_in_queue = False
                        if (
                            isinstance(
                                adjacency_matrix[vertex][next_vertex], np.ndarray
                            )
                            and (adjacency_matrix[vertex][next_vertex] != 0).any()
                        ):
                            is_in_queue = True
                        elif (
                            isinstance(
                                adjacency_matrix[next_vertex][vertex], np.ndarray
                            )
                            and (adjacency_matrix[next_vertex][vertex] != 0).any()
                        ):
                            is_in_queue = True
                        elif adjacency_matrix[vertex][next_vertex] != 0:
                            is_in_queue = True
                        elif adjacency_matrix[next_vertex][vertex] != 0:
                            is_in_queue = True
                        if is_in_queue:
                            component_ids[next_vertex] = next_component_id
                            queue.append(next_vertex)
            # Start a new connected component
            next_component_id += 1
        # Create a new Diagram object for each connected component
        result_diagrams = []
        for component_id in range(next_component_id):
            # Find vertices belonging to this connected component
            vertices = []
            for i in range(num_vertex):
                if component_ids[i] == component_id:
                    vertices.append(i)
            # Sort vertices based on self.vertex_list
            # When vertices are equal, consider all possible orders
            # First, sort based on time and vertex type
            vertices.sort(key=lambda v: (self.time_list[v], self.vertex_list[v]))

            # Check if there are same vertices
            has_same_vertices = False
            for i in range(len(vertices) - 1):
                if (
                    self.time_list[vertices[i]] == self.time_list[vertices[i + 1]]
                    and self.vertex_list[vertices[i]]
                    == self.vertex_list[vertices[i + 1]]
                ):
                    has_same_vertices = True
                    break

            # If there are same vertices, consider all possible orders
            if has_same_vertices:
                # Group by time and vertex type
                from itertools import groupby
                from itertools import permutations

                # Group by time and vertex type
                groups = []
                for k, g in groupby(
                    vertices, key=lambda v: (self.time_list[v], self.vertex_list[v])
                ):
                    groups.append(list(g))

                # Only sort vertices in each group, keeping group order
                all_possible_orders = []
                for g in groups:
                    perms = [list(p) for p in permutations(g)]
                    all_possible_orders.append(perms)

                # Use itertools.product to get all combinations
                from itertools import product

                all_permutations = list(product(*all_possible_orders))

                # Flatten nested list to match vertices format
                all_possible_vertices = []
                for perm_combination in all_permutations:
                    flattened_vertices = []
                    for group_perm in perm_combination:
                        flattened_vertices.extend(group_perm)
                    all_possible_vertices.append(flattened_vertices)
            else:
                all_possible_vertices = [vertices]
            # Create adjacency matrix for this connected component, size of connected component
            min_hash = float("inf")
            for vertices in all_possible_vertices:
                component_size = len(vertices)
                component_matrix = [
                    [
                        adjacency_matrix[vertices[j]][vertices[i]]
                        for i in range(component_size)
                    ]
                    for j in range(component_size)
                ]
                new_quark_diagram = QuarkDiagram(component_matrix)
                new_time_list = [self.time_list[i] for i in vertices]
                new_vertex_list = [self.vertex_list[i] for i in vertices]
                # Create new propagator_list, only retain used propagators
                used_propagators = set([])  # 0 is default value, always retained

                # Traverse component_matrix to find all used propagators
                for i in range(component_size):
                    for j in range(component_size):
                        value = component_matrix[i][j]
                        if isinstance(value, int):
                            if value != 0:
                                used_propagators.add(value)
                        elif isinstance(value, np.ndarray):
                            # For array type, add propagator indices of all non-zero elements
                            for prop_idx in value.flatten():
                                if prop_idx != 0:
                                    used_propagators.add(int(prop_idx))
                        elif isinstance(value, list):
                            # For list type, add all non-zero elements
                            # Handle nested lists, similar to ndarray's flatten operation
                            flat_values = []

                            def flatten_list(lst):
                                for item in lst:
                                    if isinstance(item, list):
                                        flatten_list(item)
                                    else:
                                        flat_values.append(item)

                            flatten_list(value)
                            for prop_idx in flat_values:
                                if prop_idx != 0:
                                    used_propagators.add(prop_idx)

                # Sort used propagator indices in original order
                used_propagators = sorted(
                    list(used_propagators), key=lambda x: self.propagator_list[x]
                )
                used_propagators = [0] + used_propagators
                new_propagator_list = [
                    self.propagator_list[i] for i in used_propagators
                ]

                # Update propagator indices in component_matrix
                old_to_new = {
                    old_idx: new_idx for new_idx, old_idx in enumerate(used_propagators)
                }
                for i in range(component_size):
                    for j in range(component_size):
                        value = component_matrix[i][j]
                        if isinstance(value, int):
                            if value != 0:
                                component_matrix[i][j] = old_to_new[value]
                        elif isinstance(value, np.ndarray):
                            # For array type, update all non-zero elements
                            new_array = np.zeros_like(value)
                            for idx in np.ndindex(value.shape):
                                if value[idx] != 0:
                                    new_array[idx] = old_to_new[int(value[idx])]
                            component_matrix[i][j] = new_array
                        elif isinstance(value, list):
                            # For list type, update all non-zero elements
                            # Handle nested list case
                            def update_nested_list(lst):
                                result = []
                                for item in lst:
                                    if isinstance(item, list):
                                        result.append(update_nested_list(item))
                                    else:
                                        result.append(
                                            old_to_new[item] if item != 0 else 0
                                        )
                                return result

                            component_matrix[i][j] = update_nested_list(value)

                # Use stable hash algorithm to calculate graph hash
                current_hash = int(
                    hashlib.sha256(
                        str(new_quark_diagram.adjacency_matrix).encode()
                    ).hexdigest(),
                    16,
                )
                if min_hash > current_hash:
                    min_hash = current_hash
                    new_diagram = Diagram(
                        new_quark_diagram,
                        new_time_list,
                        new_vertex_list,
                        new_propagator_list,
                    )

            result_diagrams.append(new_diagram)

        # Convert results to product expression
        result = S(1)
        for diagram in result_diagrams:
            result = Mul(result, diagram, evaluate=False)
        return sp.simplify(result)

    def replace_propagator(self, propagator_map: Dict):
        """
        Replace propagators in Diagram
        """
        for i, propagator in enumerate(self.propagator_list):
            if propagator in propagator_map:
                self.propagator_list[i] = propagator_map[propagator]

    def replace_vertex(self, vertex_map: Callable):
        """
        Replace vertices in Diagram
        """
        for i, vertex in enumerate(self.vertex_list):
            result = vertex_map(vertex)
            if result is not None:
                self.vertex_list[i] = result

    def replace_time(self, time_map: Dict):
        """
        Replace time in Diagram
        """
        for i, time in enumerate(self.time_list):
            if time in time_map:
                self.time_list[i] = time_map[time]


def diagram_vertice_replace(
    expr: Union[Expr, List, Any], indice_map: Dict
) -> Union[Expr, List, Any]:
    """
    Replace vertices in all Diagram object in the expr,list,dict,tuple or any data structure
    """
    if isinstance(expr, Diagram):
        # Replace vertices in the Diagram object
        new_vertice_list = [indice_map[v] for v in expr.vertex_list]
        new_diagram = Diagram(
            expr.diagram, expr.time_list, new_vertice_list, expr.propagator_list
        )
        return new_diagram
    elif isinstance(expr, list):
        # Recursively process list elements
        return [diagram_vertice_replace(item, indice_map) for item in expr]
    elif isinstance(expr, tuple):
        # Recursively process tuple elements
        return tuple(diagram_vertice_replace(item, indice_map) for item in expr)
    elif isinstance(expr, dict):
        # Recursively process dictionary values
        return {
            key: diagram_vertice_replace(value, indice_map)
            for key, value in expr.items()
        }
    elif isinstance(expr, Add):
        # Process sympy Add expression
        return Add(*[diagram_vertice_replace(arg, indice_map) for arg in expr.args])
    elif isinstance(expr, Mul):
        # Process sympy Mul expression
        return Mul(*[diagram_vertice_replace(arg, indice_map) for arg in expr.args])
    elif hasattr(expr, "args") and expr.args:
        # For other expressions with args attribute
        return expr.func(
            *[diagram_vertice_replace(arg, indice_map) for arg in expr.args]
        )
    else:
        # Return unchanged for other types
        return expr


def diagram_simplify(expr: Union[Expr, List, Any]) -> Union[Expr, List, Any]:
    """
    Recursively simplify expressions containing Diagram objects

    Call each Diagram object's simplify method, which integrates the following functionalities:
    1. Remove redundant vertices
    2. Sort vertices and propagators
    3. Split graph into connected components

    Supports processing various data structures, including:
    - Single Diagram object
    - sympy expressions (e.g., Add, Mul, Pow, etc.)
    - Nested lists, tuples, dictionaries
    - NumPy arrays

    Args:
        expr: Expression or data structure containing Diagram objects

    Returns:
        Simplified expression or data structure, original expression unchanged
    """
    from sympy import Add, Mul, Pow, Number, Symbol
    import numpy as np

    # Handle None or unsupported types
    if expr is None:
        return expr

    # Base case: process single Diagram object
    if isinstance(expr, Diagram):
        try:
            # Apply simplification operations
            # expr = expr.remove_redundant()
            # expr = expr.sort_vertex_and_propagator()
            splited_expr = expr.simplify()
            result = sp.simplify(splited_expr)
            return result

        except Exception as e:
            # If an exception occurs, return original expression and print error
            logger.debug(f"Warning: Simplification of Diagram failed: {e}")
            import traceback

            traceback.print_exc()
            return expr

    # Process general list - recursively process each element in the list
    elif isinstance(expr, list):
        return [diagram_simplify(item) for item in expr]

    # Process numpy ndarray
    elif hasattr(expr, "__array__") and hasattr(expr, "shape"):  # Detect ndarray
        # Get original shape
        original_shape = expr.shape

        # Flatten ndarray to 1D array, process each element, then restore original shape
        flattened = expr.flatten() if hasattr(expr, "flatten") else expr.ravel()
        result = np.array([diagram_simplify(item) for item in flattened], dtype=object)

        # Restore original shape
        return result.reshape(original_shape)

    # Recursively process addition expression
    elif isinstance(expr, Add):
        terms = []
        for term in expr.args:
            simplified_term = diagram_simplify(term)
            terms.append(simplified_term)
        return Add(*terms)

    # Recursively process multiplication expression
    elif isinstance(expr, Mul):
        factors = []
        for factor in expr.args:
            simplified_factor = diagram_simplify(factor)
            # Handle multiplication nested cases
            if isinstance(simplified_factor, Mul):
                factors.extend(simplified_factor.args)
            else:
                factors.append(simplified_factor)
        return Mul(*factors)

    # Recursively process power expression
    elif isinstance(expr, Pow):
        base = diagram_simplify(expr.args[0])
        # Keep exponent unchanged
        exponent = expr.args[1]
        return Pow(base, exponent)

    # Recursively process dictionary - process value part
    elif isinstance(expr, dict):
        return {key: diagram_simplify(value) for key, value in expr.items()}

    # Recursively process tuple - similar to list but returns tuple
    elif isinstance(expr, tuple):
        return tuple(diagram_simplify(item) for item in expr)

    # Other types of expressions remain unchanged
    else:
        return sp.simplify(expr)


def remove_unexpected_diagram(expr: Union[Expr, List, Any], condition: Callable):
    """
    Recursively find all Diagram objects and replace those with propagators meeting certain conditions with S(0).

    This function traverses through expressions, lists, dictionaries, or other nested structures to find
    Diagram objects. If a Diagram contains any propagator from the provided propagator_list, it will be
    replaced with a symbolic zero (S(0)).

    Args:
        expr: The expression or structure to process, can be a sympy expression, list, dictionary, etc.
        propagator_list: List of Propagator objects to check against

    Returns:
        The processed expression with redundant diagrams replaced by zeros
    """
    from sympy import Add, Mul, Pow, S
    import numpy as np

    # Handle None or unsupported types
    if expr is None:
        return expr

    # Process Diagram object
    if isinstance(expr, Diagram):
        # Check if this Diagram contains any propagator from the provided propagator_list
        for prop in expr.propagator_list:
            if not condition(prop):
                return S(0)  # If contains, return symbolic 0
        return expr  # If not contains, remain unchanged

    # Process list
    elif isinstance(expr, list):
        return [remove_unexpected_diagram(item, condition) for item in expr]

    # Process numpy array
    elif hasattr(expr, "__array__") and hasattr(expr, "shape"):
        original_shape = expr.shape
        flattened = expr.flatten() if hasattr(expr, "flatten") else expr.ravel()
        result = np.array(
            [remove_unexpected_diagram(item, condition) for item in flattened],
            dtype=object,
        )
        return result.reshape(original_shape)

    # Process addition expression
    elif isinstance(expr, Add):
        terms = [remove_unexpected_diagram(term, condition) for term in expr.args]
        return Add(*terms)

    # Process multiplication expression
    elif isinstance(expr, Mul):
        factors = [remove_unexpected_diagram(factor, condition) for factor in expr.args]
        return Mul(*factors)

    # Process power expression
    elif isinstance(expr, Pow):
        base = remove_unexpected_diagram(expr.args[0], condition)
        exponent = expr.args[1]  # Keep exponent unchanged
        return Pow(base, exponent)

    # Process dictionary
    elif isinstance(expr, dict):
        return {
            key: remove_unexpected_diagram(value, condition)
            for key, value in expr.items()
        }

    # Process tuple
    elif isinstance(expr, tuple):
        return tuple(remove_unexpected_diagram(item, condition) for item in expr)

    # Other types of expressions remain unchanged
    else:
        return expr


def remove_disconneted_diagram(
    expr: Union[Expr, List, Any], propagator_list: List[Propagator]
):
    """
    Recursively find all Diagram objects and replace those with propagators meeting certain conditions with S(0).

    This function traverses through expressions, lists, dictionaries, or other nested structures to find
    Diagram objects. If a Diagram contains any propagator from the provided propagator_list, it will be
    replaced with a symbolic zero (S(0)).

    Args:
        expr: The expression or structure to process, can be a sympy expression, list, dictionary, etc.
        propagator_list: List of Propagator objects to check against

    Returns:
        The processed expression with redundant diagrams replaced by zeros
    """
    from sympy import Add, Mul, Pow, S
    import numpy as np

    # Handle None or unsupported types
    if expr is None:
        return expr

    # Process Diagram object
    if isinstance(expr, Diagram):
        # Check if this Diagram contains any propagator from the provided propagator_list
        for prop in expr.propagator_list:
            if prop in propagator_list:
                return S(0)  # If contains, return symbolic 0
        return expr  # If not contains, remain unchanged

    # Process list
    elif isinstance(expr, list):
        return [remove_disconneted_diagram(item, propagator_list) for item in expr]

    # Process numpy array
    elif hasattr(expr, "__array__") and hasattr(expr, "shape"):
        original_shape = expr.shape
        flattened = expr.flatten() if hasattr(expr, "flatten") else expr.ravel()
        result = np.array(
            [remove_disconneted_diagram(item, propagator_list) for item in flattened],
            dtype=object,
        )
        return result.reshape(original_shape)

    # Process addition expression
    elif isinstance(expr, Add):
        terms = [
            remove_disconneted_diagram(term, propagator_list) for term in expr.args
        ]
        return Add(*terms)

    # Process multiplication expression
    elif isinstance(expr, Mul):
        factors = [
            remove_disconneted_diagram(factor, propagator_list) for factor in expr.args
        ]
        return Mul(*factors)

    # Process power expression
    elif isinstance(expr, Pow):
        base = remove_disconneted_diagram(expr.args[0], propagator_list)
        exponent = expr.args[1]  # Keep exponent unchanged
        return Pow(base, exponent)

    # Process dictionary
    elif isinstance(expr, dict):
        return {
            key: remove_disconneted_diagram(value, propagator_list)
            for key, value in expr.items()
        }

    # Process tuple
    elif isinstance(expr, tuple):
        return tuple(remove_disconneted_diagram(item, propagator_list) for item in expr)

    # Other types of expressions remain unchanged
    else:
        return expr


def _collect_diagrams(expr, diagram_list, save_dir, backend):
    """Recursively collect all unequal Diagram objects and return processed expr."""

    def collect_diagrams(e):
        """Recursively collect all unequal Diagram objects in the expression and set value_pointer to point to equal objects"""
        if isinstance(e, Diagram):

            # 为Diagram创建副本
            new_diagram = Diagram(
                e.diagram, e.time_list, e.vertex_list, e.propagator_list
            )

            # Check if an equal Diagram object already exists
            found_idx = None
            for idx, existing_diagram in enumerate(diagram_list):
                if new_diagram == existing_diagram:  # Use __eq__ method to compare
                    found_idx = idx
                    break

            if found_idx is not None:
                # If an equal object exists, set current object's value_pointer to point to that object
                new_diagram.value_pointer = found_idx
            else:
                # If no equal object exists, add to list and set value_pointer
                if save_dir is not None and False:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    if e.value is None and os.path.exists(f"{save_dir}/{hash(e)}.npy"):
                        new_diagram.value = backend.load(f"{save_dir}/{hash(e)}.npy")
                    else:
                        new_diagram.value_pointer = len(diagram_list)
                        diagram_list.append(new_diagram)
                else:
                    new_diagram.value_pointer = len(diagram_list)
                    diagram_list.append(new_diagram)
            return new_diagram
        elif isinstance(e, list):
            return [collect_diagrams(item) for item in e]
        elif isinstance(e, tuple):
            return tuple(collect_diagrams(item) for item in e)
        elif isinstance(e, dict):
            return {key: collect_diagrams(value) for key, value in e.items()}
        elif isinstance(e, Add):
            terms = Add.make_args(e)
            result = None
            for term in terms:
                collected_term = collect_diagrams(term)
                if result is None:
                    result = collected_term
                else:
                    result = result + collected_term
            return result
        elif isinstance(e, Mul):
            terms = Mul.make_args(e)
            result = 1
            for term in terms:
                collected_term = collect_diagrams(term)
                result = result * collected_term
            return result
        elif isinstance(e, Pow):
            base = collect_diagrams(e.base)
            exp = collect_diagrams(e.exp)
            return Pow(base, exp)
        elif hasattr(e, "__array__") and hasattr(e, "shape"):  # Process numpy array
            result = np.zeros_like(e, dtype=object)
            for index in np.ndindex(e.shape):
                result[index] = collect_diagrams(e[index])
            return result
        elif isinstance(e, sp.Basic) and e.args:
            # Generic recursion for SymPy objects (Matrix, etc.) not handled above
            return e.func(*[collect_diagrams(arg) for arg in e.args])
        elif isinstance(e, sp.Number):
            return e
        else:
            return e

    return collect_diagrams(expr)


def _build_combined(diagram_list, vertex_map, propagator_map, debug, timing=None):
    """Build combined_diagrams, all_vertices, all_propagators, all_times (without time_map)."""
    all_propagators = []
    all_time_vertex_pairs = []
    pair_to_index = {}
    propagator_to_index = {}

    t0 = perf_counter()
    for diagram in diagram_list:
        for i, (vertex, time) in enumerate(
            zip(diagram.vertex_list, diagram.time_list)
        ):
            pair = (time, vertex)
            if pair not in pair_to_index:
                pair_to_index[pair] = len(all_time_vertex_pairs)
                all_time_vertex_pairs.append(pair)
        for p in diagram.propagator_list:
            if p not in propagator_to_index:
                propagator_to_index[p] = len(all_propagators)
                all_propagators.append(p)
    if timing is not None:
        timing["build_collect_pairs"] = perf_counter() - t0
        timing["n_unique_time_vertex_pairs"] = len(all_time_vertex_pairs)
        timing["n_unique_propagators"] = len(all_propagators)

    combined_diagrams = []
    original_to_new_time_vertex = {}
    original_to_new_propagator = {}
    t0 = perf_counter()
    for diagram in diagram_list:
        did = id(diagram)
        original_to_new_time_vertex[did] = {
            i: pair_to_index[(time, vertex)]
            for i, (vertex, time) in enumerate(
                zip(diagram.vertex_list, diagram.time_list)
            )
        }
        original_to_new_propagator[did] = {
            i: propagator_to_index[p] for i, p in enumerate(diagram.propagator_list)
        }
    if timing is not None:
        timing["build_index_mapping"] = perf_counter() - t0

    t0 = perf_counter()
    for diagram in diagram_list:
        did = id(diagram)
        n_vertices = len(all_time_vertex_pairs)
        new_adjacency = [[0 for _ in range(n_vertices)] for _ in range(n_vertices)]
        tv_map = original_to_new_time_vertex[did]
        prop_map = original_to_new_propagator[did]
        old_adjacency = diagram.diagram.adjacency_matrix
        for i in range(len(diagram.time_list)):
            for j in range(len(diagram.time_list)):
                value = old_adjacency[i][j]
                if value != 0:
                    new_i = tv_map[i]
                    new_j = tv_map[j]
                    if isinstance(value, int):
                        new_adjacency[new_i][new_j] = prop_map[value]
                    elif isinstance(value, list):
                        new_adjacency[new_i][new_j] = [
                            (prop_map[v] if v != 0 else 0) for v in value
                        ]
        combined_diagrams.append(QuarkDiagram(new_adjacency))
    if timing is not None:
        timing["build_adjacency"] = perf_counter() - t0

    all_vertices = [pair[1] for pair in all_time_vertex_pairs]
    all_times = [pair[0] for pair in all_time_vertex_pairs]
    irrep_vertices = list(all_vertices)

    if not debug:
        if vertex_map is not None:
            t0 = perf_counter()
            for i, vertex in enumerate(all_vertices):
                new_vertex = vertex_map(vertex)
                if new_vertex is not None:
                    all_vertices[i] = new_vertex
            if timing is not None:
                timing["vertex_map_total"] = perf_counter() - t0
        if propagator_map is not None:
            t0 = perf_counter()
            for i, propagator in enumerate(all_propagators):
                if propagator in propagator_map:
                    all_propagators[i] = propagator_map[propagator]
            if timing is not None:
                timing["propagator_map_replace"] = perf_counter() - t0

    return combined_diagrams, all_vertices, all_propagators, all_times, irrep_vertices


def calc_diagram_prepare(
    expr: Union[Expr, List, Any],
    propagator_map: Dict = None,
    vertex_map: Callable = None,
    save_dir=None,
    debug=False,
    timing: Dict = None,
):
    """
    Prepare expression for diagram calculation. Applies collect, build, vertex_map, propagator_map.
    Use calc_diagram_eval(prepared, time_map) inside loop for time_map-varying computation.

    If timing is a dict, it will be populated with per-stage elapsed seconds, e.g.:
      collect_diagrams_1, collect_diagrams_2, build_collect_pairs, build_adjacency,
      vertex_map_total, propagator_map_replace, n_diagrams.
    """
    from sympy import Add, Mul, Pow, Symbol

    backend = get_backend()
    if debug:
        save_dir = None
    if expr is None:
        return None

    prepare_t0 = perf_counter()
    diagram_list = []
    t0 = perf_counter()
    expr = _collect_diagrams(expr, diagram_list, save_dir, backend)
    if timing is not None:
        timing["collect_diagrams_1"] = perf_counter() - t0
    t0 = perf_counter()
    expr = _collect_diagrams(expr, diagram_list, save_dir, backend)
    if timing is not None:
        timing["collect_diagrams_2"] = perf_counter() - t0
        timing["n_diagrams"] = len(diagram_list)

    if not diagram_list:
        return _CalcDiagramPrepared(expr=expr, diagram_list=[], combined_diagrams=[], all_vertices=[], all_propagators=[], all_times=[], irrep_vertices=[], save_dir=save_dir, debug=debug, backend=backend, timing=timing)

    build_timing = timing if timing is not None else None
    combined_diagrams, all_vertices, all_propagators, all_times, irrep_vertices = _build_combined(
        diagram_list, vertex_map, propagator_map, debug, timing=build_timing
    )
    t_finalize = perf_counter()
    prepared = _CalcDiagramPrepared(
        expr=expr,
        diagram_list=diagram_list,
        combined_diagrams=combined_diagrams,
        all_vertices=all_vertices,
        all_propagators=all_propagators,
        all_times=all_times,
        irrep_vertices=irrep_vertices,
        save_dir=save_dir,
        debug=debug,
        backend=backend,
        timing=timing,
    )
    if timing is not None:
        timing["prepare_finalize"] = perf_counter() - t_finalize
        _PREPARE_TIME_KEYS = (
            "collect_diagrams_1",
            "collect_diagrams_2",
            "build_collect_pairs",
            "build_index_mapping",
            "build_adjacency",
            "vertex_map_total",
            "propagator_map_replace",
            "prepare_finalize",
        )
        timed_sum = sum(timing.get(k, 0.0) for k in _PREPARE_TIME_KEYS)
        timing["prepare_unaccounted"] = perf_counter() - prepare_t0 - timed_sum
    return prepared


class _CalcDiagramPrepared:
    """Holder for prepared diagram computation state."""

    __slots__ = ("expr", "diagram_list", "combined_diagrams", "all_vertices", "all_propagators", "all_times", "irrep_vertices", "save_dir", "debug", "backend", "timing")

    def __init__(self, expr, diagram_list, combined_diagrams, all_vertices, all_propagators, all_times, irrep_vertices, save_dir, debug, backend, timing=None):
        self.expr = expr
        self.diagram_list = diagram_list
        self.combined_diagrams = combined_diagrams
        self.all_vertices = all_vertices
        self.all_propagators = all_propagators
        self.all_times = all_times
        self.irrep_vertices = irrep_vertices
        self.save_dir = save_dir
        self.debug = debug
        self.backend = backend
        self.timing = timing


def calc_diagram_bind(
    prepared: "_CalcDiagramPrepared",
    vertex_map: Callable,
    timing: Dict = None,
):
    """
    Apply cfg-dependent vertex_map (Meson.load) to prepared skeleton.
    Call once per cfg after propagator.load and before calc_diagram_eval.
    """
    if prepared is None:
        return None

    if not prepared.irrep_vertices:
        return prepared

    call_times = []
    t0 = perf_counter()
    bound_vertices = []
    for vertex in prepared.irrep_vertices:
        t_call = perf_counter()
        new_vertex = vertex_map(vertex)
        call_times.append(perf_counter() - t_call)
        if new_vertex is None:
            bound_vertices.append(vertex)
        else:
            bound_vertices.append(new_vertex)
    t_assign = perf_counter()
    prepared.all_vertices = bound_vertices
    if timing is not None:
        timing["vertex_map_total"] = perf_counter() - t0
        timing["bind_assign"] = perf_counter() - t_assign
        timing["n_unique_time_vertex_pairs"] = len(prepared.irrep_vertices)
        if call_times:
            timing["vertex_map_call_count"] = len(call_times)
            timing["vertex_map_call_sum"] = sum(call_times)
            timing["vertex_map_call_min"] = min(call_times)
            timing["vertex_map_call_max"] = max(call_times)
            timing["vertex_map_call_avg"] = sum(call_times) / len(call_times)
    return prepared


def calc_diagram_eval(prepared: _CalcDiagramPrepared, time_map: Dict = None):
    """Apply time_map, compute diagrams, replace and return result. Use inside loop."""
    from sympy import Add, Mul, Pow, Symbol

    if prepared is None:
        return None

    expr = prepared.expr
    diagram_list = prepared.diagram_list
    save_dir = prepared.save_dir
    backend = prepared.backend

    if not diagram_list:
        return _replace_diagrams(expr, diagram_list, save_dir, backend)

    all_times = list(prepared.all_times)
    if time_map is not None:
        for i, t in enumerate(all_times):
            if t in time_map:
                all_times[i] = time_map[t]

    if not prepared.debug:
        results = compute_diagrams_multitime(
            prepared.combined_diagrams,
            all_times,
            prepared.all_vertices,
            prepared.all_propagators,
            multitime_shape=True,
        )
    else:
        results = [
            Symbol("result_{}".format(i)) for i in range(len(prepared.combined_diagrams))
        ]

    for i, diagram in enumerate(diagram_list):
        diagram.value = results[i]

    return _replace_diagrams(expr, diagram_list, save_dir, backend)


def _replace_diagrams(expr, diagram_list, save_dir, backend):
    def replace_diagrams(e):
        if isinstance(e, Diagram):
            # Use value_pointer pointed value
            if e.value_pointer is not None:
                if save_dir is not None:
                    if not os.path.exists(f"{save_dir}/{hash(e)}"):
                        backend.save(
                            f"{save_dir}/{hash(e)}", diagram_list[e.value_pointer].value
                        )
                return diagram_list[e.value_pointer].value
            elif e.value is not None:
                return e.value
            else:
                logger.debug("Diagram has no value_pointer")
                logger.debug(e.diagram.adjacency_matrix)
                logger.debug(e.vertex_list)
                logger.debug(e.time_list)
                logger.debug(e.propagator_list)
                return 1
                # raise ValueError("Diagram has no value_pointer")
        elif isinstance(e, list):
            return [replace_diagrams(item) for item in e]
        elif isinstance(e, tuple):
            return tuple(replace_diagrams(item) for item in e)
        elif isinstance(e, dict):
            return {key: replace_diagrams(value) for key, value in e.items()}
        elif isinstance(e, Add):
            terms = Add.make_args(e)
            result = None
            for term in terms:
                replaced_term = replace_diagrams(term)
                if result is None:
                    result = replaced_term
                else:
                    result += replaced_term
            return result
        elif isinstance(e, Mul):
            terms = Mul.make_args(e)
            result = 1
            for term in terms:
                replaced_term = replace_diagrams(term)
                result *= replaced_term
            return result
        elif isinstance(e, Pow):
            base = replace_diagrams(e.base)
            return replace_diagrams(Pow(base, e.exp))
        elif hasattr(e, "__array__") and hasattr(e, "shape"):  # Process numpy array
            result = np.zeros_like(e, dtype=object)
            for index in np.ndindex(e.shape):
                result[index] = replace_diagrams(e[index])
            return result
        elif isinstance(e, sp.Basic) and e.args:
            # Generic recursion for SymPy objects (Matrix, etc.) not handled above
            return e.func(*[replace_diagrams(arg) for arg in e.args])
        elif isinstance(e, sp.Number):
            return complex(e)
        else:
            return e

    return replace_diagrams(expr)


def calc_diagram(
    expr: Union[Expr, List, Any],
    time_map: Dict = None,
    propagator_map: Dict = None,
    vertex_map: Callable = None,
    save_dir=None,
    debug=False,
):
    """
    Find all Diagram objects in the expression and calculate their values.
    For loop over time_map only, use calc_diagram_prepare + calc_diagram_eval instead.
    """
    prepared = calc_diagram_prepare(
        expr, propagator_map=propagator_map, vertex_map=vertex_map, save_dir=save_dir, debug=debug
    )
    return calc_diagram_eval(prepared, time_map=time_map)


def quark_contract(expr, particles, degenerate=True):
    """
    Perform quark contraction based on hadron flavor structure in the expression

    Args:
        expr: Expression containing HadronFlavorStructure objects
        particles: List of particles
        degenerate: Whether to consider u and d quark degeneracy

    Returns:
        diagrams: Contraction diagrams
        coeffs: Coefficient list
        particles: Particle list
        propagators: Propagator list
    """
    from .flavor_structure import HadronFlavorStructure, Qurak, Propagator
    from .base_types import Tag
    from .symmetry.sympy_utils import convert_pow_to_mul

    diagrams = []
    coeffs = []
    propagators = [None]
    expr = convert_pow_to_mul(expr.expand())
    num_particles = len(particles)
    # Expand expression into sum of terms
    terms = Add.make_args(expr)
    result_terms = []
    baryon_num_list = []
    time_list = []
    baryon_num_list_finished = False
    for term in terms:
        # Decompose factors
        factors = Mul.make_args(term)
        coeff = S(1)
        symbol_list = []
        hadron_id = 0
        for factor in factors:
            if isinstance(factor, HadronFlavorStructure):
                if not baryon_num_list_finished:
                    baryon_num_list.append(factor.baryon_num)
                    time_list.append(factor.time)
                # Collect quarks and anti-quarks
                if factor.baryon_num == 0:
                    symbol_list.extend(
                        [
                            Qurak(
                                factor.anti_quark_list[0],
                                Tag(hadron_id * 3, factor.time),
                                True,
                            ),
                            Qurak(
                                factor.quark_list[0],
                                Tag(hadron_id * 3, factor.time),
                                False,
                            ),
                        ]
                    )
                elif factor.baryon_num == 1:
                    quark_id = 0
                    for q in factor.quark_list:
                        symbol_list.append(
                            Qurak(q, Tag(hadron_id * 3 + quark_id, factor.time), False)
                        )
                        quark_id += 1
                elif factor.baryon_num == -1:
                    quark_id = 0
                    for q in factor.anti_quark_list:
                        symbol_list.append(
                            Qurak(q, Tag(hadron_id * 3 + quark_id, factor.time), True)
                        )
                        quark_id += 1
                hadron_id += 1
            else:
                # Non-hadron flavor structure factors as coefficients
                coeff *= factor
        baryon_num_list_finished = True
        # Perform quark contraction
        result_list = []
        result = []

        _quark_contract(symbol_list, result_list, result, degenerate)
        result_terms.append(coeff * Add(*result_list))
    # Merge results and simplify
    terms = Add.make_args(simplify(Add(*result_terms)).expand())

    for term in terms:
        diagram = [[0 for i in range(num_particles)] for j in range(num_particles)]
        for i in range(num_particles):
            for j in range(num_particles):
                if baryon_num_list[i] != 0 and baryon_num_list[j] != 0:
                    diagram[i][j] = [[0 for _ in range(3)] for _ in range(3)]
                elif baryon_num_list[i] != 0 and baryon_num_list[j] == 0:
                    diagram[i][j] = [[0 for _ in range(3)] for _ in range(1)]
                elif baryon_num_list[i] == 0 and baryon_num_list[j] != 0:
                    diagram[i][j] = [[0 for _ in range(1)] for _ in range(3)]
        factors = Mul.make_args(term)
        coeff = S(1)
        for factor in factors:
            if isinstance(factor, Propagator):
                if factor.tag not in propagators:
                    propagators.append(factor.tag)
                hadron_id_source = factor.source_tag.tag // 3
                hadron_id_sink = factor.sink_tag.tag // 3
                quark_id_source = factor.source_tag.tag % 3
                quark_id_sink = factor.sink_tag.tag % 3
                if (
                    baryon_num_list[hadron_id_source] == 0
                    and baryon_num_list[hadron_id_sink] == 0
                ):
                    diagram[hadron_id_source][hadron_id_sink] = propagators.index(
                        factor.tag
                    )
                else:
                    diagram[hadron_id_source][hadron_id_sink][quark_id_source][
                        quark_id_sink
                    ] = propagators.index(factor.tag)
            else:
                coeff *= factor

        diagrams.append(diagram)
        coeffs.append(coeff)
    diagram_expr = S(0)
    for i in range(len(diagrams)):
        diagram_expr += coeffs[i] * Diagram(
            QuarkDiagram(diagrams[i]), time_list, particles, propagators
        )
    return diagram_expr


def _quark_contract(symbol_list, result_list, result, degenerate):
    from .flavor_structure import Propagator
    from .base_types import Tag

    if symbol_list == []:
        result_list.append(Mul(*result))
        return
    for i, src in enumerate(symbol_list):
        if src.anti:
            break
    for j, snk in enumerate(symbol_list):
        if not snk.anti and snk.flavor == src.flavor:
            if i > j:
                symbol_list.pop(i)
                symbol_list.pop(j)
                factor = S(-1) ** (i - j - 1)
            else:
                symbol_list.pop(j)
                symbol_list.pop(i)
                factor = S(-1) ** (j - i)
            if degenerate and (snk.flavor == "u" or snk.flavor == "d"):
                prop = Propagator("q", src.tag, snk.tag)
            else:
                prop = Propagator(src.flavor, src.tag, snk.tag)
            result.append(factor * prop)
            _quark_contract(symbol_list, result_list, result, degenerate)
            result.pop()
            if i > j:
                symbol_list.insert(j, snk)
                symbol_list.insert(i, src)
            else:
                symbol_list.insert(i, src)
                symbol_list.insert(j, snk)
