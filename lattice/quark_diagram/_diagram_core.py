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
                logger.debug()

            # Create a new SceneExpandedDiagram for this scene combination
            scene_diagram = SceneExpandedDiagram(
                adjacency_matrix=self.adjacency_matrix,
                vertex_list=self.vertex_list,
                operands=self.operands,
                subscripts=self.subscripts,
                operands_data=self.operands_data,
                propagator_types=self.propagator_types,
                vertex_types=self.vertex_types,
                vertex_infos=getattr(self, "vertex_point_info", None),
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
                if vertex_idx < len(self.vertex_infos):
                    vertex_info = self.vertex_infos[vertex_idx]
                    logger.debug(
                        f"  Vertex {vertex_idx}, side '{side}': {vertex_info.get(side, 'N/A')}"
                    )
                else:
                    logger.debug(
                        f"  Vertex {vertex_idx}, side '{side}': vertex_idx out of range"
                    )
            logger.debug()


