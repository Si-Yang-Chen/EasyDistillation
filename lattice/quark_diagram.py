from itertools import product
import logging
import os
from fractions import Fraction
from time import perf_counter
from typing import Callable, Dict, List, Union, Any, Tuple, Optional

import numpy as np
from opt_einsum import contract
import sympy as sp
from sympy import Add, Mul, Pow, S, simplify
import hashlib

from lattice.spatial_structure import HadronIrrepRow
from lattice.scene_coefficients import labelled_set_partitions, scene_coefficients

from .backend import get_backend

# Propagator, meson and current handles live in their own module (they are a
# self-contained block sharing a loader-identity validation layer). Re-exported
# here so `from lattice.quark_diagram import Meson` keeps working.
from .propagators import (
    Particle,
    Meson,
    Current,
    Propagator,
    PropagatorLocal,
    PropagatorWithCurrent,
)

__all__ = [
    "Particle",
    "Meson",
    "Current",
    "Propagator",
    "PropagatorLocal",
    "PropagatorWithCurrent",
    "QuarkDiagram",
    "QuarkDiagramOriginal",
    "StateExpandedDiagram",
    "SceneExpandedDiagram",
    "CurrentVertexAdapter",
    "Diagram",
    "compute_diagrams",
    "compute_diagrams_multitime",
    "quark_contract",
    "diagram_simplify",
    "diagram_vertice_replace",
    "remove_unexpected_diagram",
    "remove_disconneted_diagram",
    "calc_diagram",
    "calc_diagram_prepare",
    "calc_diagram_bind",
    "calc_diagram_eval",
    "integer_partitions",
    "calculate_sampling_weight",
    "enumerate_point_scenes",
    "partition_to_constraints",
    "vertex_type_from_matrix",
]

logger = logging.getLogger(__name__)



# Fixed label sets for Einstein summation (opt_einsum)
_SUB_VECTOR = "NOPQRSTUVWXYZ"  # Eigenvector/color indices
_SUB_SPIN = "ABCDEFGHIJKLM"  # Spin indices
_SUB_POINT = "nopqrstuvwxyz"  # Point indices
_SUB_COLOR = "abcdefghijklm"  # Color indices

# Each propagator consumes TWO label slots -- ``node`` and ``node + 1``, one per end
# -- so a 13-letter alphabet caps a connected contraction group at 6 quark lines, not
# 13.  Exceeding it used to surface as a bare IndexError from inside subscript
# construction, which says nothing about the cause (02 SS8 gap 9).
MAX_PROPAGATORS_PER_GROUP = min(
    len(_SUB_SPIN), len(_SUB_VECTOR), len(_SUB_POINT), len(_SUB_COLOR)
) // 2


def _require_label_capacity(propagator_count):
    """Fail with the limit when a group has more quark lines than labels allow."""
    if propagator_count > MAX_PROPAGATORS_PER_GROUP:
        raise ValueError(
            f"this contraction group has {propagator_count} quark lines, but only "
            f"{MAX_PROPAGATORS_PER_GROUP} can be labelled: each propagator needs two "
            f"spin/eigenvector/point/colour slots and the fixed einsum alphabet has "
            f"{len(_SUB_SPIN)} of each. Split the diagram or widen the alphabet (F1.1)."
        )



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


class CurrentVertexAdapter:
    """Adapt equal-time assembled V2V current vertices to ``get(t)``.

    Temporal point-split terms are deliberately rejected here: they require
    term-wise endpoint contraction through ``contract_directed_current_v2v``.
    """

    def __init__(self, vertices_by_time):
        if not isinstance(vertices_by_time, dict) or not vertices_by_time:
            raise TypeError("CurrentVertexAdapter requires a non-empty time-to-vertex dict")
        values = {}
        source_ne = sink_ne = None
        for time, adapter in vertices_by_time.items():
            if isinstance(time, (bool, np.bool_)) or not isinstance(
                time, (int, np.integer)
            ):
                raise TypeError("CurrentVertexAdapter time keys must be integers")
            if not isinstance(adapter, dict):
                raise TypeError("CurrentVertexAdapter entries must be adapter mappings")
            if adapter.get("schema") != "lattice.current.assembler/v1":
                raise ValueError("CurrentVertexAdapter entry has an unsupported schema")
            if adapter.get("axes") != (
                "sink_spin", "source_spin", "sink_ne", "source_ne"
            ):
                raise ValueError("CurrentVertexAdapter entry has invalid V2V axes")
            term_count = adapter.get("term_count")
            if (
                isinstance(term_count, (bool, np.bool_))
                or not isinstance(term_count, (int, np.integer))
                or term_count <= 0
            ):
                raise ValueError("CurrentVertexAdapter entry has invalid term count")
            terms = adapter.get("terms")
            if not isinstance(terms, (list, tuple)) or len(terms) != term_count:
                raise TypeError("CurrentVertexAdapter entry requires complete term endpoint provenance")
            for term in terms:
                endpoints = term.get("endpoints") if isinstance(term, dict) else None
                expected = {"bar_time", "field_time", "link_origin_time", "temporal_point_split", "boundary"}
                if not isinstance(endpoints, dict) or set(endpoints) != expected:
                    raise TypeError("CurrentVertexAdapter entry has invalid endpoint provenance")
                for name in ("bar_time", "field_time", "link_origin_time"):
                    value = endpoints[name]
                    if isinstance(value, (bool, np.bool_)) or not isinstance(
                        value, (int, np.integer)
                    ):
                        raise TypeError("CurrentVertexAdapter endpoint times must be integers")
                if endpoints["boundary"] not in {"periodic", "open", "unbounded"}:
                    raise ValueError("CurrentVertexAdapter endpoint boundary is invalid")
                if not isinstance(
                    endpoints["temporal_point_split"], (bool, np.bool_)
                ):
                    raise TypeError("CurrentVertexAdapter temporal endpoint flag must be a bool")
                if endpoints["temporal_point_split"] or endpoints["bar_time"] != endpoints["field_time"]:
                    raise ValueError(
                        "CurrentVertexAdapter cannot consume point-split terms; "
                        "use contract_directed_current_v2v"
                    )
            value = np.asarray(adapter.get("vertex"))
            if (
                value.ndim != 4
                or value.shape[:2] != (4, 4)
                or not np.issubdtype(value.dtype, np.complexfloating)
                or not np.all(np.isfinite(value))
            ):
                raise ValueError(
                    "CurrentVertexAdapter vertex must be finite complex "
                    "(4, 4, sink_ne, source_ne)"
                )
            entry_ne = adapter.get("ne")
            if not isinstance(entry_ne, dict) or set(entry_ne) != {"source", "sink"}:
                raise TypeError("CurrentVertexAdapter entry is missing Ne provenance")
            counts = {}
            for axis in ("source", "sink"):
                axis_ne = entry_ne[axis]
                if not isinstance(axis_ne, dict) or set(axis_ne) != {"available", "used"}:
                    raise TypeError("CurrentVertexAdapter entry has invalid Ne provenance")
                available, used = axis_ne["available"], axis_ne["used"]
                if any(
                    isinstance(x, (bool, np.bool_))
                    or not isinstance(x, (int, np.integer))
                    for x in (available, used)
                ):
                    raise TypeError("CurrentVertexAdapter entry has invalid Ne provenance")
                available, used = int(available), int(used)
                if available < 0 or not 0 <= used <= available:
                    raise ValueError("CurrentVertexAdapter entry has invalid Ne bounds")
                counts[axis] = used
            entry_source, entry_sink = counts["source"], counts["sink"]
            if value.shape[2:] != (entry_sink, entry_source):
                raise ValueError(
                    "CurrentVertexAdapter vertex shape disagrees with Ne provenance"
                )
            if source_ne is None:
                source_ne, sink_ne = entry_source, entry_sink
            elif (source_ne, sink_ne) != (entry_source, entry_sink):
                raise ValueError("CurrentVertexAdapter entries must share source/sink Ne")
            values[int(time)] = value.transpose(1, 0, 3, 2)
        self._values = values
        self.used_source_ne = source_ne
        self.used_sink_ne = sink_ne
        self.usedNe = source_ne if source_ne == sink_ne else None
        self.smeared = False

    def get(self, time):
        if isinstance(time, (int, np.integer)) and not isinstance(
            time, (bool, np.bool_)
        ):
            try:
                return self._values[int(time)]
            except KeyError as exc:
                raise IndexError(
                    "CurrentVertexAdapter has no vertex for requested time"
                ) from exc
        indices = np.asarray(time)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise TypeError(
                "CurrentVertexAdapter time must be an integer or "
                "one-dimensional integer array"
            )
        try:
            return np.stack([self._values[int(index)] for index in indices], axis=0)
        except KeyError as exc:
            raise IndexError(
                "CurrentVertexAdapter has no vertex for requested time"
            ) from exc


def _flatten_paths_matrix(path):
    """Flatten one adjacency-matrix entry into its propagator labels.

    Same recursion as ``lattice.quark_draw._flatten_paths``; kept here so the
    validity check below does not depend on the drawing module.
    """
    if isinstance(path, int):
        return [path] if path != 0 else []
    if isinstance(path, list):
        labels = []
        for entry in path:
            labels.extend(_flatten_paths_matrix(entry))
        return labels
    raise ValueError(f"Invalid value {path} in the adjacency matrix")


def validate_adjacency_matrix(adjacency_matrix):
    """Check the quark-link invariants of an adjacency matrix.

    Every propagator entry is a quark line, so at each vertex the number of
    line-ends is that vertex's quark count. Since every quark must be linked
    (quark out, antiquark in) and there are only three hadrons, the allowed
    (outgoing, incoming) degree pairs are:

        (1, 1)  meson: one quark line in, one out — possibly both a self-loop
        (3, 0)  baryon at the source: its three quarks all leave
        (0, 3)  baryon at the sink: its three quarks all arrive

    (0, 0) is deliberately absent. A hadron with no lines would leave its quarks
    unlinked, which cannot happen; a hadron that contracts with nothing else
    contracts with itself, written as a non-zero diagonal entry
    (see gen_twopt_diagram.py's disconnected = [[2, 0], [0, 2]]).

    Anything else describes a hadron that cannot exist — a vertex with two
    quark lines out carries four quarks, one with a single dangling end leaves a
    quark unpaired — and the diagram would fail later, deep inside contraction
    or drawing, with an error far from the cause.

    Raises:
        ValueError: naming the first vertex that violates the invariant.
    """
    num_vertices = len(adjacency_matrix)
    degree = [[0, 0] for _ in range(num_vertices)]
    for i in range(num_vertices):
        row = adjacency_matrix[i]
        if len(row) != num_vertices:
            raise ValueError(
                f"Adjacency matrix must be square; row {i} has {len(row)} "
                f"entries, expected {num_vertices}."
            )
        for j in range(num_vertices):
            for _label in _flatten_paths_matrix(row[j]):
                degree[i][0] += 1   # a line leaves i
                degree[j][1] += 1   # ... and arrives at j
    for i, (outgoing, incoming) in enumerate(degree):
        if (outgoing, incoming) not in {(1, 1), (3, 0), (0, 3)}:
            hint = (
                "; a hadron that connects to nothing must self-loop, e.g. a "
                "non-zero diagonal entry"
                if (outgoing, incoming) == (0, 0)
                else ""
            )
            raise ValueError(
                f"Vertex {i} has {outgoing} outgoing and {incoming} incoming "
                f"quark lines; a hadron must be a meson (1, 1), or a baryon at "
                f"the source (3, 0) or sink (0, 3)" + hint + "."
            )


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
            _require_label_capacity(len(propagators))
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
        validate: bool = True,
        leg_offsets: List = None,
    ) -> None:
        """
        Initialize QuarkDiagram.

        Args:
            adjacency_matrix: Graph adjacency matrix representing the diagram
            vertex_list: List of vertex indices
            L: Spatial lattice size (total number of lattice points = L^3)
            usedNp: Number of sampled points (default: usedNp from Current vertex)
            debug: Enable debug output
            validate: Check the quark-link invariants (every quark carries one
                line, so a vertex is isolated, a meson, or a baryon at one end).
                Pass False only for mechanical fragments that are not complete
                diagrams.
            leg_offsets: Per-vertex term offsets, one entry per ``vertex_list``
                element.  Each entry is either ``None`` (the vertex keeps both legs
                on its anchor) or a list of ``((left_delta, right_delta), [term
                indices])`` offset classes as :func:`offset_classes` produces.  These
                come from the operator content, which is why they are an explicit
                input: a diagram built from irrep rows alone has no data handle and
                therefore no offsets.
        """
        self.adjacency_matrix = adjacency_matrix
        self.vertex_list = vertex_list
        self.leg_offsets = leg_offsets
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

        if validate:
            validate_adjacency_matrix(adjacency_matrix)

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
                leg_offsets=getattr(self, "leg_offsets", None),
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
            _require_label_capacity(len(propagators))
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
            _require_label_capacity(len(propagators))
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
        leg_offsets: List = None,
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
            leg_offsets: Per-vertex offset classes (see QuarkDiagram.leg_offsets).
        """

        # Override base class initialization for StateExpandedDiagram specific fields
        self.adjacency_matrix = adjacency_matrix
        self.vertex_list = vertex_list
        self.leg_offsets = leg_offsets
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

        _require_label_capacity(len(propagators))
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

    def _offset_terms_for(self, vertex_index):
        """Offset-class entries for one vertex, or ``None`` when it has no axis.

        One class means every term sits on the same slice, so there is nothing to
        choose and the vertex keeps the single-contraction arithmetic.
        """
        offsets = getattr(self, "leg_offsets", None)
        if not offsets or vertex_index >= len(offsets):
            return None
        entry = offsets[vertex_index]
        if not entry:
            return None
        # One class whose offsets are both zero means every term sits on the anchor:
        # the vertex block already sums them and nothing shifts, so there is no axis.
        # A single *splitting* class is different -- its legs still land on two
        # slices -- so the test is on the offsets, not on the class count.
        if len(entry) == 1 and tuple(entry[0][0]) == (0, 0):
            return None
        return list(entry)

    def expand_scenes(self) -> None:
        """
        Expand this state-expanded diagram into point coincidence scenes.

        Enumerate all possible point coincidence patterns for sampling groups and create
        SceneExpandedDiagram objects with corresponding weights and constraints.
        The results are stored in self.scene_diagrams.

        Uses L and usedNp parameters from the instance (set during initialization).
        """
        from itertools import product as iter_product

        from lattice.scene_coefficients import scene_coefficients

        # ``L`` here is the spatial extent; the number of lattice points is L**3.
        # ``calculate_sampling_weight`` applies that cube internally, so keep the
        # same convention at this boundary and cube only when solving.
        L = self.L  # spatial extent; total points = L**3
        N = self.usedNp  # Number of sampled points
        M = L**3  # Total number of lattice points (F5.6)

        # Offset-class combinations (02 SS3).  A vertex whose terms split its legs
        # contributes one axis per distinct (left_delta, right_delta) pair, and the
        # choice matters here rather than at evaluation: the pool key contains the
        # leg's time, so a (0,0) class (both legs on the anchor, one shared pool) and
        # a (0,1) class (legs split, two pools) enumerate *different* coincidence
        # scenes.  That is why the skeleton count is the sum over combinations of
        # their scene counts, not scene count times class count.
        class_axes = []
        for vertex_idx in range(len(self._vertex_state)):
            classes = self._offset_terms_for(vertex_idx)
            if classes:
                class_axes.append((vertex_idx, classes))
        if class_axes:
            combinations = []
            for choice in iter_product(
                *[range(len(classes)) for _, classes in class_axes]
            ):
                selected = [((0, 0), [None])] * len(self._vertex_state)
                for (vertex_idx, classes), class_index in zip(class_axes, choice):
                    selected[vertex_idx] = classes[class_index]
                combinations.append(tuple(selected))
        else:
            combinations = [tuple([((0, 0), [None])] * len(self._vertex_state))]

        # Initialize lists for storing scene diagrams, weights, and constraints
        self.scene_diagrams = []
        self.scene_weights = []
        self.scene_constraints = []

        for vertex_terms in combinations:
            self._expand_scenes_for(vertex_terms, M, N)

        if self.debug:
            logger.debug(
                f"Generated {len(self.scene_diagrams)} scene diagrams over "
                f"{len(combinations)} offset-class combinations"
            )

    def _expand_scenes_for(self, vertex_terms, M, N) -> None:
        """Enumerate the coincidence scenes for one offset-class combination.

        ``vertex_terms`` is the per-vertex offset-class choice; it decides the pool
        key of every point end, and therefore which ends can share a coordinate.
        """
        from itertools import product as iter_product

        self.sampling_groups = {}
        self._collect_sampling_groups(self._vertex_state, vertex_terms=vertex_terms)

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

        # Enumerate scenes for each sampling group.
        #
        # The scene coefficient is NOT w(k): a scene's contraction merges
        # subscripts, so it evaluates a free sum over its blocks and therefore also
        # counts tuples whose coincidence pattern is finer than the scene's.  The
        # unbiased weight is the solution of the zeta inversion Z^T c = w on the
        # refinement order -- see lattice/scene_coefficients.py and 02 SS5.0.
        # At r = 1 it collapses to w(1), which is what this code used to assume.
        group_scenes = {}
        for group_id, positions in self.sampling_groups.items():
            r = len(positions)
            partitions = labelled_set_partitions(r)
            coefficients = scene_coefficients(r, M, N)
            group_scenes[group_id] = [
                (partition, coefficient, positions)
                for partition, coefficient in zip(partitions, coefficients)
            ]

            if self.debug:
                logger.debug(
                    f"  Group {group_id}: {r} positions, {len(partitions)} scenes, "
                    f"coefficients={[str(c) for c in coefficients]}"
                )

        # Generate Cartesian product of scenes across all groups
        group_ids = list(group_scenes.keys())
        scene_combinations = iter_product(*[group_scenes[gid] for gid in group_ids])

        for scene_combo in scene_combinations:
            # Coefficients multiply across independently sampled groups, exactly as
            # the weights did; the value itself is now a scene coefficient rather
            # than w(k).
            total_coefficient = Fraction(1)
            all_positions = []
            all_partitions = []

            for group_idx, (partition, coefficient, positions) in enumerate(scene_combo):
                total_coefficient *= coefficient
                all_positions.extend(positions)
                all_partitions.append((partition, positions))

            # Generate constraints from partitions
            constraints = self._build_scene_constraints(all_partitions)

            if self.debug:
                logger.debug(
                    f"    Scene: coefficient={total_coefficient}, constraints={constraints}"
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
            self.scene_weights.append(total_coefficient)
            self.scene_constraints.append(constraints)

        if self.debug:
            logger.debug(f"Generated {len(self.scene_diagrams)} scene diagrams")

    def _collect_sampling_groups(self, vertex_state: tuple, vertex_terms=None) -> None:
        """Collect sampling groups from ``vertex_state``.

        A sampling group is keyed by ``(declared group id, leg time)`` (F5.1).  The
        declared id is the ``vertex_list`` entry -- that is the part the caller
        chooses -- and the leg's time comes from the vertex's anchor plus that leg's
        own term offset, which is what makes designs A and B one rule rather than two
        code paths (02 SS4):

        * two legs at the same time share the group and their coincidence patterns
          are enumerated (design B, all points distinct within a slice);
        * two legs at different times land in different groups automatically, so no
          coincidence is asserted between them -- which is the case the old code got
          silently wrong.

        Both legs of one vertex share its anchor, so "same time" inside a vertex is
        just "same offset".

        Args:
            vertex_state: Tuple of dicts, one per vertex, each with left/right -> 'v'/'p'.
            vertex_terms: Per-vertex offset-class choice, as produced by
                :func:`offset_classes`.  ``None`` means no term-wise handling.
        """
        for vertex_idx in range(len(vertex_state)):
            group_id = self.vertex_list[vertex_idx]
            for side in ("left", "right"):
                if vertex_state[vertex_idx][side] != "p":
                    continue
                leg_time = self._leg_time(vertex_idx, side, vertex_terms)
                key = (group_id, leg_time)
                self.sampling_groups.setdefault(key, []).append((vertex_idx, side))

        if self.debug:
            logger.debug(f"\n  Collected sampling groups:")
            for key, positions in self.sampling_groups.items():
                logger.debug(
                    f"    Group {key}: {positions} ({len(positions)} point positions)"
                )

    def _leg_time(self, vertex_idx, side, vertex_terms):
        """The leg's offset from its vertex anchor, for sampling-group identity.

        Only the offset matters, not the absolute time: both legs of a vertex share
        its anchor, so "same time" reduces to "same offset".  Returning the offset
        keeps the skeleton independent of the actual time, as layer B requires
        (02 SS4).

        The offsets come from the operator, which is why they enter as an explicit
        argument: a ``QuarkDiagram`` built from irrep rows alone has no data handles
        and therefore no offsets, and in that case every leg sits on the anchor.
        """
        if vertex_terms is None:
            return 0
        entry = vertex_terms[vertex_idx] if vertex_idx < len(vertex_terms) else None
        if not entry:
            return 0
        # An entry is ``((left_delta, right_delta), [term indices])``; accept a bare
        # offset pair too, so callers can supply either form.
        offsets = entry[0] if isinstance(entry[0], (tuple, list)) else entry
        left, right = offsets
        return int(left) if side == "left" else int(right)

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




def _first_term(entry):
    """The representative term index of an offset-class entry, or ``None``."""
    if not entry:
        return None
    indices = entry[1]
    return indices[0] if indices else None


def _vertex_terms_for(vertex_terms, vertex_index):
    """The offset-class entry chosen for one vertex, or ``None`` if there is none."""
    if vertex_terms is None or vertex_index >= len(vertex_terms):
        return None
    return vertex_terms[vertex_index]


def _shift_time(time, delta):
    """Shift an anchor time by a leg offset, leaving arrays alone if delta is 0.

    A zero delta must not copy the (possibly large) time array: that would be
    preparing a duplicate for a translation, which F3.3 forbids.
    """
    if delta == 0:
        return time
    if isinstance(time, (int, np.integer)):
        return int(time) + int(delta)
    return np.asarray(time) + int(delta)


def _supports_vertex_terms(vertex):
    """Whether a vertex can describe where each of its legs sits in time.

    This is the single predicate the graph uses to decide term-wise handling
    (02 SS2): a plain meson does not implement it and keeps calling ``get(t)``
    exactly as before, while a current whose terms can split the legs does.
    """
    return callable(getattr(vertex, "term_time_offsets", None))


def vertex_leg_offsets(vertex, anchor_time, term_index=None):
    """Return ``(left_delta, right_delta)`` for one vertex's two legs.

    A plain meson keeps both legs on its own slice, so this is ``(0, 0)``.  A
    current's left leg faces the sink and its right leg faces the source, so bar is
    the left delta and field the right one (02 SS2).

    ``term_index=None`` means "no term choice was made", which is only legal for a
    vertex whose terms do not split its legs; an equal-time current is contracted as
    a whole and must not be enumerated term by term.
    """
    probe = getattr(vertex, "term_time_offsets", None)
    if not callable(probe):
        return 0, 0
    if term_index is None:
        split = getattr(vertex, "is_point_split", None)
        if callable(split) and split():
            raise ValueError(
                "a vertex whose terms split its legs in time needs an explicit "
                "term choice; none was made"
            )
        return 0, 0
    left, right = probe(term_index)
    return int(left), int(right)


def offset_classes(vertex):
    """The distinct ``(left_delta, right_delta)`` pairs a vertex's terms realise.

    Terms sharing an offset pair must be contracted as one block rather than
    enumerated separately: the enumeration cost is the number of offset *classes*
    (2 for the temporal conserved current), not the number of terms, and that is why
    ten point-split currents are still only 1024 contractions (01 SSF5, F2.3).

    Returns ``[(offsets, [term indices])]`` in first-appearance order, or
    ``[((0, 0), [None])]`` for a vertex that does not split its legs.
    """
    if not _supports_vertex_terms(vertex):
        return [((0, 0), [None])]

    count = getattr(vertex, "term_count", None)
    if count is None:
        terms = getattr(vertex, "terms", None)
        if terms is None:
            raise TypeError(
                f"vertex of type {type(vertex).__name__} implements "
                "term_time_offsets but exposes neither term_count nor terms"
            )
        count = len(terms)

    classes = []
    index_of = {}
    for term_index in range(int(count)):
        offsets = vertex_leg_offsets(vertex, None, term_index)
        if offsets not in index_of:
            index_of[offsets] = len(classes)
            classes.append((offsets, []))
        classes[index_of[offsets]][1].append(term_index)
    return classes


def compute_diagrams_multitime(
    diagrams: List[QuarkDiagram],
    time_list,
    vertex_list: List[Meson],
    propagator_list: List[Propagator],
    multitime_shape: int = False,
    debug: bool = False,
    coefficients: List = None,
):
    """
    Compute diagram values with automatic sampling weight application.

    When ``coefficients`` is given, ``diagrams`` is already the list of scene
    contractions (one entry per sector x scene) and each is contracted with its own
    subscripts and multiplied by its own coefficient.  That is the form
    ``calc_diagram_prepare`` produces, and passing the coefficients explicitly keeps
    the expansion in layer B instead of re-deriving it during evaluation.

    Otherwise a Diagram carrying a current vertex is expanded here into its state
    diagrams and their point-coincidence scenes.

    The coefficients are NOT ``w(k) = C(L^3,k)/C(usedNp,k)``. A scene's contraction
    merges subscripts, so it evaluates a free sum over its blocks and therefore also
    counts tuples whose coincidence pattern is finer than the scene's; the unbiased
    coefficient is the solution of the zeta inversion ``Z^T c = w`` on the
    refinement order (see ``lattice.scene_coefficients`` and 02 SS5.0). At r = 1 it
    collapses to ``w(1)``, which is what this code used to assume.

    Each scene is contracted with **its own** subscripts and multiplied by **its own**
    coefficient, and the results are summed.  Collapsing to one coefficient per state
    would apply one scene's weight to every scene's algebra.
    """
    backend = get_backend()

    if coefficients is not None:
        if len(coefficients) != len(diagrams):
            raise ValueError(
                "coefficients must have one entry per scene contraction "
                f"({len(diagrams)} diagrams, {len(coefficients)} coefficients)"
            )
        # Already expanded by calc_diagram_prepare: the units are the diagrams.
        diagrams_to_compute = list(zip(diagrams, coefficients))
    else:
        # Expand each Diagram into its (sector x scene) contraction units.  The scene
        # is the unit that carries the coincidence constraints and the coefficient,
        # so consuming the state instead would drop both (02 SS8 gap 1/2).  A plain
        # Diagram is not itself a contraction unit -- its inner QuarkDiagram is.
        diagrams_to_compute = []
        for diagram in diagrams:
            for scene_diagram, coefficient in _scene_units(diagram):
                unit = getattr(scene_diagram, "operands", None)
                if unit is not None:
                    diagrams_to_compute.append((scene_diagram, coefficient))
                else:
                    inner = getattr(scene_diagram, "diagram", None)
                    if inner is None or not getattr(inner, "operands", None):
                        raise ValueError(
                            "a contraction unit must expose operands; got "
                            f"{type(scene_diagram).__name__}"
                        )
                    diagrams_to_compute.append((inner, coefficient))

    if debug:
        logger.debug(
            f"Total scene contractions: {len(diagrams_to_compute)} "
            f"(from {len(diagrams)} input diagrams)"
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

    # Offset classes (02 SS3).  A vertex whose terms split its legs in time
    # contributes one axis per distinct ``(left_delta, right_delta)`` pair; terms
    # sharing a pair are contracted as one block, so the cost is the number of
    # classes rather than the number of terms (01 SSF5, F2.3).  A plain meson or an
    # equal-time current yields no axis and keeps the single-contraction path.
    axes = []
    for vertex_index, vertex in enumerate(vertex_list):
        if not _supports_vertex_terms(vertex):
            continue
        classes = offset_classes(vertex)
        if len(classes) == 1 and tuple(classes[0][0]) == (0, 0):
            # Every term sits on the anchor: no axis, and the vertex block already
            # sums its terms, so this must stay byte-identical to today's path.  A
            # single *splitting* class is different and still needs its shifted legs.
            continue
        anchor = time_list[vertex_index]
        if not isinstance(anchor, (int, np.integer)):
            # A multitime vertex has no single anchor, so its legs have no single
            # offset: "each element of the scan gets its own offsets" contradicts
            # per-term offsets (N2).  This is a semantic conflict, so say so rather
            # than let it surface as an einsum rank error.
            raise ValueError(
                f"vertex {vertex_index} is both a multitime vertex and a point-split "
                "vertex: a scanned time has no single anchor for the per-term leg "
                "offsets to be measured from.  Scan a different vertex, or fix this "
                "one's time (N2, F8.2)."
            )
        axes.append((vertex_index, classes))

    if axes:
        from itertools import product as _iter_product

        combinations = []
        for choice in _iter_product(*[range(len(classes)) for _, classes in axes]):
            chosen = [((0, 0), [None])] * len(vertex_list)
            for (vertex_index, classes), class_index in zip(axes, choice):
                chosen[vertex_index] = classes[class_index]
            combinations.append(tuple(chosen))
    else:
        combinations = [(tuple([((0, 0), [None])] * len(vertex_list)))]

    for diagram_idx, (diagram, scene_coefficient) in enumerate(diagrams_to_compute):
        if debug:
            logger.debug(f"\n{'='*80}")
            logger.debug(f"Processing scene contraction {diagram_idx}")
            logger.debug(f"  coefficient: {scene_coefficient}")
            logger.debug(f"  offset-class combinations: {len(combinations)}")
            logger.debug(f"{'='*80}")
        # Each offset-class combination is one term block of the splitting vertices.
        # A contraction is linear in a vertex block, so the classes are **summed**;
        # multiplying them would be wrong as soon as a vertex has two classes.  A
        # vertex that never splits its legs yields exactly one combination, so a
        # meson and an equal-time current stay a single contraction with unchanged
        # arithmetic (F2.3, F7.1).
        scene_total = 0
        for vertex_terms in combinations:
            combination_value = 1.0
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

                    # Each end's time is its vertex's anchor shifted by that leg's
                    # offset (02 SS2).  The propagator's source end sits on the src
                    # vertex's right (source-facing) leg and its sink end on the snk
                    # vertex's left (sink-facing) leg, so each vertex contributes its own
                    # delta.  A plain meson reports (0, 0) and nothing moves.
                    source_term = _vertex_terms_for(vertex_terms, item[1])
                    sink_term = _vertex_terms_for(vertex_terms, item[2])
                    src_left, src_right = vertex_leg_offsets(
                        src_vertex, time_list[item[1]], _first_term(source_term)
                    )
                    snk_left, snk_right = vertex_leg_offsets(
                        snk_vertex, time_list[item[2]], _first_term(sink_term)
                    )
                    source_time = _shift_time(time_list[item[1]], src_right)
                    sink_time = _shift_time(time_list[item[2]], snk_left)

                    if debug and (
                        source_time is not time_list[item[1]]
                        or sink_time is not time_list[item[2]]
                    ):
                        logger.debug(
                            f"      Shifted leg times: source {time_list[item[1]]} -> "
                            f"{source_time}, sink {time_list[item[2]]} -> {sink_time}"
                        )

                    # Get propagator data based on type
                    try:
                        if prop_type == "VSV":
                            # Standard VSV: use get(t_source, t_sink) -> S_{i,j}
                            prop_data = propagator.get(source_time, sink_time)
                            # Slice both ends' usedNe
                            if usedNe_sink is not None:
                                prop_data = prop_data[..., :usedNe_sink, :]
                            if usedNe_source is not None:
                                prop_data = prop_data[..., :usedNe_source]
                            if debug:
                                logger.debug(
                                    f"      Called propagator.get(t_source={source_time}, t_sink={sink_time})"
                                )
                        elif prop_type == "VSP":
                            # VSP: sink=vector, source=point
                            # get_VSP_highmode handles usedNe_source=0 internally
                            prop_data = propagator.get_VSP_highmode(
                                source_time,
                                sink_time,
                                usedNe_source=usedNe_source,
                                usedNe_sink=usedNe_sink,
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
                                source_time,
                                sink_time,
                                usedNe_sink=usedNe_sink,
                                usedNe_source=usedNe_source,
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
                                source_time,
                                sink_time,
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
                        if not isinstance(source_time, (int, np.integer)) or not isinstance(
                            sink_time, (int, np.integer)
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
                # Distinct contraction groups of one combination multiply (they are
                # disconnected pieces of the same graph).
                combination_value = combination_value * result

                if debug:
                    logger.debug(
                        f"  Contraction successful! Result shape: {result.shape if hasattr(result, 'shape') else type(result)}"
                    )

            # Classes add: they are the terms of one vertex, and the contraction is
            # linear in that vertex's block.
            scene_total = scene_total + combination_value
        diagram_value.append(scene_total)
        # Apply this scene's own coefficient.  Every scene has been contracted with
        # its own subscripts by now, so the sum over scenes is the Horvitz-Thompson
        # estimate; a single weight per state would not be (F7.3).
        if hasattr(diagram, "scene_weights") and diagram.scene_weights:
            raise ValueError(
                "a scene must be contracted one at a time: this entry still carries "
                "a scene_weights list, so a whole state reached the contraction loop"
            )
        if scene_coefficient != 1:
            diagram_value[-1] = diagram_value[-1] * scene_coefficient
            if debug:
                logger.debug(f"\n  Applied scene coefficient: {scene_coefficient}")

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
                new_quark_diagram = QuarkDiagram(component_matrix, validate=False)
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


def vertex_type_from_matrix(adjacency_matrix, vertex_idx):
    """Infer whether a vertex is a meson or a baryon from the adjacency matrix.

    Mirrors the sizing done in ``quark_contract`` below, which builds each
    ``matrix[i][j]`` as ``[[0] * N for _ in range(M)]`` — shape ``(M, N)``, where
    the *column* dimension ``N`` belongs to vertex ``i`` and the row dimension
    ``M`` belongs to vertex ``j``::

        i=baryon, j=baryon -> (3, 3)
        i=baryon, j=meson  -> (1, 3)   # columns belong to i
        i=meson,  j=baryon -> (3, 1)
        i=meson,  j=meson  -> scalar

    A baryon carries three quark lines, so it is exactly the vertex whose row
    holds an entry with a three-wide column dimension. Mesons never reach 3.

    Reading the row dimension instead of the column silently misclassifies every
    meson in a mixed diagram; ``test_vertex_type.py`` pins all four combinations,
    so a change to the construction order fails a test instead of silently
    drawing the wrong vertex shape.

    Two written conventions reach here, and both must read as baryon:

    - ``quark_contract`` nests twice, ``matrix[i][j][source_quark][sink_quark]``;
    - hand-written matrices list the quark lines directly, ``[1, 1, 1]``.

    A one-level list of three labels is three quark lines from one vertex, which
    no meson can have (a meson is one quark line in each direction), so the list
    itself is the tell.
    """
    for other in range(len(adjacency_matrix)):
        path = adjacency_matrix[vertex_idx][other]
        if not isinstance(path, list) or not path:
            continue
        if isinstance(path[0], list):
            # two-level nesting (quark_contract): the column dimension belongs
            # to this vertex
            if len(path[0]) == 3:
                return "baryon"
        elif len(path) == 3:
            # one-level, hand-written: the entry lists the quark lines
            return "baryon"
    return "meson"


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


def _remap_paths(value, propagator_map):
    """Remap adjacency entries to combined propagator indices, recursively.

    Baryon entries nest: ``matrix[i][j][source_quark][sink_quark]`` holds one
    propagator index per quark pair.  A one-level comprehension cannot express
    that and dies with "unhashable type: 'list'" (02 SS8 gap 8), so recurse on any
    nested list rather than assuming a flat label list.
    """
    if isinstance(value, int):
        return propagator_map[value] if value != 0 else 0
    if isinstance(value, list):
        return [_remap_paths(item, propagator_map) for item in value]
    raise ValueError(f"invalid adjacency entry {value!r}")


def _scene_units(diagram):
    """Expand one Diagram into its (sector x scene) units with coefficients.

    A Diagram carrying current vertices expands into state diagrams, and each state
    expands into point-coincidence scenes.  The scene is the unit that carries the
    coincidence constraints and the coefficient, so every consumer must iterate
    scenes; consuming the state instead drops both (02 SS8 gap 1/2).

    The expansion hangs off the inner ``QuarkDiagram`` (``diagram.diagram``), not off
    the ``Diagram`` symbol itself: the symbol only carries the vertex/time lists.

    Returns:
        List of ``(scene_diagram, coefficient)``.  A plain diagram with no
        expansion yields ``[(diagram, 1)]``.
    """
    inner = getattr(diagram, "diagram", None)
    if inner is None:
        # Nothing to contract: keep the pair so the caller still sees one unit.
        return [(diagram, 1)]

    states = getattr(inner, "expanded_diagrams", None) or []
    if not states:
        return [(diagram, 1)]

    units = []
    for state in states:
        scenes = getattr(state, "scene_diagrams", None) or []
        if scenes:
            units.extend(zip(scenes, state.scene_weights))
        else:
            units.append((state, 1))
    return units


def _build_combined(diagram_list, vertex_map, propagator_map, debug, timing=None):
    """Build combined_diagrams, all_vertices, all_propagators, all_times (without time_map).

    ``combined_diagrams`` has one entry per scene contraction, and
    ``prepared.scene_groups`` records which entries belong to which input diagram
    together with each scene's coefficient.  Rebuilding the graph here without
    that information is what used to collapse four sectors into one.
    """
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
    scene_coefficients_out = []
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
    scene_ranges = []
    for diagram in diagram_list:
        did = id(diagram)
        n_vertices = len(all_time_vertex_pairs)
        tv_map = original_to_new_time_vertex[did]
        old_adjacency = diagram.diagram.adjacency_matrix
        range_start = len(combined_diagrams)

        # Rebuild one combined diagram per scene, carrying the scene's own
        # subscripts, vertex types and coefficients.  The skeleton therefore keeps
        # every sector: rebuilding once per input diagram would drop all but the
        # first (02 SS8 gap 1).
        for scene_diagram, coefficient in _scene_units(diagram):
            # A scene unit is normally a SceneExpandedDiagram, which carries its own
            # subscripts/types but not a propagator_list (its propagator indices live
            # in ``operands``); the parent Diagram's list is the one to remap.
            source = getattr(scene_diagram, "operands", None)
            source_is_scene = source is not None
            scene_prop_map = {
                i: propagator_to_index[p]
                for i, p in enumerate(diagram.propagator_list)
            }
            new_adjacency = [[0 for _ in range(n_vertices)] for _ in range(n_vertices)]
            for i in range(len(diagram.time_list)):
                for j in range(len(diagram.time_list)):
                    value = old_adjacency[i][j]
                    if value != 0:
                        new_adjacency[tv_map[i]][tv_map[j]] = _remap_paths(
                            value, scene_prop_map
                        )
            combined = QuarkDiagram(new_adjacency, validate=False)
            if source_is_scene:
                combined.operands = list(scene_diagram.operands)
                combined.subscripts = list(scene_diagram.subscripts)
                combined.propagator_types = list(scene_diagram.propagator_types)
                combined.vertex_types = list(scene_diagram.vertex_types)
                combined.vertex_infos = scene_diagram.vertex_infos
                combined.scene_constraints = list(scene_diagram.scene_constraints or [])
            combined_diagrams.append(combined)
            scene_coefficients_out.append(coefficient)
        scene_ranges.append((range_start, len(combined_diagrams)))
    if timing is not None:
        timing["build_adjacency"] = perf_counter() - t0
        timing["n_scene_contractions"] = len(combined_diagrams)

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

    return (
        combined_diagrams,
        all_vertices,
        all_propagators,
        all_times,
        irrep_vertices,
        scene_coefficients_out,
        scene_ranges,
    )


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
        return _CalcDiagramPrepared(expr=expr, diagram_list=[], combined_diagrams=[], all_vertices=[], all_propagators=[], all_times=[], irrep_vertices=[], scene_coefficients=[], scene_ranges=[], save_dir=save_dir, debug=debug, backend=backend, timing=timing)

    build_timing = timing if timing is not None else None
    (
        combined_diagrams,
        all_vertices,
        all_propagators,
        all_times,
        irrep_vertices,
        scene_coefficients_list,
        scene_range_list,
    ) = _build_combined(diagram_list, vertex_map, propagator_map, debug, timing=build_timing)
    t_finalize = perf_counter()
    prepared = _CalcDiagramPrepared(
        expr=expr,
        diagram_list=diagram_list,
        combined_diagrams=combined_diagrams,
        all_vertices=all_vertices,
        all_propagators=all_propagators,
        all_times=all_times,
        irrep_vertices=irrep_vertices,
        scene_coefficients=scene_coefficients_list,
        scene_ranges=scene_range_list,
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

    __slots__ = ("expr", "diagram_list", "combined_diagrams", "all_vertices", "all_propagators", "all_times", "irrep_vertices", "scene_coefficients", "scene_ranges", "save_dir", "debug", "backend", "timing")

    def __init__(self, expr, diagram_list, combined_diagrams, all_vertices, all_propagators, all_times, irrep_vertices, scene_coefficients=None, scene_ranges=None, save_dir=None, debug=False, backend=None, timing=None):
        self.expr = expr
        self.diagram_list = diagram_list
        self.combined_diagrams = combined_diagrams
        self.all_vertices = all_vertices
        self.all_propagators = all_propagators
        self.all_times = all_times
        self.irrep_vertices = irrep_vertices
        self.scene_coefficients = scene_coefficients if scene_coefficients is not None else []
        self.scene_ranges = scene_ranges if scene_ranges is not None else []
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
        scene_results = compute_diagrams_multitime(
            prepared.combined_diagrams,
            all_times,
            prepared.all_vertices,
            prepared.all_propagators,
            multitime_shape=True,
            coefficients=prepared.scene_coefficients or None,
        )
    else:
        scene_results = [
            Symbol("result_{}".format(i)) for i in range(len(prepared.combined_diagrams))
        ]

    # ``diagram_list`` holds one entry per distinct Diagram, while
    # ``combined_diagrams`` holds one entry per scene contraction; a diagram's value
    # is the sum over its scenes, each already weighted by its own coefficient.
    if prepared.scene_ranges:
        for i, diagram in enumerate(diagram_list):
            start, stop = prepared.scene_ranges[i]
            total = scene_results[start]
            for result in scene_results[start + 1 : stop]:
                total = total + result
            diagram.value = total
    else:
        for i, diagram in enumerate(diagram_list):
            diagram.value = scene_results[i]

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
