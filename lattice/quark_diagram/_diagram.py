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

