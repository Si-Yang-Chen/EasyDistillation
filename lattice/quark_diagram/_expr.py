
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
