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

