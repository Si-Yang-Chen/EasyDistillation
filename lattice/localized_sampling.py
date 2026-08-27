from __future__ import annotations

import numpy as np


def current_vertex_groups(source_is_current: bool, sink_is_current: bool, coincident_time: bool):
    """QuarkDiagram vertex_list for a two-point blending contraction.

    Distinct timeslices (dt != 0) give independent sparsened sets: groups 1 and 2.
    Contact time (dt = 0) shares one set: both currents use group 1. That is the
    analogue of arXiv:2505.01719 Omega^(2) versus the factorized Omega^(1)_i
    Omega^(1)_j that holds only for x_4 != y_4.

    Meson interpolators stay group 0 (no point sampling).
    """
    source_group = 1 if source_is_current else 0
    if not sink_is_current:
        sink_group = 0
    elif coincident_time and source_is_current:
        sink_group = source_group
    else:
        sink_group = 2 if source_is_current else 1
    return [source_group, sink_group]


def requires_lowmode_tensors(*used_ne_values) -> bool:
    """Ne=0 is sparse-point only: skip VSV, overlap, and I_low projection."""
    return any(int(n) > 0 for n in used_ne_values)


def split_sink_times_by_sampling(
    source_time, sink_times, independent_diagram, correlated_diagram, force_correlated
):
    """Choose correlated weights at contact time and independent weights otherwise."""
    sink_times = np.asarray(sink_times)
    if force_correlated and correlated_diagram is not None:
        return [(correlated_diagram, sink_times)]
    same_time = sink_times == source_time
    selections = []
    if np.any(~same_time):
        selections.append((independent_diagram, sink_times[~same_time]))
    if np.any(same_time):
        selections.append(
            (
                correlated_diagram
                if correlated_diagram is not None
                else independent_diagram,
                sink_times[same_time],
            )
        )
    return selections
