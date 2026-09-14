from typing import List, NamedTuple
from feynman.diagrams import Diagram
from feynman import Operator, Vertex
import matplotlib.pyplot as plt

from lattice.quark_diagram import vertex_type_from_matrix

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

r2l_u = dict(
    style="elliptic",
    ellipse_excentricity=4.0,
    ellipse_spread=-0.1,
    arrow=True,
)
r2l_d = dict(
    style="elliptic",
    ellipse_excentricity=4.0,
    ellipse_spread=0.1,
    arrow=True,
    # arrow_param={'width': 0.05},
)
l2r_u = dict(
    style="elliptic",
    ellipse_excentricity=4.0,
    ellipse_spread=0.1,
    arrow=True,
)
l2r_d = dict(
    style="elliptic",
    ellipse_excentricity=4.0,
    ellipse_spread=-0.1,
    arrow=True,
    # arrow_param={'width': 0.05},
)
d2u_l = dict(
    style="elliptic",
    ellipse_excentricity=0.25,
    ellipse_spread=0.3,
    arrow=True,
)
d2u_r = dict(
    style="elliptic",
    ellipse_excentricity=0.25,
    ellipse_spread=-0.3,
    arrow=True,
)

u2d_l = dict(
    style="elliptic",
    ellipse_excentricity=0.25,
    ellipse_spread=-0.3,
    arrow=True,
)
u2d_r = dict(
    style="elliptic",
    ellipse_excentricity=0.25,
    ellipse_spread=0.3,
    arrow=True,
)


def _flatten_paths(path):
    """Flatten one adjacency-matrix entry into its propagator labels.

    Two conventions exist and both must work:

    - hand-written matrices use a flat list of labels, ``[1, 1, 1]`` — one entry
      per quark line leaving the baryon;
    - ``quark_contract`` writes baryon contractions as two-level nesting,
      ``matrix[i][j][source_quark][sink_quark]``, where an inner ``0`` means that
      quark pair is *not* connected.

    Recursing and dropping zeros handles both. Dropping zeros is what makes the
    two-level form work: the sizing in ``quark_contract`` reserves a slot for
    every quark pair, but only the pairs that actually carry a propagator are
    non-zero — and every quark must be linked, so the surviving count is exactly
    the number of lines to draw.
    """
    if isinstance(path, int):
        return [path] if path != 0 else []
    if isinstance(path, list):
        labels = []
        for entry in path:
            labels.extend(_flatten_paths(entry))
        return labels
    raise ValueError(f"Invalid value {path} in the adjacency matrix")


def draw_diagram(diagram, adjacency_matrix, operator_list, line_color_list):
    num_vertex = len(adjacency_matrix)
    outward_idx = [0] * num_vertex
    inward_idx = [0] * num_vertex
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
                    propagators.extend(
                        [label, i, j] for label in _flatten_paths(path)
                    )
        if propagators == []:
            continue
        print(propagators)
        for propagator in propagators:
            style = {}
            op_out = operator_list[propagator[1]]
            op_in = operator_list[propagator[2]]
            xy_out = op_out.xy
            xy_in = op_in.xy
            if xy_out[0] == xy_in[0]:
                if xy_out[1] < xy_in[1]:
                    if xy_out[0] > 0.5:
                        style = d2u_l
                    if xy_out[0] < 0.5:
                        style = d2u_r
                if xy_out[1] > xy_in[1]:
                    if xy_out[0] > 0.5:
                        style = u2d_l
                    if xy_out[0] < 0.5:
                        style = u2d_r
            elif xy_out[1] == xy_in[1]:
                if xy_out[0] < xy_in[0]:
                    style = l2r_u
                if xy_out[0] > xy_in[0]:
                    style = r2l_d
            style["color"] = line_color_list[propagator[0]]
            diagram.line(
                op_out.vertex_out[outward_idx[propagator[1]]],
                op_in.vertex_in[inward_idx[propagator[2]]],
                **style,
            )
            outward_idx[propagator[1]] += 1
            inward_idx[propagator[2]] += 1


class Meson(NamedTuple):
    vertex_out: List[Vertex]
    vertex_in: List[Vertex]
    operator: Operator
    xy: tuple


class Baryon(NamedTuple):
    vertex_out: List[Vertex]
    vertex_in: List[Vertex]
    operator: Operator
    xy: tuple


def meson_source(diagram, xy, size, tag):
    vertex_out = [diagram.vertex(xy, dy=size, marker="")]
    vertex_in = [diagram.vertex(xy, dy=-size, marker="")]
    operator = diagram.operator([vertex_out[0], vertex_in[0]], c=2)
    operator.text(tag)
    return Meson(vertex_out, vertex_in, operator, xy)


def meson_sink(diagram, xy, size, tag):
    vertex_in = [diagram.vertex(xy, dy=size, marker="")]
    vertex_out = [diagram.vertex(xy, dy=-size, marker="")]
    operator = diagram.operator([vertex_in[0], vertex_out[0]], c=2)
    operator.text(tag)
    return Meson(vertex_out, vertex_in, operator, xy)


def baryon_source(diagram, xy, size, tag):
    vertex_out = []
    vertex_out.append(diagram.vertex(xy, dy=size, marker=""))
    vertex_out.append(diagram.vertex(xy, dx=size / 2, marker=""))
    vertex_out.append(diagram.vertex(xy, dy=-size, marker=""))
    operator = diagram.operator([vertex_out[0], vertex_out[2]], c=2)
    operator.text(tag)
    return Baryon(vertex_out, [], operator, xy)


def baryon_sink(diagram, xy, size, tag):
    vertex_in = []
    vertex_in.append(diagram.vertex(xy, dy=size, marker=""))
    vertex_in.append(diagram.vertex(xy, dx=-size / 2, marker=""))
    vertex_in.append(diagram.vertex(xy, dy=-size, marker=""))
    operator = diagram.operator([vertex_in[0], vertex_in[2]], c=2)
    operator.text(tag)
    return Baryon([], vertex_in, operator, xy)


def make_operator(hadron, pos, **kwargs):
    if pos == "src":
        if hadron == "meson":
            return meson_source(**kwargs)
        elif hadron == "baryon":
            return baryon_source(**kwargs)
    elif pos == "snk":
        if hadron == "meson":
            return meson_sink(**kwargs)
        elif hadron == "baryon":
            return baryon_sink(**kwargs)
    else:
        raise ValueError(f"Invalid position: {pos}.")


ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks([])
ax.set_yticks([])

D1 = Diagram(ax)

op1 = meson_source(D1, (0.2, 0.7), 0.1, R"$\pi_1$")
op2 = meson_source(D1, (0.2, 0.3), 0.1, R"$\pi_2$")
op3 = meson_sink(D1, (0.8, 0.7), 0.1, R"$\pi_3$")
op4 = meson_sink(D1, (0.8, 0.3), 0.1, R"$\pi_4$")
# draw_diagram(D1, [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], [op1, op2, op3, op4]) # direct diagram
draw_diagram(D1, [[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 3], [1, 0, 0, 0]], [op1, op2, op3, op4], [None, "r", "b", "b"])

# op1 = baryon_source(D1, (.2, .5), 0.1, R"$N$")
# op2 = baryon_sink(D1, (.8, .5), 0.1, R"$N$")
# draw_diagram(D1, [[0, [1, 1, 1]], [0, 0]], [op1, op2])

# op1 = baryon_source(D1, (.2, .5), .1, R"$N$")
# op3 = baryon_sink(D1, (.8, .3), .1, R"$N$")
# op2 = meson_sink(D1, (.8, .7), .1, R"$\pi$")
# draw_diagram(D1, [[0, 1, [1, 1]], [0, 0, 1], [0, 0, 0]], [op1, op2, op3])

# D1.plot()
# D1.show()


def _vertex_attributes_from_diagram(diagram):
    """Derive ``vertex_attribute_list`` from a quark-diagram ``Diagram``.

    A ``lattice.quark_diagram.Diagram`` already carries everything the drawing
    code needs: ``diagram.diagram.adjacency_matrix`` for the graph, and a
    ``vertex_list`` of ``HadronIrrepRow`` whose ``dagger`` flag gives the source
    /sink side and whose ``hadron_name`` gives the label.
    """
    adjacency_matrix = diagram.diagram.adjacency_matrix
    vertex_attribute_list = []
    for idx, vertex in enumerate(diagram.vertex_list):
        vertex_attribute_list.append(
            dict(
                pos="src" if vertex.dagger else "snk",
                type=vertex_type_from_matrix(adjacency_matrix, idx),
                name=rf"${vertex.hadron_name}$",
            )
        )
    return adjacency_matrix, vertex_attribute_list


def draw_quark_diagram(diagram, line_color_list=None, save_path=None):
    """Draw a quark diagram straight from a ``Diagram`` object.

    Unlike ``draw_single_diagram``, everything but the colours is read off the
    diagram itself: the adjacency matrix from ``diagram.diagram`` and the
    per-vertex side and label from ``diagram.vertex_list``.

    Args:
        diagram: a ``lattice.quark_diagram.Diagram`` instance.
        line_color_list: colour per propagator, indexed by the propagator label
            written into the adjacency matrix (labels start at 1, so the list
            needs one more entry than the highest label). Defaults to all
            ``None`` — the figure's default colour for every line.
        save_path: optional path passed through to ``draw_single_diagram``.
    """
    adjacency_matrix, vertex_attribute_list = _vertex_attributes_from_diagram(diagram)
    if line_color_list is None:
        highest = max(
            [p for row in adjacency_matrix for p in row if isinstance(p, int)]
            + [q for row in adjacency_matrix for p in row if isinstance(p, list) for q in p],
            default=0,
        )
        line_color_list = [None] * (highest + 1)
    return draw_single_diagram(
        adjacency_matrix, vertex_attribute_list, line_color_list, save_path
    )


def draw_multi_diagrams(adjacency_matrix_list, vertex_attribute_list, line_color_list, save_path=None):
    if save_path is None:
        save_path = [None] * len(adjacency_matrix_list)
    for im, isave in zip(adjacency_matrix_list, save_path):
        draw_single_diagram(im, vertex_attribute_list, line_color_list, isave)




def draw_single_diagram(adjacency_matrix, vertex_attribute_list, line_color_list, save_path=None):
    """Draw one quark diagram.

    Every vertex is drawn, including hadrons with no propagator attached: a
    quark always carries a link, but a hadron need not — disconnected pieces are
    a legitimate part of a correlation function. Vertices are therefore never
    dropped or reindexed, which keeps ``operator_list`` aligned with the vertex
    indices the propagators refer to. (The previous version filtered rows and
    columns for empty ones, which desynchronised the two when the filtered
    vertex had a lower index than a connected one.)
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    # usetex
    import matplotlib as mpl

    mpl.rc("text", usetex=True)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    diagram = Diagram(ax)
    num_vertex = len(adjacency_matrix)
    outward_idx = [0] * num_vertex
    inward_idx = [0] * num_vertex
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
                    propagators.extend(
                        [label, i, j] for label in _flatten_paths(path)
                    )
        if propagators == []:
            continue

        print(propagators)

        operator_list = [None] * len(vertex_attribute_list)
        n_src_op = sum([i["pos"] == "src" for i in vertex_attribute_list])
        n_snk_op = len(vertex_attribute_list) - n_src_op
        i_src = 0
        i_snk = 0
        size = 0.1
        for iop in range(len(vertex_attribute_list)):
            pos = vertex_attribute_list[iop]["pos"]
            type = vertex_attribute_list[iop]["type"]
            name = vertex_attribute_list[iop]["name"]
            if pos == "src":
                y_tmp = (i_src - 0.5) if n_src_op // 2 == 1 else (i_src)
                xy_tmp = (0.2, 0.5 + y_tmp / 2 * 0.6)
                print(f"src: {xy_tmp}, {n_src_op}")
                i_src += 1
                if type == "meson":
                    operator_list[iop] = meson_source(diagram, xy_tmp, size, name)
                elif type == "baryon":
                    operator_list[iop] = baryon_source(diagram, xy_tmp, size, name)
                else:
                    raise ValueError(f"Invalid hadron type: {type}.")
            elif pos == "snk":
                y_tmp = (i_snk - 0.5) if n_snk_op // 2 == 1 else (i_snk)
                xy_tmp = (0.8, 0.5 + y_tmp / 2 * 0.6)
                print(f"snk: {xy_tmp}, {n_snk_op}")
                i_snk += 1
                if type == "meson":
                    operator_list[iop] = meson_sink(diagram, xy_tmp, size, name)
                elif type == "baryon":
                    operator_list[iop] = baryon_sink(diagram, xy_tmp, size, name)
                else:
                    raise ValueError(f"Invalid hadron type: {type}.")
            else:
                raise ValueError(f"Invalid position: {pos}.")

        print(propagators)
        for propagator in propagators:
            style = {}
            visited_out_idx = propagator[1]
            visited_in_indice = propagator[2]
            op_out = operator_list[visited_out_idx]
            op_in = operator_list[visited_in_indice]
            xy_out = op_out.xy
            xy_in = op_in.xy
            if xy_out[0] == xy_in[0]:
                if xy_out[1] < xy_in[1]:
                    if xy_out[0] > 0.5:
                        style = d2u_l
                    if xy_out[0] < 0.5:
                        style = d2u_r
                if xy_out[1] > xy_in[1]:
                    if xy_out[0] > 0.5:
                        style = u2d_l
                    if xy_out[0] < 0.5:
                        style = u2d_r
            elif xy_out[1] == xy_in[1]:
                if xy_out[0] < xy_in[0]:
                    style = l2r_u
                if xy_out[0] > xy_in[0]:
                    style = r2l_d
            style["color"] = line_color_list[propagator[0]]  # add color
            diagram.line(
                op_out.vertex_out[outward_idx[visited_out_idx]],
                op_in.vertex_in[inward_idx[visited_in_indice]],
                **style,
            )
            outward_idx[visited_out_idx] += 1
            inward_idx[visited_in_indice] += 1
    diagram.plot()
    if save_path is not None:
        diagram.savefig(save_path)
    diagram.show()
