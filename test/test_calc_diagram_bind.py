import unittest
from unittest.mock import MagicMock

from lattice.quark_diagram import _CalcDiagramPrepared, calc_diagram_bind


class TestCalcDiagramBind(unittest.TestCase):
    def test_bind_replaces_vertices_from_irrep_skeleton(self):
        irrep_a = object()
        irrep_b = object()
        meson_a = MagicMock(name="meson_a")
        meson_b = MagicMock(name="meson_b")

        prepared = _CalcDiagramPrepared(
            expr=None,
            diagram_list=[],
            combined_diagrams=[],
            all_vertices=[irrep_a, irrep_b],
            all_propagators=[],
            all_times=[0, 1],
            irrep_vertices=[irrep_a, irrep_b],
            save_dir=None,
            debug=False,
            backend=None,
            timing=None,
        )

        def vertex_map(vertex):
            if vertex is irrep_a:
                return meson_a
            if vertex is irrep_b:
                return meson_b
            raise AssertionError("unexpected vertex")

        timing = {}
        calc_diagram_bind(prepared, vertex_map, timing=timing)

        self.assertEqual(prepared.all_vertices, [meson_a, meson_b])
        self.assertEqual(prepared.irrep_vertices, [irrep_a, irrep_b])
        self.assertIn("vertex_map_total", timing)
        self.assertEqual(timing["n_unique_time_vertex_pairs"], 2)

    def test_bind_is_idempotent_source(self):
        irrep = object()
        meson = MagicMock(name="meson")
        prepared = _CalcDiagramPrepared(
            expr=None,
            diagram_list=[],
            combined_diagrams=[],
            all_vertices=[irrep],
            all_propagators=[],
            all_times=[0],
            irrep_vertices=[irrep],
            save_dir=None,
            debug=False,
            backend=None,
            timing=None,
        )

        calc_diagram_bind(prepared, lambda v: meson)
        calc_diagram_bind(prepared, lambda v: meson)

        self.assertEqual(prepared.irrep_vertices, [irrep])
        self.assertEqual(prepared.all_vertices, [meson])


if __name__ == "__main__":
    unittest.main()
