import unittest
import numpy as np
from sympy import S, Matrix
import sympy as sp
from lattice.insertion import Insertion
from lattice.insertion.derivative import DerivativeName
from lattice.insertion.gamma import GammaName
from lattice.quark_diagram import remove_disconneted_diagram
from lattice.spatial_structure import (
    HadronIrrep,
    HadronIrrepRow,
)
from lattice.group_projection import (
    operator_transform,
    expr_little_group_projection,
    multi_exprs_little_group_projection,
    hadron_little_group_projection,
)
from lattice.symmetry.gen_hardcoded_rep import genLittleGroupIrrep
from lattice.base_types import Tag
from lattice.symmetry.sympy_utils import expr_simplify, find_linear_independent_exprs
from lattice.hadron import gen_correlator, operator_conjugate


class TestTransformExpr(unittest.TestCase):
    def setUp(self):
        self.tag = Tag("test", 0)
        self.hadron_row = HadronIrrepRow(
            hadron_name="rho",
            momentum=[0, 0, 0],
            irrep_name="T_1",
            row_idx=0,
            parity=-1,
            tag=self.tag,
        )
        self.hadron_row2 = HadronIrrepRow(
            hadron_name="pi",
            momentum=[1, 0, 0],
            irrep_name="A_2",
            row_idx=0,
            parity=None,
            tag=self.tag,
        )
        self.expr = self.hadron_row * self.hadron_row2

    def test_expr_transform(self):
        """Test expression transformation with group elements"""
        transformed = operator_transform(self.expr, "c4x")
        # The transformation should preserve the expression structure
        self.assertIsNotNone(transformed)
        # For this specific case, the transformation should equal the original
        self.assertEqual(transformed, self.expr)


class TestProjectionFunctions(unittest.TestCase):
    def setUp(self):
        self.tag = Tag(0, 0)
        self.hadron = HadronIrrep(
            hadron_name="rho",
            momentum=[0, 0, 0],
            irrep_name="T_1",
            parity=-1,
            tag=self.tag,
        )
        self.hadron1 = HadronIrrep(
            hadron_name="rho",
            momentum=[1, 0, 0],
            irrep_name="E",
            parity=None,
            tag=self.tag,
        )
        self.hadron2 = HadronIrrep(
            hadron_name="pi",
            momentum=[-1, 0, 0],
            irrep_name="A_2",
            parity=None,
            tag=self.tag,
        )
        self.expr = self.hadron[0]

    def test_hadron_little_group_projection_two_hadrons(self):
        """Test little group projection for two hadrons"""
        hadrons = [self.hadron1, self.hadron2]
        projected = hadron_little_group_projection(hadrons, "T_1", 0, -1, single_result=True)[0]
        parity_projected = sp.simplify(projected + operator_transform(projected, "inviden"))
        rotation_projected = operator_transform(projected, "c4x")

        # Test parity projection
        self.assertEqual(parity_projected, 0)
        # Test rotation invariance
        self.assertEqual(rotation_projected, projected)

    def test_hadron_little_group_projection_two_hadrons_conjugate(self):
        """Test little group projection with conjugated hadrons"""
        D_star_p2 = HadronIrrep("D_star", [0, 1, 1], "B_2", None, Tag(0, 0))
        D_meson_p1 = HadronIrrep("D", [0, -1, 0], "A_2", None, Tag(0, 0))
        rows_list_tmp = [
            operator_conjugate(hadron_little_group_projection([D_star_p2, D_meson_p1], "E", 0, parity=None)[0]),
            operator_conjugate(hadron_little_group_projection([D_star_p2, D_meson_p1], "E", 1, parity=None)[0]),
        ]
        little_group = genLittleGroupIrrep([0, 0, 1], "E", None)
        for ele in little_group.keys():
            for j in range(len(rows_list_tmp)):
                rotated = operator_transform(rows_list_tmp[j], ele)
                expected = S(0)
                for i in range(len(rows_list_tmp)):
                    expected += little_group[ele][i, j] * rows_list_tmp[i]
                diff = sp.simplify(expected - rotated)
                self.assertEqual(diff, 0)

    def test_expr_little_group_projection(self):
        """Test expression little group projection"""
        # Create a simple expression for testing
        expr = self.hadron[0]
        result = expr_little_group_projection(expr, "T_1", 0, parity=-1)
        self.assertIsNotNone(result)

    def test_multi_exprs_little_group_projection(self):
        """Test multiple expressions little group projection"""
        expr_list = [self.hadron[0], self.hadron[1]]
        result = multi_exprs_little_group_projection(expr_list, "T_1", 0, parity=-1)
        self.assertIsInstance(result, list)


class TestDiagramTransform(unittest.TestCase):
    def setUp(self):
        from lattice.flavor_structure import HadronFlavorStructure
        from lattice.hadron import Hadron

        charmed = [
            HadronFlavorStructure("dc"),
            HadronFlavorStructure("uc"),
        ]

        charmed_bar = [
            HadronFlavorStructure("cu"),
            -HadronFlavorStructure("cd"),
        ]
        DDbar_isoscalar = charmed[0] * charmed_bar[1] - charmed[1] * charmed_bar[0]
        DbarD_isoscalar = -charmed_bar[0] * charmed[1] + charmed_bar[1] * charmed[0]
        I0_Cm = ((DDbar_isoscalar - DbarD_isoscalar) * (1 / S(2))).expand()

        D_star_p2 = HadronIrrep("D_star", [0, 1, 1], "B_2", None, Tag(0, 0))
        D_meson_p1 = HadronIrrep("D", [0, -1, 0], "A_2", None, Tag(0, 0))
        rows_list_tmp = [
            hadron_little_group_projection([D_star_p2, D_meson_p1], "E", 0, parity=None)[0],
            hadron_little_group_projection([D_star_p2, D_meson_p1], "E", 1, parity=None)[0],
        ]

        self.hadrons_list = [Hadron(row, I0_Cm) for row in rows_list_tmp]
        self.correlator = gen_correlator([self.hadrons_list, self.hadrons_list])
        self.correlator = remove_disconneted_diagram(self.correlator, [Rf"S^c_\mathrm{{local}}"])

    def test_correlator_symmetry(self):
        """Test that correlator is invariant under time reversal"""
        rows_list_tmp = [self.correlator[0][0] + self.correlator[1][0], self.correlator[0][1] + self.correlator[1][1]]
        little_group = genLittleGroupIrrep([0, 0, 1], "E", None)
        for ele in little_group.keys():
            for j in range(len(rows_list_tmp)):
                rotated = operator_transform(rows_list_tmp[j], ele, 1)
                expected = S(0)
                for i in range(len(rows_list_tmp)):
                    expected += little_group[ele][i, j] * rows_list_tmp[i]
                diff = sp.simplify(expected - rotated)
                self.assertEqual(diff, 0)


class TestMesonLittleGroupProjection(unittest.TestCase):
    def test_ins_symmetry(self):
        """Test that the insertion is invariant under time reversal"""
        from lattice.insertion.mom_dict import momDict_mom9
        from lattice.insertion import (
            Insertion,
            Operator,
            GammaName,
            DerivativeName,
            ProjectionName,
        )
        from lattice.symmetry.hardcoded_rep import refRotateDict

        rest_rep = "T_1"
        rotation_matrix = genLittleGroupIrrep([0, 0, 0], rest_rep, -1)
        insertion_dict = {}
        for key, value in refRotateDict["0,1,1"].items():
            # if key != "-1,0,-1":
            #     continue
            # print(key, value)
            momentum = [int(ele) for ele in key.split(",")]
            insertion_dict[key] = Insertion(
                GammaName.RHO, DerivativeName.IDEN, "T_1", momDict_mom9
            ).little_group_projection(momentum, "B_1")
            # print(insertion_dict[key])
            # print(rotation_matrix[value])


if __name__ == "__main__":
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestMesonLittleGroupProjection("test_ins_symmetry"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
