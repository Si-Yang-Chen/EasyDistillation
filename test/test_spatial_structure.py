import unittest
import numpy as np
from sympy import S, Matrix
import sympy as sp
from lattice.spatial_structure import (
    HadronIrrep,
    HadronIrrepRow,
)
from lattice.symmetry.gen_hardcoded_rep import genLittleGroupIrrep
from lattice.base_types import Tag
from lattice.symmetry.sympy_utils import expr_simplify, find_linear_independent_exprs


class TestHadronIrrep(unittest.TestCase):
    def setUp(self):
        self.tag = Tag(0, 0)
        self.hadron = HadronIrrep(
            hadron_name="pi",
            momentum=[0, 0, 0],
            irrep_name="T_1",
            parity=1,
            tag=self.tag,
        )

    def test_creation(self):
        self.assertEqual(self.hadron.hadron_name, "pi")
        self.assertEqual(self.hadron.momentum, [0, 0, 0])
        self.assertEqual(self.hadron.irrep_name, "T_1")
        self.assertEqual(self.hadron.parity, 1)
        self.assertEqual(self.hadron.tag, self.tag)
        self.assertEqual(self.hadron.lenth, 3)

    def test_equality(self):
        same_hadron = HadronIrrep(
            hadron_name="pi",
            momentum=[0, 0, 0],
            irrep_name="T_1",
            parity=1,
            tag=self.tag,
        )
        different_hadron = HadronIrrep(
            hadron_name="pi",
            momentum=[1, 0, 0],
            irrep_name="T_1",
            parity=1,
            tag=self.tag,
        )
        self.assertEqual(self.hadron, same_hadron)
        self.assertNotEqual(self.hadron, different_hadron)

    def test_copy(self):
        copied = self.hadron.copy()
        self.assertEqual(self.hadron, copied)


class TestHadronIrrepRow(unittest.TestCase):
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

    def test_creation(self):
        self.assertEqual(self.hadron_row.hadron_name, "rho")
        self.assertEqual(self.hadron_row.momentum, [0, 0, 0])
        self.assertEqual(self.hadron_row.irrep_name, "T_1")
        self.assertEqual(self.hadron_row.row_idx, 0)
        self.assertEqual(self.hadron_row.parity, -1)
        self.assertEqual(self.hadron_row.tag, self.tag)

    def test_equality(self):
        same_row = HadronIrrepRow(
            hadron_name="rho",
            momentum=[0, 0, 0],
            irrep_name="T_1",
            row_idx=0,
            parity=-1,
            tag=self.tag,
        )
        different_row = HadronIrrepRow(
            hadron_name="rho",
            momentum=[0, 0, 0],
            irrep_name="T_1",
            row_idx=1,
            parity=-1,
            tag=self.tag,
        )
        self.assertEqual(self.hadron_row, same_row)
        self.assertNotEqual(self.hadron_row, different_row)

    def test_transform(self):
        # Test identity transformation
        transformed = self.hadron_row.transform("c4x^-1")
        # print("test_transform", transformed, self.hadron_row)
        self.assertEqual(transformed, self.hadron_row)
        self.assertEqual(
            sp.simplify(
                1 / S(2) * transformed
                - 1 / S(2) * sp.sqrt(2) * transformed
                - (self.hadron_row - S(1) * sp.sqrt(2) * self.hadron_row) / S(2)
            ),
            0,
        )

    def test_conjugate(self):
        conjugated = self.hadron_row.conjugate()
        self.assertEqual(conjugated.dagger, True)
        self.assertEqual(conjugated.hadron_name, self.hadron_row.hadron_name)
        self.assertEqual(conjugated.momentum, self.hadron_row.momentum)


if __name__ == "__main__":
    unittest.main()
