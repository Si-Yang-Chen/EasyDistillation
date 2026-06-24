import sympy as sp
import numpy as np
from sympy import Matrix, I, S, Pow, Mul
from sympy.physics.quantum import Operator
from typing import Dict, List, Tuple, Literal

from opt_einsum import contract
from itertools import product

from .symmetry.utils import *
from .symmetry.gen_hardcoded_rep import (
    genLittleGroupIrrep,
    reductionToLittleGroup,
    wignerRotate,
)
from .symmetry.sympy_utils import *
from .flavor_structure import *
from .spatial_structure import HadronIrrepRow, HadronIrrep
from .quark_diagram import diagram_simplify, diagram_vertice_replace, quark_contract


class Hadron:
    def __init__(self, irep_row: Expr, flavor_structure: Expr):
        self._irrep_row = irep_row
        self._flavor_structure = flavor_structure

    @property
    def irrep_row(self) -> Expr:
        return self._irrep_row

    @property
    def flavor_structure(self) -> Expr:
        return self._flavor_structure

    def set_time(self, time: int) -> "Hadron":
        """
        Create a new Hadron instance with modified time value.
        The original instance remains unchanged.

        Args:
            time: The new time value to set

        Returns:
            A new Hadron instance
        """
        # Modify the tag in irrep_row
        new_irrep_row = set_time_in_expr(self._irrep_row, time)

        # Modify the tag in flavor_structure
        new_flavor_structure = set_time_in_expr(self._flavor_structure, time)

        # Return a new Hadron instance
        return Hadron(new_irrep_row, new_flavor_structure)

    def conjugate(self) -> "Hadron":
        """
        Create a new Hadron instance with conjugated values.
        The original instance remains unchanged.
        """
        # 修改irrep_row中的dagger值
        new_irrep_row = operator_conjugate(self._irrep_row.expand())
        new_flavor_structure = operator_conjugate(self._flavor_structure)
        return Hadron(new_irrep_row, new_flavor_structure)

    def copy(self) -> "Hadron":
        """
        Create a new Hadron instance with the same values.
        The original instance remains unchanged.
        """
        return Hadron(self._irrep_row, self._flavor_structure)


def set_time_in_expr(expr: Expr, time: int) -> Expr:
    """
    Recursively traverse the expression to modify the tag.time value of leaf nodes

    Args:
        expr: The expression to modify
        time: The new time value to set

    Returns:
        The modified expression
    """
    replacements = {}

    # Traverse all sub-expressions in the expression
    for sub_expr in sp.preorder_traversal(expr):
        if isinstance(sub_expr, HadronIrrepRow):
            # Create a new Tag, only modify the time value
            new_tag = Tag(sub_expr.tag.tag, time)
            # Create a new HadronIrrepRow instance
            new_row = HadronIrrepRow(
                sub_expr.hadron_name,
                sub_expr.momentum,
                sub_expr.irrep_name,
                sub_expr.row_idx,
                sub_expr.parity,
                new_tag,
            )
            replacements[sub_expr] = new_row
        elif isinstance(sub_expr, HadronFlavorStructure):
            # For MesonFlavorStructure, create a new instance
            new_meson = HadronFlavorStructure(sub_expr.flavor_str, time)
            replacements[sub_expr] = new_meson

    # Apply replacements and return the new expression
    return expr.xreplace(replacements)


def set_time_in_list(hadron_list: List[Hadron], time: int) -> List[Hadron]:
    """
    Apply set_time_in_expr function to each expression in the list, and set_time function to each Hadron
    """
    hadron_list_new = []
    for i in range(len(hadron_list)):
        hadron = hadron_list[i]
        hadron_list_new.append(hadron.set_time(time))
    return hadron_list_new


def operator_conjugate(expr: Expr) -> Expr:
    if isinstance(expr, Add):
        terms = Add.make_args(expr)
        return Add(*[operator_conjugate(term) for term in terms])
    elif isinstance(expr, Mul):
        factors = Mul.make_args(expr)
        return Mul(*[operator_conjugate(factor) for factor in factors[::-1]])
    elif isinstance(expr, Pow):
        base, exp = expr.as_base_exp()
        return Pow(operator_conjugate(base), exp)
    elif isinstance(expr, HadronIrrepRow):
        return expr.conjugate()
    elif isinstance(expr, HadronFlavorStructure):
        return expr.conjugate()
    else:
        return expr.conjugate()


def set_dagger_in_list(hadron_list: List[Hadron], dagger: bool) -> List[Hadron]:
    """
    对列表中的每个表达式应用set_dagger_in_expr函数，并且对每个Hadron调用设置dagger方法

    Args:
        hadron_list: 要处理的哈德龙列表
        dagger: 要设置的dagger值

    Returns:
        处理后的哈德龙列表
    """
    hadron_list_new = []
    for i in range(len(hadron_list)):
        hadron = hadron_list[i]
        if dagger:
            hadron_list_new.append(hadron.conjugate())
        else:
            hadron_list_new.append(hadron.copy())
    return hadron_list_new


def gen_correlator(hadrons: List[List[Hadron]], time_slice_list=None, dagger_list: List[bool] = None):
    """
    Calculate correlation functions for multiple Hadron lists

    Args:
        hadrons: List containing multiple lists of Hadron objects
        time_slice_list: List of time slices
        dagger_list: List of dagger values to apply to hadrons

    Returns:
        Correlation function
    """
    cache = {}
    if time_slice_list is None:
        time_slice_list = [i for i in range(len(hadrons))]
    if dagger_list is None:
        dagger_list = [False for _ in range(len(hadrons))]
        # 默认把最后一个设为True
        if len(dagger_list) >= 2:
            dagger_list[-1] = True

    for i in range(len(hadrons)):
        hadrons[i] = set_time_in_list(hadrons[i], time_slice_list[i])
        hadrons[i] = set_dagger_in_list(hadrons[i], dagger_list[i])
    result_matrix = np.ndarray(
        shape=tuple([len(hadrons[i]) for i in range(len(hadrons))]), dtype=object
    )
    # Create all possible index combinations
    indices_ranges = [range(len(hadrons[i])) for i in range(len(hadrons))]
    indices_combinations = list(product(*indices_ranges))
    # Iterate through all index combinations
    for indices in indices_combinations:
        # Build corresponding hadron_tuple
        hadron_tuple = tuple(hadrons[i][indices[i]] for i in range(len(hadrons)))
        position_wavefnc = convert_pow_to_mul(
            Mul(*[hadron_tuple[i].irrep_row for i in range(len(hadron_tuple))]).expand()
        )
        flavor_wavefnc = convert_pow_to_mul(
            Mul(*[hadron_tuple[i].flavor_structure for i in range(len(hadron_tuple))]).expand()
        )
        result = S(0)
        terms = Add.make_args(position_wavefnc)
        for term in terms:
            insersion_list = []
            factors = Mul.make_args(term)
            coeff = S(1)
            for factor in factors:
                if isinstance(factor, HadronIrrepRow):
                    insersion_list.append(factor)
                else:
                    coeff *= factor
            # Convert to a hashable form suitable for dictionary key
            contraction_key = str(flavor_wavefnc.expand())
            if contraction_key not in cache:
                term_of_result = diagram_simplify(
                    quark_contract(
                        flavor_wavefnc, np.arange(len(insersion_list)), degenerate=True
                    )
                )
                cache[contraction_key] = term_of_result
            else:
                term_of_result = cache[contraction_key]
            term_of_result = diagram_vertice_replace(
                term_of_result, {i: v for i, v in enumerate(insersion_list)}
            )
            result += term_of_result * coeff

        # Store hadron_tuple in corresponding position in result
        result_matrix[indices] = sp.simplify(result)
    return result_matrix
