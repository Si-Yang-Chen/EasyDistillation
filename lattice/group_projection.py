import sympy as sp
import numpy as np
from sympy import Expr, Matrix, I, S, Pow, Mul, Add, preorder_traversal
from typing import Dict, List, Tuple, Literal
from itertools import product

from lattice.quark_diagram import Diagram

from .symmetry.gen_hardcoded_rep import (
    genLittleGroupIrrep,
    momentunSymplify,
    littleGroup,
)
from .symmetry.hardcoded_rep import OhD_inv, irrep_row_connection_dict, refRotateDict
from .symmetry.sympy_utils import find_linear_independent_exprs
from .spatial_structure import HadronIrrepRow


# def operator_transform(expr, group_element, time=None):
#     """
#     Transform expression by applying group element to all HadronIrrepRow instances

#     Args:
#         expr: Expression containing HadronIrrepRow objects
#         group_element: Group element to apply

#     Returns:
#         Transformed expression
#     """

#     # Collect all HadronIrrepRow instances from the original expression
#     instances = set()
#     for sub_expr in preorder_traversal(expr):
#         if isinstance(sub_expr, HadronIrrepRow):
#             instances.add(sub_expr)

#         if isinstance(sub_expr, Diagram):
#             instances.add(sub_expr)

#     # Create replacement mapping: each instance replaced with its transformed result
#     replacements = {inst: inst.transform(group_element) for inst in instances}

#     # Apply replacements and return new expression
#     return expr.xreplace(replacements)


def operator_transform(expr: Expr, group_element, time=None) -> Expr:
    if isinstance(expr, list):
        return [operator_transform(item, group_element) for item in expr]
    # Process numpy array
    elif hasattr(expr, "__array__") and hasattr(expr, "shape"):
        original_shape = expr.shape
        flattened = expr.flatten() if hasattr(expr, "flatten") else expr.ravel()
        result = np.array(
            [operator_transform(item, group_element) for item in flattened],
            dtype=object,
        )
        return result.reshape(original_shape)
    elif isinstance(expr, dict):
        return {k: operator_transform(v, group_element) for k, v in expr.items()}
    elif isinstance(expr, tuple):
        return tuple(operator_transform(item, group_element) for item in expr)
    elif isinstance(expr, Matrix):
        return Matrix(operator_transform(expr.tolist(), group_element))
    elif isinstance(expr, Add):
        terms = Add.make_args(expr)
        return Add(*[operator_transform(term, group_element, time) for term in terms])
    elif isinstance(expr, Mul):
        factors = Mul.make_args(expr)
        return Mul(
            *[operator_transform(factor, group_element, time) for factor in factors]
        )
    elif isinstance(expr, Pow):
        base, exp = expr.as_base_exp()
        return Pow(operator_transform(base, group_element, time), exp)
    elif isinstance(expr, HadronIrrepRow):
        return expr.transform(group_element, time)
    elif isinstance(expr, Diagram):
        return expr.transform(group_element, time)
    else:
        return expr


def expr_little_group_projection(expr, irrep_name, row_idx, parity=None, p_ref=False):
    """
    Project expression to little group irrep

    Args:
        expr: Expression to project
        irrep_name: Irrep name
        row_idx: Row index
        parity: Parity
        p_ref: Whether to return reference momentum info

    Returns:
        Projected expression
    """
    from .spatial_structure import HadronIrrepRow

    momentum = np.array([0, 0, 0], dtype=int)
    terms = Add.make_args(expr)
    factors = Mul.make_args(terms[0])
    for factor in factors:
        if isinstance(factor, HadronIrrepRow):
            momentum += np.array(factor.momentum, dtype=int)
    p = momentunSymplify(momentum)
    p_str = ",".join([str(element) for element in p])
    for ref_p_str in refRotateDict.keys():
        if p_str in refRotateDict[ref_p_str].keys():
            break
    ref_p = [int(element) for element in ref_p_str.split(",")]
    expr_tmp = operator_transform(expr, OhD_inv(refRotateDict[ref_p_str][p_str]))
    matrix_group = genLittleGroupIrrep(ref_p, irrep_name, parity)
    len_irrep = matrix_group["iden"].shape[0]
    group_size = len(matrix_group.keys())
    projected_irrep_row = S(0)
    for key in matrix_group.keys():
        projected_irrep_row += (
            S(len_irrep)
            / S(group_size)
            * matrix_group[key][0, 0]
            * operator_transform(expr_tmp, key)
        )
    if p_ref:
        return projected_irrep_row, p_str, ref_p_str
    result = S(0)
    for tup in irrep_row_connection_dict[ref_p_str][irrep_name][row_idx]:
        result += tup[0] * operator_transform(projected_irrep_row, tup[1])
    result = operator_transform(result, refRotateDict[ref_p_str][p_str])
    return result


def multi_exprs_little_group_projection(
    expr_list, irrep_name, row_idx, parity=None, single_result=False
):
    """
    Project multiple expressions to little group irrep

    Args:
        expr_list: List of expressions to project
        irrep_name: Irrep name
        row_idx: Row index
        parity: Parity
        single_result: Whether to return single result

    Returns:
        List of projected expressions
    """
    result_expr_list = []
    for expr in expr_list:
        projected_expr, p_str, ref_p_str = expr_little_group_projection(
            expr, irrep_name, row_idx, parity, p_ref=True
        )
        if projected_expr != 0:
            if single_result:
                return [projected_expr]
            else:
                result_expr_list.append(projected_expr.expand())
    # return result_expr_list
    tmp = find_linear_independent_exprs(result_expr_list)
    result = tmp.copy()
    for i in range(len(tmp)):
        result[i] = S(0)
        for tup in irrep_row_connection_dict[ref_p_str][irrep_name][row_idx]:
            result[i] += tup[0] * operator_transform(tmp[i], tup[1])
        result[i] = operator_transform(result[i], refRotateDict[ref_p_str][p_str])
    return result


def hadron_little_group_projection(
    hadron_irreps: List,
    irrep_name,
    row_idx,
    parity=None,
    single_result=False,
):
    """
    Project hadron irreps to little group irrep

    Args:
        hadron_irreps: List of HadronIrrep objects
        irrep_name: Irrep name
        row_idx: Row index
        parity: Parity
        single_result: Whether to return single result

    Returns:
        List of projected expressions
    """
    from .spatial_structure import HadronIrrep

    momentum_structure = tuple(
        hadron_irreps[i].momentum for i in range(len(hadron_irreps))
    )
    momentum_total = [
        sum([momentum_structure[i][j] for i in range(len(momentum_structure))])
        for j in range(len(momentum_structure[0]))
    ]
    little_group = littleGroup(momentum_total)

    momentum_structure_dict = {}
    rotation = genLittleGroupIrrep([0, 0, 0], "T_1", -1)
    for key in little_group.keys():
        new_momentum_structure = [
            tuple(rotation[key] @ Matrix(momentum_structure[i]))
            for i in range(len(momentum_structure))
        ]
        momentum_structure_dict[tuple(new_momentum_structure)] = None
    exprs_mul_rows = []
    for momentum_structure in momentum_structure_dict.keys():
        hadrons_list = []
        for idx in range(len(hadron_irreps)):
            hadron_irrep = hadron_irreps[idx]
            new_hadron_irrep = HadronIrrep(
                hadron_irrep.hadron_name,
                list(momentum_structure[idx]),
                hadron_irrep.irrep_name,
                hadron_irrep.parity,
                hadron_irrep.tag,
            )
            hadrons_list.append(
                [new_hadron_irrep[j] for j in range(new_hadron_irrep.lenth)]
            )
        for expr in list(product(*hadrons_list)):
            exprs_mul_rows.append(Mul(*expr))
    # for i, expr in enumerate(exprs_mul_rows):
    #     print(f"{i}:", expr)
    return multi_exprs_little_group_projection(
        exprs_mul_rows, irrep_name, row_idx, parity, single_result
    )

    # INSERT_YOUR_CODE


def diagonalize_Cij(expr):
    """
    从形如 C_{ij}*A_i*B_j 的表达式中提取系数矩阵C_{ij}，对其对角化，
    并输出对角化后的表达式 sum_i λ_i A'_i B'_i

    其中A_i、B_j自动根据expr中的symbols推断，无需输入。
    """
    import sympy as sp
    from sympy import Symbol, Matrix

    # 1. 从expr中自动找出A_i, B_j的symbol
    # 寻找所有原子项，分A_i和B_j两组
    atoms = expr.atoms(Symbol)
    # 假设A_i, B_j以形如 'A0', 'A_1', 'B0', 'B_1' 命名
    # AB只是前面一两个symbol，不一定是A或B——自动根据expr的乘法结构提取前两个不同的symbol族为A_syms、B_syms
    from sympy import Symbol

    atoms = expr.atoms(Symbol)

    # 获取所有symbol名字的前缀（如"A2"->"A"，"Q_1"->"Q"），取前两个不同的族作为A、B族
    def get_symbol_prefix(sym):
        s = str(sym)
        # 以下划线或数字断开为前缀
        for i, c in enumerate(s):
            if c == "_" or c.isdigit():
                return s[:i]
        return s

    prefixes = []
    for sym in sorted(atoms, key=lambda x: str(x)):
        pre = get_symbol_prefix(sym)
        if pre not in prefixes:
            prefixes.append(pre)
        if len(prefixes) == 2:
            break
    if len(prefixes) < 2:
        raise ValueError("Failed to find two distinct symbol families in expr")
    prefix_A, prefix_B = prefixes
    A_syms = sorted(
        [s for s in atoms if get_symbol_prefix(s) == prefix_A], key=lambda x: str(x)
    )
    B_syms = sorted(
        [s for s in atoms if get_symbol_prefix(s) == prefix_B], key=lambda x: str(x)
    )
    n, m = len(A_syms), len(B_syms)
    if n == 0 or m == 0:
        raise ValueError("No A_i or B_j found in expr")
    # 2. 提取C_{ij}矩阵
    C = sp.Matrix.zeros(n, m)
    expr = sp.expand(expr)
    for i in range(n):
        for j in range(m):
            C[i, j] = expr.coeff(A_syms[i] * B_syms[j])
    # 3. 对C对角化（若C非对称用SVD，否则用eig）
    # 标准，若C对称，则eig，否则SVD
    if n == m and C == C.T:
        # 对称可以直接对角化
        P, D = C.diagonalize()
        # 新基矢 A', B' = P^{-1} * A, P^{-1} * B
        A_vec = Matrix(A_syms)
        B_vec = Matrix(B_syms)
        A_rot = P.inv() * A_vec
        B_rot = P.inv() * B_vec
        # 表达式 sum_i lambda_i * A'_i * B'_i
        diag_expr = sum(D[i, i] * A_rot[i] * B_rot[i] for i in range(n))
    else:
        # 一般情形下用SVD : C = U S V^T
        U, S_diag, V = C.singular_value_decomposition()
        # 新基矢 A'_i = U^T*A, B'_i = V^T*B
        # Note: For SVD, U and V may not be square matrices.
        # Since U^T*U = I and V^T*V = I (orthogonal columns), we use transpose instead of inverse
        A_vec = Matrix(A_syms)
        B_vec = Matrix(B_syms)
        # Use transpose for non-square matrices, or inv() for square matrices
        if U.shape[0] == U.shape[1]:
            A_rot = U.inv() * A_vec
        else:
            A_rot = U.T * A_vec
        if V.shape[0] == V.shape[1]:
            B_rot = V.inv() * B_vec
        else:
            B_rot = V.T * B_vec
        # 表达式 sum_i S_i * A'_i * B'_i
        diag_expr = sum(S_diag[i, i] * A_rot[i] * B_rot[i] for i in range(min(n, m)))
    return diag_expr
