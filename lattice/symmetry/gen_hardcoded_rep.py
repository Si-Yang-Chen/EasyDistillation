import sympy as sp
import numpy as np
from sympy import GramSchmidt, Matrix, I, S
from sympy.physics.quantum import Operator
from .hardcoded_rep import (
    OD_irreps,
    Dic4_irreps,
    Dic2_irreps,
    Dic3_irreps,
    OhD_inv,
    OhD_mul,
    refRotateDict,
    little_group_reduction_map,
    irrep_row_connection_dict,
)
from typing import Dict, List, Tuple, Literal
from .group_generator import OhD_generator, Dic4_generator, Dic2_generator, Dic3_generator, C4_generator1, C4_generator2
from opt_einsum import contract
import copy
from itertools import product


def genMatrixGroupOhD(c4y: Matrix, c4z: Matrix, inv: Matrix = None):
    """
    Generate the irrep of group O with given generators c4y and c4z
    """
    iden = c4y.inv() @ c4y
    c4x = c4z.inv() @ c4y @ c4z
    c3 = c4x @ c4y
    c3x = (c4z @ c4y).inv()
    c3y = (c4x @ c4z).inv()
    c3z = (c4y @ c4x).inv()
    group = {
        "iden": iden,
        "c4x": c4x,
        "c2x": c4x @ c4x,
        "c4x^-1": c4x.inv(),
        "c4y": c4y,
        "c2y": c4y @ c4y,
        "c4y^-1": c4y.inv(),
        "c4z": c4z,
        "c2z": c4z @ c4z,
        "c4z^-1": c4z.inv(),
        "c3delta": c3,
        "c3delta^-1": c3.inv(),
        "c3gamma": c3x,
        "c3gamma^-1": c3x.inv(),
        "c3beta": c3y,
        "c3beta^-1": c3y.inv(),
        "c3alpha": c3z,
        "c3alpha^1": c3z.inv(),
        "c2e": c4y @ c4z @ c4y,
        "c2f": c4y @ c4y @ c4x,
        "c2c": c4y @ c4z @ c4z,
        "c2d": c4z @ c4z @ c4y,
        "c2a": c4y @ c4y @ c4z,
        "c2b": c4x @ c4x @ c4z,
    }
    for key in group.keys():
        group[key] = group[key].applyfunc(sp.simplify)
    group_tmp = group.copy()
    r_2pi = sp.simplify(c4x @ c4x @ c4x @ c4x)
    for key in group_tmp.keys():
        group[f"r{key}"] = r_2pi @ group_tmp[key]

    group_tmp = group.copy()
    if inv is not None:
        for key in group_tmp.keys():
            group[f"inv{key}"] = inv @ group_tmp[key]
    for key in group.keys():
        group[key] = group[key].applyfunc(sp.S)
        group[key] = group[key].applyfunc(sp.simplify)
    return group


def genIrrepOhD(
    irrep_name: Literal["A_1", "A_2", "E", "T_1", "T_2", "G_1", "G_2", "H"],
    parity: Literal[1, -1, None] = None,
    is_hardcoded: bool = True,
):
    """
    Generate the group element of the irrep.
    """
    if is_hardcoded:
        hardcode_irrep = OD_irreps[irrep_name].copy()
        if parity is not None:
            hardcode_irrep_tmp = copy.deepcopy(hardcode_irrep)
            for key in hardcode_irrep_tmp.keys():
                hardcode_irrep[f"inv{key}"] = S(parity) * hardcode_irrep_tmp[key]
        return hardcode_irrep
    else:
        generator_irrep = OhD_generator[irrep_name]
        c4y = generator_irrep["c4y"]
        c4z = generator_irrep["c4z"]
        if parity is not None:
            parity = Matrix(parity * np.eye(c4y.shape[0]))
        return genMatrixGroupOhD(c4y, c4z, parity)


def littleGroup(fixed_point=[0, 0, 0], group=None, elem=True):
    """
    Generate the little group element which keep the "fixed_point" fixed.
    """
    if group is None:
        group = genIrrepOhD("T_1", -1)
    little_group_element = {}
    fixed_point = Matrix(fixed_point)
    for key in group.keys():
        if (group[key] @ fixed_point - fixed_point).norm() < 0.01:
            if elem:
                little_group_element[key] = group[key]
            else:
                little_group_element[key] = None
    return little_group_element


def momentunSymplify(p):
    """
    p of the same little group is simplified to a representative element
    """
    p = [int(i) for i in p]
    classification = {0: []}
    for i in range(3):
        if abs(p[i]) in classification:
            classification[abs(p[i])].append(i)
        else:
            classification[abs(p[i])] = [i]
    p = [np.sign(ele) for ele in p]
    doublekey = 0
    singlekey = []
    for key in classification.keys():
        if len(classification[key]) != 1 and key != 0:
            doublekey = 1
        else:
            singlekey.append(key)
    for i in range(len(singlekey)):
        for index in classification[np.sort(singlekey)[i]]:
            p[index] *= i + doublekey
    return p


def genR_ref(p_ref, group=None, all=False):
    """
    Generate the R_ref that relate and reference momentum p_ref and any momentum. Stashed after hardcoding.
    """
    if group == None:
        group = genIrrepOhD("T_1", -1)
    wigner_dict = {}
    for ele in group.keys():
        pf = group[ele] @ Matrix(p_ref)
        pf_str = ",".join([str(element) for element in pf])
        if all:
            if pf_str not in wigner_dict.keys():
                wigner_dict[pf_str] = [ele]
            else:
                wigner_dict[pf_str].append(ele)
        else:
            if pf_str not in wigner_dict.keys():
                wigner_dict[pf_str] = ele
    return wigner_dict


def wignerRotate(p_i: list, ele: str):
    """
    Rotate wigner rotation for ele with initial momentum pi.
    """
    rotation = genIrrepOhD("T_1", -1)
    p_i = momentunSymplify(p_i)
    p_i_str = ",".join([str(element) for element in p_i])
    p_f = sp.sympify(rotation[ele] @ Matrix(p_i))
    p_f_str = ",".join([str(int(element)) for element in p_f])
    # 合并refRotateDict中的所有键值对到一个字典中
    merged_refRotateDict = {}
    for ref_mon, ref_dict in refRotateDict.items():
        if p_i_str in ref_dict.keys():
            ref_rotate_p_f_inv = OhD_inv(ref_dict[p_f_str])
            result = OhD_mul(OhD_mul(ref_rotate_p_f_inv, ele), ref_dict[p_i_str])
            return result
    raise NotImplementedError(f"{p_i_str} not finished")


def genLittleGroupIrrep(p, irrep_name, parity=None, p_ref=None, is_hardcoded=True, p_ref_irrep=False):
    """
    Generate the irrep of the little group with given generator and momentum p. Will be stashed after hardcoding the irrep of all little groups.
    """
    p = momentunSymplify(p)
    if sum([ele**2 for ele in p]) == 0:
        return genIrrepOhD(irrep_name, parity, is_hardcoded)
    elif sum([ele**2 for ele in p]) == 1:
        p_ref = [0, 0, 1]
        generator = Dic4_generator[irrep_name]
        hardcode_irrep = Dic4_irreps[irrep_name]
    elif sum([ele**2 for ele in p]) == 2:
        p_ref = [0, 1, 1]
        generator = Dic2_generator[irrep_name]
        hardcode_irrep = Dic2_irreps[irrep_name]
    elif sum([ele**2 for ele in p]) == 3:
        p_ref = [1, 1, 1]
        generator = Dic3_generator[irrep_name]
        hardcode_irrep = Dic3_irreps[irrep_name]
    elif sum([ele**2 for ele in p]) == 5:
        p_ref = [0, 1, 2]
        generator = C4_generator2[irrep_name]
        is_hardcoded = False
    elif sum([ele**2 for ele in p]) == 6:
        p_ref = [2, 1, 1]
        generator = C4_generator1[irrep_name]
        is_hardcoded = False
    else:
        raise NotImplementedError(f"p^2={p} not implemented")
    if is_hardcoded:
        little_group_pref = hardcode_irrep
    else:
        little_group_pref = littleGroup(p_ref, elem=False)
        nkey = len(little_group_pref.keys()) - len(generator.keys())
        for key in generator.keys():
            generator[key] = Matrix(generator[key])
            little_group_pref[key] = generator[key]
        while nkey > 0:
            for key1 in little_group_pref.keys():
                if little_group_pref[key1] is not None:
                    for key2 in generator.keys():
                        key_result = OhD_mul(key1, key2)
                        if little_group_pref[key_result] is None:
                            little_group_pref[key_result] = (little_group_pref[key1] @ generator[key2]).applyfunc(
                                sp.simplify
                            )
                            nkey -= 1
    if p_ref_irrep:
        little_group_p = little_group_pref
    else:
        little_group_p = littleGroup(p, elem=False)
        for key in little_group_p.keys():
            wignerRotated_key = wignerRotate(p, key)
            little_group_p[key] = little_group_pref[wignerRotated_key]
    return little_group_p


def gen_connection(momentum, irrep_name):
    """
    Generate the connection between a projected vector and a set of basis.
    计算标准基向量在给定基底上的投影系数。

    Args:
        momentum: The momentum vector 动量向量
        irrep_name: The name of irreducible representation 不可约表示名称

    Returns:
        与irrep_row_connection_dict格式兼容的结果，即列表的列表，
        每个内部列表包含多个(系数,群元素)元组，表示每个标准基向量的投影结果
    """
    p_s = momentunSymplify(momentum)
    momentum_str = ",".join([str(element) for element in p_s])
    print(f"简化动量: {momentum_str}")

    # Get reference rotation dictionary for this momentum
    if momentum_str not in refRotateDict:
        raise NotImplementedError(f"动量 {momentum_str} 在refRotateDict中未定义")
    refDict = refRotateDict[momentum_str]  # 这是一个子字典，键是目标动量，值是旋转操作
    print(f"参考旋转字典: {refDict}")

    # Generate little group irrep
    little_group_irrep = genLittleGroupIrrep(p_s, irrep_name)
    print(f"小群不可约表示: 包含 {len(little_group_irrep.keys())} 个元素")

    # Get basis vectors from the first column of each matrix in little_group_irrep
    basis = []
    basis_labels = []  # 存储基底的标签
    for key in little_group_irrep.keys():
        basis.append(little_group_irrep[key][:, 0])
        basis_labels.append(key)  # 使用群元素作为基底标签

    # Import necessary libraries for linear algebra operations
    import sympy as sp
    from sympy import Matrix, S

    # Convert basis to a list of column vectors (sympy matrices)
    basis_vectors = [sp.Matrix(b) for b in basis]

    # 显示原始基底向量
    print("\n原始基底向量:")
    for i, (label, vec) in enumerate(zip(basis_labels, basis_vectors)):
        print(f"基底 {i} ({label}):\n{vec}")

    # 逐个添加向量检查线性独立性，而不是使用RREF
    independent_basis = []
    independent_labels = []

    for i, (vector, label) in enumerate(zip(basis_vectors, basis_labels)):
        # 检查当前向量是否与已选向量线性独立
        if independent_basis:
            # 构建包含所有已选向量和当前向量的矩阵
            test_matrix = independent_basis + [vector]
            combined = sp.Matrix.hstack(*test_matrix)

            # 检查秩是否增加（表示线性独立）
            if combined.rank() > len(independent_basis):
                independent_basis.append(vector)
                independent_labels.append(label)
                print(f"添加基底 {i} ({label}) - 线性独立")
            else:
                print(f"跳过基底 {i} ({label}) - 线性相关")
        else:
            # 第一个非零向量总是独立的
            if vector.norm() > 1e-10:
                independent_basis.append(vector)
                independent_labels.append(label)
                print(f"添加基底 {i} ({label}) - 第一个独立向量")

    print(f"\n找到 {len(independent_basis)} 个线性独立基底向量，原始基底共有 {len(basis_vectors)} 个")
    print("线性独立基底对应的群元素:", ", ".join(independent_labels))

    # 计算度量矩阵(Gram矩阵) - 对非正交基底的处理
    gram_matrix = sp.zeros(len(independent_basis))
    for i in range(len(independent_basis)):
        for j in range(len(independent_basis)):
            # 计算内积 <v_i, v_j>
            gram_matrix[i, j] = independent_basis[i].transpose() * independent_basis[j]

    print("\n度量矩阵(Gram矩阵):")
    print(gram_matrix)

    # 计算度量矩阵的逆
    try:
        gram_inverse = gram_matrix.inv()
        print("\n度量矩阵的逆:")
        print(gram_inverse)
    except Exception as e:
        print(f"计算度量矩阵的逆时出错: {e}")
        # 如果无法求逆，可能是基底线性相关或数值问题
        # 尝试使用伪逆或奇异值分解等方法
        print("尝试使用伪逆...")
        gram_inverse = gram_matrix.pinv()
        print("\n度量矩阵的伪逆:")
        print(gram_inverse)

    # Function to project a vector onto the basis
    def project_vector(vector):
        """
        将向量投影到线性独立基底上 - 使用非正交基底的投影方法

        Args:
            vector: 要投影的向量

        Returns:
            投影系数的字典，键为基底标签，值为系数
        """
        # If the basis is empty, return empty dictionary
        if not independent_basis:
            return {}

        # 计算向量与每个基底向量的内积
        b_vector = sp.zeros(len(independent_basis), 1)
        for i in range(len(independent_basis)):
            b_vector[i] = independent_basis[i].transpose() * vector

        # 使用度量矩阵的逆计算投影系数: c = G^(-1) * b
        try:
            coeffs = gram_inverse * b_vector
            # Return the coefficients as a dictionary
            return {independent_labels[i]: coeffs[i] for i in range(len(independent_basis))}
        except Exception as e:
            print(f"计算投影系数时出错: {e}")
            return {}

    # Project each standard basis vector
    connection_results = []  # 使用与irrep_row_connection_dict兼容的格式
    dim = basis_vectors[0].shape[0] if basis_vectors else 0

    print(f"\n计算 {dim} 个标准基向量在线性独立基底上的投影系数:")

    for j in range(dim):
        # Create a standard basis vector with 1 at position j
        projected_vector = sp.zeros(dim, 1)
        projected_vector[j] = 1

        # Project onto the independent basis
        coeffs = project_vector(projected_vector)

        # 转换为(系数,群元素)元组的列表格式
        connection_row = []
        for basis_label, coeff in coeffs.items():
            # 在检查是否为0之前先化简系数
            simplified_coeff = sp.simplify(coeff, rational=True)
            if simplified_coeff != 0:  # 只保留非零系数
                connection_row.append((simplified_coeff, basis_label))

        connection_results.append(connection_row)

        print(f"\n标准基向量 e_{j} 的投影系数:")
        for coeff, basis_label in connection_row:
            print(f"  {basis_label}: {coeff}")

        # 验证投影结果
        if coeffs:
            projection = sp.zeros(dim, 1)
            for basis_label, coeff in coeffs.items():
                idx = independent_labels.index(basis_label)
                projection += coeff * independent_basis[idx]

            error = (projection - projected_vector).norm()
            error = sp.simplify(error)
            if error != 0:
                # 尝试进一步化简
                error = sp.simplify(error, rational=True)
                if error != 0:
                    # 如果仍然不为0，尝试数值评估
                    try:
                        error_eval = error.evalf()
                        if abs(error_eval) > 1e-10:
                            raise ValueError(f"标准基向量 e_{j} 的投影误差不为0: {error} (数值评估: {error_eval})")
                    except:
                        raise ValueError(f"标准基向量 e_{j} 的投影误差不为0: {error}")

    # 最终返回格式
    print("\n最终输出格式 (与irrep_row_connection_dict兼容):")
    print(f"{irrep_name}: {connection_results}")

    return connection_results


def reductionToLittleGroup(momentum, OhD_irep_name, parity, little_group_irrep_name, is_hardcoded=True):
    """
    Reduce the OhD irrep to the little group irrep.
    """
    p = momentunSymplify(momentum)
    pstr = ",".join([str(element) for element in p])
    for key in refRotateDict.keys():
        if pstr in refRotateDict[key].keys():
            pref_str = key
            pref = [int(element) for element in list(key.split(","))]
            ref_rotate_p = refRotateDict[key][pstr]
            break
    matrix_OhD_irrep = genLittleGroupIrrep([0, 0, 0], OhD_irep_name, parity)
    if is_hardcoded:
        irrep_parity = f"{OhD_irep_name}{'g' if parity == 1 else 'u'}"
        try:
            reduction_matrixs = copy.deepcopy(
                little_group_reduction_map[pref_str][irrep_parity][little_group_irrep_name]
            )
        except:
            print(f"reduction_matrixs[{pref_str}][{irrep_parity}][{little_group_irrep_name}] not found")
            return None
    else:
        matrix_little_group_irrep = genLittleGroupIrrep(pref, little_group_irrep_name)
        ndim_little_group = matrix_little_group_irrep["iden"].shape[0]
        ndim_OhD = matrix_OhD_irrep["iden"].shape[0]
        # 使用sympy的Matrix而不是numpy数组
        reduction_matrix_colinear = Matrix.zeros(ndim_OhD, ndim_OhD)
        for key in matrix_little_group_irrep.keys():
            for i in range(ndim_OhD):
                for j in range(ndim_OhD):
                    reduction_matrix_colinear[i, j] += (
                        matrix_OhD_irrep[key][j, i] * matrix_little_group_irrep[key][0, 0]
                    )
            for i in range(ndim_OhD):
                for j in range(ndim_OhD):
                    # 使用rational=True来保持有理数形式
                    reduction_matrix_colinear[i, j] = sp.simplify(reduction_matrix_colinear[i, j], rational=True)
        if reduction_matrix_colinear.is_zero_matrix:
            return None
        M = reduction_matrix_colinear
        rref_matrix, pivots = M.rref()
        vlist = [rref_matrix.row(i) for i in range(len(pivots))]
        basis = GramSchmidt(vlist, True)
        reduction_matrixs = []
        for i in range(len(basis)):
            reduction_matrix = []
            for j in range(ndim_little_group):
                connection = irrep_row_connection_dict[pref_str][little_group_irrep_name][j]
                result = None
                for coeff, rotation in connection:
                    if result is None:
                        result = coeff * basis[i] @ matrix_OhD_irrep[rotation].T
                    else:
                        result += coeff * basis[i] @ matrix_OhD_irrep[rotation].T
                # 对每个元素进行simplify，使用rational=True
                reduction_matrix.append([sp.simplify(ele, rational=True) for ele in result.tolist()[0]])
            reduction_matrixs.append(reduction_matrix)
    for i in range(len(reduction_matrixs)):
        for j in range(len(reduction_matrixs[i])):
            # 对最终结果进行simplify，使用rational=True
            reduction_matrixs[i][j] = (
                Matrix(matrix_OhD_irrep[ref_rotate_p]) @ Matrix(reduction_matrixs[i][j])
            ).T.tolist()[0]
            reduction_matrixs[i][j] = [sp.simplify(ele, rational=True) for ele in reduction_matrixs[i][j]]
    return reduction_matrixs
