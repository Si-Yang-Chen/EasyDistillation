from typing import Literal, Optional, Union
from sympy import sqrt
import numpy as np
from sympy import S, Add, Mul, Symbol, Matrix
import sympy
from lattice.symmetry.gen_hardcoded_rep import genLittleGroupIrrep
from lattice.symmetry.hardcoded_rep import OD_irreps, irrep_row_connection_dict, group_element
from lattice.symmetry.sympy_utils import find_linear_independent_exprs

# 定义规范变换字典


def genGaugeTransformDict():
    gauge_transform_dict = {}
    gauge_links = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    group_matrix = genLittleGroupIrrep([0, 0, 0], "T_1", -1)
    for ele in group_element:
        matrix = group_matrix[ele]
        gauge_transform_dict[ele] = [None] * 6
        for i in range(6):
            gauge_link = gauge_links[i]
            final_link = matrix @ Matrix(gauge_link)
            for j in range(6):
                if final_link == Matrix(gauge_links[j]):
                    gauge_transform_dict[ele][i] = j
                    break
            if gauge_transform_dict[ele][i] is None:
                raise ValueError(f"{final_link} not found in {gauge_links}")
    return gauge_transform_dict


gauge_transform_dict = {
    "iden": [0, 1, 2, 3, 4, 5],
    "c4x": [0, 2, 4, 3, 5, 1],
    "c2x": [0, 4, 5, 3, 1, 2],
    "c4x^-1": [0, 5, 1, 3, 2, 4],
    "c4y": [5, 1, 0, 2, 4, 3],
    "c2y": [3, 1, 5, 0, 4, 2],
    "c4y^-1": [2, 1, 3, 5, 4, 0],
    "c4z": [1, 3, 2, 4, 0, 5],
    "c2z": [3, 4, 2, 0, 1, 5],
    "c4z^-1": [4, 0, 2, 1, 3, 5],
    "c3delta": [1, 2, 0, 4, 5, 3],
    "c3delta^-1": [2, 0, 1, 5, 3, 4],
    "c3gamma": [4, 2, 3, 1, 5, 0],
    "c3gamma^-1": [5, 3, 1, 2, 0, 4],
    "c3beta": [4, 5, 0, 1, 2, 3],
    "c3beta^-1": [2, 3, 4, 5, 0, 1],
    "c3alpha": [1, 5, 3, 4, 2, 0],
    "c3alpha^1": [5, 0, 4, 2, 3, 1],
    "c2e": [3, 2, 1, 0, 5, 4],
    "c2f": [3, 5, 4, 0, 2, 1],
    "c2c": [2, 4, 0, 5, 1, 3],
    "c2d": [5, 4, 3, 2, 1, 0],
    "c2a": [1, 0, 5, 4, 3, 2],
    "c2b": [4, 3, 5, 1, 0, 2],
    "riden": [0, 1, 2, 3, 4, 5],
    "rc4x": [0, 2, 4, 3, 5, 1],
    "rc2x": [0, 4, 5, 3, 1, 2],
    "rc4x^-1": [0, 5, 1, 3, 2, 4],
    "rc4y": [5, 1, 0, 2, 4, 3],
    "rc2y": [3, 1, 5, 0, 4, 2],
    "rc4y^-1": [2, 1, 3, 5, 4, 0],
    "rc4z": [1, 3, 2, 4, 0, 5],
    "rc2z": [3, 4, 2, 0, 1, 5],
    "rc4z^-1": [4, 0, 2, 1, 3, 5],
    "rc3delta": [1, 2, 0, 4, 5, 3],
    "rc3delta^-1": [2, 0, 1, 5, 3, 4],
    "rc3gamma": [4, 2, 3, 1, 5, 0],
    "rc3gamma^-1": [5, 3, 1, 2, 0, 4],
    "rc3beta": [4, 5, 0, 1, 2, 3],
    "rc3beta^-1": [2, 3, 4, 5, 0, 1],
    "rc3alpha": [1, 5, 3, 4, 2, 0],
    "rc3alpha^1": [5, 0, 4, 2, 3, 1],
    "rc2e": [3, 2, 1, 0, 5, 4],
    "rc2f": [3, 5, 4, 0, 2, 1],
    "rc2c": [2, 4, 0, 5, 1, 3],
    "rc2d": [5, 4, 3, 2, 1, 0],
    "rc2a": [1, 0, 5, 4, 3, 2],
    "rc2b": [4, 3, 5, 1, 0, 2],
    "inviden": [3, 4, 5, 0, 1, 2],
    "invc4x": [3, 5, 1, 0, 2, 4],
    "invc2x": [3, 1, 2, 0, 4, 5],
    "invc4x^-1": [3, 2, 4, 0, 5, 1],
    "invc4y": [2, 4, 3, 5, 1, 0],
    "invc2y": [0, 4, 2, 3, 1, 5],
    "invc4y^-1": [5, 4, 0, 2, 1, 3],
    "invc4z": [4, 0, 5, 1, 3, 2],
    "invc2z": [0, 1, 5, 3, 4, 2],
    "invc4z^-1": [1, 3, 5, 4, 0, 2],
    "invc3delta": [4, 5, 3, 1, 2, 0],
    "invc3delta^-1": [5, 3, 4, 2, 0, 1],
    "invc3gamma": [1, 5, 0, 4, 2, 3],
    "invc3gamma^-1": [2, 0, 4, 5, 3, 1],
    "invc3beta": [1, 2, 3, 4, 5, 0],
    "invc3beta^-1": [5, 0, 1, 2, 3, 4],
    "invc3alpha": [4, 2, 0, 1, 5, 3],
    "invc3alpha^1": [2, 3, 1, 5, 0, 4],
    "invc2e": [0, 5, 4, 3, 2, 1],
    "invc2f": [0, 2, 1, 3, 5, 4],
    "invc2c": [5, 1, 3, 2, 4, 0],
    "invc2d": [2, 1, 0, 5, 4, 3],
    "invc2a": [4, 3, 2, 1, 0, 5],
    "invc2b": [1, 0, 2, 4, 3, 5],
    "invriden": [3, 4, 5, 0, 1, 2],
    "invrc4x": [3, 5, 1, 0, 2, 4],
    "invrc2x": [3, 1, 2, 0, 4, 5],
    "invrc4x^-1": [3, 2, 4, 0, 5, 1],
    "invrc4y": [2, 4, 3, 5, 1, 0],
    "invrc2y": [0, 4, 2, 3, 1, 5],
    "invrc4y^-1": [5, 4, 0, 2, 1, 3],
    "invrc4z": [4, 0, 5, 1, 3, 2],
    "invrc2z": [0, 1, 5, 3, 4, 2],
    "invrc4z^-1": [1, 3, 5, 4, 0, 2],
    "invrc3delta": [4, 5, 3, 1, 2, 0],
    "invrc3delta^-1": [5, 3, 4, 2, 0, 1],
    "invrc3gamma": [1, 5, 0, 4, 2, 3],
    "invrc3gamma^-1": [2, 0, 4, 5, 3, 1],
    "invrc3beta": [1, 2, 3, 4, 5, 0],
    "invrc3beta^-1": [5, 0, 1, 2, 3, 4],
    "invrc3alpha": [4, 2, 0, 1, 5, 3],
    "invrc3alpha^1": [2, 3, 1, 5, 0, 4],
    "invrc2e": [0, 5, 4, 3, 2, 1],
    "invrc2f": [0, 2, 1, 3, 5, 4],
    "invrc2c": [5, 1, 3, 2, 4, 0],
    "invrc2d": [2, 1, 0, 5, 4, 3],
    "invrc2a": [4, 3, 2, 1, 0, 5],
    "invrc2b": [1, 0, 2, 4, 3, 5],
}


class GaugeLink(Symbol):
    # 定义允许的下一个元素（差值不为3的元素）
    _VALID_NEXT = {
        0: [0, 1, 2, 4, 5],  # 0后面可以跟0,1,2,4,5（不能跟3）
        1: [0, 1, 2, 3, 5],  # 1后面可以跟0,1,2,3,5（不能跟4）
        2: [0, 1, 2, 3, 4],  # 2后面可以跟0,1,2,3,4（不能跟5）
        3: [1, 2, 3, 4, 5],  # 3后面可以跟0,1,2,3,4（不能跟6/0）
        4: [0, 2, 3, 4, 5],  # 4后面可以跟0,1,2,4,5（不能跟7/1）
        5: [0, 1, 3, 4, 5],  # 5后面可以跟0,1,3,4,5（不能跟8/2）
    }

    # 每个数字的有效后继数量
    _VALID_COUNT = {i: len(nexts) for i, nexts in _VALID_NEXT.items()}

    @staticmethod
    def _idx_to_gauge_list(idx):
        """将idx转换为gauge_list，基于新的编码方案"""
        lenth = 0
        for i in GaugeLink.nmax_generator(100):
            if i > idx:
                break
            else:
                lenth += 1
        if lenth == 0:
            return []
        idx = idx - list(GaugeLink.nmax_generator(lenth - 1))[-1]
        position_list = []
        for i in range(lenth):
            position = idx % 5
            position_list.append(position)
            idx = idx // 5
        position_list[-1] += idx * 5
        gauge_list = [position_list[-1]]
        for i in range(lenth - 1):
            next_gauge = GaugeLink._VALID_NEXT[gauge_list[-1]][position_list[lenth - i - 2]]
            gauge_list.append(next_gauge)
        return gauge_list

    @staticmethod
    def _gauge_list_to_name(gauge_list):
        """将gauge_list转换为name字符串"""
        return "U" + "".join(str(i) for i in gauge_list)

    @staticmethod
    def _is_valid_gauge_list(gauge_list):
        """检查gauge_list是否符合规则：相邻两项的差值绝对值不能为3"""
        if len(gauge_list) < 2:
            return True  # 单个元素或空列表总是有效的

        for i in range(len(gauge_list) - 1):
            if abs(gauge_list[i] - gauge_list[i + 1]) == 3:
                return False
        return True

    @staticmethod
    def nmax_generator(n):
        """计算n个元素的gauge_list的最大idx"""

        # 使用generator实现递归计算
        def _nmax_generator(max_n):
            result = 1  # n=0时的初始值
            yield result

            # 从n=1开始递增计算
            for i in range(1, max_n + 1):
                result = result + 6 * (5 ** (i - 1))
                yield result

        # 使用generator计算到第n个值
        return _nmax_generator(n)

    def __new__(cls, name):
        if isinstance(name, str):
            # 创建临时对象检查gauge_list是否有效
            temp_obj = super().__new__(cls, name)
            temp_obj.name = name
            temp_obj._gauge_list = None
            if not cls._is_valid_gauge_list(temp_obj.gauge_list):
                raise ValueError("Invalid gauge_list: adjacent elements cannot have difference of 3")
            obj = super().__new__(cls, name)
        elif isinstance(name, list):
            if not cls._is_valid_gauge_list(name):
                raise ValueError("Invalid gauge_list: adjacent elements cannot have difference of 3")
            obj = super().__new__(cls, cls._gauge_list_to_name(name))
        elif isinstance(name, (int,)):
            gauge_list = cls._idx_to_gauge_list(name)
            if not cls._is_valid_gauge_list(gauge_list):
                raise ValueError("Invalid gauge_list: adjacent elements cannot have difference of 3")
            obj = super().__new__(cls, cls._gauge_list_to_name(gauge_list))
        return obj

    def __init__(self, name):
        if isinstance(name, int):
            gauge_list = self._idx_to_gauge_list(name)
            self.name = self._gauge_list_to_name(gauge_list)
            self._gauge_list = gauge_list
            self._idx = name
        else:
            if isinstance(name, str):
                self.name = name
                self._gauge_list = None
            else:  # list
                self.name = self._gauge_list_to_name(name)
                self._gauge_list = list(name)
            self._idx = None
        self._displacement = None

    @property
    def gauge_list(self):
        if self._gauge_list is None:
            # 解析纯数字字符串，每个字符是一个gauge值
            self._gauge_list = [int(c) for c in self.name if c.isdigit()]
        return self._gauge_list

    @property
    def idx(self):
        if self._idx is None:
            length = len(self.gauge_list)
            if length == 0:
                self._idx = 0
            else:
                self._idx = self.gauge_list[0]
                for i in range(1, length):
                    prev_digit = self.gauge_list[i - 1]
                    curr_digit = self.gauge_list[i]
                    # 获取前一个数字的有效后继列表
                    valid_next = self._VALID_NEXT[prev_digit]
                    # 计算当前数字在有效后继中的位置
                    try:
                        position = valid_next.index(curr_digit)
                    except ValueError:
                        raise ValueError(f"Invalid gauge sequence: {curr_digit} cannot follow {prev_digit}")
                    self._idx = self._idx * 5 + position
                self._idx += list(GaugeLink.nmax_generator(length - 1))[-1]
        return self._idx

    @property
    def displacement(self):
        if self._displacement is None:
            displacement = [0, 0, 0]
            for i in range(len(self.gauge_list)):
                if self.gauge_list[i] < 3:
                    displacement[self.gauge_list[i]] += 1
                else:
                    displacement[self.gauge_list[i] - 3] -= 1
            self._displacement = displacement
        return self._displacement

    def transform(self, element: str):
        transform_list = gauge_transform_dict[element]
        new_gauge_list = [transform_list[i] for i in self.gauge_list]

        # 检查转换后的gauge_list是否有效
        if not self._is_valid_gauge_list(new_gauge_list):
            raise ValueError(f"Transform with {element} results in invalid gauge_list")

        return GaugeLink(new_gauge_list)

    def conjugate(self):
        new_gauge_list = self.gauge_list[-1::-1]
        for i in range(len(new_gauge_list)):
            new_gauge_list[i] = (new_gauge_list[i] + 3) % 6

        # 检查转换后的gauge_list是否有效
        if not self._is_valid_gauge_list(new_gauge_list):
            raise ValueError("Conjugate operation results in invalid gauge_list")

        return GaugeLink(new_gauge_list)

    def inv(self):
        new_gauge_list = [(ele + 3) % 6 for ele in self.gauge_list]
        return GaugeLink(new_gauge_list)


def gen_insertion_dict(max_lenth, insertion_form=True, max_multiplets_per_irrep=None):
    """生成gauge_irrep

    max_multiplets_per_irrep: if set to an int, each irrep key keeps at most that many
    multiplets (order follows the generation loop). None means no limit.
    """
    from lattice.symmetry.gen_hardcoded_rep import momentunSymplify, littleGroup

    max_idx = list(GaugeLink.nmax_generator(max_lenth))[-1]
    irrep_dict = {}
    representation_list = []
    irrep_name_list = ["A_1", "A_2", "E", "T_1", "T_2"]
    for irrep_name in irrep_name_list:
        for parity in ["u", "g"]:
            for charge_conjugate in ["+", "-"]:
                if irrep_name == "E":
                    irrep_dict[f"{irrep_name}{parity}{charge_conjugate}"] = []
                elif irrep_name == "T_1" or irrep_name == "T_2":
                    irrep_dict[f"{irrep_name}{parity}{charge_conjugate}"] = []
                else:
                    irrep_dict[f"{irrep_name}{parity}{charge_conjugate}"] = []
    undistribute_irrep_list = list(range(max_idx))
    while len(undistribute_irrep_list) > 0:
        representation_list.append([])
        ele = undistribute_irrep_list.pop(0)
        representation_list[-1].append(ele)
        original_gauge = GaugeLink(ele)
        for key in gauge_transform_dict.keys():
            new_gauge = original_gauge.transform(key)
            new_ele = new_gauge.idx
            if new_ele in undistribute_irrep_list:
                undistribute_irrep_list.remove(new_ele)
                representation_list[-1].append(new_ele)
        for ele in representation_list[-1]:
            original_gauge = GaugeLink(ele)
            new_gauge = original_gauge.conjugate()
            new_ele = new_gauge.idx
            if new_ele in undistribute_irrep_list:
                undistribute_irrep_list.remove(new_ele)
                representation_list[-1].append(new_ele)
    for i in range(len(representation_list)):
        representation = representation_list[i]
        ndim_representation = len(representation)
        counter = 0
        for irrep_name in irrep_name_list:
            irrep = OD_irreps[irrep_name]
            ndim_irrep = irrep["iden"].shape[0]
            final_result_dict = {}
            for i in range(ndim_representation):
                result = S(0)
                for key in irrep.keys():
                    result += irrep[key][0, 0].conjugate() * GaugeLink(representation[i]).transform(key)
                if result != S(0):
                    if result != S(0):
                        inv_result = S(0)
                        terms = Add.make_args(result)
                        for term in terms:
                            if isinstance(term, GaugeLink):
                                inv_result += term.inv()
                            elif isinstance(term, Mul):
                                # 处理乘积项，如系数乘以GaugeLink
                                coeff = 1
                                gauge_link = None
                                factors = Mul.make_args(term)
                                for factor in factors:
                                    if isinstance(factor, GaugeLink):
                                        gauge_link = factor
                                    else:
                                        coeff *= factor
                                if gauge_link:
                                    inv_result += coeff * gauge_link.inv()
                        for parity in ["u", "g"]:
                            if parity == "u":
                                projected_result = result - inv_result
                            else:
                                projected_result = result + inv_result
                            if projected_result != S(0):
                                for charge_conjugate in ["+", "-"]:
                                    if f"{irrep_name}{parity}{charge_conjugate}" not in final_result_dict.keys():
                                        final_result_dict[f"{irrep_name}{parity}{charge_conjugate}"] = []
                                    conjugate_result = S(0)
                                    terms = Add.make_args(projected_result)
                                    for term in terms:
                                        if isinstance(term, GaugeLink):
                                            conjugate_result += term.conjugate()
                                        elif isinstance(term, Mul):
                                            coeff = 1
                                            gauge_link = None
                                            factors = Mul.make_args(term)
                                            for factor in factors:
                                                if isinstance(factor, GaugeLink):
                                                    gauge_link = factor
                                                else:
                                                    coeff *= factor
                                            if gauge_link:
                                                conjugate_result += coeff * gauge_link.conjugate()
                                    if charge_conjugate == "+":
                                        final_result = projected_result + conjugate_result
                                    else:
                                        final_result = projected_result - conjugate_result
                                    if final_result != S(0):
                                        final_result_dict[f"{irrep_name}{parity}{charge_conjugate}"].append(
                                            final_result
                                        )
            for key in final_result_dict.keys():
                final_result_dict[key] = find_linear_independent_exprs(final_result_dict[key])
            for key in final_result_dict.keys():
                for i in range(len(final_result_dict[key])):
                    if max_multiplets_per_irrep is not None and len(irrep_dict[key]) >= max_multiplets_per_irrep:
                        break
                    final_result = final_result_dict[key][i]
                    irrep_row_connection = irrep_row_connection_dict["0,0,0"][key[:-2]]
                    result_irrep = []
                    for i in range(ndim_irrep):
                        result_row = S(0)
                        terms = Add.make_args(final_result)
                        for term in terms:
                            if isinstance(term, GaugeLink):
                                for j in range(len(irrep_row_connection[i])):
                                    result_row += (
                                        term.transform(irrep_row_connection[i][j][1]) * irrep_row_connection[i][j][0]
                                    )
                            elif isinstance(term, Mul):
                                coeff = 1
                                gauge_link = None
                                factors = Mul.make_args(term)
                                for factor in factors:
                                    if isinstance(factor, GaugeLink):
                                        gauge_link = factor
                                    else:
                                        coeff *= factor
                                if gauge_link:
                                    for j in range(len(irrep_row_connection[i])):
                                        result_row += (
                                            coeff
                                            * gauge_link.transform(irrep_row_connection[i][j][1])
                                            * irrep_row_connection[i][j][0]
                                        )
                        if insertion_form:
                            insertion_row = []
                            for term in Add.make_args(result_row):
                                if isinstance(term, GaugeLink):
                                    insertion_row.append([1, term.idx])
                                elif isinstance(term, Mul):
                                    coeff = 1
                                    gauge_link = None
                                    factors = Mul.make_args(term)
                                    for factor in factors:
                                        if isinstance(factor, GaugeLink):
                                            gauge_link = factor
                                        else:
                                            coeff *= factor
                                    insertion_row.append([coeff, gauge_link.idx])
                            result_irrep.append(insertion_row)
                        else:
                            result_irrep.append(result_row)
                        counter += 1
                    irrep_dict[key].append(result_irrep)
    return irrep_dict


def gen_gauge_list(insertion_dict):
    from lattice.symmetry.hardcoded_rep import little_group_irreps
    from lattice.symmetry.gen_hardcoded_rep import momentunSymplify

    gauge_list = []
    new_insertion_dict = {}
    information_dict = {}
    for key in insertion_dict.keys():
        new_insertion_dict[key] = []
        information_dict[key] = []
        for i in range(len(insertion_dict[key])):
            new_insertion_dict[key].append([])
            # projection_list = []
            for j in range(len(insertion_dict[key][i])):
                gauge_list.append(insertion_dict[key][i][j])
                new_insertion_dict[key][i].append([[1, len(gauge_list) - 1]])
                if len(information_dict[key]) < len(insertion_dict[key]):
                    information = [None, None, None]
                    displacement_distribution = {}
                    for k in range(len(insertion_dict[key][i][j])):
                        displacement = GaugeLink(insertion_dict[key][i][j][k][1]).displacement
                        if information[0] is None:
                            information[0] = sum(ele**2 for ele in displacement)
                        displacement_str = ",".join(str(i) for i in displacement)
                        displacement_str_simplified = ",".join(str(i) for i in momentunSymplify(displacement))
                        if displacement_str_simplified in little_group_irreps.keys():
                            if displacement_str not in displacement_distribution.keys():
                                displacement_distribution[displacement_str] = S(0)
                            displacement_distribution[displacement_str] += (
                                GaugeLink(insertion_dict[key][i][j][k][1]) * insertion_dict[key][i][j][k][0]
                            )
                    information[2] = displacement_distribution
                    information_dict[key].append(information)
    return gauge_list, new_insertion_dict, information_dict


def gauge_group(name):
    return name[:-2]


def gauge_parity(name):
    assert name[-2] in ["u", "g"]
    if name[-2] == "g":
        return 1
    elif name[-2] == "u":
        return -1


def gauge_charge_conjugate(name):
    assert name[-1] in ["-", "+"]
    if name[-1] == "+":
        return 1
    elif name[-1] == "-":
        return -1


def gauge_hermiticity(name):
    return gauge_charge_conjugate(name)


if __name__ == "__main__":
    insertion_dict = gen_insertion_dict(3)
    print(insertion_dict)
    # gauge_list, new_insertion_dict, information_dict = gen_gauge_list(insertion_dict)
    # print(gauge_list)
    # print(new_insertion_dict)
    # print(information_dict)
