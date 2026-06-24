from typing import Dict, List

from .gamma import (
    GammaName,
    scheme as gamma_scheme,
    group as gamma_gourp,
    parity as gamma_parity,
    charge_conjugation as gamma_charge_conjugation,
    hermiticity as gamma_hermiticity,
)
from .derivative import (
    DerivativeName,
    scheme as derivative_scheme,
    group as derivative_gourp,
    parity as derivative_parity,
    charge_conjugation as derivative_charge_conjugation,
    hermiticity as derivative_hermiticity,
)

from ..symmetry import *
from .gauge_link import gauge_transform_dict, GaugeLink,gauge_group,gauge_parity,gauge_charge_conjugate,gauge_hermiticity
from .gamma import gamma_transform


class ProjectionName:
    A1 = "A_1"
    A2 = "A_2"
    E = "E"
    T1 = "T_1"
    T2 = "T_2"


class Row(list):
    def simplify(self):
        lenth = len(self)
        tuples_list = []
        for i in range(0, lenth, 2):
            tuples_list.extend(
                [(self[i], self[i + 1][j][1], self[i + 1][j][0]) for j in range(len(self[i + 1]))]
            )
        merged_dict = {}
        # Merge tuples by summing values for the same key
        for tup in tuples_list:
            key = (tup[0], tup[1])
            value = tup[2]
            if key in merged_dict:
                merged_dict[key] += value
            else:
                merged_dict[key] = value
        # Convert the dictionary back to a list of tuples
        merged_list = [(k[0], k[1], v) for k, v in merged_dict.items()]
        merged_dict = {}
        # Group tuples by the first element and filter out zero values
        for tup in merged_list:
            key = tup[0]
            if tup[2] == 0:
                continue
            elif key in merged_dict:
                merged_dict[key].append([tup[2], tup[1]])
            else:
                merged_dict[key] = [[tup[2], tup[1]]]
        merged_list = []
        for k, v in merged_dict.items():
            merged_list.extend([k, v])
        self[:] = merged_list[:]
        return self

    def __add__(self, other):
        # Perform element-wise addition and merge tuples
        result = self.__class__(super().__add__(other))
        result.simplify()
        return result

    def __mul__(self, scalar):
        # Perform scalar multiplication on each value
        if not isinstance(scalar, (int, float, sp.Expr)):
            return NotImplemented
        result = []
        for i in range(0, len(self), 2):
            key = self[i]
            values = [[v[0] * scalar, v[1]] for v in self[i + 1]]
            result.extend([key, values])
        return Row(result)

    def __rmul__(self, scalar):
        # Ensure scalar multiplication works on the right side
        return self.__mul__(scalar)

    def __neg__(self):
        # Negate each value in the list
        return self * -1

    def __sub__(self, other):
        # Perform element-wise subtraction by negating 'other' and adding
        return self + (-other)

    def __iadd__(self, other):
        # Perform in-place addition
        self[:] = (self + other)[:]
        return self

    def __isub__(self, other):
        # Perform in-place subtraction
        self[:] = (self - other)[:]
        return self

    def __imul__(self, scalar):
        # Perform in-place scalar multiplication
        self[:] = (self * scalar)[:]
        return self


class GaugeRepRow(Row):
    def __init__(self, *args):
        super().__init__(*args)

    def transform(self, group_element):
        lenth = len(self)
        assert lenth % 2 == 0, "GaugeRepRow must have even length"
        result = copy.deepcopy(self)
        for i in range(lenth // 2):
            coeff = 1
            gamma_idx = gamma_transform(group_element, self[i * 2])
            if gamma_idx >= 16:
                coeff = -1
                gamma_idx -= 16
            for j in range(len(self[i * 2 + 1])):
                gauge_idx = self[i * 2 + 1][j][1]
                final_gauge_idx = GaugeLink(gauge_idx).transform(group_element).idx
                result[i * 2 + 1][j][1] = final_gauge_idx
                if coeff == -1:
                    result[i * 2 + 1][j][0] = -result[i * 2 + 1][j][0]
        return result



class InsertionRowMom:
    def __init__(self, row, momentum,profile=None) -> None:
        self.row = row
        self.momentum = momentum
        self.profile = profile

class Operator:
    def __init__(
        self,
        name: str,
        insertion_rows: List[InsertionRowMom],
        coefficients: List[float],
    ) -> None:
        assert len(insertion_rows) == len(
            coefficients
        ), f"Unmatched numbers of insertion rows {len(insertion_rows)} and coefficients {len(coefficients)}"
        parts = []
        for idx in range(len(insertion_rows)):
            row, momentum,profile,coefficient = (
                insertion_rows[idx].row,
                insertion_rows[idx].momentum,
                insertion_rows[idx].profile,
                coefficients[idx],
            )
            for i in range(len(row) // 2):
                parts.append(row[i * 2])
                elemental_part = []
                for derivative_coeff, derivative_idx in row[i * 2 + 1]:
                    if parts[-1] == 5 or parts[-1] == 13:
                        # gamma_3gamma_1 = -gamma(5), gamma_3gamma_1gamma_4 = -gamma(13)
                        derivative_coeff *= -1
                    elemental_part.append([coefficient * derivative_coeff, derivative_idx, momentum, profile])
                parts.append(elemental_part)

        self.name = name
        self.parts = parts

    def __str__(self) -> str:
        ret = ""
        ret += f"============== operator {self.name}, components: ===============\n"
        for irow in range(len(self.parts) // 2):
            ret += f"   gamma idx = {str(self.parts[2*irow])}, \n"
            for iterm in self.parts[2 * irow + 1]:
                coeff, derivative_idx, momentum = iterm
                ret += f"       > coeff = {coeff}, derivative_idx = {derivative_idx}, momentum = {momentum}\n"
        ret += f"================================================================\n"
        return ret

    def set_gamma(self, i_row, gamma_idx):
        """
        Set gamma indix for irow-th InsertionRow.
        """
        self.parts[2 * i_row] = gamma_idx

    def set_derivative(self, i_row, i_term, deriv_idx):
        """
        Set derivative indix for i-th term of  i-th InsertionRow.
        """
        self.parts[2 * i_row + 1][i_term][1] = deriv_idx


class OperatorDisplacement(Operator):
    def __init__(
        self,
        name: str,
        insertion_rows: List[InsertionRowMom],
        coefficients: List[float],
        distances: List[int],
    ) -> None:
        assert len(insertion_rows) == len(distances)
        super().__init__(name, insertion_rows, coefficients)
        for irow in range(len(self.parts) // 2):
            for iterm, term in enumerate(self.parts[2 * irow + 1]):
                coeff, derivative_idx, momentum = term
                assert (
                    derivative_idx == 0
                ), f"displacement operator cannot define at derivative_idx = {derivative_idx}, not 0"
                self.set_derivative(i_row=irow, i_term=iterm, deriv_idx=distances[irow])


class InsertionRow:
    def __init__(self, row, momentum_dict,profile=None) -> None:
        self.row = row
        self.momentum_dict = momentum_dict
        self.profile = profile

    def __call__(self, npx, npy, npz) -> InsertionRowMom:
        return InsertionRowMom(self.row, list(self.momentum_dict.values()).index(f"{npx} {npy} {npz}"),self.profile)

    def __str__(self) -> str:
        from .gamma import output as gamma_str
        from .derivative import output as derivative_str

        ret = ""
        parts = self.row
        for i in range(len(parts) // 2):
            derivative_part = parts[i * 2 + 1]
            derivative_str_part = ""
            for j in range(len(derivative_part)):
                derivative_str_part += f"{derivative_str(derivative_part[j])}"
                if j != len(derivative_part) - 1:
                    derivative_str_part += " + "
            ret += f"{gamma_str(parts[i*2])} * ({derivative_str_part})"
            if i != len(parts) // 2 - 1:
                ret += " + "
        return ret


class Insertion:
    def __init__(
        self,
        gamma: GammaName,
        derivative: DerivativeName,
        projection: ProjectionName,
        momentum_dict: Dict[int, str],
        profile=None,
    ) -> None:
        self.gamma = gamma_scheme(gamma)
        self.derivative = derivative_scheme(derivative)
        self.parity = gamma_parity(gamma) * derivative_parity(derivative)
        self.charge_conjugation = gamma_charge_conjugation(gamma) * derivative_charge_conjugation(derivative)
        self.hermiticity = gamma_hermiticity(gamma) * derivative_hermiticity(derivative)
        self.projection = [gamma_gourp(gamma), derivative_gourp(derivative), projection]
        self.momentum_dict = momentum_dict
        self.rows = []
        self.little_group_irreps_dict = {}
        self.profile = profile
        self.construct()

    def __getitem__(self, idx) -> InsertionRow:
        return InsertionRow(self.rows[idx], self.momentum_dict, self.profile)

    def __str__(self) -> str:
        ret = []
        for i in range(len(self.rows)):
            ret.append(str(self[i]))
        return str(ret)

    def multiply(self, coeff, derivative):
        ret = []
        for i in range(len(derivative)):
            ret.append([coeff * derivative[i][0], *derivative[i][1:]])
        return ret

    def little_group_projection(self, momentum, irrep_name, idx=0):
        if momentum == [0, 0, 0]:
            return self
        reduction_matrix = reductionToLittleGroup(momentum, self.projection[-1], self.parity, irrep_name)[idx]
        ndim_irrep = len(reduction_matrix)
        little_group_rows = []
        for i in range(ndim_irrep):
            row = Row([])
            for j in range(len(reduction_matrix[i])):
                row += reduction_matrix[i][j] * self.rows[j]
            little_group_rows.append(row)
        self.little_group_irreps_dict[str(momentum)] = little_group_rows
        self.rows = little_group_rows
        return self

    def construct(self):
        gamma = self.gamma
        derivative = self.derivative
        left, right, projection = self.projection
        length = {"A_1": 1, "A_2": 1, "E": 2, "T_1": 3, "T_2": 3}
        irrep_T1 = {
            "A_1": ["T_1"],
            "A_2": ["T_2"],
            "E": ["T_1", "T_2"],
            "T_1": ["A_1", "T_1", "T_2", "E"],
            "T_2": ["T_1", "T_2", "E", "A_2"],
        }
        if left == "A_1":
            assert right == projection, f"{left} x {right} has no irrep {projection}"
            for i in range(length[projection]):
                self.rows.append(Row([gamma[0], derivative[i]]))
        elif left == "T_1":
            assert projection in irrep_T1[right], f"{left} x {right} has no irrep {projection}"
            for i in range(length[projection]):
                if right == "A_1":
                    self.rows.append(Row([gamma[i], derivative[0]]))
                elif right == "E":
                    if projection == "T_1":
                        if i == 0:
                            row = Row([gamma[0], derivative[0], gamma[0], self.multiply(-1/sqrt(3), derivative[1])])
                        elif i == 1:
                            row = Row([gamma[1], self.multiply(-1, derivative[0]), gamma[1], self.multiply(-1/sqrt(3), derivative[1])])
                        elif i == 2:
                            row = Row([gamma[2], self.multiply(2/sqrt(3), derivative[0])])
                    elif projection == "T_2":
                        if i == 0:
                            row = Row([gamma[0], derivative[0], gamma[0], self.multiply(sqrt(3), derivative[1])])
                        elif i == 1:
                            row = Row([gamma[1], derivative[0], gamma[1], self.multiply(-sqrt(3), derivative[1])])
                        elif i == 2:
                            row = Row([gamma[2], self.multiply(-2, derivative[0])])
                    row.simplify()
                    self.rows.append(row)
                elif projection in ["A_1", "A_2"]:
                    self.rows.append(
                        Row(
                            [
                                gamma[0],
                                derivative[0],
                                gamma[1],
                                derivative[1],
                                gamma[2],
                                derivative[2],
                            ]
                        )
                    )
                elif projection in ["E"]:
                    if right == 'T_1':
                        if i == 0:
                            self.rows.append(
                                Row(
                                    [
                                        gamma[0],
                                        derivative[0],
                                        gamma[1],
                                        self.multiply(-1, derivative[1]),
                                    ]
                                )
                            )
                        else:
                            self.rows.append(
                                Row(
                                    [
                                        gamma[0],
                                        self.multiply(-1/sqrt(3), derivative[0]),
                                        gamma[1],
                                        self.multiply(-1/sqrt(3), derivative[1]),
                                        gamma[2],
                                        self.multiply(2/sqrt(3), derivative[2]),
                                    ]
                                )
                            )
                    elif right == 'T_2':
                        if i == 0:
                            self.rows.append(
                                Row(
                                    [
                                        gamma[0],
                                        self.multiply(1/sqrt(3), derivative[0]),
                                        gamma[1],
                                        self.multiply(1/sqrt(3), derivative[1]),
                                        gamma[2],
                                        self.multiply(-2/sqrt(3), derivative[2]),
                                    ]
                                )
                            )
                        elif i == 1:
                            self.rows.append(
                                Row(
                                    [
                                        gamma[0],
                                        derivative[0],
                                        gamma[1],
                                        self.multiply(-1, derivative[1]),
                                    ]
                                )
                            )
                elif projection in ["T_1", "T_2"]:
                    j = (i + 1) % 3
                    k = (i + 2) % 3
                    if right == projection:
                        self.rows.append(
                            Row(
                                [
                                    gamma[j],
                                    derivative[k],
                                    gamma[k],
                                    self.multiply(-1, derivative[j]),
                                ]
                            )
                        )
                    else:
                        self.rows.append(Row([gamma[j], derivative[k], gamma[k], derivative[j]]))

class InsertionGaugeLink(Insertion):
    def __init__(
        self,
        gamma: GammaName,
        gauge_link_irrep_name,
        gauge_link_idx,
        projection,
        momentum_dict: Dict[int, str],
        insertion_dict: Dict[int, str],
        profile=None,
    ) -> None:
        self.gamma = gamma_scheme(gamma)
        self.derivative = insertion_dict[gauge_link_irrep_name][gauge_link_idx]
        self.gauge_link_irrep_name = gauge_link_irrep_name
        self.gauge_link_idx = gauge_link_idx
        self.parity = gamma_parity(gamma)*gauge_parity(gauge_link_irrep_name)
        self.charge_conjugation = gamma_charge_conjugation(gamma)*gauge_charge_conjugate(gauge_link_irrep_name)
        self.hermiticity = gamma_hermiticity(gamma) * gauge_hermiticity(gauge_link_irrep_name)
        self.projection = [gamma_gourp(gamma), gauge_group(gauge_link_irrep_name), projection]
        self.momentum_dict = momentum_dict
        self.rows = []
        self.little_group_irreps_dict = {}
        self.profile = profile
        self.construct()

    def construct(self):
        gamma = self.gamma
        derivative = self.derivative
        left, right, projection = self.projection
        length = {"A_1": 1, "A_2": 1, "E": 2, "T_1": 3, "T_2": 3}
        irrep_T1 = {
            "A_1": ["T_1"],
            "A_2": ["T_2"],
            "E": ["T_1", "T_2"],
            "T_1": ["A_1", "T_1", "T_2", "E"],
            "T_2": ["T_1", "T_2", "E", "A_2"],
        }
        if left == "A_1":
            assert right == projection, f"{left} x {right} has no irrep {projection}"
            for i in range(length[projection]):
                self.rows.append(Row([gamma[0], derivative[i]]))
        elif left == "T_1":
            assert projection in irrep_T1[right], f"{left} x {right} has no irrep {projection}"
            for i in range(length[projection]):
                if right == "A_1":
                    self.rows.append(Row([gamma[i], derivative[0]]))
                elif right == "E":
                    if projection == "T_1":
                        if i == 0:
                            row = Row([gamma[0], derivative[0], gamma[0], self.multiply(-1/sqrt(3), derivative[1])])
                        elif i == 1:
                            row = Row([gamma[1], self.multiply(-1, derivative[0]), gamma[1], self.multiply(-1/sqrt(3), derivative[1])])
                        elif i == 2:
                            row = Row([gamma[2], self.multiply(2/sqrt(3), derivative[0])])
                    elif projection == "T_2":
                        if i == 0:
                            row = Row([gamma[0], derivative[0], gamma[0], self.multiply(sqrt(3), derivative[1])])
                        elif i == 1:
                            row = Row([gamma[1], derivative[0], gamma[1], self.multiply(-sqrt(3), derivative[1])])
                        elif i == 2:
                            row = Row([gamma[2], self.multiply(-2, derivative[0])])
                    row.simplify()
                    self.rows.append(row)
                elif projection in ["A_1", "A_2"]:
                    self.rows.append(
                        Row(
                            [
                                gamma[0],
                                derivative[0],
                                gamma[1],
                                derivative[1],
                                gamma[2],
                                derivative[2],
                            ]
                        )
                    )
                elif projection in ["E"]:
                    if right == 'T_1':
                        if i == 0:
                            self.rows.append(
                                Row(
                                    [
                                        gamma[0],
                                        derivative[0],
                                        gamma[1],
                                        self.multiply(-1, derivative[1]),
                                    ]
                                )
                            )
                        else:
                            self.rows.append(
                                Row(
                                    [
                                        gamma[0],
                                        self.multiply(-1/sqrt(3), derivative[0]),
                                        gamma[1],
                                        self.multiply(-1/sqrt(3), derivative[1]),
                                        gamma[2],
                                        self.multiply(2/sqrt(3), derivative[2]),
                                    ]
                                )
                            )
                    elif right == 'T_2':
                        if i == 0:
                            self.rows.append(
                                Row(
                                    [
                                        gamma[0],
                                        self.multiply(1/sqrt(3), derivative[0]),
                                        gamma[1],
                                        self.multiply(1/sqrt(3), derivative[1]),
                                        gamma[2],
                                        self.multiply(-2/sqrt(3), derivative[2]),
                                    ]
                                )
                            )
                        elif i == 1:
                            self.rows.append(
                                Row(
                                    [
                                        gamma[0],
                                        derivative[0],
                                        gamma[1],
                                        self.multiply(-1, derivative[1]),
                                    ]
                                )
                            )
                elif projection in ["T_1", "T_2"]:
                    j = (i + 1) % 3
                    k = (i + 2) % 3
                    if right == projection:
                        self.rows.append(
                            Row(
                                [
                                    gamma[j],
                                    derivative[k],
                                    gamma[k],
                                    self.multiply(-1, derivative[j]),
                                ]
                            )
                        )
                    else:
                        self.rows.append(Row([gamma[j], derivative[k], gamma[k], derivative[j]]))
