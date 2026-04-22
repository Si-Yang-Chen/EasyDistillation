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
from .base_types import Tag


class HadronIrrep(Symbol):
    def __new__(
        cls,
        hadron_name: str,
        momentum: List[int],
        irrep_name: str,
        parity: int,
        tag: Tag,
        dagger: bool = False,
    ):
        if parity is None:
            obj = super().__new__(
                cls,
                f"{hadron_name}({irrep_name}({tag.tag}),t={tag.time},{tuple(momentum)},{dagger})",
                commutative=False,
            )
        elif parity == -1:
            obj = super().__new__(
                cls,
                f"{hadron_name}({irrep_name}u({tag.tag}),t={tag.time},{tuple(momentum)},{parity},{dagger})",
                commutative=False,
            )
        else:
            obj = super().__new__(
                cls,
                f"{hadron_name}({irrep_name}g({tag.tag}),t={tag.time},{tuple(momentum)},{parity},{dagger})",
                commutative=False,
            )
        return obj

    def __init__(
        self,
        hadron_name: str,
        momentum: List[int],
        irrep_name: str,
        parity: int,
        tag: Tag,
        dagger: bool = False,
    ):
        """
        Initialize a HadronIrrep object.

        Args:
            name: The name of the hadron
            momentum: The momentum of the hadron
            irrep_name: The irrep name
            parity: The parity
            tag: The tag
        """
        self._hadron_name = hadron_name
        self._momentum = momentum.copy()
        self._irrep_name = irrep_name
        self._parity = parity
        self._tag = tag
        self._dagger = dagger

        if irrep_name.startswith("T"):
            self._lenth = 3
        elif irrep_name.startswith("G") or irrep_name.startswith("E"):
            self._lenth = 2
        elif irrep_name.startswith("H"):
            self._lenth = 4
        else:
            self._lenth = 1

    @property
    def hadron_name(self) -> str:
        return self._hadron_name

    @property
    def momentum(self) -> List[int]:
        return self._momentum.copy()

    @property
    def irrep_name(self) -> str:
        return self._irrep_name

    @property
    def parity(self) -> int:
        return self._parity

    @property
    def tag(self) -> Tag:
        return self._tag

    @property
    def dagger(self) -> bool:
        return self._dagger

    @property
    def lenth(self) -> int:
        return self._lenth

    def __getitem__(self, row_idx):
        return HadronIrrepRow(
            self._hadron_name,
            self._momentum,
            self._irrep_name,
            row_idx,
            self._parity,
            self._tag,
            self._dagger,
        )

    def __eq__(self, other):
        if not isinstance(other, HadronIrrep):
            return False
        else:
            return (
                self._hadron_name == other._hadron_name
                and self._momentum == other._momentum
                and self._irrep_name == other._irrep_name
                and self._parity == other._parity
                and self._tag == other._tag
                and self._dagger == other._dagger
            )

    def __hash__(self):
        return hash(
            (
                self._hadron_name,
                tuple(self._momentum),
                self._irrep_name,
                self._parity,
                self._tag,
                self._dagger,
            )
        )

    def copy(self):
        return HadronIrrep(
            self._hadron_name,
            self._momentum,
            self._irrep_name,
            self._parity,
            self._tag,
            self._dagger,
        )


class HadronIrrepRow(Symbol):
    def __new__(
        cls,
        hadron_name: str,
        momentum: List[int],
        irrep_name: str,
        row_idx: int,
        parity: int,
        tag: Tag,
        dagger: bool = False,
    ):
        if parity is None:
            obj = super().__new__(
                cls,
                rf"{hadron_name}({irrep_name},{tuple(momentum)}){'†' if dagger else ''}[{row_idx}]",
                commutative=False,
            )
        elif parity == -1:
            obj = super().__new__(
                cls,
                rf"{hadron_name}({irrep_name}u,{tuple(momentum)}){'†' if dagger else ''}[{row_idx}]",
                commutative=False,
            )
        else:
            obj = super().__new__(
                cls,
                rf"{hadron_name}({irrep_name}g,{tuple(momentum)}){'†' if dagger else ''}[{row_idx}]",
                commutative=False,
            )
        return obj

    def __init__(
        self,
        hadron_name: str,
        momentum: List[int],
        irrep_name: str,
        row_idx: int,
        parity: int,
        tag: Tag,
        dagger: bool = False,
    ):
        """
        Initialize a HadronIrrepRow object.

        Args:
            name: The name of the hadron
            momentum: The momentum of the hadron
            irrep_name: The irrep name
            row_idx: The row index
            parity: The parity
            tag: The tag
        """
        self._hadron_name = hadron_name
        self._tag = tag
        self._momentum = momentum.copy()
        self._irrep_name = irrep_name
        self._row_idx = row_idx
        self._parity = parity
        self._dagger = dagger
        self._rotate = genLittleGroupIrrep([0, 0, 0], "T_1", -1)
        self._little_group_matrix = genLittleGroupIrrep(momentum, irrep_name, parity, p_ref_irrep=True)

    @property
    def hadron_name(self) -> str:
        return self._hadron_name

    @property
    def tag(self) -> Tag:
        return self._tag

    @property
    def momentum(self) -> List[int]:
        return self._momentum.copy()

    @property
    def irrep_name(self) -> str:
        return self._irrep_name

    @property
    def row_idx(self) -> int:
        return self._row_idx

    @property
    def parity(self) -> int:
        return self._parity

    @property
    def dagger(self) -> bool:
        return self._dagger

    @property
    def rotate(self):
        return self._rotate

    @property
    def little_group_matrix(self):
        return self._little_group_matrix

    def __eq__(self, other):
        if not isinstance(other, HadronIrrepRow):
            return False
        else:
            return (
                self._hadron_name == other._hadron_name
                and self._momentum == other._momentum
                and self._irrep_name == other._irrep_name
                and self._row_idx == other._row_idx
                and self._parity == other._parity
                and self._tag == other._tag
                and self._dagger == other._dagger
            )

    def __hash__(self):
        return hash(
            (
                self._hadron_name,
                tuple(self._momentum),
                self._irrep_name,
                self._row_idx,
                self._parity,
                self._tag,
                self._dagger,
            )
        )

    def transform(self, group_element,time=None):
        momentum_final = list(self._rotate[group_element] @ Matrix(self._momentum))
        transform_matrix = self._little_group_matrix[wignerRotate(self._momentum, group_element)]
        result = S(0)
        for i in range(transform_matrix.shape[0]):
            result += transform_matrix[i, self._row_idx] * HadronIrrepRow(
                self._hadron_name,
                momentum_final,
                self._irrep_name,
                i,
                self._parity,
                self._tag,
                self._dagger,
            )
        return result

    def __lt__(self, other):
        if self._tag.time != other._tag.time:
            return self._tag.time < other._tag.time
        if self._irrep_name != other._irrep_name:
            return self._irrep_name < other._irrep_name
        if self._row_idx != other._row_idx:
            return self._row_idx < other._row_idx
        return self._tag.tag < other._tag.tag

    def __gt__(self, other):
        return other < self

    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return self > other or self == other

    def copy(self):
        return HadronIrrepRow(
            self._hadron_name,
            self._momentum,
            self._irrep_name,
            self._row_idx,
            self._parity,
            self._tag,
            self._dagger,
        )

    def conjugate(self):
        return HadronIrrepRow(
            self._hadron_name,
            self._momentum,
            self._irrep_name,
            self._row_idx,
            self._parity,
            self._tag,
            not self._dagger,
        )
