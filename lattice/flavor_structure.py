from typing import Literal, NamedTuple, List

import numpy as np
from sympy import Add, Mul, Dummy, Symbol, S, simplify, sqrt
from sympy.physics.quantum import Operator
from .symmetry.sympy_utils import convert_pow_to_mul
from .base_types import Tag, Flavor

Flavor = Literal["u", "d", "s", "c", "t", "b"]


class Qurak(Symbol):
    def __new__(cls, flavor: Flavor, tag: Tag, anti: bool, **assumptions) -> None:
        obj = super().__new__(
            cls, Rf"\bar{{{flavor}}}({tag.tag})" if anti else Rf"{flavor}({tag.tag})", commutative=False, **assumptions
        )
        return obj

    def __init__(self, flavor: Flavor, tag: Tag, anti: bool, **assumptions) -> None:
        """
        Initialize a Qurak object.

        Args:
            flavor: The flavor of the quark
            tag: The tag of the quark
            anti: Whether the quark is an anti-quark
            **assumptions: Additional assumptions for the Symbol
        """
        self.flavor = flavor
        self.tag = tag
        self.anti = anti


class Propagator(Symbol):
    """
    Symbol representing quark propagator
    """

    def __new__(cls, flavor: Flavor, source_tag: Tag, sink_tag: Tag, **assumptions) -> None:
        obj = super().__new__(cls, Rf"S^{flavor}({sink_tag.tag}, {source_tag.tag})", **assumptions)
        return obj

    def __init__(self, flavor: Flavor, source_tag: Tag, sink_tag: Tag, **assumptions) -> None:
        """
        Initialize a Propagator object.

        Args:
            flavor: The flavor of the propagator
            source_tag: The source tag of the propagator
            sink_tag: The sink tag of the propagator
            **assumptions: Additional assumptions for the Symbol
        """
        self.flavor = flavor
        self.source_tag = source_tag
        self.sink_tag = sink_tag
        self.tag = Rf"S^{flavor}_\mathrm{{local}}" if source_tag.time == sink_tag.time else Rf"S^{flavor}"


class HadronFlavorStructure(Operator):
    def __new__(cls, flavor_str: str, time: int = 0) -> None:
        if "bar" in flavor_str:
            # Handle cases like bar{uds}
            obj = super().__new__(cls, rf"bar{{{flavor_str[4:-1]}}}({time})")
        elif len(flavor_str) == 3:
            obj = super().__new__(cls, rf"{flavor_str}({time})")
        elif len(flavor_str) == 2:
            obj = super().__new__(cls, rf"{flavor_str}({time})")
        return obj

    def __init__(self, flavor_str: str, time: int = 0) -> None:
        """
        Initialize a HadronFlavorStructure object.

        Args:
            flavor_str: The flavor string
            time: The time value
        """
        self._flavor_str = flavor_str
        self._time = time

        if "bar" in flavor_str:
            # Handle cases like bar{uds}
            self._baryon_num = -1
            self._quark_list = []
            self._anti_quark_list = [c for c in flavor_str[4:-1]]
        elif len(flavor_str) == 3:
            self._baryon_num = 1
            self._quark_list = [c for c in flavor_str]
            self._anti_quark_list = []
        elif len(flavor_str) == 2:
            self._baryon_num = 0
            self._quark_list = [flavor_str[1]]
            self._anti_quark_list = [flavor_str[0]]

    @property
    def flavor_str(self) -> str:
        return self._flavor_str

    @property
    def time(self) -> int:
        return self._time

    @property
    def baryon_num(self) -> int:
        return self._baryon_num

    @property
    def quark_list(self) -> List[str]:
        return self._quark_list.copy()

    @property
    def anti_quark_list(self) -> List[str]:
        return self._anti_quark_list.copy()

    def conjugate(self) -> "HadronFlavorStructure":
        """
        Create a new HadronFlavorStructure instance with conjugated values.
        The original instance remains unchanged.
        """
        if self._baryon_num == 0:
            new_flavor_str = self._quark_list[0] + self._anti_quark_list[0]
        elif self._baryon_num == 1:
            new_flavor_str = f'bar{{{"".join(self._quark_list)}}}'
        elif self._baryon_num == -1:
            new_flavor_str = "".join(self._anti_quark_list)
        return HadronFlavorStructure(new_flavor_str, self._time)

    def _eval_is_zero(self):
        """Return True if this expression is zero, False otherwise."""
        return False

    def _eval_is_infinite(self):
        """Return True if this expression is infinite, False otherwise."""
        return False

    def _eval_is_extended_real(self):
        """Return True if this expression is extended real, False otherwise."""
        return True

    def _eval_is_finite(self):
        """Return True if this expression is finite, False otherwise."""
        return True
