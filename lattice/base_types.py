from typing import Literal, NamedTuple

Flavor = Literal["u", "d", "s", "c", "t", "b"]


class Tag(NamedTuple):
    tag: int
    time: int

    def __eq__(self, other):
        return self.tag == other.tag and self.time == other.time
