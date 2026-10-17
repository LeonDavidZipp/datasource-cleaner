from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeVar

from ._base import ColumnRule


class Comparable(Protocol):
    def __lt__(self, other: Any, /) -> bool: ...
    def __gt__(self, other: Any, /) -> bool: ...


T = TypeVar("T", bound=Comparable)

OutOfRange = Literal["null", "clamp", "error"]


@dataclass(frozen=True, kw_only=True)
class RangeRule(ColumnRule, Generic[T]):
    """
    Shared options for rules on ordered column types.

    Options are applied in the following order:
    null_values -> min_value / max_value (with out_of_range) -> fill_null
    """

    null_values: Sequence[T] = ()
    min_value: T | None = None
    max_value: T | None = None
    out_of_range: OutOfRange = "null"
    fill_null: T | None = None

    def __post_init__(self):
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError(
                "min_value must not exceed max_value, "
                f"got {self.min_value} > {self.max_value}"
            )
