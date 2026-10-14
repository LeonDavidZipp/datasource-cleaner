from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ._base import ColumnRule

Case = Literal["lower", "upper", "title"]


@dataclass(frozen=True, kw_only=True)
class StringRule(ColumnRule):
    """
    Configures how a string column is transformed.

    Options are applied in the following order:
    null_values -> trim -> collapse_whitespace -> replace -> regex_replace
    -> case -> max_length -> pad -> empty_as_null
    """

    trim_left: bool | Sequence[str] = True
    trim_right: bool | Sequence[str] = True
    null_values: Sequence[str] = ()
    null_values_ignore_case: bool = False
    collapse_whitespace: bool = False
    replace: Sequence[tuple[str, str]] = ()
    regex_replace: Sequence[tuple[str, str]] = ()
    case: Case | None = None
    max_length: int | None = None
    pad_left: int | None = None
    pad_right: int | None = None
    pad_char: str = " "
    empty_as_null: bool = False

    def __post_init__(self):
        for name in ("max_length", "pad_left", "pad_right"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if self.pad_left is not None and self.pad_right is not None:
            raise ValueError("pad_left and pad_right are mutually exclusive")
        if len(self.pad_char) != 1:
            raise ValueError(
                f"pad_char must be a single character, got {self.pad_char!r}"
            )
