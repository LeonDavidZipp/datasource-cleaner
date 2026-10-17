from dataclasses import dataclass

from ._base import ColumnRule


@dataclass(frozen=True, kw_only=True)
class BooleanRule(ColumnRule):
    """
    Configures how a boolean column is transformed.

    Options are applied in the following order:
    invert -> fill_null
    """

    invert: bool = False
    fill_null: bool | None = None
