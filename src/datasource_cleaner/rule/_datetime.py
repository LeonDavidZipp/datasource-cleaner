from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal
from zoneinfo import ZoneInfo

from ._range import RangeRule

TruncateUnit = Literal["second", "minute", "hour", "day", "month", "year"]


@dataclass(frozen=True, kw_only=True)
class DateRule(RangeRule[date]):
    """
    Configures how a date column is transformed.

    null_values is useful for sentinel dates,
    e.g. (date(1900, 1, 1), date(9999, 12, 31)).
    """


@dataclass(frozen=True, kw_only=True)
class DateTimeRule(RangeRule[datetime]):
    """
    Configures how a timestamp column is transformed.

    Options are applied in the following order:
    null_values -> assume_timezone -> timezone -> truncate
    -> min_value / max_value (with out_of_range) -> fill_null
    """

    assume_timezone: str | None = None
    timezone: str | None = None
    truncate: TruncateUnit | None = None

    def __post_init__(self):
        super().__post_init__()
        for name in ("assume_timezone", "timezone"):
            value = getattr(self, name)
            if value is not None:
                ZoneInfo(value)
