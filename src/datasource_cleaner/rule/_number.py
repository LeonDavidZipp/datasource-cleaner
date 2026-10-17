from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from ._range import RangeRule

Rounding = Literal["round", "floor", "ceil", "truncate"]


# IntegerType = Literal[
#     "int8",
#     "int16",
#     "int32",
#     "int64",
#     "int128",
#     "uint8",
#     "uint16",
#     "uint32",
#     "uint64",
#     "uint128",
# ]
# FloatType = Literal["float16", "float32", "float64"]
# NumberType = IntegerType | FloatType | Literal["decimal"]

# _NUMBER_DTYPES: dict[str, type[pl.DataType]] = {
#     "int8": pl.Int8,
#     "int16": pl.Int16,
#     "int32": pl.Int32,
#     "int64": pl.Int64,
#     "int128": pl.Int128,
#     "uint8": pl.UInt8,
#     "uint16": pl.UInt16,
#     "uint32": pl.UInt32,
#     "uint64": pl.UInt64,
#     "uint128": pl.UInt128,
#     "float16": pl.Float16,
#     "float32": pl.Float32,
#     "float64": pl.Float64,
# }
# MAX_DECIMAL_PRECISION = 38


# @dataclass(frozen=True, kw_only=True)
# class StringNumberRule(StringRule):
#     """
#     Configures how a string column holding numbers, e.g. "1.000,21",
#     is converted to a numeric column.

#     The string options of StringRule are applied first, then:
#     currency_symbols -> negative_parentheses -> thousands_separator
#     -> decimal_separator -> percent -> number_type (with precision and scale)
#     -> on_invalid
#     """

#     decimal_separator: str = "."
#     thousands_separator: str | None = None
#     currency_symbols: Sequence[str] = ()
#     negative_parentheses: bool = False
#     percent: bool = False
#     number_type: NumberType = "float64"
#     precision: int | None = None
#     scale: int | None = None
#     on_invalid: Literal["null", "error"] = "null"

#     def __post_init__(self):
#         super().__post_init__()
#         if len(self.decimal_separator) != 1:
#             raise ValueError(
#                 "decimal_separator must be a single character, "
#                 + f"got {self.decimal_separator!r}"
#             )
#         if self.thousands_separator is not None:
#             if len(self.thousands_separator) != 1:
#                 raise ValueError(
#                     "thousands_separator must be a single character, got "
#                     + f"{self.thousands_separator!r}"
#                 )
#             if self.thousands_separator == self.decimal_separator:
#                 raise ValueError(
#                     "thousands_separator and decimal_separator must differ"
#                 )
#         if self.number_type != "decimal":
#             if self.precision is not None or self.scale is not None:
#                 raise ValueError(
#                     "precision and scale are only supported for number_type 'decimal'"
#                 )
#             return
#         if self.precision is not None and not (
#             1 <= self.precision <= MAX_DECIMAL_PRECISION
#         ):
#             raise ValueError(
#                 f"precision must be between 1 and {MAX_DECIMAL_PRECISION}, "
#                 + f"got {self.precision}"
#             )
#         if self.scale is not None:
#             if self.scale < 0:
#                 raise ValueError(f"scale must be non-negative, got {self.scale}")
#             if self.scale > (self.precision or MAX_DECIMAL_PRECISION):
#                 raise ValueError(
#                     f"scale must not exceed precision, got {self.scale} > "
#                     + f"{self.precision or MAX_DECIMAL_PRECISION}"
#                 )

#     @property
#     def dtype(self) -> pl.DataType:
#         """
#         The polars dtype the column is converted to.
#         """
#         if self.number_type == "decimal":
#             return pl.Decimal(self.precision, self.scale or 0)
#         return _NUMBER_DTYPES[self.number_type]()


@dataclass(frozen=True, kw_only=True)
class NumberRule(RangeRule[int | float | Decimal]):
    """
    Configures how a numeric column is transformed.

    Options are applied in the following order:
    null_values -> min_value / max_value (with out_of_range)
    -> decimals (with rounding) -> fill_null
    """

    decimals: int | None = None
    rounding: Rounding = "round"


@dataclass(frozen=True, kw_only=True)
class IntegerRule(NumberRule):
    """
    Configures how an integer column is transformed.

    decimals must be zero or negative, e.g. -2 rounds to hundreds.
    """

    def __post_init__(self):
        super().__post_init__()
        if self.decimals is not None and self.decimals > 0:
            raise ValueError(
                f"decimals must not be positive for integers, got {self.decimals}"
            )


@dataclass(frozen=True, kw_only=True)
class FloatRule(NumberRule):
    """
    Configures how a floating point column is transformed.

    nan_as_null and infinity_as_null are applied before all other options.
    """

    nan_as_null: bool = True
    infinity_as_null: bool = False


@dataclass(frozen=True, kw_only=True)
class DecimalRule(NumberRule):
    """
    Configures how a fixed point (NUMERIC / DECIMAL) column is transformed.
    """
