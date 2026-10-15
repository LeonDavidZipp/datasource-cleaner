from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

import polars as pl

from ._base import ColumnRule, Dialect

Case = Literal["lower", "upper", "title"]


@dataclass(frozen=True, kw_only=True)
class StringRule(ColumnRule):
    """
    Configures how a string column is transformed.

    Options are applied in the following order:
    null_values -> trim -> collapse_whitespace -> replace -> regex_replace
    -> case -> max_length -> pad -> empty_as_null
    """

    trim_left: Annotated[
        bool | str,
        "Strip leading whitespace if True, or the given characters if a sequence.",
    ] = True
    trim_right: Annotated[
        bool | str,
        "Strip trailing whitespace if True, or the given characters if a sequence.",
    ] = True
    null_values: Annotated[Sequence[str], "Values that are replaced with null."] = ()
    null_values_ignore_case: Annotated[
        bool, "Match null_values case-insensitively."
    ] = False
    collapse_whitespace: Annotated[
        bool, "Replace runs of whitespace with a single space."
    ] = False
    replace: Annotated[
        Sequence[tuple[str, str]], "Literal (old, new) replacements"
    ] = ()
    regex_replace: Annotated[
        Sequence[tuple[str, str]],
        "Regex (pattern, replacement) substitutions, applied in order. "
        "Replacements reference capture groups as $1 or ${name}, not \\1.",
    ] = ()
    case: Annotated[Case | None, "Convert to this case, or leave as is if None."] = None
    max_length: Annotated[int | None, "Truncate to at most this many characters."] = (
        None
    )
    pad_left: Annotated[
        int | None, "Left-pad with pad_char to this width. Excludes pad_right."
    ] = None
    pad_right: Annotated[
        int | None, "Right-pad with pad_char to this width. Excludes pad_left."
    ] = None
    pad_char: Annotated[str, "Single character used for padding."] = " "
    empty_as_null: Annotated[
        bool, "Replace empty strings with null after all other steps."
    ] = False

    def apply(self, exp: pl.Expr) -> pl.Expr:
        exp = self._trim(exp)
        exp = self._replace_null_values(exp)
        exp = self._collapse_whitespaces_to_regular_whitespace(exp)
        exp = self._replace(exp)
        exp = self._regex_replace(exp)
        exp = self._case(exp)
        exp = self._max_length(exp)
        exp = self._pad_left(exp)
        exp = self._pad_right(exp)
        return self._empty_as_null(exp)

    def to_sql(self, dialect: Dialect) -> str: ...

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

    def _trim(self, exp: pl.Expr) -> pl.Expr:
        return self._trim_left(self._trim_right(exp))

    def _trim_left(self, exp: pl.Expr) -> pl.Expr:
        trim = self.trim_left
        if trim is False:
            return exp
        elif trim is True:
            return exp.str.strip_chars_start()
        return exp.str.strip_chars_start(trim)

    def _trim_right(self, exp: pl.Expr) -> pl.Expr:
        trim = self.trim_right
        if trim is False:
            return exp
        elif trim is True:
            return exp.str.strip_chars_end()
        return exp.str.strip_chars_end(trim)

    def _replace_null_values(self, exp: pl.Expr) -> pl.Expr:
        if self.null_values_ignore_case:
            lower_null_values = tuple(val.lower() for val in self.null_values)
            return (
                pl.when(exp.str.to_lowercase().is_in(lower_null_values))
                .then(pl.lit(None))
                .otherwise(exp)
                .name.keep()
            )
        return exp.replace(self.null_values, pl.lit(None))

    def _collapse_whitespaces_to_regular_whitespace(self, exp: pl.Expr) -> pl.Expr:
        if self.collapse_whitespace:
            return exp.str.replace_all(r"\s+", " ")
        return exp

    def _replace(self, exp: pl.Expr) -> pl.Expr:
        if not self.replace:
            return exp
        olds, news = zip(*self.replace)
        return exp.str.replace_many(list(olds), list(news))

    def _regex_replace(self, exp: pl.Expr) -> pl.Expr:
        if not self.regex_replace:
            return exp
        for pattern, replacement in self.regex_replace:
            exp = exp.str.replace_all(pattern, replacement)
        return exp

    def _case(self, exp: pl.Expr) -> pl.Expr:
        match self.case:
            case None:
                return exp
            case "lower":
                return exp.str.to_lowercase()
            case "upper":
                return exp.str.to_uppercase()
            case "title":
                return exp.str.to_titlecase()

    def _max_length(self, exp: pl.Expr) -> pl.Expr:
        if self.max_length is not None:
            return exp.str.slice(0, self.max_length)
        return exp

    def _pad_left(self, exp: pl.Expr) -> pl.Expr:
        if self.pad_left is not None:
            return exp.str.pad_start(self.pad_left, self.pad_char)
        return exp

    def _pad_right(self, exp: pl.Expr) -> pl.Expr:
        if self.pad_right is not None:
            return exp.str.pad_end(self.pad_right, self.pad_char)
        return exp

    def _empty_as_null(self, exp: pl.Expr) -> pl.Expr:
        if self.empty_as_null:
            return (
                pl.when(exp.str.len_chars() == 0)
                .then(pl.lit(None))
                .otherwise(exp)
                .name.keep()
            )
        return exp
