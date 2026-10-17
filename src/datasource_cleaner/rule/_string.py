import re
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

    def to_sql(self, dialect: Dialect) -> str:
        """
        Returns a str.format template of the SQL expression applying the rule,
        with the column as {col}.

        Differences to apply:
        - replace pairs are applied one after another, not simultaneously
        - regex patterns are passed on as is, so they follow the regex flavour of
          the dialect
        - case conversion follows the database, e.g. sqlite only converts ASCII and
          postgres' INITCAP continues words over digits

        Raises NotImplementedError for options the dialect can't express.
        """
        sql = "{col}"
        sql = self._trim_sql(sql, dialect)
        sql = self._replace_null_values_sql(sql, dialect)
        sql = self._collapse_whitespace_sql(sql, dialect)
        sql = self._replace_sql(sql, dialect)
        sql = self._regex_replace_sql(sql, dialect)
        sql = self._case_sql(sql, dialect)
        sql = self._max_length_sql(sql)
        sql = self._pad_sql(sql, dialect)
        return self._empty_as_null_sql(sql, dialect)

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

    def _trim_sql(self, sql: str, dialect: Dialect) -> str:
        for trim, func, anchored in (
            (self.trim_right, "RTRIM", "{}+\\z"),
            (self.trim_left, "LTRIM", "\\A{}+"),
        ):
            chars = _WHITESPACE if trim is True else trim
            if not chars:
                continue
            if dialect == "mysql":
                # mysql's TRIM removes a substring, not a set of characters
                pattern = anchored.format(_regex_char_class(chars, dialect))
                sql = f"REGEXP_REPLACE({sql}, {_sql_literal(pattern, dialect)}, '')"
            else:
                sql = f"{func}({sql}, {_sql_literal(chars, dialect)})"
        return sql

    def _replace_null_values_sql(self, sql: str, dialect: Dialect) -> str:
        if not self.null_values:
            return sql
        probe = sql
        null_values = self.null_values
        if self.null_values_ignore_case:
            probe = f"LOWER({probe})"
            null_values = [val.lower() for val in null_values]
        if dialect == "mysql":
            # mysql's default collations ignore case and accents
            probe = f"CAST({probe} AS BINARY)"
        literals = ", ".join(
            _sql_literal(val, dialect) for val in dict.fromkeys(null_values)
        )
        return f"CASE WHEN {probe} IN ({literals}) THEN NULL ELSE {sql} END"

    def _collapse_whitespace_sql(self, sql: str, dialect: Dialect) -> str:
        if not self.collapse_whitespace:
            return sql
        pattern = _regex_char_class(_WHITESPACE, dialect) + "+"
        return self._regexp_replace_sql(sql, pattern, " ", dialect)

    def _replace_sql(self, sql: str, dialect: Dialect) -> str:
        for old, new in self.replace:
            old, new = _sql_literal(old, dialect), _sql_literal(new, dialect)
            sql = f"REPLACE({sql}, {old}, {new})"
        return sql

    def _regex_replace_sql(self, sql: str, dialect: Dialect) -> str:
        for pattern, replacement in self.regex_replace:
            replacement = _regex_replacement(replacement, dialect)
            sql = self._regexp_replace_sql(sql, pattern, replacement, dialect)
        return sql

    def _regexp_replace_sql(
        self, sql: str, pattern: str, replacement: str, dialect: Dialect
    ) -> str:
        if dialect == "sqlite":
            raise NotImplementedError("sqlite has no regex replacement")
        args = [sql, _sql_literal(pattern, dialect), _sql_literal(replacement, dialect)]
        if dialect == "postgres":
            # postgres only replaces the first match without the g flag
            args.append("'g'")
        return f"REGEXP_REPLACE({', '.join(args)})"

    def _case_sql(self, sql: str, dialect: Dialect) -> str:
        match self.case:
            case None:
                return sql
            case "lower":
                return f"LOWER({sql})"
            case "upper":
                return f"UPPER({sql})"
            case "title":
                if dialect not in ("postgres", "bigquery"):
                    raise NotImplementedError(f"{dialect} has no title case")
                return f"INITCAP({sql})"

    def _max_length_sql(self, sql: str) -> str:
        if self.max_length is not None:
            return f"SUBSTR({sql}, 1, {self.max_length})"
        return sql

    def _pad_sql(self, sql: str, dialect: Dialect) -> str:
        for width, func in ((self.pad_left, "LPAD"), (self.pad_right, "RPAD")):
            if width is None:
                continue
            pad_char = _sql_literal(self.pad_char, dialect)
            length = "CHAR_LENGTH" if dialect == "mysql" else "LENGTH"
            if dialect == "sqlite":
                # sqlite has no LPAD/RPAD, so build the padding from width zeros
                padding = (
                    f"SUBSTR(REPLACE(HEX(ZEROBLOB({width})), '00', {pad_char}), "
                    f"1, MAX({width} - LENGTH({sql}), 0))"
                )
                return (
                    f"{padding} || {sql}" if func == "LPAD" else f"{sql} || {padding}"
                )
            # LPAD and RPAD also truncate to width, which polars doesn't
            return f"{func}({sql}, GREATEST({width}, {length}({sql})), {pad_char})"
        return sql

    def _empty_as_null_sql(self, sql: str, dialect: Dialect) -> str:
        if not self.empty_as_null:
            return sql
        if dialect == "mysql":
            # mysql's PAD SPACE collations consider "  " equal to ""
            return f"CASE WHEN CHAR_LENGTH({sql}) = 0 THEN NULL ELSE {sql} END"
        return f"NULLIF({sql}, '')"


# Unicode White_Space, which polars strips by default and matches with \s
_WHITESPACE = "".join(
    map(
        chr,
        [
            *range(0x09, 0x0E),
            0x20,
            0x85,
            0xA0,
            0x1680,
            *range(0x2000, 0x200B),
            0x2028,
            0x2029,
            0x202F,
            0x205F,
            0x3000,
        ],
    )
)

# group references in polars (Rust regex) replacements: $$, ${name} and $name
_GROUP_REF = re.compile(r"\$(?:\$|\{([0-9A-Za-z_]+)\}|([0-9A-Za-z_]+))")


def _sql_literal(value: str, dialect: Dialect) -> str:
    if dialect == "bigquery":
        value = value.replace("\\", "\\\\").replace("'", "\\'")
        # bigquery doesn't allow raw newlines and the like in quoted strings
        value = "".join(c if c.isprintable() else f"\\U{ord(c):08x}" for c in value)
    else:
        if dialect == "mysql":
            value = value.replace("\\", "\\\\")
        value = value.replace("'", "''")
    # to_sql returns a str.format template, so literal braces are doubled
    return "'" + value.replace("{", "{{").replace("}", "}}") + "'"


def _regex_char_class(chars: str, dialect: Dialect) -> str:
    if dialect == "postgres":
        return "[" + "".join(f"\\U{ord(c):08x}" for c in chars) + "]"
    return "[" + "".join(f"\\x{{{ord(c):x}}}" for c in chars) + "]"


def _regex_replacement(replacement: str, dialect: Dialect) -> str:
    """
    Translates a polars regex replacement into the syntax of the dialect.
    """

    def literal(text: str) -> str:
        if dialect == "mysql":
            return re.sub(r"([\\$])", r"\\\1", text)
        return text.replace("\\", "\\\\")

    out: list[str] = []
    pos = 0
    after_ref = False
    for match in _GROUP_REF.finditer(replacement):
        text = literal(replacement[pos : match.start()])
        # stop a following digit from being read as part of the group number
        if after_ref and dialect == "mysql" and text[:1].isdigit():
            text = "\\" + text
        out.append(text)
        pos = match.end()
        name = match.group(1) or match.group(2)
        if name is None:
            out.append(literal("$"))
            after_ref = False
        elif dialect == "mysql":
            out.append(f"${name}" if name.isdigit() else f"${{{name}}}")
            after_ref = name.isdigit()
        elif name.isdigit() and int(name) <= 9:
            out.append(f"\\{name}")
            after_ref = False
        else:
            raise NotImplementedError(
                f"{dialect} only supports capture group references $0 to $9, "
                f"got ${name}"
            )
    text = literal(replacement[pos:])
    if after_ref and text[:1].isdigit():
        text = "\\" + text
    out.append(text)
    return "".join(out)
