import sys
from collections.abc import Callable, Sequence

import polars as pl
from hypothesis import given
from hypothesis import strategies as st

from datasource_cleaner.rule import StringRule

# -------------------------------------- helpers ---------------------------------------


def _apply_string(
    values: Sequence[str | None], method: Callable[[pl.Expr], pl.Expr]
) -> list[str | None]:
    lf = pl.DataFrame({"col": values}, schema={"col": pl.String})
    return lf.select(method(pl.col("col"))).get_column("col").to_list()


trim_chars = st.lists(st.characters(codec="utf-8"), unique=True, min_size=1).map(
    "".join
)


@st.composite
def chars_and_values(draw: st.DrawFn) -> tuple[str, list[str]]:
    chars = draw(trim_chars)
    padding = st.text(alphabet=chars)
    values = draw(st.lists(st.tuples(padding, st.text(), padding).map("".join)))
    return chars, values


@st.composite
def null_values_and_values(
    draw: st.DrawFn,
) -> tuple[list[str], list[str | None]]:
    null_values = draw(st.lists(st.text(), unique=True))
    value = st.text()
    if null_values:
        hit = st.sampled_from(null_values).flatmap(
            lambda v: st.sampled_from([v, v.lower(), v.upper(), v.swapcase()])
        )
        value = st.one_of(value, hit)
    values = draw(st.lists(st.one_of(st.none(), value)))
    return null_values, values


# --------------------------------------- _trim ----------------------------------------


@given(case=chars_and_values())
def test_trim_left_chars(case: tuple[str, list[str]]):
    chars, values = case
    rule = StringRule(trim_left=chars, trim_right=False)
    assert _apply_string(values, rule._trim_left) == [v.lstrip(chars) for v in values]  # type: ignore


@given(case=chars_and_values())
def test_trim_right_chars(case: tuple[str, list[str]]):
    chars, values = case
    rule = StringRule(trim_left=False, trim_right=chars)
    assert _apply_string(values, rule._trim_right) == [v.rstrip(chars) for v in values]  # type: ignore


# -------------------------------- _replace_null_values --------------------------------


@given(case=null_values_and_values())
def test_replace_null_values(case: tuple[list[str], list[str | None]]):
    null_values, values = case
    rule = StringRule(null_values=null_values)
    expected = [None if v in null_values else v for v in values]
    assert _apply_string(values, rule._replace_null_values) == expected  # type: ignore


@given(case=null_values_and_values())
def test_replace_null_values_ignore_case(case: tuple[list[str], list[str | None]]):
    null_values, values = case
    rule = StringRule(null_values=null_values, null_values_ignore_case=True)
    lower_null_values = {v.lower() for v in null_values}
    expected = [
        None if v is not None and v.lower() in lower_null_values else v for v in values
    ]
    assert _apply_string(values, rule._replace_null_values) == expected  # type: ignore


# -------------------- _collapse_whitespaces_to_regular_whitespace ---------------------


_WHITESPACE_CHARS = [
    c
    for i in range(sys.maxunicode + 1)
    if (c := chr(i)).isspace() and c not in "\x1c\x1d\x1e\x1f"
]


words = st.text(
    alphabet=st.characters(codec="utf-8").filter(lambda c: not c.isspace()), min_size=1
)
whitespace = st.text(alphabet=st.sampled_from(_WHITESPACE_CHARS), min_size=1)


@st.composite
def whitespace_string(draw: st.DrawFn) -> tuple[str, str]:
    parts = draw(st.lists(words, min_size=2, max_size=11))
    text = parts[0]
    for part in parts[1:]:
        text += draw(whitespace) + part
    return text, " ".join(parts)


@st.composite
def whitespace_strings(draw: st.DrawFn) -> tuple[list[str], list[str]]:
    cases = draw(st.lists(whitespace_string(), min_size=1))
    texts, expected = zip(*cases)
    return list(texts), list(expected)


@given(case=whitespace_strings())
def test_collapse_whitespaces(case: tuple[list[str], list[str]]):
    values, expected = case
    rule = StringRule(collapse_whitespace=True)
    result = _apply_string(values, rule._collapse_whitespaces_to_regular_whitespace)  # type: ignore
    assert result == expected


@given(case=whitespace_strings())
def test_collapse_whitespaces_false_leaves_untouched(case: tuple[list[str], list[str]]):
    values, _ = case
    rule = StringRule(collapse_whitespace=False)
    result = _apply_string(values, rule._collapse_whitespaces_to_regular_whitespace)  # type: ignore
    assert result == values
