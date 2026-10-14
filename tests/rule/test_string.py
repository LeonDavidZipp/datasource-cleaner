from collections.abc import Callable

import polars as pl
from hypothesis import given
from hypothesis import strategies as st

from datasource_cleaner.rule import StringRule

trim_chars = st.lists(st.characters(), unique=True, min_size=1).map("".join)


@st.composite
def chars_and_values(draw: st.DrawFn) -> tuple[str, list[str]]:
    chars = draw(trim_chars)
    padding = st.text(alphabet=chars)
    values = draw(st.lists(st.tuples(padding, st.text(), padding).map("".join)))
    return chars, values


def _apply(values: list[str], method: Callable[[pl.Expr], pl.Expr]) -> list[str]:
    lf = pl.DataFrame({"col": values}, schema={"col": pl.String})
    return lf.select(method(pl.col("col"))).get_column("col").to_list()


@given(case=chars_and_values())
def test_trim_left_chars(case: tuple[str, list[str]]):
    chars, values = case
    rule = StringRule(trim_left=chars, trim_right=False)
    assert _apply(values, rule._trim_left) == [v.lstrip(chars) for v in values]  # type: ignore


@given(case=chars_and_values())
def test_trim_right_chars(case: tuple[str, list[str]]):
    chars, values = case
    rule = StringRule(trim_left=False, trim_right=chars)
    assert _apply(values, rule._trim_right) == [v.rstrip(chars) for v in values]  # type: ignore
