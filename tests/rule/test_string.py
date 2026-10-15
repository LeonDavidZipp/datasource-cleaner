import sys
from collections.abc import Callable, Sequence

import polars as pl
import pytest
import regex
from hypothesis import given
from hypothesis import strategies as st

from datasource_cleaner.rule import StringRule
from datasource_cleaner.rule._string import Case

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


# -------------------------------------- _replace --------------------------------------


@st.composite
def replacements_and_values(
    draw: st.DrawFn,
) -> tuple[dict[str, str], list[str | None]]:
    replacements = draw(
        st.dictionaries(st.characters(codec="utf-8"), st.text(), min_size=1)
    )
    olds = st.sampled_from(sorted(replacements))
    value = st.lists(st.one_of(st.characters(codec="utf-8"), olds)).map("".join)
    values = draw(st.lists(st.one_of(st.none(), value)))
    return replacements, values


@given(case=replacements_and_values())
def test_replace(case: tuple[dict[str, str], list[str | None]]):
    replacements, values = case
    rule = StringRule(replace=list(replacements.items()))
    expected = [
        None if v is None else "".join(replacements.get(c, c) for c in v)
        for v in values
    ]
    assert _apply_string(values, rule._replace) == expected  # type: ignore


@given(values=st.lists(st.one_of(st.none(), st.text())))
def test_replace_empty_leaves_untouched(values: list[str | None]):
    rule = StringRule(replace=())
    assert _apply_string(values, rule._replace) == values  # type: ignore


# polars' default (non-leftmost) matching: the match that ends first wins,
# regardless of the order the pairs are given in
@pytest.mark.parametrize(
    ("replace", "values", "expected"),
    [
        ([("ab", "1"), ("b", "2")], ["ab", "xabcx", "bab"], ["1", "x1cx", "21"]),
        ([("b", "2"), ("ab", "1")], ["ab", "xabcx", "bab"], ["1", "x1cx", "21"]),
        ([("abc", "1"), ("b", "2")], ["abc", "xabcx"], ["a2c", "xa2cx"]),
        ([("aa", "1"), ("a", "2")], ["aaa", "aab"], ["222", "22b"]),
        ([("bc", "1"), ("ab", "2")], ["abc", "bcab"], ["2c", "12"]),
        ([("ab", "ba")], ["abab", "aabb"], ["baba", "abab"]),
    ],
)
def test_replace_multi_char_patterns(
    replace: list[tuple[str, str]], values: list[str], expected: list[str]
):
    rule = StringRule(replace=replace)
    assert _apply_string(values, rule._replace) == expected  # type: ignore


# ----------------------------------- _regex_replace -----------------------------------


literal_patterns = st.text(alphabet=st.characters(categories=("L", "N")), min_size=1)
replacements = st.text(alphabet=st.characters(codec="utf-8", exclude_characters="$"))


@st.composite
def regex_replacements_and_values(
    draw: st.DrawFn,
) -> tuple[list[tuple[str, str]], list[str | None]]:
    pairs = draw(st.lists(st.tuples(literal_patterns, replacements), min_size=1))
    fragment = st.one_of(st.text(), st.sampled_from([p for p, _ in pairs]))
    value = st.lists(fragment).map("".join)
    values = draw(st.lists(st.one_of(st.none(), value)))
    return pairs, values


@given(case=regex_replacements_and_values())
def test_regex_replace_applies_in_order(
    case: tuple[list[tuple[str, str]], list[str | None]],
):
    pairs, values = case
    rule = StringRule(regex_replace=pairs)
    expected: list[str | None] = []
    for v in values:
        if v is not None:
            for pattern, replacement in pairs:
                v = v.replace(pattern, replacement)
        expected.append(v)
    assert _apply_string(values, rule._regex_replace) == expected  # type: ignore


@given(values=st.lists(st.one_of(st.none(), st.text())))
def test_regex_replace_empty_leaves_untouched(values: list[str | None]):
    rule = StringRule(regex_replace=())
    assert _apply_string(values, rule._regex_replace) == values  # type: ignore


@pytest.mark.parametrize(
    ("regex_replace", "values", "expected"),
    [
        ([(r"\d+", "#")], ["a1b22c333"], ["a#b#c#"]),
        ([(r"^\s+|\s+$", "")], ["  a b  "], ["a b"]),
        ([(r"(\w+)@(\w+)", "$2 at $1")], ["me@host"], ["host at me"]),
        ([(r"(?<user>\w+)@", "${user}:")], ["me@host"], ["me:host"]),
        ([(r"(?i)abc", "x")], ["ABC aBc abc"], ["x x x"]),
        ([(r"a", "b"), (r"b", "c")], ["ab"], ["cc"]),
        ([(r"b", "c"), (r"a", "b")], ["ab"], ["bc"]),
        ([(r"x", "y")], [None, ""], [None, ""]),
    ],
)
def test_regex_replace_examples(
    regex_replace: list[tuple[str, str]],
    values: list[str | None],
    expected: list[str | None],
):
    rule = StringRule(regex_replace=regex_replace)
    assert _apply_string(values, rule._regex_replace) == expected  # type: ignore


# --------------------------------------- _case ----------------------------------------


@pytest.mark.parametrize(
    ("case", "reference"), [("lower", str.lower), ("upper", str.upper)]
)
@given(values=st.lists(st.one_of(st.none(), st.text())))
def test_case(case: Case, reference: Callable[[str], str], values: list[str | None]):
    rule = StringRule(case=case)
    expected = [None if v is None else reference(v) for v in values]
    assert _apply_string(values, rule._case) == expected  # type: ignore


_ALPHABETIC = regex.compile(r"\p{Alphabetic}")


def _title_reference(value: str) -> str:
    out: list[str] = []
    in_word = False
    for c in value:
        out.append(c.lower() if in_word else c.title())
        in_word = _ALPHABETIC.match(c) is not None
    return "".join(out)


# regex may know newer Unicode chars than polars does, so leave out unassigned ones
title_text = st.text(alphabet=st.characters(codec="utf-8", exclude_categories=("Cn",)))


@given(values=st.lists(st.one_of(st.none(), title_text)))
def test_case_title(values: list[str | None]):
    rule = StringRule(case="title")
    expected = [None if v is None else _title_reference(v) for v in values]
    assert _apply_string(values, rule._case) == expected  # type: ignore


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["hELLO wORLD"], ["Hello World"]),
        (["o'neil", "foo-bar", "foo_bar"], ["O'Neil", "Foo-Bar", "Foo_Bar"]),
        (["a1b c2d"], ["A1B C2D"]),
        (["ǆungla"], ["ǅungla"]),
        (["straße"], ["Straße"]),
        (["ΣΑΣ"], ["Σας"]),
        (["ٮA"], ["ٮa"]),
        (["xͅA"], ["Xͅa"]),
    ],
)
def test_case_title_examples(values: list[str], expected: list[str]):
    rule = StringRule(case="title")
    assert _apply_string(values, rule._case) == expected  # type: ignore


@given(values=st.lists(st.one_of(st.none(), st.text())))
def test_case_none_leaves_untouched(values: list[str | None]):
    rule = StringRule(case=None)
    assert _apply_string(values, rule._case) == values  # type: ignore


# ------------------------------------ _max_length -------------------------------------


@given(
    max_length=st.integers(min_value=0, max_value=20),
    values=st.lists(st.one_of(st.none(), st.text())),
)
def test_max_length(max_length: int, values: list[str | None]):
    rule = StringRule(max_length=max_length)
    expected = [None if v is None else v[:max_length] for v in values]
    assert _apply_string(values, rule._max_length) == expected  # type: ignore


@given(values=st.lists(st.one_of(st.none(), st.text())))
def test_max_length_none_leaves_untouched(values: list[str | None]):
    rule = StringRule(max_length=None)
    assert _apply_string(values, rule._max_length) == values  # type: ignore


# ---------------------------------------- _pad ----------------------------------------


widths = st.integers(min_value=0, max_value=20)
pad_chars = st.characters(codec="utf-8")


@given(
    width=widths,
    pad_char=pad_chars,
    values=st.lists(st.one_of(st.none(), st.text())),
)
def test_pad_left(width: int, pad_char: str, values: list[str | None]):
    rule = StringRule(pad_left=width, pad_char=pad_char)
    expected = [None if v is None else v.rjust(width, pad_char) for v in values]
    assert _apply_string(values, rule._pad_left) == expected  # type: ignore


@given(
    width=widths,
    pad_char=pad_chars,
    values=st.lists(st.one_of(st.none(), st.text())),
)
def test_pad_right(width: int, pad_char: str, values: list[str | None]):
    rule = StringRule(pad_right=width, pad_char=pad_char)
    expected = [None if v is None else v.ljust(width, pad_char) for v in values]
    assert _apply_string(values, rule._pad_right) == expected  # type: ignore


@given(values=st.lists(st.one_of(st.none(), st.text())))
def test_pad_none_leaves_untouched(values: list[str | None]):
    rule = StringRule(pad_left=None, pad_right=None)
    assert _apply_string(values, rule._pad_left) == values  # type: ignore
    assert _apply_string(values, rule._pad_right) == values  # type: ignore


# ----------------------------------- _empty_as_null -----------------------------------


@given(values=st.lists(st.one_of(st.none(), st.just(""), st.text())))
def test_empty_as_null(values: list[str | None]):
    rule = StringRule(empty_as_null=True)
    expected = [v or None for v in values]
    assert _apply_string(values, rule._empty_as_null) == expected  # type: ignore


@given(values=st.lists(st.one_of(st.none(), st.just(""), st.text())))
def test_empty_as_null_false_leaves_untouched(values: list[str | None]):
    rule = StringRule(empty_as_null=False)
    assert _apply_string(values, rule._empty_as_null) == values  # type: ignore


# ----------------------------------- __post_init__ ------------------------------------


@pytest.mark.parametrize("name", ["max_length", "pad_left", "pad_right"])
@given(value=st.integers(max_value=-1))
def test_negative_lengths_raise(name: str, value: int):
    with pytest.raises(ValueError, match=f"{name} must be non-negative"):
        StringRule(**{name: value})  # type: ignore


@given(pad_left=widths, pad_right=widths)
def test_pad_left_and_pad_right_are_exclusive(pad_left: int, pad_right: int):
    with pytest.raises(ValueError, match="mutually exclusive"):
        StringRule(pad_left=pad_left, pad_right=pad_right)


@given(pad_char=st.text().filter(lambda s: len(s) != 1))
def test_pad_char_must_be_single_char(pad_char: str):
    with pytest.raises(ValueError, match="pad_char must be a single character"):
        StringRule(pad_char=pad_char)
