from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from ..rule import ColumnRule, Dialect


@dataclass(frozen=True, kw_only=True)
class SqlConnectionArgs:
    uri: str
    table: str
    schema: str | None = None
    dialect: Dialect
    partition_on: str | None = None
    partition_num: int | None = None


RenameFunc = Callable[[str], str]


class SqlCleaner:
    def __init__(
        self,
        columns: Sequence[str],
        rules: Sequence[ColumnRule],
        connection_args: SqlConnectionArgs,
        rename: Mapping[str, str] | RenameFunc = {},
        backup: bool = True,
    ):
        if not callable(rename):
            if a := set(rename) - set(columns):
                raise ValueError(
                    f"renames contains columns not specified in columns: {a}"
                )
        len_columns = len(columns)
        len_rules = len(rules)
        if len_columns != len_rules:
            raise ValueError(
                "Equal number of columns and rules need to be provided, "
                f"but received {len_columns=} and {len_rules=}"
            )
        self.schema = dict(zip(columns, rules, strict=True))
        self.connection_args = connection_args
        self.rename = rename
        self.backup = backup

    def _transform(self):
        args = self.connection_args
        lf = pl.read_database_uri(
            query=self._make_connection_str(),
            uri=args.uri,
            partition_on=args.partition_on,
            partition_num=args.partition_num,
        ).lazy()
        return lf.select(
            *[rule.apply(pl.col(col)).alias(col) for col, rule in self.schema.items()]
        ).rename(self.rename)

    def _make_connection_str(self) -> str:
        args = self.connection_args
        table = f"{args.schema or ''}{'.' if args.schema else ''}{args.table}"
        return f"SELECT {', '.join(self.schema.keys())} FROM {table}"
