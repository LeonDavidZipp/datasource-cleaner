from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import polars as pl

Dialect = Literal["postgres", "bigquery", "mysql", "sqlite"]


@dataclass(frozen=True)
class ColumnRule(ABC):
    def __call__(self, exp: pl.Expr) -> pl.Expr:
        return self.apply(exp)

    @abstractmethod
    def apply(self, exp: pl.Expr) -> pl.Expr:
        """
        Apply the rule to the column.
        """

    @abstractmethod
    def to_sql(self, dialect: Dialect) -> str:
        """
        Converts the instance to sql of the given dialect, as a str.format
        template with the column as {col}
        """
