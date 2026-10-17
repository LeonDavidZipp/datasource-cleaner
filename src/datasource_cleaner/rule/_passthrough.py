from polars import Expr

from ._base import ColumnRule, Dialect


class PassthroughRule(ColumnRule):
    def apply(self, exp: Expr) -> Expr:
        return exp

    def to_sql(self, dialect: Dialect) -> str:
        _ = dialect
        return "{col}"
