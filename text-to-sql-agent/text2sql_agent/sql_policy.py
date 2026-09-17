from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Literal

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.optimizer.scope import Scope, traverse_scope

Mode = Literal["read", "write"]

SYSTEM_DATABASES = {"information_schema", "mysql", "performance_schema", "sys"}
FORBIDDEN_FUNCTIONS = {
    "benchmark",
    "get_lock",
    "is_free_lock",
    "is_used_lock",
    "load_file",
    "master_pos_wait",
    "name_const",
    "release_all_locks",
    "release_lock",
    "sleep",
    "sys_exec",
    "sys_eval",
}
FORBIDDEN_SQL_PATTERNS = {
    r"\binto\s+(?:out|dump)file\b": "Writing query output to a file is prohibited.",
    r"\bfor\s+update\b": "Locking reads are prohibited.",
    r"\block\s+in\s+share\s+mode\b": "Locking reads are prohibited.",
    r"\binto\s+@": "Writing SELECT results into variables is prohibited.",
}


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    mode: Mode
    operation: str
    normalized_sql: str | None
    reasons: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)

    @property
    def requires_approval(self) -> bool:
        return self.mode == "write" and self.allowed


def sql_fingerprint(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def approval_code(sql: str) -> str:
    return f"APPROVE {sql_fingerprint(sql)[:12].upper()}"


def validate_sql(
    sql: str,
    *,
    mode: Mode,
    database: str,
    allowed_tables: set[str] | None = None,
    schema_catalog: dict[str, list[dict[str, object]]] | None = None,
    default_limit: int = 200,
    max_limit: int = 1000,
) -> PolicyDecision:
    reasons: list[str] = []
    cleaned = sql.strip()
    if not cleaned:
        return PolicyDecision(False, mode, "UNKNOWN", None, ["SQL is empty."])

    try:
        statements = [statement for statement in sqlglot.parse(cleaned, read="mysql") if statement]
    except ParseError as exc:
        return PolicyDecision(
            False,
            mode,
            "UNKNOWN",
            None,
            [f"SQL could not be parsed as MySQL: {exc}."],
        )

    if len(statements) != 1:
        return PolicyDecision(
            False,
            mode,
            "UNKNOWN",
            None,
            ["Exactly one SQL statement is required."],
        )

    expression = statements[0]
    operation = _operation(expression)
    permitted = {"SELECT"} if mode == "read" else {"INSERT", "UPDATE", "DELETE"}
    if operation not in permitted:
        reasons.append(f"{operation} is not permitted in {mode} mode.")

    table_names: list[str] = []
    allowed_lower = (
        {name.lower() for name in allowed_tables} if allowed_tables is not None else None
    )
    configured_database = database.lower()
    cte_names = {cte.alias_or_name.lower() for cte in expression.find_all(exp.CTE)}
    for table in expression.find_all(exp.Table):
        table_name = table.name
        if not table.db and table_name.lower() in cte_names:
            continue
        table_database = (table.db or configured_database).lower()
        if table_database in SYSTEM_DATABASES:
            reasons.append(f"System database `{table_database}` is prohibited.")
        elif table_database != configured_database:
            reasons.append(
                f"Cross-database access to `{table_database}` is prohibited; "
                f"only `{database}` is allowed."
            )
        if allowed_lower is not None and table_name.lower() not in allowed_lower:
            reasons.append(f"Table `{table_name}` is not in the allowed table catalog.")
        if table_name and table_name not in table_names:
            table_names.append(table_name)

    lower_sql = expression.sql(dialect="mysql").lower()
    for pattern, message in FORBIDDEN_SQL_PATTERNS.items():
        if re.search(pattern, lower_sql, flags=re.IGNORECASE):
            reasons.append(message)

    for node in expression.walk():
        function_name = _function_name(node)
        if function_name in FORBIDDEN_FUNCTIONS:
            reasons.append(f"Function `{function_name.upper()}` is prohibited.")

    if schema_catalog is not None:
        reasons.extend(_validate_column_references(expression, schema_catalog, mode))

    if mode == "read" and operation == "SELECT":
        expression, limit_reasons = _apply_read_limit(expression, default_limit, max_limit)
        reasons.extend(limit_reasons)

    if mode == "write" and operation in {"UPDATE", "DELETE"}:
        where = expression.args.get("where")
        if where is None:
            reasons.append(f"{operation} requires a restrictive WHERE clause.")
        elif _contains_obvious_tautology(where.this):
            reasons.append(f"{operation} contains an unsafe tautological WHERE clause.")

    normalized = expression.sql(dialect="mysql", pretty=True)
    return PolicyDecision(
        allowed=not reasons,
        mode=mode,
        operation=operation,
        normalized_sql=normalized,
        reasons=_dedupe(reasons),
        tables=table_names,
    )


def _operation(expression: exp.Expression) -> str:
    if isinstance(expression, exp.Query):
        return "SELECT"
    if isinstance(expression, exp.Insert):
        return "INSERT"
    if isinstance(expression, exp.Update):
        return "UPDATE"
    if isinstance(expression, exp.Delete):
        return "DELETE"
    return expression.key.upper() if expression.key else "UNKNOWN"


def _function_name(node: exp.Expression) -> str | None:
    if isinstance(node, exp.Anonymous):
        return node.name.lower()
    if isinstance(node, exp.Func):
        return node.sql_name().lower()
    return None


def _apply_read_limit(
    expression: exp.Expression,
    default_limit: int,
    max_limit: int,
) -> tuple[exp.Expression, list[str]]:
    reasons: list[str] = []
    limit = expression.args.get("limit")
    if limit is None:
        return expression.limit(default_limit), reasons

    limit_expression = limit.expression
    if not isinstance(limit_expression, exp.Literal) or not limit_expression.is_int:
        reasons.append("LIMIT must be a literal integer.")
        return expression, reasons
    value = int(limit_expression.this)
    if value <= 0:
        reasons.append("LIMIT must be greater than zero.")
    elif value > max_limit:
        reasons.append(f"LIMIT cannot exceed {max_limit} rows.")
    return expression, reasons


def _contains_obvious_tautology(predicate: exp.Expression) -> bool:
    if isinstance(predicate, exp.Paren):
        return _contains_obvious_tautology(predicate.this)
    if isinstance(predicate, exp.Boolean):
        return bool(predicate.this)
    if isinstance(predicate, exp.Literal) and predicate.is_int:
        return int(predicate.this) != 0
    if isinstance(predicate, exp.EQ):
        left, right = predicate.left, predicate.right
        if isinstance(left, exp.Literal) and isinstance(right, exp.Literal):
            return left.this == right.this
        if isinstance(left, exp.Column) and isinstance(right, exp.Column):
            return left.sql(dialect="mysql").lower() == right.sql(dialect="mysql").lower()
    if isinstance(predicate, exp.Or):
        return _contains_obvious_tautology(predicate.left) or _contains_obvious_tautology(
            predicate.right
        )
    if isinstance(predicate, exp.And):
        return _contains_obvious_tautology(predicate.left) and _contains_obvious_tautology(
            predicate.right
        )
    return False


def _validate_column_references(
    expression: exp.Expression,
    schema_catalog: dict[str, list[dict[str, object]]],
    mode: Mode,
) -> list[str]:
    """Reject ambiguous or schema-invalid model-generated column references."""
    reasons: list[str] = []
    catalog = {
        table_name.lower(): {
            str(column["column_name"]).lower()
            for column in columns
            if column.get("column_name")
        }
        for table_name, columns in schema_catalog.items()
    }

    for scope in traverse_scope(expression):
        for column in scope.columns:
            if column.is_star:
                continue
            if not column.table:
                if mode == "read" and not _is_select_alias_reference(column, scope):
                    reasons.append(
                        f"Column `{column.name}` must be qualified with a table alias."
                    )
                continue

            source = _resolve_source(scope, column.table)
            if source is None:
                reasons.append(
                    f"Column `{column.sql(dialect='mysql')}` uses unknown table alias "
                    f"`{column.table}`."
                )
                continue

            if isinstance(source, exp.Table):
                table_name = source.name.lower()
                available = catalog.get(table_name)
                if available is not None and column.name.lower() not in available:
                    reasons.append(
                        f"Column `{column.name}` does not exist on table `{source.name}` "
                        f"(referenced as `{column.table}`)."
                    )
            else:
                output_names = {name.lower() for name in source.expression.named_selects}
                if column.name.lower() not in output_names:
                    reasons.append(
                        f"Column `{column.name}` is not produced by derived source "
                        f"`{column.table}`."
                    )

    return _dedupe(reasons)


def _resolve_source(scope: Scope, alias: str) -> exp.Table | Scope | None:
    current: Scope | None = scope
    alias_lower = alias.lower()
    while current is not None:
        for source_alias, source in current.sources.items():
            if source_alias.lower() == alias_lower and isinstance(source, (exp.Table, Scope)):
                return source
        current = current.parent
    return None


def _is_select_alias_reference(column: exp.Column, scope: Scope) -> bool:
    if not isinstance(scope.expression, exp.Select):
        return False
    aliases = {
        projection.alias.lower()
        for projection in scope.expression.expressions
        if projection.alias
    }
    if column.name.lower() not in aliases:
        return False

    parent = column.parent
    while parent is not None and parent is not scope.expression:
        if isinstance(parent, (exp.Order, exp.Group, exp.Having, exp.Qualify)):
            return True
        parent = parent.parent
    return False


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
