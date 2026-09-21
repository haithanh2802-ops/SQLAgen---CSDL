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
COMPATIBLE_TEXT_COLLATION = "utf8mb4_unicode_ci"
KNOWN_OLIST_TABLE_ALIASES = {
    "customers": "olist_customers_dataset",
    "geolocation": "olist_geolocation_dataset",
    "order_items": "olist_order_items_dataset",
    "order_payments": "olist_order_payments_dataset",
    "order_reviews": "olist_order_reviews_dataset",
    "orders": "olist_orders_dataset",
    "products": "olist_products_dataset",
    "sellers": "olist_sellers_dataset",
    "product_category_translation": "product_category_name_translation",
}
OLIST_ORDER_TABLES = {"orders", "olist_orders_dataset"}
OLIST_ORDER_STATUSES = {
    "approved",
    "canceled",
    "created",
    "delivered",
    "invoiced",
    "processing",
    "shipped",
    "unavailable",
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
    if schema_catalog is not None:
        expression = _normalize_known_table_aliases(expression, schema_catalog)
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
        expression = _normalize_cross_collation_equalities(expression, schema_catalog)

    if mode == "read" and operation == "SELECT":
        expression, limit_reasons = _apply_read_limit(expression, default_limit, max_limit)
        reasons.extend(limit_reasons)

    if mode == "write" and operation in {"UPDATE", "DELETE"}:
        where = expression.args.get("where")
        if where is None:
            reasons.append(f"{operation} requires a restrictive WHERE clause.")
        elif _contains_obvious_tautology(where.this):
            reasons.append(f"{operation} contains an unsafe tautological WHERE clause.")

    if mode == "write" and operation == "UPDATE":
        reasons.extend(_normalize_order_status_update(expression))

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
                    candidates = _unqualified_column_candidates(column, scope, catalog)
                    if len(candidates) > 1:
                        aliases = ", ".join(f"`{alias}`" for alias in candidates)
                        reasons.append(
                            f"Column `{column.name}` is ambiguous across source aliases "
                            f"{aliases}; qualify it with the intended alias."
                        )
                    elif not candidates:
                        reasons.append(
                            f"Column `{column.name}` is not available from any visible source; "
                            "qualify it with a valid source alias or compute it in a CTE."
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


def _normalize_cross_collation_equalities(
    expression: exp.Expression,
    schema_catalog: dict[str, list[dict[str, object]]],
) -> exp.Expression:
    """Make equality joins deterministic when live MySQL text collations differ."""
    collations = {
        table_name.lower(): {
            str(column["column_name"]).lower(): str(column["collation_name"])
            for column in columns
            if column.get("column_name") and column.get("collation_name")
        }
        for table_name, columns in schema_catalog.items()
    }
    visited: set[int] = set()
    for scope in traverse_scope(expression):
        for equality in scope.expression.find_all(exp.EQ):
            if id(equality) in visited:
                continue
            visited.add(id(equality))
            left_column = _collated_column(equality.left)
            right_column = _collated_column(equality.right)
            if left_column is None or right_column is None:
                continue
            left_collation = _column_collation(scope, left_column, collations)
            right_collation = _column_collation(scope, right_column, collations)
            if not left_collation or not right_collation or left_collation == right_collation:
                continue
            equality.set("this", _with_compatible_collation(left_column))
            equality.set("expression", _with_compatible_collation(right_column))
    return expression


def _normalize_known_table_aliases(
    expression: exp.Expression,
    schema_catalog: dict[str, list[dict[str, object]]],
) -> exp.Expression:
    """Resolve canonical Olist names only when the verified live equivalent exists."""
    live_tables = {name.lower(): name for name in schema_catalog}
    cte_names = {cte.alias_or_name.lower() for cte in expression.find_all(exp.CTE)}
    for table in expression.find_all(exp.Table):
        current = table.name.lower()
        if current in live_tables or current in cte_names:
            continue
        mapped = KNOWN_OLIST_TABLE_ALIASES.get(current)
        if mapped and mapped in live_tables:
            table.set("this", exp.to_identifier(live_tables[mapped]))
    return expression


def _normalize_order_status_update(expression: exp.Expression) -> list[str]:
    if not isinstance(expression, exp.Update) or not isinstance(expression.this, exp.Table):
        return []
    if expression.this.name.lower() not in OLIST_ORDER_TABLES:
        return []

    reasons: list[str] = []
    for assignment in expression.expressions:
        if not isinstance(assignment, exp.EQ):
            continue
        target = assignment.left
        value = assignment.right
        if not isinstance(target, exp.Column) or target.name.lower() != "order_status":
            continue
        if not isinstance(value, exp.Literal) or not value.is_string:
            reasons.append("`order_status` must be assigned a literal Olist status value.")
            continue
        normalized = str(value.this).strip().lower()
        if normalized == "cancelled":
            normalized = "canceled"
        if normalized not in OLIST_ORDER_STATUSES:
            allowed = ", ".join(sorted(OLIST_ORDER_STATUSES))
            reasons.append(f"Unknown Olist order status `{value.this}`; allowed values: {allowed}.")
            continue
        assignment.set("expression", exp.Literal.string(normalized))
    return reasons


def _collated_column(operand: exp.Expression) -> exp.Column | None:
    if isinstance(operand, exp.Column):
        return operand
    if isinstance(operand, exp.Collate) and isinstance(operand.this, exp.Column):
        return operand.this
    return None


def _column_collation(
    scope: Scope,
    column: exp.Column,
    collations: dict[str, dict[str, str]],
) -> str | None:
    if not column.table:
        return None
    source = _resolve_source(scope, column.table)
    if isinstance(source, exp.Table):
        return collations.get(source.name.lower(), {}).get(column.name.lower())
    if isinstance(source, Scope) and isinstance(source.expression, exp.Select):
        for projection in source.expression.expressions:
            if projection.alias_or_name.lower() != column.name.lower():
                continue
            projected = projection.this if isinstance(projection, exp.Alias) else projection
            if isinstance(projected, exp.Column):
                return _column_collation(source, projected, collations)
    return None


def _with_compatible_collation(column: exp.Column) -> exp.Collate:
    return exp.Collate(
        this=column.copy(),
        expression=exp.Var(this=COMPATIBLE_TEXT_COLLATION),
    )


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


def _unqualified_column_candidates(
    column: exp.Column,
    scope: Scope,
    catalog: dict[str, set[str]],
) -> list[str]:
    """Return visible aliases that can unambiguously supply an unqualified column."""
    column_name = column.name.lower()
    current: Scope | None = scope
    while current is not None:
        candidates: list[str] = []
        for alias, (_node, source) in current.selected_sources.items():
            if isinstance(source, exp.Table):
                available = catalog.get(source.name.lower(), set())
            elif isinstance(source, Scope):
                available = {name.lower() for name in source.expression.named_selects}
            else:
                continue
            if column_name in available:
                candidates.append(alias)
        if candidates:
            return candidates
        current = current.parent
    return []


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
