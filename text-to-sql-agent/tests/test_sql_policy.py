import pytest

from text2sql_agent.sql_policy import approval_code, validate_sql

READ_ARGS = {
    "mode": "read",
    "database": "olist",
    "allowed_tables": {"orders", "customers", "order_items"},
    "default_limit": 200,
    "max_limit": 1000,
}

SCHEMA = {
    "orders": [
        {"column_name": "order_id"},
        {"column_name": "customer_id"},
        {"column_name": "order_status"},
    ],
    "customers": [
        {"column_name": "customer_id"},
        {"column_name": "customer_unique_id"},
    ],
    "order_items": [
        {"column_name": "order_id"},
        {"column_name": "price"},
    ],
}


def test_read_query_gets_a_default_limit():
    decision = validate_sql("SELECT order_id FROM orders", **READ_ARGS)
    assert decision.allowed
    assert "LIMIT 200" in decision.normalized_sql


def test_cte_alias_does_not_need_to_be_a_physical_table():
    decision = validate_sql(
        "WITH recent AS (SELECT order_id FROM orders) SELECT order_id FROM recent",
        **READ_ARGS,
    )
    assert decision.allowed


def test_unqualified_read_column_is_blocked_with_live_schema():
    decision = validate_sql(
        "SELECT customer_unique_id FROM orders AS o",
        schema_catalog=SCHEMA,
        **READ_ARGS,
    )
    assert not decision.allowed
    assert "must be qualified" in "; ".join(decision.reasons)


def test_qualified_column_must_exist_on_referenced_table():
    decision = validate_sql(
        "SELECT o.customer_unique_id FROM orders AS o",
        schema_catalog=SCHEMA,
        **READ_ARGS,
    )
    assert not decision.allowed
    assert "does not exist" in "; ".join(decision.reasons)


def test_qualified_join_columns_pass_live_schema_validation():
    decision = validate_sql(
        "SELECT c.customer_unique_id FROM customers AS c "
        "JOIN orders AS o ON o.customer_id = c.customer_id",
        schema_catalog=SCHEMA,
        **READ_ARGS,
    )
    assert decision.allowed


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM other_database.orders LIMIT 10",
        "SELECT LOAD_FILE('/etc/passwd') LIMIT 1",
        "SELECT SLEEP(30) LIMIT 1",
        "SELECT GET_LOCK('analytics', 30) LIMIT 1",
        "SELECT order_id FROM orders FOR UPDATE LIMIT 1",
        "SHOW PROCESSLIST",
        "SELECT order_id FROM unknown_table LIMIT 1",
        "SELECT order_id FROM orders LIMIT 1001",
        "SELECT order_id FROM orders; DELETE FROM orders",
    ],
)
def test_dangerous_read_queries_are_blocked(sql):
    assert not validate_sql(sql, **READ_ARGS).allowed


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "DELETE FROM orders WHERE TRUE",
        "DELETE FROM orders WHERE 1 = 1",
        "UPDATE orders SET order_status = 'x' WHERE 'a' = 'a'",
        "UPDATE orders SET order_status = 'x' WHERE order_id = order_id",
    ],
)
def test_unrestricted_writes_are_blocked(sql):
    decision = validate_sql(
        sql,
        mode="write",
        database="olist",
        allowed_tables={"orders"},
    )
    assert not decision.allowed


def test_restrictive_update_requires_approval():
    decision = validate_sql(
        "UPDATE orders SET order_status = 'cancelled' WHERE order_id = 'abc'",
        mode="write",
        database="olist",
        allowed_tables={"orders"},
    )
    assert decision.allowed
    assert decision.requires_approval
    assert decision.operation == "UPDATE"


def test_numeric_filter_is_not_mistaken_for_a_tautology():
    decision = validate_sql(
        "UPDATE order_items SET price = 12.50 WHERE order_item_id = 1",
        mode="write",
        database="olist",
        allowed_tables={"order_items"},
    )
    assert decision.allowed


def test_ddl_is_not_allowed_in_write_mode():
    decision = validate_sql(
        "DROP TABLE orders",
        mode="write",
        database="olist",
        allowed_tables={"orders"},
    )
    assert not decision.allowed


def test_empty_write_allowlist_blocks_every_table():
    decision = validate_sql(
        "UPDATE orders SET order_status = 'x' WHERE order_id = 'abc'",
        mode="write",
        database="olist",
        allowed_tables=set(),
    )
    assert not decision.allowed


def test_approval_code_is_bound_to_exact_sql():
    first = approval_code("DELETE FROM orders WHERE order_id = 'one'")
    second = approval_code("DELETE FROM orders WHERE order_id = 'two'")
    assert first.startswith("APPROVE ")
    assert first != second
