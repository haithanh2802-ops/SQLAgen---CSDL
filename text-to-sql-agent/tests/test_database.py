from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Self

from text2sql_agent.database import DatabaseGateway, _execute_literal_sql


class FakeRows(Iterable[Mapping[str, Any]]):
    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        yield {
            "TABLE_NAME": "olist_orders_dataset",
            "COLUMN_NAME": "order_id",
            "COLUMN_TYPE": "varchar(32)",
            "IS_NULLABLE": "NO",
            "COLUMN_KEY": "PRI",
            "COLUMN_DEFAULT": None,
            "EXTRA": "",
            "CHARACTER_SET_NAME": "utf8mb4",
            "COLLATION_NAME": "utf8mb4_0900_ai_ci",
            "REFERENCED_TABLE_NAME": None,
            "REFERENCED_COLUMN_NAME": None,
        }


class FakeResult:
    def mappings(self) -> FakeRows:
        return FakeRows()


class FakeConnection:
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, *_args: object, **_kwargs: object) -> FakeResult:
        return FakeResult()


class FakeEngine:
    def connect(self) -> FakeConnection:
        return FakeConnection()


def test_schema_catalog_normalizes_mysql_metadata_key_casing() -> None:
    gateway = object.__new__(DatabaseGateway)
    gateway.settings = type("Settings", (), {"mysql_database": "olist"})()
    gateway._read_engine = FakeEngine()
    gateway._write_engine = None

    catalog = gateway.schema_catalog()

    assert list(catalog) == ["olist_orders_dataset"]
    assert catalog["olist_orders_dataset"][0]["column_name"] == "order_id"
    assert catalog["olist_orders_dataset"][0]["collation_name"] == "utf8mb4_0900_ai_ci"


def test_literal_sql_disables_dbapi_percent_interpolation() -> None:
    class RecordingConnection:
        def __init__(self) -> None:
            self.statement: str | None = None
            self.execution_options: dict[str, bool] | None = None

        def exec_driver_sql(
            self,
            statement: str,
            *,
            execution_options: dict[str, bool],
        ) -> str:
            self.statement = statement
            self.execution_options = execution_options
            return "result"

    connection = RecordingConnection()
    sql = "SELECT DATE_FORMAT(order_purchase_timestamp, '%Y-%m') AS order_month"

    result = _execute_literal_sql(connection, sql)

    assert result == "result"
    assert connection.statement == sql
    assert connection.execution_options == {"no_parameters": True}
