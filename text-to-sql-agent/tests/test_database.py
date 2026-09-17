from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Self

from text2sql_agent.database import DatabaseGateway


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
