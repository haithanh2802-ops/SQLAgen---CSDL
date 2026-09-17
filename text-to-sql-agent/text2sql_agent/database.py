from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine, create_engine, text

from text2sql_agent.config import Settings
from text2sql_agent.contracts import QueryResult, WriteResult


@dataclass(frozen=True)
class ImpactPreview:
    estimated_rows: int | None
    plan: list[dict[str, Any]]


class DatabaseGateway:
    """Own the database credential boundary and all SQL execution."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._read_engine: Engine | None = None
        self._write_engine: Engine | None = None

    @property
    def read_engine(self) -> Engine:
        if self._read_engine is None:
            self._read_engine = self._make_engine(read_only=True)
        return self._read_engine

    @property
    def write_engine(self) -> Engine:
        if self._write_engine is None:
            self.settings.assert_separate_database_identities()
            self._write_engine = self._make_engine(read_only=False)
        return self._write_engine

    def _make_engine(self, *, read_only: bool) -> Engine:
        return create_engine(
            self.settings.database_url(read_only=read_only),
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args={"connect_timeout": self.settings.mysql_connect_timeout},
        )

    def health(self, *, read_only: bool) -> dict[str, Any]:
        engine = self.read_engine if read_only else self.write_engine
        with engine.connect() as connection:
            row = (
                connection.execute(
                    text("SELECT DATABASE() AS database_name, VERSION() AS mysql_version")
                )
                .mappings()
                .one()
            )
        return {"ok": True, "read_only": read_only, **dict(row)}

    def schema_catalog(self) -> dict[str, list[dict[str, Any]]]:
        query = text(
            """
            SELECT
                c.table_name,
                c.column_name,
                c.column_type,
                c.is_nullable,
                c.column_key,
                c.column_default,
                c.extra,
                k.referenced_table_name,
                k.referenced_column_name
            FROM information_schema.columns AS c
            LEFT JOIN information_schema.key_column_usage AS k
              ON k.table_schema = c.table_schema
             AND k.table_name = c.table_name
             AND k.column_name = c.column_name
             AND k.referenced_table_name IS NOT NULL
            WHERE c.table_schema = :database
            ORDER BY c.table_name, c.ordinal_position
            """
        )
        with self.read_engine.connect() as connection:
            rows = connection.execute(query, {"database": self.settings.mysql_database}).mappings()
            catalog: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                # MySQL exposes information_schema identifiers in uppercase on
                # some Windows installations, even when the SELECT expression
                # uses lowercase names. Keep the rest of the application and
                # model schema stable across server/driver configurations.
                normalized = {str(key).lower(): value for key, value in row.items()}
                catalog.setdefault(str(normalized["table_name"]), []).append(normalized)
        return catalog

    def execute_read(self, sql: str) -> QueryResult:
        started = time.perf_counter()
        cap = self.settings.sql_max_limit
        with self.read_engine.connect() as connection:
            connection.exec_driver_sql(
                f"SET SESSION MAX_EXECUTION_TIME={self.settings.mysql_max_execution_ms}"
            )
            connection.exec_driver_sql("START TRANSACTION READ ONLY")
            try:
                result = connection.exec_driver_sql(sql)
                columns = list(result.keys())
                fetched = result.mappings().fetchmany(cap + 1)
            finally:
                connection.rollback()

        truncated = len(fetched) > cap
        rows = [dict(row) for row in fetched[:cap]]
        return QueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            execution_ms=(time.perf_counter() - started) * 1000,
        )

    def preview_write(self, sql: str) -> ImpactPreview:
        with self.write_engine.connect() as connection:
            connection.exec_driver_sql(
                f"SET SESSION MAX_EXECUTION_TIME={self.settings.mysql_max_execution_ms}"
            )
            try:
                plan = [
                    dict(row) for row in connection.exec_driver_sql(f"EXPLAIN {sql}").mappings()
                ]
            finally:
                connection.rollback()

        estimates: list[int] = []
        for row in plan:
            value = row.get("rows")
            if value is not None:
                try:
                    estimates.append(int(value))
                except (TypeError, ValueError):
                    continue
        return ImpactPreview(estimated_rows=sum(estimates) if estimates else None, plan=plan)

    def execute_write(self, sql: str, *, operation: str) -> WriteResult:
        started = time.perf_counter()
        with self.write_engine.connect() as connection:
            transaction = connection.begin()
            try:
                connection.exec_driver_sql(
                    f"SET SESSION MAX_EXECUTION_TIME={self.settings.mysql_max_execution_ms}"
                )
                result = connection.exec_driver_sql(sql)
                affected_rows = max(result.rowcount, 0)
                if affected_rows > self.settings.sql_max_affected_rows:
                    raise RuntimeError(
                        f"Write affected {affected_rows} rows, exceeding the configured limit "
                        f"of {self.settings.sql_max_affected_rows}."
                    )
                transaction.commit()
            except Exception:
                transaction.rollback()
                raise
        return WriteResult(
            operation=operation,
            affected_rows=affected_rows,
            execution_ms=(time.perf_counter() - started) * 1000,
        )


def format_schema_catalog(catalog: dict[str, list[dict[str, Any]]]) -> str:
    lines: list[str] = []
    for table_name, columns in catalog.items():
        lines.append(f"Table: {table_name}")
        for column in columns:
            attributes = [str(column["column_type"])]
            attributes.append("NULL" if column["is_nullable"] == "YES" else "NOT NULL")
            if column.get("column_key"):
                attributes.append(str(column["column_key"]))
            if column.get("referenced_table_name"):
                attributes.append(
                    "REFERENCES "
                    f"{column['referenced_table_name']}({column['referenced_column_name']})"
                )
            lines.append(f"  - {column['column_name']}: {', '.join(attributes)}")
        lines.append("")
    return "\n".join(lines).strip()
