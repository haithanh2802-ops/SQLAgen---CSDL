from __future__ import annotations

import argparse
import csv

from rich.console import Console
from sqlalchemy import create_engine

from text2sql_agent.config import Settings

CSV_TABLES = [
    ("product_category_name_translation.csv", "product_category_translation"),
    ("olist_customers_dataset.csv", "customers"),
    ("olist_sellers_dataset.csv", "sellers"),
    ("olist_products_dataset.csv", "products"),
    ("olist_orders_dataset.csv", "orders"),
    ("olist_order_items_dataset.csv", "order_items"),
    ("olist_order_payments_dataset.csv", "order_payments"),
    ("olist_order_reviews_dataset.csv", "order_reviews"),
    ("olist_geolocation_dataset.csv", "geolocation"),
]

COLUMN_RENAMES = {
    "products": {
        "product_name_lenght": "product_name_length",
        "product_description_lenght": "product_description_length",
    }
}


def split_sql_statements(script: str) -> list[str]:
    return [statement.strip() for statement in script.split(";") if statement.strip()]


def prepare_database(settings: Settings) -> None:
    server_engine = create_engine(settings.setup_database_url(include_database=False))
    database_name = settings.mysql_database.replace("`", "``")
    with server_engine.begin() as connection:
        connection.exec_driver_sql(
            f"CREATE DATABASE IF NOT EXISTS `{database_name}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
    server_engine.dispose()

    schema_path = settings.project_root / "sql" / "schema.sql"
    engine = create_engine(settings.setup_database_url())
    with engine.begin() as connection:
        for statement in split_sql_statements(schema_path.read_text(encoding="utf-8")):
            connection.exec_driver_sql(statement)
    engine.dispose()


def load_csvs(settings: Settings, *, truncate: bool = False, batch_size: int = 5000) -> None:
    engine = create_engine(settings.setup_database_url())
    raw = engine.raw_connection()
    try:
        cursor = raw.cursor()
        for filename, table_name in CSV_TABLES:
            path = settings.knowledge_directory / filename
            if not path.exists():
                raise FileNotFoundError(f"Required dataset file is missing: {path}")
            if truncate:
                cursor.execute("SET FOREIGN_KEY_CHECKS=0")
                cursor.execute(f"TRUNCATE TABLE `{table_name}`")
                cursor.execute("SET FOREIGN_KEY_CHECKS=1")

            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                source_columns = list(reader.fieldnames or [])
                renames = COLUMN_RENAMES.get(table_name, {})
                target_columns = [renames.get(column, column) for column in source_columns]
                if table_name == "geolocation":
                    target_columns = [
                        column for column in target_columns if column != "geolocation_id"
                    ]
                quoted = ", ".join(f"`{column}`" for column in target_columns)
                placeholders = ", ".join(["%s"] * len(target_columns))
                sql = f"INSERT INTO `{table_name}` ({quoted}) VALUES ({placeholders})"

                batch: list[tuple[object, ...]] = []
                for row in reader:
                    batch.append(tuple(_clean(row[column]) for column in source_columns))
                    if len(batch) >= batch_size:
                        cursor.executemany(sql, batch)
                        raw.commit()
                        batch.clear()
                if batch:
                    cursor.executemany(sql, batch)
                    raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()
        engine.dispose()


def _clean(value: str) -> object:
    stripped = value.strip()
    return None if stripped == "" else stripped


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and load the local Olist MySQL database.")
    parser.add_argument("--load", action="store_true", help="Load all Olist CSV files.")
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Clear each target table before loading. Intended for a fresh local setup.",
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    settings = Settings.from_env()
    console = Console()
    prepare_database(settings)
    console.print(f"[green]Schema ready in `{settings.mysql_database}`.[/green]")
    if args.load:
        with console.status("Loading Olist CSV files into MySQL..."):
            load_csvs(settings, truncate=args.truncate, batch_size=args.batch_size)
        console.print("[green]Olist data load complete.[/green]")


if __name__ == "__main__":
    main()
