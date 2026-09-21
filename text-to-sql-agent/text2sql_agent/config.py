from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL


@dataclass(frozen=True)
class Settings:
    project_root: Path
    ollama_base_url: str
    chat_model: str
    embedding_model: str
    ollama_timeout_seconds: float
    ollama_num_gpu: int | None
    embedding_batch_size: int
    mysql_host: str
    mysql_port: int
    mysql_database: str
    mysql_read_user: str
    mysql_read_password: str
    mysql_write_user: str
    mysql_write_password: str
    mysql_setup_user: str
    mysql_setup_password: str
    mysql_writable_tables: frozenset[str]
    mysql_connect_timeout: int
    mysql_max_execution_ms: int
    sql_default_limit: int
    sql_max_limit: int
    sql_max_affected_rows: int
    result_summary_max_rows: int
    knowledge_directory: Path
    chroma_directory: Path
    chroma_collection: str
    audit_log: Path

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> Settings:
        root = (project_root or Path(__file__).resolve().parents[1]).resolve()
        load_dotenv(root / ".env")

        def local_path(name: str, default: str) -> Path:
            value = Path(os.getenv(name, default))
            return value.resolve() if value.is_absolute() else (root / value).resolve()

        def optional_int(name: str) -> int | None:
            value = os.getenv(name, "").strip()
            return int(value) if value else None

        settings = cls(
            project_root=root,
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            chat_model=os.getenv("OLLAMA_CHAT_MODEL", "qwen3-coder:30b"),
            embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:latest"),
            ollama_timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600")),
            ollama_num_gpu=optional_int("OLLAMA_NUM_GPU"),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "16")),
            mysql_host=os.getenv("MYSQL_HOST", "localhost"),
            mysql_port=int(os.getenv("MYSQL_PORT", "3306")),
            mysql_database=os.getenv("MYSQL_DATABASE", "olist"),
            mysql_read_user=os.getenv("MYSQL_READ_USER", ""),
            mysql_read_password=os.getenv("MYSQL_READ_PASSWORD", ""),
            mysql_write_user=os.getenv("MYSQL_WRITE_USER", ""),
            mysql_write_password=os.getenv("MYSQL_WRITE_PASSWORD", ""),
            mysql_setup_user=os.getenv("MYSQL_SETUP_USER", "root"),
            mysql_setup_password=os.getenv("MYSQL_SETUP_PASSWORD", ""),
            mysql_writable_tables=frozenset(
                table.strip()
                for table in os.getenv(
                    "MYSQL_WRITABLE_TABLES",
                    "customers,geolocation,order_items,order_payments,order_reviews,"
                    "orders,products,sellers,product_category_translation",
                ).split(",")
                if table.strip()
            ),
            mysql_connect_timeout=int(os.getenv("MYSQL_CONNECT_TIMEOUT", "10")),
            mysql_max_execution_ms=int(os.getenv("MYSQL_MAX_EXECUTION_MS", "10000")),
            sql_default_limit=int(os.getenv("SQL_DEFAULT_LIMIT", "200")),
            sql_max_limit=int(os.getenv("SQL_MAX_LIMIT", "1000")),
            sql_max_affected_rows=int(os.getenv("SQL_MAX_AFFECTED_ROWS", "100")),
            result_summary_max_rows=int(os.getenv("RESULT_SUMMARY_MAX_ROWS", "100")),
            knowledge_directory=local_path("KNOWLEDGE_DIRECTORY", "../dataset"),
            chroma_directory=local_path("CHROMA_DIRECTORY", ".local/chroma"),
            chroma_collection=os.getenv("CHROMA_COLLECTION", "olist_business_knowledge"),
            audit_log=local_path("AUDIT_LOG", ".local/audit.jsonl"),
        )
        settings._validate()
        return settings

    def _validate(self) -> None:
        for name in (
            "mysql_port",
            "embedding_batch_size",
            "mysql_connect_timeout",
            "mysql_max_execution_ms",
            "sql_default_limit",
            "sql_max_limit",
            "sql_max_affected_rows",
            "result_summary_max_rows",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero.")
        if self.sql_default_limit > self.sql_max_limit:
            raise ValueError("SQL_DEFAULT_LIMIT cannot exceed SQL_MAX_LIMIT.")
        if self.ollama_num_gpu is not None and self.ollama_num_gpu < -1:
            raise ValueError("OLLAMA_NUM_GPU must be -1 or greater.")

    def database_url(self, *, read_only: bool) -> URL:
        user = self.mysql_read_user if read_only else self.mysql_write_user
        password = self.mysql_read_password if read_only else self.mysql_write_password
        if not user:
            role = "MYSQL_READ_USER" if read_only else "MYSQL_WRITE_USER"
            raise RuntimeError(f"{role} must be configured.")
        return URL.create(
            "mysql+pymysql",
            username=user,
            password=password,
            host=self.mysql_host,
            port=self.mysql_port,
            database=self.mysql_database,
            query={"charset": "utf8mb4"},
        )

    def setup_database_url(self, *, include_database: bool = True) -> URL:
        if not self.mysql_setup_user:
            raise RuntimeError("MYSQL_SETUP_USER must be configured for database setup.")
        return URL.create(
            "mysql+pymysql",
            username=self.mysql_setup_user,
            password=self.mysql_setup_password,
            host=self.mysql_host,
            port=self.mysql_port,
            database=self.mysql_database if include_database else None,
            query={"charset": "utf8mb4"},
        )

    def assert_separate_database_identities(self) -> None:
        if not self.mysql_read_user or not self.mysql_write_user:
            raise RuntimeError("Both read and write MySQL identities must be configured.")
        if self.mysql_read_user == self.mysql_write_user:
            raise RuntimeError("Read and write MySQL identities must be different.")
