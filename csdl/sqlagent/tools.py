import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


WRITE_OPERATIONS = {
    "ALTER",
    "CREATE",
    "DELETE",
    "DROP",
    "INSERT",
    "RENAME",
    "TRUNCATE",
    "UPDATE",
}

READ_OPERATIONS = {"DESCRIBE", "EXPLAIN", "SELECT", "SHOW"}
FORBIDDEN_DATABASES = {"information_schema", "mysql", "performance_schema", "sys"}


@dataclass
class ValidationResult:
    allowed: bool
    operation: str
    risk: str
    reasons: list[str]
    requires_approval: bool


def get_connection(read_only: bool = True):
    try:
        import mysql.connector
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install dependencies first: pip install -r sqlagent\\requirements.txt") from exc

    user_env = "MYSQL_READ_USER" if read_only else "MYSQL_WRITE_USER"
    password_env = "MYSQL_READ_PASSWORD" if read_only else "MYSQL_WRITE_PASSWORD"

    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv(user_env, os.getenv("MYSQL_USER", "root")),
        password=os.getenv(password_env, os.getenv("MYSQL_PASSWORD", "")),
        database=os.getenv("MYSQL_DATABASE"),
        autocommit=False,
        connection_timeout=int(os.getenv("MYSQL_CONNECT_TIMEOUT", "10")),
    )


def fetch_schema() -> str:
    database = os.getenv("MYSQL_DATABASE")
    if not database:
        return "MYSQL_DATABASE is not configured."

    query = """
        SELECT
            table_name AS table_name,
            column_name AS column_name,
            column_type AS column_type,
            is_nullable AS is_nullable,
            column_key AS column_key,
            column_default AS column_default,
            extra AS extra
        FROM information_schema.columns
        WHERE table_schema = %s
        ORDER BY table_name, ordinal_position
    """

    connection = get_connection(read_only=True)
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(query, (database,))
        rows = cursor.fetchall()
    finally:
        cursor.close()
        connection.close()

    if not rows:
        return f"No tables found in database `{database}`."

    tables: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        normalized_row = {str(key).lower(): value for key, value in row.items()}
        tables.setdefault(normalized_row["table_name"], []).append(normalized_row)

    lines = [f"Database: {database}", ""]
    for table_name, columns in tables.items():
        lines.append(f"Table: {table_name}")
        for column in columns:
            details = [
                column["column_type"],
                "NULL" if column["is_nullable"] == "YES" else "NOT NULL",
            ]
            if column["column_key"]:
                details.append(column["column_key"])
            if column["column_default"] is not None:
                details.append(f"DEFAULT {column['column_default']}")
            if column["extra"]:
                details.append(column["extra"])
            lines.append(f"  - {column['column_name']}: {', '.join(details)}")
        lines.append("")

    return "\n".join(lines).strip()


def test_mysql_connection(read_only: bool = True) -> dict[str, Any]:
    connection = get_connection(read_only=read_only)
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute("SELECT DATABASE() AS database_name, VERSION() AS mysql_version")
        row = cursor.fetchone()
        connection.rollback()
        return {"ok": True, "read_only": read_only, **row}
    finally:
        cursor.close()
        connection.close()


def classify_sql(sql: str) -> str:
    cleaned = strip_sql_comments(sql).strip()
    match = re.match(r"^([a-zA-Z]+)", cleaned)
    return match.group(1).upper() if match else "UNKNOWN"


def strip_sql_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n\r]*", " ", sql)
    sql = re.sub(r"#[^\n\r]*", " ", sql)
    return sql


def validate_sql(sql: str) -> ValidationResult:
    reasons: list[str] = []
    cleaned = strip_sql_comments(sql).strip()
    operation = classify_sql(cleaned)

    if not cleaned:
        reasons.append("SQL is empty.")

    if ";" in cleaned.rstrip(";"):
        reasons.append("Multiple SQL statements are not allowed.")

    if operation == "UNKNOWN":
        reasons.append("Could not identify the SQL operation.")

    lowered = cleaned.lower()
    for database in FORBIDDEN_DATABASES:
        if re.search(rf"\b{re.escape(database)}\s*\.", lowered):
            reasons.append(f"System database `{database}` cannot be modified.")

    if operation in {"UPDATE", "DELETE"} and not re.search(r"\bwhere\b", lowered):
        reasons.append(f"{operation} requires a WHERE clause.")

    if operation == "SELECT" and not re.search(r"\blimit\b", lowered):
        reasons.append("SELECT queries must include a LIMIT clause.")

    if operation == "DROP":
        reasons.append("DROP is blocked by default. Run it manually after backup review.")

    if operation == "TRUNCATE":
        reasons.append("TRUNCATE is blocked by default. Use DELETE with a WHERE clause if appropriate.")

    is_write = operation in WRITE_OPERATIONS
    risk = "low"
    if operation in {"CREATE", "INSERT"}:
        risk = "medium"
    if operation in {"ALTER", "UPDATE", "DELETE", "DROP", "TRUNCATE", "RENAME"}:
        risk = "high"

    return ValidationResult(
        allowed=len(reasons) == 0,
        operation=operation,
        risk=risk,
        reasons=reasons,
        requires_approval=is_write,
    )


def execute_sql(sql: str, read_only: bool) -> dict[str, Any]:
    validation = validate_sql(sql)
    if not validation.allowed:
        return {
            "ok": False,
            "error": "SQL failed validation.",
            "validation_reasons": validation.reasons,
        }

    if read_only and validation.operation not in READ_OPERATIONS:
        return {
            "ok": False,
            "error": "Write SQL cannot run through the read-only connection.",
        }

    connection = get_connection(read_only=read_only)
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(f"SET SESSION MAX_EXECUTION_TIME={int(os.getenv('MYSQL_MAX_EXECUTION_MS', '5000'))}")
        cursor.execute(sql)

        if validation.operation in READ_OPERATIONS:
            rows = cursor.fetchall()
            connection.rollback()
            return {"ok": True, "operation": validation.operation, "rows": rows, "row_count": len(rows)}

        affected_rows = cursor.rowcount
        connection.commit()
        return {"ok": True, "operation": validation.operation, "affected_rows": affected_rows}
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


def append_audit_log(entry: dict[str, Any]) -> None:
    log_path = Path(os.getenv("SQL_AGENT_AUDIT_LOG", "logs/sqlagent_audit.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.now().isoformat(timespec="seconds"), **entry}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
