from pathlib import Path

import pytest
from sqlalchemy.exc import OperationalError

from text2sql_agent.audit import AuditLogger
from text2sql_agent.config import Settings
from text2sql_agent.contracts import (
    BusinessSummary,
    IntentDecision,
    QueryResult,
    SqlProposal,
    WriteResult,
)
from text2sql_agent.database import ImpactPreview
from text2sql_agent.knowledge import RetrievedContext
from text2sql_agent.service import AnalyticsAgent

CATALOG = {
    "orders": [
        {
            "column_name": "order_id",
            "column_type": "varchar(32)",
            "is_nullable": "NO",
            "column_key": "PRI",
            "referenced_table_name": None,
            "referenced_column_name": None,
        },
        {
            "column_name": "order_status",
            "column_type": "varchar(20)",
            "is_nullable": "NO",
            "column_key": "",
            "referenced_table_name": None,
            "referenced_column_name": None,
        },
    ]
}

CUSTOMER_CATALOG = {
    "olist_customers_dataset": [
        {
            "column_name": "customer_id",
            "column_type": "varchar(50)",
            "is_nullable": "NO",
            "column_key": "PRI",
            "referenced_table_name": None,
            "referenced_column_name": None,
        },
        {
            "column_name": "customer_unique_id",
            "column_type": "varchar(50)",
            "is_nullable": "YES",
            "column_key": "",
            "referenced_table_name": None,
            "referenced_column_name": None,
        },
    ],
    "olist_orders_dataset": [
        {
            "column_name": "order_id",
            "column_type": "varchar(50)",
            "is_nullable": "NO",
            "column_key": "PRI",
            "referenced_table_name": None,
            "referenced_column_name": None,
        },
        {
            "column_name": "customer_id",
            "column_type": "varchar(50)",
            "is_nullable": "YES",
            "column_key": "",
            "referenced_table_name": None,
            "referenced_column_name": None,
        },
    ],
}


class StructuredRunner:
    def __init__(self, response):
        self.response = response

    def invoke(self, _messages):
        return self.response


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)

    def with_structured_output(self, _schema, **_kwargs):
        return StructuredRunner(self.responses.pop(0))


class FakeKnowledge:
    def retrieve(self, _question):
        return RetrievedContext("Order status business definition.", ["glossary (orders)"])


class FakeDatabase:
    def __init__(self):
        self.executed_write = None

    def schema_catalog(self):
        return CATALOG

    def execute_read(self, _sql):
        return QueryResult(
            columns=["order_status", "total"],
            rows=[{"order_status": "delivered", "total": 10}],
            row_count=1,
            execution_ms=4.2,
        )

    def preview_write(self, _sql):
        return ImpactPreview(estimated_rows=1, plan=[])

    def execute_write(self, sql, *, operation):
        self.executed_write = sql
        return WriteResult(operation=operation, affected_rows=1, execution_ms=2.1)


class TimeoutThenSuccessDatabase(FakeDatabase):
    def __init__(self):
        super().__init__()
        self.read_attempts = 0

    def schema_catalog(self):
        return CUSTOMER_CATALOG

    def execute_read(self, _sql):
        self.read_attempts += 1
        if self.read_attempts == 1:
            raise OperationalError(
                "SELECT",
                {},
                Exception(3024, "maximum statement execution time exceeded"),
            )
        return QueryResult(
            columns=["repeat_customers"],
            rows=[{"repeat_customers": 3}],
            row_count=1,
            execution_ms=8.0,
        )


def settings(monkeypatch, tmp_path: Path) -> Settings:
    monkeypatch.setenv("MYSQL_READ_USER", "reader")
    monkeypatch.setenv("MYSQL_WRITE_USER", "editor")
    monkeypatch.setenv("AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    return Settings.from_env(tmp_path)


def test_read_request_returns_grounded_analysis(monkeypatch, tmp_path):
    configured = settings(monkeypatch, tmp_path)
    progress_steps = []
    model = FakeModel(
        [
            IntentDecision(action="read", reason="Analytical question"),
            SqlProposal(
                intent="Count orders by status",
                sql=(
                    "SELECT o.order_status, COUNT(*) AS total "
                    "FROM orders AS o GROUP BY o.order_status"
                ),
                explanation="Groups orders by status.",
                chart_type="bar",
                chart_x="order_status",
                chart_y=["total"],
            ),
            BusinessSummary(answer="Ten delivered orders.", insights=["Delivered: 10"]),
        ]
    )
    agent = AnalyticsAgent(
        configured,
        database=FakeDatabase(),
        knowledge=FakeKnowledge(),
        model=model,
        audit=AuditLogger(configured.audit_log),
    )

    response = agent.ask(
        "How many orders are in each status?",
        progress=progress_steps.append,
    )

    assert response.kind == "analysis"
    assert response.result.row_count == 1
    assert "LIMIT 200" in response.proposal.sql
    assert response.summary.answer == "Ten delivered orders."
    assert response.retrieved_sources == ["glossary (orders)"]
    assert progress_steps == [
        "Classifying the request",
        "Inspecting the live database schema",
        "Retrieving relevant business context",
        "Generating a SQL proposal",
        "Validating schema references and safety policy",
        "Attempting the validated read-only query",
        "Summarizing the verified database result",
    ]


def test_write_requires_exact_approval(monkeypatch, tmp_path):
    configured = settings(monkeypatch, tmp_path)
    database = FakeDatabase()
    model = FakeModel(
        [
            IntentDecision(action="write", reason="Explicit correction"),
            SqlProposal(
                intent="Correct one order",
                sql="UPDATE orders SET order_status = 'cancelled' WHERE order_id = 'abc'",
                explanation="Changes one identified order.",
            ),
        ]
    )
    agent = AnalyticsAgent(
        configured,
        database=database,
        knowledge=FakeKnowledge(),
        model=model,
        audit=AuditLogger(configured.audit_log),
    )
    response = agent.ask("Change order abc to cancelled")

    assert response.kind == "write_preview"
    assert database.executed_write is None
    with pytest.raises(PermissionError):
        agent.execute_approved(response.write_preview, "APPROVE WRONG")
    result = agent.execute_approved(response.write_preview, response.write_preview.approval_code)
    assert result.affected_rows == 1
    assert database.executed_write is not None


def test_model_generated_write_cannot_run_through_read_route(monkeypatch, tmp_path):
    configured = settings(monkeypatch, tmp_path)
    model = FakeModel(
        [
            IntentDecision(action="read", reason="Question"),
            SqlProposal(
                intent="Unsafe proposal",
                sql="DELETE FROM orders WHERE order_id = 'abc'",
                explanation="Unsafe.",
            ),
            SqlProposal(
                intent="Still unsafe",
                sql="DELETE FROM orders WHERE order_id = 'abc'",
                explanation="Unsafe.",
            ),
        ]
    )
    agent = AnalyticsAgent(
        configured,
        database=FakeDatabase(),
        knowledge=FakeKnowledge(),
        model=model,
        audit=AuditLogger(configured.audit_log),
    )

    response = agent.ask("How many orders exist?")

    assert response.kind == "error"
    assert "blocked" in response.error


def test_timeout_is_repaired_and_retried_once(monkeypatch, tmp_path):
    configured = settings(monkeypatch, tmp_path)
    database = TimeoutThenSuccessDatabase()
    model = FakeModel(
        [
            IntentDecision(action="read", reason="Analytical question"),
            SqlProposal(
                intent="Count repeat customers",
                sql=(
                    "SELECT c.customer_unique_id FROM olist_customers_dataset AS c "
                    "JOIN olist_orders_dataset AS o ON o.customer_id = c.customer_id"
                ),
                explanation="Initial query.",
            ),
            SqlProposal(
                intent="Count repeat customers efficiently",
                sql=(
                    "SELECT COUNT(DISTINCT c.customer_unique_id) AS repeat_customers "
                    "FROM olist_customers_dataset AS c "
                    "JOIN olist_orders_dataset AS o ON o.customer_id = c.customer_id"
                ),
                explanation="Rewritten after timeout.",
            ),
            BusinessSummary(answer="Three repeat customers."),
        ]
    )
    agent = AnalyticsAgent(
        configured,
        database=database,
        knowledge=FakeKnowledge(),
        model=model,
        audit=AuditLogger(configured.audit_log),
    )

    response = agent.ask("How many repeat customers are there?")

    assert response.kind == "analysis"
    assert response.result.rows == [{"repeat_customers": 3}]
    assert database.read_attempts == 2
