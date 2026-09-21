from dataclasses import replace
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
from text2sql_agent.memory import MemoryContext, MemoryRecord
from text2sql_agent.service import (
    AnalyticsAgent,
    _analysis_pattern_guidance,
    _friendly_model_error,
)

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


class FakeMemoryManager:
    def __init__(self, context: MemoryContext | None = None):
        self.context = context or MemoryContext(text="")
        self.considered = []

    def retrieve(self, _query, **_scope):
        return self.context

    def consider_turn(self, **turn):
        self.considered.append(turn)
        return ["Remembered user.analysis_style."]


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


class CollationThenSuccessDatabase(FakeDatabase):
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
                Exception(1267, "Illegal mix of collations for operation '='"),
            )
        return QueryResult(
            columns=["customer_state", "late_rate"],
            rows=[{"customer_state": "SP", "late_rate": 0.12}],
            row_count=1,
            execution_ms=5.0,
        )


def settings(monkeypatch, tmp_path: Path) -> Settings:
    monkeypatch.setenv("MYSQL_READ_USER", "reader")
    monkeypatch.setenv("MYSQL_WRITE_USER", "editor")
    monkeypatch.setenv("AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    return Settings.from_env(tmp_path)


def test_moe_backend_crash_has_actionable_error():
    error = RuntimeError("GGML_ASSERT(id >= 0 && id < n_expert) failed")
    message = _friendly_model_error(error)
    assert "OLLAMA_NUM_GPU=0" in message
    assert "qwen3:8b" in message


def test_explicit_memory_request_bypasses_database_classification(monkeypatch, tmp_path):
    configured = settings(monkeypatch, tmp_path)
    memory = FakeMemoryManager()
    agent = AnalyticsAgent(
        configured,
        database=FakeDatabase(),
        knowledge=FakeKnowledge(),
        model=FakeModel([]),
        audit=AuditLogger(configured.audit_log),
        memory=memory,
    )

    response = agent.ask("Remember that I prefer concise business summaries.")

    assert response.kind == "memory"
    assert response.memory_actions == ["Remembered user.analysis_style."]
    assert response.summary.answer == "Remembered user.analysis_style."
    assert memory.considered[0]["response_kind"] == "memory"


def test_memory_inspection_returns_retrieved_items_without_writing(monkeypatch, tmp_path):
    configured = settings(monkeypatch, tmp_path)
    record = MemoryRecord(
        memory_id="memory-1",
        score=0.9,
        payload={"text": "The user prefers concise business summaries."},
    )
    memory = FakeMemoryManager(
        MemoryContext(
            text='<memory id="memory-1">concise summaries</memory>',
            records=(record,),
        )
    )
    agent = AnalyticsAgent(
        configured,
        database=FakeDatabase(),
        knowledge=FakeKnowledge(),
        model=FakeModel([]),
        audit=AuditLogger(configured.audit_log),
        memory=memory,
    )

    response = agent.ask("What do you remember about my preferences?")

    assert response.kind == "memory"
    assert response.retrieved_memories == ["The user prefers concise business summaries."]
    assert memory.considered == []


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


def test_write_preview_degrades_safely_when_identities_match(monkeypatch, tmp_path):
    configured = settings(monkeypatch, tmp_path)
    configured = replace(configured, mysql_write_user=configured.mysql_read_user)
    database = FakeDatabase()
    model = FakeModel(
        [
            IntentDecision(action="write", reason="Explicit correction"),
            SqlProposal(
                intent="Correct one order",
                sql="UPDATE orders SET order_status = 'canceled' WHERE order_id = 'abc'",
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

    response = agent.ask("Change order abc to canceled")

    assert response.kind == "write_preview"
    assert response.write_preview.estimated_rows is None
    assert not response.write_preview.execution_available
    assert any("execution is unavailable" in warning for warning in response.write_preview.warnings)
    assert database.executed_write is None


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


def test_read_validation_can_repair_two_consecutive_column_errors(monkeypatch, tmp_path):
    configured = settings(monkeypatch, tmp_path)
    progress_steps = []
    model = FakeModel(
        [
            IntentDecision(action="read", reason="Analytical question"),
            SqlProposal(
                intent="First invalid analysis",
                sql="SELECT avg_customer_payment FROM orders AS o",
                explanation="First attempt.",
            ),
            SqlProposal(
                intent="Second invalid analysis",
                sql="SELECT customer_id FROM orders AS o",
                explanation="First repair.",
            ),
            SqlProposal(
                intent="Valid analysis",
                sql=(
                    "SELECT o.order_status, COUNT(*) AS total "
                    "FROM orders AS o GROUP BY o.order_status"
                ),
                explanation="Final repair.",
            ),
            BusinessSummary(answer="Ten delivered orders."),
        ]
    )
    agent = AnalyticsAgent(
        configured,
        database=FakeDatabase(),
        knowledge=FakeKnowledge(),
        model=model,
        audit=AuditLogger(configured.audit_log),
    )

    response = agent.ask("Compare payment behavior", progress=progress_steps.append)

    assert response.kind == "analysis"
    assert "Correcting SQL references (attempt 1 of 2)" in progress_steps
    assert "Correcting SQL references (attempt 2 of 2)" in progress_steps


def test_seller_concentration_question_gets_a_schema_valid_pattern():
    guidance = _analysis_pattern_guidance(
        "For each product category, find sellers responsible for 80% of revenue.",
        {
            "olist_order_items_dataset": [],
            "olist_products_dataset": [],
            "product_category_name_translation": [],
        },
    )

    assert "FROM olist_order_items_dataset AS oi" in guidance
    assert "JOIN olist_products_dataset AS p" in guidance
    assert "MIN(rs.seller_rank) AS sellers_count" in guidance
    assert "do not invent `df_sales`" in guidance


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


def test_collation_mismatch_is_repaired_and_retried_once(monkeypatch, tmp_path):
    configured = settings(monkeypatch, tmp_path)
    database = CollationThenSuccessDatabase()
    model = FakeModel(
        [
            IntentDecision(action="read", reason="Analytical question"),
            SqlProposal(
                intent="Compare states",
                sql=(
                    "SELECT c.customer_unique_id FROM olist_customers_dataset AS c "
                    "JOIN olist_orders_dataset AS o ON o.customer_id = c.customer_id"
                ),
                explanation="Initial query.",
            ),
            SqlProposal(
                intent="Compare states with compatible collations",
                sql=(
                    "SELECT c.customer_unique_id FROM olist_customers_dataset AS c "
                    "JOIN olist_orders_dataset AS o "
                    "ON o.customer_id COLLATE utf8mb4_unicode_ci = "
                    "c.customer_id COLLATE utf8mb4_unicode_ci"
                ),
                explanation="Uses one collation for the text join.",
            ),
            BusinessSummary(answer="SP has a 12% late-delivery rate."),
        ]
    )
    agent = AnalyticsAgent(
        configured,
        database=database,
        knowledge=FakeKnowledge(),
        model=model,
        audit=AuditLogger(configured.audit_log),
    )

    response = agent.ask("Which state has the highest late-delivery rate?")

    assert response.kind == "analysis"
    assert database.read_attempts == 2
    assert "COLLATE utf8mb4_unicode_ci" in response.proposal.sql
