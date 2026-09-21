from types import SimpleNamespace

from text2sql_agent.contracts import AgentResponse, BusinessSummary, QueryResult, SqlProposal
from text2sql_agent.evaluation import run_case


class FakeDatabase:
    def schema_catalog(self):
        return {
            "olist_orders_dataset": [
                {"column_name": "order_id"},
                {"column_name": "order_status"},
            ]
        }


class FakeAgent:
    database = FakeDatabase()
    settings = SimpleNamespace(
        mysql_database="olist",
        mysql_writable_tables=frozenset({"olist_orders_dataset"}),
        sql_default_limit=200,
        sql_max_limit=1000,
    )

    def ask(self, prompt):
        return AgentResponse(
            kind="analysis",
            question=prompt,
            proposal=SqlProposal(
                intent="Count orders",
                sql="SELECT o.order_status, COUNT(*) AS order_count "
                "FROM olist_orders_dataset AS o GROUP BY o.order_status LIMIT 200",
                explanation="Groups orders by status.",
            ),
            summary=BusinessSummary(answer="Counts are shown in the result."),
            result=QueryResult(
                columns=["order_status", "order_count"],
                rows=[{"order_status": "delivered", "order_count": 1}],
                row_count=1,
                execution_ms=1.0,
            ),
        )


def test_run_case_records_task_success_evidence():
    case = {
        "case_id": "order_status_counts",
        "prompt": "Count orders by status",
        "expected_kind": "analysis",
        "expected_mode": "read",
        "expected_table_groups": [["olist_orders_dataset"]],
        "required_sql_fragments": ["order_status"],
        "expected_result_column_groups": [["order_status", "order_count"]],
    }

    result = run_case(FakeAgent(), case)

    assert result["passed"]
    assert result["checks"]["query_executed"]
    assert result["checks"]["summary_present"]
    assert result["proposal_sql"].startswith("SELECT")
    assert result["result_columns"] == ["order_status", "order_count"]
    assert result["result_row_count"] == 1
    assert result["duration_ms"] >= 0
