from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from typing import Any

from rich.console import Console
from rich.table import Table

from text2sql_agent.config import Settings
from text2sql_agent.service import AnalyticsAgent
from text2sql_agent.sql_policy import validate_sql


def run_case(agent: AnalyticsAgent, case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    response = agent.ask(case["prompt"])
    checks = {"response_kind": response.kind == case["expected_kind"]}

    if response.proposal:
        mode = case["expected_mode"]
        catalog = agent.database.schema_catalog()
        allowed = (
            set(catalog)
            if mode == "read"
            else set(catalog).intersection(agent.settings.mysql_writable_tables)
        )
        policy = validate_sql(
            response.proposal.sql,
            mode=mode,
            database=agent.settings.mysql_database,
            allowed_tables=allowed,
            default_limit=agent.settings.sql_default_limit,
            max_limit=agent.settings.sql_max_limit,
        )
        checks["policy_allowed"] = policy.allowed
        actual_tables = {table.lower() for table in policy.tables}
        groups = [set(group) for group in case.get("expected_table_groups", [])]
        checks["expected_tables"] = not groups or any(group <= actual_tables for group in groups)
        required_fragments = [fragment.lower() for fragment in case.get("required_sql_fragments", [])]
        checks["required_sql_fragments"] = all(
            fragment in response.proposal.sql.lower() for fragment in required_fragments
        )
    else:
        checks["policy_allowed"] = False
        checks["expected_tables"] = False
        checks["required_sql_fragments"] = not case.get("required_sql_fragments")

    if case["expected_kind"] == "analysis":
        checks["query_executed"] = response.result is not None
        checks["summary_present"] = response.summary is not None
        expected_column_groups = [
            {column.lower() for column in group}
            for group in case.get("expected_result_column_groups", [])
        ]
        actual_columns = (
            {column.lower() for column in response.result.columns} if response.result else set()
        )
        checks["expected_result_columns"] = not expected_column_groups or any(
            group <= actual_columns for group in expected_column_groups
        )

    if case["expected_kind"] == "write_preview":
        checks["approval_present"] = bool(
            response.write_preview and response.write_preview.approval_code.startswith("APPROVE ")
        )

    return {
        "case_id": case["case_id"],
        "passed": all(checks.values()),
        "checks": checks,
        "response_kind": response.kind,
        "error": response.error,
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        "proposal_sql": response.proposal.sql if response.proposal else None,
        "result_columns": response.result.columns if response.result else [],
        "result_row_count": response.result.row_count if response.result else None,
        "retrieved_sources": response.retrieved_sources,
        "rubric_for_human_review": case.get("rubric", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Olist agent behavioral cases.")
    parser.add_argument("--dataset", default="evals/core_cases.json")
    parser.add_argument("--output", default="artifacts/eval_results")
    args = parser.parse_args()

    settings = Settings.from_env()
    dataset_path = settings.project_root / args.dataset
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))["cases"]
    agent = AnalyticsAgent(settings)
    console = Console()
    results = []
    for index, case in enumerate(cases, start=1):
        console.print(f"Running {index}/{len(cases)}: {case['case_id']}")
        result = run_case(agent, case)
        results.append(result)
        console.print(
            f"Completed {case['case_id']} in {result['duration_ms'] / 1000:.1f}s "
            f"({'passed' if result['passed'] else 'failed'})"
        )

    table = Table("Case", "Passed", "Checks")
    for result in results:
        table.add_row(
            result["case_id"],
            "yes" if result["passed"] else "no",
            json.dumps(result["checks"], sort_keys=True),
        )
    console.print(table)

    output_directory = settings.project_root / args.output
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = output_directory / f"results_{timestamp}.json"
    output_path.write_text(
        json.dumps({"results": results}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    console.print(f"Results: {output_path}")
    if not all(result["passed"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
