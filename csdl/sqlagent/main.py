import argparse
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

try:
    from sqlagent.tools import append_audit_log, execute_sql, fetch_schema, test_mysql_connection, validate_sql
except ModuleNotFoundError:
    from tools import append_audit_log, execute_sql, fetch_schema, test_mysql_connection, validate_sql


load_dotenv()
load_dotenv(Path(__file__).with_name(".env"))


class SqlProposal(BaseModel):
    intent: str = Field(description="Short description of what the user wants.")
    sql: str = Field(description="One SQL statement only.")
    explanation: str = Field(description="Plain-language explanation of the SQL and expected impact.")
    risk: str = Field(description="One of: low, medium, high.")
    rollback_sql: str | None = Field(
        default=None,
        description="A possible rollback SQL statement, or null if no safe rollback exists.",
    )


def proposal_schema() -> dict:
    if hasattr(SqlProposal, "model_json_schema"):
        return SqlProposal.model_json_schema()
    return SqlProposal.schema()


def parse_proposal(raw_json: str) -> SqlProposal:
    if hasattr(SqlProposal, "model_validate_json"):
        return SqlProposal.model_validate_json(raw_json)
    return SqlProposal.parse_raw(raw_json)


def proposal_to_dict(proposal: SqlProposal) -> dict:
    if hasattr(proposal, "model_dump"):
        return proposal.model_dump()
    return proposal.dict()


def render_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(no rows)"

    columns = list(rows[0])
    for row in rows[1:]:
        for column in row:
            if column not in columns:
                columns.append(column)

    text_rows = [{column: str(row.get(column, "")) for column in columns} for row in rows]
    widths = {
        column: max(len(column), *(len(row[column]) for row in text_rows))
        for column in columns
    }
    separator = "+-" + "-+-".join("-" * widths[column] for column in columns) + "-+"
    header = "| " + " | ".join(column.ljust(widths[column]) for column in columns) + " |"
    body = [
        "| " + " | ".join(row[column].ljust(widths[column]) for column in columns) + " |"
        for row in text_rows
    ]
    return "\n".join([separator, header, separator, *body, separator])


def result_to_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    if "rows" in result:
        return result["rows"]
    return [{key: value for key, value in result.items() if key != "rows"}]


def build_gemini_client():
    try:
        from google import genai
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install dependencies first: pip install -r sqlagent\\requirements.txt") from exc

    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file.")

    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


def request_sql_proposal(user_query: str, schema: str) -> SqlProposal:
    client = build_gemini_client()
    model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

    system_prompt = f"""
You are a local MySQL SQL creation and modification agent.

Your job is to propose exactly one MySQL SQL statement for the user's request.
You may create, alter, insert, update, delete, or query data, but you do not execute anything.

Rules:
- Use only the schema shown below unless the user asks to create new schema.
- Prefer safe, explicit SQL.
- UPDATE and DELETE must include a WHERE clause.
- SELECT must include a LIMIT clause.
- Do not propose DROP or TRUNCATE unless the user explicitly asks for it.
- Do not target system databases: mysql, information_schema, performance_schema, sys.
- Include a rollback statement when one is realistic. Use null when rollback is not safe or not possible.

Current database schema:
{schema}
""".strip()

    try:
        from google.genai import types
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install dependencies first: pip install -r sqlagent\\requirements.txt") from exc

    response = client.models.generate_content(
        model=model,
        contents=user_query,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0,
            response_mime_type="application/json",
            response_json_schema=proposal_schema(),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    content = response.text
    if not content:
        raise RuntimeError("The model returned an empty response.")
    return parse_proposal(content)


def approval_phrase(operation: str) -> str:
    return f"APPROVE {operation}"


def main() -> None:
    cli = argparse.ArgumentParser(description="Local Gemini-powered MySQL SQL agent.")
    cli.add_argument("--test-db", action="store_true", help="Test local MySQL read/write connections and exit.")
    args = cli.parse_args()

    if args.test_db:
        try:
            print("Read connection")
            print(test_mysql_connection(read_only=True))
            print("\nWrite connection")
            print(test_mysql_connection(read_only=False))
        except Exception as exc:
            print(f"Database connection test failed: {exc}")
            raise SystemExit(1) from exc
        return

    try:
        schema = fetch_schema()
    except Exception as exc:
        print(f"Could not read MySQL schema: {exc}")
        raise SystemExit(1) from exc

    query = input("What SQL change or query do you want? ").strip()
    try:
        proposal = request_sql_proposal(query, schema)
    except Exception as exc:
        print(f"Could not get SQL proposal from Gemini: {exc}")
        raise SystemExit(1) from exc

    validation = validate_sql(proposal.sql)

    print("\nProposed SQL")
    print(proposal.sql)
    print("\nExplanation")
    print(proposal.explanation)
    print(f"\nOperation: {validation.operation}")
    print(f"Risk: {validation.risk}")

    if proposal.rollback_sql:
        print("\nPossible rollback SQL")
        print(proposal.rollback_sql)

    if validation.reasons:
        print("\nValidation failed")
        for reason in validation.reasons:
            print(f"- {reason}")
        append_audit_log(
            {
                "user_query": query,
                "proposal": proposal_to_dict(proposal),
                "validation": validation.__dict__,
                "decision": "blocked",
            }
        )
        return

    if validation.requires_approval:
        phrase = approval_phrase(validation.operation)
        print("\nHuman approval required.")
        print(f'Type "{phrase}" to execute, or anything else to cancel.')
        decision = input("> ").strip()
        if decision != phrase:
            append_audit_log(
                {
                    "user_query": query,
                    "proposal": proposal_to_dict(proposal),
                    "validation": validation.__dict__,
                    "decision": "rejected",
                }
            )
            print("Cancelled. No SQL was executed.")
            return

    result = execute_sql(proposal.sql, read_only=not validation.requires_approval)
    append_audit_log(
        {
            "user_query": query,
            "proposal": proposal_to_dict(proposal),
            "validation": validation.__dict__,
            "decision": "approved" if validation.requires_approval else "auto_read",
            "execution_result": result,
        }
    )

    print("\nExecution result")
    print(render_table(result_to_rows(result)))


if __name__ == "__main__":
    main()
