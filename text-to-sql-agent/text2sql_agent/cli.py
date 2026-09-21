from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel

from text2sql_agent.config import Settings
from text2sql_agent.service import AnalyticsAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Olist business analytics agent.")
    parser.add_argument("question", help="Business question or explicit data-edit request.")
    args = parser.parse_args()

    console = Console()
    response = AnalyticsAgent(Settings.from_env()).ask(args.question)
    if response.kind == "analysis" and response.summary and response.result:
        console.print(
            Panel(response.summary.answer, title="Business insight", border_style="green")
        )
        for insight in response.summary.insights:
            console.print(f"- {insight}")
        console.print("\n[bold]Generated SQL[/bold]")
        console.print(response.proposal.sql if response.proposal else "")
        console.print_json(data=response.result.rows)
    elif response.kind == "write_preview" and response.write_preview:
        console.print(Panel(response.write_preview.explanation, title="Write proposal"))
        console.print(response.write_preview.sql)
        console.print(f"\nApproval code: [bold]{response.write_preview.approval_code}[/bold]")
        console.print("Use the Streamlit application to review and approve edits.")
    elif response.kind == "memory" and response.summary:
        console.print(Panel(response.summary.answer, title="Memory", border_style="blue"))
        for memory in response.retrieved_memories:
            console.print(f"- {memory}")
    else:
        console.print(Panel(response.error or "No response available.", border_style="red"))


if __name__ == "__main__":
    main()
