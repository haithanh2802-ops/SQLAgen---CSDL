from __future__ import annotations

import argparse

from rich.console import Console

from text2sql_agent.config import Settings
from text2sql_agent.knowledge import index_knowledge


def main() -> None:
    parser = argparse.ArgumentParser(description="Index local Olist business documents.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the configured collection before indexing.",
    )
    parser.add_argument(
        "--include",
        action="append",
        help='File glob to index, such as "*.docx". May be supplied more than once.',
    )
    args = parser.parse_args()
    settings = Settings.from_env()
    console = Console()

    def show_progress(completed: int, total: int) -> None:
        console.print(f"Embedded {completed}/{total} chunks")

    console.print("Extracting, chunking, and embedding business documents...")
    patterns = tuple(args.include) if args.include else ("*.pdf", "*.docx")
    count = index_knowledge(
        settings,
        reset=args.reset,
        progress=show_progress,
        patterns=patterns,
    )
    console.print(f"[green]Indexed {count} document chunks.[/green]")


if __name__ == "__main__":
    main()
