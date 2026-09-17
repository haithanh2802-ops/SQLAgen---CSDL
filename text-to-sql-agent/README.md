# Olist Business Analytics Agent

A local text-to-SQL application for exploring the Brazilian Olist e-commerce
dataset. It combines MySQL for exact analytics, Ollama for local inference and
embeddings, Chroma for document retrieval, and Streamlit for an interactive UI.

The agent answers analytical questions with generated SQL, a business summary,
a result table, and an optional chart. Data-changing requests are handled
separately: the agent shows the exact SQL and predicted impact, then requires
explicit approval in Streamlit before executing it.

## Features

- Natural-language questions translated into MySQL queries
- Answers grounded in query results and local business documents
- Generated SQL, assumptions, execution time, and row counts shown in the UI
- Interactive tables, charts, and CSV downloads
- Read-only execution for analytical queries
- Review and approval workflow for `INSERT`, `UPDATE`, and `DELETE`
- Deterministic SQL validation, result limits, and affected-row limits
- Local audit records without storing complete result sets

## Architecture

```text
Question
  -> intent classification
  -> live MySQL schema + local document retrieval
  -> structured SQL proposal
  -> deterministic SQL policy validation
     -> SELECT: read-only execution -> summary -> table/chart
     -> write: impact preview -> exact approval -> revalidation -> transaction
```

CSV rows stay in MySQL. Only unstructured business material, such as the
research DOCX, analysis PDF, glossary, metric rules, and approved examples, is
embedded in Chroma.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- MySQL 8
- [Ollama](https://ollama.com/)
- Olist CSV files in the directory configured by `KNOWLEDGE_DIRECTORY`

The default local models are:

```powershell
ollama pull qwen3:8b
ollama pull qwen3-embedding:latest
```

If your Ollama installation uses different model names, change
`OLLAMA_CHAT_MODEL` and `OLLAMA_EMBEDDING_MODEL` in `.env`. The Streamlit
sidebar also lets you switch between installed chat models for the current
browser session without editing `.env`.

## Quick start

Run all commands from the `text-to-sql-agent` directory.

### 1. Install dependencies

```powershell
uv sync
Copy-Item .env.example .env
```

Edit `.env` and provide the MySQL passwords. The setup, read, and write users
serve different purposes and should remain separate.

### 2. Create the MySQL users

Run the following with a MySQL administrator, or adjust it for your local
security policy:

```sql
CREATE DATABASE IF NOT EXISTS olist
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'olist_reader'@'localhost' IDENTIFIED BY 'replace-this';
CREATE USER 'olist_editor'@'localhost' IDENTIFIED BY 'replace-this';
GRANT SELECT ON olist.* TO 'olist_reader'@'localhost';
GRANT SELECT, INSERT, UPDATE, DELETE ON olist.* TO 'olist_editor'@'localhost';
```

Set the matching values in `.env`:

```dotenv
MYSQL_READ_USER=olist_reader
MYSQL_READ_PASSWORD=replace-this
MYSQL_WRITE_USER=olist_editor
MYSQL_WRITE_PASSWORD=replace-this
MYSQL_SETUP_USER=root
MYSQL_SETUP_PASSWORD=your-root-password
```

Do not grant the application users `FILE`, `PROCESS`, `SUPER`, user
administration, global privileges, or `GRANT OPTION`.

### 3. Create the schema and load the data

Create the schema only:

```powershell
uv run python -m text2sql_agent.mysql_setup
```

Create the schema and load all nine expected Olist CSV files:

```powershell
uv run python -m text2sql_agent.mysql_setup --load
```

By default, the loader looks in `../dataset`. Change `KNOWLEDGE_DIRECTORY` in
`.env` if the CSV files are elsewhere. The required filenames are:

```text
product_category_name_translation.csv
olist_customers_dataset.csv
olist_sellers_dataset.csv
olist_products_dataset.csv
olist_orders_dataset.csv
olist_order_items_dataset.csv
olist_order_payments_dataset.csv
olist_order_reviews_dataset.csv
olist_geolocation_dataset.csv
```

`--truncate` clears the target tables before importing. Use it only when you
intend to rebuild the local database:

```powershell
uv run python -m text2sql_agent.mysql_setup --load --truncate
```

### 4. Build the local knowledge index

Make sure Ollama is running, then execute:

```powershell
uv run olist-index --reset
```

This extracts the configured PDF and DOCX files, creates deterministic chunks,
embeds them, and stores the index in `.local/chroma`. Indexing is resumable. If
it is interrupted, rerun `uv run olist-index` without `--reset`.

To index only Word documents first:

```powershell
uv run olist-index --include "*.docx"
```

### 5. Start the application

```powershell
uv run python -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`. The
sidebar should show the configured model, MySQL database name, and number of
indexed knowledge chunks.

## How to use

### Ask an analytical question

Enter a business question in the text box and select **Analyze**. Useful
examples include:

- `How does late delivery affect review scores?`
- `Show monthly revenue and order volume for 2018.`
- `Which product categories have the highest average freight cost?`
- `Compare average delivery time by customer state.`
- `What percentage of delivered orders arrived after the estimated date?`

For an analytical request, the application displays:

1. A plain-language answer and key insights.
2. The generated SQL and any assumptions.
3. Row count, execution time, and truncation status.
4. A result table, an optional chart, and a CSV download button.
5. The local business documents used as supporting context.

Questions with a clear time range, metric, and grouping usually produce the
best results. For example, prefer `Show monthly revenue in 2018 by payment
type` over `Tell me about revenue`.

### Review a data-changing request

You can also describe a specific correction, for example:

```text
Update order review 123 so its score is 4.
```

The agent does not execute the change immediately. It presents the exact SQL,
an explanation, warnings, an estimated impact, and an approval code. To proceed:

1. Read the SQL and predicted impact carefully.
2. Select the checkbox confirming that you reviewed them.
3. Type the displayed approval code exactly.
4. Select **Execute approved edit**.

The SQL is validated again immediately before execution. The transaction is
rolled back if it violates policy or exceeds `SQL_MAX_AFFECTED_ROWS`. Write
approval is available only in Streamlit; the CLI provides a preview but cannot
approve a write.

### Use the command line

For a one-off read request:

```powershell
uv run olist-agent "How does late delivery affect review scores?"
```

The backward-compatible entry point works as well:

```powershell
uv run python agent.py "Show the top five product categories by revenue"
```

The CLI prints the business insight, generated SQL, and rows as JSON. It still
requires running MySQL, Ollama, and an existing knowledge index.

## Safety boundaries

- Exactly one parsed MySQL statement per proposal
- Configured database and known tables only
- Bounded default and maximum result limits
- System schemas and cross-database access blocked
- File, sleep, benchmark, lock, and side-effecting read functions blocked
- `SELECT ... INTO OUTFILE` and locking reads blocked
- Selective `WHERE` required for `UPDATE` and `DELETE`
- Obvious tautologies such as `WHERE TRUE` and `WHERE 1=1` blocked
- DDL and privilege administration blocked
- Separate read and write database identities
- Approval code tied to the exact normalized write SQL
- Revalidation immediately before write execution
- Transaction rollback above the configured affected-row limit
- Audit records contain SQL hashes and execution metadata, not full results

## Configuration

Important `.env` options include:

| Variable | Purpose | Default |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_CHAT_MODEL` | SQL and answer model | `qwen3:8b` |
| `OLLAMA_EMBEDDING_MODEL` | Document embedding model | `qwen3-embedding:latest` |
| `MYSQL_DATABASE` | Application database | `olist` |
| `SQL_DEFAULT_LIMIT` | Added default result limit | `200` |
| `SQL_MAX_LIMIT` | Largest permitted result limit | `1000` |
| `SQL_MAX_AFFECTED_ROWS` | Maximum rows permitted in a write | `100` |
| `KNOWLEDGE_DIRECTORY` | CSV and source-document directory | `../dataset` |
| `CHROMA_DIRECTORY` | Persistent vector index | `.local/chroma` |
| `AUDIT_LOG` | Local audit event file | `.local/audit.jsonl` |

See `.env.example` for the complete list.

## Tests and quality checks

Run the deterministic unit tests:

```powershell
uv run pytest -q
```

Run lint checks:

```powershell
uv run ruff check text2sql_agent tests streamlit_app.py agent.py
```

After MySQL, Ollama, and the knowledge index are ready, run the behavioral
evaluation cases:

```powershell
uv run python -m text2sql_agent.evaluation
```

The evaluation set covers an analytical path and an approval-gated write path.
Model behavior is evaluated separately from deterministic policy tests because
LLM output is probabilistic.

## Troubleshooting

- **MySQL is unavailable:** confirm MySQL is running and verify the host, port,
  users, passwords, and database name in `.env`.
- **A model is not found:** run `ollama list`, pull the missing model, or update
  the model name in `.env`.
- **The knowledge index is empty:** start Ollama and rerun `uv run olist-index
  --reset` from the project directory.
- **A CSV file is missing:** place all nine files under `KNOWLEDGE_DIRECTORY`
  with the exact filenames listed above.
- **A write is rejected:** check that it targets an allowed table, uses a
  selective predicate, stays under the affected-row limit, and has the exact
  current approval code.

## Project layout

```text
text2sql_agent/
  audit.py          local privacy-conscious audit events
  cli.py            command-line entry point
  config.py         environment and credential boundaries
  contracts.py      structured model and application contracts
  database.py       schema inspection and guarded execution
  evaluation.py     behavioral evaluation runner
  ingestion.py      document-index command
  knowledge.py      DOCX/PDF extraction and Chroma retrieval
  mysql_setup.py    reproducible MySQL schema and CSV loader
  service.py        controlled agent orchestration
  sql_policy.py     deterministic AST policy
evals/              behavioral evaluation cases
sql/schema.sql      Olist tables, indexes, constraints, and analytical views
streamlit_app.py    Streamlit user interface
tests/              deterministic unit tests
```
