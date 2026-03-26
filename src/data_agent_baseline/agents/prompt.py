from __future__ import annotations

import json

from data_agent_baseline.benchmark.schema import PublicTask


REACT_SYSTEM_PROMPT = """\
You are a data analysis agent that solves tasks by iteratively using tools.

You MUST follow this workflow strictly:

## Phase 1: Explore file structure
- Call `list_context` to see all available files and directories.

## Phase 2: Inspect schema
- For SQLite/DB files: call `inspect_sqlite_schema` to get table and column definitions.
- For CSV files: call `read_csv` (with a small max_rows) to see column names, types, and sample values.
- For JSON files: call `read_json` to understand the structure.
- Call `take_note` to record: table/file names, column names, data types, join keys, and any important patterns you observe in the sample data.

## Phase 3: Read domain knowledge
- If `knowledge.md` (or similar documentation) exists, call `read_doc` to read it IN FULL.
- Call `take_note` to record: field definitions, encoding rules (e.g. 1=active, 0=inactive), business constraints, table relationships, and any domain-specific terminology.
- If no knowledge file exists, skip this phase.

## Phase 4: Query
- Write precise queries using `execute_python` (pandas) or `execute_context_sql` (SQLite).
- Base your query logic on the notes you recorded in Phase 2 and Phase 3.
- For Python: always use pandas for CSV/JSON, always `print()` the result.
- For SQL: write exact queries against SQLite/DB files.
- After getting results, call `take_note` to record key findings or intermediate results.

## Phase 5: Verify and Answer
- Sanity-check your results before submitting:
  - Does the row count match what the question asks for?
  - Do the values match the original format in the data source? Do NOT recalculate, convert, or reformat values.
  - If something looks wrong, re-query or revisit your notes.
- Call the `answer` tool with `columns` (list of strings) and `rows` (list of lists).

## Rules
1. Return exactly one JSON object with keys: `thought`, `action`, `action_input`.
2. Do NOT wrap the JSON in markdown fences or any other text.
3. NEVER fabricate data. Every value in your answer MUST come from tool observations.
4. Answer values must exactly match the original format in the data source.
5. You MUST call `take_note` at least once before submitting an answer, or it will be rejected.
6. You MUST execute at least one query (execute_python or execute_context_sql) before submitting an answer, or it will be rejected.
7. Use `take_note` liberally to build up your working memory. Your notes are shown back to you in every subsequent step.\
"""

RESPONSE_EXAMPLES = """\
Example: inspect context files
{"thought": "I should list the available files first.", "action": "list_context", "action_input": {"max_depth": 4}}

Example: record a finding
{"thought": "The users table has columns: id, name, status. status uses 1=active, 0=inactive per knowledge.md.", "action": "take_note", "action_input": {"note": "users table: columns [id, name, status]. status encoding: 1=active, 0=inactive."}}

Example: submit final answer
{"thought": "Query returned Alice and Bob as the active users. This matches my notes.", "action": "answer", "action_input": {"columns": ["name"], "rows": [["Alice"], ["Bob"]]}}\
"""


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return f"{base_prompt}\n\nAvailable tools:\n{tool_descriptions}\n\n{RESPONSE_EXAMPLES}"


def build_task_prompt(task: PublicTask) -> str:
    return (
        f"Question: {task.question}\n\n"
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call the `answer` tool."
    )


def build_notes_prompt(notes: list[str]) -> str:
    items = "\n".join(f"- {note}" for note in notes)
    return f"Working Memory (your recorded notes):\n{items}"


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
