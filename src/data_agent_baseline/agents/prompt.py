from __future__ import annotations

import json

from data_agent_baseline.agents.runtime import StructuredMemory
from data_agent_baseline.benchmark.schema import PublicTask


REACT_SYSTEM_PROMPT = """\
You are a data analysis agent that solves tasks by iteratively using tools.

You MUST follow this workflow strictly:

## Phase 1: Explore file structure
- Call `list_context` to see all available files and directories.
- Record important files, tables, or join paths with `record_memory(section="schema", ...)`.

## Phase 2: Inspect schema
- For SQLite/DB files: call `inspect_sqlite_schema` to get table and column definitions.
- For CSV files: call `read_csv` (with a small max_rows) to see column names, types, and sample values.
- For JSON files: call `read_json` to understand the structure.
- Record: table/file names, column names, data types, join keys, and important patterns via `record_memory(section="schema", ...)`.

## Phase 3: Read domain knowledge
- If `knowledge.md` (or similar documentation) exists:
  - For short documents, call `read_doc`.
  - For long documents, first call `list_doc_chunks` or `search_doc`, then call `read_doc_chunk` on the most relevant chunks.
- Record: field definitions, encoding rules, business constraints, table relationships, and domain terminology via `record_memory(section="rules", ...)`.
- As soon as you identify the target ID, name, date window, or entity, record it via `record_memory(section="entities", ...)`.
- If no knowledge file exists, skip this phase.

## Phase 3.5: Map question targets to output
- Before finalizing the answer format, call `set_output_mapping`.
- Each mapping should connect the question target to:
  - the source fields or derived expression you will use,
  - the final output column names,
  - whether the answer is detail rows or an aggregate,
  - any important formatting rule.

## Phase 4: Query
- Write precise queries using `execute_python` (pandas) or `execute_context_sql` (SQLite).
- Base your query logic on your structured memory.
- For Python: always use pandas for CSV/JSON, always `print()` the result.
- Python code runs inside an isolated scratch directory, not inside the task context directory.
- In Python, read original files via `Path(context_root) / "<relative_path>"`.
- If you need temporary outputs, write them only under `Path(scratch_root)`.
- For SQL: write exact queries against SQLite/DB files.
- After getting results, call `record_memory(section="evidence", ...)` to record verified findings.
- Update `set_plan` after each major discovery so your todo list stays current.
- Once your output mapping is clear, call `set_answer_shape`.

## Phase 5: Verify and Answer
- Sanity-check your results before submitting:
  - Does the row count match what the question asks for?
  - Do the values match the original format in the data source? Do NOT recalculate, convert, or reformat values.
  - If something looks wrong, re-query or revisit your structured memory.
- Make sure `set_output_mapping`, `set_answer_shape`, and the final `answer` all describe the same output columns.
- Before answering, `set_plan` should show no unfinished work except possibly "submit answer".
- Call the `answer` tool with `columns` (list of strings) and `rows` (list of lists).

## Rules
1. Return exactly one JSON object with keys: `thought`, `action`, `action_input`.
2. Do NOT wrap the JSON in markdown fences or any other text.
3. NEVER fabricate data. Every value in your answer MUST come from tool observations.
4. Answer values must exactly match the original format in the data source.
5. You MUST call `set_output_mapping` before submitting an answer, or it will be rejected.
6. You MUST execute at least one query (execute_python or execute_context_sql) before submitting an answer, or it will be rejected.
7. You MUST record at least one verified result with `record_memory(section="evidence", ...)` before submitting an answer.
8. You MUST call `set_answer_shape` before submitting an answer, or it will be rejected.
9. You MUST call `set_plan` before submitting an answer, and the remaining todo items must be empty except for "submit answer".
10. Do not keep searching the same document once you already have the rule, threshold, or entity you need. Switch to SQL or Python and finish the computation.\
"""

RESPONSE_EXAMPLES = """\
Example: inspect context files
{"thought": "I should list the available files first.", "action": "list_context", "action_input": {"max_depth": 4}}

Example: record schema memory
{"thought": "I found the main join path, so I should store it for later queries.", "action": "record_memory", "action_input": {"section": "schema", "fact": "orders.customer_id joins customers.id"}}

Example: record an entity
{"thought": "The document identified the target entity, so I should store it and stop re-searching for it.", "action": "record_memory", "action_input": {"section": "entities", "fact": "target race_id=18"}}

Example: search a long document
{"thought": "This reference document is long, so I should search for the section that defines the target field before reading chunks.", "action": "search_doc", "action_input": {"path": "doc/reference.md", "query": "field definition status rule", "top_k": 3}}

Example: read one document chunk
{"thought": "This chunk looks relevant because it mentions the definition I need.", "action": "read_doc_chunk", "action_input": {"path": "doc/reference.md", "chunk_id": "chunk_0003"}}

Example: set output mapping
{"thought": "I know what the question is asking for and which source field it should come from, so I should map the target to the final output columns before locking the answer shape.", "action": "set_output_mapping", "action_input": {"mappings": [{"question_target": "driver surname", "source_kind": "raw_column", "source_refs": ["results.surname"], "output_columns": ["surname"], "granularity": "detail_rows", "format_rule": "preserve source formatting"}]}}

Example: set answer shape
{"thought": "The question asks for one row with two columns, so I should lock in the final answer format now.", "action": "set_answer_shape", "action_input": {"columns": ["name", "percentage"], "expected_rows": 1, "sort_requirement": "single summary row", "value_format_rules": ["preserve source formatting"]}}

Example: set plan
{"thought": "I know what is done and what remains, so I should update the plan before running the final query.", "action": "set_plan", "action_input": {"done": ["identified target entity"], "todo": ["run final aggregation", "submit answer"], "next_action": "query the results table for the final metric"}}

Example: submit final answer
{"thought": "The final query result matches my evidence and the answer shape, so I can submit the answer.", "action": "answer", "action_input": {"columns": ["name"], "rows": [["Alice"], ["Bob"]]}}\
"""


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return f"{base_prompt}\n\nAvailable tools:\n{tool_descriptions}\n\n{RESPONSE_EXAMPLES}"


def build_task_prompt(task: PublicTask) -> str:
    return (
        f"Question: {task.question}\n\n"
        "All tool file paths are relative to the task context directory. "
        "For long documents, prefer `list_doc_chunks` or `search_doc` plus `read_doc_chunk` instead of relying only on `read_doc`. "
        "For `execute_python`, you are given `context_root` and `scratch_root` as Path objects; "
        "read source files from `context_root` and write temporary files only to `scratch_root`. "
        "Use `record_memory` to store schema, rules, entities, and evidence. "
        "Use `set_output_mapping` to connect question targets to source fields or derived expressions before finalizing output columns. "
        "Use `set_answer_shape` once you know the target output format. "
        "Use `set_plan` to keep only the remaining work visible. "
        "When you have the final table, call the `answer` tool."
    )


def _render_list_section(title: str, items: list[str]) -> str:
    if not items:
        return ""
    body = "\n".join(f"- {item}" for item in items)
    return f"{title}:\n{body}"


def build_memory_prompt(memory: StructuredMemory) -> str:
    sections: list[str] = []

    if memory.answer_shape.is_set():
        sections.append(
            "Answer Shape:\n"
            f"- columns: {memory.answer_shape.columns}\n"
            f"- expected_rows: {memory.answer_shape.expected_rows}\n"
            f"- sort_requirement: {memory.answer_shape.sort_requirement or '(none)'}\n"
            f"- value_format_rules: {memory.answer_shape.value_format_rules}"
        )
    if memory.output_mapping:
        mapping_lines = []
        for item in memory.output_mapping:
            mapping_lines.append(
                "- "
                f"target={item.question_target}; "
                f"source_kind={item.source_kind}; "
                f"source_refs={item.source_refs}; "
                f"output_columns={item.output_columns}; "
                f"granularity={item.granularity or '(unspecified)'}; "
                f"format_rule={item.format_rule or '(none)'}"
            )
        sections.append("Output Mapping:\n" + "\n".join(mapping_lines))
    if memory.plan.is_set():
        sections.append(
            "Current Plan:\n"
            f"- done: {memory.plan.done}\n"
            f"- todo: {memory.plan.todo}\n"
            f"- next_action: {memory.plan.next_action or '(not set)'}"
        )

    for title, items in (
        ("Known Entities", memory.entities),
        ("Known Rules", memory.rules),
        ("Known Schema", memory.schema),
        ("Verified Evidence", memory.evidence),
    ):
        rendered = _render_list_section(title, items)
        if rendered:
            sections.append(rendered)

    return "Structured Memory:\n" + "\n\n".join(sections)


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
