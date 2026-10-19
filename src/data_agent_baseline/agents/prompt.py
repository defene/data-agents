from __future__ import annotations

import json

from data_agent_baseline.agents.runtime import StructuredMemory
from data_agent_baseline.benchmark.schema import PublicTask


REACT_SYSTEM_PROMPT = """\
You are a data analysis agent.

Work in the shortest verifiable path to the answer.

Core operating rules:
- Take exactly ONE tool action per response.
- Return exactly one JSON object with keys: `thought`, `action`, `action_input`.
- Do NOT wrap the JSON in markdown fences or extra commentary.
- Do NOT outline multiple future actions in one response.
- Prefer direct lookup, join, or aggregation over broad exploration.
- Use documentation only when it resolves an ID, definition, label, or value that structured data alone cannot answer.
- Reuse the route hint, source index, and finish guidance in memory before deciding your next action.
- Once you have enough verified evidence, stop exploring and answer.

Data integrity rules:
- NEVER fabricate data. Every answer value must come from tool observations.
- Preserve the original value formatting unless the question explicitly asks you to transform it.
- If a document search is no longer adding new information, switch back to execute_python or execute_context_sql.
- DO NOT add filters the question did not ask for. Specifically, NEVER drop rows just because a value looks like an outlier, is zero, is negative, or is unusual. SQL `AVG`/`SUM`/`COUNT` already ignore NULLs; do not also exclude zeros or extremes unless the question explicitly says so.
- The Data Profile section in the user message is the ground truth schema and value distribution from disk. Trust it. Do not call exploratory tools (read_csv, read_json, list_context, inspect_sqlite_schema) just to re-discover what the profile already shows.
- If the task ships a knowledge.md or similar guide, READ IT before computing aggregates: it may define the exact formula to use (e.g., "Average Height = AVG(height_cm)") and overrides any heuristic you might invent.

Answering rules:
- The `answer` tool is the only terminating action.
- You MUST execute at least one successful query (`execute_python` or `execute_context_sql`) before answering.
- You MUST record at least one verified finding with `record_memory(section="evidence", ...)` before answering.
- `set_output_mapping`, `set_answer_shape`, and `set_plan` are optional helpers for complex tasks. Use them when they help you finish faster or clarify a multi-column answer.
- If the finish guidance says you already have enough evidence, prioritize answering instead of more exploration.

Answer shape rules (CRITICAL — column structure must match the source):
- The grader compares value vectors per column, not column names. Missing or merged columns score zero even if values are correct.
- PRESERVE THE SOURCE COLUMN GRANULARITY. If the source has separate `first_name` and `last_name` columns, return them as TWO columns. Do not concatenate them into a single `full_name` even if the question says "full name". Same rule for any pair of columns the source keeps separate (street/city, year/month, etc.).
- Do not invent new columns that do not exist in the source unless the question asks for a derived metric. When in doubt, mirror the source columns the data was read from.
- When answering with "the X of Y", the column you return should be the actual source field used to identify Y, not a renamed alias.
- Do not return more rows than the question implies. If the question asks for "the trans_id with property P", do not return all trans_ids — return only the rows that satisfy P.\
"""

RESPONSE_EXAMPLES = """\
Example: direct lookup from a CSV
{"thought": "The primary source is a CSV, so I should directly inspect its columns and sample values first.", "action": "read_csv", "action_input": {"path": "csv/data.csv", "max_rows": 20}}

Example: record a durable fact
{"thought": "I found the join path I will reuse, so I should store it as schema memory.", "action": "record_memory", "action_input": {"section": "schema", "fact": "orders.customer_id joins customers.id"}}

Example: resolve an ID through a document
{"thought": "I already have the event id, so I should search the document for that exact identifier instead of reading broadly.", "action": "search_doc", "action_input": {"path": "doc/reference.md", "query": "rec12345", "top_k": 3}}

Example: run the core computation
{"thought": "I have enough schema context, so I should run one precise Python query to compute the final value.", "action": "execute_python", "action_input": {"code": "import pandas as pd\\ndf = pd.read_csv(context_root / 'csv/data.csv')\\nprint(df.groupby('category')['amount'].sum())"}}

Example: optional answer shape for a complex output
{"thought": "The answer has two final columns, so locking the shape will help me finish cleanly.", "action": "set_answer_shape", "action_input": {"columns": ["name", "amount"], "expected_rows": 1, "sort_requirement": "single summary row", "value_format_rules": ["preserve source formatting"]}}

Example: submit the final answer once evidence is ready
{"thought": "I already have verified evidence for the final table, so I should answer now instead of taking another exploratory step.", "action": "answer", "action_input": {"columns": ["count"], "rows": [["1"]]}}\
"""


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return f"{base_prompt}\n\nAvailable tools:\n{tool_descriptions}\n\n{RESPONSE_EXAMPLES}"


def build_task_prompt(task: PublicTask) -> str:
    return (
        f"Question: {task.question}\n\n"
        "All tool file paths are relative to the task context directory. "
        "Work one step at a time and choose the single highest-value next action. "
        "For long documents, prefer `search_doc` or `list_doc_chunks` plus `read_doc_chunk` over broad reading. "
        "For `execute_python`, you are given `context_root` and `scratch_root` as Path objects; "
        "read source files from `context_root` and write temporary files only to `scratch_root`. "
        "Use `record_memory` only for durable findings you will reuse, especially verified evidence. "
        "When you already have enough evidence for the final table, call the `answer` tool instead of exploring further."
    )


def _render_list_section(title: str, items: list[str]) -> str:
    if not items:
        return ""
    body = "\n".join(f"- {item}" for item in items)
    return f"{title}:\n{body}"


def build_memory_prompt(memory: StructuredMemory) -> str:
    sections: list[str] = []

    if memory.route.is_set():
        sections.append(
            "Route Hint:\n"
            f"- task_type: {memory.route.task_type or '(unknown)'}\n"
            f"- solver_mode: {memory.route.solver_mode or '(unknown)'}\n"
            f"- output_hint: {memory.route.output_hint or '(unspecified)'}\n"
            f"- primary_sources: {memory.route.primary_sources}\n"
            f"- supporting_sources: {memory.route.supporting_sources}\n"
            f"- rationale: {memory.route.rationale or '(none)'}"
        )
    if memory.source_index.has_content():
        sections.append(
            "Source Index:\n"
            f"- tabular_files: {memory.source_index.tabular_files}\n"
            f"- db_files: {memory.source_index.db_files}\n"
            f"- doc_files: {memory.source_index.doc_files}\n"
            f"- other_files: {memory.source_index.other_files}"
        )
    if memory.finish.is_set():
        sections.append(
            "Finish Controller:\n"
            f"- ready_to_answer: {memory.finish.ready_to_answer}\n"
            f"- missing_requirements: {memory.finish.missing_requirements}\n"
            f"- guidance: {memory.finish.guidance or '(none)'}\n"
            f"- remaining_steps: {memory.finish.remaining_steps}"
        )
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
