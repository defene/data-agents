from __future__ import annotations

import re

from data_agent_baseline.agents.policies.finish import DEFAULT_FINISH_POLICY, FinishPolicy
from data_agent_baseline.agents.router import build_source_index, infer_task_route
from data_agent_baseline.agents.runtime import AgentRuntimeState, OutputMappingItem
from data_agent_baseline.benchmark.schema import PublicTask

QUERY_ACTIONS = frozenset({"execute_python", "execute_context_sql"})
NAN_LIKE_TOKENS = frozenset({"nan", "none", "null", "n/a", "na"})


def _has_successful_query(state: AgentRuntimeState) -> bool:
    return any(step.action in QUERY_ACTIONS and step.ok for step in state.steps)


def _mapped_output_columns(state: AgentRuntimeState) -> list[str]:
    columns: list[str] = []
    for item in state.memory.output_mapping:
        for column in item.output_columns:
            if column not in columns:
                columns.append(column)
    return columns


def _is_ratio_or_percentage_question(question: str) -> bool:
    lowered = re.sub(r"\s+", " ", question.strip().lower())
    markers = ("percentage", "percent", "ratio", "proportion", "share", "how many times")
    return any(marker in lowered for marker in markers)


def _single_non_empty_cell(rows: list[object]) -> str | None:
    if len(rows) != 1 or not isinstance(rows[0], list):
        return None
    non_empty = [str(cell).strip() for cell in rows[0] if str(cell).strip()]
    if len(non_empty) != 1:
        return None
    return non_empty[0]


def _extract_ratio_numeric_value(value: str) -> str | None:
    match = re.fullmatch(
        r"\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*%?\s*(?:\([^)]*\))?\s*",
        value,
    )
    if match is None:
        return None
    return match.group(1)


def bootstrap_memory_from_context(
    state: AgentRuntimeState,
    *,
    question: str,
    context_listing: dict[str, object],
) -> None:
    source_index = build_source_index(context_listing)
    state.memory.source_index = source_index
    state.memory.route = infer_task_route(question, source_index)


def update_finish_memory(
    state: AgentRuntimeState,
    *,
    question: str,
    remaining_steps: int | None = None,
    finish_policy: FinishPolicy | None = None,
) -> None:
    state.memory.finish = (finish_policy or DEFAULT_FINISH_POLICY).build_finish_memory(
        state,
        question=question,
        remaining_steps=remaining_steps,
    )


def normalize_answer_submission(
    task: PublicTask,
    state: AgentRuntimeState,
    action_input: dict[str, object],
) -> dict[str, object]:
    raw_columns = action_input.get("columns")
    raw_rows = action_input.get("rows")
    if not isinstance(raw_columns, list):
        return action_input

    normalized_columns = [
        item.strip() for item in raw_columns if isinstance(item, str) and item.strip()
    ]
    if not normalized_columns:
        return action_input

    normalized_action_input = dict(action_input)
    normalized_action_input["columns"] = normalized_columns
    if isinstance(raw_rows, list):
        normalized_action_input["rows"] = raw_rows

    if _is_ratio_or_percentage_question(task.question) and isinstance(raw_rows, list):
        scalar_value = _single_non_empty_cell(raw_rows)
        normalized_value = (
            _extract_ratio_numeric_value(scalar_value) if scalar_value is not None else None
        )
        if normalized_value is not None:
            normalized_action_input["rows"] = [[normalized_value]]

    if not state.memory.answer_shape.columns:
        state.memory.answer_shape.columns = list(normalized_columns)
        if isinstance(normalized_action_input.get("rows"), list):
            state.memory.answer_shape.expected_rows = len(normalized_action_input["rows"])
        state.memory.answer_shape.sort_requirement = (
            "single summary row"
            if isinstance(normalized_action_input.get("rows"), list)
            and len(normalized_action_input["rows"]) <= 1
            else ""
        )
        state.memory.answer_shape.value_format_rules = ["preserve source formatting"]

    if not state.memory.output_mapping:
        state.memory.output_mapping = [
            OutputMappingItem(
                question_target=task.question.strip(),
                source_kind="derived_expression",
                source_refs=list(state.memory.evidence[:3]) or list(state.memory.route.primary_sources),
                output_columns=list(normalized_columns),
                granularity=(
                    "aggregate"
                    if isinstance(normalized_action_input.get("rows"), list)
                    and len(normalized_action_input["rows"]) <= 1
                    else "detail_rows"
                ),
                format_rule="preserve source formatting",
            )
        ]

    if not state.memory.plan.is_set():
        state.memory.plan.done = ["collected verified evidence", "prepared final table"]
        state.memory.plan.todo = ["submit answer"]
        state.memory.plan.next_action = "submit answer"

    return normalized_action_input


def validate_answer_submission(
    task: PublicTask,
    state: AgentRuntimeState,
    action_input: dict[str, object],
) -> str | None:
    if not _has_successful_query(state):
        return (
            "REJECTED: You must execute at least one successful query "
            "(execute_python or execute_context_sql) before submitting an answer."
        )
    if not state.memory.evidence:
        return "REJECTED: Record at least one verified evidence fact before submitting an answer."

    submitted_columns = action_input.get("columns")
    submitted_rows = action_input.get("rows")
    if not isinstance(submitted_columns, list) or not submitted_columns:
        return "REJECTED: answer.columns must be a non-empty list of strings."
    if not isinstance(submitted_rows, list):
        return "REJECTED: answer.rows must be a list."
    if not submitted_rows:
        return (
            "REJECTED: Do not submit an empty answer table. If the answer is a numeric zero, "
            "return one row with 0; otherwise re-check your joins, mappings, or filters."
        )

    flat_cells = [str(cell).strip() for row in submitted_rows if isinstance(row, list) for cell in row]
    flat_cells_lower = [cell.lower() for cell in flat_cells]
    if len(submitted_rows) <= 1 and any(cell in NAN_LIKE_TOKENS for cell in flat_cells_lower):
        return (
            "REJECTED: Do not submit NaN/None-like final values. Resolve the computation to a "
            "concrete numeric or text answer first."
        )
    if _is_ratio_or_percentage_question(task.question) and any("%" in cell for cell in flat_cells):
        return (
            "REJECTED: For ratio or percentage questions, submit the final numeric value without a "
            "percent sign."
        )
    if _is_ratio_or_percentage_question(task.question):
        scalar_value = _single_non_empty_cell(submitted_rows)
        if scalar_value is not None:
            normalized_value = _extract_ratio_numeric_value(scalar_value)
            if normalized_value is None and any(char.isalpha() for char in scalar_value):
                return (
                    "REJECTED: For ratio or percentage questions, return one raw numeric value only, "
                    "with no explanatory text."
                )

    if state.memory.answer_shape.columns and submitted_columns != state.memory.answer_shape.columns:
        return (
            "REJECTED: answer.columns must exactly match the final answer shape "
            "once the answer shape has been set."
        )

    if state.memory.output_mapping:
        mapped_columns = _mapped_output_columns(state)
        if mapped_columns and submitted_columns != mapped_columns:
            return (
                "REJECTED: answer.columns must match the output mapping once an output mapping "
                "has been set."
            )
    return None
