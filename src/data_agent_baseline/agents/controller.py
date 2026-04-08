from __future__ import annotations

import re

from data_agent_baseline.agents.router import build_source_index, infer_task_route
from data_agent_baseline.agents.runtime import AgentRuntimeState, FinishMemory, OutputMappingItem
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


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def _is_row_extraction_question(question: str) -> bool:
    lowered = _normalize_question(question)
    starters = (
        "list ",
        "show ",
        "which ",
        "what are ",
        "identify ",
        "write ",
        "name ",
        "return ",
    )
    return lowered.startswith(starters) or " full name " in f" {lowered} "


def _is_ratio_or_percentage_question(question: str) -> bool:
    lowered = _normalize_question(question)
    markers = ("percentage", "percent", "ratio", "proportion", "share", "how many times")
    return any(marker in lowered for marker in markers)


def _is_average_question(question: str) -> bool:
    lowered = _normalize_question(question)
    return any(marker in lowered for marker in ("average", "avg", "mean"))


def _evidence_text(state: AgentRuntimeState) -> str:
    return " ".join(state.memory.evidence).strip().lower()


def _has_nan_like_text(text: str) -> bool:
    return any(token in text for token in NAN_LIKE_TOKENS)


def _has_zero_result_text(text: str) -> bool:
    markers = (
        "no rows",
        "no match",
        "no matches",
        "empty",
        "[]",
        "count=0",
        "count: 0",
        "zero ",
        "=0",
    )
    return any(marker in text for marker in markers)


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


def _route_guidance(state: AgentRuntimeState) -> str:
    route = state.memory.route
    if route.solver_mode == "doc_link":
        return (
            "Use structured data to resolve target rows or IDs first, then read only the document "
            "chunks needed to extract the linked final values."
        )
    if route.solver_mode == "sql_first":
        return "Prefer one precise SQL query over repeated previews."
    if route.solver_mode == "tabular_first":
        return "Stay in the structured files unless a document is clearly required."
    if route.task_type == "compare":
        return "Resolve both sides of the comparison, extract the final values, then answer."
    if route.task_type == "aggregate":
        return "Filter once, aggregate once, record evidence, then answer."
    if route.task_type == "lookup":
        return "Find the target record directly, resolve any linked label, then answer."
    return "Take the shortest verifiable path to the answer."


def update_finish_memory(
    state: AgentRuntimeState,
    *,
    question: str,
    remaining_steps: int | None = None,
) -> None:
    has_query = _has_successful_query(state)
    has_evidence = bool(state.memory.evidence)
    evidence_text = _evidence_text(state)
    contract_misaligned = bool(
        state.memory.output_mapping
        and state.memory.answer_shape.columns
        and _mapped_output_columns(state) != state.memory.answer_shape.columns
    )

    missing_requirements: list[str] = []
    if not has_query:
        missing_requirements.append("one successful query")
    if not has_evidence:
        missing_requirements.append("at least one verified evidence fact")
    if contract_misaligned:
        missing_requirements.append("an aligned answer contract")
    if has_evidence and _is_average_question(question) and _has_nan_like_text(evidence_text):
        missing_requirements.append("a resolved numeric value instead of NaN/None")
    if has_evidence and _is_row_extraction_question(question) and _has_zero_result_text(evidence_text):
        missing_requirements.append("at least one concrete result row before answering")

    ready_to_answer = has_query and not missing_requirements

    if ready_to_answer and remaining_steps is not None and remaining_steps <= 3:
        guidance = "Enough evidence is already available. Stop exploring and submit the answer now."
    elif ready_to_answer:
        guidance = (
            "You already have enough verified evidence. Prefer answering next, or at most lock the "
            "final columns before answering."
        )
    elif has_evidence and _is_average_question(question) and _has_nan_like_text(evidence_text):
        guidance = "The current result is NaN-like. Resolve the mapping or computation to a concrete numeric value before answering."
    elif has_evidence and _is_row_extraction_question(question) and _has_zero_result_text(evidence_text):
        guidance = "The current evidence suggests an empty result table. Re-check joins, mappings, or filters before answering."
    elif has_query and not has_evidence:
        guidance = "Record one concise evidence fact from verified results, then answer."
    else:
        guidance = _route_guidance(state)

    if _is_ratio_or_percentage_question(question):
        guidance = (
            f"{guidance} For ratio or percentage questions, submit one raw numeric value only, "
            "with no percent sign or explanatory text."
        )

    state.memory.finish = FinishMemory(
        ready_to_answer=ready_to_answer,
        missing_requirements=missing_requirements,
        guidance=guidance,
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
