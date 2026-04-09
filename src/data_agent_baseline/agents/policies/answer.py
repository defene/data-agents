from __future__ import annotations

import re
from typing import Protocol

from data_agent_baseline.agents.runtime import AgentRuntimeState, OutputMappingItem
from data_agent_baseline.benchmark.schema import PublicTask

QUERY_ACTIONS = frozenset({"execute_python", "execute_context_sql"})
NAN_LIKE_TOKENS = frozenset({"nan", "none", "null", "n/a", "na"})
ROW_EXTRACT_STARTERS = (
    "list ",
    "show ",
    "which ",
    "what are ",
    "identify ",
    "write ",
    "name ",
    "return ",
    "tally ",
)
PROXY_PATTERNS = (
    "not found",
    "no record",
    "no records",
    "no rows",
    "no row",
    "no match",
    "no matches",
    "could not",
    "unable to",
    "saved to",
)
PATH_MARKERS = ("artifacts/runs", "artifacts\\runs", "scratch_output", "scratch\\", "scratch/")
NOTE_LIKE_COLUMNS = frozenset({"note", "notes", "message", "status", "reason", "explanation", "file", "path"})


class AnswerPolicy(Protocol):
    def normalize_submission(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        action_input: dict[str, object],
    ) -> dict[str, object]: ...

    def validate_submission(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        action_input: dict[str, object],
    ) -> str | None: ...


class AnswerFamilyPolicy(Protocol):
    def normalize_submission(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        action_input: dict[str, object],
    ) -> dict[str, object]: ...

    def validate_submission(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        action_input: dict[str, object],
    ) -> str | None: ...


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


def _is_ratio_or_percentage_question(question: str) -> bool:
    lowered = _normalize_question(question)
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


def _is_row_extraction_question(task: PublicTask, state: AgentRuntimeState) -> bool:
    question = _normalize_question(task.question)
    if state.memory.route.task_type == "table" or state.memory.route.output_hint == "multi_row_table":
        return True
    return question.startswith(ROW_EXTRACT_STARTERS)


def _first_single_cell(rows: list[object]) -> str | None:
    if len(rows) != 1 or not isinstance(rows[0], list) or len(rows[0]) != 1:
        return None
    return str(rows[0][0]).strip()


def _looks_like_proxy_answer(value: str) -> bool:
    lowered = value.strip().lower()
    if len(lowered) < 12:
        return False
    if any(marker in lowered for marker in PATH_MARKERS):
        return True
    if lowered.startswith("no ") or lowered.startswith("none "):
        return True
    return any(marker in lowered for marker in PROXY_PATTERNS)


def _question_mentions_full_name(task: PublicTask) -> bool:
    return "full name" in _normalize_question(task.question)


def _normalize_cell_text(value: object) -> str:
    return str(value).strip()


def _split_full_name_value(value: str) -> tuple[str, str] | None:
    cleaned = re.sub(r"\s+", " ", value.strip())
    if not cleaned or any(char.isdigit() for char in cleaned):
        return None
    parts = cleaned.split(" ")
    if len(parts) < 2 or len(parts) > 4:
        return None
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z'`.-]*", part) for part in parts):
        return None
    return parts[0], " ".join(parts[1:])


def _extract_prefixed_label(value: str) -> str | None:
    match = re.fullmatch(r"\s*\d+\s*-\s*(.+?)\s*", value)
    if match is None:
        return None
    label = match.group(1).strip()
    return label or None


class RatioAnswerFamilyPolicy:
    def normalize_submission(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        action_input: dict[str, object],
    ) -> dict[str, object]:
        raw_rows = action_input.get("rows")
        if not _is_ratio_or_percentage_question(task.question) or not isinstance(raw_rows, list):
            return action_input

        scalar_value = _single_non_empty_cell(raw_rows)
        normalized_value = (
            _extract_ratio_numeric_value(scalar_value) if scalar_value is not None else None
        )
        if normalized_value is None:
            return action_input

        ratio_normalized = dict(action_input)
        ratio_normalized["rows"] = [[normalized_value]]
        return ratio_normalized

    def validate_submission(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        action_input: dict[str, object],
    ) -> str | None:
        submitted_rows = action_input.get("rows")
        if not isinstance(submitted_rows, list):
            return None
        flat_cells = [
            str(cell).strip()
            for row in submitted_rows
            if isinstance(row, list)
            for cell in row
        ]
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
        return None


class RowExtractAnswerFamilyPolicy:
    def _augment_split_full_name(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        action_input: dict[str, object],
    ) -> dict[str, object]:
        if not _question_mentions_full_name(task):
            return action_input

        columns = action_input.get("columns")
        rows = action_input.get("rows")
        if not isinstance(columns, list) or len(columns) != 1 or not isinstance(rows, list) or not rows:
            return action_input

        split_rows: list[list[object]] = []
        for row in rows:
            if not isinstance(row, list) or len(row) != 1:
                return action_input
            full_name = _normalize_cell_text(row[0])
            split_value = _split_full_name_value(full_name)
            if split_value is None:
                return action_input
            first_name, last_name = split_value
            split_rows.append([full_name, first_name, last_name])

        expanded = dict(action_input)
        expanded["columns"] = [str(columns[0]).strip() or "full_name", "first_name", "last_name"]
        expanded["rows"] = split_rows
        return expanded

    def _augment_stripped_label_column(self, action_input: dict[str, object]) -> dict[str, object]:
        columns = action_input.get("columns")
        rows = action_input.get("rows")
        if not isinstance(columns, list) or len(columns) != 1 or not isinstance(rows, list) or not rows:
            return action_input

        derived_labels: list[str] = []
        for row in rows:
            if not isinstance(row, list) or len(row) != 1:
                return action_input
            label = _extract_prefixed_label(_normalize_cell_text(row[0]))
            if label is None:
                return action_input
            derived_labels.append(label)

        expanded = dict(action_input)
        expanded["columns"] = [str(columns[0]).strip() or "value", "label"]
        expanded["rows"] = [
            [_normalize_cell_text(row[0]), derived_labels[index]]
            for index, row in enumerate(rows)
        ]
        return expanded

    def normalize_submission(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        action_input: dict[str, object],
    ) -> dict[str, object]:
        if not _is_row_extraction_question(task, state):
            return action_input
        if state.memory.answer_shape.columns or state.memory.output_mapping:
            return action_input

        normalized = self._augment_split_full_name(task, state, action_input)
        normalized = self._augment_stripped_label_column(normalized)
        return normalized

    def validate_submission(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        action_input: dict[str, object],
    ) -> str | None:
        columns = action_input.get("columns")
        rows = action_input.get("rows")
        if not isinstance(columns, list) or not isinstance(rows, list):
            return None

        normalized_columns = [str(column).strip().lower() for column in columns if str(column).strip()]
        proxy_value = _first_single_cell(rows)
        if proxy_value is None:
            return None

        if normalized_columns and set(normalized_columns).issubset(NOTE_LIKE_COLUMNS) and _looks_like_proxy_answer(proxy_value):
            return (
                "REJECTED: Do not submit note-, message-, file-, or path-style proxy answers. "
                "Submit the actual final table values."
            )
        return None


class DefaultAnswerPolicy:
    def __init__(self, family_policies: tuple[AnswerFamilyPolicy, ...] | None = None) -> None:
        self.family_policies = family_policies or (
            RatioAnswerFamilyPolicy(),
            RowExtractAnswerFamilyPolicy(),
        )

    def _populate_answer_contract(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        action_input: dict[str, object],
    ) -> None:
        normalized_columns = action_input.get("columns")
        normalized_rows = action_input.get("rows")
        if not isinstance(normalized_columns, list):
            return

        if not state.memory.answer_shape.columns:
            state.memory.answer_shape.columns = list(normalized_columns)
            if isinstance(normalized_rows, list):
                state.memory.answer_shape.expected_rows = len(normalized_rows)
            state.memory.answer_shape.sort_requirement = (
                "single summary row"
                if isinstance(normalized_rows, list) and len(normalized_rows) <= 1
                else ""
            )
            state.memory.answer_shape.value_format_rules = ["preserve source formatting"]

        if not state.memory.output_mapping:
            state.memory.output_mapping = [
                OutputMappingItem(
                    question_target=task.question.strip(),
                    source_kind="derived_expression",
                    source_refs=list(state.memory.evidence[:3])
                    or list(state.memory.route.primary_sources),
                    output_columns=list(normalized_columns),
                    granularity=(
                        "aggregate"
                        if isinstance(normalized_rows, list) and len(normalized_rows) <= 1
                        else "detail_rows"
                    ),
                    format_rule="preserve source formatting",
                )
            ]

        if not state.memory.plan.is_set():
            state.memory.plan.done = ["collected verified evidence", "prepared final table"]
            state.memory.plan.todo = ["submit answer"]
            state.memory.plan.next_action = "submit answer"

    def normalize_submission(
        self,
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

        for policy in self.family_policies:
            normalized_action_input = policy.normalize_submission(
                task,
                state,
                normalized_action_input,
            )
        self._populate_answer_contract(task, state, normalized_action_input)
        return normalized_action_input

    def validate_submission(
        self,
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

        flat_cells = [
            str(cell).strip()
            for row in submitted_rows
            if isinstance(row, list)
            for cell in row
        ]
        flat_cells_lower = [cell.lower() for cell in flat_cells]
        if len(submitted_rows) <= 1 and any(cell in NAN_LIKE_TOKENS for cell in flat_cells_lower):
            return (
                "REJECTED: Do not submit NaN/None-like final values. Resolve the computation to a "
                "concrete numeric or text answer first."
            )

        proxy_value = _first_single_cell(submitted_rows)
        if proxy_value is not None and _looks_like_proxy_answer(proxy_value):
            return (
                "REJECTED: Do not submit narrative proxy answers such as 'No ... found' or file paths. "
                "Submit actual final table values instead."
            )

        for policy in self.family_policies:
            family_error = policy.validate_submission(task, state, action_input)
            if family_error is not None:
                return family_error

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


DEFAULT_ANSWER_POLICY = DefaultAnswerPolicy()
