from __future__ import annotations

import ast
import csv
import re
from pathlib import Path
from typing import Any, Protocol

from data_agent_baseline.agents.runtime import AgentRuntimeState, CandidateResultMemory
from data_agent_baseline.benchmark.schema import PublicTask

QUERY_ACTIONS = frozenset({"execute_python", "execute_context_sql"})
PATH_LIKE_PATTERN = re.compile(r"^[A-Za-z]:[\\/].+\.(?:csv|tsv|json|txt)$", flags=re.IGNORECASE)
NEGATIVE_PROXY_MARKERS = (
    "no ",
    "not found",
    "no rows",
    "no match",
    "no matches",
    "empty",
)
LIST_LABELS = ("numbers", "number", "driver_ids", "ids", "values", "dates")


class RecipePolicy(Protocol):
    def observe_tool_result(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        *,
        action: str,
        action_input: dict[str, object],
        tool_content: dict[str, object],
    ) -> None: ...

    def validate_answer_attempt(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        action_input: dict[str, object],
    ) -> str | None: ...


def _requires_candidate_result(state: AgentRuntimeState) -> bool:
    return state.memory.route.output_hint in {"single_row_lookup", "multi_row_table"}


def _candidate_strength(candidate: CandidateResultMemory) -> tuple[int, int, int]:
    kind_weight = 2 if candidate.kind == "rows" else 1
    count = candidate.row_count or candidate.value_count or 0
    return (kind_weight, count, len(candidate.columns))


def _merge_candidate(state: AgentRuntimeState, candidate: CandidateResultMemory) -> None:
    if not candidate.ready:
        return
    current = state.memory.candidate_result
    if not current.ready or _candidate_strength(candidate) >= _candidate_strength(current):
        state.memory.candidate_result = candidate


def _extract_sql_candidate(tool_content: dict[str, object]) -> CandidateResultMemory | None:
    columns = tool_content.get("columns")
    rows = tool_content.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list) or not rows:
        return None

    normalized_columns = [str(column) for column in columns]
    row_count = len(rows)
    is_scalar = row_count == 1 and len(normalized_columns) == 1
    return CandidateResultMemory(
        ready=True,
        kind="scalar" if is_scalar else "rows",
        source="execute_context_sql",
        columns=normalized_columns,
        row_count=row_count,
        value_count=1 if is_scalar else None,
        summary=(
            f"SQL query returned scalar {rows[0][0]!r}."
            if is_scalar
            else f"SQL query returned {row_count} row(s) across columns {normalized_columns}."
        ),
    )


def _extract_scratch_paths(output: str, scratch_root: Path) -> list[Path]:
    path_matches = re.findall(r"([A-Za-z]:[\\/][^\r\n]+?\.(?:csv|tsv|json))", output, flags=re.IGNORECASE)
    resolved_root = scratch_root.resolve()
    candidates: list[Path] = []
    for raw_path in path_matches:
        candidate = Path(raw_path)
        try:
            resolved_candidate = candidate.resolve()
        except OSError:
            continue
        try:
            resolved_candidate.relative_to(resolved_root)
        except ValueError:
            continue
        if resolved_candidate not in candidates:
            candidates.append(resolved_candidate)
    return candidates


def _load_tabular_candidate(path: Path) -> CandidateResultMemory | None:
    suffix = path.suffix.lower()
    delimiter = "\t" if suffix == ".tsv" else ","
    if suffix not in {".csv", ".tsv"} or not path.exists() or not path.is_file():
        return None

    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            header = next(reader, None)
            if not header:
                return None
            row_count = 0
            for row in reader:
                if any(str(cell).strip() for cell in row):
                    row_count += 1
    except OSError:
        return None

    if row_count == 0:
        return None

    normalized_columns = [str(column).strip() for column in header if str(column).strip()]
    is_scalar = row_count == 1 and len(normalized_columns) == 1
    return CandidateResultMemory(
        ready=True,
        kind="scalar" if is_scalar else "rows",
        source=f"scratch:{path.name}",
        columns=normalized_columns,
        row_count=row_count,
        value_count=1 if is_scalar else None,
        summary=(
            f"Scratch file {path.name} contains one scalar value."
            if is_scalar
            else f"Scratch file {path.name} contains {row_count} row(s) across columns {normalized_columns}."
        ),
    )


def _extract_named_list_candidate(output: str) -> CandidateResultMemory | None:
    for label in LIST_LABELS:
        match = re.search(rf"{label}\s*:\s*(\[[^\]]*\])", output, flags=re.IGNORECASE)
        if match is None:
            continue
        try:
            values = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            continue
        if not isinstance(values, list):
            continue
        normalized_values = [str(value).strip() for value in values if str(value).strip()]
        if not normalized_values:
            continue
        is_scalar = len(normalized_values) == 1
        return CandidateResultMemory(
            ready=True,
            kind="scalar" if is_scalar else "rows",
            source="execute_python",
            columns=[label.rstrip("s")],
            row_count=None if is_scalar else len(normalized_values),
            value_count=len(normalized_values),
            summary=(
                f"Python query produced scalar value {normalized_values[0]!r}."
                if is_scalar
                else f"Python query produced candidate values {normalized_values[:3]!r}."
            ),
        )
    return None


def _extract_python_candidate(tool_content: dict[str, object]) -> CandidateResultMemory | None:
    output = str(tool_content.get("output", "") or "")
    scratch_root_raw = tool_content.get("scratch_root")
    scratch_root = Path(str(scratch_root_raw)) if isinstance(scratch_root_raw, str) else None

    if scratch_root is not None:
        referenced_paths = _extract_scratch_paths(output, scratch_root)
        for path in referenced_paths:
            candidate = _load_tabular_candidate(path)
            if candidate is not None:
                return candidate

        for path in sorted(scratch_root.glob("*.csv"), key=lambda item: item.stat().st_mtime, reverse=True):
            candidate = _load_tabular_candidate(path)
            if candidate is not None:
                return candidate

    return _extract_named_list_candidate(output)


def _looks_like_proxy_answer(action_input: dict[str, object]) -> bool:
    rows = action_input.get("rows")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list) or len(rows[0]) != 1:
        return False

    cell = str(rows[0][0]).strip()
    if not cell:
        return False

    lowered = cell.lower()
    if PATH_LIKE_PATTERN.fullmatch(cell):
        return True
    return any(lowered.startswith(marker) for marker in NEGATIVE_PROXY_MARKERS)


class DefaultRecipePolicy:
    def observe_tool_result(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        *,
        action: str,
        action_input: dict[str, object],
        tool_content: dict[str, object],
    ) -> None:
        del task, action_input
        if action not in QUERY_ACTIONS or not _requires_candidate_result(state):
            return

        candidate: CandidateResultMemory | None = None
        if action == "execute_context_sql":
            candidate = _extract_sql_candidate(tool_content)
        elif action == "execute_python":
            candidate = _extract_python_candidate(tool_content)

        if candidate is not None:
            _merge_candidate(state, candidate)

    def validate_answer_attempt(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        action_input: dict[str, object],
    ) -> str | None:
        del task
        if not _requires_candidate_result(state):
            return None
        if not state.memory.candidate_result.ready:
            return (
                "REJECTED: Before answering this lookup/table task, first materialize a concrete "
                "candidate value or result row set from a successful query or scratch output."
            )
        if _looks_like_proxy_answer(action_input):
            return (
                "REJECTED: Do not answer with a file path or 'not found' narrative. Return the "
                "concrete value(s) from the candidate result instead."
            )
        return None


DEFAULT_RECIPE_POLICY = DefaultRecipePolicy()
