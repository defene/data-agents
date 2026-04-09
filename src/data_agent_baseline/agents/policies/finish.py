from __future__ import annotations

import re
from typing import Protocol

from data_agent_baseline.agents.runtime import AgentRuntimeState, FinishMemory

QUERY_ACTIONS = frozenset({"execute_python", "execute_context_sql"})
NAN_LIKE_TOKENS = frozenset({"nan", "none", "null", "n/a", "na"})


class FinishPolicy(Protocol):
    def build_finish_memory(
        self,
        state: AgentRuntimeState,
        *,
        question: str,
        remaining_steps: int | None = None,
    ) -> FinishMemory: ...


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
        "0 rows",
        "no rows",
        "no row",
        "no match",
        "no matches",
        "no record",
        "no records",
        "not found",
        "none found",
        "none labeled",
        "no exact",
        "returned 0 rows",
        "empty",
        "[]",
        "count=0",
        "count: 0",
        "zero ",
        "=0",
    )
    return any(marker in text for marker in markers)


class DefaultFinishPolicy:
    def _requires_concrete_answer(self, state: AgentRuntimeState) -> bool:
        return state.memory.route.task_type in {"lookup", "table"}

    def _route_guidance(self, state: AgentRuntimeState) -> str:
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

    def build_finish_memory(
        self,
        state: AgentRuntimeState,
        *,
        question: str,
        remaining_steps: int | None = None,
    ) -> FinishMemory:
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
        if (
            has_evidence
            and self._requires_concrete_answer(state)
            and _has_zero_result_text(evidence_text)
        ):
            missing_requirements.append(
                "a concrete result value or row instead of a negative/inconclusive evidence summary"
            )
        if has_evidence and _is_row_extraction_question(question) and _has_zero_result_text(
            evidence_text
        ):
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
            guidance = (
                "The current result is NaN-like. Resolve the mapping or computation to a concrete "
                "numeric value before answering."
            )
        elif (
            has_evidence
            and self._requires_concrete_answer(state)
            and _has_zero_result_text(evidence_text)
        ):
            guidance = (
                "The current evidence is negative or inconclusive. Do not answer with 'not found' or "
                "similar text. Run another query to extract the concrete final value or result rows."
            )
        elif has_evidence and _is_row_extraction_question(question) and _has_zero_result_text(
            evidence_text
        ):
            guidance = (
                "The current evidence suggests an empty result table. Re-check joins, mappings, or "
                "filters before answering."
            )
        elif has_query and not has_evidence:
            guidance = "Record one concise evidence fact from verified results, then answer."
        else:
            guidance = self._route_guidance(state)

        if _is_ratio_or_percentage_question(question):
            guidance = (
                f"{guidance} For ratio or percentage questions, submit one raw numeric value only, "
                "with no percent sign or explanatory text."
            )

        return FinishMemory(
            ready_to_answer=ready_to_answer,
            missing_requirements=missing_requirements,
            guidance=guidance,
            remaining_steps=remaining_steps,
        )


DEFAULT_FINISH_POLICY = DefaultFinishPolicy()
