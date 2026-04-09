from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from data_agent_baseline.benchmark.schema import AnswerTable


@dataclass(slots=True)
class SourceIndexMemory:
    tabular_files: list[str] = field(default_factory=list)
    doc_files: list[str] = field(default_factory=list)
    db_files: list[str] = field(default_factory=list)
    other_files: list[str] = field(default_factory=list)

    def has_content(self) -> bool:
        return bool(self.tabular_files or self.doc_files or self.db_files or self.other_files)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tabular_files": list(self.tabular_files),
            "doc_files": list(self.doc_files),
            "db_files": list(self.db_files),
            "other_files": list(self.other_files),
        }


@dataclass(slots=True)
class RouteMemory:
    task_type: str = ""
    solver_mode: str = ""
    question_intent: str = ""
    primary_sources: list[str] = field(default_factory=list)
    supporting_sources: list[str] = field(default_factory=list)
    rationale: str = ""
    output_hint: str = ""

    def is_set(self) -> bool:
        return bool(
            self.task_type
            or self.solver_mode
            or self.question_intent
            or self.primary_sources
            or self.supporting_sources
            or self.rationale
            or self.output_hint
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "solver_mode": self.solver_mode,
            "question_intent": self.question_intent,
            "primary_sources": list(self.primary_sources),
            "supporting_sources": list(self.supporting_sources),
            "rationale": self.rationale,
            "output_hint": self.output_hint,
        }


@dataclass(slots=True)
class FinishMemory:
    ready_to_answer: bool = False
    missing_requirements: list[str] = field(default_factory=list)
    guidance: str = ""
    remaining_steps: int | None = None

    def is_set(self) -> bool:
        return bool(
            self.ready_to_answer
            or self.missing_requirements
            or self.guidance
            or self.remaining_steps is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready_to_answer": self.ready_to_answer,
            "missing_requirements": list(self.missing_requirements),
            "guidance": self.guidance,
            "remaining_steps": self.remaining_steps,
        }


@dataclass(slots=True)
class CandidateResultMemory:
    ready: bool = False
    kind: str = ""
    source: str = ""
    columns: list[str] = field(default_factory=list)
    row_count: int | None = None
    value_count: int | None = None
    summary: str = ""

    def is_set(self) -> bool:
        return bool(
            self.ready
            or self.kind
            or self.source
            or self.columns
            or self.row_count is not None
            or self.value_count is not None
            or self.summary
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "kind": self.kind,
            "source": self.source,
            "columns": list(self.columns),
            "row_count": self.row_count,
            "value_count": self.value_count,
            "summary": self.summary,
        }


@dataclass(slots=True)
class AnswerShapeMemory:
    columns: list[str] = field(default_factory=list)
    expected_rows: int | None = None
    sort_requirement: str = ""
    value_format_rules: list[str] = field(default_factory=list)

    def is_set(self) -> bool:
        return bool(
            self.columns
            or self.expected_rows is not None
            or self.sort_requirement
            or self.value_format_rules
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "expected_rows": self.expected_rows,
            "sort_requirement": self.sort_requirement,
            "value_format_rules": list(self.value_format_rules),
        }


@dataclass(slots=True)
class OutputMappingItem:
    question_target: str
    source_kind: str
    source_refs: list[str] = field(default_factory=list)
    output_columns: list[str] = field(default_factory=list)
    granularity: str = ""
    format_rule: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_target": self.question_target,
            "source_kind": self.source_kind,
            "source_refs": list(self.source_refs),
            "output_columns": list(self.output_columns),
            "granularity": self.granularity,
            "format_rule": self.format_rule,
        }


@dataclass(slots=True)
class PlanMemory:
    done: list[str] = field(default_factory=list)
    todo: list[str] = field(default_factory=list)
    next_action: str = ""

    def is_set(self) -> bool:
        return bool(self.done or self.todo or self.next_action)

    def to_dict(self) -> dict[str, Any]:
        return {
            "done": list(self.done),
            "todo": list(self.todo),
            "next_action": self.next_action,
        }


@dataclass(slots=True)
class StructuredMemory:
    source_index: SourceIndexMemory = field(default_factory=SourceIndexMemory)
    route: RouteMemory = field(default_factory=RouteMemory)
    finish: FinishMemory = field(default_factory=FinishMemory)
    candidate_result: CandidateResultMemory = field(default_factory=CandidateResultMemory)
    schema: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    output_mapping: list[OutputMappingItem] = field(default_factory=list)
    answer_shape: AnswerShapeMemory = field(default_factory=AnswerShapeMemory)
    plan: PlanMemory = field(default_factory=PlanMemory)

    def has_content(self) -> bool:
        return bool(
            self.source_index.has_content()
            or self.route.is_set()
            or self.finish.is_set()
            or self.candidate_result.is_set()
            or self.schema
            or self.rules
            or self.entities
            or self.evidence
            or self.output_mapping
            or self.answer_shape.is_set()
            or self.plan.is_set()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_index": self.source_index.to_dict(),
            "route": self.route.to_dict(),
            "finish": self.finish.to_dict(),
            "candidate_result": self.candidate_result.to_dict(),
            "schema": list(self.schema),
            "rules": list(self.rules),
            "entities": list(self.entities),
            "evidence": list(self.evidence),
            "output_mapping": [item.to_dict() for item in self.output_mapping],
            "answer_shape": self.answer_shape.to_dict(),
            "plan": self.plan.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class StepRecord:
    step_index: int
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str
    observation: dict[str, Any]
    ok: bool
    memory_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_run_result_dict(
    task_id: str,
    *,
    answer: AnswerTable | None,
    steps: list[StepRecord],
    failure_reason: str | None,
    memory: StructuredMemory,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "answer": answer.to_dict() if answer is not None else None,
        "steps": [step.to_dict() for step in steps],
        "failure_reason": failure_reason,
        "succeeded": answer is not None and failure_reason is None,
        "memory": memory.to_dict(),
    }


@dataclass(slots=True)
class AgentRuntimeState:
    steps: list[StepRecord] = field(default_factory=list)
    memory: StructuredMemory = field(default_factory=StructuredMemory)
    answer: AnswerTable | None = None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    task_id: str
    answer: AnswerTable | None
    steps: list[StepRecord]
    failure_reason: str | None
    memory: StructuredMemory

    @property
    def succeeded(self) -> bool:
        return self.answer is not None and self.failure_reason is None

    def to_dict(self) -> dict[str, Any]:
        return build_run_result_dict(
            self.task_id,
            answer=self.answer,
            steps=self.steps,
            failure_reason=self.failure_reason,
            memory=self.memory,
        )
