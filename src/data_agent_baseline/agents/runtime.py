from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from data_agent_baseline.benchmark.schema import AnswerTable


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
    schema: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    output_mapping: list[OutputMappingItem] = field(default_factory=list)
    answer_shape: AnswerShapeMemory = field(default_factory=AnswerShapeMemory)
    plan: PlanMemory = field(default_factory=PlanMemory)

    def has_content(self) -> bool:
        return bool(
            self.schema
            or self.rules
            or self.entities
            or self.evidence
            or self.output_mapping
            or self.answer_shape.is_set()
            or self.plan.is_set()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
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
