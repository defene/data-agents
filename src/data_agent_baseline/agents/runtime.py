from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from data_agent_baseline.benchmark.schema import AnswerTable


@dataclass(frozen=True, slots=True)
class StepRecord:
    step_index: int
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str
    observation: dict[str, Any]
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_run_result_dict(
    task_id: str,
    *,
    answer: AnswerTable | None,
    steps: list[StepRecord],
    failure_reason: str | None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "answer": answer.to_dict() if answer is not None else None,
        "steps": [step.to_dict() for step in steps],
        "failure_reason": failure_reason,
        "succeeded": answer is not None and failure_reason is None,
    }


@dataclass(slots=True)
class AgentRuntimeState:
    steps: list[StepRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    answer: AnswerTable | None = None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    task_id: str
    answer: AnswerTable | None
    steps: list[StepRecord]
    failure_reason: str | None

    @property
    def succeeded(self) -> bool:
        return self.answer is not None and self.failure_reason is None

    def to_dict(self) -> dict[str, Any]:
        return build_run_result_dict(
            self.task_id,
            answer=self.answer,
            steps=self.steps,
            failure_reason=self.failure_reason,
        )
