from __future__ import annotations

from data_agent_baseline.agents.policies.answer import DEFAULT_ANSWER_POLICY, AnswerPolicy
from data_agent_baseline.agents.policies.finish import DEFAULT_FINISH_POLICY, FinishPolicy
from data_agent_baseline.agents.router import build_source_index, infer_task_route
from data_agent_baseline.agents.runtime import AgentRuntimeState
from data_agent_baseline.benchmark.schema import PublicTask

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
    answer_policy: AnswerPolicy | None = None,
) -> dict[str, object]:
    return (answer_policy or DEFAULT_ANSWER_POLICY).normalize_submission(
        task,
        state,
        action_input,
    )


def validate_answer_submission(
    task: PublicTask,
    state: AgentRuntimeState,
    action_input: dict[str, object],
    answer_policy: AnswerPolicy | None = None,
) -> str | None:
    return (answer_policy or DEFAULT_ANSWER_POLICY).validate_submission(
        task,
        state,
        action_input,
    )
