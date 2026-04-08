from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from data_agent_baseline.agents.controller import (
    bootstrap_memory_from_context,
    normalize_answer_submission,
    update_finish_memory,
    validate_answer_submission,
)
from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_memory_prompt,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import (
    AgentRunResult,
    AgentRuntimeState,
    OutputMappingItem,
    StepRecord,
)
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16
    bootstrap_context_depth: int = 3
    finish_step_buffer: int = 3


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


def _load_single_json_object(text: str) -> dict[str, Any]:
    sanitized = re.sub(r"(?<!\\)\\'", "'", text).strip()
    decoder = json.JSONDecoder()
    search_from = sanitized.find("{")
    while search_from != -1:
        try:
            payload, _end = decoder.raw_decode(sanitized[search_from:])
        except json.JSONDecodeError:
            search_from = sanitized.find("{", search_from + 1)
            continue
        if isinstance(payload, dict):
            return payload
        search_from = sanitized.find("{", search_from + 1)
    raise ValueError("Model response must contain a JSON object.")


def parse_model_step(raw_response: str) -> ModelStep:
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
    )


class ReActAgent:
    _FINISH_PRIORITY_ACTIONS = frozenset({"answer", "set_answer_shape", "set_output_mapping", "set_plan"})
    _LOW_VALUE_EXPLORATION_ACTIONS = frozenset(
        {
            "inspect_sqlite_schema",
            "list_context",
            "list_doc_chunks",
            "read_csv",
            "read_doc",
            "read_doc_chunk",
            "read_json",
            "search_doc",
        }
    )

    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
        progress_callback: Callable[[AgentRuntimeState], None] | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT
        self.progress_callback = progress_callback

    def _build_messages(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        *,
        remaining_steps: int,
    ) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]

        task_content = build_task_prompt(task) + f"\n\nRemaining step budget: {remaining_steps}"
        if state.memory.has_content():
            task_content += "\n\n" + build_memory_prompt(state.memory)
        messages.append(ModelMessage(role="user", content=task_content))

        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        return messages

    @staticmethod
    def _append_unique(items: list[str], value: str) -> None:
        if value not in items:
            items.append(value)

    def _apply_memory_patch(self, state: AgentRuntimeState, memory_patch: dict[str, Any] | None) -> None:
        if not memory_patch:
            return

        append_patch = memory_patch.get("append")
        if isinstance(append_patch, dict):
            section = append_patch.get("section")
            fact = append_patch.get("fact")
            if isinstance(section, str) and isinstance(fact, str) and fact.strip():
                bucket = getattr(state.memory, section, None)
                if isinstance(bucket, list):
                    self._append_unique(bucket, fact.strip())

        output_mapping_patch = memory_patch.get("set_output_mapping")
        if isinstance(output_mapping_patch, dict):
            raw_mappings = output_mapping_patch.get("mappings", [])
            if isinstance(raw_mappings, list):
                normalized_items: list[OutputMappingItem] = []
                for item in raw_mappings:
                    if not isinstance(item, dict):
                        continue
                    normalized_items.append(
                        OutputMappingItem(
                            question_target=str(item.get("question_target", "")),
                            source_kind=str(item.get("source_kind", "")),
                            source_refs=list(item.get("source_refs", [])),
                            output_columns=list(item.get("output_columns", [])),
                            granularity=str(item.get("granularity", "")),
                            format_rule=str(item.get("format_rule", "")),
                        )
                    )
                state.memory.output_mapping = normalized_items

        answer_shape_patch = memory_patch.get("set_answer_shape")
        if isinstance(answer_shape_patch, dict):
            state.memory.answer_shape.columns = list(answer_shape_patch.get("columns", []))
            state.memory.answer_shape.expected_rows = answer_shape_patch.get("expected_rows")
            state.memory.answer_shape.sort_requirement = str(
                answer_shape_patch.get("sort_requirement", "")
            )
            state.memory.answer_shape.value_format_rules = list(
                answer_shape_patch.get("value_format_rules", [])
            )

        plan_patch = memory_patch.get("set_plan")
        if isinstance(plan_patch, dict):
            state.memory.plan.done = list(plan_patch.get("done", []))
            state.memory.plan.todo = list(plan_patch.get("todo", []))
            state.memory.plan.next_action = str(plan_patch.get("next_action", ""))

    def _bootstrap_state(self, task: PublicTask, state: AgentRuntimeState) -> None:
        try:
            tool_result = self.tools.execute(
                task,
                "list_context",
                {"max_depth": self.config.bootstrap_context_depth},
            )
        except Exception:
            return

        if tool_result.ok:
            bootstrap_memory_from_context(
                state,
                question=task.question,
                context_listing=tool_result.content,
            )
            update_finish_memory(state, remaining_steps=self.config.max_steps)

    def _should_block_low_value_exploration(
        self,
        state: AgentRuntimeState,
        *,
        action: str,
        remaining_steps: int,
    ) -> bool:
        return (
            state.memory.finish.ready_to_answer
            and remaining_steps <= self.config.finish_step_buffer
            and action in self._LOW_VALUE_EXPLORATION_ACTIONS
        )

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        self._bootstrap_state(task, state)

        for step_index in range(1, self.config.max_steps + 1):
            remaining_steps = self.config.max_steps - step_index + 1
            update_finish_memory(state, remaining_steps=remaining_steps)
            messages = self._build_messages(task, state, remaining_steps=remaining_steps)
            raw_response = self.model.complete(messages)
            try:
                model_step = parse_model_step(raw_response)

                if self._should_block_low_value_exploration(
                    state,
                    action=model_step.action,
                    remaining_steps=remaining_steps,
                ) and model_step.action not in self._FINISH_PRIORITY_ACTIONS:
                    observation = {
                        "ok": False,
                        "error": (
                            "Controller: enough verified evidence is already available. "
                            "Stop exploring and move directly toward the final answer."
                        ),
                    }
                    state.steps.append(
                        StepRecord(
                            step_index=step_index,
                            thought=model_step.thought,
                            action=model_step.action,
                            action_input=model_step.action_input,
                            raw_response=raw_response,
                            observation=observation,
                            ok=False,
                            memory_snapshot=state.memory.to_dict(),
                        )
                    )
                    if self.progress_callback is not None:
                        self.progress_callback(state)
                    continue

                if model_step.action == "answer":
                    normalized_action_input = normalize_answer_submission(
                        task,
                        state,
                        model_step.action_input,
                    )
                    update_finish_memory(state, remaining_steps=remaining_steps)
                    guard_error = validate_answer_submission(state, normalized_action_input)
                    model_step = ModelStep(
                        thought=model_step.thought,
                        action=model_step.action,
                        action_input=normalized_action_input,
                        raw_response=raw_response,
                    )
                    if guard_error is not None:
                        observation = {"ok": False, "error": guard_error}
                        state.steps.append(
                            StepRecord(
                                step_index=step_index,
                                thought=model_step.thought,
                                action=model_step.action,
                                action_input=model_step.action_input,
                                raw_response=raw_response,
                                observation=observation,
                                ok=False,
                                memory_snapshot=state.memory.to_dict(),
                            )
                        )
                        if self.progress_callback is not None:
                            self.progress_callback(state)
                        continue

                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                if tool_result.ok:
                    self._apply_memory_patch(state, tool_result.memory_patch)
                update_finish_memory(
                    state,
                    remaining_steps=self.config.max_steps - step_index,
                )

                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought=model_step.thought,
                        action=model_step.action,
                        action_input=model_step.action_input,
                        raw_response=raw_response,
                        observation=observation,
                        ok=tool_result.ok,
                        memory_snapshot=state.memory.to_dict(),
                    )
                )
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                if self.progress_callback is not None:
                    self.progress_callback(state)
                if tool_result.is_terminal:
                    break
            except Exception as exc:
                observation = {
                    "ok": False,
                    "error": str(exc),
                }
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                        memory_snapshot=state.memory.to_dict(),
                    )
                )
                update_finish_memory(
                    state,
                    remaining_steps=self.config.max_steps - step_index,
                )
                if self.progress_callback is not None:
                    self.progress_callback(state)

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
            memory=state.memory,
        )
