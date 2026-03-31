from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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
    sanitized = re.sub(r"(?<!\\)\\'", "'", text)
    payload, end = json.JSONDecoder().raw_decode(sanitized)
    remainder = text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


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
    _QUERY_ACTIONS = frozenset({"execute_python", "execute_context_sql"})

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

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]

        task_content = build_task_prompt(task)
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

    @staticmethod
    def _outstanding_todo_items(state: AgentRuntimeState) -> list[str]:
        allowed = {"submit answer", "call answer", "final answer", "answer"}
        return [
            item
            for item in state.memory.plan.todo
            if item.strip() and item.strip().lower() not in allowed
        ]

    @staticmethod
    def _mapped_output_columns(state: AgentRuntimeState) -> list[str]:
        columns: list[str] = []
        for item in state.memory.output_mapping:
            for column in item.output_columns:
                if column not in columns:
                    columns.append(column)
        return columns

    def _check_answer_guards(
        self,
        state: AgentRuntimeState,
        action_input: dict[str, Any],
    ) -> str | None:
        has_successful_query = any(
            s.action in self._QUERY_ACTIONS and s.ok for s in state.steps
        )
        if not has_successful_query:
            return (
                "REJECTED: You must execute at least one successful query "
                "(execute_python or execute_context_sql) before submitting an answer. "
                "Go back and query the data."
            )
        if not state.memory.answer_shape.columns:
            return (
                "REJECTED: You must define the answer shape via set_answer_shape before "
                "submitting an answer. Record the final columns and row expectations first."
            )
        if not state.memory.output_mapping:
            return (
                "REJECTED: You must define an output mapping via set_output_mapping before "
                "submitting an answer. Map each question target to source fields or expressions first."
            )
        mapped_columns = self._mapped_output_columns(state)
        if mapped_columns != state.memory.answer_shape.columns:
            return (
                "REJECTED: set_output_mapping and set_answer_shape disagree about the final "
                "output columns. Update them so they match exactly before answering."
            )
        submitted_columns = action_input.get("columns")
        if submitted_columns != state.memory.answer_shape.columns:
            return (
                "REJECTED: answer.columns must exactly match the columns recorded via "
                "set_answer_shape. Update the memory or fix the answer columns."
            )
        if not state.memory.evidence:
            return (
                "REJECTED: You must record at least one verified result in memory via "
                "record_memory(section='evidence', ...) before submitting an answer."
            )
        if not state.memory.plan.is_set():
            return (
                "REJECTED: You must set a structured plan via set_plan before submitting "
                "an answer. Summarize what is done and what remains."
            )
        outstanding_todo = self._outstanding_todo_items(state)
        if outstanding_todo:
            return (
                "REJECTED: Your plan still has unfinished todo items: "
                + "; ".join(outstanding_todo)
                + ". Finish them or update the plan before answering."
            )
        return None

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        for step_index in range(1, self.config.max_steps + 1):
            messages = self._build_messages(task, state)
            raw_response = self.model.complete(messages)
            try:
                model_step = parse_model_step(raw_response)

                if model_step.action == "answer":
                    guard_error = self._check_answer_guards(state, model_step.action_input)
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
