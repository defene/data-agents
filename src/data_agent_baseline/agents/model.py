from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIError, APIStatusError, OpenAI

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0
NON_RETRYABLE_STATUS_CODES = {400, 401, 403}


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str


class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError


class OpenAIModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self._client = OpenAI(api_key=api_key, base_url=api_base.rstrip("/"))

    def complete(self, messages: list[ModelMessage]) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort

        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                choices = response.choices or []
                if not choices:
                    raise RuntimeError("Model response missing choices.")
                content = choices[0].message.content
                if not isinstance(content, str):
                    raise RuntimeError("Model response missing text content.")
                return content
            except APIStatusError as exc:
                if exc.status_code in NON_RETRYABLE_STATUS_CODES:
                    raise RuntimeError(f"Model request failed (non-retryable): {exc}") from exc
                last_exc = exc
            except APIError as exc:
                last_exc = exc

            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "API call failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt, MAX_RETRIES, delay, last_exc,
                )
                time.sleep(delay)

        raise RuntimeError(f"Model request failed after {MAX_RETRIES} attempts: {last_exc}") from last_exc


class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
