"""Thin LiteLLM wrapper: call(), with token, cost and latency capture. The only module that imports litellm."""

import logging
import time
from typing import Literal

import litellm
from litellm.exceptions import (
    APIConnectionError,
    BadGatewayError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from openai import APIError
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

RETRYABLE_ERRORS = (
    RateLimitError,
    Timeout,
    APIConnectionError,
    InternalServerError,
    ServiceUnavailableError,
    BadGatewayError,
)


class Message(BaseModel):
    """One chat message."""

    model_config = ConfigDict(extra="forbid", strict=True)

    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    """What one LLM call returned, plus what it cost."""

    model_config = ConfigDict(extra="forbid", strict=True)

    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cost_known: bool
    latency_ms: int
    model: str


class LLMAPIError(Exception):
    """The provider call failed (rate limit, timeout, 5xx, auth...). Not a validation failure."""

    def __init__(self, message: str, *, model: str, retryable: bool, status_code: int | None) -> None:
        super().__init__(message)
        self.model = model
        self.retryable = retryable
        self.status_code = status_code


def call(messages: list[Message], model: str, json_mode: bool = False) -> LLMResponse:
    """Send messages to `model` and return text, token counts, cost and latency."""
    started = time.perf_counter()
    try:
        raw = litellm.completion(
            model=model,
            messages=[m.model_dump() for m in messages],
            response_format={"type": "json_object"} if json_mode else None,
        )
    except APIError as exc:
        raise _to_api_error(exc, model) from exc
    latency_ms = round((time.perf_counter() - started) * 1000)
    return _build_response(raw, model, latency_ms)


def _to_api_error(exc: APIError, model: str) -> LLMAPIError:
    return LLMAPIError(
        f"{type(exc).__name__}: {exc}",
        model=model,
        retryable=isinstance(exc, RETRYABLE_ERRORS),
        status_code=getattr(exc, "status_code", None),
    )


def _build_response(raw: object, model: str, latency_ms: int) -> LLMResponse:
    if not isinstance(raw, litellm.ModelResponse):
        raise TypeError(f"expected a non-streaming ModelResponse, got {type(raw).__name__}")
    choice = raw.choices[0]
    if not isinstance(choice, litellm.Choices):
        raise TypeError(f"expected a non-streaming Choices, got {type(choice).__name__}")
    usage = getattr(raw, "usage", None)
    cost_usd, cost_known = _cost(raw, model)
    return LLMResponse(
        text=choice.message.content or "",
        prompt_tokens=getattr(usage, "prompt_tokens", 0),
        completion_tokens=getattr(usage, "completion_tokens", 0),
        cost_usd=cost_usd,
        cost_known=cost_known,
        latency_ms=latency_ms,
        model=model,
    )


def _cost(raw: litellm.ModelResponse, model: str) -> tuple[float, bool]:
    """Cost in USD and whether it is known. Models missing from LiteLLM's price map cost 0.0."""
    try:
        return float(litellm.completion_cost(completion_response=raw)), True
    except Exception as exc:  # LiteLLM raises a bare Exception for unmapped models
        logger.warning("cost unknown for model %r: %s", model, exc)
        return 0.0, False
