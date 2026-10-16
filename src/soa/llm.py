"""Thin LiteLLM wrapper: call(), with token, cost and latency capture. The only module that imports litellm."""

import logging
import time
from typing import Literal

import litellm
from litellm.exceptions import (
    APIConnectionError,
    BadGatewayError,
    BadRequestError,
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


OPENROUTER_PREFIX = "openrouter/"
# OpenRouter routes to any provider by default and silently drops parameters that provider lacks. With
# require_parameters it only routes to providers that support every parameter in the request (here: response_format),
# so schema mode is never quietly downgraded to plain text. https://openrouter.ai/docs/features/provider-routing
OPENROUTER_REQUIRE_PARAMETERS: dict[str, object] = {"provider": {"require_parameters": True}}
SCHEMA_REJECTION_MARKERS = ("response_format", "json_schema", "unsupported")


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


def call(messages: list[Message], model: str, response_model: type[BaseModel] | None = None) -> LLMResponse:
    """Send messages to `model` and return text, token counts, cost and latency.

    With `response_model`, the provider's schema mode is requested, built from `response_model.model_json_schema()`
    (native mode, eval setups C and D). A provider that rejects it raises a non-retryable LLMAPIError.
    """
    started = time.perf_counter()
    extra: dict[str, object] = {}
    if response_model and model.startswith(OPENROUTER_PREFIX):
        extra["extra_body"] = OPENROUTER_REQUIRE_PARAMETERS
    try:
        raw = litellm.completion(
            model=model,
            messages=[m.model_dump() for m in messages],
            response_format=_schema_format(response_model) if response_model else None,
            **extra,
        )
    except APIError as exc:
        raise _to_api_error(exc, model, schema_mode=response_model is not None) from exc
    latency_ms = round((time.perf_counter() - started) * 1000)
    return _build_response(raw, model, latency_ms)


def _schema_format(response_model: type[BaseModel]) -> dict[str, object]:
    """The OpenAI-style `json_schema` response_format; LiteLLM translates it per provider."""
    return {
        "type": "json_schema",
        "json_schema": {"name": response_model.__name__, "schema": response_model.model_json_schema(), "strict": True},
    }


def _to_api_error(exc: APIError, model: str, *, schema_mode: bool = False) -> LLMAPIError:
    # A 400 on a schema-mode request is labelled "unsupported" only when the provider's message says so (it names
    # response_format / json_schema or says unsupported). Any other 400 (e.g. a prompt that is too long) stays a
    # generic non-retryable error with the provider's text. The class name counts: UnsupportedParamsError.
    if schema_mode and isinstance(exc, BadRequestError) and _mentions_schema_rejection(f"{type(exc).__name__} {exc}"):
        return LLMAPIError(
            f"schema mode unsupported by {model}: {type(exc).__name__}: {exc}",
            model=model,
            retryable=False,
            status_code=getattr(exc, "status_code", None),
        )
    return LLMAPIError(
        f"{type(exc).__name__}: {exc}",
        model=model,
        retryable=isinstance(exc, RETRYABLE_ERRORS),
        status_code=getattr(exc, "status_code", None),
    )


def _mentions_schema_rejection(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in SCHEMA_REJECTION_MARKERS)


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
