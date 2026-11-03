"""Thin LiteLLM wrapper: call(), with token, cost and latency capture. The only module that imports litellm."""

import functools
import json
import logging
import time
import urllib.request
from typing import Literal, Self

import litellm
from litellm.exceptions import (
    APIConnectionError,
    BadGatewayError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from openai import APIError
from pydantic import BaseModel, ConfigDict, model_validator

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
# With require_parameters and no provider able to serve the request, OpenRouter answers 404 with a message such as
# "No endpoints found that can handle the requested parameters". The wording is not documented, so a 404 counts as
# "no provider supports schema mode" only when its message says "no endpoints" or names the provider.
NO_PROVIDER_MARKERS = ("no endpoints", "provider")
# OpenRouter returns the actual cost of a request in `usage.cost` (credits = USD) with no request flag needed, and
# publishes per-token list prices. https://openrouter.ai/docs/use-cases/usage-accounting
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
PRICE_FETCH_TIMEOUT_SECONDS = 10


class Message(BaseModel):
    """One chat message."""

    model_config = ConfigDict(extra="forbid", strict=True)

    role: Literal["system", "user", "assistant"]
    content: str


CostSource = Literal["provider", "estimate", "none"]


class LLMResponse(BaseModel):
    """What one LLM call returned, plus what it cost and where that number came from.

    cost_source: "provider" = the cost the provider reported for this request (OpenRouter `usage.cost`);
    "estimate" = computed from list prices (LiteLLM's price map, else OpenRouter's public model list) and the token
    counts; "none" = no cost available (cost_usd is 0.0 and means nothing). cost_known is True unless "none".
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cost_known: bool
    cost_source: CostSource
    latency_ms: int
    model: str

    @model_validator(mode="after")
    def cost_known_matches_source(self) -> Self:
        if self.cost_known != (self.cost_source != "none"):
            raise ValueError("cost_known must be True exactly when cost_source is 'provider' or 'estimate'")
        return self


class LLMAPIError(Exception):
    """The provider call failed (rate limit, timeout, 5xx, auth...). Not a validation failure.

    `schema_unsupported` is True when the failure means "this model/provider cannot do schema mode" (eval setups
    C and D report that as "not supported", never as a number).
    """

    def __init__(
        self, message: str, *, model: str, retryable: bool, status_code: int | None, schema_unsupported: bool = False
    ) -> None:
        super().__init__(message)
        self.model = model
        self.retryable = retryable
        self.status_code = status_code
        self.schema_unsupported = schema_unsupported


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
    # A 404 counts too when its message says no endpoints / no provider (see NO_PROVIDER_MARKERS).
    if schema_mode and _is_schema_rejection(exc):
        return LLMAPIError(
            f"schema mode unsupported by {model}: {type(exc).__name__}: {exc}",
            model=model,
            retryable=False,
            status_code=getattr(exc, "status_code", None),
            schema_unsupported=True,
        )
    return LLMAPIError(
        f"{type(exc).__name__}: {exc}",
        model=model,
        retryable=isinstance(exc, RETRYABLE_ERRORS),
        status_code=getattr(exc, "status_code", None),
    )


def _is_schema_rejection(exc: APIError) -> bool:
    if isinstance(exc, BadRequestError):
        return _mentions(f"{type(exc).__name__} {exc}", SCHEMA_REJECTION_MARKERS)
    if isinstance(exc, NotFoundError):
        return _mentions(str(exc), NO_PROVIDER_MARKERS)
    return False


def _mentions(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def _build_response(raw: object, model: str, latency_ms: int) -> LLMResponse:
    if not isinstance(raw, litellm.ModelResponse):
        raise TypeError(f"expected a non-streaming ModelResponse, got {type(raw).__name__}")
    choice = raw.choices[0]
    if not isinstance(choice, litellm.Choices):
        raise TypeError(f"expected a non-streaming Choices, got {type(choice).__name__}")
    usage = getattr(raw, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0)
    completion_tokens = getattr(usage, "completion_tokens", 0)
    cost_usd, cost_source = _cost(raw, model, prompt_tokens, completion_tokens)
    return LLMResponse(
        text=choice.message.content or "",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        cost_known=cost_source != "none",
        cost_source=cost_source,
        latency_ms=latency_ms,
        model=model,
    )


def _cost(raw: litellm.ModelResponse, model: str, prompt_tokens: int, completion_tokens: int) -> tuple[float, CostSource]:
    """Cost in USD and where it came from: the provider's own figure, else a list-price estimate, else none (0.0)."""
    reported = getattr(getattr(raw, "usage", None), "cost", None)
    if isinstance(reported, int | float) and not isinstance(reported, bool) and reported >= 0:
        return float(reported), "provider"
    listed = _list_price_cost(raw, model, prompt_tokens, completion_tokens)
    if listed is not None:
        return listed, "estimate"
    logger.warning("cost unknown for model %r: the provider reported none and no list price was found", model)
    return 0.0, "none"


def _list_price_cost(raw: litellm.ModelResponse, model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """LiteLLM's price map first; for an OpenRouter model missing from it, OpenRouter's public list price."""
    try:
        return float(litellm.completion_cost(completion_response=raw))
    except Exception:  # LiteLLM raises a bare Exception for unmapped models
        pass
    if not model.startswith(OPENROUTER_PREFIX):
        return None
    price = fetch_openrouter_prices().get(model.removeprefix(OPENROUTER_PREFIX))
    return None if price is None else prompt_tokens * price[0] + completion_tokens * price[1]


@functools.cache
def fetch_openrouter_prices() -> dict[str, tuple[float, float]]:
    """model slug -> (USD per prompt token, USD per completion token), from OpenRouter's public model list.

    Fetched at most once per process. A failed fetch gives {} (and a warning), so costs fall back to "none".
    """
    try:
        with urllib.request.urlopen(OPENROUTER_MODELS_URL, timeout=PRICE_FETCH_TIMEOUT_SECONDS) as response:
            listed = json.load(response)["data"]
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("could not fetch OpenRouter list prices (%s: %s)", type(exc).__name__, exc)
        return {}
    prices: dict[str, tuple[float, float]] = {}
    for entry in listed:
        try:
            prices[entry["id"]] = (float(entry["pricing"]["prompt"]), float(entry["pricing"]["completion"]))
        except (KeyError, TypeError, ValueError):
            continue  # an entry without usable prices (e.g. a variable-price router)
    return prices
