"""llm.call() with litellm.completion stubbed. No real provider is ever contacted."""

from typing import Any

import litellm
import pytest
from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
    Timeout,
    UnsupportedParamsError,
)
from pydantic import ValidationError

from soa import llm
from soa.llm import LLMAPIError, LLMResponse, Message
from soa.schemas import JobPosting

MESSAGES = [Message(role="system", content="be brief"), Message(role="user", content="hi")]


def fake_response(content: str | None = '{"ok": true}', model: str = "gpt-4o-mini") -> litellm.ModelResponse:
    return litellm.ModelResponse(
        model=model,
        choices=[{"message": {"role": "assistant", "content": content}, "finish_reason": "stop", "index": 0}],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def stub_completion(monkeypatch: pytest.MonkeyPatch, result: object) -> dict[str, Any]:
    """Replace litellm.completion; return a dict that captures the kwargs it was called with."""
    seen: dict[str, Any] = {}

    def fake(**kwargs: Any) -> object:
        seen.update(kwargs)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(litellm, "completion", fake)
    return seen


def test_call_returns_text_tokens_cost_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_completion(monkeypatch, fake_response())
    resp = llm.call(MESSAGES, "gpt-4o-mini")
    assert resp.text == '{"ok": true}'
    assert (resp.prompt_tokens, resp.completion_tokens) == (10, 5)
    assert resp.model == "gpt-4o-mini"
    assert resp.cost_known is True
    assert resp.cost_usd > 0
    assert resp.latency_ms >= 0


def test_messages_are_sent_as_role_content_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = stub_completion(monkeypatch, fake_response())
    llm.call(MESSAGES, "gpt-4o-mini")
    assert seen["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert seen["model"] == "gpt-4o-mini"


def test_no_response_model_sends_no_response_format(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = stub_completion(monkeypatch, fake_response())
    llm.call(MESSAGES, "gpt-4o-mini")
    assert seen["response_format"] is None


def test_response_model_puts_its_schema_in_response_format(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = stub_completion(monkeypatch, fake_response())
    llm.call(MESSAGES, "gpt-4o-mini", response_model=JobPosting)
    assert seen["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "JobPosting", "schema": JobPosting.model_json_schema(), "strict": True},
    }
    assert seen["response_format"]["json_schema"]["schema"]["additionalProperties"] is False  # really from the model


def test_plain_json_object_mode_is_gone() -> None:
    with pytest.raises(TypeError):
        llm.call(MESSAGES, "gpt-4o-mini", json_mode=True)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "exc",
    [
        BadRequestError("response_format json_schema is not supported", model="m", llm_provider="openrouter"),
        UnsupportedParamsError("openrouter does not support parameters: ['response_format']", llm_provider="openrouter", model="m"),
    ],
)
def test_provider_rejecting_schema_mode_raises_non_retryable_unsupported(
    monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    stub_completion(monkeypatch, exc)
    with pytest.raises(LLMAPIError) as caught:
        llm.call(MESSAGES, "openrouter/m", response_model=JobPosting)
    assert str(caught.value).startswith("schema mode unsupported by openrouter/m")
    assert caught.value.retryable is False
    assert caught.value.status_code == 400
    assert caught.value.__cause__ is exc


def test_a_400_in_schema_mode_with_an_unrelated_message_stays_generic(monkeypatch: pytest.MonkeyPatch) -> None:
    exc = BadRequestError("maximum context length is 8192 tokens", model="m", llm_provider="openrouter")
    stub_completion(monkeypatch, exc)
    with pytest.raises(LLMAPIError) as caught:
        llm.call(MESSAGES, "openrouter/m", response_model=JobPosting)
    assert "schema mode" not in str(caught.value)
    assert "maximum context length is 8192 tokens" in str(caught.value)  # the provider's text is kept
    assert caught.value.retryable is False and caught.value.status_code == 400


def test_openrouter_schema_mode_requires_all_parameters_to_be_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = stub_completion(monkeypatch, fake_response())
    llm.call(MESSAGES, "openrouter/qwen/qwen-2.5-7b-instruct", response_model=JobPosting)
    assert seen["extra_body"] == {"provider": {"require_parameters": True}}


def test_require_parameters_is_sent_only_for_openrouter_schema_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    plain = stub_completion(monkeypatch, fake_response())
    llm.call(MESSAGES, "openrouter/qwen/qwen-2.5-7b-instruct")  # no response_model
    assert "extra_body" not in plain
    other = stub_completion(monkeypatch, fake_response())
    llm.call(MESSAGES, "gpt-4o-mini", response_model=JobPosting)  # not OpenRouter
    assert "extra_body" not in other


def test_a_400_without_schema_mode_is_an_ordinary_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_completion(monkeypatch, BadRequestError("prompt too long", model="m", llm_provider="openrouter"))
    with pytest.raises(LLMAPIError) as caught:
        llm.call(MESSAGES, "openrouter/m")
    assert "schema mode" not in str(caught.value) and caught.value.retryable is False


def test_other_errors_in_schema_mode_keep_their_retry_class(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_completion(monkeypatch, RateLimitError("slow down", llm_provider="openrouter", model="m"))
    with pytest.raises(LLMAPIError) as caught:
        llm.call(MESSAGES, "openrouter/m", response_model=JobPosting)
    assert caught.value.retryable is True and "schema mode" not in str(caught.value)


def test_unknown_model_costs_zero_and_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_completion(monkeypatch, fake_response(model="gemini/not-a-real-model"))
    resp = llm.call(MESSAGES, "gemini/not-a-real-model")
    assert (resp.cost_usd, resp.cost_known) == (0.0, False)
    assert resp.text == '{"ok": true}'


def test_empty_content_becomes_empty_text(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_completion(monkeypatch, fake_response(content=None))
    assert llm.call(MESSAGES, "gpt-4o-mini").text == ""


@pytest.mark.parametrize(
    ("exc", "retryable"),
    [
        (RateLimitError("slow down", llm_provider="gemini", model="m"), True),
        (Timeout("timed out", model="m", llm_provider="gemini"), True),
        (InternalServerError("boom", llm_provider="gemini", model="m"), True),
        (AuthenticationError("bad key", llm_provider="gemini", model="m"), False),
    ],
)
def test_litellm_errors_become_llm_api_error(
    monkeypatch: pytest.MonkeyPatch, exc: Exception, retryable: bool
) -> None:
    stub_completion(monkeypatch, exc)
    with pytest.raises(LLMAPIError) as caught:
        llm.call(MESSAGES, "gemini/m")
    assert caught.value.retryable is retryable
    assert caught.value.model == "gemini/m"
    assert type(exc).__name__ in str(caught.value)
    assert caught.value.__cause__ is exc


def test_message_rejects_unknown_role_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Message(role="tool", content="x")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Message.model_validate({"role": "user", "content": "x", "name": "bob"})


def test_llm_response_is_strict() -> None:
    with pytest.raises(ValidationError):
        LLMResponse.model_validate(
            {"text": "", "prompt_tokens": "10", "completion_tokens": 0, "cost_usd": 0.0,
             "cost_known": True, "latency_ms": 1, "model": "m"}
        )
