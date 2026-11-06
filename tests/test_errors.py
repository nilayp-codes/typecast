"""redact_keys: which environment values get masked."""

import pytest

from soa.errors import redact_keys


@pytest.mark.parametrize("name", ["OPENROUTER_API_KEY", "SERVICE_KEY", "HF_TOKEN", "GITHUB_TOKEN"])
def test_values_of_key_and_token_variables_are_masked(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv(name, "value-1234567890")
    assert redact_keys("bad credential value-1234567890 sent") == "bad credential *** sent"


def test_a_value_shorter_than_8_characters_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHORT_TOKEN", "abc1234")  # 7 characters
    monkeypatch.setenv("SHORT_KEY", "xyz9876")
    assert redact_keys("abc1234 xyz9876") == "abc1234 xyz9876"


def test_other_variables_are_not_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOA_MODEL_PRIMARY", "openrouter/qwen/qwen-2.5-7b-instruct")
    monkeypatch.setenv("KEYBOARD_LAYOUT", "layout-1234567890")  # contains KEY but does not end in _KEY
    assert redact_keys("openrouter/qwen/qwen-2.5-7b-instruct layout-1234567890") == (
        "openrouter/qwen/qwen-2.5-7b-instruct layout-1234567890"
    )


def test_every_occurrence_is_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf-secret-12345")
    assert redact_keys("hf-secret-12345 and hf-secret-12345") == "*** and ***"
