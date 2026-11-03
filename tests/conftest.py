"""Test setup: keep LiteLLM offline (bundled price map, no fetch from GitHub at import)."""

import os

import pytest

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


@pytest.fixture(autouse=True)
def no_price_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """llm.call must never fetch OpenRouter's price list in a test (rule 13: tests do not touch the network)."""
    from soa import llm

    monkeypatch.setattr(llm, "fetch_openrouter_prices", lambda: {})
