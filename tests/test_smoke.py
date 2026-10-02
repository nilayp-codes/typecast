"""Smoke test: every soa module imports cleanly."""

import importlib

import pytest

MODULES = ["schemas", "tools", "llm", "agent", "logger", "eval", "cli"]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(f"soa.{name}")
