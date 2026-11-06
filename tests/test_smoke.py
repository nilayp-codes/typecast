"""Smoke test: every soa and typecast_mcp module imports cleanly."""

import importlib

import pytest

MODULES = [
    "soa.tools", "soa.llm", "soa.agent", "soa.logger", "soa.eval", "soa.cli", "soa.errors",
    "typecast_mcp.schemas", "typecast_mcp.errors", "typecast_mcp.validator", "typecast_mcp.server",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)
