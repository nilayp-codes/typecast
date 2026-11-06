"""The MCP server (docs/MCP.md sections 3 and 6): in-memory client, stdout cleanliness."""

import json
import queue
import subprocess
import sys
import threading
from collections.abc import AsyncIterator
from typing import Any

import pytest
from mcp import Client
from mcp.types import TextContent

from typecast_mcp import __version__
from typecast_mcp.schemas import JobPosting
from typecast_mcp.server import build_server
from typecast_mcp.validator import ValidationResult

VALID: dict[str, Any] = {
    "title": "Backend Engineer",
    "company": None,
    "seniority": "mid",
    "min_years_experience": 3,
    "skills": [{"name": "Python", "required": True}],
    "location": {"city": None, "country": None, "mode": "unspecified"},
    "salary_mentioned": False,
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client() -> AsyncIterator[Client]:
    async with Client(build_server(), raise_exceptions=True) as c:
        yield c


def as_result(structured: dict[str, Any]) -> ValidationResult:
    """Strict models take enum strings only from JSON text, so go through JSON (the rule docs/MCP.md 2.2 explains)."""
    return ValidationResult.model_validate_json(json.dumps(structured))


@pytest.mark.anyio
async def test_server_reports_its_name_and_package_version(client: Client) -> None:
    info = client.server_info
    assert (info.name, info.version) == ("typecast", __version__)


@pytest.mark.anyio
async def test_exactly_the_two_tools_are_listed(client: Client) -> None:
    listed = await client.list_tools()
    assert sorted(tool.name for tool in listed.tools) == ["get_job_posting_schema", "validate_job_posting"]


@pytest.mark.anyio
async def test_get_job_posting_schema_returns_the_models_schema(client: Client) -> None:
    result = await client.call_tool("get_job_posting_schema", {})
    assert not result.is_error
    assert result.structured_content == JobPosting.model_json_schema()


@pytest.mark.anyio
async def test_validate_accepts_a_valid_posting(client: Client) -> None:
    result = await client.call_tool("validate_job_posting", {"posting": json.dumps(VALID)})
    assert not result.is_error and result.structured_content is not None
    parsed = as_result(result.structured_content)
    assert parsed.valid and parsed.errors == [] and parsed.posting == JobPosting.model_validate_json(json.dumps(VALID))


@pytest.mark.anyio
async def test_validate_reports_errors_as_a_result_not_a_tool_error(client: Client) -> None:
    skills = [{"name": f"S{i}", "required": True} for i in range(39)]
    result = await client.call_tool("validate_job_posting", {"posting": json.dumps({**VALID, "skills": skills})})
    assert not result.is_error and result.structured_content is not None
    parsed = as_result(result.structured_content)
    assert not parsed.valid and parsed.posting is None
    assert parsed.errors == ["skills: has 39 items, maximum is 30 - remove 9, starting with required=false ones"]


@pytest.mark.anyio
@pytest.mark.parametrize("arguments", [{"posting": 123}, {"posting": None}, {"posting": {"title": "x"}}, {}])
async def test_a_wrong_typed_argument_is_a_tool_error_with_pydantics_text(
    client: Client, arguments: dict[str, Any]
) -> None:
    # The SDK rejects the call before validate_job_posting runs: is_error=True, no structured content, and the text is
    # pydantic's own multi-line message about the SDK's argument model, NOT our one-line format (docs/MCP.md section 6).
    result = await client.call_tool("validate_job_posting", arguments)
    assert result.is_error and result.structured_content is None
    first = result.content[0]
    assert isinstance(first, TextContent)
    assert first.text.startswith("Error executing tool validate_job_posting: 1 validation error for ")
    assert "posting" in first.text and ("Input should be a valid string" in first.text or "Field required" in first.text)


def run_stdio_session(messages: list[dict[str, Any]], replies_expected: int) -> tuple[list[str], int]:
    """Start the server as a real subprocess on stdio, send newline-delimited JSON-RPC, return (stdout lines, exit code).

    stdin stays open until every expected reply has arrived (closing it early makes the server shut down with replies
    still in flight), then it is closed and any further stdout is collected too: stray output would show up there.
    """
    proc = subprocess.Popen(
        [sys.executable, "-m", "typecast_mcp.server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    assert proc.stdin is not None and proc.stdout is not None
    lines: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(target=lambda: [lines.put(line) for line in proc.stdout], daemon=True)  # type: ignore[union-attr]
    reader.start()
    try:
        proc.stdin.write("".join(json.dumps(m) + "\n" for m in messages))
        proc.stdin.flush()
        collected = [lines.get(timeout=60) for _ in range(replies_expected)]
        proc.stdin.close()
        code = proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    reader.join(timeout=10)
    while not lines.empty():
        collected.append(lines.get_nowait())
    return collected, code


def test_stdout_carries_only_json_rpc_messages() -> None:
    import mcp_types

    lines, exit_code = run_stdio_session(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
                "name": "validate_job_posting", "arguments": {"posting": json.dumps(VALID)}}},
        ],
        replies_expected=3,
    )
    assert mcp_types.LATEST_PROTOCOL_VERSION  # the SDK is the 2.x line the spec was written against
    replies = [json.loads(line) for line in lines]  # every line must be JSON
    assert all(reply["jsonrpc"] == "2.0" for reply in replies)
    assert sorted(reply["id"] for reply in replies) == [1, 2, 3]  # replies only: no banner, no stray line
    tools = next(r for r in replies if r["id"] == 2)["result"]["tools"]
    assert sorted(t["name"] for t in tools) == ["get_job_posting_schema", "validate_job_posting"]
    assert next(r for r in replies if r["id"] == 3)["result"]["structuredContent"]["valid"] is True
    assert exit_code == 0


def test_importing_and_building_the_server_prints_nothing_and_does_not_load_litellm() -> None:
    code = (
        "import sys; from typecast_mcp.server import build_server; build_server(); "
        "print('LOADED' if 'litellm' in sys.modules else 'clean', file=sys.stderr)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout == ""
    assert done.stderr.strip().endswith("clean")
