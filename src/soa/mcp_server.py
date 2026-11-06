"""`soa-mcp`: the validator-only MCP server (docs/MCP.md). stdio transport, two tools, no LLM, no key, no network.

On a stdio transport stdout carries only MCP protocol messages, so this module never prints; diagnostics go through
`logging`, which `main()` points at stderr. `mcp` is imported inside `build_server` so that `soa-mcp` can say
"install soa[mcp]" instead of failing with a traceback when the extra is missing. Nothing here imports `soa.llm`/litellm.
"""

import logging
import sys
from typing import TYPE_CHECKING, Any

from soa.mcp_tools import get_job_posting_schema, validate_job_posting

if TYPE_CHECKING:
    from mcp.server import MCPServer

SERVER_NAME = "soa-validator"
EXIT_MISSING_EXTRA = 2

SCHEMA_DESCRIPTION = (
    "JSON Schema of a job posting. It does not contain every rule: duplicate skill names and the "
    "intern/experience rule are checked only by validate_job_posting."
)
VALIDATE_DESCRIPTION = (
    'Pass the ENTIRE job posting as ONE argument, `posting`: a JSON string, e.g. {"posting": "{\\"title\\": '
    '\\"...\\", ...}"}. Do not pass fields as separate arguments. '
    "Validate a job-posting JSON text against the JobPosting schema. Returns valid, a list of error lines "
    "('field: problem (got: value)'), and the parsed posting when valid. Fix every line and call again: the "
    "duplicate-skills and intern/experience rules are reported only once every field is valid, so a fix can "
    "surface one more error. Pass the JSON as text; no post-processing is applied."
)


def build_server() -> "MCPServer[Any]":
    """The MCP server with its two tools registered."""
    from mcp.server import MCPServer

    server: MCPServer[Any] = MCPServer(SERVER_NAME)
    server.tool(name="get_job_posting_schema", description=SCHEMA_DESCRIPTION)(get_job_posting_schema)
    server.tool(name="validate_job_posting", description=VALIDATE_DESCRIPTION)(validate_job_posting)
    return server


def main() -> None:
    """Entry point of `soa-mcp`: serve over stdio until the client disconnects."""
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    try:
        server = build_server()
    except ImportError:
        sys.stderr.write('soa-mcp needs the \'mcp\' extra: pip install "soa[mcp]"\n')
        raise SystemExit(EXIT_MISSING_EXTRA) from None
    server.run("stdio")


if __name__ == "__main__":
    main()
