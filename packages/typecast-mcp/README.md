# typecast-mcp

<!-- mcp-name: io.github.nilayp-codes/typecast -->

A validator-only [MCP](https://modelcontextprotocol.io) server. It checks a job-posting JSON against a strict
Pydantic schema and returns one line per error, in a form an agent can act on:

```
skills: has 70 items, maximum is 30 - remove 40, starting with required=false ones
```

No LLM, no API key, no network calls. Dependencies: `mcp` and `pydantic`.

It is the validation layer of [Typecast](https://github.com/nilayp-codes/typecast), a project that measures how often
LLMs fail to return valid structured output and what fixes it.

## Tools

- `get_job_posting_schema`: the JSON Schema of a job posting. It does not carry every rule: duplicate skill names
  and the intern/experience rule are only checked by the validator.
- `validate_job_posting`: takes one argument, `posting`, the whole job-posting JSON as ONE string. Returns `valid`,
  a list of error lines (`field: problem (got: value)`), and the parsed posting when valid.

## Install

Requires [uv](https://docs.astral.sh/uv/). Add this to your MCP client config (Claude Desktop, Cursor, and others):

```json
{
  "mcpServers": {
    "typecast": {
      "command": "uvx",
      "args": ["typecast-mcp"]
    }
  }
}
```

Claude Code:

```
claude mcp add typecast -- uvx typecast-mcp
```

On Windows, if the client cannot find `uvx`, put the full path to `uvx.exe` in `command`. Without uv:
`pip install typecast-mcp`, then use `typecast-mcp` as the command.

## What it does not check

Whether the values are correct. A posting can be valid and still have the wrong seniority or the wrong skills. See
[Why not just use native structured outputs?](https://github.com/nilayp-codes/typecast#why-not-just-use-native-structured-outputs)

Full spec, limits and design decisions: [docs/MCP.md](https://github.com/nilayp-codes/typecast/blob/main/docs/MCP.md).

## License

MIT
