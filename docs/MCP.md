# MCP server (v2): spec addendum

**Status: implemented and packaged** as `typecast-mcp` (PyPI), in `packages/typecast-mcp/src/typecast_mcp/`
(`validator.py`, `server.py`); install with `uvx typecast-mcp`. Packaging is section 10.
Written 2026-11-06. This document describes a small, validator-only
[Model Context Protocol](https://modelcontextprotocol.io) server for the `soa` package. It is an addendum to the
architecture in [ARCHITECTURE.md](ARCHITECTURE.md) and the schema rules in [DESIGN.md](DESIGN.md).

## 1. Purpose

Let any MCP client check a job-posting JSON object against the `JobPosting` schema and get back the same
one-line-per-error feedback that `soa`'s own retry loop sends to a model. An agent that produced a bad posting can read
the errors, fix its output and call again, without needing `soa`'s extraction pipeline.

The server contains **no LLM, needs no API key and makes no network calls**. It is a pure function of its input:
the same input always gives the same output. The validation, error wording and skill aliases are the ones the
extraction pipeline already uses; the server only exposes them.

## 2. Tools

Two tools. Both are read-only and stateless. Input and output are Pydantic models (strict, `extra="forbid"`, like every
model in `typecast_mcp.schemas`); the MCP SDK derives each tool's input and output schema from them.

### 2.1 `get_job_posting_schema() -> dict`

Returns `JobPosting.model_json_schema()` unchanged: the same schema `soa` puts in its prompt. No arguments.

The result is a JSON object, which the SDK passes through as the tool's structured content without wrapping.

Description shown to the client: *"JSON Schema of a job posting. It does not contain every rule: duplicate skill
names and the intern/experience rule are checked only by validate_job_posting."* That sentence matters: Pydantic
validators are invisible in a JSON schema, so an agent that reads only the schema will still produce postings that fail.

### 2.2 `validate_job_posting(posting: str) -> ValidationResult`

```python
class ValidationResult(BaseModel):          # strict, extra="forbid"
    valid: bool
    errors: list[str]                       # [] when valid
    posting: JobPosting | None              # the validated posting when valid, else None
    # invariant (model_validator): valid <=> errors == [] <=> posting is not None
```

`posting` is the posting as **JSON text**. A parsed object is not accepted (see "Why JSON text only" below).

**Error format.** `errors` has one string per error, exactly as `soa.errors.format_retry_feedback` renders them:
`loc: msg (got: <input, max 60 chars>)`, and a list over its maximum is count-aware. Real output of the current code:

```
seniority: Input should be 'intern', 'junior', 'mid', 'senior' or 'lead' (got: 'wizard')
skills: has 39 items, maximum is 30 - remove 9, starting with required=false ones
```

Other cases the tool returns (also real output):

| Input | `errors` |
|---|---|
| Markdown fence around the JSON | `(root): Invalid JSON: expected value at line 1 column 1 (got: '```json\n{"title": "Backend Engineer", "company": null, ...)` |
| A JSON array | `(root): Input should be an object (got: [])` |
| Missing and unknown fields | `remote_ok: Extra inputs are not permitted (got: True)`, `company: Field required (got: {'title': 'Backend Engineer', 'seniority': 'mid', 'min_ye...)` |
| Duplicate skills | `skills: Value error, duplicate skill names: 'SQL', 'sql'. Merge each group into one entry (got: [...` |
| Intern with 2 years | `(root): Value error, seniority is 'intern' but min_years_experience is 2; set min_years_experience to null or 0, or choose a different seniority (got: ...` |

A schema-valid posting that breaks a model validator reports only the validator error, because Pydantic runs
`mode="after"` validators only once every field is valid. One call can therefore need a second call to surface the
next problem. This is the same behaviour the retry loop has, and it is documented in the tool description.

**Why JSON text only.** The tool validates with `JobPosting.model_validate_json(posting)`, and the argument is a `str`,
for two reasons:

1. **The models are strict.** `StrictModel` sets `strict=True`, and under `model_validate(dict)` strict mode rejects a
   plain string for a `StrEnum` field: `"seniority": "mid"` fails in Python mode although it is valid JSON. Only
   JSON-mode validation (`model_validate_json`) accepts enum strings, so the input has to reach Pydantic as JSON text.
2. **Malformed JSON keeps its error.** With text input, a fenced or truncated reply produces the same
   `Invalid JSON: ...` line the retry loop gives a model. If the client had to parse the JSON first, that mistake would
   fail in the client and the agent would never see the feedback.

If dict support is added later, the dict must be re-serialised with `json.dumps` and validated with
`model_validate_json`, never passed to `model_validate`.

**Failure is a result, not an exception.** Invalid input, oversize input and unparseable text all return
`valid=false` with an explanatory `errors` line. The tool does not raise for bad data. (An argument of the wrong type
altogether, such as a number or an object, is rejected by the SDK before the function runs and comes back as a tool
error in the SDK's own wording; see "Verified during the build" in section 6.)

#### Decision: raw validation only, no post-processing

`validate_job_posting` validates and returns the posting **as given**. It does not run `tools.postprocess`
(skill normalisation, duplicate merging, location fill-in). Reasons:

1. **It answers the question the caller asked.** The caller wants to know whether *its own output* is valid.
   `soa`'s retry loop makes the same choice: validity is measured on the model's raw output, and post-processing runs
   afterwards.
2. **Post-processing can change the posting.** It renames skills and merges entries. A tool that says "valid" but
   returns a different object than the one submitted invites the agent to trust a rewrite it did not ask for.
3. **Part of it cannot run.** `classify_location` reads the first lines of the *source job description*, which the
   server never sees. Only `normalize_skill` could run, so the result would be half of `postprocess`.
4. **It keeps the error contract clean.** `postprocess` can raise `ToolResponseError` on its own bug; the validator
   tool should have no failure mode other than "your input is invalid".

Skill normalisation is not offered by the server at all in this version (see "Possible later" in section 8).

## 3. Transport and entry point

- **stdio only.** The client starts the server as a subprocess and talks over its stdin and stdout.
- Entry point: `typecast-mcp`, declared in `packages/typecast-mcp/pyproject.toml` as
  `typecast-mcp = "typecast_mcp.server:main"`. The command and the package share a name, so `uvx typecast-mcp` works.
- `mcp` is a hard dependency of `typecast-mcp`, so there is no missing-SDK path (the earlier `soa[mcp]` extra and its
  exit-2 message are gone, see section 10).

## 4. Dependency

Verified on 2026-11-06 against PyPI, the SDK README and the SDK's v2 migration guide:

| | |
|---|---|
| Package | `mcp` on PyPI (official SDK, repository `modelcontextprotocol/python-sdk`) |
| Latest | 2.3.0. The 1.x line (latest seen: 1.30.0, 2026-09-07) is in maintenance and gets only critical and security fixes |
| Python | `>=3.10`, so compatible with `soa`'s `>=3.11` |
| Server API | `MCPServer` (`from mcp.server import MCPServer`), decorator `@mcp.tool()`. v1's `FastMCP` was renamed `MCPServer` in v2 and the old import path no longer exists. A lowlevel `Server` also exists, documented as an advanced option |
| Tool results | Return type is the output schema. Pydantic models, dicts and dataclasses become structured content as a JSON object; scalars, lists and unions are wrapped as `{"result": ...}`. A model must be returned as a model for its schema to be sent |
| Tests | `from mcp import Client`; `Client(server)` connects in memory with no transport |

**Decision: `MCPServer` (high-level API), pinned as `mcp>=2.3,<3`.** The tools are two plain functions with typed
arguments and Pydantic results, which is exactly what the high-level API does. The lowlevel `Server` would mean writing
request routing and result wrapping by hand for no benefit. Pinning to 2.x avoids building on the 1.x line that is
now maintenance-only.

**Decision (superseded 2026-11-06, see section 10): an optional extra, `soa[mcp]`, not a core dependency.** `mcp` 2.3.0 pulls in a web stack that
`soa extract` and `soa eval` never use: `starlette`, `uvicorn`, `sse-starlette`, `httpx2`, `pyjwt[crypto]`,
`python-multipart`, `jsonschema`, `opentelemetry-api` and, on Windows, `pywin32`. A user who wants only the CLI
should not install that, and each added package adds to the audit surface of the lockfile. It requires `pydantic>=2.12`,
which `soa` already exceeds (`>=2.13.5`).

Consequences of the extra:

- `uv.lock` still resolves the extra, so `pip-audit` on the lock covers it.
- CI installs every extra: the sync step becomes `uv sync --locked --all-extras`, so the MCP tests run there.
- `tests/test_mcp_server.py` begins with `pytest.importorskip("mcp")`, so a plain `uv sync` (no extra) skips it and still
  runs everything else. `tests/test_mcp_tools.py` does not need the SDK and always runs.
- Whether `mcp`'s `httpx2` and `soa`'s `litellm` (which uses `httpx`) resolve side by side has not been tried. The first
  build task is `uv add --optional mcp "mcp>=2.3,<3"` followed by `uv lock`, before any code is written. **If the
  lock does not resolve, stop and report the conflict; do not pin around it.**

## 5. Reuse: nothing is reimplemented

| Need | Existing code the server calls |
|---|---|
| Schema | `typecast_mcp.schemas.JobPosting.model_json_schema()` |
| Validation | `typecast_mcp.schemas.JobPosting.model_validate_json(...)` |
| Error lines | `typecast_mcp.errors.format_retry_feedback(exc)`, the function that builds the retry prompt |

New code is limited to: the size guard, the `ValidationResult` model, and the `MCPServer` wiring. Two modules:

- `typecast_mcp/validator.py`: the two functions and the result model. **Does not import `mcp`**, so it is testable
  without the SDK.
- `typecast_mcp/server.py`: creates the `MCPServer`, registers the two functions, defines `main()`.

Nothing in `typecast_mcp` imports `soa` or `litellm`. That keeps the "no LLM, no key, no network" promise checkable
(a test asserts `litellm` is absent from `sys.modules`), keeps start-up fast, and avoids LiteLLM's habit of printing a
banner to stdout, which would break the transport (section 6).

The same constraints as the rest of `src/` apply: `mypy --strict`, functions of 40 lines or fewer, Pydantic models at
every boundary. The only input is a `str`. The one `dict` that leaves the server is the JSON Schema from `get_job_posting_schema`,
which is JSON by nature and comes straight from `model_json_schema()`.

## 6. Limits, safety and tests

### Input limits

| Limit | Value | Behaviour when exceeded |
|---|---|---|
| `validate_job_posting` payload | 20,000 characters of `posting` | Returns `valid=false`, `errors=["(root): payload is 23000 characters, maximum is 20000"]`; nothing is parsed |
| `validate_job_posting` error lines | 20 lines | The first 20 are returned, then one line `... N more errors omitted`. Without a cap, a 20 KB payload of unknown keys returned about 2,000 lines (92 KB). The cap is in `validator.py` only; `typecast_mcp.errors` and the retry loop are unchanged |

A minimal valid posting is about 240 characters and one at every maximum is about 2,700, so 20,000 leaves a wide margin.
`company` has no length limit in the schema, which is a further reason for the guard. Deeply nested input is bounded by
Pydantic's own recursion limit.

### stdout discipline

On a stdio transport, **stdout carries only MCP protocol messages**. One stray line (a `print`, a library banner, an
import-time message) can make the client drop the connection, and some clients then show a server with no tools. The
SDK diverts some stray stdout to stderr, but its documentation calls this best-effort. The server therefore:

- never calls `print`; diagnostics go through `logging`, configured in `main()` with `stream=sys.stderr`;
- imports nothing that prints (`soa.llm` and `litellm` stay out, section 5);
- has no module-level output.

### Tests

No test uses the network, an API key or a real LLM, and CI stays key-free.

1. **Tool functions called directly** (`tests/test_mcp_tools.py`, no SDK, always runs): a valid posting returns
   `valid=true` with the parsed posting (including a plain enum string such as `"seniority": "mid"`); each row of the
   table in 2.2 produces its documented line; the count-aware skills line appears at 39 skills; the
   `valid <=> errors == [] <=> posting is not None` invariant; payload at and over the limit; the schema equals
   `JobPosting.model_json_schema()`.
2. **In-memory client and server** (`tests/test_mcp_server.py`; also checks the server reports name `typecast` and
   the package version):
   `async with Client(server) as c:` then `await c.call_tool("validate_job_posting", {...})`. Checks that `tools/list`
   shows exactly the two tools, that the structured content of a valid and an invalid call matches `ValidationResult`,
   and what a wrong-typed argument returns (see "Verified during the build" below). Uses the project's existing `anyio`
   pytest plugin (`@pytest.mark.anyio`, backend `asyncio`).
3. **stdout cleanliness**, two checks:
   - Start `python -m typecast_mcp.server` as a subprocess, send `initialize`, `tools/list` and a `tools/call` as
     newline-delimited JSON-RPC messages, and assert that **every** line on stdout parses as JSON with `"jsonrpc"` and
     that exactly the three replies arrived.
   - Import `typecast_mcp.server` in a subprocess and assert stdout is empty and `litellm` is not in `sys.modules`.

### Verified during the build (mcp 2.3.0)

- **A wrong-typed argument is a tool error, not a result.** `validate_job_posting` called with `{"posting": 123}`,
  `{"posting": null}`, `{"posting": {"title": "x"}}` (a parsed object) or `{}` is rejected by the SDK before the function
  runs. The client gets `is_error=true`, no structured content, and one text block starting
  `Error executing tool validate_job_posting: 1 validation error for validate_job_postingArguments`, followed by
  pydantic's own multi-line message (`posting` / `Input should be a valid string [type=string_type, ...]`, or
  `Field required [type=missing, ...]`, plus a docs URL). It is **not** in the one-line `loc: msg (got: ...)` format,
  because it comes from the SDK's argument model, not from `soa`. The tool's input schema (`posting`: string, required)
  tells a client what to send. Only text that is not valid JSON, or not a valid posting, gets the one-line feedback.
  A parsed object is therefore refused, as decided in 2.2.
- **Run call:** `MCPServer.run("stdio")`; `stdio` is also the default transport.
- **Request size:** the SDK documents no maximum request size, so the 20,000-character guard is ours alone.
- **stdin must stay open until the replies have been read.** A client that closes stdin right after writing its last
  request makes the server shut down with that reply still in flight (seen: replies 1 and 2 arrived, 3 did not). Real
  clients keep the pipe open, but a script that pipes a file into `typecast-mcp` must wait for its replies first.
- **The SDK logs to stderr, not stdout.** A rejected call logs an INFO line (`Tool 'validate_job_posting' rejected
  arguments`) with the SDK's own handler; `main()` additionally points `logging` at stderr.

## 7. Manual check

From a clone, `uv sync` installs everything (the workspace includes `typecast-mcp`).

**MCP Inspector** (the SDK's own `mcp dev` command needs a further extra, `mcp[cli]`, which this server does not use;
the standalone Inspector does not):

```
npx @modelcontextprotocol/inspector uvx typecast-mcp
```

In the Inspector: *Tools* should list the two tools; run `get_job_posting_schema`; run `validate_job_posting` with
`{"posting": "{\"title\": \"x\"}"}` and expect `valid: false` with a `company: Field required ...` line.

**Claude Desktop**: add to `claude_desktop_config.json` (Windows: `%APPDATA%\Claude\claude_desktop_config.json`),
then restart the app:

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

If the app cannot find `uvx`, put the full path to `uvx.exe` in `command`. Then ask Claude to "validate this job posting
JSON with typecast" and paste a posting with one wrong field. The command lines in this section are untested
until the server exists.

## 8. Out of scope

- An `extract_job_posting` tool or anything that calls an LLM.
- HTTP, SSE or any transport other than stdio.
- Authentication, rate limiting and multi-user hosting.
- MCP resources or prompts (the schema is a tool, not a resource, so every client can use it).
- Post-processing inside `validate_job_posting` (2.2).
- Accepting an already-parsed object as `posting` (see "Why JSON text only" in 2.2).

### Possible later

- **`normalize_skills(names)`**, a tool that maps skill names through the existing alias map (`"k8s"` gives
  `"Kubernetes"`). Cut from this version because the alias map has only 40 canonical names and 110 spellings, so most
  real skills would come back unknown and the tool would rarely tell an agent anything useful. It would be worth
  revisiting if the alias map grows.

## 9. Open questions

None. The decisions above are settled and the build-time checks are recorded under "Verified during the build" in
section 6. The README links to this document from its MCP section.

## 10. Packaging (2026-11-06)

Published to PyPI as **`typecast-mcp`**, a separate package in a uv workspace, not as `soa`.

| Decision | Choice | Why |
|---|---|---|
| Distribution name | `typecast-mcp` | `soa` and `typecast` are already taken on PyPI by unrelated projects |
| What is published | Only the validator: `schemas.py`, the error-line functions, `validator.py`, `server.py` | These import only `pydantic` and each other. Publishing `soa` would install LiteLLM, openai, typer and dotenv for a server that never imports them: about 79 packages / 259 MB, against 29 packages / 33 MB now (measured in a clean venv) |
| Layout | uv workspace member `packages/typecast-mcp/`; `soa` depends on it via `[tool.uv.sources]` | One source of truth for `JobPosting`; the eval harness and the server cannot drift |
| `mcp` dependency | Hard dependency of `typecast-mcp` | `uvx typecast-mcp` must work with no extra. This reverses the `soa[mcp]` decision in section 4: `soa` now always installs the MCP SDK. Accepted because `soa` itself is not published and already pulls a large stack through LiteLLM |
| `redact_keys` | Stays in `soa.errors` | Only the CLI and eval use it; the server has no secrets |
| Server name | `typecast` (was `soa-validator`), version read from the package metadata (`importlib.metadata.version("typecast-mcp")`) | Matches the package and the client config key |
| License | MIT, `LICENSE` at the root and copied into the package (PEP 639 forbids `..` paths) | Without a license the public code was not usable |
| Registry | `server.json` at the root, name `io.github.nilayp-codes/typecast`; the PyPI README carries the `mcp-name:` marker | The MCP Registry checks the marker in the published README to prove ownership |
| Release | `.github/workflows/publish.yml`: manual run goes to TestPyPI, tag `typecast-mcp-v<version>` goes to PyPI; trusted publishing, no stored token; tests, mypy and a tag/package/server.json version check run first | Steps in [RELEASING.md](RELEASING.md) |

Verified before the first upload: wheel builds and passes `twine check`; a clean `uvx --from <wheel> typecast-mcp`
answers `initialize` (server `typecast` 0.1.0), lists the two tools, accepts a valid posting and returns
`skills: has 70 items, maximum is 30 - remove 40, ...` for a 70-skill one; no litellm, openai, typer or dotenv in the
environment.
