# Releasing typecast-mcp

`typecast-mcp` (in `packages/typecast-mcp/`) is the only package this repo publishes. `soa` is not published.

## One-time setup

1. **PyPI and TestPyPI accounts** with 2FA: <https://pypi.org/account/register/>, <https://test.pypi.org/account/register/>.
2. **Trusted publishers** (no API token needed). On each site: *Your projects → Publishing → Add a new pending
   publisher*, with:

   | Field | Value |
   |---|---|
   | PyPI project name | `typecast-mcp` |
   | Owner | `nilayp-codes` |
   | Repository | `typecast` |
   | Workflow | `publish.yml` |
   | Environment | `pypi` on PyPI, `testpypi` on TestPyPI |

3. **GitHub environments**: repo *Settings → Environments*, create `pypi` and `testpypi`. Optionally add yourself as
   a required reviewer on `pypi`, so a tag cannot publish without your click.

## Each release

1. Bump the version in **two places**, both equal: `packages/typecast-mcp/pyproject.toml` and both `version` fields in
   `server.json`. (`typecast_mcp.__version__` and the server's reported version are read from the installed package
   metadata, so there is nothing to change in `__init__.py`.) Run `uv lock`. The workflow refuses to publish if the tag, the package and `server.json` disagree.
2. Tests and types: `uv run pytest` and `uv run mypy --strict src/ packages/typecast-mcp/src/`.
3. **TestPyPI**: GitHub *Actions → Publish typecast-mcp → Run workflow*. Then check the uploaded file, not just the
   build. Download only the wheel from TestPyPI, and let its dependencies come from real PyPI (TestPyPI copies of
   `mcp` or `pydantic` can be anyone's):

   ```
   uv run --no-project --with pip python -m pip download --no-deps -d testpypi-dl -i https://test.pypi.org/simple/ typecast-mcp==0.1.0
   uvx --from ./testpypi-dl/typecast_mcp-0.1.0-py3-none-any.whl typecast-mcp
   ```

   It should start and wait silently on stdin (Ctrl+C to stop). Delete `testpypi-dl/` afterwards.
4. **PyPI**: `git tag typecast-mcp-v0.1.0 && git push origin typecast-mcp-v0.1.0`. PyPI versions are immutable: a
   broken upload is fixed only by a new version.
5. Check in a real client: `claude mcp add typecast -- uvx typecast-mcp`, then `/mcp`.

## MCP Registry (after the PyPI upload)

The registry checks that the PyPI README of the listed version contains `mcp-name: io.github.nilayp-codes/typecast`
(it is in `packages/typecast-mcp/README.md` as an HTML comment), and that the version in `server.json` is on PyPI.

```
mcp-publisher login github
mcp-publisher publish
```

Install `mcp-publisher` from <https://github.com/modelcontextprotocol/registry/releases>. `server.json` `description`
must stay at 100 characters or fewer.
