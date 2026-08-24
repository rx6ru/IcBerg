"""P3.8 (`.devdocs/PHASE3_GATES.md`): no integration surface exposes a raw executor /
database connection / cursor that would let a caller run ungoverned SQL, bypassing
`Gateway`/`GovernanceGate` entirely.

An attribute-scan gate over every object a caller of the three surfaces (SDK, MCP,
LangGraph) can get their hands on: the SDK's one-shot `govern()` result, a
`governed_connection(...)` handle, an MCP `query` tool result, and a `GovernedSQLTool`
instance. None of them may expose an ACCESSIBLE `.raw`/`.cursor`/`.connection`/`.conn`/
`._executor`/`.executor`/`.read_executor`/`.write_executor` attribute -- the exact
superset `.devdocs/PHASE3_GATES.md` P3.8 names, broader than the `{read_executor,
write_executor, connection, conn, executor}` subset the individual P3.1/P3.3/P3.4 test
files already check (this adds `raw`, `cursor`, and the private-by-convention
`_executor`, since a caller can still literally access a "private" Python attribute --
name-mangling is not an access control).
"""

from __future__ import annotations

from typing import Any

from backend.integrations.langgraph_tool import GovernedSQLTool
from backend.mcp_server import create_mcp_server
from icberg import GovernedConnection, govern, governed_connection

# The exact attribute-name superset named in P3.8's CHECK/EXPECT text.
_FORBIDDEN_HANDLE_ATTRS = {
    "raw",
    "cursor",
    "connection",
    "conn",
    "_executor",
    "executor",
    "read_executor",
    "write_executor",
}

_SENTINEL = object()


def _assert_no_raw_handle_on_object(obj: Any, label: str) -> None:
    """`obj` is a real Python object (not a dict) -- scan every name in
    `_FORBIDDEN_HANDLE_ATTRS`, public or private-by-convention, with `hasattr`/
    `getattr` (not a `dir()`-filtered set) so a leading-underscore passthrough like
    `._executor` can't hide from this check the way a "public attrs only" scan would.
    """
    for attr in _FORBIDDEN_HANDLE_ATTRS:
        value = getattr(obj, attr, _SENTINEL)
        assert value is _SENTINEL, (
            f"{label}: unexpectedly exposes an accessible `.{attr}` "
            f"(ungoverned-SQL passthrough risk) -- value: {value!r}"
        )


def _assert_no_raw_handle_on_result_dict(result: dict[str, Any], label: str) -> None:
    """`result` is a plain dict (a governed query's return value) -- assert none of the
    forbidden names leaked in as a dict key (the data-shape equivalent of an attribute
    passthrough for a caller that treats the result as a mapping), and that Python
    attribute access on the dict object itself exposes none of them either (dicts don't
    support arbitrary attribute access at all, but this keeps the assertion uniform and
    future-proof against a surface someday returning a richer result object instead of a
    plain dict).
    """
    leaked_keys = _FORBIDDEN_HANDLE_ATTRS & set(result.keys())
    assert not leaked_keys, f"{label}: result dict has forbidden key(s) {leaked_keys}"
    _assert_no_raw_handle_on_object(result, label)


class TestNoRawHandleSdkGovern:
    def test_no_raw_handle_govern_result(self, db_path: str):
        result = govern("SELECT * FROM users WHERE id=1 LIMIT 5", actor="agent-1", dsn=db_path)
        assert result["action"] == "allow"  # sanity: this is a real governed result
        _assert_no_raw_handle_on_result_dict(result, "sdk govern() result")


class TestNoRawHandleSdkGovernedConnection:
    def test_no_raw_handle_governed_connection(self, db_path: str):
        db = governed_connection(db_path)
        assert isinstance(db, GovernedConnection)
        _assert_no_raw_handle_on_object(db, "governed_connection(...)")

    def test_no_raw_handle_governed_connection_query_result(self, db_path: str):
        db = governed_connection(db_path)
        result = db.query("SELECT * FROM users WHERE id=1 LIMIT 5", actor="agent-1")
        assert result["action"] == "allow"
        _assert_no_raw_handle_on_result_dict(result, "governed_connection().query() result")


class TestNoRawHandleMcpQueryResult:
    def test_no_raw_handle_mcp_query_result_object(self, db_path: str):
        server = create_mcp_server(db_path)
        result = server.state.query("SELECT * FROM users WHERE id=1 LIMIT 5")
        assert result["action"] == "allow"  # sanity: this is a real governed result
        _assert_no_raw_handle_on_result_dict(result, "MCP query() result")

    def test_no_raw_handle_mcp_server_state(self, db_path: str):
        server = create_mcp_server(db_path)
        _assert_no_raw_handle_on_object(server.state, "MCP server.state")


class TestNoRawHandleLangGraphTool:
    def test_no_raw_handle_governed_sql_tool(self, db_path: str):
        tool = GovernedSQLTool(dsn=db_path, actor="agent-1")
        _assert_no_raw_handle_on_object(tool, "GovernedSQLTool")

    def test_no_raw_handle_governed_sql_tool_invoke_result(self, db_path: str):
        tool = GovernedSQLTool(dsn=db_path, actor="agent-1")
        result = tool.invoke({"sql": "SELECT * FROM users WHERE id=1 LIMIT 5"})
        assert result["action"] == "allow"
        _assert_no_raw_handle_on_result_dict(result, "GovernedSQLTool.invoke() result")
