"""P3.3 (`.devdocs/PHASE3_GATES.md`): the MCP server's `query` tool governs a proposed
SQL statement exactly like the SDK/REST API (`DROP` -> block, `SELECT` -> allow+redacted);
`approve`/`list_pending_approvals`/`audit_tail` are registered and callable.

Every test calls the registered tool handlers directly via `server.state.<method>` (a
plain, non-async method — see `backend/mcp_server.py`'s module docstring) or the
server's own `list_tools()`/`call_tool()` machinery run with `asyncio.run` — no live MCP
client/stdio transport is required.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from backend.mcp_server import DEFAULT_ACTOR, create_mcp_server

_FORBIDDEN_ATTRS = {"read_executor", "write_executor", "connection", "conn", "executor"}


class TestMcpGovernsQuery:
    def test_mcp_governs_select_allows_and_redacts_pii(self, db_path: str):
        server = create_mcp_server(db_path)
        result = server.state.query("SELECT * FROM users WHERE id=1 LIMIT 5")
        assert result["action"] == "allow"
        row = result["rows"][0]
        assert row["name"] == "Alice Smith"
        assert row["email"] == "[REDACTED]"
        assert row["ssn_num"] == "[REDACTED]"

    def test_mcp_governs_drop_table_blocks_and_never_executes(self, db_path: str):
        server = create_mcp_server(db_path)
        result = server.state.query("DROP TABLE users")
        assert result["action"] == "block"
        assert result["rows"] is None

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        conn.close()
        assert count == 2

    def test_mcp_governs_write_holds_then_approve_executes_exact_sql(self, db_path: str):
        server = create_mcp_server(db_path)
        held = server.state.query("UPDATE users SET admin=1 WHERE id=2")
        assert held["action"] == "hold"
        approval_id = held["approval_id"]

        pending = server.state.list_pending_approvals()
        assert any(p["id"] == approval_id for p in pending)

        approved = server.state.approve(approval_id, "human-reviewer")
        assert approved["error"] is None

        conn = sqlite3.connect(db_path)
        admin = conn.execute("SELECT admin FROM users WHERE id=2").fetchone()[0]
        conn.close()
        assert admin == 1

    def test_mcp_audit_tail_reflects_prior_decisions(self, db_path: str):
        server = create_mcp_server(db_path)
        server.state.query("SELECT * FROM users WHERE id=1 LIMIT 5")
        server.state.query("DROP TABLE users")
        tail = server.state.audit_tail(5)
        assert len(tail) == 2
        assert {entry["action"] for entry in tail} == {"allow", "block"}

    def test_mcp_uses_the_configured_actor_identity(self, db_path: str):
        server = create_mcp_server(db_path, actor="custom-actor")
        assert server.state.actor == "custom-actor" != DEFAULT_ACTOR
        server.state.query("SELECT * FROM users WHERE id=1 LIMIT 5")
        assert server.state.audit_tail(1)[0]["actor"] == "custom-actor"


class TestMcpToolRegistration:
    def test_mcp_server_registers_the_three_agent_facing_tools(self, db_path: str):
        server = create_mcp_server(db_path)
        tool_names = {t.name for t in asyncio.run(server.list_tools())}
        assert tool_names == {"query", "list_pending_approvals", "audit_tail"}

    def test_mcp_query_tool_callable_through_the_real_call_tool_dispatch(self, db_path: str):
        server = create_mcp_server(db_path)
        result = asyncio.run(server.call_tool("query", {"sql": "DROP TABLE users"}))
        assert result.is_error is False
        assert result.structured_content["action"] == "block"

    def test_mcp_server_never_exposes_raw_executor_or_connection(self, db_path: str):
        server = create_mcp_server(db_path)
        public_attrs = {a for a in dir(server.state) if not a.startswith("_")}
        assert not (public_attrs & _FORBIDDEN_ATTRS), public_attrs & _FORBIDDEN_ATTRS


class TestMcpNoSelfApproval:
    """P3.7 (`.devdocs/PHASE3_GATES.md`): the MCP tool registry has NO agent-callable
    `approve` tool -- a principal that proposes a write over MCP structurally cannot
    also approve it over MCP, because there is no MCP-reachable path to `approve` at
    all. Approval of a held write is granted out-of-band (authenticated REST call /
    human UI), never through this protocol surface.
    """

    def test_mcp_no_self_approval_tool_registry_has_no_approve(self, db_path: str):
        server = create_mcp_server(db_path)
        tool_names = {t.name for t in asyncio.run(server.list_tools())}
        assert "approve" not in tool_names
        assert tool_names == {"query", "list_pending_approvals", "audit_tail"}

    def test_mcp_no_self_approval_call_tool_dispatch_rejects_approve(self, db_path: str):
        from mcp.server.mcpserver.exceptions import ToolError

        server = create_mcp_server(db_path)
        held = server.state.query("UPDATE users SET admin=1 WHERE id=1")
        assert held["action"] == "hold"

        # The exact same actor that proposed the write cannot reach `approve` at all
        # over the MCP protocol -- "unknown tool", not merely "self-approval refused".
        with pytest.raises(ToolError):
            asyncio.run(
                server.call_tool("approve", {"id": held["approval_id"], "approver": server.state.actor})
            )

    def test_mcp_no_self_approval_out_of_band_state_approve_still_reachable_directly(self, db_path: str):
        # `_ServerState.approve` remains a plain method so an out-of-band caller (a
        # REST endpoint, a human-approval UI, test code) can still invoke it directly
        # on `server.state` -- it is simply never wrapped as an MCP tool, so no
        # MCP-connected agent/client can reach it through the protocol.
        server = create_mcp_server(db_path)
        held = server.state.query("UPDATE users SET admin=1 WHERE id=1")
        approved = server.state.approve(held["approval_id"], "human-reviewer")
        assert approved["error"] is None
