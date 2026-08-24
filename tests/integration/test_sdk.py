"""P3.1 (`.devdocs/PHASE3_GATES.md`): `icberg`'s public SDK governs a proposed SQL
statement end to end — a safe `SELECT` returns `allow` with PII redacted, a `DROP TABLE`
returns `block`, and a bounded write is `hold`ed for human approval, never auto-executed.

Also carries P3.5 (`surfaces_consistent`): the SDK, the MCP server, and the LangGraph
tool all route through the SAME `Gateway`/`GovernanceGate` — a known-blocked statement is
blocked identically on all three, and none exposes a raw executor/DB connection.
"""

from __future__ import annotations

import sqlite3

import pytest

from backend.integrations.langgraph_tool import GovernedSQLTool
from backend.mcp_server import create_mcp_server
from icberg import (
    Gateway,
    GovernanceGate,
    GovernedConnection,
    Policy,
    PolicyDecision,
    govern,
    governed_connection,
    load_policy,
)

_FORBIDDEN_ATTRS = {"read_executor", "write_executor", "connection", "conn", "executor"}


class TestSdkGovernsQuery:
    def test_sdk_governs_select_allows_and_redacts_pii(self, db_path: str):
        result = govern("SELECT * FROM users WHERE id=1 LIMIT 5", actor="agent-1", dsn=db_path)
        assert result["action"] == "allow"
        row = result["rows"][0]
        assert row["id"] == 1
        assert row["name"] == "Alice Smith"
        assert row["admin"] == 0
        assert row["email"] == "[REDACTED]"
        assert row["ssn"] == "[REDACTED]"
        assert row["ssn_num"] == "[REDACTED]"
        assert result["audit_seq"] >= 1

    def test_sdk_governs_drop_table_blocks_and_never_executes(self, db_path: str):
        result = govern("DROP TABLE users", actor="agent-1", dsn=db_path)
        assert result["action"] == "block"
        assert result["rows"] is None
        assert "ddl_blocked" in result["matched_rules"]

        # The table still exists and is queryable -- the DROP never reached the engine.
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        conn.close()
        assert count == 2

    def test_sdk_governs_bounded_write_holds_then_approve_executes(self, db_path: str):
        db = governed_connection(db_path)
        result = db.query("UPDATE users SET admin=1 WHERE id=1", actor="agent-1")
        assert result["action"] == "hold"
        approval_id = result["approval_id"]
        assert approval_id

        conn = sqlite3.connect(db_path)
        assert conn.execute("SELECT admin FROM users WHERE id=1").fetchone()[0] == 0  # not yet

        approved = db.approve(approval_id, approver="alice-approver")
        assert approved["error"] is None

        assert conn.execute("SELECT admin FROM users WHERE id=1").fetchone()[0] == 1  # now applied
        conn.close()

    def test_sdk_govern_requires_exactly_one_of_connection_or_dsn(self, db_path: str):
        import pytest

        with pytest.raises(ValueError):
            govern("SELECT 1", actor="agent-1")


class TestSdkPublicApi:
    def test_sdk_public_api_import_surface_is_stable(self):
        assert callable(govern)
        assert callable(governed_connection)
        assert callable(load_policy)
        assert Gateway is not None
        assert GovernanceGate is not None
        assert PolicyDecision is not None
        assert Policy is not None
        assert GovernedConnection is not None

    def test_sdk_governed_connection_never_exposes_raw_executor_or_connection(self, db_path: str):
        db = governed_connection(db_path)
        public_attrs = {a for a in dir(db) if not a.startswith("_")}
        assert not (public_attrs & _FORBIDDEN_ATTRS), public_attrs & _FORBIDDEN_ATTRS


class TestSurfacesConsistent:
    """P3.5: SDK, MCP server, and LangGraph tool all route through the SAME
    Gateway/GovernanceGate -- a known-blocked statement is blocked identically on all
    three, and none of the three exposes a raw executor/DB connection."""

    BLOCKED_SQL = "DROP TABLE users"

    def test_surfaces_consistent_block_identical_and_no_raw_executor_exposed(self, db_path: str):
        sdk_result = govern(self.BLOCKED_SQL, actor="agent-1", dsn=db_path)

        mcp_server = create_mcp_server(db_path)
        mcp_result = mcp_server.state.query(self.BLOCKED_SQL)

        tool = GovernedSQLTool(dsn=db_path, actor="agent-1")
        langgraph_result = tool.invoke({"sql": self.BLOCKED_SQL})

        for label, result in (("sdk", sdk_result), ("mcp", mcp_result), ("langgraph", langgraph_result)):
            assert result["action"] == "block", label
            assert result["rows"] is None, label
            assert "ddl_blocked" in result["matched_rules"], label
            assert result["reason"] == sdk_result["reason"], label

        for label, obj in (
            ("sdk", governed_connection(db_path)),
            ("mcp", mcp_server.state),
            ("langgraph", tool),
        ):
            public_attrs = {a for a in dir(obj) if not a.startswith("_")}
            assert not (public_attrs & _FORBIDDEN_ATTRS), (label, public_attrs & _FORBIDDEN_ATTRS)


class TestSurfacesParity:
    """P3.10: `surfaces_consistent` (above) proves the three surfaces block a
    known-bad statement identically -- this proves the ALLOW path is identical too
    (same redacted row shape for the same safe PII SELECT across SDK/MCP/LangGraph),
    and that `GovernedSQLTool`'s sync `.invoke`/`_run` and async `.ainvoke`/`_arun`
    govern identically (no sync/async divergence in the one integration surface that
    offers both)."""

    SAFE_PII_SELECT = "SELECT * FROM users WHERE id=1 LIMIT 5"

    def test_surfaces_parity_allow_path_identical_redacted_row_shape(self, db_path: str):
        sdk_result = govern(self.SAFE_PII_SELECT, actor="agent-1", dsn=db_path)

        mcp_server = create_mcp_server(db_path, actor="agent-1")
        mcp_result = mcp_server.state.query(self.SAFE_PII_SELECT)

        tool = GovernedSQLTool(dsn=db_path, actor="agent-1")
        langgraph_result = tool.invoke({"sql": self.SAFE_PII_SELECT})

        for label, result in (("sdk", sdk_result), ("mcp", mcp_result), ("langgraph", langgraph_result)):
            assert result["action"] == "allow", label
            assert result["rows"] == sdk_result["rows"], label
            assert result["redaction_report"] == sdk_result["redaction_report"], label

        # Pin the exact redacted shape down explicitly too, not only cross-compared --
        # three surfaces agreeing on the same wrong shape would otherwise still pass.
        assert sdk_result["rows"] == [
            {
                "id": 1,
                "name": "Alice Smith",
                "email": "[REDACTED]",
                "ssn": "[REDACTED]",
                "ssn_num": "[REDACTED]",
                "admin": 0,
            }
        ]

    @pytest.mark.asyncio
    async def test_surfaces_parity_langgraph_tool_invoke_and_ainvoke_govern_identically_on_allow(
        self, db_path: str
    ):
        tool = GovernedSQLTool(dsn=db_path, actor="agent-1")

        sync_result = tool.invoke({"sql": self.SAFE_PII_SELECT})
        async_result = await tool.ainvoke({"sql": self.SAFE_PII_SELECT})

        for key in ("action", "reason", "matched_rules", "rows", "redaction_report"):
            assert sync_result[key] == async_result[key], key

    @pytest.mark.asyncio
    async def test_surfaces_parity_langgraph_tool_invoke_and_ainvoke_govern_identically_on_block(
        self, db_path: str
    ):
        tool = GovernedSQLTool(dsn=db_path, actor="agent-1")

        sync_result = tool.invoke({"sql": "DROP TABLE users"})
        async_result = await tool.ainvoke({"sql": "DROP TABLE users"})

        for key in ("action", "reason", "matched_rules", "rows"):
            assert sync_result[key] == async_result[key], key
