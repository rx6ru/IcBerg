"""P3.4 (`.devdocs/PHASE3_GATES.md`): `GovernedSQLTool(...).invoke` governs a proposed
SQL statement exactly like the SDK/REST API/MCP server — `DROP TABLE` blocks with no
execution, a safe `SELECT` returns redacted rows, and the tool's metadata (name/
description/args schema) is valid for agent binding.
"""

from __future__ import annotations

import sqlite3

import pytest
from pydantic import BaseModel

from backend.integrations.langgraph_tool import GovernedSQLArgs, GovernedSQLTool

_FORBIDDEN_ATTRS = {"read_executor", "write_executor", "connection", "conn", "executor"}


class TestLangGraphToolGovernsInvoke:
    def test_langgraph_governs_drop_table_blocks_no_execution(self, db_path: str):
        tool = GovernedSQLTool(dsn=db_path, actor="agent-1")
        result = tool.invoke({"sql": "DROP TABLE users"})
        assert result["action"] == "block"
        assert result["rows"] is None

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        conn.close()
        assert count == 2

    def test_langgraph_governs_select_allows_and_redacts_pii(self, db_path: str):
        tool = GovernedSQLTool(dsn=db_path, actor="agent-1")
        result = tool.invoke({"sql": "SELECT * FROM users WHERE id=1 LIMIT 5"})
        assert result["action"] == "allow"
        row = result["rows"][0]
        assert row["name"] == "Alice Smith"
        assert row["email"] == "[REDACTED]"
        assert row["ssn_num"] == "[REDACTED]"

    def test_langgraph_governs_write_holds_not_auto_executed(self, db_path: str):
        tool = GovernedSQLTool(dsn=db_path, actor="agent-1")
        result = tool.invoke({"sql": "UPDATE users SET admin=1 WHERE id=1"})
        assert result["action"] == "hold"
        assert result["approval_id"]

        conn = sqlite3.connect(db_path)
        admin = conn.execute("SELECT admin FROM users WHERE id=1").fetchone()[0]
        conn.close()
        assert admin == 0

    @pytest.mark.asyncio
    async def test_langgraph_governs_via_async_ainvoke_too(self, db_path: str):
        tool = GovernedSQLTool(dsn=db_path, actor="agent-1")
        result = await tool.ainvoke({"sql": "DROP TABLE users"})
        assert result["action"] == "block"


class TestLangGraphToolMetadata:
    def test_langgraph_tool_metadata_valid_for_agent_binding(self):
        tool = GovernedSQLTool(dsn=":memory:")
        assert tool.name == "governed_sql"
        assert isinstance(tool.description, str) and len(tool.description) > 0
        assert tool.args_schema is GovernedSQLArgs
        assert issubclass(tool.args_schema, BaseModel)
        assert set(tool.args_schema.model_fields.keys()) == {"sql"}

    def test_langgraph_tool_requires_dsn_or_connection(self):
        with pytest.raises(ValueError):
            GovernedSQLTool()

    def test_langgraph_tool_never_exposes_raw_executor_or_connection(self, db_path: str):
        tool = GovernedSQLTool(dsn=db_path)
        public_attrs = {a for a in dir(tool) if not a.startswith("_")}
        assert not (public_attrs & _FORBIDDEN_ATTRS), public_attrs & _FORBIDDEN_ATTRS
