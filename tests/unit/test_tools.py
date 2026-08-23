"""Unit tests for `backend.agent.tools` -- the reference agent's governed SQL tools.

The prior version of this module tested the Titanic-pandas tools
(`get_dataset_info`/`get_statistics`/`query_data`/`visualize_data`/`set_dataframe`)
that reference-agent work repurposed away (see `backend/agent/tools.py`'s module
docstring). These tests instead exercise the governed-SQL replacements
(`create_governed_sql_tool`, `set_sql_tool`/`get_sql_tool`, `submit_sql`,
`list_tables`, `describe_table`) against a throwaway, synthetic SQLite `users` table
-- no real data -- proving governance actually happens (a destructive/injection
statement is blocked, a safe read comes back with PII redacted, and no raw
sqlite3/executor is used anywhere in this module), not just that the tools return a
string.
"""

from __future__ import annotations

import sqlite3

import pytest

import backend.agent.tools as agent_tools
from backend.agent.tools import (
    create_governed_sql_tool,
    describe_table,
    get_sql_tool,
    list_tables,
    set_sql_tool,
    submit_sql,
)
from backend.integrations.langgraph_tool import GovernedSQLTool

_FORBIDDEN_ATTRS = {"read_executor", "write_executor", "connection", "conn", "executor"}

USERS_SCHEMA_SQL = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    ssn TEXT,
    admin INTEGER
)
"""


@pytest.fixture
def db_path(tmp_path) -> str:
    """A throwaway SQLite file, seeded with two fabricated `users` rows -- no real
    data."""
    path = str(tmp_path / "unit-tools.sqlite")
    conn = sqlite3.connect(path)
    try:
        conn.execute(USERS_SCHEMA_SQL)
        conn.executemany(
            "INSERT INTO users (id, name, email, ssn, admin) VALUES (?, ?, ?, ?, ?)",
            [
                (1, "Alice Smith", "alice@example.com", "111-22-3333", 0),
                (2, "Bob Jones", "bob@example.com", "222-33-4444", 1),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return path


@pytest.fixture(autouse=True)
def wired_sql_tool(db_path):
    """Wire the module-level governed SQL tool singleton before each test, the same
    way `backend.agent.agent.bind_governed_tools()` does -- and restore whatever was
    wired before (e.g. by another test module sharing this process) afterward, so
    these tests can't leak state into -- or inherit it from -- any other test file.
    """
    previous = agent_tools._sql_tool
    tool = create_governed_sql_tool(dsn=db_path, actor="unit-test-agent")
    set_sql_tool(tool)
    yield tool
    agent_tools._sql_tool = previous


class TestCreateGovernedSqlTool:

    def test_returns_a_real_governed_sql_tool_instance(self, db_path):
        tool = create_governed_sql_tool(dsn=db_path)
        assert isinstance(tool, GovernedSQLTool)

    def test_requires_dsn_or_connection(self):
        with pytest.raises(ValueError):
            create_governed_sql_tool()


class TestSqlToolSingletonWiring:

    def test_set_and_get_round_trip(self, wired_sql_tool):
        assert get_sql_tool() is wired_sql_tool

    def test_get_raises_when_unwired(self, monkeypatch):
        monkeypatch.setattr(agent_tools, "_sql_tool", None)
        with pytest.raises(RuntimeError):
            get_sql_tool()


class TestNoRawExecutorAnywhere:
    """Mirrors the guarantee `tests/integration/test_langgraph_tool.py` and
    `tests/e2e/test_reference_agent_governance.py` already prove for the bound tool
    itself: this module never reaches a raw DB handle."""

    def test_wired_tool_exposes_no_raw_executor_attrs(self, wired_sql_tool):
        public_attrs = {a for a in dir(wired_sql_tool) if not a.startswith("_")}
        assert not (public_attrs & _FORBIDDEN_ATTRS)

    def test_module_never_imports_sqlite3_directly(self):
        assert not hasattr(agent_tools, "sqlite3")


class TestSubmitSql:

    def test_destructive_statement_is_blocked(self):
        result = submit_sql("DROP TABLE users")
        assert result["action"] == "block"
        assert result["rows"] is None

    def test_destructive_statement_leaves_db_unchanged(self, db_path):
        submit_sql("DROP TABLE users")
        conn = sqlite3.connect(db_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        finally:
            conn.close()
        assert count == 2

    def test_sql_injection_tautology_is_blocked(self):
        result = submit_sql("SELECT * FROM users WHERE id=1 OR 1=1 LIMIT 5")
        assert result["action"] == "block"
        assert result["rows"] is None

    def test_safe_bounded_select_returns_redacted_pii(self):
        result = submit_sql("SELECT * FROM users WHERE id=1 LIMIT 5")
        assert result["action"] == "allow"
        row = result["rows"][0]
        assert row["id"] == 1
        assert row["name"] == "Alice Smith"
        assert row["email"] == "[REDACTED]"
        assert row["ssn"] == "[REDACTED]"
        # Non-PII columns pass through untouched -- this is redaction, not blanket
        # scrubbing.
        assert row["admin"] == 0

    def test_unwired_raises(self, monkeypatch):
        monkeypatch.setattr(agent_tools, "_sql_tool", None)
        with pytest.raises(RuntimeError):
            submit_sql("SELECT 1")


class TestListTables:

    def test_lists_the_seeded_table(self):
        result = list_tables.invoke({})
        assert "users" in result

    def test_returns_a_string(self):
        result = list_tables.invoke({})
        assert isinstance(result, str)

    def test_surfaces_a_governance_denial_as_a_message_not_a_crash(self, monkeypatch):
        monkeypatch.setattr(
            agent_tools,
            "submit_sql",
            lambda sql: {"action": "block", "reason": "denied for test", "rows": None},
        )
        result = list_tables.invoke({})
        assert result.startswith("ERROR")
        assert "denied for test" in result


class TestDescribeTable:

    def test_describe_returns_redacted_pii_rows(self):
        result = describe_table.invoke({"table_name": "users"})
        assert "[REDACTED]" in result
        assert "alice@example.com" not in result
        assert "111-22-3333" not in result
        assert "Alice Smith" in result

    def test_rejects_non_identifier_table_name_without_querying(self, monkeypatch):
        # Not a valid Python identifier -- describe_table must reject it outright and
        # never hand it to submit_sql/governance at all (this is a defense-in-depth
        # guard in front of governance, not a substitute for it -- `TestSubmitSql`
        # above proves the governance layer itself blocks destructive/injection SQL).
        called = []
        monkeypatch.setattr(
            agent_tools, "submit_sql", lambda sql: called.append(sql) or {}
        )
        result = describe_table.invoke({"table_name": "users; DROP TABLE users--"})
        assert result.startswith("ERROR")
        assert called == []

    def test_surfaces_a_governance_denial_as_a_message_not_a_crash(self, monkeypatch):
        monkeypatch.setattr(
            agent_tools,
            "submit_sql",
            lambda sql: {"action": "hold", "reason": "held for test", "rows": None},
        )
        result = describe_table.invoke({"table_name": "users"})
        assert result.startswith("ERROR")
        assert "held for test" in result
