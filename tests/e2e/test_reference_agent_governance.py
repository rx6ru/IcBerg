"""E2E: IcBerg's reference agent (`backend/agent/*`) dogfoods its own governed SQL
tool -- proven end-to-end WITHOUT a live LLM by driving the exact tool the agent's
LangGraph node would call (`backend.agent.agent.bind_governed_tools` /
`backend.agent.tools.submit_sql`), against a demo SQLite DB seeded by
`scripts/seed_demo_db.py`. All data is synthetic/fabricated -- no real people.

Covers:
  (a) the reference agent's SQL tool IS `GovernedSQLTool` -- governed, no raw executor.
  (b) a destructive request routed through that tool is BLOCKED and the DB is
      unchanged.
  (c) a safe PII SELECT routed through that tool returns REDACTED rows.
"""

from __future__ import annotations

import sqlite3

import pytest

from backend.agent.agent import bind_governed_tools
from backend.agent.tools import get_sql_tool, submit_sql
from backend.integrations.langgraph_tool import GovernedSQLTool
from scripts.seed_demo_db import PASSENGERS, USERS, seed as seed_demo_db

_RAW_EXECUTOR_ATTRS = {"read_executor", "write_executor", "connection", "conn", "executor"}


@pytest.fixture
def demo_db_path(tmp_path) -> str:
    """A throwaway, freshly-seeded copy of the demo DB for each test."""
    path = tmp_path / "e2e-demo.sqlite"
    seed_demo_db(path)
    return str(path)


@pytest.fixture
def agent_tools(demo_db_path):
    """The reference agent's actual tool list, built the same LLM-free way
    `backend.agent.agent.create_agent` builds it internally (`bind_governed_tools`),
    just without the `create_react_agent`/chat-model wrapping step -- so this fixture
    requires no LLM API key.
    """
    return bind_governed_tools(demo_db_path, actor="e2e-test-agent")


class TestAgentSqlToolIsGoverned:
    """(a) The reference agent's SQL tool is the GovernedSQLTool -- governed, no raw
    executor -- proven without any LLM."""

    def test_bound_tool_is_governed_sql_tool(self, agent_tools):
        sql_tool = agent_tools[0]
        assert isinstance(sql_tool, GovernedSQLTool)
        assert sql_tool.name == "governed_sql"

    def test_wired_singleton_matches_bound_tool(self, agent_tools):
        # `submit_sql`'s deterministic path and the bound tool the agent calls must be
        # the literal same object -- not two separately-governed instances.
        assert get_sql_tool() is agent_tools[0]

    def test_no_raw_executor_attribute_reachable_from_the_tool(self, agent_tools):
        sql_tool = agent_tools[0]
        for attr in _RAW_EXECUTOR_ATTRS:
            assert not hasattr(sql_tool, attr), f"GovernedSQLTool must not expose `.{attr}`"

    def test_only_governed_sql_and_readonly_info_tools_are_bound(self, agent_tools):
        # No pandas/sandbox/raw-query tool anywhere in the agent's tool list.
        names = {t.name for t in agent_tools}
        assert names == {"governed_sql", "list_tables", "describe_table"}


class TestDestructiveRequestIsBlocked:
    """(b) A destructive request routed through the agent's tool is BLOCKED, and the
    database is unchanged -- proven without any LLM."""

    def test_drop_table_is_blocked(self, agent_tools):
        result = agent_tools[0].invoke({"sql": "DROP TABLE users"})
        assert result["action"] == "block"
        assert result["rows"] is None

    def test_drop_table_via_agent_tool_leaves_db_unchanged(self, agent_tools, demo_db_path):
        agent_tools[0].invoke({"sql": "DROP TABLE users"})

        conn = sqlite3.connect(demo_db_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        finally:
            conn.close()
        assert count == len(USERS)

    def test_deterministic_submit_sql_path_also_blocks_drop(self, agent_tools):
        # `submit_sql` (the deterministic, LLM-free path `create_agent`'s node would
        # use) must make the identical decision as calling the bound tool directly.
        result = submit_sql("DROP TABLE passengers")
        assert result["action"] == "block"


class TestSafePiiSelectIsRedacted:
    """(c) A safe PII SELECT returns REDACTED rows -- proven without any LLM."""

    def test_bounded_select_on_users_is_allowed_and_redacted(self, agent_tools):
        result = agent_tools[0].invoke({"sql": "SELECT * FROM users WHERE id=1 LIMIT 5"})
        assert result["action"] == "allow"
        row = result["rows"][0]
        assert row["id"] == 1
        assert row["username"] == "alice"
        assert row["email"] == "[REDACTED]"
        assert row["ssn"] == "[REDACTED]"

    def test_non_pii_table_is_allowed_without_redaction(self, agent_tools):
        result = submit_sql("SELECT * FROM passengers WHERE id=1 LIMIT 5")
        assert result["action"] == "allow"
        row = result["rows"][0]
        assert row["name"] == PASSENGERS[0][1]

    def test_describe_table_info_tool_also_redacts_pii(self, agent_tools):
        # The "read-only info" tool routes through governance too -- not a bypass.
        describe = next(t for t in agent_tools if t.name == "describe_table")
        output = describe.invoke({"table_name": "users"})
        assert "[REDACTED]" in output
        assert "alice@example.com" not in output
        assert "111-22-3333" not in output
