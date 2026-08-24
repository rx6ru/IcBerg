"""Agent tools for IcBerg's reference agent.

The agent's ONLY data-access tool is `backend.integrations.langgraph_tool
.GovernedSQLTool` -- the SAME governed adapter documented in that module. It is built
here by `create_governed_sql_tool()` and bound directly, as a real `BaseTool`
instance, into the agent's tool list in `backend/agent/agent.py` -- never wrapped or
re-implemented, so `isinstance(tool, GovernedSQLTool)` holds for the actual tool the
agent calls (see `tests/e2e`).

The two read-only info tools below (`list_tables`, `describe_table`) never write
anything, but "read-only" does NOT mean "bypasses governance": both route through the
exact same governed connection via `submit_sql()` below, never a raw sqlite3/DB
connection of their own. There is no raw executor anywhere in this module.
"""

from __future__ import annotations

from typing import Any

import structlog
from langchain_core.tools import tool

from backend.integrations.langgraph_tool import GovernedSQLTool
from icberg import GovernedConnection

logger = structlog.get_logger(__name__)

DEFAULT_ACTOR = "icberg-reference-agent"

# Module-level singleton, wired by `backend.agent.agent.bind_governed_tools()` (mirrors
# the prior module's `set_dataframe`/`_df` pattern) so the read-only info tools below
# and `submit_sql()` share the exact same governed connection/audit trail as the
# `GovernedSQLTool` instance actually bound to the agent's tool list.
_sql_tool: GovernedSQLTool | None = None


def create_governed_sql_tool(
    dsn: str | None = None,
    *,
    connection: GovernedConnection | None = None,
    policy: Any = None,
    actor: str = DEFAULT_ACTOR,
) -> GovernedSQLTool:
    """Build the ONE tool this agent uses to touch data -- a thin, explicit factory
    around `GovernedSQLTool` (kept here, rather than constructed inline, so tests and
    `examples/reference_agent.py` have a single, obvious entry point). Exactly one of
    `dsn`/`connection` must be given; see `GovernedSQLTool.__init__`.
    """
    return GovernedSQLTool(dsn=dsn, connection=connection, policy=policy, actor=actor)


def set_sql_tool(sql_tool: GovernedSQLTool) -> None:
    """Wire the singleton governed SQL tool used by `list_tables`/`describe_table`/
    `submit_sql` below. Called once by `backend.agent.agent.bind_governed_tools()`.
    """
    global _sql_tool
    _sql_tool = sql_tool


def get_sql_tool() -> GovernedSQLTool:
    """Return the currently-wired governed SQL tool, raising if none is wired yet."""
    if _sql_tool is None:
        raise RuntimeError(
            "Governed SQL tool not initialized. Call "
            "backend.agent.agent.bind_governed_tools() (or set_sql_tool() directly) first."
        )
    return _sql_tool


def submit_sql(sql: str) -> dict[str, Any]:
    """Deterministic, LLM-free path: submit one SQL string through the exact same
    governed tool the agent's LangGraph node calls (`GovernedSQLTool.invoke`). This is
    what `examples/reference_agent.py` and `tests/e2e` drive to prove governance
    end-to-end without any live LLM in the loop -- it is not a shortcut around
    governance, it IS `GovernedSQLTool.invoke` under a descriptive name.

    Returns:
        The same dict `GovernedConnection.query` returns: `action`, `reason`,
        `matched_rules`, `rows` (redacted, only on `allow`), `redaction_report`,
        `audit_seq`, `approval_id`.
    """
    return get_sql_tool().invoke({"sql": sql})


@tool
def list_tables() -> str:
    """List every table available in the governed database."""
    result = submit_sql("SELECT name FROM sqlite_master WHERE type='table' LIMIT 50")
    if result["action"] != "allow":
        return f"ERROR: could not list tables - {result['reason']}"
    names = [row["name"] for row in (result["rows"] or [])]
    return ", ".join(names) if names else "No tables found."


@tool
def describe_table(table_name: str) -> str:
    """Show up to 5 sample rows (PII redacted by governance) from one table.

    Args:
        table_name: A table name, e.g. one reported by `list_tables`.
    """
    if not table_name.isidentifier():
        return f"ERROR: '{table_name}' is not a valid table name."
    result = submit_sql(f"SELECT * FROM {table_name} WHERE 1 <> 0 LIMIT 5")
    if result["action"] != "allow":
        return f"ERROR: {result['reason']}"
    rows = result["rows"] or []
    return "\n".join(str(row) for row in rows) if rows else "No rows found."
