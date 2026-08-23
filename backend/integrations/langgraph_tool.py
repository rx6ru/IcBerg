"""`GovernedSQLTool` — a drop-in LangChain/LangGraph `BaseTool` that swaps a raw SQL tool
for a *governed* one in one line, per `.devdocs/FLAGSHIP_ROADMAP.md`'s "Framework
adapter" integration surface.

A LangGraph agent normally binds a SQL tool that hands the model's proposed query
straight to a database connection — exactly the trust boundary THREAT_MODEL.md exists to
close (the agent *proposes*, it must never itself *execute*). `GovernedSQLTool.invoke`
routes every proposed statement through `icberg.GovernedConnection` instead: the SAME
`Gateway`/`GovernanceGate` the Python SDK (`icberg`) and the MCP server
(`backend.mcp_server`) use, so a destructive statement is blocked, a write is held for
human approval, and a safe bounded read returns PII-redacted rows — never a raw
executor or DB connection reaching the agent, and never an unreviewed write executing.

Bind it into an agent exactly like any other `BaseTool`::

    from backend.integrations.langgraph_tool import GovernedSQLTool

    sql_tool = GovernedSQLTool(dsn="app.db", policy="policy.yaml", actor="agent-1")
    agent = create_react_agent(model, tools=[sql_tool])
    # agent.invoke({"messages": [("user", "show me user 1")]})

Or call it directly, the way `.devdocs/PHASE3_GATES.md` P3.4's gate does::

    sql_tool.invoke({"sql": "DROP TABLE users"})   # -> {"action": "block", ...}
    sql_tool.invoke({"sql": "SELECT * FROM users WHERE id=1 LIMIT 5"})  # -> allow, redacted

See `examples/langgraph_agent_example.py` for a runnable end-to-end example.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from icberg import GovernedConnection, Policy, governed_connection

_PolicyLike = str | dict[str, Any] | Policy | None


class GovernedSQLArgs(BaseModel):
    """`GovernedSQLTool`'s args schema — a single, untrusted `sql` field. This is the
    ENTIRE surface an agent/model can control; there is no separate "trusted" argument
    (a table name, a mode flag, ...) that could be used to route around governance."""

    sql: str = Field(
        ...,
        description="One proposed SQL statement. Untrusted — governed end to end before "
        "anything executes: a destructive or out-of-scope statement is blocked, a write "
        "is held for human approval, and a safe bounded read returns PII-redacted rows.",
    )


class GovernedSQLTool(BaseTool):
    """A drop-in replacement for a raw SQL-execution tool. `name="governed_sql"` (per
    `.devdocs/PHASE3_GATES.md` P3.4); `args_schema=GovernedSQLArgs` (`{"sql": ...}`).

    Deliberately offers no attribute that exposes a raw executor/DB connection — only a
    private `GovernedConnection`, itself governance-only (see `icberg`'s module
    docstring). `_run`/`_arun` are the ONLY execution path this tool has, and both route
    through that same connection's `.query()`, which never auto-executes a write.
    """

    name: str = "governed_sql"
    description: str = (
        "Propose one SQL statement against the governed database. Untrusted input: the "
        "statement is evaluated by IcBerg's policy gate before anything executes — a "
        "destructive or out-of-scope statement (DROP/TRUNCATE/injection/RCE/DoS/...) is "
        "blocked outright, a write is held for human approval, and a safe, bounded read "
        "is executed with PII redacted from every returned row. Use this in place of any "
        "raw SQL-execution tool."
    )
    args_schema: type[BaseModel] = GovernedSQLArgs

    _connection: GovernedConnection = PrivateAttr()
    _actor: str = PrivateAttr()

    def __init__(
        self,
        dsn: str | None = None,
        *,
        connection: GovernedConnection | None = None,
        policy: _PolicyLike = None,
        actor: str = "langgraph-agent",
        **kwargs: Any,
    ) -> None:
        """Args:
            dsn: A SQLite path/URI, `postgres://...`, or `mysql://...` — passed to
                `icberg.governed_connection`. Ignored if `connection` is given.
            connection: An already-built `icberg.GovernedConnection` to reuse (e.g. so
                several tools/agents share one audit trail/approval queue). Takes
                precedence over `dsn` when both are given.
            policy: Optional policy YAML path, dict, or `Policy` (see
                `backend.core.policy.load_policy`) — ignored when `connection` is given
                (a pre-built connection already carries its own policy).
            actor: Identity recorded on every proposal this tool makes.
            **kwargs: Forwarded to `BaseTool.__init__` (e.g. `name=`/`description=`
                overrides).

        Raises:
            ValueError: if neither `dsn` nor `connection` is given.
        """
        super().__init__(**kwargs)
        if connection is None:
            if dsn is None:
                raise ValueError("GovernedSQLTool requires either `dsn` or `connection`")
            connection = governed_connection(dsn, policy=policy)
        self._connection = connection
        self._actor = actor

    def _run(self, sql: str) -> dict[str, Any]:
        """Route `sql` through the governed connection — the ONLY execution path this
        tool offers. Returns exactly what `GovernedConnection.query` returns: `action`
        (allow/block/hold), `reason`, `matched_rules`, `rows` (redacted, only on
        `allow`), `redaction_report`, `audit_seq`, `approval_id` (only on `hold`).
        """
        return self._connection.query(sql, self._actor)

    async def _arun(self, sql: str) -> dict[str, Any]:
        # `GovernedConnection.query` is synchronous (SQLite/psycopg/pymysql calls under
        # the hood, same as every other executor in this codebase) — no separate async
        # governance path exists, or should exist, to keep exactly one code path.
        return self._run(sql)
