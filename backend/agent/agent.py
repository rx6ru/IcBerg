"""LangGraph reference agent factory -- IcBerg dogfooding its own governed SQL tool.

The agent's ONLY data-access tool is `backend.integrations.langgraph_tool
.GovernedSQLTool` -- the SAME governed adapter class the SDK (`icberg`) and the MCP
server (`backend.mcp_server`) route through. It is bound directly, as a real
`BaseTool` instance, into the agent's tool list -- never wrapped or re-implemented, and
there is no raw DB connection or executor anywhere in this module.

Two ways to use this module:
  - `bind_governed_tools(...)` -- pure, LLM-free. Builds/wires the governed SQL tool
    (plus the read-only `list_tables`/`describe_table` info tools, which also route
    through it -- see `backend.agent.tools`) and returns the tool list. This is what
    `tests/e2e` and `examples/reference_agent.py` use to prove governance end-to-end
    with no live LLM in the loop; `backend.agent.tools.submit_sql` is the matching
    deterministic query path.
  - `create_agent(llm_adapter, ...)` -- calls `bind_governed_tools` and wraps the
    result in a real `create_react_agent` graph, using `llm_adapter`'s existing
    Cerebras/Groq failover (`backend.core.llm_adapter.LLMAdapter`). Raises
    `LLMUnavailableError` if no provider key is configured -- unchanged from this
    module's behavior before it was repurposed.
"""

from __future__ import annotations

import structlog
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from backend.agent.prompts import build_system_prompt
from backend.agent.tools import (
    DEFAULT_ACTOR,
    create_governed_sql_tool,
    describe_table,
    list_tables,
    set_sql_tool,
)
from backend.core.llm_adapter import LLMAdapter
from backend.integrations.langgraph_tool import GovernedSQLTool
from icberg import GovernedConnection

logger = structlog.get_logger(__name__)

# Default demo database -- see `scripts/seed_demo_db.py`. Not auto-seeded here: this
# module only ever *queries* through governance, it never writes schema/rows itself.
DEFAULT_DEMO_DB_PATH = "data/demo.sqlite"


def bind_governed_tools(
    dsn: str | None = None,
    *,
    connection: GovernedConnection | None = None,
    policy=None,
    actor: str = DEFAULT_ACTOR,
    sql_tool: GovernedSQLTool | None = None,
) -> list[BaseTool]:
    """Build (or reuse) the governed SQL tool and wire it as this agent's tool list.

    LLM-free -- constructing this list never touches an LLM provider, so it is the
    entry point offline tests/examples use to prove governance end-to-end.

    Args:
        dsn: SQLite path/URI (or postgres/mysql DSN) for a fresh `GovernedSQLTool`.
            Defaults to `DEFAULT_DEMO_DB_PATH` if neither `connection` nor `sql_tool`
            is given -- see `scripts/seed_demo_db.py` to seed it.
        connection: An existing `icberg.GovernedConnection` to build the tool around.
        policy: Optional policy (path/dict/`Policy`) -- see `icberg.load_policy`.
        actor: Identity recorded on every proposal this agent's tool makes.
        sql_tool: An already-built `GovernedSQLTool` to reuse outright (takes
            precedence over `dsn`/`connection`/`policy`/`actor`).

    Returns:
        `[governed_sql_tool, list_tables, describe_table]` -- index 0 is always the
        real `GovernedSQLTool` instance the agent calls
        (`isinstance(tools[0], GovernedSQLTool)`).
    """
    tool = sql_tool or create_governed_sql_tool(
        dsn or DEFAULT_DEMO_DB_PATH, connection=connection, policy=policy, actor=actor
    )
    set_sql_tool(tool)
    return [tool, list_tables, describe_table]


def create_agent(
    llm_adapter: LLMAdapter,
    df=None,
    *,
    dsn: str | None = None,
    connection: GovernedConnection | None = None,
    policy=None,
    actor: str = DEFAULT_ACTOR,
    sql_tool: GovernedSQLTool | None = None,
):
    """Build the compiled ReAct agent graph, bound to the governed SQL tool.

    Args:
        llm_adapter: Initialized LLM adapter with Cerebras/Groq failover.
        df: Unused. Accepted (as the second positional argument) only for backward
            compatibility with existing callers (`backend/main.py`,
            `backend/api/routes.py`) built for the prior Titanic-pandas agent. This
            agent's only data source is the governed SQL tool -- pass
            `dsn`/`connection`/`sql_tool` to point it at real data.
        dsn/connection/policy/actor/sql_tool: See `bind_governed_tools`.

    Returns:
        Compiled LangGraph state graph, ready to `.invoke`/`.stream`.

    Raises:
        LLMUnavailableError: If `llm_adapter` has no provider key configured --
            unchanged from this module's behavior before it was repurposed.
    """
    tools = bind_governed_tools(dsn, connection=connection, policy=policy, actor=actor, sql_tool=sql_tool)

    model = llm_adapter.get_chat_model()
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=build_system_prompt(),
    )

    logger.info("agent.created", tools=[t.name for t in tools])
    return agent
