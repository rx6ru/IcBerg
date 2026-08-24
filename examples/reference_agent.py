"""Runnable reference example: IcBerg's own agent, dogfooding its governed SQL tool.

Shows `backend/agent/*`'s reference agent -- whose only data-access tool is
`GovernedSQLTool` (`backend.integrations.langgraph_tool`) -- answering a safe question
(a bounded SELECT against the demo `users` table -> PII-redacted rows) and refusing a
destructive request (`DROP TABLE users` -> blocked outright, never executed). Both run
entirely offline: no LLM API key required, because this drives the exact same governed
tool the agent's LangGraph node would call, via `backend.agent.tools.submit_sql`
(see `backend/agent/agent.py`'s `bind_governed_tools`).

If CEREBRAS_API_KEY/GROQ_API_KEY ARE configured, it also drives the full LangGraph
ReAct agent with natural-language prompts for each case, using that same governed tool
underneath.

Run from the repo root:
  .venv/bin/python examples/reference_agent.py
(The demo DB is seeded automatically -- see `scripts/seed_demo_db.py`.)
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.agent.agent import (  # noqa: E402
    DEFAULT_DEMO_DB_PATH,
    bind_governed_tools,
    create_agent,
)
from backend.agent.tools import submit_sql  # noqa: E402
from backend.core.llm_adapter import LLMAdapter  # noqa: E402
from backend.integrations.langgraph_tool import GovernedSQLTool  # noqa: E402
from scripts.seed_demo_db import seed as seed_demo_db  # noqa: E402


def main() -> None:
    db_path = PROJECT_ROOT / DEFAULT_DEMO_DB_PATH
    seed_demo_db(db_path)
    print(f"[reference_agent] Demo DB ready at {db_path}\n")

    # --- Wire the reference agent's tools -- no LLM required for this part ---
    tools = bind_governed_tools(str(db_path), actor="reference-agent-example")
    sql_tool = tools[0]
    assert isinstance(sql_tool, GovernedSQLTool), "the reference agent's SQL tool must be governed"
    print(f"Bound tool: {sql_tool.name} ({type(sql_tool).__name__})\n")

    # --- 1. Safe question: governed SELECT -> PII-redacted result ---
    print("=== Safe question: 'show me user 1' ===")
    safe = submit_sql("SELECT * FROM users WHERE id=1 LIMIT 5")
    print("action:", safe["action"])
    print("rows:  ", safe["rows"])
    assert safe["action"] == "allow"
    assert safe["rows"][0]["email"] == "[REDACTED]"
    assert safe["rows"][0]["ssn"] == "[REDACTED]"

    # --- 2. Destructive request: blocked, never executed ---
    print("\n=== Destructive request: 'drop the users table' ===")
    destructive = submit_sql("DROP TABLE users")
    print("action:", destructive["action"])
    print("reason:", destructive["reason"])
    assert destructive["action"] == "block"

    # Prove the table really is untouched -- inspect the raw file directly (the only
    # place in this example a raw sqlite3 connection is used: verifying the outcome,
    # not proposing a statement).
    conn = sqlite3.connect(str(db_path))
    remaining = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    print(f"users table still has {remaining} row(s) -- DROP never executed.\n")
    assert remaining > 0

    # --- Optional: drive the full LangGraph ReAct agent if LLM keys are configured ---
    llm_adapter = LLMAdapter()
    if not llm_adapter.is_healthy():
        print(
            "[reference_agent] No CEREBRAS_API_KEY/GROQ_API_KEY configured -- skipping "
            "the live-LLM ReAct agent demo. The governed-tool proof above already ran "
            "fully offline, which is the point: the agent degrades gracefully without "
            "any LLM provider."
        )
        return

    print("=== Live LLM demo (Cerebras/Groq configured) ===")
    agent = create_agent(llm_adapter, dsn=str(db_path), actor="reference-agent-example-llm")
    for prompt in ("Show me user 1's account details.", "Drop the users table."):
        result = agent.invoke({"messages": [("user", prompt)]}, config={"recursion_limit": 10})
        print(f"\n> {prompt}")
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
