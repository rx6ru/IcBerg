"""Runnable example: a LangGraph agent bound to IcBerg's `GovernedSQLTool` instead of a
raw SQL tool. See `.devdocs/FLAGSHIP_ROADMAP.md`'s "Framework adapter" integration
surface and `backend/integrations/langgraph_tool.py`.

This example only builds and directly invokes the tool (no LLM/API key required) so it
runs standalone -- swap the `sql_tool.invoke(...)` calls for `agent.invoke({"messages":
[...]})` once you wire in a real chat model, exactly as shown in the commented-out
`create_react_agent` block below.

Run from the repo root: `.venv/bin/python examples/langgraph_agent_example.py`
"""

from __future__ import annotations

import sqlite3
import tempfile

from backend.integrations.langgraph_tool import GovernedSQLTool


def _seed_demo_db() -> str:
    """A throwaway SQLite file with a fabricated `users` table -- no real data."""
    path = tempfile.mktemp(suffix=".sqlite")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT, admin INTEGER)"
    )
    conn.executemany(
        "INSERT INTO users (id, name, email, admin) VALUES (?, ?, ?, ?)",
        [(1, "Alice", "alice@example.com", 0), (2, "Bob", "bob@example.com", 0)],
    )
    conn.commit()
    conn.close()
    return path


def main() -> None:
    db_path = _seed_demo_db()

    sql_tool = GovernedSQLTool(dsn=db_path, actor="langgraph-agent-example")

    print("Tool bound:", sql_tool.name, "-", sql_tool.description[:60], "...")

    # --- Bind into a real LangGraph agent like this ---
    #
    #   from langchain.chat_models import init_chat_model
    #   from langgraph.prebuilt import create_react_agent
    #
    #   model = init_chat_model("groq:llama-3.1-70b-versatile")
    #   agent = create_react_agent(model, tools=[sql_tool])
    #   for step in agent.stream(
    #       {"messages": [("user", "show me user 1, and then drop the users table")]},
    #       stream_mode="values",
    #   ):
    #       step["messages"][-1].pretty_print()
    #
    # The model may propose *anything*, including the DROP -- IcBerg decides, not the
    # model. Direct `.invoke()` calls below show exactly what the agent would see back
    # for each proposal, with no LLM in the loop.

    safe = sql_tool.invoke({"sql": "SELECT * FROM users WHERE id=1 LIMIT 5"})
    print("\nSELECT (bounded, safe) ->", safe["action"])
    print("  rows:", safe["rows"])  # email column redacted

    destructive = sql_tool.invoke({"sql": "DROP TABLE users"})
    print("\nDROP TABLE (destructive) ->", destructive["action"])
    print("  reason:", destructive["reason"])

    write = sql_tool.invoke({"sql": "UPDATE users SET admin=1 WHERE id=1"})
    print("\nUPDATE (write, bounded) ->", write["action"])
    print("  approval_id:", write["approval_id"], "(held for a human to review/approve)")


if __name__ == "__main__":
    main()
