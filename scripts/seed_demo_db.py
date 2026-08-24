#!/usr/bin/env python3
"""Seed a demo SQLite database for IcBerg's reference agent (`backend/agent/*`,
`examples/reference_agent.py`, `tests/e2e/`).

Writes directly with `sqlite3` -- this script is the one place in the reference-agent
surface that touches a raw DB connection, because seeding fixture data is not a query an
agent proposes. Governance only applies once something starts *querying* this database
through `icberg`/`backend.integrations.langgraph_tool.GovernedSQLTool`, which is exactly
what `backend/agent/*` and `examples/reference_agent.py` do afterward.

Creates two tables, both entirely synthetic (no real data):
  passengers - small, fabricated Titanic-like rows. No PII.
  users      - fabricated accounts with `email`/`ssn` columns -- exactly the column
               names `backend.core.redaction` treats as PII by default, so a governed
               SELECT against this table always comes back redacted.

Idempotent: safe to run any number of times. Tables are created if missing and their
rows are always reset to the same fixed seed set, so the end state after 1 run is
identical to the end state after 100.

Usage:
  .venv/bin/python scripts/seed_demo_db.py [--db-path PATH]

The path can also be set via the ICBERG_DEMO_DB_PATH environment variable; --db-path
(if given) takes precedence over that, which takes precedence over the default
`data/demo.sqlite`.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "demo.sqlite"

PASSENGERS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS passengers (
    id INTEGER PRIMARY KEY,
    name TEXT,
    pclass INTEGER,
    sex TEXT,
    age REAL,
    survived INTEGER
)
"""

USERS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT,
    email TEXT,
    ssn TEXT
)
"""

# Small, fabricated Titanic-like rows. No PII.
PASSENGERS = [
    (1, "Braund, Mr. Owen Harris", 3, "male", 22.0, 0),
    (2, "Cumings, Mrs. John Bradley", 1, "female", 38.0, 1),
    (3, "Heikkinen, Miss. Laina", 3, "female", 26.0, 1),
    (4, "Futrelle, Mrs. Jacques Heath", 1, "female", 35.0, 1),
    (5, "Allen, Mr. William Henry", 3, "male", 35.0, 0),
    (6, "Moran, Mr. James", 3, "male", 27.0, 0),
    (7, "McCarthy, Mr. Timothy J", 1, "male", 54.0, 0),
    (8, "Palsson, Master. Gosta Leonard", 3, "male", 2.0, 0),
    (9, "Johnson, Mrs. Oscar W", 3, "female", 27.0, 1),
    (10, "Nasser, Mrs. Nicholas", 2, "female", 14.0, 1),
]

# Fabricated accounts -- synthetic PII only, no real people. `email`/`ssn` are exactly
# the column names `backend.core.redaction` treats as PII by default, so a governed
# SELECT against this table always comes back redacted regardless of policy config.
USERS = [
    (1, "alice", "alice@example.com", "111-22-3333"),
    (2, "bob", "bob@example.com", "222-33-4444"),
    (3, "carol", "carol@example.com", "333-44-5555"),
]


def seed(db_path: str | Path) -> Path:
    """Create (if needed) and (re)populate `passengers`/`users` at `db_path` with the
    fixed seed rows above. Idempotent -- safe to call repeatedly.

    Returns:
        The resolved `Path` that was seeded.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(PASSENGERS_SCHEMA_SQL)
        conn.execute(USERS_SCHEMA_SQL)

        # Idempotent: reset both tables to the fixed seed set every run, rather than
        # accumulating duplicate rows on repeated invocations.
        conn.execute("DELETE FROM passengers")
        conn.execute("DELETE FROM users")

        conn.executemany(
            "INSERT INTO passengers (id, name, pclass, sex, age, survived) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            PASSENGERS,
        )
        conn.executemany(
            "INSERT INTO users (id, username, email, ssn) VALUES (?, ?, ?, ?)",
            USERS,
        )
        conn.commit()
    finally:
        conn.close()

    return db_path


def resolve_db_path(cli_path: str | None) -> Path:
    """Precedence: --db-path > $ICBERG_DEMO_DB_PATH > data/demo.sqlite."""
    if cli_path:
        return Path(cli_path)
    env_path = os.environ.get("ICBERG_DEMO_DB_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_DB_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed IcBerg's demo SQLite database (idempotent).")
    parser.add_argument(
        "--db-path",
        help="Where to write the demo SQLite file (default: data/demo.sqlite, or $ICBERG_DEMO_DB_PATH)",
    )
    args = parser.parse_args()

    db_path = resolve_db_path(args.db_path)
    seed(db_path)
    print(
        f"[IcBerg] Seeded demo DB at {db_path} "
        f"({len(PASSENGERS)} passengers, {len(USERS)} users)."
    )


if __name__ == "__main__":
    main()
