"""Shared fixtures for the Phase 3 integration-surface tests: a throwaway SQLite
database with a synthetic `users` table, matching the schema convention
`tests/api/test_gateway_api.py` and `tests/security/test_governance_runtime.py` already
use (`users(id, name, email, ssn, ssn_num, admin)`). Every row is fabricated for these
tests — no real data.
"""

from __future__ import annotations

import sqlite3

import pytest

USERS_SCHEMA_SQL = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    ssn TEXT,
    ssn_num INTEGER,
    admin INTEGER
)
"""


@pytest.fixture
def db_path(tmp_path) -> str:
    """A throwaway SQLite file, seeded with two fabricated `users` rows."""
    path = str(tmp_path / "integration.sqlite")
    conn = sqlite3.connect(path)
    try:
        conn.execute(USERS_SCHEMA_SQL)
        conn.executemany(
            "INSERT INTO users (id, name, email, ssn, ssn_num, admin) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, "Alice Smith", "alice@example.com", "111-22-3333", 111223333, 0),
                (2, "Bob Jones", "bob@example.com", "222-33-4444", 222334444, 0),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return path
