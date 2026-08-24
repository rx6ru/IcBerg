"""Phase 1 runtime governance tests (`.devdocs/PHASE1_GATES.md`, P1.0-P1.10).

Exercises the actual execution boundary the Phase-0 `GovernanceGate` decision is wired
into: least-privilege execution (`executor.py`), PII redaction (`redaction.py`), a
tamper-evident audit log (`audit.py`), and the `Gateway` that composes all three.

Synthetic SQLite schema throughout: `users(id INTEGER, name TEXT, email TEXT, ssn TEXT,
admin INTEGER)`. All data is fabricated for the test (no real records).

The two crown gates (P1.2, P1.7) are engine-/crypto-grounded on purpose: they assert on
what SQLite itself raises and what a from-scratch hash recomputation says, not on this
module's own bookkeeping, so a bug in this test file's assertions can't self-certify.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from backend.core.audit import GENESIS_HASH, AuditLog, _compute_entry_hash, hash_result_rows
from backend.core.executor import ExecutionResult, ReadOnlyExecutor, WriteExecutor
from backend.core.gateway import Gateway
from backend.core.redaction import redact_rows, redact_text
from backend.core.schema_catalog import SchemaCatalog
from backend.core.sql_governance import GovernanceGate

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    ssn TEXT,
    ssn_num INTEGER,
    admin INTEGER
)
"""

# `orders` is deliberately schema-unknown to this module (no PII columns, no PII-keyword
# names) — it stands in for the confirmed HIGH leak's co-present base table: a top-level
# `SELECT *` that spans BOTH this table AND a derived source can't be star-expanded by
# `qualify()` (this table's real column list is genuinely unknown to `redaction.py`),
# which is exactly the condition that made `_provenance_pii_columns` fail open pre-fix.
ORDERS_SCHEMA_SQL = """
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    total INTEGER,
    note TEXT
)
"""


# Views exercising the schema-aware view-inlining fix (P1.13 extension —
# `redaction._inline_views`): `vnum` is the exact confirmed leak from the security
# review (a bare-numeric PII column, `ssn_num`, renamed through a view to a
# non-PII-looking alias, `token` — no PII `exp.Column` appears anywhere in a query that
# only names the view, so query-text-only provenance analysis had no way to trace it).
# `vnum2` is a view built on `vnum` (view-on-view), renaming AGAIN. `v_orders` is a
# genuinely non-PII view (over the non-PII `orders` table) — the no-over-redact
# counterpart proving this fix doesn't start redacting an ordinary view just because it
# IS a view. `v_email`/`v_ssn_dashed` rename already-PII-shaped values (a real email
# string, a dashed SSN string) through a view — these were already caught by the
# value-pattern scan even before this fix (see redaction.py's module docstring), kept
# here as an explicit regression guard that the new view-inlining machinery doesn't
# somehow interfere with that pre-existing, independent layer.
VIEWS_SQL = """
CREATE VIEW vnum AS SELECT id AS uid, ssn_num AS token FROM users;
CREATE VIEW vnum2 AS SELECT uid AS id2, token AS secret FROM vnum;
CREATE VIEW v_orders AS SELECT id AS uid, total AS amt FROM orders;
CREATE VIEW v_email AS SELECT id AS uid, email AS contact FROM users;
CREATE VIEW v_ssn_dashed AS SELECT id AS uid, ssn AS s FROM users;
"""

# A hand-built `SchemaCatalog` matching `SCHEMA_SQL`/`ORDERS_SCHEMA_SQL`/`VIEWS_SQL`
# above, for tests that call `redact_rows` directly (in isolation, no `Gateway`/
# `ReadOnlyExecutor` involved) with a schema — as opposed to the `..._end_to_end_via_
# gateway` variants, which exercise the REAL introspection of `db_path`'s live SQLite
# connection (`ReadOnlyExecutor.get_schema_catalog` -> `schema_catalog
# .introspect_sqlite_schema`) end to end. Both are exercised deliberately: this constant
# pins `redact_rows`'s own `schema` contract directly; the end-to-end tests prove the
# executor/gateway plumbing that produces an equivalent catalog from a real connection.
_USERS_ORDERS_SCHEMA = SchemaCatalog(
    tables={
        "users": {
            "id": "INTEGER",
            "name": "TEXT",
            "email": "TEXT",
            "ssn": "TEXT",
            "ssn_num": "INTEGER",
            "admin": "INTEGER",
        },
        "orders": {"id": "INTEGER", "user_id": "INTEGER", "total": "INTEGER", "note": "TEXT"},
    },
    views={
        "vnum": "SELECT id AS uid, ssn_num AS token FROM users",
        "vnum2": "SELECT uid AS id2, token AS secret FROM vnum",
        "v_orders": "SELECT id AS uid, total AS amt FROM orders",
    },
)


@pytest.fixture
def db_path(tmp_path) -> str:
    """A throwaway SQLite file seeded with the synthetic `users` and `orders` tables, a
    few rows each, and `VIEWS_SQL`'s views over them.

    `ssn_num` is a fabricated, dashless, INTEGER-typed mirror of `ssn` (never a real
    SSN) — added to reproduce/regression-guard HIGH #2 (numeric PII bypassing the
    `isinstance(value, str)`-gated value-pattern scan) and HIGH #1's fail-closed
    provenance tracing when that numeric column is hidden behind a subquery/CTE alias.

    `orders` (id, user_id, total, note) is fabricated, non-PII order data — added to
    reproduce/regression-guard the derived-star-JOIN HIGH leak (see
    `TestProvenanceRedaction`'s `test_provenance_redaction_star_join_*` cases): a
    top-level `SELECT *` spanning both this base table and a derived `users`-sourced
    subquery/CTE.

    Every `Gateway.handle` call against this file automatically gets a live
    `SchemaCatalog` — `ReadOnlyExecutor.get_schema_catalog` introspects this exact file,
    views included, via `schema_catalog.introspect_sqlite_schema` — so any end-to-end
    test below that queries a view name is exercising the real schema-aware path, not a
    hand-built `SchemaCatalog` fed directly into `redact_rows`.
    """
    path = str(tmp_path / "governance.sqlite")
    conn = sqlite3.connect(path)
    try:
        conn.execute(SCHEMA_SQL)
        conn.executemany(
            "INSERT INTO users (id, name, email, ssn, ssn_num, admin) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, "Alice Smith", "alice@example.com", "123-45-6789", 123456789, 0),
                (2, "Bob Jones", "bob@example.com", "987-65-4321", 987654321, 1),
                (3, "Carol Lee", "carol@example.com", "555-11-2222", 555112222, 0),
            ],
        )
        conn.execute(ORDERS_SCHEMA_SQL)
        conn.executemany(
            "INSERT INTO orders (id, user_id, total, note) VALUES (?, ?, ?, ?)",
            [
                (10, 1, 500, "ok"),
                (11, 2, 250, "pending"),
                (12, 3, 75, "ok"),
            ],
        )
        conn.executescript(VIEWS_SQL)
        conn.commit()
    finally:
        conn.close()
    return path


def _fixed_clock(start: datetime | None = None):
    """A deterministic, monotonically-advancing clock for AuditLog tests."""
    base = start or datetime(2026, 1, 1, tzinfo=timezone.utc)
    state = {"n": 0}

    def _clock() -> datetime:
        state["n"] += 1
        return base + timedelta(seconds=state["n"])

    return _clock


@pytest.fixture
def audit_log() -> AuditLog:
    return AuditLog(":memory:", clock=_fixed_clock())


# ---------------------------------------------------------------------------
# P1.1 - readonly_execute: a read query executes via the read-only executor
# ---------------------------------------------------------------------------

class TestReadonlyExecute:
    def test_readonly_execute_returns_rows(self, db_path: str) -> None:
        executor = ReadOnlyExecutor(db_path)
        result = executor.execute("SELECT id, name, email FROM users WHERE id = 1")

        assert result.error is None, result
        assert result.rowcount == 1, result
        assert result.rows == [{"id": 1, "name": "Alice Smith", "email": "alice@example.com"}]
        assert result.columns == ["id", "name", "email"]
        assert result.truncated is False
        assert result.latency_ms >= 0

    def test_readonly_execute_columns_match_select_list(self, db_path: str) -> None:
        executor = ReadOnlyExecutor(db_path)
        result = executor.execute("SELECT name, ssn FROM users ORDER BY id")

        assert result.error is None, result
        assert result.columns == ["name", "ssn"]
        assert result.rowcount == 3
        assert all(set(row.keys()) == {"name", "ssn"} for row in result.rows)


# ---------------------------------------------------------------------------
# P1.2 - CROWN: engine_rejects_write
# ---------------------------------------------------------------------------

class TestEngineRejectsWrite:
    """The read-only executor's ENGINE (SQLite's mode=ro file open), not the policy gate,
    must reject a forced write, and the underlying rows must be unchanged afterward."""

    def test_engine_rejects_write_update(self, db_path: str) -> None:
        executor = ReadOnlyExecutor(db_path)

        result = executor.execute("UPDATE users SET admin = 1 WHERE id = 1")

        assert result.error is not None, "engine must report an error, not silently succeed"
        assert "readonly database" in result.error.lower(), result.error
        assert result.rows == []

        # Re-query through the same read-only path: row must be unchanged.
        check = executor.execute("SELECT admin FROM users WHERE id = 1")
        assert check.error is None, check
        assert check.rows == [{"admin": 0}], "UPDATE must not have mutated the row"

    def test_engine_rejects_write_delete(self, db_path: str) -> None:
        executor = ReadOnlyExecutor(db_path)

        result = executor.execute("DELETE FROM users")

        assert result.error is not None
        assert "readonly database" in result.error.lower(), result.error

        check = executor.execute("SELECT COUNT(*) AS n FROM users")
        assert check.error is None, check
        assert check.rows == [{"n": 3}], "DELETE must not have removed any row"

    def test_engine_rejects_write_insert(self, db_path: str) -> None:
        executor = ReadOnlyExecutor(db_path)

        result = executor.execute("INSERT INTO users (id, name) VALUES (99, 'Mallory')")

        assert result.error is not None
        assert "readonly database" in result.error.lower(), result.error

        check = executor.execute("SELECT COUNT(*) AS n FROM users")
        assert check.rows == [{"n": 3}], "INSERT must not have added a row"

    def test_engine_rejects_write_but_write_executor_can(self, db_path: str) -> None:
        """Contrast case: the write-capable path (never used for reads) CAN mutate,
        proving the read-only rejection above is a real privilege boundary, not an
        accident of the SQL itself being malformed."""
        write_executor = WriteExecutor(db_path)
        result = write_executor.execute("UPDATE users SET admin = 1 WHERE id = 1")
        assert result.error is None, result

        check = ReadOnlyExecutor(db_path).execute("SELECT admin FROM users WHERE id = 1")
        assert check.rows == [{"admin": 1}]


# ---------------------------------------------------------------------------
# P1.3 - resource_limit: forced row cap + timeout
# ---------------------------------------------------------------------------

class TestResourceLimit:
    def test_resource_limit_row_cap_truncates(self, tmp_path) -> None:
        path = str(tmp_path / "many_rows.sqlite")
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE big (id INTEGER PRIMARY KEY, val TEXT)")
        conn.executemany("INSERT INTO big (id, val) VALUES (?, ?)", [(i, f"v{i}") for i in range(120)])
        conn.commit()
        conn.close()

        executor = ReadOnlyExecutor(path, max_rows=50)
        result = executor.execute("SELECT * FROM big")

        assert result.error is None, result
        assert result.truncated is True
        assert len(result.rows) == 50
        assert result.rowcount == 50

    def test_resource_limit_row_cap_no_truncation_when_under_cap(self, db_path: str) -> None:
        executor = ReadOnlyExecutor(db_path, max_rows=50)
        result = executor.execute("SELECT * FROM users")

        assert result.error is None, result
        assert result.truncated is False
        assert len(result.rows) == 3

    def test_resource_limit_timeout_aborts_recursive_cte(self, db_path: str) -> None:
        """A recursive-CTE row explosion must be aborted by the wall-clock watchdog well
        before it could ever finish, not allowed to run to completion."""
        executor = ReadOnlyExecutor(db_path, timeout_seconds=0.2)
        exploding_sql = """
            WITH RECURSIVE cnt(x) AS (
                SELECT 1
                UNION ALL
                SELECT x + 1 FROM cnt WHERE x < 1000000000
            )
            SELECT COUNT(*) AS n FROM cnt
        """

        result = executor.execute(exploding_sql)

        assert result.error is not None, "must abort, not run 1e9 recursion steps to completion"
        assert "timeout" in result.error.lower() or "interrupt" in result.error.lower(), result.error
        # Aborted near the configured timeout, not after actually finishing the query.
        assert result.latency_ms < 5000, f"took {result.latency_ms}ms; watchdog should have aborted near 200ms"


# ---------------------------------------------------------------------------
# P1.4 - pii_redaction: email/ssn/phone masked
# ---------------------------------------------------------------------------

class TestPiiRedaction:
    def test_pii_redaction_masks_email_and_ssn(self) -> None:
        columns = ["id", "name", "email", "ssn"]
        rows = [{"id": 1, "name": "Alice Smith", "email": "alice@example.com", "ssn": "123-45-6789"}]

        redacted, report = redact_rows(rows, columns)

        assert redacted[0]["email"] == "[REDACTED]"
        assert redacted[0]["ssn"] == "[REDACTED]"
        assert "email" in report["columns_redacted"]
        assert "ssn" in report["columns_redacted"]
        assert report["values_masked"] == 2

    def test_pii_redaction_masks_phone_column(self) -> None:
        columns = ["id", "phone"]
        rows = [{"id": 1, "phone": "555-123-4567"}]

        redacted, report = redact_rows(rows, columns)

        assert redacted[0]["phone"] == "[REDACTED]"
        assert "phone" in report["columns_redacted"]
        assert report["values_masked"] == 1

    def test_pii_redaction_catches_oddly_named_aliased_column(self) -> None:
        """`col1` carries no name signal at all; only the value scan can catch this."""
        columns = ["id", "col1"]
        rows = [{"id": 1, "col1": "carol@example.com,bob@example.com"}]

        redacted, report = redact_rows(rows, columns)

        assert "@example.com" not in redacted[0]["col1"]
        assert "EMAIL_REDACTED" in redacted[0]["col1"]
        assert "col1" in report["columns_redacted"]
        assert report["values_masked"] >= 1

    def test_pii_redaction_end_to_end_via_readonly_executor(self, db_path: str) -> None:
        executor = ReadOnlyExecutor(db_path)
        result = executor.execute("SELECT id, name, email, ssn FROM users WHERE id = 1")
        redacted, report = redact_rows(result.rows, result.columns)

        assert redacted[0]["email"] == "[REDACTED]"
        assert redacted[0]["ssn"] == "[REDACTED]"
        assert redacted[0]["name"] == "Alice Smith", "name is not a classified PII column"
        assert redacted[0]["id"] == 1


# ---------------------------------------------------------------------------
# P1.5 - no_over_redact: non-PII columns/values pass through unchanged
# ---------------------------------------------------------------------------

class TestNoOverRedact:
    def test_no_over_redact_id_admin_age_columns(self) -> None:
        columns = ["id", "admin", "age"]
        rows = [{"id": 7, "admin": 1, "age": 42}]

        redacted, report = redact_rows(rows, columns)

        assert redacted == rows, "plainly non-PII columns must pass through byte-for-byte"
        assert report["columns_redacted"] == []
        assert report["values_masked"] == 0

    def test_no_over_redact_name_column_not_classified(self) -> None:
        """`name` is deliberately NOT in the PII column-keyword list; a plain name value
        (no embedded email/phone/card/ssn pattern) must not be touched."""
        columns = ["id", "name"]
        rows = [{"id": 1, "name": "Alice Smith"}]

        redacted, report = redact_rows(rows, columns)

        assert redacted[0]["name"] == "Alice Smith"
        assert "name" not in report["columns_redacted"]

    def test_no_over_redact_mixed_row(self) -> None:
        columns = ["id", "name", "email", "admin"]
        rows = [{"id": 3, "name": "Carol Lee", "email": "carol@example.com", "admin": 0}]

        redacted, _report = redact_rows(rows, columns)

        assert redacted[0]["id"] == 3
        assert redacted[0]["admin"] == 0
        assert redacted[0]["name"] == "Carol Lee"
        assert redacted[0]["email"] == "[REDACTED]"

    def test_no_over_redact_plain_select_star_from_orders(self) -> None:
        """A bare `SELECT *` directly over a base table with no PII columns at all (no
        derived source in the query for the derived-star-JOIN fallback to walk, and no
        PII-keyword column name) must still redact nothing — pins the fallback added for
        the derived-star-JOIN HIGH leak to an empty result when there is no derived
        source present, not a change from prior no-derived-source behavior."""
        columns = ["id", "user_id", "total", "note"]
        rows = [
            {"id": 10, "user_id": 1, "total": 500, "note": "ok"},
            {"id": 11, "user_id": 2, "total": 250, "note": "pending"},
        ]
        sql = "SELECT * FROM orders LIMIT 5"

        redacted, report = redact_rows(rows, columns, sql=sql)

        assert redacted == rows
        assert report["columns_redacted"] == []
        assert report["values_masked"] == 0

    def test_no_over_redact_plain_select_star_from_orders_end_to_end_via_gateway(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        """Same case as above, through the real gateway/executor against the live SQLite
        fixture's `orders` table (bounded with `WHERE`+`LIMIT` so the Phase-0 policy
        gate auto-`allow`s it — an unbounded `SELECT * FROM orders LIMIT 5` alone is
        `hold` at the gate itself, unrelated to this redaction test's own concern)."""
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT * FROM orders WHERE id > 0 LIMIT 5",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["redaction_report"]["columns_redacted"] == []
        assert result["redaction_report"]["values_masked"] == 0
        for row in result["rows"]:
            assert set(row) == {"id", "user_id", "total", "note"}

    def test_no_over_redact_non_pii_projection_beside_star(self) -> None:
        """Boundary pin for the scalar-subquery-beside-star fix: a non-star projection
        beside a top-level star that resolves to a genuinely non-PII source column
        (`total AS c`, `total` carrying no PII keyword and no derived/PII lineage) must
        stay unredacted — the fix is provenance tracing of that projection, not a
        blanket redact-everything-beside-a-star rule."""
        columns = ["id", "user_id", "total", "note", "c"]
        rows = [{"id": 10, "user_id": 1, "total": 500, "note": "ok", "c": 500}]
        sql = "SELECT *, total AS c FROM orders LIMIT 5"

        redacted, report = redact_rows(rows, columns, sql=sql)

        assert redacted == rows
        assert report["columns_redacted"] == []
        assert report["values_masked"] == 0

    def test_no_over_redact_non_pii_projection_beside_star_end_to_end_via_gateway(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        """Same control, through the real gateway/executor (bounded with `WHERE`+
        `LIMIT` so the Phase-0 policy gate auto-`allow`s it)."""
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT *, total AS c FROM orders WHERE id > 0 LIMIT 5",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["redaction_report"]["columns_redacted"] == []
        assert result["redaction_report"]["values_masked"] == 0
        for row in result["rows"]:
            assert row["c"] == row["total"]

    def test_no_over_redact_ambiguous_column_beside_star_no_pii_source(self) -> None:
        """Fail-closed boundary pin for the generalized ambiguous-column fix (see
        `TestProvenanceRedaction.test_provenance_redaction_ambiguous_column_beside_star_numeric_ssn`):
        the exact same ambiguous-unqualified-column-beside-a-star JOIN shape as the
        `renamed` leak, but with NO PII-named column anywhere in the statement — the
        derived source projects `total` (an ordinary `orders` column), not a PII
        column. `_query_references_pii_source` must be `False` here, keeping the
        pre-existing permissive ambiguous-column fallback, so this ordinary,
        PII-free analytics query is NOT over-redacted just because it shares the same
        unresolvable-JOIN shape as the leak."""
        columns = ["s", "id", "user_id", "total", "note", "renamed"]
        rows = [{"s": 500, "id": 10, "user_id": 1, "total": 500, "note": "ok", "renamed": 500}]
        sql = (
            "SELECT *, s AS renamed FROM (SELECT total AS s FROM orders) a "
            "JOIN orders o ON o.user_id=1 WHERE o.id>0 LIMIT 5"
        )

        redacted, report = redact_rows(rows, columns, sql=sql)

        assert redacted == rows
        assert report["columns_redacted"] == []
        assert report["values_masked"] == 0

    def test_no_over_redact_ambiguous_column_beside_star_no_pii_source_end_to_end_via_gateway(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        """Same control, through the real gateway/executor against the live SQLite
        fixture."""
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT *, s AS renamed FROM (SELECT total AS s FROM orders) a "
            "JOIN orders o ON o.user_id=1 WHERE o.id>0 LIMIT 5",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["redaction_report"]["columns_redacted"] == []
        assert result["redaction_report"]["values_masked"] == 0
        for row in result["rows"]:
            assert row["renamed"] != "[REDACTED]"
            assert row["s"] != "[REDACTED]"

    def test_no_over_redact_id_admin_resolvable_in_pii_touching_query(self) -> None:
        """Positively-resolved-to-non-PII boundary pin: `id`/`admin` in a query that
        DOES reference a real PII column elsewhere in its SELECT list (`email`, making
        `_query_references_pii_source` `True` and the new fail-closed policy active)
        must still NOT be redacted — both resolve unambiguously (`users` is the sole
        source) to a base column whose own name genuinely isn't PII, which is exactly
        the "positively resolved to a proven-non-PII base column" case the fail-closed
        policy is built to exempt, not the ambiguity it targets. `email` itself must
        still be redacted for its own, unrelated (name-classification) reason."""
        columns = ["id", "admin", "email"]
        rows = [{"id": 1, "admin": 0, "email": "alice@example.com"}]
        sql = "SELECT id, admin, email FROM users WHERE id=1 LIMIT 1"

        redacted, report = redact_rows(rows, columns, sql=sql)

        assert redacted[0]["id"] == 1
        assert redacted[0]["admin"] == 0
        assert redacted[0]["email"] == "[REDACTED]"
        assert report["columns_redacted"] == ["email"]

    def test_no_over_redact_id_admin_end_to_end_via_gateway(self, db_path: str, audit_log: AuditLog) -> None:
        """The literal control query from the redaction verification pass: `SELECT id,
        admin FROM users WHERE id=1 LIMIT 1` must return `id`/`admin` unredacted
        end-to-end through the real gateway/executor."""
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT id, admin FROM users WHERE id=1 LIMIT 1",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["rows"][0] == {"id": 1, "admin": 0}
        assert result["redaction_report"]["columns_redacted"] == []
        assert result["redaction_report"]["values_masked"] == 0

    # -- Non-PII view: the schema-aware view-inlining fix must not start redacting a
    # view just because it IS a view (`v_orders`: `id AS uid, total AS amt FROM orders`,
    # neither column PII by name or by lineage) ---------------------------------------

    def test_no_over_redact_non_pii_view(self) -> None:
        columns = ["uid", "amt"]
        rows = [{"uid": 10, "amt": 500}]
        sql = "SELECT uid, amt FROM v_orders WHERE uid=10 LIMIT 5"

        redacted, report = redact_rows(rows, columns, sql=sql, schema=_USERS_ORDERS_SCHEMA)

        assert redacted == rows
        assert report["columns_redacted"] == []
        assert report["values_masked"] == 0

    def test_no_over_redact_non_pii_view_star(self) -> None:
        columns = ["uid", "amt"]
        rows = [{"uid": 10, "amt": 500}]
        sql = "SELECT * FROM v_orders WHERE uid=10 LIMIT 5"

        redacted, report = redact_rows(rows, columns, sql=sql, schema=_USERS_ORDERS_SCHEMA)

        assert redacted == rows
        assert report["columns_redacted"] == []
        assert report["values_masked"] == 0

    def test_no_over_redact_non_pii_view_end_to_end_via_gateway(self, db_path: str, audit_log: AuditLog) -> None:
        """Same control, through the real gateway/executor against the live SQLite
        fixture's actual `CREATE VIEW v_orders AS SELECT id AS uid, total AS amt FROM
        orders`."""
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT * FROM v_orders WHERE uid=10 LIMIT 5",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["rows"] == [{"uid": 10, "amt": 500}]
        assert result["redaction_report"]["columns_redacted"] == []
        assert result["redaction_report"]["values_masked"] == 0


# ---------------------------------------------------------------------------
# P1.6 - audit_chain_verifies
# ---------------------------------------------------------------------------

class TestAuditChainVerifies:
    def test_audit_chain_verifies_after_appends(self, audit_log: AuditLog) -> None:
        for i in range(5):
            audit_log.append(
                actor="agent-1",
                proposed_sql=f"SELECT {i}",
                classification="read",
                action="allow",
                matched_rules=[],
                rows_returned=1,
                latency_ms=5,
                result_hash=hash_result_rows([{"n": i}]),
            )

        ok, first_broken = audit_log.verify()
        assert ok is True
        assert first_broken is None

    def test_audit_chain_verifies_empty_log(self, audit_log: AuditLog) -> None:
        ok, first_broken = audit_log.verify()
        assert ok is True
        assert first_broken is None

    def test_audit_chain_first_entry_chains_to_genesis(self, audit_log: AuditLog) -> None:
        entry = audit_log.append(
            actor="agent-1", proposed_sql="SELECT 1", classification="read", action="allow",
            matched_rules=[], rows_returned=1, latency_ms=1, result_hash=hash_result_rows([]),
        )
        assert entry.prev_hash == GENESIS_HASH


# ---------------------------------------------------------------------------
# P1.7 - CROWN: audit_tamper_detected
# ---------------------------------------------------------------------------

class TestAuditTamperDetected:
    def test_audit_tamper_detected_mutated_action(self, audit_log: AuditLog) -> None:
        for i in range(3):
            audit_log.append(
                actor="agent-1",
                proposed_sql=f"SELECT {i}",
                classification="read",
                action="allow",
                matched_rules=[],
                rows_returned=1,
                latency_ms=1,
                result_hash=hash_result_rows([{"n": i}]),
            )

        # Simulate an attacker with direct DB access mutating a stored field on entry #2.
        # The append-only triggers (P1.14) reject an ordinary UPDATE now, so this drops
        # them first — modeling an attacker who has escalated far enough to run DDL, not
        # just DML. The point of this test is the hash-chain layer specifically: even
        # past that escalation, `verify()` must still catch the tamper.
        audit_log.conn.execute("DROP TRIGGER audit_log_no_update")
        audit_log.conn.execute("UPDATE audit_log SET action = 'block' WHERE seq = 2")
        audit_log.conn.commit()

        ok, first_broken = audit_log.verify()
        assert (ok, first_broken) == (False, 2)

    def test_audit_tamper_detected_broken_prev_hash_linkage(self, audit_log: AuditLog) -> None:
        for i in range(3):
            audit_log.append(
                actor="agent-1", proposed_sql=f"SELECT {i}", classification="read", action="allow",
                matched_rules=[], rows_returned=0, latency_ms=0, result_hash=hash_result_rows([]),
            )

        # See the comment in test_audit_tamper_detected_mutated_action above: the
        # append-only trigger (P1.14) must be dropped first to simulate this level of
        # attacker access.
        audit_log.conn.execute("DROP TRIGGER audit_log_no_update")
        audit_log.conn.execute("UPDATE audit_log SET prev_hash = ? WHERE seq = 3", ("f" * 64,))
        audit_log.conn.commit()

        ok, first_broken = audit_log.verify()
        assert (ok, first_broken) == (False, 3)

    def test_audit_tamper_detected_untampered_entries_before_break_still_valid(self, audit_log: AuditLog) -> None:
        """Entries before the tampered one must still verify individually correctly, so
        `verify()` reporting seq=2 (not seq=1) is meaningful, not an artifact of a bug."""
        entry1 = audit_log.append(
            actor="agent-1", proposed_sql="SELECT 1", classification="read", action="allow",
            matched_rules=[], rows_returned=0, latency_ms=0, result_hash=hash_result_rows([]),
        )
        audit_log.append(
            actor="agent-1", proposed_sql="SELECT 2", classification="read", action="allow",
            matched_rules=[], rows_returned=0, latency_ms=0, result_hash=hash_result_rows([]),
        )
        # See the comment in test_audit_tamper_detected_mutated_action above: the
        # append-only trigger (P1.14) must be dropped first to simulate this level of
        # attacker access.
        audit_log.conn.execute("DROP TRIGGER audit_log_no_update")
        audit_log.conn.execute("UPDATE audit_log SET actor = 'mallory' WHERE seq = 2")
        audit_log.conn.commit()

        ok, first_broken = audit_log.verify()
        assert (ok, first_broken) == (False, 2)
        assert entry1.entry_hash != ""  # entry 1 itself was computed and stored correctly


# ---------------------------------------------------------------------------
# P1.8 - audit_fields
# ---------------------------------------------------------------------------

class TestAuditFields:
    REQUIRED_FIELDS = (
        "seq", "timestamp", "actor", "proposed_sql", "classification", "action",
        "matched_rules", "rows_returned", "latency_ms", "result_hash", "prev_hash", "entry_hash",
    )

    def test_audit_fields_all_present(self, audit_log: AuditLog) -> None:
        entry = audit_log.append(
            actor="agent-1",
            proposed_sql="SELECT * FROM users WHERE id = 1 LIMIT 1",
            classification="read",
            action="allow",
            matched_rules=["pii_suspected"],
            rows_returned=1,
            latency_ms=12,
            result_hash=hash_result_rows([{"id": 1}]),
        )

        for field_name in self.REQUIRED_FIELDS:
            assert hasattr(entry, field_name), f"AuditEntry missing field {field_name}"

        assert entry.seq == 1
        assert entry.actor == "agent-1"
        assert entry.classification == "read"
        assert entry.action == "allow"
        assert entry.matched_rules == ["pii_suspected"]
        assert entry.rows_returned == 1
        assert entry.latency_ms == 12
        assert entry.prev_hash == GENESIS_HASH
        assert len(entry.entry_hash) == 64  # sha256 hex digest

    def test_audit_fields_result_hash_excludes_raw_pii(self, audit_log: AuditLog) -> None:
        redacted_rows, _report = redact_rows(
            [{"id": 1, "email": "alice@example.com", "ssn": "123-45-6789"}],
            ["id", "email", "ssn"],
        )
        entry = audit_log.append(
            actor="agent-1", proposed_sql="SELECT id, email, ssn FROM users WHERE id=1 LIMIT 1",
            classification="read", action="allow", matched_rules=[], rows_returned=1,
            latency_ms=3, result_hash=hash_result_rows(redacted_rows),
        )

        assert "alice@example.com" not in entry.result_hash
        assert "123-45-6789" not in entry.result_hash
        assert len(entry.result_hash) == 64


# ---------------------------------------------------------------------------
# P1.9 - end_to_end: allow -> execute+redact+audit; block -> audit with NO execution
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_end_to_end_allow_executes_redacts_and_audits(self, db_path: str, audit_log: AuditLog) -> None:
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT id, name, email, ssn FROM users WHERE id = 1 LIMIT 1",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["rows"] is not None
        assert result["rows"][0]["email"] == "[REDACTED]"
        assert result["rows"][0]["ssn"] == "[REDACTED]"
        assert result["rows"][0]["name"] == "Alice Smith"
        assert result["redaction_report"]["values_masked"] == 2
        assert result["audit_seq"] == 1

        entries = audit_log.entries()
        assert len(entries) == 1
        assert entries[0].action == "allow"
        assert entries[0].rows_returned == 1
        ok, _ = audit_log.verify()
        assert ok is True

    def test_end_to_end_block_audits_with_no_execution(self, db_path: str, audit_log: AuditLog) -> None:
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "DROP TABLE users",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "block", result
        assert result["rows"] is None
        assert result["redaction_report"] is None
        assert "ddl_blocked" in result["matched_rules"]

        # Table must be untouched: DROP TABLE never reached the database.
        check = executor.execute("SELECT COUNT(*) AS n FROM users")
        assert check.error is None, "table should still exist; DROP must never have executed"
        assert check.rows == [{"n": 3}]

        entries = audit_log.entries()
        assert len(entries) == 1
        assert entries[0].action == "block"
        assert entries[0].rows_returned == 0

    def test_end_to_end_hold_audits_with_no_execution(self, db_path: str, audit_log: AuditLog) -> None:
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "UPDATE users SET admin = 1 WHERE id = 1",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "hold", result
        assert result["rows"] is None
        assert "write_requires_approval" in result["matched_rules"]

        check = executor.execute("SELECT admin FROM users WHERE id = 1")
        assert check.rows == [{"admin": 0}], "held write must never have executed"

        entries = audit_log.entries()
        assert entries[0].action == "hold"

    def test_end_to_end_reason_is_pii_scrubbed(self, db_path: str, audit_log: AuditLog) -> None:
        """A synthetic check that the gateway's returned `reason` never contains a raw
        email/ssn/phone pattern, even in an error path."""
        gateway = Gateway()
        executor = ReadOnlyExecutor(db_path)
        result = gateway.handle(
            "SELECT id FROM users WHERE email = 'alice@example.com' LIMIT 1",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )
        assert "alice@example.com" not in result["reason"]
        # Confirm redact_text itself does the scrubbing it claims to.
        assert redact_text("contact alice@example.com") == "contact [EMAIL_REDACTED]"


# ---------------------------------------------------------------------------
# Sanity: the Phase-0 gate is unmodified and still importable/usable directly
# ---------------------------------------------------------------------------

def test_governance_gate_still_used_for_decisions() -> None:
    gate = GovernanceGate()
    decision = gate.evaluate("SELECT * FROM users WHERE id = 1 LIMIT 1")
    assert decision.action == "allow"


# ---------------------------------------------------------------------------
# P1.11 - attach_blocked: ATTACH DATABASE blocked at BOTH gate and connection
# ---------------------------------------------------------------------------

class TestAttachBlocked:
    def test_attach_blocked_at_gate(self) -> None:
        gate = GovernanceGate()
        decision = gate.evaluate("ATTACH DATABASE 'evil.db' AS evil")
        assert decision.action == "block", decision
        assert "attach_blocked" in decision.matched_rules

    def test_attach_blocked_bare_attach_at_gate(self) -> None:
        gate = GovernanceGate()
        decision = gate.evaluate("ATTACH 'evil.db' AS evil")
        assert decision.action == "block", decision
        assert "attach_blocked" in decision.matched_rules

    def test_attach_blocked_detach_at_gate(self) -> None:
        gate = GovernanceGate()
        decision = gate.evaluate("DETACH DATABASE evil")
        assert decision.action == "block", decision
        assert "attach_blocked" in decision.matched_rules

    def test_attach_blocked_string_literal_not_false_positive(self) -> None:
        """A benign literal that merely contains the word "attach" must not trip this
        rule — the check must ignore string-literal content."""
        gate = GovernanceGate()
        decision = gate.evaluate("SELECT * FROM users WHERE name = 'attach the file' LIMIT 1")
        assert "attach_blocked" not in decision.matched_rules

    def test_attach_blocked_column_identifier_not_false_positive(self) -> None:
        """LOW fix: a bare column/identifier literally named `attach` must not trip this
        rule — only the real ATTACH/DETACH *statement*, in statement position, should.
        Bounded so the decision is a real `allow`, not merely "not blocked"."""
        gate = GovernanceGate()
        decision = gate.evaluate("SELECT attach FROM t WHERE attach IS NOT NULL LIMIT 1")
        assert "attach_blocked" not in decision.matched_rules
        assert decision.action == "allow", decision

    def test_attach_blocked_at_connection(self, db_path: str, tmp_path) -> None:
        """A forced ATTACH through the executor (simulating a gate bypass) must be denied
        by the connection's own SQLite authorizer, and must never create the attached
        file on disk — proving this is a real engine-level boundary, not a restatement of
        the gate's own decision."""
        evil_path = tmp_path / "evil_attach_target.db"
        executor = ReadOnlyExecutor(db_path)

        result = executor.execute(f"ATTACH DATABASE '{evil_path}' AS evil")

        assert result.error is not None, "connection must deny ATTACH, not silently allow it"
        assert "authoriz" in result.error.lower(), result.error
        assert not evil_path.exists(), "ATTACH must never have created the target file"

        # Sanity: ordinary reads through the same executor still work afterward.
        check = executor.execute("SELECT COUNT(*) AS n FROM users")
        assert check.error is None, check
        assert check.rows == [{"n": 3}]


# ---------------------------------------------------------------------------
# P1.12 - set_config_blocked: GUC/session mutation via a plain SELECT
# ---------------------------------------------------------------------------

class TestSetConfigBlocked:
    def test_set_config_blocked_plain(self) -> None:
        gate = GovernanceGate()
        decision = gate.evaluate("SELECT set_config('statement_timeout', '0', false)")
        assert decision.action == "block", decision
        assert "set_config_blocked" in decision.matched_rules

    def test_set_config_blocked_schema_qualified(self) -> None:
        gate = GovernanceGate()
        decision = gate.evaluate("SELECT pg_catalog.set_config('search_path', 'public', false)")
        assert decision.action == "block", decision
        assert "set_config_blocked" in decision.matched_rules

    def test_set_config_blocked_quoted_and_cased(self) -> None:
        """Quoting/casing must not evade this — same AST function-name normalization the
        RCE/sequence/DoS deny lists already rely on."""
        gate = GovernanceGate()
        decision = gate.evaluate("SELECT \"SET_CONFIG\"('x', '0', false)")
        assert decision.action == "block", decision
        assert "set_config_blocked" in decision.matched_rules

    def test_pg_reload_conf_blocked(self) -> None:
        gate = GovernanceGate()
        decision = gate.evaluate("SELECT pg_reload_conf()")
        assert decision.action == "block", decision
        assert "set_config_blocked" in decision.matched_rules


# ---------------------------------------------------------------------------
# P1.13 - provenance_redaction: aliased/derived PII traced via sqlglot lineage
# ---------------------------------------------------------------------------

class TestProvenanceRedaction:
    def test_provenance_redaction_substr_ssn_aliased(self) -> None:
        columns = ["s"]
        rows = [{"s": "123"}]
        sql = "SELECT SUBSTR(ssn, 1, 3) AS s FROM users"

        redacted, report = redact_rows(rows, columns, sql=sql)

        assert redacted[0]["s"] == "[REDACTED]"
        assert "s" in report["columns_redacted"]

    def test_provenance_redaction_group_concat_email_aliased(self) -> None:
        columns = ["c"]
        rows = [{"c": "alice@x.co,bob@y.co"}]
        sql = "SELECT GROUP_CONCAT(email) AS c FROM users"

        redacted, report = redact_rows(rows, columns, sql=sql)

        assert redacted[0]["c"] == "[REDACTED]"
        assert "c" in report["columns_redacted"]

    def test_provenance_redaction_both_example_queries_together(self) -> None:
        columns = ["s", "c"]
        rows = [{"s": "123", "c": "alice@x.co"}]
        sql = "SELECT SUBSTR(ssn,1,3) AS s, GROUP_CONCAT(email) AS c FROM users"

        redacted, report = redact_rows(rows, columns, sql=sql)

        assert redacted[0]["s"] == "[REDACTED]"
        assert redacted[0]["c"] == "[REDACTED]"
        assert {"s", "c"} <= set(report["columns_redacted"])

    def test_provenance_redaction_end_to_end_via_gateway(self, db_path: str, audit_log: AuditLog) -> None:
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT SUBSTR(ssn,1,3) AS s, GROUP_CONCAT(email) AS c FROM users WHERE id = 1 LIMIT 1",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["rows"][0]["s"] == "[REDACTED]"
        assert result["rows"][0]["c"] == "[REDACTED]"

    def test_provenance_redaction_no_over_redact_regression(self) -> None:
        """Provenance analysis must not introduce a false positive for a plainly non-PII
        aggregate — regression guard alongside P1.5's own no_over_redact tests."""
        columns = ["total"]
        rows = [{"total": 42}]
        sql = "SELECT COUNT(*) AS total FROM users"

        redacted, report = redact_rows(rows, columns, sql=sql)

        assert redacted == rows
        assert report["columns_redacted"] == []

    # -- HIGH #1 regression: provenance must trace through nested SELECT scopes -------

    def test_provenance_redaction_subquery_hidden_substr_email(self) -> None:
        """Confirmed leak: a subquery-in-FROM that renames `SUBSTR(email,...)` to an
        innocuous alias (`c`) must still be traced back to `email` and redacted — the
        outer column's own name/expression (a bare `c`) carries no PII signal at all."""
        columns = ["c"]
        rows = [{"c": "alice"}]
        sql = "SELECT c FROM (SELECT SUBSTR(email,1,5) AS c FROM users) t WHERE t.c IS NOT NULL LIMIT 1"

        redacted, report = redact_rows(rows, columns, sql=sql)

        assert redacted[0]["c"] == "[REDACTED]"
        assert "c" in report["columns_redacted"]

    def test_provenance_redaction_cte_hidden_substr_ssn(self) -> None:
        """Same HIGH #1 gap, via a CTE instead of a FROM-subquery."""
        columns = ["c"]
        rows = [{"c": "123"}]
        sql = "WITH t AS (SELECT SUBSTR(ssn,1,3) AS c FROM users) SELECT c FROM t WHERE c IS NOT NULL LIMIT 1"

        redacted, report = redact_rows(rows, columns, sql=sql)

        assert redacted[0]["c"] == "[REDACTED]"
        assert "c" in report["columns_redacted"]

    def test_provenance_redaction_subquery_hidden_end_to_end_via_gateway(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        """Same case as above, but through the real gateway/executor against the live
        SQLite fixture — not just `redact_rows` in isolation."""
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT c FROM (SELECT SUBSTR(email,1,5) AS c FROM users) t WHERE t.c IS NOT NULL LIMIT 1",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["rows"][0]["c"] == "[REDACTED]"

    def test_provenance_redaction_numeric_ssn_num_via_subquery_end_to_end(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        """Confirmed leak combining HIGH #1 (provenance blind to nested scopes) and
        HIGH #2 (numeric PII bypassing the `isinstance(value, str)`-gated value scan):
        `ssn_num` is an INTEGER column, hidden behind a subquery alias `c`. Both fixes
        must close this — provenance must trace `c` back to `ssn_num` (PII-classified by
        name) through the nested scope, and the whole-column masking that follows must
        redact the value regardless of its Python `int` type."""
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT c FROM (SELECT ssn_num AS c FROM users) t WHERE t.c IS NOT NULL LIMIT 1",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["rows"][0]["c"] == "[REDACTED]"
        assert result["redaction_report"]["values_masked"] == 1

    # -- LOW regression: pure COUNT(<pii>) reveals no individual value -----------------

    def test_provenance_redaction_count_email_not_redacted(self) -> None:
        """A pure `COUNT(email)` aggregate reveals only a count, never an individual
        email value — must NOT be redacted, unlike `GROUP_CONCAT`/`MIN`/`MAX`/`SUBSTR`
        over the same column (see the next test)."""
        columns = ["total"]
        rows = [{"total": 3}]
        sql = "SELECT COUNT(email) AS total FROM users"

        redacted, report = redact_rows(rows, columns, sql=sql)

        assert redacted == rows
        assert report["columns_redacted"] == []

    def test_provenance_redaction_min_ssn_still_redacted(self) -> None:
        """Unlike `COUNT`, `MIN`/`MAX`/`GROUP_CONCAT`/`SUBSTR` over a PII column DO leak
        an individual (or aggregated) real value and must remain redacted."""
        columns = ["earliest"]
        rows = [{"earliest": "123-45-6789"}]
        sql = "SELECT MIN(ssn) AS earliest FROM users"

        redacted, report = redact_rows(rows, columns, sql=sql)

        assert redacted[0]["earliest"] == "[REDACTED]"
        assert "earliest" in report["columns_redacted"]

    # -- HIGH #3 regression: `SELECT *` over a derived table / ambiguous-but-resolvable
    # unqualified JOIN column must be traced via sqlglot `qualify()`, not fail open ------

    def test_provenance_redaction_select_star_over_derived_table_numeric_ssn(self) -> None:
        """Confirmed leak: a top-level `SELECT *` over a subquery-in-`FROM` that aliases
        `ssn_num` to `c` must still be redacted. Pre-fix, any top-level `*` made
        `_provenance_pii_columns` bail out entirely (`return set()`) on the assumption a
        wildcard always keeps the *source table's* own column names — true for a `*`
        directly over a base table, false here since the derived table's own SELECT
        renamed the column away from `ssn_num`, and the bare 9-digit value has no dashes
        and no keyword prefix for the value-regex backstops to catch either."""
        columns = ["c"]
        rows = [{"c": 123456789}]
        sql = "SELECT * FROM (SELECT ssn_num AS c FROM users WHERE id=1) t WHERE c IS NOT NULL LIMIT 1"

        redacted, report = redact_rows(rows, columns, sql=sql)

        assert redacted[0]["c"] == "[REDACTED]"
        assert "c" in report["columns_redacted"]
        assert report["values_masked"] == 1

    def test_provenance_redaction_select_star_over_derived_table_end_to_end_via_gateway(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        """Same leak as above, through the real gateway/executor against the live SQLite
        fixture — this is the exact query the security review reported as `allow` with
        `rows=[{'c': 123456789}]` leaked."""
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT * FROM (SELECT ssn_num AS c FROM users WHERE id=1) t WHERE c IS NOT NULL LIMIT 1",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["rows"][0]["c"] == "[REDACTED]"
        assert result["redaction_report"]["values_masked"] == 1

    def test_provenance_redaction_ambiguous_join_alias_numeric_ssn(self) -> None:
        """Confirmed leak: `s` is unqualified and ambiguous among two `JOIN` sources
        (derived table `a`, base table `b`) — pre-fix, this ambiguity made
        `_column_is_pii` fall back to classifying the *output alias* `s`'s own name
        (no PII keyword), missing that `s` only actually exists on the `a` side and
        traces back to `ssn_num`. sqlglot's own `qualify()` can prove `s` resolves to `a`
        alone (only `a` self-describingly projects a column named `s`), closing the gap
        without needing any real database schema."""
        columns = ["c"]
        rows = [{"c": 123456789}]
        sql = (
            "SELECT s AS c FROM (SELECT ssn_num AS s FROM users WHERE id=1) a "
            "JOIN users b ON b.id=1 WHERE a.s IS NOT NULL LIMIT 5"
        )

        redacted, report = redact_rows(rows, columns, sql=sql)

        assert redacted[0]["c"] == "[REDACTED]"
        assert "c" in report["columns_redacted"]
        assert report["values_masked"] == 1

    def test_provenance_redaction_ambiguous_join_alias_end_to_end_via_gateway(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        """Same leak as above, through the real gateway/executor — the exact query the
        security review reported as `allow` with `rows=[{'c': 123456789}]` leaked."""
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT s AS c FROM (SELECT ssn_num AS s FROM users WHERE id=1) a "
            "JOIN users b ON b.id=1 WHERE a.s IS NOT NULL LIMIT 5",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["rows"][0]["c"] == "[REDACTED]"
        assert result["redaction_report"]["values_masked"] == 1

    def test_provenance_redaction_qualified_derived_join_non_pii_not_over_redacted(self) -> None:
        """Over-redaction boundary pin for the HIGH #3 fix: a *qualified* reference into a
        derived-table JOIN source (`a.name`) and a *qualified* reference into a base-table
        JOIN source (`b.admin`) are both genuinely resolvable — via the same `qualify()`
        pass that closes the two leaks above — to definitively non-PII columns, and must
        stay unredacted. Output aliases (`x`, `y`) are deliberately non-signal-carrying so
        only provenance tracing (not output-name classification) is what's under test."""
        columns = ["x", "y"]
        rows = [{"x": "Alice Smith", "y": 0}]
        sql = (
            "SELECT a.name AS x, b.admin AS y FROM (SELECT name, id FROM users) a "
            "JOIN users b ON b.id = a.id WHERE b.id = 1 LIMIT 5"
        )

        redacted, report = redact_rows(rows, columns, sql=sql)

        assert redacted == rows
        assert report["columns_redacted"] == []
        assert report["values_masked"] == 0

    def test_provenance_redaction_qualified_derived_join_non_pii_end_to_end_via_gateway(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        """Same boundary-pin case as above, through the real gateway/executor."""
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT a.name AS x, b.admin AS y FROM (SELECT name, id FROM users) a "
            "JOIN users b ON b.id = a.id WHERE b.id = 1 LIMIT 5",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["redaction_report"]["columns_redacted"] == []
        assert result["redaction_report"]["values_masked"] == 0

    # -- Derived-star-JOIN HIGH leak: a top-level `SELECT *` spanning BOTH a derived
    # source (a subquery-in-FROM renaming a PII column to a non-signal-carrying alias)
    # AND a base table of unknown schema (`orders`) can't be star-expanded by
    # `qualify()` at all (the base table's schema is genuinely unknown), which pre-fix
    # made `_provenance_pii_columns` bail out completely and leak the derived column's
    # raw value. `_derived_pii_output_columns` closes this without over-redacting the
    # co-present base table's own (non-PII) columns. Reproduced in the exact shapes the
    # security review confirmed: inner/outer JOIN order, LEFT JOIN, comma-join, and a
    # 3-way JOIN. --------------------------------------------------------------------

    def test_provenance_redaction_star_join_derived_then_base_numeric_ssn(self) -> None:
        """The exact leak query from the security review: `SELECT *` over a derived
        `(SELECT ssn_num AS c ...) t` JOINed to base table `orders`. `c` must be
        redacted; `orders`'s own `id`/`user_id`/`total`/`note` must not."""
        columns = ["c", "id", "user_id", "total", "note"]
        rows = [{"c": 123456789, "id": 10, "user_id": 1, "total": 500, "note": "ok"}]
        sql = (
            "SELECT * FROM (SELECT ssn_num AS c FROM users WHERE id=1) t "
            "JOIN orders o ON o.user_id=1 WHERE o.id=10 LIMIT 1"
        )

        redacted, report = redact_rows(rows, columns, sql=sql)

        assert redacted[0] == {"c": "[REDACTED]", "id": 10, "user_id": 1, "total": 500, "note": "ok"}
        assert report["columns_redacted"] == ["c"]
        assert report["values_masked"] == 1

    def test_provenance_redaction_star_join_derived_then_base_end_to_end_via_gateway(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        """Same leak, through the real gateway/executor against the live SQLite fixture
        — real `Gateway` output: `action=allow`, `c` masked, `orders` columns intact."""
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT * FROM (SELECT ssn_num AS c FROM users WHERE id=1) t "
            "JOIN orders o ON o.user_id=1 WHERE o.id=10 LIMIT 1",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["rows"][0] == {"c": "[REDACTED]", "id": 10, "user_id": 1, "total": 500, "note": "ok"}
        assert result["redaction_report"]["columns_redacted"] == ["c"]
        assert result["redaction_report"]["values_masked"] == 1

    def test_provenance_redaction_star_join_left_join_base_then_derived(self) -> None:
        """Same leak, JOIN order reversed: base table (`orders`) first in FROM, `LEFT
        JOIN` into the derived source second."""
        columns = ["id", "user_id", "total", "note", "c"]
        rows = [{"id": 10, "user_id": 1, "total": 500, "note": "ok", "c": 123456789}]
        sql = (
            "SELECT * FROM orders o LEFT JOIN (SELECT ssn_num AS c FROM users WHERE id=1) t "
            "ON o.user_id=1 WHERE o.id=10 LIMIT 1"
        )

        redacted, report = redact_rows(rows, columns, sql=sql)

        assert redacted[0] == {"id": 10, "user_id": 1, "total": 500, "note": "ok", "c": "[REDACTED]"}
        assert report["columns_redacted"] == ["c"]
        assert report["values_masked"] == 1

    def test_provenance_redaction_star_join_left_join_end_to_end_via_gateway(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT * FROM orders o LEFT JOIN (SELECT ssn_num AS c FROM users WHERE id=1) t "
            "ON o.user_id=1 WHERE o.id=10 LIMIT 1",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["rows"][0] == {"id": 10, "user_id": 1, "total": 500, "note": "ok", "c": "[REDACTED]"}
        assert result["redaction_report"]["columns_redacted"] == ["c"]
        assert result["redaction_report"]["values_masked"] == 1

    def test_provenance_redaction_star_join_comma_join(self) -> None:
        """Same leak via an implicit comma-join (`FROM (derived) t, orders o`) instead of
        an explicit `JOIN` keyword."""
        columns = ["c", "id", "user_id", "total", "note"]
        rows = [{"c": 123456789, "id": 10, "user_id": 1, "total": 500, "note": "ok"}]
        sql = (
            "SELECT * FROM (SELECT ssn_num AS c FROM users WHERE id=1) t, orders o "
            "WHERE o.user_id=1 AND o.id=10 LIMIT 1"
        )

        redacted, report = redact_rows(rows, columns, sql=sql)

        assert redacted[0] == {"c": "[REDACTED]", "id": 10, "user_id": 1, "total": 500, "note": "ok"}
        assert report["columns_redacted"] == ["c"]
        assert report["values_masked"] == 1

    def test_provenance_redaction_star_join_comma_join_end_to_end_via_gateway(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT * FROM (SELECT ssn_num AS c FROM users WHERE id=1) t, orders o "
            "WHERE o.user_id=1 AND o.id=10 LIMIT 1",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["rows"][0] == {"c": "[REDACTED]", "id": 10, "user_id": 1, "total": 500, "note": "ok"}
        assert result["redaction_report"]["columns_redacted"] == ["c"]
        assert result["redaction_report"]["values_masked"] == 1

    def test_provenance_redaction_star_join_three_way(self) -> None:
        """Same leak with a third JOIN source (`users u`) added — the star now spans a
        derived source, `orders`, AND a second base-table `users` reference; `c`
        (derived, renamed PII) and `u`'s own PII columns (real names: `email`, `ssn`,
        `ssn_num`) must be redacted, `orders`'s columns must not."""
        columns = ["c", "id", "user_id", "total", "note", "name", "email", "ssn", "ssn_num", "admin"]
        rows = [
            {
                "c": 123456789,
                "id": 10,
                "user_id": 1,
                "total": 500,
                "note": "ok",
                "name": "Alice Smith",
                "email": "alice@example.com",
                "ssn": "123-45-6789",
                "ssn_num": 123456789,
                "admin": 0,
            }
        ]
        sql = (
            "SELECT * FROM (SELECT ssn_num AS c FROM users WHERE id=1) t "
            "JOIN orders o ON o.user_id=1 JOIN users u ON u.id=1 WHERE o.id=10 LIMIT 1"
        )

        redacted, report = redact_rows(rows, columns, sql=sql)

        assert redacted[0]["c"] == "[REDACTED]"
        assert redacted[0]["id"] == 10
        assert redacted[0]["user_id"] == 1
        assert redacted[0]["total"] == 500
        assert redacted[0]["note"] == "ok"
        assert redacted[0]["name"] == "Alice Smith"
        assert redacted[0]["admin"] == 0
        assert redacted[0]["email"] == "[REDACTED]"
        assert redacted[0]["ssn"] == "[REDACTED]"
        assert redacted[0]["ssn_num"] == "[REDACTED]"
        assert set(report["columns_redacted"]) == {"c", "email", "ssn", "ssn_num"}
        assert report["values_masked"] == 4

    def test_provenance_redaction_star_join_three_way_end_to_end_via_gateway(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        """Same 3-way-JOIN leak, through the real gateway/executor. Real `Gateway`
        output confirmed: `action=allow`, `rows=[{'c': '[REDACTED]', 'id': 10,
        'user_id': 1, 'total': 500, 'note': 'ok', 'name': 'Alice Smith', 'email':
        '[REDACTED]', 'ssn': '[REDACTED]', 'ssn_num': '[REDACTED]', 'admin': 0}]`,
        `values_masked=4` — `orders`'s columns are not over-redacted."""
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT * FROM (SELECT ssn_num AS c FROM users WHERE id=1) t "
            "JOIN orders o ON o.user_id=1 JOIN users u ON u.id=1 WHERE o.id=10 LIMIT 1",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        row = result["rows"][0]
        assert row["c"] == "[REDACTED]"
        assert row["id"] == 10
        assert row["user_id"] == 1
        assert row["total"] == 500
        assert row["note"] == "ok"
        assert row["name"] == "Alice Smith"
        assert row["admin"] == 0
        assert row["email"] == "[REDACTED]"
        assert row["ssn"] == "[REDACTED]"
        assert row["ssn_num"] == "[REDACTED]"
        assert set(result["redaction_report"]["columns_redacted"]) == {"c", "email", "ssn", "ssn_num"}
        assert result["redaction_report"]["values_masked"] == 4

    # -- Scalar-subquery-beside-star HIGH leak: a top-level `SELECT *` with a
    # *non-star* projection sitting directly beside the star in the same SELECT list
    # (a scalar subquery, or an arithmetic expression wrapping one) is invisible to
    # `_derived_pii_output_columns`, which only walks FROM/JOIN/CTE derived sources —
    # a scalar subquery used directly as a projection is none of those. Real `Gateway`
    # output confirmed the leak pre-fix: `action=allow`, `columns_redacted=[]`,
    # `rows=[{...,'c':111223333}]`. -----------------------------------------------

    def test_provenance_redaction_scalar_subquery_beside_star_numeric_ssn(self) -> None:
        """The exact leak query from the security review: `SELECT *, (SELECT ssn_num
        FROM users WHERE id=1) AS c FROM orders ...`. `c` must be redacted; `orders`'s
        own columns must not."""
        columns = ["id", "user_id", "total", "note", "c"]
        rows = [{"id": 10, "user_id": 1, "total": 500, "note": "ok", "c": 123456789}]
        sql = "SELECT *, (SELECT ssn_num FROM users WHERE id=1) AS c FROM orders WHERE id>0 LIMIT 5"

        redacted, report = redact_rows(rows, columns, sql=sql)

        expected = {"id": 10, "user_id": 1, "total": 500, "note": "ok", "c": "[REDACTED]"}
        assert redacted[0] == expected
        assert report["columns_redacted"] == ["c"]
        assert report["values_masked"] == 1

    def test_provenance_redaction_scalar_subquery_beside_star_end_to_end_via_gateway(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        """Same leak, through the real gateway/executor against the live SQLite
        fixture — real `Gateway` output: `action=allow`, `columns_redacted=['c']`,
        `values_masked=3`, `c` masked in every row, `orders`'s own columns intact."""
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT *, (SELECT ssn_num FROM users WHERE id=1) AS c FROM orders WHERE id>0 LIMIT 5",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["redaction_report"]["columns_redacted"] == ["c"]
        assert result["redaction_report"]["values_masked"] == len(result["rows"])
        for row in result["rows"]:
            assert row["c"] == "[REDACTED]"
            assert set(row) == {"id", "user_id", "total", "note", "c"}

    def test_provenance_redaction_scalar_subquery_beside_qualified_star_end_to_end_via_gateway(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        """Same leak, `o.*` variant (a qualified star instead of a bare `*`) — must
        redact identically."""
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT o.*, (SELECT ssn_num FROM users WHERE id=1) AS c FROM orders o WHERE id>0 LIMIT 5",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["redaction_report"]["columns_redacted"] == ["c"]
        assert result["redaction_report"]["values_masked"] == len(result["rows"])
        for row in result["rows"]:
            assert row["c"] == "[REDACTED]"

    def test_provenance_redaction_scalar_subquery_beside_star_arithmetic_end_to_end_via_gateway(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        """Same leak wrapped in an arithmetic expression (`(SELECT ...)+0 AS c`) —
        closing the whole non-star-projection-beside-a-star *class*, not just a bare
        scalar subquery, must also redact identically."""
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT *, (SELECT ssn_num FROM users WHERE id=1)+0 AS c FROM orders WHERE id>0 LIMIT 5",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["redaction_report"]["columns_redacted"] == ["c"]
        assert result["redaction_report"]["values_masked"] == len(result["rows"])
        for row in result["rows"]:
            assert row["c"] == "[REDACTED]"

    # -- Generalized fail-closed fix (6th round): an ambiguous, UNQUALIFIED column
    # reference beside a top-level star, among a derived-source+base-table JOIN, whose
    # own alias carries no PII signal at all. Distinct from the fourth leak above (a
    # scalar subquery beside a star): here the projection beside the star is a bare
    # column reference (`s`), ambiguous between the derived source `a` (which actually
    # projects `s`, tracing to `ssn_num`) and the base table `orders` (which doesn't).
    # `_column_is_pii`'s deliberately permissive ambiguous-column fallback name-
    # classified the alias (`renamed`) directly and missed it. Rather than adding a
    # sixth shape-specific patch, this is closed generally: `_query_references_pii_source`
    # + `strict` in `_column_is_pii` fail closed on ANY unresolved/ambiguous top-level
    # output column whenever the query references a PII column anywhere. Real `Gateway`
    # output confirmed the leak pre-fix: `action=allow`, `columns_redacted=[]`,
    # `rows=[{...,'renamed':123456789}]` (a raw SSN). ------------------------------

    def test_provenance_redaction_ambiguous_column_beside_star_numeric_ssn(self) -> None:
        """The exact leak query from the security review: `s` (ambiguous between a
        derived source that projects it and a base table that doesn't) aliased to
        `renamed`, beside a top-level `*`."""
        columns = ["s", "id", "user_id", "total", "note", "renamed"]
        rows = [{"s": 123456789, "id": 10, "user_id": 1, "total": 500, "note": "ok", "renamed": 123456789}]
        sql = (
            "SELECT *, s AS renamed FROM (SELECT ssn_num AS s FROM users) a "
            "JOIN orders o ON o.user_id=1 WHERE o.id>0 LIMIT 5"
        )

        redacted, report = redact_rows(rows, columns, sql=sql)

        assert redacted[0]["renamed"] == "[REDACTED]"
        assert redacted[0]["id"] == 10
        assert redacted[0]["user_id"] == 1
        assert redacted[0]["total"] == 500
        assert redacted[0]["note"] == "ok"
        assert "renamed" in report["columns_redacted"]

    def test_provenance_redaction_ambiguous_column_beside_star_end_to_end_via_gateway(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        """Same leak, through the real gateway/executor against the live SQLite
        fixture — this is the exact query the security review reported as `allow` with
        `renamed=123456789` leaked. `orders`'s own columns must stay intact."""
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT *, s AS renamed FROM (SELECT ssn_num AS s FROM users) a "
            "JOIN orders o ON o.user_id=1 WHERE o.id>0 LIMIT 5",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        row = result["rows"][0]
        assert row["renamed"] == "[REDACTED]"
        assert row["id"] == 10
        assert row["user_id"] == 1
        assert row["total"] == 500
        assert row["note"] == "ok"
        assert "renamed" in result["redaction_report"]["columns_redacted"]

    def test_provenance_redaction_ambiguous_column_beside_star_left_join_variant_end_to_end_via_gateway(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        """Same leak, JOIN order reversed: base table (`orders`) first in `FROM`,
        `LEFT JOIN` into the derived source second — must redact identically."""
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT *, s AS renamed FROM orders o LEFT JOIN "
            "(SELECT ssn_num AS s FROM users) a ON o.user_id=1 WHERE o.id>0 LIMIT 5",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["rows"][0]["renamed"] == "[REDACTED]"
        assert "renamed" in result["redaction_report"]["columns_redacted"]

    def test_provenance_redaction_ambiguous_column_beside_star_comma_join_variant_end_to_end_via_gateway(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        """Same leak via an implicit comma-join (`FROM (derived) a, orders o`) instead
        of an explicit `JOIN` keyword — must redact identically."""
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT *, s AS renamed FROM (SELECT ssn_num AS s FROM users) a, orders o "
            "WHERE o.user_id=1 AND o.id>0 LIMIT 5",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["rows"][0]["renamed"] == "[REDACTED]"
        assert "renamed" in result["redaction_report"]["columns_redacted"]

    # -- Schema-aware view inlining: the last allow-path leak class (bare-numeric PII
    # renamed through a view) — `redaction._inline_views`/`_resolve_view_select`, fed a
    # live `SchemaCatalog` (see `schema_catalog.py` and `gateway._introspect_schema`) --

    def test_provenance_redaction_view_renamed_numeric_ssn(self) -> None:
        """THE confirmed leak from the security review: `CREATE VIEW vnum AS SELECT id
        AS uid, ssn_num AS token FROM users; SELECT token FROM vnum WHERE uid=1` — a
        bare-numeric PII column (`ssn_num`) renamed through a view to an alias with no
        PII-looking name (`token`) at all. Pre-fix, this traced nowhere: no PII
        `exp.Column` appears anywhere in the QUERY's own text (only inside the view's
        unparsed definition), so `_query_references_pii_source` was `False` and a bare
        9-digit int has no dashes/keyword-prefix for the value-pattern layer to catch.
        Supplying a live `schema` (with `vnum`'s real body) must redact it."""
        columns = ["token"]
        rows = [{"token": 987654321}]
        sql = "SELECT token FROM vnum WHERE uid=1 LIMIT 5"

        redacted, report = redact_rows(rows, columns, sql=sql, schema=_USERS_ORDERS_SCHEMA)

        assert redacted[0]["token"] == "[REDACTED]"
        assert "token" in report["columns_redacted"]

    def test_provenance_redaction_view_renamed_numeric_ssn_no_schema_still_leaks(self) -> None:
        """Pin for the fix's exact delta: WITHOUT a schema (the default, and what every
        pre-existing test/caller in this module uses), the identical query is still the
        confirmed pre-fix leak — `redact_rows` has no way to know `vnum` is a view at
        all. This documents that the `schema` parameter is genuinely what closes the
        gap, not an incidental change to schema-less behavior, which must stay exactly
        as permissive as before for a caller with no live connection to introspect."""
        columns = ["token"]
        rows = [{"token": 987654321}]
        sql = "SELECT token FROM vnum WHERE uid=1 LIMIT 5"

        redacted, report = redact_rows(rows, columns, sql=sql)

        assert redacted == rows
        assert report["columns_redacted"] == []

    def test_provenance_redaction_view_star_renamed_numeric_ssn(self) -> None:
        """Same leak via `SELECT * FROM vnum` — the star must expand to the view's own
        real output columns (`uid`, `token`) and only `token` (tracing to `ssn_num`)
        gets redacted; `uid` (tracing to `id`, not PII) stays intact."""
        columns = ["uid", "token"]
        rows = [{"uid": 2, "token": 987654321}]
        sql = "SELECT * FROM vnum WHERE uid=2 LIMIT 5"

        redacted, report = redact_rows(rows, columns, sql=sql, schema=_USERS_ORDERS_SCHEMA)

        assert redacted[0] == {"uid": 2, "token": "[REDACTED]"}
        assert report["columns_redacted"] == ["token"]

    def test_provenance_redaction_view_on_view_numeric_ssn(self) -> None:
        """View-on-view: `vnum2` is built on `vnum`, renaming `token` AGAIN (to
        `secret`) — `_resolve_view_select`'s recursion must trace through BOTH view
        layers down to `ssn_num`."""
        columns = ["secret"]
        rows = [{"secret": 987654321}]
        sql = "SELECT secret FROM vnum2 WHERE id2=2 LIMIT 5"

        redacted, report = redact_rows(rows, columns, sql=sql, schema=_USERS_ORDERS_SCHEMA)

        assert redacted[0]["secret"] == "[REDACTED]"
        assert "secret" in report["columns_redacted"]

    def test_provenance_redaction_view_on_view_star(self) -> None:
        columns = ["id2", "secret"]
        rows = [{"id2": 1, "secret": 123456789}]
        sql = "SELECT * FROM vnum2 WHERE id2=1 LIMIT 5"

        redacted, report = redact_rows(rows, columns, sql=sql, schema=_USERS_ORDERS_SCHEMA)

        assert redacted[0] == {"id2": 1, "secret": "[REDACTED]"}
        assert report["columns_redacted"] == ["secret"]

    def test_provenance_redaction_view_unresolvable_fails_closed_no_crash(self) -> None:
        """A view name the catalog itself flags as unresolvable (`unresolved_views` —
        its stored SQL didn't parse to a single `SELECT`, e.g. a `UNION`-shaped view or
        a dialect quirk this module's `sqlglot`-based extractor couldn't handle) must
        never crash `redact_rows`, and must fail the query's ENTIRE result closed
        rather than silently pass any of it through — see `_inline_views`'s docstring
        for why a query-wide, not a per-column, guarantee is the honest one here."""
        columns = ["a", "b"]
        rows = [{"a": 1, "b": "hello"}]
        sql = "SELECT a, b FROM v_broken WHERE a=1 LIMIT 5"
        schema = SchemaCatalog(tables={}, views={}, unresolved_views=frozenset({"v_broken"}))

        redacted, report = redact_rows(rows, columns, sql=sql, schema=schema)

        assert redacted[0]["a"] == "[REDACTED]"
        assert redacted[0]["b"] == "[REDACTED]"
        assert set(report["columns_redacted"]) == {"a", "b"}

    def test_provenance_redaction_view_cycle_fails_closed_no_crash(self) -> None:
        """A malformed, should-never-happen-for-real-executing-SQL cyclical view
        definition (`va` reads from `vb`, `vb` reads from `va`) must not infinite-loop
        or crash `_resolve_view_select`'s recursion — its `chain` cycle guard must trip
        and fail the whole query closed, exactly like an unparseable view body."""
        columns = ["x"]
        rows = [{"x": 1}]
        sql = "SELECT x FROM va LIMIT 5"
        schema = SchemaCatalog(
            tables={}, views={"va": "SELECT x FROM vb", "vb": "SELECT x FROM va"}
        )

        redacted, report = redact_rows(rows, columns, sql=sql, schema=schema)

        assert redacted[0]["x"] == "[REDACTED]"
        assert "x" in report["columns_redacted"]

    def test_provenance_redaction_view_nested_unresolvable_view_fails_closed_no_crash(self) -> None:
        """Adversarial-retest regression: a view that is itself perfectly resolvable
        (`v_top`) but whose OWN body references a DIFFERENT view this module could not
        resolve (`v_bottom`, e.g. `UNION`-shaped — present only in `unresolved_views`,
        never in `views`) must fail the whole query closed too, even though the
        unresolvable name never appears in the top-level query text at all — only
        inside `v_top`'s own definition. `_resolve_view_select`'s nested `_replace`
        must check `unresolved_view_names` for every Table reference it encounters
        while resolving `v_top`'s body, not only `_inline_views`'s own top-level check
        — before this was wired through, a nested reference to a name absent from BOTH
        `views` and `unresolved_view_names` was silently treated as an ordinary unknown
        base table and the PII behind it leaked raw."""
        columns = ["x"]
        rows = [{"x": 987654321}]
        sql = "SELECT x FROM v_top WHERE x>0 LIMIT 5"
        schema = SchemaCatalog(
            tables={},
            views={"v_top": "SELECT x FROM v_bottom"},
            unresolved_views=frozenset({"v_bottom"}),
        )

        redacted, report = redact_rows(rows, columns, sql=sql, schema=schema)

        assert redacted[0]["x"] == "[REDACTED]"
        assert "x" in report["columns_redacted"]

    def test_provenance_redaction_view_nested_unresolvable_view_end_to_end_via_gateway(
        self, tmp_path, audit_log: AuditLog
    ) -> None:
        """Same nested-unresolvable-view case, through the real gateway against a live
        SQLite database with an ACTUAL `UNION`-shaped `v_bottom` (a real view SQLite
        itself executes fine, but this module's `sqlglot`-based single-`SELECT`
        extractor cannot resolve into a traceable body) referenced only from within
        `v_top`'s own definition."""
        path = str(tmp_path / "nested_unresolvable.sqlite")
        conn = sqlite3.connect(path)
        try:
            conn.execute(SCHEMA_SQL)
            conn.executemany(
                "INSERT INTO users (id, name, email, ssn, ssn_num, admin) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(1, "Alice Smith", "alice@example.com", "123-45-6789", 123456789, 0)],
            )
            conn.execute(ORDERS_SCHEMA_SQL)
            conn.executemany(
                "INSERT INTO orders (id, user_id, total, note) VALUES (?, ?, ?, ?)",
                [(10, 1, 500, "ok")],
            )
            conn.execute(
                "CREATE VIEW v_bottom AS SELECT ssn_num AS x FROM users "
                "UNION SELECT total AS x FROM orders"
            )
            conn.execute("CREATE VIEW v_top AS SELECT x FROM v_bottom")
            conn.commit()
        finally:
            conn.close()

        executor = ReadOnlyExecutor(path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT x FROM v_top WHERE x>0 LIMIT 5",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        for row in result["rows"]:
            assert row["x"] == "[REDACTED]"
        assert "x" in result["redaction_report"]["columns_redacted"]

    def test_provenance_redaction_view_uppercase_declared_name_redacted(self) -> None:
        """Adversarial-retest regression: a view DECLARED with uppercase characters in
        its own name (`CREATE VIEW VNUM_UPPER AS ...` — as opposed to merely being
        REFERENCED in uppercase, already covered by
        `test_provenance_redaction_view_renamed_numeric_ssn_end_to_end_via_gateway`'s
        case-insensitive matching elsewhere) must still be matched and inlined.
        `SchemaCatalog` stores names exactly as the database reports them (its own
        documented contract), so `_inline_views` must normalize ITS OWN dict's keys to
        lowercase before comparing against a `.lower()`-ed query-side name — comparing
        a lowercase query-side name against un-normalized, mixed-case catalog keys
        silently failed to match at all (treated as an ordinary unknown base table,
        not even flagged unresolved) and the PII leaked raw."""
        columns = ["token"]
        rows = [{"token": 123456789}]
        sql = "SELECT token FROM VNUM_UPPER WHERE uid=1 LIMIT 5"
        schema = SchemaCatalog(
            tables={},
            views={"VNUM_UPPER": "SELECT id AS uid, ssn_num AS token FROM users"},
        )

        redacted, report = redact_rows(rows, columns, sql=sql, schema=schema)

        assert redacted[0]["token"] == "[REDACTED]"
        assert "token" in report["columns_redacted"]

    def test_provenance_redaction_view_uppercase_declared_name_end_to_end_via_gateway(
        self, tmp_path, audit_log: AuditLog
    ) -> None:
        path = str(tmp_path / "uppercase_view.sqlite")
        conn = sqlite3.connect(path)
        try:
            conn.execute(SCHEMA_SQL)
            conn.executemany(
                "INSERT INTO users (id, name, email, ssn, ssn_num, admin) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(1, "Alice Smith", "alice@example.com", "123-45-6789", 123456789, 0)],
            )
            conn.execute("CREATE VIEW VNUM_UPPER AS SELECT id AS uid, ssn_num AS token FROM users")
            conn.commit()
        finally:
            conn.close()

        executor = ReadOnlyExecutor(path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT token FROM VNUM_UPPER WHERE uid=1 LIMIT 5",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["rows"] == [{"token": "[REDACTED]"}]

    def test_provenance_redaction_view_renamed_numeric_ssn_end_to_end_via_gateway(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        """The exact confirmed leak, through the REAL `Gateway`/`ReadOnlyExecutor`
        against a live SQLite database with an actual `CREATE VIEW vnum AS SELECT id AS
        uid, ssn_num AS token FROM users` (`db_path`'s fixture, `VIEWS_SQL`) — not a
        hand-built `SchemaCatalog`. `ReadOnlyExecutor.get_schema_catalog` introspects
        this same file's `sqlite_master`, and `Gateway.handle` wires that into
        `redact_rows` automatically (see `gateway._introspect_schema`)."""
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT token FROM vnum WHERE uid=1 LIMIT 5",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["rows"] == [{"token": "[REDACTED]"}]
        assert result["redaction_report"]["columns_redacted"] == ["token"]
        assert result["redaction_report"]["values_masked"] == 1

    def test_provenance_redaction_view_star_end_to_end_via_gateway(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT * FROM vnum WHERE uid=2 LIMIT 5",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["rows"] == [{"uid": 2, "token": "[REDACTED]"}]

    def test_provenance_redaction_view_on_view_end_to_end_via_gateway(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        """View-on-view, through the real gateway against the live fixture: `vnum2`
        (`CREATE VIEW vnum2 AS SELECT uid AS id2, token AS secret FROM vnum`) renames
        `vnum`'s already-renamed `token` a second time, to `secret`."""
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT secret FROM vnum2 WHERE id2=2 LIMIT 5",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["rows"] == [{"secret": "[REDACTED]"}]

    def test_provenance_redaction_view_on_view_star_end_to_end_via_gateway(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT * FROM vnum2 WHERE id2=1 LIMIT 5",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["rows"] == [{"id2": 1, "secret": "[REDACTED]"}]

    def test_provenance_redaction_view_email_still_redacted_end_to_end_via_gateway(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        """Regression guard: a real email value renamed through a view (`v_email`) was
        already caught by the value-pattern scan even before this fix (see
        redaction.py's module docstring) — the new view-inlining machinery must not
        somehow interfere with that pre-existing, independent layer."""
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT contact FROM v_email WHERE uid=1 LIMIT 5",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["rows"] == [{"contact": "[REDACTED]"}]

    def test_provenance_redaction_view_dashed_ssn_still_redacted_end_to_end_via_gateway(
        self, db_path: str, audit_log: AuditLog
    ) -> None:
        """Same regression guard for a dashed-SSN value renamed through a view
        (`v_ssn_dashed`)."""
        executor = ReadOnlyExecutor(db_path)
        gateway = Gateway()

        result = gateway.handle(
            "SELECT s FROM v_ssn_dashed WHERE uid=1 LIMIT 5",
            actor="agent-1",
            executor=executor,
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert result["rows"] == [{"s": "[REDACTED]"}]


# ---------------------------------------------------------------------------
# P1.14 - audit_append_only: direct UPDATE/DELETE on the audit table rejected
# ---------------------------------------------------------------------------

class TestAuditAppendOnly:
    def test_audit_append_only_update_raises(self, audit_log: AuditLog) -> None:
        audit_log.append(
            actor="agent-1", proposed_sql="SELECT 1", classification="read", action="allow",
            matched_rules=[], rows_returned=0, latency_ms=0, result_hash=audit_log.hash_result([]),
        )

        with pytest.raises(sqlite3.DatabaseError):
            audit_log.conn.execute("UPDATE audit_log SET action = 'block' WHERE seq = 1")

        # Nothing must have actually changed.
        assert audit_log.entries()[0].action == "allow"

    def test_audit_append_only_delete_raises(self, audit_log: AuditLog) -> None:
        audit_log.append(
            actor="agent-1", proposed_sql="SELECT 1", classification="read", action="allow",
            matched_rules=[], rows_returned=0, latency_ms=0, result_hash=audit_log.hash_result([]),
        )

        with pytest.raises(sqlite3.DatabaseError):
            audit_log.conn.execute("DELETE FROM audit_log WHERE seq = 1")

        assert len(audit_log.entries()) == 1, "row must not have been deleted"


# ---------------------------------------------------------------------------
# P1.15 - audit_anchor: external anchor detects a full, self-consistent chain rewrite
# ---------------------------------------------------------------------------

class TestAuditAnchor:
    def test_audit_anchor_written_after_append(self, tmp_path) -> None:
        anchor_path = str(tmp_path / "anchor.json")
        log = AuditLog(str(tmp_path / "log.sqlite"), clock=_fixed_clock(), anchor_path=anchor_path)

        entry = log.append(
            actor="agent-1", proposed_sql="SELECT 1", classification="read", action="allow",
            matched_rules=[], rows_returned=0, latency_ms=0, result_hash=log.hash_result([]),
        )

        with open(anchor_path) as f:
            anchor = json.load(f)
        assert anchor["seq"] == entry.seq
        assert anchor["entry_hash"] == entry.entry_hash

    def test_audit_anchor_detects_full_chain_rewrite(self, tmp_path) -> None:
        """CROWN-adjacent: rewrite the ENTIRE chain to be internally self-consistent
        (recompute every entry_hash/prev_hash correctly from a tampered field onward,
        exactly as an attacker with full table access could) without touching the
        external anchor file. A from-scratch recompute of the chain alone would NOT
        catch this — every link is genuinely consistent. `verify()` must still return
        ok=False because the recomputed head no longer matches the untouched anchor.
        """
        log = AuditLog(str(tmp_path / "log2.sqlite"), clock=_fixed_clock())
        for i in range(3):
            log.append(
                actor="agent-1", proposed_sql=f"SELECT {i}", classification="read", action="allow",
                matched_rules=[], rows_returned=1, latency_ms=1, result_hash=log.hash_result([{"n": i}]),
            )

        ok, _ = log.verify()
        assert ok is True, "sanity: untampered chain must verify before we tamper with it"

        # Attacker: mutate entry #2's action, then recompute prev_hash/entry_hash for
        # every entry from #2 onward so the chain is internally self-consistent again.
        log.conn.execute("DROP TRIGGER audit_log_no_update")
        log.conn.execute("UPDATE audit_log SET action = 'block' WHERE seq = 2")
        log.conn.commit()

        expected_prev = GENESIS_HASH
        for entry in log.entries():
            fields = {
                "seq": entry.seq,
                "timestamp": entry.timestamp,
                "actor": entry.actor,
                "proposed_sql": entry.proposed_sql,
                "classification": entry.classification,
                "action": entry.action,
                "matched_rules": entry.matched_rules,
                "rows_returned": entry.rows_returned,
                "latency_ms": entry.latency_ms,
                "result_hash": entry.result_hash,
                "prev_hash": expected_prev,
            }
            new_hash = _compute_entry_hash(fields)
            log.conn.execute(
                "UPDATE audit_log SET prev_hash = ?, entry_hash = ? WHERE seq = ?",
                (expected_prev, new_hash, entry.seq),
            )
            expected_prev = new_hash
        log.conn.commit()

        # Confirm the rewrite really is internally self-consistent by re-deriving the
        # per-entry check verify() itself does, independent of verify()'s own anchor step.
        relink_ok = True
        prev = GENESIS_HASH
        for entry in log.entries():
            if entry.prev_hash != prev or _compute_entry_hash({
                "seq": entry.seq, "timestamp": entry.timestamp, "actor": entry.actor,
                "proposed_sql": entry.proposed_sql, "classification": entry.classification,
                "action": entry.action, "matched_rules": entry.matched_rules,
                "rows_returned": entry.rows_returned, "latency_ms": entry.latency_ms,
                "result_hash": entry.result_hash, "prev_hash": entry.prev_hash,
            }) != entry.entry_hash:
                relink_ok = False
            prev = entry.entry_hash
        assert relink_ok, "the rewritten chain must be internally self-consistent for this test to be meaningful"

        ok, first_broken = log.verify()
        assert ok is False, "external anchor must still catch a full, self-consistent chain rewrite"
        assert first_broken == 3


# ---------------------------------------------------------------------------
# P1.16 - pii_not_logged: no raw PII persisted in the audit log or echoed in errors
# ---------------------------------------------------------------------------

class _StubExecutorWithPiiError:
    """Minimal executor stub that returns a synthetic engine error echoing a raw email,
    so the gateway's reason-scrubbing path can be tested deterministically — independent
    of whether a real SQLite error message happens to echo literal query text."""

    # `Gateway.handle`'s P2.19 read-path guard (`.devdocs/PHASE2_GATES.md`) requires an
    # explicit `IS_READONLY = True` marker on any executor passed to it — this stub
    # stands in for the read path here, so it declares the same marker a real
    # `ReadOnlyExecutor` does.
    IS_READONLY = True

    def execute(self, sql: str, params: tuple = ()) -> ExecutionResult:
        return ExecutionResult(error="constraint failed: duplicate email alice@example.com already exists")


class TestPiiNotLogged:
    def test_pii_not_logged_proposed_sql_scrubbed_in_entry(self, audit_log: AuditLog) -> None:
        entry = audit_log.append(
            actor="agent-1",
            proposed_sql="SELECT id FROM users WHERE email = 'alice@example.com' LIMIT 1",
            classification="read", action="allow", matched_rules=[], rows_returned=0,
            latency_ms=0, result_hash=audit_log.hash_result([]),
        )

        assert "alice@example.com" not in entry.proposed_sql
        assert "EMAIL_REDACTED" in entry.proposed_sql

    def test_pii_not_logged_proposed_sql_scrubbed_at_rest(self, audit_log: AuditLog) -> None:
        """Not just the in-memory dataclass — what's actually persisted in the table."""
        entry = audit_log.append(
            actor="agent-1",
            proposed_sql="SELECT id FROM users WHERE ssn = '123-45-6789' LIMIT 1",
            classification="read", action="allow", matched_rules=[], rows_returned=0,
            latency_ms=0, result_hash=audit_log.hash_result([]),
        )

        stored = audit_log.conn.execute(
            "SELECT proposed_sql FROM audit_log WHERE seq = ?", (entry.seq,)
        ).fetchone()[0]
        assert "123-45-6789" not in stored

    def test_pii_not_logged_proposed_sql_numeric_ssn_literal_scrubbed(self, audit_log: AuditLog) -> None:
        """MEDIUM fix: a dashless numeric SSN literal compared against an SSN-named
        column (`WHERE ssn=123456789`) must not persist raw either — same requirement
        as the dashed case above, closed via the contextual numeric-PII pattern."""
        entry = audit_log.append(
            actor="agent-1",
            proposed_sql="SELECT id FROM users WHERE ssn=123456789 LIMIT 1",
            classification="read", action="allow", matched_rules=[], rows_returned=0,
            latency_ms=0, result_hash=audit_log.hash_result([]),
        )

        assert "123456789" not in entry.proposed_sql
        assert "ssn=[REDACTED]" in entry.proposed_sql

    def test_pii_not_logged_numeric_order_id_not_over_scrubbed(self, audit_log: AuditLog) -> None:
        """The contextual numeric-PII pattern must not blanket-redact ordinary 9-digit
        ids that have no SSN/card/account keyword immediately in front of them — that
        would break routine order-id/reference-number queries."""
        entry = audit_log.append(
            actor="agent-1",
            proposed_sql="SELECT id FROM orders WHERE order_id=123456789 LIMIT 1",
            classification="read", action="allow", matched_rules=[], rows_returned=0,
            latency_ms=0, result_hash=audit_log.hash_result([]),
        )

        assert "123456789" in entry.proposed_sql

    def test_pii_not_logged_gateway_reason_scrubbed_on_engine_error(self, audit_log: AuditLog) -> None:
        gateway = Gateway()

        result = gateway.handle(
            "SELECT id FROM users WHERE id = 1 LIMIT 1",
            actor="agent-1",
            executor=_StubExecutorWithPiiError(),
            audit_log=audit_log,
        )

        assert result["action"] == "allow", result
        assert "alice@example.com" not in result["reason"]
        assert "EMAIL_REDACTED" in result["reason"]

    def test_pii_not_logged_result_hash_is_salted(self, tmp_path) -> None:
        log1 = AuditLog(str(tmp_path / "a.sqlite"), clock=_fixed_clock())
        log2 = AuditLog(str(tmp_path / "b.sqlite"), clock=_fixed_clock())
        rows = [{"id": 1, "admin": 1}]

        salted1 = log1.hash_result(rows)
        salted2 = log2.hash_result(rows)
        unsalted = hash_result_rows(rows)

        assert log1.salt and log2.salt
        assert log1.salt != log2.salt, "each log must get its own random per-instance salt"
        assert salted1 != salted2, "the same content must hash differently under different salts"
        assert salted1 != unsalted, "the salted hash must not equal the raw (unsalted) preimage"


# ---------------------------------------------------------------------------
# P1.17 - json_blob: JSON/BLOB result columns default to redacted
# ---------------------------------------------------------------------------

class TestJsonBlob:
    def test_json_blob_bytes_column_redacted(self) -> None:
        columns = ["id", "payload"]
        rows = [{"id": 1, "payload": b"\x00\x01raw-bytes-that-might-carry-pii"}]

        redacted, report = redact_rows(rows, columns)

        assert redacted[0]["payload"] == "[REDACTED]"
        assert "payload" in report["columns_redacted"]

    def test_json_blob_json_object_string_redacted(self) -> None:
        columns = ["id", "profile"]
        rows = [{"id": 1, "profile": '{"email": "alice@example.com", "age": 30}'}]

        redacted, report = redact_rows(rows, columns)

        assert redacted[0]["profile"] == "[REDACTED]"
        assert "profile" in report["columns_redacted"]

    def test_json_blob_json_array_string_redacted(self) -> None:
        columns = ["id", "tags"]
        rows = [{"id": 1, "tags": '["engineering", "alice@example.com"]'}]

        redacted, report = redact_rows(rows, columns)

        assert redacted[0]["tags"] == "[REDACTED]"
        assert "tags" in report["columns_redacted"]

    def test_json_blob_plain_string_not_over_redacted(self) -> None:
        """A plain string that merely starts with '{' but isn't valid JSON must not be
        swept up by this layer — falls through to the ordinary value-pattern scan/pass-
        through instead, same as P1.5's no-over-redaction contract."""
        columns = ["id", "note"]
        rows = [{"id": 1, "note": "{not really json, just a note"}]

        redacted, report = redact_rows(rows, columns)

        assert redacted[0]["note"] == "{not really json, just a note"
        assert "note" not in report["columns_redacted"]

    def test_json_blob_end_to_end_via_readonly_executor(self, tmp_path) -> None:
        """A real BLOB column round-tripped through SQLite must also come out redacted."""
        path = str(tmp_path / "blobs.sqlite")
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, payload BLOB)")
        conn.execute("INSERT INTO docs (id, payload) VALUES (1, ?)", (b"raw-bytes-here",))
        conn.commit()
        conn.close()

        executor = ReadOnlyExecutor(path)
        result = executor.execute("SELECT id, payload FROM docs WHERE id = 1")
        assert result.error is None, result

        redacted, _report = redact_rows(result.rows, result.columns)
        assert redacted[0]["payload"] == "[REDACTED]"
