"""Phase 2 API tests (`.devdocs/PHASE2_GATES.md`, P2.1-P2.10) — the write-approval
workflow + multi-DB connector factory + governed REST/SSE API, exercised end to end via
FastAPI's `TestClient` against a throwaway SQLite database.

Synthetic schema throughout, matching `tests/security/test_governance_runtime.py`'s own
convention: `users(id, name, email, ssn, ssn_num, admin)` and `orders(id, user_id, total,
note)`. Every row is fabricated for this test file — no real data.

The crown test, `test_approval_approve_executes_crown_updates_row_and_audits`, is the P2.4
gate: it reads the actual SQLite file directly (bypassing the API entirely) both before
and after approval to prove the mutation is real, not a stub, and inspects the real audit
chain for the `approved` transition — the same "ground the crown assertion in the engine
itself, not this module's own bookkeeping" discipline `test_governance_runtime.py` already
follows for its own crown gates.
"""

from __future__ import annotations

import json
import os
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.core.audit import AuditLog
from backend.core.connectors import ConnectorError, connector_for
from backend.core.executor import WriteExecutor
from backend.core.gateway import Gateway
from backend.core.sql_governance import GovernanceGate
from backend.gateway_app import create_gateway_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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

ORDERS_SCHEMA_SQL = """
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    total INTEGER,
    note TEXT
)
"""


@pytest.fixture
def db_path(tmp_path) -> str:
    """A throwaway SQLite file seeded with fabricated `users`/`orders` rows — no real data."""
    path = str(tmp_path / "gateway.sqlite")
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
        conn.execute(ORDERS_SCHEMA_SQL)
        conn.executemany(
            "INSERT INTO orders (id, user_id, total, note) VALUES (?, ?, ?, ?)",
            [(1, 1, 100, "ok"), (2, 2, 200, "pending")],
        )
        conn.commit()
    finally:
        conn.close()
    return path


def _make_app(db_path: str, **kwargs) -> FastAPI:
    return create_gateway_app(db_path, **kwargs)


@pytest.fixture
def app(db_path: str) -> FastAPI:
    """A gateway app with a generous rate limit — most tests aren't testing rate
    limiting and shouldn't be tripped up by it."""
    return _make_app(db_path, rate_limit_per_minute=1000)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _query_users(db_path: str, where: str = "") -> list[sqlite3.Row]:
    """Read `users` directly from the SQLite file — bypasses the API/gateway entirely,
    used to independently verify what actually happened in the database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(f"SELECT * FROM users{where}").fetchall()
    finally:
        conn.close()


def _sse_events(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


# ---------------------------------------------------------------------------
# P2.1 - query_safe_select
# ---------------------------------------------------------------------------

class TestQuerySafeSelect:
    def test_query_safe_select_allows_and_redacts_pii(self, client: TestClient):
        resp = client.post("/query", json={"sql": "SELECT * FROM users WHERE id=1 LIMIT 5", "actor": "agent-1"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["action"] == "allow"
        assert body["approval_id"] is None
        assert body["rows"] is not None and len(body["rows"]) == 1
        row = body["rows"][0]
        # Non-PII columns pass through unredacted.
        assert row["id"] == 1
        assert row["name"] == "Alice Smith"
        assert row["admin"] == 0
        # PII columns are masked outright, regardless of format (dashed string SSN AND
        # dashless numeric ssn_num alike).
        assert row["email"] == "[REDACTED]"
        assert row["ssn"] == "[REDACTED]"
        assert row["ssn_num"] == "[REDACTED]"
        assert body["audit_seq"] >= 1

    def test_query_safe_select_bounded_non_pii_columns_pass_through(self, client: TestClient):
        resp = client.post(
            "/query", json={"sql": "SELECT id, name, admin FROM users WHERE id=2 LIMIT 5", "actor": "agent-1"}
        )
        body = resp.json()
        assert body["action"] == "allow"
        assert body["rows"] == [{"id": 2, "name": "Bob Jones", "admin": 0}]


# ---------------------------------------------------------------------------
# P2.2 - query_block
# ---------------------------------------------------------------------------

class TestQueryBlock:
    def test_query_block_delete_without_where_not_executed(self, client: TestClient, db_path: str):
        before = len(_query_users(db_path))
        resp = client.post("/query", json={"sql": "DELETE FROM users", "actor": "agent-1"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["action"] == "block"
        assert body["rows"] is None
        assert body["approval_id"] is None
        assert "write_without_where" in body["matched_rules"]
        assert len(_query_users(db_path)) == before  # nothing executed

    def test_query_block_injection_tautology_audited(self, client: TestClient):
        resp = client.post(
            "/query", json={"sql": "SELECT * FROM users WHERE id=1 OR 1=1 LIMIT 5", "actor": "agent-1"}
        )
        body = resp.json()
        assert body["action"] == "block"
        assert body["rows"] is None
        assert "tautology_suspected" in body["matched_rules"]
        assert body["audit_seq"] >= 1

    def test_query_block_drop_table_ddl(self, client: TestClient):
        resp = client.post("/query", json={"sql": "DROP TABLE users", "actor": "agent-1"})
        body = resp.json()
        assert body["action"] == "block"
        assert "ddl_blocked" in body["matched_rules"]


# ---------------------------------------------------------------------------
# P2.3 - query_hold_enqueued
# ---------------------------------------------------------------------------

class TestQueryHoldEnqueued:
    def test_query_hold_enqueued_bounded_update_not_executed(self, client: TestClient, db_path: str):
        resp = client.post(
            "/query", json={"sql": "UPDATE users SET admin=1 WHERE id=1", "actor": "agent-1"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["action"] == "hold"
        assert body["rows"] is None
        assert body["approval_id"]
        assert "write_requires_approval" in body["matched_rules"]
        # Not executed yet.
        row = _query_users(db_path, " WHERE id=1")[0]
        assert row["admin"] == 0

    def test_query_hold_enqueued_visible_in_pending_approvals(self, client: TestClient):
        sql = "UPDATE users SET admin=1 WHERE id=2"
        resp = client.post("/query", json={"sql": sql, "actor": "agent-2"})
        approval_id = resp.json()["approval_id"]

        pending = client.get("/approvals").json()
        matching = [p for p in pending if p["id"] == approval_id]
        assert len(matching) == 1
        assert matching[0]["sql"] == sql
        assert matching[0]["actor"] == "agent-2"
        assert matching[0]["status"] == "pending"


# ---------------------------------------------------------------------------
# M4 - GET /approvals echoes raw SQL/actor literals
# ---------------------------------------------------------------------------

class TestApprovalsListRedaction:
    """`GET /approvals` shows a pending write's `sql`/`reason`/`actor` to whoever can
    reach it -- before this fix, a PII literal embedded in the held SQL (or a
    PII-shaped `actor` identifier) reached this endpoint completely raw, even though
    the SAME data is redacted once it lands in `/audit`. Approvers still need the
    query's STRUCTURE to decide whether to approve it; they don't need the literal
    value.
    """

    def test_approvals_list_redacts_email_and_ssn_literals_in_pending_sql(
        self, client: TestClient
    ):
        sql = "UPDATE users SET admin=1 WHERE email='alice@example.com' AND ssn='111-22-3333'"
        resp = client.post("/query", json={"sql": sql, "actor": "agent-1"})
        approval_id = resp.json()["approval_id"]

        pending = client.get("/approvals").json()
        matching = [p for p in pending if p["id"] == approval_id]
        assert len(matching) == 1

        # No raw email/SSN anywhere in this pending record.
        record_text = json.dumps(matching[0])
        assert "alice@example.com" not in record_text
        assert "111-22-3333" not in record_text

        # The query's STRUCTURE is preserved -- an approver can still see this is an
        # UPDATE against `users` filtered on `email`/`ssn`, just not the literal values.
        assert "UPDATE users SET admin=1" in matching[0]["sql"]
        assert "email" in matching[0]["sql"].lower()
        assert "ssn" in matching[0]["sql"].lower()

        # No raw PII anywhere in the FULL response either (covers actor/reason/etc.).
        full_text = json.dumps(pending)
        assert "alice@example.com" not in full_text
        assert "111-22-3333" not in full_text

    def test_approvals_list_redacts_pii_shaped_actor(self, client: TestClient):
        resp = client.post(
            "/query",
            json={"sql": "UPDATE orders SET total=0 WHERE id=1", "actor": "alice@example.com"},
        )
        approval_id = resp.json()["approval_id"]

        pending = client.get("/approvals").json()
        matching = [p for p in pending if p["id"] == approval_id][0]
        assert "alice@example.com" not in matching["actor"]


# ---------------------------------------------------------------------------
# P2.4 - approval_approve_executes (CROWN)
# ---------------------------------------------------------------------------

class TestApprovalApproveExecutes:
    def test_approval_approve_executes_crown_updates_row_and_audits(self, client: TestClient, db_path: str):
        # 1. Propose a bounded write -> held, not executed.
        resp = client.post(
            "/query", json={"sql": "UPDATE users SET admin=1 WHERE id=1", "actor": "agent-1"}
        )
        approval_id = resp.json()["approval_id"]
        assert approval_id

        # 2. While pending: the row is UNCHANGED in the real database.
        before = _query_users(db_path, " WHERE id=1")[0]
        assert before["admin"] == 0

        # 3. Approve.
        decide = client.post(f"/approvals/{approval_id}", json={"decision": "approve", "approver": "root-admin"})
        assert decide.status_code == 200
        decide_body = decide.json()
        assert decide_body["status"] == "approved"
        assert decide_body["approver"] == "root-admin"
        assert decide_body["error"] is None

        # 4. After approval: the row IS now changed in the real database.
        after = _query_users(db_path, " WHERE id=1")[0]
        assert after["admin"] == 1

        # 5. The audit log records an `approved` transition naming the approver.
        audit = client.get("/audit").json()
        assert audit["chain_valid"] is True
        approved_entries = [e for e in audit["entries"] if e["action"] == "approved"]
        assert len(approved_entries) == 1
        assert approved_entries[0]["actor"] == "root-admin"
        assert "UPDATE users SET admin=1 WHERE id=1" in approved_entries[0]["proposed_sql"]
        assert any(r == f"approval_id={approval_id}" for r in approved_entries[0]["matched_rules"])
        assert any(r == "proposed_by=agent-1" for r in approved_entries[0]["matched_rules"])

        # 6. It no longer shows up as pending.
        pending_ids = [p["id"] for p in client.get("/approvals").json()]
        assert approval_id not in pending_ids

    def test_approval_approve_executes_insert(self, client: TestClient, db_path: str):
        resp = client.post(
            "/query",
            json={"sql": "INSERT INTO orders (id, user_id, total, note) VALUES (3, 1, 50, 'new')", "actor": "agent-1"},
        )
        approval_id = resp.json()["approval_id"]
        assert approval_id

        client.post(f"/approvals/{approval_id}", json={"decision": "approve", "approver": "root-admin"})

        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute("SELECT total, note FROM orders WHERE id=3").fetchone()
        finally:
            conn.close()
        assert row == (50, "new")


# ---------------------------------------------------------------------------
# P2.5 - approval_reject
# ---------------------------------------------------------------------------

class TestApprovalReject:
    def test_approval_reject_does_not_execute(self, client: TestClient, db_path: str):
        resp = client.post("/query", json={"sql": "DELETE FROM orders WHERE id=1", "actor": "agent-1"})
        approval_id = resp.json()["approval_id"]
        assert approval_id

        decide = client.post(f"/approvals/{approval_id}", json={"decision": "reject", "approver": "root-admin"})
        assert decide.status_code == 200
        assert decide.json()["status"] == "rejected"

        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute("SELECT id FROM orders WHERE id=1").fetchone()
        finally:
            conn.close()
        assert row is not None  # not deleted

    def test_approval_reject_audits_rejected_with_approver(self, client: TestClient):
        resp = client.post("/query", json={"sql": "UPDATE orders SET total=0 WHERE id=2", "actor": "agent-3"})
        approval_id = resp.json()["approval_id"]

        client.post(f"/approvals/{approval_id}", json={"decision": "reject", "approver": "security-team"})

        audit = client.get("/audit").json()
        rejected = [e for e in audit["entries"] if e["action"] == "rejected"]
        assert len(rejected) == 1
        assert rejected[0]["actor"] == "security-team"


# ---------------------------------------------------------------------------
# P2.6 - approval_failsafe
# ---------------------------------------------------------------------------

class TestApprovalFailsafe:
    def test_approval_failsafe_unknown_id(self, client: TestClient):
        resp = client.post("/approvals/does-not-exist", json={"decision": "approve", "approver": "root-admin"})
        assert 400 <= resp.status_code < 500

    def test_approval_failsafe_expired(self, app: FastAPI, client: TestClient, db_path: str):
        gate = GovernanceGate()
        sql = "UPDATE users SET admin=1 WHERE id=2"
        decision = gate.evaluate(sql, actor="agent-1")
        # Enqueue directly against the app's own queue with a negative TTL so it is
        # already expired the instant it's created -- no need to fake the clock forward.
        approval_id = app.state.approval_queue.enqueue(decision, sql, "agent-1", ttl_seconds=-1)

        resp = client.post(f"/approvals/{approval_id}", json={"decision": "approve", "approver": "root-admin"})
        assert 400 <= resp.status_code < 500

        # No execution happened.
        row = _query_users(db_path, " WHERE id=2")[0]
        assert row["admin"] == 0

    def test_approval_failsafe_already_decided_double_approve(self, client: TestClient, db_path: str):
        resp = client.post("/query", json={"sql": "UPDATE users SET admin=1 WHERE id=2", "actor": "agent-1"})
        approval_id = resp.json()["approval_id"]

        first = client.post(f"/approvals/{approval_id}", json={"decision": "approve", "approver": "root-admin"})
        assert first.status_code == 200
        row_after_first = _query_users(db_path, " WHERE id=2")[0]["admin"]
        assert row_after_first == 1

        # A second approve of the SAME id must fail safe and must NOT execute again.
        second = client.post(f"/approvals/{approval_id}", json={"decision": "approve", "approver": "someone-else"})
        assert 400 <= second.status_code < 500

        row_after_second = _query_users(db_path, " WHERE id=2")[0]["admin"]
        assert row_after_second == 1  # unchanged by the failed second attempt

        audit = client.get("/audit").json()
        approved = [e for e in audit["entries"] if e["action"] == "approved"]
        assert len(approved) == 1  # exactly one execution, not two

    def test_approval_failsafe_reject_after_approve(self, client: TestClient):
        resp = client.post("/query", json={"sql": "UPDATE orders SET total=0 WHERE id=1", "actor": "agent-1"})
        approval_id = resp.json()["approval_id"]

        client.post(f"/approvals/{approval_id}", json={"decision": "approve", "approver": "root-admin"})
        reject_after = client.post(f"/approvals/{approval_id}", json={"decision": "reject", "approver": "root-admin"})
        assert 400 <= reject_after.status_code < 500


# ---------------------------------------------------------------------------
# P2.7 - audit_endpoint
# ---------------------------------------------------------------------------

class TestAuditEndpoint:
    def test_audit_endpoint_returns_verified_chain(self, client: TestClient):
        client.post("/query", json={"sql": "SELECT id FROM users WHERE id=1 LIMIT 1", "actor": "agent-1"})
        client.post("/query", json={"sql": "DROP TABLE users", "actor": "agent-1"})

        resp = client.get("/audit")
        assert resp.status_code == 200
        body = resp.json()
        assert body["chain_valid"] is True
        assert body["broken_at_seq"] is None
        assert len(body["entries"]) >= 2
        seqs = [e["seq"] for e in body["entries"]]
        assert seqs == sorted(seqs)

    def test_audit_endpoint_redacts_pii_in_proposed_sql(self, client: TestClient):
        client.post(
            "/query",
            json={"sql": "SELECT id FROM users WHERE email='alice@example.com' LIMIT 1", "actor": "agent-1"},
        )
        entries = client.get("/audit").json()["entries"]
        assert any("email" in e["proposed_sql"].lower() for e in entries)
        assert not any("alice@example.com" in e["proposed_sql"] for e in entries)

    # -----------------------------------------------------------------------
    # M3 - /audit leaks PII in matched_rules (proposed_by=<raw actor>)
    # -----------------------------------------------------------------------

    def test_audit_endpoint_redacts_pii_in_matched_rules_proposed_by(
        self, client: TestClient
    ):
        """`ApprovalQueue.approve`/`reject` fold `f"proposed_by={record.actor}"` into
        `matched_rules` for traceability. When `actor` is itself PII-shaped (a raw
        email, as an unauthenticated caller could legitimately supply), that value must
        not reach `/audit` raw -- `matched_rules` gets the same `redact_text` scrub as
        `proposed_sql`/`actor` do, not just those two fields.
        """
        proposer = "alice@example.com"
        resp = client.post(
            "/query", json={"sql": "UPDATE orders SET total=0 WHERE id=1", "actor": proposer}
        )
        approval_id = resp.json()["approval_id"]
        client.post(
            f"/approvals/{approval_id}", json={"decision": "approve", "approver": "root-admin"}
        )

        audit = client.get("/audit").json()
        approved = [e for e in audit["entries"] if e["action"] == "approved"]
        assert len(approved) == 1
        rules_text = json.dumps(approved[0]["matched_rules"])
        assert "proposed_by" in rules_text  # the traceability info is still present...
        assert proposer not in rules_text  # ...but never as a raw email

        # No raw email anywhere in the FULL /audit response, not just matched_rules.
        full_text = json.dumps(audit)
        assert proposer not in full_text


# ---------------------------------------------------------------------------
# P2.8 - rate_limit
# ---------------------------------------------------------------------------

class TestRateLimit:
    def test_rate_limit_returns_429_after_cap(self, db_path: str):
        low_limit_app = _make_app(db_path, rate_limit_per_minute=3)
        low_client = TestClient(low_limit_app)

        statuses = []
        for _ in range(5):
            resp = low_client.post(
                "/query", json={"sql": "SELECT id FROM users WHERE id=1 LIMIT 1", "actor": "same-actor"}
            )
            statuses.append(resp.status_code)

        assert statuses[:3] == [200, 200, 200]
        assert 429 in statuses

    def test_rate_limit_is_per_actor(self, db_path: str):
        low_limit_app = _make_app(db_path, rate_limit_per_minute=1)
        low_client = TestClient(low_limit_app)

        r1 = low_client.post("/query", json={"sql": "SELECT id FROM users WHERE id=1 LIMIT 1", "actor": "actor-a"})
        r2 = low_client.post("/query", json={"sql": "SELECT id FROM users WHERE id=1 LIMIT 1", "actor": "actor-b"})
        assert r1.status_code == 200
        assert r2.status_code == 200  # different actor, own bucket -- not starved by actor-a


# ---------------------------------------------------------------------------
# P2.9 - sse_stream
# ---------------------------------------------------------------------------

class TestSSEStream:
    def test_sse_stream_allow_yields_start_decision_rows(self, client: TestClient):
        resp = client.post(
            "/query/stream", json={"sql": "SELECT id FROM users WHERE id=1 LIMIT 1", "actor": "agent-1"}
        )
        assert resp.status_code == 200
        events = _sse_events(resp.text)
        types = [e["type"] for e in events]
        assert types == ["start", "decision", "rows"]
        assert events[1]["action"] == "allow"
        assert events[2]["rows"] == [{"id": 1}]

    def test_sse_stream_hold_yields_start_decision_held(self, client: TestClient):
        resp = client.post(
            "/query/stream", json={"sql": "UPDATE users SET admin=1 WHERE id=1", "actor": "agent-1"}
        )
        events = _sse_events(resp.text)
        types = [e["type"] for e in events]
        assert types == ["start", "decision", "held"]
        assert events[1]["action"] == "hold"
        assert events[2]["approval_id"]

    def test_sse_stream_block_yields_start_decision_blocked(self, client: TestClient):
        resp = client.post("/query/stream", json={"sql": "DROP TABLE users", "actor": "agent-1"})
        events = _sse_events(resp.text)
        assert [e["type"] for e in events] == ["start", "decision", "blocked"]


# ---------------------------------------------------------------------------
# P2.10 - connector
# ---------------------------------------------------------------------------

class TestConnector:
    def test_connector_factory_selects_sqlite_backend(self, db_path: str):
        conn = connector_for(db_path)
        assert conn.backend == "sqlite"
        result = conn.read_executor.execute("SELECT 1 AS one")
        assert result.error is None
        assert result.rows == [{"one": 1}]

    def test_connector_factory_sqlite_uri_form(self, db_path: str):
        conn = connector_for(f"sqlite://{db_path}")
        assert conn.backend == "sqlite"
        result = conn.read_executor.execute("SELECT 1")
        assert result.error is None

    def test_connector_factory_has_uniform_read_write_interface(self, db_path: str):
        conn = connector_for(db_path)
        assert hasattr(conn.read_executor, "execute")
        assert hasattr(conn.write_executor, "execute")
        assert conn.get_schema_catalog() is not None  # real SQLite introspection

    def test_connector_factory_rejects_empty_dsn(self):
        with pytest.raises(ConnectorError):
            connector_for("")

    def test_connector_factory_rejects_unknown_scheme(self):
        with pytest.raises(ConnectorError):
            connector_for("ftp://example.com/x")

    def test_connector_factory_rejects_hostless_postgres_dsn(self):
        with pytest.raises(ConnectorError):
            connector_for("postgres://")

    def test_connector_factory_postgres_constructs_without_live_server(self):
        conn = connector_for("postgres://user:pw@localhost:5432/testdb")
        assert conn.backend == "postgres"
        # No live server: execute() must degrade to an ExecutionResult.error, never raise.
        result = conn.read_executor.execute("SELECT 1")
        assert result.error is not None

    def test_connector_factory_mysql_constructs_without_live_server(self):
        conn = connector_for("mysql://user:pw@localhost:3306/testdb")
        assert conn.backend == "mysql"
        result = conn.read_executor.execute("SELECT 1")
        assert result.error is not None

    @pytest.mark.skipif(
        not os.environ.get("ICBERG_TEST_PG_DSN"),
        reason="set ICBERG_TEST_PG_DSN to run a live Postgres connector test",
    )
    def test_connector_postgres_live_roundtrip(self):
        conn = connector_for(os.environ["ICBERG_TEST_PG_DSN"])
        result = conn.read_executor.execute("SELECT 1")
        assert result.error is None

    @pytest.mark.skipif(
        not os.environ.get("ICBERG_TEST_MYSQL_DSN"),
        reason="set ICBERG_TEST_MYSQL_DSN to run a live MySQL connector test",
    )
    def test_connector_mysql_live_roundtrip(self):
        conn = connector_for(os.environ["ICBERG_TEST_MYSQL_DSN"])
        result = conn.read_executor.execute("SELECT 1")
        assert result.error is None


# ---------------------------------------------------------------------------
# Health / metrics (not gate keywords, but part of the required endpoint surface)
# ---------------------------------------------------------------------------

class TestHealthAndMetrics:
    def test_health_endpoint_reports_healthy(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["components"]["connector"] == "sqlite"

    def test_metrics_endpoint_exposes_counters_after_traffic(self, client: TestClient):
        client.post("/query", json={"sql": "SELECT id FROM users WHERE id=1 LIMIT 1", "actor": "agent-1"})
        client.post("/query", json={"sql": "DROP TABLE users", "actor": "agent-1"})
        client.post("/query", json={"sql": "UPDATE users SET admin=1 WHERE id=1", "actor": "agent-1"})

        resp = client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "icberg_queries_total 3" in text
        assert "icberg_blocks_total 1" in text
        assert "icberg_holds_total 1" in text


# ---------------------------------------------------------------------------
# P2.12 - dsn_server_pinned
# ---------------------------------------------------------------------------

class TestDsnServerPinned:
    """A caller must not be able to choose the connection target via the request
    body — the DSN is fixed once, server-side, at `create_gateway_app(dsn)` time
    (`request.app.state.connection`). `QueryRequest.model_config = ConfigDict(extra=
    "forbid")` means any extra field (a `dsn`, `connection`, `database_url`, or an
    attempted SSRF/file-read payload of any name) is rejected at validation time,
    BEFORE the request handler runs at all — so there is no code path here that could
    even attempt to honor it, let alone open a new connection to it.
    """

    @pytest.mark.parametrize(
        "malicious_field",
        [
            {"dsn": "postgresql://root@169.254.169.254/"},
            {"connection": "file:/etc/passwd"},
            {"database_url": "file:/etc/passwd"},
            {"dsn": "file:/etc/passwd?mode=ro"},
        ],
    )
    def test_dsn_server_pinned_rejects_request_supplied_connection_target(
        self, client: TestClient, db_path: str, malicious_field: dict
    ):
        body = {"sql": "SELECT id FROM users WHERE id=1 LIMIT 1", "actor": "agent-1", **malicious_field}
        resp = client.post("/query", json=body)
        # Rejected outright (422) -- never reaches the route handler, never evaluates
        # or attempts to honor the smuggled connection target at all. FastAPI's stock
        # validation-error body echoes back the submitted field's own value (standard,
        # harmless -- it's just repeating what the client itself already sent); what
        # matters for SSRF/file-read is that nothing here indicates a CONNECTION was
        # ever attempted to it (no engine error, no driver name, no "failed to open"/
        # "connect" text -- those only ever appear from an actual connection attempt).
        assert resp.status_code == 422
        text = resp.text.lower()
        for leak_indicator in ("psycopg", "pymysql", "failed to open", "failed to connect", "connection refused"):
            assert leak_indicator not in text

        # The server's own connection is untouched by the rejected attempt -- prove it
        # by using the API normally right after and getting a real result back from
        # the ONE configured throwaway SQLite database, not an error or a different DB.
        sanity = client.post(
            "/query", json={"sql": "SELECT id FROM users WHERE id=1 LIMIT 1", "actor": "agent-1"}
        )
        assert sanity.status_code == 200
        assert sanity.json()["action"] == "allow"
        assert sanity.json()["rows"] == [{"id": 1}]

    def test_dsn_server_pinned_stream_endpoint_also_rejects(self, client: TestClient):
        resp = client.post(
            "/query/stream",
            json={
                "sql": "SELECT id FROM users WHERE id=1 LIMIT 1",
                "actor": "agent-1",
                "dsn": "postgresql://root@169.254.169.254/",
            },
        )
        assert resp.status_code == 422

    def test_dsn_server_pinned_connector_factory_itself_only_uses_server_config(self, db_path: str):
        """Belt-and-suspenders unit-level proof: `connector_for` (the only place a DSN
        is ever turned into a live connection) takes exactly one argument -- there is
        no request/body object it could pull an alternate DSN from even if a caller
        somehow got one past the API layer above.
        """
        import inspect

        params = list(inspect.signature(connector_for).parameters)
        assert params[0] == "dsn"
        assert "request" not in params and "body" not in params


# ---------------------------------------------------------------------------
# P2.13 - approval_replay_blocked
# ---------------------------------------------------------------------------

class TestApprovalReplayBlocked:
    def test_approval_replay_blocked_second_approve_does_not_execute_again(
        self, client: TestClient, db_path: str
    ):
        resp = client.post(
            "/query", json={"sql": "UPDATE users SET admin=1 WHERE id=1", "actor": "agent-1"}
        )
        approval_id = resp.json()["approval_id"]

        first = client.post(f"/approvals/{approval_id}", json={"decision": "approve", "approver": "root-admin"})
        assert first.status_code == 200
        assert _query_users(db_path, " WHERE id=1")[0]["admin"] == 1

        audit_before = client.get("/audit").json()
        approved_before = [e for e in audit_before["entries"] if e["action"] == "approved"]
        assert len(approved_before) == 1

        # Replay: approve the SAME id a second time.
        second = client.post(
            f"/approvals/{approval_id}", json={"decision": "approve", "approver": "root-admin-2"}
        )
        assert second.status_code == 409  # exact status, not just "some 4xx"

        # The DB was not re-mutated by the replay (nothing to observe differently here
        # since it's already admin=1, but assert explicitly it's still exactly 1, not
        # e.g. incremented or touched again).
        assert _query_users(db_path, " WHERE id=1")[0]["admin"] == 1

        audit_after = client.get("/audit").json()
        approved_after = [e for e in audit_after["entries"] if e["action"] == "approved"]
        assert len(approved_after) == 1  # no second audit entry -- nothing executed twice


# ---------------------------------------------------------------------------
# P2.14 - approval_after_reject_blocked
# ---------------------------------------------------------------------------

class TestApprovalAfterRejectBlocked:
    def test_approval_after_reject_blocked_cannot_later_approve(self, client: TestClient, db_path: str):
        resp = client.post(
            "/query", json={"sql": "UPDATE users SET admin=1 WHERE id=2", "actor": "agent-1"}
        )
        approval_id = resp.json()["approval_id"]

        reject = client.post(f"/approvals/{approval_id}", json={"decision": "reject", "approver": "root-admin"})
        assert reject.status_code == 200
        assert reject.json()["status"] == "rejected"

        approve_after = client.post(
            f"/approvals/{approval_id}", json={"decision": "approve", "approver": "root-admin"}
        )
        assert approve_after.status_code == 409

        # Never executed -- the row is exactly as it was before the (rejected) proposal.
        assert _query_users(db_path, " WHERE id=2")[0]["admin"] == 0

        audit = client.get("/audit").json()
        approved = [e for e in audit["entries"] if e["action"] == "approved"]
        assert approved == []  # a rejected id can NEVER produce an "approved" entry


# ---------------------------------------------------------------------------
# P2.15 - approval_sql_immutable
# ---------------------------------------------------------------------------

class TestApprovalSqlImmutable:
    def test_approval_sql_immutable_toctou_ignores_request_supplied_sql(
        self, client: TestClient, db_path: str
    ):
        original_sql = "UPDATE users SET admin=1 WHERE id=1"
        resp = client.post("/query", json={"sql": original_sql, "actor": "agent-1"})
        approval_id = resp.json()["approval_id"]

        # Confirm the human reviewing GET /approvals sees the real, original SQL.
        pending = client.get("/approvals").json()
        assert [p for p in pending if p["id"] == approval_id][0]["sql"] == original_sql

        # Attempt a TOCTOU substitution: approve, but smuggle a DIFFERENT, destructive
        # SQL string into the request body. `ApprovalDecisionRequest` has no `sql`
        # field at all -- an extra field is silently ignored (default Pydantic
        # behavior), never re-parsed or substituted for the stored statement.
        malicious_decide = {
            "decision": "approve",
            "approver": "root-admin",
            "sql": "DELETE FROM users",
        }
        decide = client.post(f"/approvals/{approval_id}", json=malicious_decide)
        assert decide.status_code == 200

        # The malicious DELETE never ran -- both seeded rows are still present.
        assert len(_query_users(db_path)) == 2
        # The EXACT bytes stored at enqueue time are what executed: the original UPDATE.
        assert _query_users(db_path, " WHERE id=1")[0]["admin"] == 1

        audit = client.get("/audit").json()
        approved = [e for e in audit["entries"] if e["action"] == "approved"]
        assert len(approved) == 1
        assert approved[0]["proposed_sql"] == original_sql
        assert "DELETE FROM users" not in approved[0]["proposed_sql"]

    def test_approval_sql_immutable_mutating_original_dict_after_enqueue_has_no_effect(
        self, client: TestClient, db_path: str
    ):
        """A second flavor of the same TOCTOU property: mutating the Python dict the
        caller used to build the ORIGINAL /query request, after it has already been
        enqueued, has no effect -- the queue stored its own copy of the SQL text at
        `enqueue` time, not a live reference to anything the caller still holds.
        """
        body = {"sql": "UPDATE users SET admin=1 WHERE id=2", "actor": "agent-1"}
        resp = client.post("/query", json=body)
        approval_id = resp.json()["approval_id"]

        # Mutate the caller's own dict after the fact -- as if a buggy/adversarial
        # client tried to "edit" the statement post-submission.
        body["sql"] = "DROP TABLE users"

        decide = client.post(f"/approvals/{approval_id}", json={"decision": "approve", "approver": "root-admin"})
        assert decide.status_code == 200

        # `users` table still exists and still has exactly the seeded rows plus the
        # ORIGINAL UPDATE applied -- proving the mutated dict was never consulted.
        assert len(_query_users(db_path)) == 2
        assert _query_users(db_path, " WHERE id=2")[0]["admin"] == 1


# ---------------------------------------------------------------------------
# P2.16 - self_approval
# ---------------------------------------------------------------------------

class TestSelfApproval:
    def test_self_approval_forbidden_actor_cannot_approve_own_proposal(
        self, client: TestClient, db_path: str
    ):
        resp = client.post(
            "/query", json={"sql": "UPDATE users SET admin=1 WHERE id=1", "actor": "agent-1"}
        )
        approval_id = resp.json()["approval_id"]

        self_approve = client.post(
            f"/approvals/{approval_id}", json={"decision": "approve", "approver": "agent-1"}
        )
        assert self_approve.status_code == 403

        # Not executed.
        assert _query_users(db_path, " WHERE id=1")[0]["admin"] == 0

        audit = client.get("/audit").json()
        approved = [e for e in audit["entries"] if e["action"] == "approved"]
        assert approved == []

        # Still pending -- self-approval leaves it decidable by someone else.
        pending_ids = [p["id"] for p in client.get("/approvals").json()]
        assert approval_id in pending_ids

        # A DIFFERENT approver can legitimately approve the same proposal afterward.
        other = client.post(
            f"/approvals/{approval_id}", json={"decision": "approve", "approver": "root-admin"}
        )
        assert other.status_code == 200
        assert _query_users(db_path, " WHERE id=1")[0]["admin"] == 1

    def test_self_approval_forbidden_also_blocks_self_reject_bypass_attempt(self, client: TestClient):
        """Sanity check that the self-approval guard is specific to `approve` -- a
        proposer rejecting their OWN proposal is not a security-relevant action (it
        executes nothing either way) and must not be blocked by this control.
        """
        resp = client.post("/query", json={"sql": "UPDATE orders SET total=0 WHERE id=1", "actor": "agent-9"})
        approval_id = resp.json()["approval_id"]

        self_reject = client.post(
            f"/approvals/{approval_id}", json={"decision": "reject", "approver": "agent-9"}
        )
        assert self_reject.status_code == 200
        assert self_reject.json()["status"] == "rejected"

    # -----------------------------------------------------------------------
    # M2 - self-approval identity normalization (trailing whitespace / case)
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "approver",
        [
            "agent-1 ",   # trailing space
            " agent-1",   # leading space
            "AGENT-1",    # case variant
            "Agent-1 ",   # both
        ],
    )
    def test_self_approval_forbidden_identity_normalized_trailing_space_and_case(
        self, client: TestClient, db_path: str, approver: str
    ):
        """`approver="agent-1 "` (a trailing space) or `"AGENT-1"` (a case variant) is
        still the SAME claimed identity as `actor="agent-1"` -- a naive `==` comparison
        let either bypass the self-approval guard entirely. Normalizing both sides
        (strip + casefold) before comparing must still reject them with 403.
        """
        resp = client.post(
            "/query", json={"sql": "UPDATE users SET admin=1 WHERE id=1", "actor": "agent-1"}
        )
        approval_id = resp.json()["approval_id"]

        self_approve = client.post(
            f"/approvals/{approval_id}", json={"decision": "approve", "approver": approver}
        )
        assert self_approve.status_code == 403

        # Not executed.
        assert _query_users(db_path, " WHERE id=1")[0]["admin"] == 0
        audit = client.get("/audit").json()
        assert [e for e in audit["entries"] if e["action"] == "approved"] == []

    def test_self_approval_genuinely_different_approver_still_succeeds(
        self, client: TestClient, db_path: str
    ):
        """Sanity check the normalization doesn't over-match: a GENUINELY different
        approver identity (not just whitespace/case noise around the same one) must
        still be accepted (200), not wrongly treated as a self-approval.
        """
        resp = client.post(
            "/query", json={"sql": "UPDATE users SET admin=1 WHERE id=1", "actor": "agent-1"}
        )
        approval_id = resp.json()["approval_id"]

        approve = client.post(
            f"/approvals/{approval_id}", json={"decision": "approve", "approver": "agent-2"}
        )
        assert approve.status_code == 200
        assert _query_users(db_path, " WHERE id=1")[0]["admin"] == 1


# ---------------------------------------------------------------------------
# P2.17 - metrics_no_pii
# ---------------------------------------------------------------------------

class TestMetricsNoPii:
    def test_metrics_no_pii_only_bucketed_counters(self, client: TestClient, db_path: str):
        secret_sql = "SELECT * FROM users WHERE email='alice@example.com' LIMIT 1"
        secret_actor = "agent-super-secret-identity"
        client.post("/query", json={"sql": secret_sql, "actor": secret_actor})
        client.post("/query", json={"sql": "DROP TABLE users", "actor": secret_actor})
        client.post("/query", json={"sql": "UPDATE users SET admin=1 WHERE id=1", "actor": secret_actor})

        low_limit_app = _make_app(db_path, rate_limit_per_minute=1)
        low_client = TestClient(low_limit_app)
        low_client.post("/query", json={"sql": "SELECT 1", "actor": secret_actor})
        low_client.post("/query", json={"sql": "SELECT 1", "actor": secret_actor})  # trips 429

        for target_client in (client, low_client):
            resp = target_client.get("/metrics")
            assert resp.status_code == 200
            text = resp.text

            # No raw SQL text anywhere in the metrics output.
            assert secret_sql not in text
            assert "alice@example.com" not in text
            assert "DROP TABLE" not in text
            assert "UPDATE users" not in text
            assert "SELECT" not in text.upper()

            # No actor value leaked as a metric label or value.
            assert secret_actor not in text

            # Only bucketed, unlabeled counters -- no Prometheus label set (`{...}`,
            # which is how per-actor/per-query cardinality would show up) anywhere.
            assert "{" not in text
            value_lines = [
                line for line in text.splitlines() if line and not line.startswith("#")
            ]
            assert value_lines  # the endpoint actually returned some counters
            for line in value_lines:
                name, _, value = line.partition(" ")
                assert name.startswith("icberg_")
                assert value.strip().lstrip("-").isdigit()

        # The rate-limited bucket specifically is present and bucketed (not per-actor).
        assert "icberg_rate_limited_total" in low_client.get("/metrics").text


# ---------------------------------------------------------------------------
# P2.18 - api_error_redacted
# ---------------------------------------------------------------------------

class TestApiErrorRedacted:
    def test_api_error_redacted_scrubs_internal_error_response(self, app: FastAPI, db_path: str):
        """Force an internal error deep in the request path (inside `Gateway.handle`'s
        own `finally`-block audit write, which is NOT swallowed by `handle`'s own
        broad `except Exception` since it happens in `finally` after that block) and
        assert the API's response is a clean, generic, scrubbed 500 -- no traceback, no
        internal file path, no raw `proposed_sql`, no PII of any kind.

        Uses `raise_server_exceptions=False` on this test's own `TestClient` -- this is
        the standard way to test a registered `@app.exception_handler(Exception)`
        without pytest itself raising the underlying error (that flag exists purely as
        a *development* convenience so bugs surface loudly during testing; a real
        production ASGI server, and this test, exercise what an actual caller sees).
        """
        leaking_message = (
            "Traceback (most recent call last):\n"
            f'  File "{db_path}", line 42, in append\n'
            "    proposed_sql='UPDATE users SET admin=1 WHERE id=1'\n"
            "sqlite3.OperationalError: disk I/O error"
        )

        def _boom(*args, **kwargs):
            raise RuntimeError(leaking_message)

        app.state.audit_log.append = _boom
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/query", json={"sql": "SELECT id FROM users WHERE id=1 LIMIT 1", "actor": "agent-1"}
        )

        assert resp.status_code == 500
        body_text = resp.text

        assert "Traceback" not in body_text
        assert db_path not in body_text
        assert "proposed_sql" not in body_text
        assert "admin=1 WHERE id=1" not in body_text
        assert ".py" not in body_text
        assert "sqlite3" not in body_text.lower()
        assert "OperationalError" not in body_text
        # A generic message, nothing exception-derived.
        assert resp.json() == {"detail": "internal server error"}

    def test_api_error_redacted_approval_decision_path_too(self, app: FastAPI, db_path: str):
        resp_setup = TestClient(app).post(
            "/query", json={"sql": "UPDATE orders SET total=0 WHERE id=1", "actor": "agent-1"}
        )
        approval_id = resp_setup.json()["approval_id"]

        def _boom(*args, **kwargs):
            raise RuntimeError(f'internal secret path leak: "{db_path}" and proposed_sql="DROP TABLE orders"')

        app.state.audit_log.append = _boom
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            f"/approvals/{approval_id}", json={"decision": "approve", "approver": "root-admin"}
        )
        assert resp.status_code == 500
        assert db_path not in resp.text
        assert "proposed_sql" not in resp.text
        assert resp.json() == {"detail": "internal server error"}


# ---------------------------------------------------------------------------
# P2.19 - handle_readonly_guard
# ---------------------------------------------------------------------------

class TestHandleReadonlyGuard:
    def test_handle_readonly_guard_rejects_write_executor_on_read_path(self, db_path: str):
        """`Gateway.handle` must itself refuse to run the read path against a
        write-capable executor, regardless of what the caller intended -- it does not
        trust the caller's own bookkeeping about which executor is "the read one".
        """
        gw = Gateway(GovernanceGate())
        write_executor = WriteExecutor(db_path)
        audit_log = AuditLog(":memory:")

        with pytest.raises(TypeError):
            gw.handle("SELECT 1", "agent-1", write_executor, audit_log)

        # Nothing was audited -- the guard fires before the gate even evaluates the SQL.
        assert audit_log.entries() == []

    def test_handle_readonly_guard_rejects_executor_with_no_marker_at_all(self, db_path: str):
        """Fails CLOSED, not open: an executor-shaped object with no `IS_READONLY`
        attribute at all (e.g. a hand-rolled test double, or a future executor class
        that forgot to declare it) is rejected too, not treated as read-only by default.
        """

        class _UnmarkedExecutor:
            def execute(self, sql, params=()):  # pragma: no cover - never reached
                raise AssertionError("should never be called")

        gw = Gateway(GovernanceGate())
        audit_log = AuditLog(":memory:")

        with pytest.raises(TypeError):
            gw.handle("SELECT 1", "agent-1", _UnmarkedExecutor(), audit_log)

    def test_handle_readonly_guard_accepts_real_readonly_executor(self, db_path: str, client: TestClient):
        """Sanity check the guard doesn't false-positive: the actual gateway app (which
        wires a real `ReadOnlyExecutor` into the read path) keeps working normally.
        """
        resp = client.post(
            "/query", json={"sql": "SELECT id FROM users WHERE id=1 LIMIT 1", "actor": "agent-1"}
        )
        assert resp.status_code == 200
        assert resp.json()["action"] == "allow"
