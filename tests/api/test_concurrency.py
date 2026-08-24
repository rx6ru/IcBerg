"""H1 (HIGH) regression tests — concurrency / audit-integrity of the write-approval
workflow (`.devdocs/PHASE2_GATES.md`'s crown gates, extended): `ApprovalQueue`/
`AuditLog` share ONE SQLite connection each across every sync FastAPI route, which
Starlette runs in its own threadpool — concurrent callers on the same process are a
real, not hypothetical, scenario. Before the fix in `backend/core/approvals.py`/
`backend/core/audit.py`, N concurrent `POST /approvals/{id}` calls raced the shared
`sqlite3.Connection` objects directly and threw `InterfaceError`/`OperationalError`/
`AssertionError` as unhandled 500s, and the atomic claim-before-execute guarantee
alone did not prevent a write from executing with no corresponding audit entry landing
(the two were separate, unguarded critical sections). This file proves both are closed:
thread-safe DB access (no unhandled 500s, no chain corruption) AND execution/audit
coupling (audit-count always equals executions, never fewer).

Uses a throwaway SQLite file per test (`tmp_path`) — no real data, no external infra.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from fastapi.testclient import TestClient

from backend.gateway_app import create_gateway_app

N_CONCURRENT = 50
N_STABILITY_RUNS = 3  # run each scenario a few times to catch flakiness, not just once

USERS_SCHEMA_SQL = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT,
    admin INTEGER
)
"""

ORDERS_SCHEMA_SQL = """
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    total INTEGER
)
"""


def _make_db(path: str, n_orders: int) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(USERS_SCHEMA_SQL)
        conn.execute("INSERT INTO users (id, name, admin) VALUES (1, 'Alice', 0)")
        conn.execute(ORDERS_SCHEMA_SQL)
        conn.executemany(
            "INSERT INTO orders (id, total) VALUES (?, 0)", [(i,) for i in range(1, n_orders + 1)]
        )
        conn.commit()
    finally:
        conn.close()


def _read_users_admin(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT admin FROM users WHERE id=1").fetchone()[0]
    finally:
        conn.close()


class TestApprovalConcurrencySingleHold:
    """N concurrent `approve` calls racing on the SAME held write must produce exactly
    one execution, zero unhandled 500s, a still-valid hash chain, and an
    `approved`-audit-entry count that exactly equals the execution count (1) — the
    literal H1 assertion: never a write that ran with no audit row behind it.
    """

    @pytest.mark.parametrize("run", range(N_STABILITY_RUNS))
    def test_approval_concurrency_single_hold_exactly_one_execution(
        self, run: int, tmp_path
    ) -> None:
        db_path = str(tmp_path / f"single_{run}.sqlite")
        _make_db(db_path, n_orders=1)
        app = create_gateway_app(db_path, rate_limit_per_minute=1_000_000)
        client = TestClient(app)

        enqueue_resp = client.post(
            "/query", json={"sql": "UPDATE users SET admin=1 WHERE id=1", "actor": "agent-1"}
        )
        assert enqueue_resp.status_code == 200
        approval_id = enqueue_resp.json()["approval_id"]
        assert approval_id

        def _approve(i: int) -> int:
            resp = client.post(
                f"/approvals/{approval_id}",
                json={"decision": "approve", "approver": f"approver-{i}"},
            )
            return resp.status_code

        statuses: list[int] = []
        with ThreadPoolExecutor(max_workers=N_CONCURRENT) as pool:
            futures = [pool.submit(_approve, i) for i in range(N_CONCURRENT)]
            for future in as_completed(futures):
                statuses.append(future.result())

        counts = Counter(statuses)

        # 0 unhandled 500s from the race -- every response is either the one winner
        # (200) or a clean fail-safe 4xx (409, the loser of the atomic claim).
        assert counts[500] == 0, f"unhandled 500s from the race: {counts}"
        assert set(counts) <= {200, 409}, f"unexpected status codes: {counts}"

        # Exactly 1 execution -- the row changed exactly once, and exactly one caller
        # got the 200 that corresponds to it.
        assert counts[200] == 1, f"expected exactly 1 successful approve, got: {counts}"
        assert _read_users_admin(db_path) == 1

        # AuditLog.verify() still true -- the hash chain survived concurrent appends
        # (holds, the approved entry, and the losers' failed-claim paths, which never
        # reach `audit_log.append` at all) with no corruption from a torn read of
        # `prev_hash`/`seq`.
        audit_log = app.state.audit_log
        ok, broken_at_seq = audit_log.verify()
        assert ok is True, f"chain broken at seq={broken_at_seq}"
        assert broken_at_seq is None

        # audit-count == executions: exactly one "approved" audit entry, matching the
        # exactly-one execution above. This is the literal H1 guarantee -- a run of
        # this fixture against the pre-fix code could show 1 execution but 0 approved
        # audit entries (the unaudited-write bug), not just a mismatched count.
        approved_entries = [e for e in audit_log.entries() if e.action == "approved"]
        assert len(approved_entries) == counts[200] == 1


class TestApprovalConcurrencyDistinct:
    """N concurrent `approve` calls on N DIFFERENT held writes (no id contention at
    all) must ALL succeed, ALL be individually audited, and leave a fully-valid hash
    chain -- proving the shared `ApprovalQueue`/`AuditLog` connections are safe under
    genuine concurrent access, not merely safe because only one caller is ever
    actually inside a critical section at a time in the single-hold scenario above.
    """

    @pytest.mark.parametrize("run", range(N_STABILITY_RUNS))
    def test_approval_concurrency_distinct_all_audited_chain_valid(
        self, run: int, tmp_path
    ) -> None:
        db_path = str(tmp_path / f"distinct_{run}.sqlite")
        _make_db(db_path, n_orders=N_CONCURRENT)
        app = create_gateway_app(db_path, rate_limit_per_minute=1_000_000)
        client = TestClient(app)

        approval_ids: list[str] = []
        for order_id in range(1, N_CONCURRENT + 1):
            resp = client.post(
                "/query",
                json={
                    "sql": f"UPDATE orders SET total=total+1 WHERE id={order_id}",
                    "actor": "agent-1",
                },
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["action"] == "hold", body
            approval_ids.append(body["approval_id"])
        assert len(set(approval_ids)) == N_CONCURRENT  # all genuinely distinct ids

        def _approve(approval_id: str) -> int:
            resp = client.post(
                f"/approvals/{approval_id}",
                json={"decision": "approve", "approver": "root-admin"},
            )
            return resp.status_code

        statuses: list[int] = []
        with ThreadPoolExecutor(max_workers=N_CONCURRENT) as pool:
            futures = [pool.submit(_approve, aid) for aid in approval_ids]
            for future in as_completed(futures):
                statuses.append(future.result())

        counts = Counter(statuses)
        assert counts[500] == 0, f"unhandled 500s from the race: {counts}"
        assert counts[200] == N_CONCURRENT, f"expected all {N_CONCURRENT} to succeed, got: {counts}"

        # Every held write actually executed exactly once each.
        conn = sqlite3.connect(db_path)
        try:
            totals = conn.execute("SELECT total FROM orders").fetchall()
        finally:
            conn.close()
        assert all(t == (1,) for t in totals), totals

        audit_log = app.state.audit_log
        ok, broken_at_seq = audit_log.verify()
        assert ok is True, f"chain broken at seq={broken_at_seq}"
        assert broken_at_seq is None

        approved_entries = [e for e in audit_log.entries() if e.action == "approved"]
        assert len(approved_entries) == N_CONCURRENT  # every execution audited, none missing
