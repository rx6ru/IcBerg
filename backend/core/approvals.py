"""Human-in-the-loop write-approval queue — the Phase 2 control that finally executes a
`hold` decision `sql_governance.py`/`gateway.py` have, since Phase 0, only ever decided
and audited, never run (see THREAT_MODEL.md's "The `hold` path is still not executed
under any control" residual gap).

A `hold`ed write is `enqueue`d here with the EXACT proposed SQL text, immutably, and sits
`pending` until a human calls `approve` or `reject`. This module is deliberately small and
paranoid about one property above all others: **the SQL a human approves is the SQL that
runs — verbatim, never re-derived.** `approve()` never re-parses, reconstructs, or accepts
a caller-supplied SQL string; it looks up the row by `id` and executes exactly the `sql`
column that was written at `enqueue()` time and never touched again. This is what closes
the TOCTOU/replay threat `.devdocs/PHASE2_GATES.md` calls out explicitly: mutating the
held statement between enqueue and approve, or approving a stale/re-submitted id, has no
code path here that could substitute a different statement for the one a human actually
reviewed.

The second paranoid property is atomicity of the pending -> decided transition. `_claim`
performs it as a single `UPDATE ... WHERE id=? AND status='pending' AND expires_at > ?`
against SQLite (which serializes writes on one connection) — exactly one caller can ever
observe `rowcount == 1` for a given id, so two concurrent `approve`/`reject` calls racing
on the same id can never both proceed to execution: the loser fails safe (raises
`ApprovalError`, an unconditional 4xx at the API layer — see `backend/api/gateway_routes.py`)
and never reaches `write_executor.execute` at all. The claim happens BEFORE execution, not
after, specifically so "decided" and "executed" can never straddle a race window where a
second caller sees the row as still `pending`.

Every transition — enqueue (implicitly, via the `hold` audit entry `gateway.py` already
appends before this module is ever invoked), approve, and reject — is durably recorded:
`approve`/`reject` each append their own `AuditLog` entry (`action="approved"`/
`"rejected"`, `actor=<the approver>`, with the approval id and original proposer folded
into `matched_rules` for traceability) so the audit trail shows not just what was proposed
but who authorized or refused it and when.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable

import structlog

from backend.core.redaction import redact_rows, redact_text

if TYPE_CHECKING:
    from backend.core.audit import AuditLog
    from backend.core.executor import ExecutionResult
    from backend.core.sql_governance import PolicyDecision

logger = structlog.get_logger(__name__)

# Default time-to-live for a pending approval before it can no longer be approved/rejected
# (only its terminal-state history remains, via `get()`) — a stale hold from an agent
# session that never got human attention should not be executable indefinitely.
DEFAULT_TTL_SECONDS = 24 * 60 * 60

_STATUS_PENDING = "pending"
_STATUS_APPROVED = "approved"
_STATUS_REJECTED = "rejected"


class ApprovalError(Exception):
    """Raised by `approve`/`reject` for every fail-safe case: unknown id, expired, or
    already-decided. `status_code` is the 4xx the API layer should surface — see
    `backend/api/gateway_routes.py`. Raising (rather than returning a sentinel) is
    deliberate: a caller cannot accidentally ignore the failure and fall through to
    executing something, the way a falsy return value could be mishandled.
    """

    def __init__(self, message: str, *, status_code: int = 409) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ApprovalRecord:
    """One row of the approval queue.

    Attributes:
        id: Opaque approval id (uuid4 hex), returned by `enqueue` and used to look the
            row back up.
        sql: The EXACT proposed SQL text, stored immutably at `enqueue` time. Never
            rewritten by any method in this class.
        actor: The agent/user who originally proposed `sql` (the gateway's `hold` audit
            entry already records this; kept here too so the pending queue is
            self-describing without joining back to the audit log).
        classification: The gate's classification of `sql` ("read"/"write"/"ddl"/"unknown").
        reason: The gate's `hold` reason, for display in a `GET /approvals` listing.
        matched_rules: The gate's matched rule names for `sql`.
        status: `"pending"`, `"approved"`, or `"rejected"`.
        created_at: ISO-8601 timestamp of `enqueue`.
        expires_at: ISO-8601 timestamp after which `approve`/`reject` fail safe.
        decided_at: ISO-8601 timestamp of the approve/reject decision, or `None` while
            still `pending`.
        approver: Identifier of who approved/rejected, or `None` while still `pending`.
    """

    id: str
    sql: str
    actor: str
    classification: str
    reason: str
    matched_rules: list[str] = field(default_factory=list)
    status: str = _STATUS_PENDING
    created_at: str = ""
    expires_at: str = ""
    decided_at: str | None = None
    approver: str | None = None


def _row_to_record(row: tuple[Any, ...]) -> ApprovalRecord:
    (id_, sql, actor, classification, reason, matched_rules_json, status,
     created_at, expires_at, decided_at, approver) = row
    return ApprovalRecord(
        id=id_,
        sql=sql,
        actor=actor,
        classification=classification,
        reason=reason,
        matched_rules=json.loads(matched_rules_json),
        status=status,
        created_at=created_at,
        expires_at=expires_at,
        decided_at=decided_at,
        approver=approver,
    )


_SELECT_COLUMNS = (
    "id, sql, actor, classification, reason, matched_rules, status, "
    "created_at, expires_at, decided_at, approver"
)


def _same_identity(a: str, b: str) -> bool:
    """Case- and surrounding-whitespace-insensitive identity comparison used by the
    self-approval check (M2, `.devdocs/PHASE2_GATES.md` P2.16). A bare `a == b`
    let `approver="agent-1 "` (a trailing space) or `approver="AGENT-1"` (a case
    variant) bypass the self-approval guard entirely, even though both are plainly
    the same claimed identity as `actor="agent-1"` — normalizing both sides before
    comparing closes that without changing behavior for the ordinary exact-match or
    genuinely-different-identity cases.
    """
    return a.strip().casefold() == b.strip().casefold()


class ApprovalQueue:
    """Persistent (SQLite-backed) queue of held writes awaiting human approval/rejection.

    Args:
        db_path: SQLite path, or `":memory:"` (default) for an in-process queue that
            persists across calls for the object's lifetime (mirrors `AuditLog`'s own
            `db_path` convention).
        clock: Optional zero-arg callable returning a `datetime`, injected for
            deterministic tests — same convention as `AuditLog`. Defaults to
            `datetime.now(timezone.utc)`.
        ttl_seconds: Default approval lifetime, used by `enqueue` when its own
            `ttl_seconds` override isn't given. Defaults to `DEFAULT_TTL_SECONDS`.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        *,
        clock: Callable[[], datetime] | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.ttl_seconds = ttl_seconds
        # A per-instance lock guarding EVERY access to `self.conn` — same H1 rationale
        # as `AuditLog._lock` (see that class's `__init__` for the full explanation of
        # why `check_same_thread=False` alone does not make concurrent use of one
        # connection object from multiple threads safe). `RLock` so `approve`/`reject`
        # can call `self.get`/`self._claim` (which also acquire this lock) from within
        # their own already-locked body without deadlocking on themselves.
        #
        # `approve()` additionally holds this SAME lock across the atomic claim, the
        # write execution, AND the `audit_log.append(...)` call that records it — not
        # three separate critical sections — which is what closes the second half of
        # H1: without a single shared critical section, one thread's `approve()` could
        # claim+execute a write and be preempted before it appends the audit entry,
        # while a health check or another approval's audit append interleaves; the
        # atomic claim alone only prevented a DOUBLE execution, not a write that
        # executes with no corresponding audit row ever landing at all.
        self._lock = threading.RLock()
        # `check_same_thread=False`: same rationale as `AuditLog` — this queue may be
        # driven from a single-threaded test or from an async/threaded server context;
        # thread-safety for concurrent callers is `self._lock`'s job, not this flag's.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS approvals (
                id TEXT PRIMARY KEY,
                sql TEXT NOT NULL,
                actor TEXT NOT NULL,
                classification TEXT NOT NULL,
                reason TEXT NOT NULL,
                matched_rules TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                decided_at TEXT,
                approver TEXT
            )
            """
        )
        self.conn.commit()

    def enqueue(
        self,
        decision: "PolicyDecision",
        sql: str,
        actor: str,
        *,
        ttl_seconds: float | None = None,
    ) -> str:
        """Enqueue `sql` (a `hold`ed proposal) for human approval and return its id.

        `sql` is stored EXACTLY as given — the gateway's own `hold` decision has already
        audited this same text (see module docstring); this call never mutates,
        normalizes, or re-parses it. `decision` supplies the descriptive metadata
        (`classification`, `reason`, `matched_rules`) shown in a pending-approvals
        listing; it is never used to decide whether to enqueue — the caller (the API
        layer) only calls this for a decision it already knows is `hold`.

        `ttl_seconds`, when given, overrides this queue's own `self.ttl_seconds` default
        for this one approval only — chiefly so tests can construct an
        already-expired approval deterministically (`ttl_seconds=-1`) without faking the
        clock forward.
        """
        approval_id = uuid.uuid4().hex
        now = self._clock()
        ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
        created_at = now.isoformat()
        expires_at = (now + timedelta(seconds=ttl)).isoformat()

        with self._lock:
            self.conn.execute(
                """
                INSERT INTO approvals
                    (id, sql, actor, classification, reason, matched_rules, status,
                     created_at, expires_at, decided_at, approver)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    approval_id,
                    sql,
                    actor,
                    decision.classification,
                    decision.reason,
                    json.dumps(list(decision.matched_rules)),
                    _STATUS_PENDING,
                    created_at,
                    expires_at,
                ),
            )
            self.conn.commit()
        logger.info(
            "approvals.enqueued",
            id=approval_id,
            actor=actor,
            classification=decision.classification,
            expires_at=expires_at,
        )
        return approval_id

    def pending(self) -> list[ApprovalRecord]:
        """Every approval currently `pending` and not yet expired, oldest first."""
        now = self._clock().isoformat()
        with self._lock:
            rows = self.conn.execute(
                f"SELECT {_SELECT_COLUMNS} FROM approvals "
                "WHERE status = ? AND expires_at > ? ORDER BY created_at ASC",
                (_STATUS_PENDING, now),
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def get(self, approval_id: str) -> ApprovalRecord | None:
        """The approval `approval_id`, in whatever state it's actually in (pending,
        approved, rejected, or pending-but-expired) — unlike `pending()`, this does NOT
        filter by expiry or status, since `_claim`'s fail-safe diagnostics need to tell
        "never existed" apart from "exists but expired/already decided".
        """
        with self._lock:
            row = self.conn.execute(
                f"SELECT {_SELECT_COLUMNS} FROM approvals WHERE id = ?", (approval_id,)
            ).fetchone()
        return _row_to_record(row) if row else None

    def ping(self) -> None:
        """Thread-safe connectivity check for `GET /health` (`gateway_routes.health`) —
        that route MUST call this rather than touching `self.conn` directly, so the
        health probe is guarded by the same lock as every other access to this
        connection instead of racing a concurrent `approve`/`reject`/`enqueue`.
        """
        with self._lock:
            self.conn.execute("SELECT 1")

    def _claim(self, approval_id: str, new_status: str, approver: str) -> ApprovalRecord:
        """Atomically transition `approval_id` from `pending` (and not expired) to
        `new_status`, or fail safe. See module docstring for why this single `UPDATE`'s
        `WHERE` clause is the entire TOCTOU/replay defense: at most one caller can ever
        see `rowcount == 1` for a given id, so a losing concurrent caller never reaches
        past this method at all — no execution, no second audit entry, nothing.

        Raises `ApprovalError` (never returns a sentinel) for every failure case:
          - unknown id: 404
          - exists but not `pending` any more (already approved/rejected, including by a
            concurrent caller that won the race this same call lost): 409
          - exists, still `pending`, but past `expires_at`: 410

        Runs entirely inside `self._lock` — the `UPDATE`, its `commit()`, and (on the
        failure path) the diagnostic re-`get()` are one atomic critical section, so a
        second thread can never observe this connection mid-transition.
        """
        with self._lock:
            now_dt = self._clock()
            now = now_dt.isoformat()
            cur = self.conn.execute(
                "UPDATE approvals SET status = ?, decided_at = ?, approver = ? "
                "WHERE id = ? AND status = ? AND expires_at > ?",
                (new_status, now, approver, approval_id, _STATUS_PENDING, now),
            )
            self.conn.commit()

            if cur.rowcount != 1:
                record = self.get(approval_id)
                if record is None:
                    raise ApprovalError(f"no such approval: {approval_id}", status_code=404)
                if record.status != _STATUS_PENDING:
                    raise ApprovalError(
                        f"approval {approval_id} was already {record.status}", status_code=409
                    )
                # Still `pending` on re-read, so the UPDATE's WHERE clause failed on the
                # one remaining condition it could have: `expires_at > now`.
                raise ApprovalError(f"approval {approval_id} has expired", status_code=410)

            record = self.get(approval_id)
            if record is None:
                # The UPDATE we just committed (rowcount == 1) guarantees this row
                # exists — reaching here means that invariant was violated (e.g. a
                # bypass of this class's own API deleted the row between the UPDATE
                # and this re-read). An explicit exception here (never a bare `assert`,
                # which `python -O` strips entirely, silently turning this into a
                # `NoneType` `AttributeError` deeper in the caller instead of a clear
                # diagnostic) so this invariant violation can never be optimized away.
                raise RuntimeError(
                    f"approvals invariant violated: approval {approval_id} vanished "
                    "immediately after its own claiming UPDATE committed rowcount=1"
                )
            return record

    def approve(
        self,
        approval_id: str,
        approver: str,
        write_executor: Any,
        audit_log: "AuditLog",
    ) -> "ExecutionResult":
        """Approve `approval_id`: atomically claim it (see `_claim` — fail-safe on
        unknown/expired/already-decided, with NO execution on any failure path), then
        execute the STORED `sql` — exactly what was proposed, never re-parsed or
        substituted — via `write_executor` (a real mutation), and append an audit entry
        `action="approved"` with `actor=approver` so the trail records who authorized it.

        Self-approval is refused (`.devdocs/PHASE2_GATES.md` P2.16): if `approver`
        matches the `actor` who originally proposed this SQL (`ApprovalRecord.actor`,
        set once at `enqueue` time — compared via `_same_identity`, case- and
        whitespace-insensitively, so `approver="agent-1 "`/`"AGENT-1"` can't bypass the
        check the exact string `"agent-1"` would trip — M2), this raises
        `ApprovalError(status_code=403)` BEFORE `_claim` is ever called — the row is
        left exactly as it was (still `pending` if it was pending), so a *different*,
        legitimate approver can still decide it afterward. This is an in-process
        stopgap only, since (per THREAT_MODEL.md's Phase 2 residual risks) neither
        `actor` nor `approver` is authenticated — nothing stops one caller from
        proposing as `"agent-1"` and then calling this endpoint again claiming to be
        `"agent-2"`. It closes the *literal* self-approval case (the same identifier,
        modulo case/whitespace, proposing and approving its own write) without
        pretending to solve identity spoofing.

        Returns the `ExecutionResult` of running the approved write, with its `rows`
        passed through `redact_rows` first (defense-in-depth for the unusual case of a
        write with a `RETURNING`-style result set — see `redaction.py`; an ordinary
        `INSERT`/`UPDATE`/`DELETE` has no rows to redact at all).

        H1 (audit coupling): the self-approval check, the atomic claim, the write
        execution, AND the `approved` audit append below all run inside ONE acquisition
        of `self._lock` — the same critical section, not four separate ones. This is
        deliberate: it is what guarantees no caller can ever observe a write that
        executed with no corresponding audit entry. If `audit_log.append` itself raises
        (its own connection error, a coding bug, anything), that exception propagates
        out of this method uncaught — it is NOT swallowed into a fabricated success
        return. The write has, at that point, already happened (there is no cross-
        connection transaction spanning `write_executor` and `audit_log` to roll back —
        see THREAT_MODEL.md's Phase 2 Residual Risks for this being an inherent limit
        of two independent SQLite connections, not something this lock can fix), but
        the CALLER never gets back a 200/`ExecutionResult` for it: `gateway_routes
        .decide_approval` has no `except` for a bare exception here, so it falls through
        to `create_gateway_app`'s catch-all handler and surfaces a scrubbed 500 —
        loud failure, never a silent "approved" response with no audit row behind it.
        """
        with self._lock:
            pre_claim_record = self.get(approval_id)
            if pre_claim_record is not None and _same_identity(pre_claim_record.actor, approver):
                logger.warning("approvals.self_approval_blocked", id=approval_id, actor=approver)
                raise ApprovalError(
                    f"self-approval is not permitted: approver {approver!r} originally "
                    "proposed this SQL",
                    status_code=403,
                )

            record = self._claim(approval_id, _STATUS_APPROVED, approver)

            result = write_executor.execute(record.sql)
            redacted_rows, _ = redact_rows(result.rows, result.columns, sql=record.sql)
            result = type(result)(
                rows=redacted_rows,
                columns=result.columns,
                rowcount=result.rowcount,
                truncated=result.truncated,
                latency_ms=result.latency_ms,
                error=result.error,
            )

            # Audit append happens HERE, still holding the same lock the claim and the
            # execution above ran under — see the docstring's "H1 (audit coupling)"
            # paragraph. No `try/except` around this call: an append failure must
            # propagate, not be caught and downgraded into a quiet success.
            audit_log.append(
                actor=approver,
                proposed_sql=record.sql,
                classification=record.classification,
                action="approved",
                matched_rules=[*record.matched_rules, f"approval_id={approval_id}", f"proposed_by={record.actor}"],
                rows_returned=result.rowcount,
                latency_ms=result.latency_ms,
                result_hash=audit_log.hash_result(redacted_rows),
            )
            logger.info(
                "approvals.approved",
                id=approval_id,
                approver=approver,
                error=redact_text(result.error) if result.error else None,
            )
            return result

    def reject(self, approval_id: str, approver: str, audit_log: "AuditLog") -> ApprovalRecord:
        """Reject `approval_id`: atomically claim it (same fail-safe guarantees as
        `approve` — see `_claim`), execute NOTHING, and append an audit entry
        `action="rejected"` with `actor=approver`. Runs entirely inside `self._lock`,
        same rationale as `approve` — see that method's "H1 (audit coupling)" docstring
        paragraph (reject has no write to couple to the audit entry, but the claim and
        the audit append still must not straddle a race with a concurrent caller).
        """
        with self._lock:
            record = self._claim(approval_id, _STATUS_REJECTED, approver)
            audit_log.append(
                actor=approver,
                proposed_sql=record.sql,
                classification=record.classification,
                action="rejected",
                matched_rules=[*record.matched_rules, f"approval_id={approval_id}", f"proposed_by={record.actor}"],
                rows_returned=0,
                latency_ms=0,
                result_hash=audit_log.hash_result([]),
            )
            logger.info("approvals.rejected", id=approval_id, approver=approver)
            decided = self.get(approval_id)
            if decided is None:
                # Same invariant/rationale as `_claim`'s explicit-exception fix — never
                # a bare `assert` (see that method's docstring for why).
                raise RuntimeError(
                    f"approvals invariant violated: approval {approval_id} vanished "
                    "immediately after being rejected"
                )
            return decided
