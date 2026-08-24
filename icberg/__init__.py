"""icberg — the public SDK for IcBerg, an AI<->database governance gateway.

`pip install icberg`; wrap DB access in-process instead of handing an agent raw
credentials. Every code path this package exposes routes through the SAME governance
gateway (`backend.core.gateway.Gateway` over `backend.core.sql_governance
.GovernanceGate`) as the REST/SSE API (`backend.gateway_app`), the MCP server
(`backend.mcp_server`), and the LangChain/LangGraph adapter
(`backend.integrations.langgraph_tool`) — see `.devdocs/PHASE3_GATES.md` P3.5. There is
no code path in this package that hands a caller a raw executor or database connection;
`GovernedConnection`, the one stateful handle this module offers, deliberately has no
`.read_executor`/`.write_executor`/`.connection` attribute in its public surface.

Public API (stable):
    Gateway, GovernanceGate, PolicyDecision   - re-exported core primitives, for callers
                                                 who want to build their own wiring instead
                                                 of using the convenience helpers below.
    Policy, load_policy                        - policy YAML (see `backend.core.policy`).
    govern(sql, *, actor, connection=..., dsn=..., policy=None) -> dict
                                                 - one-shot: govern a single statement.
    governed_connection(dsn, *, policy=None) -> GovernedConnection
                                                 - a reusable, stateful governed handle.
    GovernedConnection                          - the class `governed_connection` returns.

Quickstart::

    from icberg import govern

    result = govern("SELECT * FROM users WHERE id=1 LIMIT 5", actor="agent-1", dsn="app.db")
    result["action"]   # "allow"
    result["rows"]      # [{"id": 1, "name": "...", "email": "[REDACTED]", ...}]

    result = govern("DROP TABLE users", actor="agent-1", dsn="app.db")
    result["action"]   # "block"

For repeated calls against the same database (so the audit trail/approval queue persist
across queries, rather than a fresh ephemeral log per `govern()` call — see that
function's docstring)::

    from icberg import governed_connection

    db = governed_connection("app.db", policy="policy.yaml")
    db.query("SELECT * FROM orders WHERE id=1 LIMIT 5", actor="agent-1")
    db.query("UPDATE orders SET status='shipped' WHERE id=1", actor="agent-1")  # -> hold
    db.pending_approvals()
    db.approve(approval_id, approver="alice")
    db.audit_tail(20)
"""

from __future__ import annotations

from typing import Any

from backend.core.approvals import ApprovalQueue
from backend.core.audit import AuditLog
from backend.core.connectors import Connection, connector_for
from backend.core.gateway import Gateway
from backend.core.policy import Policy, load_policy
from backend.core.redaction import redact_text
from backend.core.sql_governance import GovernanceGate, PolicyDecision

__all__ = [
    "Gateway",
    "GovernanceGate",
    "PolicyDecision",
    "Policy",
    "load_policy",
    "govern",
    "governed_connection",
    "GovernedConnection",
]

__version__ = "0.3.0"

_PolicyLike = str | dict[str, Any] | Policy | None


def _resolve_policy(policy: Any) -> Policy | None:
    """`load_policy` already accepts a `Policy`/dict/path/`None` and returns a `Policy`/
    `None` — this is a thin, explicitly-typed wrapper so every public function in this
    module documents the same accepted shapes without repeating the union type.
    """
    return load_policy(policy)


class GovernedConnection:
    """The ONLY stateful handle this SDK ever hands back for a live database.

    Deliberately does NOT expose a `.read_executor`, `.write_executor`, `.connection`,
    or any other attribute that would let a caller reach a raw executor or DB connection
    (`.devdocs/PHASE3_GATES.md` P3.5) — every method below routes through the same
    `Gateway`/`GovernanceGate` pair the REST API (`backend.gateway_app`) builds around a
    `backend.core.connectors.Connection`, and mirrors `backend/api/gateway_routes.py`'s
    `POST /query`/`GET /approvals`/`POST /approvals/{id}`/`GET /audit` behavior closely
    enough that the two surfaces make identical decisions for identical input — see
    `tests/integration/test_sdk.py`'s `surfaces_consistent` test.

    Construct via `governed_connection(dsn, ...)`, not directly, in normal use.
    """

    def __init__(
        self,
        connection: Connection,
        *,
        policy: Policy | None = None,
        audit_log: AuditLog | None = None,
        approval_queue: ApprovalQueue | None = None,
        gate: GovernanceGate | None = None,
    ) -> None:
        self._connection = connection
        self._policy = policy
        self._gate = gate or GovernanceGate()
        self._gateway = Gateway(self._gate)
        self._audit_log = audit_log or AuditLog()
        self._approval_queue = approval_queue or ApprovalQueue()

    @property
    def backend(self) -> str:
        """Which database engine this connection targets (`"sqlite"`/`"postgres"`/
        `"mysql"`) — informational only, mirroring `connectors.Connection.backend`.
        """
        return self._connection.backend

    def query(self, sql: str, actor: str) -> dict[str, Any]:
        """Propose `sql` (untrusted); governed end to end: decide -> execute-if-allowed
        -> redact -> audit, exactly like `POST /query` in `backend/api/gateway_routes.py`.
        A `hold` decision additionally enqueues an approval, with the returned dict's
        `approval_id` set to its id (`None` for `allow`/`block`).

        Returns a dict with `action`, `reason`, `matched_rules`, `rows` (redacted, or
        `None` unless `action == "allow"`), `redaction_report`, `audit_seq`, and
        `approval_id`.
        """
        # `gate.evaluate` is called once here (in addition to the identical, side-effect-
        # free evaluation `Gateway.handle` performs internally) purely to obtain the full
        # `PolicyDecision` `ApprovalQueue.enqueue` needs — same pattern as
        # `gateway_routes.query`. It is a pure function of `sql`, so calling it twice can
        # never disagree with the decision `Gateway.handle` actually acted on.
        decision = self._gate.evaluate(sql, actor=actor)
        result = self._gateway.handle(sql, actor, self._connection.read_executor, self._audit_log, policy=self._policy)

        approval_id: str | None = None
        if result["action"] == "hold":
            approval_id = self._approval_queue.enqueue(decision, sql, actor)

        return {**result, "approval_id": approval_id}

    def request_write(self, sql: str, actor: str) -> dict[str, Any]:
        """Propose a write. There is no separate "write path" to route through here —
        every proposal, read or write, goes through the exact same `.query()`/`Gateway
        .handle` decision path; a write is never auto-executed (it is always `hold`ed for
        approval, or `block`ed outright by the base gate/policy). Named separately purely
        so callers can express intent at the call site.
        """
        return self.query(sql, actor)

    def approve(self, approval_id: str, approver: str) -> dict[str, Any]:
        """Approve a pending write: executes the EXACT SQL a human/approver already
        reviewed (never re-derived — see `approvals.ApprovalQueue.approve`) via the
        connection's write executor, which this method uses internally but never
        exposes. Raises `approvals.ApprovalError` for an unknown/expired/already-decided
        id or a self-approval attempt — never silently no-ops.
        """
        result = self._approval_queue.approve(approval_id, approver, self._connection.write_executor, self._audit_log)
        return {
            "rows": result.rows,
            "rowcount": result.rowcount,
            "truncated": result.truncated,
            "latency_ms": result.latency_ms,
            "error": redact_text(result.error) if result.error else None,
        }

    def reject(self, approval_id: str, approver: str) -> dict[str, Any]:
        """Reject a pending write: executes nothing, records the decision."""
        record = self._approval_queue.reject(approval_id, approver, self._audit_log)
        return {"id": record.id, "status": record.status}

    def pending_approvals(self) -> list[dict[str, Any]]:
        """Every currently pending, not-yet-expired approval, PII-value-redacted the
        same way `GET /approvals` redacts them (`gateway_routes.list_approvals`) — the
        query's structure is preserved, literal PII values embedded in it are not.
        """
        return [_approval_record_to_dict(r) for r in self._approval_queue.pending()]

    def audit_tail(self, n: int = 20) -> list[dict[str, Any]]:
        """The last `n` entries of the tamper-evident audit trail, PII-value-redacted the
        same way `GET /audit` redacts them (`proposed_sql` is already scrubbed at write
        time by `AuditLog.append`; `actor`/`matched_rules` are scrubbed here, same as
        `gateway_routes.get_audit`).
        """
        entries = self._audit_log.entries()
        return [_audit_entry_to_dict(e) for e in entries[-n:]] if n > 0 else []

    def verify_audit(self) -> tuple[bool, int | None]:
        """`(chain_valid, broken_at_seq)` — see `audit.AuditLog.verify`."""
        return self._audit_log.verify()


def _approval_record_to_dict(record: Any) -> dict[str, Any]:
    return {
        "id": record.id,
        "sql": redact_text(record.sql),
        "actor": redact_text(record.actor),
        "classification": record.classification,
        "reason": redact_text(record.reason),
        "matched_rules": [redact_text(rule) for rule in record.matched_rules],
        "status": record.status,
        "created_at": record.created_at,
        "expires_at": record.expires_at,
        "decided_at": record.decided_at,
        "approver": redact_text(record.approver) if record.approver else record.approver,
    }


def _audit_entry_to_dict(entry: Any) -> dict[str, Any]:
    return {
        "seq": entry.seq,
        "timestamp": entry.timestamp,
        "actor": redact_text(entry.actor),
        "proposed_sql": entry.proposed_sql,
        "classification": entry.classification,
        "action": entry.action,
        "matched_rules": [redact_text(rule) for rule in entry.matched_rules],
        "rows_returned": entry.rows_returned,
        "latency_ms": entry.latency_ms,
        "result_hash": entry.result_hash,
    }


def governed_connection(dsn: str, *, policy: _PolicyLike = None) -> GovernedConnection:
    """Build a reusable `GovernedConnection` around `dsn` (a SQLite path/URI,
    `postgres://...`, or `mysql://...` — see `backend.core.connectors.connector_for` for
    exact DSN forms). `policy` may be a `backend.core.policy.Policy`, a dict, a YAML file
    path, or `None` (no policy) — see `load_policy`.

    Building never opens a connection itself (`connector_for`'s own contract) — only
    calling `.query()`/`.approve()` on the returned handle does.
    """
    connection = connector_for(dsn)
    return GovernedConnection(connection, policy=_resolve_policy(policy))


def govern(
    sql: str,
    *,
    actor: str,
    connection: Connection | None = None,
    dsn: str | None = None,
    policy: _PolicyLike = None,
) -> dict[str, Any]:
    """One-shot convenience: govern a single `sql` statement against `connection` (an
    already-built `backend.core.connectors.Connection`) or `dsn` (built fresh via
    `connector_for`) and return the governed decision — `action`, `reason`,
    `matched_rules`, `rows` (redacted PII, present only when `action == "allow"`),
    `redaction_report`, `audit_seq`, `approval_id`.

    Exactly one of `connection`/`dsn` must be given. Builds a fresh, ephemeral
    `GovernedConnection` (a private, in-memory `AuditLog`/`ApprovalQueue`) for this one
    call — a `hold` decision's `approval_id` is real and enqueued, but nothing else in
    this process can `.approve()` it unless you keep the connection around yourself; for
    an audit trail/approval queue that persists across multiple calls, use
    `governed_connection(dsn)` once and call `.query()` on it repeatedly instead.

    Raises `ValueError` if neither `connection` nor `dsn` is given, or both are.
    """
    if (connection is None) == (dsn is None):
        raise ValueError("govern() requires exactly one of `connection` or `dsn`")
    if connection is None:
        connection = connector_for(dsn)  # type: ignore[arg-type]

    handle = GovernedConnection(connection, policy=_resolve_policy(policy))
    return handle.query(sql, actor)
