"""FastAPI router for the IcBerg governance gateway (Phase 2) — the "self-hostable
service / sidecar" integration surface from `.devdocs/FLAGSHIP_ROADMAP.md`'s "End
product & integration surfaces" section: a REST + SSE API in front of a governed
database, with no change to the database itself.

Endpoints:
    POST /query            - propose one SQL statement; evaluate -> execute-if-allowed
                              -> redact -> audit -> (hold -> enqueue approval).
    POST /query/stream      - same decision, streamed as SSE events: start -> decision ->
                              rows|held|blocked.
    GET  /approvals         - list pending approvals.
    POST /approvals/{id}    - approve or reject a pending approval.
    GET  /audit              - redacted audit trail + hash-chain verification status.
    GET  /health              - component connectivity check.
    GET  /metrics              - Prometheus-format counters.

This module is intentionally decoupled from `backend/api/routes.py` (the Titanic Q&A
API) and from `backend/main.py` — it is wired into its own app by `backend/gateway_app.py`,
never into the existing one. All per-request state (the shared `GovernanceGate`,
`Gateway`, `Connection`, `AuditLog`, `ApprovalQueue`, rate-limit buckets, metrics
counters) lives on `request.app.state`, the same pattern `backend/api/routes.py` already
uses for its own dependencies (`req.app.state.llm_adapter`, etc.) — one app instance per
governed database, built by `create_gateway_app`.

Every error/response string that could echo back proposed SQL, an engine error, or a DSN
is passed through `redact_text` before leaving this module — matching `gateway.py`'s own
"reason strings are PII-scrubbed" contract, extended here to approval and rate-limit
errors too.
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal

import structlog
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from backend.core.approvals import ApprovalError
from backend.core.redaction import redact_text

logger = structlog.get_logger(__name__)

router = APIRouter()

# Metrics counter keys tracked in `request.app.state.metrics` — initialized by
# `create_gateway_app` (`backend/gateway_app.py`) and only ever incremented here. Every
# key is a single bucketed, unlabeled counter — no per-actor/per-query cardinality, no
# raw SQL — see P2.17 (`.devdocs/PHASE2_GATES.md`) and the `metrics()` handler below.
_METRIC_KEYS = (
    "queries_total",
    "blocks_total",
    "holds_total",
    "approvals_total",
    "rejections_total",
    "rate_limited_total",
)


# --------------------------------------------------------------------------------------
# Request/response schemas
# --------------------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """One proposed SQL statement. `sql` is untrusted input end to end — see
    THREAT_MODEL.md's trust boundary; nothing here presumes it is safe.

    `model_config = ConfigDict(extra="forbid")` (P2.12, `.devdocs/PHASE2_GATES.md`): the
    database/connection this API talks to is server-config-only, fixed once at
    `create_gateway_app(dsn)` time on `request.app.state.connection` — nowhere in this
    module is a request body ever passed to `connectors.connector_for`. A caller cannot
    choose, override, or redirect the connection target via the request body at all, but
    this additionally REJECTS (422) any request that tries to smuggle one in (a `dsn`,
    `connection`, `database_url`, or any other unexpected field) rather than silently
    accepting and ignoring it — fail loud on an SSRF/file-read attempt, not fail quiet.
    """
    model_config = ConfigDict(extra="forbid")

    sql: str = Field(..., min_length=1, max_length=20_000, description="Proposed SQL statement (untrusted).")
    actor: str = Field(..., min_length=1, max_length=200, description="Identifier of the proposing agent/user.")


class QueryResponse(BaseModel):
    action: Literal["allow", "block", "hold"]
    reason: str
    matched_rules: list[str]
    rows: list[dict[str, Any]] | None = None
    audit_seq: int
    approval_id: str | None = None


class ApprovalDecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]
    approver: str = Field(..., min_length=1, max_length=200, description="Identifier of the human deciding.")


class ApprovalRecordOut(BaseModel):
    id: str
    sql: str
    actor: str
    classification: str
    reason: str
    matched_rules: list[str]
    status: str
    created_at: str
    expires_at: str
    decided_at: str | None = None
    approver: str | None = None


class ApprovalDecisionResponse(BaseModel):
    id: str
    status: str
    approver: str
    rows: list[dict[str, Any]] | None = None
    rowcount: int = 0
    error: str | None = None


class AuditEntryOut(BaseModel):
    seq: int
    timestamp: str
    actor: str
    proposed_sql: str
    classification: str
    action: str
    matched_rules: list[str]
    rows_returned: int
    latency_ms: int
    result_hash: str


class AuditResponse(BaseModel):
    entries: list[AuditEntryOut]
    chain_valid: bool
    broken_at_seq: int | None


# --------------------------------------------------------------------------------------
# Rate limiting — per-actor, sliding 60s window, in-memory (per app instance)
# --------------------------------------------------------------------------------------

def _check_rate_limit(request: Request, actor: str) -> None:
    """Raise 429 if `actor` has already made `rate_limit_per_minute` requests in the
    trailing 60 seconds. State lives on `request.app.state.rate_buckets`
    (`dict[str, list[float]]`, monotonic timestamps) — one bucket per actor, so one noisy
    or misbehaving agent can never starve another's quota.
    """
    buckets: dict[str, list[float]] = request.app.state.rate_buckets
    limit: int = request.app.state.rate_limit_per_minute
    now = time.monotonic()
    window = [t for t in buckets.get(actor, []) if now - t < 60.0]
    if len(window) >= limit:
        logger.warning("gateway_api.rate_limited", actor=redact_text(actor))
        request.app.state.metrics["rate_limited_total"] += 1
        raise HTTPException(status_code=429, detail="rate limit exceeded; try again shortly")
    window.append(now)
    buckets[actor] = window


def _sse(event_type: str, payload: dict[str, Any]) -> str:
    return f"data: {json.dumps({'type': event_type, **payload})}\n\n"


# --------------------------------------------------------------------------------------
# Query endpoints
# --------------------------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request) -> QueryResponse:
    """Evaluate + (maybe) execute + redact + audit one proposed statement. A `hold`
    decision additionally enqueues an approval and returns its id, with `rows=None` — see
    module docstring. `gate.evaluate` is called once here (in addition to the identical,
    side-effect-free evaluation `gateway.handle` performs internally) purely to obtain the
    full `PolicyDecision` `ApprovalQueue.enqueue` needs (`classification`/`matched_rules`
    for the pending-approvals listing) — `evaluate()` is a pure function over `body.sql`,
    so calling it twice is redundant computation, never a second decision that could
    disagree with the first.
    """
    _check_rate_limit(request, body.actor)
    state = request.app.state

    decision = state.gate.evaluate(body.sql, actor=body.actor)
    result = state.gateway.handle(body.sql, body.actor, state.connection.read_executor, state.audit_log)

    state.metrics["queries_total"] += 1
    approval_id: str | None = None
    if result["action"] == "block":
        state.metrics["blocks_total"] += 1
    elif result["action"] == "hold":
        state.metrics["holds_total"] += 1
        approval_id = state.approval_queue.enqueue(decision, body.sql, body.actor)

    return QueryResponse(
        action=result["action"],
        reason=result["reason"],
        matched_rules=result["matched_rules"],
        rows=result["rows"],
        audit_seq=result["audit_seq"],
        approval_id=approval_id,
    )


@router.post("/query/stream")
def query_stream(body: QueryRequest, request: Request) -> StreamingResponse:
    """Same decision as `POST /query`, streamed as SSE events: `start` (immediately) ->
    `decision` (the gate's outcome) -> exactly one of `rows` (allow), `held` (hold, with
    the new approval id), or `blocked` (block) as the final event.
    """
    _check_rate_limit(request, body.actor)
    state = request.app.state

    def events():
        yield _sse("start", {"actor": body.actor})

        decision = state.gate.evaluate(body.sql, actor=body.actor)
        result = state.gateway.handle(body.sql, body.actor, state.connection.read_executor, state.audit_log)
        state.metrics["queries_total"] += 1

        yield _sse("decision", {
            "action": result["action"],
            "reason": result["reason"],
            "matched_rules": result["matched_rules"],
            "audit_seq": result["audit_seq"],
        })

        if result["action"] == "allow":
            yield _sse("rows", {"rows": result["rows"]})
        elif result["action"] == "hold":
            state.metrics["holds_total"] += 1
            approval_id = state.approval_queue.enqueue(decision, body.sql, body.actor)
            yield _sse("held", {"approval_id": approval_id})
        else:
            state.metrics["blocks_total"] += 1
            yield _sse("blocked", {"reason": result["reason"]})

    return StreamingResponse(events(), media_type="text/event-stream")


# --------------------------------------------------------------------------------------
# Approval endpoints
# --------------------------------------------------------------------------------------

@router.get("/approvals", response_model=list[ApprovalRecordOut])
def list_approvals(request: Request) -> list[ApprovalRecordOut]:
    """Every currently pending, not-yet-expired approval, with PII VALUES redacted from
    every free-text field (`sql`, `reason`, `actor`, `approver`) before they leave this
    endpoint (M3/M4 hardening — `.devdocs/PHASE2_GATES.md`'s Phase 2 residual-risk
    review). `redact_text` is the same value-pattern scrub `audit.py`'s `AuditLog
    .append` already applies to `proposed_sql`, so `UPDATE users SET x=1 WHERE
    email='alice@example.com'` renders as `... WHERE email=[EMAIL_REDACTED]` here: an
    approver still sees the query's full STRUCTURE (which table, which columns, the
    shape of the `WHERE` clause) needed to judge whether to approve it, just not the
    literal PII value embedded in it. `classification`/`status`/timestamps are not
    free text and pass through unchanged. See THREAT_MODEL.md's "## Phase 2 Residual
    Risks" for why this endpoint must still sit behind real authentication in
    production regardless of this redaction — it narrows what an UNAUTHORIZED viewer
    could scrape from this endpoint, it does not substitute for access control.
    """
    records = request.app.state.approval_queue.pending()
    return [
        ApprovalRecordOut(
            id=r.id,
            sql=redact_text(r.sql),
            actor=redact_text(r.actor),
            classification=r.classification,
            reason=redact_text(r.reason),
            matched_rules=[redact_text(rule) for rule in r.matched_rules],
            status=r.status,
            created_at=r.created_at,
            expires_at=r.expires_at,
            decided_at=r.decided_at,
            approver=redact_text(r.approver) if r.approver else r.approver,
        )
        for r in records
    ]


@router.post("/approvals/{approval_id}", response_model=ApprovalDecisionResponse)
def decide_approval(approval_id: str, body: ApprovalDecisionRequest, request: Request) -> ApprovalDecisionResponse:
    """Approve or reject one pending approval. Fails safe (4xx, no execution) for an
    unknown, expired, or already-decided id — see `approvals.ApprovalQueue._claim`; that
    fail-safe is the ONLY source of a non-2xx response here, so this handler is a thin,
    literal translation of `ApprovalError.status_code` and never itself decides whether
    execution is safe.
    """
    state = request.app.state
    try:
        if body.decision == "approve":
            result = state.approval_queue.approve(approval_id, body.approver, state.connection.write_executor, state.audit_log)
            state.metrics["approvals_total"] += 1
            return ApprovalDecisionResponse(
                id=approval_id,
                status="approved",
                approver=body.approver,
                rows=result.rows,
                rowcount=result.rowcount,
                error=redact_text(result.error) if result.error else None,
            )
        record = state.approval_queue.reject(approval_id, body.approver, state.audit_log)
        state.metrics["rejections_total"] += 1
        return ApprovalDecisionResponse(id=approval_id, status=record.status, approver=body.approver)
    except ApprovalError as exc:
        logger.warning("gateway_api.approval_failsafe", approval_id=approval_id, decision=body.decision, error=str(exc))
        raise HTTPException(status_code=exc.status_code, detail=redact_text(str(exc))) from None


# --------------------------------------------------------------------------------------
# Audit / health / metrics
# --------------------------------------------------------------------------------------

@router.get("/audit", response_model=AuditResponse)
def get_audit(request: Request) -> AuditResponse:
    """The full audit trail plus hash-chain verification status. `proposed_sql` is
    already PII-scrubbed at write time (`AuditLog.append` -> `redact_text`, see
    `audit.py`) — this endpoint applies no additional redaction to it, since doing so a
    second time on already-scrubbed text would be a no-op; `actor` is scrubbed here too,
    defense-in-depth against an actor identifier that happens to itself look like PII.

    `matched_rules` is ALSO scrubbed here (M3 hardening): `approvals.py`'s `approve`/
    `reject` fold `f"proposed_by={record.actor}"` into `matched_rules` for
    traceability, and `actor` there is the caller-supplied, UNREDACTED `QueryRequest
    .actor` from the original `/query` proposal — an email-shaped actor value
    (`proposed_by=alice@example.com`) would otherwise reach this response raw even
    though the entry's own `actor` field is redacted right above it. Every
    `matched_rules` entry is scrubbed the same way regardless of source, not just the
    `proposed_by=` ones, since a future rule name or annotation could just as easily
    carry a PII-shaped value.
    """
    audit_log = request.app.state.audit_log
    ok, broken_seq = audit_log.verify()
    entries = [
        AuditEntryOut(
            seq=e.seq,
            timestamp=e.timestamp,
            actor=redact_text(e.actor),
            proposed_sql=e.proposed_sql,
            classification=e.classification,
            action=e.action,
            matched_rules=[redact_text(rule) for rule in e.matched_rules],
            rows_returned=e.rows_returned,
            latency_ms=e.latency_ms,
            result_hash=e.result_hash,
        )
        for e in audit_log.entries()
    ]
    return AuditResponse(entries=entries, chain_valid=ok, broken_at_seq=broken_seq)


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Component connectivity check: the audit log's and approval queue's own SQLite
    connections, plus which backend the configured `Connection` targets. Goes through
    each store's own `ping()` (H1 hardening) rather than touching `.conn` directly —
    `AuditLog`/`ApprovalQueue`'s connections are only safe for concurrent, cross-thread
    access when EVERY access goes through their own lock, including this health probe;
    a raw `.conn.execute(...)` here would be exactly the kind of unguarded access H1's
    fix eliminates everywhere else.
    """
    state = request.app.state
    components: dict[str, str] = {}

    try:
        state.audit_log.ping()
        components["audit_log"] = "ok"
    except Exception:
        components["audit_log"] = "error"

    try:
        state.approval_queue.ping()
        components["approval_queue"] = "ok"
    except Exception:
        components["approval_queue"] = "error"

    components["connector"] = state.connection.backend

    errors = [k for k, v in components.items() if v == "error"]
    status = "healthy" if not errors else "unhealthy"
    return {"status": status, "components": components}


@router.get("/metrics")
def metrics(request: Request) -> Response:
    """Prometheus text-exposition-format counters: queries, blocks, holds, approvals,
    rejections, and rate-limited requests since this app instance started.

    P2.17 (`.devdocs/PHASE2_GATES.md`): every line here is one of the fixed,
    already-known `_METRIC_KEYS` — a bare counter name and an integer value, no
    Prometheus label set (`{...}`) of any kind. There is no code path from here to
    `proposed_sql`, `actor`, or any other per-request value; only the aggregate `int`
    counters on `request.app.state.metrics` are ever read. Do not add a labeled metric
    (e.g. `icberg_queries_total{actor="..."}`) to this endpoint — that would reintroduce
    exactly the high-cardinality/PII-in-labels leak this gate exists to prevent.
    """
    m: dict[str, int] = request.app.state.metrics
    lines: list[str] = []
    for key in _METRIC_KEYS:
        name = f"icberg_{key}"
        lines.append(f"# HELP {name} Total count of {key.replace('_', ' ')}.")
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name} {m[key]}")
    return Response(content="\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")
