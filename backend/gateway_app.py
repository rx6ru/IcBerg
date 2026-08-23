"""Dedicated FastAPI app for the IcBerg governance gateway (Phase 2).

This is a SEPARATE application from `backend/main.py` (the Titanic Q&A demo app) — it is
never imported by, mounted into, or otherwise wired to that app, and `backend/main.py` is
untouched by Phase 2. `create_gateway_app(dsn)` builds one self-contained governed-API
instance around a single database (per `.devdocs/FLAGSHIP_ROADMAP.md`'s "self-hostable
service / sidecar" integration surface: "a REST + SSE API in front of your DB ... with no
change to your database"), composing:

    connectors.connector_for(dsn)  -> read_executor / write_executor for that DSN's backend
    sql_governance.GovernanceGate  -> the policy decision function
    gateway.Gateway                -> decide -> execute(allow) -> redact -> audit
    audit.AuditLog                 -> hash-chained, tamper-evident audit trail
    approvals.ApprovalQueue        -> the human-in-the-loop hold/approve/reject workflow
    backend.api.gateway_routes     -> the REST + SSE surface over all of the above

Each is a factory call, not a module-level singleton, specifically so a test (or an
operator running several governed databases in one process) can build multiple, fully
isolated app instances with no shared state between them.
"""

from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from datetime import datetime
from typing import Callable

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.api.gateway_routes import router as gateway_router
from backend.core.approvals import DEFAULT_TTL_SECONDS, ApprovalQueue
from backend.core.audit import AuditLog
from backend.core.connectors import connector_for
from backend.core.gateway import Gateway
from backend.core.sql_governance import GovernanceGate

logger = structlog.get_logger(__name__)

# The one, fixed, PII-free message every unhandled exception is scrubbed down to before
# it reaches a caller (P2.18, `.devdocs/PHASE2_GATES.md`) — deliberately a constant, not
# built from `str(exc)` in any way, so there is no code path by which a traceback, an
# internal file path, a raw `proposed_sql`, or any other exception-carried detail could
# leak into a response even if a future exception type's message happened to contain one.
_GENERIC_ERROR_DETAIL = "internal server error"

# Default per-actor cap enforced by `gateway_routes._check_rate_limit` — generous enough
# for normal agent traffic; callers that need a different cap (or a low one, for a rate
# limit test) pass `rate_limit_per_minute` explicitly.
DEFAULT_RATE_LIMIT_PER_MINUTE = 60


def create_gateway_app(
    dsn: str,
    *,
    audit_db_path: str | None = None,
    approvals_db_path: str | None = None,
    rate_limit_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE,
    approval_ttl_seconds: int = DEFAULT_TTL_SECONDS,
    clock: Callable[[], datetime] | None = None,
) -> FastAPI:
    """Build one governed-API `FastAPI` app around the database named by `dsn`.

    Args:
        dsn: Passed straight to `connectors.connector_for` — a SQLite path/URI,
            `postgres://...`, or `mysql://...` (see that module for exact DSN forms and
            validation).
        audit_db_path: Where the `AuditLog` persists. Defaults to `<sqlite path>.audit
            .sqlite` for a SQLite `dsn` (so a governed SQLite database and its audit trail
            live side by side, never inside the governed database's own file/schema), or
            an isolated in-memory log for a network backend (Postgres/MySQL) where there
            is no local filesystem path to derive one from.
        approvals_db_path: Same derivation as `audit_db_path`, for the `ApprovalQueue`.
        rate_limit_per_minute: Per-actor cap — see `gateway_routes._check_rate_limit`.
        approval_ttl_seconds: Default TTL for a newly enqueued approval.
        clock: Optional zero-arg callable returning a `datetime`, forwarded to both the
            `AuditLog` and the `ApprovalQueue` for deterministic tests — same convention
            those two classes already use individually.
    """
    connection = connector_for(dsn)

    if connection.backend == "sqlite":
        sqlite_path = connection.read_executor.db_path
        default_audit_path = f"{sqlite_path}.audit.sqlite" if sqlite_path != ":memory:" else ":memory:"
        default_approvals_path = f"{sqlite_path}.approvals.sqlite" if sqlite_path != ":memory:" else ":memory:"
    else:
        # No local filesystem path to derive one from for a network-backend DSN; an
        # isolated in-memory log/queue per app instance is the honest default here.
        default_audit_path = ":memory:"
        default_approvals_path = ":memory:"

    audit_log = AuditLog(audit_db_path or default_audit_path, clock=clock)
    approval_queue = ApprovalQueue(
        approvals_db_path or default_approvals_path, clock=clock, ttl_seconds=approval_ttl_seconds
    )
    gate = GovernanceGate()

    app = FastAPI(
        title="IcBerg Governance Gateway",
        description="AI<->database governance gateway: policy decision, least-privilege "
        "execution, PII redaction, human-approval workflow, and a tamper-evident audit "
        "trail, exposed as a REST + SSE API.",
        version="0.2.0",
    )

    app.state.connection = connection
    app.state.gate = gate
    app.state.gateway = Gateway(gate)
    app.state.audit_log = audit_log
    app.state.approval_queue = approval_queue
    app.state.rate_buckets = defaultdict(list)
    app.state.rate_limit_per_minute = rate_limit_per_minute
    app.state.metrics = {
        "queries_total": 0,
        "blocks_total": 0,
        "holds_total": 0,
        "approvals_total": 0,
        "rejections_total": 0,
        "rate_limited_total": 0,
    }

    app.include_router(gateway_router)

    @app.exception_handler(Exception)
    async def _scrub_unhandled_errors(request: Request, exc: Exception) -> JSONResponse:
        """Last-resort catch-all for anything a route handler raises without itself
        turning into a clean 4xx (`HTTPException`/`ApprovalError` already are —
        FastAPI's own built-in `HTTPException` handler takes precedence over this one and
        is untouched). P2.18 (`.devdocs/PHASE2_GATES.md`): the response body here is
        ALWAYS the fixed `_GENERIC_ERROR_DETAIL` string, never anything derived from
        `exc` itself — no traceback, no internal file path, no raw `proposed_sql`, no PII,
        regardless of what the underlying exception's message happens to contain. The
        real exception is logged (type name only, not its message, out of the same
        excess caution) for server-side operators, but that log line never reaches the
        caller.
        """
        logger.error(
            "gateway_api.unhandled_error",
            path=request.url.path,
            method=request.method,
            error_type=type(exc).__name__,
        )
        return JSONResponse(status_code=500, content={"detail": _GENERIC_ERROR_DETAIL})

    return app


def __getattr__(name: str) -> FastAPI:
    """PEP 562 module `__getattr__`: builds the module-level `app` object (for `uvicorn
    backend.gateway_app:app`) lazily, on first access, rather than as an import-time side
    effect — so `from backend.gateway_app import create_gateway_app` (what every test in
    this repo does) never touches the filesystem or builds an app nobody asked for.
    Configured entirely from environment variables, with a temp-directory default so an
    ad hoc `uvicorn` smoke-test run never writes into the repo itself.
    """
    if name != "app":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    default_db_path = os.path.join(tempfile.gettempdir(), "icberg_gateway.sqlite")
    db_path = os.environ.get("ICBERG_GATEWAY_DB_PATH", default_db_path)
    rate_limit = int(os.environ.get("ICBERG_GATEWAY_RATE_LIMIT_PER_MIN", str(DEFAULT_RATE_LIMIT_PER_MINUTE)))
    built = create_gateway_app(db_path, rate_limit_per_minute=rate_limit)
    globals()["app"] = built
    return built
