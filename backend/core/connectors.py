"""Multi-database connector factory — the Phase 2 "Connectors: Postgres + SQLite solid;
MySQL added" deliverable (`.devdocs/FLAGSHIP_ROADMAP.md`).

`connector_for(dsn)` is the single entry point: it parses/validates a DSN string and
returns a uniform `Connection` — a `read_executor` (least-privilege, see `executor.py`)
and a `write_executor` (the approved-write path `approvals.py` drives), plus best-effort
schema introspection, regardless of which backend the DSN names. Nothing above this
factory (the gateway API, `approvals.py`) needs to know or care which database engine it
is actually talking to.

Backend coverage, matching the gates' honesty requirement:
  - **SQLite**: real, fully wired — reuses `executor.ReadOnlyExecutor`/`WriteExecutor`
    (SQLite's engine-level `mode=ro` boundary, process-isolated hard timeout, forced row
    cap — see `executor.py`'s module docstring) and `schema_catalog.introspect_sqlite_schema`.
    Every SQLite path in this module is exercised by `tests/api/test_gateway_api.py`
    against a throwaway file with no external infra.
  - **Postgres**: reuses `executor.PostgresReadOnlyExecutor` for reads; `PostgresWriteExecutor`
    (this module) mirrors it for the approved-write path. `psycopg` is imported lazily,
    inside the method that needs it, so its absence never breaks importing this module or
    collecting the SQLite-backed test suite.
  - **MySQL**: `MySQLReadOnlyExecutor`/`MySQLWriteExecutor` (this module), `pymysql`
    imported lazily on the same principle. MySQL has no equivalent of SQLite's `mode=ro`
    file-level open for a network connection — `SET SESSION TRANSACTION READ ONLY` is the
    best available boundary, and it is a mutable session flag, not an engine-enforced one
    (the same honest caveat `executor.py` already gives `PRAGMA query_only`). The
    least-privilege boundary that actually matters for MySQL, same as for Postgres, is a
    read-only GRANT on the connecting role — see THREAT_MODEL.md's residual-risk section.

**Both network backends are constructible with no live server at all** — building a
`Connection` never opens a socket; only calling `.execute()` on its executors does, and
every executor in this module reports a connection failure as an `ExecutionResult.error`,
never an exception. This is what lets the factory itself (DSN parsing/validation, backend
routing, interface shape) be unit-tested unconditionally with SQLite plus construction-only
Postgres/MySQL cases, while the actual *live* round-trip against a real Postgres/MySQL
server is skipped unless `ICBERG_TEST_PG_DSN`/`ICBERG_TEST_MYSQL_DSN` is set — see
`tests/api/test_gateway_api.py`.

**DSN validation** is deliberately strict and fails closed: an empty/non-string DSN, an
unrecognized scheme, or a network DSN (`postgres://`/`mysql://`) with no host at all is
rejected outright by `connector_for` with a `ConnectorError` before any connection is even
attempted — "no arbitrary host access beyond the DSN" means this factory only ever
connects to what the DSN itself names, nothing it infers or falls back to. Any DSN
embedded in a log line or an exception message is passed through `_scrub_dsn` first, which
masks a `user:password@` userinfo component — a malformed or rejected DSN must not leak a
credential into a log file or an API error string, mirroring `redaction.py`'s "never echo
a secret back" posture for the rest of this codebase.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

import structlog

from backend.core.executor import (
    MAX_ROWS,
    TIMEOUT_SECONDS,
    ExecutionResult,
    PostgresReadOnlyExecutor,
    ReadOnlyExecutor,
    WriteExecutor,
)
from backend.core.schema_catalog import SchemaCatalog

logger = structlog.get_logger(__name__)

Backend = Literal["sqlite", "postgres", "mysql"]

_SQLITE_SCHEMES = frozenset({"sqlite", "file"})
_POSTGRES_SCHEMES = frozenset({"postgres", "postgresql"})
_MYSQL_SCHEMES = frozenset({"mysql"})


class ConnectorError(ValueError):
    """Raised by `connector_for` for any DSN this module refuses to route at all — an
    empty/non-string DSN, an unrecognized scheme, or a network DSN with no host. Never
    raised by an executor's own `.execute()` (those report failure via
    `ExecutionResult.error` instead, per `executor.py`'s established contract) — this
    exception is specifically about the DSN being unusable *before* any connection is
    attempted.
    """


# `scheme://user:password@host/...` — matched non-greedily so this only ever touches the
# userinfo component immediately after `//`, never anything later in the DSN (a path or
# query string that happens to contain `:`/`@` is untouched).
_DSN_USERINFO_RE = re.compile(r"//([^/@:\s]+):([^/@\s]+)@")


def _scrub_dsn(dsn: str) -> str:
    """Mask a DSN's password before it ever reaches a log line or an exception message —
    see module docstring. A DSN with no `user:password@` userinfo component (e.g. a bare
    SQLite path) is returned unchanged.
    """
    return _DSN_USERINFO_RE.sub(lambda m: f"//{m.group(1)}:***@", dsn)


@dataclass
class Connection:
    """A uniform handle to one database, regardless of backend — what `connector_for`
    returns and everything downstream (the gateway API, `approvals.py`) actually uses.

    Attributes:
        backend: Which engine this connection targets — informational (health checks,
            logging), never branched on by callers above this module.
        dsn: The original DSN string this connection was built from (as given — NOT
            scrubbed; callers that log/display this must scrub it themselves, the same
            way `redact_text` is applied at the API boundary, not deep in core plumbing).
        read_executor: Least-privilege read path — `.execute(sql) -> ExecutionResult`,
            optionally `.get_schema_catalog() -> SchemaCatalog | None`. MUST be used for
            every `allow` decision the gateway executes; see `executor.py`.
        write_executor: The approved-write path — `.execute(sql) -> ExecutionResult`.
            MUST be used only for a human-approved `hold` (`approvals.py`'s `approve`),
            never for the gateway's own `allow` path.
    """

    backend: Backend
    dsn: str
    read_executor: Any
    write_executor: Any

    def get_schema_catalog(self) -> SchemaCatalog | None:
        """Best-effort schema introspection via `read_executor`, if it offers any (today:
        SQLite only — see `executor.ReadOnlyExecutor.get_schema_catalog`). Mirrors
        `gateway.py`'s own `_introspect_schema`: an executor with no such method, or one
        that raises, degrades to `None` rather than ever breaking a caller.
        """
        get = getattr(self.read_executor, "get_schema_catalog", None)
        if get is None:
            return None
        try:
            return get()
        except Exception as exc:  # noqa: BLE001 - schema introspection must never break a caller
            logger.warning("connectors.schema_introspection_failed", backend=self.backend, error=str(exc))
            return None


def _parse_sqlite_path(dsn: str) -> str:
    """Resolve a SQLite DSN/path to the plain filesystem path (or `:memory:`/a `file:`
    URI) `ReadOnlyExecutor`/`WriteExecutor` expect. Accepts, in order:

      - `:memory:` — passed through unchanged.
      - `file:...` — already a SQLite URI (e.g. one a caller pre-built with
        `?mode=ro`); passed through unchanged rather than re-wrapped.
      - `sqlite://<rest>` — `<rest>` is used as the path verbatim (so both
        `sqlite:///abs/path.db` -> `/abs/path.db` and `sqlite://rel/path.db` ->
        `rel/path.db` resolve the way their leading-slash count implies); an empty
        `<rest>` (`sqlite://`) resolves to `:memory:`.
      - anything else — treated as a bare filesystem path, unchanged.
    """
    if dsn == ":memory:":
        return dsn
    if dsn.startswith("file:"):
        return dsn
    if dsn.startswith("sqlite://"):
        rest = dsn[len("sqlite://"):]
        return rest or ":memory:"
    return dsn


def _require_host(dsn: str, scheme: str) -> None:
    """Reject a network-backend DSN with no host at all (junk, e.g. `postgres://`) — see
    module docstring's "no arbitrary host access beyond the DSN": this factory only ever
    connects to a host the DSN itself names, never a default/fallback host.
    """
    hostname = urlsplit(dsn).hostname
    if not hostname:
        raise ConnectorError(f"{scheme} DSN is missing a host: {_scrub_dsn(dsn)}")


def connector_for(
    dsn: str,
    *,
    max_rows: int = MAX_ROWS,
    timeout_seconds: float = TIMEOUT_SECONDS,
) -> Connection:
    """Parse `dsn` and return a uniform `Connection` for whichever backend it names.

    Never opens a connection itself (see module docstring) — this only decides which
    executor classes to construct and validates the DSN is well-formed enough to attempt
    that with. Raises `ConnectorError` (fails closed, before any connection attempt) for
    an empty/non-string DSN, an unrecognized scheme, or a `postgres`/`mysql` DSN with no
    host.
    """
    if not isinstance(dsn, str) or not dsn.strip():
        raise ConnectorError("DSN must be a non-empty string")
    dsn = dsn.strip()

    scheme = urlsplit(dsn).scheme.lower() if "://" in dsn else ""

    if not scheme or scheme in _SQLITE_SCHEMES:
        path = _parse_sqlite_path(dsn)
        return Connection(
            backend="sqlite",
            dsn=dsn,
            read_executor=ReadOnlyExecutor(path, max_rows=max_rows, timeout_seconds=timeout_seconds),
            write_executor=WriteExecutor(path, max_rows=max_rows, timeout_seconds=timeout_seconds),
        )

    if scheme in _POSTGRES_SCHEMES:
        _require_host(dsn, scheme)
        return Connection(
            backend="postgres",
            dsn=dsn,
            read_executor=PostgresReadOnlyExecutor(dsn, max_rows=max_rows, timeout_seconds=timeout_seconds),
            write_executor=PostgresWriteExecutor(dsn, max_rows=max_rows, timeout_seconds=timeout_seconds),
        )

    if scheme in _MYSQL_SCHEMES:
        _require_host(dsn, scheme)
        return Connection(
            backend="mysql",
            dsn=dsn,
            read_executor=MySQLReadOnlyExecutor(dsn, max_rows=max_rows, timeout_seconds=timeout_seconds),
            write_executor=MySQLWriteExecutor(dsn, max_rows=max_rows, timeout_seconds=timeout_seconds),
        )

    raise ConnectorError(f"unsupported DSN scheme {scheme!r}: {_scrub_dsn(dsn)}")


class PostgresWriteExecutor:
    """Write-capable Postgres executor for the approved-write path (`approvals.py`'s
    `approve`) — mirrors `executor.WriteExecutor`'s SQLite role and
    `executor.PostgresReadOnlyExecutor`'s lazy-`psycopg`-import pattern (see that class's
    docstring for why the import must stay lazy and inside the method that needs it, not
    at module scope). Untested against a live server in this repo unless
    `ICBERG_TEST_PG_DSN` is set — see `tests/api/test_gateway_api.py`.
    """

    # `Gateway.handle`'s read-path guard (P2.19) rejects any executor not explicitly
    # marked `IS_READONLY = True` — this class is write-capable, so it must never pass.
    IS_READONLY: bool = False

    def __init__(self, dsn: str, *, max_rows: int = MAX_ROWS, timeout_seconds: float = TIMEOUT_SECONDS) -> None:
        self.dsn = dsn
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> ExecutionResult:
        """Execute one write statement against Postgres and commit on success. Never raises."""
        start = time.monotonic()
        try:
            import psycopg  # lazy import: keep psycopg an optional dependency
        except ImportError as exc:
            return ExecutionResult(error=f"psycopg not installed: {exc}")
        try:
            conn = psycopg.connect(self.dsn)
        except Exception as exc:
            return ExecutionResult(error=f"failed to open Postgres write connection: {exc}")
        try:
            conn.execute(f"SET statement_timeout = {int(self.timeout_seconds * 1000)}")
            with conn.cursor() as cur:
                cur.execute(sql, params or None)
                columns = [d.name for d in cur.description] if cur.description else []
                rows: list[dict[str, Any]] = []
                truncated = False
                if cur.description:
                    for record in cur:
                        if len(rows) >= self.max_rows:
                            truncated = True
                            break
                        rows.append(dict(zip(columns, record)))
                conn.commit()
                return ExecutionResult(
                    rows=rows,
                    columns=columns,
                    rowcount=len(rows),
                    truncated=truncated,
                    latency_ms=int((time.monotonic() - start) * 1000),
                    error=None,
                )
        except Exception as exc:
            conn.rollback()
            return ExecutionResult(error=str(exc), latency_ms=int((time.monotonic() - start) * 1000))
        finally:
            conn.close()


def _mysql_connect_kwargs(dsn: str) -> dict[str, Any]:
    """Translate a `mysql://user:pass@host:port/db` DSN into `pymysql.connect(...)`
    keyword arguments. Raises `ConnectorError` if `dsn` has no host — should not normally
    happen here since `connector_for` already calls `_require_host` before constructing
    either MySQL executor, but this is called again defensively since these executors
    could in principle be constructed directly, bypassing the factory.
    """
    parts = urlsplit(dsn)
    if not parts.hostname:
        raise ConnectorError(f"mysql DSN is missing a host: {_scrub_dsn(dsn)}")
    return {
        "host": parts.hostname,
        "port": parts.port or 3306,
        "user": parts.username or "",
        "password": parts.password or "",
        "database": parts.path.lstrip("/") or None,
    }


class MySQLReadOnlyExecutor:
    """Best-effort read-only MySQL executor. `SET SESSION TRANSACTION READ ONLY` is a
    SESSION-level flag, not an engine-enforced boundary the way SQLite's `mode=ro`
    file-open is — MySQL has no equivalent of opening a read-only file descriptor for a
    network connection, so this is the best available layer short of a database-level
    read-only GRANT on the connecting role (the actual least-privilege boundary — see
    THREAT_MODEL.md's residual-risk section, and `executor.py`'s own honest caveat about
    `PRAGMA query_only` being secondary-only for the same reason). The read-only
    connection also never commits (always rolls back, success or failure) as one further,
    independent layer against an accidental write slipping through despite the session
    flag. Untested against a live server in this repo unless `ICBERG_TEST_MYSQL_DSN` is
    set — see `tests/api/test_gateway_api.py`.
    """

    # `Gateway.handle` (P2.19, `.devdocs/PHASE2_GATES.md`) asserts this before ever using
    # an executor on the read path — see `executor.ReadOnlyExecutor`'s identical marker.
    IS_READONLY: bool = True

    def __init__(self, dsn: str, *, max_rows: int = MAX_ROWS, timeout_seconds: float = TIMEOUT_SECONDS) -> None:
        self.dsn = dsn
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds

    def get_schema_catalog(self) -> SchemaCatalog | None:
        """Not yet implemented — see module docstring. Returns `None` so a caller falls
        back to schema-less redaction, mirroring `PostgresReadOnlyExecutor`'s own stub.
        """
        return None

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> ExecutionResult:
        """Execute one read-only statement against MySQL. Never raises."""
        start = time.monotonic()
        try:
            import pymysql  # lazy import: keep pymysql an optional dependency
        except ImportError as exc:
            return ExecutionResult(error=f"pymysql not installed: {exc}")
        try:
            kwargs = _mysql_connect_kwargs(self.dsn)
        except ConnectorError as exc:
            return ExecutionResult(error=str(exc))
        try:
            conn = pymysql.connect(
                **kwargs,
                cursorclass=pymysql.cursors.Cursor,
                connect_timeout=max(1, int(self.timeout_seconds)),
            )
        except Exception as exc:
            return ExecutionResult(error=f"failed to open MySQL read-only connection: {exc}")
        try:
            with conn.cursor() as cur:
                cur.execute("SET SESSION TRANSACTION READ ONLY")
                cur.execute(f"SET SESSION MAX_EXECUTION_TIME = {int(self.timeout_seconds * 1000)}")
                cur.execute(sql, params or None)
                columns = [d[0] for d in cur.description] if cur.description else []
                all_rows = cur.fetchall() if cur.description else []
                truncated = len(all_rows) > self.max_rows
                rows = [dict(zip(columns, r)) for r in all_rows[: self.max_rows]]
                conn.rollback()  # never commit on the read-only connection — defense in depth
                return ExecutionResult(
                    rows=rows,
                    columns=columns,
                    rowcount=len(rows),
                    truncated=truncated,
                    latency_ms=int((time.monotonic() - start) * 1000),
                    error=None,
                )
        except Exception as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            return ExecutionResult(
                rows=[], columns=[], rowcount=0, truncated=False, latency_ms=latency_ms, error=str(exc)
            )
        finally:
            conn.close()


class MySQLWriteExecutor:
    """Write-capable MySQL executor for the approved-write path (`approvals.py`'s
    `approve`). Untested against a live server in this repo unless
    `ICBERG_TEST_MYSQL_DSN` is set — see `tests/api/test_gateway_api.py`.
    """

    # See `PostgresWriteExecutor.IS_READONLY` — same P2.19 rationale.
    IS_READONLY: bool = False

    def __init__(self, dsn: str, *, max_rows: int = MAX_ROWS, timeout_seconds: float = TIMEOUT_SECONDS) -> None:
        self.dsn = dsn
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> ExecutionResult:
        """Execute one write statement against MySQL and commit on success. Never raises."""
        start = time.monotonic()
        try:
            import pymysql  # lazy import: keep pymysql an optional dependency
        except ImportError as exc:
            return ExecutionResult(error=f"pymysql not installed: {exc}")
        try:
            kwargs = _mysql_connect_kwargs(self.dsn)
        except ConnectorError as exc:
            return ExecutionResult(error=str(exc))
        try:
            conn = pymysql.connect(
                **kwargs,
                cursorclass=pymysql.cursors.Cursor,
                connect_timeout=max(1, int(self.timeout_seconds)),
            )
        except Exception as exc:
            return ExecutionResult(error=f"failed to open MySQL write connection: {exc}")
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params or None)
                columns = [d[0] for d in cur.description] if cur.description else []
                rows: list[dict[str, Any]] = []
                truncated = False
                if cur.description:
                    all_rows = cur.fetchall()
                    truncated = len(all_rows) > self.max_rows
                    rows = [dict(zip(columns, r)) for r in all_rows[: self.max_rows]]
                conn.commit()
                return ExecutionResult(
                    rows=rows,
                    columns=columns,
                    rowcount=len(rows) if columns else cur.rowcount,
                    truncated=truncated,
                    latency_ms=int((time.monotonic() - start) * 1000),
                    error=None,
                )
        except Exception as exc:
            conn.rollback()
            latency_ms = int((time.monotonic() - start) * 1000)
            return ExecutionResult(error=str(exc), latency_ms=latency_ms)
        finally:
            conn.close()
