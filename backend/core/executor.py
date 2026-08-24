"""Least-privilege SQL execution — the trusted execution boundary behind the gateway.

`sql_governance.py` only *decides*; it never opens a connection. This module is where a
decision an operator has actually approved for execution (`action == "allow"`) touches a
real database, and it is built so that a bypass of the policy gate — a bug, a future
regression, or a caller that skips the gate entirely — still cannot mutate data through
the read path.

The read boundary is engine-level, not just a session flag: `ReadOnlyExecutor` opens the
SQLite connection with the `file:<path>?mode=ro` URI, which SQLite enforces at the OS file
level (the file descriptor itself is opened read-only). `PRAGMA query_only = ON` is also
set as a secondary, defense-in-depth layer, but it is **not** the real boundary — it is a
mutable session-level setting a statement could in principle toggle back off, whereas
`mode=ro` cannot be undone by anything a query can execute. If a write is forced through
anyway, SQLite raises `sqlite3.OperationalError: attempt to write a readonly database`
before any row is touched; this module never catches that and retries a mutating path —
it is captured as `ExecutionResult.error` and the caller gets nothing else.

A third layer denies `ATTACH`/`DETACH DATABASE` at the connection itself: `sql_governance.py`
blocks it by rule (`attach_blocked`) before it is ever proposed for execution, but a bypass
of that string/AST-level gate must not be allowed to open a second, non-read-only database
file under this connection and write through it — see `_deny_attach` and `_connect` below.
`conn.set_authorizer(...)` returns `SQLITE_DENY` for `SQLITE_ATTACH`/`SQLITE_DETACH`
regardless of what the gate decided, so this is real defense-in-depth, not a restatement of
the gate's own rule.

Two further resource limits apply to every read, independent of whatever `LIMIT` (or lack
of one) the proposed SQL contains:
  - a forced row cap (`max_rows`, default `MAX_ROWS` = 1000): fetching stops the instant
    the cap is reached and `truncated=True` is reported, regardless of how many more rows
    the underlying result set actually has;
  - a wall-clock timeout (`timeout_seconds`, default 5s) enforced by **two** independent
    layers, not one. The first is an in-process watchdog thread that calls
    `Connection.interrupt()` — the fast path, and usually all that's needed, since
    `interrupt()` is SQLite's own designed-for-this mechanism to abort a query already
    blocked inside the VDBE (e.g. deep inside a recursive CTE). But a Python thread
    calling `interrupt()` is itself a race: nothing stops from that call landing too late
    for a query that isn't yielding control back to the interpreter often enough, or
    that's blocked on something `interrupt()` doesn't reach at all, and a Python thread
    that is stuck cannot be forcibly killed — CPython has no API for that. So
    `ReadOnlyExecutor.execute` runs the whole connect-and-query attempt (including its own
    internal interrupt watchdog) inside a **separate, forked child process**, the same
    process-isolation pattern `sandbox.py` uses for untrusted code execution, and
    `multiprocessing.Process.join(timeout=...)` on it. If the child hasn't reported back
    within `timeout_seconds` plus a small grace period, the parent kills the process
    outright (`SIGKILL`) — an OS-level guarantee no in-process blocking call, however
    pathological, can defeat. This also means a segfault or an unhandled exception in the
    query path can never propagate into or crash the caller's process; it is reported as an
    ordinary `ExecutionResult.error` either way.

A separate `WriteExecutor` (and the `Executor()` factory) exists for the *approved-write*
path Phase 2 adds (the human-approval queue in `sql_governance.py`'s `hold` decisions) —
but the read path in this module, and everywhere the gateway executes an `allow` decision,
MUST use `ReadOnlyExecutor`. A `PostgresReadOnlyExecutor` is included for the eventual
Postgres backend described in THREAT_MODEL.md (read-only role + `SET TRANSACTION READ
ONLY` + `SET statement_timeout`); it imports `psycopg` lazily, inside the method that
needs it, so its absence never breaks importing this module or running the SQLite tests
that are Phase 1's actual target.
"""

from __future__ import annotations

import multiprocessing
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from backend.core.schema_catalog import SchemaCatalog, introspect_sqlite_schema

logger = structlog.get_logger(__name__)

# Forced row cap applied to every read, independent of any LIMIT in the proposed SQL.
MAX_ROWS = 1000
# Wall-clock timeout (seconds) applied to every read via a watchdog thread + interrupt().
TIMEOUT_SECONDS = 5.0
# Extra headroom `ReadOnlyExecutor` gives the process-isolated worker beyond its own
# `timeout_seconds` before the parent gives up waiting and force-kills it. See
# `ReadOnlyExecutor.execute`'s docstring for why this exists on top of the in-process
# interrupt() watchdog, not instead of it.
_PROCESS_JOIN_GRACE_SECONDS = 1.0


@dataclass
class ExecutionResult:
    """Outcome of executing one statement through an executor.

    Attributes:
        rows: Result rows as plain dicts (column name -> value), capped at `max_rows`.
        columns: Column names, in result order (empty for a statement with no result set).
        rowcount: Number of rows actually returned in `rows` (== len(rows)).
        truncated: True if more rows existed beyond the forced cap.
        latency_ms: Wall-clock execution time in milliseconds.
        error: Engine-raised error text, or None on success. Never raised as an exception
            to the caller — a rejected write, a timeout, and a syntax error are all
            reported this way so the gateway can audit them uniformly.
    """
    rows: list[dict[str, Any]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    rowcount: int = 0
    truncated: bool = False
    latency_ms: int = 0
    error: str | None = None


def _deny_attach(action_code: int, arg1: str | None, arg2: str | None, dbname: str | None, source: str | None) -> int:
    """SQLite authorizer callback: deny `ATTACH`/`DETACH DATABASE` unconditionally,
    regardless of what `sql_governance.py`'s gate decided upstream. Every other action
    code is allowed — this is not a general allow-list authorizer, only a targeted deny
    for the one action family that can open a second, non-read-only database file under
    an already-open connection (see module docstring). `mode=ro` and `PRAGMA query_only`
    do not, on their own, stop this: an attached database is a brand-new file open with
    its own (default read-write) mode.
    """
    if action_code in (sqlite3.SQLITE_ATTACH, getattr(sqlite3, "SQLITE_DETACH", -1)):
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _run_with_watchdog(
    conn: sqlite3.Connection,
    sql: str,
    params: tuple[Any, ...],
    max_rows: int,
    timeout_seconds: float,
) -> ExecutionResult:
    """Execute `sql` on `conn` under a row cap and a wall-clock timeout.

    Shared by `ReadOnlyExecutor` and `WriteExecutor` — the only difference between the two
    is how `conn` itself was opened (read-only file URI vs. a normal read-write handle) and
    whether a successful statement is committed afterward. This is the FAST-PATH timeout
    layer only (`Connection.interrupt()` from a watchdog thread); `ReadOnlyExecutor.execute`
    layers a hard, process-level kill on top of this — see the module docstring for why
    both layers exist rather than just one.
    """
    start = time.monotonic()
    timed_out = threading.Event()

    def _abort_on_timeout() -> None:
        timed_out.set()
        try:
            conn.interrupt()
        except Exception:  # pragma: no cover - interrupt() itself should not raise
            logger.warning("executor.interrupt_failed")

    timer = threading.Timer(timeout_seconds, _abort_on_timeout)
    timer.daemon = True
    timer.start()
    try:
        cursor = conn.execute(sql, params)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows: list[dict[str, Any]] = []
        truncated = False
        for row in cursor:
            if len(rows) >= max_rows:
                truncated = True
                break
            rows.append(dict(row))
        latency_ms = int((time.monotonic() - start) * 1000)
        return ExecutionResult(
            rows=rows,
            columns=columns,
            rowcount=len(rows),
            truncated=truncated,
            latency_ms=latency_ms,
            error=None,
        )
    except sqlite3.Error as exc:
        latency_ms = int((time.monotonic() - start) * 1000)
        if timed_out.is_set():
            error = f"query exceeded timeout of {timeout_seconds}s and was aborted: {exc}"
        else:
            error = str(exc)
        logger.warning("executor.query_failed", error=error, latency_ms=latency_ms)
        return ExecutionResult(rows=[], columns=[], rowcount=0, truncated=False, latency_ms=latency_ms, error=error)
    finally:
        timer.cancel()


def _connect_readonly(db_path: str) -> sqlite3.Connection:
    """Open `db_path` read-only at the OS file level, with `PRAGMA query_only` and the
    ATTACH/DETACH-denying authorizer layered on top. Module-level (not a method) so the
    process-isolated worker below can call it without pickling a bound method.
    """
    # `mode=ro` opens the underlying file descriptor read-only at the OS level — this is
    # what actually stops a write, not a session flag a statement could revert.
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Secondary, defense-in-depth layer only (see module docstring) — never relied on
    # alone.
    conn.execute("PRAGMA query_only = ON")
    # Connection-level ATTACH/DETACH deny — see module docstring and `_deny_attach`.
    conn.set_authorizer(_deny_attach)
    return conn


def _readonly_worker(
    db_path: str,
    sql: str,
    params: tuple[Any, ...],
    max_rows: int,
    timeout_seconds: float,
    result_queue: "multiprocessing.Queue[dict[str, Any]]",
) -> None:
    """Runs inside the forked child process `ReadOnlyExecutor.execute` spawns: opens its
    own read-only connection, executes under the in-process interrupt watchdog, and puts
    a plain-dict result on `result_queue`. Wrapped in a broad `except Exception` so that
    literally anything going wrong here — a bug in this module, an unexpected engine
    error type, anything — is reported back as an `ExecutionResult.error` in the parent,
    never a silently-dead child the parent has to guess about.
    """
    start = time.monotonic()
    try:
        conn = _connect_readonly(db_path)
    except Exception as exc:
        result_queue.put({
            "rows": [], "columns": [], "rowcount": 0, "truncated": False,
            "latency_ms": int((time.monotonic() - start) * 1000),
            "error": f"failed to open read-only connection: {exc}",
        })
        return
    try:
        result = _run_with_watchdog(conn, sql, params, max_rows, timeout_seconds)
        result_queue.put({
            "rows": result.rows, "columns": result.columns, "rowcount": result.rowcount,
            "truncated": result.truncated, "latency_ms": result.latency_ms, "error": result.error,
        })
    except Exception as exc:  # never let the child die without reporting something back
        result_queue.put({
            "rows": [], "columns": [], "rowcount": 0, "truncated": False,
            "latency_ms": int((time.monotonic() - start) * 1000),
            "error": f"unexpected executor error: {exc}",
        })
    finally:
        conn.close()


class ReadOnlyExecutor:
    """Executes SQL against a SQLite database opened read-only at the file level, inside a
    process-isolated, hard-timed worker.

    Use this — and only this — for the gateway's `allow` (read) execution path. See the
    module docstring for why `mode=ro` is the real boundary and `PRAGMA query_only` is
    only a secondary layer, and for why execution runs in a forked child process rather
    than purely in-process.
    """

    # `Gateway.handle` (P2.19, `.devdocs/PHASE2_GATES.md`) asserts this attribute is
    # `True` before ever using an executor on the read path — a positive, explicit
    # self-declaration rather than an isinstance/name check, so a caller wiring in a
    # write-capable executor by mistake fails loudly instead of being trusted.
    IS_READONLY: bool = True

    def __init__(
        self,
        db_path: str,
        *,
        max_rows: int = MAX_ROWS,
        timeout_seconds: float = TIMEOUT_SECONDS,
    ) -> None:
        self.db_path = db_path
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds

    def _connect(self) -> sqlite3.Connection:
        """Kept for callers that want a raw connection with the same read-only + deny-
        ATTACH setup this executor uses internally (e.g. tests exercising the authorizer
        directly). `execute()` itself does not call this — it opens its own connection
        inside the isolated child process instead.
        """
        return _connect_readonly(self.db_path)

    def get_schema_catalog(self) -> SchemaCatalog | None:
        """Best-effort live schema for this executor's own database — see
        `schema_catalog.introspect_sqlite_schema`. `Gateway.handle` calls this (via
        `getattr`, so any executor that doesn't define it just means schema-less
        redaction) before redacting an `allow`ed query's results, so `redact_rows`'s
        provenance layer can resolve views/JOINs against the database's real schema
        instead of falling back to unknown-schema heuristics.

        Returns `None` on any introspection failure — never raises — so a caller can
        treat "no schema" and "introspection failed" identically. Re-introspects on
        every call rather than caching: the cost is a handful of cheap
        `PRAGMA`/`sqlite_master` queries against a small catalog (not a table scan),
        and a long-lived gateway process must not redact against a stale view/table
        definition if the schema legitimately changed between two proposed queries.
        """
        return introspect_sqlite_schema(self.db_path)

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> ExecutionResult:
        """Execute one read-only statement and return its (capped, time-boxed) result.

        Never raises, and the timeout is a hard wall-clock cap: the query runs inside a
        forked child process, and if it hasn't reported back within `timeout_seconds`
        plus a small grace period, the child is killed outright (`SIGKILL`) and this
        returns a timeout `ExecutionResult` — no in-process blocking call, watchdog race,
        or child crash can turn into an unhandled exception here or hang the caller past
        that hard cap. See the module docstring for the two-layer (interrupt() + process
        kill) design.
        """
        start = time.monotonic()
        result_queue: "multiprocessing.Queue[dict[str, Any]]" = multiprocessing.Queue()
        worker = multiprocessing.Process(
            target=_readonly_worker,
            args=(self.db_path, sql, params, self.max_rows, self.timeout_seconds, result_queue),
            daemon=True,
        )
        worker.start()
        worker.join(timeout=self.timeout_seconds + _PROCESS_JOIN_GRACE_SECONDS)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        if worker.is_alive():
            # Hard backstop: the in-process interrupt() watchdog should have aborted this
            # well before now (see `_run_with_watchdog`) — reaching here means it lost the
            # race against whatever the query was blocked on. Kill unconditionally.
            worker.kill()
            worker.join(timeout=2)
            logger.warning("executor.hard_timeout_kill", db_path=self.db_path, timeout_seconds=self.timeout_seconds)
            return ExecutionResult(
                error=f"query exceeded timeout of {self.timeout_seconds}s and was forcibly terminated",
                latency_ms=elapsed_ms,
            )

        if result_queue.empty():
            # Child exited without reporting anything back (OS-killed: OOM, segfault, ...).
            logger.warning("executor.worker_died_silently", db_path=self.db_path, exitcode=worker.exitcode)
            return ExecutionResult(
                error=f"query execution process exited unexpectedly (code {worker.exitcode})",
                latency_ms=elapsed_ms,
            )

        data = result_queue.get_nowait()
        return ExecutionResult(
            rows=data["rows"],
            columns=data["columns"],
            rowcount=data["rowcount"],
            truncated=data["truncated"],
            latency_ms=data["latency_ms"] or elapsed_ms,
            error=data["error"],
        )


class WriteExecutor:
    """Executes SQL with write privileges against a normal (read-write) SQLite handle.

    Reserved for Phase 2's approved-write execution flow (a `hold` decision that a human
    has approved). The gateway's `allow`/read path MUST use `ReadOnlyExecutor`, never this
    class — nothing here enforces least-privilege, by design, since a write is exactly
    what this path exists to perform once approved. `IS_READONLY = False` is what lets
    `Gateway.handle`'s read-path guard (P2.19) reject this class outright if it is ever
    handed to it by mistake.
    """

    IS_READONLY: bool = False

    def __init__(
        self,
        db_path: str,
        *,
        max_rows: int = MAX_ROWS,
        timeout_seconds: float = TIMEOUT_SECONDS,
    ) -> None:
        self.db_path = db_path
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> ExecutionResult:
        """Execute one statement with write privileges and commit on success."""
        try:
            conn = self._connect()
        except sqlite3.Error as exc:
            logger.warning("executor.connect_failed", db_path=self.db_path, error=str(exc))
            return ExecutionResult(error=f"failed to open read-write connection: {exc}")
        try:
            result = _run_with_watchdog(conn, sql, params, self.max_rows, self.timeout_seconds)
            if result.error is None:
                conn.commit()
            else:
                conn.rollback()
            return result
        finally:
            conn.close()


def Executor(
    db_path: str,
    *,
    readonly: bool = True,
    max_rows: int = MAX_ROWS,
    timeout_seconds: float = TIMEOUT_SECONDS,
) -> ReadOnlyExecutor | WriteExecutor:
    """Factory: `readonly=True` (default) returns a `ReadOnlyExecutor`; `readonly=False`
    returns a `WriteExecutor`. The gateway's read path must always request `readonly=True`
    (the default) — this factory exists so callers can spell the write path explicitly
    (`Executor(path, readonly=False)`) without importing `WriteExecutor` by name.
    """
    if readonly:
        return ReadOnlyExecutor(db_path, max_rows=max_rows, timeout_seconds=timeout_seconds)
    return WriteExecutor(db_path, max_rows=max_rows, timeout_seconds=timeout_seconds)


class PostgresReadOnlyExecutor:
    """Read-only Postgres backend: a read-only DB role + `SET TRANSACTION READ ONLY` +
    `SET statement_timeout`, layered the same way `ReadOnlyExecutor` layers SQLite's
    `mode=ro` (engine-enforced) under `PRAGMA query_only` (session-level, secondary).

    `psycopg` is imported lazily, inside `_connect`, specifically so that not having it
    installed never breaks importing this module or collecting/running the SQLite-backed
    Phase 1 test suite, which is the actual target for this phase. Untested by the Phase 1
    gates (no Postgres fixture in this repo yet) — included for the connector roadmap in
    `.devdocs/FLAGSHIP_ROADMAP.md` (Phase 2: "Connectors: Postgres + SQLite solid").
    """

    IS_READONLY: bool = True

    def __init__(
        self,
        dsn: str,
        *,
        max_rows: int = MAX_ROWS,
        timeout_seconds: float = TIMEOUT_SECONDS,
    ) -> None:
        self.dsn = dsn
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds

    def _connect(self) -> Any:
        import psycopg  # lazy import: keep psycopg an optional dependency

        conn = psycopg.connect(self.dsn)
        # The DSN's role should already be a least-privilege, non-superuser read-only
        # role; these two statements are the connection-level enforcement on top of that,
        # mirroring SQLite's mode=ro (engine-enforced) + query_only (secondary) layering.
        conn.execute("SET TRANSACTION READ ONLY")
        conn.execute(f"SET statement_timeout = {int(self.timeout_seconds * 1000)}")
        return conn

    def get_schema_catalog(self) -> SchemaCatalog | None:
        """Not yet implemented (see module and `schema_catalog.py` docstrings) —
        returns `None` so `Gateway.handle` falls back to schema-less redaction rather
        than a Postgres-specific code path breaking. A real implementation would query
        `information_schema.tables`/`.columns` for `SchemaCatalog.tables` and
        `information_schema.views`/`pg_views` for `SchemaCatalog.views`, mirroring
        `introspect_sqlite_schema`'s shape exactly — `redaction.py`'s provenance layer
        is already backend-agnostic and needs no change once this is filled in.
        """
        return None

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> ExecutionResult:
        """Execute one read-only statement against Postgres. Never raises."""
        start = time.monotonic()
        try:
            conn = self._connect()
        except Exception as exc:  # psycopg errors, or psycopg not installed
            return ExecutionResult(error=f"failed to open read-only Postgres connection: {exc}")
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params or None)
                columns = [d.name for d in cur.description] if cur.description else []
                rows: list[dict[str, Any]] = []
                truncated = False
                for record in cur:
                    if len(rows) >= self.max_rows:
                        truncated = True
                        break
                    rows.append(dict(zip(columns, record)))
                latency_ms = int((time.monotonic() - start) * 1000)
                return ExecutionResult(
                    rows=rows,
                    columns=columns,
                    rowcount=len(rows),
                    truncated=truncated,
                    latency_ms=latency_ms,
                    error=None,
                )
        except Exception as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            return ExecutionResult(
                rows=[], columns=[], rowcount=0, truncated=False, latency_ms=latency_ms, error=str(exc)
            )
        finally:
            conn.close()
