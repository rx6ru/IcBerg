"""Tamper-evident, hash-chained audit log — the auditability control (THREAT_MODEL.md
asset #5) that Phase 0 only promised ("every decision carries a `reason` and
`matched_rules`") and Phase 1 actually delivers.

Every decision the gateway makes — `allow`, `block`, or `hold`, never only the ones that
executed — is appended as an `AuditEntry`. Each entry's `entry_hash` is computed over a
canonical (sorted-key) JSON encoding of every other field, including `prev_hash`, which is
itself the previous entry's `entry_hash` (the genesis entry chains to `"0" * 64`). This
means mutating *any* stored field of *any* entry — including an old one — changes what
`entry_hash` recomputes to for that entry, which no longer matches the value stored
alongside it, and breaks the `prev_hash` linkage for every entry appended after it.
`verify()` walks the whole chain and reports the first sequence number where that check
fails, so an operator investigating an incident knows exactly where tampering started.

This module layers three independent controls on top of that hash chain, closing the gaps
an earlier version of this docstring (and THREAT_MODEL.md) called out honestly as not yet
done:

  1. **Append-only storage.** The `audit_log` table carries `BEFORE UPDATE`/`BEFORE
     DELETE` triggers that unconditionally `RAISE(ABORT, ...)` — a direct `UPDATE`/
     `DELETE` against the table (bypassing this class's own API, which never exposes
     either) fails at the storage layer itself, not just "isn't offered." This is still
     bypassable by an attacker with enough access to `DROP TRIGGER` first — SQLite has no
     true WORM mode — but it closes the common case of a bug or a lower-privileged
     compromise that can run ordinary DML but not DDL against the log.
  2. **External anchor.** After every `append()`, the new head — `{seq, entry_hash}` — is
     also written to a *separate* file (`self.anchor_path`, outside the SQLite table
     entirely). `verify()` checks the chain's own recomputed head against this anchor as
     its final step: an attacker who has rewritten every row's `entry_hash`/`prev_hash`
     to make the chain internally self-consistent again (defeating the per-entry checks
     alone) still leaves the *old* head sitting in the anchor file, which will not match
     the *new* (recomputed, doctored) head — `verify()` still returns `ok=False`. This is
     what makes the hash chain resistant to a full, self-consistent rewrite, not just a
     single-field edit.
  3. **PII-scrubbed `proposed_sql`, salted `result_hash`.** `proposed_sql` is passed
     through `redaction.redact_text` before it is stored, so a literal
     `WHERE email='alice@example.com'` never persists raw in the log at rest (the same
     scrubbing `gateway.py` already applies to the `reason` string it returns to a
     caller). `result_hash` is computed with a random, per-log salt (`self.salt`,
     generated once at construction and persisted in a metadata table) mixed into the
     digest — see `hash_result_rows`/`AuditLog.hash_result` — specifically so the stored
     hash is not a raw-value preimage: without the salt, a small/guessable result set
     (e.g. "is `admin` 0 or 1 for user 7") could be dictionary-attacked by hashing
     candidate row sets and comparing to the stored digest.

This is still tamper-*evident*, not tamper-*proof*: none of the above is a true WORM
store or an asymmetric signature binding entries to the gateway's identity, and an
attacker with full filesystem + SQLite access can defeat any single layer given enough
privilege (drop the trigger; rewrite the anchor file too; recover a salt stored alongside
the data it salts). Layering them raises the bar — a doctored chain now has to beat the
per-entry hash check *and* the anchor *and* the append-only trigger, not just one — but it
is not an unconditional guarantee. See THREAT_MODEL.md's Residual Risks for the honest
statement of what remains.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import structlog

from backend.core.redaction import redact_text

logger = structlog.get_logger(__name__)

GENESIS_HASH = "0" * 64


@dataclass
class AuditEntry:
    """One immutable record in the audit log.

    Attributes:
        seq: 1-based sequence number, strictly increasing, no gaps.
        timestamp: ISO-8601 timestamp string (from the log's injected clock).
        actor: Identifier of the proposing agent/user.
        proposed_sql: The exact SQL text that was evaluated (untrusted, verbatim).
        classification: The gate's classification ("read"/"write"/"ddl"/"unknown").
        action: The gate's decision ("allow"/"block"/"hold").
        matched_rules: Every policy rule name that fired, in evaluation order.
        rows_returned: Number of (redacted) rows actually returned to the caller (0 for
            block/hold, since nothing executes on those paths).
        latency_ms: Execution latency in milliseconds (0 for block/hold).
        result_hash: sha256 of the *redacted* result rows — see module docstring.
        prev_hash: The previous entry's `entry_hash` (or `GENESIS_HASH` for entry #1).
        entry_hash: sha256 of this entry's canonical JSON (all fields above) + `prev_hash`.
    """
    seq: int
    timestamp: str
    actor: str
    proposed_sql: str
    classification: str
    action: str
    matched_rules: list[str] = field(default_factory=list)
    rows_returned: int = 0
    latency_ms: int = 0
    result_hash: str = ""
    prev_hash: str = ""
    entry_hash: str = ""


def _canonical_json(data: dict[str, Any]) -> str:
    """Deterministic JSON encoding: sorted keys, no extraneous whitespace, so the same
    logical entry always hashes to the same bytes regardless of dict insertion order.
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def hash_result_rows(rows: list[dict[str, Any]], salt: str = "") -> str:
    """sha256 hex digest of `rows`' canonical JSON encoding, with `salt` mixed in.
    Callers MUST pass the *redacted* rows (never raw ones) — see module docstring.

    `salt` defaults to `""` (unsalted, matching this function's original behavior) for
    callers that just want a deterministic content hash with no `AuditLog` in the
    picture (e.g. computing `result_hash` before an entry exists to salt it with).
    `AuditLog.append`'s actual audit-trail usage goes through `AuditLog.hash_result`
    instead, which supplies the log's own per-instance salt — see the module docstring's
    "PII-scrubbed proposed_sql, salted result_hash" section for why that salt matters.
    """
    return hashlib.sha256((_canonical_json({"rows": rows}) + salt).encode("utf-8")).hexdigest()


def _compute_entry_hash(fields_without_entry_hash: dict[str, Any]) -> str:
    """`entry_hash = sha256(canonical_json(entry_without_entry_hash) + prev_hash)`."""
    payload = _canonical_json(fields_without_entry_hash)
    prev_hash = fields_without_entry_hash["prev_hash"]
    return hashlib.sha256((payload + prev_hash).encode("utf-8")).hexdigest()


class AuditLog:
    """Hash-chained, append-only audit log backed by a SQLite table.

    Args:
        db_path: SQLite path, or `":memory:"` (default) for an in-process log — the log's
            own `Connection` stays open for the object's lifetime, so `:memory:` persists
            across calls the way it needs to for a test or a single gateway process.
        clock: Optional zero-arg callable returning a `datetime`, injected so tests get
            deterministic, controllable timestamps instead of wall-clock time. Defaults to
            `datetime.now(timezone.utc)`.
        anchor_path: Path to the external anchor file `append()` writes `{seq,
            entry_hash}` to after every entry, and `verify()` checks the chain head
            against — see module docstring point 2. Defaults to `None`, meaning: derive
            `f"{db_path}.anchor.json"` for a file-backed `db_path`, or a fresh private
            temp file for `":memory:"` (each in-memory log is its own isolated instance
            with nothing to derive a stable path from). The anchor mechanism is always
            active — there is no way to construct an `AuditLog` without one — since an
            optional anchor an integrator forgets to pass would silently degrade back to
            "no external anchor at all."
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        *,
        clock: Callable[[], datetime] | None = None,
        anchor_path: str | None = None,
    ) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        # A per-instance lock guarding EVERY access to `self.conn` (and, in `append`,
        # the external anchor-file write in the same critical section) — see H1 in the
        # module docstring's threat-model history. `check_same_thread=False` below only
        # lifts sqlite3's own same-thread *ownership* check so this connection CAN be
        # handed to a different thread than the one that opened it (needed for a
        # sync FastAPI route, which Starlette runs in its threadpool); it does NOT make
        # concurrent use of the SAME connection object from multiple threads safe —
        # without this lock, two threads racing `append()` could both read the same
        # `prev_hash`/`seq` before either commits, corrupting the hash chain, or a bare
        # `sqlite3.Cursor` could be shared across threads and raise `InterfaceError`/
        # `ProgrammingError`. `RLock` (not `Lock`) so a locked method calling another
        # locked method on `self` from the same thread (e.g. `append` calling
        # `_next_seq`/`_last_entry_hash`, or `verify` calling `entries`) re-enters
        # cleanly instead of deadlocking on itself.
        self._lock = threading.RLock()
        # `check_same_thread=False`: the gateway may be used from a single-threaded test
        # or from an async/threaded server context; this log does no thread-affinity
        # tricks of its own, so it doesn't restrict the caller's threading model either
        # — thread-safety for concurrent callers is `self._lock`'s job, not this flag's.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                seq INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                actor TEXT NOT NULL,
                proposed_sql TEXT NOT NULL,
                classification TEXT NOT NULL,
                action TEXT NOT NULL,
                matched_rules TEXT NOT NULL,
                rows_returned INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                result_hash TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                entry_hash TEXT NOT NULL
            )
            """
        )
        # Append-only enforcement (module docstring point 1): any direct UPDATE/DELETE
        # against the table — bypassing this class's own API, which exposes neither —
        # aborts at the storage layer. `RAISE(ABORT, ...)` rolls back the statement that
        # triggered it (not the whole surrounding transaction) and raises to the caller
        # as a normal sqlite3 exception.
        self.conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS audit_log_no_update
            BEFORE UPDATE ON audit_log
            BEGIN
                SELECT RAISE(ABORT, 'audit_log is append-only: UPDATE is not permitted');
            END
            """
        )
        self.conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS audit_log_no_delete
            BEFORE DELETE ON audit_log
            BEGIN
                SELECT RAISE(ABORT, 'audit_log is append-only: DELETE is not permitted');
            END
            """
        )
        # Per-log salt for `result_hash` (module docstring point 3) — a small metadata
        # table rather than a bare instance attribute so it persists and is reused across
        # process restarts against the same file-backed `db_path`, not regenerated (and
        # so silently invalidating every prior entry's hash) on every reopen. Protected by
        # the same append-only triggers as `audit_log`, for the same reason: a salt an
        # attacker can rewrite at will stops being a useful defense against a dictionary
        # attack on `result_hash`.
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS audit_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS audit_meta_no_update
            BEFORE UPDATE ON audit_meta
            BEGIN
                SELECT RAISE(ABORT, 'audit_meta is append-only: UPDATE is not permitted');
            END
            """
        )
        self.conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS audit_meta_no_delete
            BEFORE DELETE ON audit_meta
            BEGIN
                SELECT RAISE(ABORT, 'audit_meta is append-only: DELETE is not permitted');
            END
            """
        )
        self.conn.commit()
        self.salt = self._load_or_create_salt()

        if anchor_path is None:
            if db_path != ":memory:":
                anchor_path = f"{db_path}.anchor.json"
            else:
                fd, anchor_path = tempfile.mkstemp(prefix="icberg_audit_anchor_", suffix=".json")
                os.close(fd)
        self.anchor_path = anchor_path

    def _load_or_create_salt(self) -> str:
        with self._lock:
            row = self.conn.execute("SELECT value FROM audit_meta WHERE key = 'salt'").fetchone()
            if row:
                return row[0]
            salt = secrets.token_hex(16)
            self.conn.execute("INSERT INTO audit_meta (key, value) VALUES ('salt', ?)", (salt,))
            self.conn.commit()
            return salt

    def hash_result(self, rows: list[dict[str, Any]]) -> str:
        """Salted `hash_result_rows(rows, self.salt)` — the audit trail's own `result_hash`
        must go through this (not the bare module-level function) so it carries this log's
        per-instance salt. See module docstring point 3.
        """
        return hash_result_rows(rows, salt=self.salt)

    def _read_anchor(self) -> tuple[int, str] | None:
        with self._lock:
            try:
                with open(self.anchor_path, encoding="utf-8") as f:
                    data = json.load(f)
                return int(data["seq"]), str(data["entry_hash"])
            except (OSError, ValueError, KeyError, TypeError):
                return None

    def _write_anchor(self, seq: int, entry_hash: str) -> None:
        """Write `{seq, entry_hash}` to `self.anchor_path`, outside the SQLite table
        entirely — this is what lets `verify()` detect a full, internally-self-consistent
        chain rewrite (see module docstring point 2). Written via a temp-file-then-
        `os.replace` swap so a crash mid-write can never leave a half-written, unparseable
        anchor file behind. Callers must already hold `self._lock` (this is only ever
        invoked from `append`, inside its own `with self._lock:` block) so the anchor
        write is part of the SAME atomic critical section as the row insert/commit.
        """
        payload = json.dumps({"seq": seq, "entry_hash": entry_hash})
        tmp_path = f"{self.anchor_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_path, self.anchor_path)

    def _last_entry_hash(self) -> str:
        with self._lock:
            row = self.conn.execute("SELECT entry_hash FROM audit_log ORDER BY seq DESC LIMIT 1").fetchone()
            return row[0] if row else GENESIS_HASH

    def _next_seq(self) -> int:
        with self._lock:
            row = self.conn.execute("SELECT MAX(seq) FROM audit_log").fetchone()
            return (row[0] or 0) + 1

    def ping(self) -> None:
        """Thread-safe connectivity check for `GET /health` (`gateway_routes.health`) —
        that route MUST call this rather than touching `self.conn` directly, so the
        health probe is guarded by the same lock as every other access to this
        connection instead of racing a concurrent `append()`/`verify()`.
        """
        with self._lock:
            self.conn.execute("SELECT 1")

    def append(
        self,
        *,
        actor: str,
        proposed_sql: str,
        classification: str,
        action: str,
        matched_rules: list[str],
        rows_returned: int,
        latency_ms: int,
        result_hash: str,
    ) -> AuditEntry:
        """Append one entry to the chain and return it (with `seq`/`entry_hash` filled in).

        Never mutates a previously appended entry — this is the only write path this
        class exposes; there is no `update`/`delete` method by design (and the append-only
        triggers created in `__init__` reject a caller that reaches for raw SQL instead).

        `proposed_sql` is scrubbed with `redaction.redact_text` before it is stored — see
        module docstring point 3 — so a literal PII value embedded in the proposed
        statement (`WHERE email='alice@example.com'`) never persists raw at rest. This
        happens unconditionally, inside this single write path, specifically so it can't
        be forgotten by a caller that constructs an entry some other way.

        H1 (thread-safety): reading `prev_hash`/`seq`, computing `entry_hash`, inserting
        the row, committing, AND writing the external anchor all happen inside ONE
        acquisition of `self._lock` below — not four separate locked calls. If they were
        separate, two concurrent callers could both read the same `prev_hash`/`seq`
        between their own read and write, both compute a chain-consistent-looking entry
        against the SAME predecessor, and corrupt the chain (two entries claiming the
        same `prev_hash`, or a duplicate `seq`) even though neither call individually
        raised. Holding the lock across the whole read-compute-write sequence is what
        makes `append()` atomic with respect to itself.
        """
        proposed_sql = redact_text(proposed_sql)

        with self._lock:
            seq = self._next_seq()
            prev_hash = self._last_entry_hash()
            timestamp = self._clock().isoformat()

            base_fields: dict[str, Any] = {
                "seq": seq,
                "timestamp": timestamp,
                "actor": actor,
                "proposed_sql": proposed_sql,
                "classification": classification,
                "action": action,
                "matched_rules": list(matched_rules),
                "rows_returned": rows_returned,
                "latency_ms": latency_ms,
                "result_hash": result_hash,
                "prev_hash": prev_hash,
            }
            entry_hash = _compute_entry_hash(base_fields)
            entry = AuditEntry(**base_fields, entry_hash=entry_hash)

            self.conn.execute(
                """
                INSERT INTO audit_log
                    (seq, timestamp, actor, proposed_sql, classification, action, matched_rules,
                     rows_returned, latency_ms, result_hash, prev_hash, entry_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.seq,
                    entry.timestamp,
                    entry.actor,
                    entry.proposed_sql,
                    entry.classification,
                    entry.action,
                    json.dumps(entry.matched_rules),
                    entry.rows_returned,
                    entry.latency_ms,
                    entry.result_hash,
                    entry.prev_hash,
                    entry.entry_hash,
                ),
            )
            self.conn.commit()
            self._write_anchor(entry.seq, entry.entry_hash)
            logger.info("audit.entry_appended", seq=seq, actor=actor, action=action, classification=classification)
            return entry

    def entries(self) -> list[AuditEntry]:
        """Every entry in the log, in sequence order."""
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT seq, timestamp, actor, proposed_sql, classification, action, matched_rules,
                       rows_returned, latency_ms, result_hash, prev_hash, entry_hash
                FROM audit_log ORDER BY seq ASC
                """
            ).fetchall()
        return [_row_to_entry(row) for row in rows]

    def verify(self) -> tuple[bool, int | None]:
        """Recompute the hash chain over every stored entry, then check its head against
        the external anchor file (module docstring point 2).

        Returns:
            `(True, None)` if every entry's `entry_hash` recomputes correctly, every
            entry's `prev_hash` matches the previous entry's stored `entry_hash` (genesis
            entry's `prev_hash` must equal `GENESIS_HASH`), AND — when the chain is
            non-empty — the last entry's `(seq, entry_hash)` matches what's stored in
            `self.anchor_path`. Otherwise `(False, seq)`.

            For a per-entry hash/link failure, `seq` is the sequence number of the
            *first* entry where either check fails — a broken link at entry N means
            everything after N is unverifiable too, but N itself is what should be
            reported and investigated first. For an anchor mismatch (every entry
            internally self-consistent, but the recomputed head disagrees with the
            anchor — the signature of a full, doctored chain rewrite that individually
            re-hashed every row to hide a per-entry tamper), `seq` is the last entry's
            own sequence number: that entry is the one whose content doesn't match what
            was anchored when it was first appended.
        """
        # Locked for the whole walk (not just each individual read) so `verify()` sees
        # a single, consistent snapshot even if a concurrent `append()` is racing it —
        # otherwise a chain that is perfectly valid before AND after a concurrent
        # append could be observed mid-append (entries read, anchor not yet re-checked
        # against the entry that just landed) and misreported.
        with self._lock:
            expected_prev = GENESIS_HASH
            last_entry: AuditEntry | None = None
            for entry in self.entries():
                if entry.prev_hash != expected_prev:
                    return False, entry.seq
                recomputed = _compute_entry_hash(
                    {
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
                        "prev_hash": entry.prev_hash,
                    }
                )
                if recomputed != entry.entry_hash:
                    return False, entry.seq
                expected_prev = entry.entry_hash
                last_entry = entry

            if last_entry is not None:
                anchor = self._read_anchor()
                if anchor is None or anchor != (last_entry.seq, last_entry.entry_hash):
                    # Internally self-consistent chain, but its recomputed head disagrees
                    # with the untouched external anchor — exactly the full-rewrite attack
                    # the anchor exists to catch (see module docstring point 2).
                    logger.warning(
                        "audit.verify_anchor_mismatch",
                        last_seq=last_entry.seq,
                        anchor=anchor,
                    )
                    return False, last_entry.seq
            return True, None


def _row_to_entry(row: tuple[Any, ...]) -> AuditEntry:
    (
        seq,
        timestamp,
        actor,
        proposed_sql,
        classification,
        action,
        matched_rules_json,
        rows_returned,
        latency_ms,
        result_hash,
        prev_hash,
        entry_hash,
    ) = row
    return AuditEntry(
        seq=seq,
        timestamp=timestamp,
        actor=actor,
        proposed_sql=proposed_sql,
        classification=classification,
        action=action,
        matched_rules=json.loads(matched_rules_json),
        rows_returned=rows_returned,
        latency_ms=latency_ms,
        result_hash=result_hash,
        prev_hash=prev_hash,
        entry_hash=entry_hash,
    )
