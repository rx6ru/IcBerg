"""The governance gateway — composes the Phase-0 policy gate with Phase-1's execution,
redaction, and audit controls into the single entry point an agent's proposed SQL goes
through end to end.

Per THREAT_MODEL.md's trust boundary, the agent only ever *proposes* SQL text; this
module is the trusted side that decides and executes. The sequence is fixed and always
runs, regardless of the decision:

    1. `GovernanceGate().evaluate(sql)` — decide (never executes anything itself).
    2. If `action == "allow"`: execute via the **read-only** executor, then redact PII
       from the result before it goes anywhere else. Redaction is schema-aware on a
       best-effort basis: `_introspect_schema` asks the executor for a live
       `SchemaCatalog` (tables/columns + view definitions — see `schema_catalog.py`)
       so `redact_rows`'s provenance layer can inline a view's real body and qualify
       against the database's actual schema, rather than only the proposed query's own
       text — this is what closes a PII column hidden behind a view/rename with no
       PII-looking name (e.g. `CREATE VIEW v AS SELECT ssn_num AS token FROM users`).
       If `action` is `"block"` or `"hold"`: execute nothing at all.
    3. Append one audit entry — always, on every path, including block/hold — so the
       audit trail records every proposal the gateway ever saw, not only the ones that
       ran.

Error messages that reach the caller (an engine error string, or the gate's own `reason`)
are passed through `redaction.redact_text` before being returned, since an engine error
can in principle echo back literal query text or data (THREAT_MODEL.md's redaction-
leakage concern applies to error paths, not just successful result rows).

**Observability (Phase 3, `.devdocs/FLAGSHIP_ROADMAP.md`):** `handle` is wrapped in an
OpenTelemetry root span `gateway.handle`, with child spans `parse`, `classify`, `policy`,
`execute`, `redact`, and `audit` bracketing the matching phase below (`execute`/`redact`
only exist on the `allow` path — nothing runs to trace on `block`/`hold`; `parse`/
`classify`/`policy`/`audit` always exist). `parse` and `classify` are two spans over the
SAME single call to `GovernanceGate.evaluate` rather than two calls to it: `evaluate`
parses and classifies as one atomic, deterministic operation internally, and
`sql_governance.py` is a heavily red-teamed security module this observability work
deliberately does not refactor the internals of — calling it twice would double real
parsing cost for no benefit, so `parse` brackets that one real call and `classify` reports
its outcome as an attribute immediately after. Every span attribute this module sets goes
through `telemetry.span`/`telemetry.set_span_attrs`'s own closed allow-list
(`action`, `classification`, `matched_rules_count`, `rows_returned`, `latency_ms`,
`redacted_columns_count`) — see `telemetry.py`'s module docstring for why raw SQL, `actor`,
and PII values structurally cannot reach a span attribute here, not merely "aren't passed
by convention." `Gateway.handle` also records one Prometheus decision (`GatewayMetrics
.record_decision`) per call, on every path, via the `metrics` bundle passed to
`__init__` (or `telemetry.DEFAULT_METRICS` if none was given) — mirroring the audit log's
own unconditional "one entry per proposal" guarantee.
"""

from __future__ import annotations

from typing import Any

import structlog

from backend.core import telemetry
from backend.core.audit import AuditLog
from backend.core.executor import ReadOnlyExecutor
from backend.core.policy import Policy, apply_extra_pii_redaction, apply_policy
from backend.core.redaction import redact_rows, redact_text
from backend.core.schema_catalog import SchemaCatalog
from backend.core.sql_governance import GovernanceGate

logger = structlog.get_logger(__name__)


class Gateway:
    """Runtime governance gateway: policy decision -> execution -> redaction -> audit.

    Args:
        gate: The `GovernanceGate` to evaluate proposed SQL against. Defaults to a fresh
            `GovernanceGate()`.
        metrics: The `telemetry.GatewayMetrics` bundle `handle` records each decision
            into. Defaults to `telemetry.DEFAULT_METRICS`, the module-level, process-wide
            bundle — correct for a direct/standalone `Gateway()` (every use across this
            codebase's own test suite). `backend/gateway_app.py`'s `create_gateway_app`
            passes an explicit, freshly-isolated bundle instead (`telemetry
            .new_metrics_registry()`) — see that function's and `GatewayMetrics`'s own
            docstrings for why an HTTP-facing `/metrics` route needs per-instance
            isolation that the shared default must not provide.
    """

    def __init__(
        self, gate: GovernanceGate | None = None, *, metrics: "telemetry.GatewayMetrics | None" = None
    ) -> None:
        self._gate = gate or GovernanceGate()
        self._metrics = metrics or telemetry.DEFAULT_METRICS

    @staticmethod
    def _introspect_schema(executor: ReadOnlyExecutor) -> SchemaCatalog | None:
        """Best-effort: ask `executor` for a live `SchemaCatalog` (tables/columns +
        view definitions) so `redact_rows`'s provenance layer can resolve views and
        ambiguous JOINs against the database's REAL schema — see `schema_catalog.py`'s
        module docstring for why this closes the view/rename PII leak class that
        query-text-only lineage tracing could not.

        Deliberately optional and lazy, at two levels: an `executor` with no
        `get_schema_catalog` method (any executor type that hasn't implemented schema
        introspection — e.g. today's `PostgresReadOnlyExecutor` stub) is treated
        identically to one that has it but returns `None`, and the call itself is
        wrapped so that any unexpected exception here degrades to schema-less
        redaction rather than ever breaking the gateway's own "never raises" contract.
        """
        get_schema_catalog = getattr(executor, "get_schema_catalog", None)
        if get_schema_catalog is None:
            return None
        try:
            return get_schema_catalog()
        except Exception as exc:  # noqa: BLE001 - schema introspection must never break the gateway
            logger.warning("gateway.schema_introspection_failed", error=str(exc))
            return None

    def handle(
        self,
        sql: str,
        actor: str,
        executor: ReadOnlyExecutor,
        audit_log: AuditLog,
        *,
        mode: str = "read",
        policy: Policy | None = None,
    ) -> dict[str, Any]:
        """Evaluate, (maybe) execute, redact, and always audit one proposed SQL statement.

        Args:
            sql: The proposed SQL statement text (untrusted).
            actor: Identifier of the proposing agent/user, recorded on the audit entry.
            executor: The read-only executor to run `sql` on if — and only if — the gate
                allows it. Callers MUST pass a `ReadOnlyExecutor` (or equivalent) here.
                Unlike Phase 1, this is no longer merely a documented convention: see the
                `IS_READONLY` assertion immediately below, which refuses (raises) rather
                than trusting the caller — a defense-in-depth backstop against a future
                wiring bug that hands this method a write-capable executor for the read
                path (`.devdocs/PHASE2_GATES.md` P2.19).
            audit_log: The `AuditLog` every decision is appended to, allow/block/hold alike.
            mode: Forwarded to `GovernanceGate.evaluate` (logged only; see its docstring).
            policy: Optional `backend.core.policy.Policy`, applied immediately after the
                base gate's decision (`policy.py`'s `apply_policy`) and before execution —
                it can only further-restrict that decision (deny/allow-list a table,
                force approval on a write, cap the returned row count), never loosen it.
                `None` (the default) is a complete pass-through: every pre-Phase-3 caller
                of this method is unaffected.

        Returns:
            `{"action", "reason", "matched_rules", "rows", "redaction_report", "audit_seq"}`.
            `action` is the gate's decision ("allow"/"block"/"hold"). `rows` and
            `redaction_report` are `None` unless `action == "allow"` and execution
            succeeded. `reason` is always PII-scrubbed before being returned.

        Raises:
            TypeError: if `executor` is not explicitly marked read-only (`IS_READONLY =
                True`) — this is a caller/wiring bug, not untrusted input, so it is
                raised rather than degraded to a `block` decision. Every read-only
                executor in this codebase (`executor.ReadOnlyExecutor`,
                `executor.PostgresReadOnlyExecutor`, `connectors.MySQLReadOnlyExecutor`)
                declares `IS_READONLY = True`; every write executor (`WriteExecutor`,
                `PostgresWriteExecutor`, `MySQLWriteExecutor`) declares it `False`. An
                executor of an unknown type with no such attribute at all fails closed
                (rejected), not open.
        """
        if getattr(executor, "IS_READONLY", None) is not True:
            raise TypeError(
                "Gateway.handle requires an executor explicitly marked IS_READONLY = "
                f"True for the read path; got {type(executor).__name__!r}, which is "
                "either write-capable or not marked read-only at all. This is a "
                "programming/wiring error, not a decision about the proposed SQL — "
                "the gate never even evaluated it."
            )

        with telemetry.span("gateway.handle") as root_span:
            # `parse` brackets the one real call that does both parsing and
            # classification (`GovernanceGate.evaluate`); `classify` reports that call's
            # outcome as a span of its own immediately after — see this module's
            # docstring for why these are two spans over one call, not two calls.
            with telemetry.span("parse"):
                decision = self._gate.evaluate(sql, mode=mode, actor=actor)

            with telemetry.span("classify") as classify_span:
                telemetry.set_span_attrs(
                    classify_span,
                    classification=decision.classification,
                    matched_rules_count=len(decision.matched_rules),
                )

            # Policy is applied here, on top of the base gate's decision, BEFORE
            # execution — so a policy-driven block/hold/row-cap is honored by every
            # caller of this method, not something an individual integration surface
            # could opt out of. See `policy.apply_policy`'s docstring for why this can
            # only further-restrict.
            effective_max_rows: int | None = None
            with telemetry.span("policy") as policy_span:
                if policy is not None:
                    decision, effective_max_rows = apply_policy(decision, policy)
                telemetry.set_span_attrs(
                    policy_span,
                    action=decision.action,
                    classification=decision.classification,
                    matched_rules_count=len(decision.matched_rules),
                )

            rows: list[dict[str, Any]] | None = None
            redaction_report: dict[str, Any] | None = None
            rows_returned = 0
            latency_ms = 0
            # `audit_log.hash_result` (not the bare `hash_result_rows`) so this carries the
            # log's own per-instance salt — see audit.py's module docstring point 3 (a raw,
            # unsalted `result_hash` is a dictionary-attackable preimage of the result set).
            result_hash = audit_log.hash_result([])
            reason = decision.reason

            # `executor.execute` and `redact_rows` are both documented to never raise — but
            # this `try`/`finally` is the guarantee for that promise, not a restatement of it:
            # if a future bug (or a caller-supplied executor that doesn't hold the contract)
            # raises here anyway, the audit write in `finally` still runs unconditionally
            # rather than the proposal silently vanishing from the trail. The exception is
            # swallowed into a scrubbed `reason` (never re-raised) so this method keeps its
            # own "never raises" contract end to end.
            try:
                if decision.action == "allow":
                    with telemetry.span("execute") as exec_span:
                        result = executor.execute(sql)
                        latency_ms = result.latency_ms
                        telemetry.set_span_attrs(exec_span, latency_ms=latency_ms, rows_returned=result.rowcount)

                    if result.error is not None:
                        # Fail-safe: an execution error (including the read-only engine
                        # itself rejecting a write that should never have reached "allow" in
                        # the first place) never yields rows — it is reported and audited,
                        # nothing else.
                        reason = f"query execution failed: {result.error}"
                        logger.warning("gateway.execution_failed", actor=actor, error=result.error)
                    else:
                        with telemetry.span("redact") as redact_span:
                            # `sql=sql` enables provenance (lineage) redaction on top of
                            # name/value scanning — see redaction.py's module docstring
                            # layer 3. `schema=...` (best-effort, may be None)
                            # additionally lets that layer inline views and fully
                            # qualify/expand against the real database schema instead of
                            # falling back to unknown-schema heuristics — see
                            # `_introspect_schema` and `schema_catalog.py`.
                            schema = self._introspect_schema(executor)
                            rows, redaction_report = redact_rows(result.rows, result.columns, sql=sql, schema=schema)

                            # `policy.pii_columns` — an ADDITIONAL masking pass on top of
                            # `redact_rows`'s own layered redaction, never a replacement for it
                            # (see policy.py's module docstring).
                            if policy is not None and policy.pii_columns:
                                rows, extra_cols = apply_extra_pii_redaction(rows, result.columns, policy.pii_columns)
                                if extra_cols:
                                    redaction_report = {
                                        "columns_redacted": sorted(
                                            set(redaction_report["columns_redacted"]) | set(extra_cols)
                                        ),
                                        "values_masked": redaction_report["values_masked"]
                                        + sum(
                                            1
                                            for original_row in result.rows
                                            for col in extra_cols
                                            if original_row.get(col) not in (None, "")
                                        ),
                                    }

                            # `policy.max_rows` — truncates further down from whatever the
                            # executor's own engine-level cap already produced; never used to
                            # request more rows than the engine returned (see policy.py).
                            if effective_max_rows is not None and len(rows) > effective_max_rows:
                                rows = rows[:effective_max_rows]

                            rows_returned = len(rows)
                            result_hash = audit_log.hash_result(rows)
                            telemetry.set_span_attrs(
                                redact_span,
                                rows_returned=rows_returned,
                                redacted_columns_count=len(redaction_report["columns_redacted"]),
                            )
                # block/hold: no execution at all — rows/redaction_report stay None,
                # rows_returned/latency_ms stay 0, result_hash stays the salted hash of an
                # empty result set.
            except Exception as exc:  # noqa: BLE001 - deliberately broad, see comment above
                logger.warning("gateway.unexpected_error", actor=actor, error=str(exc))
                reason = f"unexpected gateway error: {exc}"
                rows = None
                redaction_report = None
            finally:
                with telemetry.span("audit") as audit_span:
                    audit_entry = audit_log.append(
                        actor=actor,
                        proposed_sql=sql,
                        classification=decision.classification,
                        action=decision.action,
                        matched_rules=decision.matched_rules,
                        rows_returned=rows_returned,
                        latency_ms=latency_ms,
                        result_hash=result_hash,
                    )
                    telemetry.set_span_attrs(
                        audit_span, action=decision.action, rows_returned=rows_returned, latency_ms=latency_ms
                    )

                # One Prometheus decision recorded per `handle()` call, on every path —
                # mirroring the audit write immediately above. See `GatewayMetrics
                # .record_decision`'s docstring.
                self._metrics.record_decision(decision.action, latency_ms=latency_ms, rows_returned=rows_returned)

                telemetry.set_span_attrs(
                    root_span,
                    action=decision.action,
                    classification=decision.classification,
                    matched_rules_count=len(decision.matched_rules),
                    rows_returned=rows_returned,
                    latency_ms=latency_ms,
                    redacted_columns_count=len(redaction_report["columns_redacted"]) if redaction_report else 0,
                )

        return {
            "action": decision.action,
            "reason": redact_text(reason),
            "matched_rules": decision.matched_rules,
            "rows": rows,
            "redaction_report": redaction_report,
            "audit_seq": audit_entry.seq,
        }
