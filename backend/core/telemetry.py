"""Observability for the governance gateway (Phase 3, `.devdocs/FLAGSHIP_ROADMAP.md`):
an OpenTelemetry tracer plus Prometheus metrics, kept deliberately small and safe to
import from anywhere in `backend/core/*` with zero configuration.

Two hard rules this module enforces structurally, not just by code-review discipline
(THREAT_MODEL.md's redaction-leakage concern applies to telemetry exactly as much as it
applies to error paths and audit rows):

  1. **No raw SQL, actor identifiers, or PII values ever become a span attribute or a
     Prometheus label/metric name.** `span()`/`set_span_attrs()` apply a fixed, CLOSED
     allow-list of attribute keys (`_ALLOWED_SPAN_ATTRS`) — passing anything else is
     silently dropped, not attached, so a future caller that absent-mindedly writes
     `telemetry.span("execute", sql=sql)` cannot leak it: the value never reaches the
     span. There is no matching deny-list to keep in sync as new PII-shaped fields are
     invented elsewhere in the codebase; the allow-list is the only way in.
  2. **The only Prometheus metric labeled at all is `queries_total`'s `action` label**,
     and its value set is exactly `PolicyDecision.action`'s own closed, 3-valued type
     (`"allow"`/`"block"`/`"hold"`) — never an actor, a table name, or anything else with
     unbounded cardinality. Every metric this module serves over the public HTTP
     `/metrics` route (see `GatewayMetrics.public_text`) carries NO labels at all.

Safe to import with no OpenTelemetry SDK configured anywhere in the process: `_tracer`
is obtained once, at import time, via the plain `opentelemetry-api` (`trace.get_tracer`,
not the SDK) — with no `TracerProvider` ever installed, this resolves to a `ProxyTracer`
backed by the API's built-in no-op implementation, so every `span()` call below produces
a real, `is_recording() == False` span object that costs almost nothing and exports
nowhere. This module never calls `trace.set_tracer_provider(...)` itself — installing an
SDK provider (and its exporter) is entirely the embedding application's/test's choice;
`opentelemetry-api`'s own `ProxyTracer` mechanism means a provider installed AFTER this
module (and every module that imported it) is still picked up correctly by every
already-obtained tracer, not just ones obtained afterward — see the `ProxyTracer`
docstring in `opentelemetry.trace` for the mechanism this relies on.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    disable_created_metrics,
    generate_latest,
)

# `_created` gauge lines (a `Counter`/`Histogram`'s own creation timestamp, emitted by
# `prometheus_client` by default) carry a wall-clock epoch float with nothing left to
# redact but also nothing useful for this gateway's own consumers — disabled globally,
# once, at import time so no exposition text this module ever produces (rich or public)
# carries them.
disable_created_metrics()

_TRACER_NAME = "icberg.gateway"

# Obtained once, at import time — see module docstring for why this is safe even with no
# SDK configured, and why a provider installed later by an embedding app/test still works.
_tracer = trace.get_tracer(_TRACER_NAME)

# --------------------------------------------------------------------------------------
# Tracing
# --------------------------------------------------------------------------------------

# The CLOSED set of span attribute keys `span()`/`set_span_attrs()` will ever actually
# attach — see module docstring point 1. Every name here is a bucketed count, an enum-like
# decision value, or a duration; none of them can carry free-text SQL, an actor identifier,
# or a data value under any legitimate call.
_ALLOWED_SPAN_ATTRS: frozenset[str] = frozenset({
    "action",
    "classification",
    "matched_rules_count",
    "rows_returned",
    "latency_ms",
    "redacted_columns_count",
})


def _apply_safe_attrs(sp: Span, attrs: dict[str, Any]) -> None:
    """Attach only the allow-listed keys in `attrs` to `sp`, dropping (never attaching)
    anything else and any `None` value. See module docstring point 1 — this is the one
    place either `span()` or `set_span_attrs()` ever calls `Span.set_attribute`, so a
    future third entry point cannot bypass the allow-list by construction.
    """
    for key, value in attrs.items():
        if key not in _ALLOWED_SPAN_ATTRS or value is None:
            continue
        sp.set_attribute(key, value)


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span]:
    """Start a child span named `name` under whatever span is currently active (or a new
    root span if none is), yield it, and end it on exit.

    `**attrs` are applied immediately (through the same allow-list `set_span_attrs` uses)
    — pass what's already known when the span opens; call `set_span_attrs(sp, ...)` again
    inside the `with` block for anything only known once the wrapped work completes (an
    attribute set after the span has already ended, e.g. outside its own `with` block, is
    silently a no-op per the OpenTelemetry SDK's own contract — always set attributes
    while the span returned here is still open).

    An exception raised inside the `with` block is recorded as a span error (`StatusCode
    .ERROR`, the exception's type name only — never `str(exc)`, which could in principle
    echo back a value this module has no way to scrub) and then re-raised unchanged: this
    never swallows or alters caller behavior, it only annotates the trace. `record_exception`
    and `set_status_on_exception` are explicitly disabled below: the OpenTelemetry SDK's own
    defaults for both are `True`, and left at that default the SDK would overwrite this
    type-name-only status with the exception's full `str(exc)` AND attach a separate
    `exception` event carrying `exception.message`/`exception.stacktrace` — silently
    bypassing this module's own allow-list and re-introducing exactly the raw-SQL/PII leak
    point 1 above exists to close. With both disabled, this docstring's guarantee is
    actually enforced by the SDK, not just by this module's own call below.
    """
    with _tracer.start_as_current_span(name, record_exception=False, set_status_on_exception=False) as sp:
        _apply_safe_attrs(sp, attrs)
        try:
            yield sp
        except Exception as exc:  # noqa: BLE001 - re-raised unchanged immediately below
            sp.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise


def set_span_attrs(sp: Span, **attrs: Any) -> None:
    """Attach additional allow-listed attributes to an already-open span `sp` (typically
    the object `span()` yielded) once their values are known. See `span()`'s docstring for
    why this must be called before the span's own `with` block exits.
    """
    _apply_safe_attrs(sp, attrs)


# --------------------------------------------------------------------------------------
# Prometheus metrics
# --------------------------------------------------------------------------------------

# Integer-valued-float lines (`name value.0`) that `prometheus_client.generate_latest`
# always produces for a `Counter` (Prometheus samples are float64 by the spec; there is no
# library option to render a whole-number sample without the `.0`) are rewritten to plain
# integers in `GatewayMetrics.public_text()` — see that method's docstring for why: the
# gateway's public `/metrics` contract (`backend/api/gateway_routes.py`, P2.17, predates
# this module) already promised bare-integer bucketed counter values, and every existing
# caller/test of that endpoint depends on that exact shape. Anchored to end-of-line so it
# only ever touches a bare `name value` sample line, never (for example) a `# HELP` line
# that happens to end in text resembling this pattern.
_TRAILING_INTEGER_FLOAT_RE = re.compile(r"^(\S+) (-?\d+)\.0$", re.MULTILINE)

# Histogram bucket boundaries. Latency in milliseconds, spanning "fast bounded read" up to
# the executor's own hard timeout/kill ceiling (`executor.TIMEOUT_SECONDS` = 5s = 5000ms;
# a bucket above that captures the (rare) forced-timeout/process-kill path too) and beyond.
_LATENCY_BUCKETS_MS: tuple[float, ...] = (1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, float("inf"))
# Row counts, spanning "single row" up to the executor's own forced cap (`executor.MAX_ROWS`
# = 1000) and one bucket beyond it for a caller that raised `max_rows` above the default.
_ROWS_BUCKETS: tuple[float, ...] = (0, 1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000, float("inf"))


class GatewayMetrics:
    """One fully self-contained bundle of Prometheus metrics: its own `CollectorRegistry`
    instances (plural — see below) and counter/histogram objects, so two bundles never
    share state. `backend/gateway_app.py`'s `create_gateway_app` calls `new_metrics_registry()`
    once per app instance specifically so two governed-database apps — or two apps built by
    two different tests in the same pytest process — start every counter at zero and never
    observe each other's traffic; see that function's docstring.

    Two registries, deliberately, not one:

      - `registry` (rich): every metric this module tracks, including the per-action
        LABELED `queries_total` counter and the latency/rows-returned HISTOGRAMS.
        `generate_latest()` exposes this one — for `tests/observability`'s own direct
        assertions and any future internal scrape target that wants the full picture.
        Labels here are the single, closed, 3-valued `action` set (see module docstring
        point 2), never a per-actor/per-query value.
      - `public_registry` (HTTP-safe): unlabeled counters ONLY — no labels, no histograms
        (a Prometheus histogram's own `_bucket{le="..."}` line carries a label by
        construction, which is exactly the shape this registry exists to never produce) —
        mirroring the same six bucketed counter names `backend/api/gateway_routes.py`'s
        `/metrics` route has promised since Phase 2 (P2.17). `public_text()` is what that
        route actually serves; `generate_latest()`/`registry` is never wired to it.
    """

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.public_registry = CollectorRegistry()

        # --- rich (internal / test-facing) ---
        self.queries_total = Counter(
            "icberg_queries_total",
            "Total proposed SQL statements evaluated by the gateway, by decision.",
            ["action"],
            registry=self.registry,
        )
        self.query_latency_ms = Histogram(
            "icberg_query_latency_ms",
            "Gateway end-to-end execution latency in milliseconds.",
            buckets=_LATENCY_BUCKETS_MS,
            registry=self.registry,
        )
        self.rows_returned = Histogram(
            "icberg_rows_returned",
            "Rows returned per allowed query, after redaction.",
            buckets=_ROWS_BUCKETS,
            registry=self.registry,
        )
        self.blocks_total = Counter(
            "icberg_blocks_total", "Total proposed statements blocked.", registry=self.registry
        )
        self.holds_total = Counter(
            "icberg_holds_total", "Total proposed statements held for human approval.", registry=self.registry
        )
        self.approvals_total = Counter(
            "icberg_approvals_total", "Total held statements approved and executed.", registry=self.registry
        )
        self.rejections_total = Counter(
            "icberg_rejections_total", "Total held statements rejected.", registry=self.registry
        )
        self.rate_limited_total = Counter(
            "icberg_rate_limited_total",
            "Total requests rejected by the per-actor rate limit.",
            registry=self.registry,
        )

        # --- public (HTTP `/metrics`) mirrors: unlabeled, same names, separate registry ---
        self._public_queries_total = Counter(
            "icberg_queries_total", "Total proposed SQL statements evaluated by the gateway.",
            registry=self.public_registry,
        )
        self._public_blocks_total = Counter(
            "icberg_blocks_total", "Total proposed statements blocked.", registry=self.public_registry
        )
        self._public_holds_total = Counter(
            "icberg_holds_total", "Total proposed statements held for human approval.", registry=self.public_registry
        )
        self._public_approvals_total = Counter(
            "icberg_approvals_total", "Total held statements approved and executed.", registry=self.public_registry
        )
        self._public_rejections_total = Counter(
            "icberg_rejections_total", "Total held statements rejected.", registry=self.public_registry
        )
        self._public_rate_limited_total = Counter(
            "icberg_rate_limited_total",
            "Total requests rejected by the per-actor rate limit.",
            registry=self.public_registry,
        )

    def record_decision(self, action: str, *, latency_ms: int = 0, rows_returned: int = 0) -> None:
        """Record one `Gateway.handle` decision. Called exactly once per `handle()` call,
        on every path (allow/block/hold, including an unexpected-error path) — mirroring
        `AuditLog.append`'s own "always, on every path" guarantee (`gateway.py`'s module
        docstring). `action` MUST be `PolicyDecision.action`'s own value
        (`"allow"`/`"block"`/`"hold"`) — never derived from free-text input — since it is
        the one and only Prometheus label this module ever sets (module docstring point 2).
        """
        self.queries_total.labels(action=action).inc()
        self.query_latency_ms.observe(max(latency_ms, 0))
        self._public_queries_total.inc()
        if action == "allow":
            self.rows_returned.observe(max(rows_returned, 0))
        elif action == "block":
            self.blocks_total.inc()
            self._public_blocks_total.inc()
        elif action == "hold":
            self.holds_total.inc()
            self._public_holds_total.inc()

    def record_approval(self) -> None:
        """One held statement approved and executed (`ApprovalQueue.approve`, via
        `backend/api/gateway_routes.py`'s `decide_approval`)."""
        self.approvals_total.inc()
        self._public_approvals_total.inc()

    def record_rejection(self) -> None:
        """One held statement rejected (`ApprovalQueue.reject`, via `decide_approval`)."""
        self.rejections_total.inc()
        self._public_rejections_total.inc()

    def record_rate_limited(self) -> None:
        """One request rejected by the per-actor rate limit (`gateway_routes
        ._check_rate_limit`)."""
        self.rate_limited_total.inc()
        self._public_rate_limited_total.inc()

    def generate_latest(self) -> bytes:
        """Raw `prometheus_client.generate_latest()` bytes for the RICH registry — the
        full instrumentation surface (the labeled `queries_total` counter, both
        histograms, and every unlabeled convenience counter). NEVER wired to the public
        HTTP `/metrics` route — see `public_text()` and this class's docstring for why.
        """
        return generate_latest(self.registry)

    def public_text(self) -> str:
        """Prometheus text-exposition format for the PUBLIC registry only — unlabeled
        bucketed counters, no histograms, no `{...}` label set anywhere in the output —
        this is what `backend/api/gateway_routes.py`'s `/metrics` route actually returns.

        Every whole-number `Counter` sample `prometheus_client.generate_latest` emits is
        rewritten from `name 3.0` to `name 3` (see `_TRAILING_INTEGER_FLOAT_RE`): Prometheus
        samples are float64 by the exposition format's own spec, so the library always
        renders an exact integer count with a trailing `.0` — this endpoint's contract
        predates this module (P2.17, `.devdocs/PHASE2_GATES.md`) and both its existing
        callers/tests already depend on a bare-integer value, so this rewrite is what lets
        this module become the endpoint's real backing store via `generate_latest()` (as
        directed) without breaking that pre-existing, still-relied-on contract.
        """
        raw = generate_latest(self.public_registry).decode("utf-8")
        return _TRAILING_INTEGER_FLOAT_RE.sub(r"\1 \2", raw)


# Module-level default: what any `Gateway()` constructed with no explicit `metrics=`
# (every direct, non-HTTP caller across this codebase's own test suite, e.g.
# `tests/security/test_governance_runtime.py`) records into. Process-wide and
# long-lived — real Prometheus metrics are meant to accumulate for a process's whole
# lifetime, so sharing this across every such caller is correct, not a bug — but see
# `new_metrics_registry()` immediately below for why `backend/gateway_app.py`'s
# `create_gateway_app` must NOT use this one for its own, per-instance public `/metrics`
# route.
DEFAULT_METRICS = GatewayMetrics()


def new_metrics_registry() -> GatewayMetrics:
    """Factory: build a fresh, fully isolated `GatewayMetrics` bundle — its own
    registries, every counter/histogram starting at zero.

    `backend/gateway_app.py`'s `create_gateway_app` calls this once per app instance and
    hands it to that app's own `Gateway(gate, metrics=...)`, specifically so two governed-
    database app instances (or two instances built by two different tests in the same
    pytest process, as `tests/api/test_gateway_api.py` routinely does) never share a
    `/metrics` counter — sharing `DEFAULT_METRICS` here would make that endpoint's counts
    depend on test execution order and prior tests' traffic, breaking the exact-value
    assertions its existing tests already make.
    """
    return GatewayMetrics()
