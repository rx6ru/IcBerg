"""Phase 3 observability tests (`.devdocs/FLAGSHIP_ROADMAP.md`) — `backend/core
/telemetry.py`'s OpenTelemetry tracing and Prometheus metrics, exercised both directly
against `Gateway.handle` (span tree + attribute allow-list) and through the governed
HTTP API (`/metrics`).

Synthetic schema/data throughout, matching the rest of this repo's test convention
(`tests/security/test_governance_runtime.py`, `tests/api/test_gateway_api.py`): a
`users(id, name, email, admin)` table with two fabricated rows — no real data.

OpenTelemetry SDK wiring: `opentelemetry.trace.set_tracer_provider` may only be called
ONCE per process (a later call is a documented, silent no-op) — so the `TracerProvider` +
`InMemorySpanExporter` used to capture spans below are installed exactly once, at this
module's import time, and every test clears the exporter's buffer before and after
itself (`_clear_spans`, autouse) rather than reinstalling the provider. This is safe
precisely because nothing else in this test run (`tests/security`, `tests/api`) ever
touches OpenTelemetry — see `backend/core/telemetry.py`'s module docstring for why the
module-level tracer every one of those other modules' `Gateway`/`GovernanceGate` calls
transitively uses still correctly resolves to the provider installed here, even though it
was obtained (via `trace.get_tracer`) long before this module ever ran.
"""

from __future__ import annotations

import subprocess
import sys
import sqlite3
import textwrap

import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client.parser import text_string_to_metric_families

from backend.core.audit import AuditLog
from backend.core.executor import ReadOnlyExecutor
from backend.core.gateway import Gateway
from backend.gateway_app import create_gateway_app

# --------------------------------------------------------------------------------------
# OTel SDK wiring — see module docstring.
# --------------------------------------------------------------------------------------

_EXPORTER = InMemorySpanExporter()
_PROVIDER = TracerProvider()
_PROVIDER.add_span_processor(SimpleSpanProcessor(_EXPORTER))
trace.set_tracer_provider(_PROVIDER)

# The exact, closed set of span attribute keys `telemetry.span`/`telemetry
# .set_span_attrs` will ever attach — mirrors `backend/core/telemetry.py`'s own
# `_ALLOWED_SPAN_ATTRS` so a test failure here reads as "an attribute outside the
# allow-list reached a span" rather than a magic, undocumented list of strings.
_ALLOWED_SPAN_ATTRS = frozenset({
    "action",
    "classification",
    "matched_rules_count",
    "rows_returned",
    "latency_ms",
    "redacted_columns_count",
})

_CHILD_SPAN_NAMES = ("parse", "classify", "policy", "execute", "redact", "audit")


@pytest.fixture(autouse=True)
def _clear_spans():
    _EXPORTER.clear()
    yield
    _EXPORTER.clear()


def _spans_by_name(name: str):
    return [s for s in _EXPORTER.get_finished_spans() if s.name == name]


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

USERS_SCHEMA_SQL = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    admin INTEGER
)
"""


@pytest.fixture
def db_path(tmp_path) -> str:
    """A throwaway SQLite file with a `users` table (including a PII `email` column) —
    fabricated rows only, no real data."""
    path = str(tmp_path / "observability.sqlite")
    conn = sqlite3.connect(path)
    try:
        conn.execute(USERS_SCHEMA_SQL)
        conn.executemany(
            "INSERT INTO users (id, name, email, admin) VALUES (?, ?, ?, ?)",
            [
                (1, "Alice Smith", "alice@example.com", 0),
                (2, "Bob Jones", "bob@example.com", 0),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return path


# --------------------------------------------------------------------------------------
# Gateway.handle span tree
# --------------------------------------------------------------------------------------


class TestGatewayHandleSpans:
    def test_allow_path_emits_root_and_every_child_span_with_expected_attributes(self, db_path: str) -> None:
        gateway = Gateway()
        executor = ReadOnlyExecutor(db_path)
        audit_log = AuditLog()

        result = gateway.handle(
            "SELECT id, email FROM users WHERE id=1 LIMIT 1", "agent-1", executor, audit_log
        )
        assert result["action"] == "allow"

        spans = _EXPORTER.get_finished_spans()
        names = [s.name for s in spans]
        assert names.count("gateway.handle") == 1
        for child_name in _CHILD_SPAN_NAMES:
            assert names.count(child_name) == 1, f"expected exactly one {child_name!r} span, saw {names}"

        root = _spans_by_name("gateway.handle")[0]
        root_ctx = root.get_span_context()
        assert root.attributes["action"] == "allow"
        assert root.attributes["classification"] == "read"
        assert root.attributes["matched_rules_count"] == len(result["matched_rules"])
        assert root.attributes["rows_returned"] == 1
        assert isinstance(root.attributes["latency_ms"], int)
        # `email` is a PII column -> redacted -> counted on the root span.
        assert root.attributes["redacted_columns_count"] >= 1
        assert root.attributes["redacted_columns_count"] == len(result["redaction_report"]["columns_redacted"])

        # Every child span is a direct child of the root span, in the same trace.
        for child_name in _CHILD_SPAN_NAMES:
            child = _spans_by_name(child_name)[0]
            assert child.context.trace_id == root_ctx.trace_id
            assert child.parent is not None
            assert child.parent.span_id == root_ctx.span_id

        classify_span = _spans_by_name("classify")[0]
        assert classify_span.attributes["classification"] == "read"
        redact_span = _spans_by_name("redact")[0]
        assert redact_span.attributes["redacted_columns_count"] >= 1
        execute_span = _spans_by_name("execute")[0]
        assert execute_span.attributes["rows_returned"] == 1

    def test_block_path_never_emits_execute_or_redact_spans(self, db_path: str) -> None:
        gateway = Gateway()
        executor = ReadOnlyExecutor(db_path)
        audit_log = AuditLog()

        result = gateway.handle("DROP TABLE users", "agent-1", executor, audit_log)
        assert result["action"] == "block"

        names = [s.name for s in _EXPORTER.get_finished_spans()]
        assert names.count("gateway.handle") == 1
        for child_name in ("parse", "classify", "policy", "audit"):
            assert child_name in names
        # Nothing executed and nothing was redacted -- no span for either phase.
        assert "execute" not in names
        assert "redact" not in names

        root = _spans_by_name("gateway.handle")[0]
        assert root.attributes["action"] == "block"
        assert root.attributes["rows_returned"] == 0
        assert root.attributes["redacted_columns_count"] == 0

    def test_hold_path_never_emits_execute_or_redact_spans(self, db_path: str) -> None:
        gateway = Gateway()
        executor = ReadOnlyExecutor(db_path)
        audit_log = AuditLog()

        # No WHERE/LIMIT -> unbounded SELECT -> held for review, not executed.
        result = gateway.handle("SELECT id FROM users", "agent-1", executor, audit_log)
        assert result["action"] == "hold"

        names = [s.name for s in _EXPORTER.get_finished_spans()]
        assert "execute" not in names
        assert "redact" not in names
        assert _spans_by_name("gateway.handle")[0].attributes["action"] == "hold"

    def test_no_span_attribute_ever_contains_raw_sql_actor_or_pii(self, db_path: str) -> None:
        """The crown assertion for P3's telemetry gate: feed a query and an actor
        identifier that both carry a real-looking secret, then scan EVERY span emitted
        for EITHER an attribute key outside the closed allow-list OR a value containing
        that secret. `backend/core/telemetry.py`'s `_apply_safe_attrs` should make the
        first case structurally impossible; this proves it end to end through the real
        `Gateway.handle` call path, not just by reading the source.
        """
        gateway = Gateway()
        executor = ReadOnlyExecutor(db_path)
        audit_log = AuditLog()

        secret_sql = "SELECT id, email FROM users WHERE email='alice@example.com' LIMIT 1"
        secret_actor = "agent-super-secret-identity"
        gateway.handle(secret_sql, secret_actor, executor, audit_log)

        forbidden_substrings = ("SELECT", "FROM users", "alice@example.com", secret_actor, "email")

        spans = _EXPORTER.get_finished_spans()
        assert spans  # the call above must actually have produced spans
        for sp in spans:
            for key, value in sp.attributes.items():
                assert key in _ALLOWED_SPAN_ATTRS, f"disallowed span attribute key {key!r} on span {sp.name!r}"
                text = str(value)
                for forbidden in forbidden_substrings:
                    assert forbidden not in text, (
                        f"span {sp.name!r} attribute {key!r}={value!r} leaked {forbidden!r}"
                    )

    def test_span_exception_never_leaks_message_via_sdk_defaults(self) -> None:
        """Regression for the MEDIUM privacy bug: under a real (non-no-op) tracer, the
        OpenTelemetry SDK's own defaults for `start_as_current_span` are
        `record_exception=True, set_status_on_exception=True` — left at those defaults the
        SDK overwrites `telemetry.span`'s intended type-name-only status with the raised
        exception's full `str(exc)` AND attaches a separate `exception` event carrying
        `exception.message`/`exception.stacktrace`, leaking whatever SQL/PII text the
        exception message happens to contain straight into the trace. `backend/core
        /telemetry.py`'s `span()` must pass `record_exception=False,
        set_status_on_exception=False` so neither of those things can happen.
        """
        from backend.core import telemetry

        secret = "alice@example.com 123-45-6789 SELECT * FROM users"

        with pytest.raises(ValueError):
            with telemetry.span("execute"):
                raise ValueError(secret)

        spans = _spans_by_name("execute")
        assert len(spans) == 1
        sp = spans[0]

        # (a) no exception event recorded at all.
        assert len(sp.events) == 0

        # (b) status description is at most the type name -- never the exception message.
        assert sp.status.status_code == trace.StatusCode.ERROR
        assert sp.status.description == "ValueError"
        assert secret not in (sp.status.description or "")

        # (c) no field anywhere on the span (attributes, status, events, name) contains
        # the email, the SSN, or the SQL text -- individually and as the full secret.
        haystacks = [sp.name, str(sp.status.description)]
        haystacks += [str(v) for v in sp.attributes.values()]
        for event in sp.events:
            haystacks.append(event.name)
            haystacks += [str(v) for v in event.attributes.values()]
        blob = "\n".join(haystacks)
        for forbidden in ("alice@example.com", "123-45-6789", "SELECT * FROM users", secret):
            assert forbidden not in blob, f"span leaked {forbidden!r} via: {blob!r}"

    def test_telemetry_module_is_a_safe_noop_with_no_otel_sdk_configured(self) -> None:
        """`backend/core/telemetry.py` must be importable and usable even when nothing
        has ever configured an OpenTelemetry SDK provider in the process — this can only
        be proven in a FRESH interpreter (this test file's own module-level `trace
        .set_tracer_provider` call above has already permanently configured this
        process's provider, and OTel does not support un-setting it)."""
        script = textwrap.dedent(
            """
            from backend.core import telemetry

            with telemetry.span("gateway.handle", action="allow") as root:
                assert root.is_recording() is False
                with telemetry.span("execute") as child:
                    telemetry.set_span_attrs(child, rows_returned=3, latency_ms=5)
            print("OK")
            """
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout


# --------------------------------------------------------------------------------------
# Prometheus /metrics
# --------------------------------------------------------------------------------------


class TestMetricsEndpoint:
    @staticmethod
    def _client(db_path: str, **kwargs) -> TestClient:
        return TestClient(create_gateway_app(db_path, rate_limit_per_minute=1000, **kwargs))

    def test_metrics_is_valid_prometheus_text_with_only_unlabeled_safe_counters(self, db_path: str) -> None:
        client = self._client(db_path)
        secret_sql = "SELECT id, email FROM users WHERE email='alice@example.com' LIMIT 1"
        secret_actor = "agent-super-secret-identity"

        client.post("/query", json={"sql": secret_sql, "actor": secret_actor})
        client.post("/query", json={"sql": "DROP TABLE users", "actor": secret_actor})

        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        text = resp.text

        # No raw SQL, PII, or actor identifier anywhere in the exposed text.
        assert "alice@example.com" not in text
        assert secret_sql not in text
        assert "DROP TABLE" not in text
        assert secret_actor not in text

        # A real Prometheus parser must accept it, and every family/sample must be
        # exactly the safe shape this endpoint promises: an `icberg_`-prefixed name,
        # bucketed (a plain counter), and carrying NO labels at all. (`family.name`
        # strips the trailing `_total` a Counter's own sample name keeps -- e.g.
        # `icberg_queries_total`'s family is named `icberg_queries` -- so counter
        # VALUES are read off `sample.name`/`sample.value`, not the family name.)
        families = list(text_string_to_metric_families(text))
        assert families, "expected at least one metric family"
        samples = {s.name: s for f in families for s in f.samples}
        assert "icberg_queries_total" in samples
        assert "icberg_blocks_total" in samples
        for family in families:
            assert family.name.startswith("icberg_")
            assert family.type == "counter"
            for sample in family.samples:
                assert sample.labels == {}, f"unexpected label on {sample!r}"

        assert samples["icberg_queries_total"].value == 2
        assert samples["icberg_blocks_total"].value == 1

    def test_metrics_counters_are_isolated_per_app_instance(self, db_path: str) -> None:
        """Two `create_gateway_app` instances (as `tests/api/test_gateway_api.py`
        routinely builds within one test) must never share a `/metrics` count — see
        `telemetry.new_metrics_registry`'s docstring for why."""
        busy = self._client(db_path)
        quiet = self._client(db_path)

        busy.post("/query", json={"sql": "SELECT id FROM users WHERE id=1 LIMIT 1", "actor": "a"})
        busy.post("/query", json={"sql": "SELECT id FROM users WHERE id=1 LIMIT 1", "actor": "a"})

        busy_samples = {
            s.name: s for f in text_string_to_metric_families(busy.get("/metrics").text) for s in f.samples
        }
        quiet_samples = {
            s.name: s for f in text_string_to_metric_families(quiet.get("/metrics").text) for s in f.samples
        }

        assert busy_samples["icberg_queries_total"].value == 2
        assert quiet_samples["icberg_queries_total"].value == 0

    def test_rich_registry_tracks_labeled_action_counter_and_histograms_without_pii(self, db_path: str) -> None:
        """`telemetry.GatewayMetrics`'s RICH registry (never HTTP-exposed — see its
        docstring) is where the task's full instrumentation surface actually lives:
        a per-action-labeled `queries_total` and the latency/rows-returned histograms.
        Proven directly against `app.state.gateway_metrics`, not the constrained public
        `/metrics` route.
        """
        app = create_gateway_app(db_path, rate_limit_per_minute=1000)
        client = TestClient(app)

        client.post(
            "/query",
            json={"sql": "SELECT id, email FROM users WHERE email='alice@example.com' LIMIT 1", "actor": "agent-1"},
        )
        client.post("/query", json={"sql": "DROP TABLE users", "actor": "agent-1"})

        rich_text = app.state.gateway_metrics.generate_latest().decode("utf-8")
        assert 'icberg_queries_total{action="allow"} 1.0' in rich_text
        assert 'icberg_queries_total{action="block"} 1.0' in rich_text
        assert "icberg_query_latency_ms_bucket" in rich_text
        assert "icberg_rows_returned_bucket" in rich_text

        # Even the non-HTTP-exposed rich surface must never carry SQL/PII/actor values --
        # it only ever tracks bucketed counts and label values from the closed `action` set.
        assert "alice@example.com" not in rich_text
        assert "DROP TABLE" not in rich_text
        assert "agent-1" not in rich_text
