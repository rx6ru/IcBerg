# IcBerg

**A governance layer between AI agents and your database.**

An AI agent never gets a raw database connection or credentials — it only ever
*proposes* SQL text. IcBerg decides what happens to that proposal:

```
agent proposes SQL
      │
      ▼
  parse (sqlglot AST)
      │
      ▼
  classify (read / write / DDL — recursively, including CTEs & subqueries)
      │
      ▼
  policy decision (base gate + optional YAML policy)
      │
      ├── block  ──────────────────────────────────┐
      ├── hold   ── enqueue for human approval ─────┤
      └── allow  ── execute (least-privilege, RO) ──┤
                          │                          │
                          ▼                          │
                    redact PII from rows              │
                          │                           │
                          ▼                            │
                    append to hash-chained audit log ◄─┘
```

Destructive and out-of-scope statements (`DROP`, `TRUNCATE`, `ALTER`, an `UPDATE`/
`DELETE` with no `WHERE`, …) are blocked outright. Mutations with a `WHERE` clause are
held for a human to approve or reject. Bounded, scoped reads execute through an
engine-level least-privilege, read-only path, have PII redacted from the result, and —
on every path, block/hold/allow alike — are appended to a tamper-evident, hash-chained
audit log before a response goes back to the agent.

## 30 seconds

```python
from icberg import govern

govern("DROP TABLE customers", actor="agent-1", dsn="app.db")
# {"action": "block",
#  "reason": "DDL / destructive operations (CREATE, DROP, ALTER, TRUNCATE, SELECT INTO) are blocked.",
#  "matched_rules": ["ddl_blocked"], "rows": None, "audit_seq": 1, "approval_id": None}

govern("UPDATE orders SET status='refunded' WHERE id=42", actor="agent-1", dsn="app.db")
# {"action": "hold",
#  "reason": "Writes require human approval (Phase 2 approval queue); held.",
#  "matched_rules": ["write_requires_approval"], "rows": None, "audit_seq": 2,
#  "approval_id": "7e1c...-...-..."}
```

The `DROP` never reaches the database. The `UPDATE` sits in an approval queue until a
human calls `.approve(approval_id, approver=...)` — at which point the **exact** SQL a
human reviewed executes, never a re-derived or re-parsed version of it. Every one of
these decisions — including the block — is durably written to a hash-chained audit log.

## Quickstart

IcBerg is reachable three ways, all funneling through the identical decision path
(`Gateway.handle`).

### 1. Python SDK

```bash
pip install icberg
```

```python
from icberg import govern, governed_connection

# One-shot: govern a single statement.
result = govern("SELECT * FROM users WHERE id=1 LIMIT 5", actor="agent-1", dsn="app.db")
result["action"]  # "allow"
result["rows"]    # [{"id": 1, "name": "...", "email": "[REDACTED]", ...}]

# Reusable, stateful handle — audit trail and approval queue persist across calls.
db = governed_connection("app.db", policy="policy.yaml")
db.query("UPDATE orders SET status='shipped' WHERE id=1", actor="agent-1")  # -> hold
db.pending_approvals()
db.approve(approval_id, approver="alice")
db.audit_tail(20)
```

`GovernedConnection` deliberately never exposes a raw executor or database connection —
`.query()`/`.approve()`/`.reject()`/`.pending_approvals()`/`.audit_tail()` is the entire
public surface. For callers who want to build their own wiring instead of the
convenience helpers, the core primitives (`Gateway`, `GovernanceGate`, `PolicyDecision`,
`Policy`, `load_policy`) are re-exported from the same `icberg` package.

### 2. Self-hostable service

A REST + SSE API in front of a governed database — no change to the database itself.

```bash
docker compose up --build -d

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"sql": "DROP TABLE customers", "actor": "agent-1"}'
# {"action":"block","reason":"DDL / destructive operations (CREATE, DROP, ALTER, TRUNCATE, SELECT INTO) are blocked.","matched_rules":["ddl_blocked"],"rows":null,"audit_seq":1,"approval_id":null}
```

Other endpoints on the same service: `POST /query/stream` (the same decision as SSE
events), `GET /approvals` / `POST /approvals/{id}` (list / approve / reject a held
write), `GET /audit` (redacted trail + hash-chain verification status), `GET /health`,
`GET /metrics` (Prometheus counters).

### 3. MCP server

Any MCP-capable agent (Claude Desktop, Cursor, a custom client) gets governed database
tools instead of raw credentials. Copy the `icberg` entry from
[`examples/claude_desktop_mcp_config.json`](examples/claude_desktop_mcp_config.json)
into Claude Desktop's `claude_desktop_config.json` (`Settings → Developer → Edit
Config`), pointing `ICBERG_MCP_DSN` at your database.

The server exposes three governed tools, thin wrappers over the same
`GovernedConnection` the SDK uses:

- `query(sql)` — propose one statement; governed end to end.
- `list_pending_approvals()` — every pending, not-yet-expired write approval.
- `audit_tail(n)` — the last `n` entries of the tamper-evident audit trail.

Approving a held write is **deliberately not an MCP tool** — a proposing agent must not
be able to approve its own write. Approval happens out of band via the SDK
(`.approve(id, approver=...)`), the REST endpoint (`POST /approvals/{id}`), or a human
review UI.

## Features

- **Fail-safe, default-deny policy gate** — SQL is parsed to an AST (`sqlglot`) and
  recursively classified as read/write/DDL, including statements hidden inside a CTE or
  subquery; an unparseable or unclassifiable statement blocks, it never falls through to
  allow.
- **Human-in-the-loop approval workflow** — a held write is enqueued immutably; approval
  executes the exact stored SQL via an atomic claim-before-execute transition, so two
  concurrent decisions on the same approval can never both proceed, and self-approval is
  refused.
- **Least-privilege execution** — SQLite reads run through an OS-level read-only file
  open (`mode=ro`), not just a mutable session flag; every read is bounded by a forced
  row cap and a dual-layer (in-process + process-isolated) statement timeout.
- **PII redaction** — column-name classification, value-pattern scanning, and
  schema-aware provenance tracing through views and renamed columns, applied to every
  `allow` result before it leaves the gateway.
- **Tamper-evident audit log** — every decision (allow, block, or hold) is appended to a
  hash chain covering every field plus the previous entry's hash; append-only database
  triggers and an external anchor file for the chain head detect edits and full
  self-consistent rewrites alike.
- **Optional YAML policy** — a per-deployment `Policy` can deny-list tables, force
  approval on writes, cap row counts, or add PII column keywords — strictly monotonic:
  it can only further restrict the base gate's decision, never loosen it.
- **Multi-database connectors** — SQLite (fully hardened), Postgres, and MySQL behind a
  uniform `Connection` interface, with network-backend drivers imported lazily.

## Security posture

IcBerg is built as defense-in-depth, not a single choke point — each control (dangerous-
SQL detection, the read-only execution boundary, PII redaction, audit integrity, the
approval workflow) is backed by at least one independent layer beneath it. See
[`THREAT_MODEL.md`](THREAT_MODEL.md) for the full per-control breakdown.

Two things worth stating plainly rather than implying:

- **Least-privilege database GRANTs are the authoritative PII control, not redaction.**
  `redact_rows`'s name/value/provenance layers are a safety net over an `allow` result —
  name- and pattern-based, not exhaustive. The control a production deployment must
  actually rely on is a read-only database role, connected through by the gateway, that
  has no `SELECT` grant on PII columns in the first place. Redaction catches what that
  role is still granted; it does not substitute for scoping the grant correctly.
- **There is no built-in authentication yet.** `actor` (on proposing a query) and
  `approver` (on deciding a held one) are self-asserted strings, not verified
  identities. Self-approval (`approver == actor`) is refused as an in-process stopgap,
  but a different claimed identity is indistinguishable from a real one. `POST
  /approvals/{id}` and the other mutating endpoints must sit behind real authentication
  and authorization before being exposed beyond a trusted internal network.

SQLite has the fully engine-enforced read-only boundary (`mode=ro`, OS file-descriptor
level); Postgres and MySQL currently rely on mutable session-level read-only settings —
for those backends, a least-privilege connecting role is the primary control, not a
defense-in-depth nicety. Full detail, including the specific bypasses each layer was
added to close, is in `THREAT_MODEL.md`.

## Testing

500+ tests, including a dedicated red-team/adversarial suite under `tests/security/`
covering SQL-injection shapes, dangerous-function detection (RCE/sequence-mutation/DoS
primitives), parser-differential bypasses, redaction leak classes, and audit-tampering
scenarios — alongside unit, integration, contract, and API test suites.

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — component breakdown, request flow, defense-in-
  depth layering, and integration surfaces.
- [`THREAT_MODEL.md`](THREAT_MODEL.md) — assets, adversary model, trust boundary, full
  risk catalog, control↔risk mapping, and phase-by-phase residual-risk honesty sections.

## Requirements

Python 3.11–3.12. `docker compose up` requires Docker Compose v2; no external services
are required to start the gateway itself (it defaults to a local SQLite-backed database).
