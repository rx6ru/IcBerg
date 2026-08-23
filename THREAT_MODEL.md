# IcBerg — Threat Model (Phase 0 + Phase 1)

**Scope:** the SQL governance gate (`backend/core/sql_governance.py`) that sits between an
AI agent and a real database, PLUS the Phase 1 execution boundary it is now wired into:
a least-privilege **read-only executor** (`backend/core/executor.py`), column- and
value-level **PII redaction** of result rows (`backend/core/redaction.py`), and a
hash-chained, tamper-evident **audit** log (`backend/core/audit.py`), composed by
`backend/core/gateway.py`. Phase 0's decision layer (parse -> classify -> policy) is
unchanged; what changed is that an `allow` decision now actually executes through an
engine-enforced read-only boundary instead of nothing at all, and every decision —
allow, block, or hold — is now durably audited. Approval-workflow execution for `hold`
decisions and Postgres-role enforcement remain Phase 2 scope.

**Phase 1 implemented controls (see "Phase 1 — Implemented Controls" below for detail):**
- Engine-level least-privilege: reads execute through a **read-only** SQLite connection
  (`file:<path>?mode=ro`), not merely a session-level flag a statement could revert.
- Column-name + value-pattern **PII redaction** of every `allow` result before it leaves
  the gateway.
- A hash-chained, append-only **audit** log recording every proposal's decision, whether
  or not it executed.

---

## Assets

What we are protecting, in priority order:

1. **Data integrity** of the connected database(s) — rows, schemas, and constraints must
   not be corrupted or destroyed by an agent's mistake or manipulation.
2. **Data confidentiality** — customer PII, credentials, and any column an operator has
   not explicitly scoped for agent access must not be exfiltrated through query results.
3. **Database availability** — the database must stay responsive for other tenants/users;
   a single agent session must not be able to degrade or take down the instance.
4. **Host integrity** — the database server process and its underlying filesystem/OS must
   not become an execution target via DB-native features (e.g. `COPY … TO PROGRAM`,
   large-object/file functions).
5. **Auditability** — a trustworthy record of what was proposed, what was decided, and why,
   so incidents are investigable after the fact (Phase 1 delivers the hash-chained log;
   this phase only guarantees every decision carries a `reason` and `matched_rules`).

## Adversary

The threat model treats the **proposed SQL text as untrusted input**, regardless of who or
what generated it:

- **A prompt-injected agent.** The agent's LLM backend can be steered by content it reads
  (retrieved documents, tool outputs, even prior conversation turns) into emitting SQL the
  operator never intended. The agent believes it is being helpful; the resulting SQL is
  adversarial.
- **A confused / hallucinating agent.** No malicious intent required — an LLM can simply
  generate a destructive or overly broad statement (`DELETE FROM users` with no `WHERE`)
  because it "sounds right" for the requested task.
- **A malicious end user** who crafts natural-language input specifically to manipulate the
  agent into emitting an injection payload, a stacked statement, or a system-catalog query,
  using the agent as a confused deputy against the database.
- **A compromised or buggy upstream component** (a tool, a plugin, another service in the
  agent's chain) that emits SQL directly.

Critically: **the agent is "ours" (a component we built and, to a degree, trust) but the
SQL string it produces is never trusted.** Trust in the agent's intent does not transfer to
trust in its output. This is the same posture as treating any LLM completion as untrusted
user input per OWASP LLM Top 10 (LLM01: Prompt Injection).

We explicitly do **not** model a human operator with direct database credentials — that
actor bypasses the gateway entirely and is out of scope for this control.

## Trust Boundary

```
   [ AI Agent ]                    [ IcBerg Gateway ]                  [ Database ]
   untrusted SQL text  ────────►   parse → classify → policy   ────►   trusted execution
   no DB credentials               decide (allow/block/hold/redact)    path only
                                    │
                                    ▼
                              audit + reason
```

- The agent **proposes** a SQL statement (a string). It never receives, stores, or uses raw
  database credentials — it cannot reach the database except through the gateway.
- The gateway is the **sole holder of DB credentials** and the **sole decision point**. It
  parses the proposal to an AST, classifies it, applies policy, and only then decides
  whether to allow, block, hold (for human approval — Phase 2), or redact.
- The gateway **executes** on the agent's behalf; the agent never gets a live connection or
  a raw cursor. (Actual bounded execution — read-only transactions, timeouts, row caps —
  ships in Phase 1; Phase 0 only implements the decision function.)
- Everything crossing left-to-right (agent → gateway) is untrusted. Everything the gateway
  hands back (decision, redacted results) is what the agent is allowed to see.

This boundary is the entire point of the product: an agent that can construct arbitrary SQL
is safe only if something on the trusted side of the line is willing to say no.

## Risk Catalog

| ID | Risk | Example |
|----|------|---------|
| R1 | **Destructive writes** | `DROP TABLE users`, `TRUNCATE orders`, `ALTER TABLE users DROP COLUMN email` |
| R2 | **Unscoped mutation** | `DELETE FROM users` / `UPDATE users SET admin=true` with no `WHERE` clause — affects every row |
| R3 | **PII / data exfiltration** | `SELECT *` pulling unreviewed columns; broad unfiltered scans |
| R4 | **System catalog / metadata scraping** | `SELECT * FROM pg_catalog.pg_user`, `information_schema.columns` — schema and credential-adjacent reconnaissance |
| R5 | **SQL injection via string construction** | `... WHERE name='' OR 1=1 --` — tautology defeats an intended filter; trailing `--`/`/* */` comments truncate an appended clause |
| R6 | **Stacked / multi-statement injection** | `SELECT 1; DROP TABLE users` — a second, unreviewed statement riding along with a benign one |
| R7 | **Denial of service** | `pg_sleep(30)`, unbounded scans, cartesian joins — ties up connections/CPU |
| R8 | **DB-native remote code execution** | `COPY t TO PROGRAM 'sh -c ...'`, `lo_import`/`lo_export`, `pg_read_file` — Postgres features that reach the host filesystem or shell |
| R9 | **Parser/classifier bypass (unknown syntax)** | Malformed, dialect-specific, or adversarially obfuscated SQL that the parser cannot classify, used to smuggle intent past pattern checks |
| R10 | **Read-only boundary escape via ATTACH** | `ATTACH DATABASE '/path/x.db' AS x` opens a second, non-read-only database file under an already-open connection — the attached file does not inherit `mode=ro`, so this both creates/overwrites an arbitrary filesystem path and grants write access to it, from inside what was supposed to be a read-only session |
| R11 | **Session/GUC mutation via a plain SELECT** | `SELECT set_config('statement_timeout', '0', false)` / `SELECT pg_reload_conf()` — parses as an ordinary, single, bounded-looking `SELECT` but mutates session or server configuration as a side effect, including disabling the very statement timeout this gateway's least-privilege executor relies on |

## Control ↔ Risk Map

| Control (in `GovernanceGate`) | Mitigates | Notes |
|---|---|---|
| Root-node classification (Select/Insert-Update-Delete/Create-Drop-Alter-Truncate) | R1, R2 | Establishes `read`/`write`/`ddl`/`unknown` before any other rule runs |
| DDL blanket block (`DROP`, `TRUNCATE`, `ALTER`) | R1 | Fail-closed: no DDL is ever `allow` in v1 |
| `WHERE`-clause requirement on `DELETE`/`UPDATE` | R2 | Missing `WHERE` → `block`; present `WHERE` → `hold` (write approval is Phase 2, not yet auto-executed) |
| Multi-statement rejection (`sqlglot.parse` returns >1 statement) | R6 | Any batch with more than one parsed statement is blocked outright |
| Tautology / comment-truncation string checks | R5 | Regex over the raw text for `OR 1=1`-style tautologies and trailing `--`/`/* */` |
| System catalog / `pg_*` name matching (AST table names + string fallback) | R4 | Blocks `pg_catalog.*`, `information_schema.*`, and any table prefixed `pg_` |
| `COPY … TO PROGRAM` / large-object function detection (generic + `postgres` dialect parse) | R8 | Generic dialect often can't model Postgres `COPY`, so a second parse attempt with `dialect="postgres"` plus a string fallback catches it |
| `SELECT *` flag (`pii_suspected` in `matched_rules`, still `allow` in v1) | R3 | Detection only in Phase 0; hardening (column-level classification, forced redaction) is Phase 1 |
| Fail-safe default deny on parse failure/empty parse | R9 | Any unparseable or empty statement is `block`, classification `unknown` — never falls through to `allow` |
| `LIMIT` + explicit column list as a positive signal for `allow` | R3, R7 (partially) | Bounded, scoped `SELECT`s are the only statements eligible for `allow` in v1 |
| **Read-only executor** (`ReadOnlyExecutor`, engine-enforced `file:<path>?mode=ro`) | R1, R2, R9 (containment) | Phase 1. Runs *underneath* the gate as defense-in-depth: even if the string-level policy gate above is bypassed by a future parser-differential (see the `U&"..."` case below), a forced write against this connection is rejected by SQLite itself (`attempt to write a readonly database`) before any row is touched — a gate bypass degrades into "the read-only connection's own limits," not full database access |
| **Forced row cap + wall-clock timeout** (`ReadOnlyExecutor`, watchdog thread + `Connection.interrupt()`) | R7 | Phase 1. Applies to every read regardless of what `LIMIT` (or lack of one) the proposed SQL contains; a recursive-CTE / cartesian-join row explosion is aborted mid-execution, not run to completion |
| **PII redaction** (`redact_rows`: column-name classification + value-pattern scan) | R3 | Phase 1. Every `allow` result is redacted before it leaves the gateway — closes the "`SELECT *` is flagged but not blocked" gap noted below, for the columns/patterns the classifier and regexes cover (see residual gap) |
| **Hash-chained audit log** (`AuditLog.append`/`verify`) | Asset #5 (auditability) | Phase 1. Every decision — allow, block, or hold — is appended, whether or not it executed; `verify()` detects tampering with any stored field of any past entry, an append-only trigger rejects a direct `UPDATE`/`DELETE`, and an external anchor file catches a full, internally-self-consistent chain rewrite (v2 hardening — see below) |
| **ATTACH/DETACH block** (gate rule `attach_blocked` + `executor.py`'s SQLite authorizer) | R10 | v2 hardening. Blocked at the gate (raw-text-level, unconditional — sqlglot's generic/postgres dialects don't even parse this syntax) AND at the read-only connection itself (`SQLITE_ATTACH` denied via `conn.set_authorizer`), so a gate bypass still can't open a writable second database file |
| **GUC/session-mutation function block** (gate rule `set_config_blocked`) | R11 | v2 hardening. `set_config`/`pg_catalog.set_config`/`pg_reload_conf` added to the same AST-based function-name deny list as the RCE/sequence/DoS functions, so quoting/casing/schema-qualification can't evade it |
| **Provenance (lineage) redaction** (`redaction.py`, layer 3) | R3 | v2 hardening. Traces each SELECT output expression to its source column(s) via `sqlglot`; redacts the output even when aliased/wrapped (`SUBSTR(ssn,1,3) AS s`), closing a gap the name- and value-pattern layers alone missed |
| **Schema-aware view inlining** (`redaction._inline_views`, `backend/core/schema_catalog.py`) | R3 | v4 hardening. `Gateway.handle` introspects the executor's live database (tables/columns + view definitions) and feeds it into `redact_rows`; a known view referenced in `FROM`/`JOIN` is inlined with its own real `SELECT` body before lineage tracing, closing the confirmed leak where a PII column renamed through a view (e.g. `ssn_num AS token`) was invisible to a query-text-only analysis. Optional/lazy — `schema=None` (no live connection) is unchanged v1-v3 behavior; an unresolvable view (unparseable/cyclical, including one hidden behind another view) fails the whole query's result closed rather than passing anything through |
| **JSON/BLOB defaulting** (`redaction.py`, layer 4) | R3 | v2 hardening. A `bytes`/BLOB value or a string that parses as a JSON object/array defaults to redacted, since none of the other layers can see into either |
| **Process-isolated hard timeout** (`executor.py`, `ReadOnlyExecutor.execute`) | R7 | v2 hardening. The in-process `interrupt()` watchdog (v1) is a race against whatever the query is blocked on; execution now additionally runs in a forked child process with `multiprocessing.Process.join(timeout=...)`, and the parent kills the child outright (`SIGKILL`) if it hasn't reported back — an OS-level guarantee no blocking call, crash, or watchdog race can defeat |

## Phase 1 — Implemented Controls

Phase 0's decision layer is a pure function: it never opened a connection, so a bypass of
its policy (or a caller that skipped it entirely) had nothing underneath to contain it.
Phase 1 closes that gap with three controls, composed by `backend/core/gateway.py`:

1. **Engine-level least-privilege execution** (`backend/core/executor.py`,
   `ReadOnlyExecutor`). The `allow` path executes through a SQLite connection opened with
   `sqlite3.connect("file:<path>?mode=ro", uri=True)` — the file descriptor itself is
   opened read-only at the OS level. `PRAGMA query_only = ON` is set too, but only as a
   secondary layer; the real boundary is `mode=ro`, specifically because a session flag is
   mutable in a way a file-open mode is not. A forced write against this connection is
   rejected by SQLite with `attempt to write a readonly database` — captured as
   `ExecutionResult.error`, never retried against a write path, never partially applied.
   Every read is additionally bounded by a forced row cap (`MAX_ROWS`, default 1000,
   independent of any `LIMIT` in the proposed SQL) and a wall-clock timeout (default 5s)
   enforced by a watchdog thread calling `Connection.interrupt()` — the only way to abort
   a query already blocked deep inside SQLite's VDBE (e.g. mid-recursion in a recursive
   CTE), since a plain Python-level timeout cannot preempt a blocking C call. A separate
   `WriteExecutor` exists for Phase 2's approved-write flow; the gateway's `allow`/read
   path never uses it.
2. **PII redaction** (`backend/core/redaction.py`, `redact_rows`). Every `allow` result
   passes through two layers before reaching the caller: column-name classification
   (`email`, `mail`, `ssn`, `social`, `phone`, `mobile`, `card`, `credit`, `dob`, `birth`,
   `address` — case-insensitive substring match) masks every non-empty value in a matching
   column outright; a value-pattern scan (reusing `guardrails.py`'s output-scrubbing
   regexes, plus an SSN pattern this module adds) additionally catches PII sitting in a
   column whose *name* carries no signal at all (an aliased/aggregated expression, e.g. a
   bare `col1`). Plainly non-PII columns/values (`id`, `admin`, `age`) are left untouched by
   design — over-redacting would make the gateway useless for ordinary analytics.
3. **Hash-chained audit log** (`backend/core/audit.py`, `AuditLog`). Every decision the
   gateway makes — `allow`, `block`, or `hold` — is appended as an `AuditEntry`
   (`actor`, `proposed_sql`, `classification`, `action`, `matched_rules`, `rows_returned`,
   `latency_ms`, `result_hash`, `prev_hash`, `entry_hash`, `timestamp`), whether or not
   anything executed. `entry_hash` is a sha256 over the entry's canonical (sorted-key) JSON
   plus `prev_hash`, which is itself the previous entry's `entry_hash` (genesis chains to
   `"0" * 64`) — mutating any stored field of any past entry breaks that entry's own hash
   and every `prev_hash` link after it. `verify()` walks the whole chain and returns the
   first sequence number where either check fails, so an investigation knows exactly where
   tampering started. `result_hash` is computed over the *redacted* rows, never the raw
   ones, so the audit trail can prove/disprove what was returned without itself becoming a
   second at-rest copy of raw PII.

**Honest residual gaps in these three controls specifically** (in addition to the
general residual risks below, most of which Phase 1 does not change):

- **Redaction is a best-effort classifier, not a data-classification system — RESOLVED
  (v2) for aliased/derived output columns; the underlying value-visibility gap remains.**
  v1 redacted by output column *name* and by scanning output *values* against a fixed
  regex set; v2 adds a third layer that traces each SELECT output expression back to the
  *source* column(s) it references via `sqlglot` (see "Phase 1 — Implemented Controls"
  above), so `SUBSTR(ssn,1,3) AS s`/`GROUP_CONCAT(email) AS c` are now caught even though
  neither the output name nor the transformed value necessarily still looks like an
  SSN/email. What remains open regardless of layer count: a column named something
  outside the keyword list, fed by a source column *also* outside the keyword list,
  holding a value that doesn't match any of the value-level regexes — a free-text `notes`
  field containing a spelled-out address, a non-US phone/national-ID format the regexes
  don't model, or any PII expressed in natural language rather than a recognizable
  pattern — still passes through unredacted. This is the same class of gap
  `guardrails.py`'s regexes already have for LLM output; nothing in Phase 1 (v1, v2, or
  v3) replaces pattern/name matching with real data classification (no NER/ML-based PII
  detection, no per-deployment configurable column allow/deny lists). Treat every
  redaction guarantee in this document as "known US-centric structured-PII patterns and
  known-PII-named columns," not "all PII."
  **v3 update — generalized fail-closed provenance, converging the whack-a-mole.** v2's
  provenance layer closed four confirmed leak *shapes* one at a time (derived-`SELECT *`,
  ambiguous-JOIN alias, derived-star-plus-base-table JOIN, scalar-subquery-beside-star),
  but each fix was shape-specific — a fifth shape (an ambiguous, unqualified column
  reference beside a top-level star, among a derived-source+base-table JOIN: `SELECT *,
  s AS renamed FROM (SELECT ssn_num AS s FROM users) a JOIN orders o ON o.user_id=1
  WHERE o.id>0 LIMIT 5`) still leaked a raw SSN through `_column_is_pii`'s deliberately
  permissive ambiguous-column fallback. v3 (`_query_references_pii_source` +
  `_column_is_pii`'s `strict` parameter, `redaction.py`) generalizes instead of patching
  a sixth shape: whenever the query references a PII-named column *anywhere* (any scope,
  subquery, or CTE — not just among the output columns), that permissive fallback is
  removed, and ANY top-level output column this analysis cannot positively trace to a
  proven-non-PII base column is redacted — closing the whole leak *class*, current and
  future shapes alike, at the cost of some fail-safe over-redaction confined to queries
  that already touch a PII-named column. A query that references no PII column anywhere
  keeps the prior, more permissive behavior unchanged (no new over-redaction there). This
  remains, explicitly, a **static SQL-shape analysis** — best-effort defense-in-depth on
  top of the real control, not a substitute for it: see "Residual Risks" below for why
  least-privilege column grants, not this classifier, are the authoritative boundary for
  PII in production.
  **v4 update — schema-aware view inlining, closing the last allow-path leak class v1-v3
  couldn't reach at all.** v1-v3's provenance layer parsed only the *proposed query's own
  text* — a `VIEW` was an opaque base-table reference to it, indistinguishable from an
  ordinary table with unknown schema. `CREATE VIEW vnum AS SELECT id AS uid, ssn_num AS
  token FROM users; SELECT token FROM vnum WHERE uid=1` traced `token` nowhere at all: no
  PII `exp.Column` appears anywhere in the *query text* (only inside the view's own,
  unparsed definition), so `_query_references_pii_source` was `False`, and a bare 9-digit
  int (`ssn_num`) has no dashes/keyword-prefix for the value-pattern layer to catch either
  — this was a confirmed real leak against the actual `Gateway.handle`, not a theoretical
  gap. v4 closes it by giving `redact_rows` an optional, best-effort `schema` parameter (a
  live `SchemaCatalog` — `backend/core/schema_catalog.py`) that `Gateway.handle`
  introspects from the executor's own connection (`ReadOnlyExecutor.get_schema_catalog`:
  `PRAGMA table_info` for every table, `sqlite_master.sql` for every view's defining
  `SELECT`) before redacting. When present, `redaction._inline_views` replaces every
  `FROM`/`JOIN` reference to a known view — recursing for a view built on another view —
  with that view's own parsed body as a derived subquery, BEFORE lineage tracing and the
  PII-anywhere check both run; from that point on a view is, to the rest of this module's
  pre-existing lineage machinery, indistinguishable from an ordinary subquery-in-`FROM`,
  needing no view-specific redaction logic at all. The same `schema` also supplies real
  table/column definitions to `sqlglot`'s `qualify(schema=...)`, retiring the schema-less
  fallback heuristics (`_derived_pii_output_columns`, the permissive ambiguous-column
  classification) for exactly the queries where real schema is available. **What v4 does
  NOT change:** `schema=None` (no live connection — e.g. a caller that hand-builds rows
  without going through `Gateway.handle`) is a complete no-op, identical to pre-v4
  behavior — the fix is additive, not a replacement for the query-text-only analysis.
  A view whose stored definition this module cannot resolve (an unparseable body, a
  `UNION`-shaped view, or a cyclical view chain — including one hidden two-or-more layers
  deep behind an otherwise-resolvable view) fails the WHOLE query's result set closed
  (every output column redacted) rather than attempting to bound which specific columns
  it could have tainted; adversarial retesting during this fix's own development found
  and closed two real implementation gaps in this path before it was considered complete
  — a view name containing uppercase characters not matching the schema catalog's own
  (differently-cased) key, and an unresolvable view referenced only transitively through
  another, resolvable view's own body, both of which silently fell open (no redaction,
  no "unresolved" flag either) until fixed; both are now regression-tested in
  `tests/security/test_governance_runtime.py`. **What remains a genuine, honest gap even
  after v4:** (1) schema introspection is SQLite-only and tested only against SQLite —
  the Postgres path (`PostgresReadOnlyExecutor.get_schema_catalog`) is a documented stub
  returning `None`, so a Postgres-backed deployment gets v1-v3 behavior only until that is
  implemented against a live fixture; (2) introspection opens its own short-lived
  connection to the same file at call time, immediately after execution — a view/table
  DDL change landing in the narrow window between the two (an unusual, adversarial-timing
  scenario for a typically-read-only analytics database) could in principle redact against
  a schema slightly different from the one the already-executed query actually ran
  against; (3) the view-inlining match is by NAME only (case-insensitively, schema- or
  CTE-qualification aware) — it does not verify that the connecting database ROLE would
  itself have had privilege to read the view's underlying source directly, which is,
  again, why least-privilege column/view GRANTs at the database layer (not this
  classifier) remain the authoritative production boundary, stated in this document's
  very first residual-risk bullet and unchanged by this update.
- **The audit log is tamper-*evident*, not tamper-*proof* — RESOLVED (v2) for a
  single-field edit and for a full self-consistent chain rewrite; still not a WORM or
  signed store.** v1's `verify()` caught any single stored field being edited (a broken
  `entry_hash`/`prev_hash` link) but had no defense against an attacker with enough access
  to rewrite *every* downstream hash to make a doctored chain internally consistent again.
  v2 closes both remaining gaps in the SQLite-table itself and adds an independent check:
  `BEFORE UPDATE`/`BEFORE DELETE` triggers (`RAISE(ABORT, ...)`) make the table
  append-only against ordinary DML, and every `append()` writes the new chain head —
  `{seq, entry_hash}` — to a *separate* anchor file outside the table entirely, which
  `verify()` checks the recomputed head against as its final step. An attacker who
  rewrites the whole chain to be internally self-consistent still leaves the *old* head in
  the untouched anchor, so `verify()` still returns `ok=False` (see
  `TestAuditAnchor.test_audit_anchor_detects_full_chain_rewrite` in
  `tests/security/test_governance_runtime.py` for a from-scratch demonstration). This is
  **still not tamper-proof**: both defenses require ordinary filesystem/SQLite access to
  bypass, and an attacker with enough privilege to also rewrite the anchor file (or
  `DROP TRIGGER` first, as the regression tests themselves do to simulate this) defeats
  them. Neither is a WORM (write-once) storage backend, and there is no asymmetric
  signature binding entries — or the anchor itself — to the gateway's identity, so nothing
  here proves the anchor file wasn't *also* regenerated by whoever regenerated the chain,
  only that regenerating both consistently is more work than regenerating one. Genuine
  tamper-proofness needs an anchor location the same attacker can't reach (a separate
  system, a notarization service, a signed/publicly-anchored digest, or real WORM/append-
  only storage) — that remains future work, not claimed as done here.
- **`proposed_sql`/`result_hash` PII exposure at rest — RESOLVED (v2).** v1 stored
  `proposed_sql` verbatim, so a literal `WHERE email='alice@example.com'` in a proposed
  statement persisted raw in the audit log regardless of what the *result* redaction did —
  a second, distinct place raw PII was at rest. v2's `AuditLog.append` scrubs
  `proposed_sql` through `redaction.redact_text` before storing it, unconditionally, at
  the single write path this class exposes. `result_hash` is also now computed with a
  random per-`AuditLog`-instance salt (`AuditLog.hash_result`, persisted once in a
  metadata table) rather than a bare content hash, so it is not a raw-value preimage — a
  small/guessable result set (e.g. "is `admin` 0 or 1 for user 7") can no longer be
  recovered by dictionary-hashing candidate row sets and comparing to the stored digest.
  This does not make the salt secret against an attacker with full database access (it
  lives in the same SQLite file it salts); it defeats offline dictionary/rainbow-table
  matching by someone who only has the stored `result_hash` values, not someone who also
  has the database.
- **The `hold` path is still not executed under any control.** Phase 1 audits `hold`
  decisions but still does not execute them under any approval workflow — that remains
  Phase 2 scope, unchanged from Phase 0's residual-risk note below.
- **SQLite only; Postgres least-privilege is not enforced yet.** `PostgresReadOnlyExecutor`
  in `executor.py` is a stub shaped for a read-only-role + `SET TRANSACTION READ ONLY` +
  `SET statement_timeout` connection, but it has no test coverage in this repo (no Postgres
  fixture yet) and `psycopg` is an optional, lazily-imported dependency. The parser-
  differential risk this least-privilege layer exists to contain (see the `U&"..."`
  Unicode-escape case documented below) is specifically a Postgres-dialect risk — so the
  engine-level containment for it is, as of Phase 1, proven only against SQLite's
  `mode=ro`, not yet against an actual Postgres read-only role.

## Residual Risks

Being honest about what this control **does not** fully solve, by design or by phase scope.
Three headline honesty statements, expanded in detail (with the specific v1→v2→v3 change
history) in "Honest residual gaps in these three controls specifically" above:

- **Redaction is a best-effort classifier, not a data-classification system — and static
  SQL-shape analysis specifically is best-effort defense-in-depth, not the authoritative
  control.** Name-, value-pattern-, and (as of v2) provenance/lineage-based matching
  against known US-centric PII shapes and known-PII-named columns — not a guarantee
  against all PII. Free-text PII (a spelled-out address in a `notes` field, PII expressed
  in prose) and non-US phone/national-ID formats the regexes don't model can still slip
  through unredacted. As of v3 (the generalized ambiguous-column fix, `redaction.py`'s
  `_query_references_pii_source`/`strict`), the provenance layer now fails **closed**:
  whenever a query references a PII-named column anywhere, any top-level output column
  this best-effort SQL-shape analysis cannot *positively* trace to a proven-non-PII base
  column is redacted, rather than passed through on an unresolved/ambiguous lineage —
  closing the whole class of leaks six rounds of shape-specific patches were chasing one
  query shape at a time. v4 (schema-aware view inlining — see the v4 update above) closes
  a further class this static text-only analysis structurally could not: a PII column
  renamed through a `VIEW`, which is opaque to any purely-textual parse of the *query*
  alone. This raises the floor; it does not change what the classifier
  fundamentally is: parsing SQL structure and column names/values, not enforcing a real
  data boundary. **The authoritative production control for PII is least-privilege
  column grants at the database layer** — deploy the read-only role this gateway
  connects through WITHOUT `SELECT` on PII columns (or, short of real per-column GRANTs,
  restrict the queryable column set using the same PII-column denylist this module
  already classifies by, `redaction.py`'s `_PII_COLUMN_KEYWORDS`), so PII never reaches
  the gateway process at all, in rows or in query-plan side channels this SQL-shape
  analysis cannot see. Treat `redact_rows`'s output as a convenience/backstop layer for
  whatever PII columns the connecting role *is* granted — a safety net catching what
  reaches the gateway, not the boundary that should be relied on to keep it out.
- **The audit log is tamper-*evident*, not tamper-*proof*.** As of v2 it is append-only
  (`BEFORE UPDATE`/`BEFORE DELETE` triggers) and anchored externally (`{seq, entry_hash}`
  written to a separate file after every append, checked by `verify()`), which together
  catch a single-field edit, an internally-inconsistent chain, and a full self-consistent
  chain rewrite. None of that is a WORM (write-once) store or an asymmetric signature
  binding entries to the gateway's identity — an attacker with enough filesystem/SQLite
  privilege to rewrite the chain *and* the anchor file (or drop the triggers first) still
  defeats it. Real tamper-proofness needs an anchor location that attacker can't reach.
- **The Postgres executor path is unproven without a live fixture.** `PostgresReadOnlyExecutor`
  in `executor.py` is a stub shaped for a read-only role + `SET TRANSACTION READ ONLY` +
  `SET statement_timeout`, but there is no Postgres fixture in this repo to run it against —
  it has zero test coverage here, unlike `ReadOnlyExecutor`'s SQLite path, which every gate
  in `.devdocs/PHASE1_GATES.md` (P1.1–P1.17) exercises for real. Treat it as unvalidated
  design, not a proven control, until it has been.

- **`SELECT *` is still `allow`, not `block` — RESOLVED (Phase 1) for the redaction half.**
  Phase 0 only flagged it (`pii_suspected` in `matched_rules`); it did not classify columns
  or redact results. As of Phase 1, `redact_rows` (see "Phase 1 — Implemented Controls"
  above) masks classified PII columns and pattern-matched values in every `allow` result,
  including `SELECT *`. The gate decision itself is unchanged (`SELECT *` is still `allow`,
  not `block`) — what changed is that the *result* is no longer returned raw. Coverage is
  name/pattern-based, not exhaustive; see the redaction residual gap above.
- **DoS via expensive-but-syntactically-safe queries is not caught.** A bounded `SELECT …
  WHERE … LIMIT 50` can still be arbitrarily expensive to *plan and execute* (e.g. against
  an unindexed column, or a cartesian join hidden behind a `WHERE` that doesn't actually
  restrict the join). Phase 0 has no cost/`EXPLAIN` guard — that lands in Phase 1.
  `pg_sleep()` specifically is not pattern-matched in v1 and would currently pass as a
  syntactically valid, bounded-looking `SELECT`.
- **Held writes are not yet executed under any additional control.** `hold` in this phase
  is purely a decision value — there is no approval queue, no read-only-transaction
  wrapper, and no execution path at all yet. The gateway currently only decides; Phase 1/2
  build the execution and approval machinery this decision is meant to gate.
- **String-based detection (tautologies, catalog names, `COPY…TO PROGRAM`) is inherently
  incomplete.** Regex/string heuristics supplement the AST but can be evaded by sufficiently
  obfuscated or dialect-specific SQL (e.g. encoded literals, unusual whitespace, vendor
  extensions the parser doesn't recognize). `sqlglot` parsing raises the bar significantly
  over naive string matching, but it is not a formal proof of safety — a sufficiently novel
  bypass should be assumed to exist until red-teamed.
- **No connection-level enforcement yet — RESOLVED (Phase 1) for the gateway's own path.**
  The `GovernanceGate` itself is still a pure decision function with no enforcement of its
  own; what changed is that `backend/core/gateway.py`'s `Gateway.handle` now always routes
  an `allow` decision through `ReadOnlyExecutor` (see "Phase 1 — Implemented Controls"
  above), so a caller going through the gateway gets engine-level read-only containment.
  This is scoped to the gateway's own call path, not a database-wide guarantee: nothing
  stops a *different* code path in a deployment from opening its own connection to the same
  database with different (write) credentials and bypassing the gateway entirely — that is
  an operational/deployment concern (credential scoping outside this codebase), not
  something this module can enforce from inside a single process.
- **Multi-tenant / per-actor policy is not modeled.** `evaluate()` accepts an `actor`
  parameter but v1 does not yet use it to vary policy (e.g. different agents/tenants having
  different table scopes). All actors get the same rules today.
- **No unbreakable claim.** This is defense-in-depth over a single, narrow decision point.
  Every control above can, in principle, be bypassed by a sufficiently creative adversary
  supplying sufficiently unusual SQL; the goal is to make the common, cheap attacks
  (destructive DDL, unscoped writes, obvious injection, catalog scraping, known RCE vectors)
  fail loudly and by default, not to claim the surface is fully closed.
- **`SELECT *` / PII exposure — RESOLVED (Phase 1), same caveat as above.** A `SELECT *`
  over a table with unreviewed PII columns is still `allow` with `pii_suspected` in
  `matched_rules` at the decision layer, but the *result* is no longer returned raw: it now
  passes through `redact_rows` (see "Phase 1 — Implemented Controls" above) before reaching
  the caller. `action` itself was never renamed to a literal `"redact"` value — redaction is
  applied as a post-execution step on every `allow` result rather than being its own gate
  decision, since the useful unit of policy is still allow/block/hold and redaction is
  orthogonal to it.
- **The parser-differential risk: this gate's SQL model can drift from the real database
  engine's.** `sql_governance.py` decides using `sqlglot`'s parse of the proposed text —
  it is a model of SQL, not the database's own parser. A dialect quirk, a vendor
  extension the model doesn't represent, or a future `sqlglot` regression could in
  principle cause the gate's classification to disagree with what the engine actually
  executes (see G0.8/G0.9 in `.devdocs/PHASE0_GATES.md`, which exist specifically to
  red-team this gap: an execution-based differential check against a real SQLite engine,
  and explicit Postgres-dialect attack coverage that asserts a *named* control fired
  rather than trusting a bare parse failure). **This risk is not hypothetical — it has
  already been observed and fixed once:** `sqlglot` does not decode Postgres's
  `U&"..."`/`U&'...'` Unicode-escaped-identifier syntax (`U&"\0070g_read_file"` is
  `"pg_read_file"` once a real Postgres server decodes it), so the AST-based function-name
  check saw only the literal, un-decoded escape text and missed the deny list entirely
  while the string sent to a real server would have executed `pg_read_file`/`pg_sleep`/
  `nextval`/`dblink_exec` — a confirmed bypass, closed by adding raw-text-level decode-
  and-match plus an unconditional `obfuscated_identifier` fail-safe (see
  `_unicode_escape_obfuscation_decision` in `sql_governance.py`). Identifier-escape
  decoding is exactly the kind of "vendor syntax the model doesn't represent" divergence
  described above, and there is no guarantee it is the last one of its kind — a **parser**
  gap in this specific engine-syntax family, or in an entirely different one `sqlglot`
  hasn't been red-teamed against yet, should be assumed possible until proven otherwise.
  Because of this, **the SQL-string policy gate is explicitly a defense-in-depth control,
  not the sole boundary**: it must sit on top of engine-level **least-privilege** —
  read-only database roles and row-level security (RLS), and reads executed inside
  read-only transactions, at the connection the gateway actually uses — a scoped,
  non-superuser role with the minimum grants the agent's legitimate workload needs, so that
  a *future, still-unknown* parser-differential bypass of this string-level gate degrades
  into "the least-privileged role's own limits" (a read-only transaction against a role
  with no filesystem/RCE-adjacent grants), not full database access.
  **IMPLEMENTED for SQLite (Phase 1):** `ReadOnlyExecutor` (`backend/core/executor.py`)
  is that least-privilege connection — `file:<path>?mode=ro`, a forced row cap, and a
  wall-clock timeout — wired into every `allow` decision by `backend/core/gateway.py`. A
  parser-differential bypass of the string-level gate that reached this executor as a
  write would now degrade into "SQLite's own `mode=ro` rejection," not full database
  access; a bypass that reached it as an expensive/unbounded read still degrades into "the
  row cap and timeout's own limits." **NOT YET implemented for Postgres**: no read-only
  role, no `SET TRANSACTION READ ONLY`, no `statement_timeout` enforcement has been proven
  against a real Postgres instance in this repo (see the Postgres residual gap above) — for
  a Postgres deployment specifically, until that lands, this string-level gate remains the
  only control in place, so a parser-differential bypass of it there is still a full
  bypass, not a degraded one.

## Phase 2 — Approval Workflow, Multi-DB Connectors, and API

**Scope added this phase:** the human-in-the-loop write-approval queue
(`backend/core/approvals.py`), the multi-database connector factory
(`backend/core/connectors.py`), and the REST + SSE API that exposes both plus every
Phase 0/1 control over HTTP (`backend/api/gateway_routes.py`, `backend/gateway_app.py`).
This is what finally executes a `hold` decision — every prior phase only decided and
audited it (see "The `hold` path is still not executed under any control" above, now
resolved for the case where a human actually approves it).

### New risks

| ID | Risk | Example |
|----|------|---------|
| R12 | **Approval bypass / replay** | Executing a held write without a corresponding `approve` call, or re-executing the same approval a second time (double-spend on a single human decision) |
| R13 | **Approval TOCTOU** | Two concurrent `approve`/`reject` calls racing on the same id; substituting a different SQL statement for the one a human actually reviewed between enqueue and approve |
| R14 | **Approver/actor spoofing** | Nothing in this phase authenticates *who* calls `POST /query` (`actor`) or `POST /approvals/{id}` (`approver`) beyond a free-text string the caller supplies — a compromised or careless client can claim to be anyone |
| R15 | **DSN injection / credential leakage via connectors** | A malformed or adversarial DSN routing a connection to an unintended host, or a DSN's embedded credential ending up in a log line or an API error string |
| R16 | **API/rate-limit abuse** | An actor issuing enough query volume to degrade the gateway or the underlying database, or evading a per-actor cap by rotating the `actor` field across requests |

### Controls added this phase

| Control | Mitigates | Notes |
|---|---|---|
| **Atomic claim-before-execute** (`ApprovalQueue._claim`, a single `UPDATE ... WHERE id=? AND status='pending' AND expires_at > ?`) | R12, R13 | SQLite serializes writes on one connection, so exactly one caller can ever observe `rowcount == 1` for a given id; the loser raises `ApprovalError` and never reaches `write_executor.execute` at all. The claim happens BEFORE execution, not after — "decided" and "executed" can never straddle a race window. Regression-proved by `test_approval_failsafe_already_decided_double_approve` (`tests/api/test_gateway_api.py`): a second `approve` of an already-approved id is rejected (4xx) and the row is unchanged by the failed attempt, and exactly one `approved` audit entry exists regardless of how many approve calls were made |
| **Immutable stored SQL** (`ApprovalQueue.enqueue`/`approve`) | R13 | `approve` never accepts a caller-supplied SQL string and never re-parses/reconstructs one — it executes exactly the `sql` column written once at `enqueue` time. There is no code path, including the HTTP request body for `POST /approvals/{id}`, through which a different statement than the one a human reviewed in `GET /approvals` could be substituted |
| **Fail-safe on unknown/expired/already-decided** (`ApprovalQueue._claim`, surfaced as 4xx by `gateway_routes.decide_approval`) | R12 | Every failure path raises before any execution is attempted — an approval TTL (`DEFAULT_TTL_SECONDS`, 24h default) additionally bounds how long a stale hold from an abandoned agent session remains executable at all |
| **Per-decision audit trail with the approver named** (`ApprovalQueue.approve`/`reject` -> `AuditLog.append(action="approved"/"rejected", actor=<approver>, ...)`) | R12, Asset #5 | Every approve/reject is its own hash-chained entry, distinct from the original `hold` entry the gateway already wrote — an investigator can see who proposed a write AND who authorized or refused it, not just that it happened |
| **Per-actor rate limiting** (`gateway_routes._check_rate_limit`, sliding 60s window, one bucket per `actor`) | R16 | Bounds request volume per identifier before it reaches the governance gate or a database connection at all; `test_rate_limit_is_per_actor` regression-guards that one actor's bucket never starves another's. **This is a UX throttle, not a security control** — see "Phase 2 Residual Risks" below for why, and what the interim real control is |
| **DSN validation + credential scrubbing** (`connectors.connector_for`, `_scrub_dsn`) | R15 | An empty/non-string DSN, an unrecognized scheme, or a network DSN (`postgres://`/`mysql://`) with no host is rejected outright, before any connection attempt — this factory only ever connects to the host the DSN itself names, never a default or inferred one. Any DSN that does reach a log line or an exception message is passed through `_scrub_dsn` first, which masks a `user:password@` userinfo component |
| **DSN is server-config-only — request bodies cannot supply one** (`gateway_routes.QueryRequest`, `model_config = ConfigDict(extra="forbid")`) | R15 | `connector_for(dsn)` is only ever called once, at `create_gateway_app(dsn)` startup time, from a server-side config value — never from a request body. `QueryRequest` additionally rejects (422) any request containing a `dsn`/`connection`/`database_url`/any-other extra field outright, so a malicious payload (e.g. `dsn: "postgresql://root@169.254.169.254/"` targeting a cloud metadata endpoint, or `file:/etc/passwd`) is refused before the route handler runs at all — no SSRF, no arbitrary file read, structurally, not just by convention. Regression-tested by `TestDsnServerPinned` (`tests/api/test_gateway_api.py`, P2.12) |
| **Self-approval blocked** (`ApprovalQueue.approve`, `approver == proposer` check before `_claim`) | R12, R14 (partial) | `approve()` refuses (403, `ApprovalError`) before ever claiming the row if `approver` equals the `actor` who originally proposed the SQL — the same identifier cannot propose a write and then authorize its own execution. This is an in-process stopgap given the identity gap below (R14 remains open for a DIFFERENT claimed identity), not a full fix for approver/actor spoofing — see "Phase 2 Residual Risks". Regression-tested by `TestSelfApproval` (P2.16) |
| **Never-raise executors for untested/live backends** (`connectors.PostgresWriteExecutor`, `MySQLReadOnlyExecutor`, `MySQLWriteExecutor`) | R15 (partial), general robustness | Constructing a `Connection` for `postgres://`/`mysql://` never opens a socket; a connection failure, a missing driver package, or an engine error is always reported as `ExecutionResult.error`, never an unhandled exception that could crash a request handler or leak a traceback containing connection details to a caller |
| **PII scrubbing on every API error/response string, including unhandled 500s** (`redact_text` applied to `ApprovalError` messages, rate-limit-rejected actor identifiers in logs, and audit `actor`/`proposed_sql` fields; a catch-all `@app.exception_handler(Exception)` in `gateway_app.py` that returns a fixed, non-exception-derived `{"detail": "internal server error"}` body for anything else) | R3 (extended to the API surface) | Mirrors `gateway.py`'s existing "reason strings are PII-scrubbed" contract, extended to the new approval and rate-limit error paths this phase adds, and now closed for the *unhandled*-exception case too — no traceback, internal file path, raw `proposed_sql`, or other PII ever reaches a caller regardless of what internally raised. Regression-tested by `TestApiErrorRedacted` (P2.18) |
| **Defense-in-depth read-only executor assertion** (`Gateway.handle`, `IS_READONLY` marker on every executor class) | R1, R2 (containment, extending the existing read-only-executor control to distrust the caller too) | `Gateway.handle` no longer merely documents that its `executor` argument must be read-only — it asserts `getattr(executor, "IS_READONLY", None) is True` and raises `TypeError` otherwise, fail-closed against both a write-capable executor and an unmarked one. A future wiring bug that handed the read path a `WriteExecutor` (or any executor-shaped object with no `IS_READONLY = True`) is now rejected outright rather than silently trusted. Regression-tested by `TestHandleReadonlyGuard` (P2.19) |

## Phase 2 Residual Risks

Honest statement of what Phase 2's hardening (`.devdocs/PHASE2_GATES.md` P2.12–P2.19,
implemented above) does and does not close. Three items get their own explicit callout
first, since an adversarial review specifically asked whether they were named, not just
implied by a control table entry:

- **Self-approval is blocked for the same claimed identity — it is NOT identity
  verification.** `ApprovalQueue.approve` (see the "Self-approval blocked" control row
  above, P2.16) refuses to let `approver == actor` execute a write that same identifier
  proposed. This closes the literal case — the exact string `"agent-1"` cannot propose a
  write and then also approve it. It does **not** close R14 (approver/actor spoofing):
  nothing authenticates that the caller sending `approver: "root-admin"` on
  `POST /approvals/{id}` is actually a different, real human — a single adversarial
  caller can trivially propose as `actor: "agent-1"` and then approve as
  `approver: "agent-2"` (or any other string it makes up), defeating the self-approval
  check while still being the same unauthenticated party on both ends. Self-approval-by-
  same-name is now a closed control; self-approval-by-a-different-claimed-name is not,
  and cannot be closed without the identity provider this document already flags as
  missing below. `POST /approvals/{id}` MUST sit behind real authentication before it is
  exposed beyond a trusted internal network.
- **DSN is server-config-only — enforced structurally, not by convention, but only for
  the request surface this phase adds.** `connector_for(dsn)` is called exactly once, at
  `create_gateway_app(dsn)` startup, from a value the deploying operator controls; no
  request handler in `gateway_routes.py` ever calls it again or accepts a `dsn` argument.
  `QueryRequest`'s `extra="forbid"` additionally makes a request body carrying a `dsn`,
  `connection`, `database_url`, or any other unexpected field a hard 422, closing the
  SSRF/file-read vector a caller-supplied connection target would otherwise open (see
  P2.12, `TestDsnServerPinned`). What this does **not** cover: (1) the DSN itself, once
  chosen by the operator, is not validated against an allow-list of expected hosts — an
  operator who *misconfigures* `create_gateway_app` with an untrusted DSN gets exactly
  that untrusted connection, by design (this control stops a request-time hijack, not an
  operator-time misconfiguration); (2) `_scrub_dsn` masks a DSN's password before it
  reaches a log line, but the DSN's host/scheme is not secret and is not itself
  redacted — a log line can still reveal *which* host the server is configured to talk
  to (not a credential leak, but an information-disclosure one, to whoever can read
  server logs); (3) the PG/MySQL connector paths this DSN can route to remain unproven
  against a live server without `ICBERG_TEST_PG_DSN`/`ICBERG_TEST_MYSQL_DSN` set — see
  the PG/MySQL bullet below, unchanged by this hardening.
- **Per-actor rate limiting is a UX throttle, not a security control.** `actor` is an
  unauthenticated, caller-chosen free-text string (see the identity-provider gap below) —
  nothing stops a single real-world caller from rotating it (`"agent-1"`, `"agent-1-b"`,
  `"agent-1-c"`, ...) on every request and getting a fresh 60-request bucket every time,
  defeating the cap entirely with zero additional cost to the attacker. Treat
  `_check_rate_limit` as what it actually is: a courtesy limit that keeps one *well-
  behaved* agent identifier from accidentally hammering the gateway, not a defense
  against a deliberately abusive caller (R16 is only partially mitigated by it). **The
  interim real control is IP-based rate limiting** — layered in front of this API (a
  reverse proxy, an API gateway, or infrastructure-level throttling keyed on source IP or
  an authenticated client credential, not the self-reported `actor` field) — which this
  phase does not implement itself (it is a single-process FastAPI app with no reverse
  proxy of its own in this repo) but which any real deployment of this gateway MUST add
  in front of it before `actor`-based limiting can be treated as anything more than a
  developer convenience. The identity-provider work below is the durable fix (an
  authenticated `actor` can be rate-limited by a stable, unspoofable identity instead of
  a self-reported string); until then, IP-based limiting at the edge is the honest interim
  boundary, not this module's per-actor buckets.

- **No identity provider / authentication.** `actor` (on `POST /query`) and `approver` (on
  `POST /approvals/{id}`) are free-text strings supplied by the caller, with nothing in
  this phase verifying either claim — R14 is *documented*, not *closed*. In a real
  deployment, `POST /approvals/{id}` MUST sit behind real authentication and authorization
  (only specific, verified humans/roles may approve a write) before it is exposed beyond a
  trusted internal network; today, anyone who can reach the API can approve anything held,
  using any approver name they like. This is the single most important gap to close before
  this phase's approval workflow is a genuine security boundary rather than a workflow
  affordance — Phase 3's integration-surface work (SDK/MCP/framework adapters) is the
  natural place to add pluggable auth, not this phase. This is also the root cause both
  the self-approval bullet above (same-identity-only enforcement) and the rate-limit
  bullet above (per-string, not per-identity, buckets) cannot be fully closed without.
- **Approval TTL is a soft, application-level bound, not a hard one.** `_claim`'s
  `expires_at > now` check uses the queue's own injected/default clock; an operator with
  direct SQLite access to the approvals database (out of scope per this document's
  adversary model — see "human operator with direct database credentials... bypasses the
  gateway entirely") could edit `expires_at` directly. This is the same class of residual
  gap `audit.py`'s "tamper-evident, not tamper-proof" caveat already states for the audit
  log; the approvals table carries no append-only trigger or external anchor of its own
  (unlike `audit_log`), since — unlike the audit trail — its rows are legitimately
  mutable (`pending` -> `approved`/`rejected` is the whole point). The audit log itself,
  which IS append-only and anchored, remains the tamper-evident record of what was
  actually decided and executed, independent of the approvals table's own current state.
- **PG/MySQL connector paths are unproven without a live DSN.** `connector_for` routes to
  `PostgresReadOnlyExecutor`/`PostgresWriteExecutor`/`MySQLReadOnlyExecutor`/
  `MySQLWriteExecutor` correctly and every one of them degrades to `ExecutionResult.error`
  instead of raising when no server is reachable (regression-tested), but neither backend's
  actual read/write round-trip, `get_schema_catalog()` (both are documented stubs
  returning `None`), or least-privilege session/role enforcement has been exercised against
  a real server in this repo — only `ICBERG_TEST_PG_DSN`/`ICBERG_TEST_MYSQL_DSN`-gated live
  tests would prove that, and neither is set by default. Same honest status Phase 1 already
  gives `PostgresReadOnlyExecutor` alone, now extended to the write side and to MySQL.
- **MySQL's read-only boundary is session-level, not engine-level.** `SET SESSION
  TRANSACTION READ ONLY` (plus never committing) is the best available control for a MySQL
  network connection — MySQL has no equivalent of SQLite's `mode=ro` file-level open. The
  authoritative boundary remains a least-privilege, read-only database role on the
  connecting credential, exactly the same posture this document already states for
  Postgres and for SQLite's own `PRAGMA query_only` (secondary-only, `mode=ro` is primary).
- **In-memory, per-process rate limiting.** `_check_rate_limit`'s buckets live in
  `app.state`, not a shared store — a multi-process/multi-instance deployment (several
  `uvicorn` workers, a horizontally-scaled sidecar) gives each process its own independent
  per-actor quota, so the *effective* cap for a distributed deployment is `rate_limit_per_
  minute * process_count`, not the configured value. A production deployment behind more
  than one process needs a shared limiter (Redis or equivalent) — out of scope for this
  phase's single-process API.
- **The approval workflow does not re-validate policy at approval time.** `approve`
  executes the stored SQL exactly as it was classified `hold` at proposal time; it does not
  re-run `GovernanceGate.evaluate` before executing. This is deliberate (re-evaluating could
  itself be gamed by timing, and the whole point of `hold` is that a human — not the gate —
  makes the final call), but it does mean a change to policy rules between enqueue and
  approve has no effect on an already-pending approval; an operator who tightens policy
  should also review the pending queue, not assume old holds are retroactively re-screened.
