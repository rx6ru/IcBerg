r"""SQL governance gate — the policy decision layer between an AI agent and a database.

Parses a *proposed* SQL statement with `sqlglot`, classifies it (read/write/ddl/unknown),
and applies a fail-safe (default-deny) least-privilege policy: destructive and
out-of-scope operations are blocked, mutations without a `WHERE` clause are blocked,
mutations with a `WHERE` clause are held for human approval (Phase 2), and bounded,
scoped reads are allowed.

Classification is **recursive**: a write or DDL statement hidden inside a CTE or
subquery (`WITH x AS (DELETE FROM users RETURNING *) SELECT * FROM x WHERE id=1 LIMIT 1`)
is classified — and has its write/destructive rules applied — using that *inner* node,
never the outer wrapper. Trusting only the outer AST node is a confirmed bypass: the
wrapper can be shaped to look like a harmless, bounded `SELECT` while the real mutation
executes underneath it.

Dangerous-function detection (RCE, sequence-mutation, and DoS primitives such as
`pg_read_file`, `dblink_exec`, `nextval`, `pg_sleep`) is **AST-based and primary**: every
function-call node in the parsed tree is walked, its name is normalized (final identifier
component, quoting/backticks/brackets stripped, lower-cased) via sqlglot's own identifier
resolution, and matched case-insensitively against a deny list. This is what a raw-string
regex on `\bNAME\s*\(` cannot do — `"pg_read_file"('/etc/passwd')`, `PG_READ_FILE(...)`,
and `pg_catalog."pg_read_file"(...)` all reach the database as the exact same call, and
sqlglot resolves all three to the same normalized name, so quoting/casing/schema-
qualification cannot evade this layer. An earlier version of this module used only
string-level regex detection for these functions and claimed in this docstring that it
"cannot be bypassed" — that was false: double-quoting the function name (a syntax the
regex's `\s*\(` boundary does not tolerate) reached `allow`. The regexes are retained as
a **secondary, defense-in-depth layer** (they still catch patterns with no function-call
AST hook at all, e.g. `COPY ... TO PROGRAM`, comment/tautology injection) and now run
unconditionally alongside — never instead of — the AST check.

A third detection surface exists specifically for **Postgres Unicode-escaped identifiers**
(`U&"..."` / `U&'...'`, e.g. `U&"\0070g_read_file"` — Postgres decodes this to
`"pg_read_file"` server-side before it ever reaches SQL semantics): `sqlglot` does not
implement this decoding at all, so both the AST-based check above and the string-level
regexes see only the raw, un-decoded escape text and miss it completely — a confirmed
bypass of every other layer in this module. This is handled independently, at the raw-SQL-
text level, by `_unicode_escape_obfuscation_decision`: it decodes any `U&"..."`/`U&'...'`
span it finds and matches the *decoded* name against the same deny lists (for an accurate,
specific `matched_rules` entry), but — critically — it blocks on the mere *presence* of the
`U&` introducer regardless of what it decodes to or whether it decodes cleanly at all. See
that function's docstring and THREAT_MODEL.md for why unconditional blocking here is safe:
legitimate agent-generated SQL has no reason to use this syntax.

This module only *decides*. It never opens a connection and never executes SQL — Phase 1
wires the decision into an actual execution boundary (read-only transactions, statement
timeouts, row caps, audit logging). See THREAT_MODEL.md for the full threat model this
gate is one control within, including the parser-differential residual risk: this gate's
SQL model can in principle drift from the real database engine's, which is why it is one
layer of defense-in-depth on top of least-privilege DB roles/RLS, not the only boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import sqlglot
import structlog
from sqlglot import exp

logger = structlog.get_logger(__name__)

Action = Literal["allow", "block", "hold", "redact"]
Classification = Literal["read", "write", "ddl", "unknown"]

# Dialects attempted, in order, when parsing a proposed statement (when the caller does
# not pin a specific `dialect`). Generic first (matches the widest range of ANSI-ish SQL
# without dialect-specific surprises); "postgres" second, because some constructs the
# generic dialect can't model at all parse cleanly under Postgres.
_PARSE_DIALECTS: tuple[str | None, ...] = (None, "postgres")

# Mutation node types — anywhere in the parsed tree, including nested inside a CTE or
# subquery, these mean "this statement writes data" regardless of what the outer
# statement looks like.
_MUTATION_NODE_TYPES: tuple[type[exp.Expression], ...] = (exp.Insert, exp.Update, exp.Delete, exp.Merge)

# DDL / destructive node types — same "anywhere in the tree" treatment as mutations.
_DDL_NODE_TYPES: tuple[type[exp.Expression], ...] = tuple(
    t
    for t in (exp.Create, exp.Drop, exp.Alter, getattr(exp, "TruncateTable", None))
    if t is not None
)


@dataclass
class PolicyDecision:
    """The outcome of evaluating one proposed SQL statement against policy.

    Attributes:
        action: What the gateway will do with the statement.
        classification: The statement's coarse SQL category.
        reason: Human-readable explanation of the decision, for logs and audit.
        matched_rules: Names of every policy rule that fired, in evaluation order.
        tables: Table names (schema-qualified where known) referenced by the statement.
    """
    action: Action
    classification: Classification
    reason: str
    matched_rules: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)


# --- string-level detection patterns (dialect-agnostic, run even if parsing fails) ---

# `OR 1=1` / `OR 2=2` style numeric tautologies.
_TAUTOLOGY_NUM_RE = re.compile(r"\b(?:or|and)\s+(\d+(?:\.\d+)?)\s*=\s*\1\b", re.IGNORECASE)
# `OR 'a'='a'` style string tautologies.
_TAUTOLOGY_STR_RE = re.compile(r"\b(?:or|and)\s+'([^']*)'\s*=\s*'\1'", re.IGNORECASE)
# `COPY ... TO PROGRAM '...'` — Postgres DB-native RCE.
_COPY_PROGRAM_RE = re.compile(r"\bcopy\b[\s\S]*?\bto\s+program\b", re.IGNORECASE)
# File/large-object functions that reach the host filesystem, plus `dblink*` — Postgres's
# cross-database/cross-connection function family, which can open arbitrary connections
# and execute arbitrary SQL (including destructive SQL) on another server via a single
# SELECT (`dblink`, `dblink_exec`, `dblink_connect`, `dblink_connect_u`, ...).
_RCE_FUNCTION_RE = re.compile(
    r"\b(pg_read_file|pg_read_binary_file|pg_ls_dir|pg_stat_file|lo_import|lo_export|dblink\w*)\s*\(",
    re.IGNORECASE,
)
# Explicit system-catalog schema qualification.
_SYSTEM_CATALOG_RE = re.compile(r"\b(pg_catalog|information_schema)\.", re.IGNORECASE)
# Bare `pg_*` table reference (e.g. `FROM pg_user`), not necessarily schema-qualified.
_PG_TABLE_RE = re.compile(r"\b(?:from|join)\s+pg_[a-z_]+\b", re.IGNORECASE)
# Classic DoS primitive.
_PG_SLEEP_RE = re.compile(r"\bpg_sleep\s*\(", re.IGNORECASE)
# Sequence side-effect functions: `nextval`/`setval` mutate persistent sequence state as
# a side effect of what looks like an ordinary read (`SELECT nextval('s')`); `currval`
# reads that mutated state back. All three are excluded from "safe read" territory.
_SEQUENCE_FUNCTION_RE = re.compile(r"\b(nextval|setval|currval)\s*\(", re.IGNORECASE)

# --- AST-based function-name detection (PRIMARY layer; see module docstring) ---
#
# Bare, lowercase canonical names to match a normalized function-call node against.
# Quoting/backtick/bracket-stripping and case-folding are handled once, in `_func_name`/
# `_ast_function_names` below, using sqlglot's own identifier resolution — these sets
# don't need quoting variants the way the regexes above do.

_RCE_FUNCTION_NAMES: frozenset[str] = frozenset({
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "pg_stat_file",
    "lo_import",
    "lo_export",
})
# `dblink*` is a prefix family (dblink, dblink_exec, dblink_connect, dblink_connect_u,
# dblink_disconnect, dblink_send_query, dblink_is_busy, ...) — matched by prefix, mirroring
# the string-level regex above.
_RCE_FUNCTION_PREFIX = "dblink"

_SEQUENCE_FUNCTION_NAMES: frozenset[str] = frozenset({"nextval", "setval", "currval"})

# `generate_series` is deliberately NOT included: it is a legitimate, commonly-used
# reporting/reference-data function (date-series generation, etc.) with no attacker-
# specific capability by itself — blocking it outright would reintroduce the same
# over-blocking failure mode this module is being fixed for elsewhere (see the UNION
# fix in `evaluate`). `pg_terminate_backend` has no comparable legitimate-read use case
# (it kills another session/connection — pure sabotage/DoS), so it is included.
_DOS_FUNCTION_NAMES: frozenset[str] = frozenset({"pg_sleep", "pg_terminate_backend"})

# GUC/session-mutation functions: `set_config`/`pg_catalog.set_config` mutate a Postgres
# session (or transaction-local, per its third `is_local` argument) setting as a side
# effect of what parses as an ordinary, single, innocent-looking `SELECT` — e.g.
# `SELECT set_config('statement_timeout', '0', false)` disables the very statement
# timeout this module's own least-privilege executor relies on downstream, and
# `pg_reload_conf()` asks the postmaster to reload `postgresql.conf` outright. Neither
# reads a single row of application data, so there is no legitimate-analytics reason a
# proposed SELECT would ever need to call either — unconditional block, reusing the same
# AST function-name detection as the RCE/sequence/DoS lists above so quoting/casing/
# schema-qualification (`"set_config"(...)`, `pg_catalog.set_config(...)`) can't evade it.
_GUC_MUTATION_FUNCTION_NAMES: frozenset[str] = frozenset({"set_config", "pg_reload_conf"})


# --- Postgres Unicode-escaped identifier/string decoding (CRITICAL fix) ---
#
# Postgres has a `U&"..."` (identifier) / `U&'...'` (string literal) Unicode-escape
# syntax: `\XXXX` (4 hex digits) or `\+XXXXXX` (6 hex digits) inside the quoted text
# decode to the named code point *server-side*, before the identifier/string
# participates in SQL semantics at all — `U&"\0070g_read_file"` IS `"pg_read_file"` as
# far as a real Postgres server is concerned. An optional trailing `UESCAPE 'c'` clause
# swaps the escape character from `\` to a caller-chosen character `c`.
#
# `sqlglot` does not implement this decoding — confirmed empirically: it tokenizes
# `U&"..."` as the bare identifier `U`, a bitwise-AND operator, and a quoted-but-
# undecoded identifier whose `.name` is still the literal `\XXXX...` escape text. So
# every AST-based check above (`_func_name`, `_ast_function_names`,
# `_ast_system_catalog`, `_tables`) sees only that raw escaped text, never
# `pg_read_file`/`pg_sleep`/`nextval`/`dblink_exec`/`pg_catalog`, and the string-level
# regexes don't match either (the raw SQL literally doesn't contain those substrings).
# A real Postgres server decodes and executes it anyway — a parser-differential bypass
# of every other layer in this module (see THREAT_MODEL.md residual risks).

# `U&"`/`U&'` introducer. Postgres allows whitespace between `U&` and the quote, so this
# does too; `\b` avoids matching inside a longer token that merely ends in `...U&`.
_PG_UESCAPE_INTRODUCER_RE = re.compile(r"\bU&\s*(['\"])", re.IGNORECASE)

# `UESCAPE 'c'` clause immediately following the closing quote, which swaps the escape
# character from `\` to `c` for that one `U&` literal.
_PG_UESCAPE_CLAUSE_RE = re.compile(r"\s*UESCAPE\s*'([^'])'", re.IGNORECASE)


def _decode_pg_unicode_escape(content: str, escape_char: str) -> str | None:
    r"""Decode Postgres `U&` Unicode-escape sequences in `content`.

    `\XXXX` (4 hex digits) and `\+XXXXXX` (6 hex digits) — using whichever character
    `escape_char` names (`\` unless a `UESCAPE` clause overrides it) — decode to the
    named Unicode code point; a doubled escape character decodes to one literal instance
    of that character. Returns None (fail safe) if any escape sequence is malformed —
    callers must treat None as "could not verify safe", never as "safe".
    """
    out: list[str] = []
    i, n = 0, len(content)
    while i < n:
        ch = content[i]
        if ch != escape_char:
            out.append(ch)
            i += 1
            continue
        if i + 1 < n and content[i + 1] == escape_char:
            out.append(escape_char)
            i += 2
            continue
        if i + 1 < n and content[i + 1] == "+":
            digits = content[i + 2 : i + 8]
            if len(digits) == 6 and all(c in "0123456789abcdefABCDEF" for c in digits):
                out.append(chr(int(digits, 16)))
                i += 8
                continue
            return None
        digits = content[i + 1 : i + 5]
        if len(digits) == 4 and all(c in "0123456789abcdefABCDEF" for c in digits):
            out.append(chr(int(digits, 16)))
            i += 5
            continue
        return None
    return "".join(out)


def _pg_unicode_escaped_identifiers(sql: str) -> list[tuple[str, bool]]:
    """Locate every `U&"..."` / `U&'...'` span in `sql` and return each one's
    `(decoded_text, is_call)` (honoring a trailing `UESCAPE 'c'` clause). `is_call` is
    True when the span is immediately followed by `(` — i.e. used as a function-call
    name, the same distinction `_ast_function_names` (calls) vs `_tables`/
    `_ast_system_catalog` (table/schema references) draws for ordinary identifiers — so a
    decoded `pg_read_file(...)` call is matched only against the RCE/sequence/DoS
    deny lists, and a decoded `pg_catalog` table/schema reference only against the
    catalog check, instead of every decoded name being checked against every list.

    A span that fails to decode cleanly is skipped here — it is still caught by the
    unconditional `obfuscated_identifier` fail-safe in
    `_unicode_escape_obfuscation_decision`, which fires on the mere presence of the `U&`
    introducer regardless of whether decoding succeeds.
    """
    decoded: list[tuple[str, bool]] = []
    for m in _PG_UESCAPE_INTRODUCER_RE.finditer(sql):
        quote = m.group(1)
        start = m.end()
        end = sql.find(quote, start)
        if end == -1:
            continue
        content = sql[start:end]
        escape_char = "\\"
        rest = sql[end + 1 :]
        clause = _PG_UESCAPE_CLAUSE_RE.match(rest)
        if clause:
            rest = rest[clause.end() :]
        result = _decode_pg_unicode_escape(content, clause.group(1) if clause else escape_char)
        if result is not None:
            is_call = rest.lstrip().startswith("(")
            decoded.append((result, is_call))
    return decoded


def _normalize_decoded_name(name: str) -> str:
    """Normalize a decoded identifier/name for deny-list matching: final `.`-qualified
    component, quote characters stripped, lower-cased — mirrors `_func_name`'s own
    normalization so the same deny lists apply to both AST-derived and decoded names.
    """
    last = name.rsplit(".", 1)[-1]
    return last.strip().strip("\"'`[]").lower()


def _unicode_escape_obfuscation_decision(
    sql: str, dialect: str | None, string_flags: list[str]
) -> PolicyDecision | None:
    """CRITICAL fix — catch Postgres `U&"..."`/`U&'...'` Unicode-escaped identifiers that
    `sqlglot` does not decode (see the block comment above `_PG_UESCAPE_INTRODUCER_RE` and
    the module docstring).

    Two layers, both applied here:
      1. Decode-then-match: every `U&` span found is decoded and the decoded name is
         matched against the RCE/sequence/DoS/catalog deny lists, so a decoded
         `pg_read_file` etc. is caught *by name* with a specific, accurate rule in
         `matched_rules` (`db_native_rce_function`, `sequence_mutation`, `dos_suspected`,
         `system_catalog_access`) rather than only a generic catch-all.
      2. Fail-safe catch-all: regardless of what a `U&` span decodes to — or whether it
         decodes cleanly at all; a malformed escape fails safe, not silently — its mere
         presence outside of string-literal content is treated as obfuscation and blocks
         the statement outright (`obfuscated_identifier`). Legitimate agent-generated
         analytics SQL has no reason to use Postgres Unicode-escape syntax, so
         unconditionally blocking it cannot cause a real over-blocking regression —
         except when `U&` merely appears inside an unrelated string literal's data,
         which is why the presence check strips string-literal content first (below).

    Returns a `block` decision if the `U&` introducer appears outside of any string-
    literal content, or None if it doesn't (the ordinary `evaluate` path applies
    unchanged).

    The presence check itself runs on `sql` with single-quoted string-literal *contents*
    blanked out (via `_strip_string_literals`), not on the raw text: a benign literal
    whose data happens to contain `U&` right before a quote — `note = 'U&'`,
    `cat = 'menu&'`, `note = 'the U&'' operator'` — must not trip this fail-safe. This
    does not weaken detection of either real attack surface: the `U&"..."` identifier
    form always uses *double* quotes, which `_strip_string_literals` never touches, and a
    genuine `U&'...'` single-quoted Unicode-escape literal still leaves a `U&''`
    introducer-plus-empty-string behind after stripping, which still matches.
    """
    if not _PG_UESCAPE_INTRODUCER_RE.search(_strip_string_literals(sql)):
        return None

    spans = [(_normalize_decoded_name(n), is_call) for n, is_call in _pg_unicode_escaped_identifiers(sql)]
    call_names = {n for n, is_call in spans if is_call}
    ident_names = {n for n, is_call in spans if not is_call}

    matched_rules = list(string_flags)
    if any(n in _RCE_FUNCTION_NAMES or n.startswith(_RCE_FUNCTION_PREFIX) for n in call_names):
        matched_rules.append("db_native_rce_function")
    if call_names & _SEQUENCE_FUNCTION_NAMES:
        matched_rules.append("sequence_mutation")
    if call_names & _DOS_FUNCTION_NAMES:
        matched_rules.append("dos_suspected")
    if any(n in {"pg_catalog", "information_schema"} or n.startswith("pg_") for n in ident_names):
        if "system_catalog_access" not in matched_rules:
            matched_rules.append("system_catalog_access")
    matched_rules.append("obfuscated_identifier")

    # Best-effort classification/table extraction for an accurate audit record only —
    # never required for the block decision itself, which is unconditional on the
    # introducer's mere presence and never resolves to anything but `block`.
    classification: Classification = "unknown"
    tables: list[str] = []
    statements = _parse(sql, dialect=dialect)
    if statements:
        classification, _ = _classify(statements)
        tables = _tables(statements)

    return PolicyDecision(
        action="block",
        classification=classification,
        reason="Postgres Unicode-escaped identifier/string (U&\"...\" / U&'...') detected; "
        "sqlglot does not decode this syntax so it is treated as obfuscation and blocked "
        "outright, independent of what it decodes to.",
        matched_rules=matched_rules,
        tables=tables,
    )


def _strip_string_literals(sql: str) -> str:
    """Blank out single-quoted string contents so comment/keyword scans ignore literal data."""
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


# --- ATTACH / DETACH DATABASE detection ---
#
# `ATTACH DATABASE '<path>' AS <alias>` opens a second SQLite file (creating it if it
# doesn't exist) under a new schema alias within the *same* connection — and that second
# file does NOT inherit the `mode=ro` restriction the main connection was opened with in
# `executor.py`; it is opened with SQLite's own default (read-write, create-if-missing)
# mode. A statement that reaches this far has therefore escaped the read-only execution
# boundary entirely: it can create or overwrite an arbitrary file the process has
# filesystem access to and then write to it via the newly attached alias. `DETACH`
# reverses an attachment and is blocked alongside it for symmetry (and because a bare
# `DETACH <name>` is itself evidence of a session that has already ATTACHed something).
#
# `sqlglot`'s generic and `postgres` dialects (this module's default parse attempts) do
# not implement this syntax at all — `ATTACH DATABASE ... AS ...` raises `ParseError`
# under both, so without this function it would already fail safe via the ordinary
# "could not parse" block path below, but under the generic `fail_safe_parse_error` rule,
# not a specifically-named `attach_blocked` one. Only the `sqlite` dialect models it (as
# `exp.Attach`/`exp.Detach`), and even then a bare `DETACH y` is ambiguous enough that the
# *generic* dialect instead parses it as a harmless-looking `exp.Alias`. Rather than make
# a specific, auditable rule name depend on which dialect happened to be tried, detection
# here is raw-text-level and unconditional — mirroring
# `_unicode_escape_obfuscation_decision` above — so `attach_blocked` is always the
# reported rule regardless of what (if anything) the rest of `evaluate()` would parse
# this as. See `executor.py`'s SQLite authorizer for the matching connection-level deny
# (defense-in-depth: even a bypass of this gate must still be rejected at the engine).
#
# Anchored to *statement position*, not "the keyword appears anywhere": ATTACH/DETACH
# must be the statement itself — at the very start of the text (after only whitespace/
# comments) or immediately after a `;` statement separator (multi-statement smuggling,
# e.g. `SELECT 1; ATTACH DATABASE 'x' AS y`) — never merely a substring somewhere inside
# an otherwise-ordinary statement. An earlier, unanchored `\bATTACH\b`/`\bDETACH\b` match
# false-positived on a bare column/identifier named `attach` (`SELECT attach FROM t`),
# blocking a perfectly ordinary read with no ATTACH statement anywhere in it. Requiring
# statement position fixes that over-block while keeping every real ATTACH/DETACH form
# — bare or `DATABASE`-qualified, leading statement or smuggled after a `;` — caught.
_ATTACH_DETACH_RE = re.compile(
    r"(?:\A(?:\s|--[^\n]*(?:\n|\Z)|/\*.*?\*/)*|;\s*)(ATTACH|DETACH)\b",
    re.IGNORECASE | re.DOTALL,
)


def _attach_detach_decision(sql: str, string_flags: list[str]) -> PolicyDecision | None:
    """Unconditional block for `ATTACH [DATABASE] ...` / `DETACH [DATABASE] ...` in
    statement position — see the block comment above `_ATTACH_DETACH_RE`. Returns None
    if the keyword doesn't appear there (outside of string-literal content, so a literal
    like `note = 'attach the file'` does not trip this, and a bare identifier/column
    reference like `SELECT attach FROM t` does not either — it's not in statement
    position).
    """
    if not _ATTACH_DETACH_RE.search(_strip_string_literals(sql)):
        return None

    # Best-effort parse under the one dialect that actually models this syntax, purely to
    # enrich the audit record's `classification`/`tables` — never required for the block
    # decision itself, which is unconditional on the keyword's mere presence.
    classification: Classification = "ddl"
    tables: list[str] = []
    statements = _parse(sql, dialect="sqlite")
    if statements:
        tables = _tables(statements)

    return PolicyDecision(
        action="block",
        classification=classification,
        reason="ATTACH/DETACH DATABASE changes the set of database files this session can "
        "read from and write to, and the attached file does not inherit the read-only "
        "connection's restrictions; blocked outright.",
        matched_rules=[*string_flags, "attach_blocked"],
        tables=tables,
    )


def _string_level_flags(sql: str) -> list[str]:
    r"""Cheap, dialect-agnostic checks over the raw SQL text — a SECONDARY, defense-in-
    depth layer (see module docstring). The PRIMARY defense against RCE/sequence/DoS
    functions is the AST-based `_ast_function_names` check below, which normalizes
    quoting/casing/schema-qualification the way these regexes cannot: a quote character
    sitting between the function name and `(` (`"pg_read_file"(...)`) defeats
    `\bpg_read_file\s*\(` outright.

    Run unconditionally (even when parsing fails, and regardless of which `dialect` was
    requested) so a fail-safe block still carries an accurate, specific reason instead of
    a bare "couldn't parse it", and so patterns with no function-call AST hook at all
    (`COPY ... TO PROGRAM`, comment/tautology injection) are still caught even when
    parsing fails outright.
    """
    flags: list[str] = []
    stripped = _strip_string_literals(sql)

    if _TAUTOLOGY_NUM_RE.search(sql) or _TAUTOLOGY_STR_RE.search(sql):
        flags.append("tautology_suspected")
    if "--" in stripped or "/*" in stripped:
        flags.append("sql_comment_present")
    if _COPY_PROGRAM_RE.search(sql):
        flags.append("copy_to_program_rce")
    if _RCE_FUNCTION_RE.search(sql):
        flags.append("db_native_rce_function")
    if _SYSTEM_CATALOG_RE.search(sql) or _PG_TABLE_RE.search(sql):
        flags.append("system_catalog_access")
    if _PG_SLEEP_RE.search(sql):
        flags.append("dos_suspected")
    if _SEQUENCE_FUNCTION_RE.search(sql):
        flags.append("sequence_mutation")
    return flags


def _parse(sql: str, dialect: str | None = None) -> list[exp.Expression] | None:
    """Try to parse `sql`, returning the first successful, non-empty statement list.

    If `dialect` is given, parse strictly under that dialect (no generic/postgres
    fallback attempt) — the caller is asserting the SQL's real target engine. Otherwise
    try each dialect in `_PARSE_DIALECTS` in order, as before.

    Returns None if every attempt failed or every attempt yielded no real statements.
    """
    dialects: tuple[str | None, ...] = (dialect,) if dialect is not None else _PARSE_DIALECTS
    for d in dialects:
        try:
            statements = [s for s in sqlglot.parse(sql, read=d) if s is not None]
        except Exception:  # sqlglot raises its own ParseError/TokenizeError subclasses
            continue
        if statements:
            return statements
    return None


def _find_inner_write_or_ddl(statements: list[exp.Expression]) -> exp.Expression | None:
    """Recursively scan the *entire* parsed tree — including CTEs and subqueries — for
    the first mutation or DDL node, wherever it is nested.

    This is what defeats the "wrap a write in a harmless-looking SELECT" bypass: naively
    classifying only the outer/root node would see a bounded, filtered `SELECT` and
    allow it while a `DELETE`/`UPDATE`/`INSERT`/`MERGE`/DDL statement hides underneath in
    a CTE. When a top-level statement *is itself* a mutation/DDL node, `find_all` visits
    it first (it includes the node it's called on), so this is a superset of the old
    root-only behavior, not a replacement with different semantics for the simple case.
    """
    for stmt in statements:
        for node in stmt.find_all((*_MUTATION_NODE_TYPES, *_DDL_NODE_TYPES)):
            return node
    return None


def _find_select_into(statements: list[exp.Expression]) -> exp.Select | None:
    """Recursively find a `SELECT ... INTO <table> ...` anywhere in the tree.

    `SELECT INTO` implicitly creates a new table from the query result — it is DDL
    wearing a read's clothing, and `_find_inner_write_or_ddl` above does not catch it
    because `exp.Select` isn't a mutation/DDL node type.
    """
    for stmt in statements:
        for node in stmt.find_all(exp.Select):
            if node.args.get("into") is not None:
                return node
    return None


def _classify(statements: list[exp.Expression]) -> tuple[Classification, exp.Expression]:
    """Classify the *effective* statement and return the node whose shape the
    write/destructive rules should actually be applied to.

    Returns `(classification, node)`. For a plain top-level statement, `node` is just
    the root statement (unchanged behavior). For a write or DDL construct hidden inside
    a CTE/subquery, `node` is that inner construct — never the outer wrapper — so a
    `WHERE`-boundedness check (for example) reflects the real mutation, not a decoy
    `WHERE`/`LIMIT` tacked onto the wrapper.
    """
    root = statements[0]

    inner = _find_inner_write_or_ddl(statements)
    if inner is not None:
        if isinstance(inner, _DDL_NODE_TYPES):
            return "ddl", inner
        return "write", inner  # Insert / Update / Delete / Merge

    select_into = _find_select_into(statements)
    if select_into is not None:
        return "ddl", select_into  # implicit table creation

    if isinstance(root, exp.Copy):
        # COPY moves data in/out of the database (including to/from the host filesystem
        # or a shell via `TO PROGRAM`); treat it as an out-of-scope, always-blocked
        # operation like other DDL rather than a routine read/write.
        return "ddl", root

    if isinstance(root, exp.Command):
        name = str(root.args.get("this") or "").strip().upper()
        if name in {"TRUNCATE", "DROP", "ALTER", "CREATE"}:
            return "ddl", root
        return "unknown", root

    if isinstance(root, (exp.Select, exp.Union, exp.Except, exp.Intersect)):
        return "read", root

    return "unknown", root


def _has_where(node: exp.Expression) -> bool:
    return node.args.get("where") is not None


def _has_limit(node: exp.Select) -> bool:
    return node.args.get("limit") is not None


def _is_select_star(node: exp.Select) -> bool:
    for selected in node.expressions:
        if isinstance(selected, exp.Star):
            return True
        if isinstance(selected, exp.Column) and isinstance(selected.this, exp.Star):
            return True
    return False


def _tables(statements: list[exp.Expression]) -> list[str]:
    """Schema-qualified table names referenced anywhere in the statement(s)."""
    names: set[str] = set()
    for stmt in statements:
        for table in stmt.find_all(exp.Table):
            parts = [p for p in (table.args.get("catalog"), table.db, table.name) if p]
            names.add(".".join(str(p) for p in parts) if parts else table.sql())
    return sorted(names)


def _ast_tautology(statements: list[exp.Expression]) -> bool:
    """Detect `X = X` literal self-equality anywhere in the statement (e.g. `OR 1=1`)."""
    for stmt in statements:
        for eq in stmt.find_all(exp.EQ):
            left, right = eq.this, eq.expression
            if isinstance(left, exp.Literal) and isinstance(right, exp.Literal):
                if left.this == right.this and left.is_string == right.is_string:
                    return True
    return False


def _ast_system_catalog(statements: list[exp.Expression]) -> bool:
    """Detect references to system catalog schemas/tables via the parsed table list.

    More precise than the string-level regex (won't false-positive on a string literal
    that merely contains the text "pg_catalog."); kept as a supplement, not a replacement,
    since it only fires when parsing actually succeeds and resolves a `Table` node.
    """
    for stmt in statements:
        for table in stmt.find_all(exp.Table):
            db = (table.db or "").lower()
            name = (table.name or "").lower()
            if db in {"pg_catalog", "information_schema"} or name.startswith("pg_"):
                return True
    return False


def _ast_union(statements: list[exp.Expression]) -> bool:
    """Detect a `UNION`/`EXCEPT`/`INTERSECT` combination anywhere in the tree.

    Combining a benign-looking `SELECT` with a second, differently-scoped `SELECT` is a
    classic exfiltration technique (`SELECT id FROM users UNION SELECT password FROM
    users`) — the shape of the *first* branch can look perfectly bounded/safe while the
    combined result set pulls arbitrary columns from an arbitrary table.
    """
    for stmt in statements:
        for _ in stmt.find_all((exp.Union, exp.Except, exp.Intersect)):
            return True
    return False


def _func_name(node: exp.Expression) -> str | None:
    """Normalize a function-call AST node to a bare, lowercase, quote-stripped name.

    Handles both shapes sqlglot produces for a function call:
      - `exp.Anonymous` — any call sqlglot doesn't recognize as a built-in. This is what
        every Postgres-only function this gate cares about parses as: `pg_read_file`,
        `dblink_exec`, `nextval`, `pg_sleep`, ... . Its `.name` property already resolves
        the underlying `Identifier`'s text with quoting stripped, regardless of how the
        source spelled/quoted/cased it — `"pg_read_file"`, `PG_READ_FILE`, and
        `pg_read_file` all normalize to the same `.name`.
      - any other `exp.Func` subclass — a call sqlglot *does* recognize as a built-in
        (e.g. `exp.GenerateSeries`). Its canonical name lives on the class, not in an
        argument (`.name` alone returns empty for these), so `type(node).sql_name()` is
        used instead.

    Returns None for anything that isn't a function-call node, or whose name can't be
    resolved — callers must treat that as "nothing extracted", not "safe": the string-
    level regex layer and the overall fail-safe-on-parse-failure behavior remain in place
    as backstops for exactly this case.
    """
    if isinstance(node, exp.Anonymous):
        name = node.name
    elif isinstance(node, exp.Func):
        try:
            name = type(node).sql_name()
        except (NotImplementedError, AssertionError):
            return None
    else:
        return None
    # Defense in depth on top of sqlglot's own quote-stripping: peel any quoting
    # character sqlglot might not have fully unwrapped for a given dialect/quoting style
    # (double quotes, backticks, square brackets) so this can't be evaded by a quoting
    # style this gate's sqlglot version doesn't model perfectly.
    name = (name or "").strip().strip("\"'`[]")
    return name.lower() or None


def _ast_function_names(statements: list[exp.Expression], sql: str, dialect: str | None) -> set[str]:
    """Collect every normalized function-call name referenced anywhere in the parsed
    tree — the primary detection surface for the RCE/sequence/DoS deny lists (see module
    docstring). Walking the AST rather than the raw SQL text is what defeats quoted-
    identifier and case evasion: `"pg_read_file"(...)`, `PG_READ_FILE(...)`, and
    `pg_catalog."pg_read_file"(...)` all parse to a function-call node whose normalized
    name is `pg_read_file`, regardless of how the source text quoted or cased it — and
    `find_all` recurses through wrapper nodes like `exp.Dot` (schema qualification)
    automatically, so no separate handling is needed for the qualified form.

    Also *additionally* parses `sql` under the `postgres` dialect (unless that's already
    what was requested) and merges in any function names found there. This is purely
    additive — it can only add names, never remove ones the base parse already found — so
    a Postgres-only construct the caller's own dialect doesn't fully model still
    contributes its function names to the check.
    """
    names: set[str] = set()
    for stmt in statements:
        for node in stmt.find_all(exp.Func):
            name = _func_name(node)
            if name:
                names.add(name)

    if dialect != "postgres":
        pg_statements = _parse(sql, dialect="postgres")
        if pg_statements:
            for stmt in pg_statements:
                for node in stmt.find_all(exp.Func):
                    name = _func_name(node)
                    if name:
                        names.add(name)

    return names


class GovernanceGate:
    """Decides whether a proposed SQL statement may run — never executes it.

    Fail-safe by construction: anything that can't be confidently classified as a safe,
    bounded operation is blocked or held. See THREAT_MODEL.md for the risks each rule
    below maps to.
    """

    def evaluate(
        self,
        sql: str,
        *,
        mode: str = "read",
        actor: str | None = None,
        dialect: str | None = None,
    ) -> PolicyDecision:
        """Evaluate one proposed SQL statement against policy.

        Args:
            sql: The proposed SQL statement text (untrusted).
            mode: The intent the caller declares for this statement ("read" or "write").
                Reserved for future per-mode/per-actor policy scoping (e.g. Phase 2
                connection routing); v1's decision is driven entirely by what the SQL
                actually is, not by what the caller claims it is. Logged for audit.
            actor: Optional identifier of the proposing agent/user, for logging only.
            dialect: Optional sqlglot dialect (e.g. `"postgres"`) to parse `sql` under.
                When given, parsing is strict to that dialect only for classification —
                no generic/postgres fallback attempt. When omitted, behavior is
                unchanged from v1 (try the generic dialect, then postgres). Regardless of
                `dialect`, the AST-based dangerous-function check (see module docstring)
                additionally always attempts a `postgres` parse, so a Postgres-only
                construct still contributes its function names even when the caller
                asked for a different dialect or none. String-level RCE/injection regex
                detection also runs on the raw SQL text unconditionally, as a secondary
                layer — it is NOT relied on alone, since it can be evaded by identifier
                quoting; that's exactly why the AST-based function-name check is primary.

        Returns:
            A PolicyDecision. Never raises on malformed SQL — parse failure is itself
            a decision (block, classification "unknown").
        """
        string_flags = _string_level_flags(sql)

        # CRITICAL fix: Postgres `U&"..."`/`U&'...'` Unicode-escaped identifiers, which
        # `sqlglot` does not decode (see module docstring / `_unicode_escape_obfuscation_
        # decision`). Checked before parsing is even attempted so this catch-all applies
        # whether or not the rest of the statement happens to parse.
        obfuscation_decision = _unicode_escape_obfuscation_decision(sql, dialect, string_flags)
        if obfuscation_decision is not None:
            self._log(obfuscation_decision, sql, actor, mode)
            return obfuscation_decision

        # ATTACH/DETACH DATABASE: also raw-text-level and unconditional — see
        # `_attach_detach_decision`'s docstring for why this can't wait for `_parse`.
        attach_decision = _attach_detach_decision(sql, string_flags)
        if attach_decision is not None:
            self._log(attach_decision, sql, actor, mode)
            return attach_decision

        statements = _parse(sql, dialect=dialect)

        if statements is None:
            decision = PolicyDecision(
                action="block",
                classification="unknown",
                reason="Statement could not be parsed; failing safe (default deny).",
                matched_rules=["fail_safe_parse_error", *string_flags],
            )
            self._log(decision, sql, actor, mode)
            return decision

        if len(statements) > 1:
            classification, _ = _classify([statements[0]])
            decision = PolicyDecision(
                action="block",
                classification=classification,
                reason="Multiple SQL statements detected; only a single statement is permitted.",
                matched_rules=["multi_statement_rejected", *string_flags],
                tables=_tables(statements),
            )
            self._log(decision, sql, actor, mode)
            return decision

        classification, node = _classify(statements)
        tables = _tables(statements)

        matched_rules = list(string_flags)
        if _ast_tautology(statements) and "tautology_suspected" not in matched_rules:
            matched_rules.append("tautology_suspected")
        if _ast_system_catalog(statements) and "system_catalog_access" not in matched_rules:
            matched_rules.append("system_catalog_access")

        # AST-based function-name check (PRIMARY layer — see module docstring). Defeats
        # quoted-identifier / case / schema-qualification evasion of the string regexes
        # above: `"pg_read_file"(...)`, `PG_READ_FILE(...)`, and
        # `pg_catalog."pg_read_file"(...)` all normalize to the same function name here.
        ast_func_names = _ast_function_names(statements, sql, dialect)
        if (
            any(n in _RCE_FUNCTION_NAMES or n.startswith(_RCE_FUNCTION_PREFIX) for n in ast_func_names)
            and "db_native_rce_function" not in matched_rules
        ):
            matched_rules.append("db_native_rce_function")
        if (ast_func_names & _SEQUENCE_FUNCTION_NAMES) and "sequence_mutation" not in matched_rules:
            matched_rules.append("sequence_mutation")
        if (ast_func_names & _DOS_FUNCTION_NAMES) and "dos_suspected" not in matched_rules:
            matched_rules.append("dos_suspected")
        if (ast_func_names & _GUC_MUTATION_FUNCTION_NAMES) and "set_config_blocked" not in matched_rules:
            matched_rules.append("set_config_blocked")

        # Fail-safe backstop (layer 2 of the Unicode-escape fix — see module docstring):
        # even when no `U&` introducer was found in the raw text above, a function or
        # table name that still contains a literal backslash after normal AST resolution
        # is itself evidence of escape-based obfuscation this module's `U&` detector
        # doesn't recognize by name — block rather than silently normalize past it.
        if (
            any("\\" in n for n in ast_func_names) or any("\\" in t for t in tables)
        ) and "obfuscated_identifier" not in matched_rules:
            matched_rules.append("obfuscated_identifier")

        union_detected = _ast_union(statements)

        injection_rule_names = {
            "tautology_suspected",
            "sql_comment_present",
            "copy_to_program_rce",
            "db_native_rce_function",
            "system_catalog_access",
            "dos_suspected",
            "sequence_mutation",
            "obfuscated_identifier",
            "set_config_blocked",
        }
        if any(rule in injection_rule_names for rule in matched_rules):
            reason = (
                "GUC/session-configuration mutation function (set_config/pg_reload_conf) "
                "detected in a proposed SELECT; blocked outright."
                if "set_config_blocked" in matched_rules
                else "Injection, DB-native RCE, DoS, or exfiltration pattern detected in proposed SQL."
            )
            decision = PolicyDecision(
                action="block",
                classification=classification,
                reason=reason,
                matched_rules=matched_rules,
                tables=tables,
            )
            self._log(decision, sql, actor, mode)
            return decision

        # UNION/EXCEPT/INTERSECT: downgraded from a hard block to a hold (Phase 2
        # approval queue) — a bare UNION is not itself proof of an attack (legitimate
        # reporting queries combine result sets routinely), so treating every UNION as an
        # unconditional block was an over-blocking false-denial bug. It is still never
        # auto-allowed: a first branch that looks bounded (WHERE + LIMIT) doesn't make the
        # *combined* result set safe, since a second branch can pull arbitrary columns
        # from an arbitrary table — so this check overrides the normal bounded-SELECT
        # allow path below rather than deferring to it.
        if union_detected:
            matched_rules = [*matched_rules, "union_requires_review"]
            decision = PolicyDecision(
                action="hold",
                classification=classification,
                reason="UNION/EXCEPT/INTERSECT combines multiple result sets; held for human review "
                "rather than auto-blocked or auto-allowed.",
                matched_rules=matched_rules,
                tables=tables,
            )
            self._log(decision, sql, actor, mode)
            return decision

        decision = self._classify_decision(node, classification, matched_rules, tables)
        self._log(decision, sql, actor, mode)
        return decision

    def _classify_decision(
        self,
        node: exp.Expression,
        classification: Classification,
        matched_rules: list[str],
        tables: list[str],
    ) -> PolicyDecision:
        """Apply the per-classification rules once no injection/RCE/DoS pattern matched.

        `node` is the node the shape rules apply to: the outer statement for `read`, or
        the (possibly CTE/subquery-nested) inner mutation/DDL construct for `write`/`ddl`
        — see `_classify`.
        """
        if classification == "ddl":
            if isinstance(node, exp.Select) and node.args.get("into") is not None:
                matched_rules = [*matched_rules, "select_into_ddl"]
            matched_rules = [*matched_rules, "ddl_blocked"]
            return PolicyDecision(
                action="block",
                classification=classification,
                reason="DDL / destructive operations (CREATE, DROP, ALTER, TRUNCATE, SELECT INTO) are blocked.",
                matched_rules=matched_rules,
                tables=tables,
            )

        if classification == "write":
            if isinstance(node, exp.Merge):
                # MERGE can UPDATE/DELETE/INSERT matched rows in one statement driven by
                # an arbitrary join condition; there is no single "WHERE" to check for
                # boundedness the way there is for UPDATE/DELETE, so it is always blocked.
                matched_rules = [*matched_rules, "merge_blocked"]
                return PolicyDecision(
                    action="block",
                    classification=classification,
                    reason="MERGE statements are always blocked: matched-row actions cannot be bounded like a WHERE clause.",
                    matched_rules=matched_rules,
                    tables=tables,
                )

            if isinstance(node, exp.Insert):
                source = node.args.get("expression")
                if isinstance(source, (exp.Select, exp.Union, exp.Except, exp.Intersect)):
                    # INSERT ... SELECT copies an unbounded, unreviewed set of rows from
                    # one table into another (`INSERT INTO audit SELECT * FROM users`) —
                    # a duplication/exfiltration pattern, not a scoped, literal write.
                    matched_rules = [*matched_rules, "insert_select_unbounded"]
                    return PolicyDecision(
                        action="block",
                        classification=classification,
                        reason="INSERT ... SELECT copies an unbounded row set from another table; blocked.",
                        matched_rules=matched_rules,
                        tables=tables,
                    )

                conflict = node.args.get("conflict")
                if conflict is not None and "UPDATE" in str(conflict.args.get("action") or "").upper():
                    # `ON CONFLICT ... DO UPDATE` is an implicit, unconditioned UPDATE
                    # that fires on any future colliding row — it has no WHERE clause to
                    # check by construction, so it is always blocked.
                    matched_rules = [*matched_rules, "upsert_conflict_update_blocked"]
                    return PolicyDecision(
                        action="block",
                        classification=classification,
                        reason="INSERT ... ON CONFLICT DO UPDATE is an unconditioned upsert-mutation; blocked.",
                        matched_rules=matched_rules,
                        tables=tables,
                    )

                if not isinstance(node.this, exp.Schema):
                    # No explicit column list (`INSERT INTO users VALUES (...)` rather
                    # than `INSERT INTO users (col, ...) VALUES (...)`): this silently
                    # writes *every* column of the target table, including any the
                    # caller doesn't know about (e.g. an `admin` flag) — unscoped/blind,
                    # the INSERT analogue of a DELETE/UPDATE with no WHERE clause.
                    matched_rules = [*matched_rules, "insert_without_column_list"]
                    return PolicyDecision(
                        action="block",
                        classification=classification,
                        reason="INSERT without an explicit column list writes every column of the target table; blocked.",
                        matched_rules=matched_rules,
                        tables=tables,
                    )

                matched_rules = [*matched_rules, "write_requires_approval"]
                return PolicyDecision(
                    action="hold",
                    classification=classification,
                    reason="Writes require human approval (Phase 2 approval queue); held.",
                    matched_rules=matched_rules,
                    tables=tables,
                )

            # UPDATE / DELETE (possibly nested inside a CTE/subquery — `node` is the
            # inner statement itself, so this checks the real mutation's WHERE clause,
            # never a decoy WHERE on an outer wrapper).
            if not _has_where(node):
                matched_rules = [*matched_rules, "write_without_where"]
                return PolicyDecision(
                    action="block",
                    classification=classification,
                    reason="DELETE/UPDATE without a WHERE clause affects every row; blocked.",
                    matched_rules=matched_rules,
                    tables=tables,
                )

            matched_rules = [*matched_rules, "write_requires_approval"]
            return PolicyDecision(
                action="hold",
                classification=classification,
                reason="Writes require human approval (Phase 2 approval queue); held.",
                matched_rules=matched_rules,
                tables=tables,
            )

        if classification == "unknown":
            matched_rules = [*matched_rules, "unclassified_statement_type"]
            return PolicyDecision(
                action="block",
                classification=classification,
                reason="Statement type could not be classified as read/write/ddl; failing safe.",
                matched_rules=matched_rules,
                tables=tables,
            )

        # classification == "read"
        assert isinstance(node, (exp.Select, exp.Union, exp.Except, exp.Intersect))
        select_root = node if isinstance(node, exp.Select) else node.this
        has_where = isinstance(select_root, exp.Select) and _has_where(select_root)
        has_limit = isinstance(select_root, exp.Select) and _has_limit(select_root)
        is_star = isinstance(select_root, exp.Select) and _is_select_star(select_root)

        if has_where and has_limit:
            if is_star:
                matched_rules = [*matched_rules, "pii_suspected"]
            return PolicyDecision(
                action="allow",
                classification=classification,
                reason="Bounded SELECT: has WHERE and LIMIT.",
                matched_rules=matched_rules,
                tables=tables,
            )

        missing = []
        if not has_where:
            missing.append("missing_where")
        if not has_limit:
            missing.append("missing_limit")
        return PolicyDecision(
            action="hold",
            classification=classification,
            reason="Unbounded SELECT (missing WHERE and/or LIMIT); held for review.",
            matched_rules=[*matched_rules, *missing],
            tables=tables,
        )

    @staticmethod
    def _log(decision: PolicyDecision, sql: str, actor: str | None, mode: str = "read") -> None:
        log = logger.warning if decision.action in ("block", "hold") else logger.info
        log(
            "sql_governance.decision",
            action=decision.action,
            classification=decision.classification,
            reason=decision.reason,
            matched_rules=decision.matched_rules,
            tables=decision.tables,
            actor=actor,
            mode=mode,
            sql_preview=sql[:120],
        )
