"""PII redaction for query results — the confidentiality control on the `allow` path.

THREAT_MODEL.md's R3 (PII / data exfiltration) is only flagged, not stopped, by the
Phase-0 policy gate: a bounded `SELECT ... WHERE ... LIMIT` is still eligible for `allow`
even when it targets an `email`/`ssn`/`phone` column, or does a bare `SELECT *` over a
table with unreviewed PII columns. This module is where that data actually gets masked
before it leaves the gateway, using two layers together so neither a mis-scoped column
nor an oddly-aliased one slips through alone:

  1. **Column-name classification** — a column whose name contains a PII keyword
     (`email`, `mail`, `ssn`, `social`, `phone`, `mobile`, `card`, `credit`, `dob`,
     `birth`, `address`) has every non-empty value in it masked outright, regardless of
     what the value looks like. This catches the common case cheaply and catches PII
     even in values that don't match any known *pattern* (e.g. a free-text address).
  2. **Value-pattern scanning** — every value in every column *not* already classified by
     name is additionally scanned against the same PII regexes `guardrails.py` uses to
     scrub LLM output, plus an SSN pattern guardrails does not define (this module's
     contract explicitly requires SSN masking, and `_PII_PATTERNS` has no SSN regex), plus
     a narrowly-scoped dashless-SSN/contextual-numeric pattern (see below). This is what
     catches PII sitting in an oddly-named or aggregated/aliased column a name-based
     classifier has no way to see — `GROUP_CONCAT(email)` (whose column name still
     contains "email" and so is actually caught by layer 1 too) and, critically, a column
     named `col1` or an unlabeled expression result, whose *name* carries no signal at all
     but whose *value* is still a real email address, card number, etc. This scan runs
     regardless of the value's Python type — `str`, `int`, and `float` values are all
     coerced with `str()` before scanning (only `None` and the two types handled by their
     own dedicated layers below, `bytes` and JSON-shaped strings, are skipped) — so an SSN
     or card number stored as a numeric column (`ssn_num INTEGER`) cannot dodge this layer
     purely by not being a `str` at the Python level.

     The dashless-SSN signal deliberately does **not** blanket-match every bare 9-digit
     value the way the dashed `\\d{3}-\\d{2}-\\d{4}` pattern matches every dashed one — a
     table's ordinary integer id/order-number columns are routinely 9 digits, and
     redacting every one of them would make the gateway useless for ordinary analytics.
     Instead it is a *contextual* match: a 9-digit number is only treated as an SSN by
     this pattern when it's immediately preceded, in the same text, by an SSN/PII-account
     keyword and a comparison operator — `ssn = 123456789`, `ssn IN (123456789)` — which
     is precisely the shape `audit.py`'s `redact_text` needs for `proposed_sql` (a
     `WHERE ssn=123456789` literal must not persist raw in the audit log) and is safe to
     apply universally since it can never fire on a bare, contextless numeric value. A
     numeric SSN sitting in a column already classified PII by name/provenance (layer 1
     or 3) needs no pattern at all — that layer masks the whole value unconditionally,
     regardless of format — so this pattern's job is specifically the *contextless
     value* and *raw-SQL-text* cases layers 1/3 cannot reach.
  3. **Provenance (lineage) analysis** — layers 1 and 2 both only look at what a query
     *returns*: a name match on the output column, or a pattern match on the output
     value. Neither catches `SUBSTR(ssn, 1, 3) AS s` when the substring itself happens
     not to look like a full SSN, or `GROUP_CONCAT(email) AS c` when the concatenated
     result happens to dodge the email regex (e.g. a single very short local part). This
     layer instead traces each SELECT output expression, via `sqlglot`'s parsed AST, back
     to the *source* column(s) it references — `SUBSTR(ssn, ...)` references `ssn`,
     `GROUP_CONCAT(email)` references `email` — and redacts the corresponding *output*
     column whenever any referenced source column is itself PII-classified, regardless of
     what the output is named or how it's transformed. Tracing follows nested `SELECT`
     scopes too — a subquery in `FROM` or a CTE — so `SELECT c FROM (SELECT SUBSTR(email,
     1, 5) AS c FROM users) t` and `WITH t AS (SELECT SUBSTR(ssn, 1, 3) AS c FROM users)
     SELECT c FROM t` are both traced through to `email`/`ssn` even though neither the
     outer column's name (`c`) nor its immediate expression (a bare column reference)
     carries any PII signal on its own.

     Before walking the projections, this layer first tries `sqlglot.optimizer.qualify
     .qualify(..., expand_stars=True, infer_schema=True)` on the query as a whole (this
     closes the two confirmed HIGH leak shapes below). A top-level `SELECT *`/`alias.*`
     over a *derived* table (subquery-in-`FROM` or CTE) is self-describing — sqlglot can
     expand it from that derived table's own SELECT list with no real database schema
     needed — so `SELECT * FROM (SELECT ssn_num AS c FROM users) t` is expanded to `t.c`
     and traced through to `ssn_num`, instead of being wrongly assumed safe the same way
     a `SELECT *` directly over a *base* table is (a base table's wildcard expansion does
     keep the source's own column names, which layer 1 already classifies correctly on
     its own; a derived table's does not — its output names are whatever the derived
     SELECT aliased them to). Second, an *unqualified* column reference amid more than
     one `JOIN` source is qualified to the one specific source sqlglot can prove it
     resolves to (e.g. only one side is a derived table that actually projects a column
     of that name) rather than assumed ambiguous — closing the exact shape of
     `SELECT s AS c FROM (SELECT ssn_num AS s FROM users) a JOIN users b ON b.id=1`,
     where `s` only exists on the `a` side. When `qualify()` either raises (a construct
     it can't resolve/support, including a genuinely ambiguous unqualified column among
     more than one base-table source neither it nor this module has real schema for) or
     still leaves a star unexpanded (one whose source's real schema truly is unknown),
     this layer falls back to walking the *original*, unqualified tree instead, under the
     same rules as before: whenever that walk hits a point it cannot resolve with
     confidence (a nested `SELECT *` needing schema to expand, a `UNION`-shaped derived
     source, an alias with no matching projection), it **fails closed** — the output
     column is treated as PII rather than silently passed through — except for one
     narrow, deliberate exception, now reached only on this fallback path: an unqualified
     column ambiguous among more than one source falls back to classifying the column's
     own name directly, the same as a base-table reference would, rather than redacting
     every unqualified column in the query — name-classifying it directly cannot hide
     anything a qualified reference to the same column wouldn't already have exposed.
     A third confirmed HIGH leak shape sits at the boundary of the two: a top-level
     `SELECT *` that spans BOTH a derived source AND a base table of unknown schema
     (`SELECT * FROM (SELECT ssn_num AS c FROM users) t JOIN orders o ON o.user_id=1`)
     can't be expanded by `qualify()` at all — the base table's missing schema blocks
     expansion of the star as a whole, not just its own side — so the walk above still
     reaches an unexpanded top-level star. Rather than failing open outright there too,
     this fallback additionally walks every derived source's own SELECT list
     (`_derived_pii_output_columns`, recursively through any further-nested derived
     source) and adds any of *its* output column names that are PII by this same
     lineage logic to the redacted set — closing the leak for the derived side (`c`)
     while leaving the base table's own columns (`orders.id`, `.total`, `.note`, ...) to
     the name/value layers that already classify them correctly by their real names.
     A fourth confirmed HIGH leak shape sits beside that one, not inside it: a top-level
     `SELECT *` with a *non-star* projection directly beside the star in the same SELECT
     list — `SELECT *, (SELECT ssn_num FROM users WHERE id=1) AS c FROM orders ...`, and
     equally a bare function call or arithmetic expression in that position — is invisible
     to `_derived_pii_output_columns` above, since a scalar subquery/function/arithmetic
     projection is none of the FROM/JOIN/CTE derived sources that helper walks. Whenever a
     top-level star is reached, every other, non-star top-level projection is therefore
     ALSO resolved with the same per-projection lineage (`_projection_is_pii`), failing
     closed on `_ProvenanceUnresolved` exactly like the non-star SELECT path below does,
     and its alias unioned into the redacted set — closing the leak for the whole
     projection *class* beside a star, not just the one scalar-subquery shape, while a
     non-PII non-star projection beside a star (`SELECT *, total AS c FROM orders`)
     resolves through the same lineage to not-PII and is left unredacted.
     `COUNT(...)` — including `COUNT(*)` and `COUNT(some_pii_col)` — is exempted
     regardless of what it counts, since a row count reveals no individual value;
     `GROUP_CONCAT`/`MIN`/`MAX`/`SUBSTR`/string-aggregation over a PII column are
     deliberately NOT exempted, since they do leak the underlying value(s). This is a
     best-effort, single-statement analysis (see `_provenance_pii_columns`'s docstring for
     its exact scope and remaining fail-open conditions); it is additive on top of layers
     1 and 2, never a replacement.

     A fifth confirmed leak, of a different kind than the first four, is what finally
     ended the shape-by-shape chase: `SELECT *, s AS renamed FROM (SELECT ssn_num AS s
     FROM users) a JOIN orders o ON o.user_id=1 WHERE o.id>0 LIMIT 5` — a non-star
     projection beside a star (layer 3's fourth fix, above) that is itself just a bare,
     *unqualified* column reference (`s`), ambiguous between the derived source `a`
     (which actually projects `s`, tracing to `ssn_num`) and the base table `orders`
     (which doesn't). `_column_is_pii`'s deliberately permissive fallback for exactly
     this ambiguous-unqualified-column case — kept, pre-fix, so an ordinary multi-table
     `JOIN` isn't over-redacted just because this best-effort analysis can't tell which
     side an unqualified column came from — name-classified the *alias* `renamed`
     directly, found no PII keyword in it, and let the raw SSN through. Each of the
     first four fixes closed one specific *shape* this ambiguity could hide behind; this
     one generalizes instead of adding a sixth shape-specific patch: `redact_rows`'s
     `sql` argument is checked once, up front, by `_query_references_pii_source` (true
     if the statement references a PII-named column *anywhere*, at any depth). Only when
     that is true does `_column_is_pii`'s ambiguous-column fallback stop being
     permissive and start failing closed (`strict=True`, threaded through every
     resolution call in this layer) — so ANY top-level output column in a PII-touching
     query that this best-effort analysis cannot *positively* trace to a proven-non-PII
     base column is redacted, regardless of which particular JOIN/star/alias shape hides
     it, current or future. A query that references no PII column anywhere at all keeps
     the old, more permissive behavior unchanged, so ordinary PII-free analytics queries
     over an ambiguous JOIN are still not over-redacted. See `_column_is_pii`'s and
     `_provenance_pii_columns`'s docstrings for the exact mechanics.

     A sixth confirmed leak sat entirely outside what any of the first five could reach,
     because it isn't a shape *within the proposed query's own text* at all: `CREATE VIEW
     vnum AS SELECT id AS uid, ssn_num AS token FROM users; SELECT token FROM vnum WHERE
     uid=1 LIMIT 5`. A view is, to a query that merely names it, indistinguishable from an
     ordinary base table — `token` has no PII-looking name, `ssn_num` (the column that
     actually feeds it) never appears anywhere in the *query text* `_provenance_pii_columns`
     parses, and a bare 9-digit int has no dashes/keyword-prefix for the value-pattern
     layer to catch either. Closing this required a source of truth this module never had
     before: `redact_rows`'s new optional `schema` parameter (a `SchemaCatalog` — see
     `backend.core.schema_catalog` and `gateway.py`'s best-effort introspection of the live
     database connection). When supplied, `_provenance_pii_columns` first inlines every
     known view reference in `sql` with that view's own real `SELECT` body — recursing for
     a view built on another view — via `_inline_views`/`_resolve_view_select`, BEFORE
     `_query_references_pii_source` or either resolution pass runs. From that point on, a
     view reference is, to the rest of this layer's existing lineage machinery, literally
     just another derived subquery-in-`FROM` — the same shape the fourth/fifth fixes above
     already trace correctly — so no view-specific redaction logic exists anywhere past
     that one substitution step. The same `schema` also supplies `schema.tables` to
     `qualify(schema=...)` in `_qualify_root_for_provenance`, letting a `SELECT *`/
     ambiguous JOIN column over an actual BASE table resolve deterministically too,
     retiring the schema-less fallback heuristics (`_derived_pii_output_columns`, the
     permissive ambiguous-column classification) for exactly the queries where real schema
     is available — they remain in place, unchanged, as the fallback for when it isn't.
     `schema=None` (no live connection/schema available — the default, and the ONLY
     behavior every pre-existing caller/test in this module exercises) is a complete no-op
     for both of these: identical output to before this fix existed. An unresolvable known
     view (its stored body doesn't parse, or a cyclical view chain) fails the WHOLE query's
     result set closed rather than attempting to bound which specific output columns it
     could have tainted — see `_inline_views`'s docstring for why that coarser guarantee is
     the honest one this analysis can actually make for that case.
  4. **JSON/BLOB defaulting** — a column whose value is `bytes` (a SQL `BLOB`) or a string
     that parses as a JSON object/array is redacted outright, on the assumption that
     either can carry arbitrary nested PII no name/value-regex/provenance layer above can
     see into (a JSON blob's field names and values are invisible to all three).

Deliberately conservative in the other direction too: a column/value that matches none of
these layers passes through completely unchanged. `id`, `admin`, and `age` are not
redacted — over-redacting non-PII data would make the gateway useless for the ordinary
analytics queries it exists to allow. This conservatism is now itself bounded by layer 3's
generalized fail-closed policy: it only holds for queries that reference no PII-named
column anywhere; once a query does touch PII, an output column must be positively
resolved as non-PII (not merely "not disprovable") to stay unredacted.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, build_scope

from backend.core.guardrails import _PII_PATTERNS
from backend.core.schema_catalog import SchemaCatalog

# Column-name keywords (case-insensitive substring match) that mark every value in that
# column as PII, regardless of what the value itself looks like.
_PII_COLUMN_KEYWORDS: tuple[str, ...] = (
    "email",
    "mail",
    "ssn",
    "social",
    "phone",
    "mobile",
    "card",
    "credit",
    "dob",
    "birth",
    "address",
)

# Value-level scan patterns: guardrails.py's own output-scrubbing patterns (email, phone,
# card, secret, filesystem path) plus an SSN pattern this module adds, since guardrails
# has no SSN regex and this module's contract explicitly requires SSN masking.
_SSN_PATTERN: tuple[re.Pattern[str], str] = (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN_REDACTED]")

# Dashless-SSN / numeric-PII signal — deliberately CONTEXTUAL, not a blanket bare-9-digit
# match (see module docstring layer 2 for why): only fires when a 9-plus-digit number is
# immediately preceded, in the same text, by a PII/account keyword and a comparison
# operator or `IN`/`IN (` — i.e. it looks like `<keyword> <op> <number>` in raw SQL text
# (`ssn = 123456789`, `ssn IN (123456789)`, `account>=100200300`) or an equivalently-shaped
# "labeled value" string. This is what closes the MEDIUM gap in `audit.py`'s
# `redact_text(proposed_sql)`: a literal `WHERE ssn=123456789` must not persist raw in the
# audit log even though `123456789` has no dashes for `_SSN_PATTERN` above to match. The
# keyword/operator prefix requirement is the bound that keeps this from over-redacting an
# ordinary 9-digit order id or similar with no such prefix — those never match at all.
# Capture groups 1 (keyword) and 2 (operator/whitespace) are preserved in the replacement;
# only the number itself is masked.
_CONTEXTUAL_NUMERIC_PII_PATTERN: tuple[re.Pattern[str], str] = (
    re.compile(
        r"\b(ssn|social|card|ccn|account)\b(\s*(?:=|!=|<>|<=|>=|<|>|in\s*\(?)\s*)\d{3,}",
        re.IGNORECASE,
    ),
    r"\1\2[REDACTED]",
)

_VALUE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    *_PII_PATTERNS,
    _SSN_PATTERN,
    _CONTEXTUAL_NUMERIC_PII_PATTERN,
]

_REDACTED_PLACEHOLDER = "[REDACTED]"


class RedactionReport(TypedDict):
    """Report returned alongside redacted rows: which columns had anything masked, and
    how many individual values were masked in total (across all rows)."""
    columns_redacted: list[str]
    values_masked: int


def _is_pii_column(column_name: str) -> bool:
    lname = column_name.lower()
    return any(keyword in lname for keyword in _PII_COLUMN_KEYWORDS)


# Dialects attempted when parsing `sql` for provenance analysis — same order/rationale as
# `sql_governance.py`'s `_PARSE_DIALECTS`: generic first, postgres as a fallback for
# constructs the generic dialect can't model.
_PROVENANCE_PARSE_DIALECTS: tuple[str | None, ...] = (None, "postgres")


# Maximum nested-scope recursion depth `_provenance_pii_columns` will follow while
# tracing an output column back through subqueries/CTEs before giving up and failing
# closed (`_ProvenanceUnresolved`) — a safety valve against a pathologically deep chain
# of derived tables, not a limit any legitimate analytics query is expected to approach.
_MAX_PROVENANCE_DEPTH = 12


class _ProvenanceUnresolved(Exception):
    """Internal control-flow signal only — never escapes `_provenance_pii_columns`.

    Raised whenever tracing an output column's lineage reaches a point that cannot be
    resolved with confidence: a nested `SELECT *` that would need real schema knowledge
    to expand, a `UNION`/`EXCEPT`/`INTERSECT`-shaped derived source (no single SELECT to
    trace into), a derived source with no projection matching the column being resolved,
    a scalar/correlated subquery that doesn't project exactly one real column, or the
    `_MAX_PROVENANCE_DEPTH` recursion cap. The caller (`_provenance_pii_columns`) treats
    this exactly like "resolved to PII" — fail CLOSED, not fail open — which is what
    closes the HIGH #1 gap: a derived/hidden PII column this best-effort analysis can't
    positively prove safe must never silently pass through as not-PII.
    """


def _is_star_projection(node: exp.Expression) -> bool:
    return isinstance(node, exp.Star) or (isinstance(node, exp.Column) and isinstance(node.this, exp.Star))


def _projection_is_pii(expr: exp.Expression, scope: Scope, depth: int, strict: bool) -> bool:
    """True if `expr` — one SELECT projection/expression, evaluated within `scope` — is
    fed by at least one PII-classified base column, tracing through nested subqueries/
    CTEs as far as needed. Raises `_ProvenanceUnresolved` (fail closed — see that
    class's docstring) rather than returning `False` whenever tracing can't be completed
    with confidence.

    `strict` is threaded straight through to `_column_is_pii` — see that function's
    docstring and the module docstring's "generalized fail-closed" layer for what it
    changes: whether an ambiguous/unresolvable column reference is allowed to fall back
    to a permissive name-classification (`strict=False`) or must fail closed instead
    (`strict=True`).

    `COUNT(...)` — including `COUNT(*)`, `COUNT(some_pii_column)`, and
    `COUNT(DISTINCT some_pii_column)` — is special-cased to never count as PII on its
    own, regardless of what it counts: a row count reveals no individual value, unlike
    `GROUP_CONCAT`/`MIN`/`MAX`/`SUBSTR`/string-aggregation over a PII column, which DO
    leak the underlying value(s) and are deliberately NOT exempted here.
    """
    if depth > _MAX_PROVENANCE_DEPTH:
        raise _ProvenanceUnresolved()
    target = expr.this if isinstance(expr, exp.Alias) else expr
    if isinstance(target, exp.Count):
        return False
    return _node_is_pii(target, scope, depth, strict)


def _node_is_pii(node: exp.Expression, scope: Scope, depth: int, strict: bool) -> bool:
    """Recursively walk `node`'s own child expressions (not `find_all`, which would flatten
    across scope boundaries) looking for a PII-resolving `Column`, switching to a fresh
    scope whenever a nested `SELECT` (a correlated/scalar subquery embedded directly in an
    expression, e.g. `(SELECT ssn FROM other WHERE other.id = users.id)`) is encountered.
    """
    if depth > _MAX_PROVENANCE_DEPTH:
        raise _ProvenanceUnresolved()
    if isinstance(node, exp.Column):
        return _column_is_pii(node, scope, depth, strict)
    if isinstance(node, exp.Subquery):
        inner = node.this
        if not isinstance(inner, exp.Select):
            raise _ProvenanceUnresolved()
        return _scalar_select_is_pii(inner, depth, strict)
    if isinstance(node, exp.Select):
        return _scalar_select_is_pii(node, depth, strict)
    if isinstance(node, exp.Star):
        # A bare `*` reached outside of a `COUNT(*)` (already special-cased above) —
        # e.g. some other aggregate over `*` — carries no column name to classify and no
        # schema to expand against; cannot prove safe.
        raise _ProvenanceUnresolved()
    found = False
    for child in node.iter_expressions():
        if _node_is_pii(child, scope, depth + 1, strict):
            found = True
    return found


def _scalar_select_is_pii(select_node: exp.Select, depth: int, strict: bool) -> bool:
    """A `SELECT` reached as an expression (a correlated/scalar subquery embedded
    directly inside a projection) gets its own fresh scope, and must project exactly one
    real (non-star) column to be resolvable at all — anything else fails closed.
    """
    projections = select_node.selects
    real = [p for p in projections if not _is_star_projection(p)]
    if len(real) != 1 or len(real) != len(projections):
        raise _ProvenanceUnresolved()
    inner_scope = build_scope(select_node)
    if inner_scope is None:
        raise _ProvenanceUnresolved()
    return _projection_is_pii(real[0], inner_scope, depth + 1, strict)


def _column_is_pii(col: exp.Column, scope: Scope, depth: int, strict: bool) -> bool:
    """Resolve one `Column` reference to the source it reads from within `scope`.

      - Qualified (`t.c`) or the sole source in scope: resolved unambiguously. If the
        source is a base table, classify `col`'s own name (`_is_pii_column`) — the
        common case, matching pre-fix behavior exactly. If the source is itself a
        derived table (a subquery-in-`FROM` or a CTE — `sqlglot`'s `Scope.sources`
        represents both identically as a nested `Scope`), find the matching projection
        in that derived table's own SELECT by output alias name and recurse into it —
        this is the HIGH #1 fix: an output column whose own name/alias carries no PII
        signal (`c`) is no longer taken at face value when its source is itself e.g.
        `SUBSTR(email, 1, 5) AS c`.
      - Unqualified with more than one source in scope, or qualified to a table alias
        that cannot be resolved in `scope` at all (an ambiguous/unresolvable reference
        this module has no schema to disambiguate, e.g. an ordinary multi-table `JOIN`):
        when `strict` is `False`, falls back to classifying `col`'s own name directly,
        the same as a base-table column — a deliberate, narrow exception to fail-closed,
        kept ONLY for queries that reference no PII source anywhere (see
        `_query_references_pii_source`), so an ordinary PII-free analytics query over an
        ambiguous JOIN is never over-redacted. When `strict` is `True` — any query that
        *does* reference a PII-named column somewhere, per `_query_references_pii_source`
        — this ambiguity instead raises `_ProvenanceUnresolved` (fail closed): this is
        the generalized fix for the confirmed leak class where an unqualified column
        beside/among a PII-touching JOIN resolves, by sheer alias-name coincidence, to
        something that doesn't look PII (`s AS renamed`) even though it demonstrably is.
        A column that resolves unambiguously (a single source, or a qualified reference
        to a known source) is unaffected by `strict` either way — this branch is reached
        only for the genuinely ambiguous/unresolvable case.

        NOTE: `_provenance_pii_columns` only ever reaches this function with an
        *unqualified, ambiguous* column when its own `qualify()`-based resolution
        attempt (see that function's docstring) already failed to disambiguate it —
        i.e. sqlglot itself could not prove the reference resolves to one specific
        source either. This branch is the fallback for that residual case, not the
        primary resolution path.

    Fails closed for the specific indirection this analysis cannot see through at all —
    see `_ProvenanceUnresolved`'s docstring for the exact conditions — and, when
    `strict` is `True`, additionally for the ambiguous/unresolvable-reference case above
    that `strict=False` would otherwise permissively name-classify.
    """
    if depth > _MAX_PROVENANCE_DEPTH:
        raise _ProvenanceUnresolved()

    sources = scope.sources
    table_alias = col.table
    if table_alias:
        source = sources.get(table_alias)
        if source is None:
            source = next((v for k, v in sources.items() if k.lower() == table_alias.lower()), None)
        if source is None:
            if strict:
                raise _ProvenanceUnresolved()
            return _is_pii_column(col.name)
    elif len(sources) == 1:
        source = next(iter(sources.values()))
    else:
        # Zero or ambiguous (>1) sources for an unqualified column — see docstring.
        if strict:
            raise _ProvenanceUnresolved()
        return _is_pii_column(col.name)

    if not isinstance(source, Scope):
        # A base table (or anything else sqlglot's scope resolution didn't model as a
        # nested Scope) — classify the column's own name directly. Unambiguously
        # resolved, so unaffected by `strict`: this IS the "positively resolved to a
        # base column" case the fail-closed policy is deliberately built around, not the
        # ambiguity it targets.
        return _is_pii_column(col.name)

    inner_select = source.expression
    if not isinstance(inner_select, exp.Select):
        # A UNION/EXCEPT/INTERSECT-shaped derived source: no single SELECT to trace a
        # specific output column into. Cannot prove either branch is safe.
        raise _ProvenanceUnresolved()

    matching = None
    for projection in inner_select.selects:
        if _is_star_projection(projection):
            # A nested `SELECT *` would need schema-based expansion to know which real
            # column feeds `col.name` — this analysis has no schema. Cannot prove safe.
            raise _ProvenanceUnresolved()
        if projection.alias_or_name == col.name:
            matching = projection
            break
    if matching is None:
        # No projection in the derived source's SELECT list is named `col.name` at all
        # (should not normally happen for SQL that actually executes) — cannot resolve.
        raise _ProvenanceUnresolved()

    return _projection_is_pii(matching, source, depth + 1, strict)


class _ViewResolutionFailed(Exception):
    """Internal control-flow signal only — never escapes this module.

    Raised by `_resolve_view_select`'s inner `transform()` callback to unwind out of
    `sqlglot`'s tree walk the instant a nested view reference can't be resolved (its
    stored body doesn't parse to a single `SELECT`, or a view refers back to one
    already being resolved on the same chain — a cyclical view definition). Caught
    immediately around the `transform()` call that raised it; `_resolve_view_select`
    turns it into a plain `None` return, and `_inline_views` turns THAT into its own
    `unresolved` flag — see both docstrings.
    """


def _resolve_view_select(
    lname: str,
    views: dict[str, str],
    unresolved_view_names: set[str],
    dialect: str | None,
    chain: frozenset[str],
) -> exp.Select | None:
    """Parse `views[lname]` (the view's own defining `SELECT`, from `SchemaCatalog
    .views`) and recursively inline any further view references inside IT (view-on-view
    — `CREATE VIEW v2 AS SELECT uid, token FROM v1` where `v1` is itself a known view),
    so the returned tree contains no unresolved reference to another known view at all.

    `unresolved_view_names` (from `SchemaCatalog.unresolved_views`, already
    lowercase-normalized by `_inline_views`) is checked for every nested `Table`
    reference found while resolving `lname`'s own body, not only at the top level — a
    view that itself references an UNRESOLVABLE view (one whose own body didn't parse
    to a single `SELECT`, e.g. a `UNION`-shaped view two layers down) must fail exactly
    as closed as a query that references that unresolvable view directly; without this
    check, a nested reference to a name that is neither in `views` NOR checked against
    `unresolved_view_names` would be silently treated as an ordinary, unknown BASE
    table (a real fail-open gap this parameter exists to close — see
    `_inline_views`'s own top-level check, which this mirrors for the recursive case).

    `chain` is the set of view names already being resolved on the current path from
    the outermost query down to this call — a view that (directly or transitively)
    refers to one of its own ancestors is a cyclical definition (should never happen
    for SQL that actually executes, but this module has no way to prove that without
    walking it) and is treated exactly like an unparseable body: this returns `None`.

    Returns `None` — never raises — for anything it can't fully resolve: the stored SQL
    doesn't parse under `dialect`, it isn't a single top-level `SELECT` (e.g. a
    `UNION`-shaped view), or a nested view reference within it fails to resolve
    (including the cycle and unresolvable-nested-view cases above). The caller,
    `_inline_views`, treats a `None` result as "this specific view is unresolvable" and
    fails the WHOLE query closed (see that function's docstring for why a query-wide
    fail-closed, not a per-column one, is what this module can actually guarantee for
    this case).
    """
    sql = views.get(lname)
    if not sql:
        return None
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
    except Exception:  # sqlglot's own ParseError/TokenizeError subclasses
        return None
    if not isinstance(parsed, exp.Select):
        return None

    def _replace(node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.Table) or not node.name:
            return node
        inner_lname = node.name.lower()
        if inner_lname in unresolved_view_names:
            raise _ViewResolutionFailed()
        if inner_lname not in views:
            return node
        if inner_lname in chain:
            raise _ViewResolutionFailed()
        inner = _resolve_view_select(inner_lname, views, unresolved_view_names, dialect, chain | {inner_lname})
        if inner is None:
            raise _ViewResolutionFailed()
        return inner.subquery(alias=node.alias_or_name, copy=False)

    try:
        resolved = parsed.transform(_replace)
    except _ViewResolutionFailed:
        return None
    return resolved if isinstance(resolved, exp.Select) else None


def _inline_views(root: exp.Select, schema: SchemaCatalog | None, dialect: str | None) -> tuple[exp.Select, bool]:
    """Replace every `exp.Table` reference to a known view — anywhere in `root`'s tree,
    at any depth (the top-level `FROM`/`JOIN`, and every nested subquery/CTE's own) —
    with a derived subquery wrapping that view's own parsed `SELECT` body, recursing for
    a view built on another view (`_resolve_view_select`). This is the fix for the
    confirmed leak this module's docstring layer 3 addition documents: a view is
    otherwise an opaque base-table reference to `redact_rows`'s provenance analysis,
    since that analysis only ever parses the *proposed query's* own text — `CREATE VIEW
    vnum AS SELECT id AS uid, ssn_num AS token FROM users; SELECT token FROM vnum` has
    no PII `exp.Column` anywhere in the *query itself* to trace. Once this substitution
    runs, the rest of this module's existing lineage machinery — which already traces
    correctly through a subquery-in-`FROM`/CTE — needs no view-specific logic at all;
    `token` is now literally `(SELECT id AS uid, ssn_num AS token FROM users) AS vnum`,
    exactly the derived-source shape `_column_is_pii` already resolves.

    A `Table` reference is matched to a view by name (case-insensitively) ONLY when that
    name is not also a CTE alias the query itself defines (`root.find_all(exp.CTE)`,
    walked once up front) — a CTE shadows a same-named view within its own scope by
    ordinary SQL semantics, and `sqlglot`'s own scope resolution already handles a CTE
    reference correctly as a derived source; inlining the view there would silently
    substitute the wrong source. `schema=None` (no live schema available at all — see
    `redact_rows`'s docstring) or a schema with no views at all makes this a no-op,
    returning `root` unchanged — the existing schema-less behavior is preserved exactly
    for a caller/test that doesn't supply a schema.

    Returns `(new_root, unresolved)`. `unresolved` is `True` iff at least one `Table`
    reference in the query names a KNOWN view (present in `schema.views` or
    `schema.unresolved_views`) whose body this module could not resolve — an
    unparseable view definition, or a cyclical one (see `_resolve_view_select`). Rather
    than trying to bound exactly which *output columns* such a reference could taint
    (star expansions, JOINs, and nested scopes all make that genuinely unbounded without
    knowing the unresolvable view's real shape), the caller fails the WHOLE query's
    result set closed when this is `True` — the deliberately coarse, but honest, answer
    to "no crash if a view is unparseable (fail closed for that source)": every output
    column is potentially "that source" once a query touches an unresolvable view
    anywhere in its FROM/JOIN/CTE graph.
    """
    if root is None:
        return root, False
    # `SchemaCatalog` stores names exactly as the database reports them (its own
    # documented contract puts case-insensitive lookup on the caller) — normalize to
    # lowercase HERE, once, for both dicts, so every lookup below (`node.name.lower()`,
    # here and in `_resolve_view_select`) is comparing against a consistently-cased key.
    # Without this, a view declared with any uppercase in its name (`CREATE VIEW
    # VNUM_UPPER AS ...`) would never match at all — silently skipped as "not a known
    # view" rather than inlined or even flagged unresolved, a real fail-OPEN gap this
    # normalization exists specifically to close.
    views = {name.lower(): sql for name, sql in (schema.views if schema else {}).items()}
    unresolved_view_names = {name.lower() for name in (schema.unresolved_views if schema else frozenset())}
    if not views and not unresolved_view_names:
        return root, False

    cte_names = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
    state = {"unresolved": False}

    def _replace(node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.Table) or not node.name:
            return node
        lname = node.name.lower()
        if lname in cte_names:
            return node
        if lname in unresolved_view_names:
            state["unresolved"] = True
            return node
        if lname not in views:
            return node
        resolved = _resolve_view_select(lname, views, unresolved_view_names, dialect, frozenset({lname}))
        if resolved is None:
            state["unresolved"] = True
            return node
        return resolved.subquery(alias=node.alias_or_name, copy=False)

    new_root = root.transform(_replace)
    if not isinstance(new_root, exp.Select):
        # Defensive only — a Select's own transform() replaces internal nodes, never
        # the root's own type, since `_replace` never matches a bare `exp.Select`.
        return root, state["unresolved"]
    return new_root, state["unresolved"]


def _qualify_root_for_provenance(
    root: exp.Select, dialect: str | None, schema: dict[str, dict[str, str]] | None
) -> exp.Select | None:
    """Attempt to fully qualify and expand `root` via sqlglot's own optimizer
    (`qualify(..., expand_stars=True, infer_schema=True, schema=schema)`) so every
    column reference becomes an explicit `source.column` and every top-level
    `SELECT *`/`alias.*` is expanded to the real columns it projects — this is the
    HIGH #3 fix (see module docstring layer 3 for the full rationale). Three things make
    this safe/effective to attempt even with `schema=None` (no real database schema
    available at all):

      - A star over a *derived* table (subquery-in-`FROM`/CTE — including one produced
        by `_inline_views` in place of a bare view reference) is self-describing —
        sqlglot expands it straight from that derived table's own SELECT list, real
        schema or not.
      - An unqualified column amid more than one `JOIN` source is only qualified when
        sqlglot can prove it resolves to exactly one specific source (e.g. only one side
        actually projects a column of that name); a *genuinely* ambiguous reference makes
        `qualify()` raise rather than guess.
      - When `schema` IS supplied (`redact_rows`'s caller had a live `SchemaCatalog` —
        see its docstring and `schema_catalog.py`), a `SELECT *`/ambiguous JOIN column
        over an actual BASE table now resolves deterministically too, rather than only
        the two derived-source cases above — this is what "removes the unknown-schema
        ambiguous fallbacks" for a base table: `_derived_pii_output_columns` and
        `_column_is_pii`'s permissive ambiguous-column classification (see their
        docstrings) exist specifically for when this pass can't fully resolve a base
        table's real columns; supplying real `schema` is what lets it.

    Returns the qualified `exp.Select`, or `None` when qualify() either raises — a
    construct it can't resolve/support at all, including real ambiguity neither it nor
    this module has schema to settle — or still leaves a star unexpanded (its source's
    real schema is genuinely unknown — e.g. a bare `SELECT *` over a base table when
    `schema` is `None` or doesn't cover that table). Either case means the caller must
    fall back to `root` itself and the pre-existing per-column walk, which applies its
    own narrower fail-closed/fallback rules.
    """
    try:
        qualified = qualify(root.copy(), dialect=dialect, expand_stars=True, infer_schema=True, schema=schema)
    except Exception:  # sqlglot.errors.OptimizeError/SchemaError/UnsupportedError, etc.
        return None
    if not isinstance(qualified, exp.Select):
        return None
    if any(_is_star_projection(p) for p in qualified.expressions):
        return None
    return qualified


def _resolve_projections_pii(
    columns: list[str], projections: list[exp.Expression], scope: Scope, strict: bool
) -> set[str]:
    """Shared positional resolve loop for `_provenance_pii_columns`: pairs `columns[i]`
    with `projections[i]` and evaluates each via `_projection_is_pii` in `scope`, failing
    closed (treating as PII) whenever that raises `_ProvenanceUnresolved`. `strict` is
    passed straight through to `_column_is_pii` — see that function's docstring.
    """
    pii_output: set[str] = set()
    for column_name, projection in zip(columns, projections):
        try:
            is_pii = _projection_is_pii(projection, scope, 0, strict)
        except _ProvenanceUnresolved:
            is_pii = True
        if is_pii:
            pii_output.add(column_name)
    return pii_output


def _resolve_projections_pii_by_name(
    columns: list[str], projections: list[exp.Expression], scope: Scope, strict: bool
) -> set[str]:
    """Same per-projection evaluation as `_resolve_projections_pii`, but pairs each of
    `columns` with the qualified query's own output projection of the SAME NAME
    (`projection.alias_or_name`) rather than by position.

    Used only by the qualify-based pass (pass 1 in `_provenance_pii_columns`) when the
    ORIGINAL, pre-qualify top-level SELECT contained a `*`/`table.*` projection that
    `qualify(schema=..., expand_stars=True)` then expanded into multiple real
    projections: with a real `schema` supplied, sqlglot is free to expand each source's
    own columns in whatever internal order it chooses — empirically NOT always the
    source's declared `FROM`/`JOIN` position (e.g. a base table with known schema can be
    expanded before a derived source that's declared earlier in the query) — so
    `q_projections`' order cannot be assumed to line up positionally with `columns` (the
    real, engine-executed result set's own column order) the way an explicit, non-star
    projection list reliably does (see `_resolve_projections_pii`'s docstring for why
    THAT case still uses position). Matching by name instead sidesteps the ordering
    question entirely: every expanded star projection carries a real, deterministic
    column name — the source's own — which for a star expansion is exactly what the
    executing engine also names that result column.

    A `columns[i]` with no same-named projection among `projections` at all (should not
    normally happen once qualify has fully expanded every star in the query — checked
    anyway as a fail-safe, e.g. a duplicate column name collision across joined sources
    losing one name from the dict built here) fails closed (treated as PII), consistent
    with every other unresolved case in this module.
    """
    by_name: dict[str, exp.Expression] = {p.alias_or_name: p for p in projections}

    pii_output: set[str] = set()
    for column_name in columns:
        projection = by_name.get(column_name)
        if projection is None:
            pii_output.add(column_name)
            continue
        try:
            is_pii = _projection_is_pii(projection, scope, 0, strict)
        except _ProvenanceUnresolved:
            is_pii = True
        if is_pii:
            pii_output.add(column_name)
    return pii_output


def _get_arg(node: exp.Expression, *names: str) -> exp.Expression | None:
    """`node.args.get(name)` tried in order, returning the first non-`None` hit.

    sqlglot renames an `exp.Select`'s arg keys that collide with a Python keyword —
    `from` -> `"from_"`, `with` -> `"with_"` — but not ones that don't (`"joins"` stays
    `"joins"`). This small indirection is only defensive plumbing for that naming, kept
    in one place so a future sqlglot version reverting/changing the key spelling is a
    one-line fix rather than a silent no-op fallback-to-empty scattered across callers.
    """
    for name in names:
        value = node.args.get(name)
        if value is not None:
            return value
    return None


def _iter_derived_sources(node: exp.Select):
    """Yield every derived-table `exp.Select` reachable from `node`'s own `FROM`/`JOIN`
    clauses and CTEs, recursing into each yielded derived source's own `FROM`/`JOIN`/
    CTEs in turn so a chain of nested derived tables is walked all the way down.

    Used only by `_derived_pii_output_columns` (see its docstring) — this generator
    itself makes no PII judgement, it just enumerates the derived-source `SELECT`
    nodes to be classified.
    """
    with_clause = _get_arg(node, "with_", "with")
    if with_clause is not None:
        for cte in with_clause.expressions:
            inner = cte.this
            if isinstance(inner, exp.Select):
                yield inner
                yield from _iter_derived_sources(inner)

    sources: list[exp.Expression] = []
    from_clause = _get_arg(node, "from_", "from")
    if from_clause is not None:
        sources.append(from_clause.this)
    for join in _get_arg(node, "joins") or []:
        sources.append(join.this)

    for source in sources:
        if isinstance(source, exp.Subquery) and isinstance(source.this, exp.Select):
            yield source.this
            yield from _iter_derived_sources(source.this)


def _derived_pii_output_columns(root: exp.Select, strict: bool) -> set[str]:
    """Fallback for `_provenance_pii_columns` pass 2 when a top-level star can't be
    expanded (real base-table schema unknown) — this is the fix for the confirmed HIGH
    leak: a top-level `SELECT *` spanning BOTH a derived source (a subquery-in-`FROM`/
    `JOIN` or a CTE, which renamed a PII column to an alias with no PII signal of its
    own, e.g. `ssn_num AS c`) AND a base table of unknown schema (`orders`) can't be
    expanded by `qualify(expand_stars=True)` (pass 1 returns `None`), and the original
    tree still has an unexpanded star (pass 2's own expansion needs the same missing
    schema). Rather than failing open outright in that situation, this walks every
    derived source's own SELECT list — recursively, through any further-nested derived
    source — and classifies each of *its* output columns via the same lineage logic
    (`_projection_is_pii`) already used for a normal (non-star) top-level SELECT.

    Returns the set of output column *names* (aliases) that are PII by that lineage.
    `redact_rows` matches this set against the query's actual result columns by name —
    the same way layer 1 (`_is_pii_column`) already matches base-table column names —
    so a derived source's renamed PII column (`c`) is masked regardless of which other,
    unexpandable base-table columns (`orders.id`, `.total`, `.note`, ...) are also
    present in the same star's result set; those base-table columns are left alone
    here (never added to the returned set) and remain covered the same way they always
    were: layer 1 classifies each by its own real column name, and layer 2 scans each
    value against the PII regexes — a base-table wildcard expansion keeps the source's
    own column names, so neither layer needs this function's help for them.

    This can't determine whether some *other*, co-present source also happens to have
    an output column of the identical name that is not PII — over-redacting that rare
    same-name collision is accepted as the safer direction, in exchange for closing the
    leak this function exists for. It returns an empty set (contributing nothing) when
    the query has no derived source in its `FROM`/`JOIN`/CTEs at all — a plain
    `SELECT * FROM orders` still redacts nothing here, since there is nothing to walk;
    layers 1/2 remain the (already-correct) backstop for that case.

    `strict` is passed straight through to `_projection_is_pii`/`_column_is_pii` — see
    those functions' docstrings for what it changes.
    """
    pii_names: set[str] = set()
    for derived_select in _iter_derived_sources(root):
        inner_scope = build_scope(derived_select)
        if inner_scope is None:
            continue
        for projection in derived_select.selects:
            if _is_star_projection(projection):
                # A nested `SELECT *` this fallback would also need schema to expand —
                # skip it (can't name a specific output column to add), rather than
                # raising/failing the whole fallback closed; the outer star's other,
                # already-classifiable derived/base columns are still worth catching.
                continue
            name = projection.alias_or_name
            try:
                is_pii = _projection_is_pii(projection, inner_scope, 0, strict)
            except _ProvenanceUnresolved:
                is_pii = True
            if is_pii:
                pii_names.add(name)
    return pii_names


def _query_references_pii_source(root: exp.Expression) -> bool:
    """True if `root` — the parsed statement, at ANY depth (any scope, subquery, or
    CTE, not just the top-level SELECT list) — references at least one base column
    whose *name* is PII by `_is_pii_column`'s keyword classification.

    This is deliberately a coarse, whole-statement existence check, not a per-column
    lineage resolution: it does not matter here whether a given `exp.Column` leaf is
    itself ambiguous, qualified, feeds a top-level output column, or sits only in a
    `WHERE`/`JOIN ... ON`/`GROUP BY` clause — `root.find_all(exp.Column)` walks every
    `Column` node reachable from `root` regardless of clause or nesting depth, and a
    single PII-named hit anywhere is enough. That is the point: this function's only
    job is to decide whether the query touches PII *anywhere at all*, which is what
    gates the generalized fail-closed policy in `_provenance_pii_columns` below (see
    that function's and `_column_is_pii`'s docstrings for the `strict` parameter this
    feeds) — not to say which specific *output* column is fed by it.

    Returns `False` for a query that references no PII-named column anywhere (e.g.
    `SELECT *, total AS c FROM orders`, `SELECT id, admin FROM users WHERE id=1`) —
    exactly the case where the pre-existing, more permissive ambiguous-column fallback
    in `_column_is_pii` must be kept, so an ordinary PII-free (or PII-column-free)
    analytics query is never over-redacted just because it happens to contain an
    unresolvable JOIN shape.
    """
    return any(_is_pii_column(col.name) for col in root.find_all(exp.Column))


def _provenance_pii_columns(sql: str, columns: list[str], schema: SchemaCatalog | None = None) -> set[str]:
    """Best-effort column-provenance analysis: which of `columns` (by name) are fed, in
    the proposed `sql`'s SELECT, by at least one PII-classified *source* column —
    catching PII an output alias/transform hides from name- and value-based classification
    alone (`SUBSTR(ssn, 1, 3) AS s`, `GROUP_CONCAT(email) AS c`), including when that
    source is hidden behind a subquery-in-`FROM` or a CTE (see module docstring layer 3
    and `_column_is_pii`'s docstring for the nested-scope tracing and its fail-closed
    semantics).

    `schema` (optional, `None` by default — see `redact_rows`'s own docstring) is the
    live `SchemaCatalog` `redact_rows` was given, if any. Two things change when it's
    supplied, both applied BEFORE either pass below even starts:

      - **View inlining** (`_inline_views`): every `FROM`/`JOIN` reference anywhere in
        `sql`, at any depth, to a name in `schema.views` is replaced with that view's
        own parsed `SELECT` body as a derived subquery — recursing for a view built on
        another view — so a query that only ever names a view is, from this point on,
        traced exactly as if it had named the view's real underlying source directly.
        This is what closes the confirmed leak `redact_rows`'s docstring documents: a
        view was previously an opaque base-table reference this text-only analysis had
        no way to see through at all. `_query_references_pii_source` below runs on the
        ALREADY-inlined tree, so it also now sees a query that only names a PII-bearing
        view as touching PII — the `strict` fail-closed policy engages for it exactly
        as it would for a query naming the underlying table directly.
      - **Real schema for `qualify()`**: `schema.tables` (already exactly the shape
        `qualify(schema=...)` expects) is passed into pass 1 below, letting it fully
        resolve a `SELECT *`/ambiguous JOIN column over an actual base table
        deterministically — see `_qualify_root_for_provenance`'s docstring for what
        this removes from the schema-less fallback path.

    If inlining reports an unresolvable view reference anywhere (`_inline_views`'
    `unresolved` — an unparseable or cyclical view body), this function returns
    `set(columns)` immediately: every output column is treated as PII, without running
    either pass at all. See `_inline_views`'s docstring for why a query-wide, not a
    per-column, fail-closed is the honest answer here.

    **Generalized fail-closed policy (see module docstring's "generalized fail-closed"
    layer):** before either pass runs, `_query_references_pii_source(root)` decides
    `strict` once for the whole statement — `True` iff the query references a
    PII-named column *anywhere*, at any depth (not just among the top-level output
    columns). `strict` is threaded through every call below into `_column_is_pii`,
    which is the one place any ambiguous/unresolvable column reference gets decided:
    with `strict=False` (no PII source anywhere in the query) it keeps the pre-existing
    permissive fallback (name-classify the column directly); with `strict=True` (a PII
    source IS present somewhere) that same ambiguity instead raises
    `_ProvenanceUnresolved`, which every call site here already treats as fail-closed
    (redact). This closes the whole *class* of leaks the six rounds of one-shape-at-a-
    time fixes below were chasing — an unqualified/unresolvable output column in a
    query that touches PII is now redacted whenever it cannot be *positively* proven to
    trace to a non-PII base column, rather than only for the specific JOIN/star shapes
    enumerated so far — while a query that references no PII column anywhere keeps
    exactly the old, more permissive behavior, so ordinary PII-free analytics queries
    over an ambiguous JOIN are not over-redacted.

    Two resolution passes are tried, in order:

      1. **Qualify-based** (`_qualify_root_for_provenance`, HIGH #3 fix): the whole query
         is fully qualified and star-expanded via sqlglot's own optimizer first. When
         that fully succeeds (no exception, no star left unexpanded), its projections are
         used directly — every column reference is now an explicit `source.column`, so
         `_column_is_pii`'s ambiguous-unqualified-column fallback is never reached at all
         on this pass, and a `SELECT *`/`alias.*` over a derived table has already been
         expanded to the derived table's own real projections.
      2. **Legacy per-column walk**: used only when pass 1 doesn't fully succeed — a
         top-level star sqlglot couldn't expand (source schema genuinely unknown), or a
         construct `qualify()` raises on (including a truly ambiguous unqualified
         reference among base-table sources with no schema on either side, or the
         statement not parsing / not being a single top-level `SELECT` at all — a
         top-level `UNION` needs independent per-branch analysis neither pass attempts).
         When the original (unqualified) top level still has an unexpanded star, this
         pass no longer bails out unconditionally (pre-fix behavior): column-name
         classification on the actual result `columns` already covers a star expanding
         purely over a *base* table correctly on its own (a base-table wildcard
         expansion keeps the source's own column names), but a star that ALSO spans a
         *derived* source (subquery-in-`FROM`/`JOIN` or CTE) whose PII column was
         renamed to a non-signal-carrying alias needs `_derived_pii_output_columns` to
         recover it (the HIGH fix this docstring's "still fails open" list below no
         longer includes) — see that function's docstring.

    Both passes match `sql`'s top-level SELECT projections to `columns` **positionally**
    (projection i feeds `columns[i]`) rather than by alias name: this works uniformly
    whether or not a projection is aliased (an unaliased `SUBSTR(ssn,1,3)` has no
    `sqlglot` output name that reliably matches what the engine actually names the result
    column, but its *position* always does), and sidesteps needing a real column-name
    resolver at the outermost level. Nested scopes reached *while tracing* a top-level
    projection are matched by output alias name instead (`_column_is_pii`), which is safe
    and reliable at that level: a derived table's exposed column names are exactly what
    an outer `t.c` / unaliased-passthrough column reference refers to, unlike the
    outermost result set's own naming (which the executing engine controls, not
    `sqlglot`).

    Still fails open (returns an empty set — layers 1/2 in `redact_rows` remain the
    backstop) for the structural cases neither pass attempts to trace at the top level at
    all:
      - `sql` doesn't parse under any attempted dialect, or parses to more than one
        statement;
      - the (single) statement's top level isn't a plain `SELECT` (a top-level `UNION`
        would need independent per-branch analysis this layer doesn't attempt);
      - pass 1 fails AND the original (unqualified) top level still has a `*`/`table.*`
        projection, AND no derived source (subquery-in-`FROM`/`JOIN`/CTE) is present in
        the statement at all for `_derived_pii_output_columns` to walk — a bare
        `SELECT *` directly over one or more base tables of unknown schema. A wildcard
        expands to a number of actual result columns that has nothing to do with the
        number of projections, breaking the positional assumption outright, and by this
        point sqlglot itself couldn't expand it either; but this case needs no recovery
        here regardless, since column-name classification already covers it directly (a
        wildcard expansion keeps the source table's own column names). (A `SELECT *`
        *nested* inside a nested scope reached while tracing, by contrast, fails CLOSED
        — see `_column_is_pii` — since by that point a specific output column's
        provenance genuinely can't be resolved, unlike a top-level wildcard which layer
        1 already handles correctly on its own.) When a derived source IS present
        alongside the unexpandable star, this no longer fully fails open — see
        `_derived_pii_output_columns`;
      - the projection count doesn't match `len(columns)` on the pass being used (should
        not happen once the above are excluded, checked anyway as a fail-safe).
    """
    statements = None
    parse_dialect: str | None = None
    for dialect in _PROVENANCE_PARSE_DIALECTS:
        try:
            parsed = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
        except Exception:  # sqlglot's own ParseError/TokenizeError subclasses
            continue
        if parsed:
            statements = parsed
            parse_dialect = dialect
            break

    if not statements or len(statements) != 1:
        return set()

    root = statements[0]
    if not isinstance(root, exp.Select):
        return set()

    # Inline any known view reference (recursing for view-on-view) BEFORE anything else
    # below looks at `root` — see this function's docstring and `_inline_views`'s. A
    # `schema` with no views (or `schema=None` entirely) makes this a no-op that returns
    # `root` unchanged, so every existing schema-less caller/test is unaffected.
    root, unresolved_view = _inline_views(root, schema, parse_dialect)
    if unresolved_view:
        return set(columns)

    sqlglot_schema = schema.tables if schema and schema.tables else None

    # Decide the generalized fail-closed policy once for the whole statement — see the
    # docstring above and `_column_is_pii`'s docstring for exactly what this changes.
    # Runs on the already view-inlined tree, so a query that only names a PII-bearing
    # view (not the underlying table directly) is correctly seen as PII-touching too.
    strict = _query_references_pii_source(root)

    # Pass 1: qualify-based resolution (HIGH #3 fix) — see docstring above.
    qualified_root = _qualify_root_for_provenance(root, parse_dialect, sqlglot_schema)
    if qualified_root is not None:
        q_projections = qualified_root.expressions
        q_scope = build_scope(qualified_root)
        if q_scope is not None:
            if any(_is_star_projection(p) for p in root.expressions):
                # The original top-level SELECT had a star qualify() has now fully
                # expanded — match by name, not position; see
                # `_resolve_projections_pii_by_name`'s docstring for why position can't
                # be assumed here once real `schema` is involved.
                return _resolve_projections_pii_by_name(columns, q_projections, q_scope, strict)
            if len(q_projections) == len(columns):
                return _resolve_projections_pii(columns, q_projections, q_scope, strict)

    # Pass 2: legacy per-column walk over the original, unqualified tree.
    projections = root.expressions
    if any(_is_star_projection(p) for p in projections):
        # A top-level star pass 1 couldn't expand (real base-table schema unknown).
        # Column-name classification on the actual result `columns` already covers a
        # star expanding purely over a *base* table correctly on its own (layer 1 runs
        # on the real column names either way) — but when the star ALSO spans a
        # *derived* source whose PII column was renamed to an alias with no PII signal
        # in its own name (`ssn_num AS c`), that alias is invisible to layer 1 too, and
        # failing open here would leak it outright (the confirmed HIGH leak this branch
        # fixes). Rather than failing open unconditionally, recover what can still be
        # proven: any derived source's own output columns that are themselves PII by
        # lineage — see `_derived_pii_output_columns` for the exact scope/limits of
        # this fallback (returns an empty set, i.e. no change from prior behavior, when
        # no derived source is present at all).
        pii_names = _derived_pii_output_columns(root, strict)

        # A FOURTH confirmed HIGH leak shape, at a different boundary than the one
        # above: a top-level star with a *non-star* projection sitting directly beside
        # it in the same SELECT list — `SELECT *, (SELECT ssn_num FROM users WHERE
        # id=1) AS c FROM orders ...` — is not covered by `_derived_pii_output_columns`
        # at all, since that helper only walks FROM/JOIN/CTE *derived sources*; a
        # scalar subquery (or function call, or arithmetic expression) used directly as
        # a projection is none of those. Such a projection dodges layer 1 (its alias
        # `c` carries no PII keyword) and layer 2 (a bare numeric SSN has no dashes and
        # no keyword prefix in the same text for the contextual pattern to match), so
        # without this it passes straight through unredacted. Close the whole
        # projection *class* — not just the scalar-subquery shape — by resolving every
        # non-star top-level projection with the same per-projection lineage used
        # everywhere else in this module, failing closed on `_ProvenanceUnresolved`
        # exactly like `_resolve_projections_pii` does, and union its PII aliases into
        # the result. A non-PII non-star projection beside a star (`total AS c`) must
        # still resolve to non-PII and stay unredacted — this is provenance tracing,
        # not a blanket redact-everything-beside-a-star rule.
        non_star_projections = [p for p in projections if not _is_star_projection(p)]
        if non_star_projections:
            root_scope = build_scope(root)
            for projection in non_star_projections:
                try:
                    is_pii = root_scope is None or _projection_is_pii(projection, root_scope, 0, strict)
                except _ProvenanceUnresolved:
                    is_pii = True
                if is_pii:
                    pii_names.add(projection.alias_or_name)

        return pii_names

    if len(projections) != len(columns):
        return set()

    root_scope = build_scope(root)
    if root_scope is None:
        return set()

    return _resolve_projections_pii(columns, projections, root_scope, strict)


def _looks_like_json(value: str) -> bool:
    """True if `value` parses as a JSON object or array (not just any valid JSON scalar —
    a bare number/string/bool isn't the "blob carrying structured PII" case this guards
    against, and treating every quoted string as JSON would over-redact ordinary text).
    """
    stripped = value.strip()
    if not stripped or stripped[0] not in "{[":
        return False
    try:
        parsed = json.loads(stripped)
    except (ValueError, TypeError):
        return False
    return isinstance(parsed, (dict, list))


def _scan_value_patterns(value: str) -> tuple[str, bool]:
    """Apply every value-level PII pattern to `value`. Returns (masked_value, was_masked)."""
    masked = value
    matched = False
    for pattern, replacement in _VALUE_PATTERNS:
        if pattern.search(masked):
            masked = pattern.sub(replacement, masked)
            matched = True
    return masked, matched


def redact_text(text: str) -> str:
    """Scrub PII value-patterns from a plain string (e.g. an engine error message or a
    policy `reason`) using the same patterns `redact_rows` applies to non-PII-column
    result values. Used by the gateway so error/reason text returned to a caller can't
    leak PII that happened to be echoed back in an engine error message.
    """
    masked, _ = _scan_value_patterns(text)
    return masked


def redact_rows(
    rows: list[dict[str, Any]],
    columns: list[str],
    sql: str | None = None,
    schema: SchemaCatalog | None = None,
) -> tuple[list[dict[str, Any]], RedactionReport]:
    """Redact PII from query result rows.

    Args:
        rows: Result rows as plain dicts (column name -> value). Not mutated in place —
            a new list of new dicts is returned.
        columns: Column names, in result order. Used to classify PII columns by name;
            iterating this (rather than each row's own keys) also means a column that is
            entirely NULL/empty in every row is still classified and reported correctly.
        sql: The SELECT statement that produced `rows`/`columns`, optional. When given,
            enables provenance (lineage) redaction — see module docstring layer 3 and
            `_provenance_pii_columns`. Omitting it (the default) skips that layer only;
            name- and value-pattern redaction (layers 1/2) and JSON/BLOB defaulting
            (layer 4) still apply either way.
        schema: A live `SchemaCatalog` (tables/columns + view definitions —
            `backend.core.schema_catalog`), optional and `None` by default. Ignored
            unless `sql` is also given. When supplied, `_provenance_pii_columns` (a)
            inlines any known view referenced in `sql` with its own real `SELECT` body
            before tracing lineage — closing the leak class where a PII column is
            renamed through a view with no PII-looking name of its own — and (b) passes
            `schema.tables` into `sqlglot`'s `qualify(schema=...)` so a `SELECT *`/
            ambiguous JOIN column over an actual base table resolves deterministically
            instead of via the schema-less fallback heuristics. Omitting it (the
            default `None`) is exactly today's schema-less behavior — the gateway's own
            schema introspection is itself best-effort/optional (see `gateway.py`), so
            this parameter must degrade gracefully to `None`, never require a schema.

    Returns:
        `(redacted_rows, report)`. `report["columns_redacted"]` lists every column (by
        name) that had at least one value masked, by any layer, sorted for determinism.
        `report["values_masked"]` is the total count of individual cell values masked
        across all rows.
    """
    pii_columns = {c for c in columns if _is_pii_column(c)}
    if sql:
        pii_columns |= _provenance_pii_columns(sql, columns, schema=schema)
    columns_with_masks: set[str] = set()
    values_masked = 0

    redacted_rows: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        for column in columns:
            if column not in new_row:
                continue
            value = new_row[column]

            if column in pii_columns:
                if value not in (None, ""):
                    new_row[column] = _REDACTED_PLACEHOLDER
                    values_masked += 1
                    columns_with_masks.add(column)
                continue

            if isinstance(value, bytes):
                # BLOB: opaque to every text-based layer above; default to redacted (see
                # module docstring layer 4).
                if value:
                    new_row[column] = _REDACTED_PLACEHOLDER
                    values_masked += 1
                    columns_with_masks.add(column)
                continue

            if isinstance(value, str) and value and _looks_like_json(value):
                # JSON object/array: can carry arbitrary nested PII no name/value/
                # provenance layer above can see into (its keys/values are invisible to
                # all three) — default to redacted rather than passing the raw blob through.
                new_row[column] = _REDACTED_PLACEHOLDER
                values_masked += 1
                columns_with_masks.add(column)
                continue

            # HIGH #2 fix: the value-pattern scan must not be gated on `isinstance(value,
            # str)` alone — an SSN/card/phone value stored in a numeric column (e.g.
            # `ssn_num INTEGER`) is just as real a leak as one stored as text, and was
            # previously skipped entirely here purely because of its Python type. `str`
            # values are scanned as-is; `int`/`float` values are `str()`-coerced first.
            # `bool` is deliberately excluded even though it's an `int` subclass in
            # Python — "True"/"False" can never match any of these patterns, so there is
            # nothing to gain by manufacturing a string for it.
            if isinstance(value, str):
                scan_source: str | None = value
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                scan_source = str(value)
            else:
                scan_source = None

            if scan_source:
                masked_value, matched = _scan_value_patterns(scan_source)
                if matched:
                    new_row[column] = masked_value
                    values_masked += 1
                    columns_with_masks.add(column)

        redacted_rows.append(new_row)

    report: RedactionReport = {
        "columns_redacted": sorted(pii_columns | columns_with_masks),
        "values_masked": values_masked,
    }
    return redacted_rows, report
