"""Live-schema introspection for `redaction.py`'s provenance layer.

Closes the last allow-path leak class documented in `redaction.py`'s module docstring:
`redact_rows`'s provenance analysis previously only ever parsed the *query text*, so a
`VIEW` was an opaque base-table reference to it — `CREATE VIEW vnum AS SELECT id AS uid,
ssn_num AS token FROM users; SELECT token FROM vnum WHERE uid=1` traced `token` nowhere
(no PII `exp.Column` appears anywhere in the *query*, only inside the view's own,
unparsed definition), so a bare-numeric PII column renamed through a view slipped past
every layer — name classification (`token` carries no PII keyword), value-pattern
scanning (a bare 9-digit int has no dashes/keyword-prefix for the contextual pattern to
match), and provenance (nothing to trace into). This module supplies what closes that
gap: a live map of the database's actual tables/columns and view definitions, so
`redaction.py` can (a) inline a view's own `SELECT` body in place of the bare reference
to it before tracing lineage, and (b) fully qualify/expand `SELECT *` and ambiguous JOIN
columns against the database's REAL schema via `sqlglot.optimizer.qualify.qualify
(schema=...)`, instead of the schema-less heuristic fallbacks (`_derived_pii_output_columns`,
the permissive ambiguous-column classification) that `redaction.py` falls back to when no
real schema is available.

Deliberately optional/lazy at every layer: `Gateway.handle` asks the executor for a
schema catalog on a best-effort basis, `redact_rows`'s `schema` parameter defaults to
`None`, and every function in this module fails by returning `None`/an empty catalog
rather than raising — a missing/locked database file, a view whose stored SQL this
module can't extract, or any other introspection failure all just mean "fall back to
today's schema-less redaction behavior", never an exception propagating into the
gateway's own "never raises" contract.

Only SQLite is implemented/tested here (`introspect_sqlite_schema`, wired into
`executor.ReadOnlyExecutor.get_schema_catalog`). The `SchemaCatalog` shape it returns is
deliberately backend-agnostic — `{table: {column: sql_type}}` plus `{view: select_sql}`
— so a future Postgres connector can populate the same dataclass from
`information_schema.tables`/`.columns` and `information_schema.views`/`pg_views`
(`executor.PostgresReadOnlyExecutor.get_schema_catalog` is stubbed to return `None` today
— see its docstring) with no change required in `redaction.py` at all.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import sqlglot
import structlog
from sqlglot import exp

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SchemaCatalog:
    """A live database's schema, in the shape `redaction.py`'s provenance layer needs.

    Attributes:
        tables: `{table_name: {column_name: sql_type}}` — exactly the shape
            `sqlglot.optimizer.qualify.qualify`'s own `schema=` argument expects, so it
            can be passed straight through with no translation.
        views: `{view_name: view_select_sql}` — each view's defining `SELECT` statement
            text (the body only, not the `CREATE VIEW ... AS` wrapper), keyed by the
            view's own name. `redaction.py` uses this to inline a view reference with
            its parsed body before tracing lineage, and recurses for a view built on
            another view.
        unresolved_views: names of views the database itself reports (via
            `sqlite_master`) whose defining SQL this module could not extract a
            single-`SELECT` body from (a parse failure, or a `UNION`/non-`SELECT`
            shape) — present so `redaction.py` can distinguish "this name is a view
            with a known, traceable body" (in `views`), "this name is a view this
            module could not resolve" (here — `redaction.py` fails the query closed
            rather than treating it as an ordinary unknown base table), and "this name
            isn't a view at all" (absent from both).

    All three collections are keyed by the identifier's name exactly as the database
    reports it (SQLite: as declared in the original `CREATE TABLE`/`CREATE VIEW`) —
    callers that need case-insensitive lookup (SQL identifiers are conventionally
    case-insensitive) are responsible for `.lower()`-ing on both sides, the same
    convention already used elsewhere in this codebase (e.g.
    `redaction._column_is_pii`'s alias resolution).
    """

    tables: dict[str, dict[str, str]] = field(default_factory=dict)
    views: dict[str, str] = field(default_factory=dict)
    unresolved_views: frozenset[str] = field(default_factory=frozenset)

    def __bool__(self) -> bool:
        return bool(self.tables or self.views or self.unresolved_views)


def _extract_view_select(create_view_sql: str, dialect: str | None = "sqlite") -> str | None:
    """Extract the `SELECT ...` body from a `CREATE VIEW name AS SELECT ...` statement
    (as stored verbatim in `sqlite_master.sql`), via `sqlglot` rather than a text/regex
    split on `AS` — a view's own column list or expressions can legitimately contain the
    word `AS` before the real body starts, which a naive text search could split on
    incorrectly.

    Returns `None` (never raises) if the statement doesn't parse, isn't a `CREATE VIEW`,
    or its body isn't a single `SELECT` (e.g. a `UNION`-shaped view) — `redaction.py`'s
    view-inlining fails closed for a source it can't resolve this way; see its docstring.
    """
    try:
        parsed = sqlglot.parse_one(create_view_sql, read=dialect)
    except Exception:  # sqlglot's own ParseError/TokenizeError subclasses
        return None
    if not isinstance(parsed, exp.Create):
        return None
    body = parsed.args.get("expression")
    if not isinstance(body, exp.Select):
        return None
    try:
        return body.sql(dialect=dialect)
    except Exception:
        return None


def introspect_sqlite_schema(db_path: str) -> SchemaCatalog | None:
    """Best-effort schema introspection for a SQLite database file.

    Opens its own short-lived, read-only (`mode=ro`) connection to `db_path` — never the
    caller's own connection/transaction, and never capable of writing — reads every
    table's columns via `PRAGMA table_info`, and every view's defining `SELECT` (via
    `sqlite_master.sql`, extracted by `_extract_view_select`). SQLite system tables
    (`sqlite_%`) are skipped, matching what `redaction.py`'s lineage tracing would ever
    plausibly need to resolve a user query against.

    Returns `None` on any failure at all (missing file, locked database, unexpected
    schema shape) — this is optional, lazy plumbing; every caller (`ReadOnlyExecutor
    .get_schema_catalog`, and `redact_rows` beyond it) is built to fall back to
    schema-less behavior rather than propagate an exception from here.
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        logger.warning("schema_catalog.connect_failed", db_path=db_path, error=str(exc))
        return None

    try:
        tables: dict[str, dict[str, str]] = {}
        views: dict[str, str] = {}
        unresolved_views: set[str] = set()

        catalog_rows = conn.execute(
            "SELECT name, type, sql FROM sqlite_master "
            "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\'"
        ).fetchall()

        for name, kind, sql in catalog_rows:
            if kind == "table":
                columns: dict[str, str] = {}
                for row in conn.execute(f'PRAGMA table_info("{name}")'):
                    # row: (cid, name, type, notnull, dflt_value, pk)
                    col_name = row[1]
                    col_type = row[2] or "TEXT"
                    columns[col_name] = col_type
                if columns:
                    tables[name] = columns
            elif kind == "view":
                select_sql = _extract_view_select(sql) if sql else None
                if select_sql:
                    views[name] = select_sql
                else:
                    # A view SQLite itself knows about, but whose body this module
                    # could not extract — recorded (not silently dropped) so
                    # `redaction.py` can fail closed for anything referencing it
                    # instead of mistaking it for an ordinary unknown base table.
                    logger.warning("schema_catalog.view_body_unresolved", view=name)
                    unresolved_views.add(name)

        return SchemaCatalog(
            tables=tables, views=views, unresolved_views=frozenset(unresolved_views)
        )
    except sqlite3.Error as exc:
        logger.warning("schema_catalog.introspection_failed", db_path=db_path, error=str(exc))
        return None
    finally:
        conn.close()
