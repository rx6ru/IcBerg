"""Policy YAML — a per-deployment layer that further-restricts (and can NEVER loosen)
`sql_governance.GovernanceGate`'s hardcoded, fail-safe decisions.

The gate's own rules (block DDL, block writes without `WHERE`, block injection/RCE/DoS
patterns, ...) are the safety floor: they are not configurable and this module never
touches a `block` decision at all. `Policy` exists for the layer *above* that floor — the
per-deployment scoping an operator wants without editing code: "this agent may only ever
touch these tables", "cap every result at 50 rows even though the engine allows 1000",
"every write, no exceptions, needs a human, even one the gate might one day allow
outright". `apply_policy` is the single function that applies it, called from
`gateway.Gateway.handle` immediately after `GovernanceGate.evaluate` and before execution,
so a policy-driven `block`/`hold`/row-cap is honored by every integration surface that
routes through `Gateway.handle` (`icberg`, `backend.mcp_server`,
`backend.integrations.langgraph_tool`, and the REST API once it is wired to accept one) —
never something an individual surface could opt out of.

`apply_policy` is a monotonic restriction, never a decision from scratch:
  - `decision.action == "block"` is returned unchanged, always — a policy has no code
    path back to `allow`/`hold` for something the gate already refused outright.
  - `denied_tables`/`allowed_tables` violations escalate `allow`/`hold` to `block`.
  - `require_approval_for_writes` escalates a write's `allow` to `hold` (defense-in-depth:
    every write already ends up `hold`/`block` from the base gate today, but this keeps
    that guarantee explicit and policy-enforced rather than an accident of the current
    gate's own rules).
  - `max_rows` only ever narrows the row count already capped by the executor's own
    engine-level `MAX_ROWS` (`executor.py`) — `effective_max_rows` is never used to
    request MORE rows than the engine would already return, only to truncate further.

`pii_columns` (extra PII column-name keywords, beyond `redaction.py`'s built-in list) is
applied as an ADDITIONAL masking pass on top of `redact_rows`'s own layered redaction —
see `apply_extra_pii_redaction` — never a replacement for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml

from backend.core.sql_governance import PolicyDecision

logger = structlog.get_logger(__name__)

_REDACTED_PLACEHOLDER = "[REDACTED]"

# Decision actions, ranked from least to most restrictive. `apply_policy` may only move a
# decision's action to a HIGHER rank than it already has (never lower) — this is the
# concrete meaning of "a policy can only further-restrict, never loosen".
_ACTION_RANK: dict[str, int] = {"allow": 0, "hold": 1, "block": 2}


@dataclass(frozen=True)
class Policy:
    """A loaded, immutable policy. Construct via `load_policy`, not directly, in normal
    use — the dataclass itself does no YAML/dict parsing.

    Attributes:
        allowed_tables: If set, only these tables (bare name, case-insensitive; a
            schema-qualified `main.users` is compared by its last component, `users`)
            may be referenced anywhere in a proposed statement — any other table
            escalates the decision to `block`. `None` (the default) means no allow-list
            restriction at all.
        denied_tables: Tables (same bare-name, case-insensitive matching) that are
            always blocked, regardless of what the base gate decided.
        max_rows: If set, the returned row count is truncated to at most this many rows
            on an `allow` decision — but only ever DOWN from whatever the executor's own
            engine-level cap already produced, never up (see module docstring).
        require_approval_for_writes: If True, any write the base gate would have allowed
            outright is instead held for human approval.
        pii_columns: Extra column-name keywords (case-insensitive substring match, the
            same convention `redaction.py`'s own `_PII_COLUMN_KEYWORDS` uses) whose
            values are masked in addition to `redact_rows`'s own PII detection.
    """

    allowed_tables: frozenset[str] | None = None
    denied_tables: frozenset[str] = field(default_factory=frozenset)
    max_rows: int | None = None
    require_approval_for_writes: bool = False
    pii_columns: frozenset[str] = field(default_factory=frozenset)


def _leaf_name(table: str) -> str:
    """Normalize a (possibly schema-qualified) table name for policy matching: last
    `.`-qualified component, quote characters stripped, lower-cased — the same
    normalization convention `sql_governance.py`'s own identifier handling uses.
    """
    last = table.rsplit(".", 1)[-1]
    return last.strip().strip("\"'`[]").lower()


def _policy_from_mapping(data: dict[str, Any]) -> Policy:
    allowed = data.get("allowed_tables")
    return Policy(
        allowed_tables=frozenset(allowed) if allowed is not None else None,
        denied_tables=frozenset(data.get("denied_tables") or ()),
        max_rows=data.get("max_rows"),
        require_approval_for_writes=bool(data.get("require_approval_for_writes", False)),
        pii_columns=frozenset(data.get("pii_columns") or ()),
    )


def load_policy(source: str | Path | dict[str, Any] | Policy | None) -> Policy | None:
    """Load a `Policy` from a YAML file path, an already-parsed dict, an existing
    `Policy` (returned unchanged), or `None` (returns `None` — "no policy configured",
    which `apply_policy`/`Gateway.handle` treat as a pure pass-through of the base gate's
    decision).

    Expected YAML shape:

        allowed_tables: [orders, invoices]
        denied_tables: [secrets]
        max_rows: 50
        require_approval_for_writes: true
        pii_columns: [internal_notes]

    Every key is optional; an empty/absent key keeps that restriction off. Raises
    `ValueError` if a YAML file's top-level document isn't a mapping — fails loud rather
    than silently constructing a no-op policy from malformed config.
    """
    if source is None or isinstance(source, Policy):
        return source
    if isinstance(source, dict):
        return _policy_from_mapping(source)

    path = Path(source)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"policy YAML at {path} must define a mapping at the top level, got {type(data).__name__}")
    return _policy_from_mapping(data)


def apply_policy(decision: PolicyDecision, policy: Policy) -> tuple[PolicyDecision, int | None]:
    """Apply `policy` on top of `decision` (the base `GovernanceGate`'s output). Returns
    `(possibly-more-restrictive decision, effective_max_rows)` — `effective_max_rows` is
    `None` unless `policy.max_rows` applies to an `allow` decision, in which case the
    caller (`gateway.Gateway.handle`) truncates the redacted result rows to it.

    Never returns an action ranked LOWER (less restrictive) than `decision.action` — see
    `_ACTION_RANK` and the module docstring. A `block` decision is returned byte-for-byte
    unchanged: this function has no code path that can touch it.
    """
    if decision.action == "block":
        return decision, None

    matched_rules = list(decision.matched_rules)
    action = decision.action
    reason = decision.reason
    changed = False

    tables = {_leaf_name(t) for t in decision.tables}

    # Each restriction below is evaluated independently (not an if/elif chain) — a
    # policy can combine, say, an allow-list AND require_approval_for_writes, and both
    # must be checked regardless of whether the other one fired. Once `action` has
    # already been escalated to "block" by an earlier check, later checks that only
    # apply to "allow" naturally no-op (their own guard conditions already require
    # `action == "allow"`), so `block` can only ever be reached, never left.
    denied = {_leaf_name(t) for t in policy.denied_tables}
    denied_hit = tables & denied
    if denied_hit:
        action = "block"
        reason = f"Policy denies access to table(s): {', '.join(sorted(denied_hit))}."
        matched_rules.append("policy_denied_table")
        changed = True

    if action != "block" and policy.allowed_tables is not None:
        allowed = {_leaf_name(t) for t in policy.allowed_tables}
        not_allowed = tables - allowed
        if not_allowed:
            action = "block"
            reason = f"Policy allow-list does not include table(s): {', '.join(sorted(not_allowed))}."
            matched_rules.append("policy_table_not_allowed")
            changed = True

    if action == "allow" and policy.require_approval_for_writes and decision.classification == "write":
        # Defense-in-depth: the base gate never actually returns "allow" for a write
        # today (see sql_governance.py), so this branch is currently unreachable in
        # practice — kept so the guarantee is explicit and policy-enforced rather than
        # an accident of the current gate's own rules, per the contract this module's
        # docstring states.
        action = "hold"
        reason = "Policy requires human approval for all writes."
        matched_rules.append("policy_write_requires_approval")
        changed = True

    effective_max_rows: int | None = None
    if policy.max_rows is not None and action == "allow":
        effective_max_rows = policy.max_rows
        matched_rules.append(f"policy_max_rows={policy.max_rows}")
        changed = True

    if not changed:
        return decision, effective_max_rows

    assert _ACTION_RANK[action] >= _ACTION_RANK[decision.action], (
        "apply_policy must never lower a decision's action rank"
    )
    return (
        PolicyDecision(
            action=action,
            classification=decision.classification,
            reason=reason,
            matched_rules=matched_rules,
            tables=decision.tables,
        ),
        effective_max_rows,
    )


def apply_extra_pii_redaction(
    rows: list[dict[str, Any]], columns: list[str], pii_columns: frozenset[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Mask every value in any of `columns` whose name matches one of `pii_columns`
    (case-insensitive substring match, mirroring `redaction.py`'s own
    `_PII_COLUMN_KEYWORDS` convention) that `redact_rows` didn't already redact by its
    own, broader logic. Additive only: never un-redacts a value `redact_rows` already
    masked, and a `pii_columns` set with no matching column is a complete no-op.

    Returns `(new_rows, matched_columns)` — `rows` is not mutated in place. Called from
    `gateway.Gateway.handle` after `redact_rows`, only when `policy.pii_columns` is
    non-empty.
    """
    if not pii_columns:
        return rows, []
    keywords = {k.lower() for k in pii_columns}
    matched_columns = [c for c in columns if any(k in c.lower() for k in keywords)]
    if not matched_columns:
        return rows, []

    new_rows: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        for col in matched_columns:
            if col in new_row and new_row[col] not in (None, _REDACTED_PLACEHOLDER):
                new_row[col] = _REDACTED_PLACEHOLDER
        new_rows.append(new_row)
    return new_rows, matched_columns
