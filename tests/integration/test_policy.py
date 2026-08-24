"""P3.2 (`.devdocs/PHASE3_GATES.md`): a loaded `Policy` can only further-restrict
`GovernanceGate`'s hardcoded decisions, never loosen them. Denying a table (or capping
`max_rows`) blocks/caps a matching query; a deliberately permissive-looking policy (an
allow-list that includes the exact table a DROP targets) cannot resurrect a decision the
base gate already blocked outright.
"""

from __future__ import annotations

import sqlite3

import pytest
import yaml

from backend.core.policy import Policy, apply_policy, load_policy
from backend.core.sql_governance import GovernanceGate
from icberg import govern


class TestPolicyRestrictsTables:
    def test_policy_restricts_denied_table_blocks_matching_select(self, db_path: str):
        policy = Policy(denied_tables=frozenset({"users"}))
        result = govern("SELECT * FROM users WHERE id=1 LIMIT 5", actor="agent-1", dsn=db_path, policy=policy)
        assert result["action"] == "block"
        assert "policy_denied_table" in result["matched_rules"]
        assert result["rows"] is None

    def test_policy_restricts_allowed_tables_blocks_table_not_listed(self, db_path: str):
        policy = Policy(allowed_tables=frozenset({"orders"}))
        result = govern("SELECT * FROM users WHERE id=1 LIMIT 5", actor="agent-1", dsn=db_path, policy=policy)
        assert result["action"] == "block"
        assert "policy_table_not_allowed" in result["matched_rules"]

    def test_policy_allowed_tables_permits_a_listed_table(self, db_path: str):
        policy = Policy(allowed_tables=frozenset({"users"}))
        result = govern("SELECT id, name FROM users WHERE id=1 LIMIT 5", actor="agent-1", dsn=db_path, policy=policy)
        assert result["action"] == "allow"


class TestPolicyRestrictsRowCount:
    def test_policy_restricts_max_rows_caps_below_engine_default(self, db_path: str):
        policy = Policy(max_rows=1)
        result = govern("SELECT id FROM users WHERE id>0 LIMIT 5", actor="agent-1", dsn=db_path, policy=policy)
        assert result["action"] == "allow"
        assert len(result["rows"]) == 1
        assert any(rule.startswith("policy_max_rows=") for rule in result["matched_rules"])

    def test_policy_max_rows_never_raises_effective_cap_above_engine_result(self, db_path: str):
        # A policy cap looser than what the query actually returns is a pure no-op --
        # it must never cause MORE rows to come back than the ungoverned query would.
        policy = Policy(max_rows=1000)
        result = govern("SELECT id FROM users WHERE id>0 LIMIT 5", actor="agent-1", dsn=db_path, policy=policy)
        assert result["action"] == "allow"
        assert len(result["rows"]) == 2  # exactly what the ungoverned query itself returns


class TestPolicyCannotLoosenSafetyFloor:
    def test_policy_cannot_unblock_drop_table_even_with_permissive_allow_list(self, db_path: str):
        policy = Policy(allowed_tables=frozenset({"users"}))
        result = govern("DROP TABLE users", actor="agent-1", dsn=db_path, policy=policy)
        assert result["action"] == "block"
        assert "ddl_blocked" in result["matched_rules"]

    def test_policy_cannot_unblock_injection_pattern(self, db_path: str):
        policy = Policy(allowed_tables=frozenset({"users"}), max_rows=1000)
        result = govern(
            "SELECT * FROM users WHERE id=1 OR 1=1 LIMIT 5", actor="agent-1", dsn=db_path, policy=policy
        )
        assert result["action"] == "block"
        assert "tautology_suspected" in result["matched_rules"]

    def test_apply_policy_never_lowers_an_already_blocked_decision(self):
        gate = GovernanceGate()
        decision = gate.evaluate("DROP TABLE users")
        assert decision.action == "block"

        policy = Policy(allowed_tables=frozenset({"users"}), max_rows=1000)
        restricted, effective_max_rows = apply_policy(decision, policy)

        assert restricted.action == "block"
        assert restricted is decision  # apply_policy never even inspects a block decision
        assert effective_max_rows is None


class TestPolicyCannotLoosen:
    """P3.9 (`.devdocs/PHASE3_GATES.md`): `Policy.apply` (`apply_policy`) runs only on an
    `allow`/`hold` decision and may only tighten it -- never loosen. A maximally-
    permissive/malformed policy (`allowed_tables={'*'}`, `require_approval_for_writes=
    False`, `pii_columns=[]`) must not unblock a `DROP`, must not promote a
    write-with-WHERE `hold` to `allow`, and must not disable base PII redaction.
    """

    # Exactly the malformed/permissive policy P3.9 names: a literal `'*'` is NOT a glob
    # anywhere in `apply_policy` (`_leaf_name` does plain string comparison, no wildcard
    # expansion) -- so this allow-list, taken at face value, denies every real table
    # name including `users`. That is itself proof the policy can only ever restrict:
    # even a config author's typo/misunderstanding ("I meant this to allow everything")
    # cannot cause a *more* permissive outcome than having no policy at all.
    PERMISSIVE_POLICY = Policy(
        allowed_tables=frozenset({"*"}),
        require_approval_for_writes=False,
        pii_columns=frozenset(),
    )

    def test_policy_cannot_loosen_drop_table_stays_blocked(self, db_path: str):
        result = govern("DROP TABLE users", actor="agent-1", dsn=db_path, policy=self.PERMISSIVE_POLICY)
        assert result["action"] == "block"
        assert "ddl_blocked" in result["matched_rules"]
        assert result["rows"] is None

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        conn.close()
        assert count == 2  # DROP never reached the engine

    def test_policy_cannot_loosen_write_with_where_hold_never_becomes_allow(self, db_path: str):
        result = govern(
            "UPDATE users SET admin=1 WHERE id=1", actor="agent-1", dsn=db_path, policy=self.PERMISSIVE_POLICY
        )
        # A base `hold` may only ever be escalated (to `block`, e.g. by the same
        # allow-list mismatch above) or left as `hold` -- `require_approval_for_writes=
        # False` and every other permissive-looking field on this policy have no code
        # path back down to `allow`.
        assert result["action"] in ("hold", "block")
        assert result["action"] != "allow"
        assert result["rows"] is None

        conn = sqlite3.connect(db_path)
        admin = conn.execute("SELECT admin FROM users WHERE id=1").fetchone()[0]
        conn.close()
        assert admin == 0  # never auto-executed

    def test_policy_cannot_loosen_apply_policy_never_downgrades_hold_to_allow(self):
        gate = GovernanceGate()
        decision = gate.evaluate("UPDATE users SET admin=1 WHERE id=1")
        assert decision.action == "hold"

        restricted, effective_max_rows = apply_policy(decision, self.PERMISSIVE_POLICY)
        assert restricted.action != "allow"
        assert effective_max_rows is None

    def test_policy_cannot_loosen_base_pii_redaction_stays_active(self, db_path: str):
        # The table-matching half of `PERMISSIVE_POLICY` (`allowed_tables={'*'}`) would
        # itself block any query against `users` (see the class docstring), which would
        # make it impossible to observe row-level redaction at all here -- so this one
        # assertion isolates the field P3.9 actually cares about for redaction
        # (`pii_columns=[]`, alongside the same permissive `require_approval_for_writes
        # =False`) from that separate, already-covered table-matching behavior. This
        # policy is equally "maximally permissive" for what redaction depends on: an
        # empty `pii_columns` is a pure no-op ADDITION on top of `redact_rows`'s own
        # unconditional base redaction (see `policy.py`'s `apply_extra_pii_redaction`
        # and `gateway.py`'s `Gateway.handle`), never a switch that can turn it off.
        redaction_policy = Policy(require_approval_for_writes=False, pii_columns=frozenset())
        result = govern(
            "SELECT * FROM users WHERE id=1 LIMIT 5", actor="agent-1", dsn=db_path, policy=redaction_policy
        )
        assert result["action"] == "allow"
        row = result["rows"][0]
        assert row["id"] == 1
        assert row["name"] == "Alice Smith"
        assert row["email"] == "[REDACTED]"
        assert row["ssn"] == "[REDACTED]"
        assert row["ssn_num"] == "[REDACTED]"


class TestPolicyYamlLoading:
    def test_policy_denied_table_loaded_from_yaml_file_blocks(self, tmp_path, db_path: str):
        policy_path = tmp_path / "policy.yaml"
        policy_path.write_text(yaml.safe_dump({"denied_tables": ["users"]}))

        policy = load_policy(str(policy_path))
        assert isinstance(policy, Policy)

        result = govern("SELECT * FROM users WHERE id=1 LIMIT 5", actor="agent-1", dsn=db_path, policy=policy)
        assert result["action"] == "block"

    def test_load_policy_none_is_a_pass_through(self):
        assert load_policy(None) is None

    def test_load_policy_rejects_non_mapping_yaml_document(self, tmp_path):
        policy_path = tmp_path / "bad.yaml"
        policy_path.write_text("- just\n- a\n- list\n")
        with pytest.raises(ValueError):
            load_policy(str(policy_path))
