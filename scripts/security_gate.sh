#!/usr/bin/env bash
#
# security_gate.sh — IcBerg's security regression gate.
#
# Runs the security-critical test suites (red-team SQL governance, PII redaction,
# approval-workflow, and API-level tests) followed by the full test suite, and fails
# (non-zero exit) if ANY of it fails. This is the gate CI runs before a merge is
# allowed — a red-team/PII/approval regression must block, not just warn.
#
# Usage:
#   chmod +x scripts/security_gate.sh
#   ./scripts/security_gate.sh
#
# Run from the repository root. Prefers the project's uv-managed `.venv` if present,
# falling back to whatever `python`/`pytest` is on PATH otherwise (e.g. inside CI after
# `pip install -e ".[dev]"` into the runner's own environment).

set -euo pipefail

if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python"
fi

banner() {
  echo ""
  echo "======================================================================"
  echo "$1"
  echo "======================================================================"
}

banner "IcBerg security gate — using: ${PYTHON}"

echo ""
echo "--- Stage 1/2: security-critical suites (tests/security tests/api tests/integration) ---"
if "${PYTHON}" -m pytest tests/security tests/api tests/integration -q; then
  echo ""
  echo "Stage 1/2 PASSED: security-critical suites are green."
else
  banner "SECURITY GATE: FAIL (security-critical suite failure)"
  echo "One or more tests in tests/security, tests/api, or tests/integration failed."
  echo "This blocks merge — do not bypass. Investigate before proceeding."
  exit 1
fi

echo ""
echo "--- Stage 2/2: full test suite ---"
if "${PYTHON}" -m pytest -q; then
  echo ""
  echo "Stage 2/2 PASSED: full suite is green."
else
  banner "SECURITY GATE: FAIL (full suite failure)"
  echo "The security-critical suites passed, but the full test suite did not."
  echo "This blocks merge — do not bypass. Investigate before proceeding."
  exit 1
fi

banner "SECURITY GATE: PASS"
exit 0
