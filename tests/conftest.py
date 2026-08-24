"""Shared fixtures for all tests.

(Previously held pandas-based Titanic fixtures -- mock_titanic_df, sample_valid_code,
sample_dangerous_code, sample_timeout_code, engineered_df -- used only by the legacy
backend/core/sandbox.py, backend/core/validator.py, and backend/data/loader.py tests,
which were removed along with those modules. No remaining test uses them, so they were
dropped here too rather than left as dead fixtures pinning an unused `pandas`
dependency. See tests/integration/conftest.py for the current shared DB fixture.)
"""
