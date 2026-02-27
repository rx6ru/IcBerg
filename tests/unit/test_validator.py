"""Tests for the AST code validator."""

import pytest
from backend.core.validator import validate_generated_code, ValidationResult


class TestDangerousImports:

    def test_os_blocked(self):
        result = validate_generated_code("import os")
        assert not result.is_valid
        assert any("os" in v for v in result.violations)

    def test_sys_blocked(self):
        result = validate_generated_code("import sys")
        assert not result.is_valid

    def test_subprocess_blocked(self):
        result = validate_generated_code("import subprocess")
        assert not result.is_valid

    def test_shutil_blocked(self):
        result = validate_generated_code("import shutil")
        assert not result.is_valid

    def test_from_os_blocked(self):
        result = validate_generated_code("from os import path")
        assert not result.is_valid

    def test_from_os_path_blocked(self):
        result = validate_generated_code("from os.path import join")
        assert not result.is_valid


class TestAllowedImports:

    def test_pandas(self):
        assert validate_generated_code("import pandas as pd").is_valid

    def test_numpy(self):
        assert validate_generated_code("import numpy as np").is_valid

    def test_math(self):
        assert validate_generated_code("import math").is_valid

    def test_statistics(self):
        assert validate_generated_code("import statistics").is_valid


class TestDangerousBuiltins:

    def test_exec_blocked(self):
        assert not validate_generated_code("exec('print(1)')").is_valid

    def test_eval_blocked(self):
        assert not validate_generated_code("eval('1+1')").is_valid

    def test_open_blocked(self):
        assert not validate_generated_code("f = open('/etc/passwd')").is_valid

    def test_dunder_import_blocked(self):
        assert not validate_generated_code("__import__('os')").is_valid

    def test_compile_blocked(self):
        assert not validate_generated_code("compile('code', '<string>', 'exec')").is_valid

    def test_globals_blocked(self):
        assert not validate_generated_code("g = globals()").is_valid


class TestValidCode:

    def test_simple_assignment(self):
        result = validate_generated_code("result = df['Age'].mean()")
        assert result.is_valid
        assert len(result.violations) == 0

    def test_pandas_operations(self):
        assert validate_generated_code("result = df.groupby('Sex')['Survived'].mean()").is_valid

    def test_multiline(self):
        code = """
ages = df['Age'].dropna()
mean_age = ages.mean()
result = f"Mean age: {mean_age:.2f}"
"""
        assert validate_generated_code(code).is_valid

    def test_matplotlib(self):
        code = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
df['Age'].hist(ax=ax)
"""
        assert validate_generated_code(code).is_valid


class TestSyntaxErrors:

    def test_syntax_error(self):
        result = validate_generated_code("def foo(")
        assert not result.is_valid
        assert any("syntax" in v.lower() for v in result.violations)

    def test_empty_code(self):
        assert not validate_generated_code("").is_valid


class TestValidationResult:

    def test_dataclass_fields(self):
        result = ValidationResult(is_valid=True, violations=[], sanitized_code="x = 1")
        assert result.is_valid is True
        assert result.violations == []
        assert result.sanitized_code == "x = 1"
