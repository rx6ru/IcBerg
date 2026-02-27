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

    def test_builtins_blocked(self):
        assert not validate_generated_code("__builtins__['open']('file')").is_valid

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


class TestDangerousAttributes:

    def test_dunder_class_blocked(self):
        assert not validate_generated_code("x = [].__class__").is_valid

    def test_dunder_subclasses_blocked(self):
        code = "x = object.__subclasses__()"
        assert not validate_generated_code(code).is_valid

    def test_dunder_globals_attr_blocked(self):
        code = "x = f.__globals__"
        assert not validate_generated_code(code).is_valid

    def test_introspection_attack_blocked(self):
        code = """
for c in [].__class__.__base__.__subclasses__():
    if c.__init__.__globals__ and 'sys' in c.__init__.__globals__:
        sys = c.__init__.__globals__['sys']
        os = sys.modules['os']
        result = os.popen('whoami').read()
"""
        result = validate_generated_code(code)
        assert not result.is_valid
        assert any("__class__" in v for v in result.violations)

    def test_private_attr_blocked(self):
        assert not validate_generated_code("x = obj._private_method()").is_valid


class TestTransientColumns:

    def test_assign_and_read_transient_column(self):
        from backend.core.validator import set_known_columns
        set_known_columns(["A", "B"])
        code = """
df['C'] = df['A'] + df['B']
result = df['C'].sum()
"""
        result = validate_generated_code(code)
        assert result.is_valid, f"Violations: {result.violations}"

    def test_unknown_column_still_blocked(self):
        from backend.core.validator import set_known_columns
        set_known_columns(["A", "B"])
        code = "result = df['Z'].mean()"
        assert not validate_generated_code(code).is_valid

    def test_transient_column_without_assignment(self):
        from backend.core.validator import set_known_columns
        set_known_columns(["A"])
        code = "result = df['NotAssigned'].mean()"
        assert not validate_generated_code(code).is_valid


class TestValidationResult:

    def test_dataclass_fields(self):
        result = ValidationResult(is_valid=True, violations=[], sanitized_code="x = 1")
        assert result.is_valid is True
        assert result.violations == []
        assert result.sanitized_code == "x = 1"

