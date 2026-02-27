"""AST-based validator for generated Python code.

Walks the syntax tree and blocks dangerous imports/builtins
before code reaches the sandbox.
"""

import ast
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of AST validation.

    Attributes:
        is_valid: Whether the code passed all security checks.
        violations: List of human-readable violation descriptions.
        sanitized_code: Original code if valid, empty string if not.
    """
    is_valid: bool
    violations: list[str] = field(default_factory=list)
    sanitized_code: str = ""


ALLOWED_IMPORT_ROOTS = frozenset({
    "pandas", "numpy", "matplotlib", "seaborn", "math", "statistics",
})

DANGEROUS_BUILTINS = frozenset({
    "exec", "eval", "open", "__import__", "compile",
    "globals", "locals", "getattr", "setattr", "delattr", "breakpoint",
})


def validate_generated_code(code: str) -> ValidationResult:
    """Parse code into AST and check for blocked imports and dangerous calls.

    Args:
        code: Python source code string to validate.

    Returns:
        ValidationResult with validity status, violations, and sanitized code.
    """
    violations: list[str] = []

    if not code or not code.strip():
        return ValidationResult(is_valid=False, violations=["Empty code is not allowed"])

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return ValidationResult(is_valid=False, violations=[f"Syntax error: {e}"])

    for node in ast.walk(tree):
        _check_imports(node, violations)
        _check_dangerous_calls(node, violations)

    is_valid = len(violations) == 0
    return ValidationResult(
        is_valid=is_valid,
        violations=violations,
        sanitized_code=code if is_valid else "",
    )


def _check_imports(node: ast.AST, violations: list[str]) -> None:
    """Reject any import whose root module isn't whitelisted."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                violations.append(f"Blocked import: '{alias.name}' — not in allowed list")

    elif isinstance(node, ast.ImportFrom) and node.module:
        root = node.module.split(".")[0]
        if root not in ALLOWED_IMPORT_ROOTS:
            violations.append(f"Blocked import: 'from {node.module}' — not in allowed list")


def _check_dangerous_calls(node: ast.AST, violations: list[str]) -> None:
    """Reject direct calls to dangerous builtins (exec, eval, open, etc.)."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in DANGEROUS_BUILTINS:
            violations.append(f"Blocked builtin call: '{node.func.id}()' — dangerous operation")
