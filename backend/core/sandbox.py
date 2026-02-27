"""Sandboxed code execution with timeout and error classification.

Runs validated Python code against a DataFrame copy, enforces a
SIGALRM-based timeout (default 5s), and classifies errors as
retryable (wrong column, type mismatch) vs non-retryable (syntax, name).
"""

import signal
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ExecutionResult:
    """Result of sandboxed code execution.

    Attributes:
        success: Whether execution completed without error.
        output: Value of the ``result`` variable after execution.
        output_type: One of scalar, dataframe, series, string, none.
        error: Error message if execution failed, None otherwise.
        retryable: Whether the LLM should retry with fixed code.
        execution_time_ms: Wall-clock time in milliseconds.
    """
    success: bool
    output: Any = None
    output_type: str = "none"
    error: str | None = None
    retryable: bool = False
    execution_time_ms: int = 0


import builtins
from backend.core.validator import ALLOWED_IMPORT_ROOTS

RETRYABLE_ERRORS = (KeyError, TypeError, ValueError, AttributeError, IndexError, ZeroDivisionError)
NON_RETRYABLE_ERRORS = (SyntaxError, NameError, ImportError, MemoryError, RecursionError)

_DANGEROUS = frozenset({
    "exec", "eval", "open", "__import__", "compile",
    "globals", "locals", "getattr", "setattr", "delattr", "breakpoint", "input", "print"
})


def _safe_import(name: str, globals: dict | None = None, locals: dict | None = None, fromlist: tuple = (), level: int = 0) -> Any:
    """Wrapper around __import__ to enforce the same whitelist as the AST validator at runtime."""
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"Dynamic import of '{name}' is blocked by sandbox")
    return __import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in dir(builtins)
    if name not in _DANGEROUS and not name.startswith("_")
}
SAFE_BUILTINS["__build_class__"] = builtins.__build_class__
SAFE_BUILTINS["__name__"] = "__main__"
SAFE_BUILTINS["__import__"] = _safe_import

class _TimeoutError(Exception):
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    raise _TimeoutError("Code execution exceeded time limit")


def _detect_output_type(value: Any) -> str:
    """Classify the type of the result variable.

    Args:
        value: The object assigned to ``result`` in the executed code.

    Returns:
        One of: 'scalar', 'dataframe', 'series', 'string', 'none'.
    """
    if value is None:
        return "none"
    if isinstance(value, pd.DataFrame):
        return "dataframe"
    if isinstance(value, pd.Series):
        return "series"
    if isinstance(value, str):
        return "string"
    return "scalar"


def _classify_error(error: Exception) -> bool:
    """Check if the error is something the LLM can fix by rewriting code.

    Args:
        error: The caught exception.

    Returns:
        True if retryable (KeyError, TypeError, etc.), False otherwise.
    """
    if isinstance(error, RETRYABLE_ERRORS):
        return True
    return False


def execute_code(code: str, df: pd.DataFrame, timeout: int = 5) -> ExecutionResult:
    """Run code in a restricted namespace with ``df`` available.

    The executed code should assign its output to a variable named ``result``.
    The original DataFrame is never mutated — a copy is used internally.

    Args:
        code: Validated Python source code.
        df: Titanic DataFrame (will be copied, not mutated).
        timeout: Max execution time in seconds.

    Returns:
        ExecutionResult with output, type classification, and error info.
    """
    start_time = time.monotonic()
    df_copy = df.copy()

    exec_globals: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    exec_locals: dict[str, Any] = {"df": df_copy}

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)

    try:
        signal.alarm(timeout)
        exec(code, exec_globals, exec_locals)  # noqa: S102
        signal.alarm(0)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        output = exec_locals.get("result", None)

        return ExecutionResult(
            success=True,
            output=output,
            output_type=_detect_output_type(output),
            execution_time_ms=elapsed_ms,
        )

    except _TimeoutError:
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        return ExecutionResult(
            success=False,
            error=f"Timeout: code execution exceeded {timeout}s limit",
            retryable=False,
            execution_time_ms=elapsed_ms,
        )

    except SyntaxError as e:
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        return ExecutionResult(
            success=False,
            error=f"SyntaxError: {e}",
            retryable=False,
            execution_time_ms=elapsed_ms,
        )

    except Exception as e:
        signal.alarm(0)
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        return ExecutionResult(
            success=False,
            error=f"{type(e).__name__}: {e}",
            retryable=_classify_error(e),
            execution_time_ms=elapsed_ms,
        )

    finally:
        signal.signal(signal.SIGALRM, old_handler)
