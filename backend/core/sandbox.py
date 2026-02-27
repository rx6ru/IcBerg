"""Sandboxed code execution with process isolation, timeout, and error classification.

Runs validated Python code against a DataFrame copy inside a separate
child process. This provides full OOM and crash isolation — if the user
code allocates too much memory, the child process is killed without
affecting the parent FastAPI server.

Timeout is enforced via `process.join(timeout)`.  Memory limit is
applied via `resource.setrlimit(RLIMIT_AS)` inside the child.
"""

import multiprocessing
import os
import resource
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

# Extra memory headroom the child process is allowed to allocate beyond
# its inherited virtual footprint.  Default: 1024 MB (1 GB).
# This prevents catastrophic multi-GB allocations while letting normal
# Pandas operations run freely.
SANDBOX_HEADROOM_MB = int(os.environ.get("SANDBOX_MEMORY_LIMIT_MB", "1024"))


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


def _get_current_vm_bytes() -> int:
    """Read current virtual memory size from /proc/self/status."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmSize:"):
                    return int(line.split()[1]) * 1024  # kB → bytes
    except (OSError, ValueError, IndexError):
        pass
    return 0


# Maximum characters allowed in IPC output to prevent deserialization OOM
# in the parent process.  100 KB of text is more than enough for any
# analytical result the LLM needs to interpret.
_IPC_OUTPUT_LIMIT = 100_000


def _worker(code: str, df: pd.DataFrame, result_queue: multiprocessing.Queue,
            headroom_mb: int) -> None:
    """Child process worker — applies memory limit and executes code.

    This function runs in a completely isolated process. If it crashes
    (OOM, segfault), only this child process dies.
    """
    # Apply memory limit: current footprint + headroom
    try:
        current_vm = _get_current_vm_bytes()
        headroom_bytes = headroom_mb * 1024 * 1024
        limit = current_vm + headroom_bytes if current_vm > 0 else headroom_bytes
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ValueError, resource.error):
        pass  # Non-fatal: some systems don't support RLIMIT_AS

    df_copy = df.copy()
    exec_globals: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    exec_locals: dict[str, Any] = {"df": df_copy}

    try:
        exec(code, exec_globals, exec_locals)  # noqa: S102

        output = exec_locals.get("result", None)
        output_type = _detect_output_type(output)

        # Serialize output for inter-process transfer
        # DataFrames and Series must be converted to strings for the queue
        if output_type == "dataframe":
            output = output.to_string()
        elif output_type == "series":
            output = output.to_string()

        # Truncate to prevent IPC memory bomb: a malicious payload could
        # generate an 800MB string under RLIMIT_AS and OOM the parent
        # during pickle deserialization.
        if isinstance(output, str) and len(output) > _IPC_OUTPUT_LIMIT:
            output = output[:_IPC_OUTPUT_LIMIT] + "\n...[TRUNCATED]"

        result_queue.put({
            "success": True,
            "output": output,
            "output_type": output_type,
            "error": None,
            "retryable": False,
        })

    except SyntaxError as e:
        result_queue.put({
            "success": False,
            "output": None,
            "output_type": "none",
            "error": f"SyntaxError: {e}",
            "retryable": False,
        })

    except MemoryError:
        result_queue.put({
            "success": False,
            "output": None,
            "output_type": "none",
            "error": f"MemoryError: code exceeded {headroom_mb}MB memory limit",
            "retryable": False,
        })

    except Exception as e:
        retryable = _classify_error(e)
        result_queue.put({
            "success": False,
            "output": None,
            "output_type": "none",
            "error": f"{type(e).__name__}: {e}",
            "retryable": retryable,
        })


def execute_code(code: str, df: pd.DataFrame, timeout: int = 5) -> ExecutionResult:
    """Run code in an isolated child process with resource limits.

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
    result_queue: multiprocessing.Queue = multiprocessing.Queue()

    worker = multiprocessing.Process(
        target=_worker,
        args=(code, df, result_queue, SANDBOX_HEADROOM_MB),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=timeout)

    elapsed_ms = int((time.monotonic() - start_time) * 1000)

    if worker.is_alive():
        # Timeout: kill the child process
        worker.kill()
        worker.join(timeout=2)
        return ExecutionResult(
            success=False,
            error=f"Timeout: code execution exceeded {timeout}s limit",
            retryable=False,
            execution_time_ms=elapsed_ms,
        )

    # Child exited — check how
    if worker.exitcode != 0 and result_queue.empty():
        # Process was killed (OOM, segfault, etc.)
        if worker.exitcode == -9:  # SIGKILL (OOM killer)
            error_msg = "Process killed: memory limit exceeded (OOM)"
        elif worker.exitcode and worker.exitcode < 0:
            error_msg = f"Process crashed with signal {-worker.exitcode}"
        else:
            error_msg = f"Process exited with code {worker.exitcode}"
        return ExecutionResult(
            success=False,
            error=error_msg,
            retryable=False,
            execution_time_ms=elapsed_ms,
        )

    # Normal exit — read the result from the queue
    if result_queue.empty():
        return ExecutionResult(
            success=False,
            error="No result returned from sandbox worker",
            retryable=False,
            execution_time_ms=elapsed_ms,
        )

    result_data = result_queue.get_nowait()
    return ExecutionResult(
        success=result_data["success"],
        output=result_data["output"],
        output_type=result_data["output_type"],
        error=result_data["error"],
        retryable=result_data.get("retryable", False),
        execution_time_ms=elapsed_ms,
    )
