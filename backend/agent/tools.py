"""Agent tools for Titanic dataset analysis.

Four tools that the LangGraph ReAct agent can call. All return strings
(LangGraph serialization requirement). query_data and visualize_data
use the validator + sandbox pipeline with retry logic.
"""

import base64
import io

import pandas as pd
import structlog
from langchain_core.tools import tool

from backend.core.sandbox import execute_code
from backend.core.validator import validate_generated_code
from backend.data.loader import get_schema_metadata

logger = structlog.get_logger(__name__)

# Module-level DataFrame reference, set by agent.py during startup
_df: pd.DataFrame | None = None

MAX_RETRIES = 3


def set_dataframe(df: pd.DataFrame) -> None:
    """Set the singleton DataFrame reference for all tools.

    Args:
        df: The loaded and engineered Titanic DataFrame.
    """
    global _df
    _df = df


def _get_df() -> pd.DataFrame:
    """Get the current DataFrame, raising if not initialized."""
    if _df is None:
        raise RuntimeError("DataFrame not initialized. Call set_dataframe() first.")
    return _df


@tool
def get_dataset_info() -> str:
    """Get the dataset schema: column names, dtypes, null counts, and sample values."""
    return get_schema_metadata(_get_df())


@tool
def get_statistics() -> str:
    """Get descriptive statistics (count, mean, std, min, quartiles, max) for all numeric columns."""
    return _get_df().describe().to_string()


@tool
def query_data(operation: str) -> str:
    """Execute a pandas operation on the Titanic dataset.

    Args:
        operation: Valid Python/pandas code. Must assign the result to a variable named `result`.
                   Example: result = df['Survived'].mean()

    Returns:
        The computed result as a string, or an error message prefixed with "ERROR:".
    """
    df = _get_df()

    # Validate code before execution
    validation = validate_generated_code(operation)
    if not validation.is_valid:
        reasons = "; ".join(validation.violations)
        logger.warning("tool.query_data.validation_failed", reasons=reasons)
        return f"ERROR: Code validation failed — {reasons}"

    # Execute with retry on retryable errors
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        result = execute_code(operation, df)

        if result.success:
            output = result.output
            if isinstance(output, pd.DataFrame):
                return output.to_string()
            if isinstance(output, pd.Series):
                return output.to_string()
            return str(output) if output is not None else "No result produced."

        last_error = result.error
        if not result.retryable:
            logger.warning("tool.query_data.non_retryable", error=last_error, attempt=attempt)
            break

        logger.debug("tool.query_data.retry", error=last_error, attempt=attempt)

    return f"ERROR: {last_error}"


@tool
def visualize_data(chart_code: str) -> str:
    """Generate a chart from the Titanic dataset using matplotlib or seaborn.

    Args:
        chart_code: Python code that creates a matplotlib/seaborn figure.
                    Must use `df` as the DataFrame variable.
                    Example: import matplotlib.pyplot as plt; fig, ax = plt.subplots(); df['Age'].hist(ax=ax); result = fig

    Returns:
        Base64-encoded PNG prefixed with "BASE64:", or an error message prefixed with "ERROR:".
    """
    df = _get_df()

    validation = validate_generated_code(chart_code)
    if not validation.is_valid:
        reasons = "; ".join(validation.violations)
        return f"ERROR: Code validation failed — {reasons}"

    # Wrap the user's chart code to render to base64 inside the subprocess.
    # This avoids passing matplotlib Figure objects across process boundaries.
    wrapped_code = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "import base64, io\n"
        f"{chart_code}\n"
        "# --- Auto-injected chart capture ---\n"
        "_fig = result if hasattr(result, 'savefig') else plt.gcf()\n"
        "_buf = io.BytesIO()\n"
        "_fig.savefig(_buf, format='png', bbox_inches='tight', dpi=100)\n"
        "_buf.seek(0)\n"
        "result = base64.b64encode(_buf.read()).decode('utf-8')\n"
        "plt.close('all')\n"
    )

    exec_result = execute_code(wrapped_code, df, timeout=10)

    if not exec_result.success:
        return f"ERROR: {exec_result.error}"

    encoded = exec_result.output
    if encoded and isinstance(encoded, str):
        return f"BASE64:{encoded}"
    return "ERROR: Chart rendering produced no output"
