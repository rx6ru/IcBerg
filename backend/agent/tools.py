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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = _get_df()

    validation = validate_generated_code(chart_code)
    if not validation.is_valid:
        reasons = "; ".join(validation.violations)
        return f"ERROR: Code validation failed — {reasons}"

    # Execute the chart code
    result = execute_code(chart_code, df)

    if not result.success:
        return f"ERROR: {result.error}"

    # Try to capture the figure
    fig = result.output
    if fig is None:
        # Fallback: grab current figure if code didn't assign to `result`
        fig = plt.gcf()

    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return f"BASE64:{encoded}"
    except Exception as e:
        plt.close("all")
        return f"ERROR: Failed to encode chart — {e}"
