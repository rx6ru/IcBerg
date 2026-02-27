"""Unit tests for agent tools."""

import pytest
import pandas as pd

from backend.agent.tools import (
    get_dataset_info,
    get_statistics,
    query_data,
    visualize_data,
    set_dataframe,
)
from backend.core.validator import set_known_columns


@pytest.fixture(autouse=True)
def setup_tools(engineered_df):
    """Wire up the module-level DataFrame and known columns before each test."""
    set_dataframe(engineered_df)
    set_known_columns(list(engineered_df.columns))
    yield


class TestGetDatasetInfo:

    def test_returns_schema_with_all_columns(self, engineered_df):
        result = get_dataset_info.invoke({})
        for col in engineered_df.columns:
            assert col in result

    def test_returns_string(self):
        result = get_dataset_info.invoke({})
        assert isinstance(result, str)


class TestGetStatistics:

    def test_contains_describe_fields(self):
        result = get_statistics.invoke({})
        for stat in ["mean", "std", "min", "max"]:
            assert stat in result

    def test_returns_string(self):
        result = get_statistics.invoke({})
        assert isinstance(result, str)


class TestQueryData:

    def test_valid_operation(self):
        result = query_data.invoke({"operation": "result = df['Survived'].mean()"})
        assert "ERROR" not in result
        assert "0.5" in result  # 5/10 survived

    def test_invalid_code_returns_error(self):
        result = query_data.invoke({"operation": "import os"})
        assert result.startswith("ERROR:")

    def test_non_retryable_stops_immediately(self):
        # SyntaxError is non-retryable - should fail without retrying
        result = query_data.invoke({"operation": "result = df[[["})
        assert result.startswith("ERROR:")

    def test_unknown_column_caught_by_validator(self):
        result = query_data.invoke({"operation": "result = df['NonExistent'].mean()"})
        assert result.startswith("ERROR:")
        assert "NonExistent" in result

    def test_retryable_error(self):
        # KeyError is retryable - will fail after MAX_RETRIES attempts
        result = query_data.invoke({"operation": "result = df['AlsoNotReal'].sum()"})
        assert result.startswith("ERROR:")


class TestVisualizeData:

    def test_valid_chart_returns_base64(self):
        code = (
            "import matplotlib.pyplot as plt\n"
            "fig, ax = plt.subplots()\n"
            "df['Survived'].value_counts().plot(kind='bar', ax=ax)\n"
            "result = fig"
        )
        result = visualize_data.invoke({"chart_code": code})
        assert result.startswith("BASE64:")

    def test_invalid_code_returns_error(self):
        result = visualize_data.invoke({"chart_code": "import os"})
        assert result.startswith("ERROR:")

    def test_syntax_error_returns_error(self):
        result = visualize_data.invoke({"chart_code": "result = df[[["})
        assert result.startswith("ERROR:")
