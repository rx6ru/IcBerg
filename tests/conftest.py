"""Shared fixtures for all tests."""

import pandas as pd
import pytest


@pytest.fixture
def mock_titanic_df() -> pd.DataFrame:
    """10-row mock Titanic dataset covering all edge cases."""
    data = {
        "Survived": [0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
        "Pclass": [3, 1, 3, 1, 3, 3, 1, 3, 2, 2],
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley",
            "Heikkinen, Miss. Laina",
            "Futrelle, Mrs. Jacques Heath",
            "Allen, Mr. William Henry",
            "Moran, Mr. James",
            "McCarthy, Mr. Timothy J",
            "Palsson, Master. Gosta Leonard",
            "Johnson, Mrs. Oscar W",
            "Nasser, Dr. Nicholas",
        ],
        "Sex": [
            "male", "female", "female", "female", "male",
            "male", "male", "male", "female", "male",
        ],
        "Age": [22.0, 38.0, 26.0, 35.0, 35.0, None, 54.0, 2.0, 15.0, 66.0],
        "SibSp": [1, 1, 0, 1, 0, 0, 0, 3, 0, 1],
        "Parch": [0, 0, 0, 0, 0, 0, 0, 1, 2, 0],
        "Ticket": [
            "A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "373450",
            "330877", "17463", "349909", "347742", "237736",
        ],
        "Fare": [7.25, 71.28, 7.92, 53.10, 8.05, 8.46, 51.86, 21.07, 11.50, 108.90],
        "Cabin": [None, "C85", None, "C123", None, None, "E46", None, None, "C76"],
        "Embarked": ["S", "C", "S", "S", "S", "Q", "S", "S", "S", "C"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_valid_code() -> str:
    return "result = df['Age'].mean()"


@pytest.fixture
def sample_dangerous_code() -> str:
    return "import os; os.system('rm -rf /')"


@pytest.fixture
def sample_timeout_code() -> str:
    return "import time; time.sleep(10)"
