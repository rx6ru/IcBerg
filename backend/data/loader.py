"""Titanic dataset loader with feature engineering.

Loads CSV, creates derived columns (Title, AgeGroup, FareGroup, FamilySize),
and drops leakage columns before returning a clean DataFrame.
"""

import math
import re

import pandas as pd


def extract_title(name: str) -> str:
    """Pull the title (Mr, Mrs, Miss, Master, Dr) from a passenger name.
    Anything else maps to 'Other'.

    Args:
        name: Full passenger name, e.g. "Braund, Mr. Owen Harris"

    Returns:
        Extracted title string.
    """
    match = re.search(r",\s*(\w+)\.", name)
    if not match:
        return "Other"

    title = match.group(1)
    return title if title in {"Mr", "Mrs", "Miss", "Master", "Dr"} else "Other"


def assign_age_group(age: float) -> str:
    """Bin age into Child(<13) / Teen(13-17) / YoungAdult(18-30) /
    Adult(31-60) / Senior(>60). NaN maps to Unknown.

    Args:
        age: Passenger age (may be NaN).

    Returns:
        Age group label.
    """
    if age is None or (isinstance(age, float) and math.isnan(age)):
        return "Unknown"

    if age < 13:
        return "Child"
    if age < 18:
        return "Teen"
    if age <= 30:
        return "YoungAdult"
    if age <= 60:
        return "Adult"
    return "Senior"


def assign_fare_group(fare: float) -> str:
    """Bin fare into Budget(<8) / Economy(8-30) / Comfort(30-100) / Luxury(>100).

    Args:
        fare: Ticket fare amount.

    Returns:
        Fare group label.
    """
    if fare < 8:
        return "Budget"
    if fare <= 30:
        return "Economy"
    if fare <= 100:
        return "Comfort"
    return "Luxury"


def compute_family_size(sibsp: int, parch: int) -> int:
    """Calculate total family size: SibSp + Parch + 1 (self).

    Args:
        sibsp: Number of siblings/spouses aboard.
        parch: Number of parents/children aboard.

    Returns:
        Total family size including the passenger.
    """
    return sibsp + parch + 1


_LEAKAGE_COLUMNS = ["Cabin", "Ticket"]


def _apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns to raw dataframe. Must run before dropping Name."""
    df = df.copy()
    df["Title"] = df["Name"].apply(extract_title)
    df["FamilySize"] = df.apply(
        lambda row: compute_family_size(row["SibSp"], row["Parch"]),
        axis=1,
    )
    df["AgeGroup"] = df["Age"].apply(assign_age_group)
    df["FareGroup"] = df["Fare"].apply(assign_fare_group)
    return df


def load_dataframe(csv_path: str) -> pd.DataFrame:
    """Load titanic CSV, engineer features, drop leakage columns.

    Args:
        csv_path: Path to the titanic.csv file.

    Returns:
        Cleaned DataFrame with engineered columns.

    Raises:
        FileNotFoundError: If CSV path doesn't exist.
    """
    df = pd.read_csv(csv_path)
    df = _apply_feature_engineering(df)

    cols_to_drop = [c for c in _LEAKAGE_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    return df


def get_schema_metadata(df: pd.DataFrame) -> str:
    """Build a human-readable schema string for the LLM system prompt.

    Args:
        df: The engineered DataFrame.

    Returns:
        Multi-line schema description with dtypes and sample values.
    """
    lines = ["Dataset Schema (Titanic - Engineered):"]
    lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    lines.append("")

    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isna().sum()
        samples = df[col].dropna().unique()[:5]
        sample_str = ", ".join(str(s) for s in samples)

        null_info = f", {null_count} nulls" if null_count > 0 else ""
        lines.append(f"- {col} ({dtype}{null_info}): {sample_str}")

    return "\n".join(lines)
