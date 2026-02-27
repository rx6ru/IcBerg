"""Tests for the data loader and feature engineering pipeline."""

import pandas as pd
import pytest

from backend.data.loader import (
    assign_age_group,
    assign_fare_group,
    compute_family_size,
    extract_title,
    get_schema_metadata,
    load_dataframe,
)


class TestExtractTitle:

    def test_mr(self):
        assert extract_title("Braund, Mr. Owen Harris") == "Mr"

    def test_mrs(self):
        assert extract_title("Cumings, Mrs. John Bradley") == "Mrs"

    def test_miss(self):
        assert extract_title("Heikkinen, Miss. Laina") == "Miss"

    def test_master(self):
        assert extract_title("Palsson, Master. Gosta Leonard") == "Master"

    def test_dr(self):
        assert extract_title("Nasser, Dr. Nicholas") == "Dr"

    def test_rare_maps_to_other(self):
        assert extract_title("Byles, Rev. Thomas Roussel Davids") == "Other"
        assert extract_title("Weir, Col. John") == "Other"

    def test_no_title(self):
        assert extract_title("Some Random Name") == "Other"


class TestAssignAgeGroup:

    def test_child(self):
        assert assign_age_group(5.0) == "Child"

    def test_child_boundary(self):
        assert assign_age_group(12.9) == "Child"

    def test_teen(self):
        assert assign_age_group(15.0) == "Teen"

    def test_young_adult(self):
        assert assign_age_group(25.0) == "YoungAdult"

    def test_adult(self):
        assert assign_age_group(45.0) == "Adult"

    def test_senior(self):
        assert assign_age_group(65.0) == "Senior"

    def test_nan_returns_unknown(self):
        assert assign_age_group(float("nan")) == "Unknown"


class TestAssignFareGroup:

    def test_budget(self):
        assert assign_fare_group(5.0) == "Budget"

    def test_economy(self):
        assert assign_fare_group(15.0) == "Economy"

    def test_comfort(self):
        assert assign_fare_group(50.0) == "Comfort"

    def test_luxury(self):
        assert assign_fare_group(150.0) == "Luxury"

    def test_boundary_budget_economy(self):
        assert assign_fare_group(8.0) == "Economy"


class TestComputeFamilySize:

    def test_solo_traveler(self):
        assert compute_family_size(0, 0) == 1

    def test_with_siblings(self):
        assert compute_family_size(2, 0) == 3

    def test_with_parents_children(self):
        assert compute_family_size(0, 2) == 3

    def test_full_family(self):
        assert compute_family_size(3, 1) == 5


class TestLoadDataframe:

    def test_loads_real_csv(self, tmp_path):
        import shutil
        csv_src = "backend/data/titanic.csv"
        csv_dst = tmp_path / "titanic.csv"
        shutil.copy(csv_src, csv_dst)

        df = load_dataframe(str(csv_dst))

        assert len(df) == 891
        assert "Title" in df.columns
        assert "FamilySize" in df.columns
        assert "AgeGroup" in df.columns
        assert "FareGroup" in df.columns

        # leakage columns dropped (only Cabin and Ticket)
        assert "Cabin" not in df.columns
        assert "Ticket" not in df.columns

        # Name and PassengerId are kept for analysis
        assert "Name" in df.columns
        assert "PassengerId" in df.columns

    def test_ground_truth_male_percentage(self, tmp_path):
        import shutil
        csv_src = "backend/data/titanic.csv"
        csv_dst = tmp_path / "titanic.csv"
        shutil.copy(csv_src, csv_dst)

        df = load_dataframe(str(csv_dst))
        male_pct = (df["Sex"] == "male").mean() * 100
        assert abs(male_pct - 64.76) < 0.1

    def test_ground_truth_mean_fare(self, tmp_path):
        import shutil
        csv_src = "backend/data/titanic.csv"
        csv_dst = tmp_path / "titanic.csv"
        shutil.copy(csv_src, csv_dst)

        df = load_dataframe(str(csv_dst))
        assert abs(df["Fare"].mean() - 32.20) < 0.1

    def test_ground_truth_embarked_counts(self, tmp_path):
        import shutil
        csv_src = "backend/data/titanic.csv"
        csv_dst = tmp_path / "titanic.csv"
        shutil.copy(csv_src, csv_dst)

        df = load_dataframe(str(csv_dst))
        counts = df["Embarked"].value_counts()
        assert counts["S"] == 644
        assert counts["C"] == 168
        assert counts["Q"] == 77


class TestGetSchemaMetadata:

    def test_returns_string(self, mock_titanic_df):
        mock_titanic_df["Title"] = mock_titanic_df["Name"].apply(extract_title)
        mock_titanic_df["FamilySize"] = mock_titanic_df.apply(
            lambda r: compute_family_size(r["SibSp"], r["Parch"]), axis=1
        )
        mock_titanic_df["AgeGroup"] = mock_titanic_df["Age"].apply(assign_age_group)
        mock_titanic_df["FareGroup"] = mock_titanic_df["Fare"].apply(assign_fare_group)

        metadata = get_schema_metadata(mock_titanic_df)
        assert isinstance(metadata, str)
        assert len(metadata) > 0

    def test_contains_column_names(self, mock_titanic_df):
        mock_titanic_df["Title"] = mock_titanic_df["Name"].apply(extract_title)
        metadata = get_schema_metadata(mock_titanic_df)
        assert "Survived" in metadata
        assert "Sex" in metadata
        assert "Title" in metadata
