"""
Unit Tests for Data Processing Module
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocess import (
    handle_missing_values,
    encode_target,
    remove_duplicates,
    check_data_quality,
)


class TestHandleMissingValues:
    """Tests for missing value handling"""

    def test_no_missing_values(self):
        """Test when there are no missing values"""
        df = pd.DataFrame(
            {"age": [45, 50, 55], "chol": [200, 250, 300], "target": [0, 1, 0]}
        )

        result = handle_missing_values(df)

        assert result.isnull().sum().sum() == 0
        assert len(result) == 3

    def test_missing_numeric_median_strategy(self):
        """Test median filling for numeric columns"""
        df = pd.DataFrame(
            {"age": [45, np.nan, 55], "chol": [200, 250, np.nan], "target": [0, 1, 0]}
        )

        result = handle_missing_values(df, strategy="median")

        assert result.isnull().sum().sum() == 0
        # Median of [45, 55] is 50
        assert result["age"].iloc[1] == 50.0
        # Median of [200, 250] is 225
        assert result["chol"].iloc[2] == 225.0

    def test_preserves_non_missing_values(self):
        """Test that non-missing values are preserved"""
        df = pd.DataFrame(
            {"age": [45, np.nan, 55], "chol": [200, 250, 300], "target": [0, 1, 0]}
        )

        result = handle_missing_values(df)

        assert result["age"].iloc[0] == 45
        assert result["age"].iloc[2] == 55
        assert list(result["chol"]) == [200, 250, 300]


class TestEncodeTarget:
    """Tests for target encoding"""

    def test_binary_encoding(self):
        """Test that target is correctly encoded to binary"""
        df = pd.DataFrame({"target": [0, 1, 2, 3, 4]})

        result = encode_target(df)

        expected = [0, 1, 1, 1, 1]
        assert list(result["target"]) == expected

    def test_preserves_zero(self):
        """Test that 0 remains 0"""
        df = pd.DataFrame({"target": [0, 0, 0]})

        result = encode_target(df)

        assert all(result["target"] == 0)

    def test_all_disease(self):
        """Test when all values indicate disease"""
        df = pd.DataFrame({"target": [1, 2, 3, 4]})

        result = encode_target(df)

        assert all(result["target"] == 1)


class TestRemoveDuplicates:
    """Tests for duplicate removal"""

    def test_remove_exact_duplicates(self):
        """Test removal of exact duplicate rows"""
        df = pd.DataFrame(
            {"age": [45, 45, 50], "chol": [200, 200, 250], "target": [0, 0, 1]}
        )

        result = remove_duplicates(df)

        assert len(result) == 2

    def test_no_duplicates(self):
        """Test when there are no duplicates"""
        df = pd.DataFrame(
            {"age": [45, 50, 55], "chol": [200, 250, 300], "target": [0, 1, 0]}
        )

        result = remove_duplicates(df)

        assert len(result) == 3


class TestCheckDataQuality:
    """Tests for data quality checking"""

    def test_quality_report_structure(self):
        """Test that quality report has expected keys"""
        df = pd.DataFrame({"age": [45, 50, 55], "target": [0, 1, 0]})

        report = check_data_quality(df)

        assert "total_records" in report
        assert "total_features" in report
        assert "missing_values" in report
        assert "duplicate_records" in report

    def test_correct_record_count(self):
        """Test that record count is correct"""
        df = pd.DataFrame({"age": [45, 50, 55, 60], "target": [0, 1, 0, 1]})

        report = check_data_quality(df)

        assert report["total_records"] == 4
        assert report["total_features"] == 2
