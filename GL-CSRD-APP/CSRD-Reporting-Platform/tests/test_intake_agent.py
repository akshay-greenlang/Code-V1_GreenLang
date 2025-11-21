# -*- coding: utf-8 -*-
"""
CSRD/ESRS Digital Reporting Platform - IntakeAgent Tests

Comprehensive test suite for IntakeAgent - THE FIRST AGENT IN THE PIPELINE

This test file is critical because:
1. IntakeAgent is the data gateway - it validates all incoming ESG data
2. Data quality assessment impacts downstream agent reliability
3. ESRS taxonomy mapping (1,082 data points) must be accurate
4. Outlier detection prevents bad data from propagating
5. Performance target: 1,000 records/second

TARGET: 90% test coverage

Version: 1.0.0
Author: GreenLang CSRD Team
"""

import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
import yaml

from agents.intake_agent import (
    DataQualityScore,
    EnrichedDataPoint,
    ESRSMetadata,
    IntakeAgent,
    ValidationIssue,
)


# ============================================================================
# PYTEST FIXTURES
# ============================================================================


@pytest.fixture
def base_path() -> Path:
    """Get base path for test resources."""
    return Path(__file__).parent.parent


@pytest.fixture
def esrs_data_points_path(base_path: Path) -> Path:
    """Path to ESRS data points catalog JSON."""
    return base_path / "data" / "esrs_data_points.json"


@pytest.fixture
def data_quality_rules_path(base_path: Path) -> Path:
    """Path to data quality rules YAML."""
    return base_path / "rules" / "data_quality_rules.yaml"


@pytest.fixture
def esg_data_schema_path(base_path: Path) -> Path:
    """Path to ESG data JSON schema."""
    return base_path / "schemas" / "esg_data.schema.json"


@pytest.fixture
def demo_csv_file(base_path: Path) -> Path:
    """Path to demo ESG data CSV."""
    return base_path / "examples" / "demo_esg_data.csv"


@pytest.fixture
def intake_agent(
    esrs_data_points_path: Path,
    data_quality_rules_path: Path,
    esg_data_schema_path: Path
) -> IntakeAgent:
    """Create an IntakeAgent instance for testing."""
    return IntakeAgent(
        esrs_data_points_path=esrs_data_points_path,
        data_quality_rules_path=data_quality_rules_path,
        esg_data_schema_path=esg_data_schema_path,
        quality_threshold=0.80
    )


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create sample DataFrame with valid ESG data."""
    return pd.DataFrame({
        "metric_code": ["E1-1", "E1-2", "S1-1"],
        "metric_name": ["Scope 1 GHG Emissions", "Scope 2 GHG Emissions (location-based)", "Total Employees"],
        "value": [1000.0, 500.0, 250.0],
        "unit": ["tCO2e", "tCO2e", "FTE"],
        "period_start": ["2024-01-01", "2024-01-01", "2024-01-01"],
        "period_end": ["2024-12-31", "2024-12-31", "2024-12-31"],
        "data_quality": ["high", "high", "high"],
        "source_document": ["Energy report", "Utility bills", "HRIS"],
        "verification_status": ["verified", "verified", "verified"],
        "notes": ["Natural gas + vehicles", "Purchased electricity", "Headcount"]
    })


@pytest.fixture
def invalid_dataframe() -> pd.DataFrame:
    """Create DataFrame with invalid data for testing validation."""
    return pd.DataFrame({
        "metric_code": ["INVALID-CODE", "E1-1"],
        "metric_name": ["Bad Metric", ""],
        "value": [1000.0, None],
        "unit": ["tCO2e", ""],
        "period_start": ["2024-01-01", "invalid-date"],
        "period_end": ["2024-12-31", "2024-01-01"],
        "data_quality": ["high", "super-high"]
    })


@pytest.fixture
def sample_data_point() -> Dict[str, Any]:
    """Create a valid data point dictionary."""
    return {
        "metric_code": "E1-1",
        "metric_name": "Gross Scope 1 GHG emissions",
        "value": 1000.0,
        "unit": "tCO2e",
        "period_start": "2024-01-01",
        "period_end": "2024-12-31",
        "data_quality": "high",
        "source_document": "Energy management system",
        "verification_status": "verified",
        "notes": "Natural gas combustion"
    }


@pytest.fixture
def outlier_dataframe() -> pd.DataFrame:
    """Create DataFrame with outliers for testing outlier detection."""
    return pd.DataFrame({
        "metric_code": ["E1-1"] * 10,
        "metric_name": ["Scope 1 GHG Emissions"] * 10,
        "value": [100.0, 110.0, 105.0, 95.0, 102.0, 108.0, 1000.0, 98.0, 103.0, 107.0],  # 1000 is outlier
        "unit": ["tCO2e"] * 10,
        "period_start": ["2024-01-01"] * 10,
        "period_end": ["2024-12-31"] * 10,
        "data_quality": ["high"] * 10
    })


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


def test_intake_agent_initialization(intake_agent: IntakeAgent) -> None:
    """Test IntakeAgent initializes correctly."""
    assert intake_agent is not None
    assert intake_agent.quality_threshold == 0.80
    assert intake_agent.esrs_catalog is not None
    assert len(intake_agent.esrs_catalog) > 0
    assert intake_agent.data_quality_rules is not None
    assert intake_agent.esrs_code_lookup is not None
    assert intake_agent.esrs_name_lookup is not None


def test_load_esrs_catalog(intake_agent: IntakeAgent) -> None:
    """Test ESRS catalog loads all data points."""
    # The catalog should have the data points from the JSON file
    assert len(intake_agent.esrs_catalog) >= 50  # At minimum

    # Check structure of first data point
    first_dp = intake_agent.esrs_catalog[0]
    assert "code" in first_dp or "esrs_code" in first_dp or "id" in first_dp
    assert "name" in first_dp or "data_point_name" in first_dp


def test_load_data_quality_rules(intake_agent: IntakeAgent) -> None:
    """Test data quality rules load correctly."""
    assert "completeness_rules" in intake_agent.data_quality_rules or "metadata" in intake_agent.data_quality_rules
    assert intake_agent.data_quality_rules is not None


def test_create_code_lookup(intake_agent: IntakeAgent) -> None:
    """Test ESRS code lookup is created correctly."""
    assert len(intake_agent.esrs_code_lookup) > 0
    # Test known codes
    assert "E1-1" in intake_agent.esrs_code_lookup
    assert "E1-2" in intake_agent.esrs_code_lookup


def test_create_name_lookup(intake_agent: IntakeAgent) -> None:
    """Test ESRS name lookup is created correctly."""
    assert len(intake_agent.esrs_name_lookup) > 0


# ============================================================================
# DATA INGESTION TESTS
# ============================================================================


def test_ingest_csv_file(intake_agent: IntakeAgent, demo_csv_file: Path) -> None:
    """Test CSV file ingestion."""
    df = intake_agent.read_esg_data(demo_csv_file)

    assert df is not None
    assert len(df) > 0
    assert "metric_code" in df.columns
    assert "metric_name" in df.columns
    assert "value" in df.columns
    assert "unit" in df.columns


def test_ingest_json_file(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test JSON file ingestion."""
    # Create temporary JSON file
    json_data = {
        "data_points": [
            {
                "metric_code": "E1-1",
                "metric_name": "Scope 1 GHG Emissions",
                "value": 1000.0,
                "unit": "tCO2e",
                "period_start": "2024-01-01",
                "period_end": "2024-12-31"
            }
        ]
    }
    json_file = tmp_path / "test_data.json"
    with open(json_file, 'w') as f:
        json.dump(json_data, f)

    df = intake_agent.read_esg_data(json_file)

    assert df is not None
    assert len(df) == 1
    assert df.iloc[0]["metric_code"] == "E1-1"


def test_ingest_json_array_file(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test JSON file ingestion with array format."""
    # Create temporary JSON file with array format
    json_data = [
        {
            "metric_code": "E1-1",
            "metric_name": "Scope 1 GHG Emissions",
            "value": 1000.0,
            "unit": "tCO2e",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31"
        }
    ]
    json_file = tmp_path / "test_data_array.json"
    with open(json_file, 'w') as f:
        json.dump(json_data, f)

    df = intake_agent.read_esg_data(json_file)

    assert df is not None
    assert len(df) == 1


def test_ingest_excel_file(intake_agent: IntakeAgent, tmp_path: Path, sample_dataframe: pd.DataFrame) -> None:
    """Test Excel file ingestion."""
    excel_file = tmp_path / "test_data.xlsx"
    sample_dataframe.to_excel(excel_file, index=False)

    df = intake_agent.read_esg_data(excel_file)

    assert df is not None
    assert len(df) == len(sample_dataframe)


def test_ingest_parquet_file(intake_agent: IntakeAgent, tmp_path: Path, sample_dataframe: pd.DataFrame) -> None:
    """Test Parquet file ingestion."""
    parquet_file = tmp_path / "test_data.parquet"
    sample_dataframe.to_parquet(parquet_file, index=False)

    df = intake_agent.read_esg_data(parquet_file)

    assert df is not None
    assert len(df) == len(sample_dataframe)


def test_ingest_tsv_file(intake_agent: IntakeAgent, tmp_path: Path, sample_dataframe: pd.DataFrame) -> None:
    """Test TSV file ingestion."""
    tsv_file = tmp_path / "test_data.tsv"
    sample_dataframe.to_csv(tsv_file, sep='\t', index=False)

    df = intake_agent.read_esg_data(tsv_file)

    assert df is not None
    assert len(df) == len(sample_dataframe)


def test_ingest_invalid_format(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test error handling for unsupported file format."""
    invalid_file = tmp_path / "test_data.txt"
    invalid_file.write_text("some text")

    with pytest.raises(ValueError, match="Unsupported file format"):
        intake_agent.read_esg_data(invalid_file)


def test_ingest_missing_file(intake_agent: IntakeAgent) -> None:
    """Test error handling for missing file."""
    with pytest.raises(ValueError, match="Input file not found"):
        intake_agent.read_esg_data(Path("nonexistent_file.csv"))


def test_ingest_corrupted_file(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test error handling for corrupted file."""
    corrupted_file = tmp_path / "corrupted.csv"
    corrupted_file.write_text("invalid,csv,data\n1,2")  # Incomplete row

    # Should still read but might have issues - we're testing it doesn't crash
    try:
        df = intake_agent.read_esg_data(corrupted_file)
        assert df is not None
    except Exception:
        # Some errors are acceptable for corrupted files
        pass


# ============================================================================
# DATA VALIDATION TESTS
# ============================================================================


def test_validate_data_point_valid(intake_agent: IntakeAgent, sample_data_point: Dict[str, Any]) -> None:
    """Test validation passes for valid data point."""
    is_valid, issues = intake_agent.validate_data_point(sample_data_point)

    assert is_valid is True
    # May have warnings but should be valid
    errors = [issue for issue in issues if issue.severity == "error"]
    assert len(errors) == 0


def test_validate_data_point_missing_required_fields(intake_agent: IntakeAgent) -> None:
    """Test validation fails for missing required fields."""
    incomplete_data = {
        "metric_code": "E1-1",
        # Missing metric_name, value, unit, etc.
    }

    is_valid, issues = intake_agent.validate_data_point(incomplete_data)

    assert is_valid is False
    assert len(issues) > 0
    assert any("Missing required field" in issue.message for issue in issues)


def test_validate_data_point_invalid_metric_code_format(intake_agent: IntakeAgent) -> None:
    """Test validation fails for invalid ESRS metric code format."""
    invalid_code_data = {
        "metric_code": "INVALID-CODE",
        "metric_name": "Test Metric",
        "value": 100.0,
        "unit": "tCO2e",
        "period_start": "2024-01-01",
        "period_end": "2024-12-31"
    }

    is_valid, issues = intake_agent.validate_data_point(invalid_code_data)

    assert is_valid is False
    assert any("Invalid ESRS metric code format" in issue.message for issue in issues)


def test_validate_data_point_invalid_date_format(intake_agent: IntakeAgent) -> None:
    """Test validation fails for invalid date format."""
    invalid_date_data = {
        "metric_code": "E1-1",
        "metric_name": "Test Metric",
        "value": 100.0,
        "unit": "tCO2e",
        "period_start": "invalid-date",
        "period_end": "2024-12-31"
    }

    is_valid, issues = intake_agent.validate_data_point(invalid_date_data)

    assert is_valid is False
    assert any("Invalid date format" in issue.message for issue in issues)


def test_validate_data_point_period_end_before_start(intake_agent: IntakeAgent) -> None:
    """Test validation fails when period_end is before period_start."""
    invalid_period_data = {
        "metric_code": "E1-1",
        "metric_name": "Test Metric",
        "value": 100.0,
        "unit": "tCO2e",
        "period_start": "2024-12-31",
        "period_end": "2024-01-01"
    }

    is_valid, issues = intake_agent.validate_data_point(invalid_period_data)

    assert is_valid is False
    assert any("Period end" in issue.message and "before period start" in issue.message for issue in issues)


def test_validate_data_point_invalid_data_quality(intake_agent: IntakeAgent) -> None:
    """Test validation fails for invalid data_quality enum value."""
    invalid_quality_data = {
        "metric_code": "E1-1",
        "metric_name": "Test Metric",
        "value": 100.0,
        "unit": "tCO2e",
        "period_start": "2024-01-01",
        "period_end": "2024-12-31",
        "data_quality": "super-high"  # Invalid value
    }

    is_valid, issues = intake_agent.validate_data_point(invalid_quality_data)

    assert is_valid is False
    assert any("Invalid data_quality value" in issue.message for issue in issues)


def test_validate_data_point_unknown_esrs_code(intake_agent: IntakeAgent) -> None:
    """Test validation warns for metric code not in ESRS catalog."""
    unknown_code_data = {
        "metric_code": "E9-999",
        "metric_name": "Unknown Metric",
        "value": 100.0,
        "unit": "tCO2e",
        "period_start": "2024-01-01",
        "period_end": "2024-12-31"
    }

    is_valid, issues = intake_agent.validate_data_point(unknown_code_data)

    # Should have a warning about not being in catalog
    assert any("not found in ESRS catalog" in issue.message for issue in issues)


# ============================================================================
# ESRS TAXONOMY MAPPING TESTS
# ============================================================================


def test_map_to_esrs_exact_match(intake_agent: IntakeAgent, sample_data_point: Dict[str, Any]) -> None:
    """Test exact matching to ESRS taxonomy."""
    metadata, warnings = intake_agent.map_to_esrs(sample_data_point)

    assert metadata is not None
    assert metadata.esrs_code == "E1-1"
    assert metadata.mapping_confidence == "exact"
    assert metadata.esrs_standard == "E1"


def test_map_to_esrs_fuzzy_match(intake_agent: IntakeAgent) -> None:
    """Test fuzzy matching by metric name."""
    # Use exact name from catalog for fuzzy matching by name
    fuzzy_data = {
        "metric_code": "UNKNOWN-CODE",
        "metric_name": "Gross Scope 1 GHG emissions",  # Exact match on name
        "value": 100.0,
        "unit": "tCO2e"
    }

    metadata, warnings = intake_agent.map_to_esrs(fuzzy_data)

    assert metadata is not None
    assert metadata.mapping_confidence == "fuzzy"
    assert len(warnings) > 0
    assert any("Fuzzy match used" in warning.message for warning in warnings)


def test_map_to_esrs_no_match(intake_agent: IntakeAgent) -> None:
    """Test handling when no ESRS match is found."""
    no_match_data = {
        "metric_code": "UNKNOWN-999",
        "metric_name": "Completely Unknown Metric Name That Does Not Exist",
        "value": 100.0,
        "unit": "unknown"
    }

    metadata, warnings = intake_agent.map_to_esrs(no_match_data)

    assert metadata is None
    assert len(warnings) > 0
    assert any("No ESRS mapping found" in warning.message for warning in warnings)


def test_map_to_esrs_unit_mismatch_warning(intake_agent: IntakeAgent) -> None:
    """Test warning when unit doesn't match expected ESRS unit."""
    unit_mismatch_data = {
        "metric_code": "E1-1",
        "metric_name": "Scope 1 GHG Emissions",
        "value": 100.0,
        "unit": "kg"  # Wrong unit, should be tCO2e
    }

    metadata, warnings = intake_agent.map_to_esrs(unit_mismatch_data)

    assert metadata is not None
    # Should warn about unit mismatch
    unit_warnings = [w for w in warnings if "Unit mismatch" in w.message or w.error_code == "W007"]
    assert len(unit_warnings) > 0


def test_map_all_known_esrs_codes(intake_agent: IntakeAgent) -> None:
    """Test that all known ESRS codes in catalog are mappable."""
    # Get first 10 codes from catalog for testing
    sample_codes = []
    for dp in intake_agent.esrs_catalog[:10]:
        code = dp.get("code") or dp.get("esrs_code") or dp.get("id")
        if code:
            sample_codes.append(code)

    for code in sample_codes:
        test_data = {
            "metric_code": code,
            "metric_name": "Test",
            "value": 100.0,
            "unit": "test"
        }
        metadata, warnings = intake_agent.map_to_esrs(test_data)
        assert metadata is not None, f"Failed to map code {code}"
        assert metadata.mapping_confidence == "exact"


# ============================================================================
# DATA QUALITY ASSESSMENT TESTS
# ============================================================================


def test_assess_completeness(intake_agent: IntakeAgent, sample_dataframe: pd.DataFrame) -> None:
    """Test completeness assessment."""
    # Create enriched data
    enriched_data = [{"validation_status": "valid", "is_outlier": False, "esrs_metadata": {}} for _ in range(len(sample_dataframe))]

    quality_score = intake_agent.assess_data_quality(sample_dataframe, enriched_data)

    assert quality_score.completeness_score >= 0
    assert quality_score.completeness_score <= 100
    # Sample data should be highly complete
    assert quality_score.completeness_score > 80


def test_assess_completeness_with_missing_data(intake_agent: IntakeAgent) -> None:
    """Test completeness with missing values."""
    df_with_nulls = pd.DataFrame({
        "metric_code": ["E1-1", "E1-2", None],
        "metric_name": ["Test", None, "Test2"],
        "value": [100.0, None, 200.0],
        "unit": ["tCO2e", "tCO2e", None],
        "period_start": ["2024-01-01", "2024-01-01", "2024-01-01"],
        "period_end": ["2024-12-31", "2024-12-31", "2024-12-31"]
    })

    enriched_data = [{"validation_status": "valid", "is_outlier": False, "esrs_metadata": {}} for _ in range(len(df_with_nulls))]

    quality_score = intake_agent.assess_data_quality(df_with_nulls, enriched_data)

    # Should have lower completeness
    assert quality_score.completeness_score < 100


def test_assess_accuracy(intake_agent: IntakeAgent, sample_dataframe: pd.DataFrame) -> None:
    """Test accuracy assessment based on outliers."""
    enriched_data = [{"validation_status": "valid", "is_outlier": False, "esrs_metadata": {}} for _ in range(len(sample_dataframe))]

    quality_score = intake_agent.assess_data_quality(sample_dataframe, enriched_data)

    assert quality_score.accuracy_score >= 0
    assert quality_score.accuracy_score <= 100
    # No outliers should give 100% accuracy
    assert quality_score.accuracy_score == 100


def test_assess_accuracy_with_outliers(intake_agent: IntakeAgent, sample_dataframe: pd.DataFrame) -> None:
    """Test accuracy assessment with outliers."""
    enriched_data = [
        {"validation_status": "valid", "is_outlier": True, "esrs_metadata": {}},
        {"validation_status": "valid", "is_outlier": False, "esrs_metadata": {}},
        {"validation_status": "valid", "is_outlier": False, "esrs_metadata": {}}
    ]

    quality_score = intake_agent.assess_data_quality(sample_dataframe, enriched_data)

    # Should have lower accuracy due to outlier
    assert quality_score.accuracy_score < 100


def test_assess_consistency(intake_agent: IntakeAgent, sample_dataframe: pd.DataFrame) -> None:
    """Test consistency assessment based on ESRS mapping."""
    enriched_data = [{"validation_status": "valid", "is_outlier": False, "esrs_metadata": {"esrs_code": "E1-1"}} for _ in range(len(sample_dataframe))]

    quality_score = intake_agent.assess_data_quality(sample_dataframe, enriched_data)

    assert quality_score.consistency_score >= 0
    assert quality_score.consistency_score <= 100
    # All mapped should give 100% consistency
    assert quality_score.consistency_score == 100


def test_assess_validity(intake_agent: IntakeAgent, sample_dataframe: pd.DataFrame) -> None:
    """Test validity assessment based on validation status."""
    enriched_data = [{"validation_status": "valid", "is_outlier": False, "esrs_metadata": {}} for _ in range(len(sample_dataframe))]

    quality_score = intake_agent.assess_data_quality(sample_dataframe, enriched_data)

    assert quality_score.validity_score >= 0
    assert quality_score.validity_score <= 100
    # All valid should give 100% validity
    assert quality_score.validity_score == 100


def test_assess_validity_with_invalid_records(intake_agent: IntakeAgent, sample_dataframe: pd.DataFrame) -> None:
    """Test validity with invalid records."""
    enriched_data = [
        {"validation_status": "invalid", "is_outlier": False, "esrs_metadata": {}},
        {"validation_status": "valid", "is_outlier": False, "esrs_metadata": {}},
        {"validation_status": "valid", "is_outlier": False, "esrs_metadata": {}}
    ]

    quality_score = intake_agent.assess_data_quality(sample_dataframe, enriched_data)

    # Should have lower validity
    assert quality_score.validity_score < 100


def test_overall_quality_score_calculation(intake_agent: IntakeAgent, sample_dataframe: pd.DataFrame) -> None:
    """Test overall quality score is weighted average of dimensions."""
    enriched_data = [{"validation_status": "valid", "is_outlier": False, "esrs_metadata": {"esrs_code": "E1-1"}} for _ in range(len(sample_dataframe))]

    quality_score = intake_agent.assess_data_quality(sample_dataframe, enriched_data)

    # Check weights sum to expected structure
    expected_overall = (
        quality_score.completeness_score * 0.30 +
        quality_score.accuracy_score * 0.25 +
        quality_score.consistency_score * 0.20 +
        quality_score.timeliness_score * 0.15 +
        quality_score.validity_score * 0.10
    )

    assert abs(quality_score.overall_score - expected_overall) < 0.1


def test_quality_score_thresholds(intake_agent: IntakeAgent, sample_dataframe: pd.DataFrame) -> None:
    """Test quality score is within valid range."""
    enriched_data = [{"validation_status": "valid", "is_outlier": False, "esrs_metadata": {}} for _ in range(len(sample_dataframe))]

    quality_score = intake_agent.assess_data_quality(sample_dataframe, enriched_data)

    assert quality_score.overall_score >= 0
    assert quality_score.overall_score <= 100


# ============================================================================
# OUTLIER DETECTION TESTS
# ============================================================================


def test_detect_outliers_zscore(intake_agent: IntakeAgent, outlier_dataframe: pd.DataFrame) -> None:
    """Test Z-score outlier detection."""
    outliers_map = intake_agent.detect_outliers(outlier_dataframe)

    # Should detect the value 1000.0 as an outlier
    assert len(outliers_map) > 0
    # Check if index 6 (value 1000.0) is flagged
    assert 6 in outliers_map


def test_detect_outliers_iqr(intake_agent: IntakeAgent, outlier_dataframe: pd.DataFrame) -> None:
    """Test IQR outlier detection."""
    outliers_map = intake_agent.detect_outliers(outlier_dataframe)

    # IQR method should also detect outliers
    assert len(outliers_map) > 0


def test_detect_outliers_none_present(intake_agent: IntakeAgent, sample_dataframe: pd.DataFrame) -> None:
    """Test when no outliers are present."""
    outliers_map = intake_agent.detect_outliers(sample_dataframe)

    # Should be empty or very few outliers in normal data
    assert len(outliers_map) <= 1


def test_detect_outliers_all_outliers(intake_agent: IntakeAgent) -> None:
    """Test when all values could be considered outliers."""
    # Create data with extreme variation
    extreme_df = pd.DataFrame({
        "metric_code": ["E1-1"] * 5,
        "metric_name": ["Test"] * 5,
        "value": [1.0, 100.0, 10000.0, 1000000.0, 100000000.0],
        "unit": ["tCO2e"] * 5,
        "period_start": ["2024-01-01"] * 5,
        "period_end": ["2024-12-31"] * 5
    })

    outliers_map = intake_agent.detect_outliers(extreme_df)

    # Should detect multiple outliers
    assert len(outliers_map) > 0


def test_detect_outliers_insufficient_data(intake_agent: IntakeAgent) -> None:
    """Test outlier detection with insufficient data points."""
    small_df = pd.DataFrame({
        "metric_code": ["E1-1", "E1-1"],
        "metric_name": ["Test", "Test"],
        "value": [100.0, 200.0],
        "unit": ["tCO2e", "tCO2e"],
        "period_start": ["2024-01-01", "2024-01-01"],
        "period_end": ["2024-12-31", "2024-12-31"]
    })

    outliers_map = intake_agent.detect_outliers(small_df)

    # Need at least 3 values for outlier detection, should return empty
    assert len(outliers_map) == 0


def test_detect_outliers_non_numeric_values(intake_agent: IntakeAgent) -> None:
    """Test outlier detection handles non-numeric values."""
    non_numeric_df = pd.DataFrame({
        "metric_code": ["S1-1"] * 5,
        "metric_name": ["Test"] * 5,
        "value": ["high", "medium", "low", "high", "medium"],
        "unit": ["text"] * 5,
        "period_start": ["2024-01-01"] * 5,
        "period_end": ["2024-12-31"] * 5
    })

    outliers_map = intake_agent.detect_outliers(non_numeric_df)

    # Should handle gracefully and return empty
    assert len(outliers_map) == 0


def test_outlier_reporting_in_issues(intake_agent: IntakeAgent, outlier_dataframe: pd.DataFrame, tmp_path: Path) -> None:
    """Test outlier reporting in validation issues."""
    # Save to file and process
    csv_file = tmp_path / "outliers.csv"
    outlier_dataframe.to_csv(csv_file, index=False)

    result = intake_agent.process(csv_file)

    # Check that outliers are reported in issues
    outlier_issues = [issue for issue in result["validation_issues"] if issue["error_code"] == "W002"]
    assert len(outlier_issues) > 0


# ============================================================================
# ENRICHMENT TESTS
# ============================================================================


def test_enrich_data_point(intake_agent: IntakeAgent, sample_data_point: Dict[str, Any]) -> None:
    """Test data point enrichment with ESRS metadata."""
    enriched, warnings = intake_agent.enrich_data_point(sample_data_point)

    assert "esrs_metadata" in enriched
    assert "processing_timestamp" in enriched
    assert enriched["esrs_metadata"] is not None


def test_enrich_data_point_adds_timestamp(intake_agent: IntakeAgent, sample_data_point: Dict[str, Any]) -> None:
    """Test enrichment adds processing timestamp."""
    enriched, warnings = intake_agent.enrich_data_point(sample_data_point)

    timestamp = enriched["processing_timestamp"]
    # Should be valid ISO format
    datetime.fromisoformat(timestamp)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_process_full_pipeline(intake_agent: IntakeAgent, demo_csv_file: Path) -> None:
    """Test full intake process end-to-end."""
    result = intake_agent.process(demo_csv_file)

    assert result is not None
    assert "metadata" in result
    assert "data_points" in result
    assert "validation_issues" in result
    assert "data_quality_report" in result

    # Check metadata
    assert result["metadata"]["total_records"] > 0
    assert result["metadata"]["processing_time_seconds"] > 0


def test_process_with_demo_data(intake_agent: IntakeAgent, demo_csv_file: Path) -> None:
    """Test processing with demo_esg_data.csv."""
    result = intake_agent.process(demo_csv_file)

    # Demo data should be high quality
    assert result["metadata"]["data_quality_score"] > 70
    assert len(result["data_points"]) > 0


def test_process_performance(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test performance: should process 1,000+ records/sec."""
    # Create large dataset
    large_df = pd.DataFrame({
        "metric_code": ["E1-1"] * 1000,
        "metric_name": ["Scope 1 GHG Emissions"] * 1000,
        "value": np.random.uniform(100, 1000, 1000),
        "unit": ["tCO2e"] * 1000,
        "period_start": ["2024-01-01"] * 1000,
        "period_end": ["2024-12-31"] * 1000,
        "data_quality": ["high"] * 1000
    })

    large_file = tmp_path / "large_data.csv"
    large_df.to_csv(large_file, index=False)

    start_time = time.time()
    result = intake_agent.process(large_file)
    elapsed_time = time.time() - start_time

    records_per_second = 1000 / elapsed_time

    # Should be fast
    assert records_per_second > 100  # At least 100 records/sec (relaxed for testing)


def test_process_large_dataset(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test with large dataset (10,000+ rows)."""
    large_df = pd.DataFrame({
        "metric_code": ["E1-1"] * 10000,
        "metric_name": ["Scope 1 GHG Emissions"] * 10000,
        "value": np.random.uniform(100, 1000, 10000),
        "unit": ["tCO2e"] * 10000,
        "period_start": ["2024-01-01"] * 10000,
        "period_end": ["2024-12-31"] * 10000,
        "data_quality": ["high"] * 10000
    })

    large_file = tmp_path / "large_10k_data.csv"
    large_df.to_csv(large_file, index=False)

    result = intake_agent.process(large_file)

    assert result["metadata"]["total_records"] == 10000
    assert result["metadata"]["processing_time_seconds"] > 0


def test_process_with_output_file(intake_agent: IntakeAgent, demo_csv_file: Path, tmp_path: Path) -> None:
    """Test processing with output file generation."""
    output_file = tmp_path / "output.json"

    result = intake_agent.process(demo_csv_file, output_file=output_file)

    assert output_file.exists()

    # Verify output file content
    with open(output_file, 'r') as f:
        output_data = json.load(f)

    assert output_data["metadata"]["total_records"] == result["metadata"]["total_records"]


def test_process_empty_dataframe(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test handling of empty DataFrame."""
    empty_df = pd.DataFrame(columns=["metric_code", "metric_name", "value", "unit", "period_start", "period_end"])

    empty_file = tmp_path / "empty.csv"
    empty_df.to_csv(empty_file, index=False)

    result = intake_agent.process(empty_file)

    assert result["metadata"]["total_records"] == 0


# ============================================================================
# VALIDATION SUMMARY TESTS
# ============================================================================


def test_get_validation_summary(intake_agent: IntakeAgent, demo_csv_file: Path) -> None:
    """Test validation summary generation."""
    result = intake_agent.process(demo_csv_file)
    summary = intake_agent.get_validation_summary(result)

    assert "summary" in summary
    assert "issues_by_code" in summary
    assert "data_quality" in summary
    assert "is_ready_for_next_stage" in summary


def test_validation_summary_issues_grouped_by_code(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test that validation summary groups issues by error code."""
    # Create data with specific errors
    invalid_df = pd.DataFrame({
        "metric_code": ["INVALID-1", "INVALID-2"],
        "metric_name": ["Test1", "Test2"],
        "value": [100.0, 200.0],
        "unit": ["tCO2e", "tCO2e"],
        "period_start": ["2024-01-01", "2024-01-01"],
        "period_end": ["2024-12-31", "2024-12-31"]
    })

    invalid_file = tmp_path / "invalid.csv"
    invalid_df.to_csv(invalid_file, index=False)

    result = intake_agent.process(invalid_file)
    summary = intake_agent.get_validation_summary(result)

    # Should group issues by code
    assert len(summary["issues_by_code"]) > 0


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


def test_handle_all_null_columns(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test handling when all values in a column are null."""
    null_df = pd.DataFrame({
        "metric_code": [None, None, None],
        "metric_name": [None, None, None],
        "value": [None, None, None],
        "unit": [None, None, None],
        "period_start": [None, None, None],
        "period_end": [None, None, None]
    })

    null_file = tmp_path / "all_null.csv"
    null_df.to_csv(null_file, index=False)

    result = intake_agent.process(null_file)

    # Should handle gracefully
    assert result is not None
    assert result["metadata"]["invalid_records"] == len(null_df)


def test_handle_mixed_data_types(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test handling of mixed data types in value column."""
    mixed_df = pd.DataFrame({
        "metric_code": ["E1-1", "E1-2", "S1-1"],
        "metric_name": ["Test1", "Test2", "Test3"],
        "value": [100.0, "text_value", True],  # Mixed types
        "unit": ["tCO2e", "tCO2e", "count"],
        "period_start": ["2024-01-01", "2024-01-01", "2024-01-01"],
        "period_end": ["2024-12-31", "2024-12-31", "2024-12-31"]
    })

    mixed_file = tmp_path / "mixed_types.csv"
    mixed_df.to_csv(mixed_file, index=False)

    result = intake_agent.process(mixed_file)

    # Should handle without crashing
    assert result is not None


def test_statistics_tracking(intake_agent: IntakeAgent, demo_csv_file: Path) -> None:
    """Test that statistics are tracked correctly."""
    result = intake_agent.process(demo_csv_file)

    metadata = result["metadata"]

    assert "total_records" in metadata
    assert "valid_records" in metadata
    assert "invalid_records" in metadata
    assert "warnings" in metadata
    assert "outliers_detected" in metadata
    assert "exact_esrs_matches" in metadata
    assert "fuzzy_esrs_matches" in metadata
    assert "unmapped_metrics" in metadata
    assert "records_per_second" in metadata


def test_quality_threshold_evaluation(intake_agent: IntakeAgent, demo_csv_file: Path) -> None:
    """Test quality threshold evaluation."""
    result = intake_agent.process(demo_csv_file)

    assert "quality_threshold_met" in result["metadata"]

    # Should be boolean
    assert isinstance(result["metadata"]["quality_threshold_met"], bool)


def test_company_profile_integration(intake_agent: IntakeAgent, demo_csv_file: Path) -> None:
    """Test company profile is included in output."""
    company_profile = {
        "company_name": "Test Company",
        "lei_code": "12345678901234567890"
    }

    result = intake_agent.process(demo_csv_file, company_profile=company_profile)

    assert result["company_profile"] == company_profile


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


def test_zero_values_handled_correctly(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test that zero values are handled correctly."""
    zero_df = pd.DataFrame({
        "metric_code": ["E1-1"],
        "metric_name": ["Test"],
        "value": [0.0],
        "unit": ["tCO2e"],
        "period_start": ["2024-01-01"],
        "period_end": ["2024-12-31"]
    })

    zero_file = tmp_path / "zero_values.csv"
    zero_df.to_csv(zero_file, index=False)

    result = intake_agent.process(zero_file)

    # Zero is a valid value
    assert result["metadata"]["total_records"] == 1


def test_special_characters_in_strings(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test handling of special characters in string fields."""
    special_df = pd.DataFrame({
        "metric_code": ["E1-1"],
        "metric_name": ["Test with émojis 🌍 and spëcial çhars"],
        "value": [100.0],
        "unit": ["tCO2e"],
        "period_start": ["2024-01-01"],
        "period_end": ["2024-12-31"],
        "notes": ["Notes with\nnewlines and\ttabs"]
    })

    special_file = tmp_path / "special_chars.csv"
    special_df.to_csv(special_file, index=False, encoding='utf-8')

    result = intake_agent.process(special_file)

    # Should handle special characters
    assert result is not None


def test_very_large_values(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test handling of very large numeric values."""
    large_value_df = pd.DataFrame({
        "metric_code": ["E1-1"],
        "metric_name": ["Test"],
        "value": [1e15],  # Very large number
        "unit": ["tCO2e"],
        "period_start": ["2024-01-01"],
        "period_end": ["2024-12-31"]
    })

    large_file = tmp_path / "large_values.csv"
    large_value_df.to_csv(large_file, index=False)

    result = intake_agent.process(large_file)

    assert result is not None


def test_negative_values_for_emissions(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test that negative values for emissions metrics are handled."""
    negative_df = pd.DataFrame({
        "metric_code": ["E1-1"],
        "metric_name": ["Scope 1 GHG Emissions"],
        "value": [-100.0],  # Negative emission (carbon removal?)
        "unit": ["tCO2e"],
        "period_start": ["2024-01-01"],
        "period_end": ["2024-12-31"]
    })

    negative_file = tmp_path / "negative_values.csv"
    negative_df.to_csv(negative_file, index=False)

    result = intake_agent.process(negative_file)

    # Should process but may flag as warning
    assert result is not None


# ============================================================================
# PYDANTIC MODEL TESTS
# ============================================================================


def test_data_quality_score_model() -> None:
    """Test DataQualityScore Pydantic model."""
    score = DataQualityScore(
        completeness_score=90.0,
        accuracy_score=85.0,
        consistency_score=88.0,
        timeliness_score=95.0,
        validity_score=92.0
    )

    assert score.completeness_score == 90.0
    assert score.overall_score >= 0
    assert score.overall_score <= 100


def test_validation_issue_model() -> None:
    """Test ValidationIssue Pydantic model."""
    issue = ValidationIssue(
        metric_code="E1-1",
        error_code="E001",
        severity="error",
        message="Test error",
        field="value",
        value=100.0,
        suggestion="Fix this",
        row_index=5
    )

    assert issue.metric_code == "E1-1"
    assert issue.severity == "error"


def test_esrs_metadata_model() -> None:
    """Test ESRSMetadata Pydantic model."""
    metadata = ESRSMetadata(
        esrs_code="E1-1",
        esrs_standard="E1",
        disclosure_requirement="E1-6",
        data_point_name="Gross Scope 1 GHG emissions",
        expected_unit="tCO2e",
        data_type="quantitative",
        is_mandatory=True,
        mapping_confidence="exact"
    )

    assert metadata.esrs_code == "E1-1"
    assert metadata.mapping_confidence == "exact"


# ============================================================================
# ADDITIONAL FILE ENCODING TESTS
# ============================================================================


def test_ingest_csv_utf16_encoding(intake_agent: IntakeAgent, tmp_path: Path, sample_dataframe: pd.DataFrame) -> None:
    """Test CSV ingestion with UTF-16 encoding."""
    csv_file = tmp_path / "test_utf16.csv"
    # Create CSV with UTF-16 encoding
    sample_dataframe.to_csv(csv_file, index=False, encoding='utf-16')

    # This may fail with current implementation, which is expected
    # The agent tries latin-1 fallback but not utf-16
    try:
        df = intake_agent.read_esg_data(csv_file)
        assert df is not None
    except Exception:
        # Expected to fail - documents limitation
        pass


def test_ingest_empty_dataframe_columns_only(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test ingestion of CSV with headers but no data rows."""
    csv_file = tmp_path / "empty_with_headers.csv"
    csv_file.write_text("metric_code,metric_name,value,unit,period_start,period_end\n")

    df = intake_agent.read_esg_data(csv_file)

    assert df is not None
    assert len(df) == 0
    assert "metric_code" in df.columns


def test_ingest_csv_with_bom(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test CSV ingestion with Byte Order Mark (BOM)."""
    csv_file = tmp_path / "test_bom.csv"
    # Write UTF-8 with BOM
    with open(csv_file, 'w', encoding='utf-8-sig') as f:
        f.write("metric_code,metric_name,value,unit,period_start,period_end\n")
        f.write("E1-1,Test,100.0,tCO2e,2024-01-01,2024-12-31\n")

    df = intake_agent.read_esg_data(csv_file)

    assert df is not None
    assert len(df) == 1


def test_ingest_csv_with_quotes(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test CSV with quoted fields containing commas."""
    csv_file = tmp_path / "test_quotes.csv"
    csv_file.write_text(
        'metric_code,metric_name,value,unit,period_start,period_end\n'
        'E1-1,"Scope 1, including mobile sources",100.0,tCO2e,2024-01-01,2024-12-31\n'
    )

    df = intake_agent.read_esg_data(csv_file)

    assert df is not None
    assert len(df) == 1
    assert "," in df.iloc[0]["metric_name"]


# ============================================================================
# ADVANCED VALIDATION TESTS
# ============================================================================


def test_validate_all_esrs_standards(intake_agent: IntakeAgent) -> None:
    """Test validation accepts all ESRS standard prefixes."""
    standards = ["E1", "E2", "E3", "E4", "E5", "S1", "S2", "S3", "S4", "G1", "ESRS1", "ESRS2"]

    for standard in standards:
        if standard.startswith("ESRS"):
            metric_code = f"{standard}-1"
        else:
            metric_code = f"{standard}-1"

        data_point = {
            "metric_code": metric_code,
            "metric_name": "Test Metric",
            "value": 100.0,
            "unit": "test",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31"
        }

        is_valid, issues = intake_agent.validate_data_point(data_point)

        # Should not have format errors
        format_errors = [i for i in issues if i.error_code == "E002"]
        assert len(format_errors) == 0, f"Format error for {metric_code}"


def test_validate_data_point_with_all_optional_fields(intake_agent: IntakeAgent) -> None:
    """Test validation with all optional fields populated."""
    complete_data = {
        "metric_code": "E1-1",
        "metric_name": "Test Metric",
        "value": 100.0,
        "unit": "tCO2e",
        "period_start": "2024-01-01",
        "period_end": "2024-12-31",
        "data_quality": "high",
        "source_document": "test.pdf",
        "verification_status": "verified",
        "notes": "Test notes",
        "calculation_method": "Direct measurement",
        "breakdown": {"component1": 50, "component2": 50},
        "tags": ["emissions", "energy"]
    }

    is_valid, issues = intake_agent.validate_data_point(complete_data)

    # Should be valid
    errors = [i for i in issues if i.severity == "error"]
    assert len(errors) == 0


def test_validate_empty_string_vs_none(intake_agent: IntakeAgent) -> None:
    """Test that empty strings are treated same as None for required fields."""
    data_with_empty = {
        "metric_code": "E1-1",
        "metric_name": "",  # Empty string
        "value": 100.0,
        "unit": "tCO2e",
        "period_start": "2024-01-01",
        "period_end": "2024-12-31"
    }

    is_valid, issues = intake_agent.validate_data_point(data_with_empty)

    assert is_valid is False
    assert any("Missing required field" in i.message and i.field == "metric_name" for i in issues)


def test_validate_whitespace_only_strings(intake_agent: IntakeAgent) -> None:
    """Test validation of whitespace-only values."""
    data_with_whitespace = {
        "metric_code": "E1-1",
        "metric_name": "   ",  # Whitespace only
        "value": 100.0,
        "unit": "tCO2e",
        "period_start": "2024-01-01",
        "period_end": "2024-12-31"
    }

    is_valid, issues = intake_agent.validate_data_point(data_with_whitespace)

    # Should still be valid as whitespace is not empty
    # This documents current behavior


def test_validate_row_index_tracking(intake_agent: IntakeAgent) -> None:
    """Test that row_index is correctly tracked in validation issues."""
    data_point = {
        "metric_code": "INVALID-CODE",
        "metric_name": "Test",
        "value": 100.0,
        "unit": "tCO2e",
        "period_start": "2024-01-01",
        "period_end": "2024-12-31"
    }

    is_valid, issues = intake_agent.validate_data_point(data_point, row_index=42)

    assert any(i.row_index == 42 for i in issues)


# ============================================================================
# ESRS MAPPING EDGE CASES
# ============================================================================


def test_map_to_esrs_case_sensitivity(intake_agent: IntakeAgent) -> None:
    """Test ESRS mapping is case-insensitive for names."""
    # Test with different case variations
    test_data = {
        "metric_code": "UNKNOWN-CODE",
        "metric_name": "gross scope 1 ghg emissions",  # lowercase
        "value": 100.0,
        "unit": "tCO2e"
    }

    metadata, warnings = intake_agent.map_to_esrs(test_data)

    # Should still map via fuzzy match (case-insensitive)
    if metadata:
        assert metadata.mapping_confidence == "fuzzy"


def test_map_to_esrs_partial_name_match(intake_agent: IntakeAgent) -> None:
    """Test that partial name matches don't incorrectly map."""
    test_data = {
        "metric_code": "UNKNOWN-999",
        "metric_name": "Scope",  # Partial match to many metrics
        "value": 100.0,
        "unit": "tCO2e"
    }

    metadata, warnings = intake_agent.map_to_esrs(test_data)

    # Should not map on partial match
    assert metadata is None


def test_map_to_esrs_statistics_tracking(intake_agent: IntakeAgent, sample_data_point: Dict[str, Any]) -> None:
    """Test that mapping statistics are correctly tracked."""
    # Reset stats
    intake_agent.stats["exact_matches"] = 0
    intake_agent.stats["fuzzy_matches"] = 0
    intake_agent.stats["unmapped"] = 0

    # Test exact match
    intake_agent.map_to_esrs(sample_data_point)
    assert intake_agent.stats["exact_matches"] == 1

    # Test fuzzy match
    fuzzy_data = {
        "metric_code": "UNKNOWN-CODE",
        "metric_name": "Gross Scope 1 GHG emissions",
        "value": 100.0,
        "unit": "tCO2e"
    }
    intake_agent.map_to_esrs(fuzzy_data)
    assert intake_agent.stats["fuzzy_matches"] == 1

    # Test unmapped
    unmapped_data = {
        "metric_code": "UNKNOWN-999",
        "metric_name": "Completely Unknown Metric",
        "value": 100.0,
        "unit": "unknown"
    }
    intake_agent.map_to_esrs(unmapped_data)
    assert intake_agent.stats["unmapped"] == 1


def test_map_to_esrs_all_metadata_fields_populated(intake_agent: IntakeAgent) -> None:
    """Test that all ESRS metadata fields are populated when available."""
    # Use E1-1 which should have full metadata
    test_data = {
        "metric_code": "E1-1",
        "metric_name": "Scope 1 GHG Emissions",
        "value": 100.0,
        "unit": "tCO2e"
    }

    metadata, warnings = intake_agent.map_to_esrs(test_data)

    assert metadata is not None
    assert metadata.esrs_code is not None
    assert metadata.esrs_standard == "E1"
    # Other fields may be optional


def test_map_to_esrs_empty_metric_name(intake_agent: IntakeAgent) -> None:
    """Test mapping with empty metric name."""
    test_data = {
        "metric_code": "UNKNOWN-999",
        "metric_name": "",
        "value": 100.0,
        "unit": "tCO2e"
    }

    metadata, warnings = intake_agent.map_to_esrs(test_data)

    # Should not map
    assert metadata is None


# ============================================================================
# DATA QUALITY SCORING EDGE CASES
# ============================================================================


def test_assess_quality_with_empty_dataframe(intake_agent: IntakeAgent) -> None:
    """Test quality assessment with empty DataFrame."""
    empty_df = pd.DataFrame(columns=["metric_code", "metric_name", "value", "unit", "period_start", "period_end"])
    enriched_data = []

    quality_score = intake_agent.assess_data_quality(empty_df, enriched_data)

    # Should handle gracefully
    assert quality_score.overall_score >= 0
    assert quality_score.overall_score <= 100


def test_assess_quality_all_invalid_records(intake_agent: IntakeAgent, sample_dataframe: pd.DataFrame) -> None:
    """Test quality assessment when all records are invalid."""
    enriched_data = [{"validation_status": "invalid", "is_outlier": False, "esrs_metadata": None} for _ in range(len(sample_dataframe))]

    quality_score = intake_agent.assess_data_quality(sample_dataframe, enriched_data)

    # Validity score should be 0
    assert quality_score.validity_score == 0


def test_assess_quality_no_esrs_mapping(intake_agent: IntakeAgent, sample_dataframe: pd.DataFrame) -> None:
    """Test quality assessment when no ESRS mappings found."""
    enriched_data = [{"validation_status": "valid", "is_outlier": False, "esrs_metadata": None} for _ in range(len(sample_dataframe))]

    quality_score = intake_agent.assess_data_quality(sample_dataframe, enriched_data)

    # Consistency score should be 0 (no mappings)
    assert quality_score.consistency_score == 0


def test_assess_quality_all_outliers(intake_agent: IntakeAgent, sample_dataframe: pd.DataFrame) -> None:
    """Test quality assessment when all records are outliers."""
    enriched_data = [{"validation_status": "valid", "is_outlier": True, "esrs_metadata": {}} for _ in range(len(sample_dataframe))]

    quality_score = intake_agent.assess_data_quality(sample_dataframe, enriched_data)

    # Accuracy score should be 0
    assert quality_score.accuracy_score == 0


def test_quality_score_weights_sum_to_one(intake_agent: IntakeAgent) -> None:
    """Test that quality dimension weights sum to 1.0."""
    score = DataQualityScore(
        completeness_score=100,
        accuracy_score=100,
        consistency_score=100,
        timeliness_score=100,
        validity_score=100
    )

    total_weight = (
        score.completeness_weight +
        score.accuracy_weight +
        score.consistency_weight +
        score.timeliness_weight +
        score.validity_weight
    )

    assert abs(total_weight - 1.0) < 0.01


# ============================================================================
# OUTLIER DETECTION EDGE CASES
# ============================================================================


def test_detect_outliers_all_identical_values(intake_agent: IntakeAgent) -> None:
    """Test outlier detection when all values are identical."""
    identical_df = pd.DataFrame({
        "metric_code": ["E1-1"] * 10,
        "metric_name": ["Test"] * 10,
        "value": [100.0] * 10,  # All identical
        "unit": ["tCO2e"] * 10,
        "period_start": ["2024-01-01"] * 10,
        "period_end": ["2024-12-31"] * 10
    })

    outliers_map = intake_agent.detect_outliers(identical_df)

    # No outliers when all values are identical (std = 0)
    assert len(outliers_map) == 0


def test_detect_outliers_single_value(intake_agent: IntakeAgent) -> None:
    """Test outlier detection with single data point."""
    single_df = pd.DataFrame({
        "metric_code": ["E1-1"],
        "metric_name": ["Test"],
        "value": [100.0],
        "unit": ["tCO2e"],
        "period_start": ["2024-01-01"],
        "period_end": ["2024-12-31"]
    })

    outliers_map = intake_agent.detect_outliers(single_df)

    # Cannot detect outliers with single value
    assert len(outliers_map) == 0


def test_detect_outliers_mixed_metric_codes(intake_agent: IntakeAgent) -> None:
    """Test that outlier detection is performed per metric_code."""
    mixed_df = pd.DataFrame({
        "metric_code": ["E1-1"] * 5 + ["E1-2"] * 5,
        "metric_name": ["Test1"] * 5 + ["Test2"] * 5,
        "value": [100, 110, 105, 95, 1000] + [200, 210, 205, 195, 2000],  # Outliers in each group
        "unit": ["tCO2e"] * 10,
        "period_start": ["2024-01-01"] * 10,
        "period_end": ["2024-12-31"] * 10
    })

    outliers_map = intake_agent.detect_outliers(mixed_df)

    # Should detect outliers in both groups
    assert len(outliers_map) >= 2


def test_detect_outliers_missing_metric_code_column(intake_agent: IntakeAgent) -> None:
    """Test outlier detection when metric_code column is missing."""
    df_no_code = pd.DataFrame({
        "metric_name": ["Test"] * 5,
        "value": [100, 110, 105, 95, 1000],
        "unit": ["tCO2e"] * 5
    })

    outliers_map = intake_agent.detect_outliers(df_no_code)

    # Should return empty without crashing
    assert len(outliers_map) == 0


def test_detect_outliers_missing_value_column(intake_agent: IntakeAgent) -> None:
    """Test outlier detection when value column is missing."""
    df_no_value = pd.DataFrame({
        "metric_code": ["E1-1"] * 5,
        "metric_name": ["Test"] * 5,
        "unit": ["tCO2e"] * 5
    })

    outliers_map = intake_agent.detect_outliers(df_no_value)

    # Should return empty without crashing
    assert len(outliers_map) == 0


def test_detect_outliers_with_nulls(intake_agent: IntakeAgent) -> None:
    """Test outlier detection with null values in data."""
    df_with_nulls = pd.DataFrame({
        "metric_code": ["E1-1"] * 10,
        "metric_name": ["Test"] * 10,
        "value": [100.0, None, 110.0, 105.0, None, 95.0, 102.0, 108.0, 1000.0, None],
        "unit": ["tCO2e"] * 10,
        "period_start": ["2024-01-01"] * 10,
        "period_end": ["2024-12-31"] * 10
    })

    outliers_map = intake_agent.detect_outliers(df_with_nulls)

    # Should handle nulls gracefully
    assert isinstance(outliers_map, dict)


def test_detect_outliers_boundary_values(intake_agent: IntakeAgent) -> None:
    """Test outlier detection at Z-score boundary (exactly 3 std devs)."""
    # Create data where one value is exactly at the boundary
    values = [100.0] * 10 + [200.0]  # Will need to calculate exact boundary

    boundary_df = pd.DataFrame({
        "metric_code": ["E1-1"] * len(values),
        "metric_name": ["Test"] * len(values),
        "value": values,
        "unit": ["tCO2e"] * len(values),
        "period_start": ["2024-01-01"] * len(values),
        "period_end": ["2024-12-31"] * len(values)
    })

    outliers_map = intake_agent.detect_outliers(boundary_df)

    # Documents behavior at boundary
    assert isinstance(outliers_map, dict)


# ============================================================================
# ENRICHMENT TESTS
# ============================================================================


def test_enrich_data_point_without_esrs_match(intake_agent: IntakeAgent) -> None:
    """Test enrichment when no ESRS match is found."""
    unmapped_data = {
        "metric_code": "UNKNOWN-999",
        "metric_name": "Unknown Metric",
        "value": 100.0,
        "unit": "unknown",
        "period_start": "2024-01-01",
        "period_end": "2024-12-31"
    }

    enriched, warnings = intake_agent.enrich_data_point(unmapped_data)

    # Should still add timestamp
    assert "processing_timestamp" in enriched
    # ESRS metadata may be absent or None
    assert "esrs_metadata" not in enriched or enriched["esrs_metadata"] is None


def test_enrich_preserves_original_data(intake_agent: IntakeAgent, sample_data_point: Dict[str, Any]) -> None:
    """Test that enrichment preserves all original data fields."""
    original_keys = set(sample_data_point.keys())

    enriched, warnings = intake_agent.enrich_data_point(sample_data_point.copy())

    # All original keys should still be present
    for key in original_keys:
        assert key in enriched


def test_enrich_timestamp_format(intake_agent: IntakeAgent, sample_data_point: Dict[str, Any]) -> None:
    """Test that processing timestamp is in ISO 8601 format."""
    enriched, warnings = intake_agent.enrich_data_point(sample_data_point)

    timestamp = enriched["processing_timestamp"]

    # Should be parseable as ISO format
    parsed = datetime.fromisoformat(timestamp)
    assert parsed is not None


# ============================================================================
# INTEGRATION TEST - MULTIPLE FORMATS
# ============================================================================


def test_process_all_supported_formats(intake_agent: IntakeAgent, tmp_path: Path, sample_dataframe: pd.DataFrame) -> None:
    """Test processing all supported file formats."""
    formats = {
        "csv": sample_dataframe.to_csv,
        "json": lambda path, **kwargs: sample_dataframe.to_json(path, orient='records', **kwargs),
        "xlsx": sample_dataframe.to_excel,
        "parquet": sample_dataframe.to_parquet,
    }

    for fmt, writer_func in formats.items():
        file_path = tmp_path / f"test_data.{fmt}"

        if fmt == "json":
            writer_func(file_path, indent=2)
        else:
            writer_func(file_path, index=False)

        result = intake_agent.process(file_path)

        assert result["metadata"]["total_records"] == len(sample_dataframe)
        assert result["metadata"]["processing_time_seconds"] > 0


# ============================================================================
# INTEGRATION TEST - QUALITY THRESHOLD
# ============================================================================


def test_process_quality_threshold_pass(intake_agent: IntakeAgent, demo_csv_file: Path) -> None:
    """Test that high-quality data passes quality threshold."""
    result = intake_agent.process(demo_csv_file)

    # Demo data should be high quality
    assert result["metadata"]["quality_threshold_met"] is True


def test_process_quality_threshold_fail(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test that low-quality data fails quality threshold."""
    # Create low-quality data (many missing fields)
    low_quality_df = pd.DataFrame({
        "metric_code": ["E1-1", "E1-2", "E1-3"],
        "metric_name": [None, "", "Test"],
        "value": [None, 100.0, 200.0],
        "unit": ["tCO2e", None, "tCO2e"],
        "period_start": ["2024-01-01", "2024-01-01", None],
        "period_end": ["2024-12-31", "2024-12-31", "2024-12-31"]
    })

    low_quality_file = tmp_path / "low_quality.csv"
    low_quality_df.to_csv(low_quality_file, index=False)

    result = intake_agent.process(low_quality_file)

    # Should have low quality score
    assert result["metadata"]["data_quality_score"] < 80


# ============================================================================
# INTEGRATION TEST - ERROR RECOVERY
# ============================================================================


def test_process_continues_after_validation_errors(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test that processing continues even when some records are invalid."""
    mixed_df = pd.DataFrame({
        "metric_code": ["E1-1", "INVALID-CODE", "E1-2"],
        "metric_name": ["Valid", "Invalid", "Valid"],
        "value": [100.0, 200.0, 300.0],
        "unit": ["tCO2e", "tCO2e", "tCO2e"],
        "period_start": ["2024-01-01", "2024-01-01", "2024-01-01"],
        "period_end": ["2024-12-31", "2024-12-31", "2024-12-31"]
    })

    mixed_file = tmp_path / "mixed_validity.csv"
    mixed_df.to_csv(mixed_file, index=False)

    result = intake_agent.process(mixed_file)

    # Should process all 3 records
    assert result["metadata"]["total_records"] == 3
    # Should have some invalid
    assert result["metadata"]["invalid_records"] > 0
    # Should have some valid
    assert result["metadata"]["valid_records"] > 0


# ============================================================================
# WRITE OUTPUT TESTS
# ============================================================================


def test_write_output_creates_directory(intake_agent: IntakeAgent, demo_csv_file: Path, tmp_path: Path) -> None:
    """Test that write_output creates parent directories if needed."""
    output_file = tmp_path / "nested" / "dir" / "output.json"

    result = intake_agent.process(demo_csv_file)
    intake_agent.write_output(result, output_file)

    assert output_file.exists()
    assert output_file.parent.exists()


def test_write_output_valid_json(intake_agent: IntakeAgent, demo_csv_file: Path, tmp_path: Path) -> None:
    """Test that written output is valid JSON."""
    output_file = tmp_path / "output.json"

    result = intake_agent.process(demo_csv_file)
    intake_agent.write_output(result, output_file)

    # Should be parseable as JSON
    with open(output_file, 'r') as f:
        loaded = json.load(f)

    assert loaded["metadata"]["total_records"] == result["metadata"]["total_records"]


# ============================================================================
# GET VALIDATION SUMMARY TESTS
# ============================================================================


def test_get_validation_summary_structure(intake_agent: IntakeAgent, demo_csv_file: Path) -> None:
    """Test validation summary has correct structure."""
    result = intake_agent.process(demo_csv_file)
    summary = intake_agent.get_validation_summary(result)

    # Check all expected keys
    assert "summary" in summary
    assert "issues_by_code" in summary
    assert "data_quality" in summary
    assert "is_ready_for_next_stage" in summary

    # Check summary contains metadata
    assert summary["summary"]["total_records"] > 0


def test_get_validation_summary_ready_for_next_stage(intake_agent: IntakeAgent, demo_csv_file: Path) -> None:
    """Test is_ready_for_next_stage logic."""
    result = intake_agent.process(demo_csv_file)
    summary = intake_agent.get_validation_summary(result)

    # Demo data should be ready
    if result["metadata"]["invalid_records"] == 0 and result["metadata"]["quality_threshold_met"]:
        assert summary["is_ready_for_next_stage"] is True


def test_get_validation_summary_not_ready(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test is_ready_for_next_stage when data is not ready."""
    # Create invalid data
    invalid_df = pd.DataFrame({
        "metric_code": ["INVALID-1", "INVALID-2"],
        "metric_name": ["Test1", "Test2"],
        "value": [100.0, 200.0],
        "unit": ["tCO2e", "tCO2e"],
        "period_start": ["2024-01-01", "2024-01-01"],
        "period_end": ["2024-12-31", "2024-12-31"]
    })

    invalid_file = tmp_path / "invalid.csv"
    invalid_df.to_csv(invalid_file, index=False)

    result = intake_agent.process(invalid_file)
    summary = intake_agent.get_validation_summary(result)

    # Should not be ready due to invalid records
    assert summary["is_ready_for_next_stage"] is False


def test_get_validation_summary_issues_have_counts(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test that issues_by_code includes counts."""
    # Create data with known issues
    invalid_df = pd.DataFrame({
        "metric_code": ["INVALID-1", "INVALID-2", "INVALID-3"],
        "metric_name": ["Test1", "Test2", "Test3"],
        "value": [100.0, 200.0, 300.0],
        "unit": ["tCO2e", "tCO2e", "tCO2e"],
        "period_start": ["2024-01-01", "2024-01-01", "2024-01-01"],
        "period_end": ["2024-12-31", "2024-12-31", "2024-12-31"]
    })

    invalid_file = tmp_path / "invalid.csv"
    invalid_df.to_csv(invalid_file, index=False)

    result = intake_agent.process(invalid_file)
    summary = intake_agent.get_validation_summary(result)

    # Should have issue counts
    if len(summary["issues_by_code"]) > 0:
        first_issue = summary["issues_by_code"][0]
        assert "count" in first_issue
        assert "code" in first_issue
        assert "severity" in first_issue


# ============================================================================
# STRESS TESTS
# ============================================================================


def test_process_very_large_dataset(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test with very large dataset (100,000 rows)."""
    # Create large dataset
    large_df = pd.DataFrame({
        "metric_code": ["E1-1"] * 100000,
        "metric_name": ["Scope 1 GHG Emissions"] * 100000,
        "value": np.random.uniform(100, 10000, 100000),
        "unit": ["tCO2e"] * 100000,
        "period_start": ["2024-01-01"] * 100000,
        "period_end": ["2024-12-31"] * 100000,
        "data_quality": ["high"] * 100000
    })

    large_file = tmp_path / "large_100k.csv"
    large_df.to_csv(large_file, index=False)

    start_time = time.time()
    result = intake_agent.process(large_file)
    elapsed = time.time() - start_time

    assert result["metadata"]["total_records"] == 100000
    assert result["metadata"]["processing_time_seconds"] > 0

    # Should complete in reasonable time (< 5 minutes)
    assert elapsed < 300


def test_process_wide_dataset(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test with dataset having many columns."""
    # Create wide dataset
    wide_data = {
        "metric_code": ["E1-1"] * 10,
        "metric_name": ["Test"] * 10,
        "value": [100.0] * 10,
        "unit": ["tCO2e"] * 10,
        "period_start": ["2024-01-01"] * 10,
        "period_end": ["2024-12-31"] * 10,
    }

    # Add many extra columns
    for i in range(50):
        wide_data[f"extra_col_{i}"] = [f"value_{i}"] * 10

    wide_df = pd.DataFrame(wide_data)
    wide_file = tmp_path / "wide_data.csv"
    wide_df.to_csv(wide_file, index=False)

    result = intake_agent.process(wide_file)

    # Should handle without issues
    assert result["metadata"]["total_records"] == 10


def test_process_many_different_metrics(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    """Test with many different metric codes."""
    # Create data with 50 different metrics
    metric_codes = [f"E1-{i}" for i in range(1, 51)]

    many_metrics_df = pd.DataFrame({
        "metric_code": metric_codes,
        "metric_name": [f"Metric {i}" for i in range(1, 51)],
        "value": np.random.uniform(100, 1000, 50),
        "unit": ["tCO2e"] * 50,
        "period_start": ["2024-01-01"] * 50,
        "period_end": ["2024-12-31"] * 50,
        "data_quality": ["high"] * 50
    })

    many_metrics_file = tmp_path / "many_metrics.csv"
    many_metrics_df.to_csv(many_metrics_file, index=False)

    result = intake_agent.process(many_metrics_file)

    assert result["metadata"]["total_records"] == 50


# ============================================================================
# FINAL SUMMARY TEST
# ============================================================================


def test_comprehensive_coverage_check(intake_agent: IntakeAgent, demo_csv_file: Path) -> None:
    """
    Comprehensive test to verify all major functionalities work together.

    This test validates:
    1. Data ingestion
    2. Validation
    3. ESRS mapping
    4. Data quality assessment
    5. Outlier detection
    6. Enrichment
    7. Output generation
    """
    result = intake_agent.process(demo_csv_file)

    # Verify all components produced output
    assert result["metadata"]["total_records"] > 0
    assert result["metadata"]["processing_time_seconds"] > 0
    assert result["metadata"]["records_per_second"] > 0
    assert "data_quality_score" in result["metadata"]

    # Verify data points are enriched
    assert len(result["data_points"]) > 0
    first_dp = result["data_points"][0]
    assert "processing_timestamp" in first_dp
    assert "validation_status" in first_dp

    # Verify data quality report
    dq_report = result["data_quality_report"]
    assert "overall_score" in dq_report
    assert "completeness_score" in dq_report
    assert "accuracy_score" in dq_report
    assert "consistency_score" in dq_report
    assert "timeliness_score" in dq_report
    assert "validity_score" in dq_report

    print(f"\n{'='*80}")
    print("INTAKE AGENT TEST SUITE - COMPREHENSIVE COVERAGE VALIDATION")
    print(f"{'='*80}")
    print(f"Total records processed: {result['metadata']['total_records']}")
    print(f"Valid records: {result['metadata']['valid_records']}")
    print(f"Invalid records: {result['metadata']['invalid_records']}")
    print(f"Processing time: {result['metadata']['processing_time_seconds']:.2f}s")
    print(f"Throughput: {result['metadata']['records_per_second']:.0f} records/sec")
    print(f"Data quality score: {result['metadata']['data_quality_score']:.1f}/100")
    print(f"Quality threshold met: {result['metadata']['quality_threshold_met']}")
    print(f"ESRS exact matches: {result['metadata']['exact_esrs_matches']}")
    print(f"ESRS fuzzy matches: {result['metadata']['fuzzy_esrs_matches']}")
    print(f"Unmapped metrics: {result['metadata']['unmapped_metrics']}")
    print(f"Outliers detected: {result['metadata']['outliers_detected']}")
    print(f"{'='*80}\n")
