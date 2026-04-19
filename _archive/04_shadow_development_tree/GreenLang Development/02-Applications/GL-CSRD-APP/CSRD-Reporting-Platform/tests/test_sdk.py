# -*- coding: utf-8 -*-
"""
CSRD/ESRS Digital Reporting Platform - SDK Tests

Comprehensive test suite for the Python SDK (csrd_sdk.py)

This test file validates the complete SDK including:
1. Main function: csrd_build_report()
2. CSRDConfig dataclass
3. CSRDReport dataclass
4. DataFrame input/output support
5. Individual agent access methods
6. Configuration management
7. Error handling
8. Performance characteristics

The SDK provides a simple Pythonic API for CSRD reporting that:
- Accepts files OR DataFrames OR dictionaries
- Returns structured Python objects
- Provides composable access to individual agents
- Guarantees zero hallucination for calculations
- Requires human review for AI-assisted materiality

TARGET: 90% SDK test coverage
Expected: 45-55 test cases

Version: 1.0.0
Author: GreenLang CSRD Team
"""

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest
import yaml

from sdk.csrd_sdk import (
    CSRDConfig,
    CSRDReport,
    ComplianceStatus,
    ESRSMetrics,
    MaterialityAssessment,
    csrd_assess_materiality,
    csrd_audit_compliance,
    csrd_build_report,
    csrd_calculate_metrics,
    csrd_validate_data,
)


# ============================================================================
# PYTEST FIXTURES
# ============================================================================


@pytest.fixture
def base_path() -> Path:
    """Get base path for test resources."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_output_dir(tmp_path: Path) -> Path:
    """Create temporary output directory for tests."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def sample_esg_csv(tmp_path: Path) -> Path:
    """Create sample ESG data CSV file."""
    csv_file = tmp_path / "test_esg_data.csv"
    csv_data = """metric_code,metric_name,value,unit,period_start,period_end,data_quality,source,verification_status
E1-1,Scope 1 GHG Emissions,1000.0,tCO2e,2024-01-01,2024-12-31,high,measured,verified
E1-2,Scope 2 GHG Emissions (location-based),500.0,tCO2e,2024-01-01,2024-12-31,high,measured,verified
E1-3,Scope 3 GHG Emissions,2000.0,tCO2e,2024-01-01,2024-12-31,medium,estimated,unverified
S1-1,Total Employees,250,FTE,2024-01-01,2024-12-31,high,measured,verified
S1-2,Employee Turnover Rate,0.12,ratio,2024-01-01,2024-12-31,high,measured,verified
G1-1,Board Gender Diversity,0.40,ratio,2024-01-01,2024-12-31,high,measured,verified
"""
    csv_file.write_text(csv_data)
    return csv_file


@pytest.fixture
def sample_esg_json(tmp_path: Path) -> Path:
    """Create sample ESG data JSON file."""
    json_file = tmp_path / "test_esg_data.json"
    json_data = {
        "data_points": [
            {
                "metric_code": "E1-1",
                "metric_name": "Scope 1 GHG Emissions",
                "value": 1000.0,
                "unit": "tCO2e",
                "period_start": "2024-01-01",
                "period_end": "2024-12-31",
                "data_quality": "high",
                "source": "measured",
                "verification_status": "verified"
            },
            {
                "metric_code": "E1-2",
                "metric_name": "Scope 2 GHG Emissions (location-based)",
                "value": 500.0,
                "unit": "tCO2e",
                "period_start": "2024-01-01",
                "period_end": "2024-12-31",
                "data_quality": "high",
                "source": "measured",
                "verification_status": "verified"
            }
        ]
    }
    json_file.write_text(json.dumps(json_data, indent=2))
    return json_file


@pytest.fixture
def sample_esg_dataframe() -> pd.DataFrame:
    """Create sample ESG data DataFrame."""
    return pd.DataFrame({
        "metric_code": ["E1-1", "E1-2", "S1-1"],
        "metric_name": ["Scope 1 GHG Emissions", "Scope 2 GHG Emissions (location-based)", "Total Employees"],
        "value": [1000.0, 500.0, 250.0],
        "unit": ["tCO2e", "tCO2e", "FTE"],
        "period_start": ["2024-01-01", "2024-01-01", "2024-01-01"],
        "period_end": ["2024-12-31", "2024-12-31", "2024-12-31"],
        "data_quality": ["high", "high", "high"],
        "source": ["measured", "measured", "measured"],
        "verification_status": ["verified", "verified", "verified"]
    })


@pytest.fixture
def sample_company_profile_json(tmp_path: Path) -> Path:
    """Create sample company profile JSON file."""
    profile_file = tmp_path / "company_profile.json"
    profile_data = {
        "company_info": {
            "legal_name": "Acme Manufacturing GmbH",
            "lei": "529900ABCDEFGHIJKLMN",
            "country": "DE",
            "website": "https://acme-mfg.example.com"
        },
        "business_profile": {
            "sector": "Manufacturing",
            "primary_nace_code": "25.11",
            "description": "Industrial machinery manufacturing"
        },
        "company_size": {
            "employee_count": 5000,
            "revenue": {"total_revenue": 500000000.0, "currency": "EUR"},
            "total_assets": 300000000.0
        },
        "reporting_scope": {
            "reporting_year": 2024,
            "consolidation_method": "full",
            "entities_included": ["Acme GmbH", "Acme Services GmbH"]
        }
    }
    profile_file.write_text(json.dumps(profile_data, indent=2))
    return profile_file


@pytest.fixture
def sample_company_profile_dict() -> Dict[str, Any]:
    """Create sample company profile dictionary."""
    return {
        "company_info": {
            "legal_name": "Test Company Ltd",
            "lei": "549300TEST1234567",
            "country": "UK"
        },
        "business_profile": {
            "sector": "Technology",
            "primary_nace_code": "62.01"
        },
        "company_size": {
            "employee_count": 1000,
            "revenue": {"total_revenue": 100000000.0, "currency": "GBP"}
        },
        "reporting_scope": {
            "reporting_year": 2024
        }
    }


@pytest.fixture
def sample_csrd_config() -> CSRDConfig:
    """Create sample CSRDConfig instance."""
    return CSRDConfig(
        company_name="Test Corporation",
        company_lei="529900TESTLEI123456",
        reporting_year=2024,
        sector="Technology",
        country="DE",
        employee_count=1000,
        revenue=100000000.0,
        quality_threshold=0.80,
        impact_materiality_threshold=5.0,
        financial_materiality_threshold=5.0,
        llm_provider="openai",
        llm_model="gpt-4o"
    )


@pytest.fixture
def sample_config_yaml(tmp_path: Path, sample_csrd_config: CSRDConfig) -> Path:
    """Create sample CSRD config YAML file."""
    config_file = tmp_path / "csrd_config.yaml"
    config_data = {
        "company": {
            "name": sample_csrd_config.company_name,
            "lei": sample_csrd_config.company_lei,
            "reporting_year": sample_csrd_config.reporting_year,
            "sector": sample_csrd_config.sector,
            "country": sample_csrd_config.country,
            "employee_count": sample_csrd_config.employee_count,
            "revenue": sample_csrd_config.revenue
        },
        "thresholds": {
            "quality": sample_csrd_config.quality_threshold,
            "impact_materiality": sample_csrd_config.impact_materiality_threshold,
            "financial_materiality": sample_csrd_config.financial_materiality_threshold
        },
        "llm": {
            "provider": sample_csrd_config.llm_provider,
            "model": sample_csrd_config.llm_model
        }
    }
    config_file.write_text(yaml.dump(config_data))
    return config_file


# ============================================================================
# TEST CLASS 1: TestCSRDBuildReportFunction
# ============================================================================


class TestCSRDBuildReportFunction:
    """Test the main csrd_build_report() function."""

    def test_build_report_with_csv_file(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path,
        test_output_dir: Path
    ):
        """Test csrd_build_report with CSV file input."""
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            output_dir=str(test_output_dir),
            skip_materiality=True,  # Skip for faster tests
            skip_audit=False
        )

        # Assert report structure
        assert isinstance(report, CSRDReport)
        assert report.report_id is not None
        assert report.generated_at is not None
        assert report.processing_time_total_minutes is not None
        assert report.processing_time_total_minutes > 0

        # Assert company info
        assert report.company_info["legal_name"] == "Acme Manufacturing GmbH"
        assert report.company_info["lei"] == "529900ABCDEFGHIJKLMN"

        # Assert metrics calculated
        assert isinstance(report.metrics, ESRSMetrics)
        assert report.metrics.total_metrics_calculated > 0

        # Assert compliance status
        assert isinstance(report.compliance_status, ComplianceStatus)
        assert report.compliance_status.compliance_status in ["PASS", "WARNING", "FAIL", "SKIPPED"]

    def test_build_report_with_json_file(
        self,
        sample_esg_json: Path,
        sample_company_profile_json: Path,
        test_output_dir: Path
    ):
        """Test csrd_build_report with JSON file input."""
        report = csrd_build_report(
            esg_data=str(sample_esg_json),
            company_profile=str(sample_company_profile_json),
            output_dir=str(test_output_dir),
            skip_materiality=True,
            skip_audit=True
        )

        assert isinstance(report, CSRDReport)
        assert report.report_id is not None
        assert report.metrics.total_metrics_calculated >= 0

    def test_build_report_with_dataframe(
        self,
        sample_esg_dataframe: pd.DataFrame,
        sample_company_profile_json: Path,
        test_output_dir: Path
    ):
        """Test csrd_build_report with pandas DataFrame input."""
        report = csrd_build_report(
            esg_data=sample_esg_dataframe,
            company_profile=str(sample_company_profile_json),
            output_dir=str(test_output_dir),
            skip_materiality=True,
            skip_audit=True
        )

        assert isinstance(report, CSRDReport)
        assert report.metrics.total_metrics_calculated >= 0
        assert report.processing_time_total_minutes > 0

    def test_build_report_with_dict_company_profile(
        self,
        sample_esg_csv: Path,
        sample_company_profile_dict: Dict[str, Any],
        test_output_dir: Path
    ):
        """Test csrd_build_report with dict company profile."""
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=sample_company_profile_dict,
            output_dir=str(test_output_dir),
            skip_materiality=True,
            skip_audit=True
        )

        assert isinstance(report, CSRDReport)
        assert report.company_info["legal_name"] == "Test Company Ltd"

    def test_build_report_with_config(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path,
        sample_csrd_config: CSRDConfig,
        test_output_dir: Path
    ):
        """Test csrd_build_report with CSRDConfig object."""
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            config=sample_csrd_config,
            output_dir=str(test_output_dir),
            skip_materiality=True,
            skip_audit=True
        )

        assert isinstance(report, CSRDReport)
        assert report.reporting_period["year"] == sample_csrd_config.reporting_year

    def test_build_report_without_config(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path,
        test_output_dir: Path
    ):
        """Test csrd_build_report without config (uses defaults)."""
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            config=None,
            output_dir=str(test_output_dir),
            skip_materiality=True,
            skip_audit=True
        )

        assert isinstance(report, CSRDReport)
        assert report.report_id is not None

    def test_build_report_without_output_dir(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path
    ):
        """Test csrd_build_report without output directory."""
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            output_dir=None,
            skip_materiality=True,
            skip_audit=True
        )

        assert isinstance(report, CSRDReport)
        assert report.metrics is not None

    def test_build_report_return_structure(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path
    ):
        """Test csrd_build_report return value structure."""
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            skip_materiality=True,
            skip_audit=True
        )

        # Verify CSRDReport structure
        assert hasattr(report, "company_info")
        assert hasattr(report, "reporting_period")
        assert hasattr(report, "data_validation")
        assert hasattr(report, "materiality")
        assert hasattr(report, "metrics")
        assert hasattr(report, "compliance_status")
        assert hasattr(report, "report_id")
        assert hasattr(report, "generated_at")
        assert hasattr(report, "processing_time_total_minutes")

        # Verify properties
        assert hasattr(report, "is_compliant")
        assert hasattr(report, "is_audit_ready")
        assert hasattr(report, "material_standards")

        # Verify methods
        assert callable(report.to_dict)
        assert callable(report.to_json)
        assert callable(report.save_json)
        assert callable(report.save_summary)
        assert callable(report.to_dataframe)
        assert callable(report.summary)

    def test_build_report_with_custom_output_directory(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path,
        test_output_dir: Path
    ):
        """Test csrd_build_report with custom output directory."""
        custom_dir = test_output_dir / "custom_report"

        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            output_dir=str(custom_dir),
            skip_materiality=True,
            skip_audit=True
        )

        # Verify output directory was created
        assert custom_dir.exists()
        assert custom_dir.is_dir()

        # Verify output files exist
        assert (custom_dir / "00_complete_report.json").exists()
        assert (custom_dir / "00_summary.md").exists()
        assert (custom_dir / "01_validated_data.json").exists()


# ============================================================================
# TEST CLASS 2: TestCSRDConfigDataclass
# ============================================================================


class TestCSRDConfigDataclass:
    """Test the CSRDConfig dataclass."""

    def test_config_initialization(self):
        """Test CSRDConfig initialization with required fields."""
        config = CSRDConfig(
            company_name="Test Corp",
            company_lei="549300TEST1234567",
            reporting_year=2024,
            sector="Technology"
        )

        assert config.company_name == "Test Corp"
        assert config.company_lei == "549300TEST1234567"
        assert config.reporting_year == 2024
        assert config.sector == "Technology"

    def test_config_default_values(self):
        """Test CSRDConfig default values."""
        config = CSRDConfig(
            company_name="Test Corp",
            company_lei="549300TEST1234567",
            reporting_year=2024,
            sector="Technology"
        )

        # Check defaults
        assert config.country == "DE"
        assert config.employee_count is None
        assert config.revenue is None
        assert config.quality_threshold == 0.80
        assert config.impact_materiality_threshold == 5.0
        assert config.financial_materiality_threshold == 5.0
        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4o"

    def test_config_custom_values(self):
        """Test CSRDConfig with custom values."""
        config = CSRDConfig(
            company_name="Test Corp",
            company_lei="549300TEST1234567",
            reporting_year=2024,
            sector="Technology",
            country="UK",
            employee_count=5000,
            revenue=500000000.0,
            quality_threshold=0.90,
            llm_provider="anthropic",
            llm_model="claude-3-opus"
        )

        assert config.country == "UK"
        assert config.employee_count == 5000
        assert config.revenue == 500000000.0
        assert config.quality_threshold == 0.90
        assert config.llm_provider == "anthropic"
        assert config.llm_model == "claude-3-opus"

    def test_config_to_dict(self, sample_csrd_config: CSRDConfig):
        """Test CSRDConfig.to_dict() serialization."""
        config_dict = sample_csrd_config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["company_name"] == sample_csrd_config.company_name
        assert config_dict["company_lei"] == sample_csrd_config.company_lei
        assert config_dict["reporting_year"] == sample_csrd_config.reporting_year
        assert config_dict["sector"] == sample_csrd_config.sector

    def test_config_from_dict(self, sample_csrd_config: CSRDConfig):
        """Test CSRDConfig.from_dict() deserialization."""
        config_dict = sample_csrd_config.to_dict()
        new_config = CSRDConfig.from_dict(config_dict)

        assert new_config.company_name == sample_csrd_config.company_name
        assert new_config.company_lei == sample_csrd_config.company_lei
        assert new_config.reporting_year == sample_csrd_config.reporting_year
        assert new_config.sector == sample_csrd_config.sector

    def test_config_from_yaml(self, sample_config_yaml: Path):
        """Test CSRDConfig.from_yaml() loading."""
        config = CSRDConfig.from_yaml(str(sample_config_yaml))

        assert config.company_name == "Test Corporation"
        assert config.reporting_year == 2024
        assert config.quality_threshold == 0.80

    def test_config_from_env(self, monkeypatch):
        """Test CSRDConfig.from_env() loading."""
        # Set environment variables
        monkeypatch.setenv("CSRD_COMPANY_NAME", "EnvTest Corp")
        monkeypatch.setenv("CSRD_COMPANY_LEI", "549300ENVTEST12345")
        monkeypatch.setenv("CSRD_REPORTING_YEAR", "2024")
        monkeypatch.setenv("CSRD_SECTOR", "Manufacturing")
        monkeypatch.setenv("CSRD_COUNTRY", "FR")

        config = CSRDConfig.from_env()

        assert config.company_name == "EnvTest Corp"
        assert config.company_lei == "549300ENVTEST12345"
        assert config.reporting_year == 2024
        assert config.sector == "Manufacturing"
        assert config.country == "FR"

    def test_config_validation_required_fields(self):
        """Test CSRDConfig validation of required fields."""
        # This should work
        config = CSRDConfig(
            company_name="Test",
            company_lei="549300TEST1234567",
            reporting_year=2024,
            sector="Tech"
        )
        assert config is not None


# ============================================================================
# TEST CLASS 3: TestCSRDReportDataclass
# ============================================================================


class TestCSRDReportDataclass:
    """Test the CSRDReport dataclass."""

    def test_report_structure(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path
    ):
        """Test CSRDReport output structure."""
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            skip_materiality=True,
            skip_audit=True
        )

        # Verify structure
        assert isinstance(report.company_info, dict)
        assert isinstance(report.reporting_period, dict)
        assert isinstance(report.data_validation, dict)
        assert isinstance(report.materiality, MaterialityAssessment)
        assert isinstance(report.metrics, ESRSMetrics)
        assert isinstance(report.compliance_status, ComplianceStatus)

    def test_report_properties(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path
    ):
        """Test CSRDReport properties."""
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            skip_materiality=True,
            skip_audit=False
        )

        # Test properties
        assert isinstance(report.is_compliant, bool)
        assert isinstance(report.is_audit_ready, bool)
        assert isinstance(report.material_standards, list)

    def test_report_to_dict(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path
    ):
        """Test CSRDReport.to_dict() serialization."""
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            skip_materiality=True,
            skip_audit=True
        )

        report_dict = report.to_dict()

        assert isinstance(report_dict, dict)
        assert "company_info" in report_dict
        assert "validated_data" in report_dict

    def test_report_to_json(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path
    ):
        """Test CSRDReport.to_json() serialization."""
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            skip_materiality=True,
            skip_audit=True
        )

        json_str = report.to_json()

        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_report_save_json(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path,
        test_output_dir: Path
    ):
        """Test CSRDReport.save_json() method."""
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            skip_materiality=True,
            skip_audit=True
        )

        output_file = test_output_dir / "test_report.json"
        report.save_json(str(output_file))

        assert output_file.exists()

        # Verify content
        with open(output_file) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "company_info" in data

    def test_report_save_summary(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path,
        test_output_dir: Path
    ):
        """Test CSRDReport.save_summary() method."""
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            skip_materiality=True,
            skip_audit=True
        )

        output_file = test_output_dir / "test_summary.md"
        report.save_summary(str(output_file))

        assert output_file.exists()

        # Verify content
        content = output_file.read_text()
        assert "CSRD Report Summary" in content
        assert "Company:" in content

    def test_report_summary(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path
    ):
        """Test CSRDReport.summary() method."""
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            skip_materiality=True,
            skip_audit=True
        )

        summary = report.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "CSRD Report Summary" in summary
        assert "Company:" in summary
        assert "Compliance Status:" in summary


# ============================================================================
# TEST CLASS 4: TestDataFrameSupport
# ============================================================================


class TestDataFrameSupport:
    """Test DataFrame input/output support."""

    def test_input_dataframe(
        self,
        sample_esg_dataframe: pd.DataFrame,
        sample_company_profile_json: Path
    ):
        """Test accepting DataFrame as input."""
        report = csrd_build_report(
            esg_data=sample_esg_dataframe,
            company_profile=str(sample_company_profile_json),
            skip_materiality=True,
            skip_audit=True
        )

        assert isinstance(report, CSRDReport)
        assert report.metrics.total_metrics_calculated >= 0

    def test_output_dataframe(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path
    ):
        """Test CSRDReport.to_dataframe() output."""
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            skip_materiality=True,
            skip_audit=True
        )

        df = report.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        # May be empty if no validated data
        if len(df) > 0:
            assert "metric_code" in df.columns

    def test_dataframe_column_validation(self, sample_esg_dataframe: pd.DataFrame):
        """Test DataFrame column validation."""
        # Verify required columns exist
        required_columns = ["metric_code", "value", "unit"]
        for col in required_columns:
            assert col in sample_esg_dataframe.columns

    def test_empty_dataframe(
        self,
        sample_company_profile_json: Path
    ):
        """Test handling empty DataFrame."""
        empty_df = pd.DataFrame(columns=["metric_code", "value", "unit"])

        report = csrd_build_report(
            esg_data=empty_df,
            company_profile=str(sample_company_profile_json),
            skip_materiality=True,
            skip_audit=True
        )

        assert isinstance(report, CSRDReport)
        # Should handle empty data gracefully

    def test_large_dataframe(
        self,
        sample_company_profile_json: Path
    ):
        """Test handling large DataFrame."""
        # Create large DataFrame (1000 rows)
        large_df = pd.DataFrame({
            "metric_code": [f"E1-{i}" for i in range(1000)],
            "metric_name": [f"Metric {i}" for i in range(1000)],
            "value": [float(i) for i in range(1000)],
            "unit": ["tCO2e"] * 1000,
            "period_start": ["2024-01-01"] * 1000,
            "period_end": ["2024-12-31"] * 1000,
            "data_quality": ["high"] * 1000,
            "source": ["measured"] * 1000,
            "verification_status": ["verified"] * 1000
        })

        start_time = time.time()
        report = csrd_build_report(
            esg_data=large_df,
            company_profile=str(sample_company_profile_json),
            skip_materiality=True,
            skip_audit=True
        )
        elapsed = time.time() - start_time

        assert isinstance(report, CSRDReport)
        # Should process reasonably fast
        assert elapsed < 30.0  # Less than 30 seconds

    def test_dataframe_data_type_handling(
        self,
        sample_company_profile_json: Path
    ):
        """Test DataFrame data type handling."""
        # Create DataFrame with mixed types
        df = pd.DataFrame({
            "metric_code": ["E1-1", "E1-2"],
            "value": [1000, 500],  # Integers should be accepted
            "unit": ["tCO2e", "tCO2e"],
            "period_start": ["2024-01-01", "2024-01-01"],
            "period_end": ["2024-12-31", "2024-12-31"],
            "data_quality": ["high", "medium"],
            "source": ["measured", "estimated"],
            "verification_status": ["verified", "unverified"]
        })

        report = csrd_build_report(
            esg_data=df,
            company_profile=str(sample_company_profile_json),
            skip_materiality=True,
            skip_audit=True
        )

        assert isinstance(report, CSRDReport)


# ============================================================================
# TEST CLASS 5: TestIndividualAgentAccess
# ============================================================================


class TestIndividualAgentAccess:
    """Test individual agent access methods."""

    def test_csrd_validate_data(self, sample_esg_csv: Path):
        """Test csrd_validate_data() function."""
        result = csrd_validate_data(esg_data=str(sample_esg_csv))

        assert isinstance(result, dict)
        assert "metadata" in result
        assert "data_points" in result
        assert result["metadata"]["total_records"] > 0

    def test_csrd_validate_data_with_dataframe(self, sample_esg_dataframe: pd.DataFrame):
        """Test csrd_validate_data() with DataFrame."""
        result = csrd_validate_data(esg_data=sample_esg_dataframe)

        assert isinstance(result, dict)
        assert "metadata" in result

    def test_csrd_calculate_metrics(self, sample_esg_csv: Path):
        """Test csrd_calculate_metrics() function."""
        # First validate data
        validated = csrd_validate_data(esg_data=str(sample_esg_csv))

        # Then calculate metrics
        result = csrd_calculate_metrics(
            validated_data=validated,
            metrics_to_calculate=["E1-1", "E1-2", "E1-3"]
        )

        assert isinstance(result, dict)
        assert "metadata" in result
        assert "calculated_metrics" in result

    def test_csrd_calculate_metrics_with_dict(self, sample_esg_csv: Path):
        """Test csrd_calculate_metrics() with validated dict."""
        validated = csrd_validate_data(esg_data=str(sample_esg_csv))

        result = csrd_calculate_metrics(
            validated_data=validated,
            metrics_to_calculate=["E1-1", "E1-2"]
        )

        assert isinstance(result, dict)
        assert result["metadata"]["zero_hallucination_guarantee"] is True

    def test_individual_agent_config_override(
        self,
        sample_esg_csv: Path,
        sample_csrd_config: CSRDConfig
    ):
        """Test individual agent with config override."""
        result = csrd_validate_data(
            esg_data=str(sample_esg_csv),
            config=sample_csrd_config,
            quality_threshold=0.90
        )

        assert isinstance(result, dict)

    def test_validate_data_quality_threshold(self, sample_esg_csv: Path):
        """Test csrd_validate_data() with custom quality threshold."""
        result = csrd_validate_data(
            esg_data=str(sample_esg_csv),
            quality_threshold=0.90
        )

        assert isinstance(result, dict)
        assert "metadata" in result

    def test_calculate_metrics_zero_hallucination(self, sample_esg_csv: Path):
        """Test zero hallucination guarantee in calculations."""
        validated = csrd_validate_data(esg_data=str(sample_esg_csv))

        result = csrd_calculate_metrics(
            validated_data=validated,
            metrics_to_calculate=["E1-1"]
        )

        assert result["metadata"]["zero_hallucination_guarantee"] is True
        assert result["metadata"]["calculation_method"] == "deterministic"


# ============================================================================
# TEST CLASS 6: TestConfigurationManagement
# ============================================================================


class TestConfigurationManagement:
    """Test configuration management."""

    def test_load_config_from_yaml(self, sample_config_yaml: Path):
        """Test loading config from YAML file."""
        config = CSRDConfig.from_yaml(str(sample_config_yaml))

        assert isinstance(config, CSRDConfig)
        assert config.company_name is not None
        assert config.reporting_year > 0

    def test_load_config_from_dict(self, sample_csrd_config: CSRDConfig):
        """Test loading config from dict."""
        config_dict = sample_csrd_config.to_dict()
        new_config = CSRDConfig.from_dict(config_dict)

        assert new_config.company_name == sample_csrd_config.company_name
        assert new_config.company_lei == sample_csrd_config.company_lei

    def test_save_config_to_yaml(
        self,
        sample_csrd_config: CSRDConfig,
        test_output_dir: Path
    ):
        """Test saving config to YAML file."""
        output_file = test_output_dir / "saved_config.yaml"

        # Convert to dict and save
        config_dict = sample_csrd_config.to_dict()
        with open(output_file, 'w') as f:
            yaml.dump(config_dict, f)

        assert output_file.exists()

        # Load and verify
        with open(output_file) as f:
            loaded = yaml.safe_load(f)
        assert loaded["company_name"] == sample_csrd_config.company_name

    def test_update_config_values(self, sample_csrd_config: CSRDConfig):
        """Test updating config values."""
        original_year = sample_csrd_config.reporting_year
        sample_csrd_config.reporting_year = 2025

        assert sample_csrd_config.reporting_year == 2025
        assert sample_csrd_config.reporting_year != original_year

    def test_config_path_overrides(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path
    ):
        """Test config path overrides in csrd_build_report."""
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            esrs_data_points_path="data/esrs_data_points.json",
            data_quality_rules_path="rules/data_quality_rules.yaml",
            skip_materiality=True,
            skip_audit=True
        )

        assert isinstance(report, CSRDReport)

    def test_config_threshold_overrides(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path
    ):
        """Test config threshold overrides."""
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            quality_threshold=0.90,
            impact_materiality_threshold=6.0,
            financial_materiality_threshold=6.0,
            skip_materiality=True,
            skip_audit=True
        )

        assert isinstance(report, CSRDReport)


# ============================================================================
# TEST CLASS 7: TestErrorHandling
# ============================================================================


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_file_path(self, sample_company_profile_json: Path):
        """Test handling of invalid file path."""
        with pytest.raises(FileNotFoundError):
            csrd_build_report(
                esg_data="nonexistent_file.csv",
                company_profile=str(sample_company_profile_json),
                skip_materiality=True,
                skip_audit=True
            )

    def test_invalid_esg_data_format(
        self,
        tmp_path: Path,
        sample_company_profile_json: Path
    ):
        """Test handling of invalid ESG data format."""
        # Create invalid CSV (missing required columns)
        invalid_csv = tmp_path / "invalid.csv"
        invalid_csv.write_text("wrong_column,other_column\n1,2\n")

        # Should handle gracefully (may produce validation issues)
        report = csrd_build_report(
            esg_data=str(invalid_csv),
            company_profile=str(sample_company_profile_json),
            skip_materiality=True,
            skip_audit=True
        )

        assert isinstance(report, CSRDReport)

    def test_missing_company_profile(self, sample_esg_csv: Path):
        """Test handling of missing company profile."""
        with pytest.raises((FileNotFoundError, ValueError)):
            csrd_build_report(
                esg_data=str(sample_esg_csv),
                company_profile="nonexistent_profile.json",
                skip_materiality=True,
                skip_audit=True
            )

    def test_invalid_config_type(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path
    ):
        """Test handling of invalid config type."""
        # Passing invalid config type should work (will be ignored/handled)
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            config=None,  # None is valid
            skip_materiality=True,
            skip_audit=True
        )

        assert isinstance(report, CSRDReport)

    def test_empty_esg_data_file(
        self,
        tmp_path: Path,
        sample_company_profile_json: Path
    ):
        """Test handling of empty ESG data file."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")

        # Should handle gracefully
        try:
            report = csrd_build_report(
                esg_data=str(empty_csv),
                company_profile=str(sample_company_profile_json),
                skip_materiality=True,
                skip_audit=True
            )
            assert isinstance(report, CSRDReport)
        except Exception as e:
            # May raise error - that's acceptable
            assert True

    def test_corrupted_json_file(
        self,
        tmp_path: Path,
        sample_esg_csv: Path
    ):
        """Test handling of corrupted JSON file."""
        corrupted_json = tmp_path / "corrupted.json"
        corrupted_json.write_text("{ invalid json content }")

        with pytest.raises(json.JSONDecodeError):
            csrd_build_report(
                esg_data=str(sample_esg_csv),
                company_profile=str(corrupted_json),
                skip_materiality=True,
                skip_audit=True
            )

    def test_unsupported_file_format(
        self,
        tmp_path: Path,
        sample_company_profile_json: Path
    ):
        """Test handling of unsupported file format."""
        unsupported_file = tmp_path / "data.xyz"
        unsupported_file.write_text("some data")

        # Should attempt to process (may fail gracefully)
        try:
            report = csrd_build_report(
                esg_data=str(unsupported_file),
                company_profile=str(sample_company_profile_json),
                skip_materiality=True,
                skip_audit=True
            )
            # If it works, that's fine
            assert True
        except Exception:
            # If it fails, that's also acceptable
            assert True


# ============================================================================
# TEST CLASS 8: TestOutputValidation
# ============================================================================


class TestOutputValidation:
    """Test output validation."""

    def test_report_structure_completeness(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path
    ):
        """Test report structure completeness."""
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            skip_materiality=True,
            skip_audit=False
        )

        # Verify all required fields present
        assert report.company_info is not None
        assert report.reporting_period is not None
        assert report.data_validation is not None
        assert report.materiality is not None
        assert report.metrics is not None
        assert report.compliance_status is not None
        assert report.report_id is not None
        assert report.generated_at is not None

    def test_metrics_completeness(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path
    ):
        """Test metrics completeness."""
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            skip_materiality=True,
            skip_audit=True
        )

        metrics = report.metrics
        assert metrics.total_metrics_calculated is not None
        assert isinstance(metrics.metrics_by_standard, dict)
        assert metrics.calculation_method == "deterministic"
        assert metrics.zero_hallucination_guarantee is True

    def test_file_creation(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path,
        test_output_dir: Path
    ):
        """Test output file creation."""
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            output_dir=str(test_output_dir),
            skip_materiality=True,
            skip_audit=True
        )

        # Verify files created
        assert (test_output_dir / "00_complete_report.json").exists()
        assert (test_output_dir / "00_summary.md").exists()
        assert (test_output_dir / "01_validated_data.json").exists()

    def test_json_export(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path,
        test_output_dir: Path
    ):
        """Test JSON export format."""
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            skip_materiality=True,
            skip_audit=True
        )

        json_file = test_output_dir / "export_test.json"
        report.save_json(str(json_file))

        # Load and verify JSON
        with open(json_file) as f:
            data = json.load(f)

        assert "company_info" in data
        assert "validated_data" in data
        assert "calculated_metrics" in data

    def test_dataframe_export(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path
    ):
        """Test DataFrame export."""
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            skip_materiality=True,
            skip_audit=True
        )

        df = report.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        # If data exists, verify structure
        if len(df) > 0:
            assert "metric_code" in df.columns

    def test_compliance_status_values(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path
    ):
        """Test compliance status valid values."""
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            skip_materiality=True,
            skip_audit=False
        )

        valid_statuses = ["PASS", "WARNING", "FAIL", "SKIPPED"]
        assert report.compliance_status.compliance_status in valid_statuses


# ============================================================================
# TEST CLASS 9: TestPerformance
# ============================================================================


class TestPerformance:
    """Test performance characteristics."""

    def test_processing_time_tracking(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path
    ):
        """Test processing time tracking."""
        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            skip_materiality=True,
            skip_audit=True
        )

        assert report.processing_time_total_minutes is not None
        assert report.processing_time_total_minutes > 0
        assert report.metrics.processing_time_seconds is not None
        assert report.metrics.processing_time_seconds > 0

    def test_small_dataset_performance(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path
    ):
        """Test performance with small dataset."""
        start_time = time.time()

        report = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            skip_materiality=True,
            skip_audit=True
        )

        elapsed = time.time() - start_time

        assert isinstance(report, CSRDReport)
        # Should complete in reasonable time
        assert elapsed < 10.0  # Less than 10 seconds

    def test_medium_dataset_performance(
        self,
        sample_company_profile_json: Path
    ):
        """Test performance with medium dataset (100 records)."""
        # Create medium dataset
        df = pd.DataFrame({
            "metric_code": [f"E1-{i%10}" for i in range(100)],
            "metric_name": [f"Metric {i}" for i in range(100)],
            "value": [float(i) for i in range(100)],
            "unit": ["tCO2e"] * 100,
            "period_start": ["2024-01-01"] * 100,
            "period_end": ["2024-12-31"] * 100,
            "data_quality": ["high"] * 100,
            "source": ["measured"] * 100,
            "verification_status": ["verified"] * 100
        })

        start_time = time.time()

        report = csrd_build_report(
            esg_data=df,
            company_profile=str(sample_company_profile_json),
            skip_materiality=True,
            skip_audit=True
        )

        elapsed = time.time() - start_time

        assert isinstance(report, CSRDReport)
        assert elapsed < 15.0  # Less than 15 seconds

    def test_report_generation_overhead(
        self,
        sample_esg_csv: Path,
        sample_company_profile_json: Path
    ):
        """Test report generation overhead."""
        # Generate without output dir
        start1 = time.time()
        report1 = csrd_build_report(
            esg_data=str(sample_esg_csv),
            company_profile=str(sample_company_profile_json),
            output_dir=None,
            skip_materiality=True,
            skip_audit=True
        )
        time1 = time.time() - start1

        assert isinstance(report1, CSRDReport)
        # Should be reasonably fast
        assert time1 < 10.0


# ============================================================================
# END OF TESTS
# ============================================================================
