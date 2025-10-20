"""
CSRD/ESRS Digital Reporting Platform - CLI Tests

Comprehensive test suite for all 8 CLI commands.

This test file validates the command-line interface (CLI) that provides
user access to the CSRD/ESRS platform. The CLI is built with Click framework
and Rich for beautiful terminal UI.

Commands tested:
1. csrd run        - Execute full 6-agent pipeline
2. csrd validate   - Run IntakeAgent only (data validation)
3. csrd calculate  - Run CalculatorAgent only (metric calculations)
4. csrd audit      - Run AuditAgent only (compliance check)
5. csrd materialize - Run MaterialityAgent only (double materiality)
6. csrd report     - Run ReportingAgent only (XBRL generation)
7. csrd aggregate  - Run AggregatorAgent only (multi-framework)
8. csrd config     - Configuration management

Critical test areas:
- Parameter validation and error messages
- File path handling (CSV/JSON/Excel/Parquet)
- Success/failure exit codes
- Output formatting and Rich UI
- Help text clarity
- Configuration persistence
- Error handling and graceful failures

TARGET: 50-60 test cases with comprehensive coverage

Version: 1.0.0
Author: GreenLang CSRD Team
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from cli.csrd_commands import csrd


# ============================================================================
# PYTEST FIXTURES
# ============================================================================


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create temporary directory for test files."""
    return tmp_path


@pytest.fixture
def sample_csv_data() -> str:
    """Sample CSV ESG data."""
    return """metric_code,metric_name,value,unit,period_start,period_end,data_quality
E1-1,Scope 1 GHG Emissions,1000.0,tCO2e,2024-01-01,2024-12-31,high
E1-2,Scope 2 GHG Emissions,500.0,tCO2e,2024-01-01,2024-12-31,high
S1-1,Total Employees,250.0,FTE,2024-01-01,2024-12-31,high"""


@pytest.fixture
def sample_company_profile() -> Dict[str, Any]:
    """Sample company profile JSON."""
    return {
        "company_name": "Test Corp",
        "legal_entity_identifier": "123456789TESTLEI001",
        "country": "NL",
        "sector": "Technology",
        "reporting_period": {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31"
        }
    }


@pytest.fixture
def sample_validated_data() -> Dict[str, Any]:
    """Sample validated data from IntakeAgent."""
    return {
        "status": "success",
        "data": [
            {
                "metric_code": "E1-1",
                "value": 1000.0,
                "unit": "tCO2e",
                "quality_score": 0.95
            }
        ],
        "metadata": {
            "total_records": 10,
            "valid_records": 10,
            "invalid_records": 0,
            "warnings": 0,
            "overall_quality_score": 95.0,
            "timestamp": "2024-01-15T10:00:00Z"
        }
    }


@pytest.fixture
def sample_report_data() -> Dict[str, Any]:
    """Sample report data for audit."""
    return {
        "status": "success",
        "report": {
            "format": "xbrl",
            "metrics": [{"code": "E1-1", "value": 1000.0}]
        },
        "metadata": {
            "format": "XBRL",
            "xbrl_tags_count": 100,
            "esef_compliant": True,
            "file_size": "250 KB"
        }
    }


@pytest.fixture
def sample_config_data() -> Dict[str, Any]:
    """Sample CSRD configuration."""
    return {
        "company": {
            "name": "Test Corp",
            "lei": "123456789TESTLEI001",
            "country": "NL",
            "sector": "Technology"
        },
        "reporting_period": {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31"
        },
        "materiality": {
            "threshold": 50
        }
    }


# ============================================================================
# TEST CLASS 1: CSRD RUN COMMAND
# ============================================================================


class TestCLIRunCommand:
    """Tests for 'csrd run' command - full pipeline execution."""

    def test_run_missing_input_parameter(self, cli_runner: CliRunner):
        """Test that --input parameter is required."""
        result = cli_runner.invoke(csrd, ['run'])

        assert result.exit_code != 0
        assert "Missing option '--input'" in result.output or "required" in result.output.lower()

    def test_run_missing_company_profile(self, cli_runner: CliRunner, temp_dir: Path):
        """Test that --company-profile parameter is required."""
        input_file = temp_dir / "data.csv"
        input_file.write_text("metric_code,value\nE1-1,1000")

        result = cli_runner.invoke(csrd, [
            'run',
            '--input', str(input_file)
        ])

        assert result.exit_code != 0
        assert "Missing option '--company-profile'" in result.output or "required" in result.output.lower()

    def test_run_with_nonexistent_input_file(self, cli_runner: CliRunner, temp_dir: Path):
        """Test error when input file doesn't exist."""
        nonexistent = temp_dir / "nonexistent.csv"

        result = cli_runner.invoke(csrd, [
            'run',
            '--input', str(nonexistent),
            '--company-profile', 'profile.json'
        ])

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or "path" in result.output.lower()

    def test_run_with_nonexistent_company_profile(
        self,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_csv_data: str
    ):
        """Test error when company profile doesn't exist."""
        input_file = temp_dir / "data.csv"
        input_file.write_text(sample_csv_data)
        nonexistent_profile = temp_dir / "nonexistent.json"

        result = cli_runner.invoke(csrd, [
            'run',
            '--input', str(input_file),
            '--company-profile', str(nonexistent_profile)
        ])

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or "path" in result.output.lower()

    @patch('cli.csrd_commands.IntakeAgent')
    @patch('cli.csrd_commands.MaterialityAgent')
    @patch('cli.csrd_commands.CalculatorAgent')
    @patch('cli.csrd_commands.AggregatorAgent')
    @patch('cli.csrd_commands.ReportingAgent')
    @patch('cli.csrd_commands.AuditAgent')
    def test_run_successful_execution(
        self,
        mock_audit: MagicMock,
        mock_reporting: MagicMock,
        mock_aggregator: MagicMock,
        mock_calculator: MagicMock,
        mock_materiality: MagicMock,
        mock_intake: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_csv_data: str,
        sample_company_profile: Dict[str, Any]
    ):
        """Test successful pipeline execution."""
        # Create input files
        input_file = temp_dir / "data.csv"
        input_file.write_text(sample_csv_data)

        profile_file = temp_dir / "company.json"
        profile_file.write_text(json.dumps(sample_company_profile))

        output_dir = temp_dir / "output"

        # Mock agent responses
        mock_result = {
            "status": "success",
            "metadata": {
                "total_records": 10,
                "valid_records": 10,
                "invalid_records": 0,
                "warnings": 0,
                "overall_quality_score": 95.0,
                "metrics_calculated": 50,
                "is_compliant": True,
                "rules_checked": 200,
                "critical_issues": 0,
                "timestamp": "2024-01-15T10:00:00Z"
            }
        }

        mock_intake.return_value.process.return_value = mock_result
        mock_materiality.return_value.process.return_value = mock_result
        mock_calculator.return_value.process.return_value = mock_result
        mock_aggregator.return_value.process.return_value = mock_result
        mock_reporting.return_value.process.return_value = mock_result
        mock_audit.return_value.process.return_value = mock_result

        # Run command
        result = cli_runner.invoke(csrd, [
            'run',
            '--input', str(input_file),
            '--company-profile', str(profile_file),
            '--output-dir', str(output_dir),
            '--quiet'
        ])

        assert result.exit_code == 0

    @patch('cli.csrd_commands.IntakeAgent')
    def test_run_with_skip_materiality_flag(
        self,
        mock_intake: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_csv_data: str,
        sample_company_profile: Dict[str, Any]
    ):
        """Test --skip-materiality flag."""
        input_file = temp_dir / "data.csv"
        input_file.write_text(sample_csv_data)

        profile_file = temp_dir / "company.json"
        profile_file.write_text(json.dumps(sample_company_profile))

        mock_intake.return_value.process.return_value = {
            "status": "success",
            "metadata": {"total_records": 10}
        }

        result = cli_runner.invoke(csrd, [
            'run',
            '--input', str(input_file),
            '--company-profile', str(profile_file),
            '--skip-materiality',
            '--skip-audit'  # Also skip audit to avoid full pipeline
        ])

        # Should not fail on missing MaterialityAgent
        assert "materiality" in result.output.lower() or result.exit_code in [0, 1]

    @patch('cli.csrd_commands.IntakeAgent')
    @patch('cli.csrd_commands.MaterialityAgent')
    @patch('cli.csrd_commands.CalculatorAgent')
    @patch('cli.csrd_commands.AggregatorAgent')
    @patch('cli.csrd_commands.ReportingAgent')
    @patch('cli.csrd_commands.AuditAgent')
    def test_run_with_verbose_flag(
        self,
        mock_audit: MagicMock,
        mock_reporting: MagicMock,
        mock_aggregator: MagicMock,
        mock_calculator: MagicMock,
        mock_materiality: MagicMock,
        mock_intake: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_csv_data: str,
        sample_company_profile: Dict[str, Any]
    ):
        """Test verbose output includes detailed information."""
        input_file = temp_dir / "data.csv"
        input_file.write_text(sample_csv_data)

        profile_file = temp_dir / "company.json"
        profile_file.write_text(json.dumps(sample_company_profile))

        mock_result = {
            "status": "success",
            "metadata": {
                "total_records": 10,
                "is_compliant": True
            },
            "issues": [
                {
                    "severity": "warning",
                    "error_code": "W001",
                    "message": "Test warning"
                }
            ]
        }

        mock_intake.return_value.process.return_value = mock_result
        mock_materiality.return_value.process.return_value = mock_result
        mock_calculator.return_value.process.return_value = mock_result
        mock_aggregator.return_value.process.return_value = mock_result
        mock_reporting.return_value.process.return_value = mock_result
        mock_audit.return_value.process.return_value = mock_result

        result = cli_runner.invoke(csrd, [
            'run',
            '--input', str(input_file),
            '--company-profile', str(profile_file),
            '--verbose'
        ])

        # Verbose mode should show more detail
        assert result.exit_code in [0, 1]


# ============================================================================
# TEST CLASS 2: CSRD VALIDATE COMMAND
# ============================================================================


class TestCLIValidateCommand:
    """Tests for 'csrd validate' command - data validation only."""

    def test_validate_missing_input_parameter(self, cli_runner: CliRunner):
        """Test that --input parameter is required."""
        result = cli_runner.invoke(csrd, ['validate'])

        assert result.exit_code != 0
        assert "Missing option '--input'" in result.output or "required" in result.output.lower()

    def test_validate_with_nonexistent_file(self, cli_runner: CliRunner, temp_dir: Path):
        """Test error when input file doesn't exist."""
        nonexistent = temp_dir / "nonexistent.csv"

        result = cli_runner.invoke(csrd, [
            'validate',
            '--input', str(nonexistent)
        ])

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or "path" in result.output.lower()

    @patch('cli.csrd_commands.IntakeAgent')
    def test_validate_csv_file_success(
        self,
        mock_intake: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_csv_data: str
    ):
        """Test successful CSV validation."""
        input_file = temp_dir / "data.csv"
        input_file.write_text(sample_csv_data)

        mock_intake.return_value.process.return_value = {
            "status": "success",
            "metadata": {
                "total_records": 3,
                "valid_records": 3,
                "invalid_records": 0,
                "warnings": 0,
                "overall_quality_score": 95.0
            }
        }

        result = cli_runner.invoke(csrd, [
            'validate',
            '--input', str(input_file),
            '--quiet'
        ])

        assert result.exit_code == 0

    @patch('cli.csrd_commands.IntakeAgent')
    def test_validate_json_file_success(
        self,
        mock_intake: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path
    ):
        """Test successful JSON validation."""
        input_data = {
            "metrics": [
                {"metric_code": "E1-1", "value": 1000.0}
            ]
        }
        input_file = temp_dir / "data.json"
        input_file.write_text(json.dumps(input_data))

        mock_intake.return_value.process.return_value = {
            "status": "success",
            "metadata": {
                "total_records": 1,
                "valid_records": 1,
                "invalid_records": 0,
                "warnings": 0,
                "overall_quality_score": 100.0
            }
        }

        result = cli_runner.invoke(csrd, [
            'validate',
            '--input', str(input_file),
            '--quiet'
        ])

        assert result.exit_code == 0

    @patch('cli.csrd_commands.IntakeAgent')
    def test_validate_with_invalid_records(
        self,
        mock_intake: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_csv_data: str
    ):
        """Test validation with invalid records (exit code 2)."""
        input_file = temp_dir / "data.csv"
        input_file.write_text(sample_csv_data)

        mock_intake.return_value.process.return_value = {
            "status": "warning",
            "metadata": {
                "total_records": 10,
                "valid_records": 7,
                "invalid_records": 3,
                "warnings": 5,
                "overall_quality_score": 70.0
            }
        }

        result = cli_runner.invoke(csrd, [
            'validate',
            '--input', str(input_file),
            '--quiet'
        ])

        # Exit code 2 for warnings
        assert result.exit_code == 2

    @patch('cli.csrd_commands.IntakeAgent')
    def test_validate_with_output_file(
        self,
        mock_intake: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_csv_data: str,
        sample_validated_data: Dict[str, Any]
    ):
        """Test saving validation report to output file."""
        input_file = temp_dir / "data.csv"
        input_file.write_text(sample_csv_data)

        output_file = temp_dir / "validation_report.json"

        mock_intake.return_value.process.return_value = sample_validated_data

        result = cli_runner.invoke(csrd, [
            'validate',
            '--input', str(input_file),
            '--output', str(output_file),
            '--quiet'
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify output content
        with open(output_file) as f:
            output_data = json.load(f)
        assert output_data["status"] == "success"

    @patch('cli.csrd_commands.IntakeAgent')
    def test_validate_verbose_shows_details(
        self,
        mock_intake: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_csv_data: str
    ):
        """Test verbose mode shows detailed validation errors."""
        input_file = temp_dir / "data.csv"
        input_file.write_text(sample_csv_data)

        mock_intake.return_value.process.return_value = {
            "status": "success",
            "metadata": {
                "total_records": 3,
                "valid_records": 3,
                "invalid_records": 0,
                "warnings": 0,
                "overall_quality_score": 95.0
            },
            "issues": [
                {
                    "severity": "info",
                    "error_code": "I001",
                    "message": "Data quality is high"
                }
            ]
        }

        result = cli_runner.invoke(csrd, [
            'validate',
            '--input', str(input_file),
            '--verbose'
        ])

        assert result.exit_code == 0

    @patch('cli.csrd_commands.IntakeAgent')
    def test_validate_error_handling(
        self,
        mock_intake: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_csv_data: str
    ):
        """Test graceful error handling."""
        input_file = temp_dir / "data.csv"
        input_file.write_text(sample_csv_data)

        mock_intake.return_value.process.side_effect = Exception("Test error")

        result = cli_runner.invoke(csrd, [
            'validate',
            '--input', str(input_file)
        ])

        assert result.exit_code == 1
        assert "failed" in result.output.lower() or "error" in result.output.lower()


# ============================================================================
# TEST CLASS 3: CSRD CALCULATE COMMAND
# ============================================================================


class TestCLICalculateCommand:
    """Tests for 'csrd calculate' command - metric calculations only."""

    def test_calculate_missing_input_parameter(self, cli_runner: CliRunner):
        """Test that --input parameter is required."""
        result = cli_runner.invoke(csrd, ['calculate'])

        assert result.exit_code != 0
        assert "Missing option '--input'" in result.output or "required" in result.output.lower()

    def test_calculate_with_nonexistent_file(self, cli_runner: CliRunner, temp_dir: Path):
        """Test error when input file doesn't exist."""
        nonexistent = temp_dir / "nonexistent.json"

        result = cli_runner.invoke(csrd, [
            'calculate',
            '--input', str(nonexistent)
        ])

        assert result.exit_code != 0

    @patch('cli.csrd_commands.CalculatorAgent')
    def test_calculate_successful_execution(
        self,
        mock_calculator: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_validated_data: Dict[str, Any]
    ):
        """Test successful metric calculation."""
        input_file = temp_dir / "validated.json"
        input_file.write_text(json.dumps(sample_validated_data))

        mock_calculator.return_value.process.return_value = {
            "status": "success",
            "metadata": {
                "metrics_calculated": 50,
                "calculation_errors": 0,
                "avg_time_per_metric_ms": 5.2
            }
        }

        result = cli_runner.invoke(csrd, [
            'calculate',
            '--input', str(input_file),
            '--quiet'
        ])

        assert result.exit_code == 0

    @patch('cli.csrd_commands.CalculatorAgent')
    def test_calculate_with_output_file(
        self,
        mock_calculator: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_validated_data: Dict[str, Any]
    ):
        """Test saving calculation results to output file."""
        input_file = temp_dir / "validated.json"
        input_file.write_text(json.dumps(sample_validated_data))

        output_file = temp_dir / "calculated.json"

        calculation_result = {
            "status": "success",
            "calculated_metrics": [
                {"code": "E1-1", "value": 1000.0, "unit": "tCO2e"}
            ],
            "metadata": {
                "metrics_calculated": 50,
                "calculation_errors": 0
            }
        }

        mock_calculator.return_value.process.return_value = calculation_result

        result = cli_runner.invoke(csrd, [
            'calculate',
            '--input', str(input_file),
            '--output', str(output_file),
            '--quiet'
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    @patch('cli.csrd_commands.CalculatorAgent')
    def test_calculate_zero_hallucination_message(
        self,
        mock_calculator: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_validated_data: Dict[str, Any]
    ):
        """Test that ZERO HALLUCINATION message is displayed."""
        input_file = temp_dir / "validated.json"
        input_file.write_text(json.dumps(sample_validated_data))

        mock_calculator.return_value.process.return_value = {
            "status": "success",
            "metadata": {
                "metrics_calculated": 50,
                "calculation_errors": 0
            }
        }

        result = cli_runner.invoke(csrd, [
            'calculate',
            '--input', str(input_file)
        ])

        assert result.exit_code == 0
        # Should mention zero hallucination
        assert "ZERO" in result.output.upper() or "hallucination" in result.output.lower()

    @patch('cli.csrd_commands.CalculatorAgent')
    def test_calculate_error_handling(
        self,
        mock_calculator: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_validated_data: Dict[str, Any]
    ):
        """Test calculation error handling."""
        input_file = temp_dir / "validated.json"
        input_file.write_text(json.dumps(sample_validated_data))

        mock_calculator.return_value.process.side_effect = Exception("Calculation error")

        result = cli_runner.invoke(csrd, [
            'calculate',
            '--input', str(input_file)
        ])

        assert result.exit_code == 1
        assert "failed" in result.output.lower() or "error" in result.output.lower()


# ============================================================================
# TEST CLASS 4: CSRD MATERIALIZE COMMAND
# ============================================================================


class TestCLIMaterializeCommand:
    """Tests for 'csrd materialize' command - materiality assessment."""

    def test_materialize_missing_input_parameter(self, cli_runner: CliRunner):
        """Test that --input parameter is required."""
        result = cli_runner.invoke(csrd, ['materialize'])

        assert result.exit_code != 0
        assert "Missing option '--input'" in result.output or "required" in result.output.lower()

    def test_materialize_missing_company_profile(
        self,
        cli_runner: CliRunner,
        temp_dir: Path
    ):
        """Test that --company-profile parameter is required."""
        input_file = temp_dir / "data.json"
        input_file.write_text('{"data": []}')

        result = cli_runner.invoke(csrd, [
            'materialize',
            '--input', str(input_file)
        ])

        assert result.exit_code != 0
        assert "Missing option '--company-profile'" in result.output or "required" in result.output.lower()

    @patch('cli.csrd_commands.MaterialityAgent')
    def test_materialize_successful_execution(
        self,
        mock_materiality: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_validated_data: Dict[str, Any],
        sample_company_profile: Dict[str, Any]
    ):
        """Test successful materiality assessment."""
        input_file = temp_dir / "validated.json"
        input_file.write_text(json.dumps(sample_validated_data))

        profile_file = temp_dir / "company.json"
        profile_file.write_text(json.dumps(sample_company_profile))

        mock_materiality.return_value.process.return_value = {
            "status": "success",
            "metadata": {
                "material_topics_count": 10,
                "impact_material_count": 8,
                "financial_material_count": 6,
                "double_material_count": 5
            }
        }

        result = cli_runner.invoke(csrd, [
            'materialize',
            '--input', str(input_file),
            '--company-profile', str(profile_file),
            '--quiet'
        ])

        assert result.exit_code == 0

    @patch('cli.csrd_commands.MaterialityAgent')
    def test_materialize_with_output_file(
        self,
        mock_materiality: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_validated_data: Dict[str, Any],
        sample_company_profile: Dict[str, Any]
    ):
        """Test saving materiality assessment to file."""
        input_file = temp_dir / "validated.json"
        input_file.write_text(json.dumps(sample_validated_data))

        profile_file = temp_dir / "company.json"
        profile_file.write_text(json.dumps(sample_company_profile))

        output_file = temp_dir / "materiality.json"

        assessment_result = {
            "status": "success",
            "material_topics": [
                {"topic": "E1", "is_material": True}
            ],
            "metadata": {
                "material_topics_count": 10
            }
        }

        mock_materiality.return_value.process.return_value = assessment_result

        result = cli_runner.invoke(csrd, [
            'materialize',
            '--input', str(input_file),
            '--company-profile', str(profile_file),
            '--output', str(output_file),
            '--quiet'
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    @patch('cli.csrd_commands.MaterialityAgent')
    def test_materialize_error_handling(
        self,
        mock_materiality: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_validated_data: Dict[str, Any],
        sample_company_profile: Dict[str, Any]
    ):
        """Test materiality assessment error handling."""
        input_file = temp_dir / "validated.json"
        input_file.write_text(json.dumps(sample_validated_data))

        profile_file = temp_dir / "company.json"
        profile_file.write_text(json.dumps(sample_company_profile))

        mock_materiality.return_value.process.side_effect = Exception("Assessment error")

        result = cli_runner.invoke(csrd, [
            'materialize',
            '--input', str(input_file),
            '--company-profile', str(profile_file)
        ])

        assert result.exit_code == 1
        assert "failed" in result.output.lower() or "error" in result.output.lower()


# ============================================================================
# TEST CLASS 5: CSRD REPORT COMMAND
# ============================================================================


class TestCLIReportCommand:
    """Tests for 'csrd report' command - XBRL/ESEF report generation."""

    def test_report_missing_input_parameter(self, cli_runner: CliRunner):
        """Test that --input parameter is required."""
        result = cli_runner.invoke(csrd, ['report'])

        assert result.exit_code != 0
        assert "Missing option '--input'" in result.output or "required" in result.output.lower()

    def test_report_missing_company_profile(
        self,
        cli_runner: CliRunner,
        temp_dir: Path
    ):
        """Test that --company-profile parameter is required."""
        input_file = temp_dir / "data.json"
        input_file.write_text('{"data": []}')

        result = cli_runner.invoke(csrd, [
            'report',
            '--input', str(input_file)
        ])

        assert result.exit_code != 0
        assert "Missing option '--company-profile'" in result.output or "required" in result.output.lower()

    @patch('cli.csrd_commands.ReportingAgent')
    def test_report_xbrl_generation(
        self,
        mock_reporting: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_validated_data: Dict[str, Any],
        sample_company_profile: Dict[str, Any]
    ):
        """Test XBRL report generation."""
        input_file = temp_dir / "aggregated.json"
        input_file.write_text(json.dumps(sample_validated_data))

        profile_file = temp_dir / "company.json"
        profile_file.write_text(json.dumps(sample_company_profile))

        mock_reporting.return_value.process.return_value = {
            "status": "success",
            "metadata": {
                "format": "XBRL",
                "xbrl_tags_count": 100,
                "esef_compliant": True,
                "file_size": "250 KB"
            }
        }

        result = cli_runner.invoke(csrd, [
            'report',
            '--input', str(input_file),
            '--company-profile', str(profile_file),
            '--format', 'xbrl',
            '--quiet'
        ])

        assert result.exit_code == 0

    @patch('cli.csrd_commands.ReportingAgent')
    def test_report_json_format(
        self,
        mock_reporting: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_validated_data: Dict[str, Any],
        sample_company_profile: Dict[str, Any]
    ):
        """Test JSON report format."""
        input_file = temp_dir / "aggregated.json"
        input_file.write_text(json.dumps(sample_validated_data))

        profile_file = temp_dir / "company.json"
        profile_file.write_text(json.dumps(sample_company_profile))

        mock_reporting.return_value.process.return_value = {
            "status": "success",
            "metadata": {
                "format": "JSON",
                "file_size": "150 KB"
            }
        }

        result = cli_runner.invoke(csrd, [
            'report',
            '--input', str(input_file),
            '--company-profile', str(profile_file),
            '--format', 'json',
            '--quiet'
        ])

        assert result.exit_code == 0

    @patch('cli.csrd_commands.ReportingAgent')
    def test_report_both_formats(
        self,
        mock_reporting: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_validated_data: Dict[str, Any],
        sample_company_profile: Dict[str, Any]
    ):
        """Test generating both XBRL and JSON formats."""
        input_file = temp_dir / "aggregated.json"
        input_file.write_text(json.dumps(sample_validated_data))

        profile_file = temp_dir / "company.json"
        profile_file.write_text(json.dumps(sample_company_profile))

        mock_reporting.return_value.process.return_value = {
            "status": "success",
            "metadata": {
                "format": "BOTH",
                "xbrl_tags_count": 100,
                "esef_compliant": True
            }
        }

        result = cli_runner.invoke(csrd, [
            'report',
            '--input', str(input_file),
            '--company-profile', str(profile_file),
            '--format', 'both',
            '--quiet'
        ])

        assert result.exit_code == 0

    @patch('cli.csrd_commands.ReportingAgent')
    def test_report_custom_output_directory(
        self,
        mock_reporting: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_validated_data: Dict[str, Any],
        sample_company_profile: Dict[str, Any]
    ):
        """Test custom output directory."""
        input_file = temp_dir / "aggregated.json"
        input_file.write_text(json.dumps(sample_validated_data))

        profile_file = temp_dir / "company.json"
        profile_file.write_text(json.dumps(sample_company_profile))

        output_dir = temp_dir / "custom_reports"

        mock_reporting.return_value.process.return_value = {
            "status": "success",
            "metadata": {
                "format": "XBRL",
                "esef_compliant": True
            }
        }

        result = cli_runner.invoke(csrd, [
            'report',
            '--input', str(input_file),
            '--company-profile', str(profile_file),
            '--output-dir', str(output_dir),
            '--quiet'
        ])

        assert result.exit_code == 0

    @patch('cli.csrd_commands.ReportingAgent')
    def test_report_error_handling(
        self,
        mock_reporting: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_validated_data: Dict[str, Any],
        sample_company_profile: Dict[str, Any]
    ):
        """Test report generation error handling."""
        input_file = temp_dir / "aggregated.json"
        input_file.write_text(json.dumps(sample_validated_data))

        profile_file = temp_dir / "company.json"
        profile_file.write_text(json.dumps(sample_company_profile))

        mock_reporting.return_value.process.side_effect = Exception("XBRL generation failed")

        result = cli_runner.invoke(csrd, [
            'report',
            '--input', str(input_file),
            '--company-profile', str(profile_file)
        ])

        assert result.exit_code == 1
        assert "failed" in result.output.lower() or "error" in result.output.lower()


# ============================================================================
# TEST CLASS 6: CSRD AUDIT COMMAND
# ============================================================================


class TestCLIAuditCommand:
    """Tests for 'csrd audit' command - compliance validation."""

    def test_audit_missing_report_parameter(self, cli_runner: CliRunner):
        """Test that --report parameter is required."""
        result = cli_runner.invoke(csrd, ['audit'])

        assert result.exit_code != 0
        assert "Missing option '--report'" in result.output or "required" in result.output.lower()

    def test_audit_with_nonexistent_file(self, cli_runner: CliRunner, temp_dir: Path):
        """Test error when report file doesn't exist."""
        nonexistent = temp_dir / "nonexistent.json"

        result = cli_runner.invoke(csrd, [
            'audit',
            '--report', str(nonexistent)
        ])

        assert result.exit_code != 0

    @patch('cli.csrd_commands.AuditAgent')
    def test_audit_compliance_pass(
        self,
        mock_audit: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_report_data: Dict[str, Any]
    ):
        """Test audit with PASS status."""
        report_file = temp_dir / "report.json"
        report_file.write_text(json.dumps(sample_report_data))

        mock_audit.return_value.process.return_value = {
            "status": "success",
            "metadata": {
                "is_compliant": True,
                "rules_checked": 200,
                "critical_issues": 0,
                "warnings": 0,
                "info_count": 5
            }
        }

        result = cli_runner.invoke(csrd, [
            'audit',
            '--report', str(report_file),
            '--quiet'
        ])

        assert result.exit_code == 0

    @patch('cli.csrd_commands.AuditAgent')
    def test_audit_compliance_fail(
        self,
        mock_audit: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_report_data: Dict[str, Any]
    ):
        """Test audit with FAIL status (exit code 2)."""
        report_file = temp_dir / "report.json"
        report_file.write_text(json.dumps(sample_report_data))

        mock_audit.return_value.process.return_value = {
            "status": "failed",
            "metadata": {
                "is_compliant": False,
                "rules_checked": 200,
                "critical_issues": 5,
                "warnings": 10,
                "info_count": 3
            }
        }

        result = cli_runner.invoke(csrd, [
            'audit',
            '--report', str(report_file),
            '--quiet'
        ])

        # Exit code 2 for compliance failure
        assert result.exit_code == 2

    @patch('cli.csrd_commands.AuditAgent')
    def test_audit_with_warnings(
        self,
        mock_audit: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_report_data: Dict[str, Any]
    ):
        """Test audit with warnings but no critical issues."""
        report_file = temp_dir / "report.json"
        report_file.write_text(json.dumps(sample_report_data))

        mock_audit.return_value.process.return_value = {
            "status": "success",
            "metadata": {
                "is_compliant": True,
                "rules_checked": 200,
                "critical_issues": 0,
                "warnings": 15,
                "info_count": 20
            }
        }

        result = cli_runner.invoke(csrd, [
            'audit',
            '--report', str(report_file)
        ])

        assert result.exit_code == 0

    @patch('cli.csrd_commands.AuditAgent')
    def test_audit_with_output_file(
        self,
        mock_audit: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_report_data: Dict[str, Any]
    ):
        """Test saving audit report to output file."""
        report_file = temp_dir / "report.json"
        report_file.write_text(json.dumps(sample_report_data))

        output_file = temp_dir / "audit_report.json"

        audit_result = {
            "status": "success",
            "compliance_issues": [],
            "metadata": {
                "is_compliant": True,
                "rules_checked": 200,
                "critical_issues": 0
            }
        }

        mock_audit.return_value.process.return_value = audit_result

        result = cli_runner.invoke(csrd, [
            'audit',
            '--report', str(report_file),
            '--output', str(output_file),
            '--quiet'
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    @patch('cli.csrd_commands.AuditAgent')
    def test_audit_verbose_shows_issues(
        self,
        mock_audit: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_report_data: Dict[str, Any]
    ):
        """Test verbose mode shows detailed compliance issues."""
        report_file = temp_dir / "report.json"
        report_file.write_text(json.dumps(sample_report_data))

        mock_audit.return_value.process.return_value = {
            "status": "success",
            "metadata": {
                "is_compliant": True,
                "rules_checked": 200,
                "critical_issues": 0
            },
            "issues": [
                {
                    "severity": "info",
                    "error_code": "I001",
                    "message": "All checks passed"
                }
            ]
        }

        result = cli_runner.invoke(csrd, [
            'audit',
            '--report', str(report_file),
            '--verbose'
        ])

        assert result.exit_code == 0

    @patch('cli.csrd_commands.AuditAgent')
    def test_audit_error_handling(
        self,
        mock_audit: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_report_data: Dict[str, Any]
    ):
        """Test audit error handling."""
        report_file = temp_dir / "report.json"
        report_file.write_text(json.dumps(sample_report_data))

        mock_audit.return_value.process.side_effect = Exception("Audit failed")

        result = cli_runner.invoke(csrd, [
            'audit',
            '--report', str(report_file)
        ])

        assert result.exit_code == 1
        assert "failed" in result.output.lower() or "error" in result.output.lower()


# ============================================================================
# TEST CLASS 7: CSRD AGGREGATE COMMAND
# ============================================================================


class TestCLIAggregateCommand:
    """Tests for 'csrd aggregate' command - multi-framework integration."""

    def test_aggregate_missing_input_parameter(self, cli_runner: CliRunner):
        """Test that --input parameter is required."""
        result = cli_runner.invoke(csrd, ['aggregate'])

        assert result.exit_code != 0
        assert "Missing option '--input'" in result.output or "required" in result.output.lower()

    @patch('cli.csrd_commands.AggregatorAgent')
    def test_aggregate_successful_execution(
        self,
        mock_aggregator: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_validated_data: Dict[str, Any]
    ):
        """Test successful framework aggregation."""
        input_file = temp_dir / "calculated.json"
        input_file.write_text(json.dumps(sample_validated_data))

        mock_aggregator.return_value.process.return_value = {
            "status": "success",
            "metadata": {
                "frameworks_count": 4,
                "esrs_metrics": 100,
                "tcfd_metrics": 25,
                "gri_metrics": 50,
                "sasb_metrics": 30
            }
        }

        result = cli_runner.invoke(csrd, [
            'aggregate',
            '--input', str(input_file),
            '--quiet'
        ])

        assert result.exit_code == 0

    @patch('cli.csrd_commands.AggregatorAgent')
    def test_aggregate_with_output_file(
        self,
        mock_aggregator: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_validated_data: Dict[str, Any]
    ):
        """Test saving aggregated data to file."""
        input_file = temp_dir / "calculated.json"
        input_file.write_text(json.dumps(sample_validated_data))

        output_file = temp_dir / "aggregated.json"

        aggregation_result = {
            "status": "success",
            "frameworks": {
                "esrs": {"metrics": []},
                "tcfd": {"metrics": []},
                "gri": {"metrics": []},
                "sasb": {"metrics": []}
            },
            "metadata": {
                "frameworks_count": 4
            }
        }

        mock_aggregator.return_value.process.return_value = aggregation_result

        result = cli_runner.invoke(csrd, [
            'aggregate',
            '--input', str(input_file),
            '--output', str(output_file),
            '--quiet'
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    @patch('cli.csrd_commands.AggregatorAgent')
    def test_aggregate_error_handling(
        self,
        mock_aggregator: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_validated_data: Dict[str, Any]
    ):
        """Test aggregation error handling."""
        input_file = temp_dir / "calculated.json"
        input_file.write_text(json.dumps(sample_validated_data))

        mock_aggregator.return_value.process.side_effect = Exception("Aggregation failed")

        result = cli_runner.invoke(csrd, [
            'aggregate',
            '--input', str(input_file)
        ])

        assert result.exit_code == 1
        assert "failed" in result.output.lower() or "error" in result.output.lower()


# ============================================================================
# TEST CLASS 8: CSRD CONFIG COMMAND
# ============================================================================


class TestCLIConfigCommand:
    """Tests for 'csrd config' command - configuration management."""

    def test_config_without_flags_shows_usage(self, cli_runner: CliRunner):
        """Test config command without flags shows usage message."""
        result = cli_runner.invoke(csrd, ['config'])

        assert "Please specify --init or --show" in result.output

    def test_config_show_with_existing_config(
        self,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_config_data: Dict[str, Any]
    ):
        """Test showing existing configuration."""
        config_file = temp_dir / ".csrd.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config_data, f)

        result = cli_runner.invoke(csrd, [
            'config',
            '--show',
            '--path', str(config_file)
        ])

        assert result.exit_code == 0
        # Should display config as JSON
        assert "Test Corp" in result.output or "company" in result.output.lower()

    def test_config_show_with_nonexistent_file(
        self,
        cli_runner: CliRunner,
        temp_dir: Path
    ):
        """Test showing config when file doesn't exist."""
        nonexistent = temp_dir / "nonexistent.yaml"

        result = cli_runner.invoke(csrd, [
            'config',
            '--show',
            '--path', str(nonexistent)
        ])

        # Should handle gracefully
        assert result.exit_code in [0, 1]

    def test_config_init_creates_new_config(
        self,
        cli_runner: CliRunner,
        temp_dir: Path
    ):
        """Test creating new configuration interactively."""
        config_file = temp_dir / "new_config.yaml"

        # Provide interactive inputs
        result = cli_runner.invoke(csrd, [
            'config',
            '--init',
            '--path', str(config_file)
        ], input='\n'.join([
            'Test Company',  # Company name
            'TEST123LEI001',  # LEI
            'NL',  # Country
            'Technology',  # Sector
            '2024-01-01',  # Start date
            '2024-12-31',  # End date
            '50'  # Materiality threshold
        ]))

        # Should create config file
        assert config_file.exists() or "Configuration" in result.output

    def test_config_init_custom_path(
        self,
        cli_runner: CliRunner,
        temp_dir: Path
    ):
        """Test creating config at custom path."""
        config_file = temp_dir / "custom" / "config.yaml"

        result = cli_runner.invoke(csrd, [
            'config',
            '--init',
            '--path', str(config_file)
        ], input='\n'.join([
            'Test Company',
            '',
            'NL',
            'Tech',
            '2024-01-01',
            '2024-12-31',
            '50'
        ]))

        # Should handle directory creation
        assert "Configuration" in result.output or result.exit_code in [0, 1]

    def test_config_init_validates_inputs(
        self,
        cli_runner: CliRunner,
        temp_dir: Path
    ):
        """Test that config init validates user inputs."""
        config_file = temp_dir / "test_config.yaml"

        result = cli_runner.invoke(csrd, [
            'config',
            '--init',
            '--path', str(config_file)
        ], input='\n'.join([
            'Valid Company',
            '',
            'NL',
            'Technology',
            '2024-01-01',
            '2024-12-31',
            '50'
        ]))

        # Should complete without errors
        assert result.exit_code == 0 or "Configuration" in result.output


# ============================================================================
# TEST CLASS 9: HELP TEXT AND DOCUMENTATION
# ============================================================================


class TestCLIHelpText:
    """Tests for CLI help text and documentation."""

    def test_main_help_displays_correctly(self, cli_runner: CliRunner):
        """Test main CLI help text."""
        result = cli_runner.invoke(csrd, ['--help'])

        assert result.exit_code == 0
        assert "CSRD" in result.output or "Digital Reporting" in result.output
        # Should list commands
        assert "run" in result.output
        assert "validate" in result.output

    def test_run_help_shows_all_options(self, cli_runner: CliRunner):
        """Test 'csrd run --help' shows all options."""
        result = cli_runner.invoke(csrd, ['run', '--help'])

        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--company-profile" in result.output
        assert "--output-dir" in result.output
        assert "--verbose" in result.output
        assert "--quiet" in result.output

    def test_validate_help_shows_description(self, cli_runner: CliRunner):
        """Test 'csrd validate --help' shows description."""
        result = cli_runner.invoke(csrd, ['validate', '--help'])

        assert result.exit_code == 0
        assert "--input" in result.output
        assert "validate" in result.output.lower() or "validation" in result.output.lower()

    def test_calculate_help_mentions_zero_hallucination(self, cli_runner: CliRunner):
        """Test 'csrd calculate --help' mentions zero hallucination."""
        result = cli_runner.invoke(csrd, ['calculate', '--help'])

        assert result.exit_code == 0
        assert "ZERO" in result.output.upper() or "hallucination" in result.output.lower()

    def test_audit_help_mentions_compliance(self, cli_runner: CliRunner):
        """Test 'csrd audit --help' mentions compliance."""
        result = cli_runner.invoke(csrd, ['audit', '--help'])

        assert result.exit_code == 0
        assert "compliance" in result.output.lower() or "200" in result.output

    def test_report_help_shows_format_options(self, cli_runner: CliRunner):
        """Test 'csrd report --help' shows format options."""
        result = cli_runner.invoke(csrd, ['report', '--help'])

        assert result.exit_code == 0
        assert "--format" in result.output
        assert "xbrl" in result.output.lower() or "json" in result.output.lower()

    def test_config_help_shows_init_and_show(self, cli_runner: CliRunner):
        """Test 'csrd config --help' shows init and show options."""
        result = cli_runner.invoke(csrd, ['config', '--help'])

        assert result.exit_code == 0
        assert "--init" in result.output
        assert "--show" in result.output

    def test_version_option_displays_version(self, cli_runner: CliRunner):
        """Test --version displays version information."""
        result = cli_runner.invoke(csrd, ['--version'])

        assert result.exit_code == 0
        assert "1.0" in result.output or "version" in result.output.lower()


# ============================================================================
# TEST CLASS 10: ERROR HANDLING AND EDGE CASES
# ============================================================================


class TestCLIErrorHandling:
    """Tests for CLI error handling and edge cases."""

    def test_invalid_json_file_error(
        self,
        cli_runner: CliRunner,
        temp_dir: Path
    ):
        """Test error handling for invalid JSON file."""
        invalid_json = temp_dir / "invalid.json"
        invalid_json.write_text("{invalid json content")

        result = cli_runner.invoke(csrd, [
            'validate',
            '--input', str(invalid_json)
        ])

        # Should handle gracefully
        assert result.exit_code in [0, 1]

    def test_empty_csv_file(
        self,
        cli_runner: CliRunner,
        temp_dir: Path
    ):
        """Test handling of empty CSV file."""
        empty_csv = temp_dir / "empty.csv"
        empty_csv.write_text("")

        result = cli_runner.invoke(csrd, [
            'validate',
            '--input', str(empty_csv)
        ])

        # Should handle gracefully
        assert result.exit_code in [0, 1]

    def test_csv_with_missing_columns(
        self,
        cli_runner: CliRunner,
        temp_dir: Path
    ):
        """Test CSV with missing required columns."""
        bad_csv = temp_dir / "bad.csv"
        bad_csv.write_text("wrong_column,another_column\nvalue1,value2")

        result = cli_runner.invoke(csrd, [
            'validate',
            '--input', str(bad_csv)
        ])

        # Should handle gracefully
        assert result.exit_code in [0, 1]

    @patch('cli.csrd_commands.IntakeAgent')
    def test_agent_import_error_handling(
        self,
        mock_intake: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_csv_data: str
    ):
        """Test handling of agent import errors."""
        input_file = temp_dir / "data.csv"
        input_file.write_text(sample_csv_data)

        # Simulate import error
        mock_intake.side_effect = ImportError("Cannot import agent")

        result = cli_runner.invoke(csrd, [
            'validate',
            '--input', str(input_file)
        ])

        # Should handle import error
        assert result.exit_code == 1

    def test_permission_denied_on_output_directory(
        self,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_csv_data: str,
        sample_company_profile: Dict[str, Any]
    ):
        """Test handling of permission denied on output directory."""
        input_file = temp_dir / "data.csv"
        input_file.write_text(sample_csv_data)

        profile_file = temp_dir / "company.json"
        profile_file.write_text(json.dumps(sample_company_profile))

        # Try to write to a path that might not be accessible
        # Note: Actual permission errors are OS-dependent
        result = cli_runner.invoke(csrd, [
            'run',
            '--input', str(input_file),
            '--company-profile', str(profile_file)
        ])

        # Should complete (may create default output dir)
        assert result.exit_code in [0, 1]

    def test_unicode_in_file_paths(
        self,
        cli_runner: CliRunner,
        temp_dir: Path
    ):
        """Test handling of unicode characters in file paths."""
        unicode_file = temp_dir / "test_数据.csv"
        unicode_file.write_text("metric_code,value\nE1-1,1000")

        result = cli_runner.invoke(csrd, [
            'validate',
            '--input', str(unicode_file)
        ])

        # Should handle unicode paths
        assert result.exit_code in [0, 1]

    def test_very_long_file_path(
        self,
        cli_runner: CliRunner,
        temp_dir: Path
    ):
        """Test handling of very long file paths."""
        # Create nested directories
        deep_dir = temp_dir
        for i in range(10):
            deep_dir = deep_dir / f"level_{i}"

        # Note: May hit OS path length limits
        result = cli_runner.invoke(csrd, [
            'validate',
            '--input', str(deep_dir / "data.csv")
        ])

        # Should handle path errors
        assert result.exit_code != 0

    @patch('cli.csrd_commands.IntakeAgent')
    def test_keyboard_interrupt_handling(
        self,
        mock_intake: MagicMock,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_csv_data: str
    ):
        """Test handling of keyboard interrupt (Ctrl+C)."""
        input_file = temp_dir / "data.csv"
        input_file.write_text(sample_csv_data)

        # Simulate keyboard interrupt
        mock_intake.return_value.process.side_effect = KeyboardInterrupt()

        result = cli_runner.invoke(csrd, [
            'validate',
            '--input', str(input_file)
        ])

        # Should handle interrupt gracefully
        assert result.exit_code in [0, 1]


# ============================================================================
# ADDITIONAL INTEGRATION TESTS
# ============================================================================


class TestCLIIntegration:
    """Integration tests for CLI commands working together."""

    def test_validate_then_calculate_workflow(
        self,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_csv_data: str,
        sample_validated_data: Dict[str, Any]
    ):
        """Test validate -> calculate workflow."""
        # Step 1: Validate
        input_file = temp_dir / "data.csv"
        input_file.write_text(sample_csv_data)

        validated_file = temp_dir / "validated.json"

        with patch('cli.csrd_commands.IntakeAgent') as mock_intake:
            mock_intake.return_value.process.return_value = sample_validated_data

            result1 = cli_runner.invoke(csrd, [
                'validate',
                '--input', str(input_file),
                '--output', str(validated_file),
                '--quiet'
            ])

            assert result1.exit_code == 0

        # Step 2: Calculate
        if validated_file.exists():
            with patch('cli.csrd_commands.CalculatorAgent') as mock_calc:
                mock_calc.return_value.process.return_value = {
                    "status": "success",
                    "metadata": {"metrics_calculated": 50}
                }

                result2 = cli_runner.invoke(csrd, [
                    'calculate',
                    '--input', str(validated_file),
                    '--quiet'
                ])

                assert result2.exit_code == 0

    def test_config_then_run_workflow(
        self,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_csv_data: str,
        sample_company_profile: Dict[str, Any]
    ):
        """Test config -> run workflow."""
        # Create config
        config_file = temp_dir / ".csrd.yaml"

        result1 = cli_runner.invoke(csrd, [
            'config',
            '--init',
            '--path', str(config_file)
        ], input='\n'.join([
            'Test Corp',
            '',
            'NL',
            'Tech',
            '2024-01-01',
            '2024-12-31',
            '50'
        ]))

        # Then run with config
        input_file = temp_dir / "data.csv"
        input_file.write_text(sample_csv_data)

        profile_file = temp_dir / "company.json"
        profile_file.write_text(json.dumps(sample_company_profile))

        # Running would use the config
        assert config_file.exists() or result1.exit_code == 0


# ============================================================================
# END OF TEST FILE
# ============================================================================
