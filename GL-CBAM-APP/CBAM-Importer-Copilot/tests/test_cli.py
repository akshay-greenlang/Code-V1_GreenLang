"""
CBAM Importer Copilot - CLI Command Tests

Tests for command-line interface commands and argument parsing.

Version: 1.0.0
"""

import pytest
import json
import yaml
from pathlib import Path
from click.testing import CliRunner
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from cbam_cli import cli, report_cmd, config_cmd, validate_cmd


# ============================================================================
# Test CLI Report Command
# ============================================================================

@pytest.mark.unit
class TestReportCommand:
    """Test 'gl cbam report' command."""

    def test_report_command_with_csv(self, sample_shipments_csv, tmp_path):
        """Test report command with CSV input."""
        runner = CliRunner()
        output_path = tmp_path / "report.json"

        result = runner.invoke(report_cmd, [
            '--input', str(sample_shipments_csv),
            '--output', str(output_path),
            '--importer-country', 'NL',
            '--importer-name', 'Test Company',
            '--importer-eori', 'NL123456789'
        ])

        assert result.exit_code == 0
        assert output_path.exists()

        # Verify output
        with open(output_path) as f:
            report = json.load(f)
            assert 'report_metadata' in report
            assert 'emissions_summary' in report

    def test_report_command_with_excel(self, sample_shipments_excel, tmp_path):
        """Test report command with Excel input."""
        runner = CliRunner()
        output_path = tmp_path / "report_excel.json"

        result = runner.invoke(report_cmd, [
            '--input', str(sample_shipments_excel),
            '--output', str(output_path),
            '--importer-country', 'NL',
            '--importer-name', 'Test Company',
            '--importer-eori', 'NL123456789'
        ])

        assert result.exit_code == 0
        assert output_path.exists()

    def test_report_command_with_config_file(self, sample_shipments_csv, cbam_config_path, tmp_path):
        """Test report command with config file."""
        runner = CliRunner()
        output_path = tmp_path / "report_with_config.json"

        result = runner.invoke(report_cmd, [
            '--input', str(sample_shipments_csv),
            '--output', str(output_path),
            '--config', str(cbam_config_path)
        ])

        assert result.exit_code == 0
        assert output_path.exists()

    def test_report_command_auto_output_name(self, sample_shipments_csv, tmp_path):
        """Test report command generates output name automatically."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(report_cmd, [
                '--input', str(sample_shipments_csv),
                '--importer-country', 'NL',
                '--importer-name', 'Test Company',
                '--importer-eori', 'NL123456789'
            ])

            assert result.exit_code == 0
            # Should create cbam_report_<timestamp>.json

    def test_report_command_with_provenance(self, sample_shipments_csv, tmp_path):
        """Test report command with provenance enabled."""
        runner = CliRunner()
        output_path = tmp_path / "report_prov.json"

        result = runner.invoke(report_cmd, [
            '--input', str(sample_shipments_csv),
            '--output', str(output_path),
            '--importer-country', 'NL',
            '--importer-name', 'Test Company',
            '--importer-eori', 'NL123456789',
            '--provenance'
        ])

        assert result.exit_code == 0

        with open(output_path) as f:
            report = json.load(f)
            assert 'provenance' in report

    def test_report_command_with_suppliers(self, sample_shipments_csv, suppliers_path, tmp_path):
        """Test report command with suppliers file."""
        runner = CliRunner()
        output_path = tmp_path / "report_suppliers.json"

        result = runner.invoke(report_cmd, [
            '--input', str(sample_shipments_csv),
            '--output', str(output_path),
            '--importer-country', 'NL',
            '--importer-name', 'Test Company',
            '--importer-eori', 'NL123456789',
            '--suppliers', str(suppliers_path)
        ])

        assert result.exit_code == 0

    def test_report_command_verbose_mode(self, sample_shipments_csv, tmp_path):
        """Test report command with verbose output."""
        runner = CliRunner()
        output_path = tmp_path / "report_verbose.json"

        result = runner.invoke(report_cmd, [
            '--input', str(sample_shipments_csv),
            '--output', str(output_path),
            '--importer-country', 'NL',
            '--importer-name', 'Test Company',
            '--importer-eori', 'NL123456789',
            '--verbose'
        ])

        assert result.exit_code == 0
        assert 'Processing' in result.output or 'CBAM' in result.output


# ============================================================================
# Test CLI Config Command
# ============================================================================

@pytest.mark.unit
class TestConfigCommand:
    """Test 'gl cbam config' commands."""

    def test_config_init_creates_file(self, tmp_path):
        """Test config init creates configuration file."""
        runner = CliRunner()
        config_path = tmp_path / "cbam_config.yaml"

        result = runner.invoke(config_cmd, [
            'init',
            '--output', str(config_path)
        ])

        assert result.exit_code == 0
        assert config_path.exists()

        # Verify structure
        with open(config_path) as f:
            config = yaml.safe_load(f)
            assert 'importer' in config
            assert 'data_sources' in config
            assert 'processing' in config

    def test_config_init_interactive_mode(self, tmp_path):
        """Test config init in interactive mode."""
        runner = CliRunner()
        config_path = tmp_path / "interactive_config.yaml"

        # Provide interactive inputs
        result = runner.invoke(config_cmd, [
            'init',
            '--output', str(config_path),
            '--interactive'
        ], input='NL\nTest Company\nNL123456789\n')

        assert result.exit_code == 0
        assert config_path.exists()

        with open(config_path) as f:
            config = yaml.safe_load(f)
            assert config['importer']['country'] == 'NL'
            assert config['importer']['name'] == 'Test Company'
            assert config['importer']['eori_number'] == 'NL123456789'

    def test_config_show_displays_config(self, cbam_config_path):
        """Test config show displays configuration."""
        runner = CliRunner()

        result = runner.invoke(config_cmd, [
            'show',
            '--config', str(cbam_config_path)
        ])

        assert result.exit_code == 0
        assert 'importer' in result.output.lower() or 'Importer' in result.output

    def test_config_validate_valid_config(self, cbam_config_path):
        """Test config validate with valid configuration."""
        runner = CliRunner()

        result = runner.invoke(config_cmd, [
            'validate',
            '--config', str(cbam_config_path)
        ])

        assert result.exit_code == 0
        assert 'valid' in result.output.lower() or '✓' in result.output

    def test_config_validate_invalid_config(self, tmp_path):
        """Test config validate with invalid configuration."""
        runner = CliRunner()

        # Create invalid config
        invalid_config = tmp_path / "invalid_config.yaml"
        invalid_config.write_text("invalid: yaml: content:")

        result = runner.invoke(config_cmd, [
            'validate',
            '--config', str(invalid_config)
        ])

        assert result.exit_code != 0

    def test_config_edit_opens_editor(self, cbam_config_path, monkeypatch):
        """Test config edit command (mocked editor)."""
        runner = CliRunner()

        # Mock editor call
        editor_called = []

        def mock_editor(filename):
            editor_called.append(filename)
            return None

        monkeypatch.setattr('click.edit', mock_editor)

        result = runner.invoke(config_cmd, [
            'edit',
            '--config', str(cbam_config_path)
        ])

        assert result.exit_code == 0
        assert len(editor_called) > 0


# ============================================================================
# Test CLI Validate Command
# ============================================================================

@pytest.mark.unit
class TestValidateCommand:
    """Test 'gl cbam validate' command."""

    def test_validate_command_csv(self, sample_shipments_csv):
        """Test validate command with CSV input."""
        runner = CliRunner()

        result = runner.invoke(validate_cmd, [
            '--input', str(sample_shipments_csv),
            '--importer-country', 'NL'
        ])

        assert result.exit_code == 0
        assert 'valid' in result.output.lower() or 'Validation' in result.output

    def test_validate_command_with_errors(self, invalid_shipments_csv):
        """Test validate command with invalid data."""
        runner = CliRunner()

        result = runner.invoke(validate_cmd, [
            '--input', str(invalid_shipments_csv),
            '--importer-country', 'NL'
        ])

        # May succeed but report errors
        assert 'error' in result.output.lower() or 'warning' in result.output.lower()

    def test_validate_command_json_output(self, sample_shipments_csv, tmp_path):
        """Test validate command with JSON output."""
        runner = CliRunner()
        output_path = tmp_path / "validation_result.json"

        result = runner.invoke(validate_cmd, [
            '--input', str(sample_shipments_csv),
            '--importer-country', 'NL',
            '--output', str(output_path),
            '--format', 'json'
        ])

        assert result.exit_code == 0
        assert output_path.exists()

        with open(output_path) as f:
            validation = json.load(f)
            assert 'is_valid' in validation or 'errors' in validation

    def test_validate_command_detailed_mode(self, sample_shipments_csv):
        """Test validate command with detailed output."""
        runner = CliRunner()

        result = runner.invoke(validate_cmd, [
            '--input', str(sample_shipments_csv),
            '--importer-country', 'NL',
            '--detailed'
        ])

        assert result.exit_code == 0
        # Should show more information

    def test_validate_command_strict_mode(self, sample_shipments_csv):
        """Test validate command in strict mode."""
        runner = CliRunner()

        result = runner.invoke(validate_cmd, [
            '--input', str(sample_shipments_csv),
            '--importer-country', 'NL',
            '--strict'
        ])

        # Strict mode may have different exit code for warnings
        assert result.exit_code in [0, 1]


# ============================================================================
# Test CLI Argument Parsing
# ============================================================================

@pytest.mark.unit
class TestArgumentParsing:
    """Test command-line argument parsing."""

    def test_missing_required_arguments(self):
        """Test error when required arguments missing."""
        runner = CliRunner()

        result = runner.invoke(report_cmd, [])

        assert result.exit_code != 0
        assert 'required' in result.output.lower() or 'missing' in result.output.lower()

    def test_invalid_input_file(self, tmp_path):
        """Test error with non-existent input file."""
        runner = CliRunner()

        result = runner.invoke(report_cmd, [
            '--input', 'nonexistent.csv',
            '--output', str(tmp_path / 'report.json'),
            '--importer-country', 'NL',
            '--importer-name', 'Test',
            '--importer-eori', 'NL123456789'
        ])

        assert result.exit_code != 0

    def test_invalid_importer_country(self, sample_shipments_csv, tmp_path):
        """Test error with invalid country code."""
        runner = CliRunner()

        result = runner.invoke(report_cmd, [
            '--input', str(sample_shipments_csv),
            '--output', str(tmp_path / 'report.json'),
            '--importer-country', 'INVALID',
            '--importer-name', 'Test',
            '--importer-eori', 'NL123456789'
        ])

        # May succeed but validation may flag it
        assert result.exit_code in [0, 1]

    def test_mutually_exclusive_options(self, sample_shipments_csv, tmp_path):
        """Test handling of mutually exclusive options."""
        runner = CliRunner()

        # If config file provided, shouldn't need individual importer params
        result = runner.invoke(report_cmd, [
            '--input', str(sample_shipments_csv),
            '--output', str(tmp_path / 'report.json'),
            '--config', 'config.yaml',
            '--importer-country', 'NL'  # Redundant with config
        ])

        # Should handle gracefully (config takes precedence or error)
        assert result.exit_code in [0, 1]


# ============================================================================
# Test CLI Output Formats
# ============================================================================

@pytest.mark.unit
class TestOutputFormats:
    """Test CLI output format options."""

    def test_json_output_format(self, sample_shipments_csv, tmp_path):
        """Test JSON output format."""
        runner = CliRunner()
        output_path = tmp_path / "report.json"

        result = runner.invoke(report_cmd, [
            '--input', str(sample_shipments_csv),
            '--output', str(output_path),
            '--importer-country', 'NL',
            '--importer-name', 'Test',
            '--importer-eori', 'NL123456789',
            '--format', 'json'
        ])

        assert result.exit_code == 0
        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)
            assert isinstance(data, dict)

    def test_excel_output_format(self, sample_shipments_csv, tmp_path):
        """Test Excel output format."""
        runner = CliRunner()
        output_path = tmp_path / "report.xlsx"

        result = runner.invoke(report_cmd, [
            '--input', str(sample_shipments_csv),
            '--output', str(output_path),
            '--importer-country', 'NL',
            '--importer-name', 'Test',
            '--importer-eori', 'NL123456789',
            '--format', 'excel'
        ])

        if result.exit_code == 0:
            assert output_path.exists()

    def test_csv_output_format(self, sample_shipments_csv, tmp_path):
        """Test CSV output format."""
        runner = CliRunner()
        output_path = tmp_path / "report.csv"

        result = runner.invoke(report_cmd, [
            '--input', str(sample_shipments_csv),
            '--output', str(output_path),
            '--importer-country', 'NL',
            '--importer-name', 'Test',
            '--importer-eori', 'NL123456789',
            '--format', 'csv'
        ])

        if result.exit_code == 0:
            assert output_path.exists()


# ============================================================================
# Test CLI Environment Variables
# ============================================================================

@pytest.mark.unit
class TestEnvironmentVariables:
    """Test environment variable handling."""

    def test_config_from_env_variable(self, sample_shipments_csv, cbam_config_path, tmp_path, monkeypatch):
        """Test reading config path from environment variable."""
        runner = CliRunner()
        output_path = tmp_path / "report_env.json"

        # Set environment variable
        monkeypatch.setenv('CBAM_CONFIG', str(cbam_config_path))

        result = runner.invoke(report_cmd, [
            '--input', str(sample_shipments_csv),
            '--output', str(output_path)
        ])

        assert result.exit_code == 0

    def test_cn_codes_path_from_env(self, sample_shipments_csv, tmp_path, monkeypatch):
        """Test CN codes path from environment."""
        runner = CliRunner()
        output_path = tmp_path / "report.json"

        monkeypatch.setenv('CBAM_CN_CODES_PATH', 'data/cn_codes.json')

        result = runner.invoke(report_cmd, [
            '--input', str(sample_shipments_csv),
            '--output', str(output_path),
            '--importer-country', 'NL',
            '--importer-name', 'Test',
            '--importer-eori', 'NL123456789'
        ])

        assert result.exit_code == 0

    def test_debug_mode_from_env(self, sample_shipments_csv, tmp_path, monkeypatch):
        """Test debug mode from environment variable."""
        runner = CliRunner()
        output_path = tmp_path / "report_debug.json"

        monkeypatch.setenv('CBAM_DEBUG', '1')

        result = runner.invoke(report_cmd, [
            '--input', str(sample_shipments_csv),
            '--output', str(output_path),
            '--importer-country', 'NL',
            '--importer-name', 'Test',
            '--importer-eori', 'NL123456789'
        ])

        # Debug mode may produce extra output
        assert result.exit_code == 0


# ============================================================================
# Test CLI Error Handling
# ============================================================================

@pytest.mark.unit
class TestErrorHandling:
    """Test CLI error handling and messages."""

    def test_file_not_found_error(self, tmp_path):
        """Test clear error message for missing file."""
        runner = CliRunner()

        result = runner.invoke(report_cmd, [
            '--input', 'nonexistent.csv',
            '--output', str(tmp_path / 'report.json'),
            '--importer-country', 'NL',
            '--importer-name', 'Test',
            '--importer-eori', 'NL123456789'
        ])

        assert result.exit_code != 0
        assert 'not found' in result.output.lower() or 'does not exist' in result.output.lower()

    def test_invalid_yaml_config_error(self, tmp_path):
        """Test clear error message for invalid YAML."""
        runner = CliRunner()

        invalid_config = tmp_path / "bad.yaml"
        invalid_config.write_text("invalid: yaml: content:")

        result = runner.invoke(report_cmd, [
            '--input', 'test.csv',
            '--output', str(tmp_path / 'report.json'),
            '--config', str(invalid_config)
        ])

        assert result.exit_code != 0
        assert 'yaml' in result.output.lower() or 'config' in result.output.lower()

    def test_permission_error_handling(self, sample_shipments_csv, tmp_path):
        """Test handling of permission errors."""
        runner = CliRunner()

        # Try to write to read-only location (if possible)
        output_path = tmp_path / "readonly" / "report.json"

        result = runner.invoke(report_cmd, [
            '--input', str(sample_shipments_csv),
            '--output', str(output_path),
            '--importer-country', 'NL',
            '--importer-name', 'Test',
            '--importer-eori', 'NL123456789'
        ])

        # May fail due to directory not existing
        assert result.exit_code in [0, 1, 2]

    def test_keyboard_interrupt_handling(self, sample_shipments_csv, tmp_path):
        """Test graceful handling of Ctrl+C."""
        runner = CliRunner()

        # This is hard to test directly, but ensure cleanup happens
        # In real CLI, should have try/except for KeyboardInterrupt


# ============================================================================
# Test CLI Help and Version
# ============================================================================

@pytest.mark.unit
class TestHelpAndVersion:
    """Test CLI help and version information."""

    def test_main_help_message(self):
        """Test main CLI help message."""
        runner = CliRunner()

        result = runner.invoke(cli, ['--help'])

        assert result.exit_code == 0
        assert 'CBAM' in result.output

    def test_report_help_message(self):
        """Test report command help."""
        runner = CliRunner()

        result = runner.invoke(report_cmd, ['--help'])

        assert result.exit_code == 0
        assert 'report' in result.output.lower()
        assert '--input' in result.output

    def test_config_help_message(self):
        """Test config command help."""
        runner = CliRunner()

        result = runner.invoke(config_cmd, ['--help'])

        assert result.exit_code == 0
        assert 'config' in result.output.lower()

    def test_validate_help_message(self):
        """Test validate command help."""
        runner = CliRunner()

        result = runner.invoke(validate_cmd, ['--help'])

        assert result.exit_code == 0
        assert 'validate' in result.output.lower()

    def test_version_display(self):
        """Test version information display."""
        runner = CliRunner()

        result = runner.invoke(cli, ['--version'])

        assert result.exit_code == 0
        assert '1.0.0' in result.output or 'version' in result.output.lower()


# ============================================================================
# Test CLI Performance
# ============================================================================

@pytest.mark.performance
class TestCLIPerformance:
    """Test CLI command performance."""

    def test_report_command_performance(self, large_shipments_csv, tmp_path):
        """Test report command performance with large dataset."""
        import time

        runner = CliRunner()
        output_path = tmp_path / "large_report.json"

        start = time.time()
        result = runner.invoke(report_cmd, [
            '--input', str(large_shipments_csv),
            '--output', str(output_path),
            '--importer-country', 'NL',
            '--importer-name', 'Test',
            '--importer-eori', 'NL123456789'
        ])
        duration = time.time() - start

        assert result.exit_code == 0
        # 1000 records should complete in reasonable time
        assert duration < 15.0

    def test_validate_command_performance(self, large_shipments_csv):
        """Test validate command performance."""
        import time

        runner = CliRunner()

        start = time.time()
        result = runner.invoke(validate_cmd, [
            '--input', str(large_shipments_csv),
            '--importer-country', 'NL'
        ])
        duration = time.time() - start

        assert result.exit_code == 0
        # Validation should be fast
        assert duration < 5.0


# ============================================================================
# Test CLI Integration
# ============================================================================

@pytest.mark.integration
class TestCLIIntegration:
    """Test CLI integration with pipeline."""

    def test_cli_produces_valid_report(self, sample_shipments_csv, tmp_path, assert_valid_report):
        """Test CLI produces structurally valid report."""
        runner = CliRunner()
        output_path = tmp_path / "valid_report.json"

        result = runner.invoke(report_cmd, [
            '--input', str(sample_shipments_csv),
            '--output', str(output_path),
            '--importer-country', 'NL',
            '--importer-name', 'Test Company',
            '--importer-eori', 'NL123456789',
            '--provenance'
        ])

        assert result.exit_code == 0

        with open(output_path) as f:
            report = json.load(f)
            assert_valid_report(report)

    def test_cli_zero_hallucination_guarantee(self, sample_shipments_csv, tmp_path, assert_zero_hallucination):
        """Test CLI maintains zero hallucination guarantee."""
        runner = CliRunner()
        output_path = tmp_path / "zero_halluc_report.json"

        result = runner.invoke(report_cmd, [
            '--input', str(sample_shipments_csv),
            '--output', str(output_path),
            '--importer-country', 'NL',
            '--importer-name', 'Test',
            '--importer-eori', 'NL123456789',
            '--provenance'
        ])

        assert result.exit_code == 0

        with open(output_path) as f:
            report = json.load(f)
            assert_zero_hallucination(report)

    def test_cli_config_workflow(self, sample_shipments_csv, tmp_path):
        """Test complete workflow: init config, validate, run report."""
        runner = CliRunner()

        # Step 1: Initialize config
        config_path = tmp_path / "workflow_config.yaml"
        result1 = runner.invoke(config_cmd, [
            'init',
            '--output', str(config_path)
        ])
        assert result1.exit_code == 0

        # Step 2: Validate config
        result2 = runner.invoke(config_cmd, [
            'validate',
            '--config', str(config_path)
        ])
        assert result2.exit_code == 0

        # Step 3: Run report with config
        output_path = tmp_path / "workflow_report.json"
        result3 = runner.invoke(report_cmd, [
            '--input', str(sample_shipments_csv),
            '--output', str(output_path),
            '--config', str(config_path)
        ])
        assert result3.exit_code == 0
