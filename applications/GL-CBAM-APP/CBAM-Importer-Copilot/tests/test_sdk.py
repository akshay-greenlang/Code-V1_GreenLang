# -*- coding: utf-8 -*-
"""
CBAM Importer Copilot - SDK Tests

Unit tests for Python SDK functionality.

Version: 1.0.0
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sdk.cbam_sdk import (
    cbam_build_report,
    cbam_validate_shipments,
    cbam_calculate_emissions,
    CBAMConfig,
    CBAMReport
)


# ============================================================================
# Test cbam_build_report() - Main SDK Function
# ============================================================================

@pytest.mark.unit
class TestCBAMBuildReport:
    """Test main SDK function cbam_build_report()."""

    def test_builds_report_from_csv(self, sample_shipments_csv, cbam_config):
        """Test builds report from CSV file."""
        report = cbam_build_report(
            input_file=sample_shipments_csv,
            config=cbam_config,
            save_output=False
        )

        assert report is not None
        assert isinstance(report, CBAMReport)
        assert report.total_emissions_tco2 > 0

    def test_builds_report_from_excel(self, sample_shipments_excel, cbam_config):
        """Test builds report from Excel file."""
        report = cbam_build_report(
            input_file=sample_shipments_excel,
            config=cbam_config,
            save_output=False
        )

        assert report is not None
        assert report.total_emissions_tco2 > 0

    def test_builds_report_from_json(self, sample_shipments_json, cbam_config):
        """Test builds report from JSON file."""
        report = cbam_build_report(
            input_file=sample_shipments_json,
            config=cbam_config,
            save_output=False
        )

        assert report is not None
        assert report.total_emissions_tco2 > 0

    def test_builds_report_from_dataframe(self, sample_shipments_dataframe, cbam_config):
        """Test builds report from pandas DataFrame."""
        report = cbam_build_report(
            input_dataframe=sample_shipments_dataframe,
            config=cbam_config,
            save_output=False
        )

        assert report is not None
        assert isinstance(report, CBAMReport)

    def test_works_without_config_object(self, sample_shipments_csv):
        """Test works with individual parameters instead of config."""
        report = cbam_build_report(
            input_file=sample_shipments_csv,
            importer_name="Test Company",
            importer_country="NL",
            importer_eori="NL123456789012",
            save_output=False
        )

        assert report is not None
        assert report.total_emissions_tco2 > 0

    def test_saves_output_when_requested(self, sample_shipments_csv, cbam_config, test_output_dir):
        """Test saves output file when save_output=True."""
        output_path = f"{test_output_dir}/sdk_report.json"

        report = cbam_build_report(
            input_file=sample_shipments_csv,
            config=cbam_config,
            output_path=output_path,
            save_output=True
        )

        assert Path(output_path).exists()

    def test_returns_cbam_report_object(self, sample_shipments_csv, cbam_config):
        """Test returns CBAMReport object with properties."""
        report = cbam_build_report(
            input_file=sample_shipments_csv,
            config=cbam_config,
            save_output=False
        )

        # Should have convenient properties
        assert hasattr(report, 'total_emissions_tco2')
        assert hasattr(report, 'total_shipments')
        assert hasattr(report, 'unique_cn_codes')
        assert hasattr(report, 'is_valid')

    def test_raises_error_without_input(self, cbam_config):
        """Test raises error when no input provided."""
        with pytest.raises(ValueError):
            cbam_build_report(config=cbam_config, save_output=False)

    def test_raises_error_without_importer_info(self, sample_shipments_csv):
        """Test raises error when importer info missing."""
        with pytest.raises(ValueError):
            cbam_build_report(input_file=sample_shipments_csv, save_output=False)


# ============================================================================
# Test CBAMConfig Class
# ============================================================================

@pytest.mark.unit
class TestCBAMConfig:
    """Test CBAMConfig dataclass."""

    def test_creates_config_with_required_fields(self):
        """Test creates config with required fields."""
        config = CBAMConfig(
            importer_name="Test Company",
            importer_country="NL",
            importer_eori="NL123456789012"
        )

        assert config.importer_name == "Test Company"
        assert config.importer_country == "NL"
        assert config.importer_eori == "NL123456789012"

    def test_creates_config_with_optional_fields(self):
        """Test creates config with optional fields."""
        config = CBAMConfig(
            importer_name="Test Company",
            importer_country="NL",
            importer_eori="NL123456789012",
            declarant_name="John Smith",
            declarant_position="Compliance Officer"
        )

        assert config.declarant_name == "John Smith"
        assert config.declarant_position == "Compliance Officer"

    def test_config_has_default_paths(self):
        """Test config has sensible default paths."""
        config = CBAMConfig(
            importer_name="Test",
            importer_country="NL",
            importer_eori="NL123456789012"
        )

        assert config.cn_codes_path == "data/cn_codes.json"
        assert config.cbam_rules_path == "rules/cbam_rules.yaml"

    def test_config_from_yaml(self, tmp_path):
        """Test creates config from YAML file."""
        yaml_content = """
importer:
  name: "Test Company"
  country: "NL"
  eori: "NL123456789012"

declarant:
  name: "John Smith"
  position: "Compliance Officer"
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        config = CBAMConfig.from_yaml(str(yaml_file))

        assert config.importer_name == "Test Company"
        assert config.importer_country == "NL"
        assert config.declarant_name == "John Smith"


# ============================================================================
# Test CBAMReport Class
# ============================================================================

@pytest.mark.unit
class TestCBAMReport:
    """Test CBAMReport dataclass and properties."""

    def test_report_has_total_emissions_property(self, sample_shipments_csv, cbam_config):
        """Test report has total_emissions_tco2 property."""
        report = cbam_build_report(
            input_file=sample_shipments_csv,
            config=cbam_config,
            save_output=False
        )

        emissions = report.total_emissions_tco2
        assert isinstance(emissions, (int, float))
        assert emissions > 0

    def test_report_has_total_shipments_property(self, sample_shipments_csv, cbam_config):
        """Test report has total_shipments property."""
        report = cbam_build_report(
            input_file=sample_shipments_csv,
            config=cbam_config,
            save_output=False
        )

        shipments = report.total_shipments
        assert isinstance(shipments, int)
        assert shipments == 5  # Our sample data has 5 records

    def test_report_has_unique_cn_codes_property(self, sample_shipments_csv, cbam_config):
        """Test report has unique_cn_codes property."""
        report = cbam_build_report(
            input_file=sample_shipments_csv,
            config=cbam_config,
            save_output=False
        )

        cn_codes = report.unique_cn_codes
        assert isinstance(cn_codes, int)
        assert cn_codes > 0

    def test_report_has_is_valid_property(self, sample_shipments_csv, cbam_config):
        """Test report has is_valid property."""
        report = cbam_build_report(
            input_file=sample_shipments_csv,
            config=cbam_config,
            save_output=False
        )

        is_valid = report.is_valid
        assert isinstance(is_valid, bool)

    def test_report_to_dataframe(self, sample_shipments_csv, cbam_config):
        """Test converts report to DataFrame."""
        report = cbam_build_report(
            input_file=sample_shipments_csv,
            config=cbam_config,
            save_output=False
        )

        df = report.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'cn_code' in df.columns
        assert 'embedded_emissions_tco2' in df.columns

    def test_report_save_method(self, sample_shipments_csv, cbam_config, test_output_dir):
        """Test report save() method."""
        report = cbam_build_report(
            input_file=sample_shipments_csv,
            config=cbam_config,
            save_output=False
        )

        output_path = f"{test_output_dir}/saved_report.json"
        report.save(output_path)

        assert Path(output_path).exists()

    def test_report_to_excel_method(self, sample_shipments_csv, cbam_config, test_output_dir):
        """Test report to_excel() method."""
        report = cbam_build_report(
            input_file=sample_shipments_csv,
            config=cbam_config,
            save_output=False
        )

        output_path = f"{test_output_dir}/report.xlsx"
        report.to_excel(output_path)

        assert Path(output_path).exists()


# ============================================================================
# Test cbam_validate_shipments()
# ============================================================================

@pytest.mark.unit
class TestCBAMValidateShipments:
    """Test validation function."""

    def test_validates_csv_file(self, sample_shipments_csv):
        """Test validates CSV file."""
        result = cbam_validate_shipments(
            input_file=sample_shipments_csv,
            importer_country="NL"
        )

        assert result is not None
        assert 'metadata' in result
        assert 'validated_shipments' in result

    def test_validates_dataframe(self, sample_shipments_dataframe):
        """Test validates DataFrame."""
        result = cbam_validate_shipments(
            input_dataframe=sample_shipments_dataframe,
            importer_country="NL"
        )

        assert result is not None
        assert 'metadata' in result

    def test_returns_error_count(self, sample_shipments_csv):
        """Test returns error count in metadata."""
        result = cbam_validate_shipments(
            input_file=sample_shipments_csv,
            importer_country="NL"
        )

        metadata = result['metadata']
        assert 'error_count' in metadata
        assert 'warning_count' in metadata

    def test_detects_invalid_data(self, invalid_shipments_data, tmp_path):
        """Test detects invalid data."""
        csv_path = tmp_path / "invalid.csv"
        df = pd.DataFrame(invalid_shipments_data)
        df.to_csv(csv_path, index=False)

        result = cbam_validate_shipments(
            input_file=str(csv_path),
            importer_country="NL"
        )

        # Should have errors
        error_count = result['metadata']['error_count']
        assert error_count > 0


# ============================================================================
# Test cbam_calculate_emissions()
# ============================================================================

@pytest.mark.unit
class TestCBAMCalculateEmissions:
    """Test emissions calculation function."""

    def test_calculates_emissions_from_validated_data(self, sample_shipments_csv):
        """Test calculates emissions from validated data."""
        # First validate
        validated = cbam_validate_shipments(
            input_file=sample_shipments_csv,
            importer_country="NL"
        )

        # Then calculate
        emissions = cbam_calculate_emissions(
            validated_shipments=validated
        )

        assert emissions is not None
        assert 'total_emissions_tco2' in emissions
        assert emissions['total_emissions_tco2'] > 0

    def test_returns_shipments_with_emissions(self, sample_shipments_csv):
        """Test returns shipments with emissions calculated."""
        validated = cbam_validate_shipments(
            input_file=sample_shipments_csv,
            importer_country="NL"
        )

        emissions = cbam_calculate_emissions(validated_shipments=validated)

        assert 'shipments_with_emissions' in emissions
        assert len(emissions['shipments_with_emissions']) > 0

        # Each shipment should have emissions
        for shipment in emissions['shipments_with_emissions']:
            assert 'embedded_emissions_tco2' in shipment


# ============================================================================
# Test SDK Integration
# ============================================================================

@pytest.mark.integration
class TestSDKIntegration:
    """Test SDK functions work together."""

    def test_validate_then_calculate_then_report(self, sample_shipments_csv, cbam_config):
        """Test can chain SDK functions."""
        # Step 1: Validate
        validated = cbam_validate_shipments(
            input_file=sample_shipments_csv,
            importer_country="NL"
        )

        assert validated['metadata']['error_count'] == 0

        # Step 2: Calculate
        emissions = cbam_calculate_emissions(validated_shipments=validated)

        assert emissions['total_emissions_tco2'] > 0

        # Step 3: Full report (combines both)
        report = cbam_build_report(
            input_file=sample_shipments_csv,
            config=cbam_config,
            save_output=False
        )

        assert report.total_emissions_tco2 > 0

    def test_dataframe_workflow(self, cbam_config):
        """Test complete DataFrame workflow."""
        # Create DataFrame
        df = pd.DataFrame({
            'cn_code': ['72071100', '76011000'],
            'country_of_origin': ['CN', 'RU'],
            'quantity_tons': [10.0, 15.0],
            'import_date': ['2025-09-15', '2025-09-20']
        })

        # Generate report
        report = cbam_build_report(
            input_dataframe=df,
            config=cbam_config,
            save_output=False
        )

        assert report.total_shipments == 2

        # Convert back to DataFrame
        result_df = report.to_dataframe()

        assert len(result_df) == 2
        assert 'embedded_emissions_tco2' in result_df.columns


# ============================================================================
# Test SDK Error Handling
# ============================================================================

@pytest.mark.unit
class TestSDKErrorHandling:
    """Test SDK error handling."""

    def test_handles_missing_file(self, cbam_config):
        """Test handles missing file gracefully."""
        with pytest.raises(FileNotFoundError):
            cbam_build_report(
                input_file="nonexistent.csv",
                config=cbam_config,
                save_output=False
            )

    def test_handles_invalid_config(self, sample_shipments_csv):
        """Test handles invalid configuration."""
        with pytest.raises((ValueError, AttributeError, TypeError)):
            cbam_build_report(
                input_file=sample_shipments_csv,
                config="invalid",  # Wrong type
                save_output=False
            )

    def test_handles_both_file_and_dataframe(self, sample_shipments_csv,
                                               sample_shipments_dataframe, cbam_config):
        """Test raises error when both file and DataFrame provided."""
        # Should prefer one over the other or raise error
        # Implementation dependent
        try:
            report = cbam_build_report(
                input_file=sample_shipments_csv,
                input_dataframe=sample_shipments_dataframe,
                config=cbam_config,
                save_output=False
            )
            # If successful, one was chosen
            assert report is not None
        except ValueError:
            # Or raises error - both are acceptable
            pass


# ============================================================================
# Test SDK Performance
# ============================================================================

@pytest.mark.unit
@pytest.mark.performance
class TestSDKPerformance:
    """Test SDK performance."""

    def test_sdk_builds_report_quickly(self, sample_shipments_csv, cbam_config):
        """Test SDK builds report quickly."""
        import time

        start = time.time()
        report = cbam_build_report(
            input_file=sample_shipments_csv,
            config=cbam_config,
            save_output=False
        )
        duration = time.time() - start

        # Should be fast for small dataset
        assert duration < 3.0, f"SDK too slow: {duration:.2f}s"

    def test_sdk_handles_large_dataset(self, large_shipments_csv, cbam_config):
        """Test SDK handles large dataset efficiently."""
        import time

        start = time.time()
        report = cbam_build_report(
            input_file=large_shipments_csv,
            config=cbam_config,
            save_output=False
        )
        duration = time.time() - start

        # 1000 records should be processed in reasonable time
        assert duration < 10.0, \
            f"SDK large dataset too slow: {duration:.2f}s for {report.total_shipments} records"


# ============================================================================
# Test SDK Convenience Features
# ============================================================================

@pytest.mark.unit
class TestSDKConvenience:
    """Test SDK convenience features."""

    def test_config_can_be_reused(self, sample_shipments_csv):
        """Test config object can be reused for multiple reports."""
        config = CBAMConfig(
            importer_name="Test Company",
            importer_country="NL",
            importer_eori="NL123456789012"
        )

        # Generate multiple reports with same config
        report1 = cbam_build_report(sample_shipments_csv, config=config, save_output=False)
        report2 = cbam_build_report(sample_shipments_csv, config=config, save_output=False)

        # Both should succeed
        assert report1 is not None
        assert report2 is not None

    def test_dataframe_round_trip(self, sample_shipments_csv, cbam_config):
        """Test can convert report to DataFrame and back."""
        report = cbam_build_report(sample_shipments_csv, config=cbam_config, save_output=False)

        # Convert to DataFrame
        df = report.to_dataframe()

        # Use DataFrame as input for new report
        report2 = cbam_build_report(
            input_dataframe=df[['cn_code', 'country_of_origin', 'quantity_tons', 'import_date']],
            config=cbam_config,
            save_output=False
        )

        # Should produce similar results
        assert abs(report.total_emissions_tco2 - report2.total_emissions_tco2) < 1.0
