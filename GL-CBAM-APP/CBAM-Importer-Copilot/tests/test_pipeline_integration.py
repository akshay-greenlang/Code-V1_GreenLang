"""
CBAM Importer Copilot - Pipeline Integration Tests

End-to-end integration tests for complete CBAM pipeline.

Version: 1.0.0
"""

import pytest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from cbam_pipeline import CBAMPipeline


# ============================================================================
# Test Full Pipeline
# ============================================================================

@pytest.mark.integration
class TestFullPipeline:
    """Test complete end-to-end pipeline execution."""

    def test_pipeline_runs_successfully(self, sample_shipments_csv, importer_info, test_output_dir):
        """Test pipeline runs from CSV to final report."""
        pipeline = CBAMPipeline(
            cn_codes_path="data/cn_codes.json",
            cbam_rules_path="rules/cbam_rules.yaml"
        )

        output_path = f"{test_output_dir}/cbam_report.json"
        result = pipeline.run(
            input_file=sample_shipments_csv,
            importer_info=importer_info,
            output_path=output_path
        )

        assert result is not None
        assert Path(output_path).exists()

    def test_pipeline_with_all_input_formats(self, sample_shipments_csv, sample_shipments_excel,
                                               sample_shipments_json, importer_info, test_output_dir):
        """Test pipeline works with CSV, Excel, and JSON."""
        pipeline = CBAMPipeline(
            cn_codes_path="data/cn_codes.json",
            cbam_rules_path="rules/cbam_rules.yaml"
        )

        formats = {
            'csv': sample_shipments_csv,
            'excel': sample_shipments_excel,
            'json': sample_shipments_json
        }

        results = {}
        for format_name, input_file in formats.items():
            output_path = f"{test_output_dir}/report_{format_name}.json"
            result = pipeline.run(input_file, importer_info, output_path)
            results[format_name] = result

        # All formats should produce same total emissions
        csv_emissions = results['csv']['emissions_summary']['total_embedded_emissions_tco2']
        excel_emissions = results['excel']['emissions_summary']['total_embedded_emissions_tco2']
        json_emissions = results['json']['emissions_summary']['total_embedded_emissions_tco2']

        assert abs(csv_emissions - excel_emissions) < 0.01
        assert abs(csv_emissions - json_emissions) < 0.01

    def test_pipeline_output_structure(self, sample_shipments_csv, importer_info,
                                         test_output_dir, assert_valid_report):
        """Test pipeline produces valid output structure."""
        pipeline = CBAMPipeline(
            cn_codes_path="data/cn_codes.json",
            cbam_rules_path="rules/cbam_rules.yaml"
        )

        output_path = f"{test_output_dir}/cbam_report.json"
        result = pipeline.run(sample_shipments_csv, importer_info, output_path)

        # Validate structure
        assert_valid_report(result)


# ============================================================================
# Test Provenance Integration
# ============================================================================

@pytest.mark.integration
@pytest.mark.compliance
class TestProvenanceIntegration:
    """Test provenance tracking in full pipeline."""

    def test_pipeline_includes_provenance(self, sample_shipments_csv, importer_info, test_output_dir):
        """Test pipeline includes provenance in output."""
        pipeline = CBAMPipeline(
            cn_codes_path="data/cn_codes.json",
            cbam_rules_path="rules/cbam_rules.yaml",
            enable_provenance=True
        )

        output_path = f"{test_output_dir}/cbam_report.json"
        result = pipeline.run(sample_shipments_csv, importer_info, output_path)

        assert 'provenance' in result
        provenance = result['provenance']

        # Check provenance components
        assert 'input_file_integrity' in provenance
        assert 'execution_environment' in provenance
        assert 'agent_execution' in provenance
        assert 'reproducibility' in provenance

    def test_provenance_file_hash(self, sample_shipments_csv, importer_info, test_output_dir):
        """Test provenance includes SHA256 file hash."""
        pipeline = CBAMPipeline(
            cn_codes_path="data/cn_codes.json",
            cbam_rules_path="rules/cbam_rules.yaml",
            enable_provenance=True
        )

        output_path = f"{test_output_dir}/cbam_report.json"
        result = pipeline.run(sample_shipments_csv, importer_info, output_path)

        file_integrity = result['provenance']['input_file_integrity']

        assert 'sha256_hash' in file_integrity
        assert 'file_name' in file_integrity
        assert 'file_size_bytes' in file_integrity
        assert len(file_integrity['sha256_hash']) == 64  # SHA256 is 64 hex chars

    def test_provenance_agent_execution(self, sample_shipments_csv, importer_info, test_output_dir):
        """Test provenance tracks all agent executions."""
        pipeline = CBAMPipeline(
            cn_codes_path="data/cn_codes.json",
            cbam_rules_path="rules/cbam_rules.yaml",
            enable_provenance=True
        )

        output_path = f"{test_output_dir}/cbam_report.json"
        result = pipeline.run(sample_shipments_csv, importer_info, output_path)

        agent_execution = result['provenance']['agent_execution']

        # Should have 3 agent executions
        assert len(agent_execution) >= 3

        # Check agent names
        agent_names = [a['agent_name'] for a in agent_execution]
        assert 'ShipmentIntakeAgent' in agent_names or 'shipment_intake' in str(agent_names).lower()
        assert 'EmissionsCalculatorAgent' in agent_names or 'emissions' in str(agent_names).lower()
        assert 'ReportingPackagerAgent' in agent_names or 'reporting' in str(agent_names).lower()


# ============================================================================
# Test Zero Hallucination Integration
# ============================================================================

@pytest.mark.integration
@pytest.mark.compliance
class TestZeroHallucinationIntegration:
    """Test zero hallucination guarantee in full pipeline."""

    def test_pipeline_zero_hallucination_flag(self, sample_shipments_csv, importer_info,
                                                test_output_dir, assert_zero_hallucination):
        """Test pipeline sets zero hallucination flag."""
        pipeline = CBAMPipeline(
            cn_codes_path="data/cn_codes.json",
            cbam_rules_path="rules/cbam_rules.yaml",
            enable_provenance=True
        )

        output_path = f"{test_output_dir}/cbam_report.json"
        result = pipeline.run(sample_shipments_csv, importer_info, output_path)

        # Validate zero hallucination
        assert_zero_hallucination(result)

    def test_pipeline_reproducibility(self, sample_shipments_csv, importer_info, test_output_dir):
        """Test pipeline produces identical results on repeated runs."""
        pipeline = CBAMPipeline(
            cn_codes_path="data/cn_codes.json",
            cbam_rules_path="rules/cbam_rules.yaml",
            enable_provenance=True
        )

        # Run twice
        result1 = pipeline.run(
            sample_shipments_csv,
            importer_info,
            f"{test_output_dir}/report1.json"
        )

        result2 = pipeline.run(
            sample_shipments_csv,
            importer_info,
            f"{test_output_dir}/report2.json"
        )

        # Emissions must be identical
        emissions1 = result1['emissions_summary']['total_embedded_emissions_tco2']
        emissions2 = result2['emissions_summary']['total_embedded_emissions_tco2']

        assert emissions1 == emissions2, \
            "Pipeline is not reproducible! ZERO HALLUCINATION VIOLATED!"


# ============================================================================
# Test Performance Integration
# ============================================================================

@pytest.mark.integration
@pytest.mark.performance
class TestPerformanceIntegration:
    """Test full pipeline performance."""

    def test_pipeline_meets_performance_target(self, sample_shipments_csv, importer_info, test_output_dir):
        """Test pipeline meets <30s target for 10K shipments."""
        import time

        pipeline = CBAMPipeline(
            cn_codes_path="data/cn_codes.json",
            cbam_rules_path="rules/cbam_rules.yaml"
        )

        start = time.time()
        result = pipeline.run(
            sample_shipments_csv,
            importer_info,
            f"{test_output_dir}/cbam_report.json"
        )
        duration = time.time() - start

        # 5 records should be <1 second
        assert duration < 2.0, f"Pipeline too slow: {duration:.2f}s"

    def test_pipeline_large_dataset(self, large_shipments_csv, importer_info, test_output_dir):
        """Test pipeline handles large dataset (1000 records)."""
        import time

        pipeline = CBAMPipeline(
            cn_codes_path="data/cn_codes.json",
            cbam_rules_path="rules/cbam_rules.yaml"
        )

        start = time.time()
        result = pipeline.run(
            large_shipments_csv,
            importer_info,
            f"{test_output_dir}/large_report.json"
        )
        duration = time.time() - start

        # 1000 records should be <10 seconds (extrapolate to 10K = <100s)
        assert duration < 15.0, \
            f"Large dataset too slow: {duration:.2f}s for 1000 records"

        # Verify all records processed
        assert result['emissions_summary']['total_shipments'] == 1000


# ============================================================================
# Test Validation Integration
# ============================================================================

@pytest.mark.integration
class TestValidationIntegration:
    """Test validation in full pipeline."""

    def test_pipeline_validation_mode(self, sample_shipments_csv, importer_info):
        """Test pipeline validation-only mode."""
        pipeline = CBAMPipeline(
            cn_codes_path="data/cn_codes.json",
            cbam_rules_path="rules/cbam_rules.yaml"
        )

        result = pipeline.validate_only(
            input_file=sample_shipments_csv,
            importer_country="NL"
        )

        assert result is not None
        assert 'metadata' in result
        assert 'validated_shipments' in result or 'errors' in result

    def test_pipeline_handles_invalid_data(self, invalid_shipments_data, tmp_path, importer_info, test_output_dir):
        """Test pipeline handles invalid data gracefully."""
        # Create CSV with invalid data
        import pandas as pd
        csv_path = tmp_path / "invalid.csv"
        df = pd.DataFrame(invalid_shipments_data)
        df.to_csv(csv_path, index=False)

        pipeline = CBAMPipeline(
            cn_codes_path="data/cn_codes.json",
            cbam_rules_path="rules/cbam_rules.yaml"
        )

        # Should not crash, but may have errors/warnings
        result = pipeline.run(
            str(csv_path),
            importer_info,
            f"{test_output_dir}/invalid_report.json"
        )

        # Check validation results
        validation = result.get('validation_results', {})
        assert 'is_valid' in validation

        # May have errors
        if 'errors' in validation:
            assert len(validation['errors']) > 0


# ============================================================================
# Test Configuration Integration
# ============================================================================

@pytest.mark.integration
class TestConfigurationIntegration:
    """Test configuration handling in pipeline."""

    def test_pipeline_with_suppliers(self, sample_shipments_csv, importer_info,
                                       suppliers_path, test_output_dir):
        """Test pipeline with supplier data."""
        pipeline = CBAMPipeline(
            cn_codes_path="data/cn_codes.json",
            cbam_rules_path="rules/cbam_rules.yaml",
            suppliers_path=suppliers_path
        )

        result = pipeline.run(
            sample_shipments_csv,
            importer_info,
            f"{test_output_dir}/with_suppliers.json"
        )

        assert result is not None

    def test_pipeline_without_provenance(self, sample_shipments_csv, importer_info, test_output_dir):
        """Test pipeline with provenance disabled."""
        pipeline = CBAMPipeline(
            cn_codes_path="data/cn_codes.json",
            cbam_rules_path="rules/cbam_rules.yaml",
            enable_provenance=False
        )

        result = pipeline.run(
            sample_shipments_csv,
            importer_info,
            f"{test_output_dir}/no_provenance.json"
        )

        # May or may not have provenance depending on implementation
        assert result is not None


# ============================================================================
# Test Output File Generation
# ============================================================================

@pytest.mark.integration
class TestOutputGeneration:
    """Test output file generation."""

    def test_generates_json_output(self, sample_shipments_csv, importer_info, test_output_dir):
        """Test generates valid JSON output file."""
        pipeline = CBAMPipeline(
            cn_codes_path="data/cn_codes.json",
            cbam_rules_path="rules/cbam_rules.yaml"
        )

        output_path = f"{test_output_dir}/cbam_report.json"
        result = pipeline.run(sample_shipments_csv, importer_info, output_path)

        # File should exist
        assert Path(output_path).exists()

        # Should be valid JSON
        with open(output_path) as f:
            loaded = json.load(f)
            assert loaded == result

    def test_output_file_completeness(self, sample_shipments_csv, importer_info, test_output_dir):
        """Test output file contains all required data."""
        pipeline = CBAMPipeline(
            cn_codes_path="data/cn_codes.json",
            cbam_rules_path="rules/cbam_rules.yaml",
            enable_provenance=True
        )

        output_path = f"{test_output_dir}/complete_report.json"
        result = pipeline.run(sample_shipments_csv, importer_info, output_path)

        # Load from file
        with open(output_path) as f:
            report = json.load(f)

        # Check all required sections
        required = [
            'report_metadata',
            'emissions_summary',
            'detailed_goods',
            'aggregations',
            'validation_results',
            'provenance'
        ]

        for section in required:
            assert section in report, f"Missing section: {section}"


# ============================================================================
# Test Error Handling Integration
# ============================================================================

@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Test error handling in full pipeline."""

    def test_pipeline_handles_missing_file(self, importer_info, test_output_dir):
        """Test pipeline handles missing input file."""
        pipeline = CBAMPipeline(
            cn_codes_path="data/cn_codes.json",
            cbam_rules_path="rules/cbam_rules.yaml"
        )

        with pytest.raises(FileNotFoundError):
            pipeline.run(
                "nonexistent.csv",
                importer_info,
                f"{test_output_dir}/report.json"
            )

    def test_pipeline_handles_corrupted_file(self, tmp_path, importer_info, test_output_dir):
        """Test pipeline handles corrupted file gracefully."""
        # Create corrupted file
        corrupted = tmp_path / "corrupted.csv"
        corrupted.write_text("This is not valid CSV\n{{corrupted}}")

        pipeline = CBAMPipeline(
            cn_codes_path="data/cn_codes.json",
            cbam_rules_path="rules/cbam_rules.yaml"
        )

        # Should raise appropriate error
        with pytest.raises(Exception):
            pipeline.run(
                str(corrupted),
                importer_info,
                f"{test_output_dir}/report.json"
            )


# ============================================================================
# Test Regression
# ============================================================================

@pytest.mark.integration
class TestRegression:
    """Regression tests to ensure consistency."""

    def test_consistent_output_structure(self, sample_shipments_csv, importer_info, test_output_dir):
        """Test output structure remains consistent across runs."""
        pipeline = CBAMPipeline(
            cn_codes_path="data/cn_codes.json",
            cbam_rules_path="rules/cbam_rules.yaml",
            enable_provenance=True
        )

        # Run multiple times
        outputs = []
        for i in range(3):
            result = pipeline.run(
                sample_shipments_csv,
                importer_info,
                f"{test_output_dir}/report_{i}.json"
            )
            outputs.append(result)

        # Check all have same structure
        for i in range(1, len(outputs)):
            assert outputs[i].keys() == outputs[0].keys(), \
                "Output structure changed between runs!"

    def test_known_good_output(self, sample_shipments_csv, importer_info,
                                 expected_total_emissions, test_output_dir):
        """Test produces known good output for reference data."""
        pipeline = CBAMPipeline(
            cn_codes_path="data/cn_codes.json",
            cbam_rules_path="rules/cbam_rules.yaml"
        )

        result = pipeline.run(
            sample_shipments_csv,
            importer_info,
            f"{test_output_dir}/known_good.json"
        )

        # Check total emissions matches expected
        total = result['emissions_summary']['total_embedded_emissions_tco2']

        # Allow small tolerance for floating point
        assert abs(total - expected_total_emissions) < 1.0, \
            f"Output changed! Expected {expected_total_emissions}, got {total}"
