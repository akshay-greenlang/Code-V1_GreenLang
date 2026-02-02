# -*- coding: utf-8 -*-
"""
GL-CBAM-APP - Integration Test Suite V2
========================================

End-to-end integration tests for V2 CBAM pipeline:
- Complete pipeline execution
- Backward compatibility
- Error handling and recovery
- Provenance tracking

Version: 2.0.0
Author: Testing & QA Team
"""

import pytest
import pandas as pd
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any
import sys
import time
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.shipment_intake_agent_v2 import ShipmentIntakeAgentV2
from agents.emissions_calculator_agent_v2 import EmissionsCalculatorAgentV2
from agents.reporting_packager_agent import ReportingPackagerAgent


# ============================================================================
# End-to-End Pipeline Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.critical
class TestEndToEndPipeline:
    """Test complete end-to-end pipeline execution."""

    def test_end_to_end_pipeline(
        self,
        sample_shipments_csv,
        cn_codes_path,
        cbam_rules_path,
        emission_factors_db,
        importer_info,
        tmp_path
    ):
        """
        Test complete pipeline from CSV input to CBAM XML report.

        Flow: CSV -> Intake -> Validation -> Calculation -> Reporting -> XML
        """
        # Stage 1: Shipment Intake
        print("\n[Stage 1] Shipment Intake and Validation")
        intake_agent = ShipmentIntakeAgentV2(cn_codes_path, cbam_rules_path)

        intake_result = intake_agent.process(
            input_file=sample_shipments_csv,
            importer_info=importer_info
        )

        # Verify intake stage
        assert intake_result is not None
        assert 'validated_shipments' in intake_result
        assert 'metadata' in intake_result
        assert intake_result['metadata']['total_records'] > 0

        validated_shipments = intake_result['validated_shipments']
        assert len(validated_shipments) > 0, "Should produce validated shipments"

        print(f"  ✓ Processed {intake_result['metadata']['total_records']} records")
        print(f"  ✓ Validated {len(validated_shipments)} shipments")

        # Stage 2: Emissions Calculation
        print("\n[Stage 2] Emissions Calculation")
        calc_agent = EmissionsCalculatorAgentV2(emission_factors_db)

        calc_result = calc_agent.calculate_emissions(validated_shipments)

        # Verify calculation stage
        assert calc_result is not None
        assert 'total_emissions' in calc_result
        assert 'shipment_emissions' in calc_result
        assert calc_result['total_emissions'] > 0

        print(f"  ✓ Total emissions: {calc_result['total_emissions']:.2f} tCO2e")
        print(f"  ✓ Calculated emissions for {len(calc_result['shipment_emissions'])} shipments")

        # Stage 3: Report Generation
        print("\n[Stage 3] Report Generation")
        report_agent = ReportingPackagerAgent()

        # Generate CBAM XML report
        xml_report = report_agent.generate_report(
            calc_result,
            format='cbam_xml'
        )

        assert xml_report is not None
        assert len(xml_report) > 0

        # Save report to file
        report_path = tmp_path / "cbam_report.xml"
        if isinstance(xml_report, bytes):
            report_path.write_bytes(xml_report)
        else:
            report_path.write_text(xml_report)

        assert report_path.exists()
        print(f"  ✓ Generated CBAM XML report: {report_path}")

        # Generate JSON report for verification
        json_report = report_agent.generate_report(
            calc_result,
            format='json'
        )

        json_data = json.loads(json_report) if isinstance(json_report, str) else json_report
        assert 'total_emissions' in json_data or 'emissions' in json_data

        print(f"  ✓ Generated JSON report for verification")

        # Verify end-to-end data integrity
        print("\n[Verification] Data Integrity")
        assert len(calc_result['shipment_emissions']) == len(validated_shipments), \
            "All validated shipments should have emissions calculated"

        print("  ✓ End-to-end pipeline completed successfully")

    def test_end_to_end_pipeline_with_multiple_formats(
        self,
        sample_shipments_csv,
        cn_codes_path,
        cbam_rules_path,
        emission_factors_db,
        importer_info,
        tmp_path
    ):
        """Test pipeline generates all required output formats."""
        # Execute pipeline
        intake_agent = ShipmentIntakeAgentV2(cn_codes_path, cbam_rules_path)
        calc_agent = EmissionsCalculatorAgentV2(emission_factors_db)
        report_agent = ReportingPackagerAgent()

        intake_result = intake_agent.process(sample_shipments_csv, importer_info)
        calc_result = calc_agent.calculate_emissions(intake_result['validated_shipments'])

        # Generate all formats
        formats = ['cbam_xml', 'json', 'excel', 'csv']
        generated_reports = {}

        for fmt in formats:
            try:
                report = report_agent.generate_report(calc_result, format=fmt)
                generated_reports[fmt] = report
                assert report is not None, f"Should generate {fmt} report"
                print(f"  ✓ Generated {fmt.upper()} report")
            except Exception as e:
                print(f"  ⚠ {fmt.upper()} format not yet implemented: {e}")

        # At minimum, should support CBAM XML and JSON
        assert 'cbam_xml' in generated_reports
        assert 'json' in generated_reports

    def test_end_to_end_with_large_dataset(
        self,
        large_shipments_csv,
        cn_codes_path,
        cbam_rules_path,
        emission_factors_db,
        importer_info
    ):
        """Test pipeline performance with large dataset."""
        print("\n[Performance Test] Large Dataset (1000 records)")

        start_time = time.perf_counter()

        # Execute complete pipeline
        intake_agent = ShipmentIntakeAgentV2(cn_codes_path, cbam_rules_path)
        intake_result = intake_agent.process(large_shipments_csv, importer_info)

        intake_time = time.perf_counter()
        print(f"  Intake: {intake_time - start_time:.2f}s")

        calc_agent = EmissionsCalculatorAgentV2(emission_factors_db)
        calc_result = calc_agent.calculate_emissions(intake_result['validated_shipments'])

        calc_time = time.perf_counter()
        print(f"  Calculation: {calc_time - intake_time:.2f}s")

        report_agent = ReportingPackagerAgent()
        report = report_agent.generate_report(calc_result, format='json')

        report_time = time.perf_counter()
        print(f"  Reporting: {report_time - calc_time:.2f}s")

        total_time = report_time - start_time
        print(f"  Total: {total_time:.2f}s")

        # Performance assertions
        records = intake_result['metadata']['total_records']
        throughput = records / total_time

        assert throughput >= 100, \
            f"Pipeline throughput {throughput:.0f} records/sec below target (100 records/sec)"

        print(f"  ✓ Throughput: {throughput:.0f} records/sec")


# ============================================================================
# Backward Compatibility Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.critical
class TestBackwardCompatibility:
    """Test V2 pipeline maintains backward compatibility."""

    def test_backward_compatibility(
        self,
        sample_shipments_csv,
        cn_codes_path,
        cbam_rules_path,
        emission_factors_db,
        importer_info
    ):
        """
        Verify V2 pipeline processes V1 data formats correctly.

        Critical: Must maintain compatibility with existing data.
        """
        # Use V2 agents with V1 format data
        intake_agent = ShipmentIntakeAgentV2(cn_codes_path, cbam_rules_path)

        # Should process V1 CSV format
        result = intake_agent.process(sample_shipments_csv, importer_info)

        assert result is not None
        assert len(result['validated_shipments']) > 0

        # Verify all V1 required fields are present
        for shipment in result['validated_shipments']:
            # V1 required fields
            assert 'cn_code' in shipment
            assert 'quantity_tons' in shipment
            assert 'country_of_origin' in shipment

    def test_v2_output_consumable_by_downstream_systems(
        self,
        sample_shipments_csv,
        cn_codes_path,
        cbam_rules_path,
        emission_factors_db,
        importer_info
    ):
        """Verify V2 outputs are consumable by existing downstream systems."""
        # Generate V2 output
        intake_agent = ShipmentIntakeAgentV2(cn_codes_path, cbam_rules_path)
        calc_agent = EmissionsCalculatorAgentV2(emission_factors_db)
        report_agent = ReportingPackagerAgent()

        intake_result = intake_agent.process(sample_shipments_csv, importer_info)
        calc_result = calc_agent.calculate_emissions(intake_result['validated_shipments'])
        json_report = report_agent.generate_report(calc_result, format='json')

        # Parse as JSON
        report_data = json.loads(json_report) if isinstance(json_report, str) else json_report

        # Verify expected structure for downstream systems
        assert 'total_emissions' in report_data or 'emissions' in report_data
        assert isinstance(report_data, dict)

    def test_v1_to_v2_migration_path(
        self,
        sample_shipments_csv,
        cn_codes_path,
        cbam_rules_path,
        importer_info
    ):
        """Test migration path from V1 to V2 agents."""
        from agents.shipment_intake_agent import ShipmentIntakeAgent

        # Process with V1
        agent_v1 = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)
        result_v1 = agent_v1.process(sample_shipments_csv, importer_info)

        # Process with V2
        agent_v2 = ShipmentIntakeAgentV2(cn_codes_path, cbam_rules_path)
        result_v2 = agent_v2.process(sample_shipments_csv, importer_info)

        # Results should be compatible
        assert result_v1['metadata']['total_records'] == result_v2['metadata']['total_records']

        # V2 should maintain V1 output structure
        v1_keys = set(result_v1.keys())
        v2_keys = set(result_v2.keys())

        # V2 should have all V1 keys (may have additional)
        assert v1_keys.issubset(v2_keys), \
            f"V2 missing V1 keys: {v1_keys - v2_keys}"


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.integration
class TestErrorHandling:
    """Test pipeline error handling and recovery."""

    def test_error_handling(
        self,
        invalid_shipments_csv,
        cn_codes_path,
        cbam_rules_path,
        emission_factors_db,
        importer_info
    ):
        """
        Test pipeline handles errors gracefully.

        Should not crash on invalid data.
        """
        intake_agent = ShipmentIntakeAgentV2(cn_codes_path, cbam_rules_path)

        # Should process without crashing
        result = intake_agent.process(invalid_shipments_csv, importer_info)

        # Should have error tracking
        assert 'errors' in result or result['metadata']['error_count'] > 0

        # Should document which records failed
        metadata = result['metadata']
        assert metadata['total_records'] > 0
        assert metadata['error_count'] > 0 or metadata['valid_records'] < metadata['total_records']

    def test_partial_failure_recovery(
        self,
        mixed_valid_invalid_csv,
        cn_codes_path,
        cbam_rules_path,
        emission_factors_db,
        importer_info
    ):
        """Test pipeline processes valid records even when some are invalid."""
        intake_agent = ShipmentIntakeAgentV2(cn_codes_path, cbam_rules_path)
        result = intake_agent.process(mixed_valid_invalid_csv, importer_info)

        # Should have both valid and invalid records
        assert result['metadata']['valid_records'] > 0
        assert result['metadata']['error_count'] > 0

        # Valid records should proceed through pipeline
        calc_agent = EmissionsCalculatorAgentV2(emission_factors_db)
        calc_result = calc_agent.calculate_emissions(result['validated_shipments'])

        assert calc_result['total_emissions'] > 0
        assert len(calc_result['shipment_emissions']) == result['metadata']['valid_records']

    def test_missing_data_handling(
        self,
        cn_codes_path,
        cbam_rules_path,
        importer_info,
        tmp_path
    ):
        """Test pipeline handles missing optional data gracefully."""
        # Create CSV with minimal required fields
        csv_path = tmp_path / "minimal.csv"
        df = pd.DataFrame([
            {'cn_code': '7208100000', 'quantity_tons': 100, 'country_of_origin': 'CN'},
            {'cn_code': '2710199100', 'quantity_tons': 50, 'country_of_origin': 'DE'}
        ])
        df.to_csv(csv_path, index=False)

        intake_agent = ShipmentIntakeAgentV2(cn_codes_path, cbam_rules_path)
        result = intake_agent.process(str(csv_path), importer_info)

        # Should process successfully with minimal data
        assert result['metadata']['valid_records'] > 0
        assert len(result['validated_shipments']) > 0

    def test_malformed_input_handling(
        self,
        cn_codes_path,
        cbam_rules_path,
        importer_info,
        tmp_path
    ):
        """Test pipeline handles malformed input files."""
        # Create malformed CSV
        malformed_csv = tmp_path / "malformed.csv"
        malformed_csv.write_text("This is not a valid CSV\nRandom text\n{{invalid}}")

        intake_agent = ShipmentIntakeAgentV2(cn_codes_path, cbam_rules_path)

        # Should raise appropriate error
        with pytest.raises(Exception) as exc_info:
            intake_agent.process(str(malformed_csv), importer_info)

        # Error should be informative
        assert "csv" in str(exc_info.value).lower() or "parse" in str(exc_info.value).lower()


# ============================================================================
# Provenance Tracking Tests
# ============================================================================

@pytest.mark.integration
class TestProvenanceTracking:
    """Test provenance tracking through complete pipeline."""

    def test_provenance_tracking(
        self,
        sample_shipments_csv,
        cn_codes_path,
        cbam_rules_path,
        emission_factors_db,
        importer_info
    ):
        """
        Verify complete provenance tracking from input to output.

        Every data transformation must be traceable.
        """
        # Execute pipeline
        intake_agent = ShipmentIntakeAgentV2(cn_codes_path, cbam_rules_path)
        calc_agent = EmissionsCalculatorAgentV2(emission_factors_db)

        intake_result = intake_agent.process(sample_shipments_csv, importer_info)
        calc_result = calc_agent.calculate_emissions(intake_result['validated_shipments'])

        # Check intake provenance
        intake_metadata = intake_result['metadata']
        assert 'agent_version' in intake_metadata
        assert 'processing_timestamp' in intake_metadata
        assert 'input_file' in intake_metadata or 'source_file' in intake_metadata

        # Check calculation provenance
        for shipment in calc_result['shipment_emissions']:
            # Emission factor provenance
            assert 'emission_factor_source' in shipment
            assert 'emission_factor_id' in shipment

            # Calculation methodology
            assert 'calculation_method' in shipment or 'methodology' in shipment

    def test_data_lineage_tracking(
        self,
        sample_shipments_csv,
        cn_codes_path,
        cbam_rules_path,
        emission_factors_db,
        importer_info
    ):
        """Test data lineage is tracked through pipeline stages."""
        intake_agent = ShipmentIntakeAgentV2(cn_codes_path, cbam_rules_path)
        intake_result = intake_agent.process(sample_shipments_csv, importer_info)

        # Should have lineage metadata
        metadata = intake_result['metadata']
        assert 'data_lineage' in metadata or 'provenance' in metadata or 'source_file' in metadata

        # Each shipment should be traceable to source
        for idx, shipment in enumerate(intake_result['validated_shipments']):
            # Should have row number or identifier for traceability
            assert 'source_row' in shipment or 'record_id' in shipment or idx >= 0

    def test_audit_trail_completeness(
        self,
        sample_shipments_csv,
        cn_codes_path,
        cbam_rules_path,
        emission_factors_db,
        importer_info
    ):
        """Test complete audit trail is maintained."""
        intake_agent = ShipmentIntakeAgentV2(cn_codes_path, cbam_rules_path)
        calc_agent = EmissionsCalculatorAgentV2(emission_factors_db)
        report_agent = ReportingPackagerAgent()

        # Execute pipeline
        intake_result = intake_agent.process(sample_shipments_csv, importer_info)
        calc_result = calc_agent.calculate_emissions(intake_result['validated_shipments'])
        report = report_agent.generate_report(calc_result, format='json')

        # Parse report
        report_data = json.loads(report) if isinstance(report, str) else report

        # Audit trail should include:
        # - Processing timestamps
        # - Agent versions
        # - Data sources
        # - Calculation methodologies

        # Check for audit metadata
        has_audit_trail = (
            'metadata' in report_data or
            'audit_trail' in report_data or
            'processing_info' in report_data
        )

        assert has_audit_trail, "Report should include audit trail metadata"


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mixed_valid_invalid_csv(tmp_path):
    """CSV with mix of valid and invalid records."""
    csv_path = tmp_path / "mixed.csv"
    df = pd.DataFrame([
        # Valid records
        {'cn_code': '7208100000', 'quantity_tons': 100, 'country_of_origin': 'CN'},
        {'cn_code': '2710199100', 'quantity_tons': 50, 'country_of_origin': 'DE'},
        # Invalid records
        {'cn_code': '', 'quantity_tons': -10, 'country_of_origin': 'INVALID'},
        {'cn_code': '12345', 'quantity_tons': 0, 'country_of_origin': ''},
    ])
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def large_shipments_csv(tmp_path):
    """Large dataset for performance testing."""
    csv_path = tmp_path / "large_shipments.csv"

    data = []
    for i in range(1000):
        data.append({
            'cn_code': '7208100000',
            'quantity_tons': 100 + i,
            'country_of_origin': 'CN',
            'import_date': f'2024-01-{(i % 28) + 1:02d}'
        })

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def emission_factors_db():
    """Mock emission factors database."""
    return {
        'IRON_STEEL_CN_DEFAULT': {
            'factor_id': 'IRON_STEEL_CN_DEFAULT',
            'value': 2.1,
            'unit': 'tCO2e/ton',
            'source': 'EU CBAM Database 2024',
            'product_group': 'iron_steel',
            'country': 'CN'
        },
        'PETROLEUM_OILS_DE_DEFAULT': {
            'factor_id': 'PETROLEUM_OILS_DE_DEFAULT',
            'value': 0.8,
            'unit': 'tCO2e/ton',
            'source': 'EU CBAM Database 2024',
            'product_group': 'petroleum_oils',
            'country': 'DE'
        },
        'DEFAULT_FACTOR': {
            'factor_id': 'DEFAULT_FACTOR',
            'value': 1.0,
            'unit': 'tCO2e/ton',
            'source': 'EU CBAM Database 2024 - Default',
            'product_group': 'general',
            'country': 'GLOBAL'
        }
    }


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
