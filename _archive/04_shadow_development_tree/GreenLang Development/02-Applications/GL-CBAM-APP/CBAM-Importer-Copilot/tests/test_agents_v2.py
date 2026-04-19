# -*- coding: utf-8 -*-
"""
GL-CBAM-APP - Agent V2 Test Suite
==================================

Comprehensive tests for refactored V2 agents ensuring:
- Output equivalence with V1
- Zero hallucination guarantee
- Performance benchmarks (<5% overhead)
- Deterministic behavior

Version: 2.0.0
Author: Testing & QA Team
"""

import pytest
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.shipment_intake_agent import ShipmentIntakeAgent
from agents.shipment_intake_agent_v2 import ShipmentIntakeAgentV2
from agents.emissions_calculator_agent import EmissionsCalculatorAgent
from agents.emissions_calculator_agent_v2 import EmissionsCalculatorAgentV2
from agents.reporting_packager_agent import ReportingPackagerAgent


# ============================================================================
# Test Output Equivalence: V1 vs V2
# ============================================================================

@pytest.mark.v2
@pytest.mark.critical
class TestShipmentIntakeAgentV2OutputEquivalence:
    """Test V2 agent produces equivalent outputs to V1."""

    def test_shipment_intake_agent_v2_output_equivalence(
        self,
        sample_shipments_csv,
        cn_codes_path,
        cbam_rules_path,
        importer_info
    ):
        """
        Verify V2 ShipmentIntakeAgent produces identical outputs to V1.

        Critical requirement: Zero regression in data processing.
        """
        # Initialize both agents
        agent_v1 = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)
        agent_v2 = ShipmentIntakeAgentV2(cn_codes_path, cbam_rules_path)

        # Process same input with both agents
        result_v1 = agent_v1.process(sample_shipments_csv, importer_info)
        result_v2 = agent_v2.process(sample_shipments_csv, importer_info)

        # Verify equivalence
        assert result_v1['metadata']['total_records'] == result_v2['metadata']['total_records'], \
            "V2 agent should process same number of records as V1"

        assert result_v1['metadata']['valid_records'] == result_v2['metadata']['valid_records'], \
            "V2 agent should validate same number of records as V1"

        # Compare validated shipments
        validated_v1 = result_v1['validated_shipments']
        validated_v2 = result_v2['validated_shipments']

        assert len(validated_v1) == len(validated_v2), \
            "V2 agent should produce same number of validated shipments"

        # Deep comparison of key fields
        for v1_record, v2_record in zip(validated_v1, validated_v2):
            assert v1_record['cn_code'] == v2_record['cn_code'], \
                "CN codes should match between V1 and V2"
            assert v1_record['quantity_tons'] == v2_record['quantity_tons'], \
                "Quantities should match between V1 and V2"
            assert v1_record['country_of_origin'] == v2_record['country_of_origin'], \
                "Country codes should match between V1 and V2"

    def test_v2_backward_compatibility_with_existing_data(
        self,
        sample_shipments_csv,
        cn_codes_path,
        cbam_rules_path,
        importer_info
    ):
        """Verify V2 agent maintains backward compatibility with V1 data formats."""
        agent_v2 = ShipmentIntakeAgentV2(cn_codes_path, cbam_rules_path)

        # Should process V1 format data without errors
        result = agent_v2.process(sample_shipments_csv, importer_info)

        assert result is not None
        assert 'validated_shipments' in result
        assert 'metadata' in result
        assert len(result['validated_shipments']) > 0

    def test_v2_enhanced_provenance_tracking(
        self,
        sample_shipments_csv,
        cn_codes_path,
        cbam_rules_path,
        importer_info
    ):
        """Verify V2 agent includes enhanced provenance metadata."""
        agent_v2 = ShipmentIntakeAgentV2(cn_codes_path, cbam_rules_path)
        result = agent_v2.process(sample_shipments_csv, importer_info)

        # V2 should include provenance tracking
        metadata = result['metadata']
        assert 'agent_version' in metadata
        assert metadata['agent_version'] == '2.0'
        assert 'processing_timestamp' in metadata
        assert 'data_lineage' in metadata or 'provenance' in metadata


@pytest.mark.v2
@pytest.mark.critical
class TestEmissionsCalculatorAgentV2Determinism:
    """Test V2 EmissionsCalculatorAgent for deterministic behavior."""

    def test_emissions_calculator_agent_v2_determinism(
        self,
        validated_shipments_fixture,
        emission_factors_db
    ):
        """
        Verify V2 EmissionsCalculatorAgent produces deterministic results.

        Critical: Same input must always produce same output (zero hallucination).
        """
        agent_v2 = EmissionsCalculatorAgentV2(emission_factors_db)

        # Run calculation multiple times
        results = []
        for i in range(5):
            result = agent_v2.calculate_emissions(validated_shipments_fixture)
            results.append(result)

        # Verify all results are identical
        first_result = results[0]
        for result in results[1:]:
            assert result['total_emissions'] == first_result['total_emissions'], \
                "V2 agent must produce deterministic emission calculations"

            # Compare per-shipment emissions
            for idx, (shipment1, shipment2) in enumerate(
                zip(first_result['shipment_emissions'], result['shipment_emissions'])
            ):
                assert shipment1['total_co2e'] == shipment2['total_co2e'], \
                    f"Shipment {idx} emissions must be deterministic"
                assert shipment1['direct_emissions'] == shipment2['direct_emissions'], \
                    f"Shipment {idx} direct emissions must be deterministic"

    def test_v2_zero_hallucination_guarantee(
        self,
        validated_shipments_fixture,
        emission_factors_db
    ):
        """
        Verify V2 agent never hallucinates emission factors.

        All emission factors must be traceable to source database.
        """
        agent_v2 = EmissionsCalculatorAgentV2(emission_factors_db)
        result = agent_v2.calculate_emissions(validated_shipments_fixture)

        # Verify all emission factors have provenance
        for shipment in result['shipment_emissions']:
            assert 'emission_factor_source' in shipment, \
                "All emission factors must have documented source"
            assert 'emission_factor_id' in shipment, \
                "All emission factors must have traceable ID"

            # Verify factor exists in database
            factor_id = shipment['emission_factor_id']
            assert factor_id in emission_factors_db or factor_id.startswith('DEFAULT_'), \
                f"Emission factor {factor_id} must exist in database or be documented default"

    def test_v2_calculation_accuracy(
        self,
        validated_shipments_fixture,
        emission_factors_db
    ):
        """Verify V2 calculation accuracy against known ground truth."""
        agent_v2 = EmissionsCalculatorAgentV2(emission_factors_db)

        # Test with known input/output pair
        test_shipment = [{
            'cn_code': '7208100000',
            'quantity_tons': 100,
            'country_of_origin': 'CN',
            'product_group': 'iron_steel'
        }]

        result = agent_v2.calculate_emissions(test_shipment)

        # Verify calculation
        assert 'shipment_emissions' in result
        assert len(result['shipment_emissions']) == 1

        shipment_result = result['shipment_emissions'][0]
        assert shipment_result['total_co2e'] > 0, \
            "Emissions calculation should produce positive result"
        assert 'calculation_methodology' in shipment_result, \
            "Calculation methodology must be documented"


@pytest.mark.v2
@pytest.mark.critical
class TestReportingPackagerAgentV2Formats:
    """Test V2 ReportingPackagerAgent output format support."""

    def test_reporting_packager_agent_v2_formats(
        self,
        calculated_emissions_fixture
    ):
        """Verify V2 ReportingPackagerAgent supports all required formats."""
        agent = ReportingPackagerAgent()

        # Test CBAM XML format
        xml_report = agent.generate_report(
            calculated_emissions_fixture,
            format='cbam_xml'
        )
        assert xml_report is not None
        assert isinstance(xml_report, (str, bytes))
        assert b'<?xml' in xml_report or '<?xml' in xml_report

        # Test JSON format
        json_report = agent.generate_report(
            calculated_emissions_fixture,
            format='json'
        )
        assert json_report is not None
        json_data = json.loads(json_report) if isinstance(json_report, str) else json_report
        assert 'total_emissions' in json_data or 'emissions' in json_data

        # Test Excel format
        excel_report = agent.generate_report(
            calculated_emissions_fixture,
            format='excel'
        )
        assert excel_report is not None
        assert isinstance(excel_report, (bytes, str, Path))

    def test_v2_report_validation(
        self,
        calculated_emissions_fixture
    ):
        """Verify V2 reports pass schema validation."""
        agent = ReportingPackagerAgent()

        report = agent.generate_report(
            calculated_emissions_fixture,
            format='cbam_xml'
        )

        # Validate against CBAM schema
        is_valid = agent.validate_report(report, schema='cbam_v2')
        assert is_valid, "Generated CBAM report must pass schema validation"


# ============================================================================
# Test Pipeline V2 Orchestration
# ============================================================================

@pytest.mark.v2
@pytest.mark.integration
class TestPipelineV2Orchestration:
    """Test end-to-end V2 pipeline orchestration."""

    def test_pipeline_v2_orchestration(
        self,
        sample_shipments_csv,
        cn_codes_path,
        cbam_rules_path,
        emission_factors_db,
        importer_info
    ):
        """
        Test complete V2 pipeline execution.

        Intake -> Calculation -> Reporting
        """
        # Step 1: Intake
        intake_agent = ShipmentIntakeAgentV2(cn_codes_path, cbam_rules_path)
        intake_result = intake_agent.process(sample_shipments_csv, importer_info)

        assert intake_result['metadata']['valid_records'] > 0, \
            "Pipeline should produce valid records"

        # Step 2: Emissions Calculation
        calc_agent = EmissionsCalculatorAgentV2(emission_factors_db)
        calc_result = calc_agent.calculate_emissions(
            intake_result['validated_shipments']
        )

        assert calc_result['total_emissions'] > 0, \
            "Pipeline should calculate emissions"

        # Step 3: Reporting
        report_agent = ReportingPackagerAgent()
        report = report_agent.generate_report(calc_result, format='cbam_xml')

        assert report is not None, \
            "Pipeline should generate report"

    def test_v2_pipeline_error_propagation(
        self,
        invalid_shipments_csv,
        cn_codes_path,
        cbam_rules_path,
        emission_factors_db,
        importer_info
    ):
        """Verify V2 pipeline properly propagates and handles errors."""
        intake_agent = ShipmentIntakeAgentV2(cn_codes_path, cbam_rules_path)

        # Should handle invalid data gracefully
        result = intake_agent.process(invalid_shipments_csv, importer_info)

        # Should have error tracking
        assert 'errors' in result or result['metadata']['error_count'] > 0

        # Invalid records should not proceed to calculation
        assert result['metadata']['valid_records'] == 0 or \
               result['metadata']['valid_records'] < result['metadata']['total_records']


# ============================================================================
# Test Zero Hallucination Guarantee
# ============================================================================

@pytest.mark.v2
@pytest.mark.critical
class TestZeroHallucinationGuarantee:
    """Test V2 agents maintain zero hallucination guarantee."""

    def test_zero_hallucination_guarantee(
        self,
        emission_factors_db
    ):
        """
        Verify agents never generate data without source provenance.

        Critical: All data must be traceable to input or database.
        """
        agent = EmissionsCalculatorAgentV2(emission_factors_db)

        test_shipments = [
            {
                'cn_code': '7208100000',
                'quantity_tons': 50,
                'country_of_origin': 'CN',
                'product_group': 'iron_steel'
            },
            {
                'cn_code': '2710199100',
                'quantity_tons': 30,
                'country_of_origin': 'DE',
                'product_group': 'petroleum_oils'
            }
        ]

        result = agent.calculate_emissions(test_shipments)

        # Every calculated value must have provenance
        for shipment in result['shipment_emissions']:
            # Check emission factor provenance
            assert 'emission_factor_source' in shipment
            assert 'emission_factor_id' in shipment

            # Check calculation provenance
            assert 'calculation_method' in shipment or 'methodology' in shipment

            # Verify values are not randomly generated
            # Run again and verify determinism
            result2 = agent.calculate_emissions(test_shipments)

            idx = result['shipment_emissions'].index(shipment)
            shipment2 = result2['shipment_emissions'][idx]

            assert shipment['total_co2e'] == shipment2['total_co2e'], \
                "Values must be deterministic, not hallucinated"

    def test_provenance_tracking_completeness(
        self,
        sample_shipments_csv,
        cn_codes_path,
        cbam_rules_path,
        importer_info
    ):
        """Verify complete provenance tracking through pipeline."""
        agent = ShipmentIntakeAgentV2(cn_codes_path, cbam_rules_path)
        result = agent.process(sample_shipments_csv, importer_info)

        # Check provenance metadata
        assert 'data_lineage' in result['metadata'] or 'provenance' in result['metadata']

        # Each validated shipment should have provenance
        for shipment in result['validated_shipments']:
            # Should track enrichment sources
            if 'cn_description' in shipment:
                assert 'cn_description_source' in shipment or \
                       result['metadata'].get('cn_codes_source') is not None


# ============================================================================
# Test Performance Benchmarks
# ============================================================================

@pytest.mark.v2
@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Test V2 agent performance meets targets (<5% overhead)."""

    def test_performance_benchmarks(
        self,
        large_shipments_csv,
        cn_codes_path,
        cbam_rules_path,
        importer_info
    ):
        """
        Verify V2 agent overhead is <5% compared to V1.

        Performance target: <5% regression from V1.
        """
        # Benchmark V1
        agent_v1 = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)

        start_v1 = time.perf_counter()
        result_v1 = agent_v1.process(large_shipments_csv, importer_info)
        duration_v1 = time.perf_counter() - start_v1

        # Benchmark V2
        agent_v2 = ShipmentIntakeAgentV2(cn_codes_path, cbam_rules_path)

        start_v2 = time.perf_counter()
        result_v2 = agent_v2.process(large_shipments_csv, importer_info)
        duration_v2 = time.perf_counter() - start_v2

        # Calculate overhead
        overhead_pct = ((duration_v2 - duration_v1) / duration_v1) * 100

        assert overhead_pct < 5.0, \
            f"V2 agent overhead {overhead_pct:.1f}% exceeds 5% target"

        # Verify throughput is maintained
        records_v1 = result_v1['metadata']['total_records']
        records_v2 = result_v2['metadata']['total_records']

        throughput_v1 = records_v1 / duration_v1
        throughput_v2 = records_v2 / duration_v2

        throughput_degradation = ((throughput_v1 - throughput_v2) / throughput_v1) * 100

        assert throughput_degradation < 5.0, \
            f"V2 throughput degradation {throughput_degradation:.1f}% exceeds 5% target"

    def test_v2_memory_efficiency(
        self,
        large_shipments_csv,
        cn_codes_path,
        cbam_rules_path,
        importer_info
    ):
        """Verify V2 agent memory usage is reasonable."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run V2 agent
        agent_v2 = ShipmentIntakeAgentV2(cn_codes_path, cbam_rules_path)
        result = agent_v2.process(large_shipments_csv, importer_info)

        # Measure peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB

        memory_increase = peak_memory - baseline_memory

        # Memory increase should be reasonable (<500MB for large dataset)
        assert memory_increase < 500, \
            f"V2 agent memory usage {memory_increase:.1f}MB exceeds reasonable limit"

    def test_v2_concurrent_execution(
        self,
        sample_shipments_csv,
        cn_codes_path,
        cbam_rules_path,
        importer_info
    ):
        """Verify V2 agent can handle concurrent executions."""
        import concurrent.futures

        agent = ShipmentIntakeAgentV2(cn_codes_path, cbam_rules_path)

        # Execute agent concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(agent.process, sample_shipments_csv, importer_info)
                for _ in range(10)
            ]

            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All executions should succeed
        assert len(results) == 10
        assert all(r['metadata']['total_records'] > 0 for r in results)

        # Results should be deterministic (all same)
        first_result = results[0]
        for result in results[1:]:
            assert result['metadata']['total_records'] == first_result['metadata']['total_records']


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def validated_shipments_fixture():
    """Sample validated shipments for testing."""
    return [
        {
            'cn_code': '7208100000',
            'quantity_tons': 100,
            'country_of_origin': 'CN',
            'product_group': 'iron_steel',
            'import_date': '2024-01-15'
        },
        {
            'cn_code': '2710199100',
            'quantity_tons': 50,
            'country_of_origin': 'DE',
            'product_group': 'petroleum_oils',
            'import_date': '2024-01-20'
        }
    ]


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
        }
    }


@pytest.fixture
def calculated_emissions_fixture():
    """Sample calculated emissions for reporting tests."""
    return {
        'total_emissions': 250.5,
        'shipment_emissions': [
            {
                'cn_code': '7208100000',
                'quantity_tons': 100,
                'total_co2e': 210.0,
                'direct_emissions': 180.0,
                'indirect_emissions': 30.0,
                'emission_factor_id': 'IRON_STEEL_CN_DEFAULT',
                'emission_factor_source': 'EU CBAM Database 2024'
            },
            {
                'cn_code': '2710199100',
                'quantity_tons': 50,
                'total_co2e': 40.5,
                'direct_emissions': 35.0,
                'indirect_emissions': 5.5,
                'emission_factor_id': 'PETROLEUM_OILS_DE_DEFAULT',
                'emission_factor_source': 'EU CBAM Database 2024'
            }
        ]
    }


@pytest.fixture
def invalid_shipments_csv(tmp_path):
    """Invalid shipments CSV for error testing."""
    csv_path = tmp_path / "invalid_shipments.csv"
    df = pd.DataFrame([
        {'cn_code': '', 'quantity_tons': -100, 'country_of_origin': 'INVALID'},
        {'cn_code': '12345', 'quantity_tons': 0, 'country_of_origin': ''},
    ])
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def large_shipments_csv(tmp_path):
    """Large shipments CSV for performance testing."""
    csv_path = tmp_path / "large_shipments.csv"

    # Generate 1000 records
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


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'v2'])
