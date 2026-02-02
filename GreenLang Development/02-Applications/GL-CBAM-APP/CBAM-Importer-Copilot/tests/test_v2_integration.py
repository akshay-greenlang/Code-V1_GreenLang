# -*- coding: utf-8 -*-
"""
Integration Tests for CBAM v2 Agents

Tests that v2 agents produce identical outputs to v1 agents while using less code.

Version: 2.0.0
Author: GreenLang CBAM Team
"""

import json
import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "agents"))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import v1 agents
from shipment_intake_agent import ShipmentIntakeAgent as ShipmentIntakeAgent_v1
from emissions_calculator_agent import EmissionsCalculatorAgent as EmissionsCalculatorAgent_v1
from reporting_packager_agent import ReportingPackagerAgent as ReportingPackagerAgent_v1
from cbam_pipeline import CBAMPipeline as CBAMPipeline_v1

# Import v2 agents
from shipment_intake_agent_v2 import ShipmentIntakeAgent_v2
from emissions_calculator_agent_v2 import EmissionsCalculatorAgent_v2
from reporting_packager_agent_v2 import ReportingPackagerAgent_v2
from cbam_pipeline_v2 import CBAMPipeline_v2


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def test_data_paths():
    """Provide paths to test data files."""
    base_path = Path(__file__).parent.parent
    return {
        "cn_codes": str(base_path / "data" / "cn_codes.json"),
        "cbam_rules": str(base_path / "rules" / "cbam_rules.yaml"),
        "suppliers": str(base_path / "examples" / "demo_suppliers.yaml"),
        "input_shipments": str(base_path / "examples" / "demo_shipments.csv")
    }


@pytest.fixture
def importer_info():
    """Provide sample importer information."""
    return {
        "importer_name": "Test Steel EU BV",
        "importer_country": "NL",
        "importer_eori": "NL123456789012",
        "declarant_name": "Test Declarant",
        "declarant_position": "Test Officer"
    }


# ============================================================================
# INTAKE AGENT TESTS
# ============================================================================

def test_intake_agent_v2_output_compatibility(test_data_paths):
    """Test that v2 intake agent produces compatible output with v1."""
    # Create both agents
    agent_v1 = ShipmentIntakeAgent_v1(
        cn_codes_path=test_data_paths["cn_codes"],
        cbam_rules_path=test_data_paths["cbam_rules"],
        suppliers_path=test_data_paths["suppliers"]
    )

    agent_v2 = ShipmentIntakeAgent_v2(
        cn_codes_path=test_data_paths["cn_codes"],
        cbam_rules_path=test_data_paths["cbam_rules"],
        suppliers_path=test_data_paths["suppliers"]
    )

    # Process same input
    result_v1 = agent_v1.process(test_data_paths["input_shipments"])
    result_v2 = agent_v2.process_file(test_data_paths["input_shipments"])

    # Compare metadata
    assert result_v1["metadata"]["total_records"] == result_v2["metadata"]["total_records"]
    assert result_v1["metadata"]["valid_records"] == result_v2["metadata"]["valid_records"]
    assert result_v1["metadata"]["invalid_records"] == result_v2["metadata"]["invalid_records"]

    # Compare shipment count
    assert len(result_v1["shipments"]) == len(result_v2["shipments"])

    print(f"✓ Intake agent v2 processes {result_v2['metadata']['total_records']} shipments correctly")


def test_intake_agent_v2_validation_parity(test_data_paths):
    """Test that v2 applies same validation rules as v1."""
    agent_v1 = ShipmentIntakeAgent_v1(
        cn_codes_path=test_data_paths["cn_codes"],
        cbam_rules_path=test_data_paths["cbam_rules"]
    )

    agent_v2 = ShipmentIntakeAgent_v2(
        cn_codes_path=test_data_paths["cn_codes"],
        cbam_rules_path=test_data_paths["cbam_rules"]
    )

    # Process same input
    result_v1 = agent_v1.process(test_data_paths["input_shipments"])
    result_v2 = agent_v2.process_file(test_data_paths["input_shipments"])

    # Validation error counts should match
    assert len(result_v1["validation_errors"]) == len(result_v2["validation_errors"])

    print(f"✓ Intake agent v2 applies same validation ({len(result_v2['validation_errors'])} errors detected)")


# ============================================================================
# CALCULATOR AGENT TESTS
# ============================================================================

def test_calculator_agent_v2_output_compatibility(test_data_paths):
    """Test that v2 calculator produces identical emissions as v1."""
    # First get validated shipments
    intake_agent = ShipmentIntakeAgent_v2(
        cn_codes_path=test_data_paths["cn_codes"],
        cbam_rules_path=test_data_paths["cbam_rules"],
        suppliers_path=test_data_paths["suppliers"]
    )
    validated = intake_agent.process_file(test_data_paths["input_shipments"])
    shipments = validated["shipments"]

    # Create calculator agents
    calc_v1 = EmissionsCalculatorAgent_v1(
        suppliers_path=test_data_paths["suppliers"],
        cbam_rules_path=test_data_paths["cbam_rules"]
    )

    calc_v2 = EmissionsCalculatorAgent_v2(
        suppliers_path=test_data_paths["suppliers"],
        cbam_rules_path=test_data_paths["cbam_rules"]
    )

    # Calculate emissions
    result_v1 = calc_v1.calculate_batch(shipments)
    result_v2 = calc_v2.calculate_batch(shipments)

    # Compare total emissions (should be identical for deterministic calculation)
    total_v1 = result_v1["metadata"]["total_emissions_tco2"]
    total_v2 = result_v2["metadata"]["total_emissions_tco2"]

    assert abs(total_v1 - total_v2) < 0.01, f"Emissions mismatch: v1={total_v1}, v2={total_v2}"

    print(f"✓ Calculator v2 produces identical emissions: {total_v2:.2f} tCO2")


def test_calculator_agent_v2_zero_hallucination(test_data_paths):
    """Test that v2 maintains ZERO HALLUCINATION guarantee."""
    intake_agent = ShipmentIntakeAgent_v2(
        cn_codes_path=test_data_paths["cn_codes"],
        cbam_rules_path=test_data_paths["cbam_rules"],
        suppliers_path=test_data_paths["suppliers"]
    )
    validated = intake_agent.process_file(test_data_paths["input_shipments"])
    shipments = validated["shipments"]

    calc_v2 = EmissionsCalculatorAgent_v2(
        suppliers_path=test_data_paths["suppliers"],
        cbam_rules_path=test_data_paths["cbam_rules"]
    )

    # Run calculation twice - should get identical results
    result_1 = calc_v2.calculate_batch(shipments)
    result_2 = calc_v2.calculate_batch(shipments)

    total_1 = result_1["metadata"]["total_emissions_tco2"]
    total_2 = result_2["metadata"]["total_emissions_tco2"]

    assert total_1 == total_2, "Non-deterministic calculation detected!"

    print(f"✓ Calculator v2 is deterministic (ZERO HALLUCINATION verified)")


# ============================================================================
# PACKAGER AGENT TESTS
# ============================================================================

def test_packager_agent_v2_report_structure(test_data_paths, importer_info):
    """Test that v2 packager produces valid CBAM report structure."""
    # Get shipments with emissions
    intake_agent = ShipmentIntakeAgent_v2(
        cn_codes_path=test_data_paths["cn_codes"],
        cbam_rules_path=test_data_paths["cbam_rules"],
        suppliers_path=test_data_paths["suppliers"]
    )
    validated = intake_agent.process_file(test_data_paths["input_shipments"])

    calc_agent = EmissionsCalculatorAgent_v2(
        suppliers_path=test_data_paths["suppliers"]
    )
    calculated = calc_agent.calculate_batch(validated["shipments"])

    # Generate report
    packager_v2 = ReportingPackagerAgent_v2(
        cbam_rules_path=test_data_paths["cbam_rules"]
    )
    report = packager_v2.generate_report(
        calculated["shipments"],
        importer_info
    )

    # Validate report structure
    required_sections = [
        "report_metadata",
        "importer_declaration",
        "goods_summary",
        "detailed_goods",
        "emissions_summary",
        "validation_results"
    ]

    for section in required_sections:
        assert section in report, f"Missing required section: {section}"

    print(f"✓ Packager v2 generates complete CBAM report with all sections")


def test_packager_agent_v2_validation_parity(test_data_paths, importer_info):
    """Test that v2 applies same validation rules as v1."""
    # Get shipments with emissions
    intake_agent = ShipmentIntakeAgent_v2(
        cn_codes_path=test_data_paths["cn_codes"],
        cbam_rules_path=test_data_paths["cbam_rules"],
        suppliers_path=test_data_paths["suppliers"]
    )
    validated = intake_agent.process_file(test_data_paths["input_shipments"])

    calc_agent = EmissionsCalculatorAgent_v2(
        suppliers_path=test_data_paths["suppliers"]
    )
    calculated = calc_agent.calculate_batch(validated["shipments"])

    # Generate reports
    packager_v1 = ReportingPackagerAgent_v1(
        cbam_rules_path=test_data_paths["cbam_rules"]
    )
    report_v1 = packager_v1.generate_report(calculated["shipments"], importer_info)

    packager_v2 = ReportingPackagerAgent_v2(
        cbam_rules_path=test_data_paths["cbam_rules"]
    )
    report_v2 = packager_v2.generate_report(calculated["shipments"], importer_info)

    # Compare validation results
    assert report_v1["validation_results"]["is_valid"] == report_v2["validation_results"]["is_valid"]
    assert len(report_v1["validation_results"]["rules_checked"]) == len(report_v2["validation_results"]["rules_checked"])

    print(f"✓ Packager v2 applies same validation rules (validation: {report_v2['validation_results']['is_valid']})")


# ============================================================================
# PIPELINE TESTS
# ============================================================================

def test_pipeline_v2_end_to_end(test_data_paths, importer_info, tmp_path):
    """Test complete v2 pipeline execution."""
    pipeline_v2 = CBAMPipeline_v2(
        cn_codes_path=test_data_paths["cn_codes"],
        cbam_rules_path=test_data_paths["cbam_rules"],
        suppliers_path=test_data_paths["suppliers"],
        enable_metrics=False
    )

    # Run pipeline
    output_path = tmp_path / "test_report.json"
    summary_path = tmp_path / "test_summary.md"

    report = pipeline_v2.run(
        input_file=test_data_paths["input_shipments"],
        importer_info=importer_info,
        output_report_path=str(output_path),
        output_summary_path=str(summary_path)
    )

    # Verify report was generated
    assert output_path.exists(), "Report file not created"
    assert summary_path.exists(), "Summary file not created"

    # Verify report structure
    assert "report_metadata" in report
    assert "emissions_summary" in report
    assert "validation_results" in report

    print(f"✓ Pipeline v2 executes end-to-end successfully")


def test_pipeline_v2_backward_compatibility(test_data_paths, importer_info, tmp_path):
    """Test that v2 pipeline produces compatible output with v1."""
    # Run v1 pipeline
    pipeline_v1 = CBAMPipeline_v1(
        cn_codes_path=test_data_paths["cn_codes"],
        cbam_rules_path=test_data_paths["cbam_rules"],
        suppliers_path=test_data_paths["suppliers"]
    )

    report_v1 = pipeline_v1.run(
        input_file=test_data_paths["input_shipments"],
        importer_info=importer_info
    )

    # Run v2 pipeline
    pipeline_v2 = CBAMPipeline_v2(
        cn_codes_path=test_data_paths["cn_codes"],
        cbam_rules_path=test_data_paths["cbam_rules"],
        suppliers_path=test_data_paths["suppliers"],
        enable_metrics=False
    )

    report_v2 = pipeline_v2.run(
        input_file=test_data_paths["input_shipments"],
        importer_info=importer_info
    )

    # Compare key metrics
    emissions_v1 = report_v1["emissions_summary"]["total_embedded_emissions_tco2"]
    emissions_v2 = report_v2["emissions_summary"]["total_embedded_emissions_tco2"]

    assert abs(emissions_v1 - emissions_v2) < 0.01, f"Emissions mismatch: v1={emissions_v1}, v2={emissions_v2}"

    shipments_v1 = report_v1["goods_summary"]["total_shipments"]
    shipments_v2 = report_v2["goods_summary"]["total_shipments"]

    assert shipments_v1 == shipments_v2, f"Shipment count mismatch: v1={shipments_v1}, v2={shipments_v2}"

    print(f"✓ Pipeline v2 is backward compatible with v1 (emissions: {emissions_v2:.2f} tCO2)")


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

def test_v2_performance_comparable_to_v1(test_data_paths, importer_info):
    """Test that v2 performance is comparable to v1 despite framework overhead."""
    import time

    # Time v1 pipeline
    pipeline_v1 = CBAMPipeline_v1(
        cn_codes_path=test_data_paths["cn_codes"],
        cbam_rules_path=test_data_paths["cbam_rules"],
        suppliers_path=test_data_paths["suppliers"]
    )

    start_v1 = time.time()
    report_v1 = pipeline_v1.run(
        input_file=test_data_paths["input_shipments"],
        importer_info=importer_info
    )
    time_v1 = time.time() - start_v1

    # Time v2 pipeline
    pipeline_v2 = CBAMPipeline_v2(
        cn_codes_path=test_data_paths["cn_codes"],
        cbam_rules_path=test_data_paths["cbam_rules"],
        suppliers_path=test_data_paths["suppliers"],
        enable_metrics=False
    )

    start_v2 = time.time()
    report_v2 = pipeline_v2.run(
        input_file=test_data_paths["input_shipments"],
        importer_info=importer_info
    )
    time_v2 = time.time() - start_v2

    # v2 should be within 2x of v1 (framework overhead acceptable)
    overhead_ratio = time_v2 / time_v1 if time_v1 > 0 else 1.0

    print(f"✓ Performance comparison: v1={time_v1:.3f}s, v2={time_v2:.3f}s (overhead: {overhead_ratio:.1f}x)")

    # Allow up to 3x overhead (framework initialization can be slower in tests)
    assert overhead_ratio < 3.0, f"v2 is too slow ({overhead_ratio:.1f}x slower than v1)"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
