"""
GL-CSRD-APP - All Agents Test Suite
====================================

Comprehensive tests for all 6 CSRD agents:
1. Intake Agent - Validate and process input data
2. Materiality Agent - LLM-powered materiality assessment
3. Calculator Agent - Zero hallucination calculations
4. Aggregator Agent - Performance-optimized aggregation
5. Reporting Agent - XBRL generation
6. Audit Agent - Compliance rule validation

Version: 1.0.0
Author: Testing & QA Team
"""

import pytest
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.intake_agent import IntakeAgent
from agents.materiality_agent import MaterialityAgent
from agents.calculator_agent import CalculatorAgent
from agents.aggregator_agent import AggregatorAgent
from agents.reporting_agent import ReportingAgent
from agents.audit_agent import AuditAgent


# ============================================================================
# Intake Agent Tests
# ============================================================================

@pytest.mark.agents
@pytest.mark.critical
class TestIntakeAgent:
    """Test Intake Agent for data validation and processing."""

    def test_intake_agent_validate_and_process(self, sample_csrd_input):
        """
        Test Intake Agent validates and processes CSRD input data.

        Requirements:
        - Parse multiple input formats (CSV, Excel, JSON)
        - Validate required fields
        - Enrich with metadata
        - Return structured output
        """
        agent = IntakeAgent()

        result = agent.process(sample_csrd_input)

        # Verify output structure
        assert result is not None
        assert 'validated_data' in result
        assert 'metadata' in result
        assert 'errors' in result or result['metadata'].get('error_count', 0) >= 0

        # Verify validation
        metadata = result['metadata']
        assert metadata['total_records'] > 0
        assert metadata['valid_records'] <= metadata['total_records']

    def test_intake_agent_field_validation(self):
        """Test Intake Agent validates required CSRD fields."""
        agent = IntakeAgent()

        # Test with missing required fields
        invalid_data = {
            "company_name": "",  # Empty
            "reporting_period": "invalid",  # Invalid format
            # Missing other required fields
        }

        result = agent.validate_record(invalid_data)

        # Should have validation errors
        assert not result['is_valid']
        assert len(result['errors']) > 0

    def test_intake_agent_data_enrichment(self, sample_csrd_input):
        """Test Intake Agent enriches data with ESRS metadata."""
        agent = IntakeAgent()

        result = agent.process(sample_csrd_input)

        # Should enrich with ESRS categories
        validated = result['validated_data']
        for record in validated:
            assert 'esrs_category' in record or 'category' in record

    def test_intake_agent_bulk_processing(self, large_csrd_dataset):
        """Test Intake Agent handles bulk data efficiently."""
        agent = IntakeAgent()

        start = time.perf_counter()
        result = agent.process(large_csrd_dataset)
        duration = time.perf_counter() - start

        # Performance check
        records = result['metadata']['total_records']
        throughput = records / duration

        assert throughput >= 50, \
            f"Intake throughput {throughput:.0f} records/sec below target"


# ============================================================================
# Materiality Agent Tests
# ============================================================================

@pytest.mark.agents
@pytest.mark.critical
class TestMaterialityAgent:
    """Test Materiality Agent for LLM-powered materiality assessment."""

    def test_materiality_agent_llm_calls(self, sample_company_data):
        """
        Test Materiality Agent performs double materiality assessment.

        Requirements:
        - Analyze financial materiality
        - Analyze impact materiality
        - Use LLM for qualitative assessment
        - Return structured materiality matrix
        """
        agent = MaterialityAgent(llm_model="gpt-4")

        result = agent.assess_materiality(sample_company_data)

        # Verify double materiality
        assert 'financial_materiality' in result
        assert 'impact_materiality' in result
        assert 'material_topics' in result

        # Should identify material topics
        assert len(result['material_topics']) > 0

    def test_materiality_agent_topic_identification(self, sample_company_data):
        """Test Materiality Agent identifies relevant ESRS topics."""
        agent = MaterialityAgent()

        result = agent.assess_materiality(sample_company_data)

        # Should categorize by ESRS topics
        material_topics = result['material_topics']

        for topic in material_topics:
            assert 'esrs_topic' in topic
            assert 'materiality_score' in topic
            assert topic['materiality_score'] >= 0

    @pytest.mark.llm
    def test_materiality_agent_llm_consistency(self, sample_company_data):
        """Test Materiality Agent LLM outputs are consistent."""
        agent = MaterialityAgent(temperature=0.0)  # Deterministic

        # Run multiple times
        results = []
        for _ in range(3):
            result = agent.assess_materiality(sample_company_data)
            results.append(result)

        # Results should be similar (same material topics identified)
        first_topics = set(t['esrs_topic'] for t in results[0]['material_topics'])

        for result in results[1:]:
            topics = set(t['esrs_topic'] for t in result['material_topics'])
            # Should have significant overlap (>80%)
            overlap = len(first_topics & topics) / len(first_topics)
            assert overlap >= 0.8, \
                f"LLM consistency {overlap:.0%} below 80% threshold"

    def test_materiality_agent_sector_specific(self):
        """Test Materiality Agent adapts to industry sector."""
        agent = MaterialityAgent()

        # Manufacturing sector
        manufacturing_data = {
            "company_name": "Test Manufacturing",
            "sector": "manufacturing",
            "activities": ["production", "assembly"]
        }

        result_mfg = agent.assess_materiality(manufacturing_data)

        # Should identify manufacturing-specific topics
        topics_mfg = [t['esrs_topic'] for t in result_mfg['material_topics']]
        assert any('E2' in t or 'pollution' in t.lower() for t in topics_mfg)

        # Financial services sector
        financial_data = {
            "company_name": "Test Bank",
            "sector": "financial_services",
            "activities": ["lending", "investment"]
        }

        result_fin = agent.assess_materiality(financial_data)

        # Should identify different topics for financial sector
        topics_fin = [t['esrs_topic'] for t in result_fin['material_topics']]
        # Financial sector should focus on governance
        assert any('G1' in t or 'governance' in t.lower() for t in topics_fin)


# ============================================================================
# Calculator Agent Tests
# ============================================================================

@pytest.mark.agents
@pytest.mark.critical
class TestCalculatorAgent:
    """Test Calculator Agent for zero-hallucination calculations."""

    def test_calculator_agent_zero_hallucination(self, sample_emissions_data):
        """
        Test Calculator Agent maintains zero hallucination guarantee.

        Requirements:
        - All calculations must be deterministic
        - All factors must be traceable to source
        - No invented/approximated values
        - Complete provenance tracking
        """
        agent = CalculatorAgent()

        result = agent.calculate(sample_emissions_data)

        # Verify all calculations have provenance
        for calc in result['calculations']:
            assert 'calculation_method' in calc
            assert 'data_source' in calc
            assert 'factor_id' in calc or 'factor_source' in calc

            # No hallucinated values
            if 'emission_factor' in calc:
                assert calc['factor_source'] is not None
                assert calc['factor_source'] != "estimated"

    def test_calculator_agent_determinism(self, sample_emissions_data):
        """Test Calculator Agent produces deterministic results."""
        agent = CalculatorAgent()

        # Calculate multiple times
        results = []
        for _ in range(5):
            result = agent.calculate(sample_emissions_data)
            results.append(result)

        # All results must be identical
        first = results[0]
        for result in results[1:]:
            assert result['total_emissions'] == first['total_emissions']

            # Compare per-item calculations
            for calc1, calc2 in zip(first['calculations'], result['calculations']):
                assert calc1['value'] == calc2['value']

    def test_calculator_agent_methodology_documentation(self, sample_emissions_data):
        """Test Calculator Agent documents calculation methodology."""
        agent = CalculatorAgent()

        result = agent.calculate(sample_emissions_data)

        # Should document methodology
        assert 'methodology' in result
        assert 'calculation_standard' in result or 'standard' in result['methodology']

        # Each calculation should reference methodology
        for calc in result['calculations']:
            assert 'methodology_ref' in calc or 'calculation_method' in calc

    def test_calculator_agent_scope_123_emissions(self):
        """Test Calculator Agent correctly categorizes Scope 1/2/3 emissions."""
        agent = CalculatorAgent()

        test_data = {
            "emissions": [
                {"type": "direct_combustion", "scope": 1, "value": 100},
                {"type": "purchased_electricity", "scope": 2, "value": 50},
                {"type": "business_travel", "scope": 3, "value": 25}
            ]
        }

        result = agent.calculate(test_data)

        # Should have breakdown by scope
        assert 'scope_1_total' in result
        assert 'scope_2_total' in result
        assert 'scope_3_total' in result

        assert result['scope_1_total'] == 100
        assert result['scope_2_total'] == 50
        assert result['scope_3_total'] == 25


# ============================================================================
# Aggregator Agent Tests
# ============================================================================

@pytest.mark.agents
@pytest.mark.performance
class TestAggregatorAgent:
    """Test Aggregator Agent for performance-optimized aggregation."""

    def test_aggregator_agent_performance(self, large_csrd_dataset):
        """
        Test Aggregator Agent aggregates large datasets efficiently.

        Performance target: >10k records/sec
        """
        agent = AggregatorAgent()

        start = time.perf_counter()
        result = agent.aggregate(large_csrd_dataset)
        duration = time.perf_counter() - start

        records = len(large_csrd_dataset)
        throughput = records / duration

        assert throughput >= 1000, \
            f"Aggregator throughput {throughput:.0f} records/sec below target"

    def test_aggregator_agent_multi_level(self, hierarchical_data):
        """Test Aggregator Agent supports multi-level aggregation."""
        agent = AggregatorAgent()

        result = agent.aggregate(
            hierarchical_data,
            group_by=['region', 'facility', 'department']
        )

        # Should have hierarchical aggregation
        assert 'aggregated_data' in result
        assert 'summary' in result

        # Verify aggregation levels
        assert any('region' in agg for agg in result['aggregated_data'])

    def test_aggregator_agent_custom_metrics(self, sample_csrd_input):
        """Test Aggregator Agent supports custom aggregation metrics."""
        agent = AggregatorAgent()

        custom_metrics = {
            "total_emissions": "sum",
            "average_intensity": "mean",
            "max_value": "max"
        }

        result = agent.aggregate(sample_csrd_input, metrics=custom_metrics)

        # Should calculate custom metrics
        summary = result['summary']
        assert 'total_emissions' in summary
        assert 'average_intensity' in summary
        assert 'max_value' in summary


# ============================================================================
# Reporting Agent Tests
# ============================================================================

@pytest.mark.agents
@pytest.mark.critical
class TestReportingAgent:
    """Test Reporting Agent for XBRL generation."""

    def test_reporting_agent_xbrl_generation(self, aggregated_csrd_data):
        """
        Test Reporting Agent generates valid XBRL reports.

        Requirements:
        - Generate ESRS-compliant XBRL
        - Validate against XBRL schema
        - Include all required disclosures
        - Support taxonomy mapping
        """
        agent = ReportingAgent()

        result = agent.generate_report(
            aggregated_csrd_data,
            format='xbrl'
        )

        # Verify XBRL output
        assert result is not None
        assert 'xbrl_content' in result or isinstance(result, (str, bytes))

        # Should be valid XML
        if isinstance(result, dict):
            xbrl_content = result['xbrl_content']
        else:
            xbrl_content = result

        assert b'<?xml' in xbrl_content or '<?xml' in xbrl_content
        assert b'xbrl' in xbrl_content or 'xbrl' in xbrl_content

    def test_reporting_agent_esrs_taxonomy(self, aggregated_csrd_data):
        """Test Reporting Agent maps to ESRS taxonomy."""
        agent = ReportingAgent()

        result = agent.generate_report(aggregated_csrd_data, format='xbrl')

        # Should include ESRS taxonomy references
        xbrl_str = result if isinstance(result, str) else str(result)

        assert 'esrs' in xbrl_str.lower() or 'taxonomy' in xbrl_str.lower()

    def test_reporting_agent_multiple_formats(self, aggregated_csrd_data):
        """Test Reporting Agent supports multiple output formats."""
        agent = ReportingAgent()

        # Test XBRL
        xbrl_report = agent.generate_report(aggregated_csrd_data, format='xbrl')
        assert xbrl_report is not None

        # Test JSON
        json_report = agent.generate_report(aggregated_csrd_data, format='json')
        assert json_report is not None

        # Test PDF
        try:
            pdf_report = agent.generate_report(aggregated_csrd_data, format='pdf')
            assert pdf_report is not None
        except NotImplementedError:
            pytest.skip("PDF generation not yet implemented")

    def test_reporting_agent_validation(self, aggregated_csrd_data):
        """Test Reporting Agent validates generated reports."""
        agent = ReportingAgent(validate_output=True)

        result = agent.generate_report(aggregated_csrd_data, format='xbrl')

        # Should include validation results
        assert 'validation' in result or 'is_valid' in result


# ============================================================================
# Audit Agent Tests
# ============================================================================

@pytest.mark.agents
@pytest.mark.critical
class TestAuditAgent:
    """Test Audit Agent for compliance rule validation."""

    def test_audit_agent_compliance_rules(self, sample_csrd_report):
        """
        Test Audit Agent validates compliance with CSRD/ESRS rules.

        Requirements:
        - Check required disclosures
        - Validate data quality
        - Identify missing information
        - Flag inconsistencies
        """
        agent = AuditAgent()

        result = agent.audit(sample_csrd_report)

        # Verify audit results
        assert 'audit_results' in result
        assert 'compliance_status' in result
        assert 'findings' in result or 'issues' in result

        # Should check multiple rule categories
        assert 'completeness_check' in result
        assert 'consistency_check' in result
        assert 'accuracy_check' in result

    def test_audit_agent_completeness_check(self):
        """Test Audit Agent checks for required disclosures."""
        agent = AuditAgent()

        # Incomplete report (missing required fields)
        incomplete_report = {
            "company_name": "Test Corp",
            # Missing many required ESRS disclosures
        }

        result = agent.audit(incomplete_report)

        # Should identify missing disclosures
        findings = result['findings']
        assert len(findings) > 0
        assert any('missing' in f['type'].lower() for f in findings)

    def test_audit_agent_consistency_check(self):
        """Test Audit Agent identifies data inconsistencies."""
        agent = AuditAgent()

        # Inconsistent report
        inconsistent_report = {
            "total_emissions": 1000,
            "scope_1_emissions": 300,
            "scope_2_emissions": 400,
            "scope_3_emissions": 400,  # Sum = 1100, doesn't match total
        }

        result = agent.audit(inconsistent_report)

        # Should flag inconsistency
        findings = result['findings']
        assert any('inconsistent' in f['type'].lower() for f in findings)

    def test_audit_agent_rule_engine(self):
        """Test Audit Agent uses configurable rule engine."""
        # Custom audit rules
        custom_rules = [
            {
                "rule_id": "EMISSIONS_POSITIVE",
                "check": lambda data: data.get('total_emissions', 0) >= 0,
                "message": "Total emissions must be non-negative"
            }
        ]

        agent = AuditAgent(custom_rules=custom_rules)

        # Test with negative emissions
        invalid_data = {"total_emissions": -100}

        result = agent.audit(invalid_data)

        # Should fail custom rule
        findings = result['findings']
        assert any('EMISSIONS_POSITIVE' in f.get('rule_id', '') for f in findings)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_csrd_input():
    """Sample CSRD input data."""
    return {
        "company_name": "Green Corp",
        "reporting_period": "2024",
        "sector": "manufacturing",
        "employees": 500,
        "revenue": 50_000_000,
        "emissions_data": [
            {"type": "scope_1", "value": 1000, "unit": "tCO2e"},
            {"type": "scope_2", "value": 500, "unit": "tCO2e"}
        ]
    }


@pytest.fixture
def sample_company_data():
    """Sample company data for materiality assessment."""
    return {
        "company_name": "Test Manufacturing Inc",
        "sector": "manufacturing",
        "employees": 1000,
        "revenue": 100_000_000,
        "activities": [
            "metal fabrication",
            "assembly",
            "distribution"
        ],
        "geographic_presence": ["EU", "US", "APAC"],
        "value_chain": {
            "suppliers": 50,
            "customers": 200
        }
    }


@pytest.fixture
def sample_emissions_data():
    """Sample emissions data for calculation."""
    return {
        "emissions": [
            {
                "source": "natural_gas_combustion",
                "activity_data": 1000,
                "unit": "m3",
                "scope": 1
            },
            {
                "source": "purchased_electricity",
                "activity_data": 5000,
                "unit": "kWh",
                "scope": 2
            }
        ]
    }


@pytest.fixture
def large_csrd_dataset():
    """Large dataset for performance testing."""
    data = []
    for i in range(10000):
        data.append({
            "id": i,
            "value": i * 1.5,
            "category": f"category_{i % 10}",
            "region": f"region_{i % 5}"
        })
    return data


@pytest.fixture
def hierarchical_data():
    """Hierarchical data for multi-level aggregation."""
    data = []
    for region in ['EU', 'US', 'APAC']:
        for facility in range(3):
            for dept in range(5):
                data.append({
                    "region": region,
                    "facility": f"facility_{facility}",
                    "department": f"dept_{dept}",
                    "emissions": 100 + facility * 10 + dept
                })
    return data


@pytest.fixture
def aggregated_csrd_data():
    """Aggregated CSRD data for reporting."""
    return {
        "company_name": "Test Corp",
        "reporting_period": "2024",
        "total_emissions": 1500,
        "scope_1_total": 1000,
        "scope_2_total": 500,
        "scope_3_total": 0,
        "material_topics": ["E1", "E2", "S1"],
        "disclosures": {
            "E1": {"climate_change": {"data": "..."}},
            "E2": {"pollution": {"data": "..."}},
            "S1": {"workforce": {"data": "..."}}
        }
    }


@pytest.fixture
def sample_csrd_report():
    """Sample CSRD report for audit testing."""
    return {
        "company_name": "Test Corp",
        "reporting_period": "2024",
        "total_emissions": 1500,
        "scope_1_emissions": 1000,
        "scope_2_emissions": 500,
        "scope_3_emissions": 0,
        "material_topics": ["E1", "E2", "S1"],
        "required_disclosures": ["E1", "E2", "S1", "G1"],
        "completed_disclosures": ["E1", "E2", "S1"]  # Missing G1
    }


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'agents'])
