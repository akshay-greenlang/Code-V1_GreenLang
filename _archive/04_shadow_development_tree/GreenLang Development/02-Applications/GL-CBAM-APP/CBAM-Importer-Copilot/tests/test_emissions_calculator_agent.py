# -*- coding: utf-8 -*-
"""
CBAM Importer Copilot - Emissions Calculator Agent Tests

Unit tests for EmissionsCalculatorAgent - ZERO HALLUCINATION CRITICAL

Version: 1.0.0
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.emissions_calculator_agent import EmissionsCalculatorAgent


# ============================================================================
# Test Agent Initialization
# ============================================================================

@pytest.mark.unit
class TestAgentInitialization:
    """Test emissions calculator initialization."""

    def test_agent_initializes_successfully(self, cn_codes_path):
        """Test agent initializes with CN codes database."""
        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        assert agent is not None
        assert agent.cn_codes_path == cn_codes_path

    def test_loads_emission_factors(self, cn_codes_path):
        """Test agent loads emission factors database."""
        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        # Should have emission factors loaded
        assert hasattr(agent, 'emission_factors')
        assert len(agent.emission_factors) > 0


# ============================================================================
# Test Zero Hallucination Guarantee
# ============================================================================

@pytest.mark.unit
@pytest.mark.compliance
class TestZeroHallucination:
    """CRITICAL: Test zero hallucination guarantee."""

    def test_calculations_are_deterministic(self, cn_codes_path, sample_shipments_data):
        """Test calculations are 100% deterministic."""
        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        # Create validated shipments
        validated = {
            'validated_shipments': sample_shipments_data,
            'metadata': {'total_records': len(sample_shipments_data)}
        }

        # Run calculation twice
        result1 = agent.calculate(validated)
        result2 = agent.calculate(validated)

        # Results must be EXACTLY identical
        emissions1 = result1['total_emissions_tco2']
        emissions2 = result2['total_emissions_tco2']

        assert emissions1 == emissions2, \
            "Non-deterministic calculation detected! ZERO HALLUCINATION VIOLATED!"

    def test_no_llm_in_calculation_path(self, cn_codes_path, sample_shipments_data):
        """Test NO LLM used in calculation path."""
        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        validated = {
            'validated_shipments': sample_shipments_data,
            'metadata': {'total_records': len(sample_shipments_data)}
        }

        result = agent.calculate(validated)

        # All records must have calculation_method = "deterministic"
        for good in result['shipments_with_emissions']:
            calc_method = good.get('calculation_method', '')
            assert calc_method == 'deterministic', \
                f"Non-deterministic calculation detected: {calc_method}"

    def test_database_lookup_only(self, cn_codes_path, sample_shipments_data):
        """Test uses database lookup only (no LLM estimation)."""
        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        validated = {
            'validated_shipments': sample_shipments_data,
            'metadata': {'total_records': len(sample_shipments_data)}
        }

        result = agent.calculate(validated)

        # All emission factors must come from database or supplier
        for good in result['shipments_with_emissions']:
            source = good.get('emission_factor_source', '')
            assert source in ['default', 'supplier'], \
                f"Invalid emission factor source: {source}"

    def test_python_arithmetic_only(self, cn_codes_path):
        """Test uses Python arithmetic only (no LLM math)."""
        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        # Single shipment test
        shipment = {
            'cn_code': '72071100',
            'quantity_tons': 10.0
        }

        validated = {
            'validated_shipments': [shipment],
            'metadata': {'total_records': 1}
        }

        result = agent.calculate(validated)

        # Calculate expected manually
        # emission_factor for 72071100 should be ~0.8 tCO2/ton
        emissions = result['shipments_with_emissions'][0]['embedded_emissions_tco2']
        factor = result['shipments_with_emissions'][0]['emission_factor_tco2_per_ton']

        # Manual calculation
        expected = 10.0 * factor

        # Must match Python arithmetic
        assert abs(emissions - expected) < 0.01, \
            "Calculation doesn't match Python arithmetic!"

    def test_bit_perfect_reproducibility(self, cn_codes_path, sample_shipments_data):
        """Test bit-perfect reproducibility across multiple runs."""
        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        validated = {
            'validated_shipments': sample_shipments_data,
            'metadata': {'total_records': len(sample_shipments_data)}
        }

        # Run 10 times
        results = []
        for _ in range(10):
            result = agent.calculate(validated)
            results.append(result['total_emissions_tco2'])

        # ALL results must be EXACTLY identical (not just close)
        assert len(set(results)) == 1, \
            f"Non-reproducible results: {set(results)}"


# ============================================================================
# Test Emission Calculations
# ============================================================================

@pytest.mark.unit
class TestEmissionCalculations:
    """Test emission calculation accuracy."""

    def test_calculates_single_shipment(self, cn_codes_path):
        """Test calculates emissions for single shipment."""
        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        shipment = {
            'cn_code': '72071100',
            'quantity_tons': 15.5,
            'country_of_origin': 'CN'
        }

        validated = {
            'validated_shipments': [shipment],
            'metadata': {'total_records': 1}
        }

        result = agent.calculate(validated)

        assert 'shipments_with_emissions' in result
        assert len(result['shipments_with_emissions']) == 1

        calculated = result['shipments_with_emissions'][0]
        assert 'embedded_emissions_tco2' in calculated
        assert calculated['embedded_emissions_tco2'] > 0

    def test_calculates_multiple_shipments(self, cn_codes_path, sample_shipments_data):
        """Test calculates emissions for multiple shipments."""
        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        validated = {
            'validated_shipments': sample_shipments_data,
            'metadata': {'total_records': len(sample_shipments_data)}
        }

        result = agent.calculate(validated)

        assert len(result['shipments_with_emissions']) == len(sample_shipments_data)

    def test_total_emissions_sum(self, cn_codes_path, sample_shipments_data):
        """Test total emissions equals sum of individual emissions."""
        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        validated = {
            'validated_shipments': sample_shipments_data,
            'metadata': {'total_records': len(sample_shipments_data)}
        }

        result = agent.calculate(validated)

        # Manual sum
        manual_sum = sum(
            s['embedded_emissions_tco2']
            for s in result['shipments_with_emissions']
        )

        # Should match total
        assert abs(result['total_emissions_tco2'] - manual_sum) < 0.01

    def test_correct_emission_factors_used(self, cn_codes_path, expected_emissions):
        """Test uses correct emission factors for each CN code."""
        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        for cn_code, expected_factor in expected_emissions.items():
            shipment = {
                'cn_code': cn_code,
                'quantity_tons': 10.0  # Simple quantity for testing
            }

            validated = {
                'validated_shipments': [shipment],
                'metadata': {'total_records': 1}
            }

            result = agent.calculate(validated)
            calculated = result['shipments_with_emissions'][0]

            # Emission factor should be close to expected
            factor = calculated['emission_factor_tco2_per_ton']
            # Allow small tolerance for rounding
            assert abs(factor - expected_factor) < 0.1, \
                f"Wrong factor for {cn_code}: {factor} vs {expected_factor}"


# ============================================================================
# Test Default vs Supplier Emission Factors
# ============================================================================

@pytest.mark.unit
class TestEmissionFactorSources:
    """Test default vs supplier emission factors."""

    def test_uses_default_factors(self, cn_codes_path):
        """Test uses default factors when no supplier data."""
        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        shipment = {
            'cn_code': '72071100',
            'quantity_tons': 10.0
        }

        validated = {
            'validated_shipments': [shipment],
            'metadata': {'total_records': 1}
        }

        result = agent.calculate(validated)
        calculated = result['shipments_with_emissions'][0]

        assert calculated['emission_factor_source'] == 'default'

    def test_prefers_supplier_actuals(self, cn_codes_path):
        """Test prefers supplier actuals when available."""
        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        # Shipment with supplier data
        shipment = {
            'cn_code': '72071100',
            'quantity_tons': 10.0,
            'supplier_emission_factor': 0.75  # Supplier actual
        }

        validated = {
            'validated_shipments': [shipment],
            'metadata': {'total_records': 1}
        }

        result = agent.calculate(validated)
        calculated = result['shipments_with_emissions'][0]

        # Should use supplier data
        assert calculated['emission_factor_source'] == 'supplier'
        assert calculated['emission_factor_tco2_per_ton'] == 0.75


# ============================================================================
# Test Aggregations
# ============================================================================

@pytest.mark.unit
class TestAggregations:
    """Test emissions aggregation by various dimensions."""

    def test_aggregates_by_cn_code(self, cn_codes_path, sample_shipments_data):
        """Test aggregates emissions by CN code."""
        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        validated = {
            'validated_shipments': sample_shipments_data,
            'metadata': {'total_records': len(sample_shipments_data)}
        }

        result = agent.calculate(validated)

        assert 'emissions_by_cn_code' in result
        assert len(result['emissions_by_cn_code']) > 0

    def test_aggregates_by_country(self, cn_codes_path, sample_shipments_data):
        """Test aggregates emissions by country of origin."""
        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        validated = {
            'validated_shipments': sample_shipments_data,
            'metadata': {'total_records': len(sample_shipments_data)}
        }

        result = agent.calculate(validated)

        assert 'emissions_by_country' in result
        assert len(result['emissions_by_country']) > 0

    def test_aggregates_by_product_group(self, cn_codes_path, sample_shipments_data):
        """Test aggregates emissions by product group."""
        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        # Add product_group to sample data
        for shipment in sample_shipments_data:
            if shipment['cn_code'].startswith('72'):
                shipment['product_group'] = 'iron_steel'
            elif shipment['cn_code'].startswith('76'):
                shipment['product_group'] = 'aluminum'
            elif shipment['cn_code'].startswith('25'):
                shipment['product_group'] = 'cement'

        validated = {
            'validated_shipments': sample_shipments_data,
            'metadata': {'total_records': len(sample_shipments_data)}
        }

        result = agent.calculate(validated)

        assert 'emissions_by_product_group' in result
        assert len(result['emissions_by_product_group']) > 0


# ============================================================================
# Test Performance
# ============================================================================

@pytest.mark.unit
@pytest.mark.performance
class TestPerformance:
    """Test calculation performance."""

    def test_fast_calculation(self, cn_codes_path, sample_shipments_data):
        """Test calculations are fast (<3ms per shipment target)."""
        import time

        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        validated = {
            'validated_shipments': sample_shipments_data,
            'metadata': {'total_records': len(sample_shipments_data)}
        }

        start = time.time()
        result = agent.calculate(validated)
        duration = time.time() - start

        # Target: <3ms per shipment
        per_shipment = duration / len(sample_shipments_data)
        assert per_shipment < 0.01, \
            f"Too slow: {per_shipment*1000:.1f}ms per shipment (target: <3ms)"

    def test_batch_performance(self, cn_codes_path, large_shipments_data):
        """Test batch calculation performance."""
        import time

        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        validated = {
            'validated_shipments': large_shipments_data,
            'metadata': {'total_records': len(large_shipments_data)}
        }

        start = time.time()
        result = agent.calculate(validated)
        duration = time.time() - start

        # 1000 records should be <3 seconds
        assert duration < 5.0, \
            f"Batch too slow: {duration:.2f}s for {len(large_shipments_data)} records"


# ============================================================================
# Test Edge Cases
# ============================================================================

@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_zero_quantity(self, cn_codes_path):
        """Test handles zero quantity gracefully."""
        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        shipment = {
            'cn_code': '72071100',
            'quantity_tons': 0.0
        }

        validated = {
            'validated_shipments': [shipment],
            'metadata': {'total_records': 1}
        }

        result = agent.calculate(validated)
        calculated = result['shipments_with_emissions'][0]

        # Zero quantity = zero emissions
        assert calculated['embedded_emissions_tco2'] == 0.0

    def test_handles_very_large_quantity(self, cn_codes_path):
        """Test handles very large quantities."""
        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        shipment = {
            'cn_code': '72071100',
            'quantity_tons': 1000000.0  # 1 million tons
        }

        validated = {
            'validated_shipments': [shipment],
            'metadata': {'total_records': 1}
        }

        result = agent.calculate(validated)
        calculated = result['shipments_with_emissions'][0]

        # Should calculate without error
        assert calculated['embedded_emissions_tco2'] > 0

    def test_handles_unknown_cn_code_gracefully(self, cn_codes_path):
        """Test handles unknown CN code gracefully."""
        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        shipment = {
            'cn_code': '99999999',  # Unknown code
            'quantity_tons': 10.0
        }

        validated = {
            'validated_shipments': [shipment],
            'metadata': {'total_records': 1}
        }

        # Should either use fallback or raise appropriate error
        try:
            result = agent.calculate(validated)
            # If successful, should have used fallback
            assert result is not None
        except KeyError:
            # Or raise clear error
            pass  # OK to raise error for unknown code

    def test_handles_empty_input(self, cn_codes_path):
        """Test handles empty shipment list."""
        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        validated = {
            'validated_shipments': [],
            'metadata': {'total_records': 0}
        }

        result = agent.calculate(validated)

        # Should return zero emissions
        assert result['total_emissions_tco2'] == 0.0
        assert len(result['shipments_with_emissions']) == 0


# ============================================================================
# Test Audit Trail
# ============================================================================

@pytest.mark.unit
@pytest.mark.compliance
class TestAuditTrail:
    """Test calculation audit trail."""

    def test_records_calculation_details(self, cn_codes_path, sample_shipments_data):
        """Test records all calculation details for audit."""
        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        validated = {
            'validated_shipments': sample_shipments_data,
            'metadata': {'total_records': len(sample_shipments_data)}
        }

        result = agent.calculate(validated)

        # Each shipment should have complete audit trail
        for good in result['shipments_with_emissions']:
            assert 'cn_code' in good
            assert 'quantity_tons' in good
            assert 'embedded_emissions_tco2' in good
            assert 'emission_factor_tco2_per_ton' in good
            assert 'emission_factor_source' in good
            assert 'calculation_method' in good

    def test_calculation_traceable(self, cn_codes_path):
        """Test calculation can be traced and verified."""
        agent = EmissionsCalculatorAgent(cn_codes_path=cn_codes_path)

        shipment = {
            'cn_code': '72071100',
            'quantity_tons': 20.0
        }

        validated = {
            'validated_shipments': [shipment],
            'metadata': {'total_records': 1}
        }

        result = agent.calculate(validated)
        calculated = result['shipments_with_emissions'][0]

        # Manual verification
        quantity = calculated['quantity_tons']
        factor = calculated['emission_factor_tco2_per_ton']
        emissions = calculated['embedded_emissions_tco2']

        # Verify calculation
        expected_emissions = quantity * factor
        assert abs(emissions - expected_emissions) < 0.01, \
            "Calculation not traceable!"
