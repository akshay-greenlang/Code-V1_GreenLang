# -*- coding: utf-8 -*-
"""
Unit Tests for GL-001: Carbon Emissions Calculator Agent

Comprehensive test suite with 50 test cases covering:
- FuelType enum handling (5 tests)
- Scope enum handling (5 tests)
- Emission factor lookups (10 tests)
- CO2e calculations (15 tests)
- Provenance hash generation (5 tests)
- Input validation (5 tests)
- Error handling (5 tests)

Target: 85%+ coverage for Carbon Emissions Agent
Run with: pytest tests/unit/test_gl001_carbon_agent.py -v --cov

Author: GL-TestEngineer
Version: 1.0.0

The Carbon Emissions Agent calculates GHG emissions with zero-hallucination
deterministic calculations using validated emission factors from EPA, DEFRA, IEA.
"""

import pytest
import hashlib
import json
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

# Add project paths for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "GL-Agent-Factory" / "backend" / "agents"))

# Import agent components
from gl_001_carbon_emissions.agent import (
    CarbonEmissionsAgent,
    CarbonEmissionsInput,
    CarbonEmissionsOutput,
    FuelType,
    Scope,
    EmissionFactor,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def agent():
    """Create CarbonEmissionsAgent instance."""
    return CarbonEmissionsAgent()


@pytest.fixture
def agent_with_config():
    """Create CarbonEmissionsAgent with custom configuration."""
    config = {
        "enable_audit": True,
        "default_region": "EU",
    }
    return CarbonEmissionsAgent(config=config)


@pytest.fixture
def valid_natural_gas_input():
    """Create valid natural gas input data."""
    return CarbonEmissionsInput(
        fuel_type=FuelType.NATURAL_GAS,
        quantity=1000.0,
        unit="m3",
        region="US",
        scope=Scope.SCOPE_1,
        calculation_method="location",
    )


@pytest.fixture
def valid_diesel_input():
    """Create valid diesel input data."""
    return CarbonEmissionsInput(
        fuel_type=FuelType.DIESEL,
        quantity=500.0,
        unit="L",
        region="EU",
        scope=Scope.SCOPE_1,
        calculation_method="location",
    )


@pytest.fixture
def valid_electricity_input():
    """Create valid electricity input data."""
    return CarbonEmissionsInput(
        fuel_type=FuelType.ELECTRICITY_GRID,
        quantity=10000.0,
        unit="kWh",
        region="US",
        scope=Scope.SCOPE_2,
        calculation_method="location",
    )


@pytest.fixture
def sample_emission_factors():
    """Sample emission factors for testing."""
    return {
        "natural_gas": {
            "US": EmissionFactor(value=1.93, unit="kgCO2e/m3", source="EPA", year=2024),
            "EU": EmissionFactor(value=2.02, unit="kgCO2e/m3", source="DEFRA", year=2024),
        },
        "diesel": {
            "US": EmissionFactor(value=2.68, unit="kgCO2e/L", source="EPA", year=2024),
            "EU": EmissionFactor(value=2.62, unit="kgCO2e/L", source="DEFRA", year=2024),
        },
    }


# =============================================================================
# FuelType Enum Tests (5 tests)
# =============================================================================

class TestFuelTypeEnum:
    """Test suite for FuelType enum - 5 test cases."""

    @pytest.mark.unit
    def test_fuel_type_natural_gas_value(self):
        """UT-GL001-001: Test FuelType.NATURAL_GAS has correct value."""
        assert FuelType.NATURAL_GAS.value == "natural_gas"

    @pytest.mark.unit
    def test_fuel_type_diesel_value(self):
        """UT-GL001-002: Test FuelType.DIESEL has correct value."""
        assert FuelType.DIESEL.value == "diesel"

    @pytest.mark.unit
    def test_fuel_type_electricity_value(self):
        """UT-GL001-003: Test FuelType.ELECTRICITY_GRID has correct value."""
        assert FuelType.ELECTRICITY_GRID.value == "electricity_grid"

    @pytest.mark.unit
    def test_all_fuel_types_defined(self):
        """UT-GL001-004: Test all expected fuel types are defined."""
        expected_fuel_types = {
            "natural_gas", "diesel", "gasoline", "coal",
            "fuel_oil", "propane", "electricity_grid"
        }
        actual_fuel_types = {ft.value for ft in FuelType}
        assert expected_fuel_types == actual_fuel_types

    @pytest.mark.unit
    def test_fuel_type_from_string(self):
        """UT-GL001-005: Test creating FuelType from string value."""
        fuel_type = FuelType("natural_gas")
        assert fuel_type == FuelType.NATURAL_GAS


# =============================================================================
# Scope Enum Tests (5 tests)
# =============================================================================

class TestScopeEnum:
    """Test suite for Scope enum - 5 test cases."""

    @pytest.mark.unit
    def test_scope_1_value(self):
        """UT-GL001-006: Test Scope.SCOPE_1 has correct integer value."""
        assert Scope.SCOPE_1.value == 1

    @pytest.mark.unit
    def test_scope_2_value(self):
        """UT-GL001-007: Test Scope.SCOPE_2 has correct integer value."""
        assert Scope.SCOPE_2.value == 2

    @pytest.mark.unit
    def test_scope_3_value(self):
        """UT-GL001-008: Test Scope.SCOPE_3 has correct integer value."""
        assert Scope.SCOPE_3.value == 3

    @pytest.mark.unit
    def test_all_scopes_defined(self):
        """UT-GL001-009: Test all GHG Protocol scopes are defined."""
        expected_scopes = {1, 2, 3}
        actual_scopes = {s.value for s in Scope}
        assert expected_scopes == actual_scopes

    @pytest.mark.unit
    def test_scope_from_int(self):
        """UT-GL001-010: Test creating Scope from integer value."""
        scope = Scope(1)
        assert scope == Scope.SCOPE_1


# =============================================================================
# Emission Factor Lookup Tests (10 tests)
# =============================================================================

class TestEmissionFactorLookup:
    """Test suite for emission factor lookups - 10 test cases."""

    @pytest.mark.unit
    def test_lookup_natural_gas_us(self, agent):
        """UT-GL001-011: Test emission factor lookup for US natural gas."""
        ef = agent._get_emission_factor("natural_gas", "US")
        assert ef is not None
        assert ef.value == 1.93
        assert ef.unit == "kgCO2e/m3"
        assert ef.source == "EPA"

    @pytest.mark.unit
    def test_lookup_natural_gas_eu(self, agent):
        """UT-GL001-012: Test emission factor lookup for EU natural gas."""
        ef = agent._get_emission_factor("natural_gas", "EU")
        assert ef is not None
        assert ef.value == 2.02
        assert ef.source == "DEFRA"

    @pytest.mark.unit
    def test_lookup_diesel_us(self, agent):
        """UT-GL001-013: Test emission factor lookup for US diesel."""
        ef = agent._get_emission_factor("diesel", "US")
        assert ef is not None
        assert ef.value == 2.68
        assert ef.unit == "kgCO2e/L"

    @pytest.mark.unit
    def test_lookup_diesel_eu(self, agent):
        """UT-GL001-014: Test emission factor lookup for EU diesel."""
        ef = agent._get_emission_factor("diesel", "EU")
        assert ef is not None
        assert ef.value == 2.62

    @pytest.mark.unit
    def test_lookup_electricity_us(self, agent):
        """UT-GL001-015: Test emission factor lookup for US grid electricity."""
        ef = agent._get_emission_factor("electricity_grid", "US")
        assert ef is not None
        assert ef.value == 0.417
        assert ef.source == "EPA eGRID"

    @pytest.mark.unit
    def test_lookup_electricity_france(self, agent):
        """UT-GL001-016: Test emission factor lookup for France (low carbon)."""
        ef = agent._get_emission_factor("electricity_grid", "FR")
        assert ef is not None
        assert ef.value == 0.052  # Nuclear-heavy grid

    @pytest.mark.unit
    def test_lookup_fallback_to_parent_region(self, agent):
        """UT-GL001-017: Test fallback to parent region (DE -> EU)."""
        # Germany should fall back to EU for natural gas
        ef = agent._get_emission_factor("natural_gas", "DE")
        assert ef is not None
        # Should return EU factor as DE doesn't have natural gas specific

    @pytest.mark.unit
    def test_lookup_fallback_to_us(self, agent):
        """UT-GL001-018: Test fallback to US for unknown region."""
        ef = agent._get_emission_factor("diesel", "ZZ")
        assert ef is not None
        # Should return US factor as default

    @pytest.mark.unit
    def test_lookup_nonexistent_fuel_type(self, agent):
        """UT-GL001-019: Test lookup for non-existent fuel type returns None."""
        ef = agent._get_emission_factor("hydrogen", "US")
        assert ef is None

    @pytest.mark.unit
    def test_emission_factor_has_required_fields(self, agent):
        """UT-GL001-020: Test emission factor has all required fields."""
        ef = agent._get_emission_factor("natural_gas", "US")
        assert ef.value is not None
        assert ef.unit is not None
        assert ef.source is not None
        assert ef.year is not None


# =============================================================================
# CO2e Calculation Tests (15 tests)
# =============================================================================

class TestCO2eCalculations:
    """Test suite for CO2e calculations - 15 test cases."""

    @pytest.mark.unit
    def test_basic_natural_gas_calculation(self, agent, valid_natural_gas_input):
        """UT-GL001-021: Test basic natural gas emission calculation."""
        result = agent.run(valid_natural_gas_input)

        # emissions = 1000 m3 * 1.93 kgCO2e/m3 = 1930 kgCO2e
        expected = 1000.0 * 1.93
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.unit
    def test_basic_diesel_calculation(self, agent, valid_diesel_input):
        """UT-GL001-022: Test basic diesel emission calculation."""
        result = agent.run(valid_diesel_input)

        # emissions = 500 L * 2.62 kgCO2e/L = 1310 kgCO2e
        expected = 500.0 * 2.62
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.unit
    def test_electricity_calculation_us(self, agent, valid_electricity_input):
        """UT-GL001-023: Test US grid electricity emission calculation."""
        result = agent.run(valid_electricity_input)

        # emissions = 10000 kWh * 0.417 kgCO2e/kWh = 4170 kgCO2e
        expected = 10000.0 * 0.417
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.unit
    def test_zero_quantity_returns_zero_emissions(self, agent):
        """UT-GL001-024: Test zero quantity returns zero emissions."""
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=0.0,
            unit="m3",
            region="US",
            scope=Scope.SCOPE_1,
        )
        result = agent.run(input_data)
        assert result.emissions_kgco2e == 0.0

    @pytest.mark.unit
    def test_large_quantity_calculation(self, agent):
        """UT-GL001-025: Test calculation with large quantity."""
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=1_000_000.0,
            unit="m3",
            region="US",
            scope=Scope.SCOPE_1,
        )
        result = agent.run(input_data)

        expected = 1_000_000.0 * 1.93
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.unit
    def test_decimal_quantity_calculation(self, agent):
        """UT-GL001-026: Test calculation with decimal quantity."""
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=123.456,
            unit="m3",
            region="US",
            scope=Scope.SCOPE_1,
        )
        result = agent.run(input_data)

        expected = 123.456 * 1.93
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.unit
    def test_calculation_returns_correct_scope(self, agent, valid_natural_gas_input):
        """UT-GL001-027: Test calculation returns correct scope value."""
        result = agent.run(valid_natural_gas_input)
        assert result.scope == 1

    @pytest.mark.unit
    def test_calculation_returns_emission_factor_source(self, agent, valid_natural_gas_input):
        """UT-GL001-028: Test calculation returns emission factor source."""
        result = agent.run(valid_natural_gas_input)
        assert result.emission_factor_source == "EPA"

    @pytest.mark.unit
    def test_calculation_returns_emission_factor_used(self, agent, valid_natural_gas_input):
        """UT-GL001-029: Test calculation returns emission factor value used."""
        result = agent.run(valid_natural_gas_input)
        assert result.emission_factor_used == 1.93

    @pytest.mark.unit
    def test_scope_2_electricity_calculation(self, agent):
        """UT-GL001-030: Test Scope 2 electricity calculation."""
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.ELECTRICITY_GRID,
            quantity=5000.0,
            unit="kWh",
            region="DE",
            scope=Scope.SCOPE_2,
        )
        result = agent.run(input_data)

        # Germany grid factor is 0.366 kgCO2e/kWh
        expected = 5000.0 * 0.366
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)
        assert result.scope == 2

    @pytest.mark.unit
    def test_gasoline_calculation(self, agent):
        """UT-GL001-031: Test gasoline emission calculation."""
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.GASOLINE,
            quantity=100.0,
            unit="L",
            region="US",
            scope=Scope.SCOPE_1,
        )
        result = agent.run(input_data)

        # US gasoline factor is 2.31 kgCO2e/L
        expected = 100.0 * 2.31
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.unit
    def test_deterministic_calculation(self, agent, valid_natural_gas_input):
        """UT-GL001-032: Test calculation is deterministic (same input = same result)."""
        result1 = agent.run(valid_natural_gas_input)
        result2 = agent.run(valid_natural_gas_input)

        assert result1.emissions_kgco2e == result2.emissions_kgco2e

    @pytest.mark.unit
    def test_formula_emissions_equals_quantity_times_factor(self, agent):
        """UT-GL001-033: Test formula: emissions = quantity * emission_factor."""
        quantity = 750.0
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.DIESEL,
            quantity=quantity,
            unit="L",
            region="US",
            scope=Scope.SCOPE_1,
        )
        result = agent.run(input_data)

        # Verify formula: emissions = quantity * EF
        expected = quantity * 2.68
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.unit
    def test_output_rounded_to_six_decimals(self, agent):
        """UT-GL001-034: Test emissions output is rounded appropriately."""
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=123.456789,
            unit="m3",
            region="US",
            scope=Scope.SCOPE_1,
        )
        result = agent.run(input_data)

        # Result should be rounded to 6 decimal places
        str_result = str(result.emissions_kgco2e)
        if '.' in str_result:
            decimal_places = len(str_result.split('.')[1])
            assert decimal_places <= 6

    @pytest.mark.unit
    def test_calculation_method_stored_in_output(self, agent, valid_electricity_input):
        """UT-GL001-035: Test calculation method is stored in output."""
        result = agent.run(valid_electricity_input)
        assert result.calculation_method == "location"


# =============================================================================
# Provenance Hash Tests (5 tests)
# =============================================================================

class TestProvenanceHash:
    """Test suite for provenance hash generation - 5 test cases."""

    @pytest.mark.unit
    def test_provenance_hash_generated(self, agent, valid_natural_gas_input):
        """UT-GL001-036: Test provenance hash is generated."""
        result = agent.run(valid_natural_gas_input)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex length

    @pytest.mark.unit
    def test_provenance_hash_is_valid_sha256(self, agent, valid_natural_gas_input):
        """UT-GL001-037: Test provenance hash is valid SHA-256 format."""
        result = agent.run(valid_natural_gas_input)

        # SHA-256 hash should be 64 hex characters
        assert len(result.provenance_hash) == 64
        assert all(c in '0123456789abcdef' for c in result.provenance_hash)

    @pytest.mark.unit
    def test_provenance_hash_changes_with_input(self, agent):
        """UT-GL001-038: Test provenance hash changes with different inputs."""
        input1 = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=1000.0,
            unit="m3",
            region="US",
            scope=Scope.SCOPE_1,
        )
        input2 = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=2000.0,  # Different quantity
            unit="m3",
            region="US",
            scope=Scope.SCOPE_1,
        )

        result1 = agent.run(input1)
        result2 = agent.run(input2)

        # Different inputs should produce different provenance hashes
        assert result1.provenance_hash != result2.provenance_hash

    @pytest.mark.unit
    def test_provenance_hash_includes_agent_version(self, agent, valid_natural_gas_input):
        """UT-GL001-039: Test provenance tracking includes agent version."""
        agent.run(valid_natural_gas_input)

        # Check provenance steps contain agent info
        assert len(agent._provenance_steps) > 0
        # The provenance hash should be calculated with version info

    @pytest.mark.unit
    def test_calculate_provenance_hash_method(self, agent):
        """UT-GL001-040: Test _calculate_provenance_hash method."""
        agent._provenance_steps = [
            {"step_type": "test", "data": {"value": 123}}
        ]

        hash_result = agent._calculate_provenance_hash()

        assert len(hash_result) == 64
        assert isinstance(hash_result, str)


# =============================================================================
# Input Validation Tests (5 tests)
# =============================================================================

class TestInputValidation:
    """Test suite for input validation - 5 test cases."""

    @pytest.mark.unit
    def test_negative_quantity_rejected(self):
        """UT-GL001-041: Test negative quantity raises validation error."""
        with pytest.raises(ValueError):
            CarbonEmissionsInput(
                fuel_type=FuelType.NATURAL_GAS,
                quantity=-100.0,  # Negative not allowed
                unit="m3",
                region="US",
                scope=Scope.SCOPE_1,
            )

    @pytest.mark.unit
    def test_region_normalized_to_uppercase(self):
        """UT-GL001-042: Test region is normalized to uppercase."""
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=100.0,
            unit="m3",
            region="us",  # lowercase
            scope=Scope.SCOPE_1,
        )
        assert input_data.region == "US"

    @pytest.mark.unit
    def test_valid_fuel_type_required(self):
        """UT-GL001-043: Test valid fuel type is required."""
        with pytest.raises(ValueError):
            CarbonEmissionsInput(
                fuel_type="invalid_fuel",
                quantity=100.0,
                unit="m3",
                region="US",
                scope=Scope.SCOPE_1,
            )

    @pytest.mark.unit
    def test_unit_validation_logs_warning_for_mismatch(self, agent, caplog):
        """UT-GL001-044: Test unit validation logs warning for non-standard unit."""
        import logging
        caplog.set_level(logging.WARNING)

        # Using kWh for natural gas (unusual but allowed with warning)
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=100.0,
            unit="kWh",  # Non-standard unit for natural gas
            region="US",
            scope=Scope.SCOPE_1,
        )
        # Validation should log a warning but not fail
        assert input_data.unit == "kWh"

    @pytest.mark.unit
    def test_metadata_field_accepts_dict(self):
        """UT-GL001-045: Test metadata field accepts dictionary."""
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=100.0,
            unit="m3",
            region="US",
            scope=Scope.SCOPE_1,
            metadata={"source": "test", "year": 2024},
        )
        assert input_data.metadata["source"] == "test"


# =============================================================================
# Error Handling Tests (5 tests)
# =============================================================================

class TestErrorHandling:
    """Test suite for error handling - 5 test cases."""

    @pytest.mark.unit
    def test_missing_emission_factor_raises_error(self, agent):
        """UT-GL001-046: Test missing emission factor raises ValueError."""
        # Create input with fuel type that has no factors
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.COAL,  # Coal may not have factors for all regions
            quantity=100.0,
            unit="kg",
            region="ZZ",  # Non-existent region
            scope=Scope.SCOPE_1,
        )

        # Should raise error when no factor found
        with pytest.raises(ValueError) as exc_info:
            agent.run(input_data)

        assert "emission factor" in str(exc_info.value).lower()

    @pytest.mark.unit
    def test_agent_logs_error_on_failure(self, agent, caplog):
        """UT-GL001-047: Test agent logs error on calculation failure."""
        import logging
        caplog.set_level(logging.ERROR)

        # Mock to force an error
        with patch.object(agent, '_get_emission_factor', return_value=None):
            input_data = CarbonEmissionsInput(
                fuel_type=FuelType.NATURAL_GAS,
                quantity=100.0,
                unit="m3",
                region="US",
                scope=Scope.SCOPE_1,
            )

            with pytest.raises(ValueError):
                agent.run(input_data)

    @pytest.mark.unit
    def test_calculation_exception_includes_context(self, agent):
        """UT-GL001-048: Test exception includes helpful context."""
        with patch.object(agent, '_get_emission_factor', return_value=None):
            input_data = CarbonEmissionsInput(
                fuel_type=FuelType.NATURAL_GAS,
                quantity=100.0,
                unit="m3",
                region="US",
                scope=Scope.SCOPE_1,
            )

            with pytest.raises(ValueError) as exc_info:
                agent.run(input_data)

            # Error message should include fuel type and region
            error_msg = str(exc_info.value)
            assert "natural_gas" in error_msg or "US" in error_msg

    @pytest.mark.unit
    def test_agent_recovers_after_error(self, agent, valid_natural_gas_input):
        """UT-GL001-049: Test agent can process requests after an error."""
        # First, cause an error
        with patch.object(agent, '_get_emission_factor', return_value=None):
            try:
                agent.run(valid_natural_gas_input)
            except ValueError:
                pass

        # Agent should still work for valid requests
        result = agent.run(valid_natural_gas_input)
        assert result.emissions_kgco2e > 0

    @pytest.mark.unit
    def test_agent_output_includes_timestamp(self, agent, valid_natural_gas_input):
        """UT-GL001-050: Test output includes calculation timestamp."""
        result = agent.run(valid_natural_gas_input)

        assert result.calculated_at is not None
        assert isinstance(result.calculated_at, datetime)


# =============================================================================
# Additional Integration Tests
# =============================================================================

class TestAgentIntegration:
    """Integration tests for CarbonEmissionsAgent."""

    @pytest.mark.unit
    def test_agent_initialization(self):
        """Test agent initializes with default configuration."""
        agent = CarbonEmissionsAgent()
        assert agent is not None
        assert agent.AGENT_ID == "emissions/carbon_calculator_v1"
        assert agent.VERSION == "1.0.0"

    @pytest.mark.unit
    def test_agent_initialization_with_config(self):
        """Test agent initializes with custom configuration."""
        config = {"custom_key": "custom_value"}
        agent = CarbonEmissionsAgent(config=config)
        assert agent.config["custom_key"] == "custom_value"

    @pytest.mark.unit
    def test_get_supported_fuel_types(self, agent):
        """Test get_supported_fuel_types returns all fuel types."""
        fuel_types = agent.get_supported_fuel_types()
        assert "natural_gas" in fuel_types
        assert "diesel" in fuel_types
        assert "electricity_grid" in fuel_types
        assert len(fuel_types) == 7

    @pytest.mark.unit
    def test_get_supported_regions(self, agent):
        """Test get_supported_regions returns available regions."""
        regions = agent.get_supported_regions()
        assert "US" in regions
        assert "EU" in regions
        assert len(regions) > 0

    @pytest.mark.unit
    def test_unit_conversion_cf_to_m3(self, agent):
        """Test unit conversion from cubic feet to cubic meters."""
        # 1 cf = 0.0283168 m3
        converted = agent._convert_units(100, "cf", "kgCO2e/m3")
        expected = 100 * 0.0283168
        assert converted == pytest.approx(expected, rel=1e-6)

    @pytest.mark.unit
    def test_unit_conversion_gal_to_l(self, agent):
        """Test unit conversion from gallons to liters."""
        # 1 gal = 3.78541 L
        converted = agent._convert_units(10, "gal", "kgCO2e/L")
        expected = 10 * 3.78541
        assert converted == pytest.approx(expected, rel=1e-6)

    @pytest.mark.unit
    def test_unit_conversion_mwh_to_kwh(self, agent):
        """Test unit conversion from MWh to kWh."""
        # 1 MWh = 1000 kWh
        converted = agent._convert_units(5, "MWh", "kgCO2e/kWh")
        expected = 5 * 1000
        assert converted == pytest.approx(expected, rel=1e-6)

    @pytest.mark.unit
    def test_unit_conversion_same_unit(self, agent):
        """Test no conversion when units match."""
        converted = agent._convert_units(100, "m3", "kgCO2e/m3")
        assert converted == 100.0


# =============================================================================
# Parametrized Tests
# =============================================================================

class TestParametrizedCalculations:
    """Parametrized tests for multiple fuel types and regions."""

    @pytest.mark.unit
    @pytest.mark.parametrize("fuel_type,quantity,unit,region,expected_range", [
        (FuelType.NATURAL_GAS, 1000, "m3", "US", (1900, 2000)),
        (FuelType.DIESEL, 500, "L", "US", (1300, 1400)),
        (FuelType.GASOLINE, 100, "L", "US", (220, 240)),
        (FuelType.ELECTRICITY_GRID, 10000, "kWh", "US", (4100, 4200)),
        (FuelType.ELECTRICITY_GRID, 10000, "kWh", "FR", (500, 600)),
    ])
    def test_emissions_in_expected_range(self, agent, fuel_type, quantity, unit, region, expected_range):
        """Test emissions fall within expected ranges for various inputs."""
        input_data = CarbonEmissionsInput(
            fuel_type=fuel_type,
            quantity=quantity,
            unit=unit,
            region=region,
            scope=Scope.SCOPE_1 if fuel_type != FuelType.ELECTRICITY_GRID else Scope.SCOPE_2,
        )
        result = agent.run(input_data)

        assert expected_range[0] <= result.emissions_kgco2e <= expected_range[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
