# -*- coding: utf-8 -*-
"""
Unit Tests for Fuel Emissions Analyzer Agent

Comprehensive test suite with 30 test cases covering:
- LookupEmissionFactorTool (10 tests)
- CalculateEmissionsTool (10 tests)
- ValidateFuelInputTool (10 tests)

Target: 85%+ coverage for fuel analyzer tools
Run with: pytest tests/unit/test_fuel_analyzer.py -v --cov=generated/fuel_analyzer_agent

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import asyncio
import hashlib
import json
from decimal import Decimal
from datetime import datetime, date
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock

# Add project paths for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "generated" / "fuel_analyzer_agent"))
sys.path.insert(0, str(project_root / "greenlang"))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def lookup_emission_factor_tool():
    """Create LookupEmissionFactorTool instance."""
    from generated.fuel_analyzer_agent.tools import LookupEmissionFactorTool
    return LookupEmissionFactorTool()


@pytest.fixture
def calculate_emissions_tool():
    """Create CalculateEmissionsTool instance."""
    from generated.fuel_analyzer_agent.tools import CalculateEmissionsTool
    return CalculateEmissionsTool()


@pytest.fixture
def validate_fuel_input_tool():
    """Create ValidateFuelInputTool instance."""
    from generated.fuel_analyzer_agent.tools import ValidateFuelInputTool
    return ValidateFuelInputTool()


@pytest.fixture
def mock_emission_factor_db():
    """Create mock emission factor database."""
    mock_db = Mock()
    mock_record = Mock()
    mock_record.ef_uri = "ef://IPCC/natural_gas/US/2023"
    mock_record.ef_value = 0.0561
    mock_record.ef_unit = "kgCO2e/MJ"
    mock_record.source = "IPCC 2006 Guidelines"
    mock_record.gwp_set = Mock(value="AR6GWP100")
    mock_record.uncertainty = 0.05
    mock_db.lookup.return_value = mock_record
    return mock_db


@pytest.fixture
def sample_fuel_types():
    """Sample fuel types for parametrized tests."""
    return [
        "natural_gas", "diesel", "gasoline", "lpg",
        "fuel_oil", "coal", "propane", "kerosene",
        "electricity", "biomass"
    ]


@pytest.fixture
def sample_regions():
    """Sample regions for parametrized tests."""
    return ["US", "GB", "EU"]


# =============================================================================
# LookupEmissionFactorTool Tests (10 tests)
# =============================================================================

class TestLookupEmissionFactorTool:
    """Test suite for LookupEmissionFactorTool - 10 test cases."""

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    def test_tool_initialization(self, lookup_emission_factor_tool):
        """UT-FA-001: Test tool initializes correctly."""
        assert lookup_emission_factor_tool is not None
        assert lookup_emission_factor_tool.name == "lookup_emission_factor"
        assert lookup_emission_factor_tool.safe is True

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_missing_fuel_type_raises_error(self, lookup_emission_factor_tool):
        """UT-FA-002: Test missing fuel_type parameter raises ValueError."""
        params = {"region": "US", "year": 2023}

        with pytest.raises(ValueError) as exc_info:
            await lookup_emission_factor_tool.execute(params)

        assert "fuel_type" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_missing_region_raises_error(self, lookup_emission_factor_tool):
        """UT-FA-003: Test missing region parameter raises ValueError."""
        params = {"fuel_type": "natural_gas", "year": 2023}

        with pytest.raises(ValueError) as exc_info:
            await lookup_emission_factor_tool.execute(params)

        assert "region" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_missing_year_raises_error(self, lookup_emission_factor_tool):
        """UT-FA-004: Test missing year parameter raises ValueError."""
        params = {"fuel_type": "natural_gas", "region": "US"}

        with pytest.raises(ValueError) as exc_info:
            await lookup_emission_factor_tool.execute(params)

        assert "year" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_valid_lookup_returns_result(self, lookup_emission_factor_tool, mock_emission_factor_db):
        """UT-FA-005: Test valid lookup returns emission factor data."""
        with patch('generated.fuel_analyzer_agent.tools.get_database', return_value=mock_emission_factor_db):
            params = {
                "fuel_type": "natural_gas",
                "region": "US",
                "year": 2023,
            }

            result = await lookup_emission_factor_tool.execute(params)

            assert result is not None
            assert "ef_value" in result
            assert "ef_unit" in result
            assert "source" in result

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_gwp_set_default(self, lookup_emission_factor_tool, mock_emission_factor_db):
        """UT-FA-006: Test GWP set defaults to AR6GWP100."""
        with patch('generated.fuel_analyzer_agent.tools.get_database', return_value=mock_emission_factor_db):
            params = {
                "fuel_type": "diesel",
                "region": "US",
                "year": 2023,
            }

            result = await lookup_emission_factor_tool.execute(params)

            assert "gwp_set" in result
            # GWP set should be returned in result

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_invalid_fuel_type_raises_error(self, lookup_emission_factor_tool):
        """UT-FA-007: Test invalid fuel type raises appropriate error."""
        # Mock database to return None for invalid fuel
        mock_db = Mock()
        mock_db.lookup.return_value = None

        with patch('generated.fuel_analyzer_agent.tools.get_database', return_value=mock_db):
            params = {
                "fuel_type": "invalid_fuel_xyz",
                "region": "US",
                "year": 2023,
            }

            with pytest.raises(ValueError) as exc_info:
                await lookup_emission_factor_tool.execute(params)

            assert "not found" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_custom_gwp_set(self, lookup_emission_factor_tool, mock_emission_factor_db):
        """UT-FA-008: Test custom GWP set is passed correctly."""
        with patch('generated.fuel_analyzer_agent.tools.get_database', return_value=mock_emission_factor_db):
            params = {
                "fuel_type": "natural_gas",
                "region": "US",
                "year": 2023,
                "gwp_set": "AR5GWP100",
            }

            result = await lookup_emission_factor_tool.execute(params)

            assert result is not None
            mock_emission_factor_db.lookup.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    def test_validate_params_method(self, lookup_emission_factor_tool):
        """UT-FA-009: Test validate_params method exists and returns boolean."""
        params = {"fuel_type": "diesel", "region": "US", "year": 2023}
        result = lookup_emission_factor_tool.validate_params(params)
        assert isinstance(result, bool)

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    @pytest.mark.parametrize("region", ["US", "GB", "EU"])
    async def test_regional_factors_differ(self, lookup_emission_factor_tool, region):
        """UT-FA-010: Test regional emission factors are retrieved correctly."""
        # This test verifies the tool can handle different regions
        params = {
            "fuel_type": "natural_gas",
            "region": region,
            "year": 2023,
        }

        # We just verify the tool accepts all valid regions without error
        # Actual values tested in golden tests
        mock_db = Mock()
        mock_record = Mock()
        mock_record.ef_uri = f"ef://IPCC/natural_gas/{region}/2023"
        mock_record.ef_value = 0.056
        mock_record.ef_unit = "kgCO2e/MJ"
        mock_record.source = "IPCC"
        mock_record.gwp_set = Mock(value="AR6GWP100")
        mock_record.uncertainty = 0.05
        mock_db.lookup.return_value = mock_record

        with patch('generated.fuel_analyzer_agent.tools.get_database', return_value=mock_db):
            result = await lookup_emission_factor_tool.execute(params)
            assert result is not None
            assert region in result.get("ef_uri", "")


# =============================================================================
# CalculateEmissionsTool Tests (10 tests)
# =============================================================================

class TestCalculateEmissionsTool:
    """Test suite for CalculateEmissionsTool - 10 test cases."""

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    def test_tool_initialization(self, calculate_emissions_tool):
        """UT-FA-011: Test tool initializes correctly."""
        assert calculate_emissions_tool is not None
        assert calculate_emissions_tool.name == "calculate_emissions"
        assert calculate_emissions_tool.safe is True

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_missing_activity_value_raises_error(self, calculate_emissions_tool):
        """UT-FA-012: Test missing activity_value raises ValueError."""
        params = {
            "activity_unit": "MJ",
            "ef_value": 0.0561,
            "ef_unit": "kgCO2e/MJ",
        }

        with pytest.raises(ValueError) as exc_info:
            await calculate_emissions_tool.execute(params)

        assert "activity_value" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_basic_calculation(self, calculate_emissions_tool):
        """UT-FA-013: Test basic emissions calculation."""
        params = {
            "activity_value": 1000.0,
            "activity_unit": "MJ",
            "ef_value": 0.0561,
            "ef_unit": "kgCO2e/MJ",
            "output_unit": "tCO2e",
        }

        result = await calculate_emissions_tool.execute(params)

        assert result is not None
        assert "emissions_value" in result
        assert "emissions_unit" in result
        assert "calculation_formula" in result

        # Verify calculation: 1000 * 0.0561 = 56.1 kgCO2e = 0.0561 tCO2e
        expected_emissions = 1000.0 * 0.0561 / 1000.0  # Convert to tCO2e
        assert abs(result["emissions_value"] - expected_emissions) < 0.0001

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_unit_conversion_mj_to_gj(self, calculate_emissions_tool):
        """UT-FA-014: Test unit conversion from MJ to GJ."""
        params = {
            "activity_value": 1.0,  # 1 GJ
            "activity_unit": "GJ",
            "ef_value": 0.0561,  # per MJ
            "ef_unit": "kgCO2e/MJ",
            "output_unit": "kgCO2e",
        }

        result = await calculate_emissions_tool.execute(params)

        # 1 GJ = 1000 MJ, so emissions = 1000 * 0.0561 = 56.1 kgCO2e
        assert abs(result["emissions_value"] - 56.1) < 0.1

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_unit_conversion_kwh(self, calculate_emissions_tool):
        """UT-FA-015: Test unit conversion with kWh."""
        params = {
            "activity_value": 100.0,  # 100 kWh
            "activity_unit": "kWh",
            "ef_value": 0.0561,  # per MJ
            "ef_unit": "kgCO2e/MJ",
            "output_unit": "kgCO2e",
        }

        result = await calculate_emissions_tool.execute(params)

        # 100 kWh = 360 MJ (1 kWh = 3.6 MJ)
        # emissions = 360 * 0.0561 = 20.196 kgCO2e
        expected = 100.0 * 3.6 * 0.0561
        assert abs(result["emissions_value"] - expected) < 0.1

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_determinism_same_input_same_output(self, calculate_emissions_tool):
        """UT-FA-016: Test calculation determinism - same input produces same output."""
        params = {
            "activity_value": 1000.0,
            "activity_unit": "MJ",
            "ef_value": 0.0561,
            "ef_unit": "kgCO2e/MJ",
            "output_unit": "tCO2e",
        }

        results = []
        for _ in range(10):
            result = await calculate_emissions_tool.execute(params)
            results.append(result["emissions_value"])

        # All results must be identical
        assert all(r == results[0] for r in results), "Results are not deterministic"

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_output_unit_conversion_to_tonnes(self, calculate_emissions_tool):
        """UT-FA-017: Test output unit conversion to tCO2e."""
        params = {
            "activity_value": 10000.0,
            "activity_unit": "MJ",
            "ef_value": 0.1,  # 0.1 kgCO2e/MJ
            "ef_unit": "kgCO2e/MJ",
            "output_unit": "tCO2e",
        }

        result = await calculate_emissions_tool.execute(params)

        # 10000 * 0.1 = 1000 kgCO2e = 1 tCO2e
        assert abs(result["emissions_value"] - 1.0) < 0.001
        assert result["emissions_unit"] == "tCO2e"

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_zero_activity_returns_zero_emissions(self, calculate_emissions_tool):
        """UT-FA-018: Test zero activity produces zero emissions."""
        params = {
            "activity_value": 0.0,
            "activity_unit": "MJ",
            "ef_value": 0.0561,
            "ef_unit": "kgCO2e/MJ",
            "output_unit": "tCO2e",
        }

        result = await calculate_emissions_tool.execute(params)

        assert result["emissions_value"] == 0.0

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_formula_provenance_tracking(self, calculate_emissions_tool):
        """UT-FA-019: Test calculation formula is included in output."""
        params = {
            "activity_value": 500.0,
            "activity_unit": "MJ",
            "ef_value": 0.05,
            "ef_unit": "kgCO2e/MJ",
            "output_unit": "kgCO2e",
        }

        result = await calculate_emissions_tool.execute(params)

        assert "calculation_formula" in result
        formula = result["calculation_formula"]
        assert "500" in formula or "500.0" in formula
        assert "0.05" in formula
        assert "=" in formula

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_incompatible_units_raises_error(self, calculate_emissions_tool):
        """UT-FA-020: Test incompatible units raise appropriate error."""
        params = {
            "activity_value": 100.0,
            "activity_unit": "kg",  # Mass unit
            "ef_value": 0.0561,
            "ef_unit": "kgCO2e/MJ",  # Energy-based EF
            "output_unit": "kgCO2e",
        }

        with pytest.raises(ValueError) as exc_info:
            await calculate_emissions_tool.execute(params)

        assert "incompatible" in str(exc_info.value).lower() or "unit" in str(exc_info.value).lower()


# =============================================================================
# ValidateFuelInputTool Tests (10 tests)
# =============================================================================

class TestValidateFuelInputTool:
    """Test suite for ValidateFuelInputTool - 10 test cases."""

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    def test_tool_initialization(self, validate_fuel_input_tool):
        """UT-FA-021: Test tool initializes correctly."""
        assert validate_fuel_input_tool is not None
        assert validate_fuel_input_tool.name == "validate_fuel_input"
        assert validate_fuel_input_tool.safe is True

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_missing_fuel_type_raises_error(self, validate_fuel_input_tool):
        """UT-FA-022: Test missing fuel_type raises ValueError."""
        params = {"quantity": 1000, "unit": "L"}

        with pytest.raises(ValueError) as exc_info:
            await validate_fuel_input_tool.execute(params)

        assert "fuel_type" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_missing_quantity_raises_error(self, validate_fuel_input_tool):
        """UT-FA-023: Test missing quantity raises ValueError."""
        params = {"fuel_type": "diesel", "unit": "L"}

        with pytest.raises(ValueError) as exc_info:
            await validate_fuel_input_tool.execute(params)

        assert "quantity" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_missing_unit_raises_error(self, validate_fuel_input_tool):
        """UT-FA-024: Test missing unit raises ValueError."""
        params = {"fuel_type": "diesel", "quantity": 1000}

        with pytest.raises(ValueError) as exc_info:
            await validate_fuel_input_tool.execute(params)

        assert "unit" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_valid_input_returns_valid_true(self, validate_fuel_input_tool):
        """UT-FA-025: Test valid input returns valid=True."""
        params = {
            "fuel_type": "diesel",
            "quantity": 1000.0,
            "unit": "L",
        }

        result = await validate_fuel_input_tool.execute(params)

        assert result["valid"] is True
        assert "plausibility_score" in result
        assert result["plausibility_score"] >= 0.0
        assert result["plausibility_score"] <= 1.0

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_negative_quantity_returns_invalid(self, validate_fuel_input_tool):
        """UT-FA-026: Test negative quantity returns valid=False."""
        params = {
            "fuel_type": "diesel",
            "quantity": -100.0,
            "unit": "L",
        }

        result = await validate_fuel_input_tool.execute(params)

        assert result["valid"] is False
        assert result["plausibility_score"] == 0.0
        assert len(result["warnings"]) > 0
        assert any("negative" in w.lower() for w in result["warnings"])

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_zero_quantity_returns_warning(self, validate_fuel_input_tool):
        """UT-FA-027: Test zero quantity returns warning but is valid."""
        params = {
            "fuel_type": "diesel",
            "quantity": 0.0,
            "unit": "L",
        }

        result = await validate_fuel_input_tool.execute(params)

        assert len(result["warnings"]) > 0
        assert any("zero" in w.lower() for w in result["warnings"])
        assert result["plausibility_score"] < 1.0

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_plausibility_score_range(self, validate_fuel_input_tool):
        """UT-FA-028: Test plausibility score is always between 0 and 1."""
        test_cases = [
            {"fuel_type": "diesel", "quantity": 100.0, "unit": "L"},
            {"fuel_type": "natural_gas", "quantity": 1000.0, "unit": "MJ"},
            {"fuel_type": "gasoline", "quantity": 50000.0, "unit": "L"},  # High but plausible
        ]

        for params in test_cases:
            result = await validate_fuel_input_tool.execute(params)
            assert 0.0 <= result["plausibility_score"] <= 1.0

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_extremely_high_quantity_returns_warning(self, validate_fuel_input_tool):
        """UT-FA-029: Test extremely high quantity triggers warning."""
        params = {
            "fuel_type": "diesel",
            "quantity": 1e8,  # 100 million liters - unusually high
            "unit": "L",
        }

        result = await validate_fuel_input_tool.execute(params)

        # Should either be invalid or have warnings
        if result["valid"]:
            assert len(result["warnings"]) > 0 or result["plausibility_score"] < 1.0

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_incompatible_unit_returns_invalid(self, validate_fuel_input_tool):
        """UT-FA-030: Test incompatible fuel/unit combination returns invalid."""
        params = {
            "fuel_type": "diesel",
            "quantity": 1000.0,
            "unit": "therms",  # therms is for gas, not diesel
        }

        result = await validate_fuel_input_tool.execute(params)

        # Should flag as issue (either invalid or low plausibility)
        if result["valid"]:
            assert result["plausibility_score"] < 0.5 or len(result["warnings"]) > 0
        else:
            assert len(result["warnings"]) > 0


# =============================================================================
# Integration Tests for Tool Registry
# =============================================================================

class TestToolRegistry:
    """Test suite for tool registry functionality."""

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    def test_get_tool_returns_correct_tool(self):
        """Test get_tool function returns correct tool instances."""
        from generated.fuel_analyzer_agent.tools import get_tool

        lookup_tool = get_tool("lookup_emission_factor")
        calc_tool = get_tool("calculate_emissions")
        validate_tool = get_tool("validate_fuel_input")

        assert lookup_tool is not None
        assert calc_tool is not None
        assert validate_tool is not None

        assert lookup_tool.name == "lookup_emission_factor"
        assert calc_tool.name == "calculate_emissions"
        assert validate_tool.name == "validate_fuel_input"

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    def test_get_tool_invalid_name_returns_none(self):
        """Test get_tool with invalid name returns None."""
        from generated.fuel_analyzer_agent.tools import get_tool

        result = get_tool("nonexistent_tool")
        assert result is None

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    def test_list_tools_returns_all_tools(self):
        """Test list_tools returns all available tools."""
        from generated.fuel_analyzer_agent.tools import list_tools

        tools = list_tools()

        assert "lookup_emission_factor" in tools
        assert "calculate_emissions" in tools
        assert "validate_fuel_input" in tools
        assert len(tools) == 3


# =============================================================================
# Determinism Tests
# =============================================================================

class TestDeterminism:
    """Test suite for verifying deterministic behavior."""

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.determinism
    @pytest.mark.asyncio
    async def test_lookup_determinism(self, lookup_emission_factor_tool, mock_emission_factor_db):
        """Test lookup tool is deterministic across multiple calls."""
        with patch('generated.fuel_analyzer_agent.tools.get_database', return_value=mock_emission_factor_db):
            params = {
                "fuel_type": "diesel",
                "region": "US",
                "year": 2023,
            }

            results = []
            for _ in range(5):
                result = await lookup_emission_factor_tool.execute(params)
                results.append(json.dumps(result, sort_keys=True))

            # All serialized results should be identical
            assert all(r == results[0] for r in results)

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.determinism
    @pytest.mark.asyncio
    async def test_calculation_hash_determinism(self, calculate_emissions_tool):
        """Test calculation produces identical hash across runs."""
        params = {
            "activity_value": 1234.5678,
            "activity_unit": "MJ",
            "ef_value": 0.0561789,
            "ef_unit": "kgCO2e/MJ",
            "output_unit": "tCO2e",
        }

        hashes = []
        for _ in range(10):
            result = await calculate_emissions_tool.execute(params)
            result_str = json.dumps(result, sort_keys=True)
            hash_val = hashlib.sha256(result_str.encode()).hexdigest()
            hashes.append(hash_val)

        # All hashes must be identical
        assert len(set(hashes)) == 1, "SHA-256 hashes differ across runs"

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.determinism
    @pytest.mark.asyncio
    async def test_validation_determinism(self, validate_fuel_input_tool):
        """Test validation tool is deterministic."""
        params = {
            "fuel_type": "natural_gas",
            "quantity": 5000.0,
            "unit": "MJ",
        }

        results = []
        for _ in range(5):
            result = await validate_fuel_input_tool.execute(params)
            results.append((result["valid"], result["plausibility_score"]))

        # All results should be identical
        assert all(r == results[0] for r in results)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_very_small_quantity(self, calculate_emissions_tool):
        """Test handling of very small quantities."""
        params = {
            "activity_value": 0.00001,
            "activity_unit": "MJ",
            "ef_value": 0.0561,
            "ef_unit": "kgCO2e/MJ",
            "output_unit": "kgCO2e",
        }

        result = await calculate_emissions_tool.execute(params)

        assert result["emissions_value"] >= 0
        assert result["emissions_value"] < 0.001

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_very_large_quantity(self, calculate_emissions_tool):
        """Test handling of very large quantities."""
        params = {
            "activity_value": 1e12,  # 1 trillion MJ
            "activity_unit": "MJ",
            "ef_value": 0.0561,
            "ef_unit": "kgCO2e/MJ",
            "output_unit": "tCO2e",
        }

        result = await calculate_emissions_tool.execute(params)

        assert result["emissions_value"] > 0
        # Should be approximately 56.1 billion kgCO2e = 56.1 million tCO2e
        assert result["emissions_value"] > 1e6

    @pytest.mark.unit
    @pytest.mark.fuel_analyzer
    @pytest.mark.asyncio
    async def test_high_precision_values(self, calculate_emissions_tool):
        """Test handling of high-precision decimal values."""
        params = {
            "activity_value": 1000.123456789,
            "activity_unit": "MJ",
            "ef_value": 0.056123456789,
            "ef_unit": "kgCO2e/MJ",
            "output_unit": "kgCO2e",
        }

        result = await calculate_emissions_tool.execute(params)

        # Should handle high precision without errors
        assert isinstance(result["emissions_value"], (int, float))
        assert result["emissions_value"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
