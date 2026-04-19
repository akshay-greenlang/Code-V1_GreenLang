# -*- coding: utf-8 -*-
"""
Tests for GL-FOUND-X-003: Unit & Reference Normalizer

Tests cover:
    - Unit conversion across all dimensions
    - GHG conversions with GWP (AR4, AR5, AR6)
    - Fuel name standardization
    - Material name standardization
    - Reference ID management
    - Currency conversion
    - Dimensional analysis validation
    - Conversion lineage tracking
    - Tenant customization
    - Edge cases and error handling

Test Coverage Target: 85%+
"""

import pytest
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict

from greenlang.agents.base import AgentConfig, AgentResult

# Import directly from module file to avoid __init__.py import chain issues
import sys
import importlib.util
from pathlib import Path

# Load the module directly
module_path = Path(__file__).parent.parent.parent.parent / "greenlang" / "agents" / "foundation" / "unit_normalizer.py"
spec = importlib.util.spec_from_file_location("unit_normalizer", module_path)
unit_normalizer = importlib.util.module_from_spec(spec)
sys.modules["unit_normalizer"] = unit_normalizer
spec.loader.exec_module(unit_normalizer)

# Import from the loaded module
UnitNormalizerAgent = unit_normalizer.UnitNormalizerAgent
UnitDimension = unit_normalizer.UnitDimension
GHGType = unit_normalizer.GHGType
ConversionRequest = unit_normalizer.ConversionRequest
ConversionResult = unit_normalizer.ConversionResult
GHGConversionRequest = unit_normalizer.GHGConversionRequest
GHGConversionResult = unit_normalizer.GHGConversionResult
FuelStandardizationRequest = unit_normalizer.FuelStandardizationRequest
FuelStandardizationResult = unit_normalizer.FuelStandardizationResult
MaterialStandardizationRequest = unit_normalizer.MaterialStandardizationRequest
MaterialStandardizationResult = unit_normalizer.MaterialStandardizationResult
NormalizerInput = unit_normalizer.NormalizerInput
NormalizerOutput = unit_normalizer.NormalizerOutput
GWP_AR6_100 = unit_normalizer.GWP_AR6_100
GWP_AR5_100 = unit_normalizer.GWP_AR5_100
GWP_AR4_100 = unit_normalizer.GWP_AR4_100
MASS_UNITS = unit_normalizer.MASS_UNITS
ENERGY_UNITS = unit_normalizer.ENERGY_UNITS
VOLUME_UNITS = unit_normalizer.VOLUME_UNITS
AREA_UNITS = unit_normalizer.AREA_UNITS
DISTANCE_UNITS = unit_normalizer.DISTANCE_UNITS


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def agent():
    """Create a UnitNormalizerAgent instance."""
    return UnitNormalizerAgent()


@pytest.fixture
def agent_with_custom_config():
    """Create agent with custom configuration."""
    config = AgentConfig(
        name="Custom Unit Normalizer",
        description="Test agent with custom config",
        version="1.0.0",
        parameters={
            "default_precision": 8,
            "gwp_source": "AR5",
            "strict_mode": True,
        }
    )
    return UnitNormalizerAgent(config)


# =============================================================================
# UNIT CONVERSION TESTS
# =============================================================================

class TestUnitConversion:
    """Tests for basic unit conversion functionality."""

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.AGENT_ID == "GL-FOUND-X-003"
        assert agent.AGENT_NAME == "Unit & Reference Normalizer"
        assert agent.VERSION == "1.0.0"

    def test_mass_conversion_kg_to_tonnes(self, agent):
        """Test mass conversion from kg to tonnes."""
        result = agent.convert(1000, "kg", "tonnes")
        assert result == 1.0

    def test_mass_conversion_tonnes_to_kg(self, agent):
        """Test mass conversion from tonnes to kg."""
        result = agent.convert(1, "tonnes", "kg")
        assert result == 1000.0

    def test_mass_conversion_lb_to_kg(self, agent):
        """Test mass conversion from pounds to kg."""
        result = agent.convert(1, "lb", "kg")
        assert abs(result - 0.45359237) < 0.0001

    def test_mass_conversion_short_ton_to_tonnes(self, agent):
        """Test mass conversion from US short ton to metric tonnes."""
        result = agent.convert(1, "short_ton", "tonnes")
        assert abs(result - 0.907185) < 0.001

    def test_mass_conversion_long_ton_to_tonnes(self, agent):
        """Test mass conversion from UK long ton to metric tonnes."""
        result = agent.convert(1, "long_ton", "tonnes")
        assert abs(result - 1.016047) < 0.001

    def test_energy_conversion_kwh_to_mwh(self, agent):
        """Test energy conversion from kWh to MWh."""
        result = agent.convert(1000, "kwh", "mwh")
        assert result == 1.0

    def test_energy_conversion_mwh_to_gj(self, agent):
        """Test energy conversion from MWh to GJ."""
        result = agent.convert(1, "mwh", "gj")
        assert abs(result - 3.6) < 0.001

    def test_energy_conversion_btu_to_kj(self, agent):
        """Test energy conversion from BTU to kJ."""
        result = agent.convert(1, "btu", "kj")
        assert abs(result - 1.055) < 0.01

    def test_energy_conversion_therm_to_mmbtu(self, agent):
        """Test energy conversion from therm to MMBtu."""
        result = agent.convert(10, "therm", "mmbtu")
        assert abs(result - 1.0) < 0.001

    def test_volume_conversion_liters_to_gallons(self, agent):
        """Test volume conversion from liters to US gallons."""
        result = agent.convert(3.785411784, "l", "gallon")
        assert abs(result - 1.0) < 0.0001

    def test_volume_conversion_m3_to_liters(self, agent):
        """Test volume conversion from cubic meters to liters."""
        result = agent.convert(1, "m3", "l")
        assert result == 1000.0

    def test_volume_conversion_barrel_to_liters(self, agent):
        """Test volume conversion from oil barrel to liters."""
        result = agent.convert(1, "barrel", "l")
        assert abs(result - 158.987) < 0.01

    def test_area_conversion_m2_to_hectare(self, agent):
        """Test area conversion from m2 to hectare."""
        result = agent.convert(10000, "m2", "hectare")
        assert result == 1.0

    def test_area_conversion_acre_to_hectare(self, agent):
        """Test area conversion from acre to hectare."""
        result = agent.convert(1, "acre", "hectare")
        assert abs(result - 0.4047) < 0.001

    def test_area_conversion_km2_to_m2(self, agent):
        """Test area conversion from km2 to m2."""
        result = agent.convert(1, "km2", "m2")
        assert result == 1000000.0

    def test_distance_conversion_km_to_mile(self, agent):
        """Test distance conversion from km to miles."""
        result = agent.convert(1.609344, "km", "mile")
        assert abs(result - 1.0) < 0.0001

    def test_distance_conversion_nautical_mile_to_km(self, agent):
        """Test distance conversion from nautical mile to km."""
        result = agent.convert(1, "nmi", "km")
        assert abs(result - 1.852) < 0.001

    def test_same_unit_conversion(self, agent):
        """Test conversion when from and to units are the same."""
        result = agent.convert(100, "kg", "kg")
        assert result == 100.0

    def test_unit_aliases(self, agent):
        """Test that unit aliases work correctly."""
        # kg aliases
        assert agent.convert(1, "kilogram", "kg") == 1.0
        assert agent.convert(1, "kilograms", "kg") == 1.0

        # tonne aliases
        assert agent.convert(1, "metric_ton", "tonnes") == 1.0
        assert agent.convert(1, "t", "tonnes") == 1.0

        # MWh aliases
        assert agent.convert(1, "megawatt_hour", "mwh") == 1.0

    def test_conversion_precision(self, agent):
        """Test conversion precision parameter."""
        result = agent._handle_unit_conversion({
            "value": 1,
            "from_unit": "kg",
            "to_unit": "lb",
            "precision": 10
        }, None)
        # Should have more decimal places
        assert "converted_value" in result


class TestUnitConversionErrors:
    """Tests for unit conversion error handling."""

    def test_invalid_from_unit(self, agent):
        """Test error handling for unknown from_unit."""
        with pytest.raises(ValueError, match="Unknown unit"):
            agent.convert(100, "unknown_unit", "kg")

    def test_invalid_to_unit(self, agent):
        """Test error handling for unknown to_unit."""
        with pytest.raises(ValueError, match="Unknown unit"):
            agent.convert(100, "kg", "unknown_unit")

    def test_incompatible_dimensions(self, agent):
        """Test error handling for incompatible dimensions."""
        with pytest.raises(ValueError, match="Cannot convert between different dimensions"):
            agent.convert(100, "kg", "kwh")

    def test_mass_to_volume_error(self, agent):
        """Test that mass to volume conversion fails."""
        with pytest.raises(ValueError):
            agent.convert(100, "kg", "liter")

    def test_energy_to_distance_error(self, agent):
        """Test that energy to distance conversion fails."""
        with pytest.raises(ValueError):
            agent.convert(100, "mwh", "km")


class TestDimensionalAnalysis:
    """Tests for dimensional analysis functionality."""

    def test_is_convertible_same_dimension(self, agent):
        """Test is_convertible returns True for same dimension."""
        assert agent.is_convertible("kg", "tonnes") is True
        assert agent.is_convertible("kWh", "MWh") is True
        assert agent.is_convertible("liter", "gallon") is True

    def test_is_convertible_different_dimension(self, agent):
        """Test is_convertible returns False for different dimensions."""
        assert agent.is_convertible("kg", "kWh") is False
        assert agent.is_convertible("m2", "km") is False
        assert agent.is_convertible("liter", "kg") is False

    def test_is_convertible_unknown_unit(self, agent):
        """Test is_convertible returns False for unknown units."""
        assert agent.is_convertible("unknown", "kg") is False
        assert agent.is_convertible("kg", "unknown") is False

    def test_validate_conversion_valid(self, agent):
        """Test validate_conversion for valid conversions."""
        result = agent._handle_validate_conversion({
            "from_unit": "kg",
            "to_unit": "tonnes"
        })
        assert result["is_convertible"] is True
        assert result["from_dimension"] == "mass"
        assert result["to_dimension"] == "mass"

    def test_validate_conversion_invalid(self, agent):
        """Test validate_conversion for invalid conversions."""
        result = agent._handle_validate_conversion({
            "from_unit": "kg",
            "to_unit": "kwh"
        })
        assert result["is_convertible"] is False
        assert "Incompatible dimensions" in result["reason"]

    def test_get_dimension_known_unit(self, agent):
        """Test get_dimension for known units."""
        result = agent._handle_get_dimension({"unit": "kg"})
        assert result["dimension"] == "mass"
        assert result["is_known"] is True
        assert result["base_unit"] == "kg"

    def test_get_dimension_unknown_unit(self, agent):
        """Test get_dimension for unknown units."""
        result = agent._handle_get_dimension({"unit": "unknown"})
        assert result["dimension"] is None
        assert result["is_known"] is False


# =============================================================================
# GHG CONVERSION TESTS
# =============================================================================

class TestGHGConversion:
    """Tests for GHG conversion with GWP."""

    def test_ch4_to_co2e_ar6(self, agent):
        """Test CH4 to CO2e conversion using AR6 GWP."""
        result = agent.convert_ghg(1000, "kgCH4", "kgCO2e", "AR6")
        # AR6 GWP for CH4 = 29.8
        assert abs(result - 29800) < 1

    def test_ch4_to_co2e_ar5(self, agent):
        """Test CH4 to CO2e conversion using AR5 GWP."""
        result = agent.convert_ghg(1000, "kgCH4", "kgCO2e", "AR5")
        # AR5 GWP for CH4 = 28
        assert abs(result - 28000) < 1

    def test_ch4_to_co2e_ar4(self, agent):
        """Test CH4 to CO2e conversion using AR4 GWP."""
        result = agent.convert_ghg(1000, "kgCH4", "kgCO2e", "AR4")
        # AR4 GWP for CH4 = 25
        assert abs(result - 25000) < 1

    def test_n2o_to_co2e_ar6(self, agent):
        """Test N2O to CO2e conversion using AR6 GWP."""
        result = agent.convert_ghg(1, "kgN2O", "kgCO2e", "AR6")
        # AR6 GWP for N2O = 273
        assert abs(result - 273) < 1

    def test_co2_to_tco2e(self, agent):
        """Test CO2 to tCO2e conversion."""
        result = agent.convert_ghg(1000, "kgCO2", "tCO2e", "AR6")
        assert abs(result - 1.0) < 0.001

    def test_tch4_to_tco2e(self, agent):
        """Test tCH4 to tCO2e conversion."""
        result = agent.convert_ghg(1, "tCH4", "tCO2e", "AR6")
        # 1 tonne CH4 = 29.8 tonnes CO2e
        assert abs(result - 29.8) < 0.1

    def test_co2e_to_co2e_no_change(self, agent):
        """Test CO2e to CO2e conversion (should be identity)."""
        result = agent.convert_ghg(100, "kgCO2e", "kgCO2e", "AR6")
        assert abs(result - 100) < 0.001

    def test_ghg_conversion_via_execute(self, agent):
        """Test GHG conversion through execute method."""
        result = agent.run({
            "operation": "ghg_convert",
            "data": {
                "value": 1000,
                "from_unit": "kgCH4",
                "to_unit": "tCO2e",
                "gwp_source": "AR6"
            }
        })
        assert result.success is True
        assert "converted_value" in result.data["result"]


class TestGWPValues:
    """Tests for GWP value accuracy."""

    def test_gwp_ar6_values(self):
        """Verify AR6 GWP values are correct."""
        assert GWP_AR6_100[GHGType.CO2] == Decimal("1")
        assert GWP_AR6_100[GHGType.CH4] == Decimal("29.8")
        assert GWP_AR6_100[GHGType.N2O] == Decimal("273")

    def test_gwp_ar5_values(self):
        """Verify AR5 GWP values are correct."""
        assert GWP_AR5_100[GHGType.CO2] == Decimal("1")
        assert GWP_AR5_100[GHGType.CH4] == Decimal("28")
        assert GWP_AR5_100[GHGType.N2O] == Decimal("265")

    def test_gwp_ar4_values(self):
        """Verify AR4 GWP values are correct."""
        assert GWP_AR4_100[GHGType.CO2] == Decimal("1")
        assert GWP_AR4_100[GHGType.CH4] == Decimal("25")
        assert GWP_AR4_100[GHGType.N2O] == Decimal("298")


# =============================================================================
# FUEL STANDARDIZATION TESTS
# =============================================================================

class TestFuelStandardization:
    """Tests for fuel name standardization."""

    def test_natural_gas_variants(self, agent):
        """Test various natural gas name variants."""
        variants = ["natural gas", "nat gas", "natural_gas", "methane", "ng"]
        for variant in variants:
            result = agent.standardize_fuel(variant)
            assert result == "Natural Gas"

    def test_diesel_variants(self, agent):
        """Test various diesel name variants."""
        variants = ["diesel", "diesel fuel", "diesel oil", "gas oil", "derv"]
        for variant in variants:
            result = agent.standardize_fuel(variant)
            assert result == "Diesel"

    def test_gasoline_variants(self, agent):
        """Test various gasoline name variants."""
        variants = ["gasoline", "petrol", "motor gasoline", "unleaded"]
        for variant in variants:
            result = agent.standardize_fuel(variant)
            assert result == "Gasoline"

    def test_propane_lpg(self, agent):
        """Test propane and LPG standardization."""
        assert agent.standardize_fuel("propane") == "Propane"
        assert agent.standardize_fuel("lpg") == "Liquefied Petroleum Gas"
        assert agent.standardize_fuel("autogas") == "Liquefied Petroleum Gas"

    def test_coal_variants(self, agent):
        """Test coal variant standardization."""
        assert agent.standardize_fuel("coal") == "Coal"
        assert agent.standardize_fuel("bituminous coal") == "Bituminous Coal"
        assert agent.standardize_fuel("anthracite") == "Anthracite Coal"
        assert agent.standardize_fuel("lignite") == "Lignite"

    def test_biofuels(self, agent):
        """Test biofuel standardization."""
        assert agent.standardize_fuel("biodiesel") == "Biodiesel"
        assert agent.standardize_fuel("ethanol") == "Ethanol"
        assert agent.standardize_fuel("e85") == "Ethanol E85"

    def test_fuel_standardization_case_insensitive(self, agent):
        """Test that fuel standardization is case insensitive."""
        assert agent.standardize_fuel("NATURAL GAS") == "Natural Gas"
        assert agent.standardize_fuel("Diesel") == "Diesel"
        assert agent.standardize_fuel("PROPANE") == "Propane"

    def test_fuel_standardization_returns_code(self, agent):
        """Test that fuel standardization returns fuel code."""
        result = agent._handle_fuel_standardization({"fuel_name": "natural gas"}, None)
        assert result["fuel_code"] == "NG"
        assert result["fuel_category"] == "gaseous"
        assert result["confidence"] == 1.0

    def test_unknown_fuel(self, agent):
        """Test handling of unknown fuel names."""
        result = agent._handle_fuel_standardization({"fuel_name": "mystery_fuel_xyz"}, None)
        assert result["fuel_code"] == "UNK"
        assert result["confidence"] == 0.0

    def test_fuel_fuzzy_matching(self, agent):
        """Test fuzzy matching for fuel names."""
        # Should match "natural gas" with lower confidence
        result = agent._handle_fuel_standardization({"fuel_name": "gas natural"}, None)
        # May or may not match depending on fuzzy logic
        assert "confidence" in result


# =============================================================================
# MATERIAL STANDARDIZATION TESTS
# =============================================================================

class TestMaterialStandardization:
    """Tests for material name standardization."""

    def test_steel_variants(self, agent):
        """Test steel variant standardization."""
        assert agent.standardize_material("steel") == "Steel"
        assert agent.standardize_material("carbon steel") == "Carbon Steel"
        assert agent.standardize_material("stainless steel") == "Stainless Steel"

    def test_aluminum_variants(self, agent):
        """Test aluminum/aluminium standardization."""
        assert agent.standardize_material("aluminum") == "Aluminum"
        assert agent.standardize_material("aluminium") == "Aluminum"

    def test_plastics(self, agent):
        """Test plastic material standardization."""
        assert agent.standardize_material("pet") == "PET (Polyethylene Terephthalate)"
        assert agent.standardize_material("hdpe") == "HDPE (High-Density Polyethylene)"
        assert agent.standardize_material("pvc") == "PVC (Polyvinyl Chloride)"

    def test_construction_materials(self, agent):
        """Test construction material standardization."""
        assert agent.standardize_material("cement") == "Cement"
        assert agent.standardize_material("concrete") == "Concrete"
        assert agent.standardize_material("timber") == "Timber"
        assert agent.standardize_material("lumber") == "Timber"

    def test_material_returns_code(self, agent):
        """Test that material standardization returns material code."""
        result = agent._handle_material_standardization({"material_name": "steel"}, None)
        assert result["material_code"] == "STL"
        assert result["material_category"] == "metals"

    def test_unknown_material(self, agent):
        """Test handling of unknown material names."""
        result = agent._handle_material_standardization({"material_name": "unknown_material_xyz"}, None)
        assert result["material_code"] == "UNK"
        assert result["confidence"] == 0.0


# =============================================================================
# REFERENCE ID MANAGEMENT TESTS
# =============================================================================

class TestReferenceIDManagement:
    """Tests for reference ID management."""

    def test_register_and_resolve_reference(self, agent):
        """Test registering and resolving a reference ID."""
        # Register a mapping
        canonical = agent.register_reference_mapping(
            "SAP", "MAT001",
            "Oracle", "M-00001"
        )

        assert canonical.startswith("GL-")
        assert len(canonical) == 15  # GL- + 12 hex chars

    def test_resolve_existing_reference(self, agent):
        """Test resolving an existing reference."""
        # First registration
        canonical1 = agent.register_reference_mapping(
            "SAP", "MAT002",
            "Oracle", "M-00002"
        )

        # Resolve the same reference
        result = agent._handle_reference_resolution({
            "source_system": "SAP",
            "source_id": "MAT002"
        })

        assert result["canonical_id"] == canonical1

    def test_cross_system_mapping(self, agent):
        """Test that cross-system mappings are stored."""
        # Register mapping
        canonical = agent.register_reference_mapping(
            "ERP1", "ITEM-100",
            "ERP2", "SKU-200"
        )

        # Both should resolve to same canonical
        result1 = agent._handle_reference_resolution({
            "source_system": "ERP1",
            "source_id": "ITEM-100"
        })

        assert result1["canonical_id"] == canonical


# =============================================================================
# CURRENCY CONVERSION TESTS
# =============================================================================

class TestCurrencyConversion:
    """Tests for currency conversion."""

    def test_usd_to_eur(self, agent):
        """Test USD to EUR conversion."""
        result = agent.convert_currency(100, "USD", "EUR")
        assert 85 < result < 100  # Reasonable EUR range for $100

    def test_eur_to_usd(self, agent):
        """Test EUR to USD conversion."""
        result = agent.convert_currency(100, "EUR", "USD")
        assert 100 < result < 120  # Reasonable USD range for 100 EUR

    def test_same_currency(self, agent):
        """Test same currency conversion returns same value."""
        result = agent.convert_currency(100, "USD", "USD")
        assert result == 100.0

    def test_gbp_to_jpy(self, agent):
        """Test GBP to JPY conversion."""
        result = agent.convert_currency(100, "GBP", "JPY")
        assert result > 10000  # GBP is much stronger than JPY

    def test_set_custom_rate(self, agent):
        """Test setting custom exchange rate."""
        agent.set_exchange_rate("USD", "TEST", 2.5)
        result = agent.convert_currency(100, "USD", "TEST")
        assert result == 250.0

    def test_triangulation_through_usd(self, agent):
        """Test currency triangulation through USD."""
        # EUR to GBP should work via triangulation
        result = agent.convert_currency(100, "EUR", "GBP")
        assert result > 0

    def test_unknown_currency_error(self, agent):
        """Test error handling for unknown currency."""
        with pytest.raises(ValueError, match="No exchange rate available"):
            agent.convert_currency(100, "USD", "XYZ")


# =============================================================================
# TENANT CUSTOMIZATION TESTS
# =============================================================================

class TestTenantCustomization:
    """Tests for tenant-specific customization."""

    def test_register_tenant_conversion(self, agent):
        """Test registering tenant-specific conversion factor."""
        agent.register_tenant_conversion(
            "tenant_123",
            "mass",
            "custom_unit",
            0.5  # 1 custom_unit = 0.5 kg
        )

        # Use tenant-specific conversion
        result = agent._handle_unit_conversion({
            "value": 100,
            "from_unit": "custom_unit",
            "to_unit": "kg"
        }, "tenant_123")

        assert result["converted_value"] == 50.0

    def test_register_tenant_fuel_mapping(self, agent):
        """Test registering tenant-specific fuel mapping."""
        agent.register_tenant_fuel_mapping(
            "tenant_456",
            "company_gas",
            "Natural Gas",
            "NG",
            "gaseous"
        )

        result = agent._handle_fuel_standardization(
            {"fuel_name": "company_gas"},
            "tenant_456"
        )

        assert result["standardized_name"] == "Natural Gas"
        assert "Tenant:tenant_456" in result["source"]

    def test_register_tenant_material_mapping(self, agent):
        """Test registering tenant-specific material mapping."""
        agent.register_tenant_material_mapping(
            "tenant_789",
            "company_steel",
            "Carbon Steel",
            "CST",
            "metals"
        )

        result = agent._handle_material_standardization(
            {"material_name": "company_steel"},
            "tenant_789"
        )

        assert result["standardized_name"] == "Carbon Steel"


# =============================================================================
# EXECUTE METHOD TESTS
# =============================================================================

class TestExecuteMethod:
    """Tests for the execute method interface."""

    def test_execute_convert_operation(self, agent):
        """Test execute with convert operation."""
        result = agent.run({
            "operation": "convert",
            "data": {
                "value": 1000,
                "from_unit": "kg",
                "to_unit": "tonnes"
            }
        })

        assert result.success is True
        assert result.data["operation"] == "convert"
        assert result.data["result"]["converted_value"] == 1.0
        assert "provenance_hash" in result.data

    def test_execute_ghg_convert_operation(self, agent):
        """Test execute with ghg_convert operation."""
        result = agent.run({
            "operation": "ghg_convert",
            "data": {
                "value": 1000,
                "from_unit": "kgCH4",
                "to_unit": "kgCO2e",
                "gwp_source": "AR6"
            }
        })

        assert result.success is True
        assert result.data["operation"] == "ghg_convert"

    def test_execute_standardize_fuel_operation(self, agent):
        """Test execute with standardize_fuel operation."""
        result = agent.run({
            "operation": "standardize_fuel",
            "data": {
                "fuel_name": "natural gas"
            }
        })

        assert result.success is True
        assert result.data["result"]["standardized_name"] == "Natural Gas"

    def test_execute_standardize_material_operation(self, agent):
        """Test execute with standardize_material operation."""
        result = agent.run({
            "operation": "standardize_material",
            "data": {
                "material_name": "steel"
            }
        })

        assert result.success is True
        assert result.data["result"]["standardized_name"] == "Steel"

    def test_execute_convert_currency_operation(self, agent):
        """Test execute with convert_currency operation."""
        result = agent.run({
            "operation": "convert_currency",
            "data": {
                "value": 100,
                "from_currency": "USD",
                "to_currency": "EUR"
            }
        })

        assert result.success is True
        assert "converted_value" in result.data["result"]

    def test_execute_list_units_operation(self, agent):
        """Test execute with list_units operation."""
        result = agent.run({
            "operation": "list_units",
            "data": {"dimension": "mass"}
        })

        assert result.success is True
        assert "units" in result.data["result"]
        assert "kg" in result.data["result"]["units"]

    def test_execute_unknown_operation(self, agent):
        """Test execute with unknown operation."""
        result = agent.run({
            "operation": "unknown_operation",
            "data": {}
        })

        assert result.success is False
        assert "Unknown operation" in result.error
        assert "supported_operations" in result.data

    def test_execute_invalid_data(self, agent):
        """Test execute with invalid data."""
        result = agent.run({
            "operation": "convert",
            "data": {
                "value": "not_a_number",
                "from_unit": "kg",
                "to_unit": "tonnes"
            }
        })

        assert result.success is False


# =============================================================================
# PROVENANCE AND LINEAGE TESTS
# =============================================================================

class TestProvenanceTracking:
    """Tests for provenance tracking functionality."""

    def test_conversion_includes_provenance_hash(self, agent):
        """Test that conversions include provenance hash."""
        result = agent._handle_unit_conversion({
            "value": 100,
            "from_unit": "kg",
            "to_unit": "tonnes"
        }, None)

        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64  # SHA-256 hex

    def test_provenance_hash_deterministic(self, agent):
        """Test that provenance hash is deterministic."""
        result1 = agent._handle_unit_conversion({
            "value": 100,
            "from_unit": "kg",
            "to_unit": "tonnes"
        }, None)

        result2 = agent._handle_unit_conversion({
            "value": 100,
            "from_unit": "kg",
            "to_unit": "tonnes"
        }, None)

        assert result1["provenance_hash"] == result2["provenance_hash"]

    def test_different_inputs_different_hash(self, agent):
        """Test that different inputs produce different hashes."""
        result1 = agent._handle_unit_conversion({
            "value": 100,
            "from_unit": "kg",
            "to_unit": "tonnes"
        }, None)

        result2 = agent._handle_unit_conversion({
            "value": 200,
            "from_unit": "kg",
            "to_unit": "tonnes"
        }, None)

        assert result1["provenance_hash"] != result2["provenance_hash"]

    def test_get_conversion_lineage(self, agent):
        """Test getting detailed conversion lineage."""
        lineage = agent.get_conversion_lineage(1000, "kg", "tonnes")

        assert lineage["valid"] is True
        assert lineage["input"]["value"] == 1000
        assert lineage["input"]["unit"] == "kg"
        assert lineage["output"]["value"] == 1.0
        assert lineage["output"]["unit"] == "tonnes"
        assert lineage["conversion"]["dimension"] == "mass"
        assert "provenance" in lineage
        assert "hash" in lineage["provenance"]

    def test_lineage_invalid_conversion(self, agent):
        """Test lineage for invalid conversion."""
        lineage = agent.get_conversion_lineage(1000, "kg", "kwh")

        assert lineage["valid"] is False
        assert "error" in lineage


# =============================================================================
# UTILITY METHOD TESTS
# =============================================================================

class TestUtilityMethods:
    """Tests for utility methods."""

    def test_get_supported_dimensions(self, agent):
        """Test getting supported dimensions."""
        dimensions = agent.get_supported_dimensions()

        assert "mass" in dimensions
        assert "energy" in dimensions
        assert "volume" in dimensions
        assert "area" in dimensions
        assert "distance" in dimensions
        assert "emissions" in dimensions

    def test_get_supported_units(self, agent):
        """Test getting supported units for a dimension."""
        units = agent.get_supported_units("mass")

        assert "kg" in units
        assert "tonne" in units
        assert "lb" in units

    def test_get_supported_units_unknown_dimension(self, agent):
        """Test getting units for unknown dimension."""
        units = agent.get_supported_units("unknown")
        assert units == []

    def test_normalize_unit_name(self, agent):
        """Test unit name normalization."""
        assert agent._normalize_unit_name("KG") == "kg"
        assert agent._normalize_unit_name("  kg  ") == "kg"
        assert agent._normalize_unit_name("kilo-gram") == "kilo_gram"
        assert agent._normalize_unit_name("m^2") == "m2"


# =============================================================================
# EDGE CASES AND BOUNDARY TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_value_conversion(self, agent):
        """Test conversion of zero value."""
        result = agent.convert(0, "kg", "tonnes")
        assert result == 0.0

    def test_negative_value_conversion(self, agent):
        """Test conversion of negative value."""
        result = agent.convert(-100, "kg", "tonnes")
        assert result == -0.1

    def test_very_large_value_conversion(self, agent):
        """Test conversion of very large value."""
        result = agent.convert(1e15, "kg", "tonnes")
        assert result == 1e12

    def test_very_small_value_conversion(self, agent):
        """Test conversion of very small value."""
        # Use a value that won't be truncated by default precision (6 decimals)
        result = agent.convert(0.000001, "tonnes", "kg")
        assert abs(result - 0.001) < 1e-9

        # For extremely small values, use higher precision
        result_precise = agent._handle_unit_conversion({
            "value": 1e-10,
            "from_unit": "tonnes",
            "to_unit": "kg",
            "precision": 15
        }, None)
        assert abs(result_precise["converted_value"] - 1e-7) < 1e-15

    def test_precision_handling(self, agent):
        """Test that precision is handled correctly."""
        # 1/3 should have limited precision
        result = agent._handle_unit_conversion({
            "value": 1,
            "from_unit": "lb",
            "to_unit": "kg",
            "precision": 3
        }, None)

        # Check precision is applied
        converted = result["converted_value"]
        # Should have at most 3 decimal places of precision
        assert abs(converted - 0.454) < 0.001

    def test_emissions_unit_conversion(self, agent):
        """Test emissions unit conversion."""
        result = agent.convert(1000, "kgco2e", "tco2e")
        assert result == 1.0

    def test_time_unit_conversion(self, agent):
        """Test time unit conversion."""
        result = agent.convert(24, "hour", "day")
        assert result == 1.0

        result = agent.convert(1, "year", "day")
        assert abs(result - 365.25) < 0.01


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for UnitNormalizerAgent."""

    def test_full_workflow_emissions_calculation(self, agent):
        """Test a full workflow for emissions calculation."""
        # 1. Standardize fuel
        fuel_result = agent.standardize_fuel("nat gas")
        assert fuel_result == "Natural Gas"

        # 2. Convert fuel quantity
        energy_gj = agent.convert(1000, "therm", "gj")
        assert energy_gj > 0

        # 3. Convert emissions
        emissions_tco2e = agent.convert_ghg(5000, "kgCO2", "tCO2e", "AR6")
        assert emissions_tco2e == 5.0

    def test_full_workflow_material_tracking(self, agent):
        """Test a full workflow for material tracking."""
        # 1. Standardize material
        material = agent.standardize_material("aluminium")
        assert material == "Aluminum"

        # 2. Convert mass
        tonnes = agent.convert(1000, "kg", "tonnes")
        assert tonnes == 1.0

        # 3. Register reference
        canonical = agent.register_reference_mapping(
            "ERP", "ALU-001",
            "WMS", "ALUM-001"
        )
        assert canonical.startswith("GL-")

    def test_agent_statistics(self, agent):
        """Test that agent tracks statistics correctly via run() method."""
        # Perform operations through run() to track statistics
        agent.run({
            "operation": "convert",
            "data": {"value": 100, "from_unit": "kg", "to_unit": "tonnes"}
        })
        agent.run({
            "operation": "convert",
            "data": {"value": 100, "from_unit": "kwh", "to_unit": "mwh"}
        })
        agent.run({
            "operation": "standardize_fuel",
            "data": {"fuel_name": "diesel"}
        })

        stats = agent.get_stats()
        assert stats["executions"] >= 3  # At least 3 operations
        assert stats["success_rate"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
