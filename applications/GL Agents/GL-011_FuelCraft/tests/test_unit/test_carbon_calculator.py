# -*- coding: utf-8 -*-
"""
Unit Tests for CarbonCalculator

Tests all methods of CarbonCalculator with 85%+ coverage.
Validates:
- TTW/WTT/WTW boundary emissions
- Energy-weighted blend intensity
- Emission factor governance with date validity
- GHG Protocol scope attribution
- Provenance hash generation

Author: GL-TestEngineer
Date: 2025-01-01
"""

import pytest
from decimal import Decimal
from datetime import date, datetime, timezone

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.carbon_calculator import (
    CarbonCalculator,
    CarbonInput,
    CarbonResult,
    EmissionFactor,
    EmissionBoundary,
    EmissionScope,
    GHGType,
    DEFAULT_EMISSION_FACTORS,
)


@pytest.mark.unit
class TestCarbonCalculatorInitialization:
    """Tests for CarbonCalculator initialization."""

    def test_default_initialization(self):
        """Test calculator initializes with default emission factors."""
        calc = CarbonCalculator()

        assert calc.NAME == "CarbonCalculator"
        assert calc.VERSION == "1.0.0"
        assert calc._factors == DEFAULT_EMISSION_FACTORS

    def test_custom_factors_initialization(self):
        """Test calculator initializes with custom emission factors."""
        custom_factors = {
            "custom_fuel": [
                EmissionFactor(
                    factor_id="custom_ttw",
                    fuel_type="custom_fuel",
                    boundary=EmissionBoundary.TTW,
                    factor_value=Decimal("0.0800"),
                    factor_unit="MJ",
                    ghg_type=GHGType.CO2E,
                    scope=EmissionScope.SCOPE_1,
                    source_standard="Custom Standard",
                    effective_date=date(2024, 1, 1),
                    expiry_date=None,
                    region="Global",
                    uncertainty_pct=Decimal("5.0"),
                )
            ]
        }

        calc = CarbonCalculator(emission_factors=custom_factors)

        assert "custom_fuel" in calc._factors
        assert calc._factors["custom_fuel"][0].factor_id == "custom_ttw"


@pytest.mark.unit
class TestCarbonCalculatorTTWEmissions:
    """Tests for Tank-to-Wake (direct combustion) emissions."""

    def test_ttw_emissions_diesel(self, carbon_calculator):
        """Test TTW emissions for diesel combustion."""
        carbon_input = CarbonInput(
            fuel_type="diesel",
            energy_mj=Decimal("1000"),
            boundary=EmissionBoundary.TTW,
            region="Global",
            reference_date=date(2024, 1, 1),
        )

        result = carbon_calculator.calculate(carbon_input)

        # Diesel TTW: 0.0741 kgCO2e/MJ
        expected_emissions = Decimal("1000") * Decimal("0.0741")
        assert result.ttw_emissions_kg_co2e == pytest.approx(
            expected_emissions, rel=Decimal("0.001")
        )
        assert result.ttw_intensity_kg_co2e_mj == Decimal("0.074100")

    def test_ttw_emissions_natural_gas(self, carbon_calculator):
        """Test TTW emissions for natural gas combustion."""
        carbon_input = CarbonInput(
            fuel_type="natural_gas",
            energy_mj=Decimal("5000"),
            boundary=EmissionBoundary.TTW,
            region="Global",
            reference_date=date(2024, 1, 1),
        )

        result = carbon_calculator.calculate(carbon_input)

        # Natural gas TTW: 0.0561 kgCO2e/MJ
        expected_emissions = Decimal("5000") * Decimal("0.0561")
        assert result.ttw_emissions_kg_co2e == pytest.approx(
            expected_emissions, rel=Decimal("0.001")
        )

    def test_ttw_emissions_hydrogen_zero(self, carbon_calculator):
        """Test TTW emissions for hydrogen (zero combustion CO2)."""
        carbon_input = CarbonInput(
            fuel_type="hydrogen",
            energy_mj=Decimal("10000"),
            boundary=EmissionBoundary.TTW,
            region="Global",
            reference_date=date(2024, 1, 1),
        )

        result = carbon_calculator.calculate(carbon_input)

        # Green hydrogen has zero TTW emissions
        assert result.ttw_emissions_kg_co2e == Decimal("0.000000")
        assert result.ttw_intensity_kg_co2e_mj == Decimal("0.000000")


@pytest.mark.unit
class TestCarbonCalculatorWTWEmissions:
    """Tests for Well-to-Wake (full lifecycle) emissions."""

    def test_wtw_emissions_includes_wtt(self, carbon_calculator):
        """Test WTW emissions include both TTW and WTT components."""
        carbon_input = CarbonInput(
            fuel_type="diesel",
            energy_mj=Decimal("1000"),
            boundary=EmissionBoundary.WTW,
            region="Global",
            reference_date=date(2024, 1, 1),
        )

        result = carbon_calculator.calculate(carbon_input)

        # WTW = TTW + WTT
        assert result.wtw_emissions_kg_co2e == (
            result.ttw_emissions_kg_co2e + result.wtt_emissions_kg_co2e
        )
        assert result.wtw_intensity_kg_co2e_mj == (
            result.ttw_intensity_kg_co2e_mj + result.wtt_intensity_kg_co2e_mj
        )

    def test_wtw_emissions_heavy_fuel_oil(self, carbon_calculator):
        """Test WTW emissions for heavy fuel oil."""
        carbon_input = CarbonInput(
            fuel_type="heavy_fuel_oil",
            energy_mj=Decimal("2000"),
            boundary=EmissionBoundary.WTW,
            region="Global",
            reference_date=date(2024, 1, 1),
        )

        result = carbon_calculator.calculate(carbon_input)

        # HFO has both TTW and WTT components
        assert result.ttw_emissions_kg_co2e > Decimal("0")
        assert result.wtt_emissions_kg_co2e > Decimal("0")
        assert result.wtw_emissions_kg_co2e > result.ttw_emissions_kg_co2e


@pytest.mark.unit
class TestCarbonCalculatorBlendIntensity:
    """Tests for energy-weighted blend carbon intensity."""

    def test_blend_intensity_single_fuel(self, carbon_calculator):
        """Test blend intensity with single fuel."""
        fuel_energies = [("diesel", Decimal("1000"))]

        intensity, details = carbon_calculator.calculate_blend_intensity(
            fuel_energies=fuel_energies,
            boundary=EmissionBoundary.WTW,
            reference_date=date(2024, 1, 1),
        )

        # Single fuel should have its own intensity
        assert intensity > Decimal("0")
        assert len(details) == 1

    def test_blend_intensity_multiple_fuels(self, carbon_calculator):
        """Test energy-weighted blend intensity with multiple fuels."""
        fuel_energies = [
            ("diesel", Decimal("700")),      # Higher CI
            ("natural_gas", Decimal("300")), # Lower CI
        ]

        intensity, details = carbon_calculator.calculate_blend_intensity(
            fuel_energies=fuel_energies,
            boundary=EmissionBoundary.WTW,
            reference_date=date(2024, 1, 1),
        )

        # Blend intensity should be between individual fuels
        diesel_intensity = Decimal("0.0902")  # diesel WTW
        ng_intensity = Decimal("0.0744")      # natural gas WTW

        assert intensity < diesel_intensity
        assert intensity > ng_intensity

    def test_blend_intensity_zero_energy(self, carbon_calculator):
        """Test blend intensity with zero total energy."""
        fuel_energies = [
            ("diesel", Decimal("0")),
            ("natural_gas", Decimal("0")),
        ]

        intensity, details = carbon_calculator.calculate_blend_intensity(
            fuel_energies=fuel_energies,
            boundary=EmissionBoundary.WTW,
        )

        # Zero energy should return zero intensity
        assert intensity == Decimal("0")


@pytest.mark.unit
class TestCarbonCalculatorFactorGovernance:
    """Tests for emission factor governance and date validity."""

    def test_factor_date_validity(self, carbon_calculator):
        """Test that factors are selected based on date validity."""
        # Use a date within factor validity
        carbon_input = CarbonInput(
            fuel_type="diesel",
            energy_mj=Decimal("1000"),
            boundary=EmissionBoundary.TTW,
            reference_date=date(2024, 1, 1),
        )

        result = carbon_calculator.calculate(carbon_input)

        # Should use valid factor
        assert len(result.factors_used) > 0
        for factor in result.factors_used:
            assert factor.is_valid(date(2024, 1, 1))

    def test_factor_region_preference(self, carbon_calculator):
        """Test that region-specific factors are preferred."""
        # Use EU region for diesel (has region-specific WTT factor)
        carbon_input = CarbonInput(
            fuel_type="diesel",
            energy_mj=Decimal("1000"),
            boundary=EmissionBoundary.WTW,
            region="EU",
            reference_date=date(2024, 1, 1),
        )

        result = carbon_calculator.calculate(carbon_input)

        # Check that EU-specific WTT factor was used
        wtt_factors = [f for f in result.factors_used if f.boundary == EmissionBoundary.WTT]
        if wtt_factors:
            eu_factors = [f for f in wtt_factors if f.region == "EU"]
            assert len(eu_factors) > 0

    def test_unknown_fuel_raises_error(self, carbon_calculator):
        """Test that unknown fuel type raises ValueError."""
        carbon_input = CarbonInput(
            fuel_type="unknown_fuel",
            energy_mj=Decimal("1000"),
        )

        with pytest.raises(ValueError, match="Unknown fuel type"):
            carbon_calculator.calculate(carbon_input)


@pytest.mark.unit
class TestEmissionFactorMethods:
    """Tests for EmissionFactor data class methods."""

    def test_emission_factor_is_valid_in_range(self):
        """Test factor validity when date is in range."""
        factor = EmissionFactor(
            factor_id="test_factor",
            fuel_type="test_fuel",
            boundary=EmissionBoundary.TTW,
            factor_value=Decimal("0.0750"),
            factor_unit="MJ",
            ghg_type=GHGType.CO2E,
            scope=EmissionScope.SCOPE_1,
            source_standard="Test",
            effective_date=date(2020, 1, 1),
            expiry_date=date(2030, 12, 31),
            region="Global",
            uncertainty_pct=Decimal("5.0"),
        )

        assert factor.is_valid(date(2024, 6, 15)) is True
        assert factor.is_valid(date(2020, 1, 1)) is True  # Boundary
        assert factor.is_valid(date(2030, 12, 31)) is True  # Boundary

    def test_emission_factor_is_valid_before_effective(self):
        """Test factor validity when date is before effective date."""
        factor = EmissionFactor(
            factor_id="test_factor",
            fuel_type="test_fuel",
            boundary=EmissionBoundary.TTW,
            factor_value=Decimal("0.0750"),
            factor_unit="MJ",
            ghg_type=GHGType.CO2E,
            scope=EmissionScope.SCOPE_1,
            source_standard="Test",
            effective_date=date(2020, 1, 1),
            expiry_date=None,
            region="Global",
            uncertainty_pct=Decimal("5.0"),
        )

        assert factor.is_valid(date(2019, 12, 31)) is False

    def test_emission_factor_is_valid_after_expiry(self):
        """Test factor validity when date is after expiry."""
        factor = EmissionFactor(
            factor_id="test_factor",
            fuel_type="test_fuel",
            boundary=EmissionBoundary.TTW,
            factor_value=Decimal("0.0750"),
            factor_unit="MJ",
            ghg_type=GHGType.CO2E,
            scope=EmissionScope.SCOPE_1,
            source_standard="Test",
            effective_date=date(2020, 1, 1),
            expiry_date=date(2023, 12, 31),
            region="Global",
            uncertainty_pct=Decimal("5.0"),
        )

        assert factor.is_valid(date(2024, 1, 1)) is False

    def test_emission_factor_to_dict(self):
        """Test EmissionFactor serialization."""
        factor = EmissionFactor(
            factor_id="diesel_ttw_ipcc2006",
            fuel_type="diesel",
            boundary=EmissionBoundary.TTW,
            factor_value=Decimal("0.0741"),
            factor_unit="MJ",
            ghg_type=GHGType.CO2E,
            scope=EmissionScope.SCOPE_1,
            source_standard="IPCC 2006",
            effective_date=date(2006, 1, 1),
            expiry_date=None,
            region="Global",
            uncertainty_pct=Decimal("5.0"),
            notes="Test factor",
        )

        data = factor.to_dict()

        assert data["factor_id"] == "diesel_ttw_ipcc2006"
        assert data["fuel_type"] == "diesel"
        assert data["boundary"] == "TTW"
        assert data["factor_value"] == "0.0741"
        assert data["ghg_type"] == "CO2e"
        assert data["scope"] == "scope_1"


@pytest.mark.unit
class TestCarbonResultMethods:
    """Tests for CarbonResult data class methods."""

    def test_carbon_result_provenance_hash(self, carbon_calculator):
        """Test CarbonResult generates provenance hash."""
        carbon_input = CarbonInput(
            fuel_type="diesel",
            energy_mj=Decimal("1000"),
        )

        result = carbon_calculator.calculate(carbon_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256

    def test_carbon_result_to_dict(self, carbon_calculator):
        """Test CarbonResult serialization."""
        carbon_input = CarbonInput(
            fuel_type="diesel",
            energy_mj=Decimal("1000"),
        )

        result = carbon_calculator.calculate(carbon_input)
        data = result.to_dict()

        assert "fuel_type" in data
        assert "energy_mj" in data
        assert "ttw_emissions_kg_co2e" in data
        assert "wtt_emissions_kg_co2e" in data
        assert "wtw_emissions_kg_co2e" in data
        assert "provenance_hash" in data
        assert "factors_used" in data


@pytest.mark.unit
class TestCarbonCalculatorHelperMethods:
    """Tests for helper methods."""

    def test_list_fuels(self, carbon_calculator):
        """Test list of available fuel types."""
        fuels = carbon_calculator.list_fuels()

        assert "diesel" in fuels
        assert "natural_gas" in fuels
        assert "heavy_fuel_oil" in fuels
        assert "hydrogen" in fuels

    def test_list_factors(self, carbon_calculator):
        """Test list of factors for a fuel type."""
        factors = carbon_calculator.list_factors("diesel")

        assert len(factors) > 0
        for factor in factors:
            assert factor.fuel_type == "diesel"

    def test_get_factor_public_interface(self, carbon_calculator):
        """Test public interface to get emission factor."""
        factor = carbon_calculator.get_factor(
            fuel_type="diesel",
            boundary=EmissionBoundary.TTW,
            reference_date=date(2024, 1, 1),
        )

        assert factor is not None
        assert factor.fuel_type == "diesel"
        assert factor.boundary == EmissionBoundary.TTW


@pytest.mark.unit
class TestCarbonCalculatorPrecision:
    """Tests for calculation precision."""

    @pytest.mark.parametrize("precision", [3, 6, 9])
    def test_output_precision(self, carbon_calculator, precision):
        """Test output respects precision parameter."""
        carbon_input = CarbonInput(
            fuel_type="diesel",
            energy_mj=Decimal("1234.567890123"),
        )

        result = carbon_calculator.calculate(carbon_input, precision=precision)

        # Check decimal places
        ttw_str = str(result.ttw_emissions_kg_co2e)
        if "." in ttw_str:
            decimal_places = len(ttw_str.split(".")[1])
            assert decimal_places == precision

    def test_calculation_steps_recorded(self, carbon_calculator):
        """Test that calculation steps are recorded for audit."""
        carbon_input = CarbonInput(
            fuel_type="diesel",
            energy_mj=Decimal("1000"),
        )

        result = carbon_calculator.calculate(carbon_input)

        assert len(result.calculation_steps) > 0
        # Verify TTW calculation step exists
        ttw_steps = [s for s in result.calculation_steps if s.get("operation") == "calculate_ttw"]
        assert len(ttw_steps) == 1


@pytest.mark.unit
class TestEmissionBoundaryEnum:
    """Tests for EmissionBoundary enumeration."""

    def test_emission_boundary_values(self):
        """Test EmissionBoundary enum values."""
        assert EmissionBoundary.TTW.value == "TTW"
        assert EmissionBoundary.WTT.value == "WTT"
        assert EmissionBoundary.WTW.value == "WTW"


@pytest.mark.unit
class TestEmissionScopeEnum:
    """Tests for EmissionScope enumeration."""

    def test_emission_scope_values(self):
        """Test EmissionScope enum values."""
        assert EmissionScope.SCOPE_1.value == "scope_1"
        assert EmissionScope.SCOPE_2.value == "scope_2"
        assert EmissionScope.SCOPE_3.value == "scope_3"


@pytest.mark.unit
class TestGHGTypeEnum:
    """Tests for GHGType enumeration."""

    def test_ghg_type_values(self):
        """Test GHGType enum values."""
        assert GHGType.CO2.value == "CO2"
        assert GHGType.CH4.value == "CH4"
        assert GHGType.N2O.value == "N2O"
        assert GHGType.CO2E.value == "CO2e"
