# -*- coding: utf-8 -*-
"""
Unit Tests for HeatingValueCalculator

Tests all methods of HeatingValueCalculator with 85%+ coverage.
Validates:
- LHV/HHV lookup and calculation
- LHV <-> HHV conversions using hydrogen content
- Energy content from mass/volume quantities
- Temperature-corrected density for volume inputs
- Unit conversions (MJ, MMBtu, kg, m3)

Author: GL-TestEngineer
Date: 2025-01-01
"""

import pytest
from decimal import Decimal
from datetime import date

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.heating_value_calculator import (
    HeatingValueCalculator,
    HeatingValueInput,
    HeatingValueResult,
    FuelProperties,
    HeatingValueType,
    FuelType,
    DEFAULT_FUEL_PROPERTIES,
)


@pytest.mark.unit
class TestHeatingValueCalculatorInitialization:
    """Tests for HeatingValueCalculator initialization."""

    def test_default_initialization(self):
        """Test calculator initializes with default fuel properties."""
        calc = HeatingValueCalculator()

        assert calc.NAME == "HeatingValueCalculator"
        assert calc.VERSION == "1.0.0"
        assert calc._fuel_db == DEFAULT_FUEL_PROPERTIES

    def test_custom_fuel_database_initialization(self):
        """Test calculator initializes with custom fuel database."""
        custom_db = {
            "custom_fuel": FuelProperties(
                fuel_type="custom_fuel",
                hhv_mj_kg=Decimal("48.00"),
                lhv_mj_kg=Decimal("45.00"),
                density_kg_m3=Decimal("850.0"),
                hydrogen_wt_pct=Decimal("14.0"),
                carbon_wt_pct=Decimal("85.0"),
                sulfur_wt_pct=Decimal("0.05"),
                ash_wt_pct=Decimal("0.01"),
                moisture_wt_pct=Decimal("0.0"),
                source_standard="Custom",
                effective_date=date(2024, 1, 1),
                uncertainty_pct=Decimal("2.0"),
            )
        }

        calc = HeatingValueCalculator(fuel_database=custom_db)

        assert "custom_fuel" in calc._fuel_db
        assert calc._fuel_db["custom_fuel"].hhv_mj_kg == Decimal("48.00")


@pytest.mark.unit
class TestHeatingValueCalculatorMassInputs:
    """Tests for heating value calculations with mass inputs."""

    def test_calculate_energy_from_mass_kg(self, heating_value_calculator):
        """Test energy calculation from mass in kg."""
        fuel_input = HeatingValueInput(
            fuel_type="diesel",
            quantity=Decimal("1000"),  # 1000 kg
            quantity_unit="kg",
        )

        result = heating_value_calculator.calculate(fuel_input)

        # Diesel LHV = 43.00 MJ/kg
        expected_lhv = Decimal("1000") * Decimal("43.00")
        assert result.lhv_mj == expected_lhv
        assert result.mass_kg == Decimal("1000.000000")

    def test_calculate_energy_from_mass_tonne(self, heating_value_calculator):
        """Test energy calculation from mass in tonnes."""
        fuel_input = HeatingValueInput(
            fuel_type="diesel",
            quantity=Decimal("5"),  # 5 tonnes = 5000 kg
            quantity_unit="tonne",
        )

        result = heating_value_calculator.calculate(fuel_input)

        assert result.mass_kg == Decimal("5000.000000")
        # Diesel LHV = 43.00 MJ/kg
        expected_lhv = Decimal("5000") * Decimal("43.00")
        assert result.lhv_mj == expected_lhv

    def test_calculate_energy_from_mass_lb(self, heating_value_calculator):
        """Test energy calculation from mass in pounds."""
        fuel_input = HeatingValueInput(
            fuel_type="diesel",
            quantity=Decimal("2204.62"),  # ~1000 kg in lb
            quantity_unit="lb",
        )

        result = heating_value_calculator.calculate(fuel_input)

        # 2204.62 lb * 0.45359237 kg/lb ~ 1000 kg
        assert result.mass_kg == pytest.approx(Decimal("1000"), rel=Decimal("0.001"))


@pytest.mark.unit
class TestHeatingValueCalculatorVolumeInputs:
    """Tests for heating value calculations with volume inputs."""

    def test_calculate_energy_from_volume_m3(self, heating_value_calculator):
        """Test energy calculation from volume in cubic meters."""
        fuel_input = HeatingValueInput(
            fuel_type="diesel",
            quantity=Decimal("1.0"),  # 1 m3
            quantity_unit="m3",
        )

        result = heating_value_calculator.calculate(fuel_input)

        # Diesel density = 840 kg/m3, so 1 m3 = 840 kg
        assert result.mass_kg == Decimal("840.000000")
        expected_lhv = Decimal("840") * Decimal("43.00")
        assert result.lhv_mj == expected_lhv

    def test_calculate_energy_from_volume_liters(self, heating_value_calculator):
        """Test energy calculation from volume in liters."""
        fuel_input = HeatingValueInput(
            fuel_type="diesel",
            quantity=Decimal("1000"),  # 1000 L = 1 m3
            quantity_unit="L",
        )

        result = heating_value_calculator.calculate(fuel_input)

        # 1000 L = 1 m3, diesel density = 840 kg/m3
        assert result.mass_kg == Decimal("840.000000")

    def test_calculate_energy_from_volume_bbl(self, heating_value_calculator):
        """Test energy calculation from volume in barrels."""
        fuel_input = HeatingValueInput(
            fuel_type="diesel",
            quantity=Decimal("1"),  # 1 barrel = 158.987 L = 0.158987 m3
            quantity_unit="bbl",
        )

        result = heating_value_calculator.calculate(fuel_input)

        # 1 bbl = 0.158987294928 m3
        expected_mass = Decimal("0.158987294928") * Decimal("840.0")
        assert result.mass_kg == pytest.approx(expected_mass, rel=Decimal("0.001"))

    def test_calculate_energy_from_volume_gal(self, heating_value_calculator):
        """Test energy calculation from volume in gallons."""
        fuel_input = HeatingValueInput(
            fuel_type="diesel",
            quantity=Decimal("42"),  # 42 gal = 1 barrel
            quantity_unit="gal",
        )

        result = heating_value_calculator.calculate(fuel_input)

        # 42 gal * 0.003785411784 m3/gal * 840 kg/m3
        expected_mass = Decimal("42") * Decimal("0.003785411784") * Decimal("840.0")
        assert result.mass_kg == pytest.approx(expected_mass, rel=Decimal("0.001"))


@pytest.mark.unit
class TestHeatingValueCalculatorTemperatureCorrection:
    """Tests for temperature-corrected density calculations."""

    def test_density_at_standard_temperature(self, heating_value_calculator):
        """Test density at standard temperature (15C) is unchanged."""
        density_15c = Decimal("840.0")
        result = heating_value_calculator.calculate_density_at_temperature(
            density_15c_kg_m3=density_15c,
            temperature_c=Decimal("15.0"),
            fuel_type="diesel",
        )

        # At 15C, density should be unchanged
        assert result == density_15c

    def test_density_at_higher_temperature(self, heating_value_calculator):
        """Test density at higher temperature (expansion)."""
        density_15c = Decimal("840.0")
        result = heating_value_calculator.calculate_density_at_temperature(
            density_15c_kg_m3=density_15c,
            temperature_c=Decimal("30.0"),
            fuel_type="diesel",
        )

        # At higher temperature, density should be lower
        assert result < density_15c

    def test_density_at_lower_temperature(self, heating_value_calculator):
        """Test density at lower temperature (contraction)."""
        density_15c = Decimal("840.0")
        result = heating_value_calculator.calculate_density_at_temperature(
            density_15c_kg_m3=density_15c,
            temperature_c=Decimal("0.0"),
            fuel_type="diesel",
        )

        # At lower temperature, density should be higher
        assert result > density_15c

    def test_volume_with_temperature_correction(self, heating_value_calculator):
        """Test volume calculation with temperature correction."""
        fuel_input = HeatingValueInput(
            fuel_type="diesel",
            quantity=Decimal("1000"),
            quantity_unit="L",
            observed_temperature_c=Decimal("30.0"),
        )

        result = heating_value_calculator.calculate(fuel_input)

        # At 30C, density is lower, so mass should be lower
        # Compare to calculation at 15C
        fuel_input_15c = HeatingValueInput(
            fuel_type="diesel",
            quantity=Decimal("1000"),
            quantity_unit="L",
            observed_temperature_c=None,  # Uses 15C reference
        )

        result_15c = heating_value_calculator.calculate(fuel_input_15c)

        # Mass at 30C should be less than mass at 15C for same volume
        assert result.mass_kg < result_15c.mass_kg


@pytest.mark.unit
class TestHeatingValueCalculatorLHVHHVConversion:
    """Tests for LHV <-> HHV conversions using hydrogen content."""

    def test_convert_hhv_to_lhv(self, heating_value_calculator):
        """Test HHV to LHV conversion using hydrogen content."""
        # Diesel: HHV = 45.80, hydrogen = 13.5%
        hhv = Decimal("45.80")
        hydrogen_pct = Decimal("13.5")

        lhv = heating_value_calculator.convert_hhv_to_lhv(hhv, hydrogen_pct)

        # LHV = HHV - (H% / 100) * 9 * 2.442
        # LHV = 45.80 - 0.135 * 9 * 2.442 = 45.80 - 2.967 = ~42.83
        expected_lhv = hhv - (hydrogen_pct / Decimal("100")) * Decimal("9") * Decimal("2.442")
        assert lhv == pytest.approx(expected_lhv, rel=Decimal("0.001"))

    def test_convert_lhv_to_hhv(self, heating_value_calculator):
        """Test LHV to HHV conversion using hydrogen content."""
        # Diesel: LHV = 43.00, hydrogen = 13.5%
        lhv = Decimal("43.00")
        hydrogen_pct = Decimal("13.5")

        hhv = heating_value_calculator.convert_lhv_to_hhv(lhv, hydrogen_pct)

        # HHV = LHV + (H% / 100) * 9 * 2.442
        expected_hhv = lhv + (hydrogen_pct / Decimal("100")) * Decimal("9") * Decimal("2.442")
        assert hhv == pytest.approx(expected_hhv, rel=Decimal("0.001"))

    def test_hhv_lhv_roundtrip(self, heating_value_calculator):
        """Test that HHV->LHV->HHV roundtrip is consistent."""
        original_hhv = Decimal("45.80")
        hydrogen_pct = Decimal("13.5")

        lhv = heating_value_calculator.convert_hhv_to_lhv(original_hhv, hydrogen_pct)
        recovered_hhv = heating_value_calculator.convert_lhv_to_hhv(lhv, hydrogen_pct)

        assert recovered_hhv == pytest.approx(original_hhv, rel=Decimal("0.0001"))

    def test_hydrogen_override_in_calculation(self, heating_value_calculator):
        """Test that hydrogen content override affects LHV calculation."""
        fuel_input_default = HeatingValueInput(
            fuel_type="diesel",
            quantity=Decimal("1000"),
            quantity_unit="kg",
        )

        fuel_input_override = HeatingValueInput(
            fuel_type="diesel",
            quantity=Decimal("1000"),
            quantity_unit="kg",
            hydrogen_content_override=Decimal("15.0"),  # Different hydrogen content
        )

        result_default = heating_value_calculator.calculate(fuel_input_default)
        result_override = heating_value_calculator.calculate(fuel_input_override)

        # Different hydrogen content should give different LHV
        assert result_default.lhv_mj != result_override.lhv_mj


@pytest.mark.unit
class TestHeatingValueCalculatorFuelTypes:
    """Tests for different fuel types."""

    @pytest.mark.parametrize("fuel_type", [
        "natural_gas",
        "diesel",
        "gasoline",
        "lpg",
        "hydrogen",
        "heavy_fuel_oil",
        "marine_fuel_oil",
        "coal_bituminous",
    ])
    def test_supported_fuel_types(self, heating_value_calculator, fuel_type):
        """Test all supported fuel types can be calculated."""
        fuel_input = HeatingValueInput(
            fuel_type=fuel_type,
            quantity=Decimal("1000"),
            quantity_unit="kg",
        )

        result = heating_value_calculator.calculate(fuel_input)

        assert result.lhv_mj > Decimal("0")
        assert result.hhv_mj > Decimal("0")
        assert result.hhv_mj > result.lhv_mj  # HHV always > LHV

    def test_unknown_fuel_type_raises_error(self, heating_value_calculator):
        """Test that unknown fuel type raises ValueError."""
        fuel_input = HeatingValueInput(
            fuel_type="unknown_fuel",
            quantity=Decimal("1000"),
            quantity_unit="kg",
        )

        with pytest.raises(ValueError, match="Unknown fuel type"):
            heating_value_calculator.calculate(fuel_input)


@pytest.mark.unit
class TestHeatingValueCalculatorProvenance:
    """Tests for provenance tracking."""

    def test_provenance_hash_generated(self, heating_value_calculator):
        """Test that provenance hash is generated."""
        fuel_input = HeatingValueInput(
            fuel_type="diesel",
            quantity=Decimal("1000"),
            quantity_unit="kg",
        )

        result = heating_value_calculator.calculate(fuel_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256

    def test_calculation_steps_recorded(self, heating_value_calculator):
        """Test that calculation steps are recorded."""
        fuel_input = HeatingValueInput(
            fuel_type="diesel",
            quantity=Decimal("1000"),
            quantity_unit="kg",
        )

        result = heating_value_calculator.calculate(fuel_input)

        assert len(result.calculation_steps) > 0
        # Verify step structure
        for step in result.calculation_steps:
            assert "step" in step
            assert "operation" in step


@pytest.mark.unit
class TestHeatingValueCalculatorPrecision:
    """Tests for calculation precision."""

    @pytest.mark.parametrize("precision", [3, 6, 9])
    def test_output_precision(self, heating_value_calculator, precision):
        """Test output respects precision parameter."""
        fuel_input = HeatingValueInput(
            fuel_type="diesel",
            quantity=Decimal("1234.567890123"),
            quantity_unit="kg",
        )

        result = heating_value_calculator.calculate(fuel_input, precision=precision)

        # Check decimal places
        lhv_str = str(result.lhv_mj)
        if "." in lhv_str:
            decimal_places = len(lhv_str.split(".")[1])
            assert decimal_places == precision


@pytest.mark.unit
class TestHeatingValueCalculatorHelperMethods:
    """Tests for helper methods."""

    def test_get_fuel_properties(self, heating_value_calculator):
        """Test getting fuel properties."""
        props = heating_value_calculator.get_fuel_properties("diesel")

        assert props.fuel_type == "diesel"
        assert props.lhv_mj_kg == Decimal("43.00")
        assert props.hhv_mj_kg == Decimal("45.80")
        assert props.density_kg_m3 == Decimal("840.0")

    def test_get_fuel_properties_unknown(self, heating_value_calculator):
        """Test getting properties for unknown fuel raises error."""
        with pytest.raises(ValueError, match="Unknown fuel type"):
            heating_value_calculator.get_fuel_properties("unknown_fuel")

    def test_list_fuels(self, heating_value_calculator):
        """Test listing available fuels."""
        fuels = heating_value_calculator.list_fuels()

        assert "diesel" in fuels
        assert "natural_gas" in fuels
        assert "hydrogen" in fuels
        assert len(fuels) > 0


@pytest.mark.unit
class TestFuelPropertiesDataClass:
    """Tests for FuelProperties data class."""

    def test_fuel_properties_to_dict(self):
        """Test FuelProperties serialization."""
        props = DEFAULT_FUEL_PROPERTIES["diesel"]
        data = props.to_dict()

        assert data["fuel_type"] == "diesel"
        assert "lhv_mj_kg" in data
        assert "hhv_mj_kg" in data
        assert "density_kg_m3" in data
        assert "hydrogen_wt_pct" in data
        assert "source_standard" in data
        assert "effective_date" in data


@pytest.mark.unit
class TestHeatingValueResultDataClass:
    """Tests for HeatingValueResult data class."""

    def test_heating_value_result_to_dict(self, heating_value_calculator):
        """Test HeatingValueResult serialization."""
        fuel_input = HeatingValueInput(
            fuel_type="diesel",
            quantity=Decimal("1000"),
            quantity_unit="kg",
        )

        result = heating_value_calculator.calculate(fuel_input)
        data = result.to_dict()

        assert "input_fuel_type" in data
        assert "input_quantity" in data
        assert "lhv_mj" in data
        assert "hhv_mj" in data
        assert "mass_kg" in data
        assert "provenance_hash" in data
        assert "fuel_properties" in data


@pytest.mark.unit
class TestUnsupportedUnits:
    """Tests for unsupported unit handling."""

    def test_unsupported_unit_raises_error(self, heating_value_calculator):
        """Test that unsupported unit raises ValueError."""
        fuel_input = HeatingValueInput(
            fuel_type="diesel",
            quantity=Decimal("1000"),
            quantity_unit="unknown_unit",
        )

        with pytest.raises(ValueError, match="Unsupported unit"):
            heating_value_calculator.calculate(fuel_input)
