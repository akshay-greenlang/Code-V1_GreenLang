# -*- coding: utf-8 -*-
"""
tests/utils/test_unit_conversion.py

Unit Conversion Tests for FuelAgentAI v2 Internationalization

Tests:
- Volume conversions (gallons, liters, m3)
- Energy conversions (therms, kWh, MJ, GJ)
- Mass conversions (tons, tonnes, kg)
- Temperature conversions (F, C, K)
- Pressure conversions (psi, bar, kPa)
- Regional defaults
- Number formatting

Author: GreenLang Framework Team
Date: October 2025
"""

import pytest
from greenlang.utils.unit_conversion import (
    UnitConverter,
    VolumeUnit,
    EnergyUnit,
    MassUnit,
    TemperatureUnit,
    PressureUnit,
    get_regional_defaults,
    format_number,
    REGIONAL_DEFAULTS,
)


class TestVolumeConversions:
    """Test volume unit conversions."""

    def setup_method(self):
        """Setup test fixtures."""
        self.converter = UnitConverter()

    def test_gallons_to_liters(self):
        """Test US gallons to liters conversion."""
        result = self.converter.convert_volume(1.0, "gallons", "liters")
        assert abs(result - 3.78541) < 0.001, "1 gallon should equal 3.78541 liters"

    def test_liters_to_gallons(self):
        """Test liters to US gallons conversion."""
        result = self.converter.convert_volume(3.78541, "liters", "gallons")
        assert abs(result - 1.0) < 0.001, "3.78541 liters should equal 1 gallon"

    def test_gallons_uk_to_liters(self):
        """Test UK gallons to liters conversion."""
        result = self.converter.convert_volume(1.0, "gallons_uk", "liters")
        assert abs(result - 4.54609) < 0.001, "1 UK gallon should equal 4.54609 liters"

    def test_cubic_meters_to_liters(self):
        """Test cubic meters to liters conversion."""
        result = self.converter.convert_volume(1.0, "m3", "liters")
        assert result == 1000.0, "1 m3 should equal 1000 liters"

    def test_roundtrip_conversion(self):
        """Test roundtrip conversion (gallons -> liters -> gallons)."""
        original = 100.0
        liters = self.converter.convert_volume(original, "gallons", "liters")
        back = self.converter.convert_volume(liters, "liters", "gallons")
        assert abs(back - original) < 0.001, "Roundtrip conversion should preserve value"


class TestEnergyConversions:
    """Test energy unit conversions."""

    def setup_method(self):
        """Setup test fixtures."""
        self.converter = UnitConverter()

    def test_therms_to_kwh(self):
        """Test therms to kWh conversion."""
        result = self.converter.convert_energy(1.0, "therms", "kWh")
        assert abs(result - 29.3001) < 0.01, "1 therm should equal 29.3001 kWh"

    def test_kwh_to_therms(self):
        """Test kWh to therms conversion."""
        result = self.converter.convert_energy(29.3001, "kWh", "therms")
        assert abs(result - 1.0) < 0.001, "29.3001 kWh should equal 1 therm"

    def test_mj_to_kwh(self):
        """Test MJ to kWh conversion."""
        result = self.converter.convert_energy(3.6, "MJ", "kWh")
        assert abs(result - 1.0) < 0.001, "3.6 MJ should equal 1 kWh"

    def test_gj_to_kwh(self):
        """Test GJ to kWh conversion."""
        result = self.converter.convert_energy(1.0, "GJ", "kWh")
        assert abs(result - 277.778) < 0.1, "1 GJ should equal 277.778 kWh"

    def test_btu_to_kwh(self):
        """Test BTU to kWh conversion."""
        result = self.converter.convert_energy(3412.14, "BTU", "kWh")
        assert abs(result - 1.0) < 0.01, "3412.14 BTU should equal 1 kWh"


class TestMassConversions:
    """Test mass unit conversions."""

    def setup_method(self):
        """Setup test fixtures."""
        self.converter = UnitConverter()

    def test_tons_us_to_kg(self):
        """Test US tons to kg conversion."""
        result = self.converter.convert_mass(1.0, "tons", "kg")
        assert abs(result - 907.185) < 0.1, "1 US ton should equal 907.185 kg"

    def test_tonnes_to_kg(self):
        """Test metric tonnes to kg conversion."""
        result = self.converter.convert_mass(1.0, "tonnes", "kg")
        assert result == 1000.0, "1 tonne should equal 1000 kg"

    def test_lbs_to_kg(self):
        """Test pounds to kg conversion."""
        result = self.converter.convert_mass(2.20462, "lbs", "kg")
        assert abs(result - 1.0) < 0.001, "2.20462 lbs should equal 1 kg"

    def test_tons_to_tonnes(self):
        """Test US tons to metric tonnes conversion."""
        result = self.converter.convert_mass(1.0, "tons", "tonnes")
        assert abs(result - 0.907185) < 0.001, "1 US ton should equal 0.907185 tonnes"


class TestTemperatureConversions:
    """Test temperature unit conversions."""

    def setup_method(self):
        """Setup test fixtures."""
        self.converter = UnitConverter()

    def test_fahrenheit_to_celsius(self):
        """Test Fahrenheit to Celsius conversion."""
        result = self.converter.convert_temperature(32.0, "F", "C")
        assert abs(result - 0.0) < 0.001, "32°F should equal 0°C"

    def test_celsius_to_fahrenheit(self):
        """Test Celsius to Fahrenheit conversion."""
        result = self.converter.convert_temperature(0.0, "C", "F")
        assert abs(result - 32.0) < 0.001, "0°C should equal 32°F"

    def test_celsius_to_kelvin(self):
        """Test Celsius to Kelvin conversion."""
        result = self.converter.convert_temperature(0.0, "C", "K")
        assert abs(result - 273.15) < 0.01, "0°C should equal 273.15 K"

    def test_fahrenheit_to_kelvin(self):
        """Test Fahrenheit to Kelvin conversion."""
        result = self.converter.convert_temperature(32.0, "F", "K")
        assert abs(result - 273.15) < 0.01, "32°F should equal 273.15 K"

    def test_boiling_point(self):
        """Test boiling point conversion (100°C = 212°F = 373.15 K)."""
        celsius = 100.0
        fahrenheit = self.converter.convert_temperature(celsius, "C", "F")
        kelvin = self.converter.convert_temperature(celsius, "C", "K")

        assert abs(fahrenheit - 212.0) < 0.1, "100°C should equal 212°F"
        assert abs(kelvin - 373.15) < 0.1, "100°C should equal 373.15 K"


class TestPressureConversions:
    """Test pressure unit conversions."""

    def setup_method(self):
        """Setup test fixtures."""
        self.converter = UnitConverter()

    def test_psi_to_kpa(self):
        """Test PSI to kPa conversion."""
        result = self.converter.convert_pressure(14.5038, "psi", "kPa")
        assert abs(result - 100.0) < 0.1, "14.5038 psi should equal 100 kPa"

    def test_bar_to_kpa(self):
        """Test bar to kPa conversion."""
        result = self.converter.convert_pressure(1.0, "bar", "kPa")
        assert result == 100.0, "1 bar should equal 100 kPa"

    def test_atm_to_kpa(self):
        """Test atmosphere to kPa conversion."""
        result = self.converter.convert_pressure(1.0, "atm", "kPa")
        assert abs(result - 101.325) < 0.001, "1 atm should equal 101.325 kPa"


class TestAutoDetectConversion:
    """Test auto-detect unit type and convert."""

    def setup_method(self):
        """Setup test fixtures."""
        self.converter = UnitConverter()

    def test_auto_volume_conversion(self):
        """Test auto-detect and convert volume."""
        result = self.converter.convert(10.0, "gallons", "liters")
        expected = 10.0 * 3.78541
        assert abs(result - expected) < 0.01, "Auto-detect should handle volume conversion"

    def test_auto_energy_conversion(self):
        """Test auto-detect and convert energy."""
        result = self.converter.convert(100.0, "therms", "kWh")
        expected = 100.0 * 29.3001
        assert abs(result - expected) < 1.0, "Auto-detect should handle energy conversion"

    def test_incompatible_units_error(self):
        """Test error on incompatible unit types."""
        with pytest.raises(ValueError, match="Incompatible unit types"):
            self.converter.convert(10.0, "gallons", "kWh")


class TestRegionalDefaults:
    """Test regional defaults."""

    def test_us_defaults(self):
        """Test US regional defaults."""
        defaults = get_regional_defaults("US")

        assert defaults["volume_unit"] == VolumeUnit.GALLONS_US
        assert defaults["energy_unit"] == EnergyUnit.THERMS
        assert defaults["mass_unit"] == MassUnit.TONS_US
        assert defaults["temperature_unit"] == TemperatureUnit.FAHRENHEIT
        assert defaults["currency"] == "USD"
        assert defaults["decimal_separator"] == "."
        assert defaults["thousands_separator"] == ","

    def test_uk_defaults(self):
        """Test UK regional defaults."""
        defaults = get_regional_defaults("UK")

        assert defaults["volume_unit"] == VolumeUnit.LITERS
        assert defaults["energy_unit"] == EnergyUnit.KWH
        assert defaults["mass_unit"] == MassUnit.TONNES
        assert defaults["temperature_unit"] == TemperatureUnit.CELSIUS
        assert defaults["currency"] == "GBP"

    def test_eu_defaults(self):
        """Test EU regional defaults."""
        defaults = get_regional_defaults("EU")

        assert defaults["decimal_separator"] == ","
        assert defaults["thousands_separator"] == "."
        assert defaults["currency"] == "EUR"

    def test_canada_defaults(self):
        """Test Canada regional defaults."""
        defaults = get_regional_defaults("CA")

        assert defaults["volume_unit"] == VolumeUnit.LITERS
        assert defaults["temperature_unit"] == TemperatureUnit.CELSIUS
        assert defaults["currency"] == "CAD"

    def test_unknown_region_fallback(self):
        """Test fallback to US for unknown region."""
        defaults = get_regional_defaults("UNKNOWN")

        assert defaults["currency"] == "USD"


class TestNumberFormatting:
    """Test number formatting by region."""

    def test_us_number_formatting(self):
        """Test US number formatting (1,234.56)."""
        formatted = format_number(1234.56, "US", decimals=2)
        assert formatted == "1,234.56", "US format should use comma for thousands"

    def test_eu_number_formatting(self):
        """Test EU number formatting (1.234,56)."""
        formatted = format_number(1234.56, "EU", decimals=2)
        assert formatted == "1.234,56", "EU format should use period for thousands, comma for decimal"

    def test_large_number_formatting(self):
        """Test large number formatting."""
        formatted = format_number(1234567.89, "US", decimals=2)
        assert formatted == "1,234,567.89", "Should format large numbers with thousands separators"

    def test_decimal_precision(self):
        """Test decimal precision control."""
        formatted_2 = format_number(123.456789, "US", decimals=2)
        formatted_4 = format_number(123.456789, "US", decimals=4)

        assert formatted_2 == "123.46", "Should round to 2 decimals"
        assert formatted_4 == "123.4568", "Should round to 4 decimals"


class TestComprehensiveConversions:
    """Test comprehensive real-world conversion scenarios."""

    def setup_method(self):
        """Setup test fixtures."""
        self.converter = UnitConverter()

    def test_diesel_us_to_eu(self):
        """Test diesel conversion: US gallons to EU liters."""
        # 1000 US gallons diesel
        us_gallons = 1000.0

        # Convert to liters (EU standard)
        eu_liters = self.converter.convert_volume(us_gallons, "gallons", "liters")

        # Expected: 1000 × 3.78541 = 3785.41 liters
        assert abs(eu_liters - 3785.41) < 0.1, \
            "1000 US gallons should equal 3785.41 liters"

    def test_natural_gas_therms_to_kwh(self):
        """Test natural gas conversion: therms to kWh."""
        # 10,000 therms natural gas
        therms = 10000.0

        # Convert to kWh (international standard)
        kwh = self.converter.convert_energy(therms, "therms", "kWh")

        # Expected: 10,000 × 29.3001 = 293,001 kWh
        assert abs(kwh - 293001) < 100, \
            "10,000 therms should equal 293,001 kWh"

    def test_coal_tons_to_tonnes(self):
        """Test coal conversion: US tons to metric tonnes."""
        # 100 US tons coal
        us_tons = 100.0

        # Convert to metric tonnes (international standard)
        tonnes = self.converter.convert_mass(us_tons, "tons", "tonnes")

        # Expected: 100 × 0.907185 = 90.7185 tonnes
        assert abs(tonnes - 90.7185) < 0.01, \
            "100 US tons should equal 90.7185 metric tonnes"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
