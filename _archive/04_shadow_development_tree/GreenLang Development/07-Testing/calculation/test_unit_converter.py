# -*- coding: utf-8 -*-
"""
Unit Tests for Unit Converter

Tests deterministic unit conversions for energy, mass, volume, etc.
"""

import pytest
from decimal import Decimal
from greenlang.agents.calculation.emissions.unit_converter import (
    UnitConverter,
    UnitConversionError,
)


class TestUnitConverter:
    """Test UnitConverter functionality"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test converter"""
        self.converter = UnitConverter()

    # Energy Conversions
    def test_kwh_to_mwh(self):
        """Test kWh to MWh conversion"""
        result = self.converter.convert(
            value=1000,
            from_unit="kwh",
            to_unit="mwh"
        )
        assert abs(result - 1.0) < 1e-9

    def test_kwh_to_gj(self):
        """Test kWh to GJ conversion"""
        result = self.converter.convert(
            value=277.778,
            from_unit="kwh",
            to_unit="gj"
        )
        assert abs(result - 1.0) < 0.01

    def test_therms_to_kwh(self):
        """Test therms to kWh conversion"""
        result = self.converter.convert(
            value=1,
            from_unit="therms",
            to_unit="kwh"
        )
        # 1 therm = 29.3 kWh
        assert abs(result - 29.3071) < 0.1

    # Volume Conversions
    def test_liters_to_gallons(self):
        """Test liters to gallons conversion"""
        result = self.converter.convert(
            value=3.78541,
            from_unit="liters",
            to_unit="gallons"
        )
        assert abs(result - 1.0) < 0.001

    def test_cubic_meters_to_liters(self):
        """Test cubic meters to liters"""
        result = self.converter.convert(
            value=1,
            from_unit="cubic_meter",
            to_unit="liters"
        )
        assert abs(result - 1000.0) < 1e-9

    # Mass Conversions
    def test_kg_to_tonnes(self):
        """Test kg to tonnes conversion"""
        result = self.converter.convert(
            value=1000,
            from_unit="kg",
            to_unit="tonnes"
        )
        assert abs(result - 1.0) < 1e-9

    def test_lb_to_kg(self):
        """Test pounds to kg conversion"""
        result = self.converter.convert(
            value=2.20462,
            from_unit="lb",
            to_unit="kg"
        )
        assert abs(result - 1.0) < 0.001

    # Determinism Tests
    def test_conversion_determinism(self):
        """Test conversion is deterministic"""
        result1 = self.converter.convert(100, "kwh", "mwh")
        result2 = self.converter.convert(100, "kwh", "mwh")

        assert result1 == result2

    def test_round_trip_conversion(self):
        """Test round-trip conversion preserves value"""
        original = 100.0

        # Convert kWh -> MWh -> kWh
        mwh = self.converter.convert(original, "kwh", "mwh")
        back_to_kwh = self.converter.convert(mwh, "mwh", "kwh")

        assert abs(back_to_kwh - original) < 0.0001

    # Error Handling
    def test_unknown_from_unit(self):
        """Test error on unknown source unit"""
        with pytest.raises(UnitConversionError, match="Unknown unit"):
            self.converter.convert(100, "invalid_unit", "kwh")

    def test_unknown_to_unit(self):
        """Test error on unknown target unit"""
        with pytest.raises(UnitConversionError, match="Unknown unit"):
            self.converter.convert(100, "kwh", "invalid_unit")

    def test_incompatible_units(self):
        """Test error on incompatible unit types"""
        with pytest.raises(UnitConversionError, match="Cannot convert between different unit types"):
            self.converter.convert(100, "kwh", "kg")  # Energy to mass

    def test_negative_value(self):
        """Test negative value conversion (allowed for temperature offsets, etc)"""
        # Note: Negative values are allowed in the consolidated implementation
        result = self.converter.convert(-100, "kwh", "mwh")
        assert abs(result - (-0.1)) < 1e-9

    # Precision Tests
    @pytest.mark.parametrize("value,from_unit,to_unit,expected", [
        (1, "kwh", "kwh", 1),  # Same unit
        (0, "kwh", "mwh", 0),  # Zero value
        (1000000, "kwh", "gwh", 1),  # Large value
        (0.001, "mwh", "kwh", 1),  # Small value
    ])
    def test_conversion_accuracy(self, value, from_unit, to_unit, expected):
        """Test conversion accuracy for various values"""
        result = self.converter.convert(value, from_unit, to_unit)
        assert abs(result - float(expected)) < 0.001


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
