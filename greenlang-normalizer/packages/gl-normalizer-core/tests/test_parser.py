"""
Tests for the UnitParser module.
"""

import pytest
from gl_normalizer_core.parser import UnitParser, Quantity, ParseResult, UnitSystem


class TestQuantity:
    """Tests for Quantity model."""

    def test_create_quantity(self):
        """Test creating a basic quantity."""
        q = Quantity(magnitude=100.0, unit="kilogram")
        assert q.magnitude == 100.0
        assert q.unit == "kilogram"

    def test_quantity_to_tuple(self):
        """Test converting quantity to tuple."""
        q = Quantity(magnitude=50.5, unit="liter")
        assert q.to_tuple() == (50.5, "liter")

    def test_quantity_string_representation(self):
        """Test string representation."""
        q = Quantity(magnitude=1.5, unit="metric_ton")
        assert str(q) == "1.5 metric_ton"

    def test_quantity_with_uncertainty(self):
        """Test quantity with uncertainty."""
        q = Quantity(magnitude=100.0, unit="kg", uncertainty=0.5)
        assert q.uncertainty == 0.5


class TestUnitParser:
    """Tests for UnitParser class."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return UnitParser()

    def test_parse_simple_quantity(self, parser):
        """Test parsing a simple quantity string."""
        result = parser.parse("100 kg")
        assert result.success
        assert result.quantity.magnitude == 100.0
        assert result.quantity.unit == "kilogram"

    def test_parse_with_decimal(self, parser):
        """Test parsing quantity with decimal."""
        result = parser.parse("1.5 t")
        assert result.success
        assert result.quantity.magnitude == 1.5

    def test_parse_scientific_notation(self, parser):
        """Test parsing scientific notation."""
        result = parser.parse("1.5e3 kg")
        assert result.success
        assert result.quantity.magnitude == 1500.0

    def test_parse_emissions_unit(self, parser):
        """Test parsing emissions unit."""
        result = parser.parse("100 kg CO2e")
        assert result.success
        assert "CO2" in result.quantity.unit

    def test_parse_energy_unit(self, parser):
        """Test parsing energy unit."""
        result = parser.parse("500 kWh")
        assert result.success
        assert result.quantity.unit == "kilowatt_hour"

    def test_parse_empty_string(self, parser):
        """Test parsing empty string fails gracefully."""
        result = parser.parse("")
        assert not result.success

    def test_parse_invalid_string(self, parser):
        """Test parsing invalid string."""
        result = parser.parse("not a quantity")
        assert not result.success

    def test_strict_mode_raises(self):
        """Test strict mode raises exception."""
        parser = UnitParser(strict_mode=True)
        with pytest.raises(Exception):
            parser.parse("")

    def test_provenance_hash_generated(self, parser):
        """Test provenance hash is generated."""
        result = parser.parse("100 kg")
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64  # SHA-256 hex

    def test_add_custom_alias(self, parser):
        """Test adding custom alias."""
        parser.add_alias("custom_unit", "normalized_unit")
        assert parser.aliases["custom_unit"] == "normalized_unit"

    def test_parse_time_recorded(self, parser):
        """Test parse time is recorded."""
        result = parser.parse("100 kg")
        assert result.parse_time_ms >= 0
