"""
GL-001 ThermalCommand: Tag Dictionary Tests

Comprehensive tests for the tag dictionary and unit conversion system.

Test Coverage:
- Tag definition validation
- Tag registration and lookup
- Unit conversion accuracy
- SCADA/OPC mapping
- Tag value validation
"""

import pytest
from datetime import datetime, timezone

import sys
sys.path.insert(0, str(__file__).replace('\\', '/').rsplit('/tests/', 1)[0])

from data_contracts.tag_dictionary import (
    # Enums
    TagDataType,
    TagCategory,
    UnitCategory,
    QualityCode,
    # Models
    TagDefinition,
    TagValue,
    # Classes
    TagDictionary,
    UnitConversion,
    # Functions
    get_tag_dictionary,
)


# =============================================================================
# UnitConversion Tests
# =============================================================================

class TestUnitConversion:
    """Tests for UnitConversion class."""

    # -------------------------------------------------------------------------
    # Pressure Conversion Tests
    # -------------------------------------------------------------------------

    def test_pressure_bar_to_psi(self):
        """Test bar to psi conversion."""
        result = UnitConversion.convert_pressure(10.0, "bar", "psi")
        assert abs(result - 145.038) < 0.01  # 10 bar ~ 145 psi

    def test_pressure_psi_to_bar(self):
        """Test psi to bar conversion."""
        result = UnitConversion.convert_pressure(100.0, "psi", "bar")
        assert abs(result - 6.89476) < 0.001  # 100 psi ~ 6.89 bar

    def test_pressure_kpa_to_bar(self):
        """Test kPa to bar conversion."""
        result = UnitConversion.convert_pressure(1000.0, "kPa", "bar")
        assert abs(result - 10.0) < 0.001

    def test_pressure_atm_to_bar(self):
        """Test atm to bar conversion."""
        result = UnitConversion.convert_pressure(1.0, "atm", "bar")
        assert abs(result - 1.01325) < 0.001

    def test_pressure_identity(self):
        """Test same unit conversion returns same value."""
        result = UnitConversion.convert_pressure(42.5, "bar", "bar")
        assert result == 42.5

    # -------------------------------------------------------------------------
    # Temperature Conversion Tests
    # -------------------------------------------------------------------------

    def test_temperature_c_to_f(self):
        """Test Celsius to Fahrenheit conversion."""
        result = UnitConversion.convert_temperature(100.0, "C", "F")
        assert abs(result - 212.0) < 0.001  # Boiling point

        result = UnitConversion.convert_temperature(0.0, "C", "F")
        assert abs(result - 32.0) < 0.001  # Freezing point

    def test_temperature_f_to_c(self):
        """Test Fahrenheit to Celsius conversion."""
        result = UnitConversion.convert_temperature(212.0, "F", "C")
        assert abs(result - 100.0) < 0.001

        result = UnitConversion.convert_temperature(32.0, "F", "C")
        assert abs(result - 0.0) < 0.001

    def test_temperature_c_to_k(self):
        """Test Celsius to Kelvin conversion."""
        result = UnitConversion.convert_temperature(0.0, "C", "K")
        assert abs(result - 273.15) < 0.001

        result = UnitConversion.convert_temperature(100.0, "C", "K")
        assert abs(result - 373.15) < 0.001

    def test_temperature_k_to_c(self):
        """Test Kelvin to Celsius conversion."""
        result = UnitConversion.convert_temperature(273.15, "K", "C")
        assert abs(result - 0.0) < 0.001

    def test_temperature_negative(self):
        """Test negative temperature conversion."""
        result = UnitConversion.convert_temperature(-40.0, "C", "F")
        assert abs(result - (-40.0)) < 0.001  # -40C = -40F

    # -------------------------------------------------------------------------
    # Mass Flow Conversion Tests
    # -------------------------------------------------------------------------

    def test_mass_flow_kgh_to_tph(self):
        """Test kg/h to t/h conversion."""
        result = UnitConversion.convert_mass_flow(1000.0, "kg/h", "t/h")
        assert abs(result - 1.0) < 0.001

    def test_mass_flow_tph_to_kgh(self):
        """Test t/h to kg/h conversion."""
        result = UnitConversion.convert_mass_flow(1.0, "t/h", "kg/h")
        assert abs(result - 1000.0) < 0.001

    def test_mass_flow_kgs_to_kgh(self):
        """Test kg/s to kg/h conversion."""
        result = UnitConversion.convert_mass_flow(1.0, "kg/s", "kg/h")
        assert abs(result - 3600.0) < 0.001

    # -------------------------------------------------------------------------
    # Power Conversion Tests
    # -------------------------------------------------------------------------

    def test_power_mw_to_kw(self):
        """Test MW to kW conversion."""
        result = UnitConversion.convert_power(10.0, "MW", "kW")
        assert abs(result - 10000.0) < 0.001

    def test_power_kw_to_mw(self):
        """Test kW to MW conversion."""
        result = UnitConversion.convert_power(5000.0, "kW", "MW")
        assert abs(result - 5.0) < 0.001

    def test_power_hp_to_kw(self):
        """Test hp to kW conversion."""
        result = UnitConversion.convert_power(100.0, "hp", "kW")
        assert abs(result - 74.57) < 0.01  # 100 hp ~ 74.57 kW

    # -------------------------------------------------------------------------
    # Energy Conversion Tests
    # -------------------------------------------------------------------------

    def test_energy_mwh_to_gj(self):
        """Test MWh to GJ conversion."""
        result = UnitConversion.convert_energy(1.0, "MWh", "GJ")
        assert abs(result - 3.6) < 0.001  # 1 MWh = 3.6 GJ

    def test_energy_mmbtu_to_mwh(self):
        """Test MMBtu to MWh conversion."""
        result = UnitConversion.convert_energy(1.0, "MMBtu", "MWh")
        assert abs(result - 0.293071) < 0.001

    # -------------------------------------------------------------------------
    # Error Handling Tests
    # -------------------------------------------------------------------------

    def test_invalid_pressure_unit(self):
        """Test error on invalid pressure unit."""
        with pytest.raises(ValueError, match="Unknown pressure unit"):
            UnitConversion.convert_pressure(10.0, "invalid", "bar")

    def test_invalid_temperature_unit(self):
        """Test error on invalid temperature unit."""
        with pytest.raises(ValueError, match="Unknown temperature unit"):
            UnitConversion.convert_temperature(100.0, "X", "C")


# =============================================================================
# TagDefinition Tests
# =============================================================================

class TestTagDefinition:
    """Tests for TagDefinition model."""

    def test_create_float_tag(self):
        """Test creating a float tag definition."""
        tag = TagDefinition(
            tag_name="steam.headerA.pressure",
            display_name="Header A Pressure",
            description="Steam header A pressure",
            category=TagCategory.STEAM,
            data_type=TagDataType.FLOAT,
            unit="bar(g)",
            unit_category=UnitCategory.PRESSURE,
            low_limit=0.0,
            high_limit=50.0,
        )
        assert tag.tag_name == "steam.headerA.pressure"
        assert tag.data_type == TagDataType.FLOAT

    def test_create_bool_tag(self):
        """Test creating a boolean tag definition."""
        tag = TagDefinition(
            tag_name="sis.permissive.dispatch_enabled",
            display_name="Dispatch Enabled",
            description="SIS dispatch permissive",
            category=TagCategory.SIS,
            data_type=TagDataType.BOOL,
            unit="",
            unit_category=UnitCategory.DIMENSIONLESS,
        )
        assert tag.data_type == TagDataType.BOOL

    def test_tag_name_pattern_valid(self):
        """Test valid tag name patterns."""
        valid_names = [
            "steam.headerA.pressure",
            "boiler.B1.fuel_flow",
            "unit.U1.heat_demand",
            "price.electricity.rt",
        ]

        for name in valid_names:
            tag = TagDefinition(
                tag_name=name,
                display_name="Test",
                description="Test",
                category=TagCategory.STEAM,
                data_type=TagDataType.FLOAT,
                unit="bar",
                unit_category=UnitCategory.PRESSURE,
            )
            assert tag.tag_name == name

    def test_validate_value_float_valid(self):
        """Test float value validation - valid."""
        tag = TagDefinition(
            tag_name="steam.headerA.pressure",
            display_name="Test",
            description="Test",
            category=TagCategory.STEAM,
            data_type=TagDataType.FLOAT,
            unit="bar(g)",
            unit_category=UnitCategory.PRESSURE,
            low_limit=0.0,
            high_limit=50.0,
        )

        is_valid, error = tag.validate_value(25.0)
        assert is_valid is True
        assert error is None

    def test_validate_value_float_out_of_range(self):
        """Test float value validation - out of range."""
        tag = TagDefinition(
            tag_name="steam.headerA.pressure",
            display_name="Test",
            description="Test",
            category=TagCategory.STEAM,
            data_type=TagDataType.FLOAT,
            unit="bar(g)",
            unit_category=UnitCategory.PRESSURE,
            low_limit=0.0,
            high_limit=50.0,
        )

        is_valid, error = tag.validate_value(60.0)
        assert is_valid is False
        assert "above" in error.lower()

    def test_validate_value_wrong_type(self):
        """Test value validation - wrong type."""
        tag = TagDefinition(
            tag_name="sis.permissive.dispatch_enabled",
            display_name="Test",
            description="Test",
            category=TagCategory.SIS,
            data_type=TagDataType.BOOL,
            unit="",
            unit_category=UnitCategory.DIMENSIONLESS,
        )

        is_valid, error = tag.validate_value("not_a_bool")
        assert is_valid is False
        assert "bool" in error.lower()

    def test_get_alarm_status(self):
        """Test alarm status determination."""
        tag = TagDefinition(
            tag_name="steam.headerA.pressure",
            display_name="Test",
            description="Test",
            category=TagCategory.STEAM,
            data_type=TagDataType.FLOAT,
            unit="bar(g)",
            unit_category=UnitCategory.PRESSURE,
            low_alarm=5.0,
            low_warning=10.0,
            high_warning=40.0,
            high_alarm=45.0,
        )

        assert tag.get_alarm_status(25.0) == "normal"
        assert tag.get_alarm_status(8.0) == "low_warning"
        assert tag.get_alarm_status(3.0) == "low_alarm"
        assert tag.get_alarm_status(42.0) == "high_warning"
        assert tag.get_alarm_status(47.0) == "high_alarm"


# =============================================================================
# TagDictionary Tests
# =============================================================================

class TestTagDictionary:
    """Tests for TagDictionary class."""

    @pytest.fixture
    def tag_dict(self) -> TagDictionary:
        """Create fresh tag dictionary for each test."""
        return TagDictionary()

    def test_minimum_tags_registered(self, tag_dict):
        """Test that minimum required tags are registered."""
        minimum_tags = [
            "steam.headerA.pressure",
            "steam.headerA.temperature",
            "steam.headerA.flow_total",
            "boiler.B1.fuel_flow",
            "boiler.B1.max_rate",
            "unit.U1.heat_demand",
            "sis.permissive.dispatch_enabled",
            "alarm.high_pressure.headerA",
            "price.electricity.rt",
            "weather.temp_forecast",
            "cmms.asset.B1.health_score",
        ]

        for tag_name in minimum_tags:
            tag = tag_dict.get_tag(tag_name)
            assert tag is not None, f"Missing required tag: {tag_name}"

    def test_get_tag_by_name(self, tag_dict):
        """Test getting tag by name."""
        tag = tag_dict.get_tag("steam.headerA.pressure")
        assert tag is not None
        assert tag.unit == "bar(g)"
        assert tag.category == TagCategory.STEAM

    def test_get_nonexistent_tag(self, tag_dict):
        """Test getting non-existent tag returns None."""
        tag = tag_dict.get_tag("nonexistent.tag.name")
        assert tag is None

    def test_get_tags_by_category(self, tag_dict):
        """Test getting tags by category."""
        steam_tags = tag_dict.get_tags_by_category(TagCategory.STEAM)
        assert len(steam_tags) > 0
        assert all(t.category == TagCategory.STEAM for t in steam_tags)

    def test_get_tags_by_pattern(self, tag_dict):
        """Test getting tags by regex pattern."""
        # Get all boiler tags
        tags = tag_dict.get_tags_by_pattern(r"^boiler\..*")
        assert len(tags) > 0
        assert all(t.tag_name.startswith("boiler.") for t in tags)

    def test_resolve_scada_tag(self, tag_dict):
        """Test SCADA tag resolution."""
        # Find tag with SCADA mapping
        tag = tag_dict.get_tag("steam.headerA.pressure")
        if tag and tag.scada_tag:
            resolved = tag_dict.resolve_scada_tag(tag.scada_tag)
            assert resolved == "steam.headerA.pressure"

    def test_validate_value(self, tag_dict):
        """Test value validation through dictionary."""
        is_valid, error = tag_dict.validate_value("steam.headerA.pressure", 25.0)
        assert is_valid is True

        is_valid, error = tag_dict.validate_value("steam.headerA.pressure", 250.0)
        assert is_valid is False

    def test_validate_unknown_tag(self, tag_dict):
        """Test validation of unknown tag."""
        is_valid, error = tag_dict.validate_value("unknown.tag", 100.0)
        assert is_valid is False
        assert "unknown" in error.lower()

    def test_convert_value(self, tag_dict):
        """Test value conversion through dictionary."""
        # Convert pressure from psi to bar(g)
        result = tag_dict.convert_value(
            "steam.headerA.pressure",
            145.0,
            from_unit="psi",
            to_unit="bar(g)"
        )
        assert abs(result - 10.0) < 0.1

    def test_get_critical_tags(self, tag_dict):
        """Test getting critical tags."""
        critical = tag_dict.get_critical_tags()
        assert len(critical) > 0
        assert all(t.is_critical for t in critical)

    def test_register_custom_tag(self, tag_dict):
        """Test registering a custom tag."""
        custom_tag = TagDefinition(
            tag_name="custom.test.tag",
            display_name="Custom Tag",
            description="A custom test tag",
            category=TagCategory.STEAM,
            data_type=TagDataType.FLOAT,
            unit="bar",
            unit_category=UnitCategory.PRESSURE,
        )

        tag_dict.register_tag(custom_tag)
        retrieved = tag_dict.get_tag("custom.test.tag")
        assert retrieved is not None
        assert retrieved.display_name == "Custom Tag"

    def test_dictionary_length(self, tag_dict):
        """Test dictionary length."""
        assert len(tag_dict) > 10  # At least minimum tags

    def test_dictionary_contains(self, tag_dict):
        """Test 'in' operator."""
        assert "steam.headerA.pressure" in tag_dict
        assert "nonexistent.tag" not in tag_dict

    def test_dictionary_iteration(self, tag_dict):
        """Test iteration over dictionary."""
        count = 0
        for tag in tag_dict:
            assert isinstance(tag, TagDefinition)
            count += 1
        assert count == len(tag_dict)

    def test_export_to_dict(self, tag_dict):
        """Test exporting dictionary to dict."""
        export = tag_dict.export_to_dict()
        assert "version" in export
        assert "tags" in export
        assert "tag_count" in export
        assert export["tag_count"] == len(tag_dict)


# =============================================================================
# TagValue Tests
# =============================================================================

class TestTagValue:
    """Tests for TagValue model."""

    def test_create_tag_value(self):
        """Test creating a tag value."""
        value = TagValue(
            tag_name="steam.headerA.pressure",
            value=42.5,
            timestamp=datetime.now(timezone.utc),
            quality=QualityCode.GOOD,
        )
        assert value.value == 42.5
        assert value.quality == QualityCode.GOOD

    def test_tag_value_with_source(self):
        """Test tag value with source."""
        value = TagValue(
            tag_name="steam.headerA.pressure",
            value=42.5,
            timestamp=datetime.now(timezone.utc),
            source="SCADA_OPC",
        )
        assert value.source == "SCADA_OPC"

    def test_quality_codes(self):
        """Test various quality codes."""
        for quality in QualityCode:
            value = TagValue(
                tag_name="test.tag",
                value=0.0,
                timestamp=datetime.now(timezone.utc),
                quality=quality,
            )
            assert value.quality == quality


# =============================================================================
# Singleton Function Tests
# =============================================================================

class TestGetTagDictionary:
    """Tests for get_tag_dictionary singleton function."""

    def test_singleton_returns_same_instance(self):
        """Test that singleton returns same instance."""
        dict1 = get_tag_dictionary()
        dict2 = get_tag_dictionary()
        assert dict1 is dict2

    def test_singleton_has_minimum_tags(self):
        """Test singleton has minimum required tags."""
        tag_dict = get_tag_dictionary()
        assert "steam.headerA.pressure" in tag_dict


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
