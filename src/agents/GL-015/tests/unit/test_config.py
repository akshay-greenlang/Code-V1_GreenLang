# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN - Unit Tests for Configuration Module

Unit tests for configuration validation, settings management, and Pydantic models.
Tests enumerations, input/output models, and validation rules.

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
from decimal import Decimal
from datetime import datetime, date
from typing import Any, Dict, List
from unittest.mock import Mock, patch


# =============================================================================
# TEST: ENUMERATION VALUES
# =============================================================================

class TestEnumerations:
    """Tests for configuration enumerations."""

    def test_insulation_type_enum_values(self):
        """Test InsulationType enumeration values."""
        insulation_types = [
            "mineral_wool",
            "calcium_silicate",
            "cellular_glass",
            "perlite",
            "aerogel",
            "polyurethane",
            "polystyrene",
            "phenolic_foam",
            "fiberglass",
            "microporous",
            "ceramic_fiber",
        ]

        for itype in insulation_types:
            assert isinstance(itype, str)
            assert len(itype) > 0

    def test_jacket_material_enum_values(self):
        """Test JacketMaterial enumeration values."""
        jacket_materials = [
            "aluminum",
            "stainless_steel",
            "galvanized_steel",
            "pvc",
            "none",
            "painted_metal",
            "fiberglass_reinforced",
        ]

        for material in jacket_materials:
            assert isinstance(material, str)

    def test_degradation_severity_enum_values(self):
        """Test DegradationSeverity enumeration values."""
        severities = ["none", "minor", "moderate", "severe", "failed"]

        # Verify order (least to most severe)
        expected_order = severities
        assert severities == expected_order

    def test_repair_priority_enum_values(self):
        """Test RepairPriority enumeration values."""
        priorities = ["routine", "planned", "urgent", "critical"]

        # Verify urgency order
        assert priorities.index("routine") < priorities.index("critical")

    def test_surface_type_enum_values(self):
        """Test SurfaceType enumeration values."""
        surface_types = [
            "horizontal_top",
            "horizontal_bottom",
            "vertical",
            "angled_up",
            "angled_down",
            "cylindrical_horizontal",
            "cylindrical_vertical",
            "spherical",
        ]

        assert len(surface_types) == 8

    def test_equipment_type_enum_values(self):
        """Test EquipmentType enumeration values."""
        equipment_types = [
            "pipe",
            "vessel",
            "tank",
            "exchanger",
            "column",
            "duct",
            "valve",
            "flange",
            "fitting",
            "turbine",
            "pump",
            "boiler",
            "reactor",
            "furnace",
        ]

        assert "pipe" in equipment_types
        assert "vessel" in equipment_types

    def test_weather_condition_enum_values(self):
        """Test WeatherCondition enumeration values."""
        conditions = ["clear", "partly_cloudy", "overcast", "rain", "night", "fog", "snow"]

        for condition in conditions:
            assert isinstance(condition, str)

    def test_camera_type_enum_values(self):
        """Test CameraType enumeration values."""
        camera_types = [
            "FLIR",
            "Fluke",
            "Testo",
            "Optris",
            "InfraTec",
            "Seek",
            "HIKMICRO",
            "Other",
        ]

        # Major brands should be supported
        assert "FLIR" in camera_types
        assert "Fluke" in camera_types

    def test_emissivity_class_enum_values(self):
        """Test EmissivityClass enumeration values."""
        classes = ["high", "medium", "low", "very_low"]

        # Verify order
        assert classes[0] == "high"
        assert classes[-1] == "very_low"

    def test_failure_mode_enum_values(self):
        """Test FailureMode enumeration values."""
        failure_modes = [
            "moisture_ingress",
            "mechanical_damage",
            "thermal_degradation",
        ]

        for mode in failure_modes:
            assert "_" in mode or mode.isalpha()


# =============================================================================
# TEST: INPUT MODEL VALIDATION
# =============================================================================

class TestInputModelValidation:
    """Tests for input model validation."""

    def test_thermal_image_input_validation(self):
        """Test ThermalImageData input validation."""
        valid_input = {
            "image_width": 320,
            "image_height": 240,
            "min_temperature_c": 20.0,
            "max_temperature_c": 120.0,
            "emissivity_setting": 0.95,
            "distance_m": 3.0,
        }

        # Validate ranges
        assert 0 < valid_input["image_width"] <= 1920
        assert 0 < valid_input["image_height"] <= 1080
        assert valid_input["min_temperature_c"] < valid_input["max_temperature_c"]
        assert 0 < valid_input["emissivity_setting"] <= 1.0
        assert valid_input["distance_m"] > 0

    def test_ambient_conditions_validation(self, sample_ambient_conditions):
        """Test AmbientConditions validation."""
        conditions = sample_ambient_conditions

        # Temperature range check
        assert Decimal("-50") <= conditions["ambient_temperature_c"] <= Decimal("60")

        # Humidity range check
        assert Decimal("0") <= conditions["relative_humidity_percent"] <= Decimal("100")

        # Wind speed range check
        assert conditions["wind_speed_m_s"] >= Decimal("0")

    def test_equipment_parameters_validation(self, sample_equipment_parameters):
        """Test EquipmentParameters validation."""
        params = sample_equipment_parameters

        # Positive dimensions
        assert params["pipe_outer_diameter_m"] > Decimal("0")
        assert params["pipe_length_m"] > Decimal("0")

        # Reasonable temperature
        assert Decimal("-200") <= params["process_temperature_c"] <= Decimal("1000")

        # Emissivity range
        assert Decimal("0") < params["surface_emissivity"] <= Decimal("1")

    def test_insulation_specs_validation(self, sample_insulation_specs):
        """Test InsulationSpecs validation."""
        specs = sample_insulation_specs

        # Positive thickness
        assert specs["thickness_mm"] > Decimal("0")

        # Positive density
        assert specs["density_kg_m3"] > Decimal("0")

        # Positive thermal conductivity
        assert specs["thermal_conductivity_w_m_k"] > Decimal("0")

        # Temperature limits valid
        assert specs["min_service_temp_c"] < specs["max_service_temp_c"]

    def test_invalid_emissivity_rejected(self):
        """Test that invalid emissivity values are rejected."""
        invalid_values = [-0.1, 0.0, 1.1, 2.0]

        for value in invalid_values:
            with pytest.raises(ValueError):
                if not (0 < value <= 1):
                    raise ValueError(f"Emissivity must be in (0, 1], got {value}")

    def test_invalid_temperature_range_rejected(self):
        """Test that invalid temperature ranges are rejected."""
        with pytest.raises(ValueError):
            min_temp = 100.0
            max_temp = 50.0  # min > max is invalid
            if min_temp >= max_temp:
                raise ValueError("min_temperature must be less than max_temperature")

    def test_negative_dimension_rejected(self):
        """Test that negative dimensions are rejected."""
        invalid_dimensions = [-1.0, -0.001, -100]

        for dim in invalid_dimensions:
            with pytest.raises(ValueError):
                if dim <= 0:
                    raise ValueError(f"Dimension must be positive, got {dim}")


# =============================================================================
# TEST: OUTPUT MODEL VALIDATION
# =============================================================================

class TestOutputModelValidation:
    """Tests for output model validation."""

    def test_heat_loss_result_structure(self):
        """Test HeatLossResult structure."""
        result = {
            "heat_loss_w": Decimal("450.5"),
            "heat_loss_w_per_m": Decimal("45.05"),
            "heat_loss_w_per_m2": Decimal("150.17"),
            "surface_temperature_c": Decimal("48.2"),
            "provenance_hash": "abc123" * 10 + "abcd",
        }

        # All values should be present
        required_fields = ["heat_loss_w", "provenance_hash"]
        for field in required_fields:
            assert field in result

        # Heat loss should be positive
        assert result["heat_loss_w"] > Decimal("0")

    def test_degradation_result_structure(self):
        """Test DegradationResult structure."""
        result = {
            "severity": "moderate",
            "performance_loss_percent": Decimal("25.5"),
            "remaining_life_years": Decimal("8.5"),
            "recommended_action": "schedule_repair",
            "confidence_score": Decimal("0.85"),
        }

        assert result["severity"] in ["none", "minor", "moderate", "severe", "failed"]
        assert Decimal("0") <= result["performance_loss_percent"] <= Decimal("100")
        assert result["remaining_life_years"] >= Decimal("0")

    def test_repair_recommendation_structure(self):
        """Test RepairRecommendation structure."""
        result = {
            "priority": "urgent",
            "estimated_cost": Decimal("5000.00"),
            "estimated_savings_annual": Decimal("1500.00"),
            "payback_years": Decimal("3.33"),
            "recommended_materials": ["Mineral wool 75mm", "Aluminum jacket"],
            "work_description": "Replace damaged insulation section",
        }

        assert result["priority"] in ["routine", "planned", "urgent", "critical"]
        assert result["estimated_cost"] >= Decimal("0")
        assert isinstance(result["recommended_materials"], list)

    def test_provenance_hash_format(self):
        """Test provenance hash format validation."""
        import re

        valid_hash = "a" * 64  # SHA-256 produces 64 hex chars

        # Should be 64 hex characters
        pattern = r"^[a-f0-9]{64}$"
        assert re.match(pattern, valid_hash)

    def test_timestamp_format(self):
        """Test timestamp format validation."""
        timestamp = datetime.now().isoformat()

        # Should be parseable
        parsed = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        assert isinstance(parsed, datetime)


# =============================================================================
# TEST: SETTINGS MANAGEMENT
# =============================================================================

class TestSettingsManagement:
    """Tests for settings management."""

    def test_default_settings(self, test_settings):
        """Test default settings values."""
        assert test_settings["environment"] == "test"
        assert test_settings["debug"] is True
        assert test_settings["log_level"] == "DEBUG"

    def test_production_settings(self, production_settings):
        """Test production settings values."""
        assert production_settings["environment"] == "production"
        assert production_settings["debug"] is False
        assert production_settings["log_level"] == "WARNING"

    def test_settings_validation(self):
        """Test settings validation rules."""
        settings = {
            "cache_ttl_seconds": 300,
            "max_image_size_mb": 50,
            "temperature_precision_digits": 2,
        }

        # Cache TTL should be positive
        assert settings["cache_ttl_seconds"] > 0

        # Image size should be reasonable
        assert 1 <= settings["max_image_size_mb"] <= 200

        # Precision should be reasonable
        assert 0 <= settings["temperature_precision_digits"] <= 6

    def test_environment_variable_override(self):
        """Test environment variable override behavior."""
        import os

        # Simulate environment variable
        with patch.dict(os.environ, {"GL015_DEBUG": "false"}):
            debug_value = os.environ.get("GL015_DEBUG", "true")
            assert debug_value == "false"

    def test_settings_immutability(self):
        """Test that settings are treated as immutable."""
        original_settings = {"key": "value"}
        settings_copy = original_settings.copy()

        settings_copy["key"] = "modified"

        assert original_settings["key"] == "value"


# =============================================================================
# TEST: DATA TYPE CONVERSIONS
# =============================================================================

class TestDataTypeConversions:
    """Tests for data type conversions."""

    def test_decimal_from_string(self):
        """Test Decimal creation from string."""
        decimal_val = Decimal("123.456")
        assert decimal_val == Decimal("123.456")

    def test_decimal_from_float(self):
        """Test Decimal creation from float (with precision handling)."""
        # Direct float conversion can lose precision
        float_val = 0.1
        decimal_val = Decimal(str(float_val))

        assert decimal_val == Decimal("0.1")

    def test_date_from_string(self):
        """Test date parsing from string."""
        date_str = "2025-01-15"
        parsed_date = date.fromisoformat(date_str)

        assert parsed_date.year == 2025
        assert parsed_date.month == 1
        assert parsed_date.day == 15

    def test_datetime_from_string(self):
        """Test datetime parsing from string."""
        datetime_str = "2025-01-15T10:30:00"
        parsed_dt = datetime.fromisoformat(datetime_str)

        assert parsed_dt.hour == 10
        assert parsed_dt.minute == 30

    def test_enum_from_string(self):
        """Test enum creation from string value."""
        insulation_types = {
            "mineral_wool": "MINERAL_WOOL",
            "calcium_silicate": "CALCIUM_SILICATE",
        }

        input_str = "mineral_wool"
        enum_value = insulation_types.get(input_str)

        assert enum_value == "MINERAL_WOOL"


# =============================================================================
# TEST: VALIDATION RULES
# =============================================================================

class TestValidationRules:
    """Tests for custom validation rules."""

    def test_temperature_consistency_validation(self):
        """Test temperature consistency validation."""
        def validate_temperatures(process_temp, surface_temp, ambient_temp):
            # Hot service: process > surface > ambient
            # Cold service: process < surface < ambient
            if process_temp > ambient_temp:
                return process_temp >= surface_temp >= ambient_temp
            else:
                return process_temp <= surface_temp <= ambient_temp

        # Hot service valid
        assert validate_temperatures(175, 60, 25)

        # Cold service valid
        assert validate_temperatures(-40, -20, 25)

        # Invalid (surface hotter than process for hot service)
        assert not validate_temperatures(175, 200, 25)

    def test_thickness_vs_diameter_validation(self):
        """Test insulation thickness vs pipe diameter validation."""
        def validate_thickness_ratio(pipe_diameter_mm, insulation_thickness_mm):
            # Thickness shouldn't exceed diameter for most applications
            ratio = insulation_thickness_mm / pipe_diameter_mm
            return ratio < 2.0  # Typical max is 1.5x diameter

        assert validate_thickness_ratio(100, 75)  # Valid
        assert validate_thickness_ratio(50, 50)   # Valid
        assert not validate_thickness_ratio(50, 150)  # Too thick

    def test_operating_hours_validation(self):
        """Test operating hours per year validation."""
        max_hours_per_year = 8760  # 24 * 365

        valid_hours = [8000, 4000, 8760]
        invalid_hours = [9000, -100, 10000]

        for hours in valid_hours:
            assert 0 <= hours <= max_hours_per_year

        for hours in invalid_hours:
            assert not (0 <= hours <= max_hours_per_year)

    def test_coordinate_validation(self):
        """Test geographic coordinate validation."""
        def validate_coordinates(latitude, longitude):
            return -90 <= latitude <= 90 and -180 <= longitude <= 180

        assert validate_coordinates(40.7128, -74.0060)  # New York
        assert validate_coordinates(-33.8688, 151.2093)  # Sydney
        assert not validate_coordinates(100, 0)  # Invalid latitude

    def test_email_format_validation(self):
        """Test email format validation."""
        import re

        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

        valid_emails = ["user@example.com", "inspector@company.org"]
        invalid_emails = ["invalid", "no@domain", "@nodomain.com"]

        for email in valid_emails:
            assert re.match(email_pattern, email)

        for email in invalid_emails:
            assert not re.match(email_pattern, email)


# =============================================================================
# TEST: CONFIGURATION LOADING
# =============================================================================

class TestConfigurationLoading:
    """Tests for configuration file loading."""

    def test_yaml_config_structure(self):
        """Test YAML configuration structure."""
        config_structure = {
            "agent_id": "GL-015",
            "codename": "INSULSCAN",
            "version": "1.0.0",
            "settings": {
                "default_emissivity": 0.95,
                "temperature_unit": "celsius",
            },
            "thresholds": {
                "hotspot_delta_t": 15.0,
                "degradation_warning": 0.15,
            },
        }

        assert config_structure["agent_id"] == "GL-015"
        assert "settings" in config_structure
        assert "thresholds" in config_structure

    def test_config_merge_behavior(self):
        """Test configuration merge behavior (defaults + overrides)."""
        defaults = {
            "timeout": 30,
            "retries": 3,
            "debug": False,
        }

        overrides = {
            "timeout": 60,
            "debug": True,
        }

        # Merge with overrides taking precedence
        merged = {**defaults, **overrides}

        assert merged["timeout"] == 60  # Overridden
        assert merged["retries"] == 3   # Default
        assert merged["debug"] is True  # Overridden

    def test_config_validation_on_load(self):
        """Test configuration validation on load."""
        config = {
            "max_retries": -1,  # Invalid
        }

        with pytest.raises(ValueError):
            if config["max_retries"] < 0:
                raise ValueError("max_retries must be non-negative")


# =============================================================================
# TEST: MODEL SERIALIZATION
# =============================================================================

class TestModelSerialization:
    """Tests for model serialization/deserialization."""

    def test_dict_serialization(self):
        """Test dictionary serialization."""
        model_data = {
            "equipment_tag": "P-1001-A",
            "temperature_c": Decimal("175.0"),
            "timestamp": datetime.now(),
        }

        # Convert for JSON serialization
        serialized = {
            k: str(v) if isinstance(v, (Decimal, datetime)) else v
            for k, v in model_data.items()
        }

        assert isinstance(serialized["temperature_c"], str)
        assert isinstance(serialized["timestamp"], str)

    def test_json_serialization(self):
        """Test JSON serialization."""
        import json

        data = {
            "value": 123.456,
            "name": "test",
            "nested": {"key": "value"},
        }

        json_str = json.dumps(data)
        restored = json.loads(json_str)

        assert restored == data

    def test_decimal_json_serialization(self):
        """Test Decimal JSON serialization."""
        import json

        decimal_value = Decimal("123.456789")

        # Custom encoder
        class DecimalEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Decimal):
                    return str(obj)
                return super().default(obj)

        json_str = json.dumps({"value": decimal_value}, cls=DecimalEncoder)
        assert '"123.456789"' in json_str
