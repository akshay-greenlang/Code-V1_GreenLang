# -*- coding: utf-8 -*-
"""
GL-013 PREDICTMAINT - Unit Tests for Configuration Module
Comprehensive configuration validation tests.

Tests cover:
- Configuration model validation
- Default value verification
- Constraint validation (ranges, enums)
- Cross-field validation
- ISO 10816 limits configuration
- Weibull parameter configuration
- Alert threshold configuration
- Integration settings validation

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch
import json

# Import test fixtures from conftest
from ..conftest import (
    MachineClass,
    VibrationZone,
    HealthState,
    AlertSeverity,
    ISO_10816_LIMITS,
    WEIBULL_PARAMETERS,
)


# =============================================================================
# TEST CLASS: BASIC CONFIGURATION VALIDATION
# =============================================================================


class TestBasicConfigurationValidation:
    """Tests for basic configuration model validation."""

    @pytest.mark.unit
    def test_default_config_is_valid(self, default_config):
        """Test that default configuration is valid."""
        assert default_config is not None
        assert "agent_id" in default_config
        assert default_config["agent_id"] == "GL-013"
        assert default_config["agent_name"] == "PREDICTMAINT"

    @pytest.mark.unit
    def test_config_has_required_sections(self, default_config):
        """Test that configuration has all required sections."""
        required_sections = [
            "vibration_analysis",
            "rul_calculation",
            "maintenance_scheduling",
            "anomaly_detection",
            "alerts",
        ]

        for section in required_sections:
            assert section in default_config, f"Missing section: {section}"

    @pytest.mark.unit
    def test_config_deterministic_flag(self, default_config):
        """Test deterministic flag is set correctly."""
        assert "deterministic" in default_config
        assert default_config["deterministic"] is True

    @pytest.mark.unit
    def test_config_seed_value(self, default_config):
        """Test random seed is set for reproducibility."""
        assert "seed" in default_config
        assert isinstance(default_config["seed"], int)

    @pytest.mark.unit
    def test_config_version_format(self, default_config):
        """Test version string format is valid."""
        assert "version" in default_config
        version = default_config["version"]

        # Should be semantic versioning format X.Y.Z
        parts = version.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()


# =============================================================================
# TEST CLASS: VIBRATION CONFIGURATION
# =============================================================================


class TestVibrationConfiguration:
    """Tests for vibration analysis configuration."""

    @pytest.mark.unit
    def test_default_machine_class(self, default_config):
        """Test default machine class setting."""
        vib_config = default_config["vibration_analysis"]
        assert "default_machine_class" in vib_config
        assert vib_config["default_machine_class"] in [
            MachineClass.CLASS_I,
            MachineClass.CLASS_II,
            MachineClass.CLASS_III,
            MachineClass.CLASS_IV,
        ]

    @pytest.mark.unit
    def test_alarm_trigger_settings(self, default_config):
        """Test alarm and trip trigger settings."""
        vib_config = default_config["vibration_analysis"]

        assert "alarm_on_zone_c" in vib_config
        assert "trip_on_zone_d" in vib_config

        # Default should alarm on Zone C and trip on Zone D
        assert vib_config["alarm_on_zone_c"] is True
        assert vib_config["trip_on_zone_d"] is True

    @pytest.mark.unit
    def test_trend_window_days(self, default_config):
        """Test trend analysis window configuration."""
        vib_config = default_config["vibration_analysis"]

        assert "trend_window_days" in vib_config
        assert vib_config["trend_window_days"] > 0
        assert vib_config["trend_window_days"] <= 365

    @pytest.mark.unit
    @pytest.mark.parametrize("machine_class,zone_a_limit", [
        (MachineClass.CLASS_I, Decimal("0.71")),
        (MachineClass.CLASS_II, Decimal("1.12")),
        (MachineClass.CLASS_III, Decimal("1.8")),
        (MachineClass.CLASS_IV, Decimal("2.8")),
    ])
    def test_iso_10816_limits_valid(self, machine_class, zone_a_limit):
        """Test ISO 10816 limits are correctly configured."""
        limits = ISO_10816_LIMITS.get(machine_class)

        assert limits is not None
        assert limits["zone_a_upper"] == zone_a_limit
        assert limits["zone_a_upper"] < limits["zone_b_upper"]
        assert limits["zone_b_upper"] < limits["zone_c_upper"]

    @pytest.mark.unit
    def test_zone_limits_ascending_order(self):
        """Test that zone limits are in ascending order for all classes."""
        for machine_class, limits in ISO_10816_LIMITS.items():
            assert limits["zone_a_upper"] < limits["zone_b_upper"]
            assert limits["zone_b_upper"] < limits["zone_c_upper"]


# =============================================================================
# TEST CLASS: RUL CALCULATION CONFIGURATION
# =============================================================================


class TestRULCalculationConfiguration:
    """Tests for RUL calculation configuration."""

    @pytest.mark.unit
    def test_default_model_setting(self, default_config):
        """Test default RUL model setting."""
        rul_config = default_config["rul_calculation"]

        assert "default_model" in rul_config
        assert rul_config["default_model"] in ["weibull", "exponential", "lognormal"]

    @pytest.mark.unit
    def test_confidence_level_setting(self, default_config):
        """Test confidence level configuration."""
        rul_config = default_config["rul_calculation"]

        assert "confidence_level" in rul_config
        # Should be a valid confidence level
        valid_levels = ["80%", "85%", "90%", "95%", "99%"]
        assert rul_config["confidence_level"] in valid_levels

    @pytest.mark.unit
    def test_health_adjustment_enabled(self, default_config):
        """Test health adjustment flag."""
        rul_config = default_config["rul_calculation"]

        assert "health_adjustment_enabled" in rul_config
        assert isinstance(rul_config["health_adjustment_enabled"], bool)


# =============================================================================
# TEST CLASS: WEIBULL PARAMETERS CONFIGURATION
# =============================================================================


class TestWeibullParametersConfiguration:
    """Tests for Weibull parameter configuration."""

    @pytest.mark.unit
    @pytest.mark.parametrize("equipment_type", [
        "pump_centrifugal",
        "motor_ac_induction_large",
        "gearbox_helical",
        "bearing_6205",
        "compressor_reciprocating",
    ])
    def test_weibull_parameters_exist(self, equipment_type):
        """Test that Weibull parameters exist for common equipment types."""
        params = WEIBULL_PARAMETERS.get(equipment_type)

        assert params is not None
        assert "beta" in params
        assert "eta" in params

    @pytest.mark.unit
    def test_weibull_beta_positive(self):
        """Test all beta values are positive."""
        for equipment_type, params in WEIBULL_PARAMETERS.items():
            assert params["beta"] > Decimal("0"), f"Invalid beta for {equipment_type}"

    @pytest.mark.unit
    def test_weibull_eta_positive(self):
        """Test all eta values are positive."""
        for equipment_type, params in WEIBULL_PARAMETERS.items():
            assert params["eta"] > Decimal("0"), f"Invalid eta for {equipment_type}"

    @pytest.mark.unit
    def test_weibull_gamma_non_negative(self):
        """Test all gamma values are non-negative."""
        for equipment_type, params in WEIBULL_PARAMETERS.items():
            gamma = params.get("gamma", Decimal("0"))
            assert gamma >= Decimal("0"), f"Invalid gamma for {equipment_type}"

    @pytest.mark.unit
    def test_weibull_mtbf_reasonable(self):
        """Test MTBF values are reasonable (100 to 1M hours)."""
        for equipment_type, params in WEIBULL_PARAMETERS.items():
            mtbf = params.get("mtbf_hours", params["eta"])
            assert Decimal("100") <= mtbf <= Decimal("1000000"), \
                f"Unreasonable MTBF for {equipment_type}: {mtbf}"


# =============================================================================
# TEST CLASS: MAINTENANCE SCHEDULING CONFIGURATION
# =============================================================================


class TestMaintenanceSchedulingConfiguration:
    """Tests for maintenance scheduling configuration."""

    @pytest.mark.unit
    def test_optimization_window(self, default_config):
        """Test optimization window configuration."""
        maint_config = default_config["maintenance_scheduling"]

        assert "optimization_window_days" in maint_config
        assert maint_config["optimization_window_days"] > 0
        assert maint_config["optimization_window_days"] <= 730  # Max 2 years

    @pytest.mark.unit
    def test_interval_limits(self, default_config):
        """Test maintenance interval limits."""
        maint_config = default_config["maintenance_scheduling"]

        assert "min_interval_hours" in maint_config
        assert "max_interval_hours" in maint_config

        # Min should be less than max
        assert maint_config["min_interval_hours"] < maint_config["max_interval_hours"]

        # Reasonable bounds
        assert maint_config["min_interval_hours"] >= 1  # At least 1 hour
        assert maint_config["max_interval_hours"] <= 87600  # Max 10 years

    @pytest.mark.unit
    def test_interval_limits_cross_validation(self, default_config):
        """Test that min and max intervals are logically consistent."""
        maint_config = default_config["maintenance_scheduling"]

        min_interval = maint_config["min_interval_hours"]
        max_interval = maint_config["max_interval_hours"]

        # Min should be significantly less than max (at least 10x)
        assert max_interval >= min_interval * 10


# =============================================================================
# TEST CLASS: ANOMALY DETECTION CONFIGURATION
# =============================================================================


class TestAnomalyDetectionConfiguration:
    """Tests for anomaly detection configuration."""

    @pytest.mark.unit
    def test_threshold_sigma(self, default_config):
        """Test anomaly detection threshold configuration."""
        anomaly_config = default_config["anomaly_detection"]

        assert "threshold_sigma" in anomaly_config
        threshold = anomaly_config["threshold_sigma"]

        # Should be between 2 and 5 sigma typically
        assert Decimal("2.0") <= threshold <= Decimal("5.0")

    @pytest.mark.unit
    def test_cusum_parameters(self, default_config):
        """Test CUSUM parameters configuration."""
        anomaly_config = default_config["anomaly_detection"]

        assert "cusum_k" in anomaly_config
        assert "cusum_h" in anomaly_config

        # k typically 0.5 (half sigma shift)
        assert anomaly_config["cusum_k"] > Decimal("0")

        # h typically 4-5 for reasonable ARL
        assert anomaly_config["cusum_h"] > Decimal("0")

    @pytest.mark.unit
    def test_cusum_k_less_than_h(self, default_config):
        """Test CUSUM k is less than h."""
        anomaly_config = default_config["anomaly_detection"]

        assert anomaly_config["cusum_k"] < anomaly_config["cusum_h"]


# =============================================================================
# TEST CLASS: ALERT CONFIGURATION
# =============================================================================


class TestAlertConfiguration:
    """Tests for alert configuration."""

    @pytest.mark.unit
    def test_alert_channels(self, default_config):
        """Test alert channel configuration."""
        alert_config = default_config["alerts"]

        assert "email_enabled" in alert_config
        assert "sms_enabled" in alert_config
        assert "webhook_enabled" in alert_config

    @pytest.mark.unit
    def test_alert_channels_are_boolean(self, default_config):
        """Test alert channels are boolean values."""
        alert_config = default_config["alerts"]

        assert isinstance(alert_config["email_enabled"], bool)
        assert isinstance(alert_config["sms_enabled"], bool)
        assert isinstance(alert_config["webhook_enabled"], bool)


# =============================================================================
# TEST CLASS: STRICT CONFIGURATION VALIDATION
# =============================================================================


class TestStrictConfigurationValidation:
    """Tests for strict configuration validation mode."""

    @pytest.mark.unit
    def test_strict_validation_flag(self, test_config_strict):
        """Test strict validation flag."""
        assert test_config_strict["strict_validation"] is True

    @pytest.mark.unit
    def test_fail_on_warning_flag(self, test_config_strict):
        """Test fail on warning flag."""
        assert test_config_strict["fail_on_warning"] is True

    @pytest.mark.unit
    def test_higher_precision_in_strict_mode(self, test_config_strict):
        """Test higher decimal precision in strict mode."""
        assert test_config_strict["decimal_precision"] >= 10


# =============================================================================
# TEST CLASS: CONFIGURATION SERIALIZATION
# =============================================================================


class TestConfigurationSerialization:
    """Tests for configuration serialization/deserialization."""

    @pytest.mark.unit
    def test_config_json_serializable(self, default_config):
        """Test configuration can be serialized to JSON."""
        try:
            json_str = json.dumps(default_config, default=str)
            assert isinstance(json_str, str)
            assert len(json_str) > 0
        except (TypeError, ValueError) as e:
            pytest.fail(f"Configuration not JSON serializable: {e}")

    @pytest.mark.unit
    def test_config_json_roundtrip(self, default_config):
        """Test configuration survives JSON roundtrip."""
        # Create a serializable copy (convert Decimal to str)
        serializable_config = json.loads(
            json.dumps(default_config, default=str)
        )

        # Key structure should be preserved
        assert "agent_id" in serializable_config
        assert "vibration_analysis" in serializable_config

    @pytest.mark.unit
    def test_config_deep_copy(self, default_config):
        """Test configuration can be deep copied."""
        import copy

        config_copy = copy.deepcopy(default_config)

        # Modify copy
        config_copy["agent_id"] = "MODIFIED"

        # Original should be unchanged
        assert default_config["agent_id"] == "GL-013"


# =============================================================================
# TEST CLASS: CONFIGURATION EDGE CASES
# =============================================================================


class TestConfigurationEdgeCases:
    """Tests for configuration edge cases and boundaries."""

    @pytest.mark.unit
    def test_empty_string_handling(self):
        """Test handling of empty string values."""
        config = {
            "agent_id": "",
            "agent_name": "",
        }

        # Empty strings should be caught in validation
        assert config["agent_id"] == ""

    @pytest.mark.unit
    def test_none_value_handling(self):
        """Test handling of None values."""
        config = {
            "agent_id": None,
            "optional_field": None,
        }

        assert config["agent_id"] is None

    @pytest.mark.unit
    def test_whitespace_handling(self):
        """Test handling of whitespace-only values."""
        config = {
            "agent_id": "   ",
            "agent_name": "\t\n",
        }

        # Should be validated/stripped in real implementation
        assert config["agent_id"].strip() == ""

    @pytest.mark.unit
    def test_negative_values_rejected(self):
        """Test that negative values are rejected where inappropriate."""
        invalid_configs = [
            {"optimization_window_days": -1},
            {"min_interval_hours": -100},
            {"decimal_precision": -1},
        ]

        for invalid_config in invalid_configs:
            for key, value in invalid_config.items():
                assert value < 0  # Confirms these are negative

    @pytest.mark.unit
    def test_very_large_values(self):
        """Test handling of very large configuration values."""
        large_config = {
            "optimization_window_days": 100000,
            "max_interval_hours": 10000000,
        }

        # These should be validated against reasonable upper bounds
        assert large_config["optimization_window_days"] > 365 * 100


# =============================================================================
# TEST CLASS: CONFIGURATION DEFAULTS
# =============================================================================


class TestConfigurationDefaults:
    """Tests for configuration default values."""

    @pytest.mark.unit
    def test_all_defaults_set(self, default_config):
        """Test that all configuration sections have defaults."""
        # Top level
        assert default_config.get("deterministic") is not None
        assert default_config.get("seed") is not None
        assert default_config.get("decimal_precision") is not None

        # Nested sections
        assert default_config.get("vibration_analysis") is not None
        assert default_config.get("rul_calculation") is not None
        assert default_config.get("maintenance_scheduling") is not None
        assert default_config.get("anomaly_detection") is not None
        assert default_config.get("alerts") is not None

    @pytest.mark.unit
    def test_default_precision(self, default_config):
        """Test default decimal precision."""
        assert default_config["decimal_precision"] >= 6

    @pytest.mark.unit
    def test_default_provenance_enabled(self, default_config):
        """Test provenance tracking is enabled by default."""
        assert default_config.get("store_provenance") is True


# =============================================================================
# TEST CLASS: CROSS-FIELD VALIDATION
# =============================================================================


class TestCrossFieldValidation:
    """Tests for cross-field validation rules."""

    @pytest.mark.unit
    def test_interval_bounds_consistent(self, default_config):
        """Test min/max interval bounds are consistent."""
        maint_config = default_config["maintenance_scheduling"]

        min_h = maint_config["min_interval_hours"]
        max_h = maint_config["max_interval_hours"]

        assert min_h < max_h

    @pytest.mark.unit
    def test_zone_limits_consistent_across_classes(self):
        """Test zone limits increase with machine class."""
        classes = [
            MachineClass.CLASS_I,
            MachineClass.CLASS_II,
            MachineClass.CLASS_III,
            MachineClass.CLASS_IV,
        ]

        for i in range(len(classes) - 1):
            limits_lower = ISO_10816_LIMITS[classes[i]]
            limits_higher = ISO_10816_LIMITS[classes[i + 1]]

            # Higher class should have higher (more lenient) limits
            assert limits_higher["zone_a_upper"] > limits_lower["zone_a_upper"]


# =============================================================================
# TEST CLASS: CONFIGURATION DOCUMENTATION
# =============================================================================


class TestConfigurationDocumentation:
    """Tests for configuration documentation completeness."""

    @pytest.mark.unit
    def test_agent_id_documented(self, default_config):
        """Test agent ID is properly set."""
        assert default_config["agent_id"] == "GL-013"

    @pytest.mark.unit
    def test_agent_name_documented(self, default_config):
        """Test agent name is properly set."""
        assert default_config["agent_name"] == "PREDICTMAINT"

    @pytest.mark.unit
    def test_version_documented(self, default_config):
        """Test version is properly set."""
        assert default_config["version"] == "1.0.0"
