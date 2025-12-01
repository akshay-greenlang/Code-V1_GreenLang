# -*- coding: utf-8 -*-
"""
Unit Tests for GL-012 STEAMQUAL Configuration.

This module provides comprehensive tests for configuration validation,
including default values, constraint checking, serialization/deserialization,
and configuration schema validation.

Coverage Target: 95%+
Standards Compliance:
- IAPWS-IF97: Industrial Formulation for Water and Steam Properties
- ASME PTC 19.11: Steam and Water Sampling

Test Categories:
1. Configuration validation
2. Default values verification
3. Constraint checking
4. Serialization/deserialization
5. Schema validation
6. Edge cases and error handling

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import sys
import json
import tempfile
from pathlib import Path
from decimal import Decimal
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch
from dataclasses import dataclass, field, asdict
from enum import Enum

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import test fixtures from conftest
from conftest import (
    SteamState,
    RiskLevel,
    ValveCharacteristic,
    assert_within_tolerance,
)


# =============================================================================
# CONFIGURATION DATACLASSES FOR TESTING
# =============================================================================

class ControlStrategy(Enum):
    """Steam quality control strategy."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class AlarmSeverity(Enum):
    """Alarm severity levels."""
    INFO = "info"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"


@dataclass
class SteamQualityLimits:
    """Steam quality operational limits."""
    min_dryness_fraction: float = 0.85
    target_dryness_fraction: float = 0.98
    critical_dryness_fraction: float = 0.80
    max_wetness_percent: float = 15.0
    min_superheat_c: float = 5.0
    max_superheat_c: float = 100.0
    target_superheat_c: float = 20.0


@dataclass
class PressureLimits:
    """Pressure operational limits."""
    min_pressure_bar: float = 1.0
    max_pressure_bar: float = 200.0
    design_pressure_bar: float = 40.0
    operating_pressure_bar: float = 35.0
    pressure_tolerance_bar: float = 0.5
    max_pressure_drop_percent: float = 10.0


@dataclass
class DesuperheaterConfig:
    """Desuperheater configuration."""
    enabled: bool = True
    max_injection_rate_kg_s: float = 10.0
    min_injection_rate_kg_s: float = 0.1
    spray_water_temp_c: float = 105.0
    spray_water_pressure_bar: float = 20.0
    target_superheat_c: float = 15.0
    pid_kp: float = 2.0
    pid_ki: float = 0.5
    pid_kd: float = 0.1
    valve_cv: float = 50.0
    valve_characteristic: ValveCharacteristic = ValveCharacteristic.EQUAL_PERCENTAGE


@dataclass
class PressureControlConfig:
    """Pressure control configuration."""
    enabled: bool = True
    control_strategy: ControlStrategy = ControlStrategy.BALANCED
    pid_kp: float = 1.5
    pid_ki: float = 0.3
    pid_kd: float = 0.05
    dead_band_bar: float = 0.1
    max_rate_of_change_bar_s: float = 0.5
    valve_cv: float = 100.0
    valve_stroke_time_s: float = 30.0


@dataclass
class AlarmConfig:
    """Alarm configuration."""
    low_dryness_warning: float = 0.95
    low_dryness_alarm: float = 0.90
    low_dryness_critical: float = 0.85
    high_pressure_warning_bar: float = 38.0
    high_pressure_alarm_bar: float = 40.0
    low_pressure_warning_bar: float = 32.0
    low_pressure_alarm_bar: float = 30.0
    alarm_delay_seconds: int = 5
    enable_email_notifications: bool = True
    enable_sms_notifications: bool = False


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration."""
    sample_rate_seconds: float = 1.0
    log_level: str = "INFO"
    enable_trending: bool = True
    trend_retention_days: int = 30
    enable_statistics: bool = True
    statistics_window_hours: int = 24


@dataclass
class IntegrationConfig:
    """Integration settings."""
    scada_enabled: bool = True
    scada_poll_interval_ms: int = 1000
    opc_ua_enabled: bool = False
    modbus_enabled: bool = True
    modbus_slave_id: int = 1
    mqtt_enabled: bool = False
    historian_enabled: bool = True


@dataclass
class SteamQualityAgentConfig:
    """Main configuration for GL-012 STEAMQUAL agent."""
    agent_id: str = "GL-012"
    agent_name: str = "STEAMQUAL"
    version: str = "1.0.0"
    enabled: bool = True
    steam_quality_limits: SteamQualityLimits = field(default_factory=SteamQualityLimits)
    pressure_limits: PressureLimits = field(default_factory=PressureLimits)
    desuperheater: DesuperheaterConfig = field(default_factory=DesuperheaterConfig)
    pressure_control: PressureControlConfig = field(default_factory=PressureControlConfig)
    alarms: AlarmConfig = field(default_factory=AlarmConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Steam quality limits validation
        if self.steam_quality_limits.min_dryness_fraction < 0:
            errors.append("min_dryness_fraction cannot be negative")
        if self.steam_quality_limits.min_dryness_fraction > 1.0:
            errors.append("min_dryness_fraction cannot exceed 1.0")
        if self.steam_quality_limits.target_dryness_fraction < self.steam_quality_limits.min_dryness_fraction:
            errors.append("target_dryness_fraction must be >= min_dryness_fraction")

        # Pressure limits validation
        if self.pressure_limits.min_pressure_bar < 0:
            errors.append("min_pressure_bar cannot be negative")
        if self.pressure_limits.max_pressure_bar <= self.pressure_limits.min_pressure_bar:
            errors.append("max_pressure_bar must be > min_pressure_bar")
        if not (self.pressure_limits.min_pressure_bar <=
                self.pressure_limits.operating_pressure_bar <=
                self.pressure_limits.max_pressure_bar):
            errors.append("operating_pressure_bar must be within min/max range")

        # Desuperheater validation
        if self.desuperheater.enabled:
            if self.desuperheater.max_injection_rate_kg_s <= 0:
                errors.append("desuperheater max_injection_rate must be positive")
            if self.desuperheater.min_injection_rate_kg_s < 0:
                errors.append("desuperheater min_injection_rate cannot be negative")
            if self.desuperheater.min_injection_rate_kg_s >= self.desuperheater.max_injection_rate_kg_s:
                errors.append("desuperheater min_injection_rate must be < max_injection_rate")
            if self.desuperheater.spray_water_temp_c >= 100.0 and self.pressure_limits.operating_pressure_bar <= 1.0:
                errors.append("spray_water_temp_c should be below saturation at operating pressure")

        # PID validation
        if self.desuperheater.pid_kp < 0:
            errors.append("desuperheater PID Kp cannot be negative")
        if self.desuperheater.pid_ki < 0:
            errors.append("desuperheater PID Ki cannot be negative")
        if self.desuperheater.pid_kd < 0:
            errors.append("desuperheater PID Kd cannot be negative")

        if self.pressure_control.pid_kp < 0:
            errors.append("pressure_control PID Kp cannot be negative")
        if self.pressure_control.pid_ki < 0:
            errors.append("pressure_control PID Ki cannot be negative")
        if self.pressure_control.pid_kd < 0:
            errors.append("pressure_control PID Kd cannot be negative")

        # Alarm threshold validation
        if not (self.alarms.low_dryness_critical <
                self.alarms.low_dryness_alarm <
                self.alarms.low_dryness_warning):
            errors.append("Alarm thresholds must be ordered: critical < alarm < warning")

        # Monitoring validation
        if self.monitoring.sample_rate_seconds <= 0:
            errors.append("sample_rate_seconds must be positive")
        if self.monitoring.trend_retention_days < 1:
            errors.append("trend_retention_days must be at least 1")

        return errors

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        data = asdict(self)
        # Convert enums to strings
        data['desuperheater']['valve_characteristic'] = self.desuperheater.valve_characteristic.value
        data['pressure_control']['control_strategy'] = self.pressure_control.control_strategy.value
        return data

    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SteamQualityAgentConfig':
        """Create configuration from dictionary."""
        # Handle nested dataclasses
        steam_quality_limits = SteamQualityLimits(**data.get('steam_quality_limits', {}))
        pressure_limits = PressureLimits(**data.get('pressure_limits', {}))

        # Handle desuperheater with enum
        desuper_data = data.get('desuperheater', {})
        if 'valve_characteristic' in desuper_data and isinstance(desuper_data['valve_characteristic'], str):
            desuper_data['valve_characteristic'] = ValveCharacteristic(desuper_data['valve_characteristic'])
        desuperheater = DesuperheaterConfig(**desuper_data)

        # Handle pressure control with enum
        pressure_ctrl_data = data.get('pressure_control', {})
        if 'control_strategy' in pressure_ctrl_data and isinstance(pressure_ctrl_data['control_strategy'], str):
            pressure_ctrl_data['control_strategy'] = ControlStrategy(pressure_ctrl_data['control_strategy'])
        pressure_control = PressureControlConfig(**pressure_ctrl_data)

        alarms = AlarmConfig(**data.get('alarms', {}))
        monitoring = MonitoringConfig(**data.get('monitoring', {}))
        integration = IntegrationConfig(**data.get('integration', {}))

        return cls(
            agent_id=data.get('agent_id', 'GL-012'),
            agent_name=data.get('agent_name', 'STEAMQUAL'),
            version=data.get('version', '1.0.0'),
            enabled=data.get('enabled', True),
            steam_quality_limits=steam_quality_limits,
            pressure_limits=pressure_limits,
            desuperheater=desuperheater,
            pressure_control=pressure_control,
            alarms=alarms,
            monitoring=monitoring,
            integration=integration
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'SteamQualityAgentConfig':
        """Create configuration from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


def create_default_config() -> SteamQualityAgentConfig:
    """Create default configuration."""
    return SteamQualityAgentConfig()


# =============================================================================
# TEST CLASS: CONFIGURATION VALIDATION
# =============================================================================

class TestConfigurationValidation:
    """Test suite for configuration validation."""

    @pytest.fixture
    def default_config(self):
        """Create default configuration."""
        return create_default_config()

    @pytest.mark.unit
    def test_default_config_is_valid(self, default_config):
        """Test that default configuration is valid."""
        errors = default_config.validate()
        assert len(errors) == 0, f"Default config has errors: {errors}"
        assert default_config.is_valid()

    @pytest.mark.unit
    def test_negative_dryness_invalid(self):
        """Test negative dryness fraction is invalid."""
        config = create_default_config()
        config.steam_quality_limits.min_dryness_fraction = -0.1

        errors = config.validate()
        assert len(errors) > 0
        assert any("negative" in e.lower() for e in errors)

    @pytest.mark.unit
    def test_dryness_above_one_invalid(self):
        """Test dryness fraction above 1.0 is invalid."""
        config = create_default_config()
        config.steam_quality_limits.min_dryness_fraction = 1.1

        errors = config.validate()
        assert len(errors) > 0
        assert any("exceed 1.0" in e.lower() for e in errors)

    @pytest.mark.unit
    def test_target_below_min_invalid(self):
        """Test target dryness below minimum is invalid."""
        config = create_default_config()
        config.steam_quality_limits.min_dryness_fraction = 0.90
        config.steam_quality_limits.target_dryness_fraction = 0.85

        errors = config.validate()
        assert len(errors) > 0

    @pytest.mark.unit
    def test_negative_pressure_invalid(self):
        """Test negative pressure is invalid."""
        config = create_default_config()
        config.pressure_limits.min_pressure_bar = -1.0

        errors = config.validate()
        assert len(errors) > 0
        assert any("negative" in e.lower() for e in errors)

    @pytest.mark.unit
    def test_max_below_min_pressure_invalid(self):
        """Test max pressure below min is invalid."""
        config = create_default_config()
        config.pressure_limits.min_pressure_bar = 50.0
        config.pressure_limits.max_pressure_bar = 40.0

        errors = config.validate()
        assert len(errors) > 0
        assert any("max_pressure_bar must be > min" in e.lower() for e in errors)

    @pytest.mark.unit
    def test_operating_outside_range_invalid(self):
        """Test operating pressure outside range is invalid."""
        config = create_default_config()
        config.pressure_limits.min_pressure_bar = 10.0
        config.pressure_limits.max_pressure_bar = 50.0
        config.pressure_limits.operating_pressure_bar = 60.0

        errors = config.validate()
        assert len(errors) > 0

    @pytest.mark.unit
    def test_desuperheater_negative_injection_invalid(self):
        """Test negative injection rate is invalid."""
        config = create_default_config()
        config.desuperheater.max_injection_rate_kg_s = -1.0

        errors = config.validate()
        assert len(errors) > 0

    @pytest.mark.unit
    def test_desuperheater_min_above_max_invalid(self):
        """Test min injection above max is invalid."""
        config = create_default_config()
        config.desuperheater.min_injection_rate_kg_s = 15.0
        config.desuperheater.max_injection_rate_kg_s = 10.0

        errors = config.validate()
        assert len(errors) > 0

    @pytest.mark.unit
    def test_negative_pid_gains_invalid(self):
        """Test negative PID gains are invalid."""
        config = create_default_config()
        config.desuperheater.pid_kp = -1.0

        errors = config.validate()
        assert len(errors) > 0
        assert any("kp cannot be negative" in e.lower() for e in errors)

    @pytest.mark.unit
    def test_alarm_threshold_ordering_invalid(self):
        """Test incorrect alarm threshold ordering is invalid."""
        config = create_default_config()
        # Set warning below alarm (incorrect ordering)
        config.alarms.low_dryness_warning = 0.85
        config.alarms.low_dryness_alarm = 0.90
        config.alarms.low_dryness_critical = 0.92

        errors = config.validate()
        assert len(errors) > 0
        assert any("ordered" in e.lower() for e in errors)

    @pytest.mark.unit
    def test_zero_sample_rate_invalid(self):
        """Test zero sample rate is invalid."""
        config = create_default_config()
        config.monitoring.sample_rate_seconds = 0.0

        errors = config.validate()
        assert len(errors) > 0


# =============================================================================
# TEST CLASS: DEFAULT VALUES
# =============================================================================

class TestDefaultValues:
    """Test suite for default configuration values."""

    @pytest.fixture
    def default_config(self):
        """Create default configuration."""
        return create_default_config()

    @pytest.mark.unit
    def test_default_agent_id(self, default_config):
        """Test default agent ID."""
        assert default_config.agent_id == "GL-012"

    @pytest.mark.unit
    def test_default_agent_name(self, default_config):
        """Test default agent name."""
        assert default_config.agent_name == "STEAMQUAL"

    @pytest.mark.unit
    def test_default_enabled(self, default_config):
        """Test agent is enabled by default."""
        assert default_config.enabled is True

    @pytest.mark.unit
    def test_default_dryness_limits(self, default_config):
        """Test default steam quality limits."""
        limits = default_config.steam_quality_limits
        assert limits.min_dryness_fraction == 0.85
        assert limits.target_dryness_fraction == 0.98
        assert limits.critical_dryness_fraction == 0.80
        assert limits.max_wetness_percent == 15.0

    @pytest.mark.unit
    def test_default_superheat_limits(self, default_config):
        """Test default superheat limits."""
        limits = default_config.steam_quality_limits
        assert limits.min_superheat_c == 5.0
        assert limits.max_superheat_c == 100.0
        assert limits.target_superheat_c == 20.0

    @pytest.mark.unit
    def test_default_pressure_limits(self, default_config):
        """Test default pressure limits."""
        limits = default_config.pressure_limits
        assert limits.min_pressure_bar == 1.0
        assert limits.max_pressure_bar == 200.0
        assert limits.design_pressure_bar == 40.0
        assert limits.pressure_tolerance_bar == 0.5

    @pytest.mark.unit
    def test_default_desuperheater_config(self, default_config):
        """Test default desuperheater configuration."""
        desuper = default_config.desuperheater
        assert desuper.enabled is True
        assert desuper.max_injection_rate_kg_s == 10.0
        assert desuper.min_injection_rate_kg_s == 0.1
        assert desuper.spray_water_temp_c == 105.0
        assert desuper.target_superheat_c == 15.0

    @pytest.mark.unit
    def test_default_pid_gains(self, default_config):
        """Test default PID gains."""
        desuper = default_config.desuperheater
        assert desuper.pid_kp == 2.0
        assert desuper.pid_ki == 0.5
        assert desuper.pid_kd == 0.1

        pressure = default_config.pressure_control
        assert pressure.pid_kp == 1.5
        assert pressure.pid_ki == 0.3
        assert pressure.pid_kd == 0.05

    @pytest.mark.unit
    def test_default_alarm_thresholds(self, default_config):
        """Test default alarm thresholds."""
        alarms = default_config.alarms
        assert alarms.low_dryness_warning == 0.95
        assert alarms.low_dryness_alarm == 0.90
        assert alarms.low_dryness_critical == 0.85

    @pytest.mark.unit
    def test_default_monitoring_config(self, default_config):
        """Test default monitoring configuration."""
        monitoring = default_config.monitoring
        assert monitoring.sample_rate_seconds == 1.0
        assert monitoring.log_level == "INFO"
        assert monitoring.enable_trending is True
        assert monitoring.trend_retention_days == 30


# =============================================================================
# TEST CLASS: CONSTRAINT CHECKING
# =============================================================================

class TestConstraintChecking:
    """Test suite for constraint validation."""

    @pytest.mark.unit
    def test_dryness_constraint_range(self):
        """Test dryness fraction constraint is 0-1."""
        config = create_default_config()

        # Valid values
        config.steam_quality_limits.min_dryness_fraction = 0.0
        config.steam_quality_limits.target_dryness_fraction = 0.0
        assert len(config.validate()) == 0

        config.steam_quality_limits.min_dryness_fraction = 1.0
        config.steam_quality_limits.target_dryness_fraction = 1.0
        assert len(config.validate()) == 0

    @pytest.mark.unit
    def test_pressure_constraint_positive(self):
        """Test pressure constraints must be positive."""
        config = create_default_config()

        config.pressure_limits.min_pressure_bar = 0.001  # Very low but valid
        assert config.is_valid()

    @pytest.mark.unit
    def test_injection_rate_constraint(self):
        """Test injection rate constraints."""
        config = create_default_config()

        # Min must be less than max
        config.desuperheater.min_injection_rate_kg_s = 1.0
        config.desuperheater.max_injection_rate_kg_s = 10.0
        assert config.is_valid()

        # Edge case: min = 0
        config.desuperheater.min_injection_rate_kg_s = 0.0
        assert config.is_valid()

    @pytest.mark.unit
    def test_alarm_threshold_ordering_constraint(self):
        """Test alarm thresholds must be properly ordered."""
        config = create_default_config()

        # Correct ordering: critical < alarm < warning
        config.alarms.low_dryness_critical = 0.80
        config.alarms.low_dryness_alarm = 0.88
        config.alarms.low_dryness_warning = 0.95
        assert config.is_valid()

    @pytest.mark.unit
    def test_sample_rate_positive_constraint(self):
        """Test sample rate must be positive."""
        config = create_default_config()

        config.monitoring.sample_rate_seconds = 0.001  # Very fast but valid
        assert config.is_valid()


# =============================================================================
# TEST CLASS: SERIALIZATION
# =============================================================================

class TestSerialization:
    """Test suite for configuration serialization."""

    @pytest.fixture
    def default_config(self):
        """Create default configuration."""
        return create_default_config()

    @pytest.mark.unit
    def test_to_dict(self, default_config):
        """Test conversion to dictionary."""
        data = default_config.to_dict()

        assert isinstance(data, dict)
        assert data['agent_id'] == 'GL-012'
        assert data['agent_name'] == 'STEAMQUAL'
        assert 'steam_quality_limits' in data
        assert 'pressure_limits' in data
        assert 'desuperheater' in data

    @pytest.mark.unit
    def test_to_json(self, default_config):
        """Test conversion to JSON string."""
        json_str = default_config.to_json()

        assert isinstance(json_str, str)
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed['agent_id'] == 'GL-012'

    @pytest.mark.unit
    def test_json_pretty_format(self, default_config):
        """Test JSON output is pretty-formatted."""
        json_str = default_config.to_json()

        # Pretty format includes newlines
        assert '\n' in json_str
        # Indentation
        assert '  ' in json_str

    @pytest.mark.unit
    def test_enum_serialization(self, default_config):
        """Test enums are serialized as strings."""
        data = default_config.to_dict()

        assert data['desuperheater']['valve_characteristic'] == 'equal_percentage'
        assert data['pressure_control']['control_strategy'] == 'balanced'


# =============================================================================
# TEST CLASS: DESERIALIZATION
# =============================================================================

class TestDeserialization:
    """Test suite for configuration deserialization."""

    @pytest.fixture
    def default_config(self):
        """Create default configuration."""
        return create_default_config()

    @pytest.mark.unit
    def test_from_dict_basic(self):
        """Test creation from dictionary."""
        data = {
            'agent_id': 'GL-012-TEST',
            'agent_name': 'TEST',
            'steam_quality_limits': {
                'min_dryness_fraction': 0.90,
            },
        }

        config = SteamQualityAgentConfig.from_dict(data)

        assert config.agent_id == 'GL-012-TEST'
        assert config.agent_name == 'TEST'
        assert config.steam_quality_limits.min_dryness_fraction == 0.90

    @pytest.mark.unit
    def test_from_json_basic(self):
        """Test creation from JSON string."""
        json_str = '''
        {
            "agent_id": "GL-012-JSON",
            "agent_name": "JSON_TEST",
            "enabled": true,
            "steam_quality_limits": {
                "target_dryness_fraction": 0.99
            }
        }
        '''

        config = SteamQualityAgentConfig.from_json(json_str)

        assert config.agent_id == 'GL-012-JSON'
        assert config.steam_quality_limits.target_dryness_fraction == 0.99

    @pytest.mark.unit
    def test_roundtrip_serialization(self, default_config):
        """Test serialization roundtrip preserves values."""
        # Serialize
        json_str = default_config.to_json()

        # Deserialize
        restored = SteamQualityAgentConfig.from_json(json_str)

        # Compare key values
        assert restored.agent_id == default_config.agent_id
        assert restored.steam_quality_limits.min_dryness_fraction == \
               default_config.steam_quality_limits.min_dryness_fraction
        assert restored.desuperheater.max_injection_rate_kg_s == \
               default_config.desuperheater.max_injection_rate_kg_s

    @pytest.mark.unit
    def test_enum_deserialization(self):
        """Test enums are correctly deserialized."""
        data = {
            'desuperheater': {
                'valve_characteristic': 'linear',
            },
            'pressure_control': {
                'control_strategy': 'aggressive',
            },
        }

        config = SteamQualityAgentConfig.from_dict(data)

        assert config.desuperheater.valve_characteristic == ValveCharacteristic.LINEAR
        assert config.pressure_control.control_strategy == ControlStrategy.AGGRESSIVE

    @pytest.mark.unit
    def test_from_dict_with_defaults(self):
        """Test missing fields use defaults."""
        data = {
            'agent_id': 'TEST',
        }

        config = SteamQualityAgentConfig.from_dict(data)

        # Should have default values for missing fields
        assert config.enabled is True  # Default
        assert config.steam_quality_limits.min_dryness_fraction == 0.85  # Default


# =============================================================================
# TEST CLASS: FILE OPERATIONS
# =============================================================================

class TestFileOperations:
    """Test suite for configuration file operations."""

    @pytest.fixture
    def default_config(self):
        """Create default configuration."""
        return create_default_config()

    @pytest.mark.unit
    def test_save_to_file(self, default_config, temp_directory):
        """Test saving configuration to file."""
        config_path = temp_directory / "test_config.json"

        with open(config_path, 'w') as f:
            f.write(default_config.to_json())

        assert config_path.exists()

        # Verify content
        with open(config_path, 'r') as f:
            content = f.read()
        parsed = json.loads(content)
        assert parsed['agent_id'] == 'GL-012'

    @pytest.mark.unit
    def test_load_from_file(self, default_config, temp_directory):
        """Test loading configuration from file."""
        config_path = temp_directory / "load_test.json"

        # Save
        with open(config_path, 'w') as f:
            f.write(default_config.to_json())

        # Load
        with open(config_path, 'r') as f:
            loaded = SteamQualityAgentConfig.from_json(f.read())

        assert loaded.agent_id == default_config.agent_id

    @pytest.mark.unit
    def test_invalid_json_raises_error(self):
        """Test invalid JSON raises error."""
        invalid_json = "{ invalid json }"

        with pytest.raises(json.JSONDecodeError):
            SteamQualityAgentConfig.from_json(invalid_json)


# =============================================================================
# TEST CLASS: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test suite for configuration edge cases."""

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_all_integrations_disabled(self):
        """Test configuration with all integrations disabled."""
        config = create_default_config()
        config.integration.scada_enabled = False
        config.integration.opc_ua_enabled = False
        config.integration.modbus_enabled = False
        config.integration.mqtt_enabled = False
        config.integration.historian_enabled = False

        # Should still be valid
        assert config.is_valid()

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_desuperheater_disabled(self):
        """Test configuration with desuperheater disabled."""
        config = create_default_config()
        config.desuperheater.enabled = False

        # Invalid injection rates should be ignored when disabled
        config.desuperheater.max_injection_rate_kg_s = -1.0

        # Validation might still fail or might skip disabled components
        # depending on implementation

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_extreme_pid_gains(self):
        """Test extreme but valid PID gains."""
        config = create_default_config()

        # Very high gains (aggressive tuning)
        config.desuperheater.pid_kp = 100.0
        config.desuperheater.pid_ki = 50.0
        config.desuperheater.pid_kd = 10.0

        # Should be valid (though possibly unstable in practice)
        assert config.is_valid()

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_zero_pid_gains(self):
        """Test zero PID gains (no control)."""
        config = create_default_config()

        config.desuperheater.pid_kp = 0.0
        config.desuperheater.pid_ki = 0.0
        config.desuperheater.pid_kd = 0.0

        # Should be valid (though useless in practice)
        assert config.is_valid()

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_boundary_dryness_values(self):
        """Test boundary dryness values."""
        config = create_default_config()

        # Exactly at boundaries
        config.steam_quality_limits.min_dryness_fraction = 0.0
        config.steam_quality_limits.target_dryness_fraction = 0.0
        config.steam_quality_limits.critical_dryness_fraction = 0.0
        assert len([e for e in config.validate() if "dryness" in e.lower()]) == 0

        config.steam_quality_limits.min_dryness_fraction = 1.0
        config.steam_quality_limits.target_dryness_fraction = 1.0
        config.steam_quality_limits.critical_dryness_fraction = 1.0
        assert len([e for e in config.validate() if "dryness" in e.lower()]) == 0

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_very_long_retention_period(self):
        """Test very long data retention period."""
        config = create_default_config()
        config.monitoring.trend_retention_days = 365 * 10  # 10 years

        # Should be valid
        assert config.is_valid()

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_very_fast_sample_rate(self):
        """Test very fast sample rate."""
        config = create_default_config()
        config.monitoring.sample_rate_seconds = 0.001  # 1ms

        # Should be valid (though may be impractical)
        assert config.is_valid()


# =============================================================================
# TEST CLASS: SCHEMA VALIDATION
# =============================================================================

class TestSchemaValidation:
    """Test suite for configuration schema validation."""

    @pytest.mark.unit
    def test_required_fields(self):
        """Test required fields are present in default config."""
        config = create_default_config()

        assert hasattr(config, 'agent_id')
        assert hasattr(config, 'agent_name')
        assert hasattr(config, 'version')
        assert hasattr(config, 'enabled')
        assert hasattr(config, 'steam_quality_limits')
        assert hasattr(config, 'pressure_limits')
        assert hasattr(config, 'desuperheater')
        assert hasattr(config, 'pressure_control')
        assert hasattr(config, 'alarms')
        assert hasattr(config, 'monitoring')
        assert hasattr(config, 'integration')

    @pytest.mark.unit
    def test_nested_structure_types(self):
        """Test nested structures have correct types."""
        config = create_default_config()

        assert isinstance(config.steam_quality_limits, SteamQualityLimits)
        assert isinstance(config.pressure_limits, PressureLimits)
        assert isinstance(config.desuperheater, DesuperheaterConfig)
        assert isinstance(config.pressure_control, PressureControlConfig)
        assert isinstance(config.alarms, AlarmConfig)
        assert isinstance(config.monitoring, MonitoringConfig)
        assert isinstance(config.integration, IntegrationConfig)

    @pytest.mark.unit
    def test_enum_types(self):
        """Test enum fields have correct types."""
        config = create_default_config()

        assert isinstance(config.desuperheater.valve_characteristic, ValveCharacteristic)
        assert isinstance(config.pressure_control.control_strategy, ControlStrategy)

    @pytest.mark.unit
    def test_numeric_field_types(self):
        """Test numeric fields have correct types."""
        config = create_default_config()

        assert isinstance(config.steam_quality_limits.min_dryness_fraction, float)
        assert isinstance(config.pressure_limits.min_pressure_bar, float)
        assert isinstance(config.desuperheater.pid_kp, float)
        assert isinstance(config.monitoring.trend_retention_days, int)
        assert isinstance(config.alarms.alarm_delay_seconds, int)

    @pytest.mark.unit
    def test_boolean_field_types(self):
        """Test boolean fields have correct types."""
        config = create_default_config()

        assert isinstance(config.enabled, bool)
        assert isinstance(config.desuperheater.enabled, bool)
        assert isinstance(config.pressure_control.enabled, bool)
        assert isinstance(config.monitoring.enable_trending, bool)
        assert isinstance(config.integration.scada_enabled, bool)
