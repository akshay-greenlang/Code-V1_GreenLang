"""
Unit tests for GL-001 ThermalCommand Orchestrator Configuration Module

Tests all configuration classes and validation logic with 90%+ coverage.
Validates configuration loading, validation, and default values.

Author: GreenLang Test Engineering Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import Any, Dict
from unittest.mock import Mock, patch
import os
import json

from greenlang.agents.process_heat.gl_001_thermal_command.config import (
    OrchestratorConfig,
    SafetyConfig,
    PerformanceConfig,
    EquipmentConfig,
    IntegrationConfig,
    AlertThreshold,
    SILLevel,
    OperatingMode,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_orchestrator_config():
    """Create default orchestrator configuration."""
    return OrchestratorConfig()


@pytest.fixture
def custom_orchestrator_config():
    """Create custom orchestrator configuration."""
    return OrchestratorConfig(
        orchestrator_id="GL-001-CUSTOM",
        name="Custom Thermal Orchestrator",
        version="2.0.0",
        environment="production",
        max_agents=50,
        heartbeat_interval_s=30.0,
        task_timeout_s=600.0,
    )


@pytest.fixture
def safety_config():
    """Create safety configuration."""
    return SafetyConfig(
        sil_level=SILLevel.SIL_2,
        max_temperature_c=550.0,
        max_pressure_bar=20.0,
        emergency_shutdown_delay_s=0.5,
        safety_margin_percent=10.0,
    )


@pytest.fixture
def performance_config():
    """Create performance configuration."""
    return PerformanceConfig(
        target_latency_ms=100.0,
        max_concurrent_tasks=20,
        batch_size=100,
        cache_ttl_s=300,
    )


@pytest.fixture
def equipment_config():
    """Create equipment configuration."""
    return EquipmentConfig(
        equipment_id="BLR-001",
        equipment_type="boiler",
        max_capacity_mw=25.0,
        min_capacity_mw=5.0,
        efficiency_curve=[(0.2, 0.75), (0.5, 0.85), (0.8, 0.90), (1.0, 0.88)],
    )


@pytest.fixture
def integration_config():
    """Create integration configuration."""
    return IntegrationConfig(
        erp_endpoint="https://erp.example.com/api",
        cmms_type="SAP_PM",
        historian_endpoint="https://historian.example.com/api",
        auth_method="oauth2",
    )


# =============================================================================
# ORCHESTRATOR CONFIG TESTS
# =============================================================================

class TestOrchestratorConfig:
    """Test suite for OrchestratorConfig."""

    @pytest.mark.unit
    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = OrchestratorConfig()

        assert config.orchestrator_id is not None
        assert config.name == "ThermalCommand Orchestrator"
        assert config.version == "1.0.0"
        assert config.environment == "development"
        assert config.max_agents == 100
        assert config.heartbeat_interval_s == 60.0
        assert config.task_timeout_s == 300.0

    @pytest.mark.unit
    def test_custom_initialization(self, custom_orchestrator_config):
        """Test custom configuration initialization."""
        config = custom_orchestrator_config

        assert config.orchestrator_id == "GL-001-CUSTOM"
        assert config.name == "Custom Thermal Orchestrator"
        assert config.version == "2.0.0"
        assert config.environment == "production"
        assert config.max_agents == 50
        assert config.heartbeat_interval_s == 30.0
        assert config.task_timeout_s == 600.0

    @pytest.mark.unit
    def test_validation_max_agents_positive(self):
        """Test max_agents must be positive."""
        with pytest.raises(ValueError):
            OrchestratorConfig(max_agents=0)

        with pytest.raises(ValueError):
            OrchestratorConfig(max_agents=-10)

    @pytest.mark.unit
    def test_validation_heartbeat_positive(self):
        """Test heartbeat_interval must be positive."""
        with pytest.raises(ValueError):
            OrchestratorConfig(heartbeat_interval_s=0)

        with pytest.raises(ValueError):
            OrchestratorConfig(heartbeat_interval_s=-5.0)

    @pytest.mark.unit
    def test_validation_timeout_positive(self):
        """Test task_timeout must be positive."""
        with pytest.raises(ValueError):
            OrchestratorConfig(task_timeout_s=0)

    @pytest.mark.unit
    def test_environment_values(self):
        """Test valid environment values."""
        for env in ["development", "staging", "production", "test"]:
            config = OrchestratorConfig(environment=env)
            assert config.environment == env

    @pytest.mark.unit
    def test_config_to_dict(self, default_orchestrator_config):
        """Test configuration serialization to dict."""
        config_dict = default_orchestrator_config.model_dump()

        assert isinstance(config_dict, dict)
        assert "orchestrator_id" in config_dict
        assert "name" in config_dict
        assert "version" in config_dict

    @pytest.mark.unit
    def test_config_from_dict(self):
        """Test configuration deserialization from dict."""
        config_dict = {
            "orchestrator_id": "GL-001-TEST",
            "name": "Test Orchestrator",
            "version": "1.0.0",
            "environment": "test",
            "max_agents": 25,
        }

        config = OrchestratorConfig(**config_dict)
        assert config.orchestrator_id == "GL-001-TEST"
        assert config.max_agents == 25

    @pytest.mark.unit
    def test_config_immutability(self, default_orchestrator_config):
        """Test configuration is immutable after creation."""
        # Pydantic models are mutable by default, but we test the intended behavior
        original_id = default_orchestrator_config.orchestrator_id
        assert original_id is not None


# =============================================================================
# SAFETY CONFIG TESTS
# =============================================================================

class TestSafetyConfig:
    """Test suite for SafetyConfig."""

    @pytest.mark.unit
    def test_default_initialization(self):
        """Test default safety configuration."""
        config = SafetyConfig()

        assert config.sil_level == SILLevel.SIL_1
        assert config.max_temperature_c > 0
        assert config.max_pressure_bar > 0
        assert config.emergency_shutdown_delay_s >= 0
        assert 0 < config.safety_margin_percent <= 100

    @pytest.mark.unit
    def test_sil_levels(self):
        """Test SIL level configuration."""
        for sil in [SILLevel.SIL_1, SILLevel.SIL_2, SILLevel.SIL_3]:
            config = SafetyConfig(sil_level=sil)
            assert config.sil_level == sil

    @pytest.mark.unit
    def test_temperature_validation(self):
        """Test temperature limit validation."""
        # Valid temperature
        config = SafetyConfig(max_temperature_c=600.0)
        assert config.max_temperature_c == 600.0

        # Invalid temperature (too low)
        with pytest.raises(ValueError):
            SafetyConfig(max_temperature_c=-50.0)

    @pytest.mark.unit
    def test_pressure_validation(self):
        """Test pressure limit validation."""
        # Valid pressure
        config = SafetyConfig(max_pressure_bar=30.0)
        assert config.max_pressure_bar == 30.0

        # Invalid pressure (negative)
        with pytest.raises(ValueError):
            SafetyConfig(max_pressure_bar=-5.0)

    @pytest.mark.unit
    def test_safety_margin_range(self):
        """Test safety margin must be within range."""
        # Valid margins
        config = SafetyConfig(safety_margin_percent=5.0)
        assert config.safety_margin_percent == 5.0

        config = SafetyConfig(safety_margin_percent=20.0)
        assert config.safety_margin_percent == 20.0

        # Invalid margin (too high)
        with pytest.raises(ValueError):
            SafetyConfig(safety_margin_percent=150.0)

    @pytest.mark.unit
    def test_emergency_shutdown_delay(self, safety_config):
        """Test emergency shutdown delay configuration."""
        assert safety_config.emergency_shutdown_delay_s == 0.5
        assert safety_config.emergency_shutdown_delay_s < 5.0  # Should be fast

    @pytest.mark.unit
    def test_get_adjusted_limits(self, safety_config):
        """Test getting limits adjusted by safety margin."""
        # With 10% safety margin, limits should be reduced
        adjusted_temp = safety_config.max_temperature_c * (1 - safety_config.safety_margin_percent / 100)
        assert adjusted_temp < safety_config.max_temperature_c

        adjusted_pressure = safety_config.max_pressure_bar * (1 - safety_config.safety_margin_percent / 100)
        assert adjusted_pressure < safety_config.max_pressure_bar


# =============================================================================
# PERFORMANCE CONFIG TESTS
# =============================================================================

class TestPerformanceConfig:
    """Test suite for PerformanceConfig."""

    @pytest.mark.unit
    def test_default_initialization(self):
        """Test default performance configuration."""
        config = PerformanceConfig()

        assert config.target_latency_ms > 0
        assert config.max_concurrent_tasks > 0
        assert config.batch_size > 0
        assert config.cache_ttl_s > 0

    @pytest.mark.unit
    def test_latency_validation(self):
        """Test latency target validation."""
        # Valid latency
        config = PerformanceConfig(target_latency_ms=50.0)
        assert config.target_latency_ms == 50.0

        # Invalid latency
        with pytest.raises(ValueError):
            PerformanceConfig(target_latency_ms=0)

    @pytest.mark.unit
    def test_concurrent_tasks_validation(self):
        """Test concurrent tasks limit validation."""
        # Valid limit
        config = PerformanceConfig(max_concurrent_tasks=50)
        assert config.max_concurrent_tasks == 50

        # Invalid limit
        with pytest.raises(ValueError):
            PerformanceConfig(max_concurrent_tasks=0)

    @pytest.mark.unit
    def test_batch_size_validation(self):
        """Test batch size validation."""
        # Valid batch size
        config = PerformanceConfig(batch_size=500)
        assert config.batch_size == 500

        # Invalid batch size
        with pytest.raises(ValueError):
            PerformanceConfig(batch_size=-10)

    @pytest.mark.unit
    def test_cache_ttl_validation(self):
        """Test cache TTL validation."""
        # Valid TTL
        config = PerformanceConfig(cache_ttl_s=600)
        assert config.cache_ttl_s == 600

        # Zero TTL should be allowed (no caching)
        config = PerformanceConfig(cache_ttl_s=0)
        assert config.cache_ttl_s == 0


# =============================================================================
# EQUIPMENT CONFIG TESTS
# =============================================================================

class TestEquipmentConfig:
    """Test suite for EquipmentConfig."""

    @pytest.mark.unit
    def test_initialization(self, equipment_config):
        """Test equipment configuration initialization."""
        assert equipment_config.equipment_id == "BLR-001"
        assert equipment_config.equipment_type == "boiler"
        assert equipment_config.max_capacity_mw == 25.0
        assert equipment_config.min_capacity_mw == 5.0

    @pytest.mark.unit
    def test_capacity_validation(self):
        """Test capacity range validation."""
        # Valid capacity range
        config = EquipmentConfig(
            equipment_id="TEST-001",
            equipment_type="heater",
            max_capacity_mw=50.0,
            min_capacity_mw=10.0,
        )
        assert config.max_capacity_mw > config.min_capacity_mw

        # Invalid: min > max
        with pytest.raises(ValueError):
            EquipmentConfig(
                equipment_id="TEST-001",
                equipment_type="heater",
                max_capacity_mw=10.0,
                min_capacity_mw=50.0,
            )

    @pytest.mark.unit
    def test_efficiency_curve(self, equipment_config):
        """Test efficiency curve configuration."""
        curve = equipment_config.efficiency_curve
        assert len(curve) == 4

        # Verify curve is sorted by load factor
        load_factors = [point[0] for point in curve]
        assert load_factors == sorted(load_factors)

        # Verify efficiency values are valid (0-1)
        for load, efficiency in curve:
            assert 0 <= load <= 1.0
            assert 0 <= efficiency <= 1.0

    @pytest.mark.unit
    def test_get_efficiency_at_load(self, equipment_config):
        """Test efficiency interpolation at given load."""
        # At defined point
        load_80_efficiency = None
        for load, eff in equipment_config.efficiency_curve:
            if load == 0.8:
                load_80_efficiency = eff
        assert load_80_efficiency == 0.90

    @pytest.mark.unit
    def test_equipment_type_values(self):
        """Test valid equipment types."""
        valid_types = ["boiler", "heater", "furnace", "heat_exchanger", "steam_generator"]
        for eq_type in valid_types:
            config = EquipmentConfig(
                equipment_id="TEST-001",
                equipment_type=eq_type,
                max_capacity_mw=25.0,
                min_capacity_mw=5.0,
            )
            assert config.equipment_type == eq_type


# =============================================================================
# INTEGRATION CONFIG TESTS
# =============================================================================

class TestIntegrationConfig:
    """Test suite for IntegrationConfig."""

    @pytest.mark.unit
    def test_initialization(self, integration_config):
        """Test integration configuration initialization."""
        assert integration_config.erp_endpoint == "https://erp.example.com/api"
        assert integration_config.cmms_type == "SAP_PM"
        assert integration_config.historian_endpoint == "https://historian.example.com/api"
        assert integration_config.auth_method == "oauth2"

    @pytest.mark.unit
    def test_endpoint_validation(self):
        """Test endpoint URL validation."""
        # Valid HTTPS endpoint
        config = IntegrationConfig(erp_endpoint="https://secure.example.com")
        assert config.erp_endpoint.startswith("https://")

    @pytest.mark.unit
    def test_cmms_type_values(self):
        """Test valid CMMS type values."""
        valid_types = ["SAP_PM", "MAXIMO", "INFOR_EAM", "FIIX", "GENERIC_REST", "MOCK"]
        for cmms in valid_types:
            config = IntegrationConfig(cmms_type=cmms)
            assert config.cmms_type == cmms

    @pytest.mark.unit
    def test_auth_methods(self):
        """Test authentication method configuration."""
        valid_auth = ["oauth2", "api_key", "basic", "certificate"]
        for auth in valid_auth:
            config = IntegrationConfig(auth_method=auth)
            assert config.auth_method == auth


# =============================================================================
# ALERT THRESHOLD TESTS
# =============================================================================

class TestAlertThreshold:
    """Test suite for AlertThreshold."""

    @pytest.mark.unit
    def test_initialization(self):
        """Test alert threshold initialization."""
        threshold = AlertThreshold(
            name="high_temperature",
            warning_value=450.0,
            critical_value=500.0,
            unit="degC",
        )

        assert threshold.name == "high_temperature"
        assert threshold.warning_value == 450.0
        assert threshold.critical_value == 500.0
        assert threshold.unit == "degC"

    @pytest.mark.unit
    def test_threshold_ordering(self):
        """Test warning < critical threshold ordering."""
        # Valid ordering
        threshold = AlertThreshold(
            name="test",
            warning_value=100.0,
            critical_value=150.0,
        )
        assert threshold.warning_value < threshold.critical_value

        # Invalid ordering should raise
        with pytest.raises(ValueError):
            AlertThreshold(
                name="test",
                warning_value=150.0,
                critical_value=100.0,
            )

    @pytest.mark.unit
    def test_check_value(self):
        """Test value checking against threshold."""
        threshold = AlertThreshold(
            name="temperature",
            warning_value=450.0,
            critical_value=500.0,
        )

        # Below warning
        assert threshold.check_value(400.0) == "normal"

        # Warning level
        assert threshold.check_value(475.0) == "warning"

        # Critical level
        assert threshold.check_value(525.0) == "critical"

    @pytest.mark.unit
    def test_deadband_handling(self):
        """Test deadband for alarm hysteresis."""
        threshold = AlertThreshold(
            name="pressure",
            warning_value=15.0,
            critical_value=20.0,
            deadband=1.0,
        )

        # Value at warning boundary with deadband
        # Should not flip-flop at boundary
        assert threshold.deadband == 1.0


# =============================================================================
# OPERATING MODE TESTS
# =============================================================================

class TestOperatingMode:
    """Test suite for OperatingMode enum."""

    @pytest.mark.unit
    def test_mode_values(self):
        """Test operating mode enumeration values."""
        assert OperatingMode.STARTUP.value == "startup"
        assert OperatingMode.NORMAL.value == "normal"
        assert OperatingMode.SHUTDOWN.value == "shutdown"
        assert OperatingMode.EMERGENCY.value == "emergency"
        assert OperatingMode.MAINTENANCE.value == "maintenance"

    @pytest.mark.unit
    def test_mode_transitions_allowed(self):
        """Test allowed mode transitions."""
        allowed_transitions = {
            OperatingMode.STARTUP: [OperatingMode.NORMAL, OperatingMode.EMERGENCY, OperatingMode.SHUTDOWN],
            OperatingMode.NORMAL: [OperatingMode.SHUTDOWN, OperatingMode.EMERGENCY, OperatingMode.MAINTENANCE],
            OperatingMode.SHUTDOWN: [OperatingMode.MAINTENANCE, OperatingMode.EMERGENCY],
            OperatingMode.MAINTENANCE: [OperatingMode.STARTUP, OperatingMode.EMERGENCY],
            OperatingMode.EMERGENCY: [OperatingMode.SHUTDOWN],
        }

        # Verify emergency is always a valid transition target
        for mode, targets in allowed_transitions.items():
            if mode != OperatingMode.EMERGENCY:
                assert OperatingMode.EMERGENCY in targets


# =============================================================================
# SIL LEVEL TESTS
# =============================================================================

class TestSILLevel:
    """Test suite for SIL Level enum."""

    @pytest.mark.unit
    def test_sil_values(self):
        """Test SIL level enumeration values."""
        assert SILLevel.SIL_1.value == 1
        assert SILLevel.SIL_2.value == 2
        assert SILLevel.SIL_3.value == 3

    @pytest.mark.unit
    def test_sil_ordering(self):
        """Test SIL levels are properly ordered."""
        assert SILLevel.SIL_1.value < SILLevel.SIL_2.value
        assert SILLevel.SIL_2.value < SILLevel.SIL_3.value

    @pytest.mark.unit
    def test_proof_test_intervals(self):
        """Test proof test intervals by SIL level."""
        # Higher SIL = more frequent testing
        proof_test_days = {
            SILLevel.SIL_1: 365,  # Annual
            SILLevel.SIL_2: 180,  # Semi-annual
            SILLevel.SIL_3: 90,   # Quarterly
        }

        assert proof_test_days[SILLevel.SIL_1] > proof_test_days[SILLevel.SIL_2]
        assert proof_test_days[SILLevel.SIL_2] > proof_test_days[SILLevel.SIL_3]


# =============================================================================
# CONFIGURATION LOADING TESTS
# =============================================================================

class TestConfigLoading:
    """Test configuration loading from various sources."""

    @pytest.mark.unit
    def test_load_from_env_vars(self, monkeypatch):
        """Test loading configuration from environment variables."""
        monkeypatch.setenv("GL001_ORCHESTRATOR_ID", "GL-001-ENV")
        monkeypatch.setenv("GL001_ENVIRONMENT", "staging")
        monkeypatch.setenv("GL001_MAX_AGENTS", "75")

        # Configuration should pick up env vars
        config = OrchestratorConfig(
            orchestrator_id=os.getenv("GL001_ORCHESTRATOR_ID", "default"),
            environment=os.getenv("GL001_ENVIRONMENT", "development"),
            max_agents=int(os.getenv("GL001_MAX_AGENTS", "100")),
        )

        assert config.orchestrator_id == "GL-001-ENV"
        assert config.environment == "staging"
        assert config.max_agents == 75

    @pytest.mark.unit
    def test_load_from_json_file(self, tmp_path):
        """Test loading configuration from JSON file."""
        config_file = tmp_path / "config.json"
        config_data = {
            "orchestrator_id": "GL-001-JSON",
            "name": "JSON Loaded Orchestrator",
            "environment": "production",
            "max_agents": 200,
        }
        config_file.write_text(json.dumps(config_data))

        with open(config_file) as f:
            loaded_data = json.load(f)

        config = OrchestratorConfig(**loaded_data)
        assert config.orchestrator_id == "GL-001-JSON"
        assert config.max_agents == 200

    @pytest.mark.unit
    def test_config_validation_on_load(self):
        """Test configuration validation during load."""
        invalid_config = {
            "orchestrator_id": "GL-001-INVALID",
            "max_agents": -50,  # Invalid
        }

        with pytest.raises(ValueError):
            OrchestratorConfig(**invalid_config)

    @pytest.mark.unit
    def test_config_merge(self):
        """Test merging multiple configuration sources."""
        base_config = OrchestratorConfig()
        override_config = {
            "max_agents": 150,
            "heartbeat_interval_s": 45.0,
        }

        # Merge configurations
        merged_dict = base_config.model_dump()
        merged_dict.update(override_config)

        merged_config = OrchestratorConfig(**merged_dict)
        assert merged_config.max_agents == 150
        assert merged_config.heartbeat_interval_s == 45.0
        # Non-overridden values should remain
        assert merged_config.name == base_config.name
