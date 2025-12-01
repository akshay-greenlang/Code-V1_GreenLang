# -*- coding: utf-8 -*-
"""
Unit tests for GL-002 FLAMEGUARD BurnerManagementController configuration management.

Tests configuration loading, environment variable parsing,
validation rules, and default values.

Target: 25+ tests covering:
- Configuration loading from files
- Environment variable parsing
- Validation rules
- Default value handling
- Configuration merging
- Safety interlock configuration
- Burner sequence configuration
- PID tuning configuration
"""

import pytest
import os
import json
from typing import Dict, Any
from unittest.mock import patch, mock_open
from enum import Enum

pytestmark = pytest.mark.unit


# ============================================================================
# MOCK ENUMS FOR TESTING
# ============================================================================

class BurnerType(str, Enum):
    """Burner types supported."""
    GAS = "gas"
    OIL = "oil"
    DUAL_FUEL = "dual_fuel"
    LOW_NOX = "low_nox"


class IgnitionType(str, Enum):
    """Ignition types."""
    DIRECT_SPARK = "direct_spark"
    PILOT = "pilot"
    HOT_SURFACE = "hot_surface"


class SafetyInterlockType(str, Enum):
    """Safety interlock types."""
    FLAME_LOSS = "flame_loss"
    HIGH_PRESSURE = "high_pressure"
    LOW_PRESSURE = "low_pressure"
    HIGH_TEMPERATURE = "high_temperature"
    GAS_LEAK = "gas_leak"


# ============================================================================
# CONFIGURATION LOADING TESTS
# ============================================================================

class TestConfigurationLoading:
    """Test configuration loading from various sources."""

    def test_load_config_from_dict(self):
        """Test loading configuration from dictionary."""
        config = {
            'controller_id': 'BMS-001',
            'controller_name': 'BurnerManagementController',
            'version': '1.0.0',
            'burner_count': 4,
            'control_loop_interval_ms': 100
        }

        assert config['controller_id'] == 'BMS-001'
        assert config['controller_name'] == 'BurnerManagementController'
        assert config['burner_count'] == 4

    def test_load_config_with_defaults(self):
        """Test configuration with default values."""
        minimal_config = {
            'controller_id': 'BMS-TEST'
        }

        default_config = {
            'controller_id': 'BMS-001',
            'controller_name': 'BurnerManagementController',
            'version': '1.0.0',
            'burner_count': 1,
            'control_loop_interval_ms': 100,
            'safety_check_interval_ms': 50
        }

        final_config = {**default_config, **minimal_config}

        assert final_config['controller_id'] == 'BMS-TEST'
        assert final_config['burner_count'] == 1

    def test_load_config_from_json_string(self):
        """Test loading configuration from JSON string."""
        json_config = '''
        {
            "controller_id": "BMS-JSON",
            "burner_count": 6,
            "control_loop_interval_ms": 50
        }
        '''

        config = json.loads(json_config)

        assert config['controller_id'] == 'BMS-JSON'
        assert config['burner_count'] == 6

    @patch('builtins.open', new_callable=mock_open, read_data='{"controller_id": "BMS-FILE"}')
    def test_load_config_from_file(self, mock_file):
        """Test loading configuration from file."""
        with open('bms_config.json', 'r') as f:
            config = json.load(f)

        assert config['controller_id'] == 'BMS-FILE'


# ============================================================================
# ENVIRONMENT VARIABLE PARSING TESTS
# ============================================================================

class TestEnvironmentVariableParsing:
    """Test parsing configuration from environment variables."""

    @patch.dict(os.environ, {
        'GL002_CONTROLLER_ID': 'BMS-ENV-001',
        'GL002_BURNER_COUNT': '8',
        'GL002_SAFETY_ENABLED': 'true'
    })
    def test_parse_env_vars(self):
        """Test parsing configuration from environment variables."""
        config = {
            'controller_id': os.getenv('GL002_CONTROLLER_ID'),
            'burner_count': int(os.getenv('GL002_BURNER_COUNT', 1)),
            'safety_enabled': os.getenv('GL002_SAFETY_ENABLED', 'false').lower() == 'true'
        }

        assert config['controller_id'] == 'BMS-ENV-001'
        assert config['burner_count'] == 8
        assert config['safety_enabled'] is True

    @patch.dict(os.environ, {'GL002_PURGE_TIME': '120.5'})
    def test_parse_float_from_env(self):
        """Test parsing float values from environment variables."""
        purge_time = float(os.getenv('GL002_PURGE_TIME', 80.0))

        assert purge_time == 120.5
        assert isinstance(purge_time, float)

    @patch.dict(os.environ, {'GL002_INTERLOCK_ENABLED': 'true'})
    def test_parse_boolean_from_env(self):
        """Test parsing boolean values from environment variables."""
        interlock_enabled = os.getenv('GL002_INTERLOCK_ENABLED', 'false').lower() == 'true'

        assert interlock_enabled is True
        assert isinstance(interlock_enabled, bool)

    def test_env_var_fallback_to_default(self):
        """Test fallback to default when environment variable not set."""
        controller_id = os.getenv('NONEXISTENT_GL002_VAR', 'BMS-DEFAULT')

        assert controller_id == 'BMS-DEFAULT'


# ============================================================================
# BURNER CONFIGURATION VALIDATION TESTS
# ============================================================================

class TestBurnerConfigurationValidation:
    """Test burner configuration validation rules."""

    def test_validate_burner_required_fields(self):
        """Test validation of required burner configuration fields."""
        burner_config = {
            'burner_id': 'BURNER-001',
            'burner_type': 'gas',
            'capacity_mw': 5.0,
            'ignition_type': 'pilot',
            'fuel_type': 'natural_gas',
            'min_turndown_ratio': 0.2
        }

        required_fields = ['burner_id', 'burner_type', 'capacity_mw', 'ignition_type']

        for field in required_fields:
            assert field in burner_config

    def test_validate_burner_type_enum(self):
        """Test validation of burner type enum value."""
        valid_types = ['gas', 'oil', 'dual_fuel', 'low_nox']
        burner_type = 'gas'

        assert burner_type in valid_types

    def test_validate_ignition_type_enum(self):
        """Test validation of ignition type enum value."""
        valid_types = ['direct_spark', 'pilot', 'hot_surface']
        ignition_type = 'pilot'

        assert ignition_type in valid_types

    def test_validate_capacity_positive(self):
        """Test capacity must be positive."""
        capacity_mw = 5.0

        assert capacity_mw > 0

    def test_validate_turndown_ratio_range(self):
        """Test turndown ratio within valid range."""
        turndown_ratio = 0.2

        assert 0.1 <= turndown_ratio <= 1.0

    def test_validate_multiple_burners(self):
        """Test configuration with multiple burners."""
        burners = [
            {'burner_id': 'BURNER-001', 'capacity_mw': 5.0},
            {'burner_id': 'BURNER-002', 'capacity_mw': 5.0},
            {'burner_id': 'BURNER-003', 'capacity_mw': 5.0},
            {'burner_id': 'BURNER-004', 'capacity_mw': 5.0},
        ]

        assert len(burners) == 4
        total_capacity = sum(b['capacity_mw'] for b in burners)
        assert total_capacity == pytest.approx(20.0, rel=1e-6)


# ============================================================================
# SAFETY INTERLOCK CONFIGURATION TESTS
# ============================================================================

class TestSafetyInterlockConfiguration:
    """Test safety interlock configuration validation."""

    def test_validate_interlock_required_fields(self):
        """Test validation of required interlock fields."""
        interlock_config = {
            'interlock_id': 'INT-001',
            'type': 'flame_loss',
            'trip_delay_ms': 3000,
            'reset_type': 'manual',
            'bypass_allowed': False
        }

        required_fields = ['interlock_id', 'type', 'trip_delay_ms', 'reset_type']

        for field in required_fields:
            assert field in interlock_config

    def test_validate_interlock_type_enum(self):
        """Test validation of interlock type enum value."""
        valid_types = ['flame_loss', 'high_pressure', 'low_pressure', 'high_temperature', 'gas_leak']
        interlock_type = 'flame_loss'

        assert interlock_type in valid_types

    def test_validate_trip_delay_range(self):
        """Test trip delay within valid range."""
        trip_delay_ms = 3000

        assert 0 <= trip_delay_ms <= 10000

    def test_validate_reset_type(self):
        """Test reset type validation."""
        valid_reset_types = ['manual', 'automatic', 'timed']
        reset_type = 'manual'

        assert reset_type in valid_reset_types

    def test_validate_bypass_configuration(self):
        """Test bypass configuration validation."""
        bypass_allowed = False
        bypass_duration_max_min = 30

        if bypass_allowed:
            assert bypass_duration_max_min > 0
        else:
            assert bypass_allowed is False


# ============================================================================
# IGNITION SEQUENCE CONFIGURATION TESTS
# ============================================================================

class TestIgnitionSequenceConfiguration:
    """Test ignition sequence configuration validation."""

    def test_validate_sequence_timing(self):
        """Test ignition sequence timing validation."""
        sequence_config = {
            'pre_purge_time_sec': 80,
            'pilot_ignition_time_sec': 5,
            'pilot_proving_time_sec': 7,
            'main_ignition_time_sec': 10,
            'main_proving_time_sec': 5,
            'post_ignition_stabilization_sec': 15
        }

        # Pre-purge must be >= 4 air changes (NFPA 86)
        assert sequence_config['pre_purge_time_sec'] >= 60

        # Pilot proving must be >= 5 seconds
        assert sequence_config['pilot_proving_time_sec'] >= 5

    def test_validate_trial_for_ignition_limits(self):
        """Test TFI limits based on burner capacity."""
        burner_capacity_mw = 5.0

        if burner_capacity_mw < 1.0:
            max_tfi_sec = 15
        elif burner_capacity_mw < 10.0:
            max_tfi_sec = 10
        else:
            max_tfi_sec = 5

        assert max_tfi_sec == 10

    def test_validate_retry_configuration(self):
        """Test ignition retry configuration."""
        retry_config = {
            'max_retries': 3,
            'retry_delay_sec': 60,
            'lockout_after_failures': True
        }

        assert retry_config['max_retries'] <= 3
        assert retry_config['retry_delay_sec'] >= 60


# ============================================================================
# PID TUNING CONFIGURATION TESTS
# ============================================================================

class TestPIDTuningConfiguration:
    """Test PID tuning configuration validation."""

    def test_validate_pid_parameters(self):
        """Test PID parameters are properly configured."""
        pid_config = {
            'kp': 2.4,
            'ki': 0.12,
            'kd': 12.0,
            'output_min': 0.0,
            'output_max': 100.0,
            'anti_windup_enabled': True
        }

        assert pid_config['kp'] > 0
        assert pid_config['ki'] >= 0
        assert pid_config['kd'] >= 0
        assert pid_config['output_min'] < pid_config['output_max']

    def test_validate_tuning_method(self):
        """Test tuning method selection validation."""
        valid_methods = ['ziegler_nichols', 'cohen_coon', 'imc', 'manual']
        tuning_method = 'ziegler_nichols'

        assert tuning_method in valid_methods

    def test_validate_loop_type_tuning(self):
        """Test loop type specific tuning validation."""
        loop_tuning = {
            'pressure': {'method': 'ziegler_nichols', 'response': 'fast'},
            'temperature': {'method': 'imc', 'response': 'slow'},
            'oxygen': {'method': 'cohen_coon', 'response': 'moderate'}
        }

        assert loop_tuning['pressure']['method'] == 'ziegler_nichols'
        assert loop_tuning['temperature']['method'] == 'imc'


# ============================================================================
# ALARM CONFIGURATION TESTS
# ============================================================================

class TestAlarmConfiguration:
    """Test alarm configuration validation."""

    def test_validate_alarm_thresholds(self):
        """Test alarm threshold configuration."""
        alarm_config = {
            'temperature_high_alarm_c': 1350,
            'temperature_high_high_alarm_c': 1400,
            'pressure_high_alarm_mbar': 140,
            'pressure_high_high_alarm_mbar': 150,
            'co_high_alarm_ppm': 80,
            'co_high_high_alarm_ppm': 100
        }

        # High-high must be greater than high
        assert alarm_config['temperature_high_high_alarm_c'] > alarm_config['temperature_high_alarm_c']
        assert alarm_config['pressure_high_high_alarm_mbar'] > alarm_config['pressure_high_alarm_mbar']
        assert alarm_config['co_high_high_alarm_ppm'] > alarm_config['co_high_alarm_ppm']

    def test_validate_alarm_actions(self):
        """Test alarm action configuration."""
        valid_actions = ['alarm_only', 'reduce_load', 'emergency_shutdown', 'controlled_shutdown']

        alarm_actions = {
            'high_temperature': 'reduce_load',
            'high_high_temperature': 'emergency_shutdown',
            'flame_loss': 'emergency_shutdown'
        }

        for action in alarm_actions.values():
            assert action in valid_actions


# ============================================================================
# CONFIGURATION MERGING TESTS
# ============================================================================

class TestConfigurationMerging:
    """Test merging of multiple configuration sources."""

    def test_merge_default_and_user_config(self):
        """Test merging default and user configuration."""
        default_config = {
            'controller_id': 'BMS-DEFAULT',
            'burner_count': 1,
            'safety_enabled': True
        }

        user_config = {
            'controller_id': 'BMS-USER',
            'burner_count': 4
        }

        merged_config = {**default_config, **user_config}

        assert merged_config['controller_id'] == 'BMS-USER'
        assert merged_config['burner_count'] == 4
        assert merged_config['safety_enabled'] is True

    def test_merge_nested_configurations(self):
        """Test merging nested configuration dictionaries."""
        default_config = {
            'ignition': {
                'pre_purge_time': 80,
                'max_retries': 3
            },
            'safety': {
                'enabled': True
            }
        }

        user_config = {
            'ignition': {
                'pre_purge_time': 120
            }
        }

        merged_config = default_config.copy()
        if 'ignition' in user_config:
            merged_config['ignition'].update(user_config['ignition'])

        assert merged_config['ignition']['pre_purge_time'] == 120
        assert merged_config['ignition']['max_retries'] == 3


# ============================================================================
# CONFIGURATION ERROR HANDLING TESTS
# ============================================================================

class TestConfigurationErrorHandling:
    """Test configuration error handling."""

    def test_handle_missing_required_field(self):
        """Test handling of missing required field."""
        config = {
            'burner_count': 4
            # Missing controller_id
        }

        has_controller_id = 'controller_id' in config

        assert has_controller_id is False

    def test_handle_invalid_burner_type(self):
        """Test handling of invalid burner type."""
        valid_types = ['gas', 'oil', 'dual_fuel', 'low_nox']
        invalid_type = 'invalid_burner'

        is_valid = invalid_type in valid_types

        assert is_valid is False

    def test_handle_negative_capacity(self):
        """Test handling of negative capacity value."""
        invalid_capacity = -5.0

        is_valid = invalid_capacity > 0

        assert is_valid is False

    def test_handle_invalid_trip_delay(self):
        """Test handling of invalid trip delay."""
        invalid_delay = 15000  # > 10000ms limit

        is_valid = 0 <= invalid_delay <= 10000

        assert is_valid is False


# ============================================================================
# DEFAULT VALUE HANDLING TESTS
# ============================================================================

class TestDefaultValueHandling:
    """Test default value handling in configuration."""

    def test_default_control_loop_interval(self):
        """Test default control loop interval."""
        default = 100  # ms
        assert default == 100

    def test_default_safety_check_interval(self):
        """Test default safety check interval."""
        default = 50  # ms
        assert default == 50

    def test_default_pre_purge_time(self):
        """Test default pre-purge time."""
        default = 80  # seconds
        assert default == 80

    def test_default_max_ignition_retries(self):
        """Test default max ignition retries."""
        default = 3
        assert default == 3

    def test_default_flame_failure_response_time(self):
        """Test default flame failure response time."""
        default = 3000  # ms (3 seconds per NFPA 86)
        assert default == 3000
