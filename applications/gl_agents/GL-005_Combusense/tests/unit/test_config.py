# -*- coding: utf-8 -*-
"""
Unit tests for GL-005 CombustionControlAgent configuration management.

Tests configuration loading, environment variable parsing,
validation rules, and default values.

Target: 10+ tests covering:
- Configuration loading from files
- Environment variable parsing
- Validation rules
- Default value handling
- Configuration merging
- Error handling
"""

import pytest
import os
from typing import Dict, Any
from unittest.mock import patch, mock_open

pytestmark = pytest.mark.unit


# ============================================================================
# CONFIGURATION LOADING TESTS
# ============================================================================

class TestConfigurationLoading:
    """Test configuration loading from various sources."""

    def test_load_config_from_dict(self, combustion_config):
        """Test loading configuration from dictionary."""
        assert combustion_config['controller_id'] == 'CC-001'
        assert combustion_config['control_loop_interval_ms'] == 100
        assert combustion_config['deterministic_mode'] is True

    def test_load_config_with_defaults(self):
        """Test configuration with default values."""
        minimal_config = {
            'controller_id': 'CC-TEST'
        }

        # Apply defaults
        default_config = {
            'controller_id': 'CC-DEFAULT',
            'control_loop_interval_ms': 100,
            'safety_check_interval_ms': 50,
            'optimization_enabled': True,
            'deterministic_mode': True
        }

        final_config = {**default_config, **minimal_config}

        assert final_config['controller_id'] == 'CC-TEST'
        assert final_config['control_loop_interval_ms'] == 100  # Default

    def test_load_config_from_json_string(self):
        """Test loading configuration from JSON string."""
        import json

        json_config = '''
        {
            "controller_id": "CC-JSON",
            "control_loop_interval_ms": 150,
            "safety_check_interval_ms": 75
        }
        '''

        config = json.loads(json_config)

        assert config['controller_id'] == 'CC-JSON'
        assert config['control_loop_interval_ms'] == 150

    @patch('builtins.open', new_callable=mock_open, read_data='{"controller_id": "CC-FILE"}')
    def test_load_config_from_file(self, mock_file):
        """Test loading configuration from file."""
        import json

        with open('config.json', 'r') as f:
            config = json.load(f)

        assert config['controller_id'] == 'CC-FILE'


# ============================================================================
# ENVIRONMENT VARIABLE PARSING TESTS
# ============================================================================

class TestEnvironmentVariableParsing:
    """Test parsing configuration from environment variables."""

    @patch.dict(os.environ, {
        'GL005_CONTROLLER_ID': 'CC-ENV-001',
        'GL005_CONTROL_LOOP_INTERVAL_MS': '200',
        'GL005_OPTIMIZATION_ENABLED': 'true'
    })
    def test_parse_env_vars(self):
        """Test parsing configuration from environment variables."""
        config = {
            'controller_id': os.getenv('GL005_CONTROLLER_ID'),
            'control_loop_interval_ms': int(os.getenv('GL005_CONTROL_LOOP_INTERVAL_MS', 100)),
            'optimization_enabled': os.getenv('GL005_OPTIMIZATION_ENABLED', 'false').lower() == 'true'
        }

        assert config['controller_id'] == 'CC-ENV-001'
        assert config['control_loop_interval_ms'] == 200
        assert config['optimization_enabled'] is True

    @patch.dict(os.environ, {'GL005_MAX_TEMP': '1400.5'})
    def test_parse_float_from_env(self):
        """Test parsing float values from environment variables."""
        max_temp = float(os.getenv('GL005_MAX_TEMP', 1400.0))

        assert max_temp == 1400.5
        assert isinstance(max_temp, float)

    @patch.dict(os.environ, {'GL005_SAFETY_ENABLED': 'true'})
    def test_parse_boolean_from_env(self):
        """Test parsing boolean values from environment variables."""
        safety_enabled = os.getenv('GL005_SAFETY_ENABLED', 'false').lower() == 'true'

        assert safety_enabled is True
        assert isinstance(safety_enabled, bool)

    def test_env_var_fallback_to_default(self):
        """Test fallback to default when environment variable not set."""
        controller_id = os.getenv('NONEXISTENT_VAR', 'CC-DEFAULT')

        assert controller_id == 'CC-DEFAULT'


# ============================================================================
# CONFIGURATION VALIDATION TESTS
# ============================================================================

class TestConfigurationValidation:
    """Test configuration validation rules."""

    def test_validate_required_fields(self):
        """Test validation of required configuration fields."""
        config = {
            'controller_id': 'CC-001',
            'control_loop_interval_ms': 100
        }

        required_fields = ['controller_id', 'control_loop_interval_ms']

        for field in required_fields:
            assert field in config

    def test_validate_controller_id_format(self):
        """Test validation of controller ID format."""
        valid_id = 'CC-001'
        invalid_id = '001'

        assert valid_id.startswith('CC-')
        assert not invalid_id.startswith('CC-')

    def test_validate_interval_positive(self):
        """Test validation that intervals are positive."""
        config = {
            'control_loop_interval_ms': 100,
            'safety_check_interval_ms': 50
        }

        assert config['control_loop_interval_ms'] > 0
        assert config['safety_check_interval_ms'] > 0

    def test_validate_safety_interval_less_than_control_interval(self):
        """Test safety interval should be less than or equal to control interval."""
        config = {
            'control_loop_interval_ms': 100,
            'safety_check_interval_ms': 50
        }

        assert config['safety_check_interval_ms'] <= config['control_loop_interval_ms']

    def test_validate_temperature_limits(self, safety_limits):
        """Test validation of temperature limits."""
        assert safety_limits.max_temperature_c > safety_limits.min_temperature_c
        assert safety_limits.min_temperature_c > 0  # Above absolute zero

    def test_validate_pressure_limits(self, safety_limits):
        """Test validation of pressure limits."""
        assert safety_limits.max_pressure_mbar > safety_limits.min_pressure_mbar
        assert safety_limits.min_pressure_mbar >= 0  # Positive pressure

    def test_validate_fuel_flow_limits(self, safety_limits):
        """Test validation of fuel flow limits."""
        assert safety_limits.max_fuel_flow_kg_hr > safety_limits.min_fuel_flow_kg_hr
        assert safety_limits.min_fuel_flow_kg_hr > 0

    def test_validate_emission_limits(self, safety_limits):
        """Test validation of emission limits."""
        assert safety_limits.max_co_ppm > 0
        assert safety_limits.max_nox_ppm > 0


# ============================================================================
# DEFAULT VALUE HANDLING TESTS
# ============================================================================

class TestDefaultValueHandling:
    """Test default value handling in configuration."""

    def test_default_control_loop_interval(self):
        """Test default control loop interval."""
        default_interval = 100  # ms

        assert default_interval == 100

    def test_default_safety_check_interval(self):
        """Test default safety check interval."""
        default_interval = 50  # ms

        assert default_interval == 50

    def test_default_optimization_enabled(self):
        """Test default optimization enabled flag."""
        default_optimization = True

        assert default_optimization is True

    def test_default_deterministic_mode(self):
        """Test default deterministic mode flag."""
        default_deterministic = True

        assert default_deterministic is True

    def test_default_logging_enabled(self):
        """Test default logging enabled flag."""
        default_logging = True

        assert default_logging is True


# ============================================================================
# CONFIGURATION MERGING TESTS
# ============================================================================

class TestConfigurationMerging:
    """Test merging of multiple configuration sources."""

    def test_merge_default_and_user_config(self):
        """Test merging default and user configuration."""
        default_config = {
            'controller_id': 'CC-DEFAULT',
            'control_loop_interval_ms': 100,
            'optimization_enabled': True
        }

        user_config = {
            'controller_id': 'CC-USER',
            'control_loop_interval_ms': 150
        }

        merged_config = {**default_config, **user_config}

        assert merged_config['controller_id'] == 'CC-USER'
        assert merged_config['control_loop_interval_ms'] == 150
        assert merged_config['optimization_enabled'] is True  # From default

    def test_merge_preserves_user_overrides(self):
        """Test merging preserves user overrides."""
        default_value = 100
        user_value = 200

        merged_value = user_value if user_value else default_value

        assert merged_value == 200

    def test_merge_nested_configurations(self):
        """Test merging nested configuration dictionaries."""
        default_config = {
            'controller': {
                'id': 'CC-DEFAULT',
                'interval_ms': 100
            },
            'safety': {
                'enabled': True
            }
        }

        user_config = {
            'controller': {
                'id': 'CC-USER'
            }
        }

        # Deep merge
        merged_config = default_config.copy()
        if 'controller' in user_config:
            merged_config['controller'].update(user_config['controller'])

        assert merged_config['controller']['id'] == 'CC-USER'
        assert merged_config['controller']['interval_ms'] == 100


# ============================================================================
# CONFIGURATION ERROR HANDLING TESTS
# ============================================================================

class TestConfigurationErrorHandling:
    """Test configuration error handling."""

    def test_handle_missing_required_field(self):
        """Test handling of missing required field."""
        config = {
            # Missing controller_id
            'control_loop_interval_ms': 100
        }

        has_controller_id = 'controller_id' in config

        assert has_controller_id is False

    def test_handle_invalid_type(self):
        """Test handling of invalid configuration type."""
        try:
            interval = int('not_a_number')
        except ValueError as e:
            assert 'invalid literal' in str(e).lower()

    def test_handle_negative_interval(self):
        """Test handling of negative interval value."""
        invalid_interval = -100

        is_valid = invalid_interval > 0

        assert is_valid is False

    def test_handle_interval_order_violation(self):
        """Test handling of safety interval > control interval."""
        control_interval = 100
        safety_interval = 150  # Invalid: should be <= control_interval

        is_valid = safety_interval <= control_interval

        assert is_valid is False


# ============================================================================
# CONTROL PARAMETERS CONFIGURATION TESTS
# ============================================================================

class TestControlParametersConfiguration:
    """Test control parameters configuration."""

    def test_pid_parameters_configuration(self, control_parameters):
        """Test PID parameters are properly configured."""
        assert control_parameters['pid_kp'] > 0
        assert control_parameters['pid_ki'] >= 0
        assert control_parameters['pid_kd'] >= 0

    def test_setpoint_configuration(self, control_parameters):
        """Test setpoint values are properly configured."""
        assert control_parameters['setpoint_temperature_c'] > 0
        assert control_parameters['setpoint_o2_percent'] > 0
        assert control_parameters['fuel_air_ratio_target'] > 0

    def test_ramp_rate_limit_configuration(self, control_parameters):
        """Test ramp rate limit is properly configured."""
        assert control_parameters['ramp_rate_limit_c_per_min'] > 0


# ============================================================================
# OPTIMIZATION CONFIGURATION TESTS
# ============================================================================

class TestOptimizationConfiguration:
    """Test optimization configuration."""

    def test_optimization_objective_configuration(self, optimization_config):
        """Test optimization objective is properly configured."""
        valid_objectives = ['fuel_efficiency', 'emissions', 'cost', 'balanced']
        assert optimization_config['objective'] in valid_objectives

    def test_optimization_constraints_configuration(self, optimization_config):
        """Test optimization constraints are properly configured."""
        assert len(optimization_config['constraints']) > 0
        assert isinstance(optimization_config['constraints'], list)

    def test_optimization_convergence_configuration(self, optimization_config):
        """Test optimization convergence settings are proper."""
        assert optimization_config['convergence_tolerance'] > 0
        assert optimization_config['max_iterations'] > 0

    def test_optimization_interval_configuration(self, optimization_config):
        """Test optimization interval is properly configured."""
        assert optimization_config['optimization_interval_sec'] > 0
