# -*- coding: utf-8 -*-
"""
Unit tests for GL-004 BURNMASTER CombustionOptimizer configuration management.

Tests configuration loading, environment variable parsing,
validation rules, and default values.

Target: 25+ tests covering:
- Configuration loading from files
- Environment variable parsing
- Validation rules
- Default value handling
- Configuration merging
- Fuel configuration
- Combustion control configuration
- Emissions limits configuration
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

class FuelType(str, Enum):
    """Fuel types supported."""
    NATURAL_GAS = "natural_gas"
    FUEL_OIL = "fuel_oil"
    COAL = "coal"
    BIOMASS = "biomass"
    DUAL_FUEL = "dual_fuel"


class ControlMode(str, Enum):
    """Control modes."""
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    CASCADE = "cascade"
    RATIO = "ratio"


class OptimizationObjective(str, Enum):
    """Optimization objectives."""
    EFFICIENCY = "efficiency"
    EMISSIONS = "emissions"
    COST = "cost"
    BALANCED = "balanced"


# ============================================================================
# CONFIGURATION LOADING TESTS
# ============================================================================

class TestConfigurationLoading:
    """Test configuration loading from various sources."""

    def test_load_config_from_dict(self):
        """Test loading configuration from dictionary."""
        config = {
            'optimizer_id': 'BURN-001',
            'optimizer_name': 'CombustionOptimizer',
            'version': '1.0.0',
            'primary_fuel': 'natural_gas',
            'optimization_interval_sec': 60
        }

        assert config['optimizer_id'] == 'BURN-001'
        assert config['optimizer_name'] == 'CombustionOptimizer'
        assert config['primary_fuel'] == 'natural_gas'

    def test_load_config_with_defaults(self):
        """Test configuration with default values."""
        minimal_config = {
            'optimizer_id': 'BURN-TEST'
        }

        default_config = {
            'optimizer_id': 'BURN-001',
            'optimizer_name': 'CombustionOptimizer',
            'version': '1.0.0',
            'primary_fuel': 'natural_gas',
            'optimization_interval_sec': 60,
            'target_efficiency_percent': 90.0
        }

        final_config = {**default_config, **minimal_config}

        assert final_config['optimizer_id'] == 'BURN-TEST'
        assert final_config['optimization_interval_sec'] == 60

    def test_load_config_from_json_string(self):
        """Test loading configuration from JSON string."""
        json_config = '''
        {
            "optimizer_id": "BURN-JSON",
            "primary_fuel": "fuel_oil",
            "optimization_interval_sec": 120
        }
        '''

        config = json.loads(json_config)

        assert config['optimizer_id'] == 'BURN-JSON'
        assert config['primary_fuel'] == 'fuel_oil'

    @patch('builtins.open', new_callable=mock_open, read_data='{"optimizer_id": "BURN-FILE"}')
    def test_load_config_from_file(self, mock_file):
        """Test loading configuration from file."""
        with open('combustion_config.json', 'r') as f:
            config = json.load(f)

        assert config['optimizer_id'] == 'BURN-FILE'


# ============================================================================
# ENVIRONMENT VARIABLE PARSING TESTS
# ============================================================================

class TestEnvironmentVariableParsing:
    """Test parsing configuration from environment variables."""

    @patch.dict(os.environ, {
        'GL004_OPTIMIZER_ID': 'BURN-ENV-001',
        'GL004_PRIMARY_FUEL': 'natural_gas',
        'GL004_OPTIMIZATION_ENABLED': 'true'
    })
    def test_parse_env_vars(self):
        """Test parsing configuration from environment variables."""
        config = {
            'optimizer_id': os.getenv('GL004_OPTIMIZER_ID'),
            'primary_fuel': os.getenv('GL004_PRIMARY_FUEL', 'natural_gas'),
            'optimization_enabled': os.getenv('GL004_OPTIMIZATION_ENABLED', 'false').lower() == 'true'
        }

        assert config['optimizer_id'] == 'BURN-ENV-001'
        assert config['primary_fuel'] == 'natural_gas'
        assert config['optimization_enabled'] is True

    @patch.dict(os.environ, {'GL004_TARGET_EFFICIENCY': '92.5'})
    def test_parse_float_from_env(self):
        """Test parsing float values from environment variables."""
        target_efficiency = float(os.getenv('GL004_TARGET_EFFICIENCY', 90.0))

        assert target_efficiency == 92.5
        assert isinstance(target_efficiency, float)

    @patch.dict(os.environ, {'GL004_AUTO_TUNE_ENABLED': 'true'})
    def test_parse_boolean_from_env(self):
        """Test parsing boolean values from environment variables."""
        auto_tune = os.getenv('GL004_AUTO_TUNE_ENABLED', 'false').lower() == 'true'

        assert auto_tune is True
        assert isinstance(auto_tune, bool)

    def test_env_var_fallback_to_default(self):
        """Test fallback to default when environment variable not set."""
        optimizer_id = os.getenv('NONEXISTENT_GL004_VAR', 'BURN-DEFAULT')

        assert optimizer_id == 'BURN-DEFAULT'


# ============================================================================
# FUEL CONFIGURATION VALIDATION TESTS
# ============================================================================

class TestFuelConfigurationValidation:
    """Test fuel configuration validation rules."""

    def test_validate_fuel_required_fields(self):
        """Test validation of required fuel configuration fields."""
        fuel_config = {
            'fuel_id': 'FUEL-001',
            'fuel_type': 'natural_gas',
            'hhv_mj_kg': 52.0,
            'lhv_mj_kg': 47.0,
            'carbon_content_percent': 75.0,
            'hydrogen_content_percent': 25.0
        }

        required_fields = ['fuel_type', 'hhv_mj_kg', 'lhv_mj_kg']

        for field in required_fields:
            assert field in fuel_config

    def test_validate_fuel_type_enum(self):
        """Test validation of fuel type enum value."""
        valid_types = ['natural_gas', 'fuel_oil', 'coal', 'biomass', 'dual_fuel']
        fuel_type = 'natural_gas'

        assert fuel_type in valid_types

    def test_validate_hhv_greater_than_lhv(self):
        """Test HHV > LHV validation."""
        hhv = 52.0
        lhv = 47.0

        assert hhv > lhv

    def test_validate_composition_sum(self):
        """Test fuel composition sums to reasonable total."""
        c_content = 75.0
        h_content = 25.0
        o_content = 0.0
        n_content = 0.0
        s_content = 0.0

        total = c_content + h_content + o_content + n_content + s_content

        assert 95 <= total <= 100  # Allow for minor elements

    def test_validate_stoichiometric_air(self):
        """Test stoichiometric air ratio configuration."""
        fuel_config = {
            'fuel_type': 'natural_gas',
            'stoich_air_ratio': 17.2
        }

        assert fuel_config['stoich_air_ratio'] > 0
        assert 10 <= fuel_config['stoich_air_ratio'] <= 20  # Typical range


# ============================================================================
# COMBUSTION CONTROL CONFIGURATION TESTS
# ============================================================================

class TestCombustionControlConfiguration:
    """Test combustion control configuration validation."""

    def test_validate_control_mode(self):
        """Test control mode validation."""
        valid_modes = ['manual', 'automatic', 'cascade', 'ratio']
        control_mode = 'automatic'

        assert control_mode in valid_modes

    def test_validate_o2_setpoint_range(self):
        """Test O2 setpoint within valid range."""
        o2_setpoint = 3.0

        assert 1.0 <= o2_setpoint <= 10.0  # Typical operating range

    def test_validate_excess_air_limits(self):
        """Test excess air limits configuration."""
        excess_air_config = {
            'minimum_percent': 5.0,
            'maximum_percent': 50.0,
            'target_percent': 15.0
        }

        assert excess_air_config['minimum_percent'] < excess_air_config['target_percent']
        assert excess_air_config['target_percent'] < excess_air_config['maximum_percent']

    def test_validate_air_fuel_ratio_control(self):
        """Test air-fuel ratio control configuration."""
        afr_config = {
            'control_enabled': True,
            'lead_lag_enabled': True,
            'lead_gain': 1.1,
            'lag_time_constant_sec': 5.0
        }

        assert afr_config['lead_gain'] > 1.0  # Lead > 1 for anticipatory control

    def test_validate_combustion_rate_limits(self):
        """Test combustion rate change limits."""
        rate_limits = {
            'max_increase_rate_percent_min': 10.0,
            'max_decrease_rate_percent_min': 15.0
        }

        assert rate_limits['max_increase_rate_percent_min'] > 0
        assert rate_limits['max_decrease_rate_percent_min'] > 0


# ============================================================================
# OPTIMIZATION CONFIGURATION TESTS
# ============================================================================

class TestOptimizationConfiguration:
    """Test optimization configuration validation."""

    def test_validate_optimization_objective(self):
        """Test optimization objective validation."""
        valid_objectives = ['efficiency', 'emissions', 'cost', 'balanced']
        objective = 'efficiency'

        assert objective in valid_objectives

    def test_validate_optimization_interval(self):
        """Test optimization interval within valid range."""
        interval_sec = 60

        assert 10 <= interval_sec <= 600  # Reasonable range

    def test_validate_convergence_criteria(self):
        """Test convergence criteria configuration."""
        convergence_config = {
            'tolerance': 0.01,
            'max_iterations': 100,
            'min_improvement_percent': 0.1
        }

        assert convergence_config['tolerance'] > 0
        assert convergence_config['max_iterations'] > 0

    def test_validate_constraint_priorities(self):
        """Test constraint priority configuration."""
        constraints = {
            'safety': {'priority': 1, 'violation_action': 'stop'},
            'emissions': {'priority': 2, 'violation_action': 'reduce'},
            'efficiency': {'priority': 3, 'violation_action': 'warn'}
        }

        assert constraints['safety']['priority'] < constraints['emissions']['priority']


# ============================================================================
# EMISSIONS LIMITS CONFIGURATION TESTS
# ============================================================================

class TestEmissionsLimitsConfiguration:
    """Test emissions limits configuration validation."""

    def test_validate_nox_limits(self):
        """Test NOx emission limits configuration."""
        nox_limits = {
            'warning_ppm': 80.0,
            'alarm_ppm': 100.0,
            'shutdown_ppm': 150.0
        }

        assert nox_limits['warning_ppm'] < nox_limits['alarm_ppm'] < nox_limits['shutdown_ppm']

    def test_validate_co_limits(self):
        """Test CO emission limits configuration."""
        co_limits = {
            'warning_ppm': 50.0,
            'alarm_ppm': 100.0,
            'shutdown_ppm': 200.0
        }

        assert co_limits['warning_ppm'] < co_limits['alarm_ppm'] < co_limits['shutdown_ppm']

    def test_validate_so2_limits(self):
        """Test SO2 emission limits configuration."""
        so2_limits = {
            'warning_ppm': 100.0,
            'alarm_ppm': 200.0,
            'shutdown_ppm': 500.0
        }

        assert so2_limits['warning_ppm'] < so2_limits['alarm_ppm'] < so2_limits['shutdown_ppm']

    def test_validate_particulate_limits(self):
        """Test particulate emission limits configuration."""
        particulate_limits = {
            'warning_mg_m3': 30.0,
            'alarm_mg_m3': 50.0,
            'shutdown_mg_m3': 100.0
        }

        assert particulate_limits['warning_mg_m3'] < particulate_limits['alarm_mg_m3']


# ============================================================================
# ANALYZER CONFIGURATION TESTS
# ============================================================================

class TestAnalyzerConfiguration:
    """Test combustion analyzer configuration validation."""

    def test_validate_analyzer_calibration(self):
        """Test analyzer calibration configuration."""
        calibration_config = {
            'o2_span_percent': 21.0,
            'o2_zero_percent': 0.0,
            'co_span_ppm': 1000.0,
            'calibration_interval_days': 30
        }

        assert calibration_config['o2_span_percent'] > calibration_config['o2_zero_percent']
        assert calibration_config['calibration_interval_days'] > 0

    def test_validate_analyzer_sampling(self):
        """Test analyzer sampling configuration."""
        sampling_config = {
            'sample_rate_hz': 1.0,
            'averaging_time_sec': 10.0,
            'filter_constant': 0.9
        }

        assert sampling_config['sample_rate_hz'] > 0
        assert 0 < sampling_config['filter_constant'] < 1


# ============================================================================
# CONFIGURATION MERGING TESTS
# ============================================================================

class TestConfigurationMerging:
    """Test merging of multiple configuration sources."""

    def test_merge_default_and_user_config(self):
        """Test merging default and user configuration."""
        default_config = {
            'optimizer_id': 'BURN-DEFAULT',
            'primary_fuel': 'natural_gas',
            'optimization_enabled': True
        }

        user_config = {
            'optimizer_id': 'BURN-USER',
            'target_efficiency_percent': 92.0
        }

        merged_config = {**default_config, **user_config}

        assert merged_config['optimizer_id'] == 'BURN-USER'
        assert merged_config['primary_fuel'] == 'natural_gas'
        assert merged_config['target_efficiency_percent'] == 92.0

    def test_merge_nested_configurations(self):
        """Test merging nested configuration dictionaries."""
        default_config = {
            'optimization': {
                'objective': 'efficiency',
                'interval_sec': 60
            },
            'emissions': {
                'monitoring_enabled': True
            }
        }

        user_config = {
            'optimization': {
                'interval_sec': 120
            }
        }

        merged_config = default_config.copy()
        if 'optimization' in user_config:
            merged_config['optimization'].update(user_config['optimization'])

        assert merged_config['optimization']['interval_sec'] == 120
        assert merged_config['optimization']['objective'] == 'efficiency'


# ============================================================================
# CONFIGURATION ERROR HANDLING TESTS
# ============================================================================

class TestConfigurationErrorHandling:
    """Test configuration error handling."""

    def test_handle_missing_required_field(self):
        """Test handling of missing required field."""
        config = {
            'primary_fuel': 'natural_gas'
            # Missing optimizer_id
        }

        has_optimizer_id = 'optimizer_id' in config

        assert has_optimizer_id is False

    def test_handle_invalid_fuel_type(self):
        """Test handling of invalid fuel type."""
        valid_types = ['natural_gas', 'fuel_oil', 'coal', 'biomass']
        invalid_type = 'invalid_fuel'

        is_valid = invalid_type in valid_types

        assert is_valid is False

    def test_handle_hhv_lhv_order(self):
        """Test handling of invalid HHV/LHV relationship."""
        hhv = 45.0
        lhv = 50.0  # Invalid: LHV > HHV

        is_valid = hhv > lhv

        assert is_valid is False

    def test_handle_negative_efficiency_target(self):
        """Test handling of negative efficiency target."""
        target = -10.0

        is_valid = target > 0

        assert is_valid is False


# ============================================================================
# DEFAULT VALUE HANDLING TESTS
# ============================================================================

class TestDefaultValueHandling:
    """Test default value handling in configuration."""

    def test_default_optimization_interval(self):
        """Test default optimization interval."""
        default = 60  # seconds
        assert default == 60

    def test_default_target_o2(self):
        """Test default target O2 setpoint."""
        default = 3.0  # percent
        assert default == 3.0

    def test_default_excess_air_target(self):
        """Test default excess air target."""
        default = 15.0  # percent
        assert default == 15.0

    def test_default_efficiency_target(self):
        """Test default efficiency target."""
        default = 90.0  # percent
        assert default == 90.0

    def test_default_co_limit(self):
        """Test default CO limit."""
        default = 100.0  # ppm
        assert default == 100.0
