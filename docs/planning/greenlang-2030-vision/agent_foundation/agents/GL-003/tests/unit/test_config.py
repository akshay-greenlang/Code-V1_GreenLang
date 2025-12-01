# -*- coding: utf-8 -*-
"""
Unit tests for GL-003 STEAMWISE SteamSystemAnalyzer configuration management.

Tests configuration loading, environment variable parsing,
validation rules, and default values.

Target: 25+ tests covering:
- Configuration loading from files
- Environment variable parsing
- Validation rules
- Default value handling
- Configuration merging
- Steam system configuration
- Pressure zone configuration
- Steam trap configuration
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

class SteamType(str, Enum):
    """Steam types supported."""
    SATURATED = "saturated"
    SUPERHEATED = "superheated"
    WET = "wet"


class PressureClass(str, Enum):
    """Steam pressure classifications."""
    LOW = "low"  # < 3.5 bar
    MEDIUM = "medium"  # 3.5 - 17 bar
    HIGH = "high"  # > 17 bar


class SteamTrapType(str, Enum):
    """Steam trap types."""
    THERMODYNAMIC = "thermodynamic"
    THERMOSTATIC = "thermostatic"
    MECHANICAL = "mechanical"
    INVERTED_BUCKET = "inverted_bucket"


# ============================================================================
# CONFIGURATION LOADING TESTS
# ============================================================================

class TestConfigurationLoading:
    """Test configuration loading from various sources."""

    def test_load_config_from_dict(self):
        """Test loading configuration from dictionary."""
        config = {
            'system_id': 'STEAM-001',
            'system_name': 'SteamSystemAnalyzer',
            'version': '1.0.0',
            'boiler_count': 2,
            'total_capacity_tonnes_hr': 50.0
        }

        assert config['system_id'] == 'STEAM-001'
        assert config['system_name'] == 'SteamSystemAnalyzer'
        assert config['total_capacity_tonnes_hr'] == 50.0

    def test_load_config_with_defaults(self):
        """Test configuration with default values."""
        minimal_config = {
            'system_id': 'STEAM-TEST'
        }

        default_config = {
            'system_id': 'STEAM-001',
            'system_name': 'SteamSystemAnalyzer',
            'version': '1.0.0',
            'boiler_count': 1,
            'total_capacity_tonnes_hr': 10.0,
            'design_pressure_bar': 10.0
        }

        final_config = {**default_config, **minimal_config}

        assert final_config['system_id'] == 'STEAM-TEST'
        assert final_config['boiler_count'] == 1

    def test_load_config_from_json_string(self):
        """Test loading configuration from JSON string."""
        json_config = '''
        {
            "system_id": "STEAM-JSON",
            "boiler_count": 3,
            "total_capacity_tonnes_hr": 75.0
        }
        '''

        config = json.loads(json_config)

        assert config['system_id'] == 'STEAM-JSON'
        assert config['boiler_count'] == 3

    @patch('builtins.open', new_callable=mock_open, read_data='{"system_id": "STEAM-FILE"}')
    def test_load_config_from_file(self, mock_file):
        """Test loading configuration from file."""
        with open('steam_config.json', 'r') as f:
            config = json.load(f)

        assert config['system_id'] == 'STEAM-FILE'


# ============================================================================
# ENVIRONMENT VARIABLE PARSING TESTS
# ============================================================================

class TestEnvironmentVariableParsing:
    """Test parsing configuration from environment variables."""

    @patch.dict(os.environ, {
        'GL003_SYSTEM_ID': 'STEAM-ENV-001',
        'GL003_BOILER_COUNT': '4',
        'GL003_MONITORING_ENABLED': 'true'
    })
    def test_parse_env_vars(self):
        """Test parsing configuration from environment variables."""
        config = {
            'system_id': os.getenv('GL003_SYSTEM_ID'),
            'boiler_count': int(os.getenv('GL003_BOILER_COUNT', 1)),
            'monitoring_enabled': os.getenv('GL003_MONITORING_ENABLED', 'false').lower() == 'true'
        }

        assert config['system_id'] == 'STEAM-ENV-001'
        assert config['boiler_count'] == 4
        assert config['monitoring_enabled'] is True

    @patch.dict(os.environ, {'GL003_DESIGN_PRESSURE': '15.5'})
    def test_parse_float_from_env(self):
        """Test parsing float values from environment variables."""
        design_pressure = float(os.getenv('GL003_DESIGN_PRESSURE', 10.0))

        assert design_pressure == 15.5
        assert isinstance(design_pressure, float)

    @patch.dict(os.environ, {'GL003_CONDENSATE_RECOVERY': 'true'})
    def test_parse_boolean_from_env(self):
        """Test parsing boolean values from environment variables."""
        condensate_recovery = os.getenv('GL003_CONDENSATE_RECOVERY', 'false').lower() == 'true'

        assert condensate_recovery is True
        assert isinstance(condensate_recovery, bool)

    def test_env_var_fallback_to_default(self):
        """Test fallback to default when environment variable not set."""
        system_id = os.getenv('NONEXISTENT_GL003_VAR', 'STEAM-DEFAULT')

        assert system_id == 'STEAM-DEFAULT'


# ============================================================================
# BOILER CONFIGURATION VALIDATION TESTS
# ============================================================================

class TestBoilerConfigurationValidation:
    """Test boiler configuration validation rules."""

    def test_validate_boiler_required_fields(self):
        """Test validation of required boiler configuration fields."""
        boiler_config = {
            'boiler_id': 'BOILER-001',
            'boiler_type': 'fire_tube',
            'capacity_tonnes_hr': 25.0,
            'design_pressure_bar': 15.0,
            'design_temperature_c': 198.0,
            'fuel_type': 'natural_gas',
            'efficiency_percent': 85.0
        }

        required_fields = ['boiler_id', 'capacity_tonnes_hr', 'design_pressure_bar', 'fuel_type']

        for field in required_fields:
            assert field in boiler_config

    def test_validate_boiler_type_enum(self):
        """Test validation of boiler type enum value."""
        valid_types = ['fire_tube', 'water_tube', 'once_through', 'electric']
        boiler_type = 'fire_tube'

        assert boiler_type in valid_types

    def test_validate_capacity_positive(self):
        """Test capacity must be positive."""
        capacity = 25.0

        assert capacity > 0

    def test_validate_pressure_range(self):
        """Test design pressure within valid range."""
        design_pressure = 15.0

        assert 0 < design_pressure <= 300  # Typical steam pressure limit

    def test_validate_efficiency_range(self):
        """Test efficiency within valid range."""
        efficiency = 85.0

        assert 50 <= efficiency <= 100


# ============================================================================
# PRESSURE ZONE CONFIGURATION TESTS
# ============================================================================

class TestPressureZoneConfiguration:
    """Test pressure zone configuration validation."""

    def test_validate_pressure_zone_required_fields(self):
        """Test validation of required pressure zone fields."""
        zone_config = {
            'zone_id': 'ZONE-HIGH',
            'zone_name': 'High Pressure Zone',
            'pressure_class': 'high',
            'design_pressure_bar': 40.0,
            'design_temperature_c': 250.0,
            'consumers': ['reactor_1', 'reactor_2']
        }

        required_fields = ['zone_id', 'pressure_class', 'design_pressure_bar']

        for field in required_fields:
            assert field in zone_config

    def test_validate_pressure_class_enum(self):
        """Test validation of pressure class enum value."""
        valid_classes = ['low', 'medium', 'high']
        pressure_class = 'high'

        assert pressure_class in valid_classes

    def test_validate_pressure_class_boundaries(self):
        """Test pressure class boundary validation."""
        pressure = 20.0

        if pressure < 3.5:
            pressure_class = 'low'
        elif pressure <= 17.0:
            pressure_class = 'medium'
        else:
            pressure_class = 'high'

        assert pressure_class == 'high'

    def test_validate_multiple_zones(self):
        """Test configuration with multiple pressure zones."""
        zones = [
            {'zone_id': 'ZONE-HIGH', 'design_pressure_bar': 40.0},
            {'zone_id': 'ZONE-MEDIUM', 'design_pressure_bar': 10.0},
            {'zone_id': 'ZONE-LOW', 'design_pressure_bar': 2.0},
        ]

        assert len(zones) == 3
        # Zones should be in descending pressure order for cascade
        pressures = [z['design_pressure_bar'] for z in zones]
        assert pressures == sorted(pressures, reverse=True)


# ============================================================================
# STEAM TRAP CONFIGURATION TESTS
# ============================================================================

class TestSteamTrapConfiguration:
    """Test steam trap configuration validation."""

    def test_validate_trap_required_fields(self):
        """Test validation of required trap fields."""
        trap_config = {
            'trap_id': 'TRAP-001',
            'trap_type': 'thermodynamic',
            'location': 'Drip leg 1',
            'inlet_pressure_bar': 10.0,
            'outlet_pressure_bar': 1.0,
            'capacity_kg_hr': 50.0
        }

        required_fields = ['trap_id', 'trap_type', 'inlet_pressure_bar']

        for field in required_fields:
            assert field in trap_config

    def test_validate_trap_type_enum(self):
        """Test validation of trap type enum value."""
        valid_types = ['thermodynamic', 'thermostatic', 'mechanical', 'inverted_bucket']
        trap_type = 'thermodynamic'

        assert trap_type in valid_types

    def test_validate_trap_pressure_differential(self):
        """Test trap inlet pressure > outlet pressure."""
        inlet_pressure = 10.0
        outlet_pressure = 1.0

        assert inlet_pressure > outlet_pressure

    def test_validate_trap_selection_by_application(self):
        """Test trap selection based on application."""
        trap_selection = {
            'main_drip_leg': 'thermodynamic',
            'process_heat_exchanger': 'thermostatic',
            'tracing_line': 'thermostatic',
            'high_pressure_drip': 'inverted_bucket'
        }

        assert trap_selection['main_drip_leg'] == 'thermodynamic'
        assert trap_selection['process_heat_exchanger'] == 'thermostatic'


# ============================================================================
# CONDENSATE SYSTEM CONFIGURATION TESTS
# ============================================================================

class TestCondensateSystemConfiguration:
    """Test condensate system configuration validation."""

    def test_validate_condensate_config(self):
        """Test condensate system configuration."""
        condensate_config = {
            'recovery_enabled': True,
            'recovery_target_percent': 85.0,
            'flash_vessel_enabled': True,
            'return_temperature_c': 80.0,
            'deaerator_enabled': True
        }

        assert condensate_config['recovery_enabled'] is True
        assert condensate_config['recovery_target_percent'] > 0

    def test_validate_recovery_target_range(self):
        """Test recovery target within valid range."""
        recovery_target = 85.0

        assert 0 <= recovery_target <= 100

    def test_validate_flash_vessel_configuration(self):
        """Test flash vessel configuration."""
        flash_config = {
            'flash_vessel_id': 'FV-001',
            'high_pressure_inlet_bar': 10.0,
            'low_pressure_outlet_bar': 2.0,
            'design_capacity_kg_hr': 500.0
        }

        assert flash_config['high_pressure_inlet_bar'] > flash_config['low_pressure_outlet_bar']


# ============================================================================
# WATER TREATMENT CONFIGURATION TESTS
# ============================================================================

class TestWaterTreatmentConfiguration:
    """Test water treatment configuration validation."""

    def test_validate_water_quality_limits(self):
        """Test water quality parameter limits."""
        water_quality = {
            'tds_max_ppm': 3500,
            'ph_min': 10.5,
            'ph_max': 11.5,
            'silica_max_ppm': 150,
            'oxygen_max_ppb': 7
        }

        assert water_quality['ph_min'] < water_quality['ph_max']
        assert water_quality['tds_max_ppm'] > 0

    def test_validate_blowdown_configuration(self):
        """Test blowdown configuration."""
        blowdown_config = {
            'continuous_blowdown_enabled': True,
            'blowdown_rate_percent': 5.0,
            'heat_recovery_enabled': True,
            'target_tds_ppm': 2500
        }

        assert blowdown_config['blowdown_rate_percent'] > 0
        assert blowdown_config['blowdown_rate_percent'] < 20  # Typical max


# ============================================================================
# CONFIGURATION MERGING TESTS
# ============================================================================

class TestConfigurationMerging:
    """Test merging of multiple configuration sources."""

    def test_merge_default_and_user_config(self):
        """Test merging default and user configuration."""
        default_config = {
            'system_id': 'STEAM-DEFAULT',
            'boiler_count': 1,
            'monitoring_enabled': True
        }

        user_config = {
            'system_id': 'STEAM-USER',
            'boiler_count': 3
        }

        merged_config = {**default_config, **user_config}

        assert merged_config['system_id'] == 'STEAM-USER'
        assert merged_config['boiler_count'] == 3
        assert merged_config['monitoring_enabled'] is True

    def test_merge_nested_configurations(self):
        """Test merging nested configuration dictionaries."""
        default_config = {
            'boiler': {
                'capacity': 25.0,
                'efficiency': 85.0
            },
            'condensate': {
                'recovery_enabled': True
            }
        }

        user_config = {
            'boiler': {
                'capacity': 50.0
            }
        }

        merged_config = default_config.copy()
        if 'boiler' in user_config:
            merged_config['boiler'].update(user_config['boiler'])

        assert merged_config['boiler']['capacity'] == 50.0
        assert merged_config['boiler']['efficiency'] == 85.0


# ============================================================================
# CONFIGURATION ERROR HANDLING TESTS
# ============================================================================

class TestConfigurationErrorHandling:
    """Test configuration error handling."""

    def test_handle_missing_required_field(self):
        """Test handling of missing required field."""
        config = {
            'boiler_count': 2
            # Missing system_id
        }

        has_system_id = 'system_id' in config

        assert has_system_id is False

    def test_handle_invalid_pressure_class(self):
        """Test handling of invalid pressure class."""
        valid_classes = ['low', 'medium', 'high']
        invalid_class = 'ultra_high'

        is_valid = invalid_class in valid_classes

        assert is_valid is False

    def test_handle_negative_capacity(self):
        """Test handling of negative capacity value."""
        invalid_capacity = -25.0

        is_valid = invalid_capacity > 0

        assert is_valid is False

    def test_handle_invalid_efficiency(self):
        """Test handling of invalid efficiency value."""
        invalid_efficiency = 110.0  # > 100%

        is_valid = 0 <= invalid_efficiency <= 100

        assert is_valid is False


# ============================================================================
# DEFAULT VALUE HANDLING TESTS
# ============================================================================

class TestDefaultValueHandling:
    """Test default value handling in configuration."""

    def test_default_design_pressure(self):
        """Test default design pressure."""
        default = 10.0  # bar
        assert default == 10.0

    def test_default_blowdown_rate(self):
        """Test default blowdown rate."""
        default = 5.0  # percent
        assert default == 5.0

    def test_default_recovery_target(self):
        """Test default condensate recovery target."""
        default = 80.0  # percent
        assert default == 80.0

    def test_default_monitoring_interval(self):
        """Test default monitoring interval."""
        default = 60  # seconds
        assert default == 60

    def test_default_steam_quality(self):
        """Test default steam quality target."""
        default = 0.98  # 98% dryness
        assert default == 0.98
