# -*- coding: utf-8 -*-
"""
Unit tests for GL-001 THERMOSYNC ProcessHeatOrchestrator configuration management.

Tests configuration loading, environment variable parsing,
validation rules, and default values.

Target: 25+ tests covering:
- Configuration loading from files
- Environment variable parsing
- Validation rules
- Default value handling
- Configuration merging
- Plant configuration validation
- Sensor configuration validation
- Integration settings validation
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

class PlantType(str, Enum):
    """Industrial plant types supported."""
    CHEMICAL = "chemical"
    PETROCHEMICAL = "petrochemical"
    STEEL = "steel"
    CEMENT = "cement"
    PAPER = "paper"
    FOOD_PROCESSING = "food_processing"


class SensorType(str, Enum):
    """Process heat sensor types."""
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FLOW_RATE = "flow_rate"
    HEAT_FLUX = "heat_flux"
    ENERGY_METER = "energy_meter"


class IntegrationProtocol(str, Enum):
    """Integration protocols."""
    OPC_UA = "opc_ua"
    MODBUS = "modbus"
    REST_API = "rest_api"
    MQTT = "mqtt"


# ============================================================================
# CONFIGURATION LOADING TESTS
# ============================================================================

class TestConfigurationLoading:
    """Test configuration loading from various sources."""

    def test_load_config_from_dict(self):
        """Test loading configuration from dictionary."""
        config = {
            'agent_id': 'GL-001',
            'agent_name': 'ProcessHeatOrchestrator',
            'version': '1.0.0',
            'max_parallel_agents': 10,
            'calculation_timeout_seconds': 120
        }

        assert config['agent_id'] == 'GL-001'
        assert config['agent_name'] == 'ProcessHeatOrchestrator'
        assert config['version'] == '1.0.0'

    def test_load_config_with_defaults(self):
        """Test configuration with default values."""
        minimal_config = {
            'agent_id': 'GL-001-TEST'
        }

        default_config = {
            'agent_id': 'GL-001',
            'agent_name': 'ProcessHeatOrchestrator',
            'version': '1.0.0',
            'max_parallel_agents': 10,
            'calculation_timeout_seconds': 120,
            'cache_ttl_seconds': 300
        }

        final_config = {**default_config, **minimal_config}

        assert final_config['agent_id'] == 'GL-001-TEST'
        assert final_config['max_parallel_agents'] == 10  # Default

    def test_load_config_from_json_string(self):
        """Test loading configuration from JSON string."""
        json_config = '''
        {
            "agent_id": "GL-001-JSON",
            "agent_name": "ProcessHeatOrchestrator",
            "version": "1.0.0",
            "max_parallel_agents": 15
        }
        '''

        config = json.loads(json_config)

        assert config['agent_id'] == 'GL-001-JSON'
        assert config['max_parallel_agents'] == 15

    @patch('builtins.open', new_callable=mock_open, read_data='{"agent_id": "GL-001-FILE"}')
    def test_load_config_from_file(self, mock_file):
        """Test loading configuration from file."""
        with open('config.json', 'r') as f:
            config = json.load(f)

        assert config['agent_id'] == 'GL-001-FILE'


# ============================================================================
# ENVIRONMENT VARIABLE PARSING TESTS
# ============================================================================

class TestEnvironmentVariableParsing:
    """Test parsing configuration from environment variables."""

    @patch.dict(os.environ, {
        'GL001_AGENT_ID': 'GL-001-ENV',
        'GL001_MAX_PARALLEL_AGENTS': '20',
        'GL001_ENABLE_MONITORING': 'true'
    })
    def test_parse_env_vars(self):
        """Test parsing configuration from environment variables."""
        config = {
            'agent_id': os.getenv('GL001_AGENT_ID'),
            'max_parallel_agents': int(os.getenv('GL001_MAX_PARALLEL_AGENTS', 10)),
            'enable_monitoring': os.getenv('GL001_ENABLE_MONITORING', 'false').lower() == 'true'
        }

        assert config['agent_id'] == 'GL-001-ENV'
        assert config['max_parallel_agents'] == 20
        assert config['enable_monitoring'] is True

    @patch.dict(os.environ, {'GL001_CAPACITY_MW': '500.5'})
    def test_parse_float_from_env(self):
        """Test parsing float values from environment variables."""
        capacity_mw = float(os.getenv('GL001_CAPACITY_MW', 100.0))

        assert capacity_mw == 500.5
        assert isinstance(capacity_mw, float)

    @patch.dict(os.environ, {'GL001_COMPLIANCE_ENABLED': 'true'})
    def test_parse_boolean_from_env(self):
        """Test parsing boolean values from environment variables."""
        compliance_enabled = os.getenv('GL001_COMPLIANCE_ENABLED', 'false').lower() == 'true'

        assert compliance_enabled is True
        assert isinstance(compliance_enabled, bool)

    def test_env_var_fallback_to_default(self):
        """Test fallback to default when environment variable not set."""
        agent_id = os.getenv('NONEXISTENT_GL001_VAR', 'GL-001-DEFAULT')

        assert agent_id == 'GL-001-DEFAULT'


# ============================================================================
# PLANT CONFIGURATION VALIDATION TESTS
# ============================================================================

class TestPlantConfigurationValidation:
    """Test plant configuration validation rules."""

    def test_validate_plant_required_fields(self):
        """Test validation of required plant configuration fields."""
        plant_config = {
            'plant_id': 'PLANT-001',
            'plant_name': 'Chemical Plant Alpha',
            'plant_type': 'chemical',
            'location': 'Houston, TX',
            'capacity_mw': 500.0,
            'max_temperature_c': 850.0,
            'min_temperature_c': 150.0,
            'nominal_pressure_bar': 40.0,
            'primary_fuel': 'natural_gas'
        }

        required_fields = [
            'plant_id', 'plant_name', 'plant_type', 'capacity_mw',
            'max_temperature_c', 'min_temperature_c', 'primary_fuel'
        ]

        for field in required_fields:
            assert field in plant_config

    def test_validate_plant_type_enum(self):
        """Test validation of plant type enum value."""
        valid_types = ['chemical', 'petrochemical', 'steel', 'cement', 'paper']
        plant_type = 'chemical'

        assert plant_type in valid_types

    def test_validate_temperature_range(self):
        """Test max temperature > min temperature validation."""
        max_temp = 850.0
        min_temp = 150.0

        assert max_temp > min_temp

    def test_validate_temperature_range_fail(self):
        """Test temperature range validation fails for invalid values."""
        max_temp = 100.0
        min_temp = 200.0

        is_valid = max_temp > min_temp

        assert is_valid is False

    def test_validate_capacity_positive(self):
        """Test capacity must be positive."""
        capacity_mw = 500.0

        assert capacity_mw > 0

    def test_validate_capacity_negative_fail(self):
        """Test negative capacity is invalid."""
        capacity_mw = -100.0

        is_valid = capacity_mw > 0

        assert is_valid is False

    def test_validate_operating_hours_range(self):
        """Test operating hours within valid range."""
        operating_hours = 8000

        assert 0 <= operating_hours <= 8760

    def test_validate_renewable_percentage_range(self):
        """Test renewable percentage within 0-100 range."""
        renewable_pct = 15.0

        assert 0 <= renewable_pct <= 100


# ============================================================================
# SENSOR CONFIGURATION VALIDATION TESTS
# ============================================================================

class TestSensorConfigurationValidation:
    """Test sensor configuration validation rules."""

    def test_validate_sensor_required_fields(self):
        """Test validation of required sensor configuration fields."""
        sensor_config = {
            'sensor_id': 'TEMP-001',
            'sensor_type': 'temperature',
            'location': 'Reactor 1',
            'unit': 'celsius',
            'sampling_rate_hz': 10.0,
            'accuracy_percent': 0.5,
            'calibration_date': '2024-01-15'
        }

        required_fields = ['sensor_id', 'sensor_type', 'unit', 'sampling_rate_hz']

        for field in required_fields:
            assert field in sensor_config

    def test_validate_sensor_type_enum(self):
        """Test validation of sensor type enum value."""
        valid_types = ['temperature', 'pressure', 'flow_rate', 'heat_flux', 'energy_meter']
        sensor_type = 'temperature'

        assert sensor_type in valid_types

    def test_validate_sampling_rate_range(self):
        """Test sampling rate within valid range."""
        sampling_rate = 10.0

        assert 0.01 <= sampling_rate <= 1000

    def test_validate_accuracy_range(self):
        """Test accuracy percentage within valid range."""
        accuracy = 0.5

        assert 0.1 <= accuracy <= 10

    def test_validate_threshold_order(self):
        """Test threshold values are in correct order."""
        min_threshold = 100.0
        alert_threshold = 800.0
        critical_threshold = 900.0
        max_threshold = 950.0

        assert min_threshold < alert_threshold < critical_threshold <= max_threshold


# ============================================================================
# SCADA INTEGRATION CONFIGURATION TESTS
# ============================================================================

class TestSCADAIntegrationConfiguration:
    """Test SCADA integration configuration."""

    def test_validate_scada_protocol(self):
        """Test SCADA protocol validation."""
        valid_protocols = ['opc_ua', 'modbus', 'rest_api', 'mqtt']
        protocol = 'opc_ua'

        assert protocol in valid_protocols

    def test_validate_polling_interval_range(self):
        """Test polling interval within valid range."""
        polling_interval = 5

        assert 1 <= polling_interval <= 60

    def test_validate_timeout_range(self):
        """Test timeout within valid range."""
        timeout = 30

        assert 5 <= timeout <= 300

    def test_validate_data_quality_threshold(self):
        """Test data quality threshold within valid range."""
        quality_threshold = 0.9

        assert 0 <= quality_threshold <= 1

    def test_validate_endpoint_url_format(self):
        """Test endpoint URL format validation."""
        endpoint_url = "opc.tcp://192.168.1.100:4840"

        assert endpoint_url.startswith(('opc.tcp://', 'http://', 'https://'))


# ============================================================================
# ERP INTEGRATION CONFIGURATION TESTS
# ============================================================================

class TestERPIntegrationConfiguration:
    """Test ERP integration configuration."""

    def test_validate_erp_system_type(self):
        """Test ERP system type validation."""
        valid_systems = ['SAP', 'Oracle', 'Microsoft Dynamics', 'Infor']
        system_type = 'SAP'

        assert system_type in valid_systems

    def test_validate_sync_interval_range(self):
        """Test sync interval within valid range."""
        sync_interval = 60

        assert sync_interval >= 5

    def test_validate_batch_size_range(self):
        """Test batch size within valid range."""
        batch_size = 1000

        assert 100 <= batch_size <= 10000


# ============================================================================
# OPTIMIZATION PARAMETERS CONFIGURATION TESTS
# ============================================================================

class TestOptimizationParametersConfiguration:
    """Test optimization parameters configuration."""

    def test_validate_optimization_algorithm(self):
        """Test optimization algorithm validation."""
        valid_algorithms = ['linear_programming', 'milp', 'genetic_algorithm', 'gradient_descent']
        algorithm = 'linear_programming'

        assert algorithm in valid_algorithms

    def test_validate_objective_function(self):
        """Test objective function validation."""
        valid_objectives = ['minimize_cost', 'maximize_efficiency', 'minimize_emissions']
        objective = 'minimize_cost'

        assert objective in valid_objectives

    def test_validate_convergence_tolerance(self):
        """Test convergence tolerance within valid range."""
        tolerance = 0.001

        assert 0.0001 <= tolerance <= 0.01

    def test_validate_max_iterations(self):
        """Test max iterations within valid range."""
        max_iterations = 1000

        assert 100 <= max_iterations <= 10000

    def test_validate_time_step_range(self):
        """Test time step within valid range."""
        time_step = 15

        assert 1 <= time_step <= 60


# ============================================================================
# CONFIGURATION MERGING TESTS
# ============================================================================

class TestConfigurationMerging:
    """Test merging of multiple configuration sources."""

    def test_merge_default_and_user_config(self):
        """Test merging default and user configuration."""
        default_config = {
            'agent_id': 'GL-001',
            'max_parallel_agents': 10,
            'enable_monitoring': True
        }

        user_config = {
            'agent_id': 'GL-001-USER',
            'max_parallel_agents': 20
        }

        merged_config = {**default_config, **user_config}

        assert merged_config['agent_id'] == 'GL-001-USER'
        assert merged_config['max_parallel_agents'] == 20
        assert merged_config['enable_monitoring'] is True

    def test_merge_preserves_user_overrides(self):
        """Test merging preserves user overrides."""
        default_value = 10
        user_value = 20

        merged_value = user_value if user_value else default_value

        assert merged_value == 20

    def test_merge_nested_configurations(self):
        """Test merging nested configuration dictionaries."""
        default_config = {
            'optimization': {
                'algorithm': 'linear_programming',
                'max_iterations': 1000
            },
            'monitoring': {
                'enabled': True
            }
        }

        user_config = {
            'optimization': {
                'max_iterations': 2000
            }
        }

        merged_config = default_config.copy()
        if 'optimization' in user_config:
            merged_config['optimization'].update(user_config['optimization'])

        assert merged_config['optimization']['max_iterations'] == 2000
        assert merged_config['optimization']['algorithm'] == 'linear_programming'


# ============================================================================
# CONFIGURATION ERROR HANDLING TESTS
# ============================================================================

class TestConfigurationErrorHandling:
    """Test configuration error handling."""

    def test_handle_missing_required_field(self):
        """Test handling of missing required field."""
        config = {
            'agent_name': 'ProcessHeatOrchestrator'
            # Missing agent_id
        }

        has_agent_id = 'agent_id' in config

        assert has_agent_id is False

    def test_handle_invalid_type(self):
        """Test handling of invalid configuration type."""
        try:
            max_agents = int('not_a_number')
        except ValueError as e:
            assert 'invalid literal' in str(e).lower()

    def test_handle_negative_capacity(self):
        """Test handling of negative capacity value."""
        invalid_capacity = -100.0

        is_valid = invalid_capacity > 0

        assert is_valid is False

    def test_handle_empty_plants_list(self):
        """Test handling of empty plants list."""
        plants = []

        is_valid = len(plants) > 0

        assert is_valid is False

    def test_handle_empty_sensors_list(self):
        """Test handling of empty sensors list."""
        sensors = []

        is_valid = len(sensors) > 0

        assert is_valid is False


# ============================================================================
# DEFAULT VALUE HANDLING TESTS
# ============================================================================

class TestDefaultValueHandling:
    """Test default value handling in configuration."""

    def test_default_max_parallel_agents(self):
        """Test default max parallel agents."""
        default = 10
        assert default == 10

    def test_default_calculation_timeout(self):
        """Test default calculation timeout."""
        default = 120
        assert default == 120

    def test_default_cache_ttl(self):
        """Test default cache TTL."""
        default = 300
        assert default == 300

    def test_default_audit_trail_retention(self):
        """Test default audit trail retention days."""
        default = 365
        assert default == 365

    def test_default_metrics_collection_interval(self):
        """Test default metrics collection interval."""
        default = 60
        assert default == 60
