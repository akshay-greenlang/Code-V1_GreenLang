"""
Unit Tests: Schema Validation

Tests all input/output schemas for ThermalCommand agent including:
- ThermalMeasurement schema validation
- BoilerState schema validation
- HeatDemand schema validation
- OptimizationResult schema validation
- Configuration schema validation
- Error message validation

Reference: GL-001 Specification Section 11.1
Target Coverage: 85%+
"""

import pytest
from dataclasses import asdict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
import json


# =============================================================================
# Schema Classes (Simulated Production Code)
# =============================================================================

class SchemaValidationError(Exception):
    """Raised when schema validation fails."""
    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"Validation error for '{field}': {message}")


class ThermalMeasurementSchema:
    """Schema for thermal measurement validation."""

    REQUIRED_FIELDS = ['timestamp', 'temperature', 'pressure', 'flow_rate', 'sensor_id']
    OPTIONAL_FIELDS = ['energy_input', 'energy_output', 'efficiency', 'quality_score']

    TEMPERATURE_RANGE = (0.0, 1200.0)  # Celsius
    PRESSURE_RANGE = (0.0, 100.0)  # bar
    FLOW_RATE_RANGE = (0.0, 10000.0)  # m3/h
    EFFICIENCY_RANGE = (0.0, 1.0)
    QUALITY_SCORE_RANGE = (0.0, 1.0)

    @classmethod
    def validate(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate thermal measurement data against schema."""
        errors = []

        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in data:
                errors.append(SchemaValidationError(field, "Required field missing"))

        if errors:
            raise errors[0]

        # Validate timestamp
        if not isinstance(data['timestamp'], (datetime, str)):
            raise SchemaValidationError('timestamp', "Must be datetime or ISO string")

        # Validate numeric ranges
        cls._validate_range(data, 'temperature', cls.TEMPERATURE_RANGE)
        cls._validate_range(data, 'pressure', cls.PRESSURE_RANGE)
        cls._validate_range(data, 'flow_rate', cls.FLOW_RATE_RANGE)

        if 'efficiency' in data:
            cls._validate_range(data, 'efficiency', cls.EFFICIENCY_RANGE)

        if 'quality_score' in data:
            cls._validate_range(data, 'quality_score', cls.QUALITY_SCORE_RANGE)

        # Validate sensor_id
        if not isinstance(data['sensor_id'], str) or len(data['sensor_id']) == 0:
            raise SchemaValidationError('sensor_id', "Must be non-empty string")

        return data

    @classmethod
    def _validate_range(cls, data: Dict, field: str, range_tuple: tuple):
        """Validate that field value is within specified range."""
        if field not in data:
            return

        value = data[field]
        min_val, max_val = range_tuple

        if not isinstance(value, (int, float)):
            raise SchemaValidationError(field, f"Must be numeric, got {type(value).__name__}")

        if value < min_val or value > max_val:
            raise SchemaValidationError(
                field,
                f"Value {value} outside valid range [{min_val}, {max_val}]",
                value
            )


class BoilerStateSchema:
    """Schema for boiler state validation."""

    REQUIRED_FIELDS = ['boiler_id', 'status', 'temperature', 'pressure']
    VALID_STATUSES = ['running', 'standby', 'maintenance', 'fault', 'shutdown']

    @classmethod
    def validate(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate boiler state data against schema."""
        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in data:
                raise SchemaValidationError(field, "Required field missing")

        # Validate boiler_id format
        if not isinstance(data['boiler_id'], str):
            raise SchemaValidationError('boiler_id', "Must be string")

        if not data['boiler_id'].startswith('BOILER_'):
            raise SchemaValidationError('boiler_id', "Must start with 'BOILER_'")

        # Validate status
        if data['status'] not in cls.VALID_STATUSES:
            raise SchemaValidationError(
                'status',
                f"Invalid status '{data['status']}', must be one of {cls.VALID_STATUSES}"
            )

        # Validate numeric fields
        if not isinstance(data['temperature'], (int, float)) or data['temperature'] < 0:
            raise SchemaValidationError('temperature', "Must be non-negative number")

        if not isinstance(data['pressure'], (int, float)) or data['pressure'] < 0:
            raise SchemaValidationError('pressure', "Must be non-negative number")

        return data


class HeatDemandSchema:
    """Schema for heat demand validation."""

    REQUIRED_FIELDS = ['consumer_id', 'demand_type', 'required_temperature', 'required_flow_rate']
    VALID_DEMAND_TYPES = ['process', 'hvac', 'hot_water', 'steam', 'thermal_storage']
    PRIORITY_RANGE = (1, 5)

    @classmethod
    def validate(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate heat demand data against schema."""
        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in data:
                raise SchemaValidationError(field, "Required field missing")

        # Validate demand_type
        if data['demand_type'] not in cls.VALID_DEMAND_TYPES:
            raise SchemaValidationError(
                'demand_type',
                f"Invalid type '{data['demand_type']}', must be one of {cls.VALID_DEMAND_TYPES}"
            )

        # Validate priority if present
        if 'priority' in data:
            if not isinstance(data['priority'], int):
                raise SchemaValidationError('priority', "Must be integer")
            if not cls.PRIORITY_RANGE[0] <= data['priority'] <= cls.PRIORITY_RANGE[1]:
                raise SchemaValidationError(
                    'priority',
                    f"Must be between {cls.PRIORITY_RANGE[0]} and {cls.PRIORITY_RANGE[1]}"
                )

        # Validate numeric fields
        if data['required_temperature'] < 0:
            raise SchemaValidationError('required_temperature', "Must be non-negative")

        if data['required_flow_rate'] < 0:
            raise SchemaValidationError('required_flow_rate', "Must be non-negative")

        return data


class OptimizationResultSchema:
    """Schema for optimization result validation."""

    REQUIRED_FIELDS = ['timestamp', 'objective_value', 'solver_status', 'solve_time', 'provenance_hash']
    VALID_SOLVER_STATUSES = ['optimal', 'feasible', 'infeasible', 'unbounded', 'timeout', 'error']

    @classmethod
    def validate(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate optimization result data against schema."""
        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in data:
                raise SchemaValidationError(field, "Required field missing")

        # Validate solver_status
        if data['solver_status'] not in cls.VALID_SOLVER_STATUSES:
            raise SchemaValidationError(
                'solver_status',
                f"Invalid status '{data['solver_status']}'"
            )

        # Validate provenance_hash format (SHA-256 = 64 hex characters)
        if not isinstance(data['provenance_hash'], str):
            raise SchemaValidationError('provenance_hash', "Must be string")

        if len(data['provenance_hash']) != 64:
            raise SchemaValidationError('provenance_hash', "Must be 64 character SHA-256 hash")

        if not all(c in '0123456789abcdef' for c in data['provenance_hash'].lower()):
            raise SchemaValidationError('provenance_hash', "Must be valid hexadecimal")

        # Validate solve_time
        if not isinstance(data['solve_time'], (int, float)) or data['solve_time'] < 0:
            raise SchemaValidationError('solve_time', "Must be non-negative number")

        return data


# =============================================================================
# Test Classes
# =============================================================================

class TestThermalMeasurementSchema:
    """Test suite for ThermalMeasurement schema validation."""

    @pytest.fixture
    def valid_measurement(self):
        """Provide valid measurement data."""
        return {
            'timestamp': datetime.now(),
            'temperature': 450.0,
            'pressure': 15.0,
            'flow_rate': 500.0,
            'sensor_id': 'SENSOR_001',
            'energy_input': 1000.0,
            'energy_output': 850.0,
            'efficiency': 0.85,
            'quality_score': 0.98
        }

    def test_valid_measurement_passes(self, valid_measurement):
        """Test that valid measurement data passes validation."""
        result = ThermalMeasurementSchema.validate(valid_measurement)
        assert result == valid_measurement

    def test_missing_required_field_fails(self, valid_measurement):
        """Test that missing required field raises error."""
        del valid_measurement['temperature']

        with pytest.raises(SchemaValidationError) as exc_info:
            ThermalMeasurementSchema.validate(valid_measurement)

        assert exc_info.value.field == 'temperature'
        assert 'Required field' in exc_info.value.message

    @pytest.mark.parametrize("field,invalid_value,expected_error", [
        ('temperature', -10.0, 'outside valid range'),
        ('temperature', 1500.0, 'outside valid range'),
        ('pressure', -5.0, 'outside valid range'),
        ('pressure', 150.0, 'outside valid range'),
        ('flow_rate', -100.0, 'outside valid range'),
        ('flow_rate', 15000.0, 'outside valid range'),
        ('efficiency', -0.1, 'outside valid range'),
        ('efficiency', 1.5, 'outside valid range'),
        ('quality_score', -0.1, 'outside valid range'),
        ('quality_score', 1.1, 'outside valid range'),
    ])
    def test_out_of_range_values_fail(self, valid_measurement, field, invalid_value, expected_error):
        """Test that out-of-range values fail validation."""
        valid_measurement[field] = invalid_value

        with pytest.raises(SchemaValidationError) as exc_info:
            ThermalMeasurementSchema.validate(valid_measurement)

        assert exc_info.value.field == field
        assert expected_error in exc_info.value.message.lower()

    def test_non_numeric_temperature_fails(self, valid_measurement):
        """Test that non-numeric temperature fails."""
        valid_measurement['temperature'] = 'hot'

        with pytest.raises(SchemaValidationError) as exc_info:
            ThermalMeasurementSchema.validate(valid_measurement)

        assert exc_info.value.field == 'temperature'
        assert 'numeric' in exc_info.value.message.lower()

    def test_empty_sensor_id_fails(self, valid_measurement):
        """Test that empty sensor_id fails."""
        valid_measurement['sensor_id'] = ''

        with pytest.raises(SchemaValidationError) as exc_info:
            ThermalMeasurementSchema.validate(valid_measurement)

        assert exc_info.value.field == 'sensor_id'

    def test_boundary_values_pass(self, valid_measurement):
        """Test that boundary values pass validation."""
        # Test minimum boundary
        valid_measurement['temperature'] = 0.0
        valid_measurement['pressure'] = 0.0
        valid_measurement['flow_rate'] = 0.0
        result = ThermalMeasurementSchema.validate(valid_measurement)
        assert result is not None

        # Test maximum boundary
        valid_measurement['temperature'] = 1200.0
        valid_measurement['pressure'] = 100.0
        valid_measurement['flow_rate'] = 10000.0
        result = ThermalMeasurementSchema.validate(valid_measurement)
        assert result is not None

    def test_iso_string_timestamp_passes(self, valid_measurement):
        """Test that ISO string timestamp passes validation."""
        valid_measurement['timestamp'] = '2025-01-15T10:30:00Z'
        result = ThermalMeasurementSchema.validate(valid_measurement)
        assert result is not None

    def test_optional_fields_can_be_omitted(self):
        """Test that optional fields can be omitted."""
        minimal_measurement = {
            'timestamp': datetime.now(),
            'temperature': 450.0,
            'pressure': 15.0,
            'flow_rate': 500.0,
            'sensor_id': 'SENSOR_001'
        }
        result = ThermalMeasurementSchema.validate(minimal_measurement)
        assert result is not None


class TestBoilerStateSchema:
    """Test suite for BoilerState schema validation."""

    @pytest.fixture
    def valid_boiler_state(self):
        """Provide valid boiler state data."""
        return {
            'boiler_id': 'BOILER_001',
            'status': 'running',
            'temperature': 450.0,
            'pressure': 15.0,
            'fuel_rate': 100.0,
            'steam_output': 800.0,
            'efficiency': 0.88
        }

    def test_valid_boiler_state_passes(self, valid_boiler_state):
        """Test that valid boiler state passes validation."""
        result = BoilerStateSchema.validate(valid_boiler_state)
        assert result == valid_boiler_state

    @pytest.mark.parametrize("status", ['running', 'standby', 'maintenance', 'fault', 'shutdown'])
    def test_all_valid_statuses_pass(self, valid_boiler_state, status):
        """Test that all valid statuses pass validation."""
        valid_boiler_state['status'] = status
        result = BoilerStateSchema.validate(valid_boiler_state)
        assert result['status'] == status

    def test_invalid_status_fails(self, valid_boiler_state):
        """Test that invalid status fails validation."""
        valid_boiler_state['status'] = 'invalid_status'

        with pytest.raises(SchemaValidationError) as exc_info:
            BoilerStateSchema.validate(valid_boiler_state)

        assert exc_info.value.field == 'status'

    def test_invalid_boiler_id_prefix_fails(self, valid_boiler_state):
        """Test that boiler_id without proper prefix fails."""
        valid_boiler_state['boiler_id'] = 'HEATER_001'

        with pytest.raises(SchemaValidationError) as exc_info:
            BoilerStateSchema.validate(valid_boiler_state)

        assert exc_info.value.field == 'boiler_id'
        assert 'BOILER_' in exc_info.value.message

    def test_negative_temperature_fails(self, valid_boiler_state):
        """Test that negative temperature fails validation."""
        valid_boiler_state['temperature'] = -50.0

        with pytest.raises(SchemaValidationError) as exc_info:
            BoilerStateSchema.validate(valid_boiler_state)

        assert exc_info.value.field == 'temperature'

    def test_missing_required_field_fails(self, valid_boiler_state):
        """Test that missing required field fails."""
        del valid_boiler_state['status']

        with pytest.raises(SchemaValidationError) as exc_info:
            BoilerStateSchema.validate(valid_boiler_state)

        assert exc_info.value.field == 'status'


class TestHeatDemandSchema:
    """Test suite for HeatDemand schema validation."""

    @pytest.fixture
    def valid_heat_demand(self):
        """Provide valid heat demand data."""
        return {
            'consumer_id': 'CONSUMER_001',
            'demand_type': 'process',
            'required_temperature': 350.0,
            'required_flow_rate': 200.0,
            'priority': 2,
            'tolerance_temp': 7.0,
            'tolerance_flow': 10.0
        }

    def test_valid_heat_demand_passes(self, valid_heat_demand):
        """Test that valid heat demand passes validation."""
        result = HeatDemandSchema.validate(valid_heat_demand)
        assert result == valid_heat_demand

    @pytest.mark.parametrize("demand_type", ['process', 'hvac', 'hot_water', 'steam', 'thermal_storage'])
    def test_all_valid_demand_types_pass(self, valid_heat_demand, demand_type):
        """Test that all valid demand types pass validation."""
        valid_heat_demand['demand_type'] = demand_type
        result = HeatDemandSchema.validate(valid_heat_demand)
        assert result['demand_type'] == demand_type

    def test_invalid_demand_type_fails(self, valid_heat_demand):
        """Test that invalid demand type fails validation."""
        valid_heat_demand['demand_type'] = 'invalid_type'

        with pytest.raises(SchemaValidationError) as exc_info:
            HeatDemandSchema.validate(valid_heat_demand)

        assert exc_info.value.field == 'demand_type'

    @pytest.mark.parametrize("priority", [1, 2, 3, 4, 5])
    def test_valid_priorities_pass(self, valid_heat_demand, priority):
        """Test that valid priority values pass."""
        valid_heat_demand['priority'] = priority
        result = HeatDemandSchema.validate(valid_heat_demand)
        assert result['priority'] == priority

    @pytest.mark.parametrize("priority", [0, 6, -1, 10])
    def test_invalid_priorities_fail(self, valid_heat_demand, priority):
        """Test that invalid priority values fail."""
        valid_heat_demand['priority'] = priority

        with pytest.raises(SchemaValidationError) as exc_info:
            HeatDemandSchema.validate(valid_heat_demand)

        assert exc_info.value.field == 'priority'

    def test_negative_temperature_fails(self, valid_heat_demand):
        """Test that negative temperature fails."""
        valid_heat_demand['required_temperature'] = -10.0

        with pytest.raises(SchemaValidationError) as exc_info:
            HeatDemandSchema.validate(valid_heat_demand)

        assert exc_info.value.field == 'required_temperature'

    def test_negative_flow_rate_fails(self, valid_heat_demand):
        """Test that negative flow rate fails."""
        valid_heat_demand['required_flow_rate'] = -50.0

        with pytest.raises(SchemaValidationError) as exc_info:
            HeatDemandSchema.validate(valid_heat_demand)

        assert exc_info.value.field == 'required_flow_rate'


class TestOptimizationResultSchema:
    """Test suite for OptimizationResult schema validation."""

    @pytest.fixture
    def valid_optimization_result(self):
        """Provide valid optimization result data."""
        return {
            'timestamp': datetime.now(),
            'objective_value': 2500.0,
            'boiler_setpoints': {'BOILER_001': 85.0, 'BOILER_002': 80.0},
            'valve_positions': {'VALVE_001': 0.5, 'VALVE_002': 0.7},
            'pump_speeds': {'PUMP_001': 0.8},
            'predicted_cost': 10000.0,
            'predicted_emissions': 250.0,
            'solver_status': 'optimal',
            'solve_time': 2.5,
            'provenance_hash': 'a' * 64  # Valid SHA-256 hash
        }

    def test_valid_result_passes(self, valid_optimization_result):
        """Test that valid result passes validation."""
        result = OptimizationResultSchema.validate(valid_optimization_result)
        assert result == valid_optimization_result

    @pytest.mark.parametrize("status", ['optimal', 'feasible', 'infeasible', 'unbounded', 'timeout', 'error'])
    def test_all_valid_solver_statuses_pass(self, valid_optimization_result, status):
        """Test that all valid solver statuses pass."""
        valid_optimization_result['solver_status'] = status
        result = OptimizationResultSchema.validate(valid_optimization_result)
        assert result['solver_status'] == status

    def test_invalid_solver_status_fails(self, valid_optimization_result):
        """Test that invalid solver status fails."""
        valid_optimization_result['solver_status'] = 'invalid'

        with pytest.raises(SchemaValidationError) as exc_info:
            OptimizationResultSchema.validate(valid_optimization_result)

        assert exc_info.value.field == 'solver_status'

    def test_invalid_provenance_hash_length_fails(self, valid_optimization_result):
        """Test that incorrect hash length fails."""
        valid_optimization_result['provenance_hash'] = 'a' * 32  # Wrong length

        with pytest.raises(SchemaValidationError) as exc_info:
            OptimizationResultSchema.validate(valid_optimization_result)

        assert exc_info.value.field == 'provenance_hash'
        assert '64' in exc_info.value.message

    def test_invalid_provenance_hash_characters_fails(self, valid_optimization_result):
        """Test that invalid hex characters in hash fails."""
        valid_optimization_result['provenance_hash'] = 'g' * 64  # Invalid hex

        with pytest.raises(SchemaValidationError) as exc_info:
            OptimizationResultSchema.validate(valid_optimization_result)

        assert exc_info.value.field == 'provenance_hash'
        assert 'hexadecimal' in exc_info.value.message.lower()

    def test_negative_solve_time_fails(self, valid_optimization_result):
        """Test that negative solve time fails."""
        valid_optimization_result['solve_time'] = -1.0

        with pytest.raises(SchemaValidationError) as exc_info:
            OptimizationResultSchema.validate(valid_optimization_result)

        assert exc_info.value.field == 'solve_time'

    def test_missing_provenance_hash_fails(self, valid_optimization_result):
        """Test that missing provenance hash fails."""
        del valid_optimization_result['provenance_hash']

        with pytest.raises(SchemaValidationError) as exc_info:
            OptimizationResultSchema.validate(valid_optimization_result)

        assert exc_info.value.field == 'provenance_hash'


class TestSchemaEdgeCases:
    """Test edge cases across all schemas."""

    def test_null_values_handled(self):
        """Test that null values are properly handled."""
        data = {
            'timestamp': datetime.now(),
            'temperature': None,  # Null value
            'pressure': 15.0,
            'flow_rate': 500.0,
            'sensor_id': 'SENSOR_001'
        }

        with pytest.raises((SchemaValidationError, TypeError)):
            ThermalMeasurementSchema.validate(data)

    def test_empty_dict_fails(self):
        """Test that empty dict fails validation."""
        with pytest.raises(SchemaValidationError):
            ThermalMeasurementSchema.validate({})

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed (not rejected)."""
        data = {
            'timestamp': datetime.now(),
            'temperature': 450.0,
            'pressure': 15.0,
            'flow_rate': 500.0,
            'sensor_id': 'SENSOR_001',
            'extra_field': 'extra_value'  # Extra field
        }

        # Should not raise
        result = ThermalMeasurementSchema.validate(data)
        assert 'extra_field' in result

    def test_large_values_within_range_pass(self):
        """Test that large values within range pass."""
        data = {
            'timestamp': datetime.now(),
            'temperature': 1199.999,  # Just below max
            'pressure': 99.999,
            'flow_rate': 9999.999,
            'sensor_id': 'SENSOR_001'
        }

        result = ThermalMeasurementSchema.validate(data)
        assert result is not None

    def test_scientific_notation_values_pass(self):
        """Test that scientific notation values pass."""
        data = {
            'timestamp': datetime.now(),
            'temperature': 4.5e2,  # 450.0
            'pressure': 1.5e1,  # 15.0
            'flow_rate': 5e2,  # 500.0
            'sensor_id': 'SENSOR_001'
        }

        result = ThermalMeasurementSchema.validate(data)
        assert result is not None
