"""
Unit tests for GL-005 CombustionControlAgent tool schemas and validation.

Tests tool schema validation, input validation, output validation,
and error handling for all agent tools.

Target: 12+ tests covering:
- Tool schema validation
- Input parameter validation
- Output format validation
- Error handling
- Type checking
- Boundary conditions
"""

import pytest
from typing import Dict, Any
from pydantic import ValidationError, BaseModel
from datetime import datetime, timezone

pytestmark = pytest.mark.unit


# ============================================================================
# MOCK TOOL SCHEMAS (Pydantic Models)
# ============================================================================

class ControlCycleInput(BaseModel):
    """Input schema for control cycle execution."""
    controller_id: str
    timestamp: datetime
    fuel_flow_setpoint: float
    air_flow_setpoint: float
    temperature_setpoint: float

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ControlCycleOutput(BaseModel):
    """Output schema for control cycle execution."""
    cycle_id: str
    success: bool
    fuel_flow_actual: float
    air_flow_actual: float
    temperature_actual: float
    stability_index: float
    provenance_hash: str


class SafetyCheckInput(BaseModel):
    """Input schema for safety validation."""
    temperature_c: float
    pressure_mbar: float
    fuel_flow_kg_hr: float
    co_ppm: float
    flame_detected: bool


class SafetyCheckOutput(BaseModel):
    """Output schema for safety validation."""
    is_safe: bool
    violations: list[str]
    action_required: str
    timestamp: datetime


# ============================================================================
# TOOL INPUT VALIDATION TESTS
# ============================================================================

class TestToolInputValidation:
    """Test tool input validation."""

    def test_control_cycle_input_valid(self):
        """Test valid control cycle input passes validation."""
        input_data = {
            'controller_id': 'CC-001',
            'timestamp': datetime.now(timezone.utc),
            'fuel_flow_setpoint': 500.0,
            'air_flow_setpoint': 5000.0,
            'temperature_setpoint': 1200.0
        }

        tool_input = ControlCycleInput(**input_data)

        assert tool_input.controller_id == 'CC-001'
        assert tool_input.fuel_flow_setpoint == 500.0
        assert isinstance(tool_input.timestamp, datetime)

    def test_control_cycle_input_missing_required_field(self):
        """Test control cycle input validation fails with missing field."""
        input_data = {
            'controller_id': 'CC-001',
            # Missing timestamp
            'fuel_flow_setpoint': 500.0,
            'air_flow_setpoint': 5000.0,
            'temperature_setpoint': 1200.0
        }

        with pytest.raises(ValidationError) as exc_info:
            ControlCycleInput(**input_data)

        assert 'timestamp' in str(exc_info.value)

    def test_control_cycle_input_invalid_type(self):
        """Test control cycle input validation fails with wrong type."""
        input_data = {
            'controller_id': 'CC-001',
            'timestamp': datetime.now(timezone.utc),
            'fuel_flow_setpoint': 'not_a_number',  # Invalid type
            'air_flow_setpoint': 5000.0,
            'temperature_setpoint': 1200.0
        }

        with pytest.raises(ValidationError) as exc_info:
            ControlCycleInput(**input_data)

        assert 'fuel_flow_setpoint' in str(exc_info.value)

    def test_safety_check_input_valid(self):
        """Test valid safety check input passes validation."""
        input_data = {
            'temperature_c': 1200.0,
            'pressure_mbar': 100.0,
            'fuel_flow_kg_hr': 500.0,
            'co_ppm': 25.0,
            'flame_detected': True
        }

        tool_input = SafetyCheckInput(**input_data)

        assert tool_input.temperature_c == 1200.0
        assert tool_input.flame_detected is True

    def test_safety_check_input_negative_values(self):
        """Test safety check handles negative values."""
        input_data = {
            'temperature_c': -100.0,  # Negative temperature
            'pressure_mbar': 100.0,
            'fuel_flow_kg_hr': 500.0,
            'co_ppm': 25.0,
            'flame_detected': True
        }

        # Validation should still accept it (business logic will reject)
        tool_input = SafetyCheckInput(**input_data)
        assert tool_input.temperature_c == -100.0


# ============================================================================
# TOOL OUTPUT VALIDATION TESTS
# ============================================================================

class TestToolOutputValidation:
    """Test tool output validation."""

    def test_control_cycle_output_valid(self):
        """Test valid control cycle output passes validation."""
        output_data = {
            'cycle_id': 'CYCLE-001',
            'success': True,
            'fuel_flow_actual': 505.0,
            'air_flow_actual': 5050.0,
            'temperature_actual': 1205.0,
            'stability_index': 0.95,
            'provenance_hash': 'a' * 64  # SHA-256 hash
        }

        tool_output = ControlCycleOutput(**output_data)

        assert tool_output.success is True
        assert tool_output.stability_index == 0.95
        assert len(tool_output.provenance_hash) == 64

    def test_control_cycle_output_missing_field(self):
        """Test control cycle output validation fails with missing field."""
        output_data = {
            'cycle_id': 'CYCLE-001',
            'success': True,
            # Missing actual values
            'stability_index': 0.95,
            'provenance_hash': 'a' * 64
        }

        with pytest.raises(ValidationError):
            ControlCycleOutput(**output_data)

    def test_safety_check_output_valid(self):
        """Test valid safety check output passes validation."""
        output_data = {
            'is_safe': True,
            'violations': [],
            'action_required': 'none',
            'timestamp': datetime.now(timezone.utc)
        }

        tool_output = SafetyCheckOutput(**output_data)

        assert tool_output.is_safe is True
        assert len(tool_output.violations) == 0

    def test_safety_check_output_with_violations(self):
        """Test safety check output with violations."""
        output_data = {
            'is_safe': False,
            'violations': ['HIGH_TEMPERATURE', 'HIGH_CO'],
            'action_required': 'emergency_shutdown',
            'timestamp': datetime.now(timezone.utc)
        }

        tool_output = SafetyCheckOutput(**output_data)

        assert tool_output.is_safe is False
        assert len(tool_output.violations) == 2
        assert 'HIGH_TEMPERATURE' in tool_output.violations


# ============================================================================
# TOOL SCHEMA TYPE CHECKING TESTS
# ============================================================================

class TestToolSchemaTypeChecking:
    """Test tool schema type checking."""

    def test_schema_enforces_string_type(self):
        """Test schema enforces string type for controller_id."""
        with pytest.raises(ValidationError) as exc_info:
            ControlCycleInput(
                controller_id=12345,  # Should be string
                timestamp=datetime.now(timezone.utc),
                fuel_flow_setpoint=500.0,
                air_flow_setpoint=5000.0,
                temperature_setpoint=1200.0
            )

        assert 'controller_id' in str(exc_info.value)

    def test_schema_enforces_float_type(self):
        """Test schema enforces float type for numeric fields."""
        # Should accept int and convert to float
        tool_input = ControlCycleInput(
            controller_id='CC-001',
            timestamp=datetime.now(timezone.utc),
            fuel_flow_setpoint=500,  # int, should convert to float
            air_flow_setpoint=5000,
            temperature_setpoint=1200
        )

        assert isinstance(tool_input.fuel_flow_setpoint, float)

    def test_schema_enforces_boolean_type(self):
        """Test schema enforces boolean type."""
        with pytest.raises(ValidationError):
            SafetyCheckInput(
                temperature_c=1200.0,
                pressure_mbar=100.0,
                fuel_flow_kg_hr=500.0,
                co_ppm=25.0,
                flame_detected='yes'  # Should be bool
            )

    def test_schema_enforces_datetime_type(self):
        """Test schema enforces datetime type."""
        with pytest.raises(ValidationError):
            ControlCycleInput(
                controller_id='CC-001',
                timestamp='2025-01-01',  # Should be datetime object
                fuel_flow_setpoint=500.0,
                air_flow_setpoint=5000.0,
                temperature_setpoint=1200.0
            )


# ============================================================================
# TOOL ERROR HANDLING TESTS
# ============================================================================

class TestToolErrorHandling:
    """Test tool error handling."""

    def test_handle_none_value_in_required_field(self):
        """Test handling of None in required field."""
        with pytest.raises(ValidationError):
            ControlCycleInput(
                controller_id=None,  # Required field
                timestamp=datetime.now(timezone.utc),
                fuel_flow_setpoint=500.0,
                air_flow_setpoint=5000.0,
                temperature_setpoint=1200.0
            )

    def test_handle_empty_string_in_required_field(self):
        """Test handling of empty string in required field."""
        # Empty string is valid for string type
        tool_input = ControlCycleInput(
            controller_id='',  # Empty but valid
            timestamp=datetime.now(timezone.utc),
            fuel_flow_setpoint=500.0,
            air_flow_setpoint=5000.0,
            temperature_setpoint=1200.0
        )

        assert tool_input.controller_id == ''

    def test_handle_infinity_value(self):
        """Test handling of infinity value."""
        tool_input = SafetyCheckInput(
            temperature_c=float('inf'),  # Infinity
            pressure_mbar=100.0,
            fuel_flow_kg_hr=500.0,
            co_ppm=25.0,
            flame_detected=True
        )

        assert tool_input.temperature_c == float('inf')

    def test_handle_nan_value(self):
        """Test handling of NaN value."""
        import math

        tool_input = SafetyCheckInput(
            temperature_c=float('nan'),  # NaN
            pressure_mbar=100.0,
            fuel_flow_kg_hr=500.0,
            co_ppm=25.0,
            flame_detected=True
        )

        assert math.isnan(tool_input.temperature_c)


# ============================================================================
# TOOL BOUNDARY CONDITION TESTS
# ============================================================================

@pytest.mark.boundary
class TestToolBoundaryConditions:
    """Test tool handling of boundary conditions."""

    def test_zero_value_acceptance(self):
        """Test tool accepts zero values."""
        tool_input = SafetyCheckInput(
            temperature_c=0.0,
            pressure_mbar=0.0,
            fuel_flow_kg_hr=0.0,
            co_ppm=0.0,
            flame_detected=False
        )

        assert tool_input.fuel_flow_kg_hr == 0.0

    def test_very_large_value_acceptance(self):
        """Test tool accepts very large values."""
        tool_input = ControlCycleInput(
            controller_id='CC-001',
            timestamp=datetime.now(timezone.utc),
            fuel_flow_setpoint=1e10,
            air_flow_setpoint=1e10,
            temperature_setpoint=1e10
        )

        assert tool_input.fuel_flow_setpoint == 1e10

    def test_very_small_value_acceptance(self):
        """Test tool accepts very small values."""
        tool_input = ControlCycleInput(
            controller_id='CC-001',
            timestamp=datetime.now(timezone.utc),
            fuel_flow_setpoint=1e-10,
            air_flow_setpoint=1e-10,
            temperature_setpoint=1e-10
        )

        assert tool_input.fuel_flow_setpoint == 1e-10

    def test_negative_value_acceptance(self):
        """Test tool accepts negative values (business logic may reject)."""
        tool_input = SafetyCheckInput(
            temperature_c=-273.15,  # Absolute zero
            pressure_mbar=-100.0,
            fuel_flow_kg_hr=-500.0,
            co_ppm=-25.0,
            flame_detected=True
        )

        assert tool_input.temperature_c == -273.15


# ============================================================================
# TOOL SERIALIZATION TESTS
# ============================================================================

class TestToolSerialization:
    """Test tool input/output serialization."""

    def test_tool_input_to_dict(self):
        """Test tool input serialization to dict."""
        timestamp = datetime.now(timezone.utc)
        tool_input = ControlCycleInput(
            controller_id='CC-001',
            timestamp=timestamp,
            fuel_flow_setpoint=500.0,
            air_flow_setpoint=5000.0,
            temperature_setpoint=1200.0
        )

        input_dict = tool_input.model_dump()

        assert input_dict['controller_id'] == 'CC-001'
        assert input_dict['fuel_flow_setpoint'] == 500.0

    def test_tool_output_to_json(self):
        """Test tool output serialization to JSON."""
        tool_output = SafetyCheckOutput(
            is_safe=True,
            violations=[],
            action_required='none',
            timestamp=datetime.now(timezone.utc)
        )

        output_json = tool_output.model_dump_json()

        assert 'is_safe' in output_json
        assert 'true' in output_json.lower()

    def test_tool_input_from_dict(self):
        """Test tool input deserialization from dict."""
        input_dict = {
            'controller_id': 'CC-001',
            'timestamp': datetime.now(timezone.utc),
            'fuel_flow_setpoint': 500.0,
            'air_flow_setpoint': 5000.0,
            'temperature_setpoint': 1200.0
        }

        tool_input = ControlCycleInput(**input_dict)

        assert tool_input.controller_id == 'CC-001'
