"""
SafetyFunction - Safety Function Modeling and Specification

This module implements Safety Instrumented Function (SIF) modeling
per IEC 61511-1. A Safety Function is the complete implementation
of a safety requirement, including sensors, logic solver, and actuators.

Key concepts:
- Safety Function: Complete SIF from sensor to actuator
- Input Sensor: Field devices detecting hazardous conditions
- Logic Solver: Safety controller processing inputs
- Output Actuator: Final elements achieving safe state

Reference: IEC 61511-1 Clause 10, Clause 11

Example:
    >>> from greenlang.safety.srs.safety_function import SafetyFunction
    >>> sf = SafetyFunction(
    ...     function_id="SIF-001",
    ...     name="High Pressure Shutdown",
    ...     sil_level=2
    ... )
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import hashlib
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class SensorType(str, Enum):
    """Types of safety sensors per IEC 61511."""

    PRESSURE_TRANSMITTER = "pressure_transmitter"
    TEMPERATURE_TRANSMITTER = "temperature_transmitter"
    LEVEL_TRANSMITTER = "level_transmitter"
    FLOW_TRANSMITTER = "flow_transmitter"
    FLAME_DETECTOR = "flame_detector"
    GAS_DETECTOR = "gas_detector"
    VIBRATION_SENSOR = "vibration_sensor"
    POSITION_SWITCH = "position_switch"
    SPEED_SENSOR = "speed_sensor"
    ANALYZER = "analyzer"


class ActuatorType(str, Enum):
    """Types of safety actuators per IEC 61511."""

    SHUTDOWN_VALVE = "shutdown_valve"  # SDV
    BLOWDOWN_VALVE = "blowdown_valve"  # BDV
    EMERGENCY_ISOLATION_VALVE = "emergency_isolation_valve"  # EIV
    CONTROL_VALVE_TRIP = "control_valve_trip"
    PUMP_TRIP = "pump_trip"
    COMPRESSOR_TRIP = "compressor_trip"
    MOTOR_TRIP = "motor_trip"
    BURNER_TRIP = "burner_trip"
    ALARM = "alarm"
    INTERLOCK = "interlock"


class InputSensor(BaseModel):
    """Input sensor specification for a Safety Function."""

    sensor_id: str = Field(
        ...,
        description="Sensor identifier/tag"
    )
    sensor_type: SensorType = Field(
        ...,
        description="Type of sensor"
    )
    description: str = Field(
        default="",
        description="Sensor description"
    )
    measurement_variable: str = Field(
        ...,
        description="Process variable measured"
    )
    engineering_units: str = Field(
        default="",
        description="Engineering units"
    )
    normal_range_low: float = Field(
        ...,
        description="Normal operating range low"
    )
    normal_range_high: float = Field(
        ...,
        description="Normal operating range high"
    )
    trip_setpoint: float = Field(
        ...,
        description="Trip setpoint value"
    )
    trip_direction: str = Field(
        default="high",
        description="Trip direction (high/low)"
    )
    response_time_ms: float = Field(
        default=100.0,
        gt=0,
        description="Sensor response time (ms)"
    )
    voting_position: str = Field(
        default="A",
        description="Position in voting (A, B, C for 2oo3)"
    )
    failure_rate_du: float = Field(
        default=1e-6,
        ge=0,
        description="Dangerous undetected failure rate (per hour)"
    )
    failure_rate_dd: float = Field(
        default=1e-6,
        ge=0,
        description="Dangerous detected failure rate (per hour)"
    )
    proof_test_coverage: float = Field(
        default=0.9,
        ge=0,
        le=1,
        description="Proof test coverage"
    )
    manufacturer: str = Field(
        default="",
        description="Sensor manufacturer"
    )
    model: str = Field(
        default="",
        description="Sensor model"
    )
    sil_capability: int = Field(
        default=2,
        ge=1,
        le=4,
        description="SIL capability per manufacturer"
    )


class OutputActuator(BaseModel):
    """Output actuator specification for a Safety Function."""

    actuator_id: str = Field(
        ...,
        description="Actuator identifier/tag"
    )
    actuator_type: ActuatorType = Field(
        ...,
        description="Type of actuator"
    )
    description: str = Field(
        default="",
        description="Actuator description"
    )
    safe_state: str = Field(
        ...,
        description="Safe state (e.g., 'CLOSED', 'OPEN', 'OFF')"
    )
    fail_safe_position: str = Field(
        ...,
        description="Position on air/power failure"
    )
    stroke_time_ms: float = Field(
        default=5000.0,
        gt=0,
        description="Full stroke time (ms)"
    )
    partial_stroke_capable: bool = Field(
        default=False,
        description="Supports partial stroke testing"
    )
    voting_position: str = Field(
        default="A",
        description="Position in voting (A, B for 1oo2)"
    )
    failure_rate_du: float = Field(
        default=2e-6,
        ge=0,
        description="Dangerous undetected failure rate (per hour)"
    )
    failure_rate_dd: float = Field(
        default=1e-6,
        ge=0,
        description="Dangerous detected failure rate (per hour)"
    )
    proof_test_coverage: float = Field(
        default=0.9,
        ge=0,
        le=1,
        description="Proof test coverage"
    )
    manufacturer: str = Field(
        default="",
        description="Actuator manufacturer"
    )
    model: str = Field(
        default="",
        description="Actuator model"
    )
    sil_capability: int = Field(
        default=2,
        ge=1,
        le=4,
        description="SIL capability per manufacturer"
    )


class LogicSolverSpec(BaseModel):
    """Logic solver specification."""

    solver_id: str = Field(
        ...,
        description="Logic solver identifier"
    )
    solver_type: str = Field(
        default="SIS_PLC",
        description="Type (SIS_PLC, relay, etc.)"
    )
    manufacturer: str = Field(
        default="",
        description="Manufacturer"
    )
    model: str = Field(
        default="",
        description="Model"
    )
    sil_capability: int = Field(
        default=3,
        ge=1,
        le=4,
        description="SIL capability"
    )
    scan_time_ms: float = Field(
        default=50.0,
        gt=0,
        description="Program scan time (ms)"
    )
    input_processing_time_ms: float = Field(
        default=20.0,
        gt=0,
        description="Input processing time (ms)"
    )
    output_processing_time_ms: float = Field(
        default=20.0,
        gt=0,
        description="Output processing time (ms)"
    )
    voting_logic: str = Field(
        default="1oo1",
        description="Voting logic (1oo1, 1oo2, 2oo3)"
    )
    failure_rate_du: float = Field(
        default=1e-7,
        ge=0,
        description="Dangerous undetected failure rate (per hour)"
    )


class SafetyFunctionSpec(BaseModel):
    """Complete Safety Function specification."""

    function_id: str = Field(
        ...,
        description="Safety function identifier"
    )
    name: str = Field(
        ...,
        description="Safety function name"
    )
    description: str = Field(
        default="",
        description="Detailed description"
    )
    sil_level: int = Field(
        ...,
        ge=1,
        le=4,
        description="Required SIL level"
    )
    sil_basis: str = Field(
        default="LOPA",
        description="SIL determination method (LOPA, risk graph)"
    )
    hazard_description: str = Field(
        default="",
        description="Hazard this function protects against"
    )
    initiating_cause: str = Field(
        default="",
        description="Initiating event/cause"
    )
    consequence: str = Field(
        default="",
        description="Potential consequence"
    )
    safe_state_id: str = Field(
        ...,
        description="Reference to safe state definition"
    )
    action_on_trip: str = Field(
        ...,
        description="Action taken on trip"
    )
    input_sensors: List[InputSensor] = Field(
        default_factory=list,
        description="Input sensors"
    )
    logic_solver: Optional[LogicSolverSpec] = Field(
        None,
        description="Logic solver specification"
    )
    output_actuators: List[OutputActuator] = Field(
        default_factory=list,
        description="Output actuators"
    )
    input_voting: str = Field(
        default="1oo1",
        description="Input voting logic"
    )
    output_voting: str = Field(
        default="1oo1",
        description="Output voting logic"
    )
    process_safety_time_ms: float = Field(
        ...,
        gt=0,
        description="Process safety time (ms)"
    )
    required_response_time_ms: float = Field(
        ...,
        gt=0,
        description="Required response time (ms)"
    )
    proof_test_interval_hours: float = Field(
        ...,
        gt=0,
        description="Proof test interval (hours)"
    )
    pfd_target: float = Field(
        ...,
        gt=0,
        lt=1,
        description="Target PFD average"
    )
    diagnostic_coverage_target: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="Target diagnostic coverage"
    )
    bypass_permitted: bool = Field(
        default=False,
        description="Bypass permitted"
    )
    manual_initiation: bool = Field(
        default=True,
        description="Manual initiation capability"
    )
    reset_type: str = Field(
        default="manual",
        description="Reset type (manual/automatic)"
    )
    created_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation date"
    )
    last_modified: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last modification date"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SafetyFunction:
    """
    Safety Function Manager.

    Manages Safety Instrumented Function (SIF) specifications
    per IEC 61511. Provides:
    - SIF specification creation
    - Response time analysis
    - PFD calculation inputs
    - Completeness validation

    The manager follows zero-hallucination principles:
    - All calculations are deterministic
    - Complete audit trail

    Example:
        >>> sf = SafetyFunction()
        >>> spec = sf.create_function("SIF-001", "High Pressure SD", sil_level=2)
    """

    def __init__(self):
        """Initialize SafetyFunction manager."""
        self.functions: Dict[str, SafetyFunctionSpec] = {}
        logger.info("SafetyFunction manager initialized")

    def create_function(
        self,
        function_id: str,
        name: str,
        sil_level: int,
        safe_state_id: str,
        action_on_trip: str,
        process_safety_time_ms: float,
        required_response_time_ms: float,
        proof_test_interval_hours: float,
        pfd_target: float,
        description: str = "",
        hazard_description: str = "",
        initiating_cause: str = "",
        consequence: str = ""
    ) -> SafetyFunctionSpec:
        """
        Create a new Safety Function specification.

        Args:
            function_id: Unique function identifier
            name: Function name
            sil_level: Required SIL (1-4)
            safe_state_id: Reference to safe state
            action_on_trip: Action description
            process_safety_time_ms: PST in milliseconds
            required_response_time_ms: Required response time
            proof_test_interval_hours: Proof test interval
            pfd_target: Target PFD average
            description: Detailed description
            hazard_description: Hazard protected against
            initiating_cause: Initiating event
            consequence: Potential consequence

        Returns:
            SafetyFunctionSpec object

        Raises:
            ValueError: If parameters are invalid
        """
        logger.info(f"Creating safety function: {function_id} - {name}")

        # Validate response time vs PST
        if required_response_time_ms >= process_safety_time_ms:
            raise ValueError(
                f"Response time ({required_response_time_ms}ms) must be less "
                f"than PST ({process_safety_time_ms}ms)"
            )

        # Validate PFD vs SIL
        sil_pfd_limits = {
            1: (1e-2, 1e-1),
            2: (1e-3, 1e-2),
            3: (1e-4, 1e-3),
            4: (1e-5, 1e-4),
        }
        lower, upper = sil_pfd_limits[sil_level]
        if not (lower <= pfd_target < upper):
            logger.warning(
                f"PFD target {pfd_target} may not align with SIL {sil_level}"
            )

        spec = SafetyFunctionSpec(
            function_id=function_id,
            name=name,
            description=description,
            sil_level=sil_level,
            hazard_description=hazard_description,
            initiating_cause=initiating_cause,
            consequence=consequence,
            safe_state_id=safe_state_id,
            action_on_trip=action_on_trip,
            process_safety_time_ms=process_safety_time_ms,
            required_response_time_ms=required_response_time_ms,
            proof_test_interval_hours=proof_test_interval_hours,
            pfd_target=pfd_target,
        )

        # Calculate provenance hash
        spec.provenance_hash = self._calculate_provenance(spec)

        self.functions[function_id] = spec

        logger.info(f"Safety function created: {function_id}")

        return spec

    def add_input_sensor(
        self,
        function_id: str,
        sensor: InputSensor
    ) -> SafetyFunctionSpec:
        """
        Add input sensor to safety function.

        Args:
            function_id: Safety function ID
            sensor: InputSensor specification

        Returns:
            Updated SafetyFunctionSpec
        """
        if function_id not in self.functions:
            raise ValueError(f"Safety function {function_id} not found")

        spec = self.functions[function_id]
        spec.input_sensors.append(sensor)
        spec.last_modified = datetime.utcnow()
        spec.provenance_hash = self._calculate_provenance(spec)

        # Update voting based on sensor count
        sensor_count = len(spec.input_sensors)
        if sensor_count == 1:
            spec.input_voting = "1oo1"
        elif sensor_count == 2:
            spec.input_voting = "1oo2"
        elif sensor_count == 3:
            spec.input_voting = "2oo3"

        logger.info(
            f"Added sensor {sensor.sensor_id} to {function_id}. "
            f"Voting: {spec.input_voting}"
        )

        return spec

    def add_output_actuator(
        self,
        function_id: str,
        actuator: OutputActuator
    ) -> SafetyFunctionSpec:
        """
        Add output actuator to safety function.

        Args:
            function_id: Safety function ID
            actuator: OutputActuator specification

        Returns:
            Updated SafetyFunctionSpec
        """
        if function_id not in self.functions:
            raise ValueError(f"Safety function {function_id} not found")

        spec = self.functions[function_id]
        spec.output_actuators.append(actuator)
        spec.last_modified = datetime.utcnow()
        spec.provenance_hash = self._calculate_provenance(spec)

        # Update voting based on actuator count
        actuator_count = len(spec.output_actuators)
        if actuator_count == 1:
            spec.output_voting = "1oo1"
        elif actuator_count == 2:
            spec.output_voting = "1oo2"

        logger.info(
            f"Added actuator {actuator.actuator_id} to {function_id}. "
            f"Voting: {spec.output_voting}"
        )

        return spec

    def set_logic_solver(
        self,
        function_id: str,
        solver: LogicSolverSpec
    ) -> SafetyFunctionSpec:
        """
        Set logic solver for safety function.

        Args:
            function_id: Safety function ID
            solver: LogicSolverSpec specification

        Returns:
            Updated SafetyFunctionSpec
        """
        if function_id not in self.functions:
            raise ValueError(f"Safety function {function_id} not found")

        spec = self.functions[function_id]
        spec.logic_solver = solver
        spec.last_modified = datetime.utcnow()
        spec.provenance_hash = self._calculate_provenance(spec)

        logger.info(f"Set logic solver {solver.solver_id} for {function_id}")

        return spec

    def calculate_response_time(
        self,
        function_id: str
    ) -> Dict[str, float]:
        """
        Calculate total response time for safety function.

        Response Time = Sensor Time + Logic Time + Actuator Time

        Args:
            function_id: Safety function ID

        Returns:
            Dict with response time breakdown
        """
        if function_id not in self.functions:
            raise ValueError(f"Safety function {function_id} not found")

        spec = self.functions[function_id]

        # Sensor response time (use max if multiple)
        sensor_time = max(
            (s.response_time_ms for s in spec.input_sensors),
            default=100.0
        )

        # Logic solver time
        logic_time = 0.0
        if spec.logic_solver:
            logic_time = (
                spec.logic_solver.input_processing_time_ms +
                spec.logic_solver.scan_time_ms +
                spec.logic_solver.output_processing_time_ms
            )

        # Actuator stroke time (use max if multiple)
        actuator_time = max(
            (a.stroke_time_ms for a in spec.output_actuators),
            default=5000.0
        )

        total_time = sensor_time + logic_time + actuator_time

        result = {
            "sensor_time_ms": sensor_time,
            "logic_time_ms": logic_time,
            "actuator_time_ms": actuator_time,
            "total_response_time_ms": total_time,
            "required_response_time_ms": spec.required_response_time_ms,
            "process_safety_time_ms": spec.process_safety_time_ms,
            "margin_ms": spec.process_safety_time_ms - total_time,
            "meets_requirement": total_time <= spec.required_response_time_ms,
        }

        if not result["meets_requirement"]:
            logger.warning(
                f"Response time {total_time}ms exceeds requirement "
                f"{spec.required_response_time_ms}ms for {function_id}"
            )

        return result

    def validate_function(
        self,
        function_id: str
    ) -> Dict[str, Any]:
        """
        Validate safety function completeness.

        Checks:
        - Has input sensors
        - Has output actuators
        - Has logic solver
        - Response time meets requirement
        - SIL capability of components

        Args:
            function_id: Safety function ID

        Returns:
            Validation result dictionary
        """
        if function_id not in self.functions:
            raise ValueError(f"Safety function {function_id} not found")

        spec = self.functions[function_id]
        errors: List[str] = []
        warnings: List[str] = []

        # Check inputs
        if not spec.input_sensors:
            errors.append("No input sensors defined")

        # Check outputs
        if not spec.output_actuators:
            errors.append("No output actuators defined")

        # Check logic solver
        if not spec.logic_solver:
            errors.append("No logic solver defined")

        # Check response time
        if spec.input_sensors and spec.output_actuators:
            response = self.calculate_response_time(function_id)
            if not response["meets_requirement"]:
                errors.append(
                    f"Response time ({response['total_response_time_ms']}ms) "
                    f"exceeds requirement ({spec.required_response_time_ms}ms)"
                )

        # Check SIL capability
        for sensor in spec.input_sensors:
            if sensor.sil_capability < spec.sil_level:
                warnings.append(
                    f"Sensor {sensor.sensor_id} SIL capability "
                    f"({sensor.sil_capability}) < required SIL ({spec.sil_level})"
                )

        for actuator in spec.output_actuators:
            if actuator.sil_capability < spec.sil_level:
                warnings.append(
                    f"Actuator {actuator.actuator_id} SIL capability "
                    f"({actuator.sil_capability}) < required SIL ({spec.sil_level})"
                )

        if spec.logic_solver and spec.logic_solver.sil_capability < spec.sil_level:
            warnings.append(
                f"Logic solver SIL capability "
                f"({spec.logic_solver.sil_capability}) < required SIL ({spec.sil_level})"
            )

        return {
            "function_id": function_id,
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "input_count": len(spec.input_sensors),
            "output_count": len(spec.output_actuators),
            "has_logic_solver": spec.logic_solver is not None,
        }

    def get_pfd_inputs(
        self,
        function_id: str
    ) -> Dict[str, Any]:
        """
        Get inputs for PFD calculation.

        Args:
            function_id: Safety function ID

        Returns:
            Dict with failure rates and architecture info
        """
        if function_id not in self.functions:
            raise ValueError(f"Safety function {function_id} not found")

        spec = self.functions[function_id]

        # Aggregate sensor failure rates
        sensor_lambda_du = sum(s.failure_rate_du for s in spec.input_sensors)
        sensor_lambda_dd = sum(s.failure_rate_dd for s in spec.input_sensors)

        # Aggregate actuator failure rates
        actuator_lambda_du = sum(a.failure_rate_du for a in spec.output_actuators)
        actuator_lambda_dd = sum(a.failure_rate_dd for a in spec.output_actuators)

        # Logic solver failure rates
        logic_lambda_du = 0.0
        if spec.logic_solver:
            logic_lambda_du = spec.logic_solver.failure_rate_du

        return {
            "function_id": function_id,
            "input_architecture": spec.input_voting,
            "output_architecture": spec.output_voting,
            "sensor_lambda_du": sensor_lambda_du,
            "sensor_lambda_dd": sensor_lambda_dd,
            "logic_lambda_du": logic_lambda_du,
            "actuator_lambda_du": actuator_lambda_du,
            "actuator_lambda_dd": actuator_lambda_dd,
            "total_lambda_du": sensor_lambda_du + logic_lambda_du + actuator_lambda_du,
            "proof_test_interval_hours": spec.proof_test_interval_hours,
            "target_pfd": spec.pfd_target,
            "target_sil": spec.sil_level,
        }

    def _calculate_provenance(self, spec: SafetyFunctionSpec) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{spec.function_id}|{spec.name}|{spec.sil_level}|"
            f"{len(spec.input_sensors)}|{len(spec.output_actuators)}|"
            f"{spec.last_modified.isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
