"""
Safety Validator for GL-005 CombustionControlAgent

Validates all safety interlocks and checks operating limits for combustion systems.
Zero-hallucination design using deterministic safety logic and industry standards.

Reference Standards:
- NFPA 85: Boiler and Combustion Systems Hazards Code
- NFPA 86: Standard for Ovens and Furnaces
- API 556: Fired Heaters for General Refinery Service
- IEC 61508: Functional Safety of Electrical/Electronic/Programmable Systems
- ISA-84: Functional Safety - Safety Instrumented Systems

Safety Principles:
- Fail-safe design (dangerous failure leads to safe state)
- Redundancy for critical measurements
- Diverse protection layers (Defense in Depth)
- Predictable shutdown sequences
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
from pydantic import BaseModel, Field, validator
from enum import Enum
import math
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class SafetyLevel(str, Enum):
    """Safety integrity levels"""
    SAFE = "safe"
    ADVISORY = "advisory"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


class InterlockStatus(str, Enum):
    """Interlock status"""
    OK = "ok"
    BYPASSED = "bypassed"
    TRIPPED = "tripped"
    FAILED = "failed"


class AlarmPriority(str, Enum):
    """Alarm priority levels per ISA-18.2"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyInterlock:
    """Individual safety interlock definition"""
    name: str
    description: str
    status: InterlockStatus
    trip_value: Optional[float] = None
    current_value: Optional[float] = None
    trip_reason: Optional[str] = None


@dataclass
class SafetyAlarm:
    """Safety alarm definition"""
    alarm_id: str
    description: str
    priority: AlarmPriority
    timestamp: float
    current_value: float
    setpoint: float
    deviation: float


class SafetyLimits(BaseModel):
    """Safety operating limits"""

    # Temperature limits (Celsius)
    max_combustion_temperature: float = Field(
        default=1500,
        ge=0,
        le=2000,
        description="Maximum combustion chamber temperature"
    )
    max_flue_gas_temperature: float = Field(
        default=500,
        ge=0,
        le=1000,
        description="Maximum flue gas temperature"
    )
    min_operating_temperature: float = Field(
        default=200,
        ge=0,
        description="Minimum stable combustion temperature"
    )

    # Pressure limits (Pa)
    max_combustion_pressure: float = Field(
        default=10000,
        ge=0,
        le=100000,
        description="Maximum combustion chamber pressure"
    )
    min_combustion_pressure: float = Field(
        default=-1000,
        ge=-5000,
        le=0,
        description="Minimum combustion chamber pressure (draft)"
    )
    max_fuel_supply_pressure: float = Field(
        default=500000,
        ge=0,
        description="Maximum fuel supply pressure"
    )
    min_fuel_supply_pressure: float = Field(
        default=50000,
        ge=0,
        description="Minimum fuel supply pressure"
    )

    # Flow limits (kg/hr)
    max_fuel_flow_rate: float = Field(
        ...,
        gt=0,
        description="Maximum fuel flow rate"
    )
    min_fuel_flow_rate: float = Field(
        default=0,
        ge=0,
        description="Minimum fuel flow rate (turndown)"
    )
    max_air_flow_rate: float = Field(
        ...,
        gt=0,
        description="Maximum air flow rate"
    )
    min_air_flow_rate: float = Field(
        default=0,
        ge=0,
        description="Minimum air flow rate"
    )

    # Oxygen limits (%)
    min_o2_percent: float = Field(
        default=2.0,
        ge=0,
        le=21,
        description="Minimum O2 for safe combustion"
    )
    max_o2_percent: float = Field(
        default=15.0,
        ge=0,
        le=21,
        description="Maximum O2 (efficiency limit)"
    )

    # CO limits (ppm)
    max_co_ppm: float = Field(
        default=400,
        ge=0,
        description="Maximum CO for safe operation"
    )

    # Rate of change limits
    max_temperature_rate_c_per_min: float = Field(
        default=50,
        gt=0,
        description="Maximum temperature ramp rate"
    )
    max_pressure_rate_pa_per_sec: float = Field(
        default=1000,
        gt=0,
        description="Maximum pressure change rate"
    )


class SafetyValidatorInput(BaseModel):
    """Input for safety validation"""

    # Current measurements
    combustion_temperature_c: float = Field(..., description="Current combustion temperature")
    flue_gas_temperature_c: float = Field(..., description="Current flue gas temperature")
    combustion_pressure_pa: float = Field(..., description="Current combustion pressure")
    fuel_supply_pressure_pa: float = Field(..., description="Current fuel supply pressure")
    fuel_flow_rate_kg_per_hr: float = Field(..., description="Current fuel flow rate")
    air_flow_rate_kg_per_hr: float = Field(..., description="Current air flow rate")
    o2_percent: float = Field(..., description="Measured O2 in flue gas")
    co_ppm: float = Field(..., description="Measured CO in ppm")

    # Historical data (for rate of change)
    previous_temperature_c: Optional[float] = None
    previous_pressure_pa: Optional[float] = None
    time_delta_seconds: float = Field(default=1.0, gt=0)

    # Safety limits
    safety_limits: SafetyLimits

    # Operational state
    burner_firing: bool = Field(..., description="Is burner currently firing")
    flame_detected: bool = Field(..., description="Is flame detected")
    purge_complete: bool = Field(default=True, description="Has pre-purge been completed")

    # Redundant measurements (if available)
    backup_temperature_c: Optional[float] = None
    backup_pressure_pa: Optional[float] = None

    # Emergency conditions
    fire_detected: bool = Field(default=False)
    gas_leak_detected: bool = Field(default=False)
    operator_emergency_stop: bool = Field(default=False)


class SafetyValidatorOutput(BaseModel):
    """Safety validation output"""

    # Overall safety status
    safety_level: SafetyLevel
    is_safe_to_operate: bool
    requires_shutdown: bool

    # Risk assessment
    overall_risk_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Overall risk score (0=no risk, 1=maximum risk)"
    )

    # Interlock status
    interlocks: List[Dict[str, any]] = Field(
        default_factory=list,
        description="Status of all safety interlocks"
    )
    tripped_interlocks: List[str] = Field(
        default_factory=list,
        description="List of tripped interlock names"
    )

    # Alarms
    active_alarms: List[Dict[str, any]] = Field(
        default_factory=list,
        description="Active safety alarms"
    )
    alarm_count_by_priority: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of alarms by priority"
    )

    # Limit violations
    temperature_violations: List[str] = Field(default_factory=list)
    pressure_violations: List[str] = Field(default_factory=list)
    flow_violations: List[str] = Field(default_factory=list)
    emission_violations: List[str] = Field(default_factory=list)

    # Rate of change checks
    excessive_rate_of_change: bool = False
    rate_of_change_details: List[str] = Field(default_factory=list)

    # Redundancy check
    redundant_sensor_mismatch: bool = False
    sensor_mismatch_details: List[str] = Field(default_factory=list)

    # Emergency conditions
    emergency_shutdown_required: bool = False
    emergency_shutdown_reason: Optional[str] = None

    # Corrective actions
    required_actions: List[str] = Field(
        default_factory=list,
        description="Required corrective actions"
    )
    shutdown_sequence: Optional[List[str]] = None


class SafetyValidator:
    """
    Comprehensive safety validator for combustion control systems.

    This validator implements multiple layers of protection (Defense in Depth):

    Layer 1: Process Control (normal operation)
    Layer 2: Alarms and Operator Intervention
    Layer 3: Safety Instrumented System (SIS)
    Layer 4: Physical Protection (relief valves, rupture discs)

    Safety Philosophy:
        - All checks are deterministic (no probabilistic risk)
        - Conservative limits (safety factors built in)
        - Fail-safe design (sensor failure → shutdown)
        - Redundancy for critical measurements
        - Clear alarm prioritization per ISA-18.2
    """

    # Safety factors (built-in margin)
    TEMPERATURE_SAFETY_FACTOR = 0.95  # Trip at 95% of absolute limit
    PRESSURE_SAFETY_FACTOR = 0.95
    FLOW_SAFETY_FACTOR = 0.98

    # Alarm thresholds (% of limit)
    ADVISORY_THRESHOLD = 0.75  # 75% of limit
    WARNING_THRESHOLD = 0.85   # 85% of limit
    ALARM_THRESHOLD = 0.95     # 95% of limit
    TRIP_THRESHOLD = 1.00      # 100% of limit

    # Sensor mismatch tolerance
    TEMPERATURE_MISMATCH_TOLERANCE = 10.0  # °C
    PRESSURE_MISMATCH_TOLERANCE = 1000.0   # Pa

    def __init__(self):
        """Initialize safety validator"""
        self.logger = logging.getLogger(__name__)
        self.interlock_history = []
        self.alarm_history = []

    def validate_all_safety_interlocks(
        self,
        validator_input: SafetyValidatorInput
    ) -> SafetyValidatorOutput:
        """
        Validate all safety interlocks and operating limits.

        Safety Check Sequence:
            1. Check emergency conditions (fire, gas leak, E-stop)
            2. Validate temperature limits
            3. Validate pressure limits
            4. Validate flow limits
            5. Validate emission limits
            6. Check rate of change limits
            7. Verify flame safety (if firing)
            8. Check redundant sensors
            9. Calculate overall risk score
            10. Determine required actions

        Args:
            validator_input: Safety validation inputs

        Returns:
            SafetyValidatorOutput with complete safety status
        """
        self.logger.info("Performing comprehensive safety validation")

        # Initialize result containers
        interlocks = []
        alarms = []
        required_actions = []
        temp_violations = []
        pressure_violations = []
        flow_violations = []
        emission_violations = []
        rate_violations = []
        sensor_mismatches = []

        # Step 1: Check emergency conditions
        emergency_shutdown, emergency_reason = self._check_emergency_conditions(validator_input)

        if emergency_shutdown:
            return self._create_emergency_shutdown_output(emergency_reason)

        # Step 2: Validate temperature limits
        temp_interlocks, temp_alarms, temp_viols = self.validate_temperature_limits(
            validator_input.combustion_temperature_c,
            validator_input.flue_gas_temperature_c,
            validator_input.safety_limits
        )
        interlocks.extend(temp_interlocks)
        alarms.extend(temp_alarms)
        temp_violations.extend(temp_viols)

        # Step 3: Validate pressure limits
        press_interlocks, press_alarms, press_viols = self.validate_pressure_limits(
            validator_input.combustion_pressure_pa,
            validator_input.fuel_supply_pressure_pa,
            validator_input.safety_limits
        )
        interlocks.extend(press_interlocks)
        alarms.extend(press_alarms)
        pressure_violations.extend(press_viols)

        # Step 4: Validate flow limits
        flow_interlocks, flow_alarms, flow_viols = self.validate_fuel_flow_limits(
            validator_input.fuel_flow_rate_kg_per_hr,
            validator_input.air_flow_rate_kg_per_hr,
            validator_input.safety_limits
        )
        interlocks.extend(flow_interlocks)
        alarms.extend(flow_alarms)
        flow_violations.extend(flow_viols)

        # Step 5: Validate emission limits
        emission_interlocks, emission_alarms, emission_viols = self._validate_emission_limits(
            validator_input.o2_percent,
            validator_input.co_ppm,
            validator_input.safety_limits
        )
        interlocks.extend(emission_interlocks)
        alarms.extend(emission_alarms)
        emission_violations.extend(emission_viols)

        # Step 6: Check rate of change
        if validator_input.previous_temperature_c is not None:
            rate_ok, rate_violations_list = self._check_rate_of_change(validator_input)
            rate_violations.extend(rate_violations_list)

        # Step 7: Verify flame safety
        if validator_input.burner_firing:
            flame_ok, flame_violations = self._check_flame_safety(
                validator_input.flame_detected,
                validator_input.purge_complete,
                validator_input.fuel_flow_rate_kg_per_hr
            )
            if not flame_ok:
                interlocks.append(SafetyInterlock(
                    name="flame_safety",
                    description="Flame safety interlock",
                    status=InterlockStatus.TRIPPED,
                    trip_reason="Flame lost with fuel flowing"
                ))
                required_actions.append("IMMEDIATE SHUTDOWN: Flame lost with fuel flowing")

        # Step 8: Check redundant sensors
        if validator_input.backup_temperature_c is not None:
            sensor_ok, sensor_details = self._check_redundant_sensors(
                validator_input.combustion_temperature_c,
                validator_input.backup_temperature_c,
                validator_input.combustion_pressure_pa,
                validator_input.backup_pressure_pa
            )
            if not sensor_ok:
                sensor_mismatches.extend(sensor_details)

        # Step 9: Calculate overall risk score
        risk_score = self.calculate_risk_score(
            temp_violations,
            pressure_violations,
            flow_violations,
            emission_violations,
            len([i for i in interlocks if i.status == InterlockStatus.TRIPPED])
        )

        # Step 10: Determine safety level
        safety_level = self._determine_safety_level(risk_score, interlocks, alarms)

        # Step 11: Determine if shutdown required
        tripped_interlocks = [i.name for i in interlocks if i.status == InterlockStatus.TRIPPED]
        requires_shutdown = len(tripped_interlocks) > 0 or risk_score > 0.8

        # Step 12: Generate required actions
        if not requires_shutdown:
            required_actions.extend(self._generate_corrective_actions(
                alarms, temp_violations, pressure_violations, flow_violations
            ))

        # Step 13: Generate shutdown sequence if needed
        shutdown_sequence = None
        if requires_shutdown:
            shutdown_sequence = self._generate_shutdown_sequence(validator_input)

        # Count alarms by priority
        alarm_counts = {
            AlarmPriority.LOW.value: len([a for a in alarms if a.priority == AlarmPriority.LOW]),
            AlarmPriority.MEDIUM.value: len([a for a in alarms if a.priority == AlarmPriority.MEDIUM]),
            AlarmPriority.HIGH.value: len([a for a in alarms if a.priority == AlarmPriority.HIGH]),
            AlarmPriority.CRITICAL.value: len([a for a in alarms if a.priority == AlarmPriority.CRITICAL]),
        }

        return SafetyValidatorOutput(
            safety_level=safety_level,
            is_safe_to_operate=safety_level in [SafetyLevel.SAFE, SafetyLevel.ADVISORY],
            requires_shutdown=requires_shutdown,
            overall_risk_score=self._round_decimal(risk_score, 4),
            interlocks=[{
                'name': i.name,
                'description': i.description,
                'status': i.status.value,
                'trip_value': i.trip_value,
                'current_value': i.current_value,
                'trip_reason': i.trip_reason
            } for i in interlocks],
            tripped_interlocks=tripped_interlocks,
            active_alarms=[{
                'alarm_id': a.alarm_id,
                'description': a.description,
                'priority': a.priority.value,
                'current_value': a.current_value,
                'setpoint': a.setpoint,
                'deviation': a.deviation
            } for a in alarms],
            alarm_count_by_priority=alarm_counts,
            temperature_violations=temp_violations,
            pressure_violations=pressure_violations,
            flow_violations=flow_violations,
            emission_violations=emission_violations,
            excessive_rate_of_change=len(rate_violations) > 0,
            rate_of_change_details=rate_violations,
            redundant_sensor_mismatch=len(sensor_mismatches) > 0,
            sensor_mismatch_details=sensor_mismatches,
            emergency_shutdown_required=emergency_shutdown,
            emergency_shutdown_reason=emergency_reason,
            required_actions=required_actions,
            shutdown_sequence=shutdown_sequence
        )

    def validate_temperature_limits(
        self,
        combustion_temp: float,
        flue_gas_temp: float,
        limits: SafetyLimits
    ) -> Tuple[List[SafetyInterlock], List[SafetyAlarm], List[str]]:
        """
        Validate temperature against safety limits.

        Returns:
            Tuple of (interlocks, alarms, violations)
        """
        interlocks = []
        alarms = []
        violations = []

        # Check combustion temperature high limit
        limit = limits.max_combustion_temperature * self.TEMPERATURE_SAFETY_FACTOR
        if combustion_temp > limit:
            interlocks.append(SafetyInterlock(
                name="combustion_temp_high",
                description="Combustion temperature high interlock",
                status=InterlockStatus.TRIPPED,
                trip_value=limit,
                current_value=combustion_temp,
                trip_reason=f"Temperature {combustion_temp:.1f}°C exceeds limit {limit:.1f}°C"
            ))
            violations.append(f"Combustion temperature HIGH: {combustion_temp:.1f}°C > {limit:.1f}°C")

        elif combustion_temp > limits.max_combustion_temperature * self.ALARM_THRESHOLD:
            alarms.append(SafetyAlarm(
                alarm_id="TEMP_HIGH_ALARM",
                description="Combustion temperature high alarm",
                priority=AlarmPriority.HIGH,
                timestamp=datetime.now().timestamp(),
                current_value=combustion_temp,
                setpoint=limits.max_combustion_temperature,
                deviation=combustion_temp - limits.max_combustion_temperature
            ))

        # Check combustion temperature low limit
        if combustion_temp < limits.min_operating_temperature:
            violations.append(f"Combustion temperature LOW: {combustion_temp:.1f}°C < {limits.min_operating_temperature:.1f}°C")
            alarms.append(SafetyAlarm(
                alarm_id="TEMP_LOW_ALARM",
                description="Combustion temperature low - unstable combustion",
                priority=AlarmPriority.MEDIUM,
                timestamp=datetime.now().timestamp(),
                current_value=combustion_temp,
                setpoint=limits.min_operating_temperature,
                deviation=limits.min_operating_temperature - combustion_temp
            ))

        # Check flue gas temperature
        if flue_gas_temp > limits.max_flue_gas_temperature:
            violations.append(f"Flue gas temperature HIGH: {flue_gas_temp:.1f}°C > {limits.max_flue_gas_temperature:.1f}°C")
            interlocks.append(SafetyInterlock(
                name="flue_gas_temp_high",
                description="Flue gas temperature high interlock",
                status=InterlockStatus.TRIPPED,
                trip_value=limits.max_flue_gas_temperature,
                current_value=flue_gas_temp
            ))

        return interlocks, alarms, violations

    def validate_pressure_limits(
        self,
        combustion_pressure: float,
        fuel_supply_pressure: float,
        limits: SafetyLimits
    ) -> Tuple[List[SafetyInterlock], List[SafetyAlarm], List[str]]:
        """Validate pressure against safety limits"""
        interlocks = []
        alarms = []
        violations = []

        # Check combustion pressure high limit
        if combustion_pressure > limits.max_combustion_pressure * self.PRESSURE_SAFETY_FACTOR:
            interlocks.append(SafetyInterlock(
                name="combustion_pressure_high",
                description="Combustion pressure high interlock",
                status=InterlockStatus.TRIPPED,
                trip_value=limits.max_combustion_pressure,
                current_value=combustion_pressure
            ))
            violations.append(f"Combustion pressure HIGH: {combustion_pressure:.0f} Pa")

        # Check combustion pressure low limit (draft)
        if combustion_pressure < limits.min_combustion_pressure:
            violations.append(f"Combustion pressure LOW: {combustion_pressure:.0f} Pa")
            alarms.append(SafetyAlarm(
                alarm_id="PRESSURE_LOW_ALARM",
                description="Combustion pressure too low - check draft",
                priority=AlarmPriority.MEDIUM,
                timestamp=datetime.now().timestamp(),
                current_value=combustion_pressure,
                setpoint=limits.min_combustion_pressure,
                deviation=limits.min_combustion_pressure - combustion_pressure
            ))

        # Check fuel supply pressure
        if fuel_supply_pressure > limits.max_fuel_supply_pressure:
            interlocks.append(SafetyInterlock(
                name="fuel_pressure_high",
                description="Fuel supply pressure high",
                status=InterlockStatus.TRIPPED,
                trip_value=limits.max_fuel_supply_pressure,
                current_value=fuel_supply_pressure
            ))
            violations.append(f"Fuel supply pressure HIGH: {fuel_supply_pressure:.0f} Pa")

        if fuel_supply_pressure < limits.min_fuel_supply_pressure:
            violations.append(f"Fuel supply pressure LOW: {fuel_supply_pressure:.0f} Pa")

        return interlocks, alarms, violations

    def validate_fuel_flow_limits(
        self,
        fuel_flow: float,
        air_flow: float,
        limits: SafetyLimits
    ) -> Tuple[List[SafetyInterlock], List[SafetyAlarm], List[str]]:
        """Validate flow rates against safety limits"""
        interlocks = []
        alarms = []
        violations = []

        # Check fuel flow high limit
        if fuel_flow > limits.max_fuel_flow_rate * self.FLOW_SAFETY_FACTOR:
            interlocks.append(SafetyInterlock(
                name="fuel_flow_high",
                description="Fuel flow rate high interlock",
                status=InterlockStatus.TRIPPED,
                trip_value=limits.max_fuel_flow_rate,
                current_value=fuel_flow
            ))
            violations.append(f"Fuel flow HIGH: {fuel_flow:.1f} kg/hr")

        # Check air flow high limit
        if air_flow > limits.max_air_flow_rate:
            violations.append(f"Air flow HIGH: {air_flow:.1f} kg/hr")

        # Check minimum flows during firing
        if fuel_flow > 0 and fuel_flow < limits.min_fuel_flow_rate:
            violations.append(f"Fuel flow below turndown: {fuel_flow:.1f} kg/hr")

        return interlocks, alarms, violations

    def _validate_emission_limits(
        self,
        o2_percent: float,
        co_ppm: float,
        limits: SafetyLimits
    ) -> Tuple[List[SafetyInterlock], List[SafetyAlarm], List[str]]:
        """Validate emissions against safety limits"""
        interlocks = []
        alarms = []
        violations = []

        # Check O2 limits
        if o2_percent < limits.min_o2_percent:
            violations.append(f"O2 too low: {o2_percent:.1f}% - risk of incomplete combustion")
            alarms.append(SafetyAlarm(
                alarm_id="O2_LOW_ALARM",
                description="O2 below safe limit",
                priority=AlarmPriority.HIGH,
                timestamp=datetime.now().timestamp(),
                current_value=o2_percent,
                setpoint=limits.min_o2_percent,
                deviation=limits.min_o2_percent - o2_percent
            ))

        if o2_percent > limits.max_o2_percent:
            violations.append(f"O2 too high: {o2_percent:.1f}% - low efficiency")

        # Check CO limits
        if co_ppm > limits.max_co_ppm:
            violations.append(f"CO HIGH: {co_ppm:.0f} ppm - incomplete combustion")
            if co_ppm > limits.max_co_ppm * 2:
                interlocks.append(SafetyInterlock(
                    name="co_high",
                    description="CO critically high - incomplete combustion",
                    status=InterlockStatus.TRIPPED,
                    trip_value=limits.max_co_ppm * 2,
                    current_value=co_ppm
                ))

        return interlocks, alarms, violations

    def check_emergency_conditions(
        self,
        fire_detected: bool,
        gas_leak_detected: bool,
        operator_stop: bool
    ) -> Tuple[bool, Optional[str]]:
        """
        Check for emergency conditions requiring immediate shutdown.

        Returns:
            Tuple of (emergency_shutdown_required, reason)
        """
        if fire_detected:
            return True, "FIRE DETECTED - Emergency shutdown"

        if gas_leak_detected:
            return True, "GAS LEAK DETECTED - Emergency shutdown"

        if operator_stop:
            return True, "OPERATOR EMERGENCY STOP"

        return False, None

    def _check_emergency_conditions(
        self,
        validator_input: SafetyValidatorInput
    ) -> Tuple[bool, Optional[str]]:
        """Internal emergency condition check"""
        return self.check_emergency_conditions(
            validator_input.fire_detected,
            validator_input.gas_leak_detected,
            validator_input.operator_emergency_stop
        )

    def _check_rate_of_change(
        self,
        validator_input: SafetyValidatorInput
    ) -> Tuple[bool, List[str]]:
        """Check rate of change limits"""
        violations = []

        if validator_input.previous_temperature_c is not None:
            temp_rate = (
                (validator_input.combustion_temperature_c - validator_input.previous_temperature_c) /
                validator_input.time_delta_seconds * 60  # Convert to per minute
            )

            if abs(temp_rate) > validator_input.safety_limits.max_temperature_rate_c_per_min:
                violations.append(
                    f"Temperature rate of change excessive: {temp_rate:.1f} °C/min"
                )

        if validator_input.previous_pressure_pa is not None:
            pressure_rate = (
                (validator_input.combustion_pressure_pa - validator_input.previous_pressure_pa) /
                validator_input.time_delta_seconds
            )

            if abs(pressure_rate) > validator_input.safety_limits.max_pressure_rate_pa_per_sec:
                violations.append(
                    f"Pressure rate of change excessive: {pressure_rate:.0f} Pa/s"
                )

        return len(violations) == 0, violations

    def _check_flame_safety(
        self,
        flame_detected: bool,
        purge_complete: bool,
        fuel_flow: float
    ) -> Tuple[bool, List[str]]:
        """Check flame safety interlocks"""
        violations = []

        # Flame must be detected if fuel is flowing
        if fuel_flow > 0 and not flame_detected:
            violations.append("CRITICAL: Fuel flowing without flame detection")

        # Purge must be complete before ignition
        if fuel_flow > 0 and not purge_complete:
            violations.append("Purge incomplete - cannot start fuel flow")

        return len(violations) == 0, violations

    def _check_redundant_sensors(
        self,
        temp_primary: float,
        temp_backup: Optional[float],
        pressure_primary: float,
        pressure_backup: Optional[float]
    ) -> Tuple[bool, List[str]]:
        """Check redundant sensor agreement"""
        mismatches = []

        if temp_backup is not None:
            temp_diff = abs(temp_primary - temp_backup)
            if temp_diff > self.TEMPERATURE_MISMATCH_TOLERANCE:
                mismatches.append(
                    f"Temperature sensor mismatch: {temp_diff:.1f}°C difference"
                )

        if pressure_backup is not None:
            pressure_diff = abs(pressure_primary - pressure_backup)
            if pressure_diff > self.PRESSURE_MISMATCH_TOLERANCE:
                mismatches.append(
                    f"Pressure sensor mismatch: {pressure_diff:.0f} Pa difference"
                )

        return len(mismatches) == 0, mismatches

    def calculate_risk_score(
        self,
        temp_violations: List[str],
        pressure_violations: List[str],
        flow_violations: List[str],
        emission_violations: List[str],
        tripped_interlock_count: int
    ) -> float:
        """
        Calculate overall risk score.

        Formula:
            Risk = w1*temp_risk + w2*pressure_risk + w3*flow_risk +
                   w4*emission_risk + w5*interlock_risk

        Returns:
            Risk score from 0 (no risk) to 1 (maximum risk)
        """
        # Normalize violation counts
        temp_risk = min(len(temp_violations) / 3.0, 1.0)
        pressure_risk = min(len(pressure_violations) / 3.0, 1.0)
        flow_risk = min(len(flow_violations) / 2.0, 1.0)
        emission_risk = min(len(emission_violations) / 2.0, 1.0)
        interlock_risk = min(tripped_interlock_count / 2.0, 1.0)

        # Weighted combination
        risk_score = (
            0.25 * temp_risk +
            0.25 * pressure_risk +
            0.15 * flow_risk +
            0.15 * emission_risk +
            0.20 * interlock_risk
        )

        return min(risk_score, 1.0)

    def _determine_safety_level(
        self,
        risk_score: float,
        interlocks: List[SafetyInterlock],
        alarms: List[SafetyAlarm]
    ) -> SafetyLevel:
        """Determine overall safety level"""
        # Check for tripped interlocks (highest priority)
        if any(i.status == InterlockStatus.TRIPPED for i in interlocks):
            return SafetyLevel.EMERGENCY_SHUTDOWN

        # Check for critical alarms
        if any(a.priority == AlarmPriority.CRITICAL for a in alarms):
            return SafetyLevel.CRITICAL

        # Check risk score
        if risk_score > 0.8:
            return SafetyLevel.CRITICAL
        elif risk_score > 0.6:
            return SafetyLevel.ALARM
        elif risk_score > 0.4:
            return SafetyLevel.WARNING
        elif risk_score > 0.2:
            return SafetyLevel.ADVISORY
        else:
            return SafetyLevel.SAFE

    def _generate_corrective_actions(
        self,
        alarms: List[SafetyAlarm],
        temp_violations: List[str],
        pressure_violations: List[str],
        flow_violations: List[str]
    ) -> List[str]:
        """Generate corrective action recommendations"""
        actions = []

        if temp_violations:
            actions.append("Monitor temperature closely - approaching limits")
        if pressure_violations:
            actions.append("Check pressure controls and relief systems")
        if flow_violations:
            actions.append("Adjust flow rates to within safe operating range")

        critical_alarms = [a for a in alarms if a.priority == AlarmPriority.CRITICAL]
        if critical_alarms:
            actions.append(f"Address {len(critical_alarms)} critical alarms immediately")

        return actions

    def _generate_shutdown_sequence(
        self,
        validator_input: SafetyValidatorInput
    ) -> List[str]:
        """Generate safe shutdown sequence"""
        sequence = [
            "1. Stop fuel flow immediately",
            "2. Continue air flow for purge (minimum 5 air changes)",
            "3. Monitor temperature decrease",
            "4. Close all fuel isolation valves",
            "5. Stop air flow after purge complete",
            "6. Verify all interlocks reset",
            "7. Log shutdown event and reasons",
            "8. Notify operations and maintenance"
        ]
        return sequence

    def _create_emergency_shutdown_output(self, reason: str) -> SafetyValidatorOutput:
        """Create output for emergency shutdown condition"""
        return SafetyValidatorOutput(
            safety_level=SafetyLevel.EMERGENCY_SHUTDOWN,
            is_safe_to_operate=False,
            requires_shutdown=True,
            overall_risk_score=1.0,
            emergency_shutdown_required=True,
            emergency_shutdown_reason=reason,
            required_actions=[
                "EMERGENCY SHUTDOWN INITIATED",
                reason,
                "Execute emergency shutdown sequence immediately"
            ],
            shutdown_sequence=self._generate_shutdown_sequence(None)
        )

    def _round_decimal(self, value: float, places: int) -> float:
        """Round to specified decimal places using ROUND_HALF_UP"""
        decimal_value = Decimal(str(value))
        quantize_string = '0.' + '0' * places if places > 0 else '1'
        rounded = decimal_value.quantize(Decimal(quantize_string), rounding=ROUND_HALF_UP)
        return float(rounded)
