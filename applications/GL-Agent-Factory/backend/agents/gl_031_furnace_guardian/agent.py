"""
GL-031: Furnace Guardian Agent (FURNACE-GUARDIAN)

This module implements the FurnaceGuardianAgent for furnace safety monitoring
in industrial process heat systems.

The agent provides:
- Interlock validation per NFPA 86, API 560, EN 746
- Purge verification and timing
- Flame supervision with UV/IR scanner validation
- Temperature and pressure limit monitoring
- Complete SHA-256 provenance tracking

Standards Compliance:
- NFPA 86: Standard for Ovens and Furnaces
- API 560: Fired Heaters for General Refinery Service
- EN 746: Industrial Thermoprocessing Equipment

Example:
    >>> agent = FurnaceGuardianAgent()
    >>> result = agent.run(FurnaceGuardianInput(
    ...     furnace_id="FRN-001",
    ...     temps=[TemperatureReading(sensor_id="T1", value_celsius=850, ...)],
    ...     pressures=[PressureReading(sensor_id="P1", value_kpa=101.3, ...)],
    ...     flame_status=FlameStatus(is_detected=True, ...),
    ...     interlocks=[InterlockStatus(interlock_type="FLAME_FAILURE", ...)]
    ... ))
    >>> print(f"Safety Score: {result.safety_score}")
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from .models import (
    RiskLevel,
    InterlockType,
    ViolationSeverity,
    FlameDetectorType,
    FurnaceType,
    ComplianceStandard,
    PurgeStatus,
)
from .formulas import (
    verify_purge_complete,
    calculate_flame_signal_quality,
    calculate_flame_response_time,
    calculate_safety_score,
    check_temperature_limits,
    check_pressure_limits,
    calculate_interlock_reliability,
)

logger = logging.getLogger(__name__)


# =============================================================================
# INPUT MODELS
# =============================================================================

class TemperatureReading(BaseModel):
    """Temperature sensor reading."""

    sensor_id: str = Field(..., description="Temperature sensor identifier")
    value_celsius: float = Field(..., description="Temperature value in Celsius")
    low_limit: float = Field(default=0.0, description="Low alarm limit")
    high_limit: float = Field(default=1000.0, description="High alarm limit")
    high_high_limit: Optional[float] = Field(None, description="High-high trip limit")
    location: Optional[str] = Field(None, description="Sensor location description")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PressureReading(BaseModel):
    """Pressure sensor reading."""

    sensor_id: str = Field(..., description="Pressure sensor identifier")
    value_kpa: float = Field(..., description="Pressure value in kPa")
    low_limit: float = Field(default=-10.0, description="Low alarm limit")
    high_limit: float = Field(default=200.0, description="High alarm limit")
    low_low_limit: Optional[float] = Field(None, description="Low-low trip limit")
    high_high_limit: Optional[float] = Field(None, description="High-high trip limit")
    location: Optional[str] = Field(None, description="Sensor location description")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FlameStatus(BaseModel):
    """Flame detector status."""

    is_detected: bool = Field(..., description="Whether flame is detected")
    detector_type: FlameDetectorType = Field(
        default=FlameDetectorType.UV_IR_COMBINED,
        description="Type of flame detector"
    )
    signal_strength: float = Field(
        default=0.0, ge=0, le=100,
        description="Signal strength 0-100"
    )
    noise_level: float = Field(default=5.0, ge=0, description="Background noise level")
    detector_id: str = Field(default="FD-001", description="Detector identifier")
    flame_on_timestamp: Optional[datetime] = Field(None, description="When flame ignited")
    detection_timestamp: Optional[datetime] = Field(None, description="When flame detected")


class InterlockStatus(BaseModel):
    """Safety interlock status."""

    interlock_type: InterlockType = Field(..., description="Type of interlock")
    is_ok: bool = Field(..., description="Whether interlock is in OK state")
    is_bypassed: bool = Field(default=False, description="Whether interlock is bypassed")
    bypass_reason: Optional[str] = Field(None, description="Reason for bypass if bypassed")
    last_test_date: Optional[datetime] = Field(None, description="Last proof test date")
    successful_tests: int = Field(default=0, ge=0, description="Count of successful tests")
    total_tests: int = Field(default=0, ge=0, description="Total proof tests")


class PurgeData(BaseModel):
    """Furnace purge cycle data."""

    status: PurgeStatus = Field(..., description="Current purge status")
    airflow_cfm: float = Field(default=0.0, ge=0, description="Purge airflow in CFM")
    furnace_volume_cubic_feet: float = Field(
        default=1000.0, gt=0,
        description="Furnace internal volume"
    )
    purge_time_seconds: float = Field(default=0.0, ge=0, description="Purge duration")
    furnace_class: str = Field(default="A", description="NFPA 86 furnace class (A or B)")
    start_timestamp: Optional[datetime] = Field(None, description="Purge start time")
    end_timestamp: Optional[datetime] = Field(None, description="Purge end time")


class FurnaceGuardianInput(BaseModel):
    """Input data model for FurnaceGuardianAgent."""

    furnace_id: str = Field(..., min_length=1, description="Unique furnace identifier")
    furnace_type: FurnaceType = Field(
        default=FurnaceType.PROCESS_FURNACE,
        description="Type of furnace"
    )
    temps: List[TemperatureReading] = Field(
        default_factory=list,
        description="Temperature sensor readings"
    )
    pressures: List[PressureReading] = Field(
        default_factory=list,
        description="Pressure sensor readings"
    )
    flame_status: FlameStatus = Field(..., description="Flame detector status")
    interlocks: List[InterlockStatus] = Field(
        default_factory=list,
        description="Safety interlock statuses"
    )
    purge_data: Optional[PurgeData] = Field(None, description="Purge cycle data")
    compliance_standards: List[ComplianceStandard] = Field(
        default=[ComplianceStandard.NFPA_86],
        description="Applicable compliance standards"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class SafetyViolation(BaseModel):
    """Safety violation record."""

    violation_id: str = Field(..., description="Unique violation identifier")
    severity: ViolationSeverity = Field(..., description="Violation severity")
    category: str = Field(..., description="Violation category")
    description: str = Field(..., description="Violation description")
    affected_component: str = Field(..., description="Affected component ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    standard_reference: Optional[str] = Field(None, description="Relevant standard section")


class CorrectiveAction(BaseModel):
    """Recommended corrective action."""

    action_id: str = Field(..., description="Unique action identifier")
    priority: str = Field(..., description="Action priority (IMMEDIATE, HIGH, MEDIUM, LOW)")
    description: str = Field(..., description="Action description")
    affected_component: str = Field(..., description="Component to address")
    estimated_time_minutes: Optional[int] = Field(None, description="Estimated completion time")


class ComplianceStatus(BaseModel):
    """Compliance status for a standard."""

    standard: ComplianceStandard = Field(..., description="Compliance standard")
    is_compliant: bool = Field(..., description="Whether compliant")
    violations_count: int = Field(default=0, description="Number of violations")
    last_audit_date: Optional[datetime] = Field(None, description="Last compliance audit")


class ProvenanceRecord(BaseModel):
    """Provenance tracking record."""

    operation: str = Field(..., description="Operation performed")
    timestamp: datetime = Field(..., description="Operation timestamp")
    input_hash: str = Field(..., description="SHA-256 hash of inputs")
    output_hash: str = Field(..., description="SHA-256 hash of outputs")
    tool_name: str = Field(..., description="Tool/calculator used")
    parameters: Dict[str, Any] = Field(default_factory=dict)


class FurnaceGuardianOutput(BaseModel):
    """Output data model for FurnaceGuardianAgent."""

    # Identification
    analysis_id: str = Field(..., description="Unique analysis identifier")
    furnace_id: str = Field(..., description="Furnace identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Safety Score
    safety_score: float = Field(..., ge=0, le=100, description="Overall safety score 0-100")
    risk_level: RiskLevel = Field(..., description="Risk level classification")

    # Component Scores
    interlock_score: float = Field(..., ge=0, le=100, description="Interlock subsystem score")
    purge_score: float = Field(..., ge=0, le=100, description="Purge verification score")
    flame_score: float = Field(..., ge=0, le=100, description="Flame supervision score")
    temperature_score: float = Field(..., ge=0, le=100, description="Temperature limits score")
    pressure_score: float = Field(..., ge=0, le=100, description="Pressure limits score")

    # Violations
    violations: List[SafetyViolation] = Field(
        default_factory=list,
        description="Safety violations detected"
    )

    # Corrective Actions
    corrective_actions: List[CorrectiveAction] = Field(
        default_factory=list,
        description="Recommended corrective actions"
    )

    # Compliance Status
    compliance_status: List[ComplianceStatus] = Field(
        default_factory=list,
        description="Compliance status per standard"
    )

    # Provenance
    provenance_chain: List[ProvenanceRecord] = Field(
        default_factory=list,
        description="Complete audit trail"
    )
    provenance_hash: str = Field(..., description="SHA-256 hash of provenance chain")

    # Processing Metadata
    processing_time_ms: float = Field(..., description="Processing duration in ms")
    validation_status: str = Field(..., description="PASS or FAIL")
    validation_errors: List[str] = Field(default_factory=list)


# =============================================================================
# FURNACE GUARDIAN AGENT
# =============================================================================

class FurnaceGuardianAgent:
    """
    GL-031: Furnace Guardian Agent (FURNACE-GUARDIAN).

    This agent monitors furnace safety systems and provides real-time
    safety scoring, violation detection, and compliance status per
    NFPA 86, API 560, and EN 746 standards.

    Zero-Hallucination Guarantee:
    - All calculations use deterministic formulas from published standards
    - No LLM inference in calculation path
    - Complete audit trail for regulatory compliance

    Attributes:
        AGENT_ID: Unique agent identifier (GL-031)
        AGENT_NAME: Agent name (FURNACE-GUARDIAN)
        VERSION: Agent version
    """

    AGENT_ID = "GL-031"
    AGENT_NAME = "FURNACE-GUARDIAN"
    VERSION = "1.0.0"
    DESCRIPTION = "Furnace Safety Monitoring Agent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the FurnaceGuardianAgent."""
        self.config = config or {}
        self._provenance_steps: List[Dict[str, Any]] = []
        self._validation_errors: List[str] = []
        self._violations: List[SafetyViolation] = []
        self._actions: List[CorrectiveAction] = []

        logger.info(
            f"FurnaceGuardianAgent initialized "
            f"(ID: {self.AGENT_ID}, Name: {self.AGENT_NAME}, Version: {self.VERSION})"
        )

    def run(self, input_data: FurnaceGuardianInput) -> FurnaceGuardianOutput:
        """
        Execute furnace safety analysis.

        This method performs comprehensive safety analysis:
        1. Validate all interlocks
        2. Verify purge cycle completion
        3. Check flame supervision
        4. Verify temperature limits
        5. Verify pressure limits
        6. Calculate overall safety score
        7. Generate corrective actions

        Args:
            input_data: Validated furnace input data

        Returns:
            Complete safety analysis output with provenance hash
        """
        start_time = datetime.utcnow()
        self._provenance_steps = []
        self._validation_errors = []
        self._violations = []
        self._actions = []

        logger.info(f"Starting safety analysis for furnace {input_data.furnace_id}")

        try:
            # Step 1: Validate interlocks
            interlocks_ok, interlocks_total = self._check_interlocks(input_data.interlocks)
            self._track_provenance(
                "interlock_validation",
                {"interlocks_count": len(input_data.interlocks)},
                {"ok": interlocks_ok, "total": interlocks_total},
                "interlock_validator"
            )

            # Step 2: Verify purge
            purge_valid = self._verify_purge(input_data.purge_data)
            self._track_provenance(
                "purge_verification",
                {"purge_status": input_data.purge_data.status.value if input_data.purge_data else "NONE"},
                {"valid": purge_valid},
                "purge_verifier"
            )

            # Step 3: Check flame supervision
            flame_ok, flame_quality = self._check_flame(input_data.flame_status)
            self._track_provenance(
                "flame_supervision",
                {
                    "detected": input_data.flame_status.is_detected,
                    "signal": input_data.flame_status.signal_strength
                },
                {"quality": flame_quality, "ok": flame_ok},
                "flame_checker"
            )

            # Step 4: Check temperatures
            temps_ok, temps_total = self._check_temperatures(input_data.temps)
            self._track_provenance(
                "temperature_check",
                {"sensor_count": len(input_data.temps)},
                {"ok": temps_ok, "total": temps_total},
                "temperature_checker"
            )

            # Step 5: Check pressures
            pressures_ok, pressures_total = self._check_pressures(input_data.pressures)
            self._track_provenance(
                "pressure_check",
                {"sensor_count": len(input_data.pressures)},
                {"ok": pressures_ok, "total": pressures_total},
                "pressure_checker"
            )

            # Step 6: Calculate safety score
            score_result = calculate_safety_score(
                interlocks_ok=interlocks_ok,
                interlocks_total=interlocks_total,
                purge_valid=purge_valid,
                flame_detected=flame_ok,
                flame_signal_quality=flame_quality,
                temps_in_range=temps_ok,
                temps_total=temps_total,
                pressures_in_range=pressures_ok,
                pressures_total=pressures_total
            )

            # Step 7: Check compliance
            compliance_list = self._check_compliance(input_data.compliance_standards)

            # Step 8: Generate corrective actions
            self._generate_corrective_actions()

            # Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash()

            # Processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Generate analysis ID
            analysis_id = (
                f"FG-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-"
                f"{hashlib.sha256(input_data.furnace_id.encode()).hexdigest()[:8]}"
            )

            output = FurnaceGuardianOutput(
                analysis_id=analysis_id,
                furnace_id=input_data.furnace_id,
                safety_score=score_result.score,
                risk_level=RiskLevel(score_result.risk_level),
                interlock_score=score_result.interlock_score,
                purge_score=score_result.purge_score,
                flame_score=score_result.flame_score,
                temperature_score=score_result.temperature_score,
                pressure_score=score_result.pressure_score,
                violations=self._violations,
                corrective_actions=self._actions,
                compliance_status=compliance_list,
                provenance_chain=[
                    ProvenanceRecord(
                        operation=s["operation"],
                        timestamp=s["timestamp"],
                        input_hash=s["input_hash"],
                        output_hash=s["output_hash"],
                        tool_name=s["tool_name"],
                        parameters=s.get("parameters", {})
                    )
                    for s in self._provenance_steps
                ],
                provenance_hash=provenance_hash,
                processing_time_ms=round(processing_time, 2),
                validation_status="PASS" if not self._validation_errors else "FAIL",
                validation_errors=self._validation_errors
            )

            logger.info(
                f"Safety analysis complete for {input_data.furnace_id}: "
                f"score={score_result.score}, risk={score_result.risk_level}, "
                f"violations={len(self._violations)} (duration: {processing_time:.1f}ms)"
            )

            return output

        except Exception as e:
            logger.error(f"Safety analysis failed: {str(e)}", exc_info=True)
            raise

    def _check_interlocks(self, interlocks: List[InterlockStatus]) -> tuple:
        """Check all interlocks and return (ok_count, total_count)."""
        if not interlocks:
            return 0, 0

        ok_count = 0
        for interlock in interlocks:
            if interlock.is_ok and not interlock.is_bypassed:
                ok_count += 1
            elif interlock.is_bypassed:
                self._violations.append(SafetyViolation(
                    violation_id=f"INT-BYP-{interlock.interlock_type.value}",
                    severity=ViolationSeverity.WARNING,
                    category="INTERLOCK_BYPASS",
                    description=f"Interlock {interlock.interlock_type.value} is bypassed: {interlock.bypass_reason}",
                    affected_component=interlock.interlock_type.value,
                    standard_reference="NFPA 86 Section 8.5"
                ))
            elif not interlock.is_ok:
                self._violations.append(SafetyViolation(
                    violation_id=f"INT-FAIL-{interlock.interlock_type.value}",
                    severity=ViolationSeverity.ALARM,
                    category="INTERLOCK_FAILURE",
                    description=f"Interlock {interlock.interlock_type.value} is in FAULT state",
                    affected_component=interlock.interlock_type.value,
                    standard_reference="NFPA 86 Section 8.5"
                ))

        return ok_count, len(interlocks)

    def _verify_purge(self, purge_data: Optional[PurgeData]) -> bool:
        """Verify purge cycle meets requirements."""
        if not purge_data:
            return True  # No purge data means not applicable

        if purge_data.status == PurgeStatus.COMPLETE:
            result = verify_purge_complete(
                airflow_cfm=purge_data.airflow_cfm,
                furnace_volume_cubic_feet=purge_data.furnace_volume_cubic_feet,
                purge_time_seconds=purge_data.purge_time_seconds,
                furnace_class=purge_data.furnace_class
            )
            if not result.is_valid:
                self._violations.append(SafetyViolation(
                    violation_id="PURGE-INCOMPLETE",
                    severity=ViolationSeverity.TRIP,
                    category="PURGE_FAILURE",
                    description=result.message,
                    affected_component="PURGE_SYSTEM",
                    standard_reference="NFPA 86 Section 8.6"
                ))
            return result.is_valid
        elif purge_data.status == PurgeStatus.BYPASSED:
            self._violations.append(SafetyViolation(
                violation_id="PURGE-BYPASSED",
                severity=ViolationSeverity.EMERGENCY,
                category="PURGE_BYPASS",
                description="Furnace purge cycle was bypassed - CRITICAL SAFETY VIOLATION",
                affected_component="PURGE_SYSTEM",
                standard_reference="NFPA 86 Section 8.6"
            ))
            return False
        elif purge_data.status == PurgeStatus.FAILED:
            self._violations.append(SafetyViolation(
                violation_id="PURGE-FAILED",
                severity=ViolationSeverity.TRIP,
                category="PURGE_FAILURE",
                description="Furnace purge cycle failed to complete",
                affected_component="PURGE_SYSTEM",
                standard_reference="NFPA 86 Section 8.6"
            ))
            return False

        return purge_data.status == PurgeStatus.NOT_STARTED

    def _check_flame(self, flame: FlameStatus) -> tuple:
        """Check flame detection status and quality."""
        if not flame.is_detected:
            return False, "POOR"

        quality, is_acceptable = calculate_flame_signal_quality(
            flame.signal_strength,
            flame.noise_level
        )

        if not is_acceptable:
            self._violations.append(SafetyViolation(
                violation_id="FLAME-QUALITY",
                severity=ViolationSeverity.WARNING,
                category="FLAME_SUPERVISION",
                description=f"Flame signal quality is {quality} - signal strength: {flame.signal_strength}",
                affected_component=flame.detector_id,
                standard_reference="NFPA 86 Section 8.8"
            ))

        return is_acceptable, quality

    def _check_temperatures(self, temps: List[TemperatureReading]) -> tuple:
        """Check all temperature readings against limits."""
        if not temps:
            return 0, 0

        ok_count = 0
        for temp in temps:
            status, is_safe = check_temperature_limits(
                temp.value_celsius,
                temp.low_limit,
                temp.high_limit,
                temp.high_high_limit
            )
            if is_safe:
                ok_count += 1
            else:
                severity = ViolationSeverity.TRIP if status == "TRIP" else ViolationSeverity.ALARM
                self._violations.append(SafetyViolation(
                    violation_id=f"TEMP-{status}-{temp.sensor_id}",
                    severity=severity,
                    category="TEMPERATURE_LIMIT",
                    description=f"Temperature {temp.sensor_id}: {temp.value_celsius}C is {status}",
                    affected_component=temp.sensor_id,
                    standard_reference="NFPA 86 Section 8.7"
                ))

        return ok_count, len(temps)

    def _check_pressures(self, pressures: List[PressureReading]) -> tuple:
        """Check all pressure readings against limits."""
        if not pressures:
            return 0, 0

        ok_count = 0
        for pressure in pressures:
            status, is_safe = check_pressure_limits(
                pressure.value_kpa,
                pressure.low_limit,
                pressure.high_limit,
                pressure.low_low_limit,
                pressure.high_high_limit
            )
            if is_safe:
                ok_count += 1
            else:
                severity = ViolationSeverity.TRIP if "TRIP" in status else ViolationSeverity.ALARM
                self._violations.append(SafetyViolation(
                    violation_id=f"PRES-{status}-{pressure.sensor_id}",
                    severity=severity,
                    category="PRESSURE_LIMIT",
                    description=f"Pressure {pressure.sensor_id}: {pressure.value_kpa}kPa is {status}",
                    affected_component=pressure.sensor_id,
                    standard_reference="API 560 Section 6"
                ))

        return ok_count, len(pressures)

    def _check_compliance(self, standards: List[ComplianceStandard]) -> List[ComplianceStatus]:
        """Check compliance with specified standards."""
        result = []
        for standard in standards:
            violations_for_standard = [
                v for v in self._violations
                if v.standard_reference and standard.value in v.standard_reference
            ]
            result.append(ComplianceStatus(
                standard=standard,
                is_compliant=len(violations_for_standard) == 0,
                violations_count=len(violations_for_standard)
            ))
        return result

    def _generate_corrective_actions(self):
        """Generate corrective actions based on violations."""
        action_id = 0
        for violation in self._violations:
            action_id += 1
            priority = "IMMEDIATE" if violation.severity in [ViolationSeverity.EMERGENCY, ViolationSeverity.TRIP] else \
                       "HIGH" if violation.severity == ViolationSeverity.ALARM else "MEDIUM"

            action_map = {
                "INTERLOCK_BYPASS": "Remove bypass and verify interlock function",
                "INTERLOCK_FAILURE": "Repair or replace failed interlock",
                "PURGE_FAILURE": "Investigate purge system, check airflow and timing",
                "PURGE_BYPASS": "IMMEDIATE SHUTDOWN required - purge bypass is critical violation",
                "FLAME_SUPERVISION": "Clean or calibrate flame detector",
                "TEMPERATURE_LIMIT": "Investigate temperature excursion cause",
                "PRESSURE_LIMIT": "Check pressure relief and control systems"
            }

            self._actions.append(CorrectiveAction(
                action_id=f"CA-{action_id:03d}",
                priority=priority,
                description=action_map.get(violation.category, "Investigate and correct"),
                affected_component=violation.affected_component,
                estimated_time_minutes=30 if priority == "IMMEDIATE" else 60
            ))

    def _track_provenance(
        self,
        operation: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        tool_name: str
    ):
        """Track a calculation step for audit trail."""
        input_str = json.dumps(inputs, sort_keys=True, default=str)
        output_str = json.dumps(outputs, sort_keys=True, default=str)

        self._provenance_steps.append({
            "operation": operation,
            "timestamp": datetime.utcnow(),
            "input_hash": hashlib.sha256(input_str.encode()).hexdigest(),
            "output_hash": hashlib.sha256(output_str.encode()).hexdigest(),
            "tool_name": tool_name,
            "parameters": inputs
        })

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash of complete provenance chain."""
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "agent_name": self.AGENT_NAME,
            "version": self.VERSION,
            "steps": [
                {
                    "operation": s["operation"],
                    "input_hash": s["input_hash"],
                    "output_hash": s["output_hash"]
                }
                for s in self._provenance_steps
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        json_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


# =============================================================================
# PACK SPECIFICATION
# =============================================================================

PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-031",
    "name": "FURNACE-GUARDIAN - Furnace Safety Monitoring Agent",
    "version": "1.0.0",
    "summary": "Furnace safety monitoring with interlock validation, purge verification, and flame supervision",
    "tags": [
        "furnace",
        "safety",
        "interlock",
        "purge",
        "flame-detection",
        "NFPA-86",
        "API-560",
        "EN-746"
    ],
    "owners": ["process-heat-safety-team"],
    "compute": {
        "entrypoint": "python://agents.gl_031_furnace_guardian.agent:FurnaceGuardianAgent",
        "deterministic": True
    },
    "standards": [
        {"ref": "NFPA 86", "description": "Standard for Ovens and Furnaces"},
        {"ref": "API 560", "description": "Fired Heaters for General Refinery Service"},
        {"ref": "EN 746", "description": "Industrial Thermoprocessing Equipment"}
    ],
    "provenance": {
        "calculation_verified": True,
        "enable_audit": True
    }
}
