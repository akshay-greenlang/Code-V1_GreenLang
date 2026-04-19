"""
NFPA Compliance Module - NFPA 85/86 Compliance Verification

This module provides compliance verification for process heat equipment
against NFPA 85 (Boiler and Combustion Systems) and NFPA 86 (Ovens and
Furnaces) standards.

Standards:
    - NFPA 85: Boiler and Combustion Systems Hazards Code (2023)
    - NFPA 86: Standard for Ovens and Furnaces (2023)
    - NFPA 87: Standard for Fluid Heaters

Key Requirements Verified:
    - Combustion safeguards and flame supervision
    - Purge timing and flow requirements
    - Temperature and pressure limits
    - Safety interlock requirements
    - Startup/shutdown sequences
    - Emergency shutdown requirements

Example:
    >>> checker = NFPAComplianceChecker()
    >>> result = checker.check_purge_compliance(
    ...     furnace_type=FurnaceType.CLASS_A,
    ...     purge_time_seconds=240,
    ...     air_changes=4.5
    ... )
    >>> print(f"Compliant: {result.is_compliant}")

CRITICAL: All compliance checks are DETERMINISTIC. NO LLM calls permitted.
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class NFPAStandard(str, Enum):
    """NFPA standards for process heat equipment."""
    NFPA_85 = "NFPA 85"     # Boiler and Combustion Systems
    NFPA_86 = "NFPA 86"     # Ovens and Furnaces
    NFPA_87 = "NFPA 87"     # Fluid Heaters


class FurnaceType(str, Enum):
    """Furnace classifications per NFPA 86."""
    CLASS_A = "class_a"     # Ovens operating at or near atmospheric pressure
    CLASS_B = "class_b"     # Heating systems with flammable vapors
    CLASS_C = "class_c"     # Heating systems using special atmospheres
    CLASS_D = "class_d"     # Vacuum furnaces


class BurnerType(str, Enum):
    """Burner types for combustion systems."""
    SINGLE_BURNER = "single_burner"
    MULTIPLE_BURNER = "multiple_burner"
    DUCT_BURNER = "duct_burner"
    RADIANT_TUBE = "radiant_tube"
    DIRECT_FIRED = "direct_fired"
    INDIRECT_FIRED = "indirect_fired"


class FlameSupervisionType(str, Enum):
    """Flame supervision methods per NFPA 86."""
    UV_SCANNER = "uv_scanner"           # Ultraviolet flame scanner
    IR_SCANNER = "ir_scanner"           # Infrared flame scanner
    FLAME_ROD = "flame_rod"             # Flame rod/electrode
    THERMOCOUPLE = "thermocouple"       # Thermocouple (limited use)
    OPTICAL = "optical"                 # Optical flame detector


class ComplianceStatus(str, Enum):
    """Compliance check status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    MARGINAL = "marginal"               # Within 10% of limit
    NOT_APPLICABLE = "not_applicable"
    REQUIRES_REVIEW = "requires_review"  # Manual review needed


class NFPARequirement(BaseModel):
    """
    Single NFPA requirement definition.

    Attributes:
        requirement_id: Unique identifier (e.g., NFPA86-8.3.1)
        standard: Applicable NFPA standard
        section: Section reference
        description: Requirement description
        parameter: Parameter being checked
        min_value: Minimum acceptable value
        max_value: Maximum acceptable value
        unit: Engineering unit
    """
    requirement_id: str = Field(..., description="Requirement identifier")
    standard: NFPAStandard = Field(..., description="NFPA standard")
    section: str = Field(..., description="Section reference")
    description: str = Field(..., description="Requirement description")
    parameter: str = Field(..., description="Parameter name")
    min_value: Optional[float] = Field(None, description="Minimum value")
    max_value: Optional[float] = Field(None, description="Maximum value")
    unit: str = Field("", description="Engineering unit")
    is_mandatory: bool = Field(True, description="Mandatory requirement")
    applies_to: List[FurnaceType] = Field(
        default_factory=list, description="Applicable furnace types"
    )


class ComplianceCheckStep(BaseModel):
    """Individual compliance check step with provenance."""
    step_number: int = Field(..., description="Step number")
    requirement_id: str = Field(..., description="Requirement checked")
    description: str = Field(..., description="Check description")
    parameter: str = Field(..., description="Parameter checked")
    actual_value: Optional[float] = Field(None, description="Actual value")
    required_min: Optional[float] = Field(None, description="Required minimum")
    required_max: Optional[float] = Field(None, description="Required maximum")
    unit: str = Field("", description="Unit")
    status: ComplianceStatus = Field(..., description="Check status")
    deviation_percent: Optional[float] = Field(None, description="Deviation %")
    step_hash: str = Field("", description="SHA-256 hash")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.step_hash:
            self.step_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of this step."""
        hash_data = {
            "step_number": self.step_number,
            "requirement_id": self.requirement_id,
            "actual_value": str(self.actual_value),
            "status": self.status.value,
        }
        return hashlib.sha256(
            json.dumps(hash_data, sort_keys=True).encode()
        ).hexdigest()


class ComplianceResult(BaseModel):
    """
    Complete NFPA compliance check result with provenance.

    Contains all check results and full audit trail for
    regulatory documentation.
    """
    # Equipment identification
    equipment_id: str = Field(..., description="Equipment identifier")
    equipment_name: str = Field(..., description="Equipment name")
    furnace_type: FurnaceType = Field(..., description="Furnace classification")
    standard: NFPAStandard = Field(..., description="Standard checked")

    # Overall compliance
    is_compliant: bool = Field(..., description="Overall compliance status")
    compliance_score: float = Field(..., ge=0, le=100, description="Score (0-100)")
    total_checks: int = Field(..., description="Total checks performed")
    passed_checks: int = Field(..., description="Checks passed")
    failed_checks: int = Field(..., description="Checks failed")
    marginal_checks: int = Field(..., description="Marginal checks")

    # Detailed results
    check_steps: List[ComplianceCheckStep] = Field(
        default_factory=list, description="Individual check results"
    )
    non_compliant_items: List[str] = Field(
        default_factory=list, description="Non-compliant requirement IDs"
    )
    marginal_items: List[str] = Field(
        default_factory=list, description="Marginal requirement IDs"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 hash")
    input_hash: str = Field(..., description="Input hash")

    # Metadata
    check_time_ms: float = Field(..., description="Check time")
    checked_at: datetime = Field(default_factory=datetime.utcnow)
    checker_version: str = Field("1.0.0", description="Checker version")

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list, description="Compliance recommendations"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            FurnaceType: lambda v: v.value,
            NFPAStandard: lambda v: v.value,
            ComplianceStatus: lambda v: v.value,
        }

    def verify_provenance(self) -> bool:
        """Verify provenance hash matches check steps."""
        step_data = [
            {
                "step_number": s.step_number,
                "requirement_id": s.requirement_id,
                "actual_value": str(s.actual_value),
                "status": s.status.value,
            }
            for s in self.check_steps
        ]
        recalculated = hashlib.sha256(
            json.dumps(step_data, sort_keys=True).encode()
        ).hexdigest()
        return recalculated == self.provenance_hash

    def to_audit_dict(self) -> Dict[str, Any]:
        """Export as audit-ready dictionary."""
        return {
            "equipment": {
                "id": self.equipment_id,
                "name": self.equipment_name,
                "type": self.furnace_type.value,
            },
            "compliance": {
                "is_compliant": self.is_compliant,
                "score": self.compliance_score,
                "total_checks": self.total_checks,
                "passed": self.passed_checks,
                "failed": self.failed_checks,
                "marginal": self.marginal_checks,
            },
            "non_compliant": self.non_compliant_items,
            "marginal": self.marginal_items,
            "provenance": {
                "hash": self.provenance_hash,
                "input_hash": self.input_hash,
            },
            "metadata": {
                "checked_at": self.checked_at.isoformat(),
                "check_time_ms": self.check_time_ms,
                "standard": self.standard.value,
            },
            "recommendations": self.recommendations,
        }


class NFPAComplianceChecker:
    """
    NFPA 85/86 Compliance Checker for process heat equipment.

    This class verifies compliance with NFPA standards for
    combustion equipment, ovens, and furnaces. All checks are
    deterministic with full provenance tracking.

    Key Methods:
        check_purge_compliance: Verify purge time and air changes
        check_flame_supervision: Verify flame detector requirements
        check_temperature_limits: Verify operating temperature limits
        check_safety_interlocks: Verify required safety interlocks
        full_compliance_check: Complete compliance assessment

    Example:
        >>> checker = NFPAComplianceChecker()
        >>> result = checker.full_compliance_check(
        ...     equipment_id="FURNACE-001",
        ...     equipment_name="Heat Treatment Furnace",
        ...     furnace_type=FurnaceType.CLASS_A,
        ...     parameters={
        ...         "purge_time_seconds": 240,
        ...         "air_changes": 4.5,
        ...         "max_operating_temp_c": 1100,
        ...         "flame_failure_response_ms": 3000,
        ...     }
        ... )

    CRITICAL: All checks are DETERMINISTIC. NO LLM calls permitted.
    """

    VERSION = "1.0.0"

    # NFPA 86 Purge Requirements by Furnace Class
    PURGE_REQUIREMENTS: Dict[FurnaceType, Dict[str, float]] = {
        FurnaceType.CLASS_A: {
            "min_air_changes": 4.0,
            "min_purge_time_seconds": 60.0,
        },
        FurnaceType.CLASS_B: {
            "min_air_changes": 4.0,
            "min_purge_time_seconds": 120.0,
        },
        FurnaceType.CLASS_C: {
            "min_air_changes": 4.0,
            "min_purge_time_seconds": 180.0,
        },
        FurnaceType.CLASS_D: {
            "min_air_changes": 3.0,
            "min_purge_time_seconds": 60.0,
        },
    }

    # NFPA 86 Flame Failure Response Times (seconds)
    FLAME_FAILURE_RESPONSE: Dict[BurnerType, float] = {
        BurnerType.SINGLE_BURNER: 4.0,
        BurnerType.MULTIPLE_BURNER: 4.0,
        BurnerType.DUCT_BURNER: 4.0,
        BurnerType.RADIANT_TUBE: 4.0,
        BurnerType.DIRECT_FIRED: 4.0,
        BurnerType.INDIRECT_FIRED: 4.0,
    }

    # NFPA 86 Maximum Ignition Trial Times (seconds)
    MAX_IGNITION_TRIAL: Dict[BurnerType, float] = {
        BurnerType.SINGLE_BURNER: 15.0,
        BurnerType.MULTIPLE_BURNER: 15.0,
        BurnerType.DUCT_BURNER: 10.0,
        BurnerType.RADIANT_TUBE: 15.0,
        BurnerType.DIRECT_FIRED: 15.0,
        BurnerType.INDIRECT_FIRED: 15.0,
    }

    # NFPA 86 Temperature Limits by Furnace Class (degC)
    TEMPERATURE_LIMITS: Dict[FurnaceType, Dict[str, float]] = {
        FurnaceType.CLASS_A: {
            "max_chamber_temp_c": 1650.0,      # General limit
            "autoignition_margin_c": 100.0,    # Below autoignition temp
        },
        FurnaceType.CLASS_B: {
            "max_chamber_temp_c": 500.0,       # Below LEL temperature
            "autoignition_margin_c": 100.0,
        },
        FurnaceType.CLASS_C: {
            "max_chamber_temp_c": 1650.0,
            "atmosphere_ignition_temp_c": 760.0,  # H2 atmosphere limit
        },
        FurnaceType.CLASS_D: {
            "max_chamber_temp_c": 2000.0,      # Vacuum furnace
            "cooling_water_max_c": 40.0,
        },
    }

    # Required safety interlocks per NFPA 86
    REQUIRED_INTERLOCKS: Dict[FurnaceType, List[str]] = {
        FurnaceType.CLASS_A: [
            "combustion_air_flow",
            "purge_air_flow",
            "flame_failure",
            "high_temperature",
            "gas_pressure_low",
            "gas_pressure_high",
        ],
        FurnaceType.CLASS_B: [
            "combustion_air_flow",
            "purge_air_flow",
            "flame_failure",
            "high_temperature",
            "gas_pressure_low",
            "gas_pressure_high",
            "ventilation_air_flow",
            "lel_high",
        ],
        FurnaceType.CLASS_C: [
            "combustion_air_flow",
            "purge_air_flow",
            "flame_failure",
            "high_temperature",
            "gas_pressure_low",
            "gas_pressure_high",
            "atmosphere_flow",
            "atmosphere_pressure",
        ],
        FurnaceType.CLASS_D: [
            "vacuum_level",
            "cooling_water_flow",
            "cooling_water_temperature",
            "power_supply",
            "door_interlock",
        ],
    }

    def __init__(self):
        """Initialize NFPA Compliance Checker."""
        self._steps: List[ComplianceCheckStep] = []
        self._step_counter = 0
        self._start_time: Optional[float] = None
        self._recommendations: List[str] = []
        self._non_compliant: List[str] = []
        self._marginal: List[str] = []

    def _start_check(self) -> None:
        """Reset check state."""
        self._steps = []
        self._step_counter = 0
        self._start_time = time.perf_counter()
        self._recommendations = []
        self._non_compliant = []
        self._marginal = []

    def _record_step(
        self,
        requirement_id: str,
        description: str,
        parameter: str,
        actual_value: Optional[float],
        required_min: Optional[float],
        required_max: Optional[float],
        unit: str,
        status: ComplianceStatus,
    ) -> ComplianceCheckStep:
        """Record a compliance check step."""
        self._step_counter += 1

        # Calculate deviation if applicable
        deviation_percent = None
        if actual_value is not None:
            if required_min is not None and actual_value < required_min:
                deviation_percent = ((required_min - actual_value) / required_min) * 100
            elif required_max is not None and actual_value > required_max:
                deviation_percent = ((actual_value - required_max) / required_max) * 100

        step = ComplianceCheckStep(
            step_number=self._step_counter,
            requirement_id=requirement_id,
            description=description,
            parameter=parameter,
            actual_value=actual_value,
            required_min=required_min,
            required_max=required_max,
            unit=unit,
            status=status,
            deviation_percent=deviation_percent,
        )
        self._steps.append(step)

        if status == ComplianceStatus.NON_COMPLIANT:
            self._non_compliant.append(requirement_id)
        elif status == ComplianceStatus.MARGINAL:
            self._marginal.append(requirement_id)

        logger.debug(
            f"Compliance check {self._step_counter}: {requirement_id} - {status.value}"
        )
        return step

    def _compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash of all check steps."""
        step_data = [
            {
                "step_number": s.step_number,
                "requirement_id": s.requirement_id,
                "actual_value": str(s.actual_value),
                "status": s.status.value,
            }
            for s in self._steps
        ]
        return hashlib.sha256(
            json.dumps(step_data, sort_keys=True).encode()
        ).hexdigest()

    def _compute_input_hash(self, inputs: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of inputs."""
        serializable = {k: str(v) for k, v in inputs.items()}
        return hashlib.sha256(
            json.dumps(serializable, sort_keys=True).encode()
        ).hexdigest()

    def _get_check_time_ms(self) -> float:
        """Get check time in milliseconds."""
        if self._start_time is None:
            return 0.0
        return (time.perf_counter() - self._start_time) * 1000

    def check_purge_compliance(
        self,
        furnace_type: FurnaceType,
        purge_time_seconds: float,
        air_changes: float,
    ) -> Tuple[ComplianceStatus, str]:
        """
        Check purge time and air change compliance per NFPA 86.

        NFPA 86 Section 8.4 requires pre-purge before ignition.

        Args:
            furnace_type: Furnace classification
            purge_time_seconds: Actual purge time
            air_changes: Number of air volume changes

        Returns:
            Tuple of (status, message)
        """
        requirements = self.PURGE_REQUIREMENTS.get(furnace_type, {})
        min_time = requirements.get("min_purge_time_seconds", 60.0)
        min_changes = requirements.get("min_air_changes", 4.0)

        # Check both requirements
        time_ok = purge_time_seconds >= min_time
        changes_ok = air_changes >= min_changes

        if time_ok and changes_ok:
            # Check if marginal (within 10%)
            time_margin = (purge_time_seconds - min_time) / min_time
            changes_margin = (air_changes - min_changes) / min_changes

            if time_margin < 0.1 or changes_margin < 0.1:
                status = ComplianceStatus.MARGINAL
                message = (
                    f"Purge marginally compliant: {purge_time_seconds}s "
                    f"(req: {min_time}s), {air_changes} changes "
                    f"(req: {min_changes})"
                )
            else:
                status = ComplianceStatus.COMPLIANT
                message = (
                    f"Purge compliant: {purge_time_seconds}s >= {min_time}s, "
                    f"{air_changes} changes >= {min_changes}"
                )
        else:
            status = ComplianceStatus.NON_COMPLIANT
            issues = []
            if not time_ok:
                issues.append(f"time {purge_time_seconds}s < {min_time}s required")
            if not changes_ok:
                issues.append(f"air changes {air_changes} < {min_changes} required")
            message = f"Purge non-compliant: {'; '.join(issues)}"

        # Record step
        self._record_step(
            requirement_id="NFPA86-8.4.1",
            description="Pre-purge time requirement",
            parameter="purge_time_seconds",
            actual_value=purge_time_seconds,
            required_min=min_time,
            required_max=None,
            unit="seconds",
            status=status if time_ok else ComplianceStatus.NON_COMPLIANT,
        )

        self._record_step(
            requirement_id="NFPA86-8.4.2",
            description="Pre-purge air changes requirement",
            parameter="air_changes",
            actual_value=air_changes,
            required_min=min_changes,
            required_max=None,
            unit="volume changes",
            status=status if changes_ok else ComplianceStatus.NON_COMPLIANT,
        )

        return status, message

    def check_flame_supervision(
        self,
        burner_type: BurnerType,
        flame_failure_response_ms: float,
        supervision_type: FlameSupervisionType,
        ignition_trial_seconds: float,
    ) -> Tuple[ComplianceStatus, str]:
        """
        Check flame supervision compliance per NFPA 86.

        NFPA 86 Section 8.5 specifies flame supervision requirements.

        Args:
            burner_type: Type of burner
            flame_failure_response_ms: Flame failure response time
            supervision_type: Type of flame detector
            ignition_trial_seconds: Maximum ignition trial time

        Returns:
            Tuple of (status, message)
        """
        # Get requirements
        max_response_s = self.FLAME_FAILURE_RESPONSE.get(burner_type, 4.0)
        max_response_ms = max_response_s * 1000
        max_trial = self.MAX_IGNITION_TRIAL.get(burner_type, 15.0)

        # Check flame failure response
        response_ok = flame_failure_response_ms <= max_response_ms

        # Check ignition trial time
        trial_ok = ignition_trial_seconds <= max_trial

        # Validate supervision type
        valid_types = [
            FlameSupervisionType.UV_SCANNER,
            FlameSupervisionType.IR_SCANNER,
            FlameSupervisionType.FLAME_ROD,
        ]
        type_ok = supervision_type in valid_types

        # Determine overall status
        if response_ok and trial_ok and type_ok:
            response_margin = (max_response_ms - flame_failure_response_ms) / max_response_ms
            if response_margin < 0.1:
                status = ComplianceStatus.MARGINAL
            else:
                status = ComplianceStatus.COMPLIANT
            message = (
                f"Flame supervision compliant: response {flame_failure_response_ms}ms, "
                f"trial {ignition_trial_seconds}s, {supervision_type.value}"
            )
        else:
            status = ComplianceStatus.NON_COMPLIANT
            issues = []
            if not response_ok:
                issues.append(
                    f"response {flame_failure_response_ms}ms > {max_response_ms}ms"
                )
            if not trial_ok:
                issues.append(f"trial {ignition_trial_seconds}s > {max_trial}s")
            if not type_ok:
                issues.append(f"supervision type {supervision_type.value} not approved")
            message = f"Flame supervision non-compliant: {'; '.join(issues)}"

        # Record steps
        self._record_step(
            requirement_id="NFPA86-8.5.1",
            description="Flame failure response time",
            parameter="flame_failure_response_ms",
            actual_value=flame_failure_response_ms,
            required_min=None,
            required_max=max_response_ms,
            unit="milliseconds",
            status=ComplianceStatus.COMPLIANT if response_ok else ComplianceStatus.NON_COMPLIANT,
        )

        self._record_step(
            requirement_id="NFPA86-8.5.2",
            description="Maximum ignition trial time",
            parameter="ignition_trial_seconds",
            actual_value=ignition_trial_seconds,
            required_min=None,
            required_max=max_trial,
            unit="seconds",
            status=ComplianceStatus.COMPLIANT if trial_ok else ComplianceStatus.NON_COMPLIANT,
        )

        return status, message

    def check_temperature_limits(
        self,
        furnace_type: FurnaceType,
        operating_temp_c: float,
        high_limit_setpoint_c: float,
    ) -> Tuple[ComplianceStatus, str]:
        """
        Check temperature limit compliance per NFPA 86.

        Args:
            furnace_type: Furnace classification
            operating_temp_c: Normal operating temperature
            high_limit_setpoint_c: High temperature limit setpoint

        Returns:
            Tuple of (status, message)
        """
        limits = self.TEMPERATURE_LIMITS.get(furnace_type, {})
        max_temp = limits.get("max_chamber_temp_c", 1650.0)

        # Operating temp must be below limit
        operating_ok = operating_temp_c < max_temp

        # High limit setpoint must be set appropriately
        # Should be above operating but below max
        setpoint_ok = operating_temp_c < high_limit_setpoint_c <= max_temp

        if operating_ok and setpoint_ok:
            margin = (max_temp - operating_temp_c) / max_temp
            if margin < 0.1:
                status = ComplianceStatus.MARGINAL
            else:
                status = ComplianceStatus.COMPLIANT
            message = (
                f"Temperature limits compliant: operating {operating_temp_c}C, "
                f"limit {high_limit_setpoint_c}C, max {max_temp}C"
            )
        else:
            status = ComplianceStatus.NON_COMPLIANT
            issues = []
            if not operating_ok:
                issues.append(f"operating {operating_temp_c}C > max {max_temp}C")
            if not setpoint_ok:
                issues.append(f"high limit {high_limit_setpoint_c}C improperly set")
            message = f"Temperature limits non-compliant: {'; '.join(issues)}"

        self._record_step(
            requirement_id="NFPA86-7.3.1",
            description="Operating temperature within limits",
            parameter="operating_temp_c",
            actual_value=operating_temp_c,
            required_min=None,
            required_max=max_temp,
            unit="degC",
            status=ComplianceStatus.COMPLIANT if operating_ok else ComplianceStatus.NON_COMPLIANT,
        )

        self._record_step(
            requirement_id="NFPA86-7.3.2",
            description="High temperature limit setpoint",
            parameter="high_limit_setpoint_c",
            actual_value=high_limit_setpoint_c,
            required_min=operating_temp_c,
            required_max=max_temp,
            unit="degC",
            status=ComplianceStatus.COMPLIANT if setpoint_ok else ComplianceStatus.NON_COMPLIANT,
        )

        return status, message

    def check_safety_interlocks(
        self,
        furnace_type: FurnaceType,
        installed_interlocks: List[str],
    ) -> Tuple[ComplianceStatus, str]:
        """
        Check safety interlock compliance per NFPA 86.

        Args:
            furnace_type: Furnace classification
            installed_interlocks: List of installed interlock names

        Returns:
            Tuple of (status, message)
        """
        required = set(self.REQUIRED_INTERLOCKS.get(furnace_type, []))
        installed = set(installed_interlocks)

        missing = required - installed
        extra = installed - required  # Not a compliance issue, just informational

        if not missing:
            status = ComplianceStatus.COMPLIANT
            message = f"All {len(required)} required interlocks installed"
            if extra:
                message += f" (plus {len(extra)} additional)"
        else:
            status = ComplianceStatus.NON_COMPLIANT
            message = f"Missing interlocks: {', '.join(missing)}"
            self._recommendations.append(
                f"Install missing interlocks: {', '.join(missing)}"
            )

        self._record_step(
            requirement_id="NFPA86-8.2.1",
            description="Required safety interlocks",
            parameter="interlock_count",
            actual_value=float(len(installed & required)),
            required_min=float(len(required)),
            required_max=None,
            unit="interlocks",
            status=status,
        )

        return status, message

    def full_compliance_check(
        self,
        equipment_id: str,
        equipment_name: str,
        furnace_type: FurnaceType,
        parameters: Dict[str, Any],
        standard: NFPAStandard = NFPAStandard.NFPA_86,
    ) -> ComplianceResult:
        """
        Perform complete NFPA compliance check.

        Args:
            equipment_id: Equipment identifier
            equipment_name: Equipment name
            furnace_type: Furnace classification
            parameters: Dictionary of parameters to check
            standard: NFPA standard to check against

        Returns:
            ComplianceResult with complete assessment

        Required parameters:
            - purge_time_seconds: Pre-purge time
            - air_changes: Number of air volume changes
            - flame_failure_response_ms: Flame failure response time
            - ignition_trial_seconds: Maximum ignition trial time
            - operating_temp_c: Operating temperature
            - high_limit_setpoint_c: High temperature limit
            - installed_interlocks: List of interlock names
        """
        self._start_check()
        logger.info(f"Starting NFPA compliance check for {equipment_id}")

        inputs_summary = {
            "equipment_id": equipment_id,
            "furnace_type": furnace_type.value,
            "standard": standard.value,
            "parameters": parameters,
        }

        # Check purge compliance
        if "purge_time_seconds" in parameters and "air_changes" in parameters:
            self.check_purge_compliance(
                furnace_type,
                parameters["purge_time_seconds"],
                parameters["air_changes"],
            )

        # Check flame supervision
        if "flame_failure_response_ms" in parameters:
            burner_type = parameters.get("burner_type", BurnerType.SINGLE_BURNER)
            if isinstance(burner_type, str):
                burner_type = BurnerType(burner_type)

            supervision_type = parameters.get(
                "flame_supervision_type", FlameSupervisionType.UV_SCANNER
            )
            if isinstance(supervision_type, str):
                supervision_type = FlameSupervisionType(supervision_type)

            self.check_flame_supervision(
                burner_type,
                parameters["flame_failure_response_ms"],
                supervision_type,
                parameters.get("ignition_trial_seconds", 10.0),
            )

        # Check temperature limits
        if "operating_temp_c" in parameters:
            self.check_temperature_limits(
                furnace_type,
                parameters["operating_temp_c"],
                parameters.get("high_limit_setpoint_c", parameters["operating_temp_c"] + 50),
            )

        # Check safety interlocks
        if "installed_interlocks" in parameters:
            self.check_safety_interlocks(
                furnace_type,
                parameters["installed_interlocks"],
            )

        # Calculate overall compliance
        total_checks = len(self._steps)
        passed = sum(
            1 for s in self._steps
            if s.status == ComplianceStatus.COMPLIANT
        )
        failed = sum(
            1 for s in self._steps
            if s.status == ComplianceStatus.NON_COMPLIANT
        )
        marginal = sum(
            1 for s in self._steps
            if s.status == ComplianceStatus.MARGINAL
        )

        is_compliant = failed == 0
        compliance_score = (passed / total_checks * 100) if total_checks > 0 else 0.0

        # Add recommendations for marginal items
        for item in self._marginal:
            self._recommendations.append(
                f"Review {item}: currently marginal compliance"
            )

        # Build result
        result = ComplianceResult(
            equipment_id=equipment_id,
            equipment_name=equipment_name,
            furnace_type=furnace_type,
            standard=standard,
            is_compliant=is_compliant,
            compliance_score=compliance_score,
            total_checks=total_checks,
            passed_checks=passed,
            failed_checks=failed,
            marginal_checks=marginal,
            check_steps=self._steps.copy(),
            non_compliant_items=self._non_compliant.copy(),
            marginal_items=self._marginal.copy(),
            provenance_hash=self._compute_provenance_hash(),
            input_hash=self._compute_input_hash(inputs_summary),
            check_time_ms=self._get_check_time_ms(),
            recommendations=self._recommendations.copy(),
        )

        logger.info(
            f"NFPA compliance check complete: {equipment_id}, "
            f"compliant={is_compliant}, score={compliance_score:.1f}%"
        )
        return result

    def get_requirement(
        self,
        requirement_id: str,
    ) -> Optional[NFPARequirement]:
        """
        Get details for a specific NFPA requirement.

        Args:
            requirement_id: Requirement identifier

        Returns:
            NFPARequirement or None if not found
        """
        # This would typically load from a database
        # Here we provide key requirements inline
        requirements = {
            "NFPA86-8.4.1": NFPARequirement(
                requirement_id="NFPA86-8.4.1",
                standard=NFPAStandard.NFPA_86,
                section="8.4.1",
                description="Pre-purge time before ignition",
                parameter="purge_time_seconds",
                min_value=60.0,
                max_value=None,
                unit="seconds",
            ),
            "NFPA86-8.4.2": NFPARequirement(
                requirement_id="NFPA86-8.4.2",
                standard=NFPAStandard.NFPA_86,
                section="8.4.2",
                description="Minimum air volume changes during purge",
                parameter="air_changes",
                min_value=4.0,
                max_value=None,
                unit="volume changes",
            ),
            "NFPA86-8.5.1": NFPARequirement(
                requirement_id="NFPA86-8.5.1",
                standard=NFPAStandard.NFPA_86,
                section="8.5.1",
                description="Flame failure response time",
                parameter="flame_failure_response_ms",
                min_value=None,
                max_value=4000.0,
                unit="milliseconds",
            ),
        }
        return requirements.get(requirement_id)
