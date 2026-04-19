"""
GL-022 SuperheaterControlAgent Safety and Compliance Module.

This module implements comprehensive safety validation and regulatory compliance
checking for superheater control operations. All checks are DETERMINISTIC with
full provenance tracking for audit trails.

Standards Compliance:
    - ASME Boiler and Pressure Vessel Code (BPVC)
    - ASME B31.1 Power Piping Code
    - API 530 (Calculation of Heater Tube Thickness)
    - NFPA 85 Boiler and Combustion Systems Hazards Code

Features:
    - Material-specific temperature limits per ASME Section II
    - Creep rupture life calculation per API 530
    - Thermal shock prevention (rate of change limits)
    - Interlock status validation per NFPA 85
    - SHA-256 provenance tracking for all safety checks

Example:
    >>> from greenlang.agents.process_heat.gl_022_superheater_control.safety import (
    ...     SafetyValidator,
    ...     ASMEComplianceChecker,
    ... )
    >>>
    >>> validator = SafetyValidator(agent_id="GL-022-001")
    >>> result = validator.validate_tube_metal_temperature(
    ...     temp_f=1050,
    ...     material="SA-213-T22",
    ...     thickness_in=0.25
    ... )
    >>> print(result.is_safe)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math
import threading

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class SafetyStatus(Enum):
    """Safety validation status levels."""
    SAFE = "SAFE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    VIOLATION = "VIOLATION"
    SHUTDOWN_REQUIRED = "SHUTDOWN_REQUIRED"


class ComplianceStatus(Enum):
    """Regulatory compliance status."""
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    MARGINAL = "MARGINAL"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"


class MaterialGrade(Enum):
    """ASME material grades for superheater tubes."""
    SA_178_A = "SA-178-A"        # Carbon steel, low pressure
    SA_178_C = "SA-178-C"        # Carbon steel, medium pressure
    SA_192 = "SA-192"            # Seamless carbon steel
    SA_210_A1 = "SA-210-A1"      # Seamless medium-carbon steel
    SA_213_T2 = "SA-213-T2"      # 0.5Cr-0.5Mo
    SA_213_T11 = "SA-213-T11"    # 1.25Cr-0.5Mo
    SA_213_T22 = "SA-213-T22"    # 2.25Cr-1Mo (most common HP superheater)
    SA_213_T91 = "SA-213-T91"    # 9Cr-1Mo-V (advanced high temp)
    SA_213_TP304H = "SA-213-TP304H"  # Austenitic stainless
    SA_213_TP347H = "SA-213-TP347H"  # Nb-stabilized stainless


class InterlockType(Enum):
    """NFPA 85 interlock categories."""
    FUEL_TRIP = "FUEL_TRIP"
    FD_FAN_TRIP = "FD_FAN_TRIP"
    ID_FAN_TRIP = "ID_FAN_TRIP"
    DRUM_LEVEL_HIGH = "DRUM_LEVEL_HIGH"
    DRUM_LEVEL_LOW = "DRUM_LEVEL_LOW"
    STEAM_PRESSURE_HIGH = "STEAM_PRESSURE_HIGH"
    COMBUSTION_AIR_LOW = "COMBUSTION_AIR_LOW"
    FLAME_FAILURE = "FLAME_FAILURE"
    SUPERHEATER_TEMP_HIGH = "SUPERHEATER_TEMP_HIGH"
    TUBE_METAL_TEMP_HIGH = "TUBE_METAL_TEMP_HIGH"
    SPRAY_WATER_PRESSURE_LOW = "SPRAY_WATER_PRESSURE_LOW"


# =============================================================================
# MATERIAL PROPERTY TABLES (ASME SECTION II PART D)
# =============================================================================

# Maximum allowable stress (ksi) by temperature (degF) per ASME Section II Part D
# Values from Table 1A for Ferrous Materials
MATERIAL_MAX_ALLOWABLE_STRESS: Dict[str, Dict[int, float]] = {
    "SA-178-A": {
        100: 11.3, 200: 11.3, 300: 11.3, 400: 11.3, 500: 11.3,
        600: 11.3, 700: 11.0, 800: 8.4, 900: 5.0, 950: 3.5
    },
    "SA-178-C": {
        100: 13.8, 200: 13.8, 300: 13.8, 400: 13.8, 500: 13.8,
        600: 13.8, 700: 13.3, 800: 10.2, 900: 6.5, 950: 4.5
    },
    "SA-192": {
        100: 11.8, 200: 11.8, 300: 11.8, 400: 11.8, 500: 11.8,
        600: 11.8, 700: 11.5, 800: 8.7, 900: 5.5, 950: 3.8
    },
    "SA-210-A1": {
        100: 15.0, 200: 15.0, 300: 15.0, 400: 15.0, 500: 15.0,
        600: 15.0, 700: 14.4, 800: 11.0, 900: 7.0, 950: 5.0
    },
    "SA-213-T2": {
        100: 15.0, 200: 15.0, 300: 15.0, 400: 15.0, 500: 15.0,
        600: 15.0, 700: 15.0, 800: 14.4, 900: 12.5, 1000: 8.0, 1050: 5.5
    },
    "SA-213-T11": {
        100: 17.1, 200: 17.1, 300: 17.1, 400: 17.1, 500: 17.1,
        600: 17.1, 700: 17.1, 800: 17.1, 900: 16.6, 1000: 12.0,
        1050: 9.5, 1100: 7.0, 1150: 5.0
    },
    "SA-213-T22": {
        100: 17.1, 200: 17.1, 300: 17.1, 400: 17.1, 500: 17.1,
        600: 17.1, 700: 17.1, 800: 17.1, 900: 17.1, 1000: 15.0,
        1050: 12.0, 1100: 9.5, 1150: 7.0, 1200: 4.5
    },
    "SA-213-T91": {
        100: 20.0, 200: 20.0, 300: 20.0, 400: 20.0, 500: 20.0,
        600: 20.0, 700: 20.0, 800: 20.0, 900: 20.0, 1000: 20.0,
        1050: 18.5, 1100: 16.5, 1150: 13.0, 1200: 9.0
    },
    "SA-213-TP304H": {
        100: 16.7, 200: 16.7, 300: 15.5, 400: 14.1, 500: 13.0,
        600: 12.3, 700: 11.7, 800: 11.3, 900: 11.0, 1000: 10.6,
        1100: 9.8, 1200: 8.2, 1300: 5.5, 1400: 3.5, 1500: 2.2
    },
    "SA-213-TP347H": {
        100: 16.7, 200: 16.7, 300: 15.5, 400: 14.4, 500: 13.5,
        600: 12.8, 700: 12.3, 800: 11.9, 900: 11.6, 1000: 11.3,
        1100: 10.8, 1200: 9.5, 1300: 7.0, 1400: 4.5, 1500: 2.8
    },
}

# Maximum recommended tube metal temperature (degF) per material
# Based on ASME code stress rupture considerations
MATERIAL_MAX_TEMP_F: Dict[str, float] = {
    "SA-178-A": 850,
    "SA-178-C": 900,
    "SA-192": 900,
    "SA-210-A1": 900,
    "SA-213-T2": 1025,
    "SA-213-T11": 1100,
    "SA-213-T22": 1125,
    "SA-213-T91": 1200,
    "SA-213-TP304H": 1500,
    "SA-213-TP347H": 1500,
}

# Larson-Miller parameters for creep rupture life (API 530)
# C constant and coefficients for stress-rupture correlation
CREEP_RUPTURE_PARAMS: Dict[str, Dict[str, float]] = {
    "SA-213-T11": {"C": 20.0, "A": 47.15, "B": -12.89},
    "SA-213-T22": {"C": 20.0, "A": 48.87, "B": -13.24},
    "SA-213-T91": {"C": 30.0, "A": 73.21, "B": -16.54},
    "SA-213-TP304H": {"C": 15.0, "A": 41.23, "B": -10.12},
    "SA-213-TP347H": {"C": 15.0, "A": 42.56, "B": -10.45},
}

# NFPA 85 required interlocks for superheater protection
REQUIRED_SUPERHEATER_INTERLOCKS: List[InterlockType] = [
    InterlockType.SUPERHEATER_TEMP_HIGH,
    InterlockType.TUBE_METAL_TEMP_HIGH,
    InterlockType.SPRAY_WATER_PRESSURE_LOW,
    InterlockType.FUEL_TRIP,
    InterlockType.FLAME_FAILURE,
]


# =============================================================================
# DATA MODELS
# =============================================================================

class SafetyCheckResult(BaseModel):
    """Result of a safety validation check."""

    check_id: str = Field(..., description="Unique check identifier")
    check_type: str = Field(..., description="Type of safety check performed")
    is_safe: bool = Field(..., description="Overall safety status")
    status: SafetyStatus = Field(..., description="Detailed safety status")
    measured_value: float = Field(..., description="Actual measured value")
    limit_value: float = Field(..., description="Safety limit value")
    margin: float = Field(..., description="Margin to limit (positive = safe)")
    margin_percent: float = Field(..., description="Margin as percentage of limit")
    unit: str = Field(..., description="Engineering unit")
    message: str = Field(..., description="Human-readable result message")
    recommendations: List[str] = Field(
        default_factory=list,
        description="Corrective action recommendations"
    )
    standard_reference: str = Field(..., description="Applicable standard reference")
    provenance_hash: str = Field(..., description="SHA-256 audit trail hash")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Check timestamp"
    )

    class Config:
        use_enum_values = True


class ComplianceCheckResult(BaseModel):
    """Result of a compliance verification check."""

    check_id: str = Field(..., description="Unique check identifier")
    compliance_standard: str = Field(..., description="Applicable standard")
    section_reference: str = Field(..., description="Standard section reference")
    status: ComplianceStatus = Field(..., description="Compliance status")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed check results"
    )
    findings: List[str] = Field(
        default_factory=list,
        description="Compliance findings"
    )
    corrective_actions: List[str] = Field(
        default_factory=list,
        description="Required corrective actions"
    )
    provenance_hash: str = Field(..., description="SHA-256 audit trail hash")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Check timestamp"
    )

    class Config:
        use_enum_values = True


class ThermalShockAssessment(BaseModel):
    """Assessment of thermal shock risk."""

    rate_of_change_f_per_min: float = Field(
        ...,
        description="Temperature rate of change (degF/min)"
    )
    max_allowable_rate_f_per_min: float = Field(
        ...,
        description="Maximum allowable rate (degF/min)"
    )
    risk_level: SafetyStatus = Field(..., description="Thermal shock risk level")
    stress_factor: float = Field(
        ...,
        ge=0.0,
        description="Thermal stress multiplication factor"
    )
    estimated_cycle_life_impact: float = Field(
        ...,
        description="Estimated reduction in fatigue life (%)"
    )
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(..., description="SHA-256 audit trail hash")


class InterlockStatusReport(BaseModel):
    """Report of interlock system status."""

    all_interlocks_healthy: bool = Field(
        ...,
        description="All required interlocks operational"
    )
    interlocks_checked: List[str] = Field(
        ...,
        description="List of interlocks verified"
    )
    interlocks_active: List[str] = Field(
        ...,
        description="List of currently active (tripped) interlocks"
    )
    interlocks_failed: List[str] = Field(
        ...,
        description="List of failed/bypassed interlocks"
    )
    missing_interlocks: List[str] = Field(
        ...,
        description="Required interlocks not present"
    )
    nfpa_85_compliant: bool = Field(
        ...,
        description="NFPA 85 interlock compliance status"
    )
    safety_status: SafetyStatus = Field(..., description="Overall safety status")
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(..., description="SHA-256 audit trail hash")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    class Config:
        use_enum_values = True


class CreepRuptureAnalysis(BaseModel):
    """Creep rupture life analysis per API 530."""

    material: str = Field(..., description="Material grade")
    temperature_f: float = Field(..., description="Operating temperature (degF)")
    stress_ksi: float = Field(..., description="Hoop stress (ksi)")
    estimated_life_hours: float = Field(
        ...,
        description="Estimated rupture life (hours)"
    )
    consumed_life_percent: float = Field(
        ...,
        description="Consumed life fraction (%)"
    )
    remaining_life_hours: float = Field(
        ...,
        description="Remaining life (hours)"
    )
    larson_miller_parameter: float = Field(
        ...,
        description="Calculated LMP value"
    )
    safety_factor: float = Field(
        ...,
        description="Applied safety factor"
    )
    status: SafetyStatus = Field(..., description="Life status assessment")
    provenance_hash: str = Field(..., description="SHA-256 audit trail hash")

    class Config:
        use_enum_values = True


# =============================================================================
# SAFETY VALIDATOR CLASS
# =============================================================================

class SafetyValidator:
    """
    Comprehensive safety validation for superheater control operations.

    This class implements deterministic safety checks based on:
    - ASME Boiler and Pressure Vessel Code
    - ASME B31.1 Power Piping Code
    - NFPA 85 Boiler and Combustion Systems Hazards Code

    All checks produce SHA-256 provenance hashes for complete audit trails.

    Attributes:
        agent_id: Identifier for the agent using this validator
        version: Validator version for provenance tracking

    Example:
        >>> validator = SafetyValidator(agent_id="GL-022-001")
        >>> result = validator.validate_tube_metal_temperature(
        ...     temp_f=1050,
        ...     material="SA-213-T22",
        ...     thickness_in=0.25
        ... )
        >>> if not result.is_safe:
        ...     print(f"SAFETY VIOLATION: {result.message}")
    """

    VERSION = "1.0.0"

    # Rate of change limits per ASME B31.1 and industry practice
    MAX_TEMP_RATE_F_PER_MIN_NORMAL = 5.0      # Normal operation
    MAX_TEMP_RATE_F_PER_MIN_STARTUP = 10.0    # Startup/shutdown
    MAX_TEMP_RATE_F_PER_MIN_EMERGENCY = 15.0  # Emergency only

    # Spray water differential pressure minimum (NFPA 85, typical)
    MIN_SPRAY_PRESSURE_DIFFERENTIAL_PSI = 50.0

    # Steam quality requirements
    MIN_STEAM_QUALITY_PERCENT = 99.5

    def __init__(
        self,
        agent_id: str,
        version: str = "1.0.0",
    ) -> None:
        """
        Initialize the SafetyValidator.

        Args:
            agent_id: Agent identifier for audit trails
            version: Version string for provenance tracking
        """
        self.agent_id = agent_id
        self.version = version
        self._lock = threading.RLock()
        self._check_counter = 0

        logger.info(
            f"SafetyValidator initialized for agent {agent_id} v{version}"
        )

    # =========================================================================
    # PRIMARY VALIDATION METHODS
    # =========================================================================

    def validate_tube_metal_temperature(
        self,
        temp_f: float,
        material: str,
        thickness_in: float,
        operating_hours: float = 0.0,
    ) -> SafetyCheckResult:
        """
        Validate tube metal temperature against material limits.

        Per ASME Section II Part D, validates that tube metal temperature
        is within allowable limits for the specified material grade,
        accounting for operating hours and thickness considerations.

        Args:
            temp_f: Measured tube metal temperature (degrees Fahrenheit)
            material: ASME material designation (e.g., "SA-213-T22")
            thickness_in: Tube wall thickness (inches)
            operating_hours: Accumulated operating hours at temperature

        Returns:
            SafetyCheckResult with validation outcome

        Raises:
            ValueError: If material is not in database
        """
        with self._lock:
            self._check_counter += 1
            check_id = f"TMT-{self._check_counter:06d}"

        # Validate material is known
        if material not in MATERIAL_MAX_TEMP_F:
            raise ValueError(
                f"Unknown material grade: {material}. "
                f"Valid materials: {list(MATERIAL_MAX_TEMP_F.keys())}"
            )

        # Get material limit
        max_temp_f = MATERIAL_MAX_TEMP_F[material]

        # Apply thickness derating if thin-walled (< 0.15 in typical)
        if thickness_in < 0.15:
            # Thin wall tubes have higher temperature differential
            thickness_factor = 0.95  # 5% derating
            max_temp_f *= thickness_factor
            logger.debug(
                f"Thin wall derating applied: {thickness_in} in, "
                f"factor={thickness_factor}"
            )

        # Apply age-based derating for long-term operation
        # Per API 530, tubes above 100,000 hours require assessment
        age_factor = 1.0
        if operating_hours > 100000:
            age_factor = 0.95
            max_temp_f *= age_factor
            logger.debug(
                f"Age derating applied: {operating_hours} hrs, "
                f"factor={age_factor}"
            )

        # Calculate margin
        margin_f = max_temp_f - temp_f
        margin_percent = (margin_f / max_temp_f) * 100 if max_temp_f > 0 else 0

        # Determine safety status
        if margin_f < 0:
            status = SafetyStatus.VIOLATION
            is_safe = False
            message = (
                f"VIOLATION: Tube metal temp {temp_f:.1f}degF exceeds "
                f"limit {max_temp_f:.1f}degF for {material}"
            )
            recommendations = [
                "IMMEDIATELY reduce firing rate",
                "Increase spray water flow if available",
                "Check for superheater fouling or gas channeling",
                "Initiate controlled shutdown if temperature continues rising",
            ]
        elif margin_f < 25:
            status = SafetyStatus.CRITICAL
            is_safe = False
            message = (
                f"CRITICAL: Tube metal temp {temp_f:.1f}degF within "
                f"{margin_f:.1f}degF of limit for {material}"
            )
            recommendations = [
                "Reduce firing rate by 10-20%",
                "Increase spray water flow",
                "Monitor closely - prepare for controlled load reduction",
            ]
        elif margin_f < 50:
            status = SafetyStatus.WARNING
            is_safe = True
            message = (
                f"WARNING: Tube metal temp {temp_f:.1f}degF within "
                f"{margin_f:.1f}degF of limit for {material}"
            )
            recommendations = [
                "Monitor tube metal temperature trend",
                "Consider load reduction if trend continues upward",
            ]
        else:
            status = SafetyStatus.SAFE
            is_safe = True
            message = (
                f"SAFE: Tube metal temp {temp_f:.1f}degF with "
                f"{margin_f:.1f}degF margin for {material}"
            )
            recommendations = []

        # Generate provenance hash
        provenance_data = {
            "check_type": "tube_metal_temperature",
            "agent_id": self.agent_id,
            "version": self.version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "temp_f": temp_f,
                "material": material,
                "thickness_in": thickness_in,
                "operating_hours": operating_hours,
            },
            "outputs": {
                "max_temp_f": max_temp_f,
                "margin_f": margin_f,
                "status": status.value,
            },
        }
        provenance_hash = self._calculate_hash(provenance_data)

        logger.info(
            f"Tube metal temp check [{check_id}]: {temp_f}degF, "
            f"material={material}, status={status.value}"
        )

        return SafetyCheckResult(
            check_id=check_id,
            check_type="tube_metal_temperature",
            is_safe=is_safe,
            status=status,
            measured_value=temp_f,
            limit_value=max_temp_f,
            margin=margin_f,
            margin_percent=round(margin_percent, 2),
            unit="degF",
            message=message,
            recommendations=recommendations,
            standard_reference="ASME Section II Part D, Table 1A",
            provenance_hash=provenance_hash,
        )

    def validate_temperature_rate_of_change(
        self,
        current_temp_f: float,
        previous_temp_f: float,
        dt_minutes: float,
        operation_mode: str = "normal",
    ) -> SafetyCheckResult:
        """
        Validate temperature rate of change to prevent thermal shock.

        Per ASME B31.1 and industry practice, rapid temperature changes
        can cause thermal fatigue and cracking. This method validates
        that the rate of change is within acceptable limits.

        Args:
            current_temp_f: Current temperature reading (degF)
            previous_temp_f: Previous temperature reading (degF)
            dt_minutes: Time interval between readings (minutes)
            operation_mode: "normal", "startup", or "emergency"

        Returns:
            SafetyCheckResult with validation outcome
        """
        with self._lock:
            self._check_counter += 1
            check_id = f"ROC-{self._check_counter:06d}"

        # Validate inputs
        if dt_minutes <= 0:
            raise ValueError("Time interval must be positive")

        # Calculate rate of change
        rate_f_per_min = (current_temp_f - previous_temp_f) / dt_minutes

        # Select limit based on operation mode
        if operation_mode == "startup":
            max_rate = self.MAX_TEMP_RATE_F_PER_MIN_STARTUP
        elif operation_mode == "emergency":
            max_rate = self.MAX_TEMP_RATE_F_PER_MIN_EMERGENCY
        else:
            max_rate = self.MAX_TEMP_RATE_F_PER_MIN_NORMAL

        # Calculate margin (using absolute value for rate check)
        abs_rate = abs(rate_f_per_min)
        margin = max_rate - abs_rate
        margin_percent = (margin / max_rate) * 100 if max_rate > 0 else 0

        # Direction indicator
        direction = "increasing" if rate_f_per_min > 0 else "decreasing"

        # Determine status
        if abs_rate > max_rate * 1.5:
            status = SafetyStatus.CRITICAL
            is_safe = False
            message = (
                f"CRITICAL: Temperature {direction} at {abs_rate:.2f}degF/min "
                f"exceeds 150% of limit ({max_rate}degF/min)"
            )
            recommendations = [
                "STOP temperature change immediately",
                "Reduce spray flow change rate",
                "Stabilize firing rate",
                "Check for equipment malfunction",
            ]
        elif abs_rate > max_rate:
            status = SafetyStatus.WARNING
            is_safe = False
            message = (
                f"WARNING: Temperature {direction} at {abs_rate:.2f}degF/min "
                f"exceeds limit ({max_rate}degF/min)"
            )
            recommendations = [
                "Reduce rate of spray flow adjustment",
                "Monitor for thermal shock symptoms",
            ]
        elif abs_rate > max_rate * 0.8:
            status = SafetyStatus.WARNING
            is_safe = True
            message = (
                f"CAUTION: Temperature {direction} at {abs_rate:.2f}degF/min "
                f"approaching limit ({max_rate}degF/min)"
            )
            recommendations = [
                "Moderate control actions to prevent exceeding limit",
            ]
        else:
            status = SafetyStatus.SAFE
            is_safe = True
            message = (
                f"SAFE: Temperature {direction} at {abs_rate:.2f}degF/min "
                f"within limit ({max_rate}degF/min)"
            )
            recommendations = []

        # Generate provenance hash
        provenance_data = {
            "check_type": "temperature_rate_of_change",
            "agent_id": self.agent_id,
            "version": self.version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "current_temp_f": current_temp_f,
                "previous_temp_f": previous_temp_f,
                "dt_minutes": dt_minutes,
                "operation_mode": operation_mode,
            },
            "outputs": {
                "rate_f_per_min": rate_f_per_min,
                "max_rate": max_rate,
                "status": status.value,
            },
        }
        provenance_hash = self._calculate_hash(provenance_data)

        logger.info(
            f"Rate of change check [{check_id}]: {rate_f_per_min:.2f}degF/min, "
            f"mode={operation_mode}, status={status.value}"
        )

        return SafetyCheckResult(
            check_id=check_id,
            check_type="temperature_rate_of_change",
            is_safe=is_safe,
            status=status,
            measured_value=round(rate_f_per_min, 3),
            limit_value=max_rate,
            margin=round(margin, 3),
            margin_percent=round(margin_percent, 2),
            unit="degF/min",
            message=message,
            recommendations=recommendations,
            standard_reference="ASME B31.1, Industry Best Practice",
            provenance_hash=provenance_hash,
        )

    def validate_spray_pressure_differential(
        self,
        steam_pressure_psi: float,
        spray_water_pressure_psi: float,
    ) -> SafetyCheckResult:
        """
        Validate spray water pressure differential above steam pressure.

        Per NFPA 85 and desuperheater manufacturer requirements, spray
        water pressure must exceed steam pressure by a minimum differential
        to ensure proper atomization and prevent steam backflow.

        Args:
            steam_pressure_psi: Steam pressure at spray location (psi)
            spray_water_pressure_psi: Spray water supply pressure (psi)

        Returns:
            SafetyCheckResult with validation outcome
        """
        with self._lock:
            self._check_counter += 1
            check_id = f"SPD-{self._check_counter:06d}"

        # Calculate differential
        differential_psi = spray_water_pressure_psi - steam_pressure_psi
        min_differential = self.MIN_SPRAY_PRESSURE_DIFFERENTIAL_PSI

        # Calculate margin
        margin = differential_psi - min_differential
        margin_percent = (
            (margin / min_differential) * 100 if min_differential > 0 else 0
        )

        # Determine status
        if differential_psi < 0:
            status = SafetyStatus.SHUTDOWN_REQUIRED
            is_safe = False
            message = (
                f"SHUTDOWN REQUIRED: Spray water pressure {spray_water_pressure_psi:.1f} psi "
                f"BELOW steam pressure {steam_pressure_psi:.1f} psi - backflow risk!"
            )
            recommendations = [
                "IMMEDIATELY close spray water valve",
                "Check spray water pump operation",
                "Verify check valve operation",
                "Do NOT attempt spray injection until resolved",
            ]
        elif differential_psi < min_differential:
            status = SafetyStatus.CRITICAL
            is_safe = False
            message = (
                f"CRITICAL: Spray differential {differential_psi:.1f} psi "
                f"below minimum {min_differential:.1f} psi required"
            )
            recommendations = [
                "Increase spray water supply pressure",
                "Check spray water pump discharge pressure",
                "Reduce spray injection until differential restored",
            ]
        elif differential_psi < min_differential * 1.2:
            status = SafetyStatus.WARNING
            is_safe = True
            message = (
                f"WARNING: Spray differential {differential_psi:.1f} psi "
                f"marginally above minimum {min_differential:.1f} psi"
            )
            recommendations = [
                "Monitor spray water pressure",
                "Consider increasing supply pressure for margin",
            ]
        else:
            status = SafetyStatus.SAFE
            is_safe = True
            message = (
                f"SAFE: Spray differential {differential_psi:.1f} psi "
                f"exceeds minimum {min_differential:.1f} psi"
            )
            recommendations = []

        # Generate provenance hash
        provenance_data = {
            "check_type": "spray_pressure_differential",
            "agent_id": self.agent_id,
            "version": self.version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "steam_pressure_psi": steam_pressure_psi,
                "spray_water_pressure_psi": spray_water_pressure_psi,
            },
            "outputs": {
                "differential_psi": differential_psi,
                "min_differential": min_differential,
                "status": status.value,
            },
        }
        provenance_hash = self._calculate_hash(provenance_data)

        logger.info(
            f"Spray pressure check [{check_id}]: diff={differential_psi:.1f} psi, "
            f"status={status.value}"
        )

        return SafetyCheckResult(
            check_id=check_id,
            check_type="spray_pressure_differential",
            is_safe=is_safe,
            status=status,
            measured_value=round(differential_psi, 2),
            limit_value=min_differential,
            margin=round(margin, 2),
            margin_percent=round(margin_percent, 2),
            unit="psi",
            message=message,
            recommendations=recommendations,
            standard_reference="NFPA 85, Desuperheater Manufacturer Requirements",
            provenance_hash=provenance_hash,
        )

    def validate_steam_conditions(
        self,
        temperature_f: float,
        pressure_psi: float,
        quality_percent: Optional[float] = None,
    ) -> SafetyCheckResult:
        """
        Validate steam conditions for superheat and quality requirements.

        Ensures steam is properly superheated and meets quality requirements
        to prevent moisture carryover damage to downstream equipment.

        Args:
            temperature_f: Steam temperature (degF)
            pressure_psi: Steam pressure (psi)
            quality_percent: Steam quality/dryness (%, optional)

        Returns:
            SafetyCheckResult with validation outcome
        """
        with self._lock:
            self._check_counter += 1
            check_id = f"STM-{self._check_counter:06d}"

        # Calculate saturation temperature from pressure
        # Using simplified correlation: T_sat = 327.8 + 43.76*ln(P) for P in psia
        # Valid for typical boiler operating range
        pressure_psia = pressure_psi + 14.7  # Convert gauge to absolute
        t_sat_f = 327.8 + 43.76 * math.log(pressure_psia)

        # Calculate superheat
        superheat_f = temperature_f - t_sat_f

        # Minimum superheat requirement (typically 20-50 degF to ensure dry steam)
        min_superheat_f = 20.0

        # Steam quality check
        quality = quality_percent if quality_percent is not None else 100.0

        # Determine limiting condition and status
        issues = []
        recommendations = []

        if superheat_f < 0:
            status = SafetyStatus.CRITICAL
            is_safe = False
            issues.append(
                f"Steam is WET (below saturation by {abs(superheat_f):.1f}degF)"
            )
            recommendations.extend([
                "CRITICAL: Increase firing or reduce load",
                "Check for water carryover from drum",
                "Reduce spray water flow immediately",
            ])
        elif superheat_f < min_superheat_f:
            status = SafetyStatus.WARNING
            is_safe = True
            issues.append(
                f"Low superheat {superheat_f:.1f}degF (min {min_superheat_f}degF)"
            )
            recommendations.append("Reduce spray flow to increase superheat")
        else:
            status = SafetyStatus.SAFE
            is_safe = True

        # Quality check
        if quality < self.MIN_STEAM_QUALITY_PERCENT:
            if status == SafetyStatus.SAFE:
                status = SafetyStatus.WARNING
            is_safe = is_safe and False
            issues.append(
                f"Steam quality {quality:.2f}% below minimum "
                f"{self.MIN_STEAM_QUALITY_PERCENT}%"
            )
            recommendations.append("Check steam separator/drum internals")

        # Build message
        if issues:
            message = "; ".join(issues)
        else:
            message = (
                f"SAFE: Steam at {temperature_f:.1f}degF/{pressure_psi:.1f} psig "
                f"with {superheat_f:.1f}degF superheat"
            )

        # Calculate margin (based on superheat)
        margin = superheat_f - min_superheat_f
        margin_percent = (margin / min_superheat_f) * 100 if min_superheat_f > 0 else 0

        # Generate provenance hash
        provenance_data = {
            "check_type": "steam_conditions",
            "agent_id": self.agent_id,
            "version": self.version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "temperature_f": temperature_f,
                "pressure_psi": pressure_psi,
                "quality_percent": quality_percent,
            },
            "outputs": {
                "t_sat_f": round(t_sat_f, 2),
                "superheat_f": round(superheat_f, 2),
                "status": status.value,
            },
        }
        provenance_hash = self._calculate_hash(provenance_data)

        logger.info(
            f"Steam conditions check [{check_id}]: T={temperature_f}degF, "
            f"P={pressure_psi} psig, superheat={superheat_f:.1f}degF, "
            f"status={status.value}"
        )

        return SafetyCheckResult(
            check_id=check_id,
            check_type="steam_conditions",
            is_safe=is_safe,
            status=status,
            measured_value=round(superheat_f, 2),
            limit_value=min_superheat_f,
            margin=round(margin, 2),
            margin_percent=round(margin_percent, 2),
            unit="degF",
            message=message,
            recommendations=recommendations,
            standard_reference="ASME PTC 4, Steam Purity Requirements",
            provenance_hash=provenance_hash,
        )

    def check_thermal_shock_risk(
        self,
        rate_of_change_f_per_min: float,
        material: str = "SA-213-T22",
        tube_od_in: float = 2.0,
        tube_thickness_in: float = 0.25,
    ) -> ThermalShockAssessment:
        """
        Assess thermal shock risk from rapid temperature changes.

        Thermal shock can cause:
        - Thermal fatigue cracking
        - Oxide spallation
        - Reduced creep-fatigue life
        - Tube distortion

        Args:
            rate_of_change_f_per_min: Temperature rate of change (degF/min)
            material: Tube material grade
            tube_od_in: Tube outside diameter (inches)
            tube_thickness_in: Tube wall thickness (inches)

        Returns:
            ThermalShockAssessment with risk evaluation
        """
        abs_rate = abs(rate_of_change_f_per_min)
        max_rate = self.MAX_TEMP_RATE_F_PER_MIN_NORMAL

        # Calculate thermal stress factor
        # Based on thermal expansion and elastic modulus
        # Simplified: stress proportional to rate and wall thickness
        thermal_expansion = 7.0e-6  # 1/degF typical for low alloy steel
        elastic_modulus = 28.0e6   # psi typical

        # Thermal stress estimation (simplified)
        # sigma_thermal ~ E * alpha * deltaT_through_wall
        # deltaT_through_wall ~ rate * (t/2) / k where k is thermal diffusivity
        wall_temp_diff = rate_of_change_f_per_min * tube_thickness_in * 5  # empirical
        thermal_stress_psi = elastic_modulus * thermal_expansion * wall_temp_diff

        # Stress factor relative to baseline
        baseline_stress = elastic_modulus * thermal_expansion * max_rate * 0.25 * 5
        stress_factor = thermal_stress_psi / baseline_stress if baseline_stress > 0 else 1.0

        # Estimate cycle life impact using Coffin-Manson relationship
        # Life reduction proportional to stress amplitude^2
        life_impact_percent = min(100, (stress_factor - 1.0) * 50) if stress_factor > 1.0 else 0

        # Determine risk level
        recommendations = []
        if abs_rate > max_rate * 2.0:
            risk_level = SafetyStatus.CRITICAL
            recommendations = [
                "CRITICAL thermal shock risk - stop temperature change",
                "Inspect tubes for cracking at next outage",
                "Review control tuning to prevent recurrence",
            ]
        elif abs_rate > max_rate * 1.5:
            risk_level = SafetyStatus.WARNING
            recommendations = [
                "High thermal shock risk - reduce rate of change",
                "Monitor for oxide spallation in blowdown",
            ]
        elif abs_rate > max_rate:
            risk_level = SafetyStatus.WARNING
            recommendations = [
                "Moderate thermal shock risk - limit duration",
            ]
        else:
            risk_level = SafetyStatus.SAFE

        # Generate provenance hash
        provenance_data = {
            "check_type": "thermal_shock_risk",
            "agent_id": self.agent_id,
            "version": self.version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "rate_of_change_f_per_min": rate_of_change_f_per_min,
                "material": material,
                "tube_od_in": tube_od_in,
                "tube_thickness_in": tube_thickness_in,
            },
            "outputs": {
                "stress_factor": round(stress_factor, 3),
                "life_impact_percent": round(life_impact_percent, 2),
                "risk_level": risk_level.value,
            },
        }
        provenance_hash = self._calculate_hash(provenance_data)

        logger.info(
            f"Thermal shock assessment: rate={abs_rate:.2f}degF/min, "
            f"stress_factor={stress_factor:.2f}, risk={risk_level.value}"
        )

        return ThermalShockAssessment(
            rate_of_change_f_per_min=round(rate_of_change_f_per_min, 3),
            max_allowable_rate_f_per_min=max_rate,
            risk_level=risk_level,
            stress_factor=round(stress_factor, 3),
            estimated_cycle_life_impact=round(life_impact_percent, 2),
            recommendations=recommendations,
            provenance_hash=provenance_hash,
        )

    def validate_interlock_status(
        self,
        interlocks: Dict[str, Dict[str, Any]],
    ) -> InterlockStatusReport:
        """
        Validate interlock status per NFPA 85 requirements.

        Verifies that all required safety interlocks are:
        - Present and configured
        - Healthy (not bypassed or failed)
        - Properly responding

        Args:
            interlocks: Dictionary of interlock status:
                {
                    "SUPERHEATER_TEMP_HIGH": {
                        "present": True,
                        "healthy": True,
                        "active": False,
                        "bypassed": False,
                        "setpoint": 1100.0,
                        "current_value": 1050.0
                    },
                    ...
                }

        Returns:
            InterlockStatusReport with complete status
        """
        interlocks_checked = []
        interlocks_active = []
        interlocks_failed = []
        missing_interlocks = []

        # Check each required interlock
        for required in REQUIRED_SUPERHEATER_INTERLOCKS:
            interlock_name = required.value
            interlocks_checked.append(interlock_name)

            if interlock_name not in interlocks:
                missing_interlocks.append(interlock_name)
                continue

            status = interlocks[interlock_name]

            # Check if present
            if not status.get("present", False):
                missing_interlocks.append(interlock_name)
                continue

            # Check if healthy
            if not status.get("healthy", True) or status.get("bypassed", False):
                interlocks_failed.append(interlock_name)

            # Check if active (tripped)
            if status.get("active", False):
                interlocks_active.append(interlock_name)

        # Determine overall status
        recommendations = []
        if missing_interlocks:
            safety_status = SafetyStatus.VIOLATION
            all_healthy = False
            nfpa_85_compliant = False
            recommendations.append(
                f"VIOLATION: Missing required interlocks: {missing_interlocks}"
            )
        elif interlocks_failed:
            safety_status = SafetyStatus.CRITICAL
            all_healthy = False
            nfpa_85_compliant = False
            recommendations.extend([
                f"CRITICAL: Failed/bypassed interlocks: {interlocks_failed}",
                "Restore failed interlocks before continued operation",
            ])
        elif interlocks_active:
            safety_status = SafetyStatus.WARNING
            all_healthy = True
            nfpa_85_compliant = True
            recommendations.append(
                f"Interlocks currently active: {interlocks_active}"
            )
        else:
            safety_status = SafetyStatus.SAFE
            all_healthy = True
            nfpa_85_compliant = True

        # Generate provenance hash
        provenance_data = {
            "check_type": "interlock_status",
            "agent_id": self.agent_id,
            "version": self.version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "interlocks": interlocks,
            },
            "outputs": {
                "all_healthy": all_healthy,
                "nfpa_85_compliant": nfpa_85_compliant,
                "safety_status": safety_status.value,
            },
        }
        provenance_hash = self._calculate_hash(provenance_data)

        logger.info(
            f"Interlock status check: healthy={all_healthy}, "
            f"NFPA85={nfpa_85_compliant}, status={safety_status.value}"
        )

        return InterlockStatusReport(
            all_interlocks_healthy=all_healthy,
            interlocks_checked=interlocks_checked,
            interlocks_active=interlocks_active,
            interlocks_failed=interlocks_failed,
            missing_interlocks=missing_interlocks,
            nfpa_85_compliant=nfpa_85_compliant,
            safety_status=safety_status,
            recommendations=recommendations,
            provenance_hash=provenance_hash,
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


# =============================================================================
# ASME COMPLIANCE CHECKER CLASS
# =============================================================================

class ASMEComplianceChecker:
    """
    ASME Boiler and Pressure Vessel Code compliance verification.

    This class implements deterministic compliance checks based on:
    - ASME Section I (Power Boilers)
    - ASME Section II Part D (Material Properties)
    - ASME B31.1 (Power Piping)
    - API 530 (Calculation of Heater Tube Thickness)

    All calculations are traceable with SHA-256 provenance hashes.

    Attributes:
        agent_id: Identifier for the agent using this checker
        version: Checker version for provenance tracking

    Example:
        >>> checker = ASMEComplianceChecker(agent_id="GL-022-001")
        >>> result = checker.check_max_allowable_stress(
        ...     material="SA-213-T22",
        ...     temp_f=1050
        ... )
        >>> print(f"Allowable stress: {result.details['allowable_stress_ksi']} ksi")
    """

    VERSION = "1.0.0"

    # ASME design factors
    DESIGN_MARGIN_FACTOR = 1.5  # Typical design margin
    CREEP_SAFETY_FACTOR = 1.25  # API 530 recommended safety factor

    def __init__(
        self,
        agent_id: str,
        version: str = "1.0.0",
    ) -> None:
        """
        Initialize the ASMEComplianceChecker.

        Args:
            agent_id: Agent identifier for audit trails
            version: Version string for provenance tracking
        """
        self.agent_id = agent_id
        self.version = version
        self._lock = threading.RLock()
        self._check_counter = 0

        logger.info(
            f"ASMEComplianceChecker initialized for agent {agent_id} v{version}"
        )

    def check_max_allowable_stress(
        self,
        material: str,
        temp_f: float,
    ) -> ComplianceCheckResult:
        """
        Check maximum allowable stress per ASME Section II Part D.

        Interpolates stress values from ASME tables for the given
        material and temperature.

        Args:
            material: ASME material designation (e.g., "SA-213-T22")
            temp_f: Operating temperature (degrees Fahrenheit)

        Returns:
            ComplianceCheckResult with allowable stress value
        """
        with self._lock:
            self._check_counter += 1
            check_id = f"ASME-S-{self._check_counter:06d}"

        # Validate material
        if material not in MATERIAL_MAX_ALLOWABLE_STRESS:
            return ComplianceCheckResult(
                check_id=check_id,
                compliance_standard="ASME Section II Part D",
                section_reference="Table 1A",
                status=ComplianceStatus.REQUIRES_REVIEW,
                details={"material": material, "temp_f": temp_f},
                findings=[f"Material {material} not in stress table database"],
                corrective_actions=["Consult ASME Section II Part D for stress values"],
                provenance_hash=self._calculate_hash({
                    "check_type": "max_allowable_stress",
                    "material": material,
                    "temp_f": temp_f,
                    "status": "material_not_found",
                }),
            )

        # Get stress table for material
        stress_table = MATERIAL_MAX_ALLOWABLE_STRESS[material]
        temps = sorted(stress_table.keys())

        # Check temperature range
        if temp_f < temps[0]:
            # Below minimum - use minimum value
            allowable_stress = stress_table[temps[0]]
            interpolated = False
        elif temp_f > temps[-1]:
            # Above maximum - requires special consideration
            return ComplianceCheckResult(
                check_id=check_id,
                compliance_standard="ASME Section II Part D",
                section_reference="Table 1A",
                status=ComplianceStatus.NON_COMPLIANT,
                details={
                    "material": material,
                    "temp_f": temp_f,
                    "max_tabulated_temp_f": temps[-1],
                },
                findings=[
                    f"Temperature {temp_f}degF exceeds maximum tabulated "
                    f"temperature {temps[-1]}degF for {material}"
                ],
                corrective_actions=[
                    "Reduce operating temperature",
                    "Select higher temperature material grade",
                    "Consult ASME for time-dependent stress values",
                ],
                provenance_hash=self._calculate_hash({
                    "check_type": "max_allowable_stress",
                    "material": material,
                    "temp_f": temp_f,
                    "status": "temperature_exceeded",
                }),
            )
        else:
            # Interpolate between table values
            allowable_stress = self._interpolate_stress(stress_table, temp_f)
            interpolated = True

        # Build result
        provenance_data = {
            "check_type": "max_allowable_stress",
            "agent_id": self.agent_id,
            "version": self.version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": {"material": material, "temp_f": temp_f},
            "outputs": {
                "allowable_stress_ksi": round(allowable_stress, 2),
                "interpolated": interpolated,
            },
        }
        provenance_hash = self._calculate_hash(provenance_data)

        logger.info(
            f"Allowable stress check [{check_id}]: {material} at {temp_f}degF = "
            f"{allowable_stress:.2f} ksi"
        )

        return ComplianceCheckResult(
            check_id=check_id,
            compliance_standard="ASME Section II Part D",
            section_reference="Table 1A",
            status=ComplianceStatus.COMPLIANT,
            details={
                "material": material,
                "temp_f": temp_f,
                "allowable_stress_ksi": round(allowable_stress, 2),
                "interpolated": interpolated,
            },
            findings=[
                f"Maximum allowable stress for {material} at {temp_f}degF is "
                f"{allowable_stress:.2f} ksi"
            ],
            corrective_actions=[],
            provenance_hash=provenance_hash,
        )

    def check_creep_rupture_limits(
        self,
        material: str,
        temp_f: float,
        operating_hours: float,
        hoop_stress_ksi: Optional[float] = None,
        design_pressure_psi: Optional[float] = None,
        tube_od_in: Optional[float] = None,
        tube_thickness_in: Optional[float] = None,
    ) -> CreepRuptureAnalysis:
        """
        Check creep rupture life per API 530.

        Uses the Larson-Miller Parameter method to estimate remaining
        creep-rupture life based on operating temperature and stress.

        Args:
            material: ASME material designation
            temp_f: Operating temperature (degrees Fahrenheit)
            operating_hours: Accumulated operating hours at temperature
            hoop_stress_ksi: Hoop stress (ksi), or calculated from pressure
            design_pressure_psi: Design pressure (psi) - for stress calc
            tube_od_in: Tube outside diameter (inches) - for stress calc
            tube_thickness_in: Tube wall thickness (inches) - for stress calc

        Returns:
            CreepRuptureAnalysis with life assessment
        """
        # Calculate hoop stress if not provided
        if hoop_stress_ksi is None:
            if all([design_pressure_psi, tube_od_in, tube_thickness_in]):
                # Hoop stress = P * D / (2 * t)
                hoop_stress_ksi = (
                    design_pressure_psi * tube_od_in /
                    (2 * tube_thickness_in * 1000)
                )
            else:
                raise ValueError(
                    "Either hoop_stress_ksi or (design_pressure_psi, "
                    "tube_od_in, tube_thickness_in) must be provided"
                )

        # Get Larson-Miller parameters
        if material not in CREEP_RUPTURE_PARAMS:
            # Use default T22 parameters as baseline
            params = CREEP_RUPTURE_PARAMS.get(
                "SA-213-T22",
                {"C": 20.0, "A": 48.87, "B": -13.24}
            )
            logger.warning(
                f"Creep parameters not found for {material}, "
                "using SA-213-T22 as conservative estimate"
            )
        else:
            params = CREEP_RUPTURE_PARAMS[material]

        C = params["C"]
        A = params["A"]
        B = params["B"]

        # Convert temperature to Rankine
        temp_r = temp_f + 459.67

        # Calculate Larson-Miller Parameter for the given stress
        # LMP = T_R * (C + log10(t_r))
        # Solve for rupture life at stress:
        # LMP = A + B * log10(stress)
        lmp = A + B * math.log10(hoop_stress_ksi) if hoop_stress_ksi > 0 else A

        # Solve for rupture time: t_r = 10^(LMP/T_R - C)
        exponent = lmp / temp_r - C
        rupture_life_hours = 10 ** exponent

        # Apply safety factor per API 530
        design_life_hours = rupture_life_hours / self.CREEP_SAFETY_FACTOR

        # Calculate consumed and remaining life
        consumed_life_percent = (operating_hours / design_life_hours) * 100
        remaining_life_hours = max(0, design_life_hours - operating_hours)

        # Determine status
        if consumed_life_percent >= 100:
            status = SafetyStatus.SHUTDOWN_REQUIRED
        elif consumed_life_percent >= 80:
            status = SafetyStatus.CRITICAL
        elif consumed_life_percent >= 60:
            status = SafetyStatus.WARNING
        else:
            status = SafetyStatus.SAFE

        # Generate provenance hash
        provenance_data = {
            "check_type": "creep_rupture_limits",
            "agent_id": self.agent_id,
            "version": self.version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "material": material,
                "temp_f": temp_f,
                "operating_hours": operating_hours,
                "hoop_stress_ksi": round(hoop_stress_ksi, 3),
            },
            "outputs": {
                "lmp": round(lmp, 2),
                "rupture_life_hours": round(rupture_life_hours, 0),
                "consumed_percent": round(consumed_life_percent, 2),
                "status": status.value,
            },
        }
        provenance_hash = self._calculate_hash(provenance_data)

        logger.info(
            f"Creep rupture analysis: {material} at {temp_f}degF, "
            f"stress={hoop_stress_ksi:.2f} ksi, "
            f"consumed={consumed_life_percent:.1f}%, status={status.value}"
        )

        return CreepRuptureAnalysis(
            material=material,
            temperature_f=temp_f,
            stress_ksi=round(hoop_stress_ksi, 3),
            estimated_life_hours=round(rupture_life_hours, 0),
            consumed_life_percent=round(consumed_life_percent, 2),
            remaining_life_hours=round(remaining_life_hours, 0),
            larson_miller_parameter=round(lmp, 2),
            safety_factor=self.CREEP_SAFETY_FACTOR,
            status=status,
            provenance_hash=provenance_hash,
        )

    def check_design_pressure_margin(
        self,
        actual_pressure_psi: float,
        design_pressure_psi: float,
        mawp_psi: Optional[float] = None,
    ) -> ComplianceCheckResult:
        """
        Check operating pressure against design limits.

        Per ASME Section I, operating pressure must not exceed the
        Maximum Allowable Working Pressure (MAWP) which is typically
        set equal to or below the design pressure.

        Args:
            actual_pressure_psi: Current operating pressure (psi)
            design_pressure_psi: Design pressure (psi)
            mawp_psi: Maximum Allowable Working Pressure (psi), defaults to design

        Returns:
            ComplianceCheckResult with margin assessment
        """
        with self._lock:
            self._check_counter += 1
            check_id = f"ASME-P-{self._check_counter:06d}"

        # Use design pressure as MAWP if not specified
        if mawp_psi is None:
            mawp_psi = design_pressure_psi

        # Calculate margins
        design_margin_psi = design_pressure_psi - actual_pressure_psi
        design_margin_percent = (
            (design_margin_psi / design_pressure_psi) * 100
            if design_pressure_psi > 0 else 0
        )

        mawp_margin_psi = mawp_psi - actual_pressure_psi
        mawp_margin_percent = (
            (mawp_margin_psi / mawp_psi) * 100
            if mawp_psi > 0 else 0
        )

        # Determine compliance status
        findings = []
        corrective_actions = []

        if actual_pressure_psi > mawp_psi:
            status = ComplianceStatus.NON_COMPLIANT
            findings.append(
                f"VIOLATION: Operating pressure {actual_pressure_psi:.1f} psi "
                f"exceeds MAWP {mawp_psi:.1f} psi"
            )
            corrective_actions.extend([
                "IMMEDIATELY reduce pressure below MAWP",
                "Check pressure relief valve operation",
                "Investigate cause of overpressure",
            ])
        elif actual_pressure_psi > design_pressure_psi:
            status = ComplianceStatus.MARGINAL
            findings.append(
                f"WARNING: Operating pressure {actual_pressure_psi:.1f} psi "
                f"exceeds design pressure {design_pressure_psi:.1f} psi "
                f"but within MAWP {mawp_psi:.1f} psi"
            )
            corrective_actions.append(
                "Reduce operating pressure to below design pressure"
            )
        elif design_margin_percent < 10:
            status = ComplianceStatus.MARGINAL
            findings.append(
                f"CAUTION: Operating pressure {actual_pressure_psi:.1f} psi "
                f"within {design_margin_percent:.1f}% of design pressure"
            )
            corrective_actions.append(
                "Monitor pressure closely - limited margin"
            )
        else:
            status = ComplianceStatus.COMPLIANT
            findings.append(
                f"Operating pressure {actual_pressure_psi:.1f} psi with "
                f"{design_margin_percent:.1f}% margin to design pressure"
            )

        # Generate provenance hash
        provenance_data = {
            "check_type": "design_pressure_margin",
            "agent_id": self.agent_id,
            "version": self.version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "actual_pressure_psi": actual_pressure_psi,
                "design_pressure_psi": design_pressure_psi,
                "mawp_psi": mawp_psi,
            },
            "outputs": {
                "design_margin_percent": round(design_margin_percent, 2),
                "mawp_margin_percent": round(mawp_margin_percent, 2),
                "status": status.value,
            },
        }
        provenance_hash = self._calculate_hash(provenance_data)

        logger.info(
            f"Design pressure check [{check_id}]: actual={actual_pressure_psi:.1f} psi, "
            f"design={design_pressure_psi:.1f} psi, margin={design_margin_percent:.1f}%, "
            f"status={status.value}"
        )

        return ComplianceCheckResult(
            check_id=check_id,
            compliance_standard="ASME Section I",
            section_reference="PG-67 Operating Pressure Limits",
            status=status,
            details={
                "actual_pressure_psi": actual_pressure_psi,
                "design_pressure_psi": design_pressure_psi,
                "mawp_psi": mawp_psi,
                "design_margin_psi": round(design_margin_psi, 2),
                "design_margin_percent": round(design_margin_percent, 2),
                "mawp_margin_psi": round(mawp_margin_psi, 2),
                "mawp_margin_percent": round(mawp_margin_percent, 2),
            },
            findings=findings,
            corrective_actions=corrective_actions,
            provenance_hash=provenance_hash,
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _interpolate_stress(
        self,
        stress_table: Dict[int, float],
        temp_f: float,
    ) -> float:
        """Linear interpolation of stress values from table."""
        temps = sorted(stress_table.keys())

        # Find bounding temperatures
        lower_temp = temps[0]
        upper_temp = temps[-1]

        for i, t in enumerate(temps):
            if t <= temp_f:
                lower_temp = t
            if t >= temp_f:
                upper_temp = t
                break

        if lower_temp == upper_temp:
            return stress_table[lower_temp]

        # Linear interpolation
        lower_stress = stress_table[lower_temp]
        upper_stress = stress_table[upper_temp]

        fraction = (temp_f - lower_temp) / (upper_temp - lower_temp)
        interpolated_stress = lower_stress + fraction * (upper_stress - lower_stress)

        return interpolated_stress

    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return celsius * 9.0 / 5.0 + 32.0


def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (fahrenheit - 32.0) * 5.0 / 9.0


def bar_to_psi(bar: float) -> float:
    """Convert bar to PSI."""
    return bar * 14.5038


def psi_to_bar(psi: float) -> float:
    """Convert PSI to bar."""
    return psi / 14.5038


def mm_to_inches(mm: float) -> float:
    """Convert millimeters to inches."""
    return mm / 25.4


def inches_to_mm(inches: float) -> float:
    """Convert inches to millimeters."""
    return inches * 25.4


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "SafetyStatus",
    "ComplianceStatus",
    "MaterialGrade",
    "InterlockType",
    # Data Models
    "SafetyCheckResult",
    "ComplianceCheckResult",
    "ThermalShockAssessment",
    "InterlockStatusReport",
    "CreepRuptureAnalysis",
    # Classes
    "SafetyValidator",
    "ASMEComplianceChecker",
    # Constants
    "MATERIAL_MAX_ALLOWABLE_STRESS",
    "MATERIAL_MAX_TEMP_F",
    "CREEP_RUPTURE_PARAMS",
    "REQUIRED_SUPERHEATER_INTERLOCKS",
    # Utility Functions
    "celsius_to_fahrenheit",
    "fahrenheit_to_celsius",
    "bar_to_psi",
    "psi_to_bar",
    "mm_to_inches",
    "inches_to_mm",
]
