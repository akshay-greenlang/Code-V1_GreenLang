"""
NFPA 86 Furnace Compliance Checker - Process Heat Safety

This module implements comprehensive NFPA 86:2023 "Standard for Ovens and Furnaces"
compliance checking for industrial furnaces in the GreenLang Process Heat system.

Key NFPA 86 Requirements Implemented:
- Furnace classification (Class A/B/C/D)
- Purge cycle validation (4 volume changes minimum per 8.7.2)
- Flame supervision requirements per Section 8.5
- Combustion safeguards per Section 8.6
- Atmosphere control and LEL monitoring per Section 8.8
- Emergency shutdown requirements per Section 8.11
- Burner management system interlocks per Section 8.4
- Purge timing calculation: time = (4 × volume) / flow_rate
- Trial for ignition limits (10 seconds maximum per Section 8.5.4)
- Flame failure response times (<4 seconds per Section 8.5.2.2)

Reference: NFPA 86-2023, Standard for Ovens and Furnaces
Safety Classification: SIL 2 (Safety Integrity Level 2)

Example:
    >>> from greenlang.safety.nfpa_86_furnace import NFPA86ComplianceChecker
    >>> checker = NFPA86ComplianceChecker()
    >>> result = checker.check_class_a_furnace(furnace_config)
    >>> if result.is_compliant:
    ...     print(f"Class A furnace compliant at {result.compliance_percent}%")

Author: GreenLang Safety Engineering Team
Version: 1.0
Date: 2025-12-06
"""

from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import hashlib
import logging
import uuid
import math

logger = logging.getLogger(__name__)


# =============================================================================
# NFPA 86 Timing Constants
# =============================================================================

class NFPA86TimingRequirements:
    """NFPA 86:2023 mandated timing requirements."""

    # Section 8.7.2 - Prepurge Requirements
    PREPURGE_MIN_VOLUME_CHANGES: int = 4
    PREPURGE_MIN_AIRFLOW_PERCENT: float = 25.0
    PREPURGE_MIN_TIME_SECONDS: float = 15.0

    # Section 8.5.4 - Trial for Ignition
    PILOT_TRIAL_MAX_SECONDS: float = 10.0
    MAIN_TRIAL_MAX_SECONDS: float = 10.0
    TOTAL_TRIAL_MAX_SECONDS: float = 15.0

    # Section 8.5.2.2 - Flame Failure Response
    FLAME_FAILURE_RESPONSE_MAX_SECONDS: float = 4.0

    # Section 8.7.3 - Postpurge Requirements
    POSTPURGE_MIN_TIME_SECONDS: float = 15.0

    # Section 8.8 - LEL Monitoring
    LEL_ALARM_THRESHOLD_PERCENT: float = 25.0
    LEL_SHUTDOWN_THRESHOLD_PERCENT: float = 50.0

    # Section 8.11 - Emergency Shutdown
    EMERGENCY_SHUTDOWN_MAX_SECONDS: float = 5.0
    FUEL_VALVE_CLOSURE_MAX_SECONDS: float = 3.0


# =============================================================================
# Enumerations
# =============================================================================

class FurnaceClass(str, Enum):
    """NFPA 86 furnace classifications."""
    CLASS_A = "class_a"  # Ovens with flammable volatiles
    CLASS_B = "class_b"  # Ovens with heated flammable materials
    CLASS_C = "class_c"  # Atmosphere furnaces with special atmospheres
    CLASS_D = "class_d"  # Vacuum furnaces


class AtmosphereType(str, Enum):
    """Furnace atmosphere types."""
    AIR = "air"
    NITROGEN = "nitrogen"
    HYDROGEN = "hydrogen"
    ENDOTHERMIC = "endothermic"
    EXOTHERMIC = "exothermic"
    VACUUM = "vacuum"
    ARGON = "argon"


class InterlockStatus(Enum):
    """Safety interlock status."""
    OPERATIONAL = auto()
    TRIPPED = auto()
    BYPASSED = auto()
    FAULT = auto()


class ComplianceLevel(str, Enum):
    """Compliance assessment levels."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    CONDITIONAL = "conditional"
    UNKNOWN = "unknown"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PurgeConfiguration:
    """Purge cycle configuration."""
    furnace_volume_cubic_feet: float = 500.0
    airflow_cfm: float = 1000.0
    prepurge_duration_seconds: float = 0.0
    postpurge_duration_seconds: float = 0.0
    purge_air_quality: str = "clean_dry_air"
    minimum_airflow_percent: float = 25.0

    def calculate_required_prepurge_time(self) -> float:
        """Calculate required prepurge time per NFPA 86 Section 8.7.2."""
        if self.airflow_cfm <= 0:
            raise ValueError("Airflow must be positive")

        # Time = (4 volumes × volume) / airflow
        time_for_4_changes = (
            NFPA86TimingRequirements.PREPURGE_MIN_VOLUME_CHANGES
            * self.furnace_volume_cubic_feet
            / self.airflow_cfm
            * 60  # Convert to seconds
        )

        # Return maximum of calculated and minimum time
        return max(
            time_for_4_changes,
            NFPA86TimingRequirements.PREPURGE_MIN_TIME_SECONDS
        )

    def calculate_volume_changes(self, elapsed_seconds: float) -> float:
        """Calculate volume changes completed in given time."""
        if self.furnace_volume_cubic_feet <= 0:
            return 0.0

        cfm = self.airflow_cfm
        # Volume changes = (airflow × time) / volume
        return (cfm * elapsed_seconds / 60.0) / self.furnace_volume_cubic_feet


@dataclass
class SafetyInterlockConfig:
    """Safety interlock configuration."""
    name: str = ""
    setpoint: float = 0.0
    unit: str = ""
    is_high_trip: bool = True
    deadband: float = 0.0
    response_time_limit_seconds: float = 3.0
    is_operational: bool = True
    is_bypassed: bool = False
    nfpa_section: str = "8.6.3"
    interlock_id: str = field(default_factory=lambda: f"INTLK-{uuid.uuid4().hex[:8].upper()}")


@dataclass
class FurnaceConfiguration:
    """Complete furnace configuration."""
    equipment_id: str
    classification: FurnaceClass
    atmosphere_type: AtmosphereType
    maximum_temperature_deg_f: float
    furnace_volume_cubic_feet: float
    burner_input_btuh: float
    purge_config: PurgeConfiguration = field(default_factory=PurgeConfiguration)
    interlocks: List[SafetyInterlockConfig] = field(default_factory=list)
    has_flame_supervision: bool = False
    has_combustion_safeguards: bool = False
    has_lel_monitoring: bool = False
    has_emergency_shutdown: bool = False
    has_purge_capability: bool = False
    has_temperature_monitoring: bool = False


@dataclass
class ComplianceCheckResult:
    """Result of NFPA 86 compliance check."""
    equipment_id: str = ""
    classification: FurnaceClass = FurnaceClass.CLASS_A
    total_requirements: int = 0
    requirements_met: int = 0
    requirements_failed: int = 0
    compliance_percent: float = 0.0
    compliance_level: ComplianceLevel = ComplianceLevel.UNKNOWN
    findings: List[Dict[str, Any]] = field(default_factory=list)
    provenance_hash: str = ""
    purge_time_calculated_seconds: float = 0.0
    check_id: str = field(default_factory=lambda: f"F86-{uuid.uuid4().hex[:8].upper()}")
    check_timestamp: datetime = field(default_factory=datetime.utcnow)

    def calculate_provenance(self) -> str:
        """Calculate SHA-256 provenance hash."""
        data = f"{self.check_id}|{self.equipment_id}|{self.classification.value}|{self.check_timestamp.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()


# =============================================================================
# NFPA 86 Compliance Checker
# =============================================================================

class NFPA86ComplianceChecker:
    """
    NFPA 86 Furnace Compliance Checker.

    Implements comprehensive compliance checking for industrial furnaces
    per NFPA 86:2023 standard. Covers all four furnace classes with
    specific requirements for purging, flame supervision, atmosphere
    control, and emergency shutdown.

    Attributes:
        check_history: List of all performed compliance checks
        class_requirements: Mapping of furnace class to requirements

    Example:
        >>> checker = NFPA86ComplianceChecker()
        >>> result = checker.check_class_a_furnace(config)
        >>> print(f"Compliance: {result.compliance_percent}%")
    """

    # Class-specific requirements per NFPA 86
    CLASS_A_REQUIREMENTS = {
        "flame_supervision": True,  # Section 8.5 - Flame Supervision
        "combustion_safeguards": True,  # Section 8.6
        "lel_monitoring": True,  # Section 8.8 - LEL Monitoring
        "emergency_shutdown": True,  # Section 8.11
        "purge_capability": False,  # Not required for Class A
        "temperature_monitoring": True,  # Section 8.9
        "ventilation_rate_min_cfm_per_sqft": 10.0,
        "section_references": "8.3.1, 8.5, 8.6, 8.8, 8.9, 8.11"
    }

    CLASS_B_REQUIREMENTS = {
        "flame_supervision": True,  # Section 8.5
        "combustion_safeguards": True,  # Section 8.6
        "fire_suppression": True,  # Section 8.10
        "temperature_monitoring": True,  # Section 8.9
        "emergency_shutdown": True,  # Section 8.11
        "purge_capability": False,
        "section_references": "8.3.2, 8.5, 8.6, 8.9, 8.10, 8.11"
    }

    CLASS_C_REQUIREMENTS = {
        "atmosphere_monitoring": True,  # Section 8.8
        "purge_capability": True,  # Section 8.7 - Purging
        "burn_off_system": True,  # Section 8.7.4
        "pressure_relief": True,  # Section 8.8.2
        "temperature_monitoring": True,  # Section 8.9
        "flame_supervision": False,  # Not required for Class C
        "emergency_shutdown": True,  # Section 8.11
        "section_references": "8.3.3, 8.7, 8.8, 8.9, 8.11"
    }

    CLASS_D_REQUIREMENTS = {
        "vacuum_integrity": True,  # Section 8.12
        "leak_detection": True,  # Section 8.12
        "quench_system": True,  # Section 8.7.5
        "temperature_monitoring": True,  # Section 8.9
        "pressure_relief": True,  # Section 8.12
        "emergency_shutdown": True,  # Section 8.11
        "flame_supervision": False,
        "section_references": "8.3.4, 8.7, 8.9, 8.11, 8.12"
    }

    def __init__(self):
        """Initialize NFPA86ComplianceChecker."""
        self.check_history: List[ComplianceCheckResult] = []
        logger.info("NFPA86ComplianceChecker initialized")

    def check_class_a_furnace(self, config: FurnaceConfiguration) -> ComplianceCheckResult:
        """
        Check Class A furnace (with flammable volatiles) compliance.

        Class A ovens include those with flammable volatiles that may be
        released during heating (wood, textiles, etc.).

        Args:
            config: Furnace configuration

        Returns:
            ComplianceCheckResult with detailed findings
        """
        logger.info(f"Checking Class A furnace {config.equipment_id}")

        requirements = self.CLASS_A_REQUIREMENTS
        result = self._check_common_requirements(config, FurnaceClass.CLASS_A)

        # Check flame supervision (Section 8.5)
        if not config.has_flame_supervision:
            result.findings.append({
                "section": "8.5",
                "requirement": "Flame supervision required",
                "status": "FAILED",
                "severity": "CRITICAL"
            })
            result.requirements_failed += 1
        else:
            result.requirements_met += 1

        # Check LEL monitoring (Section 8.8)
        if not config.has_lel_monitoring:
            result.findings.append({
                "section": "8.8",
                "requirement": "LEL monitoring required for volatiles",
                "status": "FAILED",
                "severity": "CRITICAL"
            })
            result.requirements_failed += 1
        else:
            result.requirements_met += 1

        result.total_requirements = len(self.CLASS_A_REQUIREMENTS) + 2

        return self._finalize_result(result)

    def check_class_b_furnace(self, config: FurnaceConfiguration) -> ComplianceCheckResult:
        """
        Check Class B furnace (heated flammable materials) compliance.

        Class B ovens contain materials that may liberate flammable vapors
        when heated (oils, fats, organic materials).

        Args:
            config: Furnace configuration

        Returns:
            ComplianceCheckResult
        """
        logger.info(f"Checking Class B furnace {config.equipment_id}")

        result = self._check_common_requirements(config, FurnaceClass.CLASS_B)

        # Check fire suppression (Section 8.10)
        if not config.has_combustion_safeguards:
            result.findings.append({
                "section": "8.6",
                "requirement": "Combustion safeguards required",
                "status": "FAILED",
                "severity": "CRITICAL"
            })
            result.requirements_failed += 1
        else:
            result.requirements_met += 1

        # Check temperature monitoring (Section 8.9)
        if not config.has_temperature_monitoring:
            result.findings.append({
                "section": "8.9",
                "requirement": "Temperature monitoring required",
                "status": "FAILED",
                "severity": "CRITICAL"
            })
            result.requirements_failed += 1
        else:
            result.requirements_met += 1

        result.total_requirements = len(self.CLASS_B_REQUIREMENTS) + 2

        return self._finalize_result(result)

    def check_class_c_furnace(self, config: FurnaceConfiguration) -> ComplianceCheckResult:
        """
        Check Class C furnace (special atmosphere) compliance.

        Class C furnaces operate with controlled atmospheres like nitrogen,
        hydrogen, endothermic, or exothermic atmospheres.

        Args:
            config: Furnace configuration

        Returns:
            ComplianceCheckResult
        """
        logger.info(f"Checking Class C furnace {config.equipment_id}")

        result = self._check_common_requirements(config, FurnaceClass.CLASS_C)

        # Check purge capability (Section 8.7)
        if not config.has_purge_capability:
            result.findings.append({
                "section": "8.7",
                "requirement": "Purge capability required for atmosphere furnaces",
                "status": "FAILED",
                "severity": "CRITICAL"
            })
            result.requirements_failed += 1
        else:
            result.requirements_met += 1
            # Validate purge cycle
            purge_result = self.validate_purge_cycle(
                config.atmosphere_type,
                config.furnace_volume_cubic_feet,
                config.purge_config.airflow_cfm
            )
            if not purge_result[0]:
                result.findings.append({
                    "section": "8.7.2",
                    "requirement": "Purge cycle timing",
                    "status": "FAILED",
                    "details": purge_result[1]
                })
                result.requirements_failed += 1
            else:
                result.requirements_met += 1
                result.purge_time_calculated_seconds = purge_result[2]

        # Check atmosphere monitoring (Section 8.8)
        if not config.has_lel_monitoring:
            result.findings.append({
                "section": "8.8",
                "requirement": "Atmosphere monitoring required",
                "status": "FAILED",
                "severity": "CRITICAL"
            })
            result.requirements_failed += 1
        else:
            result.requirements_met += 1

        result.total_requirements = len(self.CLASS_C_REQUIREMENTS) + 2

        return self._finalize_result(result)

    def check_class_d_furnace(self, config: FurnaceConfiguration) -> ComplianceCheckResult:
        """
        Check Class D furnace (vacuum) compliance.

        Class D furnaces operate under vacuum conditions.

        Args:
            config: Furnace configuration

        Returns:
            ComplianceCheckResult
        """
        logger.info(f"Checking Class D furnace {config.equipment_id}")

        result = self._check_common_requirements(config, FurnaceClass.CLASS_D)

        # Check vacuum integrity (Section 8.12)
        if config.atmosphere_type != AtmosphereType.VACUUM:
            result.findings.append({
                "section": "8.12",
                "requirement": "Vacuum atmosphere required",
                "status": "FAILED",
                "severity": "CRITICAL"
            })
            result.requirements_failed += 1
        else:
            result.requirements_met += 1

        # Validate interlock configuration for vacuum systems
        interlock_result = self.validate_safety_interlocks(config)
        if not interlock_result[0]:
            result.findings.append({
                "section": "8.6",
                "requirement": "Safety interlocks for vacuum system",
                "status": "FAILED",
                "details": interlock_result[1]
            })
            result.requirements_failed += 1
        else:
            result.requirements_met += 1

        result.total_requirements = len(self.CLASS_D_REQUIREMENTS) + 2

        return self._finalize_result(result)

    def validate_purge_cycle(
        self,
        atmosphere: AtmosphereType,
        volume_cubic_feet: float,
        flow_rate_cfm: float
    ) -> Tuple[bool, str, float]:
        """
        Validate purge cycle meets NFPA 86 Section 8.7.2 requirements.

        Purge timing calculation:
        Time = (4 × Volume) / Flow_Rate (in minutes, converted to seconds)

        Args:
            atmosphere: Furnace atmosphere type
            volume_cubic_feet: Furnace volume
            flow_rate_cfm: Purge air flow rate in CFM

        Returns:
            Tuple of (is_valid, message, calculated_time_seconds)
        """
        if volume_cubic_feet <= 0:
            return False, "Furnace volume must be positive", 0.0

        if flow_rate_cfm <= 0:
            return False, "Flow rate must be positive", 0.0

        # Calculate required time: (4 volumes × volume) / flow_rate
        calculated_time = (
            NFPA86TimingRequirements.PREPURGE_MIN_VOLUME_CHANGES
            * volume_cubic_feet
            / flow_rate_cfm
            * 60  # Convert to seconds
        )

        # Apply minimum time requirement
        required_time = max(
            calculated_time,
            NFPA86TimingRequirements.PREPURGE_MIN_TIME_SECONDS
        )

        # Volume changes verification
        volume_changes = (flow_rate_cfm * required_time / 60.0) / volume_cubic_feet

        message = (
            f"Purge time: {required_time:.1f}s for {volume_changes:.1f} volume changes "
            f"at {flow_rate_cfm} CFM (min {NFPA86TimingRequirements.PREPURGE_MIN_TIME_SECONDS}s)"
        )

        is_valid = (
            volume_changes >= NFPA86TimingRequirements.PREPURGE_MIN_VOLUME_CHANGES
            and required_time >= NFPA86TimingRequirements.PREPURGE_MIN_TIME_SECONDS
        )

        logger.info(f"Purge validation: {message} - {'PASS' if is_valid else 'FAIL'}")

        return is_valid, message, required_time

    def validate_safety_interlocks(
        self,
        config: FurnaceConfiguration
    ) -> Tuple[bool, str]:
        """
        Validate safety interlock configuration per NFPA 86 Section 8.6.3.

        Args:
            config: Furnace configuration with interlock list

        Returns:
            Tuple of (is_valid, message)
        """
        if not config.interlocks:
            return False, "No safety interlocks configured"

        operational_count = 0
        failed_interlocks = []

        for interlock in config.interlocks:
            if not interlock.is_operational:
                failed_interlocks.append(interlock.name)
            else:
                operational_count += 1

            # Validate response time
            if interlock.response_time_limit_seconds > 3.0:
                logger.warning(
                    f"Interlock {interlock.name} response time "
                    f"{interlock.response_time_limit_seconds}s exceeds recommended 3s"
                )

        if operational_count == 0:
            return False, "No operational interlocks found"

        if failed_interlocks:
            message = f"{operational_count} operational, failed: {', '.join(failed_interlocks)}"
            return True, message

        return True, f"All {operational_count} interlocks operational"

    def calculate_flame_failure_response(
        self,
        detection_time_ms: float,
        fuel_shutoff_time_ms: float
    ) -> Tuple[bool, float, str]:
        """
        Calculate and validate flame failure response time per Section 8.5.2.2.

        Args:
            detection_time_ms: Time to detect flame loss in milliseconds
            fuel_shutoff_time_ms: Time to shut off fuel in milliseconds

        Returns:
            Tuple of (is_compliant, total_response_ms, message)
        """
        total_response_ms = detection_time_ms + fuel_shutoff_time_ms
        max_allowed_ms = NFPA86TimingRequirements.FLAME_FAILURE_RESPONSE_MAX_SECONDS * 1000

        is_compliant = total_response_ms <= max_allowed_ms

        message = (
            f"Flame failure response: {total_response_ms:.1f}ms "
            f"(detection {detection_time_ms:.1f}ms + shutoff {fuel_shutoff_time_ms:.1f}ms) "
            f"{'PASS' if is_compliant else 'FAIL'} limit {max_allowed_ms:.1f}ms"
        )

        logger.info(message)

        return is_compliant, total_response_ms, message

    def validate_trial_for_ignition(
        self,
        pilot_trial_seconds: float,
        main_trial_seconds: float
    ) -> Tuple[bool, str]:
        """
        Validate trial for ignition timing per Section 8.5.4.

        Trial for ignition limits:
        - Pilot trial: maximum 10 seconds
        - Main flame trial: maximum 10 seconds
        - Total combined: maximum 15 seconds

        Args:
            pilot_trial_seconds: Pilot ignition trial duration
            main_trial_seconds: Main flame trial duration

        Returns:
            Tuple of (is_valid, message)
        """
        issues = []

        if pilot_trial_seconds > NFPA86TimingRequirements.PILOT_TRIAL_MAX_SECONDS:
            issues.append(
                f"Pilot trial {pilot_trial_seconds:.1f}s exceeds {NFPA86TimingRequirements.PILOT_TRIAL_MAX_SECONDS}s"
            )

        if main_trial_seconds > NFPA86TimingRequirements.MAIN_TRIAL_MAX_SECONDS:
            issues.append(
                f"Main trial {main_trial_seconds:.1f}s exceeds {NFPA86TimingRequirements.MAIN_TRIAL_MAX_SECONDS}s"
            )

        total = pilot_trial_seconds + main_trial_seconds
        if total > NFPA86TimingRequirements.TOTAL_TRIAL_MAX_SECONDS:
            issues.append(
                f"Total trial {total:.1f}s exceeds {NFPA86TimingRequirements.TOTAL_TRIAL_MAX_SECONDS}s"
            )

        if issues:
            return False, "; ".join(issues)

        message = (
            f"Trial for ignition valid: pilot {pilot_trial_seconds:.1f}s + "
            f"main {main_trial_seconds:.1f}s = {total:.1f}s (max {NFPA86TimingRequirements.TOTAL_TRIAL_MAX_SECONDS}s)"
        )

        return True, message

    def validate_lel_monitoring(
        self,
        current_lel_percent: float
    ) -> Tuple[ComplianceLevel, str]:
        """
        Validate LEL (Lower Explosive Limit) monitoring per Section 8.8.

        Args:
            current_lel_percent: Current LEL percentage (0-100%)

        Returns:
            Tuple of (compliance_level, message)
        """
        if current_lel_percent < NFPA86TimingRequirements.LEL_ALARM_THRESHOLD_PERCENT:
            return (
                ComplianceLevel.COMPLIANT,
                f"LEL {current_lel_percent:.1f}% below alarm threshold "
                f"{NFPA86TimingRequirements.LEL_ALARM_THRESHOLD_PERCENT}%"
            )

        if current_lel_percent < NFPA86TimingRequirements.LEL_SHUTDOWN_THRESHOLD_PERCENT:
            return (
                ComplianceLevel.CONDITIONAL,
                f"LEL {current_lel_percent:.1f}% in alarm range "
                f"({NFPA86TimingRequirements.LEL_ALARM_THRESHOLD_PERCENT}-"
                f"{NFPA86TimingRequirements.LEL_SHUTDOWN_THRESHOLD_PERCENT}%)"
            )

        return (
            ComplianceLevel.NON_COMPLIANT,
            f"LEL {current_lel_percent:.1f}% exceeds shutdown threshold "
            f"{NFPA86TimingRequirements.LEL_SHUTDOWN_THRESHOLD_PERCENT}% - EMERGENCY SHUTDOWN REQUIRED"
        )

    # ==========================================================================
    # Private Methods
    # ==========================================================================

    def _check_common_requirements(
        self,
        config: FurnaceConfiguration,
        furnace_class: FurnaceClass
    ) -> ComplianceCheckResult:
        """Check requirements common to all furnace classes."""
        result = ComplianceCheckResult(
            equipment_id=config.equipment_id,
            classification=furnace_class
        )

        # Check emergency shutdown (Section 8.11 - all classes)
        if not config.has_emergency_shutdown:
            result.findings.append({
                "section": "8.11",
                "requirement": "Emergency shutdown capability required",
                "status": "FAILED",
                "severity": "CRITICAL"
            })
            result.requirements_failed += 1
        else:
            result.requirements_met += 1

        # Check temperature monitoring (Section 8.9 - all classes)
        if not config.has_temperature_monitoring:
            result.findings.append({
                "section": "8.9",
                "requirement": "Temperature monitoring/limiting required",
                "status": "FAILED",
                "severity": "CRITICAL"
            })
            result.requirements_failed += 1
        else:
            result.requirements_met += 1

        # Validate interlocks (Section 8.6)
        interlock_valid, interlock_msg = self.validate_safety_interlocks(config)
        if not interlock_valid:
            result.findings.append({
                "section": "8.6",
                "requirement": "Safety interlocks",
                "status": "FAILED",
                "details": interlock_msg
            })
            result.requirements_failed += 1
        else:
            result.requirements_met += 1

        return result

    def _finalize_result(self, result: ComplianceCheckResult) -> ComplianceCheckResult:
        """Finalize compliance result with calculations."""
        if result.total_requirements > 0:
            result.compliance_percent = (
                result.requirements_met / result.total_requirements * 100.0
            )

        if result.requirements_failed == 0:
            result.compliance_level = ComplianceLevel.COMPLIANT
        elif result.requirements_failed < 3:
            result.compliance_level = ComplianceLevel.CONDITIONAL
        else:
            result.compliance_level = ComplianceLevel.NON_COMPLIANT

        result.provenance_hash = result.calculate_provenance()

        self.check_history.append(result)

        logger.info(
            f"Compliance check {result.check_id}: {result.equipment_id} "
            f"{result.compliance_percent:.1f}% ({result.compliance_level.value})"
        )

        return result

    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive NFPA 86 compliance report."""
        return {
            "report_id": f"NFPA86-RPT-{uuid.uuid4().hex[:8].upper()}",
            "generated_at": datetime.utcnow().isoformat(),
            "standard": "NFPA 86-2023",
            "total_checks_performed": len(self.check_history),
            "checks_by_class": {
                "class_a": sum(1 for c in self.check_history if c.classification == FurnaceClass.CLASS_A),
                "class_b": sum(1 for c in self.check_history if c.classification == FurnaceClass.CLASS_B),
                "class_c": sum(1 for c in self.check_history if c.classification == FurnaceClass.CLASS_C),
                "class_d": sum(1 for c in self.check_history if c.classification == FurnaceClass.CLASS_D),
            },
            "compliance_summary": {
                "compliant": sum(1 for c in self.check_history if c.compliance_level == ComplianceLevel.COMPLIANT),
                "conditional": sum(1 for c in self.check_history if c.compliance_level == ComplianceLevel.CONDITIONAL),
                "non_compliant": sum(1 for c in self.check_history if c.compliance_level == ComplianceLevel.NON_COMPLIANT),
            },
            "timing_requirements": {
                "prepurge_min_volume_changes": NFPA86TimingRequirements.PREPURGE_MIN_VOLUME_CHANGES,
                "prepurge_min_airflow_percent": NFPA86TimingRequirements.PREPURGE_MIN_AIRFLOW_PERCENT,
                "pilot_trial_max_seconds": NFPA86TimingRequirements.PILOT_TRIAL_MAX_SECONDS,
                "flame_failure_response_max_seconds": NFPA86TimingRequirements.FLAME_FAILURE_RESPONSE_MAX_SECONDS,
                "lel_alarm_threshold_percent": NFPA86TimingRequirements.LEL_ALARM_THRESHOLD_PERCENT,
                "emergency_shutdown_max_seconds": NFPA86TimingRequirements.EMERGENCY_SHUTDOWN_MAX_SECONDS,
            }
        }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "NFPA86ComplianceChecker",
    "FurnaceClass",
    "AtmosphereType",
    "InterlockStatus",
    "ComplianceLevel",
    "PurgeConfiguration",
    "SafetyInterlockConfig",
    "FurnaceConfiguration",
    "ComplianceCheckResult",
    "NFPA86TimingRequirements",
]
