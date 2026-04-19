# -*- coding: utf-8 -*-
"""
GL-002 FLAMEGUARD - NFPA 85 Compliance Validator

Validates boiler operations against NFPA 85 requirements.

NFPA 85 (Boiler and Combustion Systems Hazards Code) Key Requirements:
- Chapter 5: Single Burner Boilers
- Purge timing (minimum 4 volume changes)
- Flame detection timeout (max 10 seconds for ignition)
- Trial for ignition timing
- Post-purge requirements
- Safety interlock requirements

Reference: NFPA 85-2023
Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib
import logging

logger = logging.getLogger(__name__)


class NFPA85Requirement(str, Enum):
    """NFPA 85 requirement categories."""
    PURGE_TIMING = "5.6.4"
    FLAME_DETECTION = "5.3.3"
    TRIAL_FOR_IGNITION = "5.6.5"
    FLAME_FAILURE = "5.3.5"
    POST_PURGE = "5.7.2"
    SAFETY_INTERLOCKS = "5.3.1"
    EMERGENCY_SHUTDOWN = "5.3.4"
    PERMISSIVE_LOGIC = "5.4.1"


class ComplianceStatus(str, Enum):
    """Compliance check result status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    requirement: NFPA85Requirement
    status: ComplianceStatus
    description: str
    actual_value: Any
    required_value: Any
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    recommendation: Optional[str] = None
    evidence: Optional[Dict[str, Any]] = None


@dataclass
class NFPA85ValidationReport:
    """Complete NFPA 85 validation report."""
    report_id: str
    boiler_id: str
    timestamp: datetime
    overall_status: ComplianceStatus
    results: List[ValidationResult]
    compliance_score: float  # 0-100
    critical_failures: List[ValidationResult]
    warnings: List[ValidationResult]
    provenance_hash: str = ""

    def __post_init__(self):
        if not self.provenance_hash:
            content = f"{self.report_id}|{self.boiler_id}|{self.timestamp.isoformat()}"
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


@dataclass
class PurgeParameters:
    """Purge sequence parameters."""
    furnace_volume_cuft: float
    air_flow_rate_cfm: float
    purge_duration_s: float
    air_flow_percentage: float = 100.0  # % of full load airflow


@dataclass
class FlameDetectionParameters:
    """Flame detection system parameters."""
    scanner_type: str  # UV, IR, UV/IR
    response_time_ms: float
    self_check_interval_s: float
    voting_scheme: str  # "1oo1", "1oo2", "2oo2", "2oo3"
    flame_signal_percent: float


@dataclass
class TrialForIgnitionParameters:
    """Trial for ignition timing parameters."""
    pilot_trial_time_s: float
    main_flame_trial_time_s: float
    pilot_flame_signal: float
    main_flame_signal: float


class NFPA85Validator:
    """
    NFPA 85 Compliance Validator.

    Validates boiler operations against NFPA 85-2023 requirements.
    """

    VERSION = "1.0.0"

    # NFPA 85 Timing Requirements
    MIN_PURGE_VOLUME_CHANGES = 4
    MIN_PURGE_AIRFLOW_PERCENT = 25.0
    MAX_PILOT_TRIAL_TIME_S = 10.0
    MAX_MAIN_FLAME_TRIAL_TIME_S = 10.0
    MAX_FLAME_FAILURE_RESPONSE_S = 4.0
    MIN_FLAME_SIGNAL_PERCENT = 10.0
    MIN_POST_PURGE_TIME_S = 60.0

    def __init__(self, boiler_id: str) -> None:
        """Initialize validator for a specific boiler."""
        self.boiler_id = boiler_id
        self._validation_history: List[NFPA85ValidationReport] = []
        logger.info(f"NFPA85Validator initialized for {boiler_id}")

    def validate_purge_timing(
        self,
        params: PurgeParameters,
    ) -> ValidationResult:
        """
        Validate purge sequence timing per NFPA 85 5.6.4.

        Requirements:
        - Minimum 4 volume changes
        - Airflow at least 25% of full load
        - Continuous operation during purge
        """
        # Calculate volume changes
        purge_volume = params.air_flow_rate_cfm * (params.purge_duration_s / 60)
        volume_changes = purge_volume / params.furnace_volume_cuft

        # Check airflow percentage
        airflow_ok = params.air_flow_percentage >= self.MIN_PURGE_AIRFLOW_PERCENT

        # Determine compliance
        if volume_changes >= self.MIN_PURGE_VOLUME_CHANGES and airflow_ok:
            status = ComplianceStatus.COMPLIANT
            description = f"Purge achieves {volume_changes:.1f} volume changes with {params.air_flow_percentage}% airflow"
            recommendation = None
        elif volume_changes >= self.MIN_PURGE_VOLUME_CHANGES * 0.9:
            status = ComplianceStatus.WARNING
            description = f"Purge marginally compliant with {volume_changes:.1f} volume changes"
            recommendation = "Consider increasing purge duration for safety margin"
        else:
            status = ComplianceStatus.NON_COMPLIANT
            description = f"Purge achieves only {volume_changes:.1f} volume changes (min: {self.MIN_PURGE_VOLUME_CHANGES})"
            recommendation = f"Increase purge duration to achieve {self.MIN_PURGE_VOLUME_CHANGES} volume changes"

        return ValidationResult(
            requirement=NFPA85Requirement.PURGE_TIMING,
            status=status,
            description=description,
            actual_value=volume_changes,
            required_value=self.MIN_PURGE_VOLUME_CHANGES,
            recommendation=recommendation,
            evidence={
                "furnace_volume_cuft": params.furnace_volume_cuft,
                "air_flow_rate_cfm": params.air_flow_rate_cfm,
                "purge_duration_s": params.purge_duration_s,
                "calculated_volume_changes": volume_changes,
                "airflow_percentage": params.air_flow_percentage,
            }
        )

    def validate_flame_detection(
        self,
        params: FlameDetectionParameters,
    ) -> ValidationResult:
        """
        Validate flame detection system per NFPA 85 5.3.3.

        Requirements:
        - Self-checking flame scanner
        - Appropriate voting scheme for SIL rating
        - Response time suitable for application
        """
        issues = []

        # Check flame signal threshold
        if params.flame_signal_percent < self.MIN_FLAME_SIGNAL_PERCENT:
            issues.append(f"Flame signal {params.flame_signal_percent}% below {self.MIN_FLAME_SIGNAL_PERCENT}% threshold")

        # Check response time (typical requirement <1 second)
        if params.response_time_ms > 1000:
            issues.append(f"Response time {params.response_time_ms}ms exceeds 1000ms")

        # Check self-check interval (should be continuous or periodic)
        if params.self_check_interval_s > 60:
            issues.append(f"Self-check interval {params.self_check_interval_s}s exceeds 60s")

        # Validate voting scheme
        valid_schemes = ["1oo1", "1oo2", "2oo2", "2oo3"]
        if params.voting_scheme not in valid_schemes:
            issues.append(f"Invalid voting scheme {params.voting_scheme}")

        if not issues:
            status = ComplianceStatus.COMPLIANT
            description = f"Flame detection system ({params.scanner_type}, {params.voting_scheme}) compliant"
        else:
            status = ComplianceStatus.NON_COMPLIANT
            description = "; ".join(issues)

        return ValidationResult(
            requirement=NFPA85Requirement.FLAME_DETECTION,
            status=status,
            description=description,
            actual_value={
                "scanner_type": params.scanner_type,
                "voting_scheme": params.voting_scheme,
                "response_time_ms": params.response_time_ms,
            },
            required_value={
                "min_flame_signal": self.MIN_FLAME_SIGNAL_PERCENT,
                "max_response_ms": 1000,
            },
            recommendation="Ensure flame scanner is properly maintained" if issues else None,
            evidence={"issues": issues}
        )

    def validate_trial_for_ignition(
        self,
        params: TrialForIgnitionParameters,
    ) -> ValidationResult:
        """
        Validate trial for ignition timing per NFPA 85 5.6.5.

        Requirements:
        - Pilot trial: 10 seconds maximum
        - Main flame trial: 10 seconds maximum
        - Flame must be proven within trial period
        """
        issues = []

        if params.pilot_trial_time_s > self.MAX_PILOT_TRIAL_TIME_S:
            issues.append(
                f"Pilot trial {params.pilot_trial_time_s}s exceeds {self.MAX_PILOT_TRIAL_TIME_S}s limit"
            )

        if params.main_flame_trial_time_s > self.MAX_MAIN_FLAME_TRIAL_TIME_S:
            issues.append(
                f"Main flame trial {params.main_flame_trial_time_s}s exceeds {self.MAX_MAIN_FLAME_TRIAL_TIME_S}s limit"
            )

        if not issues:
            status = ComplianceStatus.COMPLIANT
            description = (
                f"Trial times compliant: pilot={params.pilot_trial_time_s}s, "
                f"main={params.main_flame_trial_time_s}s"
            )
        else:
            status = ComplianceStatus.NON_COMPLIANT
            description = "; ".join(issues)

        return ValidationResult(
            requirement=NFPA85Requirement.TRIAL_FOR_IGNITION,
            status=status,
            description=description,
            actual_value={
                "pilot_trial_s": params.pilot_trial_time_s,
                "main_trial_s": params.main_flame_trial_time_s,
            },
            required_value={
                "max_pilot_trial_s": self.MAX_PILOT_TRIAL_TIME_S,
                "max_main_trial_s": self.MAX_MAIN_FLAME_TRIAL_TIME_S,
            },
            recommendation="Reduce trial times to comply" if issues else None,
        )

    def validate_flame_failure_response(
        self,
        response_time_s: float,
    ) -> ValidationResult:
        """
        Validate flame failure response time per NFPA 85 5.3.5.2.

        Requirement: 4 seconds maximum for fuel shutoff.
        """
        if response_time_s <= self.MAX_FLAME_FAILURE_RESPONSE_S:
            status = ComplianceStatus.COMPLIANT
            description = f"Flame failure response {response_time_s}s within {self.MAX_FLAME_FAILURE_RESPONSE_S}s limit"
        else:
            status = ComplianceStatus.NON_COMPLIANT
            description = f"Flame failure response {response_time_s}s exceeds {self.MAX_FLAME_FAILURE_RESPONSE_S}s limit"

        return ValidationResult(
            requirement=NFPA85Requirement.FLAME_FAILURE,
            status=status,
            description=description,
            actual_value=response_time_s,
            required_value=self.MAX_FLAME_FAILURE_RESPONSE_S,
            recommendation="Upgrade flame safeguard system" if response_time_s > self.MAX_FLAME_FAILURE_RESPONSE_S else None,
        )

    def validate_post_purge(
        self,
        post_purge_duration_s: float,
    ) -> ValidationResult:
        """
        Validate post-purge timing per NFPA 85 5.7.2.

        Recommendation: Minimum 1 minute post-purge.
        """
        if post_purge_duration_s >= self.MIN_POST_PURGE_TIME_S:
            status = ComplianceStatus.COMPLIANT
            description = f"Post-purge duration {post_purge_duration_s}s meets minimum"
        else:
            status = ComplianceStatus.WARNING
            description = f"Post-purge duration {post_purge_duration_s}s below recommended {self.MIN_POST_PURGE_TIME_S}s"

        return ValidationResult(
            requirement=NFPA85Requirement.POST_PURGE,
            status=status,
            description=description,
            actual_value=post_purge_duration_s,
            required_value=self.MIN_POST_PURGE_TIME_S,
        )

    def validate_safety_interlocks(
        self,
        interlocks: Dict[str, bool],
    ) -> ValidationResult:
        """
        Validate required safety interlocks per NFPA 85 5.3.1.

        Required interlocks:
        - Combustion air proving
        - Fuel pressure proving
        - Flame detection
        - Low water cutoff
        - High steam pressure
        """
        required_interlocks = [
            "combustion_air_proving",
            "fuel_pressure_proving",
            "flame_detection",
            "low_water_cutoff",
            "high_steam_pressure",
        ]

        missing = []
        for interlock in required_interlocks:
            if interlock not in interlocks:
                missing.append(interlock)
            elif not interlocks[interlock]:
                missing.append(f"{interlock} (not active)")

        if not missing:
            status = ComplianceStatus.COMPLIANT
            description = "All required safety interlocks present and active"
        else:
            status = ComplianceStatus.NON_COMPLIANT
            description = f"Missing or inactive interlocks: {', '.join(missing)}"

        return ValidationResult(
            requirement=NFPA85Requirement.SAFETY_INTERLOCKS,
            status=status,
            description=description,
            actual_value=interlocks,
            required_value=required_interlocks,
            recommendation="Install/activate missing interlocks" if missing else None,
        )

    def generate_compliance_report(
        self,
        purge_params: Optional[PurgeParameters] = None,
        flame_params: Optional[FlameDetectionParameters] = None,
        trial_params: Optional[TrialForIgnitionParameters] = None,
        flame_failure_response_s: Optional[float] = None,
        post_purge_duration_s: Optional[float] = None,
        interlocks: Optional[Dict[str, bool]] = None,
    ) -> NFPA85ValidationReport:
        """
        Generate comprehensive NFPA 85 compliance report.
        """
        results = []
        critical_failures = []
        warnings = []

        # Validate purge timing
        if purge_params:
            result = self.validate_purge_timing(purge_params)
            results.append(result)
            if result.status == ComplianceStatus.NON_COMPLIANT:
                critical_failures.append(result)
            elif result.status == ComplianceStatus.WARNING:
                warnings.append(result)

        # Validate flame detection
        if flame_params:
            result = self.validate_flame_detection(flame_params)
            results.append(result)
            if result.status == ComplianceStatus.NON_COMPLIANT:
                critical_failures.append(result)

        # Validate trial for ignition
        if trial_params:
            result = self.validate_trial_for_ignition(trial_params)
            results.append(result)
            if result.status == ComplianceStatus.NON_COMPLIANT:
                critical_failures.append(result)

        # Validate flame failure response
        if flame_failure_response_s is not None:
            result = self.validate_flame_failure_response(flame_failure_response_s)
            results.append(result)
            if result.status == ComplianceStatus.NON_COMPLIANT:
                critical_failures.append(result)

        # Validate post-purge
        if post_purge_duration_s is not None:
            result = self.validate_post_purge(post_purge_duration_s)
            results.append(result)
            if result.status == ComplianceStatus.WARNING:
                warnings.append(result)

        # Validate safety interlocks
        if interlocks:
            result = self.validate_safety_interlocks(interlocks)
            results.append(result)
            if result.status == ComplianceStatus.NON_COMPLIANT:
                critical_failures.append(result)

        # Calculate compliance score
        if results:
            compliant_count = sum(1 for r in results if r.status == ComplianceStatus.COMPLIANT)
            compliance_score = (compliant_count / len(results)) * 100
        else:
            compliance_score = 0.0

        # Determine overall status
        if critical_failures:
            overall_status = ComplianceStatus.NON_COMPLIANT
        elif warnings:
            overall_status = ComplianceStatus.WARNING
        elif results:
            overall_status = ComplianceStatus.COMPLIANT
        else:
            overall_status = ComplianceStatus.NOT_APPLICABLE

        # Generate report
        report = NFPA85ValidationReport(
            report_id=f"NFPA85-{self.boiler_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            boiler_id=self.boiler_id,
            timestamp=datetime.now(timezone.utc),
            overall_status=overall_status,
            results=results,
            compliance_score=compliance_score,
            critical_failures=critical_failures,
            warnings=warnings,
        )

        self._validation_history.append(report)
        return report

    def get_validation_history(self) -> List[NFPA85ValidationReport]:
        """Get history of validation reports."""
        return self._validation_history


# Convenience functions for common validations
def validate_purge_volume_changes(
    furnace_volume_cuft: float,
    air_flow_cfm: float,
    purge_time_s: float,
) -> Tuple[float, bool]:
    """
    Calculate and validate purge volume changes.

    Returns:
        Tuple of (volume_changes, is_compliant)
    """
    purge_volume = air_flow_cfm * (purge_time_s / 60)
    volume_changes = purge_volume / furnace_volume_cuft
    is_compliant = volume_changes >= NFPA85Validator.MIN_PURGE_VOLUME_CHANGES
    return volume_changes, is_compliant


def calculate_required_purge_time(
    furnace_volume_cuft: float,
    air_flow_cfm: float,
    target_volume_changes: float = 4.0,
) -> float:
    """
    Calculate required purge time to achieve target volume changes.

    Returns:
        Required purge time in seconds
    """
    required_purge_volume = furnace_volume_cuft * target_volume_changes
    required_time_minutes = required_purge_volume / air_flow_cfm
    return required_time_minutes * 60  # Convert to seconds


__all__ = [
    "NFPA85Requirement",
    "ComplianceStatus",
    "ValidationResult",
    "NFPA85ValidationReport",
    "PurgeParameters",
    "FlameDetectionParameters",
    "TrialForIgnitionParameters",
    "NFPA85Validator",
    "validate_purge_volume_changes",
    "calculate_required_purge_time",
]
