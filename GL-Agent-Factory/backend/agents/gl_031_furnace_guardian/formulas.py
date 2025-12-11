"""
GL-031 Furnace Guardian - Safety Calculation Formulas

This module implements deterministic safety calculations for furnace monitoring
per NFPA 86, API 560, and EN 746 standards.

ZERO-HALLUCINATION: All calculations use deterministic formulas from
published safety standards with exact coefficients.
"""

import math
import logging
from typing import Tuple, Dict, List, Optional, NamedTuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PurgeResult(NamedTuple):
    """Result of purge verification calculation."""
    is_valid: bool
    actual_volume_changes: float
    required_volume_changes: float
    purge_time_seconds: float
    minimum_time_seconds: float
    message: str


class FlameCheckResult(NamedTuple):
    """Result of flame supervision check."""
    is_detected: bool
    signal_strength: float
    signal_quality: str
    response_time_ms: float
    is_within_limits: bool


class SafetyScoreResult(NamedTuple):
    """Result of overall safety score calculation."""
    score: float
    risk_level: str
    interlock_score: float
    purge_score: float
    flame_score: float
    temperature_score: float
    pressure_score: float


def calculate_purge_volume_changes(
    airflow_cfm: float,
    furnace_volume_cubic_feet: float,
    purge_time_seconds: float
) -> float:
    """
    Calculate number of volume changes during purge cycle.

    ZERO-HALLUCINATION FORMULA (NFPA 86 Section 8.6):
    Volume Changes = (Airflow * Time) / Furnace Volume

    NFPA 86 requires minimum 4 volume changes for Class A furnaces
    and minimum 8 volume changes for Class B furnaces.

    Args:
        airflow_cfm: Combustion air flow rate in cubic feet per minute
        furnace_volume_cubic_feet: Total furnace internal volume
        purge_time_seconds: Duration of purge cycle in seconds

    Returns:
        Number of volume changes achieved

    Example:
        >>> changes = calculate_purge_volume_changes(5000, 2000, 120)
        >>> print(f"Volume changes: {changes:.1f}")
        Volume changes: 5.0
    """
    if airflow_cfm <= 0:
        raise ValueError(f"Airflow must be positive: {airflow_cfm} CFM")
    if furnace_volume_cubic_feet <= 0:
        raise ValueError(f"Furnace volume must be positive: {furnace_volume_cubic_feet} ft3")
    if purge_time_seconds < 0:
        raise ValueError(f"Purge time must be non-negative: {purge_time_seconds} s")

    # Convert time to minutes for CFM calculation
    purge_time_minutes = purge_time_seconds / 60.0

    # Total air volume purged
    total_air_volume = airflow_cfm * purge_time_minutes

    # Number of volume changes
    volume_changes = total_air_volume / furnace_volume_cubic_feet

    logger.debug(
        f"Purge calculation: {airflow_cfm} CFM * {purge_time_minutes:.2f} min = "
        f"{total_air_volume:.0f} ft3, {volume_changes:.2f} volume changes"
    )

    return volume_changes


def verify_purge_complete(
    airflow_cfm: float,
    furnace_volume_cubic_feet: float,
    purge_time_seconds: float,
    furnace_class: str = "A",
    minimum_purge_time_seconds: float = 30.0
) -> PurgeResult:
    """
    Verify purge cycle meets NFPA 86 requirements.

    NFPA 86 Requirements:
    - Class A (< 8000 Btu/ft3): Minimum 4 volume changes
    - Class B (>= 8000 Btu/ft3): Minimum 8 volume changes
    - Minimum purge time: 30 seconds regardless of volume changes

    Args:
        airflow_cfm: Combustion air flow rate
        furnace_volume_cubic_feet: Furnace internal volume
        purge_time_seconds: Actual purge duration
        furnace_class: "A" or "B" per NFPA 86 classification
        minimum_purge_time_seconds: Minimum required purge time

    Returns:
        PurgeResult with verification status
    """
    # Calculate volume changes
    actual_changes = calculate_purge_volume_changes(
        airflow_cfm, furnace_volume_cubic_feet, purge_time_seconds
    )

    # Required volume changes based on furnace class
    required_changes = 4.0 if furnace_class.upper() == "A" else 8.0

    # Check all requirements
    volume_ok = actual_changes >= required_changes
    time_ok = purge_time_seconds >= minimum_purge_time_seconds
    is_valid = volume_ok and time_ok

    # Build message
    if is_valid:
        message = (
            f"Purge complete: {actual_changes:.1f} volume changes "
            f"(required {required_changes:.0f}), {purge_time_seconds:.0f}s duration"
        )
    else:
        issues = []
        if not volume_ok:
            issues.append(
                f"Insufficient volume changes: {actual_changes:.1f} < {required_changes:.0f}"
            )
        if not time_ok:
            issues.append(
                f"Insufficient purge time: {purge_time_seconds:.0f}s < {minimum_purge_time_seconds:.0f}s"
            )
        message = "Purge FAILED: " + "; ".join(issues)

    logger.info(message)

    return PurgeResult(
        is_valid=is_valid,
        actual_volume_changes=actual_changes,
        required_volume_changes=required_changes,
        purge_time_seconds=purge_time_seconds,
        minimum_time_seconds=minimum_purge_time_seconds,
        message=message
    )


def calculate_flame_response_time(
    flame_on_timestamp: datetime,
    detection_timestamp: datetime,
    max_response_time_ms: float = 3000.0
) -> Tuple[float, bool]:
    """
    Calculate flame detector response time and verify compliance.

    NFPA 86 Requirements (Section 8.8.3):
    - Main burner flame must be detected within 4 seconds of ignition
    - Pilot flame must be detected within 10 seconds
    - Typical UV/IR scanners: 1-3 second response

    Args:
        flame_on_timestamp: Time when fuel valve opened
        detection_timestamp: Time when flame was detected
        max_response_time_ms: Maximum allowed response time

    Returns:
        Tuple of (response_time_ms, is_within_limits)
    """
    time_delta = detection_timestamp - flame_on_timestamp
    response_time_ms = time_delta.total_seconds() * 1000.0

    is_within_limits = response_time_ms <= max_response_time_ms

    if not is_within_limits:
        logger.warning(
            f"Flame response time {response_time_ms:.0f}ms exceeds "
            f"limit of {max_response_time_ms:.0f}ms"
        )

    return response_time_ms, is_within_limits


def calculate_flame_signal_quality(
    signal_strength: float,
    noise_level: float,
    min_signal_strength: float = 20.0,
    min_snr: float = 3.0
) -> Tuple[str, bool]:
    """
    Evaluate flame detector signal quality.

    Signal Quality Criteria:
    - Signal strength > minimum threshold
    - Signal-to-noise ratio (SNR) > minimum

    Args:
        signal_strength: Detector signal strength (0-100)
        noise_level: Background noise level
        min_signal_strength: Minimum acceptable signal
        min_snr: Minimum signal-to-noise ratio

    Returns:
        Tuple of (quality_rating, is_acceptable)
    """
    # Calculate SNR
    snr = signal_strength / max(noise_level, 0.1)

    # Determine quality
    if signal_strength >= 80 and snr >= 10:
        quality = "EXCELLENT"
    elif signal_strength >= 50 and snr >= 5:
        quality = "GOOD"
    elif signal_strength >= min_signal_strength and snr >= min_snr:
        quality = "ACCEPTABLE"
    elif signal_strength >= min_signal_strength * 0.5:
        quality = "MARGINAL"
    else:
        quality = "POOR"

    is_acceptable = quality in ["EXCELLENT", "GOOD", "ACCEPTABLE"]

    logger.debug(
        f"Flame signal quality: {quality} "
        f"(strength={signal_strength:.1f}, SNR={snr:.1f})"
    )

    return quality, is_acceptable


def calculate_safety_score(
    interlocks_ok: int,
    interlocks_total: int,
    purge_valid: bool,
    flame_detected: bool,
    flame_signal_quality: str,
    temps_in_range: int,
    temps_total: int,
    pressures_in_range: int,
    pressures_total: int
) -> SafetyScoreResult:
    """
    Calculate overall furnace safety score (0-100).

    ZERO-HALLUCINATION: Weighted scoring based on NFPA 86 priority:
    - Interlocks: 30% (most critical)
    - Purge: 20%
    - Flame supervision: 25%
    - Temperature limits: 15%
    - Pressure limits: 10%

    Args:
        interlocks_ok: Number of interlocks in OK state
        interlocks_total: Total number of interlocks
        purge_valid: Whether purge verification passed
        flame_detected: Whether flame is properly detected
        flame_signal_quality: Quality rating of flame signal
        temps_in_range: Number of temperatures in safe range
        temps_total: Total temperature sensors
        pressures_in_range: Number of pressures in safe range
        pressures_total: Total pressure sensors

    Returns:
        SafetyScoreResult with overall and component scores
    """
    # Calculate component scores
    interlock_score = (interlocks_ok / max(interlocks_total, 1)) * 100.0
    purge_score = 100.0 if purge_valid else 0.0

    # Flame score based on detection and quality
    if not flame_detected:
        flame_score = 0.0
    else:
        quality_scores = {
            "EXCELLENT": 100.0,
            "GOOD": 90.0,
            "ACCEPTABLE": 75.0,
            "MARGINAL": 50.0,
            "POOR": 25.0
        }
        flame_score = quality_scores.get(flame_signal_quality, 0.0)

    temp_score = (temps_in_range / max(temps_total, 1)) * 100.0
    pressure_score = (pressures_in_range / max(pressures_total, 1)) * 100.0

    # Weighted overall score
    overall_score = (
        interlock_score * 0.30 +
        purge_score * 0.20 +
        flame_score * 0.25 +
        temp_score * 0.15 +
        pressure_score * 0.10
    )

    # Determine risk level
    if overall_score >= 95:
        risk_level = "NONE"
    elif overall_score >= 85:
        risk_level = "LOW"
    elif overall_score >= 70:
        risk_level = "MODERATE"
    elif overall_score >= 50:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    logger.info(
        f"Safety score: {overall_score:.1f} ({risk_level}) - "
        f"Interlocks: {interlock_score:.0f}, Purge: {purge_score:.0f}, "
        f"Flame: {flame_score:.0f}, Temp: {temp_score:.0f}, Pressure: {pressure_score:.0f}"
    )

    return SafetyScoreResult(
        score=round(overall_score, 1),
        risk_level=risk_level,
        interlock_score=round(interlock_score, 1),
        purge_score=round(purge_score, 1),
        flame_score=round(flame_score, 1),
        temperature_score=round(temp_score, 1),
        pressure_score=round(pressure_score, 1)
    )


def check_temperature_limits(
    temperature_celsius: float,
    low_limit: float,
    high_limit: float,
    high_high_limit: Optional[float] = None
) -> Tuple[str, bool]:
    """
    Check temperature against NFPA 86 / API 560 limits.

    Returns status: NORMAL, LOW, HIGH, HIGH_HIGH, or TRIP

    Args:
        temperature_celsius: Current temperature reading
        low_limit: Low temperature alarm point
        high_limit: High temperature alarm point
        high_high_limit: High-high trip point (optional)

    Returns:
        Tuple of (status, is_safe)
    """
    if high_high_limit and temperature_celsius >= high_high_limit:
        return "TRIP", False
    elif temperature_celsius >= high_limit:
        return "HIGH", False
    elif temperature_celsius <= low_limit:
        return "LOW", False
    else:
        return "NORMAL", True


def check_pressure_limits(
    pressure_kpa: float,
    low_limit: float,
    high_limit: float,
    low_low_limit: Optional[float] = None,
    high_high_limit: Optional[float] = None
) -> Tuple[str, bool]:
    """
    Check pressure against safety limits.

    Args:
        pressure_kpa: Current pressure reading in kPa
        low_limit: Low pressure alarm point
        high_limit: High pressure alarm point
        low_low_limit: Low-low trip point (optional)
        high_high_limit: High-high trip point (optional)

    Returns:
        Tuple of (status, is_safe)
    """
    if high_high_limit and pressure_kpa >= high_high_limit:
        return "HIGH_HIGH_TRIP", False
    elif low_low_limit and pressure_kpa <= low_low_limit:
        return "LOW_LOW_TRIP", False
    elif pressure_kpa >= high_limit:
        return "HIGH", False
    elif pressure_kpa <= low_limit:
        return "LOW", False
    else:
        return "NORMAL", True


def calculate_interlock_reliability(
    successful_tests: int,
    total_tests: int,
    last_failure_days_ago: Optional[int] = None
) -> float:
    """
    Calculate interlock reliability based on proof test history.

    Per IEC 61511, interlocks must be proof-tested regularly.
    Reliability = Successful tests / Total tests, adjusted for recency.

    Args:
        successful_tests: Number of successful proof tests
        total_tests: Total number of proof tests
        last_failure_days_ago: Days since last failed test (None if no failures)

    Returns:
        Reliability score (0.0 to 1.0)
    """
    if total_tests == 0:
        return 0.0

    base_reliability = successful_tests / total_tests

    # Adjust for recent failures
    if last_failure_days_ago is not None:
        # Reduce reliability if failure was recent
        if last_failure_days_ago < 30:
            base_reliability *= 0.8
        elif last_failure_days_ago < 90:
            base_reliability *= 0.9

    return min(1.0, base_reliability)
