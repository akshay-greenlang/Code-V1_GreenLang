# -*- coding: utf-8 -*-
"""
DataValidationEngine - PACK-039 Energy Monitoring Engine 3
===========================================================

12-check automated data quality validation engine per ASHRAE Guideline 14.
Validates energy meter readings against range limits, spike detection,
stuck-value identification, gap analysis, rollover detection, negative
value checks, sum checks, phase balance, power factor range, timestamp
integrity, duplicate detection, and completeness scoring.

Calculation Methodology:
    Range Check:
        PASS if min_limit <= value <= max_limit

    Spike Detection:
        spike if |value - prev_value| > spike_threshold * std_dev

    Stuck Value:
        stuck if value == prev_value for consecutive_count > threshold

    Gap Detection:
        gap if time_delta > expected_interval * gap_factor

    Rollover Detection:
        rollover if value_n < value_n-1 (for cumulative registers)

    Sum Check:
        PASS if |main_meter - SUM(submeters)| / main_meter <= tolerance

    Phase Balance:
        imbalance_pct = (max_phase - min_phase) / avg_phase * 100

    Completeness Score:
        completeness = valid_readings / expected_readings * 100

    Quality Score (weighted):
        score = SUM(check_weight * check_pass_rate) / SUM(check_weight)

    Quality Grade (per ASHRAE 14):
        A: score >= 95, completeness >= 99
        B: score >= 85, completeness >= 95
        C: score >= 70, completeness >= 90
        D: score >= 50, completeness >= 80
        F: below D thresholds

Regulatory References:
    - ASHRAE Guideline 14-2014 - Measurement of Energy and Demand Savings
    - IPMVP Volume I - Data quality requirements
    - ISO 50001:2018 - Monitoring, measurement, analysis
    - IEC 62053-21/22 - Revenue meter accuracy
    - ANSI C12.1 - Data validation requirements
    - EN 50160:2010 - Voltage quality characteristics

Zero-Hallucination:
    - All 12 checks use deterministic numeric comparisons
    - Quality scoring uses fixed weighted formula
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-039 Energy Monitoring
Engine:  3 of 5
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ValidationCheck(str, Enum):
    """Validation check type identifiers per ASHRAE Guideline 14.

    RANGE:          Value within expected min/max bounds.
    SPIKE:          Sudden change exceeding threshold.
    STUCK_VALUE:    Consecutive identical readings.
    GAP:            Missing data intervals.
    ROLLOVER:       Cumulative register rollover.
    NEGATIVE:       Unexpected negative values.
    SUM_CHECK:      Main meter vs submeter sum reconciliation.
    PHASE_BALANCE:  Three-phase current imbalance.
    PF_RANGE:       Power factor within expected range.
    TIMESTAMP:      Timestamp ordering and validity.
    DUPLICATE:      Duplicate readings at same timestamp.
    COMPLETENESS:   Overall data completeness.
    """
    RANGE = "range"
    SPIKE = "spike"
    STUCK_VALUE = "stuck_value"
    GAP = "gap"
    ROLLOVER = "rollover"
    NEGATIVE = "negative"
    SUM_CHECK = "sum_check"
    PHASE_BALANCE = "phase_balance"
    PF_RANGE = "pf_range"
    TIMESTAMP = "timestamp"
    DUPLICATE = "duplicate"
    COMPLETENESS = "completeness"

class ValidationSeverity(str, Enum):
    """Severity of a validation finding.

    INFO:      Informational note, no action required.
    WARNING:   Potential issue, review recommended.
    ERROR:     Significant issue, correction needed.
    CRITICAL:  Data integrity compromised, immediate action.
    """
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class QualityGrade(str, Enum):
    """Data quality grade per ASHRAE Guideline 14 criteria.

    A_EXCELLENT: Score >= 95%, completeness >= 99%.
    B_GOOD:      Score >= 85%, completeness >= 95%.
    C_FAIR:      Score >= 70%, completeness >= 90%.
    D_POOR:      Score >= 50%, completeness >= 80%.
    F_FAIL:      Below D thresholds.
    """
    A_EXCELLENT = "A_excellent"
    B_GOOD = "B_good"
    C_FAIR = "C_fair"
    D_POOR = "D_poor"
    F_FAIL = "F_fail"

class CorrectionMethod(str, Enum):
    """Method for correcting invalid data.

    INTERPOLATE:   Linear interpolation from adjacent values.
    FORWARD_FILL:  Carry last valid value forward.
    AVERAGE:       Replace with moving average.
    MANUAL:        Flag for manual review.
    REJECT:        Reject and exclude from analysis.
    """
    INTERPOLATE = "interpolate"
    FORWARD_FILL = "forward_fill"
    AVERAGE = "average"
    MANUAL = "manual"
    REJECT = "reject"

class DataSource(str, Enum):
    """Origin of the data being validated.

    METER:  Direct meter reading.
    BMS:    Building management system.
    SCADA:  SCADA system.
    AMI:    Advanced metering infrastructure.
    IOT:    IoT sensor platform.
    MANUAL: Manual data entry.
    """
    METER = "meter"
    BMS = "bms"
    SCADA = "scada"
    AMI = "ami"
    IOT = "iot"
    MANUAL = "manual"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default check weights for quality score calculation.
CHECK_WEIGHTS: Dict[str, Decimal] = {
    ValidationCheck.RANGE.value: Decimal("15"),
    ValidationCheck.SPIKE.value: Decimal("12"),
    ValidationCheck.STUCK_VALUE.value: Decimal("8"),
    ValidationCheck.GAP.value: Decimal("15"),
    ValidationCheck.ROLLOVER.value: Decimal("5"),
    ValidationCheck.NEGATIVE.value: Decimal("10"),
    ValidationCheck.SUM_CHECK.value: Decimal("10"),
    ValidationCheck.PHASE_BALANCE.value: Decimal("5"),
    ValidationCheck.PF_RANGE.value: Decimal("5"),
    ValidationCheck.TIMESTAMP.value: Decimal("5"),
    ValidationCheck.DUPLICATE.value: Decimal("5"),
    ValidationCheck.COMPLETENESS.value: Decimal("5"),
}

# Default spike detection threshold (multiples of std dev).
DEFAULT_SPIKE_THRESHOLD: Decimal = Decimal("4.0")

# Default stuck value consecutive count threshold.
DEFAULT_STUCK_THRESHOLD: int = 6

# Default gap detection factor (gap if interval > expected * factor).
DEFAULT_GAP_FACTOR: Decimal = Decimal("1.5")

# Power factor acceptable range.
PF_MIN: Decimal = Decimal("0.70")
PF_MAX: Decimal = Decimal("1.00")

# Phase imbalance threshold percentage.
PHASE_IMBALANCE_THRESHOLD: Decimal = Decimal("10")

# Sum check tolerance percentage.
SUM_CHECK_TOLERANCE: Decimal = Decimal("5")

# Quality grade thresholds (score, completeness, grade).
QUALITY_GRADE_THRESHOLDS: List[Tuple[Decimal, Decimal, QualityGrade]] = [
    (Decimal("95"), Decimal("99"), QualityGrade.A_EXCELLENT),
    (Decimal("85"), Decimal("95"), QualityGrade.B_GOOD),
    (Decimal("70"), Decimal("90"), QualityGrade.C_FAIR),
    (Decimal("50"), Decimal("80"), QualityGrade.D_POOR),
    (Decimal("0"), Decimal("0"), QualityGrade.F_FAIL),
]

# Severity escalation thresholds (percentage of failures).
SEVERITY_THRESHOLDS: Dict[str, Decimal] = {
    "critical": Decimal("20"),
    "error": Decimal("10"),
    "warning": Decimal("5"),
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ValidationRule(BaseModel):
    """Configuration for a single validation check.

    Attributes:
        rule_id:         Unique rule identifier.
        check_type:      Type of validation check.
        enabled:         Whether this check is active.
        weight:          Weight in quality score (0-100).
        min_value:       Minimum allowed value (for range check).
        max_value:       Maximum allowed value (for range check).
        spike_threshold: Spike detection multiplier.
        stuck_threshold: Consecutive stuck count.
        gap_factor:      Gap detection factor.
        tolerance_pct:   Tolerance percentage (sum check).
        severity:        Default severity for failures.
        correction:      Default correction method.
    """
    rule_id: str = Field(default_factory=_new_uuid)
    check_type: ValidationCheck = Field(default=ValidationCheck.RANGE)
    enabled: bool = Field(default=True)
    weight: Decimal = Field(default=Decimal("10"))
    min_value: Decimal = Field(default=Decimal("0"))
    max_value: Decimal = Field(default=Decimal("999999"))
    spike_threshold: Decimal = Field(default=DEFAULT_SPIKE_THRESHOLD)
    stuck_threshold: int = Field(default=DEFAULT_STUCK_THRESHOLD, ge=2)
    gap_factor: Decimal = Field(default=DEFAULT_GAP_FACTOR)
    tolerance_pct: Decimal = Field(default=SUM_CHECK_TOLERANCE)
    severity: ValidationSeverity = Field(default=ValidationSeverity.WARNING)
    correction: CorrectionMethod = Field(default=CorrectionMethod.MANUAL)

class ValidationFinding(BaseModel):
    """A single validation finding (issue detected).

    Attributes:
        finding_id:    Unique finding identifier.
        check_type:    Which check detected this issue.
        severity:      Finding severity.
        index:         Index of the problematic reading.
        timestamp:     Timestamp of the problematic reading.
        value:         The value that failed validation.
        expected_min:  Expected minimum value.
        expected_max:  Expected maximum value.
        message:       Human-readable description.
        correction:    Recommended correction method.
        corrected_value: Value after correction (if applied).
        is_corrected:  Whether correction was applied.
    """
    finding_id: str = Field(default_factory=_new_uuid)
    check_type: ValidationCheck = Field(default=ValidationCheck.RANGE)
    severity: ValidationSeverity = Field(default=ValidationSeverity.WARNING)
    index: int = Field(default=0, ge=0)
    timestamp: Optional[datetime] = Field(default=None)
    value: Decimal = Field(default=Decimal("0"))
    expected_min: Decimal = Field(default=Decimal("0"))
    expected_max: Decimal = Field(default=Decimal("0"))
    message: str = Field(default="", max_length=1000)
    correction: CorrectionMethod = Field(default=CorrectionMethod.MANUAL)
    corrected_value: Optional[Decimal] = Field(default=None)
    is_corrected: bool = Field(default=False)

class QualityScore(BaseModel):
    """Composite data quality score.

    Attributes:
        overall_score:      Weighted quality score (0-100).
        grade:              Quality grade letter.
        completeness_pct:   Data completeness percentage.
        check_scores:       Individual check pass rates.
        check_weights:      Weights used for each check.
        total_readings:     Total readings evaluated.
        valid_readings:     Readings that passed all checks.
        invalid_readings:   Readings that failed at least one check.
        finding_count:      Total findings.
        critical_count:     Critical findings.
        error_count:        Error findings.
        warning_count:      Warning findings.
        info_count:         Info findings.
        calculated_at:      Calculation timestamp.
        provenance_hash:    SHA-256 audit hash.
    """
    overall_score: Decimal = Field(default=Decimal("0"))
    grade: QualityGrade = Field(default=QualityGrade.F_FAIL)
    completeness_pct: Decimal = Field(default=Decimal("0"))
    check_scores: Dict[str, Decimal] = Field(default_factory=dict)
    check_weights: Dict[str, Decimal] = Field(default_factory=dict)
    total_readings: int = Field(default=0, ge=0)
    valid_readings: int = Field(default=0, ge=0)
    invalid_readings: int = Field(default=0, ge=0)
    finding_count: int = Field(default=0, ge=0)
    critical_count: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)
    warning_count: int = Field(default=0, ge=0)
    info_count: int = Field(default=0, ge=0)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class DataCorrection(BaseModel):
    """Applied data correction record.

    Attributes:
        correction_id:    Unique correction identifier.
        index:            Index of corrected reading.
        original_value:   Value before correction.
        corrected_value:  Value after correction.
        method:           Correction method used.
        reason:           Reason for correction.
        confidence_pct:   Confidence in corrected value.
    """
    correction_id: str = Field(default_factory=_new_uuid)
    index: int = Field(default=0, ge=0)
    original_value: Decimal = Field(default=Decimal("0"))
    corrected_value: Decimal = Field(default=Decimal("0"))
    method: CorrectionMethod = Field(default=CorrectionMethod.INTERPOLATE)
    reason: str = Field(default="", max_length=500)
    confidence_pct: Decimal = Field(default=Decimal("100"))

class ValidationReport(BaseModel):
    """Complete validation report for a set of readings.

    Attributes:
        report_id:        Unique report identifier.
        meter_id:         Meter being validated.
        source:           Data source.
        period_start:     Validation period start.
        period_end:       Validation period end.
        quality_score:    Composite quality score.
        findings:         List of validation findings.
        corrections:      List of applied corrections.
        check_summary:    Summary by check type.
        recommendations:  List of recommendations.
        processing_time_ms: Processing duration.
        calculated_at:    Calculation timestamp.
        provenance_hash:  SHA-256 audit hash.
    """
    report_id: str = Field(default_factory=_new_uuid)
    meter_id: str = Field(default="")
    source: DataSource = Field(default=DataSource.METER)
    period_start: datetime = Field(default_factory=utcnow)
    period_end: datetime = Field(default_factory=utcnow)
    quality_score: QualityScore = Field(default_factory=QualityScore)
    findings: List[ValidationFinding] = Field(default_factory=list)
    corrections: List[DataCorrection] = Field(default_factory=list)
    check_summary: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DataValidationEngine:
    """12-check automated data quality validation engine per ASHRAE 14.

    Validates energy meter readings with range, spike, stuck-value, gap,
    rollover, negative, sum-check, phase-balance, PF-range, timestamp,
    duplicate, and completeness checks.  Produces quality scores, grades,
    and correction recommendations.

    Usage::

        engine = DataValidationEngine()
        readings = [{"timestamp": ..., "value": Decimal("1234.5")}, ...]
        report = engine.validate_readings("M-001", readings)
        print(f"Grade: {report.quality_score.grade.value}")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise DataValidationEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - spike_threshold (float): override spike detection threshold
                - stuck_threshold (int): override stuck value threshold
                - gap_factor (float): override gap detection factor
                - pf_min (float): override minimum power factor
                - pf_max (float): override maximum power factor
                - sum_check_tolerance (float): override sum check tolerance
        """
        self.config = config or {}
        self._spike_threshold = _decimal(
            self.config.get("spike_threshold", DEFAULT_SPIKE_THRESHOLD)
        )
        self._stuck_threshold = int(
            self.config.get("stuck_threshold", DEFAULT_STUCK_THRESHOLD)
        )
        self._gap_factor = _decimal(
            self.config.get("gap_factor", DEFAULT_GAP_FACTOR)
        )
        self._pf_min = _decimal(self.config.get("pf_min", PF_MIN))
        self._pf_max = _decimal(self.config.get("pf_max", PF_MAX))
        self._sum_tolerance = _decimal(
            self.config.get("sum_check_tolerance", SUM_CHECK_TOLERANCE)
        )
        self._rules: Dict[str, ValidationRule] = self._build_default_rules()
        logger.info(
            "DataValidationEngine v%s initialised (spike=%.1f, stuck=%d, "
            "gap_factor=%.1f)",
            self.engine_version, float(self._spike_threshold),
            self._stuck_threshold, float(self._gap_factor),
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def validate_readings(
        self,
        meter_id: str,
        readings: List[Dict[str, Any]],
        source: DataSource = DataSource.METER,
        expected_interval_sec: int = 900,
        submeter_values: Optional[List[Decimal]] = None,
    ) -> ValidationReport:
        """Run all 12 validation checks on a set of readings.

        Args:
            meter_id:              Meter identifier.
            readings:              List of dicts with 'timestamp' and 'value'.
            source:                Data source type.
            expected_interval_sec: Expected interval in seconds.
            submeter_values:       Submeter values for sum check.

        Returns:
            ValidationReport with findings, scores, and recommendations.
        """
        t0 = time.perf_counter()
        logger.info(
            "Validating %d readings for meter %s", len(readings), meter_id[:12]
        )

        if not readings:
            report = ValidationReport(
                meter_id=meter_id, source=source,
            )
            report.provenance_hash = _compute_hash(report)
            return report

        all_findings: List[ValidationFinding] = []

        # Extract values and timestamps
        values = [_decimal(r.get("value", 0)) for r in readings]
        timestamps = [r.get("timestamp", utcnow()) for r in readings]

        # Run each check
        check_results: Dict[str, Tuple[int, int]] = {}

        # 1. Range check
        range_findings = self.check_range(values, timestamps)
        all_findings.extend(range_findings)
        check_results[ValidationCheck.RANGE.value] = (
            len(values), len(values) - len(range_findings),
        )

        # 2. Spike check
        spike_findings = self.check_spikes(values, timestamps)
        all_findings.extend(spike_findings)
        check_results[ValidationCheck.SPIKE.value] = (
            max(len(values) - 1, 1), max(len(values) - 1, 1) - len(spike_findings),
        )

        # 3. Stuck value check
        stuck_findings = self._check_stuck_values(values, timestamps)
        all_findings.extend(stuck_findings)
        check_results[ValidationCheck.STUCK_VALUE.value] = (
            len(values), len(values) - len(stuck_findings),
        )

        # 4. Gap check
        gap_findings = self.detect_gaps(
            timestamps, expected_interval_sec,
        )
        all_findings.extend(gap_findings)
        check_results[ValidationCheck.GAP.value] = (
            max(len(timestamps) - 1, 1),
            max(len(timestamps) - 1, 1) - len(gap_findings),
        )

        # 5. Rollover check
        rollover_findings = self._check_rollover(values, timestamps)
        all_findings.extend(rollover_findings)
        check_results[ValidationCheck.ROLLOVER.value] = (
            max(len(values) - 1, 1),
            max(len(values) - 1, 1) - len(rollover_findings),
        )

        # 6. Negative value check
        neg_findings = self._check_negative(values, timestamps)
        all_findings.extend(neg_findings)
        check_results[ValidationCheck.NEGATIVE.value] = (
            len(values), len(values) - len(neg_findings),
        )

        # 7. Sum check
        sum_findings = self._check_sum(values, submeter_values)
        all_findings.extend(sum_findings)
        check_results[ValidationCheck.SUM_CHECK.value] = (
            1 if submeter_values else 0,
            0 if sum_findings else (1 if submeter_values else 0),
        )

        # 8. Phase balance check
        phase_data = [r.get("phases") for r in readings if r.get("phases")]
        phase_findings = self._check_phase_balance(phase_data, timestamps)
        all_findings.extend(phase_findings)
        check_results[ValidationCheck.PHASE_BALANCE.value] = (
            len(phase_data), len(phase_data) - len(phase_findings),
        )

        # 9. PF range check
        pf_values = [_decimal(r.get("power_factor", 0)) for r in readings if r.get("power_factor")]
        pf_findings = self._check_pf_range(pf_values, timestamps)
        all_findings.extend(pf_findings)
        check_results[ValidationCheck.PF_RANGE.value] = (
            len(pf_values), len(pf_values) - len(pf_findings),
        )

        # 10. Timestamp check
        ts_findings = self._check_timestamps(timestamps)
        all_findings.extend(ts_findings)
        check_results[ValidationCheck.TIMESTAMP.value] = (
            len(timestamps), len(timestamps) - len(ts_findings),
        )

        # 11. Duplicate check
        dup_findings = self._check_duplicates(timestamps, values)
        all_findings.extend(dup_findings)
        check_results[ValidationCheck.DUPLICATE.value] = (
            len(timestamps), len(timestamps) - len(dup_findings),
        )

        # 12. Completeness check
        comp_findings, completeness = self._check_completeness(
            timestamps, expected_interval_sec,
        )
        all_findings.extend(comp_findings)
        check_results[ValidationCheck.COMPLETENESS.value] = (1, 0 if comp_findings else 1)

        # Score quality
        quality_score = self.score_quality(
            check_results, all_findings, len(readings), completeness,
        )

        # Build check summary
        check_summary = self._build_check_summary(check_results, all_findings)

        # Recommendations
        recommendations = self._generate_recommendations(
            quality_score, all_findings, check_results,
        )

        period_start = min(timestamps) if timestamps else utcnow()
        period_end = max(timestamps) if timestamps else utcnow()
        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000.0)

        report = ValidationReport(
            meter_id=meter_id,
            source=source,
            period_start=period_start,
            period_end=period_end,
            quality_score=quality_score,
            findings=all_findings[:500],
            check_summary=check_summary,
            recommendations=recommendations,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        report.provenance_hash = _compute_hash(report)

        logger.info(
            "Validation complete: meter=%s, grade=%s, score=%.1f, "
            "findings=%d, hash=%s (%.1f ms)",
            meter_id[:12], quality_score.grade.value,
            float(quality_score.overall_score), len(all_findings),
            report.provenance_hash[:16], float(elapsed_ms),
        )
        return report

    def check_range(
        self,
        values: List[Decimal],
        timestamps: List[Any],
        min_val: Optional[Decimal] = None,
        max_val: Optional[Decimal] = None,
    ) -> List[ValidationFinding]:
        """Check if values fall within expected range.

        Args:
            values:     List of values to check.
            timestamps: Corresponding timestamps.
            min_val:    Minimum allowed value override.
            max_val:    Maximum allowed value override.

        Returns:
            List of findings for out-of-range values.
        """
        rule = self._rules.get(ValidationCheck.RANGE.value)
        lo = min_val if min_val is not None else (rule.min_value if rule else Decimal("0"))
        hi = max_val if max_val is not None else (rule.max_value if rule else Decimal("999999"))

        findings: List[ValidationFinding] = []
        for i, v in enumerate(values):
            if v < lo or v > hi:
                ts = timestamps[i] if i < len(timestamps) else None
                findings.append(ValidationFinding(
                    check_type=ValidationCheck.RANGE,
                    severity=ValidationSeverity.ERROR,
                    index=i,
                    timestamp=ts,
                    value=v,
                    expected_min=lo,
                    expected_max=hi,
                    message=f"Value {v} outside range [{lo}, {hi}]",
                ))

        return findings

    def check_spikes(
        self,
        values: List[Decimal],
        timestamps: List[Any],
    ) -> List[ValidationFinding]:
        """Detect sudden spikes using z-score of inter-reading deltas.

        A spike is flagged when |value_n - value_n-1| > threshold * std_dev.

        Args:
            values:     List of values.
            timestamps: Corresponding timestamps.

        Returns:
            List of findings for detected spikes.
        """
        if len(values) < 3:
            return []

        deltas = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
        mean_delta = _safe_divide(
            sum(deltas, Decimal("0")), _decimal(len(deltas))
        )
        variance = _safe_divide(
            sum(((d - mean_delta) ** 2 for d in deltas), Decimal("0")),
            _decimal(len(deltas)),
        )
        std_dev = _decimal(math.sqrt(float(variance)))

        findings: List[ValidationFinding] = []
        if std_dev == Decimal("0"):
            return findings

        for i, delta in enumerate(deltas):
            z = _safe_divide(delta, std_dev)
            if z > self._spike_threshold:
                idx = i + 1
                ts = timestamps[idx] if idx < len(timestamps) else None
                findings.append(ValidationFinding(
                    check_type=ValidationCheck.SPIKE,
                    severity=ValidationSeverity.WARNING,
                    index=idx,
                    timestamp=ts,
                    value=values[idx],
                    expected_min=values[i] - self._spike_threshold * std_dev,
                    expected_max=values[i] + self._spike_threshold * std_dev,
                    message=f"Spike detected: delta={delta}, z-score={_round_val(z, 2)}",
                ))

        return findings

    def detect_gaps(
        self,
        timestamps: List[Any],
        expected_interval_sec: int = 900,
    ) -> List[ValidationFinding]:
        """Detect data gaps in a timestamp sequence.

        A gap is flagged when the time delta between consecutive readings
        exceeds expected_interval * gap_factor.

        Args:
            timestamps:            List of timestamps.
            expected_interval_sec: Expected interval in seconds.

        Returns:
            List of findings for detected gaps.
        """
        if len(timestamps) < 2:
            return []

        threshold_sec = float(self._gap_factor) * expected_interval_sec
        findings: List[ValidationFinding] = []

        for i in range(1, len(timestamps)):
            ts_curr = timestamps[i]
            ts_prev = timestamps[i - 1]
            if isinstance(ts_curr, datetime) and isinstance(ts_prev, datetime):
                delta_sec = abs((ts_curr - ts_prev).total_seconds())
            else:
                continue

            if delta_sec > threshold_sec:
                gap_intervals = int(delta_sec / expected_interval_sec) - 1
                findings.append(ValidationFinding(
                    check_type=ValidationCheck.GAP,
                    severity=ValidationSeverity.ERROR if gap_intervals > 4 else ValidationSeverity.WARNING,
                    index=i,
                    timestamp=ts_curr,
                    value=_decimal(delta_sec),
                    expected_min=Decimal("0"),
                    expected_max=_decimal(threshold_sec),
                    message=(
                        f"Gap detected: {gap_intervals} missing interval(s), "
                        f"delta={delta_sec:.0f}s vs expected {expected_interval_sec}s"
                    ),
                ))

        return findings

    def score_quality(
        self,
        check_results: Dict[str, Tuple[int, int]],
        findings: List[ValidationFinding],
        total_readings: int,
        completeness_pct: Decimal,
    ) -> QualityScore:
        """Compute composite quality score from check results.

        score = SUM(check_weight * check_pass_rate) / SUM(check_weight)

        Args:
            check_results:   Dict of check -> (total, passed).
            findings:        All findings.
            total_readings:  Total readings evaluated.
            completeness_pct: Data completeness percentage.

        Returns:
            QualityScore with weighted score and grade.
        """
        t0 = time.perf_counter()

        weighted_sum = Decimal("0")
        weight_total = Decimal("0")
        check_scores: Dict[str, Decimal] = {}

        for check_name, (total, passed) in check_results.items():
            weight = CHECK_WEIGHTS.get(check_name, Decimal("5"))
            if total > 0:
                pass_rate = _safe_pct(_decimal(passed), _decimal(total))
            else:
                pass_rate = Decimal("100")
            check_scores[check_name] = _round_val(pass_rate, 2)
            weighted_sum += weight * pass_rate
            weight_total += weight

        overall = _safe_divide(weighted_sum, weight_total)

        # Determine grade
        grade = QualityGrade.F_FAIL
        for score_thresh, comp_thresh, g in QUALITY_GRADE_THRESHOLDS:
            if overall >= score_thresh and completeness_pct >= comp_thresh:
                grade = g
                break

        # Count findings by severity
        critical = sum(1 for f in findings if f.severity == ValidationSeverity.CRITICAL)
        error = sum(1 for f in findings if f.severity == ValidationSeverity.ERROR)
        warning = sum(1 for f in findings if f.severity == ValidationSeverity.WARNING)
        info = sum(1 for f in findings if f.severity == ValidationSeverity.INFO)

        # Count invalid readings (unique indices with findings)
        invalid_indices = set(f.index for f in findings)
        invalid_count = len(invalid_indices)
        valid_count = max(total_readings - invalid_count, 0)

        result = QualityScore(
            overall_score=_round_val(overall, 2),
            grade=grade,
            completeness_pct=_round_val(completeness_pct, 2),
            check_scores=check_scores,
            check_weights={k: str(v) for k, v in CHECK_WEIGHTS.items()},
            total_readings=total_readings,
            valid_readings=valid_count,
            invalid_readings=invalid_count,
            finding_count=len(findings),
            critical_count=critical,
            error_count=error,
            warning_count=warning,
            info_count=info,
        )
        result.provenance_hash = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Quality scored: %.1f%% (%s), findings=%d (C=%d/E=%d/W=%d) (%.1f ms)",
            float(overall), grade.value, len(findings),
            critical, error, warning, elapsed,
        )
        return result

    def apply_corrections(
        self,
        values: List[Decimal],
        findings: List[ValidationFinding],
        method: CorrectionMethod = CorrectionMethod.INTERPOLATE,
    ) -> Tuple[List[Decimal], List[DataCorrection]]:
        """Apply corrections to values based on validation findings.

        Args:
            values:   Original values.
            findings: Findings indicating which values to correct.
            method:   Correction method to apply.

        Returns:
            Tuple of (corrected_values, correction_records).
        """
        t0 = time.perf_counter()
        logger.info(
            "Applying corrections: %d findings, method=%s",
            len(findings), method.value,
        )

        corrected = list(values)
        records: List[DataCorrection] = []
        invalid_indices = {f.index for f in findings if f.severity in (
            ValidationSeverity.ERROR, ValidationSeverity.CRITICAL,
        )}

        for idx in sorted(invalid_indices):
            if idx >= len(corrected):
                continue

            original = corrected[idx]
            new_val = original
            confidence = Decimal("0")

            if method == CorrectionMethod.INTERPOLATE:
                new_val, confidence = self._interpolate_correction(
                    corrected, idx,
                )
            elif method == CorrectionMethod.FORWARD_FILL:
                new_val, confidence = self._forward_fill_correction(
                    corrected, idx,
                )
            elif method == CorrectionMethod.AVERAGE:
                new_val, confidence = self._average_correction(
                    corrected, idx,
                )
            elif method == CorrectionMethod.REJECT:
                new_val = Decimal("0")
                confidence = Decimal("0")

            corrected[idx] = new_val
            records.append(DataCorrection(
                index=idx,
                original_value=original,
                corrected_value=new_val,
                method=method,
                reason=f"Auto-correction at index {idx}",
                confidence_pct=confidence,
            ))

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Corrections applied: %d corrections (%.1f ms)",
            len(records), elapsed,
        )
        return corrected, records

    # ------------------------------------------------------------------ #
    # Internal Methods                                                     #
    # ------------------------------------------------------------------ #

    def _build_default_rules(self) -> Dict[str, ValidationRule]:
        """Build default validation rules for all 12 checks.

        Returns:
            Dict of check_type -> ValidationRule.
        """
        rules: Dict[str, ValidationRule] = {}
        for check in ValidationCheck:
            weight = CHECK_WEIGHTS.get(check.value, Decimal("5"))
            rules[check.value] = ValidationRule(
                check_type=check,
                weight=weight,
                spike_threshold=self._spike_threshold,
                stuck_threshold=self._stuck_threshold,
                gap_factor=self._gap_factor,
                tolerance_pct=self._sum_tolerance,
            )
        return rules

    def _check_stuck_values(
        self, values: List[Decimal], timestamps: List[Any],
    ) -> List[ValidationFinding]:
        """Check for consecutive identical values (stuck sensor)."""
        findings: List[ValidationFinding] = []
        if len(values) < self._stuck_threshold:
            return findings

        consecutive = 1
        for i in range(1, len(values)):
            if values[i] == values[i - 1]:
                consecutive += 1
                if consecutive >= self._stuck_threshold:
                    ts = timestamps[i] if i < len(timestamps) else None
                    findings.append(ValidationFinding(
                        check_type=ValidationCheck.STUCK_VALUE,
                        severity=ValidationSeverity.WARNING,
                        index=i,
                        timestamp=ts,
                        value=values[i],
                        message=f"Stuck value: {values[i]} repeated {consecutive} times",
                    ))
            else:
                consecutive = 1

        return findings

    def _check_rollover(
        self, values: List[Decimal], timestamps: List[Any],
    ) -> List[ValidationFinding]:
        """Check for potential cumulative register rollovers."""
        findings: List[ValidationFinding] = []
        for i in range(1, len(values)):
            if values[i] < values[i - 1] and values[i - 1] > Decimal("0"):
                ts = timestamps[i] if i < len(timestamps) else None
                findings.append(ValidationFinding(
                    check_type=ValidationCheck.ROLLOVER,
                    severity=ValidationSeverity.INFO,
                    index=i,
                    timestamp=ts,
                    value=values[i],
                    expected_min=values[i - 1],
                    message=f"Possible rollover: {values[i-1]} -> {values[i]}",
                ))
        return findings

    def _check_negative(
        self, values: List[Decimal], timestamps: List[Any],
    ) -> List[ValidationFinding]:
        """Check for unexpected negative values."""
        findings: List[ValidationFinding] = []
        for i, v in enumerate(values):
            if v < Decimal("0"):
                ts = timestamps[i] if i < len(timestamps) else None
                findings.append(ValidationFinding(
                    check_type=ValidationCheck.NEGATIVE,
                    severity=ValidationSeverity.ERROR,
                    index=i,
                    timestamp=ts,
                    value=v,
                    expected_min=Decimal("0"),
                    message=f"Negative value: {v}",
                ))
        return findings

    def _check_sum(
        self,
        main_values: List[Decimal],
        submeter_values: Optional[List[Decimal]],
    ) -> List[ValidationFinding]:
        """Check main meter vs submeter sum reconciliation."""
        if not submeter_values:
            return []

        main_total = sum(main_values, Decimal("0"))
        sub_total = sum(submeter_values, Decimal("0"))

        if main_total == Decimal("0"):
            return []

        diff_pct = abs(_safe_pct(main_total - sub_total, main_total))
        if diff_pct > self._sum_tolerance:
            return [ValidationFinding(
                check_type=ValidationCheck.SUM_CHECK,
                severity=ValidationSeverity.ERROR,
                index=0,
                value=diff_pct,
                expected_max=self._sum_tolerance,
                message=(
                    f"Sum check failed: main={main_total}, "
                    f"submeters={sub_total}, diff={_round_val(diff_pct, 2)}%"
                ),
            )]
        return []

    def _check_phase_balance(
        self, phase_data: List[Any], timestamps: List[Any],
    ) -> List[ValidationFinding]:
        """Check three-phase current balance."""
        findings: List[ValidationFinding] = []
        for i, phases in enumerate(phase_data):
            if not isinstance(phases, (list, tuple)) or len(phases) < 3:
                continue
            p = [_decimal(x) for x in phases[:3]]
            avg = _safe_divide(sum(p, Decimal("0")), Decimal("3"))
            if avg == Decimal("0"):
                continue
            max_p = max(p)
            min_p = min(p)
            imbalance = _safe_pct(max_p - min_p, avg)
            if imbalance > PHASE_IMBALANCE_THRESHOLD:
                ts = timestamps[i] if i < len(timestamps) else None
                findings.append(ValidationFinding(
                    check_type=ValidationCheck.PHASE_BALANCE,
                    severity=ValidationSeverity.WARNING,
                    index=i,
                    timestamp=ts,
                    value=imbalance,
                    expected_max=PHASE_IMBALANCE_THRESHOLD,
                    message=f"Phase imbalance: {_round_val(imbalance, 2)}%",
                ))
        return findings

    def _check_pf_range(
        self, pf_values: List[Decimal], timestamps: List[Any],
    ) -> List[ValidationFinding]:
        """Check power factor values within acceptable range."""
        findings: List[ValidationFinding] = []
        for i, pf in enumerate(pf_values):
            if pf < self._pf_min or pf > self._pf_max:
                ts = timestamps[i] if i < len(timestamps) else None
                findings.append(ValidationFinding(
                    check_type=ValidationCheck.PF_RANGE,
                    severity=ValidationSeverity.WARNING,
                    index=i,
                    timestamp=ts,
                    value=pf,
                    expected_min=self._pf_min,
                    expected_max=self._pf_max,
                    message=f"PF out of range: {pf}",
                ))
        return findings

    def _check_timestamps(
        self, timestamps: List[Any],
    ) -> List[ValidationFinding]:
        """Check timestamp ordering and validity."""
        findings: List[ValidationFinding] = []
        for i in range(1, len(timestamps)):
            ts_curr = timestamps[i]
            ts_prev = timestamps[i - 1]
            if isinstance(ts_curr, datetime) and isinstance(ts_prev, datetime):
                if ts_curr <= ts_prev:
                    findings.append(ValidationFinding(
                        check_type=ValidationCheck.TIMESTAMP,
                        severity=ValidationSeverity.ERROR,
                        index=i,
                        timestamp=ts_curr,
                        value=Decimal("0"),
                        message=f"Timestamp out of order: {ts_curr} <= {ts_prev}",
                    ))
        return findings

    def _check_duplicates(
        self, timestamps: List[Any], values: List[Decimal],
    ) -> List[ValidationFinding]:
        """Check for duplicate timestamp-value pairs."""
        findings: List[ValidationFinding] = []
        seen: Dict[str, int] = {}
        for i in range(len(timestamps)):
            key = f"{timestamps[i]}:{values[i]}"
            if key in seen:
                findings.append(ValidationFinding(
                    check_type=ValidationCheck.DUPLICATE,
                    severity=ValidationSeverity.WARNING,
                    index=i,
                    timestamp=timestamps[i] if isinstance(timestamps[i], datetime) else None,
                    value=values[i],
                    message=f"Duplicate of reading at index {seen[key]}",
                ))
            else:
                seen[key] = i
        return findings

    def _check_completeness(
        self, timestamps: List[Any], expected_interval_sec: int,
    ) -> Tuple[List[ValidationFinding], Decimal]:
        """Check data completeness."""
        if len(timestamps) < 2:
            return [], Decimal("100") if timestamps else Decimal("0")

        valid_ts = [t for t in timestamps if isinstance(t, datetime)]
        if len(valid_ts) < 2:
            return [], Decimal("100")

        period_sec = (max(valid_ts) - min(valid_ts)).total_seconds()
        expected_count = max(int(period_sec / expected_interval_sec) + 1, 1)
        actual_count = len(valid_ts)
        completeness = _safe_pct(_decimal(actual_count), _decimal(expected_count))
        completeness = min(completeness, Decimal("100"))

        findings: List[ValidationFinding] = []
        if completeness < Decimal("90"):
            findings.append(ValidationFinding(
                check_type=ValidationCheck.COMPLETENESS,
                severity=ValidationSeverity.ERROR if completeness < Decimal("80") else ValidationSeverity.WARNING,
                index=0,
                value=completeness,
                expected_min=Decimal("90"),
                message=f"Data completeness: {_round_val(completeness, 2)}%",
            ))

        return findings, completeness

    def _build_check_summary(
        self,
        check_results: Dict[str, Tuple[int, int]],
        findings: List[ValidationFinding],
    ) -> Dict[str, Dict[str, Any]]:
        """Build a per-check summary."""
        summary: Dict[str, Dict[str, Any]] = {}
        finding_counts: Dict[str, int] = {}
        for f in findings:
            ct = f.check_type.value
            finding_counts[ct] = finding_counts.get(ct, 0) + 1

        for check_name, (total, passed) in check_results.items():
            pass_rate = _safe_pct(_decimal(passed), _decimal(total)) if total > 0 else Decimal("100")
            summary[check_name] = {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate_pct": str(_round_val(pass_rate, 2)),
                "findings": finding_counts.get(check_name, 0),
            }
        return summary

    def _interpolate_correction(
        self, values: List[Decimal], idx: int,
    ) -> Tuple[Decimal, Decimal]:
        """Interpolate correction from adjacent valid values."""
        before = values[idx - 1] if idx > 0 else None
        after = values[idx + 1] if idx < len(values) - 1 else None
        if before is not None and after is not None:
            return _round_val((before + after) / Decimal("2"), 4), Decimal("85")
        if before is not None:
            return before, Decimal("70")
        if after is not None:
            return after, Decimal("70")
        return Decimal("0"), Decimal("0")

    def _forward_fill_correction(
        self, values: List[Decimal], idx: int,
    ) -> Tuple[Decimal, Decimal]:
        """Forward fill correction from last valid value."""
        for i in range(idx - 1, -1, -1):
            return values[i], Decimal("70")
        return Decimal("0"), Decimal("0")

    def _average_correction(
        self, values: List[Decimal], idx: int,
    ) -> Tuple[Decimal, Decimal]:
        """Moving average correction from surrounding values."""
        window = []
        for offset in [-3, -2, -1, 1, 2, 3]:
            adj = idx + offset
            if 0 <= adj < len(values) and adj not in {idx}:
                window.append(values[adj])
        if window:
            avg = _safe_divide(sum(window, Decimal("0")), _decimal(len(window)))
            return _round_val(avg, 4), Decimal("75")
        return Decimal("0"), Decimal("0")

    def _generate_recommendations(
        self,
        quality: QualityScore,
        findings: List[ValidationFinding],
        check_results: Dict[str, Tuple[int, int]],
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recs: List[str] = []

        if quality.grade == QualityGrade.F_FAIL:
            recs.append(
                "Data quality is failing. Immediate investigation of meter "
                "hardware and communication is required."
            )

        if quality.critical_count > 0:
            recs.append(
                f"{quality.critical_count} critical finding(s) detected. "
                "Address these before using data for M&V calculations."
            )

        gap_total, gap_passed = check_results.get(ValidationCheck.GAP.value, (0, 0))
        if gap_total > 0 and gap_passed < gap_total:
            recs.append(
                "Data gaps detected. Verify meter communication stability "
                "and consider installing redundant data paths."
            )

        spike_total, spike_passed = check_results.get(ValidationCheck.SPIKE.value, (0, 0))
        if spike_total > 0 and spike_passed < spike_total * 0.95:
            recs.append(
                "Excessive spikes detected. Check for electrical transients "
                "or meter calibration issues."
            )

        if quality.completeness_pct < Decimal("95"):
            recs.append(
                f"Data completeness is {quality.completeness_pct}%. "
                "Target 99% for ASHRAE Guideline 14 compliance."
            )

        if not recs:
            recs.append(
                "Data quality meets ASHRAE Guideline 14 requirements. "
                "Continue routine monitoring."
            )

        return recs
