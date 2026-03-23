# -*- coding: utf-8 -*-
"""
PersistenceEngine - PACK-040 M&V Engine 9
============================================

Long-term energy savings persistence tracking and degradation analysis.
Monitors year-over-year savings, calculates degradation rates (linear,
exponential, step-change), triggers re-commissioning when persistence
factor drops below thresholds, and supports ESCO/EPC performance
guarantee verification.

Calculation Methodology:
    Persistence Factor:
        PF_n = actual_savings_year_n / expected_savings_year_n
        where expected_savings = Year 1 verified savings (baseline)

    Linear Degradation:
        S(t) = S0 * (1 - d * t)
        where d = annual degradation rate, t = years
        d = (S1 - Sn) / (S1 * (n - 1))

    Exponential Decay:
        S(t) = S0 * exp(-lambda * t)
        lambda = -ln(Sn / S0) / t

    Step-Change Detection:
        |S(t) - S(t-1)| / S(t-1) > step_threshold
        flags sudden performance drop / improvement

    Re-Commissioning Trigger:
        Trigger when PF < 0.80 (80% of expected savings)
        or degradation rate > configurable threshold

    Performance Guarantee:
        guarantee_met = actual_cumulative >= guaranteed_cumulative
        shortfall = max(0, guaranteed - actual)
        penalty = shortfall * penalty_rate

    Seasonal Pattern Analysis:
        Decompose savings into seasonal indices
        Identify seasonal anomalies vs. expected pattern

Regulatory References:
    - IPMVP Core Concepts 2022 - Persistence of savings
    - ASHRAE Guideline 14-2014 - Long-term savings tracking
    - ISO 50015:2014 - Savings persistence monitoring
    - FEMP M&V Guidelines 4.0 - Multi-year savings verification
    - EU EPC Directive 2012/27/EU Art. 18 - Performance contracts
    - ISO 50001:2018 Cl. 9.1 - Continual improvement tracking

Zero-Hallucination:
    - Persistence factor via deterministic ratio calculation
    - Degradation models use standard decay formulae
    - Step-change detection via threshold comparison (no ML)
    - Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-040 M&V
Engine:  9 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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


class DegradationModel(str, Enum):
    """Savings degradation model type.

    LINEAR:        Linear decay: S(t) = S0 * (1 - d*t).
    EXPONENTIAL:   Exponential decay: S(t) = S0 * exp(-lambda*t).
    STEP_CHANGE:   Discrete step-change events.
    NONE:          No degradation model applied.
    """
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP_CHANGE = "step_change"
    NONE = "none"


class PersistenceStatus(str, Enum):
    """Overall persistence status assessment.

    EXCELLENT:    PF >= 0.95 (savings holding strong).
    GOOD:         PF 0.85-0.95 (minor degradation).
    ACCEPTABLE:   PF 0.80-0.85 (within tolerance).
    DEGRADED:     PF 0.60-0.80 (significant degradation).
    CRITICAL:     PF < 0.60 (re-commissioning required).
    """
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class AlertLevel(str, Enum):
    """Alert level for persistence issues.

    INFO:       Informational (normal operation).
    WARNING:    Early degradation detected.
    ALERT:      Persistence below threshold.
    CRITICAL:   Immediate action required.
    """
    INFO = "info"
    WARNING = "warning"
    ALERT = "alert"
    CRITICAL = "critical"


class ContractType(str, Enum):
    """Performance contract type.

    ESPC:    Energy Savings Performance Contract (FEMP).
    EPC:     Energy Performance Contract (EU).
    SHARED:  Shared savings contract.
    GUARANTEED: Guaranteed savings contract.
    CUSTOM:  Custom contract structure.
    """
    ESPC = "espc"
    EPC = "epc"
    SHARED = "shared"
    GUARANTEED = "guaranteed"
    CUSTOM = "custom"


class SeasonType(str, Enum):
    """Season classification for seasonal analysis.

    WINTER:    December-February (Northern Hemisphere).
    SPRING:    March-May.
    SUMMER:    June-August.
    AUTUMN:    September-November.
    ANNUAL:    Full year.
    """
    WINTER = "winter"
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    ANNUAL = "annual"


class TrendDirection(str, Enum):
    """Savings trend direction.

    IMPROVING:   Savings increasing over time.
    STABLE:      Savings holding steady.
    DECLINING:   Savings decreasing over time.
    VOLATILE:    No clear trend; high variability.
    """
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default persistence trigger thresholds
DEFAULT_RECOMISSION_THRESHOLD = Decimal("0.80")  # 80%
DEFAULT_WARNING_THRESHOLD = Decimal("0.90")      # 90%
DEFAULT_CRITICAL_THRESHOLD = Decimal("0.60")     # 60%

# Default penalty rate ($/kWh shortfall) for guarantee contracts
DEFAULT_PENALTY_RATE = Decimal("0.12")


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class AnnualSavingsRecord(BaseModel):
    """Savings record for a single year."""

    record_id: str = Field(default_factory=_new_uuid, description="Record ID")
    year: int = Field(..., ge=1, description="Year number (1 = first year)")
    period_label: str = Field(default="", description="Period label (e.g. '2025')")
    expected_savings_kwh: Decimal = Field(
        default=Decimal("0"), description="Expected savings (kWh)"
    )
    actual_savings_kwh: Decimal = Field(
        default=Decimal("0"), description="Actual verified savings (kWh)"
    )
    expected_savings_cost: Decimal = Field(
        default=Decimal("0"), description="Expected cost savings ($)"
    )
    actual_savings_cost: Decimal = Field(
        default=Decimal("0"), description="Actual cost savings ($)"
    )
    persistence_factor: Decimal = Field(
        default=Decimal("0"), description="PF = actual / expected"
    )
    status: PersistenceStatus = Field(
        default=PersistenceStatus.GOOD, description="Persistence status"
    )
    notes: str = Field(default="", description="Notes")


class DegradationResult(BaseModel):
    """Degradation analysis result."""

    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    model_type: DegradationModel = Field(
        default=DegradationModel.LINEAR, description="Degradation model used"
    )
    annual_degradation_rate: Decimal = Field(
        default=Decimal("0"), description="Annual degradation rate (fraction)"
    )
    annual_degradation_pct: Decimal = Field(
        default=Decimal("0"), description="Annual degradation rate (%)"
    )
    lambda_decay: Decimal = Field(
        default=Decimal("0"), description="Exponential decay constant"
    )
    half_life_years: Decimal = Field(
        default=Decimal("0"), description="Half-life in years (exponential model)"
    )
    predicted_savings: List[Decimal] = Field(
        default_factory=list, description="Model-predicted savings per year"
    )
    residuals: List[Decimal] = Field(
        default_factory=list, description="Actual - predicted per year"
    )
    r_squared: Decimal = Field(default=Decimal("0"), description="Model fit R-squared")
    years_to_80pct: Decimal = Field(
        default=Decimal("0"), description="Years until savings drop to 80%"
    )
    step_changes: List[int] = Field(
        default_factory=list, description="Year indices with step-changes"
    )
    calculated_at: datetime = Field(default_factory=_utcnow)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    provenance_hash: str = Field(default="")


class PersistenceAlert(BaseModel):
    """Alert generated by persistence monitoring."""

    alert_id: str = Field(default_factory=_new_uuid, description="Alert ID")
    year: int = Field(default=0, description="Year that triggered the alert")
    level: AlertLevel = Field(default=AlertLevel.INFO, description="Alert severity")
    message: str = Field(default="", description="Alert message")
    persistence_factor: Decimal = Field(
        default=Decimal("0"), description="PF at time of alert"
    )
    threshold: Decimal = Field(
        default=Decimal("0"), description="Threshold that was breached"
    )
    recommended_action: str = Field(
        default="", description="Recommended corrective action"
    )
    created_at: datetime = Field(default_factory=_utcnow)


class GuaranteeTrackingResult(BaseModel):
    """Performance guarantee tracking result."""

    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    contract_type: ContractType = Field(
        default=ContractType.GUARANTEED, description="Contract type"
    )
    contract_term_years: int = Field(default=10, description="Contract term (years)")
    guaranteed_annual_kwh: Decimal = Field(
        default=Decimal("0"), description="Guaranteed annual savings (kWh)"
    )
    guaranteed_annual_cost: Decimal = Field(
        default=Decimal("0"), description="Guaranteed annual cost savings ($)"
    )
    cumulative_guaranteed_kwh: Decimal = Field(
        default=Decimal("0"), description="Cumulative guaranteed savings to date"
    )
    cumulative_actual_kwh: Decimal = Field(
        default=Decimal("0"), description="Cumulative actual savings to date"
    )
    cumulative_surplus_kwh: Decimal = Field(
        default=Decimal("0"), description="Surplus (positive) or shortfall (negative)"
    )
    guarantee_met: bool = Field(
        default=True, description="Whether cumulative guarantee is met"
    )
    shortfall_kwh: Decimal = Field(
        default=Decimal("0"), description="Shortfall if guarantee not met"
    )
    shortfall_cost: Decimal = Field(
        default=Decimal("0"), description="Financial value of shortfall"
    )
    penalty_amount: Decimal = Field(
        default=Decimal("0"), description="Penalty amount if applicable"
    )
    years_tracked: int = Field(default=0, description="Number of years tracked")
    annual_guarantee_status: List[bool] = Field(
        default_factory=list, description="Met/not-met per year"
    )
    calculated_at: datetime = Field(default_factory=_utcnow)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    provenance_hash: str = Field(default="")


class SeasonalPattern(BaseModel):
    """Seasonal savings pattern for a single season."""

    season: SeasonType = Field(..., description="Season")
    average_savings_kwh: Decimal = Field(
        default=Decimal("0"), description="Average savings for season"
    )
    seasonal_index: Decimal = Field(
        default=Decimal("1"), description="Seasonal index (1.0 = average)"
    )
    trend: TrendDirection = Field(
        default=TrendDirection.STABLE, description="Trend within season"
    )
    year_values: List[Decimal] = Field(
        default_factory=list, description="Savings for this season across years"
    )


class SeasonalAnalysisResult(BaseModel):
    """Seasonal savings pattern analysis."""

    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    patterns: List[SeasonalPattern] = Field(
        default_factory=list, description="Seasonal patterns"
    )
    dominant_season: SeasonType = Field(
        default=SeasonType.ANNUAL, description="Season with highest savings"
    )
    seasonal_variation_pct: Decimal = Field(
        default=Decimal("0"), description="Variation across seasons (%)"
    )
    anomalies: List[str] = Field(
        default_factory=list, description="Detected anomalies"
    )
    calculated_at: datetime = Field(default_factory=_utcnow)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    provenance_hash: str = Field(default="")


class PersistenceResult(BaseModel):
    """Comprehensive persistence analysis result."""

    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    project_id: str = Field(default="", description="M&V project reference")
    ecm_id: str = Field(default="", description="ECM reference")
    annual_records: List[AnnualSavingsRecord] = Field(
        default_factory=list, description="Year-by-year savings records"
    )
    overall_persistence_factor: Decimal = Field(
        default=Decimal("0"), description="Overall PF (latest year)"
    )
    overall_status: PersistenceStatus = Field(
        default=PersistenceStatus.GOOD, description="Overall persistence status"
    )
    degradation: Optional[DegradationResult] = Field(
        None, description="Degradation analysis"
    )
    trend: TrendDirection = Field(
        default=TrendDirection.STABLE, description="Overall trend"
    )
    alerts: List[PersistenceAlert] = Field(
        default_factory=list, description="Generated alerts"
    )
    recommissioning_recommended: bool = Field(
        default=False, description="Whether re-commissioning is recommended"
    )
    total_expected_kwh: Decimal = Field(
        default=Decimal("0"), description="Total expected savings (all years)"
    )
    total_actual_kwh: Decimal = Field(
        default=Decimal("0"), description="Total actual savings (all years)"
    )
    total_lost_kwh: Decimal = Field(
        default=Decimal("0"), description="Total lost savings (expected - actual)"
    )
    calculated_at: datetime = Field(default_factory=_utcnow)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    provenance_hash: str = Field(default="")


class YearOverYearComparison(BaseModel):
    """Year-over-year savings comparison."""

    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    year_a: int = Field(default=0, description="First year")
    year_b: int = Field(default=0, description="Second year")
    savings_a_kwh: Decimal = Field(default=Decimal("0"), description="Year A savings")
    savings_b_kwh: Decimal = Field(default=Decimal("0"), description="Year B savings")
    change_kwh: Decimal = Field(default=Decimal("0"), description="Absolute change")
    change_pct: Decimal = Field(default=Decimal("0"), description="Percentage change")
    direction: TrendDirection = Field(
        default=TrendDirection.STABLE, description="Direction of change"
    )
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine Class
# ---------------------------------------------------------------------------


class PersistenceEngine:
    """Multi-year savings persistence tracking and degradation analysis.

    Monitors year-over-year savings, calculates persistence factors,
    fits degradation models, generates alerts, tracks performance
    guarantees, and analyses seasonal savings patterns.

    All calculations are deterministic (zero-hallucination) using
    Decimal arithmetic and SHA-256 provenance hashing.

    Attributes:
        _module_version: Engine version string.
        _recommission_threshold: PF below which re-commissioning is triggered.
        _warning_threshold: PF below which a warning is issued.
        _critical_threshold: PF below which a critical alert is issued.

    Example:
        >>> engine = PersistenceEngine()
        >>> records = [
        ...     AnnualSavingsRecord(year=1, expected_savings_kwh=100000,
        ...                         actual_savings_kwh=98000),
        ...     AnnualSavingsRecord(year=2, expected_savings_kwh=100000,
        ...                         actual_savings_kwh=92000),
        ... ]
        >>> result = engine.analyse_persistence("proj-1", records)
        >>> assert result.overall_status != PersistenceStatus.CRITICAL
    """

    def __init__(
        self,
        recommission_threshold: Decimal = DEFAULT_RECOMISSION_THRESHOLD,
        warning_threshold: Decimal = DEFAULT_WARNING_THRESHOLD,
        critical_threshold: Decimal = DEFAULT_CRITICAL_THRESHOLD,
    ) -> None:
        """Initialise the PersistenceEngine.

        Args:
            recommission_threshold: PF threshold for re-commissioning.
            warning_threshold: PF threshold for warning alerts.
            critical_threshold: PF threshold for critical alerts.
        """
        self._module_version: str = _MODULE_VERSION
        self._recommission_threshold = recommission_threshold
        self._warning_threshold = warning_threshold
        self._critical_threshold = critical_threshold
        logger.info(
            "PersistenceEngine v%s initialised (thresholds: warn=%.2f, "
            "recommission=%.2f, critical=%.2f)",
            self._module_version,
            float(warning_threshold),
            float(recommission_threshold),
            float(critical_threshold),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse_persistence(
        self,
        project_id: str,
        annual_records: List[AnnualSavingsRecord],
        ecm_id: str = "",
        degradation_model: DegradationModel = DegradationModel.LINEAR,
    ) -> PersistenceResult:
        """Perform comprehensive persistence analysis.

        Computes persistence factors, fits degradation model, generates
        alerts, and determines re-commissioning need.

        Args:
            project_id: M&V project identifier.
            annual_records: Year-by-year savings records.
            ecm_id: ECM identifier.
            degradation_model: Degradation model to fit.

        Returns:
            PersistenceResult with full analysis.
        """
        t0 = time.perf_counter()
        logger.info(
            "Analysing persistence: project=%s, %d years, model=%s",
            project_id, len(annual_records), degradation_model.value,
        )

        # Compute persistence factors for each year
        records = self._compute_persistence_factors(annual_records)

        # Generate alerts
        alerts = self._generate_alerts(records)

        # Fit degradation model
        degradation: Optional[DegradationResult] = None
        if len(records) >= 2:
            degradation = self.fit_degradation(records, degradation_model)

        # Overall metrics
        total_expected = sum(r.expected_savings_kwh for r in records)
        total_actual = sum(r.actual_savings_kwh for r in records)
        total_lost = max(total_expected - total_actual, Decimal("0"))

        # Overall persistence factor (latest year)
        latest_pf = records[-1].persistence_factor if records else Decimal("0")
        overall_status = self._classify_status(latest_pf)

        # Trend
        trend = self._determine_trend(records)

        # Re-commissioning check
        recommission = latest_pf < self._recommission_threshold

        elapsed = (time.perf_counter() - t0) * 1000.0
        result = PersistenceResult(
            project_id=project_id,
            ecm_id=ecm_id,
            annual_records=records,
            overall_persistence_factor=_round_val(latest_pf, 4),
            overall_status=overall_status,
            degradation=degradation,
            trend=trend,
            alerts=alerts,
            recommissioning_recommended=recommission,
            total_expected_kwh=_round_val(total_expected, 2),
            total_actual_kwh=_round_val(total_actual, 2),
            total_lost_kwh=_round_val(total_lost, 2),
            processing_time_ms=_round_val(_decimal(elapsed), 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Persistence analysis: PF=%.3f, status=%s, trend=%s, "
            "recommission=%s, hash=%s (%.1f ms)",
            float(latest_pf), overall_status.value, trend.value,
            recommission, result.provenance_hash[:16], elapsed,
        )
        return result

    def fit_degradation(
        self,
        records: List[AnnualSavingsRecord],
        model_type: DegradationModel = DegradationModel.LINEAR,
        step_threshold: Decimal = Decimal("0.15"),
    ) -> DegradationResult:
        """Fit a degradation model to the savings history.

        Args:
            records: Annual savings records with actual values.
            model_type: Degradation model type.
            step_threshold: Threshold for step-change detection (fraction).

        Returns:
            DegradationResult with fitted model parameters.
        """
        t0 = time.perf_counter()
        n = len(records)
        logger.info(
            "Fitting degradation model: %s, %d years", model_type.value, n,
        )

        actual = [r.actual_savings_kwh for r in records]
        s0 = actual[0] if actual else Decimal("0")

        if model_type == DegradationModel.LINEAR:
            deg_result = self._fit_linear_degradation(actual, s0)
        elif model_type == DegradationModel.EXPONENTIAL:
            deg_result = self._fit_exponential_degradation(actual, s0)
        elif model_type == DegradationModel.STEP_CHANGE:
            deg_result = self._detect_step_changes(actual, step_threshold)
        else:
            deg_result = DegradationResult(model_type=DegradationModel.NONE)

        elapsed = (time.perf_counter() - t0) * 1000.0
        deg_result.processing_time_ms = _round_val(_decimal(elapsed), 2)
        deg_result.provenance_hash = _compute_hash(deg_result)

        logger.info(
            "Degradation fitted: model=%s, rate=%.4f%%/yr, R2=%.4f, "
            "hash=%s (%.1f ms)",
            model_type.value,
            float(deg_result.annual_degradation_pct),
            float(deg_result.r_squared),
            deg_result.provenance_hash[:16], elapsed,
        )
        return deg_result

    def track_guarantee(
        self,
        annual_records: List[AnnualSavingsRecord],
        guaranteed_annual_kwh: Decimal,
        guaranteed_annual_cost: Decimal = Decimal("0"),
        contract_type: ContractType = ContractType.GUARANTEED,
        contract_term_years: int = 10,
        penalty_rate: Decimal = DEFAULT_PENALTY_RATE,
    ) -> GuaranteeTrackingResult:
        """Track performance guarantee compliance.

        Args:
            annual_records: Year-by-year actual savings.
            guaranteed_annual_kwh: Guaranteed annual savings (kWh).
            guaranteed_annual_cost: Guaranteed annual cost savings ($).
            contract_type: Contract type.
            contract_term_years: Contract duration.
            penalty_rate: Penalty rate per kWh shortfall.

        Returns:
            GuaranteeTrackingResult with compliance status.
        """
        t0 = time.perf_counter()
        n = len(annual_records)
        logger.info(
            "Tracking guarantee: %d years, guaranteed=%.0f kWh/yr, "
            "contract=%s, term=%d yr",
            n, float(guaranteed_annual_kwh), contract_type.value,
            contract_term_years,
        )

        cumulative_guaranteed = Decimal("0")
        cumulative_actual = Decimal("0")
        annual_status: List[bool] = []

        for rec in annual_records:
            cumulative_guaranteed += guaranteed_annual_kwh
            cumulative_actual += rec.actual_savings_kwh
            annual_status.append(rec.actual_savings_kwh >= guaranteed_annual_kwh)

        surplus = cumulative_actual - cumulative_guaranteed
        guarantee_met = surplus >= Decimal("0")
        shortfall = max(-surplus, Decimal("0"))
        shortfall_cost = shortfall * penalty_rate
        penalty = shortfall_cost if not guarantee_met else Decimal("0")

        elapsed = (time.perf_counter() - t0) * 1000.0
        result = GuaranteeTrackingResult(
            contract_type=contract_type,
            contract_term_years=contract_term_years,
            guaranteed_annual_kwh=_round_val(guaranteed_annual_kwh, 2),
            guaranteed_annual_cost=_round_val(guaranteed_annual_cost, 2),
            cumulative_guaranteed_kwh=_round_val(cumulative_guaranteed, 2),
            cumulative_actual_kwh=_round_val(cumulative_actual, 2),
            cumulative_surplus_kwh=_round_val(surplus, 2),
            guarantee_met=guarantee_met,
            shortfall_kwh=_round_val(shortfall, 2),
            shortfall_cost=_round_val(shortfall_cost, 2),
            penalty_amount=_round_val(penalty, 2),
            years_tracked=n,
            annual_guarantee_status=annual_status,
            processing_time_ms=_round_val(_decimal(elapsed), 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Guarantee tracking: met=%s, surplus=%.0f kWh, penalty=%.2f, "
            "hash=%s (%.1f ms)",
            guarantee_met, float(surplus), float(penalty),
            result.provenance_hash[:16], elapsed,
        )
        return result

    def analyse_seasonal_patterns(
        self,
        quarterly_savings: Dict[int, Dict[str, Decimal]],
    ) -> SeasonalAnalysisResult:
        """Analyse seasonal savings patterns across years.

        Args:
            quarterly_savings: Dict mapping year -> {season: savings_kwh}.
                Seasons: "winter", "spring", "summer", "autumn".

        Returns:
            SeasonalAnalysisResult with patterns and anomalies.
        """
        t0 = time.perf_counter()
        logger.info("Analysing seasonal patterns: %d years", len(quarterly_savings))

        seasons = [SeasonType.WINTER, SeasonType.SPRING,
                    SeasonType.SUMMER, SeasonType.AUTUMN]
        season_values: Dict[str, List[Decimal]] = {s.value: [] for s in seasons}

        for year in sorted(quarterly_savings.keys()):
            year_data = quarterly_savings[year]
            for s in seasons:
                val = year_data.get(s.value, Decimal("0"))
                season_values[s.value].append(_decimal(val))

        # Compute seasonal patterns
        patterns: List[SeasonalPattern] = []
        overall_avg = Decimal("0")
        season_avgs: Dict[str, Decimal] = {}

        for s in seasons:
            vals = season_values[s.value]
            if vals:
                avg = _safe_divide(sum(vals), _decimal(len(vals)))
            else:
                avg = Decimal("0")
            season_avgs[s.value] = avg
            overall_avg += avg

        overall_avg = _safe_divide(overall_avg, Decimal("4")) if overall_avg > Decimal("0") else Decimal("1")

        anomalies: List[str] = []
        dominant_season = SeasonType.ANNUAL
        max_avg = Decimal("-1")

        for s in seasons:
            vals = season_values[s.value]
            avg = season_avgs[s.value]
            index = _safe_divide(avg, overall_avg, Decimal("1"))

            # Trend within season
            trend = self._determine_list_trend(vals)

            if avg > max_avg:
                max_avg = avg
                dominant_season = s

            # Anomaly detection (any year > 2x or < 0.5x of average)
            for i, v in enumerate(vals):
                if avg > Decimal("0") and (v > avg * Decimal("2") or v < avg * Decimal("0.5")):
                    year_num = sorted(quarterly_savings.keys())[i] if i < len(quarterly_savings) else i + 1
                    anomalies.append(
                        f"{s.value} year {year_num}: savings={float(v):.0f} "
                        f"vs avg={float(avg):.0f}"
                    )

            patterns.append(SeasonalPattern(
                season=s,
                average_savings_kwh=_round_val(avg, 2),
                seasonal_index=_round_val(index, 4),
                trend=trend,
                year_values=[_round_val(v, 2) for v in vals],
            ))

        # Seasonal variation
        if season_avgs:
            avg_vals = list(season_avgs.values())
            s_max = max(avg_vals)
            s_min = min(avg_vals)
            variation = _safe_pct(s_max - s_min, overall_avg) if overall_avg > Decimal("0") else Decimal("0")
        else:
            variation = Decimal("0")

        elapsed = (time.perf_counter() - t0) * 1000.0
        result = SeasonalAnalysisResult(
            patterns=patterns,
            dominant_season=dominant_season,
            seasonal_variation_pct=_round_val(variation, 2),
            anomalies=anomalies,
            processing_time_ms=_round_val(_decimal(elapsed), 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Seasonal analysis: dominant=%s, variation=%.1f%%, "
            "%d anomalies, hash=%s (%.1f ms)",
            dominant_season.value, float(variation), len(anomalies),
            result.provenance_hash[:16], elapsed,
        )
        return result

    def compare_year_over_year(
        self,
        records: List[AnnualSavingsRecord],
    ) -> List[YearOverYearComparison]:
        """Compare savings year-over-year for trend analysis.

        Args:
            records: Annual savings records.

        Returns:
            List of YearOverYearComparison for consecutive years.
        """
        t0 = time.perf_counter()
        comparisons: List[YearOverYearComparison] = []

        for i in range(1, len(records)):
            a = records[i - 1]
            b = records[i]
            change = b.actual_savings_kwh - a.actual_savings_kwh
            change_pct = _safe_pct(change, a.actual_savings_kwh) if a.actual_savings_kwh != Decimal("0") else Decimal("0")

            if change_pct > Decimal("5"):
                direction = TrendDirection.IMPROVING
            elif change_pct < Decimal("-5"):
                direction = TrendDirection.DECLINING
            else:
                direction = TrendDirection.STABLE

            comp = YearOverYearComparison(
                year_a=a.year,
                year_b=b.year,
                savings_a_kwh=_round_val(a.actual_savings_kwh, 2),
                savings_b_kwh=_round_val(b.actual_savings_kwh, 2),
                change_kwh=_round_val(change, 2),
                change_pct=_round_val(change_pct, 2),
                direction=direction,
            )
            comp.provenance_hash = _compute_hash(comp)
            comparisons.append(comp)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Year-over-year: %d comparisons (%.1f ms)", len(comparisons), elapsed,
        )
        return comparisons

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_persistence_factors(
        self, records: List[AnnualSavingsRecord],
    ) -> List[AnnualSavingsRecord]:
        """Compute persistence factor and status for each year."""
        updated = []
        for rec in records:
            r = rec.model_copy(deep=True)
            r.persistence_factor = _safe_divide(
                r.actual_savings_kwh, r.expected_savings_kwh, Decimal("0")
            )
            r.persistence_factor = _round_val(r.persistence_factor, 4)
            r.status = self._classify_status(r.persistence_factor)
            updated.append(r)
        return updated

    def _classify_status(self, pf: Decimal) -> PersistenceStatus:
        """Classify persistence factor into status category."""
        if pf >= Decimal("0.95"):
            return PersistenceStatus.EXCELLENT
        if pf >= Decimal("0.85"):
            return PersistenceStatus.GOOD
        if pf >= Decimal("0.80"):
            return PersistenceStatus.ACCEPTABLE
        if pf >= Decimal("0.60"):
            return PersistenceStatus.DEGRADED
        return PersistenceStatus.CRITICAL

    def _generate_alerts(
        self, records: List[AnnualSavingsRecord],
    ) -> List[PersistenceAlert]:
        """Generate alerts based on persistence factors."""
        alerts: List[PersistenceAlert] = []
        for rec in records:
            pf = rec.persistence_factor
            if pf < self._critical_threshold:
                alerts.append(PersistenceAlert(
                    year=rec.year,
                    level=AlertLevel.CRITICAL,
                    message=(
                        f"Year {rec.year}: Persistence factor {float(pf):.2f} "
                        f"below critical threshold {float(self._critical_threshold)}"
                    ),
                    persistence_factor=pf,
                    threshold=self._critical_threshold,
                    recommended_action="Immediate re-commissioning required",
                ))
            elif pf < self._recommission_threshold:
                alerts.append(PersistenceAlert(
                    year=rec.year,
                    level=AlertLevel.ALERT,
                    message=(
                        f"Year {rec.year}: Persistence factor {float(pf):.2f} "
                        f"below re-commissioning threshold {float(self._recommission_threshold)}"
                    ),
                    persistence_factor=pf,
                    threshold=self._recommission_threshold,
                    recommended_action="Schedule re-commissioning investigation",
                ))
            elif pf < self._warning_threshold:
                alerts.append(PersistenceAlert(
                    year=rec.year,
                    level=AlertLevel.WARNING,
                    message=(
                        f"Year {rec.year}: Persistence factor {float(pf):.2f} "
                        f"below warning threshold {float(self._warning_threshold)}"
                    ),
                    persistence_factor=pf,
                    threshold=self._warning_threshold,
                    recommended_action="Monitor closely; investigate root cause",
                ))
        return alerts

    def _determine_trend(
        self, records: List[AnnualSavingsRecord],
    ) -> TrendDirection:
        """Determine overall savings trend direction."""
        if len(records) < 2:
            return TrendDirection.STABLE
        pf_values = [r.persistence_factor for r in records]
        return self._determine_list_trend(pf_values)

    def _determine_list_trend(self, values: List[Decimal]) -> TrendDirection:
        """Determine trend direction from a list of values."""
        if len(values) < 2:
            return TrendDirection.STABLE

        # Simple linear trend via least squares slope sign
        n = len(values)
        x_mean = _decimal(n - 1) / Decimal("2")
        y_mean = _safe_divide(sum(values), _decimal(n))
        numerator = sum((_decimal(i) - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((_decimal(i) - x_mean) ** 2 for i in range(n))
        slope = _safe_divide(numerator, denominator)

        # Coefficient of variation to check volatility
        if y_mean > Decimal("0"):
            variance = _safe_divide(
                sum((v - y_mean) ** 2 for v in values), _decimal(n)
            )
            cv = _decimal(math.sqrt(float(variance))) / y_mean if variance > Decimal("0") else Decimal("0")
        else:
            cv = Decimal("0")

        if cv > Decimal("0.3"):
            return TrendDirection.VOLATILE
        if slope > Decimal("0.01"):
            return TrendDirection.IMPROVING
        if slope < Decimal("-0.01"):
            return TrendDirection.DECLINING
        return TrendDirection.STABLE

    def _fit_linear_degradation(
        self, actual: List[Decimal], s0: Decimal,
    ) -> DegradationResult:
        """Fit linear degradation: S(t) = S0 * (1 - d*t)."""
        n = len(actual)
        if n < 2 or s0 == Decimal("0"):
            return DegradationResult(model_type=DegradationModel.LINEAR)

        # Least-squares fit of S(t)/S0 = 1 - d*t
        # Rewrite as y = a - d*x where y = S(t)/S0, x = t-1
        ratios = [_safe_divide(actual[i], s0) for i in range(n)]
        x = [_decimal(i) for i in range(n)]

        # Simple linear regression
        n_dec = _decimal(n)
        x_mean = _safe_divide(sum(x), n_dec)
        y_mean = _safe_divide(sum(ratios), n_dec)
        num = sum((x[i] - x_mean) * (ratios[i] - y_mean) for i in range(n))
        den = sum((x[i] - x_mean) ** 2 for i in range(n))
        slope = _safe_divide(num, den)  # This is -d (degradation rate)
        intercept = y_mean - slope * x_mean

        d = -slope  # Positive d means savings are decreasing
        d_pct = d * Decimal("100")

        # Predicted and residuals
        predicted = [s0 * (intercept + slope * _decimal(i)) for i in range(n)]
        residuals = [actual[i] - predicted[i] for i in range(n)]

        # R-squared
        ss_tot = sum((actual[i] - _safe_divide(sum(actual), n_dec)) ** 2 for i in range(n))
        ss_res = sum(r ** 2 for r in residuals)
        r2 = Decimal("1") - _safe_divide(ss_res, ss_tot) if ss_tot > Decimal("0") else Decimal("0")
        r2 = max(r2, Decimal("0"))

        # Years to 80%
        years_to_80 = _safe_divide(Decimal("0.20"), d) if d > Decimal("0") else Decimal("999")

        return DegradationResult(
            model_type=DegradationModel.LINEAR,
            annual_degradation_rate=_round_val(abs(d), 6),
            annual_degradation_pct=_round_val(abs(d_pct), 4),
            predicted_savings=[_round_val(p, 2) for p in predicted],
            residuals=[_round_val(r, 2) for r in residuals],
            r_squared=_round_val(r2, 6),
            years_to_80pct=_round_val(min(years_to_80, Decimal("999")), 2),
        )

    def _fit_exponential_degradation(
        self, actual: List[Decimal], s0: Decimal,
    ) -> DegradationResult:
        """Fit exponential decay: S(t) = S0 * exp(-lambda * t)."""
        n = len(actual)
        if n < 2 or s0 <= Decimal("0"):
            return DegradationResult(model_type=DegradationModel.EXPONENTIAL)

        # Take log: ln(S/S0) = -lambda * t
        # Linear regression on ln_ratios vs t
        ln_ratios: List[Decimal] = []
        valid_x: List[Decimal] = []
        for i in range(n):
            ratio = _safe_divide(actual[i], s0)
            if ratio > Decimal("0"):
                ln_ratios.append(_decimal(math.log(float(ratio))))
                valid_x.append(_decimal(i))

        if len(ln_ratios) < 2:
            return DegradationResult(model_type=DegradationModel.EXPONENTIAL)

        m = len(ln_ratios)
        m_dec = _decimal(m)
        x_mean = _safe_divide(sum(valid_x), m_dec)
        y_mean = _safe_divide(sum(ln_ratios), m_dec)
        num = sum((valid_x[i] - x_mean) * (ln_ratios[i] - y_mean) for i in range(m))
        den = sum((valid_x[i] - x_mean) ** 2 for i in range(m))
        neg_lambda = _safe_divide(num, den)  # Should be negative for decay
        lam = abs(neg_lambda)

        # Half-life = ln(2) / lambda
        half_life = _safe_divide(_decimal(math.log(2)), lam, Decimal("999"))

        # Annual degradation rate approximation
        d_annual = Decimal("1") - _decimal(math.exp(-float(lam)))
        d_annual_pct = d_annual * Decimal("100")

        # Predicted
        predicted = [s0 * _decimal(math.exp(-float(lam) * i)) for i in range(n)]
        residuals = [actual[i] - predicted[i] for i in range(n)]

        # R-squared
        a_mean = _safe_divide(sum(actual), _decimal(n))
        ss_tot = sum((actual[i] - a_mean) ** 2 for i in range(n))
        ss_res = sum(r ** 2 for r in residuals)
        r2 = Decimal("1") - _safe_divide(ss_res, ss_tot) if ss_tot > Decimal("0") else Decimal("0")
        r2 = max(r2, Decimal("0"))

        # Years to 80%: 0.8 = exp(-lambda*t) => t = -ln(0.8)/lambda
        years_to_80 = _safe_divide(
            _decimal(-math.log(0.8)), lam, Decimal("999")
        )

        return DegradationResult(
            model_type=DegradationModel.EXPONENTIAL,
            annual_degradation_rate=_round_val(d_annual, 6),
            annual_degradation_pct=_round_val(d_annual_pct, 4),
            lambda_decay=_round_val(lam, 6),
            half_life_years=_round_val(min(half_life, Decimal("999")), 2),
            predicted_savings=[_round_val(p, 2) for p in predicted],
            residuals=[_round_val(r, 2) for r in residuals],
            r_squared=_round_val(r2, 6),
            years_to_80pct=_round_val(min(years_to_80, Decimal("999")), 2),
        )

    def _detect_step_changes(
        self,
        actual: List[Decimal],
        threshold: Decimal = Decimal("0.15"),
    ) -> DegradationResult:
        """Detect step-changes in savings history."""
        step_years: List[int] = []
        for i in range(1, len(actual)):
            if actual[i - 1] != Decimal("0"):
                change_pct = abs(actual[i] - actual[i - 1]) / actual[i - 1]
                if change_pct > threshold:
                    step_years.append(i + 1)  # Year number

        return DegradationResult(
            model_type=DegradationModel.STEP_CHANGE,
            step_changes=step_years,
            annual_degradation_rate=Decimal("0"),
            annual_degradation_pct=Decimal("0"),
        )
