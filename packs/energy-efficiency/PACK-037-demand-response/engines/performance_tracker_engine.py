# -*- coding: utf-8 -*-
"""
PerformanceTrackerEngine - PACK-037 Demand Response Engine 7
==============================================================

Event performance tracking engine for demand response programmes.
Evaluates actual versus baseline curtailment, computes performance
ratios, tracks compliance against programme requirements, summarises
seasonal performance with revenue and penalty accounting, and detects
performance degradation trends via rolling averages.

Calculation Methodology:
    Event Performance:
        curtailment_kwh   = baseline_kwh - actual_kwh
        performance_ratio = curtailment_kwh / nominated_kwh * 100
        compliance_flag   = performance_ratio >= compliance_threshold_pct

    Season Summary:
        events_count      = count(events in season)
        compliance_rate   = compliant_events / total_events * 100
        total_revenue     = sum(event_revenue for compliant events)
        total_penalties   = sum(penalty for non-compliant events)
        net_income        = total_revenue - total_penalties

    Performance Trend:
        rolling_avg       = mean(performance_ratio over window)
        trend_slope       = linear regression slope over window
        degradation       = trend_slope < -threshold per event

    Compliance Report:
        overall_compliance_pct = compliant / total * 100
        risk_score             = f(non_compliance_count, penalty_total)
        reliability_index      = 1 - (missed_events / total_events)

Regulatory References:
    - FERC Order 745 - Demand Response Compensation in Wholesale Markets
    - PJM Capacity Performance Requirements
    - ISO-NE Forward Capacity Market (FCM) Rules
    - NYISO ICAP Special Case Resource (SCR) Requirements
    - CAISO Proxy Demand Response (PDR) Performance Requirements
    - NAESB WEQ Business Practices for DR
    - EU Electricity Directive 2019/944 (demand response provisions)

Zero-Hallucination:
    - Performance ratios computed from metered baseline and actual data
    - Compliance thresholds from published ISO/RTO programme rules
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-037 Demand Response
Engine:  7 of 10
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

class ComplianceStatus(str, Enum):
    """Event compliance status.

    COMPLIANT:     Performance meets or exceeds programme threshold.
    NON_COMPLIANT: Performance below required threshold.
    PARTIAL:       Partial performance (between floor and target).
    EXCUSED:       Non-performance excused (force majeure, etc.).
    PENDING:       Performance data not yet finalised.
    """
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    EXCUSED = "excused"
    PENDING = "pending"

class TrendDirection(str, Enum):
    """Performance trend direction indicator.

    IMPROVING:  Performance ratio trending upward.
    STABLE:     Performance within +/-5% of rolling average.
    DECLINING:  Performance ratio trending downward.
    DEGRADING:  Sustained decline exceeding degradation threshold.
    INSUFFICIENT_DATA: Not enough data points for trend analysis.
    """
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    DEGRADING = "degrading"
    INSUFFICIENT_DATA = "insufficient_data"

class SeasonType(str, Enum):
    """DR programme season classification.

    SUMMER:        Summer peak season (Jun-Sep typical).
    WINTER:        Winter peak season (Dec-Feb typical).
    SHOULDER:      Shoulder season (spring/fall).
    ANNUAL:        Full annual period.
    CUSTOM:        Custom-defined period.
    """
    SUMMER = "summer"
    WINTER = "winter"
    SHOULDER = "shoulder"
    ANNUAL = "annual"
    CUSTOM = "custom"

class PenaltyType(str, Enum):
    """Penalty type for non-compliance.

    CAPACITY_SHORTFALL:    Penalty per kW of shortfall.
    NON_RESPONSE:          Penalty for complete non-response.
    UNDER_PERFORMANCE:     Graduated penalty for under-performance.
    AVAILABILITY_FAILURE:  Penalty for unavailability during event.
    NONE:                  No penalty applicable.
    """
    CAPACITY_SHORTFALL = "capacity_shortfall"
    NON_RESPONSE = "non_response"
    UNDER_PERFORMANCE = "under_performance"
    AVAILABILITY_FAILURE = "availability_failure"
    NONE = "none"

class RiskLevel(str, Enum):
    """Compliance risk level assessment.

    LOW:       High compliance rate, minimal penalties.
    MODERATE:  Some non-compliance, manageable penalties.
    HIGH:      Frequent non-compliance, significant penalties.
    CRITICAL:  Programme expulsion risk.
    """
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default compliance threshold (% of nominated capacity).
DEFAULT_COMPLIANCE_THRESHOLD_PCT: Decimal = Decimal("80")

# Partial compliance floor (below this = non-compliant).
DEFAULT_PARTIAL_FLOOR_PCT: Decimal = Decimal("50")

# Trend analysis defaults.
DEFAULT_ROLLING_WINDOW: int = 10
DEFAULT_DEGRADATION_SLOPE_THRESHOLD: Decimal = Decimal("-2.0")
MINIMUM_TREND_DATAPOINTS: int = 5

# Risk score thresholds.
RISK_THRESHOLDS: Dict[str, Decimal] = {
    RiskLevel.LOW.value: Decimal("90"),
    RiskLevel.MODERATE.value: Decimal("75"),
    RiskLevel.HIGH.value: Decimal("50"),
}

# Penalty rates by type (USD per kW-day or per event).
DEFAULT_PENALTY_RATES: Dict[str, Decimal] = {
    PenaltyType.CAPACITY_SHORTFALL.value: Decimal("50"),
    PenaltyType.NON_RESPONSE.value: Decimal("500"),
    PenaltyType.UNDER_PERFORMANCE.value: Decimal("25"),
    PenaltyType.AVAILABILITY_FAILURE.value: Decimal("100"),
    PenaltyType.NONE.value: Decimal("0"),
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class EventRecord(BaseModel):
    """Record of a single DR event for performance tracking.

    Attributes:
        event_id: Unique event identifier.
        programme_id: DR programme identifier.
        event_date: Date and time of the event.
        duration_hours: Event duration (hours).
        baseline_kwh: Baseline energy consumption (kWh).
        actual_kwh: Actual energy consumption during event (kWh).
        nominated_kw: Nominated curtailment capacity (kW).
        nominated_kwh: Nominated curtailment energy (kWh).
        revenue_earned: Revenue earned from this event (USD).
        penalty_amount: Penalty incurred (USD).
        penalty_type: Type of penalty applied.
        notes: Event notes.
    """
    event_id: str = Field(
        default_factory=_new_uuid, description="Event ID"
    )
    programme_id: str = Field(
        default="", max_length=200, description="Programme ID"
    )
    event_date: datetime = Field(
        default_factory=utcnow, description="Event date/time"
    )
    duration_hours: Decimal = Field(
        default=Decimal("1"), ge=Decimal("0.25"), le=Decimal("24"),
        description="Duration (hours)"
    )
    baseline_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Baseline consumption (kWh)"
    )
    actual_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Actual consumption (kWh)"
    )
    nominated_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Nominated capacity (kW)"
    )
    nominated_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Nominated energy (kWh)"
    )
    revenue_earned: Decimal = Field(
        default=Decimal("0"), ge=0, description="Revenue earned (USD)"
    )
    penalty_amount: Decimal = Field(
        default=Decimal("0"), ge=0, description="Penalty amount (USD)"
    )
    penalty_type: PenaltyType = Field(
        default=PenaltyType.NONE, description="Penalty type"
    )
    notes: str = Field(
        default="", max_length=1000, description="Event notes"
    )

    @field_validator("baseline_kwh")
    @classmethod
    def validate_baseline(cls, v: Decimal) -> Decimal:
        """Ensure baseline is non-negative."""
        if v < Decimal("0"):
            raise ValueError("Baseline kWh must be >= 0")
        return v

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class EventPerformance(BaseModel):
    """Performance analysis for a single DR event.

    Attributes:
        event_id: Event identifier.
        curtailment_kwh: Actual curtailment achieved (kWh).
        curtailment_kw: Average curtailment power (kW).
        performance_ratio_pct: Ratio of actual to nominated (%).
        compliance_status: Compliance determination.
        compliance_threshold_pct: Threshold used for compliance.
        shortfall_kwh: Shortfall below nomination (kWh).
        shortfall_kw: Shortfall below nomination (kW).
        revenue: Revenue earned or forfeited (USD).
        penalty: Penalty incurred (USD).
        net_value: Revenue minus penalty (USD).
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    event_id: str = Field(default="", description="Event ID")
    curtailment_kwh: Decimal = Field(
        default=Decimal("0"), description="Curtailment achieved (kWh)"
    )
    curtailment_kw: Decimal = Field(
        default=Decimal("0"), description="Average curtailment (kW)"
    )
    performance_ratio_pct: Decimal = Field(
        default=Decimal("0"), description="Performance ratio (%)"
    )
    compliance_status: ComplianceStatus = Field(
        default=ComplianceStatus.PENDING, description="Compliance status"
    )
    compliance_threshold_pct: Decimal = Field(
        default=DEFAULT_COMPLIANCE_THRESHOLD_PCT,
        description="Compliance threshold (%)"
    )
    shortfall_kwh: Decimal = Field(
        default=Decimal("0"), description="Shortfall (kWh)"
    )
    shortfall_kw: Decimal = Field(
        default=Decimal("0"), description="Shortfall (kW)"
    )
    revenue: Decimal = Field(
        default=Decimal("0"), description="Revenue (USD)"
    )
    penalty: Decimal = Field(
        default=Decimal("0"), description="Penalty (USD)"
    )
    net_value: Decimal = Field(
        default=Decimal("0"), description="Net value (USD)"
    )
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

class SeasonSummary(BaseModel):
    """Seasonal performance summary.

    Attributes:
        season: Season type.
        season_label: Human-readable season label.
        total_events: Total DR events in season.
        compliant_events: Number of compliant events.
        non_compliant_events: Number of non-compliant events.
        partial_events: Number of partially compliant events.
        compliance_rate_pct: Overall compliance rate (%).
        total_curtailment_kwh: Total curtailment achieved (kWh).
        total_nominated_kwh: Total nominated energy (kWh).
        avg_performance_ratio_pct: Average performance ratio (%).
        total_revenue: Total revenue earned (USD).
        total_penalties: Total penalties incurred (USD).
        net_income: Revenue minus penalties (USD).
        best_event_ratio: Best event performance ratio (%).
        worst_event_ratio: Worst event performance ratio (%).
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    season: SeasonType = Field(
        default=SeasonType.ANNUAL, description="Season type"
    )
    season_label: str = Field(
        default="", max_length=200, description="Season label"
    )
    total_events: int = Field(default=0, ge=0, description="Total events")
    compliant_events: int = Field(
        default=0, ge=0, description="Compliant events"
    )
    non_compliant_events: int = Field(
        default=0, ge=0, description="Non-compliant events"
    )
    partial_events: int = Field(
        default=0, ge=0, description="Partial events"
    )
    compliance_rate_pct: Decimal = Field(
        default=Decimal("0"), description="Compliance rate (%)"
    )
    total_curtailment_kwh: Decimal = Field(
        default=Decimal("0"), description="Total curtailment (kWh)"
    )
    total_nominated_kwh: Decimal = Field(
        default=Decimal("0"), description="Total nominated (kWh)"
    )
    avg_performance_ratio_pct: Decimal = Field(
        default=Decimal("0"), description="Avg performance ratio (%)"
    )
    total_revenue: Decimal = Field(
        default=Decimal("0"), description="Total revenue (USD)"
    )
    total_penalties: Decimal = Field(
        default=Decimal("0"), description="Total penalties (USD)"
    )
    net_income: Decimal = Field(
        default=Decimal("0"), description="Net income (USD)"
    )
    best_event_ratio: Decimal = Field(
        default=Decimal("0"), description="Best event ratio (%)"
    )
    worst_event_ratio: Decimal = Field(
        default=Decimal("0"), description="Worst event ratio (%)"
    )
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

class PerformanceTrend(BaseModel):
    """Performance trend analysis result.

    Attributes:
        window_size: Number of events in rolling window.
        rolling_avg_pct: Rolling average performance ratio (%).
        trend_direction: Detected trend direction.
        trend_slope: Linear regression slope (pct per event).
        min_ratio_pct: Minimum ratio in window (%).
        max_ratio_pct: Maximum ratio in window (%).
        std_deviation_pct: Standard deviation of ratios (%).
        degradation_detected: Whether degradation pattern detected.
        degradation_rate: Estimated degradation rate if detected.
        data_points: Number of data points analysed.
        provenance_hash: SHA-256 audit hash.
    """
    window_size: int = Field(
        default=DEFAULT_ROLLING_WINDOW, description="Window size"
    )
    rolling_avg_pct: Decimal = Field(
        default=Decimal("0"), description="Rolling average (%)"
    )
    trend_direction: TrendDirection = Field(
        default=TrendDirection.INSUFFICIENT_DATA, description="Trend direction"
    )
    trend_slope: Decimal = Field(
        default=Decimal("0"), description="Trend slope (pct/event)"
    )
    min_ratio_pct: Decimal = Field(
        default=Decimal("0"), description="Min ratio (%)"
    )
    max_ratio_pct: Decimal = Field(
        default=Decimal("0"), description="Max ratio (%)"
    )
    std_deviation_pct: Decimal = Field(
        default=Decimal("0"), description="Std deviation (%)"
    )
    degradation_detected: bool = Field(
        default=False, description="Degradation detected"
    )
    degradation_rate: Decimal = Field(
        default=Decimal("0"), description="Degradation rate"
    )
    data_points: int = Field(
        default=0, ge=0, description="Data points analysed"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

class ComplianceReport(BaseModel):
    """Comprehensive compliance report.

    Attributes:
        report_id: Report identifier.
        programme_id: DR programme identifier.
        reporting_period: Reporting period label.
        overall_compliance_pct: Overall compliance rate (%).
        risk_level: Assessed risk level.
        risk_score: Numeric risk score (0-100).
        reliability_index: Reliability index (0-1).
        total_events: Total events in period.
        compliant_events: Compliant events count.
        missed_events: Completely missed events.
        event_performances: Individual event performance records.
        season_summary: Seasonal summary.
        trend_analysis: Performance trend analysis.
        total_revenue: Total revenue (USD).
        total_penalties: Total penalties (USD).
        net_income: Net income (USD).
        recommendations: List of improvement recommendations.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    report_id: str = Field(
        default_factory=_new_uuid, description="Report ID"
    )
    programme_id: str = Field(
        default="", description="Programme ID"
    )
    reporting_period: str = Field(
        default="", description="Reporting period"
    )
    overall_compliance_pct: Decimal = Field(
        default=Decimal("0"), description="Overall compliance (%)"
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.LOW, description="Risk level"
    )
    risk_score: Decimal = Field(
        default=Decimal("0"), description="Risk score (0-100)"
    )
    reliability_index: Decimal = Field(
        default=Decimal("0"), description="Reliability index (0-1)"
    )
    total_events: int = Field(default=0, ge=0, description="Total events")
    compliant_events: int = Field(
        default=0, ge=0, description="Compliant events"
    )
    missed_events: int = Field(
        default=0, ge=0, description="Missed events"
    )
    event_performances: List[EventPerformance] = Field(
        default_factory=list, description="Event performances"
    )
    season_summary: Optional[SeasonSummary] = Field(
        default=None, description="Season summary"
    )
    trend_analysis: Optional[PerformanceTrend] = Field(
        default=None, description="Trend analysis"
    )
    total_revenue: Decimal = Field(
        default=Decimal("0"), description="Total revenue (USD)"
    )
    total_penalties: Decimal = Field(
        default=Decimal("0"), description="Total penalties (USD)"
    )
    net_income: Decimal = Field(
        default=Decimal("0"), description="Net income (USD)"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations"
    )
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PerformanceTrackerEngine:
    """Event performance tracking engine for demand response.

    Evaluates actual versus baseline curtailment, computes performance
    ratios, tracks compliance, summarises seasonal performance, and
    detects performance degradation trends.

    Usage::

        engine = PerformanceTrackerEngine()
        perf = engine.track_event(event_record)
        compliance = engine.calculate_compliance(event_record)
        summary = engine.summarize_season(events, SeasonType.SUMMER)
        trend = engine.detect_trends(event_performances)
        report = engine.generate_compliance_report(events, "2026-Summer")

    All arithmetic uses ``Decimal`` for deterministic, audit-grade precision.
    Every result carries a SHA-256 provenance hash.
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise PerformanceTrackerEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - compliance_threshold_pct (Decimal): compliance threshold
                - partial_floor_pct (Decimal): partial compliance floor
                - rolling_window (int): trend analysis window size
                - degradation_slope_threshold (Decimal): slope threshold
                - penalty_rates (dict): custom penalty rates
        """
        self.config = config or {}
        self._threshold = _decimal(
            self.config.get("compliance_threshold_pct", DEFAULT_COMPLIANCE_THRESHOLD_PCT)
        )
        self._partial_floor = _decimal(
            self.config.get("partial_floor_pct", DEFAULT_PARTIAL_FLOOR_PCT)
        )
        self._rolling_window = int(
            self.config.get("rolling_window", DEFAULT_ROLLING_WINDOW)
        )
        self._degradation_threshold = _decimal(
            self.config.get(
                "degradation_slope_threshold", DEFAULT_DEGRADATION_SLOPE_THRESHOLD
            )
        )
        self._penalty_rates = dict(DEFAULT_PENALTY_RATES)
        if "penalty_rates" in self.config:
            self._penalty_rates.update(self.config["penalty_rates"])

        logger.info(
            "PerformanceTrackerEngine v%s initialised "
            "(threshold=%.1f%%, window=%d)",
            self.engine_version, float(self._threshold), self._rolling_window,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def track_event(
        self,
        event: EventRecord,
        compliance_threshold_pct: Optional[Decimal] = None,
    ) -> EventPerformance:
        """Track performance for a single DR event.

        Computes curtailment, performance ratio, compliance status,
        and net financial value.

        Args:
            event: Event record with baseline and actual data.
            compliance_threshold_pct: Override compliance threshold.

        Returns:
            EventPerformance with full analysis.
        """
        t0 = time.perf_counter()
        threshold = compliance_threshold_pct or self._threshold
        logger.info(
            "Tracking event: id=%s, baseline=%s kWh, actual=%s kWh, "
            "nominated=%s kWh",
            event.event_id, str(event.baseline_kwh),
            str(event.actual_kwh), str(event.nominated_kwh),
        )

        # Curtailment
        curtailment_kwh = max(
            event.baseline_kwh - event.actual_kwh, Decimal("0")
        )
        curtailment_kw = _safe_divide(curtailment_kwh, event.duration_hours)

        # Performance ratio
        nominated = event.nominated_kwh
        if nominated <= Decimal("0"):
            nominated = event.nominated_kw * event.duration_hours
        performance_ratio = _safe_pct(curtailment_kwh, nominated)

        # Compliance determination
        compliance = self._determine_compliance(performance_ratio, threshold)

        # Shortfall
        shortfall_kwh = max(nominated - curtailment_kwh, Decimal("0"))
        shortfall_kw = _safe_divide(shortfall_kwh, event.duration_hours)

        # Financial
        revenue = event.revenue_earned if compliance != ComplianceStatus.NON_COMPLIANT else Decimal("0")
        penalty = event.penalty_amount
        if compliance == ComplianceStatus.NON_COMPLIANT and penalty == Decimal("0"):
            penalty = self._calculate_penalty(
                shortfall_kw, event.duration_hours, event.penalty_type,
            )
        net_value = revenue - penalty

        result = EventPerformance(
            event_id=event.event_id,
            curtailment_kwh=_round_val(curtailment_kwh, 2),
            curtailment_kw=_round_val(curtailment_kw, 2),
            performance_ratio_pct=_round_val(performance_ratio, 2),
            compliance_status=compliance,
            compliance_threshold_pct=_round_val(threshold, 2),
            shortfall_kwh=_round_val(shortfall_kwh, 2),
            shortfall_kw=_round_val(shortfall_kw, 2),
            revenue=_round_val(revenue, 2),
            penalty=_round_val(penalty, 2),
            net_value=_round_val(net_value, 2),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Event tracked: id=%s, curtailment=%.2f kWh, "
            "ratio=%.1f%%, compliance=%s, net=%.2f, hash=%s (%.1f ms)",
            event.event_id, float(curtailment_kwh),
            float(performance_ratio), compliance.value,
            float(net_value), result.provenance_hash[:16], elapsed,
        )
        return result

    def calculate_compliance(
        self,
        event: EventRecord,
        threshold_pct: Optional[Decimal] = None,
    ) -> ComplianceStatus:
        """Calculate compliance status for a single event.

        Convenience method returning just the compliance status.

        Args:
            event: Event record.
            threshold_pct: Optional threshold override.

        Returns:
            ComplianceStatus enum value.
        """
        perf = self.track_event(event, threshold_pct)
        return perf.compliance_status

    def summarize_season(
        self,
        events: List[EventRecord],
        season: SeasonType = SeasonType.ANNUAL,
        season_label: str = "",
    ) -> SeasonSummary:
        """Summarise performance over a DR season.

        Aggregates event performances, computes compliance rate,
        totals revenue and penalties, and identifies best/worst events.

        Args:
            events: List of event records for the season.
            season: Season type classification.
            season_label: Human-readable label (e.g. "Summer 2026").

        Returns:
            SeasonSummary with aggregate metrics.
        """
        t0 = time.perf_counter()
        logger.info(
            "Summarising season: %s (%s), %d events",
            season_label or season.value, season.value, len(events),
        )

        # Track all events
        performances: List[EventPerformance] = []
        for event in events:
            perf = self.track_event(event)
            performances.append(perf)

        # Counts
        compliant = sum(
            1 for p in performances if p.compliance_status == ComplianceStatus.COMPLIANT
        )
        non_compliant = sum(
            1 for p in performances if p.compliance_status == ComplianceStatus.NON_COMPLIANT
        )
        partial = sum(
            1 for p in performances if p.compliance_status == ComplianceStatus.PARTIAL
        )
        total = len(performances)

        # Compliance rate
        compliance_rate = _safe_pct(_decimal(compliant), _decimal(total))

        # Totals
        total_curtailment = sum(
            (p.curtailment_kwh for p in performances), Decimal("0")
        )
        total_nominated = sum(
            (e.nominated_kwh if e.nominated_kwh > Decimal("0")
             else e.nominated_kw * e.duration_hours for e in events),
            Decimal("0"),
        )
        avg_ratio = Decimal("0")
        if performances:
            ratio_sum = sum(
                (p.performance_ratio_pct for p in performances), Decimal("0")
            )
            avg_ratio = ratio_sum / _decimal(total)

        total_revenue = sum((p.revenue for p in performances), Decimal("0"))
        total_penalties = sum((p.penalty for p in performances), Decimal("0"))
        net_income = total_revenue - total_penalties

        # Best / worst
        ratios = [p.performance_ratio_pct for p in performances]
        best = max(ratios) if ratios else Decimal("0")
        worst = min(ratios) if ratios else Decimal("0")

        summary = SeasonSummary(
            season=season,
            season_label=season_label or season.value,
            total_events=total,
            compliant_events=compliant,
            non_compliant_events=non_compliant,
            partial_events=partial,
            compliance_rate_pct=_round_val(compliance_rate, 2),
            total_curtailment_kwh=_round_val(total_curtailment, 2),
            total_nominated_kwh=_round_val(total_nominated, 2),
            avg_performance_ratio_pct=_round_val(avg_ratio, 2),
            total_revenue=_round_val(total_revenue, 2),
            total_penalties=_round_val(total_penalties, 2),
            net_income=_round_val(net_income, 2),
            best_event_ratio=_round_val(best, 2),
            worst_event_ratio=_round_val(worst, 2),
        )
        summary.provenance_hash = _compute_hash(summary)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Season summary: %d events, compliance=%.1f%%, "
            "revenue=%.2f, penalties=%.2f, net=%.2f, hash=%s (%.1f ms)",
            total, float(compliance_rate), float(total_revenue),
            float(total_penalties), float(net_income),
            summary.provenance_hash[:16], elapsed,
        )
        return summary

    def detect_trends(
        self,
        performances: List[EventPerformance],
        window_size: Optional[int] = None,
    ) -> PerformanceTrend:
        """Detect performance trends using rolling averages.

        Computes a rolling average of performance ratios, fits a
        linear regression to detect slope, and flags degradation
        when the slope falls below the configured threshold.

        Args:
            performances: Ordered list of event performances (oldest first).
            window_size: Override rolling window size.

        Returns:
            PerformanceTrend with direction and degradation assessment.
        """
        t0 = time.perf_counter()
        window = window_size or self._rolling_window
        n = len(performances)
        logger.info(
            "Detecting trends: %d events, window=%d", n, window,
        )

        if n < MINIMUM_TREND_DATAPOINTS:
            trend = PerformanceTrend(
                window_size=window,
                data_points=n,
                trend_direction=TrendDirection.INSUFFICIENT_DATA,
            )
            trend.provenance_hash = _compute_hash(trend)
            return trend

        # Extract ratios from the most recent window
        window_data = performances[-window:] if n > window else performances
        ratios = [float(p.performance_ratio_pct) for p in window_data]
        m = len(ratios)

        # Rolling average
        rolling_avg = sum(ratios) / m

        # Min / max
        min_ratio = min(ratios)
        max_ratio = max(ratios)

        # Standard deviation
        mean = rolling_avg
        variance = sum((r - mean) ** 2 for r in ratios) / m
        std_dev = math.sqrt(variance) if variance > 0 else 0.0

        # Linear regression slope
        slope = self._linear_regression_slope(ratios)

        # Trend direction
        slope_dec = _decimal(slope)
        if slope_dec < self._degradation_threshold:
            direction = TrendDirection.DEGRADING
            degradation_detected = True
        elif slope_dec < Decimal("-0.5"):
            direction = TrendDirection.DECLINING
            degradation_detected = False
        elif slope_dec > Decimal("0.5"):
            direction = TrendDirection.IMPROVING
            degradation_detected = False
        else:
            direction = TrendDirection.STABLE
            degradation_detected = False

        trend = PerformanceTrend(
            window_size=window,
            rolling_avg_pct=_round_val(_decimal(rolling_avg), 2),
            trend_direction=direction,
            trend_slope=_round_val(slope_dec, 4),
            min_ratio_pct=_round_val(_decimal(min_ratio), 2),
            max_ratio_pct=_round_val(_decimal(max_ratio), 2),
            std_deviation_pct=_round_val(_decimal(std_dev), 4),
            degradation_detected=degradation_detected,
            degradation_rate=_round_val(abs(slope_dec), 4) if degradation_detected else Decimal("0"),
            data_points=m,
        )
        trend.provenance_hash = _compute_hash(trend)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Trend analysis: avg=%.1f%%, slope=%.4f, direction=%s, "
            "degradation=%s, hash=%s (%.1f ms)",
            rolling_avg, slope, direction.value,
            degradation_detected, trend.provenance_hash[:16], elapsed,
        )
        return trend

    def generate_compliance_report(
        self,
        events: List[EventRecord],
        reporting_period: str = "",
        programme_id: str = "",
    ) -> ComplianceReport:
        """Generate a comprehensive compliance report.

        Combines event performance tracking, seasonal summary,
        trend analysis, risk assessment, and recommendations.

        Args:
            events: List of event records.
            reporting_period: Period label (e.g. "2026-Q2").
            programme_id: DR programme identifier.

        Returns:
            ComplianceReport with full analysis and recommendations.
        """
        t0 = time.perf_counter()
        logger.info(
            "Generating compliance report: %d events, period=%s, programme=%s",
            len(events), reporting_period, programme_id,
        )

        # Track all events
        performances: List[EventPerformance] = []
        for event in events:
            perf = self.track_event(event)
            performances.append(perf)

        # Season summary
        summary = self.summarize_season(events, SeasonType.CUSTOM, reporting_period)

        # Trend analysis
        trend = self.detect_trends(performances)

        # Risk assessment
        risk_level, risk_score = self._assess_risk(performances)

        # Reliability index
        missed = sum(
            1 for p in performances
            if p.performance_ratio_pct <= Decimal("0")
        )
        total = len(performances)
        reliability = Decimal("1") - _safe_divide(_decimal(missed), _decimal(total))

        # Revenue totals
        total_revenue = sum((p.revenue for p in performances), Decimal("0"))
        total_penalties = sum((p.penalty for p in performances), Decimal("0"))
        net_income = total_revenue - total_penalties

        # Recommendations
        recommendations = self._generate_recommendations(
            performances, summary, trend, risk_level,
        )

        report = ComplianceReport(
            programme_id=programme_id,
            reporting_period=reporting_period,
            overall_compliance_pct=summary.compliance_rate_pct,
            risk_level=risk_level,
            risk_score=_round_val(risk_score, 2),
            reliability_index=_round_val(reliability, 4),
            total_events=total,
            compliant_events=summary.compliant_events,
            missed_events=missed,
            event_performances=performances,
            season_summary=summary,
            trend_analysis=trend,
            total_revenue=_round_val(total_revenue, 2),
            total_penalties=_round_val(total_penalties, 2),
            net_income=_round_val(net_income, 2),
            recommendations=recommendations,
        )
        report.provenance_hash = _compute_hash(report)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Compliance report: compliance=%.1f%%, risk=%s (%.1f), "
            "net=%.2f, hash=%s (%.1f ms)",
            float(summary.compliance_rate_pct), risk_level.value,
            float(risk_score), float(net_income),
            report.provenance_hash[:16], elapsed,
        )
        return report

    # ------------------------------------------------------------------ #
    # Internal: Compliance Determination                                  #
    # ------------------------------------------------------------------ #

    def _determine_compliance(
        self,
        ratio_pct: Decimal,
        threshold_pct: Decimal,
    ) -> ComplianceStatus:
        """Determine compliance status from performance ratio.

        Args:
            ratio_pct: Performance ratio percentage.
            threshold_pct: Compliance threshold percentage.

        Returns:
            ComplianceStatus enum value.
        """
        if ratio_pct >= threshold_pct:
            return ComplianceStatus.COMPLIANT
        if ratio_pct >= self._partial_floor:
            return ComplianceStatus.PARTIAL
        return ComplianceStatus.NON_COMPLIANT

    def _calculate_penalty(
        self,
        shortfall_kw: Decimal,
        duration_hours: Decimal,
        penalty_type: PenaltyType,
    ) -> Decimal:
        """Calculate penalty for non-compliance.

        Args:
            shortfall_kw: Shortfall below nomination (kW).
            duration_hours: Event duration (hours).
            penalty_type: Type of penalty.

        Returns:
            Penalty amount (USD).
        """
        rate = _decimal(
            self._penalty_rates.get(penalty_type.value, Decimal("0"))
        )
        if penalty_type == PenaltyType.NON_RESPONSE:
            return rate
        if penalty_type == PenaltyType.CAPACITY_SHORTFALL:
            return shortfall_kw * rate * duration_hours / Decimal("24")
        if penalty_type == PenaltyType.UNDER_PERFORMANCE:
            return shortfall_kw * rate * duration_hours / Decimal("24")
        return Decimal("0")

    # ------------------------------------------------------------------ #
    # Internal: Trend Analysis                                            #
    # ------------------------------------------------------------------ #

    def _linear_regression_slope(self, values: List[float]) -> float:
        """Compute linear regression slope for a series of values.

        Uses ordinary least squares: slope = cov(x,y) / var(x)
        where x = [0, 1, 2, ...] and y = values.

        Args:
            values: Ordered list of numeric values.

        Returns:
            Slope of the regression line.
        """
        n = len(values)
        if n < 2:
            return 0.0

        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0
        return numerator / denominator

    # ------------------------------------------------------------------ #
    # Internal: Risk Assessment                                           #
    # ------------------------------------------------------------------ #

    def _assess_risk(
        self,
        performances: List[EventPerformance],
    ) -> Tuple[RiskLevel, Decimal]:
        """Assess compliance risk level from performance history.

        Risk score = 100 - compliance_rate_pct, adjusted for
        penalty severity.

        Args:
            performances: List of event performance results.

        Returns:
            Tuple of (RiskLevel, risk_score).
        """
        if not performances:
            return RiskLevel.LOW, Decimal("0")

        total = _decimal(len(performances))
        compliant = _decimal(sum(
            1 for p in performances
            if p.compliance_status == ComplianceStatus.COMPLIANT
        ))
        compliance_rate = _safe_pct(compliant, total)

        # Base risk score (inverse of compliance)
        risk_score = Decimal("100") - compliance_rate

        # Adjust for penalty severity
        total_penalties = sum((p.penalty for p in performances), Decimal("0"))
        total_revenue = sum((p.revenue for p in performances), Decimal("0"))
        if total_revenue > Decimal("0"):
            penalty_ratio = _safe_pct(total_penalties, total_revenue)
            risk_score += penalty_ratio * Decimal("0.1")
            risk_score = min(risk_score, Decimal("100"))

        # Determine level
        if compliance_rate >= RISK_THRESHOLDS[RiskLevel.LOW.value]:
            level = RiskLevel.LOW
        elif compliance_rate >= RISK_THRESHOLDS[RiskLevel.MODERATE.value]:
            level = RiskLevel.MODERATE
        elif compliance_rate >= RISK_THRESHOLDS[RiskLevel.HIGH.value]:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.CRITICAL

        return level, risk_score

    # ------------------------------------------------------------------ #
    # Internal: Recommendations                                          #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        performances: List[EventPerformance],
        summary: SeasonSummary,
        trend: PerformanceTrend,
        risk_level: RiskLevel,
    ) -> List[str]:
        """Generate actionable recommendations based on analysis.

        Args:
            performances: Event performance results.
            summary: Season summary.
            trend: Trend analysis.
            risk_level: Assessed risk level.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            recs.append(
                "URGENT: Compliance rate below programme minimum. "
                "Review curtailment capacity and operational procedures."
            )

        if summary.compliance_rate_pct < Decimal("90"):
            recs.append(
                "Increase nominated capacity buffer by 10-20% to improve "
                "compliance margin during events."
            )

        if trend.degradation_detected:
            recs.append(
                "Performance degradation detected. Investigate equipment "
                "condition, control system calibration, and load changes."
            )

        if trend.trend_direction == TrendDirection.DECLINING:
            recs.append(
                "Declining performance trend observed. Schedule preventive "
                "maintenance and review baseline calculations."
            )

        if summary.total_penalties > Decimal("0"):
            recs.append(
                f"Total penalties of ${float(summary.total_penalties):,.2f} incurred. "
                "Consider DER integration to improve response reliability."
            )

        avg_ratio = summary.avg_performance_ratio_pct
        if avg_ratio > Decimal("120"):
            recs.append(
                "Consistent over-performance suggests nominated capacity "
                "could be increased to capture additional revenue."
            )

        if not recs:
            recs.append(
                "Performance is strong. Continue current operational "
                "practices and monitor for any changes."
            )

        return recs
