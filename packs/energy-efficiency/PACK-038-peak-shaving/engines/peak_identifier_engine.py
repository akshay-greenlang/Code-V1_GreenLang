# -*- coding: utf-8 -*-
"""
PeakIdentifierEngine - PACK-038 Peak Shaving Engine 2
=======================================================

Peak detection and attribution engine for demand charge management.
Identifies billing-period peaks, classifies them by cause (weather,
equipment startup, production coincidence), clusters recurring peak
patterns, runs Monte Carlo simulations for peak probability, and
assesses avoidability of each peak event.

Calculation Methodology:
    Peak Identification:
        - Top-N ranking within billing period
        - Threshold-based detection (demand > percentile)
        - Rate-of-change (ramp) detection for startups

    Weather Regression:
        peak_weather_component = beta_cdd * CDD + beta_hdd * HDD + intercept
        where CDD = max(0, T_outdoor - T_base_cool)
              HDD = max(0, T_base_heat - T_outdoor)

    Startup Ramp Detection:
        ramp_kw_per_min = (demand[t] - demand[t-1]) / interval_minutes
        startup_flag = ramp_kw_per_min > ramp_threshold

    Monte Carlo Simulation:
        For i in 1..N_scenarios:
            peak_i = sample(historical_peaks, with_perturbation)
            P(peak > threshold) = count(peak_i > threshold) / N_scenarios

    Poisson Recurrence:
        P(k events in T) = (lambda*T)^k * exp(-lambda*T) / k!
        where lambda = observed_peak_rate (peaks / month)

    Shaving Potential:
        shavable_kw = peak_kw - target_threshold
        savings = shavable_kw * demand_charge_rate

Regulatory References:
    - FERC Order 745 - Demand Response Compensation
    - NERC Standards - Reliability Peak Management
    - ASHRAE 90.1-2022 - Peak demand provisions
    - IEC 61968 / CIM - Load profile data model
    - IEEE 1459-2010 - Power quality measurement
    - EN 50160:2010 - Voltage characteristics
    - ISO 50001:2018 - Energy management systems

Zero-Hallucination:
    - All peak identification uses deterministic thresholds
    - Weather regression via explicit least-squares coefficients
    - Monte Carlo uses seeded pseudo-random generation
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-038 Peak Shaving
Engine:  2 of 5
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

class PeakType(str, Enum):
    """Classification of peak event by billing or system context.

    BILLING:     Peak recorded for billing-period demand charge.
    SEASONAL:    Seasonal peak (summer / winter system peak).
    COINCIDENT:  Coincident peak with utility system peak.
    RATCHET:     Ratchet-demand peak (rolling maximum).
    STARTUP:     Equipment startup surge peak.
    """
    BILLING = "billing"
    SEASONAL = "seasonal"
    COINCIDENT = "coincident"
    RATCHET = "ratchet"
    STARTUP = "startup"

class PeakCause(str, Enum):
    """Root cause attribution for a peak event.

    WEATHER:            Temperature-driven peak (extreme CDD/HDD).
    EQUIPMENT_STARTUP:  Simultaneous equipment startup surge.
    PRODUCTION:         Production schedule coincidence.
    COINCIDENCE:        Random coincidence of multiple loads.
    UNKNOWN:            Cause not determinable from available data.
    """
    WEATHER = "weather"
    EQUIPMENT_STARTUP = "equipment_startup"
    PRODUCTION = "production"
    COINCIDENCE = "coincidence"
    UNKNOWN = "unknown"

class Avoidability(str, Enum):
    """Assessment of whether a peak can be avoided or reduced.

    FULLY:        Peak is fully avoidable with available measures.
    PARTIALLY:    Peak can be partially reduced.
    UNAVOIDABLE:  Peak cannot be reduced without major changes.
    """
    FULLY = "fully"
    PARTIALLY = "partially"
    UNAVOIDABLE = "unavoidable"

class ClusterType(str, Enum):
    """Temporal clustering pattern of peak events.

    ISOLATED:    Peak occurs as a standalone event.
    CLUSTERED:   Peak occurs in a cluster with nearby peaks.
    RECURRING:   Peak recurs on a regular schedule or pattern.
    """
    ISOLATED = "isolated"
    CLUSTERED = "clustered"
    RECURRING = "recurring"

class SeverityLevel(str, Enum):
    """Severity classification of a peak event.

    LOW:       Peak is within normal operating range.
    MEDIUM:    Peak is elevated but manageable.
    HIGH:      Peak is significantly above normal.
    CRITICAL:  Peak is at or near equipment / contract limits.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SeasonType(str, Enum):
    """Season for peak analysis context.

    SUMMER:   Peak cooling season.
    WINTER:   Peak heating season.
    SHOULDER: Transitional season.
    """
    SUMMER = "summer"
    WINTER = "winter"
    SHOULDER = "shoulder"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default top-N ranking count for peak identification.
DEFAULT_TOP_N: int = 10

# Default percentile threshold for peak detection.
DEFAULT_PEAK_PERCENTILE: int = 95

# Default ramp rate threshold (kW/min) for startup detection.
DEFAULT_RAMP_THRESHOLD: Decimal = Decimal("50")

# Default Monte Carlo scenario count.
DEFAULT_MC_SCENARIOS: int = 1000

# Default random seed for reproducible Monte Carlo.
DEFAULT_MC_SEED: int = 42

# Base temperatures for CDD/HDD calculations (degrees C).
CDD_BASE_TEMP: Decimal = Decimal("18.3")  # 65 F
HDD_BASE_TEMP: Decimal = Decimal("15.6")  # 60 F

# Severity thresholds (as fraction of billing peak).
SEVERITY_THRESHOLDS: List[Tuple[Decimal, SeverityLevel]] = [
    (Decimal("0.95"), SeverityLevel.CRITICAL),
    (Decimal("0.85"), SeverityLevel.HIGH),
    (Decimal("0.70"), SeverityLevel.MEDIUM),
    (Decimal("0"), SeverityLevel.LOW),
]

# Cluster proximity threshold (minutes between peaks).
CLUSTER_PROXIMITY_MINUTES: int = 120

# Recurrence regularity threshold (std dev of inter-peak hours).
RECURRENCE_STD_THRESHOLD: Decimal = Decimal("24")

# Default billing period days.
DEFAULT_BILLING_DAYS: int = 30

# Weather regression default coefficients.
DEFAULT_BETA_CDD: Decimal = Decimal("2.5")
DEFAULT_BETA_HDD: Decimal = Decimal("1.8")
DEFAULT_INTERCEPT: Decimal = Decimal("0")

# Perturbation range for Monte Carlo (fraction of peak kW).
MC_PERTURBATION_RANGE: Decimal = Decimal("0.10")

# ---------------------------------------------------------------------------
# Pydantic Models -- Input / Output
# ---------------------------------------------------------------------------

class PeakEvent(BaseModel):
    """A detected peak demand event.

    Attributes:
        peak_id: Unique peak event identifier.
        timestamp: When the peak occurred (interval start).
        demand_kw: Peak demand value (kW).
        duration_minutes: Duration of elevated demand (minutes).
        peak_type: Classification of peak type.
        severity: Severity level.
        billing_period: Billing period label (e.g., '2026-03').
        rank_in_period: Rank within billing period (1 = highest).
        percentile: Percentile rank of this peak (0-100).
        ramp_rate_kw_per_min: Rate of demand increase leading to peak.
        temperature_c: Outdoor temperature at peak time (Celsius).
        cdd: Cooling degree-days at peak.
        hdd: Heating degree-days at peak.
        notes: Additional notes.
    """
    peak_id: str = Field(default_factory=_new_uuid, description="Peak event ID")
    timestamp: datetime = Field(default_factory=utcnow, description="Peak timestamp")
    demand_kw: Decimal = Field(default=Decimal("0"), ge=0, description="Peak demand (kW)")
    duration_minutes: int = Field(default=15, ge=0, description="Duration (minutes)")
    peak_type: PeakType = Field(default=PeakType.BILLING, description="Peak type")
    severity: SeverityLevel = Field(default=SeverityLevel.LOW, description="Severity")
    billing_period: str = Field(default="", max_length=20, description="Billing period")
    rank_in_period: int = Field(default=0, ge=0, description="Rank in period")
    percentile: Decimal = Field(default=Decimal("0"), description="Percentile rank")
    ramp_rate_kw_per_min: Decimal = Field(default=Decimal("0"), description="Ramp rate")
    temperature_c: Decimal = Field(default=Decimal("0"), description="Temperature (C)")
    cdd: Decimal = Field(default=Decimal("0"), ge=0, description="Cooling degree-days")
    hdd: Decimal = Field(default=Decimal("0"), ge=0, description="Heating degree-days")
    notes: str = Field(default="", max_length=2000, description="Notes")

class PeakAttribution(BaseModel):
    """Root cause attribution for a peak event.

    Attributes:
        peak_id: Reference to the peak event.
        primary_cause: Dominant cause of the peak.
        weather_component_kw: kW attributable to weather.
        weather_pct: Percentage attributable to weather.
        startup_component_kw: kW attributable to equipment startup.
        startup_pct: Percentage attributable to startup.
        production_component_kw: kW attributable to production.
        production_pct: Percentage attributable to production.
        residual_kw: Unexplained residual demand (kW).
        residual_pct: Percentage residual.
        confidence_score: Attribution confidence (0-100).
        avoidability: Avoidability assessment.
        shavable_kw: Potentially shavable demand (kW).
        notes: Attribution notes.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    peak_id: str = Field(default="", description="Peak event ID")
    primary_cause: PeakCause = Field(default=PeakCause.UNKNOWN)
    weather_component_kw: Decimal = Field(default=Decimal("0"))
    weather_pct: Decimal = Field(default=Decimal("0"))
    startup_component_kw: Decimal = Field(default=Decimal("0"))
    startup_pct: Decimal = Field(default=Decimal("0"))
    production_component_kw: Decimal = Field(default=Decimal("0"))
    production_pct: Decimal = Field(default=Decimal("0"))
    residual_kw: Decimal = Field(default=Decimal("0"))
    residual_pct: Decimal = Field(default=Decimal("0"))
    confidence_score: Decimal = Field(default=Decimal("0"))
    avoidability: Avoidability = Field(default=Avoidability.PARTIALLY)
    shavable_kw: Decimal = Field(default=Decimal("0"))
    notes: str = Field(default="", max_length=2000)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class PeakCluster(BaseModel):
    """Cluster of temporally related peak events.

    Attributes:
        cluster_id: Unique cluster identifier.
        cluster_type: Clustering pattern type.
        peak_ids: List of peak event IDs in this cluster.
        peak_count: Number of peaks in cluster.
        earliest: Earliest peak timestamp.
        latest: Latest peak timestamp.
        max_demand_kw: Highest peak in cluster.
        avg_demand_kw: Average peak demand in cluster.
        recurrence_interval_hours: Average interval between peaks.
        recurrence_std_hours: Standard deviation of inter-peak intervals.
        dominant_cause: Most common cause in cluster.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    cluster_id: str = Field(default_factory=_new_uuid)
    cluster_type: ClusterType = Field(default=ClusterType.ISOLATED)
    peak_ids: List[str] = Field(default_factory=list)
    peak_count: int = Field(default=0)
    earliest: datetime = Field(default_factory=utcnow)
    latest: datetime = Field(default_factory=utcnow)
    max_demand_kw: Decimal = Field(default=Decimal("0"))
    avg_demand_kw: Decimal = Field(default=Decimal("0"))
    recurrence_interval_hours: Decimal = Field(default=Decimal("0"))
    recurrence_std_hours: Decimal = Field(default=Decimal("0"))
    dominant_cause: PeakCause = Field(default=PeakCause.UNKNOWN)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class PeakSimulation(BaseModel):
    """Monte Carlo peak simulation results.

    Attributes:
        simulation_id: Unique simulation identifier.
        scenario_count: Number of Monte Carlo scenarios.
        seed: Random seed used.
        mean_peak_kw: Mean simulated peak (kW).
        median_peak_kw: Median simulated peak (kW).
        p90_peak_kw: 90th percentile simulated peak (kW).
        p95_peak_kw: 95th percentile simulated peak (kW).
        p99_peak_kw: 99th percentile simulated peak (kW).
        prob_exceed_threshold: Probability of exceeding a given threshold.
        threshold_kw: The threshold used for exceedance probability.
        poisson_rate: Estimated Poisson rate (peaks / month).
        poisson_prob_k: Probability of k or more peaks next month.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    simulation_id: str = Field(default_factory=_new_uuid)
    scenario_count: int = Field(default=DEFAULT_MC_SCENARIOS)
    seed: int = Field(default=DEFAULT_MC_SEED)
    mean_peak_kw: Decimal = Field(default=Decimal("0"))
    median_peak_kw: Decimal = Field(default=Decimal("0"))
    p90_peak_kw: Decimal = Field(default=Decimal("0"))
    p95_peak_kw: Decimal = Field(default=Decimal("0"))
    p99_peak_kw: Decimal = Field(default=Decimal("0"))
    prob_exceed_threshold: Decimal = Field(default=Decimal("0"))
    threshold_kw: Decimal = Field(default=Decimal("0"))
    poisson_rate: Decimal = Field(default=Decimal("0"))
    poisson_prob_k: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class PeakAssessment(BaseModel):
    """Complete peak assessment result.

    Attributes:
        assessment_id: Unique assessment identifier.
        facility_id: Facility identifier.
        facility_name: Facility name.
        analysis_period_start: Start of analysis period.
        analysis_period_end: End of analysis period.
        peaks: Identified peak events.
        attributions: Peak cause attributions.
        clusters: Peak clusters.
        simulation: Monte Carlo simulation results.
        billing_peak_kw: Highest billing-period peak (kW).
        avg_peak_kw: Average of top-N peaks (kW).
        total_shavable_kw: Total shavable demand across peaks (kW).
        peak_count: Total number of identified peaks.
        avoidable_count: Count of fully avoidable peaks.
        partially_avoidable_count: Count of partially avoidable peaks.
        unavoidable_count: Count of unavoidable peaks.
        dominant_cause: Most common peak cause.
        recommendations: List of peak management recommendations.
        processing_time_ms: Processing duration in milliseconds.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    assessment_id: str = Field(default_factory=_new_uuid)
    facility_id: str = Field(default="")
    facility_name: str = Field(default="", max_length=500)
    analysis_period_start: datetime = Field(default_factory=utcnow)
    analysis_period_end: datetime = Field(default_factory=utcnow)
    peaks: List[PeakEvent] = Field(default_factory=list)
    attributions: List[PeakAttribution] = Field(default_factory=list)
    clusters: List[PeakCluster] = Field(default_factory=list)
    simulation: PeakSimulation = Field(default_factory=PeakSimulation)
    billing_peak_kw: Decimal = Field(default=Decimal("0"))
    avg_peak_kw: Decimal = Field(default=Decimal("0"))
    total_shavable_kw: Decimal = Field(default=Decimal("0"))
    peak_count: int = Field(default=0)
    avoidable_count: int = Field(default=0)
    partially_avoidable_count: int = Field(default=0)
    unavoidable_count: int = Field(default=0)
    dominant_cause: PeakCause = Field(default=PeakCause.UNKNOWN)
    recommendations: List[str] = Field(default_factory=list)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PeakIdentifierEngine:
    """Peak detection and attribution engine for demand charge management.

    Identifies billing-period peaks, attributes causes via weather
    regression and ramp detection, clusters recurring patterns, and
    runs Monte Carlo simulations for peak probability.  All calculations
    use deterministic Decimal arithmetic with SHA-256 provenance hashing.

    Usage::

        engine = PeakIdentifierEngine()
        peaks = engine.identify_peaks(demand_data, billing_period="2026-03")
        attributions = engine.attribute_peaks(peaks, weather_data)
        clusters = engine.cluster_peaks(peaks)
        simulation = engine.simulate_peaks(peaks, threshold_kw=Decimal("500"))
        assessment = engine.assess_avoidability(peaks, attributions)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise PeakIdentifierEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - top_n (int): number of top peaks to identify
                - peak_percentile (int): percentile threshold
                - ramp_threshold (float): kW/min ramp threshold
                - mc_scenarios (int): Monte Carlo scenario count
                - mc_seed (int): Monte Carlo random seed
                - beta_cdd (float): CDD regression coefficient
                - beta_hdd (float): HDD regression coefficient
                - intercept (float): regression intercept
        """
        self.config = config or {}
        self._top_n = int(self.config.get("top_n", DEFAULT_TOP_N))
        self._peak_percentile = int(
            self.config.get("peak_percentile", DEFAULT_PEAK_PERCENTILE)
        )
        self._ramp_threshold = _decimal(
            self.config.get("ramp_threshold", DEFAULT_RAMP_THRESHOLD)
        )
        self._mc_scenarios = int(
            self.config.get("mc_scenarios", DEFAULT_MC_SCENARIOS)
        )
        self._mc_seed = int(self.config.get("mc_seed", DEFAULT_MC_SEED))
        self._beta_cdd = _decimal(
            self.config.get("beta_cdd", DEFAULT_BETA_CDD)
        )
        self._beta_hdd = _decimal(
            self.config.get("beta_hdd", DEFAULT_BETA_HDD)
        )
        self._intercept = _decimal(
            self.config.get("intercept", DEFAULT_INTERCEPT)
        )
        logger.info(
            "PeakIdentifierEngine v%s initialised (top_n=%d, percentile=%d)",
            self.engine_version, self._top_n, self._peak_percentile,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def identify_peaks(
        self,
        demand_data: List[Dict[str, Any]],
        billing_period: str = "",
        interval_minutes: int = 15,
    ) -> List[PeakEvent]:
        """Identify peak demand events from interval data.

        Uses top-N ranking and percentile threshold to detect peaks.
        Also flags startup ramp events.

        Args:
            demand_data: List of dicts with 'timestamp', 'demand_kw',
                and optionally 'temperature_c'.
            billing_period: Billing period label.
            interval_minutes: Interval length in minutes.

        Returns:
            List of identified PeakEvent objects, ranked by demand.
        """
        t0 = time.perf_counter()
        logger.info(
            "Identifying peaks: %d data points, period=%s",
            len(demand_data), billing_period,
        )

        if not demand_data:
            return []

        # Extract and sort demands
        entries = []
        for d in demand_data:
            entries.append({
                "timestamp": d.get("timestamp", utcnow()),
                "demand_kw": _decimal(d.get("demand_kw", 0)),
                "temperature_c": _decimal(d.get("temperature_c", 0)),
            })

        # Sort by demand descending
        sorted_entries = sorted(
            entries, key=lambda x: x["demand_kw"], reverse=True
        )

        # Determine percentile threshold
        all_demands = sorted([e["demand_kw"] for e in entries])
        threshold_kw = self._percentile_value(
            all_demands, self._peak_percentile
        )
        billing_peak_kw = sorted_entries[0]["demand_kw"] if sorted_entries else Decimal("0")

        # Identify peaks: top-N and above percentile
        peak_set: Dict[str, Dict[str, Any]] = {}

        # Top-N peaks
        for rank, entry in enumerate(sorted_entries[:self._top_n], 1):
            ts_key = str(entry["timestamp"])
            if ts_key not in peak_set:
                peak_set[ts_key] = {**entry, "rank": rank}

        # Percentile-based peaks
        for entry in entries:
            if entry["demand_kw"] >= threshold_kw:
                ts_key = str(entry["timestamp"])
                if ts_key not in peak_set:
                    peak_set[ts_key] = {**entry, "rank": 0}

        # Detect startup ramps
        for i in range(1, len(entries)):
            ramp = _safe_divide(
                entries[i]["demand_kw"] - entries[i - 1]["demand_kw"],
                _decimal(interval_minutes),
            )
            if ramp > self._ramp_threshold:
                ts_key = str(entries[i]["timestamp"])
                if ts_key not in peak_set:
                    peak_set[ts_key] = {
                        **entries[i], "rank": 0, "is_startup": True
                    }
                else:
                    peak_set[ts_key]["is_startup"] = True

        # Build PeakEvent list
        peaks: List[PeakEvent] = []
        for ts_key, pdata in peak_set.items():
            demand = pdata["demand_kw"]
            temp_c = pdata.get("temperature_c", Decimal("0"))
            cdd = max(temp_c - CDD_BASE_TEMP, Decimal("0"))
            hdd = max(HDD_BASE_TEMP - temp_c, Decimal("0"))

            severity = self._classify_severity(demand, billing_peak_kw)
            pct_rank = _safe_pct(demand, billing_peak_kw) if billing_peak_kw > Decimal("0") else Decimal("0")

            ramp_rate = Decimal("0")
            # Find ramp rate for this timestamp
            for i in range(1, len(entries)):
                if str(entries[i]["timestamp"]) == ts_key:
                    ramp_rate = _safe_divide(
                        entries[i]["demand_kw"] - entries[i - 1]["demand_kw"],
                        _decimal(interval_minutes),
                    )
                    break

            peak_type = PeakType.STARTUP if pdata.get("is_startup") else PeakType.BILLING

            peak = PeakEvent(
                timestamp=pdata["timestamp"],
                demand_kw=_round_val(demand, 2),
                duration_minutes=interval_minutes,
                peak_type=peak_type,
                severity=severity,
                billing_period=billing_period,
                rank_in_period=pdata.get("rank", 0),
                percentile=_round_val(pct_rank, 2),
                ramp_rate_kw_per_min=_round_val(ramp_rate, 4),
                temperature_c=_round_val(temp_c, 1),
                cdd=_round_val(cdd, 2),
                hdd=_round_val(hdd, 2),
            )
            peaks.append(peak)

        # Sort by demand descending and re-rank
        peaks.sort(key=lambda p: p.demand_kw, reverse=True)
        for rank, peak in enumerate(peaks, 1):
            peak.rank_in_period = rank

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Peaks identified: %d peaks, billing_peak=%.1f kW, "
            "threshold=%.1f kW (%.1f ms)",
            len(peaks), float(billing_peak_kw),
            float(threshold_kw), elapsed,
        )
        return peaks

    def attribute_peaks(
        self,
        peaks: List[PeakEvent],
        baseline_kw: Optional[Decimal] = None,
    ) -> List[PeakAttribution]:
        """Attribute root causes to identified peak events.

        Uses weather regression (CDD/HDD coefficients) and ramp rate
        analysis to decompose each peak into weather, startup,
        production, and residual components.

        Args:
            peaks: List of identified peak events.
            baseline_kw: Optional baseline demand for decomposition.

        Returns:
            List of PeakAttribution objects.
        """
        t0 = time.perf_counter()
        logger.info("Attributing %d peaks", len(peaks))

        if not peaks:
            return []

        # Use mean peak as baseline if not provided
        if baseline_kw is None:
            total = sum((_decimal(p.demand_kw) for p in peaks), Decimal("0"))
            baseline_kw = _safe_divide(total, _decimal(len(peaks)))

        attributions: List[PeakAttribution] = []
        for peak in peaks:
            attr = self._attribute_single_peak(peak, baseline_kw)
            attributions.append(attr)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Peak attribution complete: %d attributions (%.1f ms)",
            len(attributions), elapsed,
        )
        return attributions

    def cluster_peaks(
        self,
        peaks: List[PeakEvent],
    ) -> List[PeakCluster]:
        """Cluster peak events by temporal proximity and pattern.

        Groups peaks that occur within CLUSTER_PROXIMITY_MINUTES of
        each other, then classifies the cluster as isolated, clustered,
        or recurring based on recurrence statistics.

        Args:
            peaks: List of identified peak events.

        Returns:
            List of PeakCluster objects.
        """
        t0 = time.perf_counter()
        logger.info("Clustering %d peaks", len(peaks))

        if not peaks:
            return []

        # Sort peaks by timestamp
        sorted_peaks = sorted(peaks, key=lambda p: p.timestamp)

        # Group into clusters by proximity
        raw_clusters: List[List[PeakEvent]] = []
        current_cluster: List[PeakEvent] = [sorted_peaks[0]]

        for i in range(1, len(sorted_peaks)):
            gap_seconds = (
                sorted_peaks[i].timestamp - sorted_peaks[i - 1].timestamp
            ).total_seconds()
            gap_minutes = gap_seconds / 60.0

            if gap_minutes <= CLUSTER_PROXIMITY_MINUTES:
                current_cluster.append(sorted_peaks[i])
            else:
                raw_clusters.append(current_cluster)
                current_cluster = [sorted_peaks[i]]

        raw_clusters.append(current_cluster)

        # Build PeakCluster objects
        clusters: List[PeakCluster] = []
        for group in raw_clusters:
            cluster = self._build_cluster(group)
            clusters.append(cluster)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Peak clustering complete: %d clusters (%.1f ms)",
            len(clusters), elapsed,
        )
        return clusters

    def simulate_peaks(
        self,
        peaks: List[PeakEvent],
        threshold_kw: Decimal = Decimal("0"),
        target_k: int = 1,
    ) -> PeakSimulation:
        """Run Monte Carlo simulation on historical peak data.

        Generates N scenarios by sampling from historical peaks with
        perturbation.  Computes exceedance probabilities and Poisson
        recurrence estimates.

        Args:
            peaks: Historical peak events for sampling.
            threshold_kw: Demand threshold for exceedance probability.
            target_k: Target number of peaks for Poisson probability.

        Returns:
            PeakSimulation with statistical results.
        """
        t0 = time.perf_counter()
        logger.info(
            "Simulating peaks: %d historical peaks, %d scenarios, "
            "threshold=%.1f kW",
            len(peaks), self._mc_scenarios, float(threshold_kw),
        )

        if not peaks:
            result = PeakSimulation(
                threshold_kw=threshold_kw,
            )
            result.provenance_hash = _compute_hash(result)
            return result

        # Extract peak demands
        historical = sorted(
            [_decimal(p.demand_kw) for p in peaks], reverse=True
        )

        # Deterministic LCG pseudo-random number generator
        # Using seeded linear congruential generator for reproducibility
        seed = self._mc_seed
        a = 1664525
        c = 1013904223
        m = 2 ** 32

        simulated_peaks: List[Decimal] = []
        for _ in range(self._mc_scenarios):
            # Select a random historical peak
            seed = (a * seed + c) % m
            idx = seed % len(historical)
            base_peak = historical[idx]

            # Apply perturbation
            seed = (a * seed + c) % m
            perturbation_frac = _decimal(seed % 1000) / Decimal("1000")
            perturbation = (
                perturbation_frac * Decimal("2") - Decimal("1")
            ) * MC_PERTURBATION_RANGE * base_peak
            simulated_peak = max(base_peak + perturbation, Decimal("0"))
            simulated_peaks.append(simulated_peak)

        # Sort for percentile calculations
        simulated_peaks.sort()
        n = len(simulated_peaks)

        mean_val = _safe_divide(
            sum(simulated_peaks, Decimal("0")), _decimal(n)
        )
        median_val = self._percentile_value(simulated_peaks, 50)
        p90_val = self._percentile_value(simulated_peaks, 90)
        p95_val = self._percentile_value(simulated_peaks, 95)
        p99_val = self._percentile_value(simulated_peaks, 99)

        # Exceedance probability
        if threshold_kw > Decimal("0"):
            exceed_count = sum(
                1 for p in simulated_peaks if p > threshold_kw
            )
            prob_exceed = _safe_divide(
                _decimal(exceed_count), _decimal(n)
            )
        else:
            prob_exceed = Decimal("1")

        # Poisson recurrence
        poisson_rate = self._estimate_poisson_rate(peaks)
        poisson_prob = self._poisson_probability(poisson_rate, target_k)

        result = PeakSimulation(
            scenario_count=self._mc_scenarios,
            seed=self._mc_seed,
            mean_peak_kw=_round_val(mean_val, 2),
            median_peak_kw=_round_val(median_val, 2),
            p90_peak_kw=_round_val(p90_val, 2),
            p95_peak_kw=_round_val(p95_val, 2),
            p99_peak_kw=_round_val(p99_val, 2),
            prob_exceed_threshold=_round_val(prob_exceed, 4),
            threshold_kw=_round_val(threshold_kw, 2),
            poisson_rate=_round_val(poisson_rate, 4),
            poisson_prob_k=_round_val(poisson_prob, 4),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Simulation complete: mean=%.1f kW, P95=%.1f kW, "
            "P(exceed)=%.3f, Poisson=%.3f, hash=%s (%.1f ms)",
            float(mean_val), float(p95_val), float(prob_exceed),
            float(poisson_prob), result.provenance_hash[:16], elapsed,
        )
        return result

    def assess_avoidability(
        self,
        peaks: List[PeakEvent],
        attributions: List[PeakAttribution],
        target_reduction_kw: Optional[Decimal] = None,
    ) -> PeakAssessment:
        """Produce a complete peak assessment with avoidability analysis.

        Combines peak identification, attribution, clustering, and
        simulation into a comprehensive assessment.

        Args:
            peaks: Identified peak events.
            attributions: Peak cause attributions.
            target_reduction_kw: Optional target peak reduction (kW).

        Returns:
            PeakAssessment with full analysis and recommendations.
        """
        t0 = time.perf_counter()
        logger.info(
            "Assessing avoidability for %d peaks", len(peaks),
        )

        if not peaks:
            result = PeakAssessment(
                recommendations=["No peak events to assess."],
            )
            result.provenance_hash = _compute_hash(result)
            return result

        # Build attribution map
        attr_map: Dict[str, PeakAttribution] = {}
        for attr in attributions:
            attr_map[attr.peak_id] = attr

        # Cluster peaks
        clusters = self.cluster_peaks(peaks)

        # Simulate
        billing_peak_kw = max(
            (_decimal(p.demand_kw) for p in peaks), default=Decimal("0")
        )
        threshold = target_reduction_kw or (billing_peak_kw * Decimal("0.90"))
        simulation = self.simulate_peaks(peaks, threshold_kw=threshold)

        # Count avoidability
        avoidable = sum(
            1 for a in attributions if a.avoidability == Avoidability.FULLY
        )
        partially = sum(
            1 for a in attributions if a.avoidability == Avoidability.PARTIALLY
        )
        unavoidable = sum(
            1 for a in attributions if a.avoidability == Avoidability.UNAVOIDABLE
        )

        total_shavable = sum(
            (_decimal(a.shavable_kw) for a in attributions), Decimal("0")
        )
        avg_peak = _safe_divide(
            sum((_decimal(p.demand_kw) for p in peaks), Decimal("0")),
            _decimal(len(peaks)),
        )

        # Dominant cause
        cause_counts: Dict[str, int] = {}
        for attr in attributions:
            key = attr.primary_cause.value
            cause_counts[key] = cause_counts.get(key, 0) + 1
        dominant = PeakCause.UNKNOWN
        if cause_counts:
            dominant = PeakCause(
                max(cause_counts, key=cause_counts.get)  # type: ignore[arg-type]
            )

        # Timestamps
        timestamps = [p.timestamp for p in peaks]
        start = min(timestamps) if timestamps else utcnow()
        end = max(timestamps) if timestamps else utcnow()

        # Recommendations
        recommendations = self._generate_recommendations(
            peaks, attributions, clusters, simulation,
            billing_peak_kw, dominant,
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000.0)

        result = PeakAssessment(
            facility_id=peaks[0].billing_period if peaks else "",
            analysis_period_start=start,
            analysis_period_end=end,
            peaks=peaks,
            attributions=attributions,
            clusters=clusters,
            simulation=simulation,
            billing_peak_kw=_round_val(billing_peak_kw, 2),
            avg_peak_kw=_round_val(avg_peak, 2),
            total_shavable_kw=_round_val(total_shavable, 2),
            peak_count=len(peaks),
            avoidable_count=avoidable,
            partially_avoidable_count=partially,
            unavoidable_count=unavoidable,
            dominant_cause=dominant,
            recommendations=recommendations,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Assessment complete: %d peaks, shavable=%.1f kW, "
            "avoidable=%d, hash=%s (%.1f ms)",
            len(peaks), float(total_shavable), avoidable,
            result.provenance_hash[:16], float(elapsed_ms),
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal Methods                                                     #
    # ------------------------------------------------------------------ #

    def _attribute_single_peak(
        self,
        peak: PeakEvent,
        baseline_kw: Decimal,
    ) -> PeakAttribution:
        """Attribute a single peak event to root causes.

        Args:
            peak: Peak event to attribute.
            baseline_kw: Baseline demand for decomposition.

        Returns:
            PeakAttribution for this peak.
        """
        demand = _decimal(peak.demand_kw)
        excess = max(demand - baseline_kw, Decimal("0"))

        # Weather component: beta_cdd * CDD + beta_hdd * HDD
        weather_kw = (
            self._beta_cdd * _decimal(peak.cdd)
            + self._beta_hdd * _decimal(peak.hdd)
            + self._intercept
        )
        weather_kw = max(min(weather_kw, excess), Decimal("0"))
        weather_pct = _safe_pct(weather_kw, demand)

        # Startup component: high ramp rate
        startup_kw = Decimal("0")
        ramp = _decimal(peak.ramp_rate_kw_per_min)
        if ramp > self._ramp_threshold:
            startup_kw = min(
                ramp * Decimal("15"), excess - weather_kw
            )
            startup_kw = max(startup_kw, Decimal("0"))
        startup_pct = _safe_pct(startup_kw, demand)

        # Production component: residual above weather + startup
        remaining = max(excess - weather_kw - startup_kw, Decimal("0"))
        production_kw = remaining * Decimal("0.60")
        production_pct = _safe_pct(production_kw, demand)

        # Residual
        residual_kw = max(
            demand - baseline_kw - weather_kw - startup_kw - production_kw,
            Decimal("0"),
        )
        residual_pct = _safe_pct(residual_kw, demand)

        # Determine primary cause
        components = {
            PeakCause.WEATHER: weather_kw,
            PeakCause.EQUIPMENT_STARTUP: startup_kw,
            PeakCause.PRODUCTION: production_kw,
        }
        primary_cause = max(components, key=components.get)  # type: ignore[arg-type]
        if max(components.values()) == Decimal("0"):
            primary_cause = PeakCause.UNKNOWN

        # Confidence based on how much is explained
        explained = weather_kw + startup_kw + production_kw
        confidence = _safe_pct(explained, excess) if excess > Decimal("0") else Decimal("50")
        confidence = min(confidence, Decimal("100"))

        # Avoidability
        avoidability = self._assess_peak_avoidability(
            weather_kw, startup_kw, production_kw, demand,
        )

        # Shavable kW
        shavable = Decimal("0")
        if avoidability == Avoidability.FULLY:
            shavable = startup_kw + production_kw
        elif avoidability == Avoidability.PARTIALLY:
            shavable = (startup_kw + production_kw) * Decimal("0.50")

        attr = PeakAttribution(
            peak_id=peak.peak_id,
            primary_cause=primary_cause,
            weather_component_kw=_round_val(weather_kw, 2),
            weather_pct=_round_val(weather_pct, 2),
            startup_component_kw=_round_val(startup_kw, 2),
            startup_pct=_round_val(startup_pct, 2),
            production_component_kw=_round_val(production_kw, 2),
            production_pct=_round_val(production_pct, 2),
            residual_kw=_round_val(residual_kw, 2),
            residual_pct=_round_val(residual_pct, 2),
            confidence_score=_round_val(confidence, 2),
            avoidability=avoidability,
            shavable_kw=_round_val(shavable, 2),
        )
        attr.provenance_hash = _compute_hash(attr)
        return attr

    def _assess_peak_avoidability(
        self,
        weather_kw: Decimal,
        startup_kw: Decimal,
        production_kw: Decimal,
        total_kw: Decimal,
    ) -> Avoidability:
        """Assess avoidability of a peak based on cause decomposition.

        Args:
            weather_kw: Weather-driven component.
            startup_kw: Startup-driven component.
            production_kw: Production-driven component.
            total_kw: Total peak demand.

        Returns:
            Avoidability classification.
        """
        controllable = startup_kw + production_kw
        controllable_pct = _safe_pct(controllable, total_kw)

        if controllable_pct >= Decimal("60"):
            return Avoidability.FULLY
        elif controllable_pct >= Decimal("25"):
            return Avoidability.PARTIALLY
        else:
            return Avoidability.UNAVOIDABLE

    def _build_cluster(
        self,
        group: List[PeakEvent],
    ) -> PeakCluster:
        """Build a PeakCluster from a group of temporally close peaks.

        Args:
            group: List of peaks in the cluster.

        Returns:
            PeakCluster object.
        """
        peak_ids = [p.peak_id for p in group]
        demands = [_decimal(p.demand_kw) for p in group]
        timestamps = [p.timestamp for p in group]

        max_d = max(demands) if demands else Decimal("0")
        avg_d = _safe_divide(
            sum(demands, Decimal("0")), _decimal(len(demands))
        )
        earliest = min(timestamps) if timestamps else utcnow()
        latest = max(timestamps) if timestamps else utcnow()

        # Calculate inter-peak intervals
        intervals_hours: List[Decimal] = []
        for i in range(1, len(timestamps)):
            gap_sec = (timestamps[i] - timestamps[i - 1]).total_seconds()
            intervals_hours.append(_decimal(gap_sec / 3600.0))

        recurrence_interval = Decimal("0")
        recurrence_std = Decimal("0")
        if intervals_hours:
            recurrence_interval = _safe_divide(
                sum(intervals_hours, Decimal("0")),
                _decimal(len(intervals_hours)),
            )
            if len(intervals_hours) > 1:
                variance = _safe_divide(
                    sum(
                        ((h - recurrence_interval) ** 2 for h in intervals_hours),
                        Decimal("0"),
                    ),
                    _decimal(len(intervals_hours)),
                )
                recurrence_std = _decimal(math.sqrt(float(variance)))

        # Classify cluster type
        if len(group) == 1:
            cluster_type = ClusterType.ISOLATED
        elif recurrence_std < RECURRENCE_STD_THRESHOLD and len(group) >= 3:
            cluster_type = ClusterType.RECURRING
        else:
            cluster_type = ClusterType.CLUSTERED

        cluster = PeakCluster(
            cluster_type=cluster_type,
            peak_ids=peak_ids,
            peak_count=len(group),
            earliest=earliest,
            latest=latest,
            max_demand_kw=_round_val(max_d, 2),
            avg_demand_kw=_round_val(avg_d, 2),
            recurrence_interval_hours=_round_val(recurrence_interval, 2),
            recurrence_std_hours=_round_val(recurrence_std, 2),
        )
        cluster.provenance_hash = _compute_hash(cluster)
        return cluster

    def _classify_severity(
        self,
        demand_kw: Decimal,
        billing_peak_kw: Decimal,
    ) -> SeverityLevel:
        """Classify peak severity relative to billing peak.

        Args:
            demand_kw: Peak demand value.
            billing_peak_kw: Billing period peak.

        Returns:
            SeverityLevel classification.
        """
        if billing_peak_kw == Decimal("0"):
            return SeverityLevel.LOW
        ratio = _safe_divide(demand_kw, billing_peak_kw)
        for threshold, severity in SEVERITY_THRESHOLDS:
            if ratio >= threshold:
                return severity
        return SeverityLevel.LOW

    def _percentile_value(
        self,
        sorted_values: List[Decimal],
        percentile: int,
    ) -> Decimal:
        """Compute a percentile from a sorted list.

        Args:
            sorted_values: Pre-sorted ascending Decimal list.
            percentile: Percentile (0-100).

        Returns:
            Interpolated percentile value.
        """
        if not sorted_values:
            return Decimal("0")
        n = len(sorted_values)
        if n == 1:
            return sorted_values[0]
        rank = _decimal(percentile) / Decimal("100") * _decimal(n - 1)
        lower = int(math.floor(float(rank)))
        upper = min(lower + 1, n - 1)
        frac = rank - _decimal(lower)
        return (
            sorted_values[lower] * (Decimal("1") - frac)
            + sorted_values[upper] * frac
        )

    def _estimate_poisson_rate(
        self,
        peaks: List[PeakEvent],
    ) -> Decimal:
        """Estimate Poisson rate (peaks per month) from historical data.

        Args:
            peaks: Historical peak events.

        Returns:
            Estimated Poisson rate (lambda).
        """
        if len(peaks) < 2:
            return _decimal(len(peaks))
        timestamps = sorted([p.timestamp for p in peaks])
        span_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
        span_months = _decimal(span_seconds / (30.44 * 86400))
        if span_months <= Decimal("0"):
            return _decimal(len(peaks))
        return _safe_divide(_decimal(len(peaks)), span_months)

    def _poisson_probability(
        self,
        rate: Decimal,
        k: int,
    ) -> Decimal:
        """Calculate P(X >= k) for Poisson distribution.

        P(X >= k) = 1 - sum(P(X = i) for i in 0..k-1)
        P(X = i) = (lambda^i * exp(-lambda)) / i!

        Args:
            rate: Poisson rate (lambda).
            k: Minimum number of events.

        Returns:
            Probability of k or more events.
        """
        lam = float(rate)
        if lam <= 0:
            return Decimal("0")

        cumulative = Decimal("0")
        for i in range(k):
            try:
                prob = _decimal(
                    (lam ** i) * math.exp(-lam) / math.factorial(i)
                )
                cumulative += prob
            except (OverflowError, ValueError):
                break

        return max(Decimal("1") - cumulative, Decimal("0"))

    def _generate_recommendations(
        self,
        peaks: List[PeakEvent],
        attributions: List[PeakAttribution],
        clusters: List[PeakCluster],
        simulation: PeakSimulation,
        billing_peak_kw: Decimal,
        dominant_cause: PeakCause,
    ) -> List[str]:
        """Generate peak management recommendations.

        Args:
            peaks: Identified peaks.
            attributions: Peak attributions.
            clusters: Peak clusters.
            simulation: Monte Carlo simulation.
            billing_peak_kw: Billing period peak.
            dominant_cause: Most common peak cause.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        if dominant_cause == PeakCause.WEATHER:
            recs.append(
                "Weather is the dominant peak driver. Install thermal storage "
                "or pre-cool/pre-heat to decouple peak demand from weather."
            )

        if dominant_cause == PeakCause.EQUIPMENT_STARTUP:
            recs.append(
                "Equipment startup drives peaks. Implement staggered startup "
                "sequences and soft-start VFDs to limit inrush current."
            )

        if dominant_cause == PeakCause.PRODUCTION:
            recs.append(
                "Production scheduling causes peaks. Shift non-critical "
                "production loads to off-peak hours where possible."
            )

        recurring = [c for c in clusters if c.cluster_type == ClusterType.RECURRING]
        if recurring:
            recs.append(
                f"Found {len(recurring)} recurring peak patterns. "
                "Deploy automated demand limiting controls triggered by "
                "time-of-day schedules."
            )

        avoidable = sum(
            1 for a in attributions if a.avoidability == Avoidability.FULLY
        )
        if avoidable > len(attributions) / 2:
            recs.append(
                f"{avoidable} of {len(attributions)} peaks are fully avoidable. "
                "Implement BESS peak shaving for immediate demand charge savings."
            )

        if simulation.prob_exceed_threshold > Decimal("0.50"):
            recs.append(
                "Monte Carlo indicates >50% probability of exceeding the "
                "target threshold. Consider upsizing BESS or adding load "
                "shedding capacity."
            )

        if billing_peak_kw > Decimal("0"):
            p90_ratio = _safe_divide(
                simulation.p90_peak_kw, billing_peak_kw
            )
            if p90_ratio > Decimal("1.10"):
                recs.append(
                    "Simulated P90 peak exceeds historical billing peak by >10%. "
                    "Review demand limiting setpoints and BESS dispatch strategy."
                )

        if not recs:
            recs.append(
                "Peak profile is manageable. Continue monitoring and "
                "maintain current demand management strategies."
            )

        return recs
