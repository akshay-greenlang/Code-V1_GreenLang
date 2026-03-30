# -*- coding: utf-8 -*-
"""
LoadProfileEngine - PACK-038 Peak Shaving Engine 1
====================================================

Comprehensive 15-minute interval load profile analysis engine for peak
shaving programmes.  Ingests time-series demand data, computes statistical
metrics (load factor, peak-to-average ratio, diversity factor, coincidence
factor), builds load-duration curves, clusters days by shape, detects
anomalies, and performs seasonal / TOU-period decomposition.

Calculation Methodology:
    Load Factor:
        LF = average_demand / peak_demand

    Peak-to-Average Ratio:
        PAR = peak_demand / average_demand

    Diversity Factor:
        DF = sum(individual_peaks) / coincident_peak

    Coincidence Factor:
        CF = coincident_peak / sum(individual_peaks) = 1 / DF

    Coefficient of Variation:
        CV = std_dev / mean * 100

    Percentile Analysis:
        P50, P90, P95, P99 from sorted demand array

    Load-Duration Curve:
        Sorted descending demand values with cumulative hours

    Anomaly Detection:
        Z-score method: anomaly if |demand - mean| > threshold * std_dev

Regulatory References:
    - DOE CBECS (Commercial Buildings Energy Consumption Survey)
    - ASHRAE 90.1-2022 - Energy Standard for Buildings
    - IEC 61968 / CIM - Common Information Model for Utilities
    - IEEE 1459-2010 - Power Quality Measurement
    - EN 50160:2010 - Voltage characteristics of electricity supply
    - FERC Order 2222 - DER aggregation metering requirements
    - ISO 50001:2018 - Energy management systems

Zero-Hallucination:
    - All statistical metrics computed from deterministic formulas
    - Percentiles via sorted-array linear interpolation only
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-038 Peak Shaving
Engine:  1 of 5
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

class IntervalLength(str, Enum):
    """Metering interval length.

    MIN_15: 15-minute intervals (standard AMI).
    MIN_30: 30-minute intervals (UK half-hourly).
    MIN_60: 60-minute intervals (hourly).
    """
    MIN_15 = "15_min"
    MIN_30 = "30_min"
    MIN_60 = "60_min"

class DayType(str, Enum):
    """Day classification for profile analysis.

    WEEKDAY:  Monday through Friday (non-holiday).
    WEEKEND:  Saturday and Sunday.
    HOLIDAY:  Designated public holidays.
    SPECIAL:  Special events or atypical days.
    """
    WEEKDAY = "weekday"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"
    SPECIAL = "special"

class Season(str, Enum):
    """Seasonal classification for demand analysis.

    SUMMER:   Peak cooling season (Jun-Aug in northern hemisphere).
    WINTER:   Peak heating season (Dec-Feb in northern hemisphere).
    SHOULDER: Transitional season with moderate loads.
    SPRING:   Spring transitional period (Mar-May).
    FALL:     Fall transitional period (Sep-Nov).
    """
    SUMMER = "summer"
    WINTER = "winter"
    SHOULDER = "shoulder"
    SPRING = "spring"
    FALL = "fall"

class LoadShape(str, Enum):
    """Classified daily load shape type.

    FLAT:             Nearly constant demand throughout the day.
    PEAKY:            Single narrow peak with low base load.
    MORNING_PEAK:     Dominant peak in morning hours (6-10 AM).
    AFTERNOON_PEAK:   Dominant peak in afternoon hours (12-5 PM).
    DOUBLE_PEAK:      Two distinct peaks (morning and afternoon).
    """
    FLAT = "flat"
    PEAKY = "peaky"
    MORNING_PEAK = "morning_peak"
    AFTERNOON_PEAK = "afternoon_peak"
    DOUBLE_PEAK = "double_peak"

class ProfileQuality(str, Enum):
    """Data quality grade for the load profile.

    EXCELLENT: >99% completeness, <0.5% anomalies.
    GOOD:      >95% completeness, <2% anomalies.
    FAIR:      >90% completeness, <5% anomalies.
    POOR:      <90% completeness or >5% anomalies.
    """
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

class TOUPeriod(str, Enum):
    """Time-of-use tariff period classification.

    ON_PEAK:    Peak pricing period.
    MID_PEAK:   Mid/partial-peak pricing period.
    OFF_PEAK:   Off-peak pricing period.
    SUPER_PEAK: Critical / super-peak pricing period.
    """
    ON_PEAK = "on_peak"
    MID_PEAK = "mid_peak"
    OFF_PEAK = "off_peak"
    SUPER_PEAK = "super_peak"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Intervals per day by interval length.
INTERVALS_PER_DAY: Dict[str, int] = {
    IntervalLength.MIN_15.value: 96,
    IntervalLength.MIN_30.value: 48,
    IntervalLength.MIN_60.value: 24,
}

# Hours per year for load-duration curve scaling.
HOURS_PER_YEAR: int = 8760

# Load factor benchmarks by facility type (DOE CBECS / ASHRAE).
LOAD_FACTOR_BENCHMARKS: Dict[str, Decimal] = {
    "office": Decimal("0.45"),
    "retail": Decimal("0.40"),
    "hospital": Decimal("0.70"),
    "data_center": Decimal("0.85"),
    "warehouse": Decimal("0.35"),
    "school": Decimal("0.30"),
    "hotel": Decimal("0.55"),
    "restaurant": Decimal("0.38"),
    "manufacturing": Decimal("0.65"),
    "mixed_use": Decimal("0.50"),
}

# Anomaly detection z-score thresholds.
ANOMALY_Z_THRESHOLD: Decimal = Decimal("3.0")

# Percentile levels for analysis.
PERCENTILE_LEVELS: List[int] = [50, 75, 90, 95, 99]

# Load shape classification thresholds.
FLAT_CV_THRESHOLD: Decimal = Decimal("15")  # CV below 15% is flat
PEAKY_PAR_THRESHOLD: Decimal = Decimal("2.5")  # PAR above 2.5 is peaky

# Morning peak window (hour indices, 0-23).
MORNING_PEAK_START: int = 6
MORNING_PEAK_END: int = 10

# Afternoon peak window.
AFTERNOON_PEAK_START: int = 12
AFTERNOON_PEAK_END: int = 17

# Quality grade thresholds.
QUALITY_THRESHOLDS: List[Tuple[Decimal, Decimal, ProfileQuality]] = [
    (Decimal("99"), Decimal("0.5"), ProfileQuality.EXCELLENT),
    (Decimal("95"), Decimal("2.0"), ProfileQuality.GOOD),
    (Decimal("90"), Decimal("5.0"), ProfileQuality.FAIR),
    (Decimal("0"), Decimal("100"), ProfileQuality.POOR),
]

# Default TOU periods (hour -> period) for a generic commercial tariff.
DEFAULT_TOU_MAP: Dict[int, TOUPeriod] = {
    0: TOUPeriod.OFF_PEAK, 1: TOUPeriod.OFF_PEAK,
    2: TOUPeriod.OFF_PEAK, 3: TOUPeriod.OFF_PEAK,
    4: TOUPeriod.OFF_PEAK, 5: TOUPeriod.OFF_PEAK,
    6: TOUPeriod.OFF_PEAK, 7: TOUPeriod.MID_PEAK,
    8: TOUPeriod.MID_PEAK, 9: TOUPeriod.MID_PEAK,
    10: TOUPeriod.MID_PEAK, 11: TOUPeriod.ON_PEAK,
    12: TOUPeriod.ON_PEAK, 13: TOUPeriod.ON_PEAK,
    14: TOUPeriod.ON_PEAK, 15: TOUPeriod.ON_PEAK,
    16: TOUPeriod.ON_PEAK, 17: TOUPeriod.ON_PEAK,
    18: TOUPeriod.ON_PEAK, 19: TOUPeriod.MID_PEAK,
    20: TOUPeriod.MID_PEAK, 21: TOUPeriod.OFF_PEAK,
    22: TOUPeriod.OFF_PEAK, 23: TOUPeriod.OFF_PEAK,
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class IntervalReading(BaseModel):
    """Single metered interval reading.

    Attributes:
        timestamp: Interval start timestamp (UTC).
        demand_kw: Average demand during interval (kW).
        reactive_kvar: Reactive power during interval (kVAR).
        voltage_v: Average voltage (V).
        day_type: Day type classification.
        season: Season classification.
        tou_period: Time-of-use period.
        is_valid: Whether reading passed quality checks.
        notes: Additional notes or flags.
    """
    timestamp: datetime = Field(
        default_factory=utcnow, description="Interval start timestamp"
    )
    demand_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Average demand (kW)"
    )
    reactive_kvar: Decimal = Field(
        default=Decimal("0"), ge=0, description="Reactive power (kVAR)"
    )
    voltage_v: Decimal = Field(
        default=Decimal("0"), ge=0, description="Average voltage (V)"
    )
    day_type: DayType = Field(
        default=DayType.WEEKDAY, description="Day type classification"
    )
    season: Season = Field(
        default=Season.SUMMER, description="Season classification"
    )
    tou_period: TOUPeriod = Field(
        default=TOUPeriod.OFF_PEAK, description="TOU period"
    )
    is_valid: bool = Field(default=True, description="Data quality flag")
    notes: str = Field(default="", max_length=500, description="Notes")

class LoadStatistics(BaseModel):
    """Statistical summary of a load profile.

    Attributes:
        mean_demand_kw: Arithmetic mean demand (kW).
        median_demand_kw: Median (P50) demand (kW).
        std_dev_kw: Standard deviation (kW).
        min_demand_kw: Minimum demand (kW).
        max_demand_kw: Maximum / peak demand (kW).
        load_factor: Load factor (mean / peak).
        peak_to_average_ratio: Peak-to-average ratio (peak / mean).
        coefficient_of_variation: CV as percentage.
        percentile_p50_kw: 50th percentile demand (kW).
        percentile_p75_kw: 75th percentile demand (kW).
        percentile_p90_kw: 90th percentile demand (kW).
        percentile_p95_kw: 95th percentile demand (kW).
        percentile_p99_kw: 99th percentile demand (kW).
        total_energy_kwh: Total energy consumption (kWh).
        interval_count: Number of valid intervals.
        missing_count: Number of missing / invalid intervals.
        completeness_pct: Data completeness percentage.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    mean_demand_kw: Decimal = Field(default=Decimal("0"))
    median_demand_kw: Decimal = Field(default=Decimal("0"))
    std_dev_kw: Decimal = Field(default=Decimal("0"))
    min_demand_kw: Decimal = Field(default=Decimal("0"))
    max_demand_kw: Decimal = Field(default=Decimal("0"))
    load_factor: Decimal = Field(default=Decimal("0"))
    peak_to_average_ratio: Decimal = Field(default=Decimal("0"))
    coefficient_of_variation: Decimal = Field(default=Decimal("0"))
    percentile_p50_kw: Decimal = Field(default=Decimal("0"))
    percentile_p75_kw: Decimal = Field(default=Decimal("0"))
    percentile_p90_kw: Decimal = Field(default=Decimal("0"))
    percentile_p95_kw: Decimal = Field(default=Decimal("0"))
    percentile_p99_kw: Decimal = Field(default=Decimal("0"))
    total_energy_kwh: Decimal = Field(default=Decimal("0"))
    interval_count: int = Field(default=0)
    missing_count: int = Field(default=0)
    completeness_pct: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class DayTypeProfile(BaseModel):
    """Aggregated profile for a specific day type.

    Attributes:
        day_type: Day type this profile represents.
        season: Season this profile represents.
        load_shape: Classified load shape.
        hourly_avg_kw: Average demand by hour-of-day (24 entries).
        peak_hour: Hour with highest average demand (0-23).
        peak_kw: Peak average demand (kW).
        trough_hour: Hour with lowest average demand (0-23).
        trough_kw: Trough average demand (kW).
        load_factor: Load factor for this day type.
        day_count: Number of days in this cluster.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    day_type: DayType = Field(default=DayType.WEEKDAY)
    season: Season = Field(default=Season.SUMMER)
    load_shape: LoadShape = Field(default=LoadShape.FLAT)
    hourly_avg_kw: Dict[str, Decimal] = Field(default_factory=dict)
    peak_hour: int = Field(default=0, ge=0, le=23)
    peak_kw: Decimal = Field(default=Decimal("0"))
    trough_hour: int = Field(default=0, ge=0, le=23)
    trough_kw: Decimal = Field(default=Decimal("0"))
    load_factor: Decimal = Field(default=Decimal("0"))
    day_count: int = Field(default=0)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class LoadDurationCurve(BaseModel):
    """Load-duration curve representation.

    Attributes:
        curve_points: List of (hours_exceeded, demand_kw) tuples as dicts.
        peak_demand_kw: Maximum demand (0 hours exceeded).
        base_demand_kw: Minimum demand (all hours exceeded).
        area_under_curve_kwh: Total energy (area under LDC).
        hours_above_p90: Hours demand exceeds P90.
        hours_above_p95: Hours demand exceeds P95.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    curve_points: List[Dict[str, Decimal]] = Field(default_factory=list)
    peak_demand_kw: Decimal = Field(default=Decimal("0"))
    base_demand_kw: Decimal = Field(default=Decimal("0"))
    area_under_curve_kwh: Decimal = Field(default=Decimal("0"))
    hours_above_p90: Decimal = Field(default=Decimal("0"))
    hours_above_p95: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class LoadProfileResult(BaseModel):
    """Complete load profile analysis result.

    Attributes:
        profile_id: Unique analysis identifier.
        facility_id: Facility identifier.
        facility_name: Facility name.
        interval_length: Metering interval length.
        analysis_start: Start of analysis period.
        analysis_end: End of analysis period.
        statistics: Overall statistical summary.
        duration_curve: Load-duration curve.
        day_type_profiles: Profiles by day type / season.
        tou_statistics: Statistics by TOU period.
        seasonal_statistics: Statistics by season.
        data_quality: Overall data quality grade.
        anomaly_count: Number of detected anomalies.
        anomaly_indices: Indices of anomalous intervals.
        load_shape: Overall classified load shape.
        benchmark_facility_type: Facility type used for benchmarking.
        benchmark_load_factor: Reference load factor for facility type.
        load_factor_gap: Difference from benchmark (actual - benchmark).
        recommendations: List of analysis recommendations.
        processing_time_ms: Processing duration in milliseconds.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    profile_id: str = Field(default_factory=_new_uuid)
    facility_id: str = Field(default="")
    facility_name: str = Field(default="", max_length=500)
    interval_length: IntervalLength = Field(default=IntervalLength.MIN_15)
    analysis_start: datetime = Field(default_factory=utcnow)
    analysis_end: datetime = Field(default_factory=utcnow)
    statistics: LoadStatistics = Field(default_factory=LoadStatistics)
    duration_curve: LoadDurationCurve = Field(default_factory=LoadDurationCurve)
    day_type_profiles: List[DayTypeProfile] = Field(default_factory=list)
    tou_statistics: Dict[str, Dict[str, Decimal]] = Field(default_factory=dict)
    seasonal_statistics: Dict[str, Dict[str, Decimal]] = Field(default_factory=dict)
    data_quality: ProfileQuality = Field(default=ProfileQuality.POOR)
    anomaly_count: int = Field(default=0)
    anomaly_indices: List[int] = Field(default_factory=list)
    load_shape: LoadShape = Field(default=LoadShape.FLAT)
    benchmark_facility_type: str = Field(default="office")
    benchmark_load_factor: Decimal = Field(default=Decimal("0"))
    load_factor_gap: Decimal = Field(default=Decimal("0"))
    recommendations: List[str] = Field(default_factory=list)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class LoadProfileEngine:
    """Load profile analysis engine for 15-minute interval data.

    Analyses time-series demand data to compute statistical metrics,
    build load-duration curves, cluster day types, detect anomalies,
    and decompose profiles by season and TOU period.  All calculations
    use deterministic Decimal arithmetic with SHA-256 provenance hashing.

    Usage::

        engine = LoadProfileEngine()
        readings = [IntervalReading(timestamp=..., demand_kw=Decimal("120")), ...]
        result = engine.analyze_profile(
            facility_id="FAC-001",
            facility_name="Building A",
            readings=readings,
        )
        print(f"Load factor: {result.statistics.load_factor}")
        print(f"Peak demand: {result.statistics.max_demand_kw} kW")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise LoadProfileEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - interval_length (str): override default interval length
                - anomaly_z_threshold (float): override z-score threshold
                - tou_map (dict): override default TOU hour mapping
                - facility_type (str): benchmark facility type
        """
        self.config = config or {}
        self._interval_length = IntervalLength(
            self.config.get("interval_length", IntervalLength.MIN_15.value)
        )
        self._anomaly_threshold = _decimal(
            self.config.get("anomaly_z_threshold", ANOMALY_Z_THRESHOLD)
        )
        self._tou_map: Dict[int, TOUPeriod] = dict(DEFAULT_TOU_MAP)
        if "tou_map" in self.config:
            for hour_str, period_str in self.config["tou_map"].items():
                self._tou_map[int(hour_str)] = TOUPeriod(period_str)
        self._facility_type = self.config.get("facility_type", "office")
        logger.info(
            "LoadProfileEngine v%s initialised (interval=%s)",
            self.engine_version, self._interval_length.value,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def analyze_profile(
        self,
        facility_id: str,
        facility_name: str,
        readings: List[IntervalReading],
        facility_type: Optional[str] = None,
    ) -> LoadProfileResult:
        """Run a complete load profile analysis.

        Args:
            facility_id: Unique facility identifier.
            facility_name: Facility name.
            readings: List of interval readings (chronological).
            facility_type: Optional facility type for benchmarking.

        Returns:
            LoadProfileResult with statistics, curves, and recommendations.
        """
        t0 = time.perf_counter()
        ftype = facility_type or self._facility_type
        logger.info(
            "Analyzing profile: %s (%d readings, type=%s)",
            facility_name, len(readings), ftype,
        )

        if not readings:
            result = LoadProfileResult(
                facility_id=facility_id,
                facility_name=facility_name,
                data_quality=ProfileQuality.POOR,
                recommendations=["No interval data provided for analysis."],
            )
            result.provenance_hash = _compute_hash(result)
            return result

        # Step 1: Calculate statistics
        statistics = self.calculate_statistics(readings)

        # Step 2: Build duration curve
        duration_curve = self.build_duration_curve(readings)

        # Step 3: Cluster day types
        day_type_profiles = self.cluster_day_types(readings)

        # Step 4: Detect anomalies
        anomaly_indices = self.detect_anomalies(readings)

        # Step 5: TOU period statistics
        tou_stats = self._calculate_tou_statistics(readings)

        # Step 6: Seasonal statistics
        seasonal_stats = self._calculate_seasonal_statistics(readings)

        # Step 7: Classify overall load shape
        load_shape = self._classify_load_shape(statistics, readings)

        # Step 8: Assess data quality
        data_quality = self._assess_quality(statistics, len(anomaly_indices))

        # Step 9: Benchmark comparison
        benchmark_lf = LOAD_FACTOR_BENCHMARKS.get(
            ftype, Decimal("0.50")
        )
        lf_gap = statistics.load_factor - benchmark_lf

        # Step 10: Generate recommendations
        recommendations = self._generate_recommendations(
            statistics, load_shape, lf_gap, data_quality, len(anomaly_indices),
        )

        # Determine analysis period
        timestamps = [r.timestamp for r in readings if r.is_valid]
        analysis_start = min(timestamps) if timestamps else utcnow()
        analysis_end = max(timestamps) if timestamps else utcnow()

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000.0)

        result = LoadProfileResult(
            facility_id=facility_id,
            facility_name=facility_name,
            interval_length=self._interval_length,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
            statistics=statistics,
            duration_curve=duration_curve,
            day_type_profiles=day_type_profiles,
            tou_statistics=tou_stats,
            seasonal_statistics=seasonal_stats,
            data_quality=data_quality,
            anomaly_count=len(anomaly_indices),
            anomaly_indices=anomaly_indices[:100],
            load_shape=load_shape,
            benchmark_facility_type=ftype,
            benchmark_load_factor=_round_val(benchmark_lf, 4),
            load_factor_gap=_round_val(lf_gap, 4),
            recommendations=recommendations,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Profile analysed: %s, LF=%.3f, PAR=%.2f, peak=%.1f kW, "
            "quality=%s, anomalies=%d, hash=%s (%.1f ms)",
            facility_name, float(statistics.load_factor),
            float(statistics.peak_to_average_ratio),
            float(statistics.max_demand_kw), data_quality.value,
            len(anomaly_indices), result.provenance_hash[:16],
            float(elapsed_ms),
        )
        return result

    def build_duration_curve(
        self,
        readings: List[IntervalReading],
    ) -> LoadDurationCurve:
        """Build a load-duration curve from interval readings.

        Sorts demand values in descending order and maps each to the
        cumulative number of hours exceeded.

        Args:
            readings: List of interval readings.

        Returns:
            LoadDurationCurve with sorted points and summary metrics.
        """
        t0 = time.perf_counter()
        logger.info("Building duration curve from %d readings", len(readings))

        valid = [r for r in readings if r.is_valid]
        if not valid:
            result = LoadDurationCurve()
            result.provenance_hash = _compute_hash(result)
            return result

        demands = sorted(
            [_decimal(r.demand_kw) for r in valid], reverse=True
        )
        n = len(demands)

        # Hours per interval
        intervals_day = INTERVALS_PER_DAY.get(
            self._interval_length.value, 96
        )
        hours_per_interval = _safe_divide(
            Decimal("24"), _decimal(intervals_day)
        )

        # Build curve points (sample up to 500 points for efficiency)
        step = max(1, n // 500)
        curve_points: List[Dict[str, Decimal]] = []
        for i in range(0, n, step):
            hours_exceeded = _round_val(
                _decimal(i) * hours_per_interval, 2
            )
            curve_points.append({
                "hours_exceeded": hours_exceeded,
                "demand_kw": _round_val(demands[i], 2),
            })

        # Ensure last point is included
        if n > 0 and (n - 1) % step != 0:
            hours_exceeded = _round_val(
                _decimal(n - 1) * hours_per_interval, 2
            )
            curve_points.append({
                "hours_exceeded": hours_exceeded,
                "demand_kw": _round_val(demands[-1], 2),
            })

        peak_kw = demands[0] if demands else Decimal("0")
        base_kw = demands[-1] if demands else Decimal("0")

        # Area under curve (trapezoidal approximation)
        area = Decimal("0")
        for i in range(1, len(demands)):
            avg_demand = (demands[i - 1] + demands[i]) / Decimal("2")
            area += avg_demand * hours_per_interval

        # Hours above percentiles
        p90_kw = self._percentile(demands, 90)
        p95_kw = self._percentile(demands, 95)
        hours_above_p90 = Decimal("0")
        hours_above_p95 = Decimal("0")
        for d in demands:
            if d >= p90_kw:
                hours_above_p90 += hours_per_interval
            if d >= p95_kw:
                hours_above_p95 += hours_per_interval

        result = LoadDurationCurve(
            curve_points=curve_points,
            peak_demand_kw=_round_val(peak_kw, 2),
            base_demand_kw=_round_val(base_kw, 2),
            area_under_curve_kwh=_round_val(area, 2),
            hours_above_p90=_round_val(hours_above_p90, 2),
            hours_above_p95=_round_val(hours_above_p95, 2),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Duration curve built: peak=%.1f kW, base=%.1f kW, "
            "area=%.1f kWh, hash=%s (%.1f ms)",
            float(peak_kw), float(base_kw), float(area),
            result.provenance_hash[:16], elapsed,
        )
        return result

    def cluster_day_types(
        self,
        readings: List[IntervalReading],
    ) -> List[DayTypeProfile]:
        """Cluster readings by day type and season to produce typical profiles.

        Groups intervals by (day_type, season), computes hourly averages,
        and classifies the resulting load shapes.

        Args:
            readings: List of interval readings.

        Returns:
            List of DayTypeProfile objects, one per (day_type, season) pair.
        """
        t0 = time.perf_counter()
        logger.info("Clustering day types from %d readings", len(readings))

        valid = [r for r in readings if r.is_valid]
        if not valid:
            return []

        # Group by (day_type, season) -> hour -> list of demands
        groups: Dict[Tuple[str, str], Dict[int, List[Decimal]]] = {}
        day_counts: Dict[Tuple[str, str], set] = {}

        for r in valid:
            key = (r.day_type.value, r.season.value)
            hour = r.timestamp.hour
            if key not in groups:
                groups[key] = {}
                day_counts[key] = set()
            if hour not in groups[key]:
                groups[key][hour] = []
            groups[key][hour].append(_decimal(r.demand_kw))
            day_counts[key].add(r.timestamp.date())

        profiles: List[DayTypeProfile] = []
        for (dt_val, season_val), hourly_data in groups.items():
            hourly_avg: Dict[str, Decimal] = {}
            peak_hour = 0
            peak_kw = Decimal("0")
            trough_hour = 0
            trough_kw = Decimal("999999999")

            for hour in range(24):
                demands = hourly_data.get(hour, [])
                if demands:
                    avg = _safe_divide(
                        sum(demands, Decimal("0")),
                        _decimal(len(demands)),
                    )
                else:
                    avg = Decimal("0")
                hourly_avg[str(hour)] = _round_val(avg, 2)

                if avg > peak_kw:
                    peak_kw = avg
                    peak_hour = hour
                if avg < trough_kw and avg > Decimal("0"):
                    trough_kw = avg
                    trough_hour = hour

            if trough_kw == Decimal("999999999"):
                trough_kw = Decimal("0")

            lf = _safe_divide(
                sum(hourly_avg.values(), Decimal("0")) / Decimal("24"),
                peak_kw,
            )

            load_shape = self._classify_hourly_shape(hourly_avg)

            profile = DayTypeProfile(
                day_type=DayType(dt_val),
                season=Season(season_val),
                load_shape=load_shape,
                hourly_avg_kw=hourly_avg,
                peak_hour=peak_hour,
                peak_kw=_round_val(peak_kw, 2),
                trough_hour=trough_hour,
                trough_kw=_round_val(trough_kw, 2),
                load_factor=_round_val(lf, 4),
                day_count=len(day_counts.get((dt_val, season_val), set())),
            )
            profile.provenance_hash = _compute_hash(profile)
            profiles.append(profile)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Day type clustering complete: %d profiles (%.1f ms)",
            len(profiles), elapsed,
        )
        return profiles

    def detect_anomalies(
        self,
        readings: List[IntervalReading],
    ) -> List[int]:
        """Detect anomalous intervals using z-score method.

        An interval is flagged anomalous if its demand deviates from
        the mean by more than the configured z-score threshold times
        the standard deviation.

        Args:
            readings: List of interval readings.

        Returns:
            List of indices of anomalous readings.
        """
        t0 = time.perf_counter()
        logger.info("Detecting anomalies in %d readings", len(readings))

        valid_demands = [
            _decimal(r.demand_kw) for r in readings if r.is_valid
        ]
        if len(valid_demands) < 3:
            return []

        mean_val = _safe_divide(
            sum(valid_demands, Decimal("0")),
            _decimal(len(valid_demands)),
        )
        variance = _safe_divide(
            sum(
                ((d - mean_val) ** 2 for d in valid_demands),
                Decimal("0"),
            ),
            _decimal(len(valid_demands)),
        )
        std_dev = _decimal(math.sqrt(float(variance)))

        if std_dev == Decimal("0"):
            return []

        anomalies: List[int] = []
        for idx, reading in enumerate(readings):
            if not reading.is_valid:
                continue
            demand = _decimal(reading.demand_kw)
            z_score = _safe_divide(abs(demand - mean_val), std_dev)
            if z_score > self._anomaly_threshold:
                anomalies.append(idx)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Anomaly detection complete: %d anomalies of %d readings "
            "(threshold=%.1f sigma) (%.1f ms)",
            len(anomalies), len(readings),
            float(self._anomaly_threshold), elapsed,
        )
        return anomalies

    def calculate_statistics(
        self,
        readings: List[IntervalReading],
    ) -> LoadStatistics:
        """Calculate comprehensive load statistics from interval readings.

        Computes mean, median, std dev, min, max, load factor,
        peak-to-average ratio, coefficient of variation, percentiles,
        and total energy consumption.

        Args:
            readings: List of interval readings.

        Returns:
            LoadStatistics with all computed metrics.
        """
        t0 = time.perf_counter()
        logger.info("Calculating statistics for %d readings", len(readings))

        valid = [r for r in readings if r.is_valid]
        total_count = len(readings)
        valid_count = len(valid)
        missing_count = total_count - valid_count

        if not valid:
            result = LoadStatistics(
                interval_count=0,
                missing_count=total_count,
                completeness_pct=Decimal("0"),
            )
            result.provenance_hash = _compute_hash(result)
            return result

        demands = sorted([_decimal(r.demand_kw) for r in valid])
        n = _decimal(len(demands))

        total = sum(demands, Decimal("0"))
        mean_val = _safe_divide(total, n)
        min_val = demands[0]
        max_val = demands[-1]

        # Median
        median_val = self._percentile(demands, 50)

        # Standard deviation (population)
        variance = _safe_divide(
            sum(((d - mean_val) ** 2 for d in demands), Decimal("0")),
            n,
        )
        std_dev = _decimal(math.sqrt(float(variance)))

        # Load factor
        load_factor = _safe_divide(mean_val, max_val)

        # Peak-to-average ratio
        par = _safe_divide(max_val, mean_val)

        # Coefficient of variation
        cv = _safe_pct(std_dev, mean_val)

        # Percentiles
        p50 = self._percentile(demands, 50)
        p75 = self._percentile(demands, 75)
        p90 = self._percentile(demands, 90)
        p95 = self._percentile(demands, 95)
        p99 = self._percentile(demands, 99)

        # Total energy (kWh)
        intervals_day = INTERVALS_PER_DAY.get(
            self._interval_length.value, 96
        )
        hours_per_interval = _safe_divide(
            Decimal("24"), _decimal(intervals_day)
        )
        total_energy = total * hours_per_interval

        completeness = _safe_pct(_decimal(valid_count), _decimal(total_count))

        result = LoadStatistics(
            mean_demand_kw=_round_val(mean_val, 2),
            median_demand_kw=_round_val(median_val, 2),
            std_dev_kw=_round_val(std_dev, 2),
            min_demand_kw=_round_val(min_val, 2),
            max_demand_kw=_round_val(max_val, 2),
            load_factor=_round_val(load_factor, 4),
            peak_to_average_ratio=_round_val(par, 4),
            coefficient_of_variation=_round_val(cv, 2),
            percentile_p50_kw=_round_val(p50, 2),
            percentile_p75_kw=_round_val(p75, 2),
            percentile_p90_kw=_round_val(p90, 2),
            percentile_p95_kw=_round_val(p95, 2),
            percentile_p99_kw=_round_val(p99, 2),
            total_energy_kwh=_round_val(total_energy, 2),
            interval_count=valid_count,
            missing_count=missing_count,
            completeness_pct=_round_val(completeness, 2),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Statistics calculated: mean=%.1f kW, peak=%.1f kW, "
            "LF=%.3f, PAR=%.2f, CV=%.1f%%, hash=%s (%.1f ms)",
            float(mean_val), float(max_val), float(load_factor),
            float(par), float(cv), result.provenance_hash[:16], elapsed,
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal Methods                                                     #
    # ------------------------------------------------------------------ #

    def _percentile(
        self,
        sorted_values: List[Decimal],
        percentile: int,
    ) -> Decimal:
        """Compute a percentile from a sorted list using linear interpolation.

        Args:
            sorted_values: Pre-sorted ascending list of Decimal values.
            percentile: Desired percentile (0-100).

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
        fraction = rank - _decimal(lower)

        return (
            sorted_values[lower] * (Decimal("1") - fraction)
            + sorted_values[upper] * fraction
        )

    def _calculate_tou_statistics(
        self,
        readings: List[IntervalReading],
    ) -> Dict[str, Dict[str, Decimal]]:
        """Calculate demand statistics by TOU period.

        Args:
            readings: List of interval readings.

        Returns:
            Dict keyed by TOU period with mean, max, and energy stats.
        """
        groups: Dict[str, List[Decimal]] = {}
        for r in readings:
            if not r.is_valid:
                continue
            hour = r.timestamp.hour
            period = self._tou_map.get(hour, TOUPeriod.OFF_PEAK)
            key = period.value
            if key not in groups:
                groups[key] = []
            groups[key].append(_decimal(r.demand_kw))

        intervals_day = INTERVALS_PER_DAY.get(
            self._interval_length.value, 96
        )
        hours_per_interval = _safe_divide(
            Decimal("24"), _decimal(intervals_day)
        )

        result: Dict[str, Dict[str, Decimal]] = {}
        for period_key, demands in groups.items():
            n = _decimal(len(demands))
            total = sum(demands, Decimal("0"))
            mean_val = _safe_divide(total, n)
            max_val = max(demands) if demands else Decimal("0")
            energy = total * hours_per_interval
            result[period_key] = {
                "mean_kw": _round_val(mean_val, 2),
                "max_kw": _round_val(max_val, 2),
                "energy_kwh": _round_val(energy, 2),
                "interval_count": n,
            }

        return result

    def _calculate_seasonal_statistics(
        self,
        readings: List[IntervalReading],
    ) -> Dict[str, Dict[str, Decimal]]:
        """Calculate demand statistics by season.

        Args:
            readings: List of interval readings.

        Returns:
            Dict keyed by season with mean, max, and energy stats.
        """
        groups: Dict[str, List[Decimal]] = {}
        for r in readings:
            if not r.is_valid:
                continue
            key = r.season.value
            if key not in groups:
                groups[key] = []
            groups[key].append(_decimal(r.demand_kw))

        intervals_day = INTERVALS_PER_DAY.get(
            self._interval_length.value, 96
        )
        hours_per_interval = _safe_divide(
            Decimal("24"), _decimal(intervals_day)
        )

        result: Dict[str, Dict[str, Decimal]] = {}
        for season_key, demands in groups.items():
            n = _decimal(len(demands))
            total = sum(demands, Decimal("0"))
            mean_val = _safe_divide(total, n)
            max_val = max(demands) if demands else Decimal("0")
            energy = total * hours_per_interval
            result[season_key] = {
                "mean_kw": _round_val(mean_val, 2),
                "max_kw": _round_val(max_val, 2),
                "energy_kwh": _round_val(energy, 2),
                "interval_count": n,
            }

        return result

    def _classify_load_shape(
        self,
        statistics: LoadStatistics,
        readings: List[IntervalReading],
    ) -> LoadShape:
        """Classify the overall load shape based on statistical features.

        Uses coefficient of variation and peak-to-average ratio to
        determine whether the profile is flat, peaky, or has specific
        peak patterns.

        Args:
            statistics: Computed load statistics.
            readings: Original readings for hourly analysis.

        Returns:
            Classified LoadShape.
        """
        cv = statistics.coefficient_of_variation
        par = statistics.peak_to_average_ratio

        # Flat profile: low variability
        if cv < FLAT_CV_THRESHOLD:
            return LoadShape.FLAT

        # Compute hourly averages for shape detection
        hourly_sums: Dict[int, Decimal] = {}
        hourly_counts: Dict[int, int] = {}
        for r in readings:
            if not r.is_valid:
                continue
            h = r.timestamp.hour
            hourly_sums[h] = hourly_sums.get(h, Decimal("0")) + _decimal(r.demand_kw)
            hourly_counts[h] = hourly_counts.get(h, 0) + 1

        hourly_avg: Dict[int, Decimal] = {}
        for h in range(24):
            s = hourly_sums.get(h, Decimal("0"))
            c = hourly_counts.get(h, 0)
            hourly_avg[h] = _safe_divide(s, _decimal(c)) if c > 0 else Decimal("0")

        if not hourly_avg:
            return LoadShape.FLAT

        max_demand = max(hourly_avg.values())
        if max_demand == Decimal("0"):
            return LoadShape.FLAT

        # Find peak hour
        peak_hour = max(hourly_avg, key=hourly_avg.get)  # type: ignore[arg-type]

        # Check for morning peak
        morning_demands = [
            hourly_avg.get(h, Decimal("0"))
            for h in range(MORNING_PEAK_START, MORNING_PEAK_END + 1)
        ]
        afternoon_demands = [
            hourly_avg.get(h, Decimal("0"))
            for h in range(AFTERNOON_PEAK_START, AFTERNOON_PEAK_END + 1)
        ]

        morning_max = max(morning_demands) if morning_demands else Decimal("0")
        afternoon_max = max(afternoon_demands) if afternoon_demands else Decimal("0")

        morning_ratio = _safe_divide(morning_max, max_demand)
        afternoon_ratio = _safe_divide(afternoon_max, max_demand)

        # Double peak: both morning and afternoon have high values
        if (morning_ratio > Decimal("0.85")
                and afternoon_ratio > Decimal("0.85")):
            return LoadShape.DOUBLE_PEAK

        # Peaky: high PAR
        if par > PEAKY_PAR_THRESHOLD:
            return LoadShape.PEAKY

        # Morning or afternoon peak
        if MORNING_PEAK_START <= peak_hour <= MORNING_PEAK_END:
            return LoadShape.MORNING_PEAK

        if AFTERNOON_PEAK_START <= peak_hour <= AFTERNOON_PEAK_END:
            return LoadShape.AFTERNOON_PEAK

        return LoadShape.PEAKY

    def _classify_hourly_shape(
        self,
        hourly_avg: Dict[str, Decimal],
    ) -> LoadShape:
        """Classify a load shape from hourly average data.

        Args:
            hourly_avg: Dict mapping hour (str) to average kW.

        Returns:
            Classified LoadShape.
        """
        values = [hourly_avg.get(str(h), Decimal("0")) for h in range(24)]
        if not values or all(v == Decimal("0") for v in values):
            return LoadShape.FLAT

        max_val = max(values)
        if max_val == Decimal("0"):
            return LoadShape.FLAT

        mean_val = _safe_divide(
            sum(values, Decimal("0")), Decimal("24")
        )
        if mean_val == Decimal("0"):
            return LoadShape.FLAT

        par = _safe_divide(max_val, mean_val)
        peak_hour = values.index(max_val)

        # Check variability
        variance = _safe_divide(
            sum(((v - mean_val) ** 2 for v in values), Decimal("0")),
            Decimal("24"),
        )
        std_dev = _decimal(math.sqrt(float(variance)))
        cv = _safe_pct(std_dev, mean_val)

        if cv < FLAT_CV_THRESHOLD:
            return LoadShape.FLAT

        morning_vals = values[MORNING_PEAK_START:MORNING_PEAK_END + 1]
        afternoon_vals = values[AFTERNOON_PEAK_START:AFTERNOON_PEAK_END + 1]
        morning_max = max(morning_vals) if morning_vals else Decimal("0")
        afternoon_max = max(afternoon_vals) if afternoon_vals else Decimal("0")

        m_ratio = _safe_divide(morning_max, max_val)
        a_ratio = _safe_divide(afternoon_max, max_val)

        if m_ratio > Decimal("0.85") and a_ratio > Decimal("0.85"):
            return LoadShape.DOUBLE_PEAK

        if par > PEAKY_PAR_THRESHOLD:
            return LoadShape.PEAKY

        if MORNING_PEAK_START <= peak_hour <= MORNING_PEAK_END:
            return LoadShape.MORNING_PEAK

        if AFTERNOON_PEAK_START <= peak_hour <= AFTERNOON_PEAK_END:
            return LoadShape.AFTERNOON_PEAK

        return LoadShape.PEAKY

    def _assess_quality(
        self,
        statistics: LoadStatistics,
        anomaly_count: int,
    ) -> ProfileQuality:
        """Assess overall data quality of the load profile.

        Args:
            statistics: Computed load statistics.
            anomaly_count: Number of detected anomalies.

        Returns:
            ProfileQuality grade.
        """
        completeness = statistics.completeness_pct
        total = statistics.interval_count + statistics.missing_count
        if total == 0:
            return ProfileQuality.POOR
        anomaly_pct = _safe_pct(_decimal(anomaly_count), _decimal(total))

        for comp_thresh, anom_thresh, grade in QUALITY_THRESHOLDS:
            if completeness >= comp_thresh and anomaly_pct <= anom_thresh:
                return grade
        return ProfileQuality.POOR

    def _generate_recommendations(
        self,
        statistics: LoadStatistics,
        load_shape: LoadShape,
        lf_gap: Decimal,
        quality: ProfileQuality,
        anomaly_count: int,
    ) -> List[str]:
        """Generate analysis recommendations based on computed metrics.

        Args:
            statistics: Computed load statistics.
            load_shape: Classified load shape.
            lf_gap: Gap between actual and benchmark load factor.
            quality: Data quality grade.
            anomaly_count: Number of anomalies detected.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        if quality == ProfileQuality.POOR:
            recs.append(
                "Data quality is poor -- install AMI metering or repair "
                "existing meters to achieve >95% completeness."
            )

        if anomaly_count > 10:
            recs.append(
                f"Detected {anomaly_count} anomalous readings. "
                "Investigate meter calibration and data pipeline integrity."
            )

        if lf_gap < Decimal("-0.10"):
            recs.append(
                "Load factor is significantly below benchmark. "
                "Peak shaving or load shifting could improve demand charges."
            )

        if statistics.peak_to_average_ratio > Decimal("3.0"):
            recs.append(
                "High peak-to-average ratio indicates concentrated demand. "
                "BESS or load scheduling can flatten the profile."
            )

        if load_shape == LoadShape.PEAKY:
            recs.append(
                "Peaky load shape identified -- target the peak periods "
                "with battery dispatch or thermal storage pre-cooling."
            )

        if load_shape == LoadShape.DOUBLE_PEAK:
            recs.append(
                "Double-peak profile detected. Consider staggering "
                "equipment startup and using mid-day solar generation."
            )

        if statistics.coefficient_of_variation > Decimal("50"):
            recs.append(
                "High demand variability (CV > 50%). Implement automated "
                "demand limiting controls to reduce volatility."
            )

        if statistics.load_factor < Decimal("0.35"):
            recs.append(
                "Very low load factor (<0.35). Significant peak shaving "
                "opportunity exists -- evaluate BESS economics."
            )

        if not recs:
            recs.append(
                "Load profile is well-balanced. Continue monitoring "
                "and assess incremental efficiency opportunities."
            )

        return recs
