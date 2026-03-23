# -*- coding: utf-8 -*-
"""
WeatherEngine - PACK-040 M&V Engine 7
========================================

Weather normalisation engine for Measurement & Verification calculations.
Calculates heating degree-days (HDD), cooling degree-days (CDD),
optimises balance-point temperatures via iterative regression, performs
TMY (Typical Meteorological Year) normalisation, and executes degree-day
regression for weather-dependent savings.

Calculation Methodology:
    Heating Degree-Days:
        HDD = max(0, Tbase - Tavg) per day
        Monthly HDD = sum of daily HDD

    Cooling Degree-Days:
        CDD = max(0, Tavg - Tbase) per day
        Monthly CDD = sum of daily CDD

    Balance Point Optimisation (ASHRAE method):
        For Tbase in [5, 30] step 0.5:
            Compute HDD/CDD at Tbase
            Fit E = a + bh*HDD + bc*CDD
            Track R-squared
        Select Tbase maximising R-squared

    Degree-Day Regression:
        E = a + bh * HDD + bc * CDD
        where bh = energy per HDD, bc = energy per CDD

    TMY Normalisation:
        TMY_HDD = sum of HDD using TMY3 temperature data
        TMY_CDD = sum of CDD using TMY3 temperature data
        Normalised_savings = model(TMY_conditions) - actual

    Weather Data Quality:
        Completeness = (non-null values / expected values) * 100
        Range check: -60 <= Tavg <= 60 degC
        Consistency: |Tavg[i] - Tavg[i-1]| < threshold
        Temporal coverage: no gaps > max_gap_hours

Regulatory References:
    - ASHRAE Guideline 14-2014 - Weather normalisation for M&V
    - IPMVP Core Concepts 2022 - Routine weather adjustment
    - ISO 50015:2014 - Weather as relevant variable
    - ISO 50006:2014 - Baseline energy performance / weather
    - NOAA TMY3 - Typical Meteorological Year data standard
    - FEMP M&V Guidelines 4.0 - Weather normalisation

Zero-Hallucination:
    - HDD/CDD computed by deterministic max(0, ...) formula
    - Balance-point search by exhaustive grid (no ML)
    - TMY data treated as fixed lookup (no generation)
    - Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-040 M&V
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
from datetime import datetime, timezone, date, timedelta
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


class DegreeDayType(str, Enum):
    """Type of degree-day calculation.

    HDD:   Heating Degree-Days.
    CDD:   Cooling Degree-Days.
    BOTH:  Calculate both HDD and CDD.
    """
    HDD = "hdd"
    CDD = "cdd"
    BOTH = "both"


class TemperatureUnit(str, Enum):
    """Temperature measurement unit.

    CELSIUS:     Degrees Celsius.
    FAHRENHEIT:  Degrees Fahrenheit.
    KELVIN:      Kelvin.
    """
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"
    KELVIN = "kelvin"


class WeatherDataSource(str, Enum):
    """Source of weather data.

    ONSITE:          On-site weather station.
    NOAA_ISD:        NOAA Integrated Surface Database.
    NOAA_TMY3:       NOAA Typical Meteorological Year 3.
    AIRPORT:         Local airport weather station.
    SATELLITE:       Satellite-derived temperature.
    UTILITY:         Utility-provided degree-days.
    CUSTOM:          User-uploaded data.
    """
    ONSITE = "onsite"
    NOAA_ISD = "noaa_isd"
    NOAA_TMY3 = "noaa_tmy3"
    AIRPORT = "airport"
    SATELLITE = "satellite"
    UTILITY = "utility"
    CUSTOM = "custom"


class QualityFlag(str, Enum):
    """Weather data quality flag.

    GOOD:        Value within expected range and passes all checks.
    SUSPECT:     Value near range boundary or small gap-filled.
    ESTIMATED:   Value gap-filled via interpolation.
    MISSING:     Value not available (null).
    OUT_OF_RANGE: Value outside physical limits.
    """
    GOOD = "good"
    SUSPECT = "suspect"
    ESTIMATED = "estimated"
    MISSING = "missing"
    OUT_OF_RANGE = "out_of_range"


class GapFillMethod(str, Enum):
    """Method used to fill missing weather data.

    LINEAR:       Linear interpolation between neighbours.
    PREVIOUS:     Forward-fill with previous value.
    AVERAGE:      Use monthly/weekly average.
    NEARBY:       Use nearby station data.
    TMY:          Fill from TMY dataset.
    NONE:         No gap-fill applied.
    """
    LINEAR = "linear"
    PREVIOUS = "previous"
    AVERAGE = "average"
    NEARBY = "nearby"
    TMY = "tmy"
    NONE = "none"


class NormalisationMethod(str, Enum):
    """Weather normalisation method.

    TMY:              Normalise to Typical Meteorological Year.
    LONG_TERM_AVG:    Normalise to long-term average degree-days.
    REFERENCE_YEAR:   Normalise to a specific reference year.
    REPORTING_PERIOD: Normalise baseline to reporting-period weather.
    """
    TMY = "tmy"
    LONG_TERM_AVG = "long_term_avg"
    REFERENCE_YEAR = "reference_year"
    REPORTING_PERIOD = "reporting_period"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Physical temperature range limits (Celsius)
TEMP_MIN_C = Decimal("-60")
TEMP_MAX_C = Decimal("60")

# Default balance point search range (Celsius)
BP_SEARCH_MIN_C = Decimal("5")
BP_SEARCH_MAX_C = Decimal("30")
BP_SEARCH_STEP_C = Decimal("0.5")

# Maximum acceptable gap (hours) before data is flagged
MAX_GAP_HOURS_DEFAULT = 6


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class DailyTemperature(BaseModel):
    """Daily temperature observation."""

    observation_date: str = Field(..., description="Date string YYYY-MM-DD")
    temp_avg_c: Decimal = Field(..., description="Average temperature (C)")
    temp_min_c: Optional[Decimal] = Field(None, description="Minimum temperature (C)")
    temp_max_c: Optional[Decimal] = Field(None, description="Maximum temperature (C)")
    quality_flag: QualityFlag = Field(
        default=QualityFlag.GOOD, description="Data quality flag"
    )
    source: WeatherDataSource = Field(
        default=WeatherDataSource.NOAA_ISD, description="Data source"
    )

    @field_validator("temp_avg_c", mode="before")
    @classmethod
    def _coerce_temp(cls, v: Any) -> Decimal:
        return _decimal(v)


class DegreeDayRecord(BaseModel):
    """Degree-day calculation for a single period."""

    period_label: str = Field(default="", description="Period label (date or month)")
    hdd: Decimal = Field(default=Decimal("0"), description="Heating degree-days")
    cdd: Decimal = Field(default=Decimal("0"), description="Cooling degree-days")
    avg_temp_c: Decimal = Field(default=Decimal("0"), description="Average temp for period")
    n_days: int = Field(default=0, description="Number of days in period")
    base_temp_heating_c: Decimal = Field(default=Decimal("18"), description="HDD base temp")
    base_temp_cooling_c: Decimal = Field(default=Decimal("18"), description="CDD base temp")


class BalancePointResult(BaseModel):
    """Result of balance-point optimisation."""

    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    optimal_heating_bp_c: Decimal = Field(
        default=Decimal("18"), description="Optimal heating balance point (C)"
    )
    optimal_cooling_bp_c: Decimal = Field(
        default=Decimal("18"), description="Optimal cooling balance point (C)"
    )
    best_r_squared: Decimal = Field(default=Decimal("0"), description="Best R-squared")
    heating_slope: Decimal = Field(default=Decimal("0"), description="Heating slope (energy/HDD)")
    cooling_slope: Decimal = Field(default=Decimal("0"), description="Cooling slope (energy/CDD)")
    intercept: Decimal = Field(default=Decimal("0"), description="Regression intercept")
    candidates_tested: int = Field(default=0, description="Number of candidates evaluated")
    calculated_at: datetime = Field(default_factory=_utcnow)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    provenance_hash: str = Field(default="")


class DegreeDayRegressionResult(BaseModel):
    """Result of degree-day regression E = a + bh*HDD + bc*CDD."""

    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    intercept: Decimal = Field(default=Decimal("0"), description="Intercept (baseload)")
    heating_slope: Decimal = Field(default=Decimal("0"), description="Energy per HDD")
    cooling_slope: Decimal = Field(default=Decimal("0"), description="Energy per CDD")
    base_temp_heating_c: Decimal = Field(default=Decimal("18"), description="Heating base temp")
    base_temp_cooling_c: Decimal = Field(default=Decimal("18"), description="Cooling base temp")
    r_squared: Decimal = Field(default=Decimal("0"), description="R-squared")
    adj_r_squared: Decimal = Field(default=Decimal("0"), description="Adjusted R-squared")
    cvrmse: Decimal = Field(default=Decimal("0"), description="CV(RMSE) %")
    nmbe: Decimal = Field(default=Decimal("0"), description="NMBE %")
    n_observations: int = Field(default=0, description="Number of observations")
    predicted: List[Decimal] = Field(default_factory=list, description="Predicted values")
    residuals: List[Decimal] = Field(default_factory=list, description="Residuals")
    calculated_at: datetime = Field(default_factory=_utcnow)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    provenance_hash: str = Field(default="")


class TMYNormalisationResult(BaseModel):
    """Result of TMY weather normalisation."""

    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    normalisation_method: NormalisationMethod = Field(
        default=NormalisationMethod.TMY, description="Normalisation method used"
    )
    baseline_hdd: Decimal = Field(default=Decimal("0"), description="Baseline period HDD")
    baseline_cdd: Decimal = Field(default=Decimal("0"), description="Baseline period CDD")
    tmy_hdd: Decimal = Field(default=Decimal("0"), description="TMY / normal HDD")
    tmy_cdd: Decimal = Field(default=Decimal("0"), description="TMY / normal CDD")
    reporting_hdd: Decimal = Field(default=Decimal("0"), description="Reporting period HDD")
    reporting_cdd: Decimal = Field(default=Decimal("0"), description="Reporting period CDD")
    normalised_baseline_energy: Decimal = Field(
        default=Decimal("0"), description="Baseline energy normalised to TMY weather"
    )
    normalised_reporting_energy: Decimal = Field(
        default=Decimal("0"), description="Reporting energy normalised to TMY weather"
    )
    normalisation_factor_hdd: Decimal = Field(
        default=Decimal("1"), description="HDD normalisation factor"
    )
    normalisation_factor_cdd: Decimal = Field(
        default=Decimal("1"), description="CDD normalisation factor"
    )
    calculated_at: datetime = Field(default_factory=_utcnow)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    provenance_hash: str = Field(default="")


class WeatherQualityReport(BaseModel):
    """Weather data quality assessment report."""

    report_id: str = Field(default_factory=_new_uuid, description="Report ID")
    source: WeatherDataSource = Field(
        default=WeatherDataSource.NOAA_ISD, description="Data source"
    )
    total_records: int = Field(default=0, description="Total records")
    valid_records: int = Field(default=0, description="Valid (non-null) records")
    completeness_pct: Decimal = Field(default=Decimal("0"), description="Completeness %")
    out_of_range_count: int = Field(default=0, description="Out of range values")
    suspect_count: int = Field(default=0, description="Suspect values")
    gap_count: int = Field(default=0, description="Number of gaps detected")
    max_gap_hours: int = Field(default=0, description="Longest gap in hours")
    avg_temp_c: Decimal = Field(default=Decimal("0"), description="Average temperature")
    min_temp_c: Decimal = Field(default=Decimal("0"), description="Minimum temperature")
    max_temp_c: Decimal = Field(default=Decimal("0"), description="Maximum temperature")
    consistency_score: Decimal = Field(default=Decimal("100"), description="Consistency 0-100")
    overall_grade: QualityFlag = Field(default=QualityFlag.GOOD, description="Overall grade")
    issues: List[str] = Field(default_factory=list, description="Identified issues")
    calculated_at: datetime = Field(default_factory=_utcnow)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    provenance_hash: str = Field(default="")


class WeatherReconciliationResult(BaseModel):
    """Result of multi-source weather data reconciliation."""

    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    sources_compared: List[str] = Field(default_factory=list, description="Source names")
    n_periods: int = Field(default=0, description="Number of periods compared")
    mean_abs_diff_c: Decimal = Field(default=Decimal("0"), description="Mean absolute diff (C)")
    max_abs_diff_c: Decimal = Field(default=Decimal("0"), description="Max absolute diff (C)")
    correlation: Decimal = Field(default=Decimal("0"), description="Pearson correlation")
    recommended_source: str = Field(default="", description="Recommended source")
    recommendation_reason: str = Field(default="", description="Reason for recommendation")
    reconciled_values: List[Decimal] = Field(
        default_factory=list, description="Reconciled temperature values"
    )
    calculated_at: datetime = Field(default_factory=_utcnow)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Linear Algebra Helpers (minimal - for degree-day regression)
# ---------------------------------------------------------------------------


def _ols_2var(
    y: List[Decimal],
    x1: List[Decimal],
    x2: List[Decimal],
) -> Optional[Tuple[Decimal, Decimal, Decimal, Decimal]]:
    """Fit E = a + b1*x1 + b2*x2 via normal equation.

    Returns (intercept, b1, b2, r_squared) or None if singular.
    """
    n = len(y)
    if n < 4:
        return None

    # Build X'X and X'y for [1, x1, x2]
    s_1 = _decimal(n)
    s_x1 = sum(x1)
    s_x2 = sum(x2)
    s_x1x1 = sum(a * a for a in x1)
    s_x2x2 = sum(a * a for a in x2)
    s_x1x2 = sum(x1[i] * x2[i] for i in range(n))
    s_y = sum(y)
    s_x1y = sum(x1[i] * y[i] for i in range(n))
    s_x2y = sum(x2[i] * y[i] for i in range(n))

    # 3x3 matrix inversion via Cramer's rule
    xtx = [
        [s_1, s_x1, s_x2],
        [s_x1, s_x1x1, s_x1x2],
        [s_x2, s_x1x2, s_x2x2],
    ]
    xty = [s_y, s_x1y, s_x2y]

    # Determinant
    det = (xtx[0][0] * (xtx[1][1] * xtx[2][2] - xtx[1][2] * xtx[2][1])
           - xtx[0][1] * (xtx[1][0] * xtx[2][2] - xtx[1][2] * xtx[2][0])
           + xtx[0][2] * (xtx[1][0] * xtx[2][1] - xtx[1][1] * xtx[2][0]))

    if abs(det) < Decimal("1e-30"):
        return None

    # Adjugate (cofactor) approach
    inv = [[Decimal("0")] * 3 for _ in range(3)]
    inv[0][0] = (xtx[1][1] * xtx[2][2] - xtx[1][2] * xtx[2][1]) / det
    inv[0][1] = (xtx[0][2] * xtx[2][1] - xtx[0][1] * xtx[2][2]) / det
    inv[0][2] = (xtx[0][1] * xtx[1][2] - xtx[0][2] * xtx[1][1]) / det
    inv[1][0] = (xtx[1][2] * xtx[2][0] - xtx[1][0] * xtx[2][2]) / det
    inv[1][1] = (xtx[0][0] * xtx[2][2] - xtx[0][2] * xtx[2][0]) / det
    inv[1][2] = (xtx[0][2] * xtx[1][0] - xtx[0][0] * xtx[1][2]) / det
    inv[2][0] = (xtx[1][0] * xtx[2][1] - xtx[1][1] * xtx[2][0]) / det
    inv[2][1] = (xtx[0][1] * xtx[2][0] - xtx[0][0] * xtx[2][1]) / det
    inv[2][2] = (xtx[0][0] * xtx[1][1] - xtx[0][1] * xtx[1][0]) / det

    a = sum(inv[0][j] * xty[j] for j in range(3))
    b1 = sum(inv[1][j] * xty[j] for j in range(3))
    b2 = sum(inv[2][j] * xty[j] for j in range(3))

    # R-squared
    y_mean = _safe_divide(s_y, _decimal(n))
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum((y[i] - a - b1 * x1[i] - b2 * x2[i]) ** 2 for i in range(n))
    r2 = Decimal("1") - _safe_divide(ss_res, ss_tot) if ss_tot > Decimal("0") else Decimal("0")
    r2 = max(r2, Decimal("0"))

    return a, b1, b2, r2


def _ols_1var(
    y: List[Decimal], x: List[Decimal],
) -> Optional[Tuple[Decimal, Decimal, Decimal]]:
    """Simple OLS: E = a + b*x.  Returns (a, b, r_squared) or None."""
    n = len(y)
    if n < 3:
        return None
    s_x = sum(x)
    s_y = sum(y)
    s_xx = sum(xi * xi for xi in x)
    s_xy = sum(x[i] * y[i] for i in range(n))
    nd = _decimal(n)

    denom = nd * s_xx - s_x * s_x
    if abs(denom) < Decimal("1e-30"):
        return None

    b = (nd * s_xy - s_x * s_y) / denom
    a = (s_y - b * s_x) / nd

    y_mean = s_y / nd
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum((y[i] - a - b * x[i]) ** 2 for i in range(n))
    r2 = Decimal("1") - _safe_divide(ss_res, ss_tot) if ss_tot > Decimal("0") else Decimal("0")
    r2 = max(r2, Decimal("0"))

    return a, b, r2


# ---------------------------------------------------------------------------
# Engine Class
# ---------------------------------------------------------------------------


class WeatherEngine:
    """Weather normalisation engine for M&V calculations.

    Provides HDD/CDD calculation, balance-point optimisation via
    iterative regression, TMY normalisation, degree-day regression,
    weather data quality assessment, and multi-source reconciliation.

    All calculations are deterministic (zero-hallucination) with
    Decimal arithmetic and SHA-256 provenance hashing.

    Attributes:
        _module_version: Engine version string.

    Example:
        >>> engine = WeatherEngine()
        >>> temps = [DailyTemperature(
        ...     observation_date="2025-01-01", temp_avg_c=Decimal("-5")
        ... )]
        >>> dd = engine.calculate_degree_days(temps, base_temp_c=Decimal("18"))
        >>> assert dd[0].hdd == Decimal("23")
    """

    def __init__(self) -> None:
        """Initialise the WeatherEngine."""
        self._module_version: str = _MODULE_VERSION
        logger.info("WeatherEngine v%s initialised", self._module_version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_degree_days(
        self,
        temperatures: List[DailyTemperature],
        base_temp_c: Decimal = Decimal("18"),
        dd_type: DegreeDayType = DegreeDayType.BOTH,
    ) -> List[DegreeDayRecord]:
        """Calculate daily degree-days from temperature observations.

        HDD = max(0, base_temp - Tavg)
        CDD = max(0, Tavg - base_temp)

        Args:
            temperatures: Daily temperature observations.
            base_temp_c: Base temperature for degree-day calculation.
            dd_type: Which degree-day types to calculate.

        Returns:
            List of DegreeDayRecord, one per day.
        """
        t0 = time.perf_counter()
        logger.info(
            "Calculating degree-days: %d days, base=%.1f C, type=%s",
            len(temperatures), float(base_temp_c), dd_type.value,
        )

        records: List[DegreeDayRecord] = []
        for obs in temperatures:
            tavg = obs.temp_avg_c
            hdd = Decimal("0")
            cdd = Decimal("0")

            if dd_type in (DegreeDayType.HDD, DegreeDayType.BOTH):
                hdd = max(Decimal("0"), base_temp_c - tavg)

            if dd_type in (DegreeDayType.CDD, DegreeDayType.BOTH):
                cdd = max(Decimal("0"), tavg - base_temp_c)

            records.append(DegreeDayRecord(
                period_label=obs.observation_date,
                hdd=_round_val(hdd, 4),
                cdd=_round_val(cdd, 4),
                avg_temp_c=_round_val(tavg, 2),
                n_days=1,
                base_temp_heating_c=base_temp_c,
                base_temp_cooling_c=base_temp_c,
            ))

        elapsed = (time.perf_counter() - t0) * 1000.0
        total_hdd = sum(r.hdd for r in records)
        total_cdd = sum(r.cdd for r in records)
        logger.info(
            "Degree-days calculated: total_hdd=%.1f, total_cdd=%.1f (%.1f ms)",
            float(total_hdd), float(total_cdd), elapsed,
        )
        return records

    def aggregate_degree_days_monthly(
        self,
        daily_records: List[DegreeDayRecord],
    ) -> List[DegreeDayRecord]:
        """Aggregate daily degree-day records into monthly totals.

        Args:
            daily_records: List of daily DegreeDayRecord.

        Returns:
            List of monthly DegreeDayRecord.
        """
        t0 = time.perf_counter()
        monthly: Dict[str, List[DegreeDayRecord]] = {}
        for rec in daily_records:
            month_key = rec.period_label[:7] if len(rec.period_label) >= 7 else rec.period_label
            monthly.setdefault(month_key, []).append(rec)

        results: List[DegreeDayRecord] = []
        for month_key in sorted(monthly.keys()):
            recs = monthly[month_key]
            n_days = len(recs)
            total_hdd = sum(r.hdd for r in recs)
            total_cdd = sum(r.cdd for r in recs)
            avg_temp = _safe_divide(
                sum(r.avg_temp_c for r in recs), _decimal(n_days)
            )
            results.append(DegreeDayRecord(
                period_label=month_key,
                hdd=_round_val(total_hdd, 2),
                cdd=_round_val(total_cdd, 2),
                avg_temp_c=_round_val(avg_temp, 2),
                n_days=n_days,
                base_temp_heating_c=recs[0].base_temp_heating_c if recs else Decimal("18"),
                base_temp_cooling_c=recs[0].base_temp_cooling_c if recs else Decimal("18"),
            ))

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Monthly aggregation: %d months from %d days (%.1f ms)",
            len(results), len(daily_records), elapsed,
        )
        return results

    def optimise_balance_point(
        self,
        energy_values: List[Decimal],
        temperatures: List[DailyTemperature],
        search_min_c: Decimal = BP_SEARCH_MIN_C,
        search_max_c: Decimal = BP_SEARCH_MAX_C,
        search_step_c: Decimal = BP_SEARCH_STEP_C,
        separate_balance_points: bool = False,
    ) -> BalancePointResult:
        """Optimise balance-point temperature to maximise R-squared.

        Iterates through candidate balance points, computes HDD/CDD at
        each, fits E = a + bh*HDD + bc*CDD, and selects the balance
        point(s) that maximise R-squared.

        Args:
            energy_values: Energy consumption values aligned with temperatures.
            temperatures: Daily or monthly temperature observations.
            search_min_c: Lower bound of balance-point search (C).
            search_max_c: Upper bound of balance-point search (C).
            search_step_c: Step size for search grid (C).
            separate_balance_points: If True, search heating and cooling
                balance points independently (5P style).

        Returns:
            BalancePointResult with optimal balance point(s) and fit quality.
        """
        t0 = time.perf_counter()
        logger.info(
            "Optimising balance point: %d obs, range=[%.1f, %.1f], step=%.1f",
            len(energy_values), float(search_min_c), float(search_max_c),
            float(search_step_c),
        )

        n = min(len(energy_values), len(temperatures))
        y = list(energy_values[:n])
        tavg = [t.temp_avg_c for t in temperatures[:n]]

        best_r2 = Decimal("-1")
        best_h_bp = Decimal("18")
        best_c_bp = Decimal("18")
        best_a = Decimal("0")
        best_bh = Decimal("0")
        best_bc = Decimal("0")
        candidates = 0

        if separate_balance_points:
            # 5P-style: independent heating and cooling balance points
            h_bp = search_min_c
            while h_bp <= search_max_c:
                c_bp = h_bp
                while c_bp <= search_max_c:
                    candidates += 1
                    hdd_vals = [max(Decimal("0"), h_bp - tavg[i]) for i in range(n)]
                    cdd_vals = [max(Decimal("0"), tavg[i] - c_bp) for i in range(n)]
                    result = _ols_2var(y, hdd_vals, cdd_vals)
                    if result is not None:
                        a, bh, bc, r2 = result
                        if r2 > best_r2:
                            best_r2 = r2
                            best_h_bp = h_bp
                            best_c_bp = c_bp
                            best_a = a
                            best_bh = bh
                            best_bc = bc
                    c_bp += search_step_c
                h_bp += search_step_c
        else:
            # 4P-style: single balance point for both
            bp = search_min_c
            while bp <= search_max_c:
                candidates += 1
                hdd_vals = [max(Decimal("0"), bp - tavg[i]) for i in range(n)]
                cdd_vals = [max(Decimal("0"), tavg[i] - bp) for i in range(n)]
                result = _ols_2var(y, hdd_vals, cdd_vals)
                if result is not None:
                    a, bh, bc, r2 = result
                    if r2 > best_r2:
                        best_r2 = r2
                        best_h_bp = bp
                        best_c_bp = bp
                        best_a = a
                        best_bh = bh
                        best_bc = bc
                bp += search_step_c

        elapsed = (time.perf_counter() - t0) * 1000.0
        bp_result = BalancePointResult(
            optimal_heating_bp_c=_round_val(best_h_bp, 2),
            optimal_cooling_bp_c=_round_val(best_c_bp, 2),
            best_r_squared=_round_val(max(best_r2, Decimal("0")), 6),
            heating_slope=_round_val(best_bh, 6),
            cooling_slope=_round_val(best_bc, 6),
            intercept=_round_val(best_a, 4),
            candidates_tested=candidates,
            processing_time_ms=_round_val(_decimal(elapsed), 2),
        )
        bp_result.provenance_hash = _compute_hash(bp_result)

        logger.info(
            "Balance point optimised: heat_bp=%.1f, cool_bp=%.1f, R2=%.4f, "
            "%d candidates, hash=%s (%.1f ms)",
            float(best_h_bp), float(best_c_bp), float(max(best_r2, Decimal("0"))),
            candidates, bp_result.provenance_hash[:16], elapsed,
        )
        return bp_result

    def fit_degree_day_regression(
        self,
        energy_values: List[Decimal],
        hdd_values: List[Decimal],
        cdd_values: List[Decimal],
    ) -> DegreeDayRegressionResult:
        """Fit degree-day regression: E = a + bh*HDD + bc*CDD.

        Args:
            energy_values: Consumption values.
            hdd_values: Heating degree-day values.
            cdd_values: Cooling degree-day values.

        Returns:
            DegreeDayRegressionResult with coefficients and statistics.

        Raises:
            ValueError: If fewer than 4 observations.
        """
        t0 = time.perf_counter()
        n = min(len(energy_values), len(hdd_values), len(cdd_values))
        if n < 4:
            raise ValueError(f"Need >= 4 observations for regression, got {n}")

        y = list(energy_values[:n])
        x1 = list(hdd_values[:n])
        x2 = list(cdd_values[:n])

        ols_result = _ols_2var(y, x1, x2)
        if ols_result is None:
            raise ValueError("Degree-day regression matrix is singular")

        a, bh, bc, r2 = ols_result

        # Compute adj R-squared, CVRMSE, NMBE
        y_mean = _safe_divide(sum(y), _decimal(n))
        y_hat = [a + bh * x1[i] + bc * x2[i] for i in range(n)]
        residuals_list = [y[i] - y_hat[i] for i in range(n)]
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        ss_res = sum(r ** 2 for r in residuals_list)

        df_res = n - 3
        adj_r2 = (Decimal("1") - (Decimal("1") - r2) * _decimal(n - 1) / _decimal(df_res)
                  ) if df_res > 0 else Decimal("0")
        mse = _safe_divide(ss_res, _decimal(df_res)) if df_res > 0 else Decimal("0")
        rmse = _decimal(math.sqrt(float(mse))) if mse > Decimal("0") else Decimal("0")
        cvrmse = _safe_pct(rmse, y_mean) if y_mean != Decimal("0") else Decimal("0")
        nmbe = _safe_divide(
            sum(residuals_list) * Decimal("100"),
            _decimal(n) * y_mean,
        ) if y_mean != Decimal("0") else Decimal("0")

        elapsed = (time.perf_counter() - t0) * 1000.0
        result = DegreeDayRegressionResult(
            intercept=_round_val(a, 4),
            heating_slope=_round_val(bh, 6),
            cooling_slope=_round_val(bc, 6),
            r_squared=_round_val(r2, 6),
            adj_r_squared=_round_val(adj_r2, 6),
            cvrmse=_round_val(cvrmse, 4),
            nmbe=_round_val(nmbe, 4),
            n_observations=n,
            predicted=[_round_val(yh, 4) for yh in y_hat],
            residuals=[_round_val(r, 6) for r in residuals_list],
            processing_time_ms=_round_val(_decimal(elapsed), 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "DD regression fit: a=%.2f, bh=%.4f, bc=%.4f, R2=%.4f, "
            "CVRMSE=%.2f%% (%.1f ms)",
            float(a), float(bh), float(bc), float(r2),
            float(cvrmse), elapsed,
        )
        return result

    def normalise_to_tmy(
        self,
        regression: DegreeDayRegressionResult,
        tmy_hdd: Decimal,
        tmy_cdd: Decimal,
        actual_energy: Decimal,
        actual_hdd: Decimal,
        actual_cdd: Decimal,
        method: NormalisationMethod = NormalisationMethod.TMY,
    ) -> TMYNormalisationResult:
        """Normalise energy consumption to TMY weather conditions.

        Normalised = intercept + bh * TMY_HDD + bc * TMY_CDD

        Args:
            regression: Fitted degree-day regression result.
            tmy_hdd: TMY heating degree-days for the period.
            tmy_cdd: TMY cooling degree-days for the period.
            actual_energy: Actual energy consumption.
            actual_hdd: Actual period HDD.
            actual_cdd: Actual period CDD.
            method: Normalisation method.

        Returns:
            TMYNormalisationResult with normalised values.
        """
        t0 = time.perf_counter()
        logger.info(
            "TMY normalisation: method=%s, TMY_HDD=%.1f, TMY_CDD=%.1f",
            method.value, float(tmy_hdd), float(tmy_cdd),
        )

        # Normalised energy = model prediction at TMY conditions
        normalised_baseline = (
            regression.intercept
            + regression.heating_slope * tmy_hdd
            + regression.cooling_slope * tmy_cdd
        )

        # Normalisation factors
        norm_factor_hdd = _safe_divide(tmy_hdd, actual_hdd, Decimal("1"))
        norm_factor_cdd = _safe_divide(tmy_cdd, actual_cdd, Decimal("1"))

        # Normalised reporting energy using the same weather adjustment
        weather_effect_actual = (
            regression.heating_slope * actual_hdd
            + regression.cooling_slope * actual_cdd
        )
        weather_effect_tmy = (
            regression.heating_slope * tmy_hdd
            + regression.cooling_slope * tmy_cdd
        )
        weather_adjustment = weather_effect_tmy - weather_effect_actual
        normalised_reporting = actual_energy + weather_adjustment

        elapsed = (time.perf_counter() - t0) * 1000.0
        result = TMYNormalisationResult(
            normalisation_method=method,
            baseline_hdd=_round_val(actual_hdd, 2),
            baseline_cdd=_round_val(actual_cdd, 2),
            tmy_hdd=_round_val(tmy_hdd, 2),
            tmy_cdd=_round_val(tmy_cdd, 2),
            reporting_hdd=_round_val(actual_hdd, 2),
            reporting_cdd=_round_val(actual_cdd, 2),
            normalised_baseline_energy=_round_val(normalised_baseline, 2),
            normalised_reporting_energy=_round_val(normalised_reporting, 2),
            normalisation_factor_hdd=_round_val(norm_factor_hdd, 6),
            normalisation_factor_cdd=_round_val(norm_factor_cdd, 6),
            processing_time_ms=_round_val(_decimal(elapsed), 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "TMY normalisation complete: norm_baseline=%.2f, norm_reporting=%.2f, "
            "hash=%s (%.1f ms)",
            float(normalised_baseline), float(normalised_reporting),
            result.provenance_hash[:16], elapsed,
        )
        return result

    def assess_weather_quality(
        self,
        temperatures: List[DailyTemperature],
        expected_days: Optional[int] = None,
        max_gap_hours: int = MAX_GAP_HOURS_DEFAULT,
        max_step_c: Decimal = Decimal("15"),
    ) -> WeatherQualityReport:
        """Assess quality of weather data for M&V use.

        Checks completeness, range, consistency, and temporal coverage.

        Args:
            temperatures: Daily temperature observations.
            expected_days: Expected number of days (defaults to len).
            max_gap_hours: Max acceptable gap in hours.
            max_step_c: Max acceptable day-to-day change in temperature.

        Returns:
            WeatherQualityReport with quality metrics.
        """
        t0 = time.perf_counter()
        n = len(temperatures)
        if expected_days is None:
            expected_days = n

        logger.info(
            "Assessing weather quality: %d records, expected=%d days",
            n, expected_days,
        )

        issues: List[str] = []
        valid_count = 0
        oor_count = 0
        suspect_count = 0
        temp_vals: List[Decimal] = []

        for obs in temperatures:
            if obs.quality_flag == QualityFlag.MISSING:
                continue
            valid_count += 1
            temp_vals.append(obs.temp_avg_c)

            if obs.temp_avg_c < TEMP_MIN_C or obs.temp_avg_c > TEMP_MAX_C:
                oor_count += 1
            elif obs.quality_flag == QualityFlag.SUSPECT:
                suspect_count += 1

        completeness = _safe_pct(_decimal(valid_count), _decimal(expected_days))

        # Gap detection (simplified - count consecutive missing)
        gap_count = 0
        current_gap = 0
        longest_gap = 0
        for obs in temperatures:
            if obs.quality_flag == QualityFlag.MISSING:
                current_gap += 1
            else:
                if current_gap > 0:
                    gap_count += 1
                    longest_gap = max(longest_gap, current_gap)
                current_gap = 0
        if current_gap > 0:
            gap_count += 1
            longest_gap = max(longest_gap, current_gap)

        longest_gap_hours = longest_gap * 24  # daily resolution

        # Consistency check (day-to-day jumps)
        inconsistent = 0
        for i in range(1, len(temp_vals)):
            if abs(temp_vals[i] - temp_vals[i - 1]) > max_step_c:
                inconsistent += 1

        consistency_score = Decimal("100")
        if temp_vals:
            consistency_score = Decimal("100") - _safe_pct(
                _decimal(inconsistent), _decimal(len(temp_vals))
            )
            consistency_score = max(consistency_score, Decimal("0"))

        # Averages
        avg_temp = _safe_divide(sum(temp_vals), _decimal(len(temp_vals))) if temp_vals else Decimal("0")
        min_temp = min(temp_vals) if temp_vals else Decimal("0")
        max_temp = max(temp_vals) if temp_vals else Decimal("0")

        # Issues
        if completeness < Decimal("90"):
            issues.append(f"Low completeness: {float(completeness):.1f}% (< 90%)")
        if oor_count > 0:
            issues.append(f"{oor_count} out-of-range values detected")
        if longest_gap_hours > max_gap_hours:
            issues.append(f"Longest gap {longest_gap_hours}h exceeds max {max_gap_hours}h")
        if inconsistent > 0:
            issues.append(f"{inconsistent} day-to-day jumps > {float(max_step_c)} C")

        # Overall grade
        overall = QualityFlag.GOOD
        if issues:
            overall = QualityFlag.SUSPECT
        if completeness < Decimal("80") or oor_count > n * 0.05:
            overall = QualityFlag.OUT_OF_RANGE

        source = temperatures[0].source if temperatures else WeatherDataSource.NOAA_ISD

        elapsed = (time.perf_counter() - t0) * 1000.0
        report = WeatherQualityReport(
            source=source,
            total_records=n,
            valid_records=valid_count,
            completeness_pct=_round_val(completeness, 2),
            out_of_range_count=oor_count,
            suspect_count=suspect_count,
            gap_count=gap_count,
            max_gap_hours=longest_gap_hours,
            avg_temp_c=_round_val(avg_temp, 2),
            min_temp_c=_round_val(min_temp, 2),
            max_temp_c=_round_val(max_temp, 2),
            consistency_score=_round_val(consistency_score, 2),
            overall_grade=overall,
            issues=issues,
            processing_time_ms=_round_val(_decimal(elapsed), 2),
        )
        report.provenance_hash = _compute_hash(report)

        logger.info(
            "Weather quality assessed: completeness=%.1f%%, grade=%s, "
            "%d issues, hash=%s (%.1f ms)",
            float(completeness), overall.value, len(issues),
            report.provenance_hash[:16], elapsed,
        )
        return report

    def reconcile_sources(
        self,
        source_a_temps: List[Decimal],
        source_b_temps: List[Decimal],
        source_a_name: str = "source_a",
        source_b_name: str = "source_b",
        preference: Optional[str] = None,
    ) -> WeatherReconciliationResult:
        """Reconcile weather data from two sources.

        Computes differences, correlation, and recommends a source.
        If both sources are usable, returns averaged values.

        Args:
            source_a_temps: Temperatures from source A.
            source_b_temps: Temperatures from source B.
            source_a_name: Label for source A.
            source_b_name: Label for source B.
            preference: Optional source preference override.

        Returns:
            WeatherReconciliationResult with reconciled values.
        """
        t0 = time.perf_counter()
        n = min(len(source_a_temps), len(source_b_temps))
        logger.info(
            "Reconciling %d periods: %s vs %s",
            n, source_a_name, source_b_name,
        )

        if n == 0:
            result = WeatherReconciliationResult(
                sources_compared=[source_a_name, source_b_name],
                n_periods=0,
                recommended_source=preference or source_a_name,
                recommendation_reason="No data to compare",
            )
            result.provenance_hash = _compute_hash(result)
            return result

        a_vals = list(source_a_temps[:n])
        b_vals = list(source_b_temps[:n])

        # Absolute differences
        diffs = [abs(a_vals[i] - b_vals[i]) for i in range(n)]
        mean_diff = _safe_divide(sum(diffs), _decimal(n))
        max_diff = max(diffs)

        # Pearson correlation
        a_mean = _safe_divide(sum(a_vals), _decimal(n))
        b_mean = _safe_divide(sum(b_vals), _decimal(n))
        cov_ab = sum((a_vals[i] - a_mean) * (b_vals[i] - b_mean) for i in range(n))
        var_a = sum((a_vals[i] - a_mean) ** 2 for i in range(n))
        var_b = sum((b_vals[i] - b_mean) ** 2 for i in range(n))
        denom = _decimal(math.sqrt(float(var_a * var_b))) if var_a > Decimal("0") and var_b > Decimal("0") else Decimal("1")
        corr = _safe_divide(cov_ab, denom)
        corr = max(min(corr, Decimal("1")), Decimal("-1"))

        # Recommendation
        if preference:
            recommended = preference
            reason = f"User preference for {preference}"
        elif mean_diff < Decimal("1"):
            recommended = source_a_name
            reason = (
                f"Sources agree well (mean diff {float(mean_diff):.2f} C); "
                f"using {source_a_name}"
            )
        else:
            # Prefer on-site over remote
            recommended = source_a_name
            reason = (
                f"Sources diverge (mean diff {float(mean_diff):.2f} C); "
                f"defaulting to {source_a_name}"
            )

        # Reconciled = average of both sources
        reconciled = [
            _round_val((a_vals[i] + b_vals[i]) / Decimal("2"), 2)
            for i in range(n)
        ]

        elapsed = (time.perf_counter() - t0) * 1000.0
        result = WeatherReconciliationResult(
            sources_compared=[source_a_name, source_b_name],
            n_periods=n,
            mean_abs_diff_c=_round_val(mean_diff, 4),
            max_abs_diff_c=_round_val(max_diff, 4),
            correlation=_round_val(corr, 6),
            recommended_source=recommended,
            recommendation_reason=reason,
            reconciled_values=reconciled,
            processing_time_ms=_round_val(_decimal(elapsed), 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Reconciliation: mean_diff=%.2f C, corr=%.4f, recommended=%s, "
            "hash=%s (%.1f ms)",
            float(mean_diff), float(corr), recommended,
            result.provenance_hash[:16], elapsed,
        )
        return result

    def convert_temperature(
        self,
        value: Decimal,
        from_unit: TemperatureUnit,
        to_unit: TemperatureUnit,
    ) -> Decimal:
        """Convert temperature between units.

        Args:
            value: Temperature value to convert.
            from_unit: Source temperature unit.
            to_unit: Target temperature unit.

        Returns:
            Converted temperature value.
        """
        if from_unit == to_unit:
            return value

        # Convert to Celsius first
        celsius: Decimal
        if from_unit == TemperatureUnit.CELSIUS:
            celsius = value
        elif from_unit == TemperatureUnit.FAHRENHEIT:
            celsius = (value - Decimal("32")) * Decimal("5") / Decimal("9")
        elif from_unit == TemperatureUnit.KELVIN:
            celsius = value - Decimal("273.15")
        else:
            celsius = value

        # Convert from Celsius to target
        if to_unit == TemperatureUnit.CELSIUS:
            return _round_val(celsius, 4)
        elif to_unit == TemperatureUnit.FAHRENHEIT:
            return _round_val(celsius * Decimal("9") / Decimal("5") + Decimal("32"), 4)
        elif to_unit == TemperatureUnit.KELVIN:
            return _round_val(celsius + Decimal("273.15"), 4)
        return _round_val(celsius, 4)

    def fill_gaps(
        self,
        temperatures: List[DailyTemperature],
        method: GapFillMethod = GapFillMethod.LINEAR,
    ) -> List[DailyTemperature]:
        """Fill gaps in temperature data using the specified method.

        Args:
            temperatures: Temperature observations (may have MISSING flags).
            method: Gap-filling method to apply.

        Returns:
            New list with gaps filled and quality flags updated.
        """
        t0 = time.perf_counter()
        filled = [t.model_copy(deep=True) for t in temperatures]
        gaps_filled = 0

        if method == GapFillMethod.LINEAR:
            gaps_filled = self._fill_linear(filled)
        elif method == GapFillMethod.PREVIOUS:
            gaps_filled = self._fill_previous(filled)
        elif method == GapFillMethod.AVERAGE:
            gaps_filled = self._fill_average(filled)
        else:
            logger.warning("Gap-fill method %s not implemented; no fill applied", method.value)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Gap fill (%s): %d gaps filled in %d records (%.1f ms)",
            method.value, gaps_filled, len(filled), elapsed,
        )
        return filled

    # ------------------------------------------------------------------
    # Internal gap-fill methods
    # ------------------------------------------------------------------

    def _fill_linear(self, records: List[DailyTemperature]) -> int:
        """Linear interpolation gap-fill.  Modifies records in place."""
        n = len(records)
        filled = 0
        i = 0
        while i < n:
            if records[i].quality_flag == QualityFlag.MISSING:
                # Find bounds
                left_idx = i - 1
                right_idx = i + 1
                while right_idx < n and records[right_idx].quality_flag == QualityFlag.MISSING:
                    right_idx += 1

                if left_idx >= 0 and right_idx < n:
                    left_val = records[left_idx].temp_avg_c
                    right_val = records[right_idx].temp_avg_c
                    gap_len = right_idx - left_idx
                    for j in range(left_idx + 1, right_idx):
                        frac = _decimal(j - left_idx) / _decimal(gap_len)
                        interp = left_val + frac * (right_val - left_val)
                        records[j].temp_avg_c = _round_val(interp, 2)
                        records[j].quality_flag = QualityFlag.ESTIMATED
                        filled += 1
                i = right_idx
            else:
                i += 1
        return filled

    def _fill_previous(self, records: List[DailyTemperature]) -> int:
        """Forward-fill gap-fill.  Modifies records in place."""
        filled = 0
        last_good = Decimal("15")  # default
        for rec in records:
            if rec.quality_flag == QualityFlag.MISSING:
                rec.temp_avg_c = last_good
                rec.quality_flag = QualityFlag.ESTIMATED
                filled += 1
            else:
                last_good = rec.temp_avg_c
        return filled

    def _fill_average(self, records: List[DailyTemperature]) -> int:
        """Average-fill: use average of all valid values."""
        valid_temps = [
            r.temp_avg_c for r in records
            if r.quality_flag not in (QualityFlag.MISSING, QualityFlag.OUT_OF_RANGE)
        ]
        if not valid_temps:
            return 0
        avg = _safe_divide(sum(valid_temps), _decimal(len(valid_temps)))
        filled = 0
        for rec in records:
            if rec.quality_flag == QualityFlag.MISSING:
                rec.temp_avg_c = _round_val(avg, 2)
                rec.quality_flag = QualityFlag.ESTIMATED
                filled += 1
        return filled
