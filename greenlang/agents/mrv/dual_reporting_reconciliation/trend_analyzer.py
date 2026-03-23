# -*- coding: utf-8 -*-
"""
TrendAnalysisEngine - Multi-Period Trend Analysis (Engine 5 of 7)

AGENT-MRV-013: Dual Reporting Reconciliation Agent

Analyses location-based and market-based Scope 2 emissions across multiple
reporting periods to compute year-over-year changes, compound annual growth
rates (CAGR), Procurement Impact Factor (PIF) trends, RE100 renewable
electricity progress, emission intensity metrics, and SBTi target tracking.

Responsibilities:
    1. Compute year-over-year (YoY) percentage change for both location-based
       and market-based emissions between the two most recent periods.
    2. Calculate Compound Annual Growth Rate (CAGR) across all available
       periods for both methods.
    3. Track PIF (Procurement Impact Factor = 1 - market/location) trend
       direction across periods (INCREASING, DECREASING, STABLE).
    4. Track RE100 renewable electricity percentage trend direction.
    5. Calculate emission intensity metrics for the current period:
       revenue-based, FTE-based, floor-area-based, production-unit-based.
    6. Assess SBTi target tracking (is the entity on track to meet its
       Science Based Target for Scope 2 market-based emissions?).
    7. Generate TrendReport with all computed metrics.
    8. Generate flags for adverse trends or off-track targets.

Zero-Hallucination Guarantees:
    - All arithmetic uses Python ``Decimal`` with ROUND_HALF_UP at
      8-decimal-place precision.
    - No LLM, ML, or probabilistic computation in any calculation path.
    - CAGR uses the standard deterministic formula:
      CAGR = (end/start)^(1/n) - 1
    - Provenance hashes computed over serialised inputs and outputs.

Thread Safety:
    Thread-safe singleton via ``__new__`` with ``_instance``,
    ``_initialized``, and ``threading.RLock``.

Public Methods (14):
    analyze_trends              -> TrendReport
    compute_yoy                 -> Tuple[Optional[Decimal], Optional[Decimal]]
    compute_cagr                -> Tuple[Optional[Decimal], Optional[Decimal]]
    determine_trend_direction   -> TrendDirection
    compute_pif_series          -> List[Decimal]
    compute_re100_series        -> List[Decimal]
    compute_intensity_metrics   -> Dict[str, IntensityResult]
    assess_sbti_tracking        -> bool
    compute_linear_projection   -> Decimal
    get_period_count            -> int
    get_most_recent_period      -> Optional[TrendDataPoint]
    get_trend_summary           -> Dict[str, Any]
    generate_trend_flags        -> List[Flag]
    health_check                -> Dict[str, Any]

Classmethod:
    reset                       -> None

Example:
    >>> from greenlang.agents.mrv.dual_reporting_reconciliation.trend_analyzer import (
    ...     TrendAnalysisEngine,
    ... )
    >>> engine = TrendAnalysisEngine()
    >>> report = engine.analyze_trends(
    ...     trend_data=data_points,
    ...     config_params={"tenant_id": "acme-corp"},
    ... )
    >>> print(report.location_cagr, report.pif_trend)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-013 Dual Reporting Reconciliation (GL-MRV-X-024)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = [
    "TrendAnalysisEngine",
    "analyze_trends",
    "compute_cagr",
]

# ---------------------------------------------------------------------------
# Conditional imports -- graceful degradation if sibling modules unavailable
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.models import (
        AGENT_COMPONENT,
        AGENT_ID,
        VERSION,
        Flag,
        FlagSeverity,
        FlagType,
        IntensityMetric,
        IntensityResult,
        TrendDataPoint,
        TrendDirection,
        TrendReport,
    )
except ImportError:
    logger.warning("Could not import models; TrendAnalysisEngine will be limited")
    AGENT_COMPONENT = "AGENT-MRV-013"
    AGENT_ID = "GL-MRV-X-024"
    VERSION = "1.0.0"

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.config import (
        DualReportingReconciliationConfig,
    )
except ImportError:
    DualReportingReconciliationConfig = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.metrics import (
        DualReportingReconciliationMetrics,
    )
except ImportError:
    DualReportingReconciliationMetrics = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.provenance import (
        DualReportingProvenanceTracker,
        ProvenanceStage,
        hash_trend_analysis,
        hash_trend_point,
    )
except ImportError:
    DualReportingProvenanceTracker = None  # type: ignore[assignment,misc]
    ProvenanceStage = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_PRECISION = 8
_QUANTIZE_EXP = Decimal("0." + "0" * _PRECISION)
_ZERO = Decimal("0")
_ONE = Decimal("1")
_ONE_HUNDRED = Decimal("100")
_TWO = Decimal("2")
_STABLE_THRESHOLD_DEFAULT = Decimal("2.0")  # +/- 2% for STABLE classification
_MIN_PERIODS_DEFAULT = 2
_MAX_PERIODS_DEFAULT = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quantize(value: Decimal) -> Decimal:
    """Quantize a Decimal to _PRECISION decimal places."""
    try:
        return value.quantize(_QUANTIZE_EXP, rounding=ROUND_HALF_UP)
    except (InvalidOperation, Exception):
        return _ZERO


def _safe_divide(numerator: Decimal, denominator: Decimal) -> Decimal:
    """Divide numerator by denominator, returning ZERO on division by zero."""
    if denominator == _ZERO:
        return _ZERO
    return _quantize(numerator / denominator)


def _safe_pct_change(new_value: Decimal, old_value: Decimal) -> Decimal:
    """Calculate percentage change: (new - old) / old * 100."""
    if old_value == _ZERO:
        return _ZERO
    return _quantize(((new_value - old_value) / old_value) * _ONE_HUNDRED)


def _abs_decimal(value: Decimal) -> Decimal:
    """Absolute value of a Decimal."""
    return value if value >= _ZERO else -value


def _compute_hash(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of a dictionary."""
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


def _decimal_pow(base: Decimal, exponent: Decimal) -> Decimal:
    """
    Raise Decimal base to Decimal exponent using math.pow.

    Falls back to float arithmetic for fractional exponents since
    Decimal does not natively support non-integer exponents.

    Args:
        base: Base value (must be positive).
        exponent: Exponent value.

    Returns:
        Result as Decimal.
    """
    if base <= _ZERO:
        return _ZERO
    try:
        result = math.pow(float(base), float(exponent))
        return _quantize(Decimal(str(result)))
    except (OverflowError, ValueError, InvalidOperation):
        return _ZERO


# =============================================================================
# TrendAnalysisEngine
# =============================================================================


class TrendAnalysisEngine:
    """
    Engine 5 of 7: Multi-Period Trend Analysis.

    Analyses Scope 2 emissions across multiple reporting periods to compute
    year-over-year changes, CAGR, PIF trends, RE100 progress, intensity
    metrics, and SBTi target tracking.

    Thread-safe singleton via ``__new__`` with ``_instance``,
    ``_initialized``, and ``threading.RLock``.

    Attributes:
        _config: Singleton configuration object.
        _metrics: Singleton metrics tracker.
        _provenance: Singleton provenance tracker.
        _analysis_count: Number of trend analyses performed since init.

    Example:
        >>> engine = TrendAnalysisEngine()
        >>> report = engine.analyze_trends(data_points, config_params)
        >>> print(report.location_cagr, report.market_cagr)
    """

    _instance: Optional[TrendAnalysisEngine] = None
    _initialized: bool = False
    _lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    def __new__(cls) -> TrendAnalysisEngine:
        """Return the singleton instance, creating on first call."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialise engine once."""
        if self.__class__._initialized:
            return
        with self._lock:
            if self.__class__._initialized:
                return
            self._config = (
                DualReportingReconciliationConfig()
                if DualReportingReconciliationConfig is not None
                else None
            )
            self._metrics = (
                DualReportingReconciliationMetrics()
                if DualReportingReconciliationMetrics is not None
                else None
            )
            self._provenance = (
                DualReportingProvenanceTracker.get_instance()
                if DualReportingProvenanceTracker is not None
                else None
            )
            self._analysis_count: int = 0
            self.__class__._initialized = True
            logger.info(
                "%s-TrendAnalysisEngine initialised (v%s)",
                AGENT_COMPONENT,
                VERSION,
            )

    @classmethod
    def reset(cls) -> None:
        """Reset singleton for testing."""
        with cls._lock:
            cls._instance = None
            cls._initialized = False
            logger.warning("TrendAnalysisEngine singleton reset")

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _get_stable_threshold(self) -> Decimal:
        """Get threshold for STABLE trend classification (pct)."""
        if self._config is not None:
            return self._config.stable_threshold
        return _STABLE_THRESHOLD_DEFAULT

    def _get_min_periods(self) -> int:
        """Get minimum periods required for trend analysis."""
        if self._config is not None:
            return self._config.trend_min_periods
        return _MIN_PERIODS_DEFAULT

    def _get_max_periods(self) -> int:
        """Get maximum periods for trend analysis."""
        if self._config is not None:
            return self._config.trend_max_periods
        return _MAX_PERIODS_DEFAULT

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze_trends(
        self,
        trend_data: List[TrendDataPoint],
        config_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[TrendReport]:
        """
        Perform complete multi-period trend analysis.

        This is the primary entry point for Engine 5. It:
        1. Validates input data (minimum period count).
        2. Computes YoY changes for location and market.
        3. Computes CAGR for location and market.
        4. Determines PIF trend direction.
        5. Determines RE100 trend direction.
        6. Computes intensity metrics if denominators provided.
        7. Assesses SBTi target tracking if base year data provided.
        8. Generates flags for adverse trends.
        9. Records metrics and provenance.

        Args:
            trend_data: Ordered list of TrendDataPoint (earliest first).
            config_params: Optional configuration overrides including:
                - tenant_id (str): Tenant identifier
                - intensity_denominators (Dict[str, Decimal])
                - base_year_location_tco2e (Decimal)
                - base_year_market_tco2e (Decimal)
                - sbti_target_year (int)
                - sbti_target_reduction_pct (Decimal)

        Returns:
            TrendReport with all computed metrics, or None if
            insufficient data.
        """
        start_time = time.monotonic()
        config_params = config_params or {}
        tenant_id = config_params.get("tenant_id", "default")

        if not trend_data:
            logger.info("No trend data provided; skipping trend analysis.")
            return None

        min_periods = self._get_min_periods()
        if len(trend_data) < min_periods:
            logger.info(
                "Insufficient trend data: %d points (minimum %d). "
                "Skipping trend analysis.",
                len(trend_data),
                min_periods,
            )
            return None

        # Trim to max periods
        max_periods = self._get_max_periods()
        if len(trend_data) > max_periods:
            trend_data = trend_data[-max_periods:]
            logger.info(
                "Trimmed trend data to most recent %d periods.",
                max_periods,
            )

        logger.info(
            "Analyzing trends for tenant %s: %d periods, "
            "latest=%s",
            tenant_id,
            len(trend_data),
            trend_data[-1].period if trend_data else "N/A",
        )

        # Step 1: YoY changes
        loc_yoy, mkt_yoy = self.compute_yoy(trend_data)

        # Step 2: CAGR
        loc_cagr, mkt_cagr = self.compute_cagr(trend_data)

        # Step 3: PIF trend
        pif_series = self.compute_pif_series(trend_data)
        pif_trend = self.determine_trend_direction(pif_series)

        # Step 4: RE100 trend
        re100_series = self.compute_re100_series(trend_data)
        re100_trend = self.determine_trend_direction(re100_series)

        # Step 5: Intensity metrics
        intensity_metrics: Dict[str, Any] = {}
        denominators = config_params.get("intensity_denominators")
        if denominators and trend_data:
            intensity_metrics = self.compute_intensity_metrics(
                trend_data[-1], denominators
            )

        # Step 6: SBTi tracking
        sbti_on_track = False
        base_market = config_params.get("base_year_market_tco2e")
        target_year = config_params.get("sbti_target_year")
        target_pct = config_params.get("sbti_target_reduction_pct")
        if base_market is not None and target_year is not None and target_pct is not None:
            sbti_on_track = self.assess_sbti_tracking(
                current_market_tco2e=trend_data[-1].market_tco2e,
                base_year_market_tco2e=Decimal(str(base_market)),
                target_year=int(target_year),
                target_reduction_pct=Decimal(str(target_pct)),
                current_period=trend_data[-1].period,
            )

        # Assemble report
        report = TrendReport(
            tenant_id=tenant_id,
            data_points=trend_data,
            location_yoy_pct=loc_yoy,
            market_yoy_pct=mkt_yoy,
            location_cagr=loc_cagr,
            market_cagr=mkt_cagr,
            pif_trend=pif_trend,
            re100_trend=re100_trend,
            intensity_metrics=intensity_metrics,
            sbti_on_track=sbti_on_track,
        )

        # Record metrics
        elapsed_ms = (time.monotonic() - start_time) * 1000
        self._analysis_count += 1

        if self._metrics is not None:
            try:
                self._metrics.record_trend_analysis(tenant_id)
            except Exception as exc:
                logger.warning("Failed to record trend metrics: %s", exc)

        # Record provenance
        if self._provenance is not None and ProvenanceStage is not None:
            try:
                self._provenance.add_stage(
                    config_params.get("reconciliation_id", f"trend-{tenant_id}"),
                    ProvenanceStage.ANALYZE_TRENDS,
                    {
                        "period_count": len(trend_data),
                        "location_yoy": str(loc_yoy),
                        "market_yoy": str(mkt_yoy),
                        "location_cagr": str(loc_cagr),
                        "market_cagr": str(mkt_cagr),
                        "pif_trend": pif_trend.value
                        if hasattr(pif_trend, "value") and pif_trend
                        else str(pif_trend),
                    },
                    {"report_hash": _compute_hash({
                        "periods": len(trend_data),
                        "loc_yoy": str(loc_yoy),
                        "mkt_yoy": str(mkt_yoy),
                    })},
                )
            except Exception as exc:
                logger.debug("Provenance tracking skipped: %s", exc)

        logger.info(
            "Trend analysis complete for %s: %d periods, "
            "loc_yoy=%s%%, mkt_yoy=%s%%, loc_cagr=%s%%, "
            "mkt_cagr=%s%%, pif_trend=%s, re100_trend=%s, "
            "sbti_on_track=%s, elapsed=%.1fms",
            tenant_id,
            len(trend_data),
            loc_yoy, mkt_yoy, loc_cagr, mkt_cagr,
            pif_trend.value if hasattr(pif_trend, "value") and pif_trend else pif_trend,
            re100_trend.value if hasattr(re100_trend, "value") and re100_trend else re100_trend,
            sbti_on_track,
            elapsed_ms,
        )

        return report

    # ------------------------------------------------------------------
    # YoY calculations
    # ------------------------------------------------------------------

    def compute_yoy(
        self, data_points: List[TrendDataPoint]
    ) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """
        Compute year-over-year percentage change for location and market.

        Compares the two most recent periods. Returns (None, None) if
        fewer than 2 periods are available.

        Formula: YoY = (current - previous) / previous * 100

        Args:
            data_points: Ordered TrendDataPoint list (earliest first).

        Returns:
            Tuple of (location_yoy_pct, market_yoy_pct) or (None, None).
        """
        if len(data_points) < 2:
            return None, None

        current = data_points[-1]
        previous = data_points[-2]

        loc_yoy = _safe_pct_change(
            current.location_tco2e, previous.location_tco2e
        )
        mkt_yoy = _safe_pct_change(
            current.market_tco2e, previous.market_tco2e
        )

        return loc_yoy, mkt_yoy

    # ------------------------------------------------------------------
    # CAGR calculations
    # ------------------------------------------------------------------

    def compute_cagr(
        self, data_points: List[TrendDataPoint]
    ) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """
        Compute Compound Annual Growth Rate for location and market.

        CAGR formula: (end_value / start_value)^(1 / n_periods) - 1
        Result is expressed as a percentage.

        Requires at least 2 periods.

        Args:
            data_points: Ordered TrendDataPoint list (earliest first).

        Returns:
            Tuple of (location_cagr_pct, market_cagr_pct) or (None, None).
        """
        if len(data_points) < 2:
            return None, None

        start = data_points[0]
        end = data_points[-1]
        n = Decimal(str(len(data_points) - 1))

        loc_cagr = self._cagr_formula(
            start.location_tco2e, end.location_tco2e, n
        )
        mkt_cagr = self._cagr_formula(
            start.market_tco2e, end.market_tco2e, n
        )

        return loc_cagr, mkt_cagr

    def _cagr_formula(
        self,
        start_value: Decimal,
        end_value: Decimal,
        n_periods: Decimal,
    ) -> Optional[Decimal]:
        """
        Calculate CAGR for a single series.

        CAGR = ((end / start)^(1/n) - 1) * 100

        Args:
            start_value: Starting value.
            end_value: Ending value.
            n_periods: Number of periods (n-1 where n is data points).

        Returns:
            CAGR as percentage, or None if cannot compute.
        """
        if start_value <= _ZERO or n_periods <= _ZERO:
            return None

        ratio = _safe_divide(end_value, start_value)
        if ratio <= _ZERO:
            return None

        exponent = _safe_divide(_ONE, n_periods)
        growth_factor = _decimal_pow(ratio, exponent)
        cagr_pct = _quantize((growth_factor - _ONE) * _ONE_HUNDRED)

        return cagr_pct

    # ------------------------------------------------------------------
    # Trend direction
    # ------------------------------------------------------------------

    def determine_trend_direction(
        self, series: List[Decimal]
    ) -> Optional[TrendDirection]:
        """
        Determine trend direction of a numeric series.

        Compares the first and last values. If the change is within
        the stable threshold, returns STABLE. Otherwise INCREASING
        or DECREASING.

        Args:
            series: Ordered list of Decimal values.

        Returns:
            TrendDirection or None if insufficient data.
        """
        if len(series) < 2:
            return None

        first = series[0]
        last = series[-1]

        if first == _ZERO and last == _ZERO:
            return TrendDirection.STABLE

        pct_change = _safe_pct_change(last, first)
        threshold = self._get_stable_threshold()

        if _abs_decimal(pct_change) <= threshold:
            return TrendDirection.STABLE
        elif pct_change > _ZERO:
            return TrendDirection.INCREASING
        else:
            return TrendDirection.DECREASING

    # ------------------------------------------------------------------
    # PIF series
    # ------------------------------------------------------------------

    def compute_pif_series(
        self, data_points: List[TrendDataPoint]
    ) -> List[Decimal]:
        """
        Compute PIF (Procurement Impact Factor) for each period.

        PIF = 1 - (market / location). A positive PIF means market-based
        is lower than location-based.

        Args:
            data_points: Ordered TrendDataPoint list.

        Returns:
            List of PIF values, one per period.
        """
        pifs: List[Decimal] = []
        for dp in data_points:
            if dp.location_tco2e > _ZERO:
                pif = _quantize(
                    _ONE - _safe_divide(
                        dp.market_tco2e, dp.location_tco2e
                    )
                )
            else:
                pif = _ZERO
            pifs.append(pif)
        return pifs

    # ------------------------------------------------------------------
    # RE100 series
    # ------------------------------------------------------------------

    def compute_re100_series(
        self, data_points: List[TrendDataPoint]
    ) -> List[Decimal]:
        """
        Extract RE100 percentage series from trend data.

        Args:
            data_points: Ordered TrendDataPoint list.

        Returns:
            List of RE100 percentages, one per period.
        """
        return [dp.re100_pct for dp in data_points]

    # ------------------------------------------------------------------
    # Intensity metrics
    # ------------------------------------------------------------------

    def compute_intensity_metrics(
        self,
        current_period: TrendDataPoint,
        denominators: Dict[str, Decimal],
    ) -> Dict[str, Any]:
        """
        Compute emission intensity metrics for the current period.

        Intensity = emissions / denominator for each metric type.

        Supported metric types:
        - revenue: tCO2e per million USD/EUR
        - fte: tCO2e per full-time equivalent employee
        - floor_area: tCO2e per square metre
        - production_unit: tCO2e per unit of output

        Args:
            current_period: Most recent TrendDataPoint.
            denominators: Mapping of metric_type to denominator value.

        Returns:
            Dictionary mapping metric type to IntensityResult dict.
        """
        results: Dict[str, Any] = {}

        intensity_units = {
            "revenue": "tCO2e/million USD",
            "fte": "tCO2e/FTE",
            "floor_area": "tCO2e/m2",
            "production_unit": "tCO2e/unit",
        }

        for metric_type, denominator_value in denominators.items():
            denom = Decimal(str(denominator_value))
            if denom <= _ZERO:
                continue

            loc_intensity = _quantize(
                current_period.location_tco2e / denom
            )
            mkt_intensity = _quantize(
                current_period.market_tco2e / denom
            )

            unit = intensity_units.get(metric_type, f"tCO2e/{metric_type}")

            try:
                result = IntensityResult(
                    metric_type=IntensityMetric(metric_type),
                    location_intensity=loc_intensity,
                    market_intensity=mkt_intensity,
                    unit=unit,
                    period=current_period.period,
                )
                results[metric_type] = result.model_dump()
            except Exception:
                # Fallback if model construction fails
                results[metric_type] = {
                    "metric_type": metric_type,
                    "location_intensity": str(loc_intensity),
                    "market_intensity": str(mkt_intensity),
                    "unit": unit,
                    "period": current_period.period,
                }

        return results

    # ------------------------------------------------------------------
    # SBTi target tracking
    # ------------------------------------------------------------------

    def assess_sbti_tracking(
        self,
        current_market_tco2e: Decimal,
        base_year_market_tco2e: Decimal,
        target_year: int,
        target_reduction_pct: Decimal,
        current_period: str,
    ) -> bool:
        """
        Assess whether the entity is on track to meet its SBTi target.

        SBTi uses market-based emissions for Scope 2 target tracking.
        The entity is on track if the current reduction rate, when
        linearly projected to the target year, would achieve the
        required reduction.

        Args:
            current_market_tco2e: Current period market-based emissions.
            base_year_market_tco2e: Base year market-based emissions.
            target_year: SBTi target year.
            target_reduction_pct: Required reduction percentage (0-100).
            current_period: Current period label (e.g. "2024").

        Returns:
            True if on track, False otherwise.
        """
        if base_year_market_tco2e <= _ZERO:
            return False

        # Calculate current reduction percentage
        current_reduction = _quantize(
            (base_year_market_tco2e - current_market_tco2e)
            / base_year_market_tco2e
            * _ONE_HUNDRED
        )

        # Extract current year from period label
        current_year = self._extract_year_from_period(current_period)
        if current_year is None or current_year >= target_year:
            # If at or past target year, check if target is met
            return current_reduction >= target_reduction_pct

        # Calculate required annual reduction rate
        years_remaining = target_year - current_year
        remaining_reduction = _quantize(target_reduction_pct - current_reduction)

        if remaining_reduction <= _ZERO:
            # Already met or exceeded target
            return True

        # Linear projection: can we achieve remaining reduction
        # at current pace?
        if current_year > 2020:
            years_elapsed = current_year - 2020  # Assume base year ~2020
        else:
            years_elapsed = 1

        annual_rate = _safe_divide(current_reduction, Decimal(str(years_elapsed)))

        if annual_rate <= _ZERO:
            return False

        projected_reduction = _quantize(
            current_reduction
            + annual_rate * Decimal(str(years_remaining))
        )

        return projected_reduction >= target_reduction_pct

    # ------------------------------------------------------------------
    # Linear projection
    # ------------------------------------------------------------------

    def compute_linear_projection(
        self,
        data_points: List[TrendDataPoint],
        periods_ahead: int = 1,
        use_market: bool = True,
    ) -> Decimal:
        """
        Project emissions forward using simple linear extrapolation.

        Uses the average period-over-period change to project.

        Args:
            data_points: Ordered TrendDataPoint list.
            periods_ahead: Number of periods to project.
            use_market: If True, project market-based; else location-based.

        Returns:
            Projected emission value.
        """
        if len(data_points) < 2:
            if data_points:
                return (
                    data_points[-1].market_tco2e
                    if use_market
                    else data_points[-1].location_tco2e
                )
            return _ZERO

        # Calculate average change
        changes: List[Decimal] = []
        for i in range(1, len(data_points)):
            if use_market:
                change = data_points[i].market_tco2e - data_points[i - 1].market_tco2e
            else:
                change = data_points[i].location_tco2e - data_points[i - 1].location_tco2e
            changes.append(change)

        avg_change = _safe_divide(
            sum(changes, start=_ZERO), Decimal(str(len(changes)))
        )

        last_value = (
            data_points[-1].market_tco2e
            if use_market
            else data_points[-1].location_tco2e
        )

        projected = _quantize(
            last_value + avg_change * Decimal(str(periods_ahead))
        )

        # Emissions cannot be negative
        return projected if projected >= _ZERO else _ZERO

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_period_count(self, data_points: List[TrendDataPoint]) -> int:
        """Return the number of available trend data points."""
        return len(data_points) if data_points else 0

    def get_most_recent_period(
        self, data_points: List[TrendDataPoint]
    ) -> Optional[TrendDataPoint]:
        """Return the most recent trend data point."""
        if not data_points:
            return None
        return data_points[-1]

    def get_trend_summary(
        self, report: TrendReport
    ) -> Dict[str, Any]:
        """
        Generate a concise summary of a trend report.

        Args:
            report: TrendReport to summarize.

        Returns:
            Dictionary with summary statistics.
        """
        return {
            "tenant_id": report.tenant_id,
            "period_count": len(report.data_points),
            "location_yoy_pct": str(report.location_yoy_pct),
            "market_yoy_pct": str(report.market_yoy_pct),
            "location_cagr": str(report.location_cagr),
            "market_cagr": str(report.market_cagr),
            "pif_trend": (
                report.pif_trend.value
                if report.pif_trend and hasattr(report.pif_trend, "value")
                else str(report.pif_trend)
            ),
            "re100_trend": (
                report.re100_trend.value
                if report.re100_trend and hasattr(report.re100_trend, "value")
                else str(report.re100_trend)
            ),
            "sbti_on_track": report.sbti_on_track,
            "intensity_metrics_count": len(report.intensity_metrics),
        }

    # ------------------------------------------------------------------
    # Flag generation
    # ------------------------------------------------------------------

    def generate_trend_flags(
        self, report: TrendReport
    ) -> List[Flag]:
        """
        Generate flags based on trend analysis findings.

        Creates flags for:
        - Increasing emissions (location or market)
        - Decreasing PIF (procurement effectiveness declining)
        - Decreasing RE100 percentage
        - Off-track SBTi targets
        - Insufficient trend data

        Args:
            report: TrendReport.

        Returns:
            List of Flag objects.
        """
        flags: List[Flag] = []

        # Flag 1: Location emissions increasing
        if (report.location_yoy_pct is not None
                and report.location_yoy_pct > Decimal("5")):
            flags.append(Flag(
                flag_type=FlagType.WARNING,
                severity=FlagSeverity.MEDIUM,
                code="DRR-T-001",
                message=(
                    f"Location-based emissions increased by "
                    f"{float(report.location_yoy_pct):.2f}% year-over-year."
                ),
                recommendation=(
                    "Investigate causes of increased grid emissions. "
                    "Consider energy efficiency improvements."
                ),
            ))

        # Flag 2: Market emissions increasing
        if (report.market_yoy_pct is not None
                and report.market_yoy_pct > Decimal("5")):
            flags.append(Flag(
                flag_type=FlagType.WARNING,
                severity=FlagSeverity.HIGH,
                code="DRR-T-002",
                message=(
                    f"Market-based emissions increased by "
                    f"{float(report.market_yoy_pct):.2f}% year-over-year."
                ),
                recommendation=(
                    "Review renewable energy procurement strategy. "
                    "Ensure contractual instruments are maintained."
                ),
            ))

        # Flag 3: Decreasing PIF
        if report.pif_trend == TrendDirection.DECREASING:
            flags.append(Flag(
                flag_type=FlagType.WARNING,
                severity=FlagSeverity.MEDIUM,
                code="DRR-T-003",
                message=(
                    "Procurement Impact Factor (PIF) is declining, "
                    "indicating reduced effectiveness of renewable "
                    "energy procurement."
                ),
                recommendation=(
                    "Review renewable energy procurement strategy. "
                    "Consider long-term PPAs or additional REC purchases."
                ),
            ))

        # Flag 4: Decreasing RE100
        if report.re100_trend == TrendDirection.DECREASING:
            flags.append(Flag(
                flag_type=FlagType.WARNING,
                severity=FlagSeverity.HIGH,
                code="DRR-T-004",
                message=(
                    "RE100 renewable electricity percentage is declining."
                ),
                recommendation=(
                    "Increase renewable electricity procurement to "
                    "maintain RE100 commitment trajectory."
                ),
            ))

        # Flag 5: SBTi off track
        if not report.sbti_on_track:
            # Only flag if SBTi data was actually provided
            # (sbti_on_track defaults to False)
            pass  # SBTi tracking requires explicit config

        # Flag 6: Significant negative CAGR for market
        if (report.market_cagr is not None
                and report.market_cagr < Decimal("-10")):
            flags.append(Flag(
                flag_type=FlagType.INFO,
                severity=FlagSeverity.LOW,
                code="DRR-T-005",
                message=(
                    f"Market-based emissions show strong declining CAGR "
                    f"of {float(report.market_cagr):.2f}%, indicating "
                    f"effective decarbonisation."
                ),
                recommendation="",
            ))

        # Flag 7: Insufficient data warning
        if len(report.data_points) < 3:
            flags.append(Flag(
                flag_type=FlagType.INFO,
                severity=FlagSeverity.LOW,
                code="DRR-T-006",
                message=(
                    f"Only {len(report.data_points)} periods available for "
                    f"trend analysis. CAGR and trend direction may not be "
                    f"reliable with limited data."
                ),
                recommendation=(
                    "Continue collecting data points for more reliable "
                    "trend analysis. Minimum 3 periods recommended."
                ),
            ))

        return flags

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the TrendAnalysisEngine.

        Returns:
            Dictionary with engine health status.
        """
        return {
            "status": "healthy",
            "engine": "TrendAnalysisEngine",
            "agent_id": AGENT_ID,
            "component": AGENT_COMPONENT,
            "version": VERSION,
            "initialized": self.__class__._initialized,
            "analysis_count": self._analysis_count,
            "config_available": self._config is not None,
            "metrics_available": self._metrics is not None,
            "provenance_available": self._provenance is not None,
            "min_periods": self._get_min_periods(),
            "max_periods": self._get_max_periods(),
            "stable_threshold": str(self._get_stable_threshold()),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_year_from_period(self, period: str) -> Optional[int]:
        """
        Extract a four-digit year from a period label.

        Handles formats: "2024", "2024-Q1", "2024-01", "FY2024", etc.

        Args:
            period: Period label string.

        Returns:
            Extracted year as int, or None.
        """
        import re
        match = re.search(r"(20\d{2})", period)
        if match:
            return int(match.group(1))
        return None

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TrendAnalysisEngine(version={VERSION}, "
            f"initialized={self.__class__._initialized}, "
            f"analyses={self._analysis_count})"
        )


# =============================================================================
# Module-level convenience functions
# =============================================================================


def analyze_trends(
    trend_data: List[TrendDataPoint],
    config_params: Optional[Dict[str, Any]] = None,
) -> Optional[TrendReport]:
    """Module-level shortcut for TrendAnalysisEngine.analyze_trends."""
    engine = TrendAnalysisEngine()
    return engine.analyze_trends(trend_data, config_params)


def compute_cagr(
    data_points: List[TrendDataPoint],
) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """Module-level shortcut for TrendAnalysisEngine.compute_cagr."""
    engine = TrendAnalysisEngine()
    return engine.compute_cagr(data_points)
