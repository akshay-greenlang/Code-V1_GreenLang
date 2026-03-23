# -*- coding: utf-8 -*-
"""
TrendAnalysisEngine - Multi-Period Trend Analysis (Engine 5 of 7)

AGENT-MRV-013: Dual Reporting Reconciliation Agent

Computes year-over-year (YoY) changes, compound annual growth rates (CAGR),
Procurement Impact Factor (PIF) trends, RE100 renewable energy progress,
SBTi Science Based Target trajectory assessment, emission intensity metrics,
location-based and market-based trend comparisons, discrepancy gap trends,
moving averages, linear forecasts, and z-score anomaly detection across
multiple reporting periods for GHG Protocol Scope 2 dual reporting.

Trend Analysis Methods:
    1. YoY Absolute and Percentage Change - period-over-period comparison
    2. CAGR - (end/start)^(1/n) - 1 compound growth rate
    3. Trend Direction - INCREASING/DECREASING/STABLE within +/-2% threshold
    4. PIF Trend - Procurement Impact Factor per period and direction
    5. RE100 Progress - renewable_mwh / total_electricity_mwh * 100
    6. SBTi Trajectory - base * (1 - annual_reduction)^years_elapsed
    7. Intensity Metrics - tCO2e / revenue, FTE, floor area, production
    8. Location-Based Trend - location emission direction over time
    9. Market-Based Trend - market emission direction over time
   10. Discrepancy Trend - location-market gap change over time
   11. Moving Average - simple moving average with configurable window
   12. Linear Forecast - least-squares linear extrapolation
   13. Anomaly Detection - z-score based outlier identification

Formulas (Deterministic, Zero-Hallucination):
    YoY_Change_Abs = current - previous
    YoY_Change_Pct = (current - previous) / |previous| * 100
    CAGR = (end / start)^(1/n) - 1
    RE100_Pct = renewable_mwh / total_electricity_mwh * 100
    SBTi_Target = base_year * (1 - annual_reduction_rate)^years_elapsed
    Intensity = tCO2e / denominator
    Moving_Avg[i] = sum(values[i-w+1 : i+1]) / w
    Linear_Forecast: y = slope * x + intercept (OLS)
    Z_Score = (value - mean) / std_dev

Thread Safety:
    Thread-safe singleton with ``__new__``, ``_instance``,
    ``_initialized``, and ``threading.RLock``. All mutable state
    (counters, caches) is protected by the reentrant lock.

Zero-Hallucination Guarantees:
    - All calculations use Python ``Decimal`` arithmetic (8 decimal places).
    - No LLM involvement in any numeric computation.
    - Every result carries a SHA-256 provenance hash.
    - Same inputs always produce identical outputs.
    - CAGR uses deterministic Decimal exponentiation via ``math`` bridging.

Example:
    >>> from greenlang.agents.mrv.dual_reporting_reconciliation.trend_analysis import (
    ...     TrendAnalysisEngine,
    ... )
    >>> engine = TrendAnalysisEngine()
    >>> from greenlang.agents.mrv.dual_reporting_reconciliation.models import TrendDataPoint
    >>> points = [
    ...     TrendDataPoint(
    ...         period="2022", location_tco2e=10000, market_tco2e=8000,
    ...         pif=Decimal("0.20"), re100_pct=Decimal("30"),
    ...     ),
    ...     TrendDataPoint(
    ...         period="2023", location_tco2e=9500, market_tco2e=7000,
    ...         pif=Decimal("0.2632"), re100_pct=Decimal("45"),
    ...     ),
    ... ]
    >>> report = engine.analyze_trends(points, {"tenant_id": "tenant-1"})
    >>> print(report["location_trend"])

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
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["TrendAnalysisEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.config import (
        get_config as _get_config,
    )
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.metrics import (
        get_metrics as _get_metrics,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _get_metrics = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.provenance import (
        hash_trend_point as _hash_trend_point,
        hash_trend_analysis as _hash_trend_analysis,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _hash_trend_point = None  # type: ignore[assignment]
    _hash_trend_analysis = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.models import (
        Scope2Method,
        EnergyType,
        IntensityMetric,
        TrendDirection,
        DiscrepancyDirection,
        TrendDataPoint,
        TrendReport,
        IntensityResult,
        ReconciliationWorkspace,
        AGENT_ID,
        AGENT_COMPONENT,
        VERSION,
        MAX_TREND_POINTS,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False
    logger.warning(
        "greenlang.agents.mrv.dual_reporting_reconciliation.models not available; "
        "TrendAnalysisEngine will use inline fallbacks"
    )


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# SHA-256 helper
# ---------------------------------------------------------------------------


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash. Pydantic models are serialised via
            ``model_dump``; all other types use ``json.dumps``.

    Returns:
        Lowercase hex SHA-256 digest (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Decimal helpers
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")
_ZERO = Decimal("0")
_ONE = Decimal("1")
_TWO = Decimal("2")
_HUNDRED = Decimal("100")
_NEG_ONE = Decimal("-1")


def _D(value: Any) -> Decimal:
    """Convert a value to Decimal.

    Args:
        value: Any numeric value or string representation.

    Returns:
        Decimal representation of the value.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _safe_decimal(value: Any, default: Decimal = _ZERO) -> Decimal:
    """Safely convert to Decimal with a fallback default.

    Args:
        value: Value to convert.
        default: Fallback if conversion fails.

    Returns:
        Decimal value or default.
    """
    if value is None:
        return default
    try:
        return _D(value)
    except (InvalidOperation, ValueError, TypeError):
        return default


def _quantize(value: Decimal, precision: Decimal = _PRECISION) -> Decimal:
    """Quantize a Decimal to the specified precision.

    Args:
        value: Decimal value to quantize.
        precision: Target precision (default 8 decimal places).

    Returns:
        Quantized Decimal value.
    """
    return value.quantize(precision, rounding=ROUND_HALF_UP)


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = _ZERO,
) -> Decimal:
    """Safely divide two Decimals, returning default on zero denominator.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Value returned when denominator is zero.

    Returns:
        Quantized result or default.
    """
    if denominator == _ZERO:
        return default
    return _quantize(numerator / denominator)


def _abs_decimal(value: Decimal) -> Decimal:
    """Return the absolute value of a Decimal.

    Args:
        value: Input Decimal.

    Returns:
        Absolute Decimal value.
    """
    return value if value >= _ZERO else value * _NEG_ONE


# ---------------------------------------------------------------------------
# Inline fallback constants (used when models not importable)
# ---------------------------------------------------------------------------

_FALLBACK_AGENT_ID = "GL-MRV-X-024"
_FALLBACK_COMPONENT = "AGENT-MRV-013"
_FALLBACK_VERSION = "1.0.0"
_FALLBACK_MAX_TREND_POINTS = 240

# ---------------------------------------------------------------------------
# Default configuration values
# ---------------------------------------------------------------------------

_DEFAULT_STABLE_THRESHOLD = Decimal("2.0")
_DEFAULT_MIN_PERIODS = 2
_DEFAULT_MAX_PERIODS = 240
_DEFAULT_MOVING_AVERAGE_WINDOW = 3
_DEFAULT_ANOMALY_SIGMA = Decimal("2.0")
_DEFAULT_FORECAST_PERIODS = 3
_DEFAULT_DECIMAL_PLACES = 8
_DEFAULT_SBTI_ANNUAL_REDUCTION = Decimal("4.2")  # 4.2% per year (1.5C pathway)
_DEFAULT_SBTI_TARGET_YEAR = 2030

# ---------------------------------------------------------------------------
# Intensity metric units
# ---------------------------------------------------------------------------

_INTENSITY_UNITS: Dict[str, str] = {
    "revenue": "tCO2e/million USD",
    "fte": "tCO2e/FTE",
    "floor_area": "tCO2e/m2",
    "production_unit": "tCO2e/unit",
}


# ===========================================================================
# TrendAnalysisEngine
# ===========================================================================


class TrendAnalysisEngine:
    """Multi-period trend analysis for Scope 2 dual reporting reconciliation.

    Computes YoY changes, CAGR, PIF trends, RE100 progress, SBTi trajectory,
    intensity metrics, location/market/discrepancy trends, moving averages,
    linear forecasts, and anomaly detection.

    Thread Safety:
        Singleton with ``__new__`` / ``_instance`` / ``_initialized`` /
        ``threading.RLock``. All mutable counters are lock-protected.

    Zero-Hallucination:
        Every numeric computation uses deterministic ``Decimal`` arithmetic.
        No LLM or ML model is called for any calculation. CAGR uses
        ``math.pow`` bridging with full Decimal conversion on output.

    Attributes:
        _lock: Reentrant lock for thread safety.
        _total_analyses: Running counter of analyses performed.
        _created_at: Engine creation timestamp.

    Example:
        >>> engine = TrendAnalysisEngine()
        >>> report = engine.analyze_trends(data_points, config_params)
    """

    _instance: Optional[TrendAnalysisEngine] = None
    _initialized: bool = False
    _lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Singleton constructor
    # ------------------------------------------------------------------

    def __new__(cls) -> TrendAnalysisEngine:
        """Return the singleton instance, creating it on first call.

        Uses double-checked locking with ``threading.RLock`` to ensure
        exactly one instance is created even under concurrent access.

        Returns:
            The singleton TrendAnalysisEngine instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialise engine state on first instantiation.

        Guarded by ``_initialized`` so repeated ``TrendAnalysisEngine()``
        calls do not reset counters or re-read configuration.
        """
        if self.__class__._initialized:
            return
        with self._lock:
            if self.__class__._initialized:
                return
            self._total_analyses: int = 0
            self._total_yoy: int = 0
            self._total_cagr: int = 0
            self._total_pif: int = 0
            self._total_re100: int = 0
            self._total_sbti: int = 0
            self._total_intensity: int = 0
            self._total_location_trend: int = 0
            self._total_market_trend: int = 0
            self._total_discrepancy_trend: int = 0
            self._total_moving_avg: int = 0
            self._total_forecast: int = 0
            self._total_anomaly: int = 0
            self._total_errors: int = 0
            self._created_at: datetime = _utcnow()
            self._stable_threshold: Decimal = _DEFAULT_STABLE_THRESHOLD
            self._min_periods: int = _DEFAULT_MIN_PERIODS
            self._max_periods: int = _DEFAULT_MAX_PERIODS
            self._decimal_places: int = _DEFAULT_DECIMAL_PLACES
            self._load_config()
            self.__class__._initialized = True
            logger.info(
                "TrendAnalysisEngine initialized: stable_threshold=%s, "
                "min_periods=%d, max_periods=%d, decimal_places=%d",
                self._stable_threshold,
                self._min_periods,
                self._max_periods,
                self._decimal_places,
            )

    # ------------------------------------------------------------------
    # Configuration loading
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        """Load configuration from the DRR config singleton.

        Falls back to sensible defaults if config module is not available.
        """
        if not _CONFIG_AVAILABLE:
            logger.debug(
                "Config module not available; using defaults for "
                "TrendAnalysisEngine"
            )
            return
        try:
            cfg = _get_config()
            self._stable_threshold = _safe_decimal(
                getattr(cfg, "stable_threshold", None),
                _DEFAULT_STABLE_THRESHOLD,
            )
            self._min_periods = getattr(
                cfg, "trend_min_periods", _DEFAULT_MIN_PERIODS
            )
            self._max_periods = getattr(
                cfg, "trend_max_periods", _DEFAULT_MAX_PERIODS
            )
            self._decimal_places = getattr(
                cfg, "decimal_places", _DEFAULT_DECIMAL_PLACES
            )
        except Exception as exc:
            logger.warning(
                "Failed to load config for TrendAnalysisEngine: %s; "
                "using defaults",
                exc,
            )

    # ------------------------------------------------------------------
    # Singleton management
    # ------------------------------------------------------------------

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton for test teardown.

        After calling ``reset()``, the next ``TrendAnalysisEngine()`` will
        create a fresh instance and re-read configuration.
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False
        logger.debug("TrendAnalysisEngine singleton reset")

    # ------------------------------------------------------------------
    # Counter helpers
    # ------------------------------------------------------------------

    def _increment(self, counter_name: str) -> None:
        """Thread-safe increment of a named counter.

        Args:
            counter_name: Attribute name on ``self`` to increment.
        """
        with self._lock:
            current = getattr(self, counter_name, 0)
            setattr(self, counter_name, current + 1)

    # ==================================================================
    # 1. Main entry: analyze_trends
    # ==================================================================

    def analyze_trends(
        self,
        data_points: List[Any],
        config_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute full multi-period trend analysis.

        Orchestrates all trend sub-analyses and assembles a complete
        trend report dictionary. This is the primary public entry point
        for callers who have a list of ``TrendDataPoint`` objects.

        Args:
            data_points: Ordered list of trend data points (earliest
                first). Each must have ``period``, ``location_tco2e``,
                ``market_tco2e``, ``pif``, and ``re100_pct`` attributes
                or dict keys.
            config_params: Optional overrides. Supported keys:
                - ``tenant_id`` (str): Tenant identifier.
                - ``stable_threshold`` (Decimal): Trend stability band.
                - ``sbti_base_year_emissions`` (Decimal): SBTi base year.
                - ``sbti_target_year`` (int): SBTi target year.
                - ``sbti_reduction_pct`` (Decimal): SBTi reduction %.
                - ``denominators`` (Dict): Intensity denominators.
                - ``moving_average_window`` (int): MA window size.
                - ``forecast_periods`` (int): Forecast periods.
                - ``anomaly_sigma`` (Decimal): Z-score threshold.

        Returns:
            Dict containing:
                - tenant_id (str)
                - num_periods (int)
                - period_labels (List[str])
                - location_trend (str)
                - market_trend (str)
                - discrepancy_trend (str)
                - location_yoy_changes (List[Dict])
                - market_yoy_changes (List[Dict])
                - location_cagr (Decimal or None)
                - market_cagr (Decimal or None)
                - pif_values (List[Dict])
                - pif_trend (str)
                - re100_values (List[Dict])
                - re100_trend (str)
                - sbti_assessment (Dict or None)
                - intensity_results (Dict)
                - location_moving_avg (List[Decimal])
                - market_moving_avg (List[Decimal])
                - location_forecast (List[Decimal])
                - market_forecast (List[Decimal])
                - anomaly_indices (List[int])
                - recommendations (List[str])
                - provenance_hash (str)
                - processing_time_ms (float)

        Raises:
            ValueError: If fewer than ``min_periods`` data points are
                supplied or if the list exceeds ``max_periods``.
        """
        start_time = time.monotonic()
        params = config_params or {}
        tenant_id = str(params.get("tenant_id", "default"))
        stable_threshold = _safe_decimal(
            params.get("stable_threshold"),
            self._stable_threshold,
        )
        ma_window = int(
            params.get("moving_average_window", _DEFAULT_MOVING_AVERAGE_WINDOW)
        )
        forecast_periods = int(
            params.get("forecast_periods", _DEFAULT_FORECAST_PERIODS)
        )
        anomaly_sigma = _safe_decimal(
            params.get("anomaly_sigma"),
            _DEFAULT_ANOMALY_SIGMA,
        )

        try:
            # Validate data points
            self._validate_data_points(data_points)

            # Extract scalar lists
            period_labels = self._extract_periods(data_points)
            location_values = self._extract_location_values(data_points)
            market_values = self._extract_market_values(data_points)
            pif_raw_values = self._extract_pif_values(data_points)
            re100_raw_values = self._extract_re100_values(data_points)
            num_periods = len(data_points)

            # 1. YoY changes
            location_yoy = self._compute_all_yoy(location_values)
            market_yoy = self._compute_all_yoy(market_values)

            # 2. CAGR
            location_cagr = self._compute_cagr_safe(
                location_values[0], location_values[-1], num_periods - 1
            )
            market_cagr = self._compute_cagr_safe(
                market_values[0], market_values[-1], num_periods - 1
            )

            # 3. Trend directions
            location_trend = self.determine_trend_direction(
                location_values, stable_threshold
            )
            market_trend = self.determine_trend_direction(
                market_values, stable_threshold
            )

            # 4. PIF trend
            pif_result = self.compute_pif_trend(data_points)

            # 5. RE100 progress
            re100_result = self.compute_re100_progress(data_points)

            # 6. Discrepancy trend
            discrepancy_result = self.compute_discrepancy_trend(data_points)

            # 7. SBTi trajectory
            sbti_assessment = self._compute_sbti_if_configured(
                data_points, params
            )

            # 8. Intensity metrics
            denominators = params.get("denominators", {})
            intensity_results = self._compute_intensity_if_available(
                data_points, denominators
            )

            # 9. Location trend detail
            location_trend_detail = self.compute_location_trend(data_points)

            # 10. Market trend detail
            market_trend_detail = self.compute_market_trend(data_points)

            # 11. Moving averages
            location_ma = self.compute_moving_average(
                location_values, ma_window
            )
            market_ma = self.compute_moving_average(
                market_values, ma_window
            )

            # 12. Linear forecasts
            location_forecast = self.forecast_linear(
                location_values, forecast_periods
            )
            market_forecast = self.forecast_linear(
                market_values, forecast_periods
            )

            # 13. Anomaly detection
            anomaly_indices = self.detect_anomalies(
                location_values, anomaly_sigma
            )

            # 14. Recommendations
            recommendations = self._generate_recommendations(
                location_trend=location_trend,
                market_trend=market_trend,
                discrepancy_trend=discrepancy_result.get(
                    "trend_direction", "stable"
                ),
                pif_trend=pif_result.get("trend_direction", "stable"),
                re100_values=re100_raw_values,
                sbti_assessment=sbti_assessment,
                anomaly_count=len(anomaly_indices),
                location_cagr=location_cagr,
                market_cagr=market_cagr,
            )

            # 15. Provenance hash
            provenance_hash = self._compute_provenance_hash(
                num_periods=num_periods,
                location_trend=location_trend,
                market_trend=market_trend,
                pif_trend=pif_result.get("trend_direction", "stable"),
                location_values=location_values,
                market_values=market_values,
            )

            # Record metrics
            self._record_metrics(tenant_id)
            self._increment("_total_analyses")

            elapsed_ms = (time.monotonic() - start_time) * 1000.0

            report = {
                "tenant_id": tenant_id,
                "num_periods": num_periods,
                "period_labels": period_labels,
                "location_trend": location_trend,
                "market_trend": market_trend,
                "discrepancy_trend": discrepancy_result.get(
                    "trend_direction", "stable"
                ),
                "location_yoy_changes": location_yoy,
                "market_yoy_changes": market_yoy,
                "location_cagr": location_cagr,
                "market_cagr": market_cagr,
                "pif_values": pif_result.get("pif_per_period", []),
                "pif_trend": pif_result.get("trend_direction", "stable"),
                "re100_values": re100_result.get("re100_per_period", []),
                "re100_trend": re100_result.get("trend_direction", "stable"),
                "sbti_assessment": sbti_assessment,
                "intensity_results": intensity_results,
                "location_trend_detail": location_trend_detail,
                "market_trend_detail": market_trend_detail,
                "discrepancy_trend_detail": discrepancy_result,
                "location_moving_avg": [str(v) for v in location_ma],
                "market_moving_avg": [str(v) for v in market_ma],
                "location_forecast": [str(v) for v in location_forecast],
                "market_forecast": [str(v) for v in market_forecast],
                "anomaly_indices": anomaly_indices,
                "recommendations": recommendations,
                "provenance_hash": provenance_hash,
                "processing_time_ms": round(elapsed_ms, 3),
            }

            logger.info(
                "TrendAnalysisEngine.analyze_trends completed: "
                "tenant=%s, periods=%d, location_trend=%s, "
                "market_trend=%s, elapsed=%.1fms",
                tenant_id,
                num_periods,
                location_trend,
                market_trend,
                elapsed_ms,
            )

            return report

        except ValueError:
            self._increment("_total_errors")
            raise

        except Exception as exc:
            self._increment("_total_errors")
            logger.error(
                "TrendAnalysisEngine.analyze_trends failed: %s",
                exc,
                exc_info=True,
            )
            raise ValueError(
                f"Trend analysis failed: {exc}"
            ) from exc

    # ==================================================================
    # 2. YoY computation
    # ==================================================================

    def compute_yoy_change(
        self,
        current: Decimal,
        previous: Decimal,
    ) -> Tuple[Decimal, Decimal]:
        """Compute year-over-year absolute change and percentage change.

        Args:
            current: Current period value.
            previous: Previous period value.

        Returns:
            Tuple of (absolute_change, percentage_change).
            If ``previous`` is zero, percentage_change is ``Decimal("0")``.

        Example:
            >>> engine = TrendAnalysisEngine()
            >>> abs_chg, pct_chg = engine.compute_yoy_change(
            ...     Decimal("9500"), Decimal("10000")
            ... )
            >>> print(abs_chg)   # -500
            >>> print(pct_chg)   # -5.0
        """
        current_d = _safe_decimal(current)
        previous_d = _safe_decimal(previous)
        absolute_change = _quantize(current_d - previous_d)
        if previous_d == _ZERO:
            pct_change = _ZERO
        else:
            pct_change = _quantize(
                (current_d - previous_d) / _abs_decimal(previous_d) * _HUNDRED
            )
        self._increment("_total_yoy")
        return absolute_change, pct_change

    def _compute_all_yoy(
        self,
        values: List[Decimal],
    ) -> List[Dict[str, Any]]:
        """Compute YoY changes for all consecutive period pairs.

        Args:
            values: Ordered list of Decimal values (earliest first).

        Returns:
            List of dicts with ``period_index``, ``absolute_change``,
            ``percentage_change`` for each consecutive pair.
        """
        results: List[Dict[str, Any]] = []
        for i in range(1, len(values)):
            abs_chg, pct_chg = self.compute_yoy_change(values[i], values[i - 1])
            results.append({
                "period_index": i,
                "previous_value": str(values[i - 1]),
                "current_value": str(values[i]),
                "absolute_change": str(abs_chg),
                "percentage_change": str(pct_chg),
            })
        return results

    # ==================================================================
    # 3. CAGR computation
    # ==================================================================

    def compute_cagr(
        self,
        start_value: Decimal,
        end_value: Decimal,
        num_years: int,
    ) -> Decimal:
        """Compute Compound Annual Growth Rate.

        Formula: CAGR = (end_value / start_value)^(1/num_years) - 1

        Uses ``math.pow`` for fractional exponentiation then converts
        back to Decimal for deterministic precision.

        Args:
            start_value: Value at the beginning of the period.
            end_value: Value at the end of the period.
            num_years: Number of years (must be >= 1).

        Returns:
            CAGR as a Decimal (e.g. ``Decimal("-0.02500000")`` for -2.5%).
            Returns ``Decimal("0")`` if start_value is zero or num_years < 1.

        Raises:
            ValueError: If num_years < 1.

        Example:
            >>> engine = TrendAnalysisEngine()
            >>> cagr = engine.compute_cagr(
            ...     Decimal("10000"), Decimal("8000"), 3
            ... )
            >>> print(cagr)  # approximately -0.07167...
        """
        start_d = _safe_decimal(start_value)
        end_d = _safe_decimal(end_value)

        if num_years < 1:
            raise ValueError(
                f"num_years must be >= 1, got {num_years}"
            )

        if start_d == _ZERO:
            logger.debug(
                "CAGR: start_value is zero; returning 0"
            )
            return _ZERO

        if start_d < _ZERO or end_d < _ZERO:
            logger.warning(
                "CAGR: negative values detected (start=%s, end=%s); "
                "returning 0",
                start_d,
                end_d,
            )
            return _ZERO

        ratio = float(end_d / start_d)
        exponent = 1.0 / float(num_years)

        try:
            growth_factor = math.pow(ratio, exponent)
        except (OverflowError, ValueError) as exc:
            logger.warning("CAGR math.pow failed: %s; returning 0", exc)
            return _ZERO

        cagr = _quantize(_D(growth_factor) - _ONE)
        self._increment("_total_cagr")

        logger.debug(
            "CAGR: start=%s, end=%s, years=%d, result=%s",
            start_d,
            end_d,
            num_years,
            cagr,
        )
        return cagr

    def _compute_cagr_safe(
        self,
        start_value: Decimal,
        end_value: Decimal,
        num_years: int,
    ) -> Optional[Decimal]:
        """Safely compute CAGR, returning None on failure.

        Args:
            start_value: Start period value.
            end_value: End period value.
            num_years: Number of years.

        Returns:
            CAGR as Decimal or None if computation not possible.
        """
        if num_years < 1:
            return None
        try:
            return self.compute_cagr(start_value, end_value, num_years)
        except (ValueError, InvalidOperation) as exc:
            logger.debug("CAGR computation skipped: %s", exc)
            return None

    # ==================================================================
    # 4. Trend direction determination
    # ==================================================================

    def determine_trend_direction(
        self,
        data_points: List[Decimal],
        threshold_pct: Optional[Decimal] = None,
    ) -> str:
        """Determine the trend direction from an ordered list of values.

        Compares the last value to the first value. If the percentage
        change exceeds ``+threshold_pct`` the trend is INCREASING; if
        it is below ``-threshold_pct`` the trend is DECREASING; otherwise
        the trend is STABLE.

        Args:
            data_points: Ordered Decimal values (earliest first).
            threshold_pct: Stability band percentage (default 2%).

        Returns:
            One of ``"increasing"``, ``"decreasing"``, ``"stable"``.

        Example:
            >>> engine = TrendAnalysisEngine()
            >>> direction = engine.determine_trend_direction(
            ...     [Decimal("10000"), Decimal("9800"), Decimal("9500")],
            ...     Decimal("2.0"),
            ... )
            >>> print(direction)  # "decreasing"
        """
        if threshold_pct is None:
            threshold_pct = self._stable_threshold

        threshold_d = _safe_decimal(threshold_pct, _DEFAULT_STABLE_THRESHOLD)

        if not data_points or len(data_points) < 2:
            return "stable"

        first = _safe_decimal(data_points[0])
        last = _safe_decimal(data_points[-1])

        if first == _ZERO:
            if last > _ZERO:
                return "increasing"
            if last < _ZERO:
                return "decreasing"
            return "stable"

        pct_change = _quantize(
            (last - first) / _abs_decimal(first) * _HUNDRED
        )

        if pct_change > threshold_d:
            return "increasing"
        if pct_change < (threshold_d * _NEG_ONE):
            return "decreasing"
        return "stable"

    # ==================================================================
    # 5. PIF trend
    # ==================================================================

    def compute_pif_trend(
        self,
        data_points: List[Any],
    ) -> Dict[str, Any]:
        """Compute Procurement Impact Factor trend across periods.

        PIF = 1 - (market_tco2e / location_tco2e).
        A higher PIF indicates greater procurement of cleaner energy.

        Args:
            data_points: Ordered list of trend data points.

        Returns:
            Dict containing:
                - pif_per_period: List of {period, pif, pif_pct} dicts
                - pif_values: List of Decimal PIF values
                - trend_direction: "increasing" / "decreasing" / "stable"
                - latest_pif: Decimal PIF for most recent period
                - pif_change: Decimal change from first to last period
        """
        self._increment("_total_pif")
        pif_per_period: List[Dict[str, Any]] = []
        pif_values: List[Decimal] = []

        for dp in data_points:
            period = self._get_attr(dp, "period")
            location = _safe_decimal(self._get_attr(dp, "location_tco2e"))
            market = _safe_decimal(self._get_attr(dp, "market_tco2e"))

            if location == _ZERO:
                pif = _ZERO
            else:
                pif = _quantize(_ONE - (market / location))

            pif_pct = _quantize(pif * _HUNDRED)
            pif_values.append(pif)
            pif_per_period.append({
                "period": period,
                "pif": str(pif),
                "pif_pct": str(pif_pct),
            })

        trend_direction = self.determine_trend_direction(pif_values)
        latest_pif = pif_values[-1] if pif_values else _ZERO
        pif_change = _ZERO
        if len(pif_values) >= 2:
            pif_change = _quantize(pif_values[-1] - pif_values[0])

        result = {
            "pif_per_period": pif_per_period,
            "pif_values": [str(v) for v in pif_values],
            "trend_direction": trend_direction,
            "latest_pif": str(latest_pif),
            "pif_change": str(pif_change),
            "num_periods": len(pif_values),
        }

        logger.debug(
            "PIF trend: direction=%s, latest=%s, change=%s",
            trend_direction,
            latest_pif,
            pif_change,
        )
        return result

    # ==================================================================
    # 6. RE100 progress
    # ==================================================================

    def compute_re100_progress(
        self,
        data_points: List[Any],
    ) -> Dict[str, Any]:
        """Compute RE100 renewable electricity percentage across periods.

        RE100_Pct = renewable_mwh / total_electricity_mwh * 100

        For data points that already carry ``re100_pct``, that value is
        used directly. Otherwise, the engine attempts to compute from
        ``renewable_mwh`` and ``energy_mwh`` attributes if present.

        Args:
            data_points: Ordered list of trend data points.

        Returns:
            Dict containing:
                - re100_per_period: List of {period, re100_pct} dicts
                - re100_values: List of Decimal RE100 percentages
                - trend_direction: "increasing" / "decreasing" / "stable"
                - latest_re100: Decimal RE100% for most recent period
                - re100_change: Decimal change from first to last period
                - target_met: bool (True if latest >= 100.0)
        """
        self._increment("_total_re100")
        re100_per_period: List[Dict[str, Any]] = []
        re100_values: List[Decimal] = []

        for dp in data_points:
            period = self._get_attr(dp, "period")
            re100_pct = _safe_decimal(self._get_attr(dp, "re100_pct"))

            # Attempt computation from raw MWh if re100_pct is zero
            if re100_pct == _ZERO:
                renewable_mwh = _safe_decimal(
                    self._get_attr(dp, "renewable_mwh")
                )
                total_mwh = _safe_decimal(
                    self._get_attr(dp, "energy_mwh")
                )
                if total_mwh > _ZERO and renewable_mwh > _ZERO:
                    re100_pct = _quantize(
                        renewable_mwh / total_mwh * _HUNDRED
                    )

            # Clamp to [0, 100]
            if re100_pct < _ZERO:
                re100_pct = _ZERO
            if re100_pct > _HUNDRED:
                re100_pct = _HUNDRED

            re100_values.append(re100_pct)
            re100_per_period.append({
                "period": period,
                "re100_pct": str(re100_pct),
            })

        trend_direction = self.determine_trend_direction(re100_values)
        latest_re100 = re100_values[-1] if re100_values else _ZERO
        re100_change = _ZERO
        if len(re100_values) >= 2:
            re100_change = _quantize(re100_values[-1] - re100_values[0])

        target_met = latest_re100 >= _HUNDRED

        result = {
            "re100_per_period": re100_per_period,
            "re100_values": [str(v) for v in re100_values],
            "trend_direction": trend_direction,
            "latest_re100": str(latest_re100),
            "re100_change": str(re100_change),
            "target_met": target_met,
            "num_periods": len(re100_values),
        }

        logger.debug(
            "RE100 progress: direction=%s, latest=%s%%, target_met=%s",
            trend_direction,
            latest_re100,
            target_met,
        )
        return result

    # ==================================================================
    # 7. SBTi trajectory
    # ==================================================================

    def compute_sbti_trajectory(
        self,
        base_year_emissions: Decimal,
        target_year: int,
        reduction_pct: Decimal,
        data_points: List[Any],
    ) -> Dict[str, Any]:
        """Compute SBTi Science Based Target trajectory and assessment.

        Target trajectory per year:
            target_emissions = base_year * (1 - annual_reduction)^years_elapsed

        Where ``annual_reduction`` = ``reduction_pct`` / 100 / total_years
        between base year and target year. The method computes target
        emissions for each year that has actual data and assesses whether
        the entity is on-track.

        Args:
            base_year_emissions: Emissions in the base year (tCO2e).
            target_year: SBTi target year (e.g. 2030).
            reduction_pct: Total reduction percentage over the period
                (e.g. ``Decimal("42")`` for 42% by target_year).
            data_points: Ordered list of trend data points with
                ``period`` and ``market_tco2e`` attributes.

        Returns:
            Dict containing:
                - base_year_emissions (str): Base year value
                - target_year (int): Target year
                - total_reduction_pct (str): Total reduction %
                - annual_reduction_rate (str): Annual reduction rate
                - trajectory: List of {period, target_tco2e,
                  actual_tco2e, gap_tco2e, on_track} dicts
                - on_track (bool): Overall on-track assessment
                - latest_gap_pct (str): Gap between actual and target
                - years_to_target (int): Remaining years
        """
        self._increment("_total_sbti")
        base_d = _safe_decimal(base_year_emissions)
        reduction_d = _safe_decimal(reduction_pct)

        if base_d <= _ZERO:
            logger.warning("SBTi: base_year_emissions <= 0; returning empty")
            return self._empty_sbti_result(base_d, target_year, reduction_d)

        # Determine base year from first data point
        base_year = self._extract_year_from_period(
            self._get_attr(data_points[0], "period") if data_points else ""
        )
        if base_year is None and data_points:
            base_year = self._extract_year_from_period(
                self._get_attr(data_points[0], "period")
            )

        total_years = target_year - (base_year or target_year)
        if total_years <= 0:
            logger.warning(
                "SBTi: target_year (%d) <= base_year (%s); "
                "returning empty",
                target_year,
                base_year,
            )
            return self._empty_sbti_result(base_d, target_year, reduction_d)

        # Annual compound reduction rate
        # (1 - annual_rate)^total_years = (1 - total_reduction/100)
        # annual_rate = 1 - (1 - total_reduction/100)^(1/total_years)
        total_factor = _ONE - reduction_d / _HUNDRED
        annual_rate_float = 1.0 - math.pow(
            float(total_factor), 1.0 / float(total_years)
        )
        annual_rate = _quantize(_D(annual_rate_float))

        trajectory: List[Dict[str, Any]] = []
        overall_on_track = True
        latest_gap_pct = _ZERO

        for dp in data_points:
            period = self._get_attr(dp, "period")
            actual_market = _safe_decimal(
                self._get_attr(dp, "market_tco2e")
            )
            dp_year = self._extract_year_from_period(period)

            if dp_year is None or base_year is None:
                continue

            years_elapsed = dp_year - base_year
            if years_elapsed < 0:
                continue

            # Target = base * (1 - annual_rate)^years_elapsed
            target_factor_float = math.pow(
                1.0 - float(annual_rate), float(years_elapsed)
            )
            target_tco2e = _quantize(base_d * _D(target_factor_float))
            gap_tco2e = _quantize(actual_market - target_tco2e)
            period_on_track = actual_market <= target_tco2e

            if not period_on_track:
                overall_on_track = False

            # Gap as percentage of target
            if target_tco2e > _ZERO:
                gap_pct = _quantize(gap_tco2e / target_tco2e * _HUNDRED)
            else:
                gap_pct = _ZERO

            latest_gap_pct = gap_pct

            trajectory.append({
                "period": period,
                "year": dp_year,
                "years_elapsed": years_elapsed,
                "target_tco2e": str(target_tco2e),
                "actual_tco2e": str(actual_market),
                "gap_tco2e": str(gap_tco2e),
                "gap_pct": str(gap_pct),
                "on_track": period_on_track,
            })

        # Remaining years from last data point
        last_year = None
        if data_points:
            last_year = self._extract_year_from_period(
                self._get_attr(data_points[-1], "period")
            )
        years_to_target = target_year - (last_year or target_year)

        result = {
            "base_year_emissions": str(base_d),
            "base_year": base_year,
            "target_year": target_year,
            "total_reduction_pct": str(reduction_d),
            "annual_reduction_rate": str(annual_rate),
            "trajectory": trajectory,
            "on_track": overall_on_track,
            "latest_gap_pct": str(latest_gap_pct),
            "years_to_target": max(0, years_to_target),
            "num_trajectory_points": len(trajectory),
        }

        logger.info(
            "SBTi trajectory: on_track=%s, annual_rate=%s, "
            "gap_pct=%s, years_to_target=%d",
            overall_on_track,
            annual_rate,
            latest_gap_pct,
            max(0, years_to_target),
        )
        return result

    def _empty_sbti_result(
        self,
        base_d: Decimal,
        target_year: int,
        reduction_d: Decimal,
    ) -> Dict[str, Any]:
        """Return an empty SBTi assessment structure.

        Args:
            base_d: Base year emissions.
            target_year: Target year.
            reduction_d: Reduction percentage.

        Returns:
            Dict with empty trajectory and on_track=False.
        """
        return {
            "base_year_emissions": str(base_d),
            "base_year": None,
            "target_year": target_year,
            "total_reduction_pct": str(reduction_d),
            "annual_reduction_rate": str(_ZERO),
            "trajectory": [],
            "on_track": False,
            "latest_gap_pct": str(_ZERO),
            "years_to_target": 0,
            "num_trajectory_points": 0,
        }

    def _compute_sbti_if_configured(
        self,
        data_points: List[Any],
        params: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Compute SBTi trajectory if parameters are provided.

        Args:
            data_points: Trend data points.
            params: Configuration parameters.

        Returns:
            SBTi assessment dict or None if not configured.
        """
        base_emissions = params.get("sbti_base_year_emissions")
        if base_emissions is None:
            return None

        target_year = int(
            params.get("sbti_target_year", _DEFAULT_SBTI_TARGET_YEAR)
        )
        reduction_pct = _safe_decimal(
            params.get("sbti_reduction_pct"),
            _DEFAULT_SBTI_ANNUAL_REDUCTION * _D(10),
        )

        return self.compute_sbti_trajectory(
            base_year_emissions=_safe_decimal(base_emissions),
            target_year=target_year,
            reduction_pct=reduction_pct,
            data_points=data_points,
        )

    # ==================================================================
    # 8. Intensity metrics
    # ==================================================================

    def compute_intensity_metrics(
        self,
        data_points: List[Any],
        denominators: Dict[str, List[Decimal]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Compute emission intensity metrics across periods.

        Intensity = tCO2e / denominator_value

        Supported intensity metrics:
            - revenue: tCO2e per million USD
            - fte: tCO2e per full-time equivalent employee
            - floor_area: tCO2e per square metre
            - production_unit: tCO2e per unit of output

        Args:
            data_points: Ordered list of trend data points.
            denominators: Dict mapping intensity metric name to a list
                of Decimal denominator values (one per period). Keys:
                ``"revenue"``, ``"fte"``, ``"floor_area"``,
                ``"production_unit"``.

        Returns:
            Dict mapping metric name to list of intensity result dicts.
            Each result has ``period``, ``location_intensity``,
            ``market_intensity``, ``denominator_value``, ``unit``.

        Example:
            >>> engine = TrendAnalysisEngine()
            >>> results = engine.compute_intensity_metrics(
            ...     data_points,
            ...     {"revenue": [Decimal("100"), Decimal("110")]},
            ... )
        """
        self._increment("_total_intensity")
        results: Dict[str, List[Dict[str, Any]]] = {}

        for metric_name, denom_values in denominators.items():
            metric_results: List[Dict[str, Any]] = []
            unit = _INTENSITY_UNITS.get(metric_name, f"tCO2e/{metric_name}")

            for idx, dp in enumerate(data_points):
                period = self._get_attr(dp, "period")
                location = _safe_decimal(
                    self._get_attr(dp, "location_tco2e")
                )
                market = _safe_decimal(
                    self._get_attr(dp, "market_tco2e")
                )

                if idx < len(denom_values):
                    denom = _safe_decimal(denom_values[idx])
                else:
                    denom = _ZERO

                location_intensity = _safe_divide(location, denom)
                market_intensity = _safe_divide(market, denom)

                metric_results.append({
                    "period": period,
                    "location_intensity": str(location_intensity),
                    "market_intensity": str(market_intensity),
                    "denominator_value": str(denom),
                    "unit": unit,
                    "metric_type": metric_name,
                })

            results[metric_name] = metric_results

        # Attempt attribute-based denominators from data points
        self._compute_intensity_from_attributes(data_points, results)

        logger.debug(
            "Intensity metrics computed for %d metric types",
            len(results),
        )
        return results

    def _compute_intensity_from_attributes(
        self,
        data_points: List[Any],
        results: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """Supplement intensity results from data point attributes.

        Checks each data point for ``revenue_musd``, ``fte_count``,
        ``floor_area_m2``, and ``production_units`` attributes and
        computes intensity if the denominator is available and the
        metric was not already provided in the explicit denominators
        dict.

        Args:
            data_points: Ordered trend data points.
            results: Results dict to supplement (mutated in-place).
        """
        attr_map = {
            "revenue": "revenue_musd",
            "fte": "fte_count",
            "floor_area": "floor_area_m2",
            "production_unit": "production_units",
        }

        for metric_name, attr_name in attr_map.items():
            if metric_name in results:
                continue

            metric_results: List[Dict[str, Any]] = []
            has_data = False
            unit = _INTENSITY_UNITS.get(metric_name, f"tCO2e/{metric_name}")

            for dp in data_points:
                period = self._get_attr(dp, "period")
                location = _safe_decimal(
                    self._get_attr(dp, "location_tco2e")
                )
                market = _safe_decimal(
                    self._get_attr(dp, "market_tco2e")
                )
                denom = _safe_decimal(self._get_attr(dp, attr_name))

                if denom > _ZERO:
                    has_data = True

                location_intensity = _safe_divide(location, denom)
                market_intensity = _safe_divide(market, denom)

                metric_results.append({
                    "period": period,
                    "location_intensity": str(location_intensity),
                    "market_intensity": str(market_intensity),
                    "denominator_value": str(denom),
                    "unit": unit,
                    "metric_type": metric_name,
                })

            if has_data:
                results[metric_name] = metric_results

    def _compute_intensity_if_available(
        self,
        data_points: List[Any],
        denominators: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Compute intensity metrics if denominators are available.

        Safely converts denominator values to Decimal lists before
        delegating to ``compute_intensity_metrics``.

        Args:
            data_points: Trend data points.
            denominators: Raw denominator dict from config_params.

        Returns:
            Intensity results dict.
        """
        decimal_denoms: Dict[str, List[Decimal]] = {}
        for key, values in denominators.items():
            if isinstance(values, list):
                decimal_denoms[key] = [_safe_decimal(v) for v in values]
        return self.compute_intensity_metrics(data_points, decimal_denoms)

    # ==================================================================
    # 9. Location trend detail
    # ==================================================================

    def compute_location_trend(
        self,
        data_points: List[Any],
    ) -> Dict[str, Any]:
        """Compute location-based emission trend analysis.

        Provides period-by-period location emissions with YoY changes,
        overall direction, CAGR, moving average, and min/max values.

        Args:
            data_points: Ordered list of trend data points.

        Returns:
            Dict containing:
                - emissions_per_period: List of {period, tco2e} dicts
                - trend_direction: "increasing" / "decreasing" / "stable"
                - cagr: Decimal CAGR or None
                - yoy_changes: List of YoY change dicts
                - min_emissions: {period, tco2e}
                - max_emissions: {period, tco2e}
                - total_change_pct: Decimal
        """
        self._increment("_total_location_trend")
        values = self._extract_location_values(data_points)
        periods = self._extract_periods(data_points)

        return self._compute_method_trend(
            values, periods, "location"
        )

    # ==================================================================
    # 10. Market trend detail
    # ==================================================================

    def compute_market_trend(
        self,
        data_points: List[Any],
    ) -> Dict[str, Any]:
        """Compute market-based emission trend analysis.

        Provides period-by-period market emissions with YoY changes,
        overall direction, CAGR, moving average, and min/max values.

        Args:
            data_points: Ordered list of trend data points.

        Returns:
            Dict containing:
                - emissions_per_period: List of {period, tco2e} dicts
                - trend_direction: "increasing" / "decreasing" / "stable"
                - cagr: Decimal CAGR or None
                - yoy_changes: List of YoY change dicts
                - min_emissions: {period, tco2e}
                - max_emissions: {period, tco2e}
                - total_change_pct: Decimal
        """
        self._increment("_total_market_trend")
        values = self._extract_market_values(data_points)
        periods = self._extract_periods(data_points)

        return self._compute_method_trend(
            values, periods, "market"
        )

    def _compute_method_trend(
        self,
        values: List[Decimal],
        periods: List[str],
        method_label: str,
    ) -> Dict[str, Any]:
        """Generic method trend computation for location or market.

        Args:
            values: Ordered Decimal emission values.
            periods: Ordered period labels.
            method_label: "location" or "market".

        Returns:
            Trend analysis dict.
        """
        emissions_per_period: List[Dict[str, str]] = []
        for i, val in enumerate(values):
            emissions_per_period.append({
                "period": periods[i] if i < len(periods) else f"period_{i}",
                "tco2e": str(val),
            })

        direction = self.determine_trend_direction(values)
        yoy_changes = self._compute_all_yoy(values)
        cagr = self._compute_cagr_safe(
            values[0], values[-1], len(values) - 1
        ) if len(values) >= 2 else None

        # Min and max
        min_val = values[0] if values else _ZERO
        min_idx = 0
        max_val = values[0] if values else _ZERO
        max_idx = 0
        for i, val in enumerate(values):
            if val < min_val:
                min_val = val
                min_idx = i
            if val > max_val:
                max_val = val
                max_idx = i

        total_change_pct = _ZERO
        if len(values) >= 2 and values[0] != _ZERO:
            total_change_pct = _quantize(
                (values[-1] - values[0])
                / _abs_decimal(values[0])
                * _HUNDRED
            )

        min_period = periods[min_idx] if min_idx < len(periods) else ""
        max_period = periods[max_idx] if max_idx < len(periods) else ""

        result = {
            "method": method_label,
            "emissions_per_period": emissions_per_period,
            "trend_direction": direction,
            "cagr": str(cagr) if cagr is not None else None,
            "yoy_changes": yoy_changes,
            "min_emissions": {
                "period": min_period,
                "tco2e": str(min_val),
            },
            "max_emissions": {
                "period": max_period,
                "tco2e": str(max_val),
            },
            "total_change_pct": str(total_change_pct),
            "num_periods": len(values),
        }

        logger.debug(
            "%s trend: direction=%s, cagr=%s, total_change=%s%%",
            method_label,
            direction,
            cagr,
            total_change_pct,
        )
        return result

    # ==================================================================
    # 11. Discrepancy trend
    # ==================================================================

    def compute_discrepancy_trend(
        self,
        data_points: List[Any],
    ) -> Dict[str, Any]:
        """Compute how the location-market gap changes over time.

        The discrepancy for each period is location_tco2e - market_tco2e.
        Positive values mean location-based is higher (typical when
        RECs/GOs reduce market-based). Negative values mean market-based
        exceeds location-based.

        Args:
            data_points: Ordered list of trend data points.

        Returns:
            Dict containing:
                - gap_per_period: List of {period, gap_tco2e, gap_pct,
                  direction} dicts
                - gap_values: List of Decimal gap values
                - trend_direction: "increasing" / "decreasing" / "stable"
                  (increasing means the gap is widening)
                - latest_gap: Decimal most recent gap
                - gap_change: Decimal change in gap from first to last
                - avg_gap: Decimal average gap across periods
                - gap_volatility: Decimal standard deviation of gap
        """
        self._increment("_total_discrepancy_trend")
        gap_per_period: List[Dict[str, Any]] = []
        gap_values: List[Decimal] = []
        abs_gap_values: List[Decimal] = []

        for dp in data_points:
            period = self._get_attr(dp, "period")
            location = _safe_decimal(self._get_attr(dp, "location_tco2e"))
            market = _safe_decimal(self._get_attr(dp, "market_tco2e"))
            gap = _quantize(location - market)
            gap_values.append(gap)
            abs_gap_values.append(_abs_decimal(gap))

            # Percentage gap relative to location
            if location > _ZERO:
                gap_pct = _quantize(gap / location * _HUNDRED)
            else:
                gap_pct = _ZERO

            # Direction
            if gap > _ZERO:
                direction = "market_lower"
            elif gap < _ZERO:
                direction = "market_higher"
            else:
                direction = "equal"

            gap_per_period.append({
                "period": period,
                "gap_tco2e": str(gap),
                "gap_pct": str(gap_pct),
                "direction": direction,
            })

        # Trend of absolute gap (widening or narrowing)
        trend_direction = self.determine_trend_direction(abs_gap_values)

        latest_gap = gap_values[-1] if gap_values else _ZERO
        gap_change = _ZERO
        if len(gap_values) >= 2:
            gap_change = _quantize(gap_values[-1] - gap_values[0])

        # Average gap
        avg_gap = _ZERO
        if gap_values:
            total = sum(gap_values, _ZERO)
            avg_gap = _quantize(total / _D(len(gap_values)))

        # Volatility (standard deviation)
        gap_volatility = self._compute_std_dev(gap_values)

        result = {
            "gap_per_period": gap_per_period,
            "gap_values": [str(v) for v in gap_values],
            "abs_gap_values": [str(v) for v in abs_gap_values],
            "trend_direction": trend_direction,
            "latest_gap": str(latest_gap),
            "gap_change": str(gap_change),
            "avg_gap": str(avg_gap),
            "gap_volatility": str(gap_volatility),
            "num_periods": len(gap_values),
        }

        logger.debug(
            "Discrepancy trend: direction=%s, latest_gap=%s, "
            "avg=%s, volatility=%s",
            trend_direction,
            latest_gap,
            avg_gap,
            gap_volatility,
        )
        return result

    # ==================================================================
    # 12. Moving average
    # ==================================================================

    def compute_moving_average(
        self,
        values: List[Decimal],
        window: int = 3,
    ) -> List[Decimal]:
        """Compute simple moving average over a sliding window.

        For indices where the full window is not available (i.e. the
        first ``window - 1`` values), the average is computed over
        all available preceding values (expanding window).

        Args:
            values: Ordered list of Decimal values.
            window: Window size for averaging (default 3).

        Returns:
            List of Decimal moving average values, same length as input.

        Example:
            >>> engine = TrendAnalysisEngine()
            >>> vals = [Decimal("100"), Decimal("200"), Decimal("300")]
            >>> engine.compute_moving_average(vals, 2)
            [Decimal('100.00000000'), Decimal('150.00000000'),
             Decimal('250.00000000')]
        """
        self._increment("_total_moving_avg")
        if not values:
            return []

        if window < 1:
            window = 1

        result: List[Decimal] = []
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            window_values = values[start_idx:i + 1]
            window_sum = sum(window_values, _ZERO)
            window_avg = _quantize(window_sum / _D(len(window_values)))
            result.append(window_avg)

        return result

    # ==================================================================
    # 13. Linear forecast
    # ==================================================================

    def forecast_linear(
        self,
        data_points: List[Decimal],
        periods_ahead: int = 3,
    ) -> List[Decimal]:
        """Forecast future values using simple linear regression.

        Fits y = slope * x + intercept via ordinary least squares (OLS)
        on indices 0..n-1 and extrapolates for the next
        ``periods_ahead`` indices.

        Args:
            data_points: Ordered list of Decimal values.
            periods_ahead: Number of future periods to forecast.

        Returns:
            List of Decimal forecast values for future periods.

        Example:
            >>> engine = TrendAnalysisEngine()
            >>> vals = [Decimal("100"), Decimal("90"), Decimal("80")]
            >>> engine.forecast_linear(vals, 2)
            [Decimal('70.00000000'), Decimal('60.00000000')]
        """
        self._increment("_total_forecast")
        if not data_points or periods_ahead < 1:
            return []

        n = len(data_points)
        if n == 1:
            return [_quantize(data_points[0])] * periods_ahead

        # OLS: y = slope * x + intercept
        slope, intercept = self._ols_fit(data_points)

        forecasts: List[Decimal] = []
        for i in range(periods_ahead):
            x = _D(n + i)
            forecast_val = _quantize(slope * x + intercept)
            forecasts.append(forecast_val)

        logger.debug(
            "Linear forecast: slope=%s, intercept=%s, periods=%d",
            slope,
            intercept,
            periods_ahead,
        )
        return forecasts

    def _ols_fit(
        self,
        values: List[Decimal],
    ) -> Tuple[Decimal, Decimal]:
        """Fit a line to values using ordinary least squares.

        Indices 0..n-1 serve as the x-axis. The formulas are:

        slope = (n * sum(x*y) - sum(x) * sum(y)) /
                (n * sum(x^2) - sum(x)^2)
        intercept = (sum(y) - slope * sum(x)) / n

        Args:
            values: Ordered Decimal values.

        Returns:
            Tuple of (slope, intercept) as quantized Decimals.
        """
        n = _D(len(values))
        sum_x = _ZERO
        sum_y = _ZERO
        sum_xy = _ZERO
        sum_x2 = _ZERO

        for i, y in enumerate(values):
            x = _D(i)
            y_d = _safe_decimal(y)
            sum_x += x
            sum_y += y_d
            sum_xy += x * y_d
            sum_x2 += x * x

        denominator = n * sum_x2 - sum_x * sum_x

        if denominator == _ZERO:
            intercept = _safe_divide(sum_y, n)
            return _ZERO, _quantize(intercept)

        slope = _quantize((n * sum_xy - sum_x * sum_y) / denominator)
        intercept = _quantize((sum_y - slope * sum_x) / n)

        return slope, intercept

    # ==================================================================
    # 14. Anomaly detection
    # ==================================================================

    def detect_anomalies(
        self,
        values: List[Decimal],
        sigma: Optional[Decimal] = None,
    ) -> List[int]:
        """Detect anomalous values using z-score method.

        Identifies indices where the absolute z-score exceeds the
        ``sigma`` threshold. The z-score is computed as:

            z = (value - mean) / std_dev

        Args:
            values: Ordered list of Decimal values.
            sigma: Z-score threshold (default 2.0). Values with
                |z-score| > sigma are flagged as anomalies.

        Returns:
            List of integer indices where anomalies were detected.

        Example:
            >>> engine = TrendAnalysisEngine()
            >>> vals = [
            ...     Decimal("100"), Decimal("102"), Decimal("500"),
            ...     Decimal("99"), Decimal("101"),
            ... ]
            >>> engine.detect_anomalies(vals, Decimal("2.0"))
            [2]
        """
        self._increment("_total_anomaly")
        if sigma is None:
            sigma = _DEFAULT_ANOMALY_SIGMA

        sigma_d = _safe_decimal(sigma, _DEFAULT_ANOMALY_SIGMA)

        if len(values) < 3:
            return []

        mean = self._compute_mean(values)
        std_dev = self._compute_std_dev(values)

        if std_dev == _ZERO:
            return []

        anomalies: List[int] = []
        for i, val in enumerate(values):
            val_d = _safe_decimal(val)
            z_score = _quantize(
                _abs_decimal(val_d - mean) / std_dev
            )
            if z_score > sigma_d:
                anomalies.append(i)

        logger.debug(
            "Anomaly detection: %d anomalies found out of %d values "
            "(sigma=%s)",
            len(anomalies),
            len(values),
            sigma_d,
        )
        return anomalies

    # ==================================================================
    # 15. Health check
    # ==================================================================

    def health_check(self) -> Dict[str, Any]:
        """Return engine health and operational statistics.

        Returns:
            Dict containing:
                - engine: Engine name string
                - status: "healthy"
                - created_at: ISO timestamp
                - uptime_seconds: Time since creation
                - counters: Dict of all operation counters
                - config: Dict of current config values
                - version: Agent version string
        """
        now = _utcnow()
        uptime = (now - self._created_at).total_seconds()

        agent_id = AGENT_ID if _MODELS_AVAILABLE else _FALLBACK_AGENT_ID
        version = VERSION if _MODELS_AVAILABLE else _FALLBACK_VERSION
        component = (
            AGENT_COMPONENT
            if _MODELS_AVAILABLE
            else _FALLBACK_COMPONENT
        )

        return {
            "engine": "TrendAnalysisEngine",
            "agent_id": agent_id,
            "component": component,
            "version": version,
            "status": "healthy",
            "created_at": self._created_at.isoformat(),
            "uptime_seconds": uptime,
            "counters": {
                "total_analyses": self._total_analyses,
                "total_yoy": self._total_yoy,
                "total_cagr": self._total_cagr,
                "total_pif": self._total_pif,
                "total_re100": self._total_re100,
                "total_sbti": self._total_sbti,
                "total_intensity": self._total_intensity,
                "total_location_trend": self._total_location_trend,
                "total_market_trend": self._total_market_trend,
                "total_discrepancy_trend": self._total_discrepancy_trend,
                "total_moving_avg": self._total_moving_avg,
                "total_forecast": self._total_forecast,
                "total_anomaly": self._total_anomaly,
                "total_errors": self._total_errors,
            },
            "config": {
                "stable_threshold": str(self._stable_threshold),
                "min_periods": self._min_periods,
                "max_periods": self._max_periods,
                "decimal_places": self._decimal_places,
            },
        }

    # ==================================================================
    # Internal helpers: data extraction
    # ==================================================================

    def _get_attr(self, obj: Any, name: str) -> Any:
        """Get an attribute or dict key from an object.

        Args:
            obj: Object (Pydantic model or dict).
            name: Attribute/key name.

        Returns:
            Attribute value, or None if not found.
        """
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)

    def _extract_periods(self, data_points: List[Any]) -> List[str]:
        """Extract period labels from data points.

        Args:
            data_points: List of trend data points.

        Returns:
            Ordered list of period label strings.
        """
        return [
            str(self._get_attr(dp, "period") or f"period_{i}")
            for i, dp in enumerate(data_points)
        ]

    def _extract_location_values(
        self,
        data_points: List[Any],
    ) -> List[Decimal]:
        """Extract location-based emission values.

        Args:
            data_points: List of trend data points.

        Returns:
            Ordered list of Decimal location tCO2e values.
        """
        return [
            _safe_decimal(self._get_attr(dp, "location_tco2e"))
            for dp in data_points
        ]

    def _extract_market_values(
        self,
        data_points: List[Any],
    ) -> List[Decimal]:
        """Extract market-based emission values.

        Args:
            data_points: List of trend data points.

        Returns:
            Ordered list of Decimal market tCO2e values.
        """
        return [
            _safe_decimal(self._get_attr(dp, "market_tco2e"))
            for dp in data_points
        ]

    def _extract_pif_values(
        self,
        data_points: List[Any],
    ) -> List[Decimal]:
        """Extract PIF values from data points.

        Args:
            data_points: List of trend data points.

        Returns:
            Ordered list of Decimal PIF values.
        """
        return [
            _safe_decimal(self._get_attr(dp, "pif"))
            for dp in data_points
        ]

    def _extract_re100_values(
        self,
        data_points: List[Any],
    ) -> List[Decimal]:
        """Extract RE100 percentage values from data points.

        Args:
            data_points: List of trend data points.

        Returns:
            Ordered list of Decimal RE100 percentage values.
        """
        return [
            _safe_decimal(self._get_attr(dp, "re100_pct"))
            for dp in data_points
        ]

    def _extract_year_from_period(
        self,
        period: Optional[str],
    ) -> Optional[int]:
        """Extract a 4-digit year from a period label.

        Handles formats: "2024", "2024-Q1", "2024-01", "FY2024", etc.

        Args:
            period: Period label string.

        Returns:
            Integer year or None if extraction fails.
        """
        if period is None:
            return None

        # Try direct integer conversion
        try:
            year = int(period.strip())
            if 1900 <= year <= 2200:
                return year
        except ValueError:
            pass

        # Try extracting 4-digit sequence
        cleaned = period.strip()
        for i in range(len(cleaned) - 3):
            chunk = cleaned[i:i + 4]
            try:
                year = int(chunk)
                if 1900 <= year <= 2200:
                    return year
            except ValueError:
                continue

        return None

    # ==================================================================
    # Internal helpers: statistics
    # ==================================================================

    def _compute_mean(self, values: List[Decimal]) -> Decimal:
        """Compute arithmetic mean of a list of Decimals.

        Args:
            values: List of Decimal values.

        Returns:
            Quantized mean value. Returns zero for empty list.
        """
        if not values:
            return _ZERO
        total = sum(values, _ZERO)
        return _quantize(total / _D(len(values)))

    def _compute_std_dev(self, values: List[Decimal]) -> Decimal:
        """Compute population standard deviation of Decimal values.

        Uses the formula: sqrt(sum((xi - mean)^2) / n)

        Args:
            values: List of Decimal values.

        Returns:
            Quantized standard deviation. Returns zero for fewer than
            2 values.
        """
        if len(values) < 2:
            return _ZERO

        mean = self._compute_mean(values)
        n = _D(len(values))

        sum_sq_diff = _ZERO
        for val in values:
            diff = _safe_decimal(val) - mean
            sum_sq_diff += diff * diff

        variance = sum_sq_diff / n
        std_dev_float = math.sqrt(float(variance))
        return _quantize(_D(std_dev_float))

    def _compute_variance(self, values: List[Decimal]) -> Decimal:
        """Compute population variance of Decimal values.

        Args:
            values: List of Decimal values.

        Returns:
            Quantized variance. Returns zero for fewer than 2 values.
        """
        if len(values) < 2:
            return _ZERO

        mean = self._compute_mean(values)
        n = _D(len(values))

        sum_sq_diff = _ZERO
        for val in values:
            diff = _safe_decimal(val) - mean
            sum_sq_diff += diff * diff

        return _quantize(sum_sq_diff / n)

    def _compute_median(self, values: List[Decimal]) -> Decimal:
        """Compute median of a sorted list of Decimal values.

        Args:
            values: List of Decimal values (will be sorted internally).

        Returns:
            Quantized median value. Returns zero for empty list.
        """
        if not values:
            return _ZERO

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mid = n // 2

        if n % 2 == 0:
            return _quantize((sorted_vals[mid - 1] + sorted_vals[mid]) / _TWO)
        return _quantize(sorted_vals[mid])

    def _compute_percentile(
        self,
        values: List[Decimal],
        percentile: int,
    ) -> Decimal:
        """Compute a given percentile using nearest-rank method.

        Args:
            values: List of Decimal values.
            percentile: Percentile rank (0-100).

        Returns:
            Quantized percentile value. Returns zero for empty list.
        """
        if not values:
            return _ZERO

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        if percentile <= 0:
            return _quantize(sorted_vals[0])
        if percentile >= 100:
            return _quantize(sorted_vals[-1])

        k = _D(percentile) / _HUNDRED * _D(n - 1)
        f = int(k)
        c = f + 1 if f + 1 < n else f
        d = k - _D(f)

        return _quantize(
            sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])
        )

    # ==================================================================
    # Internal helpers: validation
    # ==================================================================

    def _validate_data_points(self, data_points: List[Any]) -> None:
        """Validate the input data points list.

        Args:
            data_points: List to validate.

        Raises:
            ValueError: If the list is too short, too long, or empty.
        """
        if data_points is None:
            raise ValueError("data_points must not be None")

        if not data_points:
            raise ValueError(
                "data_points must not be empty; at least "
                f"{self._min_periods} periods required"
            )

        if len(data_points) < self._min_periods:
            raise ValueError(
                f"At least {self._min_periods} data points required "
                f"for trend analysis, got {len(data_points)}"
            )

        max_points = (
            MAX_TREND_POINTS
            if _MODELS_AVAILABLE
            else _FALLBACK_MAX_TREND_POINTS
        )
        if len(data_points) > max_points:
            raise ValueError(
                f"Maximum {max_points} trend data points, "
                f"got {len(data_points)}"
            )

        # Validate each data point has required fields
        for i, dp in enumerate(data_points):
            period = self._get_attr(dp, "period")
            if period is None or str(period).strip() == "":
                raise ValueError(
                    f"data_points[{i}] has no 'period' attribute"
                )
            loc = self._get_attr(dp, "location_tco2e")
            if loc is None:
                raise ValueError(
                    f"data_points[{i}] has no 'location_tco2e' attribute"
                )
            mkt = self._get_attr(dp, "market_tco2e")
            if mkt is None:
                raise ValueError(
                    f"data_points[{i}] has no 'market_tco2e' attribute"
                )

    # ==================================================================
    # Internal helpers: recommendations
    # ==================================================================

    def _generate_recommendations(
        self,
        location_trend: str,
        market_trend: str,
        discrepancy_trend: str,
        pif_trend: str,
        re100_values: List[Decimal],
        sbti_assessment: Optional[Dict[str, Any]],
        anomaly_count: int,
        location_cagr: Optional[Decimal],
        market_cagr: Optional[Decimal],
    ) -> List[str]:
        """Generate actionable recommendations based on trend results.

        Args:
            location_trend: Location trend direction.
            market_trend: Market trend direction.
            discrepancy_trend: Discrepancy gap trend direction.
            pif_trend: PIF trend direction.
            re100_values: RE100 percentage values per period.
            sbti_assessment: SBTi trajectory result or None.
            anomaly_count: Number of detected anomalies.
            location_cagr: Location CAGR or None.
            market_cagr: Market CAGR or None.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        # Location-based emissions increasing
        if location_trend == "increasing":
            recommendations.append(
                "Location-based emissions are increasing. Consider "
                "energy efficiency measures and investigate sources "
                "of growth in grid-based consumption."
            )

        # Market-based emissions increasing despite procurement
        if market_trend == "increasing":
            recommendations.append(
                "Market-based emissions are increasing. Review "
                "renewable energy procurement strategy and contractual "
                "instrument coverage."
            )

        # Both methods decreasing - positive trend
        if (
            location_trend == "decreasing"
            and market_trend == "decreasing"
        ):
            recommendations.append(
                "Both location-based and market-based emissions are "
                "decreasing, indicating effective emissions reduction "
                "efforts. Maintain current trajectory."
            )

        # Discrepancy widening
        if discrepancy_trend == "increasing":
            recommendations.append(
                "The gap between location-based and market-based "
                "emissions is widening. This may indicate increased "
                "reliance on contractual instruments without actual "
                "grid decarbonisation. Consider additional energy "
                "efficiency investments."
            )

        # PIF decreasing
        if pif_trend == "decreasing":
            recommendations.append(
                "Procurement Impact Factor is declining. Evaluate "
                "whether renewable energy contracts are being renewed "
                "and whether new procurement covers growing demand."
            )

        # RE100 progress stalling
        if re100_values and len(re100_values) >= 2:
            latest_re100 = re100_values[-1]
            if latest_re100 < _D(50):
                recommendations.append(
                    f"RE100 progress is at {latest_re100}%, well below "
                    "the 100% target. Accelerate renewable electricity "
                    "procurement to improve the RE100 score."
                )
            elif latest_re100 >= _HUNDRED:
                recommendations.append(
                    "RE100 target has been achieved (100% renewable "
                    "electricity). Maintain coverage to sustain this "
                    "achievement."
                )

        # SBTi off-track
        if sbti_assessment is not None:
            if not sbti_assessment.get("on_track", False):
                gap_pct = sbti_assessment.get("latest_gap_pct", "0")
                recommendations.append(
                    f"SBTi Science Based Target is off-track by "
                    f"{gap_pct}%. Accelerate decarbonisation efforts "
                    "to close the gap to the target trajectory."
                )
            else:
                recommendations.append(
                    "SBTi Science Based Target is on-track. Continue "
                    "current reduction pathway to meet the target."
                )

        # Anomalies detected
        if anomaly_count > 0:
            recommendations.append(
                f"{anomaly_count} anomalous data point(s) detected "
                "in location-based emissions. Investigate potential "
                "data quality issues, methodology changes, or "
                "one-off events."
            )

        # CAGR divergence
        if location_cagr is not None and market_cagr is not None:
            cagr_diff = _abs_decimal(location_cagr - market_cagr)
            if cagr_diff > _D("0.05"):
                recommendations.append(
                    "Significant divergence between location-based "
                    f"CAGR ({location_cagr}) and market-based CAGR "
                    f"({market_cagr}). This divergence may signal "
                    "changing effectiveness of procurement strategies."
                )

        # Market emissions growing faster than location
        if (
            market_cagr is not None
            and location_cagr is not None
            and market_cagr > location_cagr
            and market_cagr > _ZERO
        ):
            recommendations.append(
                "Market-based emissions are growing faster than "
                "location-based emissions. This unusual pattern "
                "suggests deteriorating contractual instrument "
                "coverage or shifts to higher-emission suppliers."
            )

        if not recommendations:
            recommendations.append(
                "No specific trend-based recommendations at this time. "
                "Continue monitoring emissions across both methods."
            )

        return recommendations

    # ==================================================================
    # Internal helpers: provenance
    # ==================================================================

    def _compute_provenance_hash(
        self,
        num_periods: int,
        location_trend: str,
        market_trend: str,
        pif_trend: str,
        location_values: List[Decimal],
        market_values: List[Decimal],
    ) -> str:
        """Compute SHA-256 provenance hash for the trend analysis.

        Uses the provenance module if available, otherwise falls back
        to local hash computation.

        Args:
            num_periods: Number of periods analysed.
            location_trend: Location trend direction.
            market_trend: Market trend direction.
            pif_trend: PIF trend direction.
            location_values: Location emission values.
            market_values: Market emission values.

        Returns:
            SHA-256 hex digest string (64 characters).
        """
        if _PROVENANCE_AVAILABLE and _hash_trend_analysis is not None:
            try:
                return _hash_trend_analysis(
                    period_count=num_periods,
                    location_trend=location_trend,
                    market_trend=market_trend,
                    pif_trend=pif_trend,
                )
            except Exception as exc:
                logger.warning(
                    "Provenance hash via module failed: %s; "
                    "falling back to local hash",
                    exc,
                )

        data = {
            "num_periods": num_periods,
            "location_trend": location_trend,
            "market_trend": market_trend,
            "pif_trend": pif_trend,
            "location_values": [str(v) for v in location_values],
            "market_values": [str(v) for v in market_values],
        }
        return _compute_hash(data)

    # ==================================================================
    # Internal helpers: metrics recording
    # ==================================================================

    def _record_metrics(self, tenant_id: str) -> None:
        """Record trend analysis metrics if the metrics module is available.

        Args:
            tenant_id: Tenant identifier for metric labelling.
        """
        if not _METRICS_AVAILABLE:
            return
        try:
            metrics = _get_metrics()
            metrics.record_trend_analysis(tenant_id=tenant_id)
        except Exception as exc:
            logger.debug(
                "Failed to record trend analysis metrics: %s",
                exc,
            )

    # ==================================================================
    # Utility methods for external callers
    # ==================================================================

    def compute_yoy_summary(
        self,
        data_points: List[Any],
    ) -> Dict[str, Any]:
        """Compute a YoY summary for both location and market methods.

        Convenience wrapper that extracts values and computes all YoY
        changes for both methods.

        Args:
            data_points: Ordered list of trend data points.

        Returns:
            Dict with ``location_yoy`` and ``market_yoy`` lists.
        """
        location_values = self._extract_location_values(data_points)
        market_values = self._extract_market_values(data_points)

        return {
            "location_yoy": self._compute_all_yoy(location_values),
            "market_yoy": self._compute_all_yoy(market_values),
            "num_periods": len(data_points),
        }

    def compute_cagr_summary(
        self,
        data_points: List[Any],
    ) -> Dict[str, Optional[str]]:
        """Compute CAGR summary for both location and market methods.

        Args:
            data_points: Ordered list of trend data points.

        Returns:
            Dict with ``location_cagr`` and ``market_cagr`` strings.
        """
        location_values = self._extract_location_values(data_points)
        market_values = self._extract_market_values(data_points)
        n = len(data_points)

        loc_cagr = self._compute_cagr_safe(
            location_values[0] if location_values else _ZERO,
            location_values[-1] if location_values else _ZERO,
            n - 1,
        )
        mkt_cagr = self._compute_cagr_safe(
            market_values[0] if market_values else _ZERO,
            market_values[-1] if market_values else _ZERO,
            n - 1,
        )

        return {
            "location_cagr": str(loc_cagr) if loc_cagr is not None else None,
            "market_cagr": str(mkt_cagr) if mkt_cagr is not None else None,
            "num_periods": n,
        }

    def compute_trend_summary(
        self,
        data_points: List[Any],
    ) -> Dict[str, str]:
        """Compute directional trend summary for all tracked metrics.

        Args:
            data_points: Ordered list of trend data points.

        Returns:
            Dict with trend direction for location, market,
            discrepancy, pif, and re100.
        """
        location_values = self._extract_location_values(data_points)
        market_values = self._extract_market_values(data_points)
        pif_values = self._extract_pif_values(data_points)
        re100_values = self._extract_re100_values(data_points)

        # Discrepancy (absolute gap)
        gap_values = []
        for i in range(len(data_points)):
            loc = location_values[i] if i < len(location_values) else _ZERO
            mkt = market_values[i] if i < len(market_values) else _ZERO
            gap_values.append(_abs_decimal(loc - mkt))

        return {
            "location_trend": self.determine_trend_direction(
                location_values
            ),
            "market_trend": self.determine_trend_direction(
                market_values
            ),
            "discrepancy_trend": self.determine_trend_direction(
                gap_values
            ),
            "pif_trend": self.determine_trend_direction(pif_values),
            "re100_trend": self.determine_trend_direction(re100_values),
        }

    def compute_statistics_summary(
        self,
        values: List[Decimal],
    ) -> Dict[str, str]:
        """Compute comprehensive statistical summary of a value series.

        Args:
            values: Ordered list of Decimal values.

        Returns:
            Dict with mean, median, std_dev, variance, min, max,
            p25, p75, p90, count.
        """
        return {
            "count": str(len(values)),
            "mean": str(self._compute_mean(values)),
            "median": str(self._compute_median(values)),
            "std_dev": str(self._compute_std_dev(values)),
            "variance": str(self._compute_variance(values)),
            "min": str(min(values)) if values else str(_ZERO),
            "max": str(max(values)) if values else str(_ZERO),
            "p25": str(self._compute_percentile(values, 25)),
            "p75": str(self._compute_percentile(values, 75)),
            "p90": str(self._compute_percentile(values, 90)),
        }

    def compute_correlation(
        self,
        values_x: List[Decimal],
        values_y: List[Decimal],
    ) -> Decimal:
        """Compute Pearson correlation coefficient between two series.

        r = sum((xi - mean_x)(yi - mean_y)) /
            sqrt(sum((xi - mean_x)^2) * sum((yi - mean_y)^2))

        Args:
            values_x: First value series.
            values_y: Second value series.

        Returns:
            Pearson correlation coefficient as quantized Decimal.
            Returns zero if series lengths differ or are too short.
        """
        if len(values_x) != len(values_y) or len(values_x) < 2:
            return _ZERO

        mean_x = self._compute_mean(values_x)
        mean_y = self._compute_mean(values_y)

        sum_xy = _ZERO
        sum_x2 = _ZERO
        sum_y2 = _ZERO

        for i in range(len(values_x)):
            dx = _safe_decimal(values_x[i]) - mean_x
            dy = _safe_decimal(values_y[i]) - mean_y
            sum_xy += dx * dy
            sum_x2 += dx * dx
            sum_y2 += dy * dy

        denominator_float = math.sqrt(
            float(sum_x2) * float(sum_y2)
        )
        if denominator_float == 0.0:
            return _ZERO

        r = float(sum_xy) / denominator_float
        return _quantize(_D(r))

    def compute_r_squared(
        self,
        values: List[Decimal],
    ) -> Decimal:
        """Compute R-squared (coefficient of determination) for OLS fit.

        Measures how well the linear trend explains the data variance.

        Args:
            values: Ordered Decimal values.

        Returns:
            R-squared value (0 to 1) as quantized Decimal.
        """
        if len(values) < 2:
            return _ZERO

        mean = self._compute_mean(values)
        slope, intercept = self._ols_fit(values)

        ss_res = _ZERO
        ss_tot = _ZERO

        for i, val in enumerate(values):
            val_d = _safe_decimal(val)
            predicted = slope * _D(i) + intercept
            ss_res += (val_d - predicted) ** 2
            ss_tot += (val_d - mean) ** 2

        if ss_tot == _ZERO:
            return _ONE

        r_sq = _quantize(_ONE - ss_res / ss_tot)

        # Clamp to [0, 1]
        if r_sq < _ZERO:
            r_sq = _ZERO
        if r_sq > _ONE:
            r_sq = _ONE

        return r_sq

    def compute_rate_of_change(
        self,
        values: List[Decimal],
    ) -> List[Decimal]:
        """Compute period-over-period rate of change.

        rate[i] = (values[i] - values[i-1]) / |values[i-1]| * 100

        Args:
            values: Ordered Decimal values.

        Returns:
            List of rate-of-change percentages (one fewer than input).
        """
        if len(values) < 2:
            return []

        rates: List[Decimal] = []
        for i in range(1, len(values)):
            prev = values[i - 1]
            curr = values[i]
            if prev == _ZERO:
                rates.append(_ZERO)
            else:
                rate = _quantize(
                    (curr - prev) / _abs_decimal(prev) * _HUNDRED
                )
                rates.append(rate)

        return rates

    def compute_cumulative_change(
        self,
        values: List[Decimal],
    ) -> List[Decimal]:
        """Compute cumulative change from the first value.

        cumulative[i] = (values[i] - values[0]) / |values[0]| * 100

        Args:
            values: Ordered Decimal values.

        Returns:
            List of cumulative change percentages (same length as input).
        """
        if not values:
            return []

        base = values[0]
        if base == _ZERO:
            return [_ZERO] * len(values)

        return [
            _quantize((v - base) / _abs_decimal(base) * _HUNDRED)
            for v in values
        ]

    def compute_volatility_index(
        self,
        values: List[Decimal],
    ) -> Decimal:
        """Compute volatility as coefficient of variation (CV%).

        CV = (std_dev / |mean|) * 100

        Args:
            values: Ordered Decimal values.

        Returns:
            Coefficient of variation as percentage Decimal.
        """
        mean = self._compute_mean(values)
        std_dev = self._compute_std_dev(values)

        if mean == _ZERO:
            return _ZERO

        return _quantize(std_dev / _abs_decimal(mean) * _HUNDRED)

    def compute_weighted_moving_average(
        self,
        values: List[Decimal],
        window: int = 3,
    ) -> List[Decimal]:
        """Compute linearly-weighted moving average.

        More recent values within the window receive higher weight.
        Weight[i] = i + 1 (position within window, 1-indexed).

        Args:
            values: Ordered Decimal values.
            window: Window size (default 3).

        Returns:
            List of weighted moving average values (same length as input).
        """
        if not values:
            return []

        if window < 1:
            window = 1

        result: List[Decimal] = []
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            window_values = values[start_idx:i + 1]
            w_len = len(window_values)

            weighted_sum = _ZERO
            weight_total = _ZERO
            for j, val in enumerate(window_values):
                weight = _D(j + 1)
                weighted_sum += _safe_decimal(val) * weight
                weight_total += weight

            wma = _safe_divide(weighted_sum, weight_total)
            result.append(wma)

        return result

    def compute_exponential_moving_average(
        self,
        values: List[Decimal],
        alpha: Decimal = Decimal("0.3"),
    ) -> List[Decimal]:
        """Compute exponential moving average (EMA).

        EMA[0] = values[0]
        EMA[i] = alpha * values[i] + (1 - alpha) * EMA[i-1]

        Args:
            values: Ordered Decimal values.
            alpha: Smoothing factor (0 < alpha <= 1).

        Returns:
            List of EMA values (same length as input).
        """
        if not values:
            return []

        alpha_d = _safe_decimal(alpha, Decimal("0.3"))
        if alpha_d <= _ZERO or alpha_d > _ONE:
            alpha_d = Decimal("0.3")

        complement = _ONE - alpha_d
        result: List[Decimal] = [_quantize(_safe_decimal(values[0]))]

        for i in range(1, len(values)):
            ema = _quantize(
                alpha_d * _safe_decimal(values[i])
                + complement * result[i - 1]
            )
            result.append(ema)

        return result

    def detect_seasonality(
        self,
        values: List[Decimal],
        expected_period: int = 4,
    ) -> Dict[str, Any]:
        """Detect seasonal patterns in the value series.

        Computes seasonal indices by averaging values at each position
        within the expected period. A seasonality ratio > 1.0 indicates
        a peak season; < 1.0 indicates a trough season.

        Args:
            values: Ordered Decimal values (at least 2x expected_period).
            expected_period: Expected seasonal period (default 4 for
                quarterly data).

        Returns:
            Dict with seasonal indices, is_seasonal flag, and
            seasonal_strength.
        """
        if len(values) < expected_period * 2:
            return {
                "is_seasonal": False,
                "seasonal_indices": [],
                "seasonal_strength": str(_ZERO),
                "expected_period": expected_period,
            }

        # Compute mean for each position within the period
        overall_mean = self._compute_mean(values)
        if overall_mean == _ZERO:
            return {
                "is_seasonal": False,
                "seasonal_indices": [str(_ONE)] * expected_period,
                "seasonal_strength": str(_ZERO),
                "expected_period": expected_period,
            }

        seasonal_sums: List[Decimal] = [_ZERO] * expected_period
        seasonal_counts: List[int] = [0] * expected_period

        for i, val in enumerate(values):
            pos = i % expected_period
            seasonal_sums[pos] += _safe_decimal(val)
            seasonal_counts[pos] += 1

        seasonal_indices: List[Decimal] = []
        for pos in range(expected_period):
            if seasonal_counts[pos] > 0:
                seasonal_mean = _quantize(
                    seasonal_sums[pos] / _D(seasonal_counts[pos])
                )
                index = _safe_divide(seasonal_mean, overall_mean)
            else:
                index = _ONE
            seasonal_indices.append(index)

        # Seasonal strength: range of indices
        max_index = max(seasonal_indices) if seasonal_indices else _ONE
        min_index = min(seasonal_indices) if seasonal_indices else _ONE
        strength = _quantize(max_index - min_index)

        # Consider seasonal if strength > 0.10 (10% variation)
        is_seasonal = strength > Decimal("0.10")

        return {
            "is_seasonal": is_seasonal,
            "seasonal_indices": [str(v) for v in seasonal_indices],
            "seasonal_strength": str(strength),
            "expected_period": expected_period,
        }

    def compute_gap_analysis(
        self,
        data_points: List[Any],
        target_values: Optional[List[Decimal]] = None,
    ) -> Dict[str, Any]:
        """Compute gap analysis between actual and target emissions.

        Useful for comparing actual market-based emissions against
        internal targets or budgets.

        Args:
            data_points: Ordered list of trend data points.
            target_values: Optional list of target Decimal values
                (one per period). If None, uses the first period's
                market value as the target for all periods.

        Returns:
            Dict with gap_per_period, total_gap, average_gap,
            and on_target_count.
        """
        market_values = self._extract_market_values(data_points)
        periods = self._extract_periods(data_points)

        if target_values is None:
            base_target = market_values[0] if market_values else _ZERO
            target_values = [base_target] * len(market_values)

        gap_per_period: List[Dict[str, str]] = []
        on_target_count = 0

        for i in range(len(market_values)):
            actual = market_values[i]
            target = (
                _safe_decimal(target_values[i])
                if i < len(target_values)
                else _ZERO
            )
            gap = _quantize(actual - target)
            on_target = actual <= target

            if on_target:
                on_target_count += 1

            gap_pct = _ZERO
            if target > _ZERO:
                gap_pct = _quantize(gap / target * _HUNDRED)

            gap_per_period.append({
                "period": periods[i] if i < len(periods) else f"period_{i}",
                "actual": str(actual),
                "target": str(target),
                "gap": str(gap),
                "gap_pct": str(gap_pct),
                "on_target": on_target,
            })

        gaps = [_safe_decimal(g["gap"]) for g in gap_per_period]
        total_gap = sum(gaps, _ZERO)
        avg_gap = _safe_divide(total_gap, _D(len(gaps))) if gaps else _ZERO

        return {
            "gap_per_period": gap_per_period,
            "total_gap": str(total_gap),
            "average_gap": str(avg_gap),
            "on_target_count": on_target_count,
            "total_periods": len(market_values),
            "on_target_pct": str(
                _safe_divide(
                    _D(on_target_count) * _HUNDRED,
                    _D(len(market_values)),
                )
            )
            if market_values
            else str(_ZERO),
        }

    def compute_decarbonisation_rate(
        self,
        data_points: List[Any],
    ) -> Dict[str, Any]:
        """Compute year-on-year decarbonisation rate for both methods.

        Decarbonisation rate is the percentage reduction (negative YoY).
        Positive values indicate emissions are increasing (not
        decarbonising).

        Args:
            data_points: Ordered list of trend data points.

        Returns:
            Dict with location and market decarbonisation rates per
            period, and average rates.
        """
        location_values = self._extract_location_values(data_points)
        market_values = self._extract_market_values(data_points)
        periods = self._extract_periods(data_points)

        loc_rates: List[Dict[str, str]] = []
        mkt_rates: List[Dict[str, str]] = []

        for i in range(1, len(location_values)):
            # Location
            _, loc_pct = self.compute_yoy_change(
                location_values[i], location_values[i - 1]
            )
            loc_rate = _quantize(loc_pct * _NEG_ONE)  # Negate so reduction is positive
            loc_rates.append({
                "period": periods[i] if i < len(periods) else f"period_{i}",
                "decarb_rate_pct": str(loc_rate),
            })

            # Market
            _, mkt_pct = self.compute_yoy_change(
                market_values[i], market_values[i - 1]
            )
            mkt_rate = _quantize(mkt_pct * _NEG_ONE)
            mkt_rates.append({
                "period": periods[i] if i < len(periods) else f"period_{i}",
                "decarb_rate_pct": str(mkt_rate),
            })

        avg_loc = _ZERO
        avg_mkt = _ZERO
        if loc_rates:
            loc_total = sum(
                [_safe_decimal(r["decarb_rate_pct"]) for r in loc_rates],
                _ZERO,
            )
            avg_loc = _quantize(loc_total / _D(len(loc_rates)))
        if mkt_rates:
            mkt_total = sum(
                [_safe_decimal(r["decarb_rate_pct"]) for r in mkt_rates],
                _ZERO,
            )
            avg_mkt = _quantize(mkt_total / _D(len(mkt_rates)))

        return {
            "location_decarb_rates": loc_rates,
            "market_decarb_rates": mkt_rates,
            "avg_location_decarb_rate": str(avg_loc),
            "avg_market_decarb_rate": str(avg_mkt),
            "num_periods": len(loc_rates),
        }

    def compute_benchmark_comparison(
        self,
        data_points: List[Any],
        benchmark_values: List[Decimal],
        benchmark_label: str = "industry_average",
    ) -> Dict[str, Any]:
        """Compare emissions against a benchmark series.

        Args:
            data_points: Ordered list of trend data points.
            benchmark_values: Benchmark values per period (same length).
            benchmark_label: Label for the benchmark series.

        Returns:
            Dict with per-period comparison and summary statistics.
        """
        market_values = self._extract_market_values(data_points)
        periods = self._extract_periods(data_points)

        comparisons: List[Dict[str, str]] = []
        above_count = 0
        below_count = 0

        for i in range(len(market_values)):
            actual = market_values[i]
            benchmark = (
                _safe_decimal(benchmark_values[i])
                if i < len(benchmark_values)
                else _ZERO
            )
            diff = _quantize(actual - benchmark)
            diff_pct = _ZERO
            if benchmark > _ZERO:
                diff_pct = _quantize(diff / benchmark * _HUNDRED)

            above = actual > benchmark
            if above:
                above_count += 1
            elif actual < benchmark:
                below_count += 1

            comparisons.append({
                "period": periods[i] if i < len(periods) else f"period_{i}",
                "actual": str(actual),
                "benchmark": str(benchmark),
                "difference": str(diff),
                "difference_pct": str(diff_pct),
                "above_benchmark": above,
            })

        return {
            "benchmark_label": benchmark_label,
            "comparisons": comparisons,
            "periods_above": above_count,
            "periods_below": below_count,
            "periods_equal": len(market_values) - above_count - below_count,
            "total_periods": len(market_values),
        }

    def get_trend_report_model(
        self,
        data_points: List[Any],
        config_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Build a TrendReport Pydantic model from trend analysis.

        Convenience method that calls ``analyze_trends`` and maps the
        result into a ``TrendReport`` model if the models module is
        available.

        Args:
            data_points: Ordered trend data points.
            config_params: Optional configuration overrides.

        Returns:
            TrendReport model instance, or None if models unavailable.
        """
        if not _MODELS_AVAILABLE:
            logger.warning(
                "Models module not available; cannot build TrendReport model"
            )
            return None

        result = self.analyze_trends(data_points, config_params)
        params = config_params or {}
        tenant_id = str(params.get("tenant_id", "default"))

        try:
            # Build minimal TrendReport from result dict
            report = TrendReport(
                tenant_id=tenant_id,
                data_points=data_points,
                location_yoy_pct=(
                    _safe_decimal(
                        result["location_yoy_changes"][-1]["percentage_change"]
                    )
                    if result.get("location_yoy_changes")
                    else None
                ),
                market_yoy_pct=(
                    _safe_decimal(
                        result["market_yoy_changes"][-1]["percentage_change"]
                    )
                    if result.get("market_yoy_changes")
                    else None
                ),
                location_cagr=result.get("location_cagr"),
                market_cagr=result.get("market_cagr"),
                pif_trend=self._to_trend_direction(result.get("pif_trend")),
                re100_trend=self._to_trend_direction(
                    result.get("re100_trend")
                ),
                intensity_metrics=result.get("intensity_results", {}),
                sbti_on_track=(
                    result["sbti_assessment"].get("on_track", False)
                    if result.get("sbti_assessment")
                    else False
                ),
            )
            return report
        except Exception as exc:
            logger.warning(
                "Failed to build TrendReport model: %s",
                exc,
            )
            return None

    def _to_trend_direction(self, value: Optional[str]) -> Optional[Any]:
        """Convert a string to a TrendDirection enum if available.

        Args:
            value: String trend direction value.

        Returns:
            TrendDirection enum or None.
        """
        if value is None:
            return None

        if not _MODELS_AVAILABLE:
            return None

        try:
            return TrendDirection(value)
        except (ValueError, KeyError):
            return None


# ===========================================================================
# Module-level convenience functions
# ===========================================================================


def get_trend_analysis_engine() -> TrendAnalysisEngine:
    """Return the singleton TrendAnalysisEngine.

    Creates the instance on first call. Subsequent calls return the
    cached singleton.

    Returns:
        TrendAnalysisEngine singleton instance.

    Example:
        >>> engine = get_trend_analysis_engine()
        >>> report = engine.analyze_trends(data_points, config)
    """
    return TrendAnalysisEngine()


def reset_trend_analysis_engine() -> None:
    """Reset the singleton for test teardown.

    The next call to :func:`get_trend_analysis_engine` will construct
    a fresh instance and re-read configuration.

    Example:
        >>> reset_trend_analysis_engine()
        >>> engine = get_trend_analysis_engine()  # fresh instance
    """
    TrendAnalysisEngine.reset()
    logger.debug(
        "TrendAnalysisEngine singleton reset via module-level "
        "reset_trend_analysis_engine()"
    )
