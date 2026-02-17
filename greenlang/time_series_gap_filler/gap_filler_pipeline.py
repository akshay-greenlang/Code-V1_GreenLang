# -*- coding: utf-8 -*-
"""
Gap Filler Pipeline Engine - AGENT-DATA-014 Time Series Gap Filler

Engine 7 of 7.  Orchestrates the full gap filling workflow by
composing the six upstream engines (GapDetector, FrequencyAnalyzer,
InterpolationEngine, SeasonalFillerEngine, TrendExtrapolatorEngine,
CrossSeriesFillerEngine) into a deterministic pipeline.

Pipeline stages:
    1. DETECT   -- identify gap segments via GapDetectorEngine
    2. ANALYZE  -- determine series frequency via FrequencyAnalyzerEngine
    3. SELECT   -- choose the best fill strategy per gap
    4. FILL     -- execute the chosen strategy via the appropriate engine
    5. VALIDATE -- verify fill quality (confidence, continuity, range)
    6. REPORT   -- aggregate results and generate a fill report

Zero-Hallucination: All calculations use deterministic Python arithmetic.
Strategy selection uses rule-based heuristics (gap length, seasonality
strength, trend R-squared, cross-series correlation).  No LLM calls for
numeric computations.  Every operation is traced through SHA-256
provenance chains.

Example:
    >>> from greenlang.time_series_gap_filler.gap_filler_pipeline import (
    ...     GapFillerPipelineEngine,
    ... )
    >>> engine = GapFillerPipelineEngine()
    >>> result = engine.run_pipeline([1.0, 2.0, None, None, 5.0, 6.0])
    >>> assert result["status"] == "completed"
    >>> assert result["gaps_filled"] > 0

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
Status: Production Ready
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from greenlang.time_series_gap_filler.config import get_config
from greenlang.time_series_gap_filler.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graceful imports for sibling engines (metrics + engines)
# ---------------------------------------------------------------------------

try:
    from greenlang.time_series_gap_filler import metrics as _metrics_mod
    _METRICS_AVAILABLE = True
except ImportError:
    _metrics_mod = None  # type: ignore[assignment]
    _METRICS_AVAILABLE = False

try:
    from greenlang.time_series_gap_filler.gap_detector import GapDetectorEngine
    _GAP_DETECTOR_AVAILABLE = True
except ImportError:
    GapDetectorEngine = None  # type: ignore[misc, assignment]
    _GAP_DETECTOR_AVAILABLE = False

try:
    from greenlang.time_series_gap_filler.frequency_analyzer import (
        FrequencyAnalyzerEngine,
    )
    _FREQ_ANALYZER_AVAILABLE = True
except ImportError:
    FrequencyAnalyzerEngine = None  # type: ignore[misc, assignment]
    _FREQ_ANALYZER_AVAILABLE = False

try:
    from greenlang.time_series_gap_filler.interpolation_engine import (
        InterpolationEngine,
    )
    _INTERP_AVAILABLE = True
except ImportError:
    InterpolationEngine = None  # type: ignore[misc, assignment]
    _INTERP_AVAILABLE = False

try:
    from greenlang.time_series_gap_filler.seasonal_filler import (
        SeasonalFillerEngine,
    )
    _SEASONAL_AVAILABLE = True
except ImportError:
    SeasonalFillerEngine = None  # type: ignore[misc, assignment]
    _SEASONAL_AVAILABLE = False

try:
    from greenlang.time_series_gap_filler.trend_extrapolator import (
        TrendExtrapolatorEngine,
    )
    _TREND_AVAILABLE = True
except ImportError:
    TrendExtrapolatorEngine = None  # type: ignore[misc, assignment]
    _TREND_AVAILABLE = False

try:
    from greenlang.time_series_gap_filler.cross_series_filler import (
        CrossSeriesFillerEngine,
        ReferenceSeries,
    )
    _CROSS_AVAILABLE = True
except ImportError:
    CrossSeriesFillerEngine = None  # type: ignore[misc, assignment]
    ReferenceSeries = None  # type: ignore[misc, assignment]
    _CROSS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _is_missing(value: Any) -> bool:
    """Determine whether a value represents a missing data point.

    Treats None, float('nan'), and float('inf') as missing.

    Args:
        value: Value to check.

    Returns:
        True if the value is considered missing.
    """
    if value is None:
        return True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    return False


def _safe_mean(values: List[float]) -> float:
    """Compute arithmetic mean, returning 0.0 for empty lists.

    Args:
        values: List of numeric values.

    Returns:
        Arithmetic mean or 0.0.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Local metric helper stubs (delegate when metrics module is present)
# ---------------------------------------------------------------------------


def _inc_jobs_processed(status: str) -> None:
    """Increment the jobs processed counter.

    Args:
        status: Job status (completed, failed).
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        _metrics_mod.inc_jobs_processed(status)


def _inc_gaps_filled(method: str, count: int = 1) -> None:
    """Increment the gaps-filled counter.

    Args:
        method: Fill method name.
        count: Number of gaps filled.
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        _metrics_mod.inc_gaps_filled(method, count)


def _observe_confidence(confidence: float) -> None:
    """Observe a fill confidence score.

    Args:
        confidence: Confidence value (0.0-1.0).
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        _metrics_mod.observe_confidence(confidence)


def _observe_duration(operation: str, duration: float) -> None:
    """Observe processing duration in seconds.

    Args:
        operation: Operation label.
        duration: Duration in seconds.
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        _metrics_mod.observe_duration(operation, duration)


def _inc_strategies(strategy: str) -> None:
    """Increment the strategy selection counter.

    Args:
        strategy: Strategy name.
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        _metrics_mod.inc_strategies(strategy)


def _inc_errors(error_type: str) -> None:
    """Record a processing error event.

    Args:
        error_type: Error classification.
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        _metrics_mod.inc_errors(error_type)


def _inc_validations(result: str) -> None:
    """Increment the validations counter.

    Args:
        result: Validation result (passed, failed, warning).
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        _metrics_mod.inc_validations(result)


def _set_active_jobs(count: int) -> None:
    """Set the active jobs gauge.

    Args:
        count: Active job count.
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        _metrics_mod.set_active_jobs(count)


# ---------------------------------------------------------------------------
# Pipeline stage enumeration
# ---------------------------------------------------------------------------


_STAGES = ("detect", "analyze", "select", "fill", "validate", "report")


# ---------------------------------------------------------------------------
# Local data models
# ---------------------------------------------------------------------------


@dataclass
class GapSegment:
    """A contiguous gap segment for pipeline processing.

    Attributes:
        start: First missing index.
        end: Last missing index (inclusive).
        length: Number of consecutive missing values.
        position: Position classification (leading, trailing, interior).
        strategy: Selected fill strategy name.
        confidence: Achieved fill confidence for this gap.
        filled: Whether this gap was successfully filled.
    """

    start: int = 0
    end: int = 0
    length: int = 1
    position: str = "interior"
    strategy: str = "auto"
    confidence: float = 0.0
    filled: bool = False


@dataclass
class ValidationResult:
    """Result of fill quality validation.

    Attributes:
        level: Validation level (pass, warn, fail).
        confidence_check: Whether mean confidence meets threshold.
        continuity_check: Whether filled values maintain continuity.
        range_check: Whether filled values stay within expected range.
        messages: Validation messages and warnings.
        mean_confidence: Mean confidence of filled points.
        min_confidence: Minimum confidence of filled points.
    """

    level: str = "pass"
    confidence_check: bool = True
    continuity_check: bool = True
    range_check: bool = True
    messages: List[str] = field(default_factory=list)
    mean_confidence: float = 0.0
    min_confidence: float = 0.0


@dataclass
class PipelineStatistics:
    """Aggregated pipeline execution statistics.

    Attributes:
        total_runs: Total pipeline executions.
        total_gaps_detected: Total gaps detected across all runs.
        total_gaps_filled: Total gaps filled across all runs.
        total_points_filled: Total individual points filled.
        avg_confidence: Weighted average confidence across runs.
        by_strategy: Fill counts per strategy.
        by_status: Run counts per status.
        by_validation: Validation counts per level.
    """

    total_runs: int = 0
    total_gaps_detected: int = 0
    total_gaps_filled: int = 0
    total_points_filled: int = 0
    avg_confidence: float = 0.0
    by_strategy: Dict[str, int] = field(default_factory=dict)
    by_status: Dict[str, int] = field(default_factory=dict)
    by_validation: Dict[str, int] = field(default_factory=dict)


# ============================================================================
# GapFillerPipelineEngine
# ============================================================================


class GapFillerPipelineEngine:
    """Orchestrates the full gap filling pipeline across all engines.

    Composes six upstream engines into a sequential pipeline:
    detect -> analyze -> select -> fill -> validate -> report.

    Strategy selection uses rule-based heuristics based on gap length,
    trend strength (R-squared), seasonal autocorrelation, and
    cross-series correlation.  All decisions are deterministic.

    Attributes:
        _config: TimeSeriesGapFillerConfig singleton.
        _provenance: SHA-256 provenance tracker.
        _gap_detector: Gap detection engine instance.
        _freq_analyzer: Frequency analysis engine instance.
        _interpolation: Interpolation engine instance.
        _seasonal: Seasonal filler engine instance.
        _trend: Trend extrapolation engine instance.
        _cross_series: Cross-series filler engine instance.
        _statistics: Running pipeline statistics.

    Example:
        >>> engine = GapFillerPipelineEngine()
        >>> result = engine.run_pipeline([1.0, 2.0, None, None, 5.0, 6.0])
        >>> assert result["status"] == "completed"
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize GapFillerPipelineEngine with all sub-engines.

        Args:
            config: Optional TimeSeriesGapFillerConfig override.
                Falls back to the singleton from ``get_config()``.
        """
        self._config = config or get_config()
        self._provenance: ProvenanceTracker = get_provenance_tracker()
        self._statistics = PipelineStatistics()

        # Initialize sub-engines (graceful when not available)
        self._gap_detector = (
            GapDetectorEngine(self._config)
            if _GAP_DETECTOR_AVAILABLE else None
        )
        self._freq_analyzer = (
            FrequencyAnalyzerEngine(self._config)
            if _FREQ_ANALYZER_AVAILABLE else None
        )
        self._interpolation = (
            InterpolationEngine(self._config)
            if _INTERP_AVAILABLE else None
        )
        self._seasonal = (
            SeasonalFillerEngine(self._config)
            if _SEASONAL_AVAILABLE else None
        )
        self._trend = (
            TrendExtrapolatorEngine(self._config)
            if _TREND_AVAILABLE else None
        )
        self._cross_series = (
            CrossSeriesFillerEngine(self._config)
            if _CROSS_AVAILABLE else None
        )

        available_engines = sum([
            _GAP_DETECTOR_AVAILABLE,
            _FREQ_ANALYZER_AVAILABLE,
            _INTERP_AVAILABLE,
            _SEASONAL_AVAILABLE,
            _TREND_AVAILABLE,
            _CROSS_AVAILABLE,
        ])
        logger.info(
            "GapFillerPipelineEngine initialized (%d/6 sub-engines available)",
            available_engines,
        )

    # ------------------------------------------------------------------
    # 1. run_pipeline - Full pipeline execution
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        values: List[Optional[float]],
        timestamps: Optional[List[datetime]] = None,
        strategy: str = "auto",
        reference_series: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Execute the full gap filling pipeline on a single series.

        Runs all six pipeline stages in order and returns a comprehensive
        result dictionary with filled values, per-gap details, validation
        outcome, and provenance chain.

        Args:
            values: Time series values (None or NaN for missing).
            timestamps: Optional aligned timestamps.
            strategy: Strategy selection mode.  ``'auto'`` uses
                rule-based heuristics; other values force a specific
                method (``'linear'``, ``'cubic_spline'``, ``'seasonal'``,
                ``'trend'``, ``'moving_average'``, ``'cross_series'``).
            reference_series: Optional list of reference series dicts
                for cross-series filling.  Each dict must have
                ``'series_id'`` and ``'values'`` keys.

        Returns:
            Dict with keys:
                - pipeline_id (str): Unique pipeline run identifier.
                - status (str): 'completed' or 'failed'.
                - filled_values (list): Series with gaps filled.
                - original_values (list): Original input series.
                - gaps_detected (int): Number of gaps found.
                - gaps_filled (int): Number of gaps filled.
                - points_filled (int): Individual points filled.
                - fill_details (list): Per-gap fill information.
                - validation (dict): Validation result.
                - report (dict): Summary report.
                - stage_timings (dict): Duration per stage in ms.
                - total_time_ms (float): Total pipeline time.
                - provenance_hash (str): SHA-256 chain hash.
                - error (str or None): Error message if failed.
        """
        pipeline_id = str(uuid4())
        pipeline_start = time.time()
        stage_timings: Dict[str, float] = {}

        logger.info(
            "Pipeline %s starting: n=%d, strategy=%s",
            pipeline_id[:8], len(values), strategy,
        )

        _set_active_jobs(1)

        result: Dict[str, Any] = {
            "pipeline_id": pipeline_id,
            "status": "running",
            "filled_values": list(values),
            "original_values": list(values),
            "gaps_detected": 0,
            "gaps_filled": 0,
            "points_filled": 0,
            "fill_details": [],
            "validation": {},
            "report": {},
            "stage_timings": stage_timings,
            "total_time_ms": 0.0,
            "provenance_hash": "",
            "error": None,
        }

        try:
            # -- Stage 1: DETECT --
            t0 = time.time()
            gap_segments = self._stage_detect(values, timestamps)
            stage_timings["detect"] = (time.time() - t0) * 1000.0

            result["gaps_detected"] = len(gap_segments)
            total_missing = sum(g.length for g in gap_segments)

            if not gap_segments:
                logger.info("Pipeline %s: no gaps detected", pipeline_id[:8])
                result["status"] = "completed"
                result["validation"] = {
                    "level": "pass",
                    "messages": ["No gaps to fill"],
                }
                result["report"] = self._generate_report(
                    values, values, gap_segments, [], 1.0,
                )
                self._finalize_pipeline(
                    result, pipeline_start, stage_timings, pipeline_id,
                )
                return result

            # Validate gap ratio
            n = len(values)
            gap_ratio = total_missing / n if n > 0 else 1.0
            if gap_ratio > self._config.max_gap_ratio:
                logger.warning(
                    "Pipeline %s: gap ratio %.2f exceeds max %.2f",
                    pipeline_id[:8], gap_ratio, self._config.max_gap_ratio,
                )
                result["status"] = "failed"
                result["error"] = (
                    f"Gap ratio {gap_ratio:.2f} exceeds maximum "
                    f"{self._config.max_gap_ratio:.2f}"
                )
                _inc_errors("gap_ratio_exceeded")
                self._finalize_pipeline(
                    result, pipeline_start, stage_timings, pipeline_id,
                )
                return result

            # Check minimum data points
            valid_count = n - total_missing
            if valid_count < self._config.min_data_points:
                logger.warning(
                    "Pipeline %s: insufficient data (%d < %d)",
                    pipeline_id[:8], valid_count, self._config.min_data_points,
                )
                result["status"] = "failed"
                result["error"] = (
                    f"Insufficient data: {valid_count} valid points, "
                    f"need {self._config.min_data_points}"
                )
                _inc_errors("insufficient_data")
                self._finalize_pipeline(
                    result, pipeline_start, stage_timings, pipeline_id,
                )
                return result

            # -- Stage 2: ANALYZE --
            t0 = time.time()
            analysis = self._stage_analyze(values, timestamps)
            stage_timings["analyze"] = (time.time() - t0) * 1000.0

            # -- Stage 3: SELECT --
            t0 = time.time()
            gap_segments = self._stage_select(
                gap_segments, analysis, strategy, values,
            )
            stage_timings["select"] = (time.time() - t0) * 1000.0

            # Register reference series if provided
            if reference_series and self._cross_series is not None:
                for ref_dict in reference_series:
                    sid = ref_dict.get("series_id", str(uuid4()))
                    ref_vals = ref_dict.get("values", [])
                    if ref_vals:
                        self._cross_series.register_reference_series(
                            sid, ref_vals,
                        )

            # -- Stage 4: FILL --
            t0 = time.time()
            filled_values, fill_details = self._stage_fill(
                values, gap_segments, timestamps,
            )
            stage_timings["fill"] = (time.time() - t0) * 1000.0

            result["filled_values"] = filled_values
            result["fill_details"] = fill_details
            result["gaps_filled"] = sum(
                1 for g in gap_segments if g.filled
            )
            result["points_filled"] = sum(
                g.length for g in gap_segments if g.filled
            )

            # -- Stage 5: VALIDATE --
            t0 = time.time()
            validation = self._stage_validate(
                values, filled_values, gap_segments,
            )
            stage_timings["validate"] = (time.time() - t0) * 1000.0
            result["validation"] = {
                "level": validation.level,
                "confidence_check": validation.confidence_check,
                "continuity_check": validation.continuity_check,
                "range_check": validation.range_check,
                "messages": validation.messages,
                "mean_confidence": validation.mean_confidence,
                "min_confidence": validation.min_confidence,
            }

            # -- Stage 6: REPORT --
            t0 = time.time()
            report = self._generate_report(
                values, filled_values, gap_segments, fill_details,
                validation.mean_confidence,
            )
            stage_timings["report"] = (time.time() - t0) * 1000.0
            result["report"] = report

            result["status"] = "completed"

        except Exception as exc:
            logger.error(
                "Pipeline %s failed: %s", pipeline_id[:8], str(exc),
                exc_info=True,
            )
            result["status"] = "failed"
            result["error"] = str(exc)
            _inc_errors("pipeline")

        self._finalize_pipeline(
            result, pipeline_start, stage_timings, pipeline_id,
        )
        return result

    # ------------------------------------------------------------------
    # 2. run_batch_pipeline - Batch processing
    # ------------------------------------------------------------------

    def run_batch_pipeline(
        self,
        series_list: List[Dict[str, Any]],
        strategy: str = "auto",
    ) -> Dict[str, Any]:
        """Execute the pipeline on multiple series in batch.

        Processes each series independently and aggregates results.

        Args:
            series_list: List of dicts with keys:
                - ``'series_id'`` (str): Unique series identifier.
                - ``'values'`` (list): Series values.
                - ``'timestamps'`` (list, optional): Timestamps.
                - ``'reference_series'`` (list, optional): Reference dicts.
            strategy: Strategy selection mode applied to all series.

        Returns:
            Dict with keys:
                - batch_id (str): Unique batch identifier.
                - status (str): 'completed' or 'partial'.
                - total_series (int): Number of series processed.
                - successful (int): Number of successful pipelines.
                - failed (int): Number of failed pipelines.
                - results (list): Per-series pipeline results.
                - total_gaps_filled (int): Aggregate gaps filled.
                - total_points_filled (int): Aggregate points filled.
                - avg_confidence (float): Weighted average confidence.
                - total_time_ms (float): Total batch processing time.
                - provenance_hash (str): SHA-256 chain hash.
        """
        batch_id = str(uuid4())
        batch_start = time.time()

        logger.info(
            "Batch %s starting: %d series, strategy=%s",
            batch_id[:8], len(series_list), strategy,
        )

        results: List[Dict[str, Any]] = []
        total_gaps_filled = 0
        total_points_filled = 0
        all_confidences: List[float] = []
        successful = 0
        failed = 0

        batch_size = self._config.batch_size
        for idx, series_dict in enumerate(series_list):
            if idx >= self._config.max_records:
                logger.warning(
                    "Batch %s: max_records limit (%d) reached at index %d",
                    batch_id[:8], self._config.max_records, idx,
                )
                break

            series_id = series_dict.get("series_id", f"series_{idx}")
            series_values = series_dict.get("values", [])
            series_timestamps = series_dict.get("timestamps")
            series_refs = series_dict.get("reference_series")

            if not series_values:
                logger.warning(
                    "Batch %s: skipping empty series '%s'",
                    batch_id[:8], series_id,
                )
                continue

            try:
                pipeline_result = self.run_pipeline(
                    values=series_values,
                    timestamps=series_timestamps,
                    strategy=strategy,
                    reference_series=series_refs,
                )
                pipeline_result["series_id"] = series_id
                results.append(pipeline_result)

                if pipeline_result["status"] == "completed":
                    successful += 1
                    total_gaps_filled += pipeline_result.get("gaps_filled", 0)
                    total_points_filled += pipeline_result.get("points_filled", 0)

                    validation = pipeline_result.get("validation", {})
                    mean_conf = validation.get("mean_confidence", 0.0)
                    if isinstance(mean_conf, (int, float)) and mean_conf > 0:
                        all_confidences.append(float(mean_conf))
                else:
                    failed += 1

            except Exception as exc:
                logger.error(
                    "Batch %s: series '%s' raised %s: %s",
                    batch_id[:8], series_id, type(exc).__name__, str(exc),
                )
                results.append({
                    "series_id": series_id,
                    "status": "failed",
                    "error": str(exc),
                })
                failed += 1

        avg_confidence = _safe_mean(all_confidences)
        batch_status = "completed" if failed == 0 else "partial"

        # Provenance
        provenance_hash = self._provenance.build_hash({
            "operation": "run_batch_pipeline",
            "batch_id": batch_id,
            "total_series": len(results),
            "successful": successful,
            "failed": failed,
            "total_gaps_filled": total_gaps_filled,
        })
        self._provenance.record(
            "pipeline_batch", batch_id, "batch", provenance_hash,
        )

        elapsed = time.time() - batch_start
        _observe_duration("batch_pipeline", elapsed)

        logger.info(
            "Batch %s completed: %d/%d successful, %d gaps filled, "
            "avg_conf=%.3f, %.1fms",
            batch_id[:8], successful, len(results), total_gaps_filled,
            avg_confidence, elapsed * 1000.0,
        )

        return {
            "batch_id": batch_id,
            "status": batch_status,
            "total_series": len(results),
            "successful": successful,
            "failed": failed,
            "results": results,
            "total_gaps_filled": total_gaps_filled,
            "total_points_filled": total_points_filled,
            "avg_confidence": avg_confidence,
            "total_time_ms": elapsed * 1000.0,
            "provenance_hash": provenance_hash,
        }

    # ------------------------------------------------------------------
    # 3. get_statistics - Pipeline statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregated pipeline execution statistics.

        Returns:
            Dict with total_runs, total_gaps_detected, total_gaps_filled,
            total_points_filled, avg_confidence, by_strategy, by_status,
            by_validation, and engine_availability.
        """
        return {
            "total_runs": self._statistics.total_runs,
            "total_gaps_detected": self._statistics.total_gaps_detected,
            "total_gaps_filled": self._statistics.total_gaps_filled,
            "total_points_filled": self._statistics.total_points_filled,
            "avg_confidence": self._statistics.avg_confidence,
            "by_strategy": dict(self._statistics.by_strategy),
            "by_status": dict(self._statistics.by_status),
            "by_validation": dict(self._statistics.by_validation),
            "engine_availability": {
                "gap_detector": _GAP_DETECTOR_AVAILABLE,
                "frequency_analyzer": _FREQ_ANALYZER_AVAILABLE,
                "interpolation": _INTERP_AVAILABLE,
                "seasonal": _SEASONAL_AVAILABLE,
                "trend": _TREND_AVAILABLE,
                "cross_series": _CROSS_AVAILABLE,
            },
            "provenance_chain_length": self._provenance.get_chain_length(),
        }

    # ------------------------------------------------------------------
    # 4. get_engine_health - Health check
    # ------------------------------------------------------------------

    def get_engine_health(self) -> Dict[str, Any]:
        """Return the health status of the pipeline and sub-engines.

        Returns:
            Dict with overall status, per-engine availability, and
            configuration summary.
        """
        engines = {
            "gap_detector": _GAP_DETECTOR_AVAILABLE,
            "frequency_analyzer": _FREQ_ANALYZER_AVAILABLE,
            "interpolation": _INTERP_AVAILABLE,
            "seasonal": _SEASONAL_AVAILABLE,
            "trend": _TREND_AVAILABLE,
            "cross_series": _CROSS_AVAILABLE,
        }
        available = sum(engines.values())
        total = len(engines)

        if available == total:
            overall = "healthy"
        elif available >= 3:
            overall = "degraded"
        else:
            overall = "unhealthy"

        return {
            "status": overall,
            "engines_available": available,
            "engines_total": total,
            "engines": engines,
            "config": {
                "default_strategy": self._config.default_strategy,
                "max_gap_ratio": self._config.max_gap_ratio,
                "min_data_points": self._config.min_data_points,
                "confidence_threshold": self._config.confidence_threshold,
                "correlation_threshold": self._config.correlation_threshold,
                "seasonal_periods": self._config.seasonal_periods,
                "enable_seasonal": self._config.enable_seasonal,
                "enable_cross_series": self._config.enable_cross_series,
                "enable_provenance": self._config.enable_provenance,
            },
            "statistics": {
                "total_runs": self._statistics.total_runs,
                "total_gaps_filled": self._statistics.total_gaps_filled,
            },
        }

    # ==================================================================
    # Private: Stage 1 - DETECT
    # ==================================================================

    def _stage_detect(
        self,
        values: List[Optional[float]],
        timestamps: Optional[List[datetime]] = None,
    ) -> List[GapSegment]:
        """Detect gap segments in the series.

        Uses GapDetectorEngine if available, otherwise falls back to
        a simple scan for consecutive missing values.

        Args:
            values: Time series values.
            timestamps: Optional timestamps.

        Returns:
            List of GapSegment dataclasses.
        """
        n = len(values)
        segments: List[GapSegment] = []

        if self._gap_detector is not None:
            try:
                detection_result = self._gap_detector.detect_gaps(
                    values, timestamps=timestamps,
                )
                for gap in detection_result.gaps:
                    position = self._classify_position(
                        gap.start_index, gap.end_index, n,
                    )
                    segments.append(GapSegment(
                        start=gap.start_index,
                        end=gap.end_index,
                        length=gap.length,
                        position=position,
                    ))
                logger.debug(
                    "Stage DETECT (via GapDetectorEngine): %d gaps found",
                    len(segments),
                )
                return segments
            except Exception as exc:
                logger.warning(
                    "GapDetectorEngine failed, falling back: %s", str(exc),
                )

        # Fallback: simple scan
        segments = self._simple_gap_scan(values)
        logger.debug(
            "Stage DETECT (fallback): %d gaps found", len(segments),
        )
        return segments

    # ==================================================================
    # Private: Stage 2 - ANALYZE
    # ==================================================================

    def _stage_analyze(
        self,
        values: List[Optional[float]],
        timestamps: Optional[List[datetime]] = None,
    ) -> Dict[str, Any]:
        """Analyze the series for frequency, trend, and seasonality.

        Collects analysis signals used by the strategy selector.

        Args:
            values: Time series values.
            timestamps: Optional timestamps.

        Returns:
            Dict with analysis results including frequency_level,
            regularity_score, trend_type, r_squared, is_seasonal,
            dominant_period, and seasonal_strength.
        """
        analysis: Dict[str, Any] = {
            "frequency_level": "unknown",
            "regularity_score": 0.0,
            "trend_type": "unknown",
            "r_squared": 0.0,
            "is_seasonal": False,
            "dominant_period": 0,
            "seasonal_strength": 0.0,
        }

        # Frequency analysis
        if self._freq_analyzer is not None and timestamps is not None:
            try:
                freq_result = self._freq_analyzer.analyze_frequency(
                    timestamps,
                )
                analysis["frequency_level"] = freq_result.get(
                    "frequency_level", "unknown",
                )
                analysis["regularity_score"] = freq_result.get(
                    "regularity_score", 0.0,
                )
            except Exception as exc:
                logger.warning(
                    "FrequencyAnalyzerEngine failed: %s", str(exc),
                )

        # Trend analysis
        if self._trend is not None:
            try:
                trend_result = self._trend.analyze_trend(values)
                # analyze_trend may return a TrendAnalysis model or a dict
                if hasattr(trend_result, "trend_type"):
                    analysis["trend_type"] = (
                        trend_result.trend_type.value
                        if hasattr(trend_result.trend_type, "value")
                        else str(trend_result.trend_type)
                    )
                    analysis["r_squared"] = getattr(
                        trend_result, "r_squared", 0.0,
                    )
                elif isinstance(trend_result, dict):
                    analysis["trend_type"] = trend_result.get(
                        "trend_type", "unknown",
                    )
                    analysis["r_squared"] = trend_result.get(
                        "r_squared", 0.0,
                    )
            except Exception as exc:
                logger.warning(
                    "TrendExtrapolatorEngine.analyze_trend failed: %s",
                    str(exc),
                )

        # Seasonal analysis
        if self._seasonal is not None and self._config.enable_seasonal:
            try:
                seasonal_result = self._seasonal.detect_seasonality(values)
                analysis["is_seasonal"] = seasonal_result.get(
                    "is_seasonal", False,
                )
                analysis["dominant_period"] = seasonal_result.get(
                    "dominant_period", 0,
                )
                analysis["seasonal_strength"] = seasonal_result.get(
                    "confidence", 0.0,
                )
            except Exception as exc:
                logger.warning(
                    "SeasonalFillerEngine.detect_seasonality failed: %s",
                    str(exc),
                )

        logger.debug(
            "Stage ANALYZE: freq=%s, trend=%s(R2=%.3f), seasonal=%s(p=%d)",
            analysis["frequency_level"],
            analysis["trend_type"],
            analysis["r_squared"],
            analysis["is_seasonal"],
            analysis["dominant_period"],
        )

        return analysis

    # ==================================================================
    # Private: Stage 3 - SELECT
    # ==================================================================

    def _stage_select(
        self,
        gap_segments: List[GapSegment],
        analysis: Dict[str, Any],
        strategy: str,
        values: List[Optional[float]],
    ) -> List[GapSegment]:
        """Select a fill strategy for each gap segment.

        When strategy is ``'auto'``, uses rule-based heuristics:
            1. Short interior gaps (length <= short_gap_limit):
               -> linear interpolation
            2. Strongly seasonal series (seasonal_strength > 0.5):
               -> seasonal decomposition
            3. Strong trend (R-squared > 0.7):
               -> linear trend extrapolation
            4. Leading/trailing gaps:
               -> moving average
            5. Default:
               -> linear interpolation

        When strategy is not ``'auto'``, all gaps receive that strategy.

        Args:
            gap_segments: Gap segments from the DETECT stage.
            analysis: Analysis signals from the ANALYZE stage.
            strategy: Strategy mode.
            values: Original series for context.

        Returns:
            Updated gap segments with strategy assignments.
        """
        short_limit = self._config.short_gap_limit
        long_limit = self._config.long_gap_limit
        r_squared = analysis.get("r_squared", 0.0)
        is_seasonal = analysis.get("is_seasonal", False)
        seasonal_strength = analysis.get("seasonal_strength", 0.0)

        for gap in gap_segments:
            if strategy != "auto":
                gap.strategy = strategy
                _inc_strategies(strategy)
                continue

            selected = self._select_strategy(
                gap, short_limit, long_limit, r_squared,
                is_seasonal, seasonal_strength,
            )
            gap.strategy = selected
            _inc_strategies(selected)

        # Log strategy distribution
        strategy_counts: Dict[str, int] = {}
        for g in gap_segments:
            strategy_counts[g.strategy] = strategy_counts.get(g.strategy, 0) + 1

        logger.debug(
            "Stage SELECT: %d gaps, strategies=%s",
            len(gap_segments), strategy_counts,
        )

        return gap_segments

    def _select_strategy(
        self,
        gap: GapSegment,
        short_limit: int,
        long_limit: int,
        r_squared: float,
        is_seasonal: bool,
        seasonal_strength: float,
    ) -> str:
        """Select the best fill strategy for a single gap segment.

        Args:
            gap: The gap segment.
            short_limit: Maximum length for short gaps.
            long_limit: Maximum length for long gaps.
            r_squared: Trend R-squared from analysis.
            is_seasonal: Whether seasonal patterns were detected.
            seasonal_strength: Strength of the seasonal signal.

        Returns:
            Strategy name string.
        """
        # Leading gaps: use moving average or trend
        if gap.position == "leading":
            if r_squared >= 0.7 and _TREND_AVAILABLE:
                return "linear_trend"
            return "moving_average"

        # Trailing gaps: use trend or moving average
        if gap.position == "trailing":
            if r_squared >= 0.7 and _TREND_AVAILABLE:
                return "linear_trend"
            return "moving_average"

        # Interior gaps
        if gap.length <= short_limit:
            # Short interior gaps: prefer linear interpolation
            if _INTERP_AVAILABLE:
                return "linear"
            return "moving_average"

        if gap.length <= long_limit:
            # Medium gaps: seasonal if strong, trend if moderate, else cubic
            if is_seasonal and seasonal_strength > 0.5 and _SEASONAL_AVAILABLE:
                return "seasonal"
            if r_squared >= 0.6 and _TREND_AVAILABLE:
                return "linear_trend"
            if _INTERP_AVAILABLE:
                return "cubic_spline"
            return "moving_average"

        # Long gaps: seasonal if available, else trend, else MA
        if is_seasonal and seasonal_strength > 0.4 and _SEASONAL_AVAILABLE:
            return "seasonal"
        if r_squared >= 0.5 and _TREND_AVAILABLE:
            return "linear_trend"
        if _TREND_AVAILABLE:
            return "exponential_smoothing"
        return "moving_average"

    # ==================================================================
    # Private: Stage 4 - FILL
    # ==================================================================

    def _stage_fill(
        self,
        values: List[Optional[float]],
        gap_segments: List[GapSegment],
        timestamps: Optional[List[datetime]] = None,
    ) -> Tuple[List[Optional[float]], List[Dict[str, Any]]]:
        """Execute fills for each gap segment using the selected strategy.

        Applies the fill to the series in-place and records per-gap
        fill details.

        Args:
            values: Original series (will not be mutated).
            gap_segments: Gap segments with strategies assigned.
            timestamps: Optional timestamps.

        Returns:
            Tuple of (filled_values, fill_details).
        """
        filled = list(values)
        fill_details: List[Dict[str, Any]] = []

        for gap in gap_segments:
            t0 = time.time()
            try:
                gap_filled_values, confidence = self._fill_gap(
                    filled, gap, timestamps,
                )

                # Apply filled values
                for idx in range(gap.start, gap.end + 1):
                    if idx < len(gap_filled_values):
                        filled[idx] = gap_filled_values[idx]

                gap.confidence = confidence
                gap.filled = True

                fill_details.append({
                    "gap_start": gap.start,
                    "gap_end": gap.end,
                    "gap_length": gap.length,
                    "position": gap.position,
                    "strategy": gap.strategy,
                    "confidence": confidence,
                    "filled": True,
                    "processing_time_ms": (time.time() - t0) * 1000.0,
                })

                _inc_gaps_filled(gap.strategy)

            except Exception as exc:
                logger.warning(
                    "Fill failed for gap [%d:%d] with strategy '%s': %s",
                    gap.start, gap.end, gap.strategy, str(exc),
                )
                gap.filled = False
                gap.confidence = 0.0
                _inc_errors("fill")

                fill_details.append({
                    "gap_start": gap.start,
                    "gap_end": gap.end,
                    "gap_length": gap.length,
                    "position": gap.position,
                    "strategy": gap.strategy,
                    "confidence": 0.0,
                    "filled": False,
                    "error": str(exc),
                    "processing_time_ms": (time.time() - t0) * 1000.0,
                })

        return filled, fill_details

    def _fill_gap(
        self,
        current_values: List[Optional[float]],
        gap: GapSegment,
        timestamps: Optional[List[datetime]] = None,
    ) -> Tuple[List[Optional[float]], float]:
        """Fill a single gap using the assigned strategy.

        Routes to the appropriate engine based on gap.strategy and
        returns the updated series with the gap filled.

        Args:
            current_values: Current state of the series (may have
                some gaps already filled from earlier segments).
            gap: Gap segment with strategy assigned.
            timestamps: Optional timestamps.

        Returns:
            Tuple of (series_with_gap_filled, confidence).
        """
        strategy = gap.strategy
        series = list(current_values)

        # Route to the appropriate engine
        if strategy in ("linear", "cubic_spline", "pchip", "akima",
                        "polynomial", "nearest"):
            return self._fill_with_interpolation(series, strategy)

        if strategy == "seasonal":
            return self._fill_with_seasonal(series)

        if strategy in ("linear_trend", "exponential_smoothing",
                        "double_exponential", "holt_winters"):
            return self._fill_with_trend(series, strategy)

        if strategy == "moving_average":
            return self._fill_with_moving_average(series)

        if strategy == "cross_series":
            return self._fill_with_cross_series(series)

        # Default fallback: linear interpolation
        logger.warning(
            "Unknown strategy '%s', falling back to linear", strategy,
        )
        return self._fill_with_interpolation(series, "linear")

    def _fill_with_interpolation(
        self,
        values: List[Optional[float]],
        method: str,
    ) -> Tuple[List[Optional[float]], float]:
        """Fill using InterpolationEngine.

        Args:
            values: Series with gaps.
            method: Interpolation method name.

        Returns:
            Tuple of (filled_series, confidence).
        """
        if self._interpolation is None:
            raise RuntimeError("InterpolationEngine not available")

        result = self._interpolation.fill_gaps(values, method=method)

        # Extract filled values from the result
        if hasattr(result, "filled_values") and result.filled_values:
            return list(result.filled_values), getattr(
                result, "mean_confidence", 0.8,
            )

        # If fill_gaps returns a local FillResult with values attribute
        filled_vals = getattr(result, "values", None)
        if filled_vals:
            return list(filled_vals), getattr(result, "confidence", 0.8)

        return list(values), 0.0

    def _fill_with_seasonal(
        self,
        values: List[Optional[float]],
    ) -> Tuple[List[Optional[float]], float]:
        """Fill using SeasonalFillerEngine.

        Args:
            values: Series with gaps.

        Returns:
            Tuple of (filled_series, confidence).
        """
        if self._seasonal is None:
            raise RuntimeError("SeasonalFillerEngine not available")

        result = self._seasonal.fill_seasonal(values)
        filled_vals = getattr(result, "values", list(values))
        confidence = getattr(result, "confidence", 0.7)

        return list(filled_vals), float(confidence)

    def _fill_with_trend(
        self,
        values: List[Optional[float]],
        strategy: str,
    ) -> Tuple[List[Optional[float]], float]:
        """Fill using TrendExtrapolatorEngine.

        Args:
            values: Series with gaps.
            strategy: Specific trend method name.

        Returns:
            Tuple of (filled_series, confidence).
        """
        if self._trend is None:
            raise RuntimeError("TrendExtrapolatorEngine not available")

        # Route to appropriate method
        method_map = {
            "linear_trend": "fill_linear_trend",
            "exponential_smoothing": "fill_exponential_smoothing",
            "double_exponential": "fill_double_exponential",
            "holt_winters": "fill_holt_winters",
        }

        method_name = method_map.get(strategy, "fill_linear_trend")
        fill_fn = getattr(self._trend, method_name, None)

        if fill_fn is None:
            raise RuntimeError(
                f"TrendExtrapolatorEngine has no method '{method_name}'"
            )

        result = fill_fn(values)

        # Extract from dict (new format) or FillResult object (legacy)
        if isinstance(result, dict):
            filled = result.get("filled_values", list(values))
            conf = result.get("mean_confidence", 0.75)
        elif hasattr(result, "filled_values") and result.filled_values:
            filled = list(result.filled_values)
            conf = getattr(result, "mean_confidence", 0.75)
        else:
            filled = getattr(result, "values", list(values))
            if isinstance(filled, list):
                filled = list(filled)
            else:
                filled = list(values)
            conf = getattr(result, "confidence", 0.75)

        return list(filled), float(conf)

    def _fill_with_moving_average(
        self,
        values: List[Optional[float]],
    ) -> Tuple[List[Optional[float]], float]:
        """Fill using TrendExtrapolatorEngine.fill_moving_average.

        Falls back to a simple local mean if the trend engine is
        not available.

        Args:
            values: Series with gaps.

        Returns:
            Tuple of (filled_series, confidence).
        """
        if self._trend is not None:
            try:
                result = self._trend.fill_moving_average(values)
                if isinstance(result, dict):
                    return (
                        list(result.get("filled_values", values)),
                        result.get("mean_confidence", 0.6),
                    )
                if hasattr(result, "filled_values") and result.filled_values:
                    return list(result.filled_values), getattr(
                        result, "mean_confidence", 0.6,
                    )
                filled = getattr(result, "values", list(values))
                return list(filled), getattr(result, "confidence", 0.6)
            except Exception as exc:
                logger.warning(
                    "fill_moving_average via trend engine failed: %s",
                    str(exc),
                )

        # Fallback: simple local window mean
        return self._simple_moving_average_fill(values)

    def _fill_with_cross_series(
        self,
        values: List[Optional[float]],
    ) -> Tuple[List[Optional[float]], float]:
        """Fill using CrossSeriesFillerEngine.

        Uses auto_fill_consensus which leverages all registered
        reference series.

        Args:
            values: Series with gaps.

        Returns:
            Tuple of (filled_series, confidence).
        """
        if self._cross_series is None:
            raise RuntimeError("CrossSeriesFillerEngine not available")

        result = self._cross_series.auto_fill_consensus(values)
        filled = getattr(result, "values", list(values))
        confidence = getattr(result, "avg_confidence", 0.6)

        return list(filled), float(confidence)

    # ==================================================================
    # Private: Stage 5 - VALIDATE
    # ==================================================================

    def _stage_validate(
        self,
        original: List[Optional[float]],
        filled: List[Optional[float]],
        gap_segments: List[GapSegment],
    ) -> ValidationResult:
        """Validate fill quality against configured thresholds.

        Checks:
            1. Confidence threshold: mean confidence >= config threshold.
            2. Continuity: no abrupt jumps at gap boundaries.
            3. Range: filled values within data range or reasonable
               extrapolation bounds.

        Args:
            original: Original series with gaps.
            filled: Series after gap filling.
            gap_segments: Gap segments with fill results.

        Returns:
            ValidationResult with pass/warn/fail level.
        """
        result = ValidationResult()
        messages: List[str] = []

        filled_gaps = [g for g in gap_segments if g.filled]
        unfilled_gaps = [g for g in gap_segments if not g.filled]

        # 1. Confidence check
        if filled_gaps:
            confidences = [g.confidence for g in filled_gaps]
            result.mean_confidence = _safe_mean(confidences)
            result.min_confidence = min(confidences) if confidences else 0.0

            threshold = self._config.confidence_threshold
            if result.mean_confidence < threshold:
                result.confidence_check = False
                messages.append(
                    f"Mean confidence {result.mean_confidence:.3f} "
                    f"below threshold {threshold:.3f}"
                )
        else:
            result.mean_confidence = 0.0
            result.min_confidence = 0.0
            if gap_segments:
                messages.append("No gaps were successfully filled")

        # 2. Continuity check
        continuity_issues = self._check_continuity(original, filled, gap_segments)
        if continuity_issues:
            result.continuity_check = False
            for issue in continuity_issues:
                messages.append(issue)

        # 3. Range check
        range_issues = self._check_range(original, filled)
        if range_issues:
            result.range_check = False
            for issue in range_issues:
                messages.append(issue)

        # Determine level
        if not result.confidence_check or not result.range_check:
            result.level = "fail"
        elif not result.continuity_check or unfilled_gaps:
            result.level = "warn"
        else:
            result.level = "pass"

        if unfilled_gaps:
            messages.append(
                f"{len(unfilled_gaps)} of {len(gap_segments)} gaps "
                f"could not be filled"
            )

        result.messages = messages

        # Record metric
        _inc_validations(result.level)

        logger.info(
            "Stage VALIDATE: level=%s, mean_conf=%.3f, messages=%d",
            result.level, result.mean_confidence, len(messages),
        )

        return result

    def _check_continuity(
        self,
        original: List[Optional[float]],
        filled: List[Optional[float]],
        gap_segments: List[GapSegment],
    ) -> List[str]:
        """Check for abrupt discontinuities at gap boundaries.

        A discontinuity is flagged when the absolute difference between
        a filled value at a gap boundary and the nearest original value
        exceeds 3 times the standard deviation of the original data.

        Args:
            original: Original series.
            filled: Filled series.
            gap_segments: Gap segments.

        Returns:
            List of warning message strings (empty if no issues).
        """
        issues: List[str] = []

        # Compute std of original non-missing values
        valid_vals = [
            float(v) for v in original  # type: ignore[arg-type]
            if not _is_missing(v)
        ]
        if len(valid_vals) < 2:
            return issues

        mean_val = _safe_mean(valid_vals)
        variance = sum((v - mean_val) ** 2 for v in valid_vals) / len(valid_vals)
        std_val = math.sqrt(variance) if variance > 0 else 0.0

        if std_val < 1e-12:
            return issues

        threshold = 3.0 * std_val

        for gap in gap_segments:
            if not gap.filled:
                continue

            # Check left boundary
            if gap.start > 0 and not _is_missing(original[gap.start - 1]):
                left_orig = float(original[gap.start - 1])  # type: ignore[arg-type]
                fill_val = filled[gap.start]
                if fill_val is not None:
                    diff = abs(float(fill_val) - left_orig)
                    if diff > threshold:
                        issues.append(
                            f"Discontinuity at left boundary of gap "
                            f"[{gap.start}:{gap.end}]: "
                            f"jump={diff:.2f} > {threshold:.2f}"
                        )

            # Check right boundary
            if (gap.end + 1 < len(original)
                    and not _is_missing(original[gap.end + 1])):
                right_orig = float(original[gap.end + 1])  # type: ignore[arg-type]
                fill_val = filled[gap.end]
                if fill_val is not None:
                    diff = abs(float(fill_val) - right_orig)
                    if diff > threshold:
                        issues.append(
                            f"Discontinuity at right boundary of gap "
                            f"[{gap.start}:{gap.end}]: "
                            f"jump={diff:.2f} > {threshold:.2f}"
                        )

        return issues

    def _check_range(
        self,
        original: List[Optional[float]],
        filled: List[Optional[float]],
    ) -> List[str]:
        """Check whether filled values are within a reasonable range.

        Flags filled values that exceed the original data range
        extended by 50% on each side.

        Args:
            original: Original series.
            filled: Filled series.

        Returns:
            List of warning message strings (empty if no issues).
        """
        issues: List[str] = []

        valid_vals = [
            float(v) for v in original  # type: ignore[arg-type]
            if not _is_missing(v)
        ]
        if not valid_vals:
            return issues

        data_min = min(valid_vals)
        data_max = max(valid_vals)
        data_range = data_max - data_min
        margin = max(data_range * 0.5, abs(data_max) * 0.1, 1.0)

        lower_bound = data_min - margin
        upper_bound = data_max + margin

        out_of_range_count = 0
        for i, (orig, fill) in enumerate(zip(original, filled)):
            if not _is_missing(orig):
                continue  # Only check filled positions
            if fill is None:
                continue
            fv = float(fill)
            if fv < lower_bound or fv > upper_bound:
                out_of_range_count += 1

        if out_of_range_count > 0:
            issues.append(
                f"{out_of_range_count} filled values outside expected "
                f"range [{lower_bound:.2f}, {upper_bound:.2f}]"
            )

        return issues

    # ==================================================================
    # Private: Stage 6 - REPORT
    # ==================================================================

    def _generate_report(
        self,
        original: List[Optional[float]],
        filled: List[Optional[float]],
        gap_segments: List[GapSegment],
        fill_details: List[Dict[str, Any]],
        mean_confidence: float,
    ) -> Dict[str, Any]:
        """Generate a comprehensive fill report.

        Args:
            original: Original series.
            filled: Filled series.
            gap_segments: Gap segments with fill results.
            fill_details: Per-gap fill detail dicts.
            mean_confidence: Overall mean confidence.

        Returns:
            Dict with summary statistics and impact analysis.
        """
        n = len(original)
        total_gaps = len(gap_segments)
        total_filled = sum(1 for g in gap_segments if g.filled)
        total_unfilled = total_gaps - total_filled
        total_missing = sum(g.length for g in gap_segments)
        total_points_filled = sum(
            g.length for g in gap_segments if g.filled
        )

        fill_rate = total_filled / total_gaps if total_gaps > 0 else 1.0

        # Strategy breakdown
        by_strategy: Dict[str, int] = {}
        for g in gap_segments:
            if g.filled:
                by_strategy[g.strategy] = by_strategy.get(g.strategy, 0) + 1

        # Position breakdown
        by_position: Dict[str, int] = {}
        for g in gap_segments:
            by_position[g.position] = by_position.get(g.position, 0) + 1

        # Confidence breakdown
        conf_values = [g.confidence for g in gap_segments if g.filled]
        min_confidence = min(conf_values) if conf_values else 0.0
        max_confidence = max(conf_values) if conf_values else 0.0

        # Compute impact
        impact = self._compute_impact(original, filled)

        report: Dict[str, Any] = {
            "series_length": n,
            "total_gaps": total_gaps,
            "total_filled": total_filled,
            "total_unfilled": total_unfilled,
            "total_missing_points": total_missing,
            "total_points_filled": total_points_filled,
            "fill_rate": fill_rate,
            "mean_confidence": mean_confidence,
            "min_confidence": min_confidence,
            "max_confidence": max_confidence,
            "by_strategy": by_strategy,
            "by_position": by_position,
            "impact": impact,
            "gap_details": fill_details,
            "generated_at": _utcnow().isoformat(),
        }

        logger.debug(
            "Stage REPORT: %d/%d gaps filled, fill_rate=%.2f, "
            "mean_conf=%.3f",
            total_filled, total_gaps, fill_rate, mean_confidence,
        )

        return report

    def _compute_impact(
        self,
        original: List[Optional[float]],
        filled: List[Optional[float]],
    ) -> Dict[str, Any]:
        """Compute the statistical impact of gap filling on the series.

        Compares descriptive statistics (mean, std, min, max) of the
        original non-missing values with the complete filled series.

        Args:
            original: Original series with gaps.
            filled: Filled series.

        Returns:
            Dict with before/after statistics and percentage changes.
        """
        orig_vals = [
            float(v) for v in original  # type: ignore[arg-type]
            if not _is_missing(v)
        ]
        fill_vals = [
            float(v) for v in filled  # type: ignore[arg-type]
            if not _is_missing(v)
        ]

        if not orig_vals:
            return {"before": {}, "after": {}, "change": {}}

        orig_mean = _safe_mean(orig_vals)
        fill_mean = _safe_mean(fill_vals)

        orig_std = self._compute_std(orig_vals)
        fill_std = self._compute_std(fill_vals)

        before = {
            "count": len(orig_vals),
            "mean": orig_mean,
            "std": orig_std,
            "min": min(orig_vals),
            "max": max(orig_vals),
        }
        after = {
            "count": len(fill_vals),
            "mean": fill_mean,
            "std": fill_std,
            "min": min(fill_vals) if fill_vals else 0.0,
            "max": max(fill_vals) if fill_vals else 0.0,
        }

        # Percentage changes
        def _pct_change(old: float, new: float) -> float:
            if abs(old) < 1e-12:
                return 0.0
            return ((new - old) / abs(old)) * 100.0

        change = {
            "count_added": len(fill_vals) - len(orig_vals),
            "mean_pct_change": _pct_change(orig_mean, fill_mean),
            "std_pct_change": _pct_change(orig_std, fill_std),
        }

        return {"before": before, "after": after, "change": change}

    # ==================================================================
    # Private: utility helpers
    # ==================================================================

    @staticmethod
    def _classify_position(start: int, end: int, n: int) -> str:
        """Classify a gap's position in the series.

        Args:
            start: First missing index.
            end: Last missing index.
            n: Series length.

        Returns:
            'leading', 'trailing', or 'interior'.
        """
        if start == 0:
            return "leading"
        if end >= n - 1:
            return "trailing"
        return "interior"

    @staticmethod
    def _simple_gap_scan(
        values: List[Optional[float]],
    ) -> List[GapSegment]:
        """Fallback gap detection via simple sequential scan.

        Args:
            values: Series values.

        Returns:
            List of GapSegment for each contiguous run of missing values.
        """
        n = len(values)
        segments: List[GapSegment] = []
        i = 0

        while i < n:
            if _is_missing(values[i]):
                start = i
                while i < n and _is_missing(values[i]):
                    i += 1
                end = i - 1
                length = end - start + 1

                if start == 0:
                    position = "leading"
                elif end >= n - 1:
                    position = "trailing"
                else:
                    position = "interior"

                segments.append(GapSegment(
                    start=start,
                    end=end,
                    length=length,
                    position=position,
                ))
            else:
                i += 1

        return segments

    @staticmethod
    def _simple_moving_average_fill(
        values: List[Optional[float]],
    ) -> Tuple[List[Optional[float]], float]:
        """Simple fallback: fill gaps with local window mean.

        Uses a window of 5 non-missing values on each side.

        Args:
            values: Series with gaps.

        Returns:
            Tuple of (filled_series, confidence).
        """
        n = len(values)
        filled = list(values)
        window = 5
        filled_count = 0

        for i in range(n):
            if not _is_missing(values[i]):
                continue

            # Collect neighbours
            neighbours: List[float] = []
            for direction in (-1, 1):
                pos = i + direction
                count = 0
                while 0 <= pos < n and count < window:
                    if not _is_missing(values[pos]):
                        neighbours.append(float(values[pos]))  # type: ignore[arg-type]
                        count += 1
                    pos += direction

            if neighbours:
                filled[i] = _safe_mean(neighbours)
                filled_count += 1

        total_missing = sum(1 for v in values if _is_missing(v))
        confidence = 0.5 * (filled_count / max(total_missing, 1))
        return filled, min(1.0, confidence)

    @staticmethod
    def _compute_std(values: List[float]) -> float:
        """Compute population standard deviation.

        Args:
            values: Numeric values.

        Returns:
            Standard deviation or 0.0.
        """
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        return math.sqrt(variance)

    def _finalize_pipeline(
        self,
        result: Dict[str, Any],
        pipeline_start: float,
        stage_timings: Dict[str, float],
        pipeline_id: str,
    ) -> None:
        """Finalize pipeline result with timing, provenance, and metrics.

        Args:
            result: Pipeline result dict to finalize.
            pipeline_start: time.time() at pipeline start.
            stage_timings: Per-stage timing dict.
            pipeline_id: Unique pipeline run identifier.
        """
        elapsed = time.time() - pipeline_start
        result["total_time_ms"] = elapsed * 1000.0
        result["stage_timings"] = stage_timings

        status = result.get("status", "unknown")

        # Provenance
        provenance_hash = self._provenance.build_hash({
            "operation": "run_pipeline",
            "pipeline_id": pipeline_id,
            "status": status,
            "gaps_detected": result.get("gaps_detected", 0),
            "gaps_filled": result.get("gaps_filled", 0),
            "points_filled": result.get("points_filled", 0),
            "total_time_ms": result["total_time_ms"],
        })
        self._provenance.record(
            "pipeline", pipeline_id, "pipeline", provenance_hash,
        )
        result["provenance_hash"] = provenance_hash

        # Metrics
        _inc_jobs_processed(status)
        _observe_duration("pipeline", elapsed)
        _set_active_jobs(0)

        if status == "completed" and result.get("validation"):
            validation = result["validation"]
            mean_conf = validation.get("mean_confidence", 0.0)
            if isinstance(mean_conf, (int, float)) and mean_conf > 0:
                _observe_confidence(float(mean_conf))

        # Update statistics
        self._statistics.total_runs += 1
        self._statistics.total_gaps_detected += result.get("gaps_detected", 0)
        self._statistics.total_gaps_filled += result.get("gaps_filled", 0)
        self._statistics.total_points_filled += result.get("points_filled", 0)

        # Update by_status
        status_key = status if status else "unknown"
        self._statistics.by_status[status_key] = (
            self._statistics.by_status.get(status_key, 0) + 1
        )

        # Update by_strategy from fill_details
        for detail in result.get("fill_details", []):
            if detail.get("filled"):
                strat = detail.get("strategy", "unknown")
                self._statistics.by_strategy[strat] = (
                    self._statistics.by_strategy.get(strat, 0) + 1
                )

        # Update by_validation
        validation = result.get("validation", {})
        val_level = validation.get("level", "unknown") if isinstance(
            validation, dict,
        ) else "unknown"
        self._statistics.by_validation[val_level] = (
            self._statistics.by_validation.get(val_level, 0) + 1
        )

        # Update avg_confidence (running weighted average)
        if result.get("gaps_filled", 0) > 0 and isinstance(validation, dict):
            new_conf = validation.get("mean_confidence", 0.0)
            old_total = self._statistics.total_runs - 1
            if old_total > 0 and isinstance(new_conf, (int, float)):
                self._statistics.avg_confidence = (
                    (self._statistics.avg_confidence * old_total + float(new_conf))
                    / self._statistics.total_runs
                )
            elif isinstance(new_conf, (int, float)):
                self._statistics.avg_confidence = float(new_conf)

        logger.info(
            "Pipeline %s finalized: status=%s, gaps=%d/%d filled, %.1fms",
            pipeline_id[:8], status,
            result.get("gaps_filled", 0),
            result.get("gaps_detected", 0),
            result["total_time_ms"],
        )


__all__ = [
    "GapFillerPipelineEngine",
    "GapSegment",
    "ValidationResult",
    "PipelineStatistics",
]
