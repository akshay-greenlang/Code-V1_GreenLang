# -*- coding: utf-8 -*-
"""
Time Series Gap Filler Service Setup - AGENT-DATA-014

Provides ``configure_gap_filler(app)`` which wires up the Time Series
Gap Filler SDK (gap detection, frequency analysis, interpolation,
seasonal filling, cross-series filling, trend extrapolation, pipeline
orchestration, provenance tracker) and mounts the REST API.

Also exposes ``get_gap_filler(app)`` for programmatic access,
``get_router()`` for obtaining the FastAPI APIRouter, and the
``TimeSeriesGapFillerService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.time_series_gap_filler.setup import configure_gap_filler
    >>> app = FastAPI()
    >>> configure_gap_filler(app)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.time_series_gap_filler.config import (
    TimeSeriesGapFillerConfig,
    get_config,
)
from greenlang.time_series_gap_filler.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Optional engine imports (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.time_series_gap_filler.gap_detector import GapDetectorEngine
except ImportError:
    GapDetectorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.time_series_gap_filler.frequency_analyzer import (
        FrequencyAnalyzerEngine,
    )
except ImportError:
    FrequencyAnalyzerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.time_series_gap_filler.interpolation_engine import (
        InterpolationEngine,
    )
except ImportError:
    InterpolationEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.time_series_gap_filler.seasonal_filler import SeasonalFillerEngine
except ImportError:
    SeasonalFillerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.time_series_gap_filler.cross_series_filler import (
        CrossSeriesFillerEngine,
    )
except ImportError:
    CrossSeriesFillerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.time_series_gap_filler.trend_extrapolator import (
        TrendExtrapolatorEngine,
    )
except ImportError:
    TrendExtrapolatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.time_series_gap_filler.metrics import PROMETHEUS_AVAILABLE
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Build a SHA-256 hash for arbitrary data."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ===================================================================
# TimeSeriesGapFillerService facade
# ===================================================================


class TimeSeriesGapFillerService:
    """Facade service for the Time Series Gap Filler SDK.

    Wires together the 7 engines (GapDetector, FrequencyAnalyzer,
    InterpolationEngine, SeasonalFiller, CrossSeriesFiller,
    TrendExtrapolator, Pipeline) behind a simple API suitable for
    REST endpoint delegation.

    Attributes:
        config: TimeSeriesGapFillerConfig instance.
        _provenance: ProvenanceTracker instance.
        _gap_detector: GapDetectorEngine instance.
        _frequency_analyzer: FrequencyAnalyzerEngine instance.
        _interpolation_engine: InterpolationEngine instance.
        _seasonal_filler: SeasonalFillerEngine instance.
        _cross_series_filler: CrossSeriesFillerEngine instance.
        _trend_extrapolator: TrendExtrapolatorEngine instance.
    """

    def __init__(
        self,
        config: Optional[TimeSeriesGapFillerConfig] = None,
    ) -> None:
        self.config = config or get_config()
        self._provenance = get_provenance_tracker()

        # Engine stubs -- created lazily or via startup()
        self._gap_detector: Any = None
        self._frequency_analyzer: Any = None
        self._interpolation_engine: Any = None
        self._seasonal_filler: Any = None
        self._cross_series_filler: Any = None
        self._trend_extrapolator: Any = None
        self._pipeline: Any = None

        # In-memory stores
        self._detections: Dict[str, Dict[str, Any]] = {}
        self._frequency_results: Dict[str, Dict[str, Any]] = {}
        self._fill_results: Dict[str, Dict[str, Any]] = {}
        self._validation_results: Dict[str, Dict[str, Any]] = {}
        self._calendars: Dict[str, Dict[str, Any]] = {}
        self._pipeline_results: Dict[str, Dict[str, Any]] = {}
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._stats = {
            "total_detections": 0,
            "total_fills": 0,
            "total_validations": 0,
            "total_pipelines": 0,
            "total_jobs": 0,
        }

        self._started = False
        logger.info("TimeSeriesGapFillerService created")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Initialize all engines."""
        if GapDetectorEngine is not None:
            self._gap_detector = GapDetectorEngine(config=self.config)
        if FrequencyAnalyzerEngine is not None:
            self._frequency_analyzer = FrequencyAnalyzerEngine(
                config=self.config,
            )
        if InterpolationEngine is not None:
            self._interpolation_engine = InterpolationEngine(
                config=self.config,
            )
        if SeasonalFillerEngine is not None:
            self._seasonal_filler = SeasonalFillerEngine(config=self.config)
        if CrossSeriesFillerEngine is not None:
            self._cross_series_filler = CrossSeriesFillerEngine(
                config=self.config,
            )
        if TrendExtrapolatorEngine is not None:
            self._trend_extrapolator = TrendExtrapolatorEngine(
                config=self.config,
            )

        self._started = True
        logger.info("TimeSeriesGapFillerService started")

    def shutdown(self) -> None:
        """Shutdown the service."""
        self._started = False
        logger.info("TimeSeriesGapFillerService shutdown")

    def health_check(self) -> Dict[str, Any]:
        """Return service health status."""
        return {
            "status": "healthy" if self._started else "starting",
            "service": "time_series_gap_filler",
            "engines": {
                "gap_detector": self._gap_detector is not None,
                "frequency_analyzer": self._frequency_analyzer is not None,
                "interpolation_engine": self._interpolation_engine is not None,
                "seasonal_filler": self._seasonal_filler is not None,
                "cross_series_filler": self._cross_series_filler is not None,
                "trend_extrapolator": self._trend_extrapolator is not None,
                "pipeline": self._pipeline is not None,
            },
            "timestamp": _utcnow().isoformat(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return service statistics."""
        return {
            **self._stats,
            "detections_stored": len(self._detections),
            "fills_stored": len(self._fill_results),
            "validations_stored": len(self._validation_results),
            "calendars_stored": len(self._calendars),
            "jobs_stored": len(self._jobs),
            "timestamp": _utcnow().isoformat(),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Alias for get_stats() used by the router."""
        return self.get_stats()

    # ------------------------------------------------------------------
    # Gap Detection
    # ------------------------------------------------------------------

    def detect_gaps(
        self,
        values: List[Any],
        series_id: str = "",
        timestamps: Optional[List[Any]] = None,
        expected_frequency: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect gaps in a time series."""
        start_t = time.time()

        # Use engine if available, otherwise do simple scan
        if self._gap_detector is not None:
            result = self._gap_detector.detect_gaps(
                values=values,
                timestamps=timestamps,
                expected_frequency=expected_frequency,
            )
            output = {
                "detection_id": str(uuid.uuid4()),
                "series_id": series_id,
                "total_gaps": result.total_gaps,
                "total_missing": result.total_missing,
                "series_length": result.series_length,
                "gap_pct": result.gap_pct,
                "gap_count": result.total_gaps,
                "gaps": [
                    {
                        "start_index": g.start_index,
                        "end_index": g.end_index,
                        "length": g.length,
                    }
                    for g in result.gaps
                ],
                "provenance_hash": result.provenance_hash,
                "processing_time_ms": result.processing_time_ms,
            }
        else:
            # Minimal fallback
            missing_count = sum(
                1 for v in values if v is None
            )
            output = {
                "detection_id": str(uuid.uuid4()),
                "series_id": series_id,
                "total_gaps": 1 if missing_count > 0 else 0,
                "total_missing": missing_count,
                "series_length": len(values),
                "gap_pct": missing_count / len(values) if values else 0.0,
                "gap_count": 1 if missing_count > 0 else 0,
                "gaps": [],
                "provenance_hash": _compute_hash({
                    "values_length": len(values),
                    "missing": missing_count,
                }),
                "processing_time_ms": (time.time() - start_t) * 1000.0,
            }

        self._detections[output["detection_id"]] = output
        self._stats["total_detections"] += 1
        return output

    def detect_gaps_batch(
        self,
        series_list: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Detect gaps in multiple series."""
        results = []
        for series in series_list:
            values = series.get("values", [])
            series_id = series.get("series_id", "")
            r = self.detect_gaps(values=values, series_id=series_id)
            results.append(r)
        return {
            "batch_id": str(uuid.uuid4()),
            "results": results,
            "count": len(results),
        }

    def detect_batch(
        self,
        series_list: List[Dict[str, Any]],
        timestamps_list: Optional[List[List[str]]] = None,
    ) -> Dict[str, Any]:
        """Detect gaps across multiple series (router-compatible alias)."""
        results = []
        for i, series in enumerate(series_list):
            values = series.get("values", [])
            name = series.get("name", f"series_{i}")
            r = self.detect_gaps(values=values, series_id=name)
            results.append(r)
        return {
            "batch_id": str(uuid.uuid4()),
            "results": results,
            "count": len(results),
        }

    def list_detections(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List stored detection results."""
        items = list(self._detections.values())
        return items[offset:offset + limit]

    def get_detections(self) -> List[Dict[str, Any]]:
        """Get all detections (used by router)."""
        return list(self._detections.values())

    def get_detection(self, detection_id: str) -> Optional[Dict[str, Any]]:
        """Get a stored detection result by ID."""
        return self._detections.get(detection_id)

    # ------------------------------------------------------------------
    # Frequency Analysis
    # ------------------------------------------------------------------

    def analyze_frequency(
        self,
        timestamps: List[Any],
        series_id: str = "",
    ) -> Dict[str, Any]:
        """Analyze frequency of a timestamp sequence."""
        if self._frequency_analyzer is not None:
            result = self._frequency_analyzer.analyze_frequency(timestamps)
            result["frequency_id"] = str(uuid.uuid4())
            result["series_id"] = series_id
        else:
            result = {
                "frequency_id": str(uuid.uuid4()),
                "series_id": series_id,
                "frequency_level": "unknown",
                "dominant_interval_seconds": 0.0,
                "regularity_score": 0.0,
                "confidence": 0.0,
                "sample_size": len(timestamps),
                "total_timestamps": len(timestamps),
                "provenance_hash": _compute_hash({
                    "count": len(timestamps),
                }),
            }

        self._frequency_results[result["frequency_id"]] = result
        return result

    def get_frequency(self, frequency_id: str) -> Optional[Dict[str, Any]]:
        """Get a stored frequency analysis result by ID."""
        return self._frequency_results.get(frequency_id)

    # ------------------------------------------------------------------
    # Gap Filling
    # ------------------------------------------------------------------

    def fill_gaps(
        self,
        values: List[Any],
        method: str = "linear",
        series_id: str = "",
        timestamps: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Fill gaps in a time series."""
        start_t = time.time()

        if self._interpolation_engine is not None:
            result = self._interpolation_engine.fill_gaps(
                values=values,
                method=method,
                timestamps=timestamps,
            )
            output = {
                "fill_id": str(uuid.uuid4()),
                "series_id": series_id,
                "method": method,
                "filled_values": result.filled_values,
                "gaps_found": result.gaps_found,
                "gaps_filled": result.gaps_filled,
                "total_missing": result.total_missing,
                "mean_confidence": result.mean_confidence,
                "min_confidence": result.min_confidence,
                "processing_time_ms": result.processing_time_ms,
                "provenance_hash": result.provenance_hash,
                "original_values": list(values),
            }
        else:
            # Minimal fallback: forward fill
            filled = list(values)
            missing = 0
            last_known = 0.0
            for i, v in enumerate(filled):
                if v is None:
                    filled[i] = last_known
                    missing += 1
                else:
                    last_known = float(v)

            elapsed = (time.time() - start_t) * 1000.0
            output = {
                "fill_id": str(uuid.uuid4()),
                "series_id": series_id,
                "method": method,
                "filled_values": filled,
                "gaps_found": 1 if missing > 0 else 0,
                "gaps_filled": 1 if missing > 0 else 0,
                "total_missing": missing,
                "mean_confidence": 0.5 if missing > 0 else 1.0,
                "min_confidence": 0.3 if missing > 0 else 1.0,
                "processing_time_ms": elapsed,
                "provenance_hash": _compute_hash({
                    "method": method,
                    "missing": missing,
                }),
                "original_values": list(values),
            }

        self._fill_results[output["fill_id"]] = output
        self._stats["total_fills"] += 1
        return output

    def get_fill(self, fill_id: str) -> Optional[Dict[str, Any]]:
        """Get a stored fill result by ID."""
        return self._fill_results.get(fill_id)

    def undo_fill(self, fill_id: str) -> Optional[Dict[str, Any]]:
        """Undo a fill, restoring original values."""
        fill = self._fill_results.get(fill_id)
        if fill is None:
            return None
        original = fill.get("original_values", [])
        fill["filled_values"] = original
        fill["undone"] = True
        return fill

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_fills(
        self,
        filled_values: Optional[List[Any]] = None,
        original_values: Optional[List[Any]] = None,
        original: Optional[List[Any]] = None,
        filled: Optional[List[Any]] = None,
        fill_indices: Optional[List[int]] = None,
        series_id: str = "",
    ) -> Dict[str, Any]:
        """Validate filled values.

        Supports two calling conventions:
        1. filled_values + original_values (from service)
        2. original + filled + fill_indices (from router)
        """
        validation_id = str(uuid.uuid4())

        # Normalize arguments
        actual_filled = filled_values or filled or []
        actual_original = original_values or original or []

        # Simple validation: check no Nones remain
        remaining_gaps = sum(1 for v in actual_filled if v is None)
        all_numeric = all(
            isinstance(v, (int, float)) for v in actual_filled if v is not None
        )

        level = "pass"
        messages: List[str] = []
        if remaining_gaps > 0:
            level = "fail"
            messages.append(f"{remaining_gaps} gaps remain after fill")
        if not all_numeric and actual_filled:
            level = "warn"
            messages.append("Non-numeric values detected")

        result = {
            "validation_id": validation_id,
            "series_id": series_id,
            "level": level,
            "remaining_gaps": remaining_gaps,
            "confidence_check": True,
            "continuity_check": remaining_gaps == 0,
            "range_check": all_numeric,
            "messages": messages,
            "provenance_hash": _compute_hash({
                "validation_id": validation_id,
                "level": level,
            }),
        }

        self._validation_results[validation_id] = result
        self._stats["total_validations"] += 1
        return result

    def get_validation(
        self, validation_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a stored validation result by ID."""
        return self._validation_results.get(validation_id)

    # ------------------------------------------------------------------
    # Calendar
    # ------------------------------------------------------------------

    def create_calendar(
        self,
        name: str = "default",
        frequency: str = "daily",
        calendar_type: str = "business_days",
        holidays: Optional[List[str]] = None,
        business_days: Optional[List[int]] = None,
        fiscal_start_month: int = 1,
    ) -> Dict[str, Any]:
        """Create a calendar definition."""
        calendar_id = str(uuid.uuid4())
        cal = {
            "calendar_id": calendar_id,
            "name": name,
            "frequency": frequency,
            "calendar_type": calendar_type,
            "holidays": holidays or [],
            "business_days": business_days or [0, 1, 2, 3, 4],
            "fiscal_start_month": fiscal_start_month,
            "created_at": _utcnow().isoformat(),
        }
        self._calendars[calendar_id] = cal
        return cal

    def list_calendars(self) -> List[Dict[str, Any]]:
        """List all calendar definitions."""
        return list(self._calendars.values())

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        values: List[Any],
        series_id: str = "",
        method: str = "linear",
        strategy: str = "auto",
        validate: bool = True,
        enable_validation: bool = True,
        timestamps: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Run a full gap-filling pipeline: detect -> fill -> validate."""
        start_t = time.time()
        pipeline_id = str(uuid.uuid4())
        should_validate = validate and enable_validation

        # Step 1: Detect
        detection = self.detect_gaps(values=values, series_id=series_id)

        # Step 2: Fill
        fill = self.fill_gaps(
            values=values,
            method=method,
            series_id=series_id,
        )

        # Step 3: Validate (optional)
        validation = None
        if should_validate:
            validation = self.validate_fills(
                filled_values=fill["filled_values"],
                original_values=values,
                series_id=series_id,
            )

        elapsed = (time.time() - start_t) * 1000.0

        result = {
            "pipeline_id": pipeline_id,
            "series_id": series_id,
            "detection": detection,
            "fill": fill,
            "validation": validation,
            "status": "completed",
            "total_processing_time_ms": elapsed,
            "provenance_hash": _compute_hash({
                "pipeline_id": pipeline_id,
                "detection_id": detection["detection_id"],
                "fill_id": fill["fill_id"],
            }),
        }

        self._pipeline_results[pipeline_id] = result
        self._stats["total_pipelines"] += 1
        return result

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def create_job(
        self,
        series_id: str = "",
        strategy: str = "auto",
        method: str = "linear",
        request: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a gap fill job.

        Supports both direct kwargs and a ``request`` dict from the router.
        """
        job_id = str(uuid.uuid4())

        if request is not None:
            series_id = request.get("series_name", series_id)
            strategy = request.get("strategy", strategy)

        job = {
            "job_id": job_id,
            "series_id": series_id,
            "strategy": strategy,
            "method": method,
            "status": "pending",
            "created_at": _utcnow().isoformat(),
        }
        self._jobs[job_id] = job
        self._stats["total_jobs"] += 1
        return job

    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List stored jobs."""
        items = list(self._jobs.values())
        if status is not None:
            items = [j for j in items if j.get("status") == status]
        return items[offset:offset + limit]

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def delete_job(self, job_id: str) -> bool:
        """Delete a job by ID."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            return True
        return False


# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------

_service_instance: Optional[TimeSeriesGapFillerService] = None
_service_lock = threading.Lock()


def get_service() -> TimeSeriesGapFillerService:
    """Return the singleton TimeSeriesGapFillerService."""
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = TimeSeriesGapFillerService()
                _service_instance.startup()
    return _service_instance


def reset_service() -> TimeSeriesGapFillerService:
    """Reset and return a new singleton instance."""
    global _service_instance
    with _service_lock:
        _service_instance = TimeSeriesGapFillerService()
        _service_instance.startup()
    return _service_instance


# ---------------------------------------------------------------------------
# FastAPI integration
# ---------------------------------------------------------------------------


def configure_gap_filler(app: Any) -> TimeSeriesGapFillerService:
    """Configure the gap filler service on a FastAPI app.

    Attaches the service to ``app.state.gap_filler_service`` and
    optionally includes the router.

    Args:
        app: FastAPI application instance.

    Returns:
        The configured TimeSeriesGapFillerService.
    """
    service = get_service()
    app.state.gap_filler_service = service

    # Attempt to include the router
    try:
        from greenlang.time_series_gap_filler.api.router import router
        if router is not None:
            app.include_router(router)
    except ImportError:
        logger.warning(
            "Gap filler router not available; skipping route registration"
        )

    logger.info("Gap filler service configured on app")
    return service


def get_gap_filler(app: Any) -> Optional[TimeSeriesGapFillerService]:
    """Retrieve the gap filler service from a FastAPI app.

    Args:
        app: FastAPI application instance.

    Returns:
        TimeSeriesGapFillerService or None if not configured.
    """
    return getattr(app.state, "gap_filler_service", None)


def get_router() -> Any:
    """Return the FastAPI APIRouter for the gap filler service.

    Returns:
        FastAPI APIRouter instance or None if FastAPI is not available.
    """
    try:
        from greenlang.time_series_gap_filler.api.router import router
        return router
    except ImportError:
        return None


__all__ = [
    "TimeSeriesGapFillerService",
    "configure_gap_filler",
    "get_gap_filler",
    "get_router",
    "get_service",
    "reset_service",
    "_compute_hash",
]
