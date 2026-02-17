# -*- coding: utf-8 -*-
"""
Refresh Predictor Engine - AGENT-DATA-016: Data Freshness Monitor (GL-DATA-X-019)

Predicts when the next data refresh will arrive for monitored datasets based
on historical refresh patterns. Uses a tiered prediction strategy that selects
the optimal algorithm depending on the number of historical samples available:

    - < prediction_min_samples:  cadence-based prediction (low confidence)
    - 5-10 samples:              mean-interval prediction (medium confidence)
    - 10+ samples:               weighted-recent prediction (high confidence)

Confidence scores incorporate three components:
    - Base confidence from sample count (50% weight)
    - Regularity bonus from coefficient of variation (30% weight)
    - Recency bonus from last-5 interval consistency (20% weight)

Zero-Hallucination Guarantees:
    - All predictions use deterministic arithmetic on datetime intervals
    - Confidence scores are computed from statistical properties only
    - No ML/LLM calls in the prediction path
    - SHA-256 provenance on every prediction mutation
    - Thread-safe in-memory storage with locking

Example:
    >>> from greenlang.data_freshness_monitor.refresh_predictor import RefreshPredictorEngine
    >>> engine = RefreshPredictorEngine()
    >>> from datetime import datetime, timezone, timedelta
    >>> now = datetime.now(timezone.utc)
    >>> history = [now - timedelta(hours=h) for h in [72, 48, 24]]
    >>> pred = engine.predict_next_refresh("ds-001", history, cadence_hours=24.0)
    >>> print(pred.predicted_at, pred.confidence)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

__all__ = [
    "PredictionStatus",
    "RefreshPrediction",
    "RefreshPredictorEngine",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _generate_id(prefix: str = "RPR") -> str:
    """Generate a unique identifier with the given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        String of the form ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _compute_provenance(operation: str, data_repr: str) -> str:
    """Compute SHA-256 provenance hash for a prediction operation.

    Args:
        operation: Name of the operation.
        data_repr: Serialised representation of the data involved.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    payload = f"{operation}:{data_repr}:{_utcnow().isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_stdev(values: List[float]) -> float:
    """Compute sample standard deviation, 0.0 for fewer than 2 values.

    Args:
        values: List of numeric values.

    Returns:
        Sample standard deviation or 0.0.
    """
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _compute_intervals_hours(refresh_history: List[datetime]) -> List[float]:
    """Compute consecutive intervals in hours from sorted refresh timestamps.

    Args:
        refresh_history: Sorted (ascending) list of UTC datetimes.

    Returns:
        List of intervals in hours between consecutive refreshes.
    """
    if len(refresh_history) < 2:
        return []
    sorted_ts = sorted(refresh_history)
    intervals: List[float] = []
    for i in range(1, len(sorted_ts)):
        delta_hours = (sorted_ts[i] - sorted_ts[i - 1]).total_seconds() / 3600.0
        intervals.append(max(0.0, delta_hours))
    return intervals


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class PredictionStatus(str, Enum):
    """Status of a refresh prediction evaluation.

    Values:
        PENDING: Prediction has not yet been evaluated against actual refresh.
        ON_TIME: Actual refresh arrived within 25% of cadence from predicted.
        LATE: Actual arrived after predicted, error within one cadence period.
        VERY_LATE: Actual arrived after predicted, error exceeds one cadence.
        MISSED: Refresh never arrived (set manually by operator).
    """

    PENDING = "pending"
    ON_TIME = "on_time"
    LATE = "late"
    VERY_LATE = "very_late"
    MISSED = "missed"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class RefreshPrediction(BaseModel):
    """A single refresh prediction for a monitored dataset.

    Contains the predicted next refresh time, the confidence of the
    prediction, the algorithm used, and provenance metadata.

    Attributes:
        prediction_id: Unique identifier for this prediction.
        dataset_id: Identifier of the dataset being predicted.
        predicted_at: Predicted UTC datetime of next refresh.
        confidence: Confidence score (0.0 to 1.0).
        algorithm: Name of the prediction algorithm used.
        cadence_hours: Expected cadence in hours.
        sample_count: Number of historical samples used.
        interval_stats: Statistical summary of historical intervals.
        status: Evaluation status of this prediction.
        actual_at: Actual refresh time (set after evaluation).
        error_hours: Prediction error in hours (set after evaluation).
        provenance_hash: SHA-256 provenance chain hash for audit trail.
        created_at: Timestamp when prediction was created.
    """

    prediction_id: str = Field(
        default_factory=lambda: _generate_id("RPR"),
        description="Unique identifier for this prediction",
    )
    dataset_id: str = Field(
        ..., description="Identifier of the dataset being predicted",
    )
    predicted_at: datetime = Field(
        ..., description="Predicted UTC datetime of next refresh",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence score (0.0 to 1.0)",
    )
    algorithm: str = Field(
        default="cadence",
        description="Name of the prediction algorithm used",
    )
    cadence_hours: float = Field(
        default=24.0, ge=0.0,
        description="Expected cadence in hours",
    )
    sample_count: int = Field(
        default=0, ge=0,
        description="Number of historical samples used",
    )
    interval_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Statistical summary of historical intervals",
    )
    status: PredictionStatus = Field(
        default=PredictionStatus.PENDING,
        description="Evaluation status of this prediction",
    )
    actual_at: Optional[datetime] = Field(
        None,
        description="Actual refresh time (set after evaluation)",
    )
    error_hours: Optional[float] = Field(
        None, ge=0.0,
        description="Prediction error in hours (set after evaluation)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when prediction was created",
    )

    model_config = {"extra": "forbid"}

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, v: str) -> str:
        """Validate dataset_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("dataset_id must be non-empty")
        return v


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default minimum samples before switching from cadence-based prediction.
_DEFAULT_MIN_SAMPLES = 5

#: Default exponential decay factor for weighted-recent estimation.
_DEFAULT_DECAY_FACTOR = 0.85

#: Threshold for coefficient of variation above which confidence is penalised.
_CV_IRREGULARITY_THRESHOLD = 0.3

#: Maximum prediction confidence (hard cap).
_MAX_CONFIDENCE = 0.95

#: Anomalous delay factor threshold (1.5 = 50% late).
_ANOMALY_DELAY_FACTOR = 1.5

#: Recency window size for the recency bonus calculation.
_RECENCY_WINDOW = 5

#: Recency tolerance multiplier relative to cadence.
_RECENCY_TOLERANCE_FACTOR = 1.5


# ---------------------------------------------------------------------------
# RefreshPredictorEngine
# ---------------------------------------------------------------------------


class RefreshPredictorEngine:
    """Predicts when the next data refresh will arrive for monitored datasets.

    Implements a tiered prediction strategy that selects the optimal
    algorithm depending on the number of historical samples available.
    Thread-safe: all mutations to internal storage are protected by a
    threading lock. SHA-256 provenance hashes on every prediction.

    Attributes:
        _prediction_min_samples: Minimum samples for non-cadence prediction.
        _default_decay_factor: Default exponential decay factor.
        _predictions: In-memory storage of predictions keyed by dataset_id.
        _evaluated: List of evaluated prediction records.
        _lock: Threading lock for thread-safe storage access.
        _stats: Aggregate tracking statistics.

    Example:
        >>> engine = RefreshPredictorEngine()
        >>> from datetime import datetime, timezone, timedelta
        >>> now = datetime.now(timezone.utc)
        >>> history = [now - timedelta(hours=h) for h in [72, 48, 24]]
        >>> pred = engine.predict_next_refresh("ds-001", history, 24.0)
        >>> assert pred.confidence > 0
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize RefreshPredictorEngine.

        Args:
            config: Optional configuration dict. Recognised keys:
                - ``prediction_min_samples``: int (default 5)
                - ``default_decay_factor``: float (default 0.85)
                - ``max_confidence``: float (default 0.95)
                - ``anomaly_delay_factor``: float (default 1.5)
        """
        self._config = config or {}
        self._prediction_min_samples: int = self._config.get(
            "prediction_min_samples", _DEFAULT_MIN_SAMPLES,
        )
        self._default_decay_factor: float = self._config.get(
            "default_decay_factor", _DEFAULT_DECAY_FACTOR,
        )
        self._max_confidence: float = self._config.get(
            "max_confidence", _MAX_CONFIDENCE,
        )
        self._anomaly_delay_factor: float = self._config.get(
            "anomaly_delay_factor", _ANOMALY_DELAY_FACTOR,
        )

        # Internal storage
        self._predictions: Dict[str, List[RefreshPrediction]] = {}
        self._evaluated: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "predictions_created": 0,
            "predictions_evaluated": 0,
            "anomalies_detected": 0,
            "total_prediction_time_ms": 0.0,
            "total_error_hours": 0.0,
        }

        logger.info(
            "RefreshPredictorEngine initialized: min_samples=%d, "
            "decay_factor=%.2f, max_confidence=%.2f, "
            "anomaly_delay_factor=%.1f",
            self._prediction_min_samples,
            self._default_decay_factor,
            self._max_confidence,
            self._anomaly_delay_factor,
        )

    # ------------------------------------------------------------------
    # Public API - Predict Next Refresh
    # ------------------------------------------------------------------

    def predict_next_refresh(
        self,
        dataset_id: str,
        refresh_history: List[datetime],
        cadence_hours: float,
    ) -> RefreshPrediction:
        """Predict when the next data refresh will arrive for a dataset.

        Uses a tiered strategy based on sample count:
            1. < min_samples: cadence-based prediction (low confidence ~0.3)
            2. 5-10 samples: mean-interval prediction (medium confidence)
            3. 10+ samples: weighted-recent prediction (high confidence)

        Args:
            dataset_id: Unique identifier of the monitored dataset.
            refresh_history: List of UTC datetimes when past refreshes
                occurred. Need not be sorted; engine sorts internally.
            cadence_hours: Expected refresh cadence in hours.

        Returns:
            RefreshPrediction containing the predicted time, confidence,
            algorithm used, interval statistics, and provenance hash.

        Raises:
            ValueError: If dataset_id is empty or cadence_hours <= 0.
        """
        start = time.monotonic()

        if not dataset_id or not dataset_id.strip():
            raise ValueError("dataset_id must be non-empty")
        if cadence_hours <= 0:
            raise ValueError("cadence_hours must be > 0")

        # Sort history ascending
        sorted_history = sorted(refresh_history)
        sample_count = len(sorted_history)

        # Compute interval statistics (may be empty)
        interval_stats = self.compute_interval_statistics(sorted_history)

        # Select prediction algorithm and estimate next refresh
        predicted_at, algorithm, confidence = self._select_and_predict(
            sorted_history, cadence_hours, interval_stats,
        )

        # Build provenance
        provenance_data = json.dumps({
            "dataset_id": dataset_id,
            "sample_count": sample_count,
            "cadence_hours": cadence_hours,
            "algorithm": algorithm,
            "predicted_at": predicted_at.isoformat(),
            "confidence": confidence,
        }, sort_keys=True, default=str)
        provenance_hash = _compute_provenance("predict_next_refresh", provenance_data)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        prediction = RefreshPrediction(
            dataset_id=dataset_id,
            predicted_at=predicted_at,
            confidence=confidence,
            algorithm=algorithm,
            cadence_hours=cadence_hours,
            sample_count=sample_count,
            interval_stats=interval_stats,
            status=PredictionStatus.PENDING,
            provenance_hash=provenance_hash,
        )

        # Store prediction
        with self._lock:
            if dataset_id not in self._predictions:
                self._predictions[dataset_id] = []
            self._predictions[dataset_id].append(prediction)
            self._stats["predictions_created"] += 1
            self._stats["total_prediction_time_ms"] += elapsed_ms

        logger.info(
            "Prediction created: id=%s, dataset=%s, algorithm=%s, "
            "confidence=%.3f, predicted_at=%s, elapsed=%.2fms",
            prediction.prediction_id, dataset_id, algorithm,
            confidence, predicted_at.isoformat(), elapsed_ms,
        )
        return prediction

    def predict_batch(
        self,
        predictions: List[Dict[str, Any]],
    ) -> List[RefreshPrediction]:
        """Predict next refresh for multiple datasets in a single call.

        Each entry in the list must contain:
            - ``dataset_id``: str
            - ``refresh_history``: List[datetime]
            - ``cadence_hours``: float

        Args:
            predictions: List of prediction request dicts.

        Returns:
            List of RefreshPrediction results, one per input.

        Raises:
            ValueError: If any individual prediction request is invalid.
        """
        results: List[RefreshPrediction] = []
        for entry in predictions:
            dataset_id = entry.get("dataset_id", "")
            refresh_history = entry.get("refresh_history", [])
            cadence_hours = entry.get("cadence_hours", 24.0)

            pred = self.predict_next_refresh(
                dataset_id=dataset_id,
                refresh_history=refresh_history,
                cadence_hours=cadence_hours,
            )
            results.append(pred)

        logger.info("Batch prediction completed: %d datasets", len(results))
        return results

    # ------------------------------------------------------------------
    # Interval Statistics
    # ------------------------------------------------------------------

    def compute_mean_interval(
        self,
        refresh_history: List[datetime],
    ) -> float:
        """Compute the mean interval between consecutive refreshes in hours.

        Args:
            refresh_history: List of UTC refresh datetimes.

        Returns:
            Mean interval in hours, or 0.0 if fewer than 2 samples.
        """
        intervals = _compute_intervals_hours(refresh_history)
        if not intervals:
            return 0.0
        return statistics.mean(intervals)

    def compute_median_interval(
        self,
        refresh_history: List[datetime],
    ) -> float:
        """Compute the median interval between consecutive refreshes in hours.

        Args:
            refresh_history: List of UTC refresh datetimes.

        Returns:
            Median interval in hours, or 0.0 if fewer than 2 samples.
        """
        intervals = _compute_intervals_hours(refresh_history)
        if not intervals:
            return 0.0
        return statistics.median(intervals)

    def compute_interval_statistics(
        self,
        refresh_history: List[datetime],
    ) -> Dict[str, Any]:
        """Compute comprehensive interval statistics for a refresh history.

        Returns mean, median, standard deviation, min, max, coefficient
        of variation, and interval count.

        Args:
            refresh_history: List of UTC refresh datetimes.

        Returns:
            Dict with keys: mean, median, stddev, min, max, cv, count.
        """
        intervals = _compute_intervals_hours(refresh_history)

        if not intervals:
            return {
                "mean": 0.0,
                "median": 0.0,
                "stddev": 0.0,
                "min": 0.0,
                "max": 0.0,
                "cv": 0.0,
                "count": 0,
            }

        mean_val = statistics.mean(intervals)
        median_val = statistics.median(intervals)
        stddev_val = _safe_stdev(intervals)
        min_val = min(intervals)
        max_val = max(intervals)

        # Coefficient of variation
        cv = stddev_val / mean_val if mean_val > 0 else 0.0

        return {
            "mean": round(mean_val, 4),
            "median": round(median_val, 4),
            "stddev": round(stddev_val, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "cv": round(cv, 4),
            "count": len(intervals),
        }

    # ------------------------------------------------------------------
    # Estimation Methods
    # ------------------------------------------------------------------

    def estimate_next_by_mean(
        self,
        refresh_history: List[datetime],
    ) -> datetime:
        """Estimate next refresh time using the mean interval.

        Adds the mean interval to the last (most recent) refresh time.

        Args:
            refresh_history: List of UTC refresh datetimes (min 2 required).

        Returns:
            Estimated next refresh datetime.

        Raises:
            ValueError: If fewer than 2 samples provided.
        """
        if len(refresh_history) < 2:
            raise ValueError(
                "At least 2 samples required for mean-interval estimation"
            )
        sorted_ts = sorted(refresh_history)
        mean_hours = self.compute_mean_interval(sorted_ts)
        last_refresh = sorted_ts[-1]
        return last_refresh + timedelta(hours=mean_hours)

    def estimate_next_by_median(
        self,
        refresh_history: List[datetime],
    ) -> datetime:
        """Estimate next refresh time using the median interval.

        Adds the median interval to the last (most recent) refresh time.

        Args:
            refresh_history: List of UTC refresh datetimes (min 2 required).

        Returns:
            Estimated next refresh datetime.

        Raises:
            ValueError: If fewer than 2 samples provided.
        """
        if len(refresh_history) < 2:
            raise ValueError(
                "At least 2 samples required for median-interval estimation"
            )
        sorted_ts = sorted(refresh_history)
        median_hours = self.compute_median_interval(sorted_ts)
        last_refresh = sorted_ts[-1]
        return last_refresh + timedelta(hours=median_hours)

    def estimate_next_by_weighted_recent(
        self,
        refresh_history: List[datetime],
        decay_factor: Optional[float] = None,
    ) -> datetime:
        """Estimate next refresh using exponentially weighted recent intervals.

        Applies exponential decay weighting so that more recent intervals
        carry higher weight. The most recent interval receives weight 1.0,
        the second most recent receives ``decay_factor``, the third receives
        ``decay_factor^2``, and so on.

        Args:
            refresh_history: List of UTC refresh datetimes (min 2 required).
            decay_factor: Exponential decay factor (0 < decay < 1).
                Defaults to ``self._default_decay_factor`` (0.85).

        Returns:
            Estimated next refresh datetime.

        Raises:
            ValueError: If fewer than 2 samples or invalid decay_factor.
        """
        if len(refresh_history) < 2:
            raise ValueError(
                "At least 2 samples required for weighted-recent estimation"
            )
        decay = decay_factor if decay_factor is not None else self._default_decay_factor
        if not (0.0 < decay < 1.0):
            raise ValueError(
                f"decay_factor must be between 0.0 and 1.0 exclusive, got {decay}"
            )

        sorted_ts = sorted(refresh_history)
        intervals = _compute_intervals_hours(sorted_ts)

        if not intervals:
            return sorted_ts[-1] + timedelta(hours=24.0)

        # Compute weights: most recent interval gets weight 1.0
        # Intervals are ordered oldest-first, so reverse for weighting
        reversed_intervals = list(reversed(intervals))
        weights: List[float] = []
        for i in range(len(reversed_intervals)):
            weights.append(decay ** i)

        # Weighted average
        total_weight = sum(weights)
        weighted_sum = sum(
            interval * weight
            for interval, weight in zip(reversed_intervals, weights)
        )
        weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0.0

        last_refresh = sorted_ts[-1]
        return last_refresh + timedelta(hours=weighted_avg)

    def estimate_next_by_cadence(
        self,
        last_refresh: datetime,
        cadence_hours: float,
    ) -> datetime:
        """Estimate next refresh time by adding the cadence to the last refresh.

        Simple cadence-based prediction used as a fallback when insufficient
        historical data is available.

        Args:
            last_refresh: Datetime of the most recent refresh.
            cadence_hours: Expected cadence in hours.

        Returns:
            Estimated next refresh datetime.
        """
        return last_refresh + timedelta(hours=cadence_hours)

    # ------------------------------------------------------------------
    # Confidence Calculation
    # ------------------------------------------------------------------

    def compute_prediction_confidence(
        self,
        refresh_history: List[datetime],
        cadence_hours: float,
    ) -> float:
        """Compute prediction confidence score from 0.0 to 1.0.

        Confidence is composed of three weighted components:

        1. **Base confidence (50%)**: ``min(1.0, count / 20) * 0.5``
           More historical samples yield higher base confidence.

        2. **Regularity bonus (30%)**: ``(1.0 - min(1.0, cv)) * 0.3``
           Lower coefficient of variation (more regular intervals) yields
           a higher regularity bonus.

        3. **Recency bonus (20%)**: 0.2 if the last 5 intervals are each
           within ``1.5 * cadence_hours``, otherwise 0.0.

        The final confidence is capped at ``max_confidence`` (default 0.95).

        Args:
            refresh_history: List of UTC refresh datetimes.
            cadence_hours: Expected cadence in hours.

        Returns:
            Confidence float between 0.0 and max_confidence.
        """
        intervals = _compute_intervals_hours(refresh_history)
        count = len(intervals)

        if count == 0:
            return 0.0

        # Component 1: Base confidence from sample count
        base = min(1.0, count / 20.0) * 0.5

        # Component 2: Regularity bonus from coefficient of variation
        mean_interval = statistics.mean(intervals) if intervals else 0.0
        stddev_interval = _safe_stdev(intervals)
        cv = stddev_interval / mean_interval if mean_interval > 0 else 0.0
        regularity_bonus = (1.0 - min(1.0, cv)) * 0.3

        # Component 3: Recency bonus from last N intervals
        recency_bonus = self._compute_recency_bonus(intervals, cadence_hours)

        confidence = base + regularity_bonus + recency_bonus
        confidence = min(confidence, self._max_confidence)
        confidence = max(0.0, confidence)

        return round(confidence, 4)

    def _compute_recency_bonus(
        self,
        intervals: List[float],
        cadence_hours: float,
    ) -> float:
        """Compute recency bonus for confidence calculation.

        Awards 0.2 if the last 5 intervals are each within
        1.5x the cadence. Awards 0.0 otherwise.

        Args:
            intervals: List of interval durations in hours (oldest first).
            cadence_hours: Expected cadence in hours.

        Returns:
            Recency bonus: 0.2 or 0.0.
        """
        if not intervals or cadence_hours <= 0:
            return 0.0

        window = intervals[-_RECENCY_WINDOW:]
        if len(window) < _RECENCY_WINDOW:
            # Not enough recent intervals for a full window bonus
            return 0.0

        tolerance = cadence_hours * _RECENCY_TOLERANCE_FACTOR
        for interval in window:
            if interval > tolerance:
                return 0.0

        return 0.2

    # ------------------------------------------------------------------
    # Prediction Evaluation
    # ------------------------------------------------------------------

    def evaluate_prediction(
        self,
        prediction_id: str,
        actual_refresh_at: datetime,
    ) -> Dict[str, Any]:
        """Evaluate a prediction against the actual refresh time.

        Computes the error and assigns a status:
            - on_time: error <= cadence * 0.25
            - late: actual > predicted AND error <= cadence
            - very_late: actual > predicted AND error > cadence
            - missed: set manually (not by this method)

        Args:
            prediction_id: The prediction ID to evaluate.
            actual_refresh_at: The actual UTC datetime of the refresh.

        Returns:
            Dict with: prediction_id, error_hours, status,
            predicted_at, actual_at, cadence_hours, provenance_hash.

        Raises:
            ValueError: If prediction_id is not found.
        """
        prediction = self._find_prediction(prediction_id)
        if prediction is None:
            raise ValueError(
                f"Prediction not found: {prediction_id}"
            )

        error_hours = abs(
            (prediction.predicted_at - actual_refresh_at).total_seconds() / 3600.0
        )
        cadence = prediction.cadence_hours

        # Determine status
        status = self._classify_prediction_error(
            prediction.predicted_at, actual_refresh_at, error_hours, cadence,
        )

        # Update prediction in storage
        with self._lock:
            prediction.actual_at = actual_refresh_at
            prediction.error_hours = round(error_hours, 4)
            prediction.status = status

        # Build evaluation record
        eval_record = {
            "prediction_id": prediction_id,
            "dataset_id": prediction.dataset_id,
            "predicted_at": prediction.predicted_at.isoformat(),
            "actual_at": actual_refresh_at.isoformat(),
            "error_hours": round(error_hours, 4),
            "status": status.value,
            "cadence_hours": cadence,
        }

        # Provenance
        provenance_data = json.dumps(eval_record, sort_keys=True, default=str)
        provenance_hash = _compute_provenance("evaluate_prediction", provenance_data)
        eval_record["provenance_hash"] = provenance_hash

        with self._lock:
            self._evaluated.append(eval_record)
            self._stats["predictions_evaluated"] += 1
            self._stats["total_error_hours"] += error_hours

        logger.info(
            "Prediction evaluated: id=%s, error=%.2fh, status=%s",
            prediction_id, error_hours, status.value,
        )
        return eval_record

    def _classify_prediction_error(
        self,
        predicted_at: datetime,
        actual_at: datetime,
        error_hours: float,
        cadence_hours: float,
    ) -> PredictionStatus:
        """Classify the prediction evaluation status.

        Args:
            predicted_at: Predicted refresh time.
            actual_at: Actual refresh time.
            error_hours: Absolute error in hours.
            cadence_hours: Expected cadence in hours.

        Returns:
            PredictionStatus enumeration value.
        """
        threshold_on_time = cadence_hours * 0.25

        if error_hours <= threshold_on_time:
            return PredictionStatus.ON_TIME

        if actual_at > predicted_at:
            if error_hours <= cadence_hours:
                return PredictionStatus.LATE
            return PredictionStatus.VERY_LATE

        # Actual arrived earlier than predicted but beyond on_time threshold
        # Treat as on_time since early arrival is generally not an issue,
        # but if the error is large, classify as late prediction
        if error_hours <= cadence_hours:
            return PredictionStatus.ON_TIME
        return PredictionStatus.LATE

    # ------------------------------------------------------------------
    # Anomaly Detection
    # ------------------------------------------------------------------

    def detect_anomalous_delay(
        self,
        last_refresh: datetime,
        cadence_hours: float,
        current_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Detect whether a dataset refresh is anomalously delayed.

        A delay is considered anomalous when the delay_factor exceeds
        the configured anomaly threshold (default 1.5, meaning 50% late).

        Args:
            last_refresh: Datetime of the most recent refresh.
            cadence_hours: Expected cadence in hours.
            current_time: Optional current UTC time. Defaults to now.

        Returns:
            Dict with: is_anomalous, expected_at, delay_hours,
            delay_factor, current_time, provenance_hash.
        """
        now = current_time if current_time is not None else _utcnow()
        expected_at = last_refresh + timedelta(hours=cadence_hours)
        delay_seconds = (now - expected_at).total_seconds()
        delay_hours = delay_seconds / 3600.0

        if delay_hours < 0:
            delay_hours = 0.0

        delay_factor = delay_hours / cadence_hours if cadence_hours > 0 else 0.0
        is_anomalous = delay_factor > self._anomaly_delay_factor

        provenance_data = json.dumps({
            "last_refresh": last_refresh.isoformat(),
            "cadence_hours": cadence_hours,
            "delay_hours": round(delay_hours, 4),
            "delay_factor": round(delay_factor, 4),
            "is_anomalous": is_anomalous,
        }, sort_keys=True, default=str)
        provenance_hash = _compute_provenance("detect_anomalous_delay", provenance_data)

        if is_anomalous:
            with self._lock:
                self._stats["anomalies_detected"] += 1
            logger.warning(
                "Anomalous delay detected: delay=%.2fh, factor=%.2f, "
                "expected_at=%s",
                delay_hours, delay_factor, expected_at.isoformat(),
            )

        return {
            "is_anomalous": is_anomalous,
            "expected_at": expected_at.isoformat(),
            "delay_hours": round(delay_hours, 4),
            "delay_factor": round(delay_factor, 4),
            "current_time": now.isoformat(),
            "last_refresh": last_refresh.isoformat(),
            "cadence_hours": cadence_hours,
            "provenance_hash": provenance_hash,
        }

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_predictions(
        self,
        dataset_id: str,
    ) -> List[RefreshPrediction]:
        """Retrieve all predictions for a given dataset.

        Args:
            dataset_id: The dataset identifier.

        Returns:
            List of RefreshPrediction objects, newest first.
        """
        with self._lock:
            preds = list(self._predictions.get(dataset_id, []))
        return list(reversed(preds))

    def get_prediction_accuracy(self) -> Dict[str, Any]:
        """Compute overall prediction accuracy statistics.

        Returns:
            Dict with: total, evaluated, mean_error_hours,
            median_error_hours, by_status counts.
        """
        with self._lock:
            evaluated = list(self._evaluated)

        total_predictions = 0
        with self._lock:
            for preds in self._predictions.values():
                total_predictions += len(preds)

        if not evaluated:
            return {
                "total": total_predictions,
                "evaluated": 0,
                "mean_error_hours": 0.0,
                "median_error_hours": 0.0,
                "by_status": {},
            }

        errors = [e["error_hours"] for e in evaluated]
        mean_error = statistics.mean(errors)
        median_error = statistics.median(errors)

        # Count by status
        status_counts: Dict[str, int] = {}
        for e in evaluated:
            st = e.get("status", "unknown")
            status_counts[st] = status_counts.get(st, 0) + 1

        return {
            "total": total_predictions,
            "evaluated": len(evaluated),
            "mean_error_hours": round(mean_error, 4),
            "median_error_hours": round(median_error, 4),
            "by_status": status_counts,
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate tracking statistics for the prediction engine.

        Returns:
            Dict with prediction counts, evaluation counts, anomaly
            counts, timing, and accuracy information.
        """
        with self._lock:
            created = self._stats["predictions_created"]
            evaluated = self._stats["predictions_evaluated"]
            anomalies = self._stats["anomalies_detected"]
            total_time = self._stats["total_prediction_time_ms"]
            total_error = self._stats["total_error_hours"]
            dataset_count = len(self._predictions)
            total_preds = sum(
                len(preds) for preds in self._predictions.values()
            )

        avg_time = total_time / created if created > 0 else 0.0
        avg_error = total_error / evaluated if evaluated > 0 else 0.0

        return {
            "predictions_created": created,
            "predictions_evaluated": evaluated,
            "anomalies_detected": anomalies,
            "datasets_tracked": dataset_count,
            "total_predictions_stored": total_preds,
            "total_prediction_time_ms": round(total_time, 2),
            "avg_prediction_time_ms": round(avg_time, 2),
            "avg_error_hours": round(avg_error, 4),
            "total_evaluated_records": len(self._evaluated),
            "timestamp": _utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all internal storage and statistics.

        Clears predictions, evaluations, and resets all counters.
        Primarily used for testing and development.
        """
        with self._lock:
            self._predictions.clear()
            self._evaluated.clear()
            self._stats = {
                "predictions_created": 0,
                "predictions_evaluated": 0,
                "anomalies_detected": 0,
                "total_prediction_time_ms": 0.0,
                "total_error_hours": 0.0,
            }
        logger.info("RefreshPredictorEngine reset: all state cleared")

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _select_and_predict(
        self,
        sorted_history: List[datetime],
        cadence_hours: float,
        interval_stats: Dict[str, Any],
    ) -> Tuple[datetime, str, float]:
        """Select the best prediction algorithm and compute the prediction.

        Decision logic:
            1. If fewer than min_samples refreshes, use cadence-based.
            2. If 5-10 samples, use mean-interval.
            3. If 10+ samples, use weighted-recent.
            4. Adjust confidence down if CV > 0.3.

        Args:
            sorted_history: Sorted (ascending) list of refresh datetimes.
            cadence_hours: Expected cadence in hours.
            interval_stats: Pre-computed interval statistics dict.

        Returns:
            Tuple of (predicted_at, algorithm_name, confidence).
        """
        count = len(sorted_history)

        # Case 1: Insufficient history -- cadence-based fallback
        if count < self._prediction_min_samples:
            return self._predict_with_cadence(
                sorted_history, cadence_hours,
            )

        # Case 2: Medium history (5-10 samples) -- mean-interval
        if count <= 10:
            return self._predict_with_mean(
                sorted_history, cadence_hours, interval_stats,
            )

        # Case 3: Rich history (10+ samples) -- weighted-recent
        return self._predict_with_weighted_recent(
            sorted_history, cadence_hours, interval_stats,
        )

    def _predict_with_cadence(
        self,
        sorted_history: List[datetime],
        cadence_hours: float,
    ) -> Tuple[datetime, str, float]:
        """Cadence-based prediction for sparse history.

        Args:
            sorted_history: Sorted refresh datetimes.
            cadence_hours: Expected cadence in hours.

        Returns:
            Tuple of (predicted_at, "cadence", confidence).
        """
        if sorted_history:
            last_refresh = sorted_history[-1]
        else:
            last_refresh = _utcnow()

        predicted_at = self.estimate_next_by_cadence(last_refresh, cadence_hours)
        confidence = 0.3  # Low confidence for cadence-only prediction

        return predicted_at, "cadence", confidence

    def _predict_with_mean(
        self,
        sorted_history: List[datetime],
        cadence_hours: float,
        interval_stats: Dict[str, Any],
    ) -> Tuple[datetime, str, float]:
        """Mean-interval prediction for medium-size history.

        Args:
            sorted_history: Sorted refresh datetimes.
            cadence_hours: Expected cadence in hours.
            interval_stats: Pre-computed interval statistics.

        Returns:
            Tuple of (predicted_at, "mean_interval", confidence).
        """
        predicted_at = self.estimate_next_by_mean(sorted_history)
        confidence = self.compute_prediction_confidence(
            sorted_history, cadence_hours,
        )

        # Adjust confidence down if CV indicates irregular refreshes
        cv = interval_stats.get("cv", 0.0)
        if cv > _CV_IRREGULARITY_THRESHOLD:
            penalty = min(0.2, (cv - _CV_IRREGULARITY_THRESHOLD) * 0.5)
            confidence = max(0.0, confidence - penalty)

        return predicted_at, "mean_interval", round(confidence, 4)

    def _predict_with_weighted_recent(
        self,
        sorted_history: List[datetime],
        cadence_hours: float,
        interval_stats: Dict[str, Any],
    ) -> Tuple[datetime, str, float]:
        """Weighted-recent prediction for rich history.

        Args:
            sorted_history: Sorted refresh datetimes.
            cadence_hours: Expected cadence in hours.
            interval_stats: Pre-computed interval statistics.

        Returns:
            Tuple of (predicted_at, "weighted_recent", confidence).
        """
        predicted_at = self.estimate_next_by_weighted_recent(
            sorted_history, self._default_decay_factor,
        )
        confidence = self.compute_prediction_confidence(
            sorted_history, cadence_hours,
        )

        # Adjust confidence down if CV indicates irregular refreshes
        cv = interval_stats.get("cv", 0.0)
        if cv > _CV_IRREGULARITY_THRESHOLD:
            penalty = min(0.2, (cv - _CV_IRREGULARITY_THRESHOLD) * 0.5)
            confidence = max(0.0, confidence - penalty)

        return predicted_at, "weighted_recent", round(confidence, 4)

    def _find_prediction(
        self,
        prediction_id: str,
    ) -> Optional[RefreshPrediction]:
        """Look up a prediction by its ID across all datasets.

        Args:
            prediction_id: The prediction identifier to find.

        Returns:
            RefreshPrediction if found, None otherwise.
        """
        with self._lock:
            for preds in self._predictions.values():
                for pred in preds:
                    if pred.prediction_id == prediction_id:
                        return pred
        return None
