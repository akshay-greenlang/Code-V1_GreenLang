# -*- coding: utf-8 -*-
"""
Freshness Checker Engine - AGENT-DATA-016

Performs freshness checks on registered datasets by computing age,
freshness score, and SLA status.  Supports single-dataset and batch
checks, dataset group aggregation, SLA compliance summaries, freshness
heatmaps, check history tracking, and stale dataset identification.

Engine 3 of 7 in the Data Freshness Monitor Agent SDK.

Zero-Hallucination: All calculations use deterministic Python arithmetic
(math, datetime, hashlib). No LLM calls for numeric computations or
freshness scoring logic. No external numerical libraries required.

Freshness Score Algorithm (piecewise-linear, 5-tier):
    - age <= excellent_hours -> score = 1.0 (EXCELLENT)
    - excellent < age <= good_hours -> linear 1.0 -> 0.85 (GOOD)
    - good < age <= fair_hours -> linear 0.85 -> 0.70 (FAIR)
    - fair < age <= poor_hours -> linear 0.70 -> 0.50 (POOR)
    - age > poor_hours -> max(0.0, 0.50 - (age-poor)/(poor*2)*0.50) (STALE)

Example:
    >>> from greenlang.data_freshness_monitor.freshness_checker import FreshnessCheckerEngine
    >>> engine = FreshnessCheckerEngine()
    >>> from datetime import datetime, timezone, timedelta
    >>> now = datetime.now(timezone.utc)
    >>> check = engine.check_freshness("ds-001", now - timedelta(hours=0.5), 24.0, 72.0)
    >>> assert check.freshness_level == "excellent"

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["FreshnessCheckerEngine"]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FreshnessLevel(str, Enum):
    """Freshness tier classification for a dataset."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    STALE = "stale"


class SLAStatus(str, Enum):
    """SLA compliance status for a dataset freshness check."""

    COMPLIANT = "compliant"
    WARNING = "warning"
    BREACHED = "breached"


# ---------------------------------------------------------------------------
# Data Models (self-contained until models.py is built)
# ---------------------------------------------------------------------------


@dataclass
class FreshnessCheck:
    """Result of a single freshness check on a dataset.

    Attributes:
        check_id: Unique identifier for this check (FC-<uuid4_hex[:12]>).
        dataset_id: Identifier of the dataset being checked.
        last_refreshed_at: UTC timestamp of the dataset's last refresh.
        checked_at: UTC timestamp when this check was performed.
        age_hours: Age of the dataset in hours since last refresh.
        freshness_score: Freshness score between 0.0 and 1.0.
        freshness_level: Freshness tier classification.
        sla_warning_hours: Warning threshold in hours.
        sla_critical_hours: Critical threshold in hours.
        sla_status: SLA compliance status.
        provenance_hash: SHA-256 hash for audit trail.
    """

    check_id: str = ""
    dataset_id: str = ""
    last_refreshed_at: str = ""
    checked_at: str = ""
    age_hours: float = 0.0
    freshness_score: float = 1.0
    freshness_level: str = FreshnessLevel.EXCELLENT.value
    sla_warning_hours: float = 24.0
    sla_critical_hours: float = 72.0
    sla_status: str = SLAStatus.COMPLIANT.value
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the freshness check.
        """
        return asdict(self)


@dataclass
class FreshnessSummary:
    """Summary of a stale dataset identified by freshness checks.

    Attributes:
        dataset_id: Identifier of the stale dataset.
        age_hours: Current age in hours since last refresh.
        freshness_score: Current freshness score.
        freshness_level: Current freshness level classification.
        sla_status: Current SLA compliance status.
        last_checked_at: ISO timestamp of the most recent check.
    """

    dataset_id: str = ""
    age_hours: float = 0.0
    freshness_score: float = 0.0
    freshness_level: str = FreshnessLevel.STALE.value
    sla_status: str = SLAStatus.BREACHED.value
    last_checked_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the freshness summary.
        """
        return asdict(self)


# ---------------------------------------------------------------------------
# Metrics helpers (safe when prometheus_client is absent)
# ---------------------------------------------------------------------------

try:
    from greenlang.data_freshness_monitor.metrics import (
        record_check as _record_check_raw,
        observe_freshness_score as _observe_freshness_score_raw,
        observe_data_age as _observe_data_age_raw,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_check_raw = None  # type: ignore[assignment]
    _observe_freshness_score_raw = None  # type: ignore[assignment]
    _observe_data_age_raw = None  # type: ignore[assignment]


def _safe_record_check(status: str, count: int = 1) -> None:
    """Safely record a freshness check metric.

    Args:
        status: SLA status category (compliant, warning, breached).
        count: Number of checks to record.
    """
    if _METRICS_AVAILABLE and _record_check_raw is not None:
        try:
            _record_check_raw(status, count)
        except Exception:
            pass


def _safe_observe_freshness_score(score: float) -> None:
    """Safely observe a freshness score metric.

    Args:
        score: Freshness score between 0.0 and 1.0.
    """
    if _METRICS_AVAILABLE and _observe_freshness_score_raw is not None:
        try:
            _observe_freshness_score_raw(score)
        except Exception:
            pass


def _safe_observe_data_age(age_hours: float) -> None:
    """Safely observe a data age metric.

    Args:
        age_hours: Age of the dataset in hours.
    """
    if _METRICS_AVAILABLE and _observe_data_age_raw is not None:
        try:
            _observe_data_age_raw(age_hours)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Provenance helper (safe when provenance module is absent)
# ---------------------------------------------------------------------------

try:
    from greenlang.data_freshness_monitor.provenance import (
        ProvenanceTracker,
    )
    _PROVENANCE_MODULE_AVAILABLE = True
except ImportError:
    _PROVENANCE_MODULE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _build_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Sorts dictionary keys and serializes to JSON for reproducibility.

    Args:
        data: Data to hash (dict, list, str, numeric, or other).

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _generate_check_id() -> str:
    """Generate a unique freshness check ID.

    Returns:
        Check ID with FC- prefix and 12-character hex suffix.
    """
    return f"FC-{uuid4().hex[:12]}"


# ===========================================================================
# FreshnessCheckerEngine
# ===========================================================================


class FreshnessCheckerEngine:
    """Dataset freshness check engine with piecewise-linear scoring.

    Performs freshness checks on registered datasets by computing age
    in hours since last refresh, applying a 5-tier piecewise-linear
    freshness scoring algorithm, classifying the freshness level, and
    evaluating SLA compliance status. Supports batch checks, dataset
    group aggregation with optional weighting, check history tracking,
    stale dataset identification, SLA compliance summaries, and
    freshness heatmap generation.

    All arithmetic is deterministic Python (zero-hallucination).
    Every freshness check produces a SHA-256 provenance hash for
    audit trail tracking.

    Attributes:
        _config: DataFreshnessMonitorConfig for threshold boundaries.
        _check_history: Per-dataset ordered list of FreshnessCheck results.
        _check_count: Running count of total freshness checks performed.
        _provenance: SHA-256 provenance tracker for audit trails.
        _lock: Thread-safety lock for concurrent access.

    Example:
        >>> engine = FreshnessCheckerEngine()
        >>> from datetime import datetime, timezone, timedelta
        >>> now = datetime.now(timezone.utc)
        >>> check = engine.check_freshness(
        ...     "ds-001", now - timedelta(hours=0.5), 24.0, 72.0
        ... )
        >>> assert check.freshness_level == "excellent"
        >>> assert check.sla_status == "compliant"
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize FreshnessCheckerEngine.

        Args:
            config: Optional DataFreshnessMonitorConfig instance.
                If None, imports and calls get_config() from the
                config module.
        """
        if config is None:
            from greenlang.data_freshness_monitor.config import get_config
            self._config = get_config()
        else:
            self._config = config

        self._check_history: Dict[str, List[FreshnessCheck]] = {}
        self._check_count: int = 0

        if _PROVENANCE_MODULE_AVAILABLE:
            self._provenance: Any = ProvenanceTracker()
        else:
            self._provenance = None

        self._lock = threading.Lock()

        logger.info("FreshnessCheckerEngine initialized (data freshness monitor)")

    # ------------------------------------------------------------------
    # 1. check_freshness
    # ------------------------------------------------------------------

    def check_freshness(
        self,
        dataset_id: str,
        last_refreshed_at: datetime,
        sla_warning_hours: float,
        sla_critical_hours: float,
    ) -> FreshnessCheck:
        """Perform a freshness check on a single dataset.

        Computes age in hours, freshness score (piecewise-linear 5-tier),
        freshness level classification, and SLA compliance status.
        Records the check in history and emits metrics and provenance.

        Args:
            dataset_id: Unique identifier for the dataset.
            last_refreshed_at: UTC datetime of the dataset's last refresh.
            sla_warning_hours: Warning threshold in hours for SLA.
            sla_critical_hours: Critical threshold in hours for SLA.

        Returns:
            FreshnessCheck with computed age, score, level, and SLA status.

        Raises:
            ValueError: If dataset_id is empty or SLA thresholds are
                invalid (warning >= critical or non-positive).

        Example:
            >>> engine = FreshnessCheckerEngine()
            >>> from datetime import datetime, timezone, timedelta
            >>> now = datetime.now(timezone.utc)
            >>> check = engine.check_freshness(
            ...     "ds-001", now - timedelta(hours=3), 24.0, 72.0
            ... )
            >>> assert check.freshness_score > 0.8
        """
        start_time = time.monotonic()

        # Validate inputs
        if not dataset_id or not dataset_id.strip():
            raise ValueError("dataset_id must not be empty")
        if sla_warning_hours <= 0.0:
            raise ValueError("sla_warning_hours must be > 0.0")
        if sla_critical_hours <= 0.0:
            raise ValueError("sla_critical_hours must be > 0.0")
        if sla_warning_hours >= sla_critical_hours:
            raise ValueError(
                "sla_warning_hours must be < sla_critical_hours"
            )

        current_time = _utcnow()
        check_id = _generate_check_id()

        # Compute age
        age_hours = self.compute_age_hours(last_refreshed_at, current_time)

        # Compute freshness score (piecewise-linear 5-tier)
        freshness_score = self.compute_freshness_score(age_hours)

        # Classify freshness level
        freshness_level = self.classify_freshness_level(age_hours)

        # Evaluate SLA status
        sla_status = self.evaluate_sla_status(
            age_hours, sla_warning_hours, sla_critical_hours,
        )

        # Compute provenance hash
        provenance_hash = self._compute_provenance(
            operation="check_freshness",
            input_data={
                "dataset_id": dataset_id,
                "last_refreshed_at": last_refreshed_at.isoformat(),
                "sla_warning_hours": sla_warning_hours,
                "sla_critical_hours": sla_critical_hours,
            },
            output_data={
                "age_hours": age_hours,
                "freshness_score": freshness_score,
                "freshness_level": freshness_level.value,
                "sla_status": sla_status.value,
            },
        )

        # Build result
        check = FreshnessCheck(
            check_id=check_id,
            dataset_id=dataset_id,
            last_refreshed_at=last_refreshed_at.isoformat(),
            checked_at=current_time.isoformat(),
            age_hours=round(age_hours, 6),
            freshness_score=round(freshness_score, 6),
            freshness_level=freshness_level.value,
            sla_warning_hours=sla_warning_hours,
            sla_critical_hours=sla_critical_hours,
            sla_status=sla_status.value,
            provenance_hash=provenance_hash,
        )

        # Record in history and emit metrics
        self._store_check(check)
        self._record_metrics_and_provenance(
            "check_freshness", start_time, check,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "check_freshness completed: dataset=%s age=%.2fh score=%.4f "
            "level=%s sla=%s in %.3fms",
            dataset_id, age_hours, freshness_score,
            freshness_level.value, sla_status.value, elapsed_ms,
        )
        return check

    # ------------------------------------------------------------------
    # 2. batch_check
    # ------------------------------------------------------------------

    def batch_check(
        self,
        checks: List[Dict[str, Any]],
    ) -> List[FreshnessCheck]:
        """Perform freshness checks on multiple datasets in batch.

        Each entry in the checks list must contain the keys:
        ``dataset_id``, ``last_refreshed_at`` (datetime),
        ``sla_warning_hours`` (float), ``sla_critical_hours`` (float).

        Args:
            checks: List of dictionaries with check parameters.

        Returns:
            List of FreshnessCheck results in the same order as input.

        Raises:
            ValueError: If any individual check fails validation.

        Example:
            >>> engine = FreshnessCheckerEngine()
            >>> from datetime import datetime, timezone, timedelta
            >>> now = datetime.now(timezone.utc)
            >>> results = engine.batch_check([
            ...     {
            ...         "dataset_id": "ds-001",
            ...         "last_refreshed_at": now - timedelta(hours=1),
            ...         "sla_warning_hours": 24.0,
            ...         "sla_critical_hours": 72.0,
            ...     },
            ... ])
            >>> assert len(results) == 1
        """
        start_time = time.monotonic()
        results: List[FreshnessCheck] = []

        for entry in checks:
            check = self.check_freshness(
                dataset_id=entry["dataset_id"],
                last_refreshed_at=entry["last_refreshed_at"],
                sla_warning_hours=entry["sla_warning_hours"],
                sla_critical_hours=entry["sla_critical_hours"],
            )
            results.append(check)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "batch_check completed: %d datasets in %.3fms",
            len(results), elapsed_ms,
        )
        return results

    # ------------------------------------------------------------------
    # 3. compute_age_hours
    # ------------------------------------------------------------------

    def compute_age_hours(
        self,
        last_refreshed_at: datetime,
        current_time: Optional[datetime] = None,
    ) -> float:
        """Compute the age of a dataset in hours since last refresh.

        If the last_refreshed_at timestamp is in the future relative
        to current_time, age is clamped to 0.0.

        Args:
            last_refreshed_at: UTC datetime of the last refresh.
            current_time: Optional reference time. Defaults to UTC now.

        Returns:
            Age in hours as a non-negative float.

        Example:
            >>> engine = FreshnessCheckerEngine()
            >>> from datetime import datetime, timezone, timedelta
            >>> now = datetime.now(timezone.utc)
            >>> age = engine.compute_age_hours(now - timedelta(hours=5), now)
            >>> assert 4.9 < age < 5.1
        """
        if current_time is None:
            current_time = _utcnow()

        # Ensure both are timezone-aware for subtraction
        if last_refreshed_at.tzinfo is None:
            last_refreshed_at = last_refreshed_at.replace(tzinfo=timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        delta = current_time - last_refreshed_at
        total_seconds = delta.total_seconds()

        # Clamp to non-negative (future timestamps yield 0)
        if total_seconds < 0.0:
            return 0.0

        return total_seconds / 3600.0

    # ------------------------------------------------------------------
    # 4. compute_freshness_score
    # ------------------------------------------------------------------

    def compute_freshness_score(self, age_hours: float) -> float:
        """Compute a freshness score using piecewise-linear 5-tier algorithm.

        Tier boundaries are sourced from config:
          - age <= excellent_hours -> 1.0
          - excellent < age <= good_hours -> linear 1.0 -> 0.85
          - good < age <= fair_hours -> linear 0.85 -> 0.70
          - fair < age <= poor_hours -> linear 0.70 -> 0.50
          - age > poor_hours -> max(0.0, 0.50 - (age-poor)/(poor*2)*0.50)

        Args:
            age_hours: Age of the dataset in hours.

        Returns:
            Freshness score between 0.0 and 1.0.

        Example:
            >>> engine = FreshnessCheckerEngine()
            >>> score = engine.compute_freshness_score(0.5)
            >>> assert score == 1.0
        """
        excellent = self._config.freshness_excellent_hours
        good = self._config.freshness_good_hours
        fair = self._config.freshness_fair_hours
        poor = self._config.freshness_poor_hours

        # Clamp negative age to 0
        if age_hours < 0.0:
            age_hours = 0.0

        # Tier 1: EXCELLENT
        if age_hours <= excellent:
            return 1.0

        # Tier 2: GOOD (linear 1.0 -> 0.85)
        if age_hours <= good:
            return self._linear_interpolate(
                age_hours, excellent, good, 1.0, 0.85,
            )

        # Tier 3: FAIR (linear 0.85 -> 0.70)
        if age_hours <= fair:
            return self._linear_interpolate(
                age_hours, good, fair, 0.85, 0.70,
            )

        # Tier 4: POOR (linear 0.70 -> 0.50)
        if age_hours <= poor:
            return self._linear_interpolate(
                age_hours, fair, poor, 0.70, 0.50,
            )

        # Tier 5: STALE (decays from 0.50 toward 0.0)
        decay = (age_hours - poor) / (poor * 2.0) * 0.50
        score = 0.50 - decay
        return max(0.0, score)

    # ------------------------------------------------------------------
    # 5. classify_freshness_level
    # ------------------------------------------------------------------

    def classify_freshness_level(self, age_hours: float) -> FreshnessLevel:
        """Classify a dataset's freshness into one of five tiers.

        Uses config boundaries to determine the tier:
          - age <= excellent_hours -> EXCELLENT
          - age <= good_hours -> GOOD
          - age <= fair_hours -> FAIR
          - age <= poor_hours -> POOR
          - age > poor_hours -> STALE

        Args:
            age_hours: Age of the dataset in hours.

        Returns:
            FreshnessLevel enumeration value.

        Example:
            >>> engine = FreshnessCheckerEngine()
            >>> level = engine.classify_freshness_level(0.5)
            >>> assert level == FreshnessLevel.EXCELLENT
        """
        if age_hours < 0.0:
            age_hours = 0.0

        if age_hours <= self._config.freshness_excellent_hours:
            return FreshnessLevel.EXCELLENT
        if age_hours <= self._config.freshness_good_hours:
            return FreshnessLevel.GOOD
        if age_hours <= self._config.freshness_fair_hours:
            return FreshnessLevel.FAIR
        if age_hours <= self._config.freshness_poor_hours:
            return FreshnessLevel.POOR
        return FreshnessLevel.STALE

    # ------------------------------------------------------------------
    # 6. evaluate_sla_status
    # ------------------------------------------------------------------

    def evaluate_sla_status(
        self,
        age_hours: float,
        warning_hours: float,
        critical_hours: float,
    ) -> SLAStatus:
        """Evaluate SLA compliance status for a dataset's age.

        Args:
            age_hours: Age of the dataset in hours.
            warning_hours: Warning threshold in hours.
            critical_hours: Critical threshold in hours.

        Returns:
            SLAStatus enumeration value.

        Example:
            >>> engine = FreshnessCheckerEngine()
            >>> status = engine.evaluate_sla_status(12.0, 24.0, 72.0)
            >>> assert status == SLAStatus.COMPLIANT
        """
        if age_hours < 0.0:
            age_hours = 0.0

        if age_hours >= critical_hours:
            return SLAStatus.BREACHED
        if age_hours >= warning_hours:
            return SLAStatus.WARNING
        return SLAStatus.COMPLIANT

    # ------------------------------------------------------------------
    # 7. check_dataset_group
    # ------------------------------------------------------------------

    def check_dataset_group(
        self,
        group_datasets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Check freshness for a group of related datasets.

        Performs individual checks on each dataset in the group and
        produces an aggregated group summary including the weighted
        (or equal-weight) group freshness score, worst-case SLA
        status, and per-dataset results.

        Each entry in group_datasets must contain:
          - ``dataset_id`` (str)
          - ``last_refreshed_at`` (datetime)
          - ``sla_warning_hours`` (float)
          - ``sla_critical_hours`` (float)
          - ``weight`` (float, optional, defaults to 1.0)

        Args:
            group_datasets: List of dataset configurations.

        Returns:
            Dictionary with keys:
              - ``checks``: List of FreshnessCheck results
              - ``group_freshness_score``: Aggregated weighted score
              - ``worst_sla_status``: Worst SLA status in the group
              - ``total_datasets``: Number of datasets checked
              - ``stale_count``: Number of stale datasets
              - ``provenance_hash``: SHA-256 hash of the group result

        Example:
            >>> engine = FreshnessCheckerEngine()
            >>> from datetime import datetime, timezone, timedelta
            >>> now = datetime.now(timezone.utc)
            >>> result = engine.check_dataset_group([
            ...     {
            ...         "dataset_id": "ds-001",
            ...         "last_refreshed_at": now - timedelta(hours=0.5),
            ...         "sla_warning_hours": 24.0,
            ...         "sla_critical_hours": 72.0,
            ...     },
            ... ])
            >>> assert result["group_freshness_score"] > 0.9
        """
        start_time = time.monotonic()
        checks: List[FreshnessCheck] = []
        weights: Dict[str, float] = {}

        for entry in group_datasets:
            check = self.check_freshness(
                dataset_id=entry["dataset_id"],
                last_refreshed_at=entry["last_refreshed_at"],
                sla_warning_hours=entry["sla_warning_hours"],
                sla_critical_hours=entry["sla_critical_hours"],
            )
            checks.append(check)
            weights[entry["dataset_id"]] = entry.get("weight", 1.0)

        # Compute weighted group freshness score
        group_score = self.compute_group_freshness_score(checks, weights)

        # Determine worst SLA status
        worst_sla = self._worst_sla_status(checks)

        # Count stale datasets
        stale_count = sum(
            1 for c in checks
            if c.freshness_level == FreshnessLevel.STALE.value
        )

        # Provenance hash for the group result
        provenance_hash = self._compute_provenance(
            operation="check_dataset_group",
            input_data={
                "dataset_ids": [c.dataset_id for c in checks],
            },
            output_data={
                "group_freshness_score": group_score,
                "worst_sla_status": worst_sla,
                "stale_count": stale_count,
            },
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "check_dataset_group completed: %d datasets, "
            "group_score=%.4f, worst_sla=%s, stale=%d in %.3fms",
            len(checks), group_score, worst_sla, stale_count, elapsed_ms,
        )

        return {
            "checks": checks,
            "group_freshness_score": round(group_score, 6),
            "worst_sla_status": worst_sla,
            "total_datasets": len(checks),
            "stale_count": stale_count,
            "provenance_hash": provenance_hash,
        }

    # ------------------------------------------------------------------
    # 8. compute_group_freshness_score
    # ------------------------------------------------------------------

    def compute_group_freshness_score(
        self,
        checks: List[FreshnessCheck],
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute a weighted average freshness score for a group.

        If weights are not provided, all datasets receive equal weight.
        Datasets with weight <= 0 are excluded from the average.

        Args:
            checks: List of FreshnessCheck results.
            weights: Optional mapping of dataset_id to weight. Defaults
                to equal weight (1.0) for all datasets.

        Returns:
            Weighted average freshness score between 0.0 and 1.0.
            Returns 0.0 if no checks are provided or total weight is 0.

        Example:
            >>> engine = FreshnessCheckerEngine()
            >>> # Assume checks is a list of FreshnessCheck objects
            >>> score = engine.compute_group_freshness_score([], None)
            >>> assert score == 0.0
        """
        if not checks:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for check in checks:
            w = 1.0
            if weights is not None and check.dataset_id in weights:
                w = weights[check.dataset_id]
            if w <= 0.0:
                continue
            weighted_sum += check.freshness_score * w
            total_weight += w

        if total_weight <= 0.0:
            return 0.0

        return weighted_sum / total_weight

    # ------------------------------------------------------------------
    # 9. get_check_history
    # ------------------------------------------------------------------

    def get_check_history(
        self,
        dataset_id: str,
        limit: int = 100,
    ) -> List[FreshnessCheck]:
        """Retrieve check history for a specific dataset.

        Returns the most recent checks in reverse chronological order,
        limited by the ``limit`` parameter.

        Args:
            dataset_id: Identifier of the dataset.
            limit: Maximum number of checks to return. Defaults to 100.

        Returns:
            List of FreshnessCheck results, most recent first.

        Example:
            >>> engine = FreshnessCheckerEngine()
            >>> history = engine.get_check_history("ds-001", limit=10)
            >>> assert isinstance(history, list)
        """
        with self._lock:
            history = self._check_history.get(dataset_id, [])
            # Return most recent first, capped at limit
            return list(reversed(history[-limit:]))

    # ------------------------------------------------------------------
    # 10. get_latest_check
    # ------------------------------------------------------------------

    def get_latest_check(
        self,
        dataset_id: str,
    ) -> Optional[FreshnessCheck]:
        """Retrieve the most recent freshness check for a dataset.

        Args:
            dataset_id: Identifier of the dataset.

        Returns:
            Most recent FreshnessCheck, or None if no checks exist.

        Example:
            >>> engine = FreshnessCheckerEngine()
            >>> latest = engine.get_latest_check("ds-nonexistent")
            >>> assert latest is None
        """
        with self._lock:
            history = self._check_history.get(dataset_id)
            if history and len(history) > 0:
                return history[-1]
            return None

    # ------------------------------------------------------------------
    # 11. get_stale_datasets
    # ------------------------------------------------------------------

    def get_stale_datasets(
        self,
        threshold_hours: float,
    ) -> List[FreshnessSummary]:
        """Identify all datasets whose latest check age exceeds a threshold.

        Scans the check history for all tracked datasets and returns
        summaries for those whose most recent age_hours exceeds the
        given threshold.

        Args:
            threshold_hours: Age threshold in hours. Datasets older
                than this are considered stale.

        Returns:
            List of FreshnessSummary for stale datasets, sorted by
            age_hours descending (stalest first).

        Example:
            >>> engine = FreshnessCheckerEngine()
            >>> stale = engine.get_stale_datasets(48.0)
            >>> assert isinstance(stale, list)
        """
        stale: List[FreshnessSummary] = []

        with self._lock:
            for dataset_id, history in self._check_history.items():
                if not history:
                    continue
                latest = history[-1]
                if latest.age_hours > threshold_hours:
                    summary = FreshnessSummary(
                        dataset_id=dataset_id,
                        age_hours=latest.age_hours,
                        freshness_score=latest.freshness_score,
                        freshness_level=latest.freshness_level,
                        sla_status=latest.sla_status,
                        last_checked_at=latest.checked_at,
                    )
                    stale.append(summary)

        # Sort by age descending (stalest first)
        stale.sort(key=lambda s: s.age_hours, reverse=True)
        return stale

    # ------------------------------------------------------------------
    # 12. get_sla_compliance_summary
    # ------------------------------------------------------------------

    def get_sla_compliance_summary(self) -> Dict[str, Any]:
        """Compute SLA compliance summary across all tracked datasets.

        Examines the latest check for every tracked dataset and
        produces aggregate counts and percentages by SLA status.

        Returns:
            Dictionary with keys:
              - ``total``: Total number of tracked datasets
              - ``compliant``: Count of compliant datasets
              - ``warning``: Count of datasets in warning state
              - ``breached``: Count of datasets with breached SLAs
              - ``compliant_pct``: Percentage compliant
              - ``warning_pct``: Percentage in warning
              - ``breached_pct``: Percentage breached
              - ``provenance_hash``: SHA-256 hash of the summary

        Example:
            >>> engine = FreshnessCheckerEngine()
            >>> summary = engine.get_sla_compliance_summary()
            >>> assert summary["total"] == 0
        """
        compliant = 0
        warning = 0
        breached = 0

        with self._lock:
            for history in self._check_history.values():
                if not history:
                    continue
                latest = history[-1]
                if latest.sla_status == SLAStatus.COMPLIANT.value:
                    compliant += 1
                elif latest.sla_status == SLAStatus.WARNING.value:
                    warning += 1
                elif latest.sla_status == SLAStatus.BREACHED.value:
                    breached += 1

        total = compliant + warning + breached

        compliant_pct = (compliant / total * 100.0) if total > 0 else 0.0
        warning_pct = (warning / total * 100.0) if total > 0 else 0.0
        breached_pct = (breached / total * 100.0) if total > 0 else 0.0

        provenance_hash = self._compute_provenance(
            operation="get_sla_compliance_summary",
            input_data={"total": total},
            output_data={
                "compliant": compliant,
                "warning": warning,
                "breached": breached,
            },
        )

        return {
            "total": total,
            "compliant": compliant,
            "warning": warning,
            "breached": breached,
            "compliant_pct": round(compliant_pct, 2),
            "warning_pct": round(warning_pct, 2),
            "breached_pct": round(breached_pct, 2),
            "provenance_hash": provenance_hash,
        }

    # ------------------------------------------------------------------
    # 13. compute_freshness_heatmap
    # ------------------------------------------------------------------

    def compute_freshness_heatmap(
        self,
        dataset_ids: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Generate a freshness heatmap for the specified datasets.

        For each dataset_id, returns the latest freshness score,
        freshness level, and age in hours. Datasets without any
        check history are included with None values.

        Args:
            dataset_ids: List of dataset identifiers to include.

        Returns:
            Dictionary keyed by dataset_id, each value being a dict
            with keys ``score``, ``level``, and ``age_hours``.

        Example:
            >>> engine = FreshnessCheckerEngine()
            >>> heatmap = engine.compute_freshness_heatmap(["ds-001"])
            >>> assert "ds-001" in heatmap
        """
        heatmap: Dict[str, Dict[str, Any]] = {}

        with self._lock:
            for dataset_id in dataset_ids:
                history = self._check_history.get(dataset_id)
                if history and len(history) > 0:
                    latest = history[-1]
                    heatmap[dataset_id] = {
                        "score": latest.freshness_score,
                        "level": latest.freshness_level,
                        "age_hours": latest.age_hours,
                    }
                else:
                    heatmap[dataset_id] = {
                        "score": None,
                        "level": None,
                        "age_hours": None,
                    }

        return heatmap

    # ------------------------------------------------------------------
    # 14. get_check_count
    # ------------------------------------------------------------------

    def get_check_count(self) -> int:
        """Return the total number of freshness checks performed.

        Returns:
            Non-negative integer count of all checks.
        """
        return self._check_count

    # ------------------------------------------------------------------
    # 15. get_statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine statistics summary.

        Provides aggregate statistics about all freshness checks
        performed by this engine instance, including total checks,
        tracked datasets, average freshness score, and SLA status
        distribution.

        Returns:
            Dictionary with keys:
              - ``total_checks``: Total freshness checks performed
              - ``tracked_datasets``: Number of unique datasets tracked
              - ``average_freshness_score``: Mean score across latest checks
              - ``freshness_level_distribution``: Count by FreshnessLevel
              - ``sla_status_distribution``: Count by SLAStatus
              - ``provenance_chain_length``: Length of provenance chain

        Example:
            >>> engine = FreshnessCheckerEngine()
            >>> stats = engine.get_statistics()
            >>> assert stats["total_checks"] == 0
        """
        level_dist: Dict[str, int] = {
            level.value: 0 for level in FreshnessLevel
        }
        sla_dist: Dict[str, int] = {
            status.value: 0 for status in SLAStatus
        }
        scores: List[float] = []

        with self._lock:
            for history in self._check_history.values():
                if not history:
                    continue
                latest = history[-1]
                scores.append(latest.freshness_score)
                if latest.freshness_level in level_dist:
                    level_dist[latest.freshness_level] += 1
                if latest.sla_status in sla_dist:
                    sla_dist[latest.sla_status] += 1

        avg_score = sum(scores) / len(scores) if scores else 0.0

        prov_length = 0
        if self._provenance is not None:
            try:
                prov_length = self._provenance.get_chain_length()
            except Exception:
                prov_length = 0

        return {
            "total_checks": self._check_count,
            "tracked_datasets": len(self._check_history),
            "average_freshness_score": round(avg_score, 6),
            "freshness_level_distribution": level_dist,
            "sla_status_distribution": sla_dist,
            "provenance_chain_length": prov_length,
        }

    # ------------------------------------------------------------------
    # 16. reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all internal state, clearing history and counters.

        Useful for testing and re-initialization. Reinitializes the
        provenance tracker if available.

        Example:
            >>> engine = FreshnessCheckerEngine()
            >>> engine.reset()
            >>> assert engine.get_check_count() == 0
        """
        with self._lock:
            self._check_history.clear()
            self._check_count = 0

        if _PROVENANCE_MODULE_AVAILABLE:
            self._provenance = ProvenanceTracker()
        else:
            self._provenance = None

        logger.info("FreshnessCheckerEngine reset: all state cleared")

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _store_check(self, check: FreshnessCheck) -> None:
        """Store a freshness check in the per-dataset history.

        Thread-safe: acquires _lock before mutating shared state.

        Args:
            check: FreshnessCheck to store.
        """
        with self._lock:
            if check.dataset_id not in self._check_history:
                self._check_history[check.dataset_id] = []
            self._check_history[check.dataset_id].append(check)
            self._check_count += 1

    def _linear_interpolate(
        self,
        x: float,
        x_start: float,
        x_end: float,
        y_start: float,
        y_end: float,
    ) -> float:
        """Perform linear interpolation between two points.

        Computes y for a given x using the linear equation between
        (x_start, y_start) and (x_end, y_end).

        Args:
            x: Input value.
            x_start: Start of the x range.
            x_end: End of the x range.
            y_start: Start of the y range (score at x_start).
            y_end: End of the y range (score at x_end).

        Returns:
            Interpolated y value.
        """
        if x_end == x_start:
            return y_start
        fraction = (x - x_start) / (x_end - x_start)
        return y_start + (y_end - y_start) * fraction

    def _worst_sla_status(
        self,
        checks: List[FreshnessCheck],
    ) -> str:
        """Determine the worst SLA status in a list of checks.

        Priority: BREACHED > WARNING > COMPLIANT.

        Args:
            checks: List of FreshnessCheck results.

        Returns:
            Worst SLA status as a string value.
        """
        if not checks:
            return SLAStatus.COMPLIANT.value

        status_priority = {
            SLAStatus.COMPLIANT.value: 0,
            SLAStatus.WARNING.value: 1,
            SLAStatus.BREACHED.value: 2,
        }

        worst = SLAStatus.COMPLIANT.value
        worst_priority = 0

        for check in checks:
            priority = status_priority.get(check.sla_status, 0)
            if priority > worst_priority:
                worst = check.sla_status
                worst_priority = priority

        return worst

    def _compute_provenance(
        self,
        operation: str,
        input_data: Any,
        output_data: Any,
    ) -> str:
        """Compute a SHA-256 provenance hash for an operation.

        Combines the operation name, input data, output data, and
        current UTC timestamp into a deterministic hash.

        Args:
            operation: Name of the operation performed.
            input_data: Input data for the operation.
            output_data: Output data from the operation.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        timestamp = _utcnow().isoformat()
        payload = {
            "operation": operation,
            "input": input_data,
            "output": output_data,
            "timestamp": timestamp,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _record_metrics_and_provenance(
        self,
        operation: str,
        start_time: float,
        check: FreshnessCheck,
    ) -> None:
        """Record metric observations and provenance for a freshness check.

        Emits Prometheus metrics (if available) and appends a
        provenance chain entry (if available) for the given check.

        Args:
            operation: Name of the freshness check operation.
            start_time: Monotonic start time from time.monotonic().
            check: The FreshnessCheck result.
        """
        # Emit Prometheus metrics
        _safe_record_check(check.sla_status, 1)
        _safe_observe_freshness_score(check.freshness_score)
        _safe_observe_data_age(check.age_hours)

        # Record provenance chain entry
        if self._provenance is not None:
            try:
                input_hash = _build_hash({
                    "dataset_id": check.dataset_id,
                    "last_refreshed_at": check.last_refreshed_at,
                })
                output_hash = _build_hash({
                    "age_hours": check.age_hours,
                    "freshness_score": check.freshness_score,
                    "sla_status": check.sla_status,
                })
                self._provenance.add_to_chain(
                    operation=operation,
                    input_hash=input_hash,
                    output_hash=output_hash,
                    metadata={
                        "dataset_id": check.dataset_id,
                        "check_id": check.check_id,
                    },
                )
            except Exception:
                logger.debug(
                    "Provenance recording skipped for %s", operation,
                    exc_info=True,
                )
