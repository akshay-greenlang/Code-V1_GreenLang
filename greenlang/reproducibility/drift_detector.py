# -*- coding: utf-8 -*-
"""
Drift Detection Engine - AGENT-FOUND-008: Reproducibility Agent

Detects drift between baseline and current execution results using
configurable soft and hard thresholds. Manages named baselines for
persistent drift monitoring.

Zero-Hallucination Guarantees:
    - All drift calculations use deterministic arithmetic
    - Severity classification uses fixed threshold comparison
    - No probabilistic or ML-based drift detection
    - Complete provenance for all baselines

Example:
    >>> from greenlang.reproducibility.drift_detector import DriftDetector
    >>> from greenlang.reproducibility.artifact_hasher import ArtifactHasher
    >>> from greenlang.reproducibility.config import ReproducibilityConfig
    >>> config = ReproducibilityConfig()
    >>> hasher = ArtifactHasher(config)
    >>> detector = DriftDetector(config, hasher)
    >>> baseline = detector.create_baseline("v1", "Initial", {"emissions": 100.0})
    >>> drift = detector.compare_to_baseline(baseline.baseline_id, {"emissions": 101.0})

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-008 Reproducibility Agent
Status: Production Ready
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.reproducibility.config import ReproducibilityConfig
from greenlang.reproducibility.artifact_hasher import ArtifactHasher
from greenlang.reproducibility.models import (
    DriftSeverity,
    DriftDetection,
    DriftBaseline,
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_DRIFT_SOFT_THRESHOLD,
    DEFAULT_DRIFT_HARD_THRESHOLD,
)
from greenlang.reproducibility.metrics import record_drift

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


class DriftDetector:
    """Drift detection engine.

    Compares current execution results against stored baselines to detect
    drift, classify severity, and determine acceptability.

    Attributes:
        _config: Reproducibility configuration.
        _hasher: ArtifactHasher for deterministic hashing.
        _baselines: In-memory store of drift baselines.

    Example:
        >>> detector = DriftDetector(config, hasher)
        >>> bl = detector.create_baseline("v1", "Init", {"co2": 100.0})
        >>> result = detector.detect_drift({"co2": 100.0}, {"co2": 101.0})
        >>> print(result.severity)
    """

    def __init__(
        self,
        config: ReproducibilityConfig,
        hasher: ArtifactHasher,
    ) -> None:
        """Initialize DriftDetector.

        Args:
            config: Reproducibility configuration instance.
            hasher: ArtifactHasher instance for hash computation.
        """
        self._config = config
        self._hasher = hasher
        self._baselines: Dict[str, DriftBaseline] = {}
        logger.info(
            "DriftDetector initialized: soft=%s, hard=%s",
            config.drift_soft_threshold, config.drift_hard_threshold,
        )

    def detect_drift(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any],
        soft_threshold: float = DEFAULT_DRIFT_SOFT_THRESHOLD,
        hard_threshold: float = DEFAULT_DRIFT_HARD_THRESHOLD,
        tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    ) -> DriftDetection:
        """Detect drift between baseline and current result data.

        Computes hashes, recursively compares fields, calculates
        maximum drift percentage, and classifies severity.

        Args:
            baseline: Baseline result data dictionary.
            current: Current result data dictionary.
            soft_threshold: Soft threshold for drift warning (fraction).
            hard_threshold: Hard threshold for drift failure (fraction).
            tolerance: Absolute tolerance for numeric comparisons.

        Returns:
            DriftDetection result with severity and field-level details.
        """
        baseline_hash = self._hasher.compute_hash(baseline)
        current_hash = self._hasher.compute_hash(current)

        drifted_fields: List[str] = []
        drift_details: Dict[str, Dict[str, Any]] = {}

        self._compare_for_drift(
            baseline, current, "", drifted_fields, drift_details, tolerance,
        )

        # Calculate max drift percentage
        max_drift = 0.0
        for details in drift_details.values():
            drift_pct = details.get("drift_percentage", 0.0)
            max_drift = max(max_drift, abs(drift_pct))

        # Classify severity
        severity = self._classify_severity(
            max_drift, soft_threshold, hard_threshold, baseline_hash, current_hash,
        )
        is_acceptable = severity in (DriftSeverity.NONE, DriftSeverity.MINOR)

        result = DriftDetection(
            baseline_hash=baseline_hash,
            current_hash=current_hash,
            severity=severity,
            drift_percentage=max_drift * 100,
            drifted_fields=drifted_fields,
            drift_details=drift_details,
            is_acceptable=is_acceptable,
        )

        # Record metric
        record_drift(severity.value, max_drift * 100)

        logger.info(
            "Drift detection: severity=%s, drift=%.4f%%, fields=%d",
            severity.value, max_drift * 100, len(drifted_fields),
        )

        return result

    def _calculate_field_drift(
        self,
        baseline_val: Any,
        current_val: Any,
        tolerance: float,
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate drift for a single field value.

        Args:
            baseline_val: Baseline value.
            current_val: Current value.
            tolerance: Absolute tolerance.

        Returns:
            Tuple of (drift_percentage, detail_dict).
        """
        if isinstance(baseline_val, (int, float)) and isinstance(current_val, (int, float)):
            abs_diff = abs(current_val - baseline_val)
            if abs_diff <= tolerance:
                return 0.0, {}

            drift_pct = 0.0
            if baseline_val != 0:
                drift_pct = abs_diff / abs(baseline_val)

            return drift_pct, {
                "baseline": baseline_val,
                "current": current_val,
                "difference": current_val - baseline_val,
                "drift_percentage": drift_pct,
            }

        # Non-numeric: binary match
        if baseline_val != current_val:
            return 1.0, {
                "baseline": str(baseline_val),
                "current": str(current_val),
                "type_mismatch": type(baseline_val).__name__ != type(current_val).__name__,
            }

        return 0.0, {}

    def _classify_severity(
        self,
        max_drift: float,
        soft_threshold: float,
        hard_threshold: float,
        baseline_hash: str,
        current_hash: str,
    ) -> DriftSeverity:
        """Classify drift severity based on thresholds.

        Args:
            max_drift: Maximum drift percentage (as fraction).
            soft_threshold: Soft threshold for warnings.
            hard_threshold: Hard threshold for failures.
            baseline_hash: Hash of baseline data.
            current_hash: Hash of current data.

        Returns:
            DriftSeverity classification.
        """
        if max_drift == 0 and baseline_hash == current_hash:
            return DriftSeverity.NONE

        if max_drift <= soft_threshold:
            return DriftSeverity.MINOR

        if max_drift <= hard_threshold:
            return DriftSeverity.MODERATE

        return DriftSeverity.CRITICAL

    def _compare_for_drift(
        self,
        baseline: Any,
        current: Any,
        path: str,
        drifted_fields: List[str],
        drift_details: Dict[str, Dict[str, Any]],
        tolerance: float,
    ) -> None:
        """Recursively compare baseline and current data for drift.

        Args:
            baseline: Baseline value.
            current: Current value.
            path: Current field path for reporting.
            drifted_fields: Accumulator for drifted field names.
            drift_details: Accumulator for drift detail dicts.
            tolerance: Absolute tolerance.
        """
        if isinstance(baseline, dict) and isinstance(current, dict):
            all_keys = set(baseline.keys()) | set(current.keys())
            for key in sorted(all_keys):
                new_path = f"{path}.{key}" if path else key
                b_val = baseline.get(key)
                c_val = current.get(key)
                if key not in baseline or key not in current:
                    drifted_fields.append(new_path)
                    drift_details[new_path] = {
                        "baseline": str(b_val),
                        "current": str(c_val),
                        "field_missing": True,
                    }
                else:
                    self._compare_for_drift(
                        b_val, c_val, new_path,
                        drifted_fields, drift_details, tolerance,
                    )

        elif isinstance(baseline, (list, tuple)) and isinstance(current, (list, tuple)):
            for i, (b_item, c_item) in enumerate(zip(baseline, current)):
                new_path = f"{path}[{i}]"
                self._compare_for_drift(
                    b_item, c_item, new_path,
                    drifted_fields, drift_details, tolerance,
                )
            # Handle length differences
            if len(baseline) != len(current):
                drifted_fields.append(f"{path}.__length__")
                drift_details[f"{path}.__length__"] = {
                    "baseline_length": len(baseline),
                    "current_length": len(current),
                }

        else:
            drift_pct, detail = self._calculate_field_drift(
                baseline, current, tolerance,
            )
            if detail:
                drifted_fields.append(path)
                drift_details[path] = detail

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    def create_baseline(
        self,
        name: str,
        description: str,
        baseline_data: Dict[str, Any],
    ) -> DriftBaseline:
        """Create a new drift baseline.

        Args:
            name: Human-readable baseline name.
            description: Description of the baseline.
            baseline_data: Snapshot data to use as baseline.

        Returns:
            DriftBaseline record.
        """
        baseline_hash = self._hasher.compute_hash(baseline_data)

        baseline = DriftBaseline(
            name=name,
            description=description,
            baseline_data=baseline_data,
            baseline_hash=baseline_hash,
        )

        self._baselines[baseline.baseline_id] = baseline
        logger.info(
            "Created drift baseline: id=%s, name=%s",
            baseline.baseline_id[:8], name,
        )
        return baseline

    def update_baseline(
        self,
        baseline_id: str,
        new_data: Dict[str, Any],
    ) -> DriftBaseline:
        """Update an existing drift baseline with new data.

        Args:
            baseline_id: ID of the baseline to update.
            new_data: New baseline data.

        Returns:
            Updated DriftBaseline record.

        Raises:
            ValueError: If the baseline is not found.
        """
        baseline = self._baselines.get(baseline_id)
        if baseline is None:
            raise ValueError(f"Baseline {baseline_id} not found")

        new_hash = self._hasher.compute_hash(new_data)

        updated = DriftBaseline(
            baseline_id=baseline.baseline_id,
            name=baseline.name,
            description=baseline.description,
            baseline_data=new_data,
            baseline_hash=new_hash,
            created_at=baseline.created_at,
            updated_at=_utcnow(),
            is_active=baseline.is_active,
        )

        self._baselines[baseline_id] = updated
        logger.info("Updated drift baseline: id=%s", baseline_id[:8])
        return updated

    def get_baseline(self, baseline_id: str) -> Optional[DriftBaseline]:
        """Get a drift baseline by ID.

        Args:
            baseline_id: Unique baseline identifier.

        Returns:
            DriftBaseline or None if not found.
        """
        return self._baselines.get(baseline_id)

    def list_baselines(self, active_only: bool = True) -> List[DriftBaseline]:
        """List drift baselines.

        Args:
            active_only: Whether to return only active baselines.

        Returns:
            List of DriftBaseline records, newest first.
        """
        baselines = list(self._baselines.values())
        if active_only:
            baselines = [b for b in baselines if b.is_active]
        baselines.sort(key=lambda b: b.created_at, reverse=True)
        return baselines

    def delete_baseline(self, baseline_id: str) -> bool:
        """Delete a drift baseline (soft delete by deactivation).

        Args:
            baseline_id: ID of the baseline to delete.

        Returns:
            True if deleted, False if not found.
        """
        baseline = self._baselines.get(baseline_id)
        if baseline is None:
            return False

        updated = DriftBaseline(
            baseline_id=baseline.baseline_id,
            name=baseline.name,
            description=baseline.description,
            baseline_data=baseline.baseline_data,
            baseline_hash=baseline.baseline_hash,
            created_at=baseline.created_at,
            updated_at=_utcnow(),
            is_active=False,
        )
        self._baselines[baseline_id] = updated
        logger.info("Deactivated drift baseline: id=%s", baseline_id[:8])
        return True

    def compare_to_baseline(
        self,
        baseline_id: str,
        current_data: Dict[str, Any],
        soft_threshold: Optional[float] = None,
        hard_threshold: Optional[float] = None,
        tolerance: Optional[float] = None,
    ) -> DriftDetection:
        """Compare current data against a stored baseline.

        Args:
            baseline_id: ID of the baseline to compare against.
            current_data: Current output data.
            soft_threshold: Override soft threshold (uses config default).
            hard_threshold: Override hard threshold (uses config default).
            tolerance: Override tolerance (uses config default).

        Returns:
            DriftDetection result.

        Raises:
            ValueError: If the baseline is not found.
        """
        baseline = self._baselines.get(baseline_id)
        if baseline is None:
            raise ValueError(f"Baseline {baseline_id} not found")

        return self.detect_drift(
            baseline=baseline.baseline_data,
            current=current_data,
            soft_threshold=soft_threshold or self._config.drift_soft_threshold,
            hard_threshold=hard_threshold or self._config.drift_hard_threshold,
            tolerance=tolerance or self._config.default_absolute_tolerance,
        )


__all__ = [
    "DriftDetector",
]
