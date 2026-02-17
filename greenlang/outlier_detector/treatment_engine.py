# -*- coding: utf-8 -*-
"""
Outlier Treatment Engine - AGENT-DATA-013

Applies treatment strategies to detected outliers including capping,
winsorization, flagging, removal, replacement with imputed values,
and routing to investigation queues. Supports treatment undo and
impact analysis.

Zero-Hallucination: All treatment computations use deterministic
Python arithmetic. No LLM calls for value calculations.

Example:
    >>> from greenlang.outlier_detector.treatment_engine import TreatmentEngine
    >>> engine = TreatmentEngine()
    >>> results = engine.flag_outliers(records, detections)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
Status: Production Ready
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional

from greenlang.outlier_detector.config import get_config
from greenlang.outlier_detector.models import (
    ImpactAnalysis,
    OutlierScore,
    TreatmentRecord,
    TreatmentResult,
    TreatmentStrategy,
)
from greenlang.outlier_detector.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


def _safe_mean(values: List[float]) -> float:
    """Compute arithmetic mean, returning 0.0 for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_std(values: List[float], mean: Optional[float] = None) -> float:
    """Compute population standard deviation."""
    if len(values) < 2:
        return 0.0
    m = mean if mean is not None else _safe_mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _safe_median(values: List[float]) -> float:
    """Compute median of values."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]


def _percentile(values: List[float], pct: float) -> float:
    """Compute percentile using linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n == 1:
        return s[0]
    k = pct * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


class TreatmentEngine:
    """Outlier treatment engine.

    Applies configurable treatment strategies to detected outliers,
    maintains a treatment record log for undo support, and computes
    impact analysis comparing original and treated data.

    Attributes:
        _config: Outlier detector configuration.
        _provenance: SHA-256 provenance tracker.
        _treatment_log: In-memory treatment record log for undo.

    Example:
        >>> engine = TreatmentEngine()
        >>> results = engine.cap_values(records, "emissions", lower=0, upper=1000)
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize TreatmentEngine.

        Args:
            config: Optional OutlierDetectorConfig override.
        """
        self._config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._treatment_log: Dict[str, TreatmentRecord] = {}
        logger.info("TreatmentEngine initialized")

    # ------------------------------------------------------------------
    # Strategy dispatcher
    # ------------------------------------------------------------------

    def apply_treatment(
        self,
        records: List[Dict[str, Any]],
        detections: List[OutlierScore],
        strategy: TreatmentStrategy,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[TreatmentResult]:
        """Apply a treatment strategy to detected outliers.

        Dispatches to the appropriate treatment method based on
        the strategy parameter.

        Args:
            records: Original record dictionaries.
            detections: Outlier scores from detection stage.
            strategy: Treatment strategy to apply.
            options: Optional treatment-specific options.

        Returns:
            List of TreatmentResult for each treated point.
        """
        start = time.time()
        opts = options or {}
        column_name = ""
        if detections:
            column_name = detections[0].column_name

        dispatch: Dict[TreatmentStrategy, Any] = {
            TreatmentStrategy.CAP: lambda: self.cap_values(
                records, column_name,
                lower=opts.get("lower"),
                upper=opts.get("upper"),
                detections=detections,
            ),
            TreatmentStrategy.WINSORIZE: lambda: self.winsorize(
                records, column_name,
                pct=opts.get("pct"),
                detections=detections,
            ),
            TreatmentStrategy.FLAG: lambda: self.flag_outliers(
                records, detections,
            ),
            TreatmentStrategy.REMOVE: lambda: self.remove_outliers(
                records, detections,
            ),
            TreatmentStrategy.REPLACE: lambda: self.replace_with_imputed(
                records, detections,
                method=opts.get("method", "median"),
            ),
            TreatmentStrategy.INVESTIGATE: lambda: self.mark_for_investigation(
                records, detections,
            ),
        }

        fn = dispatch.get(strategy)
        if fn is None:
            logger.warning("Unknown strategy %s, defaulting to flag", strategy)
            fn = lambda: self.flag_outliers(records, detections)  # noqa: E731

        results = fn()
        elapsed = time.time() - start
        logger.debug(
            "Treatment %s: %d results in %.3fs",
            strategy.value, len(results), elapsed,
        )
        return results

    # ------------------------------------------------------------------
    # Cap values
    # ------------------------------------------------------------------

    def cap_values(
        self,
        records: List[Dict[str, Any]],
        column: str,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        detections: Optional[List[OutlierScore]] = None,
    ) -> List[TreatmentResult]:
        """Cap outlier values at specified bounds.

        Values below lower are set to lower; values above upper are
        set to upper. Defaults to 5th and 95th percentile if not specified.

        Args:
            records: Record dictionaries.
            column: Column to cap.
            lower: Lower cap value (default: 5th percentile).
            upper: Upper cap value (default: 95th percentile).
            detections: Optional detection scores for filtering.

        Returns:
            List of TreatmentResult for each capped value.
        """
        values = self._extract_column(records, column)

        if lower is None:
            lower = _percentile(values, 0.05) if values else 0.0
        if upper is None:
            upper = _percentile(values, 0.95) if values else 0.0

        outlier_indices = self._outlier_indices(detections)
        results: List[TreatmentResult] = []

        for i, rec in enumerate(records):
            if outlier_indices and i not in outlier_indices:
                continue

            val = rec.get(column)
            if val is None:
                continue
            try:
                v = float(val)
            except (ValueError, TypeError):
                continue

            if v < lower or v > upper:
                capped = max(lower, min(upper, v))
                provenance_hash = self._provenance.build_hash({
                    "treatment": "cap", "index": i,
                    "original": v, "capped": capped,
                })

                result = TreatmentResult(
                    record_index=i,
                    column_name=column,
                    original_value=v,
                    treated_value=capped,
                    strategy=TreatmentStrategy.CAP,
                    reason=f"Capped from {v} to [{lower}, {upper}]",
                    reversible=True,
                    confidence=0.8,
                    provenance_hash=provenance_hash,
                )
                results.append(result)
                self._log_treatment(result)

        return results

    # ------------------------------------------------------------------
    # Winsorize
    # ------------------------------------------------------------------

    def winsorize(
        self,
        records: List[Dict[str, Any]],
        column: str,
        pct: Optional[float] = None,
        detections: Optional[List[OutlierScore]] = None,
    ) -> List[TreatmentResult]:
        """Winsorize outlier values by replacing with percentile values.

        Replaces values below the lower percentile with that percentile
        value, and values above the upper percentile with that value.

        Args:
            records: Record dictionaries.
            column: Column to winsorize.
            pct: Percentile fraction (default from config, e.g. 0.05).
            detections: Optional detection scores for filtering.

        Returns:
            List of TreatmentResult for each winsorized value.
        """
        p = pct if pct is not None else self._config.winsorize_pct
        values = self._extract_column(records, column)

        lower_val = _percentile(values, p) if values else 0.0
        upper_val = _percentile(values, 1.0 - p) if values else 0.0

        outlier_indices = self._outlier_indices(detections)
        results: List[TreatmentResult] = []

        for i, rec in enumerate(records):
            if outlier_indices and i not in outlier_indices:
                continue

            val = rec.get(column)
            if val is None:
                continue
            try:
                v = float(val)
            except (ValueError, TypeError):
                continue

            if v < lower_val or v > upper_val:
                winsorized = lower_val if v < lower_val else upper_val
                provenance_hash = self._provenance.build_hash({
                    "treatment": "winsorize", "index": i,
                    "original": v, "winsorized": winsorized, "pct": p,
                })

                result = TreatmentResult(
                    record_index=i,
                    column_name=column,
                    original_value=v,
                    treated_value=winsorized,
                    strategy=TreatmentStrategy.WINSORIZE,
                    reason=f"Winsorized at {p:.1%}/{1.0 - p:.1%} percentiles",
                    reversible=True,
                    confidence=0.75,
                    provenance_hash=provenance_hash,
                )
                results.append(result)
                self._log_treatment(result)

        return results

    # ------------------------------------------------------------------
    # Flag outliers
    # ------------------------------------------------------------------

    def flag_outliers(
        self,
        records: List[Dict[str, Any]],
        detections: List[OutlierScore],
    ) -> List[TreatmentResult]:
        """Flag outliers without modifying data.

        Creates treatment records that mark points as outliers for
        review but do not change any values.

        Args:
            records: Record dictionaries.
            detections: Outlier scores.

        Returns:
            List of TreatmentResult for each flagged outlier.
        """
        results: List[TreatmentResult] = []

        for detection in detections:
            if not detection.is_outlier:
                continue

            provenance_hash = self._provenance.build_hash({
                "treatment": "flag", "index": detection.record_index,
                "value": detection.value, "score": detection.score,
            })

            result = TreatmentResult(
                record_index=detection.record_index,
                column_name=detection.column_name,
                original_value=detection.value,
                treated_value=detection.value,  # No change
                strategy=TreatmentStrategy.FLAG,
                reason=f"Flagged as outlier (score={detection.score:.2f})",
                reversible=True,
                confidence=detection.confidence,
                provenance_hash=provenance_hash,
            )
            results.append(result)
            self._log_treatment(result)

        return results

    # ------------------------------------------------------------------
    # Remove outliers
    # ------------------------------------------------------------------

    def remove_outliers(
        self,
        records: List[Dict[str, Any]],
        detections: List[OutlierScore],
    ) -> List[TreatmentResult]:
        """Mark outliers for exclusion from analysis.

        Sets the treated value to None, indicating the record should
        be excluded from downstream processing.

        Args:
            records: Record dictionaries.
            detections: Outlier scores.

        Returns:
            List of TreatmentResult for each removed outlier.
        """
        results: List[TreatmentResult] = []

        for detection in detections:
            if not detection.is_outlier:
                continue

            provenance_hash = self._provenance.build_hash({
                "treatment": "remove", "index": detection.record_index,
                "value": detection.value,
            })

            result = TreatmentResult(
                record_index=detection.record_index,
                column_name=detection.column_name,
                original_value=detection.value,
                treated_value=None,
                strategy=TreatmentStrategy.REMOVE,
                reason="Marked for exclusion",
                reversible=True,
                confidence=detection.confidence,
                provenance_hash=provenance_hash,
            )
            results.append(result)
            self._log_treatment(result)

        return results

    # ------------------------------------------------------------------
    # Replace with imputed value
    # ------------------------------------------------------------------

    def replace_with_imputed(
        self,
        records: List[Dict[str, Any]],
        detections: List[OutlierScore],
        method: str = "median",
    ) -> List[TreatmentResult]:
        """Replace outlier values with imputed values.

        Computes a replacement value (mean, median, or mode of
        non-outlier values) and substitutes it for each outlier.

        Args:
            records: Record dictionaries.
            detections: Outlier scores.
            method: Imputation method (mean, median, mode).

        Returns:
            List of TreatmentResult for each replaced outlier.
        """
        if not detections:
            return []

        column = detections[0].column_name
        outlier_indices = {d.record_index for d in detections if d.is_outlier}

        # Compute replacement from non-outlier values
        non_outlier_values: List[float] = []
        for i, rec in enumerate(records):
            if i in outlier_indices:
                continue
            val = rec.get(column)
            if val is not None:
                try:
                    non_outlier_values.append(float(val))
                except (ValueError, TypeError):
                    pass

        if method == "mean":
            replacement = _safe_mean(non_outlier_values)
        elif method == "mode":
            replacement = self._compute_mode(non_outlier_values)
        else:
            replacement = _safe_median(non_outlier_values)

        results: List[TreatmentResult] = []

        for detection in detections:
            if not detection.is_outlier:
                continue

            provenance_hash = self._provenance.build_hash({
                "treatment": "replace", "index": detection.record_index,
                "original": detection.value, "replacement": replacement,
                "method": method,
            })

            result = TreatmentResult(
                record_index=detection.record_index,
                column_name=column,
                original_value=detection.value,
                treated_value=replacement,
                strategy=TreatmentStrategy.REPLACE,
                reason=f"Replaced with {method}: {replacement:.4f}",
                reversible=True,
                confidence=0.7,
                provenance_hash=provenance_hash,
            )
            results.append(result)
            self._log_treatment(result)

        return results

    # ------------------------------------------------------------------
    # Mark for investigation
    # ------------------------------------------------------------------

    def mark_for_investigation(
        self,
        records: List[Dict[str, Any]],
        detections: List[OutlierScore],
    ) -> List[TreatmentResult]:
        """Route outliers to human investigation queue.

        Creates treatment records that preserve original values but
        flag them for manual review by domain experts.

        Args:
            records: Record dictionaries.
            detections: Outlier scores.

        Returns:
            List of TreatmentResult for each investigation-queued outlier.
        """
        results: List[TreatmentResult] = []

        for detection in detections:
            if not detection.is_outlier:
                continue

            provenance_hash = self._provenance.build_hash({
                "treatment": "investigate", "index": detection.record_index,
                "value": detection.value, "score": detection.score,
            })

            result = TreatmentResult(
                record_index=detection.record_index,
                column_name=detection.column_name,
                original_value=detection.value,
                treated_value=detection.value,  # Preserved
                strategy=TreatmentStrategy.INVESTIGATE,
                reason=f"Routed to investigation (score={detection.score:.2f})",
                reversible=True,
                confidence=detection.confidence,
                provenance_hash=provenance_hash,
            )
            results.append(result)
            self._log_treatment(result)

        return results

    # ------------------------------------------------------------------
    # Undo treatment
    # ------------------------------------------------------------------

    def undo_treatment(
        self,
        treatment_id: str,
    ) -> TreatmentResult:
        """Reverse a previously applied treatment.

        Looks up the treatment record and creates a new treatment
        result restoring the original value.

        Args:
            treatment_id: Treatment ID to undo.

        Returns:
            TreatmentResult with original value restored.

        Raises:
            ValueError: If treatment_id is not found.
        """
        record = self._treatment_log.get(treatment_id)
        if record is None:
            raise ValueError(f"Treatment {treatment_id} not found")

        if record.undone:
            raise ValueError(f"Treatment {treatment_id} already undone")

        provenance_hash = self._provenance.build_hash({
            "treatment": "undo", "treatment_id": treatment_id,
            "original": record.original_value,
            "treated": record.treated_value,
        })

        result = TreatmentResult(
            record_index=record.record_index,
            column_name=record.column_name,
            original_value=record.treated_value,
            treated_value=record.original_value,
            strategy=record.strategy,
            reason=f"Undo of treatment {treatment_id}",
            reversible=False,
            confidence=1.0,
            provenance_hash=provenance_hash,
        )

        # Mark record as undone
        record.undone = True
        from datetime import datetime, timezone
        record.undone_at = datetime.now(timezone.utc).replace(microsecond=0)

        logger.info("Treatment %s undone", treatment_id)
        return result

    # ------------------------------------------------------------------
    # Impact analysis
    # ------------------------------------------------------------------

    def compute_impact(
        self,
        original: List[Dict[str, Any]],
        treated: List[Dict[str, Any]],
        column: str,
    ) -> ImpactAnalysis:
        """Quantify the statistical impact of treatment.

        Compares original and treated datasets to measure changes
        in mean, standard deviation, and distribution shape.

        Args:
            original: Original record dictionaries.
            treated: Treated record dictionaries.
            column: Column to analyze.

        Returns:
            ImpactAnalysis with before/after comparison.
        """
        orig_vals = self._extract_column(original, column)
        treat_vals = self._extract_column(treated, column)

        orig_mean = _safe_mean(orig_vals)
        treat_mean = _safe_mean(treat_vals)
        orig_std = _safe_std(orig_vals, orig_mean)
        treat_std = _safe_std(treat_vals, treat_mean)
        orig_median = _safe_median(orig_vals)
        treat_median = _safe_median(treat_vals)

        mean_change = (
            abs(treat_mean - orig_mean) / abs(orig_mean) * 100.0
            if orig_mean != 0 else 0.0
        )
        std_change = (
            abs(treat_std - orig_std) / abs(orig_std) * 100.0
            if orig_std != 0 else 0.0
        )

        # Distribution shift: Kolmogorov-Smirnov-like max difference
        dist_shift = self._distribution_shift(orig_vals, treat_vals)

        records_affected = sum(
            1 for i in range(min(len(orig_vals), len(treat_vals)))
            if i < len(orig_vals) and i < len(treat_vals)
            and orig_vals[i] != treat_vals[i]
        )

        provenance_hash = self._provenance.build_hash({
            "impact": "analysis", "column": column,
            "orig_mean": orig_mean, "treat_mean": treat_mean,
            "affected": records_affected,
        })

        return ImpactAnalysis(
            column_name=column,
            records_affected=records_affected,
            original_mean=orig_mean,
            treated_mean=treat_mean,
            original_std=orig_std,
            treated_std=treat_std,
            original_median=orig_median,
            treated_median=treat_median,
            mean_change_pct=mean_change,
            std_change_pct=std_change,
            distribution_shift=dist_shift,
            provenance_hash=provenance_hash,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_column(
        self,
        records: List[Dict[str, Any]],
        column: str,
    ) -> List[float]:
        """Extract numeric values from a column."""
        values: List[float] = []
        for rec in records:
            val = rec.get(column)
            if val is not None:
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    pass
        return values

    def _outlier_indices(
        self,
        detections: Optional[List[OutlierScore]],
    ) -> set:
        """Get set of outlier record indices from detections."""
        if detections is None:
            return set()
        return {d.record_index for d in detections if d.is_outlier}

    def _log_treatment(self, result: TreatmentResult) -> None:
        """Log a treatment result for undo support.

        Args:
            result: Treatment result to log.
        """
        record = TreatmentRecord(
            treatment_id=result.treatment_id,
            column_name=result.column_name,
            record_index=result.record_index,
            original_value=result.original_value,
            treated_value=result.treated_value,
            strategy=result.strategy,
            provenance_hash=result.provenance_hash,
        )
        self._treatment_log[result.treatment_id] = record

    @staticmethod
    def _compute_mode(values: List[float]) -> float:
        """Compute mode of numeric values (most frequent).

        Args:
            values: List of numeric values.

        Returns:
            Mode value or median as fallback.
        """
        if not values:
            return 0.0
        counts: Dict[float, int] = {}
        for v in values:
            rounded = round(v, 6)
            counts[rounded] = counts.get(rounded, 0) + 1
        return max(counts, key=counts.get)  # type: ignore[arg-type]

    @staticmethod
    def _distribution_shift(
        original: List[float],
        treated: List[float],
    ) -> float:
        """Compute a simple distribution shift measure.

        Uses the maximum absolute difference between the empirical
        CDFs of the original and treated distributions.

        Args:
            original: Original values.
            treated: Treated values.

        Returns:
            Distribution shift measure (0.0-1.0).
        """
        if not original or not treated:
            return 0.0

        all_vals = sorted(set(original + treated))
        if not all_vals:
            return 0.0

        max_diff = 0.0
        n_orig = len(original)
        n_treat = len(treated)

        orig_sorted = sorted(original)
        treat_sorted = sorted(treated)
        orig_idx = 0
        treat_idx = 0

        for val in all_vals:
            while orig_idx < n_orig and orig_sorted[orig_idx] <= val:
                orig_idx += 1
            while treat_idx < n_treat and treat_sorted[treat_idx] <= val:
                treat_idx += 1
            cdf_orig = orig_idx / n_orig
            cdf_treat = treat_idx / n_treat
            max_diff = max(max_diff, abs(cdf_orig - cdf_treat))

        return min(1.0, max_diff)


__all__ = [
    "TreatmentEngine",
]
