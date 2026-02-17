# -*- coding: utf-8 -*-
"""
Outlier Detection Pipeline Engine - AGENT-DATA-013

End-to-end pipeline orchestrating all outlier detection stages:
detect -> classify -> treat -> validate -> document. Supports
checkpointing, ensemble combination, and provenance tracking.

Zero-Hallucination: All computations use deterministic Python
arithmetic. Pipeline orchestration is pure control flow.

Example:
    >>> from greenlang.outlier_detector.outlier_pipeline import OutlierPipelineEngine
    >>> engine = OutlierPipelineEngine()
    >>> result = engine.run_pipeline(records)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.outlier_detector.config import get_config
from greenlang.outlier_detector.contextual_detector import ContextualDetectorEngine
from greenlang.outlier_detector.models import (
    BatchDetectionResult,
    ColumnOutlierSummary,
    DetectionMethod,
    DetectionResult,
    EnsembleMethod,
    EnsembleResult,
    OutlierClassification,
    OutlierReport,
    OutlierScore,
    OutlierStatus,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    SeverityLevel,
    TreatmentResult,
    TreatmentStrategy,
)
from greenlang.outlier_detector.multivariate_detector import MultivariateDetectorEngine
from greenlang.outlier_detector.outlier_classifier import OutlierClassifierEngine
from greenlang.outlier_detector.provenance import ProvenanceTracker
from greenlang.outlier_detector.statistical_detector import StatisticalDetectorEngine
from greenlang.outlier_detector.temporal_detector import TemporalDetectorEngine
from greenlang.outlier_detector.treatment_engine import TreatmentEngine

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _safe_mean(values: List[float]) -> float:
    """Compute arithmetic mean, returning 0.0 for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _severity_from_score(score: float) -> SeverityLevel:
    """Map normalised score to severity level."""
    if score >= 0.95:
        return SeverityLevel.CRITICAL
    if score >= 0.80:
        return SeverityLevel.HIGH
    if score >= 0.60:
        return SeverityLevel.MEDIUM
    if score >= 0.40:
        return SeverityLevel.LOW
    return SeverityLevel.INFO


class OutlierPipelineEngine:
    """End-to-end outlier detection pipeline engine.

    Orchestrates the full detect -> classify -> treat -> validate ->
    document pipeline with configurable methods, ensemble combination,
    checkpointing, and complete provenance tracking.

    Attributes:
        _config: Outlier detector configuration.
        _provenance: SHA-256 provenance tracker.
        _statistical: Statistical detector engine.
        _contextual: Contextual detector engine.
        _temporal: Temporal detector engine.
        _multivariate: Multivariate detector engine.
        _classifier: Outlier classifier engine.
        _treatment: Treatment engine.
        _checkpoints: In-memory checkpoint store.
        _stats: Pipeline execution statistics.

    Example:
        >>> engine = OutlierPipelineEngine()
        >>> result = engine.run_pipeline(
        ...     records=[{"val": 1}, {"val": 2}, {"val": 100}],
        ... )
        >>> print(result.status, result.total_processing_time_ms)
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize OutlierPipelineEngine.

        Args:
            config: Optional OutlierDetectorConfig override.
        """
        self._config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._statistical = StatisticalDetectorEngine(self._config)
        self._contextual = ContextualDetectorEngine(self._config)
        self._temporal = TemporalDetectorEngine(self._config)
        self._multivariate = MultivariateDetectorEngine(self._config)
        self._classifier = OutlierClassifierEngine(self._config)
        self._treatment = TreatmentEngine(self._config)
        self._checkpoints: Dict[str, Dict[str, Any]] = {}
        self._stats: Dict[str, Any] = {
            "pipelines_run": 0,
            "total_records": 0,
            "total_outliers": 0,
            "total_treatments": 0,
        }
        logger.info("OutlierPipelineEngine initialized")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        records: List[Dict[str, Any]],
        config: Optional[PipelineConfig] = None,
    ) -> PipelineResult:
        """Run the full detect -> classify -> treat -> validate -> document pipeline.

        Args:
            records: List of record dictionaries to analyze.
            config: Optional pipeline configuration override.

        Returns:
            PipelineResult with all stage outputs.
        """
        start = time.time()
        cfg = config or PipelineConfig()
        pipeline_id = str(hash(time.time()))

        logger.info(
            "Pipeline started: %d records, %d methods, ensemble=%s",
            len(records), len(cfg.methods), cfg.ensemble_method.value,
        )

        try:
            # Stage 1: Detect
            self._save_checkpoint(pipeline_id, "detect_start", {
                "record_count": len(records),
            })
            batch = self.detect_stage(records, cfg.columns, cfg)
            detection_results = batch.results
            self._save_checkpoint(pipeline_id, "detect_done", {
                "outliers": batch.total_outliers,
            })

            # Stage 2: Ensemble combine
            ensemble_results = self._ensemble_combine_from_batch(batch, cfg)

            # Collect all outlier scores for downstream stages
            all_outlier_scores: List[OutlierScore] = []
            for er in ensemble_results:
                if er.is_outlier:
                    all_outlier_scores.append(OutlierScore(
                        record_index=er.record_index,
                        column_name=er.column_name,
                        value=er.value,
                        method=DetectionMethod.IQR,
                        score=er.ensemble_score,
                        is_outlier=True,
                        threshold=0.5,
                        severity=er.severity,
                        details={"ensemble_score": er.ensemble_score,
                                 "methods_flagged": er.methods_flagged},
                        confidence=er.confidence,
                        provenance_hash=er.provenance_hash,
                    ))

            # Stage 3: Classify
            classifications: List[OutlierClassification] = []
            if cfg.enable_classification and all_outlier_scores:
                self._save_checkpoint(pipeline_id, "classify_start", {})
                classifications = self.classify_stage(
                    all_outlier_scores, records,
                )
                self._save_checkpoint(pipeline_id, "classify_done", {
                    "classified": len(classifications),
                })

            # Stage 4: Treat
            treatments: List[TreatmentResult] = []
            if all_outlier_scores:
                self._save_checkpoint(pipeline_id, "treat_start", {})
                treatments = self.treat_stage(
                    records, classifications, cfg.treatment_strategy,
                    all_outlier_scores,
                )
                self._save_checkpoint(pipeline_id, "treat_done", {
                    "treated": len(treatments),
                })

            # Stage 5: Validate
            self._save_checkpoint(pipeline_id, "validate_start", {})
            validation_summary = self.validate_stage(
                records, treatments,
            )
            self._save_checkpoint(pipeline_id, "validate_done", {})

            # Stage 6: Document
            doc = self.document_stage(
                records, detection_results, ensemble_results,
                classifications, treatments, validation_summary,
            )

            elapsed_ms = (time.time() - start) * 1000.0
            total_outliers = sum(
                1 for er in ensemble_results if er.is_outlier
            )

            # Update stats
            self._stats["pipelines_run"] += 1
            self._stats["total_records"] += len(records)
            self._stats["total_outliers"] += total_outliers
            self._stats["total_treatments"] += len(treatments)

            provenance_hash = self._provenance.add_to_chain(
                "pipeline",
                self._provenance.build_hash({"records": len(records)}),
                self._provenance.build_hash({
                    "outliers": total_outliers,
                    "classifications": len(classifications),
                    "treatments": len(treatments),
                }),
                metadata={"pipeline_id": pipeline_id},
            )

            # Build report
            report = self._build_report(
                records, detection_results, ensemble_results,
                classifications, treatments, doc,
            )

            result = PipelineResult(
                pipeline_id=pipeline_id,
                stage=PipelineStage.DOCUMENT,
                status=OutlierStatus.COMPLETED,
                detection_results=detection_results,
                ensemble_results=ensemble_results,
                classifications=classifications,
                treatments=treatments,
                validation_summary=validation_summary,
                report=report,
                total_processing_time_ms=elapsed_ms,
                provenance_hash=provenance_hash,
            )

            logger.info(
                "Pipeline completed: %d outliers, %d classified, "
                "%d treated, %.1fms",
                total_outliers, len(classifications),
                len(treatments), elapsed_ms,
            )
            return result

        except Exception as e:
            logger.error("Pipeline failed: %s", str(e), exc_info=True)
            elapsed_ms = (time.time() - start) * 1000.0

            return PipelineResult(
                pipeline_id=pipeline_id,
                stage=PipelineStage.DETECT,
                status=OutlierStatus.FAILED,
                total_processing_time_ms=elapsed_ms,
                validation_summary={"error": str(e)},
                provenance_hash=self._provenance.build_hash({
                    "error": str(e),
                }),
            )

    # ------------------------------------------------------------------
    # Detect stage
    # ------------------------------------------------------------------

    def detect_stage(
        self,
        records: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
        config: Optional[PipelineConfig] = None,
    ) -> BatchDetectionResult:
        """Run all enabled detection methods on the dataset.

        Args:
            records: Record dictionaries.
            columns: Specific columns to analyze (None = auto-detect numeric).
            config: Optional pipeline config.

        Returns:
            BatchDetectionResult with per-column results.
        """
        start = time.time()
        cfg = config or PipelineConfig()

        # Auto-detect numeric columns
        cols = columns or []
        if not cols:
            cols = self._detect_numeric_columns(records)

        methods = cfg.methods
        results: List[DetectionResult] = []
        total_outliers = 0

        for col in cols:
            values = self._extract_column(records, col)
            if len(values) < 3:
                continue

            for method in methods:
                col_result = self._run_single_detection(
                    values, col, method,
                )
                results.append(col_result)
                total_outliers += col_result.outliers_found

        # Contextual detection
        if cfg.enable_contextual and cfg.group_columns:
            ctx_results = self._contextual.detect_by_group(
                records,
                value_column=cols[0] if cols else "",
                group_column=cfg.group_columns[0],
            )
            for cr in ctx_results:
                total_outliers += cr.outliers_found

        # Temporal detection
        if cfg.enable_temporal and cfg.time_column:
            for col in cols:
                values = self._extract_column(records, col)
                if len(values) >= 10:
                    temp_results = self._temporal.detect_ewma(
                        values, column_name=col,
                    )
                    for tr in temp_results:
                        total_outliers += tr.anomalies_found

        elapsed_ms = (time.time() - start) * 1000.0
        provenance_hash = self._provenance.build_hash({
            "stage": "detect", "columns": cols,
            "methods": [m.value for m in methods],
            "total_outliers": total_outliers,
        })

        logger.debug("Detect stage: %d columns, %d results, %d outliers, %.1fms",
                      len(cols), len(results), total_outliers, elapsed_ms)

        return BatchDetectionResult(
            results=results,
            total_outliers=total_outliers,
            columns_analyzed=len(cols),
            processing_time_ms=elapsed_ms,
            provenance_hash=provenance_hash,
        )

    # ------------------------------------------------------------------
    # Classify stage
    # ------------------------------------------------------------------

    def classify_stage(
        self,
        detections: List[OutlierScore],
        records: List[Dict[str, Any]],
    ) -> List[OutlierClassification]:
        """Classify all detected outliers.

        Args:
            detections: Outlier scores from detect stage.
            records: Original records.

        Returns:
            List of OutlierClassification.
        """
        return self._classifier.classify_outliers(detections, records)

    # ------------------------------------------------------------------
    # Treat stage
    # ------------------------------------------------------------------

    def treat_stage(
        self,
        records: List[Dict[str, Any]],
        classifications: List[OutlierClassification],
        strategy: Optional[TreatmentStrategy] = None,
        detections: Optional[List[OutlierScore]] = None,
    ) -> List[TreatmentResult]:
        """Apply treatment to detected outliers.

        Uses classifications to determine the best treatment per
        outlier, or applies the specified strategy uniformly.

        Args:
            records: Original records.
            classifications: Outlier classifications.
            strategy: Override treatment strategy (None = use classification).
            detections: Outlier detection scores.

        Returns:
            List of TreatmentResult.
        """
        if not detections:
            return []

        strat = strategy or TreatmentStrategy(self._config.default_treatment)

        # If we have classifications, use recommended treatments
        if classifications and strategy is None:
            treatments: List[TreatmentResult] = []
            for cls in classifications:
                # Find the matching detection
                matching = [
                    d for d in detections
                    if d.record_index == cls.record_index
                ]
                if matching:
                    t_results = self._treatment.apply_treatment(
                        records, matching,
                        cls.recommended_treatment,
                    )
                    treatments.extend(t_results)
            return treatments

        # Uniform strategy
        return self._treatment.apply_treatment(
            records, detections, strat,
        )

    # ------------------------------------------------------------------
    # Validate stage
    # ------------------------------------------------------------------

    def validate_stage(
        self,
        original: List[Dict[str, Any]],
        treatments: List[TreatmentResult],
    ) -> Dict[str, Any]:
        """Validate treatment results by comparing before/after statistics.

        Args:
            original: Original records.
            treatments: Applied treatments.

        Returns:
            Dict with validation statistics.
        """
        if not treatments:
            return {"status": "no_treatments", "valid": True}

        # Group treatments by column
        by_column: Dict[str, List[TreatmentResult]] = {}
        for t in treatments:
            col = t.column_name
            if col not in by_column:
                by_column[col] = []
            by_column[col].append(t)

        validation: Dict[str, Any] = {
            "status": "validated",
            "columns": {},
            "total_treatments": len(treatments),
            "valid": True,
        }

        for col, col_treatments in by_column.items():
            orig_vals = self._extract_column(original, col)
            treated_vals = list(orig_vals)
            for t in col_treatments:
                idx = t.record_index
                if idx < len(treated_vals) and t.treated_value is not None:
                    try:
                        treated_vals[idx] = float(t.treated_value)
                    except (ValueError, TypeError):
                        pass

            orig_mean = _safe_mean(orig_vals)
            treated_mean = _safe_mean(treated_vals)
            mean_change = (
                abs(treated_mean - orig_mean) / abs(orig_mean) * 100.0
                if orig_mean != 0 else 0.0
            )

            col_valid = mean_change < 50.0  # Alert if mean changes >50%

            validation["columns"][col] = {
                "treatments": len(col_treatments),
                "original_mean": orig_mean,
                "treated_mean": treated_mean,
                "mean_change_pct": mean_change,
                "valid": col_valid,
            }

            if not col_valid:
                validation["valid"] = False

        provenance_hash = self._provenance.build_hash(validation)
        validation["provenance_hash"] = provenance_hash

        return validation

    # ------------------------------------------------------------------
    # Document stage
    # ------------------------------------------------------------------

    def document_stage(
        self,
        records: List[Dict[str, Any]],
        detection_results: List[DetectionResult],
        ensemble_results: List[EnsembleResult],
        classifications: List[OutlierClassification],
        treatments: List[TreatmentResult],
        validation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate documentation and provenance for the pipeline run.

        Args:
            records: Original records.
            detection_results: Detection results.
            ensemble_results: Ensemble results.
            classifications: Classifications.
            treatments: Treatments.
            validation: Validation summary.

        Returns:
            Dict with documentation and provenance.
        """
        doc: Dict[str, Any] = {
            "methodology": {
                "detection_methods": list({
                    r.method.value for r in detection_results
                }),
                "classification_enabled": len(classifications) > 0,
                "treatment_strategies": list({
                    t.strategy.value for t in treatments
                }),
            },
            "summary": {
                "total_records": len(records),
                "total_detection_results": len(detection_results),
                "total_ensemble_outliers": sum(
                    1 for e in ensemble_results if e.is_outlier
                ),
                "total_classifications": len(classifications),
                "total_treatments": len(treatments),
                "validation_passed": validation.get("valid", False),
            },
            "provenance": {
                "chain_length": self._provenance.get_chain_length(),
                "genesis_hash": ProvenanceTracker.GENESIS_HASH,
            },
            "generated_at": _utcnow().isoformat(),
        }

        provenance_hash = self._provenance.build_hash(doc)
        doc["provenance_hash"] = provenance_hash

        return doc

    # ------------------------------------------------------------------
    # Ensemble combination
    # ------------------------------------------------------------------

    def _ensemble_combine_from_batch(
        self,
        batch: BatchDetectionResult,
        config: PipelineConfig,
    ) -> List[EnsembleResult]:
        """Combine batch detection results into ensemble scores.

        Args:
            batch: Batch detection results.
            config: Pipeline configuration.

        Returns:
            List of EnsembleResult per data point per column.
        """
        from greenlang.outlier_detector.models import DEFAULT_METHOD_WEIGHTS

        # Group results by column and then by record index
        col_method_scores: Dict[str, Dict[int, Dict[str, float]]] = {}
        col_method_flags: Dict[str, Dict[int, int]] = {}
        col_values: Dict[str, Dict[int, Any]] = {}
        col_methods: Dict[str, int] = {}

        for result in batch.results:
            col = result.column_name
            if col not in col_method_scores:
                col_method_scores[col] = {}
                col_method_flags[col] = {}
                col_values[col] = {}
                col_methods[col] = 0

            col_methods[col] += 1

            for score in result.scores:
                idx = score.record_index
                if idx not in col_method_scores[col]:
                    col_method_scores[col][idx] = {}
                    col_method_flags[col][idx] = 0
                    col_values[col][idx] = score.value

                col_method_scores[col][idx][result.method.value] = score.score
                if score.is_outlier:
                    col_method_flags[col][idx] += 1

        # Combine
        weights = DEFAULT_METHOD_WEIGHTS
        ensemble_method = config.ensemble_method
        min_consensus = config.min_consensus
        ensemble_results: List[EnsembleResult] = []

        for col, idx_scores in col_method_scores.items():
            total_methods = col_methods.get(col, 1)

            for idx, method_scores in sorted(idx_scores.items()):
                combined = self._combine_scores(
                    method_scores, weights, ensemble_method,
                )
                flags = col_method_flags[col].get(idx, 0)

                if ensemble_method == EnsembleMethod.MAJORITY_VOTE:
                    is_outlier = flags >= min_consensus
                else:
                    is_outlier = combined >= 0.5 and flags >= min_consensus

                provenance_hash = self._provenance.build_hash({
                    "ensemble": True, "column": col, "index": idx,
                    "combined": combined, "flags": flags,
                })

                ensemble_results.append(EnsembleResult(
                    record_index=idx,
                    column_name=col,
                    value=col_values[col].get(idx),
                    ensemble_score=combined,
                    is_outlier=is_outlier,
                    method_scores=method_scores,
                    methods_flagged=flags,
                    total_methods=total_methods,
                    ensemble_method=ensemble_method,
                    severity=_severity_from_score(combined),
                    confidence=min(1.0, 0.4 + combined * 0.3
                                   + flags / max(total_methods, 1) * 0.3),
                    provenance_hash=provenance_hash,
                ))

        return ensemble_results

    def _ensemble_combine(
        self,
        scores_by_method: Dict[str, List[OutlierScore]],
    ) -> List[EnsembleResult]:
        """Combine per-method scores into ensemble results (standalone).

        Args:
            scores_by_method: Dict mapping method name to list of scores.

        Returns:
            List of EnsembleResult.
        """
        from greenlang.outlier_detector.models import DEFAULT_METHOD_WEIGHTS
        weights = DEFAULT_METHOD_WEIGHTS
        ensemble_method = EnsembleMethod(self._config.ensemble_method)
        min_consensus = self._config.min_consensus

        # Find all record indices
        all_indices: set = set()
        for scores in scores_by_method.values():
            for s in scores:
                all_indices.add(s.record_index)

        results: List[EnsembleResult] = []
        for idx in sorted(all_indices):
            method_scores: Dict[str, float] = {}
            flags = 0
            value = None
            column = ""

            for method_name, scores in scores_by_method.items():
                for s in scores:
                    if s.record_index == idx:
                        method_scores[method_name] = s.score
                        if s.is_outlier:
                            flags += 1
                        value = s.value
                        column = s.column_name
                        break

            combined = self._combine_scores(
                method_scores, weights, ensemble_method,
            )

            if ensemble_method == EnsembleMethod.MAJORITY_VOTE:
                is_outlier = flags >= min_consensus
            else:
                is_outlier = combined >= 0.5 and flags >= min_consensus

            provenance_hash = self._provenance.build_hash({
                "ensemble": True, "index": idx, "combined": combined,
            })

            results.append(EnsembleResult(
                record_index=idx,
                column_name=column,
                value=value,
                ensemble_score=combined,
                is_outlier=is_outlier,
                method_scores=method_scores,
                methods_flagged=flags,
                total_methods=len(scores_by_method),
                ensemble_method=ensemble_method,
                severity=_severity_from_score(combined),
                confidence=min(1.0, 0.4 + combined * 0.6),
                provenance_hash=provenance_hash,
            ))

        return results

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def create_checkpoint(
        self,
        pipeline_id: str,
        stage: str,
        data: Dict[str, Any],
    ) -> str:
        """Create a named checkpoint for pipeline state.

        Args:
            pipeline_id: Pipeline run identifier.
            stage: Stage name.
            data: Checkpoint data.

        Returns:
            Checkpoint key.
        """
        return self._save_checkpoint(pipeline_id, stage, data)

    def resume_from_checkpoint(
        self,
        pipeline_id: str,
        stage: str,
    ) -> Optional[Dict[str, Any]]:
        """Resume pipeline from a checkpoint.

        Args:
            pipeline_id: Pipeline run identifier.
            stage: Stage name to resume from.

        Returns:
            Checkpoint data or None if not found.
        """
        key = f"{pipeline_id}:{stage}"
        checkpoint = self._checkpoints.get(key)
        if checkpoint:
            logger.info("Resuming from checkpoint: %s", key)
        return checkpoint

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline execution statistics.

        Returns:
            Dict with pipeline statistics.
        """
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_single_detection(
        self,
        values: List[float],
        column: str,
        method: DetectionMethod,
    ) -> DetectionResult:
        """Run a single detection method on a column.

        Args:
            values: Numeric values.
            column: Column name.
            method: Detection method.

        Returns:
            DetectionResult for this method/column.
        """
        start = time.time()
        scores: List[OutlierScore] = []

        dispatch = {
            DetectionMethod.IQR: lambda: self._statistical.detect_iqr(
                values, column_name=column,
            ),
            DetectionMethod.ZSCORE: lambda: self._statistical.detect_zscore(
                values, column_name=column,
            ),
            DetectionMethod.MODIFIED_ZSCORE: lambda: self._statistical.detect_modified_zscore(
                values, column_name=column,
            ),
            DetectionMethod.MAD: lambda: self._statistical.detect_mad(
                values, column_name=column,
            ),
            DetectionMethod.GRUBBS: lambda: self._statistical.detect_grubbs(
                values, column_name=column,
            ),
            DetectionMethod.TUKEY: lambda: self._statistical.detect_tukey(
                values, column_name=column,
            ),
            DetectionMethod.PERCENTILE: lambda: self._statistical.detect_percentile(
                values, column_name=column,
            ),
        }

        fn = dispatch.get(method)
        if fn is not None:
            scores = fn()

        elapsed_ms = (time.time() - start) * 1000.0
        outliers_found = sum(1 for s in scores if s.is_outlier)

        lower_fence = None
        upper_fence = None
        if scores and scores[0].details:
            lower_fence = scores[0].details.get("lower_fence")
            upper_fence = scores[0].details.get("upper_fence")

        provenance_hash = self._provenance.build_hash({
            "detection": method.value, "column": column,
            "points": len(values), "outliers": outliers_found,
        })

        return DetectionResult(
            column_name=column,
            method=method,
            total_points=len(values),
            outliers_found=outliers_found,
            outlier_pct=outliers_found / len(values) if values else 0.0,
            scores=scores,
            lower_fence=lower_fence,
            upper_fence=upper_fence,
            processing_time_ms=elapsed_ms,
            provenance_hash=provenance_hash,
        )

    def _detect_numeric_columns(
        self,
        records: List[Dict[str, Any]],
    ) -> List[str]:
        """Auto-detect numeric columns from records.

        Args:
            records: Record dictionaries.

        Returns:
            List of column names that contain numeric data.
        """
        if not records:
            return []

        # Sample first few records to detect types
        sample = records[:min(10, len(records))]
        all_keys: set = set()
        for rec in sample:
            all_keys.update(rec.keys())

        numeric_cols: List[str] = []
        for key in sorted(all_keys):
            numeric_count = 0
            total_count = 0
            for rec in sample:
                val = rec.get(key)
                if val is not None:
                    total_count += 1
                    try:
                        float(val)
                        numeric_count += 1
                    except (ValueError, TypeError):
                        pass

            if total_count > 0 and numeric_count / total_count >= 0.8:
                numeric_cols.append(key)

        return numeric_cols

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

    def _combine_scores(
        self,
        method_scores: Dict[str, float],
        weights: Dict[str, float],
        ensemble_method: EnsembleMethod,
    ) -> float:
        """Combine per-method scores using ensemble method."""
        if not method_scores:
            return 0.0

        if ensemble_method == EnsembleMethod.MAX_SCORE:
            return max(method_scores.values())

        if ensemble_method == EnsembleMethod.MEAN_SCORE:
            return _safe_mean(list(method_scores.values()))

        if ensemble_method == EnsembleMethod.MAJORITY_VOTE:
            votes = sum(1 for s in method_scores.values() if s >= 0.5)
            return votes / len(method_scores)

        # WEIGHTED_AVERAGE
        total_weight = 0.0
        weighted_sum = 0.0
        for method_name, score in method_scores.items():
            w = weights.get(method_name, 1.0)
            weighted_sum += score * w
            total_weight += w

        if total_weight > 0:
            return min(1.0, weighted_sum / total_weight)
        return 0.0

    def _save_checkpoint(
        self,
        pipeline_id: str,
        stage: str,
        data: Dict[str, Any],
    ) -> str:
        """Save a checkpoint.

        Args:
            pipeline_id: Pipeline identifier.
            stage: Stage name.
            data: Checkpoint data.

        Returns:
            Checkpoint key.
        """
        key = f"{pipeline_id}:{stage}"
        self._checkpoints[key] = {
            "stage": stage,
            "data": data,
            "timestamp": _utcnow().isoformat(),
        }
        logger.debug("Checkpoint saved: %s", key)
        return key

    def _build_report(
        self,
        records: List[Dict[str, Any]],
        detection_results: List[DetectionResult],
        ensemble_results: List[EnsembleResult],
        classifications: List[OutlierClassification],
        treatments: List[TreatmentResult],
        doc: Dict[str, Any],
    ) -> OutlierReport:
        """Build the final outlier report.

        Args:
            records: Original records.
            detection_results: Detection results.
            ensemble_results: Ensemble results.
            classifications: Classifications.
            treatments: Treatments.
            doc: Documentation dict.

        Returns:
            OutlierReport.
        """
        total_outliers = sum(1 for e in ensemble_results if e.is_outlier)
        total_records = len(records)

        # By method
        by_method: Dict[str, int] = {}
        for r in detection_results:
            m = r.method.value
            by_method[m] = by_method.get(m, 0) + r.outliers_found

        # By class
        by_class: Dict[str, int] = {}
        for c in classifications:
            k = c.outlier_class.value
            by_class[k] = by_class.get(k, 0) + 1

        # By treatment
        by_treatment: Dict[str, int] = {}
        for t in treatments:
            k = t.strategy.value
            by_treatment[k] = by_treatment.get(k, 0) + 1

        # Column summaries
        columns_analyzed = set()
        for r in detection_results:
            columns_analyzed.add(r.column_name)

        column_summaries: List[Dict[str, Any]] = []
        for col in sorted(columns_analyzed):
            col_results = [r for r in detection_results if r.column_name == col]
            col_outliers = sum(r.outliers_found for r in col_results)
            col_points = max((r.total_points for r in col_results), default=0)
            col_methods = [r.method.value for r in col_results]

            column_summaries.append({
                "column_name": col,
                "total_points": col_points,
                "outliers_detected": col_outliers,
                "methods_used": col_methods,
            })

        provenance_hash = self._provenance.build_hash({
            "report": True,
            "records": total_records,
            "outliers": total_outliers,
        })

        return OutlierReport(
            total_records=total_records,
            total_columns_analyzed=len(columns_analyzed),
            total_outliers=total_outliers,
            outlier_pct=total_outliers / total_records if total_records > 0 else 0.0,
            by_method=by_method,
            by_class=by_class,
            by_treatment=by_treatment,
            column_summaries=column_summaries,
            provenance_hash=provenance_hash,
        )


__all__ = [
    "OutlierPipelineEngine",
]
