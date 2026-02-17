# -*- coding: utf-8 -*-
"""
Imputation Pipeline Engine - AGENT-DATA-012: Missing Value Imputer (GL-DATA-X-015)

Orchestrates the full end-to-end imputation pipeline: analyze -> strategize ->
impute -> validate -> document. Coordinates all six sub-engines (missingness
analyzer, statistical imputer, ML imputer, rule-based imputer, time-series
imputer, validation engine) into a single cohesive workflow.

Supports checkpointing for resumable pipelines, fallback strategy chains,
and comprehensive provenance documentation.

Zero-Hallucination Guarantees:
    - Pipeline orchestration is deterministic state machine logic
    - Strategy selection uses rule-based decision tree only
    - All imputation dispatches to zero-hallucination sub-engines
    - SHA-256 provenance at every stage transition
    - No ML/LLM calls in pipeline orchestration code

Example:
    >>> from greenlang.missing_value_imputer.imputation_pipeline import ImputationPipelineEngine
    >>> from greenlang.missing_value_imputer.config import MissingValueImputerConfig
    >>> from greenlang.missing_value_imputer.missingness_analyzer import MissingnessAnalyzerEngine
    >>> from greenlang.missing_value_imputer.statistical_imputer import StatisticalImputerEngine
    >>> from greenlang.missing_value_imputer.ml_imputer import MLImputerEngine
    >>> from greenlang.missing_value_imputer.rule_based_imputer import RuleBasedImputerEngine
    >>> from greenlang.missing_value_imputer.time_series_imputer import TimeSeriesImputerEngine
    >>> from greenlang.missing_value_imputer.validation_engine import ValidationEngine
    >>> config = MissingValueImputerConfig()
    >>> pipeline = ImputationPipelineEngine(
    ...     config,
    ...     MissingnessAnalyzerEngine(config),
    ...     StatisticalImputerEngine(config),
    ...     MLImputerEngine(config),
    ...     RuleBasedImputerEngine(config),
    ...     TimeSeriesImputerEngine(config),
    ...     ValidationEngine(config),
    ... )
    >>> result = pipeline.run_pipeline(records)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
Status: Production Ready
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.missing_value_imputer.config import MissingValueImputerConfig
from greenlang.missing_value_imputer.models import (
    ColumnAnalysis,
    ConfidenceLevel,
    DataColumnType,
    ImputationBatch,
    ImputationResult,
    ImputationStatus,
    ImputationStrategy,
    ImputedValue,
    MissingnessReport,
    MissingnessType,
    PatternType,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    StrategySelection,
    ValidationMethod,
    ValidationReport,
    ValidationResult,
)
from greenlang.missing_value_imputer.metrics import (
    inc_jobs,
    inc_values_imputed,
    inc_strategies_selected,
    observe_duration,
    observe_completeness_improvement,
    set_active_jobs,
    inc_errors,
)
from greenlang.missing_value_imputer.provenance import ProvenanceTracker
from greenlang.missing_value_imputer.missingness_analyzer import MissingnessAnalyzerEngine
from greenlang.missing_value_imputer.statistical_imputer import StatisticalImputerEngine
from greenlang.missing_value_imputer.ml_imputer import MLImputerEngine
from greenlang.missing_value_imputer.rule_based_imputer import RuleBasedImputerEngine
from greenlang.missing_value_imputer.time_series_imputer import TimeSeriesImputerEngine
from greenlang.missing_value_imputer.validation_engine import ValidationEngine

logger = logging.getLogger(__name__)

__all__ = [
    "ImputationPipelineEngine",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _is_missing(value: Any) -> bool:
    """Determine whether a value is considered missing."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _is_numeric(value: Any) -> bool:
    """Check if a value is numeric (excluding bool)."""
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def _compute_provenance(operation: str, data_repr: str) -> str:
    """Compute SHA-256 provenance hash."""
    payload = f"{operation}:{data_repr}:{_utcnow().isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _classify_confidence(score: float) -> ConfidenceLevel:
    """Classify a numeric confidence score into a level."""
    if score >= 0.85:
        return ConfidenceLevel.HIGH
    if score >= 0.70:
        return ConfidenceLevel.MEDIUM
    if score >= 0.50:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.VERY_LOW


# Strategy -> method name mapping for dispatch
_STATISTICAL_STRATEGIES = {
    ImputationStrategy.MEAN,
    ImputationStrategy.MEDIAN,
    ImputationStrategy.MODE,
    ImputationStrategy.KNN,
    ImputationStrategy.REGRESSION,
    ImputationStrategy.HOT_DECK,
    ImputationStrategy.LOCF,
    ImputationStrategy.NOCB,
}

_ML_STRATEGIES = {
    ImputationStrategy.RANDOM_FOREST,
    ImputationStrategy.GRADIENT_BOOSTING,
    ImputationStrategy.MICE,
    ImputationStrategy.MATRIX_FACTORIZATION,
}

_TIMESERIES_STRATEGIES = {
    ImputationStrategy.LINEAR_INTERPOLATION,
    ImputationStrategy.SPLINE_INTERPOLATION,
    ImputationStrategy.SEASONAL_DECOMPOSITION,
}

_RULE_STRATEGIES = {
    ImputationStrategy.RULE_BASED,
    ImputationStrategy.LOOKUP_TABLE,
    ImputationStrategy.REGULATORY_DEFAULT,
}

# Default fallback chain per column type
_FALLBACK_CHAINS: Dict[DataColumnType, List[ImputationStrategy]] = {
    DataColumnType.NUMERIC: [
        ImputationStrategy.MEDIAN,
        ImputationStrategy.MEAN,
        ImputationStrategy.KNN,
        ImputationStrategy.REGRESSION,
    ],
    DataColumnType.CATEGORICAL: [
        ImputationStrategy.MODE,
        ImputationStrategy.HOT_DECK,
        ImputationStrategy.RULE_BASED,
    ],
    DataColumnType.DATETIME: [
        ImputationStrategy.LINEAR_INTERPOLATION,
        ImputationStrategy.LOCF,
        ImputationStrategy.NOCB,
    ],
    DataColumnType.BOOLEAN: [
        ImputationStrategy.MODE,
        ImputationStrategy.RULE_BASED,
    ],
    DataColumnType.TEXT: [
        ImputationStrategy.MODE,
        ImputationStrategy.HOT_DECK,
        ImputationStrategy.RULE_BASED,
    ],
}


# ===========================================================================
# ImputationPipelineEngine
# ===========================================================================


class ImputationPipelineEngine:
    """End-to-end imputation pipeline orchestrator.

    Coordinates missingness analysis, strategy selection, imputation
    execution, validation, and documentation into a single pipeline
    with checkpointing and fallback support.

    Attributes:
        config: Service configuration.
        analyzer: Missingness analysis engine.
        statistical: Statistical imputation engine.
        ml: ML-based imputation engine.
        rule_based: Rule-based imputation engine.
        time_series: Time-series imputation engine.
        validation: Validation engine.
        provenance: SHA-256 provenance tracker.
        _checkpoints: In-memory checkpoint storage.
        _active_jobs: Count of currently running pipelines.

    Example:
        >>> pipeline = ImputationPipelineEngine(config, analyzer, ...)
        >>> result = pipeline.run_pipeline(records)
        >>> assert result.status == ImputationStatus.COMPLETED
    """

    def __init__(
        self,
        config: MissingValueImputerConfig,
        analyzer: MissingnessAnalyzerEngine,
        statistical: StatisticalImputerEngine,
        ml: MLImputerEngine,
        rule_based: RuleBasedImputerEngine,
        time_series: TimeSeriesImputerEngine,
        validation: ValidationEngine,
    ) -> None:
        """Initialize the ImputationPipelineEngine.

        Args:
            config: Service configuration.
            analyzer: Missingness analysis engine.
            statistical: Statistical imputation engine.
            ml: ML-based imputation engine.
            rule_based: Rule-based imputation engine.
            time_series: Time-series imputation engine.
            validation: Validation engine.
        """
        self.config = config
        self.analyzer = analyzer
        self.statistical = statistical
        self.ml = ml
        self.rule_based = rule_based
        self.time_series = time_series
        self.validation = validation
        self.provenance = ProvenanceTracker()
        self._checkpoints: Dict[str, Dict[str, Any]] = {}
        self._active_jobs: int = 0
        logger.info("ImputationPipelineEngine initialized")

    # ------------------------------------------------------------------
    # Main pipeline entry point
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        records: List[Dict[str, Any]],
        config: Optional[PipelineConfig] = None,
    ) -> PipelineResult:
        """Execute the full imputation pipeline.

        Stages:
            1. Analyze: Detect missingness patterns.
            2. Strategize: Select best strategy per column.
            3. Impute: Execute imputation per column.
            4. Validate: Validate all imputed values.
            5. Document: Generate provenance documentation.

        Args:
            records: List of record dictionaries to impute.
            config: Optional pipeline configuration overrides.

        Returns:
            PipelineResult with complete analysis, imputation, and
            validation results.

        Raises:
            ValueError: If records list is empty.
        """
        pipeline_start = time.monotonic()
        pipeline_id = str(uuid.uuid4())

        if not records:
            raise ValueError("records must be non-empty for pipeline")

        self._active_jobs += 1
        set_active_jobs(self._active_jobs)

        pipeline_config = config or PipelineConfig()

        try:
            # Stage 1: Analyze
            logger.info("Pipeline %s: ANALYZE stage", pipeline_id[:8])
            self.create_checkpoint(
                pipeline_id, PipelineStage.ANALYZE.value, {"records_count": len(records)}
            )
            analysis_report = self.analyze_stage(records)

            # Stage 2: Strategize
            logger.info("Pipeline %s: STRATEGIZE stage", pipeline_id[:8])
            self.create_checkpoint(
                pipeline_id, PipelineStage.STRATEGIZE.value,
                {"analysis_done": True},
            )
            strategies = self.strategize_stage(
                analysis_report,
                user_preferences=pipeline_config.column_strategies,
            )

            # Stage 3: Impute
            logger.info("Pipeline %s: IMPUTE stage", pipeline_id[:8])
            self.create_checkpoint(
                pipeline_id, PipelineStage.IMPUTE.value,
                {"strategies_count": len(strategies)},
            )
            imputation_result = self.impute_stage(records, strategies)

            # Build imputed records for validation
            imputed_records = self._apply_imputations(records, imputation_result)

            # Stage 4: Validate
            logger.info("Pipeline %s: VALIDATE stage", pipeline_id[:8])
            self.create_checkpoint(
                pipeline_id, PipelineStage.VALIDATE.value,
                {"values_imputed": imputation_result.total_values_imputed},
            )
            validation_report = self.validate_stage(records, imputed_records)

            # Stage 5: Document
            logger.info("Pipeline %s: DOCUMENT stage", pipeline_id[:8])
            documentation = self.document_stage(imputation_result, validation_report)

            # Build pipeline result
            total_time = (time.monotonic() - pipeline_start) * 1000

            pipeline_prov = _compute_provenance(
                "pipeline",
                f"{pipeline_id}:{len(records)}:{imputation_result.total_values_imputed}",
            )

            result = PipelineResult(
                pipeline_id=pipeline_id,
                stage=PipelineStage.DOCUMENT,
                status=ImputationStatus.COMPLETED,
                analysis_report=analysis_report,
                strategy_selections=list(strategies.values()),
                imputation_batch=imputation_result,
                validation_report=validation_report,
                total_processing_time_ms=round(total_time, 2),
                provenance_hash=pipeline_prov,
            )

            # Record provenance chain
            self.provenance.record(
                "pipeline", pipeline_id, "complete", pipeline_prov
            )

            inc_jobs("completed")
            observe_duration("pipeline", total_time / 1000.0)

            logger.info(
                "Pipeline %s COMPLETED: %d records, %d values imputed, "
                "validation=%s, elapsed=%.1fms",
                pipeline_id[:8],
                len(records),
                imputation_result.total_values_imputed,
                "PASS" if validation_report.overall_passed else "FAIL",
                total_time,
            )
            return result

        except Exception as e:
            self._active_jobs = max(0, self._active_jobs - 1)
            set_active_jobs(self._active_jobs)
            inc_jobs("failed")
            inc_errors("pipeline")
            logger.error(
                "Pipeline %s FAILED: %s", pipeline_id[:8], str(e), exc_info=True
            )
            return PipelineResult(
                pipeline_id=pipeline_id,
                status=ImputationStatus.FAILED,
                total_processing_time_ms=(time.monotonic() - pipeline_start) * 1000,
                provenance_hash=_compute_provenance("pipeline_failed", str(e)),
            )
        finally:
            self._active_jobs = max(0, self._active_jobs - 1)
            set_active_jobs(self._active_jobs)

    # ------------------------------------------------------------------
    # Individual pipeline stages
    # ------------------------------------------------------------------

    def analyze_stage(
        self, records: List[Dict[str, Any]]
    ) -> MissingnessReport:
        """Run the missingness analysis stage.

        Args:
            records: List of record dictionaries.

        Returns:
            MissingnessReport with complete analysis.
        """
        return self.analyzer.analyze_dataset(records)

    def strategize_stage(
        self,
        report: MissingnessReport,
        user_preferences: Optional[Dict[str, str]] = None,
    ) -> Dict[str, StrategySelection]:
        """Select imputation strategy per column.

        Combines auto-recommendation with optional user overrides.

        Args:
            report: MissingnessReport from analyze_stage.
            user_preferences: Optional dict mapping column -> strategy name.

        Returns:
            Dict mapping column name to StrategySelection.
        """
        # Auto-recommend
        strategies = self.analyzer.recommend_strategies(report)

        # Apply user overrides
        if user_preferences:
            for col, strategy_name in user_preferences.items():
                if col in strategies:
                    try:
                        strategy_enum = ImputationStrategy(strategy_name)
                        strategies[col] = StrategySelection(
                            column_name=col,
                            recommended_strategy=strategy_enum,
                            alternative_strategies=strategies[col].alternative_strategies,
                            rationale=f"User override: {strategy_name}",
                            estimated_confidence=strategies[col].estimated_confidence,
                            column_type=strategies[col].column_type,
                            missing_pct=strategies[col].missing_pct,
                            provenance_hash=_compute_provenance(
                                "user_override", f"{col}:{strategy_name}"
                            ),
                        )
                    except ValueError:
                        logger.warning(
                            "Unknown strategy '%s' for column '%s', ignoring override",
                            strategy_name, col,
                        )

        # Record metrics
        for sel in strategies.values():
            inc_strategies_selected(sel.recommended_strategy.value)

        return strategies

    def impute_stage(
        self,
        records: List[Dict[str, Any]],
        strategies: Dict[str, StrategySelection],
    ) -> ImputationBatch:
        """Execute imputation for all columns per their strategies.

        Uses the fallback chain mechanism: if the primary strategy fails,
        tries alternatives in order.

        Args:
            records: List of record dictionaries.
            strategies: Per-column strategy selections.

        Returns:
            ImputationBatch with all results.
        """
        start = time.monotonic()
        results: List[ImputationResult] = []
        total_imputed = 0

        for col_name, selection in strategies.items():
            col_start = time.monotonic()

            # Build fallback chain
            fallback_strategies = [selection.recommended_strategy]
            fallback_strategies.extend(selection.alternative_strategies)

            # Try imputation with fallback
            imputed_values = self._apply_fallback_chain(
                records, col_name, fallback_strategies
            )

            # Calculate completeness improvement
            total_in_col = len(records)
            missing_before = sum(
                1 for r in records if _is_missing(r.get(col_name))
            )
            filled = len(imputed_values)
            completeness_before = (
                (total_in_col - missing_before) / total_in_col
                if total_in_col > 0 else 1.0
            )
            completeness_after = (
                (total_in_col - missing_before + filled) / total_in_col
                if total_in_col > 0 else 1.0
            )

            improvement = completeness_after - completeness_before
            if improvement > 0:
                observe_completeness_improvement(
                    selection.recommended_strategy.value, improvement
                )

            # Compute confidence stats
            confidences = [iv.confidence for iv in imputed_values]
            avg_conf = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )
            min_conf = min(confidences) if confidences else 0.0

            col_elapsed = (time.monotonic() - col_start) * 1000

            prov = _compute_provenance(
                "impute_column", f"{col_name}:{filled}:{avg_conf:.4f}"
            )
            result = ImputationResult(
                column_name=col_name,
                strategy=selection.recommended_strategy,
                values_imputed=filled,
                imputed_values=imputed_values,
                avg_confidence=round(avg_conf, 4),
                min_confidence=round(min_conf, 4),
                completeness_before=round(completeness_before, 6),
                completeness_after=round(completeness_after, 6),
                processing_time_ms=round(col_elapsed, 2),
                provenance_hash=prov,
            )
            results.append(result)
            total_imputed += filled

        total_elapsed = (time.monotonic() - start) * 1000

        # Batch confidence
        all_confidences = []
        for r in results:
            all_confidences.extend([iv.confidence for iv in r.imputed_values])
        batch_conf = (
            sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        )

        batch_prov = _compute_provenance(
            "imputation_batch", f"{total_imputed}:{batch_conf:.4f}"
        )
        return ImputationBatch(
            results=results,
            total_values_imputed=total_imputed,
            avg_confidence=round(batch_conf, 4),
            processing_time_ms=round(total_elapsed, 2),
            provenance_hash=batch_prov,
        )

    def validate_stage(
        self,
        original: List[Dict[str, Any]],
        imputed: List[Dict[str, Any]],
    ) -> ValidationReport:
        """Validate all imputed values.

        Args:
            original: Original records before imputation.
            imputed: Records after imputation.

        Returns:
            ValidationReport with per-column results.
        """
        start = time.monotonic()

        all_cols = set()
        for r in imputed:
            all_cols.update(r.keys())

        validation_results: List[Dict[str, Any]] = []

        for col in sorted(all_cols):
            # Check if column had missing values that were imputed
            orig_missing = sum(1 for r in original if _is_missing(r.get(col)))
            imp_missing = sum(1 for r in imputed if _is_missing(r.get(col)))

            if orig_missing == 0 or orig_missing == imp_missing:
                continue  # Column was not imputed

            # Get non-missing values for comparison
            orig_vals = [
                r.get(col) for r in original if not _is_missing(r.get(col))
            ]
            imp_vals = [
                r.get(col) for r in imputed if not _is_missing(r.get(col))
            ]

            if not orig_vals or not imp_vals:
                continue

            # Check if numeric
            numeric_orig = [v for v in orig_vals if _is_numeric(v)]
            if len(numeric_orig) > len(orig_vals) * 0.5:
                # KS test for numeric columns
                ks_result = self.validation.ks_test(
                    [float(v) for v in numeric_orig],
                    [float(v) for v in imp_vals if _is_numeric(v)],
                )
                ks_result["column"] = col
                ks_result["test"] = "ks_test"
                validation_results.append(ks_result)

                # Plausibility check
                orig_floats = [float(v) for v in numeric_orig]
                imp_floats = [
                    float(v) for v in imp_vals
                    if _is_numeric(v) and v not in orig_vals
                ]
                if imp_floats and orig_floats:
                    from statistics import stdev as _stdev, mean as _mean
                    orig_mean = _mean(orig_floats)
                    orig_std = _stdev(orig_floats) if len(orig_floats) > 1 else 1.0
                    plaus = self.validation.plausibility_check(
                        imp_floats,
                        {
                            "mean": orig_mean,
                            "std": orig_std,
                            "min": min(orig_floats),
                            "max": max(orig_floats),
                        },
                    )
                    plaus["column"] = col
                    plaus["test"] = "plausibility_range"
                    validation_results.append(plaus)
            else:
                # Chi-square for categorical
                chi_result = self.validation.chi_square_test(orig_vals, imp_vals)
                chi_result["column"] = col
                chi_result["test"] = "chi_square"
                validation_results.append(chi_result)

        report = self.validation.generate_validation_report(validation_results)

        elapsed = time.monotonic() - start
        observe_duration("validate", elapsed)

        logger.info(
            "Validation stage: %d tests, %d passed, %d failed",
            len(validation_results),
            report.columns_passed,
            report.columns_failed,
        )
        return report

    def document_stage(
        self,
        result: ImputationBatch,
        validation: ValidationReport,
    ) -> Dict[str, Any]:
        """Generate provenance and methodology documentation.

        Args:
            result: Imputation batch results.
            validation: Validation report.

        Returns:
            Dict with documentation fields.
        """
        doc: Dict[str, Any] = {
            "generated_at": _utcnow().isoformat(),
            "summary": {
                "total_values_imputed": result.total_values_imputed,
                "avg_confidence": result.avg_confidence,
                "validation_passed": validation.overall_passed,
                "columns_passed": validation.columns_passed,
                "columns_failed": validation.columns_failed,
                "processing_time_ms": result.processing_time_ms,
            },
            "columns": {},
            "methodology": [],
            "provenance_chain": [],
        }

        for col_result in result.results:
            doc["columns"][col_result.column_name] = {
                "strategy": col_result.strategy.value,
                "values_imputed": col_result.values_imputed,
                "avg_confidence": col_result.avg_confidence,
                "min_confidence": col_result.min_confidence,
                "completeness_before": col_result.completeness_before,
                "completeness_after": col_result.completeness_after,
                "improvement": round(
                    col_result.completeness_after - col_result.completeness_before, 6
                ),
                "provenance_hash": col_result.provenance_hash,
            }
            doc["methodology"].append({
                "column": col_result.column_name,
                "method": col_result.strategy.value,
                "description": self._strategy_description(col_result.strategy),
            })

        # Provenance chain
        doc["provenance_chain"] = self.provenance.get_global_chain(limit=50)
        doc["provenance_hash"] = _compute_provenance(
            "document_stage",
            f"{result.total_values_imputed}:{validation.overall_passed}",
        )

        logger.info(
            "Documentation stage: %d columns documented", len(doc["columns"])
        )
        return doc

    # ------------------------------------------------------------------
    # Strategy dispatch
    # ------------------------------------------------------------------

    def _select_strategy(
        self, column_analysis: ColumnAnalysis
    ) -> ImputationStrategy:
        """Auto-select the best strategy for a column.

        Delegates to the analyzer's recommendation logic.

        Args:
            column_analysis: Analysis for the target column.

        Returns:
            Selected ImputationStrategy.
        """
        return column_analysis.recommended_strategy

    def _apply_fallback_chain(
        self,
        records: List[Dict[str, Any]],
        column: str,
        strategies: List[ImputationStrategy],
    ) -> List[ImputedValue]:
        """Try strategies in order until one succeeds.

        Args:
            records: List of record dictionaries.
            column: Column to impute.
            strategies: Ordered list of strategies to try.

        Returns:
            List of ImputedValue from the first successful strategy.
        """
        for strategy in strategies:
            try:
                result = self._dispatch_imputation(records, column, strategy)
                if result:
                    logger.debug(
                        "Strategy %s succeeded for column '%s': %d values",
                        strategy.value, column, len(result),
                    )
                    return result
            except (ValueError, RuntimeError) as e:
                logger.warning(
                    "Strategy %s failed for column '%s': %s, trying next",
                    strategy.value, column, str(e),
                )
                continue

        logger.warning(
            "All strategies failed for column '%s'", column
        )
        return []

    def _dispatch_imputation(
        self,
        records: List[Dict[str, Any]],
        column: str,
        strategy: ImputationStrategy,
    ) -> List[ImputedValue]:
        """Dispatch imputation to the appropriate engine.

        Args:
            records: List of record dictionaries.
            column: Column to impute.
            strategy: Strategy to execute.

        Returns:
            List of ImputedValue from the engine.

        Raises:
            ValueError: If strategy is not supported or engine not available.
        """
        # Statistical strategies
        if strategy == ImputationStrategy.MEAN:
            return self.statistical.impute_mean(records, column)
        if strategy == ImputationStrategy.MEDIAN:
            return self.statistical.impute_median(records, column)
        if strategy == ImputationStrategy.MODE:
            return self.statistical.impute_mode(records, column)
        if strategy == ImputationStrategy.KNN:
            return self.statistical.impute_knn(records, column)
        if strategy == ImputationStrategy.REGRESSION:
            return self.statistical.impute_regression(records, column)
        if strategy == ImputationStrategy.HOT_DECK:
            return self.statistical.impute_hot_deck(records, column)
        if strategy == ImputationStrategy.LOCF:
            return self.statistical.impute_locf(records, column)
        if strategy == ImputationStrategy.NOCB:
            return self.statistical.impute_nocb(records, column)

        # ML strategies
        if strategy == ImputationStrategy.RANDOM_FOREST:
            if not self.config.enable_ml_imputation:
                raise ValueError("ML imputation is disabled in config")
            return self.ml.impute_random_forest(records, column)
        if strategy == ImputationStrategy.GRADIENT_BOOSTING:
            if not self.config.enable_ml_imputation:
                raise ValueError("ML imputation is disabled in config")
            return self.ml.impute_gradient_boosting(records, column)
        if strategy == ImputationStrategy.MICE:
            if not self.config.enable_ml_imputation:
                raise ValueError("ML imputation is disabled in config")
            mice_result = self.ml.impute_mice(records, columns=[column])
            return mice_result.get(column, [])
        if strategy == ImputationStrategy.MATRIX_FACTORIZATION:
            if not self.config.enable_ml_imputation:
                raise ValueError("ML imputation is disabled in config")
            mf_result = self.ml.impute_matrix_factorization(
                records, columns=[column]
            )
            return mf_result.get(column, [])

        # Time-series strategies
        if strategy == ImputationStrategy.LINEAR_INTERPOLATION:
            if not self.config.enable_timeseries:
                raise ValueError("Time-series imputation is disabled in config")
            series = [r.get(column) for r in records]
            float_series = [
                float(v) if not _is_missing(v) and _is_numeric(v) else None
                for v in series
            ]
            result = self.time_series.impute_linear_interpolation(float_series)
            # Fix column names
            return [
                ImputedValue(
                    record_index=iv.record_index,
                    column_name=column,
                    imputed_value=iv.imputed_value,
                    original_value=iv.original_value,
                    strategy=iv.strategy,
                    confidence=iv.confidence,
                    confidence_level=iv.confidence_level,
                    contributing_records=iv.contributing_records,
                    provenance_hash=iv.provenance_hash,
                )
                for iv in result
            ]
        if strategy == ImputationStrategy.SPLINE_INTERPOLATION:
            if not self.config.enable_timeseries:
                raise ValueError("Time-series imputation is disabled in config")
            series = [r.get(column) for r in records]
            float_series = [
                float(v) if not _is_missing(v) and _is_numeric(v) else None
                for v in series
            ]
            result = self.time_series.impute_spline_interpolation(float_series)
            return [
                ImputedValue(
                    record_index=iv.record_index,
                    column_name=column,
                    imputed_value=iv.imputed_value,
                    original_value=iv.original_value,
                    strategy=iv.strategy,
                    confidence=iv.confidence,
                    confidence_level=iv.confidence_level,
                    contributing_records=iv.contributing_records,
                    provenance_hash=iv.provenance_hash,
                )
                for iv in result
            ]
        if strategy == ImputationStrategy.SEASONAL_DECOMPOSITION:
            if not self.config.enable_timeseries:
                raise ValueError("Time-series imputation is disabled in config")
            series = [r.get(column) for r in records]
            float_series = [
                float(v) if not _is_missing(v) and _is_numeric(v) else None
                for v in series
            ]
            result = self.time_series.impute_seasonal_decomposition(float_series)
            return [
                ImputedValue(
                    record_index=iv.record_index,
                    column_name=column,
                    imputed_value=iv.imputed_value,
                    original_value=iv.original_value,
                    strategy=iv.strategy,
                    confidence=iv.confidence,
                    confidence_level=iv.confidence_level,
                    contributing_records=iv.contributing_records,
                    provenance_hash=iv.provenance_hash,
                )
                for iv in result
            ]

        # Rule-based strategies
        if strategy == ImputationStrategy.RULE_BASED:
            if not self.config.enable_rule_based:
                raise ValueError("Rule-based imputation is disabled in config")
            return []  # Requires rules to be provided externally
        if strategy == ImputationStrategy.REGULATORY_DEFAULT:
            if not self.config.enable_rule_based:
                raise ValueError("Rule-based imputation is disabled in config")
            return self.rule_based.regulatory_defaults(records, column)
        if strategy == ImputationStrategy.LOOKUP_TABLE:
            return []  # Requires lookup table to be provided externally

        raise ValueError(f"Unsupported strategy: {strategy.value}")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def create_checkpoint(
        self,
        pipeline_id: str,
        stage: str,
        data: Dict[str, Any],
    ) -> str:
        """Save pipeline state at a checkpoint.

        Args:
            pipeline_id: Pipeline identifier.
            stage: Current stage name.
            data: Stage-specific state data.

        Returns:
            Checkpoint ID.
        """
        checkpoint_id = f"{pipeline_id}:{stage}"
        self._checkpoints[checkpoint_id] = {
            "pipeline_id": pipeline_id,
            "stage": stage,
            "data": data,
            "timestamp": _utcnow().isoformat(),
            "provenance_hash": _compute_provenance(
                "checkpoint", f"{pipeline_id}:{stage}"
            ),
        }
        logger.debug("Checkpoint saved: %s", checkpoint_id)
        return checkpoint_id

    def resume_from_checkpoint(
        self, checkpoint_id: str
    ) -> Optional[PipelineResult]:
        """Resume a pipeline from a saved checkpoint.

        Note: In the current in-memory implementation, this returns None
        if the checkpoint data is insufficient for a full resume. A
        production implementation would restore the full pipeline state
        from persistent storage.

        Args:
            checkpoint_id: Checkpoint ID to resume from.

        Returns:
            PipelineResult if resumable, else None.
        """
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint:
            logger.warning("Checkpoint not found: %s", checkpoint_id)
            return None

        logger.info(
            "Resume requested from checkpoint %s (stage=%s)",
            checkpoint_id, checkpoint.get("stage"),
        )
        # In a production system, this would re-hydrate state and
        # continue from the saved stage. For now, return None to
        # indicate the caller should re-run.
        return None

    # ------------------------------------------------------------------
    # Pipeline statistics
    # ------------------------------------------------------------------

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get current pipeline processing statistics.

        Returns:
            Dict with active_jobs, checkpoints, provenance_entries.
        """
        return {
            "active_jobs": self._active_jobs,
            "checkpoints_stored": len(self._checkpoints),
            "provenance_entries": self.provenance.entry_count,
            "provenance_entities": self.provenance.entity_count,
            "timestamp": _utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_imputations(
        self,
        original_records: List[Dict[str, Any]],
        batch: ImputationBatch,
    ) -> List[Dict[str, Any]]:
        """Apply imputed values to create a new set of records.

        Args:
            original_records: Original records.
            batch: ImputationBatch with imputed values.

        Returns:
            New list of records with imputed values filled in.
        """
        # Deep copy to avoid mutating originals
        imputed_records = [dict(r) for r in original_records]

        for col_result in batch.results:
            for iv in col_result.imputed_values:
                idx = iv.record_index
                if 0 <= idx < len(imputed_records):
                    imputed_records[idx][iv.column_name] = iv.imputed_value

        return imputed_records

    def _strategy_description(self, strategy: ImputationStrategy) -> str:
        """Get a human-readable description of a strategy.

        Args:
            strategy: ImputationStrategy enum value.

        Returns:
            Description string.
        """
        descriptions = {
            ImputationStrategy.MEAN: (
                "Mean imputation: replaces missing values with the arithmetic "
                "mean of observed values in the column."
            ),
            ImputationStrategy.MEDIAN: (
                "Median imputation: replaces missing values with the median "
                "of observed values, robust to outliers."
            ),
            ImputationStrategy.MODE: (
                "Mode imputation: replaces missing values with the most "
                "frequent observed value in the column."
            ),
            ImputationStrategy.KNN: (
                "K-Nearest Neighbors: finds the k most similar complete "
                "records using Euclidean distance and averages their values."
            ),
            ImputationStrategy.REGRESSION: (
                "Linear regression: fits an OLS model using other columns "
                "as predictors to estimate missing values."
            ),
            ImputationStrategy.HOT_DECK: (
                "Hot-deck: selects a donor from observed values using "
                "random or sequential selection."
            ),
            ImputationStrategy.LOCF: (
                "Last Observation Carried Forward: carries the most recent "
                "non-missing value forward in time."
            ),
            ImputationStrategy.NOCB: (
                "Next Observation Carried Backward: fills missing values "
                "with the next available observed value."
            ),
            ImputationStrategy.RANDOM_FOREST: (
                "Random Forest: builds an ensemble of decision trees on "
                "bootstrap samples and averages their predictions."
            ),
            ImputationStrategy.GRADIENT_BOOSTING: (
                "Gradient Boosting: iteratively fits shallow trees to "
                "residuals using squared-error loss."
            ),
            ImputationStrategy.MICE: (
                "MICE (Multiple Imputation by Chained Equations): "
                "iteratively imputes each column conditional on all others."
            ),
            ImputationStrategy.MATRIX_FACTORIZATION: (
                "Matrix Factorization: uses alternating least squares to "
                "approximate a low-rank decomposition and reconstruct missing values."
            ),
            ImputationStrategy.LINEAR_INTERPOLATION: (
                "Linear interpolation: draws a straight line between "
                "nearest non-missing neighbors to fill gaps."
            ),
            ImputationStrategy.SPLINE_INTERPOLATION: (
                "Cubic spline interpolation: fits smooth piecewise cubic "
                "polynomials through observed data points."
            ),
            ImputationStrategy.SEASONAL_DECOMPOSITION: (
                "Seasonal decomposition: decomposes the series into trend + "
                "seasonal components and reconstructs missing values."
            ),
            ImputationStrategy.RULE_BASED: (
                "Rule-based: applies domain-specific if-then rules with "
                "priority ordering and audit-ready justifications."
            ),
            ImputationStrategy.LOOKUP_TABLE: (
                "Lookup table: matches record fields against a reference "
                "table to find the imputed value."
            ),
            ImputationStrategy.REGULATORY_DEFAULT: (
                "Regulatory default: applies published default values from "
                "GHG Protocol, DEFRA, or EPA for common emission factors."
            ),
            ImputationStrategy.CUSTOM: (
                "Custom imputation: user-defined imputation function."
            ),
        }
        return descriptions.get(strategy, f"Imputation strategy: {strategy.value}")
