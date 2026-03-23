# -*- coding: utf-8 -*-
"""
QualityScorerEngine - 4-Dimension Quality Scoring with Composite Grading (Engine 3 of 7)

AGENT-MRV-013: Dual Reporting Reconciliation Agent

Core quality assessment engine scoring reconciliation data across four
dimensions defined by the GHG Protocol Scope 2 Guidance:

    1. Completeness (weight 0.30): Are all energy types, facilities, and
       periods covered by both location-based and market-based results?
    2. Consistency (weight 0.25): Are methodological choices, GWP values,
       reporting boundaries, and time periods consistent between the two
       methods?
    3. Accuracy (weight 0.25): Are emission factors from authoritative
       sources, and are calculations free from arithmetic errors?
    4. Transparency (weight 0.20): Are all assumptions, data sources,
       emission factors, and contractual instruments fully documented
       with provenance hashes?

Composite Scoring Formula:
    composite = 0.30 * completeness + 0.25 * consistency
              + 0.25 * accuracy + 0.20 * transparency

Grade Thresholds:
    A  >= 0.90  (Assurance-Ready)
    B  >= 0.80  (High Quality)
    C  >= 0.65  (Acceptable)
    D  >= 0.50  (Needs Improvement)
    F  <  0.50  (Insufficient)

Emission Factor Hierarchy Quality Scores (GHG Protocol Scope 2):
    supplier_with_cert  = 1.00
    supplier_no_cert    = 0.85
    bundled_cert        = 0.75
    unbundled_cert      = 0.65
    residual_mix        = 0.40
    grid_average        = 0.20

Cross-Check:
    For each upstream result: emissions_tco2e == energy_quantity_mwh * ef_used
    within a tolerance of 0.01 tCO2e.

EF Hierarchy Distribution:
    Counts market-based results at each GHG Protocol EF hierarchy level.

Zero-Hallucination Guarantees:
    - All scores are computed with deterministic Decimal arithmetic (8dp).
    - No LLM calls in the scoring path.
    - Every assessment carries a SHA-256 provenance hash.
    - Identical inputs always produce identical outputs.

Thread Safety:
    Thread-safe singleton using threading.RLock with _instance/_initialized
    double-checked locking pattern. All scoring operations are stateless
    and re-entrant. Shared counters protected by reentrant lock.

Example:
    >>> from greenlang.agents.mrv.dual_reporting_reconciliation.quality_scorer import (
    ...     QualityScorerEngine,
    ... )
    >>> engine = QualityScorerEngine()
    >>> assessment = engine.score_quality(workspace)
    >>> print(assessment.grade, assessment.composite_score)
    A 0.92000000

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-013 Dual Reporting Reconciliation (GL-MRV-X-024)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Engine identifier for registry and logging.
ENGINE_ID: str = "QualityScorerEngine"

#: Engine version string.
ENGINE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Conditional imports -- graceful degradation when peer modules unavailable
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.models import (
        EnergyType,
        Scope2Method,
        QualityDimension,
        QualityGrade,
        EFHierarchyPriority,
        FlagType,
        FlagSeverity,
        DataQualityTier,
        GWPSource,
        ReportingFramework,
        QUALITY_WEIGHTS,
        QUALITY_GRADE_THRESHOLDS,
        EF_HIERARCHY_QUALITY_SCORES,
        ReconciliationWorkspace,
        QualityScore,
        QualityAssessment,
        Flag,
        UpstreamResult,
        AGENT_ID,
        AGENT_COMPONENT,
        VERSION,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False
    logger.warning(
        "QualityScorerEngine: models module not available; "
        "engine will operate in degraded mode"
    )

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.config import (
        DualReportingReconciliationConfig as _Config,
    )
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _Config = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.provenance import (
        DualReportingProvenanceTracker as _ProvenanceTracker,
        ProvenanceStage as _ProvenanceStage,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _ProvenanceTracker = None  # type: ignore[misc,assignment]
    _ProvenanceStage = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.metrics import (
        DualReportingReconciliationMetrics as _Metrics,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _Metrics = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Decimal precision constants
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")  # 8 decimal places
_ZERO = Decimal("0")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")

# ---------------------------------------------------------------------------
# Score boundaries
# ---------------------------------------------------------------------------

SCORE_FLOOR: Decimal = Decimal("0")
SCORE_CEILING: Decimal = Decimal("1")

# ---------------------------------------------------------------------------
# All energy types / methods frozen sets (when models available)
# ---------------------------------------------------------------------------

if _MODELS_AVAILABLE:
    ALL_ENERGY_TYPES: frozenset = frozenset(EnergyType)
    BOTH_METHODS: frozenset = frozenset(Scope2Method)
    ASSURANCE_READY_GRADES: frozenset = frozenset({
        QualityGrade.A, QualityGrade.B
    })
else:
    ALL_ENERGY_TYPES = frozenset()
    BOTH_METHODS = frozenset()
    ASSURANCE_READY_GRADES = frozenset()

# ---------------------------------------------------------------------------
# Tier quality scores for accuracy dimension
# ---------------------------------------------------------------------------

TIER_QUALITY_SCORES: Dict[str, Decimal] = {
    "tier_3": Decimal("1.00"),
    "tier_2": Decimal("0.70"),
    "tier_1": Decimal("0.40"),
}

# ---------------------------------------------------------------------------
# Sub-dimension weights for completeness
# ---------------------------------------------------------------------------

COMPLETENESS_SUB_WEIGHTS: Dict[str, Decimal] = {
    "energy_type": Decimal("0.30"),
    "dual_method_bonus": Decimal("0.10"),
    "facility": Decimal("0.40"),
    "period": Decimal("0.20"),
}

# ---------------------------------------------------------------------------
# Sub-dimension weights for consistency
# ---------------------------------------------------------------------------

CONSISTENCY_SUB_WEIGHTS: Dict[str, Decimal] = {
    "gwp": Decimal("0.25"),
    "period": Decimal("0.25"),
    "boundary": Decimal("0.25"),
    "ef_vintage": Decimal("0.25"),
}

# ---------------------------------------------------------------------------
# Sub-dimension weights for accuracy
# ---------------------------------------------------------------------------

ACCURACY_SUB_WEIGHTS: Dict[str, Decimal] = {
    "ef_hierarchy": Decimal("0.40"),
    "tier": Decimal("0.30"),
    "cross_check": Decimal("0.30"),
}

# ---------------------------------------------------------------------------
# Sub-dimension weights for transparency
# ---------------------------------------------------------------------------

TRANSPARENCY_SUB_WEIGHTS: Dict[str, Decimal] = {
    "provenance": Decimal("0.30"),
    "ef_source": Decimal("0.25"),
    "metadata": Decimal("0.20"),
    "ef_hierarchy_doc": Decimal("0.25"),
}

# ---------------------------------------------------------------------------
# Cross-check tolerance
# ---------------------------------------------------------------------------

CROSS_CHECK_TOLERANCE: Decimal = Decimal("0.01")

# ---------------------------------------------------------------------------
# Flag thresholds
# ---------------------------------------------------------------------------

FLAG_WARNING_THRESHOLD: Decimal = Decimal("0.65")
FLAG_ERROR_THRESHOLD: Decimal = Decimal("0.50")
FLAG_CODE_PREFIX: str = "DRR-Q"

# ---------------------------------------------------------------------------
# Grade labels
# ---------------------------------------------------------------------------

GRADE_LABELS: Dict[str, str] = {
    "A": "Assurance-Ready",
    "B": "High Quality",
    "C": "Acceptable",
    "D": "Needs Improvement",
    "F": "Insufficient",
}

# ---------------------------------------------------------------------------
# Recognised EF databases for source quality scoring
# ---------------------------------------------------------------------------

RECOGNISED_EF_DATABASES: frozenset = frozenset({
    "egrid", "aib", "iea", "defra", "epa", "ecoinvent",
    "gabi", "re-diss", "green-e", "ember", "iges",
    "residual mix", "residual_mix", "unfccc",
})


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _clamp(
    value: Decimal,
    floor: Decimal = SCORE_FLOOR,
    ceiling: Decimal = SCORE_CEILING,
) -> Decimal:
    """Clamp a Decimal value between floor and ceiling.

    Args:
        value: The value to clamp.
        floor: Minimum allowed value (default 0).
        ceiling: Maximum allowed value (default 1).

    Returns:
        Clamped Decimal value.
    """
    if value < floor:
        return floor
    if value > ceiling:
        return ceiling
    return value


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = _ZERO,
) -> Decimal:
    """Safely divide two Decimals, returning default on zero denominator.

    Args:
        numerator: The numerator.
        denominator: The denominator.
        default: Value to return if denominator is zero.

    Returns:
        Result of division or default.
    """
    if denominator == _ZERO:
        return default
    return numerator / denominator


def _quantize(value: Decimal) -> Decimal:
    """Quantize a Decimal to 8 decimal places with ROUND_HALF_UP.

    Args:
        value: Decimal value to quantize.

    Returns:
        Quantized Decimal with 8 decimal places.
    """
    return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)


def _D(value: Any) -> Decimal:
    """Convert a value to Decimal.

    Args:
        value: Value to convert (int, float, str, or Decimal).

    Returns:
        Decimal representation.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _safe_decimal(value: Any, default: Decimal = _ZERO) -> Decimal:
    """Safely convert a value to Decimal, returning default on failure.

    Args:
        value: Value to convert.
        default: Default on conversion failure.

    Returns:
        Decimal representation or default.
    """
    if value is None:
        return default
    try:
        return _D(value)
    except (InvalidOperation, ValueError, TypeError):
        return default


def _hash_dict(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of a dictionary.

    Args:
        data: Dictionary to hash.

    Returns:
        Hexadecimal SHA-256 digest string (64 chars).
    """
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Handles Pydantic models (via model_dump), dicts, lists, and scalars.

    Args:
        data: Data to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest string (64 chars).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = {
            k: (v.model_dump(mode="json") if hasattr(v, "model_dump") else v)
            for k, v in data.items()
        }
    elif isinstance(data, (list, tuple)):
        serializable = [
            (item.model_dump(mode="json") if hasattr(item, "model_dump") else item)
            for item in data
        ]
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _contains_year(text: str) -> bool:
    """Check whether text contains a plausible year reference (2000-2099).

    Args:
        text: Text to search.

    Returns:
        True if a year reference is found.
    """
    matches = re.findall(r'\b(20\d{2})\b', text)
    return len(matches) > 0


def _extract_vintage_year(ef_source: str) -> Optional[str]:
    """Extract a 4-digit year from an emission factor source string.

    Looks for patterns like ``eGRID 2023``, ``AIB Residual Mix 2024``.
    Returns the first 4-digit year found in the 2000-2099 range.

    Args:
        ef_source: Emission factor source description string.

    Returns:
        4-digit year string or None if no year found.
    """
    if not ef_source:
        return None
    matches = re.findall(r'\b(20\d{2})\b', ef_source)
    return matches[0] if matches else None


# ===========================================================================
# QualityScorerEngine
# ===========================================================================


class QualityScorerEngine:
    """Engine 3: Scores data quality across 4 dimensions with composite grading.

    Thread-safe singleton. Evaluates completeness, consistency, accuracy,
    and transparency of dual Scope 2 reporting with weighted composite
    scoring and A-F grading per GHG Protocol Scope 2 Guidance quality
    hierarchy.

    The engine computes:
    - Per-dimension scores (0.0 to 1.0) for completeness, consistency,
      accuracy, and transparency.
    - A weighted composite score using configurable dimension weights
      (default: 0.30 / 0.25 / 0.25 / 0.20).
    - A letter grade (A through F) based on composite score thresholds.
    - An assurance-readiness flag (True if grade is A or B).
    - Arithmetic cross-checks on all upstream results.
    - EF hierarchy distribution for market-based results.
    - Flags and recommendations for quality improvement.

    Attributes:
        _config: Agent configuration singleton (optional).
        _metrics: Prometheus metrics collector singleton (optional).
        _provenance: Provenance tracker singleton (optional).
        _lock: Reentrant lock protecting mutable counters.
        _total_assessments: Counter of total assessments performed.
        _total_cross_checks: Counter of total cross-checks performed.
        _total_flags_generated: Counter of total flags generated.
        _created_at: UTC timestamp of engine creation.

    Example:
        >>> engine = QualityScorerEngine()
        >>> assessment = engine.score_quality(workspace)
        >>> assert assessment.grade in (QualityGrade.A, QualityGrade.B)
        >>> assert assessment.assurance_ready is True

        >>> # Singleton behaviour
        >>> engine2 = QualityScorerEngine()
        >>> assert engine is engine2

        >>> # Reset for testing
        >>> QualityScorerEngine.reset()
    """

    _instance: Optional[QualityScorerEngine] = None
    _initialized: bool = False
    _lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Singleton lifecycle
    # ------------------------------------------------------------------

    def __new__(cls) -> QualityScorerEngine:
        """Return the singleton instance, creating it on first call.

        Uses double-checked locking with threading.RLock for thread safety.

        Returns:
            The singleton QualityScorerEngine instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialise engine dependencies.

        Guarded by _initialized flag so repeated instantiation is a no-op.
        Loads optional references to config, metrics, and provenance.
        """
        if self.__class__._initialized:
            return
        with self._lock:
            if self.__class__._initialized:
                return
            self._setup()
            self.__class__._initialized = True
            logger.info(
                "%s v%s initialised: config=%s, metrics=%s, "
                "provenance=%s, models=%s",
                ENGINE_ID,
                ENGINE_VERSION,
                _CONFIG_AVAILABLE,
                _METRICS_AVAILABLE,
                _PROVENANCE_AVAILABLE,
                _MODELS_AVAILABLE,
            )

    def _setup(self) -> None:
        """Internal initialisation of engine state.

        Loads optional peer module references and initialises
        mutable counters. Called once during singleton creation.
        """
        # Peer module references
        self._config = None
        if _CONFIG_AVAILABLE and _Config is not None:
            try:
                self._config = _Config()
            except Exception as exc:
                logger.warning(
                    "QualityScorerEngine: config load failed: %s", exc
                )

        self._metrics = None
        if _METRICS_AVAILABLE and _Metrics is not None:
            try:
                self._metrics = _Metrics()
            except Exception as exc:
                logger.warning(
                    "QualityScorerEngine: metrics load failed: %s", exc
                )

        self._provenance = None
        if _PROVENANCE_AVAILABLE and _ProvenanceTracker is not None:
            try:
                self._provenance = _ProvenanceTracker.get_instance()
            except Exception as exc:
                logger.warning(
                    "QualityScorerEngine: provenance load failed: %s", exc
                )

        # Mutable counters (protected by _lock)
        self._total_assessments: int = 0
        self._total_cross_checks: int = 0
        self._total_flags_generated: int = 0
        self._created_at: datetime = _utcnow()

    # ------------------------------------------------------------------
    # Singleton management
    # ------------------------------------------------------------------

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance for testing.

        After calling reset(), the next instantiation will create a
        fresh engine with re-read configuration.

        Warning:
            Only call this in test setups, never in production code.
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False
            logger.info("%s singleton reset", ENGINE_ID)

    # ------------------------------------------------------------------
    # Counter helpers
    # ------------------------------------------------------------------

    def _increment_assessments(self) -> None:
        """Thread-safe increment of the assessment counter."""
        with self._lock:
            self._total_assessments += 1

    def _increment_cross_checks(self) -> None:
        """Thread-safe increment of the cross-check counter."""
        with self._lock:
            self._total_cross_checks += 1

    def _increment_flags(self, count: int = 1) -> None:
        """Thread-safe increment of the flag counter.

        Args:
            count: Number of flags to add.
        """
        with self._lock:
            self._total_flags_generated += count

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def get_quality_weights(self) -> Dict[str, Decimal]:
        """Return quality dimension weights from config or model defaults.

        Returns:
            Dictionary mapping dimension value strings to Decimal weights.
        """
        if (
            self._config is not None
            and hasattr(self._config, "completeness_weight")
        ):
            return {
                "completeness": _safe_decimal(
                    self._config.completeness_weight, Decimal("0.30"),
                ),
                "consistency": _safe_decimal(
                    self._config.consistency_weight, Decimal("0.25"),
                ),
                "accuracy": _safe_decimal(
                    self._config.accuracy_weight, Decimal("0.25"),
                ),
                "transparency": _safe_decimal(
                    self._config.transparency_weight, Decimal("0.20"),
                ),
            }
        if _MODELS_AVAILABLE:
            return {
                dim.value: weight
                for dim, weight in QUALITY_WEIGHTS.items()
            }
        return {
            "completeness": Decimal("0.30"),
            "consistency": Decimal("0.25"),
            "accuracy": Decimal("0.25"),
            "transparency": Decimal("0.20"),
        }

    def _get_assurance_threshold(self) -> Decimal:
        """Return the assurance-ready composite score threshold.

        Returns:
            Decimal threshold (default 0.80 for grade B).
        """
        if (
            self._config is not None
            and hasattr(self._config, "assurance_threshold")
        ):
            return _safe_decimal(
                self._config.assurance_threshold, Decimal("0.80"),
            )
        return Decimal("0.80")

    # ==================================================================
    # 1. score_quality -- Main entry point
    # ==================================================================

    def score_quality(
        self, workspace: ReconciliationWorkspace,
    ) -> QualityAssessment:
        """Run full 4-dimension quality assessment on a workspace.

        Scores completeness, consistency, accuracy, and transparency,
        computes a weighted composite score, assigns a letter grade,
        and generates flags and recommendations. Also performs an
        arithmetic cross-check and assesses EF hierarchy distribution.

        Args:
            workspace: The reconciliation workspace containing all
                location-based and market-based upstream results.

        Returns:
            A complete QualityAssessment with dimension scores,
            composite score, grade, and aggregate findings.

        Raises:
            ValueError: If workspace has no upstream results.
            RuntimeError: If models module is not available.
        """
        start_time = time.monotonic()

        if not _MODELS_AVAILABLE:
            raise RuntimeError(
                "QualityScorerEngine: models module is required for "
                "quality scoring but is not available"
            )

        if not workspace.location_results and not workspace.market_results:
            raise ValueError(
                "Cannot score quality for workspace with no upstream results"
            )

        logger.info(
            "Scoring quality for reconciliation %s (tenant=%s, "
            "location_results=%d, market_results=%d)",
            workspace.reconciliation_id,
            workspace.tenant_id,
            len(workspace.location_results),
            len(workspace.market_results),
        )

        try:
            # Step 1: Score each dimension
            completeness_qs = self.score_completeness(workspace)
            consistency_qs = self.score_consistency(workspace)
            accuracy_qs = self.score_accuracy(workspace)
            transparency_qs = self.score_transparency(workspace)

            # Step 2: Collect dimension scores into dict
            dimension_scores_dict: Dict[str, QualityScore] = {
                QualityDimension.COMPLETENESS.value: completeness_qs,
                QualityDimension.CONSISTENCY.value: consistency_qs,
                QualityDimension.ACCURACY.value: accuracy_qs,
                QualityDimension.TRANSPARENCY.value: transparency_qs,
            }

            # Step 3: Compute weighted composite score
            composite = self.compute_composite_score(dimension_scores_dict)

            # Step 4: Assign letter grade
            grade = self.assign_grade(composite)

            # Step 5: Determine assurance readiness
            assurance_ready = grade in ASSURANCE_READY_GRADES

            # Stricter check: no dimension below 0.50
            for qs in [completeness_qs, consistency_qs,
                       accuracy_qs, transparency_qs]:
                if qs.score < Decimal("0.50"):
                    assurance_ready = False
                    break

            # Step 6: Aggregate findings from all dimensions
            all_findings: List[str] = []
            for qs in [completeness_qs, consistency_qs,
                       accuracy_qs, transparency_qs]:
                all_findings.extend(qs.findings)

            # Step 7: Cross-check and EF hierarchy distribution
            cross_check_passed, cross_check_issues = (
                self.cross_check_emissions(workspace)
            )
            ef_dist = self.assess_ef_hierarchy_distribution(workspace)

            if not cross_check_passed:
                all_findings.append(
                    f"Arithmetic cross-check failed with "
                    f"{len(cross_check_issues)} issue(s)"
                )
            else:
                all_findings.append(
                    "Arithmetic cross-check passed for all results"
                )

            # Step 8: Generate flags
            flags = self.generate_quality_flags(
                composite, grade, assurance_ready, dimension_scores_dict,
            )

            # Step 9: Generate recommendations
            recommendations = self.generate_recommendations(
                composite, grade, assurance_ready, dimension_scores_dict,
            )

            if recommendations:
                all_findings.append(
                    f"{len(recommendations)} improvement recommendation(s) "
                    "generated"
                )

            # Step 10: Build provenance hash
            provenance_input = {
                "reconciliation_id": workspace.reconciliation_id,
                "dimension_scores": {
                    k: str(v.score) for k, v in dimension_scores_dict.items()
                },
                "composite_score": str(composite),
                "grade": grade.value,
                "cross_check_passed": cross_check_passed,
                "ef_hierarchy_distribution": ef_dist,
            }
            provenance_hash = _hash_dict(provenance_input)
            all_findings.append(
                f"Quality provenance hash: {provenance_hash[:16]}..."
            )

            # Step 11: Build assessment
            scores_list = [
                completeness_qs,
                consistency_qs,
                accuracy_qs,
                transparency_qs,
            ]

            assessment = QualityAssessment(
                reconciliation_id=workspace.reconciliation_id,
                scores=scores_list,
                composite_score=composite,
                grade=grade,
                assurance_ready=assurance_ready,
                findings=all_findings,
            )

            # Step 12: Record metrics
            elapsed_s = time.monotonic() - start_time
            self._increment_assessments()
            self._record_quality_metrics(assessment, elapsed_s)

            # Step 13: Record provenance stages
            self._record_provenance_stages(
                workspace.reconciliation_id, composite, grade,
            )

            logger.info(
                "Quality assessment complete for %s: composite=%.8f, "
                "grade=%s (%s), assurance_ready=%s, findings=%d, "
                "flags=%d, recommendations=%d, cross_check=%s (%.3fs)",
                workspace.reconciliation_id,
                composite,
                grade.value,
                GRADE_LABELS.get(grade.value, "Unknown"),
                assurance_ready,
                len(all_findings),
                len(flags),
                len(recommendations),
                cross_check_passed,
                elapsed_s,
            )

            return assessment

        except Exception:
            elapsed_s = time.monotonic() - start_time
            logger.error(
                "Quality scoring failed for %s after %.3fs",
                workspace.reconciliation_id,
                elapsed_s,
                exc_info=True,
            )
            if self._metrics is not None:
                try:
                    self._metrics.record_error(
                        "calculation_error", "score_quality"
                    )
                except Exception:
                    pass
            raise

    # ==================================================================
    # 2. score_completeness
    # ==================================================================

    def score_completeness(
        self, workspace: ReconciliationWorkspace,
    ) -> QualityScore:
        """Score completeness of the reconciliation workspace (0.0 to 1.0).

        Scoring components:
            - Energy type presence (30%): each of 4 energy types found
              in at least one result contributes 0.25 of the sub-score.
            - Dual method bonus (10%): for each energy type with both
              location and market results, add 0.05 (max 0.20 total).
            - Facility coverage (40%): proportion of facilities that
              have both methods, weighted by emission share.
            - Period coverage (20%): fraction of results covering the
              full workspace reporting period.

        Args:
            workspace: Populated reconciliation workspace.

        Returns:
            QualityScore for the completeness dimension.
        """
        findings: List[str] = []

        # --- Sub-dimension 1: Energy type coverage (30%) ---
        location_types = {r.energy_type for r in workspace.location_results}
        market_types = {r.energy_type for r in workspace.market_results}
        all_present_types = location_types | market_types

        total_possible = len(EnergyType)
        types_present = len(all_present_types)
        energy_type_score = _safe_divide(
            _D(types_present), _D(total_possible), _ZERO
        )

        if types_present == total_possible:
            findings.append(
                f"All {total_possible} energy types present"
            )
        else:
            missing = set(EnergyType) - all_present_types
            missing_names = sorted(m.value for m in missing)
            findings.append(
                f"{types_present}/{total_possible} energy types present; "
                f"missing: {', '.join(missing_names)}"
            )

        # --- Sub-dimension 2: Dual method bonus (10%) ---
        dual_count = 0
        for et in EnergyType:
            has_loc = et in location_types
            has_mkt = et in market_types
            if has_loc and has_mkt:
                dual_count += 1
            elif has_loc:
                findings.append(
                    f"Energy type {et.value}: location-based only"
                )
            elif has_mkt:
                findings.append(
                    f"Energy type {et.value}: market-based only"
                )

        dual_method_score = _safe_divide(
            _D(dual_count), _D(total_possible), _ZERO
        )

        # --- Sub-dimension 3: Facility coverage (40%) ---
        facility_score = self._score_facility_coverage(workspace, findings)

        # --- Sub-dimension 4: Period coverage (20%) ---
        period_score = self._score_period_coverage(workspace, findings)

        # --- Weighted composite ---
        raw_score = (
            COMPLETENESS_SUB_WEIGHTS["energy_type"] * energy_type_score
            + COMPLETENESS_SUB_WEIGHTS["dual_method_bonus"] * dual_method_score
            + COMPLETENESS_SUB_WEIGHTS["facility"] * facility_score
            + COMPLETENESS_SUB_WEIGHTS["period"] * period_score
        )
        final_score = _clamp(_quantize(raw_score))

        logger.debug(
            "Completeness for %s: energy_type=%.4f, dual=%.4f, "
            "facility=%.4f, period=%.4f => %.8f",
            workspace.reconciliation_id,
            energy_type_score, dual_method_score,
            facility_score, period_score, final_score,
        )

        return QualityScore(
            dimension=QualityDimension.COMPLETENESS,
            score=final_score,
            max_score=_ONE,
            findings=findings,
        )

    # ==================================================================
    # 3. score_consistency
    # ==================================================================

    def score_consistency(
        self, workspace: ReconciliationWorkspace,
    ) -> QualityScore:
        """Score consistency of the reconciliation workspace (0.0 to 1.0).

        Scoring components (each 25%):
            1. GWP source: All results use the same GWP source?
            2. Reporting period: All results cover the same dates?
            3. Organisational boundary: All results share same tenant_id?
            4. EF vintage: All EF sources reference the same year?

        Args:
            workspace: Populated reconciliation workspace.

        Returns:
            QualityScore for the consistency dimension.
        """
        findings: List[str] = []
        all_results = (
            list(workspace.location_results)
            + list(workspace.market_results)
        )

        if not all_results:
            findings.append(
                "No upstream results available for consistency scoring"
            )
            return QualityScore(
                dimension=QualityDimension.CONSISTENCY,
                score=_ZERO,
                max_score=_ONE,
                findings=findings,
            )

        # Sub 1: GWP source
        gwp_score = self._score_gwp_consistency(all_results, findings)

        # Sub 2: Reporting period
        period_score = self._score_period_consistency(all_results, findings)

        # Sub 3: Organisational boundary (tenant_id)
        boundary_score = self._score_boundary_consistency(
            all_results, findings
        )

        # Sub 4: EF vintage
        vintage_score = self._score_ef_vintage_consistency(
            all_results, findings
        )

        raw_score = (
            CONSISTENCY_SUB_WEIGHTS["gwp"] * gwp_score
            + CONSISTENCY_SUB_WEIGHTS["period"] * period_score
            + CONSISTENCY_SUB_WEIGHTS["boundary"] * boundary_score
            + CONSISTENCY_SUB_WEIGHTS["ef_vintage"] * vintage_score
        )
        final_score = _clamp(_quantize(raw_score))

        logger.debug(
            "Consistency for %s: gwp=%.4f, period=%.4f, "
            "boundary=%.4f, vintage=%.4f => %.8f",
            workspace.reconciliation_id,
            gwp_score, period_score, boundary_score,
            vintage_score, final_score,
        )

        return QualityScore(
            dimension=QualityDimension.CONSISTENCY,
            score=final_score,
            max_score=_ONE,
            findings=findings,
        )

    # ==================================================================
    # 4. score_accuracy
    # ==================================================================

    def score_accuracy(
        self, workspace: ReconciliationWorkspace,
    ) -> QualityScore:
        """Score accuracy of the reconciliation workspace (0.0 to 1.0).

        Scoring components:
            1. EF hierarchy quality for market results (40%):
               Average EF_HIERARCHY_QUALITY_SCORES across market results.
            2. Data quality tier (30%):
               Tier3=1.0, Tier2=0.7, Tier1=0.4, averaged.
            3. Cross-check pass rate (30%):
               Fraction of results where emissions == MWh * EF.

        Args:
            workspace: Populated reconciliation workspace.

        Returns:
            QualityScore for the accuracy dimension.
        """
        findings: List[str] = []
        all_results = (
            list(workspace.location_results)
            + list(workspace.market_results)
        )

        if not all_results:
            findings.append(
                "No upstream results available for accuracy scoring"
            )
            return QualityScore(
                dimension=QualityDimension.ACCURACY,
                score=_ZERO,
                max_score=_ONE,
                findings=findings,
            )

        # Sub 1: EF hierarchy quality (market-based only)
        ef_score = self._score_ef_hierarchy_quality(
            list(workspace.market_results), findings,
        )

        # Sub 2: Data quality tier
        tier_score = self._score_data_quality_tier(all_results, findings)

        # Sub 3: Cross-check pass rate
        cc_score = self._score_cross_check_rate(all_results, findings)

        raw_score = (
            ACCURACY_SUB_WEIGHTS["ef_hierarchy"] * ef_score
            + ACCURACY_SUB_WEIGHTS["tier"] * tier_score
            + ACCURACY_SUB_WEIGHTS["cross_check"] * cc_score
        )
        final_score = _clamp(_quantize(raw_score))

        logger.debug(
            "Accuracy for %s: ef_hierarchy=%.4f, tier=%.4f, "
            "cross_check=%.4f => %.8f",
            workspace.reconciliation_id,
            ef_score, tier_score, cc_score, final_score,
        )

        return QualityScore(
            dimension=QualityDimension.ACCURACY,
            score=final_score,
            max_score=_ONE,
            findings=findings,
        )

    # ==================================================================
    # 5. score_transparency
    # ==================================================================

    def score_transparency(
        self, workspace: ReconciliationWorkspace,
    ) -> QualityScore:
        """Score transparency of the reconciliation workspace (0.0 to 1.0).

        Scoring components:
            1. Provenance hash coverage (30%): Fraction of results with
               non-empty provenance_hash.
            2. EF source documentation (25%): Fraction of results with
               documented ef_source string.
            3. Metadata coverage (20%): Fraction of results with
               non-empty metadata dictionaries.
            4. EF hierarchy documentation for market results (25%):
               Fraction of market results with ef_hierarchy set.

        Args:
            workspace: Populated reconciliation workspace.

        Returns:
            QualityScore for the transparency dimension.
        """
        findings: List[str] = []
        all_results = (
            list(workspace.location_results)
            + list(workspace.market_results)
        )

        if not all_results:
            findings.append(
                "No upstream results available for transparency scoring"
            )
            return QualityScore(
                dimension=QualityDimension.TRANSPARENCY,
                score=_ZERO,
                max_score=_ONE,
                findings=findings,
            )

        total_count = _D(len(all_results))
        market_count = _D(len(workspace.market_results))

        # Sub 1: Provenance hash coverage
        prov_present = sum(
            1 for r in all_results
            if r.provenance_hash and len(r.provenance_hash.strip()) > 0
        )
        prov_score = _safe_divide(_D(prov_present), total_count, _ZERO)
        findings.append(
            f"Provenance hash: {prov_present}/{len(all_results)} results"
        )

        # Sub 2: EF source documentation
        ef_src_present = sum(
            1 for r in all_results
            if r.ef_source and len(r.ef_source.strip()) > 0
        )
        ef_src_score = _safe_divide(_D(ef_src_present), total_count, _ZERO)
        findings.append(
            f"EF source documented: {ef_src_present}/{len(all_results)} results"
        )

        # Sub 3: Metadata coverage
        meta_present = sum(
            1 for r in all_results
            if r.metadata and len(r.metadata) > 0
        )
        meta_score = _safe_divide(_D(meta_present), total_count, _ZERO)
        findings.append(
            f"Metadata present: {meta_present}/{len(all_results)} results"
        )

        # Sub 4: EF hierarchy documentation (market only)
        if workspace.market_results:
            hier_present = sum(
                1 for r in workspace.market_results
                if r.ef_hierarchy is not None
            )
            hier_score = _safe_divide(
                _D(hier_present), market_count, _ZERO
            )
            findings.append(
                f"EF hierarchy documented: {hier_present}/"
                f"{len(workspace.market_results)} market results"
            )
        else:
            hier_score = _ZERO
            findings.append(
                "No market-based results for EF hierarchy documentation"
            )

        raw_score = (
            TRANSPARENCY_SUB_WEIGHTS["provenance"] * prov_score
            + TRANSPARENCY_SUB_WEIGHTS["ef_source"] * ef_src_score
            + TRANSPARENCY_SUB_WEIGHTS["metadata"] * meta_score
            + TRANSPARENCY_SUB_WEIGHTS["ef_hierarchy_doc"] * hier_score
        )
        final_score = _clamp(_quantize(raw_score))

        logger.debug(
            "Transparency for %s: prov=%.4f, ef_src=%.4f, "
            "meta=%.4f, hier=%.4f => %.8f",
            workspace.reconciliation_id,
            prov_score, ef_src_score, meta_score,
            hier_score, final_score,
        )

        return QualityScore(
            dimension=QualityDimension.TRANSPARENCY,
            score=final_score,
            max_score=_ONE,
            findings=findings,
        )

    # ==================================================================
    # 6. compute_composite_score
    # ==================================================================

    def compute_composite_score(
        self, dimension_scores: Dict[str, QualityScore],
    ) -> Decimal:
        """Compute weighted average composite score from dimension scores.

        Uses QUALITY_WEIGHTS (completeness=0.30, consistency=0.25,
        accuracy=0.25, transparency=0.20).

        Formula:
            composite = sum(score_i * weight_i) / sum(weight_i)

        Args:
            dimension_scores: Dictionary mapping dimension value strings
                to QualityScore objects.

        Returns:
            Weighted composite score as Decimal between 0.0 and 1.0.

        Raises:
            ValueError: If dimension_scores is empty.
        """
        if not dimension_scores:
            raise ValueError(
                "compute_composite_score: dimension_scores must not be empty"
            )

        weights = self.get_quality_weights()
        weighted_sum = _ZERO
        total_weight = _ZERO

        for dim_key, quality_score in dimension_scores.items():
            weight = weights.get(dim_key, _ZERO)
            clamped_score = _clamp(quality_score.score)
            weighted_sum += weight * clamped_score
            total_weight += weight

        if total_weight == _ZERO:
            logger.warning(
                "compute_composite_score: total weight is zero"
            )
            return _ZERO

        composite = _safe_divide(weighted_sum, total_weight, _ZERO)

        # Quantize to configured decimal places
        if (
            self._config is not None
            and hasattr(self._config, "decimal_places")
        ):
            places = self._config.decimal_places
            quantize_str = "0." + "0" * places
            composite = composite.quantize(
                Decimal(quantize_str), rounding=ROUND_HALF_UP
            )
        else:
            composite = _quantize(composite)

        return _clamp(composite)

    # ==================================================================
    # 7. assign_grade
    # ==================================================================

    def assign_grade(self, composite_score: Decimal) -> QualityGrade:
        """Map a composite score to a letter grade.

        Grade thresholds (descending):
            A >= 0.90 (Assurance-Ready)
            B >= 0.80 (High Quality)
            C >= 0.65 (Acceptable)
            D >= 0.50 (Needs Improvement)
            F <  0.50 (Insufficient)

        Args:
            composite_score: Composite score (0.0 to 1.0).

        Returns:
            QualityGrade letter grade.
        """
        clamped = _clamp(composite_score)

        if clamped >= QUALITY_GRADE_THRESHOLDS[QualityGrade.A]:
            return QualityGrade.A
        if clamped >= QUALITY_GRADE_THRESHOLDS[QualityGrade.B]:
            return QualityGrade.B
        if clamped >= QUALITY_GRADE_THRESHOLDS[QualityGrade.C]:
            return QualityGrade.C
        if clamped >= QUALITY_GRADE_THRESHOLDS[QualityGrade.D]:
            return QualityGrade.D
        return QualityGrade.F

    # ==================================================================
    # 8. generate_recommendations
    # ==================================================================

    def generate_recommendations(
        self,
        composite: Decimal,
        grade: QualityGrade,
        assurance_ready: bool,
        dimension_scores: Dict[str, QualityScore],
    ) -> List[str]:
        """Generate improvement recommendations based on low-scoring dimensions.

        Produces targeted string recommendations when dimension scores
        fall below the warning threshold (0.65).

        Args:
            composite: The composite quality score.
            grade: The assigned letter grade.
            assurance_ready: Whether the assessment is assurance-ready.
            dimension_scores: Dictionary of dimension value strings
                to QualityScore objects.

        Returns:
            Ordered list of recommendation strings, most impactful first.
        """
        recs: List[str] = []

        # Completeness recommendations
        qs = dimension_scores.get("completeness")
        if qs and qs.score < FLAG_WARNING_THRESHOLD:
            recs.append(
                "Completeness: Ensure all 4 energy types (electricity, "
                "steam, district heating, district cooling) have results "
                "for both location-based and market-based methods. Verify "
                "all facilities and reporting periods are covered."
            )
        elif qs and qs.score < Decimal("0.80"):
            recs.append(
                "Completeness: Review upstream agent outputs to identify "
                "missing energy type/facility/period combinations."
            )

        # Consistency recommendations
        qs = dimension_scores.get("consistency")
        if qs and qs.score < FLAG_WARNING_THRESHOLD:
            recs.append(
                "Consistency: Harmonise GWP source (use same IPCC AR "
                "version), align organisational boundaries, ensure "
                "identical reporting periods, and standardise EF vintages "
                "across all upstream results."
            )
        elif qs and qs.score < Decimal("0.80"):
            recs.append(
                "Consistency: Review GWP source alignment and temporal "
                "parameters across both methods."
            )

        # Accuracy recommendations
        qs = dimension_scores.get("accuracy")
        if qs and qs.score < FLAG_WARNING_THRESHOLD:
            recs.append(
                "Accuracy: Prioritise sourcing supplier-specific emission "
                "factors with third-party certification for market-based "
                "calculations. Move from Tier 1 to Tier 2/3 data. Verify "
                "emissions = energy_quantity * ef_used for all results."
            )
        elif qs and qs.score < Decimal("0.80"):
            recs.append(
                "Accuracy: Consider moving from grid average or residual "
                "mix EFs to supplier-specific or certified EFs per GHG "
                "Protocol Scope 2 hierarchy."
            )

        # Transparency recommendations
        qs = dimension_scores.get("transparency")
        if qs and qs.score < FLAG_WARNING_THRESHOLD:
            recs.append(
                "Transparency: Ensure all upstream results include SHA-256 "
                "provenance hashes, detailed EF source references with "
                "year and database name, metadata with data origin, and "
                "EF hierarchy documentation for market-based results."
            )
        elif qs and qs.score < Decimal("0.80"):
            recs.append(
                "Transparency: Complete documentation gaps in provenance "
                "hashes, EF source references, and metadata fields."
            )

        # Grade-level recommendation
        if grade == QualityGrade.F:
            recs.append(
                "Overall grade F (Insufficient): Conduct a comprehensive "
                "data quality review across all four dimensions before "
                "submitting for assurance."
            )
        elif grade == QualityGrade.D:
            recs.append(
                "Overall grade D (Needs Improvement): Focus on the "
                "lowest-scoring dimensions to raise the composite above "
                "the acceptable threshold of 0.65."
            )
        elif not assurance_ready:
            recs.append(
                "Not assurance-ready: Target a composite score of 0.80 "
                "or above (grade B) by addressing the recommendations "
                "above."
            )

        return recs

    # ==================================================================
    # 9. generate_quality_flags
    # ==================================================================

    def generate_quality_flags(
        self,
        composite: Decimal,
        grade: QualityGrade,
        assurance_ready: bool,
        dimension_scores: Dict[str, QualityScore],
    ) -> List[Flag]:
        """Generate flags for dimensions below quality thresholds.

        Creates WARNING flags for dimensions scoring below 0.65 and
        ERROR flags for dimensions scoring below 0.50. Also generates
        grade-level and assurance-readiness flags.

        Args:
            composite: The composite quality score.
            grade: The assigned letter grade.
            assurance_ready: Whether the assessment is assurance-ready.
            dimension_scores: Dict of dimension value strings to
                QualityScore objects.

        Returns:
            List of Flag objects sorted by severity (critical first).
        """
        flags: List[Flag] = []
        counter = 1

        # Dimension-level flags
        for dim_key, qs in dimension_scores.items():
            dim_label = dim_key.replace("_", " ").title()

            if qs.score < FLAG_ERROR_THRESHOLD:
                flags.append(Flag(
                    flag_type=FlagType.ERROR,
                    severity=FlagSeverity.HIGH,
                    code=f"{FLAG_CODE_PREFIX}-{counter:03d}",
                    message=(
                        f"{dim_label} score ({qs.score}) is below the "
                        f"error threshold ({FLAG_ERROR_THRESHOLD})"
                    ),
                    recommendation=(
                        f"Critically improve {dim_label.lower()} by "
                        f"addressing all findings"
                    ),
                ))
                counter += 1
            elif qs.score < FLAG_WARNING_THRESHOLD:
                flags.append(Flag(
                    flag_type=FlagType.WARNING,
                    severity=FlagSeverity.MEDIUM,
                    code=f"{FLAG_CODE_PREFIX}-{counter:03d}",
                    message=(
                        f"{dim_label} score ({qs.score}) is below the "
                        f"warning threshold ({FLAG_WARNING_THRESHOLD})"
                    ),
                    recommendation=(
                        f"Improve {dim_label.lower()} to raise score "
                        f"above {FLAG_WARNING_THRESHOLD}"
                    ),
                ))
                counter += 1

        # Grade-level flags
        if grade == QualityGrade.F:
            flags.append(Flag(
                flag_type=FlagType.ERROR,
                severity=FlagSeverity.CRITICAL,
                code=f"{FLAG_CODE_PREFIX}-{counter:03d}",
                message=(
                    f"Quality grade F (Insufficient): composite "
                    f"{composite}. Not suitable for external reporting."
                ),
                recommendation=(
                    "Conduct a comprehensive data quality review before "
                    "resubmitting"
                ),
            ))
            counter += 1
        elif grade == QualityGrade.D:
            flags.append(Flag(
                flag_type=FlagType.WARNING,
                severity=FlagSeverity.HIGH,
                code=f"{FLAG_CODE_PREFIX}-{counter:03d}",
                message=(
                    f"Quality grade D (Needs Improvement): composite "
                    f"{composite}. Significant improvements required."
                ),
                recommendation=(
                    "Focus remediation on the lowest-scoring dimensions"
                ),
            ))
            counter += 1

        # Assurance readiness flag
        if assurance_ready:
            flags.append(Flag(
                flag_type=FlagType.INFO,
                severity=FlagSeverity.LOW,
                code=f"{FLAG_CODE_PREFIX}-{counter:03d}",
                message=(
                    f"Assurance-ready: grade {grade.value} "
                    f"({GRADE_LABELS.get(grade.value, '')}), "
                    f"composite {composite}"
                ),
                recommendation="",
            ))
        else:
            flags.append(Flag(
                flag_type=FlagType.RECOMMENDATION,
                severity=FlagSeverity.MEDIUM,
                code=f"{FLAG_CODE_PREFIX}-{counter:03d}",
                message=(
                    f"Not assurance-ready: grade {grade.value}, "
                    f"composite {composite}. "
                    f"Target >= {self._get_assurance_threshold()}"
                ),
                recommendation=(
                    "Improve data quality to achieve composite score "
                    f">= {self._get_assurance_threshold()} (grade B)"
                ),
            ))

        # Sort by severity
        severity_order = {
            FlagSeverity.CRITICAL: 0,
            FlagSeverity.HIGH: 1,
            FlagSeverity.MEDIUM: 2,
            FlagSeverity.LOW: 3,
        }
        flags.sort(key=lambda f: severity_order.get(f.severity, 99))

        self._increment_flags(len(flags))
        return flags

    # ==================================================================
    # 10. cross_check_emissions
    # ==================================================================

    def cross_check_emissions(
        self, workspace: ReconciliationWorkspace,
    ) -> Tuple[bool, List[str]]:
        """Verify emissions = energy_quantity_mwh * ef_used for each result.

        Performs an arithmetic cross-check on every upstream result
        (both location-based and market-based). A result passes if the
        absolute difference between reported emissions and the computed
        product is within the tolerance of 0.01 tCO2e.

        Args:
            workspace: The reconciliation workspace.

        Returns:
            Tuple of:
                - bool: True if all results pass the cross-check.
                - List[str]: List of issue descriptions for failures.
        """
        self._increment_cross_checks()
        issues: List[str] = []

        all_results = (
            list(workspace.location_results)
            + list(workspace.market_results)
        )

        if not all_results:
            return True, []

        pass_count = 0
        fail_count = 0

        for idx, r in enumerate(all_results):
            expected = _quantize(r.energy_quantity_mwh * r.ef_used)
            actual = r.emissions_tco2e
            difference = abs(actual - expected)

            if difference <= CROSS_CHECK_TOLERANCE:
                pass_count += 1
            else:
                fail_count += 1
                method_label = (
                    r.method.value
                    if hasattr(r.method, "value")
                    else str(r.method)
                )
                energy_label = (
                    r.energy_type.value
                    if hasattr(r.energy_type, "value")
                    else str(r.energy_type)
                )
                issues.append(
                    f"Result #{idx + 1} ({method_label}, "
                    f"{energy_label}, facility={r.facility_id}): "
                    f"reported={actual}, expected={expected}, "
                    f"diff={_quantize(difference)} tCO2e"
                )

        all_passed = fail_count == 0

        logger.info(
            "cross_check_emissions for %s: pass=%d, fail=%d",
            workspace.reconciliation_id,
            pass_count, fail_count,
        )

        return all_passed, issues

    # ==================================================================
    # 11. assess_ef_hierarchy_distribution
    # ==================================================================

    def assess_ef_hierarchy_distribution(
        self, workspace: ReconciliationWorkspace,
    ) -> Dict[str, int]:
        """Count market-based results at each EF hierarchy level.

        Hierarchy levels (GHG Protocol Scope 2):
            - supplier_with_cert (highest quality)
            - supplier_no_cert
            - bundled_cert
            - unbundled_cert
            - residual_mix
            - grid_average (lowest quality)
            - undocumented (ef_hierarchy is None)

        Args:
            workspace: The reconciliation workspace.

        Returns:
            Dictionary mapping hierarchy level strings to counts.
        """
        distribution: Dict[str, int] = defaultdict(int)

        # Initialise known levels to 0
        if _MODELS_AVAILABLE:
            for priority in EFHierarchyPriority:
                distribution[priority.value] = 0
        distribution["undocumented"] = 0

        for r in workspace.market_results:
            if r.ef_hierarchy is not None:
                key = (
                    r.ef_hierarchy.value
                    if hasattr(r.ef_hierarchy, "value")
                    else str(r.ef_hierarchy)
                )
                distribution[key] += 1
            else:
                distribution["undocumented"] += 1

        result_dict = dict(distribution)

        logger.debug(
            "EF hierarchy distribution for %s: %s",
            workspace.reconciliation_id, result_dict,
        )

        return result_dict

    # ==================================================================
    # 12. health_check
    # ==================================================================

    def health_check(self) -> Dict[str, Any]:
        """Return the engine health status.

        Verifies that the engine is properly initialised, config is
        loaded, weights sum to 1.0, and grade thresholds are ordered.

        Returns:
            Dictionary with health status, engine metadata, and
            configuration verification results.
        """
        health: Dict[str, Any] = {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID if _MODELS_AVAILABLE else "GL-MRV-X-024",
            "agent_component": (
                AGENT_COMPONENT if _MODELS_AVAILABLE else "AGENT-MRV-013"
            ),
            "agent_version": VERSION if _MODELS_AVAILABLE else "1.0.0",
            "status": "healthy",
            "models_available": _MODELS_AVAILABLE,
            "config_available": _CONFIG_AVAILABLE,
            "metrics_available": _METRICS_AVAILABLE,
            "provenance_available": _PROVENANCE_AVAILABLE,
            "total_assessments": self._total_assessments,
            "total_cross_checks": self._total_cross_checks,
            "total_flags_generated": self._total_flags_generated,
            "created_at": self._created_at.isoformat(),
            "checks": {},
        }

        try:
            # Check 1: Config loaded
            config_ok = self._config is not None
            health["checks"]["config_loaded"] = config_ok

            # Check 2: Weights sum to 1.0
            weights = self.get_quality_weights()
            weight_sum = sum(weights.values())
            weights_valid = (
                Decimal("0.99") <= weight_sum <= Decimal("1.01")
            )
            health["checks"]["weights_sum_valid"] = weights_valid
            health["checks"]["weights_sum"] = str(weight_sum)

            # Check 3: Grade thresholds ordered
            if _MODELS_AVAILABLE:
                thresholds_ordered = (
                    QUALITY_GRADE_THRESHOLDS[QualityGrade.A]
                    > QUALITY_GRADE_THRESHOLDS[QualityGrade.B]
                    > QUALITY_GRADE_THRESHOLDS[QualityGrade.C]
                    > QUALITY_GRADE_THRESHOLDS[QualityGrade.D]
                    >= QUALITY_GRADE_THRESHOLDS[QualityGrade.F]
                )
                health["checks"]["grade_thresholds_ordered"] = thresholds_ordered

            # Check 4: EF hierarchy scores ordered
            if _MODELS_AVAILABLE:
                ef_scores = list(EF_HIERARCHY_QUALITY_SCORES.values())
                ef_ordered = all(
                    ef_scores[i] >= ef_scores[i + 1]
                    for i in range(len(ef_scores) - 1)
                )
                health["checks"]["ef_hierarchy_ordered"] = ef_ordered

            # Check 5: All four quality dimensions defined
            if _MODELS_AVAILABLE:
                dims_complete = len(QualityDimension) == 4
                health["checks"]["dimensions_complete"] = dims_complete

            # Overall status
            bool_checks = [
                v for v in health["checks"].values()
                if isinstance(v, bool)
            ]
            all_ok = all(bool_checks) if bool_checks else False
            health["status"] = "healthy" if all_ok else "degraded"

            # Uptime
            now = _utcnow()
            uptime = int((now - self._created_at).total_seconds())
            health["uptime_seconds"] = uptime

        except Exception as exc:
            health["status"] = "unhealthy"
            health["error"] = str(exc)
            logger.error(
                "Health check failed: %s", exc, exc_info=True
            )

        return health

    # ==================================================================
    # Extended public methods
    # ==================================================================

    def get_weakest_dimension(
        self, assessment: QualityAssessment,
    ) -> QualityDimension:
        """Identify the weakest quality dimension.

        Returns the dimension with the lowest score. In case of a tie,
        returns the first in weight order (higher weight = higher priority).

        Args:
            assessment: Complete quality assessment.

        Returns:
            The QualityDimension with the lowest score.

        Raises:
            ValueError: If assessment has no scores.
        """
        if not assessment.scores:
            raise ValueError("Assessment has no dimension scores")

        weights = self.get_quality_weights()
        weakest = min(
            assessment.scores,
            key=lambda qs: (
                qs.score,
                -weights.get(qs.dimension.value, _ZERO),
            ),
        )
        return weakest.dimension

    def get_strongest_dimension(
        self, assessment: QualityAssessment,
    ) -> QualityDimension:
        """Identify the strongest quality dimension.

        Args:
            assessment: Complete quality assessment.

        Returns:
            The QualityDimension with the highest score.

        Raises:
            ValueError: If assessment has no scores.
        """
        if not assessment.scores:
            raise ValueError("Assessment has no dimension scores")

        weights = self.get_quality_weights()
        strongest = max(
            assessment.scores,
            key=lambda qs: (
                qs.score,
                weights.get(qs.dimension.value, _ZERO),
            ),
        )
        return strongest.dimension

    def is_assurance_ready(
        self, assessment: QualityAssessment,
    ) -> bool:
        """Determine whether the reconciliation is assurance-ready.

        Assurance-ready if grade is A or B AND no individual dimension
        scores below 0.50.

        Args:
            assessment: Complete quality assessment.

        Returns:
            True if assurance-ready, False otherwise.
        """
        if assessment.grade not in ASSURANCE_READY_GRADES:
            return False

        min_threshold = Decimal("0.50")
        for qs in assessment.scores:
            if qs.score < min_threshold:
                return False

        return True

    def score_per_energy_type(
        self, workspace: ReconciliationWorkspace,
    ) -> Dict[EnergyType, QualityAssessment]:
        """Score quality per energy type.

        Creates a filtered workspace for each energy type and runs
        the full quality assessment.

        Args:
            workspace: Populated reconciliation workspace.

        Returns:
            Map of EnergyType to QualityAssessment.
        """
        result: Dict[EnergyType, QualityAssessment] = {}

        for energy_type in EnergyType:
            loc_results = [
                r for r in workspace.location_results
                if r.energy_type == energy_type
            ]
            mkt_results = [
                r for r in workspace.market_results
                if r.energy_type == energy_type
            ]

            if not loc_results and not mkt_results:
                continue

            loc_total = sum(
                (r.emissions_tco2e for r in loc_results), _ZERO
            )
            mkt_total = sum(
                (r.emissions_tco2e for r in mkt_results), _ZERO
            )

            filtered_breakdowns = [
                b for b in workspace.by_energy_type
                if b.energy_type == energy_type
            ]

            filtered_ws = ReconciliationWorkspace(
                reconciliation_id=(
                    f"{workspace.reconciliation_id}-{energy_type.value}"
                ),
                tenant_id=workspace.tenant_id,
                period_start=workspace.period_start,
                period_end=workspace.period_end,
                location_results=loc_results,
                market_results=mkt_results,
                total_location_tco2e=loc_total,
                total_market_tco2e=mkt_total,
                by_energy_type=filtered_breakdowns,
                by_facility=[],
            )

            try:
                assessment = self.score_quality(filtered_ws)
                result[energy_type] = assessment
            except ValueError:
                logger.debug(
                    "Skipping quality for energy type %s (no results)",
                    energy_type.value,
                )

        return result

    def score_per_facility(
        self, workspace: ReconciliationWorkspace,
    ) -> Dict[str, QualityAssessment]:
        """Score quality per facility.

        Creates a filtered workspace for each unique facility and runs
        the full quality assessment.

        Args:
            workspace: Populated reconciliation workspace.

        Returns:
            Map of facility_id to QualityAssessment.
        """
        result: Dict[str, QualityAssessment] = {}

        all_facilities: Set[str] = set()
        for r in workspace.location_results:
            all_facilities.add(r.facility_id)
        for r in workspace.market_results:
            all_facilities.add(r.facility_id)

        for fac_id in sorted(all_facilities):
            loc = [
                r for r in workspace.location_results
                if r.facility_id == fac_id
            ]
            mkt = [
                r for r in workspace.market_results
                if r.facility_id == fac_id
            ]

            if not loc and not mkt:
                continue

            loc_total = sum((r.emissions_tco2e for r in loc), _ZERO)
            mkt_total = sum((r.emissions_tco2e for r in mkt), _ZERO)

            fac_breakdowns = [
                b for b in workspace.by_facility
                if b.facility_id == fac_id
            ]

            filtered_ws = ReconciliationWorkspace(
                reconciliation_id=(
                    f"{workspace.reconciliation_id}-{fac_id}"
                ),
                tenant_id=workspace.tenant_id,
                period_start=workspace.period_start,
                period_end=workspace.period_end,
                location_results=loc,
                market_results=mkt,
                total_location_tco2e=loc_total,
                total_market_tco2e=mkt_total,
                by_energy_type=[],
                by_facility=fac_breakdowns,
            )

            try:
                assessment = self.score_quality(filtered_ws)
                result[fac_id] = assessment
            except ValueError:
                logger.debug(
                    "Skipping quality for facility %s (no results)",
                    fac_id,
                )

        return result

    def summarize_quality(
        self, assessment: QualityAssessment,
    ) -> Dict[str, Any]:
        """Generate a human-readable quality summary.

        Produces a dictionary suitable for JSON serialisation or
        inclusion in reporting tables.

        Args:
            assessment: Complete quality assessment.

        Returns:
            Dictionary with quality summary fields.
        """
        dim_scores: Dict[str, Any] = {}
        for qs in assessment.scores:
            dim_scores[qs.dimension.value] = {
                "score": str(qs.score),
                "max_score": str(qs.max_score),
                "percentage": str(
                    _quantize(qs.score / qs.max_score * _HUNDRED)
                    if qs.max_score > _ZERO
                    else _ZERO
                ),
                "finding_count": len(qs.findings),
                "findings": qs.findings,
            }

        weakest = None
        strongest = None
        if assessment.scores:
            weakest = self.get_weakest_dimension(assessment).value
            strongest = self.get_strongest_dimension(assessment).value

        weights = self.get_quality_weights()
        weights_str = {k: str(v) for k, v in weights.items()}

        summary: Dict[str, Any] = {
            "reconciliation_id": assessment.reconciliation_id,
            "composite_score": str(assessment.composite_score),
            "composite_percentage": str(
                _quantize(assessment.composite_score * _HUNDRED)
            ),
            "grade": assessment.grade.value,
            "grade_label": GRADE_LABELS.get(
                assessment.grade.value, "Unknown"
            ),
            "assurance_ready": assessment.assurance_ready,
            "assurance_ready_strict": self.is_assurance_ready(assessment),
            "dimension_scores": dim_scores,
            "dimension_weights": weights_str,
            "weakest_dimension": weakest,
            "strongest_dimension": strongest,
            "total_findings": len(assessment.findings),
            "finding_summary": assessment.findings[:10],
            "grade_thresholds": {
                g.value: str(t)
                for g, t in QUALITY_GRADE_THRESHOLDS.items()
            } if _MODELS_AVAILABLE else {},
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
        }

        summary["provenance_hash"] = _hash_dict(summary)
        return summary

    @staticmethod
    def get_engine_id() -> str:
        """Return the engine identifier.

        Returns:
            Engine ID string.
        """
        return ENGINE_ID

    @staticmethod
    def get_engine_version() -> str:
        """Return the engine version string.

        Returns:
            Engine version string.
        """
        return ENGINE_VERSION

    # ==================================================================
    # Private helpers -- Completeness sub-scoring
    # ==================================================================

    def _score_facility_coverage(
        self,
        workspace: ReconciliationWorkspace,
        findings: List[str],
    ) -> Decimal:
        """Score facility-level dual-method coverage weighted by emissions.

        Facilities with both methods get full weight. Facilities with
        only one method get 50% weight. Score is emission-weighted.

        Args:
            workspace: Reconciliation workspace.
            findings: Mutable list for appending findings.

        Returns:
            Facility coverage score (0.0 to 1.0).
        """
        loc_fac: Dict[str, Decimal] = {}
        for r in workspace.location_results:
            loc_fac.setdefault(r.facility_id, _ZERO)
            loc_fac[r.facility_id] += r.emissions_tco2e

        mkt_fac: Dict[str, Decimal] = {}
        for r in workspace.market_results:
            mkt_fac.setdefault(r.facility_id, _ZERO)
            mkt_fac[r.facility_id] += r.emissions_tco2e

        all_fac_ids = set(loc_fac.keys()) | set(mkt_fac.keys())
        if not all_fac_ids:
            findings.append("No facilities found in upstream results")
            return _ZERO

        total_emissions = _ZERO
        for fid in all_fac_ids:
            total_emissions += loc_fac.get(fid, _ZERO)
            total_emissions += mkt_fac.get(fid, _ZERO)

        both_count = 0
        weighted_coverage = _ZERO

        for fid in all_fac_ids:
            has_loc = fid in loc_fac
            has_mkt = fid in mkt_fac
            fac_emissions = (
                loc_fac.get(fid, _ZERO) + mkt_fac.get(fid, _ZERO)
            )
            fac_weight = _safe_divide(fac_emissions, total_emissions, _ZERO)

            if has_loc and has_mkt:
                both_count += 1
                weighted_coverage += fac_weight
            else:
                weighted_coverage += fac_weight * Decimal("0.50")

        total_fac = len(all_fac_ids)
        pct = _safe_divide(
            _D(both_count) * _HUNDRED, _D(total_fac), _ZERO,
        )
        findings.append(
            f"Facility coverage: {both_count}/{total_fac} "
            f"({_quantize(pct)}%) have both methods"
        )

        loc_only = set(loc_fac.keys()) - set(mkt_fac.keys())
        mkt_only = set(mkt_fac.keys()) - set(loc_fac.keys())
        if loc_only:
            sample = sorted(loc_only)[:3]
            findings.append(
                f"{len(loc_only)} facility(ies) location-based only "
                f"(e.g. {', '.join(sample)})"
            )
        if mkt_only:
            sample = sorted(mkt_only)[:3]
            findings.append(
                f"{len(mkt_only)} facility(ies) market-based only "
                f"(e.g. {', '.join(sample)})"
            )

        return _clamp(_quantize(weighted_coverage))

    def _score_period_coverage(
        self,
        workspace: ReconciliationWorkspace,
        findings: List[str],
    ) -> Decimal:
        """Score temporal coverage and detect period gaps.

        Checks whether the workspace's reporting period is fully covered
        by upstream results. A result passes if its period matches or
        encompasses the workspace period.

        Args:
            workspace: Reconciliation workspace.
            findings: Mutable list for appending findings.

        Returns:
            Period coverage score (0.0 to 1.0).
        """
        all_results = (
            list(workspace.location_results)
            + list(workspace.market_results)
        )
        if not all_results:
            findings.append("No results for period coverage check")
            return _ZERO

        ws_start = workspace.period_start
        ws_end = workspace.period_end

        coverage_count = 0
        for r in all_results:
            if r.period_start <= ws_start and r.period_end >= ws_end:
                coverage_count += 1
            elif (
                r.period_start == ws_start
                and r.period_end == ws_end
            ):
                coverage_count += 1

        total = len(all_results)
        gap_count = total - coverage_count

        if gap_count == 0:
            findings.append(
                f"All {total} results cover the full reporting period"
            )
        else:
            findings.append(
                f"Period coverage: {coverage_count}/{total} results "
                f"cover the full period; {gap_count} have gaps"
            )

        return _clamp(
            _safe_divide(_D(coverage_count), _D(total), _ZERO)
        )

    # ==================================================================
    # Private helpers -- Consistency sub-scoring
    # ==================================================================

    def _score_gwp_consistency(
        self, results: List[Any], findings: List[str],
    ) -> Decimal:
        """Score GWP source consistency. 1.0 if all same, otherwise majority fraction.

        Args:
            results: All upstream results.
            findings: Mutable findings list.

        Returns:
            Score 0.0 to 1.0.
        """
        gwp_counts: Dict[str, int] = defaultdict(int)
        for r in results:
            src = (
                r.gwp_source.value
                if hasattr(r.gwp_source, "value")
                else str(r.gwp_source)
            )
            gwp_counts[src] += 1

        if len(gwp_counts) <= 1:
            src_name = list(gwp_counts.keys())[0] if gwp_counts else "N/A"
            findings.append(
                f"GWP source consistent: all {len(results)} results "
                f"use {src_name}"
            )
            return _ONE

        total = sum(gwp_counts.values())
        dominant = max(gwp_counts.values())
        score = _safe_divide(_D(dominant), _D(total), _ZERO)

        summary = ", ".join(f"{k}: {v}" for k, v in sorted(gwp_counts.items()))
        findings.append(
            f"GWP inconsistency: {len(gwp_counts)} sources ({summary})"
        )
        return _quantize(score)

    def _score_period_consistency(
        self, results: List[Any], findings: List[str],
    ) -> Decimal:
        """Score reporting period consistency. 1.0 if all same.

        Args:
            results: All upstream results.
            findings: Mutable findings list.

        Returns:
            Score 0.0 to 1.0.
        """
        periods: Dict[str, int] = defaultdict(int)
        for r in results:
            key = f"{r.period_start}|{r.period_end}"
            periods[key] += 1

        if len(periods) <= 1:
            findings.append(
                f"Reporting period consistent across all {len(results)} results"
            )
            return _ONE

        total = sum(periods.values())
        dominant = max(periods.values())
        score = _safe_divide(_D(dominant), _D(total), _ZERO)

        findings.append(
            f"Period inconsistency: {len(periods)} different periods detected"
        )
        return _quantize(score)

    def _score_boundary_consistency(
        self, results: List[Any], findings: List[str],
    ) -> Decimal:
        """Score organisational boundary consistency (tenant_id).

        Args:
            results: All upstream results.
            findings: Mutable findings list.

        Returns:
            Score 0.0 to 1.0.
        """
        tenants: Dict[str, int] = defaultdict(int)
        for r in results:
            tenants[r.tenant_id] += 1

        if len(tenants) <= 1:
            tid = list(tenants.keys())[0] if tenants else "N/A"
            findings.append(
                f"Organisational boundary consistent: tenant_id '{tid}'"
            )
            return _ONE

        total = sum(tenants.values())
        dominant = max(tenants.values())
        score = _safe_divide(_D(dominant), _D(total), _ZERO)

        findings.append(
            f"Boundary inconsistency: {len(tenants)} tenant_ids detected"
        )
        return _quantize(score)

    def _score_ef_vintage_consistency(
        self, results: List[Any], findings: List[str],
    ) -> Decimal:
        """Score EF vintage consistency across all results.

        Extracts year from ef_source strings and checks uniformity.

        Args:
            results: All upstream results.
            findings: Mutable findings list.

        Returns:
            Score 0.0 to 1.0.
        """
        vintages: Dict[str, int] = defaultdict(int)
        undetermined = 0

        for r in results:
            year = _extract_vintage_year(r.ef_source)
            if year is not None:
                vintages[year] += 1
            else:
                undetermined += 1

        total_with_vintage = sum(vintages.values())

        if total_with_vintage == 0:
            findings.append(
                "EF vintage could not be determined from ef_source strings"
            )
            return Decimal("0.50")

        if len(vintages) <= 1:
            year_str = list(vintages.keys())[0]
            findings.append(
                f"EF vintage consistent: all determinable results "
                f"reference {year_str}"
            )
            if undetermined > 0:
                findings.append(
                    f"{undetermined} result(s) have undetermined EF vintage"
                )
                penalty = _safe_divide(
                    _D(undetermined),
                    _D(total_with_vintage + undetermined),
                    _ZERO,
                )
                return _clamp(
                    _quantize(_ONE - penalty * Decimal("0.20"))
                )
            return _ONE

        dominant = max(vintages.values())
        score = _safe_divide(_D(dominant), _D(total_with_vintage), _ZERO)

        summary = ", ".join(f"{y}: {c}" for y, c in sorted(vintages.items()))
        findings.append(
            f"EF vintage inconsistency: {len(vintages)} vintages ({summary})"
        )
        if undetermined > 0:
            findings.append(
                f"{undetermined} result(s) have undetermined EF vintage"
            )

        return _quantize(score)

    # ==================================================================
    # Private helpers -- Accuracy sub-scoring
    # ==================================================================

    def _score_ef_hierarchy_quality(
        self, market_results: List[Any], findings: List[str],
    ) -> Decimal:
        """Average EF hierarchy quality score for market-based results.

        Maps ef_hierarchy to EF_HIERARCHY_QUALITY_SCORES. Results
        without ef_hierarchy scored at grid_average (0.20).

        Args:
            market_results: Market-based upstream results.
            findings: Mutable findings list.

        Returns:
            Score 0.0 to 1.0.
        """
        if not market_results:
            findings.append(
                "No market results for EF hierarchy quality scoring"
            )
            return _ZERO

        total_quality = _ZERO
        hierarchy_counts: Dict[str, int] = defaultdict(int)

        for r in market_results:
            if r.ef_hierarchy is not None and _MODELS_AVAILABLE:
                quality = EF_HIERARCHY_QUALITY_SCORES.get(
                    r.ef_hierarchy, Decimal("0.20"),
                )
                key = (
                    r.ef_hierarchy.value
                    if hasattr(r.ef_hierarchy, "value")
                    else str(r.ef_hierarchy)
                )
            else:
                quality = Decimal("0.20")
                key = "undocumented"

            total_quality += quality
            hierarchy_counts[key] += 1

        avg_quality = _safe_divide(
            total_quality, _D(len(market_results)), _ZERO,
        )

        summary = ", ".join(
            f"{k}: {v}" for k, v in sorted(hierarchy_counts.items())
        )
        findings.append(
            f"EF hierarchy quality: avg {_quantize(avg_quality)} "
            f"across {len(market_results)} market results ({summary})"
        )

        if avg_quality < Decimal("0.50"):
            findings.append(
                "EF hierarchy quality is low; most market results use "
                "residual mix or grid average factors"
            )

        return _clamp(_quantize(avg_quality))

    def _score_data_quality_tier(
        self, results: List[Any], findings: List[str],
    ) -> Decimal:
        """Average data quality tier score. Tier3=1.0, Tier2=0.7, Tier1=0.4.

        Args:
            results: All upstream results.
            findings: Mutable findings list.

        Returns:
            Score 0.0 to 1.0.
        """
        if not results:
            findings.append("No results for tier quality scoring")
            return _ZERO

        total_tier = _ZERO
        tier_counts: Dict[str, int] = defaultdict(int)

        for r in results:
            tier_key = (
                r.tier.value if hasattr(r.tier, "value") else str(r.tier)
            )
            tier_score = TIER_QUALITY_SCORES.get(tier_key, Decimal("0.40"))
            total_tier += tier_score
            tier_counts[tier_key] += 1

        avg_tier = _safe_divide(total_tier, _D(len(results)), _ZERO)

        summary = ", ".join(
            f"{k}: {v}" for k, v in sorted(tier_counts.items())
        )
        findings.append(
            f"Data tier quality: avg {_quantize(avg_tier)} "
            f"across {len(results)} results ({summary})"
        )

        if avg_tier < Decimal("0.60"):
            findings.append(
                "Data quality tiers predominantly Tier 1 (global defaults)"
            )

        return _clamp(_quantize(avg_tier))

    def _score_cross_check_rate(
        self, results: List[Any], findings: List[str],
    ) -> Decimal:
        """Score pass rate of emissions == MWh * EF cross-checks.

        Args:
            results: All upstream results.
            findings: Mutable findings list.

        Returns:
            Score 0.0 to 1.0.
        """
        if not results:
            findings.append("No results for cross-check scoring")
            return _ZERO

        pass_count = 0
        for r in results:
            expected = _quantize(r.energy_quantity_mwh * r.ef_used)
            diff = abs(r.emissions_tco2e - expected)
            if diff <= CROSS_CHECK_TOLERANCE:
                pass_count += 1

        total = len(results)
        fail_count = total - pass_count
        rate = _safe_divide(_D(pass_count), _D(total), _ZERO)

        if fail_count == 0:
            findings.append(
                f"All {total} results pass arithmetic cross-check"
            )
        else:
            findings.append(
                f"Cross-check: {pass_count}/{total} pass, "
                f"{fail_count} fail"
            )

        return _clamp(_quantize(rate))

    # ==================================================================
    # Private helpers -- Metrics and provenance
    # ==================================================================

    def _record_quality_metrics(
        self, assessment: QualityAssessment, elapsed_s: float,
    ) -> None:
        """Record Prometheus metrics for a quality assessment.

        Args:
            assessment: The completed quality assessment.
            elapsed_s: Time taken for the assessment in seconds.
        """
        if self._metrics is None:
            return

        try:
            for qs in assessment.scores:
                self._metrics.record_quality_score(
                    dimension=qs.dimension.value,
                    score=float(qs.score),
                    grade=assessment.grade.value,
                )
            self._metrics.record_quality_score(
                dimension="composite",
                score=float(assessment.composite_score),
                grade=assessment.grade.value,
            )
        except Exception:
            logger.warning(
                "Failed to record quality metrics", exc_info=True,
            )

    def _record_provenance_stages(
        self,
        reconciliation_id: str,
        composite: Decimal,
        grade: QualityGrade,
    ) -> None:
        """Record provenance stages for quality scoring.

        Args:
            reconciliation_id: The reconciliation run ID.
            composite: The composite score.
            grade: The assigned grade.
        """
        if self._provenance is None or _ProvenanceStage is None:
            return

        stage_names = [
            "SCORE_COMPLETENESS",
            "SCORE_CONSISTENCY",
            "SCORE_ACCURACY",
            "SCORE_TRANSPARENCY",
        ]

        for stage_name in stage_names:
            stage_enum = getattr(_ProvenanceStage, stage_name, None)
            if stage_enum is not None:
                try:
                    self._provenance.add_stage(
                        chain_id=reconciliation_id,
                        stage=stage_enum,
                        metadata={
                            "composite_score": str(composite),
                            "grade": grade.value,
                        },
                        output_data=str(composite),
                    )
                except (ValueError, KeyError):
                    pass  # Chain may not exist yet
                except Exception as exc:
                    logger.debug(
                        "Provenance recording skipped: %s", exc,
                    )


# ===========================================================================
# Module-level convenience functions
# ===========================================================================


def score_quality(
    workspace: ReconciliationWorkspace,
) -> QualityAssessment:
    """Score data quality for a reconciliation workspace.

    Module-level convenience function that delegates to the
    QualityScorerEngine singleton.

    Args:
        workspace: Populated reconciliation workspace.

    Returns:
        Complete QualityAssessment.

    Example:
        >>> from greenlang.agents.mrv.dual_reporting_reconciliation.quality_scorer import (
        ...     score_quality,
        ... )
        >>> assessment = score_quality(workspace)
        >>> print(assessment.grade.value)
        A
    """
    engine = QualityScorerEngine()
    return engine.score_quality(workspace)


def get_quality_weights() -> Dict[str, Decimal]:
    """Get the configured quality dimension weights.

    Module-level convenience function.

    Returns:
        Map of dimension value string to weight.
    """
    engine = QualityScorerEngine()
    return engine.get_quality_weights()


def assign_grade(composite_score: Decimal) -> QualityGrade:
    """Assign a letter grade from a composite quality score.

    Module-level convenience function.

    Args:
        composite_score: Composite score (0.0 to 1.0).

    Returns:
        QualityGrade letter grade.
    """
    engine = QualityScorerEngine()
    return engine.assign_grade(composite_score)


def is_assurance_ready(assessment: QualityAssessment) -> bool:
    """Check whether a quality assessment is assurance-ready.

    Module-level convenience function.

    Args:
        assessment: Complete quality assessment.

    Returns:
        True if assurance-ready.
    """
    engine = QualityScorerEngine()
    return engine.is_assurance_ready(assessment)


def summarize_quality(assessment: QualityAssessment) -> Dict[str, Any]:
    """Generate a human-readable quality summary.

    Module-level convenience function.

    Args:
        assessment: Complete quality assessment.

    Returns:
        Dictionary with quality summary.
    """
    engine = QualityScorerEngine()
    return engine.summarize_quality(assessment)


def cross_check_emissions(
    workspace: ReconciliationWorkspace,
) -> Tuple[bool, List[str]]:
    """Cross-check emissions = MWh * EF for all results.

    Module-level convenience function.

    Args:
        workspace: Reconciliation workspace.

    Returns:
        Tuple of (all_passed, issue_list).
    """
    engine = QualityScorerEngine()
    return engine.cross_check_emissions(workspace)


def assess_ef_hierarchy_distribution(
    workspace: ReconciliationWorkspace,
) -> Dict[str, int]:
    """Assess EF hierarchy distribution for market-based results.

    Module-level convenience function.

    Args:
        workspace: Reconciliation workspace.

    Returns:
        Dict of hierarchy level to count.
    """
    engine = QualityScorerEngine()
    return engine.assess_ef_hierarchy_distribution(workspace)


# ===========================================================================
# Exports
# ===========================================================================


__all__ = [
    # Engine class
    "QualityScorerEngine",
    # Module-level convenience functions
    "score_quality",
    "get_quality_weights",
    "assign_grade",
    "is_assurance_ready",
    "summarize_quality",
    "cross_check_emissions",
    "assess_ef_hierarchy_distribution",
    # Constants
    "ENGINE_ID",
    "ENGINE_VERSION",
    "SCORE_FLOOR",
    "SCORE_CEILING",
    "COMPLETENESS_SUB_WEIGHTS",
    "CONSISTENCY_SUB_WEIGHTS",
    "ACCURACY_SUB_WEIGHTS",
    "TRANSPARENCY_SUB_WEIGHTS",
    "TIER_QUALITY_SCORES",
    "CROSS_CHECK_TOLERANCE",
    "FLAG_WARNING_THRESHOLD",
    "FLAG_ERROR_THRESHOLD",
    "FLAG_CODE_PREFIX",
    "GRADE_LABELS",
]
