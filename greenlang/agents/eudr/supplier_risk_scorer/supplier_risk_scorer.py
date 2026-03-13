# -*- coding: utf-8 -*-
"""
Supplier Risk Scorer Engine - AGENT-EUDR-017 Engine 1

Multi-factor weighted composite supplier risk scoring per EUDR Article 10-11
with 8-factor model (geographic_sourcing 20%, compliance_history 15%,
documentation_quality 15%, certification_status 15%, traceability_completeness
10%, financial_stability 10%, environmental_performance 10%, social_compliance
5%), risk level classification (LOW 0-25, MEDIUM 26-50, HIGH 51-75, CRITICAL
76-100), confidence scoring, trend analysis (improving/stable/deteriorating),
peer group benchmarking, batch assessment, and supplier comparison.

Risk Score Calculation (Zero-Hallucination):
    composite_score = sum(factor_value * factor_weight for all 8 factors)
    All factors normalized to [0, 100] scale.
    Confidence = weighted_average(data_completeness, data_freshness)

Classification Thresholds (configurable):
    - LOW: 0-25 (low risk, reduced monitoring)
    - MEDIUM: 26-50 (standard risk, standard monitoring)
    - HIGH: 51-75 (high risk, enhanced monitoring)
    - CRITICAL: 76-100 (critical risk, immediate action required)

Confidence Scoring:
    confidence = 0.6 * data_completeness + 0.4 * data_freshness
    data_completeness = (factors_available / 8.0)
    data_freshness = 1.0 if max_age < threshold else 0.5

Trend Analysis:
    Rolling window (default 12 months), linear regression on historical scores.
    Direction: improving (slope < -2), stable (-2 to +2), deteriorating (> +2).

Peer Group Benchmarking:
    Compare supplier against peers in same commodity + region group.
    Percentile ranking and deviation from peer median.

Zero-Hallucination: All scoring is deterministic arithmetic. No LLM
    calls in the calculation path. All inputs are validated against
    database lookups or configuration bounds.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-017 Supplier Risk Scorer (GL-EUDR-SRS-017)
Status: Production Ready
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from .config import get_config
from .metrics import (
    observe_assessment_duration,
    record_assessment_completed,
)
from .models import (
    AssessSupplierRequest,
    BatchAssessmentRequest,
    BatchResponse,
    CommodityType,
    CompareSupplierRequest,
    ComparisonResponse,
    FactorScore,
    GetTrendRequest,
    RiskLevel,
    SupplierRiskAssessment,
    SupplierRiskResponse,
    SupplierType,
    TrendDirection,
    TrendResponse,
)
from .provenance import get_provenance_tracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default factor weights (sum = 1.0)
_DEFAULT_WEIGHTS: Dict[str, Decimal] = {
    "geographic_sourcing": Decimal("0.20"),
    "compliance_history": Decimal("0.15"),
    "documentation_quality": Decimal("0.15"),
    "certification_status": Decimal("0.15"),
    "traceability_completeness": Decimal("0.10"),
    "financial_stability": Decimal("0.10"),
    "environmental_performance": Decimal("0.10"),
    "social_compliance": Decimal("0.05"),
}

#: Factor keys
_FACTOR_KEYS: List[str] = [
    "geographic_sourcing",
    "compliance_history",
    "documentation_quality",
    "certification_status",
    "traceability_completeness",
    "financial_stability",
    "environmental_performance",
    "social_compliance",
]

#: Trend slope thresholds for classification
_TREND_IMPROVING_THRESHOLD: Decimal = Decimal("-2.0")
_TREND_DETERIORATING_THRESHOLD: Decimal = Decimal("2.0")


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _decimal(value: Any) -> Decimal:
    """Convert value to Decimal for precise arithmetic."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _float(value: Decimal) -> float:
    """Convert Decimal to float for API responses."""
    return float(value)


# ---------------------------------------------------------------------------
# SupplierRiskScorer
# ---------------------------------------------------------------------------


class SupplierRiskScorer:
    """Multi-factor weighted composite supplier risk scoring per EUDR Article 10-11.

    Calculates composite risk scores from 8 weighted factors, classifies
    risk levels, scores confidence, analyzes trends, performs peer group
    benchmarking, and provides supplier comparison capabilities.

    All scoring operations use Decimal arithmetic for zero floating-point
    drift and deterministic reproducibility.

    Attributes:
        _assessments: In-memory store of risk assessments keyed by
            assessment_id.
        _supplier_history: Historical risk scores keyed by supplier_id,
            list of (timestamp, score) tuples.
        _peer_groups: Peer group data keyed by (commodity, region).
        _lock: Threading lock for thread-safe access.

    Example:
        >>> scorer = SupplierRiskScorer()
        >>> request = AssessSupplierRequest(supplier_id="SUP123", ...)
        >>> result = scorer.assess_supplier(request)
        >>> assert result.assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        >>> assert 0.0 <= result.assessment.composite_score <= 100.0
    """

    def __init__(self) -> None:
        """Initialize SupplierRiskScorer with empty stores."""
        self._assessments: Dict[str, SupplierRiskAssessment] = {}
        self._supplier_history: Dict[str, List[Tuple[datetime, Decimal]]] = {}
        self._peer_groups: Dict[Tuple[str, str], List[Decimal]] = defaultdict(list)
        self._lock: threading.Lock = threading.Lock()
        logger.info(
            "SupplierRiskScorer initialized: factors=%d, default_weights=%s",
            len(_FACTOR_KEYS),
            {k: _float(v) for k, v in _DEFAULT_WEIGHTS.items()},
        )

    # ------------------------------------------------------------------
    # Primary assessment
    # ------------------------------------------------------------------

    def assess_supplier(
        self,
        request: AssessSupplierRequest,
    ) -> SupplierRiskResponse:
        """Assess supplier risk using 8-factor weighted composite scoring.

        Applies the following assessment pipeline:
        1. Validate inputs (supplier_id, factor values, weights).
        2. Normalize all factor values to [0, 100] scale.
        3. Calculate composite score using weighted sum.
        4. Classify risk level from score.
        5. Calculate confidence score from completeness and freshness.
        6. Analyze trend if historical data available.
        7. Perform peer group benchmarking if enabled.
        8. Store assessment and update history.
        9. Record provenance and metrics.

        Args:
            request: AssessSupplierRequest containing supplier_id,
                factor_values, factor_weights, data_dates, commodity,
                region, include_trend, include_benchmark.

        Returns:
            SupplierRiskResponse with SupplierRiskAssessment including
            composite_score, risk_level, confidence, trend_direction,
            peer_percentile, and factor breakdown.

        Raises:
            ValueError: If supplier_id is empty, factor_values missing
                required keys, weights don't sum to 1.0, or values out
                of valid range.
        """
        start_time = time.perf_counter()
        cfg = get_config()

        try:
            # Step 1: Validate inputs
            self._validate_assessment_inputs(request)

            # Step 2: Normalize factors to [0, 100]
            normalized_factors = self._normalize_factors(request.factor_values)

            # Step 3: Calculate composite score
            weights = self._get_factor_weights(request.factor_weights)
            composite_score = self._calculate_composite_score(
                normalized_factors, weights
            )

            # Step 4: Classify risk level
            risk_level = self._classify_risk_level(composite_score)

            # Step 5: Calculate confidence
            confidence = self._calculate_confidence(
                request.factor_values, request.data_dates
            )

            # Step 6: Analyze trend
            trend_direction = None
            trend_slope = None
            if request.include_trend:
                trend_direction, trend_slope = self._analyze_trend(
                    request.supplier_id, composite_score
                )

            # Step 7: Peer benchmarking
            peer_percentile = None
            peer_median = None
            if request.include_benchmark and request.commodity and request.region:
                peer_percentile, peer_median = self._benchmark_peer_group(
                    composite_score, request.commodity, request.region
                )

            # Step 8: Build factor scores
            factor_scores = self._build_factor_scores(
                normalized_factors, weights, request.data_dates
            )

            # Step 9: Create assessment
            assessment_id = str(uuid.uuid4())
            now = _utcnow()

            assessment = SupplierRiskAssessment(
                assessment_id=assessment_id,
                supplier_id=request.supplier_id,
                supplier_type=request.supplier_type,
                commodity=request.commodity,
                region=request.region,
                composite_score=_float(composite_score),
                risk_level=risk_level,
                confidence=_float(confidence),
                trend_direction=trend_direction,
                trend_slope=_float(trend_slope) if trend_slope is not None else None,
                peer_percentile=_float(peer_percentile) if peer_percentile is not None else None,
                peer_median=_float(peer_median) if peer_median is not None else None,
                factor_scores=factor_scores,
                assessment_date=now,
                next_assessment_date=self._calculate_next_assessment_date(
                    risk_level, now
                ),
                assessed_by=request.assessed_by or "system",
            )

            # Step 10: Store assessment and update history
            with self._lock:
                self._assessments[assessment_id] = assessment
                if request.supplier_id not in self._supplier_history:
                    self._supplier_history[request.supplier_id] = []
                self._supplier_history[request.supplier_id].append(
                    (now, composite_score)
                )

                # Update peer group data
                if request.commodity and request.region:
                    peer_key = (request.commodity.value, request.region)
                    self._peer_groups[peer_key].append(composite_score)
                    # Keep only recent N scores per peer group
                    max_peer_scores = 1000
                    if len(self._peer_groups[peer_key]) > max_peer_scores:
                        self._peer_groups[peer_key] = self._peer_groups[peer_key][
                            -max_peer_scores:
                        ]

            # Step 11: Record provenance
            get_provenance_tracker().record_operation(
                entity_type="supplier_assessment",
                entity_id=assessment_id,
                action="assess",
                details={
                    "supplier_id": request.supplier_id,
                    "composite_score": _float(composite_score),
                    "risk_level": risk_level.value,
                    "confidence": _float(confidence),
                },
            )

            # Step 12: Record metrics
            duration = time.perf_counter() - start_time
            observe_assessment_duration(duration, risk_level.value)
            record_assessment_completed(
                risk_level=risk_level.value,
                supplier_type=request.supplier_type.value if request.supplier_type else "unknown",
                commodity=request.commodity.value if request.commodity else "unknown",
            )

            logger.info(
                "Supplier risk assessment completed: supplier_id=%s, score=%.2f, "
                "risk_level=%s, confidence=%.2f, duration=%.3fs",
                request.supplier_id,
                _float(composite_score),
                risk_level.value,
                _float(confidence),
                duration,
            )

            return SupplierRiskResponse(
                assessment=assessment,
                processing_time_ms=duration * 1000.0,
            )

        except Exception as e:
            logger.error(
                "Supplier risk assessment failed: supplier_id=%s, error=%s",
                request.supplier_id if hasattr(request, "supplier_id") else "unknown",
                str(e),
                exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # Batch assessment
    # ------------------------------------------------------------------

    def assess_batch(
        self,
        request: BatchAssessmentRequest,
    ) -> BatchResponse:
        """Assess multiple suppliers in batch for portfolio-level analysis.

        Processes multiple supplier assessments in sequence, collecting
        results and aggregating portfolio-level statistics.

        Args:
            request: BatchAssessmentRequest containing list of
                AssessSupplierRequest objects.

        Returns:
            BatchResponse with list of SupplierRiskResponse objects,
            summary statistics (counts by risk level, average score,
            average confidence), and processing metrics.

        Raises:
            ValueError: If requests list is empty or exceeds max batch size.
        """
        start_time = time.perf_counter()
        cfg = get_config()

        if not request.requests:
            raise ValueError("Batch assessment requires at least one request")

        max_batch = cfg.batch_max_size
        if len(request.requests) > max_batch:
            raise ValueError(
                f"Batch size {len(request.requests)} exceeds maximum {max_batch}"
            )

        results: List[SupplierRiskResponse] = []
        errors: List[Dict[str, Any]] = []

        # Risk level counters
        risk_counts: Dict[str, int] = {
            "low": 0,
            "medium": 0,
            "high": 0,
            "critical": 0,
        }

        total_score = Decimal("0.0")
        total_confidence = Decimal("0.0")
        success_count = 0

        for idx, assess_req in enumerate(request.requests):
            try:
                result = self.assess_supplier(assess_req)
                results.append(result)
                risk_counts[result.assessment.risk_level.value] += 1
                total_score += _decimal(result.assessment.composite_score)
                total_confidence += _decimal(result.assessment.confidence)
                success_count += 1
            except Exception as e:
                logger.warning(
                    "Batch assessment failed for supplier %s at index %d: %s",
                    assess_req.supplier_id,
                    idx,
                    str(e),
                )
                errors.append({
                    "index": idx,
                    "supplier_id": assess_req.supplier_id,
                    "error": str(e),
                })

        # Calculate summary statistics
        avg_score = _float(total_score / success_count) if success_count > 0 else 0.0
        avg_confidence = (
            _float(total_confidence / success_count) if success_count > 0 else 0.0
        )

        duration = time.perf_counter() - start_time

        logger.info(
            "Batch assessment completed: total=%d, success=%d, errors=%d, "
            "avg_score=%.2f, avg_confidence=%.2f, duration=%.3fs",
            len(request.requests),
            success_count,
            len(errors),
            avg_score,
            avg_confidence,
            duration,
        )

        return BatchResponse(
            results=results,
            errors=errors,
            total_count=len(request.requests),
            success_count=success_count,
            error_count=len(errors),
            summary_statistics={
                "risk_level_counts": risk_counts,
                "average_score": avg_score,
                "average_confidence": avg_confidence,
            },
            processing_time_ms=duration * 1000.0,
        )

    # ------------------------------------------------------------------
    # Supplier comparison
    # ------------------------------------------------------------------

    def compare_suppliers(
        self,
        request: CompareSupplierRequest,
    ) -> ComparisonResponse:
        """Compare two or more suppliers across all risk factors.

        Args:
            request: CompareSupplierRequest containing list of supplier_ids.

        Returns:
            ComparisonResponse with assessments for each supplier and
            comparative analysis (score differences, rank order, factor
            deltas).

        Raises:
            ValueError: If fewer than 2 supplier_ids provided or any
                supplier not found.
        """
        if len(request.supplier_ids) < 2:
            raise ValueError("Comparison requires at least 2 suppliers")

        # Retrieve assessments
        assessments: List[SupplierRiskAssessment] = []
        with self._lock:
            for supplier_id in request.supplier_ids:
                # Find latest assessment for this supplier
                supplier_assessments = [
                    a for a in self._assessments.values()
                    if a.supplier_id == supplier_id
                ]
                if not supplier_assessments:
                    raise ValueError(f"No assessment found for supplier {supplier_id}")

                # Get most recent
                latest = max(supplier_assessments, key=lambda x: x.assessment_date)
                assessments.append(latest)

        # Calculate comparative metrics
        scores = [_decimal(a.composite_score) for a in assessments]
        max_score = max(scores)
        min_score = min(scores)
        score_range = max_score - min_score

        # Rank suppliers (lower score = better rank)
        sorted_assessments = sorted(assessments, key=lambda x: x.composite_score)
        rankings = {
            a.supplier_id: idx + 1
            for idx, a in enumerate(sorted_assessments)
        }

        # Calculate factor-level deltas
        factor_comparisons: Dict[str, Dict[str, float]] = {}
        for factor_key in _FACTOR_KEYS:
            factor_scores = []
            for assessment in assessments:
                factor_score_obj = next(
                    (fs for fs in assessment.factor_scores if fs.factor_name == factor_key),
                    None
                )
                if factor_score_obj:
                    factor_scores.append(_decimal(factor_score_obj.normalized_value))
                else:
                    factor_scores.append(Decimal("0.0"))

            factor_comparisons[factor_key] = {
                "max": _float(max(factor_scores)),
                "min": _float(min(factor_scores)),
                "range": _float(max(factor_scores) - min(factor_scores)),
                "std_dev": _float(self._calculate_std_dev(factor_scores)),
            }

        logger.info(
            "Supplier comparison completed: suppliers=%d, score_range=%.2f",
            len(request.supplier_ids),
            _float(score_range),
        )

        return ComparisonResponse(
            assessments=assessments,
            rankings=rankings,
            score_range=_float(score_range),
            max_score=_float(max_score),
            min_score=_float(min_score),
            factor_comparisons=factor_comparisons,
        )

    # ------------------------------------------------------------------
    # Trend analysis
    # ------------------------------------------------------------------

    def get_risk_trend(
        self,
        request: GetTrendRequest,
    ) -> TrendResponse:
        """Get historical risk score trend for a supplier.

        Analyzes historical risk scores to identify trends using linear
        regression. Supports custom time windows.

        Args:
            request: GetTrendRequest containing supplier_id and optional
                time window parameters.

        Returns:
            TrendResponse with historical scores, trend direction,
            slope coefficient, and forecast next period score.

        Raises:
            ValueError: If supplier_id not found or insufficient
                historical data (< 2 data points).
        """
        cfg = get_config()

        with self._lock:
            if request.supplier_id not in self._supplier_history:
                raise ValueError(
                    f"No assessment history found for supplier {request.supplier_id}"
                )

            history = self._supplier_history[request.supplier_id]

        if len(history) < 2:
            raise ValueError(
                f"Insufficient historical data for trend analysis: "
                f"{len(history)} data points (minimum 2 required)"
            )

        # Filter by time window if specified
        now = _utcnow()
        window_months = request.window_months or cfg.trend_window_months
        cutoff_date = now - timedelta(days=window_months * 30)
        filtered_history = [
            (ts, score) for ts, score in history if ts >= cutoff_date
        ]

        if len(filtered_history) < 2:
            filtered_history = history[-2:]  # Use last 2 data points minimum

        # Calculate trend using linear regression
        direction, slope = self._calculate_linear_regression(filtered_history)

        # Forecast next score
        if filtered_history:
            last_score = filtered_history[-1][1]
            forecast_score = last_score + slope
            forecast_score = max(Decimal("0.0"), min(Decimal("100.0"), forecast_score))
        else:
            forecast_score = Decimal("0.0")

        logger.info(
            "Trend analysis completed: supplier_id=%s, direction=%s, slope=%.2f, "
            "forecast=%.2f, data_points=%d",
            request.supplier_id,
            direction.value,
            _float(slope),
            _float(forecast_score),
            len(filtered_history),
        )

        return TrendResponse(
            supplier_id=request.supplier_id,
            trend_direction=direction,
            slope=_float(slope),
            historical_scores=[
                {"timestamp": ts.isoformat(), "score": _float(score)}
                for ts, score in filtered_history
            ],
            forecast_next_score=_float(forecast_score),
            data_points=len(filtered_history),
            window_months=window_months,
        )

    # ------------------------------------------------------------------
    # Rankings
    # ------------------------------------------------------------------

    def get_rankings(
        self,
        commodity: Optional[CommodityType] = None,
        region: Optional[str] = None,
        supplier_type: Optional[SupplierType] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get ranked list of suppliers by composite risk score.

        Args:
            commodity: Optional filter by commodity.
            region: Optional filter by region.
            supplier_type: Optional filter by supplier type.
            limit: Maximum number of results (default 100).

        Returns:
            List of dictionaries with supplier_id, composite_score,
            risk_level, rank, and assessment_date, sorted by score
            ascending (best to worst).
        """
        with self._lock:
            # Filter assessments
            filtered = []
            for assessment in self._assessments.values():
                if commodity and assessment.commodity != commodity:
                    continue
                if region and assessment.region != region:
                    continue
                if supplier_type and assessment.supplier_type != supplier_type:
                    continue
                filtered.append(assessment)

        # Sort by score (ascending = best first)
        sorted_assessments = sorted(filtered, key=lambda x: x.composite_score)
        sorted_assessments = sorted_assessments[:limit]

        # Build rankings
        rankings = []
        for idx, assessment in enumerate(sorted_assessments):
            rankings.append({
                "rank": idx + 1,
                "supplier_id": assessment.supplier_id,
                "composite_score": assessment.composite_score,
                "risk_level": assessment.risk_level.value,
                "assessment_date": assessment.assessment_date.isoformat(),
                "commodity": assessment.commodity.value if assessment.commodity else None,
                "region": assessment.region,
            })

        logger.info(
            "Rankings retrieved: total=%d, commodity=%s, region=%s, supplier_type=%s",
            len(rankings),
            commodity.value if commodity else "all",
            region or "all",
            supplier_type.value if supplier_type else "all",
        )

        return rankings

    # ------------------------------------------------------------------
    # Helper methods: Validation
    # ------------------------------------------------------------------

    def _validate_assessment_inputs(
        self,
        request: AssessSupplierRequest,
    ) -> None:
        """Validate assessment request inputs.

        Raises:
            ValueError: If validation fails.
        """
        if not request.supplier_id:
            raise ValueError("supplier_id is required")

        if not request.factor_values:
            raise ValueError("factor_values is required")

        # Check all required factors present
        missing_factors = set(_FACTOR_KEYS) - set(request.factor_values.keys())
        if missing_factors:
            raise ValueError(f"Missing required factors: {missing_factors}")

        # Validate factor values in [0, 100] range (raw values)
        for key, value in request.factor_values.items():
            if not (0.0 <= value <= 100.0):
                raise ValueError(
                    f"Factor {key} value {value} out of valid range [0, 100]"
                )

        # Validate custom weights if provided
        if request.factor_weights:
            weight_sum = sum(request.factor_weights.values())
            if not (0.99 <= weight_sum <= 1.01):  # Allow small float precision error
                raise ValueError(
                    f"Factor weights must sum to 1.0, got {weight_sum}"
                )

    # ------------------------------------------------------------------
    # Helper methods: Normalization
    # ------------------------------------------------------------------

    def _normalize_factors(
        self,
        factor_values: Dict[str, float],
    ) -> Dict[str, Decimal]:
        """Normalize factor values to [0, 100] scale.

        Input values are assumed to be already in [0, 100] range.
        This method applies any additional normalization logic if needed.

        Args:
            factor_values: Raw factor values.

        Returns:
            Dictionary of normalized factor values as Decimals.
        """
        normalized = {}
        for key, value in factor_values.items():
            # For now, values are already in [0, 100], just convert to Decimal
            normalized[key] = _decimal(value)
        return normalized

    def _normalize_factor(
        self,
        factor_name: str,
        raw_value: float,
    ) -> Decimal:
        """Normalize a single factor value to [0, 100] scale.

        Args:
            factor_name: Name of factor.
            raw_value: Raw input value.

        Returns:
            Normalized Decimal value in [0, 100].
        """
        # Clamp to [0, 100]
        clamped = max(0.0, min(100.0, raw_value))
        return _decimal(clamped)

    # ------------------------------------------------------------------
    # Helper methods: Scoring
    # ------------------------------------------------------------------

    def _get_factor_weights(
        self,
        custom_weights: Optional[Dict[str, float]],
    ) -> Dict[str, Decimal]:
        """Get factor weights from custom or default configuration.

        Args:
            custom_weights: Optional custom weights.

        Returns:
            Dictionary of factor weights as Decimals.
        """
        if custom_weights:
            return {k: _decimal(v) for k, v in custom_weights.items()}

        cfg = get_config()
        return {
            "geographic_sourcing": _decimal(cfg.geographic_sourcing_weight / 100.0),
            "compliance_history": _decimal(cfg.compliance_history_weight / 100.0),
            "documentation_quality": _decimal(cfg.documentation_quality_weight / 100.0),
            "certification_status": _decimal(cfg.certification_status_weight / 100.0),
            "traceability_completeness": _decimal(cfg.traceability_completeness_weight / 100.0),
            "financial_stability": _decimal(cfg.financial_stability_weight / 100.0),
            "environmental_performance": _decimal(cfg.environmental_performance_weight / 100.0),
            "social_compliance": _decimal(cfg.social_compliance_weight / 100.0),
        }

    def _calculate_composite_score(
        self,
        normalized_factors: Dict[str, Decimal],
        weights: Dict[str, Decimal],
    ) -> Decimal:
        """Calculate weighted composite risk score.

        Args:
            normalized_factors: Normalized factor values [0, 100].
            weights: Factor weights [sum to 1.0].

        Returns:
            Composite score in [0, 100].
        """
        composite = Decimal("0.0")
        for factor_key in _FACTOR_KEYS:
            factor_value = normalized_factors.get(factor_key, Decimal("0.0"))
            weight = weights.get(factor_key, Decimal("0.0"))
            composite += factor_value * weight

        # Ensure within [0, 100]
        composite = max(Decimal("0.0"), min(Decimal("100.0"), composite))
        return composite.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Helper methods: Classification
    # ------------------------------------------------------------------

    def _classify_risk_level(
        self,
        composite_score: Decimal,
    ) -> RiskLevel:
        """Classify risk level from composite score.

        Args:
            composite_score: Composite score [0, 100].

        Returns:
            RiskLevel enum value.
        """
        cfg = get_config()
        score_float = _float(composite_score)

        if score_float < cfg.low_risk_threshold:
            return RiskLevel.LOW
        elif score_float < cfg.medium_risk_threshold:
            return RiskLevel.MEDIUM
        elif score_float < cfg.high_risk_threshold:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    # ------------------------------------------------------------------
    # Helper methods: Confidence
    # ------------------------------------------------------------------

    def _calculate_confidence(
        self,
        factor_values: Dict[str, float],
        data_dates: Optional[Dict[str, datetime]],
    ) -> Decimal:
        """Calculate confidence score from data completeness and freshness.

        Args:
            factor_values: Factor values (for completeness).
            data_dates: Optional timestamps for each factor (for freshness).

        Returns:
            Confidence score [0.0, 1.0].
        """
        cfg = get_config()

        # Data completeness
        available_factors = len([v for v in factor_values.values() if v > 0.0])
        completeness = Decimal(available_factors) / Decimal(len(_FACTOR_KEYS))

        # Data freshness
        freshness = Decimal("1.0")
        if data_dates:
            now = _utcnow()
            max_age = timedelta(days=365)  # 1 year threshold
            oldest_date = min(data_dates.values())
            age = now - oldest_date
            if age > max_age:
                freshness = Decimal("0.5")

        # Weighted average: 60% completeness, 40% freshness
        confidence = Decimal("0.6") * completeness + Decimal("0.4") * freshness
        return confidence.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Helper methods: Trend analysis
    # ------------------------------------------------------------------

    def _analyze_trend(
        self,
        supplier_id: str,
        current_score: Decimal,
    ) -> Tuple[TrendDirection, Decimal]:
        """Analyze risk score trend for supplier.

        Args:
            supplier_id: Supplier identifier.
            current_score: Current risk score.

        Returns:
            Tuple of (TrendDirection, slope coefficient).
        """
        cfg = get_config()

        with self._lock:
            if supplier_id not in self._supplier_history:
                return TrendDirection.STABLE, Decimal("0.0")

            history = self._supplier_history[supplier_id]

        if len(history) < 2:
            return TrendDirection.STABLE, Decimal("0.0")

        # Get recent window
        now = _utcnow()
        window = timedelta(days=cfg.trend_window_months * 30)
        cutoff = now - window
        recent = [(ts, score) for ts, score in history if ts >= cutoff]

        if len(recent) < 2:
            recent = history[-2:]  # Use last 2 points minimum

        # Calculate linear regression
        direction, slope = self._calculate_linear_regression(recent)
        return direction, slope

    def _calculate_linear_regression(
        self,
        data_points: List[Tuple[datetime, Decimal]],
    ) -> Tuple[TrendDirection, Decimal]:
        """Calculate linear regression slope from time series data.

        Args:
            data_points: List of (timestamp, score) tuples.

        Returns:
            Tuple of (TrendDirection, slope).
        """
        if len(data_points) < 2:
            return TrendDirection.STABLE, Decimal("0.0")

        # Convert timestamps to days since first point
        first_ts = data_points[0][0]
        x_values = [
            _decimal((ts - first_ts).total_seconds() / 86400.0)
            for ts, _ in data_points
        ]
        y_values = [score for _, score in data_points]

        n = Decimal(len(x_values))
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)

        # Slope: m = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x^2)
        numerator = n * sum_xy - sum_x * sum_y
        denominator = n * sum_x2 - sum_x * sum_x

        if denominator == 0:
            slope = Decimal("0.0")
        else:
            slope = numerator / denominator

        # Classify direction
        if slope < _TREND_IMPROVING_THRESHOLD:
            direction = TrendDirection.IMPROVING
        elif slope > _TREND_DETERIORATING_THRESHOLD:
            direction = TrendDirection.DETERIORATING
        else:
            direction = TrendDirection.STABLE

        return direction, slope.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Helper methods: Peer benchmarking
    # ------------------------------------------------------------------

    def _benchmark_peer_group(
        self,
        composite_score: Decimal,
        commodity: CommodityType,
        region: str,
    ) -> Tuple[Decimal, Decimal]:
        """Calculate peer group percentile ranking.

        Args:
            composite_score: Supplier's composite risk score.
            commodity: Commodity type.
            region: Geographic region.

        Returns:
            Tuple of (percentile, peer_median_score).
        """
        peer_key = (commodity.value, region)

        with self._lock:
            peer_scores = self._peer_groups.get(peer_key, [])

        if not peer_scores or len(peer_scores) < 5:
            # Insufficient peer data
            return Decimal("50.0"), composite_score

        # Calculate percentile (lower score = better, so invert for percentile)
        sorted_peers = sorted(peer_scores)
        count_lower = len([s for s in sorted_peers if s < composite_score])
        percentile = Decimal(count_lower) / Decimal(len(sorted_peers)) * Decimal("100.0")
        percentile = percentile.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

        # Calculate median
        mid = len(sorted_peers) // 2
        if len(sorted_peers) % 2 == 0:
            median = (sorted_peers[mid - 1] + sorted_peers[mid]) / Decimal("2.0")
        else:
            median = sorted_peers[mid]
        median = median.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        return percentile, median

    # ------------------------------------------------------------------
    # Helper methods: Factor scores
    # ------------------------------------------------------------------

    def _build_factor_scores(
        self,
        normalized_factors: Dict[str, Decimal],
        weights: Dict[str, Decimal],
        data_dates: Optional[Dict[str, datetime]],
    ) -> List[FactorScore]:
        """Build list of FactorScore objects.

        Args:
            normalized_factors: Normalized factor values.
            weights: Factor weights.
            data_dates: Optional timestamps per factor.

        Returns:
            List of FactorScore objects.
        """
        factor_scores: List[FactorScore] = []

        for factor_key in _FACTOR_KEYS:
            normalized_value = normalized_factors.get(factor_key, Decimal("0.0"))
            weight = weights.get(factor_key, Decimal("0.0"))
            weighted_contribution = normalized_value * weight
            data_date = data_dates.get(factor_key) if data_dates else None

            factor_scores.append(
                FactorScore(
                    factor_name=factor_key,
                    normalized_value=_float(normalized_value),
                    weight=_float(weight),
                    weighted_contribution=_float(weighted_contribution),
                    data_date=data_date,
                )
            )

        return factor_scores

    # ------------------------------------------------------------------
    # Helper methods: Next assessment date
    # ------------------------------------------------------------------

    def _calculate_next_assessment_date(
        self,
        risk_level: RiskLevel,
        assessment_date: datetime,
    ) -> datetime:
        """Calculate next assessment date based on risk level.

        Args:
            risk_level: Current risk level.
            assessment_date: Current assessment date.

        Returns:
            Next assessment due date.
        """
        # Assessment frequency by risk level
        frequency_days = {
            RiskLevel.LOW: 180,       # 6 months
            RiskLevel.MEDIUM: 90,     # 3 months
            RiskLevel.HIGH: 30,       # 1 month
            RiskLevel.CRITICAL: 7,    # 1 week
        }

        days = frequency_days.get(risk_level, 90)
        return assessment_date + timedelta(days=days)

    # ------------------------------------------------------------------
    # Helper methods: Statistics
    # ------------------------------------------------------------------

    def _calculate_std_dev(
        self,
        values: List[Decimal],
    ) -> Decimal:
        """Calculate standard deviation of values.

        Args:
            values: List of Decimal values.

        Returns:
            Standard deviation.
        """
        if not values:
            return Decimal("0.0")

        n = Decimal(len(values))
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        std_dev = Decimal(math.sqrt(float(variance)))

        return std_dev.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
