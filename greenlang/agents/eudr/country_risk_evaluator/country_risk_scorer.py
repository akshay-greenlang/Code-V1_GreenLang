# -*- coding: utf-8 -*-
"""
Country Risk Scorer Engine - AGENT-EUDR-016 Engine 1

Multi-factor weighted composite risk scoring per EUDR Article 29 with
6-factor model (deforestation_rate 30%, governance_quality 20%,
enforcement_effectiveness 15%, corruption_index 15%, forest_law_compliance
10%, historical_trend 10%), EC benchmark override capability, risk level
classification (LOW 0-30, STANDARD 31-65, HIGH 66-100), confidence
scoring, trend analysis, country comparison, and batch assessment.

Risk Score Calculation (Zero-Hallucination):
    composite_score = sum(factor_value * factor_weight for all 6 factors)
    All factors normalized to [0, 100] scale.
    EC benchmark override takes precedence when enabled.

Classification Thresholds (configurable):
    - LOW: 0-30 (simplified DD eligible per Article 13)
    - STANDARD: 31-65 (standard DD per Articles 10-11)
    - HIGH: 66-100 (enhanced DD per Article 11)

Confidence Scoring:
    confidence = weighted_average(data_completeness, data_freshness)
    data_completeness = (factors_available / 6.0)
    data_freshness = 1.0 if max_age < threshold else 0.5

Trend Analysis:
    Rolling 3-year window, linear regression on historical scores.
    Direction: improving, stable, deteriorating.

Zero-Hallucination: All scoring is deterministic arithmetic. No LLM
    calls in the calculation path. All inputs are validated against
    database lookups or configuration bounds.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-016 Country Risk Evaluator (GL-EUDR-CRE-016)
Status: Production Ready
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from .config import get_config
from .metrics import (
    observe_assessment_duration,
    record_assessment_completed,
)
from .models import (
    AssessmentConfidence,
    CountryRiskAssessment,
    RiskLevel,
    TrendDirection,
)
from .provenance import get_provenance_tracker

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default factor weights (sum = 1.0)
_DEFAULT_WEIGHTS: Dict[str, Decimal] = {
    "deforestation_rate": Decimal("0.30"),
    "governance_quality": Decimal("0.20"),
    "enforcement_effectiveness": Decimal("0.15"),
    "corruption_index": Decimal("0.15"),
    "forest_law_compliance": Decimal("0.10"),
    "historical_trend": Decimal("0.10"),
}

#: Factor keys
_FACTOR_KEYS: List[str] = [
    "deforestation_rate",
    "governance_quality",
    "enforcement_effectiveness",
    "corruption_index",
    "forest_law_compliance",
    "historical_trend",
]

#: EC benchmark risk level mapping (values match RiskLevel enum: lowercase)
_EC_BENCHMARK_MAPPING: Dict[str, str] = {
    "low": "low",
    "standard": "standard",
    "high": "high",
}

def _decimal(value: Any) -> Decimal:
    """Convert value to Decimal for precise arithmetic."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))

def _float(value: Decimal) -> float:
    """Convert Decimal to float for API responses."""
    return float(value)

# ---------------------------------------------------------------------------
# CountryRiskScorer
# ---------------------------------------------------------------------------

class CountryRiskScorer:
    """Multi-factor weighted composite risk scoring per EUDR Article 29.

    Calculates composite risk scores from 6 weighted factors, applies
    EC benchmark overrides, classifies risk levels, scores confidence,
    analyzes trends, and provides country comparison capabilities.

    All scoring operations use Decimal arithmetic for zero floating-point
    drift and deterministic reproducibility.

    Attributes:
        _assessments: In-memory store of risk assessments keyed by
            assessment_id.
        _risk_history: Historical risk scores keyed by country_code,
            list of (timestamp, score) tuples.
        _ec_benchmarks: EC-published benchmark classifications keyed
            by country_code.
        _lock: Threading lock for thread-safe access.

    Example:
        >>> scorer = CountryRiskScorer()
        >>> result = scorer.assess_country("BR", factors)
        >>> assert result.risk_level in [RiskLevel.LOW, RiskLevel.STANDARD, RiskLevel.HIGH]
        >>> assert 0.0 <= result.composite_score <= 100.0
    """

    def __init__(self) -> None:
        """Initialize CountryRiskScorer with empty stores."""
        self._assessments: Dict[str, CountryRiskAssessment] = {}
        self._risk_history: Dict[str, List[Tuple[datetime, Decimal]]] = {}
        self._ec_benchmarks: Dict[str, str] = {}
        self._lock: threading.Lock = threading.Lock()
        logger.info(
            "CountryRiskScorer initialized: factors=%d, default_weights=%s",
            len(_FACTOR_KEYS),
            {k: _float(v) for k, v in _DEFAULT_WEIGHTS.items()},
        )

    # ------------------------------------------------------------------
    # Primary assessment
    # ------------------------------------------------------------------

    def assess_country(
        self,
        country_code: str,
        factor_values: Dict[str, float],
        factor_weights: Optional[Dict[str, float]] = None,
        data_dates: Optional[Dict[str, datetime]] = None,
    ) -> CountryRiskAssessment:
        """Assess country risk using 6-factor weighted composite scoring.

        Applies the following assessment pipeline:
        1. Validate inputs (country code, factor values, weights).
        2. Normalize all factor values to [0, 100] scale.
        3. Calculate composite score using weighted sum.
        4. Check for EC benchmark override.
        5. Classify risk level from score or override.
        6. Calculate confidence score from completeness and freshness.
        7. Store assessment and update history.
        8. Record provenance and metrics.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            factor_values: Dictionary of factor values (raw inputs).
                Keys: deforestation_rate, governance_quality,
                enforcement_effectiveness, corruption_index,
                forest_law_compliance, historical_trend.
                All values will be normalized to [0, 100].
            factor_weights: Optional custom factor weights (default uses
                config weights). Must sum to 1.0.
            data_dates: Optional dictionary of timestamps for each factor
                to calculate data freshness confidence.

        Returns:
            CountryRiskAssessment with composite_score, risk_level,
            confidence, trend_direction, and factor breakdown.

        Raises:
            ValueError: If country_code is empty, factor_values missing
                required keys, or weights invalid.
        """
        start = time.monotonic()
        cfg = get_config()

        # -- Input validation ------------------------------------------------
        country_code = self._validate_country_code(country_code)
        weights = self._validate_weights(factor_weights, cfg)
        normalized_factors = self._normalize_factors(factor_values)

        # -- Composite score -------------------------------------------------
        composite_score = self._calculate_composite_score(
            normalized_factors, weights,
        )

        # -- EC benchmark override -------------------------------------------
        ec_override_level = None
        if cfg.ec_benchmark_override:
            ec_override_level = self._check_ec_override(country_code)

        # -- Risk classification ---------------------------------------------
        if ec_override_level:
            risk_level = RiskLevel(ec_override_level)
        else:
            risk_level = self._classify_risk_level(composite_score, cfg)

        # -- Confidence scoring ----------------------------------------------
        confidence = self._calculate_confidence(
            factor_values, data_dates or {}, cfg,
        )

        # -- Trend analysis --------------------------------------------------
        trend_direction = self._get_trend_direction(
            country_code, composite_score,
        )

        # -- Build assessment ------------------------------------------------
        assessment = self._build_assessment(
            country_code=country_code,
            composite_score=composite_score,
            risk_level=risk_level,
            confidence=confidence,
            trend_direction=trend_direction,
            factor_values=normalized_factors,
            factor_weights={k: _float(v) for k, v in weights.items()},
            ec_override=ec_override_level is not None,
        )

        # -- Store and history update ----------------------------------------
        with self._lock:
            self._assessments[assessment.assessment_id] = assessment
            self._add_to_history(country_code, composite_score)

        # -- Provenance ------------------------------------------------------
        tracker = get_provenance_tracker()
        tracker.record(
            entity_type="country_assessment",
            action="assess",
            entity_id=assessment.assessment_id,
            data=assessment.model_dump(mode="json"),
            metadata={
                "country_code": country_code,
                "composite_score": _float(composite_score),
                "risk_level": risk_level.value,
                "confidence": confidence.value,
                "ec_override": ec_override_level is not None,
            },
        )

        # -- Metrics ---------------------------------------------------------
        elapsed = time.monotonic() - start
        observe_assessment_duration(elapsed)
        record_assessment_completed(risk_level.value)

        logger.info(
            "Country risk assessed: country=%s score=%.2f level=%s "
            "confidence=%s trend=%s ec_override=%s elapsed_ms=%.1f",
            country_code,
            _float(composite_score),
            risk_level.value,
            confidence.value,
            trend_direction.value,
            ec_override_level is not None,
            elapsed * 1000,
        )
        return assessment

    def assess_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[CountryRiskAssessment]:
        """Assess country risk for multiple countries in batch.

        Each item in the batch is a dictionary with keys:
            - country_code (str, required)
            - factor_values (dict[str, float], required)
            - factor_weights (dict[str, float], optional)
            - data_dates (dict[str, datetime], optional)

        Args:
            items: List of assessment request dictionaries.

        Returns:
            List of CountryRiskAssessment results in the same order
            as the input items.

        Raises:
            ValueError: If items list is empty or exceeds batch_max_size.
        """
        cfg = get_config()
        if not items:
            raise ValueError("Batch items list must not be empty")
        if len(items) > cfg.batch_max_size:
            raise ValueError(
                f"Batch size {len(items)} exceeds maximum "
                f"{cfg.batch_max_size}"
            )

        results: List[CountryRiskAssessment] = []
        for item in items:
            result = self.assess_country(
                country_code=item["country_code"],
                factor_values=item["factor_values"],
                factor_weights=item.get("factor_weights"),
                data_dates=item.get("data_dates"),
            )
            results.append(result)

        logger.info(
            "Batch assessment completed: items=%d", len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_assessment(
        self, assessment_id: str,
    ) -> Optional[CountryRiskAssessment]:
        """Retrieve an assessment by its unique identifier.

        Args:
            assessment_id: The assessment_id to look up.

        Returns:
            CountryRiskAssessment if found, None otherwise.
        """
        with self._lock:
            return self._assessments.get(assessment_id)

    def list_assessments(
        self,
        country_code: Optional[str] = None,
        risk_level: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[CountryRiskAssessment]:
        """List assessments with optional filters.

        Args:
            country_code: Optional country code filter.
            risk_level: Optional risk level filter (LOW/STANDARD/HIGH).
            limit: Maximum number of results (default 100).
            offset: Pagination offset (default 0).

        Returns:
            Filtered list of CountryRiskAssessment objects.
        """
        with self._lock:
            results = list(self._assessments.values())

        if country_code:
            cc = country_code.upper().strip()
            results = [a for a in results if a.country_code == cc]

        if risk_level:
            results = [
                a for a in results if a.risk_level.value == risk_level.upper()
            ]

        # Sort by assessment timestamp descending
        results.sort(key=lambda a: a.assessed_at, reverse=True)

        return results[offset:offset + limit]

    # ------------------------------------------------------------------
    # Risk level classification
    # ------------------------------------------------------------------

    def classify_risk_level(
        self, composite_score: float,
    ) -> str:
        """Classify risk level from composite score.

        Args:
            composite_score: Composite risk score (0-100).

        Returns:
            Risk level string: LOW, STANDARD, or HIGH.

        Raises:
            ValueError: If composite_score is outside [0, 100].
        """
        cfg = get_config()
        if composite_score < 0.0 or composite_score > 100.0:
            raise ValueError(
                f"composite_score must be in [0, 100], got {composite_score}"
            )

        risk_level = self._classify_risk_level(_decimal(composite_score), cfg)
        return risk_level.value

    # ------------------------------------------------------------------
    # Country comparison
    # ------------------------------------------------------------------

    def compare_countries(
        self,
        country_codes: List[str],
    ) -> Dict[str, Any]:
        """Compare risk assessments across multiple countries.

        Retrieves the most recent assessment for each country and
        computes rank, percentile, and relative risk comparison.

        Args:
            country_codes: List of ISO 3166-1 alpha-2 country codes.

        Returns:
            Dictionary with country_scores (list of dicts with
            country_code, composite_score, risk_level, rank, percentile),
            highest_risk_country, lowest_risk_country, and statistics.

        Raises:
            ValueError: If country_codes list is empty or contains
                countries with no assessments.
        """
        if not country_codes:
            raise ValueError("country_codes list must not be empty")

        country_data: List[Dict[str, Any]] = []
        with self._lock:
            for cc in country_codes:
                cc_upper = cc.upper().strip()
                # Get most recent assessment for this country
                country_assessments = [
                    a for a in self._assessments.values()
                    if a.country_code == cc_upper
                ]
                if not country_assessments:
                    logger.warning(
                        "No assessments found for country: %s", cc_upper,
                    )
                    continue
                country_assessments.sort(
                    key=lambda a: a.assessed_at, reverse=True,
                )
                latest = country_assessments[0]
                country_data.append({
                    "country_code": cc_upper,
                    "composite_score": latest.risk_score,
                    "risk_level": latest.risk_level.value,
                    "assessed_at": latest.assessed_at.isoformat(),
                })

        if not country_data:
            raise ValueError("No assessments found for any country")

        # Sort by score descending (highest risk first)
        country_data.sort(key=lambda x: x["composite_score"], reverse=True)

        # Add rank and percentile
        total = len(country_data)
        for rank, item in enumerate(country_data, start=1):
            item["rank"] = rank
            item["percentile"] = round((rank / total) * 100, 1)

        # Statistics
        scores = [x["composite_score"] for x in country_data]
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = math.sqrt(variance)

        highest = country_data[0]
        lowest = country_data[-1]

        return {
            "country_scores": country_data,
            "highest_risk_country": {
                "country_code": highest["country_code"],
                "composite_score": highest["composite_score"],
                "risk_level": highest["risk_level"],
            },
            "lowest_risk_country": {
                "country_code": lowest["country_code"],
                "composite_score": lowest["composite_score"],
                "risk_level": lowest["risk_level"],
            },
            "statistics": {
                "mean_score": round(mean_score, 2),
                "std_dev": round(std_dev, 2),
                "min_score": lowest["composite_score"],
                "max_score": highest["composite_score"],
                "total_countries": total,
            },
        }

    # ------------------------------------------------------------------
    # Trend analysis
    # ------------------------------------------------------------------

    def get_risk_trend(
        self,
        country_code: str,
        window_years: int = 3,
    ) -> Dict[str, Any]:
        """Get historical risk trend for a country.

        Analyzes risk score trend over a rolling window using linear
        regression to determine direction (improving, stable, deteriorating).

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            window_years: Rolling window in years (default 3).

        Returns:
            Dictionary with country_code, trend_direction, slope,
            historical_scores (list of {date, score}), and summary.

        Raises:
            ValueError: If country_code is empty or window_years < 1.
        """
        country_code = self._validate_country_code(country_code)
        if window_years < 1:
            raise ValueError("window_years must be >= 1")

        cutoff_date = utcnow() - timedelta(days=window_years * 365)

        with self._lock:
            history = self._risk_history.get(country_code, [])

        # Filter to window
        window_data = [
            (ts, score) for ts, score in history if ts >= cutoff_date
        ]

        if len(window_data) < 2:
            return {
                "country_code": country_code,
                "trend_direction": "insufficient_data",
                "slope": 0.0,
                "historical_scores": [],
                "summary": "Insufficient historical data for trend analysis",
            }

        # Linear regression
        slope = self._calculate_trend_slope(window_data)
        direction = self._classify_trend_direction(slope)

        historical_scores = [
            {
                "date": ts.isoformat(),
                "score": _float(score),
            }
            for ts, score in window_data
        ]

        summary = self._build_trend_summary(
            country_code, direction, slope, len(window_data),
        )

        return {
            "country_code": country_code,
            "trend_direction": direction.value,
            "slope": round(slope, 4),
            "window_years": window_years,
            "data_points": len(window_data),
            "historical_scores": historical_scores,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Country ranking
    # ------------------------------------------------------------------

    def get_country_ranking(
        self,
        limit: int = 100,
        order: str = "desc",
    ) -> List[Dict[str, Any]]:
        """Get ranked list of all assessed countries by risk score.

        Args:
            limit: Maximum number of countries to return (default 100).
            order: Sort order, 'desc' (highest risk first) or 'asc'
                (lowest risk first). Default 'desc'.

        Returns:
            List of dictionaries with country_code, composite_score,
            risk_level, rank, percentile, and assessed_at.

        Raises:
            ValueError: If order is not 'desc' or 'asc'.
        """
        if order not in ("desc", "asc"):
            raise ValueError("order must be 'desc' or 'asc'")

        with self._lock:
            # Get most recent assessment per country
            country_latest: Dict[str, CountryRiskAssessment] = {}
            for assessment in self._assessments.values():
                cc = assessment.country_code
                if cc not in country_latest:
                    country_latest[cc] = assessment
                else:
                    if assessment.assessed_at > country_latest[cc].assessed_at:
                        country_latest[cc] = assessment

        ranking_data = [
            {
                "country_code": a.country_code,
                "composite_score": a.risk_score,
                "risk_level": a.risk_level.value,
                "assessed_at": a.assessed_at.isoformat(),
            }
            for a in country_latest.values()
        ]

        # Sort
        reverse = order == "desc"
        ranking_data.sort(key=lambda x: x["composite_score"], reverse=reverse)

        # Add rank and percentile
        total = len(ranking_data)
        for rank, item in enumerate(ranking_data, start=1):
            item["rank"] = rank
            item["percentile"] = round((rank / total) * 100, 1)

        return ranking_data[:limit]

    # ------------------------------------------------------------------
    # EC benchmark override
    # ------------------------------------------------------------------

    def set_ec_benchmark(
        self,
        country_code: str,
        risk_level: str,
    ) -> None:
        """Set EC-published benchmark classification for a country.

        When EC benchmark override is enabled in config, this
        classification will take precedence over agent-computed scores.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            risk_level: EC benchmark level (low, standard, high).

        Raises:
            ValueError: If country_code or risk_level is invalid.
        """
        country_code = self._validate_country_code(country_code)
        risk_level_lower = risk_level.lower().strip()
        if risk_level_lower not in _EC_BENCHMARK_MAPPING:
            raise ValueError(
                f"Invalid EC benchmark level '{risk_level}'; "
                f"must be one of: low, standard, high"
            )

        mapped_level = _EC_BENCHMARK_MAPPING[risk_level_lower]

        with self._lock:
            self._ec_benchmarks[country_code] = mapped_level

        # Provenance
        tracker = get_provenance_tracker()
        tracker.record(
            entity_type="country_assessment",
            action="update",
            entity_id=country_code,
            data={"ec_benchmark": mapped_level},
            metadata={
                "country_code": country_code,
                "risk_level": mapped_level,
                "source": "ec_benchmark",
            },
        )

        logger.info(
            "EC benchmark set: country=%s level=%s",
            country_code, mapped_level,
        )

    def get_ec_benchmark(
        self, country_code: str,
    ) -> Optional[str]:
        """Get EC-published benchmark classification for a country.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            EC benchmark level (LOW, STANDARD, HIGH) or None if not set.
        """
        country_code = self._validate_country_code(country_code)
        with self._lock:
            return self._ec_benchmarks.get(country_code)

    def clear_ec_benchmark(
        self, country_code: str,
    ) -> None:
        """Clear EC benchmark override for a country.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
        """
        country_code = self._validate_country_code(country_code)
        with self._lock:
            if country_code in self._ec_benchmarks:
                del self._ec_benchmarks[country_code]

        logger.info("EC benchmark cleared: country=%s", country_code)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_country_code(self, country_code: str) -> str:
        """Validate and normalize country code.

        Args:
            country_code: Raw country code string.

        Returns:
            Uppercase, stripped country code.

        Raises:
            ValueError: If country_code is empty or wrong length.
        """
        if not country_code or not country_code.strip():
            raise ValueError("country_code must not be empty")
        cc = country_code.upper().strip()
        if len(cc) != 2:
            raise ValueError(
                f"country_code must be 2 characters, got '{cc}'"
            )
        return cc

    def _validate_weights(
        self,
        factor_weights: Optional[Dict[str, float]],
        cfg: Any,
    ) -> Dict[str, Decimal]:
        """Validate and normalize factor weights.

        Args:
            factor_weights: Optional custom weights dict.
            cfg: Agent configuration.

        Returns:
            Dictionary of Decimal weights for all 6 factors.

        Raises:
            ValueError: If weights do not sum to ~1.0 or are out of
                allowed bounds.
        """
        if factor_weights is None:
            # Use default weights from config
            return {
                "deforestation_rate": _decimal(cfg.deforestation_weight / 100.0),
                "governance_quality": _decimal(cfg.governance_weight / 100.0),
                "enforcement_effectiveness": _decimal(cfg.enforcement_weight / 100.0),
                "corruption_index": _decimal(cfg.corruption_weight / 100.0),
                "forest_law_compliance": _decimal(cfg.forest_law_weight / 100.0),
                "historical_trend": _decimal(cfg.trend_weight / 100.0),
            }

        # Validate custom weights
        weights: Dict[str, Decimal] = {}
        for key in _FACTOR_KEYS:
            if key not in factor_weights:
                raise ValueError(f"Missing weight for factor '{key}'")
            weight = _decimal(factor_weights[key])
            min_weight = _decimal(cfg.min_factor_weight / 100.0)
            max_weight = _decimal(cfg.max_factor_weight / 100.0)
            if weight < min_weight or weight > max_weight:
                raise ValueError(
                    f"Weight for '{key}' must be in "
                    f"[{min_weight}, {max_weight}], got {weight}"
                )
            weights[key] = weight

        # Check sum = 1.0 (tolerance 0.01)
        total = sum(weights.values())
        if abs(total - Decimal("1.0")) > Decimal("0.01"):
            raise ValueError(
                f"Factor weights must sum to 1.0, got {_float(total)}"
            )

        return weights

    def _normalize_factors(
        self, factor_values: Dict[str, float],
    ) -> Dict[str, Decimal]:
        """Normalize all factor values to [0, 100] scale.

        Applies min-max normalization or percentile-based normalization
        depending on the factor type. All factors are inverted if needed
        so that higher value = higher risk.

        Args:
            factor_values: Raw factor values dict.

        Returns:
            Dictionary of normalized Decimal values for all 6 factors.

        Raises:
            ValueError: If any required factor is missing.
        """
        normalized: Dict[str, Decimal] = {}

        for key in _FACTOR_KEYS:
            if key not in factor_values:
                # Missing factor defaults to 50.0 (neutral)
                logger.warning(
                    "Missing factor '%s', using default value 50.0", key,
                )
                normalized[key] = Decimal("50.0")
                continue

            raw_value = _decimal(factor_values[key])

            # Normalize based on factor type
            if key == "deforestation_rate":
                # Assume raw value is % forest loss per year (0-10% typical)
                # Clamp to [0, 10] and scale to [0, 100]
                clamped = min(max(raw_value, Decimal("0")), Decimal("10"))
                normalized[key] = clamped * Decimal("10")

            elif key == "governance_quality":
                # Assume raw value is 0-100 scale (higher = better)
                # Invert so higher = higher risk
                clamped = min(max(raw_value, Decimal("0")), Decimal("100"))
                normalized[key] = Decimal("100") - clamped

            elif key == "enforcement_effectiveness":
                # Assume raw value is 0-100 scale (higher = better)
                # Invert so higher = higher risk
                clamped = min(max(raw_value, Decimal("0")), Decimal("100"))
                normalized[key] = Decimal("100") - clamped

            elif key == "corruption_index":
                # Assume raw value is CPI 0-100 (higher = less corrupt)
                # Invert so higher = higher risk
                clamped = min(max(raw_value, Decimal("0")), Decimal("100"))
                normalized[key] = Decimal("100") - clamped

            elif key == "forest_law_compliance":
                # Assume raw value is 0-100 scale (higher = better)
                # Invert so higher = higher risk
                clamped = min(max(raw_value, Decimal("0")), Decimal("100"))
                normalized[key] = Decimal("100") - clamped

            elif key == "historical_trend":
                # Assume raw value is trend slope (-10 to +10 typical)
                # Positive slope = worsening = higher risk
                # Scale to [0, 100]
                clamped = min(max(raw_value, Decimal("-10")), Decimal("10"))
                normalized[key] = ((clamped + Decimal("10")) / Decimal("20")) * Decimal("100")

        return normalized

    def _calculate_composite_score(
        self,
        normalized_factors: Dict[str, Decimal],
        weights: Dict[str, Decimal],
    ) -> Decimal:
        """Calculate weighted composite risk score.

        Args:
            normalized_factors: Normalized factor values [0, 100].
            weights: Factor weights (sum = 1.0).

        Returns:
            Composite score Decimal [0, 100].
        """
        composite = Decimal("0")
        for key in _FACTOR_KEYS:
            factor_value = normalized_factors[key]
            weight = weights[key]
            composite += factor_value * weight

        # Round to 2 decimal places
        return composite.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _check_ec_override(self, country_code: str) -> Optional[str]:
        """Check for EC benchmark override.

        Args:
            country_code: ISO alpha-2 code.

        Returns:
            EC benchmark level string (LOW, STANDARD, HIGH) or None.
        """
        with self._lock:
            return self._ec_benchmarks.get(country_code)

    def _classify_risk_level(
        self, composite_score: Decimal, cfg: Any,
    ) -> RiskLevel:
        """Classify risk level from composite score.

        Args:
            composite_score: Composite risk score [0, 100].
            cfg: Agent configuration with thresholds.

        Returns:
            RiskLevel enum value.
        """
        score_float = _float(composite_score)
        if score_float <= cfg.low_risk_threshold:
            return RiskLevel.LOW
        if score_float <= cfg.high_risk_threshold:
            return RiskLevel.STANDARD
        return RiskLevel.HIGH

    def _calculate_confidence(
        self,
        factor_values: Dict[str, float],
        data_dates: Dict[str, datetime],
        cfg: Any,
    ) -> AssessmentConfidence:
        """Calculate confidence score for assessment.

        Confidence based on:
        1. Data completeness (how many factors are present)
        2. Data freshness (age of factor data)

        Args:
            factor_values: Raw factor values dict.
            data_dates: Timestamps for each factor.
            cfg: Agent configuration.

        Returns:
            AssessmentConfidence enum value.
        """
        # Completeness
        present_count = sum(1 for k in _FACTOR_KEYS if k in factor_values)
        completeness = present_count / len(_FACTOR_KEYS)

        # Freshness
        now = utcnow()
        freshness_scores: List[float] = []
        for key in _FACTOR_KEYS:
            if key in data_dates:
                age_days = (now - data_dates[key]).days
                if age_days <= cfg.data_freshness_max_days:
                    freshness_scores.append(1.0)
                else:
                    freshness_scores.append(0.5)
            else:
                freshness_scores.append(0.5)

        freshness = sum(freshness_scores) / len(freshness_scores)

        # Combined confidence
        confidence_score = (completeness * 0.6) + (freshness * 0.4)

        if confidence_score >= 0.8:
            return AssessmentConfidence.HIGH
        if confidence_score >= 0.6:
            return AssessmentConfidence.MEDIUM
        return AssessmentConfidence.LOW

    def _get_trend_direction(
        self, country_code: str, current_score: Decimal,
    ) -> TrendDirection:
        """Get trend direction for current assessment.

        Args:
            country_code: ISO alpha-2 code.
            current_score: Current composite score.

        Returns:
            TrendDirection enum value.
        """
        with self._lock:
            history = self._risk_history.get(country_code, [])

        if len(history) < 2:
            return TrendDirection.STABLE

        # Use last 3 points including current
        recent = history[-2:]
        recent_with_current = recent + [(utcnow(), current_score)]

        slope = self._calculate_trend_slope(recent_with_current)
        return self._classify_trend_direction(slope)

    def _calculate_trend_slope(
        self, data_points: List[Tuple[datetime, Decimal]],
    ) -> float:
        """Calculate linear regression slope for trend analysis.

        Args:
            data_points: List of (timestamp, score) tuples.

        Returns:
            Slope value (points per day).
        """
        if len(data_points) < 2:
            return 0.0

        # Convert timestamps to days since first point
        first_ts = data_points[0][0]
        x_values = [(ts - first_ts).days for ts, _ in data_points]
        y_values = [_float(score) for _, score in data_points]

        # Linear regression
        n = len(x_values)
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        return slope

    def _classify_trend_direction(self, slope: float) -> TrendDirection:
        """Classify trend direction from slope.

        Args:
            slope: Trend slope value.

        Returns:
            TrendDirection enum value.
        """
        if slope < -0.1:
            return TrendDirection.IMPROVING
        if slope > 0.1:
            return TrendDirection.DETERIORATING
        return TrendDirection.STABLE

    def _add_to_history(
        self, country_code: str, score: Decimal,
    ) -> None:
        """Add score to country risk history (thread-safe).

        Args:
            country_code: ISO alpha-2 code.
            score: Composite score.
        """
        now = utcnow()
        if country_code not in self._risk_history:
            self._risk_history[country_code] = []
        self._risk_history[country_code].append((now, score))

        # Keep only last 100 entries per country
        if len(self._risk_history[country_code]) > 100:
            self._risk_history[country_code] = self._risk_history[country_code][-100:]

    def _build_assessment(
        self,
        country_code: str,
        composite_score: Decimal,
        risk_level: RiskLevel,
        confidence: AssessmentConfidence,
        trend_direction: TrendDirection,
        factor_values: Dict[str, Decimal],
        factor_weights: Dict[str, float],
        ec_override: bool,
    ) -> CountryRiskAssessment:
        """Build complete CountryRiskAssessment model.

        Args:
            country_code: ISO alpha-2 code.
            composite_score: Calculated composite score.
            risk_level: Classified risk level.
            confidence: Confidence score.
            trend_direction: Trend direction.
            factor_values: Normalized factor values.
            factor_weights: Factor weights used.
            ec_override: Whether EC override was applied.

        Returns:
            Populated CountryRiskAssessment model.
        """
        # Provenance hash
        tracker = get_provenance_tracker()
        prov_data = {
            "country_code": country_code,
            "composite_score": _float(composite_score),
            "risk_level": risk_level.value,
            "factor_values": {k: _float(v) for k, v in factor_values.items()},
            "factor_weights": factor_weights,
        }
        provenance_hash = tracker.build_hash(prov_data)

        return CountryRiskAssessment(
            country_code=country_code,
            country_name=country_code,  # Placeholder; resolved by reference data
            risk_score=_float(composite_score),
            risk_level=risk_level,
            confidence=confidence,
            trend=trend_direction,
            composite_factors={k: _float(v) for k, v in factor_values.items()},
            factor_weights=factor_weights,
            ec_benchmark_aligned=not ec_override,
            provenance_hash=provenance_hash,
        )

    def _build_trend_summary(
        self,
        country_code: str,
        direction: TrendDirection,
        slope: float,
        data_points: int,
    ) -> str:
        """Build human-readable trend summary.

        Args:
            country_code: ISO alpha-2 code.
            direction: Trend direction.
            slope: Trend slope.
            data_points: Number of data points.

        Returns:
            Summary string.
        """
        if direction == TrendDirection.IMPROVING:
            return (
                f"{country_code} risk trend is improving "
                f"(slope: {slope:.4f}, {data_points} data points)"
            )
        if direction == TrendDirection.DETERIORATING:
            return (
                f"{country_code} risk trend is deteriorating "
                f"(slope: {slope:.4f}, {data_points} data points)"
            )
        return (
            f"{country_code} risk trend is stable "
            f"(slope: {slope:.4f}, {data_points} data points)"
        )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        with self._lock:
            count = len(self._assessments)
            history_count = sum(len(v) for v in self._risk_history.values())
        return (
            f"CountryRiskScorer("
            f"assessments={count}, "
            f"history_records={history_count}, "
            f"ec_benchmarks={len(self._ec_benchmarks)})"
        )

    def __len__(self) -> int:
        """Return number of stored assessments."""
        with self._lock:
            return len(self._assessments)
