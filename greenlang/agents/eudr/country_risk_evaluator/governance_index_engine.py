# -*- coding: utf-8 -*-
"""
Governance Index Engine - AGENT-EUDR-016 Engine 4

Multi-source governance quality evaluation using World Bank Worldwide
Governance Indicators (WGI) 6 dimensions, Transparency International
Corruption Perceptions Index (CPI), and forest-specific governance
indicators from FAO/ITTO. Integrates enforcement effectiveness scoring
based on prosecution rates and penalty adequacy, composite governance
score calculation with configurable weights, data source freshness
tracking and staleness detection, historical governance trend analysis,
regional governance benchmarking, and gap analysis to identify weakest
governance dimensions.

World Bank WGI 6 Dimensions (percentile rank 0-100):
    1. Voice and Accountability (civil liberties, political rights)
    2. Political Stability and Absence of Violence
    3. Government Effectiveness (quality of public services)
    4. Regulatory Quality (policy formulation and implementation)
    5. Rule of Law (contract enforcement, property rights, courts)
    6. Control of Corruption (public power for private gain)

Transparency International CPI:
    - Score 0-100 (0 = highly corrupt, 100 = very clean)
    - Aggregates 13 data sources from 12 institutions
    - Published annually for 180+ countries

Forest Governance Quality:
    - Forest law quality (0-100): comprehensiveness of forest legislation
    - Enforcement capacity (0-100): resources, personnel, technology
    - Institutional strength (0-100): inter-agency coordination, capacity

Enforcement Effectiveness:
    - Prosecution rate: (cases prosecuted / violations detected)
    - Penalty adequacy: (penalties collected / penalties imposed)
    - Composite: (prosecution_rate * 0.6) + (penalty_adequacy * 0.4)

Composite Governance Score:
    governance_score = (WGI * wgi_weight) + (CPI * cpi_weight) +
                       (forest_gov * forest_weight) +
                       (enforcement * enforcement_weight)
    Default weights: WGI 40%, CPI 30%, forest_gov 20%, enforcement 10%

Zero-Hallucination: All governance scores are deterministic arithmetic
    from published indices and database lookups. No LLM calls.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-016 Country Risk Evaluator (GL-EUDR-CRE-016)
Status: Production Ready
"""

from __future__ import annotations

import logging
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
    GovernanceIndex,
    GovernanceIndicator,
    TrendDirection,
)
from .provenance import get_provenance_tracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Composite governance score weights
_DEFAULT_WEIGHTS: Dict[str, Decimal] = {
    "wgi": Decimal("0.40"),
    "cpi": Decimal("0.30"),
    "forest_governance": Decimal("0.20"),
    "enforcement": Decimal("0.10"),
}

#: WGI dimension names (6 dimensions)
_WGI_DIMENSIONS: List[str] = [
    "voice_accountability",
    "political_stability",
    "government_effectiveness",
    "regulatory_quality",
    "rule_of_law",
    "control_of_corruption",
]

#: Forest governance indicators (3 dimensions)
_FOREST_INDICATORS: List[str] = [
    "forest_law_quality",
    "enforcement_capacity",
    "institutional_strength",
]


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
# GovernanceIndexEngine
# ---------------------------------------------------------------------------


class GovernanceIndexEngine:
    """Multi-source governance quality evaluation engine.

    Integrates World Bank WGI, Transparency International CPI, and
    forest-specific governance indicators to produce composite governance
    scores, assess enforcement effectiveness, track data freshness,
    analyze historical trends, benchmark regional performance, and
    identify governance gaps.

    All scoring operations use Decimal arithmetic for zero floating-point
    drift and deterministic reproducibility.

    Attributes:
        _evaluations: In-memory store of governance evaluations keyed by
            evaluation_id.
        _wgi_data: WGI dimension scores by country_code.
        _cpi_data: CPI scores by country_code.
        _forest_gov_data: Forest governance scores by country_code.
        _enforcement_data: Enforcement effectiveness data by country_code.
        _governance_history: Historical governance scores by country_code.
        _lock: Threading lock for thread-safe access.

    Example:
        >>> engine = GovernanceIndexEngine()
        >>> result = engine.evaluate_governance("BR")
        >>> assert 0.0 <= result.overall_score <= 100.0
        >>> assert len(result.wgi_scores) == 6
    """

    def __init__(self) -> None:
        """Initialize GovernanceIndexEngine with empty stores."""
        self._evaluations: Dict[str, GovernanceIndex] = {}
        self._wgi_data: Dict[str, Dict[str, float]] = {}
        self._cpi_data: Dict[str, float] = {}
        self._forest_gov_data: Dict[str, Dict[str, float]] = {}
        self._enforcement_data: Dict[str, Dict[str, float]] = {}
        self._governance_history: Dict[str, List[Tuple[datetime, Decimal]]] = {}
        self._lock: threading.Lock = threading.Lock()
        logger.info(
            "GovernanceIndexEngine initialized: "
            "wgi_dimensions=%d, forest_indicators=%d",
            len(_WGI_DIMENSIONS),
            len(_FOREST_INDICATORS),
        )

    # ------------------------------------------------------------------
    # Primary evaluation
    # ------------------------------------------------------------------

    def evaluate_governance(
        self,
        country_code: str,
        wgi_scores: Optional[Dict[str, float]] = None,
        cpi_score: Optional[float] = None,
        forest_governance_scores: Optional[Dict[str, float]] = None,
        enforcement_data: Optional[Dict[str, float]] = None,
    ) -> GovernanceIndex:
        """Evaluate governance quality for a country.

        Applies the following evaluation pipeline:
        1. Validate inputs (country code).
        2. Load WGI scores (from args or cache).
        3. Load CPI score (from args or cache).
        4. Load forest governance scores (from args or cache).
        5. Load enforcement data (from args or cache).
        6. Calculate composite governance score.
        7. Assess data completeness and freshness.
        8. Determine governance trend.
        9. Store evaluation and record provenance/metrics.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            wgi_scores: Optional WGI dimension scores (0-100). Keys:
                voice_accountability, political_stability,
                government_effectiveness, regulatory_quality, rule_of_law,
                control_of_corruption.
            cpi_score: Optional Transparency International CPI score (0-100).
            forest_governance_scores: Optional forest governance scores (0-100).
                Keys: forest_law_quality, enforcement_capacity,
                institutional_strength.
            enforcement_data: Optional enforcement effectiveness data.
                Keys: prosecution_rate, penalty_adequacy (0.0-1.0).

        Returns:
            GovernanceIndex with composite_score, wgi_scores, cpi_score,
            forest_governance_scores, enforcement_score, confidence,
            and trend_direction.

        Raises:
            ValueError: If country_code is empty.
        """
        start = time.monotonic()
        cfg = get_config()

        # -- Input validation ------------------------------------------------
        country_code = self._validate_country_code(country_code)

        # -- Load/cache data -------------------------------------------------
        wgi = self._load_wgi_scores(country_code, wgi_scores)
        cpi = self._load_cpi_score(country_code, cpi_score)
        forest_gov = self._load_forest_governance(
            country_code, forest_governance_scores,
        )
        enforcement = self._load_enforcement_data(
            country_code, enforcement_data,
        )

        # -- Composite scores ------------------------------------------------
        wgi_composite = self._calculate_wgi_composite(wgi)
        forest_gov_composite = self._calculate_forest_gov_composite(forest_gov)
        enforcement_score = self._calculate_enforcement_score(enforcement)

        # -- Overall composite -----------------------------------------------
        composite_score = self._calculate_composite_score(
            wgi_composite=wgi_composite,
            cpi_score=_decimal(cpi) if cpi is not None else None,
            forest_gov_composite=forest_gov_composite,
            enforcement_score=enforcement_score,
            cfg=cfg,
        )

        # -- Confidence ------------------------------------------------------
        confidence = self._calculate_confidence(
            wgi, cpi, forest_gov, enforcement,
        )

        # -- Trend analysis --------------------------------------------------
        trend_direction = self._get_trend_direction(
            country_code, composite_score,
        )

        # -- Build evaluation ------------------------------------------------
        evaluation = self._build_evaluation(
            country_code=country_code,
            composite_score=composite_score,
            wgi_scores=wgi,
            cpi_score=cpi,
            forest_governance_scores=forest_gov,
            enforcement_score=enforcement_score,
            confidence=confidence,
            trend_direction=trend_direction,
        )

        # -- Store and history update ----------------------------------------
        with self._lock:
            self._evaluations[evaluation.index_id] = evaluation
            self._add_to_history(country_code, composite_score)

        # -- Provenance ------------------------------------------------------
        tracker = get_provenance_tracker()
        tracker.record(
            entity_type="governance_evaluation",
            action="evaluate",
            entity_id=evaluation.index_id,
            data=evaluation.model_dump(mode="json"),
            metadata={
                "country_code": country_code,
                "composite_score": _float(composite_score),
                "confidence": confidence.value,
            },
        )

        # -- Metrics ---------------------------------------------------------
        elapsed = time.monotonic() - start
        observe_assessment_duration(elapsed)
        record_assessment_completed("governance")

        logger.info(
            "Governance evaluated: country=%s composite=%.2f "
            "wgi=%.2f cpi=%.1f forest=%.2f enforcement=%.2f "
            "confidence=%s trend=%s elapsed_ms=%.1f",
            country_code,
            _float(composite_score),
            _float(wgi_composite) if wgi_composite else 0.0,
            cpi if cpi else 0.0,
            _float(forest_gov_composite) if forest_gov_composite else 0.0,
            _float(enforcement_score) if enforcement_score else 0.0,
            confidence.value,
            trend_direction.value,
            elapsed * 1000,
        )
        return evaluation

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_evaluation(
        self, evaluation_id: str,
    ) -> Optional[GovernanceIndex]:
        """Retrieve a governance evaluation by its unique identifier.

        Args:
            evaluation_id: The evaluation_id to look up.

        Returns:
            GovernanceIndex if found, None otherwise.
        """
        with self._lock:
            return self._evaluations.get(evaluation_id)

    def list_evaluations(
        self,
        country_code: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[GovernanceIndex]:
        """List governance evaluations with optional filters.

        Args:
            country_code: Optional country code filter.
            limit: Maximum number of results (default 100).
            offset: Pagination offset (default 0).

        Returns:
            Filtered list of GovernanceIndex objects.
        """
        with self._lock:
            results = list(self._evaluations.values())

        if country_code:
            cc = country_code.upper().strip()
            results = [e for e in results if e.country_code == cc]

        # Sort by evaluation timestamp descending
        results.sort(key=lambda e: e.assessed_at, reverse=True)

        return results[offset:offset + limit]

    # ------------------------------------------------------------------
    # WGI scores
    # ------------------------------------------------------------------

    def get_wgi_scores(
        self, country_code: str,
    ) -> Dict[str, float]:
        """Get World Bank WGI scores for a country.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            Dictionary with WGI dimension scores (0-100).

        Raises:
            ValueError: If country_code is empty.
        """
        country_code = self._validate_country_code(country_code)
        with self._lock:
            return self._wgi_data.get(country_code, {})

    def set_wgi_scores(
        self,
        country_code: str,
        wgi_scores: Dict[str, float],
    ) -> None:
        """Set World Bank WGI scores for a country.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            wgi_scores: Dictionary with WGI dimension scores (0-100).

        Raises:
            ValueError: If country_code is empty or wgi_scores invalid.
        """
        country_code = self._validate_country_code(country_code)
        self._validate_wgi_scores(wgi_scores)

        with self._lock:
            self._wgi_data[country_code] = dict(wgi_scores)

        logger.info(
            "WGI scores set: country=%s dimensions=%d",
            country_code, len(wgi_scores),
        )

    # ------------------------------------------------------------------
    # CPI score
    # ------------------------------------------------------------------

    def get_cpi_score(
        self, country_code: str,
    ) -> Optional[float]:
        """Get Transparency International CPI score for a country.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            CPI score (0-100) or None if not set.

        Raises:
            ValueError: If country_code is empty.
        """
        country_code = self._validate_country_code(country_code)
        with self._lock:
            return self._cpi_data.get(country_code)

    def set_cpi_score(
        self,
        country_code: str,
        cpi_score: float,
    ) -> None:
        """Set Transparency International CPI score for a country.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            cpi_score: CPI score (0-100).

        Raises:
            ValueError: If country_code is empty or cpi_score invalid.
        """
        country_code = self._validate_country_code(country_code)
        if cpi_score < 0.0 or cpi_score > 100.0:
            raise ValueError(
                f"cpi_score must be in [0, 100], got {cpi_score}"
            )

        with self._lock:
            self._cpi_data[country_code] = cpi_score

        logger.info(
            "CPI score set: country=%s score=%.1f",
            country_code, cpi_score,
        )

    # ------------------------------------------------------------------
    # Forest governance
    # ------------------------------------------------------------------

    def assess_forest_governance(
        self,
        country_code: str,
        forest_law_quality: float,
        enforcement_capacity: float,
        institutional_strength: float,
    ) -> Dict[str, Any]:
        """Assess forest governance quality for a country.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            forest_law_quality: Forest law quality score (0-100).
            enforcement_capacity: Enforcement capacity score (0-100).
            institutional_strength: Institutional strength score (0-100).

        Returns:
            Dictionary with individual scores, composite score, and
            assessment category (weak/moderate/strong).

        Raises:
            ValueError: If country_code is empty or scores invalid.
        """
        country_code = self._validate_country_code(country_code)
        self._validate_score(forest_law_quality, "forest_law_quality")
        self._validate_score(enforcement_capacity, "enforcement_capacity")
        self._validate_score(institutional_strength, "institutional_strength")

        scores = {
            "forest_law_quality": forest_law_quality,
            "enforcement_capacity": enforcement_capacity,
            "institutional_strength": institutional_strength,
        }

        composite = self._calculate_forest_gov_composite(scores)

        category = "weak"
        if composite and _float(composite) >= 70.0:
            category = "strong"
        elif composite and _float(composite) >= 40.0:
            category = "moderate"

        with self._lock:
            self._forest_gov_data[country_code] = scores

        return {
            "country_code": country_code,
            "forest_law_quality": forest_law_quality,
            "enforcement_capacity": enforcement_capacity,
            "institutional_strength": institutional_strength,
            "composite_score": _float(composite) if composite else None,
            "category": category,
        }

    # ------------------------------------------------------------------
    # Enforcement effectiveness
    # ------------------------------------------------------------------

    def assess_enforcement(
        self,
        country_code: str,
        prosecution_rate: float,
        penalty_adequacy: float,
    ) -> Dict[str, Any]:
        """Assess enforcement effectiveness for a country.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            prosecution_rate: Prosecution rate (0.0-1.0).
            penalty_adequacy: Penalty adequacy (0.0-1.0).

        Returns:
            Dictionary with prosecution_rate, penalty_adequacy,
            enforcement_score (0-100), and effectiveness_category
            (low/moderate/high).

        Raises:
            ValueError: If country_code is empty or rates invalid.
        """
        country_code = self._validate_country_code(country_code)
        if prosecution_rate < 0.0 or prosecution_rate > 1.0:
            raise ValueError(
                f"prosecution_rate must be in [0.0, 1.0], got {prosecution_rate}"
            )
        if penalty_adequacy < 0.0 or penalty_adequacy > 1.0:
            raise ValueError(
                f"penalty_adequacy must be in [0.0, 1.0], got {penalty_adequacy}"
            )

        enforcement_data = {
            "prosecution_rate": prosecution_rate,
            "penalty_adequacy": penalty_adequacy,
        }

        enforcement_score = self._calculate_enforcement_score(enforcement_data)

        category = "low"
        if enforcement_score and _float(enforcement_score) >= 70.0:
            category = "high"
        elif enforcement_score and _float(enforcement_score) >= 40.0:
            category = "moderate"

        with self._lock:
            self._enforcement_data[country_code] = enforcement_data

        return {
            "country_code": country_code,
            "prosecution_rate": prosecution_rate,
            "penalty_adequacy": penalty_adequacy,
            "enforcement_score": _float(enforcement_score) if enforcement_score else None,
            "effectiveness_category": category,
        }

    # ------------------------------------------------------------------
    # Composite calculation
    # ------------------------------------------------------------------

    def calculate_composite(
        self,
        wgi_composite: Optional[float],
        cpi_score: Optional[float],
        forest_gov_composite: Optional[float],
        enforcement_score: Optional[float],
    ) -> float:
        """Calculate overall composite governance score.

        Args:
            wgi_composite: WGI composite score (0-100).
            cpi_score: CPI score (0-100).
            forest_gov_composite: Forest governance composite (0-100).
            enforcement_score: Enforcement score (0-100).

        Returns:
            Composite governance score (0-100).
        """
        cfg = get_config()
        composite = self._calculate_composite_score(
            wgi_composite=_decimal(wgi_composite) if wgi_composite is not None else None,
            cpi_score=_decimal(cpi_score) if cpi_score is not None else None,
            forest_gov_composite=_decimal(forest_gov_composite) if forest_gov_composite is not None else None,
            enforcement_score=_decimal(enforcement_score) if enforcement_score is not None else None,
            cfg=cfg,
        )
        return _float(composite)

    # ------------------------------------------------------------------
    # Trend analysis
    # ------------------------------------------------------------------

    def get_governance_trend(
        self,
        country_code: str,
        window_years: int = 3,
    ) -> Dict[str, Any]:
        """Get historical governance trend for a country.

        Analyzes governance score trend over a rolling window using
        linear regression to determine direction (improving, stable,
        deteriorating).

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

        cutoff_date = _utcnow() - timedelta(days=window_years * 365)

        with self._lock:
            history = self._governance_history.get(country_code, [])

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

        # Linear regression (simplified)
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
    # Regional benchmarking
    # ------------------------------------------------------------------

    def benchmark_regional(
        self,
        country_code: str,
        region_countries: List[str],
    ) -> Dict[str, Any]:
        """Benchmark country governance against regional peers.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            region_countries: List of peer country codes in the region.

        Returns:
            Dictionary with country_score, regional_mean, regional_median,
            country_rank, percentile, and comparison.

        Raises:
            ValueError: If country_code is empty or region_countries empty.
        """
        country_code = self._validate_country_code(country_code)
        if not region_countries:
            raise ValueError("region_countries list must not be empty")

        # Get latest evaluation for target country
        country_eval = self._get_latest_evaluation(country_code)
        if not country_eval:
            raise ValueError(f"No governance evaluation found for {country_code}")

        country_score = country_eval.overall_score

        # Get scores for region countries
        regional_scores: List[Tuple[str, float]] = []
        for cc in region_countries:
            cc_upper = cc.upper().strip()
            eval_data = self._get_latest_evaluation(cc_upper)
            if eval_data:
                regional_scores.append((cc_upper, eval_data.overall_score))

        if not regional_scores:
            raise ValueError("No governance evaluations found for region countries")

        scores_only = [s for _, s in regional_scores]
        regional_mean = sum(scores_only) / len(scores_only)

        sorted_scores = sorted(scores_only)
        n = len(sorted_scores)
        if n % 2 == 0:
            regional_median = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
        else:
            regional_median = sorted_scores[n // 2]

        # Rank (1 = best)
        sorted_scores_desc = sorted(scores_only, reverse=True)
        country_rank = sorted_scores_desc.index(country_score) + 1
        percentile = (country_rank / len(sorted_scores_desc)) * 100

        comparison = "below_average"
        if country_score >= regional_mean:
            comparison = "above_average"
        if country_score >= regional_median:
            comparison = "above_median"

        return {
            "country_code": country_code,
            "country_score": country_score,
            "regional_mean": round(regional_mean, 2),
            "regional_median": round(regional_median, 2),
            "country_rank": country_rank,
            "percentile": round(percentile, 1),
            "comparison": comparison,
            "peer_count": len(regional_scores),
        }

    # ------------------------------------------------------------------
    # Gap analysis
    # ------------------------------------------------------------------

    def analyze_gaps(
        self, country_code: str,
    ) -> Dict[str, Any]:
        """Identify weakest governance dimensions for a country.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            Dictionary with weakest_dimensions (list of tuples with
            dimension name and score), gap_severity (low/moderate/high),
            and recommendations.

        Raises:
            ValueError: If country_code is empty or no evaluation found.
        """
        country_code = self._validate_country_code(country_code)

        eval_data = self._get_latest_evaluation(country_code)
        if not eval_data:
            raise ValueError(f"No governance evaluation found for {country_code}")

        # Collect all dimension scores
        dimension_scores: List[Tuple[str, float]] = []

        if eval_data.wgi_scores:
            for dim, score in eval_data.wgi_scores.items():
                dimension_scores.append((f"wgi_{dim}", score))

        if eval_data.cpi_score is not None:
            dimension_scores.append(("cpi", eval_data.cpi_score))

        if eval_data.forest_governance_scores:
            for dim, score in eval_data.forest_governance_scores.items():
                dimension_scores.append((f"forest_{dim}", score))

        if eval_data.enforcement_score is not None:
            dimension_scores.append(("enforcement", eval_data.enforcement_score))

        # Sort by score ascending (weakest first)
        dimension_scores.sort(key=lambda x: x[1])

        # Take bottom 3
        weakest = dimension_scores[:3]

        # Determine gap severity
        weakest_score = weakest[0][1] if weakest else 100.0
        gap_severity = "low"
        if weakest_score < 30.0:
            gap_severity = "high"
        elif weakest_score < 50.0:
            gap_severity = "moderate"

        recommendations = self._build_gap_recommendations(weakest, gap_severity)

        return {
            "country_code": country_code,
            "weakest_dimensions": [
                {"dimension": dim, "score": score}
                for dim, score in weakest
            ],
            "gap_severity": gap_severity,
            "recommendations": recommendations,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_country_code(self, country_code: str) -> str:
        """Validate and normalize country code."""
        if not country_code or not country_code.strip():
            raise ValueError("country_code must not be empty")
        cc = country_code.upper().strip()
        if len(cc) != 2:
            raise ValueError(
                f"country_code must be 2 characters, got '{cc}'"
            )
        return cc

    def _validate_wgi_scores(self, wgi_scores: Dict[str, float]) -> None:
        """Validate WGI scores dictionary."""
        for dim in _WGI_DIMENSIONS:
            if dim not in wgi_scores:
                raise ValueError(f"Missing WGI dimension '{dim}'")
            score = wgi_scores[dim]
            if score < 0.0 or score > 100.0:
                raise ValueError(
                    f"WGI score for '{dim}' must be in [0, 100], got {score}"
                )

    def _validate_score(self, score: float, name: str) -> None:
        """Validate a score is in [0, 100]."""
        if score < 0.0 or score > 100.0:
            raise ValueError(
                f"{name} must be in [0, 100], got {score}"
            )

    def _load_wgi_scores(
        self, country_code: str, wgi_scores: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """Load WGI scores from args or cache."""
        if wgi_scores is not None:
            self._validate_wgi_scores(wgi_scores)
            return dict(wgi_scores)

        with self._lock:
            return self._wgi_data.get(country_code, {})

    def _load_cpi_score(
        self, country_code: str, cpi_score: Optional[float],
    ) -> Optional[float]:
        """Load CPI score from args or cache."""
        if cpi_score is not None:
            self._validate_score(cpi_score, "cpi_score")
            return cpi_score

        with self._lock:
            return self._cpi_data.get(country_code)

    def _load_forest_governance(
        self, country_code: str, forest_scores: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """Load forest governance scores from args or cache."""
        if forest_scores is not None:
            for key in _FOREST_INDICATORS:
                if key in forest_scores:
                    self._validate_score(forest_scores[key], key)
            return dict(forest_scores)

        with self._lock:
            return self._forest_gov_data.get(country_code, {})

    def _load_enforcement_data(
        self, country_code: str, enforcement_data: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """Load enforcement data from args or cache."""
        if enforcement_data is not None:
            return dict(enforcement_data)

        with self._lock:
            return self._enforcement_data.get(country_code, {})

    def _calculate_wgi_composite(
        self, wgi_scores: Dict[str, float],
    ) -> Optional[Decimal]:
        """Calculate WGI composite score (average of 6 dimensions)."""
        if not wgi_scores:
            return None

        total = Decimal("0")
        count = 0
        for dim in _WGI_DIMENSIONS:
            if dim in wgi_scores:
                total += _decimal(wgi_scores[dim])
                count += 1

        if count == 0:
            return None

        composite = total / Decimal(str(count))
        return composite.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _calculate_forest_gov_composite(
        self, forest_scores: Dict[str, float],
    ) -> Optional[Decimal]:
        """Calculate forest governance composite score."""
        if not forest_scores:
            return None

        total = Decimal("0")
        count = 0
        for ind in _FOREST_INDICATORS:
            if ind in forest_scores:
                total += _decimal(forest_scores[ind])
                count += 1

        if count == 0:
            return None

        composite = total / Decimal(str(count))
        return composite.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _calculate_enforcement_score(
        self, enforcement_data: Dict[str, float],
    ) -> Optional[Decimal]:
        """Calculate enforcement effectiveness score."""
        if not enforcement_data:
            return None

        prosecution_rate = enforcement_data.get("prosecution_rate")
        penalty_adequacy = enforcement_data.get("penalty_adequacy")

        if prosecution_rate is None or penalty_adequacy is None:
            return None

        # Composite: (prosecution * 0.6) + (penalty * 0.4)
        score = (
            (_decimal(prosecution_rate) * Decimal("0.6")) +
            (_decimal(penalty_adequacy) * Decimal("0.4"))
        ) * Decimal("100")

        return score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _calculate_composite_score(
        self,
        wgi_composite: Optional[Decimal],
        cpi_score: Optional[Decimal],
        forest_gov_composite: Optional[Decimal],
        enforcement_score: Optional[Decimal],
        cfg: Any,
    ) -> Decimal:
        """Calculate overall composite governance score."""
        weights = {
            "wgi": _decimal(cfg.wgi_weight / 100.0),
            "cpi": _decimal(cfg.cpi_weight / 100.0),
            "forest_governance": _decimal(cfg.forest_governance_weight / 100.0),
            "enforcement": _decimal(cfg.gov_enforcement_weight / 100.0),
        }

        composite = Decimal("0")
        total_weight = Decimal("0")

        if wgi_composite is not None:
            composite += wgi_composite * weights["wgi"]
            total_weight += weights["wgi"]

        if cpi_score is not None:
            composite += cpi_score * weights["cpi"]
            total_weight += weights["cpi"]

        if forest_gov_composite is not None:
            composite += forest_gov_composite * weights["forest_governance"]
            total_weight += weights["forest_governance"]

        if enforcement_score is not None:
            composite += enforcement_score * weights["enforcement"]
            total_weight += weights["enforcement"]

        # Normalize by total weight (in case some components missing)
        if total_weight > Decimal("0"):
            composite = composite / total_weight

        return composite.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _calculate_confidence(
        self,
        wgi_scores: Dict[str, float],
        cpi_score: Optional[float],
        forest_gov: Dict[str, float],
        enforcement: Dict[str, float],
    ) -> AssessmentConfidence:
        """Calculate confidence based on data completeness."""
        total_components = 4
        present_components = 0

        if wgi_scores:
            present_components += 1
        if cpi_score is not None:
            present_components += 1
        if forest_gov:
            present_components += 1
        if enforcement:
            present_components += 1

        completeness = present_components / total_components

        if completeness >= 0.75:
            return AssessmentConfidence.HIGH
        if completeness >= 0.5:
            return AssessmentConfidence.MEDIUM
        return AssessmentConfidence.LOW

    def _get_trend_direction(
        self, country_code: str, current_score: Decimal,
    ) -> TrendDirection:
        """Get trend direction for current evaluation."""
        with self._lock:
            history = self._governance_history.get(country_code, [])

        if len(history) < 2:
            return TrendDirection.STABLE

        # Use last 2 points including current
        recent = history[-2:]
        recent_with_current = recent + [(_utcnow(), current_score)]

        slope = self._calculate_trend_slope(recent_with_current)
        return self._classify_trend_direction(slope)

    def _calculate_trend_slope(
        self, data_points: List[Tuple[datetime, Decimal]],
    ) -> float:
        """Calculate linear regression slope for trend analysis."""
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
        """Classify trend direction from slope."""
        if slope > 0.1:
            return TrendDirection.IMPROVING
        if slope < -0.1:
            return TrendDirection.DETERIORATING
        return TrendDirection.STABLE

    def _add_to_history(
        self, country_code: str, score: Decimal,
    ) -> None:
        """Add score to governance history (thread-safe)."""
        now = _utcnow()
        if country_code not in self._governance_history:
            self._governance_history[country_code] = []
        self._governance_history[country_code].append((now, score))

        # Keep only last 100 entries
        if len(self._governance_history[country_code]) > 100:
            self._governance_history[country_code] = (
                self._governance_history[country_code][-100:]
            )

    def _get_latest_evaluation(
        self, country_code: str,
    ) -> Optional[GovernanceIndex]:
        """Get most recent evaluation for a country."""
        with self._lock:
            country_evals = [
                e for e in self._evaluations.values()
                if e.country_code == country_code
            ]

        if not country_evals:
            return None

        country_evals.sort(key=lambda e: e.assessed_at, reverse=True)
        return country_evals[0]

    def _build_evaluation(
        self,
        country_code: str,
        composite_score: Decimal,
        wgi_scores: Dict[str, float],
        cpi_score: Optional[float],
        forest_governance_scores: Dict[str, float],
        enforcement_score: Optional[Decimal],
        confidence: AssessmentConfidence,
        trend_direction: TrendDirection,
    ) -> GovernanceIndex:
        """Build complete GovernanceIndex model."""
        # Provenance hash
        tracker = get_provenance_tracker()
        prov_data = {
            "country_code": country_code,
            "composite_score": _float(composite_score),
            "wgi_scores": wgi_scores,
            "cpi_score": cpi_score,
        }
        provenance_hash = tracker.build_hash(prov_data)

        return GovernanceIndex(
            country_code=country_code,
            overall_score=_float(composite_score),
            indicators=wgi_scores if wgi_scores else {},
            cpi_score=cpi_score,
            forest_governance_score=(
                sum(forest_governance_scores.values()) / len(forest_governance_scores)
                if forest_governance_scores else None
            ),
            enforcement_effectiveness=(
                _float(enforcement_score) if enforcement_score else None
            ),
            provenance_hash=provenance_hash,
        )

    def _build_trend_summary(
        self,
        country_code: str,
        direction: TrendDirection,
        slope: float,
        data_points: int,
    ) -> str:
        """Build human-readable trend summary."""
        if direction == TrendDirection.IMPROVING:
            return (
                f"{country_code} governance trend is improving "
                f"(slope: {slope:.4f}, {data_points} data points)"
            )
        if direction == TrendDirection.DETERIORATING:
            return (
                f"{country_code} governance trend is deteriorating "
                f"(slope: {slope:.4f}, {data_points} data points)"
            )
        return (
            f"{country_code} governance trend is stable "
            f"(slope: {slope:.4f}, {data_points} data points)"
        )

    def _build_gap_recommendations(
        self,
        weakest_dimensions: List[Tuple[str, float]],
        gap_severity: str,
    ) -> List[str]:
        """Build recommendations for addressing governance gaps."""
        recommendations: List[str] = []

        if gap_severity == "high":
            recommendations.append(
                "URGENT: Critical governance deficiencies identified. "
                "Enhanced due diligence required."
            )

        for dim, score in weakest_dimensions:
            if "wgi_voice" in dim:
                recommendations.append(
                    f"Improve civil liberties and political rights (score: {score:.1f})"
                )
            elif "wgi_political" in dim:
                recommendations.append(
                    f"Enhance political stability mechanisms (score: {score:.1f})"
                )
            elif "wgi_government" in dim:
                recommendations.append(
                    f"Strengthen government effectiveness and public services (score: {score:.1f})"
                )
            elif "wgi_regulatory" in dim:
                recommendations.append(
                    f"Improve regulatory quality and policy implementation (score: {score:.1f})"
                )
            elif "wgi_rule" in dim:
                recommendations.append(
                    f"Strengthen rule of law and judicial system (score: {score:.1f})"
                )
            elif "wgi_control" in dim:
                recommendations.append(
                    f"Enhance anti-corruption measures (score: {score:.1f})"
                )
            elif "cpi" in dim:
                recommendations.append(
                    f"Address corruption issues (CPI score: {score:.1f})"
                )
            elif "forest_law" in dim:
                recommendations.append(
                    f"Strengthen forest legislation (score: {score:.1f})"
                )
            elif "enforcement_capacity" in dim:
                recommendations.append(
                    f"Increase enforcement resources and capacity (score: {score:.1f})"
                )
            elif "institutional" in dim:
                recommendations.append(
                    f"Improve inter-agency coordination (score: {score:.1f})"
                )
            elif "enforcement" == dim:
                recommendations.append(
                    f"Improve prosecution and penalty enforcement (score: {score:.1f})"
                )

        return recommendations

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        with self._lock:
            count = len(self._evaluations)
            history_count = sum(len(v) for v in self._governance_history.values())
        return (
            f"GovernanceIndexEngine("
            f"evaluations={count}, "
            f"history_records={history_count})"
        )

    def __len__(self) -> int:
        """Return number of stored evaluations."""
        with self._lock:
            return len(self._evaluations)
