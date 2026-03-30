# -*- coding: utf-8 -*-
"""
BenchmarkingEngine - PACK-002 CSRD Professional Engine 4

Peer comparison and ESG rating alignment engine that enables companies to
benchmark their ESRS disclosures against anonymized sector peers. Provides
percentile ranking, quartile analysis, ESG rating prediction, multi-year
trend analysis, and improvement priority recommendations.

Features:
    - Load anonymized peer benchmark datasets by sector, geography, size
    - Compute percentile rank and quartile for each metric
    - Predict ESG ratings across MSCI, Sustainalytics, and CDP frameworks
    - Analyze multi-year trends with CAGR and volatility
    - Identify improvement priorities based on gap analysis
    - SHA-256 provenance hashing on all benchmark outputs

Zero-Hallucination:
    - All percentile calculations use deterministic statistics
    - ESG predictions use rule-based scoring matrices
    - No LLM involvement in numeric benchmarking
    - CAGR uses standard compound growth formula

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-002 CSRD Professional
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _percentile(values: List[float], pct: float) -> float:
    """Compute the pct-th percentile of a sorted list.

    Uses linear interpolation consistent with numpy's default method.

    Args:
        values: Sorted list of numeric values.
        pct: Percentile (0-100).

    Returns:
        Interpolated percentile value.
    """
    if not values:
        return 0.0
    n = len(values)
    if n == 1:
        return values[0]

    k = (pct / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] * (c - k) + values[c] * (k - f)

def _percentile_rank(values: List[float], target: float) -> float:
    """Compute percentile rank of target within values.

    Args:
        values: List of peer values.
        target: Company value to rank.

    Returns:
        Percentile rank (0-100).
    """
    if not values:
        return 50.0
    below = sum(1 for v in values if v < target)
    equal = sum(1 for v in values if v == target)
    rank = ((below + 0.5 * equal) / len(values)) * 100
    return round(rank, 2)

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class BenchmarkDataset(BaseModel):
    """Anonymized peer benchmark dataset."""

    dataset_id: str = Field(default_factory=_new_uuid, description="Dataset ID")
    sector: str = Field(..., description="NACE sector code or name")
    geography: str = Field(..., description="Geographic region (EU, Global, etc.)")
    company_size: str = Field(
        "all", description="Company size filter (large/medium/small/all)"
    )
    year: int = Field(..., ge=2020, le=2030, description="Reporting year")
    metrics: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Metric name -> list of anonymized peer values",
    )
    peer_count: int = Field(0, ge=0, description="Number of peers in dataset")

    @field_validator("company_size")
    @classmethod
    def validate_size(cls, v: str) -> str:
        """Validate company size category."""
        allowed = {"large", "medium", "small", "all"}
        if v.lower() not in allowed:
            raise ValueError(f"company_size must be one of {allowed}")
        return v.lower()

class PeerComparison(BaseModel):
    """Comparison of a single metric against peers."""

    metric: str = Field(..., description="Metric name")
    company_value: float = Field(..., description="Company's value")
    peer_count: int = Field(0, description="Number of peers compared")
    peer_median: float = Field(0.0, description="Peer median value")
    peer_p25: float = Field(0.0, description="Peer 25th percentile")
    peer_p75: float = Field(0.0, description="Peer 75th percentile")
    peer_min: float = Field(0.0, description="Peer minimum value")
    peer_max: float = Field(0.0, description="Peer maximum value")
    percentile_rank: float = Field(
        0.0, ge=0.0, le=100.0, description="Company percentile rank"
    )
    quartile: int = Field(1, ge=1, le=4, description="Company quartile (1=top)")

class ESGRatingPrediction(BaseModel):
    """Predicted ESG rating for a specific framework."""

    prediction_id: str = Field(default_factory=_new_uuid, description="Prediction ID")
    framework: str = Field(
        ..., description="Rating framework (MSCI, Sustainalytics, CDP)"
    )
    predicted_score: str = Field(
        ..., description="Predicted rating or score"
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Prediction confidence (0-1)"
    )
    key_positive_drivers: List[str] = Field(
        default_factory=list, description="Positive rating drivers"
    )
    key_negative_drivers: List[str] = Field(
        default_factory=list, description="Negative rating drivers"
    )
    improvement_actions: List[str] = Field(
        default_factory=list, description="Actions to improve rating"
    )
    provenance_hash: str = Field("", description="SHA-256 hash")

class TrendAnalysis(BaseModel):
    """Multi-year trend analysis for a metric."""

    metric: str = Field(..., description="Metric name")
    years: List[int] = Field(default_factory=list, description="Years analyzed")
    values: List[float] = Field(default_factory=list, description="Values per year")
    cagr: float = Field(0.0, description="Compound annual growth rate")
    trend_direction: str = Field(
        "stable", description="improving/declining/stable"
    )
    volatility: float = Field(0.0, description="Standard deviation of values")
    projection_next_year: float = Field(0.0, description="Projected next year value")

class BenchmarkReport(BaseModel):
    """Complete benchmark analysis report."""

    report_id: str = Field(default_factory=_new_uuid, description="Report ID")
    comparisons: List[PeerComparison] = Field(
        default_factory=list, description="Peer comparisons"
    )
    esg_predictions: List[ESGRatingPrediction] = Field(
        default_factory=list, description="ESG rating predictions"
    )
    trends: List[TrendAnalysis] = Field(
        default_factory=list, description="Multi-year trends"
    )
    improvement_priorities: List[Dict[str, Any]] = Field(
        default_factory=list, description="Prioritized improvement areas"
    )
    generated_at: datetime = Field(default_factory=utcnow, description="Report time")
    provenance_hash: str = Field("", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class BenchmarkingConfig(BaseModel):
    """Configuration for the benchmarking engine."""

    default_sector: str = Field("all", description="Default sector for benchmarking")
    default_geography: str = Field("EU", description="Default geography")
    min_peer_count: int = Field(
        5, ge=1, description="Minimum peers for valid comparison"
    )
    higher_is_better_metrics: List[str] = Field(
        default_factory=lambda: [
            "renewable_energy_pct",
            "recycling_rate",
            "board_diversity_pct",
            "esrs_disclosure_score",
            "water_reuse_pct",
        ],
        description="Metrics where higher values are better",
    )
    lower_is_better_metrics: List[str] = Field(
        default_factory=lambda: [
            "ghg_intensity",
            "scope1_emissions",
            "scope2_emissions",
            "waste_intensity",
            "water_intensity",
            "injury_rate",
        ],
        description="Metrics where lower values are better",
    )

# ---------------------------------------------------------------------------
# ESG Rating Scoring Matrices
# ---------------------------------------------------------------------------

# MSCI ESG ratings: AAA, AA, A, BBB, BB, B, CCC
_MSCI_SCORING: List[Tuple[float, str]] = [
    (85.0, "AAA"),
    (70.0, "AA"),
    (60.0, "A"),
    (50.0, "BBB"),
    (35.0, "BB"),
    (20.0, "B"),
    (0.0, "CCC"),
]

# Sustainalytics: 0-10 negligible, 10-20 low, 20-30 medium, 30-40 high, 40+ severe
_SUSTAINALYTICS_SCORING: List[Tuple[float, str, str]] = [
    (85.0, "0-10", "Negligible Risk"),
    (70.0, "10-20", "Low Risk"),
    (50.0, "20-30", "Medium Risk"),
    (30.0, "30-40", "High Risk"),
    (0.0, "40+", "Severe Risk"),
]

# CDP: A, A-, B, B-, C, C-, D, D-
_CDP_SCORING: List[Tuple[float, str]] = [
    (90.0, "A"),
    (80.0, "A-"),
    (70.0, "B"),
    (60.0, "B-"),
    (50.0, "C"),
    (40.0, "C-"),
    (25.0, "D"),
    (0.0, "D-"),
]

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BenchmarkingEngine:
    """Peer comparison and ESG rating alignment engine.

    Enables companies to benchmark ESRS disclosures against anonymized
    sector peers, predict ESG ratings, and identify improvement priorities.

    Attributes:
        config: Engine configuration.
        datasets: Loaded benchmark datasets keyed by composite key.

    Example:
        >>> engine = BenchmarkingEngine()
        >>> engine.load_benchmark_data("C20", "EU", 2025)
        >>> comparisons = await engine.compare_to_peers(company_data, "C20", "EU")
    """

    def __init__(self, config: Optional[BenchmarkingConfig] = None) -> None:
        """Initialize BenchmarkingEngine.

        Args:
            config: Engine configuration. Uses defaults if not provided.
        """
        self.config = config or BenchmarkingConfig()
        self.datasets: Dict[str, BenchmarkDataset] = {}
        logger.info("BenchmarkingEngine initialized (version=%s)", _MODULE_VERSION)

    # -- Data Loading -------------------------------------------------------

    def load_benchmark_data(
        self,
        sector: str,
        geography: str,
        year: int,
        metrics: Optional[Dict[str, List[float]]] = None,
        company_size: str = "all",
    ) -> str:
        """Load anonymized peer benchmark data.

        Args:
            sector: NACE sector code.
            geography: Geographic region.
            year: Reporting year.
            metrics: Metric name -> list of peer values.
            company_size: Company size filter.

        Returns:
            Dataset identifier key.
        """
        key = f"{sector}_{geography}_{year}_{company_size}"

        dataset = BenchmarkDataset(
            sector=sector,
            geography=geography,
            company_size=company_size,
            year=year,
            metrics=metrics or {},
            peer_count=max(len(v) for v in (metrics or {}).values()) if metrics else 0,
        )

        self.datasets[key] = dataset
        logger.info(
            "Benchmark data loaded: sector=%s, geo=%s, year=%d, peers=%d, metrics=%d",
            sector,
            geography,
            year,
            dataset.peer_count,
            len(dataset.metrics),
        )
        return key

    # -- Peer Comparison ----------------------------------------------------

    async def compare_to_peers(
        self,
        company_data: Dict[str, float],
        sector: str,
        geography: str,
        year: Optional[int] = None,
        company_size: str = "all",
    ) -> List[PeerComparison]:
        """Compare company metrics against peer benchmarks.

        Args:
            company_data: Company metric values keyed by metric name.
            sector: Sector to compare against.
            geography: Geography to compare against.
            year: Reporting year (uses latest available if None).
            company_size: Company size filter.

        Returns:
            List of PeerComparison for each metric with available peer data.

        Raises:
            ValueError: If no benchmark data available for parameters.
        """
        dataset = self._find_dataset(sector, geography, year, company_size)

        comparisons: List[PeerComparison] = []

        for metric_name, company_value in company_data.items():
            peer_values = dataset.metrics.get(metric_name)
            if not peer_values:
                logger.debug("No peer data for metric '%s', skipping", metric_name)
                continue

            if len(peer_values) < self.config.min_peer_count:
                logger.warning(
                    "Insufficient peers for '%s' (%d < %d)",
                    metric_name,
                    len(peer_values),
                    self.config.min_peer_count,
                )
                continue

            sorted_peers = sorted(peer_values)
            p_rank = _percentile_rank(sorted_peers, company_value)

            # Determine if higher or lower is better for quartile
            is_higher_better = metric_name in self.config.higher_is_better_metrics
            if is_higher_better:
                quartile = self._compute_quartile(p_rank, higher_is_better=True)
            else:
                quartile = self._compute_quartile(p_rank, higher_is_better=False)

            comparison = PeerComparison(
                metric=metric_name,
                company_value=company_value,
                peer_count=len(sorted_peers),
                peer_median=_percentile(sorted_peers, 50.0),
                peer_p25=_percentile(sorted_peers, 25.0),
                peer_p75=_percentile(sorted_peers, 75.0),
                peer_min=sorted_peers[0],
                peer_max=sorted_peers[-1],
                percentile_rank=p_rank,
                quartile=quartile,
            )
            comparisons.append(comparison)

        logger.info(
            "Peer comparison complete: %d metrics compared against %s/%s",
            len(comparisons),
            sector,
            geography,
        )
        return comparisons

    # -- ESG Rating Prediction ----------------------------------------------

    async def predict_esg_rating(
        self,
        company_data: Dict[str, float],
        framework: str,
    ) -> ESGRatingPrediction:
        """Predict ESG rating for a given framework.

        Uses rule-based scoring matrices derived from public rating
        methodologies. This is a directional estimate, not an official rating.

        Args:
            company_data: Company ESG metrics.
            framework: Rating framework (MSCI, Sustainalytics, CDP).

        Returns:
            ESGRatingPrediction with predicted score and drivers.

        Raises:
            ValueError: If framework is not supported.
        """
        framework_upper = framework.upper()
        if framework_upper not in ("MSCI", "SUSTAINALYTICS", "CDP"):
            raise ValueError(
                f"Unsupported framework: {framework}. Use MSCI, Sustainalytics, or CDP."
            )

        composite = self._compute_composite_score(company_data)
        positive_drivers = self._identify_positive_drivers(company_data)
        negative_drivers = self._identify_negative_drivers(company_data)

        if framework_upper == "MSCI":
            predicted_score = self._map_msci_rating(composite)
            confidence = self._calculate_confidence(company_data, "msci")
        elif framework_upper == "SUSTAINALYTICS":
            predicted_score = self._map_sustainalytics_rating(composite)
            confidence = self._calculate_confidence(company_data, "sustainalytics")
        else:
            predicted_score = self._map_cdp_rating(composite)
            confidence = self._calculate_confidence(company_data, "cdp")

        improvement_actions = self._generate_improvement_actions(
            negative_drivers, framework_upper
        )

        prediction = ESGRatingPrediction(
            framework=framework_upper,
            predicted_score=predicted_score,
            confidence=confidence,
            key_positive_drivers=positive_drivers,
            key_negative_drivers=negative_drivers,
            improvement_actions=improvement_actions,
        )
        prediction.provenance_hash = _compute_hash(prediction)

        logger.info(
            "ESG rating prediction: framework=%s, score=%s, confidence=%.2f",
            framework_upper,
            predicted_score,
            confidence,
        )
        return prediction

    # -- Trend Analysis -----------------------------------------------------

    async def analyze_trends(
        self, multi_year_data: Dict[str, Dict[int, float]]
    ) -> List[TrendAnalysis]:
        """Analyze multi-year trends for company metrics.

        Args:
            multi_year_data: Metric name -> {year: value} mapping.

        Returns:
            List of TrendAnalysis for each metric.
        """
        trends: List[TrendAnalysis] = []

        for metric_name, yearly_data in multi_year_data.items():
            if len(yearly_data) < 2:
                logger.debug(
                    "Insufficient years for trend on '%s', skipping", metric_name
                )
                continue

            sorted_years = sorted(yearly_data.keys())
            values = [yearly_data[y] for y in sorted_years]

            cagr = self._calculate_cagr(values[0], values[-1], len(values) - 1)
            volatility = self._calculate_volatility(values)
            direction = self._determine_trend_direction(values, metric_name)
            projection = self._project_next_year(values, cagr)

            trend = TrendAnalysis(
                metric=metric_name,
                years=sorted_years,
                values=values,
                cagr=round(cagr, 4),
                trend_direction=direction,
                volatility=round(volatility, 4),
                projection_next_year=round(projection, 4),
            )
            trends.append(trend)

        logger.info("Trend analysis complete: %d metrics analyzed", len(trends))
        return trends

    # -- Improvement Priorities ---------------------------------------------

    async def identify_improvement_priorities(
        self, comparisons: List[PeerComparison]
    ) -> List[Dict[str, Any]]:
        """Identify and prioritize areas for improvement.

        Ranks metrics by gap-to-peer-median and assigns priority levels.

        Args:
            comparisons: Peer comparison results.

        Returns:
            Prioritized list of improvement areas.
        """
        priorities: List[Dict[str, Any]] = []

        for comp in comparisons:
            is_higher_better = comp.metric in self.config.higher_is_better_metrics
            is_lower_better = comp.metric in self.config.lower_is_better_metrics

            if is_higher_better:
                gap = comp.peer_median - comp.company_value
                needs_improvement = comp.company_value < comp.peer_median
            elif is_lower_better:
                gap = comp.company_value - comp.peer_median
                needs_improvement = comp.company_value > comp.peer_median
            else:
                gap = abs(comp.company_value - comp.peer_median)
                needs_improvement = comp.quartile >= 3

            if not needs_improvement:
                continue

            # Calculate gap percentage
            if comp.peer_median != 0:
                gap_pct = abs(gap / comp.peer_median) * 100
            else:
                gap_pct = 100.0

            if gap_pct > 50:
                priority = "critical"
            elif gap_pct > 25:
                priority = "high"
            elif gap_pct > 10:
                priority = "medium"
            else:
                priority = "low"

            priorities.append({
                "metric": comp.metric,
                "company_value": comp.company_value,
                "peer_median": comp.peer_median,
                "gap": round(gap, 4),
                "gap_pct": round(gap_pct, 2),
                "current_quartile": comp.quartile,
                "priority": priority,
                "direction": "increase" if is_higher_better else "decrease",
                "target_value": comp.peer_p25 if is_higher_better else comp.peer_p25,
            })

        priorities.sort(
            key=lambda p: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(
                p["priority"], 99
            )
        )

        return priorities

    # -- Full Report --------------------------------------------------------

    async def generate_benchmark_report(
        self,
        company_data: Dict[str, float],
        sector: str = "all",
        geography: str = "EU",
        multi_year_data: Optional[Dict[str, Dict[int, float]]] = None,
    ) -> BenchmarkReport:
        """Generate a comprehensive benchmark report.

        Args:
            company_data: Company metric values.
            sector: Sector for peer comparison.
            geography: Geography for peer comparison.
            multi_year_data: Optional multi-year data for trends.

        Returns:
            Complete BenchmarkReport.
        """
        start = utcnow()

        # Peer comparisons
        try:
            comparisons = await self.compare_to_peers(
                company_data, sector, geography
            )
        except ValueError:
            comparisons = []
            logger.warning("No benchmark data available for %s/%s", sector, geography)

        # ESG predictions
        predictions: List[ESGRatingPrediction] = []
        for framework in ("MSCI", "Sustainalytics", "CDP"):
            try:
                pred = await self.predict_esg_rating(company_data, framework)
                predictions.append(pred)
            except Exception as exc:
                logger.warning("ESG prediction failed for %s: %s", framework, exc)

        # Trends
        trends: List[TrendAnalysis] = []
        if multi_year_data:
            trends = await self.analyze_trends(multi_year_data)

        # Priorities
        improvement_priorities = await self.identify_improvement_priorities(comparisons)

        report = BenchmarkReport(
            comparisons=comparisons,
            esg_predictions=predictions,
            trends=trends,
            improvement_priorities=improvement_priorities,
        )
        report.provenance_hash = _compute_hash(report)

        elapsed = (utcnow() - start).total_seconds() * 1000
        logger.info(
            "Benchmark report generated: %d comparisons, %d predictions, %d trends (%.1fms)",
            len(comparisons),
            len(predictions),
            len(trends),
            elapsed,
        )
        return report

    # -- Internal Helpers ---------------------------------------------------

    def _find_dataset(
        self,
        sector: str,
        geography: str,
        year: Optional[int],
        company_size: str,
    ) -> BenchmarkDataset:
        """Find the matching benchmark dataset.

        Args:
            sector: Sector code.
            geography: Geographic region.
            year: Reporting year (None = latest).
            company_size: Company size filter.

        Returns:
            Matching BenchmarkDataset.

        Raises:
            ValueError: If no matching dataset found.
        """
        if year:
            key = f"{sector}_{geography}_{year}_{company_size}"
            dataset = self.datasets.get(key)
            if dataset:
                return dataset

        # Try to find any matching dataset
        candidates = [
            ds
            for ds in self.datasets.values()
            if ds.sector == sector and ds.geography == geography
        ]
        if company_size != "all":
            sized = [ds for ds in candidates if ds.company_size == company_size]
            if sized:
                candidates = sized

        if not candidates:
            raise ValueError(
                f"No benchmark data for sector={sector}, geography={geography}"
            )

        # Return the latest year
        return max(candidates, key=lambda ds: ds.year)

    def _compute_quartile(
        self, percentile_rank: float, higher_is_better: bool
    ) -> int:
        """Compute quartile from percentile rank.

        Q1 = top quartile (best), Q4 = bottom quartile (worst).

        Args:
            percentile_rank: Percentile rank (0-100).
            higher_is_better: Whether higher values are better.

        Returns:
            Quartile number (1-4).
        """
        if higher_is_better:
            if percentile_rank >= 75:
                return 1
            elif percentile_rank >= 50:
                return 2
            elif percentile_rank >= 25:
                return 3
            else:
                return 4
        else:
            # For lower-is-better, low percentile = good
            if percentile_rank <= 25:
                return 1
            elif percentile_rank <= 50:
                return 2
            elif percentile_rank <= 75:
                return 3
            else:
                return 4

    def _compute_composite_score(self, company_data: Dict[str, float]) -> float:
        """Compute a composite ESG score from company data.

        Uses a weighted average of key ESG metrics normalized to 0-100.

        Args:
            company_data: Company metric values.

        Returns:
            Composite score (0-100).
        """
        scoring_weights: Dict[str, Tuple[float, bool]] = {
            "ghg_intensity": (0.15, False),
            "renewable_energy_pct": (0.10, True),
            "scope1_emissions": (0.10, False),
            "scope2_emissions": (0.08, False),
            "waste_intensity": (0.07, False),
            "water_intensity": (0.07, False),
            "recycling_rate": (0.08, True),
            "board_diversity_pct": (0.08, True),
            "esrs_disclosure_score": (0.12, True),
            "injury_rate": (0.05, False),
            "training_hours_per_employee": (0.05, True),
            "supplier_assessment_pct": (0.05, True),
        }

        total_weight = 0.0
        weighted_score = 0.0

        for metric, (weight, higher_better) in scoring_weights.items():
            value = company_data.get(metric)
            if value is None:
                continue

            # Normalize to 0-100 scale
            if higher_better:
                normalized = min(100.0, max(0.0, value))
            else:
                # Invert: lower is better, cap at reasonable range
                normalized = max(0.0, 100.0 - min(100.0, value))

            weighted_score += normalized * weight
            total_weight += weight

        if total_weight == 0:
            return 50.0

        return round(weighted_score / total_weight, 2)

    def _map_msci_rating(self, composite: float) -> str:
        """Map composite score to MSCI ESG rating."""
        for threshold, rating in _MSCI_SCORING:
            if composite >= threshold:
                return rating
        return "CCC"

    def _map_sustainalytics_rating(self, composite: float) -> str:
        """Map composite score to Sustainalytics risk category."""
        for threshold, score_range, risk_label in _SUSTAINALYTICS_SCORING:
            if composite >= threshold:
                return f"{risk_label} ({score_range})"
        return "Severe Risk (40+)"

    def _map_cdp_rating(self, composite: float) -> str:
        """Map composite score to CDP score."""
        for threshold, rating in _CDP_SCORING:
            if composite >= threshold:
                return rating
        return "D-"

    def _calculate_confidence(
        self, company_data: Dict[str, float], framework: str
    ) -> float:
        """Calculate prediction confidence based on data completeness.

        Args:
            company_data: Company metric values.
            framework: Rating framework.

        Returns:
            Confidence score (0-1).
        """
        key_metrics = [
            "ghg_intensity",
            "scope1_emissions",
            "scope2_emissions",
            "renewable_energy_pct",
            "esrs_disclosure_score",
            "board_diversity_pct",
        ]
        present = sum(1 for m in key_metrics if m in company_data)
        base_confidence = present / len(key_metrics)

        # Discount if very few total metrics
        total_metrics = len(company_data)
        if total_metrics < 3:
            base_confidence *= 0.5
        elif total_metrics < 6:
            base_confidence *= 0.75

        return round(min(0.95, base_confidence), 2)

    def _identify_positive_drivers(
        self, company_data: Dict[str, float]
    ) -> List[str]:
        """Identify positive ESG rating drivers."""
        drivers: List[str] = []

        if company_data.get("renewable_energy_pct", 0) > 50:
            drivers.append("High renewable energy usage")
        if company_data.get("esrs_disclosure_score", 0) > 80:
            drivers.append("Strong ESRS disclosure coverage")
        if company_data.get("board_diversity_pct", 0) > 30:
            drivers.append("Good board diversity")
        if company_data.get("recycling_rate", 0) > 70:
            drivers.append("High recycling rate")
        if company_data.get("ghg_intensity", 100) < 20:
            drivers.append("Low GHG intensity")
        if company_data.get("supplier_assessment_pct", 0) > 75:
            drivers.append("Comprehensive supplier assessment")

        return drivers

    def _identify_negative_drivers(
        self, company_data: Dict[str, float]
    ) -> List[str]:
        """Identify negative ESG rating drivers."""
        drivers: List[str] = []

        if company_data.get("ghg_intensity", 0) > 80:
            drivers.append("High GHG intensity")
        if company_data.get("injury_rate", 0) > 5:
            drivers.append("Elevated injury rate")
        if company_data.get("renewable_energy_pct", 0) < 10:
            drivers.append("Low renewable energy adoption")
        if company_data.get("board_diversity_pct", 0) < 15:
            drivers.append("Low board diversity")
        if company_data.get("esrs_disclosure_score", 0) < 50:
            drivers.append("Insufficient ESRS disclosures")
        if company_data.get("waste_intensity", 0) > 50:
            drivers.append("High waste intensity")

        return drivers

    def _generate_improvement_actions(
        self, negative_drivers: List[str], framework: str
    ) -> List[str]:
        """Generate improvement actions based on negative drivers.

        Args:
            negative_drivers: Identified negative drivers.
            framework: Target rating framework.

        Returns:
            List of actionable improvement recommendations.
        """
        actions: List[str] = []

        driver_action_map: Dict[str, str] = {
            "High GHG intensity": "Develop emission reduction roadmap with science-based targets",
            "Elevated injury rate": "Strengthen occupational health and safety management system",
            "Low renewable energy adoption": "Set renewable energy procurement targets and explore PPAs",
            "Low board diversity": "Implement board diversity policy with measurable targets",
            "Insufficient ESRS disclosures": "Complete ESRS gap analysis and populate missing data points",
            "High waste intensity": "Implement circular economy practices and waste reduction programs",
        }

        for driver in negative_drivers:
            action = driver_action_map.get(driver)
            if action:
                actions.append(action)

        if framework == "CDP" and not any("emission" in a.lower() for a in actions):
            actions.append("Enhance climate risk disclosure per CDP questionnaire requirements")

        return actions

    def _calculate_cagr(
        self, start_value: float, end_value: float, years: int
    ) -> float:
        """Calculate Compound Annual Growth Rate.

        Args:
            start_value: Starting value.
            end_value: Ending value.
            years: Number of years.

        Returns:
            CAGR as a decimal (e.g., 0.05 = 5%).
        """
        if years <= 0 or start_value <= 0:
            return 0.0
        if end_value <= 0:
            return -1.0

        return (end_value / start_value) ** (1.0 / years) - 1.0

    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate standard deviation of values.

        Args:
            values: List of numeric values.

        Returns:
            Standard deviation.
        """
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance)

    def _determine_trend_direction(
        self, values: List[float], metric_name: str
    ) -> str:
        """Determine whether trend is improving, declining, or stable.

        Args:
            values: Time series values (oldest to newest).
            metric_name: Metric name for direction interpretation.

        Returns:
            'improving', 'declining', or 'stable'.
        """
        if len(values) < 2:
            return "stable"

        first_half = values[: len(values) // 2]
        second_half = values[len(values) // 2 :]

        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)

        if avg_first == 0:
            change_pct = 0.0
        else:
            change_pct = (avg_second - avg_first) / abs(avg_first) * 100

        is_lower_better = metric_name in self.config.lower_is_better_metrics

        if abs(change_pct) < 5.0:
            return "stable"
        elif change_pct > 0:
            return "declining" if is_lower_better else "improving"
        else:
            return "improving" if is_lower_better else "declining"

    def _project_next_year(self, values: List[float], cagr: float) -> float:
        """Project next year value using CAGR.

        Args:
            values: Historical values.
            cagr: Compound annual growth rate.

        Returns:
            Projected value for next year.
        """
        if not values:
            return 0.0
        return values[-1] * (1 + cagr)
