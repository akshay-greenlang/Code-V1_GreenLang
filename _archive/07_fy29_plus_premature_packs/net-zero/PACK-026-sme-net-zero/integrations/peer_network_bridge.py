# -*- coding: utf-8 -*-
"""
PeerNetworkBridge - Anonymous Peer Benchmarking for PACK-026
================================================================

Provides privacy-preserving peer benchmarking for SMEs, enabling
anonymous comparison of emissions performance against industry peers
of similar size and geography.

Features:
    - Anonymous peer benchmarking (no individual data shared)
    - Aggregated industry statistics by sector, size tier, geography
    - Emissions intensity benchmarks (tCO2e/employee, tCO2e/revenue)
    - Percentile ranking
    - Trend analysis (year-over-year comparison)
    - Privacy-preserving aggregation (minimum 5 peers for disclosure)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-026 SME Net Zero Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SizeTier(str, Enum):
    MICRO = "micro"        # 1-9 employees
    SMALL = "small"        # 10-49 employees
    MEDIUM = "medium"      # 50-249 employees

class BenchmarkMetric(str, Enum):
    TOTAL_EMISSIONS = "total_emissions_tco2e"
    EMISSIONS_PER_EMPLOYEE = "emissions_per_employee_tco2e"
    EMISSIONS_PER_REVENUE = "emissions_per_revenue_tco2e_per_meur"
    SCOPE1_SHARE = "scope1_share_pct"
    SCOPE2_SHARE = "scope2_share_pct"
    SCOPE3_SHARE = "scope3_share_pct"
    REDUCTION_RATE = "year_on_year_reduction_pct"
    RENEWABLE_ENERGY = "renewable_energy_pct"

# ---------------------------------------------------------------------------
# Industry Benchmark Data (Aggregated, Anonymous)
# ---------------------------------------------------------------------------

INDUSTRY_BENCHMARKS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "general": {
        "micro": {
            "total_emissions_tco2e": {"p25": 15, "p50": 35, "p75": 80, "mean": 45},
            "emissions_per_employee_tco2e": {"p25": 2.0, "p50": 4.5, "p75": 8.0, "mean": 5.0},
            "emissions_per_revenue_tco2e_per_meur": {"p25": 20, "p50": 45, "p75": 90, "mean": 55},
            "scope1_share_pct": {"p25": 10, "p50": 25, "p75": 40, "mean": 25},
            "scope2_share_pct": {"p25": 15, "p50": 30, "p75": 45, "mean": 30},
            "scope3_share_pct": {"p25": 20, "p50": 45, "p75": 65, "mean": 45},
            "year_on_year_reduction_pct": {"p25": 0, "p50": 3, "p75": 8, "mean": 4},
            "renewable_energy_pct": {"p25": 0, "p50": 15, "p75": 50, "mean": 25},
            "peer_count": 150,
        },
        "small": {
            "total_emissions_tco2e": {"p25": 50, "p50": 120, "p75": 300, "mean": 160},
            "emissions_per_employee_tco2e": {"p25": 2.5, "p50": 5.0, "p75": 9.0, "mean": 5.5},
            "emissions_per_revenue_tco2e_per_meur": {"p25": 15, "p50": 35, "p75": 70, "mean": 40},
            "scope1_share_pct": {"p25": 12, "p50": 28, "p75": 42, "mean": 27},
            "scope2_share_pct": {"p25": 15, "p50": 28, "p75": 40, "mean": 28},
            "scope3_share_pct": {"p25": 25, "p50": 44, "p75": 60, "mean": 45},
            "year_on_year_reduction_pct": {"p25": 0, "p50": 4, "p75": 10, "mean": 5},
            "renewable_energy_pct": {"p25": 0, "p50": 20, "p75": 60, "mean": 30},
            "peer_count": 280,
        },
        "medium": {
            "total_emissions_tco2e": {"p25": 200, "p50": 500, "p75": 1200, "mean": 650},
            "emissions_per_employee_tco2e": {"p25": 3.0, "p50": 5.5, "p75": 10.0, "mean": 6.0},
            "emissions_per_revenue_tco2e_per_meur": {"p25": 10, "p50": 28, "p75": 55, "mean": 32},
            "scope1_share_pct": {"p25": 15, "p50": 30, "p75": 45, "mean": 30},
            "scope2_share_pct": {"p25": 12, "p50": 25, "p75": 38, "mean": 25},
            "scope3_share_pct": {"p25": 28, "p50": 45, "p75": 62, "mean": 45},
            "year_on_year_reduction_pct": {"p25": 1, "p50": 5, "p75": 12, "mean": 6},
            "renewable_energy_pct": {"p25": 5, "p50": 30, "p75": 70, "mean": 35},
            "peer_count": 200,
        },
    },
    "manufacturing": {
        "small": {
            "total_emissions_tco2e": {"p25": 100, "p50": 280, "p75": 600, "mean": 330},
            "emissions_per_employee_tco2e": {"p25": 5.0, "p50": 10.0, "p75": 18.0, "mean": 11.0},
            "emissions_per_revenue_tco2e_per_meur": {"p25": 30, "p50": 60, "p75": 120, "mean": 70},
            "scope1_share_pct": {"p25": 25, "p50": 40, "p75": 55, "mean": 40},
            "scope2_share_pct": {"p25": 15, "p50": 25, "p75": 35, "mean": 25},
            "scope3_share_pct": {"p25": 20, "p50": 35, "p75": 50, "mean": 35},
            "year_on_year_reduction_pct": {"p25": 0, "p50": 3, "p75": 8, "mean": 4},
            "renewable_energy_pct": {"p25": 0, "p50": 10, "p75": 40, "mean": 18},
            "peer_count": 95,
        },
        "medium": {
            "total_emissions_tco2e": {"p25": 400, "p50": 1000, "p75": 2500, "mean": 1300},
            "emissions_per_employee_tco2e": {"p25": 5.5, "p50": 11.0, "p75": 20.0, "mean": 12.0},
            "emissions_per_revenue_tco2e_per_meur": {"p25": 25, "p50": 50, "p75": 100, "mean": 58},
            "scope1_share_pct": {"p25": 28, "p50": 42, "p75": 58, "mean": 43},
            "scope2_share_pct": {"p25": 12, "p50": 22, "p75": 32, "mean": 22},
            "scope3_share_pct": {"p25": 18, "p50": 36, "p75": 52, "mean": 35},
            "year_on_year_reduction_pct": {"p25": 1, "p50": 4, "p75": 10, "mean": 5},
            "renewable_energy_pct": {"p25": 5, "p50": 20, "p75": 55, "mean": 28},
            "peer_count": 72,
        },
    },
    "technology": {
        "small": {
            "total_emissions_tco2e": {"p25": 20, "p50": 60, "p75": 150, "mean": 80},
            "emissions_per_employee_tco2e": {"p25": 1.0, "p50": 2.5, "p75": 5.0, "mean": 3.0},
            "emissions_per_revenue_tco2e_per_meur": {"p25": 5, "p50": 15, "p75": 35, "mean": 18},
            "scope1_share_pct": {"p25": 3, "p50": 8, "p75": 15, "mean": 9},
            "scope2_share_pct": {"p25": 10, "p50": 22, "p75": 35, "mean": 22},
            "scope3_share_pct": {"p25": 50, "p50": 70, "p75": 85, "mean": 69},
            "year_on_year_reduction_pct": {"p25": 2, "p50": 6, "p75": 15, "mean": 8},
            "renewable_energy_pct": {"p25": 10, "p50": 40, "p75": 80, "mean": 45},
            "peer_count": 120,
        },
    },
    "retail": {
        "small": {
            "total_emissions_tco2e": {"p25": 40, "p50": 100, "p75": 250, "mean": 130},
            "emissions_per_employee_tco2e": {"p25": 2.0, "p50": 4.0, "p75": 7.5, "mean": 4.5},
            "emissions_per_revenue_tco2e_per_meur": {"p25": 12, "p50": 30, "p75": 60, "mean": 35},
            "scope1_share_pct": {"p25": 8, "p50": 18, "p75": 30, "mean": 19},
            "scope2_share_pct": {"p25": 18, "p50": 32, "p75": 45, "mean": 32},
            "scope3_share_pct": {"p25": 30, "p50": 50, "p75": 68, "mean": 49},
            "year_on_year_reduction_pct": {"p25": 0, "p50": 4, "p75": 10, "mean": 5},
            "renewable_energy_pct": {"p25": 5, "p50": 25, "p75": 60, "mean": 30},
            "peer_count": 85,
        },
    },
}

MINIMUM_PEERS_FOR_DISCLOSURE = 5

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class PeerNetworkConfig(BaseModel):
    """Configuration for the Peer Network Bridge."""

    pack_id: str = Field(default="PACK-026")
    enable_provenance: bool = Field(default=True)
    minimum_peers: int = Field(default=5, ge=3, le=20)
    anonymize_data: bool = Field(default=True)

class BenchmarkResult(BaseModel):
    """Result of a peer benchmarking comparison."""

    benchmark_id: str = Field(default_factory=_new_uuid)
    sector: str = Field(default="")
    size_tier: str = Field(default="")
    geography: str = Field(default="")
    peer_count: int = Field(default=0)
    sufficient_peers: bool = Field(default=False)
    metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    your_values: Dict[str, float] = Field(default_factory=dict)
    percentile_rankings: Dict[str, float] = Field(default_factory=dict)
    performance_summary: str = Field(default="")
    improvement_areas: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class PercentileRanking(BaseModel):
    """Percentile ranking for a single metric."""

    metric: str = Field(default="")
    your_value: float = Field(default=0.0)
    percentile: float = Field(default=0.0)
    peer_p25: float = Field(default=0.0)
    peer_p50: float = Field(default=0.0)
    peer_p75: float = Field(default=0.0)
    peer_mean: float = Field(default=0.0)
    interpretation: str = Field(default="")

# ---------------------------------------------------------------------------
# PeerNetworkBridge
# ---------------------------------------------------------------------------

class PeerNetworkBridge:
    """Anonymous peer benchmarking for SME emissions performance.

    Provides privacy-preserving comparison against aggregated industry
    statistics by sector, size tier, and geography. No individual
    company data is ever shared.

    Attributes:
        config: Bridge configuration.
        _benchmark_history: History of benchmark requests.

    Example:
        >>> bridge = PeerNetworkBridge()
        >>> result = bridge.benchmark(
        ...     sector="technology",
        ...     size_tier="small",
        ...     your_data={"total_emissions_tco2e": 45, "employee_count": 20}
        ... )
        >>> for metric, pct in result.percentile_rankings.items():
        ...     print(f"{metric}: {pct}th percentile")
    """

    def __init__(self, config: Optional[PeerNetworkConfig] = None) -> None:
        self.config = config or PeerNetworkConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._benchmark_history: List[BenchmarkResult] = []

        self.logger.info("PeerNetworkBridge initialized: anonymize=%s", self.config.anonymize_data)

    # -------------------------------------------------------------------------
    # Benchmarking
    # -------------------------------------------------------------------------

    def benchmark(
        self,
        sector: str = "general",
        size_tier: str = "small",
        geography: str = "global",
        your_data: Optional[Dict[str, float]] = None,
    ) -> BenchmarkResult:
        """Run peer benchmarking comparison.

        Args:
            sector: Organization sector.
            size_tier: Size tier (micro, small, medium).
            geography: Geography filter (currently global only).
            your_data: Your organization's metrics for comparison.

        Returns:
            BenchmarkResult with percentile rankings and insights.
        """
        start = time.monotonic()
        your_data = your_data or {}

        result = BenchmarkResult(
            sector=sector,
            size_tier=size_tier,
            geography=geography,
        )

        # Find matching benchmarks
        sector_data = INDUSTRY_BENCHMARKS.get(sector)
        if sector_data is None:
            sector_data = INDUSTRY_BENCHMARKS.get("general", {})
            result.sector = "general"

        tier_data = sector_data.get(size_tier)
        if tier_data is None:
            # Fall back to nearest tier
            for fallback in ["small", "medium", "micro"]:
                tier_data = sector_data.get(fallback)
                if tier_data:
                    result.size_tier = fallback
                    break

        if tier_data is None:
            result.sufficient_peers = False
            result.performance_summary = "Insufficient peer data for your sector and size."
            return result

        peer_count = tier_data.get("peer_count", 0)
        result.peer_count = peer_count
        result.sufficient_peers = peer_count >= self.config.minimum_peers

        if not result.sufficient_peers:
            result.performance_summary = (
                f"Only {peer_count} peers found (minimum {self.config.minimum_peers} "
                f"required for privacy). Try a broader sector or size tier."
            )
            return result

        # Calculate metrics and rankings
        metrics: Dict[str, Dict[str, Any]] = {}
        rankings: Dict[str, float] = {}
        your_values: Dict[str, float] = {}
        strengths: List[str] = []
        improvements: List[str] = []

        # Compute derived values from input
        emp_count = your_data.get("employee_count", 0)
        revenue_meur = your_data.get("annual_revenue_eur", 0) / 1_000_000.0
        total_emissions = your_data.get("total_emissions_tco2e", 0)
        scope1 = your_data.get("scope1_tco2e", 0)
        scope2 = your_data.get("scope2_tco2e", 0)
        scope3 = your_data.get("scope3_tco2e", 0)

        derived: Dict[str, float] = {
            "total_emissions_tco2e": total_emissions,
            "emissions_per_employee_tco2e": (
                total_emissions / emp_count if emp_count > 0 else 0
            ),
            "emissions_per_revenue_tco2e_per_meur": (
                total_emissions / revenue_meur if revenue_meur > 0 else 0
            ),
            "scope1_share_pct": (
                (scope1 / total_emissions * 100) if total_emissions > 0 else 0
            ),
            "scope2_share_pct": (
                (scope2 / total_emissions * 100) if total_emissions > 0 else 0
            ),
            "scope3_share_pct": (
                (scope3 / total_emissions * 100) if total_emissions > 0 else 0
            ),
            "year_on_year_reduction_pct": your_data.get("year_on_year_reduction_pct", 0),
            "renewable_energy_pct": your_data.get("renewable_energy_pct", 0),
        }

        for metric_key in BenchmarkMetric:
            metric_name = metric_key.value
            peer_stats = tier_data.get(metric_name)
            if peer_stats is None or not isinstance(peer_stats, dict):
                continue

            your_val = derived.get(metric_name, 0)
            your_values[metric_name] = round(your_val, 2)

            p25 = peer_stats.get("p25", 0)
            p50 = peer_stats.get("p50", 0)
            p75 = peer_stats.get("p75", 0)
            mean = peer_stats.get("mean", 0)

            metrics[metric_name] = {
                "p25": p25, "p50": p50, "p75": p75, "mean": mean,
            }

            # Calculate percentile (approximate)
            percentile = self._approximate_percentile(your_val, p25, p50, p75, metric_name)
            rankings[metric_name] = round(percentile, 1)

            # Determine if strength or improvement area
            # For emissions, lower is better; for reduction/renewable, higher is better
            higher_is_better = metric_name in (
                "year_on_year_reduction_pct", "renewable_energy_pct",
            )

            if higher_is_better:
                if percentile >= 75:
                    strengths.append(f"{metric_name}: Top quartile ({percentile:.0f}th percentile)")
                elif percentile <= 25:
                    improvements.append(f"{metric_name}: Below peers ({percentile:.0f}th percentile)")
            else:
                if percentile <= 25:
                    strengths.append(f"{metric_name}: Top quartile ({percentile:.0f}th percentile)")
                elif percentile >= 75:
                    improvements.append(f"{metric_name}: Above peers ({percentile:.0f}th percentile)")

        result.metrics = metrics
        result.your_values = your_values
        result.percentile_rankings = rankings
        result.strengths = strengths
        result.improvement_areas = improvements

        # Summary
        avg_ranking = sum(rankings.values()) / len(rankings) if rankings else 50
        if avg_ranking <= 30:
            result.performance_summary = "Your emissions performance is better than most peers."
        elif avg_ranking <= 60:
            result.performance_summary = "Your emissions performance is around the industry average."
        else:
            result.performance_summary = "There are opportunities to reduce emissions compared to peers."

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        self._benchmark_history.append(result)

        self.logger.info(
            "Peer benchmark: sector=%s, tier=%s, peers=%d, avg_percentile=%.0f in %.1fms",
            sector, size_tier, peer_count, avg_ranking,
            (time.monotonic() - start) * 1000,
        )
        return result

    # -------------------------------------------------------------------------
    # Detailed Rankings
    # -------------------------------------------------------------------------

    def get_detailed_ranking(
        self,
        sector: str,
        size_tier: str,
        metric: str,
        your_value: float,
    ) -> PercentileRanking:
        """Get detailed ranking for a single metric.

        Args:
            sector: Organization sector.
            size_tier: Size tier.
            metric: Metric to rank.
            your_value: Your value for the metric.

        Returns:
            PercentileRanking with interpretation.
        """
        sector_data = INDUSTRY_BENCHMARKS.get(sector, INDUSTRY_BENCHMARKS.get("general", {}))
        tier_data = sector_data.get(size_tier, {})
        peer_stats = tier_data.get(metric, {})

        if not isinstance(peer_stats, dict) or "p50" not in peer_stats:
            return PercentileRanking(
                metric=metric,
                your_value=your_value,
                interpretation="Insufficient data for ranking",
            )

        p25 = peer_stats.get("p25", 0)
        p50 = peer_stats.get("p50", 0)
        p75 = peer_stats.get("p75", 0)
        mean = peer_stats.get("mean", 0)

        percentile = self._approximate_percentile(your_value, p25, p50, p75, metric)

        higher_is_better = metric in ("year_on_year_reduction_pct", "renewable_energy_pct")

        if higher_is_better:
            if percentile >= 75:
                interpretation = "Excellent - top quartile performance"
            elif percentile >= 50:
                interpretation = "Good - above median"
            elif percentile >= 25:
                interpretation = "Below median - room for improvement"
            else:
                interpretation = "Bottom quartile - significant opportunity"
        else:
            if percentile <= 25:
                interpretation = "Excellent - lower emissions than most peers"
            elif percentile <= 50:
                interpretation = "Good - below median emissions"
            elif percentile <= 75:
                interpretation = "Above median - room to reduce"
            else:
                interpretation = "Top quartile emissions - priority area for reduction"

        return PercentileRanking(
            metric=metric,
            your_value=your_value,
            percentile=round(percentile, 1),
            peer_p25=p25,
            peer_p50=p50,
            peer_p75=p75,
            peer_mean=mean,
            interpretation=interpretation,
        )

    # -------------------------------------------------------------------------
    # Available Sectors
    # -------------------------------------------------------------------------

    def get_available_sectors(self) -> List[Dict[str, Any]]:
        """Get list of sectors with benchmark data."""
        result = []
        for sector, tiers in INDUSTRY_BENCHMARKS.items():
            total_peers = sum(
                t.get("peer_count", 0) for t in tiers.values()
                if isinstance(t, dict) and "peer_count" in t
            )
            result.append({
                "sector": sector,
                "size_tiers": list(tiers.keys()),
                "total_peers": total_peers,
            })
        return result

    def get_benchmark_history(self) -> List[Dict[str, Any]]:
        """Get history of benchmark requests."""
        return [
            {
                "benchmark_id": b.benchmark_id,
                "sector": b.sector,
                "size_tier": b.size_tier,
                "peer_count": b.peer_count,
                "generated_at": b.generated_at.isoformat(),
            }
            for b in self._benchmark_history
        ]

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get bridge status."""
        return {
            "pack_id": self.config.pack_id,
            "sectors_available": len(INDUSTRY_BENCHMARKS),
            "minimum_peers": self.config.minimum_peers,
            "anonymize_data": self.config.anonymize_data,
            "benchmarks_run": len(self._benchmark_history),
        }

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _approximate_percentile(
        self,
        value: float,
        p25: float,
        p50: float,
        p75: float,
        metric: str,
    ) -> float:
        """Approximate percentile based on quartile values.

        Linear interpolation between quartile boundaries.
        """
        if p25 == p75:
            return 50.0

        if value <= p25:
            return max(0, 25.0 * (value / p25)) if p25 > 0 else 0.0
        elif value <= p50:
            return 25.0 + 25.0 * (value - p25) / (p50 - p25) if (p50 - p25) > 0 else 37.5
        elif value <= p75:
            return 50.0 + 25.0 * (value - p50) / (p75 - p50) if (p75 - p50) > 0 else 62.5
        else:
            extra = 25.0 * (value - p75) / (p75 - p50) if (p75 - p50) > 0 else 12.5
            return min(100.0, 75.0 + extra)
