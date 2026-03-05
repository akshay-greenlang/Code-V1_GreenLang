"""
CDP Benchmarking Engine -- Peer Comparison and Sector Analysis

This module provides peer benchmarking capabilities including sector-level
comparison (GICS classification), regional benchmarking, score distribution
analysis, category-by-category comparison, A-list rate calculation, custom
peer groups, and historical sector trends.

Key capabilities:
  - Sector benchmarking using GICS classification
  - Regional benchmarking by geography
  - Score distribution analysis (histogram, percentiles)
  - Category-by-category comparison against sector average
  - A-list rate calculation per sector
  - Custom peer group definition and comparison
  - Historical sector trend analysis
  - Anonymous benchmarking (no company names exposed)

Example:
    >>> engine = BenchmarkingEngine(config, scoring_simulator)
    >>> benchmark = engine.benchmark_organization("q-123", sector="20")
    >>> print(f"Sector rank: {benchmark.sector_rank}")
"""

from __future__ import annotations

import logging
import math
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import (
    CDPAppConfig,
    GICS_SECTOR_NAMES,
    SCORING_CATEGORY_WEIGHTS,
    SCORING_LEVEL_BANDS,
    SCORING_LEVEL_THRESHOLDS,
    SECTOR_BENCHMARK_SCORES,
    ScoringBand,
    ScoringLevel,
)
from .models import (
    Benchmark,
    PeerComparison,
    SectorDistribution,
    _new_id,
    _now,
)
from .scoring_simulator import ScoringSimulator

logger = logging.getLogger(__name__)


class BenchmarkingEngine:
    """
    CDP Benchmarking Engine -- compares organizations against peers.

    Provides sector, regional, and custom peer group benchmarking with
    anonymous score distribution analysis and category-level comparisons.

    Attributes:
        config: Application configuration.
        scoring_simulator: Reference to scoring simulator.
        _peer_data: In-memory peer data store.
        _custom_groups: Custom peer group definitions.
        _benchmarks: Benchmark result cache.

    Example:
        >>> engine = BenchmarkingEngine(config, scorer)
        >>> result = engine.benchmark_organization("q-123", sector="20")
    """

    def __init__(
        self,
        config: CDPAppConfig,
        scoring_simulator: ScoringSimulator,
    ) -> None:
        """Initialize the Benchmarking Engine."""
        self.config = config
        self.scoring_simulator = scoring_simulator
        self._peer_data: Dict[str, List[PeerComparison]] = {}  # sector -> peers
        self._custom_groups: Dict[str, List[str]] = {}
        self._benchmarks: Dict[str, Benchmark] = {}
        self._initialize_sector_data()
        logger.info("BenchmarkingEngine initialized with %d sectors", len(GICS_SECTOR_NAMES))

    # ------------------------------------------------------------------
    # Main Benchmarking
    # ------------------------------------------------------------------

    def benchmark_organization(
        self,
        questionnaire_id: str,
        sector: Optional[str] = None,
        region: Optional[str] = None,
        custom_peer_group_id: Optional[str] = None,
    ) -> Benchmark:
        """
        Benchmark an organization against its peers.

        Args:
            questionnaire_id: Questionnaire ID to benchmark.
            sector: GICS sector code for sector comparison.
            region: Geographic region for regional comparison.
            custom_peer_group_id: ID of custom peer group.

        Returns:
            Complete Benchmark result.
        """
        start_time = datetime.utcnow()

        # Calculate organization score
        org_score = self.scoring_simulator.calculate_score(questionnaire_id)
        org_score_pct = org_score.overall_score_pct
        org_level = org_score.overall_level

        # Get sector distribution
        sector_dist = None
        sector_rank = None
        sector_percentile = None

        if sector:
            sector_dist = self._get_sector_distribution(sector)
            if sector_dist and sector_dist.total_respondents > 0:
                sector_rank, sector_percentile = self._calculate_rank(
                    org_score_pct, sector,
                )

        # Regional ranking
        regional_rank = None
        regional_percentile = None
        if region:
            regional_rank, regional_percentile = self._calculate_regional_rank(
                org_score_pct, region,
            )

        # Category comparison
        category_comparison = self._build_category_comparison(
            org_score.category_scores, sector,
        )

        # Get peer data
        peers = self._get_sector_peers(sector) if sector else []

        # Historical trends
        historical = self._get_sector_trend(sector) if sector else []

        elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000

        benchmark = Benchmark(
            questionnaire_id=questionnaire_id,
            org_id=org_score.org_id,
            year=org_score.year,
            org_score=round(org_score_pct, 2),
            org_level=org_level,
            sector_distribution=sector_dist,
            sector_rank=sector_rank,
            sector_percentile=round(sector_percentile, 1) if sector_percentile else None,
            regional_rank=regional_rank,
            regional_percentile=round(regional_percentile, 1) if regional_percentile else None,
            category_comparison=category_comparison,
            peers=peers[:20],  # Limit to 20 anonymized peers
            custom_peer_group_id=custom_peer_group_id,
            historical_trend=historical,
        )

        self._benchmarks[questionnaire_id] = benchmark

        logger.info(
            "Benchmarked %s: score=%.1f%%, sector=%s, rank=%s (%.1f ms)",
            questionnaire_id, org_score_pct, sector, sector_rank, elapsed,
        )
        return benchmark

    # ------------------------------------------------------------------
    # Sector Analysis
    # ------------------------------------------------------------------

    def get_sector_distribution(
        self,
        sector_code: str,
    ) -> Optional[SectorDistribution]:
        """Get the score distribution for a sector."""
        return self._get_sector_distribution(sector_code)

    def get_sector_a_list_rate(self, sector_code: str) -> Dict[str, Any]:
        """
        Get the A-list rate for a sector.

        Returns the percentage and count of organizations achieving A/A-.
        """
        benchmark_data = SECTOR_BENCHMARK_SCORES.get(sector_code)
        if not benchmark_data:
            return {"sector": sector_code, "a_list_pct": 0.0, "a_list_count": 0}

        return {
            "sector_code": sector_code,
            "sector_name": benchmark_data.get("name", ""),
            "a_list_pct": benchmark_data.get("a_list_pct", 0.0),
            "a_list_count": round(benchmark_data.get("a_list_pct", 0) * 0.5),
            "median_score": benchmark_data.get("median", ""),
            "avg_score": benchmark_data.get("avg_score", 0.0),
        }

    def get_all_sector_rankings(self) -> List[Dict[str, Any]]:
        """Get A-list rates and average scores for all sectors."""
        rankings = []
        for code, data in SECTOR_BENCHMARK_SCORES.items():
            rankings.append({
                "sector_code": code,
                "sector_name": data.get("name", ""),
                "avg_score": data.get("avg_score", 0.0),
                "a_list_pct": data.get("a_list_pct", 0.0),
                "median": data.get("median", ""),
            })
        rankings.sort(key=lambda x: x["avg_score"], reverse=True)
        return rankings

    # ------------------------------------------------------------------
    # Custom Peer Groups
    # ------------------------------------------------------------------

    def create_custom_peer_group(
        self,
        group_name: str,
        peer_org_ids: List[str],
    ) -> str:
        """
        Create a custom peer group for comparison.

        Args:
            group_name: Display name for the peer group.
            peer_org_ids: List of peer organization IDs.

        Returns:
            Group ID.
        """
        group_id = _new_id()
        self._custom_groups[group_id] = peer_org_ids
        logger.info(
            "Created custom peer group '%s' with %d peers",
            group_name, len(peer_org_ids),
        )
        return group_id

    def get_custom_group_comparison(
        self,
        questionnaire_id: str,
        group_id: str,
    ) -> Dict[str, Any]:
        """Compare against a custom peer group."""
        peer_ids = self._custom_groups.get(group_id, [])
        if not peer_ids:
            return {"error": "Custom peer group not found or empty"}

        org_score = self.scoring_simulator.calculate_score(questionnaire_id)

        return {
            "org_score": round(org_score.overall_score_pct, 2),
            "org_level": org_score.overall_level.value,
            "peer_count": len(peer_ids),
            "group_id": group_id,
        }

    # ------------------------------------------------------------------
    # Category-Level Comparison
    # ------------------------------------------------------------------

    def compare_categories(
        self,
        questionnaire_id: str,
        sector_code: str,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare all 17 categories against sector averages.

        Returns a dict keyed by category ID with org and sector scores.
        """
        org_score = self.scoring_simulator.calculate_score(questionnaire_id)
        return self._build_category_comparison(org_score.category_scores, sector_code)

    # ------------------------------------------------------------------
    # Historical Trends
    # ------------------------------------------------------------------

    def get_sector_trend(
        self,
        sector_code: str,
        years: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get historical average score trend for a sector.

        Args:
            sector_code: GICS sector code.
            years: Number of years of history.

        Returns:
            List of year -> avg_score data points.
        """
        return self._get_sector_trend(sector_code, years)

    # ------------------------------------------------------------------
    # Peer Data Management
    # ------------------------------------------------------------------

    def load_peer_data(
        self,
        sector_code: str,
        peers: List[PeerComparison],
    ) -> None:
        """Load peer comparison data for a sector."""
        self._peer_data[sector_code] = peers
        logger.info(
            "Loaded %d peer data points for sector %s",
            len(peers), sector_code,
        )

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    def _initialize_sector_data(self) -> None:
        """Initialize synthetic sector peer data for benchmarking."""
        for sector_code, data in SECTOR_BENCHMARK_SCORES.items():
            avg = data.get("avg_score", 40.0)
            a_pct = data.get("a_list_pct", 5.0)

            # Generate synthetic peer distribution
            peers = self._generate_synthetic_peers(sector_code, avg, a_pct)
            self._peer_data[sector_code] = peers

    def _generate_synthetic_peers(
        self,
        sector_code: str,
        avg_score: float,
        a_list_pct: float,
        count: int = 50,
    ) -> List[PeerComparison]:
        """Generate synthetic peer data using normal distribution."""
        peers = []
        std_dev = 18.0  # Typical CDP score standard deviation
        rng = random.Random(hash(sector_code))  # Deterministic per sector

        for i in range(count):
            score = max(0.0, min(100.0, rng.gauss(avg_score, std_dev)))
            level = self._score_to_level(score)

            # Generate category scores
            cat_scores = {}
            for cat_id in SCORING_CATEGORY_WEIGHTS:
                cat_score = max(0.0, min(100.0, score + rng.gauss(0, 12)))
                cat_scores[cat_id] = round(cat_score, 1)

            peers.append(PeerComparison(
                sector=sector_code,
                overall_score=round(score, 2),
                level=level,
                category_scores=cat_scores,
                year=2026,
            ))

        peers.sort(key=lambda p: p.overall_score, reverse=True)
        return peers

    def _get_sector_distribution(
        self,
        sector_code: str,
    ) -> Optional[SectorDistribution]:
        """Build sector distribution statistics."""
        peers = self._peer_data.get(sector_code, [])
        if not peers:
            return None

        scores = sorted([p.overall_score for p in peers])
        n = len(scores)

        if n == 0:
            return None

        a_list = sum(1 for s in scores if s >= 70.0)

        # Level distribution
        level_dist: Dict[str, int] = {}
        for s in scores:
            level = self._score_to_level(s)
            level_dist[level.value] = level_dist.get(level.value, 0) + 1

        return SectorDistribution(
            sector_code=sector_code,
            sector_name=GICS_SECTOR_NAMES.get(sector_code, ""),
            total_respondents=n,
            mean_score=round(sum(scores) / n, 2),
            median_score=round(scores[n // 2], 2),
            p25_score=round(scores[n // 4], 2),
            p75_score=round(scores[3 * n // 4], 2),
            min_score=round(scores[0], 2),
            max_score=round(scores[-1], 2),
            a_list_count=a_list,
            a_list_pct=round(a_list / n * 100, 2) if n > 0 else 0.0,
            level_distribution=level_dist,
        )

    def _calculate_rank(
        self,
        org_score: float,
        sector_code: str,
    ) -> tuple:
        """Calculate organization's rank within its sector."""
        peers = self._peer_data.get(sector_code, [])
        if not peers:
            return (None, None)

        scores = sorted([p.overall_score for p in peers], reverse=True)
        # Find rank position
        rank = 1
        for s in scores:
            if org_score >= s:
                break
            rank += 1

        percentile = (1 - (rank - 1) / max(len(scores), 1)) * 100
        return (rank, percentile)

    def _calculate_regional_rank(
        self,
        org_score: float,
        region: str,
    ) -> tuple:
        """Calculate regional rank (uses all sector data for the region)."""
        all_scores = []
        for peers in self._peer_data.values():
            for p in peers:
                if p.region == region or region == "global":
                    all_scores.append(p.overall_score)

        if not all_scores:
            # Fall back to all data
            for peers in self._peer_data.values():
                all_scores.extend(p.overall_score for p in peers)

        if not all_scores:
            return (None, None)

        all_scores.sort(reverse=True)
        rank = 1
        for s in all_scores:
            if org_score >= s:
                break
            rank += 1

        percentile = (1 - (rank - 1) / max(len(all_scores), 1)) * 100
        return (rank, percentile)

    def _build_category_comparison(
        self,
        org_category_scores: list,
        sector_code: Optional[str],
    ) -> Dict[str, Dict[str, float]]:
        """Build category-by-category comparison against sector average."""
        comparison: Dict[str, Dict[str, float]] = {}

        org_cat_map = {cs.category_id: cs.score_pct for cs in org_category_scores}

        for cat_id, cat_info in SCORING_CATEGORY_WEIGHTS.items():
            org_val = org_cat_map.get(cat_id, 0.0)

            # Get sector average for this category
            sector_avg = 0.0
            if sector_code:
                peers = self._peer_data.get(sector_code, [])
                cat_values = [
                    p.category_scores.get(cat_id, 0.0) for p in peers
                ]
                if cat_values:
                    sector_avg = sum(cat_values) / len(cat_values)

            comparison[cat_id] = {
                "name": cat_info.get("name", ""),
                "org_score": round(org_val, 2),
                "sector_avg": round(sector_avg, 2),
                "delta": round(org_val - sector_avg, 2),
                "above_average": org_val > sector_avg,
            }

        return comparison

    def _get_sector_peers(
        self,
        sector_code: Optional[str],
    ) -> List[PeerComparison]:
        """Get anonymized peer data for a sector."""
        if not sector_code:
            return []
        return self._peer_data.get(sector_code, [])

    def _get_sector_trend(
        self,
        sector_code: Optional[str],
        years: int = 5,
    ) -> List[Dict[str, Any]]:
        """Generate historical sector trend data."""
        if not sector_code:
            return []

        base_data = SECTOR_BENCHMARK_SCORES.get(sector_code)
        if not base_data:
            return []

        base_score = base_data.get("avg_score", 40.0)
        trend = []

        for i in range(years, 0, -1):
            year = 2026 - i
            # Simulate gradual improvement (2-3% per year on average)
            year_score = max(10.0, base_score - (i * 2.5))
            trend.append({
                "year": year,
                "avg_score": round(year_score, 1),
                "a_list_pct": max(0.5, base_data.get("a_list_pct", 5.0) - (i * 0.8)),
                "respondent_count": max(20, 50 - i * 3),
            })

        # Add current year
        trend.append({
            "year": 2026,
            "avg_score": base_score,
            "a_list_pct": base_data.get("a_list_pct", 5.0),
            "respondent_count": 50,
        })

        return trend

    def _score_to_level(self, score: float) -> ScoringLevel:
        """Convert a score to a scoring level."""
        for level, (min_s, max_s) in SCORING_LEVEL_THRESHOLDS.items():
            if min_s <= score <= max_s:
                return level
        if score >= 80.0:
            return ScoringLevel.A
        return ScoringLevel.D_MINUS
