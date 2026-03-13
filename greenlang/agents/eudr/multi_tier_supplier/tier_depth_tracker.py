# -*- coding: utf-8 -*-
"""
Tier Depth Tracker - AGENT-EUDR-008 Multi-Tier Supplier Tracker

Engine 3 of 8: Tracks and scores supply chain tier depth for EUDR
compliance. Calculates visibility scores (percentage of known suppliers
at each tier), coverage scores (volume-weighted traceability), detects
tier gaps, provides commodity-specific typical chain depth references,
tracks depth trends over time, benchmarks against industry averages,
and generates hierarchical tier map data.

EUDR Regulatory Context:
    Article 4 requires due diligence on the FULL supply chain, not just
    Tier 1. This engine quantifies how deep an operator's visibility
    extends and identifies where it drops off, enabling targeted
    remediation to meet Article 9 traceability requirements.

Typical Supply Chain Depths (PRD Appendix A):
    - Cocoa:  6-8 tiers (Farmer -> Coop -> Aggregator -> Processor -> ...)
    - Coffee: 5-7 tiers (Farmer -> Coop -> Mill -> Exporter -> ...)
    - Palm:   5-7 tiers (Smallholder -> Mill -> Refinery -> ...)
    - Soya:   4-6 tiers (Farm -> Silo -> Crusher -> ...)
    - Rubber: 5-7 tiers (Smallholder -> Dealer -> Factory -> ...)
    - Cattle: 3-5 tiers (Ranch -> Feedlot -> Slaughterhouse -> ...)
    - Wood:   4-6 tiers (Forest -> Sawmill -> Processor -> ...)

Zero-Hallucination Principle:
    All scoring uses deterministic arithmetic formulas. Industry
    benchmarks are sourced from static reference data, not LLM
    predictions.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Engine version string.
ENGINE_VERSION: str = "1.0.0"

#: Prometheus metric prefix.
METRIC_PREFIX: str = "gl_eudr_mst_"

#: Default batch size for batch operations.
DEFAULT_BATCH_SIZE: int = 1000

#: Maximum supported tier depth.
MAX_TIER_DEPTH: int = 50


# ---------------------------------------------------------------------------
# Reference Data: Typical Supply Chain Depths (PRD Appendix A)
# ---------------------------------------------------------------------------

#: Typical tier depth ranges per commodity (min, typical, max).
COMMODITY_TIER_DEPTHS: Dict[str, Dict[str, Any]] = {
    "cocoa": {
        "min_tiers": 6,
        "typical_tiers": 7,
        "max_tiers": 8,
        "chain_structure": [
            "Farmer/Smallholder",
            "Cooperative",
            "Aggregator",
            "Processor",
            "Trader",
            "Refiner",
            "Importer",
        ],
        "industry_avg_visibility": 3.2,
        "best_in_class_visibility": 6.5,
    },
    "coffee": {
        "min_tiers": 5,
        "typical_tiers": 6,
        "max_tiers": 7,
        "chain_structure": [
            "Farmer/Smallholder",
            "Cooperative",
            "Mill/Wet Processing",
            "Exporter",
            "Trader",
            "Roaster",
            "Importer",
        ],
        "industry_avg_visibility": 3.5,
        "best_in_class_visibility": 6.0,
    },
    "palm_oil": {
        "min_tiers": 5,
        "typical_tiers": 6,
        "max_tiers": 7,
        "chain_structure": [
            "Smallholder",
            "Mill",
            "Refinery",
            "Trader",
            "Processor",
            "Importer",
        ],
        "industry_avg_visibility": 3.0,
        "best_in_class_visibility": 5.5,
    },
    "soya": {
        "min_tiers": 4,
        "typical_tiers": 5,
        "max_tiers": 6,
        "chain_structure": [
            "Farm",
            "Silo/Storage",
            "Crusher",
            "Trader",
            "Processor",
            "Importer",
        ],
        "industry_avg_visibility": 2.8,
        "best_in_class_visibility": 5.0,
    },
    "rubber": {
        "min_tiers": 5,
        "typical_tiers": 6,
        "max_tiers": 7,
        "chain_structure": [
            "Smallholder",
            "Dealer",
            "Factory",
            "Trader",
            "Processor",
            "Importer",
        ],
        "industry_avg_visibility": 2.5,
        "best_in_class_visibility": 5.0,
    },
    "cattle": {
        "min_tiers": 3,
        "typical_tiers": 4,
        "max_tiers": 5,
        "chain_structure": [
            "Ranch",
            "Feedlot",
            "Slaughterhouse",
            "Trader",
            "Importer",
        ],
        "industry_avg_visibility": 2.5,
        "best_in_class_visibility": 4.0,
    },
    "wood": {
        "min_tiers": 4,
        "typical_tiers": 5,
        "max_tiers": 6,
        "chain_structure": [
            "Forest/Concession",
            "Sawmill",
            "Processor",
            "Trader",
            "Importer",
        ],
        "industry_avg_visibility": 3.0,
        "best_in_class_visibility": 5.0,
    },
}

#: Industry benchmark categories for depth assessment.
BENCHMARK_CATEGORIES: Dict[str, Dict[str, float]] = {
    "leading": {"min_ratio": 0.90, "label": "Industry Leading"},
    "above_average": {
        "min_ratio": 0.75,
        "label": "Above Average",
    },
    "average": {"min_ratio": 0.50, "label": "Industry Average"},
    "below_average": {
        "min_ratio": 0.25,
        "label": "Below Average",
    },
    "lagging": {"min_ratio": 0.0, "label": "Industry Lagging"},
}


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class VisibilityLevel(str, Enum):
    """Classification of tier visibility level."""

    FULL = "full"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    NONE = "none"


class GapSeverity(str, Enum):
    """Severity of a tier gap for compliance."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class TrendDirection(str, Enum):
    """Direction of a depth/visibility trend."""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"


class EUDRCommodity(str, Enum):
    """EUDR regulated commodities (7 commodities per Article 1)."""

    COCOA = "cocoa"
    COFFEE = "coffee"
    PALM_OIL = "palm_oil"
    SOYA = "soya"
    RUBBER = "rubber"
    CATTLE = "cattle"
    WOOD = "wood"


# ---------------------------------------------------------------------------
# Data Classes (local, independent of models.py)
# ---------------------------------------------------------------------------


@dataclass
class SupplierNode:
    """A supplier node in the tier hierarchy.

    Attributes:
        supplier_id: Unique supplier identifier.
        name: Supplier name.
        tier_level: Tier level relative to the operator.
        country_code: ISO 3166-1 alpha-2 country code.
        commodity: EUDR commodity.
        volume_tonnes: Annual volume in tonnes.
        has_gps: Whether supplier has GPS coordinates.
        has_certification: Whether supplier has active certifications.
        has_dds: Whether supplier has DDS references.
        upstream_supplier_ids: List of upstream supplier IDs.
        downstream_buyer_ids: List of downstream buyer IDs.
        metadata: Additional key-value metadata.
    """

    supplier_id: str = ""
    name: str = ""
    tier_level: int = 0
    country_code: str = ""
    commodity: str = ""
    volume_tonnes: float = 0.0
    has_gps: bool = False
    has_certification: bool = False
    has_dds: bool = False
    upstream_supplier_ids: List[str] = field(default_factory=list)
    downstream_buyer_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate ID if not provided."""
        if not self.supplier_id:
            self.supplier_id = str(uuid.uuid4())


@dataclass
class SupplierChain:
    """A supply chain structure with suppliers at various tiers.

    Attributes:
        chain_id: Unique chain identifier.
        root_supplier_id: ID of the Tier 1 root supplier.
        commodity: Primary commodity for this chain.
        nodes: Dictionary of supplier nodes keyed by supplier ID.
        total_volume_tonnes: Total volume through this chain.
        metadata: Additional key-value metadata.
    """

    chain_id: str = ""
    root_supplier_id: str = ""
    commodity: str = ""
    nodes: Dict[str, SupplierNode] = field(default_factory=dict)
    total_volume_tonnes: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate ID if not provided."""
        if not self.chain_id:
            self.chain_id = str(uuid.uuid4())


@dataclass
class TierDepthScore:
    """Result of a tier depth assessment.

    Attributes:
        assessment_id: Unique assessment identifier.
        supplier_id: Root supplier assessed.
        commodity: Commodity assessed.
        max_depth_reached: Deepest tier level with known suppliers.
        expected_depth: Expected/typical depth for this commodity.
        depth_ratio: Ratio of reached to expected depth.
        visibility_score: Percentage of known suppliers (0-100).
        coverage_score: Volume-weighted traceability (0-100).
        tier_breakdown: Supplier count per tier level.
        volume_breakdown: Volume per tier level.
        gaps: List of detected tier gaps.
        benchmark_category: Industry benchmark classification.
        benchmark_label: Human-readable benchmark label.
        visibility_level: Categorical visibility level.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 provenance hash.
        timestamp: Assessment timestamp.
        metadata: Additional key-value metadata.
    """

    assessment_id: str = ""
    supplier_id: str = ""
    commodity: str = ""
    max_depth_reached: int = 0
    expected_depth: int = 0
    depth_ratio: float = 0.0
    visibility_score: float = 0.0
    coverage_score: float = 0.0
    tier_breakdown: Dict[int, int] = field(default_factory=dict)
    volume_breakdown: Dict[int, float] = field(default_factory=dict)
    gaps: List[Dict[str, Any]] = field(default_factory=list)
    benchmark_category: str = ""
    benchmark_label: str = ""
    visibility_level: str = VisibilityLevel.NONE.value
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate ID and timestamp if not provided."""
        if not self.assessment_id:
            self.assessment_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class TierGap:
    """A gap in tier coverage.

    Attributes:
        gap_id: Unique gap identifier.
        tier_level: Tier level with the gap.
        gap_type: Type of gap (no_suppliers, incomplete, unverified).
        severity: Gap severity.
        description: Human-readable description.
        affected_volume_tonnes: Volume affected by this gap.
        affected_volume_percentage: Percentage of total volume affected.
        remediation_suggestion: Suggested remediation action.
        metadata: Additional metadata.
    """

    gap_id: str = ""
    tier_level: int = 0
    gap_type: str = "no_suppliers"
    severity: str = GapSeverity.MAJOR.value
    description: str = ""
    affected_volume_tonnes: float = 0.0
    affected_volume_percentage: float = 0.0
    remediation_suggestion: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate ID if not provided."""
        if not self.gap_id:
            self.gap_id = str(uuid.uuid4())


@dataclass
class DepthTrendPoint:
    """A single point in a depth trend time series.

    Attributes:
        timestamp: Measurement timestamp (ISO 8601).
        depth_reached: Max depth at this point.
        visibility_score: Visibility score at this point.
        coverage_score: Coverage score at this point.
        supplier_count: Total supplier count at this point.
    """

    timestamp: str = ""
    depth_reached: int = 0
    visibility_score: float = 0.0
    coverage_score: float = 0.0
    supplier_count: int = 0


@dataclass
class DepthTrend:
    """Time-series depth tracking result.

    Attributes:
        supplier_id: Supplier being tracked.
        commodity: Commodity being tracked.
        data_points: List of trend data points.
        trend_direction: Overall trend direction.
        depth_change: Change in depth over the period.
        visibility_change: Change in visibility score.
        period_start: Start of the tracking period.
        period_end: End of the tracking period.
        provenance_hash: SHA-256 provenance hash.
    """

    supplier_id: str = ""
    commodity: str = ""
    data_points: List[DepthTrendPoint] = field(default_factory=list)
    trend_direction: str = TrendDirection.STABLE.value
    depth_change: int = 0
    visibility_change: float = 0.0
    period_start: str = ""
    period_end: str = ""
    provenance_hash: str = ""


@dataclass
class BenchmarkResult:
    """Result of benchmarking depth against industry averages.

    Attributes:
        supplier_id: Supplier benchmarked.
        commodity: Commodity benchmarked.
        actual_depth: Actual achieved depth.
        industry_avg: Industry average depth.
        best_in_class: Best-in-class depth.
        typical_depth: Typical expected depth.
        depth_ratio: Ratio of actual to typical.
        category: Benchmark category.
        label: Human-readable label.
        percentile: Estimated percentile ranking.
        provenance_hash: SHA-256 provenance hash.
    """

    supplier_id: str = ""
    commodity: str = ""
    actual_depth: int = 0
    industry_avg: float = 0.0
    best_in_class: float = 0.0
    typical_depth: int = 0
    depth_ratio: float = 0.0
    category: str = ""
    label: str = ""
    percentile: float = 0.0
    provenance_hash: str = ""


@dataclass
class TierMapNode:
    """A node in the hierarchical tier map.

    Attributes:
        supplier_id: Supplier identifier.
        name: Supplier name.
        tier_level: Tier level.
        country_code: ISO country code.
        volume_tonnes: Volume through this node.
        has_gps: Whether GPS data exists.
        has_certification: Whether certifications exist.
        children: List of child (upstream) nodes.
        metadata: Additional metadata.
    """

    supplier_id: str = ""
    name: str = ""
    tier_level: int = 0
    country_code: str = ""
    volume_tonnes: float = 0.0
    has_gps: bool = False
    has_certification: bool = False
    children: List[TierMapNode] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchTierResult:
    """Result of a batch tier assessment.

    Attributes:
        batch_id: Unique batch identifier.
        total_assessed: Number of chains assessed.
        avg_depth: Average depth across all chains.
        avg_visibility: Average visibility score.
        avg_coverage: Average coverage score.
        individual_results: Per-chain results.
        processing_time_ms: Total processing duration.
        provenance_hash: SHA-256 provenance hash.
        timestamp: Result generation timestamp.
    """

    batch_id: str = ""
    total_assessed: int = 0
    avg_depth: float = 0.0
    avg_visibility: float = 0.0
    avg_coverage: float = 0.0
    individual_results: List[TierDepthScore] = field(
        default_factory=list
    )
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        """Generate ID and timestamp if not provided."""
        if not self.batch_id:
            self.batch_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    """Return current UTC timestamp as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _compute_provenance_hash(data: Any) -> str:
    """Compute SHA-256 provenance hash for any serializable data.

    Args:
        data: Data to hash.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    try:
        if hasattr(data, "__dict__"):
            serialized = json.dumps(
                data.__dict__, sort_keys=True, default=str
            )
        else:
            serialized = json.dumps(data, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized = str(data)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _classify_visibility(score: float) -> str:
    """Classify a visibility score into a visibility level.

    Args:
        score: Visibility score (0-100).

    Returns:
        VisibilityLevel value string.
    """
    if score >= 90.0:
        return VisibilityLevel.FULL.value
    if score >= 70.0:
        return VisibilityLevel.HIGH.value
    if score >= 40.0:
        return VisibilityLevel.MODERATE.value
    if score > 0.0:
        return VisibilityLevel.LOW.value
    return VisibilityLevel.NONE.value


def _classify_benchmark(depth_ratio: float) -> Tuple[str, str]:
    """Classify a depth ratio into a benchmark category.

    Args:
        depth_ratio: Ratio of actual depth to typical depth.

    Returns:
        Tuple of (category_key, human-readable label).
    """
    for category, info in BENCHMARK_CATEGORIES.items():
        if depth_ratio >= info["min_ratio"]:
            return category, info["label"]
    return "lagging", "Industry Lagging"


# ---------------------------------------------------------------------------
# TierDepthTracker
# ---------------------------------------------------------------------------


class TierDepthTracker:
    """Engine 3: Tracks and scores supply chain tier depth.

    Provides comprehensive tier depth assessment, visibility scoring,
    coverage scoring, gap detection, commodity-specific depth
    references, trend tracking, industry benchmarking, and
    hierarchical tier map generation.

    All scoring is deterministic with SHA-256 provenance hashes.

    Attributes:
        _assessment_history: Historical assessments for trend tracking.
        _assessment_count: Running count for metrics.

    Example:
        >>> tracker = TierDepthTracker()
        >>> chain = SupplierChain(
        ...     root_supplier_id="SUP-001",
        ...     commodity="cocoa",
        ...     nodes={"SUP-001": SupplierNode(tier_level=1)},
        ... )
        >>> score = tracker.assess_tier_depth("SUP-001", chain)
        >>> assert 0 <= score.visibility_score <= 100
    """

    def __init__(self) -> None:
        """Initialize TierDepthTracker."""
        self._assessment_history: Dict[
            str, List[TierDepthScore]
        ] = defaultdict(list)
        self._assessment_count: int = 0

        logger.info(
            "TierDepthTracker initialized: version=%s, "
            "commodity_ref_count=%d",
            ENGINE_VERSION,
            len(COMMODITY_TIER_DEPTHS),
        )

    # ------------------------------------------------------------------
    # Tier Depth Assessment
    # ------------------------------------------------------------------

    def assess_tier_depth(
        self,
        supplier_id: str,
        chain: SupplierChain,
    ) -> TierDepthScore:
        """Calculate tier depth for a supplier chain.

        Analyzes the supply chain structure to determine maximum
        depth reached, visibility and coverage scores, tier
        breakdown, and gaps.

        Args:
            supplier_id: Root supplier to assess from.
            chain: The supply chain structure.

        Returns:
            TierDepthScore with comprehensive depth metrics.
        """
        start_time = time.monotonic()

        logger.info(
            "Assessing tier depth: supplier_id=%s, commodity=%s, "
            "nodes=%d",
            supplier_id,
            chain.commodity,
            len(chain.nodes),
        )

        # Calculate tier breakdown (count per tier)
        tier_breakdown: Dict[int, int] = defaultdict(int)
        volume_breakdown: Dict[int, float] = defaultdict(float)

        for node in chain.nodes.values():
            tier_breakdown[node.tier_level] += 1
            volume_breakdown[node.tier_level] += node.volume_tonnes

        # Find max depth
        max_depth = max(tier_breakdown.keys()) if tier_breakdown else 0

        # Get expected depth for commodity
        commodity_ref = COMMODITY_TIER_DEPTHS.get(
            chain.commodity, {}
        )
        expected_depth = commodity_ref.get("typical_tiers", 5)

        # Calculate depth ratio
        depth_ratio = (
            max_depth / expected_depth if expected_depth > 0 else 0.0
        )

        # Calculate visibility score
        visibility_score = self.calculate_visibility_score(chain)

        # Calculate coverage score
        coverage_score = self.calculate_coverage_score(chain)

        # Detect gaps
        gaps = self.detect_tier_gaps(chain)

        # Benchmark
        benchmark_cat, benchmark_label = _classify_benchmark(
            depth_ratio
        )
        visibility_level = _classify_visibility(visibility_score)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Build provenance
        provenance_data = {
            "supplier_id": supplier_id,
            "commodity": chain.commodity,
            "max_depth": max_depth,
            "expected_depth": expected_depth,
            "visibility_score": visibility_score,
            "coverage_score": coverage_score,
            "node_count": len(chain.nodes),
            "timestamp": _utcnow_iso(),
        }
        provenance_hash = _compute_provenance_hash(provenance_data)

        score = TierDepthScore(
            supplier_id=supplier_id,
            commodity=chain.commodity,
            max_depth_reached=max_depth,
            expected_depth=expected_depth,
            depth_ratio=round(depth_ratio, 3),
            visibility_score=round(visibility_score, 1),
            coverage_score=round(coverage_score, 1),
            tier_breakdown=dict(tier_breakdown),
            volume_breakdown={
                k: round(v, 2) for k, v in volume_breakdown.items()
            },
            gaps=[self._gap_to_dict(g) for g in gaps],
            benchmark_category=benchmark_cat,
            benchmark_label=benchmark_label,
            visibility_level=visibility_level,
            processing_time_ms=round(elapsed_ms, 2),
            provenance_hash=provenance_hash,
        )

        # Store for trend tracking
        self._assessment_history[supplier_id].append(score)
        self._assessment_count += 1

        logger.info(
            "Tier depth assessed: supplier_id=%s, "
            "max_depth=%d, expected=%d, ratio=%.2f, "
            "visibility=%.1f, coverage=%.1f, "
            "benchmark=%s, gaps=%d, duration_ms=%.2f",
            supplier_id,
            max_depth,
            expected_depth,
            depth_ratio,
            visibility_score,
            coverage_score,
            benchmark_label,
            len(gaps),
            elapsed_ms,
        )

        return score

    # ------------------------------------------------------------------
    # Visibility Score
    # ------------------------------------------------------------------

    def calculate_visibility_score(
        self, chain: SupplierChain
    ) -> float:
        """Score visibility coverage percentage across tiers.

        Visibility score represents the percentage of expected
        tier levels that have at least one known supplier.
        Weighted by tier importance (deeper tiers have slightly
        lower weight since they are harder to map).

        Formula:
            visibility = sum(tier_weight[i] * has_supplier[i])
                         / sum(tier_weight[i]) * 100

        Where tier_weight[i] = 1.0 - (i-1) * decay_factor

        Args:
            chain: Supply chain to score.

        Returns:
            Visibility score between 0.0 and 100.0.
        """
        if not chain.nodes:
            return 0.0

        commodity_ref = COMMODITY_TIER_DEPTHS.get(
            chain.commodity, {}
        )
        expected_depth = commodity_ref.get("typical_tiers", 5)

        # Tier weight decay factor (deeper = slightly less weight)
        decay_factor = 0.05

        # Build tier presence map
        tier_presence: Dict[int, bool] = {}
        for node in chain.nodes.values():
            tier_presence[node.tier_level] = True

        weighted_sum = 0.0
        weight_total = 0.0

        for tier in range(1, expected_depth + 1):
            weight = max(0.1, 1.0 - (tier - 1) * decay_factor)
            weight_total += weight
            if tier_presence.get(tier, False):
                weighted_sum += weight

        if weight_total <= 0:
            return 0.0

        score = (weighted_sum / weight_total) * 100.0
        return max(0.0, min(100.0, score))

    # ------------------------------------------------------------------
    # Coverage Score
    # ------------------------------------------------------------------

    def calculate_coverage_score(
        self, chain: SupplierChain
    ) -> float:
        """Volume-weighted coverage percentage.

        Coverage score represents what percentage of the total
        commodity volume flowing through this chain can be traced
        to its origin (deepest known tier). Higher volumes at
        deeper tiers increase the score.

        Formula:
            coverage = (traced_volume / total_volume) * 100

        Where traced_volume is the sum of volumes at the deepest
        tier that has known suppliers.

        Args:
            chain: Supply chain to score.

        Returns:
            Coverage score between 0.0 and 100.0.
        """
        if not chain.nodes:
            return 0.0

        total_volume = chain.total_volume_tonnes
        if total_volume <= 0:
            # If no volume data, fall back to count-based coverage
            return self._count_based_coverage(chain)

        # Volume at the deepest tier with suppliers = traced volume
        tier_volumes: Dict[int, float] = defaultdict(float)
        for node in chain.nodes.values():
            tier_volumes[node.tier_level] += node.volume_tonnes

        if not tier_volumes:
            return 0.0

        max_tier = max(tier_volumes.keys())
        deepest_volume = tier_volumes.get(max_tier, 0.0)

        # Also consider partial volume at intermediate tiers
        traced_volume = 0.0
        for tier in sorted(tier_volumes.keys(), reverse=True):
            tier_vol = tier_volumes[tier]
            if tier_vol > 0:
                traced_volume += tier_vol
                break

        if total_volume <= 0:
            return 0.0

        coverage = (traced_volume / total_volume) * 100.0
        return max(0.0, min(100.0, coverage))

    def _count_based_coverage(
        self, chain: SupplierChain
    ) -> float:
        """Calculate coverage based on supplier count when no volume data.

        Args:
            chain: Supply chain to score.

        Returns:
            Count-based coverage score (0-100).
        """
        if not chain.nodes:
            return 0.0

        commodity_ref = COMMODITY_TIER_DEPTHS.get(
            chain.commodity, {}
        )
        expected_depth = commodity_ref.get("typical_tiers", 5)

        tier_counts: Dict[int, int] = defaultdict(int)
        for node in chain.nodes.values():
            tier_counts[node.tier_level] += 1

        max_tier = max(tier_counts.keys()) if tier_counts else 0
        covered_tiers = sum(
            1 for t in range(1, expected_depth + 1)
            if tier_counts.get(t, 0) > 0
        )

        if expected_depth <= 0:
            return 0.0

        return (covered_tiers / expected_depth) * 100.0

    # ------------------------------------------------------------------
    # Gap Detection
    # ------------------------------------------------------------------

    def detect_tier_gaps(
        self, chain: SupplierChain
    ) -> List[TierGap]:
        """Identify tiers with no known suppliers (gaps).

        Scans from Tier 1 through the expected depth for the
        commodity, identifying tiers where no suppliers are known.
        Classifies gap severity based on position and impact.

        Args:
            chain: Supply chain to analyze.

        Returns:
            List of TierGap instances describing each gap.
        """
        if not chain.nodes:
            return []

        commodity_ref = COMMODITY_TIER_DEPTHS.get(
            chain.commodity, {}
        )
        expected_depth = commodity_ref.get("typical_tiers", 5)
        chain_structure = commodity_ref.get("chain_structure", [])

        # Build tier presence and volume maps
        tier_counts: Dict[int, int] = defaultdict(int)
        tier_volumes: Dict[int, float] = defaultdict(float)
        for node in chain.nodes.values():
            tier_counts[node.tier_level] += 1
            tier_volumes[node.tier_level] += node.volume_tonnes

        gaps: List[TierGap] = []
        total_volume = chain.total_volume_tonnes

        for tier in range(1, expected_depth + 1):
            count = tier_counts.get(tier, 0)

            if count == 0:
                # No suppliers at this tier
                severity = self._classify_gap_severity(
                    tier, expected_depth
                )
                role_name = (
                    chain_structure[tier - 1]
                    if tier <= len(chain_structure)
                    else f"Tier {tier}"
                )

                # Estimate affected volume (carry forward from
                # nearest known tier above)
                affected_volume = 0.0
                for check_tier in range(tier - 1, 0, -1):
                    if tier_volumes.get(check_tier, 0) > 0:
                        affected_volume = tier_volumes[check_tier]
                        break

                affected_pct = (
                    (affected_volume / total_volume * 100.0)
                    if total_volume > 0
                    else 0.0
                )

                gap = TierGap(
                    tier_level=tier,
                    gap_type="no_suppliers",
                    severity=severity,
                    description=(
                        f"No known suppliers at Tier {tier} "
                        f"({role_name}). Expected {role_name} "
                        f"entities in a typical {chain.commodity} "
                        f"supply chain."
                    ),
                    affected_volume_tonnes=affected_volume,
                    affected_volume_percentage=round(
                        affected_pct, 1
                    ),
                    remediation_suggestion=(
                        f"Request Tier {tier - 1} suppliers to "
                        f"declare their upstream {role_name} "
                        f"partners via supplier questionnaire."
                    ),
                )
                gaps.append(gap)

            else:
                # Check for data quality gaps at this tier
                quality_gaps = self._check_tier_data_quality(
                    chain, tier
                )
                gaps.extend(quality_gaps)

        logger.debug(
            "Gap detection completed: commodity=%s, "
            "expected_depth=%d, gaps_found=%d",
            chain.commodity,
            expected_depth,
            len(gaps),
        )

        return gaps

    def _classify_gap_severity(
        self, tier: int, expected_depth: int
    ) -> str:
        """Classify gap severity based on tier position.

        Args:
            tier: Tier level with the gap.
            expected_depth: Expected total depth.

        Returns:
            GapSeverity value string.
        """
        if tier <= 2:
            return GapSeverity.CRITICAL.value
        if tier <= expected_depth // 2:
            return GapSeverity.MAJOR.value
        if tier <= expected_depth:
            return GapSeverity.MINOR.value
        return GapSeverity.INFO.value

    def _check_tier_data_quality(
        self, chain: SupplierChain, tier: int
    ) -> List[TierGap]:
        """Check data quality at a tier with known suppliers.

        Args:
            chain: Supply chain.
            tier: Tier level to check.

        Returns:
            List of data quality TierGap instances.
        """
        gaps: List[TierGap] = []
        tier_nodes = [
            n for n in chain.nodes.values()
            if n.tier_level == tier
        ]

        if not tier_nodes:
            return gaps

        # Check GPS coverage
        gps_count = sum(1 for n in tier_nodes if n.has_gps)
        gps_pct = (gps_count / len(tier_nodes)) * 100.0

        if gps_pct < 50.0:
            gaps.append(TierGap(
                tier_level=tier,
                gap_type="incomplete",
                severity=GapSeverity.MAJOR.value
                if tier <= 3
                else GapSeverity.MINOR.value,
                description=(
                    f"Only {gps_pct:.0f}% of Tier {tier} suppliers "
                    f"have GPS coordinates ({gps_count}/{len(tier_nodes)})"
                ),
                remediation_suggestion=(
                    f"Request GPS coordinates from Tier {tier} "
                    f"suppliers without geolocation data."
                ),
            ))

        # Check certification coverage
        cert_count = sum(
            1 for n in tier_nodes if n.has_certification
        )
        cert_pct = (cert_count / len(tier_nodes)) * 100.0

        if cert_pct < 50.0 and tier <= 3:
            gaps.append(TierGap(
                tier_level=tier,
                gap_type="unverified",
                severity=GapSeverity.MINOR.value,
                description=(
                    f"Only {cert_pct:.0f}% of Tier {tier} suppliers "
                    f"have certifications "
                    f"({cert_count}/{len(tier_nodes)})"
                ),
                remediation_suggestion=(
                    f"Encourage Tier {tier} suppliers to obtain "
                    f"relevant certifications (FSC, RSPO, UTZ)."
                ),
            ))

        return gaps

    # ------------------------------------------------------------------
    # Commodity Tier Depth
    # ------------------------------------------------------------------

    def get_commodity_tier_depth(
        self, commodity: str
    ) -> Dict[str, Any]:
        """Get commodity-specific typical chain depth reference data.

        Returns the expected tier structure, typical depths,
        chain composition, and industry average visibility
        for the specified EUDR commodity.

        Args:
            commodity: EUDR commodity name.

        Returns:
            Dictionary with commodity tier depth reference data.
            Returns empty dict if commodity is not recognized.
        """
        commodity_lower = commodity.lower().strip()
        ref = COMMODITY_TIER_DEPTHS.get(commodity_lower)

        if ref is None:
            logger.warning(
                "Unknown commodity for tier depth: %s", commodity
            )
            return {}

        logger.debug(
            "Commodity tier depth retrieved: commodity=%s, "
            "typical=%d, range=%d-%d",
            commodity,
            ref["typical_tiers"],
            ref["min_tiers"],
            ref["max_tiers"],
        )

        return dict(ref)

    # ------------------------------------------------------------------
    # Trend Tracking
    # ------------------------------------------------------------------

    def track_depth_trend(
        self, supplier_id: str
    ) -> DepthTrend:
        """Time-series depth tracking for a supplier.

        Analyzes historical assessment data to determine whether
        tier depth visibility is improving, stable, or degrading
        over time.

        Args:
            supplier_id: Supplier to track.

        Returns:
            DepthTrend with historical data and trend direction.
        """
        history = self._assessment_history.get(supplier_id, [])

        if not history:
            logger.debug(
                "No trend data for supplier: %s", supplier_id
            )
            return DepthTrend(
                supplier_id=supplier_id,
                trend_direction=TrendDirection.STABLE.value,
            )

        data_points: List[DepthTrendPoint] = []
        for assessment in history:
            point = DepthTrendPoint(
                timestamp=assessment.timestamp,
                depth_reached=assessment.max_depth_reached,
                visibility_score=assessment.visibility_score,
                coverage_score=assessment.coverage_score,
                supplier_count=sum(
                    assessment.tier_breakdown.values()
                ),
            )
            data_points.append(point)

        # Determine trend direction
        trend_direction = TrendDirection.STABLE.value
        depth_change = 0
        visibility_change = 0.0

        if len(data_points) >= 2:
            first = data_points[0]
            last = data_points[-1]
            depth_change = (
                last.depth_reached - first.depth_reached
            )
            visibility_change = (
                last.visibility_score - first.visibility_score
            )

            if depth_change > 0 or visibility_change > 5.0:
                trend_direction = TrendDirection.IMPROVING.value
            elif depth_change < 0 or visibility_change < -5.0:
                trend_direction = TrendDirection.DEGRADING.value

        trend = DepthTrend(
            supplier_id=supplier_id,
            commodity=(
                history[-1].commodity if history else ""
            ),
            data_points=data_points,
            trend_direction=trend_direction,
            depth_change=depth_change,
            visibility_change=round(visibility_change, 1),
            period_start=(
                data_points[0].timestamp if data_points else ""
            ),
            period_end=(
                data_points[-1].timestamp if data_points else ""
            ),
            provenance_hash=_compute_provenance_hash({
                "supplier_id": supplier_id,
                "points": len(data_points),
                "trend": trend_direction,
            }),
        )

        logger.info(
            "Depth trend tracked: supplier_id=%s, "
            "points=%d, trend=%s, depth_change=%d, "
            "visibility_change=%.1f",
            supplier_id,
            len(data_points),
            trend_direction,
            depth_change,
            visibility_change,
        )

        return trend

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def benchmark_depth(
        self,
        depth: int,
        commodity: str,
        supplier_id: str = "",
    ) -> BenchmarkResult:
        """Compare tier depth against industry averages.

        Benchmarks the achieved depth against the typical depth
        for the commodity, classifying into categories: leading,
        above_average, average, below_average, lagging.

        Args:
            depth: Actual tier depth achieved.
            commodity: EUDR commodity for comparison.
            supplier_id: Optional supplier ID for context.

        Returns:
            BenchmarkResult with comparison metrics.
        """
        commodity_ref = COMMODITY_TIER_DEPTHS.get(
            commodity.lower().strip(), {}
        )
        typical = commodity_ref.get("typical_tiers", 5)
        industry_avg = commodity_ref.get(
            "industry_avg_visibility", 3.0
        )
        best_class = commodity_ref.get(
            "best_in_class_visibility", 5.0
        )

        depth_ratio = depth / typical if typical > 0 else 0.0
        category, label = _classify_benchmark(depth_ratio)

        # Estimate percentile based on depth ratio
        percentile = min(99.0, max(1.0, depth_ratio * 50.0))
        if depth_ratio >= 1.0:
            percentile = min(99.0, 50.0 + (depth_ratio - 1.0) * 50.0)

        result = BenchmarkResult(
            supplier_id=supplier_id,
            commodity=commodity,
            actual_depth=depth,
            industry_avg=industry_avg,
            best_in_class=best_class,
            typical_depth=typical,
            depth_ratio=round(depth_ratio, 3),
            category=category,
            label=label,
            percentile=round(percentile, 1),
            provenance_hash=_compute_provenance_hash({
                "supplier_id": supplier_id,
                "commodity": commodity,
                "depth": depth,
                "typical": typical,
                "ratio": depth_ratio,
            }),
        )

        logger.info(
            "Depth benchmarked: commodity=%s, depth=%d, "
            "typical=%d, ratio=%.2f, category=%s, "
            "percentile=%.1f",
            commodity,
            depth,
            typical,
            depth_ratio,
            label,
            percentile,
        )

        return result

    # ------------------------------------------------------------------
    # Tier Map Generation
    # ------------------------------------------------------------------

    def generate_tier_map(
        self, root_supplier: SupplierNode, chain: SupplierChain
    ) -> TierMapNode:
        """Generate hierarchical tier map data for visualization.

        Builds a tree structure from the supply chain data,
        suitable for rendering as a hierarchical visualization.

        Args:
            root_supplier: The root (Tier 1) supplier node.
            chain: The full supply chain.

        Returns:
            TierMapNode tree rooted at the given supplier.
        """
        start_time = time.monotonic()

        logger.info(
            "Generating tier map: root=%s (%s), nodes=%d",
            root_supplier.name,
            root_supplier.supplier_id,
            len(chain.nodes),
        )

        # Build adjacency: downstream -> list of upstream
        adjacency: Dict[str, List[str]] = defaultdict(list)
        for node in chain.nodes.values():
            for buyer_id in node.downstream_buyer_ids:
                adjacency[buyer_id].append(node.supplier_id)

        # Build tree via BFS from root
        visited: Set[str] = set()
        root_map = self._build_map_node(
            root_supplier, chain, adjacency, visited
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "Tier map generated: root=%s, visited=%d, "
            "duration_ms=%.2f",
            root_supplier.name,
            len(visited),
            elapsed_ms,
        )

        return root_map

    def _build_map_node(
        self,
        node: SupplierNode,
        chain: SupplierChain,
        adjacency: Dict[str, List[str]],
        visited: Set[str],
    ) -> TierMapNode:
        """Recursively build a TierMapNode tree.

        Args:
            node: Current supplier node.
            chain: Full supply chain.
            adjacency: Downstream -> upstream adjacency map.
            visited: Set of already-visited supplier IDs.

        Returns:
            TierMapNode with children populated.
        """
        visited.add(node.supplier_id)

        children: List[TierMapNode] = []
        upstream_ids = adjacency.get(node.supplier_id, [])

        for upstream_id in upstream_ids:
            if upstream_id in visited:
                continue
            upstream_node = chain.nodes.get(upstream_id)
            if upstream_node is not None:
                child_map = self._build_map_node(
                    upstream_node, chain, adjacency, visited
                )
                children.append(child_map)

        return TierMapNode(
            supplier_id=node.supplier_id,
            name=node.name,
            tier_level=node.tier_level,
            country_code=node.country_code,
            volume_tonnes=node.volume_tonnes,
            has_gps=node.has_gps,
            has_certification=node.has_certification,
            children=children,
            metadata=node.metadata,
        )

    # ------------------------------------------------------------------
    # Batch Assessment
    # ------------------------------------------------------------------

    def batch_assess(
        self, suppliers: List[Tuple[str, SupplierChain]]
    ) -> BatchTierResult:
        """Batch tier assessment across multiple supply chains.

        Args:
            suppliers: List of (supplier_id, chain) tuples to assess.

        Returns:
            BatchTierResult with aggregated metrics.
        """
        start_time = time.monotonic()
        batch_id = str(uuid.uuid4())

        logger.info(
            "Starting batch tier assessment: batch_id=%s, count=%d",
            batch_id,
            len(suppliers),
        )

        individual_results: List[TierDepthScore] = []
        total_depth = 0.0
        total_visibility = 0.0
        total_coverage = 0.0

        for supplier_id, chain in suppliers:
            try:
                result = self.assess_tier_depth(supplier_id, chain)
                individual_results.append(result)
                total_depth += result.max_depth_reached
                total_visibility += result.visibility_score
                total_coverage += result.coverage_score
            except Exception as exc:
                logger.warning(
                    "Batch assessment failed for %s: %s",
                    supplier_id,
                    exc,
                )

        count = len(individual_results) or 1
        elapsed_ms = (time.monotonic() - start_time) * 1000

        batch_result = BatchTierResult(
            batch_id=batch_id,
            total_assessed=len(individual_results),
            avg_depth=round(total_depth / count, 1),
            avg_visibility=round(total_visibility / count, 1),
            avg_coverage=round(total_coverage / count, 1),
            individual_results=individual_results,
            processing_time_ms=round(elapsed_ms, 2),
            provenance_hash=_compute_provenance_hash({
                "batch_id": batch_id,
                "count": len(individual_results),
                "avg_depth": total_depth / count,
            }),
        )

        logger.info(
            "Batch tier assessment completed: batch_id=%s, "
            "assessed=%d, avg_depth=%.1f, "
            "avg_visibility=%.1f, avg_coverage=%.1f, "
            "duration_ms=%.2f",
            batch_id,
            len(individual_results),
            batch_result.avg_depth,
            batch_result.avg_visibility,
            batch_result.avg_coverage,
            elapsed_ms,
        )

        return batch_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gap_to_dict(gap: TierGap) -> Dict[str, Any]:
        """Convert a TierGap to a dictionary.

        Args:
            gap: TierGap instance.

        Returns:
            Dictionary representation.
        """
        return {
            "gap_id": gap.gap_id,
            "tier_level": gap.tier_level,
            "gap_type": gap.gap_type,
            "severity": gap.severity,
            "description": gap.description,
            "affected_volume_tonnes": gap.affected_volume_tonnes,
            "affected_volume_percentage": (
                gap.affected_volume_percentage
            ),
            "remediation_suggestion": gap.remediation_suggestion,
        }

    # ------------------------------------------------------------------
    # Metrics accessors
    # ------------------------------------------------------------------

    @property
    def total_assessments(self) -> int:
        """Return total number of assessments performed.

        Returns:
            Running assessment count.
        """
        return self._assessment_count

    def reset_metrics(self) -> None:
        """Reset internal metrics counters."""
        self._assessment_count = 0
        self._assessment_history.clear()
        logger.debug("TierDepthTracker metrics reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    # Engine
    "TierDepthTracker",
    # Enums
    "VisibilityLevel",
    "GapSeverity",
    "TrendDirection",
    "EUDRCommodity",
    # Data classes
    "SupplierNode",
    "SupplierChain",
    "TierDepthScore",
    "TierGap",
    "DepthTrendPoint",
    "DepthTrend",
    "BenchmarkResult",
    "TierMapNode",
    "BatchTierResult",
    # Reference data
    "COMMODITY_TIER_DEPTHS",
    "BENCHMARK_CATEGORIES",
    # Constants
    "ENGINE_VERSION",
    "METRIC_PREFIX",
    "DEFAULT_BATCH_SIZE",
    "MAX_TIER_DEPTH",
]
