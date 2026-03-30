# -*- coding: utf-8 -*-
"""
PeerGroupConstructionEngine - PACK-047 GHG Emissions Benchmark Engine 1
====================================================================

Constructs statistically robust peer groups for GHG emissions benchmarking
using multi-dimensional similarity scoring across sector classification,
revenue band, geographic mix, and value chain position.

Calculation Methodology:
    Sector Similarity (configurable weights):
        sim(A, B) = SUM(w_i * match(A_i, B_i))

        Where:
            w_i         = weight for classification level i
            match(A, B) = 1.0 if codes match at level i, else 0.0
            Levels:     GICS (sector/industry group/industry/sub-industry)
                        NACE (section/division/group/class)
                        ISIC (section/division/group/class)
                        SIC  (division/major group/industry group/industry)

    Size Band Distance:
        d_size = |ln(rev_A) - ln(rev_B)| / ln(10)

        Revenue bands:
            MICRO:      <2M
            SMALL:      2M - 10M
            MEDIUM:     10M - 50M
            LARGE:      50M - 250M
            ENTERPRISE: 250M - 1B
            MEGA:       1B+

    Geographic Similarity:
        sim_geo = 1 - |ef_A - ef_B| / max(ef_A, ef_B)

        Where ef = grid emission factor (tCO2e/MWh) for the entity's primary region.

    Composite Similarity:
        S_total = w_sector * sim_sector + w_size * (1 - d_size) + w_geo * sim_geo + w_vc * match_vc

    Peer Quality Scoring:
        Q = w_recency * recency_score + w_scope * scope_score + w_verification * verification_score

    Outlier Detection (IQR method):
        IQR       = Q3 - Q1
        lower     = Q1 - k * IQR
        upper     = Q3 + k * IQR
        outlier   = value < lower OR value > upper
        Default k = 1.5

Regulatory References:
    - GHG Protocol Corporate Standard: Peer comparison guidance
    - ESRS E1-6: Sector benchmark context
    - CDP Climate Change C0.3: Sector classification
    - TPI: Sector-based peer grouping methodology
    - SBTi SDA: Sector classification for target-setting
    - GICS, NACE Rev. 2, ISIC Rev. 4, SIC classification systems

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Peer group construction uses published classification systems only
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-047 GHG Emissions Benchmark
Engine:  1 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

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
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0"),
) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _round2(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _round4(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

def _round6(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))

def _median_decimal(values: List[Decimal]) -> Decimal:
    if not values:
        return Decimal("0")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / Decimal("2")

def _percentile_decimal(values: List[Decimal], pct: Decimal) -> Decimal:
    """Compute the p-th percentile using linear interpolation."""
    if not values:
        return Decimal("0")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    rank = (pct / Decimal("100")) * Decimal(str(n - 1))
    lower = int(rank)
    upper = lower + 1
    if upper >= n:
        return sorted_vals[-1]
    frac = rank - Decimal(str(lower))
    return sorted_vals[lower] + frac * (sorted_vals[upper] - sorted_vals[lower])

def _std_deviation_decimal(values: List[Decimal]) -> Decimal:
    if len(values) < 2:
        return Decimal("0")
    n = Decimal(str(len(values)))
    mean = sum(values) / n
    squared_diffs = [(v - mean) ** 2 for v in values]
    variance = sum(squared_diffs) / n
    std_float = float(variance) ** 0.5
    return _decimal(std_float)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ClassificationSystem(str, Enum):
    """Sector classification system.

    GICS:  Global Industry Classification Standard (MSCI/S&P).
    NACE:  Statistical Classification of Economic Activities (EU).
    ISIC:  International Standard Industrial Classification (UN).
    SIC:   Standard Industrial Classification (US).
    """
    GICS = "GICS"
    NACE = "NACE"
    ISIC = "ISIC"
    SIC = "SIC"

class RevenueBand(str, Enum):
    """Revenue-band sizing for peer grouping.

    MICRO:      Revenue < 2M.
    SMALL:      Revenue 2M - 10M.
    MEDIUM:     Revenue 10M - 50M.
    LARGE:      Revenue 50M - 250M.
    ENTERPRISE: Revenue 250M - 1B.
    MEGA:       Revenue 1B+.
    """
    MICRO = "micro"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    ENTERPRISE = "enterprise"
    MEGA = "mega"

class ValueChainPosition(str, Enum):
    """Value chain position for filtering.

    UPSTREAM:   Raw materials, extraction, primary processing.
    MIDSTREAM:  Manufacturing, transformation, assembly.
    DOWNSTREAM: Distribution, retail, end-use services.
    INTEGRATED: Vertically integrated across multiple positions.
    """
    UPSTREAM = "upstream"
    MIDSTREAM = "midstream"
    DOWNSTREAM = "downstream"
    INTEGRATED = "integrated"

class VerificationStatus(str, Enum):
    """Verification status of reported data.

    THIRD_PARTY_ASSURED:   Third-party limited/reasonable assurance.
    SELF_REPORTED:         Self-reported by entity.
    ESTIMATED:             Estimated by data provider.
    MODELLED:              Modelled from sector averages.
    """
    THIRD_PARTY_ASSURED = "third_party_assured"
    SELF_REPORTED = "self_reported"
    ESTIMATED = "estimated"
    MODELLED = "modelled"

class ScopeCompleteness(str, Enum):
    """Scope boundary completeness.

    S1_ONLY:       Scope 1 only.
    S1_S2:         Scope 1 + 2.
    S1_S2_S3_PARTIAL: Scope 1 + 2 + partial Scope 3.
    S1_S2_S3_FULL:    Scope 1 + 2 + full Scope 3.
    """
    S1_ONLY = "s1_only"
    S1_S2 = "s1_s2"
    S1_S2_S3_PARTIAL = "s1_s2_s3_partial"
    S1_S2_S3_FULL = "s1_s2_s3_full"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Revenue band boundaries in millions (lower, upper)
REVENUE_BAND_BOUNDARIES: Dict[str, Tuple[Decimal, Decimal]] = {
    RevenueBand.MICRO.value: (Decimal("0"), Decimal("2")),
    RevenueBand.SMALL.value: (Decimal("2"), Decimal("10")),
    RevenueBand.MEDIUM.value: (Decimal("10"), Decimal("50")),
    RevenueBand.LARGE.value: (Decimal("50"), Decimal("250")),
    RevenueBand.ENTERPRISE.value: (Decimal("250"), Decimal("1000")),
    RevenueBand.MEGA.value: (Decimal("1000"), Decimal("999999999")),
}

# Default similarity weights
DEFAULT_SECTOR_WEIGHT: Decimal = Decimal("0.40")
DEFAULT_SIZE_WEIGHT: Decimal = Decimal("0.25")
DEFAULT_GEO_WEIGHT: Decimal = Decimal("0.20")
DEFAULT_VALUE_CHAIN_WEIGHT: Decimal = Decimal("0.15")

# Default quality weights
DEFAULT_RECENCY_WEIGHT: Decimal = Decimal("0.40")
DEFAULT_SCOPE_WEIGHT: Decimal = Decimal("0.35")
DEFAULT_VERIFICATION_WEIGHT: Decimal = Decimal("0.25")

# Sector matching depth weights (level 1-4)
DEFAULT_SECTOR_LEVEL_WEIGHTS: List[Decimal] = [
    Decimal("0.10"),  # Level 1 (sector / section / division)
    Decimal("0.20"),  # Level 2 (industry group / division / group)
    Decimal("0.30"),  # Level 3 (industry / group / class)
    Decimal("0.40"),  # Level 4 (sub-industry / class)
]

# Verification status scores
VERIFICATION_SCORES: Dict[str, Decimal] = {
    VerificationStatus.THIRD_PARTY_ASSURED.value: Decimal("1.0"),
    VerificationStatus.SELF_REPORTED.value: Decimal("0.7"),
    VerificationStatus.ESTIMATED.value: Decimal("0.4"),
    VerificationStatus.MODELLED.value: Decimal("0.2"),
}

# Scope completeness scores
SCOPE_COMPLETENESS_SCORES: Dict[str, Decimal] = {
    ScopeCompleteness.S1_ONLY.value: Decimal("0.25"),
    ScopeCompleteness.S1_S2.value: Decimal("0.50"),
    ScopeCompleteness.S1_S2_S3_PARTIAL.value: Decimal("0.75"),
    ScopeCompleteness.S1_S2_S3_FULL.value: Decimal("1.00"),
}

# Peer count bounds
DEFAULT_MIN_PEERS: int = 10
MAX_PEER_CANDIDATES: int = 50000
DEFAULT_IQR_MULTIPLIER: Decimal = Decimal("1.5")
MAX_DATA_AGE_YEARS: int = 5

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class SectorMapping(BaseModel):
    """Sector classification mapping for an entity.

    Attributes:
        system:          Classification system.
        codes:           Hierarchical codes from broad to specific.
        descriptions:    Human-readable descriptions per level.
    """
    system: ClassificationSystem = Field(..., description="Classification system")
    codes: List[str] = Field(default_factory=list, description="Hierarchical sector codes")
    descriptions: List[str] = Field(default_factory=list, description="Level descriptions")

class PeerCandidate(BaseModel):
    """A candidate entity for peer group inclusion.

    Attributes:
        entity_id:              Unique entity identifier.
        entity_name:            Human-readable name.
        sector_mappings:        Sector classifications.
        revenue_millions:       Annual revenue in millions (common currency).
        grid_emission_factor:   Grid emission factor (tCO2e/MWh) for primary region.
        region:                 Primary geographic region.
        value_chain_position:   Position in value chain.
        total_emissions_tco2e:  Total reported emissions.
        intensity_value:        Emissions intensity.
        intensity_unit:         Intensity unit.
        reporting_year:         Year of reported data.
        scope_completeness:     Scope boundary completeness.
        verification_status:    Verification status of data.
        data_source:            Data source identifier.
    """
    entity_id: str = Field(..., description="Entity ID")
    entity_name: str = Field(default="", description="Entity name")
    sector_mappings: List[SectorMapping] = Field(
        default_factory=list, description="Sector classifications"
    )
    revenue_millions: Decimal = Field(default=Decimal("0"), ge=0, description="Revenue (millions)")
    grid_emission_factor: Decimal = Field(
        default=Decimal("0.5"), ge=0, description="Grid EF (tCO2e/MWh)"
    )
    region: str = Field(default="", description="Primary region")
    value_chain_position: ValueChainPosition = Field(
        default=ValueChainPosition.INTEGRATED, description="Value chain position"
    )
    total_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total emissions (tCO2e)"
    )
    intensity_value: Decimal = Field(default=Decimal("0"), ge=0, description="Intensity value")
    intensity_unit: str = Field(default="tCO2e/unit", description="Intensity unit")
    reporting_year: int = Field(default=2024, description="Reporting year")
    scope_completeness: ScopeCompleteness = Field(
        default=ScopeCompleteness.S1_S2, description="Scope completeness"
    )
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.SELF_REPORTED, description="Verification status"
    )
    data_source: str = Field(default="", description="Data source")

    @field_validator("revenue_millions", mode="before")
    @classmethod
    def coerce_revenue(cls, v: Any) -> Decimal:
        return _decimal(v)

    @field_validator("grid_emission_factor", mode="before")
    @classmethod
    def coerce_ef(cls, v: Any) -> Decimal:
        return _decimal(v)

    @field_validator("total_emissions_tco2e", "intensity_value", mode="before")
    @classmethod
    def coerce_emissions(cls, v: Any) -> Decimal:
        return _decimal(v)

class SimilarityWeights(BaseModel):
    """Weights for composite similarity calculation.

    Attributes:
        sector_weight:          Weight for sector similarity.
        size_weight:            Weight for size similarity.
        geo_weight:             Weight for geographic similarity.
        value_chain_weight:     Weight for value chain match.
        sector_level_weights:   Weights per sector classification depth.
    """
    sector_weight: Decimal = Field(default=DEFAULT_SECTOR_WEIGHT, ge=0, le=1)
    size_weight: Decimal = Field(default=DEFAULT_SIZE_WEIGHT, ge=0, le=1)
    geo_weight: Decimal = Field(default=DEFAULT_GEO_WEIGHT, ge=0, le=1)
    value_chain_weight: Decimal = Field(default=DEFAULT_VALUE_CHAIN_WEIGHT, ge=0, le=1)
    sector_level_weights: List[Decimal] = Field(
        default_factory=lambda: list(DEFAULT_SECTOR_LEVEL_WEIGHTS),
        description="Weights per classification depth level",
    )

    @model_validator(mode="after")
    def check_weights_sum(self) -> "SimilarityWeights":
        total = self.sector_weight + self.size_weight + self.geo_weight + self.value_chain_weight
        if abs(total - Decimal("1")) > Decimal("0.01"):
            logger.warning(
                "Similarity weights sum to %s (expected ~1.0). Results may be skewed.", total
            )
        return self

class QualityWeights(BaseModel):
    """Weights for peer data quality scoring.

    Attributes:
        recency_weight:         Weight for data recency.
        scope_weight:           Weight for scope completeness.
        verification_weight:    Weight for verification status.
    """
    recency_weight: Decimal = Field(default=DEFAULT_RECENCY_WEIGHT, ge=0, le=1)
    scope_weight: Decimal = Field(default=DEFAULT_SCOPE_WEIGHT, ge=0, le=1)
    verification_weight: Decimal = Field(default=DEFAULT_VERIFICATION_WEIGHT, ge=0, le=1)

class PeerGroupConfig(BaseModel):
    """Configuration for peer group construction.

    Attributes:
        reference_year:         Reference year for comparison.
        min_peer_count:         Minimum peers required.
        max_peer_count:         Maximum peers to return.
        min_similarity_score:   Minimum similarity threshold for inclusion.
        similarity_weights:     Similarity dimension weights.
        quality_weights:        Quality dimension weights.
        iqr_multiplier:         IQR multiplier for outlier detection.
        remove_outliers:        Whether to remove outliers.
        filter_value_chain:     Restrict to specific value chain positions.
        filter_regions:         Restrict to specific regions.
        max_data_age_years:     Maximum age of data to include.
        classification_system:  Preferred classification system.
        output_precision:       Decimal places for output.
    """
    reference_year: int = Field(default=2024, description="Reference year")
    min_peer_count: int = Field(default=DEFAULT_MIN_PEERS, ge=1, le=1000)
    max_peer_count: int = Field(default=100, ge=1, le=MAX_PEER_CANDIDATES)
    min_similarity_score: Decimal = Field(default=Decimal("0.30"), ge=0, le=1)
    similarity_weights: SimilarityWeights = Field(
        default_factory=SimilarityWeights, description="Similarity weights"
    )
    quality_weights: QualityWeights = Field(
        default_factory=QualityWeights, description="Quality weights"
    )
    iqr_multiplier: Decimal = Field(default=DEFAULT_IQR_MULTIPLIER, gt=0)
    remove_outliers: bool = Field(default=True, description="Remove outliers")
    filter_value_chain: Optional[List[ValueChainPosition]] = Field(
        default=None, description="Value chain filter"
    )
    filter_regions: Optional[List[str]] = Field(default=None, description="Region filter")
    max_data_age_years: int = Field(default=MAX_DATA_AGE_YEARS, ge=1, le=20)
    classification_system: ClassificationSystem = Field(
        default=ClassificationSystem.GICS, description="Preferred classification"
    )
    output_precision: int = Field(default=4, ge=0, le=12, description="Output precision")

class PeerGroupInput(BaseModel):
    """Input for peer group construction.

    Attributes:
        organisation_id:    Organisation identifier.
        organisation:       The reference organisation.
        candidates:         Pool of candidate peers.
        config:             Construction configuration.
    """
    organisation_id: str = Field(default="", description="Organisation ID")
    organisation: PeerCandidate = Field(..., description="Reference organisation")
    candidates: List[PeerCandidate] = Field(default_factory=list, description="Candidate pool")
    config: PeerGroupConfig = Field(
        default_factory=PeerGroupConfig, description="Configuration"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class PeerQualityScore(BaseModel):
    """Data quality score for a peer.

    Attributes:
        entity_id:          Entity identifier.
        recency_score:      Recency sub-score (0-1).
        scope_score:        Scope completeness sub-score (0-1).
        verification_score: Verification sub-score (0-1).
        composite_score:    Weighted composite quality score (0-1).
    """
    entity_id: str = Field(..., description="Entity ID")
    recency_score: Decimal = Field(default=Decimal("0"), description="Recency score")
    scope_score: Decimal = Field(default=Decimal("0"), description="Scope score")
    verification_score: Decimal = Field(default=Decimal("0"), description="Verification score")
    composite_score: Decimal = Field(default=Decimal("0"), description="Composite quality")

class PeerSimilarityDetail(BaseModel):
    """Detailed similarity breakdown for a peer.

    Attributes:
        entity_id:              Entity identifier.
        entity_name:            Entity name.
        sector_similarity:      Sector similarity sub-score.
        size_similarity:        Size similarity sub-score.
        geo_similarity:         Geographic similarity sub-score.
        value_chain_match:      Value chain match score (0 or 1).
        composite_similarity:   Weighted composite similarity.
        quality_score:          Data quality composite.
        is_outlier:             Whether peer is flagged as outlier.
        revenue_band:           Revenue band.
        intensity_value:        Emissions intensity.
    """
    entity_id: str = Field(..., description="Entity ID")
    entity_name: str = Field(default="", description="Entity name")
    sector_similarity: Decimal = Field(default=Decimal("0"), description="Sector similarity")
    size_similarity: Decimal = Field(default=Decimal("0"), description="Size similarity")
    geo_similarity: Decimal = Field(default=Decimal("0"), description="Geo similarity")
    value_chain_match: Decimal = Field(default=Decimal("0"), description="VC match")
    composite_similarity: Decimal = Field(default=Decimal("0"), description="Composite similarity")
    quality_score: Decimal = Field(default=Decimal("0"), description="Quality score")
    is_outlier: bool = Field(default=False, description="Outlier flag")
    revenue_band: str = Field(default="", description="Revenue band")
    intensity_value: Decimal = Field(default=Decimal("0"), description="Intensity")

class OutlierSummary(BaseModel):
    """Summary of outlier detection.

    Attributes:
        method:             Outlier detection method.
        iqr_multiplier:     IQR multiplier used.
        lower_fence:        Lower fence value.
        upper_fence:        Upper fence value.
        q1:                 First quartile.
        q3:                 Third quartile.
        iqr:                Interquartile range.
        outliers_removed:   Count of outliers removed.
        outlier_entity_ids: IDs of removed outliers.
    """
    method: str = Field(default="IQR", description="Detection method")
    iqr_multiplier: Decimal = Field(default=DEFAULT_IQR_MULTIPLIER, description="IQR multiplier")
    lower_fence: Decimal = Field(default=Decimal("0"), description="Lower fence")
    upper_fence: Decimal = Field(default=Decimal("0"), description="Upper fence")
    q1: Decimal = Field(default=Decimal("0"), description="Q1")
    q3: Decimal = Field(default=Decimal("0"), description="Q3")
    iqr: Decimal = Field(default=Decimal("0"), description="IQR")
    outliers_removed: int = Field(default=0, description="Outliers removed")
    outlier_entity_ids: List[str] = Field(default_factory=list, description="Outlier IDs")

class PeerGroupStats(BaseModel):
    """Distribution statistics for the constructed peer group.

    Attributes:
        count:      Number of peers.
        mean:       Mean intensity.
        median:     Median intensity.
        std_dev:    Standard deviation.
        min_val:    Minimum.
        max_val:    Maximum.
        p10:        10th percentile.
        p25:        25th percentile.
        p75:        75th percentile.
        p90:        90th percentile.
    """
    count: int = Field(default=0, description="Peer count")
    mean: Decimal = Field(default=Decimal("0"), description="Mean")
    median: Decimal = Field(default=Decimal("0"), description="Median")
    std_dev: Decimal = Field(default=Decimal("0"), description="Std dev")
    min_val: Decimal = Field(default=Decimal("0"), description="Minimum")
    max_val: Decimal = Field(default=Decimal("0"), description="Maximum")
    p10: Decimal = Field(default=Decimal("0"), description="10th percentile")
    p25: Decimal = Field(default=Decimal("0"), description="25th percentile")
    p75: Decimal = Field(default=Decimal("0"), description="75th percentile")
    p90: Decimal = Field(default=Decimal("0"), description="90th percentile")

class PeerGroup(BaseModel):
    """Constructed peer group result.

    Attributes:
        group_id:               Unique group identifier.
        organisation_id:        Reference organisation ID.
        peers:                  Included peers with similarity details.
        excluded_peers:         Peers excluded (below threshold or outliers).
        statistics:             Group distribution statistics.
        outlier_summary:        Outlier detection summary.
        avg_similarity:         Average composite similarity of included peers.
        avg_quality:            Average quality score of included peers.
        revenue_band_distribution: Count of peers per revenue band.
        region_distribution:    Count of peers per region.
    """
    group_id: str = Field(default_factory=_new_uuid, description="Group ID")
    organisation_id: str = Field(default="", description="Org ID")
    peers: List[PeerSimilarityDetail] = Field(default_factory=list, description="Included peers")
    excluded_peers: List[PeerSimilarityDetail] = Field(
        default_factory=list, description="Excluded peers"
    )
    statistics: PeerGroupStats = Field(default_factory=PeerGroupStats, description="Stats")
    outlier_summary: Optional[OutlierSummary] = Field(default=None, description="Outlier summary")
    avg_similarity: Decimal = Field(default=Decimal("0"), description="Avg similarity")
    avg_quality: Decimal = Field(default=Decimal("0"), description="Avg quality")
    revenue_band_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Revenue band dist"
    )
    region_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Region distribution"
    )

class PeerGroupResult(BaseModel):
    """Complete result of peer group construction.

    Attributes:
        result_id:              Unique result identifier.
        organisation_id:        Organisation ID.
        peer_group:             Constructed peer group.
        total_candidates:       Total candidates evaluated.
        candidates_passing:     Candidates passing similarity threshold.
        candidates_after_outlier: Candidates after outlier removal.
        final_peer_count:       Final peer count.
        min_peers_met:          Whether minimum peer count was met.
        classification_used:    Classification system used.
        warnings:               Warnings.
        calculated_at:          Timestamp.
        processing_time_ms:     Processing time (ms).
        provenance_hash:        SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    organisation_id: str = Field(default="", description="Organisation ID")
    peer_group: PeerGroup = Field(default_factory=PeerGroup, description="Peer group")
    total_candidates: int = Field(default=0, description="Total candidates")
    candidates_passing: int = Field(default=0, description="Candidates passing threshold")
    candidates_after_outlier: int = Field(default=0, description="After outlier removal")
    final_peer_count: int = Field(default=0, description="Final peer count")
    min_peers_met: bool = Field(default=False, description="Minimum met")
    classification_used: str = Field(default="", description="Classification used")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: str = Field(default="", description="Timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PeerGroupConstructionEngine:
    """Constructs statistically robust peer groups for GHG emissions benchmarking.

    Uses multi-dimensional similarity scoring across sector classification,
    revenue band, geographic grid emission factor similarity, and value chain
    position.  Includes outlier detection via IQR method and peer quality
    scoring for data recency, scope completeness, and verification status.

    Guarantees:
        - Deterministic: Same inputs always produce the same output.
        - Reproducible: Full provenance tracking with SHA-256 hashes.
        - Auditable: Every similarity score documented.
        - Zero-Hallucination: No LLM in any calculation path.
    """

    def __init__(self) -> None:
        self._version = _MODULE_VERSION
        logger.info("PeerGroupConstructionEngine v%s initialised", self._version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, input_data: PeerGroupInput) -> PeerGroupResult:
        """Construct a peer group from candidate pool.

        Args:
            input_data: Organisation, candidates, and configuration.

        Returns:
            PeerGroupResult with constructed peer group, statistics, and outlier details.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []
        config = input_data.config
        prec = config.output_precision
        prec_str = "0." + "0" * prec

        org = input_data.organisation
        candidates = input_data.candidates
        total_candidates = len(candidates)

        if total_candidates > MAX_PEER_CANDIDATES:
            raise ValueError(
                f"Maximum {MAX_PEER_CANDIDATES} candidates allowed (got {total_candidates})"
            )

        # Step 1: Filter by data age
        max_age = config.max_data_age_years
        min_year = config.reference_year - max_age
        filtered = [c for c in candidates if c.reporting_year >= min_year]
        if len(filtered) < len(candidates):
            warnings.append(
                f"Excluded {len(candidates) - len(filtered)} candidates "
                f"with data older than {max_age} years."
            )

        # Step 2: Filter by value chain position
        if config.filter_value_chain:
            vc_set = set(config.filter_value_chain)
            before = len(filtered)
            filtered = [c for c in filtered if c.value_chain_position in vc_set]
            if len(filtered) < before:
                warnings.append(
                    f"Excluded {before - len(filtered)} candidates "
                    f"outside value chain filter."
                )

        # Step 3: Filter by region
        if config.filter_regions:
            region_set = {r.lower() for r in config.filter_regions}
            before = len(filtered)
            filtered = [c for c in filtered if c.region.lower() in region_set]
            if len(filtered) < before:
                warnings.append(
                    f"Excluded {before - len(filtered)} candidates outside region filter."
                )

        # Step 4: Compute similarity and quality scores
        scored_peers: List[PeerSimilarityDetail] = []
        for candidate in filtered:
            sim_detail = self._score_candidate(org, candidate, config, prec_str)
            scored_peers.append(sim_detail)

        # Step 5: Filter by minimum similarity threshold
        passing = [p for p in scored_peers if p.composite_similarity >= config.min_similarity_score]
        excluded_low_sim = [
            p for p in scored_peers if p.composite_similarity < config.min_similarity_score
        ]
        candidates_passing = len(passing)

        if candidates_passing < config.min_peer_count:
            warnings.append(
                f"Only {candidates_passing} candidates pass similarity threshold "
                f"{config.min_similarity_score}. Minimum is {config.min_peer_count}. "
                f"Consider lowering threshold."
            )

        # Step 6: Outlier detection on intensity values
        outlier_summary: Optional[OutlierSummary] = None
        excluded_outliers: List[PeerSimilarityDetail] = []
        if config.remove_outliers and len(passing) >= 4:
            passing, excluded_outliers, outlier_summary = self._remove_outliers(
                passing, config.iqr_multiplier, prec_str
            )

        candidates_after_outlier = len(passing)

        # Step 7: Sort by composite similarity (descending) and truncate
        passing.sort(key=lambda p: p.composite_similarity, reverse=True)
        final_peers = passing[: config.max_peer_count]
        final_peer_count = len(final_peers)

        # Combine all excluded
        all_excluded = excluded_low_sim + excluded_outliers

        # Step 8: Compute group statistics
        intensity_values = [p.intensity_value for p in final_peers if p.intensity_value > Decimal("0")]
        stats = self._compute_stats(intensity_values, prec_str)

        # Step 9: Compute aggregate metrics
        avg_sim = Decimal("0")
        avg_qual = Decimal("0")
        if final_peer_count > 0:
            avg_sim = (
                sum(p.composite_similarity for p in final_peers)
                / Decimal(str(final_peer_count))
            ).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
            avg_qual = (
                sum(p.quality_score for p in final_peers)
                / Decimal(str(final_peer_count))
            ).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        # Revenue band distribution
        rev_dist: Dict[str, int] = {}
        for p in final_peers:
            band = p.revenue_band or "unknown"
            rev_dist[band] = rev_dist.get(band, 0) + 1

        # Region distribution
        region_dist: Dict[str, int] = {}
        for candidate in filtered:
            if any(p.entity_id == candidate.entity_id for p in final_peers):
                r = candidate.region or "unknown"
                region_dist[r] = region_dist.get(r, 0) + 1

        min_peers_met = final_peer_count >= config.min_peer_count

        peer_group = PeerGroup(
            organisation_id=input_data.organisation_id,
            peers=final_peers,
            excluded_peers=all_excluded,
            statistics=stats,
            outlier_summary=outlier_summary,
            avg_similarity=avg_sim,
            avg_quality=avg_qual,
            revenue_band_distribution=rev_dist,
            region_distribution=region_dist,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = PeerGroupResult(
            organisation_id=input_data.organisation_id,
            peer_group=peer_group,
            total_candidates=total_candidates,
            candidates_passing=candidates_passing,
            candidates_after_outlier=candidates_after_outlier,
            final_peer_count=final_peer_count,
            min_peers_met=min_peers_met,
            classification_used=config.classification_system.value,
            warnings=warnings,
            calculated_at=utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def compute_sector_similarity(
        self,
        org_codes: List[str],
        peer_codes: List[str],
        level_weights: Optional[List[Decimal]] = None,
    ) -> Decimal:
        """Compute sector similarity between two entities.

        Formula: sim(A,B) = SUM(w_i * match(A_i, B_i))

        Args:
            org_codes:      Organisation sector codes (hierarchical).
            peer_codes:     Peer sector codes (hierarchical).
            level_weights:  Weights per level (default: DEFAULT_SECTOR_LEVEL_WEIGHTS).

        Returns:
            Similarity score (0-1).
        """
        return self._sector_similarity(org_codes, peer_codes, level_weights)

    def compute_size_distance(
        self, rev_a: Decimal, rev_b: Decimal,
    ) -> Decimal:
        """Compute size band distance between two entities.

        Formula: d_size = |ln(rev_A) - ln(rev_B)| / ln(10)

        Args:
            rev_a: Revenue A (millions).
            rev_b: Revenue B (millions).

        Returns:
            Size distance (0+, lower = more similar).
        """
        return self._size_distance(rev_a, rev_b)

    def compute_geo_similarity(
        self, ef_a: Decimal, ef_b: Decimal,
    ) -> Decimal:
        """Compute geographic similarity by grid emission factor.

        Formula: sim_geo = 1 - |ef_A - ef_B| / max(ef_A, ef_B)

        Args:
            ef_a: Grid emission factor for entity A.
            ef_b: Grid emission factor for entity B.

        Returns:
            Similarity score (0-1).
        """
        return self._geo_similarity(ef_a, ef_b)

    def classify_revenue_band(self, revenue_millions: Decimal) -> str:
        """Classify an entity into a revenue band.

        Args:
            revenue_millions: Revenue in millions.

        Returns:
            Revenue band name.
        """
        return self._classify_revenue_band(revenue_millions)

    # ------------------------------------------------------------------
    # Internal: Similarity Scoring
    # ------------------------------------------------------------------

    def _score_candidate(
        self,
        org: PeerCandidate,
        candidate: PeerCandidate,
        config: PeerGroupConfig,
        prec_str: str,
    ) -> PeerSimilarityDetail:
        """Score a candidate against the reference organisation."""
        weights = config.similarity_weights

        # Sector similarity
        org_codes = self._get_codes_for_system(org.sector_mappings, config.classification_system)
        cand_codes = self._get_codes_for_system(
            candidate.sector_mappings, config.classification_system
        )
        sector_sim = self._sector_similarity(
            org_codes, cand_codes, weights.sector_level_weights
        )

        # Size similarity (1 - distance, clamped to 0)
        size_dist = self._size_distance(org.revenue_millions, candidate.revenue_millions)
        size_sim = max(Decimal("1") - size_dist, Decimal("0"))

        # Geographic similarity
        geo_sim = self._geo_similarity(org.grid_emission_factor, candidate.grid_emission_factor)

        # Value chain match
        vc_match = Decimal("1") if org.value_chain_position == candidate.value_chain_position else Decimal("0")

        # Composite
        composite = (
            weights.sector_weight * sector_sim
            + weights.size_weight * size_sim
            + weights.geo_weight * geo_sim
            + weights.value_chain_weight * vc_match
        ).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        # Quality score
        quality = self._compute_quality_score(candidate, config, prec_str)

        # Revenue band
        rev_band = self._classify_revenue_band(candidate.revenue_millions)

        return PeerSimilarityDetail(
            entity_id=candidate.entity_id,
            entity_name=candidate.entity_name,
            sector_similarity=sector_sim.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            size_similarity=size_sim.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            geo_similarity=geo_sim.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            value_chain_match=vc_match,
            composite_similarity=composite,
            quality_score=quality,
            is_outlier=False,
            revenue_band=rev_band,
            intensity_value=candidate.intensity_value,
        )

    def _sector_similarity(
        self,
        org_codes: List[str],
        peer_codes: List[str],
        level_weights: Optional[List[Decimal]] = None,
    ) -> Decimal:
        """Compute sector similarity: sim(A,B) = SUM(w_i * match(A_i, B_i))."""
        weights = level_weights or DEFAULT_SECTOR_LEVEL_WEIGHTS
        max_levels = min(len(org_codes), len(peer_codes), len(weights))
        if max_levels == 0:
            return Decimal("0")

        total_weight = sum(weights[:max_levels])
        if total_weight == Decimal("0"):
            return Decimal("0")

        score = Decimal("0")
        for i in range(max_levels):
            if org_codes[i].strip().lower() == peer_codes[i].strip().lower():
                score += weights[i]
            else:
                break  # Classification hierarchy: stop at first divergence

        return _safe_divide(score, total_weight, Decimal("0"))

    def _size_distance(self, rev_a: Decimal, rev_b: Decimal) -> Decimal:
        """Size band distance: d_size = |ln(rev_A) - ln(rev_B)| / ln(10)."""
        # Clamp to minimum 0.01 to avoid log(0)
        a = max(float(rev_a), 0.01)
        b = max(float(rev_b), 0.01)
        distance = abs(math.log(a) - math.log(b)) / math.log(10)
        return _decimal(distance).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    def _geo_similarity(self, ef_a: Decimal, ef_b: Decimal) -> Decimal:
        """Geographic similarity: sim_geo = 1 - |ef_A - ef_B| / max(ef_A, ef_B)."""
        max_ef = max(ef_a, ef_b)
        if max_ef == Decimal("0"):
            return Decimal("1")  # Both zero = identical
        diff = abs(ef_a - ef_b)
        sim = Decimal("1") - _safe_divide(diff, max_ef)
        return max(sim, Decimal("0"))

    def _classify_revenue_band(self, revenue_millions: Decimal) -> str:
        """Classify revenue into band."""
        for band_name, (lower, upper) in REVENUE_BAND_BOUNDARIES.items():
            if lower <= revenue_millions < upper:
                return band_name
        return RevenueBand.MEGA.value

    def _get_codes_for_system(
        self, mappings: List[SectorMapping], system: ClassificationSystem,
    ) -> List[str]:
        """Extract codes for a specific classification system."""
        for mapping in mappings:
            if mapping.system == system:
                return mapping.codes
        # Fallback: use first available
        if mappings:
            return mappings[0].codes
        return []

    # ------------------------------------------------------------------
    # Internal: Quality Scoring
    # ------------------------------------------------------------------

    def _compute_quality_score(
        self,
        candidate: PeerCandidate,
        config: PeerGroupConfig,
        prec_str: str,
    ) -> Decimal:
        """Compute peer data quality composite score.

        Q = w_recency * recency + w_scope * scope + w_verify * verify
        """
        qw = config.quality_weights

        # Recency: 1.0 for current year, decreasing by 0.2 per year
        age = max(config.reference_year - candidate.reporting_year, 0)
        recency = max(Decimal("1") - Decimal("0.2") * Decimal(str(age)), Decimal("0"))

        # Scope completeness
        scope = SCOPE_COMPLETENESS_SCORES.get(
            candidate.scope_completeness.value, Decimal("0.25")
        )

        # Verification
        verification = VERIFICATION_SCORES.get(
            candidate.verification_status.value, Decimal("0.2")
        )

        total_weight = qw.recency_weight + qw.scope_weight + qw.verification_weight
        if total_weight == Decimal("0"):
            return Decimal("0")

        composite = (
            qw.recency_weight * recency
            + qw.scope_weight * scope
            + qw.verification_weight * verification
        ) / total_weight

        return composite.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Internal: Outlier Detection
    # ------------------------------------------------------------------

    def _remove_outliers(
        self,
        peers: List[PeerSimilarityDetail],
        iqr_multiplier: Decimal,
        prec_str: str,
    ) -> Tuple[List[PeerSimilarityDetail], List[PeerSimilarityDetail], OutlierSummary]:
        """Remove outliers using IQR method on intensity values.

        IQR   = Q3 - Q1
        lower = Q1 - k * IQR
        upper = Q3 + k * IQR
        """
        values = [p.intensity_value for p in peers]
        q1 = _percentile_decimal(values, Decimal("25"))
        q3 = _percentile_decimal(values, Decimal("75"))
        iqr = q3 - q1

        lower_fence = q1 - iqr_multiplier * iqr
        upper_fence = q3 + iqr_multiplier * iqr

        included: List[PeerSimilarityDetail] = []
        excluded: List[PeerSimilarityDetail] = []
        outlier_ids: List[str] = []

        for peer in peers:
            if peer.intensity_value < lower_fence or peer.intensity_value > upper_fence:
                peer_copy = peer.model_copy(update={"is_outlier": True})
                excluded.append(peer_copy)
                outlier_ids.append(peer.entity_id)
            else:
                included.append(peer)

        summary = OutlierSummary(
            method="IQR",
            iqr_multiplier=iqr_multiplier,
            lower_fence=lower_fence.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            upper_fence=upper_fence.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            q1=q1.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            q3=q3.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            iqr=iqr.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            outliers_removed=len(excluded),
            outlier_entity_ids=outlier_ids,
        )

        return included, excluded, summary

    # ------------------------------------------------------------------
    # Internal: Statistics
    # ------------------------------------------------------------------

    def _compute_stats(self, values: List[Decimal], prec_str: str) -> PeerGroupStats:
        """Compute distribution statistics for the peer group."""
        if not values:
            return PeerGroupStats()

        n = len(values)
        sorted_vals = sorted(values)
        total = sum(values)
        mean = (total / Decimal(str(n))).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
        median = _median_decimal(values).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
        std = _std_deviation_decimal(values).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        return PeerGroupStats(
            count=n,
            mean=mean,
            median=median,
            std_dev=std,
            min_val=sorted_vals[0].quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            max_val=sorted_vals[-1].quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            p10=_percentile_decimal(sorted_vals, Decimal("10")).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            ),
            p25=_percentile_decimal(sorted_vals, Decimal("25")).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            ),
            p75=_percentile_decimal(sorted_vals, Decimal("75")).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            ),
            p90=_percentile_decimal(sorted_vals, Decimal("90")).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            ),
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_version(self) -> str:
        return self._version

# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "ClassificationSystem",
    "RevenueBand",
    "ValueChainPosition",
    "VerificationStatus",
    "ScopeCompleteness",
    # Input Models
    "SectorMapping",
    "PeerCandidate",
    "SimilarityWeights",
    "QualityWeights",
    "PeerGroupConfig",
    "PeerGroupInput",
    # Output Models
    "PeerQualityScore",
    "PeerSimilarityDetail",
    "OutlierSummary",
    "PeerGroupStats",
    "PeerGroup",
    "PeerGroupResult",
    # Engine
    "PeerGroupConstructionEngine",
]
