# -*- coding: utf-8 -*-
"""
Peer Group Setup Workflow
====================================

5-phase workflow for peer group identification, sector mapping, size banding,
geographic weighting, and validation within PACK-047 GHG Emissions Benchmark
Pack.

Phases:
    1. SectorMapping              -- Map the reporting entity to GICS, NACE,
                                     and ISIC sector classification systems,
                                     determine primary sector, sub-sector, and
                                     cross-referenced codes for peer matching.
    2. SizeBanding                -- Apply revenue-based size banding to
                                     partition the peer universe into comparable
                                     cohorts (micro, small, medium, large,
                                     mega-cap), ensuring like-for-like
                                     benchmarking across scale dimensions.
    3. GeographicWeighting        -- Weight candidate peers by geographic
                                     emission factor similarity, adjusting for
                                     grid-mix differences, regional policy
                                     exposure, and climate zone alignment.
    4. PeerScoring                -- Score each candidate peer on data quality
                                     (PCAF 1-5), recency (years since latest
                                     report), completeness (scope coverage),
                                     and relevance (sector/size/geography
                                     composite score).
    5. Validation                 -- Validate the final peer group: minimum
                                     peer count (>=5), outlier removal (IQR
                                     fence), statistical distribution checks,
                                     and final group summary statistics.

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    ESRS E1-6 (2024) - Benchmark comparison requirements
    CDP Climate Change C7 (2026) - Peer benchmarking guidance
    SBTi Corporate Manual v2.1 - Sectoral benchmark alignment
    TCFD Recommendations - Metrics and Targets peer comparison
    PCAF Global Standard (2022) - Data quality scoring
    GRI 305-4 (2016) - Intensity benchmarking
    IFRS S2 (2023) - Climate-related peer disclosures
    SEC Climate Disclosure Rules (2024) - Comparable metrics

Schedule: Annually or upon significant peer universe change
Estimated duration: 1-2 weeks depending on peer data availability

Author: GreenLang Team
Version: 47.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.utcnow().isoformat() + "Z"


def _new_uuid() -> str:
    """Return a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of JSON-serialisable data."""
    serialised = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class PeerSetupPhase(str, Enum):
    """Peer group setup workflow phases."""

    SECTOR_MAPPING = "sector_mapping"
    SIZE_BANDING = "size_banding"
    GEOGRAPHIC_WEIGHTING = "geographic_weighting"
    PEER_SCORING = "peer_scoring"
    VALIDATION = "validation"


class SectorSystem(str, Enum):
    """Sector classification system."""

    GICS = "gics"
    NACE = "nace"
    ISIC = "isic"


class GICSsector(str, Enum):
    """GICS sector classification."""

    ENERGY = "energy"
    MATERIALS = "materials"
    INDUSTRIALS = "industrials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    HEALTHCARE = "healthcare"
    FINANCIALS = "financials"
    INFORMATION_TECHNOLOGY = "information_technology"
    COMMUNICATION_SERVICES = "communication_services"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"


class SizeBand(str, Enum):
    """Revenue-based size banding."""

    MICRO = "micro"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    MEGA = "mega"


class GeographicRegion(str, Enum):
    """Geographic region for emission factor similarity."""

    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    LATIN_AMERICA = "latin_america"
    MIDDLE_EAST_AFRICA = "middle_east_africa"
    OCEANIA = "oceania"
    GLOBAL = "global"


class DataQualityLevel(str, Enum):
    """PCAF-aligned data quality level (1=best, 5=worst)."""

    LEVEL_1 = "level_1"
    LEVEL_2 = "level_2"
    LEVEL_3 = "level_3"
    LEVEL_4 = "level_4"
    LEVEL_5 = "level_5"


class PeerStatus(str, Enum):
    """Status of a peer candidate in the selection process."""

    CANDIDATE = "candidate"
    SELECTED = "selected"
    EXCLUDED_OUTLIER = "excluded_outlier"
    EXCLUDED_QUALITY = "excluded_quality"
    EXCLUDED_SIZE = "excluded_size"
    EXCLUDED_GEO = "excluded_geo"


class ValidationSeverity(str, Enum):
    """Severity level for validation findings."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# =============================================================================
# SECTOR CROSS-REFERENCE (Zero-Hallucination Reference Data)
# =============================================================================

GICS_TO_NACE: Dict[str, str] = {
    "energy": "B",
    "materials": "C20-C23",
    "industrials": "C24-C33",
    "consumer_discretionary": "G-I",
    "consumer_staples": "C10-C12",
    "healthcare": "Q",
    "financials": "K",
    "information_technology": "J",
    "communication_services": "J61",
    "utilities": "D",
    "real_estate": "L",
}

GICS_TO_ISIC: Dict[str, str] = {
    "energy": "05-09",
    "materials": "20-23",
    "industrials": "24-33",
    "consumer_discretionary": "45-47",
    "consumer_staples": "10-12",
    "healthcare": "86-88",
    "financials": "64-66",
    "information_technology": "58-63",
    "communication_services": "61",
    "utilities": "35",
    "real_estate": "68",
}

SIZE_BAND_THRESHOLDS_USD_M: Dict[str, Tuple[float, float]] = {
    "micro": (0.0, 50.0),
    "small": (50.0, 500.0),
    "medium": (500.0, 5000.0),
    "large": (5000.0, 50000.0),
    "mega": (50000.0, float("inf")),
}

REGION_GRID_INTENSITY_KG_CO2_KWH: Dict[str, float] = {
    "north_america": 0.42,
    "europe": 0.30,
    "asia_pacific": 0.58,
    "latin_america": 0.22,
    "middle_east_africa": 0.55,
    "oceania": 0.65,
    "global": 0.45,
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class SectorMapping(BaseModel):
    """Cross-referenced sector classification for an entity."""

    gics_sector: GICSsector = Field(...)
    gics_sub_industry: str = Field(default="")
    nace_code: str = Field(default="")
    isic_code: str = Field(default="")
    sector_description: str = Field(default="")
    cross_reference_confidence: float = Field(default=0.0, ge=0.0, le=100.0)


class PeerCandidate(BaseModel):
    """A candidate peer entity for benchmarking."""

    entity_id: str = Field(...)
    entity_name: str = Field(default="")
    gics_sector: GICSsector = Field(default=GICSsector.INDUSTRIALS)
    revenue_usd_m: float = Field(default=0.0, ge=0.0)
    size_band: SizeBand = Field(default=SizeBand.MEDIUM)
    region: GeographicRegion = Field(default=GeographicRegion.GLOBAL)
    emissions_tco2e: float = Field(default=0.0, ge=0.0)
    intensity_value: float = Field(default=0.0, ge=0.0)
    data_quality: DataQualityLevel = Field(default=DataQualityLevel.LEVEL_3)
    reporting_year: int = Field(default=2024, ge=2015, le=2030)
    scope_coverage: str = Field(default="scope_1_2")
    geographic_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    composite_score: float = Field(default=0.0, ge=0.0, le=100.0)
    status: PeerStatus = Field(default=PeerStatus.CANDIDATE)
    exclusion_reason: str = Field(default="")
    provenance_hash: str = Field(default="")


class ValidationFinding(BaseModel):
    """A finding from peer group validation."""

    finding_id: str = Field(default_factory=lambda: f"vf-{_new_uuid()[:8]}")
    severity: ValidationSeverity = Field(...)
    check_name: str = Field(default="")
    message: str = Field(default="")
    recommendation: str = Field(default="")


class PeerGroupStats(BaseModel):
    """Summary statistics of the final peer group."""

    peer_count: int = Field(default=0, ge=0)
    median_intensity: float = Field(default=0.0, ge=0.0)
    mean_intensity: float = Field(default=0.0, ge=0.0)
    std_dev_intensity: float = Field(default=0.0, ge=0.0)
    min_intensity: float = Field(default=0.0, ge=0.0)
    max_intensity: float = Field(default=0.0, ge=0.0)
    p25_intensity: float = Field(default=0.0, ge=0.0)
    p75_intensity: float = Field(default=0.0, ge=0.0)
    iqr: float = Field(default=0.0, ge=0.0)
    avg_data_quality_score: float = Field(default=0.0, ge=0.0, le=5.0)
    avg_reporting_recency_years: float = Field(default=0.0, ge=0.0)
    sector_concentration_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class PeerGroupSetupInput(BaseModel):
    """Input data model for PeerGroupSetupWorkflow."""

    organization_id: str = Field(..., min_length=1, description="Organisation identifier")
    organization_name: str = Field(default="", description="Organisation display name")
    gics_sector: GICSsector = Field(
        default=GICSsector.INDUSTRIALS,
        description="Primary GICS sector classification",
    )
    gics_sub_industry: str = Field(default="", description="GICS sub-industry code")
    revenue_usd_m: float = Field(
        default=0.0, ge=0.0,
        description="Organisation annual revenue in USD millions",
    )
    region: GeographicRegion = Field(
        default=GeographicRegion.GLOBAL,
        description="Primary operating region",
    )
    candidate_peers: List[PeerCandidate] = Field(
        default_factory=list,
        description="Pre-selected candidate peers for evaluation",
    )
    minimum_peer_count: int = Field(
        default=5, ge=3, le=50,
        description="Minimum number of peers required",
    )
    maximum_peer_count: int = Field(
        default=30, ge=5, le=100,
        description="Maximum number of peers to include",
    )
    size_band_tolerance: int = Field(
        default=1, ge=0, le=2,
        description="Number of adjacent size bands to include (0=exact, 1=+/-1, 2=+/-2)",
    )
    min_data_quality: DataQualityLevel = Field(
        default=DataQualityLevel.LEVEL_4,
        description="Minimum acceptable data quality level",
    )
    max_reporting_age_years: int = Field(
        default=3, ge=1, le=5,
        description="Maximum years since latest report",
    )
    outlier_iqr_multiplier: float = Field(
        default=1.5, ge=1.0, le=3.0,
        description="IQR multiplier for outlier fence",
    )
    current_year: int = Field(default=2025, ge=2020, le=2035)
    tenant_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class PeerGroupSetupResult(BaseModel):
    """Complete result from peer group setup workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="peer_group_setup")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_id: str = Field(default="")
    sector_mapping: Optional[SectorMapping] = Field(default=None)
    organization_size_band: SizeBand = Field(default=SizeBand.MEDIUM)
    all_candidates: List[PeerCandidate] = Field(default_factory=list)
    selected_peers: List[PeerCandidate] = Field(default_factory=list)
    excluded_peers: List[PeerCandidate] = Field(default_factory=list)
    validation_findings: List[ValidationFinding] = Field(default_factory=list)
    peer_group_stats: Optional[PeerGroupStats] = Field(default=None)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class PeerGroupSetupWorkflow:
    """
    5-phase workflow for peer group identification, sector mapping, size
    banding, geographic weighting, scoring, and validation.

    Identifies the entity's sector classifications, assigns size bands, weights
    peers by geographic emission factor similarity, scores each candidate on
    data quality and relevance, and validates the final peer group for
    statistical soundness.

    Zero-hallucination: sector cross-references use deterministic mapping
    tables; size bands use fixed revenue thresholds; geographic weights use
    published grid intensity factors; no LLM calls in scoring path; SHA-256
    provenance on every output.

    Attributes:
        workflow_id: Unique execution identifier.
        _phase_results: Ordered phase outputs.
        _sector_mapping: Cross-referenced sector classification.
        _org_size_band: Organisation size band.
        _candidates: All peer candidates with scores.
        _selected: Final selected peers.
        _excluded: Excluded peers with reasons.
        _findings: Validation findings.
        _stats: Peer group summary statistics.

    Example:
        >>> wf = PeerGroupSetupWorkflow()
        >>> inp = PeerGroupSetupInput(
        ...     organization_id="org-001",
        ...     gics_sector=GICSsector.INDUSTRIALS,
        ...     revenue_usd_m=2500.0,
        ...     candidate_peers=[...],
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_SEQUENCE: List[PeerSetupPhase] = [
        PeerSetupPhase.SECTOR_MAPPING,
        PeerSetupPhase.SIZE_BANDING,
        PeerSetupPhase.GEOGRAPHIC_WEIGHTING,
        PeerSetupPhase.PEER_SCORING,
        PeerSetupPhase.VALIDATION,
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize PeerGroupSetupWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._sector_mapping: Optional[SectorMapping] = None
        self._org_size_band: SizeBand = SizeBand.MEDIUM
        self._candidates: List[PeerCandidate] = []
        self._selected: List[PeerCandidate] = []
        self._excluded: List[PeerCandidate] = []
        self._findings: List[ValidationFinding] = []
        self._stats: Optional[PeerGroupStats] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, input_data: PeerGroupSetupInput) -> PeerGroupSetupResult:
        """
        Execute the 5-phase peer group setup workflow.

        Args:
            input_data: Organisation details and candidate peer list.

        Returns:
            PeerGroupSetupResult with selected peer group and statistics.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting peer group setup %s org=%s sector=%s",
            self.workflow_id, input_data.organization_id,
            input_data.gics_sector.value,
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        phase_methods = [
            self._phase_1_sector_mapping,
            self._phase_2_size_banding,
            self._phase_3_geographic_weighting,
            self._phase_4_peer_scoring,
            self._phase_5_validation,
        ]

        try:
            for idx, phase_fn in enumerate(phase_methods, start=1):
                phase_result = await self._run_phase(phase_fn, input_data, idx)
                self._phase_results.append(phase_result)
                if phase_result.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {idx} failed: {phase_result.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Peer group setup failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        result = PeerGroupSetupResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_id=input_data.organization_id,
            sector_mapping=self._sector_mapping,
            organization_size_band=self._org_size_band,
            all_candidates=self._candidates,
            selected_peers=self._selected,
            excluded_peers=self._excluded,
            validation_findings=self._findings,
            peer_group_stats=self._stats,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Peer group setup %s completed in %.2fs status=%s selected=%d excluded=%d",
            self.workflow_id, elapsed, overall_status.value,
            len(self._selected), len(self._excluded),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Sector Mapping
    # -------------------------------------------------------------------------

    async def _phase_1_sector_mapping(
        self, input_data: PeerGroupSetupInput,
    ) -> PhaseResult:
        """Map organisation to GICS, NACE, and ISIC sector classifications."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        gics = input_data.gics_sector
        gics_key = gics.value

        nace_code = GICS_TO_NACE.get(gics_key, "")
        isic_code = GICS_TO_ISIC.get(gics_key, "")

        if not nace_code:
            warnings.append(f"No NACE cross-reference for GICS sector {gics_key}")
        if not isic_code:
            warnings.append(f"No ISIC cross-reference for GICS sector {gics_key}")

        confidence = 100.0
        if not nace_code or not isic_code:
            confidence = 70.0
        if input_data.gics_sub_industry:
            confidence = min(confidence + 10.0, 100.0)

        self._sector_mapping = SectorMapping(
            gics_sector=gics,
            gics_sub_industry=input_data.gics_sub_industry,
            nace_code=nace_code,
            isic_code=isic_code,
            sector_description=f"GICS {gics_key} / NACE {nace_code} / ISIC {isic_code}",
            cross_reference_confidence=round(confidence, 2),
        )

        outputs["gics_sector"] = gics_key
        outputs["gics_sub_industry"] = input_data.gics_sub_industry
        outputs["nace_code"] = nace_code
        outputs["isic_code"] = isic_code
        outputs["cross_reference_confidence"] = confidence

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 1 SectorMapping: gics=%s nace=%s isic=%s confidence=%.1f%%",
            gics_key, nace_code, isic_code, confidence,
        )
        return PhaseResult(
            phase_name="sector_mapping", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Size Banding
    # -------------------------------------------------------------------------

    async def _phase_2_size_banding(
        self, input_data: PeerGroupSetupInput,
    ) -> PhaseResult:
        """Apply revenue-based size banding to organisation and peers."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Determine organisation size band
        self._org_size_band = self._classify_size_band(input_data.revenue_usd_m)

        # Determine allowed bands based on tolerance
        allowed_bands = self._get_allowed_bands(
            self._org_size_band, input_data.size_band_tolerance,
        )

        # Classify and filter candidates
        self._candidates = list(input_data.candidate_peers)
        within_band_count = 0
        excluded_size_count = 0

        for peer in self._candidates:
            peer.size_band = self._classify_size_band(peer.revenue_usd_m)
            if peer.size_band not in allowed_bands:
                peer.status = PeerStatus.EXCLUDED_SIZE
                peer.exclusion_reason = (
                    f"Size band {peer.size_band.value} outside tolerance "
                    f"of {self._org_size_band.value} +/- {input_data.size_band_tolerance}"
                )
                excluded_size_count += 1
            else:
                within_band_count += 1

        outputs["organization_size_band"] = self._org_size_band.value
        outputs["organization_revenue_usd_m"] = input_data.revenue_usd_m
        outputs["allowed_bands"] = [b.value for b in allowed_bands]
        outputs["candidates_within_band"] = within_band_count
        outputs["candidates_excluded_size"] = excluded_size_count
        outputs["total_candidates"] = len(self._candidates)

        if within_band_count < input_data.minimum_peer_count:
            warnings.append(
                f"Only {within_band_count} peers within size band tolerance; "
                f"minimum is {input_data.minimum_peer_count}"
            )

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 2 SizeBanding: org=%s within=%d excluded=%d",
            self._org_size_band.value, within_band_count, excluded_size_count,
        )
        return PhaseResult(
            phase_name="size_banding", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Geographic Weighting
    # -------------------------------------------------------------------------

    async def _phase_3_geographic_weighting(
        self, input_data: PeerGroupSetupInput,
    ) -> PhaseResult:
        """Weight candidate peers by geographic emission factor similarity."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        org_region = input_data.region
        org_grid_intensity = REGION_GRID_INTENSITY_KG_CO2_KWH.get(
            org_region.value, 0.45,
        )

        weighted_count = 0
        for peer in self._candidates:
            if peer.status != PeerStatus.CANDIDATE:
                continue

            peer_grid_intensity = REGION_GRID_INTENSITY_KG_CO2_KWH.get(
                peer.region.value, 0.45,
            )

            # Geographic similarity: 1.0 = identical, 0.0 = maximally different
            # Based on ratio of grid intensities (closer = higher weight)
            if org_grid_intensity > 0 and peer_grid_intensity > 0:
                ratio = min(org_grid_intensity, peer_grid_intensity) / max(
                    org_grid_intensity, peer_grid_intensity,
                )
                geo_weight = round(ratio, 4)
            else:
                geo_weight = 0.5

            # Same region bonus
            if peer.region == org_region:
                geo_weight = min(geo_weight + 0.2, 1.0)

            peer.geographic_weight = round(geo_weight, 4)
            weighted_count += 1

        # Compute weight statistics
        active_peers = [
            p for p in self._candidates if p.status == PeerStatus.CANDIDATE
        ]
        weights = [p.geographic_weight for p in active_peers]

        outputs["organization_region"] = org_region.value
        outputs["organization_grid_intensity_kg_co2_kwh"] = org_grid_intensity
        outputs["peers_weighted"] = weighted_count
        outputs["weight_min"] = round(min(weights), 4) if weights else 0.0
        outputs["weight_max"] = round(max(weights), 4) if weights else 0.0
        outputs["weight_mean"] = (
            round(sum(weights) / len(weights), 4) if weights else 0.0
        )

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 3 GeographicWeighting: %d peers weighted, mean=%.3f",
            weighted_count,
            outputs["weight_mean"],
        )
        return PhaseResult(
            phase_name="geographic_weighting", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Peer Scoring
    # -------------------------------------------------------------------------

    async def _phase_4_peer_scoring(
        self, input_data: PeerGroupSetupInput,
    ) -> PhaseResult:
        """Score each candidate peer on quality, recency, completeness, relevance."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        quality_map: Dict[DataQualityLevel, float] = {
            DataQualityLevel.LEVEL_1: 100.0,
            DataQualityLevel.LEVEL_2: 80.0,
            DataQualityLevel.LEVEL_3: 60.0,
            DataQualityLevel.LEVEL_4: 40.0,
            DataQualityLevel.LEVEL_5: 20.0,
        }

        min_quality_map: Dict[DataQualityLevel, int] = {
            DataQualityLevel.LEVEL_1: 1,
            DataQualityLevel.LEVEL_2: 2,
            DataQualityLevel.LEVEL_3: 3,
            DataQualityLevel.LEVEL_4: 4,
            DataQualityLevel.LEVEL_5: 5,
        }
        min_quality_num = min_quality_map.get(input_data.min_data_quality, 4)

        scored_count = 0
        excluded_quality_count = 0

        for peer in self._candidates:
            if peer.status != PeerStatus.CANDIDATE:
                continue

            # Check data quality threshold
            peer_quality_num = min_quality_map.get(peer.data_quality, 5)
            if peer_quality_num > min_quality_num:
                peer.status = PeerStatus.EXCLUDED_QUALITY
                peer.exclusion_reason = (
                    f"Data quality {peer.data_quality.value} below minimum "
                    f"{input_data.min_data_quality.value}"
                )
                excluded_quality_count += 1
                continue

            # Check reporting recency
            reporting_age = input_data.current_year - peer.reporting_year
            if reporting_age > input_data.max_reporting_age_years:
                peer.status = PeerStatus.EXCLUDED_QUALITY
                peer.exclusion_reason = (
                    f"Reporting year {peer.reporting_year} is {reporting_age} years old "
                    f"(max {input_data.max_reporting_age_years})"
                )
                excluded_quality_count += 1
                continue

            # Compute composite score (weighted: quality 30%, recency 20%,
            # completeness 20%, geographic 30%)
            quality_score = quality_map.get(peer.data_quality, 20.0)

            recency_score = max(
                0.0,
                100.0 - (reporting_age * (100.0 / input_data.max_reporting_age_years)),
            )

            completeness_score = 80.0
            if "scope_1_2_3" in peer.scope_coverage:
                completeness_score = 100.0
            elif "scope_1_2" in peer.scope_coverage:
                completeness_score = 80.0
            elif "scope_1" in peer.scope_coverage:
                completeness_score = 50.0

            geo_score = peer.geographic_weight * 100.0

            composite = (
                quality_score * 0.30
                + recency_score * 0.20
                + completeness_score * 0.20
                + geo_score * 0.30
            )

            peer.composite_score = round(min(composite, 100.0), 2)

            peer_data = {
                "entity_id": peer.entity_id,
                "composite_score": peer.composite_score,
                "quality": quality_score,
                "recency": recency_score,
                "completeness": completeness_score,
                "geo": geo_score,
            }
            peer.provenance_hash = _compute_hash(peer_data)
            scored_count += 1

        outputs["peers_scored"] = scored_count
        outputs["peers_excluded_quality"] = excluded_quality_count
        if scored_count > 0:
            scored_peers = [
                p for p in self._candidates if p.status == PeerStatus.CANDIDATE
            ]
            scores = [p.composite_score for p in scored_peers]
            outputs["score_min"] = round(min(scores), 2) if scores else 0.0
            outputs["score_max"] = round(max(scores), 2) if scores else 0.0
            outputs["score_mean"] = (
                round(sum(scores) / len(scores), 2) if scores else 0.0
            )
        else:
            outputs["score_min"] = 0.0
            outputs["score_max"] = 0.0
            outputs["score_mean"] = 0.0
            warnings.append("No peers passed quality and recency filters")

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 4 PeerScoring: %d scored, %d excluded (quality/recency)",
            scored_count, excluded_quality_count,
        )
        return PhaseResult(
            phase_name="peer_scoring", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Validation
    # -------------------------------------------------------------------------

    async def _phase_5_validation(
        self, input_data: PeerGroupSetupInput,
    ) -> PhaseResult:
        """Validate final peer group: outlier removal, count, distribution."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._findings = []

        # Get active candidates sorted by composite score descending
        active_peers = sorted(
            [p for p in self._candidates if p.status == PeerStatus.CANDIDATE],
            key=lambda p: p.composite_score,
            reverse=True,
        )

        # Outlier removal using IQR fence on intensity values
        intensities = [p.intensity_value for p in active_peers if p.intensity_value > 0]
        outlier_ids: set = set()

        if len(intensities) >= 4:
            sorted_vals = sorted(intensities)
            n = len(sorted_vals)
            q1 = sorted_vals[n // 4]
            q3 = sorted_vals[(3 * n) // 4]
            iqr = q3 - q1
            lower_fence = q1 - input_data.outlier_iqr_multiplier * iqr
            upper_fence = q3 + input_data.outlier_iqr_multiplier * iqr

            for peer in active_peers:
                if peer.intensity_value > 0:
                    if (peer.intensity_value < lower_fence
                            or peer.intensity_value > upper_fence):
                        peer.status = PeerStatus.EXCLUDED_OUTLIER
                        peer.exclusion_reason = (
                            f"Intensity {peer.intensity_value:.4f} outside IQR fence "
                            f"[{lower_fence:.4f}, {upper_fence:.4f}]"
                        )
                        outlier_ids.add(peer.entity_id)
                        self._findings.append(ValidationFinding(
                            severity=ValidationSeverity.INFO,
                            check_name="outlier_removal",
                            message=(
                                f"Peer {peer.entity_name or peer.entity_id} excluded: "
                                f"intensity={peer.intensity_value:.4f}"
                            ),
                            recommendation="Review if entity is comparable",
                        ))

        # Select final peers (up to maximum, by composite score)
        remaining = [
            p for p in active_peers if p.status == PeerStatus.CANDIDATE
        ]
        selected = remaining[:input_data.maximum_peer_count]
        for p in selected:
            p.status = PeerStatus.SELECTED

        self._selected = selected
        self._excluded = [
            p for p in self._candidates if p.status not in (
                PeerStatus.CANDIDATE, PeerStatus.SELECTED,
            )
        ]

        # Validate minimum count
        if len(self._selected) < input_data.minimum_peer_count:
            self._findings.append(ValidationFinding(
                severity=ValidationSeverity.ERROR,
                check_name="minimum_peer_count",
                message=(
                    f"Only {len(self._selected)} peers selected; "
                    f"minimum is {input_data.minimum_peer_count}"
                ),
                recommendation="Expand candidate pool or relax size/quality filters",
            ))

        # Compute peer group statistics
        self._stats = self._compute_peer_stats(self._selected)

        # Distribution normality check (coefficient of variation)
        if self._stats and self._stats.mean_intensity > 0:
            cv = self._stats.std_dev_intensity / self._stats.mean_intensity
            if cv > 1.5:
                self._findings.append(ValidationFinding(
                    severity=ValidationSeverity.WARNING,
                    check_name="distribution_spread",
                    message=(
                        f"High coefficient of variation ({cv:.2f}); "
                        f"peer group may be heterogeneous"
                    ),
                    recommendation="Consider tightening sector or size filters",
                ))
            else:
                self._findings.append(ValidationFinding(
                    severity=ValidationSeverity.INFO,
                    check_name="distribution_spread",
                    message=f"Coefficient of variation ({cv:.2f}) within acceptable range",
                ))

        error_count = sum(
            1 for f in self._findings if f.severity == ValidationSeverity.ERROR
        )
        warning_count = sum(
            1 for f in self._findings if f.severity == ValidationSeverity.WARNING
        )

        outputs["peers_selected"] = len(self._selected)
        outputs["peers_excluded_total"] = len(self._excluded)
        outputs["outliers_removed"] = len(outlier_ids)
        outputs["validation_errors"] = error_count
        outputs["validation_warnings"] = warning_count
        outputs["peer_group_valid"] = error_count == 0
        if self._stats:
            outputs["median_intensity"] = self._stats.median_intensity
            outputs["mean_intensity"] = self._stats.mean_intensity
            outputs["std_dev_intensity"] = self._stats.std_dev_intensity

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 5 Validation: %d selected, %d excluded, %d outliers, valid=%s",
            len(self._selected), len(self._excluded),
            len(outlier_ids), error_count == 0,
        )
        return PhaseResult(
            phase_name="validation", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase Execution Wrapper
    # -------------------------------------------------------------------------

    async def _run_phase(
        self, phase_fn: Any, input_data: PeerGroupSetupInput, phase_number: int,
    ) -> PhaseResult:
        """Execute a phase with exponential backoff retry."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await phase_fn(input_data)
            except Exception as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    delay = self.BASE_RETRY_DELAY_S * (2 ** (attempt - 1))
                    self.logger.warning(
                        "Phase %d attempt %d/%d failed: %s. Retrying in %.1fs",
                        phase_number, attempt, self.MAX_RETRIES, exc, delay,
                    )
                    import asyncio
                    await asyncio.sleep(delay)
        return PhaseResult(
            phase_name=f"phase_{phase_number}_failed",
            phase_number=phase_number,
            status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._phase_results = []
        self._sector_mapping = None
        self._org_size_band = SizeBand.MEDIUM
        self._candidates = []
        self._selected = []
        self._excluded = []
        self._findings = []
        self._stats = None

    def _classify_size_band(self, revenue_usd_m: float) -> SizeBand:
        """Classify revenue into a size band."""
        for band_name, (lower, upper) in SIZE_BAND_THRESHOLDS_USD_M.items():
            if lower <= revenue_usd_m < upper:
                return SizeBand(band_name)
        return SizeBand.MEGA

    def _get_allowed_bands(
        self, org_band: SizeBand, tolerance: int,
    ) -> List[SizeBand]:
        """Get allowed size bands based on tolerance."""
        band_order = [
            SizeBand.MICRO, SizeBand.SMALL, SizeBand.MEDIUM,
            SizeBand.LARGE, SizeBand.MEGA,
        ]
        try:
            idx = band_order.index(org_band)
        except ValueError:
            return band_order

        start = max(0, idx - tolerance)
        end = min(len(band_order), idx + tolerance + 1)
        return band_order[start:end]

    def _compute_peer_stats(self, peers: List[PeerCandidate]) -> PeerGroupStats:
        """Compute summary statistics for the peer group."""
        if not peers:
            return PeerGroupStats(provenance_hash=_compute_hash({"empty": True}))

        intensities = sorted(
            [p.intensity_value for p in peers if p.intensity_value > 0],
        )
        n = len(intensities)

        if n == 0:
            return PeerGroupStats(
                peer_count=len(peers),
                provenance_hash=_compute_hash({"count": len(peers)}),
            )

        mean_val = sum(intensities) / n
        variance = sum((v - mean_val) ** 2 for v in intensities) / max(n, 1)
        std_dev = math.sqrt(variance)

        p25 = intensities[n // 4] if n >= 4 else intensities[0]
        p75 = intensities[(3 * n) // 4] if n >= 4 else intensities[-1]
        median = intensities[n // 2]

        quality_map: Dict[DataQualityLevel, float] = {
            DataQualityLevel.LEVEL_1: 1.0,
            DataQualityLevel.LEVEL_2: 2.0,
            DataQualityLevel.LEVEL_3: 3.0,
            DataQualityLevel.LEVEL_4: 4.0,
            DataQualityLevel.LEVEL_5: 5.0,
        }
        avg_quality = sum(
            quality_map.get(p.data_quality, 3.0) for p in peers
        ) / len(peers)

        current_year = datetime.utcnow().year
        avg_recency = sum(
            current_year - p.reporting_year for p in peers
        ) / len(peers)

        # Sector concentration: % of peers in same GICS sector as mode
        sector_counts: Dict[str, int] = {}
        for p in peers:
            s = p.gics_sector.value
            sector_counts[s] = sector_counts.get(s, 0) + 1
        max_sector_count = max(sector_counts.values()) if sector_counts else 0
        sector_concentration = (max_sector_count / len(peers)) * 100.0

        stats_data = {
            "count": n, "mean": round(mean_val, 6),
            "std": round(std_dev, 6), "median": round(median, 6),
        }

        return PeerGroupStats(
            peer_count=len(peers),
            median_intensity=round(median, 6),
            mean_intensity=round(mean_val, 6),
            std_dev_intensity=round(std_dev, 6),
            min_intensity=round(intensities[0], 6),
            max_intensity=round(intensities[-1], 6),
            p25_intensity=round(p25, 6),
            p75_intensity=round(p75, 6),
            iqr=round(p75 - p25, 6),
            avg_data_quality_score=round(avg_quality, 2),
            avg_reporting_recency_years=round(avg_recency, 2),
            sector_concentration_pct=round(sector_concentration, 2),
            provenance_hash=_compute_hash(stats_data),
        )

    def _compute_provenance(self, result: PeerGroupSetupResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += (
            f"|{result.workflow_id}|{result.organization_id}"
            f"|{len(result.selected_peers)}"
        )
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
