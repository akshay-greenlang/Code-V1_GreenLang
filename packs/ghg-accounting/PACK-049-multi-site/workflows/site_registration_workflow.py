# -*- coding: utf-8 -*-
"""
Site Registration Workflow
====================================

5-phase workflow for multi-site GHG inventory site registration covering
site discovery, classification, characteristics capture, boundary assignment,
and activation within PACK-049 GHG Multi-Site Management Pack.

Phases:
    1. SiteDiscovery            -- Discover sites from input data sources,
                                   generating a candidate site list with
                                   preliminary metadata (name, address,
                                   country, legal entity).
    2. Classification           -- Classify each candidate site by facility
                                   type, sector (GICS/NACE), geography,
                                   and business unit assignment.
    3. CharacteristicsCapture   -- Capture operational characteristics
                                   including floor area (sqm), headcount,
                                   operating hours (h/yr), grid region,
                                   primary fuel, and production output.
    4. BoundaryAssignment       -- Assign each site to the organisational
                                   boundary with consolidation approach
                                   (equity share / financial control /
                                   operational control) and ownership %.
    5. Activation               -- Validate completeness, activate sites
                                   in the registry, and generate the
                                   final registration result with SHA-256
                                   provenance hash.

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    GHG Protocol Corporate Standard (Ch. 3) -- Organisational Boundaries
    GHG Protocol Corporate Standard (Ch. 4) -- Operational Boundaries
    ISO 14064-1:2018 -- Organisational level GHG quantification
    CSRD / ESRS E1 -- Climate change disclosure (scope & boundary)

Author: GreenLang Team
Version: 49.0.0
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


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


class RegistrationPhase(str, Enum):
    """Site registration workflow phases."""
    SITE_DISCOVERY = "site_discovery"
    CLASSIFICATION = "classification"
    CHARACTERISTICS_CAPTURE = "characteristics_capture"
    BOUNDARY_ASSIGNMENT = "boundary_assignment"
    ACTIVATION = "activation"


class FacilityType(str, Enum):
    """GHG inventory facility type classification."""
    MANUFACTURING = "manufacturing"
    OFFICE = "office"
    WAREHOUSE = "warehouse"
    DATA_CENTER = "data_center"
    RETAIL = "retail"
    LABORATORY = "laboratory"
    REFINERY = "refinery"
    POWER_PLANT = "power_plant"
    TRANSPORT_HUB = "transport_hub"
    MIXED_USE = "mixed_use"
    OTHER = "other"


class SectorClassification(str, Enum):
    """Industry sector classification scheme."""
    ENERGY = "energy"
    MATERIALS = "materials"
    INDUSTRIALS = "industrials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    HEALTH_CARE = "health_care"
    FINANCIALS = "financials"
    INFORMATION_TECHNOLOGY = "information_technology"
    COMMUNICATION_SERVICES = "communication_services"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"


class ConsolidationApproach(str, Enum):
    """GHG Protocol organisational boundary consolidation approach."""
    EQUITY_SHARE = "equity_share"
    FINANCIAL_CONTROL = "financial_control"
    OPERATIONAL_CONTROL = "operational_control"


class SiteStatus(str, Enum):
    """Site lifecycle status."""
    CANDIDATE = "candidate"
    CLASSIFIED = "classified"
    CHARACTERIZED = "characterized"
    ASSIGNED = "assigned"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DECOMMISSIONED = "decommissioned"


class GeographicRegion(str, Enum):
    """High-level geographic region."""
    NORTH_AMERICA = "north_america"
    LATIN_AMERICA = "latin_america"
    EUROPE = "europe"
    MIDDLE_EAST_AFRICA = "middle_east_africa"
    ASIA_PACIFIC = "asia_pacific"
    OCEANIA = "oceania"


# =============================================================================
# REFERENCE DATA
# =============================================================================

FACILITY_TYPE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "manufacturing": {"default_hours_yr": 6000, "intensity_driver": "production_output"},
    "office": {"default_hours_yr": 2500, "intensity_driver": "headcount"},
    "warehouse": {"default_hours_yr": 4000, "intensity_driver": "floor_area_sqm"},
    "data_center": {"default_hours_yr": 8760, "intensity_driver": "it_load_kw"},
    "retail": {"default_hours_yr": 3500, "intensity_driver": "revenue"},
    "laboratory": {"default_hours_yr": 3000, "intensity_driver": "headcount"},
    "refinery": {"default_hours_yr": 8400, "intensity_driver": "throughput"},
    "power_plant": {"default_hours_yr": 7500, "intensity_driver": "generation_mwh"},
    "transport_hub": {"default_hours_yr": 6000, "intensity_driver": "throughput"},
    "mixed_use": {"default_hours_yr": 3500, "intensity_driver": "floor_area_sqm"},
    "other": {"default_hours_yr": 2500, "intensity_driver": "headcount"},
}

REGION_MAPPING: Dict[str, str] = {
    "US": "north_america", "CA": "north_america", "MX": "latin_america",
    "BR": "latin_america", "AR": "latin_america", "CL": "latin_america",
    "GB": "europe", "DE": "europe", "FR": "europe", "IT": "europe",
    "ES": "europe", "NL": "europe", "SE": "europe", "NO": "europe",
    "DK": "europe", "FI": "europe", "PL": "europe", "CH": "europe",
    "AT": "europe", "BE": "europe", "IE": "europe", "PT": "europe",
    "ZA": "middle_east_africa", "AE": "middle_east_africa",
    "SA": "middle_east_africa", "NG": "middle_east_africa",
    "CN": "asia_pacific", "JP": "asia_pacific", "KR": "asia_pacific",
    "IN": "asia_pacific", "SG": "asia_pacific", "TH": "asia_pacific",
    "AU": "oceania", "NZ": "oceania",
}

MINIMUM_COMPLETENESS_THRESHOLD = Decimal("0.80")


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class CandidateSite(BaseModel):
    """A discovered candidate site before classification."""
    site_id: str = Field(default_factory=_new_uuid, description="Unique site ID")
    site_name: str = Field(..., description="Human-readable site name")
    legal_entity: str = Field("", description="Legal entity name")
    address_line_1: str = Field("", description="Street address")
    city: str = Field("", description="City")
    state_province: str = Field("", description="State or province")
    country_code: str = Field("", description="ISO 3166-1 alpha-2 country code")
    postal_code: str = Field("", description="Postal / ZIP code")
    latitude: Optional[Decimal] = Field(None, description="Latitude")
    longitude: Optional[Decimal] = Field(None, description="Longitude")
    source: str = Field("manual", description="Discovery source")
    discovered_at: str = Field(default_factory=lambda: _utcnow().isoformat())


class ClassifiedSite(BaseModel):
    """A site with classification attributes assigned."""
    site_id: str = Field(..., description="Unique site ID")
    site_name: str = Field(..., description="Site name")
    legal_entity: str = Field("", description="Legal entity")
    country_code: str = Field("", description="ISO country code")
    facility_type: FacilityType = Field(FacilityType.OTHER, description="Facility type")
    sector: SectorClassification = Field(
        SectorClassification.INDUSTRIALS, description="Industry sector"
    )
    geographic_region: GeographicRegion = Field(
        GeographicRegion.EUROPE, description="Geographic region"
    )
    business_unit: str = Field("", description="Business unit assignment")
    classification_confidence: Decimal = Field(
        Decimal("1.00"), description="Classification confidence 0-1"
    )


class SiteCharacteristics(BaseModel):
    """Operational characteristics for a site."""
    site_id: str = Field(..., description="Site ID")
    floor_area_sqm: Optional[Decimal] = Field(None, ge=0, description="Floor area m2")
    headcount: Optional[int] = Field(None, ge=0, description="Employee headcount")
    operating_hours_yr: Optional[int] = Field(None, ge=0, description="Operating hours per year")
    grid_region: str = Field("", description="Electricity grid region code")
    primary_fuel: str = Field("", description="Primary fuel type")
    production_output: Optional[Decimal] = Field(None, ge=0, description="Annual production output")
    production_unit: str = Field("", description="Unit of production output")
    revenue_local: Optional[Decimal] = Field(None, ge=0, description="Revenue in local currency")
    currency_code: str = Field("USD", description="Currency code")
    it_load_kw: Optional[Decimal] = Field(None, ge=0, description="IT load for data centres")
    has_onsite_generation: bool = Field(False, description="On-site generation present")
    has_fleet_vehicles: bool = Field(False, description="Fleet vehicles present")
    data_completeness_pct: Decimal = Field(Decimal("0"), description="Data completeness 0-100")


class BoundaryAssignment(BaseModel):
    """Organisational boundary assignment for a site."""
    site_id: str = Field(..., description="Site ID")
    consolidation_approach: ConsolidationApproach = Field(
        ..., description="Consolidation approach"
    )
    ownership_pct: Decimal = Field(
        Decimal("100.00"), ge=Decimal("0"), le=Decimal("100"),
        description="Ownership percentage"
    )
    is_joint_venture: bool = Field(False, description="Joint venture flag")
    parent_entity: str = Field("", description="Parent entity name")
    control_type: str = Field("", description="Control type detail")
    inclusion_rationale: str = Field("", description="Reason for inclusion")
    exclusion_reason: str = Field("", description="Reason if excluded")
    is_included: bool = Field(True, description="Included in boundary")
    effective_date: str = Field("", description="Effective date ISO")


class RegisteredSite(BaseModel):
    """A fully registered and activated site."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    site_id: str = Field(..., description="Unique site ID")
    site_name: str = Field(..., description="Site name")
    legal_entity: str = Field("", description="Legal entity")
    country_code: str = Field("", description="Country code")
    facility_type: FacilityType = Field(FacilityType.OTHER)
    sector: SectorClassification = Field(SectorClassification.INDUSTRIALS)
    geographic_region: GeographicRegion = Field(GeographicRegion.EUROPE)
    business_unit: str = Field("")
    floor_area_sqm: Optional[Decimal] = Field(None)
    headcount: Optional[int] = Field(None)
    operating_hours_yr: Optional[int] = Field(None)
    grid_region: str = Field("")
    primary_fuel: str = Field("")
    consolidation_approach: ConsolidationApproach = Field(
        ConsolidationApproach.OPERATIONAL_CONTROL
    )
    ownership_pct: Decimal = Field(Decimal("100.00"))
    is_included: bool = Field(True)
    status: SiteStatus = Field(SiteStatus.ACTIVE)
    activated_at: str = Field("")
    provenance_hash: str = Field("")


class SiteRegistrationInput(BaseModel):
    """Input for the site registration workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    organisation_id: str = Field(..., description="Organisation identifier")
    organisation_name: str = Field("", description="Organisation display name")
    reporting_year: int = Field(..., description="Reporting year")
    default_consolidation: ConsolidationApproach = Field(
        ConsolidationApproach.OPERATIONAL_CONTROL,
        description="Default consolidation approach",
    )
    candidate_sites: List[Dict[str, Any]] = Field(
        default_factory=list, description="Raw candidate site data"
    )
    existing_sites: List[Dict[str, Any]] = Field(
        default_factory=list, description="Already-registered sites for delta"
    )
    auto_classify: bool = Field(True, description="Enable auto-classification")
    completeness_threshold: Decimal = Field(
        MINIMUM_COMPLETENESS_THRESHOLD,
        description="Minimum completeness for activation",
    )
    skip_phases: List[str] = Field(
        default_factory=list, description="Phase names to skip"
    )


class SiteRegistrationResult(BaseModel):
    """Output from the site registration workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    workflow_id: str = Field(default_factory=_new_uuid, description="Workflow run ID")
    organisation_id: str = Field("", description="Organisation ID")
    reporting_year: int = Field(0, description="Reporting year")
    status: WorkflowStatus = Field(WorkflowStatus.PENDING)
    phase_results: List[PhaseResult] = Field(default_factory=list)
    registered_sites: List[RegisteredSite] = Field(default_factory=list)
    total_candidates: int = Field(0, description="Number of candidate sites")
    total_activated: int = Field(0, description="Number of activated sites")
    total_excluded: int = Field(0, description="Number of excluded sites")
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    duration_seconds: float = Field(0.0)
    provenance_hash: str = Field("")
    started_at: str = Field("")
    completed_at: str = Field("")


# =============================================================================
# WORKFLOW CLASS
# =============================================================================


class SiteRegistrationWorkflow:
    """
    5-phase site registration workflow for multi-site GHG inventories.

    Discovers candidate sites, classifies them by facility type and sector,
    captures operational characteristics, assigns organisational boundary
    parameters (consolidation approach, ownership %), and activates sites
    in the registry with full SHA-256 provenance.

    Attributes:
        config: Optional workflow configuration overrides.
        _candidates: Internal list of candidate sites.
        _classified: Internal list of classified sites.
        _characteristics: Internal map site_id -> characteristics.
        _assignments: Internal map site_id -> boundary assignment.

    Example:
        >>> wf = SiteRegistrationWorkflow()
        >>> inp = SiteRegistrationInput(
        ...     organisation_id="ORG-001",
        ...     reporting_year=2025,
        ...     candidate_sites=[{"site_name": "Plant A", "country_code": "DE"}],
        ... )
        >>> result = wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_ORDER: List[RegistrationPhase] = [
        RegistrationPhase.SITE_DISCOVERY,
        RegistrationPhase.CLASSIFICATION,
        RegistrationPhase.CHARACTERISTICS_CAPTURE,
        RegistrationPhase.BOUNDARY_ASSIGNMENT,
        RegistrationPhase.ACTIVATION,
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SiteRegistrationWorkflow."""
        self.config = config or {}
        self._candidates: List[CandidateSite] = []
        self._classified: List[ClassifiedSite] = []
        self._characteristics: Dict[str, SiteCharacteristics] = {}
        self._assignments: Dict[str, BoundaryAssignment] = {}

    # -----------------------------------------------------------------
    # PUBLIC ENTRY POINT
    # -----------------------------------------------------------------

    def execute(self, input_data: SiteRegistrationInput) -> SiteRegistrationResult:
        """
        Execute the full 5-phase site registration workflow.

        Args:
            input_data: Validated workflow input.

        Returns:
            SiteRegistrationResult with registered sites and provenance.
        """
        start = _utcnow()
        result = SiteRegistrationResult(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            status=WorkflowStatus.RUNNING,
            started_at=start.isoformat(),
        )

        phase_methods = {
            RegistrationPhase.SITE_DISCOVERY: self._phase_site_discovery,
            RegistrationPhase.CLASSIFICATION: self._phase_classification,
            RegistrationPhase.CHARACTERISTICS_CAPTURE: self._phase_characteristics_capture,
            RegistrationPhase.BOUNDARY_ASSIGNMENT: self._phase_boundary_assignment,
            RegistrationPhase.ACTIVATION: self._phase_activation,
        }

        for idx, phase in enumerate(self.PHASE_ORDER, 1):
            if phase.value in input_data.skip_phases:
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value,
                    phase_number=idx,
                    status=PhaseStatus.SKIPPED,
                ))
                continue

            phase_start = _utcnow()
            try:
                phase_out = phase_methods[phase](input_data, result)
                elapsed = (_utcnow() - phase_start).total_seconds()
                ph_hash = _compute_hash(str(phase_out))
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value,
                    phase_number=idx,
                    status=PhaseStatus.COMPLETED,
                    duration_seconds=elapsed,
                    outputs=phase_out,
                    provenance_hash=ph_hash,
                ))
            except Exception as exc:
                elapsed = (_utcnow() - phase_start).total_seconds()
                logger.error("Phase %s failed: %s", phase.value, exc, exc_info=True)
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value,
                    phase_number=idx,
                    status=PhaseStatus.FAILED,
                    duration_seconds=elapsed,
                    errors=[str(exc)],
                ))
                result.status = WorkflowStatus.FAILED
                result.errors.append(f"Phase {phase.value} failed: {exc}")
                break

        if result.status != WorkflowStatus.FAILED:
            result.status = WorkflowStatus.COMPLETED

        end = _utcnow()
        result.completed_at = end.isoformat()
        result.duration_seconds = (end - start).total_seconds()
        result.provenance_hash = _compute_hash(
            f"{result.workflow_id}|{result.organisation_id}|"
            f"{result.total_activated}|{result.completed_at}"
        )
        return result

    # -----------------------------------------------------------------
    # PHASE 1 -- SITE DISCOVERY
    # -----------------------------------------------------------------

    def _phase_site_discovery(
        self,
        input_data: SiteRegistrationInput,
        result: SiteRegistrationResult,
    ) -> Dict[str, Any]:
        """
        Discover candidate sites from input data.

        Parses raw candidate dicts into CandidateSite models, deduplicates
        by name+country, and logs discovery statistics.
        """
        logger.info("Phase 1 -- Site Discovery: %d raw candidates", len(input_data.candidate_sites))
        seen: Dict[str, bool] = {}
        candidates: List[CandidateSite] = []

        for raw in input_data.candidate_sites:
            name = raw.get("site_name", "").strip()
            country = raw.get("country_code", "").strip().upper()
            dedup_key = f"{name}|{country}".lower()

            if not name:
                result.warnings.append(f"Skipped candidate with empty name: {raw}")
                continue

            if dedup_key in seen:
                result.warnings.append(f"Duplicate site skipped: {name} ({country})")
                continue
            seen[dedup_key] = True

            candidate = CandidateSite(
                site_name=name,
                legal_entity=raw.get("legal_entity", ""),
                address_line_1=raw.get("address_line_1", raw.get("address", "")),
                city=raw.get("city", ""),
                state_province=raw.get("state_province", raw.get("state", "")),
                country_code=country,
                postal_code=raw.get("postal_code", ""),
                latitude=Decimal(str(raw["latitude"])) if raw.get("latitude") else None,
                longitude=Decimal(str(raw["longitude"])) if raw.get("longitude") else None,
                source=raw.get("source", "manual"),
            )
            candidates.append(candidate)

        # Merge with existing sites to detect new vs. returning
        existing_keys = set()
        for ex in input_data.existing_sites:
            ek = f"{ex.get('site_name', '')}|{ex.get('country_code', '')}".lower()
            existing_keys.add(ek)

        new_count = 0
        returning_count = 0
        for c in candidates:
            ck = f"{c.site_name}|{c.country_code}".lower()
            if ck in existing_keys:
                returning_count += 1
            else:
                new_count += 1

        self._candidates = candidates
        result.total_candidates = len(candidates)

        logger.info(
            "Discovery complete: %d candidates (%d new, %d returning)",
            len(candidates), new_count, returning_count,
        )
        return {
            "candidates_discovered": len(candidates),
            "new_sites": new_count,
            "returning_sites": returning_count,
            "duplicates_removed": len(input_data.candidate_sites) - len(candidates),
        }

    # -----------------------------------------------------------------
    # PHASE 2 -- CLASSIFICATION
    # -----------------------------------------------------------------

    def _phase_classification(
        self,
        input_data: SiteRegistrationInput,
        result: SiteRegistrationResult,
    ) -> Dict[str, Any]:
        """
        Classify each candidate site by facility type, sector, geography,
        and business unit.
        """
        logger.info("Phase 2 -- Classification: %d candidates", len(self._candidates))
        classified: List[ClassifiedSite] = []
        unclassified_count = 0

        for cand in self._candidates:
            facility_type = self._infer_facility_type(cand)
            sector = self._infer_sector(cand, facility_type)
            region = self._resolve_region(cand.country_code)
            bu = self._resolve_business_unit(cand)

            confidence = Decimal("1.00")
            if input_data.auto_classify:
                confidence = self._compute_classification_confidence(cand, facility_type)

            if confidence < Decimal("0.30"):
                unclassified_count += 1
                result.warnings.append(
                    f"Low classification confidence for {cand.site_name}: {confidence}"
                )

            cs = ClassifiedSite(
                site_id=cand.site_id,
                site_name=cand.site_name,
                legal_entity=cand.legal_entity,
                country_code=cand.country_code,
                facility_type=facility_type,
                sector=sector,
                geographic_region=region,
                business_unit=bu,
                classification_confidence=confidence,
            )
            classified.append(cs)

        self._classified = classified

        type_dist: Dict[str, int] = {}
        for c in classified:
            key = c.facility_type.value
            type_dist[key] = type_dist.get(key, 0) + 1

        region_dist: Dict[str, int] = {}
        for c in classified:
            key = c.geographic_region.value
            region_dist[key] = region_dist.get(key, 0) + 1

        logger.info("Classification complete: %d classified, %d low-confidence",
                     len(classified), unclassified_count)
        return {
            "classified_count": len(classified),
            "unclassified_count": unclassified_count,
            "facility_type_distribution": type_dist,
            "region_distribution": region_dist,
        }

    def _infer_facility_type(self, candidate: CandidateSite) -> FacilityType:
        """Infer facility type from site name keywords (deterministic)."""
        name_lower = candidate.site_name.lower()
        keyword_map = {
            "plant": FacilityType.MANUFACTURING,
            "factory": FacilityType.MANUFACTURING,
            "mill": FacilityType.MANUFACTURING,
            "office": FacilityType.OFFICE,
            "hq": FacilityType.OFFICE,
            "headquarters": FacilityType.OFFICE,
            "warehouse": FacilityType.WAREHOUSE,
            "distribution": FacilityType.WAREHOUSE,
            "dc": FacilityType.DATA_CENTER,
            "data center": FacilityType.DATA_CENTER,
            "data centre": FacilityType.DATA_CENTER,
            "store": FacilityType.RETAIL,
            "retail": FacilityType.RETAIL,
            "shop": FacilityType.RETAIL,
            "lab": FacilityType.LABORATORY,
            "refinery": FacilityType.REFINERY,
            "power": FacilityType.POWER_PLANT,
            "transport": FacilityType.TRANSPORT_HUB,
            "depot": FacilityType.TRANSPORT_HUB,
        }
        for kw, ft in keyword_map.items():
            if kw in name_lower:
                return ft
        return FacilityType.OTHER

    def _infer_sector(
        self, candidate: CandidateSite, facility_type: FacilityType
    ) -> SectorClassification:
        """Infer sector from facility type (deterministic mapping)."""
        mapping: Dict[FacilityType, SectorClassification] = {
            FacilityType.MANUFACTURING: SectorClassification.INDUSTRIALS,
            FacilityType.OFFICE: SectorClassification.FINANCIALS,
            FacilityType.WAREHOUSE: SectorClassification.INDUSTRIALS,
            FacilityType.DATA_CENTER: SectorClassification.INFORMATION_TECHNOLOGY,
            FacilityType.RETAIL: SectorClassification.CONSUMER_DISCRETIONARY,
            FacilityType.LABORATORY: SectorClassification.HEALTH_CARE,
            FacilityType.REFINERY: SectorClassification.ENERGY,
            FacilityType.POWER_PLANT: SectorClassification.UTILITIES,
            FacilityType.TRANSPORT_HUB: SectorClassification.INDUSTRIALS,
            FacilityType.MIXED_USE: SectorClassification.REAL_ESTATE,
            FacilityType.OTHER: SectorClassification.INDUSTRIALS,
        }
        return mapping.get(facility_type, SectorClassification.INDUSTRIALS)

    def _resolve_region(self, country_code: str) -> GeographicRegion:
        """Resolve geographic region from ISO country code."""
        region_str = REGION_MAPPING.get(country_code.upper(), "europe")
        try:
            return GeographicRegion(region_str)
        except ValueError:
            return GeographicRegion.EUROPE

    def _resolve_business_unit(self, candidate: CandidateSite) -> str:
        """Resolve business unit from legal entity or configuration."""
        bu_map = self.config.get("business_unit_map", {})
        if candidate.legal_entity in bu_map:
            return bu_map[candidate.legal_entity]
        return candidate.legal_entity or "Default"

    def _compute_classification_confidence(
        self, candidate: CandidateSite, facility_type: FacilityType
    ) -> Decimal:
        """Compute deterministic classification confidence score."""
        score = Decimal("0.50")
        if candidate.site_name and len(candidate.site_name) > 3:
            score += Decimal("0.15")
        if candidate.country_code:
            score += Decimal("0.10")
        if candidate.legal_entity:
            score += Decimal("0.10")
        if facility_type != FacilityType.OTHER:
            score += Decimal("0.15")
        return min(score, Decimal("1.00"))

    # -----------------------------------------------------------------
    # PHASE 3 -- CHARACTERISTICS CAPTURE
    # -----------------------------------------------------------------

    def _phase_characteristics_capture(
        self,
        input_data: SiteRegistrationInput,
        result: SiteRegistrationResult,
    ) -> Dict[str, Any]:
        """
        Capture operational characteristics for each classified site.

        Populates floor area, headcount, operating hours, grid region,
        primary fuel, and production output. Applies facility-type
        defaults where data is missing.
        """
        logger.info("Phase 3 -- Characteristics Capture: %d sites", len(self._classified))
        characteristics: Dict[str, SiteCharacteristics] = {}
        raw_lookup: Dict[str, Dict[str, Any]] = {}
        for raw in input_data.candidate_sites:
            key = raw.get("site_name", "").strip().lower()
            raw_lookup[key] = raw

        complete_count = 0
        partial_count = 0

        for cs in self._classified:
            raw = raw_lookup.get(cs.site_name.lower(), {})
            defaults = FACILITY_TYPE_DEFAULTS.get(cs.facility_type.value, {})

            floor_area = self._parse_decimal(raw.get("floor_area_sqm"))
            headcount = raw.get("headcount")
            if isinstance(headcount, str):
                headcount = int(headcount) if headcount.isdigit() else None
            operating_hours = raw.get("operating_hours_yr")
            if operating_hours is None:
                operating_hours = defaults.get("default_hours_yr")
            grid_region = raw.get("grid_region", "")
            primary_fuel = raw.get("primary_fuel", "")
            production_output = self._parse_decimal(raw.get("production_output"))
            production_unit = raw.get("production_unit", "")
            revenue = self._parse_decimal(raw.get("revenue_local"))
            currency = raw.get("currency_code", "USD")
            it_load = self._parse_decimal(raw.get("it_load_kw"))
            has_gen = bool(raw.get("has_onsite_generation", False))
            has_fleet = bool(raw.get("has_fleet_vehicles", False))

            # Compute data completeness
            fields = [floor_area, headcount, operating_hours, grid_region, primary_fuel]
            filled = sum(1 for f in fields if f is not None and f != "" and f != 0)
            completeness = Decimal(str(filled)) / Decimal("5") * Decimal("100")
            completeness = completeness.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            char = SiteCharacteristics(
                site_id=cs.site_id,
                floor_area_sqm=floor_area,
                headcount=headcount,
                operating_hours_yr=operating_hours,
                grid_region=grid_region,
                primary_fuel=primary_fuel,
                production_output=production_output,
                production_unit=production_unit,
                revenue_local=revenue,
                currency_code=currency,
                it_load_kw=it_load,
                has_onsite_generation=has_gen,
                has_fleet_vehicles=has_fleet,
                data_completeness_pct=completeness,
            )
            characteristics[cs.site_id] = char

            if completeness >= Decimal("80"):
                complete_count += 1
            else:
                partial_count += 1

        self._characteristics = characteristics

        avg_completeness = Decimal("0")
        if characteristics:
            total_comp = sum(c.data_completeness_pct for c in characteristics.values())
            avg_completeness = (total_comp / Decimal(str(len(characteristics)))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        logger.info(
            "Characteristics captured: %d complete, %d partial, avg %.1f%%",
            complete_count, partial_count, float(avg_completeness),
        )
        return {
            "sites_characterized": len(characteristics),
            "complete_count": complete_count,
            "partial_count": partial_count,
            "average_completeness_pct": float(avg_completeness),
        }

    def _parse_decimal(self, value: Any) -> Optional[Decimal]:
        """Safely parse a value to Decimal or return None."""
        if value is None:
            return None
        try:
            return Decimal(str(value))
        except Exception:
            return None

    # -----------------------------------------------------------------
    # PHASE 4 -- BOUNDARY ASSIGNMENT
    # -----------------------------------------------------------------

    def _phase_boundary_assignment(
        self,
        input_data: SiteRegistrationInput,
        result: SiteRegistrationResult,
    ) -> Dict[str, Any]:
        """
        Assign each site to the organisational boundary per GHG Protocol Ch. 3.

        Applies the default consolidation approach and ownership percentages.
        Flags joint ventures and determines inclusion/exclusion.
        """
        logger.info("Phase 4 -- Boundary Assignment: %d sites", len(self._classified))
        assignments: Dict[str, BoundaryAssignment] = {}
        raw_lookup: Dict[str, Dict[str, Any]] = {}
        for raw in input_data.candidate_sites:
            key = raw.get("site_name", "").strip().lower()
            raw_lookup[key] = raw

        included_count = 0
        excluded_count = 0
        jv_count = 0

        for cs in self._classified:
            raw = raw_lookup.get(cs.site_name.lower(), {})

            approach_str = raw.get("consolidation_approach", "")
            try:
                approach = ConsolidationApproach(approach_str)
            except ValueError:
                approach = input_data.default_consolidation

            ownership_str = raw.get("ownership_pct", "100")
            try:
                ownership = Decimal(str(ownership_str))
            except Exception:
                ownership = Decimal("100.00")
            ownership = min(max(ownership, Decimal("0")), Decimal("100"))

            is_jv = ownership < Decimal("100") or bool(raw.get("is_joint_venture", False))
            if is_jv:
                jv_count += 1

            # Determine inclusion based on approach and ownership
            is_included = True
            exclusion_reason = ""
            inclusion_rationale = self._build_inclusion_rationale(approach, ownership, is_jv)

            if approach == ConsolidationApproach.EQUITY_SHARE and ownership <= Decimal("0"):
                is_included = False
                exclusion_reason = "Zero equity share"
            elif approach == ConsolidationApproach.OPERATIONAL_CONTROL:
                has_control = raw.get("has_operational_control", True)
                if not has_control:
                    is_included = False
                    exclusion_reason = "No operational control"
            elif approach == ConsolidationApproach.FINANCIAL_CONTROL:
                has_control = raw.get("has_financial_control", True)
                if not has_control:
                    is_included = False
                    exclusion_reason = "No financial control"

            if is_included:
                included_count += 1
            else:
                excluded_count += 1

            assignment = BoundaryAssignment(
                site_id=cs.site_id,
                consolidation_approach=approach,
                ownership_pct=ownership.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                is_joint_venture=is_jv,
                parent_entity=raw.get("parent_entity", cs.legal_entity),
                control_type=approach.value,
                inclusion_rationale=inclusion_rationale,
                exclusion_reason=exclusion_reason,
                is_included=is_included,
                effective_date=raw.get("effective_date", _utcnow().date().isoformat()),
            )
            assignments[cs.site_id] = assignment

        self._assignments = assignments

        approach_dist: Dict[str, int] = {}
        for a in assignments.values():
            key = a.consolidation_approach.value
            approach_dist[key] = approach_dist.get(key, 0) + 1

        logger.info(
            "Boundary assignment: %d included, %d excluded, %d JVs",
            included_count, excluded_count, jv_count,
        )
        return {
            "included_count": included_count,
            "excluded_count": excluded_count,
            "joint_venture_count": jv_count,
            "approach_distribution": approach_dist,
        }

    def _build_inclusion_rationale(
        self,
        approach: ConsolidationApproach,
        ownership: Decimal,
        is_jv: bool,
    ) -> str:
        """Build a textual inclusion rationale."""
        if approach == ConsolidationApproach.EQUITY_SHARE:
            return (
                f"Included under equity share approach at {ownership}% ownership. "
                f"Emissions reported proportional to equity interest per GHG Protocol Ch. 3."
            )
        elif approach == ConsolidationApproach.OPERATIONAL_CONTROL:
            jv_note = " (joint venture)" if is_jv else ""
            return (
                f"Included under operational control approach{jv_note}. "
                f"Organisation has operational control; 100% of emissions reported."
            )
        else:
            return (
                f"Included under financial control approach at {ownership}% ownership. "
                f"Organisation has financial control per GHG Protocol Ch. 3."
            )

    # -----------------------------------------------------------------
    # PHASE 5 -- ACTIVATION
    # -----------------------------------------------------------------

    def _phase_activation(
        self,
        input_data: SiteRegistrationInput,
        result: SiteRegistrationResult,
    ) -> Dict[str, Any]:
        """
        Validate completeness and activate sites in the registry.

        Sites meeting the completeness threshold and included in the
        boundary are activated. Generates per-site provenance hashes.
        """
        logger.info("Phase 5 -- Activation")
        registered: List[RegisteredSite] = []
        activated_count = 0
        below_threshold_count = 0
        excluded_count = 0
        now_iso = _utcnow().isoformat()

        threshold = input_data.completeness_threshold * Decimal("100")

        for cs in self._classified:
            sid = cs.site_id
            chars = self._characteristics.get(sid)
            assignment = self._assignments.get(sid)

            if assignment and not assignment.is_included:
                excluded_count += 1
                continue

            completeness = chars.data_completeness_pct if chars else Decimal("0")
            is_complete = completeness >= threshold

            status = SiteStatus.ACTIVE if is_complete else SiteStatus.CANDIDATE
            if not is_complete:
                below_threshold_count += 1
                result.warnings.append(
                    f"Site {cs.site_name} below completeness threshold: "
                    f"{completeness}% < {threshold}%"
                )

            if is_complete:
                activated_count += 1

            prov_input = (
                f"{sid}|{cs.site_name}|{cs.country_code}|"
                f"{cs.facility_type.value}|{assignment.ownership_pct if assignment else 100}|"
                f"{now_iso}"
            )
            prov_hash = _compute_hash(prov_input)

            reg = RegisteredSite(
                site_id=sid,
                site_name=cs.site_name,
                legal_entity=cs.legal_entity,
                country_code=cs.country_code,
                facility_type=cs.facility_type,
                sector=cs.sector,
                geographic_region=cs.geographic_region,
                business_unit=cs.business_unit,
                floor_area_sqm=chars.floor_area_sqm if chars else None,
                headcount=chars.headcount if chars else None,
                operating_hours_yr=chars.operating_hours_yr if chars else None,
                grid_region=chars.grid_region if chars else "",
                primary_fuel=chars.primary_fuel if chars else "",
                consolidation_approach=(
                    assignment.consolidation_approach
                    if assignment
                    else input_data.default_consolidation
                ),
                ownership_pct=(
                    assignment.ownership_pct if assignment else Decimal("100.00")
                ),
                is_included=True,
                status=status,
                activated_at=now_iso if is_complete else "",
                provenance_hash=prov_hash,
            )
            registered.append(reg)

        result.registered_sites = registered
        result.total_activated = activated_count
        result.total_excluded = excluded_count

        logger.info(
            "Activation complete: %d activated, %d below threshold, %d excluded",
            activated_count, below_threshold_count, excluded_count,
        )
        return {
            "activated_count": activated_count,
            "below_threshold_count": below_threshold_count,
            "excluded_count": excluded_count,
            "total_registered": len(registered),
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "SiteRegistrationWorkflow",
    "SiteRegistrationInput",
    "SiteRegistrationResult",
    "RegistrationPhase",
    "PhaseStatus",
    "WorkflowStatus",
    "FacilityType",
    "SectorClassification",
    "ConsolidationApproach",
    "SiteStatus",
    "GeographicRegion",
    "CandidateSite",
    "ClassifiedSite",
    "SiteCharacteristics",
    "BoundaryAssignment",
    "RegisteredSite",
    "PhaseResult",
]
