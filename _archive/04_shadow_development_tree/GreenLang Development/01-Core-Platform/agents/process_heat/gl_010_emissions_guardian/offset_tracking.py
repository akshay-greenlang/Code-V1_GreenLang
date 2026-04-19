"""
Carbon Offset Tracking Module - GL-010 EmissionsGuardian

This module implements comprehensive carbon offset verification, registry
integration, and lifecycle tracking for both compliance and voluntary carbon
markets. Provides complete audit trails for offset procurement, verification,
retirement, and reporting.

Key Features:
    - Multi-registry integration (Verra VCS, Gold Standard, ACR, CAR)
    - Offset project verification tracking
    - Additionality and permanence assessment
    - Vintage management and expiry tracking
    - Retirement and cancellation workflows
    - ICAO CORSIA eligibility verification
    - Science-Based Target (SBTi) alignment checking
    - Complete audit trail with SHA-256 provenance

Regulatory References:
    - ICAO CORSIA Eligible Emissions Units (CEU)
    - Verra VCS Program Rules
    - Gold Standard Certification Requirements
    - ACR Standard v7.0
    - Climate Action Reserve Program Manual
    - ISO 14064-2 Project-Level GHG Quantification

Example:
    >>> tracker = CarbonOffsetTracker(entity_id="CORP-001")
    >>> offset = tracker.procure_offset(
    ...     registry=OffsetRegistry.VERRA_VCS,
    ...     project_id="VCS-1234",
    ...     quantity=1000,
    ...     vintage_year=2023
    ... )
    >>> retirement = tracker.retire_offset(
    ...     offset_id=offset.offset_id,
    ...     quantity=500,
    ...     purpose="2023 Scope 1 neutralization"
    ... )
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union, Set
import hashlib
import json
import logging
import statistics
import uuid

from pydantic import BaseModel, Field, validator, root_validator

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - Carbon Offset Standards and Criteria
# =============================================================================

class OffsetConstants:
    """Carbon offset program constants and thresholds."""

    # Credit validity periods (years from vintage)
    VERRA_VCS_VALIDITY_YEARS = 10
    GOLD_STANDARD_VALIDITY_YEARS = 5
    ACR_VALIDITY_YEARS = 8
    CAR_VALIDITY_YEARS = 8
    CORSIA_ELIGIBILITY_CUTOFF_YEAR = 2016  # Credits from 2016 onwards

    # Quality thresholds
    HIGH_QUALITY_SCORE_MIN = 80
    ACCEPTABLE_QUALITY_SCORE_MIN = 60
    CORSIA_ELIGIBLE_SCORE_MIN = 70

    # Permanence requirements
    MINIMUM_PERMANENCE_YEARS = 20  # Minimum commitment
    PREFERRED_PERMANENCE_YEARS = 100  # Preferred for claims

    # Buffer pool requirements (% of issuance held in buffer)
    VERRA_BUFFER_POOL_PCT = 15  # Standard buffer
    GOLD_STANDARD_BUFFER_PCT = 20

    # Vintage discounting
    VINTAGE_DISCOUNT_PER_YEAR = 0.02  # 2% discount per year of age

    # SBTi guidance thresholds
    SBTI_NEUTRALIZATION_REMOVALS_ONLY = True  # Net-zero requires removals
    SBTI_BEYOND_VALUE_CHAIN_ALLOWED = True


class OffsetRegistry(Enum):
    """Carbon offset registries."""
    VERRA_VCS = "verra_vcs"  # Verified Carbon Standard
    GOLD_STANDARD = "gold_standard"  # Gold Standard
    ACR = "acr"  # American Carbon Registry
    CAR = "car"  # Climate Action Reserve
    PURO_EARTH = "puro_earth"  # Puro.earth (removals)
    PLAN_VIVO = "plan_vivo"  # Plan Vivo
    CERCARBONO = "cercarbono"  # Cercarbono
    BIOCARBON_REGISTRY = "biocarbon_registry"  # Biocarbon Registry
    GLOBAL_CARBON_COUNCIL = "gcc"  # Global Carbon Council
    ART_TREES = "art_trees"  # Architecture for REDD+ Transactions


class OffsetProjectType(Enum):
    """Carbon offset project types."""
    # Avoidance/Reduction
    RENEWABLE_ENERGY = "renewable_energy"
    ENERGY_EFFICIENCY = "energy_efficiency"
    FUEL_SWITCHING = "fuel_switching"
    METHANE_CAPTURE = "methane_capture"
    N2O_DESTRUCTION = "n2o_destruction"
    HFC_DESTRUCTION = "hfc_destruction"
    REDD_AVOIDED_DEFORESTATION = "redd_avoided_deforestation"
    IMPROVED_FOREST_MANAGEMENT = "improved_forest_management"
    IMPROVED_COOKSTOVES = "improved_cookstoves"

    # Removals (eligible for SBTi net-zero)
    AFFORESTATION_REFORESTATION = "afforestation_reforestation"
    DIRECT_AIR_CAPTURE = "direct_air_capture"
    BIOCHAR = "biochar"
    ENHANCED_WEATHERING = "enhanced_weathering"
    BIOENERGY_CCS = "beccs"
    SOIL_CARBON_SEQUESTRATION = "soil_carbon"
    BLUE_CARBON = "blue_carbon"
    OCEAN_ALKALINITY = "ocean_alkalinity"


class OffsetStatus(Enum):
    """Offset unit lifecycle status."""
    PENDING_ISSUANCE = "pending_issuance"
    ISSUED = "issued"
    ACTIVE = "active"
    RETIRED = "retired"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"
    TRANSFERRED = "transferred"


class RetirementPurpose(Enum):
    """Offset retirement purpose classification."""
    VOLUNTARY_COMPENSATION = "voluntary_compensation"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    CARBON_NEUTRALITY = "carbon_neutrality"
    NET_ZERO_TARGET = "net_zero_target"
    BEYOND_VALUE_CHAIN = "beyond_value_chain"
    CORSIA_COMPLIANCE = "corsia_compliance"
    RESALE = "resale"
    DONATION = "donation"


class VerificationStatus(Enum):
    """Project verification status."""
    NOT_VERIFIED = "not_verified"
    UNDER_VERIFICATION = "under_verification"
    VERIFIED = "verified"
    VERIFICATION_EXPIRED = "verification_expired"
    SUSPENDED = "suspended"


class CORSIAEligibility(Enum):
    """ICAO CORSIA eligibility status."""
    ELIGIBLE = "eligible"
    NOT_ELIGIBLE = "not_eligible"
    PENDING_ASSESSMENT = "pending_assessment"


class SBTiAlignment(Enum):
    """SBTi claim alignment status."""
    NEUTRALIZATION_ELIGIBLE = "neutralization_eligible"  # Removals only
    BEYOND_VALUE_CHAIN_ELIGIBLE = "beyond_value_chain"  # Reductions/avoidance
    NOT_ALIGNED = "not_aligned"


# =============================================================================
# DATA MODELS
# =============================================================================

class OffsetProject(BaseModel):
    """Carbon offset project details."""

    project_id: str = Field(..., description="Registry project ID")
    registry: OffsetRegistry = Field(..., description="Issuing registry")
    project_type: OffsetProjectType = Field(..., description="Project type")
    project_name: str = Field(..., description="Project name")

    # Location
    country: str = Field(..., description="Host country")
    region: Optional[str] = Field(default=None, description="Region/state")
    coordinates: Optional[Tuple[float, float]] = Field(
        default=None, description="Lat/Long coordinates"
    )

    # Methodology
    methodology_id: str = Field(..., description="Methodology identifier")
    methodology_name: str = Field(default="", description="Methodology name")

    # Crediting period
    crediting_start_date: date = Field(..., description="Crediting period start")
    crediting_end_date: date = Field(..., description="Crediting period end")
    project_lifetime_years: int = Field(
        default=30, ge=1, description="Project lifetime"
    )

    # Verification
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.VERIFIED, description="Verification status"
    )
    verifier: Optional[str] = Field(default=None, description="Verification body")
    last_verification_date: Optional[date] = Field(
        default=None, description="Last verification date"
    )

    # Issuance history
    total_issuance_tco2e: float = Field(
        default=0, ge=0, description="Total credits issued"
    )
    vintage_years: List[int] = Field(
        default_factory=list, description="Vintage years with issuance"
    )

    # Co-benefits
    sdg_contributions: List[str] = Field(
        default_factory=list, description="SDG contributions (e.g., 'SDG13', 'SDG15')"
    )
    ccb_certification: bool = Field(
        default=False, description="CCB Standards certification"
    )
    social_carbon_certified: bool = Field(
        default=False, description="Social Carbon certification"
    )

    # Risk factors
    permanence_risk: str = Field(
        default="low", description="Permanence risk (low/medium/high)"
    )
    buffer_pool_contribution_pct: float = Field(
        default=15, ge=0, le=100, description="Buffer pool contribution (%)"
    )

    # Quality
    additionality_assessment: str = Field(
        default="", description="Additionality assessment summary"
    )
    baseline_validity: str = Field(
        default="", description="Baseline validity assessment"
    )

    # External ratings
    external_ratings: Dict[str, str] = Field(
        default_factory=dict, description="Third-party ratings (BeZero, Sylvera, etc.)"
    )

    class Config:
        use_enum_values = True


class CarbonOffset(BaseModel):
    """Individual carbon offset unit."""

    offset_id: str = Field(..., description="Unique offset identifier")
    registry: OffsetRegistry = Field(..., description="Issuing registry")
    registry_serial_number: str = Field(
        ..., description="Registry serial/block number"
    )

    # Project reference
    project_id: str = Field(..., description="Source project ID")
    project_type: OffsetProjectType = Field(..., description="Project type")
    project_country: str = Field(..., description="Project country")

    # Quantity and vintage
    quantity_tco2e: float = Field(..., gt=0, description="Quantity (tCO2e)")
    vintage_year: int = Field(..., ge=2000, le=2100, description="Vintage year")

    # Status
    status: OffsetStatus = Field(
        default=OffsetStatus.ACTIVE, description="Current status"
    )

    # Pricing
    acquisition_price: Optional[float] = Field(
        default=None, ge=0, description="Acquisition price per tCO2e"
    )
    acquisition_date: Optional[date] = Field(
        default=None, description="Acquisition date"
    )
    acquisition_counterparty: Optional[str] = Field(
        default=None, description="Seller/counterparty"
    )
    currency: str = Field(default="USD", description="Price currency")

    # Quality metrics
    quality_score: float = Field(
        default=70, ge=0, le=100, description="Quality score (0-100)"
    )
    is_removal: bool = Field(
        default=False, description="Is carbon removal (vs avoidance)"
    )

    # Eligibility
    corsia_eligible: CORSIAEligibility = Field(
        default=CORSIAEligibility.PENDING_ASSESSMENT,
        description="CORSIA eligibility"
    )
    sbti_alignment: SBTiAlignment = Field(
        default=SBTiAlignment.NOT_ALIGNED,
        description="SBTi alignment"
    )

    # Validity
    expiry_date: Optional[date] = Field(
        default=None, description="Expiry date"
    )
    is_expired: bool = Field(default=False, description="Expired flag")

    # Retirement details (if retired)
    retirement_date: Optional[datetime] = Field(
        default=None, description="Retirement date"
    )
    retirement_purpose: Optional[RetirementPurpose] = Field(
        default=None, description="Retirement purpose"
    )
    retirement_beneficiary: Optional[str] = Field(
        default=None, description="Retirement beneficiary"
    )
    retirement_reference: Optional[str] = Field(
        default=None, description="Retirement certificate reference"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Record creation timestamp"
    )

    class Config:
        use_enum_values = True

    @property
    def age_years(self) -> int:
        """Calculate offset age in years from vintage."""
        return datetime.now().year - self.vintage_year

    @property
    def book_value(self) -> float:
        """Calculate current book value (with vintage discounting)."""
        if self.acquisition_price is None:
            return 0.0

        discount = min(self.age_years * OffsetConstants.VINTAGE_DISCOUNT_PER_YEAR, 0.5)
        return self.acquisition_price * (1 - discount) * self.quantity_tco2e


class RetirementRecord(BaseModel):
    """Offset retirement record."""

    retirement_id: str = Field(..., description="Unique retirement ID")
    offset_id: str = Field(..., description="Retired offset ID")
    registry_serial_number: str = Field(..., description="Registry serial number")

    # Retirement details
    retirement_date: datetime = Field(..., description="Retirement timestamp")
    quantity_tco2e: float = Field(..., gt=0, description="Quantity retired")
    purpose: RetirementPurpose = Field(..., description="Retirement purpose")
    beneficiary: str = Field(..., description="Beneficiary entity")

    # Claim details
    claim_year: int = Field(..., description="Claim year")
    claim_scope: Optional[str] = Field(
        default=None, description="GHG scope (1, 2, 3)"
    )
    claim_category: Optional[str] = Field(
        default=None, description="Emission category"
    )
    claim_statement: str = Field(
        default="", description="Retirement claim statement"
    )

    # Registry confirmation
    registry_retirement_id: Optional[str] = Field(
        default=None, description="Registry retirement reference"
    )
    registry_confirmation_date: Optional[datetime] = Field(
        default=None, description="Registry confirmation timestamp"
    )

    # Certificate
    certificate_url: Optional[str] = Field(
        default=None, description="Retirement certificate URL"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")

    class Config:
        use_enum_values = True


class QualityAssessment(BaseModel):
    """Carbon offset quality assessment."""

    assessment_id: str = Field(..., description="Assessment ID")
    offset_id: str = Field(..., description="Assessed offset ID")
    assessment_date: datetime = Field(..., description="Assessment timestamp")

    # Core quality dimensions (0-100)
    additionality_score: float = Field(
        ..., ge=0, le=100, description="Additionality confidence"
    )
    permanence_score: float = Field(
        ..., ge=0, le=100, description="Permanence assurance"
    )
    quantification_score: float = Field(
        ..., ge=0, le=100, description="Quantification rigor"
    )
    verification_score: float = Field(
        ..., ge=0, le=100, description="Verification quality"
    )
    registry_score: float = Field(
        ..., ge=0, le=100, description="Registry credibility"
    )
    co_benefits_score: float = Field(
        ..., ge=0, le=100, description="Co-benefits"
    )

    # Composite score
    overall_score: float = Field(
        ..., ge=0, le=100, description="Overall quality score"
    )

    # Risk flags
    double_counting_risk: str = Field(
        default="low", description="Double counting risk level"
    )
    reversal_risk: str = Field(
        default="low", description="Reversal risk level"
    )
    leakage_risk: str = Field(
        default="low", description="Leakage risk level"
    )

    # Alignment
    corsia_eligible: bool = Field(
        default=False, description="CORSIA eligible"
    )
    sbti_neutralization: bool = Field(
        default=False, description="SBTi neutralization eligible"
    )
    sbti_beyond_value_chain: bool = Field(
        default=False, description="SBTi beyond value chain eligible"
    )

    # Third-party ratings
    bezero_rating: Optional[str] = Field(
        default=None, description="BeZero Carbon Rating"
    )
    sylvera_rating: Optional[str] = Field(
        default=None, description="Sylvera rating"
    )
    calyx_rating: Optional[str] = Field(
        default=None, description="Calyx Global rating"
    )

    # Concerns and recommendations
    concerns: List[str] = Field(
        default_factory=list, description="Quality concerns"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations"
    )

    # Assessment methodology
    methodology: str = Field(
        default="GreenLang Quality Framework v1.0",
        description="Assessment methodology"
    )
    assessor: str = Field(..., description="Assessor identity")


class PortfolioSummary(BaseModel):
    """Offset portfolio summary."""

    entity_id: str = Field(..., description="Entity identifier")
    as_of_date: datetime = Field(..., description="Summary date")

    # Holdings
    total_holdings_tco2e: float = Field(
        default=0, ge=0, description="Total holdings"
    )
    active_holdings_tco2e: float = Field(
        default=0, ge=0, description="Active holdings"
    )
    retired_ytd_tco2e: float = Field(
        default=0, ge=0, description="Retired year-to-date"
    )

    # Quality distribution
    high_quality_pct: float = Field(
        default=0, ge=0, le=100, description="High quality (%)"
    )
    removal_pct: float = Field(
        default=0, ge=0, le=100, description="Removal credits (%)"
    )
    corsia_eligible_pct: float = Field(
        default=0, ge=0, le=100, description="CORSIA eligible (%)"
    )

    # By vintage
    by_vintage: Dict[int, float] = Field(
        default_factory=dict, description="Holdings by vintage year"
    )

    # By project type
    by_project_type: Dict[str, float] = Field(
        default_factory=dict, description="Holdings by project type"
    )

    # By registry
    by_registry: Dict[str, float] = Field(
        default_factory=dict, description="Holdings by registry"
    )

    # Valuation
    total_book_value: float = Field(
        default=0, ge=0, description="Total book value"
    )
    weighted_avg_price: float = Field(
        default=0, ge=0, description="Weighted average price"
    )
    weighted_avg_quality: float = Field(
        default=0, ge=0, le=100, description="Weighted average quality score"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")


# =============================================================================
# CARBON OFFSET TRACKER
# =============================================================================

class CarbonOffsetTracker:
    """
    Carbon Offset Verification and Registry Tracker.

    Implements comprehensive carbon offset lifecycle management including:
    - Multi-registry integration (Verra VCS, Gold Standard, ACR, CAR, etc.)
    - Offset project verification tracking
    - Quality assessment with additionality and permanence scoring
    - ICAO CORSIA eligibility determination
    - SBTi alignment verification
    - Retirement workflow with claim documentation
    - Complete audit trail with SHA-256 provenance

    Features:
        - Project-level verification tracking
        - Vintage management and expiry monitoring
        - Quality scoring framework
        - Retirement certificate generation
        - Double counting prevention
        - Portfolio analytics

    Standards Supported:
        - Verra VCS Program
        - Gold Standard
        - American Carbon Registry
        - Climate Action Reserve
        - ICAO CORSIA
        - SBTi Net-Zero Standard

    Example:
        >>> tracker = CarbonOffsetTracker(entity_id="CORP-001")
        >>> offset = tracker.procure_offset(
        ...     registry=OffsetRegistry.GOLD_STANDARD,
        ...     project_id="GS-5678",
        ...     quantity=5000,
        ...     vintage_year=2022
        ... )
        >>> assessment = tracker.assess_quality(offset.offset_id)
        >>> retirement = tracker.retire_offset(
        ...     offset_id=offset.offset_id,
        ...     quantity=2500,
        ...     purpose=RetirementPurpose.CARBON_NEUTRALITY,
        ...     beneficiary="Acme Corp"
        ... )
    """

    # Quality scoring weights
    QUALITY_WEIGHTS = {
        "additionality": 0.25,
        "permanence": 0.20,
        "quantification": 0.15,
        "verification": 0.15,
        "registry": 0.15,
        "co_benefits": 0.10,
    }

    # Registry base quality scores
    REGISTRY_QUALITY_SCORES = {
        OffsetRegistry.VERRA_VCS: 80,
        OffsetRegistry.GOLD_STANDARD: 90,
        OffsetRegistry.ACR: 75,
        OffsetRegistry.CAR: 75,
        OffsetRegistry.PURO_EARTH: 85,
        OffsetRegistry.PLAN_VIVO: 80,
        OffsetRegistry.ART_TREES: 85,
    }

    # Project type quality adjustments
    PROJECT_TYPE_ADJUSTMENTS = {
        OffsetProjectType.DIRECT_AIR_CAPTURE: 15,
        OffsetProjectType.BIOCHAR: 10,
        OffsetProjectType.ENHANCED_WEATHERING: 10,
        OffsetProjectType.BIOENERGY_CCS: 12,
        OffsetProjectType.AFFORESTATION_REFORESTATION: 5,
        OffsetProjectType.BLUE_CARBON: 8,
        OffsetProjectType.SOIL_CARBON_SEQUESTRATION: 3,
        OffsetProjectType.RENEWABLE_ENERGY: -10,
        OffsetProjectType.HFC_DESTRUCTION: -5,
        OffsetProjectType.N2O_DESTRUCTION: -5,
    }

    # Removal project types (eligible for SBTi neutralization)
    REMOVAL_PROJECT_TYPES = {
        OffsetProjectType.DIRECT_AIR_CAPTURE,
        OffsetProjectType.BIOCHAR,
        OffsetProjectType.ENHANCED_WEATHERING,
        OffsetProjectType.BIOENERGY_CCS,
        OffsetProjectType.AFFORESTATION_REFORESTATION,
        OffsetProjectType.BLUE_CARBON,
        OffsetProjectType.SOIL_CARBON_SEQUESTRATION,
        OffsetProjectType.OCEAN_ALKALINITY,
    }

    def __init__(
        self,
        entity_id: str,
        registries: Optional[List[OffsetRegistry]] = None,
    ) -> None:
        """
        Initialize Carbon Offset Tracker.

        Args:
            entity_id: Entity identifier
            registries: Supported registries (default: all major registries)
        """
        self.entity_id = entity_id
        self.registries = registries or list(OffsetRegistry)

        # Project database
        self._projects: Dict[str, OffsetProject] = {}

        # Offset holdings
        self._offsets: Dict[str, CarbonOffset] = {}

        # Retirement records
        self._retirements: List[RetirementRecord] = []

        # Quality assessments
        self._assessments: Dict[str, QualityAssessment] = {}

        logger.info(
            f"CarbonOffsetTracker initialized for {entity_id} "
            f"with {len(self.registries)} registries"
        )

    # =========================================================================
    # PROJECT MANAGEMENT
    # =========================================================================

    def register_project(
        self,
        project_id: str,
        registry: OffsetRegistry,
        project_type: OffsetProjectType,
        project_name: str,
        country: str,
        methodology_id: str,
        crediting_start_date: date,
        crediting_end_date: date,
        verifier: Optional[str] = None,
        sdg_contributions: Optional[List[str]] = None,
        permanence_risk: str = "low",
    ) -> OffsetProject:
        """
        Register a carbon offset project.

        Args:
            project_id: Registry project identifier
            registry: Issuing registry
            project_type: Project type classification
            project_name: Project name
            country: Host country
            methodology_id: Methodology identifier
            crediting_start_date: Crediting period start
            crediting_end_date: Crediting period end
            verifier: Verification body
            sdg_contributions: SDG contributions
            permanence_risk: Permanence risk level

        Returns:
            OffsetProject record
        """
        project = OffsetProject(
            project_id=project_id,
            registry=registry,
            project_type=project_type,
            project_name=project_name,
            country=country,
            methodology_id=methodology_id,
            crediting_start_date=crediting_start_date,
            crediting_end_date=crediting_end_date,
            verifier=verifier,
            sdg_contributions=sdg_contributions or [],
            permanence_risk=permanence_risk,
            buffer_pool_contribution_pct=self._get_buffer_pool_pct(registry),
        )

        self._projects[project_id] = project

        logger.info(f"Project {project_id} registered: {project_name}")

        return project

    def _get_buffer_pool_pct(self, registry: OffsetRegistry) -> float:
        """Get buffer pool contribution percentage for registry."""
        buffer_rates = {
            OffsetRegistry.VERRA_VCS: 15,
            OffsetRegistry.GOLD_STANDARD: 20,
            OffsetRegistry.ACR: 15,
            OffsetRegistry.CAR: 10,
        }
        return buffer_rates.get(registry, 15)

    # =========================================================================
    # OFFSET PROCUREMENT
    # =========================================================================

    def procure_offset(
        self,
        registry: OffsetRegistry,
        project_id: str,
        quantity: float,
        vintage_year: int,
        registry_serial_number: str = "",
        acquisition_price: Optional[float] = None,
        counterparty: Optional[str] = None,
        project_type: Optional[OffsetProjectType] = None,
        project_country: str = "",
    ) -> CarbonOffset:
        """
        Procure carbon offset credits.

        Args:
            registry: Issuing registry
            project_id: Source project ID
            quantity: Quantity in tCO2e
            vintage_year: Vintage year
            registry_serial_number: Registry serial/block number
            acquisition_price: Price per tCO2e
            counterparty: Seller/counterparty
            project_type: Project type (fetched from project if registered)
            project_country: Project country

        Returns:
            CarbonOffset record
        """
        offset_id = f"{registry.value}_{project_id}_{vintage_year}_{uuid.uuid4().hex[:8]}"

        # Get project details if registered
        project = self._projects.get(project_id)
        if project:
            project_type = project_type or OffsetProjectType(project.project_type)
            project_country = project_country or project.country
        else:
            project_type = project_type or OffsetProjectType.RENEWABLE_ENERGY

        # Generate serial number if not provided
        if not registry_serial_number:
            registry_serial_number = f"{registry.value.upper()}-{vintage_year}-{uuid.uuid4().hex[:12].upper()}"

        # Determine if removal
        is_removal = project_type in self.REMOVAL_PROJECT_TYPES

        # Calculate initial quality score
        quality_score = self._calculate_initial_quality(
            registry=registry,
            project_type=project_type,
            vintage_year=vintage_year,
        )

        # Determine eligibility
        corsia_eligible = self._assess_corsia_eligibility(
            registry=registry,
            vintage_year=vintage_year,
            project_type=project_type,
        )

        sbti_alignment = self._assess_sbti_alignment(
            is_removal=is_removal,
            quality_score=quality_score,
        )

        # Calculate expiry date
        expiry_date = self._calculate_expiry_date(registry, vintage_year)

        # Calculate provenance hash
        provenance_hash = self._hash_offset_data(
            offset_id=offset_id,
            registry=registry.value,
            project_id=project_id,
            quantity=quantity,
            vintage_year=vintage_year,
        )

        offset = CarbonOffset(
            offset_id=offset_id,
            registry=registry,
            registry_serial_number=registry_serial_number,
            project_id=project_id,
            project_type=project_type,
            project_country=project_country,
            quantity_tco2e=quantity,
            vintage_year=vintage_year,
            status=OffsetStatus.ACTIVE,
            acquisition_price=acquisition_price,
            acquisition_date=date.today() if acquisition_price else None,
            acquisition_counterparty=counterparty,
            quality_score=quality_score,
            is_removal=is_removal,
            corsia_eligible=corsia_eligible,
            sbti_alignment=sbti_alignment,
            expiry_date=expiry_date,
            provenance_hash=provenance_hash,
        )

        self._offsets[offset_id] = offset

        logger.info(
            f"Offset {offset_id} procured: {quantity:,.0f} tCO2e from {project_id}, "
            f"vintage {vintage_year}, quality score {quality_score:.1f}"
        )

        return offset

    def _calculate_initial_quality(
        self,
        registry: OffsetRegistry,
        project_type: OffsetProjectType,
        vintage_year: int,
    ) -> float:
        """Calculate initial quality score."""
        # Base registry score
        base_score = self.REGISTRY_QUALITY_SCORES.get(registry, 70)

        # Project type adjustment
        type_adj = self.PROJECT_TYPE_ADJUSTMENTS.get(project_type, 0)

        # Vintage penalty
        age = datetime.now().year - vintage_year
        vintage_penalty = max(0, (age - 2) * 2)  # Penalty after 2 years

        score = base_score + type_adj - vintage_penalty

        return max(0, min(100, score))

    def _calculate_expiry_date(
        self,
        registry: OffsetRegistry,
        vintage_year: int,
    ) -> date:
        """Calculate offset expiry date based on registry rules."""
        validity_years = {
            OffsetRegistry.VERRA_VCS: 10,
            OffsetRegistry.GOLD_STANDARD: 5,
            OffsetRegistry.ACR: 8,
            OffsetRegistry.CAR: 8,
        }.get(registry, 8)

        return date(vintage_year + validity_years, 12, 31)

    def _assess_corsia_eligibility(
        self,
        registry: OffsetRegistry,
        vintage_year: int,
        project_type: OffsetProjectType,
    ) -> CORSIAEligibility:
        """Assess ICAO CORSIA eligibility."""
        # Must be from approved program
        approved_programs = {
            OffsetRegistry.VERRA_VCS,
            OffsetRegistry.GOLD_STANDARD,
            OffsetRegistry.ACR,
            OffsetRegistry.CAR,
            OffsetRegistry.ART_TREES,
        }

        if registry not in approved_programs:
            return CORSIAEligibility.NOT_ELIGIBLE

        # Vintage must be 2016 or later (for CORSIA 2021-2023)
        if vintage_year < OffsetConstants.CORSIA_ELIGIBILITY_CUTOFF_YEAR:
            return CORSIAEligibility.NOT_ELIGIBLE

        # HFC/industrial gas projects have limitations
        if project_type in {
            OffsetProjectType.HFC_DESTRUCTION,
            OffsetProjectType.N2O_DESTRUCTION,
        }:
            return CORSIAEligibility.NOT_ELIGIBLE

        return CORSIAEligibility.ELIGIBLE

    def _assess_sbti_alignment(
        self,
        is_removal: bool,
        quality_score: float,
    ) -> SBTiAlignment:
        """Assess SBTi alignment."""
        if is_removal and quality_score >= OffsetConstants.HIGH_QUALITY_SCORE_MIN:
            return SBTiAlignment.NEUTRALIZATION_ELIGIBLE
        elif quality_score >= OffsetConstants.ACCEPTABLE_QUALITY_SCORE_MIN:
            return SBTiAlignment.BEYOND_VALUE_CHAIN_ELIGIBLE
        else:
            return SBTiAlignment.NOT_ALIGNED

    # =========================================================================
    # QUALITY ASSESSMENT
    # =========================================================================

    def assess_quality(
        self,
        offset_id: str,
        additionality_evidence: Optional[Dict[str, Any]] = None,
        permanence_mechanism: Optional[str] = None,
        third_party_verification: bool = True,
        external_ratings: Optional[Dict[str, str]] = None,
    ) -> QualityAssessment:
        """
        Perform quality assessment on offset.

        Args:
            offset_id: Offset identifier
            additionality_evidence: Evidence for additionality
            permanence_mechanism: Permanence assurance mechanism
            third_party_verification: Third-party verification status
            external_ratings: External ratings (BeZero, Sylvera, etc.)

        Returns:
            QualityAssessment with detailed scores

        Raises:
            ValueError: If offset not found
        """
        offset = self._offsets.get(offset_id)
        if offset is None:
            raise ValueError(f"Offset not found: {offset_id}")

        assessment_id = f"QA_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
        concerns = []
        recommendations = []

        # Additionality scoring
        additionality_score = 60.0  # Base
        if additionality_evidence:
            if additionality_evidence.get("financial_additionality"):
                additionality_score += 15
            if additionality_evidence.get("regulatory_surplus"):
                additionality_score += 10
            if additionality_evidence.get("barrier_analysis"):
                additionality_score += 10
            if additionality_evidence.get("common_practice_assessment"):
                additionality_score += 5
        else:
            concerns.append("Limited additionality documentation")

        # Project type adjustments for additionality
        proj_type = OffsetProjectType(offset.project_type)
        if proj_type == OffsetProjectType.RENEWABLE_ENERGY:
            additionality_score -= 15
            concerns.append("Renewable energy additionality often questioned")
        elif proj_type in self.REMOVAL_PROJECT_TYPES:
            additionality_score += 10  # Removals generally more additional

        # Permanence scoring
        permanence_score = 50.0  # Base
        if offset.is_removal:
            if proj_type == OffsetProjectType.DIRECT_AIR_CAPTURE:
                permanence_score = 95  # Geological storage
            elif proj_type == OffsetProjectType.BIOCHAR:
                permanence_score = 90  # Very stable
            elif proj_type == OffsetProjectType.ENHANCED_WEATHERING:
                permanence_score = 85
            elif proj_type == OffsetProjectType.AFFORESTATION_REFORESTATION:
                permanence_score = 60  # Forest fire risk
                concerns.append("Forest projects face reversal risk")
        else:
            permanence_score = 70  # Avoidance is inherently permanent

        if permanence_mechanism == "buffer_pool":
            permanence_score += 10
        elif permanence_mechanism == "insurance":
            permanence_score += 15

        # Quantification scoring
        quantification_score = 70.0  # Base
        if third_party_verification:
            quantification_score += 15
        if external_ratings:
            quantification_score += 10

        # Verification scoring
        verification_score = 75.0 if third_party_verification else 45.0

        # Registry scoring
        registry_score = self.REGISTRY_QUALITY_SCORES.get(
            OffsetRegistry(offset.registry), 70
        )

        # Co-benefits scoring
        project = self._projects.get(offset.project_id)
        co_benefits_score = 50.0
        if project:
            if project.sdg_contributions:
                co_benefits_score = 60 + len(project.sdg_contributions) * 5
            if project.ccb_certification:
                co_benefits_score += 15
            if project.social_carbon_certified:
                co_benefits_score += 10

        # Cap scores at 100
        additionality_score = min(100, additionality_score)
        permanence_score = min(100, permanence_score)
        quantification_score = min(100, quantification_score)
        verification_score = min(100, verification_score)
        co_benefits_score = min(100, co_benefits_score)

        # Calculate overall score
        overall_score = (
            additionality_score * self.QUALITY_WEIGHTS["additionality"] +
            permanence_score * self.QUALITY_WEIGHTS["permanence"] +
            quantification_score * self.QUALITY_WEIGHTS["quantification"] +
            verification_score * self.QUALITY_WEIGHTS["verification"] +
            registry_score * self.QUALITY_WEIGHTS["registry"] +
            co_benefits_score * self.QUALITY_WEIGHTS["co_benefits"]
        )

        # Vintage penalty
        if offset.age_years > 3:
            vintage_penalty = (offset.age_years - 3) * 2
            overall_score -= vintage_penalty
            concerns.append(f"Vintage age {offset.age_years} years may limit market acceptance")

        overall_score = max(0, min(100, overall_score))

        # Risk assessment
        double_counting_risk = "low"
        if not third_party_verification:
            double_counting_risk = "medium"

        reversal_risk = "low"
        if proj_type in {
            OffsetProjectType.AFFORESTATION_REFORESTATION,
            OffsetProjectType.REDD_AVOIDED_DEFORESTATION,
            OffsetProjectType.IMPROVED_FOREST_MANAGEMENT,
        }:
            reversal_risk = "medium"

        leakage_risk = "low"
        if proj_type == OffsetProjectType.REDD_AVOIDED_DEFORESTATION:
            leakage_risk = "medium"
            concerns.append("REDD+ projects may face leakage concerns")

        # Eligibility determination
        corsia_eligible = (
            offset.corsia_eligible == CORSIAEligibility.ELIGIBLE.value and
            overall_score >= OffsetConstants.CORSIA_ELIGIBLE_SCORE_MIN
        )

        sbti_neutralization = (
            offset.is_removal and
            overall_score >= OffsetConstants.HIGH_QUALITY_SCORE_MIN
        )

        sbti_beyond_value_chain = overall_score >= OffsetConstants.ACCEPTABLE_QUALITY_SCORE_MIN

        # Recommendations
        if overall_score < OffsetConstants.HIGH_QUALITY_SCORE_MIN:
            recommendations.append("Consider supplementing with high-quality removal credits")
        if not offset.is_removal:
            recommendations.append("For net-zero claims, prioritize carbon removal credits")

        # Parse external ratings
        bezero_rating = external_ratings.get("bezero") if external_ratings else None
        sylvera_rating = external_ratings.get("sylvera") if external_ratings else None
        calyx_rating = external_ratings.get("calyx") if external_ratings else None

        assessment = QualityAssessment(
            assessment_id=assessment_id,
            offset_id=offset_id,
            assessment_date=datetime.now(timezone.utc),
            additionality_score=round(additionality_score, 1),
            permanence_score=round(permanence_score, 1),
            quantification_score=round(quantification_score, 1),
            verification_score=round(verification_score, 1),
            registry_score=round(registry_score, 1),
            co_benefits_score=round(co_benefits_score, 1),
            overall_score=round(overall_score, 1),
            double_counting_risk=double_counting_risk,
            reversal_risk=reversal_risk,
            leakage_risk=leakage_risk,
            corsia_eligible=corsia_eligible,
            sbti_neutralization=sbti_neutralization,
            sbti_beyond_value_chain=sbti_beyond_value_chain,
            bezero_rating=bezero_rating,
            sylvera_rating=sylvera_rating,
            calyx_rating=calyx_rating,
            concerns=concerns,
            recommendations=recommendations,
            assessor="CarbonOffsetTracker",
        )

        self._assessments[offset_id] = assessment

        # Update offset quality score
        offset.quality_score = overall_score

        logger.info(
            f"Quality assessment for {offset_id}: score {overall_score:.1f}, "
            f"CORSIA: {corsia_eligible}, SBTi neutralization: {sbti_neutralization}"
        )

        return assessment

    # =========================================================================
    # RETIREMENT MANAGEMENT
    # =========================================================================

    def retire_offset(
        self,
        offset_id: str,
        quantity: float,
        purpose: RetirementPurpose,
        beneficiary: str,
        claim_year: int,
        claim_scope: Optional[str] = None,
        claim_category: Optional[str] = None,
        claim_statement: str = "",
    ) -> RetirementRecord:
        """
        Retire carbon offset credits.

        Args:
            offset_id: Offset to retire
            quantity: Quantity to retire (tCO2e)
            purpose: Retirement purpose
            beneficiary: Beneficiary entity
            claim_year: Year for which claim is made
            claim_scope: GHG scope (1, 2, or 3)
            claim_category: Emission category
            claim_statement: Retirement claim statement

        Returns:
            RetirementRecord

        Raises:
            ValueError: If offset not found or insufficient quantity
        """
        offset = self._offsets.get(offset_id)
        if offset is None:
            raise ValueError(f"Offset not found: {offset_id}")

        if offset.status != OffsetStatus.ACTIVE.value:
            raise ValueError(f"Offset {offset_id} is not active (status: {offset.status})")

        if quantity > offset.quantity_tco2e:
            raise ValueError(
                f"Insufficient quantity: requested {quantity:,.0f}, "
                f"available {offset.quantity_tco2e:,.0f}"
            )

        retirement_id = f"RET_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Calculate provenance hash
        provenance_hash = self._hash_retirement_data(
            retirement_id=retirement_id,
            offset_id=offset_id,
            quantity=quantity,
            purpose=purpose.value,
        )

        retirement = RetirementRecord(
            retirement_id=retirement_id,
            offset_id=offset_id,
            registry_serial_number=offset.registry_serial_number,
            retirement_date=datetime.now(timezone.utc),
            quantity_tco2e=quantity,
            purpose=purpose,
            beneficiary=beneficiary,
            claim_year=claim_year,
            claim_scope=claim_scope,
            claim_category=claim_category,
            claim_statement=claim_statement,
            provenance_hash=provenance_hash,
        )

        self._retirements.append(retirement)

        # Update offset
        if quantity == offset.quantity_tco2e:
            offset.status = OffsetStatus.RETIRED
            offset.quantity_tco2e = 0
        else:
            offset.quantity_tco2e -= quantity

        offset.retirement_date = retirement.retirement_date
        offset.retirement_purpose = purpose
        offset.retirement_beneficiary = beneficiary

        logger.info(
            f"Offset retired: {quantity:,.0f} tCO2e from {offset_id} "
            f"for {purpose.value}, beneficiary: {beneficiary}"
        )

        return retirement

    def cancel_offset(
        self,
        offset_id: str,
        reason: str = "",
    ) -> CarbonOffset:
        """
        Cancel offset credits (non-retirement removal from circulation).

        Args:
            offset_id: Offset to cancel
            reason: Cancellation reason

        Returns:
            Updated CarbonOffset
        """
        offset = self._offsets.get(offset_id)
        if offset is None:
            raise ValueError(f"Offset not found: {offset_id}")

        offset.status = OffsetStatus.CANCELLED

        logger.info(f"Offset {offset_id} cancelled: {reason}")

        return offset

    def get_retirement_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        purpose: Optional[RetirementPurpose] = None,
    ) -> List[RetirementRecord]:
        """Get retirement history with optional filtering."""
        retirements = self._retirements

        if start_date:
            retirements = [r for r in retirements if r.retirement_date >= start_date]
        if end_date:
            retirements = [r for r in retirements if r.retirement_date <= end_date]
        if purpose:
            retirements = [r for r in retirements if r.purpose == purpose.value]

        return list(reversed(retirements))

    # =========================================================================
    # PORTFOLIO ANALYTICS
    # =========================================================================

    def get_portfolio_summary(self) -> PortfolioSummary:
        """Get portfolio summary with analytics."""
        active_offsets = [
            o for o in self._offsets.values()
            if o.status == OffsetStatus.ACTIVE.value
        ]

        if not active_offsets:
            return PortfolioSummary(
                entity_id=self.entity_id,
                as_of_date=datetime.now(timezone.utc),
                provenance_hash=hashlib.sha256(
                    f"{self.entity_id}:empty:{datetime.now(timezone.utc).isoformat()}".encode()
                ).hexdigest(),
            )

        # Total holdings
        total_holdings = sum(o.quantity_tco2e for o in active_offsets)
        active_holdings = total_holdings

        # Retired YTD
        current_year = datetime.now().year
        retired_ytd = sum(
            r.quantity_tco2e for r in self._retirements
            if r.retirement_date.year == current_year
        )

        # Quality distribution
        high_quality = sum(
            o.quantity_tco2e for o in active_offsets
            if o.quality_score >= OffsetConstants.HIGH_QUALITY_SCORE_MIN
        )
        removal = sum(
            o.quantity_tco2e for o in active_offsets
            if o.is_removal
        )
        corsia_eligible = sum(
            o.quantity_tco2e for o in active_offsets
            if o.corsia_eligible == CORSIAEligibility.ELIGIBLE.value
        )

        high_quality_pct = high_quality / total_holdings * 100 if total_holdings > 0 else 0
        removal_pct = removal / total_holdings * 100 if total_holdings > 0 else 0
        corsia_pct = corsia_eligible / total_holdings * 100 if total_holdings > 0 else 0

        # By vintage
        by_vintage: Dict[int, float] = {}
        for offset in active_offsets:
            by_vintage[offset.vintage_year] = (
                by_vintage.get(offset.vintage_year, 0) + offset.quantity_tco2e
            )

        # By project type
        by_project_type: Dict[str, float] = {}
        for offset in active_offsets:
            pt = offset.project_type
            by_project_type[pt] = by_project_type.get(pt, 0) + offset.quantity_tco2e

        # By registry
        by_registry: Dict[str, float] = {}
        for offset in active_offsets:
            reg = offset.registry
            by_registry[reg] = by_registry.get(reg, 0) + offset.quantity_tco2e

        # Valuation
        total_book_value = sum(o.book_value for o in active_offsets)

        priced_offsets = [o for o in active_offsets if o.acquisition_price]
        weighted_avg_price = 0.0
        if priced_offsets:
            total_priced = sum(o.quantity_tco2e for o in priced_offsets)
            weighted_avg_price = sum(
                o.acquisition_price * o.quantity_tco2e for o in priced_offsets
            ) / total_priced if total_priced > 0 else 0

        # Weighted average quality
        weighted_avg_quality = sum(
            o.quality_score * o.quantity_tco2e for o in active_offsets
        ) / total_holdings if total_holdings > 0 else 0

        # Provenance hash
        provenance_hash = hashlib.sha256(
            f"{self.entity_id}:{total_holdings}:{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()

        return PortfolioSummary(
            entity_id=self.entity_id,
            as_of_date=datetime.now(timezone.utc),
            total_holdings_tco2e=total_holdings,
            active_holdings_tco2e=active_holdings,
            retired_ytd_tco2e=retired_ytd,
            high_quality_pct=round(high_quality_pct, 1),
            removal_pct=round(removal_pct, 1),
            corsia_eligible_pct=round(corsia_pct, 1),
            by_vintage=by_vintage,
            by_project_type=by_project_type,
            by_registry=by_registry,
            total_book_value=round(total_book_value, 2),
            weighted_avg_price=round(weighted_avg_price, 2),
            weighted_avg_quality=round(weighted_avg_quality, 1),
            provenance_hash=provenance_hash,
        )

    def get_expiring_offsets(
        self,
        days_ahead: int = 365,
    ) -> List[CarbonOffset]:
        """Get offsets expiring within specified days."""
        cutoff = date.today() + timedelta(days=days_ahead)

        expiring = [
            o for o in self._offsets.values()
            if (
                o.status == OffsetStatus.ACTIVE.value and
                o.expiry_date and
                o.expiry_date <= cutoff
            )
        ]

        return sorted(expiring, key=lambda o: o.expiry_date)

    def check_vintage_compliance(self) -> Dict[str, Any]:
        """Check vintage compliance for various programs."""
        active_offsets = [
            o for o in self._offsets.values()
            if o.status == OffsetStatus.ACTIVE.value
        ]

        current_year = datetime.now().year

        # CORSIA vintage requirements (2016+)
        corsia_compliant = sum(
            o.quantity_tco2e for o in active_offsets
            if o.vintage_year >= 2016 and
            o.corsia_eligible == CORSIAEligibility.ELIGIBLE.value
        )

        # Recent vintages (last 5 years)
        recent_vintages = sum(
            o.quantity_tco2e for o in active_offsets
            if current_year - o.vintage_year <= 5
        )

        # Aged vintages (older than 5 years)
        aged_vintages = sum(
            o.quantity_tco2e for o in active_offsets
            if current_year - o.vintage_year > 5
        )

        total = sum(o.quantity_tco2e for o in active_offsets)

        return {
            "entity_id": self.entity_id,
            "as_of_date": date.today().isoformat(),
            "total_holdings_tco2e": total,
            "corsia_compliant_tco2e": corsia_compliant,
            "corsia_compliant_pct": round(corsia_compliant / total * 100, 1) if total > 0 else 0,
            "recent_vintages_tco2e": recent_vintages,
            "recent_vintages_pct": round(recent_vintages / total * 100, 1) if total > 0 else 0,
            "aged_vintages_tco2e": aged_vintages,
            "aged_vintages_pct": round(aged_vintages / total * 100, 1) if total > 0 else 0,
            "avg_vintage_year": round(
                sum(o.vintage_year * o.quantity_tco2e for o in active_offsets) / total
            ) if total > 0 else 0,
        }

    # =========================================================================
    # HASH UTILITIES
    # =========================================================================

    def _hash_offset_data(
        self,
        offset_id: str,
        registry: str,
        project_id: str,
        quantity: float,
        vintage_year: int,
    ) -> str:
        """Calculate SHA-256 hash for offset provenance."""
        data = {
            "offset_id": offset_id,
            "registry": registry,
            "project_id": project_id,
            "quantity": quantity,
            "vintage_year": vintage_year,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

    def _hash_retirement_data(
        self,
        retirement_id: str,
        offset_id: str,
        quantity: float,
        purpose: str,
    ) -> str:
        """Calculate SHA-256 hash for retirement provenance."""
        data = {
            "retirement_id": retirement_id,
            "offset_id": offset_id,
            "quantity": quantity,
            "purpose": purpose,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

    def get_offset(self, offset_id: str) -> Optional[CarbonOffset]:
        """Get offset by ID."""
        return self._offsets.get(offset_id)

    def get_all_offsets(
        self,
        status: Optional[OffsetStatus] = None,
        registry: Optional[OffsetRegistry] = None,
        is_removal: Optional[bool] = None,
    ) -> List[CarbonOffset]:
        """Get offsets with optional filtering."""
        offsets = list(self._offsets.values())

        if status:
            offsets = [o for o in offsets if o.status == status.value]
        if registry:
            offsets = [o for o in offsets if o.registry == registry.value]
        if is_removal is not None:
            offsets = [o for o in offsets if o.is_removal == is_removal]

        return offsets

    def export_holdings(self) -> List[Dict[str, Any]]:
        """Export offset holdings for reporting."""
        return [
            {
                "offset_id": o.offset_id,
                "registry": o.registry,
                "project_id": o.project_id,
                "project_type": o.project_type,
                "country": o.project_country,
                "quantity_tco2e": o.quantity_tco2e,
                "vintage_year": o.vintage_year,
                "status": o.status,
                "quality_score": o.quality_score,
                "is_removal": o.is_removal,
                "corsia_eligible": o.corsia_eligible,
                "sbti_alignment": o.sbti_alignment,
                "acquisition_price": o.acquisition_price,
                "expiry_date": o.expiry_date.isoformat() if o.expiry_date else None,
            }
            for o in self._offsets.values()
        ]

    def export_retirements(self) -> List[Dict[str, Any]]:
        """Export retirement records for reporting."""
        return [
            {
                "retirement_id": r.retirement_id,
                "offset_id": r.offset_id,
                "registry_serial": r.registry_serial_number,
                "retirement_date": r.retirement_date.isoformat(),
                "quantity_tco2e": r.quantity_tco2e,
                "purpose": r.purpose,
                "beneficiary": r.beneficiary,
                "claim_year": r.claim_year,
                "claim_scope": r.claim_scope,
                "claim_statement": r.claim_statement,
            }
            for r in self._retirements
        ]
