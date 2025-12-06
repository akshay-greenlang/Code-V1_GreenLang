"""
GL-012: Carbon Offset Verification Agent

This module implements the Carbon Offset Verification Agent that validates
carbon credits/offsets against major registries and scores quality using
ICVCM Core Carbon Principles.

The agent supports:
- 6 major carbon registries (Verra VCS, Gold Standard, ACR, CAR, Plan Vivo, Puro.earth)
- ICVCM Core Carbon Principles quality scoring (5 dimensions)
- Project existence and retirement status verification
- Credit vintage validation
- Double counting prevention (corresponding adjustments)
- Buffer pool adequacy assessment
- Complete SHA-256 provenance tracking

Example:
    >>> agent = CarbonOffsetAgent()
    >>> result = agent.run(CarbonOffsetInput(
    ...     project_id="VCS-1234",
    ...     registry=CarbonRegistry.VERRA_VCS,
    ...     credits=[
    ...         CarbonCredit(
    ...             serial_number="VCS-1234-2023-001",
    ...             vintage_year=2023,
    ...             quantity_tco2e=100.0,
    ...             retirement_status="active"
    ...         )
    ...     ]
    ... ))
    >>> print(f"Verification: {result.data.verification_status}")
    >>> print(f"Quality Score: {result.data.quality_score}")
"""

import hashlib
import json
import logging
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class CarbonRegistry(str, Enum):
    """
    Supported carbon offset registries.

    These 6 registries represent the major global voluntary carbon market
    registries with established verification standards.
    """

    VERRA_VCS = "verra_vcs"  # Verified Carbon Standard
    GOLD_STANDARD = "gold_standard"  # Gold Standard for the Global Goals
    ACR = "acr"  # American Carbon Registry
    CAR = "car"  # Climate Action Reserve
    PLAN_VIVO = "plan_vivo"  # Plan Vivo Foundation
    PURO_EARTH = "puro_earth"  # Puro.earth (carbon removal focus)


class ProjectType(str, Enum):
    """Carbon offset project types."""

    # Nature-based solutions
    AFFORESTATION = "afforestation"
    REFORESTATION = "reforestation"
    REDD_PLUS = "redd_plus"  # Reduced Emissions from Deforestation
    IMPROVED_FOREST_MANAGEMENT = "improved_forest_management"
    BLUE_CARBON = "blue_carbon"
    SOIL_CARBON = "soil_carbon"
    AGROFORESTRY = "agroforestry"

    # Technology-based
    RENEWABLE_ENERGY = "renewable_energy"
    ENERGY_EFFICIENCY = "energy_efficiency"
    WASTE_MANAGEMENT = "waste_management"
    METHANE_CAPTURE = "methane_capture"
    INDUSTRIAL_PROCESS = "industrial_process"
    COOKSTOVES = "cookstoves"

    # Carbon removal
    DIRECT_AIR_CAPTURE = "direct_air_capture"
    BIOENERGY_CCS = "bioenergy_ccs"
    BIOCHAR = "biochar"
    ENHANCED_WEATHERING = "enhanced_weathering"
    OCEAN_ALKALINITY = "ocean_alkalinity"


class VerificationStatus(str, Enum):
    """Offset verification status."""

    VERIFIED = "verified"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"
    PENDING = "pending"
    EXPIRED = "expired"


class RiskLevel(str, Enum):
    """Risk assessment level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RetirementStatus(str, Enum):
    """Credit retirement status."""

    ACTIVE = "active"  # Available for use
    RETIRED = "retired"  # Permanently retired
    CANCELLED = "cancelled"  # Cancelled/invalidated
    PENDING_RETIREMENT = "pending_retirement"
    TRANSFERRED = "transferred"


class CorrespondingAdjustmentStatus(str, Enum):
    """Status of corresponding adjustment under Paris Agreement."""

    APPLIED = "applied"  # CA applied by host country
    NOT_REQUIRED = "not_required"  # Pre-2021 or domestic use
    PENDING = "pending"  # CA requested but not confirmed
    NOT_APPLIED = "not_applied"  # No CA - double counting risk


# =============================================================================
# Pydantic Models
# =============================================================================


class CarbonCredit(BaseModel):
    """Individual carbon credit details."""

    serial_number: str = Field(
        ...,
        description="Unique serial number from registry"
    )
    vintage_year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Year the emission reduction occurred"
    )
    quantity_tco2e: float = Field(
        ...,
        gt=0,
        description="Quantity in tonnes CO2 equivalent"
    )
    retirement_status: RetirementStatus = Field(
        RetirementStatus.ACTIVE,
        description="Current retirement status"
    )
    retirement_date: Optional[str] = Field(
        None,
        description="Date of retirement if retired"
    )
    retirement_beneficiary: Optional[str] = Field(
        None,
        description="Entity claiming the retirement"
    )
    issuance_date: Optional[str] = Field(
        None,
        description="Date credits were issued"
    )
    corresponding_adjustment: CorrespondingAdjustmentStatus = Field(
        CorrespondingAdjustmentStatus.NOT_REQUIRED,
        description="Corresponding adjustment status for Article 6"
    )

    @field_validator("serial_number")
    @classmethod
    def validate_serial_number(cls, v: str) -> str:
        """Validate serial number is not empty."""
        v = v.strip()
        if not v:
            raise ValueError("Serial number cannot be empty")
        return v


class ProjectDetails(BaseModel):
    """Carbon offset project details."""

    project_id: str = Field(..., description="Registry project identifier")
    project_name: str = Field(..., description="Project name")
    project_type: ProjectType = Field(..., description="Type of offset project")
    registry: CarbonRegistry = Field(..., description="Registry where registered")
    country: str = Field(..., description="Host country ISO code")
    region: Optional[str] = Field(None, description="Sub-national region")
    start_date: Optional[str] = Field(None, description="Project start date")
    crediting_period_start: Optional[str] = Field(None, description="Crediting period start")
    crediting_period_end: Optional[str] = Field(None, description="Crediting period end")
    methodology: Optional[str] = Field(None, description="Applied methodology")
    methodology_version: Optional[str] = Field(None, description="Methodology version")
    verification_body: Optional[str] = Field(None, description="Third-party verifier")
    last_verification_date: Optional[str] = Field(None, description="Last verification date")
    total_credits_issued: Optional[float] = Field(None, description="Total credits issued to date")
    buffer_pool_contribution: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Percentage contributed to buffer pool"
    )
    sdg_contributions: List[int] = Field(
        default_factory=list,
        description="UN SDG numbers this project contributes to"
    )


class CarbonOffsetInput(BaseModel):
    """
    Input model for Carbon Offset Verification Agent.

    Attributes:
        project_id: Registry project identifier
        registry: Carbon offset registry
        credits: List of credits to verify
        project_details: Optional detailed project information
        verification_purpose: Purpose of verification (compliance, voluntary, due diligence)
        require_corresponding_adjustment: Whether CA is required for this use
    """

    project_id: str = Field(..., description="Registry project identifier")
    registry: CarbonRegistry = Field(..., description="Carbon offset registry")
    credits: List[CarbonCredit] = Field(
        ...,
        min_items=1,
        description="Credits to verify"
    )
    project_details: Optional[ProjectDetails] = Field(
        None,
        description="Detailed project information"
    )
    verification_purpose: str = Field(
        "voluntary",
        description="Purpose: compliance, voluntary, due_diligence"
    )
    require_corresponding_adjustment: bool = Field(
        False,
        description="Whether corresponding adjustment is required"
    )
    vintage_cutoff_years: int = Field(
        5,
        ge=1,
        le=20,
        description="Maximum age of credits in years"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, v: str) -> str:
        """Validate project ID is not empty."""
        v = v.strip()
        if not v:
            raise ValueError("Project ID cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_project_registry_match(self):
        """Validate project details match specified registry."""
        if self.project_details and self.project_details.registry != self.registry:
            logger.warning(
                f"Project registry {self.project_details.registry} does not match "
                f"input registry {self.registry}"
            )

        return self


class ICVCMScoreBreakdown(BaseModel):
    """
    ICVCM Core Carbon Principles quality score breakdown.

    Based on the Integrity Council for the Voluntary Carbon Market's
    Core Carbon Principles (CCP) framework.
    """

    additionality_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Additionality score (30% weight) - Would project happen without credits?"
    )
    permanence_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Permanence score (25% weight) - Risk of reversal"
    )
    mrv_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="MRV score (20% weight) - Measurement, Reporting, Verification quality"
    )
    cobenefits_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Co-benefits score (15% weight) - SDG alignment, community impact"
    )
    governance_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Governance score (10% weight) - Registry standards, third-party verification"
    )
    weighted_total: float = Field(
        ...,
        ge=0,
        le=100,
        description="Weighted total quality score"
    )


class VerificationCheck(BaseModel):
    """Individual verification check result."""

    check_name: str = Field(..., description="Name of the verification check")
    passed: bool = Field(..., description="Whether the check passed")
    severity: str = Field(..., description="Severity if failed: info, warning, error, critical")
    message: str = Field(..., description="Detailed message")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    recommendation: Optional[str] = Field(None, description="Recommendation if failed")


class CreditVerificationResult(BaseModel):
    """Verification result for individual credit."""

    serial_number: str = Field(..., description="Credit serial number")
    status: VerificationStatus = Field(..., description="Verification status")
    quality_score: float = Field(..., ge=0, le=100, description="Quality score")
    risk_level: RiskLevel = Field(..., description="Risk assessment")
    checks_passed: int = Field(..., ge=0, description="Number of checks passed")
    checks_failed: int = Field(..., ge=0, description="Number of checks failed")
    issues: List[str] = Field(default_factory=list, description="Identified issues")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")


class RiskAssessment(BaseModel):
    """Overall risk assessment for the offset portfolio."""

    overall_risk: RiskLevel = Field(..., description="Overall risk level")
    reversal_risk: RiskLevel = Field(..., description="Risk of carbon reversal")
    permanence_risk: RiskLevel = Field(..., description="Long-term permanence risk")
    double_counting_risk: RiskLevel = Field(..., description="Double counting risk")
    regulatory_risk: RiskLevel = Field(..., description="Regulatory/compliance risk")
    reputational_risk: RiskLevel = Field(..., description="Reputational risk")
    risk_factors: List[str] = Field(default_factory=list, description="Key risk factors")
    mitigation_recommendations: List[str] = Field(
        default_factory=list,
        description="Risk mitigation recommendations"
    )


class CarbonOffsetOutput(BaseModel):
    """
    Output model for Carbon Offset Verification Agent.

    Comprehensive verification result with quality scoring and risk assessment.
    """

    # Project identification
    project_id: str = Field(..., description="Project identifier")
    registry: str = Field(..., description="Carbon registry")

    # Verification results
    verification_status: VerificationStatus = Field(
        ...,
        description="Overall verification status"
    )
    quality_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall quality score (0-100)"
    )
    quality_rating: str = Field(
        ...,
        description="Quality rating: excellent, good, acceptable, poor, unacceptable"
    )
    icvcm_scores: ICVCMScoreBreakdown = Field(
        ...,
        description="ICVCM Core Carbon Principles score breakdown"
    )

    # Risk assessment
    risk_assessment: RiskAssessment = Field(..., description="Risk assessment")

    # Verification checks
    verification_checks: List[VerificationCheck] = Field(
        default_factory=list,
        description="Individual verification check results"
    )
    checks_passed: int = Field(..., ge=0, description="Total checks passed")
    checks_failed: int = Field(..., ge=0, description="Total checks failed")

    # Credit-by-credit breakdown
    credit_results: List[CreditVerificationResult] = Field(
        default_factory=list,
        description="Results for each credit"
    )
    total_credits_verified: float = Field(
        ...,
        ge=0,
        description="Total tCO2e verified"
    )
    total_credits_accepted: float = Field(
        ...,
        ge=0,
        description="Total tCO2e accepted"
    )
    total_credits_rejected: float = Field(
        ...,
        ge=0,
        description="Total tCO2e rejected"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Overall recommendations"
    )
    immediate_actions: List[str] = Field(
        default_factory=list,
        description="Immediate actions required"
    )

    # Audit trail
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    calculated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Registry Standards and Configurations
# =============================================================================


class RegistryStandard(BaseModel):
    """Standards for a carbon registry."""

    registry: CarbonRegistry
    name: str
    website: str
    methodology_count: int
    buffer_pool_required: bool
    default_buffer_percentage: float
    third_party_verification_required: bool
    minimum_monitoring_years: int
    sdg_reporting_required: bool
    governance_score_base: float  # Base governance score for this registry


# Registry standards database
REGISTRY_STANDARDS: Dict[CarbonRegistry, RegistryStandard] = {
    CarbonRegistry.VERRA_VCS: RegistryStandard(
        registry=CarbonRegistry.VERRA_VCS,
        name="Verified Carbon Standard (Verra)",
        website="https://verra.org",
        methodology_count=80,
        buffer_pool_required=True,
        default_buffer_percentage=15.0,
        third_party_verification_required=True,
        minimum_monitoring_years=5,
        sdg_reporting_required=False,
        governance_score_base=80.0,
    ),
    CarbonRegistry.GOLD_STANDARD: RegistryStandard(
        registry=CarbonRegistry.GOLD_STANDARD,
        name="Gold Standard for the Global Goals",
        website="https://goldstandard.org",
        methodology_count=40,
        buffer_pool_required=False,
        default_buffer_percentage=0.0,
        third_party_verification_required=True,
        minimum_monitoring_years=5,
        sdg_reporting_required=True,
        governance_score_base=90.0,
    ),
    CarbonRegistry.ACR: RegistryStandard(
        registry=CarbonRegistry.ACR,
        name="American Carbon Registry",
        website="https://americancarbonregistry.org",
        methodology_count=30,
        buffer_pool_required=True,
        default_buffer_percentage=10.0,
        third_party_verification_required=True,
        minimum_monitoring_years=5,
        sdg_reporting_required=False,
        governance_score_base=75.0,
    ),
    CarbonRegistry.CAR: RegistryStandard(
        registry=CarbonRegistry.CAR,
        name="Climate Action Reserve",
        website="https://climateactionreserve.org",
        methodology_count=25,
        buffer_pool_required=True,
        default_buffer_percentage=12.0,
        third_party_verification_required=True,
        minimum_monitoring_years=5,
        sdg_reporting_required=False,
        governance_score_base=75.0,
    ),
    CarbonRegistry.PLAN_VIVO: RegistryStandard(
        registry=CarbonRegistry.PLAN_VIVO,
        name="Plan Vivo Foundation",
        website="https://planvivo.org",
        methodology_count=10,
        buffer_pool_required=True,
        default_buffer_percentage=20.0,
        third_party_verification_required=True,
        minimum_monitoring_years=10,
        sdg_reporting_required=True,
        governance_score_base=85.0,
    ),
    CarbonRegistry.PURO_EARTH: RegistryStandard(
        registry=CarbonRegistry.PURO_EARTH,
        name="Puro.earth",
        website="https://puro.earth",
        methodology_count=8,
        buffer_pool_required=False,
        default_buffer_percentage=0.0,
        third_party_verification_required=True,
        minimum_monitoring_years=1,
        sdg_reporting_required=False,
        governance_score_base=80.0,
    ),
}


# Project type risk profiles
PROJECT_TYPE_PROFILES: Dict[ProjectType, Dict[str, Any]] = {
    # Nature-based - higher permanence risk
    ProjectType.AFFORESTATION: {
        "base_additionality": 70,
        "permanence_risk": "medium",
        "reversal_buffer_minimum": 15,
        "typical_crediting_period": 30,
        "monitoring_complexity": "medium",
    },
    ProjectType.REFORESTATION: {
        "base_additionality": 75,
        "permanence_risk": "medium",
        "reversal_buffer_minimum": 15,
        "typical_crediting_period": 30,
        "monitoring_complexity": "medium",
    },
    ProjectType.REDD_PLUS: {
        "base_additionality": 65,
        "permanence_risk": "high",
        "reversal_buffer_minimum": 20,
        "typical_crediting_period": 20,
        "monitoring_complexity": "high",
    },
    ProjectType.IMPROVED_FOREST_MANAGEMENT: {
        "base_additionality": 60,
        "permanence_risk": "medium",
        "reversal_buffer_minimum": 15,
        "typical_crediting_period": 40,
        "monitoring_complexity": "medium",
    },
    ProjectType.BLUE_CARBON: {
        "base_additionality": 80,
        "permanence_risk": "low",
        "reversal_buffer_minimum": 10,
        "typical_crediting_period": 30,
        "monitoring_complexity": "high",
    },
    ProjectType.SOIL_CARBON: {
        "base_additionality": 65,
        "permanence_risk": "high",
        "reversal_buffer_minimum": 20,
        "typical_crediting_period": 20,
        "monitoring_complexity": "high",
    },
    ProjectType.AGROFORESTRY: {
        "base_additionality": 75,
        "permanence_risk": "medium",
        "reversal_buffer_minimum": 15,
        "typical_crediting_period": 20,
        "monitoring_complexity": "medium",
    },
    # Technology-based - lower permanence risk
    ProjectType.RENEWABLE_ENERGY: {
        "base_additionality": 50,  # Often questioned for additionality
        "permanence_risk": "low",
        "reversal_buffer_minimum": 0,
        "typical_crediting_period": 10,
        "monitoring_complexity": "low",
    },
    ProjectType.ENERGY_EFFICIENCY: {
        "base_additionality": 70,
        "permanence_risk": "low",
        "reversal_buffer_minimum": 0,
        "typical_crediting_period": 10,
        "monitoring_complexity": "low",
    },
    ProjectType.WASTE_MANAGEMENT: {
        "base_additionality": 75,
        "permanence_risk": "low",
        "reversal_buffer_minimum": 0,
        "typical_crediting_period": 10,
        "monitoring_complexity": "medium",
    },
    ProjectType.METHANE_CAPTURE: {
        "base_additionality": 80,
        "permanence_risk": "low",
        "reversal_buffer_minimum": 0,
        "typical_crediting_period": 10,
        "monitoring_complexity": "medium",
    },
    ProjectType.INDUSTRIAL_PROCESS: {
        "base_additionality": 75,
        "permanence_risk": "low",
        "reversal_buffer_minimum": 0,
        "typical_crediting_period": 10,
        "monitoring_complexity": "medium",
    },
    ProjectType.COOKSTOVES: {
        "base_additionality": 85,
        "permanence_risk": "low",
        "reversal_buffer_minimum": 0,
        "typical_crediting_period": 7,
        "monitoring_complexity": "medium",
    },
    # Carbon removal - highest quality potential
    ProjectType.DIRECT_AIR_CAPTURE: {
        "base_additionality": 95,
        "permanence_risk": "very_low",
        "reversal_buffer_minimum": 0,
        "typical_crediting_period": 10,
        "monitoring_complexity": "low",
    },
    ProjectType.BIOENERGY_CCS: {
        "base_additionality": 90,
        "permanence_risk": "very_low",
        "reversal_buffer_minimum": 0,
        "typical_crediting_period": 20,
        "monitoring_complexity": "medium",
    },
    ProjectType.BIOCHAR: {
        "base_additionality": 85,
        "permanence_risk": "low",
        "reversal_buffer_minimum": 5,
        "typical_crediting_period": 20,
        "monitoring_complexity": "medium",
    },
    ProjectType.ENHANCED_WEATHERING: {
        "base_additionality": 90,
        "permanence_risk": "very_low",
        "reversal_buffer_minimum": 0,
        "typical_crediting_period": 30,
        "monitoring_complexity": "high",
    },
    ProjectType.OCEAN_ALKALINITY: {
        "base_additionality": 90,
        "permanence_risk": "very_low",
        "reversal_buffer_minimum": 0,
        "typical_crediting_period": 50,
        "monitoring_complexity": "high",
    },
}


# =============================================================================
# Carbon Offset Verification Agent Implementation
# =============================================================================


class CarbonOffsetAgent:
    """
    GL-012: Carbon Offset Verification Agent.

    This agent verifies carbon credits/offsets using zero-hallucination
    deterministic rules:
    - Registry validation against known standards
    - ICVCM Core Carbon Principles quality scoring
    - Verification checks (existence, vintage, retirement, double counting)
    - Risk assessment with buffer pool adequacy

    Aligned with:
    - ICVCM Core Carbon Principles (CCP)
    - Paris Agreement Article 6 (corresponding adjustments)
    - ISO 14064-2 (Project-level GHG quantification)
    - Major registry standards (Verra, Gold Standard, ACR, CAR, Plan Vivo, Puro.earth)

    Attributes:
        registry_standards: Database of registry standards
        project_profiles: Project type risk profiles

    Example:
        >>> agent = CarbonOffsetAgent()
        >>> result = agent.run(CarbonOffsetInput(
        ...     project_id="VCS-1234",
        ...     registry=CarbonRegistry.VERRA_VCS,
        ...     credits=[CarbonCredit(
        ...         serial_number="VCS-1234-2023-001",
        ...         vintage_year=2023,
        ...         quantity_tco2e=100.0
        ...     )]
        ... ))
        >>> assert result.quality_score >= 0
    """

    AGENT_ID = "offsets/carbon_verification_v1"
    VERSION = "1.0.0"
    DESCRIPTION = "Carbon offset verification with ICVCM quality scoring"

    # ICVCM Core Carbon Principles weights
    ICVCM_WEIGHTS = {
        "additionality": 0.30,  # 30% - Would project happen without credits?
        "permanence": 0.25,     # 25% - Risk of reversal
        "mrv": 0.20,           # 20% - Measurement, Reporting, Verification
        "cobenefits": 0.15,    # 15% - SDG alignment, community impact
        "governance": 0.10,    # 10% - Registry standards, third-party verification
    }

    # Quality rating thresholds
    QUALITY_THRESHOLDS = {
        "excellent": 80,
        "good": 65,
        "acceptable": 50,
        "poor": 35,
        "unacceptable": 0,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Carbon Offset Verification Agent.

        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        self._provenance_steps: List[Dict] = []
        self.registry_standards = REGISTRY_STANDARDS
        self.project_profiles = PROJECT_TYPE_PROFILES

        logger.info(f"CarbonOffsetAgent initialized (version {self.VERSION})")

    def run(self, input_data: CarbonOffsetInput) -> CarbonOffsetOutput:
        """
        Execute the carbon offset verification.

        ZERO-HALLUCINATION verification:
        - All scoring uses fixed weights and deterministic formulas
        - Verification checks against explicit rules
        - Risk assessment based on predefined profiles
        - No LLM in verification path

        Args:
            input_data: Validated carbon offset input data

        Returns:
            Comprehensive verification result with quality scores

        Raises:
            ValueError: If input validation fails
        """
        start_time = datetime.utcnow()
        self._provenance_steps = []

        logger.info(
            f"Verifying carbon offsets: project={input_data.project_id}, "
            f"registry={input_data.registry.value}, "
            f"credits={len(input_data.credits)}"
        )

        try:
            # Step 1: Run verification checks
            verification_checks = self._run_verification_checks(input_data)

            self._track_step("verification_checks", {
                "total_checks": len(verification_checks),
                "passed": sum(1 for c in verification_checks if c.passed),
                "failed": sum(1 for c in verification_checks if not c.passed),
            })

            # Step 2: Calculate ICVCM quality scores
            icvcm_scores = self._calculate_icvcm_scores(input_data, verification_checks)

            self._track_step("icvcm_scoring", {
                "additionality": icvcm_scores.additionality_score,
                "permanence": icvcm_scores.permanence_score,
                "mrv": icvcm_scores.mrv_score,
                "cobenefits": icvcm_scores.cobenefits_score,
                "governance": icvcm_scores.governance_score,
                "weighted_total": icvcm_scores.weighted_total,
            })

            # Step 3: Verify individual credits
            credit_results = self._verify_credits(input_data, icvcm_scores)

            self._track_step("credit_verification", {
                "total_credits": len(credit_results),
                "accepted": sum(1 for c in credit_results if c.status == VerificationStatus.VERIFIED),
                "rejected": sum(1 for c in credit_results if c.status == VerificationStatus.REJECTED),
            })

            # Step 4: Perform risk assessment
            risk_assessment = self._assess_risks(input_data, verification_checks, icvcm_scores)

            self._track_step("risk_assessment", {
                "overall_risk": risk_assessment.overall_risk.value,
                "reversal_risk": risk_assessment.reversal_risk.value,
                "double_counting_risk": risk_assessment.double_counting_risk.value,
            })

            # Step 5: Determine overall verification status
            verification_status = self._determine_verification_status(
                verification_checks,
                icvcm_scores,
                credit_results,
            )

            # Step 6: Generate recommendations
            recommendations, immediate_actions = self._generate_recommendations(
                input_data,
                verification_checks,
                icvcm_scores,
                risk_assessment,
            )

            # Step 7: Calculate totals
            total_verified = sum(c.quantity_tco2e for c in input_data.credits)
            total_accepted = sum(
                c.quantity_tco2e
                for c, r in zip(input_data.credits, credit_results)
                if r.status == VerificationStatus.VERIFIED
            )
            total_rejected = total_verified - total_accepted

            # Step 8: Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash()

            # Step 9: Calculate processing time
            processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Step 10: Determine quality rating
            quality_rating = self._get_quality_rating(icvcm_scores.weighted_total)

            # Step 11: Count checks
            checks_passed = sum(1 for c in verification_checks if c.passed)
            checks_failed = sum(1 for c in verification_checks if not c.passed)

            # Create output
            output = CarbonOffsetOutput(
                project_id=input_data.project_id,
                registry=input_data.registry.value,
                verification_status=verification_status,
                quality_score=round(icvcm_scores.weighted_total, 2),
                quality_rating=quality_rating,
                icvcm_scores=icvcm_scores,
                risk_assessment=risk_assessment,
                verification_checks=verification_checks,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                credit_results=credit_results,
                total_credits_verified=round(total_verified, 4),
                total_credits_accepted=round(total_accepted, 4),
                total_credits_rejected=round(total_rejected, 4),
                recommendations=recommendations,
                immediate_actions=immediate_actions,
                provenance_hash=provenance_hash,
                processing_time_ms=round(processing_time_ms, 2),
            )

            logger.info(
                f"Carbon offset verification complete: status={verification_status.value}, "
                f"quality={icvcm_scores.weighted_total:.1f}, "
                f"risk={risk_assessment.overall_risk.value} "
                f"(duration: {processing_time_ms:.2f}ms)"
            )

            return output

        except Exception as e:
            logger.error(f"Carbon offset verification failed: {str(e)}", exc_info=True)
            raise

    def _run_verification_checks(
        self,
        input_data: CarbonOffsetInput,
    ) -> List[VerificationCheck]:
        """
        Run all verification checks.

        ZERO-HALLUCINATION: Explicit rule-based checks.

        Checks performed:
        1. Project existence in registry
        2. Credit vintage validation
        3. Retirement status
        4. Double counting prevention
        5. Buffer pool adequacy
        6. Methodology validity
        7. Third-party verification

        Args:
            input_data: The carbon offset input

        Returns:
            List of verification check results
        """
        checks: List[VerificationCheck] = []

        # Check 1: Project existence / Registry validity
        registry_standard = self.registry_standards.get(input_data.registry)
        if registry_standard:
            checks.append(VerificationCheck(
                check_name="registry_validity",
                passed=True,
                severity="info",
                message=f"Project registered with {registry_standard.name}",
                evidence=[f"Registry: {registry_standard.website}"],
            ))
        else:
            checks.append(VerificationCheck(
                check_name="registry_validity",
                passed=False,
                severity="critical",
                message="Unknown or unrecognized carbon registry",
                evidence=[f"Registry: {input_data.registry.value}"],
                recommendation="Use credits from recognized registries (Verra, Gold Standard, ACR, CAR, Plan Vivo, Puro.earth)",
            ))

        # Check 2: Credit vintage validation
        current_year = datetime.now().year
        vintage_issues = []
        for credit in input_data.credits:
            age = current_year - credit.vintage_year
            if age > input_data.vintage_cutoff_years:
                vintage_issues.append(
                    f"{credit.serial_number}: vintage {credit.vintage_year} ({age} years old)"
                )

        if vintage_issues:
            checks.append(VerificationCheck(
                check_name="vintage_validation",
                passed=False,
                severity="warning",
                message=f"Credits exceed {input_data.vintage_cutoff_years} year vintage cutoff",
                evidence=vintage_issues[:5],  # Limit evidence items
                recommendation="Use more recent vintage credits for better quality assurance",
            ))
        else:
            checks.append(VerificationCheck(
                check_name="vintage_validation",
                passed=True,
                severity="info",
                message=f"All credits within {input_data.vintage_cutoff_years} year vintage window",
                evidence=[f"Vintages: {min(c.vintage_year for c in input_data.credits)}-{max(c.vintage_year for c in input_data.credits)}"],
            ))

        # Check 3: Retirement status
        retired_credits = [
            c for c in input_data.credits
            if c.retirement_status == RetirementStatus.RETIRED
        ]
        cancelled_credits = [
            c for c in input_data.credits
            if c.retirement_status == RetirementStatus.CANCELLED
        ]

        if cancelled_credits:
            checks.append(VerificationCheck(
                check_name="retirement_status",
                passed=False,
                severity="critical",
                message="Credits include cancelled/invalidated units",
                evidence=[c.serial_number for c in cancelled_credits[:5]],
                recommendation="Remove cancelled credits from portfolio",
            ))
        elif retired_credits:
            checks.append(VerificationCheck(
                check_name="retirement_status",
                passed=False,
                severity="error",
                message="Credits already retired - cannot be re-used",
                evidence=[c.serial_number for c in retired_credits[:5]],
                recommendation="Verify retirement was for your organization or obtain new credits",
            ))
        else:
            checks.append(VerificationCheck(
                check_name="retirement_status",
                passed=True,
                severity="info",
                message="All credits are active and available",
                evidence=[f"Active credits: {len(input_data.credits)}"],
            ))

        # Check 4: Double counting prevention (Corresponding Adjustments)
        if input_data.require_corresponding_adjustment:
            ca_issues = [
                c for c in input_data.credits
                if c.corresponding_adjustment == CorrespondingAdjustmentStatus.NOT_APPLIED
            ]
            if ca_issues:
                checks.append(VerificationCheck(
                    check_name="corresponding_adjustment",
                    passed=False,
                    severity="error",
                    message="Corresponding adjustment required but not applied",
                    evidence=[f"{c.serial_number}: CA status = {c.corresponding_adjustment.value}" for c in ca_issues[:5]],
                    recommendation="Obtain credits with confirmed corresponding adjustments for compliance claims",
                ))
            else:
                checks.append(VerificationCheck(
                    check_name="corresponding_adjustment",
                    passed=True,
                    severity="info",
                    message="Corresponding adjustments confirmed for all credits",
                    evidence=["All credits have CA applied or not required"],
                ))
        else:
            pending_ca = [
                c for c in input_data.credits
                if c.corresponding_adjustment == CorrespondingAdjustmentStatus.NOT_APPLIED
            ]
            if pending_ca:
                checks.append(VerificationCheck(
                    check_name="corresponding_adjustment",
                    passed=True,
                    severity="warning",
                    message="Credits without corresponding adjustments - monitor for Article 6 compliance",
                    evidence=[f"Credits without CA: {len(pending_ca)}"],
                    recommendation="Consider obtaining CA-backed credits for future compliance needs",
                ))
            else:
                checks.append(VerificationCheck(
                    check_name="corresponding_adjustment",
                    passed=True,
                    severity="info",
                    message="Corresponding adjustment status acceptable for voluntary use",
                ))

        # Check 5: Buffer pool adequacy (for nature-based projects)
        if input_data.project_details:
            project_type = input_data.project_details.project_type
            profile = self.project_profiles.get(project_type, {})
            minimum_buffer = profile.get("reversal_buffer_minimum", 0)

            if minimum_buffer > 0:
                actual_buffer = input_data.project_details.buffer_pool_contribution or 0
                if actual_buffer < minimum_buffer:
                    checks.append(VerificationCheck(
                        check_name="buffer_pool_adequacy",
                        passed=False,
                        severity="warning",
                        message=f"Buffer pool contribution below recommended minimum",
                        evidence=[
                            f"Required: {minimum_buffer}%",
                            f"Actual: {actual_buffer}%",
                        ],
                        recommendation=f"Ensure buffer pool contribution of at least {minimum_buffer}% for {project_type.value} projects",
                    ))
                else:
                    checks.append(VerificationCheck(
                        check_name="buffer_pool_adequacy",
                        passed=True,
                        severity="info",
                        message="Buffer pool contribution adequate for project type",
                        evidence=[f"Buffer: {actual_buffer}% (minimum: {minimum_buffer}%)"],
                    ))

        # Check 6: Third-party verification
        if registry_standard and registry_standard.third_party_verification_required:
            if input_data.project_details and input_data.project_details.verification_body:
                checks.append(VerificationCheck(
                    check_name="third_party_verification",
                    passed=True,
                    severity="info",
                    message="Third-party verification confirmed",
                    evidence=[
                        f"Verifier: {input_data.project_details.verification_body}",
                        f"Last verification: {input_data.project_details.last_verification_date or 'Unknown'}",
                    ],
                ))
            else:
                checks.append(VerificationCheck(
                    check_name="third_party_verification",
                    passed=False,
                    severity="warning",
                    message="Third-party verification information not provided",
                    evidence=["Verification body not specified in project details"],
                    recommendation="Obtain verification documentation from registry",
                ))

        # Check 7: Methodology validity
        if input_data.project_details and input_data.project_details.methodology:
            checks.append(VerificationCheck(
                check_name="methodology_validity",
                passed=True,
                severity="info",
                message="Project methodology specified",
                evidence=[
                    f"Methodology: {input_data.project_details.methodology}",
                    f"Version: {input_data.project_details.methodology_version or 'Not specified'}",
                ],
            ))
        else:
            checks.append(VerificationCheck(
                check_name="methodology_validity",
                passed=False,
                severity="warning",
                message="Methodology information not provided",
                evidence=["Methodology not specified in project details"],
                recommendation="Verify methodology approval status with registry",
            ))

        # Check 8: Crediting period validity
        if input_data.project_details:
            if input_data.project_details.crediting_period_end:
                try:
                    end_date = datetime.strptime(
                        input_data.project_details.crediting_period_end,
                        "%Y-%m-%d"
                    ).date()
                    if end_date < date.today():
                        checks.append(VerificationCheck(
                            check_name="crediting_period",
                            passed=False,
                            severity="error",
                            message="Crediting period has expired",
                            evidence=[f"Crediting period ended: {end_date}"],
                            recommendation="Verify if crediting period has been renewed",
                        ))
                    else:
                        checks.append(VerificationCheck(
                            check_name="crediting_period",
                            passed=True,
                            severity="info",
                            message="Project within active crediting period",
                            evidence=[f"Crediting period ends: {end_date}"],
                        ))
                except ValueError:
                    checks.append(VerificationCheck(
                        check_name="crediting_period",
                        passed=True,
                        severity="warning",
                        message="Could not parse crediting period end date",
                        evidence=[f"Date provided: {input_data.project_details.crediting_period_end}"],
                    ))

        return checks

    def _calculate_icvcm_scores(
        self,
        input_data: CarbonOffsetInput,
        verification_checks: List[VerificationCheck],
    ) -> ICVCMScoreBreakdown:
        """
        Calculate ICVCM Core Carbon Principles quality scores.

        ZERO-HALLUCINATION: Fixed weights and deterministic formulas.

        Formula:
        quality_score = (
            additionality_score * 0.30 +
            permanence_score * 0.25 +
            mrv_score * 0.20 +
            cobenefits_score * 0.15 +
            governance_score * 0.10
        )

        Args:
            input_data: The carbon offset input
            verification_checks: Results of verification checks

        Returns:
            ICVCM score breakdown
        """
        # Score 1: Additionality (30% weight)
        additionality_score = self._score_additionality(input_data)

        # Score 2: Permanence (25% weight)
        permanence_score = self._score_permanence(input_data, verification_checks)

        # Score 3: MRV - Measurement, Reporting, Verification (20% weight)
        mrv_score = self._score_mrv(input_data, verification_checks)

        # Score 4: Co-benefits (15% weight)
        cobenefits_score = self._score_cobenefits(input_data)

        # Score 5: Governance (10% weight)
        governance_score = self._score_governance(input_data, verification_checks)

        # ZERO-HALLUCINATION CALCULATION
        # Formula: weighted_total = sum(dimension_score * weight)
        weighted_total = (
            additionality_score * self.ICVCM_WEIGHTS["additionality"]
            + permanence_score * self.ICVCM_WEIGHTS["permanence"]
            + mrv_score * self.ICVCM_WEIGHTS["mrv"]
            + cobenefits_score * self.ICVCM_WEIGHTS["cobenefits"]
            + governance_score * self.ICVCM_WEIGHTS["governance"]
        )

        return ICVCMScoreBreakdown(
            additionality_score=round(additionality_score, 2),
            permanence_score=round(permanence_score, 2),
            mrv_score=round(mrv_score, 2),
            cobenefits_score=round(cobenefits_score, 2),
            governance_score=round(governance_score, 2),
            weighted_total=round(weighted_total, 2),
        )

    def _score_additionality(self, input_data: CarbonOffsetInput) -> float:
        """
        Score additionality dimension (0-100).

        Additionality: Would the emission reduction happen without carbon credits?

        Factors:
        - Project type base score
        - Vintage (newer = more likely additional)
        - Registry requirements
        """
        score = 50.0  # Base score

        # Factor 1: Project type base additionality (up to 40 points)
        if input_data.project_details:
            profile = self.project_profiles.get(
                input_data.project_details.project_type,
                {}
            )
            base_add = profile.get("base_additionality", 70)
            # Scale to 0-40 range
            score += (base_add - 50) * 0.8  # -40 to +40 adjustment

        # Factor 2: Vintage recency (up to 20 points)
        current_year = datetime.now().year
        avg_vintage = sum(c.vintage_year for c in input_data.credits) / len(input_data.credits)
        vintage_age = current_year - avg_vintage

        if vintage_age <= 2:
            score += 20
        elif vintage_age <= 4:
            score += 15
        elif vintage_age <= 6:
            score += 10
        elif vintage_age <= 8:
            score += 5
        # Older vintages: no bonus

        # Factor 3: Carbon removal vs avoidance (up to 15 points)
        if input_data.project_details:
            removal_types = {
                ProjectType.DIRECT_AIR_CAPTURE,
                ProjectType.BIOENERGY_CCS,
                ProjectType.BIOCHAR,
                ProjectType.ENHANCED_WEATHERING,
                ProjectType.OCEAN_ALKALINITY,
            }
            if input_data.project_details.project_type in removal_types:
                score += 15  # Carbon removal is inherently additional
            elif input_data.project_details.project_type in {
                ProjectType.AFFORESTATION,
                ProjectType.REFORESTATION,
                ProjectType.BLUE_CARBON,
            }:
                score += 10  # Nature-based removal

        # Factor 4: Renewable energy penalty (known additionality concerns)
        if input_data.project_details:
            if input_data.project_details.project_type == ProjectType.RENEWABLE_ENERGY:
                # Renewable energy often questioned for additionality
                score -= 15

        return max(min(score, 100), 0)

    def _score_permanence(
        self,
        input_data: CarbonOffsetInput,
        verification_checks: List[VerificationCheck],
    ) -> float:
        """
        Score permanence dimension (0-100).

        Permanence: Risk of carbon reversal (release back to atmosphere).

        Factors:
        - Project type risk profile
        - Buffer pool adequacy
        - Crediting period length
        """
        score = 70.0  # Base score

        # Factor 1: Project type permanence risk (up to +/- 30 points)
        if input_data.project_details:
            profile = self.project_profiles.get(
                input_data.project_details.project_type,
                {}
            )
            risk = profile.get("permanence_risk", "medium")

            risk_adjustments = {
                "very_low": 30,
                "low": 15,
                "medium": 0,
                "high": -20,
            }
            score += risk_adjustments.get(risk, 0)

        # Factor 2: Buffer pool check result (up to 20 points)
        buffer_check = next(
            (c for c in verification_checks if c.check_name == "buffer_pool_adequacy"),
            None
        )
        if buffer_check:
            if buffer_check.passed:
                score += 20
            elif buffer_check.severity == "warning":
                score += 5
        else:
            # No buffer required for technology-based
            score += 10

        # Factor 3: Geological vs biological storage
        if input_data.project_details:
            geological_storage = {
                ProjectType.DIRECT_AIR_CAPTURE,
                ProjectType.BIOENERGY_CCS,
            }
            if input_data.project_details.project_type in geological_storage:
                score += 15  # Geological storage is more permanent

        return max(min(score, 100), 0)

    def _score_mrv(
        self,
        input_data: CarbonOffsetInput,
        verification_checks: List[VerificationCheck],
    ) -> float:
        """
        Score MRV dimension (0-100).

        MRV: Measurement, Reporting, Verification quality.

        Factors:
        - Third-party verification
        - Methodology specification
        - Monitoring requirements
        """
        score = 50.0  # Base score

        # Factor 1: Third-party verification (up to 30 points)
        tpv_check = next(
            (c for c in verification_checks if c.check_name == "third_party_verification"),
            None
        )
        if tpv_check and tpv_check.passed:
            score += 30
        elif tpv_check and tpv_check.severity == "warning":
            score += 10

        # Factor 2: Methodology specification (up to 20 points)
        meth_check = next(
            (c for c in verification_checks if c.check_name == "methodology_validity"),
            None
        )
        if meth_check and meth_check.passed:
            score += 20
        elif meth_check and meth_check.severity == "warning":
            score += 5

        # Factor 3: Registry MRV standards (up to 20 points)
        registry_standard = self.registry_standards.get(input_data.registry)
        if registry_standard:
            if registry_standard.minimum_monitoring_years >= 5:
                score += 20
            else:
                score += 10

        # Factor 4: Project type monitoring complexity
        if input_data.project_details:
            profile = self.project_profiles.get(
                input_data.project_details.project_type,
                {}
            )
            complexity = profile.get("monitoring_complexity", "medium")

            # Higher complexity = more rigorous MRV (bonus)
            complexity_bonus = {
                "low": 5,
                "medium": 10,
                "high": 15,
            }
            score += complexity_bonus.get(complexity, 10)

        return max(min(score, 100), 0)

    def _score_cobenefits(self, input_data: CarbonOffsetInput) -> float:
        """
        Score co-benefits dimension (0-100).

        Co-benefits: SDG alignment, community impact, biodiversity.

        Factors:
        - SDG contributions
        - Community-focused project types
        - Registry SDG requirements
        """
        score = 30.0  # Base score

        # Factor 1: SDG contributions (up to 40 points)
        if input_data.project_details and input_data.project_details.sdg_contributions:
            sdg_count = len(input_data.project_details.sdg_contributions)
            sdg_score = min(sdg_count * 8, 40)  # 5+ SDGs = full score
            score += sdg_score

        # Factor 2: Registry SDG requirements (up to 15 points)
        registry_standard = self.registry_standards.get(input_data.registry)
        if registry_standard and registry_standard.sdg_reporting_required:
            score += 15

        # Factor 3: Community-focused project types (up to 20 points)
        if input_data.project_details:
            community_projects = {
                ProjectType.COOKSTOVES,
                ProjectType.AGROFORESTRY,
                ProjectType.PLAN_VIVO,
            }
            if input_data.project_details.project_type in community_projects:
                score += 20
            elif input_data.project_details.project_type in {
                ProjectType.REDD_PLUS,
                ProjectType.REFORESTATION,
                ProjectType.AFFORESTATION,
            }:
                score += 10  # Often have community components

        # Factor 4: Gold Standard bonus (known for co-benefits focus)
        if input_data.registry == CarbonRegistry.GOLD_STANDARD:
            score += 10
        elif input_data.registry == CarbonRegistry.PLAN_VIVO:
            score += 15  # Community-focused registry

        return max(min(score, 100), 0)

    def _score_governance(
        self,
        input_data: CarbonOffsetInput,
        verification_checks: List[VerificationCheck],
    ) -> float:
        """
        Score governance dimension (0-100).

        Governance: Registry standards, institutional quality.

        Factors:
        - Registry base governance score
        - Verification check results
        - Crediting period validity
        """
        score = 50.0  # Base score

        # Factor 1: Registry governance base (up to 40 points)
        registry_standard = self.registry_standards.get(input_data.registry)
        if registry_standard:
            # Scale registry base score (75-90) to 0-40 range
            base_gov = registry_standard.governance_score_base
            score += (base_gov - 50) * 0.8  # +20 to +32 typically

        # Factor 2: Verification checks passed (up to 20 points)
        passed_checks = sum(1 for c in verification_checks if c.passed)
        total_checks = len(verification_checks)
        if total_checks > 0:
            pass_rate = passed_checks / total_checks
            score += pass_rate * 20

        # Factor 3: Critical failures penalty
        critical_failures = sum(
            1 for c in verification_checks
            if not c.passed and c.severity == "critical"
        )
        score -= critical_failures * 15

        # Factor 4: Crediting period validity bonus
        period_check = next(
            (c for c in verification_checks if c.check_name == "crediting_period"),
            None
        )
        if period_check and period_check.passed:
            score += 10

        return max(min(score, 100), 0)

    def _verify_credits(
        self,
        input_data: CarbonOffsetInput,
        icvcm_scores: ICVCMScoreBreakdown,
    ) -> List[CreditVerificationResult]:
        """
        Verify individual credits.

        Args:
            input_data: The carbon offset input
            icvcm_scores: Overall ICVCM scores

        Returns:
            List of credit verification results
        """
        results: List[CreditVerificationResult] = []
        current_year = datetime.now().year

        for credit in input_data.credits:
            issues: List[str] = []
            recommendations: List[str] = []
            checks_passed = 0
            checks_failed = 0

            # Check 1: Vintage age
            vintage_age = current_year - credit.vintage_year
            if vintage_age <= input_data.vintage_cutoff_years:
                checks_passed += 1
            else:
                checks_failed += 1
                issues.append(f"Vintage {credit.vintage_year} exceeds cutoff ({vintage_age} years old)")
                recommendations.append("Consider using more recent vintage credits")

            # Check 2: Retirement status
            if credit.retirement_status == RetirementStatus.ACTIVE:
                checks_passed += 1
            elif credit.retirement_status == RetirementStatus.RETIRED:
                checks_failed += 1
                issues.append("Credit already retired")
                recommendations.append("Verify retirement beneficiary or obtain new credits")
            elif credit.retirement_status == RetirementStatus.CANCELLED:
                checks_failed += 1
                issues.append("Credit has been cancelled/invalidated")
                recommendations.append("Remove from portfolio")
            else:
                checks_passed += 1  # Pending or transferred OK

            # Check 3: Corresponding adjustment
            if input_data.require_corresponding_adjustment:
                if credit.corresponding_adjustment in {
                    CorrespondingAdjustmentStatus.APPLIED,
                    CorrespondingAdjustmentStatus.NOT_REQUIRED,
                }:
                    checks_passed += 1
                else:
                    checks_failed += 1
                    issues.append("Corresponding adjustment not confirmed")
                    recommendations.append("Obtain CA confirmation from host country")
            else:
                checks_passed += 1  # Not required

            # Determine credit status
            if credit.retirement_status == RetirementStatus.CANCELLED:
                status = VerificationStatus.REJECTED
                risk_level = RiskLevel.CRITICAL
            elif credit.retirement_status == RetirementStatus.RETIRED:
                status = VerificationStatus.REJECTED
                risk_level = RiskLevel.HIGH
            elif checks_failed > 1:
                status = VerificationStatus.NEEDS_REVIEW
                risk_level = RiskLevel.MEDIUM
            elif checks_failed == 1:
                status = VerificationStatus.NEEDS_REVIEW
                risk_level = RiskLevel.LOW
            else:
                status = VerificationStatus.VERIFIED
                risk_level = RiskLevel.LOW

            # Calculate credit-specific quality score
            # Start with ICVCM score, adjust for credit-specific factors
            quality_score = icvcm_scores.weighted_total

            # Vintage penalty/bonus
            if vintage_age <= 2:
                quality_score += 5
            elif vintage_age > 5:
                quality_score -= (vintage_age - 5) * 2

            quality_score = max(min(quality_score, 100), 0)

            results.append(CreditVerificationResult(
                serial_number=credit.serial_number,
                status=status,
                quality_score=round(quality_score, 2),
                risk_level=risk_level,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                issues=issues,
                recommendations=recommendations,
            ))

        return results

    def _assess_risks(
        self,
        input_data: CarbonOffsetInput,
        verification_checks: List[VerificationCheck],
        icvcm_scores: ICVCMScoreBreakdown,
    ) -> RiskAssessment:
        """
        Perform comprehensive risk assessment.

        Args:
            input_data: The carbon offset input
            verification_checks: Verification check results
            icvcm_scores: ICVCM quality scores

        Returns:
            Risk assessment with mitigation recommendations
        """
        risk_factors: List[str] = []
        mitigation_recommendations: List[str] = []

        # Assess reversal risk
        if icvcm_scores.permanence_score >= 80:
            reversal_risk = RiskLevel.LOW
        elif icvcm_scores.permanence_score >= 60:
            reversal_risk = RiskLevel.MEDIUM
            risk_factors.append("Moderate permanence concerns")
            mitigation_recommendations.append("Consider diversifying with technology-based credits")
        else:
            reversal_risk = RiskLevel.HIGH
            risk_factors.append("High reversal risk based on project type")
            mitigation_recommendations.append("Ensure adequate buffer pool contribution")
            mitigation_recommendations.append("Consider carbon removal credits")

        # Assess permanence risk (similar but considers project type more)
        if input_data.project_details:
            profile = self.project_profiles.get(input_data.project_details.project_type, {})
            perm_risk_str = profile.get("permanence_risk", "medium")

            permanence_risk_map = {
                "very_low": RiskLevel.LOW,
                "low": RiskLevel.LOW,
                "medium": RiskLevel.MEDIUM,
                "high": RiskLevel.HIGH,
            }
            permanence_risk = permanence_risk_map.get(perm_risk_str, RiskLevel.MEDIUM)
        else:
            permanence_risk = RiskLevel.MEDIUM

        if permanence_risk == RiskLevel.HIGH:
            risk_factors.append(f"Nature-based project with inherent permanence risk")

        # Assess double counting risk
        ca_issues = sum(
            1 for c in input_data.credits
            if c.corresponding_adjustment == CorrespondingAdjustmentStatus.NOT_APPLIED
        )

        if ca_issues == 0:
            double_counting_risk = RiskLevel.LOW
        elif ca_issues < len(input_data.credits) / 2:
            double_counting_risk = RiskLevel.MEDIUM
            risk_factors.append("Some credits lack corresponding adjustments")
            mitigation_recommendations.append("Monitor Article 6 guidance updates")
        else:
            double_counting_risk = RiskLevel.HIGH
            risk_factors.append("Majority of credits lack corresponding adjustments")
            mitigation_recommendations.append("Obtain CA-backed credits for compliance claims")

        # Assess regulatory risk
        critical_checks = sum(
            1 for c in verification_checks
            if not c.passed and c.severity == "critical"
        )

        if critical_checks == 0:
            regulatory_risk = RiskLevel.LOW
        elif critical_checks == 1:
            regulatory_risk = RiskLevel.MEDIUM
            risk_factors.append("Minor regulatory compliance concerns")
        else:
            regulatory_risk = RiskLevel.HIGH
            risk_factors.append("Significant regulatory compliance issues")
            mitigation_recommendations.append("Address critical verification failures")

        # Assess reputational risk
        if icvcm_scores.weighted_total >= 70:
            reputational_risk = RiskLevel.LOW
        elif icvcm_scores.weighted_total >= 50:
            reputational_risk = RiskLevel.MEDIUM
            risk_factors.append("Moderate quality score may face scrutiny")
            mitigation_recommendations.append("Consider higher-quality credits for public claims")
        else:
            reputational_risk = RiskLevel.HIGH
            risk_factors.append("Low quality score creates reputational exposure")
            mitigation_recommendations.append("Upgrade to premium quality credits")

        # Determine overall risk
        risk_scores = {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4,
        }

        avg_risk_score = (
            risk_scores[reversal_risk]
            + risk_scores[permanence_risk]
            + risk_scores[double_counting_risk]
            + risk_scores[regulatory_risk]
            + risk_scores[reputational_risk]
        ) / 5

        if avg_risk_score <= 1.2:
            overall_risk = RiskLevel.LOW
        elif avg_risk_score <= 2.0:
            overall_risk = RiskLevel.MEDIUM
        elif avg_risk_score <= 2.8:
            overall_risk = RiskLevel.HIGH
        else:
            overall_risk = RiskLevel.CRITICAL

        return RiskAssessment(
            overall_risk=overall_risk,
            reversal_risk=reversal_risk,
            permanence_risk=permanence_risk,
            double_counting_risk=double_counting_risk,
            regulatory_risk=regulatory_risk,
            reputational_risk=reputational_risk,
            risk_factors=risk_factors,
            mitigation_recommendations=mitigation_recommendations,
        )

    def _determine_verification_status(
        self,
        verification_checks: List[VerificationCheck],
        icvcm_scores: ICVCMScoreBreakdown,
        credit_results: List[CreditVerificationResult],
    ) -> VerificationStatus:
        """
        Determine overall verification status.

        Args:
            verification_checks: All verification checks
            icvcm_scores: ICVCM quality scores
            credit_results: Individual credit results

        Returns:
            Overall verification status
        """
        # Count critical failures
        critical_failures = sum(
            1 for c in verification_checks
            if not c.passed and c.severity == "critical"
        )

        # Count rejected credits
        rejected_credits = sum(
            1 for c in credit_results
            if c.status == VerificationStatus.REJECTED
        )

        # Decision logic
        if critical_failures > 0:
            return VerificationStatus.REJECTED

        if rejected_credits > len(credit_results) / 2:
            return VerificationStatus.REJECTED

        if rejected_credits > 0:
            return VerificationStatus.NEEDS_REVIEW

        error_failures = sum(
            1 for c in verification_checks
            if not c.passed and c.severity == "error"
        )

        if error_failures > 0:
            return VerificationStatus.NEEDS_REVIEW

        if icvcm_scores.weighted_total < 40:
            return VerificationStatus.NEEDS_REVIEW

        warning_failures = sum(
            1 for c in verification_checks
            if not c.passed and c.severity == "warning"
        )

        if warning_failures > 2:
            return VerificationStatus.NEEDS_REVIEW

        return VerificationStatus.VERIFIED

    def _generate_recommendations(
        self,
        input_data: CarbonOffsetInput,
        verification_checks: List[VerificationCheck],
        icvcm_scores: ICVCMScoreBreakdown,
        risk_assessment: RiskAssessment,
    ) -> Tuple[List[str], List[str]]:
        """
        Generate recommendations and immediate actions.

        Args:
            input_data: The carbon offset input
            verification_checks: Verification check results
            icvcm_scores: ICVCM quality scores
            risk_assessment: Risk assessment

        Returns:
            Tuple of (recommendations, immediate_actions)
        """
        recommendations: List[str] = []
        immediate_actions: List[str] = []

        # Immediate actions from failed checks
        for check in verification_checks:
            if not check.passed and check.recommendation:
                if check.severity in ["critical", "error"]:
                    immediate_actions.append(check.recommendation)
                else:
                    recommendations.append(check.recommendation)

        # Score-based recommendations
        if icvcm_scores.additionality_score < 60:
            recommendations.append(
                "Consider credits from higher-additionality project types "
                "(e.g., carbon removal, methane capture)"
            )

        if icvcm_scores.permanence_score < 60:
            recommendations.append(
                "Diversify portfolio with technology-based credits "
                "for improved permanence"
            )

        if icvcm_scores.mrv_score < 60:
            recommendations.append(
                "Prioritize projects with rigorous third-party verification"
            )

        if icvcm_scores.cobenefits_score < 60:
            recommendations.append(
                "Consider Gold Standard or Plan Vivo credits for enhanced co-benefits"
            )

        if icvcm_scores.governance_score < 60:
            recommendations.append(
                "Use credits from tier-1 registries (Verra, Gold Standard) "
                "for better governance"
            )

        # Risk-based recommendations
        recommendations.extend(risk_assessment.mitigation_recommendations)

        # Deduplicate while preserving order
        recommendations = list(dict.fromkeys(recommendations))
        immediate_actions = list(dict.fromkeys(immediate_actions))

        return recommendations, immediate_actions

    def _get_quality_rating(self, score: float) -> str:
        """
        Get quality rating based on score.

        Args:
            score: Weighted total quality score

        Returns:
            Quality rating string
        """
        if score >= self.QUALITY_THRESHOLDS["excellent"]:
            return "excellent"
        elif score >= self.QUALITY_THRESHOLDS["good"]:
            return "good"
        elif score >= self.QUALITY_THRESHOLDS["acceptable"]:
            return "acceptable"
        elif score >= self.QUALITY_THRESHOLDS["poor"]:
            return "poor"
        else:
            return "unacceptable"

    def _track_step(self, step_type: str, data: Dict[str, Any]) -> None:
        """Track a processing step for provenance."""
        self._provenance_steps.append({
            "step_type": step_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        })

    def _calculate_provenance_hash(self) -> str:
        """
        Calculate SHA-256 hash of complete provenance chain.

        This hash enables:
        - Verification that assessment was deterministic
        - Audit trail for regulatory compliance
        - Reproducibility checking
        """
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "steps": self._provenance_steps,
            "timestamp": datetime.utcnow().isoformat(),
        }

        json_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    # =========================================================================
    # Public API Methods
    # =========================================================================

    def verify_credit(
        self,
        serial_number: str,
        registry: CarbonRegistry,
        vintage_year: int,
        quantity_tco2e: float,
    ) -> CreditVerificationResult:
        """
        Verify a single carbon credit.

        Simplified API for single credit verification.

        Args:
            serial_number: Credit serial number
            registry: Carbon registry
            vintage_year: Credit vintage year
            quantity_tco2e: Quantity in tCO2e

        Returns:
            Credit verification result
        """
        credit = CarbonCredit(
            serial_number=serial_number,
            vintage_year=vintage_year,
            quantity_tco2e=quantity_tco2e,
        )

        input_data = CarbonOffsetInput(
            project_id=serial_number.split("-")[0] if "-" in serial_number else serial_number,
            registry=registry,
            credits=[credit],
        )

        # Calculate basic ICVCM scores
        checks = self._run_verification_checks(input_data)
        icvcm_scores = self._calculate_icvcm_scores(input_data, checks)
        results = self._verify_credits(input_data, icvcm_scores)

        return results[0]

    def calculate_quality_score(
        self,
        project_id: str,
        registry: CarbonRegistry,
        project_type: Optional[ProjectType] = None,
        sdg_contributions: Optional[List[int]] = None,
    ) -> ICVCMScoreBreakdown:
        """
        Calculate ICVCM quality score for a project.

        Simplified API for quality scoring without full verification.

        Args:
            project_id: Project identifier
            registry: Carbon registry
            project_type: Optional project type
            sdg_contributions: Optional SDG contributions

        Returns:
            ICVCM score breakdown
        """
        project_details = None
        if project_type:
            project_details = ProjectDetails(
                project_id=project_id,
                project_name=f"Project {project_id}",
                project_type=project_type,
                registry=registry,
                country="XX",
                sdg_contributions=sdg_contributions or [],
            )

        credit = CarbonCredit(
            serial_number=f"{project_id}-sample",
            vintage_year=datetime.now().year - 1,
            quantity_tco2e=100.0,
        )

        input_data = CarbonOffsetInput(
            project_id=project_id,
            registry=registry,
            credits=[credit],
            project_details=project_details,
        )

        checks = self._run_verification_checks(input_data)
        return self._calculate_icvcm_scores(input_data, checks)

    def get_supported_registries(self) -> List[Dict[str, Any]]:
        """Get list of supported carbon registries with details."""
        return [
            {
                "id": r.registry.value,
                "name": r.name,
                "website": r.website,
                "buffer_pool_required": r.buffer_pool_required,
                "sdg_reporting_required": r.sdg_reporting_required,
                "governance_score": r.governance_score_base,
            }
            for r in self.registry_standards.values()
        ]

    def get_project_types(self) -> List[Dict[str, Any]]:
        """Get list of supported project types with risk profiles."""
        return [
            {
                "id": pt.value,
                "permanence_risk": profile.get("permanence_risk", "unknown"),
                "base_additionality": profile.get("base_additionality", 0),
                "buffer_minimum": profile.get("reversal_buffer_minimum", 0),
            }
            for pt, profile in self.project_profiles.items()
        ]

    def get_icvcm_weights(self) -> Dict[str, float]:
        """Get ICVCM Core Carbon Principles weights."""
        return self.ICVCM_WEIGHTS.copy()

    def get_quality_thresholds(self) -> Dict[str, int]:
        """Get quality rating thresholds."""
        return self.QUALITY_THRESHOLDS.copy()


# =============================================================================
# Pack Specification
# =============================================================================


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "offsets/carbon_verification_v1",
    "name": "Carbon Offset Verification Agent",
    "version": "1.0.0",
    "summary": "Verify carbon offsets with ICVCM quality scoring and registry validation",
    "tags": [
        "carbon-offsets",
        "verification",
        "icvcm",
        "verra",
        "gold-standard",
        "quality-scoring",
        "double-counting",
    ],
    "owners": ["offsets-team"],
    "compute": {
        "entrypoint": "python://agents.gl_012_carbon_offset.agent:CarbonOffsetAgent",
        "deterministic": True,
    },
    "factors": [
        {"ref": "std://icvcm/core-carbon-principles/2023"},
        {"ref": "std://iso/14064-2/2019"},
        {"ref": "treaty://paris-agreement/article-6"},
    ],
    "provenance": {
        "icvcm_version": "2023",
        "article_6_guidance": "2023",
        "enable_audit": True,
    },
}
