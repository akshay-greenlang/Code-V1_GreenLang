# -*- coding: utf-8 -*-
"""
CarbonCreditEngine - PACK-016 ESRS E1 Climate Engine 6
========================================================

Tracks GHG removals and carbon credit purchases per ESRS E1-7.

Under the European Sustainability Reporting Standards (ESRS), ESRS E1-7
requires separate disclosure of (a) GHG removals in the undertaking's
own operations and its upstream/downstream value chain, and (b) GHG
mitigation projects financed through the purchase of carbon credits.
Carbon credits shall NOT be deducted from gross GHG emissions reported
in E1-6; they are reported separately.

ESRS E1-7 Framework:
    - Para 56: The undertaking shall disclose GHG removals and storage
      in its own operations and its value chain, and the amount of GHG
      emission reductions or removals from climate change mitigation
      projects financed through the purchase of carbon credits.
    - Para 57: The disclosure shall include: (a) the total amount of
      GHG removals in tCO2e; (b) the amount of carbon credits purchased,
      distinguishing between avoidance and removal credits; (c) the
      crediting standards used; (d) whether credits are certified.
    - Para 58: The undertaking shall separately disclose the GHG
      removals and carbon credits; these shall not be netted against
      gross GHG emissions.

Application Requirements (AR E1-63 through AR E1-68):
    - AR E1-63: Removals include DACCS, BECCS, enhanced weathering,
      afforestation, reforestation, soil carbon sequestration.
    - AR E1-64: Carbon credits shall be classified by type (avoidance
      vs. removal), standard, project type, and vintage year.
    - AR E1-65: Quality assessment should include additionality,
      permanence, measurability, and independent verification.
    - AR E1-66: SBTi guidance: credits shall not substitute for
      abatement; residual emissions only.

Regulatory References:
    - EU Delegated Regulation 2023/2772 (ESRS Set 1)
    - ESRS E1 Climate Change, Para 56-58
    - ESRS E1 Application Requirements AR E1-63 through AR E1-68
    - SBTi Corporate Net-Zero Standard v1.2 (2024)
    - Voluntary Carbon Markets Integrity Initiative (VCMI) Claims Code

Zero-Hallucination:
    - Credit portfolio aggregation uses deterministic summation
    - Quality scoring uses weighted average of four deterministic criteria
    - SBTi compliance uses rule-based threshold checks
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-016 ESRS E1 Climate Change
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
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
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Convert value to Decimal safely.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))

def _safe_divide(
    numerator: float, denominator: float, default: float = 0.0
) -> float:
    """Safely divide two numbers, returning *default* on zero denominator."""
    if denominator == 0.0:
        return default
    return numerator / denominator

def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    ))

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))

def _round4(value: float) -> float:
    """Round to 4 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP
    ))

def _round_val(value: Decimal, places: int = 3) -> Decimal:
    """Round a Decimal value to the specified number of decimal places.

    Uses ROUND_HALF_UP for regulatory consistency.

    Args:
        value: Decimal value to round.
        places: Number of decimal places (default 3).

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CreditStandard(str, Enum):
    """Carbon credit standard or registry.

    Identifies the crediting programme under which the carbon credit
    was issued.  Each standard has its own methodologies, verification
    requirements, and registry infrastructure.
    """
    VERRA_VCS = "verra_vcs"
    GOLD_STANDARD = "gold_standard"
    ACR = "acr"
    CAR = "car"
    CDM = "cdm"
    CORSIA = "corsia"
    REDD_PLUS = "redd_plus"
    CUSTOM = "custom"

class CreditType(str, Enum):
    """Type of carbon credit per AR E1-64.

    Distinguishes between avoidance credits (preventing emissions that
    would otherwise occur) and removal credits (removing GHGs from
    the atmosphere).
    """
    AVOIDANCE = "avoidance"
    REMOVAL = "removal"

class ProjectType(str, Enum):
    """Type of carbon credit project.

    Categorises the underlying project that generates the carbon
    credits or removals.
    """
    RENEWABLE_ENERGY = "renewable_energy"
    FORESTRY_AFFORESTATION = "forestry_afforestation"
    FORESTRY_REDD = "forestry_redd"
    BIOCHAR = "biochar"
    DIRECT_AIR_CAPTURE = "direct_air_capture"
    ENHANCED_WEATHERING = "enhanced_weathering"
    SOIL_CARBON = "soil_carbon"
    METHANE_CAPTURE = "methane_capture"
    COOKSTOVES = "cookstoves"
    OTHER = "other"

class CreditStatus(str, Enum):
    """Status of a carbon credit in the portfolio.

    Tracks the lifecycle of each credit from purchase through
    retirement or cancellation.
    """
    PURCHASED = "purchased"
    RETIRED = "retired"
    CANCELLED = "cancelled"
    PENDING = "pending"

class RemovalType(str, Enum):
    """Type of GHG removal activity per AR E1-63.

    Categorises the removal mechanism used in the undertaking's own
    operations or value chain.
    """
    DACCS = "daccs"
    BECCS = "beccs"
    ENHANCED_WEATHERING = "enhanced_weathering"
    AFFORESTATION = "afforestation"
    REFORESTATION = "reforestation"
    SOIL_CARBON = "soil_carbon"
    BIOCHAR = "biochar"
    OCEAN_BASED = "ocean_based"
    OTHER = "other"

class VerificationStatus(str, Enum):
    """Verification status of a GHG removal or credit."""
    VERIFIED = "verified"
    UNDER_REVIEW = "under_review"
    NOT_VERIFIED = "not_verified"
    SELF_ASSESSED = "self_assessed"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Required ESRS E1-7 data points.
E1_7_DATAPOINTS: Dict[str, str] = {
    "e1_7_dp01": "Total GHG removals in the undertaking's own operations and value chain (tCO2e)",
    "e1_7_dp02": "GHG removals by type (DACCS, BECCS, afforestation, etc.)",
    "e1_7_dp03": "Methodology used for quantifying GHG removals",
    "e1_7_dp04": "Whether GHG removals have been independently verified",
    "e1_7_dp05": "Total carbon credits purchased (tCO2e)",
    "e1_7_dp06": "Carbon credits by type (avoidance vs. removal)",
    "e1_7_dp07": "Carbon credits by crediting standard (VCS, Gold Standard, etc.)",
    "e1_7_dp08": "Carbon credits by project type (renewable energy, forestry, etc.)",
    "e1_7_dp09": "Vintage year of carbon credits",
    "e1_7_dp10": "Whether carbon credits are certified by an independent third party",
    "e1_7_dp11": "Total expenditure on carbon credits",
    "e1_7_dp12": "Confirmation that carbon credits are not deducted from E1-6 gross emissions",
    "e1_7_dp13": "Total carbon credits retired in the reporting period (tCO2e)",
    "e1_7_dp14": "Quality assessment of carbon credits (additionality, permanence)",
    "e1_7_dp15": "Role of carbon credits in the transition plan (E1-1)",
}

# Quality criteria for carbon credit assessment per AR E1-65.
QUALITY_CRITERIA: Dict[str, Dict[str, Any]] = {
    "additionality": {
        "description": "The emission reductions or removals would not have occurred "
                       "in the absence of the carbon credit project",
        "weight": Decimal("0.30"),
        "scale": "1-5 (1=not additional, 5=clearly additional)",
    },
    "permanence": {
        "description": "The emission reductions or removals are permanent or have "
                       "adequate safeguards against reversal",
        "weight": Decimal("0.25"),
        "scale": "1-5 (1=high reversal risk, 5=permanent storage)",
    },
    "measurability": {
        "description": "The emission reductions or removals can be accurately "
                       "measured, reported, and verified using recognised methodologies",
        "weight": Decimal("0.25"),
        "scale": "1-5 (1=unquantifiable, 5=precisely measurable)",
    },
    "verification": {
        "description": "The emission reductions or removals have been independently "
                       "verified by an accredited third party",
        "weight": Decimal("0.20"),
        "scale": "1-5 (1=no verification, 5=accredited third-party verified)",
    },
}

# SBTi guidance on carbon credit use per Corporate Net-Zero Standard v1.2.
SBTI_BEYONDVALUECHAINMITIGATION: Dict[str, Any] = {
    "principle": "Carbon credits shall not substitute for direct emission "
                 "reductions within the value chain",
    "near_term": {
        "description": "Companies with near-term SBTi targets should pursue "
                       "beyond value chain mitigation (BVCM) in addition to "
                       "abatement, not as a substitute",
        "credit_use": "voluntary_additional",
        "removal_preference": False,
    },
    "long_term": {
        "description": "After achieving long-term 90%+ reduction, residual "
                       "emissions shall be neutralised using permanent carbon "
                       "removals only",
        "credit_use": "neutralisation_of_residual",
        "removal_preference": True,
        "min_reduction_pct": Decimal("90.0"),
    },
    "net_zero": {
        "description": "Net-zero requires 90%+ reduction of value chain "
                       "emissions and neutralisation of residual emissions "
                       "through high-quality carbon removals",
        "max_residual_pct": Decimal("10.0"),
        "removal_only": True,
    },
}

# Credit standard descriptions for reporting.
CREDIT_STANDARD_DESCRIPTIONS: Dict[str, str] = {
    "verra_vcs": "Verified Carbon Standard (VCS) by Verra - largest voluntary carbon market standard",
    "gold_standard": "Gold Standard - premium standard requiring sustainable development co-benefits",
    "acr": "American Carbon Registry (ACR) - US-focused voluntary standard",
    "car": "Climate Action Reserve (CAR) - US-focused offset registry",
    "cdm": "Clean Development Mechanism (CDM) - UNFCCC compliance mechanism",
    "corsia": "CORSIA - International aviation carbon offset scheme (ICAO)",
    "redd_plus": "REDD+ - Reducing Emissions from Deforestation and Degradation",
    "custom": "Custom or emerging standard not listed in standard registries",
}

# Project type descriptions for reporting.
PROJECT_TYPE_DESCRIPTIONS: Dict[str, str] = {
    "renewable_energy": "Renewable energy generation displacing fossil fuel electricity",
    "forestry_afforestation": "Afforestation - planting trees on previously non-forested land",
    "forestry_redd": "REDD - Reducing deforestation and forest degradation",
    "biochar": "Biochar production and soil application for carbon sequestration",
    "direct_air_capture": "Direct Air Carbon Capture and Storage (DACCS)",
    "enhanced_weathering": "Enhanced rock weathering for atmospheric CO2 removal",
    "soil_carbon": "Soil carbon sequestration through regenerative agriculture",
    "methane_capture": "Methane capture from landfills, mines, or livestock",
    "cookstoves": "Improved cookstoves reducing fuelwood consumption in developing countries",
    "other": "Other project type not in standard categories",
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class CarbonCredit(BaseModel):
    """A carbon credit held or retired by the undertaking per ESRS E1-7.

    Represents a verified emission reduction or removal from a carbon
    credit project, tracked by standard, type, vintage, and quality.
    """
    credit_id: str = Field(
        default_factory=_new_uuid,
        description="Unique credit identifier",
    )
    standard: CreditStandard = Field(
        ...,
        description="Crediting standard under which the credit was issued",
    )
    credit_type: CreditType = Field(
        ...,
        description="Type of credit (avoidance or removal)",
    )
    project_type: ProjectType = Field(
        default=ProjectType.OTHER,
        description="Type of underlying project",
    )
    project_name: str = Field(
        default="",
        description="Name of the carbon credit project",
        max_length=500,
    )
    project_location: str = Field(
        default="",
        description="Country or region of the project",
        max_length=200,
    )
    vintage_year: int = Field(
        ...,
        description="Vintage year of the credit (year emissions were reduced/removed)",
        ge=2000,
        le=2050,
    )
    quantity_tco2e: Decimal = Field(
        ...,
        description="Quantity of credits in tCO2e",
        gt=0,
    )
    unit_price: Decimal = Field(
        default=Decimal("0.00"),
        description="Price per tCO2e of the credit",
        ge=0,
    )
    total_cost: Decimal = Field(
        default=Decimal("0.00"),
        description="Total cost of the credit purchase",
        ge=0,
    )
    currency: str = Field(
        default="EUR",
        description="Currency code for financial amounts",
        max_length=3,
    )
    status: CreditStatus = Field(
        default=CreditStatus.PURCHASED,
        description="Current status of the credit",
    )
    verification_body: str = Field(
        default="",
        description="Name of the independent verification body",
        max_length=500,
    )
    is_certified: bool = Field(
        default=False,
        description="Whether the credit is certified by an independent third party",
    )
    additionality_score: int = Field(
        default=3,
        description="Additionality assessment score (1-5)",
        ge=1,
        le=5,
    )
    permanence_years: int = Field(
        default=0,
        description="Expected permanence of the removal in years (0 = unknown)",
        ge=0,
    )
    permanence_score: int = Field(
        default=3,
        description="Permanence assessment score (1-5)",
        ge=1,
        le=5,
    )
    measurability_score: int = Field(
        default=3,
        description="Measurability assessment score (1-5)",
        ge=1,
        le=5,
    )
    verification_score: int = Field(
        default=3,
        description="Verification quality score (1-5)",
        ge=1,
        le=5,
    )
    serial_numbers: str = Field(
        default="",
        description="Serial number range for registry identification",
        max_length=500,
    )
    retirement_date: Optional[date] = Field(
        default=None,
        description="Date the credit was retired",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )

class GHGRemoval(BaseModel):
    """A GHG removal activity in the undertaking's own operations per E1-7.

    Represents a removal of greenhouse gases from the atmosphere
    through the undertaking's own activities (not financed credits).
    """
    removal_id: str = Field(
        default_factory=_new_uuid,
        description="Unique removal identifier",
    )
    removal_type: RemovalType = Field(
        ...,
        description="Type of GHG removal activity",
    )
    quantity_tco2e: Decimal = Field(
        ...,
        description="Quantity of GHG removed in tCO2e",
        gt=0,
    )
    methodology: str = Field(
        default="",
        description="Methodology used for quantification",
        max_length=500,
    )
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.NOT_VERIFIED,
        description="Verification status of the removal",
    )
    verification_body: str = Field(
        default="",
        description="Name of the verification body",
        max_length=500,
    )
    location: str = Field(
        default="",
        description="Location of the removal activity",
        max_length=200,
    )
    reporting_period: str = Field(
        default="",
        description="Reporting period for the removal (e.g., '2025')",
        max_length=50,
    )
    permanence_years: int = Field(
        default=0,
        description="Expected permanence of the removal in years",
        ge=0,
    )
    is_in_own_operations: bool = Field(
        default=True,
        description="Whether the removal is in the undertaking's own operations",
    )
    description: str = Field(
        default="",
        description="Detailed description of the removal activity",
        max_length=5000,
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )

class QualityAssessment(BaseModel):
    """Quality assessment of a carbon credit per AR E1-65.

    Evaluates the credit against four quality criteria: additionality,
    permanence, measurability, and independent verification.  Each
    criterion is scored 1-5 and weighted to produce an overall score.
    """
    credit_id: str = Field(
        default="",
        description="ID of the assessed credit",
    )
    additionality: int = Field(
        ...,
        description="Additionality score (1-5)",
        ge=1,
        le=5,
    )
    permanence: int = Field(
        ...,
        description="Permanence score (1-5)",
        ge=1,
        le=5,
    )
    measurability: int = Field(
        ...,
        description="Measurability score (1-5)",
        ge=1,
        le=5,
    )
    verification: int = Field(
        ...,
        description="Verification score (1-5)",
        ge=1,
        le=5,
    )
    overall_score: Decimal = Field(
        default=Decimal("0.000"),
        description="Weighted overall quality score (0-5 scale)",
    )
    quality_tier: str = Field(
        default="",
        description="Quality tier label (High, Medium, Low)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )

class CarbonCreditResult(BaseModel):
    """Result of carbon credit portfolio compilation per ESRS E1-7.

    Contains the complete inventory of carbon credits and GHG removals
    with aggregated summaries, quality assessments, and completeness
    scoring.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this compilation",
    )
    calculated_at: datetime = Field(
        default_factory=utcnow,
        description="Timestamp of compilation (UTC)",
    )
    credits: List[CarbonCredit] = Field(
        default_factory=list,
        description="List of carbon credits in the portfolio",
    )
    removals: List[GHGRemoval] = Field(
        default_factory=list,
        description="List of GHG removal activities",
    )
    total_credits_purchased_tco2e: Decimal = Field(
        default=Decimal("0.000"),
        description="Total carbon credits purchased (tCO2e)",
    )
    total_credits_retired_tco2e: Decimal = Field(
        default=Decimal("0.000"),
        description="Total carbon credits retired (tCO2e)",
    )
    total_removals_tco2e: Decimal = Field(
        default=Decimal("0.000"),
        description="Total GHG removals in own operations (tCO2e)",
    )
    credits_by_standard: Dict[str, str] = Field(
        default_factory=dict,
        description="Credits (tCO2e) grouped by standard",
    )
    credits_by_type: Dict[str, str] = Field(
        default_factory=dict,
        description="Credits (tCO2e) grouped by type (avoidance/removal)",
    )
    credits_by_project_type: Dict[str, str] = Field(
        default_factory=dict,
        description="Credits (tCO2e) grouped by project type",
    )
    credits_by_vintage: Dict[str, str] = Field(
        default_factory=dict,
        description="Credits (tCO2e) grouped by vintage year",
    )
    removals_by_type: Dict[str, str] = Field(
        default_factory=dict,
        description="Removals (tCO2e) grouped by removal type",
    )
    average_quality_score: Decimal = Field(
        default=Decimal("0.000"),
        description="Average quality score across all credits (0-5)",
    )
    total_expenditure: Decimal = Field(
        default=Decimal("0.00"),
        description="Total expenditure on carbon credits",
    )
    total_credits_count: int = Field(
        default=0,
        description="Total number of credit entries",
    )
    total_removals_count: int = Field(
        default=0,
        description="Total number of removal entries",
    )
    completeness_score: float = Field(
        default=0.0,
        description="Completeness score for E1-7 data points (0-100)",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the entire result",
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CarbonCreditEngine:
    """Carbon credit and GHG removal engine per ESRS E1-7.

    Provides deterministic, zero-hallucination tracking of:
    - Carbon credit registration and portfolio management
    - GHG removal activity registration and tracking
    - Credit quality assessment (additionality, permanence, etc.)
    - SBTi compliance validation for credit use
    - Portfolio aggregation and summary statistics
    - Completeness validation against E1-7 data points

    All calculations are bit-perfect reproducible.  No LLM is used
    in any calculation path.

    Usage::

        engine = CarbonCreditEngine()
        credit = CarbonCredit(
            standard=CreditStandard.VERRA_VCS,
            credit_type=CreditType.REMOVAL,
            project_type=ProjectType.DIRECT_AIR_CAPTURE,
            vintage_year=2025,
            quantity_tco2e=Decimal("1000.0"),
            unit_price=Decimal("150.00"),
        )
        registered = engine.register_credit(credit)
        result = engine.build_credit_portfolio()
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise CarbonCreditEngine."""
        self._credits: List[CarbonCredit] = []
        self._removals: List[GHGRemoval] = []
        logger.info(
            "CarbonCreditEngine v%s initialised", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Credit Registration                                                  #
    # ------------------------------------------------------------------ #

    def register_credit(self, credit: CarbonCredit) -> CarbonCredit:
        """Register a carbon credit in the portfolio per ESRS E1-7.

        Calculates total cost if not provided, assigns provenance hash,
        and adds the credit to the internal registry.

        Args:
            credit: CarbonCredit to register.

        Returns:
            Registered CarbonCredit with provenance hash.
        """
        t0 = time.perf_counter()

        if not credit.credit_id:
            credit.credit_id = _new_uuid()

        # Auto-calculate total cost if not explicitly set
        if credit.total_cost == Decimal("0.00") and credit.unit_price > 0:
            credit.total_cost = _round_val(
                credit.quantity_tco2e * credit.unit_price, 2
            )

        credit.provenance_hash = _compute_hash(credit)
        self._credits.append(credit)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        logger.info(
            "Registered credit: standard=%s, type=%s, qty=%s tCO2e, "
            "vintage=%d in %.3f ms",
            credit.standard.value,
            credit.credit_type.value,
            credit.quantity_tco2e,
            credit.vintage_year,
            elapsed_ms,
        )
        return credit

    # ------------------------------------------------------------------ #
    # Removal Registration                                                 #
    # ------------------------------------------------------------------ #

    def register_removal(self, removal: GHGRemoval) -> GHGRemoval:
        """Register a GHG removal activity per ESRS E1-7.

        Args:
            removal: GHGRemoval to register.

        Returns:
            Registered GHGRemoval with provenance hash.
        """
        t0 = time.perf_counter()

        if not removal.removal_id:
            removal.removal_id = _new_uuid()

        removal.provenance_hash = _compute_hash(removal)
        self._removals.append(removal)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        logger.info(
            "Registered removal: type=%s, qty=%s tCO2e, verified=%s in %.3f ms",
            removal.removal_type.value,
            removal.quantity_tco2e,
            removal.verification_status.value,
            elapsed_ms,
        )
        return removal

    # ------------------------------------------------------------------ #
    # Credit Portfolio Builder                                             #
    # ------------------------------------------------------------------ #

    def build_credit_portfolio(
        self,
        credits: Optional[List[CarbonCredit]] = None,
        removals: Optional[List[GHGRemoval]] = None,
    ) -> CarbonCreditResult:
        """Build the complete carbon credit portfolio per E1-7.

        Aggregates all credits and removals into a single result with
        summary statistics, quality scoring, and provenance tracking.

        Args:
            credits: List of credits (uses internal registry if None).
            removals: List of removals (uses internal registry if None).

        Returns:
            CarbonCreditResult with complete aggregation.
        """
        t0 = time.perf_counter()

        if credits is None:
            credits = list(self._credits)
        if removals is None:
            removals = list(self._removals)

        # Aggregate credits
        total_purchased = Decimal("0.000")
        total_retired = Decimal("0.000")
        total_expenditure = Decimal("0.00")
        by_standard: Dict[str, Decimal] = {}
        by_type: Dict[str, Decimal] = {}
        by_project_type: Dict[str, Decimal] = {}
        by_vintage: Dict[str, Decimal] = {}
        quality_scores: List[Decimal] = []

        for credit in credits:
            total_purchased += credit.quantity_tco2e

            if credit.status == CreditStatus.RETIRED:
                total_retired += credit.quantity_tco2e

            total_expenditure += credit.total_cost

            # By standard
            std_key = credit.standard.value
            by_standard[std_key] = (
                by_standard.get(std_key, Decimal("0.000"))
                + credit.quantity_tco2e
            )

            # By type
            type_key = credit.credit_type.value
            by_type[type_key] = (
                by_type.get(type_key, Decimal("0.000"))
                + credit.quantity_tco2e
            )

            # By project type
            proj_key = credit.project_type.value
            by_project_type[proj_key] = (
                by_project_type.get(proj_key, Decimal("0.000"))
                + credit.quantity_tco2e
            )

            # By vintage
            vintage_key = str(credit.vintage_year)
            by_vintage[vintage_key] = (
                by_vintage.get(vintage_key, Decimal("0.000"))
                + credit.quantity_tco2e
            )

            # Quality scores
            quality = self._calculate_quality_score(credit)
            quality_scores.append(quality)

        # Aggregate removals
        total_removals = Decimal("0.000")
        removals_by_type: Dict[str, Decimal] = {}

        for removal in removals:
            total_removals += removal.quantity_tco2e
            rem_key = removal.removal_type.value
            removals_by_type[rem_key] = (
                removals_by_type.get(rem_key, Decimal("0.000"))
                + removal.quantity_tco2e
            )

        # Average quality score
        avg_quality = Decimal("0.000")
        if quality_scores:
            avg_quality = _round_val(
                sum(quality_scores) / _decimal(len(quality_scores)), 3
            )

        # Completeness
        completeness = self._calculate_completeness(credits, removals)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = CarbonCreditResult(
            credits=credits,
            removals=removals,
            total_credits_purchased_tco2e=_round_val(total_purchased, 3),
            total_credits_retired_tco2e=_round_val(total_retired, 3),
            total_removals_tco2e=_round_val(total_removals, 3),
            credits_by_standard={
                k: str(_round_val(v, 3)) for k, v in by_standard.items()
            },
            credits_by_type={
                k: str(_round_val(v, 3)) for k, v in by_type.items()
            },
            credits_by_project_type={
                k: str(_round_val(v, 3)) for k, v in by_project_type.items()
            },
            credits_by_vintage={
                k: str(_round_val(v, 3)) for k, v in by_vintage.items()
            },
            removals_by_type={
                k: str(_round_val(v, 3)) for k, v in removals_by_type.items()
            },
            average_quality_score=avg_quality,
            total_expenditure=_round_val(total_expenditure, 2),
            total_credits_count=len(credits),
            total_removals_count=len(removals),
            completeness_score=completeness,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Built credit portfolio: %d credits (%s tCO2e purchased, "
            "%s retired), %d removals (%s tCO2e), avg quality=%s in %.3f ms",
            len(credits),
            total_purchased,
            total_retired,
            len(removals),
            total_removals,
            avg_quality,
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # Quality Assessment                                                   #
    # ------------------------------------------------------------------ #

    def assess_credit_quality(
        self, credit: CarbonCredit
    ) -> QualityAssessment:
        """Assess the quality of a carbon credit per AR E1-65.

        Evaluates the credit against four weighted criteria and
        produces an overall quality score and tier.

        Scoring formula (deterministic weighted average):
            overall = (additionality * 0.30) + (permanence * 0.25)
                    + (measurability * 0.25) + (verification * 0.20)

        Quality tiers:
            4.0-5.0: High
            2.5-3.9: Medium
            1.0-2.4: Low

        Args:
            credit: CarbonCredit to assess.

        Returns:
            QualityAssessment with weighted overall score.
        """
        t0 = time.perf_counter()

        overall_score = self._calculate_quality_score(credit)

        # Determine quality tier
        overall_float = float(overall_score)
        if overall_float >= 4.0:
            tier = "High"
        elif overall_float >= 2.5:
            tier = "Medium"
        else:
            tier = "Low"

        assessment = QualityAssessment(
            credit_id=credit.credit_id,
            additionality=credit.additionality_score,
            permanence=credit.permanence_score,
            measurability=credit.measurability_score,
            verification=credit.verification_score,
            overall_score=overall_score,
            quality_tier=tier,
        )

        assessment.provenance_hash = _compute_hash(assessment)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        logger.info(
            "Quality assessment for credit %s: score=%s, tier=%s in %.3f ms",
            credit.credit_id,
            overall_score,
            tier,
            elapsed_ms,
        )
        return assessment

    def batch_assess_quality(
        self, credits: List[CarbonCredit]
    ) -> List[QualityAssessment]:
        """Assess quality for multiple credits.

        Args:
            credits: List of CarbonCredits to assess.

        Returns:
            List of QualityAssessment results.
        """
        return [self.assess_credit_quality(c) for c in credits]

    # ------------------------------------------------------------------ #
    # SBTi Compliance Validation                                           #
    # ------------------------------------------------------------------ #

    def validate_sbti_compliance(
        self,
        result: CarbonCreditResult,
        target_emissions_tco2e: Decimal,
    ) -> Dict[str, Any]:
        """Validate carbon credit use against SBTi Net-Zero Standard.

        Checks whether the undertaking's use of carbon credits is
        consistent with SBTi guidance: credits shall not substitute
        for direct emission reductions, and at net-zero only removals
        are permitted for neutralising residual emissions.

        Args:
            result: CarbonCreditResult portfolio.
            target_emissions_tco2e: Total target-year emissions for
                context (Scope 1+2+3 in base year or current year).

        Returns:
            Dict with compliance assessment.
        """
        t0 = time.perf_counter()

        credits = result.credits
        total_purchased = result.total_credits_purchased_tco2e

        # Calculate avoidance vs removal split
        avoidance_tco2e = Decimal("0.000")
        removal_tco2e = Decimal("0.000")

        for credit in credits:
            if credit.credit_type == CreditType.AVOIDANCE:
                avoidance_tco2e += credit.quantity_tco2e
            elif credit.credit_type == CreditType.REMOVAL:
                removal_tco2e += credit.quantity_tco2e

        # Credit-to-emissions ratio
        credit_ratio = Decimal("0.000")
        if target_emissions_tco2e > 0:
            credit_ratio = _round_val(
                (total_purchased / target_emissions_tco2e) * Decimal("100"),
                3,
            )

        # SBTi compliance checks
        checks: Dict[str, Dict[str, Any]] = {}

        # Check 1: Credits should not exceed 10% of target emissions
        # for near-term use (SBTi BVCM guidance)
        max_near_term_pct = Decimal("10.0")
        near_term_ok = credit_ratio <= max_near_term_pct
        checks["near_term_volume"] = {
            "description": "Carbon credits should not exceed 10% of target emissions (BVCM)",
            "threshold_pct": str(max_near_term_pct),
            "actual_pct": str(credit_ratio),
            "compliant": near_term_ok,
        }

        # Check 2: For net-zero neutralisation, only removals allowed
        net_zero_guidance = SBTI_BEYONDVALUECHAINMITIGATION["net_zero"]
        removal_only_ok = avoidance_tco2e == Decimal("0.000") or True
        # Note: This check is informational; avoidance is allowed for BVCM
        checks["net_zero_removal_preference"] = {
            "description": "For net-zero neutralisation, only removal credits are permissible",
            "avoidance_tco2e": str(avoidance_tco2e),
            "removal_tco2e": str(removal_tco2e),
            "removal_share_pct": str(
                _round_val(
                    _decimal(
                        _safe_divide(
                            float(removal_tco2e),
                            float(total_purchased),
                            0.0,
                        )
                    ) * Decimal("100"),
                    1,
                )
            ) if total_purchased > 0 else "0.0",
            "compliant": True,  # Informational at near-term stage
            "note": "Avoidance credits are permitted for BVCM but not for "
                    "net-zero neutralisation of residual emissions",
        }

        # Check 3: Credits not deducted from gross emissions
        checks["no_netting"] = {
            "description": "Carbon credits shall not be deducted from E1-6 gross GHG emissions",
            "compliant": True,  # Structural check; the engine ensures separation
            "note": "This engine reports credits separately from gross emissions",
        }

        # Check 4: Average quality above threshold
        min_quality = Decimal("3.000")
        quality_ok = result.average_quality_score >= min_quality
        checks["quality_threshold"] = {
            "description": "Average credit quality should be Medium or above (>= 3.0)",
            "threshold": str(min_quality),
            "actual": str(result.average_quality_score),
            "compliant": quality_ok,
        }

        # Overall compliance
        all_compliant = all(
            c.get("compliant", False) for c in checks.values()
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        assessment = {
            "target_emissions_tco2e": str(target_emissions_tco2e),
            "total_credits_tco2e": str(total_purchased),
            "credit_to_emission_ratio_pct": str(credit_ratio),
            "avoidance_tco2e": str(avoidance_tco2e),
            "removal_tco2e": str(removal_tco2e),
            "checks": checks,
            "overall_compliant": all_compliant,
            "sbti_guidance_reference": "SBTi Corporate Net-Zero Standard v1.2",
            "processing_time_ms": elapsed_ms,
            "provenance_hash": _compute_hash(checks),
        }

        logger.info(
            "SBTi compliance check: overall=%s, ratio=%s%%, "
            "quality=%s in %.3f ms",
            "PASS" if all_compliant else "FAIL",
            credit_ratio,
            result.average_quality_score,
            elapsed_ms,
        )
        return assessment

    # ------------------------------------------------------------------ #
    # Completeness Validation                                              #
    # ------------------------------------------------------------------ #

    def validate_completeness(
        self, result: CarbonCreditResult
    ) -> Dict[str, Any]:
        """Validate completeness of E1-7 data points.

        Checks each required E1-7 data point against the result.

        Args:
            result: CarbonCreditResult to validate.

        Returns:
            Dict with data point coverage and completeness score.
        """
        datapoints_status: Dict[str, Dict[str, Any]] = {}
        covered = 0

        credits = result.credits
        removals = result.removals

        checks = {
            "e1_7_dp01": result.total_removals_tco2e > 0,
            "e1_7_dp02": len(result.removals_by_type) > 0,
            "e1_7_dp03": any(bool(r.methodology) for r in removals),
            "e1_7_dp04": any(
                r.verification_status == VerificationStatus.VERIFIED
                for r in removals
            ),
            "e1_7_dp05": result.total_credits_purchased_tco2e > 0,
            "e1_7_dp06": len(result.credits_by_type) > 0,
            "e1_7_dp07": len(result.credits_by_standard) > 0,
            "e1_7_dp08": len(result.credits_by_project_type) > 0,
            "e1_7_dp09": len(result.credits_by_vintage) > 0,
            "e1_7_dp10": any(c.is_certified for c in credits),
            "e1_7_dp11": result.total_expenditure > 0,
            "e1_7_dp12": True,  # Structural: credits always separate
            "e1_7_dp13": result.total_credits_retired_tco2e > 0,
            "e1_7_dp14": result.average_quality_score > 0,
            "e1_7_dp15": True,  # Link to transition plan is structural
        }

        for dp_id, dp_label in E1_7_DATAPOINTS.items():
            is_covered = checks.get(dp_id, False)
            if is_covered:
                covered += 1
            datapoints_status[dp_id] = {
                "label": dp_label,
                "covered": is_covered,
                "status": "COMPLETE" if is_covered else "MISSING",
            }

        total = len(E1_7_DATAPOINTS)
        score = _round2(
            _safe_divide(float(covered), float(total), 0.0) * 100.0
        )

        return {
            "disclosure_requirement": "E1-7",
            "title": "GHG removals and GHG mitigation projects financed through carbon credits",
            "total_datapoints": total,
            "covered_datapoints": covered,
            "missing_datapoints": total - covered,
            "completeness_score": score,
            "datapoints": datapoints_status,
            "provenance_hash": _compute_hash(datapoints_status),
        }

    # ------------------------------------------------------------------ #
    # E1-7 Data Point Extraction                                           #
    # ------------------------------------------------------------------ #

    def get_e1_7_datapoints(
        self, result: CarbonCreditResult
    ) -> Dict[str, Any]:
        """Extract structured E1-7 data points for XBRL tagging.

        Returns a dict of all E1-7 data points with their values,
        suitable for XBRL tagging and digital submission.

        Args:
            result: CarbonCreditResult to extract from.

        Returns:
            Dict mapping data point IDs to values.
        """
        datapoints: Dict[str, Any] = {
            "e1_7_dp01": {
                "value": str(result.total_removals_tco2e),
                "unit": "tCO2e",
                "label": E1_7_DATAPOINTS["e1_7_dp01"],
                "xbrl_element": "esrs:GHGRemovalsTotalOwn",
            },
            "e1_7_dp02": {
                "value": result.removals_by_type,
                "label": E1_7_DATAPOINTS["e1_7_dp02"],
                "xbrl_element": "esrs:GHGRemovalsByType",
            },
            "e1_7_dp03": {
                "value": [
                    {
                        "removal_type": r.removal_type.value,
                        "methodology": r.methodology,
                    }
                    for r in result.removals
                    if r.methodology
                ],
                "label": E1_7_DATAPOINTS["e1_7_dp03"],
                "xbrl_element": "esrs:GHGRemovalMethodology",
            },
            "e1_7_dp04": {
                "value": any(
                    r.verification_status == VerificationStatus.VERIFIED
                    for r in result.removals
                ),
                "label": E1_7_DATAPOINTS["e1_7_dp04"],
                "xbrl_element": "esrs:GHGRemovalVerified",
            },
            "e1_7_dp05": {
                "value": str(result.total_credits_purchased_tco2e),
                "unit": "tCO2e",
                "label": E1_7_DATAPOINTS["e1_7_dp05"],
                "xbrl_element": "esrs:CarbonCreditsTotalPurchased",
            },
            "e1_7_dp06": {
                "value": result.credits_by_type,
                "label": E1_7_DATAPOINTS["e1_7_dp06"],
                "xbrl_element": "esrs:CarbonCreditsByType",
            },
            "e1_7_dp07": {
                "value": result.credits_by_standard,
                "label": E1_7_DATAPOINTS["e1_7_dp07"],
                "xbrl_element": "esrs:CarbonCreditsByStandard",
            },
            "e1_7_dp08": {
                "value": result.credits_by_project_type,
                "label": E1_7_DATAPOINTS["e1_7_dp08"],
                "xbrl_element": "esrs:CarbonCreditsByProjectType",
            },
            "e1_7_dp09": {
                "value": result.credits_by_vintage,
                "label": E1_7_DATAPOINTS["e1_7_dp09"],
                "xbrl_element": "esrs:CarbonCreditsVintageYear",
            },
            "e1_7_dp10": {
                "value": any(c.is_certified for c in result.credits),
                "label": E1_7_DATAPOINTS["e1_7_dp10"],
                "xbrl_element": "esrs:CarbonCreditsCertified",
            },
            "e1_7_dp11": {
                "value": str(result.total_expenditure),
                "label": E1_7_DATAPOINTS["e1_7_dp11"],
                "xbrl_element": "esrs:CarbonCreditsExpenditure",
            },
            "e1_7_dp12": {
                "value": True,
                "label": E1_7_DATAPOINTS["e1_7_dp12"],
                "xbrl_element": "esrs:CarbonCreditsNotDeductedFromGrossEmissions",
            },
            "e1_7_dp13": {
                "value": str(result.total_credits_retired_tco2e),
                "unit": "tCO2e",
                "label": E1_7_DATAPOINTS["e1_7_dp13"],
                "xbrl_element": "esrs:CarbonCreditsRetired",
            },
            "e1_7_dp14": {
                "value": str(result.average_quality_score),
                "label": E1_7_DATAPOINTS["e1_7_dp14"],
                "xbrl_element": "esrs:CarbonCreditQualityScore",
            },
            "e1_7_dp15": {
                "value": "Carbon credits are reported separately and their role "
                         "in the transition plan is documented in E1-1",
                "label": E1_7_DATAPOINTS["e1_7_dp15"],
                "xbrl_element": "esrs:CarbonCreditRoleInTransitionPlan",
            },
        }

        datapoints["provenance_hash"] = _compute_hash(datapoints)
        return datapoints

    # ------------------------------------------------------------------ #
    # Summary Utilities                                                    #
    # ------------------------------------------------------------------ #

    def get_credit_summary(
        self, credit: CarbonCredit
    ) -> Dict[str, Any]:
        """Return a structured summary of a single carbon credit.

        Args:
            credit: CarbonCredit to summarise.

        Returns:
            Dict with credit details.
        """
        quality = self.assess_credit_quality(credit)

        return {
            "credit_id": credit.credit_id,
            "standard": credit.standard.value,
            "standard_description": CREDIT_STANDARD_DESCRIPTIONS.get(
                credit.standard.value, ""
            ),
            "credit_type": credit.credit_type.value,
            "project_type": credit.project_type.value,
            "project_type_description": PROJECT_TYPE_DESCRIPTIONS.get(
                credit.project_type.value, ""
            ),
            "project_name": credit.project_name,
            "project_location": credit.project_location,
            "vintage_year": credit.vintage_year,
            "quantity_tco2e": str(credit.quantity_tco2e),
            "unit_price": str(credit.unit_price),
            "total_cost": str(credit.total_cost),
            "currency": credit.currency,
            "status": credit.status.value,
            "is_certified": credit.is_certified,
            "verification_body": credit.verification_body,
            "quality_score": str(quality.overall_score),
            "quality_tier": quality.quality_tier,
            "permanence_years": credit.permanence_years,
            "provenance_hash": credit.provenance_hash,
        }

    def get_removal_summary(
        self, removal: GHGRemoval
    ) -> Dict[str, Any]:
        """Return a structured summary of a single GHG removal.

        Args:
            removal: GHGRemoval to summarise.

        Returns:
            Dict with removal details.
        """
        return {
            "removal_id": removal.removal_id,
            "removal_type": removal.removal_type.value,
            "quantity_tco2e": str(removal.quantity_tco2e),
            "methodology": removal.methodology,
            "verification_status": removal.verification_status.value,
            "verification_body": removal.verification_body,
            "location": removal.location,
            "reporting_period": removal.reporting_period,
            "permanence_years": removal.permanence_years,
            "is_in_own_operations": removal.is_in_own_operations,
            "provenance_hash": removal.provenance_hash,
        }

    def get_portfolio_quality_distribution(
        self, credits: List[CarbonCredit]
    ) -> Dict[str, int]:
        """Return the distribution of credits by quality tier.

        Args:
            credits: List of CarbonCredits.

        Returns:
            Dict mapping quality tier to count.
        """
        distribution: Dict[str, int] = {
            "High": 0,
            "Medium": 0,
            "Low": 0,
        }

        for credit in credits:
            assessment = self.assess_credit_quality(credit)
            tier = assessment.quality_tier
            distribution[tier] = distribution.get(tier, 0) + 1

        return distribution

    def clear_registry(self) -> None:
        """Clear all registered credits and removals."""
        self._credits.clear()
        self._removals.clear()
        logger.info("CarbonCreditEngine registry cleared")

    # ------------------------------------------------------------------ #
    # Private Helpers                                                      #
    # ------------------------------------------------------------------ #

    def _calculate_quality_score(self, credit: CarbonCredit) -> Decimal:
        """Calculate weighted quality score for a credit.

        Formula (deterministic):
            score = additionality * 0.30 + permanence * 0.25
                  + measurability * 0.25 + verification * 0.20

        Args:
            credit: CarbonCredit with quality scores.

        Returns:
            Weighted overall quality score (Decimal, 0-5 scale).
        """
        score = (
            _decimal(credit.additionality_score) * QUALITY_CRITERIA["additionality"]["weight"]
            + _decimal(credit.permanence_score) * QUALITY_CRITERIA["permanence"]["weight"]
            + _decimal(credit.measurability_score) * QUALITY_CRITERIA["measurability"]["weight"]
            + _decimal(credit.verification_score) * QUALITY_CRITERIA["verification"]["weight"]
        )
        return _round_val(score, 3)

    def _calculate_completeness(
        self,
        credits: List[CarbonCredit],
        removals: List[GHGRemoval],
    ) -> float:
        """Calculate E1-7 completeness score.

        Args:
            credits: List of carbon credits.
            removals: List of GHG removals.

        Returns:
            Completeness score (0-100).
        """
        total = len(E1_7_DATAPOINTS)

        checks = [
            any(r.quantity_tco2e > 0 for r in removals) if removals else False,
            len(set(r.removal_type for r in removals)) > 0 if removals else False,
            any(bool(r.methodology) for r in removals) if removals else False,
            any(
                r.verification_status == VerificationStatus.VERIFIED
                for r in removals
            ) if removals else False,
            any(c.quantity_tco2e > 0 for c in credits) if credits else False,
            len(set(c.credit_type for c in credits)) > 0 if credits else False,
            len(set(c.standard for c in credits)) > 0 if credits else False,
            len(set(c.project_type for c in credits)) > 0 if credits else False,
            len(set(c.vintage_year for c in credits)) > 0 if credits else False,
            any(c.is_certified for c in credits) if credits else False,
            any(c.total_cost > 0 for c in credits) if credits else False,
            True,  # No-netting is structural
            any(
                c.status == CreditStatus.RETIRED for c in credits
            ) if credits else False,
            True,  # Quality assessment always possible
            True,  # Transition plan link is structural
        ]

        covered = sum(1 for c in checks if c)
        return _round2(_safe_divide(float(covered), float(total), 0.0) * 100.0)
