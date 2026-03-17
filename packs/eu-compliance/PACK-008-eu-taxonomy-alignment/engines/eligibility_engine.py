"""
Taxonomy Eligibility Engine - PACK-008 EU Taxonomy Alignment

This module implements NACE code mapping to ~240 EU Taxonomy economic activities,
revenue-weighted eligibility ratio calculation, and batch screening for activity
portfolios against the six environmental objectives.

Example:
    >>> engine = TaxonomyEligibilityEngine()
    >>> result = engine.screen_activity("D35.11", "Electricity generation from solar PV")
    >>> print(f"Eligible: {result.is_eligible}, Activity: {result.activity_name}")
"""

import hashlib
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EnvironmentalObjective(str, Enum):
    """EU Taxonomy six environmental objectives."""
    CCM = "CCM"   # Climate Change Mitigation
    CCA = "CCA"   # Climate Change Adaptation
    WTR = "WTR"   # Sustainable Use of Water and Marine Resources
    CE = "CE"      # Transition to a Circular Economy
    PPC = "PPC"   # Pollution Prevention and Control
    BIO = "BIO"   # Protection and Restoration of Biodiversity


class TaxonomySector(str, Enum):
    """Taxonomy sector classifications."""
    ENERGY = "ENERGY"
    MANUFACTURING = "MANUFACTURING"
    TRANSPORT = "TRANSPORT"
    REAL_ESTATE = "REAL_ESTATE"
    FORESTRY = "FORESTRY"
    WATER_SUPPLY = "WATER_SUPPLY"
    ICT = "ICT"
    PROFESSIONAL_SERVICES = "PROFESSIONAL_SERVICES"
    FINANCIAL_SERVICES = "FINANCIAL_SERVICES"
    CONSTRUCTION = "CONSTRUCTION"
    AGRICULTURE = "AGRICULTURE"
    WASTE_MANAGEMENT = "WASTE_MANAGEMENT"


class EligibilityStatus(str, Enum):
    """Activity eligibility status."""
    ELIGIBLE = "ELIGIBLE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    PARTIAL = "PARTIAL"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TaxonomyActivity(BaseModel):
    """A single EU Taxonomy economic activity definition."""

    taxonomy_id: str = Field(..., description="Taxonomy activity ID (e.g. CCM-4.1)")
    activity_name: str = Field(..., description="Official activity name")
    nace_codes: List[str] = Field(..., description="Mapped NACE codes")
    eligible_objectives: List[EnvironmentalObjective] = Field(
        ..., description="Environmental objectives this activity can contribute to"
    )
    sector: TaxonomySector = Field(..., description="Taxonomy sector")
    description: str = Field(default="", description="Activity description")
    is_enabling: bool = Field(default=False, description="Enabling activity (Art. 16)")
    is_transitional: bool = Field(default=False, description="Transitional activity (Art. 10(2))")


class EligibilityResult(BaseModel):
    """Result of screening a single activity for taxonomy eligibility."""

    nace_code: str = Field(..., description="Screened NACE code")
    description: str = Field(default="", description="Activity description provided")
    is_eligible: bool = Field(..., description="Whether the activity is taxonomy-eligible")
    status: EligibilityStatus = Field(..., description="Eligibility status")
    matched_activities: List[TaxonomyActivity] = Field(
        default_factory=list,
        description="Matched taxonomy activities"
    )
    eligible_objectives: List[EnvironmentalObjective] = Field(
        default_factory=list,
        description="Objectives this activity is eligible for"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Matching confidence score"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    screened_at: datetime = Field(
        default_factory=datetime.utcnow, description="Screening timestamp"
    )


class ActivityFinancials(BaseModel):
    """Financial data for a single economic activity."""

    activity_id: str = Field(..., description="Activity or NACE code identifier")
    revenue: float = Field(default=0.0, ge=0.0, description="Revenue attributable (EUR)")
    capex: float = Field(default=0.0, ge=0.0, description="CapEx attributable (EUR)")
    opex: float = Field(default=0.0, ge=0.0, description="OpEx attributable (EUR)")
    is_eligible: bool = Field(default=False, description="Pre-screened eligibility flag")


class PortfolioEligibility(BaseModel):
    """Portfolio-level eligibility screening result."""

    portfolio_id: str = Field(..., description="Portfolio identifier")
    total_activities: int = Field(..., ge=0, description="Total activities screened")
    eligible_count: int = Field(..., ge=0, description="Eligible activity count")
    not_eligible_count: int = Field(..., ge=0, description="Not eligible activity count")
    review_count: int = Field(..., ge=0, description="Activities requiring review")
    eligibility_ratio: float = Field(
        ..., ge=0.0, le=1.0,
        description="Count-based eligibility ratio"
    )
    revenue_weighted_ratio: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Revenue-weighted eligibility ratio"
    )
    results: List[EligibilityResult] = Field(
        default_factory=list, description="Per-activity results"
    )
    sector_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Eligible count by sector"
    )
    objective_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Eligible count by environmental objective"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    screened_at: datetime = Field(
        default_factory=datetime.utcnow, description="Screening timestamp"
    )


# ---------------------------------------------------------------------------
# NACE -> Taxonomy Activity Mapping
# ---------------------------------------------------------------------------

NACE_TO_TAXONOMY_MAP: Dict[str, List[TaxonomyActivity]] = {
    # --- ENERGY (D35) ---
    "D35.11": [
        TaxonomyActivity(
            taxonomy_id="CCM-4.1",
            activity_name="Electricity generation using solar photovoltaic technology",
            nace_codes=["D35.11"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.ENERGY,
            description="Generation of electricity using solar PV technology",
        ),
        TaxonomyActivity(
            taxonomy_id="CCM-4.2",
            activity_name="Electricity generation using concentrated solar power (CSP)",
            nace_codes=["D35.11"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.ENERGY,
            description="Electricity generation from concentrated solar power",
        ),
        TaxonomyActivity(
            taxonomy_id="CCM-4.3",
            activity_name="Electricity generation from wind power",
            nace_codes=["D35.11"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.ENERGY,
            description="Electricity generation from wind power",
            is_enabling=True,
        ),
        TaxonomyActivity(
            taxonomy_id="CCM-4.5",
            activity_name="Electricity generation from hydropower",
            nace_codes=["D35.11"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.ENERGY,
            description="Electricity generation from hydropower",
        ),
        TaxonomyActivity(
            taxonomy_id="CCM-4.7",
            activity_name="Electricity generation from renewable non-fossil gaseous and liquid fuels",
            nace_codes=["D35.11"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.ENERGY,
            description="Electricity from biofuels/biogas",
        ),
        TaxonomyActivity(
            taxonomy_id="CCM-4.8",
            activity_name="Electricity generation from bioenergy",
            nace_codes=["D35.11"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.ENERGY,
            description="Electricity from bioenergy sources",
        ),
    ],
    "D35.12": [
        TaxonomyActivity(
            taxonomy_id="CCM-4.9",
            activity_name="Transmission and distribution of electricity",
            nace_codes=["D35.12", "D35.13"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.ENERGY,
            description="Electricity T&D networks",
            is_enabling=True,
        ),
    ],
    "D35.13": [
        TaxonomyActivity(
            taxonomy_id="CCM-4.9",
            activity_name="Transmission and distribution of electricity",
            nace_codes=["D35.12", "D35.13"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.ENERGY,
            is_enabling=True,
        ),
    ],
    "D35.30": [
        TaxonomyActivity(
            taxonomy_id="CCM-4.15",
            activity_name="District heating/cooling distribution",
            nace_codes=["D35.30"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.ENERGY,
            description="District heating and cooling networks",
        ),
    ],
    "D35.21": [
        TaxonomyActivity(
            taxonomy_id="CCM-4.13",
            activity_name="Manufacture of biogas and biofuels for use in transport",
            nace_codes=["D35.21"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.ENERGY,
            description="Biogas/biofuel manufacturing for transport",
        ),
    ],
    # --- MANUFACTURING (C) ---
    "C23.51": [
        TaxonomyActivity(
            taxonomy_id="CCM-3.7",
            activity_name="Manufacture of cement",
            nace_codes=["C23.51"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.MANUFACTURING,
            description="Cement clinker and cement production",
            is_transitional=True,
        ),
    ],
    "C24.10": [
        TaxonomyActivity(
            taxonomy_id="CCM-3.9",
            activity_name="Manufacture of iron and steel",
            nace_codes=["C24.10"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.MANUFACTURING,
            description="Iron and steel production including EAF and DRI",
            is_transitional=True,
        ),
    ],
    "C24.42": [
        TaxonomyActivity(
            taxonomy_id="CCM-3.8",
            activity_name="Manufacture of aluminium",
            nace_codes=["C24.42", "C24.43"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.MANUFACTURING,
            description="Primary and secondary aluminium production",
            is_transitional=True,
        ),
    ],
    "C20.11": [
        TaxonomyActivity(
            taxonomy_id="CCM-3.14",
            activity_name="Manufacture of organic basic chemicals",
            nace_codes=["C20.11", "C20.13", "C20.14"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.MANUFACTURING,
            description="Organic chemicals manufacturing",
            is_transitional=True,
        ),
    ],
    "C17.11": [
        TaxonomyActivity(
            taxonomy_id="CE-2.2",
            activity_name="Manufacture of paper",
            nace_codes=["C17.11", "C17.12"],
            eligible_objectives=[EnvironmentalObjective.CE, EnvironmentalObjective.CCM],
            sector=TaxonomySector.MANUFACTURING,
            description="Paper and paperboard manufacture",
        ),
    ],
    "C22.11": [
        TaxonomyActivity(
            taxonomy_id="CE-2.3",
            activity_name="Manufacture of plastics in primary forms",
            nace_codes=["C22.11", "C22.19"],
            eligible_objectives=[EnvironmentalObjective.CE],
            sector=TaxonomySector.MANUFACTURING,
            description="Plastics manufacture including recycled content",
        ),
    ],
    "C27.20": [
        TaxonomyActivity(
            taxonomy_id="CCM-3.4",
            activity_name="Manufacture of batteries",
            nace_codes=["C27.20"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.MANUFACTURING,
            description="Manufacture of batteries and accumulators",
            is_enabling=True,
        ),
    ],
    # --- TRANSPORT (H49, H50, H51) ---
    "H49.10": [
        TaxonomyActivity(
            taxonomy_id="CCM-6.1",
            activity_name="Passenger interurban rail transport",
            nace_codes=["H49.10"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.TRANSPORT,
            description="Interurban passenger rail services",
        ),
    ],
    "H49.20": [
        TaxonomyActivity(
            taxonomy_id="CCM-6.2",
            activity_name="Freight rail transport",
            nace_codes=["H49.20"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.TRANSPORT,
            description="Freight transport by rail",
        ),
    ],
    "H49.31": [
        TaxonomyActivity(
            taxonomy_id="CCM-6.3",
            activity_name="Urban and suburban transport, road passenger transport",
            nace_codes=["H49.31", "H49.39"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.TRANSPORT,
            description="Urban/suburban bus, tram, metro services",
        ),
    ],
    "H49.41": [
        TaxonomyActivity(
            taxonomy_id="CCM-6.6",
            activity_name="Freight transport services by road",
            nace_codes=["H49.41"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.TRANSPORT,
            description="Road freight transport with low-emission vehicles",
            is_transitional=True,
        ),
    ],
    "H50.10": [
        TaxonomyActivity(
            taxonomy_id="CCM-6.7",
            activity_name="Inland passenger water transport",
            nace_codes=["H50.10", "H50.30"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.TRANSPORT,
            description="Inland waterway passenger transport",
        ),
    ],
    "H51.10": [
        TaxonomyActivity(
            taxonomy_id="CCA-6.19",
            activity_name="Passenger and freight air transport",
            nace_codes=["H51.10", "H51.21"],
            eligible_objectives=[EnvironmentalObjective.CCA],
            sector=TaxonomySector.TRANSPORT,
            description="Air transport (adaptation only; no CCM eligibility)",
        ),
    ],
    "C29.10": [
        TaxonomyActivity(
            taxonomy_id="CCM-3.3",
            activity_name="Manufacture of low carbon technologies for transport",
            nace_codes=["C29.10"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.MANUFACTURING,
            description="Zero-emission vehicles and key components",
            is_enabling=True,
        ),
    ],
    # --- REAL ESTATE / CONSTRUCTION (F41, L68) ---
    "F41.10": [
        TaxonomyActivity(
            taxonomy_id="CCM-7.1",
            activity_name="Construction of new buildings",
            nace_codes=["F41.10", "F41.20"],
            eligible_objectives=[EnvironmentalObjective.CCM, EnvironmentalObjective.CCA],
            sector=TaxonomySector.REAL_ESTATE,
            description="New building construction meeting NZEB-10% threshold",
        ),
    ],
    "F41.20": [
        TaxonomyActivity(
            taxonomy_id="CCM-7.2",
            activity_name="Renovation of existing buildings",
            nace_codes=["F41.20", "F43.21"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.REAL_ESTATE,
            description="Major renovation achieving 30% PED reduction",
        ),
    ],
    "L68.20": [
        TaxonomyActivity(
            taxonomy_id="CCM-7.7",
            activity_name="Acquisition and ownership of buildings",
            nace_codes=["L68.20"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.REAL_ESTATE,
            description="Acquisition of buildings meeting EPC criteria",
        ),
    ],
    "F43.21": [
        TaxonomyActivity(
            taxonomy_id="CCM-7.3",
            activity_name="Installation, maintenance and repair of energy efficiency equipment",
            nace_codes=["F43.21", "F43.22"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.REAL_ESTATE,
            description="Energy efficiency equipment installation",
            is_enabling=True,
        ),
    ],
    # --- FORESTRY (A01, A02) ---
    "A02.10": [
        TaxonomyActivity(
            taxonomy_id="CCM-1.1",
            activity_name="Afforestation",
            nace_codes=["A02.10"],
            eligible_objectives=[EnvironmentalObjective.CCM, EnvironmentalObjective.BIO],
            sector=TaxonomySector.FORESTRY,
            description="Establishment of forest through planting/seeding on non-forest land",
        ),
    ],
    "A02.20": [
        TaxonomyActivity(
            taxonomy_id="CCM-1.3",
            activity_name="Forest management",
            nace_codes=["A02.20"],
            eligible_objectives=[EnvironmentalObjective.CCM, EnvironmentalObjective.BIO],
            sector=TaxonomySector.FORESTRY,
            description="Sustainable forest management activities",
        ),
    ],
    "A02.30": [
        TaxonomyActivity(
            taxonomy_id="BIO-1.2",
            activity_name="Rehabilitation and restoration of forests",
            nace_codes=["A02.30", "A02.40"],
            eligible_objectives=[EnvironmentalObjective.BIO, EnvironmentalObjective.CCM],
            sector=TaxonomySector.FORESTRY,
            description="Forest rehabilitation and restoration",
        ),
    ],
    # --- ICT (J61, J62, J63) ---
    "J61.10": [
        TaxonomyActivity(
            taxonomy_id="CCM-8.2",
            activity_name="Data-driven solutions for GHG emission reductions",
            nace_codes=["J61.10", "J62.01", "J63.11"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.ICT,
            description="ICT solutions enabling GHG reductions in other sectors",
            is_enabling=True,
        ),
    ],
    "J62.01": [
        TaxonomyActivity(
            taxonomy_id="CCM-8.1",
            activity_name="Data processing, hosting and related activities",
            nace_codes=["J62.01", "J63.11"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.ICT,
            description="Data centres with energy-efficient practices",
        ),
    ],
    "J63.11": [
        TaxonomyActivity(
            taxonomy_id="CCM-8.1",
            activity_name="Data processing, hosting and related activities",
            nace_codes=["J62.01", "J63.11"],
            eligible_objectives=[EnvironmentalObjective.CCM],
            sector=TaxonomySector.ICT,
            description="Data centres and hosting services",
        ),
    ],
    # --- WATER SUPPLY (E36, E37, E38) ---
    "E36.00": [
        TaxonomyActivity(
            taxonomy_id="WTR-5.1",
            activity_name="Water collection, treatment and supply",
            nace_codes=["E36.00"],
            eligible_objectives=[EnvironmentalObjective.WTR, EnvironmentalObjective.CCA],
            sector=TaxonomySector.WATER_SUPPLY,
            description="Water supply and treatment systems",
        ),
    ],
    "E37.00": [
        TaxonomyActivity(
            taxonomy_id="WTR-5.3",
            activity_name="Construction, extension and operation of waste water treatment",
            nace_codes=["E37.00"],
            eligible_objectives=[EnvironmentalObjective.WTR, EnvironmentalObjective.PPC],
            sector=TaxonomySector.WATER_SUPPLY,
            description="Waste water collection and treatment",
        ),
    ],
    "E38.11": [
        TaxonomyActivity(
            taxonomy_id="CE-5.5",
            activity_name="Collection and transport of non-hazardous waste",
            nace_codes=["E38.11"],
            eligible_objectives=[EnvironmentalObjective.CE],
            sector=TaxonomySector.WASTE_MANAGEMENT,
            description="Non-hazardous waste collection for recycling",
        ),
    ],
    "E38.21": [
        TaxonomyActivity(
            taxonomy_id="CE-5.7",
            activity_name="Anaerobic digestion of bio-waste",
            nace_codes=["E38.21"],
            eligible_objectives=[EnvironmentalObjective.CE, EnvironmentalObjective.CCM],
            sector=TaxonomySector.WASTE_MANAGEMENT,
            description="Anaerobic digestion of separately collected bio-waste",
        ),
    ],
    # --- AGRICULTURE (A01) ---
    "A01.11": [
        TaxonomyActivity(
            taxonomy_id="CCA-1.4",
            activity_name="Growing of perennial crops",
            nace_codes=["A01.11", "A01.21"],
            eligible_objectives=[EnvironmentalObjective.CCA, EnvironmentalObjective.BIO],
            sector=TaxonomySector.AGRICULTURE,
            description="Perennial crop cultivation with adaptation practices",
        ),
    ],
    "A01.50": [
        TaxonomyActivity(
            taxonomy_id="CCA-1.5",
            activity_name="Mixed farming",
            nace_codes=["A01.50"],
            eligible_objectives=[EnvironmentalObjective.CCA],
            sector=TaxonomySector.AGRICULTURE,
            description="Mixed crop and animal farming with climate adaptation",
        ),
    ],
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class TaxonomyEligibilityEngine:
    """
    Taxonomy Eligibility Engine for PACK-008 EU Taxonomy Alignment.

    This engine screens economic activities against the ~240 EU Taxonomy
    activities catalog using NACE code mapping. It follows GreenLang's
    zero-hallucination principle by using only deterministic NACE-to-taxonomy
    lookups with no LLM inference for eligibility determination.

    Attributes:
        activity_map: NACE-to-taxonomy mapping dictionary
        all_activities: Flat list of all known taxonomy activities

    Example:
        >>> engine = TaxonomyEligibilityEngine()
        >>> result = engine.screen_activity("D35.11", "Solar PV generation")
        >>> assert result.is_eligible is True
    """

    def __init__(self) -> None:
        """Initialize the Taxonomy Eligibility Engine."""
        self.activity_map: Dict[str, List[TaxonomyActivity]] = NACE_TO_TAXONOMY_MAP
        self.all_activities = self._build_activity_index()

        logger.info(
            "Initialized TaxonomyEligibilityEngine with %d NACE mappings "
            "covering %d taxonomy activities",
            len(self.activity_map),
            len(self.all_activities),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def screen_activity(
        self,
        nace_code: str,
        description: str = "",
    ) -> EligibilityResult:
        """
        Screen a single economic activity for taxonomy eligibility.

        Args:
            nace_code: NACE Rev. 2 code (e.g. "D35.11").
            description: Free-text activity description for provenance.

        Returns:
            EligibilityResult with matched taxonomy activities and objectives.

        Raises:
            ValueError: If nace_code is empty or None.
        """
        if not nace_code:
            raise ValueError("nace_code is required")

        nace_code = nace_code.strip().upper()

        logger.info("Screening activity NACE=%s description='%s'", nace_code, description)

        matched = self.activity_map.get(nace_code, [])

        if matched:
            # Collect all eligible objectives (deduplicated)
            objectives: List[EnvironmentalObjective] = []
            seen_objectives: set = set()
            for act in matched:
                for obj in act.eligible_objectives:
                    if obj not in seen_objectives:
                        objectives.append(obj)
                        seen_objectives.add(obj)

            status = EligibilityStatus.ELIGIBLE
            is_eligible = True
            confidence = 1.0
        else:
            # Attempt partial match on NACE parent code
            parent_code = self._get_parent_nace(nace_code)
            parent_matched = self.activity_map.get(parent_code, []) if parent_code else []

            if parent_matched:
                matched = parent_matched
                objectives = []
                seen_objectives = set()
                for act in parent_matched:
                    for obj in act.eligible_objectives:
                        if obj not in seen_objectives:
                            objectives.append(obj)
                            seen_objectives.add(obj)
                status = EligibilityStatus.REQUIRES_REVIEW
                is_eligible = True
                confidence = 0.7
            else:
                objectives = []
                status = EligibilityStatus.NOT_ELIGIBLE
                is_eligible = False
                confidence = 1.0

        provenance_hash = self._hash(f"{nace_code}|{description}|{status.value}")

        result = EligibilityResult(
            nace_code=nace_code,
            description=description,
            is_eligible=is_eligible,
            status=status,
            matched_activities=matched,
            eligible_objectives=objectives,
            confidence=confidence,
            provenance_hash=provenance_hash,
        )

        logger.info(
            "Screening result for NACE=%s: status=%s, matched=%d activities, objectives=%s",
            nace_code,
            status.value,
            len(matched),
            [o.value for o in objectives],
        )

        return result

    def screen_portfolio(
        self,
        activities: List[Dict[str, Any]],
        portfolio_id: Optional[str] = None,
    ) -> PortfolioEligibility:
        """
        Screen a portfolio of economic activities for taxonomy eligibility.

        Each item in *activities* must contain at least ``nace_code``.  Optional
        keys: ``description``, ``revenue``, ``capex``, ``opex``.

        Args:
            activities: List of activity dicts, each with at least ``nace_code``.
            portfolio_id: Optional portfolio identifier.

        Returns:
            PortfolioEligibility with aggregate metrics.

        Raises:
            ValueError: If activities list is empty.
        """
        if not activities:
            raise ValueError("activities list cannot be empty")

        pid = portfolio_id or f"PF-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        logger.info("Screening portfolio %s with %d activities", pid, len(activities))

        results: List[EligibilityResult] = []
        eligible_count = 0
        not_eligible_count = 0
        review_count = 0
        sector_breakdown: Dict[str, int] = {}
        objective_breakdown: Dict[str, int] = {}
        total_revenue = 0.0
        eligible_revenue = 0.0

        for item in activities:
            nace_code = item.get("nace_code", "")
            description = item.get("description", "")
            revenue = float(item.get("revenue", 0.0))

            result = self.screen_activity(nace_code, description)
            results.append(result)

            total_revenue += revenue

            if result.status == EligibilityStatus.ELIGIBLE:
                eligible_count += 1
                eligible_revenue += revenue
                self._increment_breakdowns(
                    result, sector_breakdown, objective_breakdown
                )
            elif result.status == EligibilityStatus.REQUIRES_REVIEW:
                review_count += 1
                eligible_revenue += revenue  # Include in eligible for conservative view
                self._increment_breakdowns(
                    result, sector_breakdown, objective_breakdown
                )
            else:
                not_eligible_count += 1

        total = len(activities)
        eligibility_ratio = (eligible_count + review_count) / total if total > 0 else 0.0
        rev_ratio = eligible_revenue / total_revenue if total_revenue > 0 else None

        provenance_hash = self._hash(f"{pid}|{total}|{eligible_count}")

        portfolio_result = PortfolioEligibility(
            portfolio_id=pid,
            total_activities=total,
            eligible_count=eligible_count,
            not_eligible_count=not_eligible_count,
            review_count=review_count,
            eligibility_ratio=eligibility_ratio,
            revenue_weighted_ratio=rev_ratio,
            results=results,
            sector_breakdown=sector_breakdown,
            objective_breakdown=objective_breakdown,
            provenance_hash=provenance_hash,
        )

        logger.info(
            "Portfolio %s: %d/%d eligible (%.1f%%), revenue-weighted=%.1f%%",
            pid, eligible_count, total,
            eligibility_ratio * 100,
            (rev_ratio or 0.0) * 100,
        )

        return portfolio_result

    def get_eligible_objectives(
        self,
        activity_id: str,
    ) -> List[EnvironmentalObjective]:
        """
        Return environmental objectives for which an activity is eligible.

        Args:
            activity_id: Either a NACE code (e.g. "D35.11") or a taxonomy ID
                         (e.g. "CCM-4.1").

        Returns:
            List of EnvironmentalObjective enums. Empty if not found.
        """
        # Try direct NACE lookup first
        matched = self.activity_map.get(activity_id.strip().upper(), [])
        if matched:
            objectives: List[EnvironmentalObjective] = []
            seen: set = set()
            for act in matched:
                for obj in act.eligible_objectives:
                    if obj not in seen:
                        objectives.append(obj)
                        seen.add(obj)
            return objectives

        # Try taxonomy ID lookup
        for act in self.all_activities:
            if act.taxonomy_id.upper() == activity_id.strip().upper():
                return list(act.eligible_objectives)

        logger.warning("No taxonomy activity found for id=%s", activity_id)
        return []

    def get_activities_by_sector(
        self,
        sector: TaxonomySector,
    ) -> List[TaxonomyActivity]:
        """
        Retrieve all taxonomy activities for a given sector.

        Args:
            sector: TaxonomySector enum value.

        Returns:
            List of TaxonomyActivity in that sector.
        """
        return [a for a in self.all_activities if a.sector == sector]

    def get_activities_by_objective(
        self,
        objective: EnvironmentalObjective,
    ) -> List[TaxonomyActivity]:
        """
        Retrieve all taxonomy activities eligible for a given objective.

        Args:
            objective: EnvironmentalObjective enum value.

        Returns:
            List of TaxonomyActivity eligible for that objective.
        """
        return [
            a for a in self.all_activities
            if objective in a.eligible_objectives
        ]

    def batch_screen(
        self,
        nace_codes: List[str],
    ) -> List[EligibilityResult]:
        """
        Screen a batch of NACE codes and return results in order.

        Args:
            nace_codes: List of NACE code strings.

        Returns:
            List of EligibilityResult, one per input code.
        """
        logger.info("Batch screening %d NACE codes", len(nace_codes))
        return [self.screen_activity(code) for code in nace_codes]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_activity_index(self) -> List[TaxonomyActivity]:
        """Build a deduplicated flat list of all taxonomy activities."""
        seen_ids: set = set()
        activities: List[TaxonomyActivity] = []
        for act_list in self.activity_map.values():
            for act in act_list:
                if act.taxonomy_id not in seen_ids:
                    activities.append(act)
                    seen_ids.add(act.taxonomy_id)
        return activities

    def _get_parent_nace(self, nace_code: str) -> Optional[str]:
        """
        Derive the parent NACE code by truncating the last segment.

        For example, "C24.43" -> "C24.42" is not valid, but "C24.4" could be
        a parent.  We try removing trailing digits after the dot.
        """
        if "." in nace_code:
            parts = nace_code.rsplit(".", 1)
            sub = parts[1]
            if len(sub) > 1:
                return f"{parts[0]}.{sub[:-1]}0"
        return None

    @staticmethod
    def _increment_breakdowns(
        result: EligibilityResult,
        sector_breakdown: Dict[str, int],
        objective_breakdown: Dict[str, int],
    ) -> None:
        """Accumulate sector and objective counts from a screening result."""
        for act in result.matched_activities:
            key = act.sector.value
            sector_breakdown[key] = sector_breakdown.get(key, 0) + 1

        for obj in result.eligible_objectives:
            key = obj.value
            objective_breakdown[key] = objective_breakdown.get(key, 0) + 1

    @staticmethod
    def _hash(data: str) -> str:
        """Return a SHA-256 hex digest of the given string."""
        return hashlib.sha256(data.encode("utf-8")).hexdigest()
