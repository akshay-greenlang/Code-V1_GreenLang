"""
GL-TCFD-APP Transition Risk API

Manages transition risk assessment across four TCFD sub-categories:
policy/regulatory, technology, market, and reputation.  Provides composite
scoring, sector transition profiles, carbon cost exposure, asset stranding
analysis, regulation timeline tracking, and disclosure text generation.

Transition Risk Sub-categories:
    - Policy & Legal: Carbon pricing, emissions mandates, litigation risk
    - Technology: Substitution, disruption, adoption costs
    - Market: Demand shifts, commodity price changes, stranded revenue
    - Reputation: Consumer preference, stakeholder pressure, ESG ratings

ISSB/IFRS S2 references: paragraphs 10-12 (transition risks).
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/tcfd/transition-risk", tags=["Transition Risk"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TransitionRiskCategory(str, Enum):
    POLICY = "policy"
    TECHNOLOGY = "technology"
    MARKET = "market"
    REPUTATION = "reputation"


class PolicyRiskDriver(str, Enum):
    CARBON_TAX = "carbon_tax"
    ETS_SCHEME = "ets_scheme"
    CBAM = "cbam"
    EMISSIONS_STANDARD = "emissions_standard"
    MANDATORY_DISCLOSURE = "mandatory_disclosure"
    LITIGATION = "litigation"
    SUBSIDY_REMOVAL = "subsidy_removal"


class TechRiskDriver(str, Enum):
    RENEWABLE_SUBSTITUTION = "renewable_substitution"
    ELECTRIFICATION = "electrification"
    HYDROGEN_ECONOMY = "hydrogen_economy"
    CCS_REQUIREMENT = "ccs_requirement"
    DIGITAL_DISRUPTION = "digital_disruption"
    EFFICIENCY_MANDATE = "efficiency_mandate"


class MarketRiskDriver(str, Enum):
    DEMAND_SHIFT = "demand_shift"
    COMMODITY_PRICE = "commodity_price"
    RAW_MATERIAL_COST = "raw_material_cost"
    CONSUMER_PREFERENCE = "consumer_preference"
    GREEN_COMPETITION = "green_competition"


class ReputationRiskDriver(str, Enum):
    ESG_RATING_DOWNGRADE = "esg_rating_downgrade"
    GREENWASHING_ALLEGATION = "greenwashing_allegation"
    INVESTOR_DIVESTMENT = "investor_divestment"
    EMPLOYEE_RETENTION = "employee_retention"
    COMMUNITY_OPPOSITION = "community_opposition"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class AssessPolicyRiskRequest(BaseModel):
    """Request to assess policy/regulatory transition risk."""
    org_id: str = Field(..., description="Organization ID")
    risk_drivers: List[PolicyRiskDriver] = Field(..., description="Applicable policy risk drivers")
    current_carbon_cost_usd_per_tco2e: float = Field(0.0, ge=0, description="Current carbon cost")
    scope1_emissions_tco2e: float = Field(0.0, ge=0, description="Annual Scope 1 emissions")
    scope2_emissions_tco2e: float = Field(0.0, ge=0, description="Annual Scope 2 emissions")
    jurisdictions: List[str] = Field(default_factory=list, description="Operating jurisdictions")
    annual_revenue_usd: float = Field(0.0, ge=0, description="Annual revenue")

    class Config:
        json_schema_extra = {
            "example": {
                "org_id": "org_abc123",
                "risk_drivers": ["carbon_tax", "cbam", "mandatory_disclosure"],
                "current_carbon_cost_usd_per_tco2e": 30,
                "scope1_emissions_tco2e": 25000,
                "scope2_emissions_tco2e": 15000,
                "jurisdictions": ["EU", "UK", "US"],
                "annual_revenue_usd": 500000000,
            }
        }


class AssessTechRiskRequest(BaseModel):
    """Request to assess technology transition risk."""
    org_id: str = Field(..., description="Organization ID")
    risk_drivers: List[TechRiskDriver] = Field(..., description="Applicable technology risk drivers")
    technology_capex_usd: float = Field(0.0, ge=0, description="Required technology transition capex")
    stranded_tech_value_usd: float = Field(0.0, ge=0, description="Value of potentially stranded technology")
    rd_budget_pct_of_revenue: float = Field(0.0, ge=0, le=100, description="R&D budget as % of revenue")
    sector: str = Field("", description="Industry sector")


class AssessMarketRiskRequest(BaseModel):
    """Request to assess market transition risk."""
    org_id: str = Field(..., description="Organization ID")
    risk_drivers: List[MarketRiskDriver] = Field(..., description="Applicable market risk drivers")
    carbon_intensive_revenue_pct: float = Field(0.0, ge=0, le=100, description="% revenue from carbon-intensive products")
    annual_revenue_usd: float = Field(0.0, ge=0, description="Annual revenue")
    input_cost_exposure_usd: float = Field(0.0, ge=0, description="Annual input cost exposed to climate")
    sector: str = Field("", description="Industry sector")


class AssessReputationRiskRequest(BaseModel):
    """Request to assess reputation transition risk."""
    org_id: str = Field(..., description="Organization ID")
    risk_drivers: List[ReputationRiskDriver] = Field(..., description="Applicable reputation risk drivers")
    current_esg_score: float = Field(50.0, ge=0, le=100, description="Current ESG score (0-100)")
    has_net_zero_target: bool = Field(False, description="Has a net zero commitment")
    has_sbti_target: bool = Field(False, description="Has SBTi validated target")
    brand_value_usd: float = Field(0.0, ge=0, description="Estimated brand value")


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class TransitionRiskAssessmentResponse(BaseModel):
    """Transition risk assessment result."""
    assessment_id: str
    org_id: str
    category: str
    risk_score: float
    risk_rating: str
    risk_drivers: List[str]
    financial_exposure_usd: float
    revenue_at_risk_pct: float
    key_findings: List[str]
    mitigation_actions: List[str]
    assessed_at: datetime


class CompositeTransitionRiskResponse(BaseModel):
    """Composite transition risk score across all categories."""
    org_id: str
    composite_score: float
    composite_rating: str
    policy_score: float
    technology_score: float
    market_score: float
    reputation_score: float
    total_exposure_usd: float
    category_breakdown: Dict[str, Dict[str, Any]]
    trend: str
    peer_comparison: Dict[str, float]
    generated_at: datetime


class SectorProfileResponse(BaseModel):
    """Sector-level transition risk profile."""
    sector: str
    overall_transition_risk: str
    policy_exposure: str
    technology_disruption: str
    market_shift: str
    reputation_sensitivity: str
    carbon_intensity: str
    decarbonization_pathway: str
    key_regulations: List[str]
    technology_outlook: List[str]
    peer_leaders: List[str]
    generated_at: datetime


class CarbonExposureResponse(BaseModel):
    """Carbon cost exposure analysis."""
    org_id: str
    current_carbon_cost_usd: float
    projected_cost_2030_usd: float
    projected_cost_2040_usd: float
    projected_cost_2050_usd: float
    cumulative_cost_2025_2050_usd: float
    cost_as_pct_of_revenue: Dict[str, float]
    breakeven_carbon_price_usd: float
    generated_at: datetime


class StrandingAnalysisResponse(BaseModel):
    """Asset stranding analysis."""
    org_id: str
    total_asset_value_usd: float
    stranded_asset_value_usd: float
    stranding_ratio_pct: float
    by_asset_category: Dict[str, Dict[str, float]]
    stranding_timeline: Dict[str, float]
    writedown_risk_usd: float
    recommendations: List[str]
    generated_at: datetime


class RegulationTimelineResponse(BaseModel):
    """Regulatory timeline affecting the organization."""
    org_id: str
    regulations: List[Dict[str, Any]]
    total_compliance_cost_usd: float
    critical_deadlines: List[Dict[str, str]]
    generated_at: datetime


class TransitionDisclosureResponse(BaseModel):
    """Transition risk disclosure text."""
    org_id: str
    disclosure_text: str
    word_count: int
    risk_categories_covered: List[str]
    compliance_score: float
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_transition_assessments: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


def _rating(score: float) -> str:
    if score >= 80:
        return "very_high"
    if score >= 60:
        return "high"
    if score >= 40:
        return "medium"
    if score >= 20:
        return "low"
    return "negligible"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/assess/policy",
    response_model=TransitionRiskAssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Assess policy risk",
    description=(
        "Assess policy and regulatory transition risk including carbon pricing, "
        "emissions trading, CBAM, disclosure mandates, and litigation risk."
    ),
)
async def assess_policy_risk(request: AssessPolicyRiskRequest) -> TransitionRiskAssessmentResponse:
    """Assess policy/regulatory transition risk."""
    assessment_id = _generate_id("trp")
    driver_count = len(request.risk_drivers)
    base_score = min(driver_count * 12, 50)
    carbon_exposure = request.scope1_emissions_tco2e + request.scope2_emissions_tco2e
    carbon_cost_pct = (carbon_exposure * request.current_carbon_cost_usd_per_tco2e / request.annual_revenue_usd * 100) if request.annual_revenue_usd > 0 else 0
    if carbon_cost_pct > 2:
        base_score += 25
    elif carbon_cost_pct > 1:
        base_score += 15
    else:
        base_score += 5
    jurisdiction_score = min(len(request.jurisdictions) * 5, 25)
    total_score = min(round(base_score + jurisdiction_score, 1), 100)
    financial_exposure = round(carbon_exposure * 100, 2)  # Projected carbon cost at $100/tCO2e
    revenue_at_risk = round(financial_exposure / request.annual_revenue_usd * 100, 2) if request.annual_revenue_usd > 0 else 0

    findings = []
    if PolicyRiskDriver.CBAM in request.risk_drivers:
        findings.append("CBAM exposure may increase import costs for carbon-intensive goods")
    if PolicyRiskDriver.CARBON_TAX in request.risk_drivers:
        findings.append(f"Carbon tax exposure: {carbon_exposure:,.0f} tCO2e at risk of pricing")
    if PolicyRiskDriver.MANDATORY_DISCLOSURE in request.risk_drivers:
        findings.append("Mandatory disclosure requirements increase compliance burden")
    if PolicyRiskDriver.LITIGATION in request.risk_drivers:
        findings.append("Climate litigation risk is rising across jurisdictions")

    mitigations = [
        "Accelerate emissions reduction to lower carbon price exposure",
        "Engage with regulators on transition timeline",
        "Build internal carbon pricing into investment decisions",
        "Prepare for mandatory disclosure requirements (CSRD, SEC, ISSB)",
    ]

    result = {
        "assessment_id": assessment_id,
        "org_id": request.org_id,
        "category": TransitionRiskCategory.POLICY.value,
        "risk_score": total_score,
        "risk_rating": _rating(total_score),
        "risk_drivers": [d.value for d in request.risk_drivers],
        "financial_exposure_usd": financial_exposure,
        "revenue_at_risk_pct": revenue_at_risk,
        "key_findings": findings,
        "mitigation_actions": mitigations,
        "assessed_at": _now(),
    }
    _transition_assessments[assessment_id] = result
    return TransitionRiskAssessmentResponse(**result)


@router.post(
    "/assess/technology",
    response_model=TransitionRiskAssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Assess technology risk",
    description="Assess technology transition risk including substitution, disruption, and capex requirements.",
)
async def assess_technology_risk(request: AssessTechRiskRequest) -> TransitionRiskAssessmentResponse:
    """Assess technology transition risk."""
    assessment_id = _generate_id("trt")
    base_score = min(len(request.risk_drivers) * 14, 55)
    if request.stranded_tech_value_usd > 0:
        base_score += 20
    if request.rd_budget_pct_of_revenue < 2:
        base_score += 15
    elif request.rd_budget_pct_of_revenue < 5:
        base_score += 8
    total_score = min(round(base_score, 1), 100)
    financial_exposure = round(request.technology_capex_usd + request.stranded_tech_value_usd * 0.5, 2)

    findings = []
    if TechRiskDriver.RENEWABLE_SUBSTITUTION in request.risk_drivers:
        findings.append("Renewable energy substitution may strand existing fossil fuel infrastructure")
    if TechRiskDriver.ELECTRIFICATION in request.risk_drivers:
        findings.append("Electrification trend requires significant fleet and process conversion capex")
    if request.rd_budget_pct_of_revenue < 2:
        findings.append("Low R&D investment may limit ability to adapt to technology shifts")

    result = {
        "assessment_id": assessment_id,
        "org_id": request.org_id,
        "category": TransitionRiskCategory.TECHNOLOGY.value,
        "risk_score": total_score,
        "risk_rating": _rating(total_score),
        "risk_drivers": [d.value for d in request.risk_drivers],
        "financial_exposure_usd": financial_exposure,
        "revenue_at_risk_pct": round(financial_exposure / 500000000 * 100, 2),
        "key_findings": findings,
        "mitigation_actions": [
            "Increase R&D allocation towards low-carbon technologies",
            "Develop technology transition roadmap with milestones",
            "Partner with technology innovators for early adoption",
            "Assess stranded technology exposure and plan managed phase-out",
        ],
        "assessed_at": _now(),
    }
    _transition_assessments[assessment_id] = result
    return TransitionRiskAssessmentResponse(**result)


@router.post(
    "/assess/market",
    response_model=TransitionRiskAssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Assess market risk",
    description="Assess market transition risk including demand shifts, commodity changes, and competition.",
)
async def assess_market_risk(request: AssessMarketRiskRequest) -> TransitionRiskAssessmentResponse:
    """Assess market transition risk."""
    assessment_id = _generate_id("trm")
    base_score = min(len(request.risk_drivers) * 13, 50)
    if request.carbon_intensive_revenue_pct > 50:
        base_score += 30
    elif request.carbon_intensive_revenue_pct > 25:
        base_score += 20
    else:
        base_score += 10
    total_score = min(round(base_score, 1), 100)
    revenue_at_risk = round(request.annual_revenue_usd * request.carbon_intensive_revenue_pct / 100, 2)

    findings = []
    if request.carbon_intensive_revenue_pct > 50:
        findings.append(f"{request.carbon_intensive_revenue_pct}% of revenue from carbon-intensive products is a significant exposure")
    if MarketRiskDriver.GREEN_COMPETITION in request.risk_drivers:
        findings.append("Growing low-carbon competition threatens market share")
    if MarketRiskDriver.DEMAND_SHIFT in request.risk_drivers:
        findings.append("Consumer and B2B demand shifting towards low-carbon alternatives")

    result = {
        "assessment_id": assessment_id,
        "org_id": request.org_id,
        "category": TransitionRiskCategory.MARKET.value,
        "risk_score": total_score,
        "risk_rating": _rating(total_score),
        "risk_drivers": [d.value for d in request.risk_drivers],
        "financial_exposure_usd": revenue_at_risk,
        "revenue_at_risk_pct": request.carbon_intensive_revenue_pct,
        "key_findings": findings,
        "mitigation_actions": [
            "Diversify product portfolio with low-carbon alternatives",
            "Invest in sustainable product R&D",
            "Build green brand positioning",
            "Monitor market demand signals for timing of portfolio shifts",
        ],
        "assessed_at": _now(),
    }
    _transition_assessments[assessment_id] = result
    return TransitionRiskAssessmentResponse(**result)


@router.post(
    "/assess/reputation",
    response_model=TransitionRiskAssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Assess reputation risk",
    description="Assess reputational transition risk including ESG ratings, stakeholder pressure, and greenwashing.",
)
async def assess_reputation_risk(request: AssessReputationRiskRequest) -> TransitionRiskAssessmentResponse:
    """Assess reputation transition risk."""
    assessment_id = _generate_id("trr")
    base_score = min(len(request.risk_drivers) * 15, 50)
    if request.current_esg_score < 40:
        base_score += 30
    elif request.current_esg_score < 60:
        base_score += 15
    if not request.has_net_zero_target:
        base_score += 10
    if not request.has_sbti_target:
        base_score += 10
    total_score = min(round(base_score, 1), 100)
    financial_exposure = round(request.brand_value_usd * total_score / 100 * 0.2, 2)

    findings = []
    if not request.has_net_zero_target:
        findings.append("Absence of net-zero commitment increases reputational exposure")
    if ReputationRiskDriver.GREENWASHING_ALLEGATION in request.risk_drivers:
        findings.append("Greenwashing risk requires robust data substantiation")
    if ReputationRiskDriver.INVESTOR_DIVESTMENT in request.risk_drivers:
        findings.append("Investor divestment pressure may affect access to capital")

    result = {
        "assessment_id": assessment_id,
        "org_id": request.org_id,
        "category": TransitionRiskCategory.REPUTATION.value,
        "risk_score": total_score,
        "risk_rating": _rating(total_score),
        "risk_drivers": [d.value for d in request.risk_drivers],
        "financial_exposure_usd": financial_exposure,
        "revenue_at_risk_pct": round(total_score * 0.15, 2),
        "key_findings": findings,
        "mitigation_actions": [
            "Set and publish science-based net-zero targets",
            "Ensure transparent and data-backed climate communications",
            "Improve ESG reporting and disclosure quality",
            "Engage with investors on climate transition strategy",
        ],
        "assessed_at": _now(),
    }
    _transition_assessments[assessment_id] = result
    return TransitionRiskAssessmentResponse(**result)


@router.get(
    "/results/{org_id}",
    response_model=List[TransitionRiskAssessmentResponse],
    summary="List transition risk results",
    description="Retrieve all transition risk assessments for an organization.",
)
async def list_transition_results(
    org_id: str,
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
) -> List[TransitionRiskAssessmentResponse]:
    """List transition risk results."""
    results = [r for r in _transition_assessments.values() if r["org_id"] == org_id]
    if category:
        results = [r for r in results if r["category"] == category]
    results.sort(key=lambda r: r["risk_score"], reverse=True)
    return [TransitionRiskAssessmentResponse(**r) for r in results[:limit]]


@router.get(
    "/composite/{org_id}",
    response_model=CompositeTransitionRiskResponse,
    summary="Composite transition risk score",
    description="Calculate composite transition risk score across all four categories.",
)
async def get_composite_score(org_id: str) -> CompositeTransitionRiskResponse:
    """Calculate composite transition risk score."""
    org_results = [r for r in _transition_assessments.values() if r["org_id"] == org_id]
    cat_scores: Dict[str, List[float]] = {c.value: [] for c in TransitionRiskCategory}
    cat_exposure: Dict[str, float] = {c.value: 0 for c in TransitionRiskCategory}

    for r in org_results:
        cat = r["category"]
        cat_scores[cat].append(r["risk_score"])
        cat_exposure[cat] += r["financial_exposure_usd"]

    policy_avg = round(sum(cat_scores["policy"]) / len(cat_scores["policy"]), 1) if cat_scores["policy"] else 0
    tech_avg = round(sum(cat_scores["technology"]) / len(cat_scores["technology"]), 1) if cat_scores["technology"] else 0
    market_avg = round(sum(cat_scores["market"]) / len(cat_scores["market"]), 1) if cat_scores["market"] else 0
    rep_avg = round(sum(cat_scores["reputation"]) / len(cat_scores["reputation"]), 1) if cat_scores["reputation"] else 0

    composite = round(policy_avg * 0.35 + tech_avg * 0.25 + market_avg * 0.25 + rep_avg * 0.15, 1)
    total_exposure = round(sum(cat_exposure.values()), 2)

    breakdown = {}
    for cat in TransitionRiskCategory:
        c = cat.value
        avg = round(sum(cat_scores[c]) / len(cat_scores[c]), 1) if cat_scores[c] else 0
        breakdown[c] = {
            "score": avg,
            "rating": _rating(avg),
            "exposure_usd": round(cat_exposure[c], 2),
            "assessment_count": len(cat_scores[c]),
        }

    return CompositeTransitionRiskResponse(
        org_id=org_id,
        composite_score=composite,
        composite_rating=_rating(composite),
        policy_score=policy_avg,
        technology_score=tech_avg,
        market_score=market_avg,
        reputation_score=rep_avg,
        total_exposure_usd=total_exposure,
        category_breakdown=breakdown,
        trend="increasing" if composite > 50 else "stable",
        peer_comparison={"org": composite, "sector_avg": 52.0, "peer_median": 48.5},
        generated_at=_now(),
    )


@router.get(
    "/sector-profile/{sector}",
    response_model=SectorProfileResponse,
    summary="Sector transition profile",
    description="Get the transition risk profile for a specific industry sector.",
)
async def get_sector_profile(sector: str) -> SectorProfileResponse:
    """Get sector transition profile."""
    profiles = {
        "energy": {
            "overall_transition_risk": "very_high",
            "policy_exposure": "very_high", "technology_disruption": "very_high",
            "market_shift": "high", "reputation_sensitivity": "very_high",
            "carbon_intensity": "very_high", "decarbonization_pathway": "Renewables + CCS + hydrogen",
            "key_regulations": ["EU ETS", "US EPA regulations", "CBAM", "Methane regulation"],
            "technology_outlook": ["Solar/wind cost decline", "Battery storage", "Green hydrogen", "CCS deployment"],
            "peer_leaders": ["Orsted", "NextEra Energy", "Enel"],
        },
        "manufacturing": {
            "overall_transition_risk": "high",
            "policy_exposure": "high", "technology_disruption": "medium",
            "market_shift": "high", "reputation_sensitivity": "medium",
            "carbon_intensity": "high", "decarbonization_pathway": "Electrification + efficiency + circularity",
            "key_regulations": ["EU ETS", "CBAM", "CSRD", "Product efficiency standards"],
            "technology_outlook": ["Industrial heat pumps", "Electric furnaces", "Process optimization AI"],
            "peer_leaders": ["Schneider Electric", "Siemens", "ABB"],
        },
        "financial": {
            "overall_transition_risk": "medium",
            "policy_exposure": "medium", "technology_disruption": "low",
            "market_shift": "medium", "reputation_sensitivity": "high",
            "carbon_intensity": "low", "decarbonization_pathway": "Portfolio decarbonization + green finance",
            "key_regulations": ["SFDR", "EU Taxonomy", "TCFD mandates", "Green Bond Standards"],
            "technology_outlook": ["Climate risk analytics", "Green fintech", "ESG data platforms"],
            "peer_leaders": ["ING", "BNP Paribas", "HSBC"],
        },
    }
    profile = profiles.get(sector.lower())
    if not profile:
        profile = {
            "overall_transition_risk": "medium",
            "policy_exposure": "medium", "technology_disruption": "medium",
            "market_shift": "medium", "reputation_sensitivity": "medium",
            "carbon_intensity": "medium", "decarbonization_pathway": "Sector-specific pathway under development",
            "key_regulations": ["CSRD", "ISSB/IFRS S2", "National carbon pricing"],
            "technology_outlook": ["Energy efficiency", "Renewable procurement", "Digitalization"],
            "peer_leaders": [],
        }

    return SectorProfileResponse(sector=sector, **profile, generated_at=_now())


@router.get(
    "/carbon-exposure/{org_id}",
    response_model=CarbonExposureResponse,
    summary="Carbon cost exposure",
    description="Analyze cumulative carbon cost exposure under projected carbon pricing trajectories.",
)
async def get_carbon_exposure(
    org_id: str,
    scope1_tco2e: float = Query(25000, ge=0, description="Annual Scope 1 emissions"),
    scope2_tco2e: float = Query(15000, ge=0, description="Annual Scope 2 emissions"),
    annual_revenue_usd: float = Query(500000000, ge=0, description="Annual revenue"),
    current_carbon_price: float = Query(30, ge=0, description="Current carbon price USD/tCO2e"),
) -> CarbonExposureResponse:
    """Analyze carbon cost exposure."""
    total_emissions = scope1_tco2e + scope2_tco2e
    current_cost = round(total_emissions * current_carbon_price, 2)
    # Projected costs assuming reduction trajectory
    cost_2030 = round(total_emissions * 0.85 * 100, 2)   # $100/tCO2e
    cost_2040 = round(total_emissions * 0.60 * 175, 2)   # $175/tCO2e
    cost_2050 = round(total_emissions * 0.40 * 250, 2)   # $250/tCO2e
    cumulative = round(sum([
        current_cost * 5,
        cost_2030 * 5,
        (cost_2030 + cost_2040) / 2 * 5,
        (cost_2040 + cost_2050) / 2 * 5,
    ]), 2)

    rev = annual_revenue_usd if annual_revenue_usd > 0 else 1
    cost_pct = {
        "2025": round(current_cost / rev * 100, 2),
        "2030": round(cost_2030 / rev * 100, 2),
        "2040": round(cost_2040 / rev * 100, 2),
        "2050": round(cost_2050 / rev * 100, 2),
    }
    breakeven = round(rev * 0.05 / total_emissions, 2) if total_emissions > 0 else 0

    return CarbonExposureResponse(
        org_id=org_id,
        current_carbon_cost_usd=current_cost,
        projected_cost_2030_usd=cost_2030,
        projected_cost_2040_usd=cost_2040,
        projected_cost_2050_usd=cost_2050,
        cumulative_cost_2025_2050_usd=cumulative,
        cost_as_pct_of_revenue=cost_pct,
        breakeven_carbon_price_usd=breakeven,
        generated_at=_now(),
    )


@router.get(
    "/stranding/{org_id}",
    response_model=StrandingAnalysisResponse,
    summary="Asset stranding analysis",
    description="Analyze the risk of asset stranding under climate transition scenarios.",
)
async def get_stranding_analysis(org_id: str) -> StrandingAnalysisResponse:
    """Analyze asset stranding risk."""
    # Simulated stranding data
    total_assets = 850000000
    by_category = {
        "fossil_fuel_infrastructure": {"value_usd": 120000000, "stranding_pct": 60, "stranded_usd": 72000000},
        "carbon_intensive_equipment": {"value_usd": 200000000, "stranding_pct": 30, "stranded_usd": 60000000},
        "real_estate_coastal": {"value_usd": 80000000, "stranding_pct": 15, "stranded_usd": 12000000},
        "vehicle_fleet_ice": {"value_usd": 45000000, "stranding_pct": 50, "stranded_usd": 22500000},
        "other_assets": {"value_usd": 405000000, "stranding_pct": 2, "stranded_usd": 8100000},
    }
    total_stranded = sum(c["stranded_usd"] for c in by_category.values())
    stranding_ratio = round(total_stranded / total_assets * 100, 2)

    timeline = {
        "2030": round(total_stranded * 0.15, 2),
        "2035": round(total_stranded * 0.35, 2),
        "2040": round(total_stranded * 0.60, 2),
        "2045": round(total_stranded * 0.85, 2),
        "2050": round(total_stranded, 2),
    }

    return StrandingAnalysisResponse(
        org_id=org_id,
        total_asset_value_usd=total_assets,
        stranded_asset_value_usd=round(total_stranded, 2),
        stranding_ratio_pct=stranding_ratio,
        by_asset_category=by_category,
        stranding_timeline=timeline,
        writedown_risk_usd=round(total_stranded * 0.7, 2),
        recommendations=[
            "Develop managed phase-out plan for fossil fuel infrastructure",
            "Accelerate fleet electrification to avoid ICE vehicle stranding",
            "Reassess coastal real estate exposure under sea level rise projections",
            "Build depreciation schedules reflecting climate transition timelines",
        ],
        generated_at=_now(),
    )


@router.get(
    "/regulations/{org_id}",
    response_model=RegulationTimelineResponse,
    summary="Regulation timeline",
    description="Get a timeline of climate regulations affecting the organization.",
)
async def get_regulation_timeline(org_id: str) -> RegulationTimelineResponse:
    """Get regulation timeline."""
    regulations = [
        {"regulation": "EU CSRD", "jurisdiction": "EU", "effective_date": "2025-01-01", "impact": "Mandatory climate disclosure for large companies", "compliance_cost_usd": 500000, "status": "in_effect"},
        {"regulation": "EU CBAM Full Phase-in", "jurisdiction": "EU", "effective_date": "2026-01-01", "impact": "Carbon border adjustment for imports", "compliance_cost_usd": 2000000, "status": "transitional"},
        {"regulation": "SEC Climate Rule", "jurisdiction": "US", "effective_date": "2026-01-01", "impact": "Mandatory Scope 1/2 disclosure for US-listed companies", "compliance_cost_usd": 350000, "status": "pending"},
        {"regulation": "ISSB/IFRS S2", "jurisdiction": "Global", "effective_date": "2025-01-01", "impact": "Global sustainability disclosure baseline", "compliance_cost_usd": 400000, "status": "in_effect"},
        {"regulation": "EU ETS Phase 4+", "jurisdiction": "EU", "effective_date": "2026-01-01", "impact": "Reduced free allowances, higher carbon price", "compliance_cost_usd": 3000000, "status": "in_effect"},
        {"regulation": "UK Net Zero Strategy", "jurisdiction": "UK", "effective_date": "2025-04-01", "impact": "Mandatory transition plans for listed companies", "compliance_cost_usd": 250000, "status": "in_effect"},
    ]
    total_cost = sum(r["compliance_cost_usd"] for r in regulations)
    deadlines = [
        {"regulation": r["regulation"], "deadline": r["effective_date"]}
        for r in sorted(regulations, key=lambda x: x["effective_date"])
    ]

    return RegulationTimelineResponse(
        org_id=org_id,
        regulations=regulations,
        total_compliance_cost_usd=total_cost,
        critical_deadlines=deadlines,
        generated_at=_now(),
    )


@router.get(
    "/disclosure/{org_id}",
    response_model=TransitionDisclosureResponse,
    summary="Transition risk disclosure text",
    description="Generate transition risk disclosure text for TCFD reporting.",
)
async def get_transition_disclosure(org_id: str) -> TransitionDisclosureResponse:
    """Generate transition risk disclosure text."""
    org_results = [r for r in _transition_assessments.values() if r["org_id"] == org_id]
    categories_covered = list(set(r["category"] for r in org_results))
    avg_score = round(sum(r["risk_score"] for r in org_results) / len(org_results), 1) if org_results else 0

    text = (
        f"The organization has assessed transition risks across "
        f"{len(categories_covered)} categories: {', '.join(categories_covered) or 'none yet assessed'}. "
        f"The composite transition risk score is {avg_score}/100 ({_rating(avg_score)}). "
        f"Policy and regulatory risks represent the most significant exposure, driven by "
        f"evolving carbon pricing mechanisms, mandatory disclosure requirements (CSRD, ISSB), "
        f"and sector-specific emissions standards. Technology transition risks relate to the "
        f"need to invest in low-carbon alternatives and manage potential stranding of existing "
        f"carbon-intensive assets. Market risks include shifting customer preferences towards "
        f"sustainable products and growing competition from low-carbon alternatives. Reputational "
        f"risks are managed through transparent disclosure and science-based target setting."
    )

    compliance = min(len(categories_covered) * 25.0, 100.0)

    return TransitionDisclosureResponse(
        org_id=org_id,
        disclosure_text=text,
        word_count=len(text.split()),
        risk_categories_covered=categories_covered,
        compliance_score=compliance,
        generated_at=_now(),
    )
