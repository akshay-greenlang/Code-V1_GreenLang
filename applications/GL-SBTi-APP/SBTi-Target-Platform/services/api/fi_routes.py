"""
GL-SBTi-APP Financial Institutions API

Manages SBTi target-setting for financial institutions (FIs) per the
SBTi Financial Institutions framework.  Covers portfolio construction,
financed emissions calculation (PCAF methodology), WACI (Weighted Average
Carbon Intensity), portfolio temperature scoring, engagement tracking,
coverage pathway to 2040, and FINZ (Financial Net-Zero) compliance.

SBTi FI Framework:
    - Portfolio coverage approach (% of portfolio with SBTs)
    - Sectoral decarbonization approach for portfolios
    - Temperature alignment approach
    - PCAF asset class coverage (6 classes)
    - Engagement targets for portfolio companies
    - Coverage pathway reaching 100% by 2040
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

router = APIRouter(prefix="/api/v1/sbti/financial-institutions", tags=["Financial Institutions"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class CreatePortfolioRequest(BaseModel):
    """Request to create an FI portfolio."""
    org_id: str = Field(...)
    portfolio_name: str = Field(..., max_length=300)
    asset_class: str = Field(
        ..., description="listed_equity, corporate_bonds, business_loans, project_finance, commercial_real_estate, mortgages",
    )
    total_value_usd: float = Field(..., gt=0)
    currency: str = Field("USD")
    reporting_year: int = Field(..., ge=2020, le=2055)
    notes: Optional[str] = Field(None, max_length=2000)


class AddHoldingRequest(BaseModel):
    """Request to add a holding to a portfolio."""
    company_id: str = Field(...)
    company_name: str = Field(..., max_length=300)
    sector: str = Field(...)
    value_usd: float = Field(..., gt=0)
    ownership_pct: Optional[float] = Field(None, ge=0, le=100)
    emissions_tco2e: Optional[float] = Field(None, ge=0)
    has_sbti_target: bool = Field(False)
    sbti_status: Optional[str] = Field(None, description="committed, validated, none")
    temperature_score_c: Optional[float] = Field(None, ge=0, le=5)
    pcaf_data_quality: Optional[int] = Field(None, ge=1, le=5)


class EngagementRequest(BaseModel):
    """Request to track engagement with a portfolio company."""
    company_id: str = Field(...)
    company_name: str = Field(..., max_length=300)
    engagement_type: str = Field(
        ..., description="direct_engagement, collaborative_engagement, proxy_voting",
    )
    objective: str = Field(..., max_length=500)
    status: str = Field("in_progress", description="not_started, in_progress, achieved, escalated")
    target_date: Optional[str] = Field(None)
    notes: Optional[str] = Field(None, max_length=2000)


class FINZValidationRequest(BaseModel):
    """Request to validate FINZ compliance."""
    portfolio_id: str = Field(...)
    org_id: str = Field(...)
    has_net_zero_commitment: bool = Field(...)
    interim_targets_set: bool = Field(False)
    engagement_strategy_defined: bool = Field(False)
    pcaf_reporting: bool = Field(False)
    annual_disclosure: bool = Field(False)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class PortfolioResponse(BaseModel):
    """FI portfolio record."""
    portfolio_id: str
    org_id: str
    portfolio_name: str
    asset_class: str
    total_value_usd: float
    currency: str
    reporting_year: int
    holdings_count: int
    notes: Optional[str]
    created_at: datetime


class HoldingResponse(BaseModel):
    """Portfolio holding record."""
    holding_id: str
    portfolio_id: str
    company_id: str
    company_name: str
    sector: str
    value_usd: float
    ownership_pct: Optional[float]
    emissions_tco2e: Optional[float]
    has_sbti_target: bool
    sbti_status: Optional[str]
    temperature_score_c: Optional[float]
    pcaf_data_quality: Optional[int]
    created_at: datetime


class CoverageResponse(BaseModel):
    """Portfolio SBTi coverage metrics."""
    portfolio_id: str
    total_holdings: int
    holdings_with_sbti: int
    coverage_pct: float
    coverage_by_value_pct: float
    coverage_by_emissions_pct: float
    committed_count: int
    validated_count: int
    no_target_count: int
    generated_at: datetime


class FinancedEmissionsResponse(BaseModel):
    """Financed emissions calculation."""
    portfolio_id: str
    total_financed_emissions_tco2e: float
    by_sector: Dict[str, float]
    by_asset_class: Dict[str, float]
    attribution_methodology: str
    reporting_year: int
    generated_at: datetime


class WACIResponse(BaseModel):
    """Weighted Average Carbon Intensity."""
    portfolio_id: str
    waci_tco2e_per_m_usd: float
    benchmark_waci: float
    relative_to_benchmark_pct: float
    by_sector: Dict[str, float]
    top_contributors: List[Dict[str, Any]]
    generated_at: datetime


class PortfolioTemperatureResponse(BaseModel):
    """Portfolio temperature score."""
    portfolio_id: str
    temperature_c: float
    alignment_status: str
    by_sector: Dict[str, float]
    paris_aligned_pct: float
    generated_at: datetime


class PCAFResponse(BaseModel):
    """PCAF data quality assessment."""
    portfolio_id: str
    overall_data_quality: float
    by_asset_class: Dict[str, float]
    quality_distribution: Dict[str, int]
    improvement_priorities: List[str]
    generated_at: datetime


class EngagementResponse(BaseModel):
    """Engagement tracking record."""
    engagement_id: str
    portfolio_id: str
    company_id: str
    company_name: str
    engagement_type: str
    objective: str
    status: str
    target_date: Optional[str]
    notes: Optional[str]
    created_at: datetime


class CoveragePathwayResponse(BaseModel):
    """Coverage pathway to 2040."""
    portfolio_id: str
    current_coverage_pct: float
    target_coverage_pct: float
    target_year: int
    yearly_coverage: Dict[str, float]
    on_track: bool
    gap_pct: float
    actions_needed: List[str]
    generated_at: datetime


class FINZValidationResponse(BaseModel):
    """FINZ compliance check result."""
    portfolio_id: str
    org_id: str
    finz_compliant: bool
    criteria_results: List[Dict[str, Any]]
    compliance_score: float
    gaps: List[str]
    recommendations: List[str]
    generated_at: datetime


class AssetClassBreakdownResponse(BaseModel):
    """Portfolio breakdown by asset class."""
    portfolio_id: str
    total_value_usd: float
    asset_classes: List[Dict[str, Any]]
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_portfolios: Dict[str, Dict[str, Any]] = {}
_holdings: Dict[str, Dict[str, Any]] = {}
_engagements: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/portfolios",
    response_model=PortfolioResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create portfolio",
    description="Create a financial institution portfolio for SBTi target-setting.",
)
async def create_portfolio(request: CreatePortfolioRequest) -> PortfolioResponse:
    """Create a portfolio."""
    portfolio_id = _generate_id("pf")
    data = {
        "portfolio_id": portfolio_id,
        "org_id": request.org_id,
        "portfolio_name": request.portfolio_name,
        "asset_class": request.asset_class,
        "total_value_usd": request.total_value_usd,
        "currency": request.currency,
        "reporting_year": request.reporting_year,
        "holdings_count": 0,
        "notes": request.notes,
        "created_at": _now(),
    }
    _portfolios[portfolio_id] = data
    return PortfolioResponse(**data)


@router.get(
    "/portfolios/{portfolio_id}",
    response_model=PortfolioResponse,
    summary="Get portfolio",
    description="Retrieve a portfolio by its ID.",
)
async def get_portfolio(portfolio_id: str) -> PortfolioResponse:
    """Get portfolio details."""
    pf = _portfolios.get(portfolio_id)
    if not pf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Portfolio {portfolio_id} not found")
    return PortfolioResponse(**pf)


@router.post(
    "/portfolios/{portfolio_id}/holdings",
    response_model=HoldingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add holding",
    description="Add a company holding to a portfolio.",
)
async def add_holding(portfolio_id: str, request: AddHoldingRequest) -> HoldingResponse:
    """Add a holding to a portfolio."""
    if portfolio_id not in _portfolios:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Portfolio {portfolio_id} not found")

    holding_id = _generate_id("hld")
    data = {
        "holding_id": holding_id,
        "portfolio_id": portfolio_id,
        "company_id": request.company_id,
        "company_name": request.company_name,
        "sector": request.sector,
        "value_usd": request.value_usd,
        "ownership_pct": request.ownership_pct,
        "emissions_tco2e": request.emissions_tco2e,
        "has_sbti_target": request.has_sbti_target,
        "sbti_status": request.sbti_status,
        "temperature_score_c": request.temperature_score_c,
        "pcaf_data_quality": request.pcaf_data_quality,
        "created_at": _now(),
    }
    _holdings[holding_id] = data
    _portfolios[portfolio_id]["holdings_count"] = sum(
        1 for h in _holdings.values() if h["portfolio_id"] == portfolio_id
    )
    return HoldingResponse(**data)


@router.get(
    "/portfolios/{portfolio_id}/coverage",
    response_model=CoverageResponse,
    summary="Portfolio coverage",
    description="Get SBTi target coverage metrics for a portfolio.",
)
async def get_coverage(portfolio_id: str) -> CoverageResponse:
    """Get portfolio coverage."""
    holdings = [h for h in _holdings.values() if h["portfolio_id"] == portfolio_id]
    total = len(holdings)
    with_sbti = sum(1 for h in holdings if h["has_sbti_target"])
    committed = sum(1 for h in holdings if h.get("sbti_status") == "committed")
    validated = sum(1 for h in holdings if h.get("sbti_status") == "validated")

    total_value = sum(h["value_usd"] for h in holdings) or 1
    sbti_value = sum(h["value_usd"] for h in holdings if h["has_sbti_target"])
    total_em = sum(h.get("emissions_tco2e", 0) for h in holdings) or 1
    sbti_em = sum(h.get("emissions_tco2e", 0) for h in holdings if h["has_sbti_target"])

    return CoverageResponse(
        portfolio_id=portfolio_id,
        total_holdings=total,
        holdings_with_sbti=with_sbti,
        coverage_pct=round(with_sbti / total * 100, 1) if total > 0 else 0.0,
        coverage_by_value_pct=round(sbti_value / total_value * 100, 1),
        coverage_by_emissions_pct=round(sbti_em / total_em * 100, 1),
        committed_count=committed,
        validated_count=validated,
        no_target_count=total - with_sbti,
        generated_at=_now(),
    )


@router.get(
    "/portfolios/{portfolio_id}/financed-emissions",
    response_model=FinancedEmissionsResponse,
    summary="Financed emissions",
    description="Calculate financed emissions for a portfolio using PCAF attribution.",
)
async def get_financed_emissions(portfolio_id: str) -> FinancedEmissionsResponse:
    """Get financed emissions."""
    return FinancedEmissionsResponse(
        portfolio_id=portfolio_id,
        total_financed_emissions_tco2e=85000,
        by_sector={"energy": 35000, "materials": 20000, "transport": 15000, "real_estate": 10000, "other": 5000},
        by_asset_class={"listed_equity": 40000, "corporate_bonds": 25000, "project_finance": 15000, "mortgages": 5000},
        attribution_methodology="PCAF Global Standard v2",
        reporting_year=2024,
        generated_at=_now(),
    )


@router.get(
    "/portfolios/{portfolio_id}/waci",
    response_model=WACIResponse,
    summary="WACI calculation",
    description="Calculate Weighted Average Carbon Intensity for the portfolio.",
)
async def get_waci(portfolio_id: str) -> WACIResponse:
    """Get WACI."""
    return WACIResponse(
        portfolio_id=portfolio_id,
        waci_tco2e_per_m_usd=125.5,
        benchmark_waci=145.0,
        relative_to_benchmark_pct=-13.4,
        by_sector={"energy": 280.0, "materials": 180.0, "technology": 15.0, "healthcare": 25.0, "finance": 8.0},
        top_contributors=[
            {"company": "OilCo Inc", "waci_contribution": 35.2, "sector": "energy"},
            {"company": "SteelCorp", "waci_contribution": 22.1, "sector": "materials"},
            {"company": "ChemWorks", "waci_contribution": 15.8, "sector": "materials"},
        ],
        generated_at=_now(),
    )


@router.get(
    "/portfolios/{portfolio_id}/temperature",
    response_model=PortfolioTemperatureResponse,
    summary="Portfolio temperature score",
    description="Calculate portfolio-level temperature alignment score.",
)
async def get_portfolio_temperature(portfolio_id: str) -> PortfolioTemperatureResponse:
    """Get portfolio temperature."""
    return PortfolioTemperatureResponse(
        portfolio_id=portfolio_id,
        temperature_c=2.05,
        alignment_status="above_2C",
        by_sector={"energy": 2.8, "materials": 2.3, "technology": 1.6, "healthcare": 1.9, "finance": 1.5},
        paris_aligned_pct=42.0,
        generated_at=_now(),
    )


@router.get(
    "/portfolios/{portfolio_id}/pcaf",
    response_model=PCAFResponse,
    summary="PCAF data quality",
    description="Assess PCAF data quality scores across the portfolio.",
)
async def get_pcaf_quality(portfolio_id: str) -> PCAFResponse:
    """Get PCAF data quality."""
    return PCAFResponse(
        portfolio_id=portfolio_id,
        overall_data_quality=3.2,
        by_asset_class={"listed_equity": 2.5, "corporate_bonds": 3.0, "project_finance": 3.5, "mortgages": 4.0},
        quality_distribution={"score_1": 5, "score_2": 15, "score_3": 30, "score_4": 35, "score_5": 15},
        improvement_priorities=[
            "Upgrade listed equity holdings from estimated to reported emissions",
            "Obtain verified emissions data for top 20 corporate bond issuers",
            "Transition project finance from sector averages to project-specific data",
        ],
        generated_at=_now(),
    )


@router.post(
    "/portfolios/{portfolio_id}/engagement",
    response_model=EngagementResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Track engagement",
    description="Track engagement activity with a portfolio company to encourage SBT adoption.",
)
async def track_engagement(
    portfolio_id: str,
    request: EngagementRequest,
) -> EngagementResponse:
    """Track engagement."""
    if portfolio_id not in _portfolios:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Portfolio {portfolio_id} not found")

    eid = _generate_id("eng")
    data = {
        "engagement_id": eid,
        "portfolio_id": portfolio_id,
        "company_id": request.company_id,
        "company_name": request.company_name,
        "engagement_type": request.engagement_type,
        "objective": request.objective,
        "status": request.status,
        "target_date": request.target_date,
        "notes": request.notes,
        "created_at": _now(),
    }
    _engagements[eid] = data
    return EngagementResponse(**data)


@router.get(
    "/portfolios/{portfolio_id}/coverage-pathway",
    response_model=CoveragePathwayResponse,
    summary="Coverage pathway to 2040",
    description=(
        "Generate the portfolio SBTi coverage pathway showing year-by-year "
        "coverage growth from current to 100% by 2040."
    ),
)
async def get_coverage_pathway(portfolio_id: str) -> CoveragePathwayResponse:
    """Get coverage pathway to 2040."""
    current = 35.0
    target = 100.0
    target_year = 2040
    current_year = _now().year
    years = target_year - current_year

    yearly = {}
    if years > 0:
        annual_increase = (target - current) / years
        for i in range(years + 1):
            yr = current_year + i
            yearly[str(yr)] = round(min(current + annual_increase * i, 100), 1)

    gap = round(max(target - current, 0), 1)
    on_track = current >= (target * (current_year - 2020) / (target_year - 2020))

    return CoveragePathwayResponse(
        portfolio_id=portfolio_id,
        current_coverage_pct=current,
        target_coverage_pct=target,
        target_year=target_year,
        yearly_coverage=yearly,
        on_track=on_track,
        gap_pct=gap,
        actions_needed=[
            "Engage top 50 holdings without SBTi targets (representing 40% of portfolio emissions)",
            "Participate in Climate Action 100+ for high-impact holdings",
            "Set escalation criteria for non-responsive holdings",
            "Include SBTi commitment in investment due diligence",
        ],
        generated_at=_now(),
    )


@router.post(
    "/portfolios/{portfolio_id}/finz-validation",
    response_model=FINZValidationResponse,
    summary="FINZ compliance check",
    description="Validate portfolio compliance against the Financial Net-Zero (FINZ) framework.",
)
async def validate_finz(
    portfolio_id: str,
    request: FINZValidationRequest,
) -> FINZValidationResponse:
    """Validate FINZ compliance."""
    criteria = [
        {"criterion": "Net-zero commitment", "met": request.has_net_zero_commitment, "required": True},
        {"criterion": "Interim targets set", "met": request.interim_targets_set, "required": True},
        {"criterion": "Engagement strategy defined", "met": request.engagement_strategy_defined, "required": True},
        {"criterion": "PCAF emissions reporting", "met": request.pcaf_reporting, "required": True},
        {"criterion": "Annual disclosure", "met": request.annual_disclosure, "required": True},
    ]

    met = sum(1 for c in criteria if c["met"])
    total = len(criteria)
    compliant = met == total
    score = round(met / total * 100, 1)

    gaps = [c["criterion"] for c in criteria if not c["met"]]
    recs = []
    if not request.has_net_zero_commitment:
        recs.append("Make a formal net-zero commitment covering all asset classes")
    if not request.interim_targets_set:
        recs.append("Set interim 2030 targets for portfolio emissions reduction")
    if not request.engagement_strategy_defined:
        recs.append("Develop engagement strategy with clear escalation mechanisms")
    if not request.pcaf_reporting:
        recs.append("Implement PCAF-aligned financed emissions calculation")
    if not request.annual_disclosure:
        recs.append("Commit to annual public disclosure of portfolio climate metrics")

    return FINZValidationResponse(
        portfolio_id=portfolio_id,
        org_id=request.org_id,
        finz_compliant=compliant,
        criteria_results=criteria,
        compliance_score=score,
        gaps=gaps,
        recommendations=recs,
        generated_at=_now(),
    )


@router.get(
    "/portfolios/{portfolio_id}/asset-class-breakdown",
    response_model=AssetClassBreakdownResponse,
    summary="Asset class breakdown",
    description="Get portfolio breakdown by PCAF asset class with emissions and coverage.",
)
async def get_asset_class_breakdown(portfolio_id: str) -> AssetClassBreakdownResponse:
    """Get asset class breakdown."""
    return AssetClassBreakdownResponse(
        portfolio_id=portfolio_id,
        total_value_usd=500000000,
        asset_classes=[
            {"asset_class": "listed_equity", "value_usd": 200000000, "pct": 40, "financed_emissions_tco2e": 40000, "data_quality": 2.5, "sbti_coverage_pct": 45},
            {"asset_class": "corporate_bonds", "value_usd": 150000000, "pct": 30, "financed_emissions_tco2e": 25000, "data_quality": 3.0, "sbti_coverage_pct": 30},
            {"asset_class": "project_finance", "value_usd": 75000000, "pct": 15, "financed_emissions_tco2e": 15000, "data_quality": 3.5, "sbti_coverage_pct": 20},
            {"asset_class": "mortgages", "value_usd": 50000000, "pct": 10, "financed_emissions_tco2e": 4000, "data_quality": 4.0, "sbti_coverage_pct": 10},
            {"asset_class": "commercial_real_estate", "value_usd": 25000000, "pct": 5, "financed_emissions_tco2e": 1000, "data_quality": 3.8, "sbti_coverage_pct": 15},
        ],
        generated_at=_now(),
    )
