"""
GL-Taxonomy-APP Green Asset Ratio (GAR) API

Calculates Green Asset Ratio (GAR) and Banking Book Taxonomy Alignment
Ratio (BTAR) for financial institutions per EBA Pillar III disclosure
requirements (ITS on ESG Risks, Commission Delegated Regulation 2022/2453).

GAR Types:
    - GAR Stock:  Taxonomy-aligned assets / Covered assets (balance sheet)
    - GAR Flow:   New taxonomy-aligned originations / Total new originations
    - BTAR:       Banking book taxonomy alignment (extended scope)

Asset Classes:
    - Loans and advances to NFCs (non-financial corporates)
    - Loans and advances to households (mortgages, vehicle loans)
    - Debt securities (corporate bonds, covered bonds)
    - Equity holdings
    - Repossessed collaterals (real estate)

EPC Integration:
    - Residential mortgages: EPC A/B for alignment
    - Commercial real estate: Top 15% energy performance
    - Vehicle loans: Zero-emission vehicles
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

router = APIRouter(prefix="/api/v1/taxonomy/gar", tags=["GAR / BTAR"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class GARStockRequest(BaseModel):
    """Calculate GAR stock ratio."""
    institution_id: str = Field(...)
    reporting_date: str = Field(..., description="ISO date (quarter end)")
    total_covered_assets_eur: float = Field(..., gt=0)
    taxonomy_aligned_assets_eur: float = Field(..., ge=0)
    taxonomy_eligible_assets_eur: float = Field(..., ge=0)
    excluded_assets_eur: float = Field(0, ge=0, description="Sovereigns, central banks, trading book")
    asset_class_breakdown: Optional[Dict[str, float]] = Field(
        None, description="Aligned amounts by asset class",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "institution_id": "bank_001",
                "reporting_date": "2025-12-31",
                "total_covered_assets_eur": 50000000000,
                "taxonomy_aligned_assets_eur": 8500000000,
                "taxonomy_eligible_assets_eur": 22000000000,
                "excluded_assets_eur": 12000000000,
            }
        }


class GARFlowRequest(BaseModel):
    """Calculate GAR flow ratio."""
    institution_id: str = Field(...)
    reporting_period_start: str = Field(...)
    reporting_period_end: str = Field(...)
    total_new_originations_eur: float = Field(..., gt=0)
    aligned_new_originations_eur: float = Field(..., ge=0)
    eligible_new_originations_eur: float = Field(..., ge=0)


class BTARRequest(BaseModel):
    """Calculate BTAR."""
    institution_id: str = Field(...)
    reporting_date: str = Field(...)
    total_banking_book_eur: float = Field(..., gt=0)
    aligned_banking_book_eur: float = Field(..., ge=0)
    eligible_banking_book_eur: float = Field(..., ge=0)
    sme_loans_aligned_eur: float = Field(0, ge=0)
    retail_aligned_eur: float = Field(0, ge=0)


class ExposureClassifyRequest(BaseModel):
    """Classify a single exposure."""
    institution_id: str = Field(...)
    counterparty_name: str = Field(...)
    counterparty_type: str = Field(..., description="nfc, household, sovereign, fi, other")
    exposure_eur: float = Field(..., gt=0)
    nace_code: Optional[str] = Field(None)
    counterparty_turnover_alignment_pct: Optional[float] = Field(None, ge=0, le=100)
    counterparty_capex_alignment_pct: Optional[float] = Field(None, ge=0, le=100)
    epc_rating: Optional[str] = Field(None, description="A, B, C, D, E, F, G (for mortgages)")
    vehicle_emission_gkm: Optional[float] = Field(None, ge=0, description="For vehicle loans")


class EBATemplateRequest(BaseModel):
    """Generate EBA Pillar III ESG template data."""
    institution_id: str = Field(...)
    reporting_date: str = Field(...)
    include_template_0: bool = Field(True, description="Summary of KPIs")
    include_template_1: bool = Field(True, description="Scope of consolidation")
    include_template_2: bool = Field(True, description="Banking book - climate change mitigation")
    include_template_4: bool = Field(True, description="GAR stock")
    include_template_5: bool = Field(True, description="GAR flow")


class MortgageCheckRequest(BaseModel):
    """Check mortgage alignment."""
    institution_id: str = Field(...)
    property_type: str = Field(..., description="residential or commercial")
    epc_rating: str = Field(..., description="A through G")
    primary_energy_demand_kwh_m2: Optional[float] = Field(None)
    country_code: str = Field("DE", max_length=2)
    building_year: Optional[int] = Field(None, ge=1900, le=2030)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class GARStockResponse(BaseModel):
    """GAR stock calculation result."""
    calculation_id: str
    institution_id: str
    reporting_date: str
    gar_stock_pct: float
    eligibility_pct: float
    total_covered_assets_eur: float
    taxonomy_aligned_eur: float
    taxonomy_eligible_eur: float
    excluded_assets_eur: float
    non_eligible_eur: float
    asset_class_breakdown: Dict[str, float]
    generated_at: datetime


class GARFlowResponse(BaseModel):
    """GAR flow calculation result."""
    calculation_id: str
    institution_id: str
    reporting_period: str
    gar_flow_pct: float
    eligibility_flow_pct: float
    total_new_originations_eur: float
    aligned_originations_eur: float
    eligible_originations_eur: float
    generated_at: datetime


class BTARResponse(BaseModel):
    """BTAR calculation result."""
    calculation_id: str
    institution_id: str
    reporting_date: str
    btar_pct: float
    eligibility_pct: float
    total_banking_book_eur: float
    aligned_banking_book_eur: float
    sme_contribution_pct: float
    retail_contribution_pct: float
    generated_at: datetime


class ExposureClassifyResponse(BaseModel):
    """Exposure classification result."""
    classification_id: str
    institution_id: str
    counterparty_name: str
    counterparty_type: str
    exposure_eur: float
    taxonomy_eligible: bool
    taxonomy_aligned: bool
    aligned_amount_eur: float
    alignment_method: str
    classification_rationale: str
    generated_at: datetime


class SectorBreakdownResponse(BaseModel):
    """GAR sector breakdown."""
    institution_id: str
    sectors: List[Dict[str, Any]]
    total_aligned_eur: float
    total_covered_eur: float
    top_sector: str
    generated_at: datetime


class GARTrendsResponse(BaseModel):
    """GAR trends over time."""
    institution_id: str
    periods: List[Dict[str, Any]]
    gar_trend: str
    avg_quarterly_change_pct: float
    generated_at: datetime


class GARCompareResponse(BaseModel):
    """GAR vs BTAR comparison."""
    institution_id: str
    gar_stock_pct: float
    gar_flow_pct: float
    btar_pct: float
    gar_vs_btar_diff: float
    analysis: str
    generated_at: datetime


class EBATemplateResponse(BaseModel):
    """EBA template data."""
    template_id: str
    institution_id: str
    reporting_date: str
    templates: Dict[str, Dict[str, Any]]
    completeness_pct: float
    generated_at: datetime


class AssetClassSummaryResponse(BaseModel):
    """Asset class summary."""
    institution_id: str
    asset_classes: List[Dict[str, Any]]
    total_covered_eur: float
    total_aligned_eur: float
    overall_gar_pct: float
    generated_at: datetime


class MortgageCheckResponse(BaseModel):
    """Mortgage alignment check result."""
    institution_id: str
    property_type: str
    epc_rating: str
    country_code: str
    taxonomy_aligned: bool
    alignment_rationale: str
    epc_threshold: str
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_gar_calculations: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/stock",
    response_model=GARStockResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate GAR stock",
    description=(
        "Calculate the Green Asset Ratio (stock) as taxonomy-aligned assets "
        "divided by total covered assets, excluding sovereigns and trading book."
    ),
)
async def calculate_gar_stock(request: GARStockRequest) -> GARStockResponse:
    """Calculate GAR stock ratio."""
    calc_id = _generate_id("gar_s")
    net_covered = request.total_covered_assets_eur - request.excluded_assets_eur
    gar = round((request.taxonomy_aligned_assets_eur / net_covered) * 100, 2) if net_covered > 0 else 0
    elig = round((request.taxonomy_eligible_assets_eur / net_covered) * 100, 2) if net_covered > 0 else 0
    non_elig = round(net_covered - request.taxonomy_eligible_assets_eur, 2)

    breakdown = request.asset_class_breakdown or {
        "loans_nfc": round(request.taxonomy_aligned_assets_eur * 0.45, 2),
        "mortgages_residential": round(request.taxonomy_aligned_assets_eur * 0.30, 2),
        "mortgages_commercial": round(request.taxonomy_aligned_assets_eur * 0.10, 2),
        "debt_securities": round(request.taxonomy_aligned_assets_eur * 0.10, 2),
        "equity": round(request.taxonomy_aligned_assets_eur * 0.05, 2),
    }

    data = {
        "calculation_id": calc_id,
        "institution_id": request.institution_id,
        "reporting_date": request.reporting_date,
        "gar_stock_pct": gar,
        "eligibility_pct": elig,
        "total_covered_assets_eur": round(net_covered, 2),
        "taxonomy_aligned_eur": request.taxonomy_aligned_assets_eur,
        "taxonomy_eligible_eur": request.taxonomy_eligible_assets_eur,
        "excluded_assets_eur": request.excluded_assets_eur,
        "non_eligible_eur": non_elig,
        "asset_class_breakdown": breakdown,
        "generated_at": _now(),
    }
    _gar_calculations[calc_id] = data
    return GARStockResponse(**data)


@router.post(
    "/flow",
    response_model=GARFlowResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate GAR flow",
    description="Calculate the GAR flow ratio based on new originations in the period.",
)
async def calculate_gar_flow(request: GARFlowRequest) -> GARFlowResponse:
    """Calculate GAR flow ratio."""
    calc_id = _generate_id("gar_f")
    gar_flow = round((request.aligned_new_originations_eur / request.total_new_originations_eur) * 100, 2) if request.total_new_originations_eur > 0 else 0
    elig_flow = round((request.eligible_new_originations_eur / request.total_new_originations_eur) * 100, 2) if request.total_new_originations_eur > 0 else 0

    return GARFlowResponse(
        calculation_id=calc_id,
        institution_id=request.institution_id,
        reporting_period=f"{request.reporting_period_start} to {request.reporting_period_end}",
        gar_flow_pct=gar_flow,
        eligibility_flow_pct=elig_flow,
        total_new_originations_eur=request.total_new_originations_eur,
        aligned_originations_eur=request.aligned_new_originations_eur,
        eligible_originations_eur=request.eligible_new_originations_eur,
        generated_at=_now(),
    )


@router.post(
    "/btar",
    response_model=BTARResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate BTAR",
    description="Calculate the Banking Book Taxonomy Alignment Ratio (BTAR).",
)
async def calculate_btar(request: BTARRequest) -> BTARResponse:
    """Calculate BTAR."""
    calc_id = _generate_id("btar")
    btar = round((request.aligned_banking_book_eur / request.total_banking_book_eur) * 100, 2) if request.total_banking_book_eur > 0 else 0
    elig = round((request.eligible_banking_book_eur / request.total_banking_book_eur) * 100, 2) if request.total_banking_book_eur > 0 else 0
    sme_pct = round((request.sme_loans_aligned_eur / request.aligned_banking_book_eur) * 100, 2) if request.aligned_banking_book_eur > 0 else 0
    retail_pct = round((request.retail_aligned_eur / request.aligned_banking_book_eur) * 100, 2) if request.aligned_banking_book_eur > 0 else 0

    return BTARResponse(
        calculation_id=calc_id,
        institution_id=request.institution_id,
        reporting_date=request.reporting_date,
        btar_pct=btar,
        eligibility_pct=elig,
        total_banking_book_eur=request.total_banking_book_eur,
        aligned_banking_book_eur=request.aligned_banking_book_eur,
        sme_contribution_pct=sme_pct,
        retail_contribution_pct=retail_pct,
        generated_at=_now(),
    )


@router.post(
    "/classify-exposure",
    response_model=ExposureClassifyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Classify exposure",
    description=(
        "Classify a single exposure for taxonomy alignment. Uses counterparty "
        "NACE code and taxonomy KPIs, or EPC rating for mortgages."
    ),
)
async def classify_exposure(request: ExposureClassifyRequest) -> ExposureClassifyResponse:
    """Classify a single exposure."""
    classification_id = _generate_id("exp")
    aligned = False
    eligible = False
    aligned_amount = 0.0
    method = "not_applicable"
    rationale = ""

    if request.counterparty_type == "sovereign":
        rationale = "Sovereign exposures excluded from GAR scope."
        method = "excluded"
    elif request.counterparty_type == "household":
        # Mortgage / vehicle loan logic
        if request.epc_rating and request.epc_rating in ("A", "B"):
            eligible = True
            aligned = True
            aligned_amount = request.exposure_eur
            method = "epc_rating"
            rationale = f"Residential mortgage with EPC {request.epc_rating} meets alignment criteria."
        elif request.vehicle_emission_gkm is not None and request.vehicle_emission_gkm == 0:
            eligible = True
            aligned = True
            aligned_amount = request.exposure_eur
            method = "zero_emission_vehicle"
            rationale = "Vehicle loan for zero-emission vehicle meets alignment criteria."
        elif request.epc_rating:
            eligible = True
            aligned = False
            method = "epc_rating"
            rationale = f"Residential mortgage with EPC {request.epc_rating} is eligible but not aligned (requires A or B)."
        else:
            rationale = "Household exposure without EPC or vehicle data cannot be classified."
            method = "insufficient_data"
    elif request.counterparty_type == "nfc":
        eligible = request.nace_code is not None
        if request.counterparty_turnover_alignment_pct is not None:
            alignment_pct = request.counterparty_turnover_alignment_pct / 100.0
            aligned_amount = round(request.exposure_eur * alignment_pct, 2)
            aligned = aligned_amount > 0
            method = "turnover_weighted"
            rationale = (
                f"NFC exposure weighted by counterparty taxonomy-aligned turnover "
                f"({request.counterparty_turnover_alignment_pct}%)."
            )
        elif request.counterparty_capex_alignment_pct is not None:
            alignment_pct = request.counterparty_capex_alignment_pct / 100.0
            aligned_amount = round(request.exposure_eur * alignment_pct, 2)
            aligned = aligned_amount > 0
            method = "capex_weighted"
            rationale = f"NFC exposure weighted by counterparty CapEx alignment ({request.counterparty_capex_alignment_pct}%)."
        else:
            rationale = "NFC exposure eligible but no counterparty alignment data available."
            method = "no_counterparty_data"
    else:
        rationale = f"Counterparty type '{request.counterparty_type}' handling pending."
        method = "manual_review"

    return ExposureClassifyResponse(
        classification_id=classification_id,
        institution_id=request.institution_id,
        counterparty_name=request.counterparty_name,
        counterparty_type=request.counterparty_type,
        exposure_eur=request.exposure_eur,
        taxonomy_eligible=eligible,
        taxonomy_aligned=aligned,
        aligned_amount_eur=aligned_amount,
        alignment_method=method,
        classification_rationale=rationale,
        generated_at=_now(),
    )


@router.get(
    "/{institution_id}/sector-breakdown",
    response_model=SectorBreakdownResponse,
    summary="GAR by sector",
    description="Get GAR broken down by NACE sector.",
)
async def get_sector_breakdown(institution_id: str) -> SectorBreakdownResponse:
    """Get GAR sector breakdown."""
    sectors = [
        {"sector": "Energy", "nace": "D35", "aligned_eur": 2800000000, "covered_eur": 8000000000, "gar_pct": 35.0},
        {"sector": "Construction & Real Estate", "nace": "F41-43/L68", "aligned_eur": 2200000000, "covered_eur": 12000000000, "gar_pct": 18.3},
        {"sector": "Transport", "nace": "H49-52", "aligned_eur": 1500000000, "covered_eur": 6000000000, "gar_pct": 25.0},
        {"sector": "Manufacturing", "nace": "C20-29", "aligned_eur": 1200000000, "covered_eur": 10000000000, "gar_pct": 12.0},
        {"sector": "ICT", "nace": "J61-63", "aligned_eur": 500000000, "covered_eur": 3000000000, "gar_pct": 16.7},
        {"sector": "Water & Waste", "nace": "E36-39", "aligned_eur": 300000000, "covered_eur": 1000000000, "gar_pct": 30.0},
    ]

    return SectorBreakdownResponse(
        institution_id=institution_id,
        sectors=sectors,
        total_aligned_eur=sum(s["aligned_eur"] for s in sectors),
        total_covered_eur=sum(s["covered_eur"] for s in sectors),
        top_sector="Energy",
        generated_at=_now(),
    )


@router.get(
    "/{institution_id}/trends",
    response_model=GARTrendsResponse,
    summary="GAR trends over time",
    description="Get GAR trends across reporting periods.",
)
async def get_trends(institution_id: str) -> GARTrendsResponse:
    """Get GAR trends."""
    periods = [
        {"period": "2023-Q4", "gar_stock_pct": 12.5, "gar_flow_pct": 18.0, "btar_pct": 14.2},
        {"period": "2024-Q1", "gar_stock_pct": 13.1, "gar_flow_pct": 19.5, "btar_pct": 15.0},
        {"period": "2024-Q2", "gar_stock_pct": 14.0, "gar_flow_pct": 21.0, "btar_pct": 15.8},
        {"period": "2024-Q3", "gar_stock_pct": 14.8, "gar_flow_pct": 22.5, "btar_pct": 16.5},
        {"period": "2024-Q4", "gar_stock_pct": 15.5, "gar_flow_pct": 24.0, "btar_pct": 17.2},
        {"period": "2025-Q1", "gar_stock_pct": 16.2, "gar_flow_pct": 25.5, "btar_pct": 18.0},
    ]

    avg_change = round((periods[-1]["gar_stock_pct"] - periods[0]["gar_stock_pct"]) / (len(periods) - 1), 2)

    return GARTrendsResponse(
        institution_id=institution_id,
        periods=periods,
        gar_trend="improving",
        avg_quarterly_change_pct=avg_change,
        generated_at=_now(),
    )


@router.get(
    "/{institution_id}/compare",
    response_model=GARCompareResponse,
    summary="Compare GAR vs BTAR",
    description="Compare GAR stock, GAR flow, and BTAR metrics.",
)
async def compare_gar_btar(institution_id: str) -> GARCompareResponse:
    """Compare GAR vs BTAR."""
    gar_stock = 16.2
    gar_flow = 25.5
    btar = 18.0
    diff = round(btar - gar_stock, 1)

    analysis = (
        f"BTAR ({btar}%) exceeds GAR stock ({gar_stock}%) by {diff} pp, indicating "
        f"the extended banking book scope captures additional aligned exposures. "
        f"GAR flow ({gar_flow}%) significantly exceeds GAR stock, reflecting the "
        f"institution's transition toward greener new originations."
    )

    return GARCompareResponse(
        institution_id=institution_id,
        gar_stock_pct=gar_stock,
        gar_flow_pct=gar_flow,
        btar_pct=btar,
        gar_vs_btar_diff=diff,
        analysis=analysis,
        generated_at=_now(),
    )


@router.post(
    "/eba-template",
    response_model=EBATemplateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate EBA template",
    description=(
        "Generate EBA Pillar III ESG disclosure template data per "
        "ITS on Prudential Disclosures on ESG Risks."
    ),
)
async def generate_eba_template(request: EBATemplateRequest) -> EBATemplateResponse:
    """Generate EBA template data."""
    template_id = _generate_id("eba")
    templates: Dict[str, Dict[str, Any]] = {}
    completed = 0
    total = 0

    if request.include_template_0:
        total += 1
        completed += 1
        templates["template_0_summary"] = {
            "title": "Summary of KPIs on taxonomy-aligned exposures",
            "gar_stock_pct": 16.2, "gar_flow_pct": 25.5, "btar_pct": 18.0,
            "status": "complete",
        }
    if request.include_template_1:
        total += 1
        completed += 1
        templates["template_1_scope"] = {
            "title": "Scope of consolidation",
            "total_assets_eur": 50000000000, "covered_assets_eur": 38000000000,
            "excluded_eur": 12000000000, "status": "complete",
        }
    if request.include_template_2:
        total += 1
        completed += 1
        templates["template_2_ccm"] = {
            "title": "Banking book - Climate change mitigation",
            "aligned_eur": 6500000000, "eligible_eur": 18000000000,
            "by_sector": {"energy": 2800000000, "construction": 2200000000, "transport": 1500000000},
            "status": "complete",
        }
    if request.include_template_4:
        total += 1
        completed += 1
        templates["template_4_gar_stock"] = {
            "title": "GAR - Stock of loans and advances, debt securities and equity instruments",
            "gar_pct": 16.2, "asset_breakdown": {"loans_nfc": 45, "mortgages": 30, "securities": 15, "equity": 10},
            "status": "complete",
        }
    if request.include_template_5:
        total += 1
        completed += 1
        templates["template_5_gar_flow"] = {
            "title": "GAR - Flow of new loans and advances, debt securities and equity instruments",
            "gar_flow_pct": 25.5, "new_originations_eur": 8000000000,
            "status": "complete",
        }

    completeness = round((completed / total) * 100, 1) if total > 0 else 0

    return EBATemplateResponse(
        template_id=template_id,
        institution_id=request.institution_id,
        reporting_date=request.reporting_date,
        templates=templates,
        completeness_pct=completeness,
        generated_at=_now(),
    )


@router.get(
    "/{institution_id}/asset-class-summary",
    response_model=AssetClassSummaryResponse,
    summary="Asset class summary",
    description="Get GAR summary by asset class.",
)
async def get_asset_class_summary(institution_id: str) -> AssetClassSummaryResponse:
    """Get asset class summary."""
    asset_classes = [
        {"asset_class": "Loans to NFCs", "covered_eur": 18000000000, "aligned_eur": 3825000000, "gar_pct": 21.3, "nfc_count": 450},
        {"asset_class": "Residential Mortgages", "covered_eur": 12000000000, "aligned_eur": 2550000000, "gar_pct": 21.3, "epc_ab_pct": 32.0},
        {"asset_class": "Commercial Real Estate", "covered_eur": 5000000000, "aligned_eur": 850000000, "gar_pct": 17.0, "epc_ab_pct": 22.0},
        {"asset_class": "Debt Securities", "covered_eur": 2000000000, "aligned_eur": 850000000, "gar_pct": 42.5},
        {"asset_class": "Equity Holdings", "covered_eur": 1000000000, "aligned_eur": 425000000, "gar_pct": 42.5},
    ]

    total_covered = sum(a["covered_eur"] for a in asset_classes)
    total_aligned = sum(a["aligned_eur"] for a in asset_classes)

    return AssetClassSummaryResponse(
        institution_id=institution_id,
        asset_classes=asset_classes,
        total_covered_eur=total_covered,
        total_aligned_eur=total_aligned,
        overall_gar_pct=round((total_aligned / total_covered) * 100, 1) if total_covered > 0 else 0,
        generated_at=_now(),
    )


@router.post(
    "/mortgage-check",
    response_model=MortgageCheckResponse,
    summary="Check mortgage alignment (EPC)",
    description="Check whether a mortgage is taxonomy-aligned based on EPC rating.",
)
async def mortgage_check(request: MortgageCheckRequest) -> MortgageCheckResponse:
    """Check mortgage alignment."""
    if request.property_type == "residential":
        aligned = request.epc_rating in ("A", "B")
        threshold = "EPC A or B"
        if aligned:
            rationale = f"Residential property with EPC {request.epc_rating} meets taxonomy alignment criteria."
        else:
            rationale = f"Residential property with EPC {request.epc_rating} does not meet minimum EPC A/B threshold."
    elif request.property_type == "commercial":
        aligned = request.epc_rating == "A"
        threshold = "EPC A or top 15% energy performance"
        if request.primary_energy_demand_kwh_m2 is not None:
            country_thresholds = {"DE": 75, "FR": 70, "NL": 65, "IT": 80, "ES": 85}
            country_threshold = country_thresholds.get(request.country_code, 80)
            if request.primary_energy_demand_kwh_m2 <= country_threshold:
                aligned = True
                rationale = (
                    f"Commercial property with PED {request.primary_energy_demand_kwh_m2} kWh/m2 "
                    f"is within top 15% threshold ({country_threshold} kWh/m2) for {request.country_code}."
                )
            else:
                rationale = (
                    f"Commercial property with PED {request.primary_energy_demand_kwh_m2} kWh/m2 "
                    f"exceeds top 15% threshold ({country_threshold} kWh/m2) for {request.country_code}."
                )
        elif aligned:
            rationale = f"Commercial property with EPC {request.epc_rating} meets alignment criteria."
        else:
            rationale = f"Commercial property with EPC {request.epc_rating} does not meet minimum threshold. Consider primary energy demand data."
    else:
        aligned = False
        threshold = "N/A"
        rationale = f"Unknown property type '{request.property_type}'."

    return MortgageCheckResponse(
        institution_id=request.institution_id,
        property_type=request.property_type,
        epc_rating=request.epc_rating,
        country_code=request.country_code,
        taxonomy_aligned=aligned,
        alignment_rationale=rationale,
        epc_threshold=threshold,
        generated_at=_now(),
    )
