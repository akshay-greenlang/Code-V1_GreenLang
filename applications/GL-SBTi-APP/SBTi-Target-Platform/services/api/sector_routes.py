"""
GL-SBTi-APP Sector Pathways API

Manages SBTi sector classifications and sector-specific decarbonization
pathways.  Provides sector detection from standard industry codes
(ISIC, NACE, NAICS), sector-specific pathway calculation using SDA
methodology, multi-sector blending for diversified companies, and
sector benchmark comparisons.

SBTi Sector Coverage:
    - Power generation (SDA: tCO2e/MWh)
    - Transport (SDA: gCO2e/pkm or gCO2e/tkm)
    - Buildings (SDA: kgCO2e/m2)
    - Cement (SDA: tCO2e/tonne clinker)
    - Iron & Steel (SDA: tCO2e/tonne steel)
    - Aluminium (SDA: tCO2e/tonne aluminium)
    - Pulp & Paper (SDA: tCO2e/tonne product)
    - Chemicals (ACA or SDA)
    - Oil & Gas (ACA or SDA)
    - Aviation (gCO2e/RTK)
    - Shipping (gCO2e/tnm)
    - All other sectors (ACA)
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

router = APIRouter(prefix="/api/v1/sbti/sectors", tags=["Sector Pathways"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class SectorDetectRequest(BaseModel):
    """Request to auto-detect sector from industry codes."""
    isic_code: Optional[str] = Field(None, description="ISIC Rev 4 code")
    nace_code: Optional[str] = Field(None, description="NACE Rev 2 code")
    naics_code: Optional[str] = Field(None, description="NAICS code")
    company_description: Optional[str] = Field(None, max_length=2000)


class SectorCalculateRequest(BaseModel):
    """Request to calculate sector-specific pathway."""
    org_id: str = Field(...)
    base_year: int = Field(..., ge=2015, le=2025)
    base_year_intensity: float = Field(..., gt=0)
    base_year_activity: float = Field(..., gt=0)
    target_year: int = Field(..., ge=2025, le=2055)
    projected_activity: float = Field(..., gt=0)
    ambition: str = Field("1.5C")


class SectorBlendRequest(BaseModel):
    """Request to blend pathways for multi-sector companies."""
    org_id: str = Field(...)
    sector_weights: List[Dict[str, Any]] = Field(
        ..., description="List of {sector, weight_pct, emissions_tco2e}",
    )
    base_year: int = Field(..., ge=2015, le=2025)
    target_year: int = Field(..., ge=2025, le=2055)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class SectorInfo(BaseModel):
    """SBTi sector information."""
    sector_id: str
    name: str
    sda_available: bool
    intensity_unit: str
    convergence_year: int
    convergence_intensity: float
    methodology: str
    description: str
    sub_sectors: List[str]


class SectorPathwayResponse(BaseModel):
    """Sector-specific pathway data."""
    sector_id: str
    name: str
    base_year: int
    convergence_year: int
    base_intensity: float
    convergence_intensity: float
    intensity_unit: str
    yearly_intensity: Dict[str, float]
    reduction_pct: float
    generated_at: datetime


class SectorCalculateResponse(BaseModel):
    """Sector pathway calculation result."""
    pathway_id: str
    org_id: str
    sector_id: str
    base_year: int
    target_year: int
    base_intensity: float
    target_intensity: float
    intensity_unit: str
    base_emissions_tco2e: float
    target_emissions_tco2e: float
    reduction_pct: float
    annual_reduction_pct: float
    yearly_budgets: Dict[str, float]
    methodology_notes: str
    generated_at: datetime


class SectorDetectResponse(BaseModel):
    """Sector detection result."""
    detected_sector: str
    confidence: float
    sector_name: str
    sda_available: bool
    intensity_unit: str
    alternative_sectors: List[Dict[str, Any]]
    input_codes: Dict[str, Optional[str]]
    generated_at: datetime


class SectorBlendResponse(BaseModel):
    """Blended pathway for multi-sector companies."""
    org_id: str
    blended_reduction_pct: float
    blended_annual_rate: float
    sector_contributions: List[Dict[str, Any]]
    total_base_emissions: float
    total_target_emissions: float
    methodology_notes: str
    generated_at: datetime


class SectorBenchmarkResponse(BaseModel):
    """Sector benchmark data."""
    sector_id: str
    sector_name: str
    benchmark_intensity: float
    intensity_unit: str
    peer_average_intensity: float
    top_quartile_intensity: float
    bottom_quartile_intensity: float
    sbti_convergence_target: float
    year: int
    data_sources: List[str]
    generated_at: datetime


# ---------------------------------------------------------------------------
# Reference Data
# ---------------------------------------------------------------------------

SECTORS: List[Dict[str, Any]] = [
    {"sector_id": "power_generation", "name": "Power Generation", "sda_available": True,
     "intensity_unit": "tCO2e/MWh", "convergence_year": 2050, "convergence_intensity": 0.01,
     "methodology": "SDA", "description": "Electricity generation from all fuel sources.",
     "sub_sectors": ["thermal", "renewable", "nuclear", "combined_cycle"]},
    {"sector_id": "transport_passenger", "name": "Passenger Transport", "sda_available": True,
     "intensity_unit": "gCO2e/pkm", "convergence_year": 2050, "convergence_intensity": 12.0,
     "methodology": "SDA", "description": "Passenger transportation across all modes.",
     "sub_sectors": ["road", "rail", "air", "maritime"]},
    {"sector_id": "transport_freight", "name": "Freight Transport", "sda_available": True,
     "intensity_unit": "gCO2e/tkm", "convergence_year": 2050, "convergence_intensity": 8.0,
     "methodology": "SDA", "description": "Freight transportation and logistics.",
     "sub_sectors": ["road", "rail", "air", "maritime"]},
    {"sector_id": "buildings_commercial", "name": "Commercial Buildings", "sda_available": True,
     "intensity_unit": "kgCO2e/m2", "convergence_year": 2050, "convergence_intensity": 2.0,
     "methodology": "SDA", "description": "Commercial and office buildings energy use.",
     "sub_sectors": ["office", "retail", "hospitality", "healthcare"]},
    {"sector_id": "buildings_residential", "name": "Residential Buildings", "sda_available": True,
     "intensity_unit": "kgCO2e/m2", "convergence_year": 2050, "convergence_intensity": 3.0,
     "methodology": "SDA", "description": "Residential buildings energy use.",
     "sub_sectors": ["single_family", "multi_family", "social_housing"]},
    {"sector_id": "cement", "name": "Cement", "sda_available": True,
     "intensity_unit": "tCO2e/tonne_clinker", "convergence_year": 2050, "convergence_intensity": 0.42,
     "methodology": "SDA", "description": "Cement and clinker production.",
     "sub_sectors": ["portland", "blended", "specialty"]},
    {"sector_id": "iron_steel", "name": "Iron & Steel", "sda_available": True,
     "intensity_unit": "tCO2e/tonne_steel", "convergence_year": 2050, "convergence_intensity": 0.31,
     "methodology": "SDA", "description": "Iron and steel production.",
     "sub_sectors": ["blast_furnace", "electric_arc", "direct_reduced"]},
    {"sector_id": "aluminium", "name": "Aluminium", "sda_available": True,
     "intensity_unit": "tCO2e/tonne_aluminium", "convergence_year": 2050, "convergence_intensity": 1.2,
     "methodology": "SDA", "description": "Primary and secondary aluminium production.",
     "sub_sectors": ["smelting", "rolling", "recycling"]},
    {"sector_id": "pulp_paper", "name": "Pulp & Paper", "sda_available": True,
     "intensity_unit": "tCO2e/tonne_product", "convergence_year": 2050, "convergence_intensity": 0.15,
     "methodology": "SDA", "description": "Pulp, paper, and packaging production.",
     "sub_sectors": ["pulp", "paper", "packaging", "tissue"]},
    {"sector_id": "chemicals", "name": "Chemicals", "sda_available": False,
     "intensity_unit": "tCO2e/tonne_product", "convergence_year": 2050, "convergence_intensity": 0.0,
     "methodology": "ACA", "description": "Chemical manufacturing (ACA approach due to heterogeneity).",
     "sub_sectors": ["basic_chemicals", "specialty", "pharma", "fertilizer"]},
    {"sector_id": "oil_gas", "name": "Oil & Gas", "sda_available": False,
     "intensity_unit": "tCO2e/TJ", "convergence_year": 2050, "convergence_intensity": 0.0,
     "methodology": "ACA", "description": "Oil and gas exploration, production, and refining.",
     "sub_sectors": ["upstream", "midstream", "downstream", "integrated"]},
    {"sector_id": "aviation", "name": "Aviation", "sda_available": True,
     "intensity_unit": "gCO2e/RTK", "convergence_year": 2050, "convergence_intensity": 150.0,
     "methodology": "SDA", "description": "Commercial aviation (passenger and cargo).",
     "sub_sectors": ["passenger", "cargo", "charter"]},
    {"sector_id": "shipping", "name": "Shipping", "sda_available": True,
     "intensity_unit": "gCO2e/tnm", "convergence_year": 2050, "convergence_intensity": 3.5,
     "methodology": "SDA", "description": "International and domestic shipping.",
     "sub_sectors": ["container", "bulk", "tanker", "ro_ro"]},
    {"sector_id": "general", "name": "All Other Sectors", "sda_available": False,
     "intensity_unit": "tCO2e", "convergence_year": 2050, "convergence_intensity": 0.0,
     "methodology": "ACA", "description": "Sectors without SDA pathway use ACA.",
     "sub_sectors": ["services", "technology", "healthcare", "retail", "finance"]},
]


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


def _get_sector(sector_id: str) -> Optional[Dict[str, Any]]:
    return next((s for s in SECTORS if s["sector_id"] == sector_id), None)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/",
    response_model=List[SectorInfo],
    summary="List all SBTi sectors",
    description="List all SBTi sectors with pathway methodology and convergence data.",
)
async def list_sectors() -> List[SectorInfo]:
    """List all SBTi sectors."""
    return [SectorInfo(**s) for s in SECTORS]


@router.get(
    "/{sector}/pathway",
    response_model=SectorPathwayResponse,
    summary="Get sector pathway data",
    description=(
        "Get the reference SDA pathway for a sector showing intensity "
        "convergence from current global average to 2050 target."
    ),
)
async def get_sector_pathway(sector: str) -> SectorPathwayResponse:
    """Get sector reference pathway."""
    sec = _get_sector(sector)
    if not sec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sector '{sector}' not found. Use GET /sectors for valid values.",
        )
    if not sec["sda_available"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Sector '{sector}' uses ACA methodology and does not have an SDA convergence pathway.",
        )

    base_year = 2020
    base_intensity_map = {
        "power_generation": 0.48, "transport_passenger": 85.0,
        "transport_freight": 45.0, "buildings_commercial": 25.0,
        "buildings_residential": 30.0, "cement": 0.85,
        "iron_steel": 1.85, "aluminium": 10.5,
        "pulp_paper": 0.45, "aviation": 650.0, "shipping": 12.0,
    }
    base_intensity = base_intensity_map.get(sector, 1.0)
    conv_intensity = sec["convergence_intensity"]
    conv_year = sec["convergence_year"]
    years = conv_year - base_year

    yearly = {}
    delta = (base_intensity - conv_intensity) / years if years > 0 else 0
    for i in range(years + 1):
        yr = base_year + i
        yearly[str(yr)] = round(max(base_intensity - delta * i, conv_intensity), 4)

    reduction = round((1 - conv_intensity / base_intensity) * 100, 1) if base_intensity > 0 else 0.0

    return SectorPathwayResponse(
        sector_id=sector,
        name=sec["name"],
        base_year=base_year,
        convergence_year=conv_year,
        base_intensity=base_intensity,
        convergence_intensity=conv_intensity,
        intensity_unit=sec["intensity_unit"],
        yearly_intensity=yearly,
        reduction_pct=reduction,
        generated_at=_now(),
    )


@router.post(
    "/{sector}/calculate",
    response_model=SectorCalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate sector-specific pathway",
    description=(
        "Calculate a company-specific SDA pathway within a sector. Converges "
        "intensity to sector target while accounting for activity growth."
    ),
)
async def calculate_sector_pathway(
    sector: str,
    request: SectorCalculateRequest,
) -> SectorCalculateResponse:
    """Calculate sector pathway for organization."""
    sec = _get_sector(sector)
    if not sec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sector '{sector}' not found.",
        )

    years = request.target_year - request.base_year
    conv_year = sec["convergence_year"]
    conv_intensity = sec["convergence_intensity"]
    progress = min(years / (conv_year - request.base_year), 1.0) if conv_year > request.base_year else 1.0

    if sec["sda_available"]:
        target_intensity = round(
            request.base_year_intensity - (request.base_year_intensity - conv_intensity) * progress, 4,
        )
    else:
        # ACA fallback
        rate = 4.2 if request.ambition == "1.5C" else 2.5
        target_intensity = round(request.base_year_intensity * (1 - rate * years / 100), 4)

    base_emissions = round(request.base_year_intensity * request.base_year_activity, 1)
    target_emissions = round(target_intensity * request.projected_activity, 1)
    reduction = round(max((1 - target_emissions / base_emissions) * 100, 0), 1) if base_emissions > 0 else 0.0
    annual_rate = round(reduction / years, 2) if years > 0 else 0.0

    budgets = {}
    em_delta = (base_emissions - target_emissions) / years if years > 0 else 0
    for i in range(years + 1):
        budgets[str(request.base_year + i)] = round(
            max(base_emissions - em_delta * i, 0), 1,
        )

    pathway_id = _generate_id("pw_sec")
    return SectorCalculateResponse(
        pathway_id=pathway_id,
        org_id=request.org_id,
        sector_id=sector,
        base_year=request.base_year,
        target_year=request.target_year,
        base_intensity=request.base_year_intensity,
        target_intensity=target_intensity,
        intensity_unit=sec["intensity_unit"],
        base_emissions_tco2e=base_emissions,
        target_emissions_tco2e=target_emissions,
        reduction_pct=reduction,
        annual_reduction_pct=annual_rate,
        yearly_budgets=budgets,
        methodology_notes=(
            f"{'SDA' if sec['sda_available'] else 'ACA'} pathway for sector '{sec['name']}'. "
            f"Intensity from {request.base_year_intensity} to {target_intensity} "
            f"{sec['intensity_unit']} by {request.target_year}."
        ),
        generated_at=_now(),
    )


@router.post(
    "/detect",
    response_model=SectorDetectResponse,
    summary="Auto-detect sector",
    description=(
        "Auto-detect the most appropriate SBTi sector from ISIC, NACE, or "
        "NAICS codes. Returns the detected sector, confidence score, and "
        "alternative sector matches."
    ),
)
async def detect_sector(request: SectorDetectRequest) -> SectorDetectResponse:
    """Auto-detect sector from industry codes."""
    # Simplified detection based on code prefixes
    detected = "general"
    confidence = 0.5
    alternatives: List[Dict[str, Any]] = []

    isic = (request.isic_code or "").upper()
    naics = (request.naics_code or "").upper()

    if isic.startswith("35") or naics.startswith("2211"):
        detected, confidence = "power_generation", 0.95
        alternatives = [{"sector": "general", "confidence": 0.3}]
    elif isic.startswith("49") or naics.startswith("48"):
        detected, confidence = "transport_freight", 0.80
        alternatives = [{"sector": "transport_passenger", "confidence": 0.70}]
    elif isic.startswith("23") and "cement" in (request.company_description or "").lower():
        detected, confidence = "cement", 0.90
    elif isic.startswith("24") or naics.startswith("331"):
        detected, confidence = "iron_steel", 0.85
        alternatives = [{"sector": "aluminium", "confidence": 0.60}]
    elif isic.startswith("17") or naics.startswith("322"):
        detected, confidence = "pulp_paper", 0.90
    elif isic.startswith("20") or naics.startswith("325"):
        detected, confidence = "chemicals", 0.85
    elif isic.startswith("06") or isic.startswith("19") or naics.startswith("211"):
        detected, confidence = "oil_gas", 0.90
    elif isic.startswith("51"):
        detected, confidence = "aviation", 0.90

    sec = _get_sector(detected) or SECTORS[-1]

    return SectorDetectResponse(
        detected_sector=detected,
        confidence=confidence,
        sector_name=sec["name"],
        sda_available=sec["sda_available"],
        intensity_unit=sec["intensity_unit"],
        alternative_sectors=alternatives,
        input_codes={
            "isic_code": request.isic_code,
            "nace_code": request.nace_code,
            "naics_code": request.naics_code,
        },
        generated_at=_now(),
    )


@router.post(
    "/blend",
    response_model=SectorBlendResponse,
    summary="Blend multi-sector pathways",
    description=(
        "Blend multiple sector pathways for diversified companies operating "
        "across multiple SBTi sectors. Produces a weighted-average reduction "
        "rate based on emission share per sector."
    ),
)
async def blend_pathways(request: SectorBlendRequest) -> SectorBlendResponse:
    """Blend pathways for multi-sector companies."""
    years = request.target_year - request.base_year
    contributions = []
    total_base = 0.0
    total_target = 0.0

    for sw in request.sector_weights:
        sector_id = sw.get("sector", "general")
        weight = sw.get("weight_pct", 0)
        emissions = sw.get("emissions_tco2e", 0)
        sec = _get_sector(sector_id)

        rate = 4.2 if sec and not sec["sda_available"] else 3.5
        sector_reduction = round(min(rate * years, 100), 1)
        sector_target = round(emissions * (1 - sector_reduction / 100), 1)

        contributions.append({
            "sector": sector_id,
            "weight_pct": weight,
            "base_emissions": emissions,
            "target_emissions": sector_target,
            "reduction_pct": sector_reduction,
        })
        total_base += emissions
        total_target += sector_target

    blended_reduction = round((1 - total_target / total_base) * 100, 1) if total_base > 0 else 0.0
    blended_annual = round(blended_reduction / years, 2) if years > 0 else 0.0

    return SectorBlendResponse(
        org_id=request.org_id,
        blended_reduction_pct=blended_reduction,
        blended_annual_rate=blended_annual,
        sector_contributions=contributions,
        total_base_emissions=round(total_base, 1),
        total_target_emissions=round(total_target, 1),
        methodology_notes=(
            f"Blended pathway across {len(contributions)} sectors. "
            f"Weighted average annual reduction of {blended_annual}% over {years} years."
        ),
        generated_at=_now(),
    )


@router.get(
    "/{sector}/benchmarks",
    response_model=SectorBenchmarkResponse,
    summary="Sector benchmarks",
    description="Get sector benchmark data including peer performance and SBTi convergence targets.",
)
async def get_sector_benchmarks(
    sector: str,
    year: int = Query(2024, ge=2020, le=2050, description="Benchmark year"),
) -> SectorBenchmarkResponse:
    """Get sector benchmarks."""
    sec = _get_sector(sector)
    if not sec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sector '{sector}' not found.",
        )

    # Simulated benchmark data
    intensity_map = {
        "power_generation": (0.42, 0.35, 0.55, "IEA WEO"),
        "cement": (0.78, 0.65, 0.90, "GCCA"),
        "iron_steel": (1.70, 1.45, 2.10, "World Steel"),
        "aluminium": (9.5, 7.5, 12.0, "IAI"),
    }
    defaults = (1.0, 0.8, 1.2, "SBTi sector data")
    avg, top_q, bottom_q, source = intensity_map.get(sector, defaults)

    return SectorBenchmarkResponse(
        sector_id=sector,
        sector_name=sec["name"],
        benchmark_intensity=avg,
        intensity_unit=sec["intensity_unit"],
        peer_average_intensity=avg,
        top_quartile_intensity=top_q,
        bottom_quartile_intensity=bottom_q,
        sbti_convergence_target=sec["convergence_intensity"],
        year=year,
        data_sources=[source, "SBTi monitoring report", "CDP disclosures"],
        generated_at=_now(),
    )
