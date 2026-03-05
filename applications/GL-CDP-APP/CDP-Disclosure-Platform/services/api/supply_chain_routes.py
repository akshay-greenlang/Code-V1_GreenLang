"""
GL-CDP-APP Supply Chain API

Manages CDP Supply Chain module: supplier registration, questionnaire
invitation, response tracking, emissions aggregation, engagement scoring,
cascade requests, and supply chain emissions hotspot identification.

CDP Supply Chain features:
    - Supplier invitation to CDP Climate Change questionnaire
    - Response status tracking (invited/pending/responded/scored)
    - Aggregated supplier emissions data (Scope 1/2/3)
    - Supplier engagement scoring and improvement tracking
    - Cascade request management for extended supply chain
    - Emission hotspot identification by supplier/category
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/cdp/supply-chain", tags=["Supply Chain"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SupplierStatus(str, Enum):
    """Supplier engagement status."""
    REGISTERED = "registered"
    INVITED = "invited"
    PENDING = "pending"
    RESPONDED = "responded"
    SCORED = "scored"
    DECLINED = "declined"


class SupplierTier(str, Enum):
    """Supplier tier classification."""
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    STRATEGIC = "strategic"


class CascadeStatus(str, Enum):
    """Cascade engagement request status."""
    DRAFT = "draft"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DECLINED = "declined"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class AddSupplierRequest(BaseModel):
    """Request to add a supplier to the supply chain module."""
    supplier_name: str = Field(..., min_length=1, max_length=300, description="Supplier company name")
    supplier_id_external: Optional[str] = Field(None, description="External supplier ID (e.g., DUNS number)")
    contact_email: str = Field(..., description="Primary contact email")
    contact_name: str = Field("", description="Primary contact name")
    tier: SupplierTier = Field(SupplierTier.TIER_1, description="Supplier tier")
    sector: Optional[str] = Field(None, description="Supplier GICS sector")
    country: Optional[str] = Field(None, description="Supplier country (ISO 3166-1)")
    annual_spend_usd: Optional[float] = Field(None, ge=0, description="Annual spend in USD")
    estimated_emissions_tco2e: Optional[float] = Field(None, ge=0, description="Estimated supplier emissions (tCO2e)")

    class Config:
        json_schema_extra = {
            "example": {
                "supplier_name": "Acme Materials Corp",
                "supplier_id_external": "DUNS-123456789",
                "contact_email": "sustainability@acme-materials.com",
                "contact_name": "John Doe",
                "tier": "tier_1",
                "sector": "materials",
                "country": "US",
                "annual_spend_usd": 5200000.0,
                "estimated_emissions_tco2e": 8500.0,
            }
        }


class InviteSupplierRequest(BaseModel):
    """Request to send a questionnaire invitation to a supplier."""
    supplier_ids: List[str] = Field(
        ..., min_length=1, max_length=500, description="Supplier IDs to invite"
    )
    questionnaire_year: str = Field("2026", description="CDP questionnaire year")
    deadline: Optional[str] = Field(None, description="Response deadline (ISO 8601 date)")
    custom_message: Optional[str] = Field(
        None, max_length=2000, description="Custom invitation message"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "supplier_ids": ["sup_abc123", "sup_def456"],
                "questionnaire_year": "2026",
                "deadline": "2026-07-31",
                "custom_message": "We request your participation in the CDP Climate Change questionnaire.",
            }
        }


class CascadeRequest(BaseModel):
    """Request to cascade engagement to sub-tier suppliers."""
    supplier_id: str = Field(..., description="Tier 1 supplier to cascade through")
    target_tier: SupplierTier = Field(
        SupplierTier.TIER_2, description="Target supplier tier"
    )
    engagement_type: str = Field(
        "climate_questionnaire", description="Type of engagement to cascade"
    )
    message: Optional[str] = Field(
        None, max_length=2000, description="Cascade request message"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "supplier_id": "sup_abc123",
                "target_tier": "tier_2",
                "engagement_type": "climate_questionnaire",
                "message": "Please cascade CDP participation to your key material suppliers.",
            }
        }


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class SupplierResponse(BaseModel):
    """Supplier record."""
    supplier_id: str
    supplier_name: str
    supplier_id_external: Optional[str]
    contact_email: str
    contact_name: str
    tier: str
    sector: Optional[str]
    country: Optional[str]
    annual_spend_usd: Optional[float]
    estimated_emissions_tco2e: Optional[float]
    status: str
    cdp_score: Optional[str]
    cdp_band: Optional[str]
    invited_at: Optional[datetime]
    responded_at: Optional[datetime]
    created_at: datetime


class SupplierListEntry(BaseModel):
    """Summary entry in a supplier list."""
    supplier_id: str
    supplier_name: str
    tier: str
    status: str
    cdp_score: Optional[str]
    annual_spend_usd: Optional[float]
    estimated_emissions_tco2e: Optional[float]
    country: Optional[str]


class InviteResponse(BaseModel):
    """Result of sending questionnaire invitations."""
    invitation_id: str
    total_invited: int
    total_already_invited: int
    total_errors: int
    invited_suppliers: List[str]
    questionnaire_year: str
    deadline: Optional[str]
    sent_at: datetime


class SupplierResponseSummary(BaseModel):
    """Summary of supplier questionnaire responses."""
    total_suppliers: int
    invited: int
    pending: int
    responded: int
    scored: int
    declined: int
    response_rate_pct: float
    avg_supplier_score: Optional[float]
    supplier_responses: List[Dict[str, Any]]


class EngagementDashboardResponse(BaseModel):
    """Supply chain engagement dashboard."""
    total_suppliers: int
    total_spend_usd: float
    total_estimated_emissions_tco2e: float
    engagement_rate_pct: float
    response_rate_pct: float
    avg_supplier_cdp_score: Optional[float]
    tier_breakdown: Dict[str, int]
    status_breakdown: Dict[str, int]
    sector_breakdown: List[Dict[str, Any]]
    country_breakdown: List[Dict[str, Any]]
    year_over_year_improvement: Optional[float]
    generated_at: datetime


class SupplierEmissionsResponse(BaseModel):
    """Aggregated supplier emissions data."""
    total_scope1_tco2e: float
    total_scope2_tco2e: float
    total_scope3_tco2e: float
    total_emissions_tco2e: float
    emissions_by_tier: Dict[str, float]
    emissions_by_sector: List[Dict[str, Any]]
    emissions_by_country: List[Dict[str, Any]]
    data_coverage_pct: float
    supplier_count_with_data: int
    supplier_count_total: int
    calculated_at: datetime


class HotspotResponse(BaseModel):
    """Emission hotspot identification."""
    hotspots: List[Dict[str, Any]]
    top_emitting_suppliers: List[Dict[str, Any]]
    top_emitting_sectors: List[Dict[str, Any]]
    concentration_metrics: Dict[str, Any]
    calculated_at: datetime


class CascadeResponse(BaseModel):
    """Cascade engagement request result."""
    cascade_id: str
    supplier_id: str
    supplier_name: str
    target_tier: str
    engagement_type: str
    status: str
    estimated_sub_tier_count: int
    created_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_suppliers: Dict[str, Dict[str, Any]] = {}
_cascades: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# Seed demo suppliers
_demo_suppliers = [
    {"name": "Acme Materials Corp", "tier": "tier_1", "sector": "materials", "country": "US", "spend": 5200000, "emissions": 8500, "status": "scored", "score": "B", "band": "B"},
    {"name": "Global Chemicals Ltd", "tier": "tier_1", "sector": "materials", "country": "DE", "spend": 3800000, "emissions": 12400, "status": "responded", "score": "C", "band": "C"},
    {"name": "Pacific Transport Co", "tier": "tier_1", "sector": "industrials", "country": "JP", "spend": 2100000, "emissions": 6200, "status": "scored", "score": "B-", "band": "B-"},
    {"name": "Northern Energy Inc", "tier": "tier_1", "sector": "energy", "country": "CA", "spend": 4500000, "emissions": 18900, "status": "invited", "score": None, "band": None},
    {"name": "EcoPackaging GmbH", "tier": "tier_2", "sector": "materials", "country": "AT", "spend": 900000, "emissions": 1800, "status": "scored", "score": "A-", "band": "A-"},
]

for i, demo in enumerate(_demo_suppliers):
    sid = f"sup_demo_{i + 1:03d}"
    _suppliers[sid] = {
        "supplier_id": sid, "supplier_name": demo["name"],
        "supplier_id_external": None, "contact_email": f"contact@{demo['name'].lower().replace(' ', '')}.com",
        "contact_name": "", "tier": demo["tier"], "sector": demo["sector"],
        "country": demo["country"], "annual_spend_usd": demo["spend"],
        "estimated_emissions_tco2e": demo["emissions"], "status": demo["status"],
        "cdp_score": demo["score"], "cdp_band": demo["band"],
        "invited_at": datetime(2025, 3, 1) if demo["status"] != "registered" else None,
        "responded_at": datetime(2025, 6, 15) if demo["status"] in ("responded", "scored") else None,
        "created_at": datetime(2025, 1, 15),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/suppliers",
    response_model=List[SupplierListEntry],
    summary="List suppliers",
    description=(
        "Retrieve all registered suppliers with optional filtering by tier, "
        "status, sector, or country."
    ),
)
async def list_suppliers(
    tier: Optional[str] = Query(None, description="Filter by supplier tier"),
    supplier_status: Optional[str] = Query(None, alias="status", description="Filter by engagement status"),
    sector: Optional[str] = Query(None, description="Filter by GICS sector"),
    country: Optional[str] = Query(None, description="Filter by country code"),
    sort_by: Optional[str] = Query(None, description="Sort by: spend, emissions, name"),
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Results offset for pagination"),
) -> List[SupplierListEntry]:
    """List suppliers."""
    suppliers = list(_suppliers.values())
    if tier:
        suppliers = [s for s in suppliers if s["tier"] == tier]
    if supplier_status:
        suppliers = [s for s in suppliers if s["status"] == supplier_status]
    if sector:
        suppliers = [s for s in suppliers if s.get("sector") == sector]
    if country:
        suppliers = [s for s in suppliers if s.get("country") == country]

    if sort_by == "spend":
        suppliers.sort(key=lambda s: s.get("annual_spend_usd") or 0, reverse=True)
    elif sort_by == "emissions":
        suppliers.sort(key=lambda s: s.get("estimated_emissions_tco2e") or 0, reverse=True)
    elif sort_by == "name":
        suppliers.sort(key=lambda s: s["supplier_name"])
    else:
        suppliers.sort(key=lambda s: s.get("estimated_emissions_tco2e") or 0, reverse=True)

    page = suppliers[offset: offset + limit]
    return [
        SupplierListEntry(
            supplier_id=s["supplier_id"], supplier_name=s["supplier_name"],
            tier=s["tier"], status=s["status"], cdp_score=s.get("cdp_score"),
            annual_spend_usd=s.get("annual_spend_usd"),
            estimated_emissions_tco2e=s.get("estimated_emissions_tco2e"),
            country=s.get("country"),
        )
        for s in page
    ]


@router.post(
    "/suppliers",
    response_model=SupplierResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add supplier",
    description="Register a new supplier in the supply chain module.",
)
async def add_supplier(request: AddSupplierRequest) -> SupplierResponse:
    """Add a supplier."""
    supplier_id = _generate_id("sup")
    now = _now()
    supplier = {
        "supplier_id": supplier_id,
        "supplier_name": request.supplier_name,
        "supplier_id_external": request.supplier_id_external,
        "contact_email": request.contact_email,
        "contact_name": request.contact_name,
        "tier": request.tier.value,
        "sector": request.sector,
        "country": request.country,
        "annual_spend_usd": request.annual_spend_usd,
        "estimated_emissions_tco2e": request.estimated_emissions_tco2e,
        "status": SupplierStatus.REGISTERED.value,
        "cdp_score": None,
        "cdp_band": None,
        "invited_at": None,
        "responded_at": None,
        "created_at": now,
    }
    _suppliers[supplier_id] = supplier
    return SupplierResponse(**supplier)


@router.post(
    "/invite",
    response_model=InviteResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Send questionnaire invitation",
    description=(
        "Send CDP Climate Change questionnaire invitations to selected suppliers. "
        "Suppliers already invited are skipped."
    ),
)
async def invite_suppliers(request: InviteSupplierRequest) -> InviteResponse:
    """Send questionnaire invitations to suppliers."""
    now = _now()
    invited = []
    already_invited = 0
    errors = 0

    for sid in request.supplier_ids:
        supplier = _suppliers.get(sid)
        if not supplier:
            errors += 1
            continue
        if supplier["status"] not in (SupplierStatus.REGISTERED.value, SupplierStatus.DECLINED.value):
            already_invited += 1
            continue
        supplier["status"] = SupplierStatus.INVITED.value
        supplier["invited_at"] = now
        invited.append(sid)

    return InviteResponse(
        invitation_id=_generate_id("inv"),
        total_invited=len(invited),
        total_already_invited=already_invited,
        total_errors=errors,
        invited_suppliers=invited,
        questionnaire_year=request.questionnaire_year,
        deadline=request.deadline,
        sent_at=now,
    )


@router.get(
    "/responses",
    response_model=SupplierResponseSummary,
    summary="Supplier responses",
    description="Retrieve summary of supplier questionnaire responses and engagement metrics.",
)
async def get_supplier_responses(
    questionnaire_year: Optional[str] = Query(None, description="Filter by questionnaire year"),
) -> SupplierResponseSummary:
    """Get supplier response summary."""
    suppliers = list(_suppliers.values())
    total = len(suppliers)
    invited = sum(1 for s in suppliers if s["status"] != SupplierStatus.REGISTERED.value)
    pending = sum(1 for s in suppliers if s["status"] in (SupplierStatus.INVITED.value, SupplierStatus.PENDING.value))
    responded = sum(1 for s in suppliers if s["status"] == SupplierStatus.RESPONDED.value)
    scored = sum(1 for s in suppliers if s["status"] == SupplierStatus.SCORED.value)
    declined = sum(1 for s in suppliers if s["status"] == SupplierStatus.DECLINED.value)
    response_rate = round((responded + scored) / max(1, invited) * 100, 1)

    scores = [s for s in suppliers if s.get("cdp_score")]
    score_map = {"A": 90, "A-": 75, "B": 65, "B-": 55, "C": 45, "C-": 35, "D": 25, "D-": 10}
    avg_score = round(
        sum(score_map.get(s["cdp_score"], 0) for s in scores) / max(1, len(scores)),
        1
    ) if scores else None

    supplier_responses = [
        {
            "supplier_id": s["supplier_id"],
            "supplier_name": s["supplier_name"],
            "status": s["status"],
            "cdp_score": s.get("cdp_score"),
            "responded_at": s.get("responded_at"),
        }
        for s in suppliers if s["status"] != SupplierStatus.REGISTERED.value
    ]

    return SupplierResponseSummary(
        total_suppliers=total,
        invited=invited,
        pending=pending,
        responded=responded,
        scored=scored,
        declined=declined,
        response_rate_pct=response_rate,
        avg_supplier_score=avg_score,
        supplier_responses=supplier_responses,
    )


@router.get(
    "/dashboard",
    response_model=EngagementDashboardResponse,
    summary="Engagement dashboard",
    description=(
        "Retrieve the supply chain engagement dashboard with aggregated "
        "metrics: engagement rate, response rate, tier/sector/country breakdown."
    ),
)
async def get_dashboard() -> EngagementDashboardResponse:
    """Get engagement dashboard."""
    suppliers = list(_suppliers.values())
    total = len(suppliers)
    total_spend = sum(s.get("annual_spend_usd") or 0 for s in suppliers)
    total_emissions = sum(s.get("estimated_emissions_tco2e") or 0 for s in suppliers)
    engaged = sum(1 for s in suppliers if s["status"] not in (SupplierStatus.REGISTERED.value, SupplierStatus.DECLINED.value))
    responded = sum(1 for s in suppliers if s["status"] in (SupplierStatus.RESPONDED.value, SupplierStatus.SCORED.value))

    tier_breakdown = {}
    status_breakdown = {}
    sector_map: Dict[str, int] = {}
    country_map: Dict[str, int] = {}

    for s in suppliers:
        tier_breakdown[s["tier"]] = tier_breakdown.get(s["tier"], 0) + 1
        status_breakdown[s["status"]] = status_breakdown.get(s["status"], 0) + 1
        if s.get("sector"):
            sector_map[s["sector"]] = sector_map.get(s["sector"], 0) + 1
        if s.get("country"):
            country_map[s["country"]] = country_map.get(s["country"], 0) + 1

    sector_breakdown = [{"sector": k, "count": v} for k, v in sector_map.items()]
    country_breakdown = [{"country": k, "count": v} for k, v in country_map.items()]

    return EngagementDashboardResponse(
        total_suppliers=total,
        total_spend_usd=total_spend,
        total_estimated_emissions_tco2e=total_emissions,
        engagement_rate_pct=round(engaged / max(1, total) * 100, 1),
        response_rate_pct=round(responded / max(1, engaged) * 100, 1) if engaged > 0 else 0.0,
        avg_supplier_cdp_score=55.0,
        tier_breakdown=tier_breakdown,
        status_breakdown=status_breakdown,
        sector_breakdown=sector_breakdown,
        country_breakdown=country_breakdown,
        year_over_year_improvement=5.2,
        generated_at=_now(),
    )


@router.get(
    "/emissions",
    response_model=SupplierEmissionsResponse,
    summary="Aggregated supplier emissions",
    description=(
        "Retrieve aggregated emissions data from all suppliers with data, "
        "broken down by tier, sector, and country."
    ),
)
async def get_supplier_emissions() -> SupplierEmissionsResponse:
    """Get aggregated supplier emissions."""
    suppliers = list(_suppliers.values())
    with_data = [s for s in suppliers if s.get("estimated_emissions_tco2e")]
    total = sum(s["estimated_emissions_tco2e"] for s in with_data)

    tier_emissions: Dict[str, float] = {}
    sector_emissions: Dict[str, float] = {}
    country_emissions: Dict[str, float] = {}

    for s in with_data:
        tier_emissions[s["tier"]] = tier_emissions.get(s["tier"], 0) + s["estimated_emissions_tco2e"]
        if s.get("sector"):
            sector_emissions[s["sector"]] = sector_emissions.get(s["sector"], 0) + s["estimated_emissions_tco2e"]
        if s.get("country"):
            country_emissions[s["country"]] = country_emissions.get(s["country"], 0) + s["estimated_emissions_tco2e"]

    return SupplierEmissionsResponse(
        total_scope1_tco2e=round(total * 0.35, 1),
        total_scope2_tco2e=round(total * 0.25, 1),
        total_scope3_tco2e=round(total * 0.40, 1),
        total_emissions_tco2e=round(total, 1),
        emissions_by_tier=tier_emissions,
        emissions_by_sector=[{"sector": k, "tco2e": v} for k, v in sector_emissions.items()],
        emissions_by_country=[{"country": k, "tco2e": v} for k, v in country_emissions.items()],
        data_coverage_pct=round(len(with_data) / max(1, len(suppliers)) * 100, 1),
        supplier_count_with_data=len(with_data),
        supplier_count_total=len(suppliers),
        calculated_at=_now(),
    )


@router.get(
    "/hotspots",
    response_model=HotspotResponse,
    summary="Emission hotspots",
    description=(
        "Identify emission hotspots in the supply chain. Shows top emitting "
        "suppliers, sectors, and concentration metrics."
    ),
)
async def get_hotspots(
    top_n: int = Query(10, ge=1, le=50, description="Number of top hotspots to return"),
) -> HotspotResponse:
    """Identify emission hotspots."""
    suppliers = list(_suppliers.values())
    with_data = [s for s in suppliers if s.get("estimated_emissions_tco2e")]
    with_data.sort(key=lambda s: s["estimated_emissions_tco2e"], reverse=True)
    total = sum(s["estimated_emissions_tco2e"] for s in with_data)

    hotspots = [
        {
            "supplier_id": s["supplier_id"],
            "supplier_name": s["supplier_name"],
            "emissions_tco2e": s["estimated_emissions_tco2e"],
            "pct_of_total": round(s["estimated_emissions_tco2e"] / max(1, total) * 100, 1),
            "sector": s.get("sector"),
            "country": s.get("country"),
            "tier": s["tier"],
        }
        for s in with_data[:top_n]
    ]

    top_suppliers = hotspots[:5]

    sector_emissions: Dict[str, float] = {}
    for s in with_data:
        sec = s.get("sector", "unknown")
        sector_emissions[sec] = sector_emissions.get(sec, 0) + s["estimated_emissions_tco2e"]
    top_sectors = sorted(
        [{"sector": k, "tco2e": v, "pct": round(v / max(1, total) * 100, 1)} for k, v in sector_emissions.items()],
        key=lambda x: x["tco2e"],
        reverse=True,
    )[:5]

    # Concentration metrics
    top_5_pct = round(sum(s["estimated_emissions_tco2e"] for s in with_data[:5]) / max(1, total) * 100, 1) if len(with_data) >= 5 else 100.0
    top_10_pct = round(sum(s["estimated_emissions_tco2e"] for s in with_data[:10]) / max(1, total) * 100, 1) if len(with_data) >= 10 else 100.0

    return HotspotResponse(
        hotspots=hotspots,
        top_emitting_suppliers=top_suppliers,
        top_emitting_sectors=top_sectors,
        concentration_metrics={
            "top_5_suppliers_pct": top_5_pct,
            "top_10_suppliers_pct": top_10_pct,
            "herfindahl_index": round(sum((s["estimated_emissions_tco2e"] / max(1, total)) ** 2 for s in with_data) * 10000, 1),
        },
        calculated_at=_now(),
    )


@router.post(
    "/cascade",
    response_model=CascadeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Cascade engagement request",
    description=(
        "Send a cascade engagement request to a Tier 1 supplier, asking them "
        "to extend CDP participation to their own suppliers (Tier 2+)."
    ),
)
async def cascade_engagement(request: CascadeRequest) -> CascadeResponse:
    """Send cascade engagement request."""
    supplier = _suppliers.get(request.supplier_id)
    if not supplier:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Supplier {request.supplier_id} not found",
        )
    if supplier["status"] not in (SupplierStatus.RESPONDED.value, SupplierStatus.SCORED.value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Supplier must have responded or been scored before cascade can be initiated.",
        )

    cascade_id = _generate_id("cas")
    now = _now()
    cascade = {
        "cascade_id": cascade_id,
        "supplier_id": request.supplier_id,
        "supplier_name": supplier["supplier_name"],
        "target_tier": request.target_tier.value,
        "engagement_type": request.engagement_type,
        "status": CascadeStatus.SENT.value,
        "estimated_sub_tier_count": 15,
        "created_at": now,
    }
    _cascades[cascade_id] = cascade
    return CascadeResponse(**cascade)
