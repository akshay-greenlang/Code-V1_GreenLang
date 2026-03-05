"""
GL-TCFD-APP Metrics & Targets API

TCFD Pillar 4 -- Metrics and Targets.  Manages GHG emissions metrics (Scope
1/2/3), cross-industry metrics per ISSB/IFRS S2, industry-specific metrics,
intensity metrics, climate targets (including SBTi alignment), target progress
tracking, gap-to-target analysis, peer benchmarking, trend analysis, and
metrics disclosure text generation.

TCFD Recommended Disclosures (Metrics & Targets):
    a) Metrics used to assess climate-related risks and opportunities
    b) Scope 1, 2, 3 GHG emissions
    c) Targets used to manage climate risks/opportunities and performance

ISSB/IFRS S2 seven cross-industry metrics: GHG emissions, transition risks,
physical risks, climate opportunities, capital deployment, internal carbon
price, remuneration linkage.
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/tcfd/metrics", tags=["Metrics & Targets"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MetricType(str, Enum):
    GHG_EMISSIONS = "ghg_emissions"
    INTENSITY = "intensity"
    ENERGY = "energy"
    WATER = "water"
    WASTE = "waste"
    FINANCIAL = "financial"
    CUSTOM = "custom"


class TargetType(str, Enum):
    ABSOLUTE = "absolute"
    INTENSITY = "intensity"
    NET_ZERO = "net_zero"
    RENEWABLE_ENERGY = "renewable_energy"
    CUSTOM = "custom"


class SBTiStatus(str, Enum):
    NOT_COMMITTED = "not_committed"
    COMMITTED = "committed"
    TARGET_SET = "target_set"
    VALIDATED = "validated"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class RecordMetricRequest(BaseModel):
    """Request to record a climate metric."""
    metric_name: str = Field(..., min_length=1, max_length=300, description="Metric name")
    metric_type: MetricType = Field(..., description="Metric type")
    value: float = Field(..., description="Metric value")
    unit: str = Field(..., min_length=1, max_length=50, description="Measurement unit")
    reporting_year: int = Field(..., ge=2015, le=2100, description="Reporting year")
    scope: Optional[str] = Field(None, description="GHG scope if applicable: scope_1, scope_2, scope_3")
    methodology: Optional[str] = Field(None, max_length=500, description="Calculation methodology")
    data_quality: Optional[str] = Field(None, description="Data quality: high, medium, low")
    verified: bool = Field(False, description="Whether independently verified")

    class Config:
        json_schema_extra = {
            "example": {
                "metric_name": "Total Scope 1 GHG Emissions",
                "metric_type": "ghg_emissions",
                "value": 25000,
                "unit": "tCO2e",
                "reporting_year": 2025,
                "scope": "scope_1",
                "methodology": "Calculation based (activity data x emission factor x GWP)",
                "data_quality": "high",
                "verified": True,
            }
        }


class CreateTargetRequest(BaseModel):
    """Request to create a climate target."""
    target_name: str = Field(..., min_length=1, max_length=300, description="Target name")
    target_type: TargetType = Field(..., description="Target type")
    metric_name: str = Field(..., description="Related metric name")
    base_year: int = Field(..., ge=2000, le=2100, description="Base year")
    base_value: float = Field(..., description="Base year value")
    target_year: int = Field(..., ge=2025, le=2100, description="Target year")
    target_value: float = Field(..., description="Target value")
    interim_targets: Optional[Dict[str, float]] = Field(None, description="Interim year targets")
    is_science_based: bool = Field(False, description="Is this a science-based target")
    sbti_status: SBTiStatus = Field(SBTiStatus.NOT_COMMITTED, description="SBTi status")

    class Config:
        json_schema_extra = {
            "example": {
                "target_name": "50% Scope 1+2 reduction by 2030",
                "target_type": "absolute",
                "metric_name": "Scope 1+2 GHG Emissions",
                "base_year": 2020,
                "base_value": 50000,
                "target_year": 2030,
                "target_value": 25000,
                "interim_targets": {"2025": 37500},
                "is_science_based": True,
                "sbti_status": "validated",
            }
        }


class UpdateTargetRequest(BaseModel):
    """Request to update a target."""
    target_name: Optional[str] = Field(None, max_length=300)
    target_value: Optional[float] = None
    interim_targets: Optional[Dict[str, float]] = None
    sbti_status: Optional[SBTiStatus] = None


class RecordProgressRequest(BaseModel):
    """Request to record progress against a target."""
    reporting_year: int = Field(..., ge=2015, le=2100, description="Reporting year")
    actual_value: float = Field(..., description="Actual value achieved")
    notes: Optional[str] = Field(None, max_length=2000, description="Progress notes")


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class MetricResponse(BaseModel):
    """A recorded climate metric."""
    metric_id: str
    org_id: str
    metric_name: str
    metric_type: str
    value: float
    unit: str
    reporting_year: int
    scope: Optional[str]
    methodology: Optional[str]
    data_quality: Optional[str]
    verified: bool
    created_at: datetime


class CrossIndustryMetricsResponse(BaseModel):
    """ISSB 7 cross-industry metrics."""
    org_id: str
    ghg_emissions: Dict[str, float]
    transition_risk_amount_usd: float
    transition_risk_pct_of_assets: float
    physical_risk_amount_usd: float
    physical_risk_pct_of_assets: float
    climate_opportunity_amount_usd: float
    climate_opportunity_pct_of_revenue: float
    capital_deployed_to_climate_usd: float
    internal_carbon_price_usd: float
    remuneration_linked_to_climate_pct: float
    reporting_year: int
    generated_at: datetime


class IndustryMetricsResponse(BaseModel):
    """Industry-specific metrics."""
    org_id: str
    industry: str
    metrics: List[Dict[str, Any]]
    issb_industry_standard: str
    generated_at: datetime


class GHGEmissionsResponse(BaseModel):
    """GHG emissions summary."""
    org_id: str
    reporting_year: int
    scope_1_tco2e: float
    scope_2_location_tco2e: float
    scope_2_market_tco2e: float
    scope_3_tco2e: float
    total_tco2e: float
    by_scope_3_category: Optional[Dict[str, float]]
    yoy_change_pct: Optional[float]
    generated_at: datetime


class IntensityMetricsResponse(BaseModel):
    """Intensity metrics."""
    org_id: str
    reporting_year: int
    metrics: List[Dict[str, Any]]
    generated_at: datetime


class TargetResponse(BaseModel):
    """A climate target."""
    target_id: str
    org_id: str
    target_name: str
    target_type: str
    metric_name: str
    base_year: int
    base_value: float
    target_year: int
    target_value: float
    current_value: Optional[float]
    progress_pct: Optional[float]
    interim_targets: Optional[Dict[str, float]]
    is_science_based: bool
    sbti_status: str
    on_track: Optional[bool]
    created_at: datetime
    updated_at: datetime


class ProgressResponse(BaseModel):
    """Progress record for a target."""
    progress_id: str
    target_id: str
    reporting_year: int
    actual_value: float
    progress_pct: float
    on_track: bool
    notes: Optional[str]
    recorded_at: datetime


class SBTiAlignmentResponse(BaseModel):
    """SBTi alignment assessment."""
    org_id: str
    sbti_status: str
    near_term_target: Optional[Dict[str, Any]]
    long_term_target: Optional[Dict[str, Any]]
    net_zero_target: Optional[Dict[str, Any]]
    alignment_pathway: str
    gap_to_1_5c_pct: float
    recommendations: List[str]
    generated_at: datetime


class GapToTargetResponse(BaseModel):
    """Gap-to-target analysis."""
    org_id: str
    targets: List[Dict[str, Any]]
    total_gap_tco2e: float
    gap_closure_rate_pct: float
    years_to_close_gap: Optional[float]
    recommended_actions: List[str]
    generated_at: datetime


class MetricsBenchmarkResponse(BaseModel):
    """Peer benchmarking for metrics."""
    org_id: str
    org_emissions_tco2e: float
    org_intensity: float
    peer_avg_emissions_tco2e: float
    peer_avg_intensity: float
    percentile_rank: int
    peer_count: int
    comparison: Dict[str, Dict[str, float]]
    generated_at: datetime


class TrendAnalysisResponse(BaseModel):
    """Trend analysis across years."""
    org_id: str
    metric_name: str
    data_points: Dict[str, float]
    trend_direction: str
    cagr_pct: Optional[float]
    generated_at: datetime


class MetricsDisclosureResponse(BaseModel):
    """Metrics disclosure text."""
    org_id: str
    pillar: str
    disclosure_a: str
    disclosure_b: str
    disclosure_c: str
    word_count: int
    compliance_score: float
    issb_references: List[str]
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_metrics: Dict[str, Dict[str, Any]] = {}
_targets: Dict[str, Dict[str, Any]] = {}
_progress: Dict[str, List[Dict[str, Any]]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=MetricResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record metric",
    description="Record a climate-related metric (GHG emissions, energy, water, intensity, etc.).",
)
async def record_metric(
    org_id: str = Query(..., description="Organization ID"),
    request: RecordMetricRequest = ...,
) -> MetricResponse:
    """Record a climate metric."""
    metric_id = _generate_id("met")
    metric = {
        "metric_id": metric_id,
        "org_id": org_id,
        "metric_name": request.metric_name,
        "metric_type": request.metric_type.value,
        "value": request.value,
        "unit": request.unit,
        "reporting_year": request.reporting_year,
        "scope": request.scope,
        "methodology": request.methodology,
        "data_quality": request.data_quality,
        "verified": request.verified,
        "created_at": _now(),
    }
    _metrics[metric_id] = metric
    return MetricResponse(**metric)


@router.get(
    "/{org_id}",
    response_model=List[MetricResponse],
    summary="List metrics",
    description="Retrieve all recorded metrics for an organization.",
)
async def list_metrics(
    org_id: str,
    metric_type: Optional[str] = Query(None, description="Filter by metric type"),
    reporting_year: Optional[int] = Query(None, description="Filter by year"),
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
) -> List[MetricResponse]:
    """List metrics."""
    results = [m for m in _metrics.values() if m["org_id"] == org_id]
    if metric_type:
        results = [m for m in results if m["metric_type"] == metric_type]
    if reporting_year:
        results = [m for m in results if m["reporting_year"] == reporting_year]
    results.sort(key=lambda m: m["reporting_year"], reverse=True)
    return [MetricResponse(**m) for m in results[:limit]]


@router.get(
    "/cross-industry/{org_id}",
    response_model=CrossIndustryMetricsResponse,
    summary="ISSB 7 cross-industry metrics",
    description="Return the seven ISSB/IFRS S2 cross-industry climate metrics for an organization.",
)
async def get_cross_industry_metrics(
    org_id: str,
    reporting_year: int = Query(2025, ge=2015, le=2100),
) -> CrossIndustryMetricsResponse:
    """Get ISSB cross-industry metrics."""
    return CrossIndustryMetricsResponse(
        org_id=org_id,
        ghg_emissions={"scope_1": 25000, "scope_2_location": 15000, "scope_2_market": 12000, "scope_3": 120000, "total": 172000},
        transition_risk_amount_usd=85000000,
        transition_risk_pct_of_assets=10.0,
        physical_risk_amount_usd=45000000,
        physical_risk_pct_of_assets=5.3,
        climate_opportunity_amount_usd=60000000,
        climate_opportunity_pct_of_revenue=12.0,
        capital_deployed_to_climate_usd=35000000,
        internal_carbon_price_usd=75,
        remuneration_linked_to_climate_pct=15,
        reporting_year=reporting_year,
        generated_at=_now(),
    )


@router.get(
    "/industry/{org_id}/{industry}",
    response_model=IndustryMetricsResponse,
    summary="Industry-specific metrics",
    description="Get industry-specific climate metrics based on ISSB industry standards.",
)
async def get_industry_metrics(org_id: str, industry: str) -> IndustryMetricsResponse:
    """Get industry-specific metrics."""
    industry_data = {
        "energy": [
            {"metric": "Scope 1 emissions from methane", "value": 5000, "unit": "tCO2e"},
            {"metric": "Gross global Scope 1 CO2 emissions from power generation", "value": 18000, "unit": "tCO2e"},
            {"metric": "Percentage of power generation from renewables", "value": 35, "unit": "%"},
        ],
        "manufacturing": [
            {"metric": "Process emissions", "value": 8000, "unit": "tCO2e"},
            {"metric": "Energy intensity per unit of output", "value": 0.45, "unit": "MWh/unit"},
            {"metric": "Waste recycling rate", "value": 72, "unit": "%"},
        ],
        "financial": [
            {"metric": "Financed emissions (PCAF)", "value": 500000, "unit": "tCO2e"},
            {"metric": "WACI portfolio", "value": 125, "unit": "tCO2e/$M revenue"},
            {"metric": "Green asset ratio", "value": 18, "unit": "%"},
        ],
    }
    metrics = industry_data.get(industry.lower(), [
        {"metric": "Total GHG emissions", "value": 40000, "unit": "tCO2e"},
        {"metric": "Energy consumption", "value": 120000, "unit": "MWh"},
    ])

    return IndustryMetricsResponse(
        org_id=org_id,
        industry=industry,
        metrics=metrics,
        issb_industry_standard=f"IFRS S2 Industry-based Guidance: {industry.title()}",
        generated_at=_now(),
    )


@router.get(
    "/ghg/{org_id}/{year}",
    response_model=GHGEmissionsResponse,
    summary="GHG emissions (Scope 1/2/3)",
    description="Get GHG emissions breakdown by scope for a reporting year.",
)
async def get_ghg_emissions(org_id: str, year: int) -> GHGEmissionsResponse:
    """Get GHG emissions summary."""
    s1 = 25000.0
    s2_loc = 15000.0
    s2_mkt = 12000.0
    s3 = 120000.0
    total = s1 + s2_mkt + s3

    by_s3 = {
        "cat_1_purchased_goods": 45000, "cat_2_capital_goods": 12000,
        "cat_3_fuel_energy": 8000, "cat_4_upstream_transport": 15000,
        "cat_5_waste": 3000, "cat_6_business_travel": 5000,
        "cat_7_employee_commuting": 4000, "cat_11_use_of_sold_products": 18000,
        "cat_15_investments": 10000,
    }

    return GHGEmissionsResponse(
        org_id=org_id,
        reporting_year=year,
        scope_1_tco2e=s1,
        scope_2_location_tco2e=s2_loc,
        scope_2_market_tco2e=s2_mkt,
        scope_3_tco2e=s3,
        total_tco2e=total,
        by_scope_3_category=by_s3,
        yoy_change_pct=-4.2,
        generated_at=_now(),
    )


@router.get(
    "/intensity/{org_id}/{year}",
    response_model=IntensityMetricsResponse,
    summary="Intensity metrics",
    description="Get emissions intensity metrics normalized by revenue, employees, and production.",
)
async def get_intensity_metrics(org_id: str, year: int) -> IntensityMetricsResponse:
    """Get intensity metrics."""
    return IntensityMetricsResponse(
        org_id=org_id,
        reporting_year=year,
        metrics=[
            {"metric": "Revenue intensity", "value": 80.0, "unit": "tCO2e/$M revenue", "yoy_change_pct": -5.2},
            {"metric": "Employee intensity", "value": 8.5, "unit": "tCO2e/FTE", "yoy_change_pct": -3.8},
            {"metric": "Production intensity", "value": 0.45, "unit": "tCO2e/unit", "yoy_change_pct": -6.1},
            {"metric": "Floor area intensity", "value": 0.12, "unit": "tCO2e/m2", "yoy_change_pct": -2.5},
        ],
        generated_at=_now(),
    )


@router.post(
    "/targets",
    response_model=TargetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create target",
    description="Create a climate target with base year, target year, and optional interim milestones.",
)
async def create_target(
    org_id: str = Query(..., description="Organization ID"),
    request: CreateTargetRequest = ...,
) -> TargetResponse:
    """Create a climate target."""
    target_id = _generate_id("tgt")
    now = _now()
    target = {
        "target_id": target_id,
        "org_id": org_id,
        "target_name": request.target_name,
        "target_type": request.target_type.value,
        "metric_name": request.metric_name,
        "base_year": request.base_year,
        "base_value": request.base_value,
        "target_year": request.target_year,
        "target_value": request.target_value,
        "current_value": None,
        "progress_pct": None,
        "interim_targets": request.interim_targets,
        "is_science_based": request.is_science_based,
        "sbti_status": request.sbti_status.value,
        "on_track": None,
        "created_at": now,
        "updated_at": now,
    }
    _targets[target_id] = target
    _progress[target_id] = []
    return TargetResponse(**target)


@router.get(
    "/targets/{org_id}",
    response_model=List[TargetResponse],
    summary="List targets",
    description="List all climate targets for an organization.",
)
async def list_targets(
    org_id: str,
    target_type: Optional[str] = Query(None, description="Filter by target type"),
) -> List[TargetResponse]:
    """List climate targets."""
    results = [t for t in _targets.values() if t["org_id"] == org_id]
    if target_type:
        results = [t for t in results if t["target_type"] == target_type]
    results.sort(key=lambda t: t["target_year"])
    return [TargetResponse(**t) for t in results]


@router.put(
    "/targets/{target_id}",
    response_model=TargetResponse,
    summary="Update target",
    description="Update an existing climate target.",
)
async def update_target(target_id: str, request: UpdateTargetRequest) -> TargetResponse:
    """Update a climate target."""
    target = _targets.get(target_id)
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Target {target_id} not found")
    updates = request.model_dump(exclude_unset=True)
    if "sbti_status" in updates and hasattr(updates["sbti_status"], "value"):
        updates["sbti_status"] = updates["sbti_status"].value
    target.update(updates)
    target["updated_at"] = _now()
    return TargetResponse(**target)


@router.post(
    "/targets/{target_id}/progress",
    response_model=ProgressResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record progress",
    description="Record progress against a climate target for a specific reporting year.",
)
async def record_progress(target_id: str, request: RecordProgressRequest) -> ProgressResponse:
    """Record progress against a target."""
    target = _targets.get(target_id)
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Target {target_id} not found")

    total_change = target["base_value"] - target["target_value"]
    actual_change = target["base_value"] - request.actual_value
    progress_pct = round(actual_change / total_change * 100, 1) if total_change != 0 else 0
    years_elapsed = request.reporting_year - target["base_year"]
    years_total = target["target_year"] - target["base_year"]
    expected_pct = round(years_elapsed / years_total * 100, 1) if years_total > 0 else 0
    on_track = progress_pct >= expected_pct * 0.9

    target["current_value"] = request.actual_value
    target["progress_pct"] = progress_pct
    target["on_track"] = on_track
    target["updated_at"] = _now()

    progress_id = _generate_id("prg")
    entry = {
        "progress_id": progress_id,
        "target_id": target_id,
        "reporting_year": request.reporting_year,
        "actual_value": request.actual_value,
        "progress_pct": progress_pct,
        "on_track": on_track,
        "notes": request.notes,
        "recorded_at": _now(),
    }
    _progress.setdefault(target_id, []).append(entry)
    return ProgressResponse(**entry)


@router.get(
    "/targets/{target_id}/progress",
    response_model=List[ProgressResponse],
    summary="Get progress history",
    description="Retrieve progress history for a target across all reporting years.",
)
async def get_progress_history(target_id: str) -> List[ProgressResponse]:
    """Get progress history."""
    if target_id not in _targets:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Target {target_id} not found")
    entries = _progress.get(target_id, [])
    return [ProgressResponse(**e) for e in sorted(entries, key=lambda e: e["reporting_year"])]


@router.get(
    "/sbti/{org_id}",
    response_model=SBTiAlignmentResponse,
    summary="SBTi alignment check",
    description="Check the organization's alignment with Science Based Targets initiative requirements.",
)
async def check_sbti_alignment(org_id: str) -> SBTiAlignmentResponse:
    """Check SBTi alignment."""
    org_targets = [t for t in _targets.values() if t["org_id"] == org_id and t["is_science_based"]]
    sbti_targets = [t for t in org_targets if t["sbti_status"] in ("target_set", "validated")]

    near_term = None
    long_term = None
    net_zero = None
    for t in sbti_targets:
        if t["target_year"] <= 2030:
            near_term = {"name": t["target_name"], "year": t["target_year"], "reduction_pct": round((1 - t["target_value"] / t["base_value"]) * 100, 1) if t["base_value"] > 0 else 0}
        elif t["target_type"] == "net_zero":
            net_zero = {"name": t["target_name"], "year": t["target_year"]}
        elif t["target_year"] <= 2050:
            long_term = {"name": t["target_name"], "year": t["target_year"], "reduction_pct": round((1 - t["target_value"] / t["base_value"]) * 100, 1) if t["base_value"] > 0 else 0}

    overall_status = "validated" if any(t["sbti_status"] == "validated" for t in org_targets) else "committed" if org_targets else "not_committed"
    gap = 25.0 if not near_term else max(0, 50 - (near_term.get("reduction_pct", 0)))

    recs = []
    if not near_term:
        recs.append("Set a near-term (2030) science-based target covering Scope 1+2")
    if not long_term:
        recs.append("Set a long-term (2050) science-based target")
    if not net_zero:
        recs.append("Commit to a net-zero target aligned with SBTi Net-Zero Standard")
    if overall_status != "validated":
        recs.append("Submit targets to SBTi for validation")

    return SBTiAlignmentResponse(
        org_id=org_id,
        sbti_status=overall_status,
        near_term_target=near_term,
        long_term_target=long_term,
        net_zero_target=net_zero,
        alignment_pathway="1.5C" if near_term and near_term.get("reduction_pct", 0) >= 42 else "Well Below 2C" if near_term else "Not aligned",
        gap_to_1_5c_pct=gap,
        recommendations=recs,
        generated_at=_now(),
    )


@router.get(
    "/gap-to-target/{org_id}",
    response_model=GapToTargetResponse,
    summary="Gap analysis",
    description="Analyze the gap between current performance and climate targets.",
)
async def get_gap_to_target(org_id: str) -> GapToTargetResponse:
    """Analyze gap to targets."""
    org_targets = [t for t in _targets.values() if t["org_id"] == org_id]
    target_data = []
    total_gap = 0.0
    for t in org_targets:
        current = t.get("current_value") or t["base_value"] * 0.85
        gap = max(current - t["target_value"], 0)
        total_gap += gap
        target_data.append({
            "target_id": t["target_id"],
            "name": t["target_name"],
            "current_value": current,
            "target_value": t["target_value"],
            "gap": round(gap, 2),
            "progress_pct": t.get("progress_pct") or 0,
            "on_track": t.get("on_track"),
        })

    closure_rate = 5.0
    years = round(total_gap / (closure_rate * 1000), 1) if total_gap > 0 else 0

    return GapToTargetResponse(
        org_id=org_id,
        targets=target_data,
        total_gap_tco2e=round(total_gap, 2),
        gap_closure_rate_pct=closure_rate,
        years_to_close_gap=years if years > 0 else None,
        recommended_actions=[
            "Accelerate energy efficiency programs",
            "Scale renewable energy procurement",
            "Engage top suppliers on Scope 3 reduction",
            "Invest in process decarbonization technologies",
        ],
        generated_at=_now(),
    )


@router.get(
    "/benchmark/{org_id}",
    response_model=MetricsBenchmarkResponse,
    summary="Peer benchmarking",
    description="Benchmark the organization's climate metrics against sector peers.",
)
async def get_metrics_benchmark(org_id: str) -> MetricsBenchmarkResponse:
    """Benchmark metrics against peers."""
    org_emissions = 40000.0
    org_intensity = 80.0
    peer_emissions = [30000, 35000, 38000, 42000, 45000, 50000, 55000, 60000, 65000, 75000]
    peer_intensity = [60, 65, 70, 75, 82, 88, 95, 100, 110, 120]

    peer_avg_em = round(sum(peer_emissions) / len(peer_emissions), 1)
    peer_avg_int = round(sum(peer_intensity) / len(peer_intensity), 1)
    below = sum(1 for p in peer_emissions if p >= org_emissions)
    percentile = round(below / len(peer_emissions) * 100)

    return MetricsBenchmarkResponse(
        org_id=org_id,
        org_emissions_tco2e=org_emissions,
        org_intensity=org_intensity,
        peer_avg_emissions_tco2e=peer_avg_em,
        peer_avg_intensity=peer_avg_int,
        percentile_rank=percentile,
        peer_count=len(peer_emissions),
        comparison={
            "emissions": {"org": org_emissions, "peer_avg": peer_avg_em},
            "intensity": {"org": org_intensity, "peer_avg": peer_avg_int},
        },
        generated_at=_now(),
    )


@router.get(
    "/trends/{org_id}",
    response_model=TrendAnalysisResponse,
    summary="Trend analysis",
    description="Analyze multi-year trends for climate metrics.",
)
async def get_trend_analysis(
    org_id: str,
    metric_name: str = Query("Total GHG Emissions", description="Metric to analyze"),
) -> TrendAnalysisResponse:
    """Analyze metric trends."""
    data_points = {
        "2020": 50000, "2021": 48500, "2022": 46800,
        "2023": 44200, "2024": 42100, "2025": 40000,
    }
    values = list(data_points.values())
    if len(values) >= 2 and values[0] > 0:
        n = len(values) - 1
        cagr = round(((values[-1] / values[0]) ** (1 / n) - 1) * 100, 2)
    else:
        cagr = None
    direction = "decreasing" if cagr and cagr < 0 else "increasing" if cagr and cagr > 0 else "stable"

    return TrendAnalysisResponse(
        org_id=org_id,
        metric_name=metric_name,
        data_points=data_points,
        trend_direction=direction,
        cagr_pct=cagr,
        generated_at=_now(),
    )


@router.get(
    "/disclosure/{org_id}",
    response_model=MetricsDisclosureResponse,
    summary="Metrics disclosure text",
    description="Generate TCFD-aligned metrics and targets disclosure text.",
)
async def get_metrics_disclosure(org_id: str) -> MetricsDisclosureResponse:
    """Generate metrics disclosure text."""
    org_targets = [t for t in _targets.values() if t["org_id"] == org_id]

    disclosure_a = (
        "The organization tracks a comprehensive set of climate-related metrics "
        "aligned with ISSB/IFRS S2 cross-industry requirements. These include "
        "absolute and intensity-based GHG emissions across Scopes 1, 2, and 3, "
        "energy consumption and mix, water usage, waste metrics, and financial "
        "metrics such as internal carbon price, capital deployed to climate-related "
        "initiatives, and the proportion of executive remuneration linked to climate targets."
    )

    disclosure_b = (
        "The organization reports GHG emissions for Scope 1 (direct), Scope 2 "
        "(both location-based and market-based), and all material Scope 3 categories "
        "in accordance with the GHG Protocol. Emissions are calculated using "
        "recognized methodologies and independently verified. Year-over-year trends "
        "demonstrate a consistent reduction trajectory."
    )

    target_text = f"{len(org_targets)} climate target(s) are in place" if org_targets else "Climate targets are being developed"
    sbti_targets = [t for t in org_targets if t.get("is_science_based")]
    sbti_text = f", including {len(sbti_targets)} science-based target(s)" if sbti_targets else ""

    disclosure_c = (
        f"{target_text}{sbti_text}. Targets cover Scope 1+2 absolute emissions "
        f"reduction, emission intensity per unit of revenue, and renewable energy "
        f"procurement. Progress is tracked annually against base year baselines "
        f"with interim milestones."
    )

    word_count = sum(len(d.split()) for d in [disclosure_a, disclosure_b, disclosure_c])
    score = min(40 + len(org_targets) * 10 + len(sbti_targets) * 10, 100.0)

    return MetricsDisclosureResponse(
        org_id=org_id,
        pillar="metrics_and_targets",
        disclosure_a=disclosure_a,
        disclosure_b=disclosure_b,
        disclosure_c=disclosure_c,
        word_count=word_count,
        compliance_score=round(score, 1),
        issb_references=[
            "IFRS S2 para 29(a)-(g)", "IFRS S2 para 29(a) - GHG emissions",
            "IFRS S2 para 33-35 - Targets",
        ],
        generated_at=_now(),
    )
