"""
GL-TCFD-APP Scenario Analysis API

TCFD Strategy Recommended Disclosure (c) -- Resilience of the organization's
strategy under different climate-related scenarios, including a 2C or lower
scenario.  Provides pre-built IEA and NGFS scenarios, custom scenario
creation, comparative analysis, sensitivity testing, and resilience scoring.

Pre-built Scenario Families:
    IEA  -- International Energy Agency (World Energy Outlook)
        NZE 2050  (Net Zero Emissions by 2050, ~1.5C)
        APS       (Announced Pledges Scenario, ~1.7C)
        STEPS     (Stated Policies Scenario, ~2.5C)
    NGFS -- Network for Greening the Financial System
        Net Zero 2050       (~1.5C)
        Below 2 Degrees     (~1.7C)
        Divergent Net Zero  (~1.5C, disorderly)
        Delayed Transition  (~1.8C)
        NDCs                (~2.5C)
        Current Policies    (>3C)

ISSB/IFRS S2 references: paragraphs 22 (climate resilience), B1-B18.
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/tcfd/scenarios", tags=["Scenario Analysis"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ScenarioFamily(str, Enum):
    """Scenario family / provider."""
    IEA = "iea"
    NGFS = "ngfs"
    IPCC = "ipcc"
    CUSTOM = "custom"


class ScenarioType(str, Enum):
    """Scenario archetype classification."""
    ORDERLY = "orderly"
    DISORDERLY = "disorderly"
    HOT_HOUSE = "hot_house"
    CUSTOM = "custom"


class TemperatureOutcome(str, Enum):
    """Temperature outcome of a scenario."""
    BELOW_1_5C = "below_1_5c"
    C_1_5 = "1_5c"
    C_1_7 = "1_7c"
    C_2_0 = "2_0c"
    C_2_5 = "2_5c"
    ABOVE_3C = "above_3c"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class CreateScenarioRequest(BaseModel):
    """Request to create a custom scenario."""
    scenario_name: str = Field(..., min_length=1, max_length=300, description="Scenario name")
    description: str = Field(..., min_length=1, max_length=5000, description="Scenario narrative")
    scenario_type: ScenarioType = Field(..., description="Archetype classification")
    temperature_outcome: TemperatureOutcome = Field(..., description="Temperature outcome")
    time_horizon_years: int = Field(30, ge=5, le=100, description="Projection horizon in years")
    carbon_price_2030_usd: Optional[float] = Field(None, ge=0, description="Carbon price in 2030 (USD/tCO2)")
    carbon_price_2050_usd: Optional[float] = Field(None, ge=0, description="Carbon price in 2050 (USD/tCO2)")
    renewable_share_2030_pct: Optional[float] = Field(None, ge=0, le=100)
    renewable_share_2050_pct: Optional[float] = Field(None, ge=0, le=100)
    gdp_impact_pct: Optional[float] = Field(None, description="GDP impact relative to baseline (%)")
    physical_damage_pct: Optional[float] = Field(None, ge=0, description="Physical climate damage as pct GDP")
    policy_stringency: Optional[str] = Field(None, max_length=200)
    technology_assumptions: Optional[str] = Field(None, max_length=2000)
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")

    class Config:
        json_schema_extra = {
            "example": {
                "scenario_name": "Aggressive Green Transition",
                "description": "Rapid policy-driven decarbonization with early carbon pricing",
                "scenario_type": "orderly",
                "temperature_outcome": "1_5c",
                "time_horizon_years": 30,
                "carbon_price_2030_usd": 130.0,
                "carbon_price_2050_usd": 250.0,
                "renewable_share_2030_pct": 65.0,
                "renewable_share_2050_pct": 95.0,
            }
        }


class UpdateScenarioRequest(BaseModel):
    """Request to update a custom scenario."""
    scenario_name: Optional[str] = Field(None, min_length=1, max_length=300)
    description: Optional[str] = Field(None, max_length=5000)
    carbon_price_2030_usd: Optional[float] = Field(None, ge=0)
    carbon_price_2050_usd: Optional[float] = Field(None, ge=0)
    renewable_share_2030_pct: Optional[float] = Field(None, ge=0, le=100)
    renewable_share_2050_pct: Optional[float] = Field(None, ge=0, le=100)
    gdp_impact_pct: Optional[float] = None
    physical_damage_pct: Optional[float] = Field(None, ge=0)
    technology_assumptions: Optional[str] = Field(None, max_length=2000)
    parameters: Optional[Dict[str, Any]] = None


class RunAnalysisRequest(BaseModel):
    """Request to run scenario analysis."""
    org_id: str = Field(..., description="Organization ID")
    scenario_ids: List[str] = Field(..., min_length=1, max_length=10, description="Scenario IDs")
    base_revenue_usd: float = Field(..., gt=0, description="Base annual revenue (USD)")
    base_emissions_tco2e: float = Field(..., ge=0, description="Base annual emissions (tCO2e)")
    base_assets_usd: float = Field(..., gt=0, description="Total asset value (USD)")
    discount_rate_pct: float = Field(8.0, ge=0, le=50, description="Discount rate (%)")
    analysis_horizon_years: int = Field(30, ge=5, le=100, description="Analysis period (years)")

    class Config:
        json_schema_extra = {
            "example": {
                "org_id": "org_abc123",
                "scenario_ids": ["scn_iea_nze", "scn_ngfs_cp"],
                "base_revenue_usd": 500000000.0,
                "base_emissions_tco2e": 50000.0,
                "base_assets_usd": 1200000000.0,
            }
        }


class RunSensitivityRequest(BaseModel):
    """Request for sensitivity analysis."""
    org_id: str = Field(..., description="Organization ID")
    base_scenario_id: str = Field(..., description="Base scenario ID")
    parameter: str = Field(..., description="Parameter to vary")
    min_value: float = Field(..., description="Minimum value")
    max_value: float = Field(..., description="Maximum value")
    steps: int = Field(10, ge=3, le=50, description="Number of steps")
    base_revenue_usd: float = Field(..., gt=0)
    base_emissions_tco2e: float = Field(..., ge=0)


class RunWeightedImpactRequest(BaseModel):
    """Request for probability-weighted impact calculation."""
    org_id: str = Field(..., description="Organization ID")
    scenarios: List[Dict[str, Any]] = Field(..., description="List of {scenario_id, probability_pct}")
    base_revenue_usd: float = Field(..., gt=0)
    base_emissions_tco2e: float = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ScenarioResponse(BaseModel):
    """A scenario definition."""
    scenario_id: str
    scenario_name: str
    family: str
    scenario_type: str
    temperature_outcome: str
    description: str
    time_horizon_years: int
    carbon_price_2030_usd: Optional[float]
    carbon_price_2050_usd: Optional[float]
    renewable_share_2030_pct: Optional[float]
    renewable_share_2050_pct: Optional[float]
    gdp_impact_pct: Optional[float]
    physical_damage_pct: Optional[float]
    policy_stringency: Optional[str]
    technology_assumptions: Optional[str]
    is_prebuilt: bool
    parameters: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class AnalysisResultResponse(BaseModel):
    """Scenario analysis results."""
    analysis_id: str
    org_id: str
    scenarios_analyzed: int
    results: List[Dict[str, Any]]
    base_revenue_usd: float
    base_emissions_tco2e: float
    analysis_horizon_years: int
    generated_at: datetime


class ComparisonResponse(BaseModel):
    """Side-by-side scenario comparison."""
    scenarios: List[Dict[str, Any]]
    comparison_dimensions: List[str]
    comparison_matrix: Dict[str, Dict[str, Any]]
    key_differences: List[str]
    generated_at: datetime


class SensitivityResponse(BaseModel):
    """Sensitivity analysis result."""
    org_id: str
    base_scenario_id: str
    parameter: str
    steps: List[Dict[str, Any]]
    breakeven_value: Optional[float]
    max_impact_usd: float
    min_impact_usd: float
    elasticity: float
    generated_at: datetime


class ResilienceResponse(BaseModel):
    """Climate resilience assessment."""
    org_id: str
    overall_resilience_score: float
    resilience_grade: str
    scenario_results: List[Dict[str, Any]]
    strengths: List[str]
    vulnerabilities: List[str]
    adaptation_measures: List[Dict[str, Any]]
    strategy_viable_under_all_scenarios: bool
    generated_at: datetime


class WeightedImpactResponse(BaseModel):
    """Probability-weighted financial impact."""
    org_id: str
    expected_revenue_impact_usd: float
    expected_cost_impact_usd: float
    expected_asset_impact_usd: float
    expected_total_impact_usd: float
    scenario_contributions: List[Dict[str, Any]]
    confidence_interval_lower_usd: float
    confidence_interval_upper_usd: float
    generated_at: datetime


class NarrativeResponse(BaseModel):
    """Scenario narrative for disclosure."""
    scenario_id: str
    scenario_name: str
    narrative: str
    key_assumptions: List[str]
    word_count: int
    generated_at: datetime


# ---------------------------------------------------------------------------
# Pre-built Scenarios
# ---------------------------------------------------------------------------

PREBUILT_SCENARIOS = [
    {"scenario_id": "scn_iea_nze", "scenario_name": "IEA Net Zero Emissions by 2050", "family": "iea", "scenario_type": "orderly", "temperature_outcome": "1_5c", "description": "IEA NZE 2050 pathway requiring immediate and massive deployment of clean energy technologies.", "time_horizon_years": 30, "carbon_price_2030_usd": 130.0, "carbon_price_2050_usd": 250.0, "renewable_share_2030_pct": 60.0, "renewable_share_2050_pct": 90.0, "gdp_impact_pct": -0.5, "physical_damage_pct": 1.5, "policy_stringency": "very_high", "technology_assumptions": "Rapid CCS, green hydrogen, advanced nuclear", "is_prebuilt": True, "parameters": {"coal_phase_out": 2040, "ice_ban": 2035}},
    {"scenario_id": "scn_iea_aps", "scenario_name": "IEA Announced Pledges Scenario", "family": "iea", "scenario_type": "orderly", "temperature_outcome": "1_7c", "description": "Assumes all announced national net-zero targets met in full and on time.", "time_horizon_years": 30, "carbon_price_2030_usd": 90.0, "carbon_price_2050_usd": 160.0, "renewable_share_2030_pct": 50.0, "renewable_share_2050_pct": 75.0, "gdp_impact_pct": -0.3, "physical_damage_pct": 3.0, "policy_stringency": "high", "technology_assumptions": "Moderate CCS, growing hydrogen economy", "is_prebuilt": True, "parameters": {}},
    {"scenario_id": "scn_iea_steps", "scenario_name": "IEA Stated Policies Scenario", "family": "iea", "scenario_type": "hot_house", "temperature_outcome": "2_5c", "description": "Reflects existing policies as implemented. Insufficient for Paris Agreement.", "time_horizon_years": 30, "carbon_price_2030_usd": 30.0, "carbon_price_2050_usd": 55.0, "renewable_share_2030_pct": 35.0, "renewable_share_2050_pct": 50.0, "gdp_impact_pct": -1.5, "physical_damage_pct": 8.0, "policy_stringency": "low", "technology_assumptions": "Incremental improvements only", "is_prebuilt": True, "parameters": {}},
    {"scenario_id": "scn_ngfs_nz50", "scenario_name": "NGFS Net Zero 2050", "family": "ngfs", "scenario_type": "orderly", "temperature_outcome": "1_5c", "description": "Ambitious scenario limiting warming to 1.5C through stringent climate policies.", "time_horizon_years": 30, "carbon_price_2030_usd": 140.0, "carbon_price_2050_usd": 300.0, "renewable_share_2030_pct": 55.0, "renewable_share_2050_pct": 88.0, "gdp_impact_pct": -0.4, "physical_damage_pct": 1.5, "policy_stringency": "very_high", "technology_assumptions": "Breakthrough CCS, direct air capture at scale", "is_prebuilt": True, "parameters": {}},
    {"scenario_id": "scn_ngfs_b2d", "scenario_name": "NGFS Below 2 Degrees", "family": "ngfs", "scenario_type": "orderly", "temperature_outcome": "1_7c", "description": "Gradual strengthening of climate policies to limit warming below 2C.", "time_horizon_years": 30, "carbon_price_2030_usd": 80.0, "carbon_price_2050_usd": 180.0, "renewable_share_2030_pct": 45.0, "renewable_share_2050_pct": 72.0, "gdp_impact_pct": -0.2, "physical_damage_pct": 3.0, "policy_stringency": "high", "technology_assumptions": "Moderate clean energy ramp-up", "is_prebuilt": True, "parameters": {}},
    {"scenario_id": "scn_ngfs_dnz", "scenario_name": "NGFS Divergent Net Zero", "family": "ngfs", "scenario_type": "disorderly", "temperature_outcome": "1_5c", "description": "Net zero by 2050 via divergent policies across sectors with higher costs.", "time_horizon_years": 30, "carbon_price_2030_usd": 100.0, "carbon_price_2050_usd": 350.0, "renewable_share_2030_pct": 40.0, "renewable_share_2050_pct": 85.0, "gdp_impact_pct": -1.0, "physical_damage_pct": 1.5, "policy_stringency": "variable", "technology_assumptions": "Uneven adoption across regions", "is_prebuilt": True, "parameters": {}},
    {"scenario_id": "scn_ngfs_dt", "scenario_name": "NGFS Delayed Transition", "family": "ngfs", "scenario_type": "disorderly", "temperature_outcome": "1_7c", "description": "Policy action delayed until 2030 then aggressive catch-up creating shocks.", "time_horizon_years": 30, "carbon_price_2030_usd": 15.0, "carbon_price_2050_usd": 300.0, "renewable_share_2030_pct": 30.0, "renewable_share_2050_pct": 80.0, "gdp_impact_pct": -2.0, "physical_damage_pct": 4.0, "policy_stringency": "delayed_high", "technology_assumptions": "Technology catch-up after 2030", "is_prebuilt": True, "parameters": {}},
    {"scenario_id": "scn_ngfs_ndc", "scenario_name": "NGFS Nationally Determined Contributions", "family": "ngfs", "scenario_type": "hot_house", "temperature_outcome": "2_5c", "description": "Only current NDC commitments implemented. Insufficient for Paris targets.", "time_horizon_years": 30, "carbon_price_2030_usd": 25.0, "carbon_price_2050_usd": 50.0, "renewable_share_2030_pct": 32.0, "renewable_share_2050_pct": 48.0, "gdp_impact_pct": -1.8, "physical_damage_pct": 10.0, "policy_stringency": "low", "technology_assumptions": "Incremental progress", "is_prebuilt": True, "parameters": {}},
    {"scenario_id": "scn_ngfs_cp", "scenario_name": "NGFS Current Policies", "family": "ngfs", "scenario_type": "hot_house", "temperature_outcome": "above_3c", "description": "Only currently implemented policies. Warming exceeds 3C by 2100.", "time_horizon_years": 30, "carbon_price_2030_usd": 10.0, "carbon_price_2050_usd": 20.0, "renewable_share_2030_pct": 28.0, "renewable_share_2050_pct": 38.0, "gdp_impact_pct": -3.0, "physical_damage_pct": 18.0, "policy_stringency": "minimal", "technology_assumptions": "No material breakthroughs assumed", "is_prebuilt": True, "parameters": {}},
]


# ---------------------------------------------------------------------------
# In-Memory Store & Helpers
# ---------------------------------------------------------------------------

_custom_scenarios: Dict[str, Dict[str, Any]] = {}
_analysis_results: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


def _get_scenario(scenario_id: str) -> Optional[Dict[str, Any]]:
    """Look up a scenario by ID (prebuilt or custom)."""
    for s in PREBUILT_SCENARIOS:
        if s["scenario_id"] == scenario_id:
            return s
    return _custom_scenarios.get(scenario_id)


def _simulate_impact(scn: Dict[str, Any], base_revenue: float, base_emissions: float, base_assets: float = 0.0) -> Dict[str, Any]:
    """Simulate financial impact of a scenario."""
    cp = scn.get("carbon_price_2030_usd") or 0.0
    carbon_cost = round(base_emissions * cp, 2)
    rev_pct = (scn.get("gdp_impact_pct") or 0.0) * 2.0
    rev_impact = round(base_revenue * rev_pct / 100.0, 2)
    phys_damage = round(base_assets * (scn.get("physical_damage_pct") or 0.0) / 100.0, 2)
    total = round(rev_impact - carbon_cost - phys_damage, 2)
    return {"scenario_id": scn["scenario_id"], "scenario_name": scn["scenario_name"], "carbon_cost_usd": carbon_cost, "revenue_impact_usd": rev_impact, "physical_damage_usd": phys_damage, "total_impact_usd": total, "carbon_price_used": cp}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/prebuilt",
    response_model=List[ScenarioResponse],
    summary="List pre-built scenarios",
    description="Retrieve all pre-built IEA and NGFS climate scenarios with parameters.",
)
async def list_prebuilt(
    family: Optional[str] = Query(None, description="Filter by family: iea, ngfs"),
    scenario_type: Optional[str] = Query(None, description="Filter by type: orderly, disorderly, hot_house"),
) -> List[ScenarioResponse]:
    """List pre-built scenarios."""
    results = list(PREBUILT_SCENARIOS)
    if family:
        results = [s for s in results if s["family"] == family]
    if scenario_type:
        results = [s for s in results if s["scenario_type"] == scenario_type]
    now = _now()
    return [ScenarioResponse(**s, created_at=datetime(2025, 1, 1), updated_at=datetime(2025, 1, 1)) for s in results]


@router.post(
    "",
    response_model=ScenarioResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create custom scenario",
    description="Create a custom climate scenario with user-defined parameters.",
)
async def create_scenario(request: CreateScenarioRequest) -> ScenarioResponse:
    """Create a custom scenario."""
    sid = _generate_id("scn")
    now = _now()
    scn = {"scenario_id": sid, "scenario_name": request.scenario_name, "family": ScenarioFamily.CUSTOM.value, "scenario_type": request.scenario_type.value, "temperature_outcome": request.temperature_outcome.value, "description": request.description, "time_horizon_years": request.time_horizon_years, "carbon_price_2030_usd": request.carbon_price_2030_usd, "carbon_price_2050_usd": request.carbon_price_2050_usd, "renewable_share_2030_pct": request.renewable_share_2030_pct, "renewable_share_2050_pct": request.renewable_share_2050_pct, "gdp_impact_pct": request.gdp_impact_pct, "physical_damage_pct": request.physical_damage_pct, "policy_stringency": request.policy_stringency, "technology_assumptions": request.technology_assumptions, "is_prebuilt": False, "parameters": request.parameters, "created_at": now, "updated_at": now}
    _custom_scenarios[sid] = scn
    return ScenarioResponse(**scn)


@router.get(
    "/{scenario_id}",
    response_model=ScenarioResponse,
    summary="Get scenario with parameters",
    description="Retrieve a scenario definition by ID (pre-built or custom).",
)
async def get_scenario(scenario_id: str) -> ScenarioResponse:
    """Retrieve a scenario by ID."""
    scn = _get_scenario(scenario_id)
    if not scn:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Scenario {scenario_id} not found")
    if "created_at" not in scn:
        scn = {**scn, "created_at": datetime(2025, 1, 1), "updated_at": datetime(2025, 1, 1)}
    return ScenarioResponse(**scn)


@router.put(
    "/{scenario_id}",
    response_model=ScenarioResponse,
    summary="Update custom scenario",
    description="Update a custom scenario. Pre-built scenarios cannot be modified.",
)
async def update_scenario(scenario_id: str, request: UpdateScenarioRequest) -> ScenarioResponse:
    """Update a custom scenario."""
    scn = _custom_scenarios.get(scenario_id)
    if not scn:
        for s in PREBUILT_SCENARIOS:
            if s["scenario_id"] == scenario_id:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Pre-built scenarios cannot be modified")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Custom scenario {scenario_id} not found")
    updates = request.model_dump(exclude_unset=True)
    scn.update(updates)
    scn["updated_at"] = _now()
    return ScenarioResponse(**scn)


@router.delete(
    "/{scenario_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete custom scenario",
    description="Delete a custom scenario. Pre-built scenarios cannot be deleted.",
)
async def delete_scenario(scenario_id: str) -> None:
    """Delete a custom scenario."""
    for s in PREBUILT_SCENARIOS:
        if s["scenario_id"] == scenario_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Pre-built scenarios cannot be deleted")
    if scenario_id not in _custom_scenarios:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Custom scenario {scenario_id} not found")
    del _custom_scenarios[scenario_id]
    return None


@router.post(
    "/analyze",
    response_model=AnalysisResultResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run scenario analysis",
    description="Run scenario analysis for an organization against selected climate scenarios.",
)
async def run_analysis(request: RunAnalysisRequest) -> AnalysisResultResponse:
    """Run scenario analysis."""
    results = []
    for sid in request.scenario_ids:
        scn = _get_scenario(sid)
        if not scn:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Scenario {sid} not found")
        results.append(_simulate_impact(scn, request.base_revenue_usd, request.base_emissions_tco2e, request.base_assets_usd))
    aid = _generate_id("sca")
    analysis = {"analysis_id": aid, "org_id": request.org_id, "scenarios_analyzed": len(results), "results": results, "base_revenue_usd": request.base_revenue_usd, "base_emissions_tco2e": request.base_emissions_tco2e, "analysis_horizon_years": request.analysis_horizon_years, "generated_at": _now()}
    _analysis_results[aid] = analysis
    return AnalysisResultResponse(**analysis)


@router.get(
    "/results/{org_id}",
    response_model=List[AnalysisResultResponse],
    summary="Get scenario analysis results",
    description="Retrieve all scenario analysis results for an organization.",
)
async def get_results(org_id: str, limit: int = Query(20, ge=1, le=100)) -> List[AnalysisResultResponse]:
    """Retrieve scenario analysis results."""
    results = [a for a in _analysis_results.values() if a["org_id"] == org_id]
    results.sort(key=lambda a: a["generated_at"], reverse=True)
    return [AnalysisResultResponse(**a) for a in results[:limit]]


@router.get(
    "/compare",
    response_model=ComparisonResponse,
    summary="Compare scenarios side-by-side",
    description="Compare two or more scenarios across key dimensions.",
)
async def compare_scenarios(
    scenario_ids: str = Query(..., description="Comma-separated scenario IDs"),
) -> ComparisonResponse:
    """Compare scenarios side-by-side."""
    ids = [s.strip() for s in scenario_ids.split(",")]
    if len(ids) < 2:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least two scenario IDs required")

    scenarios = []
    for sid in ids:
        scn = _get_scenario(sid)
        if not scn:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Scenario {sid} not found")
        scenarios.append({"scenario_id": scn["scenario_id"], "scenario_name": scn["scenario_name"], "family": scn.get("family", "custom"), "temperature_outcome": scn["temperature_outcome"]})

    dimensions = ["temperature_outcome", "carbon_price_2030_usd", "carbon_price_2050_usd", "renewable_share_2030_pct", "renewable_share_2050_pct", "physical_damage_pct", "gdp_impact_pct", "policy_stringency"]
    matrix: Dict[str, Dict[str, Any]] = {}
    for dim in dimensions:
        matrix[dim] = {}
        for sid in ids:
            scn = _get_scenario(sid)
            matrix[dim][sid] = scn.get(dim) if scn else None

    differences = []
    if len(ids) >= 2:
        s1, s2 = _get_scenario(ids[0]) or {}, _get_scenario(ids[1]) or {}
        cp1, cp2 = s1.get("carbon_price_2050_usd") or 0, s2.get("carbon_price_2050_usd") or 0
        if abs(cp1 - cp2) > 50:
            differences.append(f"Carbon price diverges: ${cp1} vs ${cp2} by 2050")
        pd1, pd2 = s1.get("physical_damage_pct") or 0, s2.get("physical_damage_pct") or 0
        if abs(pd1 - pd2) > 3:
            differences.append(f"Physical damage differs by {abs(pd1 - pd2):.1f}% of GDP")
        differences.append(f"Temperature outcomes: {', '.join((_get_scenario(sid) or {}).get('temperature_outcome', '?') for sid in ids)}")

    return ComparisonResponse(scenarios=scenarios, comparison_dimensions=dimensions, comparison_matrix=matrix, key_differences=differences, generated_at=_now())


@router.post(
    "/sensitivity",
    response_model=SensitivityResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run sensitivity analysis",
    description="Vary a single scenario parameter and measure financial impact.",
)
async def run_sensitivity(request: RunSensitivityRequest) -> SensitivityResponse:
    """Run sensitivity analysis."""
    scn = _get_scenario(request.base_scenario_id)
    if not scn:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Scenario {request.base_scenario_id} not found")

    step_size = (request.max_value - request.min_value) / max(request.steps - 1, 1)
    steps = []
    breakeven = None
    prev = None
    for i in range(request.steps):
        value = round(request.min_value + i * step_size, 2)
        modified = dict(scn)
        if request.parameter == "carbon_price":
            modified["carbon_price_2030_usd"] = value
        elif request.parameter == "physical_damage":
            modified["physical_damage_pct"] = value
        else:
            modified[request.parameter] = value
        impact = _simulate_impact(modified, request.base_revenue_usd, request.base_emissions_tco2e)
        total = impact["total_impact_usd"]
        steps.append({"parameter_value": value, "total_impact_usd": total, "carbon_cost_usd": impact["carbon_cost_usd"]})
        if prev is not None and prev * total < 0 and breakeven is None:
            breakeven = round(value - step_size / 2, 2)
        prev = total

    impacts = [s["total_impact_usd"] for s in steps]
    return SensitivityResponse(
        org_id=request.org_id, base_scenario_id=request.base_scenario_id, parameter=request.parameter,
        steps=steps, breakeven_value=breakeven, max_impact_usd=max(impacts), min_impact_usd=min(impacts),
        elasticity=round((max(impacts) - min(impacts)) / max(request.max_value - request.min_value, 0.01), 2),
        generated_at=_now(),
    )


@router.get(
    "/resilience/{org_id}",
    response_model=ResilienceResponse,
    summary="Get climate resilience assessment",
    description="Assess strategy resilience across all analyzed scenarios (TCFD Disclosure c).",
)
async def get_resilience(org_id: str) -> ResilienceResponse:
    """Assess climate resilience."""
    org_analyses = [a for a in _analysis_results.values() if a["org_id"] == org_id]
    scenario_results = []
    all_viable = True
    if org_analyses:
        latest = max(org_analyses, key=lambda a: a["generated_at"])
        for r in latest.get("results", []):
            viable = r.get("total_impact_usd", 0) > -latest.get("base_revenue_usd", 1) * 0.2
            scenario_results.append({"scenario_id": r.get("scenario_id"), "scenario_name": r.get("scenario_name"), "total_impact_usd": r.get("total_impact_usd"), "viable": viable})
            if not viable:
                all_viable = False

    score = 80.0 if all_viable and scenario_results else 50.0 if not scenario_results else 65.0
    grade = "A" if score >= 80 else "B" if score >= 65 else "C" if score >= 50 else "D"
    return ResilienceResponse(
        org_id=org_id, overall_resilience_score=score, resilience_grade=grade, scenario_results=scenario_results,
        strengths=["Diversified revenue streams", "Active emissions reduction program", "Climate governance framework"],
        vulnerabilities=["Coastal facility exposure", "Supply chain concentration in vulnerable regions", "Limited low-carbon portfolio"],
        adaptation_measures=[
            {"measure": "Facility hardening", "investment_usd": 15000000, "risk_reduction_pct": 35.0},
            {"measure": "Supply chain diversification", "investment_usd": 8000000, "risk_reduction_pct": 25.0},
            {"measure": "Low-carbon R&D acceleration", "investment_usd": 20000000, "risk_reduction_pct": 20.0},
        ],
        strategy_viable_under_all_scenarios=all_viable, generated_at=_now(),
    )


@router.post(
    "/weighted-impact",
    response_model=WeightedImpactResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Calculate probability-weighted financial impact",
    description="Calculate expected financial impact weighted by scenario probabilities.",
)
async def weighted_impact(request: RunWeightedImpactRequest) -> WeightedImpactResponse:
    """Calculate probability-weighted financial impact."""
    total_prob = sum(s.get("probability_pct", 0) for s in request.scenarios)
    if abs(total_prob - 100.0) > 1.0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Probabilities must sum to 100%. Current: {total_prob}%")

    contributions = []
    w_rev, w_cost, w_total = 0.0, 0.0, 0.0
    all_totals = []
    for entry in request.scenarios:
        sid = entry.get("scenario_id")
        prob = entry.get("probability_pct", 0) / 100.0
        scn = _get_scenario(sid)
        if not scn:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Scenario {sid} not found")
        impact = _simulate_impact(scn, request.base_revenue_usd, request.base_emissions_tco2e)
        wr = round(impact["revenue_impact_usd"] * prob, 2)
        wc = round(impact["carbon_cost_usd"] * prob, 2)
        wt = round(impact["total_impact_usd"] * prob, 2)
        w_rev += wr
        w_cost += wc
        w_total += wt
        all_totals.append(impact["total_impact_usd"])
        contributions.append({"scenario_id": sid, "scenario_name": scn["scenario_name"], "probability_pct": entry.get("probability_pct"), "unweighted_impact_usd": impact["total_impact_usd"], "weighted_impact_usd": wt})

    spread = max(all_totals) - min(all_totals) if all_totals else 0
    return WeightedImpactResponse(
        org_id=request.org_id, expected_revenue_impact_usd=round(w_rev, 2), expected_cost_impact_usd=round(w_cost, 2),
        expected_asset_impact_usd=0.0, expected_total_impact_usd=round(w_total, 2), scenario_contributions=contributions,
        confidence_interval_lower_usd=round(w_total - spread * 0.3, 2), confidence_interval_upper_usd=round(w_total + spread * 0.3, 2),
        generated_at=_now(),
    )


@router.get(
    "/{scenario_id}/narrative",
    response_model=NarrativeResponse,
    summary="Get scenario narrative",
    description="Generate a disclosure-ready narrative for a scenario.",
)
async def get_narrative(scenario_id: str) -> NarrativeResponse:
    """Generate scenario narrative for disclosure."""
    scn = _get_scenario(scenario_id)
    if not scn:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Scenario {scenario_id} not found")

    name = scn["scenario_name"]
    temp = scn["temperature_outcome"].replace("_", ".").replace("c", "C")
    cp30 = scn.get("carbon_price_2030_usd") or 0
    cp50 = scn.get("carbon_price_2050_usd") or 0
    ren30 = scn.get("renewable_share_2030_pct") or 0
    ren50 = scn.get("renewable_share_2050_pct") or 0
    phys = scn.get("physical_damage_pct") or 0
    desc = scn.get("description", "")

    narrative = (
        f"Under the {name} scenario (temperature outcome: {temp}), "
        f"{desc.lower() if desc else 'the following conditions are assumed'}. "
        f"Carbon prices are projected to reach ${cp30:.0f}/tCO2 by 2030 and "
        f"${cp50:.0f}/tCO2 by 2050. Renewable energy constitutes "
        f"{ren30:.0f}% of the mix by 2030, rising to {ren50:.0f}% by 2050. "
        f"Cumulative physical climate damages are estimated at {phys:.1f}% of GDP."
    )
    assumptions = [f"Temperature: {temp}", f"Carbon price: ${cp30:.0f} (2030) to ${cp50:.0f} (2050)", f"Renewables: {ren30:.0f}% (2030) to {ren50:.0f}% (2050)", f"Physical damage: {phys:.1f}% GDP", f"Policy: {scn.get('policy_stringency', 'moderate')}"]
    if scn.get("technology_assumptions"):
        assumptions.append(f"Technology: {scn['technology_assumptions']}")

    return NarrativeResponse(scenario_id=scenario_id, scenario_name=name, narrative=narrative, key_assumptions=assumptions, word_count=len(narrative.split()), generated_at=_now())
