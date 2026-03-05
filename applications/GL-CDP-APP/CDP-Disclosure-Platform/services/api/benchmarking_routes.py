"""
GL-CDP-APP Benchmarking API

Sector-level and regional benchmarking for CDP Climate Change scores.
Provides peer comparison, score distribution histograms, category-level
comparisons, A-list rates, and custom peer group configuration.

Benchmarking dimensions:
    - Sector: GICS sector classification (11 sectors)
    - Regional: Geography-based peer groups
    - Custom: User-defined comparator sets

All benchmark data is anonymized -- no company-specific data exposed.
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/cdp/benchmarks", tags=["Benchmarking"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BenchmarkRegion(str, Enum):
    """Regional benchmark groups."""
    GLOBAL = "global"
    EUROPE = "europe"
    NORTH_AMERICA = "north_america"
    ASIA_PACIFIC = "asia_pacific"
    LATIN_AMERICA = "latin_america"
    MIDDLE_EAST_AFRICA = "middle_east_africa"


class BenchmarkSector(str, Enum):
    """GICS sector classification for benchmarking."""
    ENERGY = "energy"
    MATERIALS = "materials"
    INDUSTRIALS = "industrials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    HEALTH_CARE = "health_care"
    FINANCIALS = "financials"
    INFORMATION_TECHNOLOGY = "information_technology"
    COMMUNICATION_SERVICES = "communication_services"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class SectorInfo(BaseModel):
    """Sector information for benchmarking."""
    sector_code: str
    sector_name: str
    respondent_count: int
    avg_score: float
    avg_band: str
    a_list_count: int
    a_list_pct: float


class SectorBenchmarkResponse(BaseModel):
    """Sector-level benchmark comparison."""
    questionnaire_id: str
    sector: str
    sector_name: str
    org_score: float
    org_band: str
    sector_avg_score: float
    sector_avg_band: str
    sector_median_score: float
    sector_top_quartile: float
    sector_bottom_quartile: float
    percentile_rank: float
    respondent_count: int
    a_list_count: int
    a_list_pct: float
    outperforming_pct: float
    category_comparisons: List[Dict[str, Any]]
    benchmarked_at: datetime


class RegionalBenchmarkResponse(BaseModel):
    """Regional benchmark comparison."""
    questionnaire_id: str
    region: str
    region_name: str
    org_score: float
    org_band: str
    region_avg_score: float
    region_avg_band: str
    region_median_score: float
    percentile_rank: float
    respondent_count: int
    a_list_count: int
    outperforming_pct: float
    top_regions: List[Dict[str, Any]]
    benchmarked_at: datetime


class ScoreDistributionResponse(BaseModel):
    """Score distribution histogram for a peer group."""
    questionnaire_id: str
    peer_group: str
    respondent_count: int
    org_score: float
    org_band: str
    distribution: List[Dict[str, Any]]
    band_distribution: Dict[str, int]
    benchmarked_at: datetime


class CategoryComparisonResponse(BaseModel):
    """Category-by-category comparison against sector."""
    questionnaire_id: str
    sector: str
    categories: List[Dict[str, Any]]
    outperforming_categories: List[str]
    underperforming_categories: List[str]
    aligned_categories: List[str]
    benchmarked_at: datetime


class CustomPeerResponse(BaseModel):
    """Custom peer group benchmark result."""
    questionnaire_id: str
    peer_group_id: str
    peer_group_name: str
    peer_count: int
    org_score: float
    org_band: str
    peer_avg_score: float
    peer_avg_band: str
    peer_median_score: float
    percentile_rank: float
    category_comparisons: List[Dict[str, Any]]
    created_at: datetime


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class CustomPeerRequest(BaseModel):
    """Request to create a custom peer group benchmark."""
    peer_group_name: str = Field(
        ..., min_length=1, max_length=200, description="Custom peer group name"
    )
    sectors: Optional[List[str]] = Field(None, description="Filter peers by sectors")
    regions: Optional[List[str]] = Field(None, description="Filter peers by regions")
    min_score: Optional[float] = Field(None, ge=0, le=100, description="Minimum peer score")
    max_score: Optional[float] = Field(None, ge=0, le=100, description="Maximum peer score")
    revenue_range: Optional[Dict[str, float]] = Field(
        None, description="Revenue range filter (min/max in USD millions)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "peer_group_name": "European Materials Peers",
                "sectors": ["materials"],
                "regions": ["europe"],
                "min_score": 40.0,
                "max_score": None,
                "revenue_range": {"min": 1000.0, "max": 50000.0},
            }
        }


# ---------------------------------------------------------------------------
# Simulated Benchmark Data
# ---------------------------------------------------------------------------

SECTOR_BENCHMARKS = {
    "energy": {"name": "Energy", "count": 320, "avg": 48.5, "median": 45.0, "q1": 32.0, "q3": 62.0, "a_list": 18, "a_pct": 5.6},
    "materials": {"name": "Materials", "count": 285, "avg": 52.3, "median": 50.0, "q1": 38.0, "q3": 65.0, "a_list": 22, "a_pct": 7.7},
    "industrials": {"name": "Industrials", "count": 410, "avg": 50.8, "median": 48.0, "q1": 35.0, "q3": 63.0, "a_list": 30, "a_pct": 7.3},
    "consumer_discretionary": {"name": "Consumer Discretionary", "count": 350, "avg": 46.2, "median": 43.0, "q1": 30.0, "q3": 58.0, "a_list": 15, "a_pct": 4.3},
    "consumer_staples": {"name": "Consumer Staples", "count": 180, "avg": 55.8, "median": 54.0, "q1": 42.0, "q3": 68.0, "a_list": 20, "a_pct": 11.1},
    "health_care": {"name": "Health Care", "count": 150, "avg": 47.5, "median": 44.0, "q1": 32.0, "q3": 60.0, "a_list": 8, "a_pct": 5.3},
    "financials": {"name": "Financials", "count": 420, "avg": 54.2, "median": 52.0, "q1": 40.0, "q3": 66.0, "a_list": 40, "a_pct": 9.5},
    "information_technology": {"name": "Information Technology", "count": 280, "avg": 51.0, "median": 48.0, "q1": 36.0, "q3": 64.0, "a_list": 25, "a_pct": 8.9},
    "communication_services": {"name": "Communication Services", "count": 120, "avg": 49.5, "median": 46.0, "q1": 34.0, "q3": 62.0, "a_list": 8, "a_pct": 6.7},
    "utilities": {"name": "Utilities", "count": 200, "avg": 58.2, "median": 56.0, "q1": 44.0, "q3": 70.0, "a_list": 28, "a_pct": 14.0},
    "real_estate": {"name": "Real Estate", "count": 230, "avg": 53.5, "median": 51.0, "q1": 38.0, "q3": 66.0, "a_list": 22, "a_pct": 9.6},
}

SCORING_CATEGORIES = [
    {"id": 1, "name": "Governance"}, {"id": 2, "name": "Risk management processes"},
    {"id": 3, "name": "Risk disclosure"}, {"id": 4, "name": "Opportunity disclosure"},
    {"id": 5, "name": "Business strategy"}, {"id": 6, "name": "Scenario analysis"},
    {"id": 7, "name": "Targets"}, {"id": 8, "name": "Emissions reduction initiatives"},
    {"id": 9, "name": "Scope 1 & 2 emissions"}, {"id": 10, "name": "Scope 3 emissions"},
    {"id": 11, "name": "Energy"}, {"id": 12, "name": "Carbon pricing"},
    {"id": 13, "name": "Value chain engagement"}, {"id": 14, "name": "Public policy engagement"},
    {"id": 15, "name": "Transition plan"}, {"id": 16, "name": "Portfolio climate performance"},
    {"id": 17, "name": "Financial impact assessment"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


def _score_to_band(score_pct: float) -> str:
    if score_pct >= 80:
        return "A"
    elif score_pct >= 70:
        return "A-"
    elif score_pct >= 60:
        return "B"
    elif score_pct >= 50:
        return "B-"
    elif score_pct >= 40:
        return "C"
    elif score_pct >= 30:
        return "C-"
    elif score_pct >= 20:
        return "D"
    return "D-"


def _generate_category_comparisons(org_score_base: float, sector_avg_base: float) -> List[Dict[str, Any]]:
    """Generate per-category comparisons."""
    comparisons = []
    for cat in SCORING_CATEGORIES:
        org_score = round(org_score_base + (cat["id"] % 5 - 2) * 5, 1)
        sector_avg = round(sector_avg_base + (cat["id"] % 4 - 1.5) * 3, 1)
        delta = round(org_score - sector_avg, 1)
        comparisons.append({
            "category_id": cat["id"],
            "category_name": cat["name"],
            "org_score": max(0, min(100, org_score)),
            "sector_avg": max(0, min(100, sector_avg)),
            "delta": delta,
            "outperforming": delta > 0,
        })
    return comparisons


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/sectors",
    response_model=List[SectorInfo],
    summary="List available sectors",
    description="Retrieve all available GICS sectors with aggregate benchmark statistics.",
)
async def list_sectors() -> List[SectorInfo]:
    """List all sectors with benchmark statistics."""
    sectors = []
    for code, data in SECTOR_BENCHMARKS.items():
        sectors.append(SectorInfo(
            sector_code=code,
            sector_name=data["name"],
            respondent_count=data["count"],
            avg_score=data["avg"],
            avg_band=_score_to_band(data["avg"]),
            a_list_count=data["a_list"],
            a_list_pct=data["a_pct"],
        ))
    sectors.sort(key=lambda s: s.avg_score, reverse=True)
    return sectors


@router.get(
    "/{questionnaire_id}/sector",
    response_model=SectorBenchmarkResponse,
    summary="Sector benchmark",
    description=(
        "Compare the organization's CDP score against sector peers. "
        "Includes percentile rank, quartile positions, A-list rate, "
        "and category-by-category comparison."
    ),
)
async def get_sector_benchmark(
    questionnaire_id: str,
    sector: Optional[str] = Query(None, description="Override sector for comparison"),
) -> SectorBenchmarkResponse:
    """Get sector benchmark comparison."""
    sector_code = sector or "materials"
    benchmark = SECTOR_BENCHMARKS.get(sector_code)
    if not benchmark:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sector '{sector_code}' not found. Use GET /sectors for valid options.",
        )

    org_score = 58.7
    org_band = _score_to_band(org_score)
    outperforming = round(
        (org_score - benchmark["q1"]) / (benchmark["q3"] - benchmark["q1"]) * 50 + 25,
        1
    ) if benchmark["q3"] > benchmark["q1"] else 50.0
    percentile = max(0, min(100, outperforming))

    comparisons = _generate_category_comparisons(org_score, benchmark["avg"])

    return SectorBenchmarkResponse(
        questionnaire_id=questionnaire_id,
        sector=sector_code,
        sector_name=benchmark["name"],
        org_score=org_score,
        org_band=org_band,
        sector_avg_score=benchmark["avg"],
        sector_avg_band=_score_to_band(benchmark["avg"]),
        sector_median_score=benchmark["median"],
        sector_top_quartile=benchmark["q3"],
        sector_bottom_quartile=benchmark["q1"],
        percentile_rank=percentile,
        respondent_count=benchmark["count"],
        a_list_count=benchmark["a_list"],
        a_list_pct=benchmark["a_pct"],
        outperforming_pct=percentile,
        category_comparisons=comparisons,
        benchmarked_at=_now(),
    )


@router.get(
    "/{questionnaire_id}/regional",
    response_model=RegionalBenchmarkResponse,
    summary="Regional benchmark",
    description=(
        "Compare the organization's CDP score against regional peers. "
        "Includes top-performing regions and regional A-list rates."
    ),
)
async def get_regional_benchmark(
    questionnaire_id: str,
    region: BenchmarkRegion = Query(BenchmarkRegion.GLOBAL, description="Region for comparison"),
) -> RegionalBenchmarkResponse:
    """Get regional benchmark comparison."""
    regional_data = {
        "global": {"name": "Global", "count": 3200, "avg": 50.5, "median": 48.0, "a_list": 250},
        "europe": {"name": "Europe", "count": 1100, "avg": 55.2, "median": 53.0, "a_list": 120},
        "north_america": {"name": "North America", "count": 800, "avg": 52.0, "median": 50.0, "a_list": 65},
        "asia_pacific": {"name": "Asia Pacific", "count": 900, "avg": 48.5, "median": 45.0, "a_list": 50},
        "latin_america": {"name": "Latin America", "count": 250, "avg": 44.0, "median": 42.0, "a_list": 10},
        "middle_east_africa": {"name": "Middle East & Africa", "count": 150, "avg": 40.5, "median": 38.0, "a_list": 5},
    }

    data = regional_data.get(region.value, regional_data["global"])
    org_score = 58.7
    percentile = round(min(100, max(0, (org_score - 20) / 80 * 100)), 1)

    top_regions = sorted(
        [{"region": k, "name": v["name"], "avg_score": v["avg"], "a_list_count": v["a_list"]}
         for k, v in regional_data.items() if k != "global"],
        key=lambda r: r["avg_score"],
        reverse=True,
    )

    return RegionalBenchmarkResponse(
        questionnaire_id=questionnaire_id,
        region=region.value,
        region_name=data["name"],
        org_score=org_score,
        org_band=_score_to_band(org_score),
        region_avg_score=data["avg"],
        region_avg_band=_score_to_band(data["avg"]),
        region_median_score=data["median"],
        percentile_rank=percentile,
        respondent_count=data["count"],
        a_list_count=data["a_list"],
        outperforming_pct=percentile,
        top_regions=top_regions,
        benchmarked_at=_now(),
    )


@router.get(
    "/{questionnaire_id}/distribution",
    response_model=ScoreDistributionResponse,
    summary="Score distribution",
    description=(
        "Retrieve the score distribution histogram for the selected peer group. "
        "Shows organization position within the distribution and band breakdown."
    ),
)
async def get_distribution(
    questionnaire_id: str,
    sector: Optional[str] = Query(None, description="Sector filter"),
    region: Optional[str] = Query(None, description="Region filter"),
) -> ScoreDistributionResponse:
    """Get score distribution for peer group."""
    peer_group = sector or "global"
    org_score = 58.7

    distribution = [
        {"range": "0-10", "count": 45, "pct": 1.4},
        {"range": "10-20", "count": 190, "pct": 5.9},
        {"range": "20-30", "count": 380, "pct": 11.9},
        {"range": "30-40", "count": 520, "pct": 16.3},
        {"range": "40-50", "count": 610, "pct": 19.1},
        {"range": "50-60", "count": 580, "pct": 18.1},
        {"range": "60-70", "count": 420, "pct": 13.1},
        {"range": "70-80", "count": 280, "pct": 8.8},
        {"range": "80-90", "count": 140, "pct": 4.4},
        {"range": "90-100", "count": 35, "pct": 1.1},
    ]

    band_distribution = {
        "A": 35, "A-": 140, "B": 420, "B-": 580,
        "C": 610, "C-": 520, "D": 380, "D-": 235,
    }

    return ScoreDistributionResponse(
        questionnaire_id=questionnaire_id,
        peer_group=peer_group,
        respondent_count=3200,
        org_score=org_score,
        org_band=_score_to_band(org_score),
        distribution=distribution,
        band_distribution=band_distribution,
        benchmarked_at=_now(),
    )


@router.get(
    "/{questionnaire_id}/categories",
    response_model=CategoryComparisonResponse,
    summary="Category comparison",
    description=(
        "Compare the organization's score across all 17 CDP categories "
        "against the sector average. Identifies outperforming, underperforming, "
        "and aligned categories."
    ),
)
async def get_category_comparison(
    questionnaire_id: str,
    sector: Optional[str] = Query(None, description="Sector for comparison"),
) -> CategoryComparisonResponse:
    """Get category-by-category comparison."""
    sector_code = sector or "materials"
    benchmark = SECTOR_BENCHMARKS.get(sector_code, SECTOR_BENCHMARKS["materials"])

    org_score = 58.7
    comparisons = _generate_category_comparisons(org_score, benchmark["avg"])

    outperforming = [c["category_name"] for c in comparisons if c["delta"] > 3]
    underperforming = [c["category_name"] for c in comparisons if c["delta"] < -3]
    aligned = [c["category_name"] for c in comparisons if -3 <= c["delta"] <= 3]

    return CategoryComparisonResponse(
        questionnaire_id=questionnaire_id,
        sector=sector_code,
        categories=comparisons,
        outperforming_categories=outperforming,
        underperforming_categories=underperforming,
        aligned_categories=aligned,
        benchmarked_at=_now(),
    )


@router.post(
    "/{questionnaire_id}/custom-peers",
    response_model=CustomPeerResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Custom peer group benchmark",
    description=(
        "Create a custom peer group based on sector, region, score range, "
        "and revenue filters. Compares the organization against this "
        "custom-defined peer set."
    ),
)
async def create_custom_peer_benchmark(
    questionnaire_id: str,
    request: CustomPeerRequest,
) -> CustomPeerResponse:
    """Create custom peer group benchmark."""
    peer_group_id = _generate_id("pg")
    org_score = 58.7
    now = _now()

    # Simulate peer group based on filters
    peer_count = 85
    if request.sectors:
        peer_count = min(peer_count, sum(
            SECTOR_BENCHMARKS.get(s, {}).get("count", 0)
            for s in request.sectors
        ) // 4)
    if request.regions:
        peer_count = min(peer_count, peer_count * len(request.regions) // 5)
    peer_count = max(10, peer_count)

    peer_avg = 55.0
    if request.min_score:
        peer_avg = max(peer_avg, request.min_score + 10)

    comparisons = _generate_category_comparisons(org_score, peer_avg)

    return CustomPeerResponse(
        questionnaire_id=questionnaire_id,
        peer_group_id=peer_group_id,
        peer_group_name=request.peer_group_name,
        peer_count=peer_count,
        org_score=org_score,
        org_band=_score_to_band(org_score),
        peer_avg_score=peer_avg,
        peer_avg_band=_score_to_band(peer_avg),
        peer_median_score=round(peer_avg - 2.0, 1),
        percentile_rank=round(min(100, max(0, (org_score - 20) / 80 * 100)), 1),
        category_comparisons=comparisons,
        created_at=now,
    )
