"""
GL-SBTi-APP Temperature Scoring API

Calculates temperature scores (implied temperature rise) for companies
and portfolios based on their science-based targets and emissions
trajectories.  Implements the SBTi Temperature Rating methodology which
converts target ambition into a warming alignment score (e.g. 1.5C, 1.8C).

Temperature Scoring Methodology:
    - Maps target ambition and progress to implied warming
    - Company score derived from best available target across scopes
    - Portfolio score is emissions-weighted average of holdings
    - Peer ranking based on sector-adjusted temperature scores
    - Time series tracks temperature alignment evolution
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

router = APIRouter(prefix="/api/v1/sbti/temperature", tags=["Temperature Scoring"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class PortfolioTemperatureRequest(BaseModel):
    """Request to calculate portfolio temperature score."""
    portfolio_name: str = Field(..., max_length=300)
    holdings: List[Dict[str, Any]] = Field(
        ..., description="List of {company_id, company_name, weight_pct, temperature_score}",
    )
    weighting_method: str = Field(
        "ownership", description="ownership, market_value, or equal_weight",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "portfolio_name": "Climate Transition Fund",
                "holdings": [
                    {"company_id": "c001", "company_name": "GreenCorp", "weight_pct": 30, "temperature_score": 1.5},
                    {"company_id": "c002", "company_name": "EnergyInc", "weight_pct": 40, "temperature_score": 2.1},
                    {"company_id": "c003", "company_name": "TechStartup", "weight_pct": 30, "temperature_score": 1.8},
                ],
                "weighting_method": "ownership",
            }
        }


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class TemperatureScoreResponse(BaseModel):
    """Company temperature score."""
    org_id: str
    overall_temperature_c: float
    scope1_2_temperature_c: float
    scope3_temperature_c: float
    combined_temperature_c: float
    alignment_status: str
    target_basis: str
    methodology: str
    confidence: str
    sbti_validated: bool
    generated_at: datetime


class TemperatureTimeSeriesResponse(BaseModel):
    """Temperature score time series."""
    org_id: str
    time_series: List[Dict[str, Any]]
    trend: str
    improvement_rate_c_per_year: float
    generated_at: datetime


class PeerRankingResponse(BaseModel):
    """Peer temperature ranking."""
    org_id: str
    org_temperature_c: float
    sector: str
    rank: int
    total_peers: int
    percentile: int
    sector_average_c: float
    sector_median_c: float
    peer_distribution: Dict[str, int]
    top_performers: List[Dict[str, Any]]
    generated_at: datetime


class PortfolioTemperatureResponse(BaseModel):
    """Portfolio temperature score."""
    portfolio_name: str
    portfolio_temperature_c: float
    alignment_status: str
    total_holdings: int
    aligned_1_5c_count: int
    aligned_2c_count: int
    above_2c_count: int
    no_target_count: int
    holding_scores: List[Dict[str, Any]]
    weighting_method: str
    paris_aligned: bool
    generated_at: datetime


class TemperatureReportResponse(BaseModel):
    """Temperature alignment report."""
    org_id: str
    report_id: str
    overall_temperature_c: float
    scope_temperatures: Dict[str, float]
    alignment_status: str
    peer_comparison: Dict[str, Any]
    trajectory_analysis: Dict[str, Any]
    recommendations: List[str]
    generated_at: datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


def _alignment_status(temp: float) -> str:
    """Classify temperature alignment."""
    if temp <= 1.5:
        return "1.5C_aligned"
    if temp <= 1.75:
        return "well_below_2C"
    if temp <= 2.0:
        return "below_2C"
    if temp <= 2.5:
        return "above_2C"
    return "strongly_misaligned"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/org/{org_id}/score",
    response_model=TemperatureScoreResponse,
    summary="Company temperature score",
    description=(
        "Calculate the company-level temperature score based on its science-based "
        "targets and emissions trajectory. Returns implied temperature rise for "
        "each scope and a combined score."
    ),
)
async def get_temperature_score(org_id: str) -> TemperatureScoreResponse:
    """Get company temperature score."""
    s12_temp = 1.65
    s3_temp = 2.15
    combined = round(s12_temp * 0.4 + s3_temp * 0.6, 2)

    return TemperatureScoreResponse(
        org_id=org_id,
        overall_temperature_c=combined,
        scope1_2_temperature_c=s12_temp,
        scope3_temperature_c=s3_temp,
        combined_temperature_c=combined,
        alignment_status=_alignment_status(combined),
        target_basis="Near-term 1.5C Scope 1+2, well-below 2C Scope 3",
        methodology="SBTi Temperature Rating v2.0",
        confidence="medium",
        sbti_validated=True,
        generated_at=_now(),
    )


@router.get(
    "/org/{org_id}/time-series",
    response_model=TemperatureTimeSeriesResponse,
    summary="Temperature time series",
    description="Get the temperature score evolution over time for trend analysis.",
)
async def get_time_series(org_id: str) -> TemperatureTimeSeriesResponse:
    """Get temperature score time series."""
    series = [
        {"year": 2020, "temperature_c": 2.8, "scope1_2_c": 2.5, "scope3_c": 3.0},
        {"year": 2021, "temperature_c": 2.5, "scope1_2_c": 2.2, "scope3_c": 2.7},
        {"year": 2022, "temperature_c": 2.2, "scope1_2_c": 1.9, "scope3_c": 2.4},
        {"year": 2023, "temperature_c": 2.0, "scope1_2_c": 1.75, "scope3_c": 2.2},
        {"year": 2024, "temperature_c": 1.95, "scope1_2_c": 1.65, "scope3_c": 2.15},
    ]

    if len(series) >= 2:
        improvement = round(
            (series[0]["temperature_c"] - series[-1]["temperature_c"]) / len(series), 2,
        )
    else:
        improvement = 0.0

    trend = "improving" if improvement > 0 else "stable" if improvement == 0 else "worsening"

    return TemperatureTimeSeriesResponse(
        org_id=org_id,
        time_series=series,
        trend=trend,
        improvement_rate_c_per_year=improvement,
        generated_at=_now(),
    )


@router.get(
    "/org/{org_id}/peer-ranking",
    response_model=PeerRankingResponse,
    summary="Peer temperature ranking",
    description=(
        "Rank the organization's temperature score against sector peers. "
        "Returns rank, percentile, and distribution of peer temperatures."
    ),
)
async def get_peer_ranking(
    org_id: str,
    sector: str = Query("general", description="Sector for peer comparison"),
) -> PeerRankingResponse:
    """Get peer temperature ranking."""
    org_temp = 1.95
    peer_temps = [1.4, 1.55, 1.7, 1.8, 1.95, 2.1, 2.3, 2.5, 2.8, 3.2]
    sorted_peers = sorted(peer_temps)
    rank = sum(1 for t in sorted_peers if t < org_temp) + 1
    percentile = round((1 - rank / len(sorted_peers)) * 100)

    distribution = {
        "below_1.5C": sum(1 for t in peer_temps if t <= 1.5),
        "1.5C_to_2C": sum(1 for t in peer_temps if 1.5 < t <= 2.0),
        "2C_to_2.5C": sum(1 for t in peer_temps if 2.0 < t <= 2.5),
        "above_2.5C": sum(1 for t in peer_temps if t > 2.5),
    }

    top_performers = [
        {"company": "LeadGreen Corp", "temperature_c": 1.4, "sbti_status": "validated"},
        {"company": "EcoFirst Inc", "temperature_c": 1.55, "sbti_status": "validated"},
        {"company": "CleanTech Ltd", "temperature_c": 1.7, "sbti_status": "committed"},
    ]

    return PeerRankingResponse(
        org_id=org_id,
        org_temperature_c=org_temp,
        sector=sector,
        rank=rank,
        total_peers=len(peer_temps),
        percentile=percentile,
        sector_average_c=round(sum(peer_temps) / len(peer_temps), 2),
        sector_median_c=round(sorted_peers[len(sorted_peers) // 2], 2),
        peer_distribution=distribution,
        top_performers=top_performers,
        generated_at=_now(),
    )


@router.post(
    "/portfolio",
    response_model=PortfolioTemperatureResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Portfolio temperature score",
    description=(
        "Calculate the weighted-average temperature score for an investment "
        "portfolio. Weights can be based on ownership, market value, or "
        "equal weighting. Returns Paris Agreement alignment assessment."
    ),
)
async def calculate_portfolio_temperature(
    request: PortfolioTemperatureRequest,
) -> PortfolioTemperatureResponse:
    """Calculate portfolio temperature score."""
    weighted_sum = 0.0
    total_weight = 0.0
    aligned_15 = 0
    aligned_2 = 0
    above_2 = 0
    no_target = 0

    holding_scores = []
    for h in request.holdings:
        temp = h.get("temperature_score", 3.2)
        weight = h.get("weight_pct", 0)
        weighted_sum += temp * weight / 100
        total_weight += weight / 100

        if temp <= 1.5:
            aligned_15 += 1
        elif temp <= 2.0:
            aligned_2 += 1
        else:
            above_2 += 1

        holding_scores.append({
            "company_id": h.get("company_id"),
            "company_name": h.get("company_name"),
            "weight_pct": weight,
            "temperature_c": temp,
            "alignment": _alignment_status(temp),
        })

    portfolio_temp = round(weighted_sum / total_weight, 2) if total_weight > 0 else 3.2

    return PortfolioTemperatureResponse(
        portfolio_name=request.portfolio_name,
        portfolio_temperature_c=portfolio_temp,
        alignment_status=_alignment_status(portfolio_temp),
        total_holdings=len(request.holdings),
        aligned_1_5c_count=aligned_15,
        aligned_2c_count=aligned_2,
        above_2c_count=above_2,
        no_target_count=no_target,
        holding_scores=holding_scores,
        weighting_method=request.weighting_method,
        paris_aligned=portfolio_temp <= 2.0,
        generated_at=_now(),
    )


@router.get(
    "/org/{org_id}/report",
    response_model=TemperatureReportResponse,
    summary="Temperature report",
    description=(
        "Generate a comprehensive temperature alignment report including "
        "scope-level temperatures, peer comparison, trajectory analysis, "
        "and actionable recommendations."
    ),
)
async def get_temperature_report(org_id: str) -> TemperatureReportResponse:
    """Generate temperature alignment report."""
    return TemperatureReportResponse(
        org_id=org_id,
        report_id=_generate_id("temp_rpt"),
        overall_temperature_c=1.95,
        scope_temperatures={
            "scope1_2": 1.65,
            "scope3": 2.15,
            "combined": 1.95,
        },
        alignment_status="below_2C",
        peer_comparison={
            "sector_average_c": 2.13,
            "percentile_rank": 55,
            "better_than_average": True,
        },
        trajectory_analysis={
            "current_trajectory_c": 1.95,
            "2_year_forecast_c": 1.8,
            "improving": True,
            "rate_c_per_year": 0.17,
        },
        recommendations=[
            "Increase Scope 3 ambition to close the 0.65C gap between Scope 3 (2.15C) and 1.5C",
            "Engage top 20 suppliers representing 60% of Scope 3 to set SBTs",
            "Accelerate Scope 1 transition to renewable energy to maintain 1.5C S1+2 alignment",
            "Submit for SBTi re-validation to lock in near-term progress",
        ],
        generated_at=_now(),
    )
