"""
Risk Assessment API Routes for GL-EUDR-APP v1.0

Provides risk assessment, heatmap generation, alert management,
trend analysis, and mitigation recommendations for EUDR compliance.
Risk levels are computed from country risk, commodity risk,
deforestation proximity, and supplier compliance history.

Prefix: /api/v1/risk
Tags: Risk Assessment
"""

import uuid
import math
import random
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/risk", tags=["Risk Assessment"])

# ---------------------------------------------------------------------------
# Reference Data
# ---------------------------------------------------------------------------

COUNTRY_RISK_LEVELS: Dict[str, str] = {
    "BRA": "high", "IDN": "high", "COL": "high", "MYS": "high",
    "COG": "high", "COD": "high", "CMR": "high", "GHA": "high",
    "CIV": "high", "PNG": "high", "PER": "medium", "ECU": "medium",
    "BOL": "medium", "VNM": "medium", "THA": "medium", "MEX": "medium",
    "ARG": "low", "CHL": "low", "DEU": "low", "FRA": "low",
    "ESP": "low", "ITA": "low", "NLD": "low", "USA": "low",
}

COMMODITY_RISK_LEVELS: Dict[str, str] = {
    "oil_palm": "high", "soya": "high", "cattle": "high",
    "cocoa": "medium", "coffee": "medium", "rubber": "medium",
    "wood": "medium",
}

# Country x Commodity matrix risk scores (0-100)
RISK_MATRIX: Dict[str, Dict[str, float]] = {
    "BRA": {"soya": 85, "cattle": 90, "coffee": 45, "wood": 70, "cocoa": 55},
    "IDN": {"oil_palm": 92, "rubber": 65, "cocoa": 50, "coffee": 40, "wood": 60},
    "COL": {"coffee": 55, "cocoa": 60, "oil_palm": 75, "cattle": 70},
    "MYS": {"oil_palm": 88, "rubber": 55, "wood": 50},
    "GHA": {"cocoa": 72, "wood": 55},
    "CIV": {"cocoa": 78, "rubber": 50},
    "COD": {"wood": 80, "cocoa": 65, "coffee": 55},
    "CMR": {"cocoa": 68, "wood": 62, "oil_palm": 60},
    "PER": {"coffee": 40, "cocoa": 45, "wood": 50},
    "ECU": {"cocoa": 42, "coffee": 38},
}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class RiskAssessRequest(BaseModel):
    """Request to run a risk assessment.

    Example::

        {
            "supplier_id": "sup_abc123",
            "plot_id": "plot_xyz"
        }
    """

    supplier_id: str = Field(..., description="Supplier to assess")
    plot_id: Optional[str] = Field(None, description="Specific plot to assess")
    country_iso3: Optional[str] = Field(None, description="Country override for assessment")
    commodity: Optional[str] = Field(None, description="Commodity override")


class RiskFactor(BaseModel):
    """A single risk factor contributing to the overall assessment."""

    factor: str = Field(..., description="Risk factor name")
    level: str = Field(..., description="low | medium | high | critical")
    score: float = Field(..., ge=0, le=100, description="Factor score 0-100")
    description: str


class RiskAssessmentResponse(BaseModel):
    """Full risk assessment result."""

    assessment_id: str = Field(..., description="Unique assessment identifier")
    supplier_id: str
    plot_id: Optional[str] = None
    overall_risk_level: str = Field(..., description="low | medium | high | critical")
    overall_risk_score: float = Field(..., ge=0, le=100)
    factors: List[RiskFactor]
    country_iso3: Optional[str] = None
    commodity: Optional[str] = None
    assessed_at: datetime
    valid_until: datetime = Field(
        ..., description="Assessment validity period (typically 90 days)"
    )


class HeatmapCell(BaseModel):
    """A single cell in the country x commodity risk heatmap."""

    country_iso3: str
    commodity: str
    risk_score: float = Field(..., ge=0, le=100)
    risk_level: str
    supplier_count: int = Field(0, description="Number of suppliers in this cell")


class RiskHeatmapResponse(BaseModel):
    """Country x commodity risk heatmap."""

    cells: List[HeatmapCell]
    countries: List[str]
    commodities: List[str]
    generated_at: datetime


class RiskAlert(BaseModel):
    """A risk alert notification."""

    alert_id: str
    alert_type: str = Field(
        ..., description="deforestation | compliance | document_expiry | regulatory_change"
    )
    severity: str = Field(..., description="info | warning | critical")
    title: str
    description: str
    country_iso3: Optional[str] = None
    commodity: Optional[str] = None
    supplier_id: Optional[str] = None
    plot_id: Optional[str] = None
    created_at: datetime
    acknowledged: bool = False


class RiskAlertListResponse(BaseModel):
    """Paginated list of risk alerts."""

    items: List[RiskAlert]
    page: int
    limit: int
    total: int
    total_pages: int


class TrendDataPoint(BaseModel):
    """A single data point in a risk trend series."""

    date: str = Field(..., description="ISO date (YYYY-MM-DD)")
    risk_score: float
    risk_level: str
    deforestation_alerts: int = 0


class RiskTrendResponse(BaseModel):
    """Risk trend data for a plot over time."""

    plot_id: str
    months: int
    data_points: List[TrendDataPoint]
    trend_direction: str = Field(
        ..., description="improving | stable | worsening"
    )


class MitigationRecommendation(BaseModel):
    """A recommended mitigation action."""

    recommendation_id: str
    priority: str = Field(..., description="low | medium | high | critical")
    category: str = Field(
        ..., description="documentation | monitoring | sourcing | verification"
    )
    title: str
    description: str
    estimated_risk_reduction: float = Field(
        ..., ge=0, le=100, description="Estimated risk score reduction"
    )
    estimated_effort: str = Field(..., description="low | medium | high")


class MitigationResponse(BaseModel):
    """Risk mitigation recommendations for an assessment."""

    assessment_id: str
    overall_risk_level: str
    overall_risk_score: float
    recommendations: List[MitigationRecommendation]
    total_potential_reduction: float


# ---------------------------------------------------------------------------
# In-Memory Storage (v1.0)
# ---------------------------------------------------------------------------

_assessments: Dict[str, dict] = {}
_alerts: List[dict] = []


def _score_to_level(score: float) -> str:
    if score >= 80:
        return "critical"
    elif score >= 60:
        return "high"
    elif score >= 40:
        return "medium"
    else:
        return "low"


def _generate_seed_alerts() -> None:
    """Generate sample alerts if none exist."""
    if _alerts:
        return

    now = datetime.now(timezone.utc)
    samples = [
        {
            "alert_type": "deforestation",
            "severity": "critical",
            "title": "Deforestation detected near plot boundaries in Para, Brazil",
            "description": "Satellite imagery from GLAD alerts shows 15 hectares of tree cover loss within 5km of registered plots in Para state.",
            "country_iso3": "BRA",
            "commodity": "soya",
        },
        {
            "alert_type": "compliance",
            "severity": "warning",
            "title": "DDS submission deadline approaching for Q1 2025",
            "description": "12 suppliers have pending DDS submissions due within 30 days.",
            "country_iso3": None,
            "commodity": None,
        },
        {
            "alert_type": "document_expiry",
            "severity": "warning",
            "title": "FSC certificates expiring for 3 suppliers",
            "description": "Forest Stewardship Council certificates for 3 Indonesian suppliers expire within 60 days.",
            "country_iso3": "IDN",
            "commodity": "wood",
        },
        {
            "alert_type": "regulatory_change",
            "severity": "info",
            "title": "EU Commission updates EUDR benchmarking criteria",
            "description": "Updated country benchmarking methodology published. Risk classifications may change for 8 countries.",
            "country_iso3": None,
            "commodity": None,
        },
        {
            "alert_type": "deforestation",
            "severity": "critical",
            "title": "RADD alert: forest disturbance in Kalimantan",
            "description": "RADD system detected 42 hectares of forest disturbance in West Kalimantan, overlapping with registered oil palm plots.",
            "country_iso3": "IDN",
            "commodity": "oil_palm",
        },
    ]

    for i, sample in enumerate(samples):
        _alerts.append({
            "alert_id": f"alert_{uuid.uuid4().hex[:8]}",
            **sample,
            "supplier_id": None,
            "plot_id": None,
            "created_at": now - timedelta(hours=i * 12),
            "acknowledged": False,
        })


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/assess",
    response_model=RiskAssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Run risk assessment",
    description="Execute a risk assessment for a supplier and optionally a specific plot.",
)
async def run_risk_assessment(body: RiskAssessRequest) -> RiskAssessmentResponse:
    """
    Run a composite risk assessment.

    Evaluates country risk, commodity risk, deforestation proximity,
    supplier compliance history, and document completeness.

    Returns:
        201 with risk assessment results.
    """
    now = datetime.now(timezone.utc)
    assessment_id = f"ra_{uuid.uuid4().hex[:12]}"

    country = (body.country_iso3 or "BRA").upper()
    commodity = (body.commodity or "soya").lower()

    # Factor 1: Country risk
    country_level = COUNTRY_RISK_LEVELS.get(country, "medium")
    country_score = {"high": 75, "medium": 45, "low": 20}.get(country_level, 45)

    # Factor 2: Commodity risk
    commodity_level = COMMODITY_RISK_LEVELS.get(commodity, "medium")
    commodity_score = {"high": 70, "medium": 40, "low": 15}.get(commodity_level, 40)

    # Factor 3: Deforestation proximity (simulated)
    deforestation_score = RISK_MATRIX.get(country, {}).get(commodity, 50.0)

    # Factor 4: Compliance history (simulated baseline)
    compliance_score = 35.0

    # Factor 5: Document completeness (simulated baseline)
    document_score = 40.0

    factors = [
        RiskFactor(
            factor="country_risk",
            level=country_level,
            score=country_score,
            description=f"Country-level deforestation risk for {country}",
        ),
        RiskFactor(
            factor="commodity_risk",
            level=commodity_level,
            score=commodity_score,
            description=f"Commodity-specific risk for {commodity}",
        ),
        RiskFactor(
            factor="deforestation_proximity",
            level=_score_to_level(deforestation_score),
            score=deforestation_score,
            description="Proximity to recent deforestation events (satellite data)",
        ),
        RiskFactor(
            factor="compliance_history",
            level=_score_to_level(compliance_score),
            score=compliance_score,
            description="Historical compliance record and DDS submission history",
        ),
        RiskFactor(
            factor="document_completeness",
            level=_score_to_level(document_score),
            score=document_score,
            description="Completeness and verification status of supporting documents",
        ),
    ]

    # Weighted composite score
    weights = {
        "country_risk": 0.20,
        "commodity_risk": 0.15,
        "deforestation_proximity": 0.35,
        "compliance_history": 0.15,
        "document_completeness": 0.15,
    }
    overall_score = sum(
        f.score * weights.get(f.factor, 0.2) for f in factors
    )
    overall_score = round(min(100.0, max(0.0, overall_score)), 1)
    overall_level = _score_to_level(overall_score)

    result = {
        "assessment_id": assessment_id,
        "supplier_id": body.supplier_id,
        "plot_id": body.plot_id,
        "overall_risk_level": overall_level,
        "overall_risk_score": overall_score,
        "factors": [f.model_dump() for f in factors],
        "country_iso3": country,
        "commodity": commodity,
        "assessed_at": now,
        "valid_until": now + timedelta(days=90),
    }
    _assessments[assessment_id] = result

    logger.info(
        "Risk assessment: %s -> score=%.1f (%s)",
        assessment_id,
        overall_score,
        overall_level,
    )

    return RiskAssessmentResponse(**result)


@router.get(
    "/heatmap",
    response_model=RiskHeatmapResponse,
    summary="Risk heatmap",
    description="Generate a country x commodity risk heatmap matrix.",
)
async def get_risk_heatmap() -> RiskHeatmapResponse:
    """
    Generate a risk heatmap showing risk scores for each country and
    commodity combination.

    Returns:
        200 with heatmap data.
    """
    cells: List[HeatmapCell] = []
    countries_set = set()
    commodities_set = set()

    for country, commodities in RISK_MATRIX.items():
        for commodity, score in commodities.items():
            cells.append(HeatmapCell(
                country_iso3=country,
                commodity=commodity,
                risk_score=score,
                risk_level=_score_to_level(score),
                supplier_count=0,  # Placeholder for v1.0
            ))
            countries_set.add(country)
            commodities_set.add(commodity)

    return RiskHeatmapResponse(
        cells=cells,
        countries=sorted(countries_set),
        commodities=sorted(commodities_set),
        generated_at=datetime.now(timezone.utc),
    )


@router.get(
    "/alerts",
    response_model=RiskAlertListResponse,
    summary="Risk alerts",
    description="Retrieve risk alerts with optional filtering.",
)
async def get_risk_alerts(
    min_level: Optional[str] = Query(
        None, description="Minimum severity: info | warning | critical"
    ),
    commodity: Optional[str] = Query(None, description="Filter by commodity"),
    country: Optional[str] = Query(None, description="Filter by country ISO3"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
) -> RiskAlertListResponse:
    """
    Retrieve paginated risk alerts.

    Seed alerts are generated on first access to demonstrate the alert model.

    Returns:
        200 with paginated alert list.
    """
    _generate_seed_alerts()

    results = list(_alerts)

    severity_order = {"info": 0, "warning": 1, "critical": 2}
    if min_level:
        min_ord = severity_order.get(min_level, 0)
        results = [a for a in results if severity_order.get(a["severity"], 0) >= min_ord]

    if commodity:
        results = [a for a in results if a.get("commodity") == commodity.lower()]
    if country:
        results = [a for a in results if a.get("country_iso3") == country.upper()]

    results.sort(key=lambda a: a["created_at"], reverse=True)

    total = len(results)
    total_pages = max(1, math.ceil(total / limit))
    start = (page - 1) * limit
    page_items = results[start : start + limit]

    return RiskAlertListResponse(
        items=[RiskAlert(**a) for a in page_items],
        page=page,
        limit=limit,
        total=total,
        total_pages=total_pages,
    )


@router.get(
    "/trends/{plot_id}",
    response_model=RiskTrendResponse,
    summary="Risk trend",
    description="Retrieve risk trend data for a plot over a specified number of months.",
)
async def get_risk_trend(
    plot_id: str,
    months: int = Query(6, ge=1, le=24, description="Number of months of historical data"),
) -> RiskTrendResponse:
    """
    Generate a risk trend time series for a plot.

    In v1.0, data is simulated with a realistic pattern. Production
    would query historical assessment records.

    Returns:
        200 with trend data.
    """
    now = datetime.now(timezone.utc)
    data_points: List[TrendDataPoint] = []

    # Simulate a trend: start high, gradually improve
    random.seed(hash(plot_id))
    base_score = random.uniform(50, 80)

    for i in range(months):
        date = now - timedelta(days=(months - i) * 30)
        # Simulate gradual improvement with noise
        improvement = i * 2.5
        noise = random.uniform(-5, 5)
        score = max(10, min(95, base_score - improvement + noise))
        score = round(score, 1)

        data_points.append(TrendDataPoint(
            date=date.strftime("%Y-%m-%d"),
            risk_score=score,
            risk_level=_score_to_level(score),
            deforestation_alerts=max(0, int((score - 40) / 15)),
        ))

    # Determine trend direction
    if len(data_points) >= 2:
        first_half_avg = sum(dp.risk_score for dp in data_points[: len(data_points) // 2]) / (
            len(data_points) // 2
        )
        second_half_avg = sum(dp.risk_score for dp in data_points[len(data_points) // 2 :]) / (
            len(data_points) - len(data_points) // 2
        )
        if second_half_avg < first_half_avg - 5:
            direction = "improving"
        elif second_half_avg > first_half_avg + 5:
            direction = "worsening"
        else:
            direction = "stable"
    else:
        direction = "stable"

    return RiskTrendResponse(
        plot_id=plot_id,
        months=months,
        data_points=data_points,
        trend_direction=direction,
    )


@router.get(
    "/mitigations/{assessment_id}",
    response_model=MitigationResponse,
    summary="Risk mitigations",
    description="Get mitigation recommendations based on a risk assessment.",
)
async def get_mitigations(assessment_id: str) -> MitigationResponse:
    """
    Generate risk mitigation recommendations based on the factors
    identified in a risk assessment.

    Returns:
        200 with prioritized mitigation recommendations.

    Raises:
        404 if assessment not found.
    """
    assessment = _assessments.get(assessment_id)
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assessment '{assessment_id}' not found",
        )

    recommendations: List[MitigationRecommendation] = []
    factors = assessment.get("factors", [])

    for factor in factors:
        f_score = factor["score"] if isinstance(factor, dict) else factor.score
        f_name = factor["factor"] if isinstance(factor, dict) else factor.factor
        f_level = factor["level"] if isinstance(factor, dict) else factor.level

        if f_score < 40:
            continue  # Low-risk factors do not need mitigation

        if f_name == "country_risk":
            recommendations.append(MitigationRecommendation(
                recommendation_id=f"mit_{uuid.uuid4().hex[:8]}",
                priority="high" if f_score >= 70 else "medium",
                category="sourcing",
                title="Diversify sourcing to lower-risk origins",
                description=(
                    "Consider supplementing supply from EU-benchmarked low-risk "
                    "countries to reduce portfolio-level country risk."
                ),
                estimated_risk_reduction=min(15.0, f_score * 0.2),
                estimated_effort="high",
            ))

        elif f_name == "commodity_risk":
            recommendations.append(MitigationRecommendation(
                recommendation_id=f"mit_{uuid.uuid4().hex[:8]}",
                priority="medium",
                category="verification",
                title="Obtain third-party commodity certification",
                description=(
                    "Secure recognized certifications (FSC, RSPO, Rainforest Alliance) "
                    "to demonstrate deforestation-free sourcing."
                ),
                estimated_risk_reduction=min(12.0, f_score * 0.18),
                estimated_effort="medium",
            ))

        elif f_name == "deforestation_proximity":
            recommendations.append(MitigationRecommendation(
                recommendation_id=f"mit_{uuid.uuid4().hex[:8]}",
                priority="critical" if f_score >= 80 else "high",
                category="monitoring",
                title="Enhance satellite monitoring for affected plots",
                description=(
                    "Deploy continuous satellite monitoring (Sentinel-2, GLAD, RADD) "
                    "with automated alerts for any tree cover change within 10km of plots."
                ),
                estimated_risk_reduction=min(20.0, f_score * 0.25),
                estimated_effort="medium",
            ))

        elif f_name == "compliance_history":
            recommendations.append(MitigationRecommendation(
                recommendation_id=f"mit_{uuid.uuid4().hex[:8]}",
                priority="medium",
                category="documentation",
                title="Establish regular DDS submission schedule",
                description=(
                    "Implement a quarterly DDS generation and submission workflow "
                    "to maintain continuous compliance documentation."
                ),
                estimated_risk_reduction=min(10.0, f_score * 0.15),
                estimated_effort="low",
            ))

        elif f_name == "document_completeness":
            recommendations.append(MitigationRecommendation(
                recommendation_id=f"mit_{uuid.uuid4().hex[:8]}",
                priority="high" if f_score >= 60 else "medium",
                category="documentation",
                title="Complete document gap remediation",
                description=(
                    "Upload and verify all required EUDR documents: certificates, "
                    "permits, land titles, invoices, and transport records."
                ),
                estimated_risk_reduction=min(15.0, f_score * 0.2),
                estimated_effort="medium",
            ))

    total_reduction = round(
        sum(r.estimated_risk_reduction for r in recommendations), 1
    )

    return MitigationResponse(
        assessment_id=assessment_id,
        overall_risk_level=assessment["overall_risk_level"],
        overall_risk_score=assessment["overall_risk_score"],
        recommendations=recommendations,
        total_potential_reduction=total_reduction,
    )
