"""
GL-GHG-APP Dashboard API

Provides pre-aggregated data for frontend dashboards, charts, and
visualizations. Includes KPIs, year-over-year trends, scope breakdowns,
geographic distributions, and compliance alerts.
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

router = APIRouter(prefix="/api/v1/dashboard", tags=["Dashboard"])


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class DashboardKPIs(BaseModel):
    """Key performance indicators for the dashboard overview."""
    org_id: str
    reporting_year: int
    total_tco2e: float
    scope1_tco2e: float
    scope2_location_tco2e: float
    scope2_market_tco2e: float
    scope3_tco2e: float
    yoy_change_pct: float
    yoy_change_direction: str
    base_year_change_pct: float
    intensity_per_revenue: float
    intensity_per_employee: float
    renewable_energy_pct: float
    target_progress_pct: Optional[float]
    data_quality_score: float
    data_quality_grade: str
    inventory_completeness_pct: float
    verification_status: str
    entity_count: int
    facility_count: int


class TrendYear(BaseModel):
    """Emission data for a single year in a trend series."""
    year: int
    scope1_tco2e: float
    scope2_tco2e: float
    scope3_tco2e: float
    total_tco2e: float
    intensity_per_revenue: float


class TrendResponse(BaseModel):
    """Year-over-year trend data."""
    org_id: str
    years: List[TrendYear]
    cagr_pct: float
    trend_direction: str
    base_year: int
    base_year_total: float


class DonutSegment(BaseModel):
    """A segment in a donut/pie chart."""
    label: str
    value: float
    percentage: float
    color: str


class WaterfallStep(BaseModel):
    """A step in a waterfall chart."""
    label: str
    value: float
    cumulative: float
    type: str  # start, increase, decrease, subtotal, end


class BreakdownResponse(BaseModel):
    """Scope breakdown data for charts."""
    inventory_id: str
    donut_data: List[DonutSegment]
    waterfall_data: List[WaterfallStep]
    scope1_categories: List[DonutSegment]
    scope3_top_categories: List[DonutSegment]


class GeographicEntry(BaseModel):
    """Emissions for a single country."""
    country_code: str
    country_name: str
    total_tco2e: float
    percentage_of_total: float
    scope1_tco2e: float
    scope2_tco2e: float
    facility_count: int
    employee_count: int


class GeographicResponse(BaseModel):
    """Geographic breakdown of emissions."""
    inventory_id: str
    countries: List[GeographicEntry]
    top_country: str
    top_country_pct: float


class Alert(BaseModel):
    """A data quality or compliance alert."""
    alert_id: str
    type: str
    severity: str
    title: str
    description: str
    scope: Optional[int]
    category: Optional[str]
    created_at: datetime
    resolved: bool


class AlertsResponse(BaseModel):
    """Dashboard alerts."""
    org_id: str
    total_alerts: int
    critical_count: int
    warning_count: int
    info_count: int
    alerts: List[Alert]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/metrics/{org_id}",
    response_model=DashboardKPIs,
    summary="Dashboard KPIs",
    description=(
        "Retrieve all key performance indicators for the dashboard overview. "
        "Includes scope totals, YoY change, intensity metrics, renewable energy "
        "percentage, target progress, and data quality."
    ),
)
async def get_dashboard_metrics(
    org_id: str,
    reporting_year: int = Query(2025, ge=1990, le=2100, description="Reporting year"),
) -> DashboardKPIs:
    return DashboardKPIs(
        org_id=org_id,
        reporting_year=reporting_year,
        total_tco2e=66001.5,
        scope1_tco2e=12450.8,
        scope2_location_tco2e=8320.5,
        scope2_market_tco2e=5890.3,
        scope3_tco2e=45230.2,
        yoy_change_pct=-3.2,
        yoy_change_direction="decreasing",
        base_year_change_pct=-6.4,
        intensity_per_revenue=42.3,
        intensity_per_employee=18.7,
        renewable_energy_pct=45.2,
        target_progress_pct=32.0,
        data_quality_score=82.4,
        data_quality_grade="B",
        inventory_completeness_pct=87.5,
        verification_status="not_started",
        entity_count=8,
        facility_count=4,
    )


@router.get(
    "/trends/{org_id}",
    response_model=TrendResponse,
    summary="Year-over-year emission trends",
    description=(
        "Returns up to 5 years of emission trends by scope for trend charts. "
        "Includes CAGR and trend direction."
    ),
)
async def get_trends(
    org_id: str,
    years: int = Query(5, ge=2, le=10, description="Number of years to include"),
) -> TrendResponse:
    trend_data = [
        TrendYear(year=2021, scope1_tco2e=14200.0, scope2_tco2e=9800.0, scope3_tco2e=48000.0, total_tco2e=72000.0, intensity_per_revenue=52.1),
        TrendYear(year=2022, scope1_tco2e=13800.0, scope2_tco2e=9400.0, scope3_tco2e=47200.0, total_tco2e=70400.0, intensity_per_revenue=49.5),
        TrendYear(year=2023, scope1_tco2e=13200.0, scope2_tco2e=9100.0, scope3_tco2e=46500.0, total_tco2e=68800.0, intensity_per_revenue=47.2),
        TrendYear(year=2024, scope1_tco2e=12850.0, scope2_tco2e=8600.0, scope3_tco2e=45800.0, total_tco2e=67250.0, intensity_per_revenue=44.8),
        TrendYear(year=2025, scope1_tco2e=12450.8, scope2_tco2e=8320.5, scope3_tco2e=45230.2, total_tco2e=66001.5, intensity_per_revenue=42.3),
    ]
    return TrendResponse(
        org_id=org_id,
        years=trend_data[-years:],
        cagr_pct=-2.1,
        trend_direction="decreasing",
        base_year=2019,
        base_year_total=70500.0,
    )


@router.get(
    "/breakdown/{inventory_id}",
    response_model=BreakdownResponse,
    summary="Scope breakdown for charts",
    description=(
        "Pre-aggregated data for donut charts (scope breakdown) and "
        "waterfall charts (emission build-up). Includes Scope 1 category "
        "breakdown and top Scope 3 categories."
    ),
)
async def get_breakdown(inventory_id: str) -> BreakdownResponse:
    total = 66001.5
    scope1 = 12450.8
    scope2 = 8320.5
    scope3 = 45230.2

    donut_data = [
        DonutSegment(label="Scope 1 - Direct", value=scope1, percentage=round(scope1 / total * 100, 1), color="#E53E3E"),
        DonutSegment(label="Scope 2 - Energy Indirect", value=scope2, percentage=round(scope2 / total * 100, 1), color="#DD6B20"),
        DonutSegment(label="Scope 3 - Value Chain", value=scope3, percentage=round(scope3 / total * 100, 1), color="#3182CE"),
    ]

    waterfall_data = [
        WaterfallStep(label="Scope 1", value=scope1, cumulative=scope1, type="start"),
        WaterfallStep(label="+ Scope 2 (Location)", value=scope2, cumulative=scope1 + scope2, type="increase"),
        WaterfallStep(label="Scope 1+2 Subtotal", value=scope1 + scope2, cumulative=scope1 + scope2, type="subtotal"),
        WaterfallStep(label="+ Scope 3", value=scope3, cumulative=total, type="increase"),
        WaterfallStep(label="Total Emissions", value=total, cumulative=total, type="end"),
    ]

    scope1_cats = [
        DonutSegment(label="Stationary Combustion", value=5820.3, percentage=46.7, color="#FC8181"),
        DonutSegment(label="Mobile Combustion", value=2340.5, percentage=18.8, color="#F6AD55"),
        DonutSegment(label="Process Emissions", value=1890.0, percentage=15.2, color="#68D391"),
        DonutSegment(label="Fugitive Emissions", value=1250.0, percentage=10.0, color="#63B3ED"),
        DonutSegment(label="Refrigerants", value=1150.0, percentage=9.2, color="#B794F4"),
    ]

    scope3_top = [
        DonutSegment(label="Cat 1: Purchased Goods", value=18500.0, percentage=40.9, color="#2B6CB0"),
        DonutSegment(label="Cat 4: Upstream Transport", value=4800.0, percentage=10.6, color="#2C7A7B"),
        DonutSegment(label="Cat 11: Use of Sold Products", value=3500.0, percentage=7.7, color="#285E61"),
        DonutSegment(label="Cat 2: Capital Goods", value=3200.0, percentage=7.1, color="#2D3748"),
        DonutSegment(label="Cat 6: Business Travel", value=2800.0, percentage=6.2, color="#553C9A"),
        DonutSegment(label="Other Categories", value=12430.2, percentage=27.5, color="#A0AEC0"),
    ]

    return BreakdownResponse(
        inventory_id=inventory_id,
        donut_data=donut_data,
        waterfall_data=waterfall_data,
        scope1_categories=scope1_cats,
        scope3_top_categories=scope3_top,
    )


@router.get(
    "/geographic/{inventory_id}",
    response_model=GeographicResponse,
    summary="Geographic emissions breakdown",
    description=(
        "Breakdown of emissions by country for map visualizations. "
        "Shows per-country totals, scope splits, and facility counts."
    ),
)
async def get_geographic_breakdown(inventory_id: str) -> GeographicResponse:
    total = 66001.5
    countries = [
        GeographicEntry(
            country_code="US", country_name="United States",
            total_tco2e=52800.0, percentage_of_total=round(52800.0 / total * 100, 1),
            scope1_tco2e=10500.0, scope2_tco2e=6700.0,
            facility_count=3, employee_count=1200,
        ),
        GeographicEntry(
            country_code="DE", country_name="Germany",
            total_tco2e=7920.0, percentage_of_total=round(7920.0 / total * 100, 1),
            scope1_tco2e=1200.0, scope2_tco2e=980.0,
            facility_count=1, employee_count=280,
        ),
        GeographicEntry(
            country_code="CN", country_name="China",
            total_tco2e=3960.0, percentage_of_total=round(3960.0 / total * 100, 1),
            scope1_tco2e=550.0, scope2_tco2e=480.0,
            facility_count=0, employee_count=120,
        ),
        GeographicEntry(
            country_code="GB", country_name="United Kingdom",
            total_tco2e=1321.5, percentage_of_total=round(1321.5 / total * 100, 1),
            scope1_tco2e=200.8, scope2_tco2e=160.5,
            facility_count=0, employee_count=250,
        ),
    ]
    return GeographicResponse(
        inventory_id=inventory_id,
        countries=countries,
        top_country="United States",
        top_country_pct=round(52800.0 / total * 100, 1),
    )


@router.get(
    "/alerts/{org_id}",
    response_model=AlertsResponse,
    summary="Data quality and compliance alerts",
    description=(
        "Active alerts for data quality issues, compliance warnings, "
        "and deadline notifications. Sorted by severity."
    ),
)
async def get_alerts(
    org_id: str,
    include_resolved: bool = Query(False, description="Include resolved alerts"),
) -> AlertsResponse:
    now = _now()
    alerts = [
        Alert(
            alert_id=_generate_id("alrt"),
            type="data_quality",
            severity="critical",
            title="Scope 3 Cat 1 data quality below threshold",
            description="Purchased Goods & Services (Cat 1) uses spend-based method for 80% of spend. Data quality score is 35/100, below the 50-point threshold.",
            scope=3,
            category="Cat 1: Purchased Goods & Services",
            created_at=now,
            resolved=False,
        ),
        Alert(
            alert_id=_generate_id("alrt"),
            type="completeness",
            severity="warning",
            title="Mobile combustion Q4 data missing",
            description="Fleet fuel consumption data for October through December 2025 has not been submitted. Estimated 180 tCO2e impact.",
            scope=1,
            category="Mobile Combustion",
            created_at=now,
            resolved=False,
        ),
        Alert(
            alert_id=_generate_id("alrt"),
            type="compliance",
            severity="warning",
            title="Verification not started",
            description="Third-party verification has not been initiated. Recommended to start at least 60 days before reporting deadline.",
            scope=None,
            category=None,
            created_at=now,
            resolved=False,
        ),
        Alert(
            alert_id=_generate_id("alrt"),
            type="compliance",
            severity="info",
            title="Biogenic CO2 not reported separately",
            description="GHG Protocol recommends reporting biogenic CO2 emissions separately. Currently not tracked.",
            scope=None,
            category=None,
            created_at=now,
            resolved=False,
        ),
        Alert(
            alert_id=_generate_id("alrt"),
            type="data_quality",
            severity="info",
            title="Employee commuting survey incomplete",
            description="Response rate is 35%. Consider re-sending survey to improve Scope 3 Cat 7 accuracy.",
            scope=3,
            category="Cat 7: Employee Commuting",
            created_at=now,
            resolved=False,
        ),
    ]

    if not include_resolved:
        alerts = [a for a in alerts if not a.resolved]

    critical = sum(1 for a in alerts if a.severity == "critical")
    warning = sum(1 for a in alerts if a.severity == "warning")
    info = sum(1 for a in alerts if a.severity == "info")

    return AlertsResponse(
        org_id=org_id,
        total_alerts=len(alerts),
        critical_count=critical,
        warning_count=warning,
        info_count=info,
        alerts=alerts,
    )
