"""
GL-ISO14064-APP Dashboard API

Provides pre-aggregated metrics and KPIs for the ISO 14064-1 compliance
dashboard.  Returns emissions by category, by gas, removals, data quality
scores, verification status, management plan progress, and year-over-year
trends -- all in a single endpoint for efficient frontend rendering.
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from decimal import Decimal
import uuid

router = APIRouter(prefix="/api/v1/iso14064/dashboard", tags=["Dashboard"])


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class CategoryMetric(BaseModel):
    """Emission metric for a single ISO category."""
    category: str
    category_name: str
    tco2e: float
    percentage_of_total: float
    source_count: int
    data_quality_tier: str
    significance: str


class GasMetric(BaseModel):
    """Emission metric for a single greenhouse gas."""
    gas: str
    gas_name: str
    tco2e: float
    percentage_of_total: float


class RemovalMetric(BaseModel):
    """Removal summary metric."""
    total_gross_tco2e: float
    total_credited_tco2e: float
    removal_count: int
    by_type: Dict[str, float]


class DataQualityMetric(BaseModel):
    """Data quality summary."""
    overall_score: float
    completeness_pct: float
    tier_distribution: Dict[str, int]


class ManagementMetric(BaseModel):
    """Management plan summary."""
    plan_count: int
    total_actions: int
    completed_actions: int
    in_progress_actions: int
    planned_reduction_tco2e: float


class DashboardResponse(BaseModel):
    """Complete dashboard metrics for ISO 14064-1."""
    org_id: str
    reporting_year: int
    gross_emissions_tco2e: float
    total_removals_tco2e: float
    net_emissions_tco2e: float
    biogenic_co2_tco2e: float
    by_category: List[CategoryMetric]
    by_gas: List[GasMetric]
    removals: RemovalMetric
    data_quality: DataQualityMetric
    verification_stage: str
    management: ManagementMetric
    yoy_change_pct: Optional[float]
    mandatory_element_completeness_pct: float
    significant_categories: List[str]
    generated_at: datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Simulated Dashboard Data
# ---------------------------------------------------------------------------

CATEGORY_DATA = [
    {
        "category": "category_1_direct",
        "category_name": "Category 1 - Direct",
        "tco2e": 12450.8,
        "source_count": 42,
        "data_quality_tier": "tier_2",
        "significance": "significant",
    },
    {
        "category": "category_2_energy",
        "category_name": "Category 2 - Energy Indirect",
        "tco2e": 8320.5,
        "source_count": 12,
        "data_quality_tier": "tier_2",
        "significance": "significant",
    },
    {
        "category": "category_3_transport",
        "category_name": "Category 3 - Transportation",
        "tco2e": 9840.0,
        "source_count": 28,
        "data_quality_tier": "tier_1",
        "significance": "significant",
    },
    {
        "category": "category_4_products_used",
        "category_name": "Category 4 - Products Used",
        "tco2e": 26180.2,
        "source_count": 35,
        "data_quality_tier": "tier_1",
        "significance": "significant",
    },
    {
        "category": "category_5_products_from_org",
        "category_name": "Category 5 - Products From Org",
        "tco2e": 8450.0,
        "source_count": 18,
        "data_quality_tier": "tier_1",
        "significance": "significant",
    },
    {
        "category": "category_6_other",
        "category_name": "Category 6 - Other",
        "tco2e": 760.0,
        "source_count": 5,
        "data_quality_tier": "tier_1",
        "significance": "not_significant",
    },
]

GAS_DATA = [
    {"gas": "CO2", "gas_name": "Carbon Dioxide", "tco2e": 52800.0},
    {"gas": "CH4", "gas_name": "Methane", "tco2e": 8400.0},
    {"gas": "N2O", "gas_name": "Nitrous Oxide", "tco2e": 3200.0},
    {"gas": "HFCs", "gas_name": "Hydrofluorocarbons", "tco2e": 1450.0},
    {"gas": "PFCs", "gas_name": "Perfluorocarbons", "tco2e": 120.0},
    {"gas": "SF6", "gas_name": "Sulfur Hexafluoride", "tco2e": 31.5},
    {"gas": "NF3", "gas_name": "Nitrogen Trifluoride", "tco2e": 0.0},
]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/{org_id}",
    response_model=DashboardResponse,
    summary="Dashboard metrics",
    description=(
        "Retrieve all dashboard metrics for an organization and reporting year. "
        "Includes emissions by category, by gas, removal summary, data quality, "
        "verification status, management plan progress, and year-over-year change."
    ),
)
async def get_dashboard(
    org_id: str,
    reporting_year: int = Query(2025, ge=1990, le=2100, description="Reporting year"),
) -> DashboardResponse:
    """Retrieve complete dashboard metrics for an organization-year."""
    gross = sum(c["tco2e"] for c in CATEGORY_DATA)
    removals_gross = 2100.0
    removals_credited = 1890.0
    net = round(gross - removals_credited, 2)

    # Category metrics with percentages
    categories = []
    for c in CATEGORY_DATA:
        pct = round(c["tco2e"] / gross * 100, 1) if gross > 0 else 0.0
        categories.append(CategoryMetric(
            category=c["category"],
            category_name=c["category_name"],
            tco2e=c["tco2e"],
            percentage_of_total=pct,
            source_count=c["source_count"],
            data_quality_tier=c["data_quality_tier"],
            significance=c["significance"],
        ))

    # Gas metrics with percentages
    gases = []
    for g in GAS_DATA:
        pct = round(g["tco2e"] / gross * 100, 1) if gross > 0 else 0.0
        gases.append(GasMetric(
            gas=g["gas"],
            gas_name=g["gas_name"],
            tco2e=g["tco2e"],
            percentage_of_total=pct,
        ))

    # Removal metrics
    removals = RemovalMetric(
        total_gross_tco2e=removals_gross,
        total_credited_tco2e=removals_credited,
        removal_count=4,
        by_type={
            "forestry": 950.0,
            "soil_carbon": 540.0,
            "wetland_restoration": 280.0,
            "ccs": 120.0,
        },
    )

    # Data quality
    data_quality = DataQualityMetric(
        overall_score=78.5,
        completeness_pct=89.2,
        tier_distribution={"tier_1": 48, "tier_2": 35, "tier_3": 12, "tier_4": 5},
    )

    # Management plan
    management = ManagementMetric(
        plan_count=1,
        total_actions=8,
        completed_actions=2,
        in_progress_actions=4,
        planned_reduction_tco2e=3200.0,
    )

    significant = [
        c["category"] for c in CATEGORY_DATA if c["significance"] == "significant"
    ]

    return DashboardResponse(
        org_id=org_id,
        reporting_year=reporting_year,
        gross_emissions_tco2e=round(gross, 2),
        total_removals_tco2e=removals_credited,
        net_emissions_tco2e=net,
        biogenic_co2_tco2e=450.0,
        by_category=categories,
        by_gas=gases,
        removals=removals,
        data_quality=data_quality,
        verification_stage="draft",
        management=management,
        yoy_change_pct=-3.8,
        mandatory_element_completeness_pct=64.3,
        significant_categories=significant,
        generated_at=_now(),
    )
