# -*- coding: utf-8 -*-
"""
PAIMandatoryEngine - PACK-011 SFDR Article 9 Engine 6
=======================================================

Mandatory PAI (Principal Adverse Impact) indicator engine for SFDR Article 9
products. Article 9 products must consider ALL mandatory PAI indicators from
SFDR RTS Annex I Table 1 (18 indicators), PLUS select additional indicators
from Table 2 (environmental) and Table 3 (social).

Indicator Groups:
    Table 1 - Mandatory (18 indicators):
        PAI 1-6   : Climate & GHG (Scope 1/2/3, carbon footprint, fossil fuel)
        PAI 7-9   : Biodiversity, Water, Waste (environmental)
        PAI 10-14 : Social & Governance (UNGC, gender pay, board diversity, weapons)
        PAI 15-16 : Sovereign (country GHG intensity, social violations)
        PAI 17-18 : Real Estate (fossil fuel exposure, energy efficiency)

    Table 2 - Additional Environmental (at least 1 required):
        Inorganic pollutants, Air pollutants, Ozone-depleting substances,
        No carbon reduction initiatives, Water usage

    Table 3 - Additional Social (at least 1 required):
        No accident prevention, Accident rate, No supplier code of conduct,
        No grievance mechanism, No whistleblower protection

Additional Article 9 Requirements:
    - PAI integration in investment decision tracking
    - Action plan generation for adverse impacts
    - Minimum 70% data quality threshold
    - Period-over-period comparison

Formulas:
    PAI 1 (GHG): SUM(attribution_factor_i * total_ghg_i)
        attribution_factor = value_i / evic_i
    PAI 2 (Carbon Footprint): PAI_1_total / portfolio_value_EUR_M
    PAI 3 (WACI): SUM(weight_i * (ghg_i / revenue_i))
    PAI 4-14: Share = SUM(value_exposed) / total_nav * 100
        or Weighted Average = SUM(weight_i * metric_i)
    Coverage = holdings_with_data / total_holdings * 100

Zero-Hallucination:
    - All calculations use deterministic Python arithmetic
    - Coverage ratios track data availability per indicator
    - Data quality flags distinguish REPORTED / ESTIMATED / MODELED
    - SHA-256 provenance hashing on every result
    - No LLM involvement in any numeric calculation path

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-011 SFDR Article 9
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default on zero denominator.

    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value to return if denominator is zero.

    Returns:
        Result of division or default value.
    """
    if denominator == 0.0:
        return default
    return numerator / denominator

def _safe_pct(numerator: float, denominator: float) -> float:
    """Calculate percentage safely.

    Args:
        numerator: The dividend.
        denominator: The divisor.

    Returns:
        Percentage or 0.0 on zero denominator.
    """
    if denominator == 0.0:
        return 0.0
    return (numerator / denominator) * 100.0

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PAIMandatoryStatus(str, Enum):
    """Overall PAI compliance status for Article 9."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    INSUFFICIENT_DATA = "insufficient_data"

class DataQualityLevel(str, Enum):
    """Data quality classification for PAI data."""
    REPORTED = "REPORTED"
    ESTIMATED = "ESTIMATED"
    MODELED = "MODELED"
    NOT_AVAILABLE = "NOT_AVAILABLE"

class PAIIndicatorId(str, Enum):
    """All 18 mandatory PAI indicators plus additional Table 2/3 indicators."""
    # Table 1 - Mandatory
    PAI_1 = "PAI_1"
    PAI_2 = "PAI_2"
    PAI_3 = "PAI_3"
    PAI_4 = "PAI_4"
    PAI_5 = "PAI_5"
    PAI_6 = "PAI_6"
    PAI_7 = "PAI_7"
    PAI_8 = "PAI_8"
    PAI_9 = "PAI_9"
    PAI_10 = "PAI_10"
    PAI_11 = "PAI_11"
    PAI_12 = "PAI_12"
    PAI_13 = "PAI_13"
    PAI_14 = "PAI_14"
    PAI_15 = "PAI_15"
    PAI_16 = "PAI_16"
    PAI_17 = "PAI_17"
    PAI_18 = "PAI_18"
    # Table 2 - Additional Environmental
    T2_INORGANIC_POLLUTANTS = "T2_INORGANIC_POLLUTANTS"
    T2_AIR_POLLUTANTS = "T2_AIR_POLLUTANTS"
    T2_OZONE_DEPLETING = "T2_OZONE_DEPLETING"
    T2_NO_CARBON_INITIATIVES = "T2_NO_CARBON_INITIATIVES"
    T2_WATER_USAGE = "T2_WATER_USAGE"
    # Table 3 - Additional Social
    T3_NO_ACCIDENT_PREVENTION = "T3_NO_ACCIDENT_PREVENTION"
    T3_ACCIDENT_RATE = "T3_ACCIDENT_RATE"
    T3_NO_SUPPLIER_CODE = "T3_NO_SUPPLIER_CODE"
    T3_NO_GRIEVANCE = "T3_NO_GRIEVANCE"
    T3_NO_WHISTLEBLOWER = "T3_NO_WHISTLEBLOWER"

class PAICategory(str, Enum):
    """PAI indicator groupings."""
    CLIMATE_GHG = "climate_ghg"
    ENVIRONMENT = "environment"
    SOCIAL = "social"
    SOVEREIGN = "sovereign"
    REAL_ESTATE = "real_estate"
    ADDITIONAL_ENVIRONMENTAL = "additional_environmental"
    ADDITIONAL_SOCIAL = "additional_social"

# ---------------------------------------------------------------------------
# PAI Metadata Registry
# ---------------------------------------------------------------------------

PAI_METADATA: Dict[str, Dict[str, Any]] = {
    PAIIndicatorId.PAI_1.value: {
        "name": "GHG emissions",
        "category": PAICategory.CLIMATE_GHG,
        "unit": "tCO2eq",
        "aggregation": "attribution",
        "table": "Table 1",
    },
    PAIIndicatorId.PAI_2.value: {
        "name": "Carbon footprint",
        "category": PAICategory.CLIMATE_GHG,
        "unit": "tCO2eq/EUR M",
        "aggregation": "sum_over_nav",
        "table": "Table 1",
    },
    PAIIndicatorId.PAI_3.value: {
        "name": "GHG intensity of investee companies",
        "category": PAICategory.CLIMATE_GHG,
        "unit": "tCO2eq/EUR M revenue",
        "aggregation": "weighted_average",
        "table": "Table 1",
    },
    PAIIndicatorId.PAI_4.value: {
        "name": "Exposure to fossil fuel companies",
        "category": PAICategory.CLIMATE_GHG,
        "unit": "%",
        "aggregation": "share",
        "table": "Table 1",
    },
    PAIIndicatorId.PAI_5.value: {
        "name": "Non-renewable energy share",
        "category": PAICategory.CLIMATE_GHG,
        "unit": "%",
        "aggregation": "weighted_average",
        "table": "Table 1",
    },
    PAIIndicatorId.PAI_6.value: {
        "name": "Energy consumption intensity per high impact sector",
        "category": PAICategory.CLIMATE_GHG,
        "unit": "GWh/EUR M revenue",
        "aggregation": "weighted_average_by_sector",
        "table": "Table 1",
    },
    PAIIndicatorId.PAI_7.value: {
        "name": "Activities affecting biodiversity-sensitive areas",
        "category": PAICategory.ENVIRONMENT,
        "unit": "%",
        "aggregation": "share",
        "table": "Table 1",
    },
    PAIIndicatorId.PAI_8.value: {
        "name": "Emissions to water",
        "category": PAICategory.ENVIRONMENT,
        "unit": "tonnes",
        "aggregation": "attribution",
        "table": "Table 1",
    },
    PAIIndicatorId.PAI_9.value: {
        "name": "Hazardous waste and radioactive waste ratio",
        "category": PAICategory.ENVIRONMENT,
        "unit": "tonnes",
        "aggregation": "attribution",
        "table": "Table 1",
    },
    PAIIndicatorId.PAI_10.value: {
        "name": "UNGC/OECD principles violations",
        "category": PAICategory.SOCIAL,
        "unit": "%",
        "aggregation": "share",
        "table": "Table 1",
    },
    PAIIndicatorId.PAI_11.value: {
        "name": "Lack of UNGC/OECD compliance processes",
        "category": PAICategory.SOCIAL,
        "unit": "%",
        "aggregation": "share",
        "table": "Table 1",
    },
    PAIIndicatorId.PAI_12.value: {
        "name": "Unadjusted gender pay gap",
        "category": PAICategory.SOCIAL,
        "unit": "%",
        "aggregation": "weighted_average",
        "table": "Table 1",
    },
    PAIIndicatorId.PAI_13.value: {
        "name": "Board gender diversity",
        "category": PAICategory.SOCIAL,
        "unit": "%",
        "aggregation": "weighted_average",
        "table": "Table 1",
    },
    PAIIndicatorId.PAI_14.value: {
        "name": "Exposure to controversial weapons",
        "category": PAICategory.SOCIAL,
        "unit": "%",
        "aggregation": "share",
        "table": "Table 1",
    },
    PAIIndicatorId.PAI_15.value: {
        "name": "GHG intensity of investee countries",
        "category": PAICategory.SOVEREIGN,
        "unit": "tCO2eq/EUR M GDP",
        "aggregation": "weighted_average",
        "table": "Table 1",
    },
    PAIIndicatorId.PAI_16.value: {
        "name": "Investee countries subject to social violations",
        "category": PAICategory.SOVEREIGN,
        "unit": "%",
        "aggregation": "share",
        "table": "Table 1",
    },
    PAIIndicatorId.PAI_17.value: {
        "name": "Fossil fuels through real estate assets",
        "category": PAICategory.REAL_ESTATE,
        "unit": "%",
        "aggregation": "share",
        "table": "Table 1",
    },
    PAIIndicatorId.PAI_18.value: {
        "name": "Energy-inefficient real estate assets",
        "category": PAICategory.REAL_ESTATE,
        "unit": "%",
        "aggregation": "share",
        "table": "Table 1",
    },
    # Table 2 - Additional Environmental
    PAIIndicatorId.T2_INORGANIC_POLLUTANTS.value: {
        "name": "Inorganic pollutant emissions",
        "category": PAICategory.ADDITIONAL_ENVIRONMENTAL,
        "unit": "tonnes",
        "aggregation": "attribution",
        "table": "Table 2",
    },
    PAIIndicatorId.T2_AIR_POLLUTANTS.value: {
        "name": "Air pollutant emissions",
        "category": PAICategory.ADDITIONAL_ENVIRONMENTAL,
        "unit": "tonnes",
        "aggregation": "attribution",
        "table": "Table 2",
    },
    PAIIndicatorId.T2_OZONE_DEPLETING.value: {
        "name": "Ozone-depleting substance emissions",
        "category": PAICategory.ADDITIONAL_ENVIRONMENTAL,
        "unit": "tonnes",
        "aggregation": "attribution",
        "table": "Table 2",
    },
    PAIIndicatorId.T2_NO_CARBON_INITIATIVES.value: {
        "name": "Lack of carbon emission reduction initiatives",
        "category": PAICategory.ADDITIONAL_ENVIRONMENTAL,
        "unit": "%",
        "aggregation": "share",
        "table": "Table 2",
    },
    PAIIndicatorId.T2_WATER_USAGE.value: {
        "name": "Water usage and recycling",
        "category": PAICategory.ADDITIONAL_ENVIRONMENTAL,
        "unit": "m3/EUR M revenue",
        "aggregation": "weighted_average",
        "table": "Table 2",
    },
    # Table 3 - Additional Social
    PAIIndicatorId.T3_NO_ACCIDENT_PREVENTION.value: {
        "name": "Lack of accident prevention policy",
        "category": PAICategory.ADDITIONAL_SOCIAL,
        "unit": "%",
        "aggregation": "share",
        "table": "Table 3",
    },
    PAIIndicatorId.T3_ACCIDENT_RATE.value: {
        "name": "Accident rate",
        "category": PAICategory.ADDITIONAL_SOCIAL,
        "unit": "per 1M hours",
        "aggregation": "weighted_average",
        "table": "Table 3",
    },
    PAIIndicatorId.T3_NO_SUPPLIER_CODE.value: {
        "name": "Lack of supplier code of conduct",
        "category": PAICategory.ADDITIONAL_SOCIAL,
        "unit": "%",
        "aggregation": "share",
        "table": "Table 3",
    },
    PAIIndicatorId.T3_NO_GRIEVANCE.value: {
        "name": "Lack of grievance/complaints handling mechanism",
        "category": PAICategory.ADDITIONAL_SOCIAL,
        "unit": "%",
        "aggregation": "share",
        "table": "Table 3",
    },
    PAIIndicatorId.T3_NO_WHISTLEBLOWER.value: {
        "name": "Lack of whistleblower protection",
        "category": PAICategory.ADDITIONAL_SOCIAL,
        "unit": "%",
        "aggregation": "share",
        "table": "Table 3",
    },
}

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class InvesteeFullData(BaseModel):
    """Complete PAI data for a single investee holding.

    Contains financial exposure, all 18 mandatory PAI fields, plus
    additional Table 2 environmental and Table 3 social fields.
    """
    investee_id: str = Field(
        default_factory=_new_uuid, description="Unique investee identifier"
    )
    investee_name: str = Field(default="", description="Investee name")
    investee_type: str = Field(
        default="CORPORATE", description="CORPORATE, SOVEREIGN, or REAL_ESTATE"
    )
    # Financial data
    value_eur: float = Field(default=0.0, description="Holding value (EUR)")
    weight_pct: float = Field(default=0.0, description="Portfolio weight (%)")
    enterprise_value_eur: float = Field(
        default=0.0, description="Enterprise value including cash (EUR)"
    )
    revenue_eur: float = Field(default=0.0, description="Annual revenue (EUR)")
    # GHG data (PAI 1-3)
    scope1_tco2eq: float = Field(default=0.0, description="Scope 1 GHG emissions")
    scope2_tco2eq: float = Field(default=0.0, description="Scope 2 GHG emissions")
    scope3_tco2eq: float = Field(default=0.0, description="Scope 3 GHG emissions")
    total_ghg_tco2eq: float = Field(default=0.0, description="Total GHG emissions")
    # Energy data (PAI 4-6)
    is_fossil_fuel_company: Optional[bool] = Field(
        default=None, description="Active in fossil fuel sector"
    )
    non_renewable_energy_share_pct: Optional[float] = Field(
        default=None, description="Non-renewable energy share (%)"
    )
    energy_consumption_gwh: Optional[float] = Field(
        default=None, description="Total energy consumption (GWh)"
    )
    nace_sector: str = Field(default="", description="NACE sector code")
    # Environmental data (PAI 7-9)
    affects_biodiversity_area: Optional[bool] = Field(
        default=None, description="Activities affect biodiversity-sensitive areas"
    )
    emissions_to_water_tonnes: Optional[float] = Field(
        default=None, description="Water pollutant emissions (tonnes)"
    )
    hazardous_waste_tonnes: Optional[float] = Field(
        default=None, description="Hazardous waste (tonnes)"
    )
    radioactive_waste_tonnes: Optional[float] = Field(
        default=None, description="Radioactive waste (tonnes)"
    )
    # Social data (PAI 10-14)
    has_ungc_violations: Optional[bool] = Field(
        default=None, description="UNGC/OECD principle violations"
    )
    has_compliance_mechanisms: Optional[bool] = Field(
        default=None, description="Has UNGC/OECD compliance processes"
    )
    gender_pay_gap_pct: Optional[float] = Field(
        default=None, description="Unadjusted gender pay gap (%)"
    )
    female_board_pct: Optional[float] = Field(
        default=None, description="Female board members (%)"
    )
    involved_controversial_weapons: Optional[bool] = Field(
        default=None, description="Involved in controversial weapons"
    )
    # Sovereign data (PAI 15-16)
    country_code: str = Field(default="", description="Country ISO code")
    country_ghg_intensity: Optional[float] = Field(
        default=None, description="Country GHG intensity (tCO2eq/EUR M GDP)"
    )
    country_social_violations: Optional[bool] = Field(
        default=None, description="Country subject to social violations"
    )
    # Real estate data (PAI 17-18)
    re_fossil_fuel_involved: Optional[bool] = Field(
        default=None, description="Real estate involved in fossil fuels"
    )
    re_energy_inefficient: Optional[bool] = Field(
        default=None, description="Real estate energy-inefficient"
    )
    # Table 2 - Additional Environmental
    inorganic_pollutants_tonnes: Optional[float] = Field(
        default=None, description="Inorganic pollutant emissions (tonnes)"
    )
    air_pollutants_tonnes: Optional[float] = Field(
        default=None, description="Air pollutant emissions (tonnes)"
    )
    ozone_depleting_tonnes: Optional[float] = Field(
        default=None, description="Ozone-depleting substance emissions (tonnes)"
    )
    has_carbon_initiatives: Optional[bool] = Field(
        default=None, description="Has carbon reduction initiatives"
    )
    water_usage_m3: Optional[float] = Field(
        default=None, description="Water usage (m3)"
    )
    # Table 3 - Additional Social
    has_accident_prevention: Optional[bool] = Field(
        default=None, description="Has accident prevention policy"
    )
    accident_rate: Optional[float] = Field(
        default=None, description="Accident rate (per 1M hours worked)"
    )
    has_supplier_code: Optional[bool] = Field(
        default=None, description="Has supplier code of conduct"
    )
    has_grievance_mechanism: Optional[bool] = Field(
        default=None, description="Has grievance handling mechanism"
    )
    has_whistleblower_protection: Optional[bool] = Field(
        default=None, description="Has whistleblower protection"
    )
    # Data quality
    data_quality: DataQualityLevel = Field(
        default=DataQualityLevel.NOT_AVAILABLE, description="Data quality level"
    )

    @model_validator(mode="after")
    def _compute_total_ghg(self) -> "InvesteeFullData":
        """Auto-compute total GHG from scopes if not provided."""
        if self.total_ghg_tco2eq <= 0.0:
            self.total_ghg_tco2eq = (
                self.scope1_tco2eq + self.scope2_tco2eq + self.scope3_tco2eq
            )
        return self

class PAISingleResult(BaseModel):
    """Result for a single PAI indicator calculation."""
    indicator_id: str = Field(description="PAI indicator identifier")
    indicator_name: str = Field(default="", description="Indicator name")
    category: str = Field(default="", description="PAI category")
    table: str = Field(default="Table 1", description="RTS table reference")
    value: Optional[float] = Field(default=None, description="Calculated value")
    unit: str = Field(default="", description="Unit of measurement")
    sub_values: Dict[str, Any] = Field(
        default_factory=dict, description="Breakdown values"
    )
    total_holdings: int = Field(default=0, description="Total holdings")
    holdings_with_data: int = Field(default=0, description="Holdings with data")
    coverage_pct: float = Field(default=0.0, description="Data coverage (%)")
    coverage_by_value_pct: float = Field(
        default=0.0, description="Coverage by portfolio value (%)"
    )
    data_quality_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Quality distribution"
    )
    is_sufficient: bool = Field(
        default=False, description="Coverage meets threshold"
    )
    methodology_note: str = Field(default="", description="Calculation methodology")
    provenance_hash: str = Field(default="", description="SHA-256 hash")

class IntegrationAssessment(BaseModel):
    """Assessment of PAI integration in investment decisions."""
    assessment_id: str = Field(default_factory=_new_uuid, description="Unique identifier")
    indicators_integrated: List[str] = Field(
        default_factory=list, description="PAI indicators integrated in decisions"
    )
    integration_count: int = Field(
        default=0, description="Number of indicators actively integrated"
    )
    total_mandatory: int = Field(default=18, description="Total mandatory indicators")
    integration_ratio_pct: float = Field(
        default=0.0, description="Integration ratio (%)"
    )
    additional_env_selected: List[str] = Field(
        default_factory=list, description="Additional Table 2 indicators selected"
    )
    additional_social_selected: List[str] = Field(
        default_factory=list, description="Additional Table 3 indicators selected"
    )
    decision_examples: List[Dict[str, str]] = Field(
        default_factory=list, description="Examples of PAI-informed decisions"
    )
    assessment_notes: str = Field(default="", description="Assessment notes")
    calculated_at: datetime = Field(default_factory=utcnow, description="Timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 hash")

class ActionPlanItem(BaseModel):
    """A single action item in a PAI action plan."""
    indicator_id: str = Field(description="Related PAI indicator")
    action: str = Field(description="Recommended action")
    priority: str = Field(default="medium", description="Priority: high/medium/low")
    timeline: str = Field(default="", description="Implementation timeline")
    expected_impact: str = Field(default="", description="Expected impact")
    status: str = Field(default="proposed", description="Status")

class ActionPlan(BaseModel):
    """Action plan for addressing adverse PAI impacts."""
    plan_id: str = Field(default_factory=_new_uuid, description="Unique identifier")
    reporting_period: str = Field(default="", description="Reporting period")
    items: List[ActionPlanItem] = Field(
        default_factory=list, description="Action plan items"
    )
    total_actions: int = Field(default=0, description="Total actions")
    high_priority_count: int = Field(default=0, description="High priority actions")
    indicators_addressed: List[str] = Field(
        default_factory=list, description="PAI indicators addressed"
    )
    generated_at: datetime = Field(default_factory=utcnow, description="Timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 hash")

class DataQualityReport(BaseModel):
    """Data quality assessment report for PAI indicators."""
    report_id: str = Field(default_factory=_new_uuid, description="Unique identifier")
    overall_quality_pct: float = Field(
        default=0.0, description="Overall data quality (%)"
    )
    meets_minimum_threshold: bool = Field(
        default=False, description="Meets 70% minimum quality"
    )
    minimum_threshold_pct: float = Field(
        default=70.0, description="Minimum quality threshold (%)"
    )
    by_indicator: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Quality by indicator"
    )
    by_quality_level: Dict[str, int] = Field(
        default_factory=dict, description="Count by quality level"
    )
    improvement_recommendations: List[str] = Field(
        default_factory=list, description="Data quality improvement recommendations"
    )
    total_data_points: int = Field(default=0, description="Total data points assessed")
    sufficient_data_points: int = Field(
        default=0, description="Data points meeting threshold"
    )
    generated_at: datetime = Field(default_factory=utcnow, description="Timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 hash")

class AdditionalPAIResult(BaseModel):
    """Result for additional Table 2/3 PAI indicators."""
    table: str = Field(description="Table 2 or Table 3")
    indicators: List[PAISingleResult] = Field(
        default_factory=list, description="Indicator results"
    )
    total_indicators: int = Field(default=0, description="Total indicators calculated")
    average_coverage_pct: float = Field(
        default=0.0, description="Average data coverage (%)"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

class PAIMandatoryResult(BaseModel):
    """Complete PAI mandatory calculation result for Article 9 products.

    Contains all 18 mandatory indicators, additional Table 2/3 indicators,
    integration assessment, action plan, and data quality report.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Unique result identifier")
    status: PAIMandatoryStatus = Field(description="Overall PAI status")
    # Mandatory indicators (Table 1)
    mandatory_indicators: List[PAISingleResult] = Field(
        default_factory=list, description="All 18 mandatory PAI results"
    )
    mandatory_coverage_pct: float = Field(
        default=0.0, description="Average coverage across mandatory indicators (%)"
    )
    # Additional indicators
    additional_environmental: Optional[AdditionalPAIResult] = Field(
        default=None, description="Table 2 additional environmental indicators"
    )
    additional_social: Optional[AdditionalPAIResult] = Field(
        default=None, description="Table 3 additional social indicators"
    )
    # Integration and action plan
    integration: Optional[IntegrationAssessment] = Field(
        default=None, description="PAI integration assessment"
    )
    action_plan: Optional[ActionPlan] = Field(
        default=None, description="Action plan for adverse impacts"
    )
    data_quality: Optional[DataQualityReport] = Field(
        default=None, description="Data quality report"
    )
    # Summary
    total_nav_eur: float = Field(default=0.0, description="Total NAV (EUR)")
    total_holdings: int = Field(default=0, description="Total holdings")
    reporting_period: str = Field(default="", description="Reporting period")
    generated_at: datetime = Field(default_factory=utcnow, description="Timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------

class PAIMandatoryConfig(BaseModel):
    """Configuration for the PAIMandatoryEngine."""
    total_nav_eur: float = Field(
        default=0.0, gt=-1.0, description="Total NAV of the fund (EUR)"
    )
    reporting_period_start: Optional[datetime] = Field(
        default=None, description="Reporting period start"
    )
    reporting_period_end: Optional[datetime] = Field(
        default=None, description="Reporting period end"
    )
    min_data_quality_pct: float = Field(
        default=70.0, description="Minimum data quality threshold (%)"
    )
    coverage_threshold_pct: float = Field(
        default=50.0, description="Minimum data coverage for valid results (%)"
    )
    include_scope_3: bool = Field(
        default=True, description="Include Scope 3 in GHG totals"
    )
    selected_table2_indicators: List[str] = Field(
        default_factory=lambda: [
            PAIIndicatorId.T2_INORGANIC_POLLUTANTS.value,
            PAIIndicatorId.T2_AIR_POLLUTANTS.value,
            PAIIndicatorId.T2_OZONE_DEPLETING.value,
            PAIIndicatorId.T2_NO_CARBON_INITIATIVES.value,
            PAIIndicatorId.T2_WATER_USAGE.value,
        ],
        description="Selected Table 2 environmental indicators",
    )
    selected_table3_indicators: List[str] = Field(
        default_factory=lambda: [
            PAIIndicatorId.T3_NO_ACCIDENT_PREVENTION.value,
            PAIIndicatorId.T3_ACCIDENT_RATE.value,
            PAIIndicatorId.T3_NO_SUPPLIER_CODE.value,
            PAIIndicatorId.T3_NO_GRIEVANCE.value,
            PAIIndicatorId.T3_NO_WHISTLEBLOWER.value,
        ],
        description="Selected Table 3 social indicators",
    )
    generate_action_plans: bool = Field(
        default=True, description="Auto-generate action plans"
    )
    high_impact_nace_sectors: List[str] = Field(
        default_factory=lambda: ["A", "B", "C", "D", "E", "F", "G", "H", "L"],
        description="NACE sectors for PAI 6",
    )

# ---------------------------------------------------------------------------
# Pydantic model_rebuild
# ---------------------------------------------------------------------------

PAIMandatoryConfig.model_rebuild()
InvesteeFullData.model_rebuild()
PAISingleResult.model_rebuild()
IntegrationAssessment.model_rebuild()
ActionPlanItem.model_rebuild()
ActionPlan.model_rebuild()
DataQualityReport.model_rebuild()
AdditionalPAIResult.model_rebuild()
PAIMandatoryResult.model_rebuild()

# ---------------------------------------------------------------------------
# PAIMandatoryEngine
# ---------------------------------------------------------------------------

class PAIMandatoryEngine:
    """
    Mandatory PAI indicator engine for SFDR Article 9 products.

    Calculates all 18 mandatory PAI indicators (Table 1) plus additional
    environmental (Table 2) and social (Table 3) indicators. Provides
    integration assessment, action plan generation, and data quality
    reporting as required for Article 9 disclosures.

    Attributes:
        config: Engine configuration parameters.
        _holdings: Stored investee data.
        _total_nav: Total net asset value.

    Example:
        >>> engine = PAIMandatoryEngine({"total_nav_eur": 500_000_000})
        >>> holdings = [InvesteeFullData(
        ...     investee_name="Corp A",
        ...     value_eur=25_000_000,
        ...     scope1_tco2eq=5000,
        ...     scope2_tco2eq=2000,
        ...     enterprise_value_eur=200_000_000,
        ...     revenue_eur=50_000_000,
        ... )]
        >>> result = engine.calculate_all(holdings)
        >>> print(f"Status: {result.status.value}")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PAIMandatoryEngine.

        Args:
            config: Optional configuration dictionary.
        """
        if config and isinstance(config, dict):
            self.config = PAIMandatoryConfig(**config)
        elif config and isinstance(config, PAIMandatoryConfig):
            self.config = config
        else:
            self.config = PAIMandatoryConfig()

        self._holdings: List[InvesteeFullData] = []
        self._total_nav: float = self.config.total_nav_eur

        logger.info(
            "PAIMandatoryEngine initialized (version=%s, nav=%.0f)",
            _MODULE_VERSION,
            self._total_nav,
        )

    # ------------------------------------------------------------------
    # Full Calculation
    # ------------------------------------------------------------------

    def calculate_all(
        self,
        holdings: List[InvesteeFullData],
    ) -> PAIMandatoryResult:
        """Calculate all mandatory and additional PAI indicators.

        Args:
            holdings: List of investee data for all portfolio holdings.

        Returns:
            PAIMandatoryResult with complete PAI assessment.
        """
        start = utcnow()
        self._holdings = holdings
        self._ensure_weights(holdings)

        if self._total_nav <= 0.0:
            self._total_nav = sum(h.value_eur for h in holdings)

        # Calculate all 18 mandatory indicators
        mandatory = self._calculate_mandatory(holdings)

        # Calculate additional indicators
        additional_env = self._calculate_additional_environmental(holdings)
        additional_soc = self._calculate_additional_social(holdings)

        # Assess data quality
        all_indicators = mandatory + additional_env.indicators + additional_soc.indicators
        quality_report = self._assess_data_quality(all_indicators, holdings)

        # Generate action plan
        action_plan: Optional[ActionPlan] = None
        if self.config.generate_action_plans:
            action_plan = self._generate_action_plan(mandatory, additional_env, additional_soc)

        # Calculate coverage
        mandatory_coverage = self._average_coverage(mandatory)

        # Determine status
        if quality_report.meets_minimum_threshold and mandatory_coverage >= self.config.coverage_threshold_pct:
            status = PAIMandatoryStatus.COMPLIANT
        elif mandatory_coverage >= 30.0:
            status = PAIMandatoryStatus.PARTIAL
        elif mandatory_coverage < 10.0:
            status = PAIMandatoryStatus.INSUFFICIENT_DATA
        else:
            status = PAIMandatoryStatus.NON_COMPLIANT

        period_str = ""
        if self.config.reporting_period_start and self.config.reporting_period_end:
            period_str = (
                f"{self.config.reporting_period_start.strftime('%Y-%m-%d')} to "
                f"{self.config.reporting_period_end.strftime('%Y-%m-%d')}"
            )

        result = PAIMandatoryResult(
            status=status,
            mandatory_indicators=mandatory,
            mandatory_coverage_pct=round(mandatory_coverage, 2),
            additional_environmental=additional_env,
            additional_social=additional_soc,
            action_plan=action_plan,
            data_quality=quality_report,
            total_nav_eur=round(self._total_nav, 2),
            total_holdings=len(holdings),
            reporting_period=period_str,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "PAI mandatory calculation complete: status=%s, coverage=%.1f%%, "
            "%d indicators, %d holdings in %dms",
            status.value,
            mandatory_coverage,
            len(all_indicators),
            len(holdings),
            int((utcnow() - start).total_seconds() * 1000),
        )
        return result

    # ------------------------------------------------------------------
    # Mandatory PAI Indicators (Table 1)
    # ------------------------------------------------------------------

    def _calculate_mandatory(
        self, holdings: List[InvesteeFullData]
    ) -> List[PAISingleResult]:
        """Calculate all 18 mandatory PAI indicators.

        Args:
            holdings: Investee data.

        Returns:
            List of 18 PAISingleResult objects.
        """
        results: List[PAISingleResult] = []

        results.append(self._calc_pai_1(holdings))
        results.append(self._calc_pai_2(holdings))
        results.append(self._calc_pai_3(holdings))
        results.append(self._calc_pai_4(holdings))
        results.append(self._calc_pai_5(holdings))
        results.append(self._calc_pai_6(holdings))
        results.append(self._calc_pai_7(holdings))
        results.append(self._calc_pai_8(holdings))
        results.append(self._calc_pai_9(holdings))
        results.append(self._calc_pai_10(holdings))
        results.append(self._calc_pai_11(holdings))
        results.append(self._calc_pai_12(holdings))
        results.append(self._calc_pai_13(holdings))
        results.append(self._calc_pai_14(holdings))
        results.append(self._calc_pai_15(holdings))
        results.append(self._calc_pai_16(holdings))
        results.append(self._calc_pai_17(holdings))
        results.append(self._calc_pai_18(holdings))

        return results

    def _calc_pai_1(self, holdings: List[InvesteeFullData]) -> PAISingleResult:
        """PAI 1: GHG Emissions (Scope 1 + 2 + 3 + Total).

        Formula: SUM(attribution_factor_i * ghg_i)
            attribution_factor = value_i / evic_i
        """
        scope1_total = 0.0
        scope2_total = 0.0
        scope3_total = 0.0
        covered = 0
        quality_dist: Dict[str, int] = defaultdict(int)

        for h in self._corporate_holdings(holdings):
            evic = h.enterprise_value_eur
            if evic <= 0.0:
                continue
            attr = h.value_eur / evic
            if h.scope1_tco2eq > 0 or h.scope2_tco2eq > 0 or h.scope3_tco2eq > 0:
                scope1_total += attr * h.scope1_tco2eq
                scope2_total += attr * h.scope2_tco2eq
                if self.config.include_scope_3:
                    scope3_total += attr * h.scope3_tco2eq
                covered += 1
                quality_dist[h.data_quality.value] += 1

        total = scope1_total + scope2_total + scope3_total
        return self._build_result(
            PAIIndicatorId.PAI_1.value,
            total,
            holdings,
            covered,
            quality_dist,
            sub_values={
                "scope_1": round(scope1_total, 2),
                "scope_2": round(scope2_total, 2),
                "scope_3": round(scope3_total, 2),
                "total": round(total, 2),
            },
            note="Financed GHG emissions using attribution factor (value/EVIC)",
        )

    def _calc_pai_2(self, holdings: List[InvesteeFullData]) -> PAISingleResult:
        """PAI 2: Carbon Footprint = PAI_1_total / portfolio_value_EUR_M."""
        pai1 = self._calc_pai_1(holdings)
        nav_m = self._total_nav / 1_000_000.0 if self._total_nav > 0 else 1.0
        value = _safe_divide(pai1.value or 0.0, nav_m)

        return self._build_result(
            PAIIndicatorId.PAI_2.value,
            value,
            holdings,
            pai1.holdings_with_data,
            {},
            note="Total financed GHG per EUR M invested",
        )

    def _calc_pai_3(self, holdings: List[InvesteeFullData]) -> PAISingleResult:
        """PAI 3: GHG Intensity (WACI) = SUM(weight_i * (ghg_i / revenue_i))."""
        waci = 0.0
        covered = 0
        quality_dist: Dict[str, int] = defaultdict(int)

        for h in self._corporate_holdings(holdings):
            rev_m = h.revenue_eur / 1_000_000.0 if h.revenue_eur > 0 else 0.0
            ghg = h.total_ghg_tco2eq
            if rev_m <= 0.0 or ghg <= 0.0:
                continue
            intensity = ghg / rev_m
            weight = h.weight_pct / 100.0
            waci += weight * intensity
            covered += 1
            quality_dist[h.data_quality.value] += 1

        return self._build_result(
            PAIIndicatorId.PAI_3.value,
            waci,
            holdings,
            covered,
            quality_dist,
            note="Weighted average carbon intensity (WACI)",
        )

    def _calc_pai_4(self, holdings: List[InvesteeFullData]) -> PAISingleResult:
        """PAI 4: Exposure to fossil fuel companies (%)."""
        return self._calc_share_indicator(
            PAIIndicatorId.PAI_4.value,
            holdings,
            lambda h: h.is_fossil_fuel_company is True,
            lambda h: h.is_fossil_fuel_company is not None,
        )

    def _calc_pai_5(self, holdings: List[InvesteeFullData]) -> PAISingleResult:
        """PAI 5: Non-renewable energy share (weighted average %)."""
        return self._calc_weighted_avg(
            PAIIndicatorId.PAI_5.value,
            holdings,
            lambda h: h.non_renewable_energy_share_pct,
        )

    def _calc_pai_6(self, holdings: List[InvesteeFullData]) -> PAISingleResult:
        """PAI 6: Energy consumption intensity per high-impact NACE sector."""
        sector_intensities: Dict[str, List[float]] = defaultdict(list)
        covered = 0
        quality_dist: Dict[str, int] = defaultdict(int)

        for h in self._corporate_holdings(holdings):
            if h.nace_sector not in self.config.high_impact_nace_sectors:
                continue
            if h.energy_consumption_gwh is None or h.revenue_eur <= 0:
                continue
            rev_m = h.revenue_eur / 1_000_000.0
            intensity = h.energy_consumption_gwh / rev_m
            sector_intensities[h.nace_sector].append(intensity)
            covered += 1
            quality_dist[h.data_quality.value] += 1

        sector_avg: Dict[str, float] = {}
        for sector, values in sector_intensities.items():
            sector_avg[sector] = round(sum(values) / len(values), 4)

        overall = 0.0
        if sector_avg:
            overall = sum(sector_avg.values()) / len(sector_avg)

        return self._build_result(
            PAIIndicatorId.PAI_6.value,
            overall,
            holdings,
            covered,
            quality_dist,
            sub_values={"by_sector": sector_avg},
            note="Energy intensity per NACE high-impact sector",
        )

    def _calc_pai_7(self, holdings: List[InvesteeFullData]) -> PAISingleResult:
        """PAI 7: Activities affecting biodiversity-sensitive areas (%)."""
        return self._calc_share_indicator(
            PAIIndicatorId.PAI_7.value,
            holdings,
            lambda h: h.affects_biodiversity_area is True,
            lambda h: h.affects_biodiversity_area is not None,
        )

    def _calc_pai_8(self, holdings: List[InvesteeFullData]) -> PAISingleResult:
        """PAI 8: Emissions to water (attribution-weighted tonnes)."""
        return self._calc_attribution_indicator(
            PAIIndicatorId.PAI_8.value,
            holdings,
            lambda h: h.emissions_to_water_tonnes,
        )

    def _calc_pai_9(self, holdings: List[InvesteeFullData]) -> PAISingleResult:
        """PAI 9: Hazardous waste + radioactive waste ratio (tonnes)."""
        total = 0.0
        covered = 0
        quality_dist: Dict[str, int] = defaultdict(int)

        for h in self._corporate_holdings(holdings):
            evic = h.enterprise_value_eur
            if evic <= 0.0:
                continue
            hw = h.hazardous_waste_tonnes or 0.0
            rw = h.radioactive_waste_tonnes or 0.0
            if hw > 0 or rw > 0:
                attr = h.value_eur / evic
                total += attr * (hw + rw)
                covered += 1
                quality_dist[h.data_quality.value] += 1

        return self._build_result(
            PAIIndicatorId.PAI_9.value,
            total,
            holdings,
            covered,
            quality_dist,
            note="Attribution-weighted hazardous + radioactive waste",
        )

    def _calc_pai_10(self, holdings: List[InvesteeFullData]) -> PAISingleResult:
        """PAI 10: UNGC/OECD violations (%)."""
        return self._calc_share_indicator(
            PAIIndicatorId.PAI_10.value,
            holdings,
            lambda h: h.has_ungc_violations is True,
            lambda h: h.has_ungc_violations is not None,
        )

    def _calc_pai_11(self, holdings: List[InvesteeFullData]) -> PAISingleResult:
        """PAI 11: Lack of UNGC/OECD compliance processes (%)."""
        return self._calc_share_indicator(
            PAIIndicatorId.PAI_11.value,
            holdings,
            lambda h: h.has_compliance_mechanisms is False,
            lambda h: h.has_compliance_mechanisms is not None,
        )

    def _calc_pai_12(self, holdings: List[InvesteeFullData]) -> PAISingleResult:
        """PAI 12: Unadjusted gender pay gap (weighted average %)."""
        return self._calc_weighted_avg(
            PAIIndicatorId.PAI_12.value,
            holdings,
            lambda h: h.gender_pay_gap_pct,
        )

    def _calc_pai_13(self, holdings: List[InvesteeFullData]) -> PAISingleResult:
        """PAI 13: Board gender diversity (weighted average %)."""
        return self._calc_weighted_avg(
            PAIIndicatorId.PAI_13.value,
            holdings,
            lambda h: h.female_board_pct,
        )

    def _calc_pai_14(self, holdings: List[InvesteeFullData]) -> PAISingleResult:
        """PAI 14: Exposure to controversial weapons (%)."""
        return self._calc_share_indicator(
            PAIIndicatorId.PAI_14.value,
            holdings,
            lambda h: h.involved_controversial_weapons is True,
            lambda h: h.involved_controversial_weapons is not None,
        )

    def _calc_pai_15(self, holdings: List[InvesteeFullData]) -> PAISingleResult:
        """PAI 15: GHG intensity of investee countries (weighted avg)."""
        total_weighted = 0.0
        covered = 0
        covered_weight = 0.0
        quality_dist: Dict[str, int] = defaultdict(int)

        for h in holdings:
            if h.investee_type != "SOVEREIGN":
                continue
            if h.country_ghg_intensity is None:
                continue
            weight = h.weight_pct / 100.0
            total_weighted += weight * h.country_ghg_intensity
            covered_weight += weight
            covered += 1
            quality_dist[h.data_quality.value] += 1

        value = _safe_divide(total_weighted, covered_weight) if covered_weight > 0 else 0.0

        return self._build_result(
            PAIIndicatorId.PAI_15.value,
            value,
            holdings,
            covered,
            quality_dist,
            note="Weighted average country GHG intensity (sovereign holdings only)",
        )

    def _calc_pai_16(self, holdings: List[InvesteeFullData]) -> PAISingleResult:
        """PAI 16: Countries subject to social violations (%)."""
        exposed_value = 0.0
        total_sov_value = 0.0
        covered = 0
        quality_dist: Dict[str, int] = defaultdict(int)

        for h in holdings:
            if h.investee_type != "SOVEREIGN":
                continue
            total_sov_value += h.value_eur
            if h.country_social_violations is None:
                continue
            covered += 1
            quality_dist[h.data_quality.value] += 1
            if h.country_social_violations:
                exposed_value += h.value_eur

        value = _safe_pct(exposed_value, total_sov_value)
        sov_count = sum(1 for h in holdings if h.investee_type == "SOVEREIGN")

        return self._build_result(
            PAIIndicatorId.PAI_16.value,
            value,
            [h for h in holdings if h.investee_type == "SOVEREIGN"],
            covered,
            quality_dist,
            note="Share of sovereign investments in countries with social violations",
        )

    def _calc_pai_17(self, holdings: List[InvesteeFullData]) -> PAISingleResult:
        """PAI 17: Fossil fuels through real estate assets (%)."""
        exposed_value = 0.0
        total_re_value = 0.0
        covered = 0
        quality_dist: Dict[str, int] = defaultdict(int)

        for h in holdings:
            if h.investee_type != "REAL_ESTATE":
                continue
            total_re_value += h.value_eur
            if h.re_fossil_fuel_involved is None:
                continue
            covered += 1
            quality_dist[h.data_quality.value] += 1
            if h.re_fossil_fuel_involved:
                exposed_value += h.value_eur

        value = _safe_pct(exposed_value, total_re_value)

        return self._build_result(
            PAIIndicatorId.PAI_17.value,
            value,
            [h for h in holdings if h.investee_type == "REAL_ESTATE"],
            covered,
            quality_dist,
            note="Share of real estate involved in fossil fuels",
        )

    def _calc_pai_18(self, holdings: List[InvesteeFullData]) -> PAISingleResult:
        """PAI 18: Energy-inefficient real estate assets (%)."""
        exposed_value = 0.0
        total_re_value = 0.0
        covered = 0
        quality_dist: Dict[str, int] = defaultdict(int)

        for h in holdings:
            if h.investee_type != "REAL_ESTATE":
                continue
            total_re_value += h.value_eur
            if h.re_energy_inefficient is None:
                continue
            covered += 1
            quality_dist[h.data_quality.value] += 1
            if h.re_energy_inefficient:
                exposed_value += h.value_eur

        value = _safe_pct(exposed_value, total_re_value)

        return self._build_result(
            PAIIndicatorId.PAI_18.value,
            value,
            [h for h in holdings if h.investee_type == "REAL_ESTATE"],
            covered,
            quality_dist,
            note="Share of energy-inefficient real estate (below NZEB)",
        )

    # ------------------------------------------------------------------
    # Additional Indicators (Table 2 - Environmental)
    # ------------------------------------------------------------------

    def _calculate_additional_environmental(
        self, holdings: List[InvesteeFullData]
    ) -> AdditionalPAIResult:
        """Calculate selected Table 2 additional environmental indicators.

        Args:
            holdings: Investee data.

        Returns:
            AdditionalPAIResult with Table 2 indicator results.
        """
        selected = self.config.selected_table2_indicators
        results: List[PAISingleResult] = []

        if PAIIndicatorId.T2_INORGANIC_POLLUTANTS.value in selected:
            results.append(self._calc_attribution_indicator(
                PAIIndicatorId.T2_INORGANIC_POLLUTANTS.value,
                holdings,
                lambda h: h.inorganic_pollutants_tonnes,
            ))

        if PAIIndicatorId.T2_AIR_POLLUTANTS.value in selected:
            results.append(self._calc_attribution_indicator(
                PAIIndicatorId.T2_AIR_POLLUTANTS.value,
                holdings,
                lambda h: h.air_pollutants_tonnes,
            ))

        if PAIIndicatorId.T2_OZONE_DEPLETING.value in selected:
            results.append(self._calc_attribution_indicator(
                PAIIndicatorId.T2_OZONE_DEPLETING.value,
                holdings,
                lambda h: h.ozone_depleting_tonnes,
            ))

        if PAIIndicatorId.T2_NO_CARBON_INITIATIVES.value in selected:
            results.append(self._calc_share_indicator(
                PAIIndicatorId.T2_NO_CARBON_INITIATIVES.value,
                holdings,
                lambda h: h.has_carbon_initiatives is False,
                lambda h: h.has_carbon_initiatives is not None,
            ))

        if PAIIndicatorId.T2_WATER_USAGE.value in selected:
            results.append(self._calc_weighted_avg(
                PAIIndicatorId.T2_WATER_USAGE.value,
                holdings,
                lambda h: (
                    h.water_usage_m3 / (h.revenue_eur / 1_000_000.0)
                    if h.water_usage_m3 is not None and h.revenue_eur > 0
                    else None
                ),
            ))

        avg_coverage = self._average_coverage(results)

        additional = AdditionalPAIResult(
            table="Table 2",
            indicators=results,
            total_indicators=len(results),
            average_coverage_pct=round(avg_coverage, 2),
        )
        additional.provenance_hash = _compute_hash(additional)
        return additional

    # ------------------------------------------------------------------
    # Additional Indicators (Table 3 - Social)
    # ------------------------------------------------------------------

    def _calculate_additional_social(
        self, holdings: List[InvesteeFullData]
    ) -> AdditionalPAIResult:
        """Calculate selected Table 3 additional social indicators.

        Args:
            holdings: Investee data.

        Returns:
            AdditionalPAIResult with Table 3 indicator results.
        """
        selected = self.config.selected_table3_indicators
        results: List[PAISingleResult] = []

        if PAIIndicatorId.T3_NO_ACCIDENT_PREVENTION.value in selected:
            results.append(self._calc_share_indicator(
                PAIIndicatorId.T3_NO_ACCIDENT_PREVENTION.value,
                holdings,
                lambda h: h.has_accident_prevention is False,
                lambda h: h.has_accident_prevention is not None,
            ))

        if PAIIndicatorId.T3_ACCIDENT_RATE.value in selected:
            results.append(self._calc_weighted_avg(
                PAIIndicatorId.T3_ACCIDENT_RATE.value,
                holdings,
                lambda h: h.accident_rate,
            ))

        if PAIIndicatorId.T3_NO_SUPPLIER_CODE.value in selected:
            results.append(self._calc_share_indicator(
                PAIIndicatorId.T3_NO_SUPPLIER_CODE.value,
                holdings,
                lambda h: h.has_supplier_code is False,
                lambda h: h.has_supplier_code is not None,
            ))

        if PAIIndicatorId.T3_NO_GRIEVANCE.value in selected:
            results.append(self._calc_share_indicator(
                PAIIndicatorId.T3_NO_GRIEVANCE.value,
                holdings,
                lambda h: h.has_grievance_mechanism is False,
                lambda h: h.has_grievance_mechanism is not None,
            ))

        if PAIIndicatorId.T3_NO_WHISTLEBLOWER.value in selected:
            results.append(self._calc_share_indicator(
                PAIIndicatorId.T3_NO_WHISTLEBLOWER.value,
                holdings,
                lambda h: h.has_whistleblower_protection is False,
                lambda h: h.has_whistleblower_protection is not None,
            ))

        avg_coverage = self._average_coverage(results)

        additional = AdditionalPAIResult(
            table="Table 3",
            indicators=results,
            total_indicators=len(results),
            average_coverage_pct=round(avg_coverage, 2),
        )
        additional.provenance_hash = _compute_hash(additional)
        return additional

    # ------------------------------------------------------------------
    # Integration Assessment
    # ------------------------------------------------------------------

    def assess_integration(
        self,
        integrated_indicators: List[str],
        decision_examples: Optional[List[Dict[str, str]]] = None,
    ) -> IntegrationAssessment:
        """Assess PAI integration in investment decision-making.

        Args:
            integrated_indicators: List of PAI indicator IDs actively integrated.
            decision_examples: Optional examples of PAI-informed decisions.

        Returns:
            IntegrationAssessment with integration ratio.
        """
        mandatory_ids = [f"PAI_{i}" for i in range(1, 19)]
        integrated_mandatory = [i for i in integrated_indicators if i in mandatory_ids]
        additional_env = [i for i in integrated_indicators if i.startswith("T2_")]
        additional_soc = [i for i in integrated_indicators if i.startswith("T3_")]

        ratio = _safe_pct(len(integrated_mandatory), 18)

        assessment = IntegrationAssessment(
            indicators_integrated=integrated_indicators,
            integration_count=len(integrated_mandatory),
            total_mandatory=18,
            integration_ratio_pct=round(ratio, 2),
            additional_env_selected=additional_env,
            additional_social_selected=additional_soc,
            decision_examples=decision_examples or [],
            assessment_notes=(
                f"{len(integrated_mandatory)}/18 mandatory PAI indicators integrated. "
                f"{len(additional_env)} Table 2 and {len(additional_soc)} Table 3 "
                f"indicators also considered."
            ),
        )
        assessment.provenance_hash = _compute_hash(assessment)
        return assessment

    # ------------------------------------------------------------------
    # Action Plan Generation
    # ------------------------------------------------------------------

    def _generate_action_plan(
        self,
        mandatory: List[PAISingleResult],
        additional_env: AdditionalPAIResult,
        additional_soc: AdditionalPAIResult,
    ) -> ActionPlan:
        """Generate action plan for adverse PAI impacts.

        Identifies indicators with concerning values and generates
        recommended actions.

        Args:
            mandatory: Mandatory indicator results.
            additional_env: Additional environmental results.
            additional_soc: Additional social results.

        Returns:
            ActionPlan with prioritized action items.
        """
        items: List[ActionPlanItem] = []
        all_indicators = mandatory + additional_env.indicators + additional_soc.indicators

        for ind in all_indicators:
            # Low coverage -> recommend data improvement
            if ind.coverage_pct < self.config.min_data_quality_pct:
                items.append(ActionPlanItem(
                    indicator_id=ind.indicator_id,
                    action=f"Improve data coverage for {ind.indicator_name} "
                           f"(current: {ind.coverage_pct:.0f}%, target: "
                           f"{self.config.min_data_quality_pct:.0f}%)",
                    priority="high" if ind.coverage_pct < 30.0 else "medium",
                    timeline="Next reporting period",
                    expected_impact="Better PAI disclosure quality and compliance",
                ))

            # Specific adverse impact actions
            if ind.indicator_id == PAIIndicatorId.PAI_4.value and ind.value and ind.value > 5.0:
                items.append(ActionPlanItem(
                    indicator_id=ind.indicator_id,
                    action="Reduce fossil fuel exposure through portfolio transition",
                    priority="high",
                    timeline="12 months",
                    expected_impact=f"Reduce fossil fuel share from {ind.value:.1f}%",
                ))

            if ind.indicator_id == PAIIndicatorId.PAI_10.value and ind.value and ind.value > 0.0:
                items.append(ActionPlanItem(
                    indicator_id=ind.indicator_id,
                    action="Engage with companies having UNGC/OECD violations",
                    priority="high",
                    timeline="6 months",
                    expected_impact="Eliminate UNGC/OECD violations in portfolio",
                ))

            if ind.indicator_id == PAIIndicatorId.PAI_14.value and ind.value and ind.value > 0.0:
                items.append(ActionPlanItem(
                    indicator_id=ind.indicator_id,
                    action="Divest from controversial weapons manufacturers",
                    priority="high",
                    timeline="Immediate",
                    expected_impact="Zero exposure to controversial weapons",
                ))

        high_count = sum(1 for i in items if i.priority == "high")
        addressed = list(set(i.indicator_id for i in items))

        plan = ActionPlan(
            items=items,
            total_actions=len(items),
            high_priority_count=high_count,
            indicators_addressed=addressed,
        )
        plan.provenance_hash = _compute_hash(plan)
        return plan

    # ------------------------------------------------------------------
    # Data Quality Assessment
    # ------------------------------------------------------------------

    def _assess_data_quality(
        self,
        indicators: List[PAISingleResult],
        holdings: List[InvesteeFullData],
    ) -> DataQualityReport:
        """Assess data quality across all PAI indicators.

        Args:
            indicators: All calculated indicator results.
            holdings: Investee data.

        Returns:
            DataQualityReport with quality assessment.
        """
        by_indicator: Dict[str, Dict[str, Any]] = {}
        total_points = 0
        sufficient_points = 0
        quality_counts: Dict[str, int] = defaultdict(int)

        for ind in indicators:
            coverage = ind.coverage_pct
            sufficient = coverage >= self.config.min_data_quality_pct
            by_indicator[ind.indicator_id] = {
                "coverage_pct": round(coverage, 2),
                "sufficient": sufficient,
                "holdings_with_data": ind.holdings_with_data,
                "total_holdings": ind.total_holdings,
            }
            total_points += 1
            if sufficient:
                sufficient_points += 1

        # Count quality levels from holdings
        for h in holdings:
            quality_counts[h.data_quality.value] += 1

        overall = _safe_pct(sufficient_points, total_points)
        meets_threshold = overall >= self.config.min_data_quality_pct

        recommendations: List[str] = []
        if not meets_threshold:
            recommendations.append(
                f"Overall data quality ({overall:.0f}%) is below the "
                f"{self.config.min_data_quality_pct:.0f}% threshold. "
                f"Prioritize data collection for indicators with lowest coverage."
            )

        low_coverage = [
            ind_id for ind_id, info in by_indicator.items()
            if not info["sufficient"]
        ]
        if low_coverage:
            recommendations.append(
                f"{len(low_coverage)} indicators have insufficient data coverage: "
                f"{', '.join(low_coverage[:5])}."
            )

        not_avail_count = quality_counts.get(DataQualityLevel.NOT_AVAILABLE.value, 0)
        if not_avail_count > len(holdings) * 0.3:
            recommendations.append(
                f"{not_avail_count} holdings lack quality classification. "
                f"Request data quality flags from data providers."
            )

        report = DataQualityReport(
            overall_quality_pct=round(overall, 2),
            meets_minimum_threshold=meets_threshold,
            minimum_threshold_pct=self.config.min_data_quality_pct,
            by_indicator=by_indicator,
            by_quality_level=dict(quality_counts),
            improvement_recommendations=recommendations,
            total_data_points=total_points,
            sufficient_data_points=sufficient_points,
        )
        report.provenance_hash = _compute_hash(report)
        return report

    # ------------------------------------------------------------------
    # Generic Calculation Helpers
    # ------------------------------------------------------------------

    def _calc_share_indicator(
        self,
        indicator_id: str,
        holdings: List[InvesteeFullData],
        is_exposed_fn: Any,
        has_data_fn: Any,
    ) -> PAISingleResult:
        """Calculate a share-type PAI indicator.

        Share = SUM(value_exposed) / total_nav * 100

        Args:
            indicator_id: PAI indicator identifier.
            holdings: Investee data.
            is_exposed_fn: Function returning True if holding is exposed.
            has_data_fn: Function returning True if holding has data.

        Returns:
            PAISingleResult with share value.
        """
        exposed_value = 0.0
        covered = 0
        covered_value = 0.0
        quality_dist: Dict[str, int] = defaultdict(int)

        corp_holdings = self._corporate_holdings(holdings)
        for h in corp_holdings:
            if has_data_fn(h):
                covered += 1
                covered_value += h.value_eur
                quality_dist[h.data_quality.value] += 1
                if is_exposed_fn(h):
                    exposed_value += h.value_eur

        nav = self._total_nav if self._total_nav > 0 else sum(h.value_eur for h in corp_holdings)
        value = _safe_pct(exposed_value, nav)

        return self._build_result(
            indicator_id,
            value,
            corp_holdings,
            covered,
            quality_dist,
        )

    def _calc_weighted_avg(
        self,
        indicator_id: str,
        holdings: List[InvesteeFullData],
        value_fn: Any,
    ) -> PAISingleResult:
        """Calculate a weighted-average PAI indicator.

        Weighted Average = SUM(weight_i * metric_i) / SUM(weight_covered)

        Args:
            indicator_id: PAI indicator identifier.
            holdings: Investee data.
            value_fn: Function extracting the metric from a holding.

        Returns:
            PAISingleResult with weighted average value.
        """
        total_weighted = 0.0
        covered_weight = 0.0
        covered = 0
        quality_dist: Dict[str, int] = defaultdict(int)

        corp_holdings = self._corporate_holdings(holdings)
        for h in corp_holdings:
            val = value_fn(h)
            if val is not None:
                weight = h.weight_pct / 100.0
                total_weighted += weight * val
                covered_weight += weight
                covered += 1
                quality_dist[h.data_quality.value] += 1

        value = _safe_divide(total_weighted, covered_weight) if covered_weight > 0 else 0.0

        return self._build_result(
            indicator_id,
            value,
            corp_holdings,
            covered,
            quality_dist,
        )

    def _calc_attribution_indicator(
        self,
        indicator_id: str,
        holdings: List[InvesteeFullData],
        value_fn: Any,
    ) -> PAISingleResult:
        """Calculate an attribution-type PAI indicator.

        Attribution = SUM(value_i / evic_i * metric_i)

        Args:
            indicator_id: PAI indicator identifier.
            holdings: Investee data.
            value_fn: Function extracting the metric from a holding.

        Returns:
            PAISingleResult with attribution-weighted value.
        """
        total = 0.0
        covered = 0
        quality_dist: Dict[str, int] = defaultdict(int)

        corp_holdings = self._corporate_holdings(holdings)
        for h in corp_holdings:
            val = value_fn(h)
            evic = h.enterprise_value_eur
            if val is not None and val > 0 and evic > 0:
                attr = h.value_eur / evic
                total += attr * val
                covered += 1
                quality_dist[h.data_quality.value] += 1

        return self._build_result(
            indicator_id,
            total,
            corp_holdings,
            covered,
            quality_dist,
        )

    def _build_result(
        self,
        indicator_id: str,
        value: float,
        holdings: List[InvesteeFullData],
        covered: int,
        quality_dist: Dict[str, int],
        sub_values: Optional[Dict[str, Any]] = None,
        note: str = "",
    ) -> PAISingleResult:
        """Build a PAISingleResult with metadata and provenance.

        Args:
            indicator_id: PAI indicator identifier.
            value: Calculated value.
            holdings: Holdings used in calculation.
            covered: Number of holdings with data.
            quality_dist: Data quality distribution.
            sub_values: Optional breakdown values.
            note: Methodology note.

        Returns:
            PAISingleResult with complete metadata.
        """
        meta = PAI_METADATA.get(indicator_id, {})
        total = len(holdings)
        coverage = _safe_pct(covered, total)
        is_sufficient = coverage >= self.config.coverage_threshold_pct

        result = PAISingleResult(
            indicator_id=indicator_id,
            indicator_name=meta.get("name", indicator_id),
            category=meta.get("category", PAICategory.CLIMATE_GHG).value
                if hasattr(meta.get("category", ""), "value")
                else str(meta.get("category", "")),
            table=meta.get("table", "Table 1"),
            value=round(value, 4),
            unit=meta.get("unit", ""),
            sub_values=sub_values or {},
            total_holdings=total,
            holdings_with_data=covered,
            coverage_pct=round(coverage, 2),
            coverage_by_value_pct=round(coverage, 2),
            data_quality_distribution=dict(quality_dist),
            is_sufficient=is_sufficient,
            methodology_note=note,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _corporate_holdings(
        self, holdings: List[InvesteeFullData]
    ) -> List[InvesteeFullData]:
        """Filter to corporate-type holdings only.

        Args:
            holdings: All holdings.

        Returns:
            List of CORPORATE holdings.
        """
        return [h for h in holdings if h.investee_type == "CORPORATE"]

    def _ensure_weights(self, holdings: List[InvesteeFullData]) -> None:
        """Ensure portfolio weights are populated from holding values.

        Args:
            holdings: List of holdings to update.
        """
        total_value = sum(h.value_eur for h in holdings)
        if total_value <= 0:
            return

        for h in holdings:
            if h.weight_pct <= 0.0 and h.value_eur > 0:
                h.weight_pct = (h.value_eur / total_value) * 100.0

    def _average_coverage(self, results: List[PAISingleResult]) -> float:
        """Calculate average coverage across indicator results.

        Args:
            results: List of indicator results.

        Returns:
            Average coverage percentage.
        """
        if not results:
            return 0.0
        total_coverage = sum(r.coverage_pct for r in results)
        return total_coverage / len(results)
