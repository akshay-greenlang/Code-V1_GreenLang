# -*- coding: utf-8 -*-
"""
ClimateRiskEngine - PACK-016 ESRS E1 Climate Engine 8
=======================================================

Assesses anticipated financial effects from material climate risks
and opportunities per ESRS E1-9.

Under the European Sustainability Reporting Standards (ESRS), ESRS E1-9
requires the undertaking to disclose the anticipated financial effects
of material physical risks, transition risks, and potential climate-
related opportunities on its financial position, performance, and cash
flows, quantified over short-, medium-, and long-term time horizons.

ESRS E1-9 Framework:
    - Para 64: The undertaking shall disclose the anticipated financial
      effects of material physical risks and material transition risks
      and the potential climate-related opportunities.
    - Para 65: The disclosure of anticipated financial effects shall
      include: (a) the monetary amount and proportion of assets at
      material physical risk; (b) the monetary amount and proportion
      of assets at material transition risk; (c) the monetary amount
      and proportion of net revenue from activities at material
      physical and transition risk.
    - Para 66: For each material risk, the undertaking shall disclose:
      the nature of the risk, the time horizon, the likelihood, and
      the estimated financial impact.
    - Para 67: For climate-related opportunities, the undertaking
      shall disclose the estimated financial effects (revenue impact,
      investment required).
    - Para 68: The undertaking shall use scenario analysis where
      appropriate, referencing climate scenarios (RCP/SSP, IEA).

Application Requirements (AR E1-74 through AR E1-81):
    - AR E1-74: Physical risks include acute (extreme weather) and
      chronic (gradual climate change).
    - AR E1-75: Transition risks include policy/legal, technology,
      market, and reputation risks.
    - AR E1-76: Financial effects may affect assets, liabilities,
      revenue, operating costs, and access to capital.
    - AR E1-77: Time horizons: short-term (0-3y), medium-term (3-10y),
      long-term (10+ years).
    - AR E1-78: Scenario analysis should cover at least a 1.5C/2C
      aligned scenario and a higher warming scenario (e.g., 4C).

Regulatory References:
    - EU Delegated Regulation 2023/2772 (ESRS Set 1)
    - ESRS E1 Climate Change, Para 64-68
    - ESRS E1 Application Requirements AR E1-74 through AR E1-81
    - TCFD Recommendations (2017) and Implementation Guidance
    - NGFS Climate Scenarios for Central Banks (2024)

Zero-Hallucination:
    - Financial impact uses deterministic multiplication
    - Scenario aggregation uses deterministic summation
    - Time horizon breakdown uses deterministic grouping
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-016 ESRS E1 Climate Change
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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


def _decimal(value: Any) -> Decimal:
    """Convert value to Decimal safely.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _safe_divide(
    numerator: float, denominator: float, default: float = 0.0
) -> float:
    """Safely divide two numbers, returning *default* on zero denominator."""
    if denominator == 0.0:
        return default
    return numerator / denominator


def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    ))


def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))


def _round_val(value: Decimal, places: int = 3) -> Decimal:
    """Round a Decimal value to the specified number of decimal places.

    Uses ROUND_HALF_UP for regulatory consistency.

    Args:
        value: Decimal value to round.
        places: Number of decimal places (default 3).

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PhysicalRiskType(str, Enum):
    """Type of physical climate risk per AR E1-74.

    Physical risks are divided into acute (event-driven) risks and
    chronic (long-term shift) risks arising from climate change.
    """
    ACUTE_FLOODING = "acute_flooding"
    ACUTE_WILDFIRE = "acute_wildfire"
    ACUTE_STORM = "acute_storm"
    ACUTE_HEATWAVE = "acute_heatwave"
    CHRONIC_SEA_LEVEL = "chronic_sea_level"
    CHRONIC_TEMPERATURE = "chronic_temperature"
    CHRONIC_PRECIPITATION = "chronic_precipitation"
    CHRONIC_WATER_STRESS = "chronic_water_stress"


class TransitionRiskType(str, Enum):
    """Type of transition climate risk per AR E1-75.

    Transition risks arise from the transition to a lower-carbon
    economy, including policy, technology, market, and legal shifts.
    """
    POLICY_CARBON_PRICING = "policy_carbon_pricing"
    POLICY_REGULATION = "policy_regulation"
    TECHNOLOGY_DISRUPTION = "technology_disruption"
    MARKET_SHIFT = "market_shift"
    REPUTATION = "reputation"
    LEGAL_LIABILITY = "legal_liability"


class ClimateOpportunityType(str, Enum):
    """Type of climate-related opportunity per AR E1-76.

    Climate-related opportunities arise from resource efficiency
    gains, new energy sources, products/services, markets, and
    resilience improvements.
    """
    RESOURCE_EFFICIENCY = "resource_efficiency"
    ENERGY_SOURCE = "energy_source"
    PRODUCTS_SERVICES = "products_services"
    MARKETS = "markets"
    RESILIENCE = "resilience"


class RiskTimeHorizon(str, Enum):
    """Time horizon for risk assessment per AR E1-77.

    ESRS E1-9 requires assessment over short-, medium-, and
    long-term time horizons.
    """
    SHORT_TERM_0_3Y = "short_term_0_3y"
    MEDIUM_TERM_3_10Y = "medium_term_3_10y"
    LONG_TERM_10_PLUS = "long_term_10_plus"


class ClimateScenario(str, Enum):
    """Climate scenario for risk and opportunity assessment per AR E1-78.

    The undertaking should use at least two scenarios: one aligned
    with 1.5C/2C and one representing higher warming.
    """
    RCP_2_6 = "rcp_2_6"
    RCP_4_5 = "rcp_4_5"
    RCP_8_5 = "rcp_8_5"
    IEA_NZE = "iea_nze"
    IEA_STEPS = "iea_steps"
    NGFS_ORDERLY = "ngfs_orderly"
    NGFS_DISORDERLY = "ngfs_disorderly"
    NGFS_HOT_HOUSE = "ngfs_hot_house"


class LikelihoodLevel(str, Enum):
    """Likelihood level for risk assessment.

    Qualitative likelihood categories with implied probability
    ranges used in climate risk assessment.
    """
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Required ESRS E1-9 data points.
E1_9_DATAPOINTS: Dict[str, str] = {
    "e1_9_dp01": "Monetary amount of assets at material physical risk",
    "e1_9_dp02": "Proportion (%) of assets at material physical risk",
    "e1_9_dp03": "Monetary amount of assets at material transition risk",
    "e1_9_dp04": "Proportion (%) of assets at material transition risk",
    "e1_9_dp05": "Net revenue from activities at material physical risk",
    "e1_9_dp06": "Net revenue from activities at material transition risk",
    "e1_9_dp07": "Description of each material physical risk identified",
    "e1_9_dp08": "Description of each material transition risk identified",
    "e1_9_dp09": "Time horizon for each material risk (short/medium/long-term)",
    "e1_9_dp10": "Likelihood of each material risk",
    "e1_9_dp11": "Estimated financial impact of each material risk",
    "e1_9_dp12": "Description of climate-related opportunities identified",
    "e1_9_dp13": "Estimated financial effects of climate-related opportunities",
    "e1_9_dp14": "Adaptation and mitigation costs for physical risks",
    "e1_9_dp15": "Mitigation costs for transition risks",
    "e1_9_dp16": "Climate scenarios used for assessment (at least 2)",
    "e1_9_dp17": "Breakdown of financial effects by time horizon",
    "e1_9_dp18": "Net climate financial impact (risks minus opportunities)",
}


# Descriptions of physical risk types for reporting.
PHYSICAL_RISK_DESCRIPTIONS: Dict[str, str] = {
    "acute_flooding": "Increased frequency and severity of flooding events due to "
                      "intense precipitation and rising sea/river levels",
    "acute_wildfire": "Increased frequency and severity of wildfire events due to "
                      "higher temperatures, drought, and vegetation drying",
    "acute_storm": "Increased frequency and severity of storms (cyclones, hurricanes, "
                   "typhoons) due to higher sea surface temperatures",
    "acute_heatwave": "Increased frequency, duration, and intensity of heatwave events "
                      "affecting operations, workforce, and infrastructure",
    "chronic_sea_level": "Gradual sea level rise threatening coastal assets, "
                         "infrastructure, and supply chain routes",
    "chronic_temperature": "Gradual increase in mean temperatures affecting energy "
                           "demand, agricultural yields, and operational efficiency",
    "chronic_precipitation": "Changes in precipitation patterns affecting water "
                             "availability, agricultural operations, and logistics",
    "chronic_water_stress": "Increasing water stress and scarcity affecting "
                            "operations, cooling, and supply chain inputs",
}


# Descriptions of transition risk types for reporting.
TRANSITION_RISK_DESCRIPTIONS: Dict[str, str] = {
    "policy_carbon_pricing": "Introduction or increase of carbon pricing (ETS, carbon "
                             "tax) raising operating costs for emission-intensive activities",
    "policy_regulation": "New or strengthened climate regulations (efficiency standards, "
                         "emission limits, disclosure requirements, building codes)",
    "technology_disruption": "Disruptive low-carbon technologies rendering existing assets "
                             "or products obsolete (stranded assets, competitive displacement)",
    "market_shift": "Shifts in customer demand towards low-carbon products and services, "
                    "changes in commodity prices, reduced demand for fossil-based products",
    "reputation": "Reputational damage from perceived insufficient climate action, "
                  "leading to loss of customers, investors, or talent",
    "legal_liability": "Climate-related litigation risk including failure-to-mitigate "
                       "claims, greenwashing allegations, and fiduciary duty challenges",
}


# Likelihood level to probability mapping for quantitative assessment.
LIKELIHOOD_PROBABILITIES: Dict[str, Dict[str, Any]] = {
    "very_low": {
        "label": "Very Low",
        "probability_range": "< 5%",
        "midpoint": Decimal("0.025"),
        "weight": Decimal("0.05"),
    },
    "low": {
        "label": "Low",
        "probability_range": "5-20%",
        "midpoint": Decimal("0.125"),
        "weight": Decimal("0.15"),
    },
    "medium": {
        "label": "Medium",
        "probability_range": "20-50%",
        "midpoint": Decimal("0.350"),
        "weight": Decimal("0.35"),
    },
    "high": {
        "label": "High",
        "probability_range": "50-80%",
        "midpoint": Decimal("0.650"),
        "weight": Decimal("0.65"),
    },
    "very_high": {
        "label": "Very High",
        "probability_range": "> 80%",
        "midpoint": Decimal("0.900"),
        "weight": Decimal("0.90"),
    },
}


# Climate scenario descriptions.
SCENARIO_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "rcp_2_6": {
        "name": "RCP 2.6",
        "warming": "~1.5C by 2100",
        "description": "Aggressive mitigation pathway limiting warming to 1.5C; "
                       "rapid phase-out of fossil fuels, high carbon prices",
    },
    "rcp_4_5": {
        "name": "RCP 4.5",
        "warming": "~2.4C by 2100",
        "description": "Moderate mitigation pathway with some climate policies; "
                       "gradual energy transition, moderate physical risks",
    },
    "rcp_8_5": {
        "name": "RCP 8.5",
        "warming": "~4.3C by 2100",
        "description": "High-emission pathway with limited mitigation; severe "
                       "physical risks, high adaptation costs",
    },
    "iea_nze": {
        "name": "IEA Net Zero by 2050",
        "warming": "~1.5C by 2100",
        "description": "IEA roadmap for net-zero CO2 emissions by 2050; rapid "
                       "deployment of clean energy, phase-out of unabated fossil fuels",
    },
    "iea_steps": {
        "name": "IEA Stated Policies Scenario",
        "warming": "~2.5C by 2100",
        "description": "IEA scenario based on current stated policies; incomplete "
                       "transition, moderate physical and transition risks",
    },
    "ngfs_orderly": {
        "name": "NGFS Orderly Transition",
        "warming": "~1.5C by 2100",
        "description": "NGFS scenario with early, coordinated policy action; "
                       "low physical risk, moderate transition risk",
    },
    "ngfs_disorderly": {
        "name": "NGFS Disorderly Transition",
        "warming": "~1.8C by 2100",
        "description": "NGFS scenario with delayed, sudden policy action; "
                       "moderate physical risk, high transition risk",
    },
    "ngfs_hot_house": {
        "name": "NGFS Hot House World",
        "warming": "~3.0+C by 2100",
        "description": "NGFS scenario with minimal policy action; very high "
                       "physical risk, low transition risk",
    },
}


# Damage function parameters for financial impact estimation.
# Maps physical risk types to expected annual loss factors
# (as fraction of affected asset value) by warming scenario.
DAMAGE_FUNCTION_PARAMS: Dict[str, Dict[str, Decimal]] = {
    "acute_flooding": {
        "rcp_2_6": Decimal("0.005"),
        "rcp_4_5": Decimal("0.012"),
        "rcp_8_5": Decimal("0.035"),
    },
    "acute_wildfire": {
        "rcp_2_6": Decimal("0.003"),
        "rcp_4_5": Decimal("0.008"),
        "rcp_8_5": Decimal("0.025"),
    },
    "acute_storm": {
        "rcp_2_6": Decimal("0.004"),
        "rcp_4_5": Decimal("0.010"),
        "rcp_8_5": Decimal("0.030"),
    },
    "acute_heatwave": {
        "rcp_2_6": Decimal("0.002"),
        "rcp_4_5": Decimal("0.006"),
        "rcp_8_5": Decimal("0.020"),
    },
    "chronic_sea_level": {
        "rcp_2_6": Decimal("0.003"),
        "rcp_4_5": Decimal("0.010"),
        "rcp_8_5": Decimal("0.040"),
    },
    "chronic_temperature": {
        "rcp_2_6": Decimal("0.002"),
        "rcp_4_5": Decimal("0.005"),
        "rcp_8_5": Decimal("0.015"),
    },
    "chronic_precipitation": {
        "rcp_2_6": Decimal("0.002"),
        "rcp_4_5": Decimal("0.004"),
        "rcp_8_5": Decimal("0.012"),
    },
    "chronic_water_stress": {
        "rcp_2_6": Decimal("0.003"),
        "rcp_4_5": Decimal("0.008"),
        "rcp_8_5": Decimal("0.025"),
    },
}


# Opportunity type descriptions.
OPPORTUNITY_DESCRIPTIONS: Dict[str, str] = {
    "resource_efficiency": "Cost savings from improved resource efficiency (energy, water, "
                           "materials) driven by climate-related innovation",
    "energy_source": "Revenue or savings from shifting to lower-cost renewable energy "
                     "sources, on-site generation, or power purchase agreements",
    "products_services": "Revenue from new or enhanced low-carbon products and services "
                         "meeting growing customer demand for sustainability",
    "markets": "Access to new markets opened by climate-related demand shifts, "
               "climate-resilient infrastructure, or green finance instruments",
    "resilience": "Value creation through resilience investments that reduce "
                  "vulnerability to physical and transition risks",
}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class PhysicalRisk(BaseModel):
    """A physical climate risk per ESRS E1-9 and AR E1-74.

    Represents a material physical risk (acute or chronic) to the
    undertaking's assets, operations, or value chain from climate
    change.
    """
    risk_id: str = Field(
        default_factory=_new_uuid,
        description="Unique risk identifier",
    )
    risk_type: PhysicalRiskType = Field(
        ...,
        description="Type of physical risk",
    )
    name: str = Field(
        default="",
        description="Short name or title for the risk",
        max_length=500,
    )
    description: str = Field(
        default="",
        description="Detailed description of the risk",
        max_length=5000,
    )
    affected_assets_value: Decimal = Field(
        default=Decimal("0.00"),
        description="Monetary value of assets at risk",
        ge=0,
    )
    affected_assets_pct: Decimal = Field(
        default=Decimal("0.00"),
        description="Percentage of total assets at risk",
        ge=0,
        le=Decimal("100.00"),
    )
    affected_revenue: Decimal = Field(
        default=Decimal("0.00"),
        description="Net revenue from activities at risk",
        ge=0,
    )
    affected_revenue_pct: Decimal = Field(
        default=Decimal("0.00"),
        description="Percentage of total revenue at risk",
        ge=0,
        le=Decimal("100.00"),
    )
    likelihood: LikelihoodLevel = Field(
        default=LikelihoodLevel.MEDIUM,
        description="Likelihood of the risk materialising",
    )
    time_horizon: RiskTimeHorizon = Field(
        default=RiskTimeHorizon.MEDIUM_TERM_3_10Y,
        description="Time horizon over which the risk is most relevant",
    )
    scenario: ClimateScenario = Field(
        default=ClimateScenario.RCP_4_5,
        description="Climate scenario under which the risk is assessed",
    )
    estimated_annual_loss: Decimal = Field(
        default=Decimal("0.00"),
        description="Estimated annual financial loss from the risk",
        ge=0,
    )
    adaptation_cost: Decimal = Field(
        default=Decimal("0.00"),
        description="Cost of adaptation measures to mitigate the risk",
        ge=0,
    )
    residual_risk_value: Decimal = Field(
        default=Decimal("0.00"),
        description="Residual risk value after adaptation measures",
        ge=0,
    )
    currency: str = Field(
        default="EUR",
        description="Currency for financial amounts",
        max_length=3,
    )
    location: str = Field(
        default="",
        description="Geographic location of the risk exposure",
        max_length=500,
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )


class TransitionRisk(BaseModel):
    """A transition climate risk per ESRS E1-9 and AR E1-75.

    Represents a material transition risk from policy, technology,
    market, or reputational shifts associated with the transition
    to a lower-carbon economy.
    """
    risk_id: str = Field(
        default_factory=_new_uuid,
        description="Unique risk identifier",
    )
    risk_type: TransitionRiskType = Field(
        ...,
        description="Type of transition risk",
    )
    name: str = Field(
        default="",
        description="Short name or title for the risk",
        max_length=500,
    )
    description: str = Field(
        default="",
        description="Detailed description of the risk",
        max_length=5000,
    )
    affected_assets_value: Decimal = Field(
        default=Decimal("0.00"),
        description="Monetary value of assets at risk",
        ge=0,
    )
    affected_assets_pct: Decimal = Field(
        default=Decimal("0.00"),
        description="Percentage of total assets at risk",
        ge=0,
        le=Decimal("100.00"),
    )
    affected_revenue_pct: Decimal = Field(
        default=Decimal("0.00"),
        description="Percentage of revenue exposed to this risk",
        ge=0,
        le=Decimal("100.00"),
    )
    likelihood: LikelihoodLevel = Field(
        default=LikelihoodLevel.MEDIUM,
        description="Likelihood of the risk materialising",
    )
    time_horizon: RiskTimeHorizon = Field(
        default=RiskTimeHorizon.MEDIUM_TERM_3_10Y,
        description="Time horizon over which the risk is most relevant",
    )
    scenario: ClimateScenario = Field(
        default=ClimateScenario.IEA_NZE,
        description="Climate scenario under which the risk is assessed",
    )
    estimated_financial_impact: Decimal = Field(
        default=Decimal("0.00"),
        description="Estimated total financial impact of the risk",
        ge=0,
    )
    mitigation_cost: Decimal = Field(
        default=Decimal("0.00"),
        description="Cost of mitigation measures to address the risk",
        ge=0,
    )
    residual_risk_value: Decimal = Field(
        default=Decimal("0.00"),
        description="Residual risk value after mitigation",
        ge=0,
    )
    currency: str = Field(
        default="EUR",
        description="Currency for financial amounts",
        max_length=3,
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )


class ClimateOpportunity(BaseModel):
    """A climate-related opportunity per ESRS E1-9 and AR E1-76.

    Represents a potential opportunity arising from climate change
    mitigation or adaptation efforts.
    """
    opportunity_id: str = Field(
        default_factory=_new_uuid,
        description="Unique opportunity identifier",
    )
    opportunity_type: ClimateOpportunityType = Field(
        ...,
        description="Type of climate-related opportunity",
    )
    name: str = Field(
        default="",
        description="Short name or title for the opportunity",
        max_length=500,
    )
    description: str = Field(
        default="",
        description="Detailed description of the opportunity",
        max_length=5000,
    )
    estimated_revenue_impact: Decimal = Field(
        default=Decimal("0.00"),
        description="Estimated annual revenue impact",
        ge=0,
    )
    estimated_cost_savings: Decimal = Field(
        default=Decimal("0.00"),
        description="Estimated annual cost savings",
        ge=0,
    )
    investment_required: Decimal = Field(
        default=Decimal("0.00"),
        description="Investment required to realise the opportunity",
        ge=0,
    )
    time_horizon: RiskTimeHorizon = Field(
        default=RiskTimeHorizon.MEDIUM_TERM_3_10Y,
        description="Time horizon for realisation",
    )
    currency: str = Field(
        default="EUR",
        description="Currency for financial amounts",
        max_length=3,
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )


class ClimateRiskResult(BaseModel):
    """Result of climate risk and opportunity assessment per ESRS E1-9.

    Contains the complete inventory of physical risks, transition
    risks, and opportunities with aggregated financial effects and
    scenario-based analysis.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this assessment",
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of assessment (UTC)",
    )
    physical_risks: List[PhysicalRisk] = Field(
        default_factory=list,
        description="List of material physical risks assessed",
    )
    transition_risks: List[TransitionRisk] = Field(
        default_factory=list,
        description="List of material transition risks assessed",
    )
    opportunities: List[ClimateOpportunity] = Field(
        default_factory=list,
        description="List of climate-related opportunities assessed",
    )
    total_physical_risks: int = Field(
        default=0,
        description="Total number of physical risks",
    )
    total_transition_risks: int = Field(
        default=0,
        description="Total number of transition risks",
    )
    total_opportunities: int = Field(
        default=0,
        description="Total number of opportunities",
    )
    total_physical_risk_exposure: Decimal = Field(
        default=Decimal("0.00"),
        description="Total estimated financial exposure from physical risks",
    )
    total_transition_risk_exposure: Decimal = Field(
        default=Decimal("0.00"),
        description="Total estimated financial exposure from transition risks",
    )
    total_opportunity_value: Decimal = Field(
        default=Decimal("0.00"),
        description="Total estimated value of climate opportunities",
    )
    net_climate_financial_impact: Decimal = Field(
        default=Decimal("0.00"),
        description="Net climate financial impact (risks minus opportunities)",
    )
    total_adaptation_cost: Decimal = Field(
        default=Decimal("0.00"),
        description="Total cost of adaptation measures for physical risks",
    )
    total_mitigation_cost: Decimal = Field(
        default=Decimal("0.00"),
        description="Total cost of mitigation measures for transition risks",
    )
    by_scenario: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Financial effects grouped by climate scenario",
    )
    by_time_horizon: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Financial effects grouped by time horizon",
    )
    physical_risks_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Physical risk count by type",
    )
    transition_risks_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Transition risk count by type",
    )
    opportunities_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Opportunity count by type",
    )
    scenarios_used: List[str] = Field(
        default_factory=list,
        description="Climate scenarios used in the assessment",
    )
    completeness_score: float = Field(
        default=0.0,
        description="Completeness score for E1-9 data points (0-100)",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the entire result",
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ClimateRiskEngine:
    """Climate risk and opportunity engine per ESRS E1-9.

    Provides deterministic, zero-hallucination assessment of:
    - Physical climate risks (acute and chronic)
    - Transition climate risks (policy, technology, market, legal)
    - Climate-related opportunities
    - Anticipated financial effects by scenario and time horizon
    - Completeness validation against E1-9 data points

    All calculations are bit-perfect reproducible.  No LLM is used
    in any calculation path.

    Usage::

        engine = ClimateRiskEngine()

        physical = PhysicalRisk(
            risk_type=PhysicalRiskType.ACUTE_FLOODING,
            affected_assets_value=Decimal("50000000.00"),
            likelihood=LikelihoodLevel.HIGH,
            scenario=ClimateScenario.RCP_4_5,
        )
        assessed_physical = engine.assess_physical_risk(physical)

        result = engine.build_risk_assessment(
            physical_risks=[assessed_physical],
            transition_risks=[],
            opportunities=[],
        )
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise ClimateRiskEngine."""
        self._physical_risks: List[PhysicalRisk] = []
        self._transition_risks: List[TransitionRisk] = []
        self._opportunities: List[ClimateOpportunity] = []
        logger.info(
            "ClimateRiskEngine v%s initialised", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Physical Risk Assessment                                             #
    # ------------------------------------------------------------------ #

    def assess_physical_risk(
        self, risk: PhysicalRisk
    ) -> PhysicalRisk:
        """Assess a physical climate risk per ESRS E1-9.

        Calculates estimated annual loss using the damage function
        approach: loss = affected_assets * damage_factor * likelihood.

        If estimated_annual_loss is already set, it is preserved.
        Otherwise it is calculated from the damage function parameters.

        Residual risk is calculated as:
            residual = estimated_annual_loss - adaptation_cost
            (floored at zero)

        Args:
            risk: PhysicalRisk to assess.

        Returns:
            PhysicalRisk with calculated fields and provenance hash.
        """
        t0 = time.perf_counter()

        if not risk.risk_id:
            risk.risk_id = _new_uuid()

        # Auto-fill name from risk type if empty
        if not risk.name:
            risk.name = PHYSICAL_RISK_DESCRIPTIONS.get(
                risk.risk_type.value, risk.risk_type.value
            )[:100]

        # Calculate estimated annual loss if not provided
        if risk.estimated_annual_loss == Decimal("0.00"):
            risk.estimated_annual_loss = self._calculate_physical_loss(risk)

        # Calculate residual risk
        residual = risk.estimated_annual_loss - risk.adaptation_cost
        risk.residual_risk_value = max(residual, Decimal("0.00"))

        risk.provenance_hash = _compute_hash(risk)
        self._physical_risks.append(risk)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        logger.info(
            "Assessed physical risk: type=%s, loss=%s, "
            "residual=%s in %.3f ms",
            risk.risk_type.value,
            risk.estimated_annual_loss,
            risk.residual_risk_value,
            elapsed_ms,
        )
        return risk

    # ------------------------------------------------------------------ #
    # Transition Risk Assessment                                           #
    # ------------------------------------------------------------------ #

    def assess_transition_risk(
        self, risk: TransitionRisk
    ) -> TransitionRisk:
        """Assess a transition climate risk per ESRS E1-9.

        If estimated_financial_impact is not set, calculates it as:
            impact = affected_assets_value * likelihood_weight

        Residual risk is:
            residual = estimated_financial_impact - mitigation_cost
            (floored at zero)

        Args:
            risk: TransitionRisk to assess.

        Returns:
            TransitionRisk with calculated fields and provenance hash.
        """
        t0 = time.perf_counter()

        if not risk.risk_id:
            risk.risk_id = _new_uuid()

        if not risk.name:
            risk.name = TRANSITION_RISK_DESCRIPTIONS.get(
                risk.risk_type.value, risk.risk_type.value
            )[:100]

        # Calculate financial impact if not provided
        if risk.estimated_financial_impact == Decimal("0.00"):
            likelihood_data = LIKELIHOOD_PROBABILITIES.get(
                risk.likelihood.value, {"weight": Decimal("0.35")}
            )
            weight = likelihood_data["weight"]
            risk.estimated_financial_impact = _round_val(
                risk.affected_assets_value * weight, 2
            )

        # Calculate residual risk
        residual = risk.estimated_financial_impact - risk.mitigation_cost
        risk.residual_risk_value = max(residual, Decimal("0.00"))

        risk.provenance_hash = _compute_hash(risk)
        self._transition_risks.append(risk)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        logger.info(
            "Assessed transition risk: type=%s, impact=%s, "
            "residual=%s in %.3f ms",
            risk.risk_type.value,
            risk.estimated_financial_impact,
            risk.residual_risk_value,
            elapsed_ms,
        )
        return risk

    # ------------------------------------------------------------------ #
    # Opportunity Assessment                                               #
    # ------------------------------------------------------------------ #

    def assess_opportunity(
        self, opp: ClimateOpportunity
    ) -> ClimateOpportunity:
        """Assess a climate-related opportunity per ESRS E1-9.

        Assigns a provenance hash and adds to the registry.

        Args:
            opp: ClimateOpportunity to assess.

        Returns:
            ClimateOpportunity with provenance hash.
        """
        t0 = time.perf_counter()

        if not opp.opportunity_id:
            opp.opportunity_id = _new_uuid()

        if not opp.name:
            opp.name = OPPORTUNITY_DESCRIPTIONS.get(
                opp.opportunity_type.value, opp.opportunity_type.value
            )[:100]

        opp.provenance_hash = _compute_hash(opp)
        self._opportunities.append(opp)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        logger.info(
            "Assessed opportunity: type=%s, revenue=%s, "
            "savings=%s in %.3f ms",
            opp.opportunity_type.value,
            opp.estimated_revenue_impact,
            opp.estimated_cost_savings,
            elapsed_ms,
        )
        return opp

    # ------------------------------------------------------------------ #
    # Risk Assessment Builder                                              #
    # ------------------------------------------------------------------ #

    def build_risk_assessment(
        self,
        physical_risks: Optional[List[PhysicalRisk]] = None,
        transition_risks: Optional[List[TransitionRisk]] = None,
        opportunities: Optional[List[ClimateOpportunity]] = None,
    ) -> ClimateRiskResult:
        """Build the complete climate risk assessment per E1-9.

        Aggregates all risks and opportunities into a single result
        with financial summaries, scenario breakdowns, and time
        horizon analysis.

        Args:
            physical_risks: List of physical risks (uses registry if None).
            transition_risks: List of transition risks (uses registry if None).
            opportunities: List of opportunities (uses registry if None).

        Returns:
            ClimateRiskResult with complete aggregation.
        """
        t0 = time.perf_counter()

        if physical_risks is None:
            physical_risks = list(self._physical_risks)
        if transition_risks is None:
            transition_risks = list(self._transition_risks)
        if opportunities is None:
            opportunities = list(self._opportunities)

        # Physical risk aggregation
        total_physical = Decimal("0.00")
        total_adaptation = Decimal("0.00")
        physical_by_type: Dict[str, int] = {}

        for risk in physical_risks:
            total_physical += risk.estimated_annual_loss
            total_adaptation += risk.adaptation_cost
            key = risk.risk_type.value
            physical_by_type[key] = physical_by_type.get(key, 0) + 1

        # Transition risk aggregation
        total_transition = Decimal("0.00")
        total_mitigation = Decimal("0.00")
        transition_by_type: Dict[str, int] = {}

        for risk in transition_risks:
            total_transition += risk.estimated_financial_impact
            total_mitigation += risk.mitigation_cost
            key = risk.risk_type.value
            transition_by_type[key] = transition_by_type.get(key, 0) + 1

        # Opportunity aggregation
        total_opportunity = Decimal("0.00")
        opp_by_type: Dict[str, int] = {}

        for opp in opportunities:
            total_opportunity += (
                opp.estimated_revenue_impact + opp.estimated_cost_savings
            )
            key = opp.opportunity_type.value
            opp_by_type[key] = opp_by_type.get(key, 0) + 1

        # Net impact
        total_risk = total_physical + total_transition
        net_impact = total_risk - total_opportunity

        # Scenario breakdown
        by_scenario = self._build_scenario_breakdown(
            physical_risks, transition_risks
        )

        # Time horizon breakdown
        by_time_horizon = self._build_time_horizon_breakdown(
            physical_risks, transition_risks, opportunities
        )

        # Scenarios used
        scenarios_used = set()
        for risk in physical_risks:
            scenarios_used.add(risk.scenario.value)
        for risk in transition_risks:
            scenarios_used.add(risk.scenario.value)

        # Completeness
        completeness = self._calculate_completeness(
            physical_risks, transition_risks, opportunities
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = ClimateRiskResult(
            physical_risks=physical_risks,
            transition_risks=transition_risks,
            opportunities=opportunities,
            total_physical_risks=len(physical_risks),
            total_transition_risks=len(transition_risks),
            total_opportunities=len(opportunities),
            total_physical_risk_exposure=_round_val(total_physical, 2),
            total_transition_risk_exposure=_round_val(total_transition, 2),
            total_opportunity_value=_round_val(total_opportunity, 2),
            net_climate_financial_impact=_round_val(net_impact, 2),
            total_adaptation_cost=_round_val(total_adaptation, 2),
            total_mitigation_cost=_round_val(total_mitigation, 2),
            by_scenario=by_scenario,
            by_time_horizon=by_time_horizon,
            physical_risks_by_type=physical_by_type,
            transition_risks_by_type=transition_by_type,
            opportunities_by_type=opp_by_type,
            scenarios_used=sorted(scenarios_used),
            completeness_score=completeness,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Built risk assessment: %d physical risks (%s exposure), "
            "%d transition risks (%s exposure), %d opportunities (%s value), "
            "net impact=%s in %.3f ms",
            len(physical_risks),
            total_physical,
            len(transition_risks),
            total_transition,
            len(opportunities),
            total_opportunity,
            net_impact,
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # Scenario Impact Calculation                                          #
    # ------------------------------------------------------------------ #

    def calculate_scenario_impact(
        self, result: ClimateRiskResult, scenario: ClimateScenario
    ) -> Dict[str, Any]:
        """Calculate total financial impact for a specific scenario.

        Filters all risks by the specified scenario and aggregates
        their financial impacts.

        Args:
            result: ClimateRiskResult to analyse.
            scenario: ClimateScenario to filter by.

        Returns:
            Dict with scenario-specific financial summary.
        """
        t0 = time.perf_counter()

        physical_in_scenario = [
            r for r in result.physical_risks
            if r.scenario == scenario
        ]
        transition_in_scenario = [
            r for r in result.transition_risks
            if r.scenario == scenario
        ]

        physical_total = sum(
            (r.estimated_annual_loss for r in physical_in_scenario),
            Decimal("0.00"),
        )
        transition_total = sum(
            (r.estimated_financial_impact for r in transition_in_scenario),
            Decimal("0.00"),
        )
        adaptation_total = sum(
            (r.adaptation_cost for r in physical_in_scenario),
            Decimal("0.00"),
        )
        mitigation_total = sum(
            (r.mitigation_cost for r in transition_in_scenario),
            Decimal("0.00"),
        )

        scenario_info = SCENARIO_DESCRIPTIONS.get(
            scenario.value, {"name": scenario.value, "warming": "N/A"}
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        return {
            "scenario": scenario.value,
            "scenario_name": scenario_info.get("name", ""),
            "warming": scenario_info.get("warming", ""),
            "description": scenario_info.get("description", ""),
            "physical_risks_count": len(physical_in_scenario),
            "transition_risks_count": len(transition_in_scenario),
            "total_physical_exposure": str(_round_val(physical_total, 2)),
            "total_transition_exposure": str(_round_val(transition_total, 2)),
            "total_exposure": str(
                _round_val(physical_total + transition_total, 2)
            ),
            "total_adaptation_cost": str(_round_val(adaptation_total, 2)),
            "total_mitigation_cost": str(_round_val(mitigation_total, 2)),
            "processing_time_ms": elapsed_ms,
            "provenance_hash": _compute_hash({
                "scenario": scenario.value,
                "physical": str(physical_total),
                "transition": str(transition_total),
            }),
        }

    # ------------------------------------------------------------------ #
    # Time Horizon Breakdown                                               #
    # ------------------------------------------------------------------ #

    def calculate_time_horizon_breakdown(
        self, result: ClimateRiskResult
    ) -> Dict[str, Any]:
        """Calculate financial effects breakdown by time horizon.

        Groups all risks and opportunities by their time horizon and
        aggregates financial impacts.

        Args:
            result: ClimateRiskResult to analyse.

        Returns:
            Dict with time horizon breakdown.
        """
        return self._build_time_horizon_breakdown(
            result.physical_risks,
            result.transition_risks,
            result.opportunities,
        )

    # ------------------------------------------------------------------ #
    # Completeness Validation                                              #
    # ------------------------------------------------------------------ #

    def validate_completeness(
        self, result: ClimateRiskResult
    ) -> Dict[str, Any]:
        """Validate completeness of E1-9 data points.

        Args:
            result: ClimateRiskResult to validate.

        Returns:
            Dict with data point coverage and completeness score.
        """
        datapoints_status: Dict[str, Dict[str, Any]] = {}
        covered = 0

        physical = result.physical_risks
        transition = result.transition_risks
        opps = result.opportunities

        checks = {
            "e1_9_dp01": any(r.affected_assets_value > 0 for r in physical),
            "e1_9_dp02": any(r.affected_assets_pct > 0 for r in physical),
            "e1_9_dp03": any(r.affected_assets_value > 0 for r in transition),
            "e1_9_dp04": any(r.affected_assets_pct > 0 for r in transition),
            "e1_9_dp05": any(r.affected_revenue > 0 for r in physical),
            "e1_9_dp06": any(r.affected_revenue_pct > 0 for r in transition),
            "e1_9_dp07": len(physical) > 0,
            "e1_9_dp08": len(transition) > 0,
            "e1_9_dp09": any(
                r.time_horizon is not None for r in physical
            ) or any(r.time_horizon is not None for r in transition),
            "e1_9_dp10": any(
                r.likelihood is not None for r in physical
            ) or any(r.likelihood is not None for r in transition),
            "e1_9_dp11": result.total_physical_risk_exposure > 0
            or result.total_transition_risk_exposure > 0,
            "e1_9_dp12": len(opps) > 0,
            "e1_9_dp13": result.total_opportunity_value > 0,
            "e1_9_dp14": result.total_adaptation_cost > 0,
            "e1_9_dp15": result.total_mitigation_cost > 0,
            "e1_9_dp16": len(result.scenarios_used) >= 2,
            "e1_9_dp17": len(result.by_time_horizon) > 0,
            "e1_9_dp18": True,  # Net impact always calculated
        }

        for dp_id, dp_label in E1_9_DATAPOINTS.items():
            is_covered = checks.get(dp_id, False)
            if is_covered:
                covered += 1
            datapoints_status[dp_id] = {
                "label": dp_label,
                "covered": is_covered,
                "status": "COMPLETE" if is_covered else "MISSING",
            }

        total = len(E1_9_DATAPOINTS)
        score = _round2(
            _safe_divide(float(covered), float(total), 0.0) * 100.0
        )

        return {
            "disclosure_requirement": "E1-9",
            "title": "Anticipated financial effects from material physical and "
                     "transition risks and potential climate-related opportunities",
            "total_datapoints": total,
            "covered_datapoints": covered,
            "missing_datapoints": total - covered,
            "completeness_score": score,
            "datapoints": datapoints_status,
            "provenance_hash": _compute_hash(datapoints_status),
        }

    # ------------------------------------------------------------------ #
    # E1-9 Data Point Extraction                                           #
    # ------------------------------------------------------------------ #

    def get_e1_9_datapoints(
        self, result: ClimateRiskResult
    ) -> Dict[str, Any]:
        """Extract structured E1-9 data points for XBRL tagging.

        Args:
            result: ClimateRiskResult to extract from.

        Returns:
            Dict mapping data point IDs to values.
        """
        physical = result.physical_risks
        transition = result.transition_risks
        opps = result.opportunities

        total_physical_assets = sum(
            (r.affected_assets_value for r in physical),
            Decimal("0.00"),
        )
        total_transition_assets = sum(
            (r.affected_assets_value for r in transition),
            Decimal("0.00"),
        )
        total_physical_revenue = sum(
            (r.affected_revenue for r in physical),
            Decimal("0.00"),
        )

        datapoints: Dict[str, Any] = {
            "e1_9_dp01": {
                "value": str(_round_val(total_physical_assets, 2)),
                "label": E1_9_DATAPOINTS["e1_9_dp01"],
                "xbrl_element": "esrs:AssetsAtPhysicalRisk",
            },
            "e1_9_dp02": {
                "value": [
                    {
                        "risk_type": r.risk_type.value,
                        "pct": str(r.affected_assets_pct),
                    }
                    for r in physical
                    if r.affected_assets_pct > 0
                ],
                "label": E1_9_DATAPOINTS["e1_9_dp02"],
                "xbrl_element": "esrs:AssetsAtPhysicalRiskPct",
            },
            "e1_9_dp03": {
                "value": str(_round_val(total_transition_assets, 2)),
                "label": E1_9_DATAPOINTS["e1_9_dp03"],
                "xbrl_element": "esrs:AssetsAtTransitionRisk",
            },
            "e1_9_dp04": {
                "value": [
                    {
                        "risk_type": r.risk_type.value,
                        "pct": str(r.affected_assets_pct),
                    }
                    for r in transition
                    if r.affected_assets_pct > 0
                ],
                "label": E1_9_DATAPOINTS["e1_9_dp04"],
                "xbrl_element": "esrs:AssetsAtTransitionRiskPct",
            },
            "e1_9_dp05": {
                "value": str(_round_val(total_physical_revenue, 2)),
                "label": E1_9_DATAPOINTS["e1_9_dp05"],
                "xbrl_element": "esrs:RevenueAtPhysicalRisk",
            },
            "e1_9_dp06": {
                "value": [
                    {
                        "risk_type": r.risk_type.value,
                        "revenue_pct": str(r.affected_revenue_pct),
                    }
                    for r in transition
                    if r.affected_revenue_pct > 0
                ],
                "label": E1_9_DATAPOINTS["e1_9_dp06"],
                "xbrl_element": "esrs:RevenueAtTransitionRisk",
            },
            "e1_9_dp07": {
                "value": [
                    {
                        "risk_id": r.risk_id,
                        "type": r.risk_type.value,
                        "description": PHYSICAL_RISK_DESCRIPTIONS.get(
                            r.risk_type.value, ""
                        ),
                        "location": r.location,
                    }
                    for r in physical
                ],
                "label": E1_9_DATAPOINTS["e1_9_dp07"],
                "xbrl_element": "esrs:PhysicalRiskDescription",
            },
            "e1_9_dp08": {
                "value": [
                    {
                        "risk_id": r.risk_id,
                        "type": r.risk_type.value,
                        "description": TRANSITION_RISK_DESCRIPTIONS.get(
                            r.risk_type.value, ""
                        ),
                    }
                    for r in transition
                ],
                "label": E1_9_DATAPOINTS["e1_9_dp08"],
                "xbrl_element": "esrs:TransitionRiskDescription",
            },
            "e1_9_dp09": {
                "value": result.by_time_horizon,
                "label": E1_9_DATAPOINTS["e1_9_dp09"],
                "xbrl_element": "esrs:RiskTimeHorizon",
            },
            "e1_9_dp10": {
                "value": [
                    {
                        "risk_id": r.risk_id,
                        "type": r.risk_type.value,
                        "likelihood": r.likelihood.value,
                    }
                    for r in physical
                ] + [
                    {
                        "risk_id": r.risk_id,
                        "type": r.risk_type.value,
                        "likelihood": r.likelihood.value,
                    }
                    for r in transition
                ],
                "label": E1_9_DATAPOINTS["e1_9_dp10"],
                "xbrl_element": "esrs:RiskLikelihood",
            },
            "e1_9_dp11": {
                "value": str(
                    _round_val(
                        result.total_physical_risk_exposure
                        + result.total_transition_risk_exposure,
                        2,
                    )
                ),
                "label": E1_9_DATAPOINTS["e1_9_dp11"],
                "xbrl_element": "esrs:EstimatedFinancialImpactRisks",
            },
            "e1_9_dp12": {
                "value": [
                    {
                        "opportunity_id": o.opportunity_id,
                        "type": o.opportunity_type.value,
                        "description": OPPORTUNITY_DESCRIPTIONS.get(
                            o.opportunity_type.value, ""
                        ),
                    }
                    for o in opps
                ],
                "label": E1_9_DATAPOINTS["e1_9_dp12"],
                "xbrl_element": "esrs:ClimateOpportunityDescription",
            },
            "e1_9_dp13": {
                "value": str(result.total_opportunity_value),
                "label": E1_9_DATAPOINTS["e1_9_dp13"],
                "xbrl_element": "esrs:ClimateOpportunityFinancialEffects",
            },
            "e1_9_dp14": {
                "value": str(result.total_adaptation_cost),
                "label": E1_9_DATAPOINTS["e1_9_dp14"],
                "xbrl_element": "esrs:AdaptationCosts",
            },
            "e1_9_dp15": {
                "value": str(result.total_mitigation_cost),
                "label": E1_9_DATAPOINTS["e1_9_dp15"],
                "xbrl_element": "esrs:TransitionRiskMitigationCosts",
            },
            "e1_9_dp16": {
                "value": [
                    {
                        "scenario": s,
                        "description": SCENARIO_DESCRIPTIONS.get(
                            s, {}
                        ).get("description", ""),
                    }
                    for s in result.scenarios_used
                ],
                "label": E1_9_DATAPOINTS["e1_9_dp16"],
                "xbrl_element": "esrs:ClimateScenariosUsed",
            },
            "e1_9_dp17": {
                "value": result.by_time_horizon,
                "label": E1_9_DATAPOINTS["e1_9_dp17"],
                "xbrl_element": "esrs:FinancialEffectsByTimeHorizon",
            },
            "e1_9_dp18": {
                "value": str(result.net_climate_financial_impact),
                "label": E1_9_DATAPOINTS["e1_9_dp18"],
                "xbrl_element": "esrs:NetClimateFinancialImpact",
            },
        }

        datapoints["provenance_hash"] = _compute_hash(datapoints)
        return datapoints

    # ------------------------------------------------------------------ #
    # Summary and Reporting Utilities                                      #
    # ------------------------------------------------------------------ #

    def get_physical_risk_summary(
        self, risk: PhysicalRisk
    ) -> Dict[str, Any]:
        """Return a structured summary of a single physical risk.

        Args:
            risk: PhysicalRisk to summarise.

        Returns:
            Dict with risk details.
        """
        return {
            "risk_id": risk.risk_id,
            "risk_type": risk.risk_type.value,
            "risk_description": PHYSICAL_RISK_DESCRIPTIONS.get(
                risk.risk_type.value, ""
            ),
            "name": risk.name,
            "affected_assets_value": str(risk.affected_assets_value),
            "affected_assets_pct": str(risk.affected_assets_pct),
            "affected_revenue": str(risk.affected_revenue),
            "likelihood": risk.likelihood.value,
            "likelihood_details": LIKELIHOOD_PROBABILITIES.get(
                risk.likelihood.value, {}
            ).get("label", ""),
            "time_horizon": risk.time_horizon.value,
            "scenario": risk.scenario.value,
            "estimated_annual_loss": str(risk.estimated_annual_loss),
            "adaptation_cost": str(risk.adaptation_cost),
            "residual_risk_value": str(risk.residual_risk_value),
            "location": risk.location,
            "provenance_hash": risk.provenance_hash,
        }

    def get_transition_risk_summary(
        self, risk: TransitionRisk
    ) -> Dict[str, Any]:
        """Return a structured summary of a single transition risk.

        Args:
            risk: TransitionRisk to summarise.

        Returns:
            Dict with risk details.
        """
        return {
            "risk_id": risk.risk_id,
            "risk_type": risk.risk_type.value,
            "risk_description": TRANSITION_RISK_DESCRIPTIONS.get(
                risk.risk_type.value, ""
            ),
            "name": risk.name,
            "affected_assets_value": str(risk.affected_assets_value),
            "affected_assets_pct": str(risk.affected_assets_pct),
            "affected_revenue_pct": str(risk.affected_revenue_pct),
            "likelihood": risk.likelihood.value,
            "time_horizon": risk.time_horizon.value,
            "scenario": risk.scenario.value,
            "estimated_financial_impact": str(risk.estimated_financial_impact),
            "mitigation_cost": str(risk.mitigation_cost),
            "residual_risk_value": str(risk.residual_risk_value),
            "provenance_hash": risk.provenance_hash,
        }

    def get_opportunity_summary(
        self, opp: ClimateOpportunity
    ) -> Dict[str, Any]:
        """Return a structured summary of a single opportunity.

        Args:
            opp: ClimateOpportunity to summarise.

        Returns:
            Dict with opportunity details.
        """
        total_value = opp.estimated_revenue_impact + opp.estimated_cost_savings

        return {
            "opportunity_id": opp.opportunity_id,
            "opportunity_type": opp.opportunity_type.value,
            "description": OPPORTUNITY_DESCRIPTIONS.get(
                opp.opportunity_type.value, ""
            ),
            "name": opp.name,
            "estimated_revenue_impact": str(opp.estimated_revenue_impact),
            "estimated_cost_savings": str(opp.estimated_cost_savings),
            "total_estimated_value": str(_round_val(total_value, 2)),
            "investment_required": str(opp.investment_required),
            "time_horizon": opp.time_horizon.value,
            "provenance_hash": opp.provenance_hash,
        }

    def get_scenario_descriptions(self) -> Dict[str, Dict[str, str]]:
        """Return all climate scenario descriptions.

        Returns:
            Dict mapping scenario ID to description dict.
        """
        return dict(SCENARIO_DESCRIPTIONS)

    def clear_registry(self) -> None:
        """Clear all registered risks and opportunities."""
        self._physical_risks.clear()
        self._transition_risks.clear()
        self._opportunities.clear()
        logger.info("ClimateRiskEngine registry cleared")

    # ------------------------------------------------------------------ #
    # Private Helpers                                                      #
    # ------------------------------------------------------------------ #

    def _calculate_physical_loss(self, risk: PhysicalRisk) -> Decimal:
        """Calculate estimated annual loss for a physical risk.

        Uses the damage function approach:
            loss = affected_assets * damage_factor * likelihood_weight

        The damage factor comes from DAMAGE_FUNCTION_PARAMS, keyed by
        risk type and scenario.  The likelihood weight comes from
        LIKELIHOOD_PROBABILITIES.

        Args:
            risk: PhysicalRisk with affected_assets_value, risk_type,
                  scenario, and likelihood.

        Returns:
            Estimated annual loss as Decimal.
        """
        risk_key = risk.risk_type.value
        scenario_key = risk.scenario.value

        # Get damage factor
        damage_factors = DAMAGE_FUNCTION_PARAMS.get(risk_key, {})
        damage_factor = damage_factors.get(
            scenario_key, Decimal("0.005")
        )

        # Get likelihood weight
        likelihood_data = LIKELIHOOD_PROBABILITIES.get(
            risk.likelihood.value, {"weight": Decimal("0.35")}
        )
        likelihood_weight = likelihood_data["weight"]

        # Calculate loss
        loss = risk.affected_assets_value * damage_factor * likelihood_weight
        return _round_val(loss, 2)

    def _build_scenario_breakdown(
        self,
        physical_risks: List[PhysicalRisk],
        transition_risks: List[TransitionRisk],
    ) -> Dict[str, Dict[str, str]]:
        """Build financial effects breakdown by scenario.

        Args:
            physical_risks: List of physical risks.
            transition_risks: List of transition risks.

        Returns:
            Dict mapping scenario to financial summary.
        """
        scenarios: Dict[str, Dict[str, Decimal]] = {}

        for risk in physical_risks:
            key = risk.scenario.value
            if key not in scenarios:
                scenarios[key] = {
                    "physical_exposure": Decimal("0.00"),
                    "transition_exposure": Decimal("0.00"),
                    "total_exposure": Decimal("0.00"),
                }
            scenarios[key]["physical_exposure"] += risk.estimated_annual_loss

        for risk in transition_risks:
            key = risk.scenario.value
            if key not in scenarios:
                scenarios[key] = {
                    "physical_exposure": Decimal("0.00"),
                    "transition_exposure": Decimal("0.00"),
                    "total_exposure": Decimal("0.00"),
                }
            scenarios[key]["transition_exposure"] += (
                risk.estimated_financial_impact
            )

        # Calculate totals and convert to strings
        result: Dict[str, Dict[str, str]] = {}
        for key, data in scenarios.items():
            total = data["physical_exposure"] + data["transition_exposure"]
            result[key] = {
                "physical_exposure": str(_round_val(data["physical_exposure"], 2)),
                "transition_exposure": str(_round_val(data["transition_exposure"], 2)),
                "total_exposure": str(_round_val(total, 2)),
            }

        return result

    def _build_time_horizon_breakdown(
        self,
        physical_risks: List[PhysicalRisk],
        transition_risks: List[TransitionRisk],
        opportunities: List[ClimateOpportunity],
    ) -> Dict[str, Dict[str, str]]:
        """Build financial effects breakdown by time horizon.

        Args:
            physical_risks: List of physical risks.
            transition_risks: List of transition risks.
            opportunities: List of climate opportunities.

        Returns:
            Dict mapping time horizon to financial summary.
        """
        horizons: Dict[str, Dict[str, Decimal]] = {}

        for th in RiskTimeHorizon:
            horizons[th.value] = {
                "physical_exposure": Decimal("0.00"),
                "transition_exposure": Decimal("0.00"),
                "opportunity_value": Decimal("0.00"),
                "net_impact": Decimal("0.00"),
            }

        for risk in physical_risks:
            key = risk.time_horizon.value
            horizons[key]["physical_exposure"] += risk.estimated_annual_loss

        for risk in transition_risks:
            key = risk.time_horizon.value
            horizons[key]["transition_exposure"] += (
                risk.estimated_financial_impact
            )

        for opp in opportunities:
            key = opp.time_horizon.value
            horizons[key]["opportunity_value"] += (
                opp.estimated_revenue_impact + opp.estimated_cost_savings
            )

        # Calculate net impact and convert to strings
        result: Dict[str, Dict[str, str]] = {}
        for key, data in horizons.items():
            total_risk = data["physical_exposure"] + data["transition_exposure"]
            net = total_risk - data["opportunity_value"]
            result[key] = {
                "physical_exposure": str(_round_val(data["physical_exposure"], 2)),
                "transition_exposure": str(_round_val(data["transition_exposure"], 2)),
                "opportunity_value": str(_round_val(data["opportunity_value"], 2)),
                "total_risk_exposure": str(_round_val(total_risk, 2)),
                "net_impact": str(_round_val(net, 2)),
            }

        return result

    def _calculate_completeness(
        self,
        physical_risks: List[PhysicalRisk],
        transition_risks: List[TransitionRisk],
        opportunities: List[ClimateOpportunity],
    ) -> float:
        """Calculate E1-9 completeness score.

        Args:
            physical_risks: List of physical risks.
            transition_risks: List of transition risks.
            opportunities: List of climate opportunities.

        Returns:
            Completeness score (0-100).
        """
        total = len(E1_9_DATAPOINTS)

        scenarios_used = set()
        for r in physical_risks:
            scenarios_used.add(r.scenario.value)
        for r in transition_risks:
            scenarios_used.add(r.scenario.value)

        checks = [
            any(r.affected_assets_value > 0 for r in physical_risks) if physical_risks else False,
            any(r.affected_assets_pct > 0 for r in physical_risks) if physical_risks else False,
            any(r.affected_assets_value > 0 for r in transition_risks) if transition_risks else False,
            any(r.affected_assets_pct > 0 for r in transition_risks) if transition_risks else False,
            any(r.affected_revenue > 0 for r in physical_risks) if physical_risks else False,
            any(r.affected_revenue_pct > 0 for r in transition_risks) if transition_risks else False,
            len(physical_risks) > 0,
            len(transition_risks) > 0,
            True if physical_risks or transition_risks else False,
            True if physical_risks or transition_risks else False,
            any(r.estimated_annual_loss > 0 for r in physical_risks) or
            any(r.estimated_financial_impact > 0 for r in transition_risks),
            len(opportunities) > 0,
            any(
                (o.estimated_revenue_impact + o.estimated_cost_savings) > 0
                for o in opportunities
            ) if opportunities else False,
            any(r.adaptation_cost > 0 for r in physical_risks) if physical_risks else False,
            any(r.mitigation_cost > 0 for r in transition_risks) if transition_risks else False,
            len(scenarios_used) >= 2,
            True if physical_risks or transition_risks or opportunities else False,
            True,  # Net impact always calculated
        ]

        covered = sum(1 for c in checks if c)
        return _round2(_safe_divide(float(covered), float(total), 0.0) * 100.0)
