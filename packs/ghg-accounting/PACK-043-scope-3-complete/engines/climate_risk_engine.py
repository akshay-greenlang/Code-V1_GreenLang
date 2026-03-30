# -*- coding: utf-8 -*-
"""
ClimateRiskEngine - PACK-043 Scope 3 Complete Pack Engine 7
==============================================================

Translates Scope 3 emissions into financial risk metrics aligned
with TCFD recommendations and ISSB IFRS S2 disclosure requirements.
Quantifies transition risk (carbon pricing, policy, technology),
physical risk (acute and chronic hazards), opportunity value,
stranded-asset exposure, and runs multi-scenario analysis using
IEA NZE 2050, NGFS orderly/disorderly/hot-house pathways.

Transition Risk Formula:
    financial_exposure = scope3_tco2e * carbon_price_per_tonne
    annual_cost_at_icp = scope3_tco2e * internal_carbon_price

Carbon Pricing Scenarios (per tCO2e):
    Low:    $50   (current voluntary market baseline)
    Medium: $100  (IEA STEPS mid-range)
    High:   $150  (IEA APS pathway)
    Extreme: $200 (IEA NZE 2050 by 2030)

CBAM Exposure:
    cbam_cost = sum(import_qty * embedded_emissions * cbam_rate)
    For each product in scope: cement, steel, aluminium, fertiliser,
    electricity, hydrogen.

Physical Risk:
    disruption_probability = base_hazard_prob * exposure_factor * vulnerability_factor
    Financial impact = disruption_probability * revenue_at_risk * duration_factor

NPV of Risks and Opportunities:
    npv = sum(annual_impact_t / (1 + r)^t for t in range(horizon))

NGFS Scenarios:
    Orderly:    Early, gradual policy action; limited physical risk.
    Disorderly: Late, sudden policy action; medium physical risk.
    Hot House:  No additional policy; severe physical risk by 2050.

Regulatory References:
    - TCFD Recommendations (2017, revised 2021)
    - ISSB IFRS S2 Climate-related Disclosures (2023)
    - IEA World Energy Outlook 2024 (NZE, APS, STEPS)
    - NGFS Climate Scenarios v4 (2023)
    - EU CBAM Regulation (EU) 2023/956
    - GHG Protocol Scope 3 Standard (2011)
    - ESRS E1 para 64-68 (climate-related financial effects)

Zero-Hallucination:
    - All financial calculations use deterministic Decimal arithmetic
    - Scenario parameters from published IEA/NGFS data
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-043 Scope 3 Complete
Engine:  7 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "43.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash -- dict, Pydantic model, or other serializable.

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)

def _round2(value: Any) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _round4(value: Any) -> float:
    """Round to 4 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RiskType(str, Enum):
    """Climate risk type per TCFD taxonomy."""
    TRANSITION = "transition"
    PHYSICAL = "physical"

class TransitionRiskDriver(str, Enum):
    """Transition risk drivers per TCFD."""
    CARBON_PRICING = "carbon_pricing"
    POLICY_REGULATION = "policy_regulation"
    TECHNOLOGY_SHIFT = "technology_shift"
    MARKET_SHIFT = "market_shift"
    REPUTATION = "reputation"

class PhysicalHazardType(str, Enum):
    """Physical hazard types."""
    FLOOD = "flood"
    DROUGHT = "drought"
    HEAT_STRESS = "heat_stress"
    CYCLONE = "cyclone"
    SEA_LEVEL_RISE = "sea_level_rise"
    WILDFIRE = "wildfire"

class ScenarioType(str, Enum):
    """Climate scenario types."""
    IEA_NZE = "iea_nze_2050"
    IEA_APS = "iea_aps"
    IEA_STEPS = "iea_steps"
    NGFS_ORDERLY = "ngfs_orderly"
    NGFS_DISORDERLY = "ngfs_disorderly"
    NGFS_HOT_HOUSE = "ngfs_hot_house"

class RiskSeverity(str, Enum):
    """Risk severity classification."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"

class OpportunityType(str, Enum):
    """Climate opportunity types per TCFD."""
    RESOURCE_EFFICIENCY = "resource_efficiency"
    ENERGY_SOURCE = "energy_source"
    PRODUCTS_SERVICES = "products_services"
    MARKETS = "markets"
    RESILIENCE = "resilience"

class PCAFAssetClass(str, Enum):
    """PCAF asset classes for financed emissions."""
    LISTED_EQUITY = "listed_equity"
    CORPORATE_BONDS = "corporate_bonds"
    BUSINESS_LOANS = "business_loans"
    PROJECT_FINANCE = "project_finance"
    COMMERCIAL_REAL_ESTATE = "commercial_real_estate"
    MORTGAGES = "mortgages"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Carbon price scenarios (USD per tCO2e).
CARBON_PRICE_SCENARIOS: Dict[str, float] = {
    "low": 50.0,
    "medium": 100.0,
    "high": 150.0,
    "extreme": 200.0,
}
"""Carbon price scenarios in USD/tCO2e."""

# IEA NZE 2050 pathway carbon prices by decade.
IEA_NZE_CARBON_PRICES: Dict[int, float] = {
    2025: 75.0,
    2030: 130.0,
    2035: 160.0,
    2040: 190.0,
    2045: 200.0,
    2050: 250.0,
}
"""IEA NZE 2050 carbon price trajectory (USD/tCO2e)."""

# NGFS scenario parameters.
NGFS_SCENARIOS: Dict[str, Dict[str, Any]] = {
    ScenarioType.NGFS_ORDERLY: {
        "name": "Orderly (Net Zero 2050)",
        "description": "Early, gradual policy action starting before 2025",
        "carbon_price_2030": 130.0,
        "carbon_price_2050": 250.0,
        "physical_risk_multiplier": 1.0,
        "transition_risk_multiplier": 1.5,
        "temperature_2100": 1.5,
        "policy_stringency": "high",
    },
    ScenarioType.NGFS_DISORDERLY: {
        "name": "Disorderly (Delayed Transition)",
        "description": "Late, sudden policy action after 2030",
        "carbon_price_2030": 50.0,
        "carbon_price_2050": 350.0,
        "physical_risk_multiplier": 1.5,
        "transition_risk_multiplier": 2.5,
        "temperature_2100": 1.8,
        "policy_stringency": "sudden_high",
    },
    ScenarioType.NGFS_HOT_HOUSE: {
        "name": "Hot House World",
        "description": "No additional policy action beyond current pledges",
        "carbon_price_2030": 30.0,
        "carbon_price_2050": 50.0,
        "physical_risk_multiplier": 3.0,
        "transition_risk_multiplier": 0.5,
        "temperature_2100": 3.0,
        "policy_stringency": "low",
    },
}
"""NGFS scenario parameters (v4, 2023)."""

# IEA scenario parameters.
IEA_SCENARIOS: Dict[str, Dict[str, Any]] = {
    ScenarioType.IEA_NZE: {
        "name": "IEA Net Zero Emissions by 2050",
        "carbon_price_2030": 130.0,
        "carbon_price_2050": 250.0,
        "emission_reduction_2030_pct": 45.0,
        "emission_reduction_2050_pct": 100.0,
    },
    ScenarioType.IEA_APS: {
        "name": "IEA Announced Pledges Scenario",
        "carbon_price_2030": 90.0,
        "carbon_price_2050": 160.0,
        "emission_reduction_2030_pct": 25.0,
        "emission_reduction_2050_pct": 65.0,
    },
    ScenarioType.IEA_STEPS: {
        "name": "IEA Stated Policies Scenario",
        "carbon_price_2030": 50.0,
        "carbon_price_2050": 80.0,
        "emission_reduction_2030_pct": 10.0,
        "emission_reduction_2050_pct": 35.0,
    },
}
"""IEA World Energy Outlook 2024 scenario parameters."""

# CBAM product rates (tCO2e per tonne of product, default EU benchmarks).
CBAM_DEFAULT_RATES: Dict[str, float] = {
    "cement": 0.766,
    "steel": 1.552,
    "aluminium": 6.700,
    "fertiliser": 2.927,
    "electricity": 0.376,   # tCO2e per MWh
    "hydrogen": 9.000,
}
"""CBAM default embedded emission rates (tCO2e per tonne or per MWh)."""

# Physical hazard base probabilities (annual, moderate scenario).
HAZARD_BASE_PROBABILITIES: Dict[str, float] = {
    PhysicalHazardType.FLOOD: 0.05,
    PhysicalHazardType.DROUGHT: 0.08,
    PhysicalHazardType.HEAT_STRESS: 0.10,
    PhysicalHazardType.CYCLONE: 0.03,
    PhysicalHazardType.SEA_LEVEL_RISE: 0.02,
    PhysicalHazardType.WILDFIRE: 0.04,
}
"""Annual base probability of disruption by hazard type."""

# Regional exposure multipliers.
REGIONAL_EXPOSURE: Dict[str, Dict[str, float]] = {
    "south_asia": {"flood": 2.0, "heat_stress": 2.5, "cyclone": 2.0, "drought": 1.5},
    "southeast_asia": {"flood": 2.5, "heat_stress": 2.0, "cyclone": 2.5, "sea_level_rise": 2.5},
    "sub_saharan_africa": {"drought": 3.0, "heat_stress": 2.5, "flood": 1.5},
    "coastal_americas": {"cyclone": 2.5, "sea_level_rise": 2.0, "flood": 1.5},
    "mediterranean": {"drought": 2.0, "wildfire": 2.5, "heat_stress": 2.0},
    "northern_europe": {"flood": 1.2, "heat_stress": 1.0},
    "north_america": {"wildfire": 1.5, "cyclone": 1.5, "flood": 1.2},
    "global_average": {"flood": 1.0, "drought": 1.0, "heat_stress": 1.0, "cyclone": 1.0},
}
"""Regional exposure multipliers for physical hazards."""

# ---------------------------------------------------------------------------
# Pydantic Data Models
# ---------------------------------------------------------------------------

class Scope3CategoryData(BaseModel):
    """Scope 3 category emission data for risk analysis.

    Attributes:
        category: Scope 3 category number (1-15).
        tco2e: Emissions in tCO2e.
        spend: Annual spend associated with this category.
        methodology_tier: Data methodology tier (spend/average/supplier/product).
        key_suppliers: Number of key suppliers in this category.
        regions: Regions of origin.
    """
    category: int = Field(..., ge=1, le=15, description="Scope 3 category (1-15)")
    tco2e: float = Field(..., ge=0, description="Emissions tCO2e")
    spend: float = Field(default=0, ge=0, description="Associated spend")
    methodology_tier: str = Field(default="spend", description="Methodology tier")
    key_suppliers: int = Field(default=0, ge=0, description="Number of key suppliers")
    regions: List[str] = Field(default_factory=list, description="Regions of origin")

class SupplierLocation(BaseModel):
    """Supplier location for physical risk assessment.

    Attributes:
        supplier_id: Supplier identifier.
        supplier_name: Supplier name.
        location_name: Location name or description.
        region: Climate region classification.
        revenue_at_risk: Revenue at risk at this location.
        hazard_vulnerabilities: Known hazard vulnerabilities.
    """
    supplier_id: str = Field(..., description="Supplier identifier")
    supplier_name: str = Field(default="", description="Supplier name")
    location_name: str = Field(default="", description="Location name")
    region: str = Field(default="global_average", description="Climate region")
    revenue_at_risk: float = Field(default=0, ge=0, description="Revenue at risk")
    hazard_vulnerabilities: List[str] = Field(
        default_factory=list, description="Known hazard vulnerabilities"
    )

class ImportItem(BaseModel):
    """Import item for CBAM exposure calculation.

    Attributes:
        product_type: CBAM product type (cement, steel, etc.).
        quantity_tonnes: Import quantity in tonnes.
        origin_country: Country of origin.
        embedded_emissions_tco2e: Known embedded emissions (optional).
        cbam_rate_override: Override default CBAM rate.
    """
    product_type: str = Field(..., description="CBAM product type")
    quantity_tonnes: float = Field(..., ge=0, description="Import quantity (tonnes)")
    origin_country: str = Field(default="", description="Country of origin")
    embedded_emissions_tco2e: Optional[float] = Field(
        default=None, description="Known embedded emissions"
    )
    cbam_rate_override: Optional[float] = Field(
        default=None, description="Override CBAM rate"
    )

class MarketData(BaseModel):
    """Market data for opportunity assessment.

    Attributes:
        low_carbon_market_growth_pct: Annual growth of low-carbon alternatives.
        green_premium_pct: Price premium for low-carbon products.
        customer_willingness_to_pay_pct: Customer green premium tolerance.
        market_size_total: Total addressable market.
    """
    low_carbon_market_growth_pct: float = Field(default=15.0, description="Annual growth %")
    green_premium_pct: float = Field(default=5.0, description="Green premium %")
    customer_willingness_to_pay_pct: float = Field(default=60.0, description="WTP %")
    market_size_total: float = Field(default=0, ge=0, description="Total market size")

class SupplierAsset(BaseModel):
    """Supplier asset for stranded asset assessment.

    Attributes:
        supplier_id: Supplier identifier.
        supplier_name: Supplier name.
        sector: Sector classification.
        asset_value: Book value of high-carbon assets.
        asset_type: Type of asset (plant, reserve, equipment).
        remaining_useful_life_years: Remaining useful life.
        annual_emissions_tco2e: Annual emissions from asset.
    """
    supplier_id: str = Field(..., description="Supplier identifier")
    supplier_name: str = Field(default="", description="Supplier name")
    sector: str = Field(default="other", description="Sector")
    asset_value: float = Field(default=0, ge=0, description="Asset book value")
    asset_type: str = Field(default="plant", description="Asset type")
    remaining_useful_life_years: int = Field(default=20, ge=0, description="Remaining life")
    annual_emissions_tco2e: float = Field(default=0, ge=0, description="Annual emissions")

class TransitionRiskResult(BaseModel):
    """Transition risk quantification result.

    Attributes:
        total_scope3_tco2e: Total Scope 3 emissions assessed.
        carbon_price_usd: Carbon price used.
        annual_exposure_usd: Annual financial exposure.
        by_category: Exposure by Scope 3 category.
        severity: Risk severity classification.
        pct_of_revenue: Exposure as percentage of revenue.
        provenance_hash: SHA-256 provenance.
        calculated_at: Calculation timestamp.
    """
    total_scope3_tco2e: float
    carbon_price_usd: float
    annual_exposure_usd: float
    by_category: Dict[str, float] = Field(default_factory=dict)
    severity: str = ""
    pct_of_revenue: float = 0.0
    provenance_hash: str = ""
    calculated_at: str = ""

class PhysicalRiskResult(BaseModel):
    """Physical risk assessment result.

    Attributes:
        total_locations_assessed: Number of locations assessed.
        total_revenue_at_risk: Total revenue at risk.
        weighted_disruption_probability: Weighted disruption probability.
        expected_annual_loss: Expected annual loss.
        by_hazard: Breakdown by hazard type.
        by_location: Breakdown by location.
        severity: Risk severity.
        provenance_hash: SHA-256 provenance.
        calculated_at: Calculation timestamp.
    """
    total_locations_assessed: int
    total_revenue_at_risk: float
    weighted_disruption_probability: float
    expected_annual_loss: float
    by_hazard: Dict[str, float] = Field(default_factory=dict)
    by_location: Dict[str, float] = Field(default_factory=dict)
    severity: str = ""
    provenance_hash: str = ""
    calculated_at: str = ""

class OpportunityResult(BaseModel):
    """Climate opportunity assessment result.

    Attributes:
        opportunity_type: Type of opportunity.
        description: Description of the opportunity.
        estimated_annual_value: Estimated annual value.
        confidence: Confidence level (low/medium/high).
        investment_required: Estimated investment to capture.
        payback_years: Payback period.
        provenance_hash: SHA-256 provenance.
        calculated_at: Calculation timestamp.
    """
    opportunity_type: str
    description: str
    estimated_annual_value: float
    confidence: str = "medium"
    investment_required: float = 0.0
    payback_years: float = 0.0
    provenance_hash: str = ""
    calculated_at: str = ""

class CBAMExposure(BaseModel):
    """CBAM exposure assessment.

    Attributes:
        total_cbam_cost: Total CBAM cost.
        by_product: Cost by product type.
        total_embedded_emissions: Total embedded emissions.
        provenance_hash: SHA-256 provenance.
        calculated_at: Calculation timestamp.
    """
    total_cbam_cost: float
    by_product: Dict[str, float] = Field(default_factory=dict)
    total_embedded_emissions: float = 0.0
    provenance_hash: str = ""
    calculated_at: str = ""

class StrandedAssetResult(BaseModel):
    """Stranded asset assessment.

    Attributes:
        supplier_id: Supplier identifier.
        supplier_name: Supplier name.
        asset_value_at_risk: Value of assets at risk.
        impairment_pct: Estimated impairment percentage.
        impairment_value: Estimated impairment value.
        years_to_stranding: Estimated years to stranding.
        provenance_hash: SHA-256 provenance.
        calculated_at: Calculation timestamp.
    """
    supplier_id: str
    supplier_name: str
    asset_value_at_risk: float
    impairment_pct: float
    impairment_value: float
    years_to_stranding: float
    provenance_hash: str = ""
    calculated_at: str = ""

class FinancialNPV(BaseModel):
    """Net present value of risks and opportunities.

    Attributes:
        horizon_years: Analysis horizon.
        discount_rate: Discount rate used.
        npv_risks: NPV of all risks.
        npv_opportunities: NPV of all opportunities.
        net_npv: Net NPV (opportunities - risks).
        annual_risk_trajectory: Year-by-year risk values.
        annual_opp_trajectory: Year-by-year opportunity values.
        provenance_hash: SHA-256 provenance.
        calculated_at: Calculation timestamp.
    """
    horizon_years: int
    discount_rate: float
    npv_risks: float
    npv_opportunities: float
    net_npv: float
    annual_risk_trajectory: Dict[int, float] = Field(default_factory=dict)
    annual_opp_trajectory: Dict[int, float] = Field(default_factory=dict)
    provenance_hash: str = ""
    calculated_at: str = ""

class ScenarioResult(BaseModel):
    """Result of a scenario analysis run.

    Attributes:
        scenario: Scenario type.
        scenario_name: Human-readable scenario name.
        temperature_outcome: Temperature outcome by 2100.
        carbon_price_2030: Carbon price in 2030.
        carbon_price_2050: Carbon price in 2050.
        transition_risk_exposure: Transition risk exposure.
        physical_risk_exposure: Physical risk exposure.
        total_risk: Total risk.
        opportunities: Identified opportunities.
        net_impact: Net impact (risk - opportunity).
        description: Scenario narrative.
        provenance_hash: SHA-256 provenance.
        calculated_at: Calculation timestamp.
    """
    scenario: str
    scenario_name: str
    temperature_outcome: float = 0.0
    carbon_price_2030: float = 0.0
    carbon_price_2050: float = 0.0
    transition_risk_exposure: float = 0.0
    physical_risk_exposure: float = 0.0
    total_risk: float = 0.0
    opportunities: float = 0.0
    net_impact: float = 0.0
    description: str = ""
    provenance_hash: str = ""
    calculated_at: str = ""

# ---------------------------------------------------------------------------
# Engine Class
# ---------------------------------------------------------------------------

class ClimateRiskEngine:
    """Translates Scope 3 emissions into financial risk metrics.

    Provides deterministic quantification of transition risk, physical
    risk, climate opportunities, carbon-pricing exposure, CBAM impact,
    stranded assets, financial NPV, and multi-scenario analysis aligned
    with TCFD, ISSB S2, IEA, and NGFS frameworks.

    All calculations use ``Decimal`` arithmetic.  Every result carries
    a SHA-256 provenance hash.

    Example:
        >>> engine = ClimateRiskEngine()
        >>> result = engine.quantify_transition_risk(scope3_data, carbon_price=100.0)
        >>> print(result.annual_exposure_usd)
    """

    def __init__(self) -> None:
        """Initialise ClimateRiskEngine."""
        logger.info("ClimateRiskEngine v%s initialised", _MODULE_VERSION)

    # -------------------------------------------------------------------
    # Public -- quantify_transition_risk
    # -------------------------------------------------------------------

    def quantify_transition_risk(
        self,
        scope3_data: List[Scope3CategoryData],
        carbon_price: float = 100.0,
        revenue: float = 0.0,
    ) -> TransitionRiskResult:
        """Quantify financial exposure from carbon pricing on Scope 3.

        Args:
            scope3_data: Scope 3 category emissions.
            carbon_price: Carbon price in USD/tCO2e.
            revenue: Company revenue for percentage calculation.

        Returns:
            TransitionRiskResult with exposure by category.
        """
        start_ms = time.time()
        price_d = _decimal(carbon_price)
        total_tco2e = Decimal("0")
        by_category: Dict[str, float] = {}

        for cat in scope3_data:
            cat_d = _decimal(cat.tco2e)
            total_tco2e += cat_d
            exposure = cat_d * price_d
            by_category[f"category_{cat.category}"] = _round2(exposure)

        annual_exposure = total_tco2e * price_d

        # Severity classification.
        rev_d = _decimal(revenue)
        pct_rev = _safe_pct(annual_exposure, rev_d) if rev_d > Decimal("0") else Decimal("0")
        if pct_rev >= Decimal("10"):
            severity = RiskSeverity.CRITICAL.value
        elif pct_rev >= Decimal("5"):
            severity = RiskSeverity.HIGH.value
        elif pct_rev >= Decimal("2"):
            severity = RiskSeverity.MEDIUM.value
        elif pct_rev >= Decimal("0.5"):
            severity = RiskSeverity.LOW.value
        else:
            severity = RiskSeverity.NEGLIGIBLE.value

        result = TransitionRiskResult(
            total_scope3_tco2e=_round2(total_tco2e),
            carbon_price_usd=_round2(price_d),
            annual_exposure_usd=_round2(annual_exposure),
            by_category=by_category,
            severity=severity,
            pct_of_revenue=_round2(pct_rev),
            calculated_at=utcnow().isoformat(),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed_ms = (time.time() - start_ms) * 1000
        logger.info(
            "Transition risk: $%.0f exposure at $%.0f/tCO2e (%s) in %.1f ms",
            _round2(annual_exposure), carbon_price, severity, elapsed_ms,
        )
        return result

    # -------------------------------------------------------------------
    # Public -- quantify_physical_risk
    # -------------------------------------------------------------------

    def quantify_physical_risk(
        self,
        supplier_locations: List[SupplierLocation],
        hazard_types: Optional[List[str]] = None,
        duration_days: int = 30,
    ) -> PhysicalRiskResult:
        """Quantify physical risk from climate hazards at supplier locations.

        Args:
            supplier_locations: Supplier location data.
            hazard_types: Hazard types to assess (default: all).
            duration_days: Average disruption duration in days.

        Returns:
            PhysicalRiskResult with expected annual loss.
        """
        start_ms = time.time()
        hazards = hazard_types or [h.value for h in PhysicalHazardType]
        duration_factor = _decimal(duration_days) / Decimal("365")

        total_revenue = Decimal("0")
        weighted_prob = Decimal("0")
        by_hazard: Dict[str, Decimal] = {}
        by_location: Dict[str, Decimal] = {}

        for loc in supplier_locations:
            rev = _decimal(loc.revenue_at_risk)
            total_revenue += rev
            region_factors = REGIONAL_EXPOSURE.get(
                loc.region, REGIONAL_EXPOSURE["global_average"]
            )

            loc_expected_loss = Decimal("0")
            for hazard in hazards:
                base_prob = _decimal(
                    HAZARD_BASE_PROBABILITIES.get(hazard, 0.05)
                )
                exposure = _decimal(region_factors.get(hazard, 1.0))
                vulnerability = Decimal("1.0")
                if hazard in loc.hazard_vulnerabilities:
                    vulnerability = Decimal("1.5")

                prob = base_prob * exposure * vulnerability
                loss = prob * rev * duration_factor

                by_hazard[hazard] = by_hazard.get(hazard, Decimal("0")) + loss
                loc_expected_loss += loss

            loc_key = f"{loc.supplier_id}_{loc.location_name}"
            by_location[loc_key] = loc_expected_loss

        total_expected = sum(by_hazard.values())
        if total_revenue > Decimal("0"):
            weighted_prob_val = _safe_divide(total_expected, total_revenue * duration_factor)
        else:
            weighted_prob_val = Decimal("0")

        # Severity.
        loss_pct = _safe_pct(total_expected, total_revenue) if total_revenue > Decimal("0") else Decimal("0")
        if loss_pct >= Decimal("5"):
            severity = RiskSeverity.CRITICAL.value
        elif loss_pct >= Decimal("2"):
            severity = RiskSeverity.HIGH.value
        elif loss_pct >= Decimal("1"):
            severity = RiskSeverity.MEDIUM.value
        else:
            severity = RiskSeverity.LOW.value

        result = PhysicalRiskResult(
            total_locations_assessed=len(supplier_locations),
            total_revenue_at_risk=_round2(total_revenue),
            weighted_disruption_probability=_round4(weighted_prob_val),
            expected_annual_loss=_round2(total_expected),
            by_hazard={k: _round2(v) for k, v in by_hazard.items()},
            by_location={k: _round2(v) for k, v in by_location.items()},
            severity=severity,
            calculated_at=utcnow().isoformat(),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed_ms = (time.time() - start_ms) * 1000
        logger.info(
            "Physical risk: $%.0f expected annual loss across %d locations in %.1f ms",
            _round2(total_expected), len(supplier_locations), elapsed_ms,
        )
        return result

    # -------------------------------------------------------------------
    # Public -- assess_opportunities
    # -------------------------------------------------------------------

    def assess_opportunities(
        self,
        scope3_data: List[Scope3CategoryData],
        market_data: Optional[MarketData] = None,
        revenue: float = 0.0,
    ) -> List[OpportunityResult]:
        """Assess low-carbon opportunities from Scope 3 reduction.

        Identifies opportunities in resource efficiency, energy source
        transition, product/service innovation, market access, and
        resilience building.

        Args:
            scope3_data: Scope 3 category emissions.
            market_data: Market context data.
            revenue: Company revenue.

        Returns:
            List of OpportunityResult.
        """
        start_ms = time.time()
        mkt = market_data or MarketData()
        rev_d = _decimal(revenue)
        total_tco2e = sum(_decimal(c.tco2e) for c in scope3_data)
        total_spend = sum(_decimal(c.spend) for c in scope3_data)

        opportunities: List[OpportunityResult] = []

        # 1. Resource efficiency (reduce material/energy inputs).
        efficiency_savings = total_spend * Decimal("0.05")  # 5% spend reduction potential.
        opportunities.append(OpportunityResult(
            opportunity_type=OpportunityType.RESOURCE_EFFICIENCY.value,
            description="Material and energy efficiency improvements across supply chain",
            estimated_annual_value=_round2(efficiency_savings),
            confidence="medium",
            investment_required=_round2(efficiency_savings * Decimal("2")),
            payback_years=_round2(Decimal("2")),
            calculated_at=utcnow().isoformat(),
        ))

        # 2. Energy source (renewable energy in supply chain).
        energy_cats = [c for c in scope3_data if c.category in (3, 4, 9)]
        if energy_cats:
            energy_tco2e = sum(_decimal(c.tco2e) for c in energy_cats)
            re_value = energy_tco2e * Decimal("30")  # $30/tCO2e avoided cost.
            opportunities.append(OpportunityResult(
                opportunity_type=OpportunityType.ENERGY_SOURCE.value,
                description="Renewable energy transition in transport and energy categories",
                estimated_annual_value=_round2(re_value),
                confidence="medium",
                investment_required=_round2(re_value * Decimal("5")),
                payback_years=_round2(Decimal("5")),
                calculated_at=utcnow().isoformat(),
            ))

        # 3. Products/services (low-carbon product premium).
        growth = _decimal(mkt.low_carbon_market_growth_pct) / Decimal("100")
        premium = _decimal(mkt.green_premium_pct) / Decimal("100")
        wtp = _decimal(mkt.customer_willingness_to_pay_pct) / Decimal("100")
        product_value = rev_d * premium * wtp
        opportunities.append(OpportunityResult(
            opportunity_type=OpportunityType.PRODUCTS_SERVICES.value,
            description="Low-carbon product differentiation and green premium capture",
            estimated_annual_value=_round2(product_value),
            confidence="low" if product_value < rev_d * Decimal("0.01") else "medium",
            investment_required=_round2(product_value * Decimal("3")),
            payback_years=_round2(Decimal("3")),
            calculated_at=utcnow().isoformat(),
        ))

        # 4. Markets (access new low-carbon markets).
        market_value = rev_d * growth * Decimal("0.1")  # Capture 10% of growth.
        opportunities.append(OpportunityResult(
            opportunity_type=OpportunityType.MARKETS.value,
            description="Access to growing low-carbon markets and green procurement",
            estimated_annual_value=_round2(market_value),
            confidence="low",
            calculated_at=utcnow().isoformat(),
        ))

        # 5. Resilience (supply chain de-risking value).
        resilience_value = total_spend * Decimal("0.02")  # 2% of spend.
        opportunities.append(OpportunityResult(
            opportunity_type=OpportunityType.RESILIENCE.value,
            description="Supply chain resilience through diversification and decarbonisation",
            estimated_annual_value=_round2(resilience_value),
            confidence="medium",
            calculated_at=utcnow().isoformat(),
        ))

        for opp in opportunities:
            opp.provenance_hash = _compute_hash(opp)

        elapsed_ms = (time.time() - start_ms) * 1000
        total_opp = sum(_decimal(o.estimated_annual_value) for o in opportunities)
        logger.info(
            "Identified %d opportunities worth $%.0f/yr in %.1f ms",
            len(opportunities), _round2(total_opp), elapsed_ms,
        )
        return opportunities

    # -------------------------------------------------------------------
    # Public -- calculate_carbon_pricing_exposure
    # -------------------------------------------------------------------

    def calculate_carbon_pricing_exposure(
        self,
        scope3_tco2e: float,
        price_scenarios: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Calculate carbon pricing exposure across multiple price scenarios.

        Args:
            scope3_tco2e: Total Scope 3 emissions.
            price_scenarios: Custom price scenarios or use defaults.

        Returns:
            Dict mapping scenario name to annual cost.
        """
        scenarios = price_scenarios or CARBON_PRICE_SCENARIOS
        tco2e_d = _decimal(scope3_tco2e)
        result: Dict[str, float] = {}
        for name, price in scenarios.items():
            result[name] = _round2(tco2e_d * _decimal(price))
        return result

    # -------------------------------------------------------------------
    # Public -- model_cbam_exposure
    # -------------------------------------------------------------------

    def model_cbam_exposure(
        self,
        imports: List[ImportItem],
        carbon_price: float = 100.0,
    ) -> CBAMExposure:
        """Model CBAM border adjustment cost.

        Calculates CBAM-equivalent cost for each import product using
        embedded emissions and the applicable carbon price.

        Args:
            imports: List of import items.
            carbon_price: Carbon price for CBAM calculation.

        Returns:
            CBAMExposure with per-product breakdown.
        """
        start_ms = time.time()
        price_d = _decimal(carbon_price)
        total_cost = Decimal("0")
        total_emissions = Decimal("0")
        by_product: Dict[str, Decimal] = {}

        for item in imports:
            qty = _decimal(item.quantity_tonnes)
            if item.embedded_emissions_tco2e is not None:
                embedded = _decimal(item.embedded_emissions_tco2e)
            else:
                rate = item.cbam_rate_override or CBAM_DEFAULT_RATES.get(item.product_type, 0.5)
                embedded = qty * _decimal(rate)

            cost = embedded * price_d
            total_cost += cost
            total_emissions += embedded
            by_product[item.product_type] = (
                by_product.get(item.product_type, Decimal("0")) + cost
            )

        result = CBAMExposure(
            total_cbam_cost=_round2(total_cost),
            by_product={k: _round2(v) for k, v in by_product.items()},
            total_embedded_emissions=_round2(total_emissions),
            calculated_at=utcnow().isoformat(),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed_ms = (time.time() - start_ms) * 1000
        logger.info(
            "CBAM exposure: $%.0f across %d imports in %.1f ms",
            _round2(total_cost), len(imports), elapsed_ms,
        )
        return result

    # -------------------------------------------------------------------
    # Public -- assess_stranded_assets
    # -------------------------------------------------------------------

    def assess_stranded_assets(
        self,
        supplier_assets: List[SupplierAsset],
        carbon_price: float = 100.0,
        nze_deadline_year: int = 2050,
    ) -> List[StrandedAssetResult]:
        """Assess high-carbon asset stranding risk for suppliers.

        Estimates impairment based on remaining useful life vs.
        net-zero deadline and carbon cost of continued operation.

        Args:
            supplier_assets: Supplier asset data.
            carbon_price: Current/projected carbon price.
            nze_deadline_year: Net-zero target year.

        Returns:
            List of StrandedAssetResult.
        """
        start_ms = time.time()
        current_year = utcnow().year
        results: List[StrandedAssetResult] = []

        for asset in supplier_assets:
            asset_end_year = current_year + asset.remaining_useful_life_years
            years_beyond_nze = max(0, asset_end_year - nze_deadline_year)

            # Impairment based on years of operation beyond NZE deadline.
            if asset.remaining_useful_life_years > 0:
                impairment_pct = _safe_divide(
                    _decimal(years_beyond_nze),
                    _decimal(asset.remaining_useful_life_years),
                ) * Decimal("100")
            else:
                impairment_pct = Decimal("0")

            # Additional impairment from carbon cost.
            annual_carbon_cost = _decimal(asset.annual_emissions_tco2e) * _decimal(carbon_price)
            lifetime_carbon_cost = annual_carbon_cost * _decimal(asset.remaining_useful_life_years)
            carbon_impairment_pct = _safe_pct(lifetime_carbon_cost, _decimal(asset.asset_value))

            total_impairment_pct = min(
                impairment_pct + carbon_impairment_pct * Decimal("0.5"),
                Decimal("100"),
            )
            impairment_value = _decimal(asset.asset_value) * total_impairment_pct / Decimal("100")

            # Years to stranding.
            if annual_carbon_cost > Decimal("0") and _decimal(asset.asset_value) > Decimal("0"):
                years_to_strand = float(_safe_divide(
                    _decimal(asset.asset_value), annual_carbon_cost,
                ))
            else:
                years_to_strand = float(asset.remaining_useful_life_years)

            result = StrandedAssetResult(
                supplier_id=asset.supplier_id,
                supplier_name=asset.supplier_name,
                asset_value_at_risk=_round2(asset.asset_value),
                impairment_pct=_round2(total_impairment_pct),
                impairment_value=_round2(impairment_value),
                years_to_stranding=_round2(years_to_strand),
                calculated_at=utcnow().isoformat(),
            )
            result.provenance_hash = _compute_hash(result)
            results.append(result)

        elapsed_ms = (time.time() - start_ms) * 1000
        total_impairment = sum(_decimal(r.impairment_value) for r in results)
        logger.info(
            "Stranded assets: $%.0f total impairment across %d assets in %.1f ms",
            _round2(total_impairment), len(results), elapsed_ms,
        )
        return results

    # -------------------------------------------------------------------
    # Public -- calculate_financial_npv
    # -------------------------------------------------------------------

    def calculate_financial_npv(
        self,
        annual_risk_cost: float,
        annual_opportunity_value: float,
        discount_rate: float = 0.08,
        horizons: Optional[List[int]] = None,
    ) -> List[FinancialNPV]:
        """Calculate NPV of risks and opportunities over multiple horizons.

        Args:
            annual_risk_cost: Annual risk cost.
            annual_opportunity_value: Annual opportunity value.
            discount_rate: Discount rate.
            horizons: List of horizons in years (default: [10, 20, 30]).

        Returns:
            List of FinancialNPV for each horizon.
        """
        start_ms = time.time()
        horizons = horizons or [10, 20, 30]
        rate_d = _decimal(discount_rate)
        risk_d = _decimal(annual_risk_cost)
        opp_d = _decimal(annual_opportunity_value)
        results: List[FinancialNPV] = []

        for horizon in horizons:
            npv_risk = Decimal("0")
            npv_opp = Decimal("0")
            risk_traj: Dict[int, float] = {}
            opp_traj: Dict[int, float] = {}

            for yr in range(1, horizon + 1):
                df = Decimal("1") / (Decimal("1") + rate_d) ** _decimal(yr)
                pv_risk = risk_d * df
                pv_opp = opp_d * df
                npv_risk += pv_risk
                npv_opp += pv_opp
                risk_traj[yr] = _round2(pv_risk)
                opp_traj[yr] = _round2(pv_opp)

            net = npv_opp - npv_risk
            result = FinancialNPV(
                horizon_years=horizon,
                discount_rate=_round4(rate_d),
                npv_risks=_round2(npv_risk),
                npv_opportunities=_round2(npv_opp),
                net_npv=_round2(net),
                annual_risk_trajectory=risk_traj,
                annual_opp_trajectory=opp_traj,
                calculated_at=utcnow().isoformat(),
            )
            result.provenance_hash = _compute_hash(result)
            results.append(result)

        elapsed_ms = (time.time() - start_ms) * 1000
        logger.info(
            "Financial NPV: %d horizons computed in %.1f ms",
            len(results), elapsed_ms,
        )
        return results

    # -------------------------------------------------------------------
    # Public -- run_scenario_analysis
    # -------------------------------------------------------------------

    def run_scenario_analysis(
        self,
        scope3_data: List[Scope3CategoryData],
        scenarios: Optional[List[str]] = None,
        revenue: float = 0.0,
        supplier_locations: Optional[List[SupplierLocation]] = None,
    ) -> List[ScenarioResult]:
        """Run multi-scenario analysis across IEA and NGFS pathways.

        For each scenario, calculates transition risk exposure at the
        scenario's carbon price, physical risk exposure at the scenario's
        physical risk multiplier, and identifies opportunities.

        Args:
            scope3_data: Scope 3 category emissions.
            scenarios: Scenario types to run (default: all 6).
            revenue: Company revenue.
            supplier_locations: Supplier locations for physical risk.

        Returns:
            List of ScenarioResult.
        """
        start_ms = time.time()
        all_scenarios = {**NGFS_SCENARIOS, **IEA_SCENARIOS}
        scenario_types = scenarios or list(all_scenarios.keys())
        total_tco2e = sum(_decimal(c.tco2e) for c in scope3_data)

        results: List[ScenarioResult] = []

        for scenario_key in scenario_types:
            params = all_scenarios.get(scenario_key)
            if params is None:
                logger.warning("Unknown scenario: %s, skipping", scenario_key)
                continue

            cp_2030 = _decimal(params.get("carbon_price_2030", 100))
            cp_2050 = _decimal(params.get("carbon_price_2050", 200))
            phys_mult = _decimal(params.get("physical_risk_multiplier", 1.0))
            trans_mult = _decimal(params.get("transition_risk_multiplier", 1.0))
            temp = params.get("temperature_2100", 2.0)

            # Transition risk at 2030 price.
            transition_exposure = total_tco2e * cp_2030 * trans_mult

            # Physical risk (simplified).
            physical_exposure = Decimal("0")
            if supplier_locations:
                base_loss = Decimal("0")
                for loc in supplier_locations:
                    base_loss += _decimal(loc.revenue_at_risk) * Decimal("0.02")
                physical_exposure = base_loss * phys_mult

            total_risk = transition_exposure + physical_exposure

            # Opportunities scale with transition stringency.
            opp_value = total_tco2e * Decimal("20") * trans_mult  # $20/tCO2e opportunity baseline.

            net_impact = total_risk - opp_value

            result = ScenarioResult(
                scenario=scenario_key if isinstance(scenario_key, str) else scenario_key.value,
                scenario_name=params.get("name", str(scenario_key)),
                temperature_outcome=float(temp),
                carbon_price_2030=_round2(cp_2030),
                carbon_price_2050=_round2(cp_2050),
                transition_risk_exposure=_round2(transition_exposure),
                physical_risk_exposure=_round2(physical_exposure),
                total_risk=_round2(total_risk),
                opportunities=_round2(opp_value),
                net_impact=_round2(net_impact),
                description=params.get("description", ""),
                calculated_at=utcnow().isoformat(),
            )
            result.provenance_hash = _compute_hash(result)
            results.append(result)

        elapsed_ms = (time.time() - start_ms) * 1000
        logger.info(
            "Scenario analysis: %d scenarios completed in %.1f ms",
            len(results), elapsed_ms,
        )
        return results

    # -------------------------------------------------------------------
    # Public -- _compute_provenance
    # -------------------------------------------------------------------

    @staticmethod
    def _compute_provenance(data: Any) -> str:
        """Compute SHA-256 provenance hash for audit trail.

        Args:
            data: Data to hash.

        Returns:
            SHA-256 hex digest (64 characters).
        """
        return _compute_hash(data)

# ---------------------------------------------------------------------------
# Pydantic v2 model_rebuild for forward-reference resolution
# ---------------------------------------------------------------------------

Scope3CategoryData.model_rebuild()
SupplierLocation.model_rebuild()
ImportItem.model_rebuild()
MarketData.model_rebuild()
SupplierAsset.model_rebuild()
TransitionRiskResult.model_rebuild()
PhysicalRiskResult.model_rebuild()
OpportunityResult.model_rebuild()
CBAMExposure.model_rebuild()
StrandedAssetResult.model_rebuild()
FinancialNPV.model_rebuild()
ScenarioResult.model_rebuild()
