# -*- coding: utf-8 -*-
"""
FinancedEmissionsEngine - PACK-012 CSRD Financial Service Engine 1
====================================================================

PCAF (Partnership for Carbon Accounting Financials) methodology for
calculating financed emissions -- Scope 3 Category 15 under the GHG
Protocol.  Financial institutions attribute a share of their investees'
and borrowers' emissions proportional to their financial exposure.

Asset Classes Supported (PCAF Global Standard, 2nd Edition):
    1. Listed Equity & Corporate Bonds
    2. Business Loans & Unlisted Equity
    3. Project Finance
    4. Commercial Real Estate
    5. Mortgages
    6. Motor Vehicle Loans
    7. Sovereign Bonds
    8. Securitizations (asset-backed)
    9. Sub-Sovereign / Municipal Debt
   10. Green / Sustainability Bonds (pass-through)

Core Formulas:
    Attribution Factor  = Outstanding Amount / EVIC (or Total Equity + Debt)
    Financed Emissions  = Attribution Factor * Borrower Emissions (tCO2e)
    WACI               = SUM(weight_i * intensity_i)
    Data Quality Score  = SUM(weight_i * dq_i) across portfolio

Regulatory References:
    - PCAF Global GHG Accounting & Reporting Standard (2nd Ed., 2022)
    - EU Delegated Regulation 2021/2178 (Pillar 3 / GAR)
    - ESRS E1-6 (Financed GHG emissions, financial institutions)
    - GHG Protocol Scope 3, Category 15 (Investments)

Zero-Hallucination:
    - All calculations use deterministic Python arithmetic
    - Attribution factors are pure ratios of financial data
    - Data quality scores use PCAF 1-5 scoring rubric (deterministic)
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-012 CSRD Financial Service
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
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

def _round_val(value: float, places: int = 4) -> float:
    """Round a float to specified decimal places."""
    return round(value, places)

def _round_decimal(value: float, places: int = 4) -> float:
    """Round using Decimal for regulatory precision."""
    d = Decimal(str(value))
    q = Decimal("0." + "0" * places)
    return float(d.quantize(q, rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PCAFAssetClass(str, Enum):
    """PCAF asset class categories per the Global Standard."""
    LISTED_EQUITY = "listed_equity"
    CORPORATE_BONDS = "corporate_bonds"
    BUSINESS_LOANS = "business_loans"
    PROJECT_FINANCE = "project_finance"
    COMMERCIAL_REAL_ESTATE = "commercial_real_estate"
    MORTGAGES = "mortgages"
    MOTOR_VEHICLE_LOANS = "motor_vehicle_loans"
    SOVEREIGN_BONDS = "sovereign_bonds"
    SECURITIZATIONS = "securitizations"
    SUB_SOVEREIGN_DEBT = "sub_sovereign_debt"

class DataQualityLevel(str, Enum):
    """PCAF data quality score levels (1 = best, 5 = worst)."""
    SCORE_1 = "score_1"  # Reported, verified emissions
    SCORE_2 = "score_2"  # Reported, unverified emissions
    SCORE_3 = "score_3"  # Physical activity-based estimates
    SCORE_4 = "score_4"  # Economic activity-based estimates
    SCORE_5 = "score_5"  # Estimated using sector averages

    @property
    def numeric(self) -> int:
        """Return integer score value."""
        return int(self.value.split("_")[1])

class CurrencyCode(str, Enum):
    """Common currency codes for multi-currency normalization."""
    EUR = "EUR"
    USD = "USD"
    GBP = "GBP"
    CHF = "CHF"
    JPY = "JPY"
    CNY = "CNY"
    SEK = "SEK"
    DKK = "DKK"
    NOK = "NOK"
    AUD = "AUD"
    CAD = "CAD"

class EmissionScope(str, Enum):
    """GHG Protocol emission scopes."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"
    SCOPE_1_2 = "scope_1_2"
    SCOPE_1_2_3 = "scope_1_2_3"

# ---------------------------------------------------------------------------
# Attribution Factor Rules per Asset Class
# ---------------------------------------------------------------------------

ATTRIBUTION_RULES: Dict[str, Dict[str, Any]] = {
    PCAFAssetClass.LISTED_EQUITY.value: {
        "denominator": "evic",
        "description": "Outstanding Amount / Enterprise Value Including Cash (EVIC)",
        "fallback_denominator": "total_equity_plus_debt",
    },
    PCAFAssetClass.CORPORATE_BONDS.value: {
        "denominator": "evic",
        "description": "Outstanding Amount / EVIC",
        "fallback_denominator": "total_equity_plus_debt",
    },
    PCAFAssetClass.BUSINESS_LOANS.value: {
        "denominator": "total_equity_plus_debt",
        "description": "Outstanding Amount / (Total Equity + Debt)",
        "fallback_denominator": "total_assets",
    },
    PCAFAssetClass.PROJECT_FINANCE.value: {
        "denominator": "total_project_cost",
        "description": "Outstanding Amount / Total Project Cost",
        "fallback_denominator": "total_equity_plus_debt",
    },
    PCAFAssetClass.COMMERCIAL_REAL_ESTATE.value: {
        "denominator": "property_value",
        "description": "Outstanding Amount / Property Value at Origination",
        "fallback_denominator": "total_assets",
    },
    PCAFAssetClass.MORTGAGES.value: {
        "denominator": "property_value",
        "description": "Outstanding Amount / Property Value at Origination",
        "fallback_denominator": None,
    },
    PCAFAssetClass.MOTOR_VEHICLE_LOANS.value: {
        "denominator": "vehicle_value",
        "description": "Outstanding Amount / Vehicle Value at Origination",
        "fallback_denominator": None,
    },
    PCAFAssetClass.SOVEREIGN_BONDS.value: {
        "denominator": "government_debt",
        "description": "Outstanding Amount / PPP-Adjusted GDP",
        "fallback_denominator": "gdp_nominal",
    },
    PCAFAssetClass.SECURITIZATIONS.value: {
        "denominator": "total_pool_value",
        "description": "Outstanding Amount / Total Securitization Pool Value",
        "fallback_denominator": None,
    },
    PCAFAssetClass.SUB_SOVEREIGN_DEBT.value: {
        "denominator": "total_revenue_or_budget",
        "description": "Outstanding Amount / Total Revenue or Budget",
        "fallback_denominator": "gdp_regional",
    },
}

# Default data quality score descriptions for audit trail
DATA_QUALITY_DESCRIPTIONS: Dict[int, str] = {
    1: "Reported emissions, verified by third party (highest quality)",
    2: "Reported emissions, not externally verified",
    3: "Estimated from physical activity data (e.g., energy use, floor area)",
    4: "Estimated from economic activity data (e.g., revenue, assets)",
    5: "Estimated using sector averages only (lowest quality)",
}

# ---------------------------------------------------------------------------
# Pydantic Data Models
# ---------------------------------------------------------------------------

class CurrencyRate(BaseModel):
    """Exchange rate for currency normalization.

    All amounts are normalized to a single reporting currency before
    attribution calculations.

    Attributes:
        source_currency: ISO 4217 currency code of the source.
        target_currency: ISO 4217 currency code of the target.
        rate: Exchange rate (source -> target).
        rate_date: Date the rate was observed.
    """
    source_currency: str = Field(description="Source currency code (ISO 4217)")
    target_currency: str = Field(default="EUR", description="Target currency code")
    rate: float = Field(gt=0.0, description="Exchange rate (source -> target)")
    rate_date: str = Field(default="", description="Rate observation date (YYYY-MM-DD)")

class DataQualityScore(BaseModel):
    """PCAF data quality assessment for a single holding.

    Captures the data quality score (1-5), the methodology used,
    and the justification for the score assignment.

    Attributes:
        holding_id: Unique holding identifier.
        asset_class: PCAF asset class of the holding.
        score: Data quality score (1 = best, 5 = worst).
        level: Data quality level enum.
        methodology: Description of the methodology used.
        justification: Reason for the assigned score.
        emission_data_source: Source of emission data.
        is_estimated: Whether the emissions are estimated.
    """
    holding_id: str = Field(default="", description="Holding identifier")
    asset_class: PCAFAssetClass = Field(
        default=PCAFAssetClass.LISTED_EQUITY,
        description="PCAF asset class",
    )
    score: int = Field(default=5, ge=1, le=5, description="PCAF data quality score (1-5)")
    level: DataQualityLevel = Field(
        default=DataQualityLevel.SCORE_5,
        description="Data quality level",
    )
    methodology: str = Field(default="", description="Methodology description")
    justification: str = Field(default="", description="Score justification")
    emission_data_source: str = Field(default="", description="Source of emission data")
    is_estimated: bool = Field(default=True, description="Whether emissions are estimated")

    @model_validator(mode="after")
    def _sync_level(self) -> "DataQualityScore":
        """Ensure level matches numeric score."""
        expected = f"score_{self.score}"
        if self.level.value != expected:
            self.level = DataQualityLevel(expected)
        return self

class AssetClassData(BaseModel):
    """Input data for a single holding / exposure in the portfolio.

    Contains all financial and emissions data needed to compute
    financed emissions for one holding under the PCAF methodology.

    Attributes:
        holding_id: Unique identifier for this holding.
        holding_name: Name of the borrower / investee / entity.
        asset_class: PCAF asset class classification.
        outstanding_amount: Outstanding loan / investment amount.
        currency: Currency of outstanding_amount.
        evic: Enterprise Value Including Cash (for equity/bonds).
        total_equity_plus_debt: Total book equity plus total debt.
        total_assets: Total assets of the borrower.
        total_project_cost: Total cost for project finance.
        property_value: Property value for CRE / mortgages.
        vehicle_value: Vehicle value for motor vehicle loans.
        government_debt: Total government debt for sovereign bonds.
        gdp_nominal: Nominal GDP for sovereign bonds.
        total_pool_value: Total securitization pool value.
        total_revenue_or_budget: Revenue/budget for sub-sovereign.
        gdp_regional: Regional GDP for sub-sovereign fallback.
        scope1_emissions: Scope 1 emissions of the borrower (tCO2e).
        scope2_emissions: Scope 2 emissions of the borrower (tCO2e).
        scope3_emissions: Scope 3 emissions of the borrower (tCO2e).
        revenue: Annual revenue of the borrower.
        sector: Sector classification (NACE / GICS).
        country: Country of domicile (ISO 3166-1).
        data_quality_score: PCAF data quality score (1-5).
        emission_year: Year the emission data refers to.
        is_green_bond: Whether the holding is a green/sustainability bond.
        green_bond_use_of_proceeds_pct: Pct of proceeds for green activities.
    """
    holding_id: str = Field(default_factory=_new_uuid, description="Unique holding ID")
    holding_name: str = Field(default="", description="Borrower / investee name")
    asset_class: PCAFAssetClass = Field(description="PCAF asset class")
    outstanding_amount: float = Field(
        default=0.0, ge=0.0,
        description="Outstanding loan / investment amount",
    )
    currency: str = Field(default="EUR", description="Currency of outstanding_amount")
    # Denominators (only relevant fields per asset class)
    evic: float = Field(default=0.0, ge=0.0, description="EVIC (listed equity / bonds)")
    total_equity_plus_debt: float = Field(
        default=0.0, ge=0.0, description="Total equity + debt",
    )
    total_assets: float = Field(default=0.0, ge=0.0, description="Total assets")
    total_project_cost: float = Field(
        default=0.0, ge=0.0, description="Total project cost",
    )
    property_value: float = Field(
        default=0.0, ge=0.0, description="Property value at origination",
    )
    vehicle_value: float = Field(
        default=0.0, ge=0.0, description="Vehicle value at origination",
    )
    government_debt: float = Field(
        default=0.0, ge=0.0, description="Total government debt",
    )
    gdp_nominal: float = Field(default=0.0, ge=0.0, description="Nominal GDP")
    total_pool_value: float = Field(
        default=0.0, ge=0.0, description="Total securitization pool value",
    )
    total_revenue_or_budget: float = Field(
        default=0.0, ge=0.0, description="Revenue / budget (sub-sovereign)",
    )
    gdp_regional: float = Field(default=0.0, ge=0.0, description="Regional GDP")
    # Emissions
    scope1_emissions: float = Field(
        default=0.0, ge=0.0, description="Scope 1 emissions (tCO2e)",
    )
    scope2_emissions: float = Field(
        default=0.0, ge=0.0, description="Scope 2 emissions (tCO2e)",
    )
    scope3_emissions: float = Field(
        default=0.0, ge=0.0, description="Scope 3 emissions (tCO2e)",
    )
    revenue: float = Field(default=0.0, ge=0.0, description="Annual revenue")
    sector: str = Field(default="", description="Sector classification (NACE/GICS)")
    country: str = Field(default="", description="Country of domicile (ISO 3166)")
    data_quality_score: int = Field(
        default=5, ge=1, le=5, description="PCAF data quality score (1-5)",
    )
    emission_year: int = Field(default=2024, description="Year of emission data")
    is_green_bond: bool = Field(default=False, description="Green / sustainability bond")
    green_bond_use_of_proceeds_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of proceeds allocated to green activities",
    )

class AttributionResult(BaseModel):
    """Result of attribution factor calculation for a single holding.

    Contains the computed attribution factor, the denominator used,
    and any fallback logic that was applied.

    Attributes:
        holding_id: Holding identifier.
        asset_class: PCAF asset class.
        outstanding_amount_eur: Outstanding amount in EUR.
        denominator_value: Denominator value used in attribution.
        denominator_field: Name of the denominator field used.
        attribution_factor: Calculated attribution factor.
        used_fallback: Whether a fallback denominator was used.
        provenance_hash: SHA-256 provenance hash.
    """
    holding_id: str = Field(default="", description="Holding identifier")
    asset_class: PCAFAssetClass = Field(description="PCAF asset class")
    outstanding_amount_eur: float = Field(
        default=0.0, description="Outstanding amount in EUR",
    )
    denominator_value: float = Field(
        default=0.0, description="Denominator value used",
    )
    denominator_field: str = Field(default="", description="Denominator field name")
    attribution_factor: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Attribution factor (0.0 to 1.0)",
    )
    used_fallback: bool = Field(default=False, description="Whether fallback was used")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class HoldingEmissions(BaseModel):
    """Financed emissions for a single holding.

    Contains the full breakdown of attributed emissions per scope,
    the data quality assessment, and the attribution details.

    Attributes:
        holding_id: Holding identifier.
        holding_name: Borrower / investee name.
        asset_class: PCAF asset class.
        attribution_factor: Attribution factor used.
        financed_scope1: Financed Scope 1 (tCO2e).
        financed_scope2: Financed Scope 2 (tCO2e).
        financed_scope3: Financed Scope 3 (tCO2e).
        financed_scope1_2: Financed Scope 1+2 (tCO2e).
        financed_total: Total financed emissions (tCO2e).
        carbon_intensity: Carbon intensity (tCO2e / EUR M).
        weight_pct: Portfolio weight.
        weighted_intensity: Weight * intensity for WACI.
        data_quality: Data quality assessment.
        attribution: Attribution calculation details.
        provenance_hash: SHA-256 provenance hash.
    """
    holding_id: str = Field(default="", description="Holding identifier")
    holding_name: str = Field(default="", description="Borrower / investee name")
    asset_class: PCAFAssetClass = Field(description="PCAF asset class")
    attribution_factor: float = Field(
        default=0.0, description="Attribution factor used",
    )
    financed_scope1: float = Field(default=0.0, description="Financed Scope 1 (tCO2e)")
    financed_scope2: float = Field(default=0.0, description="Financed Scope 2 (tCO2e)")
    financed_scope3: float = Field(default=0.0, description="Financed Scope 3 (tCO2e)")
    financed_scope1_2: float = Field(
        default=0.0, description="Financed Scope 1+2 (tCO2e)",
    )
    financed_total: float = Field(default=0.0, description="Total financed emissions (tCO2e)")
    carbon_intensity: float = Field(
        default=0.0, description="Carbon intensity (tCO2e / EUR M revenue)",
    )
    weight_pct: float = Field(default=0.0, description="Portfolio weight (%)")
    weighted_intensity: float = Field(
        default=0.0, description="weight * intensity for WACI",
    )
    data_quality: Optional[DataQualityScore] = Field(
        default=None, description="Data quality assessment",
    )
    attribution: Optional[AttributionResult] = Field(
        default=None, description="Attribution calculation details",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class EmissionsByAssetClass(BaseModel):
    """Aggregated financed emissions for a single PCAF asset class.

    Provides totals, averages, and data quality for all holdings
    within one asset class.

    Attributes:
        asset_class: PCAF asset class.
        holding_count: Number of holdings in this class.
        total_outstanding_eur: Total outstanding exposure (EUR).
        total_financed_scope1: Total financed Scope 1.
        total_financed_scope2: Total financed Scope 2.
        total_financed_scope3: Total financed Scope 3.
        total_financed_scope1_2: Total financed Scope 1+2.
        total_financed_emissions: Total financed emissions.
        weighted_avg_intensity: WACI for this asset class.
        avg_data_quality_score: Average data quality score.
        weight_in_portfolio_pct: Weight of this class in portfolio.
    """
    asset_class: PCAFAssetClass = Field(description="PCAF asset class")
    holding_count: int = Field(default=0, ge=0, description="Number of holdings")
    total_outstanding_eur: float = Field(
        default=0.0, description="Total outstanding exposure (EUR)",
    )
    total_financed_scope1: float = Field(default=0.0, description="Total financed Scope 1")
    total_financed_scope2: float = Field(default=0.0, description="Total financed Scope 2")
    total_financed_scope3: float = Field(default=0.0, description="Total financed Scope 3")
    total_financed_scope1_2: float = Field(default=0.0, description="Total financed Scope 1+2")
    total_financed_emissions: float = Field(
        default=0.0, description="Total financed emissions",
    )
    weighted_avg_intensity: float = Field(
        default=0.0, description="WACI for this asset class",
    )
    avg_data_quality_score: float = Field(
        default=5.0, ge=1.0, le=5.0,
        description="Average PCAF data quality score",
    )
    weight_in_portfolio_pct: float = Field(
        default=0.0, description="Weight of this class in portfolio (%)",
    )

class YoYTrajectory(BaseModel):
    """Year-over-year emissions trajectory data point.

    Tracks the change in financed emissions between consecutive years
    for the same portfolio.

    Attributes:
        year: Reporting year.
        total_financed_emissions: Total financed emissions for the year.
        total_outstanding_eur: Total outstanding for the year.
        portfolio_waci: Portfolio WACI.
        yoy_change_pct: Year-over-year change in financed emissions (%).
        yoy_change_absolute: Absolute change in financed emissions (tCO2e).
    """
    year: int = Field(description="Reporting year")
    total_financed_emissions: float = Field(
        default=0.0, description="Total financed emissions (tCO2e)",
    )
    total_outstanding_eur: float = Field(
        default=0.0, description="Total outstanding for the year (EUR)",
    )
    portfolio_waci: float = Field(default=0.0, description="Portfolio WACI")
    yoy_change_pct: float = Field(
        default=0.0, description="Year-over-year change (%)",
    )
    yoy_change_absolute: float = Field(
        default=0.0, description="Absolute change (tCO2e)",
    )

class PortfolioEmissionsResult(BaseModel):
    """Complete portfolio-level financed emissions result.

    The top-level result object containing totals, breakdowns by
    asset class, individual holding results, data quality summary,
    and year-over-year trajectory.

    Attributes:
        result_id: Unique result identifier.
        reporting_year: Reporting period year.
        reporting_currency: Reporting currency (ISO 4217).
        total_portfolio_outstanding_eur: Total portfolio outstanding (EUR).
        total_financed_scope1: Portfolio total financed Scope 1.
        total_financed_scope2: Portfolio total financed Scope 2.
        total_financed_scope3: Portfolio total financed Scope 3.
        total_financed_scope1_2: Portfolio total financed Scope 1+2.
        total_financed_emissions: Portfolio total financed emissions.
        portfolio_waci: Portfolio weighted average carbon intensity.
        weighted_avg_data_quality: Portfolio weighted average data quality.
        data_quality_coverage_pct: Pct of portfolio with DQ score < 5.
        asset_class_breakdown: Breakdown by PCAF asset class.
        holding_results: Individual holding emission results.
        yoy_trajectory: Year-over-year trajectory points.
        double_counting_adjustments: Adjustments made for double-counting.
        total_holdings: Number of holdings in the portfolio.
        holdings_with_data: Holdings with actual emission data.
        holdings_estimated: Holdings with estimated emissions.
        methodology_notes: Methodology notes for disclosure.
        processing_time_ms: Processing time in milliseconds.
        engine_version: Engine version string.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 provenance hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Unique result identifier")
    reporting_year: int = Field(default=2024, description="Reporting period year")
    reporting_currency: str = Field(default="EUR", description="Reporting currency")
    total_portfolio_outstanding_eur: float = Field(
        default=0.0, description="Total portfolio outstanding (EUR)",
    )
    total_financed_scope1: float = Field(
        default=0.0, description="Portfolio total financed Scope 1 (tCO2e)",
    )
    total_financed_scope2: float = Field(
        default=0.0, description="Portfolio total financed Scope 2 (tCO2e)",
    )
    total_financed_scope3: float = Field(
        default=0.0, description="Portfolio total financed Scope 3 (tCO2e)",
    )
    total_financed_scope1_2: float = Field(
        default=0.0, description="Portfolio total financed Scope 1+2 (tCO2e)",
    )
    total_financed_emissions: float = Field(
        default=0.0, description="Portfolio total financed emissions (tCO2e)",
    )
    portfolio_waci: float = Field(
        default=0.0, description="Portfolio WACI (tCO2e / EUR M)",
    )
    weighted_avg_data_quality: float = Field(
        default=5.0, ge=1.0, le=5.0,
        description="Portfolio weighted average data quality score",
    )
    data_quality_coverage_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Pct of portfolio with data quality score < 5",
    )
    asset_class_breakdown: List[EmissionsByAssetClass] = Field(
        default_factory=list, description="Breakdown by PCAF asset class",
    )
    holding_results: List[HoldingEmissions] = Field(
        default_factory=list, description="Individual holding emission results",
    )
    yoy_trajectory: List[YoYTrajectory] = Field(
        default_factory=list, description="Year-over-year trajectory",
    )
    double_counting_adjustments: Dict[str, float] = Field(
        default_factory=dict, description="Double-counting adjustments applied",
    )
    total_holdings: int = Field(default=0, ge=0, description="Total holdings")
    holdings_with_data: int = Field(default=0, ge=0, description="Holdings with data")
    holdings_estimated: int = Field(default=0, ge=0, description="Holdings estimated")
    methodology_notes: List[str] = Field(
        default_factory=list, description="Methodology notes",
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Calculation timestamp",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------

class FinancedEmissionsConfig(BaseModel):
    """Configuration for the FinancedEmissionsEngine.

    Controls reporting currency, scope inclusion, data quality
    thresholds, and double-counting prevention settings.

    Attributes:
        reporting_currency: Target currency for all amounts.
        include_scope3: Whether to include Scope 3 in financed total.
        scope3_asset_classes: Asset classes where Scope 3 is included.
        max_attribution_factor: Maximum attribution factor (cap at 1.0).
        min_data_quality_for_disclosure: Minimum DQ score for disclosure.
        currency_rates: Exchange rates for multi-currency normalization.
        enable_double_counting_prevention: Enable double-counting checks.
        double_counting_entity_groups: Groups of holdings that share entities.
        reporting_year: Reporting year for the calculation.
        yoy_prior_results: Prior year results for YoY trajectory.
        precision_decimal_places: Decimal places for rounding.
    """
    reporting_currency: str = Field(
        default="EUR", description="Target reporting currency",
    )
    include_scope3: bool = Field(
        default=False,
        description="Whether to include Scope 3 in financed total",
    )
    scope3_asset_classes: List[PCAFAssetClass] = Field(
        default_factory=lambda: [
            PCAFAssetClass.LISTED_EQUITY,
            PCAFAssetClass.CORPORATE_BONDS,
            PCAFAssetClass.BUSINESS_LOANS,
        ],
        description="Asset classes where Scope 3 should be included",
    )
    max_attribution_factor: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Maximum attribution factor (capped)",
    )
    min_data_quality_for_disclosure: int = Field(
        default=5, ge=1, le=5,
        description="Minimum DQ score to include in disclosure (1-5)",
    )
    currency_rates: List[CurrencyRate] = Field(
        default_factory=list,
        description="Exchange rates for multi-currency normalization",
    )
    enable_double_counting_prevention: bool = Field(
        default=True,
        description="Enable double-counting prevention across asset classes",
    )
    double_counting_entity_groups: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Groups of holding IDs that share the same underlying entity",
    )
    reporting_year: int = Field(default=2024, description="Reporting year")
    yoy_prior_results: List[YoYTrajectory] = Field(
        default_factory=list,
        description="Prior year results for year-over-year trajectory",
    )
    precision_decimal_places: int = Field(
        default=4, ge=0, le=10,
        description="Decimal places for rounding",
    )

# ---------------------------------------------------------------------------
# Model rebuilds for forward references
# ---------------------------------------------------------------------------

CurrencyRate.model_rebuild()
DataQualityScore.model_rebuild()
AssetClassData.model_rebuild()
AttributionResult.model_rebuild()
HoldingEmissions.model_rebuild()
EmissionsByAssetClass.model_rebuild()
YoYTrajectory.model_rebuild()
PortfolioEmissionsResult.model_rebuild()
FinancedEmissionsConfig.model_rebuild()

# ---------------------------------------------------------------------------
# FinancedEmissionsEngine
# ---------------------------------------------------------------------------

class FinancedEmissionsEngine:
    """
    Financed emissions calculation engine implementing PCAF methodology.

    Computes Scope 3 Category 15 financed emissions for a portfolio of
    financial exposures across all 10 PCAF asset classes.  The engine
    normalizes currencies, calculates attribution factors per asset class,
    scores data quality, aggregates to portfolio level with double-counting
    prevention, and produces a full disclosure-ready result with SHA-256
    provenance.

    Zero-Hallucination Guarantees:
        - All calculations use deterministic Python arithmetic
        - Attribution factors are pure ratios of financial data
        - Data quality scores use PCAF 1-5 rubric (deterministic)
        - SHA-256 provenance hash on every result
        - No LLM involvement in any calculation path

    Example:
        >>> config = FinancedEmissionsConfig(reporting_currency="EUR")
        >>> engine = FinancedEmissionsEngine(config)
        >>> holdings = [AssetClassData(
        ...     asset_class=PCAFAssetClass.LISTED_EQUITY,
        ...     outstanding_amount=10_000_000,
        ...     evic=100_000_000,
        ...     scope1_emissions=50_000,
        ...     scope2_emissions=20_000,
        ... )]
        >>> result = engine.calculate_portfolio_emissions(holdings)
        >>> assert result.total_financed_emissions > 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize FinancedEmissionsEngine.

        Args:
            config: Optional FinancedEmissionsConfig or dict.
        """
        if config and isinstance(config, dict):
            self.config = FinancedEmissionsConfig(**config)
        elif config and isinstance(config, FinancedEmissionsConfig):
            self.config = config
        else:
            self.config = FinancedEmissionsConfig()

        self._currency_map: Dict[str, float] = self._build_currency_map()
        self._holdings: List[AssetClassData] = []
        self._holding_results: Dict[str, HoldingEmissions] = {}

        logger.info(
            "FinancedEmissionsEngine initialized (version=%s, currency=%s)",
            _MODULE_VERSION,
            self.config.reporting_currency,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_portfolio_emissions(
        self,
        holdings: List[AssetClassData],
    ) -> PortfolioEmissionsResult:
        """Calculate financed emissions for the entire portfolio.

        Processes each holding through the PCAF methodology pipeline:
        1. Currency normalization
        2. Attribution factor calculation
        3. Emission attribution
        4. Data quality scoring
        5. Double-counting prevention
        6. Portfolio aggregation
        7. WACI computation
        8. YoY trajectory

        Args:
            holdings: List of AssetClassData for each exposure.

        Returns:
            PortfolioEmissionsResult with full breakdown.

        Raises:
            ValueError: If holdings list is empty.
        """
        start = utcnow()

        if not holdings:
            raise ValueError("Holdings list cannot be empty")

        self._holdings = holdings
        self._holding_results = {}

        logger.info("Calculating financed emissions for %d holdings", len(holdings))

        # Step 1: Process each holding
        all_results: List[HoldingEmissions] = []
        for holding in holdings:
            result = self._process_single_holding(holding)
            self._holding_results[holding.holding_id] = result
            all_results.append(result)

        # Step 2: Double-counting prevention
        if self.config.enable_double_counting_prevention:
            all_results, dc_adjustments = self._apply_double_counting_prevention(
                all_results
            )
        else:
            dc_adjustments = {}

        # Step 3: Aggregate by asset class
        asset_class_breakdown = self._aggregate_by_asset_class(all_results)

        # Step 4: Compute portfolio totals
        total_outstanding = sum(
            r.attribution.outstanding_amount_eur
            for r in all_results
            if r.attribution
        )
        total_s1 = _round_val(sum(r.financed_scope1 for r in all_results))
        total_s2 = _round_val(sum(r.financed_scope2 for r in all_results))
        total_s3 = _round_val(sum(r.financed_scope3 for r in all_results))
        total_s1_2 = _round_val(sum(r.financed_scope1_2 for r in all_results))
        total_all = _round_val(sum(r.financed_total for r in all_results))

        # Step 5: WACI
        portfolio_waci = self._compute_waci(all_results, total_outstanding)

        # Step 6: Weighted average data quality
        weighted_dq = self._compute_weighted_data_quality(all_results, total_outstanding)

        # Step 7: Data quality coverage
        dq_coverage = self._compute_dq_coverage(all_results, total_outstanding)

        # Step 8: Count stats
        holdings_with_data = sum(
            1 for r in all_results
            if r.data_quality and r.data_quality.score < 5
        )
        holdings_estimated = len(all_results) - holdings_with_data

        # Step 9: YoY trajectory
        yoy = self._build_yoy_trajectory(total_all, total_outstanding, portfolio_waci)

        # Step 10: Methodology notes
        notes = self._generate_methodology_notes(all_results, dc_adjustments)

        end = utcnow()
        processing_ms = (end - start).total_seconds() * 1000.0

        result = PortfolioEmissionsResult(
            reporting_year=self.config.reporting_year,
            reporting_currency=self.config.reporting_currency,
            total_portfolio_outstanding_eur=_round_val(total_outstanding, 2),
            total_financed_scope1=total_s1,
            total_financed_scope2=total_s2,
            total_financed_scope3=total_s3,
            total_financed_scope1_2=total_s1_2,
            total_financed_emissions=total_all,
            portfolio_waci=_round_val(portfolio_waci),
            weighted_avg_data_quality=_round_val(weighted_dq, 2),
            data_quality_coverage_pct=_round_val(dq_coverage, 2),
            asset_class_breakdown=asset_class_breakdown,
            holding_results=all_results,
            yoy_trajectory=yoy,
            double_counting_adjustments=dc_adjustments,
            total_holdings=len(all_results),
            holdings_with_data=holdings_with_data,
            holdings_estimated=holdings_estimated,
            methodology_notes=notes,
            processing_time_ms=_round_val(processing_ms, 2),
        )

        result.provenance_hash = _compute_hash(result)
        logger.info(
            "Portfolio financed emissions: %.2f tCO2e (WACI: %.4f, DQ: %.2f)",
            result.total_financed_emissions,
            result.portfolio_waci,
            result.weighted_avg_data_quality,
        )
        return result

    def calculate_single_holding(
        self, holding: AssetClassData,
    ) -> HoldingEmissions:
        """Calculate financed emissions for a single holding.

        Convenience method for processing one holding without
        portfolio-level aggregation.

        Args:
            holding: AssetClassData for the single exposure.

        Returns:
            HoldingEmissions with attribution and provenance.
        """
        return self._process_single_holding(holding)

    def compute_attribution_factor(
        self, holding: AssetClassData,
    ) -> AttributionResult:
        """Compute the PCAF attribution factor for a holding.

        The attribution factor determines what fraction of the
        borrower's emissions is attributed to the financial institution.

        Args:
            holding: AssetClassData with financial details.

        Returns:
            AttributionResult with the factor and methodology.
        """
        outstanding_eur = self._normalize_currency(
            holding.outstanding_amount, holding.currency,
        )
        return self._compute_attribution(holding, outstanding_eur)

    def assess_data_quality(
        self, holding: AssetClassData,
    ) -> DataQualityScore:
        """Assess data quality for a single holding.

        Returns the PCAF data quality score with justification.

        Args:
            holding: AssetClassData with emission data.

        Returns:
            DataQualityScore assessment.
        """
        return self._assess_data_quality(holding)

    # ------------------------------------------------------------------
    # Internal: Single Holding Processing
    # ------------------------------------------------------------------

    def _process_single_holding(self, holding: AssetClassData) -> HoldingEmissions:
        """Process a single holding through the full PCAF pipeline.

        Steps:
            1. Normalize currency to reporting currency
            2. Compute attribution factor
            3. Attribute emissions per scope
            4. Assess data quality
            5. Compute intensity and weight

        Args:
            holding: AssetClassData input.

        Returns:
            HoldingEmissions with all calculations.
        """
        # Currency normalization
        outstanding_eur = self._normalize_currency(
            holding.outstanding_amount, holding.currency,
        )

        # Attribution factor
        attribution = self._compute_attribution(holding, outstanding_eur)
        af = attribution.attribution_factor

        # Emission attribution (deterministic multiplication)
        financed_s1 = _round_val(af * holding.scope1_emissions)
        financed_s2 = _round_val(af * holding.scope2_emissions)

        # Scope 3 inclusion depends on config and asset class
        include_s3 = (
            self.config.include_scope3
            and holding.asset_class in self.config.scope3_asset_classes
        )
        financed_s3 = _round_val(af * holding.scope3_emissions) if include_s3 else 0.0

        financed_s1_2 = _round_val(financed_s1 + financed_s2)
        financed_total = _round_val(financed_s1_2 + financed_s3)

        # Carbon intensity (tCO2e per EUR M revenue)
        revenue_m = holding.revenue / 1_000_000.0 if holding.revenue > 0 else 0.0
        total_borrower_emissions = holding.scope1_emissions + holding.scope2_emissions
        if include_s3:
            total_borrower_emissions += holding.scope3_emissions
        intensity = _safe_divide(total_borrower_emissions, revenue_m) if revenue_m > 0 else 0.0

        # Weight in portfolio (will be recalculated at portfolio level)
        weight_pct = 0.0  # Set during aggregation

        # Data quality
        dq = self._assess_data_quality(holding)

        result = HoldingEmissions(
            holding_id=holding.holding_id,
            holding_name=holding.holding_name,
            asset_class=holding.asset_class,
            attribution_factor=_round_val(af, 6),
            financed_scope1=financed_s1,
            financed_scope2=financed_s2,
            financed_scope3=financed_s3,
            financed_scope1_2=financed_s1_2,
            financed_total=financed_total,
            carbon_intensity=_round_val(intensity),
            weight_pct=weight_pct,
            weighted_intensity=0.0,  # Set during aggregation
            data_quality=dq,
            attribution=attribution,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Internal: Attribution Factor
    # ------------------------------------------------------------------

    def _compute_attribution(
        self,
        holding: AssetClassData,
        outstanding_eur: float,
    ) -> AttributionResult:
        """Compute PCAF attribution factor.

        Attribution Factor = Outstanding Amount / Denominator

        The denominator varies by asset class (see ATTRIBUTION_RULES).
        Falls back to alternative denominators if primary is zero.

        Args:
            holding: AssetClassData.
            outstanding_eur: Outstanding amount in EUR.

        Returns:
            AttributionResult.
        """
        rules = ATTRIBUTION_RULES.get(holding.asset_class.value, {})
        primary_field = rules.get("denominator", "evic")
        fallback_field = rules.get("fallback_denominator")

        # Retrieve denominator value
        denom_value = getattr(holding, primary_field, 0.0) or 0.0
        used_fallback = False
        denom_field_used = primary_field

        if denom_value <= 0.0 and fallback_field:
            denom_value = getattr(holding, fallback_field, 0.0) or 0.0
            if denom_value > 0.0:
                used_fallback = True
                denom_field_used = fallback_field

        # Calculate attribution factor
        af = _safe_divide(outstanding_eur, denom_value)

        # Cap at maximum
        af = min(af, self.config.max_attribution_factor)

        result = AttributionResult(
            holding_id=holding.holding_id,
            asset_class=holding.asset_class,
            outstanding_amount_eur=_round_val(outstanding_eur, 2),
            denominator_value=_round_val(denom_value, 2),
            denominator_field=denom_field_used,
            attribution_factor=_round_val(af, 6),
            used_fallback=used_fallback,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Internal: Data Quality Assessment
    # ------------------------------------------------------------------

    def _assess_data_quality(self, holding: AssetClassData) -> DataQualityScore:
        """Assess PCAF data quality score for a holding.

        PCAF scores (1-5):
            1: Reported, verified emissions
            2: Reported, unverified emissions
            3: Physical activity-based estimates
            4: Economic activity-based estimates
            5: Sector average estimates

        The score comes from the input data; this method validates
        and documents it.

        Args:
            holding: AssetClassData with data_quality_score.

        Returns:
            DataQualityScore with justification.
        """
        score = holding.data_quality_score
        is_estimated = score >= 3

        return DataQualityScore(
            holding_id=holding.holding_id,
            asset_class=holding.asset_class,
            score=score,
            level=DataQualityLevel(f"score_{score}"),
            methodology=DATA_QUALITY_DESCRIPTIONS.get(score, "Unknown"),
            justification=f"PCAF DQ Score {score}: {DATA_QUALITY_DESCRIPTIONS.get(score, 'N/A')}",
            emission_data_source=holding.holding_name,
            is_estimated=is_estimated,
        )

    # ------------------------------------------------------------------
    # Internal: Currency Normalization
    # ------------------------------------------------------------------

    def _build_currency_map(self) -> Dict[str, float]:
        """Build a lookup map from currency_rates config.

        Returns:
            Dict mapping source_currency -> rate to reporting currency.
        """
        cmap: Dict[str, float] = {}
        for cr in self.config.currency_rates:
            cmap[cr.source_currency] = cr.rate
        # Self-rate
        cmap[self.config.reporting_currency] = 1.0
        return cmap

    def _normalize_currency(self, amount: float, currency: str) -> float:
        """Normalize an amount to the reporting currency.

        Args:
            amount: Original amount.
            currency: Original currency code.

        Returns:
            Amount in reporting currency.
        """
        if currency == self.config.reporting_currency:
            return amount
        rate = self._currency_map.get(currency, 1.0)
        return amount * rate

    # ------------------------------------------------------------------
    # Internal: Double-Counting Prevention
    # ------------------------------------------------------------------

    def _apply_double_counting_prevention(
        self,
        results: List[HoldingEmissions],
    ) -> Tuple[List[HoldingEmissions], Dict[str, float]]:
        """Prevent double-counting when the same entity appears in
        multiple asset classes (e.g., equity + bonds of same company).

        Uses entity groups from config to identify overlaps and
        allocates emissions proportionally to outstanding amounts.

        Args:
            results: List of HoldingEmissions.

        Returns:
            Tuple of (adjusted results, adjustment details dict).
        """
        adjustments: Dict[str, float] = {}

        if not self.config.double_counting_entity_groups:
            return results, adjustments

        # Build lookup: holding_id -> result
        result_map: Dict[str, HoldingEmissions] = {
            r.holding_id: r for r in results
        }

        for group_name, holding_ids in self.config.double_counting_entity_groups.items():
            group_results = [
                result_map[hid] for hid in holding_ids
                if hid in result_map
            ]
            if len(group_results) <= 1:
                continue

            # Proportional allocation based on outstanding amount
            total_outstanding = sum(
                r.attribution.outstanding_amount_eur
                for r in group_results
                if r.attribution
            )
            if total_outstanding <= 0:
                continue

            # Use the maximum financed emissions from any single holding
            # as the reference (avoids double-counting)
            max_emissions = max(r.financed_total for r in group_results)
            current_total = sum(r.financed_total for r in group_results)
            excess = current_total - max_emissions

            if excess > 0:
                # Redistribute proportionally
                for r in group_results:
                    if r.attribution and total_outstanding > 0:
                        share = r.attribution.outstanding_amount_eur / total_outstanding
                        adjusted_total = max_emissions * share
                        original = r.financed_total
                        ratio = _safe_divide(adjusted_total, original, 1.0)
                        r.financed_scope1 = _round_val(r.financed_scope1 * ratio)
                        r.financed_scope2 = _round_val(r.financed_scope2 * ratio)
                        r.financed_scope3 = _round_val(r.financed_scope3 * ratio)
                        r.financed_scope1_2 = _round_val(r.financed_scope1 + r.financed_scope2)
                        r.financed_total = _round_val(
                            r.financed_scope1_2 + r.financed_scope3
                        )
                        adjustments[f"{group_name}:{r.holding_id}"] = original - r.financed_total

        return results, adjustments

    # ------------------------------------------------------------------
    # Internal: Aggregation
    # ------------------------------------------------------------------

    def _aggregate_by_asset_class(
        self,
        results: List[HoldingEmissions],
    ) -> List[EmissionsByAssetClass]:
        """Aggregate holding results by PCAF asset class.

        Args:
            results: List of HoldingEmissions.

        Returns:
            List of EmissionsByAssetClass summaries.
        """
        groups: Dict[PCAFAssetClass, List[HoldingEmissions]] = defaultdict(list)
        for r in results:
            groups[r.asset_class].append(r)

        total_portfolio_outstanding = sum(
            r.attribution.outstanding_amount_eur
            for r in results
            if r.attribution
        )

        breakdowns: List[EmissionsByAssetClass] = []
        for ac, group in groups.items():
            total_out = sum(
                r.attribution.outstanding_amount_eur for r in group if r.attribution
            )
            total_s1 = sum(r.financed_scope1 for r in group)
            total_s2 = sum(r.financed_scope2 for r in group)
            total_s3 = sum(r.financed_scope3 for r in group)
            total_s1_2 = sum(r.financed_scope1_2 for r in group)
            total_fe = sum(r.financed_total for r in group)

            # Weighted average intensity for this asset class
            waci = self._compute_waci(group, total_out)

            # Average data quality
            dq_scores = [r.data_quality.score for r in group if r.data_quality]
            avg_dq = sum(dq_scores) / len(dq_scores) if dq_scores else 5.0

            weight_in_portfolio = _safe_pct(total_out, total_portfolio_outstanding)

            breakdowns.append(EmissionsByAssetClass(
                asset_class=ac,
                holding_count=len(group),
                total_outstanding_eur=_round_val(total_out, 2),
                total_financed_scope1=_round_val(total_s1),
                total_financed_scope2=_round_val(total_s2),
                total_financed_scope3=_round_val(total_s3),
                total_financed_scope1_2=_round_val(total_s1_2),
                total_financed_emissions=_round_val(total_fe),
                weighted_avg_intensity=_round_val(waci),
                avg_data_quality_score=_round_val(avg_dq, 2),
                weight_in_portfolio_pct=_round_val(weight_in_portfolio, 2),
            ))

        return breakdowns

    def _compute_waci(
        self,
        results: List[HoldingEmissions],
        total_outstanding: float,
    ) -> float:
        """Compute Weighted Average Carbon Intensity (WACI).

        WACI = SUM(weight_i * intensity_i)

        where weight_i = outstanding_i / total_outstanding
        and intensity_i = borrower_emissions / revenue_M

        Args:
            results: List of HoldingEmissions.
            total_outstanding: Total outstanding for weight calculation.

        Returns:
            WACI value (tCO2e / EUR M).
        """
        if total_outstanding <= 0:
            return 0.0

        waci = 0.0
        for r in results:
            if r.attribution and r.carbon_intensity > 0:
                weight = r.attribution.outstanding_amount_eur / total_outstanding
                r.weight_pct = _round_val(weight * 100.0, 4)
                r.weighted_intensity = _round_val(weight * r.carbon_intensity, 6)
                waci += r.weighted_intensity

        return waci

    def _compute_weighted_data_quality(
        self,
        results: List[HoldingEmissions],
        total_outstanding: float,
    ) -> float:
        """Compute portfolio weighted average data quality score.

        Weighted by outstanding amount.

        Args:
            results: List of HoldingEmissions.
            total_outstanding: Total outstanding for weights.

        Returns:
            Weighted average data quality score (1.0-5.0).
        """
        if total_outstanding <= 0:
            return 5.0

        weighted_sum = 0.0
        for r in results:
            if r.attribution and r.data_quality:
                weight = r.attribution.outstanding_amount_eur / total_outstanding
                weighted_sum += weight * r.data_quality.score

        return max(1.0, min(5.0, weighted_sum))

    def _compute_dq_coverage(
        self,
        results: List[HoldingEmissions],
        total_outstanding: float,
    ) -> float:
        """Compute the percentage of the portfolio with DQ score < 5.

        Args:
            results: List of HoldingEmissions.
            total_outstanding: Total outstanding.

        Returns:
            Percentage of portfolio with DQ < 5.
        """
        if total_outstanding <= 0:
            return 0.0

        covered_outstanding = sum(
            r.attribution.outstanding_amount_eur
            for r in results
            if r.attribution and r.data_quality and r.data_quality.score < 5
        )
        return _safe_pct(covered_outstanding, total_outstanding)

    # ------------------------------------------------------------------
    # Internal: YoY Trajectory
    # ------------------------------------------------------------------

    def _build_yoy_trajectory(
        self,
        current_emissions: float,
        current_outstanding: float,
        current_waci: float,
    ) -> List[YoYTrajectory]:
        """Build year-over-year emissions trajectory.

        Includes prior results from config and appends the current year.

        Args:
            current_emissions: Current year total financed emissions.
            current_outstanding: Current year total outstanding.
            current_waci: Current year WACI.

        Returns:
            List of YoYTrajectory data points.
        """
        trajectory = list(self.config.yoy_prior_results)

        # Get prior year for YoY calculation
        prior = trajectory[-1] if trajectory else None

        yoy_change_pct = 0.0
        yoy_change_abs = 0.0
        if prior and prior.total_financed_emissions > 0:
            yoy_change_abs = current_emissions - prior.total_financed_emissions
            yoy_change_pct = _safe_pct(
                yoy_change_abs, prior.total_financed_emissions
            )

        current_point = YoYTrajectory(
            year=self.config.reporting_year,
            total_financed_emissions=_round_val(current_emissions),
            total_outstanding_eur=_round_val(current_outstanding, 2),
            portfolio_waci=_round_val(current_waci),
            yoy_change_pct=_round_val(yoy_change_pct, 2),
            yoy_change_absolute=_round_val(yoy_change_abs),
        )
        trajectory.append(current_point)
        return trajectory

    # ------------------------------------------------------------------
    # Internal: Methodology Notes
    # ------------------------------------------------------------------

    def _generate_methodology_notes(
        self,
        results: List[HoldingEmissions],
        adjustments: Dict[str, float],
    ) -> List[str]:
        """Generate methodology disclosure notes.

        Args:
            results: Holding results.
            adjustments: Double-counting adjustments.

        Returns:
            List of methodology note strings.
        """
        notes: List[str] = [
            f"Methodology: PCAF Global GHG Accounting & Reporting Standard (2nd Edition)",
            f"Reporting year: {self.config.reporting_year}",
            f"Reporting currency: {self.config.reporting_currency}",
            f"Scope 3 included: {self.config.include_scope3}",
            f"Total holdings processed: {len(results)}",
        ]

        # Asset class distribution
        ac_counts: Dict[str, int] = defaultdict(int)
        for r in results:
            ac_counts[r.asset_class.value] += 1
        for ac, count in sorted(ac_counts.items()):
            notes.append(f"Asset class {ac}: {count} holding(s)")

        # Fallback usage
        fallback_count = sum(
            1 for r in results
            if r.attribution and r.attribution.used_fallback
        )
        if fallback_count > 0:
            notes.append(
                f"Fallback denominators used for {fallback_count} holding(s)"
            )

        # Double-counting
        if adjustments:
            total_adj = sum(adjustments.values())
            notes.append(
                f"Double-counting adjustments: {len(adjustments)} adjustment(s), "
                f"total removed: {_round_val(total_adj)} tCO2e"
            )

        return notes
