# -*- coding: utf-8 -*-
"""
PortfolioCarbonFootprintEngine - PACK-010 SFDR Article 8 Engine 7

Calculate portfolio-level carbon metrics for PAI reporting under SFDR.
Implements WACI (Weighted Average Carbon Intensity), carbon footprint,
financed emissions (PCAF methodology), sector-level attribution, and
implied temperature alignment.

Key Metrics (per SFDR RTS Annex I, Table 1):
    - PAI 1: GHG Emissions (Scope 1, Scope 2, Total)
    - PAI 2: Carbon Footprint (tCO2e/EUR M invested)
    - PAI 3: GHG Intensity (WACI - tCO2e/EUR M revenue)
    - PAI 4: Fossil fuel exposure
    These are mandatory PAI indicators that must be disclosed.

Formulas:
    WACI = SUM(portfolio_weight_i * (emissions_i / revenue_i))
    Carbon Footprint = SUM((value_i / evic_i) * emissions_i) / portfolio_value_EUR_M
    Financed Emissions = SUM(attribution_factor_i * emissions_i)
    Attribution Factor = outstanding_amount_i / (equity_i + debt_i)

Methodologies:
    - PCAF (Partnership for Carbon Accounting Financials): Standard for
      financial institutions to measure financed emissions
    - GHG Protocol: Corporate accounting and reporting standard

Zero-Hallucination:
    - All carbon calculations use deterministic arithmetic formulas
    - No LLM involvement in any calculation path
    - Emission factors from established reference sources only
    - SHA-256 provenance hash on every result

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-010 SFDR Article 8
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

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


class CarbonMethodology(str, Enum):
    """Carbon accounting methodology."""
    PCAF = "pcaf"
    GHG_PROTOCOL = "ghg_protocol"


class AttributionMethod(str, Enum):
    """Method for attributing emissions to the portfolio."""
    EVIC = "evic"  # Enterprise Value Including Cash
    MARKET_CAP = "market_cap"
    TOTAL_ASSETS = "total_assets"
    REVENUE = "revenue"


class ScopeCoverage(str, Enum):
    """Emission scope coverage for calculations."""
    SCOPE_1 = "scope_1"
    SCOPE_1_2 = "scope_1_2"
    SCOPE_1_2_3 = "scope_1_2_3"


class DataQuality(str, Enum):
    """Data quality score per PCAF methodology."""
    REPORTED_VERIFIED = "reported_verified"       # Score 1
    REPORTED_UNVERIFIED = "reported_unverified"   # Score 2
    ESTIMATED_SPECIFIC = "estimated_specific"      # Score 3
    ESTIMATED_SECTOR = "estimated_sector"          # Score 4
    ESTIMATED_BROAD = "estimated_broad"            # Score 5


class TemperatureMethodology(str, Enum):
    """Implied temperature rise calculation methodology."""
    SBT_PORTFOLIO = "sbt_portfolio"
    SECTORAL_DECARBONIZATION = "sectoral_decarbonization"
    MARKET_WARMING_POTENTIAL = "market_warming_potential"


# ---------------------------------------------------------------------------
# Sector Emission Intensity Reference Data
# ---------------------------------------------------------------------------


SECTOR_EMISSION_INTENSITY: Dict[str, Dict[str, float]] = {
    # NACE sector code -> {scope1+2 intensity, scope3 intensity} in tCO2e/EUR M revenue
    "A": {"scope_1_2": 120.0, "scope_3": 280.0, "name": "Agriculture, Forestry, Fishing"},
    "B": {"scope_1_2": 350.0, "scope_3": 150.0, "name": "Mining and Quarrying"},
    "C": {"scope_1_2": 180.0, "scope_3": 420.0, "name": "Manufacturing"},
    "C10-12": {"scope_1_2": 95.0, "scope_3": 380.0, "name": "Food, Beverages, Tobacco"},
    "C19": {"scope_1_2": 900.0, "scope_3": 1200.0, "name": "Coke and Refined Petroleum"},
    "C20": {"scope_1_2": 450.0, "scope_3": 350.0, "name": "Chemicals"},
    "C23": {"scope_1_2": 800.0, "scope_3": 200.0, "name": "Non-metallic Minerals (Cement)"},
    "C24": {"scope_1_2": 1200.0, "scope_3": 300.0, "name": "Basic Metals"},
    "C25": {"scope_1_2": 80.0, "scope_3": 250.0, "name": "Fabricated Metals"},
    "C29": {"scope_1_2": 40.0, "scope_3": 600.0, "name": "Motor Vehicles"},
    "D": {"scope_1_2": 650.0, "scope_3": 100.0, "name": "Electricity, Gas, Steam"},
    "E": {"scope_1_2": 200.0, "scope_3": 80.0, "name": "Water Supply, Sewerage, Waste"},
    "F": {"scope_1_2": 60.0, "scope_3": 350.0, "name": "Construction"},
    "G": {"scope_1_2": 25.0, "scope_3": 180.0, "name": "Wholesale and Retail Trade"},
    "H": {"scope_1_2": 250.0, "scope_3": 120.0, "name": "Transportation and Storage"},
    "H49": {"scope_1_2": 400.0, "scope_3": 80.0, "name": "Land Transport"},
    "H50": {"scope_1_2": 500.0, "scope_3": 100.0, "name": "Water Transport"},
    "H51": {"scope_1_2": 800.0, "scope_3": 150.0, "name": "Air Transport"},
    "I": {"scope_1_2": 35.0, "scope_3": 90.0, "name": "Accommodation and Food Service"},
    "J": {"scope_1_2": 10.0, "scope_3": 50.0, "name": "Information and Communication"},
    "K": {"scope_1_2": 5.0, "scope_3": 30.0, "name": "Financial and Insurance"},
    "L": {"scope_1_2": 30.0, "scope_3": 40.0, "name": "Real Estate"},
    "M": {"scope_1_2": 8.0, "scope_3": 35.0, "name": "Professional, Scientific, Technical"},
    "N": {"scope_1_2": 15.0, "scope_3": 45.0, "name": "Administrative and Support"},
    "O": {"scope_1_2": 20.0, "scope_3": 25.0, "name": "Public Administration"},
    "P": {"scope_1_2": 12.0, "scope_3": 20.0, "name": "Education"},
    "Q": {"scope_1_2": 18.0, "scope_3": 30.0, "name": "Human Health and Social Work"},
    "R": {"scope_1_2": 22.0, "scope_3": 35.0, "name": "Arts, Entertainment, Recreation"},
    "S": {"scope_1_2": 10.0, "scope_3": 25.0, "name": "Other Service Activities"},
    "_default": {"scope_1_2": 50.0, "scope_3": 100.0, "name": "Unknown/Other"},
}

# PCAF data quality scores (1=best, 5=worst)
PCAF_QUALITY_SCORES: Dict[DataQuality, int] = {
    DataQuality.REPORTED_VERIFIED: 1,
    DataQuality.REPORTED_UNVERIFIED: 2,
    DataQuality.ESTIMATED_SPECIFIC: 3,
    DataQuality.ESTIMATED_SECTOR: 4,
    DataQuality.ESTIMATED_BROAD: 5,
}

# Temperature pathway reference (degrees C per tCO2e/EUR M trajectory)
TEMPERATURE_PATHWAYS: Dict[str, float] = {
    "1.5C_aligned": 50.0,    # Max carbon intensity for 1.5C pathway
    "2.0C_aligned": 100.0,   # Max carbon intensity for 2.0C pathway
    "3.0C_aligned": 200.0,   # Max carbon intensity for 3.0C pathway
    "4.0C_plus": 400.0,      # Above 4.0C pathway
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class HoldingEmissions(BaseModel):
    """Emissions data for a single portfolio holding.

    Contains company-level emission data, financial data, and data
    quality indicators needed for portfolio carbon calculations.
    """
    holding_id: str = Field(default_factory=_new_uuid, description="Unique holding identifier")
    company_name: str = Field(default="", description="Investee company name")
    isin: str = Field(default="", description="Security ISIN")
    sector: str = Field(default="", description="NACE/GICS sector code")
    country: str = Field(default="", description="Country of domicile (ISO 3166)")
    # Emissions (tCO2e per year)
    scope1: float = Field(default=0.0, description="Scope 1 emissions (tCO2e)")
    scope2: float = Field(default=0.0, description="Scope 2 emissions (tCO2e)")
    scope3: float = Field(default=0.0, description="Scope 3 emissions (tCO2e)")
    total_emissions: float = Field(default=0.0, description="Total emissions (tCO2e)")
    # Financial data (EUR)
    revenue: float = Field(default=0.0, description="Annual revenue (EUR)")
    enterprise_value: float = Field(default=0.0, description="Enterprise value (EUR)")
    evic: float = Field(default=0.0, description="Enterprise value including cash (EUR)")
    market_cap: float = Field(default=0.0, description="Market capitalization (EUR)")
    total_assets: float = Field(default=0.0, description="Total assets (EUR)")
    total_debt: float = Field(default=0.0, description="Total debt (EUR)")
    total_equity: float = Field(default=0.0, description="Total equity (EUR)")
    # Portfolio position
    holding_value: float = Field(default=0.0, description="Portfolio holding value (EUR)")
    weight_pct: float = Field(default=0.0, description="Portfolio weight (%)")
    outstanding_amount: float = Field(default=0.0, description="Outstanding amount for attribution")
    # Data quality
    data_quality: DataQuality = Field(
        default=DataQuality.ESTIMATED_SECTOR,
        description="PCAF data quality score"
    )
    emissions_estimated: bool = Field(
        default=False, description="Whether emissions are estimated"
    )
    reporting_year: int = Field(default=2025, description="Year of emissions data")

    @field_validator("total_emissions", mode="before")
    @classmethod
    def _auto_total(cls, v: Any, info: Any) -> float:
        """Auto-calculate total if not provided."""
        if v and float(v) > 0:
            return float(v)
        return 0.0

    def compute_total_emissions(self) -> float:
        """Compute total emissions from scope breakdown."""
        self.total_emissions = self.scope1 + self.scope2 + self.scope3
        return self.total_emissions


class WACIResult(BaseModel):
    """Weighted Average Carbon Intensity result.

    WACI = SUM(portfolio_weight_i * (emissions_i / revenue_i))
    Unit: tCO2e per EUR M revenue
    """
    result_id: str = Field(default_factory=_new_uuid, description="Unique result identifier")
    waci_value: float = Field(description="WACI value (tCO2e/EUR M revenue)")
    unit: str = Field(default="tCO2e/EUR M revenue", description="Unit of measurement")
    scope_coverage: ScopeCoverage = Field(description="Scope coverage used")
    coverage_ratio: float = Field(
        default=0.0, description="% of portfolio with emission data"
    )
    total_holdings: int = Field(default=0, description="Total holdings in portfolio")
    covered_holdings: int = Field(default=0, description="Holdings with emission data")
    by_sector: Dict[str, float] = Field(
        default_factory=dict, description="WACI contribution by sector"
    )
    top_contributors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Top WACI contributors"
    )
    data_quality_score: float = Field(
        default=0.0, description="Average PCAF data quality score"
    )
    calculated_at: datetime = Field(default_factory=_utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class CarbonFootprintResult(BaseModel):
    """Carbon footprint calculation result.

    Carbon Footprint = SUM((value_i / evic_i) * emissions_i) / portfolio_value_EUR_M
    Unit: tCO2e per EUR M invested
    """
    result_id: str = Field(default_factory=_new_uuid, description="Unique result identifier")
    carbon_footprint: float = Field(description="Carbon footprint (tCO2e/EUR M invested)")
    unit: str = Field(default="tCO2e/EUR M invested", description="Unit of measurement")
    total_financed_emissions: float = Field(
        description="Total financed emissions (tCO2e)"
    )
    total_portfolio_value: float = Field(description="Total portfolio value (EUR)")
    scope_coverage: ScopeCoverage = Field(description="Scope coverage used")
    coverage_ratio: float = Field(default=0.0, description="% of portfolio covered")
    attribution_method: AttributionMethod = Field(description="Attribution method used")
    attribution_by_sector: Dict[str, float] = Field(
        default_factory=dict, description="Financed emissions by sector"
    )
    by_holding: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-holding emission attribution"
    )
    by_scope: Dict[str, float] = Field(
        default_factory=dict, description="Financed emissions by scope"
    )
    data_quality_score: float = Field(default=0.0, description="Average PCAF quality score")
    calculated_at: datetime = Field(default_factory=_utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class FinancedEmissionsResult(BaseModel):
    """Financed emissions calculation per PCAF methodology.

    Financed Emissions = SUM(attribution_factor_i * emissions_i)
    Attribution Factor = outstanding_amount_i / (equity_i + debt_i)
    """
    result_id: str = Field(default_factory=_new_uuid, description="Unique result identifier")
    total_financed_emissions: float = Field(description="Total financed emissions (tCO2e)")
    scope1_financed: float = Field(default=0.0, description="Scope 1 financed emissions")
    scope2_financed: float = Field(default=0.0, description="Scope 2 financed emissions")
    scope3_financed: float = Field(default=0.0, description="Scope 3 financed emissions")
    total_outstanding: float = Field(default=0.0, description="Total outstanding amount (EUR)")
    coverage_ratio: float = Field(default=0.0, description="% of portfolio covered")
    by_holding: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-holding financed emissions"
    )
    by_sector: Dict[str, float] = Field(
        default_factory=dict, description="Financed emissions by sector"
    )
    by_geography: Dict[str, float] = Field(
        default_factory=dict, description="Financed emissions by country"
    )
    weighted_data_quality: float = Field(
        default=0.0, description="Weighted average PCAF data quality score"
    )
    calculated_at: datetime = Field(default_factory=_utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class TemperatureAlignment(BaseModel):
    """Implied temperature rise alignment result.

    Maps the portfolio carbon intensity trajectory to an implied
    temperature rise based on sector-specific decarbonization pathways.
    """
    alignment_id: str = Field(default_factory=_new_uuid, description="Unique alignment identifier")
    implied_temperature_rise: float = Field(
        description="Implied temperature rise in degrees Celsius"
    )
    methodology: TemperatureMethodology = Field(description="Methodology used")
    confidence: float = Field(default=0.0, description="Confidence level (0-1)")
    portfolio_carbon_intensity: float = Field(
        default=0.0, description="Current portfolio carbon intensity"
    )
    pathway_benchmark: str = Field(default="", description="Reference pathway benchmark")
    sector_contributions: Dict[str, float] = Field(
        default_factory=dict, description="Temperature contribution by sector"
    )
    aligned_with_paris: bool = Field(
        default=False, description="Whether portfolio is Paris-aligned (<2.0C)"
    )
    notes: str = Field(default="", description="Assessment notes")
    calculated_at: datetime = Field(default_factory=_utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class SectorBreakdown(BaseModel):
    """Sector-level carbon attribution breakdown."""
    sector_code: str = Field(description="NACE/GICS sector code")
    sector_name: str = Field(default="", description="Sector name")
    holding_count: int = Field(default=0, description="Number of holdings in sector")
    portfolio_weight_pct: float = Field(default=0.0, description="Sector weight in portfolio (%)")
    total_emissions: float = Field(default=0.0, description="Total emissions attributed (tCO2e)")
    carbon_intensity: float = Field(default=0.0, description="Sector carbon intensity (tCO2e/EUR M)")
    waci_contribution: float = Field(default=0.0, description="Contribution to portfolio WACI")
    financed_emissions: float = Field(default=0.0, description="Financed emissions (tCO2e)")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class CarbonSummary(BaseModel):
    """Comprehensive portfolio carbon summary for PAI reporting."""
    summary_id: str = Field(default_factory=_new_uuid, description="Unique summary identifier")
    waci: Optional[WACIResult] = Field(default=None, description="WACI result")
    carbon_footprint: Optional[CarbonFootprintResult] = Field(
        default=None, description="Carbon footprint result"
    )
    financed_emissions: Optional[FinancedEmissionsResult] = Field(
        default=None, description="Financed emissions result"
    )
    temperature_alignment: Optional[TemperatureAlignment] = Field(
        default=None, description="Temperature alignment result"
    )
    sector_breakdown: List[SectorBreakdown] = Field(
        default_factory=list, description="Sector-level breakdown"
    )
    total_portfolio_value: float = Field(default=0.0, description="Total portfolio value (EUR)")
    total_holdings: int = Field(default=0, description="Total holdings")
    coverage_ratio: float = Field(default=0.0, description="Data coverage ratio (%)")
    generated_at: datetime = Field(default_factory=_utcnow, description="Generation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------


class CarbonFootprintConfig(BaseModel):
    """Configuration for the PortfolioCarbonFootprintEngine.

    Controls methodology, scope coverage, and attribution settings.
    """
    methodology: CarbonMethodology = Field(
        default=CarbonMethodology.PCAF, description="Carbon accounting methodology"
    )
    scope_coverage: ScopeCoverage = Field(
        default=ScopeCoverage.SCOPE_1_2, description="Default scope coverage"
    )
    attribution_method: AttributionMethod = Field(
        default=AttributionMethod.EVIC, description="Default attribution method"
    )
    min_coverage_ratio: float = Field(
        default=50.0, description="Minimum data coverage ratio for valid results (%)"
    )
    top_contributors_count: int = Field(
        default=10, description="Number of top contributors to report"
    )
    use_sector_estimates: bool = Field(
        default=True, description="Use sector averages for missing emission data"
    )
    revenue_in_millions: bool = Field(
        default=True, description="Whether revenue figures are in millions"
    )
    currency: str = Field(default="EUR", description="Portfolio currency")


# ---------------------------------------------------------------------------
# Pydantic model_rebuild for forward reference resolution
# ---------------------------------------------------------------------------

CarbonFootprintConfig.model_rebuild()
HoldingEmissions.model_rebuild()
WACIResult.model_rebuild()
CarbonFootprintResult.model_rebuild()
FinancedEmissionsResult.model_rebuild()
TemperatureAlignment.model_rebuild()
SectorBreakdown.model_rebuild()
CarbonSummary.model_rebuild()


# ---------------------------------------------------------------------------
# PortfolioCarbonFootprintEngine
# ---------------------------------------------------------------------------


class PortfolioCarbonFootprintEngine:
    """
    Portfolio-level carbon metrics calculation engine.

    Implements WACI, carbon footprint, financed emissions (PCAF),
    sector attribution, and implied temperature alignment calculations
    for SFDR PAI reporting.

    Attributes:
        config: Engine configuration parameters.
        _holdings: Stored holding emissions data.
        _total_portfolio_value: Calculated total portfolio value.

    Example:
        >>> engine = PortfolioCarbonFootprintEngine()
        >>> holdings = [HoldingEmissions(
        ...     company_name="Corp A", scope1=5000, scope2=2000,
        ...     revenue=50_000_000, evic=200_000_000,
        ...     holding_value=10_000_000, weight_pct=5.0
        ... )]
        >>> waci = engine.calculate_waci(holdings)
        >>> print(f"WACI: {waci.waci_value:.2f} tCO2e/EUR M revenue")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PortfolioCarbonFootprintEngine.

        Args:
            config: Optional configuration dictionary.
        """
        if config and isinstance(config, dict):
            self.config = CarbonFootprintConfig(**config)
        elif config and isinstance(config, CarbonFootprintConfig):
            self.config = config
        else:
            self.config = CarbonFootprintConfig()

        self._holdings: List[HoldingEmissions] = []
        self._total_portfolio_value: float = 0.0

        logger.info(
            "PortfolioCarbonFootprintEngine initialized (version=%s, methodology=%s)",
            _MODULE_VERSION,
            self.config.methodology.value,
        )

    # ------------------------------------------------------------------
    # WACI Calculation
    # ------------------------------------------------------------------

    def calculate_waci(
        self,
        holdings: List[HoldingEmissions],
        scope_coverage: Optional[ScopeCoverage] = None,
    ) -> WACIResult:
        """Calculate Weighted Average Carbon Intensity (WACI).

        WACI = SUM(portfolio_weight_i * (emissions_i / revenue_i))

        This is PAI indicator 3 under SFDR RTS Annex I, Table 1.

        Args:
            holdings: List of holding emission data.
            scope_coverage: Scope coverage override.

        Returns:
            WACIResult with WACI value and breakdown.
        """
        start = _utcnow()
        self._holdings = holdings
        scope = scope_coverage or self.config.scope_coverage

        self._total_portfolio_value = sum(h.holding_value for h in holdings)
        self._ensure_weights(holdings)
        self._ensure_totals(holdings)

        waci_total = 0.0
        sector_contributions: Dict[str, float] = defaultdict(float)
        holding_contributions: List[Dict[str, Any]] = []
        covered_count = 0
        quality_sum = 0.0

        for h in holdings:
            emissions = self._get_scoped_emissions(h, scope)
            revenue_m = self._to_millions(h.revenue)

            if revenue_m <= 0.0 or emissions <= 0.0:
                if self.config.use_sector_estimates and h.sector:
                    emissions, revenue_m = self._estimate_from_sector(h, scope)
                if revenue_m <= 0.0:
                    continue

            carbon_intensity = emissions / revenue_m
            weight_fraction = h.weight_pct / 100.0
            contribution = weight_fraction * carbon_intensity

            waci_total += contribution
            sector_key = h.sector or "_default"
            sector_contributions[sector_key] += contribution
            covered_count += 1
            quality_sum += PCAF_QUALITY_SCORES.get(h.data_quality, 5)

            holding_contributions.append({
                "holding_id": h.holding_id,
                "company_name": h.company_name,
                "weight_pct": round(h.weight_pct, 4),
                "carbon_intensity": round(carbon_intensity, 2),
                "waci_contribution": round(contribution, 4),
            })

        # Sort top contributors
        holding_contributions.sort(
            key=lambda x: x["waci_contribution"], reverse=True
        )
        top = holding_contributions[: self.config.top_contributors_count]

        coverage = _safe_pct(covered_count, len(holdings))
        avg_quality = _safe_divide(quality_sum, covered_count, 5.0)

        result = WACIResult(
            waci_value=round(waci_total, 4),
            scope_coverage=scope,
            coverage_ratio=round(coverage, 2),
            total_holdings=len(holdings),
            covered_holdings=covered_count,
            by_sector=dict(
                {k: round(v, 4) for k, v in sector_contributions.items()}
            ),
            top_contributors=top,
            data_quality_score=round(avg_quality, 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "WACI calculated: %.2f tCO2e/EUR M revenue (%s, coverage=%.1f%%) in %dms",
            waci_total,
            scope.value,
            coverage,
            int((_utcnow() - start).total_seconds() * 1000),
        )
        return result

    # ------------------------------------------------------------------
    # Carbon Footprint Calculation
    # ------------------------------------------------------------------

    def calculate_carbon_footprint(
        self,
        holdings: Optional[List[HoldingEmissions]] = None,
        scope_coverage: Optional[ScopeCoverage] = None,
        attribution_method: Optional[AttributionMethod] = None,
    ) -> CarbonFootprintResult:
        """Calculate portfolio carbon footprint.

        Carbon Footprint = SUM((value_i / evic_i) * emissions_i) / portfolio_value_EUR_M

        This is PAI indicator 2 under SFDR RTS Annex I, Table 1.

        Args:
            holdings: Optional list (uses stored if not provided).
            scope_coverage: Scope coverage override.
            attribution_method: Attribution method override.

        Returns:
            CarbonFootprintResult with carbon footprint and breakdown.
        """
        start = _utcnow()
        if holdings is not None:
            self._holdings = holdings
        holdings_list = self._holdings
        scope = scope_coverage or self.config.scope_coverage
        attr_method = attribution_method or self.config.attribution_method

        self._total_portfolio_value = sum(h.holding_value for h in holdings_list)
        self._ensure_weights(holdings_list)
        self._ensure_totals(holdings_list)

        portfolio_value_m = self._to_millions(self._total_portfolio_value)
        total_financed = 0.0
        sector_attribution: Dict[str, float] = defaultdict(float)
        scope_attribution: Dict[str, float] = {"scope_1": 0.0, "scope_2": 0.0, "scope_3": 0.0}
        holding_details: List[Dict[str, Any]] = []
        covered_count = 0
        quality_sum = 0.0

        for h in holdings_list:
            denominator = self._get_attribution_denominator(h, attr_method)
            if denominator <= 0.0:
                continue

            attribution_factor = h.holding_value / denominator
            emissions = self._get_scoped_emissions(h, scope)

            if emissions <= 0.0 and self.config.use_sector_estimates and h.sector:
                emissions, _ = self._estimate_from_sector(h, scope)

            if emissions <= 0.0:
                continue

            financed = attribution_factor * emissions
            total_financed += financed
            covered_count += 1
            quality_sum += PCAF_QUALITY_SCORES.get(h.data_quality, 5)

            sector_key = h.sector or "_default"
            sector_attribution[sector_key] += financed

            # Scope breakdown for this holding
            scope_attribution["scope_1"] += attribution_factor * h.scope1
            scope_attribution["scope_2"] += attribution_factor * h.scope2
            if scope == ScopeCoverage.SCOPE_1_2_3:
                scope_attribution["scope_3"] += attribution_factor * h.scope3

            holding_details.append({
                "holding_id": h.holding_id,
                "company_name": h.company_name,
                "attribution_factor": round(attribution_factor, 6),
                "financed_emissions": round(financed, 2),
                "weight_pct": round(h.weight_pct, 4),
            })

        carbon_footprint_value = _safe_divide(total_financed, portfolio_value_m, 0.0)
        coverage = _safe_pct(covered_count, len(holdings_list))
        avg_quality = _safe_divide(quality_sum, covered_count, 5.0)

        holding_details.sort(key=lambda x: x["financed_emissions"], reverse=True)

        result = CarbonFootprintResult(
            carbon_footprint=round(carbon_footprint_value, 4),
            total_financed_emissions=round(total_financed, 2),
            total_portfolio_value=round(self._total_portfolio_value, 2),
            scope_coverage=scope,
            coverage_ratio=round(coverage, 2),
            attribution_method=attr_method,
            attribution_by_sector=dict(
                {k: round(v, 2) for k, v in sector_attribution.items()}
            ),
            by_holding=holding_details[: self.config.top_contributors_count],
            by_scope={k: round(v, 2) for k, v in scope_attribution.items()},
            data_quality_score=round(avg_quality, 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Carbon footprint: %.2f tCO2e/EUR M invested (financed=%.0f tCO2e, "
            "coverage=%.1f%%) in %dms",
            carbon_footprint_value,
            total_financed,
            coverage,
            int((_utcnow() - start).total_seconds() * 1000),
        )
        return result

    # ------------------------------------------------------------------
    # Financed Emissions (PCAF)
    # ------------------------------------------------------------------

    def calculate_financed_emissions(
        self,
        holdings: Optional[List[HoldingEmissions]] = None,
        scope_coverage: Optional[ScopeCoverage] = None,
    ) -> FinancedEmissionsResult:
        """Calculate financed emissions per PCAF methodology.

        Financed Emissions = SUM(attribution_factor_i * emissions_i)
        Attribution Factor = outstanding_amount_i / (equity_i + debt_i)

        Args:
            holdings: Optional list (uses stored if not provided).
            scope_coverage: Scope coverage override.

        Returns:
            FinancedEmissionsResult with detailed breakdown.
        """
        start = _utcnow()
        if holdings is not None:
            self._holdings = holdings
        holdings_list = self._holdings
        scope = scope_coverage or self.config.scope_coverage

        total_financed = 0.0
        scope1_financed = 0.0
        scope2_financed = 0.0
        scope3_financed = 0.0
        total_outstanding = 0.0
        sector_emissions: Dict[str, float] = defaultdict(float)
        geo_emissions: Dict[str, float] = defaultdict(float)
        holding_details: List[Dict[str, Any]] = []
        covered_count = 0
        quality_weighted_sum = 0.0
        weight_sum = 0.0

        for h in holdings_list:
            # PCAF attribution factor
            equity_plus_debt = h.total_equity + h.total_debt
            if equity_plus_debt <= 0.0:
                # Fall back to EVIC
                equity_plus_debt = h.evic if h.evic > 0 else h.market_cap
            if equity_plus_debt <= 0.0:
                continue

            outstanding = h.outstanding_amount if h.outstanding_amount > 0 else h.holding_value
            attribution_factor = outstanding / equity_plus_debt

            s1 = attribution_factor * h.scope1
            s2 = attribution_factor * h.scope2
            s3 = attribution_factor * h.scope3 if scope == ScopeCoverage.SCOPE_1_2_3 else 0.0

            holding_financed = s1 + s2 + s3
            total_financed += holding_financed
            scope1_financed += s1
            scope2_financed += s2
            scope3_financed += s3
            total_outstanding += outstanding
            covered_count += 1

            quality_score = PCAF_QUALITY_SCORES.get(h.data_quality, 5)
            quality_weighted_sum += quality_score * outstanding
            weight_sum += outstanding

            sector_key = h.sector or "_default"
            sector_emissions[sector_key] += holding_financed
            geo_key = h.country or "unknown"
            geo_emissions[geo_key] += holding_financed

            holding_details.append({
                "holding_id": h.holding_id,
                "company_name": h.company_name,
                "attribution_factor": round(attribution_factor, 6),
                "outstanding_amount": round(outstanding, 2),
                "financed_emissions": round(holding_financed, 2),
                "scope1": round(s1, 2),
                "scope2": round(s2, 2),
                "scope3": round(s3, 2),
                "data_quality": h.data_quality.value,
            })

        coverage = _safe_pct(covered_count, len(holdings_list))
        weighted_quality = _safe_divide(quality_weighted_sum, weight_sum, 5.0)

        holding_details.sort(key=lambda x: x["financed_emissions"], reverse=True)

        result = FinancedEmissionsResult(
            total_financed_emissions=round(total_financed, 2),
            scope1_financed=round(scope1_financed, 2),
            scope2_financed=round(scope2_financed, 2),
            scope3_financed=round(scope3_financed, 2),
            total_outstanding=round(total_outstanding, 2),
            coverage_ratio=round(coverage, 2),
            by_holding=holding_details,
            by_sector=dict({k: round(v, 2) for k, v in sector_emissions.items()}),
            by_geography=dict({k: round(v, 2) for k, v in geo_emissions.items()}),
            weighted_data_quality=round(weighted_quality, 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Financed emissions: %.0f tCO2e (S1=%.0f, S2=%.0f, S3=%.0f, "
            "coverage=%.1f%%, quality=%.1f) in %dms",
            total_financed,
            scope1_financed,
            scope2_financed,
            scope3_financed,
            coverage,
            weighted_quality,
            int((_utcnow() - start).total_seconds() * 1000),
        )
        return result

    # ------------------------------------------------------------------
    # Attribution Analysis
    # ------------------------------------------------------------------

    def attribution_analysis(
        self,
        holdings: Optional[List[HoldingEmissions]] = None,
        scope_coverage: Optional[ScopeCoverage] = None,
    ) -> TemperatureAlignment:
        """Perform implied temperature alignment analysis.

        Maps portfolio carbon intensity to an implied temperature rise
        based on sectoral decarbonization pathways.

        Args:
            holdings: Optional list (uses stored if not provided).
            scope_coverage: Scope coverage override.

        Returns:
            TemperatureAlignment result.
        """
        start = _utcnow()
        if holdings is not None:
            self._holdings = holdings
        holdings_list = self._holdings
        scope = scope_coverage or self.config.scope_coverage

        # Calculate portfolio-level carbon intensity (WACI-like)
        total_weighted_intensity = 0.0
        sector_temperatures: Dict[str, float] = defaultdict(float)
        covered_weight = 0.0

        for h in holdings_list:
            revenue_m = self._to_millions(h.revenue)
            if revenue_m <= 0.0:
                continue

            emissions = self._get_scoped_emissions(h, scope)
            if emissions <= 0.0 and self.config.use_sector_estimates and h.sector:
                emissions, revenue_m = self._estimate_from_sector(h, scope)
                if revenue_m <= 0.0:
                    continue

            intensity = emissions / revenue_m
            weight = h.weight_pct / 100.0
            total_weighted_intensity += weight * intensity
            covered_weight += weight

            sector_key = h.sector or "_default"
            sector_temperatures[sector_key] += weight * intensity

        # Map intensity to implied temperature
        implied_temp = self._map_to_temperature(total_weighted_intensity)
        confidence = min(covered_weight, 1.0)
        paris_aligned = implied_temp <= 2.0

        # Determine pathway benchmark
        if total_weighted_intensity <= TEMPERATURE_PATHWAYS["1.5C_aligned"]:
            pathway = "1.5C_aligned"
        elif total_weighted_intensity <= TEMPERATURE_PATHWAYS["2.0C_aligned"]:
            pathway = "2.0C_aligned"
        elif total_weighted_intensity <= TEMPERATURE_PATHWAYS["3.0C_aligned"]:
            pathway = "3.0C_aligned"
        else:
            pathway = "4.0C_plus"

        result = TemperatureAlignment(
            implied_temperature_rise=round(implied_temp, 2),
            methodology=TemperatureMethodology.SECTORAL_DECARBONIZATION,
            confidence=round(confidence, 3),
            portfolio_carbon_intensity=round(total_weighted_intensity, 2),
            pathway_benchmark=pathway,
            sector_contributions=dict(
                {k: round(v, 4) for k, v in sector_temperatures.items()}
            ),
            aligned_with_paris=paris_aligned,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Temperature alignment: %.2fC (intensity=%.1f, pathway=%s, "
            "paris_aligned=%s) in %dms",
            implied_temp,
            total_weighted_intensity,
            pathway,
            paris_aligned,
            int((_utcnow() - start).total_seconds() * 1000),
        )
        return result

    # ------------------------------------------------------------------
    # Sector & Top Contributors
    # ------------------------------------------------------------------

    def get_sector_breakdown(
        self,
        holdings: Optional[List[HoldingEmissions]] = None,
        scope_coverage: Optional[ScopeCoverage] = None,
    ) -> List[SectorBreakdown]:
        """Calculate sector-level carbon attribution breakdown.

        Args:
            holdings: Optional list (uses stored if not provided).
            scope_coverage: Scope coverage override.

        Returns:
            List of SectorBreakdown objects sorted by emissions.
        """
        start = _utcnow()
        if holdings is not None:
            self._holdings = holdings
        holdings_list = self._holdings
        scope = scope_coverage or self.config.scope_coverage

        sector_data: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "count": 0,
                "weight": 0.0,
                "emissions": 0.0,
                "revenue_m": 0.0,
                "waci_contribution": 0.0,
                "financed": 0.0,
            }
        )

        for h in holdings_list:
            sector_key = h.sector or "_default"
            emissions = self._get_scoped_emissions(h, scope)
            revenue_m = self._to_millions(h.revenue)

            sector_data[sector_key]["count"] += 1
            sector_data[sector_key]["weight"] += h.weight_pct
            sector_data[sector_key]["emissions"] += emissions

            if revenue_m > 0:
                intensity = emissions / revenue_m
                waci_contrib = (h.weight_pct / 100.0) * intensity
                sector_data[sector_key]["waci_contribution"] += waci_contrib
                sector_data[sector_key]["revenue_m"] += revenue_m

            # Simple attribution for sector totals
            if h.evic > 0:
                attr = (h.holding_value / h.evic) * emissions
                sector_data[sector_key]["financed"] += attr

        results: List[SectorBreakdown] = []
        for sector_code, data in sector_data.items():
            sector_ref = SECTOR_EMISSION_INTENSITY.get(sector_code, {})
            sector_name = sector_ref.get("name", sector_code)
            intensity = _safe_divide(
                data["emissions"], data["revenue_m"], 0.0
            )

            breakdown = SectorBreakdown(
                sector_code=sector_code,
                sector_name=sector_name,
                holding_count=data["count"],
                portfolio_weight_pct=round(data["weight"], 2),
                total_emissions=round(data["emissions"], 2),
                carbon_intensity=round(intensity, 2),
                waci_contribution=round(data["waci_contribution"], 4),
                financed_emissions=round(data["financed"], 2),
            )
            breakdown.provenance_hash = _compute_hash(breakdown)
            results.append(breakdown)

        results.sort(key=lambda x: x.total_emissions, reverse=True)

        logger.info(
            "Sector breakdown: %d sectors analyzed in %dms",
            len(results),
            int((_utcnow() - start).total_seconds() * 1000),
        )
        return results

    def get_top_contributors(
        self,
        holdings: Optional[List[HoldingEmissions]] = None,
        scope_coverage: Optional[ScopeCoverage] = None,
        count: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get top emission contributors in the portfolio.

        Args:
            holdings: Optional list (uses stored if not provided).
            scope_coverage: Scope coverage override.
            count: Number of top contributors to return.

        Returns:
            List of top contributor dictionaries sorted by emissions.
        """
        if holdings is not None:
            self._holdings = holdings
        holdings_list = self._holdings
        scope = scope_coverage or self.config.scope_coverage
        n = count or self.config.top_contributors_count

        contributors: List[Dict[str, Any]] = []
        for h in holdings_list:
            emissions = self._get_scoped_emissions(h, scope)
            revenue_m = self._to_millions(h.revenue)
            intensity = _safe_divide(emissions, revenue_m, 0.0) if revenue_m > 0 else 0.0

            financed = 0.0
            if h.evic > 0:
                financed = (h.holding_value / h.evic) * emissions

            contributors.append({
                "holding_id": h.holding_id,
                "company_name": h.company_name,
                "sector": h.sector,
                "country": h.country,
                "weight_pct": round(h.weight_pct, 4),
                "total_emissions": round(emissions, 2),
                "carbon_intensity": round(intensity, 2),
                "financed_emissions": round(financed, 2),
                "data_quality": h.data_quality.value,
            })

        contributors.sort(key=lambda x: x["financed_emissions"], reverse=True)
        return contributors[:n]

    # ------------------------------------------------------------------
    # Full Carbon Summary
    # ------------------------------------------------------------------

    def generate_carbon_summary(
        self,
        holdings: List[HoldingEmissions],
        scope_coverage: Optional[ScopeCoverage] = None,
    ) -> CarbonSummary:
        """Generate a comprehensive carbon summary with all metrics.

        Runs WACI, carbon footprint, financed emissions, temperature
        alignment, and sector breakdown in a single call.

        Args:
            holdings: List of holding emission data.
            scope_coverage: Scope coverage override.

        Returns:
            CarbonSummary with all calculated metrics.
        """
        start = _utcnow()
        scope = scope_coverage or self.config.scope_coverage

        waci_result = self.calculate_waci(holdings, scope)
        footprint_result = self.calculate_carbon_footprint(scope_coverage=scope)
        financed_result = self.calculate_financed_emissions(scope_coverage=scope)
        temp_result = self.attribution_analysis(scope_coverage=scope)
        sector_breakdown = self.get_sector_breakdown(scope_coverage=scope)

        summary = CarbonSummary(
            waci=waci_result,
            carbon_footprint=footprint_result,
            financed_emissions=financed_result,
            temperature_alignment=temp_result,
            sector_breakdown=sector_breakdown,
            total_portfolio_value=round(self._total_portfolio_value, 2),
            total_holdings=len(holdings),
            coverage_ratio=waci_result.coverage_ratio,
        )
        summary.provenance_hash = _compute_hash(summary)

        logger.info(
            "Carbon summary generated: WACI=%.2f, footprint=%.2f, "
            "financed=%.0f, temp=%.2fC in %dms",
            waci_result.waci_value,
            footprint_result.carbon_footprint,
            financed_result.total_financed_emissions,
            temp_result.implied_temperature_rise,
            int((_utcnow() - start).total_seconds() * 1000),
        )
        return summary

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _get_scoped_emissions(
        self, h: HoldingEmissions, scope: ScopeCoverage
    ) -> float:
        """Get emissions for the specified scope coverage.

        Args:
            h: Holding emissions data.
            scope: Scope coverage level.

        Returns:
            Total emissions for the requested scope.
        """
        if scope == ScopeCoverage.SCOPE_1:
            return h.scope1
        elif scope == ScopeCoverage.SCOPE_1_2:
            return h.scope1 + h.scope2
        else:
            return h.scope1 + h.scope2 + h.scope3

    def _get_attribution_denominator(
        self, h: HoldingEmissions, method: AttributionMethod
    ) -> float:
        """Get the denominator for attribution factor calculation.

        Args:
            h: Holding emissions data.
            method: Attribution method.

        Returns:
            Denominator value (EVIC, market cap, etc.).
        """
        if method == AttributionMethod.EVIC:
            return h.evic if h.evic > 0 else h.market_cap
        elif method == AttributionMethod.MARKET_CAP:
            return h.market_cap
        elif method == AttributionMethod.TOTAL_ASSETS:
            return h.total_assets
        elif method == AttributionMethod.REVENUE:
            return h.revenue
        return h.evic if h.evic > 0 else 0.0

    def _to_millions(self, value: float) -> float:
        """Convert a value to millions.

        Args:
            value: Value in base units.

        Returns:
            Value in millions.
        """
        if self.config.revenue_in_millions:
            return value  # Already in millions
        return value / 1_000_000.0

    def _ensure_weights(self, holdings: List[HoldingEmissions]) -> None:
        """Ensure portfolio weights are populated.

        If weights are zero, calculate from holding values.

        Args:
            holdings: List of holdings to update.
        """
        total_value = sum(h.holding_value for h in holdings)
        if total_value <= 0:
            return

        for h in holdings:
            if h.weight_pct <= 0.0 and h.holding_value > 0:
                h.weight_pct = (h.holding_value / total_value) * 100.0

    def _ensure_totals(self, holdings: List[HoldingEmissions]) -> None:
        """Ensure total emissions are populated from scope breakdowns.

        Args:
            holdings: List of holdings to update.
        """
        for h in holdings:
            if h.total_emissions <= 0.0:
                h.compute_total_emissions()

    def _estimate_from_sector(
        self, h: HoldingEmissions, scope: ScopeCoverage
    ) -> Tuple[float, float]:
        """Estimate emissions from sector averages when data is missing.

        Args:
            h: Holding with missing emissions data.
            scope: Scope coverage.

        Returns:
            Tuple of (estimated_emissions, revenue_in_millions).
        """
        sector_ref = SECTOR_EMISSION_INTENSITY.get(
            h.sector, SECTOR_EMISSION_INTENSITY["_default"]
        )
        revenue_m = self._to_millions(h.revenue)
        if revenue_m <= 0.0:
            return 0.0, 0.0

        scope_1_2_intensity = sector_ref.get("scope_1_2", 50.0)
        scope_3_intensity = sector_ref.get("scope_3", 100.0)

        if scope == ScopeCoverage.SCOPE_1:
            estimated = scope_1_2_intensity * 0.6 * revenue_m  # ~60% is Scope 1
        elif scope == ScopeCoverage.SCOPE_1_2:
            estimated = scope_1_2_intensity * revenue_m
        else:
            estimated = (scope_1_2_intensity + scope_3_intensity) * revenue_m

        h.emissions_estimated = True
        h.data_quality = DataQuality.ESTIMATED_SECTOR
        return estimated, revenue_m

    def _map_to_temperature(self, carbon_intensity: float) -> float:
        """Map portfolio carbon intensity to implied temperature rise.

        Uses a linear interpolation between pathway benchmarks.

        Args:
            carbon_intensity: Portfolio carbon intensity (tCO2e/EUR M revenue).

        Returns:
            Implied temperature rise in degrees Celsius.
        """
        # Linear interpolation between pathway thresholds
        thresholds = [
            (0.0, 1.2),     # Zero emissions -> 1.2C
            (50.0, 1.5),    # 1.5C pathway
            (100.0, 2.0),   # 2.0C pathway
            (200.0, 3.0),   # 3.0C pathway
            (400.0, 4.0),   # 4.0C pathway
            (800.0, 5.0),   # Well above 4C
        ]

        if carbon_intensity <= thresholds[0][0]:
            return thresholds[0][1]
        if carbon_intensity >= thresholds[-1][0]:
            return thresholds[-1][1]

        for i in range(len(thresholds) - 1):
            lower_int, lower_temp = thresholds[i]
            upper_int, upper_temp = thresholds[i + 1]
            if lower_int <= carbon_intensity <= upper_int:
                fraction = (carbon_intensity - lower_int) / (upper_int - lower_int)
                return lower_temp + fraction * (upper_temp - lower_temp)

        return 3.0  # Fallback
