# -*- coding: utf-8 -*-
"""
CarbonPricingEngine - PACK-016 ESRS E1 Climate Engine 7
=========================================================

Manages internal carbon pricing disclosure per ESRS E1-8.

Under the European Sustainability Reporting Standards (ESRS), ESRS E1-8
requires the undertaking to disclose whether it applies internal carbon
pricing, and if so, to provide details on the type of internal carbon
pricing scheme, the price per tonne of CO2e, the scope of activities
covered, and how the internal carbon price informs investment and
operational decisions.

ESRS E1-8 Framework:
    - Para 59: The undertaking shall disclose whether it applies
      internal carbon pricing and, if so, how.
    - Para 60: The disclosure shall include: (a) the type of internal
      carbon pricing scheme; (b) the specific price applied per tCO2e;
      (c) the scope of emissions and activities covered.
    - Para 61: The undertaking shall explain how the internal carbon
      price is integrated into its decision-making processes.

Application Requirements (AR E1-69 through AR E1-73):
    - AR E1-69: Types of internal carbon pricing include shadow price,
      internal carbon fee or levy, implicit carbon price derived from
      low-carbon investment analysis.
    - AR E1-70: Shadow price is used in investment appraisals without
      actual cash flow; internal fee involves actual charge.
    - AR E1-71: The carbon price should be disclosed in EUR per tCO2e.
    - AR E1-72: Coverage should indicate which Scopes and activities
      are subject to the internal carbon price.
    - AR E1-73: The undertaking may disclose multiple price scenarios
      (current, expected future, social cost of carbon).

Regulatory References:
    - EU Delegated Regulation 2023/2772 (ESRS Set 1)
    - ESRS E1 Climate Change, Para 59-61
    - ESRS E1 Application Requirements AR E1-69 through AR E1-73
    - EU ETS Directive 2003/87/EC (as amended)
    - IEA World Energy Outlook carbon price assumptions

Zero-Hallucination:
    - Weighted average price uses deterministic summation
    - Coverage calculation uses deterministic ratio
    - Shadow price analysis uses deterministic comparison
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

class PricingMechanism(str, Enum):
    """Type of internal carbon pricing mechanism per AR E1-69.

    Each mechanism represents a different approach to incorporating
    the cost of carbon into business decisions.
    """
    SHADOW_PRICE = "shadow_price"
    INTERNAL_FEE = "internal_fee"
    IMPLICIT_PRICE = "implicit_price"
    OFFSET_PRICE = "offset_price"

class PricingScope(str, Enum):
    """Scope of activities to which the carbon price applies per AR E1-72.

    Indicates which types of business decisions are influenced by
    the internal carbon price.
    """
    INVESTMENT_DECISIONS = "investment_decisions"
    PROCUREMENT = "procurement"
    PRODUCT_PRICING = "product_pricing"
    INTERNAL_TRANSFER = "internal_transfer"
    ALL = "all"

class CurrencyCode(str, Enum):
    """Currency codes for carbon pricing per AR E1-71.

    ESRS requires disclosure in EUR but the undertaking may also
    disclose in its functional currency.
    """
    EUR = "EUR"
    USD = "USD"
    GBP = "GBP"
    CHF = "CHF"
    JPY = "JPY"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Required ESRS E1-8 data points.
E1_8_DATAPOINTS: Dict[str, str] = {
    "e1_8_dp01": "Whether the undertaking applies internal carbon pricing",
    "e1_8_dp02": "Type of internal carbon pricing scheme (shadow price, internal fee, implicit price)",
    "e1_8_dp03": "The specific carbon price per tCO2e applied",
    "e1_8_dp04": "Currency in which the carbon price is expressed",
    "e1_8_dp05": "Effective date of the current carbon price",
    "e1_8_dp06": "Scope of emissions covered by the internal carbon price (Scope 1, 2, 3)",
    "e1_8_dp07": "Percentage of total emissions covered by the internal carbon price",
    "e1_8_dp08": "Total emissions covered by the internal carbon price (tCO2e)",
    "e1_8_dp09": "How the internal carbon price is applied to investment decisions",
    "e1_8_dp10": "How the internal carbon price is applied to procurement decisions",
    "e1_8_dp11": "Revenue generated from internal carbon fee/levy (if applicable)",
    "e1_8_dp12": "Whether shadow price scenarios are used for sensitivity analysis",
    "e1_8_dp13": "Shadow price trajectory (if applicable, price by year to 2050)",
}

# Reference carbon prices from major compliance and voluntary markets.
# All prices in EUR per tCO2e as of early 2026.
REFERENCE_CARBON_PRICES: Dict[str, Dict[str, Any]] = {
    "eu_ets": {
        "name": "EU Emissions Trading System",
        "price_eur": Decimal("85.00"),
        "currency": "EUR",
        "source": "EU ETS Phase IV (2021-2030)",
        "market_type": "compliance",
        "description": "EU ETS carbon allowance price (EUA)",
        "last_updated": "2026-01",
    },
    "uk_ets": {
        "name": "UK Emissions Trading Scheme",
        "price_eur": Decimal("58.00"),
        "currency": "EUR",
        "source": "UK ETS (post-Brexit)",
        "market_type": "compliance",
        "description": "UK ETS carbon allowance price (UKA)",
        "last_updated": "2026-01",
    },
    "california_cat": {
        "name": "California Cap-and-Trade",
        "price_eur": Decimal("32.00"),
        "currency": "EUR",
        "source": "California Cap-and-Trade Program",
        "market_type": "compliance",
        "description": "California carbon allowance price (CCA)",
        "last_updated": "2026-01",
    },
    "iea_nze_2030": {
        "name": "IEA Net Zero by 2050 - 2030 implied price",
        "price_eur": Decimal("130.00"),
        "currency": "EUR",
        "source": "IEA World Energy Outlook 2025 NZE Scenario",
        "market_type": "reference",
        "description": "IEA NZE scenario implied carbon price for 2030",
        "last_updated": "2025-10",
    },
    "iea_nze_2050": {
        "name": "IEA Net Zero by 2050 - 2050 implied price",
        "price_eur": Decimal("250.00"),
        "currency": "EUR",
        "source": "IEA World Energy Outlook 2025 NZE Scenario",
        "market_type": "reference",
        "description": "IEA NZE scenario implied carbon price for 2050",
        "last_updated": "2025-10",
    },
    "social_cost_carbon": {
        "name": "Social Cost of Carbon (US EPA)",
        "price_eur": Decimal("190.00"),
        "currency": "EUR",
        "source": "US EPA Social Cost of Greenhouse Gases (2023 update)",
        "market_type": "reference",
        "description": "Social cost of carbon at 2% discount rate",
        "last_updated": "2023-12",
    },
}

# Shadow price benchmarks for scenario analysis.
# Provides low, central, and high price trajectories from 2025 to 2050.
SHADOW_PRICE_BENCHMARKS: Dict[str, Dict[int, Decimal]] = {
    "low": {
        2025: Decimal("50.00"),
        2030: Decimal("75.00"),
        2035: Decimal("100.00"),
        2040: Decimal("125.00"),
        2045: Decimal("150.00"),
        2050: Decimal("175.00"),
    },
    "central": {
        2025: Decimal("85.00"),
        2030: Decimal("130.00"),
        2035: Decimal("175.00"),
        2040: Decimal("220.00"),
        2045: Decimal("265.00"),
        2050: Decimal("310.00"),
    },
    "high": {
        2025: Decimal("120.00"),
        2030: Decimal("200.00"),
        2035: Decimal("280.00"),
        2040: Decimal("360.00"),
        2045: Decimal("440.00"),
        2050: Decimal("520.00"),
    },
}

# Pricing mechanism descriptions for reporting.
MECHANISM_DESCRIPTIONS: Dict[str, str] = {
    "shadow_price": "Shadow carbon price used in investment appraisals and project "
                    "evaluations without generating actual cash flows",
    "internal_fee": "Internal carbon fee or levy charged to business units based "
                    "on their GHG emissions, generating actual internal revenue",
    "implicit_price": "Implicit carbon price derived from the cost of low-carbon "
                      "investments divided by the emissions they avoid",
    "offset_price": "Carbon offset price based on the average cost of carbon "
                    "credits purchased to compensate for emissions",
}

# Pricing scope descriptions.
SCOPE_DESCRIPTIONS: Dict[str, str] = {
    "investment_decisions": "Carbon price applied to evaluate capital investment proposals "
                           "and project business cases",
    "procurement": "Carbon price applied to procurement decisions and supplier selection "
                   "to incentivise low-carbon supply chains",
    "product_pricing": "Carbon price embedded in product/service pricing to reflect "
                       "the carbon cost to customers",
    "internal_transfer": "Carbon price used for internal transfer pricing between "
                         "business units or divisions",
    "all": "Carbon price applied across all business decisions (investment, procurement, "
           "product pricing, and internal transfer)",
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class CarbonPrice(BaseModel):
    """An internal carbon price per ESRS E1-8.

    Represents a specific carbon pricing mechanism applied by the
    undertaking, including the price level, scope of application,
    and emissions coverage.
    """
    price_id: str = Field(
        default_factory=_new_uuid,
        description="Unique price identifier",
    )
    mechanism: PricingMechanism = Field(
        ...,
        description="Type of carbon pricing mechanism",
    )
    price_per_tco2e: Decimal = Field(
        ...,
        description="Carbon price per tonne of CO2e",
        ge=0,
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.EUR,
        description="Currency in which the price is expressed",
    )
    effective_date: Optional[date] = Field(
        default=None,
        description="Date from which this price is effective",
    )
    scope: PricingScope = Field(
        default=PricingScope.ALL,
        description="Scope of business decisions to which the price applies",
    )
    emissions_covered_tco2e: Decimal = Field(
        default=Decimal("0.000"),
        description="Total emissions covered by this price (tCO2e)",
        ge=0,
    )
    emissions_covered_pct: Decimal = Field(
        default=Decimal("0.00"),
        description="Percentage of total emissions covered by this price",
        ge=0,
        le=Decimal("100.00"),
    )
    covers_scope_1: bool = Field(
        default=True,
        description="Whether this price covers Scope 1 emissions",
    )
    covers_scope_2: bool = Field(
        default=True,
        description="Whether this price covers Scope 2 emissions",
    )
    covers_scope_3: bool = Field(
        default=False,
        description="Whether this price covers Scope 3 emissions",
    )
    description: str = Field(
        default="",
        description="Description of how the carbon price is applied",
        max_length=5000,
    )
    revenue_generated: Decimal = Field(
        default=Decimal("0.00"),
        description="Revenue generated from internal fee/levy (if applicable)",
        ge=0,
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )

class ShadowPriceScenario(BaseModel):
    """A shadow carbon price scenario for sensitivity analysis per AR E1-73.

    Represents a price trajectory over time used to test the
    sensitivity of investment decisions to different carbon price
    assumptions.
    """
    scenario_id: str = Field(
        default_factory=_new_uuid,
        description="Unique scenario identifier",
    )
    scenario_name: str = Field(
        ...,
        description="Name of the scenario (e.g., 'IEA NZE', 'Low', 'Central', 'High')",
        min_length=1,
        max_length=200,
    )
    price_trajectory: Dict[int, Decimal] = Field(
        ...,
        description="Price trajectory mapping year to EUR/tCO2e",
    )
    applied_to: str = Field(
        default="",
        description="Description of what decisions this scenario is applied to",
        max_length=2000,
    )
    source: str = Field(
        default="",
        description="Source of the price trajectory",
        max_length=500,
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.EUR,
        description="Currency for the price trajectory",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )

    @field_validator("scenario_name")
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        """Validate that scenario name is not empty."""
        if not v.strip():
            raise ValueError("Scenario name cannot be empty")
        return v.strip()

class CarbonPricingResult(BaseModel):
    """Result of carbon pricing disclosure compilation per ESRS E1-8.

    Contains the complete inventory of carbon pricing mechanisms
    and shadow price scenarios with aggregated summaries.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this compilation",
    )
    calculated_at: datetime = Field(
        default_factory=utcnow,
        description="Timestamp of compilation (UTC)",
    )
    applies_carbon_pricing: bool = Field(
        default=False,
        description="Whether the undertaking applies internal carbon pricing",
    )
    mechanisms: List[CarbonPrice] = Field(
        default_factory=list,
        description="List of carbon pricing mechanisms",
    )
    shadow_price_scenarios: List[ShadowPriceScenario] = Field(
        default_factory=list,
        description="List of shadow price scenarios for sensitivity analysis",
    )
    total_mechanisms: int = Field(
        default=0,
        description="Total number of pricing mechanisms",
    )
    weighted_average_price: Decimal = Field(
        default=Decimal("0.00"),
        description="Emissions-weighted average carbon price (EUR/tCO2e)",
    )
    total_emissions_covered_tco2e: Decimal = Field(
        default=Decimal("0.000"),
        description="Total emissions covered by all mechanisms (tCO2e)",
    )
    coverage_pct: Decimal = Field(
        default=Decimal("0.00"),
        description="Combined coverage as percentage of total emissions",
    )
    total_revenue_from_fees: Decimal = Field(
        default=Decimal("0.00"),
        description="Total revenue generated from internal carbon fees/levies",
    )
    mechanisms_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of mechanisms by type",
    )
    price_range: Dict[str, str] = Field(
        default_factory=dict,
        description="Min and max carbon prices applied",
    )
    scope_coverage: Dict[str, bool] = Field(
        default_factory=dict,
        description="Which emission scopes are covered",
    )
    completeness_score: float = Field(
        default=0.0,
        description="Completeness score for E1-8 data points (0-100)",
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

class CarbonPricingEngine:
    """Internal carbon pricing engine per ESRS E1-8.

    Provides deterministic, zero-hallucination management of:
    - Carbon price registration and tracking
    - Weighted average price calculation
    - Emission coverage calculation
    - Shadow price scenario analysis
    - Completeness validation against E1-8 data points
    - Data point extraction for XBRL tagging

    All calculations are bit-perfect reproducible.  No LLM is used
    in any calculation path.

    Usage::

        engine = CarbonPricingEngine()
        price = CarbonPrice(
            mechanism=PricingMechanism.SHADOW_PRICE,
            price_per_tco2e=Decimal("100.00"),
            emissions_covered_tco2e=Decimal("50000.0"),
            emissions_covered_pct=Decimal("85.0"),
        )
        registered = engine.set_carbon_price(price)
        result = engine.build_pricing_disclosure(total_emissions=Decimal("58823.5"))
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise CarbonPricingEngine."""
        self._prices: List[CarbonPrice] = []
        self._scenarios: List[ShadowPriceScenario] = []
        logger.info(
            "CarbonPricingEngine v%s initialised", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Carbon Price Registration                                            #
    # ------------------------------------------------------------------ #

    def set_carbon_price(self, price: CarbonPrice) -> CarbonPrice:
        """Register an internal carbon price per ESRS E1-8.

        Args:
            price: CarbonPrice to register.

        Returns:
            Registered CarbonPrice with provenance hash.
        """
        t0 = time.perf_counter()

        if not price.price_id:
            price.price_id = _new_uuid()

        price.provenance_hash = _compute_hash(price)
        self._prices.append(price)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        logger.info(
            "Set carbon price: mechanism=%s, price=%s %s/tCO2e, "
            "coverage=%s tCO2e in %.3f ms",
            price.mechanism.value,
            price.price_per_tco2e,
            price.currency.value,
            price.emissions_covered_tco2e,
            elapsed_ms,
        )
        return price

    # ------------------------------------------------------------------ #
    # Scenario Registration                                                #
    # ------------------------------------------------------------------ #

    def add_scenario(
        self, scenario: ShadowPriceScenario
    ) -> ShadowPriceScenario:
        """Add a shadow price scenario for sensitivity analysis.

        Args:
            scenario: ShadowPriceScenario to add.

        Returns:
            Registered ShadowPriceScenario with provenance hash.
        """
        if not scenario.scenario_id:
            scenario.scenario_id = _new_uuid()

        scenario.provenance_hash = _compute_hash(scenario)
        self._scenarios.append(scenario)

        logger.info(
            "Added shadow price scenario: '%s' with %d price points",
            scenario.scenario_name,
            len(scenario.price_trajectory),
        )
        return scenario

    def add_benchmark_scenarios(self) -> List[ShadowPriceScenario]:
        """Add the standard low/central/high benchmark scenarios.

        Loads the three benchmark price trajectories from
        SHADOW_PRICE_BENCHMARKS and registers them as scenarios.

        Returns:
            List of registered ShadowPriceScenario objects.
        """
        scenarios: List[ShadowPriceScenario] = []

        for name, trajectory in SHADOW_PRICE_BENCHMARKS.items():
            scenario = ShadowPriceScenario(
                scenario_name=f"{name.capitalize()} Carbon Price Scenario",
                price_trajectory=trajectory,
                applied_to="Investment appraisals and capital allocation decisions",
                source=f"GreenLang {name} scenario based on IEA/EU ETS trajectories",
            )
            registered = self.add_scenario(scenario)
            scenarios.append(registered)

        logger.info("Added %d benchmark scenarios", len(scenarios))
        return scenarios

    # ------------------------------------------------------------------ #
    # Pricing Disclosure Builder                                           #
    # ------------------------------------------------------------------ #

    def build_pricing_disclosure(
        self,
        prices: Optional[List[CarbonPrice]] = None,
        scenarios: Optional[List[ShadowPriceScenario]] = None,
        total_emissions: Optional[Decimal] = None,
    ) -> CarbonPricingResult:
        """Build the complete carbon pricing disclosure per E1-8.

        Aggregates all pricing mechanisms and scenarios into a single
        result with weighted average price, coverage, and summaries.

        Args:
            prices: List of prices (uses internal registry if None).
            scenarios: List of scenarios (uses internal registry if None).
            total_emissions: Total emissions for coverage calculation.

        Returns:
            CarbonPricingResult with complete aggregation.
        """
        t0 = time.perf_counter()

        if prices is None:
            prices = list(self._prices)
        if scenarios is None:
            scenarios = list(self._scenarios)

        applies_pricing = len(prices) > 0

        # Weighted average price
        weighted_avg = self.calculate_weighted_average_price(prices)

        # Coverage
        total_covered = Decimal("0.000")
        total_revenue = Decimal("0.00")
        mechanisms_by_type: Dict[str, int] = {}
        min_price = Decimal("999999.99")
        max_price = Decimal("0.00")
        scope_coverage = {
            "scope_1": False,
            "scope_2": False,
            "scope_3": False,
        }

        for price in prices:
            total_covered += price.emissions_covered_tco2e
            total_revenue += price.revenue_generated

            mech_key = price.mechanism.value
            mechanisms_by_type[mech_key] = (
                mechanisms_by_type.get(mech_key, 0) + 1
            )

            if price.price_per_tco2e < min_price:
                min_price = price.price_per_tco2e
            if price.price_per_tco2e > max_price:
                max_price = price.price_per_tco2e

            if price.covers_scope_1:
                scope_coverage["scope_1"] = True
            if price.covers_scope_2:
                scope_coverage["scope_2"] = True
            if price.covers_scope_3:
                scope_coverage["scope_3"] = True

        # Coverage percentage
        coverage_pct = Decimal("0.00")
        if total_emissions and total_emissions > 0:
            coverage_pct = self.calculate_coverage(prices, total_emissions)
        elif prices:
            # Use max coverage_pct from individual prices
            max_pct = max(p.emissions_covered_pct for p in prices)
            coverage_pct = max_pct

        # Price range
        if not prices:
            min_price = Decimal("0.00")
            max_price = Decimal("0.00")

        price_range = {
            "min": str(_round_val(min_price, 2)),
            "max": str(_round_val(max_price, 2)),
            "currency": "EUR",
        }

        # Completeness
        completeness = self._calculate_completeness(prices, scenarios)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = CarbonPricingResult(
            applies_carbon_pricing=applies_pricing,
            mechanisms=prices,
            shadow_price_scenarios=scenarios,
            total_mechanisms=len(prices),
            weighted_average_price=_round_val(weighted_avg, 2),
            total_emissions_covered_tco2e=_round_val(total_covered, 3),
            coverage_pct=_round_val(coverage_pct, 2),
            total_revenue_from_fees=_round_val(total_revenue, 2),
            mechanisms_by_type=mechanisms_by_type,
            price_range=price_range,
            scope_coverage=scope_coverage,
            completeness_score=completeness,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Built pricing disclosure: %d mechanisms, weighted avg=%s EUR/tCO2e, "
            "coverage=%s%%, %d scenarios in %.3f ms",
            len(prices),
            weighted_avg,
            coverage_pct,
            len(scenarios),
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # Weighted Average Price                                               #
    # ------------------------------------------------------------------ #

    def calculate_weighted_average_price(
        self, prices: List[CarbonPrice]
    ) -> Decimal:
        """Calculate emissions-weighted average carbon price.

        The weighted average gives more weight to pricing mechanisms
        that cover a larger share of total emissions.

        Formula (deterministic):
            weighted_avg = sum(price_i * coverage_i) / sum(coverage_i)

        If no emissions are covered, returns the simple arithmetic
        mean of all prices.

        Args:
            prices: List of CarbonPrice mechanisms.

        Returns:
            Weighted average price as Decimal (EUR/tCO2e).
        """
        if not prices:
            return Decimal("0.00")

        total_weighted = Decimal("0.00")
        total_coverage = Decimal("0.000")

        for price in prices:
            total_weighted += (
                price.price_per_tco2e * price.emissions_covered_tco2e
            )
            total_coverage += price.emissions_covered_tco2e

        if total_coverage > 0:
            return _round_val(total_weighted / total_coverage, 2)

        # Fallback: simple average if no emissions coverage data
        total_price = sum(p.price_per_tco2e for p in prices)
        return _round_val(total_price / _decimal(len(prices)), 2)

    # ------------------------------------------------------------------ #
    # Coverage Calculation                                                 #
    # ------------------------------------------------------------------ #

    def calculate_coverage(
        self, prices: List[CarbonPrice], total_emissions: Decimal
    ) -> Decimal:
        """Calculate the percentage of total emissions covered by carbon pricing.

        Args:
            prices: List of CarbonPrice mechanisms.
            total_emissions: Total emissions (tCO2e) for the denominator.

        Returns:
            Coverage percentage as Decimal (0-100).
        """
        if total_emissions <= 0:
            return Decimal("0.00")

        total_covered = sum(
            p.emissions_covered_tco2e for p in prices
        )

        # Cap at 100% to avoid overlapping coverage overcounting
        pct = (total_covered / total_emissions) * Decimal("100")
        if pct > Decimal("100.00"):
            pct = Decimal("100.00")

        return _round_val(pct, 2)

    # ------------------------------------------------------------------ #
    # Shadow Price Analysis                                                #
    # ------------------------------------------------------------------ #

    def run_shadow_price_analysis(
        self,
        scenarios: Optional[List[ShadowPriceScenario]] = None,
        target_years: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Run shadow price scenario analysis per AR E1-73.

        Compares price trajectories across scenarios for specified
        target years and calculates price differentials.

        Args:
            scenarios: List of scenarios (uses internal registry if None).
            target_years: Years to analyse (default [2025, 2030, 2035, 2040, 2050]).

        Returns:
            Dict with scenario comparison and analysis.
        """
        t0 = time.perf_counter()

        if scenarios is None:
            scenarios = list(self._scenarios)
        if target_years is None:
            target_years = [2025, 2030, 2035, 2040, 2050]

        analysis: Dict[str, Any] = {
            "total_scenarios": len(scenarios),
            "target_years": target_years,
            "scenarios": {},
            "year_comparison": {},
            "reference_prices": {},
        }

        # Scenario trajectories
        for scenario in scenarios:
            traj = {}
            for year in target_years:
                price = scenario.price_trajectory.get(year)
                if price is not None:
                    traj[str(year)] = str(_round_val(price, 2))
                else:
                    # Interpolate if year is between available points
                    interp = self._interpolate_price(
                        scenario.price_trajectory, year
                    )
                    traj[str(year)] = str(_round_val(interp, 2))

            analysis["scenarios"][scenario.scenario_name] = {
                "trajectory": traj,
                "source": scenario.source,
                "applied_to": scenario.applied_to,
            }

        # Year-by-year comparison
        for year in target_years:
            year_prices: Dict[str, str] = {}
            for scenario in scenarios:
                price = scenario.price_trajectory.get(year)
                if price is not None:
                    year_prices[scenario.scenario_name] = str(
                        _round_val(price, 2)
                    )
                else:
                    interp = self._interpolate_price(
                        scenario.price_trajectory, year
                    )
                    year_prices[scenario.scenario_name] = str(
                        _round_val(interp, 2)
                    )

            analysis["year_comparison"][str(year)] = year_prices

        # Reference prices for context
        for ref_id, ref_data in REFERENCE_CARBON_PRICES.items():
            analysis["reference_prices"][ref_id] = {
                "name": ref_data["name"],
                "price_eur": str(ref_data["price_eur"]),
                "source": ref_data["source"],
                "market_type": ref_data["market_type"],
            }

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        analysis["processing_time_ms"] = elapsed_ms
        analysis["provenance_hash"] = _compute_hash(analysis)

        logger.info(
            "Shadow price analysis: %d scenarios, %d years in %.3f ms",
            len(scenarios),
            len(target_years),
            elapsed_ms,
        )
        return analysis

    # ------------------------------------------------------------------ #
    # Completeness Validation                                              #
    # ------------------------------------------------------------------ #

    def validate_completeness(
        self, result: CarbonPricingResult
    ) -> Dict[str, Any]:
        """Validate completeness of E1-8 data points.

        Args:
            result: CarbonPricingResult to validate.

        Returns:
            Dict with data point coverage and completeness score.
        """
        datapoints_status: Dict[str, Dict[str, Any]] = {}
        covered = 0

        prices = result.mechanisms
        scenarios = result.shadow_price_scenarios

        checks = {
            "e1_8_dp01": True,  # Always disclosed (yes/no)
            "e1_8_dp02": len(result.mechanisms_by_type) > 0,
            "e1_8_dp03": any(p.price_per_tco2e > 0 for p in prices),
            "e1_8_dp04": any(p.currency is not None for p in prices),
            "e1_8_dp05": any(
                p.effective_date is not None for p in prices
            ),
            "e1_8_dp06": any(
                p.covers_scope_1 or p.covers_scope_2 or p.covers_scope_3
                for p in prices
            ) if prices else False,
            "e1_8_dp07": result.coverage_pct > 0,
            "e1_8_dp08": result.total_emissions_covered_tco2e > 0,
            "e1_8_dp09": any(
                p.scope in (PricingScope.INVESTMENT_DECISIONS, PricingScope.ALL)
                for p in prices
            ) if prices else False,
            "e1_8_dp10": any(
                p.scope in (PricingScope.PROCUREMENT, PricingScope.ALL)
                for p in prices
            ) if prices else False,
            "e1_8_dp11": result.total_revenue_from_fees > 0 or any(
                p.mechanism == PricingMechanism.SHADOW_PRICE for p in prices
            ),
            "e1_8_dp12": len(scenarios) > 0,
            "e1_8_dp13": any(
                len(s.price_trajectory) > 0 for s in scenarios
            ) if scenarios else False,
        }

        for dp_id, dp_label in E1_8_DATAPOINTS.items():
            is_covered = checks.get(dp_id, False)
            if is_covered:
                covered += 1
            datapoints_status[dp_id] = {
                "label": dp_label,
                "covered": is_covered,
                "status": "COMPLETE" if is_covered else "MISSING",
            }

        total = len(E1_8_DATAPOINTS)
        score = _round2(
            _safe_divide(float(covered), float(total), 0.0) * 100.0
        )

        return {
            "disclosure_requirement": "E1-8",
            "title": "Internal carbon pricing",
            "total_datapoints": total,
            "covered_datapoints": covered,
            "missing_datapoints": total - covered,
            "completeness_score": score,
            "datapoints": datapoints_status,
            "provenance_hash": _compute_hash(datapoints_status),
        }

    # ------------------------------------------------------------------ #
    # E1-8 Data Point Extraction                                           #
    # ------------------------------------------------------------------ #

    def get_e1_8_datapoints(
        self, result: CarbonPricingResult
    ) -> Dict[str, Any]:
        """Extract structured E1-8 data points for XBRL tagging.

        Args:
            result: CarbonPricingResult to extract from.

        Returns:
            Dict mapping data point IDs to values.
        """
        prices = result.mechanisms
        scenarios = result.shadow_price_scenarios

        datapoints: Dict[str, Any] = {
            "e1_8_dp01": {
                "value": result.applies_carbon_pricing,
                "label": E1_8_DATAPOINTS["e1_8_dp01"],
                "xbrl_element": "esrs:InternalCarbonPricingApplied",
            },
            "e1_8_dp02": {
                "value": [
                    {
                        "mechanism": p.mechanism.value,
                        "description": MECHANISM_DESCRIPTIONS.get(
                            p.mechanism.value, ""
                        ),
                    }
                    for p in prices
                ],
                "label": E1_8_DATAPOINTS["e1_8_dp02"],
                "xbrl_element": "esrs:InternalCarbonPricingType",
            },
            "e1_8_dp03": {
                "value": [
                    {
                        "mechanism": p.mechanism.value,
                        "price_per_tco2e": str(p.price_per_tco2e),
                        "currency": p.currency.value,
                    }
                    for p in prices
                ],
                "label": E1_8_DATAPOINTS["e1_8_dp03"],
                "xbrl_element": "esrs:InternalCarbonPrice",
            },
            "e1_8_dp04": {
                "value": list(set(p.currency.value for p in prices))
                if prices else [],
                "label": E1_8_DATAPOINTS["e1_8_dp04"],
                "xbrl_element": "esrs:InternalCarbonPriceCurrency",
            },
            "e1_8_dp05": {
                "value": [
                    {
                        "mechanism": p.mechanism.value,
                        "effective_date": str(p.effective_date)
                        if p.effective_date else None,
                    }
                    for p in prices
                ],
                "label": E1_8_DATAPOINTS["e1_8_dp05"],
                "xbrl_element": "esrs:InternalCarbonPriceEffectiveDate",
            },
            "e1_8_dp06": {
                "value": result.scope_coverage,
                "label": E1_8_DATAPOINTS["e1_8_dp06"],
                "xbrl_element": "esrs:InternalCarbonPriceScopeCoverage",
            },
            "e1_8_dp07": {
                "value": str(result.coverage_pct),
                "label": E1_8_DATAPOINTS["e1_8_dp07"],
                "xbrl_element": "esrs:InternalCarbonPriceCoveragePct",
            },
            "e1_8_dp08": {
                "value": str(result.total_emissions_covered_tco2e),
                "unit": "tCO2e",
                "label": E1_8_DATAPOINTS["e1_8_dp08"],
                "xbrl_element": "esrs:InternalCarbonPriceEmissionsCovered",
            },
            "e1_8_dp09": {
                "value": any(
                    p.scope in (PricingScope.INVESTMENT_DECISIONS, PricingScope.ALL)
                    for p in prices
                ) if prices else False,
                "label": E1_8_DATAPOINTS["e1_8_dp09"],
                "xbrl_element": "esrs:InternalCarbonPriceInvestmentDecisions",
            },
            "e1_8_dp10": {
                "value": any(
                    p.scope in (PricingScope.PROCUREMENT, PricingScope.ALL)
                    for p in prices
                ) if prices else False,
                "label": E1_8_DATAPOINTS["e1_8_dp10"],
                "xbrl_element": "esrs:InternalCarbonPriceProcurement",
            },
            "e1_8_dp11": {
                "value": str(result.total_revenue_from_fees),
                "label": E1_8_DATAPOINTS["e1_8_dp11"],
                "xbrl_element": "esrs:InternalCarbonFeeRevenue",
            },
            "e1_8_dp12": {
                "value": len(scenarios) > 0,
                "label": E1_8_DATAPOINTS["e1_8_dp12"],
                "xbrl_element": "esrs:ShadowPriceScenariosUsed",
            },
            "e1_8_dp13": {
                "value": [
                    {
                        "scenario_name": s.scenario_name,
                        "trajectory": {
                            str(k): str(v)
                            for k, v in s.price_trajectory.items()
                        },
                    }
                    for s in scenarios
                ],
                "label": E1_8_DATAPOINTS["e1_8_dp13"],
                "xbrl_element": "esrs:ShadowPriceTrajectory",
            },
        }

        datapoints["provenance_hash"] = _compute_hash(datapoints)
        return datapoints

    # ------------------------------------------------------------------ #
    # Summary and Reporting Utilities                                      #
    # ------------------------------------------------------------------ #

    def get_price_summary(self, price: CarbonPrice) -> Dict[str, Any]:
        """Return a structured summary of a single carbon price.

        Args:
            price: CarbonPrice to summarise.

        Returns:
            Dict with price details.
        """
        return {
            "price_id": price.price_id,
            "mechanism": price.mechanism.value,
            "mechanism_description": MECHANISM_DESCRIPTIONS.get(
                price.mechanism.value, ""
            ),
            "price_per_tco2e": str(price.price_per_tco2e),
            "currency": price.currency.value,
            "effective_date": str(price.effective_date)
            if price.effective_date else None,
            "scope": price.scope.value,
            "scope_description": SCOPE_DESCRIPTIONS.get(
                price.scope.value, ""
            ),
            "emissions_covered_tco2e": str(price.emissions_covered_tco2e),
            "emissions_covered_pct": str(price.emissions_covered_pct),
            "covers_scope_1": price.covers_scope_1,
            "covers_scope_2": price.covers_scope_2,
            "covers_scope_3": price.covers_scope_3,
            "revenue_generated": str(price.revenue_generated),
            "provenance_hash": price.provenance_hash,
        }

    def compare_to_reference_prices(
        self, result: CarbonPricingResult
    ) -> Dict[str, Any]:
        """Compare the undertaking's carbon price to reference prices.

        Useful for benchmarking the internal carbon price against
        compliance market prices and policy-implied prices.

        Args:
            result: CarbonPricingResult with weighted average price.

        Returns:
            Dict with comparison to each reference price.
        """
        internal_price = result.weighted_average_price
        comparisons: Dict[str, Any] = {}

        for ref_id, ref_data in REFERENCE_CARBON_PRICES.items():
            ref_price = ref_data["price_eur"]
            diff = internal_price - ref_price
            diff_pct = Decimal("0.00")
            if ref_price > 0:
                diff_pct = _round_val(
                    (diff / ref_price) * Decimal("100"), 2
                )

            comparisons[ref_id] = {
                "name": ref_data["name"],
                "reference_price_eur": str(ref_price),
                "internal_price_eur": str(internal_price),
                "difference_eur": str(_round_val(diff, 2)),
                "difference_pct": str(diff_pct),
                "above_reference": internal_price >= ref_price,
                "market_type": ref_data["market_type"],
            }

        return {
            "internal_weighted_avg_price": str(internal_price),
            "comparisons": comparisons,
            "provenance_hash": _compute_hash(comparisons),
        }

    def get_reference_prices(self) -> Dict[str, Dict[str, Any]]:
        """Return all reference carbon prices.

        Returns:
            Dict mapping reference ID to price data.
        """
        return {
            ref_id: {
                "name": data["name"],
                "price_eur": str(data["price_eur"]),
                "source": data["source"],
                "market_type": data["market_type"],
                "description": data["description"],
            }
            for ref_id, data in REFERENCE_CARBON_PRICES.items()
        }

    def get_benchmark_scenarios(self) -> Dict[str, Dict[str, str]]:
        """Return the benchmark shadow price scenarios.

        Returns:
            Dict mapping scenario name to year-price trajectory.
        """
        return {
            name: {str(y): str(p) for y, p in traj.items()}
            for name, traj in SHADOW_PRICE_BENCHMARKS.items()
        }

    def clear_registry(self) -> None:
        """Clear all registered prices and scenarios."""
        self._prices.clear()
        self._scenarios.clear()
        logger.info("CarbonPricingEngine registry cleared")

    # ------------------------------------------------------------------ #
    # Private Helpers                                                      #
    # ------------------------------------------------------------------ #

    def _interpolate_price(
        self, trajectory: Dict[int, Decimal], target_year: int
    ) -> Decimal:
        """Linearly interpolate a price for a target year.

        If the target year is outside the trajectory range, the
        nearest endpoint is returned (no extrapolation).

        Args:
            trajectory: Year-to-price mapping.
            target_year: Year to interpolate for.

        Returns:
            Interpolated price as Decimal.
        """
        if not trajectory:
            return Decimal("0.00")

        years = sorted(trajectory.keys())

        # Exact match
        if target_year in trajectory:
            return trajectory[target_year]

        # Before first year - return first price
        if target_year <= years[0]:
            return trajectory[years[0]]

        # After last year - return last price
        if target_year >= years[-1]:
            return trajectory[years[-1]]

        # Find bracketing years
        lower_year = years[0]
        upper_year = years[-1]

        for y in years:
            if y <= target_year:
                lower_year = y
            if y >= target_year:
                upper_year = y
                break

        if lower_year == upper_year:
            return trajectory[lower_year]

        # Linear interpolation
        lower_price = trajectory[lower_year]
        upper_price = trajectory[upper_year]
        year_fraction = _decimal(target_year - lower_year) / _decimal(
            upper_year - lower_year
        )
        interpolated = lower_price + (upper_price - lower_price) * year_fraction

        return _round_val(interpolated, 2)

    def _calculate_completeness(
        self,
        prices: List[CarbonPrice],
        scenarios: List[ShadowPriceScenario],
    ) -> float:
        """Calculate E1-8 completeness score.

        Args:
            prices: List of carbon prices.
            scenarios: List of shadow price scenarios.

        Returns:
            Completeness score (0-100).
        """
        total = len(E1_8_DATAPOINTS)

        checks = [
            True,  # dp01: always disclosed
            len(prices) > 0,
            any(p.price_per_tco2e > 0 for p in prices) if prices else False,
            any(p.currency is not None for p in prices) if prices else False,
            any(p.effective_date is not None for p in prices) if prices else False,
            any(
                p.covers_scope_1 or p.covers_scope_2 or p.covers_scope_3
                for p in prices
            ) if prices else False,
            any(p.emissions_covered_pct > 0 for p in prices) if prices else False,
            any(p.emissions_covered_tco2e > 0 for p in prices) if prices else False,
            any(
                p.scope in (PricingScope.INVESTMENT_DECISIONS, PricingScope.ALL)
                for p in prices
            ) if prices else False,
            any(
                p.scope in (PricingScope.PROCUREMENT, PricingScope.ALL)
                for p in prices
            ) if prices else False,
            any(p.revenue_generated > 0 for p in prices) or any(
                p.mechanism == PricingMechanism.SHADOW_PRICE for p in prices
            ) if prices else False,
            len(scenarios) > 0,
            any(
                len(s.price_trajectory) > 0 for s in scenarios
            ) if scenarios else False,
        ]

        covered = sum(1 for c in checks if c)
        return _round2(_safe_divide(float(covered), float(total), 0.0) * 100.0)
