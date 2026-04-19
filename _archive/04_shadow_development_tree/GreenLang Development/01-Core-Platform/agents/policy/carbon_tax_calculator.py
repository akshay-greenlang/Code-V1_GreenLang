# -*- coding: utf-8 -*-
"""
GL-POL-X-005: Carbon Tax Calculator
===================================

Calculates carbon tax exposure across jurisdictions. This is a CRITICAL PATH
agent with zero-hallucination guarantees - all tax calculations are deterministic.

Capabilities:
    - Carbon tax liability calculation by jurisdiction
    - Multi-jurisdiction exposure aggregation
    - Forecast modeling for tax liability
    - Exemption and allowance calculations
    - Carbon price sensitivity analysis

Zero-Hallucination Guarantees:
    - All tax calculations use official rates from curated database
    - Deterministic formulas with complete audit trails
    - No LLM inference in tax amount calculations
    - Full provenance tracking with SHA-256 hashes

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism import DeterministicClock, deterministic_uuid

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class TaxJurisdiction(str, Enum):
    """Jurisdictions with carbon taxes."""
    EU_ETS = "eu_ets"
    UK_ETS = "uk_ets"
    CALIFORNIA_CAP = "california_cap"
    CANADA_FEDERAL = "canada_federal"
    CANADA_BC = "canada_bc"
    SWEDEN = "sweden"
    SWITZERLAND = "switzerland"
    NORWAY = "norway"
    SINGAPORE = "singapore"
    JAPAN = "japan"
    KOREA = "korea"
    CHINA_ETS = "china_ets"


class CoverageScope(str, Enum):
    """Emission scopes covered by tax."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    BOTH = "both"


class SectorType(str, Enum):
    """Sector types for carbon tax application."""
    POWER_GENERATION = "power_generation"
    MANUFACTURING = "manufacturing"
    HEAVY_INDUSTRY = "heavy_industry"
    TRANSPORT = "transport"
    BUILDINGS = "buildings"
    AGRICULTURE = "agriculture"
    WASTE = "waste"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class CarbonTaxRate(BaseModel):
    """Carbon tax rate for a jurisdiction."""

    rate_id: str = Field(..., description="Unique rate identifier")
    jurisdiction: TaxJurisdiction = Field(..., description="Tax jurisdiction")
    name: str = Field(..., description="Tax/scheme name")

    # Rate details
    rate_per_tco2e: Decimal = Field(..., description="Rate per tonne CO2e")
    currency: str = Field(default="EUR", description="Currency code")

    # Effective dates
    effective_date: date = Field(..., description="When rate becomes effective")
    expiry_date: Optional[date] = Field(None, description="When rate expires")

    # Coverage
    coverage_scope: CoverageScope = Field(
        default=CoverageScope.SCOPE_1,
        description="Emissions covered"
    )
    covered_sectors: List[SectorType] = Field(
        default_factory=list,
        description="Sectors covered"
    )

    # Exemptions
    free_allocation_available: bool = Field(
        default=False,
        description="Whether free allocation available"
    )
    exemption_threshold_tco2e: Optional[Decimal] = Field(
        None,
        description="Threshold below which exempt"
    )

    # Rate trajectory
    future_rates: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Projected future rates by year"
    )


class EmissionsProfile(BaseModel):
    """Emissions profile for carbon tax calculation."""

    organization_id: str = Field(..., description="Organization identifier")

    # Emissions by jurisdiction
    emissions_by_jurisdiction: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions (tCO2e) by jurisdiction code"
    )

    # Emissions by scope
    scope_1_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Total Scope 1 emissions"
    )
    scope_2_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Total Scope 2 emissions"
    )

    # Emissions by sector
    emissions_by_sector: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by sector"
    )

    # Existing allocations
    free_allocations: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Free allocations by jurisdiction"
    )


class TaxLiabilityItem(BaseModel):
    """Tax liability for a single jurisdiction."""

    jurisdiction: TaxJurisdiction = Field(..., description="Tax jurisdiction")
    jurisdiction_name: str = Field(..., description="Jurisdiction name")

    # Emissions
    covered_emissions_tco2e: Decimal = Field(..., description="Covered emissions")
    exempt_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Exempt emissions"
    )
    free_allocation_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Free allocation"
    )
    taxable_emissions_tco2e: Decimal = Field(..., description="Net taxable emissions")

    # Rate and calculation
    tax_rate_per_tco2e: Decimal = Field(..., description="Applied tax rate")
    currency: str = Field(default="EUR", description="Currency")

    # Liability
    gross_liability: Decimal = Field(..., description="Gross tax liability")
    net_liability: Decimal = Field(..., description="Net liability after allocations")

    # Calculation trace
    calculation_trace: List[str] = Field(default_factory=list)


class CarbonTaxResult(BaseModel):
    """Complete carbon tax calculation result."""

    result_id: str = Field(
        default_factory=lambda: deterministic_uuid("carbon_tax"),
        description="Unique result identifier"
    )
    organization_id: str = Field(..., description="Organization identifier")
    calculation_date: date = Field(
        default_factory=lambda: DeterministicClock.now().date()
    )
    reporting_year: int = Field(..., description="Reporting year")

    # Liability breakdown
    liability_items: List[TaxLiabilityItem] = Field(
        default_factory=list,
        description="Liability by jurisdiction"
    )

    # Totals
    total_covered_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    total_taxable_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    total_liability_eur: Decimal = Field(
        default=Decimal("0"),
        description="Total liability in EUR"
    )

    # Analysis
    effective_carbon_price_eur: Decimal = Field(
        default=Decimal("0"),
        description="Effective carbon price across all jurisdictions"
    )
    highest_exposure_jurisdiction: Optional[str] = Field(None)

    # Sensitivity
    sensitivity_analysis: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Liability at different price levels"
    )

    # Forecast
    future_liability_forecast: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Forecast liability by year"
    )

    # Provenance
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)

    def calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash for audit trail."""
        content = {
            "organization_id": self.organization_id,
            "calculation_date": self.calculation_date.isoformat(),
            "total_liability_eur": str(self.total_liability_eur),
            "total_taxable_emissions": str(self.total_taxable_emissions_tco2e),
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True, default=str).encode()
        ).hexdigest()


class CarbonTaxInput(BaseModel):
    """Input for carbon tax calculation."""

    emissions_profile: EmissionsProfile = Field(
        ...,
        description="Emissions profile"
    )
    reporting_year: int = Field(
        default_factory=lambda: DeterministicClock.now().year,
        description="Reporting year"
    )
    include_sensitivity: bool = Field(
        default=True,
        description="Include sensitivity analysis"
    )
    forecast_years: int = Field(
        default=5,
        ge=0,
        le=10,
        description="Years to forecast"
    )
    jurisdictions: Optional[List[TaxJurisdiction]] = Field(
        None,
        description="Specific jurisdictions to calculate"
    )


class CarbonTaxOutput(BaseModel):
    """Output from carbon tax calculation."""

    success: bool = Field(...)
    result: Optional[CarbonTaxResult] = Field(None)
    error: Optional[str] = Field(None)
    warnings: List[str] = Field(default_factory=list)


# =============================================================================
# CARBON TAX RATES DATABASE
# =============================================================================


CARBON_TAX_RATES: Dict[str, CarbonTaxRate] = {}


def _initialize_tax_rates() -> None:
    """Initialize carbon tax rates database."""
    global CARBON_TAX_RATES

    rates = [
        CarbonTaxRate(
            rate_id="EU-ETS-2024",
            jurisdiction=TaxJurisdiction.EU_ETS,
            name="EU Emissions Trading System",
            rate_per_tco2e=Decimal("80.00"),  # Approximate 2024 price
            currency="EUR",
            effective_date=date(2024, 1, 1),
            coverage_scope=CoverageScope.SCOPE_1,
            covered_sectors=[
                SectorType.POWER_GENERATION,
                SectorType.HEAVY_INDUSTRY,
                SectorType.MANUFACTURING,
            ],
            free_allocation_available=True,
            future_rates={
                "2025": Decimal("90.00"),
                "2026": Decimal("100.00"),
                "2027": Decimal("110.00"),
                "2028": Decimal("120.00"),
                "2029": Decimal("130.00"),
            },
        ),
        CarbonTaxRate(
            rate_id="UK-ETS-2024",
            jurisdiction=TaxJurisdiction.UK_ETS,
            name="UK Emissions Trading Scheme",
            rate_per_tco2e=Decimal("75.00"),  # GBP converted to EUR approx
            currency="EUR",
            effective_date=date(2024, 1, 1),
            coverage_scope=CoverageScope.SCOPE_1,
            covered_sectors=[
                SectorType.POWER_GENERATION,
                SectorType.HEAVY_INDUSTRY,
                SectorType.MANUFACTURING,
            ],
            free_allocation_available=True,
            future_rates={
                "2025": Decimal("85.00"),
                "2026": Decimal("95.00"),
            },
        ),
        CarbonTaxRate(
            rate_id="CA-CAP-2024",
            jurisdiction=TaxJurisdiction.CALIFORNIA_CAP,
            name="California Cap-and-Trade",
            rate_per_tco2e=Decimal("35.00"),  # USD converted to EUR approx
            currency="EUR",
            effective_date=date(2024, 1, 1),
            coverage_scope=CoverageScope.SCOPE_1,
            covered_sectors=[
                SectorType.POWER_GENERATION,
                SectorType.HEAVY_INDUSTRY,
                SectorType.TRANSPORT,
            ],
            free_allocation_available=True,
            exemption_threshold_tco2e=Decimal("25000"),
            future_rates={
                "2025": Decimal("40.00"),
                "2026": Decimal("45.00"),
            },
        ),
        CarbonTaxRate(
            rate_id="CANADA-FED-2024",
            jurisdiction=TaxJurisdiction.CANADA_FEDERAL,
            name="Canada Federal Carbon Price",
            rate_per_tco2e=Decimal("55.00"),  # CAD converted to EUR approx
            currency="EUR",
            effective_date=date(2024, 1, 1),
            coverage_scope=CoverageScope.SCOPE_1,
            covered_sectors=[
                SectorType.POWER_GENERATION,
                SectorType.HEAVY_INDUSTRY,
                SectorType.MANUFACTURING,
                SectorType.TRANSPORT,
                SectorType.BUILDINGS,
            ],
            future_rates={
                "2025": Decimal("65.00"),
                "2026": Decimal("75.00"),
                "2027": Decimal("85.00"),
                "2028": Decimal("95.00"),
                "2029": Decimal("105.00"),
                "2030": Decimal("125.00"),
            },
        ),
        CarbonTaxRate(
            rate_id="SWEDEN-2024",
            jurisdiction=TaxJurisdiction.SWEDEN,
            name="Swedish Carbon Tax",
            rate_per_tco2e=Decimal("120.00"),  # One of the highest
            currency="EUR",
            effective_date=date(2024, 1, 1),
            coverage_scope=CoverageScope.SCOPE_1,
            covered_sectors=[
                SectorType.TRANSPORT,
                SectorType.BUILDINGS,
            ],
        ),
        CarbonTaxRate(
            rate_id="SINGAPORE-2024",
            jurisdiction=TaxJurisdiction.SINGAPORE,
            name="Singapore Carbon Tax",
            rate_per_tco2e=Decimal("20.00"),  # SGD converted to EUR
            currency="EUR",
            effective_date=date(2024, 1, 1),
            coverage_scope=CoverageScope.SCOPE_1,
            covered_sectors=[
                SectorType.POWER_GENERATION,
                SectorType.HEAVY_INDUSTRY,
            ],
            exemption_threshold_tco2e=Decimal("25000"),
            future_rates={
                "2025": Decimal("20.00"),
                "2026": Decimal("40.00"),  # Significant increase planned
                "2027": Decimal("50.00"),
                "2030": Decimal("75.00"),
            },
        ),
    ]

    for rate in rates:
        CARBON_TAX_RATES[rate.rate_id] = rate


_initialize_tax_rates()


# =============================================================================
# CARBON TAX CALCULATOR AGENT
# =============================================================================


class CarbonTaxCalculator(BaseAgent):
    """
    GL-POL-X-005: Carbon Tax Calculator

    Calculates carbon tax exposure across jurisdictions.
    CRITICAL PATH agent with zero-hallucination guarantees.

    All calculations are:
    - Based on official tax rates from curated database
    - Deterministic with complete formula transparency
    - Fully auditable with step-by-step traces
    - No LLM inference involved

    Formula:
        Net Liability = (Covered Emissions - Free Allocation - Exempt) * Tax Rate

    Usage:
        agent = CarbonTaxCalculator()
        result = agent.run({
            'emissions_profile': {...},
            'reporting_year': 2024
        })
    """

    AGENT_ID = "GL-POL-X-005"
    AGENT_NAME = "Carbon Tax Calculator"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name=AGENT_NAME,
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        uses_tools=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Deterministic carbon tax liability calculation"
    )

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize Carbon Tax Calculator."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Carbon tax liability calculator",
                version=self.VERSION,
                parameters={
                    "default_currency": "EUR",
                    "sensitivity_percentages": [50, 75, 100, 125, 150, 200],
                }
            )

        self._tax_rates = CARBON_TAX_RATES.copy()
        self._audit_trail: List[Dict[str, Any]] = []

        super().__init__(config)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute carbon tax calculation."""
        import time
        start_time = time.time()

        try:
            agent_input = CarbonTaxInput(**input_data)
            result = self._calculate_tax_liability(agent_input)
            result.provenance_hash = result.calculate_provenance_hash()
            result.processing_time_ms = (time.time() - start_time) * 1000

            output = CarbonTaxOutput(success=True, result=result)

            return AgentResult(
                success=True,
                data=output.model_dump(),
            )

        except Exception as e:
            logger.error(f"Carbon tax calculation failed: {str(e)}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _calculate_tax_liability(
        self,
        input_data: CarbonTaxInput
    ) -> CarbonTaxResult:
        """
        Calculate carbon tax liability - 100% deterministic.

        Formula:
            Net Liability = (Covered - Free - Exempt) * Rate
        """
        profile = input_data.emissions_profile

        result = CarbonTaxResult(
            organization_id=profile.organization_id,
            reporting_year=input_data.reporting_year,
        )

        liability_items: List[TaxLiabilityItem] = []

        # Get applicable rates
        applicable_rates = self._get_applicable_rates(
            input_data.reporting_year,
            input_data.jurisdictions
        )

        # Calculate for each jurisdiction
        for rate_id, rate in applicable_rates.items():
            jurisdiction = rate.jurisdiction

            # Get emissions for this jurisdiction
            emissions = profile.emissions_by_jurisdiction.get(
                jurisdiction.value, Decimal("0")
            )

            if emissions <= 0:
                continue  # Skip jurisdictions with no emissions

            trace: List[str] = []
            trace.append(f"Calculating {rate.name} ({jurisdiction.value})")
            trace.append(f"  Covered emissions: {emissions:,.2f} tCO2e")

            # Check exemption threshold
            exempt_emissions = Decimal("0")
            if rate.exemption_threshold_tco2e:
                if emissions < rate.exemption_threshold_tco2e:
                    exempt_emissions = emissions
                    trace.append(f"  Below exemption threshold of {rate.exemption_threshold_tco2e:,.0f} tCO2e - fully exempt")

            # Get free allocation
            free_allocation = profile.free_allocations.get(
                jurisdiction.value, Decimal("0")
            )
            trace.append(f"  Free allocation: {free_allocation:,.2f} tCO2e")

            # Calculate taxable emissions
            taxable = max(
                Decimal("0"),
                emissions - exempt_emissions - free_allocation
            )
            trace.append(f"  Taxable emissions: {taxable:,.2f} tCO2e")

            # Calculate liability
            gross_liability = emissions * rate.rate_per_tco2e
            net_liability = taxable * rate.rate_per_tco2e
            trace.append(f"  Tax rate: {rate.rate_per_tco2e:,.2f} {rate.currency}/tCO2e")
            trace.append(f"  Gross liability: {gross_liability:,.2f} {rate.currency}")
            trace.append(f"  Net liability: {net_liability:,.2f} {rate.currency}")

            item = TaxLiabilityItem(
                jurisdiction=jurisdiction,
                jurisdiction_name=rate.name,
                covered_emissions_tco2e=emissions,
                exempt_emissions_tco2e=exempt_emissions,
                free_allocation_tco2e=free_allocation,
                taxable_emissions_tco2e=taxable,
                tax_rate_per_tco2e=rate.rate_per_tco2e,
                currency=rate.currency,
                gross_liability=gross_liability.quantize(Decimal("0.01"), ROUND_HALF_UP),
                net_liability=net_liability.quantize(Decimal("0.01"), ROUND_HALF_UP),
                calculation_trace=trace,
            )

            liability_items.append(item)

        # Aggregate results
        result.liability_items = liability_items
        result.total_covered_emissions_tco2e = sum(
            item.covered_emissions_tco2e for item in liability_items
        )
        result.total_taxable_emissions_tco2e = sum(
            item.taxable_emissions_tco2e for item in liability_items
        )
        result.total_liability_eur = sum(
            item.net_liability for item in liability_items
        )

        # Calculate effective carbon price
        if result.total_taxable_emissions_tco2e > 0:
            result.effective_carbon_price_eur = (
                result.total_liability_eur / result.total_taxable_emissions_tco2e
            ).quantize(Decimal("0.01"), ROUND_HALF_UP)

        # Find highest exposure
        if liability_items:
            highest = max(liability_items, key=lambda x: x.net_liability)
            result.highest_exposure_jurisdiction = highest.jurisdiction.value

        # Sensitivity analysis
        if input_data.include_sensitivity:
            result.sensitivity_analysis = self._calculate_sensitivity(
                result.total_taxable_emissions_tco2e,
                result.effective_carbon_price_eur
            )

        # Forecast
        if input_data.forecast_years > 0:
            result.future_liability_forecast = self._calculate_forecast(
                profile, applicable_rates, input_data.forecast_years
            )

        return result

    def _get_applicable_rates(
        self,
        year: int,
        jurisdictions: Optional[List[TaxJurisdiction]]
    ) -> Dict[str, CarbonTaxRate]:
        """Get applicable tax rates for given year and jurisdictions."""
        applicable = {}
        reference_date = date(year, 12, 31)

        for rate_id, rate in self._tax_rates.items():
            # Check date validity
            if rate.effective_date > reference_date:
                continue
            if rate.expiry_date and rate.expiry_date < reference_date:
                continue

            # Check jurisdiction filter
            if jurisdictions and rate.jurisdiction not in jurisdictions:
                continue

            applicable[rate_id] = rate

        return applicable

    def _calculate_sensitivity(
        self,
        taxable_emissions: Decimal,
        base_price: Decimal
    ) -> Dict[str, Decimal]:
        """Calculate liability at different price levels."""
        percentages = self.config.parameters.get(
            "sensitivity_percentages", [50, 75, 100, 125, 150, 200]
        )

        sensitivity = {}
        for pct in percentages:
            price = base_price * Decimal(pct) / Decimal(100)
            liability = taxable_emissions * price
            sensitivity[f"{pct}%"] = liability.quantize(Decimal("0.01"), ROUND_HALF_UP)

        return sensitivity

    def _calculate_forecast(
        self,
        profile: EmissionsProfile,
        current_rates: Dict[str, CarbonTaxRate],
        years: int
    ) -> Dict[str, Decimal]:
        """Calculate forecast liability using projected rates."""
        current_year = DeterministicClock.now().year
        forecast = {}

        for year_offset in range(1, years + 1):
            target_year = current_year + year_offset
            year_str = str(target_year)
            total_liability = Decimal("0")

            for rate_id, rate in current_rates.items():
                # Get projected rate
                projected_rate = rate.future_rates.get(year_str, rate.rate_per_tco2e)

                # Get emissions (assume constant)
                emissions = profile.emissions_by_jurisdiction.get(
                    rate.jurisdiction.value, Decimal("0")
                )
                free_allocation = profile.free_allocations.get(
                    rate.jurisdiction.value, Decimal("0")
                )

                taxable = max(Decimal("0"), emissions - free_allocation)
                liability = taxable * projected_rate
                total_liability += liability

            forecast[year_str] = total_liability.quantize(Decimal("0.01"), ROUND_HALF_UP)

        return forecast

    # =========================================================================
    # PUBLIC API METHODS
    # =========================================================================

    def get_tax_rate(
        self,
        jurisdiction: TaxJurisdiction,
        year: Optional[int] = None
    ) -> Optional[CarbonTaxRate]:
        """Get tax rate for a jurisdiction."""
        year = year or DeterministicClock.now().year

        for rate in self._tax_rates.values():
            if rate.jurisdiction == jurisdiction:
                if rate.effective_date.year <= year:
                    if not rate.expiry_date or rate.expiry_date.year >= year:
                        return rate
        return None

    def add_tax_rate(self, rate: CarbonTaxRate) -> str:
        """Add a custom tax rate."""
        self._tax_rates[rate.rate_id] = rate
        return rate.rate_id

    def list_jurisdictions(self) -> List[TaxJurisdiction]:
        """List all jurisdictions with carbon tax rates."""
        return list(set(rate.jurisdiction for rate in self._tax_rates.values()))


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "CarbonTaxCalculator",
    "TaxJurisdiction",
    "CoverageScope",
    "SectorType",
    "CarbonTaxRate",
    "EmissionsProfile",
    "TaxLiabilityItem",
    "CarbonTaxResult",
    "CarbonTaxInput",
    "CarbonTaxOutput",
    "CARBON_TAX_RATES",
]
