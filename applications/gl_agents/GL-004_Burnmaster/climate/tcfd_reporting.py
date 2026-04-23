"""
GL-004 BURNMASTER - TCFD Reporting

Task Force on Climate-related Financial Disclosures support for
combustion operations emissions and climate risk metrics.

References:
    - TCFD Final Report (2017)
    - TCFD Technical Supplement (2017)
    - TCFD Guidance on Metrics, Targets, and Transition Plans (2021)

Author: GreenLang Combustion Systems Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ClimateRiskCategory(str, Enum):
    """TCFD climate risk categories."""
    TRANSITION_POLICY = "transition_policy"
    TRANSITION_LEGAL = "transition_legal"
    TRANSITION_TECHNOLOGY = "transition_technology"
    TRANSITION_MARKET = "transition_market"
    TRANSITION_REPUTATION = "transition_reputation"
    PHYSICAL_ACUTE = "physical_acute"
    PHYSICAL_CHRONIC = "physical_chronic"


class ClimateOpportunityCategory(str, Enum):
    """TCFD climate opportunity categories."""
    RESOURCE_EFFICIENCY = "resource_efficiency"
    ENERGY_SOURCE = "energy_source"
    PRODUCTS_SERVICES = "products_services"
    MARKETS = "markets"
    RESILIENCE = "resilience"


class ScenarioType(str, Enum):
    """Climate scenario types for analysis."""
    IEA_NZE = "iea_nze"              # IEA Net Zero Emissions 2050
    IEA_SDS = "iea_sds"              # IEA Sustainable Development
    IEA_STEPS = "iea_steps"          # IEA Stated Policies
    NGFS_NET_ZERO = "ngfs_net_zero"  # NGFS Net Zero 2050
    NGFS_BELOW_2C = "ngfs_below_2c"  # NGFS Below 2C
    NGFS_CURRENT = "ngfs_current"    # NGFS Current Policies
    CUSTOM = "custom"


class EmissionScope(str, Enum):
    """GHG emission scopes."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


class TCFDMetrics(BaseModel):
    """TCFD-aligned climate metrics for disclosure."""
    metrics_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Entity identification
    entity_name: str = Field(..., description="Reporting entity name")
    reporting_year: int = Field(..., description="Reporting year")

    # GHG Emissions (Cross-Industry Metric)
    scope1_emissions_tco2e: Decimal = Field(..., ge=0, description="Scope 1 emissions")
    scope2_emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    scope3_emissions_tco2e: Optional[Decimal] = Field(None, description="Scope 3 if material")
    total_emissions_tco2e: Decimal = Field(..., ge=0)

    # Emissions breakdown
    emissions_by_business_unit: Dict[str, Decimal] = Field(default_factory=dict)
    emissions_by_geography: Dict[str, Decimal] = Field(default_factory=dict)

    # Emissions intensity (Cross-Industry Metric)
    revenue_based_intensity: Optional[Decimal] = Field(
        None, description="tCO2e per $M revenue"
    )
    production_based_intensity: Optional[Decimal] = Field(
        None, description="tCO2e per unit output"
    )
    energy_based_intensity: Optional[Decimal] = Field(
        None, description="tCO2e per MWh"
    )
    intensity_metric_unit: str = Field(default="tCO2e/unit")

    # Year-over-year change
    emissions_change_percent: Optional[float] = Field(
        None, description="Change from prior year"
    )
    base_year: Optional[int] = Field(None, description="Base year for tracking")
    change_from_base_year_percent: Optional[float] = Field(None)

    # Targets
    target_year: Optional[int] = Field(None, description="Target achievement year")
    target_reduction_percent: Optional[float] = Field(None, description="Target reduction %")
    progress_toward_target_percent: Optional[float] = Field(None)
    net_zero_commitment_year: Optional[int] = Field(None)
    science_based_target: bool = Field(default=False, description="SBTi validated")

    # Energy metrics
    total_energy_consumption_mwh: Optional[Decimal] = Field(None)
    renewable_energy_percent: Optional[float] = Field(None, ge=0, le=100)

    # Carbon pricing exposure
    internal_carbon_price: Optional[Decimal] = Field(
        None, description="Internal carbon price ($/tCO2e)"
    )
    carbon_cost_exposure: Optional[Decimal] = Field(
        None, description="Estimated carbon cost exposure"
    )
    percent_emissions_under_pricing: Optional[float] = Field(None, ge=0, le=100)

    # Physical risk metrics
    assets_in_high_risk_locations_percent: Optional[float] = Field(None, ge=0, le=100)
    capex_in_climate_vulnerable_areas: Optional[Decimal] = Field(None)

    # Transition risk metrics
    revenue_from_fossil_fuels_percent: Optional[float] = Field(None, ge=0, le=100)
    capex_aligned_with_transition: Optional[Decimal] = Field(None)

    provenance_hash: str = Field(default="")


class ClimateRisk(BaseModel):
    """Individual climate risk assessment."""
    risk_id: str = Field(default_factory=lambda: str(uuid4()))
    category: ClimateRiskCategory
    description: str
    time_horizon: str = Field(default="medium_term")  # short, medium, long
    likelihood: str = Field(default="likely")  # unlikely, possible, likely, very_likely
    magnitude: str = Field(default="medium")  # low, medium, high
    financial_impact_range: Optional[str] = Field(None)
    mitigation_actions: List[str] = Field(default_factory=list)


class ClimateOpportunity(BaseModel):
    """Individual climate opportunity assessment."""
    opportunity_id: str = Field(default_factory=lambda: str(uuid4()))
    category: ClimateOpportunityCategory
    description: str
    time_horizon: str = Field(default="medium_term")
    likelihood: str = Field(default="likely")
    financial_impact_range: Optional[str] = Field(None)
    actions_to_capture: List[str] = Field(default_factory=list)


class TCFDReporter:
    """
    TCFD-aligned climate disclosure reporter.

    Generates metrics and assessments aligned with TCFD recommendations
    for climate-related financial disclosures.

    Example:
        >>> reporter = TCFDReporter()
        >>> metrics = reporter.generate_metrics(
        ...     entity_name="Acme Energy",
        ...     reporting_year=2024,
        ...     scope1_tco2e=Decimal("50000"),
        ...     revenue_million=Decimal("500")
        ... )
    """

    def __init__(self, precision: int = 2):
        """Initialize TCFD reporter."""
        self.precision = precision
        self._quantize_str = "0." + "0" * precision
        self._historical_emissions: Dict[str, Dict[int, Decimal]] = {}
        logger.info("TCFDReporter initialized")

    def _quantize(self, value: Decimal) -> Decimal:
        """Apply precision rounding."""
        return value.quantize(Decimal(self._quantize_str), rounding=ROUND_HALF_UP)

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash."""
        data_str = str(sorted(data.items()))
        return hashlib.sha256(data_str.encode()).hexdigest()

    def set_historical_emissions(
        self,
        entity_name: str,
        year: int,
        emissions_tco2e: Decimal
    ) -> None:
        """Set historical emissions for tracking."""
        if entity_name not in self._historical_emissions:
            self._historical_emissions[entity_name] = {}
        self._historical_emissions[entity_name][year] = emissions_tco2e
        logger.info(f"Historical emissions set: {entity_name} {year} = {emissions_tco2e}")

    def set_base_year(
        self,
        entity_name: str,
        base_year: int,
        base_emissions_tco2e: Decimal
    ) -> None:
        """Set base year emissions for reduction tracking."""
        self.set_historical_emissions(entity_name, base_year, base_emissions_tco2e)

    def calculate_intensity(
        self,
        emissions_tco2e: Decimal,
        denominator: Decimal,
    ) -> Decimal:
        """Calculate emissions intensity."""
        if denominator == 0:
            return Decimal("0")
        return self._quantize(emissions_tco2e / denominator)

    def calculate_year_over_year_change(
        self,
        entity_name: str,
        current_year: int,
        current_emissions: Decimal
    ) -> Optional[float]:
        """Calculate year-over-year emissions change."""
        prior_year = current_year - 1
        historical = self._historical_emissions.get(entity_name, {})

        if prior_year in historical and historical[prior_year] > 0:
            change = (current_emissions - historical[prior_year]) / historical[prior_year]
            return round(float(change) * 100, 1)
        return None

    def calculate_progress_to_target(
        self,
        base_emissions: Decimal,
        current_emissions: Decimal,
        target_reduction_percent: float
    ) -> float:
        """Calculate progress toward emissions reduction target."""
        if base_emissions == 0:
            return 0.0

        target_emissions = base_emissions * Decimal(str(1 - target_reduction_percent / 100))
        required_reduction = base_emissions - target_emissions
        actual_reduction = base_emissions - current_emissions

        if required_reduction == 0:
            return 100.0 if actual_reduction >= 0 else 0.0

        progress = float(actual_reduction / required_reduction) * 100
        return round(min(100.0, max(0.0, progress)), 1)

    def generate_metrics(
        self,
        entity_name: str,
        reporting_year: int,
        scope1_tco2e: Decimal,
        scope2_tco2e: Decimal = Decimal("0"),
        scope3_tco2e: Optional[Decimal] = None,
        revenue_million: Optional[Decimal] = None,
        production_units: Optional[Decimal] = None,
        production_unit_name: str = "unit",
        energy_mwh: Optional[Decimal] = None,
        renewable_energy_mwh: Optional[Decimal] = None,
        base_year: Optional[int] = None,
        target_year: Optional[int] = None,
        target_reduction_percent: Optional[float] = None,
        net_zero_year: Optional[int] = None,
        science_based_target: bool = False,
        internal_carbon_price: Optional[Decimal] = None,
        emissions_under_pricing_tco2e: Optional[Decimal] = None,
    ) -> TCFDMetrics:
        """
        Generate TCFD-aligned climate metrics.

        Args:
            entity_name: Reporting entity name
            reporting_year: Year being reported
            scope1_tco2e: Scope 1 emissions (tonnes CO2e)
            scope2_tco2e: Scope 2 emissions (tonnes CO2e)
            scope3_tco2e: Scope 3 emissions if material
            revenue_million: Revenue in millions for intensity calculation
            production_units: Production quantity for intensity
            production_unit_name: Name of production unit
            energy_mwh: Total energy consumption in MWh
            renewable_energy_mwh: Renewable energy in MWh
            base_year: Base year for reduction tracking
            target_year: Target achievement year
            target_reduction_percent: Reduction target percentage
            net_zero_year: Net zero commitment year
            science_based_target: Whether SBTi validated
            internal_carbon_price: Internal carbon price ($/tCO2e)
            emissions_under_pricing_tco2e: Emissions subject to carbon pricing

        Returns:
            TCFDMetrics with disclosure-ready data
        """
        # Calculate totals
        total_emissions = scope1_tco2e + scope2_tco2e
        if scope3_tco2e:
            total_emissions += scope3_tco2e

        # Calculate intensities
        revenue_intensity = None
        if revenue_million and revenue_million > 0:
            revenue_intensity = self.calculate_intensity(total_emissions, revenue_million)

        production_intensity = None
        if production_units and production_units > 0:
            production_intensity = self.calculate_intensity(total_emissions, production_units)

        energy_intensity = None
        if energy_mwh and energy_mwh > 0:
            energy_intensity = self.calculate_intensity(total_emissions, energy_mwh)

        # Calculate renewable percentage
        renewable_percent = None
        if energy_mwh and renewable_energy_mwh:
            if energy_mwh > 0:
                renewable_percent = round(
                    float(renewable_energy_mwh / energy_mwh) * 100, 1
                )

        # Calculate year-over-year change
        yoy_change = self.calculate_year_over_year_change(
            entity_name, reporting_year, total_emissions
        )

        # Calculate change from base year
        base_year_change = None
        if base_year and entity_name in self._historical_emissions:
            base_emissions = self._historical_emissions[entity_name].get(base_year)
            if base_emissions and base_emissions > 0:
                change = (total_emissions - base_emissions) / base_emissions
                base_year_change = round(float(change) * 100, 1)

        # Calculate progress toward target
        progress = None
        if base_year and target_reduction_percent:
            base_emissions = self._historical_emissions.get(entity_name, {}).get(base_year)
            if base_emissions:
                progress = self.calculate_progress_to_target(
                    base_emissions, total_emissions, target_reduction_percent
                )

        # Calculate carbon cost exposure
        carbon_cost = None
        if internal_carbon_price and emissions_under_pricing_tco2e:
            carbon_cost = self._quantize(
                internal_carbon_price * emissions_under_pricing_tco2e
            )

        # Calculate percent under pricing
        pricing_percent = None
        if emissions_under_pricing_tco2e and total_emissions > 0:
            pricing_percent = round(
                float(emissions_under_pricing_tco2e / total_emissions) * 100, 1
            )

        # Store for future tracking
        self.set_historical_emissions(entity_name, reporting_year, total_emissions)

        # Compute provenance
        provenance = self._compute_hash({
            "entity_name": entity_name,
            "reporting_year": reporting_year,
            "total_emissions": str(total_emissions),
        })

        return TCFDMetrics(
            entity_name=entity_name,
            reporting_year=reporting_year,
            scope1_emissions_tco2e=scope1_tco2e,
            scope2_emissions_tco2e=scope2_tco2e,
            scope3_emissions_tco2e=scope3_tco2e,
            total_emissions_tco2e=total_emissions,
            revenue_based_intensity=revenue_intensity,
            production_based_intensity=production_intensity,
            energy_based_intensity=energy_intensity,
            intensity_metric_unit=f"tCO2e/{production_unit_name}" if production_units else "tCO2e/unit",
            emissions_change_percent=yoy_change,
            base_year=base_year,
            change_from_base_year_percent=base_year_change,
            target_year=target_year,
            target_reduction_percent=target_reduction_percent,
            progress_toward_target_percent=progress,
            net_zero_commitment_year=net_zero_year,
            science_based_target=science_based_target,
            total_energy_consumption_mwh=energy_mwh,
            renewable_energy_percent=renewable_percent,
            internal_carbon_price=internal_carbon_price,
            carbon_cost_exposure=carbon_cost,
            percent_emissions_under_pricing=pricing_percent,
            provenance_hash=provenance,
        )

    def assess_transition_risks(
        self,
        scope1_tco2e: Decimal,
        revenue_million: Decimal,
        carbon_price_current: Decimal,
        carbon_price_projected: Decimal,
    ) -> List[ClimateRisk]:
        """
        Assess transition risks for combustion operations.

        Returns list of identified climate risks.
        """
        risks = []

        # Carbon pricing risk
        current_cost = scope1_tco2e * carbon_price_current
        projected_cost = scope1_tco2e * carbon_price_projected
        cost_increase = projected_cost - current_cost

        if cost_increase > revenue_million * Decimal("0.01"):  # >1% of revenue
            risks.append(ClimateRisk(
                category=ClimateRiskCategory.TRANSITION_POLICY,
                description="Increasing carbon pricing may impact operating costs",
                time_horizon="medium_term",
                likelihood="very_likely",
                magnitude="high" if cost_increase > revenue_million * Decimal("0.05") else "medium",
                financial_impact_range=f"${cost_increase:,.0f} additional annual cost",
                mitigation_actions=[
                    "Invest in energy efficiency improvements",
                    "Evaluate fuel switching opportunities",
                    "Develop internal carbon pricing mechanisms",
                ]
            ))

        # Technology risk
        risks.append(ClimateRisk(
            category=ClimateRiskCategory.TRANSITION_TECHNOLOGY,
            description="Emergence of low-carbon combustion technologies may require capital investment",
            time_horizon="medium_term",
            likelihood="likely",
            magnitude="medium",
            mitigation_actions=[
                "Monitor hydrogen-ready burner technology",
                "Assess electrification opportunities",
                "Evaluate carbon capture feasibility",
            ]
        ))

        return risks

    def assess_opportunities(
        self,
        current_efficiency: float,
        target_efficiency: float,
        fuel_cost_per_year: Decimal,
    ) -> List[ClimateOpportunity]:
        """
        Assess climate-related opportunities.

        Returns list of identified opportunities.
        """
        opportunities = []

        # Efficiency opportunity
        efficiency_gap = target_efficiency - current_efficiency
        if efficiency_gap > 0:
            potential_savings = fuel_cost_per_year * Decimal(str(efficiency_gap / 100))
            opportunities.append(ClimateOpportunity(
                category=ClimateOpportunityCategory.RESOURCE_EFFICIENCY,
                description="Combustion efficiency improvements can reduce fuel costs and emissions",
                time_horizon="short_term",
                likelihood="very_likely",
                financial_impact_range=f"${potential_savings:,.0f} annual savings potential",
                actions_to_capture=[
                    "Implement AI-based combustion optimization",
                    "Upgrade to high-efficiency burners",
                    "Deploy advanced process controls",
                ]
            ))

        return opportunities
