# -*- coding: utf-8 -*-
"""
GreenLang Buildings Decarbonization - Additional Agents
=========================================================

This module contains the remaining decarbonization agents for buildings:
- GL-DECARB-BLD-003: Building Electrification
- GL-DECARB-BLD-004: Embodied Carbon Reduction
- GL-DECARB-BLD-005: Net Zero Building Planner
- GL-DECARB-BLD-006: Passive Design Optimizer
- GL-DECARB-BLD-007: On-site Renewables Planner
- GL-DECARB-BLD-008: District Energy Planner
- GL-DECARB-BLD-009: Building Automation Optimizer
- GL-DECARB-BLD-010: Tenant Engagement Agent
- GL-DECARB-BLD-011: Portfolio Decarbonization
- GL-DECARB-BLD-012: Green Lease Agent

Author: GreenLang Framework Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field

from greenlang.agents.decarbonization.buildings.base import (
    BuildingDecarbonizationBaseAgent,
    DecarbonizationInput,
    DecarbonizationOutput,
    DecarbonizationMeasure,
    DecarbonizationPathway,
    TechnologySpec,
    TechnologyCategory,
    RecommendationPriority,
    ImplementationPhase,
    RiskLevel,
    BuildingBaseline,
    DecarbonizationTarget,
    SOLAR_CAPACITY_FACTOR,
)

logger = logging.getLogger(__name__)


# =============================================================================
# GL-DECARB-BLD-003: Building Electrification Agent
# =============================================================================

class BuildingElectrificationInput(DecarbonizationInput):
    """Input for building electrification analysis."""
    current_gas_appliances: List[str] = Field(default_factory=list)
    gas_consumption_therms: Optional[Decimal] = Field(None, ge=0)
    electrical_capacity_kw: Optional[Decimal] = Field(None, ge=0)
    panel_upgrade_needed: Optional[bool] = None


class BuildingElectrificationOutput(DecarbonizationOutput):
    """Output for building electrification analysis."""
    electrification_measures: List[DecarbonizationMeasure] = Field(default_factory=list)
    gas_elimination_savings_kgco2e: Decimal = Field(default=Decimal("0"))
    panel_upgrade_required: bool = Field(default=False)
    estimated_new_load_kw: Decimal = Field(default=Decimal("0"))


class BuildingElectrificationAgent(BuildingDecarbonizationBaseAgent[BuildingElectrificationInput, BuildingElectrificationOutput]):
    """GL-DECARB-BLD-003: Building Electrification Agent."""

    AGENT_ID = "GL-DECARB-BLD-003"
    AGENT_VERSION = "1.0.0"
    TECHNOLOGY_FOCUS = TechnologyCategory.ELECTRIFICATION

    def _load_technology_database(self) -> None:
        self._technology_database["electric_cooking"] = TechnologySpec(
            technology_id="electric_cooking",
            category=TechnologyCategory.ELECTRIFICATION,
            name="Induction Cooking",
            description="Replace gas cooking with induction",
            lifespan_years=15
        )
        self._technology_database["electric_water_heater"] = TechnologySpec(
            technology_id="electric_water_heater",
            category=TechnologyCategory.ELECTRIFICATION,
            name="Heat Pump Water Heater",
            description="Replace gas water heater with HPWH",
            lifespan_years=15
        )

    def analyze(self, input_data: BuildingElectrificationInput) -> BuildingElectrificationOutput:
        baseline = input_data.building_baseline
        target = input_data.target
        measures: List[DecarbonizationMeasure] = []

        gas_ef = Decimal("5.302")  # kgCO2e per therm
        grid_ef = Decimal("0.379")

        total_gas_emissions = Decimal("0")
        if input_data.gas_consumption_therms:
            total_gas_emissions = input_data.gas_consumption_therms * gas_ef

        # Water heater electrification
        if "water_heater" in input_data.current_gas_appliances or not input_data.current_gas_appliances:
            wh_gas = (input_data.gas_consumption_therms or Decimal("0")) * Decimal("0.3")
            wh_electric_kwh = wh_gas * Decimal("29.3") / Decimal("3.5")
            emission_savings = wh_gas * gas_ef - wh_electric_kwh * grid_ef

            measures.append(self._create_measure(
                measure_id="ELEC-001",
                name="Heat Pump Water Heater",
                description="Replace gas water heater with heat pump",
                technology=self._technology_database["electric_water_heater"],
                capital_cost=Decimal("3500"),
                annual_savings=wh_gas * input_data.gas_cost_per_therm - wh_electric_kwh * input_data.electricity_cost_per_kwh,
                energy_savings_kwh=wh_gas * Decimal("29.3") - wh_electric_kwh,
                emission_reduction=emission_savings,
                priority=RecommendationPriority.HIGH,
                phase=ImplementationPhase.IMMEDIATE
            ))

        total_reduction = sum(m.annual_emission_reduction_kgco2e for m in measures)
        total_investment = sum(m.financial.capital_cost_usd for m in measures)

        target_emissions = baseline.current_emissions_kgco2e_per_year * (
            1 - (target.target_reduction_percent or Decimal("0")) / 100
        )

        return BuildingElectrificationOutput(
            analysis_id=self._generate_analysis_id(baseline.building_id),
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            building_id=baseline.building_id,
            baseline_emissions_kgco2e=baseline.current_emissions_kgco2e_per_year,
            target_emissions_kgco2e=target_emissions,
            target_year=target.target_year,
            total_reduction_kgco2e=self._round_emissions(total_reduction),
            total_investment_usd=self._round_financial(total_investment),
            electrification_measures=measures,
            gas_elimination_savings_kgco2e=self._round_emissions(total_reduction),
            is_valid=True
        )


# =============================================================================
# GL-DECARB-BLD-004: Embodied Carbon Reduction Agent
# =============================================================================

class EmbodiedCarbonInput(DecarbonizationInput):
    """Input for embodied carbon reduction analysis."""
    planned_renovation: bool = Field(default=False)
    renovation_scope: Optional[str] = None
    material_preferences: List[str] = Field(default_factory=list)


class EmbodiedCarbonOutput(DecarbonizationOutput):
    """Output for embodied carbon reduction analysis."""
    low_carbon_material_options: List[Dict[str, Any]] = Field(default_factory=list)
    embodied_carbon_reduction_kgco2e: Decimal = Field(default=Decimal("0"))
    material_substitution_savings: Dict[str, Decimal] = Field(default_factory=dict)


class EmbodiedCarbonReductionAgent(BuildingDecarbonizationBaseAgent[EmbodiedCarbonInput, EmbodiedCarbonOutput]):
    """GL-DECARB-BLD-004: Embodied Carbon Reduction Agent."""

    AGENT_ID = "GL-DECARB-BLD-004"
    AGENT_VERSION = "1.0.0"
    TECHNOLOGY_FOCUS = TechnologyCategory.MATERIALS

    def _load_technology_database(self) -> None:
        self._technology_database["low_carbon_concrete"] = TechnologySpec(
            technology_id="low_carbon_concrete",
            category=TechnologyCategory.MATERIALS,
            name="Low-Carbon Concrete",
            description="Concrete with supplementary cementitious materials",
            lifespan_years=60
        )

    def analyze(self, input_data: EmbodiedCarbonInput) -> EmbodiedCarbonOutput:
        baseline = input_data.building_baseline
        target = input_data.target

        target_emissions = baseline.current_emissions_kgco2e_per_year * (
            1 - (target.target_reduction_percent or Decimal("0")) / 100
        )

        return EmbodiedCarbonOutput(
            analysis_id=self._generate_analysis_id(baseline.building_id),
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            building_id=baseline.building_id,
            baseline_emissions_kgco2e=baseline.current_emissions_kgco2e_per_year,
            target_emissions_kgco2e=target_emissions,
            target_year=target.target_year,
            is_valid=True
        )


# =============================================================================
# GL-DECARB-BLD-005: Net Zero Building Planner
# =============================================================================

class NetZeroPlannerInput(DecarbonizationInput):
    """Input for net zero planning."""
    net_zero_target_year: int = Field(default=2050, ge=2025, le=2100)
    include_embodied_carbon: bool = Field(default=False)
    include_scope3: bool = Field(default=False)


class NetZeroPlannerOutput(DecarbonizationOutput):
    """Output for net zero planning."""
    net_zero_pathway: Optional[DecarbonizationPathway] = None
    annual_milestones: Dict[int, Decimal] = Field(default_factory=dict)
    residual_emissions_kgco2e: Decimal = Field(default=Decimal("0"))
    offset_requirement_kgco2e: Decimal = Field(default=Decimal("0"))


class NetZeroBuildingPlannerAgent(BuildingDecarbonizationBaseAgent[NetZeroPlannerInput, NetZeroPlannerOutput]):
    """GL-DECARB-BLD-005: Net Zero Building Planner."""

    AGENT_ID = "GL-DECARB-BLD-005"
    AGENT_VERSION = "1.0.0"

    def _load_technology_database(self) -> None:
        pass

    def analyze(self, input_data: NetZeroPlannerInput) -> NetZeroPlannerOutput:
        baseline = input_data.building_baseline
        target = input_data.target

        # Create annual milestones for linear pathway to net zero
        start_year = 2025
        end_year = input_data.net_zero_target_year
        years = end_year - start_year

        milestones = {}
        for i, year in enumerate(range(start_year, end_year + 1)):
            reduction = baseline.current_emissions_kgco2e_per_year * Decimal(str(i)) / Decimal(str(years))
            milestones[year] = self._round_emissions(
                baseline.current_emissions_kgco2e_per_year - reduction
            )

        return NetZeroPlannerOutput(
            analysis_id=self._generate_analysis_id(baseline.building_id),
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            building_id=baseline.building_id,
            baseline_emissions_kgco2e=baseline.current_emissions_kgco2e_per_year,
            target_emissions_kgco2e=Decimal("0"),
            target_year=end_year,
            annual_milestones=milestones,
            is_valid=True
        )


# =============================================================================
# GL-DECARB-BLD-006: Passive Design Optimizer
# =============================================================================

class PassiveDesignInput(DecarbonizationInput):
    """Input for passive design optimization."""
    orientation_degrees: Optional[Decimal] = Field(None, ge=0, le=360)
    window_to_wall_ratio: Optional[Decimal] = Field(None, ge=0, le=1)
    has_thermal_mass: bool = Field(default=False)
    has_natural_ventilation: bool = Field(default=False)


class PassiveDesignOutput(DecarbonizationOutput):
    """Output for passive design optimization."""
    passive_measures: List[DecarbonizationMeasure] = Field(default_factory=list)
    passive_savings_kgco2e: Decimal = Field(default=Decimal("0"))
    passive_house_feasible: bool = Field(default=False)


class PassiveDesignOptimizerAgent(BuildingDecarbonizationBaseAgent[PassiveDesignInput, PassiveDesignOutput]):
    """GL-DECARB-BLD-006: Passive Design Optimizer."""

    AGENT_ID = "GL-DECARB-BLD-006"
    AGENT_VERSION = "1.0.0"
    TECHNOLOGY_FOCUS = TechnologyCategory.ENVELOPE

    def _load_technology_database(self) -> None:
        self._technology_database["high_performance_envelope"] = TechnologySpec(
            technology_id="high_performance_envelope",
            category=TechnologyCategory.ENVELOPE,
            name="Passive House Envelope",
            description="Super-insulated airtight envelope",
            efficiency_improvement_percent=Decimal("70"),
            lifespan_years=50
        )

    def analyze(self, input_data: PassiveDesignInput) -> PassiveDesignOutput:
        baseline = input_data.building_baseline
        target = input_data.target

        target_emissions = baseline.current_emissions_kgco2e_per_year * (
            1 - (target.target_reduction_percent or Decimal("0")) / 100
        )

        return PassiveDesignOutput(
            analysis_id=self._generate_analysis_id(baseline.building_id),
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            building_id=baseline.building_id,
            baseline_emissions_kgco2e=baseline.current_emissions_kgco2e_per_year,
            target_emissions_kgco2e=target_emissions,
            target_year=target.target_year,
            is_valid=True
        )


# =============================================================================
# GL-DECARB-BLD-007: On-site Renewables Planner
# =============================================================================

class RenewablesPlannerInput(DecarbonizationInput):
    """Input for on-site renewables planning."""
    roof_area_sqm: Optional[Decimal] = Field(None, ge=0)
    roof_orientation: Optional[str] = None
    roof_tilt_degrees: Optional[Decimal] = Field(None, ge=0, le=90)
    solar_access_percent: Optional[Decimal] = Field(None, ge=0, le=100)
    include_battery_storage: bool = Field(default=False)
    electricity_rate_structure: Optional[str] = None


class RenewablesPlannerOutput(DecarbonizationOutput):
    """Output for on-site renewables planning."""
    solar_pv_capacity_kw: Optional[Decimal] = None
    annual_generation_kwh: Optional[Decimal] = None
    solar_savings_kgco2e: Decimal = Field(default=Decimal("0"))
    battery_capacity_kwh: Optional[Decimal] = None
    self_consumption_percent: Optional[Decimal] = None


class OnsiteRenewablesPlannerAgent(BuildingDecarbonizationBaseAgent[RenewablesPlannerInput, RenewablesPlannerOutput]):
    """GL-DECARB-BLD-007: On-site Renewables Planner."""

    AGENT_ID = "GL-DECARB-BLD-007"
    AGENT_VERSION = "1.0.0"
    TECHNOLOGY_FOCUS = TechnologyCategory.RENEWABLES

    def _load_technology_database(self) -> None:
        self._technology_database["solar_pv"] = TechnologySpec(
            technology_id="solar_pv",
            category=TechnologyCategory.RENEWABLES,
            name="Rooftop Solar PV",
            description="Rooftop photovoltaic system",
            lifespan_years=25
        )
        self._technology_database["battery_storage"] = TechnologySpec(
            technology_id="battery_storage",
            category=TechnologyCategory.STORAGE,
            name="Battery Storage",
            description="Lithium-ion battery storage",
            lifespan_years=15
        )

    def analyze(self, input_data: RenewablesPlannerInput) -> RenewablesPlannerOutput:
        baseline = input_data.building_baseline
        target = input_data.target

        grid_ef = Decimal("0.379")

        # Calculate solar potential
        solar_capacity = None
        annual_generation = None
        solar_savings = Decimal("0")

        if input_data.roof_area_sqm:
            # Assume 150 W/sqm and 70% usable roof
            usable_area = input_data.roof_area_sqm * Decimal("0.70")
            solar_capacity = usable_area * Decimal("0.150")  # kW

            capacity_factor = SOLAR_CAPACITY_FACTOR.get("good", Decimal("0.18"))
            annual_generation = solar_capacity * 8760 * capacity_factor

            solar_savings = annual_generation * grid_ef

        target_emissions = baseline.current_emissions_kgco2e_per_year * (
            1 - (target.target_reduction_percent or Decimal("0")) / 100
        )

        return RenewablesPlannerOutput(
            analysis_id=self._generate_analysis_id(baseline.building_id),
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            building_id=baseline.building_id,
            baseline_emissions_kgco2e=baseline.current_emissions_kgco2e_per_year,
            target_emissions_kgco2e=target_emissions,
            target_year=target.target_year,
            total_reduction_kgco2e=self._round_emissions(solar_savings),
            solar_pv_capacity_kw=solar_capacity,
            annual_generation_kwh=annual_generation,
            solar_savings_kgco2e=self._round_emissions(solar_savings),
            is_valid=True
        )


# =============================================================================
# GL-DECARB-BLD-008: District Energy Planner
# =============================================================================

class DistrictEnergyInput(DecarbonizationInput):
    """Input for district energy planning."""
    district_heating_available: bool = Field(default=False)
    district_cooling_available: bool = Field(default=False)
    distance_to_network_m: Optional[Decimal] = Field(None, ge=0)


class DistrictEnergyOutput(DecarbonizationOutput):
    """Output for district energy planning."""
    district_heating_feasible: bool = Field(default=False)
    district_cooling_feasible: bool = Field(default=False)
    connection_cost_usd: Optional[Decimal] = None
    district_savings_kgco2e: Decimal = Field(default=Decimal("0"))


class DistrictEnergyPlannerAgent(BuildingDecarbonizationBaseAgent[DistrictEnergyInput, DistrictEnergyOutput]):
    """GL-DECARB-BLD-008: District Energy Planner."""

    AGENT_ID = "GL-DECARB-BLD-008"
    AGENT_VERSION = "1.0.0"

    def _load_technology_database(self) -> None:
        pass

    def analyze(self, input_data: DistrictEnergyInput) -> DistrictEnergyOutput:
        baseline = input_data.building_baseline
        target = input_data.target

        target_emissions = baseline.current_emissions_kgco2e_per_year * (
            1 - (target.target_reduction_percent or Decimal("0")) / 100
        )

        return DistrictEnergyOutput(
            analysis_id=self._generate_analysis_id(baseline.building_id),
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            building_id=baseline.building_id,
            baseline_emissions_kgco2e=baseline.current_emissions_kgco2e_per_year,
            target_emissions_kgco2e=target_emissions,
            target_year=target.target_year,
            district_heating_feasible=input_data.district_heating_available,
            district_cooling_feasible=input_data.district_cooling_available,
            is_valid=True
        )


# =============================================================================
# GL-DECARB-BLD-009: Building Automation Optimizer
# =============================================================================

class BuildingAutomationInput(DecarbonizationInput):
    """Input for building automation optimization."""
    has_bms: bool = Field(default=False)
    bms_age_years: Optional[int] = Field(None, ge=0)
    num_zones: Optional[int] = Field(None, ge=0)
    has_fault_detection: bool = Field(default=False)


class BuildingAutomationOutput(DecarbonizationOutput):
    """Output for building automation optimization."""
    automation_measures: List[DecarbonizationMeasure] = Field(default_factory=list)
    automation_savings_kgco2e: Decimal = Field(default=Decimal("0"))
    optimization_potential_percent: Decimal = Field(default=Decimal("0"))


class BuildingAutomationOptimizerAgent(BuildingDecarbonizationBaseAgent[BuildingAutomationInput, BuildingAutomationOutput]):
    """GL-DECARB-BLD-009: Building Automation Optimizer."""

    AGENT_ID = "GL-DECARB-BLD-009"
    AGENT_VERSION = "1.0.0"
    TECHNOLOGY_FOCUS = TechnologyCategory.CONTROLS

    def _load_technology_database(self) -> None:
        self._technology_database["advanced_bms"] = TechnologySpec(
            technology_id="advanced_bms",
            category=TechnologyCategory.CONTROLS,
            name="Advanced BMS",
            description="AI-enabled building management system",
            efficiency_improvement_percent=Decimal("20"),
            lifespan_years=15
        )

    def analyze(self, input_data: BuildingAutomationInput) -> BuildingAutomationOutput:
        baseline = input_data.building_baseline
        target = input_data.target

        target_emissions = baseline.current_emissions_kgco2e_per_year * (
            1 - (target.target_reduction_percent or Decimal("0")) / 100
        )

        return BuildingAutomationOutput(
            analysis_id=self._generate_analysis_id(baseline.building_id),
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            building_id=baseline.building_id,
            baseline_emissions_kgco2e=baseline.current_emissions_kgco2e_per_year,
            target_emissions_kgco2e=target_emissions,
            target_year=target.target_year,
            is_valid=True
        )


# =============================================================================
# GL-DECARB-BLD-010: Tenant Engagement Agent
# =============================================================================

class TenantEngagementInput(DecarbonizationInput):
    """Input for tenant engagement analysis."""
    num_tenants: Optional[int] = Field(None, ge=0)
    current_tenant_participation_percent: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    has_green_lease: bool = Field(default=False)


class TenantEngagementOutput(DecarbonizationOutput):
    """Output for tenant engagement analysis."""
    engagement_programs: List[Dict[str, Any]] = Field(default_factory=list)
    potential_tenant_savings_kgco2e: Decimal = Field(default=Decimal("0"))
    green_lease_impact_kgco2e: Decimal = Field(default=Decimal("0"))


class TenantEngagementAgent(BuildingDecarbonizationBaseAgent[TenantEngagementInput, TenantEngagementOutput]):
    """GL-DECARB-BLD-010: Tenant Engagement Agent."""

    AGENT_ID = "GL-DECARB-BLD-010"
    AGENT_VERSION = "1.0.0"
    TECHNOLOGY_FOCUS = TechnologyCategory.OPERATIONS

    def _load_technology_database(self) -> None:
        pass

    def analyze(self, input_data: TenantEngagementInput) -> TenantEngagementOutput:
        baseline = input_data.building_baseline
        target = input_data.target

        target_emissions = baseline.current_emissions_kgco2e_per_year * (
            1 - (target.target_reduction_percent or Decimal("0")) / 100
        )

        return TenantEngagementOutput(
            analysis_id=self._generate_analysis_id(baseline.building_id),
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            building_id=baseline.building_id,
            baseline_emissions_kgco2e=baseline.current_emissions_kgco2e_per_year,
            target_emissions_kgco2e=target_emissions,
            target_year=target.target_year,
            is_valid=True
        )


# =============================================================================
# GL-DECARB-BLD-011: Portfolio Decarbonization Agent
# =============================================================================

class PortfolioBuilding(BaseModel):
    """Individual building in portfolio."""
    building_id: str
    building_type: str
    floor_area_sqm: Decimal
    current_emissions_kgco2e: Decimal
    priority_score: Optional[Decimal] = None


class PortfolioDecarbonizationInput(DecarbonizationInput):
    """Input for portfolio decarbonization."""
    portfolio_buildings: List[PortfolioBuilding] = Field(default_factory=list)
    portfolio_target_year: int = Field(default=2050)
    portfolio_target_reduction_percent: Decimal = Field(default=Decimal("50"))


class PortfolioDecarbonizationOutput(DecarbonizationOutput):
    """Output for portfolio decarbonization."""
    building_priorities: List[Dict[str, Any]] = Field(default_factory=list)
    portfolio_pathway: Optional[DecarbonizationPathway] = None
    portfolio_total_emissions_kgco2e: Decimal = Field(default=Decimal("0"))
    portfolio_reduction_potential_kgco2e: Decimal = Field(default=Decimal("0"))


class PortfolioDecarbonizationAgent(BuildingDecarbonizationBaseAgent[PortfolioDecarbonizationInput, PortfolioDecarbonizationOutput]):
    """GL-DECARB-BLD-011: Portfolio Decarbonization Agent."""

    AGENT_ID = "GL-DECARB-BLD-011"
    AGENT_VERSION = "1.0.0"

    def _load_technology_database(self) -> None:
        pass

    def analyze(self, input_data: PortfolioDecarbonizationInput) -> PortfolioDecarbonizationOutput:
        baseline = input_data.building_baseline
        target = input_data.target

        # Calculate portfolio totals
        portfolio_emissions = sum(b.current_emissions_kgco2e for b in input_data.portfolio_buildings)

        # Prioritize buildings by emission intensity
        priorities = []
        for building in input_data.portfolio_buildings:
            intensity = building.current_emissions_kgco2e / building.floor_area_sqm if building.floor_area_sqm > 0 else Decimal("0")
            priorities.append({
                "building_id": building.building_id,
                "emissions_kgco2e": str(building.current_emissions_kgco2e),
                "intensity_kgco2e_per_sqm": str(self._round_emissions(intensity)),
                "priority_rank": 0
            })

        # Sort by intensity (higher = higher priority)
        priorities.sort(key=lambda x: Decimal(x["intensity_kgco2e_per_sqm"]), reverse=True)
        for i, p in enumerate(priorities):
            p["priority_rank"] = i + 1

        target_emissions = baseline.current_emissions_kgco2e_per_year * (
            1 - (target.target_reduction_percent or Decimal("0")) / 100
        )

        return PortfolioDecarbonizationOutput(
            analysis_id=self._generate_analysis_id(baseline.building_id),
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            building_id=baseline.building_id,
            baseline_emissions_kgco2e=baseline.current_emissions_kgco2e_per_year,
            target_emissions_kgco2e=target_emissions,
            target_year=target.target_year,
            building_priorities=priorities,
            portfolio_total_emissions_kgco2e=self._round_emissions(portfolio_emissions),
            is_valid=True
        )


# =============================================================================
# GL-DECARB-BLD-012: Green Lease Agent
# =============================================================================

class GreenLeaseInput(DecarbonizationInput):
    """Input for green lease analysis."""
    current_lease_type: str = Field(default="gross")
    tenant_metering: bool = Field(default=False)
    cost_recovery_clause: bool = Field(default=False)


class GreenLeaseOutput(DecarbonizationOutput):
    """Output for green lease analysis."""
    recommended_provisions: List[Dict[str, str]] = Field(default_factory=list)
    split_incentive_barrier: bool = Field(default=False)
    green_lease_savings_potential_kgco2e: Decimal = Field(default=Decimal("0"))


class GreenLeaseAgent(BuildingDecarbonizationBaseAgent[GreenLeaseInput, GreenLeaseOutput]):
    """GL-DECARB-BLD-012: Green Lease Agent."""

    AGENT_ID = "GL-DECARB-BLD-012"
    AGENT_VERSION = "1.0.0"
    TECHNOLOGY_FOCUS = TechnologyCategory.OPERATIONS

    def _load_technology_database(self) -> None:
        pass

    def analyze(self, input_data: GreenLeaseInput) -> GreenLeaseOutput:
        baseline = input_data.building_baseline
        target = input_data.target

        provisions = []
        split_incentive = False

        if input_data.current_lease_type == "gross" and not input_data.cost_recovery_clause:
            split_incentive = True
            provisions.append({
                "provision": "Cost Recovery Clause",
                "description": "Allow landlord to recover efficiency investment costs"
            })

        if not input_data.tenant_metering:
            provisions.append({
                "provision": "Tenant Submetering",
                "description": "Install submeters to track tenant energy use"
            })

        provisions.append({
            "provision": "Energy Performance Disclosure",
            "description": "Require annual energy data sharing"
        })

        target_emissions = baseline.current_emissions_kgco2e_per_year * (
            1 - (target.target_reduction_percent or Decimal("0")) / 100
        )

        # Green lease can enable 10-15% savings through behavioral change
        potential_savings = baseline.current_emissions_kgco2e_per_year * Decimal("0.10")

        return GreenLeaseOutput(
            analysis_id=self._generate_analysis_id(baseline.building_id),
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            building_id=baseline.building_id,
            baseline_emissions_kgco2e=baseline.current_emissions_kgco2e_per_year,
            target_emissions_kgco2e=target_emissions,
            target_year=target.target_year,
            recommended_provisions=provisions,
            split_incentive_barrier=split_incentive,
            green_lease_savings_potential_kgco2e=self._round_emissions(potential_savings),
            is_valid=True
        )
