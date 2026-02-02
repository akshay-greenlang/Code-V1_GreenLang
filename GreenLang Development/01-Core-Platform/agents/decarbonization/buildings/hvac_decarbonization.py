# -*- coding: utf-8 -*-
"""
GL-DECARB-BLD-002: HVAC Decarbonization Agent
==============================================

Specialized agent for HVAC system decarbonization including
heat pump transitions and refrigerant management.

Author: GreenLang Framework Team
Agent ID: GL-DECARB-BLD-002
Version: 1.0.0
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Dict, List, Optional

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
    HEAT_PUMP_COP,
)

logger = logging.getLogger(__name__)


class HVACDecarbonizationInput(DecarbonizationInput):
    """Input for HVAC decarbonization analysis."""
    current_heating_system: str = Field(default="gas_boiler")
    current_cooling_system: str = Field(default="electric_chiller")
    heating_capacity_kw: Optional[Decimal] = Field(None, ge=0)
    cooling_capacity_kw: Optional[Decimal] = Field(None, ge=0)
    current_heating_efficiency: Optional[Decimal] = Field(None, ge=0)
    current_cooling_cop: Optional[Decimal] = Field(None, ge=0)
    climate_zone: Optional[str] = None
    has_existing_ductwork: bool = Field(default=True)


class HVACDecarbonizationOutput(DecarbonizationOutput):
    """Output for HVAC decarbonization analysis."""
    recommended_heat_pump_type: Optional[str] = None
    estimated_cop: Optional[Decimal] = None
    heating_electrification_savings_kgco2e: Decimal = Field(default=Decimal("0"))
    refrigerant_transition_savings_kgco2e: Decimal = Field(default=Decimal("0"))


class HVACDecarbonizationAgent(BuildingDecarbonizationBaseAgent[HVACDecarbonizationInput, HVACDecarbonizationOutput]):
    """
    GL-DECARB-BLD-002: HVAC Decarbonization Agent.

    Analyzes HVAC systems and recommends heat pump transitions
    and refrigerant management strategies.
    """

    AGENT_ID = "GL-DECARB-BLD-002"
    AGENT_VERSION = "1.0.0"
    TECHNOLOGY_FOCUS = TechnologyCategory.HVAC

    def _load_technology_database(self) -> None:
        """Load HVAC decarbonization technologies."""
        self._technology_database["air_source_heat_pump"] = TechnologySpec(
            technology_id="air_source_heat_pump",
            category=TechnologyCategory.HVAC,
            name="Air Source Heat Pump",
            description="High-efficiency ASHP for heating and cooling",
            efficiency_improvement_percent=Decimal("60"),
            lifespan_years=20
        )

        self._technology_database["ground_source_heat_pump"] = TechnologySpec(
            technology_id="ground_source_heat_pump",
            category=TechnologyCategory.HVAC,
            name="Ground Source Heat Pump",
            description="Geothermal heat pump system",
            efficiency_improvement_percent=Decimal("75"),
            lifespan_years=25
        )

        self._technology_database["vrf_system"] = TechnologySpec(
            technology_id="vrf_system",
            category=TechnologyCategory.HVAC,
            name="Variable Refrigerant Flow",
            description="VRF system for zoned heating/cooling",
            efficiency_improvement_percent=Decimal("40"),
            lifespan_years=20
        )

    def analyze(self, input_data: HVACDecarbonizationInput) -> HVACDecarbonizationOutput:
        """Analyze HVAC and recommend heat pump transition."""
        baseline = input_data.building_baseline
        target = input_data.target
        measures: List[DecarbonizationMeasure] = []

        grid_ef = Decimal("0.379")
        gas_ef = Decimal("0.181")  # kgCO2e per kWh

        # Determine best heat pump type
        recommended_hp = "air_source_heat_pump"
        estimated_cop = HEAT_PUMP_COP["air_source"]

        if input_data.climate_zone in ["5A", "5B", "6A", "6B", "7"]:
            recommended_hp = "ground_source_heat_pump"
            estimated_cop = HEAT_PUMP_COP["ground_source"]

        # Calculate heating electrification savings
        if "gas" in input_data.current_heating_system:
            heating_energy = baseline.current_energy_kwh_per_year * (baseline.natural_gas_percent / 100)
            current_efficiency = input_data.current_heating_efficiency or Decimal("0.85")

            # Heat pump requires less energy due to COP > 1
            hp_energy = heating_energy * current_efficiency / estimated_cop
            energy_savings = heating_energy - hp_energy

            # Emissions: gas heating vs electric heat pump
            current_emissions = heating_energy * gas_ef
            new_emissions = hp_energy * grid_ef
            emission_savings = current_emissions - new_emissions

            capacity = input_data.heating_capacity_kw or (baseline.gross_floor_area_sqm * Decimal("0.05"))
            hp_cost = capacity * Decimal("2500") if "ground" in recommended_hp else capacity * Decimal("1500")

            tech = self._technology_database[recommended_hp]
            measures.append(self._create_measure(
                measure_id="HP-001",
                name=tech.name,
                description=f"Replace gas heating with {tech.name.lower()}",
                technology=tech,
                capital_cost=hp_cost,
                annual_savings=energy_savings * input_data.electricity_cost_per_kwh,
                energy_savings_kwh=energy_savings,
                emission_reduction=emission_savings,
                priority=RecommendationPriority.HIGH,
                phase=ImplementationPhase.SHORT_TERM,
                discount_rate=input_data.discount_rate_percent
            ))

        total_reduction = sum(m.annual_emission_reduction_kgco2e for m in measures)
        total_investment = sum(m.financial.capital_cost_usd for m in measures)
        total_savings = sum(m.financial.annual_savings_usd for m in measures)

        pathway = DecarbonizationPathway(
            pathway_id=self._generate_analysis_id(baseline.building_id),
            name="HVAC Decarbonization Pathway",
            description="Heat pump transition pathway",
            target_year=target.target_year,
            target_reduction_percent=target.target_reduction_percent or Decimal("0"),
            short_term_measures=measures,
            total_capital_cost_usd=total_investment,
            total_annual_savings_usd=total_savings,
            total_emission_reduction_kgco2e=total_reduction
        )

        target_emissions = baseline.current_emissions_kgco2e_per_year * (
            1 - (target.target_reduction_percent or Decimal("0")) / 100
        )

        return HVACDecarbonizationOutput(
            analysis_id=self._generate_analysis_id(baseline.building_id),
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            building_id=baseline.building_id,
            baseline_emissions_kgco2e=baseline.current_emissions_kgco2e_per_year,
            target_emissions_kgco2e=target_emissions,
            target_year=target.target_year,
            pathway=pathway,
            total_reduction_kgco2e=self._round_emissions(total_reduction),
            total_reduction_percent=self._round_financial(
                total_reduction / baseline.current_emissions_kgco2e_per_year * 100
            ) if baseline.current_emissions_kgco2e_per_year > 0 else Decimal("0"),
            total_investment_usd=self._round_financial(total_investment),
            total_annual_savings_usd=self._round_financial(total_savings),
            target_achievable=total_reduction >= baseline.current_emissions_kgco2e_per_year - target_emissions,
            recommended_heat_pump_type=recommended_hp,
            estimated_cop=estimated_cop,
            heating_electrification_savings_kgco2e=self._round_emissions(total_reduction),
            is_valid=True
        )
