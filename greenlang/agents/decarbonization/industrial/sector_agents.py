# -*- coding: utf-8 -*-
"""
GreenLang Industrial Decarbonization Sector Agents
===================================================

This module contains decarbonization agents for major industrial sectors:
    - GL-DECARB-IND-001: Steel Decarbonization
    - GL-DECARB-IND-002: Cement Decarbonization
    - GL-DECARB-IND-003: Chemicals Decarbonization
    - GL-DECARB-IND-004: Aluminum Decarbonization
    - GL-DECARB-IND-005: Pulp & Paper Decarbonization
    - GL-DECARB-IND-006: Glass Decarbonization
    - GL-DECARB-IND-007: Food Processing Decarbonization
    - GL-DECARB-IND-008: Pharmaceutical Decarbonization
    - GL-DECARB-IND-009: Electronics Decarbonization
    - GL-DECARB-IND-010: Automotive Decarbonization
    - GL-DECARB-IND-011: Textiles Decarbonization
    - GL-DECARB-IND-012: Mining Decarbonization

Author: GreenLang Framework Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import List

from pydantic import Field

from .base import (
    IndustrialDecarbonizationBaseAgent,
    DecarbonizationInput,
    DecarbonizationOutput,
    DecarbonizationPathway,
    Technology,
    TechnologyReadiness,
    DecarbonizationLever,
    Milestone,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SECTOR INPUT/OUTPUT MODELS
# =============================================================================

class SteelDecarbInput(DecarbonizationInput):
    """Input for Steel Decarbonization."""
    sector: str = Field(default="Steel")
    current_route: str = Field(default="BF_BOF")
    scrap_availability_pct: Decimal = Field(default=Decimal("30"), ge=0, le=100)
    hydrogen_access: bool = Field(default=False)


class SteelDecarbOutput(DecarbonizationOutput):
    """Output for Steel Decarbonization."""
    recommended_route: str = Field(default="")
    hydrogen_requirement_tonnes: Decimal = Field(default=Decimal("0"))


# =============================================================================
# GL-DECARB-IND-001: STEEL DECARBONIZATION
# =============================================================================

class SteelDecarbonizationAgent(IndustrialDecarbonizationBaseAgent[SteelDecarbInput, SteelDecarbOutput]):
    """
    GL-DECARB-IND-001: Steel Decarbonization Agent

    Key Decarbonization Levers for Steel:
        1. Energy Efficiency: 10-15% reduction potential
        2. Scrap-based EAF: Up to 75% reduction vs BF-BOF
        3. DRI-EAF with Natural Gas: ~40% reduction
        4. H2-DRI: Up to 95% reduction (green H2)
        5. CCUS on BF-BOF: 60-90% capture potential

    Abatement Costs (2024):
        - EAF shift: -20 to +50 USD/tCO2
        - DRI-EAF: 50-100 USD/tCO2
        - H2-DRI: 80-200 USD/tCO2
        - CCUS: 80-150 USD/tCO2
    """

    AGENT_ID = "GL-DECARB-IND-001"
    AGENT_VERSION = "1.0.0"
    SECTOR = "Steel"

    def _load_technologies(self) -> None:
        """Load steel decarbonization technologies."""
        self._technologies = [
            Technology(
                technology_id="steel_efficiency",
                name="Energy Efficiency Improvements",
                description="Heat recovery, process optimization, digital twins",
                lever=DecarbonizationLever.ENERGY_EFFICIENCY,
                readiness=TechnologyReadiness.COMMERCIAL,
                abatement_potential_pct=Decimal("15"),
                abatement_cost_usd_per_tco2=Decimal("-20"),  # Cost savings
                capex_usd_per_annual_tonne=Decimal("10"),
                deployment_year_earliest=2024,
                ramp_up_years=2
            ),
            Technology(
                technology_id="steel_eaf_shift",
                name="EAF Steel Production",
                description="Shift from BF-BOF to scrap-based EAF",
                lever=DecarbonizationLever.ELECTRIFICATION,
                readiness=TechnologyReadiness.COMMERCIAL,
                abatement_potential_pct=Decimal("75"),
                abatement_cost_usd_per_tco2=Decimal("30"),
                capex_usd_per_annual_tonne=Decimal("300"),
                deployment_year_earliest=2025,
                ramp_up_years=5
            ),
            Technology(
                technology_id="steel_dri_ng",
                name="DRI-EAF with Natural Gas",
                description="Direct Reduced Iron with natural gas + EAF",
                lever=DecarbonizationLever.FUEL_SWITCHING,
                readiness=TechnologyReadiness.COMMERCIAL,
                abatement_potential_pct=Decimal("40"),
                abatement_cost_usd_per_tco2=Decimal("60"),
                capex_usd_per_annual_tonne=Decimal("350"),
                deployment_year_earliest=2025,
                ramp_up_years=4
            ),
            Technology(
                technology_id="steel_h2_dri",
                name="Hydrogen-based DRI",
                description="H2-DRI with green hydrogen + EAF",
                lever=DecarbonizationLever.HYDROGEN,
                readiness=TechnologyReadiness.EARLY_ADOPTION,
                abatement_potential_pct=Decimal("95"),
                abatement_cost_usd_per_tco2=Decimal("120"),
                capex_usd_per_annual_tonne=Decimal("500"),
                deployment_year_earliest=2028,
                ramp_up_years=7
            ),
            Technology(
                technology_id="steel_ccus",
                name="CCUS on BF-BOF",
                description="Carbon capture on existing blast furnaces",
                lever=DecarbonizationLever.CCUS,
                readiness=TechnologyReadiness.DEMONSTRATION,
                abatement_potential_pct=Decimal("70"),
                abatement_cost_usd_per_tco2=Decimal("100"),
                capex_usd_per_annual_tonne=Decimal("200"),
                deployment_year_earliest=2027,
                ramp_up_years=5
            ),
        ]

    def generate_pathway(self, input_data: SteelDecarbInput) -> SteelDecarbOutput:
        """Generate steel decarbonization pathway."""
        calc_id = self._generate_calculation_id(input_data.facility_id)

        # Filter technologies based on constraints
        available_techs = self._filter_technologies(input_data)

        # Select optimal technology mix
        selected_techs = self._select_technologies(input_data, available_techs)

        # Calculate pathway
        trajectory = self._calculate_trajectory(
            input_data.baseline_emissions_tco2e,
            input_data.baseline_year,
            input_data.target_year,
            selected_techs
        )

        # Calculate financials
        total_capex = sum(
            (tech.capex_usd_per_annual_tonne or Decimal("0")) * input_data.baseline_production_tonnes
            for tech in selected_techs
        )

        total_abatement = (
            input_data.baseline_emissions_tco2e -
            trajectory.get(input_data.target_year, input_data.baseline_emissions_tco2e)
        )

        avg_abatement_cost = sum(
            tech.abatement_cost_usd_per_tco2
            for tech in selected_techs
        ) / len(selected_techs) if selected_techs else Decimal("0")

        # Create pathway
        pathway = DecarbonizationPathway(
            pathway_id=f"pathway_{calc_id}",
            name=f"Steel Decarbonization - {input_data.facility_id}",
            baseline_emissions_tco2e=input_data.baseline_emissions_tco2e,
            baseline_year=input_data.baseline_year,
            target_year=input_data.target_year,
            target_reduction_pct=input_data.target_reduction_pct,
            annual_trajectory=trajectory,
            technologies=selected_techs,
            total_capex_usd=self._round_cost(total_capex),
            average_abatement_cost_usd_per_tco2=self._round_cost(avg_abatement_cost)
        )

        # Milestones
        milestones = [
            Milestone(year=2030, target_reduction_pct=Decimal("30"), description="Short-term target"),
            Milestone(year=2040, target_reduction_pct=Decimal("60"), description="Medium-term target"),
            Milestone(year=2050, target_reduction_pct=Decimal("90"), description="Net zero target"),
        ]

        # Determine recommended route
        h2_tech = next((t for t in selected_techs if t.technology_id == "steel_h2_dri"), None)
        recommended_route = "H2_DRI" if h2_tech else "EAF" if input_data.scrap_availability_pct > 50 else "DRI_EAF"

        return SteelDecarbOutput(
            calculation_id=calc_id,
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            facility_id=input_data.facility_id,
            sector=self.SECTOR,
            baseline_emissions_tco2e=input_data.baseline_emissions_tco2e,
            recommended_pathway=pathway,
            total_abatement_tco2e=self._round_emissions(total_abatement),
            abatement_pct=self._round_emissions(
                total_abatement / input_data.baseline_emissions_tco2e * Decimal("100")
            ),
            total_capex_usd=self._round_cost(total_capex),
            levelized_abatement_cost_usd_per_tco2=self._round_cost(avg_abatement_cost),
            key_milestones=milestones,
            first_technology_deployment_year=min(t.deployment_year_earliest for t in selected_techs),
            recommended_route=recommended_route,
            hydrogen_requirement_tonnes=self._calculate_h2_requirement(input_data, selected_techs),
            is_valid=True
        )

    def _filter_technologies(self, input_data: SteelDecarbInput) -> List[Technology]:
        """Filter technologies based on input constraints."""
        available = []
        for tech in self._technologies:
            # Check excluded levers
            if tech.lever in input_data.excluded_levers:
                continue
            # Check budget
            if input_data.budget_capex_usd and tech.capex_usd_per_annual_tonne:
                tech_cost = tech.capex_usd_per_annual_tonne * input_data.baseline_production_tonnes
                if tech_cost > input_data.budget_capex_usd:
                    continue
            # Check H2 access
            if tech.lever == DecarbonizationLever.HYDROGEN and not input_data.hydrogen_access:
                continue
            available.append(tech)
        return available

    def _select_technologies(
        self,
        input_data: SteelDecarbInput,
        available: List[Technology]
    ) -> List[Technology]:
        """Select optimal technology mix."""
        # Always include efficiency
        selected = [t for t in available if t.lever == DecarbonizationLever.ENERGY_EFFICIENCY]

        # Prioritize by abatement potential and cost
        others = [t for t in available if t not in selected]
        others.sort(key=lambda t: (
            -t.abatement_potential_pct,
            t.abatement_cost_usd_per_tco2
        ))

        # Add best options
        for tech in others[:2]:
            selected.append(tech)

        return selected

    def _calculate_h2_requirement(
        self,
        input_data: SteelDecarbInput,
        technologies: List[Technology]
    ) -> Decimal:
        """Calculate hydrogen requirement for H2-DRI."""
        h2_tech = next((t for t in technologies if "h2" in t.technology_id.lower()), None)
        if not h2_tech:
            return Decimal("0")

        # ~54 kg H2 per tonne steel for H2-DRI
        h2_per_tonne = Decimal("0.054")
        return self._round_emissions(input_data.baseline_production_tonnes * h2_per_tonne)


# =============================================================================
# GL-DECARB-IND-002 to IND-012: OTHER SECTOR AGENTS
# =============================================================================

class CementDecarbonizationAgent(IndustrialDecarbonizationBaseAgent):
    """GL-DECARB-IND-002: Cement Decarbonization Agent"""

    AGENT_ID = "GL-DECARB-IND-002"
    SECTOR = "Cement"

    def _load_technologies(self) -> None:
        self._technologies = [
            Technology(
                technology_id="cement_efficiency",
                name="Thermal and Electrical Efficiency",
                lever=DecarbonizationLever.ENERGY_EFFICIENCY,
                readiness=TechnologyReadiness.COMMERCIAL,
                abatement_potential_pct=Decimal("10"),
                abatement_cost_usd_per_tco2=Decimal("-15"),
                deployment_year_earliest=2024,
                ramp_up_years=2
            ),
            Technology(
                technology_id="cement_alt_fuels",
                name="Alternative Fuels (Biomass, Waste)",
                lever=DecarbonizationLever.FUEL_SWITCHING,
                readiness=TechnologyReadiness.COMMERCIAL,
                abatement_potential_pct=Decimal("30"),
                abatement_cost_usd_per_tco2=Decimal("20"),
                deployment_year_earliest=2024,
                ramp_up_years=3
            ),
            Technology(
                technology_id="cement_clinker_sub",
                name="Clinker Substitution (SCMs)",
                lever=DecarbonizationLever.MATERIAL_EFFICIENCY,
                readiness=TechnologyReadiness.COMMERCIAL,
                abatement_potential_pct=Decimal("25"),
                abatement_cost_usd_per_tco2=Decimal("10"),
                deployment_year_earliest=2024,
                ramp_up_years=2
            ),
            Technology(
                technology_id="cement_ccus",
                name="Carbon Capture on Cement Kilns",
                lever=DecarbonizationLever.CCUS,
                readiness=TechnologyReadiness.DEMONSTRATION,
                abatement_potential_pct=Decimal("90"),
                abatement_cost_usd_per_tco2=Decimal("80"),
                deployment_year_earliest=2028,
                ramp_up_years=6
            ),
        ]

    def generate_pathway(self, input_data: DecarbonizationInput) -> DecarbonizationOutput:
        calc_id = self._generate_calculation_id(input_data.facility_id)
        trajectory = self._calculate_trajectory(
            input_data.baseline_emissions_tco2e,
            input_data.baseline_year,
            input_data.target_year,
            self._technologies
        )
        pathway = DecarbonizationPathway(
            pathway_id=f"pathway_{calc_id}",
            name=f"Cement Decarbonization - {input_data.facility_id}",
            baseline_emissions_tco2e=input_data.baseline_emissions_tco2e,
            target_reduction_pct=input_data.target_reduction_pct,
            annual_trajectory=trajectory,
            technologies=self._technologies
        )
        total_abatement = input_data.baseline_emissions_tco2e - trajectory.get(input_data.target_year, input_data.baseline_emissions_tco2e)
        return DecarbonizationOutput(
            calculation_id=calc_id, agent_id=self.AGENT_ID, agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(), facility_id=input_data.facility_id,
            sector=self.SECTOR, baseline_emissions_tco2e=input_data.baseline_emissions_tco2e,
            recommended_pathway=pathway, total_abatement_tco2e=total_abatement,
            abatement_pct=total_abatement / input_data.baseline_emissions_tco2e * Decimal("100"),
            is_valid=True
        )


class ChemicalsDecarbonizationAgent(IndustrialDecarbonizationBaseAgent):
    """GL-DECARB-IND-003: Chemicals Decarbonization Agent"""
    AGENT_ID = "GL-DECARB-IND-003"
    SECTOR = "Chemicals"

    def _load_technologies(self) -> None:
        self._technologies = [
            Technology(
                technology_id="chem_efficiency", name="Process Integration and Heat Recovery",
                lever=DecarbonizationLever.ENERGY_EFFICIENCY, readiness=TechnologyReadiness.COMMERCIAL,
                abatement_potential_pct=Decimal("15"), abatement_cost_usd_per_tco2=Decimal("-10"),
                deployment_year_earliest=2024, ramp_up_years=2
            ),
            Technology(
                technology_id="chem_electrolysis", name="Green Hydrogen via Electrolysis",
                lever=DecarbonizationLever.HYDROGEN, readiness=TechnologyReadiness.EARLY_ADOPTION,
                abatement_potential_pct=Decimal("70"), abatement_cost_usd_per_tco2=Decimal("100"),
                deployment_year_earliest=2027, ramp_up_years=5
            ),
            Technology(
                technology_id="chem_ccus", name="Carbon Capture on Ammonia/Methanol",
                lever=DecarbonizationLever.CCUS, readiness=TechnologyReadiness.COMMERCIAL,
                abatement_potential_pct=Decimal("85"), abatement_cost_usd_per_tco2=Decimal("60"),
                deployment_year_earliest=2025, ramp_up_years=4
            ),
        ]

    def generate_pathway(self, input_data: DecarbonizationInput) -> DecarbonizationOutput:
        calc_id = self._generate_calculation_id(input_data.facility_id)
        trajectory = self._calculate_trajectory(input_data.baseline_emissions_tco2e, input_data.baseline_year, input_data.target_year, self._technologies)
        pathway = DecarbonizationPathway(pathway_id=f"pathway_{calc_id}", name=f"Chemicals Decarbonization", baseline_emissions_tco2e=input_data.baseline_emissions_tco2e, annual_trajectory=trajectory, technologies=self._technologies, target_reduction_pct=input_data.target_reduction_pct)
        total_abatement = input_data.baseline_emissions_tco2e - trajectory.get(input_data.target_year, input_data.baseline_emissions_tco2e)
        return DecarbonizationOutput(calculation_id=calc_id, agent_id=self.AGENT_ID, agent_version=self.AGENT_VERSION, timestamp=self._get_timestamp(), facility_id=input_data.facility_id, sector=self.SECTOR, baseline_emissions_tco2e=input_data.baseline_emissions_tco2e, recommended_pathway=pathway, total_abatement_tco2e=total_abatement, is_valid=True)


class AluminumDecarbonizationAgent(IndustrialDecarbonizationBaseAgent):
    """GL-DECARB-IND-004: Aluminum Decarbonization Agent"""
    AGENT_ID = "GL-DECARB-IND-004"
    SECTOR = "Aluminum"

    def _load_technologies(self) -> None:
        self._technologies = [
            Technology(technology_id="al_renewable_power", name="Renewable Electricity for Smelting", lever=DecarbonizationLever.RENEWABLE_ENERGY, readiness=TechnologyReadiness.COMMERCIAL, abatement_potential_pct=Decimal("80"), abatement_cost_usd_per_tco2=Decimal("40"), deployment_year_earliest=2024, ramp_up_years=5),
            Technology(technology_id="al_inert_anode", name="Inert Anode Technology", lever=DecarbonizationLever.PROCESS_CHANGE, readiness=TechnologyReadiness.DEMONSTRATION, abatement_potential_pct=Decimal("15"), abatement_cost_usd_per_tco2=Decimal("80"), deployment_year_earliest=2030, ramp_up_years=8),
            Technology(technology_id="al_recycling", name="Increased Secondary Aluminum", lever=DecarbonizationLever.CIRCULAR_ECONOMY, readiness=TechnologyReadiness.COMMERCIAL, abatement_potential_pct=Decimal("95"), abatement_cost_usd_per_tco2=Decimal("-30"), deployment_year_earliest=2024, ramp_up_years=3),
        ]

    def generate_pathway(self, input_data: DecarbonizationInput) -> DecarbonizationOutput:
        calc_id = self._generate_calculation_id(input_data.facility_id)
        trajectory = self._calculate_trajectory(input_data.baseline_emissions_tco2e, input_data.baseline_year, input_data.target_year, self._technologies)
        pathway = DecarbonizationPathway(pathway_id=f"pathway_{calc_id}", name=f"Aluminum Decarbonization", baseline_emissions_tco2e=input_data.baseline_emissions_tco2e, annual_trajectory=trajectory, technologies=self._technologies, target_reduction_pct=input_data.target_reduction_pct)
        total_abatement = input_data.baseline_emissions_tco2e - trajectory.get(input_data.target_year, input_data.baseline_emissions_tco2e)
        return DecarbonizationOutput(calculation_id=calc_id, agent_id=self.AGENT_ID, agent_version=self.AGENT_VERSION, timestamp=self._get_timestamp(), facility_id=input_data.facility_id, sector=self.SECTOR, baseline_emissions_tco2e=input_data.baseline_emissions_tco2e, recommended_pathway=pathway, total_abatement_tco2e=total_abatement, is_valid=True)


# Create similar agents for remaining sectors
class PulpPaperDecarbonizationAgent(IndustrialDecarbonizationBaseAgent):
    """GL-DECARB-IND-005: Pulp & Paper Decarbonization Agent"""
    AGENT_ID = "GL-DECARB-IND-005"
    SECTOR = "Pulp & Paper"
    def _load_technologies(self) -> None:
        self._technologies = [
            Technology(technology_id="pp_biomass", name="Biomass Energy", lever=DecarbonizationLever.FUEL_SWITCHING, readiness=TechnologyReadiness.COMMERCIAL, abatement_potential_pct=Decimal("60"), abatement_cost_usd_per_tco2=Decimal("20"), deployment_year_earliest=2024, ramp_up_years=3),
            Technology(technology_id="pp_efficiency", name="Process Efficiency", lever=DecarbonizationLever.ENERGY_EFFICIENCY, readiness=TechnologyReadiness.COMMERCIAL, abatement_potential_pct=Decimal("20"), abatement_cost_usd_per_tco2=Decimal("-15"), deployment_year_earliest=2024, ramp_up_years=2),
        ]
    def generate_pathway(self, input_data: DecarbonizationInput) -> DecarbonizationOutput:
        calc_id = self._generate_calculation_id(input_data.facility_id)
        trajectory = self._calculate_trajectory(input_data.baseline_emissions_tco2e, input_data.baseline_year, input_data.target_year, self._technologies)
        pathway = DecarbonizationPathway(pathway_id=f"pathway_{calc_id}", name=f"Pulp & Paper Decarbonization", baseline_emissions_tco2e=input_data.baseline_emissions_tco2e, annual_trajectory=trajectory, technologies=self._technologies, target_reduction_pct=input_data.target_reduction_pct)
        return DecarbonizationOutput(calculation_id=calc_id, agent_id=self.AGENT_ID, agent_version=self.AGENT_VERSION, timestamp=self._get_timestamp(), facility_id=input_data.facility_id, sector=self.SECTOR, baseline_emissions_tco2e=input_data.baseline_emissions_tco2e, recommended_pathway=pathway, is_valid=True)


class GlassDecarbonizationAgent(IndustrialDecarbonizationBaseAgent):
    """GL-DECARB-IND-006: Glass Decarbonization Agent"""
    AGENT_ID = "GL-DECARB-IND-006"
    SECTOR = "Glass"
    def _load_technologies(self) -> None:
        self._technologies = [
            Technology(technology_id="glass_cullet", name="Increased Cullet Usage", lever=DecarbonizationLever.CIRCULAR_ECONOMY, readiness=TechnologyReadiness.COMMERCIAL, abatement_potential_pct=Decimal("25"), abatement_cost_usd_per_tco2=Decimal("-10"), deployment_year_earliest=2024, ramp_up_years=2),
            Technology(technology_id="glass_electric", name="Electric Melting", lever=DecarbonizationLever.ELECTRIFICATION, readiness=TechnologyReadiness.EARLY_ADOPTION, abatement_potential_pct=Decimal("80"), abatement_cost_usd_per_tco2=Decimal("60"), deployment_year_earliest=2027, ramp_up_years=5),
            Technology(technology_id="glass_hydrogen", name="Hydrogen-fired Furnaces", lever=DecarbonizationLever.HYDROGEN, readiness=TechnologyReadiness.DEMONSTRATION, abatement_potential_pct=Decimal("90"), abatement_cost_usd_per_tco2=Decimal("100"), deployment_year_earliest=2030, ramp_up_years=6),
        ]
    def generate_pathway(self, input_data: DecarbonizationInput) -> DecarbonizationOutput:
        calc_id = self._generate_calculation_id(input_data.facility_id)
        trajectory = self._calculate_trajectory(input_data.baseline_emissions_tco2e, input_data.baseline_year, input_data.target_year, self._technologies)
        pathway = DecarbonizationPathway(pathway_id=f"pathway_{calc_id}", name=f"Glass Decarbonization", baseline_emissions_tco2e=input_data.baseline_emissions_tco2e, annual_trajectory=trajectory, technologies=self._technologies, target_reduction_pct=input_data.target_reduction_pct)
        return DecarbonizationOutput(calculation_id=calc_id, agent_id=self.AGENT_ID, agent_version=self.AGENT_VERSION, timestamp=self._get_timestamp(), facility_id=input_data.facility_id, sector=self.SECTOR, baseline_emissions_tco2e=input_data.baseline_emissions_tco2e, recommended_pathway=pathway, is_valid=True)


class FoodProcessingDecarbonizationAgent(IndustrialDecarbonizationBaseAgent):
    """GL-DECARB-IND-007: Food Processing Decarbonization Agent"""
    AGENT_ID = "GL-DECARB-IND-007"
    SECTOR = "Food Processing"
    def _load_technologies(self) -> None:
        self._technologies = [
            Technology(technology_id="food_heat_pump", name="Industrial Heat Pumps", lever=DecarbonizationLever.ELECTRIFICATION, readiness=TechnologyReadiness.COMMERCIAL, abatement_potential_pct=Decimal("50"), abatement_cost_usd_per_tco2=Decimal("30"), deployment_year_earliest=2024, ramp_up_years=3),
            Technology(technology_id="food_biogas", name="Biogas from Waste", lever=DecarbonizationLever.FUEL_SWITCHING, readiness=TechnologyReadiness.COMMERCIAL, abatement_potential_pct=Decimal("30"), abatement_cost_usd_per_tco2=Decimal("15"), deployment_year_earliest=2024, ramp_up_years=2),
            Technology(technology_id="food_refrigerant", name="Low-GWP Refrigerants", lever=DecarbonizationLever.PROCESS_CHANGE, readiness=TechnologyReadiness.COMMERCIAL, abatement_potential_pct=Decimal("10"), abatement_cost_usd_per_tco2=Decimal("40"), deployment_year_earliest=2024, ramp_up_years=3),
        ]
    def generate_pathway(self, input_data: DecarbonizationInput) -> DecarbonizationOutput:
        calc_id = self._generate_calculation_id(input_data.facility_id)
        trajectory = self._calculate_trajectory(input_data.baseline_emissions_tco2e, input_data.baseline_year, input_data.target_year, self._technologies)
        pathway = DecarbonizationPathway(pathway_id=f"pathway_{calc_id}", name=f"Food Processing Decarbonization", baseline_emissions_tco2e=input_data.baseline_emissions_tco2e, annual_trajectory=trajectory, technologies=self._technologies, target_reduction_pct=input_data.target_reduction_pct)
        return DecarbonizationOutput(calculation_id=calc_id, agent_id=self.AGENT_ID, agent_version=self.AGENT_VERSION, timestamp=self._get_timestamp(), facility_id=input_data.facility_id, sector=self.SECTOR, baseline_emissions_tco2e=input_data.baseline_emissions_tco2e, recommended_pathway=pathway, is_valid=True)


class PharmaceuticalDecarbonizationAgent(IndustrialDecarbonizationBaseAgent):
    """GL-DECARB-IND-008: Pharmaceutical Decarbonization Agent"""
    AGENT_ID = "GL-DECARB-IND-008"
    SECTOR = "Pharmaceutical"
    def _load_technologies(self) -> None:
        self._technologies = [
            Technology(technology_id="pharma_renewable", name="Renewable Electricity", lever=DecarbonizationLever.RENEWABLE_ENERGY, readiness=TechnologyReadiness.COMMERCIAL, abatement_potential_pct=Decimal("60"), abatement_cost_usd_per_tco2=Decimal("25"), deployment_year_earliest=2024, ramp_up_years=3),
            Technology(technology_id="pharma_heat_pump", name="Heat Pumps for HVAC", lever=DecarbonizationLever.ELECTRIFICATION, readiness=TechnologyReadiness.COMMERCIAL, abatement_potential_pct=Decimal("30"), abatement_cost_usd_per_tco2=Decimal("35"), deployment_year_earliest=2024, ramp_up_years=3),
        ]
    def generate_pathway(self, input_data: DecarbonizationInput) -> DecarbonizationOutput:
        calc_id = self._generate_calculation_id(input_data.facility_id)
        trajectory = self._calculate_trajectory(input_data.baseline_emissions_tco2e, input_data.baseline_year, input_data.target_year, self._technologies)
        pathway = DecarbonizationPathway(pathway_id=f"pathway_{calc_id}", name=f"Pharma Decarbonization", baseline_emissions_tco2e=input_data.baseline_emissions_tco2e, annual_trajectory=trajectory, technologies=self._technologies, target_reduction_pct=input_data.target_reduction_pct)
        return DecarbonizationOutput(calculation_id=calc_id, agent_id=self.AGENT_ID, agent_version=self.AGENT_VERSION, timestamp=self._get_timestamp(), facility_id=input_data.facility_id, sector=self.SECTOR, baseline_emissions_tco2e=input_data.baseline_emissions_tco2e, recommended_pathway=pathway, is_valid=True)


class ElectronicsDecarbonizationAgent(IndustrialDecarbonizationBaseAgent):
    """GL-DECARB-IND-009: Electronics Decarbonization Agent"""
    AGENT_ID = "GL-DECARB-IND-009"
    SECTOR = "Electronics"
    def _load_technologies(self) -> None:
        self._technologies = [
            Technology(technology_id="elec_renewable", name="100% Renewable Electricity", lever=DecarbonizationLever.RENEWABLE_ENERGY, readiness=TechnologyReadiness.COMMERCIAL, abatement_potential_pct=Decimal("70"), abatement_cost_usd_per_tco2=Decimal("30"), deployment_year_earliest=2024, ramp_up_years=4),
            Technology(technology_id="elec_pfc_abate", name="PFC Abatement Systems", lever=DecarbonizationLever.PROCESS_CHANGE, readiness=TechnologyReadiness.COMMERCIAL, abatement_potential_pct=Decimal("95"), abatement_cost_usd_per_tco2=Decimal("50"), deployment_year_earliest=2024, ramp_up_years=2),
        ]
    def generate_pathway(self, input_data: DecarbonizationInput) -> DecarbonizationOutput:
        calc_id = self._generate_calculation_id(input_data.facility_id)
        trajectory = self._calculate_trajectory(input_data.baseline_emissions_tco2e, input_data.baseline_year, input_data.target_year, self._technologies)
        pathway = DecarbonizationPathway(pathway_id=f"pathway_{calc_id}", name=f"Electronics Decarbonization", baseline_emissions_tco2e=input_data.baseline_emissions_tco2e, annual_trajectory=trajectory, technologies=self._technologies, target_reduction_pct=input_data.target_reduction_pct)
        return DecarbonizationOutput(calculation_id=calc_id, agent_id=self.AGENT_ID, agent_version=self.AGENT_VERSION, timestamp=self._get_timestamp(), facility_id=input_data.facility_id, sector=self.SECTOR, baseline_emissions_tco2e=input_data.baseline_emissions_tco2e, recommended_pathway=pathway, is_valid=True)


class AutomotiveDecarbonizationAgent(IndustrialDecarbonizationBaseAgent):
    """GL-DECARB-IND-010: Automotive Decarbonization Agent"""
    AGENT_ID = "GL-DECARB-IND-010"
    SECTOR = "Automotive"
    def _load_technologies(self) -> None:
        self._technologies = [
            Technology(technology_id="auto_renewable", name="Renewable Electricity", lever=DecarbonizationLever.RENEWABLE_ENERGY, readiness=TechnologyReadiness.COMMERCIAL, abatement_potential_pct=Decimal("50"), abatement_cost_usd_per_tco2=Decimal("25"), deployment_year_earliest=2024, ramp_up_years=3),
            Technology(technology_id="auto_paint", name="Low-emission Paint Shop", lever=DecarbonizationLever.PROCESS_CHANGE, readiness=TechnologyReadiness.COMMERCIAL, abatement_potential_pct=Decimal("30"), abatement_cost_usd_per_tco2=Decimal("40"), deployment_year_earliest=2025, ramp_up_years=4),
            Technology(technology_id="auto_efficiency", name="Energy Efficiency", lever=DecarbonizationLever.ENERGY_EFFICIENCY, readiness=TechnologyReadiness.COMMERCIAL, abatement_potential_pct=Decimal("15"), abatement_cost_usd_per_tco2=Decimal("-10"), deployment_year_earliest=2024, ramp_up_years=2),
        ]
    def generate_pathway(self, input_data: DecarbonizationInput) -> DecarbonizationOutput:
        calc_id = self._generate_calculation_id(input_data.facility_id)
        trajectory = self._calculate_trajectory(input_data.baseline_emissions_tco2e, input_data.baseline_year, input_data.target_year, self._technologies)
        pathway = DecarbonizationPathway(pathway_id=f"pathway_{calc_id}", name=f"Automotive Decarbonization", baseline_emissions_tco2e=input_data.baseline_emissions_tco2e, annual_trajectory=trajectory, technologies=self._technologies, target_reduction_pct=input_data.target_reduction_pct)
        return DecarbonizationOutput(calculation_id=calc_id, agent_id=self.AGENT_ID, agent_version=self.AGENT_VERSION, timestamp=self._get_timestamp(), facility_id=input_data.facility_id, sector=self.SECTOR, baseline_emissions_tco2e=input_data.baseline_emissions_tco2e, recommended_pathway=pathway, is_valid=True)


class TextilesDecarbonizationAgent(IndustrialDecarbonizationBaseAgent):
    """GL-DECARB-IND-011: Textiles Decarbonization Agent"""
    AGENT_ID = "GL-DECARB-IND-011"
    SECTOR = "Textiles"
    def _load_technologies(self) -> None:
        self._technologies = [
            Technology(technology_id="tex_heat_pump", name="Heat Pumps for Dyeing", lever=DecarbonizationLever.ELECTRIFICATION, readiness=TechnologyReadiness.COMMERCIAL, abatement_potential_pct=Decimal("50"), abatement_cost_usd_per_tco2=Decimal("35"), deployment_year_earliest=2024, ramp_up_years=3),
            Technology(technology_id="tex_solar", name="Solar Thermal", lever=DecarbonizationLever.RENEWABLE_ENERGY, readiness=TechnologyReadiness.COMMERCIAL, abatement_potential_pct=Decimal("25"), abatement_cost_usd_per_tco2=Decimal("20"), deployment_year_earliest=2024, ramp_up_years=2),
        ]
    def generate_pathway(self, input_data: DecarbonizationInput) -> DecarbonizationOutput:
        calc_id = self._generate_calculation_id(input_data.facility_id)
        trajectory = self._calculate_trajectory(input_data.baseline_emissions_tco2e, input_data.baseline_year, input_data.target_year, self._technologies)
        pathway = DecarbonizationPathway(pathway_id=f"pathway_{calc_id}", name=f"Textiles Decarbonization", baseline_emissions_tco2e=input_data.baseline_emissions_tco2e, annual_trajectory=trajectory, technologies=self._technologies, target_reduction_pct=input_data.target_reduction_pct)
        return DecarbonizationOutput(calculation_id=calc_id, agent_id=self.AGENT_ID, agent_version=self.AGENT_VERSION, timestamp=self._get_timestamp(), facility_id=input_data.facility_id, sector=self.SECTOR, baseline_emissions_tco2e=input_data.baseline_emissions_tco2e, recommended_pathway=pathway, is_valid=True)


class MiningDecarbonizationAgent(IndustrialDecarbonizationBaseAgent):
    """GL-DECARB-IND-012: Mining Decarbonization Agent"""
    AGENT_ID = "GL-DECARB-IND-012"
    SECTOR = "Mining"
    def _load_technologies(self) -> None:
        self._technologies = [
            Technology(technology_id="mine_electric", name="Electric Mining Equipment", lever=DecarbonizationLever.ELECTRIFICATION, readiness=TechnologyReadiness.EARLY_ADOPTION, abatement_potential_pct=Decimal("60"), abatement_cost_usd_per_tco2=Decimal("50"), deployment_year_earliest=2025, ramp_up_years=5),
            Technology(technology_id="mine_trolley", name="Trolley Assist Systems", lever=DecarbonizationLever.ELECTRIFICATION, readiness=TechnologyReadiness.COMMERCIAL, abatement_potential_pct=Decimal("30"), abatement_cost_usd_per_tco2=Decimal("25"), deployment_year_earliest=2024, ramp_up_years=3),
            Technology(technology_id="mine_renewable", name="On-site Renewable Generation", lever=DecarbonizationLever.RENEWABLE_ENERGY, readiness=TechnologyReadiness.COMMERCIAL, abatement_potential_pct=Decimal("40"), abatement_cost_usd_per_tco2=Decimal("30"), deployment_year_earliest=2024, ramp_up_years=3),
        ]
    def generate_pathway(self, input_data: DecarbonizationInput) -> DecarbonizationOutput:
        calc_id = self._generate_calculation_id(input_data.facility_id)
        trajectory = self._calculate_trajectory(input_data.baseline_emissions_tco2e, input_data.baseline_year, input_data.target_year, self._technologies)
        pathway = DecarbonizationPathway(pathway_id=f"pathway_{calc_id}", name=f"Mining Decarbonization", baseline_emissions_tco2e=input_data.baseline_emissions_tco2e, annual_trajectory=trajectory, technologies=self._technologies, target_reduction_pct=input_data.target_reduction_pct)
        return DecarbonizationOutput(calculation_id=calc_id, agent_id=self.AGENT_ID, agent_version=self.AGENT_VERSION, timestamp=self._get_timestamp(), facility_id=input_data.facility_id, sector=self.SECTOR, baseline_emissions_tco2e=input_data.baseline_emissions_tco2e, recommended_pathway=pathway, is_valid=True)
