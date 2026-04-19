# -*- coding: utf-8 -*-
"""
GL-DECARB-BLD-001: Building Energy Efficiency Agent
=====================================================

Comprehensive building energy efficiency and retrofit planning agent.
Analyzes buildings and recommends efficiency improvements with financial analysis.

Features:
    - Energy audit analysis
    - Retrofit measure prioritization
    - Financial payback calculation
    - Energy modeling integration
    - Incentive program matching

Standards:
    - ASHRAE Level II Energy Audit
    - DOE Building Energy Asset Score
    - IPMVP measurement and verification

Author: GreenLang Framework Team
Agent ID: GL-DECARB-BLD-001
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
    FinancialMetrics,
    BuildingBaseline,
    DecarbonizationTarget,
    ENVELOPE_SAVINGS,
    LED_SAVINGS_PERCENT,
)

logger = logging.getLogger(__name__)


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class EnergyEfficiencyInput(DecarbonizationInput):
    """Input model for energy efficiency analysis."""

    # Building envelope
    window_type: Optional[str] = Field(None, description="single, double, triple")
    wall_insulation_r_value: Optional[Decimal] = Field(None, ge=0)
    roof_insulation_r_value: Optional[Decimal] = Field(None, ge=0)
    air_leakage_cfm50: Optional[Decimal] = Field(None, ge=0)

    # HVAC
    hvac_system_age_years: Optional[int] = Field(None, ge=0)
    hvac_efficiency_percent: Optional[Decimal] = Field(None, ge=0, le=100)
    has_economizer: bool = Field(default=False)
    has_demand_ventilation: bool = Field(default=False)

    # Lighting
    led_percentage: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    has_lighting_controls: bool = Field(default=False)

    # Controls
    has_bms: bool = Field(default=False)
    has_smart_thermostat: bool = Field(default=False)

    # Energy audit data
    audit_level: Optional[str] = Field(None, description="walkthrough, level1, level2, level3")


class EnergyEfficiencyOutput(DecarbonizationOutput):
    """Output model for energy efficiency analysis."""

    # Retrofit recommendations by category
    envelope_measures: List[DecarbonizationMeasure] = Field(default_factory=list)
    hvac_measures: List[DecarbonizationMeasure] = Field(default_factory=list)
    lighting_measures: List[DecarbonizationMeasure] = Field(default_factory=list)
    controls_measures: List[DecarbonizationMeasure] = Field(default_factory=list)

    # Summary by category
    envelope_savings_kgco2e: Decimal = Field(default=Decimal("0"))
    hvac_savings_kgco2e: Decimal = Field(default=Decimal("0"))
    lighting_savings_kgco2e: Decimal = Field(default=Decimal("0"))
    controls_savings_kgco2e: Decimal = Field(default=Decimal("0"))

    # Benchmarking
    current_energy_star_score: Optional[int] = None
    projected_energy_star_score: Optional[int] = None


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class BuildingEnergyEfficiencyAgent(BuildingDecarbonizationBaseAgent[EnergyEfficiencyInput, EnergyEfficiencyOutput]):
    """
    GL-DECARB-BLD-001: Building Energy Efficiency Agent.

    Analyzes buildings for energy efficiency improvements and recommends
    retrofit measures with financial analysis and implementation planning.

    Example:
        >>> agent = BuildingEnergyEfficiencyAgent()
        >>> input_data = EnergyEfficiencyInput(
        ...     building_baseline=BuildingBaseline(
        ...         building_id="BLDG-001",
        ...         building_type="commercial_office",
        ...         gross_floor_area_sqm=Decimal("5000"),
        ...         current_energy_kwh_per_year=Decimal("890000"),
        ...         current_emissions_kgco2e_per_year=Decimal("337420")
        ...     ),
        ...     target=DecarbonizationTarget(target_year=2030, target_reduction_percent=Decimal("50")),
        ...     hvac_system_age_years=15,
        ...     led_percentage=Decimal("20")
        ... )
        >>> output = agent.process(input_data)
    """

    AGENT_ID = "GL-DECARB-BLD-001"
    AGENT_VERSION = "1.0.0"
    TECHNOLOGY_FOCUS = TechnologyCategory.ENVELOPE

    def _load_technology_database(self) -> None:
        """Load energy efficiency technology specifications."""
        # Window upgrades
        self._technology_database["window_upgrade_double"] = TechnologySpec(
            technology_id="window_upgrade_double",
            category=TechnologyCategory.ENVELOPE,
            name="Double-pane Window Upgrade",
            description="Replace single-pane windows with double-pane low-e",
            efficiency_improvement_percent=Decimal("15"),
            lifespan_years=25
        )

        self._technology_database["window_upgrade_triple"] = TechnologySpec(
            technology_id="window_upgrade_triple",
            category=TechnologyCategory.ENVELOPE,
            name="Triple-pane Window Upgrade",
            description="Install high-performance triple-pane windows",
            efficiency_improvement_percent=Decimal("25"),
            lifespan_years=30
        )

        # Insulation
        self._technology_database["wall_insulation"] = TechnologySpec(
            technology_id="wall_insulation",
            category=TechnologyCategory.ENVELOPE,
            name="Wall Insulation Upgrade",
            description="Add or upgrade wall insulation to R-19+",
            efficiency_improvement_percent=Decimal("15"),
            lifespan_years=40
        )

        self._technology_database["roof_insulation"] = TechnologySpec(
            technology_id="roof_insulation",
            category=TechnologyCategory.ENVELOPE,
            name="Roof Insulation Upgrade",
            description="Upgrade roof insulation to R-38+",
            efficiency_improvement_percent=Decimal("10"),
            lifespan_years=40
        )

        # HVAC
        self._technology_database["hvac_upgrade"] = TechnologySpec(
            technology_id="hvac_upgrade",
            category=TechnologyCategory.HVAC,
            name="High-Efficiency HVAC",
            description="Replace aging HVAC with high-efficiency system",
            efficiency_improvement_percent=Decimal("30"),
            lifespan_years=20
        )

        self._technology_database["economizer"] = TechnologySpec(
            technology_id="economizer",
            category=TechnologyCategory.HVAC,
            name="Airside Economizer",
            description="Install economizer for free cooling",
            efficiency_improvement_percent=Decimal("10"),
            lifespan_years=15
        )

        # Lighting
        self._technology_database["led_retrofit"] = TechnologySpec(
            technology_id="led_retrofit",
            category=TechnologyCategory.LIGHTING,
            name="LED Lighting Retrofit",
            description="Convert all lighting to LED",
            efficiency_improvement_percent=Decimal("50"),
            lifespan_years=15
        )

        self._technology_database["lighting_controls"] = TechnologySpec(
            technology_id="lighting_controls",
            category=TechnologyCategory.CONTROLS,
            name="Advanced Lighting Controls",
            description="Install occupancy and daylight sensors",
            efficiency_improvement_percent=Decimal("25"),
            lifespan_years=12
        )

        # Controls
        self._technology_database["bms_upgrade"] = TechnologySpec(
            technology_id="bms_upgrade",
            category=TechnologyCategory.CONTROLS,
            name="Building Management System",
            description="Install or upgrade BMS with optimization",
            efficiency_improvement_percent=Decimal("15"),
            lifespan_years=15
        )

    def analyze(
        self,
        input_data: EnergyEfficiencyInput
    ) -> EnergyEfficiencyOutput:
        """
        Analyze building and recommend energy efficiency measures.

        Methodology:
        1. Analyze building envelope opportunities
        2. Evaluate HVAC upgrade potential
        3. Calculate lighting retrofit savings
        4. Assess controls improvements
        5. Prioritize and create pathway

        Args:
            input_data: Building baseline and efficiency details

        Returns:
            Energy efficiency recommendations with financial analysis
        """
        baseline = input_data.building_baseline
        target = input_data.target

        envelope_measures: List[DecarbonizationMeasure] = []
        hvac_measures: List[DecarbonizationMeasure] = []
        lighting_measures: List[DecarbonizationMeasure] = []
        controls_measures: List[DecarbonizationMeasure] = []

        grid_ef = Decimal("0.379")  # US average kgCO2e/kWh
        floor_area = baseline.gross_floor_area_sqm

        # Step 1: Analyze envelope
        if input_data.window_type in [None, "single"]:
            window_savings_percent = ENVELOPE_SAVINGS["window_upgrade"]
            window_savings_kwh = baseline.current_energy_kwh_per_year * (window_savings_percent / 100)
            window_savings_co2 = window_savings_kwh * grid_ef
            window_cost = floor_area * Decimal("40")  # $40/sqm typical

            envelope_measures.append(self._create_measure(
                measure_id="ENV-001",
                name="Window Upgrade",
                description="Replace single-pane with double-pane low-e windows",
                technology=self._technology_database["window_upgrade_double"],
                capital_cost=window_cost,
                annual_savings=window_savings_kwh * input_data.electricity_cost_per_kwh,
                energy_savings_kwh=window_savings_kwh,
                emission_reduction=window_savings_co2,
                priority=RecommendationPriority.MEDIUM,
                phase=ImplementationPhase.MEDIUM_TERM,
                discount_rate=input_data.discount_rate_percent
            ))

        # Wall insulation
        if input_data.wall_insulation_r_value and input_data.wall_insulation_r_value < 19:
            insulation_savings_percent = ENVELOPE_SAVINGS["insulation_upgrade"]
            insulation_savings_kwh = baseline.current_energy_kwh_per_year * (insulation_savings_percent / 100)
            insulation_savings_co2 = insulation_savings_kwh * grid_ef
            insulation_cost = floor_area * Decimal("25")

            envelope_measures.append(self._create_measure(
                measure_id="ENV-002",
                name="Wall Insulation Upgrade",
                description="Upgrade wall insulation to R-19 or higher",
                technology=self._technology_database["wall_insulation"],
                capital_cost=insulation_cost,
                annual_savings=insulation_savings_kwh * input_data.electricity_cost_per_kwh,
                energy_savings_kwh=insulation_savings_kwh,
                emission_reduction=insulation_savings_co2,
                priority=RecommendationPriority.MEDIUM,
                phase=ImplementationPhase.MEDIUM_TERM,
                discount_rate=input_data.discount_rate_percent
            ))

        # Step 2: Analyze HVAC
        if input_data.hvac_system_age_years and input_data.hvac_system_age_years > 15:
            hvac_savings_percent = Decimal("30")
            hvac_savings_kwh = baseline.current_energy_kwh_per_year * Decimal("0.40") * (hvac_savings_percent / 100)
            hvac_savings_co2 = hvac_savings_kwh * grid_ef
            hvac_cost = floor_area * Decimal("120")

            hvac_measures.append(self._create_measure(
                measure_id="HVAC-001",
                name="HVAC System Replacement",
                description="Replace aging HVAC with high-efficiency system",
                technology=self._technology_database["hvac_upgrade"],
                capital_cost=hvac_cost,
                annual_savings=hvac_savings_kwh * input_data.electricity_cost_per_kwh,
                energy_savings_kwh=hvac_savings_kwh,
                emission_reduction=hvac_savings_co2,
                priority=RecommendationPriority.HIGH,
                phase=ImplementationPhase.SHORT_TERM,
                discount_rate=input_data.discount_rate_percent
            ))

        if not input_data.has_economizer:
            econ_savings_kwh = baseline.current_energy_kwh_per_year * Decimal("0.30") * Decimal("0.10")
            econ_savings_co2 = econ_savings_kwh * grid_ef
            econ_cost = Decimal("15000")

            hvac_measures.append(self._create_measure(
                measure_id="HVAC-002",
                name="Airside Economizer",
                description="Add economizer for free cooling when outdoor conditions allow",
                technology=self._technology_database["economizer"],
                capital_cost=econ_cost,
                annual_savings=econ_savings_kwh * input_data.electricity_cost_per_kwh,
                energy_savings_kwh=econ_savings_kwh,
                emission_reduction=econ_savings_co2,
                priority=RecommendationPriority.MEDIUM,
                phase=ImplementationPhase.SHORT_TERM,
                discount_rate=input_data.discount_rate_percent
            ))

        # Step 3: Analyze lighting
        if input_data.led_percentage < 100:
            non_led_fraction = (100 - input_data.led_percentage) / 100
            lighting_energy = baseline.current_energy_kwh_per_year * Decimal("0.25")
            led_savings_kwh = lighting_energy * non_led_fraction * LED_SAVINGS_PERCENT / 100
            led_savings_co2 = led_savings_kwh * grid_ef
            led_cost = floor_area * Decimal("15") * non_led_fraction

            lighting_measures.append(self._create_measure(
                measure_id="LTG-001",
                name="LED Lighting Retrofit",
                description="Convert remaining lighting to LED",
                technology=self._technology_database["led_retrofit"],
                capital_cost=led_cost,
                annual_savings=led_savings_kwh * input_data.electricity_cost_per_kwh,
                energy_savings_kwh=led_savings_kwh,
                emission_reduction=led_savings_co2,
                priority=RecommendationPriority.HIGH,
                phase=ImplementationPhase.IMMEDIATE,
                discount_rate=input_data.discount_rate_percent
            ))

        if not input_data.has_lighting_controls:
            controls_savings_kwh = baseline.current_energy_kwh_per_year * Decimal("0.25") * Decimal("0.25")
            controls_savings_co2 = controls_savings_kwh * grid_ef
            controls_cost = floor_area * Decimal("8")

            lighting_measures.append(self._create_measure(
                measure_id="LTG-002",
                name="Lighting Controls",
                description="Install occupancy sensors and daylight harvesting",
                technology=self._technology_database["lighting_controls"],
                capital_cost=controls_cost,
                annual_savings=controls_savings_kwh * input_data.electricity_cost_per_kwh,
                energy_savings_kwh=controls_savings_kwh,
                emission_reduction=controls_savings_co2,
                priority=RecommendationPriority.MEDIUM,
                phase=ImplementationPhase.IMMEDIATE,
                discount_rate=input_data.discount_rate_percent
            ))

        # Step 4: Analyze controls
        if not input_data.has_bms:
            bms_savings_kwh = baseline.current_energy_kwh_per_year * Decimal("0.15")
            bms_savings_co2 = bms_savings_kwh * grid_ef
            bms_cost = floor_area * Decimal("30")

            controls_measures.append(self._create_measure(
                measure_id="CTRL-001",
                name="Building Management System",
                description="Install BMS with optimization algorithms",
                technology=self._technology_database["bms_upgrade"],
                capital_cost=bms_cost,
                annual_savings=bms_savings_kwh * input_data.electricity_cost_per_kwh,
                energy_savings_kwh=bms_savings_kwh,
                emission_reduction=bms_savings_co2,
                priority=RecommendationPriority.HIGH,
                phase=ImplementationPhase.SHORT_TERM,
                discount_rate=input_data.discount_rate_percent
            ))

        # Calculate totals
        all_measures = envelope_measures + hvac_measures + lighting_measures + controls_measures

        total_reduction = sum(m.annual_emission_reduction_kgco2e for m in all_measures)
        total_investment = sum(m.financial.capital_cost_usd for m in all_measures)
        total_savings = sum(m.financial.annual_savings_usd for m in all_measures)

        envelope_savings = sum(m.annual_emission_reduction_kgco2e for m in envelope_measures)
        hvac_savings = sum(m.annual_emission_reduction_kgco2e for m in hvac_measures)
        lighting_savings = sum(m.annual_emission_reduction_kgco2e for m in lighting_measures)
        controls_savings = sum(m.annual_emission_reduction_kgco2e for m in controls_measures)

        # Create pathway
        pathway = DecarbonizationPathway(
            pathway_id=self._generate_analysis_id(baseline.building_id),
            name="Energy Efficiency Pathway",
            description="Comprehensive energy efficiency retrofit pathway",
            target_year=target.target_year,
            target_reduction_percent=target.target_reduction_percent or Decimal("0"),
            immediate_measures=[m for m in all_measures if m.phase == ImplementationPhase.IMMEDIATE],
            short_term_measures=[m for m in all_measures if m.phase == ImplementationPhase.SHORT_TERM],
            medium_term_measures=[m for m in all_measures if m.phase == ImplementationPhase.MEDIUM_TERM],
            long_term_measures=[m for m in all_measures if m.phase == ImplementationPhase.LONG_TERM],
            total_capital_cost_usd=total_investment,
            total_annual_savings_usd=total_savings,
            total_emission_reduction_kgco2e=total_reduction
        )

        # Check if target achievable
        target_emissions = baseline.current_emissions_kgco2e_per_year * (
            1 - (target.target_reduction_percent or Decimal("0")) / 100
        )
        gap = max(Decimal("0"), baseline.current_emissions_kgco2e_per_year - total_reduction - target_emissions)
        target_achievable = gap <= 0

        avg_payback = None
        if total_savings > 0:
            avg_payback = self._round_financial(total_investment / total_savings, 1)

        return EnergyEfficiencyOutput(
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
            average_payback_years=avg_payback,
            target_achievable=target_achievable,
            gap_to_target_kgco2e=self._round_emissions(gap),
            envelope_measures=envelope_measures,
            hvac_measures=hvac_measures,
            lighting_measures=lighting_measures,
            controls_measures=controls_measures,
            envelope_savings_kgco2e=self._round_emissions(envelope_savings),
            hvac_savings_kgco2e=self._round_emissions(hvac_savings),
            lighting_savings_kgco2e=self._round_emissions(lighting_savings),
            controls_savings_kgco2e=self._round_emissions(controls_savings),
            is_valid=True
        )
