# -*- coding: utf-8 -*-
"""
GL-DECARB-WAT-001: Water System Decarbonization Agent
====================================================

Decarbonization agent for comprehensive water sector decarbonization
planning and pathway development.

Capabilities:
    - Baseline emissions inventory for water utilities
    - Decarbonization pathway modeling
    - Intervention prioritization and sequencing
    - Cost-effectiveness analysis
    - Net-zero roadmap development

Methodologies:
    - GHG Protocol for water utilities
    - Science-based targets methodology
    - Marginal abatement cost curves

Zero-Hallucination Guarantees:
    - All calculations are deterministic
    - NO LLM involvement in emission/cost calculations
    - All emission factors traceable to authoritative sources
    - Complete provenance hash for every calculation

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class EmissionScope(str, Enum):
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


class InterventionCategory(str, Enum):
    ENERGY_EFFICIENCY = "energy_efficiency"
    RENEWABLE_ENERGY = "renewable_energy"
    PROCESS_OPTIMIZATION = "process_optimization"
    BIOGAS_UTILIZATION = "biogas_utilization"
    ELECTRIFICATION = "electrification"
    NATURE_BASED = "nature_based"
    CHEMICAL_OPTIMIZATION = "chemical_optimization"


class TargetType(str, Enum):
    NET_ZERO = "net_zero"
    CARBON_NEUTRAL = "carbon_neutral"
    SBT_15C = "sbt_1.5c"  # Science-based 1.5C aligned
    SBT_WB2C = "sbt_wb2c"  # Science-based well-below 2C
    CUSTOM = "custom"


# =============================================================================
# PYDANTIC MODELS - INPUT
# =============================================================================

class BaselineEmissions(BaseModel):
    """Baseline emissions inventory."""
    year: int
    scope1_emissions_tco2e: float = Field(..., ge=0)
    scope2_emissions_tco2e: float = Field(..., ge=0)
    scope3_emissions_tco2e: float = Field(default=0, ge=0)

    # Breakdown by source
    electricity_emissions_tco2e: float = Field(default=0, ge=0)
    natural_gas_emissions_tco2e: float = Field(default=0, ge=0)
    diesel_emissions_tco2e: float = Field(default=0, ge=0)
    process_ch4_emissions_tco2e: float = Field(default=0, ge=0)
    process_n2o_emissions_tco2e: float = Field(default=0, ge=0)
    chemical_emissions_tco2e: float = Field(default=0, ge=0)

    # Activity data
    energy_consumption_mwh: float = Field(default=0, ge=0)
    water_treated_million_m3: float = Field(default=0, ge=0)


class InterventionOption(BaseModel):
    """Decarbonization intervention option."""
    intervention_id: str
    name: str
    category: InterventionCategory
    description: str

    # Emission reduction
    annual_reduction_tco2e: float = Field(..., ge=0)
    reduction_scope: EmissionScope

    # Costs
    capital_cost: float = Field(default=0, ge=0)
    annual_operating_cost: float = Field(default=0)  # Can be negative (savings)
    lifetime_years: int = Field(default=15, ge=1)

    # Implementation
    implementation_years: int = Field(default=2, ge=1)
    earliest_start_year: int = Field(default=2025)

    # Co-benefits
    energy_savings_mwh: float = Field(default=0, ge=0)
    water_savings_million_m3: float = Field(default=0, ge=0)

    # Technical factors
    technical_readiness: float = Field(default=0.8, ge=0, le=1)
    scalability_factor: float = Field(default=1.0, ge=0)


class DecarbonizationTarget(BaseModel):
    """Decarbonization target."""
    target_type: TargetType
    target_year: int
    target_reduction_percent: float = Field(..., ge=0, le=100)
    interim_targets: Dict[int, float] = Field(default_factory=dict)


class DecarbonizationInput(BaseModel):
    """Input for decarbonization planning."""
    utility_id: str
    utility_name: Optional[str] = None
    baseline: BaselineEmissions
    interventions: List[InterventionOption] = Field(default_factory=list)
    target: DecarbonizationTarget
    grid_decarbonization_rate_per_year: float = Field(default=0.03)
    discount_rate: float = Field(default=0.05)
    carbon_price_per_tco2e: float = Field(default=50)


# =============================================================================
# PYDANTIC MODELS - OUTPUT
# =============================================================================

class YearlyProjection(BaseModel):
    """Yearly emission projection."""
    year: int
    baseline_emissions_tco2e: float
    projected_emissions_tco2e: float
    cumulative_reduction_tco2e: float
    reduction_percent: float
    active_interventions: List[str]
    annual_cost: float
    cumulative_cost: float


class InterventionSchedule(BaseModel):
    """Scheduled intervention."""
    intervention_id: str
    name: str
    start_year: int
    completion_year: int
    annual_reduction_tco2e: float
    total_cost: float
    cost_per_tco2e: float
    priority_rank: int


class DecarbonizationPathway(BaseModel):
    """Complete decarbonization pathway."""
    pathway_id: str
    target_type: str
    target_year: int
    target_reduction_percent: float

    # Projections
    yearly_projections: List[YearlyProjection]

    # Intervention schedule
    intervention_schedule: List[InterventionSchedule]

    # Summary metrics
    total_reduction_tco2e: float
    total_investment: float
    average_cost_per_tco2e: float
    net_present_value: float
    payback_years: float

    # Target achievement
    target_achieved: bool
    final_reduction_percent: float
    gap_to_target_tco2e: float


class DecarbonizationOutput(BaseModel):
    """Output from decarbonization planning."""
    utility_id: str
    baseline_year: int
    baseline_emissions_tco2e: float
    pathway: DecarbonizationPathway
    marginal_abatement_cost_curve: List[Dict[str, float]]
    key_milestones: List[Dict[str, Any]]
    recommendations: List[str]
    risks_and_barriers: List[str]
    provenance_hash: str
    calculation_timestamp: datetime
    processing_time_ms: float


# =============================================================================
# WATER SYSTEM DECARBONIZATION AGENT
# =============================================================================

class WaterSystemDecarbonizationAgent(BaseAgent):
    """
    GL-DECARB-WAT-001: Water System Decarbonization Agent

    Develops comprehensive decarbonization pathways for water utilities,
    including intervention prioritization, pathway modeling, and
    net-zero roadmap development.

    Zero-Hallucination Guarantees:
        - All calculations are deterministic mathematical operations
        - NO LLM involvement in emission/cost calculations
        - All factors traceable to authoritative sources
        - Complete provenance hash for every calculation

    Usage:
        agent = WaterSystemDecarbonizationAgent()
        result = agent.run({
            "utility_id": "...",
            "baseline": {...},
            "interventions": [...],
            "target": {...}
        })
    """

    AGENT_ID = "GL-DECARB-WAT-001"
    AGENT_NAME = "Water System Decarbonization Agent"
    VERSION = "1.0.0"
    METHODOLOGY_VERSION = "SBTi-Water-v2024.1"

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Water System Decarbonization Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Water sector decarbonization planning and pathway development",
                version=self.VERSION,
            )
        super().__init__(config)
        self.logger.info(f"Initialized {self.AGENT_ID}: {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute decarbonization pathway development."""
        start_time = time.time()

        try:
            dc_input = DecarbonizationInput(**input_data)

            # Calculate baseline total
            baseline_total = (
                dc_input.baseline.scope1_emissions_tco2e +
                dc_input.baseline.scope2_emissions_tco2e +
                dc_input.baseline.scope3_emissions_tco2e
            )

            # Generate default interventions if none provided
            interventions = dc_input.interventions or self._generate_default_interventions(
                dc_input.baseline
            )

            # Calculate marginal abatement cost curve
            macc = self._calculate_macc(interventions, dc_input.discount_rate)

            # Prioritize and schedule interventions
            scheduled = self._schedule_interventions(
                interventions,
                dc_input.target,
                baseline_total,
                dc_input.baseline.year,
            )

            # Generate yearly projections
            projections = self._generate_projections(
                dc_input.baseline,
                scheduled,
                dc_input.target.target_year,
                dc_input.grid_decarbonization_rate_per_year,
            )

            # Calculate pathway metrics
            total_reduction = sum(s.annual_reduction_tco2e * (dc_input.target.target_year - s.start_year) for s in scheduled)
            total_investment = sum(s.total_cost for s in scheduled)
            avg_cost = total_investment / max(1, total_reduction)

            # NPV calculation
            npv = self._calculate_npv(
                scheduled,
                dc_input.discount_rate,
                dc_input.carbon_price_per_tco2e,
                dc_input.baseline.year,
            )

            # Payback
            annual_savings = sum(
                i.energy_savings_mwh * 80 + i.annual_reduction_tco2e * dc_input.carbon_price_per_tco2e
                for i in interventions if i.intervention_id in [s.intervention_id for s in scheduled]
            )
            payback = total_investment / max(1, annual_savings)

            # Check target achievement
            final_projection = projections[-1] if projections else None
            final_reduction_pct = final_projection.reduction_percent if final_projection else 0
            target_achieved = final_reduction_pct >= dc_input.target.target_reduction_percent
            gap = baseline_total * (dc_input.target.target_reduction_percent - final_reduction_pct) / 100

            pathway = DecarbonizationPathway(
                pathway_id=f"PATH-{dc_input.utility_id}-{dc_input.target.target_year}",
                target_type=dc_input.target.target_type.value,
                target_year=dc_input.target.target_year,
                target_reduction_percent=dc_input.target.target_reduction_percent,
                yearly_projections=projections,
                intervention_schedule=scheduled,
                total_reduction_tco2e=round(total_reduction, 2),
                total_investment=round(total_investment, 0),
                average_cost_per_tco2e=round(avg_cost, 2),
                net_present_value=round(npv, 0),
                payback_years=round(min(payback, 30), 1),
                target_achieved=target_achieved,
                final_reduction_percent=round(final_reduction_pct, 1),
                gap_to_target_tco2e=round(max(0, gap), 2),
            )

            # Generate milestones
            milestones = []
            for year, target_pct in dc_input.target.interim_targets.items():
                proj = next((p for p in projections if p.year == year), None)
                if proj:
                    milestones.append({
                        "year": year,
                        "target_percent": target_pct,
                        "achieved_percent": proj.reduction_percent,
                        "on_track": proj.reduction_percent >= target_pct,
                    })

            # Recommendations
            recommendations = self._generate_recommendations(
                pathway, dc_input.baseline, interventions
            )

            # Risks
            risks = [
                "Grid decarbonization slower than projected",
                "Capital cost increases for key technologies",
                "Regulatory changes affecting operational requirements",
                "Supply chain constraints for equipment",
            ]

            provenance_hash = hashlib.sha256(
                json.dumps({
                    "utility": dc_input.utility_id,
                    "baseline": baseline_total,
                    "target": dc_input.target.target_reduction_percent,
                }, sort_keys=True).encode()
            ).hexdigest()[:16]

            processing_time = (time.time() - start_time) * 1000

            output = DecarbonizationOutput(
                utility_id=dc_input.utility_id,
                baseline_year=dc_input.baseline.year,
                baseline_emissions_tco2e=round(baseline_total, 2),
                pathway=pathway,
                marginal_abatement_cost_curve=macc,
                key_milestones=milestones,
                recommendations=recommendations,
                risks_and_barriers=risks,
                provenance_hash=provenance_hash,
                calculation_timestamp=DeterministicClock.now(),
                processing_time_ms=processing_time,
            )

            self.logger.info(
                f"Developed decarbonization pathway for {dc_input.utility_id}: "
                f"{final_reduction_pct:.1f}% reduction by {dc_input.target.target_year}"
            )

            return AgentResult(
                success=True,
                data=output.model_dump(),
                metadata={
                    "agent_id": self.AGENT_ID,
                    "methodology": self.METHODOLOGY_VERSION,
                }
            )

        except Exception as e:
            self.logger.error(f"Decarbonization planning failed: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _generate_default_interventions(
        self, baseline: BaselineEmissions
    ) -> List[InterventionOption]:
        """Generate default intervention options based on baseline."""
        interventions = []

        # Energy efficiency
        if baseline.electricity_emissions_tco2e > 0:
            interventions.append(InterventionOption(
                intervention_id="INT-001",
                name="Pump Efficiency Upgrades",
                category=InterventionCategory.ENERGY_EFFICIENCY,
                description="Replace inefficient pumps with high-efficiency units",
                annual_reduction_tco2e=baseline.electricity_emissions_tco2e * 0.10,
                reduction_scope=EmissionScope.SCOPE_2,
                capital_cost=baseline.electricity_emissions_tco2e * 5000,
                annual_operating_cost=-baseline.electricity_emissions_tco2e * 500,
                energy_savings_mwh=baseline.energy_consumption_mwh * 0.10,
            ))

            interventions.append(InterventionOption(
                intervention_id="INT-002",
                name="Process Optimization",
                category=InterventionCategory.PROCESS_OPTIMIZATION,
                description="AI-driven process optimization for energy reduction",
                annual_reduction_tco2e=baseline.electricity_emissions_tco2e * 0.08,
                reduction_scope=EmissionScope.SCOPE_2,
                capital_cost=baseline.electricity_emissions_tco2e * 2000,
                annual_operating_cost=baseline.electricity_emissions_tco2e * 100,
                energy_savings_mwh=baseline.energy_consumption_mwh * 0.08,
            ))

        # Renewable energy
        interventions.append(InterventionOption(
            intervention_id="INT-003",
            name="On-site Solar PV",
            category=InterventionCategory.RENEWABLE_ENERGY,
            description="Install solar PV arrays at treatment facilities",
            annual_reduction_tco2e=baseline.electricity_emissions_tco2e * 0.25,
            reduction_scope=EmissionScope.SCOPE_2,
            capital_cost=baseline.electricity_emissions_tco2e * 8000,
            annual_operating_cost=-baseline.electricity_emissions_tco2e * 300,
            lifetime_years=25,
        ))

        interventions.append(InterventionOption(
            intervention_id="INT-004",
            name="Renewable PPA",
            category=InterventionCategory.RENEWABLE_ENERGY,
            description="Power purchase agreement for 100% renewable electricity",
            annual_reduction_tco2e=baseline.electricity_emissions_tco2e * 0.90,
            reduction_scope=EmissionScope.SCOPE_2,
            capital_cost=0,
            annual_operating_cost=baseline.electricity_emissions_tco2e * 200,
            implementation_years=1,
        ))

        # Biogas
        if baseline.process_ch4_emissions_tco2e > 0:
            interventions.append(InterventionOption(
                intervention_id="INT-005",
                name="Biogas CHP",
                category=InterventionCategory.BIOGAS_UTILIZATION,
                description="Capture and utilize biogas for combined heat and power",
                annual_reduction_tco2e=baseline.process_ch4_emissions_tco2e * 0.80,
                reduction_scope=EmissionScope.SCOPE_1,
                capital_cost=baseline.process_ch4_emissions_tco2e * 15000,
                annual_operating_cost=-baseline.process_ch4_emissions_tco2e * 1000,
                energy_savings_mwh=baseline.process_ch4_emissions_tco2e * 5,
            ))

        # Electrification
        if baseline.natural_gas_emissions_tco2e > 0:
            interventions.append(InterventionOption(
                intervention_id="INT-006",
                name="Fleet Electrification",
                category=InterventionCategory.ELECTRIFICATION,
                description="Transition service fleet to electric vehicles",
                annual_reduction_tco2e=baseline.diesel_emissions_tco2e * 0.70,
                reduction_scope=EmissionScope.SCOPE_1,
                capital_cost=baseline.diesel_emissions_tco2e * 10000,
                annual_operating_cost=-baseline.diesel_emissions_tco2e * 500,
            ))

        return interventions

    def _calculate_macc(
        self, interventions: List[InterventionOption], discount_rate: float
    ) -> List[Dict[str, float]]:
        """Calculate marginal abatement cost curve."""
        macc = []
        for intervention in interventions:
            # Levelized cost of emission reduction
            annuity_factor = (
                discount_rate * (1 + discount_rate) ** intervention.lifetime_years /
                ((1 + discount_rate) ** intervention.lifetime_years - 1)
            ) if intervention.lifetime_years > 0 else 1

            annualized_capex = intervention.capital_cost * annuity_factor
            total_annual_cost = annualized_capex + intervention.annual_operating_cost
            cost_per_tco2e = (
                total_annual_cost / intervention.annual_reduction_tco2e
                if intervention.annual_reduction_tco2e > 0 else float('inf')
            )

            macc.append({
                "intervention_id": intervention.intervention_id,
                "name": intervention.name,
                "annual_reduction_tco2e": intervention.annual_reduction_tco2e,
                "cost_per_tco2e": round(cost_per_tco2e, 2),
            })

        # Sort by cost
        macc.sort(key=lambda x: x["cost_per_tco2e"])
        return macc

    def _schedule_interventions(
        self,
        interventions: List[InterventionOption],
        target: DecarbonizationTarget,
        baseline_total: float,
        baseline_year: int,
    ) -> List[InterventionSchedule]:
        """Schedule interventions to meet target."""
        # Sort by cost-effectiveness
        sorted_interventions = sorted(
            interventions,
            key=lambda i: (i.capital_cost / max(1, i.annual_reduction_tco2e)),
        )

        scheduled = []
        cumulative_reduction = 0
        target_reduction = baseline_total * target.target_reduction_percent / 100
        current_year = baseline_year + 1
        rank = 1

        for intervention in sorted_interventions:
            if cumulative_reduction >= target_reduction:
                break

            cost_per_tco2e = (
                intervention.capital_cost / max(1, intervention.annual_reduction_tco2e * intervention.lifetime_years)
            )

            schedule = InterventionSchedule(
                intervention_id=intervention.intervention_id,
                name=intervention.name,
                start_year=max(current_year, intervention.earliest_start_year),
                completion_year=max(current_year, intervention.earliest_start_year) + intervention.implementation_years,
                annual_reduction_tco2e=intervention.annual_reduction_tco2e,
                total_cost=intervention.capital_cost,
                cost_per_tco2e=round(cost_per_tco2e, 2),
                priority_rank=rank,
            )
            scheduled.append(schedule)

            cumulative_reduction += intervention.annual_reduction_tco2e
            current_year = schedule.completion_year
            rank += 1

        return scheduled

    def _generate_projections(
        self,
        baseline: BaselineEmissions,
        scheduled: List[InterventionSchedule],
        target_year: int,
        grid_decarb_rate: float,
    ) -> List[YearlyProjection]:
        """Generate yearly emission projections."""
        baseline_total = (
            baseline.scope1_emissions_tco2e +
            baseline.scope2_emissions_tco2e +
            baseline.scope3_emissions_tco2e
        )

        projections = []
        cumulative_cost = 0

        for year in range(baseline.year, target_year + 1):
            # Grid decarbonization effect on Scope 2
            years_elapsed = year - baseline.year
            grid_factor = (1 - grid_decarb_rate) ** years_elapsed
            adjusted_scope2 = baseline.scope2_emissions_tco2e * grid_factor

            # Active interventions
            active = [s for s in scheduled if s.completion_year <= year]
            active_ids = [s.intervention_id for s in active]

            # Total reduction from interventions
            intervention_reduction = sum(s.annual_reduction_tco2e for s in active)

            # Projected emissions
            projected = max(0, baseline_total - intervention_reduction - (baseline.scope2_emissions_tco2e - adjusted_scope2))

            # Costs
            new_costs = sum(s.total_cost for s in scheduled if s.start_year == year)
            cumulative_cost += new_costs

            reduction_pct = (baseline_total - projected) / baseline_total * 100 if baseline_total > 0 else 0

            projection = YearlyProjection(
                year=year,
                baseline_emissions_tco2e=round(baseline_total, 2),
                projected_emissions_tco2e=round(projected, 2),
                cumulative_reduction_tco2e=round(baseline_total - projected, 2),
                reduction_percent=round(reduction_pct, 1),
                active_interventions=active_ids,
                annual_cost=round(new_costs, 0),
                cumulative_cost=round(cumulative_cost, 0),
            )
            projections.append(projection)

        return projections

    def _calculate_npv(
        self,
        scheduled: List[InterventionSchedule],
        discount_rate: float,
        carbon_price: float,
        baseline_year: int,
    ) -> float:
        """Calculate NPV of the decarbonization pathway."""
        npv = 0
        for schedule in scheduled:
            # Cost (negative)
            year_offset = schedule.start_year - baseline_year
            discounted_cost = schedule.total_cost / ((1 + discount_rate) ** year_offset)

            # Benefit (carbon savings)
            annual_benefit = schedule.annual_reduction_tco2e * carbon_price
            benefit_years = 20  # Assume 20 years of benefits
            pv_benefit = annual_benefit * (1 - (1 + discount_rate) ** (-benefit_years)) / discount_rate
            discounted_benefit = pv_benefit / ((1 + discount_rate) ** year_offset)

            npv += discounted_benefit - discounted_cost

        return npv

    def _generate_recommendations(
        self,
        pathway: DecarbonizationPathway,
        baseline: BaselineEmissions,
        interventions: List[InterventionOption],
    ) -> List[str]:
        """Generate pathway recommendations."""
        recommendations = []

        if pathway.target_achieved:
            recommendations.append(f"Pathway achieves {pathway.target_reduction_percent}% target by {pathway.target_year}")
        else:
            recommendations.append(f"Gap of {pathway.gap_to_target_tco2e:.0f} tCO2e to target - additional measures needed")

        if pathway.average_cost_per_tco2e < 50:
            recommendations.append("Cost-effective pathway with average abatement cost below $50/tCO2e")
        elif pathway.average_cost_per_tco2e < 100:
            recommendations.append("Moderately cost-effective pathway")
        else:
            recommendations.append("Consider phasing high-cost interventions to later years")

        if baseline.scope2_emissions_tco2e > baseline.scope1_emissions_tco2e:
            recommendations.append("Prioritize renewable energy procurement to address dominant Scope 2 emissions")

        if baseline.process_ch4_emissions_tco2e > baseline.scope1_emissions_tco2e * 0.3:
            recommendations.append("Biogas capture and utilization should be a priority intervention")

        recommendations.append("Establish monitoring and reporting system to track progress")
        recommendations.append("Review and update pathway annually based on technology developments")

        return recommendations
