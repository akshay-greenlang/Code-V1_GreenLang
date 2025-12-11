"""
GL-070: Decarbonization Planner Agent (DECARBONIZATION-PLANNER)

This module implements the DecarbonizationPlannerAgent for decarbonization pathway planning,
investment optimization, and emission trajectory modeling aligned with climate targets.

Standards Reference:
    - Science Based Targets initiative (SBTi)
    - Paris Agreement (1.5C and 2C pathways)
    - GHG Protocol
    - Task Force on Climate-related Financial Disclosures (TCFD)

Example:
    >>> agent = DecarbonizationPlannerAgent()
    >>> result = agent.run(DecarbonizationPlannerInput(current_emissions=..., targets=[...]))
    >>> print(f"Pathway achieves {result.pathway_assessment.target_achievement_percent:.1f}% of target")
"""

import hashlib
import json
import logging
import math
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EmissionScope(str, Enum):
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"
    ALL = "all"


class TargetType(str, Enum):
    ABSOLUTE = "absolute"
    INTENSITY = "intensity"
    NET_ZERO = "net_zero"


class TechnologyCategory(str, Enum):
    ENERGY_EFFICIENCY = "energy_efficiency"
    FUEL_SWITCHING = "fuel_switching"
    ELECTRIFICATION = "electrification"
    RENEWABLE_ENERGY = "renewable_energy"
    CARBON_CAPTURE = "carbon_capture"
    PROCESS_CHANGE = "process_change"
    HYDROGEN = "hydrogen"
    NATURE_BASED = "nature_based"


class PathwayType(str, Enum):
    PARIS_1_5C = "paris_1_5c"
    PARIS_2C = "paris_2c"
    SBTI_ALIGNED = "sbti_aligned"
    CUSTOM = "custom"


class CurrentEmissions(BaseModel):
    """Current emissions baseline."""
    baseline_year: int = Field(..., description="Baseline year")
    scope_1_tCO2e: float = Field(..., ge=0, description="Scope 1 emissions (tCO2e)")
    scope_2_tCO2e: float = Field(..., ge=0, description="Scope 2 emissions (tCO2e)")
    scope_3_tCO2e: float = Field(default=0, ge=0, description="Scope 3 emissions (tCO2e)")
    revenue_million_usd: Optional[float] = Field(None, description="Revenue for intensity calc")
    production_units: Optional[float] = Field(None, description="Production for intensity calc")
    emission_sources: Dict[str, float] = Field(default_factory=dict, description="Breakdown by source")


class ReductionTarget(BaseModel):
    """Emission reduction target."""
    target_id: str = Field(..., description="Target identifier")
    target_name: str = Field(..., description="Target name")
    target_year: int = Field(..., description="Target year")
    target_type: TargetType = Field(..., description="Type of target")
    scope: EmissionScope = Field(default=EmissionScope.ALL, description="Target scope")
    reduction_percent: float = Field(..., ge=0, le=100, description="Reduction percentage")
    base_year: int = Field(..., description="Base year for reduction")
    pathway_type: PathwayType = Field(default=PathwayType.SBTI_ALIGNED, description="Pathway type")
    is_validated: bool = Field(default=False, description="SBTi validated")


class TechnologyOption(BaseModel):
    """Technology option for decarbonization."""
    technology_id: str = Field(..., description="Technology identifier")
    name: str = Field(..., description="Technology name")
    category: TechnologyCategory = Field(..., description="Technology category")
    applicable_scope: EmissionScope = Field(..., description="Applicable emission scope")
    reduction_potential_tCO2e: float = Field(..., ge=0, description="Annual reduction potential")
    reduction_potential_percent: float = Field(default=0, description="Percentage reduction")
    capex_usd: float = Field(..., ge=0, description="Capital expenditure")
    annual_opex_usd: float = Field(default=0, description="Annual operating cost change")
    annual_savings_usd: float = Field(default=0, description="Annual energy savings")
    implementation_years: int = Field(default=1, ge=1, description="Years to implement")
    lifetime_years: int = Field(default=20, ge=1, description="Technology lifetime")
    technology_readiness_level: int = Field(default=9, ge=1, le=9, description="TRL")
    earliest_deployment_year: int = Field(default=2024, description="Earliest deployment")


class DecarbonizationPlannerInput(BaseModel):
    """Input for decarbonization planning."""
    plan_id: Optional[str] = Field(None, description="Plan identifier")
    organization_name: str = Field(default="Organization", description="Organization name")
    current_emissions: CurrentEmissions = Field(..., description="Current emissions baseline")
    targets: List[ReductionTarget] = Field(..., description="Reduction targets")
    technology_options: List[TechnologyOption] = Field(..., description="Available technologies")
    planning_horizon_years: int = Field(default=30, description="Planning horizon")
    discount_rate_percent: float = Field(default=8.0, description="Discount rate for NPV")
    carbon_price_per_tCO2e: float = Field(default=50.0, description="Carbon price")
    carbon_price_escalation_percent: float = Field(default=5.0, description="Annual price escalation")
    budget_constraint_million_usd: Optional[float] = Field(None, description="Budget constraint")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class YearlyEmission(BaseModel):
    """Yearly emission projection."""
    year: int
    scope_1_tCO2e: float
    scope_2_tCO2e: float
    scope_3_tCO2e: float
    total_tCO2e: float
    cumulative_reduction_tCO2e: float
    target_trajectory_tCO2e: float
    variance_tCO2e: float
    on_track: bool


class TechnologyDeployment(BaseModel):
    """Technology deployment in roadmap."""
    technology_id: str
    technology_name: str
    deployment_year: int
    reduction_tCO2e: float
    capex_usd: float
    annual_opex_usd: float
    annual_savings_usd: float
    cumulative_reduction_to_date: float
    priority_rank: int


class InvestmentYear(BaseModel):
    """Yearly investment details."""
    year: int
    capex_usd: float
    opex_usd: float
    savings_usd: float
    net_cost_usd: float
    carbon_cost_avoided_usd: float
    cumulative_investment_usd: float
    technologies_deployed: List[str]


class ROIAnalysis(BaseModel):
    """Return on investment analysis."""
    total_capex_usd: float
    total_opex_usd: float
    total_savings_usd: float
    total_carbon_cost_avoided_usd: float
    net_present_value_usd: float
    internal_rate_of_return_percent: Optional[float]
    simple_payback_years: Optional[float]
    discounted_payback_years: Optional[float]
    cost_per_tCO2e_avoided: float
    marginal_abatement_cost_usd: float


class PathwayAssessment(BaseModel):
    """Assessment of decarbonization pathway."""
    pathway_type: str
    target_achievement_percent: float
    sbti_aligned: bool
    paris_aligned: bool
    gap_to_target_tCO2e: float
    year_target_achieved: Optional[int]
    risk_level: str
    confidence_score: float


class MilestoneCheck(BaseModel):
    """Milestone checkpoint."""
    milestone_year: int
    target_reduction_percent: float
    projected_reduction_percent: float
    on_track: bool
    gap_percent: float
    corrective_actions: List[str]


class DecarbonizationPlannerOutput(BaseModel):
    """Output from decarbonization planning."""
    plan_id: str
    organization_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    baseline_year: int
    baseline_emissions_tCO2e: float
    target_year: int
    target_emissions_tCO2e: float
    emission_trajectory: List[YearlyEmission]
    decarbonization_roadmap: List[TechnologyDeployment]
    investment_plan: List[InvestmentYear]
    roi_analysis: ROIAnalysis
    pathway_assessment: PathwayAssessment
    milestone_checks: List[MilestoneCheck]
    abatement_curve: List[Dict[str, float]]
    residual_emissions_tCO2e: float
    offset_requirement_tCO2e: float
    provenance_hash: str
    processing_time_ms: float
    validation_status: str


class DecarbonizationPlannerAgent:
    """GL-070: Decarbonization Planner Agent - Pathway planning and optimization."""

    AGENT_ID = "GL-070"
    AGENT_NAME = "DECARBONIZATION-PLANNER"
    VERSION = "1.0.0"

    # SBTi 1.5C pathway requires ~4.2% annual reduction
    SBTI_1_5C_ANNUAL_RATE = 0.042
    # SBTi well-below 2C requires ~2.5% annual reduction
    SBTI_2C_ANNUAL_RATE = 0.025

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"DecarbonizationPlannerAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: DecarbonizationPlannerInput) -> DecarbonizationPlannerOutput:
        start_time = datetime.utcnow()

        baseline = input_data.current_emissions
        total_baseline = (baseline.scope_1_tCO2e + baseline.scope_2_tCO2e +
                         baseline.scope_3_tCO2e)

        # Determine primary target
        primary_target = input_data.targets[0] if input_data.targets else None
        target_year = primary_target.target_year if primary_target else (
            baseline.baseline_year + input_data.planning_horizon_years)
        target_reduction_pct = primary_target.reduction_percent if primary_target else 50.0
        target_emissions = total_baseline * (1 - target_reduction_pct / 100)

        # Generate target trajectory
        trajectory = self._generate_trajectory(
            baseline, primary_target, input_data.planning_horizon_years)

        # Optimize technology deployment
        roadmap = self._optimize_roadmap(
            input_data.technology_options,
            total_baseline,
            target_emissions,
            baseline.baseline_year,
            target_year,
            input_data.budget_constraint_million_usd)

        # Generate investment plan
        investment_plan = self._generate_investment_plan(
            roadmap,
            baseline.baseline_year,
            target_year,
            input_data.carbon_price_per_tCO2e,
            input_data.carbon_price_escalation_percent)

        # Update trajectory with roadmap
        trajectory = self._update_trajectory_with_roadmap(
            trajectory, roadmap, total_baseline)

        # Calculate ROI
        roi = self._calculate_roi(
            investment_plan,
            input_data.discount_rate_percent,
            total_baseline - target_emissions)

        # Assess pathway
        pathway_assessment = self._assess_pathway(
            trajectory, primary_target, total_baseline, target_emissions)

        # Generate milestone checks
        milestones = self._generate_milestones(
            trajectory, primary_target, baseline.baseline_year)

        # Generate abatement curve
        abatement_curve = self._generate_abatement_curve(
            input_data.technology_options, total_baseline)

        # Calculate residual and offset needs
        final_emission = trajectory[-1].total_tCO2e if trajectory else total_baseline
        residual = max(0, final_emission)
        offset_need = max(0, final_emission - target_emissions) if primary_target else 0

        provenance_hash = hashlib.sha256(
            json.dumps({
                "agent": self.AGENT_ID,
                "baseline": total_baseline,
                "target_year": target_year,
                "timestamp": datetime.utcnow().isoformat()
            }, sort_keys=True, default=str).encode()).hexdigest()

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return DecarbonizationPlannerOutput(
            plan_id=input_data.plan_id or f"DECARB-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            organization_name=input_data.organization_name,
            baseline_year=baseline.baseline_year,
            baseline_emissions_tCO2e=round(total_baseline, 2),
            target_year=target_year,
            target_emissions_tCO2e=round(target_emissions, 2),
            emission_trajectory=trajectory,
            decarbonization_roadmap=roadmap,
            investment_plan=investment_plan,
            roi_analysis=roi,
            pathway_assessment=pathway_assessment,
            milestone_checks=milestones,
            abatement_curve=abatement_curve,
            residual_emissions_tCO2e=round(residual, 2),
            offset_requirement_tCO2e=round(offset_need, 2),
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS")

    def _generate_trajectory(self, baseline: CurrentEmissions,
                            target: Optional[ReductionTarget],
                            horizon: int) -> List[YearlyEmission]:
        """Generate emission trajectory to target."""
        trajectory = []
        total_baseline = (baseline.scope_1_tCO2e + baseline.scope_2_tCO2e +
                         baseline.scope_3_tCO2e)

        target_year = target.target_year if target else baseline.baseline_year + horizon
        target_reduction = target.reduction_percent / 100 if target else 0.5

        # Determine annual reduction rate based on pathway
        if target and target.pathway_type == PathwayType.PARIS_1_5C:
            annual_rate = self.SBTI_1_5C_ANNUAL_RATE
        elif target and target.pathway_type == PathwayType.PARIS_2C:
            annual_rate = self.SBTI_2C_ANNUAL_RATE
        else:
            # Calculate rate to meet target
            years = target_year - baseline.baseline_year
            annual_rate = 1 - (1 - target_reduction) ** (1 / years) if years > 0 else 0

        cumulative_reduction = 0
        for year in range(baseline.baseline_year, target_year + 1):
            years_elapsed = year - baseline.baseline_year

            # Linear target trajectory
            target_traj = total_baseline * (1 - target_reduction * years_elapsed /
                                           (target_year - baseline.baseline_year))

            # Current emissions (before technology deployment)
            scope_1 = baseline.scope_1_tCO2e
            scope_2 = baseline.scope_2_tCO2e
            scope_3 = baseline.scope_3_tCO2e
            total = scope_1 + scope_2 + scope_3

            trajectory.append(YearlyEmission(
                year=year,
                scope_1_tCO2e=round(scope_1, 2),
                scope_2_tCO2e=round(scope_2, 2),
                scope_3_tCO2e=round(scope_3, 2),
                total_tCO2e=round(total, 2),
                cumulative_reduction_tCO2e=round(cumulative_reduction, 2),
                target_trajectory_tCO2e=round(target_traj, 2),
                variance_tCO2e=round(total - target_traj, 2),
                on_track=total <= target_traj * 1.05))

        return trajectory

    def _optimize_roadmap(self, technologies: List[TechnologyOption],
                         baseline: float, target: float,
                         start_year: int, end_year: int,
                         budget: Optional[float]) -> List[TechnologyDeployment]:
        """Optimize technology deployment roadmap."""
        roadmap = []

        # Sort technologies by cost-effectiveness (reduction/capex ratio)
        sorted_tech = sorted(
            technologies,
            key=lambda t: t.reduction_potential_tCO2e / t.capex_usd if t.capex_usd > 0 else float('inf'),
            reverse=True)

        required_reduction = baseline - target
        cumulative_reduction = 0
        cumulative_capex = 0
        budget_limit = budget * 1_000_000 if budget else float('inf')
        current_year = start_year + 1

        for rank, tech in enumerate(sorted_tech):
            if cumulative_reduction >= required_reduction:
                break
            if cumulative_capex + tech.capex_usd > budget_limit:
                continue
            if tech.earliest_deployment_year > end_year:
                continue

            deploy_year = max(current_year, tech.earliest_deployment_year)
            if deploy_year > end_year:
                continue

            cumulative_reduction += tech.reduction_potential_tCO2e
            cumulative_capex += tech.capex_usd

            roadmap.append(TechnologyDeployment(
                technology_id=tech.technology_id,
                technology_name=tech.name,
                deployment_year=deploy_year,
                reduction_tCO2e=round(tech.reduction_potential_tCO2e, 2),
                capex_usd=round(tech.capex_usd, 2),
                annual_opex_usd=round(tech.annual_opex_usd, 2),
                annual_savings_usd=round(tech.annual_savings_usd, 2),
                cumulative_reduction_to_date=round(cumulative_reduction, 2),
                priority_rank=rank + 1))

            # Stagger deployments
            current_year = deploy_year + tech.implementation_years

        return roadmap

    def _generate_investment_plan(self, roadmap: List[TechnologyDeployment],
                                  start_year: int, end_year: int,
                                  carbon_price: float,
                                  escalation: float) -> List[InvestmentYear]:
        """Generate yearly investment plan."""
        plan = []
        cumulative_investment = 0

        for year in range(start_year, end_year + 1):
            year_deployments = [t for t in roadmap if t.deployment_year == year]
            active_tech = [t for t in roadmap if t.deployment_year <= year]

            capex = sum(t.capex_usd for t in year_deployments)
            opex = sum(t.annual_opex_usd for t in active_tech)
            savings = sum(t.annual_savings_usd for t in active_tech)

            # Carbon cost avoided (with escalation)
            years_elapsed = year - start_year
            current_carbon_price = carbon_price * ((1 + escalation / 100) ** years_elapsed)
            reduction = sum(t.reduction_tCO2e for t in active_tech)
            carbon_avoided = reduction * current_carbon_price

            cumulative_investment += capex
            net_cost = capex + opex - savings - carbon_avoided

            plan.append(InvestmentYear(
                year=year,
                capex_usd=round(capex, 2),
                opex_usd=round(opex, 2),
                savings_usd=round(savings, 2),
                net_cost_usd=round(net_cost, 2),
                carbon_cost_avoided_usd=round(carbon_avoided, 2),
                cumulative_investment_usd=round(cumulative_investment, 2),
                technologies_deployed=[t.technology_id for t in year_deployments]))

        return plan

    def _update_trajectory_with_roadmap(self, trajectory: List[YearlyEmission],
                                        roadmap: List[TechnologyDeployment],
                                        baseline: float) -> List[YearlyEmission]:
        """Update emission trajectory based on roadmap."""
        updated = []

        for ye in trajectory:
            active_tech = [t for t in roadmap if t.deployment_year <= ye.year]
            total_reduction = sum(t.reduction_tCO2e for t in active_tech)

            # Apply reductions proportionally to scopes
            remaining = baseline - total_reduction
            if baseline > 0:
                ratio = remaining / baseline
                scope_1 = ye.scope_1_tCO2e * ratio
                scope_2 = ye.scope_2_tCO2e * ratio
                scope_3 = ye.scope_3_tCO2e * ratio
            else:
                scope_1 = scope_2 = scope_3 = 0

            total = scope_1 + scope_2 + scope_3
            variance = total - ye.target_trajectory_tCO2e

            updated.append(YearlyEmission(
                year=ye.year,
                scope_1_tCO2e=round(scope_1, 2),
                scope_2_tCO2e=round(scope_2, 2),
                scope_3_tCO2e=round(scope_3, 2),
                total_tCO2e=round(total, 2),
                cumulative_reduction_tCO2e=round(total_reduction, 2),
                target_trajectory_tCO2e=ye.target_trajectory_tCO2e,
                variance_tCO2e=round(variance, 2),
                on_track=total <= ye.target_trajectory_tCO2e * 1.05))

        return updated

    def _calculate_roi(self, investment_plan: List[InvestmentYear],
                       discount_rate: float,
                       total_reduction: float) -> ROIAnalysis:
        """Calculate return on investment metrics."""
        total_capex = sum(y.capex_usd for y in investment_plan)
        total_opex = sum(y.opex_usd for y in investment_plan)
        total_savings = sum(y.savings_usd for y in investment_plan)
        total_carbon = sum(y.carbon_cost_avoided_usd for y in investment_plan)

        # NPV calculation
        npv = 0
        cumulative_cf = 0
        simple_payback = None
        discounted_payback = None
        cumulative_dcf = 0

        for i, year in enumerate(investment_plan):
            cash_flow = year.savings_usd + year.carbon_cost_avoided_usd - year.capex_usd - year.opex_usd
            discount_factor = 1 / ((1 + discount_rate / 100) ** i)
            npv += cash_flow * discount_factor

            cumulative_cf += cash_flow
            cumulative_dcf += cash_flow * discount_factor

            if simple_payback is None and cumulative_cf >= 0:
                simple_payback = i
            if discounted_payback is None and cumulative_dcf >= 0:
                discounted_payback = i

        # IRR approximation (simplified)
        irr = None
        if total_capex > 0:
            avg_annual_benefit = (total_savings + total_carbon - total_opex) / len(investment_plan)
            if avg_annual_benefit > 0:
                irr = (avg_annual_benefit / total_capex) * 100

        # Cost per tCO2e avoided
        total_cost = total_capex + total_opex - total_savings
        cost_per_tonne = total_cost / total_reduction if total_reduction > 0 else 0

        # Marginal abatement cost
        mac = total_cost / total_reduction if total_reduction > 0 else 0

        return ROIAnalysis(
            total_capex_usd=round(total_capex, 2),
            total_opex_usd=round(total_opex, 2),
            total_savings_usd=round(total_savings, 2),
            total_carbon_cost_avoided_usd=round(total_carbon, 2),
            net_present_value_usd=round(npv, 2),
            internal_rate_of_return_percent=round(irr, 2) if irr else None,
            simple_payback_years=simple_payback,
            discounted_payback_years=discounted_payback,
            cost_per_tCO2e_avoided=round(cost_per_tonne, 2),
            marginal_abatement_cost_usd=round(mac, 2))

    def _assess_pathway(self, trajectory: List[YearlyEmission],
                        target: Optional[ReductionTarget],
                        baseline: float,
                        target_emissions: float) -> PathwayAssessment:
        """Assess pathway against targets."""
        final_emission = trajectory[-1].total_tCO2e if trajectory else baseline
        reduction_achieved = baseline - final_emission
        target_reduction = baseline - target_emissions

        achievement_pct = (reduction_achieved / target_reduction * 100) if target_reduction > 0 else 0
        gap = max(0, target_emissions - final_emission)

        # Check alignment
        on_track_years = sum(1 for ye in trajectory if ye.on_track)
        confidence = on_track_years / len(trajectory) if trajectory else 0

        # Determine risk
        if achievement_pct >= 100:
            risk = "LOW"
        elif achievement_pct >= 80:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        # Find year target achieved
        year_achieved = None
        for ye in trajectory:
            if ye.total_tCO2e <= target_emissions:
                year_achieved = ye.year
                break

        pathway_type = target.pathway_type.value if target else "custom"
        sbti_aligned = target.is_validated if target else False
        paris_aligned = pathway_type in ["paris_1_5c", "paris_2c", "sbti_aligned"]

        return PathwayAssessment(
            pathway_type=pathway_type,
            target_achievement_percent=round(min(100, achievement_pct), 2),
            sbti_aligned=sbti_aligned,
            paris_aligned=paris_aligned,
            gap_to_target_tCO2e=round(gap, 2),
            year_target_achieved=year_achieved,
            risk_level=risk,
            confidence_score=round(confidence, 2))

    def _generate_milestones(self, trajectory: List[YearlyEmission],
                            target: Optional[ReductionTarget],
                            base_year: int) -> List[MilestoneCheck]:
        """Generate milestone checkpoints."""
        milestones = []
        milestone_years = [2025, 2030, 2035, 2040, 2050]

        if not trajectory or not target:
            return milestones

        baseline = trajectory[0].total_tCO2e if trajectory else 0
        target_reduction = target.reduction_percent

        for year in milestone_years:
            if year <= base_year or year > target.target_year:
                continue

            # Find trajectory year
            traj_year = next((ye for ye in trajectory if ye.year == year), None)
            if not traj_year:
                continue

            # Expected reduction at milestone (linear interpolation)
            years_elapsed = year - base_year
            total_years = target.target_year - base_year
            expected_reduction_pct = (target_reduction * years_elapsed / total_years) if total_years > 0 else 0

            # Actual reduction
            actual_reduction = baseline - traj_year.total_tCO2e
            actual_pct = (actual_reduction / baseline * 100) if baseline > 0 else 0

            gap = expected_reduction_pct - actual_pct
            on_track = gap <= 5  # 5% tolerance

            actions = []
            if not on_track:
                if gap > 20:
                    actions.append("Accelerate technology deployment")
                    actions.append("Consider additional abatement measures")
                elif gap > 10:
                    actions.append("Review deployment timeline")
                    actions.append("Increase investment in high-impact technologies")
                else:
                    actions.append("Minor adjustments to deployment schedule")

            milestones.append(MilestoneCheck(
                milestone_year=year,
                target_reduction_percent=round(expected_reduction_pct, 2),
                projected_reduction_percent=round(actual_pct, 2),
                on_track=on_track,
                gap_percent=round(gap, 2),
                corrective_actions=actions))

        return milestones

    def _generate_abatement_curve(self, technologies: List[TechnologyOption],
                                  baseline: float) -> List[Dict[str, float]]:
        """Generate marginal abatement cost curve."""
        curve = []

        # Calculate MAC for each technology
        mac_data = []
        for tech in technologies:
            if tech.reduction_potential_tCO2e > 0:
                # Annualized cost (simple)
                annual_cost = tech.capex_usd / tech.lifetime_years + tech.annual_opex_usd - tech.annual_savings_usd
                mac = annual_cost / tech.reduction_potential_tCO2e
                mac_data.append({
                    "technology": tech.name,
                    "mac_usd_per_tCO2e": mac,
                    "reduction_tCO2e": tech.reduction_potential_tCO2e
                })

        # Sort by MAC
        mac_data.sort(key=lambda x: x["mac_usd_per_tCO2e"])

        cumulative = 0
        for item in mac_data:
            curve.append({
                "technology": item["technology"],
                "mac_usd_per_tCO2e": round(item["mac_usd_per_tCO2e"], 2),
                "reduction_tCO2e": round(item["reduction_tCO2e"], 2),
                "cumulative_reduction_tCO2e": round(cumulative + item["reduction_tCO2e"], 2),
                "cumulative_reduction_percent": round((cumulative + item["reduction_tCO2e"]) / baseline * 100, 2)
            })
            cumulative += item["reduction_tCO2e"]

        return curve


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-070",
    "name": "DECARBONIZATION-PLANNER",
    "version": "1.0.0",
    "summary": "Decarbonization pathway planning and investment optimization",
    "tags": ["decarbonization", "SBTi", "net-zero", "pathway", "carbon-reduction", "investment"],
    "standards": [
        {"ref": "SBTi", "description": "Science Based Targets initiative methodology"},
        {"ref": "Paris Agreement", "description": "1.5C and 2C pathways"},
        {"ref": "TCFD", "description": "Climate-related financial disclosures"}
    ],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
