"""GL-084: Net Zero Pathway Agent (NET-ZERO).

Develops net zero carbon pathways for facilities.

Standards: SBTi, GHG Protocol
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EmissionScope(str, Enum):
    SCOPE_1 = "SCOPE_1"
    SCOPE_2 = "SCOPE_2"
    SCOPE_3 = "SCOPE_3"


class DecarbonizationMeasure(BaseModel):
    measure_id: str
    description: str
    scope: EmissionScope
    reduction_tonnes_co2: float = Field(ge=0)
    implementation_cost_usd: float = Field(ge=0)
    annual_savings_usd: float = Field(default=0)
    implementation_year: int = Field(ge=2024)
    technology_readiness: int = Field(ge=1, le=9)


class NetZeroInput(BaseModel):
    facility_id: str
    facility_name: str = Field(default="Facility")
    baseline_year: int = Field(default=2020)
    target_year: int = Field(default=2050)
    current_year: int = Field(default=2024)
    scope1_emissions_tonnes: float = Field(default=1000, ge=0)
    scope2_emissions_tonnes: float = Field(default=2000, ge=0)
    scope3_emissions_tonnes: float = Field(default=5000, ge=0)
    reduction_target_pct: float = Field(default=90, ge=0, le=100)
    interim_target_2030_pct: float = Field(default=50, ge=0, le=100)
    measures: List[DecarbonizationMeasure] = Field(default_factory=list)
    carbon_price_tonne: float = Field(default=100, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PathwayMilestone(BaseModel):
    year: int
    target_emissions_tonnes: float
    projected_emissions_tonnes: float
    gap_tonnes: float
    cumulative_investment_usd: float
    on_track: bool


class NetZeroOutput(BaseModel):
    facility_id: str
    total_baseline_emissions: float
    total_current_emissions: float
    target_emissions_tonnes: float
    reduction_required_tonnes: float
    identified_reductions_tonnes: float
    reduction_gap_tonnes: float
    milestones: List[PathwayMilestone]
    total_investment_required_usd: float
    total_savings_projected_usd: float
    net_cost_usd: float
    cost_per_tonne_reduced_usd: float
    sbti_aligned: bool
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class NetZeroAgent:
    AGENT_ID = "GL-084"
    AGENT_NAME = "NET-ZERO"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"NetZeroAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = NetZeroInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _calculate_pathway(self, inp: NetZeroInput, total_baseline: float) -> List[PathwayMilestone]:
        """Calculate emissions pathway milestones."""
        milestones = []
        years = range(inp.current_year, inp.target_year + 1, 5)

        # Sort measures by year
        sorted_measures = sorted(inp.measures, key=lambda x: x.implementation_year)

        for year in years:
            # Linear target trajectory
            years_from_baseline = year - inp.baseline_year
            total_years = inp.target_year - inp.baseline_year
            progress = years_from_baseline / total_years if total_years > 0 else 1

            target = total_baseline * (1 - inp.reduction_target_pct / 100 * progress)

            # Projected based on measures implemented by this year
            implemented = [m for m in sorted_measures if m.implementation_year <= year]
            reduction = sum(m.reduction_tonnes_co2 for m in implemented)
            projected = total_baseline - reduction

            # Cumulative investment
            investment = sum(m.implementation_cost_usd for m in implemented)

            gap = projected - target
            on_track = gap <= 0

            milestones.append(PathwayMilestone(
                year=year,
                target_emissions_tonnes=round(target, 1),
                projected_emissions_tonnes=round(projected, 1),
                gap_tonnes=round(gap, 1),
                cumulative_investment_usd=round(investment, 2),
                on_track=on_track
            ))

        return milestones

    def _process(self, inp: NetZeroInput) -> NetZeroOutput:
        recommendations = []

        # Total emissions
        total_baseline = inp.scope1_emissions_tonnes + inp.scope2_emissions_tonnes + inp.scope3_emissions_tonnes
        total_current = total_baseline  # Simplified - in practice would track actual

        # Targets
        target_emissions = total_baseline * (1 - inp.reduction_target_pct / 100)
        reduction_required = total_baseline - target_emissions

        # Identified reductions
        identified = sum(m.reduction_tonnes_co2 for m in inp.measures)
        gap = reduction_required - identified

        # Calculate pathway
        milestones = self._calculate_pathway(inp, total_baseline)

        # Financial analysis
        total_investment = sum(m.implementation_cost_usd for m in inp.measures)
        total_savings = sum(m.annual_savings_usd for m in inp.measures) * (inp.target_year - inp.current_year)
        net_cost = total_investment - total_savings

        # Cost per tonne
        cost_per_tonne = net_cost / identified if identified > 0 else 0

        # SBTi alignment (4.2% annual reduction for 1.5°C)
        years_to_target = inp.target_year - inp.current_year
        required_annual_rate = inp.reduction_target_pct / years_to_target if years_to_target > 0 else 0
        sbti_aligned = required_annual_rate >= 4.2

        # Recommendations
        if gap > 0:
            recommendations.append(f"Reduction gap of {gap:,.0f} tonnes CO2 - identify additional measures")

        scope1_reduction = sum(m.reduction_tonnes_co2 for m in inp.measures if m.scope == EmissionScope.SCOPE_1)
        if scope1_reduction < inp.scope1_emissions_tonnes * 0.5:
            recommendations.append("Scope 1 reductions below 50% - prioritize fuel switching")

        scope2_reduction = sum(m.reduction_tonnes_co2 for m in inp.measures if m.scope == EmissionScope.SCOPE_2)
        if scope2_reduction < inp.scope2_emissions_tonnes * 0.7:
            recommendations.append("Scope 2 reductions below 70% - increase renewable procurement")

        if not sbti_aligned:
            recommendations.append(f"Not SBTi 1.5°C aligned ({required_annual_rate:.1f}% vs 4.2% required)")

        if cost_per_tonne > 200:
            recommendations.append(f"High abatement cost ${cost_per_tonne:.0f}/tonne - prioritize low-cost measures")

        milestone_2030 = next((m for m in milestones if m.year == 2030), None)
        if milestone_2030 and not milestone_2030.on_track:
            recommendations.append(f"2030 interim target at risk - {milestone_2030.gap_tonnes:,.0f} tonnes gap")

        calc_hash = hashlib.sha256(json.dumps({
            "facility": inp.facility_id,
            "baseline": round(total_baseline, 1),
            "identified": round(identified, 1)
        }).encode()).hexdigest()

        return NetZeroOutput(
            facility_id=inp.facility_id,
            total_baseline_emissions=round(total_baseline, 1),
            total_current_emissions=round(total_current, 1),
            target_emissions_tonnes=round(target_emissions, 1),
            reduction_required_tonnes=round(reduction_required, 1),
            identified_reductions_tonnes=round(identified, 1),
            reduction_gap_tonnes=round(gap, 1),
            milestones=milestones,
            total_investment_required_usd=round(total_investment, 2),
            total_savings_projected_usd=round(total_savings, 2),
            net_cost_usd=round(net_cost, 2),
            cost_per_tonne_reduced_usd=round(cost_per_tonne, 2),
            sbti_aligned=sbti_aligned,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-084", "name": "NET-ZERO", "version": "1.0.0",
    "summary": "Net zero carbon pathway development",
    "standards": [{"ref": "SBTi"}, {"ref": "GHG Protocol"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
