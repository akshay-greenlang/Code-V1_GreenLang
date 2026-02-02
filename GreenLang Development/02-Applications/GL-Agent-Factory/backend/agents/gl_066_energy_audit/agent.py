"""GL-066: Energy Audit Agent (ENERGY-AUDIT).

Performs comprehensive energy audits and identifies savings opportunities.

Standards: ISO 50002, ASHRAE Level II
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AuditLevel(str, Enum):
    WALKTHROUGH = "WALKTHROUGH"  # Level I
    STANDARD = "STANDARD"  # Level II
    INVESTMENT = "INVESTMENT"  # Level III


class EnergySystem(BaseModel):
    system_id: str
    system_name: str
    system_type: str
    annual_consumption_kwh: float = Field(ge=0)
    annual_cost_usd: float = Field(ge=0)
    efficiency_pct: float = Field(ge=0, le=100)
    age_years: float = Field(ge=0)


class EnergyAuditInput(BaseModel):
    facility_id: str
    facility_name: str = Field(default="Facility")
    audit_level: AuditLevel = Field(default=AuditLevel.STANDARD)
    energy_systems: List[EnergySystem] = Field(default_factory=list)
    total_floor_area_m2: float = Field(default=10000, gt=0)
    annual_production_units: float = Field(default=100000, gt=0)
    electricity_rate_kwh: float = Field(default=0.10, ge=0)
    gas_rate_therm: float = Field(default=1.0, ge=0)
    industry_benchmark_kwh_m2: float = Field(default=200, gt=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SavingsOpportunity(BaseModel):
    opportunity_id: str
    description: str
    system_affected: str
    annual_savings_kwh: float
    annual_savings_usd: float
    implementation_cost_usd: float
    simple_payback_years: float
    priority: str


class EnergyAuditOutput(BaseModel):
    facility_id: str
    audit_level: str
    total_consumption_kwh: float
    total_cost_usd: float
    energy_use_intensity_kwh_m2: float
    energy_per_unit_kwh: float
    benchmark_comparison_pct: float
    opportunities: List[SavingsOpportunity]
    total_savings_potential_kwh: float
    total_savings_potential_usd: float
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class EnergyAuditAgent:
    AGENT_ID = "GL-066"
    AGENT_NAME = "ENERGY-AUDIT"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"EnergyAuditAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = EnergyAuditInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _identify_opportunities(self, system: EnergySystem, elec_rate: float) -> List[SavingsOpportunity]:
        """Identify savings opportunities for a system."""
        opportunities = []

        # Age-based replacement opportunity
        if system.age_years > 15 and system.efficiency_pct < 85:
            savings_pct = min(25, (100 - system.efficiency_pct) * 0.7)
            annual_savings = system.annual_consumption_kwh * savings_pct / 100
            annual_savings_usd = annual_savings * elec_rate
            impl_cost = system.annual_cost_usd * 2  # Rough estimate
            payback = impl_cost / annual_savings_usd if annual_savings_usd > 0 else 99

            opportunities.append(SavingsOpportunity(
                opportunity_id=f"{system.system_id}-REPLACE",
                description=f"Replace aging {system.system_name} with high-efficiency unit",
                system_affected=system.system_name,
                annual_savings_kwh=round(annual_savings, 0),
                annual_savings_usd=round(annual_savings_usd, 2),
                implementation_cost_usd=round(impl_cost, 0),
                simple_payback_years=round(payback, 1),
                priority="HIGH" if payback < 3 else "MEDIUM" if payback < 7 else "LOW"
            ))

        # Operational improvement
        if system.efficiency_pct < 90:
            savings_pct = 5  # Conservative operational savings
            annual_savings = system.annual_consumption_kwh * savings_pct / 100
            annual_savings_usd = annual_savings * elec_rate
            impl_cost = 5000  # Tune-up cost
            payback = impl_cost / annual_savings_usd if annual_savings_usd > 0 else 99

            opportunities.append(SavingsOpportunity(
                opportunity_id=f"{system.system_id}-OPTIMIZE",
                description=f"Optimize operation of {system.system_name}",
                system_affected=system.system_name,
                annual_savings_kwh=round(annual_savings, 0),
                annual_savings_usd=round(annual_savings_usd, 2),
                implementation_cost_usd=round(impl_cost, 0),
                simple_payback_years=round(payback, 1),
                priority="HIGH" if payback < 1 else "MEDIUM"
            ))

        return opportunities

    def _process(self, inp: EnergyAuditInput) -> EnergyAuditOutput:
        recommendations = []

        # Calculate totals
        total_consumption = sum(s.annual_consumption_kwh for s in inp.energy_systems)
        total_cost = sum(s.annual_cost_usd for s in inp.energy_systems)

        # Energy use intensity
        eui = total_consumption / inp.total_floor_area_m2

        # Energy per production unit
        energy_per_unit = total_consumption / inp.annual_production_units

        # Benchmark comparison
        benchmark_pct = (eui / inp.industry_benchmark_kwh_m2) * 100

        # Identify opportunities
        all_opportunities = []
        for system in inp.energy_systems:
            opps = self._identify_opportunities(system, inp.electricity_rate_kwh)
            all_opportunities.extend(opps)

        # Sort by payback
        all_opportunities.sort(key=lambda x: x.simple_payback_years)

        # Calculate total savings potential
        total_savings_kwh = sum(o.annual_savings_kwh for o in all_opportunities)
        total_savings_usd = sum(o.annual_savings_usd for o in all_opportunities)

        # Recommendations
        if benchmark_pct > 120:
            recommendations.append(f"Energy intensity {benchmark_pct:.0f}% of benchmark - significant improvement needed")
        if benchmark_pct < 80:
            recommendations.append(f"Energy intensity {benchmark_pct:.0f}% of benchmark - performing well")

        high_priority = [o for o in all_opportunities if o.priority == "HIGH"]
        if high_priority:
            recommendations.append(f"{len(high_priority)} high-priority opportunities with payback <3 years")

        old_systems = [s for s in inp.energy_systems if s.age_years > 20]
        if old_systems:
            recommendations.append(f"{len(old_systems)} systems >20 years old - plan replacements")

        if total_savings_usd > 0:
            savings_pct = total_savings_usd / total_cost * 100
            recommendations.append(f"Total savings potential: {savings_pct:.1f}% of energy costs")

        calc_hash = hashlib.sha256(json.dumps({
            "facility": inp.facility_id,
            "consumption": round(total_consumption, 0),
            "eui": round(eui, 1)
        }).encode()).hexdigest()

        return EnergyAuditOutput(
            facility_id=inp.facility_id,
            audit_level=inp.audit_level.value,
            total_consumption_kwh=round(total_consumption, 0),
            total_cost_usd=round(total_cost, 2),
            energy_use_intensity_kwh_m2=round(eui, 1),
            energy_per_unit_kwh=round(energy_per_unit, 4),
            benchmark_comparison_pct=round(benchmark_pct, 1),
            opportunities=all_opportunities,
            total_savings_potential_kwh=round(total_savings_kwh, 0),
            total_savings_potential_usd=round(total_savings_usd, 2),
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-066", "name": "ENERGY-AUDIT", "version": "1.0.0",
    "summary": "Comprehensive energy audit and savings identification",
    "standards": [{"ref": "ISO 50002"}, {"ref": "ASHRAE Level II"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
