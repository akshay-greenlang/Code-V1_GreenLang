"""GL-078: Circular Economy Agent (CIRCULAR-ECONOMY).

Evaluates circular economy opportunities for energy systems.

Standards: ISO 14001, Ellen MacArthur Foundation
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CircularStrategy(str, Enum):
    REFUSE = "REFUSE"
    REDUCE = "REDUCE"
    REUSE = "REUSE"
    REPAIR = "REPAIR"
    REFURBISH = "REFURBISH"
    REMANUFACTURE = "REMANUFACTURE"
    RECYCLE = "RECYCLE"
    RECOVER = "RECOVER"


class MaterialStream(BaseModel):
    material_id: str
    material_name: str
    quantity_kg: float = Field(ge=0)
    recyclability_pct: float = Field(ge=0, le=100)
    current_recovery_pct: float = Field(ge=0, le=100)
    virgin_cost_kg: float = Field(ge=0)
    recycled_cost_kg: float = Field(ge=0)
    embodied_carbon_kg_co2: float = Field(ge=0)


class CircularEconomyInput(BaseModel):
    facility_id: str
    facility_name: str = Field(default="Facility")
    material_streams: List[MaterialStream] = Field(default_factory=list)
    waste_disposal_cost_kg: float = Field(default=0.10, ge=0)
    carbon_price_tonne: float = Field(default=50, ge=0)
    current_recycled_content_pct: float = Field(default=10, ge=0, le=100)
    target_recycled_content_pct: float = Field(default=50, ge=0, le=100)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CircularOpportunity(BaseModel):
    material_name: str
    strategy: str
    annual_savings_usd: float
    carbon_reduction_kg: float
    implementation_cost_usd: float
    payback_years: float


class CircularEconomyOutput(BaseModel):
    facility_id: str
    current_circularity_index: float
    target_circularity_index: float
    total_material_kg: float
    recoverable_material_kg: float
    current_recovery_rate_pct: float
    opportunities: List[CircularOpportunity]
    total_savings_potential_usd: float
    total_carbon_reduction_kg: float
    waste_reduction_pct: float
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class CircularEconomyAgent:
    AGENT_ID = "GL-078"
    AGENT_NAME = "CIRCULAR-ECONOMY"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"CircularEconomyAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = CircularEconomyInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _identify_opportunity(self, stream: MaterialStream, inp: CircularEconomyInput) -> Optional[CircularOpportunity]:
        """Identify circular economy opportunity for a material stream."""
        # Recovery gap
        recovery_gap = stream.recyclability_pct - stream.current_recovery_pct

        if recovery_gap <= 5:
            return None

        # Determine strategy
        if stream.recyclability_pct >= 90:
            strategy = CircularStrategy.RECYCLE.value
        elif stream.recyclability_pct >= 70:
            strategy = CircularStrategy.REMANUFACTURE.value
        elif stream.recyclability_pct >= 50:
            strategy = CircularStrategy.REFURBISH.value
        else:
            strategy = CircularStrategy.RECOVER.value

        # Calculate savings
        recovered_kg = stream.quantity_kg * (recovery_gap / 100)

        # Material cost savings
        material_savings = recovered_kg * (stream.virgin_cost_kg - stream.recycled_cost_kg)

        # Disposal cost savings
        disposal_savings = recovered_kg * inp.waste_disposal_cost_kg

        # Carbon value
        carbon_reduction = recovered_kg * stream.embodied_carbon_kg_co2 * 0.5  # 50% credit
        carbon_value = carbon_reduction / 1000 * inp.carbon_price_tonne

        total_savings = material_savings + disposal_savings + carbon_value

        # Implementation cost (rough estimate)
        impl_cost = recovered_kg * 0.5  # $0.50/kg implementation

        payback = impl_cost / total_savings if total_savings > 0 else 99

        return CircularOpportunity(
            material_name=stream.material_name,
            strategy=strategy,
            annual_savings_usd=round(total_savings, 2),
            carbon_reduction_kg=round(carbon_reduction, 2),
            implementation_cost_usd=round(impl_cost, 2),
            payback_years=round(payback, 2)
        )

    def _process(self, inp: CircularEconomyInput) -> CircularEconomyOutput:
        recommendations = []

        # Calculate totals
        total_material = sum(s.quantity_kg for s in inp.material_streams)
        recoverable = sum(s.quantity_kg * s.recyclability_pct / 100 for s in inp.material_streams)
        current_recovered = sum(s.quantity_kg * s.current_recovery_pct / 100 for s in inp.material_streams)

        current_rate = (current_recovered / total_material * 100) if total_material > 0 else 0
        target_rate = (recoverable / total_material * 100) if total_material > 0 else 0

        # Circularity index (simplified - based on recovery rate and recycled content)
        current_index = (current_rate + inp.current_recycled_content_pct) / 2
        target_index = (target_rate + inp.target_recycled_content_pct) / 2

        # Identify opportunities
        opportunities = []
        for stream in inp.material_streams:
            opp = self._identify_opportunity(stream, inp)
            if opp:
                opportunities.append(opp)

        # Sort by payback
        opportunities.sort(key=lambda x: x.payback_years)

        # Totals
        total_savings = sum(o.annual_savings_usd for o in opportunities)
        total_carbon = sum(o.carbon_reduction_kg for o in opportunities)
        waste_reduction = (target_rate - current_rate)

        # Recommendations
        if current_index < 30:
            recommendations.append(f"Low circularity index ({current_index:.0f}%) - significant improvement potential")
        if inp.current_recycled_content_pct < 20:
            recommendations.append("Low recycled content - evaluate recycled material suppliers")
        if current_rate < 50:
            recommendations.append(f"Recovery rate {current_rate:.0f}% below potential - implement sorting program")

        quick_wins = [o for o in opportunities if o.payback_years < 1]
        if quick_wins:
            recommendations.append(f"{len(quick_wins)} quick-win opportunities with <1 year payback")

        high_impact = [o for o in opportunities if o.carbon_reduction_kg > 1000]
        if high_impact:
            recommendations.append(f"{len(high_impact)} high-impact opportunities for carbon reduction")

        calc_hash = hashlib.sha256(json.dumps({
            "facility": inp.facility_id,
            "circularity": round(current_index, 1),
            "opportunities": len(opportunities)
        }).encode()).hexdigest()

        return CircularEconomyOutput(
            facility_id=inp.facility_id,
            current_circularity_index=round(current_index, 1),
            target_circularity_index=round(target_index, 1),
            total_material_kg=round(total_material, 2),
            recoverable_material_kg=round(recoverable, 2),
            current_recovery_rate_pct=round(current_rate, 1),
            opportunities=opportunities,
            total_savings_potential_usd=round(total_savings, 2),
            total_carbon_reduction_kg=round(total_carbon, 2),
            waste_reduction_pct=round(waste_reduction, 1),
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-078", "name": "CIRCULAR-ECONOMY", "version": "1.0.0",
    "summary": "Circular economy evaluation for energy systems",
    "standards": [{"ref": "ISO 14001"}, {"ref": "Ellen MacArthur Foundation"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
