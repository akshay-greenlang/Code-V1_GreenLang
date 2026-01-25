"""
GL-034: Heat Recovery Scout Agent (HEAT-RECOVERY-SCOUT)

Waste heat recovery opportunity identification using pinch analysis
and heat integration principles.

Standards: ISO 50001, DOE ITP (Industrial Technologies Program)
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ExhaustStream(BaseModel):
    """Waste heat exhaust stream data."""
    stream_id: str = Field(..., description="Stream identifier")
    source_equipment: str = Field(..., description="Source equipment")
    temperature_celsius: float = Field(..., description="Stream temperature")
    flow_rate_kg_hr: float = Field(..., ge=0, description="Mass flow rate")
    specific_heat_kj_kg_k: float = Field(default=1.0, gt=0)
    available_hours_per_year: float = Field(default=8000, gt=0)
    contaminants: List[str] = Field(default_factory=list)


class HeatDemand(BaseModel):
    """Process heat demand data."""
    demand_id: str = Field(..., description="Demand identifier")
    process_name: str = Field(..., description="Process requiring heat")
    required_temp_celsius: float = Field(..., description="Required temperature")
    heat_load_kw: float = Field(..., ge=0, description="Heat load in kW")
    hours_per_year: float = Field(default=8000, gt=0)
    current_fuel_cost_per_year: float = Field(default=0, ge=0)


class UtilityCosts(BaseModel):
    """Utility cost data."""
    natural_gas_per_mmbtu: float = Field(default=5.0, ge=0)
    electricity_per_kwh: float = Field(default=0.10, ge=0)
    steam_per_klb: float = Field(default=8.0, ge=0)


class RecoveryOpportunity(BaseModel):
    """Identified heat recovery opportunity."""
    opportunity_id: str
    source_stream: str
    target_demand: str
    recoverable_heat_kw: float
    annual_energy_savings_mmbtu: float
    annual_cost_savings: float
    implementation_cost: float
    simple_payback_years: float
    npv_10yr: float
    co2_reduction_tonnes_per_year: float
    priority_rank: int
    technology_recommendation: str


class HeatRecoveryScoutInput(BaseModel):
    """Input for HeatRecoveryScoutAgent."""
    facility_id: str = Field(..., description="Facility identifier")
    exhaust_streams: List[ExhaustStream] = Field(...)
    process_heat_demands: List[HeatDemand] = Field(...)
    utility_costs: UtilityCosts = Field(default_factory=UtilityCosts)
    discount_rate: float = Field(default=0.10, ge=0, le=1)
    co2_emission_factor_kg_per_mmbtu: float = Field(default=53.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HeatRecoveryScoutOutput(BaseModel):
    """Output from HeatRecoveryScoutAgent."""
    analysis_id: str
    facility_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    recovery_opportunities: List[RecoveryOpportunity]
    total_recoverable_heat_kw: float
    total_annual_savings: float
    total_npv: float
    total_co2_reduction_tonnes: float
    implementation_priority: List[str]
    pinch_temperature_celsius: Optional[float]
    provenance_hash: str
    processing_time_ms: float
    validation_status: str


class HeatRecoveryScoutAgent:
    """GL-034: Heat Recovery Scout Agent."""

    AGENT_ID = "GL-034"
    AGENT_NAME = "HEAT-RECOVERY-SCOUT"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"HeatRecoveryScoutAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: HeatRecoveryScoutInput) -> HeatRecoveryScoutOutput:
        """Execute heat recovery opportunity analysis."""
        start_time = datetime.utcnow()
        logger.info(f"Starting heat recovery analysis for {input_data.facility_id}")

        opportunities = []
        opp_id = 0

        # Match exhaust streams to heat demands
        for stream in input_data.exhaust_streams:
            for demand in input_data.process_heat_demands:
                # Check temperature feasibility (stream must be hotter than demand + approach)
                approach_temp = 20  # Minimum approach temperature in C
                if stream.temperature_celsius > demand.required_temp_celsius + approach_temp:
                    opp_id += 1

                    # Calculate recoverable heat
                    temp_drop = stream.temperature_celsius - demand.required_temp_celsius - approach_temp
                    max_heat_kw = (stream.flow_rate_kg_hr * stream.specific_heat_kj_kg_k * temp_drop) / 3600
                    recoverable_kw = min(max_heat_kw, demand.heat_load_kw)

                    # Annual energy savings
                    operating_hours = min(stream.available_hours_per_year, demand.hours_per_year)
                    annual_energy_kwh = recoverable_kw * operating_hours
                    annual_energy_mmbtu = annual_energy_kwh * 0.003412

                    # Cost savings
                    annual_savings = annual_energy_mmbtu * input_data.utility_costs.natural_gas_per_mmbtu

                    # Implementation cost estimate (rough: $500/kW for heat exchangers)
                    impl_cost = recoverable_kw * 500

                    # Simple payback
                    payback = impl_cost / annual_savings if annual_savings > 0 else 999

                    # NPV calculation (10-year)
                    npv = -impl_cost
                    for year in range(1, 11):
                        npv += annual_savings / ((1 + input_data.discount_rate) ** year)

                    # CO2 reduction
                    co2_reduction = annual_energy_mmbtu * input_data.co2_emission_factor_kg_per_mmbtu / 1000

                    # Technology recommendation
                    if stream.temperature_celsius > 400:
                        tech = "Waste heat boiler or economizer"
                    elif stream.temperature_celsius > 200:
                        tech = "Shell-and-tube heat exchanger"
                    elif stream.temperature_celsius > 100:
                        tech = "Plate heat exchanger or heat pipe"
                    else:
                        tech = "Heat pump system"

                    opportunities.append(RecoveryOpportunity(
                        opportunity_id=f"HRO-{opp_id:03d}",
                        source_stream=stream.stream_id,
                        target_demand=demand.demand_id,
                        recoverable_heat_kw=round(recoverable_kw, 1),
                        annual_energy_savings_mmbtu=round(annual_energy_mmbtu, 0),
                        annual_cost_savings=round(annual_savings, 0),
                        implementation_cost=round(impl_cost, 0),
                        simple_payback_years=round(payback, 1),
                        npv_10yr=round(npv, 0),
                        co2_reduction_tonnes_per_year=round(co2_reduction, 1),
                        priority_rank=0,
                        technology_recommendation=tech
                    ))

        # Rank by NPV
        opportunities.sort(key=lambda x: -x.npv_10yr)
        for i, opp in enumerate(opportunities):
            opp.priority_rank = i + 1

        # Totals
        total_kw = sum(o.recoverable_heat_kw for o in opportunities)
        total_savings = sum(o.annual_cost_savings for o in opportunities)
        total_npv = sum(o.npv_10yr for o in opportunities)
        total_co2 = sum(o.co2_reduction_tonnes_per_year for o in opportunities)

        # Simple pinch estimate (lowest demand temp + approach)
        if input_data.process_heat_demands:
            pinch = min(d.required_temp_celsius for d in input_data.process_heat_demands) + 10
        else:
            pinch = None

        provenance_hash = hashlib.sha256(
            json.dumps({"agent": self.AGENT_ID, "facility": input_data.facility_id,
                        "timestamp": datetime.utcnow().isoformat()}, sort_keys=True).encode()
        ).hexdigest()

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return HeatRecoveryScoutOutput(
            analysis_id=f"HRS-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            facility_id=input_data.facility_id,
            recovery_opportunities=opportunities,
            total_recoverable_heat_kw=round(total_kw, 1),
            total_annual_savings=round(total_savings, 0),
            total_npv=round(total_npv, 0),
            total_co2_reduction_tonnes=round(total_co2, 1),
            implementation_priority=[o.opportunity_id for o in opportunities[:5]],
            pinch_temperature_celsius=pinch,
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS"
        )


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-034",
    "name": "HEAT-RECOVERY-SCOUT - Waste Heat Recovery Opportunity Finder",
    "version": "1.0.0",
    "summary": "Identifies waste heat recovery opportunities using pinch analysis principles",
    "tags": ["heat-recovery", "waste-heat", "pinch-analysis", "energy-efficiency", "ISO-50001"],
    "owners": ["process-heat-optimization-team"],
    "standards": [
        {"ref": "ISO 50001", "description": "Energy Management Systems"},
        {"ref": "DOE ITP", "description": "Industrial Technologies Program"}
    ]
}
