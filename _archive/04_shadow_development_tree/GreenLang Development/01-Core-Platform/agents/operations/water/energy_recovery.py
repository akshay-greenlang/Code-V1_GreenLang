# -*- coding: utf-8 -*-
"""
GL-OPS-WAT-006: Energy Recovery Agent
=====================================

Operations agent for identifying and optimizing energy recovery
opportunities in water systems.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class RecoveryType(str, Enum):
    TURBINE = "turbine"
    PUMP_AS_TURBINE = "pump_as_turbine"
    BIOGAS = "biogas"
    HEAT_RECOVERY = "heat_recovery"
    PRESSURE_REDUCTION = "pressure_reduction"


class RecoveryOpportunity(BaseModel):
    """Energy recovery opportunity."""
    opportunity_id: str
    location_id: str
    recovery_type: RecoveryType
    description: str
    available_head_m: Optional[float] = None
    available_flow_m3_hr: Optional[float] = None
    potential_power_kw: float
    annual_energy_kwh: float
    annual_co2_savings_kg: float
    estimated_capex: float
    payback_years: float
    feasibility_score: float  # 0-100


class ExistingRecoverySystem(BaseModel):
    """Existing energy recovery system."""
    system_id: str
    location_id: str
    recovery_type: RecoveryType
    installed_capacity_kw: float
    actual_output_kwh: float
    availability_percent: float
    efficiency_percent: float


class RecoveryResult(BaseModel):
    """Energy recovery analysis result."""
    total_potential_kw: float
    total_potential_annual_kwh: float
    total_co2_savings_kg: float
    opportunities: List[RecoveryOpportunity]
    priority_ranking: List[str]


class EnergyRecoveryInput(BaseModel):
    """Input for energy recovery analysis."""
    system_id: str
    prv_locations: List[Dict[str, Any]] = Field(default_factory=list)
    wastewater_plants: List[Dict[str, Any]] = Field(default_factory=list)
    existing_systems: List[ExistingRecoverySystem] = Field(default_factory=list)
    electricity_price_per_kwh: float = Field(default=0.10)
    grid_emission_factor: float = Field(default=0.417)


class EnergyRecoveryOutput(BaseModel):
    """Output from energy recovery analysis."""
    system_id: str
    recovery_result: RecoveryResult
    existing_systems_summary: Dict[str, Any]
    total_current_recovery_kwh: float
    total_potential_additional_kwh: float
    recommendations: List[str]
    provenance_hash: str
    calculation_timestamp: datetime
    processing_time_ms: float


class EnergyRecoveryAgent(BaseAgent):
    """
    GL-OPS-WAT-006: Energy Recovery Agent

    Identifies and analyzes energy recovery opportunities.
    """

    AGENT_ID = "GL-OPS-WAT-006"
    AGENT_NAME = "Energy Recovery Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Energy recovery from water systems",
                version=self.VERSION,
            )
        super().__init__(config)
        self.logger.info(f"Initialized {self.AGENT_ID}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        start_time = time.time()

        try:
            er_input = EnergyRecoveryInput(**input_data)
            opportunities = []

            # Analyze PRV locations for micro-hydro
            for i, prv in enumerate(er_input.prv_locations):
                head = prv.get("pressure_drop_m", 20)
                flow = prv.get("flow_m3_hr", 100)

                # P = rho * g * Q * H * eta / 3.6e6
                # Simplified: kW = m3/hr * m * 0.00272 * efficiency
                efficiency = 0.75
                power_kw = flow * head * 0.00272 * efficiency
                annual_kwh = power_kw * 8760 * 0.80  # 80% capacity factor

                if power_kw > 5:  # Minimum viable size
                    capex = power_kw * 2500  # $2500/kW estimate
                    annual_savings = annual_kwh * er_input.electricity_price_per_kwh
                    payback = capex / annual_savings if annual_savings > 0 else 999

                    opp = RecoveryOpportunity(
                        opportunity_id=f"OPP-PRV-{i+1}",
                        location_id=prv.get("location_id", f"PRV-{i+1}"),
                        recovery_type=RecoveryType.PUMP_AS_TURBINE,
                        description=f"Micro-hydro at PRV location with {head}m head",
                        available_head_m=head,
                        available_flow_m3_hr=flow,
                        potential_power_kw=round(power_kw, 2),
                        annual_energy_kwh=round(annual_kwh, 0),
                        annual_co2_savings_kg=round(annual_kwh * er_input.grid_emission_factor, 0),
                        estimated_capex=round(capex, 0),
                        payback_years=round(payback, 1),
                        feasibility_score=min(100, 100 - payback * 5),
                    )
                    opportunities.append(opp)

            # Analyze wastewater plants for biogas
            for i, wwtp in enumerate(er_input.wastewater_plants):
                sludge_tonnes = wwtp.get("sludge_tonnes_day", 10)

                # Biogas potential: ~25 m3/tonne sludge, ~60% CH4, 10 kWh/m3 biogas
                biogas_m3 = sludge_tonnes * 25 * 365
                energy_kwh = biogas_m3 * 6  # 6 kWh/m3 considering CHP efficiency

                if energy_kwh > 50000:
                    capex = 500000 + energy_kwh * 0.5
                    annual_savings = energy_kwh * er_input.electricity_price_per_kwh
                    payback = capex / annual_savings if annual_savings > 0 else 999

                    opp = RecoveryOpportunity(
                        opportunity_id=f"OPP-BIO-{i+1}",
                        location_id=wwtp.get("plant_id", f"WWTP-{i+1}"),
                        recovery_type=RecoveryType.BIOGAS,
                        description=f"Biogas CHP from anaerobic digestion",
                        potential_power_kw=round(energy_kwh / 8760 * 0.9, 2),
                        annual_energy_kwh=round(energy_kwh, 0),
                        annual_co2_savings_kg=round(energy_kwh * er_input.grid_emission_factor, 0),
                        estimated_capex=round(capex, 0),
                        payback_years=round(payback, 1),
                        feasibility_score=min(100, 100 - payback * 4),
                    )
                    opportunities.append(opp)

            # Sort by feasibility
            opportunities.sort(key=lambda x: x.feasibility_score, reverse=True)

            # Calculate totals
            total_power = sum(o.potential_power_kw for o in opportunities)
            total_annual = sum(o.annual_energy_kwh for o in opportunities)
            total_co2 = sum(o.annual_co2_savings_kg for o in opportunities)

            # Existing systems summary
            current_recovery = sum(s.actual_output_kwh for s in er_input.existing_systems)

            result = RecoveryResult(
                total_potential_kw=round(total_power, 2),
                total_potential_annual_kwh=round(total_annual, 0),
                total_co2_savings_kg=round(total_co2, 0),
                opportunities=opportunities,
                priority_ranking=[o.opportunity_id for o in opportunities[:5]],
            )

            # Recommendations
            recommendations = []
            if opportunities:
                best = opportunities[0]
                recommendations.append(f"Priority: Implement {best.recovery_type.value} at {best.location_id} (payback: {best.payback_years} years)")
            if total_annual > 100000:
                recommendations.append(f"Total potential recovery of {total_annual:,.0f} kWh/year available")

            provenance_hash = hashlib.sha256(
                json.dumps({"system": er_input.system_id, "opportunities": len(opportunities)}, sort_keys=True).encode()
            ).hexdigest()[:16]

            processing_time = (time.time() - start_time) * 1000

            output = EnergyRecoveryOutput(
                system_id=er_input.system_id,
                recovery_result=result,
                existing_systems_summary={
                    "count": len(er_input.existing_systems),
                    "total_capacity_kw": sum(s.installed_capacity_kw for s in er_input.existing_systems),
                    "total_output_kwh": current_recovery,
                },
                total_current_recovery_kwh=round(current_recovery, 0),
                total_potential_additional_kwh=round(total_annual, 0),
                recommendations=recommendations,
                provenance_hash=provenance_hash,
                calculation_timestamp=DeterministicClock.now(),
                processing_time_ms=processing_time,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            self.logger.error(f"Energy recovery analysis failed: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))
