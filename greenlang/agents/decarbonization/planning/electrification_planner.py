# -*- coding: utf-8 -*-
"""
GL-DECARB-X-011: Electrification Planner Agent
===============================================

Plans electrification of fossil fuel processes including heat,
transport, and industrial applications.

Author: GreenLang Team
Version: 1.0.0
"""

import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig
from greenlang.agents.base_agents import DeterministicAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism import DeterministicClock, content_hash, deterministic_id

logger = logging.getLogger(__name__)


class ProcessType(str, Enum):
    SPACE_HEATING = "space_heating"
    WATER_HEATING = "water_heating"
    PROCESS_HEAT_LOW = "process_heat_low"  # <100C
    PROCESS_HEAT_MEDIUM = "process_heat_medium"  # 100-200C
    PROCESS_HEAT_HIGH = "process_heat_high"  # >200C
    VEHICLE_FLEET = "vehicle_fleet"
    MATERIAL_HANDLING = "material_handling"
    COMPRESSED_AIR = "compressed_air"


class ElectrificationTechnology(str, Enum):
    HEAT_PUMP_AIR = "heat_pump_air"
    HEAT_PUMP_GROUND = "heat_pump_ground"
    HEAT_PUMP_HIGH_TEMP = "heat_pump_high_temp"
    ELECTRIC_BOILER = "electric_boiler"
    INDUCTION_HEATING = "induction_heating"
    ELECTRIC_ARC = "electric_arc"
    BATTERY_ELECTRIC_VEHICLE = "battery_electric_vehicle"
    ELECTRIC_FORKLIFT = "electric_forklift"


class ElectrificationProject(BaseModel):
    project_id: str = Field(...)
    name: str = Field(...)
    process_type: ProcessType = Field(...)
    technology: ElectrificationTechnology = Field(...)

    # Current state
    current_fuel: str = Field(default="natural_gas")
    current_fuel_consumption_mwh: float = Field(..., ge=0)
    current_emissions_tco2e: float = Field(..., ge=0)

    # Electrified state
    electricity_consumption_mwh: float = Field(..., ge=0)
    grid_emissions_tco2e: float = Field(..., ge=0)

    # Savings
    emission_reduction_tco2e: float = Field(...)
    reduction_percent: float = Field(...)

    # Costs
    capital_cost_usd: float = Field(default=0, ge=0)
    annual_operating_savings_usd: float = Field(default=0)
    payback_years: Optional[float] = Field(None, ge=0)

    # Technical
    cop: float = Field(default=1.0, ge=0.1, description="Coefficient of Performance")
    is_technically_feasible: bool = Field(default=True)
    barriers: List[str] = Field(default_factory=list)


class ElectrificationPlan(BaseModel):
    plan_id: str = Field(...)
    projects: List[ElectrificationProject] = Field(default_factory=list)

    total_fuel_displaced_mwh: float = Field(default=0, ge=0)
    total_electricity_added_mwh: float = Field(default=0, ge=0)
    total_emission_reduction_tco2e: float = Field(default=0, ge=0)
    total_investment_usd: float = Field(default=0, ge=0)
    total_annual_savings_usd: float = Field(default=0)

    electrification_percent: float = Field(default=0, ge=0, le=100)
    provenance_hash: str = Field(default="")


class ElectrificationInput(BaseModel):
    operation: str = Field(default="plan")
    processes: List[Dict[str, Any]] = Field(default_factory=list)
    grid_emission_factor_kgco2_mwh: float = Field(default=400, ge=0)
    natural_gas_emission_factor_kgco2_kwh: float = Field(default=0.2, ge=0)
    electricity_price_usd_mwh: float = Field(default=100, ge=0)
    natural_gas_price_usd_mwh: float = Field(default=40, ge=0)


class ElectrificationOutput(BaseModel):
    operation: str = Field(...)
    success: bool = Field(...)
    plan: Optional[ElectrificationPlan] = Field(None)
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    error_message: Optional[str] = Field(None)


class ElectrificationPlanner(DeterministicAgent):
    """
    GL-DECARB-X-011: Electrification Planner Agent

    Plans electrification of fossil fuel processes.
    """

    AGENT_ID = "GL-DECARB-X-011"
    AGENT_NAME = "Electrification Planner"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="ElectrificationPlanner",
        category=AgentCategory.CRITICAL,
        description="Plans process electrification"
    )

    # Default COP values by technology
    DEFAULT_COP = {
        ElectrificationTechnology.HEAT_PUMP_AIR: 3.0,
        ElectrificationTechnology.HEAT_PUMP_GROUND: 4.0,
        ElectrificationTechnology.HEAT_PUMP_HIGH_TEMP: 2.5,
        ElectrificationTechnology.ELECTRIC_BOILER: 0.98,
        ElectrificationTechnology.INDUCTION_HEATING: 0.95,
        ElectrificationTechnology.BATTERY_ELECTRIC_VEHICLE: 3.5,  # Miles per kWh equivalent
        ElectrificationTechnology.ELECTRIC_FORKLIFT: 0.9,
    }

    def __init__(self, config: Optional[AgentConfig] = None, enable_audit_trail: bool = True):
        super().__init__(enable_audit_trail=enable_audit_trail)
        self.config = config or AgentConfig(
            name=self.AGENT_NAME, description="Plans electrification", version=self.VERSION
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        calculation_trace = []

        try:
            elec_input = ElectrificationInput(**inputs)
            calculation_trace.append(f"Operation: {elec_input.operation}")

            if elec_input.operation == "plan":
                projects = []

                for proc in elec_input.processes:
                    process_type = ProcessType(proc.get("process_type", "space_heating"))
                    fuel_consumption = proc.get("fuel_consumption_mwh", 1000)

                    # Select technology based on process type
                    if process_type in [ProcessType.SPACE_HEATING, ProcessType.WATER_HEATING]:
                        tech = ElectrificationTechnology.HEAT_PUMP_AIR
                    elif process_type == ProcessType.PROCESS_HEAT_LOW:
                        tech = ElectrificationTechnology.HEAT_PUMP_HIGH_TEMP
                    elif process_type == ProcessType.VEHICLE_FLEET:
                        tech = ElectrificationTechnology.BATTERY_ELECTRIC_VEHICLE
                    else:
                        tech = ElectrificationTechnology.ELECTRIC_BOILER

                    cop = self.DEFAULT_COP.get(tech, 1.0)

                    # Calculate
                    current_emissions = fuel_consumption * elec_input.natural_gas_emission_factor_kgco2_kwh
                    elec_consumption = fuel_consumption / cop
                    grid_emissions = elec_consumption * elec_input.grid_emission_factor_kgco2_mwh / 1000

                    emission_reduction = current_emissions - grid_emissions
                    reduction_pct = (emission_reduction / current_emissions * 100) if current_emissions > 0 else 0

                    # Cost analysis
                    capital_cost = fuel_consumption * 200  # Rough estimate
                    annual_fuel_cost = fuel_consumption * elec_input.natural_gas_price_usd_mwh
                    annual_elec_cost = elec_consumption * elec_input.electricity_price_usd_mwh
                    annual_savings = annual_fuel_cost - annual_elec_cost
                    payback = capital_cost / annual_savings if annual_savings > 0 else None

                    project = ElectrificationProject(
                        project_id=deterministic_id({"process": process_type.value}, "elec_"),
                        name=f"Electrify {process_type.value}",
                        process_type=process_type,
                        technology=tech,
                        current_fuel_consumption_mwh=fuel_consumption,
                        current_emissions_tco2e=current_emissions / 1000,
                        electricity_consumption_mwh=elec_consumption,
                        grid_emissions_tco2e=grid_emissions,
                        emission_reduction_tco2e=emission_reduction / 1000,
                        reduction_percent=reduction_pct,
                        capital_cost_usd=capital_cost,
                        annual_operating_savings_usd=annual_savings,
                        payback_years=payback,
                        cop=cop
                    )
                    projects.append(project)

                plan = ElectrificationPlan(
                    plan_id=deterministic_id({"count": len(projects)}, "elecplan_"),
                    projects=projects,
                    total_fuel_displaced_mwh=sum(p.current_fuel_consumption_mwh for p in projects),
                    total_electricity_added_mwh=sum(p.electricity_consumption_mwh for p in projects),
                    total_emission_reduction_tco2e=sum(p.emission_reduction_tco2e for p in projects),
                    total_investment_usd=sum(p.capital_cost_usd for p in projects),
                    total_annual_savings_usd=sum(p.annual_operating_savings_usd for p in projects)
                )
                plan.provenance_hash = content_hash(plan.model_dump(exclude={"provenance_hash"}))

                calculation_trace.append(f"Planned {len(projects)} electrification projects")

                self._capture_audit_entry(
                    operation="plan",
                    inputs=inputs,
                    outputs={"projects": len(projects)},
                    calculation_trace=calculation_trace
                )

                return {
                    "operation": "plan",
                    "success": True,
                    "plan": plan.model_dump(),
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": DeterministicClock.now().isoformat()
                }
            else:
                raise ValueError(f"Unknown operation: {elec_input.operation}")

        except Exception as e:
            self.logger.error(f"Planning failed: {str(e)}", exc_info=True)
            return {
                "operation": inputs.get("operation", "unknown"),
                "success": False,
                "error_message": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": DeterministicClock.now().isoformat()
            }
