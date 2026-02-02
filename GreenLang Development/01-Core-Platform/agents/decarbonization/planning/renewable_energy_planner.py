# -*- coding: utf-8 -*-
"""
GL-DECARB-X-010: Renewable Energy Planner Agent
=================================================

Plans renewable energy adoption strategies including on-site generation,
PPAs, and renewable energy certificates.

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


class RenewableType(str, Enum):
    SOLAR_PV = "solar_pv"
    WIND_ONSHORE = "wind_onshore"
    WIND_OFFSHORE = "wind_offshore"
    BIOMASS = "biomass"
    HYDRO = "hydro"
    GEOTHERMAL = "geothermal"


class ProcurementOption(str, Enum):
    ONSITE = "onsite"
    PPA_PHYSICAL = "ppa_physical"
    PPA_VIRTUAL = "ppa_virtual"
    REC_UNBUNDLED = "rec_unbundled"
    GREEN_TARIFF = "green_tariff"


class RenewableProject(BaseModel):
    project_id: str = Field(...)
    name: str = Field(...)
    renewable_type: RenewableType = Field(...)
    procurement_option: ProcurementOption = Field(...)

    # Capacity
    capacity_kw: float = Field(..., ge=0)
    annual_generation_mwh: float = Field(..., ge=0)
    capacity_factor: float = Field(default=0.25, ge=0, le=1)

    # Impact
    emission_reduction_tco2e: float = Field(..., ge=0)
    scope_2_reduction_percent: float = Field(default=0, ge=0, le=100)

    # Cost
    capital_cost_usd: float = Field(default=0, ge=0)
    annual_cost_usd: float = Field(default=0)
    levelized_cost_per_mwh: float = Field(default=0, ge=0)

    # Timeline
    implementation_months: int = Field(default=12, ge=1)
    contract_years: Optional[int] = Field(None, ge=1)


class RenewableEnergyPlan(BaseModel):
    plan_id: str = Field(...)
    target_year: int = Field(...)
    target_renewable_percent: float = Field(..., ge=0, le=100)

    # Current state
    current_renewable_percent: float = Field(default=0, ge=0, le=100)
    current_consumption_mwh: float = Field(..., ge=0)

    # Projects
    projects: List[RenewableProject] = Field(default_factory=list)

    # Summary
    total_renewable_mwh: float = Field(default=0, ge=0)
    achieved_renewable_percent: float = Field(default=0, ge=0, le=100)
    total_emission_reduction_tco2e: float = Field(default=0, ge=0)
    total_investment_usd: float = Field(default=0, ge=0)
    total_annual_cost_usd: float = Field(default=0)

    provenance_hash: str = Field(default="")


class RenewableEnergyInput(BaseModel):
    operation: str = Field(default="plan")
    target_year: int = Field(default=2030)
    target_renewable_percent: float = Field(default=100, ge=0, le=100)
    current_consumption_mwh: float = Field(default=10000, ge=0)
    current_renewable_percent: float = Field(default=0, ge=0, le=100)
    grid_emission_factor_kgco2_mwh: float = Field(default=400, ge=0)
    available_roof_area_m2: Optional[float] = Field(None, ge=0)
    budget_constraint_usd: Optional[float] = Field(None, ge=0)


class RenewableEnergyOutput(BaseModel):
    operation: str = Field(...)
    success: bool = Field(...)
    plan: Optional[RenewableEnergyPlan] = Field(None)
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    error_message: Optional[str] = Field(None)


class RenewableEnergyPlanner(DeterministicAgent):
    """
    GL-DECARB-X-010: Renewable Energy Planner Agent

    Plans renewable energy procurement strategies.
    """

    AGENT_ID = "GL-DECARB-X-010"
    AGENT_NAME = "Renewable Energy Planner"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="RenewableEnergyPlanner",
        category=AgentCategory.CRITICAL,
        description="Plans renewable energy adoption"
    )

    def __init__(self, config: Optional[AgentConfig] = None, enable_audit_trail: bool = True):
        super().__init__(enable_audit_trail=enable_audit_trail)
        self.config = config or AgentConfig(
            name=self.AGENT_NAME, description="Plans renewable energy", version=self.VERSION
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        calculation_trace = []

        try:
            plan_input = RenewableEnergyInput(**inputs)
            calculation_trace.append(f"Operation: {plan_input.operation}")

            if plan_input.operation == "plan":
                # Calculate gap to fill
                current_renewable_mwh = plan_input.current_consumption_mwh * plan_input.current_renewable_percent / 100
                target_renewable_mwh = plan_input.current_consumption_mwh * plan_input.target_renewable_percent / 100
                gap_mwh = target_renewable_mwh - current_renewable_mwh

                calculation_trace.append(f"Gap to fill: {gap_mwh:,.0f} MWh")

                projects = []

                # Option 1: On-site solar (if roof area available)
                if plan_input.available_roof_area_m2 and plan_input.available_roof_area_m2 > 100:
                    # Assume 150W/m2 and 15% capacity factor
                    capacity_kw = plan_input.available_roof_area_m2 * 0.15
                    generation_mwh = capacity_kw * 8760 * 0.15 / 1000
                    solar_project = RenewableProject(
                        project_id=deterministic_id({"type": "solar"}, "re_"),
                        name="On-site Solar PV",
                        renewable_type=RenewableType.SOLAR_PV,
                        procurement_option=ProcurementOption.ONSITE,
                        capacity_kw=capacity_kw,
                        annual_generation_mwh=min(generation_mwh, gap_mwh * 0.3),
                        capacity_factor=0.15,
                        emission_reduction_tco2e=min(generation_mwh, gap_mwh * 0.3) * plan_input.grid_emission_factor_kgco2_mwh / 1000,
                        capital_cost_usd=capacity_kw * 1200,
                        levelized_cost_per_mwh=50,
                        implementation_months=6
                    )
                    projects.append(solar_project)
                    gap_mwh -= solar_project.annual_generation_mwh

                # Option 2: PPA for remaining
                if gap_mwh > 0:
                    ppa_project = RenewableProject(
                        project_id=deterministic_id({"type": "ppa"}, "re_"),
                        name="Wind PPA",
                        renewable_type=RenewableType.WIND_ONSHORE,
                        procurement_option=ProcurementOption.PPA_VIRTUAL,
                        capacity_kw=0,
                        annual_generation_mwh=gap_mwh,
                        capacity_factor=0.35,
                        emission_reduction_tco2e=gap_mwh * plan_input.grid_emission_factor_kgco2_mwh / 1000,
                        capital_cost_usd=0,
                        annual_cost_usd=gap_mwh * 45,
                        levelized_cost_per_mwh=45,
                        implementation_months=6,
                        contract_years=15
                    )
                    projects.append(ppa_project)

                # Calculate totals
                total_renewable = sum(p.annual_generation_mwh for p in projects)
                achieved_percent = (current_renewable_mwh + total_renewable) / plan_input.current_consumption_mwh * 100

                plan = RenewableEnergyPlan(
                    plan_id=deterministic_id({"target": plan_input.target_renewable_percent}, "replan_"),
                    target_year=plan_input.target_year,
                    target_renewable_percent=plan_input.target_renewable_percent,
                    current_renewable_percent=plan_input.current_renewable_percent,
                    current_consumption_mwh=plan_input.current_consumption_mwh,
                    projects=projects,
                    total_renewable_mwh=total_renewable,
                    achieved_renewable_percent=achieved_percent,
                    total_emission_reduction_tco2e=sum(p.emission_reduction_tco2e for p in projects),
                    total_investment_usd=sum(p.capital_cost_usd for p in projects),
                    total_annual_cost_usd=sum(p.annual_cost_usd for p in projects)
                )
                plan.provenance_hash = content_hash(plan.model_dump(exclude={"provenance_hash"}))

                calculation_trace.append(f"Planned {len(projects)} projects, {achieved_percent:.1f}% renewable")

                self._capture_audit_entry(
                    operation="plan",
                    inputs=inputs,
                    outputs={"projects": len(projects), "percent": achieved_percent},
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
                raise ValueError(f"Unknown operation: {plan_input.operation}")

        except Exception as e:
            self.logger.error(f"Planning failed: {str(e)}", exc_info=True)
            return {
                "operation": inputs.get("operation", "unknown"),
                "success": False,
                "error_message": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": DeterministicClock.now().isoformat()
            }
