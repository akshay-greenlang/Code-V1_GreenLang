# -*- coding: utf-8 -*-
"""
GL-DECARB-X-013: Energy Efficiency Identifier Agent
====================================================

Identifies energy efficiency opportunities across facilities
and processes with savings potential and implementation details.

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


class EfficiencyCategory(str, Enum):
    LIGHTING = "lighting"
    HVAC = "hvac"
    MOTORS_DRIVES = "motors_drives"
    COMPRESSED_AIR = "compressed_air"
    STEAM_SYSTEMS = "steam_systems"
    PROCESS_HEAT = "process_heat"
    BUILDING_ENVELOPE = "building_envelope"
    CONTROLS_BMS = "controls_bms"
    WASTE_HEAT_RECOVERY = "waste_heat_recovery"


class EfficiencyOpportunity(BaseModel):
    opportunity_id: str = Field(...)
    name: str = Field(...)
    category: EfficiencyCategory = Field(...)
    description: str = Field(default="")

    # Savings potential
    energy_savings_mwh: float = Field(..., ge=0)
    energy_savings_percent: float = Field(default=0, ge=0, le=100)
    emission_reduction_tco2e: float = Field(..., ge=0)
    cost_savings_usd: float = Field(default=0, ge=0)

    # Investment
    capital_cost_usd: float = Field(default=0, ge=0)
    simple_payback_years: Optional[float] = Field(None, ge=0)

    # Implementation
    implementation_months: int = Field(default=6, ge=1)
    complexity: str = Field(default="medium")
    maintenance_impact: str = Field(default="neutral")


class EfficiencyAssessment(BaseModel):
    assessment_id: str = Field(...)
    facility_name: str = Field(...)
    assessment_date: datetime = Field(default_factory=DeterministicClock.now)

    # Baseline
    baseline_consumption_mwh: float = Field(..., ge=0)
    baseline_emissions_tco2e: float = Field(..., ge=0)
    baseline_cost_usd: float = Field(..., ge=0)

    # Opportunities
    opportunities: List[EfficiencyOpportunity] = Field(default_factory=list)

    # Summary
    total_savings_potential_mwh: float = Field(default=0, ge=0)
    total_savings_percent: float = Field(default=0, ge=0, le=100)
    total_emission_reduction_tco2e: float = Field(default=0, ge=0)
    total_cost_savings_usd: float = Field(default=0, ge=0)
    total_investment_usd: float = Field(default=0, ge=0)
    average_payback_years: Optional[float] = Field(None, ge=0)

    provenance_hash: str = Field(default="")


class EnergyEfficiencyInput(BaseModel):
    operation: str = Field(default="identify")
    facility_name: str = Field(default="Main Facility")
    baseline_consumption_mwh: float = Field(default=10000, ge=0)
    baseline_cost_usd: float = Field(default=1000000, ge=0)
    emission_factor_kgco2_mwh: float = Field(default=400, ge=0)
    categories_to_assess: List[str] = Field(default_factory=list)


class EnergyEfficiencyOutput(BaseModel):
    operation: str = Field(...)
    success: bool = Field(...)
    assessment: Optional[EfficiencyAssessment] = Field(None)
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    error_message: Optional[str] = Field(None)


# Default efficiency opportunities by category (typical savings percentages)
DEFAULT_OPPORTUNITIES = {
    EfficiencyCategory.LIGHTING: {
        "name": "LED Lighting Retrofit",
        "savings_percent": 0.05,
        "cost_factor": 50,
        "payback_years": 2,
    },
    EfficiencyCategory.HVAC: {
        "name": "HVAC Optimization",
        "savings_percent": 0.08,
        "cost_factor": 100,
        "payback_years": 3,
    },
    EfficiencyCategory.MOTORS_DRIVES: {
        "name": "VFD Installation",
        "savings_percent": 0.06,
        "cost_factor": 80,
        "payback_years": 2.5,
    },
    EfficiencyCategory.COMPRESSED_AIR: {
        "name": "Compressed Air Optimization",
        "savings_percent": 0.04,
        "cost_factor": 40,
        "payback_years": 1.5,
    },
    EfficiencyCategory.CONTROLS_BMS: {
        "name": "BMS Upgrade",
        "savings_percent": 0.07,
        "cost_factor": 60,
        "payback_years": 2,
    },
}


class EnergyEfficiencyIdentifier(DeterministicAgent):
    """GL-DECARB-X-013: Energy Efficiency Identifier Agent"""

    AGENT_ID = "GL-DECARB-X-013"
    AGENT_NAME = "Energy Efficiency Identifier"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="EnergyEfficiencyIdentifier",
        category=AgentCategory.CRITICAL,
        description="Identifies energy efficiency opportunities"
    )

    def __init__(self, config: Optional[AgentConfig] = None, enable_audit_trail: bool = True):
        super().__init__(enable_audit_trail=enable_audit_trail)
        self.config = config or AgentConfig(
            name=self.AGENT_NAME, description="Identifies efficiency opportunities", version=self.VERSION
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        calculation_trace = []

        try:
            eff_input = EnergyEfficiencyInput(**inputs)
            calculation_trace.append(f"Operation: {eff_input.operation}")

            if eff_input.operation == "identify":
                baseline_emissions = eff_input.baseline_consumption_mwh * eff_input.emission_factor_kgco2_mwh / 1000

                opportunities = []
                for cat, params in DEFAULT_OPPORTUNITIES.items():
                    savings_mwh = eff_input.baseline_consumption_mwh * params["savings_percent"]
                    savings_usd = eff_input.baseline_cost_usd * params["savings_percent"]
                    capital = savings_usd * params["cost_factor"] / 100

                    opp = EfficiencyOpportunity(
                        opportunity_id=deterministic_id({"cat": cat.value}, "eff_"),
                        name=params["name"],
                        category=cat,
                        energy_savings_mwh=savings_mwh,
                        energy_savings_percent=params["savings_percent"] * 100,
                        emission_reduction_tco2e=savings_mwh * eff_input.emission_factor_kgco2_mwh / 1000,
                        cost_savings_usd=savings_usd,
                        capital_cost_usd=capital,
                        simple_payback_years=params["payback_years"]
                    )
                    opportunities.append(opp)

                # Sort by payback
                opportunities.sort(key=lambda o: o.simple_payback_years or 999)

                assessment = EfficiencyAssessment(
                    assessment_id=deterministic_id({"facility": eff_input.facility_name}, "assess_"),
                    facility_name=eff_input.facility_name,
                    baseline_consumption_mwh=eff_input.baseline_consumption_mwh,
                    baseline_emissions_tco2e=baseline_emissions,
                    baseline_cost_usd=eff_input.baseline_cost_usd,
                    opportunities=opportunities,
                    total_savings_potential_mwh=sum(o.energy_savings_mwh for o in opportunities),
                    total_emission_reduction_tco2e=sum(o.emission_reduction_tco2e for o in opportunities),
                    total_cost_savings_usd=sum(o.cost_savings_usd for o in opportunities),
                    total_investment_usd=sum(o.capital_cost_usd for o in opportunities)
                )
                assessment.total_savings_percent = (
                    assessment.total_savings_potential_mwh / eff_input.baseline_consumption_mwh * 100
                ) if eff_input.baseline_consumption_mwh > 0 else 0

                assessment.provenance_hash = content_hash(assessment.model_dump(exclude={"provenance_hash"}))

                calculation_trace.append(f"Identified {len(opportunities)} efficiency opportunities")

                self._capture_audit_entry(
                    operation="identify",
                    inputs=inputs,
                    outputs={"opportunities": len(opportunities)},
                    calculation_trace=calculation_trace
                )

                return {
                    "operation": "identify",
                    "success": True,
                    "assessment": assessment.model_dump(),
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": DeterministicClock.now().isoformat()
                }
            else:
                raise ValueError(f"Unknown operation: {eff_input.operation}")

        except Exception as e:
            self.logger.error(f"Identification failed: {str(e)}", exc_info=True)
            return {
                "operation": inputs.get("operation", "unknown"),
                "success": False,
                "error_message": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": DeterministicClock.now().isoformat()
            }
