# -*- coding: utf-8 -*-
"""
GL-DECARB-X-012: Fuel Switching Optimizer Agent
================================================

Optimizes fuel transitions to lower-carbon alternatives including
natural gas, biomethane, hydrogen, and electrification.

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


class FuelType(str, Enum):
    COAL = "coal"
    FUEL_OIL = "fuel_oil"
    DIESEL = "diesel"
    NATURAL_GAS = "natural_gas"
    BIOMETHANE = "biomethane"
    HYDROGEN_GREY = "hydrogen_grey"
    HYDROGEN_BLUE = "hydrogen_blue"
    HYDROGEN_GREEN = "hydrogen_green"
    ELECTRICITY = "electricity"
    BIOMASS = "biomass"


# Emission factors in kgCO2e/kWh
EMISSION_FACTORS = {
    FuelType.COAL: 0.34,
    FuelType.FUEL_OIL: 0.27,
    FuelType.DIESEL: 0.25,
    FuelType.NATURAL_GAS: 0.20,
    FuelType.BIOMETHANE: 0.02,
    FuelType.HYDROGEN_GREY: 0.12,
    FuelType.HYDROGEN_BLUE: 0.04,
    FuelType.HYDROGEN_GREEN: 0.0,
    FuelType.ELECTRICITY: 0.4,  # Grid average, varies by region
    FuelType.BIOMASS: 0.03,
}


class FuelSwitchOption(BaseModel):
    option_id: str = Field(...)
    from_fuel: FuelType = Field(...)
    to_fuel: FuelType = Field(...)
    consumption_mwh: float = Field(..., ge=0)

    # Emissions
    current_emissions_tco2e: float = Field(...)
    new_emissions_tco2e: float = Field(...)
    reduction_tco2e: float = Field(...)
    reduction_percent: float = Field(...)

    # Costs
    fuel_cost_change_usd_year: float = Field(default=0)
    capital_cost_usd: float = Field(default=0, ge=0)
    cost_per_tco2e: float = Field(default=0)

    # Feasibility
    is_feasible: bool = Field(default=True)
    barriers: List[str] = Field(default_factory=list)
    implementation_months: int = Field(default=12, ge=1)


class FuelSwitchingInput(BaseModel):
    operation: str = Field(default="optimize")
    current_fuel: FuelType = Field(default=FuelType.NATURAL_GAS)
    consumption_mwh: float = Field(default=10000, ge=0)
    target_reduction_percent: float = Field(default=50, ge=0, le=100)
    available_fuels: List[str] = Field(default_factory=list)
    electricity_emission_factor: float = Field(default=0.4, ge=0)


class FuelSwitchingOutput(BaseModel):
    operation: str = Field(...)
    success: bool = Field(...)
    options: List[FuelSwitchOption] = Field(default_factory=list)
    recommended_option: Optional[FuelSwitchOption] = Field(None)
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    error_message: Optional[str] = Field(None)


class FuelSwitchingOptimizer(DeterministicAgent):
    """GL-DECARB-X-012: Fuel Switching Optimizer Agent"""

    AGENT_ID = "GL-DECARB-X-012"
    AGENT_NAME = "Fuel Switching Optimizer"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="FuelSwitchingOptimizer",
        category=AgentCategory.CRITICAL,
        description="Optimizes fuel transitions"
    )

    def __init__(self, config: Optional[AgentConfig] = None, enable_audit_trail: bool = True):
        super().__init__(enable_audit_trail=enable_audit_trail)
        self.config = config or AgentConfig(
            name=self.AGENT_NAME, description="Optimizes fuel switching", version=self.VERSION
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        calculation_trace = []

        try:
            switch_input = FuelSwitchingInput(**inputs)
            calculation_trace.append(f"Operation: {switch_input.operation}")

            if switch_input.operation == "optimize":
                current_ef = EMISSION_FACTORS.get(switch_input.current_fuel, 0.2)
                current_emissions = switch_input.consumption_mwh * current_ef

                # Generate options for each potential fuel
                options = []
                for fuel in FuelType:
                    if fuel == switch_input.current_fuel:
                        continue

                    new_ef = EMISSION_FACTORS.get(fuel, 0.2)
                    if fuel == FuelType.ELECTRICITY:
                        new_ef = switch_input.electricity_emission_factor

                    new_emissions = switch_input.consumption_mwh * new_ef
                    reduction = current_emissions - new_emissions
                    reduction_pct = (reduction / current_emissions * 100) if current_emissions > 0 else 0

                    # Skip if doesn't meet target
                    if reduction_pct < 0:
                        continue

                    option = FuelSwitchOption(
                        option_id=deterministic_id({"to": fuel.value}, "fswitch_"),
                        from_fuel=switch_input.current_fuel,
                        to_fuel=fuel,
                        consumption_mwh=switch_input.consumption_mwh,
                        current_emissions_tco2e=current_emissions / 1000,
                        new_emissions_tco2e=new_emissions / 1000,
                        reduction_tco2e=reduction / 1000,
                        reduction_percent=reduction_pct
                    )
                    options.append(option)

                # Sort by reduction
                options.sort(key=lambda o: o.reduction_percent, reverse=True)
                recommended = options[0] if options else None

                calculation_trace.append(f"Generated {len(options)} fuel switching options")

                self._capture_audit_entry(
                    operation="optimize",
                    inputs=inputs,
                    outputs={"options": len(options)},
                    calculation_trace=calculation_trace
                )

                return {
                    "operation": "optimize",
                    "success": True,
                    "options": [o.model_dump() for o in options],
                    "recommended_option": recommended.model_dump() if recommended else None,
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": DeterministicClock.now().isoformat()
                }
            else:
                raise ValueError(f"Unknown operation: {switch_input.operation}")

        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}", exc_info=True)
            return {
                "operation": inputs.get("operation", "unknown"),
                "success": False,
                "error_message": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": DeterministicClock.now().isoformat()
            }
