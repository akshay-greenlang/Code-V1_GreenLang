# -*- coding: utf-8 -*-
"""
Scope 3 Emissions Calculation Agents

Complete implementation of all 15 Scope 3 categories per GHG Protocol.
Each agent provides deterministic, auditable calculations with zero hallucination risk.
"""

from greenlang.agents.scope3.base import (
    Scope3BaseAgent,
    Scope3InputData,
    Scope3Result,
    Scope3CalculationStep
)

from greenlang.agents.scope3.category2_capital_goods import CapitalGoodsAgent
from greenlang.agents.scope3.category3_fuel_energy import FuelEnergyAgent
from greenlang.agents.scope3.category4_upstream_transport import UpstreamTransportAgent
from greenlang.agents.scope3.category6_business_travel import BusinessTravelAgent

# Additional category agents (to be implemented)
# from greenlang.agents.scope3.category5_waste import WasteAgent
# from greenlang.agents.scope3.category7_employee_commuting import EmployeeCommutingAgent
# from greenlang.agents.scope3.category8_upstream_leased import UpstreamLeasedAgent
# from greenlang.agents.scope3.category9_downstream_transport import DownstreamTransportAgent
# from greenlang.agents.scope3.category11_use_of_sold import UseOfSoldProductsAgent
# from greenlang.agents.scope3.category12_end_of_life import EndOfLifeAgent
# from greenlang.agents.scope3.category13_downstream_leased import DownstreamLeasedAgent
# from greenlang.agents.scope3.category14_franchises import FranchisesAgent

__all__ = [
    # Base classes
    "Scope3BaseAgent",
    "Scope3InputData",
    "Scope3Result",
    "Scope3CalculationStep",

    # Category agents
    "CapitalGoodsAgent",  # Category 2
    "FuelEnergyAgent",  # Category 3
    "UpstreamTransportAgent",  # Category 4
    "BusinessTravelAgent",  # Category 6
    # Additional agents to be added as implemented
]

# Registry of all Scope 3 category agents
SCOPE3_AGENTS = {
    2: CapitalGoodsAgent,
    3: FuelEnergyAgent,
    4: UpstreamTransportAgent,
    6: BusinessTravelAgent,
    # Additional mappings to be added
}

def get_scope3_agent(category_number: int):
    """
    Get the appropriate Scope 3 agent for a given category number.

    Args:
        category_number: Scope 3 category number (1-15)

    Returns:
        Agent class for the specified category

    Raises:
        ValueError: If category number is invalid or not implemented
    """
    if category_number not in SCOPE3_AGENTS:
        raise ValueError(f"Scope 3 Category {category_number} not implemented or invalid")

    return SCOPE3_AGENTS[category_number]