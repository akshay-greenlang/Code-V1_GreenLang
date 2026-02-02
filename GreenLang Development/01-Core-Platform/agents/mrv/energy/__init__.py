# -*- coding: utf-8 -*-
"""
GreenLang MRV Energy Sector Agents
==================================

This package contains agents for measuring, reporting, and verifying
greenhouse gas emissions from energy sector activities.

Agents:
    GL-MRV-ENE-001: PowerGenerationMRVAgent - Power generation emissions
    GL-MRV-ENE-002: GridEmissionsTrackerAgent - Grid emission factors
    GL-MRV-ENE-003: RenewableGenerationMRVAgent - Renewable energy MRV
    GL-MRV-ENE-004: StorageSystemsMRVAgent - Energy storage emissions
    GL-MRV-ENE-005: TransmissionLossMRVAgent - T&D losses
    GL-MRV-ENE-006: FuelSupplyChainMRVAgent - Upstream fuel emissions
    GL-MRV-ENE-007: CHPSystemsMRVAgent - Combined heat & power
    GL-MRV-ENE-008: HydrogenProductionMRVAgent - Hydrogen emissions

All agents follow the CRITICAL PATH pattern with:
- Zero-hallucination guarantee (no LLM in calculation path)
- Full audit trail with SHA-256 provenance hashing
- GHG Protocol and regulatory compliance
- Deterministic, reproducible calculations

Example:
    >>> from greenlang.agents.mrv.energy import PowerGenerationMRVAgent
    >>> agent = PowerGenerationMRVAgent()
    >>> result = agent.process({
    ...     "facility_id": "PLANT-001",
    ...     "unit_id": "UNIT-1",
    ...     "generation_type": "combined_cycle_gas_turbine",
    ...     "fuel_type": "natural_gas",
    ...     "fuel_consumption": 50000,
    ...     "fuel_consumption_unit": "MMBTU",
    ...     "net_generation_mwh": 5000,
    ...     "reporting_period_start": "2024-01-01T00:00:00Z",
    ...     "reporting_period_end": "2024-01-31T23:59:59Z",
    ... })
"""

from greenlang.agents.mrv.energy.base import MRVEnergyBaseAgent

# Schemas
from greenlang.agents.mrv.energy.schemas import (
    # Enums
    FuelType,
    GenerationType,
    EmissionScope,
    HydrogenProductionMethod,
    StorageTechnology,
    GridRegion,
    UncertaintyLevel,
    # Base models
    MRVBaseInput,
    MRVBaseOutput,
    # Power Generation
    PowerGenerationInput,
    PowerGenerationOutput,
    # Grid Emissions
    GridEmissionsInput,
    GridEmissionsOutput,
    # Renewable Generation
    RenewableGenerationInput,
    RenewableGenerationOutput,
    # Storage Systems
    StorageSystemsInput,
    StorageSystemsOutput,
    # Transmission Loss
    TransmissionLossInput,
    TransmissionLossOutput,
    # Fuel Supply Chain
    FuelSupplyChainInput,
    FuelSupplyChainOutput,
    # CHP Systems
    CHPSystemsInput,
    CHPSystemsOutput,
    # Hydrogen Production
    HydrogenProductionInput,
    HydrogenProductionOutput,
)

# Agents
from greenlang.agents.mrv.energy.power_generation_mrv import (
    PowerGenerationMRVAgent,
    calculate_power_generation_emissions,
)
from greenlang.agents.mrv.energy.grid_emissions_tracker import (
    GridEmissionsTrackerAgent,
)
from greenlang.agents.mrv.energy.renewable_generation_mrv import (
    RenewableGenerationMRVAgent,
)
from greenlang.agents.mrv.energy.storage_systems_mrv import (
    StorageSystemsMRVAgent,
)
from greenlang.agents.mrv.energy.transmission_loss_mrv import (
    TransmissionLossMRVAgent,
)
from greenlang.agents.mrv.energy.fuel_supply_chain_mrv import (
    FuelSupplyChainMRVAgent,
)
from greenlang.agents.mrv.energy.chp_systems_mrv import (
    CHPSystemsMRVAgent,
)
from greenlang.agents.mrv.energy.hydrogen_production_mrv import (
    HydrogenProductionMRVAgent,
)

__version__ = "1.0.0"

__all__ = [
    # Base
    "MRVEnergyBaseAgent",
    # Enums
    "FuelType",
    "GenerationType",
    "EmissionScope",
    "HydrogenProductionMethod",
    "StorageTechnology",
    "GridRegion",
    "UncertaintyLevel",
    # Base models
    "MRVBaseInput",
    "MRVBaseOutput",
    # Power Generation (GL-MRV-ENE-001)
    "PowerGenerationMRVAgent",
    "PowerGenerationInput",
    "PowerGenerationOutput",
    "calculate_power_generation_emissions",
    # Grid Emissions (GL-MRV-ENE-002)
    "GridEmissionsTrackerAgent",
    "GridEmissionsInput",
    "GridEmissionsOutput",
    # Renewable Generation (GL-MRV-ENE-003)
    "RenewableGenerationMRVAgent",
    "RenewableGenerationInput",
    "RenewableGenerationOutput",
    # Storage Systems (GL-MRV-ENE-004)
    "StorageSystemsMRVAgent",
    "StorageSystemsInput",
    "StorageSystemsOutput",
    # Transmission Loss (GL-MRV-ENE-005)
    "TransmissionLossMRVAgent",
    "TransmissionLossInput",
    "TransmissionLossOutput",
    # Fuel Supply Chain (GL-MRV-ENE-006)
    "FuelSupplyChainMRVAgent",
    "FuelSupplyChainInput",
    "FuelSupplyChainOutput",
    # CHP Systems (GL-MRV-ENE-007)
    "CHPSystemsMRVAgent",
    "CHPSystemsInput",
    "CHPSystemsOutput",
    # Hydrogen Production (GL-MRV-ENE-008)
    "HydrogenProductionMRVAgent",
    "HydrogenProductionInput",
    "HydrogenProductionOutput",
]

# Agent registry for dynamic lookup
AGENT_REGISTRY = {
    "GL-MRV-ENE-001": PowerGenerationMRVAgent,
    "GL-MRV-ENE-002": GridEmissionsTrackerAgent,
    "GL-MRV-ENE-003": RenewableGenerationMRVAgent,
    "GL-MRV-ENE-004": StorageSystemsMRVAgent,
    "GL-MRV-ENE-005": TransmissionLossMRVAgent,
    "GL-MRV-ENE-006": FuelSupplyChainMRVAgent,
    "GL-MRV-ENE-007": CHPSystemsMRVAgent,
    "GL-MRV-ENE-008": HydrogenProductionMRVAgent,
}


def get_agent(agent_id: str):
    """
    Get an MRV Energy agent by ID.

    Args:
        agent_id: Agent identifier (e.g., "GL-MRV-ENE-001")

    Returns:
        Agent class

    Raises:
        KeyError: If agent_id not found
    """
    if agent_id not in AGENT_REGISTRY:
        raise KeyError(f"Unknown agent: {agent_id}. Available: {list(AGENT_REGISTRY.keys())}")
    return AGENT_REGISTRY[agent_id]
