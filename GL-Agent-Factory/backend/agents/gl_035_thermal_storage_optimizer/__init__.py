"""GL-035: Thermal Storage Optimizer Agent"""

from .agent import (
    ThermalStorageOptimizerAgent,
    ThermalStorageOptimizerInput,
    ThermalStorageOptimizerOutput,
    DemandProfile,
    EnergyPrice,
    HourlySchedule,
    PACK_SPEC,
)

__all__ = [
    "ThermalStorageOptimizerAgent",
    "ThermalStorageOptimizerInput",
    "ThermalStorageOptimizerOutput",
    "DemandProfile",
    "EnergyPrice",
    "HourlySchedule",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-035"
__agent_name__ = "THERMAL-STORAGE-OPTIMIZER"
