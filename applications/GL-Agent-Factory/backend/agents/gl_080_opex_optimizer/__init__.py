"""GL-080: OPEX Optimizer Agent (OPEXOPTIMIZER)"""

from .agent import (
    OpexOptimizerAgent,
    OpexOptimizerInput,
    OpexOptimizerOutput,
    OperatingCost,
    MaintenanceCost,
    EnergyCost,
    LaborCost,
    OpexBreakdown,
    OptimizationOpportunity,
    CostDriver,
    CostCategory,
    MaintenanceType,
    PACK_SPEC,
)

from .formulas import (
    calculate_annual_opex,
    calculate_maintenance_schedule,
    calculate_energy_cost,
    calculate_labor_optimization,
    project_opex_savings,
)

__all__ = [
    "OpexOptimizerAgent",
    "OpexOptimizerInput",
    "OpexOptimizerOutput",
    "OperatingCost",
    "MaintenanceCost",
    "EnergyCost",
    "LaborCost",
    "OpexBreakdown",
    "OptimizationOpportunity",
    "CostDriver",
    "CostCategory",
    "MaintenanceType",
    "PACK_SPEC",
    "calculate_annual_opex",
    "calculate_maintenance_schedule",
    "calculate_energy_cost",
    "calculate_labor_optimization",
    "project_opex_savings",
]

__version__ = "1.0.0"
__agent_id__ = "GL-080"
__agent_name__ = "OPEXOPTIMIZER"
