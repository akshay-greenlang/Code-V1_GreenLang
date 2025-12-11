"""GL-078: Tariff Optimizer Agent (TARIFFOPTIMIZER)"""

from .agent import (
    TariffOptimizerAgent,
    TariffOptimizerInput,
    TariffOptimizerOutput,
    UsageProfile,
    TariffOption,
    LoadShiftOpportunity,
    TariffRecommendation,
    SavingsAnalysis,
    DemandChargeAnalysis,
    RateSchedule,
    RateType,
    SeasonType,
    PACK_SPEC,
)

from .formulas import (
    calculate_tou_cost,
    calculate_demand_charge,
    calculate_optimal_shift,
    calculate_annual_savings,
    calculate_peak_shaving_benefit,
)

__all__ = [
    "TariffOptimizerAgent",
    "TariffOptimizerInput",
    "TariffOptimizerOutput",
    "UsageProfile",
    "TariffOption",
    "LoadShiftOpportunity",
    "TariffRecommendation",
    "SavingsAnalysis",
    "DemandChargeAnalysis",
    "RateSchedule",
    "RateType",
    "SeasonType",
    "PACK_SPEC",
    "calculate_tou_cost",
    "calculate_demand_charge",
    "calculate_optimal_shift",
    "calculate_annual_savings",
    "calculate_peak_shaving_benefit",
]

__version__ = "1.0.0"
__agent_id__ = "GL-078"
__agent_name__ = "TARIFFOPTIMIZER"
