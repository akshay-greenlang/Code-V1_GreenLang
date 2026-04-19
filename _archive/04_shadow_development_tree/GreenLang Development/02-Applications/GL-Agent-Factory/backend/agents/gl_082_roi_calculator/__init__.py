"""GL-082: ROI Calculator Agent (ROICALCULATOR)"""

from .agent import (
    ROICalculatorAgent,
    ROICalculatorInput,
    ROICalculatorOutput,
    InvestmentCost,
    CashFlow,
    ROIMetrics,
    SensitivityResult,
    InvestmentType,
    PACK_SPEC,
)

from .formulas import (
    calculate_npv,
    calculate_irr,
    calculate_payback_period,
    calculate_roi,
    calculate_mirr,
    run_sensitivity_analysis,
)

__all__ = [
    "ROICalculatorAgent",
    "ROICalculatorInput",
    "ROICalculatorOutput",
    "InvestmentCost",
    "CashFlow",
    "ROIMetrics",
    "SensitivityResult",
    "InvestmentType",
    "PACK_SPEC",
    "calculate_npv",
    "calculate_irr",
    "calculate_payback_period",
    "calculate_roi",
    "calculate_mirr",
    "run_sensitivity_analysis",
]

__version__ = "1.0.0"
__agent_id__ = "GL-082"
__agent_name__ = "ROICALCULATOR"
