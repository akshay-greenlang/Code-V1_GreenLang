"""GL-076: Carbon Market Trader Agent (CARBONTRADER)"""

from .agent import (
    CarbonMarketTraderAgent,
    CarbonMarketInput,
    EmissionAllowance,
    MarketPrice,
    ComplianceObligation,
    TradingLimits,
    MarketConditions,
    CarbonMarketOutput,
    TradingRecommendation,
    PortfolioPosition,
    RiskAssessment,
    ComplianceStatus,
    ProvenanceRecord,
    AllowanceType,
    TradingAction,
    RiskLevel,
    ComplianceState,
    PACK_SPEC,
)

from .formulas import (
    calculate_portfolio_value,
    calculate_position_risk,
    calculate_compliance_gap,
    calculate_optimal_position,
    calculate_var_monte_carlo,
    calculate_expected_shortfall,
    calculate_sharpe_ratio,
)

__all__ = [
    "CarbonMarketTraderAgent",
    "CarbonMarketInput",
    "EmissionAllowance",
    "MarketPrice",
    "ComplianceObligation",
    "TradingLimits",
    "MarketConditions",
    "CarbonMarketOutput",
    "TradingRecommendation",
    "PortfolioPosition",
    "RiskAssessment",
    "ComplianceStatus",
    "ProvenanceRecord",
    "AllowanceType",
    "TradingAction",
    "RiskLevel",
    "ComplianceState",
    "PACK_SPEC",
    "calculate_portfolio_value",
    "calculate_position_risk",
    "calculate_compliance_gap",
    "calculate_optimal_position",
    "calculate_var_monte_carlo",
    "calculate_expected_shortfall",
    "calculate_sharpe_ratio",
]

__version__ = "1.0.0"
__agent_id__ = "GL-076"
__agent_name__ = "CARBONTRADER"
