"""
GL-003 UNIFIEDSTEAM - Calculators Module

Zero-hallucination calculation engines for steam system optimization:
- Desuperheater spray water calculations
- Condensate recovery calculations
- Steam trap diagnostics
- Heat balance calculations
- Steam system KPIs

All calculators implement:
- Deterministic formulas with zero hallucination
- SHA-256 provenance hashing
- Uncertainty bounds where applicable
- ASME and engineering standards compliance
"""

from .desuperheater_calculator import (
    DesuperheaterCalculator,
    SprayRequirement,
    MaxCoolingResult,
    ErosionRiskAssessment,
    DesuperheaterInput,
)

from .condensate_calculator import (
    CondensateCalculator,
    FlashSteamResult,
    HeatRecoveryResult,
    LossSource,
    EconomicsResult,
    CondensateInput,
)

from .trap_diagnostics_calculator import (
    TrapDiagnosticsCalculator,
    TrapCondition,
    FailurePrediction,
    FailureMode,
    MaintenancePriorityList,
    TrapEconomics,
    TrapInput,
)

from .heat_balance_calculator import (
    HeatBalanceCalculator,
    HeaderBalanceResult,
    HeatDemandResult,
    HeatRateResult,
    DistributionLossResult,
    ReconciledBalance,
    HeaderData,
    UserData,
    NetworkTopology,
)

from .steam_kpi_calculator import (
    SteamKPICalculator,
    PerformanceMetrics,
    TrapHealthKPI,
    KPIDashboard,
    KPIInput,
)

from .enthalpy_balance_integration import (
    EnthalpyBalanceIntegrator,
    StreamState,
    ZoneBalance,
    PlantBalance,
    EnthalpyPoint,
    BalanceZone,
    StreamType,
    create_stream_from_pt,
    create_stream_from_ph,
)

__version__ = "1.0.0"
__author__ = "GL-CalculatorEngineer"

__all__ = [
    # Desuperheater Calculator
    "DesuperheaterCalculator",
    "SprayRequirement",
    "MaxCoolingResult",
    "ErosionRiskAssessment",
    "DesuperheaterInput",
    # Condensate Calculator
    "CondensateCalculator",
    "FlashSteamResult",
    "HeatRecoveryResult",
    "LossSource",
    "EconomicsResult",
    "CondensateInput",
    # Trap Diagnostics Calculator
    "TrapDiagnosticsCalculator",
    "TrapCondition",
    "FailurePrediction",
    "FailureMode",
    "MaintenancePriorityList",
    "TrapEconomics",
    "TrapInput",
    # Heat Balance Calculator
    "HeatBalanceCalculator",
    "HeaderBalanceResult",
    "HeatDemandResult",
    "HeatRateResult",
    "DistributionLossResult",
    "ReconciledBalance",
    "HeaderData",
    "UserData",
    "NetworkTopology",
    # Steam KPI Calculator
    "SteamKPICalculator",
    "PerformanceMetrics",
    "TrapHealthKPI",
    "KPIDashboard",
    "KPIInput",
    # Enthalpy Balance Integration
    "EnthalpyBalanceIntegrator",
    "StreamState",
    "ZoneBalance",
    "PlantBalance",
    "EnthalpyPoint",
    "BalanceZone",
    "StreamType",
    "create_stream_from_pt",
    "create_stream_from_ph",
]
