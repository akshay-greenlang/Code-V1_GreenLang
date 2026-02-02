"""
GL-003 UNIFIEDSTEAM - Optimization Module

Provides optimization components for steam system operations:
- Desuperheater spray water optimization
- Condensate recovery optimization
- Steam network-wide optimization
- Trap maintenance optimization
- Recommendation packaging and presentation

Optimization Objectives:
1. Minimize total steam generation and distribution losses
2. Maintain steam quality/temperature targets with minimal spray-water injection
3. Maximize condensate return ratio
4. Reduce unplanned downtime via early trap/equipment issue detection
5. Minimize cost and emissions (fuel + makeup water + treatment + maintenance)

All optimizers use deterministic calculations (zero-hallucination approach)
and respect safety constraints - safety is paramount.
"""

# Constraints
from .constraints import (
    # Enums
    ConstraintStatus,
    ConstraintSeverity,
    # Dataclasses
    ConstraintViolation,
    ConstraintCheckResult,
    # Safety Constraints
    PressureConstraints,
    TemperatureConstraints,
    QualityConstraints,
    RateLimitConstraints,
    SafetyConstraints,
    # Equipment Constraints
    ValveConstraints,
    NozzleConstraints,
    TrapConstraints,
    PumpConstraints,
    EquipmentConstraints,
    # Operational Constraints
    ProductionConstraints,
    MaintenanceWindow,
    MaintenanceConstraints,
    OperationalConstraints,
    # Uncertainty Constraints
    UncertaintyConstraints,
    # Master Container
    SteamSystemConstraints,
)

# Desuperheater Optimizer
from .desuperheater_optimizer import (
    DesuperheaterState,
    TargetConstraints,
    Setpoint,
    SprayOptimizationResult,
    DesuperheaterOptimizer,
)

# Condensate Recovery Optimizer
from .condensate_optimizer import (
    CondensateSourceType,
    CondensateSource,
    CondensateReceiver,
    CondensatePump,
    NetworkTopology,
    FlashConstraints,
    RoutingRecommendation,
    RoutingOptimization,
    PressureSetpoint,
    RecoveryOpportunity,
    PrioritizedList,
    CondensateRecoveryOptimizer,
)

# Steam Network Optimizer
from .steam_network_optimizer import (
    HeaderType,
    BoilerState,
    HeaderState,
    PRVState,
    DemandForecast,
    NetworkModel,
    BoilerLoadAllocation,
    LoadAllocationResult,
    HeaderOptimization,
    PRVOptimization,
    LossMinimizationResult,
    SteamNetworkOptimizer,
)

# Trap Maintenance Optimizer
from .trap_maintenance_optimizer import (
    TrapType,
    TrapStatus,
    TrapCriticality,
    TrapData,
    TrapFleet,
    FailurePrediction,
    DowntimeConstraint,
    InspectionTask,
    InspectionSchedule,
    ReplacementTask,
    ReplacementPriority,
    SparePartItem,
    SparesPlan,
    TrapMaintenanceOptimizer,
)

# Recommendation Engine
from .recommendation_engine import (
    RecommendationType,
    RecommendationPriority,
    RiskCategory,
    OperatorPreference,
    BenefitEstimate,
    RiskAssessment,
    VerificationPlan,
    EscalationPath,
    Recommendation,
    RankedList,
    RecommendationEngine,
)

__all__ = [
    # === Constraints ===
    # Enums
    "ConstraintStatus",
    "ConstraintSeverity",
    # Dataclasses
    "ConstraintViolation",
    "ConstraintCheckResult",
    # Safety
    "PressureConstraints",
    "TemperatureConstraints",
    "QualityConstraints",
    "RateLimitConstraints",
    "SafetyConstraints",
    # Equipment
    "ValveConstraints",
    "NozzleConstraints",
    "TrapConstraints",
    "PumpConstraints",
    "EquipmentConstraints",
    # Operational
    "ProductionConstraints",
    "MaintenanceWindow",
    "MaintenanceConstraints",
    "OperationalConstraints",
    # Uncertainty
    "UncertaintyConstraints",
    # Master
    "SteamSystemConstraints",
    # === Desuperheater ===
    "DesuperheaterState",
    "TargetConstraints",
    "Setpoint",
    "SprayOptimizationResult",
    "DesuperheaterOptimizer",
    # === Condensate ===
    "CondensateSourceType",
    "CondensateSource",
    "CondensateReceiver",
    "CondensatePump",
    "NetworkTopology",
    "FlashConstraints",
    "RoutingRecommendation",
    "RoutingOptimization",
    "PressureSetpoint",
    "RecoveryOpportunity",
    "PrioritizedList",
    "CondensateRecoveryOptimizer",
    # === Steam Network ===
    "HeaderType",
    "BoilerState",
    "HeaderState",
    "PRVState",
    "DemandForecast",
    "NetworkModel",
    "BoilerLoadAllocation",
    "LoadAllocationResult",
    "HeaderOptimization",
    "PRVOptimization",
    "LossMinimizationResult",
    "SteamNetworkOptimizer",
    # === Trap Maintenance ===
    "TrapType",
    "TrapStatus",
    "TrapCriticality",
    "TrapData",
    "TrapFleet",
    "FailurePrediction",
    "DowntimeConstraint",
    "InspectionTask",
    "InspectionSchedule",
    "ReplacementTask",
    "ReplacementPriority",
    "SparePartItem",
    "SparesPlan",
    "TrapMaintenanceOptimizer",
    # === Recommendation Engine ===
    "RecommendationType",
    "RecommendationPriority",
    "RiskCategory",
    "OperatorPreference",
    "BenefitEstimate",
    "RiskAssessment",
    "VerificationPlan",
    "EscalationPath",
    "Recommendation",
    "RankedList",
    "RecommendationEngine",
]
