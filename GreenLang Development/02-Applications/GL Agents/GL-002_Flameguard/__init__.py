"""
GL-002 FlameGuard - BoilerEfficiencyOptimizer
Master Orchestration for Multi-Boiler Combustion Optimization

Agent ID: GL-002
Capability: FLAMEGUARD / Boiler Efficiency Optimizer
Priority: P0 (High)
Business Impact: 3-8% fuel savings, $500K-$2M annual per boiler house
Target Release: Q4 2025
Status: Implemented (baseline) - scaling and hardening

This package provides the master orchestrator for boiler combustion optimization,
coordinating multi-boiler load dispatch, O2 trim control, and emissions monitoring
to maximize efficiency while maintaining strict safety and emissions compliance.

Key Features:
    - MILP optimization for multi-boiler load dispatch
    - Adaptive O2 trim with CO cross-limiting
    - Real-time combustion efficiency tracking
    - Predictive maintenance integration
    - Emissions monitoring and reporting (NOx, CO, CO2)
    - Kafka event streaming for real-time dashboards
    - SHAP/LIME explainability for recommendations
    - SIL-3 safety compliance with burner management

Reference Standards:
    - ASME PTC 4 (Fired Steam Generators)
    - NFPA 85 (Boiler and Combustion Systems)
    - EPA 40 CFR 60 (New Source Performance Standards)
    - IEC 61511 (Safety Instrumented Systems)
    - API 560 (Fired Heaters for General Refinery Service)

Performance Targets:
    - Efficiency improvement: 3-8%
    - O2 setpoint optimization: Within 0.5% accuracy
    - CO breakthrough detection: <10ms response
    - Emissions compliance: 100% regulatory adherence
    - Safety response: <100ms for critical events

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__agent_id__ = "GL-002"
__capability__ = "FLAMEGUARD"

# Agent metadata for discovery and registration
__agent_metadata__ = {
    "agent_id": __agent_id__,
    "name": "FlameGuard BoilerEfficiencyOptimizer",
    "capability": __capability__,
    "version": __version__,
    "category": "process_heat",
    "subcategory": "combustion_optimization",
    "priority": "P0",
    "sil_level": 3,
    "interfaces": ["OPC-UA", "MQTT", "Kafka", "REST", "GraphQL"],
    "dependencies": ["GL-001"],  # Requires ThermalCommand orchestration
    "standards": ["ASME PTC 4", "NFPA 85", "EPA 40 CFR 60", "IEC 61511"],
}

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .boiler_efficiency_orchestrator import (
        FlameGuardOrchestrator,
        FlameGuardConfig,
        OptimizationResult,
        EfficiencyResult,
        SetpointRecommendation,
        EmissionsReport,
        SafetyResponse,
        EfficiencyReport,
    )
    from .optimization.combustion_optimizer import (
        CombustionOptimizer,
        BoilerModel,
        LoadDispatchResult,
    )
    from .optimization.o2_trim_controller import (
        O2TrimController,
        TrimSetpoint,
        COBreakthroughEvent,
    )

__all__ = [
    # Orchestrator
    "FlameGuardOrchestrator",
    "FlameGuardConfig",
    # Results
    "OptimizationResult",
    "EfficiencyResult",
    "SetpointRecommendation",
    "EmissionsReport",
    "SafetyResponse",
    "EfficiencyReport",
    # Optimization
    "CombustionOptimizer",
    "BoilerModel",
    "LoadDispatchResult",
    # O2 Control
    "O2TrimController",
    "TrimSetpoint",
    "COBreakthroughEvent",
    # Metadata
    "__version__",
    "__agent_id__",
    "__capability__",
    "__agent_metadata__",
]

