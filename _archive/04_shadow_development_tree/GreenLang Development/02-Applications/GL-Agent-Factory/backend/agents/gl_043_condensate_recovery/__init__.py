"""
GL-043: Condensate-Reclaim Agent (CONDENSATE-RECLAIM)

This package provides the Condensate-Reclaim Agent for condensate recovery
monitoring and optimization in steam systems.

Key Features:
- Condensate recovery rate monitoring
- Water and energy savings calculation
- Flash steam recovery potential assessment
- System efficiency evaluation
- Complete SHA-256 provenance tracking

Standards Compliance:
- ASME PTC 19.1: Test Uncertainty
- ASHRAE Handbook - HVAC Systems and Equipment
- DOE Steam Best Practices

Example Usage:
    >>> from backend.agents.gl_043_condensate_recovery import (
    ...     CondensateReclaimAgent,
    ...     CondensateReclaimInput,
    ... )
    >>> agent = CondensateReclaimAgent()
    >>> result = agent.run(input_data)
    >>> print(f"Recovery Rate: {result.recovery_metrics.recovery_rate_percent}%")
"""

from .agent import (
    CondensateReclaimAgent,
    CondensateReclaimInput,
    CondensateReclaimOutput,
    CondensateReturn,
    MakeupWater,
    SteamConditions,
    RecoveryMetrics,
    EnergySavings,
    WaterSavings,
    FlashSteamPotential,
    PACK_SPEC,
)

__all__ = [
    "CondensateReclaimAgent",
    "CondensateReclaimInput",
    "CondensateReclaimOutput",
    "CondensateReturn",
    "MakeupWater",
    "SteamConditions",
    "RecoveryMetrics",
    "EnergySavings",
    "WaterSavings",
    "FlashSteamPotential",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-043"
__agent_name__ = "CONDENSATE-RECLAIM"
