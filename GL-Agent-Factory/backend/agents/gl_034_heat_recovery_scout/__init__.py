"""GL-034: Heat Recovery Scout Agent (HEAT-RECOVERY-SCOUT)"""

from .agent import (
    HeatRecoveryScoutAgent,
    HeatRecoveryScoutInput,
    HeatRecoveryScoutOutput,
    ExhaustStream,
    HeatDemand,
    UtilityCosts,
    RecoveryOpportunity,
    PACK_SPEC,
)

__all__ = [
    "HeatRecoveryScoutAgent",
    "HeatRecoveryScoutInput",
    "HeatRecoveryScoutOutput",
    "ExhaustStream",
    "HeatDemand",
    "UtilityCosts",
    "RecoveryOpportunity",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-034"
__agent_name__ = "HEAT-RECOVERY-SCOUT"
