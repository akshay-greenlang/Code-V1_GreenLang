"""
GL-033: Burner Balancer Agent (BURNER-BALANCER)

Multi-burner load balancing and air-fuel optimization for industrial combustion systems.
"""

from .agent import (
    BurnerBalancerAgent,
    BurnerBalancerInput,
    BurnerBalancerOutput,
    BurnerData,
    BurnerSetpoint,
    PACK_SPEC,
)
from .models import (
    BurnerType,
    FuelType,
    BurnerStatus,
    BalancingObjective,
)

__all__ = [
    "BurnerBalancerAgent",
    "BurnerBalancerInput",
    "BurnerBalancerOutput",
    "BurnerData",
    "BurnerSetpoint",
    "BurnerType",
    "FuelType",
    "BurnerStatus",
    "BalancingObjective",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-033"
__agent_name__ = "BURNER-BALANCER"
