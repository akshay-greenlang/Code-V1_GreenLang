"""GL-036: CHP Coordinator Agent"""

from .agent import (
    CHPCoordinatorAgent,
    CHPCoordinatorInput,
    CHPCoordinatorOutput,
    PACK_SPEC,
)

__all__ = [
    "CHPCoordinatorAgent",
    "CHPCoordinatorInput",
    "CHPCoordinatorOutput",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-036"
__agent_name__ = "CHP-COORDINATOR"
