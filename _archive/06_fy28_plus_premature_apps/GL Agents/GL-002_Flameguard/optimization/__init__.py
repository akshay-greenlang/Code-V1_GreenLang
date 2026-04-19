"""
GL-002 FLAMEGUARD - Optimization Module

Contains combustion optimizer, O2 trim controller, and excess air control.
"""

from .combustion_optimizer import CombustionOptimizer, LoadDispatchResult
from .o2_trim_controller import O2TrimController, TrimSetpoint

__all__ = [
    "CombustionOptimizer",
    "LoadDispatchResult",
    "O2TrimController",
    "TrimSetpoint",
]
