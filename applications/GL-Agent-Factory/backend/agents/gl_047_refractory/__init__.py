"""
GL-047 Refractory Agent Package

Refractory condition monitoring and remaining life estimation.
"""

from .agent import RefractoryAgent, RefractoryInput, RefractoryOutput, PACK_SPEC

__all__ = [
    "RefractoryAgent",
    "RefractoryInput",
    "RefractoryOutput",
    "PACK_SPEC",
]
