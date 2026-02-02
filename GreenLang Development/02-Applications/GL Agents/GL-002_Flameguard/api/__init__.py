"""
GL-002 FLAMEGUARD - API Module

REST API, GraphQL, and gRPC interfaces for boiler optimization.
"""

from .rest_api import (
    create_app,
    FlameguardAPI,
)
from .schemas import (
    BoilerStatusResponse,
    OptimizationRequest,
    OptimizationResponse,
    EfficiencyResponse,
    EmissionsResponse,
    SafetyStatusResponse,
    SetpointCommand,
)

__all__ = [
    "create_app",
    "FlameguardAPI",
    "BoilerStatusResponse",
    "OptimizationRequest",
    "OptimizationResponse",
    "EfficiencyResponse",
    "EmissionsResponse",
    "SafetyStatusResponse",
    "SetpointCommand",
]
