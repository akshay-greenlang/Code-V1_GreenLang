"""
GL-006 HEATRECLAIM - Heat Recovery Maximizer Agent

Enterprise-grade heat recovery optimization agent for industrial facilities.
Implements pinch analysis, heat exchanger network (HEN) synthesis, exergy
analysis, and multi-objective optimization with full provenance tracking.

Agent ID: GL-006
Codename: HEATRECLAIM
Domain: Heat Recovery
Class: Optimizer
Status: Production

Standards Compliance:
- Zero-hallucination calculations
- Deterministic, reproducible outputs
- SHA-256 provenance tracking
- ASME/ISO thermal standards

Copyright (c) 2025 GreenLang. All rights reserved.
"""

__version__ = "1.0.0"
__agent_id__ = "GL-006"
__codename__ = "HEATRECLAIM"
__agent_name__ = "HeatRecoveryMaximizer"

from .core.orchestrator import HeatReclaimOrchestrator
from .core.schemas import (
    HeatStream,
    HeatExchanger,
    HENDesign,
    PinchAnalysisResult,
    OptimizationRequest,
    OptimizationResult,
)

__all__ = [
    "HeatReclaimOrchestrator",
    "HeatStream",
    "HeatExchanger",
    "HENDesign",
    "PinchAnalysisResult",
    "OptimizationRequest",
    "OptimizationResult",
    "__version__",
    "__agent_id__",
    "__codename__",
    "__agent_name__",
]
