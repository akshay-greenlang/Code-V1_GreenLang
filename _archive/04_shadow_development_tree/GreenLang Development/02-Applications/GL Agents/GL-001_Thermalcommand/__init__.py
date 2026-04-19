"""
GL-001 ThermalCommand - ProcessHeatOrchestrator
Master Orchestration Specification and Execution

Agent ID: GL-001
Capability: THERMALCOMMAND / Process Heat Orchestrator
Priority: P0 (High)
Business Impact: $20B value at stake (portfolio estimate)
Target Release: Q4 2025
Status: Implemented (baseline) - scaling and hardening

This package provides the master orchestrator for all process heat operations,
coordinating heat generation, distribution, and consumption across the plant
to minimize total cost and emissions while maintaining production quality
and strict safety boundaries.

Key Features:
    - MILP optimization for load allocation
    - Cascade PID control integration
    - SIS safety boundary enforcement
    - CMMS integration for maintenance
    - OPC-UA, Kafka streaming integrations
    - GraphQL/gRPC APIs with webhooks
    - SHAP/LIME explainability
    - Uncertainty quantification
    - Comprehensive audit logging

Reference Standards:
    - ASME PTC 4.1 (Process Heat Performance)
    - ISO 50001:2018 (Energy Management)
    - IEC 61511 (Safety Instrumented Systems)
    - NFPA 86 (Ovens and Furnaces)
    - EPA 40 CFR 98 Subpart C (GHG Reporting)

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__agent_id__ = "GL-001"
__capability__ = "THERMALCOMMAND"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core.orchestrator import ThermalCommandOrchestrator
    from .core.config import OrchestratorConfig
    from .optimization.milp_optimizer import MILPLoadAllocator
    from .control.cascade_controller import CascadeController
    from .safety.boundary_engine import SafetyBoundaryEngine

__all__ = [
    "ThermalCommandOrchestrator",
    "OrchestratorConfig",
    "MILPLoadAllocator",
    "CascadeController",
    "SafetyBoundaryEngine",
    "__version__",
    "__agent_id__",
    "__capability__",
]
