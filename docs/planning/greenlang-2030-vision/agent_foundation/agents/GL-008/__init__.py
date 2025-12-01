# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER - SteamTrapInspector Agent

Automated detection and diagnosis of steam trap failures for industrial
steam systems. This agent provides comprehensive steam trap monitoring
including acoustic signature analysis, thermal differential analysis,
and IR imaging interpretation.

Agent ID: GL-008
Codename: TRAPCATCHER
Name: SteamTrapInspector
Domain: Steam Systems
Type: Monitor
Priority: P2
Market: $3B
Timeline: Q2 2026

Key Features:
- Acoustic signature analysis for ultrasonic leak detection
- Temperature differential analysis (inlet/outlet)
- IR thermal imaging interpretation
- Failed trap location identification
- Maintenance priority calculation
- Cost savings quantification
- Work order generation

Connected Systems:
- Steam trap monitoring systems
- Maintenance systems (CMMS)
- SCADA systems

Standards Compliance:
- ISO 6552: Automatic steam traps
- ASME B31.1: Power Piping
- ISO 7841: Automatic steam traps - Selection
- ASHRAE standards for steam system efficiency

Zero-Hallucination Guarantee:
All calculations are deterministic with bit-perfect reproducibility.
No LLM involved in any numeric calculation path.

Author: GreenLang Industrial Optimization Team
Date: December 2025
Version: 1.0.0
"""

from .steam_trap_orchestrator import (
    SteamTrapOrchestrator,
    SteamTrapConfig,
    TrapInspectionRequest,
    TrapInspectionResult,
    TrapType,
    TrapStatus,
    FailureMode,
    InspectionMethod,
    MaintenancePriority,
    TrapCondition,
)

__all__ = [
    # Orchestrator
    "SteamTrapOrchestrator",
    "SteamTrapConfig",
    # Request/Response
    "TrapInspectionRequest",
    "TrapInspectionResult",
    # Enums
    "TrapType",
    "TrapStatus",
    "FailureMode",
    "InspectionMethod",
    "MaintenancePriority",
    "TrapCondition",
]

__version__ = "1.0.0"
__agent_id__ = "GL-008"
__codename__ = "TRAPCATCHER"
