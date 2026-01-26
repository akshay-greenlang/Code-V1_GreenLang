# -*- coding: utf-8 -*-
"""
GreenLang Water Decarbonization Agents
======================================

This package provides agents for water sector decarbonization planning
and implementation.

Agents:
    GL-DECARB-WAT-001: WaterSystemDecarbonizationAgent - Water sector decarbonization

All agents follow the GreenLang standard patterns with:
    - Zero-hallucination calculations
    - Complete provenance tracking
    - Deterministic outputs
"""

from greenlang.agents.decarbonization.water.system_decarbonization import (
    WaterSystemDecarbonizationAgent,
    DecarbonizationInput,
    DecarbonizationOutput,
    DecarbonizationPathway,
    InterventionOption,
)

__all__ = [
    # Water System Decarbonization (GL-DECARB-WAT-001)
    "WaterSystemDecarbonizationAgent",
    "DecarbonizationInput",
    "DecarbonizationOutput",
    "DecarbonizationPathway",
    "InterventionOption",
]
