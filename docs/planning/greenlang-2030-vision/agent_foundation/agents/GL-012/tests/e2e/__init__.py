# -*- coding: utf-8 -*-
"""
GL-012 STEAMQUAL End-to-End Tests Package.

This package contains comprehensive end-to-end tests for the GL-012
SteamQualityController agent, validating complete workflows from
sensor input to control output.

Test Categories:
    - Complete quality control workflows
    - Real-time control loop validation
    - Fault tolerance scenarios
    - Multi-header coordination

Standards Compliance:
    - ASME PTC 4.4 - Gas Turbine Heat Recovery Steam Generators
    - ISA-88 - Batch Control
    - IEC 61131-3 - Programmable Controllers

Author: GreenLang Industrial Optimization Team
Agent ID: GL-012
Version: 1.0.0
"""

__version__ = "1.0.0"
__agent_id__ = "GL-012"
__codename__ = "STEAMQUAL"

E2E_MARKERS = [
    "e2e",
    "workflow",
    "control_loop",
    "fault_tolerance",
    "multi_header",
    "slow",
]
