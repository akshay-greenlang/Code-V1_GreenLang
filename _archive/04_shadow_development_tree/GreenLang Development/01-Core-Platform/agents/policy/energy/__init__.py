# -*- coding: utf-8 -*-
"""
GreenLang Policy Energy Sector Agents
======================================

Energy sector policy compliance and regulatory intelligence agents.

Agents:
    GL-POL-ENE-001
"""

from greenlang.agents.policy.energy.agents import (
    EnergyRegulatoryComplianceAgent,
)

__all__ = [
    "EnergyRegulatoryComplianceAgent",
]

AGENT_REGISTRY = {
    "GL-POL-ENE-001": EnergyRegulatoryComplianceAgent,
}
