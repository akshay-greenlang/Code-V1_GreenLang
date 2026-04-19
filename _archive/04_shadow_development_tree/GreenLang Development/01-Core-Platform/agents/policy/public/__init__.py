# -*- coding: utf-8 -*-
"""
GreenLang Policy Public Sector Agents
======================================

Public sector policy compliance and regulatory intelligence agents.

Agents:
    GL-POL-PUB-001
"""

from greenlang.agents.policy.public.agents import (
    PublicSectorClimateComplianceAgent,
)

__all__ = [
    "PublicSectorClimateComplianceAgent",
]

AGENT_REGISTRY = {
    "GL-POL-PUB-001": PublicSectorClimateComplianceAgent,
}
