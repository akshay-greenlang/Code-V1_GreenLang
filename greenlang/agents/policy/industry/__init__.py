# -*- coding: utf-8 -*-
"""
GreenLang Policy Industry Sector Agents
========================================

Industry sector policy compliance and regulatory intelligence agents.

Agents:
    GL-POL-IND-001
"""

from greenlang.agents.policy.industry.agents import (
    IndustryRegulatoryComplianceAgent,
)

__all__ = [
    "IndustryRegulatoryComplianceAgent",
]

AGENT_REGISTRY = {
    "GL-POL-IND-001": IndustryRegulatoryComplianceAgent,
}
