# -*- coding: utf-8 -*-
"""
GreenLang Operations Energy Sector Agents
==========================================

Energy operations optimization agents for grid management
and renewable integration.

Agents:
    GL-OPS-ENE-001
"""

from greenlang.agents.operations.energy.agents import (
    GridOperationsOptimizerAgent,
)

__all__ = [
    "GridOperationsOptimizerAgent",
]

AGENT_REGISTRY = {
    "GL-OPS-ENE-001": GridOperationsOptimizerAgent,
}
