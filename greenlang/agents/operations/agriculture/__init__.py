# -*- coding: utf-8 -*-
"""
GreenLang Operations Agriculture Sector Agents
===============================================

Agriculture operations optimization agents for farm management,
irrigation control, and resource efficiency.

Agents:
    GL-OPS-AGR-001 to GL-OPS-AGR-004
"""

from greenlang.agents.operations.agriculture.agents import (
    FarmOperationsOptimizerAgent,
    IrrigationControlAgent,
    CropMonitoringAgent,
    LivestockMonitoringAgent,
)

__all__ = [
    "FarmOperationsOptimizerAgent",
    "IrrigationControlAgent",
    "CropMonitoringAgent",
    "LivestockMonitoringAgent",
]

AGENT_REGISTRY = {
    "GL-OPS-AGR-001": FarmOperationsOptimizerAgent,
    "GL-OPS-AGR-002": IrrigationControlAgent,
    "GL-OPS-AGR-003": CropMonitoringAgent,
    "GL-OPS-AGR-004": LivestockMonitoringAgent,
}
