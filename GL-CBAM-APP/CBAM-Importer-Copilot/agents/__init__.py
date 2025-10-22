"""
CBAM Importer Copilot - AI Agents

This package contains the three core agents for CBAM compliance reporting:

1. ShipmentIntakeAgent_AI - Data ingestion, validation, and enrichment
2. EmissionsCalculatorAgent_AI - Emissions calculation with ZERO HALLUCINATION
3. ReportingPackagerAgent_AI - Report generation and packaging

All agents follow a tool-first architecture with deterministic calculations.

Version: 1.0.0
Author: GreenLang CBAM Team
"""

from .shipment_intake_agent import ShipmentIntakeAgent
from .emissions_calculator_agent import EmissionsCalculatorAgent
from .reporting_packager_agent import ReportingPackagerAgent

__version__ = "1.0.0"
__all__ = [
    "ShipmentIntakeAgent",
    "EmissionsCalculatorAgent",
    "ReportingPackagerAgent"
]
