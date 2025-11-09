"""
GreenLang Agent Templates
Reusable Agent Patterns for Sustainability Applications

Common agent templates that can be extended by applications:
- IntakeAgent: Multi-format data ingestion with validation
- CalculatorAgent: Zero-hallucination calculations with provenance
- ReportingAgent: Multi-format export with compliance checking

Version: 1.0.0
"""

from greenlang.agents.templates.intake_agent import IntakeAgent, IntakeResult
from greenlang.agents.templates.calculator_agent import CalculatorAgent, CalculationResult
from greenlang.agents.templates.reporting_agent import ReportingAgent, ReportResult

__all__ = [
    "IntakeAgent",
    "IntakeResult",
    "CalculatorAgent",
    "CalculationResult",
    "ReportingAgent",
    "ReportResult",
]
