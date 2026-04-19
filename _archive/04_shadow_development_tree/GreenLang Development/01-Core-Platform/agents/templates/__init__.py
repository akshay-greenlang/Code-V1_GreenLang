# -*- coding: utf-8 -*-
"""
GreenLang Agent Templates
Reusable Agent Patterns for Sustainability Applications

Common agent templates that can be extended by applications:
- IntakeAgent: Multi-format data ingestion with validation
- CalculatorAgent: Zero-hallucination calculations with provenance
- ReportingAgent: Multi-format export with compliance checking

Version: 1.0.0
"""

from greenlang.agents.templates.intake_agent import IntakeAgent, IntakeResult, DataFormat
from greenlang.agents.templates.calculator_agent import CalculatorAgent, CalculationResult
from greenlang.agents.templates.reporting_agent import ReportingAgent, ReportResult, ReportFormat

__all__ = [
    "IntakeAgent",
    "IntakeResult",
    "DataFormat",
    "CalculatorAgent",
    "CalculationResult",
    "ReportingAgent",
    "ReportResult",
    "ReportFormat",
]
