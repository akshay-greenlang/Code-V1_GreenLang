# -*- coding: utf-8 -*-
"""
CSRD Agents Module
==================

Six specialized agents for CSRD/ESRS reporting:

1. IntakeAgent - Data ingestion & validation (1,000+ records/sec)
2. MaterialityAgent - AI-powered double materiality assessment
3. CalculatorAgent - Zero-hallucination metric calculations
4. AggregatorAgent - Multi-framework integration (TCFD/GRI/SASB)
5. ReportingAgent - XBRL/ESEF report generation
6. AuditAgent - 200+ compliance rule validation

All agents follow GreenLang framework patterns.
"""

from .intake_agent import IntakeAgent
from .materiality_agent import MaterialityAgent
from .calculator_agent import CalculatorAgent
from .aggregator_agent import AggregatorAgent
from .reporting_agent import ReportingAgent
from .audit_agent import AuditAgent

__all__ = [
    "IntakeAgent",
    "MaterialityAgent",
    "CalculatorAgent",
    "AggregatorAgent",
    "ReportingAgent",
    "AuditAgent",
]
