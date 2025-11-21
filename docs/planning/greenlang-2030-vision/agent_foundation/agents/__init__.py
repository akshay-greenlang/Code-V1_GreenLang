# -*- coding: utf-8 -*-
"""
GreenLang Specialized Agents Package.

This package contains production-ready implementations of specialized agents
that demonstrate the core patterns of the GreenLang Agent Foundation.

These agents serve as templates and examples for the factory pattern that
will generate thousands of domain-specific agents.

Agent Types:
    - CalculatorAgent: Zero-hallucination deterministic calculations
    - ComplianceAgent: Regulatory compliance checking (CSRD, CBAM, EUDR, SB253)
    - IntegratorAgent: ERP and external system integration
    - ReporterAgent: Multi-format report generation
    - CoordinatorAgent: Swarm orchestration and management
    - WorkerAgent: Distributed task execution
    - MonitorAgent: System health and performance monitoring

Example:
    >>> from agents import CalculatorAgent, AgentConfig
    >>> config = AgentConfig(name="carbon_calculator", version="1.0.0")
    >>> agent = CalculatorAgent(config)
    >>> await agent.initialize()
    >>> result = await agent.execute(calculation_data)
"""

from .calculator_agent import CalculatorAgent, CalculationInput, CalculationOutput
from .compliance_agent import ComplianceAgent, ComplianceInput, ComplianceOutput
from .integrator_agent import IntegratorAgent, IntegrationInput, IntegrationOutput
from .reporter_agent import ReporterAgent, ReportInput, ReportOutput
from .coordinator_agent import CoordinatorAgent, CoordinationInput, CoordinationOutput
from .worker_agent import WorkerAgent, WorkerInput, WorkerOutput
from .monitor_agent import MonitorAgent, MonitorInput, MonitorOutput

__all__ = [
    # Calculator Agent
    "CalculatorAgent",
    "CalculationInput",
    "CalculationOutput",

    # Compliance Agent
    "ComplianceAgent",
    "ComplianceInput",
    "ComplianceOutput",

    # Integrator Agent
    "IntegratorAgent",
    "IntegrationInput",
    "IntegrationOutput",

    # Reporter Agent
    "ReporterAgent",
    "ReportInput",
    "ReportOutput",

    # Coordinator Agent
    "CoordinatorAgent",
    "CoordinationInput",
    "CoordinationOutput",

    # Worker Agent
    "WorkerAgent",
    "WorkerInput",
    "WorkerOutput",

    # Monitor Agent
    "MonitorAgent",
    "MonitorInput",
    "MonitorOutput",
]

__version__ = "1.0.0"