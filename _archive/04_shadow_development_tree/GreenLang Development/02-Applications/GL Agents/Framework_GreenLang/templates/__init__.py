"""
GreenLang Framework - Agent Templates

Pre-built templates for creating new GreenLang agents.
Templates ensure consistency and compliance with Framework standards.
"""

from .base_agent import BaseAgentTemplate, AgentConfig
from .calculator_template import CalculatorTemplate
from .optimizer_template import OptimizerTemplate

__all__ = [
    "BaseAgentTemplate",
    "AgentConfig",
    "CalculatorTemplate",
    "OptimizerTemplate",
]
