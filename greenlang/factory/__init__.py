"""
GreenLang Agent Factory

LLM-powered code generation system for GreenLang agents.

This module provides automated agent generation from AgentSpec specifications:
- 10× productivity improvement (10 min vs 2 weeks per agent)
- Tool-first architecture pattern
- Deterministic execution (temperature=0, seed=42)
- Comprehensive validation (syntax, type, lint, test)
- Multi-step generation pipeline
- Feedback loop for iterative refinement

Quick Start:
    >>> from greenlang.factory import AgentFactory
    >>> from greenlang.specs import from_yaml
    >>>
    >>> factory = AgentFactory()
    >>> spec = from_yaml("specs/my_agent.yaml")
    >>> result = await factory.generate_agent(spec)
    >>> print(f"Generated in {result.duration_seconds}s for ${result.total_cost_usd}")

Author: GreenLang Framework Team
Date: October 2025
"""

from .agent_factory import AgentFactory, GenerationResult
from .validators import CodeValidator, ValidationResult, ValidationError, DeterminismVerifier
from .templates import AgentTemplate, CodeTemplates, TestTemplates, DocumentationTemplates
from .prompts import AgentFactoryPrompts

__version__ = "0.1.0"

__all__ = [
    "AgentFactory",
    "GenerationResult",
    "CodeValidator",
    "ValidationResult",
    "ValidationError",
    "DeterminismVerifier",
    "AgentTemplate",
    "CodeTemplates",
    "TestTemplates",
    "DocumentationTemplates",
    "AgentFactoryPrompts",
]
