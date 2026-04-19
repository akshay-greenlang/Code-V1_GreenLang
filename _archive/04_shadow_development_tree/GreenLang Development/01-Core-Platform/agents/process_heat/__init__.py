# -*- coding: utf-8 -*-
"""
GreenLang Process Heat Agents Module
=====================================

This module provides specialized agents for industrial process heat applications
including combustion diagnostics, efficiency optimization, and regulatory compliance.

Agent Categories:
- GL-005: COMBUSENSE - Combustion Diagnostics (diagnostics only, no control)
- GL-017: CONDENSYNC - Condenser Optimization (HEI Standards, cooling tower)
- GL-018: Unified Combustion Control (future - control execution)

Architecture:
    Process Heat agents follow the GreenLang AgentSpec v2 pattern with
    DeterministicMixin for zero-hallucination calculations and full audit trails.

Author: GreenLang Framework Team
Status: Production Ready
"""

from greenlang.agents.process_heat.shared.base_agent import (
    BaseProcessHeatAgent,
    AgentConfig as ProcessHeatConfig,
)

# Lazy import for agents to avoid circular dependencies
def __getattr__(name: str):
    """Lazy import process heat agents."""
    if name == "CombustionDiagnosticsAgent":
        from greenlang.agents.process_heat.gl_005_combustion_diagnostics import (
            CombustionDiagnosticsAgent,
        )
        return CombustionDiagnosticsAgent
    if name == "CondenserOptimizerAgent":
        from greenlang.agents.process_heat.gl_017_condenser_optimization import (
            CondenserOptimizerAgent,
        )
        return CondenserOptimizerAgent
    if name == "CondenserOptimizationConfig":
        from greenlang.agents.process_heat.gl_017_condenser_optimization import (
            CondenserOptimizationConfig,
        )
        return CondenserOptimizationConfig
    raise AttributeError(f"module 'greenlang.agents.process_heat' has no attribute '{name}'")


__all__ = [
    # Base classes
    "BaseProcessHeatAgent",
    "ProcessHeatConfig",
    # Agents (lazy loaded)
    "CombustionDiagnosticsAgent",
    "CondenserOptimizerAgent",
    "CondenserOptimizationConfig",
]
