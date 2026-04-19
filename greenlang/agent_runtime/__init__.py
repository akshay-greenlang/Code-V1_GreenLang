# -*- coding: utf-8 -*-
"""
GreenLang Agent Runtime — v3 L3 Intelligence Layer
====================================================

Unified entry point for the GreenLang agent framework.  This module
re-exports the canonical base classes, mixins, and lifecycle utilities
under a single, clean namespace.

v3 Canonical Hierarchy
----------------------

**Sync agents** (most agents):

    BaseAgent                          # lifecycle, metrics, provenance
    ├── DeterministicAgent             # zero-hallucination (CRITICAL PATH)
    ├── ReasoningAgent                 # full LLM reasoning (RECOMMENDATION PATH)
    └── InsightAgent                   # hybrid deterministic + AI (INSIGHT PATH)

**Typed agents** (schema-validated):

    AgentSpecV2Base[InT, OutT]         # generic typing + pack.yaml validation
    + DeterministicMixin               # zero-hallucination mixin
    + ReasoningMixin                   # AI reasoning mixin
    + InsightMixin                     # hybrid mixin

**Async agents** (high-throughput I/O):

    AsyncAgentBase[InT, OutT]          # async/await lifecycle

**LLM-enhanced agents** (explanation + recommendation):

    IntelligentAgentBase               # BaseAgent + LLM provider integration

Quick Start
-----------

.. code-block:: python

    from greenlang.agent_runtime import BaseAgent, AgentResult

    class MyAgent(BaseAgent):
        def execute(self, input_data):
            return AgentResult(success=True, data={"answer": 42})

    agent = MyAgent()
    result = agent.run({"question": "everything"})

Author: GreenLang Platform Team
Version: 0.1.0
"""

from __future__ import annotations

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Canonical base class + result types (183+ importers)
# ---------------------------------------------------------------------------
from greenlang.agents.base import (
    BaseAgent,
    AgentConfig,
    AgentMetrics,
    AgentResult,
    StatsTracker,
)

# ---------------------------------------------------------------------------
# Intelligence Paradox base classes (62 importers)
# ---------------------------------------------------------------------------
from greenlang.agents.base_agents import (
    DeterministicAgent,
    ReasoningAgent,
    InsightAgent,
    AuditEntry,
)

# ---------------------------------------------------------------------------
# AgentSpec v2 typed base + category mixins
# ---------------------------------------------------------------------------
from greenlang.agents.agentspec_v2_base import (
    AgentSpecV2Base,
    AgentExecutionContext,
    AgentLifecycleState,
)
from greenlang.agents.mixins import (
    DeterministicMixin,
    ReasoningMixin,
    InsightMixin,
    get_category_mixin,
    validate_mixin_usage,
)

# ---------------------------------------------------------------------------
# Async agent base
# ---------------------------------------------------------------------------
from greenlang.agents.async_agent_base import (
    AsyncAgentBase,
    AsyncAgentExecutionContext,
    AsyncAgentLifecycleState,
)

# ---------------------------------------------------------------------------
# Intelligent (LLM-enhanced) agent base
# ---------------------------------------------------------------------------
from greenlang.agents.intelligent_base import (
    IntelligentAgentBase,
    IntelligentAgentConfig,
    IntelligenceLevel,
)

__all__ = [
    # Canonical base
    "BaseAgent",
    "AgentConfig",
    "AgentMetrics",
    "AgentResult",
    "StatsTracker",
    # Intelligence Paradox
    "DeterministicAgent",
    "ReasoningAgent",
    "InsightAgent",
    "AuditEntry",
    # AgentSpec v2
    "AgentSpecV2Base",
    "AgentExecutionContext",
    "AgentLifecycleState",
    # Category mixins
    "DeterministicMixin",
    "ReasoningMixin",
    "InsightMixin",
    "get_category_mixin",
    "validate_mixin_usage",
    # Async
    "AsyncAgentBase",
    "AsyncAgentExecutionContext",
    "AsyncAgentLifecycleState",
    # Intelligent
    "IntelligentAgentBase",
    "IntelligentAgentConfig",
    "IntelligenceLevel",
]
