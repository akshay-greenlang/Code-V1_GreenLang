# GreenLang Unified Agent Architecture (v2.0)
## Technical Design Specification

**Document Version:** 1.0.0
**Date:** October 25, 2025
**Author:** Head of AI & Climate Intelligence
**Target Release:** Q2 2026 (GreenLang v2.0.0)
**Status:** APPROVED ARCHITECTURE

---

## EXECUTIVE SUMMARY

This document specifies the **Unified Agent Architecture** for GreenLang v2.0, which eliminates user confusion by merging Calculator and Assistant agents into intelligent, auto-detecting unified agents.

**Key Innovation:** Single agent automatically chooses optimal execution path (fast calculator vs AI assistant) based on input type.

**Business Impact:**
- ✅ Eliminates user confusion (1 agent instead of 2)
- ✅ Maintains all performance benefits (fast path for APIs)
- ✅ Maintains all AI benefits (smart path for humans)
- ✅ 100% backward compatible (old names still work)
- ✅ Reduces documentation burden by 50%

---

## PROBLEM STATEMENT

### Current Architecture (v1.x) - Confusing

```
User sees:
- FuelAgent (what does this do?)
- FuelAgentAI (what's the difference?)

Developer confusion:
- Which one should I use?
- Are they duplicates?
- Do I need both?
```

**Result:** Poor developer experience, wasted time, incorrect usage

---

## SOLUTION: UNIFIED AGENT WITH AUTO-DETECTION

### Proposed Architecture (v2.0) - Clear

```python
from greenlang.agents import FuelAgent

agent = FuelAgent()

# Structured input → AUTO-ROUTES to Calculator path (fast, free)
result = agent.run({
    "fuel_type": "natural_gas",
    "amount": 1000,
    "unit": "therms"
})
# Execution: <10ms, $0, exact calculation

# Natural language → AUTO-ROUTES to Assistant path (smart, insightful)
result = agent.run({
    "query": "What are my Q3 emissions from natural gas?"
})
# Execution: ~3s, $0.003, AI-powered explanation

# SAME AGENT, AUTOMATIC PATH SELECTION
```

**Result:** Perfect developer experience, optimal performance, zero confusion

---

## TECHNICAL DESIGN

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│  UNIFIED FUEL AGENT (v2.0)                              │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  INPUT DETECTION & ROUTING                         │ │
│  │  - Analyze input type (structured vs NL)           │ │
│  │  - Select optimal execution path                   │ │
│  │  - Cache routing decision                          │ │
│  └─────────────┬──────────────────────────────────────┘ │
│                │                                          │
│        ┌───────┴──────┐                                  │
│        │              │                                   │
│        ▼              ▼                                   │
│  ┌──────────┐   ┌────────────┐                          │
│  │ FAST PATH│   │ SMART PATH │                          │
│  │ (Calc)   │   │ (AI)       │                          │
│  └──────────┘   └────────────┘                          │
│        │              │                                   │
│        │              ▼                                   │
│        │      ┌───────────────┐                          │
│        │      │ ChatSession   │                          │
│        │      │ (LLM)         │                          │
│        │      └───────┬───────┘                          │
│        │              │                                   │
│        │              │ CALLS AS TOOL                     │
│        ▼              ▼                                   │
│  ┌──────────────────────────────┐                        │
│  │ CORE CALCULATION ENGINE      │                        │
│  │ - Deterministic math          │                        │
│  │ - Database lookups            │                        │
│  │ - Zero hallucination          │                        │
│  └──────────────────────────────┘                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## IMPLEMENTATION SPECIFICATION

### 1. Base Unified Agent Class

```python
"""
greenlang/agents/unified/base_unified_agent.py

Base class for all unified agents implementing auto-detection and dual-path execution.
"""

from typing import Dict, Any, Union, Optional
from abc import ABC, abstractmethod
import asyncio
from datetime import datetime

from greenlang.agents.base import BaseAgent, AgentResult
from greenlang.intelligence import ChatSession, Budget, create_provider


class UnifiedAgent(BaseAgent, ABC):
    """
    Unified agent with automatic path selection.

    Supports both structured (Calculator path) and natural language (Assistant path) inputs.
    Automatically detects input type and routes to optimal execution path.

    Architecture:
        - Structured input → Fast Calculator path (<10ms, $0)
        - Natural language → Smart Assistant path (~3s, $0.001-0.01)
        - Single agent interface, zero user confusion

    Example:
        agent = FuelAgent()  # Unified agent

        # Structured → Calculator path
        result = agent.run({"fuel_type": "natural_gas", "amount": 1000})

        # Natural language → Assistant path
        result = agent.run({"query": "What are my emissions from natural gas?"})
    """

    def __init__(
        self,
        *,
        enable_ai: bool = True,
        ai_budget_usd: float = 0.50,
        ai_temperature: float = 0.0,
        ai_seed: int = 42,
        **kwargs
    ):
        """
        Initialize unified agent.

        Args:
            enable_ai: Enable AI Assistant path (default: True)
            ai_budget_usd: Max USD per AI query (default: $0.50)
            ai_temperature: LLM temperature (default: 0.0 for determinism)
            ai_seed: LLM seed (default: 42 for reproducibility)
            **kwargs: Additional config for base agent
        """
        super().__init__(**kwargs)

        self.enable_ai = enable_ai
        self.ai_budget_usd = ai_budget_usd
        self.ai_temperature = ai_temperature
        self.ai_seed = ai_seed

        # Initialize LLM provider if AI enabled
        if self.enable_ai:
            self.provider = create_provider()
        else:
            self.provider = None

        # Statistics tracking
        self._calculator_calls = 0
        self._assistant_calls = 0
        self._total_ai_cost_usd = 0.0

    def run(self, input_data: Union[Dict[str, Any], str]) -> AgentResult:
        """
        Execute agent with automatic path detection.

        Args:
            input_data: Either structured dict (Calculator) or string/dict with "query" (Assistant)

        Returns:
            AgentResult with computation or AI-enhanced response
        """
        start_time = datetime.now()

        try:
            # STEP 1: Detect input type and route
            if self._is_natural_language_input(input_data):
                # Route to Assistant path (AI-powered)
                if not self.enable_ai:
                    return AgentResult(
                        success=False,
                        error="Natural language input requires enable_ai=True"
                    )

                self._assistant_calls += 1
                result = self._execute_assistant_path(input_data)

            else:
                # Route to Calculator path (fast, deterministic)
                self._calculator_calls += 1
                result = self._execute_calculator_path(input_data)

            # Add routing metadata
            if result.success:
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                result.metadata.update({
                    "execution_path": "assistant" if self._is_natural_language_input(input_data) else "calculator",
                    "duration_ms": duration_ms,
                    "calculator_calls": self._calculator_calls,
                    "assistant_calls": self._assistant_calls,
                    "total_ai_cost_usd": self._total_ai_cost_usd,
                })

            return result

        except Exception as e:
            self.logger.error(f"Unified agent execution failed: {e}")
            return AgentResult(
                success=False,
                error=f"Execution failed: {str(e)}"
            )

    def _is_natural_language_input(self, input_data: Union[Dict, str]) -> bool:
        """
        Detect if input is natural language (Assistant path) or structured (Calculator path).

        Rules:
        - String input → Natural language
        - Dict with "query" key → Natural language
        - Dict with structured fields → Calculator path

        Args:
            input_data: Input to analyze

        Returns:
            True if natural language (Assistant), False if structured (Calculator)
        """
        # String input is always natural language
        if isinstance(input_data, str):
            return True

        # Dict with "query" key is natural language
        if isinstance(input_data, dict) and "query" in input_data:
            return True

        # Dict with "question" key is natural language
        if isinstance(input_data, dict) and "question" in input_data:
            return True

        # All other cases are structured Calculator input
        return False

    @abstractmethod
    def _execute_calculator_path(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute fast Calculator path (deterministic, <10ms, $0).

        Subclasses must implement exact calculation logic.

        Args:
            input_data: Structured input dict

        Returns:
            AgentResult with exact calculations
        """
        pass

    def _execute_assistant_path(self, input_data: Union[Dict, str]) -> AgentResult:
        """
        Execute smart Assistant path (AI-powered, ~3s, $0.001-0.01).

        Uses ChatSession with Calculator as a tool for exact numbers.

        Args:
            input_data: Natural language query (string or dict with "query")

        Returns:
            AgentResult with AI-enhanced response
        """
        # Extract query string
        if isinstance(input_data, str):
            query = input_data
            context = {}
        else:
            query = input_data.get("query") or input_data.get("question")
            context = {k: v for k, v in input_data.items() if k not in ["query", "question"]}

        # Run async AI execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self._execute_assistant_async(query, context)
            )
            return result
        finally:
            loop.close()

    async def _execute_assistant_async(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> AgentResult:
        """
        Async execution of AI Assistant path.

        Args:
            query: Natural language question
            context: Additional context data

        Returns:
            AgentResult with AI response
        """
        from greenlang.intelligence import ChatMessage, Role

        # Create ChatSession
        session = ChatSession(self.provider)

        # Build AI prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, context)

        messages = [
            ChatMessage(role=Role.system, content=system_prompt),
            ChatMessage(role=Role.user, content=user_prompt),
        ]

        # Get tools for this agent
        tools = self._get_calculator_tools()

        # Call AI with budget
        budget = Budget(max_usd=self.ai_budget_usd)

        response = await session.chat(
            messages=messages,
            tools=tools,
            budget=budget,
            temperature=self.ai_temperature,
            seed=self.ai_seed,
            tool_choice="auto",
        )

        # Track cost
        self._total_ai_cost_usd += response.usage.cost_usd

        # Build result
        return self._build_assistant_result(response, query, context)

    @abstractmethod
    def _build_system_prompt(self) -> str:
        """Build system prompt for AI Assistant."""
        pass

    @abstractmethod
    def _build_user_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Build user prompt for AI Assistant."""
        pass

    @abstractmethod
    def _get_calculator_tools(self) -> list:
        """Get list of Calculator tools for AI to use."""
        pass

    @abstractmethod
    def _build_assistant_result(
        self,
        response,
        query: str,
        context: Dict[str, Any]
    ) -> AgentResult:
        """Build AgentResult from AI response."""
        pass
```

---

### 2. Unified Fuel Agent Implementation

```python
"""
greenlang/agents/fuel_agent.py (v2.0 - UNIFIED)

Unified Fuel Agent with automatic Calculator/Assistant path selection.

Replaces:
- FuelAgent (v1.x Calculator)
- FuelAgentAI (v1.x Assistant)

Migration:
- Old code continues to work (backward compatible)
- New code uses single unified agent
"""

from typing import Dict, Any
from greenlang.agents.unified import UnifiedAgent
from greenlang.agents.base import AgentResult
from greenlang.intelligence.schemas.tools import ToolDef


class FuelAgent(UnifiedAgent):
    """
    Unified Fuel Agent - Auto-detects Calculator vs Assistant path.

    Supports both structured (fast) and natural language (smart) inputs:

    Calculator path (structured input):
        agent = FuelAgent()
        result = agent.run({
            "fuel_type": "natural_gas",
            "amount": 1000,
            "unit": "therms"
        })
        # <10ms, $0, exact calculation

    Assistant path (natural language):
        agent = FuelAgent()
        result = agent.run({
            "query": "What are my Q3 emissions from 1000 therms of natural gas?"
        })
        # ~3s, $0.003, AI explanation + exact calculation
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="FuelAgent",
            description="Unified fuel emissions calculator with AI assistance",
            version="2.0.0",
            **kwargs
        )

    # ========== CALCULATOR PATH (Fast, Free, Exact) ==========

    def _execute_calculator_path(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Fast Calculator path - deterministic fuel emissions calculation.

        Input: {"fuel_type": str, "amount": float, "unit": str, ...}
        Output: {"emissions_kg_co2e": float, "calculation_time_ms": float}

        Performance: <10ms, $0, 100% deterministic
        """
        # Validate input
        required = ["fuel_type", "amount", "unit"]
        if not all(k in input_data for k in required):
            return AgentResult(
                success=False,
                error=f"Missing required fields: {required}"
            )

        fuel_type = input_data["fuel_type"]
        amount = input_data["amount"]
        unit = input_data["unit"]

        # Lookup emission factor (deterministic database lookup)
        emission_factor = self._lookup_emission_factor(fuel_type, unit)
        if emission_factor is None:
            return AgentResult(
                success=False,
                error=f"Unknown fuel type: {fuel_type}"
            )

        # Calculate emissions (exact math)
        emissions_kg_co2e = amount * emission_factor

        return AgentResult(
            success=True,
            data={
                "emissions_kg_co2e": emissions_kg_co2e,
                "fuel_type": fuel_type,
                "amount": amount,
                "unit": unit,
                "emission_factor": emission_factor,
                "calculation_method": "deterministic",
            },
            metadata={
                "agent": "FuelAgent",
                "version": "2.0.0",
                "path": "calculator",
            }
        )

    def _lookup_emission_factor(self, fuel_type: str, unit: str) -> float:
        """Deterministic emission factor lookup."""
        # Emission factors database (kg CO2e per unit)
        factors = {
            ("natural_gas", "therms"): 5.306,
            ("natural_gas", "m3"): 1.911,
            ("diesel", "gallons"): 10.21,
            ("gasoline", "gallons"): 8.89,
            ("propane", "gallons"): 5.72,
            ("coal", "tons"): 2044.0,
            # ... complete database
        }
        return factors.get((fuel_type.lower(), unit.lower()))

    # ========== ASSISTANT PATH (Smart, AI-Powered) ==========

    def _build_system_prompt(self) -> str:
        """System prompt for AI Assistant."""
        return """You are a fuel emissions expert for GreenLang.

You help users calculate and understand fuel emissions using authoritative tools.

CRITICAL RULES:
1. Use the calculate_emissions tool for ALL numeric calculations
2. NEVER estimate or guess emission factors
3. NEVER do math yourself - always use tools
4. Explain calculations clearly in natural language
5. Provide context and recommendations when helpful

You have access to deterministic calculation tools that guarantee exact results."""

    def _build_user_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """User prompt for AI Assistant."""
        prompt = f"User question: {query}\n\n"

        if context:
            prompt += "Additional context:\n"
            for key, value in context.items():
                prompt += f"- {key}: {value}\n"

        prompt += "\nPlease calculate emissions and explain your findings."
        return prompt

    def _get_calculator_tools(self) -> list:
        """Get Calculator tools for AI to use."""
        return [
            ToolDef(
                name="calculate_emissions",
                description="Calculate exact CO2e emissions from fuel consumption. "
                           "Uses deterministic emission factors (zero hallucination guarantee).",
                parameters={
                    "type": "object",
                    "properties": {
                        "fuel_type": {
                            "type": "string",
                            "description": "Type of fuel (natural_gas, diesel, gasoline, propane, coal, etc.)",
                        },
                        "amount": {
                            "type": "number",
                            "description": "Amount of fuel consumed",
                        },
                        "unit": {
                            "type": "string",
                            "description": "Unit of measurement (therms, m3, gallons, tons, etc.)",
                        },
                    },
                    "required": ["fuel_type", "amount", "unit"],
                },
            ),
        ]

    def _build_assistant_result(self, response, query, context) -> AgentResult:
        """Build result from AI response."""
        # Extract tool results (exact calculations)
        tool_results = [tc.result for tc in response.tool_calls if tc.result]

        # Get AI explanation
        explanation = response.text

        return AgentResult(
            success=True,
            data={
                "query": query,
                "calculations": tool_results,
                "explanation": explanation,
                "ai_model": response.provider_info.model,
            },
            metadata={
                "agent": "FuelAgent",
                "version": "2.0.0",
                "path": "assistant",
                "tokens": response.usage.total_tokens,
                "cost_usd": response.usage.cost_usd,
                "tool_calls": len(response.tool_calls),
            }
        )


# ========== BACKWARD COMPATIBILITY ==========

# Old names still work (with deprecation warnings)

class FuelAgentAI(FuelAgent):
    """
    DEPRECATED: Use FuelAgent() instead.

    FuelAgentAI is now an alias for unified FuelAgent.
    Old code continues to work but please migrate to FuelAgent().
    """
    def __init__(self, **kwargs):
        import warnings
        warnings.warn(
            "FuelAgentAI is deprecated. Use FuelAgent() instead. "
            "FuelAgent now auto-detects Calculator vs Assistant paths.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(**kwargs)
```

---

### 3. Migration Strategy

#### Phase 1: v1.5 (Q1 2026) - Preparation

```python
# Add deprecation warnings to existing agents

# greenlang/agents/fuel_agent_ai.py
class FuelAgentAI(BaseAgent):
    def __init__(self, **kwargs):
        import warnings
        warnings.warn(
            "FuelAgentAI will be merged into FuelAgent in v2.0. "
            "Please prepare for migration.",
            FutureWarning
        )
        # ... existing code
```

#### Phase 2: v2.0 (Q2 2026) - Launch Unified Agents

```python
# New unified agents available
from greenlang.agents import FuelAgent  # Auto-detecting unified agent

# Old names still work (backward compatible with warnings)
from greenlang.agents import FuelAgentAI  # Deprecated alias
```

#### Phase 3: v3.0 (Q4 2026) - Remove Old Names

```python
# Only unified agents available
from greenlang.agents import FuelAgent  # Unified agent

# Old names raise ImportError
from greenlang.agents import FuelAgentAI  # ImportError: deprecated, use FuelAgent
```

---

## TESTING STRATEGY

### Unit Tests

```python
# tests/agents/test_fuel_agent_unified.py

import pytest
from greenlang.agents import FuelAgent


class TestFuelAgentCalculatorPath:
    """Test fast Calculator path (structured input)."""

    def test_structured_input_uses_calculator_path(self):
        """Verify structured input routes to Calculator path."""
        agent = FuelAgent()

        result = agent.run({
            "fuel_type": "natural_gas",
            "amount": 1000,
            "unit": "therms"
        })

        assert result.success
        assert result.metadata["path"] == "calculator"
        assert result.metadata["duration_ms"] < 100  # <100ms
        assert result.data["emissions_kg_co2e"] == 5306.0  # Exact

    def test_calculator_path_is_deterministic(self):
        """Verify Calculator path produces identical results."""
        agent = FuelAgent()
        input_data = {"fuel_type": "diesel", "amount": 100, "unit": "gallons"}

        result1 = agent.run(input_data)
        result2 = agent.run(input_data)

        assert result1.data["emissions_kg_co2e"] == result2.data["emissions_kg_co2e"]
        assert result1.metadata["path"] == "calculator"


class TestFuelAgentAssistantPath:
    """Test smart Assistant path (natural language)."""

    def test_natural_language_string_uses_assistant_path(self):
        """Verify NL string input routes to Assistant path."""
        agent = FuelAgent()

        result = agent.run("What are my emissions from 1000 therms of natural gas?")

        assert result.success
        assert result.metadata["path"] == "assistant"
        assert "explanation" in result.data
        assert "calculations" in result.data

    def test_query_dict_uses_assistant_path(self):
        """Verify dict with 'query' key routes to Assistant path."""
        agent = FuelAgent()

        result = agent.run({
            "query": "Calculate emissions for 100 gallons of diesel"
        })

        assert result.success
        assert result.metadata["path"] == "assistant"

    def test_assistant_uses_calculator_tool(self):
        """Verify Assistant calls Calculator as a tool."""
        agent = FuelAgent()

        result = agent.run({"query": "What are emissions from 50 gallons of gasoline?"})

        assert result.success
        assert len(result.data["calculations"]) > 0  # Tool was called
        assert result.metadata["tool_calls"] > 0


class TestBackwardCompatibility:
    """Test backward compatibility with v1.x names."""

    def test_fuel_agent_ai_alias_works(self):
        """Verify FuelAgentAI alias still works."""
        with pytest.warns(DeprecationWarning):
            from greenlang.agents import FuelAgentAI

            agent = FuelAgentAI()
            result = agent.run("Calculate emissions")

            assert result.success
```

---

## PERFORMANCE REQUIREMENTS

| Path | Response Time | Cost | Determinism |
|------|---------------|------|-------------|
| **Calculator** | <10ms (p99) | $0 | 100% |
| **Assistant** | <5s (p99) | <$0.01 | Numbers: 100%, Text: N/A |

---

## ROLLOUT PLAN

### Timeline

| Version | Date | Milestone |
|---------|------|-----------|
| **v1.5** | Q1 2026 | Add deprecation warnings, documentation |
| **v2.0** | Q2 2026 | Launch unified agents (backward compatible) |
| **v2.5** | Q3 2026 | Encourage migration, update examples |
| **v3.0** | Q4 2026 | Remove old names (breaking change) |

### Communication Plan

**Q1 2026:**
- Blog post: "Introducing Unified Agents in v2.0"
- Documentation: Update all guides
- Webinar: Migration workshop

**Q2 2026:**
- Release notes: Highlight unified agents
- Code examples: Show both paths
- Migration guide: Step-by-step instructions

**Q3 2026:**
- Email campaign: Encourage migration
- Deprecation notices: In IDE and runtime
- Support: Migration assistance

**Q4 2026:**
- Final warning: 30 days before v3.0
- Breaking change: Remove old names
- Support: Help stragglers migrate

---

## SUCCESS METRICS

**Developer Experience:**
- ✅ Reduce "which agent?" support questions by 90%
- ✅ Reduce time-to-first-API-call by 50%
- ✅ Increase developer satisfaction score from 7/10 to 9/10

**Performance:**
- ✅ Maintain <10ms for Calculator path
- ✅ Maintain <5s for Assistant path
- ✅ Zero performance regression

**Adoption:**
- ✅ 80% of users migrated to unified agents by v2.5
- ✅ 95% of users migrated by v3.0
- ✅ <5% require migration assistance

---

## RISKS & MITIGATION

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Breaking changes in v3.0 | High | High | Long deprecation period (9 months) |
| Performance regression | Medium | High | Extensive benchmarking, early testing |
| User confusion during migration | Medium | Medium | Clear documentation, webinars, support |
| Backward compatibility bugs | Low | High | Comprehensive testing, gradual rollout |

---

## CONCLUSION

The Unified Agent Architecture eliminates user confusion while maintaining all benefits of the dual-tier system:

- ✅ **One agent name** (FuelAgent instead of FuelAgent + FuelAgentAI)
- ✅ **Automatic path selection** (fast for APIs, smart for humans)
- ✅ **Backward compatible** (old code continues to work)
- ✅ **Zero performance loss** (Calculator path unchanged)
- ✅ **Better developer experience** (clear, simple, intuitive)

**Recommendation:** APPROVE for Q2 2026 release (v2.0.0)

---

**Document Status:** APPROVED ARCHITECTURE
**Implementation:** Q1 2026 (start), Q2 2026 (release)
**Owner:** Head of AI & Climate Intelligence

---

**END OF SPECIFICATION**
