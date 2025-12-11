# -*- coding: utf-8 -*-
"""
Intelligence Mixin - Drop-in LLM Intelligence for Existing Agents

This mixin provides a ZERO-BREAKING-CHANGE way to add LLM intelligence
to existing BaseAgent implementations. Simply add the mixin to your
agent's class hierarchy and gain all intelligence capabilities.

USE CASE: Retrofitting 30+ existing agents without rewriting them.

Before (No Intelligence):
    class CarbonAgent(BaseAgent):
        def execute(self, input_data):
            emissions = self._calculate(input_data)
            return AgentResult(
                success=True,
                data={"emissions": emissions, "summary": self._static_summary(emissions)}
            )

After (With Intelligence - ONE LINE CHANGE):
    class CarbonAgent(IntelligenceMixin, BaseAgent):  # Just add the mixin!
        def execute(self, input_data):
            emissions = self._calculate(input_data)
            return AgentResult(
                success=True,
                data={
                    "emissions": emissions,
                    "summary": self.generate_explanation(input_data, {"emissions": emissions}),
                    "recommendations": self.generate_recommendations({"emissions": emissions})
                }
            )

MIGRATION PATH:
1. Add IntelligenceMixin to class hierarchy (before BaseAgent)
2. Call self._init_intelligence() in __init__ (optional - auto-initialized)
3. Use self.generate_explanation(), self.generate_recommendations(), etc.
4. Existing code continues to work unchanged

Author: GreenLang Intelligence Framework
Date: December 2025
Status: Production Ready - Retrofit Tool
"""

from __future__ import annotations
import logging
import asyncio
from typing import Any, Dict, List, Optional, Callable
from pydantic import BaseModel, Field

from greenlang.intelligence import create_provider
from greenlang.intelligence.providers.base import LLMProvider
from greenlang.intelligence.runtime.session import ChatSession
from greenlang.intelligence.runtime.budget import Budget, BudgetExceeded
from greenlang.intelligence.schemas.messages import ChatMessage, Role

logger = logging.getLogger(__name__)


class IntelligenceConfig(BaseModel):
    """Configuration for the intelligence mixin."""
    enabled: bool = Field(default=True, description="Enable/disable intelligence")
    model: str = Field(default="auto", description="LLM model to use")
    max_budget_per_call_usd: float = Field(default=0.10, description="Max USD per LLM call")
    enable_explanations: bool = Field(default=True, description="Enable explanation generation")
    enable_recommendations: bool = Field(default=True, description="Enable recommendation generation")
    enable_anomaly_detection: bool = Field(default=False, description="Enable anomaly detection")
    domain_context: str = Field(default="climate and sustainability", description="Domain context")
    regulatory_context: Optional[str] = Field(default=None, description="Regulatory context")
    cache_enabled: bool = Field(default=True, description="Enable response caching")


class IntelligenceMetricsMixin(BaseModel):
    """Metrics tracked by the intelligence mixin."""
    llm_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    explanations_generated: int = 0
    recommendations_generated: int = 0
    anomalies_detected: int = 0
    cache_hits: int = 0


class IntelligenceMixin:
    """
    Mixin class that adds LLM intelligence capabilities to any BaseAgent.

    This mixin is designed for RETROFITTING existing agents without
    breaking changes. Add it to your class hierarchy and call the
    intelligence methods as needed.

    Key Methods:
        - generate_explanation(input_data, output_data) -> str
        - generate_recommendations(analysis) -> List[Dict]
        - detect_anomalies(data) -> List[Dict]
        - reason_about(question, context) -> str
        - validate_with_reasoning(input_data) -> (bool, str)

    Usage:
        # Option 1: Auto-initialization (recommended)
        class MyAgent(IntelligenceMixin, BaseAgent):
            def execute(self, input_data):
                result = self._calculate(input_data)
                explanation = self.generate_explanation(input_data, result)
                return AgentResult(success=True, data={"result": result, "explanation": explanation})

        # Option 2: Manual initialization with config
        class MyAgent(IntelligenceMixin, BaseAgent):
            def __init__(self, config):
                super().__init__(config)
                self._init_intelligence(IntelligenceConfig(
                    model="claude-3-haiku",
                    max_budget_per_call_usd=0.05
                ))

    Thread Safety:
        The mixin maintains instance-level state (provider, session, metrics).
        Each agent instance has its own LLM connection.

    Cost Control:
        - Budget enforcement per call (max_budget_per_call_usd)
        - Automatic cost tracking (available via get_intelligence_metrics())
        - Graceful degradation on budget exceeded (returns error message, doesn't crash)
    """

    # Class-level defaults
    _default_intelligence_config = IntelligenceConfig()

    def _init_intelligence(
        self,
        config: Optional[IntelligenceConfig] = None
    ) -> None:
        """
        Initialize intelligence capabilities.

        This is called automatically on first use, but you can call it
        explicitly in __init__ to customize configuration.

        Args:
            config: Optional IntelligenceConfig to override defaults
        """
        self._intel_config = config or IntelligenceConfig()
        self._intel_provider: Optional[LLMProvider] = None
        self._intel_session: Optional[ChatSession] = None
        self._intel_metrics = IntelligenceMetricsMixin()
        self._intel_initialized = True

        logger.debug(
            f"Intelligence initialized: model={self._intel_config.model}, "
            f"enabled={self._intel_config.enabled}"
        )

    def _ensure_intelligence_initialized(self) -> None:
        """Ensure intelligence is initialized (auto-init on first use)."""
        if not hasattr(self, '_intel_initialized') or not self._intel_initialized:
            self._init_intelligence()

    def _get_intel_provider(self) -> LLMProvider:
        """Get or create LLM provider."""
        self._ensure_intelligence_initialized()
        if self._intel_provider is None:
            self._intel_provider = create_provider(model=self._intel_config.model)
            self._intel_session = ChatSession(self._intel_provider)
        return self._intel_provider

    def _get_intel_session(self) -> ChatSession:
        """Get or create chat session."""
        if not hasattr(self, '_intel_session') or self._intel_session is None:
            self._get_intel_provider()
        return self._intel_session

    def _intel_budget(self) -> Budget:
        """Create a budget for an LLM call."""
        self._ensure_intelligence_initialized()
        return Budget(max_usd=self._intel_config.max_budget_per_call_usd)

    def _intel_system_prompt(self, task: str) -> str:
        """Build system prompt with context."""
        self._ensure_intelligence_initialized()

        parts = [
            f"You are an expert AI assistant for {self._intel_config.domain_context}.",
            f"Task: {task}"
        ]

        if self._intel_config.regulatory_context:
            parts.append(f"Regulatory context: {self._intel_config.regulatory_context}")

        # Add agent context if available
        if hasattr(self, 'config') and hasattr(self.config, 'name'):
            parts.append(f"Agent: {self.config.name}")

        parts.extend([
            "",
            "IMPORTANT:",
            "- Be factual and precise",
            "- Never hallucinate numbers",
            "- Be concise but comprehensive"
        ])

        return "\n".join(parts)

    async def _intel_call_llm(
        self,
        messages: List[ChatMessage],
        task: str,
        temperature: float = 0.0
    ) -> str:
        """Make an LLM call with metrics tracking."""
        self._ensure_intelligence_initialized()

        if not self._intel_config.enabled:
            return "[Intelligence disabled]"

        session = self._get_intel_session()
        budget = self._intel_budget()

        try:
            response = await session.chat(
                messages=messages,
                budget=budget,
                temperature=temperature
            )

            # Update metrics
            self._intel_metrics.llm_calls += 1
            self._intel_metrics.total_tokens += response.usage.total_tokens
            self._intel_metrics.total_cost_usd += response.usage.cost_usd

            return response.text or ""

        except BudgetExceeded as e:
            logger.warning(f"Intelligence budget exceeded for {task}: {e}")
            return f"[Budget exceeded: {e}]"
        except Exception as e:
            logger.error(f"Intelligence error for {task}: {e}")
            return f"[Error: {str(e)}]"

    def _run_intel_async(self, coro):
        """Run async coroutine synchronously."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    # =========================================================================
    # PUBLIC INTELLIGENCE METHODS
    # =========================================================================

    def generate_explanation(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        calculation_steps: Optional[List[str]] = None
    ) -> str:
        """
        Generate a natural language explanation of a calculation.

        Args:
            input_data: Input provided to the agent
            output_data: Output/results from calculation
            calculation_steps: Optional list of calculation steps

        Returns:
            Natural language explanation string
        """
        self._ensure_intelligence_initialized()

        if not self._intel_config.enable_explanations:
            return "Explanations disabled"

        user_parts = [
            "Explain this calculation clearly:",
            "",
            "INPUT:", str(input_data),
            "OUTPUT:", str(output_data)
        ]

        if calculation_steps:
            user_parts.extend(["", "STEPS:", "\n".join(f"- {s}" for s in calculation_steps)])

        user_parts.extend([
            "",
            "Provide a clear explanation suitable for a compliance report."
        ])

        messages = [
            ChatMessage(role=Role.system, content=self._intel_system_prompt("Generate explanation")),
            ChatMessage(role=Role.user, content="\n".join(user_parts))
        ]

        result = self._run_intel_async(self._intel_call_llm(messages, "explanation"))
        self._intel_metrics.explanations_generated += 1
        return result

    def generate_recommendations(
        self,
        analysis: Dict[str, Any],
        max_recommendations: int = 5,
        focus_areas: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations based on analysis.

        Args:
            analysis: Analysis data to base recommendations on
            max_recommendations: Maximum number of recommendations
            focus_areas: Optional focus areas

        Returns:
            List of recommendation dictionaries
        """
        self._ensure_intelligence_initialized()

        if not self._intel_config.enable_recommendations:
            return []

        user_parts = [
            "Provide actionable recommendations based on:",
            "",
            "ANALYSIS:", str(analysis),
            ""
        ]

        if focus_areas:
            user_parts.append(f"FOCUS: {', '.join(focus_areas)}")

        user_parts.extend([
            "",
            f"Return up to {max_recommendations} recommendations as JSON array:",
            '[{"id": "REC-001", "title": "...", "description": "...", "category": "...", "priority": "high|medium|low", "potential_impact": "..."}]',
            "",
            "Return ONLY valid JSON."
        ])

        messages = [
            ChatMessage(role=Role.system, content=self._intel_system_prompt("Generate recommendations")),
            ChatMessage(role=Role.user, content="\n".join(user_parts))
        ]

        response = self._run_intel_async(self._intel_call_llm(messages, "recommendations"))

        try:
            import json
            recommendations = json.loads(response)
            self._intel_metrics.recommendations_generated += len(recommendations)
            return recommendations
        except Exception as e:
            logger.error(f"Failed to parse recommendations: {e}")
            return []

    def detect_anomalies(
        self,
        data: Dict[str, Any],
        expected_ranges: Optional[Dict[str, tuple]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in data using LLM reasoning.

        Args:
            data: Data to analyze
            expected_ranges: Optional dict of field -> (min, max)

        Returns:
            List of anomaly dictionaries
        """
        self._ensure_intelligence_initialized()

        if not self._intel_config.enable_anomaly_detection:
            return []

        user_parts = [
            "Analyze for anomalies:",
            "",
            "DATA:", str(data),
            ""
        ]

        if expected_ranges:
            user_parts.extend(["EXPECTED RANGES:", str(expected_ranges), ""])

        user_parts.extend([
            "Return anomalies as JSON:",
            '[{"id": "ANOM-001", "description": "...", "severity": "high|medium|low", "field": "...", "reasoning": "..."}]',
            "",
            "If none, return []. Return ONLY valid JSON."
        ])

        messages = [
            ChatMessage(role=Role.system, content=self._intel_system_prompt("Detect anomalies")),
            ChatMessage(role=Role.user, content="\n".join(user_parts))
        ]

        response = self._run_intel_async(self._intel_call_llm(messages, "anomalies"))

        try:
            import json
            anomalies = json.loads(response)
            self._intel_metrics.anomalies_detected += len(anomalies)
            return anomalies
        except Exception as e:
            logger.error(f"Failed to parse anomalies: {e}")
            return []

    def reason_about(
        self,
        question: str,
        context: Dict[str, Any],
        chain_of_thought: bool = False
    ) -> str:
        """
        Use LLM to reason about a question.

        Args:
            question: Question to answer
            context: Context data
            chain_of_thought: Include reasoning steps

        Returns:
            LLM's reasoned answer
        """
        self._ensure_intelligence_initialized()

        user_parts = [
            f"QUESTION: {question}",
            "",
            "CONTEXT:", str(context)
        ]

        if chain_of_thought:
            user_parts.extend(["", "Provide step-by-step reasoning."])

        messages = [
            ChatMessage(role=Role.system, content=self._intel_system_prompt("Answer question")),
            ChatMessage(role=Role.user, content="\n".join(user_parts))
        ]

        return self._run_intel_async(self._intel_call_llm(messages, "reasoning"))

    def validate_with_reasoning(
        self,
        input_data: Dict[str, Any],
        rules: Optional[List[str]] = None
    ) -> tuple[bool, str]:
        """
        Validate input with LLM reasoning.

        Args:
            input_data: Data to validate
            rules: Optional validation rules

        Returns:
            Tuple of (is_valid, reasoning)
        """
        self._ensure_intelligence_initialized()

        user_parts = [
            "Validate this data:",
            "",
            "DATA:", str(input_data)
        ]

        if rules:
            user_parts.extend(["", "RULES:", "\n".join(f"- {r}" for r in rules)])

        user_parts.extend([
            "",
            "Respond with:",
            "VALID: true/false",
            "REASONING: explanation"
        ])

        messages = [
            ChatMessage(role=Role.system, content=self._intel_system_prompt("Validate data")),
            ChatMessage(role=Role.user, content="\n".join(user_parts))
        ]

        response = self._run_intel_async(self._intel_call_llm(messages, "validation"))

        is_valid = "valid: true" in response.lower()
        return is_valid, response

    def get_intelligence_metrics(self) -> Dict[str, Any]:
        """Get intelligence metrics as dictionary."""
        self._ensure_intelligence_initialized()
        return self._intel_metrics.dict()

    def reset_intelligence_metrics(self) -> None:
        """Reset intelligence metrics."""
        self._ensure_intelligence_initialized()
        self._intel_metrics = IntelligenceMetricsMixin()

    def set_intelligence_enabled(self, enabled: bool) -> None:
        """Enable or disable intelligence."""
        self._ensure_intelligence_initialized()
        self._intel_config.enabled = enabled

    def set_intelligence_model(self, model: str) -> None:
        """Change the LLM model (recreates provider on next call)."""
        self._ensure_intelligence_initialized()
        self._intel_config.model = model
        self._intel_provider = None
        self._intel_session = None

    def set_intelligence_budget(self, max_usd: float) -> None:
        """Set the maximum budget per LLM call."""
        self._ensure_intelligence_initialized()
        self._intel_config.max_budget_per_call_usd = max_usd


# =============================================================================
# RETROFIT HELPER FUNCTIONS
# =============================================================================

def retrofit_agent_class(
    agent_class: type,
    intelligence_config: Optional[IntelligenceConfig] = None
) -> type:
    """
    Dynamically retrofit an existing agent class with intelligence.

    This function creates a new class that inherits from both
    IntelligenceMixin and the original agent class.

    Args:
        agent_class: The original agent class to retrofit
        intelligence_config: Optional configuration

    Returns:
        A new class with intelligence capabilities

    Example:
        from greenlang.agents.carbon_agent import CarbonAgent

        IntelligentCarbonAgent = retrofit_agent_class(CarbonAgent)

        agent = IntelligentCarbonAgent()
        result = agent.run(input_data)
        explanation = agent.generate_explanation(input_data, result.data)
    """
    class RetrofittedAgent(IntelligenceMixin, agent_class):
        """Dynamically retrofitted agent with intelligence."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if intelligence_config:
                self._init_intelligence(intelligence_config)

    RetrofittedAgent.__name__ = f"Intelligent{agent_class.__name__}"
    RetrofittedAgent.__doc__ = f"Retrofitted {agent_class.__name__} with LLM intelligence."

    return RetrofittedAgent


def create_intelligent_wrapper(
    agent_instance: Any,
    intelligence_config: Optional[IntelligenceConfig] = None
) -> Any:
    """
    Create an intelligent wrapper around an existing agent instance.

    This is useful when you can't modify the agent class but want
    to add intelligence to a specific instance.

    Args:
        agent_instance: The agent instance to wrap
        intelligence_config: Optional configuration

    Returns:
        A wrapper object with intelligence methods

    Example:
        agent = CarbonAgent()
        intelligent_agent = create_intelligent_wrapper(agent)

        result = intelligent_agent.run(input_data)
        explanation = intelligent_agent.generate_explanation(input_data, result.data)
    """

    class IntelligentWrapper(IntelligenceMixin):
        """Wrapper that adds intelligence to an existing agent instance."""

        def __init__(self, wrapped_agent, config=None):
            self._wrapped = wrapped_agent
            self._init_intelligence(config or IntelligenceConfig())

        def __getattr__(self, name):
            # Delegate to wrapped agent for unknown attributes
            return getattr(self._wrapped, name)

        def run(self, input_data):
            """Run with intelligence metrics tracking."""
            self.reset_intelligence_metrics()
            result = self._wrapped.run(input_data)

            # Add intelligence metrics to result
            if hasattr(result, 'metadata'):
                result.metadata['intelligence_metrics'] = self.get_intelligence_metrics()

            return result

    return IntelligentWrapper(agent_instance, intelligence_config)


# =============================================================================
# BATCH RETROFIT UTILITIES
# =============================================================================

def retrofit_all_agents_in_module(
    module,
    intelligence_config: Optional[IntelligenceConfig] = None,
    exclude: Optional[List[str]] = None
) -> Dict[str, type]:
    """
    Retrofit all agent classes in a module.

    Args:
        module: The module containing agent classes
        intelligence_config: Configuration for all agents
        exclude: List of class names to exclude

    Returns:
        Dict of name -> retrofitted class

    Example:
        from greenlang import agents

        retrofitted = retrofit_all_agents_in_module(agents)
        IntelligentCarbonAgent = retrofitted['CarbonAgent']
    """
    from greenlang.agents.base import BaseAgent

    exclude = exclude or []
    retrofitted = {}

    for name in dir(module):
        if name.startswith('_') or name in exclude:
            continue

        obj = getattr(module, name)

        # Check if it's a BaseAgent subclass
        if isinstance(obj, type) and issubclass(obj, BaseAgent) and obj is not BaseAgent:
            retrofitted[name] = retrofit_agent_class(obj, intelligence_config)
            logger.info(f"Retrofitted agent: {name}")

    return retrofitted


# =============================================================================
# VALIDATION DECORATOR
# =============================================================================

def requires_intelligence(method: Callable) -> Callable:
    """
    Decorator to mark methods that require intelligence to be enabled.

    If intelligence is disabled, returns a default value instead of
    calling the method.

    Usage:
        @requires_intelligence
        def generate_insights(self, data):
            return self.reason_about("What insights...", data)
    """
    def wrapper(self, *args, **kwargs):
        if hasattr(self, '_intel_config') and not self._intel_config.enabled:
            logger.warning(f"Intelligence disabled, skipping {method.__name__}")
            return None
        return method(self, *args, **kwargs)

    wrapper.__name__ = method.__name__
    wrapper.__doc__ = method.__doc__
    return wrapper


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """Demonstrate the IntelligenceMixin retrofit pattern."""

    from greenlang.agents.base import BaseAgent, AgentConfig, AgentResult

    # Example: Original agent (no intelligence)
    class OriginalCarbonAgent(BaseAgent):
        """Simple carbon agent without intelligence."""

        def execute(self, input_data):
            emissions = input_data.get("amount", 0) * 10.21
            return AgentResult(
                success=True,
                data={"co2e_kg": emissions}
            )

    # Example: Retrofitted agent (with intelligence - ONE LINE CHANGE!)
    class IntelligentCarbonAgent(IntelligenceMixin, OriginalCarbonAgent):
        """Carbon agent WITH intelligence - just add the mixin!"""

        def execute(self, input_data):
            # Run original calculation
            result = super().execute(input_data)

            # Add intelligence
            explanation = self.generate_explanation(
                input_data=input_data,
                output_data=result.data
            )

            result.data["explanation"] = explanation
            return result

    # Run the retrofitted agent
    agent = IntelligentCarbonAgent(AgentConfig(
        name="IntelligentCarbonAgent",
        description="Retrofitted carbon agent"
    ))

    result = agent.run({"fuel_type": "diesel", "amount": 100})

    print(f"Result: {result.data}")
    print(f"Intelligence Metrics: {agent.get_intelligence_metrics()}")
