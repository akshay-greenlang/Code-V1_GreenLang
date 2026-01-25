# -*- coding: utf-8 -*-
"""
Intelligent Agent Base - The Foundation for AI-Native Climate OS

This module solves the "Intelligence Paradox" by bridging BaseAgent with
GreenLang's LLM infrastructure. All agents extending this base class are
truly "intelligent" - they leverage LLM reasoning for:
- Explanation generation
- Recommendation synthesis
- Anomaly detection
- Contextual reasoning
- Input validation with reasoning

CRITICAL: This is the MANDATORY base class for all new agents in GreenLang.
The AI Factory MUST generate agents that extend IntelligentAgentBase.

Architecture:
    IntelligentAgentBase (this module)
        ├── BaseAgent (lifecycle, metrics, provenance)
        └── LLM Intelligence (greenlang.intelligence)
            ├── Providers (Anthropic, OpenAI)
            ├── ChatSession (budget, telemetry)
            ├── RAG (domain knowledge)
            └── Tool system (function calling)

Example:
    >>> from greenlang.agents.intelligent_base import IntelligentAgentBase, IntelligentAgentConfig
    >>>
    >>> class CarbonAgentIntelligent(IntelligentAgentBase):
    ...     '''Intelligent carbon emissions agent with LLM reasoning'''
    ...
    ...     def execute(self, input_data):
    ...         # 1. Deterministic calculation (zero-hallucination)
    ...         emissions = self._calculate_emissions(input_data)
    ...
    ...         # 2. AI-powered explanation (this is what makes it intelligent!)
    ...         explanation = self.generate_explanation(
    ...             input_data=input_data,
    ...             output_data={"emissions": emissions}
    ...         )
    ...
    ...         # 3. AI-powered recommendations
    ...         recommendations = self.generate_recommendations(
    ...             analysis={"emissions": emissions, "trend": "increasing"}
    ...         )
    ...
    ...         return AgentResult(
    ...             success=True,
    ...             data={
    ...                 "emissions": emissions,
    ...                 "explanation": explanation,  # AI-generated
    ...                 "recommendations": recommendations  # AI-generated
    ...             }
    ...         )

Author: GreenLang Intelligence Framework
Date: December 2025
Status: Production Ready
"""

from __future__ import annotations
import logging
import asyncio
from abc import ABC
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum

from greenlang.agents.base import BaseAgent, AgentConfig, AgentResult
from greenlang.intelligence import create_provider
from greenlang.intelligence.providers.base import LLMProvider
from greenlang.intelligence.runtime.session import ChatSession
from greenlang.intelligence.runtime.budget import Budget, BudgetExceeded
from greenlang.intelligence.schemas.messages import ChatMessage, Role
from greenlang.intelligence.schemas.tools import ToolDef
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class IntelligenceLevel(str, Enum):
    """Intelligence levels for agents"""
    NONE = "none"  # No LLM usage - pure deterministic (DEPRECATED for new agents)
    BASIC = "basic"  # LLM for explanations only
    STANDARD = "standard"  # LLM for explanations + recommendations
    ADVANCED = "advanced"  # LLM for reasoning, anomaly detection, RAG
    FULL = "full"  # All capabilities including chain-of-thought


class IntelligentAgentConfig(AgentConfig):
    """Configuration for intelligent agents with LLM capabilities."""

    # Intelligence configuration
    intelligence_level: IntelligenceLevel = Field(
        default=IntelligenceLevel.STANDARD,
        description="Level of LLM intelligence to use"
    )

    # LLM provider configuration
    llm_model: str = Field(
        default="auto",
        description="LLM model to use (auto, gpt-4o, claude-3-sonnet, etc.)"
    )

    # Budget controls
    max_budget_per_call_usd: float = Field(
        default=0.10,
        ge=0.01,
        le=5.0,
        description="Maximum USD budget per LLM call"
    )
    max_budget_per_execution_usd: float = Field(
        default=0.50,
        ge=0.05,
        le=10.0,
        description="Maximum USD budget per agent execution"
    )

    # Feature flags
    enable_explanations: bool = Field(
        default=True,
        description="Enable AI-generated explanations"
    )
    enable_recommendations: bool = Field(
        default=True,
        description="Enable AI-generated recommendations"
    )
    enable_anomaly_detection: bool = Field(
        default=False,
        description="Enable AI-powered anomaly detection"
    )
    enable_rag: bool = Field(
        default=False,
        description="Enable RAG for domain knowledge"
    )

    # Caching
    enable_semantic_cache: bool = Field(
        default=True,
        description="Enable semantic caching of LLM responses"
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        description="Cache TTL in seconds"
    )

    # Domain context
    domain_context: str = Field(
        default="climate and sustainability",
        description="Domain context for LLM prompts"
    )
    regulatory_context: Optional[str] = Field(
        default=None,
        description="Regulatory context (e.g., 'CSRD', 'CBAM', 'SB253')"
    )


class Recommendation(BaseModel):
    """A recommendation generated by the intelligent agent."""
    id: str = Field(..., description="Unique recommendation ID")
    title: str = Field(..., description="Short recommendation title")
    description: str = Field(..., description="Detailed recommendation")
    category: str = Field(..., description="Category (efficiency, reduction, compliance, etc.)")
    priority: str = Field(..., description="Priority (high, medium, low)")
    potential_impact: Optional[str] = Field(None, description="Potential impact estimate")
    implementation_effort: Optional[str] = Field(None, description="Implementation effort")
    regulatory_relevance: Optional[str] = Field(None, description="Regulatory relevance")


class Anomaly(BaseModel):
    """An anomaly detected by the intelligent agent."""
    id: str = Field(..., description="Unique anomaly ID")
    description: str = Field(..., description="What the anomaly is")
    severity: str = Field(..., description="Severity (critical, high, medium, low)")
    field: str = Field(..., description="Which field/data point is anomalous")
    expected_range: Optional[str] = Field(None, description="Expected range")
    actual_value: Optional[str] = Field(None, description="Actual value found")
    reasoning: str = Field(..., description="LLM reasoning for why this is anomalous")


class IntelligenceMetrics(BaseModel):
    """Metrics from LLM intelligence operations."""
    llm_calls: int = Field(default=0, description="Number of LLM calls made")
    total_tokens: int = Field(default=0, description="Total tokens used")
    total_cost_usd: float = Field(default=0.0, description="Total cost in USD")
    cache_hits: int = Field(default=0, description="Number of cache hits")
    cache_misses: int = Field(default=0, description="Number of cache misses")
    explanations_generated: int = Field(default=0, description="Number of explanations generated")
    recommendations_generated: int = Field(default=0, description="Number of recommendations generated")
    anomalies_detected: int = Field(default=0, description="Number of anomalies detected")


class IntelligentAgentBase(BaseAgent, ABC):
    """
    Base class for AI-native intelligent agents.

    This class extends BaseAgent with full LLM intelligence capabilities,
    solving the "Intelligence Paradox" by making LLM reasoning a first-class
    citizen in every agent.

    Key capabilities:
    1. generate_explanation() - Create natural language explanations of calculations
    2. generate_recommendations() - Synthesize actionable recommendations
    3. detect_anomalies() - Identify unusual patterns in data
    4. reason_about() - General-purpose LLM reasoning
    5. validate_with_reasoning() - Input validation with LLM reasoning

    Design principles:
    - Zero-hallucination calculations: Numbers are ALWAYS deterministic
    - LLM for reasoning: Explanations and recommendations use LLM
    - Budget enforcement: Every LLM call is budgeted
    - Caching: Semantic caching to reduce costs
    - Provenance: Full audit trail of LLM calls

    Example:
        class MyIntelligentAgent(IntelligentAgentBase):
            def execute(self, input_data):
                # Deterministic calculation
                result = self.calculate(input_data)

                # AI-powered intelligence
                explanation = self.generate_explanation(input_data, result)
                recommendations = self.generate_recommendations(result)

                return AgentResult(
                    success=True,
                    data={
                        "result": result,
                        "explanation": explanation,
                        "recommendations": recommendations
                    }
                )
    """

    def __init__(self, config: Optional[IntelligentAgentConfig] = None):
        """
        Initialize intelligent agent with LLM capabilities.

        Args:
            config: Intelligent agent configuration
        """
        if config is None:
            config = IntelligentAgentConfig(
                name=self.__class__.__name__,
                description=self.__class__.__doc__ or "Intelligent agent"
            )

        super().__init__(config)
        self.intelligent_config = config

        # Initialize LLM provider lazily
        self._provider: Optional[LLMProvider] = None
        self._session: Optional[ChatSession] = None

        # Intelligence metrics
        self._intelligence_metrics = IntelligenceMetrics()

        # Execution budget tracker
        self._execution_budget: Optional[Budget] = None

        logger.info(
            f"Initialized intelligent agent: {config.name}, "
            f"intelligence_level={config.intelligence_level.value}, "
            f"model={config.llm_model}"
        )

    def _get_provider(self) -> LLMProvider:
        """Get or create LLM provider (lazy initialization)."""
        if self._provider is None:
            self._provider = create_provider(model=self.intelligent_config.llm_model)
            self._session = ChatSession(self._provider)
            logger.debug(f"Created LLM provider: {type(self._provider).__name__}")
        return self._provider

    def _get_session(self) -> ChatSession:
        """Get or create chat session."""
        if self._session is None:
            self._get_provider()  # This also creates the session
        return self._session

    def _create_call_budget(self) -> Budget:
        """Create a budget for a single LLM call."""
        return Budget(
            max_usd=self.intelligent_config.max_budget_per_call_usd,
            max_tokens=None  # Use USD limit only
        )

    def _create_execution_budget(self) -> Budget:
        """Create or return the execution budget."""
        if self._execution_budget is None:
            self._execution_budget = Budget(
                max_usd=self.intelligent_config.max_budget_per_execution_usd,
                max_tokens=None
            )
        return self._execution_budget

    def _reset_execution_budget(self):
        """Reset the execution budget for a new execution."""
        self._execution_budget = None

    def _build_system_prompt(self, task: str) -> str:
        """Build a system prompt with domain and regulatory context."""
        context_parts = [
            f"You are an expert AI assistant for {self.intelligent_config.domain_context}.",
            f"Agent: {self.config.name}",
            f"Task: {task}"
        ]

        if self.intelligent_config.regulatory_context:
            context_parts.append(
                f"Regulatory context: {self.intelligent_config.regulatory_context}"
            )

        context_parts.extend([
            "",
            "IMPORTANT RULES:",
            "1. Be factual and precise - this is regulatory compliance context",
            "2. Never hallucinate numbers - only explain calculations already done",
            "3. Cite relevant regulations when applicable",
            "4. Be concise but comprehensive"
        ])

        return "\n".join(context_parts)

    async def _call_llm(
        self,
        messages: List[ChatMessage],
        task_description: str,
        temperature: float = 0.0
    ) -> str:
        """
        Make an LLM call with budget enforcement and metrics tracking.

        Args:
            messages: Chat messages to send
            task_description: Description of the task (for logging)
            temperature: LLM temperature (0.0 = deterministic)

        Returns:
            LLM response text

        Raises:
            BudgetExceeded: If budget would be exceeded
        """
        session = self._get_session()
        budget = self._create_call_budget()

        try:
            response = await session.chat(
                messages=messages,
                budget=budget,
                temperature=temperature
            )

            # Update metrics
            self._intelligence_metrics.llm_calls += 1
            self._intelligence_metrics.total_tokens += response.usage.total_tokens
            self._intelligence_metrics.total_cost_usd += response.usage.cost_usd

            logger.debug(
                f"LLM call for {task_description}: "
                f"tokens={response.usage.total_tokens}, "
                f"cost=${response.usage.cost_usd:.4f}"
            )

            return response.text or ""

        except BudgetExceeded as e:
            logger.warning(f"Budget exceeded for {task_description}: {e}")
            raise

    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(coro)

    # =========================================================================
    # CORE INTELLIGENCE METHODS - These make agents truly intelligent
    # =========================================================================

    def generate_explanation(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        calculation_steps: Optional[List[str]] = None
    ) -> str:
        """
        Generate a natural language explanation of the calculation.

        This is the CORE intelligence method that transforms a deterministic
        calculation into a human-understandable explanation.

        Args:
            input_data: The input that was provided to the agent
            output_data: The output/results from the calculation
            calculation_steps: Optional list of calculation steps taken

        Returns:
            Natural language explanation of what was calculated and why

        Example:
            explanation = self.generate_explanation(
                input_data={"fuel_type": "diesel", "amount": 100, "unit": "gallons"},
                output_data={"co2e_kg": 1024.5, "source": "EPA 2024"},
                calculation_steps=[
                    "Retrieved emission factor: 10.245 kg CO2e/gallon",
                    "Multiplied: 100 gallons * 10.245 = 1024.5 kg CO2e"
                ]
            )
        """
        if not self.intelligent_config.enable_explanations:
            return "Explanations disabled"

        if self.intelligent_config.intelligence_level == IntelligenceLevel.NONE:
            return "Intelligence level set to NONE"

        # Build the prompt
        system_prompt = self._build_system_prompt("Generate explanation for calculation")

        user_content_parts = [
            "Please explain this calculation in clear, professional language:",
            "",
            f"INPUT DATA:",
            str(input_data),
            "",
            f"OUTPUT DATA:",
            str(output_data)
        ]

        if calculation_steps:
            user_content_parts.extend([
                "",
                "CALCULATION STEPS:",
                "\n".join(f"- {step}" for step in calculation_steps)
            ])

        user_content_parts.extend([
            "",
            "Provide a clear explanation that:",
            "1. Summarizes what was calculated",
            "2. Explains the methodology used",
            "3. Notes any relevant regulatory or industry standards",
            "4. Is suitable for inclusion in a compliance report"
        ])

        messages = [
            ChatMessage(role=Role.system, content=system_prompt),
            ChatMessage(role=Role.user, content="\n".join(user_content_parts))
        ]

        try:
            explanation = self._run_async(
                self._call_llm(messages, "generate_explanation")
            )
            self._intelligence_metrics.explanations_generated += 1
            return explanation
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return f"Unable to generate explanation: {str(e)}"

    def generate_recommendations(
        self,
        analysis: Dict[str, Any],
        max_recommendations: int = 5,
        focus_areas: Optional[List[str]] = None
    ) -> List[Recommendation]:
        """
        Generate actionable recommendations based on the analysis.

        This method synthesizes LLM intelligence to provide strategic
        recommendations for emission reduction, efficiency improvement,
        or compliance optimization.

        Args:
            analysis: The analysis data to base recommendations on
            max_recommendations: Maximum number of recommendations
            focus_areas: Optional focus areas (e.g., ["efficiency", "compliance"])

        Returns:
            List of Recommendation objects

        Example:
            recommendations = self.generate_recommendations(
                analysis={
                    "total_emissions": 5000,
                    "top_sources": ["electricity", "natural_gas"],
                    "trend": "increasing",
                    "benchmark_comparison": "above_average"
                },
                max_recommendations=3,
                focus_areas=["reduction", "quick_wins"]
            )
        """
        if not self.intelligent_config.enable_recommendations:
            return []

        if self.intelligent_config.intelligence_level in [IntelligenceLevel.NONE, IntelligenceLevel.BASIC]:
            return []

        system_prompt = self._build_system_prompt("Generate actionable recommendations")

        user_content_parts = [
            "Based on this analysis, provide actionable recommendations:",
            "",
            "ANALYSIS DATA:",
            str(analysis),
            ""
        ]

        if focus_areas:
            user_content_parts.append(f"FOCUS AREAS: {', '.join(focus_areas)}")

        user_content_parts.extend([
            "",
            f"Provide up to {max_recommendations} recommendations in this JSON format:",
            """[
  {
    "id": "REC-001",
    "title": "Short title",
    "description": "Detailed description",
    "category": "efficiency|reduction|compliance|monitoring",
    "priority": "high|medium|low",
    "potential_impact": "Estimated impact",
    "implementation_effort": "low|medium|high",
    "regulatory_relevance": "Relevant regulations if any"
  }
]""",
            "",
            "Return ONLY valid JSON, no other text."
        ])

        messages = [
            ChatMessage(role=Role.system, content=system_prompt),
            ChatMessage(role=Role.user, content="\n".join(user_content_parts))
        ]

        try:
            response = self._run_async(
                self._call_llm(messages, "generate_recommendations")
            )

            import json
            # Parse JSON response
            recommendations_data = json.loads(response)
            recommendations = [
                Recommendation(**rec) for rec in recommendations_data
            ]

            self._intelligence_metrics.recommendations_generated += len(recommendations)
            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return []

    def detect_anomalies(
        self,
        data: Dict[str, Any],
        expected_ranges: Optional[Dict[str, tuple]] = None
    ) -> List[Anomaly]:
        """
        Detect anomalies in the input or output data using LLM reasoning.

        This method uses LLM intelligence to identify unusual patterns,
        outliers, or suspicious data points that might indicate errors
        or require investigation.

        Args:
            data: The data to analyze for anomalies
            expected_ranges: Optional dict of field -> (min, max) expected ranges

        Returns:
            List of Anomaly objects

        Example:
            anomalies = self.detect_anomalies(
                data={"electricity_kwh": 1000000, "area_sqft": 1000},
                expected_ranges={
                    "electricity_kwh": (0, 100000),
                    "area_sqft": (100, 1000000)
                }
            )
        """
        if not self.intelligent_config.enable_anomaly_detection:
            return []

        if self.intelligent_config.intelligence_level not in [IntelligenceLevel.ADVANCED, IntelligenceLevel.FULL]:
            return []

        system_prompt = self._build_system_prompt("Detect anomalies in data")

        user_content_parts = [
            "Analyze this data for anomalies, outliers, or suspicious values:",
            "",
            "DATA:",
            str(data),
            ""
        ]

        if expected_ranges:
            user_content_parts.extend([
                "EXPECTED RANGES:",
                str(expected_ranges),
                ""
            ])

        user_content_parts.extend([
            "Return anomalies in this JSON format:",
            """[
  {
    "id": "ANOM-001",
    "description": "What the anomaly is",
    "severity": "critical|high|medium|low",
    "field": "field_name",
    "expected_range": "what was expected",
    "actual_value": "what was found",
    "reasoning": "Why this is anomalous"
  }
]""",
            "",
            "If no anomalies, return empty array []. Return ONLY valid JSON."
        ])

        messages = [
            ChatMessage(role=Role.system, content=system_prompt),
            ChatMessage(role=Role.user, content="\n".join(user_content_parts))
        ]

        try:
            response = self._run_async(
                self._call_llm(messages, "detect_anomalies")
            )

            import json
            anomalies_data = json.loads(response)
            anomalies = [Anomaly(**anom) for anom in anomalies_data]

            self._intelligence_metrics.anomalies_detected += len(anomalies)
            return anomalies

        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
            return []

    def reason_about(
        self,
        question: str,
        context: Dict[str, Any],
        chain_of_thought: bool = False
    ) -> str:
        """
        Use LLM to reason about a question given context.

        This is the most general intelligence method - use it for
        any reasoning task that doesn't fit the other methods.

        Args:
            question: The question to reason about
            context: Context data to inform the reasoning
            chain_of_thought: If True, include reasoning steps

        Returns:
            LLM's reasoned answer

        Example:
            answer = self.reason_about(
                question="Is this facility compliant with CSRD requirements?",
                context={
                    "emissions_reported": True,
                    "scopes_covered": [1, 2],
                    "verification_status": "unverified"
                },
                chain_of_thought=True
            )
        """
        if self.intelligent_config.intelligence_level == IntelligenceLevel.NONE:
            return "Intelligence disabled"

        system_prompt = self._build_system_prompt("Answer question with reasoning")

        user_content_parts = [
            f"QUESTION: {question}",
            "",
            "CONTEXT:",
            str(context)
        ]

        if chain_of_thought:
            user_content_parts.extend([
                "",
                "Please provide your reasoning step by step before giving the final answer."
            ])

        messages = [
            ChatMessage(role=Role.system, content=system_prompt),
            ChatMessage(role=Role.user, content="\n".join(user_content_parts))
        ]

        try:
            return self._run_async(
                self._call_llm(messages, "reason_about")
            )
        except Exception as e:
            logger.error(f"Failed to reason: {e}")
            return f"Unable to reason: {str(e)}"

    def validate_with_reasoning(
        self,
        input_data: Dict[str, Any],
        validation_rules: Optional[List[str]] = None
    ) -> tuple[bool, str]:
        """
        Validate input data with LLM-powered reasoning.

        This method goes beyond simple schema validation to apply
        domain knowledge and reasoning to validate inputs.

        Args:
            input_data: The data to validate
            validation_rules: Optional list of validation rules to apply

        Returns:
            Tuple of (is_valid, reasoning)

        Example:
            is_valid, reasoning = self.validate_with_reasoning(
                input_data={"fuel_type": "hydrogen", "amount": -100},
                validation_rules=[
                    "Amount must be positive",
                    "Fuel type must be recognized"
                ]
            )
        """
        if self.intelligent_config.intelligence_level == IntelligenceLevel.NONE:
            # Fall back to basic validation
            return self.validate_input(input_data), "Basic validation"

        system_prompt = self._build_system_prompt("Validate input data")

        user_content_parts = [
            "Validate this input data and explain any issues:",
            "",
            "DATA:",
            str(input_data)
        ]

        if validation_rules:
            user_content_parts.extend([
                "",
                "VALIDATION RULES:",
                "\n".join(f"- {rule}" for rule in validation_rules)
            ])

        user_content_parts.extend([
            "",
            "Respond in this format:",
            "VALID: true/false",
            "REASONING: your detailed reasoning"
        ])

        messages = [
            ChatMessage(role=Role.system, content=system_prompt),
            ChatMessage(role=Role.user, content="\n".join(user_content_parts))
        ]

        try:
            response = self._run_async(
                self._call_llm(messages, "validate_with_reasoning")
            )

            # Parse response
            is_valid = "VALID: true" in response.lower() or "valid: true" in response.lower()
            reasoning = response

            return is_valid, reasoning

        except Exception as e:
            logger.error(f"Failed to validate with reasoning: {e}")
            # Fall back to basic validation
            return self.validate_input(input_data), f"Fallback validation: {str(e)}"

    # =========================================================================
    # LIFECYCLE OVERRIDES
    # =========================================================================

    def run(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute the agent with intelligence tracking.

        Extends BaseAgent.run() to:
        1. Reset intelligence metrics for this execution
        2. Create execution budget
        3. Run the agent
        4. Add intelligence metrics to result
        """
        # Reset per-execution state
        self._reset_execution_budget()
        self._intelligence_metrics = IntelligenceMetrics()

        # Run the base agent
        result = super().run(input_data)

        # Add intelligence metrics to metadata
        result.metadata["intelligence_metrics"] = self._intelligence_metrics.dict()

        return result

    def get_intelligence_metrics(self) -> IntelligenceMetrics:
        """Get current intelligence metrics."""
        return self._intelligence_metrics

    def __repr__(self):
        return (
            f"{self.config.name}("
            f"version={self.config.version}, "
            f"intelligence={self.intelligent_config.intelligence_level.value}, "
            f"llm_calls={self._intelligence_metrics.llm_calls})"
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_intelligent_agent_config(
    name: str,
    description: str,
    intelligence_level: IntelligenceLevel = IntelligenceLevel.STANDARD,
    regulatory_context: Optional[str] = None,
    **kwargs
) -> IntelligentAgentConfig:
    """
    Factory function to create an IntelligentAgentConfig.

    Args:
        name: Agent name
        description: Agent description
        intelligence_level: Level of intelligence
        regulatory_context: Regulatory context (CSRD, CBAM, etc.)
        **kwargs: Additional config parameters

    Returns:
        Configured IntelligentAgentConfig
    """
    return IntelligentAgentConfig(
        name=name,
        description=description,
        intelligence_level=intelligence_level,
        regulatory_context=regulatory_context,
        **kwargs
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of IntelligentAgentBase
    """

    class ExampleIntelligentAgent(IntelligentAgentBase):
        """Example intelligent agent demonstrating LLM capabilities."""

        def execute(self, input_data: Dict[str, Any]) -> AgentResult:
            """Execute with intelligence."""
            # 1. Deterministic calculation
            fuel_amount = input_data.get("amount", 0)
            emission_factor = 10.21  # kg CO2e per gallon
            emissions = fuel_amount * emission_factor

            output_data = {
                "co2e_kg": round(emissions, 2),
                "fuel_amount": fuel_amount,
                "emission_factor": emission_factor,
                "source": "EPA 2024"
            }

            # 2. AI-powered explanation
            explanation = self.generate_explanation(
                input_data=input_data,
                output_data=output_data,
                calculation_steps=[
                    f"Retrieved emission factor: {emission_factor} kg CO2e/gallon",
                    f"Calculated: {fuel_amount} × {emission_factor} = {emissions:.2f} kg CO2e"
                ]
            )

            # 3. AI-powered recommendations
            recommendations = self.generate_recommendations(
                analysis={"emissions": emissions, "trend": "stable"},
                max_recommendations=2
            )

            return AgentResult(
                success=True,
                data={
                    **output_data,
                    "explanation": explanation,
                    "recommendations": [r.dict() for r in recommendations]
                },
                timestamp=DeterministicClock.now()
            )

    # Create and run the agent
    config = create_intelligent_agent_config(
        name="ExampleIntelligentAgent",
        description="Demonstrates intelligent agent capabilities",
        intelligence_level=IntelligenceLevel.STANDARD
    )

    agent = ExampleIntelligentAgent(config)

    result = agent.run({
        "fuel_type": "diesel",
        "amount": 100,
        "unit": "gallons"
    })

    print(f"Result: {result.success}")
    print(f"Data: {result.data}")
    print(f"Intelligence Metrics: {agent.get_intelligence_metrics()}")
