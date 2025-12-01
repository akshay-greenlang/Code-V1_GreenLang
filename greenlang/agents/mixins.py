# -*- coding: utf-8 -*-
"""
GreenLang Agent Category Mixins
================================

This module provides category mixins for the AgentSpecV2Base inheritance pattern.
These mixins standardize agent behavior based on their category (DETERMINISTIC, REASONING, INSIGHT).

Architecture:
    AgentSpecV2Base[InT, OutT]
    + DeterministicMixin → Zero-hallucination calculation agents
    + ReasoningMixin → AI-powered reasoning agents
    + InsightMixin → Hybrid calculation + AI agents

Author: GreenLang Framework Team
Date: December 2025
Status: Production Ready
"""

from abc import ABC
from typing import Any, Dict, List, Optional
import hashlib
import json
import logging
from datetime import datetime

from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# ==============================================================================
# Deterministic Mixin (Zero Hallucination)
# ==============================================================================

class DeterministicMixin(ABC):
    """
    Mixin for deterministic agents requiring zero-hallucination guarantee.

    Use this mixin for agents in the CRITICAL PATH that must produce:
    - 100% reproducible results (same input → same output)
    - Full audit trail for regulatory compliance
    - NO LLM/AI usage in calculation path
    - Provenance tracking with SHA-256 hashes

    Applicable to:
    - Emission calculations (CBAM, CSRD, GHG Protocol)
    - Compliance validation
    - Factor lookups
    - Regulatory reporting

    Example:
        >>> class EmissionsCalculator(AgentSpecV2Base[EmissionsInput, EmissionsOutput], DeterministicMixin):
        ...     def execute_impl(self, validated_input, context):
        ...         # Pure deterministic calculation
        ...         emissions = validated_input.activity_data * self.get_emission_factor(...)
        ...
        ...         # Track audit trail
        ...         self.capture_audit_entry(
        ...             operation="emissions_calculation",
        ...             inputs=validated_input.dict(),
        ...             outputs={"emissions": emissions},
        ...             calculation_trace=[f"activity_data * emission_factor = {emissions}"]
        ...         )
        ...
        ...         return EmissionsOutput(emissions=emissions)
    """

    # Category metadata
    category: str = "DETERMINISTIC"
    temperature: float = 0.0  # No temperature (no LLM)
    requires_audit_trail: bool = True
    requires_provenance: bool = True
    allows_llm_in_calculation: bool = False  # CRITICAL: No LLM in calculation path

    def __init__(self, *args, **kwargs):
        """Initialize deterministic mixin."""
        super().__init__(*args, **kwargs)
        self._audit_trail: List[Dict[str, Any]] = []
        self._calculation_traces: List[str] = []

    def capture_audit_entry(
        self,
        operation: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        calculation_trace: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Capture audit trail entry for regulatory compliance.

        This method creates a complete audit record including:
        - Timestamp (deterministic)
        - Operation name
        - Input/output hashes (SHA-256)
        - Step-by-step calculation trace
        - Provenance metadata

        Args:
            operation: Name of the operation being audited
            inputs: Input parameters (will be hashed)
            outputs: Calculation outputs (will be hashed)
            calculation_trace: Step-by-step calculation trace
            metadata: Additional metadata for audit record

        Returns:
            Audit entry dictionary

        Example:
            >>> self.capture_audit_entry(
            ...     operation="scope1_emissions",
            ...     inputs={"fuel_consumption": 100, "emission_factor": 2.3},
            ...     outputs={"emissions_kg_co2e": 230},
            ...     calculation_trace=["100 * 2.3 = 230"]
            ... )
        """
        # Create deterministic hashes
        input_hash = hashlib.sha256(
            json.dumps(inputs, sort_keys=True).encode()
        ).hexdigest()

        output_hash = hashlib.sha256(
            json.dumps(outputs, sort_keys=True).encode()
        ).hexdigest()

        audit_entry = {
            "timestamp": DeterministicClock.utcnow().isoformat() + "Z",
            "agent_name": self.__class__.__name__ if hasattr(self, '__class__') else "Unknown",
            "operation": operation,
            "inputs": inputs,
            "outputs": outputs,
            "calculation_trace": calculation_trace,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "metadata": metadata or {}
        }

        self._audit_trail.append(audit_entry)
        logger.debug(f"Audit entry captured for operation: {operation}")

        return audit_entry

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """
        Get complete audit trail for this agent execution.

        Returns:
            List of audit entries
        """
        return self._audit_trail.copy()

    def export_audit_trail(self, file_path: str) -> None:
        """
        Export audit trail to JSON file for regulatory compliance.

        Args:
            file_path: Path to save audit trail JSON
        """
        with open(file_path, "w") as f:
            json.dump(self._audit_trail, f, indent=2, default=str)

        logger.info(f"Audit trail exported to {file_path}")

    def validate_determinism(self, result: Any) -> bool:
        """
        Validate that result is deterministic (no random elements).

        This is a hook for subclasses to implement custom determinism checks.

        Args:
            result: Result to validate

        Returns:
            True if result is deterministic
        """
        # Default implementation: always True
        # Subclasses can override for custom validation
        return True

    def calculate_provenance_hash(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> str:
        """
        Calculate SHA-256 provenance hash for complete reproducibility.

        Args:
            inputs: Input data
            outputs: Output data

        Returns:
            SHA-256 hash (hex string)
        """
        provenance_str = f"{json.dumps(inputs, sort_keys=True)}{json.dumps(outputs, sort_keys=True)}"
        return hashlib.sha256(provenance_str.encode()).hexdigest()


# ==============================================================================
# Reasoning Mixin (AI-Powered)
# ==============================================================================

class ReasoningMixin(ABC):
    """
    Mixin for AI reasoning agents using full LLM capabilities.

    Use this mixin for agents in the RECOMMENDATION PATH that provide:
    - RAG-based knowledge retrieval
    - Multi-turn reasoning with ChatSession
    - Multi-tool orchestration
    - Creative recommendations (temperature ≥ 0.5)
    - NON-CRITICAL PATH ONLY (not for regulatory calculations)

    Applicable to:
    - Technology recommendations
    - Strategic planning
    - Optimization analysis
    - What-if scenarios

    Example:
        >>> class DecarbonizationPlanner(AgentSpecV2Base[PlanInput, PlanOutput], ReasoningMixin):
        ...     async def execute_impl(self, validated_input, context):
        ...         # 1. RAG retrieval for knowledge grounding
        ...         knowledge = await self.rag_retrieve(
        ...             query=f"Decarbonization for {validated_input.industry}",
        ...             collections=["case_studies", "best_practices"],
        ...             top_k=5
        ...         )
        ...
        ...         # 2. Multi-tool reasoning
        ...         response = await self.chat_session.chat(
        ...             messages=[{
        ...                 "role": "user",
        ...                 "content": f"Recommend solutions for: {validated_input}\\n\\nContext: {knowledge}"
        ...             }],
        ...             tools=[tech_db_tool, financial_tool],
        ...             temperature=0.7
        ...         )
        ...
        ...         return PlanOutput.parse_obj(response)
    """

    # Category metadata
    category: str = "REASONING"
    temperature: float = 0.7  # Allow creativity for reasoning
    requires_rag: bool = True
    requires_chat_session: bool = True
    requires_audit_trail: bool = False  # Not required for recommendations
    min_temperature: float = 0.5  # Minimum temperature for creative reasoning
    max_temperature: float = 1.0  # Maximum temperature

    def __init__(self, *args, **kwargs):
        """Initialize reasoning mixin."""
        super().__init__(*args, **kwargs)
        self._rag_engine = None
        self._chat_session = None
        self._tool_registry: Dict[str, Any] = {}

    def set_rag_engine(self, rag_engine: Any) -> None:
        """
        Set RAG engine for knowledge retrieval.

        Args:
            rag_engine: RAGEngine instance
        """
        self._rag_engine = rag_engine
        logger.debug(f"RAG engine set for {self.__class__.__name__}")

    def set_chat_session(self, chat_session: Any) -> None:
        """
        Set ChatSession for LLM reasoning.

        Args:
            chat_session: ChatSession instance
        """
        self._chat_session = chat_session
        logger.debug(f"Chat session set for {self.__class__.__name__}")

    def register_tool(self, name: str, tool: Any) -> None:
        """
        Register a tool for LLM function calling.

        Args:
            name: Tool name
            tool: Tool function or callable
        """
        self._tool_registry[name] = tool
        logger.debug(f"Tool registered: {name}")

    async def rag_retrieve(
        self,
        query: str,
        collections: List[str],
        top_k: int = 5
    ) -> Any:
        """
        Retrieve relevant knowledge from RAG collections.

        Args:
            query: Search query
            collections: RAG collection names
            top_k: Number of results to retrieve

        Returns:
            RAG query result

        Raises:
            ValueError: If RAG engine not set
        """
        if not self._rag_engine:
            raise ValueError("RAG engine not set. Call set_rag_engine() first.")

        result = await self._rag_engine.query(
            query=query,
            collections=collections,
            top_k=top_k
        )

        logger.debug(f"RAG retrieved {len(result.chunks) if hasattr(result, 'chunks') else 0} chunks")
        return result

    def format_rag_results(self, rag_result: Any) -> str:
        """
        Format RAG results for LLM context.

        Args:
            rag_result: RAG query result

        Returns:
            Formatted string for LLM prompt
        """
        if not rag_result or not hasattr(rag_result, 'chunks') or not rag_result.chunks:
            return "No relevant knowledge found."

        formatted = []
        for i, chunk in enumerate(rag_result.chunks, 1):
            formatted.append(f"{i}. {chunk.text}")

        return "\n\n".join(formatted)

    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a registered tool.

        Args:
            tool_name: Name of tool to execute
            **kwargs: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not found
        """
        if tool_name not in self._tool_registry:
            raise ValueError(f"Tool not found: {tool_name}")

        tool_func = self._tool_registry[tool_name]

        # Execute tool (supports both sync and async)
        if asyncio.iscoroutinefunction(tool_func):
            result = await tool_func(**kwargs)
        else:
            result = tool_func(**kwargs)

        logger.debug(f"Tool executed: {tool_name}")
        return result


# ==============================================================================
# Insight Mixin (Hybrid)
# ==============================================================================

class InsightMixin(ABC):
    """
    Mixin for hybrid agents combining deterministic calculations with AI insights.

    Use this mixin for agents in the INSIGHT PATH that provide:
    - Deterministic calculations (numbers, metrics) - NO LLM
    - AI-generated explanations (narratives, insights) - WITH LLM
    - Optional RAG for historical context
    - Moderate temperature (≤ 0.7) for consistency
    - Clear separation of responsibilities

    Applicable to:
    - Anomaly investigation (detect with ML, explain with AI)
    - Forecast explanation (calculate with stats, explain with AI)
    - Benchmark insights (calculate with math, interpret with AI)
    - Trend analysis (detect with algorithms, narrate with AI)

    Example:
        >>> class AnomalyInvestigator(AgentSpecV2Base[AnomalyInput, AnomalyOutput], InsightMixin):
        ...     def calculate(self, validated_input, context):
        ...         # DETERMINISTIC: No LLM in calculation path
        ...         anomalies = self.isolation_forest.detect(validated_input.data)
        ...
        ...         # Capture audit trail for calculations
        ...         self.capture_calculation_audit(
        ...             operation="anomaly_detection",
        ...             inputs=validated_input.dict(),
        ...             outputs={"anomalies": anomalies},
        ...             calculation_trace=["IsolationForest.fit_predict()"]
        ...         )
        ...
        ...         return anomalies
        ...
        ...     async def explain(self, calculation_result, validated_input, context):
        ...         # AI-POWERED: LLM for explanation only
        ...         rag_context = await self.rag_retrieve(
        ...             query=f"Anomalies in {validated_input.metric}",
        ...             collections=["historical_data"],
        ...             top_k=3
        ...         )
        ...
        ...         explanation = await self.chat_session.chat(
        ...             messages=[{
        ...                 "role": "user",
        ...                 "content": f"Explain these anomalies: {calculation_result}\\n\\nContext: {rag_context}"
        ...             }],
        ...             temperature=0.6
        ...         )
        ...
        ...         return explanation.text
        ...
        ...     def execute_impl(self, validated_input, context):
        ...         # Orchestrate: calculate first, then explain
        ...         anomalies = self.calculate(validated_input, context)
        ...         explanation = await self.explain(anomalies, validated_input, context)
        ...
        ...         return AnomalyOutput(
        ...             anomalies=anomalies,
        ...             explanation=explanation
        ...         )
    """

    # Category metadata
    category: str = "INSIGHT"
    temperature: float = 0.6  # Moderate temperature for consistency
    requires_audit_trail: bool = True  # For calculation portion
    requires_explanation: bool = True  # For AI portion
    max_temperature: float = 0.7  # Maximum temperature for insights

    def __init__(self, *args, **kwargs):
        """Initialize insight mixin."""
        super().__init__(*args, **kwargs)
        self._audit_trail: List[Dict[str, Any]] = []
        self._rag_engine = None
        self._chat_session = None

    def capture_calculation_audit(
        self,
        operation: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        calculation_trace: List[str]
    ) -> Dict[str, Any]:
        """
        Capture audit trail for deterministic calculation portion.

        Args:
            operation: Operation name
            inputs: Input parameters
            outputs: Calculation outputs
            calculation_trace: Step-by-step trace

        Returns:
            Audit entry
        """
        input_hash = hashlib.sha256(
            json.dumps(inputs, sort_keys=True).encode()
        ).hexdigest()

        output_hash = hashlib.sha256(
            json.dumps(outputs, sort_keys=True).encode()
        ).hexdigest()

        audit_entry = {
            "timestamp": DeterministicClock.utcnow().isoformat() + "Z",
            "agent_name": self.__class__.__name__ if hasattr(self, '__class__') else "Unknown",
            "operation": operation,
            "inputs": inputs,
            "outputs": outputs,
            "calculation_trace": calculation_trace,
            "input_hash": input_hash,
            "output_hash": output_hash,
        }

        self._audit_trail.append(audit_entry)
        logger.debug(f"Calculation audit captured: {operation}")

        return audit_entry

    def set_rag_engine(self, rag_engine: Any) -> None:
        """Set RAG engine for explanation context."""
        self._rag_engine = rag_engine

    def set_chat_session(self, chat_session: Any) -> None:
        """Set ChatSession for explanation generation."""
        self._chat_session = chat_session

    async def rag_retrieve(
        self,
        query: str,
        collections: List[str],
        top_k: int = 5
    ) -> Any:
        """
        Retrieve RAG context for explanation.

        Args:
            query: Search query
            collections: RAG collections
            top_k: Number of results

        Returns:
            RAG result
        """
        if not self._rag_engine:
            raise ValueError("RAG engine not set. Call set_rag_engine() first.")

        return await self._rag_engine.query(
            query=query,
            collections=collections,
            top_k=top_k
        )

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get audit trail for calculation portion."""
        return self._audit_trail.copy()


# ==============================================================================
# Helper Functions
# ==============================================================================

def get_category_mixin(category: str) -> type:
    """
    Get the appropriate mixin class for a given category.

    Args:
        category: Category name ("DETERMINISTIC", "REASONING", "INSIGHT")

    Returns:
        Mixin class

    Raises:
        ValueError: If category is unknown
    """
    category_map = {
        "DETERMINISTIC": DeterministicMixin,
        "REASONING": ReasoningMixin,
        "INSIGHT": InsightMixin,
    }

    if category not in category_map:
        raise ValueError(
            f"Unknown category: {category}. "
            f"Valid categories: {list(category_map.keys())}"
        )

    return category_map[category]


def validate_mixin_usage(agent_class: type) -> None:
    """
    Validate that an agent class uses mixins correctly.

    Checks:
    - Agent inherits from AgentSpecV2Base
    - Agent has exactly one category mixin
    - Category mixin is appropriate for agent's purpose

    Args:
        agent_class: Agent class to validate

    Raises:
        TypeError: If validation fails
    """
    # Check if inherits from AgentSpecV2Base
    from greenlang.agents.agentspec_v2_base import AgentSpecV2Base

    if not any(issubclass(base, AgentSpecV2Base) for base in agent_class.__mro__ if base != AgentSpecV2Base):
        raise TypeError(
            f"{agent_class.__name__} must inherit from AgentSpecV2Base"
        )

    # Check for category mixins
    mixins = []
    for base in agent_class.__mro__:
        if base in (DeterministicMixin, ReasoningMixin, InsightMixin):
            mixins.append(base)

    if len(mixins) == 0:
        raise TypeError(
            f"{agent_class.__name__} must inherit from exactly one category mixin "
            f"(DeterministicMixin, ReasoningMixin, or InsightMixin)"
        )

    if len(mixins) > 1:
        raise TypeError(
            f"{agent_class.__name__} inherits from multiple category mixins: {mixins}. "
            f"Agent must inherit from exactly one category mixin."
        )

    logger.info(f"Mixin validation passed for {agent_class.__name__} ({mixins[0].__name__})")


# ==============================================================================
# Module Exports
# ==============================================================================

__all__ = [
    "DeterministicMixin",
    "ReasoningMixin",
    "InsightMixin",
    "get_category_mixin",
    "validate_mixin_usage",
]
