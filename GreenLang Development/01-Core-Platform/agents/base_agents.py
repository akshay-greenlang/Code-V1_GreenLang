# -*- coding: utf-8 -*-
"""
Base Agent Classes for Intelligence Paradox Architecture
GL Intelligence Infrastructure

Three base classes for agent categorization:
1. DeterministicAgent - Pure calculations, zero AI (CRITICAL PATH)
2. ReasoningAgent - Full AI reasoning with RAG (RECOMMENDATION PATH)
3. InsightAgent - Hybrid (deterministic + AI) (INSIGHT PATH)

Version: 1.0.0
Date: 2025-11-06
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json

from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism import DeterministicClock


@dataclass
class AuditEntry:
    """
    Audit trail entry for deterministic calculations.

    Required for CRITICAL PATH agents to maintain regulatory compliance.
    """

    timestamp: str
    agent_name: str
    operation: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    calculation_trace: List[str]
    input_hash: str
    output_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit entry to dictionary."""
        return {
            "timestamp": self.timestamp,
            "agent_name": self.agent_name,
            "operation": self.operation,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "calculation_trace": self.calculation_trace,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "metadata": self.metadata,
        }


class DeterministicAgent(ABC):
    """
    Base class for CRITICAL PATH agents requiring zero hallucination guarantee.

    Characteristics:
    - 100% deterministic calculations
    - Full audit trail
    - No LLM/AI usage
    - Reproducible results
    - Regulatory compliance ready

    Use for:
    - Emission calculations (CBAM, CSRD, GHG Protocol)
    - Compliance validation
    - Factor lookups
    - Regulatory reporting

    Example:
        class EmissionsCalculator(DeterministicAgent):
            category = AgentCategory.CRITICAL

            def execute(self, inputs: Dict) -> Dict:
                # Pure deterministic calculation
                emissions = inputs["consumption_kwh"] * self.get_emission_factor(...)
                return {"emissions_tco2e": emissions}
    """

    # Agent metadata (subclasses must set)
    category: AgentCategory = AgentCategory.CRITICAL
    metadata: Optional[AgentMetadata] = None

    def __init__(self, enable_audit_trail: bool = True):
        """
        Initialize deterministic agent.

        Args:
            enable_audit_trail: Whether to capture full audit trail
        """
        self.enable_audit_trail = enable_audit_trail
        self.audit_trail: List[AuditEntry] = []

        # Validate that this is a CRITICAL agent
        if self.category != AgentCategory.CRITICAL:
            raise ValueError(
                f"{self.__class__.__name__} must have category=AgentCategory.CRITICAL"
            )

    @abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute deterministic calculation.

        This method MUST be:
        - Deterministic (same inputs → same outputs)
        - Fast (no network calls, no LLM)
        - Pure (no side effects except audit logging)
        - Traceable (all steps documented)

        Args:
            inputs: Input parameters

        Returns:
            Calculation results
        """
        pass

    def _capture_audit_entry(
        self,
        operation: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        calculation_trace: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditEntry:
        """
        Capture audit trail entry.

        Args:
            operation: Operation name
            inputs: Input parameters
            outputs: Calculation outputs
            calculation_trace: Step-by-step calculation trace
            metadata: Additional metadata

        Returns:
            AuditEntry object
        """
        # Create deterministic hashes
        input_hash = hashlib.sha256(
            json.dumps(inputs, sort_keys=True).encode()
        ).hexdigest()

        output_hash = hashlib.sha256(
            json.dumps(outputs, sort_keys=True).encode()
        ).hexdigest()

        entry = AuditEntry(
            timestamp=DeterministicClock.utcnow().isoformat() + "Z",
            agent_name=self.__class__.__name__,
            operation=operation,
            inputs=inputs,
            outputs=outputs,
            calculation_trace=calculation_trace,
            input_hash=input_hash,
            output_hash=output_hash,
            metadata=metadata or {}
        )

        if self.enable_audit_trail:
            self.audit_trail.append(entry)

        return entry

    def get_audit_trail(self) -> List[AuditEntry]:
        """Get full audit trail."""
        return self.audit_trail

    def export_audit_trail(self, file_path: str):
        """
        Export audit trail to JSON file for regulatory compliance.

        Args:
            file_path: Path to save audit trail
        """
        with open(file_path, "w") as f:
            json.dump(
                [entry.to_dict() for entry in self.audit_trail],
                f,
                indent=2
            )

    def verify_reproducibility(
        self,
        inputs: Dict[str, Any],
        expected_outputs: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify that agent produces expected outputs for given inputs.

        Used for regression testing and compliance verification.

        Args:
            inputs: Input parameters
            expected_outputs: Expected outputs

        Returns:
            Tuple of (is_reproducible, error_message)
        """
        try:
            actual_outputs = self.execute(inputs)

            # Compare outputs
            if actual_outputs == expected_outputs:
                return True, None
            else:
                return False, f"Outputs differ: {actual_outputs} != {expected_outputs}"

        except Exception as e:
            return False, f"Execution failed: {str(e)}"


class ReasoningAgent(ABC):
    """
    Base class for RECOMMENDATION PATH agents using full AI reasoning.

    Characteristics:
    - RAG for knowledge retrieval
    - ChatSession for multi-turn reasoning
    - Multi-tool orchestration
    - Temperature ≥ 0.5 for creativity
    - Non-critical path only

    Use for:
    - Technology recommendations
    - Strategic planning
    - Optimization analysis
    - What-if scenarios

    Example:
        class DecarbonizationPlanner(ReasoningAgent):
            category = AgentCategory.RECOMMENDATION

            async def reason(self, context: Dict, session, rag_engine) -> Dict:
                # 1. RAG retrieval
                knowledge = await rag_engine.query(
                    query=f"Decarbonization for {context['industry']}",
                    collections=["case_studies", "best_practices"],
                    top_k=5
                )

                # 2. Multi-tool reasoning
                response = await session.chat(
                    messages=[{
                        "role": "user",
                        "content": f"Recommend solutions for: {context}\\n\\nContext: {knowledge}"
                    }],
                    tools=[tech_db_tool, financial_tool, spatial_tool],
                    temperature=0.7
                )

                # 3. Return structured recommendation
                return parse_recommendation(response)
    """

    # Agent metadata (subclasses must set)
    category: AgentCategory = AgentCategory.RECOMMENDATION
    metadata: Optional[AgentMetadata] = None

    def __init__(self):
        """Initialize reasoning agent."""
        # Validate that this is a RECOMMENDATION agent
        if self.category != AgentCategory.RECOMMENDATION:
            raise ValueError(
                f"{self.__class__.__name__} must have category=AgentCategory.RECOMMENDATION"
            )

    @abstractmethod
    async def reason(
        self,
        context: Dict[str, Any],
        session,  # ChatSession instance
        rag_engine,  # RAGEngine instance
        tools: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute AI reasoning process.

        This method SHOULD:
        - Use RAG for knowledge grounding
        - Use ChatSession for multi-turn reasoning
        - Orchestrate multiple tool calls
        - Use temperature ≥ 0.5 for creative reasoning
        - Return structured recommendations with confidence

        Args:
            context: Input context
            session: ChatSession instance for LLM calls
            rag_engine: RAGEngine instance for knowledge retrieval
            tools: List of available tools

        Returns:
            Reasoning results with recommendations
        """
        pass

    async def _rag_retrieve(
        self,
        query: str,
        rag_engine,
        collections: List[str],
        top_k: int = 5
    ) -> Any:
        """
        Helper method for RAG retrieval.

        Args:
            query: Search query
            rag_engine: RAGEngine instance
            collections: Collections to search
            top_k: Number of results

        Returns:
            RAG query result
        """
        return await rag_engine.query(
            query=query,
            collections=collections,
            top_k=top_k
        )

    def _format_rag_results(self, rag_result: Any) -> str:
        """
        Format RAG results for LLM context.

        Args:
            rag_result: RAG query result

        Returns:
            Formatted string for LLM
        """
        if not rag_result or not rag_result.chunks:
            return "No relevant knowledge found."

        formatted = []
        for i, chunk in enumerate(rag_result.chunks, 1):
            formatted.append(f"{i}. {chunk.text}")

        return "\n\n".join(formatted)

    async def _execute_tool(
        self,
        tool_call: Dict[str, Any],
        tool_registry: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a tool call.

        Args:
            tool_call: Tool call from ChatSession
            tool_registry: Dictionary of available tools

        Returns:
            Tool execution result
        """
        tool_name = tool_call["name"]
        arguments = json.loads(tool_call["arguments"])

        if tool_name not in tool_registry:
            raise ValueError(f"Tool not found: {tool_name}")

        tool_func = tool_registry[tool_name]
        return await tool_func(**arguments)


class InsightAgent(ABC):
    """
    Base class for INSIGHT PATH agents with hybrid architecture.

    Characteristics:
    - Deterministic calculations (numbers)
    - AI-generated insights (narratives)
    - Optional RAG for context
    - Temperature ≤ 0.7 for consistency
    - Split responsibilities clearly

    Use for:
    - Anomaly investigation
    - Forecast explanation
    - Benchmark insights
    - Trend analysis

    Example:
        class AnomalyInvestigator(InsightAgent):
            category = AgentCategory.INSIGHT

            def calculate(self, data: Dict) -> Dict:
                # Deterministic anomaly detection
                anomalies = self.isolation_forest.detect(data)
                return {"anomalies": anomalies}

            async def explain(self, anomalies: Dict, session, rag_engine) -> str:
                # AI-generated explanation
                context = await rag_engine.query(
                    query=f"Anomalies in {anomalies['metric']}",
                    collections=["historical_data"],
                    top_k=3
                )

                response = await session.chat(
                    messages=[{
                        "role": "user",
                        "content": f"Explain these anomalies: {anomalies}\\n\\nContext: {context}"
                    }],
                    temperature=0.6
                )

                return response.text
    """

    # Agent metadata (subclasses must set)
    category: AgentCategory = AgentCategory.INSIGHT
    metadata: Optional[AgentMetadata] = None

    def __init__(self, enable_audit_trail: bool = False):
        """
        Initialize insight agent.

        Args:
            enable_audit_trail: Whether to capture calculation audit trail
        """
        self.enable_audit_trail = enable_audit_trail
        self.audit_trail: List[AuditEntry] = []

        # Validate that this is an INSIGHT agent
        if self.category != AgentCategory.INSIGHT:
            raise ValueError(
                f"{self.__class__.__name__} must have category=AgentCategory.INSIGHT"
            )

    @abstractmethod
    def calculate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute deterministic calculation.

        This method MUST be:
        - Deterministic (same inputs → same outputs)
        - Fast
        - Pure (no side effects)

        Args:
            inputs: Input parameters

        Returns:
            Calculation results (numbers, metrics, statistics)
        """
        pass

    @abstractmethod
    async def explain(
        self,
        calculation_result: Dict[str, Any],
        context: Dict[str, Any],
        session,  # ChatSession instance
        rag_engine,  # RAGEngine instance
        temperature: float = 0.6
    ) -> str:
        """
        Generate AI-powered explanation/insight.

        This method SHOULD:
        - Use calculation results (not recalculate)
        - Use RAG for historical context
        - Use ChatSession for narrative generation
        - Use temperature ≤ 0.7 for consistency
        - Focus on WHY, not WHAT

        Args:
            calculation_result: Results from calculate()
            context: Additional context
            session: ChatSession instance
            rag_engine: RAGEngine instance
            temperature: LLM temperature

        Returns:
            Natural language explanation/insight
        """
        pass

    def _capture_calculation_audit(
        self,
        operation: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        calculation_trace: List[str]
    ):
        """Capture audit trail for deterministic calculation."""
        if self.enable_audit_trail:
            input_hash = hashlib.sha256(
                json.dumps(inputs, sort_keys=True).encode()
            ).hexdigest()

            output_hash = hashlib.sha256(
                json.dumps(outputs, sort_keys=True).encode()
            ).hexdigest()

            entry = AuditEntry(
                timestamp=DeterministicClock.utcnow().isoformat() + "Z",
                agent_name=self.__class__.__name__,
                operation=operation,
                inputs=inputs,
                outputs=outputs,
                calculation_trace=calculation_trace,
                input_hash=input_hash,
                output_hash=output_hash,
            )

            self.audit_trail.append(entry)


if __name__ == "__main__":
    print("Base Agent Classes for Intelligence Paradox Architecture")
    print("=" * 80)

    print("\n1. DeterministicAgent (CRITICAL PATH)")
    print("   - Use for: Regulatory calculations")
    print("   - Characteristics: Zero AI, full audit trail")
    print("   - Example: EmissionsCalculator, ComplianceValidator")

    print("\n2. ReasoningAgent (RECOMMENDATION PATH)")
    print("   - Use for: AI-driven recommendations")
    print("   - Characteristics: RAG + ChatSession + tools")
    print("   - Example: DecarbonizationPlanner, TechnologyRecommender")

    print("\n3. InsightAgent (INSIGHT PATH)")
    print("   - Use for: Hybrid analysis")
    print("   - Characteristics: Deterministic calculations + AI insights")
    print("   - Example: AnomalyInvestigator, ForecastExplainer")

    print("\n" + "=" * 80)
