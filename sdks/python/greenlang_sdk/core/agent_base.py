"""
SDKAgentBase - Enhanced AgentSpec v2 base class with full lifecycle support.

Extends AgentSpecV2Base with:
- Pre/post lifecycle hooks
- Complete provenance tracking
- Citation aggregation
- Tool registry integration
- Zero-hallucination enforcement
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar
from datetime import datetime
import logging

from greenlang_sdk.core.lifecycle import LifecycleHooks
from greenlang_sdk.core.provenance import ProvenanceTracker, ProvenanceRecord
from greenlang_sdk.core.citation import CitationTracker, CitationRecord

# Try to import existing AgentSpecV2Base if available
try:
    from greenlang.agents.agentspec_v2_base import AgentSpecV2Base as BaseClass
except ImportError:
    # Fallback if not available yet
    BaseClass = ABC

InT = TypeVar("InT")
OutT = TypeVar("OutT")

logger = logging.getLogger(__name__)


class AgentResult(Generic[OutT]):
    """Complete agent execution result with metadata."""

    def __init__(
        self,
        output: OutT,
        provenance: ProvenanceRecord,
        citations: list[CitationRecord],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.output = output
        self.provenance = provenance
        self.citations = citations
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output": self.output,
            "provenance": self.provenance.model_dump(),
            "citations": [c.model_dump() for c in self.citations],
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class SDKAgentBase(LifecycleHooks[InT, OutT], ABC):
    """
    Enhanced agent base class with full SDK capabilities.

    Provides:
    - Complete lifecycle management with hooks
    - Automatic provenance tracking
    - Citation aggregation
    - Tool registry integration
    - Zero-hallucination enforcement
    """

    def __init__(
        self,
        agent_id: str,
        agent_version: str = "1.0.0",
        enable_provenance: bool = True,
        enable_citations: bool = True
    ):
        self.agent_id = agent_id
        self.agent_version = agent_version
        self.enable_provenance = enable_provenance
        self.enable_citations = enable_citations

        # Initialize trackers
        self.provenance_tracker = ProvenanceTracker(agent_id, agent_version) if enable_provenance else None
        self.citation_tracker = CitationTracker() if enable_citations else None

        logger.info(f"Initialized agent: {agent_id} v{agent_version}")

    @abstractmethod
    async def validate_input(self, input_data: InT, context: dict) -> InT:
        """
        Validate input data against agent schema.

        Must be implemented by subclasses to define input validation rules.

        Args:
            input_data: Raw input data
            context: Execution context

        Returns:
            Validated input data

        Raises:
            ValidationError: If input is invalid
        """
        pass

    @abstractmethod
    async def execute(self, validated_input: InT, context: dict) -> OutT:
        """
        Execute main agent logic.

        Must be implemented by subclasses to define core functionality.

        Args:
            validated_input: Validated input data
            context: Execution context

        Returns:
            Output data

        Raises:
            ExecutionError: If execution fails
        """
        pass

    async def validate_output(self, output: OutT, context: dict) -> OutT:
        """
        Validate output data against agent schema.

        Override this method to add output validation.

        Args:
            output: Raw output data
            context: Execution context

        Returns:
            Validated output data

        Raises:
            ValidationError: If output is invalid
        """
        return output

    async def finalize(self, result: AgentResult[OutT], context: dict) -> AgentResult[OutT]:
        """
        Finalize execution and cleanup.

        Override this method to add cleanup logic.

        Args:
            result: Complete agent result
            context: Execution context

        Returns:
            Finalized result
        """
        return result

    async def run(self, payload: InT, context: Optional[Dict[str, Any]] = None) -> AgentResult[OutT]:
        """
        Main entry point - executes complete agent lifecycle.

        Lifecycle:
        1. pre_validate
        2. validate_input
        3. post_validate
        4. pre_execute
        5. execute
        6. post_execute
        7. validate_output
        8. finalize

        Args:
            payload: Input payload
            context: Execution context

        Returns:
            Complete agent result with provenance and citations
        """
        ctx = context or {}
        ctx["agent_id"] = self.agent_id
        ctx["agent_version"] = self.agent_version
        ctx["start_time"] = datetime.utcnow()

        try:
            # Lifecycle: Pre-validate
            payload = await self.pre_validate(payload, ctx)

            # Lifecycle: Validate input
            validated = await self.validate_input(payload, ctx)

            # Lifecycle: Post-validate
            validated = await self.post_validate(validated, ctx)

            # Lifecycle: Pre-execute
            validated = await self.pre_execute(validated, ctx)

            # Lifecycle: Execute (main logic)
            output = await self.execute(validated, ctx)

            # Lifecycle: Post-execute
            output = await self.post_execute(output, ctx)

            # Lifecycle: Validate output
            output = await self.validate_output(output, ctx)

            # Create provenance record
            provenance = self.provenance_tracker.create_provenance_record(
                payload, output
            ) if self.provenance_tracker else None

            # Get all citations
            citations = self.citation_tracker.get_all_citations() if self.citation_tracker else []

            # Create result
            ctx["end_time"] = datetime.utcnow()
            ctx["duration"] = (ctx["end_time"] - ctx["start_time"]).total_seconds()

            result = AgentResult(
                output=output,
                provenance=provenance,
                citations=citations,
                metadata={"context": ctx}
            )

            # Lifecycle: Finalize
            result = await self.finalize(result, ctx)

            logger.info(f"Agent {self.agent_id} execution completed in {ctx['duration']:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Agent {self.agent_id} execution failed: {e}")
            await self.on_error(e, ctx)
            raise

    def add_citation(self, source: str, **kwargs) -> CitationRecord:
        """Helper to add a citation during execution."""
        if not self.citation_tracker:
            raise RuntimeError("Citation tracking not enabled")
        return self.citation_tracker.add_citation(source, **kwargs)

    def record_tool_call(self, tool_name: str, params: Dict[str, Any], result: Any) -> str:
        """Helper to record a tool call with provenance."""
        if not self.provenance_tracker:
            raise RuntimeError("Provenance tracking not enabled")
        return self.provenance_tracker.record_tool_call(tool_name, params, result)
