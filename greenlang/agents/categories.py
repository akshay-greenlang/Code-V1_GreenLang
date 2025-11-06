"""
Agent Category Definitions
GL Intelligence Infrastructure

Defines the three agent categories for the Intelligence Paradox fix:
- CRITICAL PATH: Regulatory/compliance calculations (100% deterministic)
- RECOMMENDATION PATH: Decision support with AI reasoning
- INSIGHT PATH: Analysis and investigation with AI enhancement

Version: 1.0.0
Date: 2025-11-06
"""

from enum import Enum
from typing import Optional
from dataclasses import dataclass


class AgentCategory(str, Enum):
    """
    Agent category classification for Intelligence Paradox architecture.

    Categories determine agent behavior and AI usage:
    - CRITICAL: Zero AI, 100% deterministic, audit trail required
    - RECOMMENDATION: AI reasoning with RAG, multi-tool orchestration
    - INSIGHT: Hybrid (deterministic calculations + AI analysis)
    - UTILITY: Framework/testing code, no specific pattern
    """

    CRITICAL = "critical_path"
    RECOMMENDATION = "recommendation_path"
    INSIGHT = "insight_path"
    UTILITY = "utility"

    @property
    def allows_llm(self) -> bool:
        """Whether this category allows LLM usage."""
        return self in (AgentCategory.RECOMMENDATION, AgentCategory.INSIGHT)

    @property
    def requires_determinism(self) -> bool:
        """Whether this category requires deterministic calculations."""
        return self in (AgentCategory.CRITICAL, AgentCategory.INSIGHT)

    @property
    def requires_audit_trail(self) -> bool:
        """Whether this category requires audit trail."""
        return self == AgentCategory.CRITICAL

    @property
    def allows_rag(self) -> bool:
        """Whether this category can use RAG for knowledge retrieval."""
        return self in (AgentCategory.RECOMMENDATION, AgentCategory.INSIGHT)

    @property
    def allows_tools(self) -> bool:
        """Whether this category can use ChatSession tools."""
        return self in (AgentCategory.RECOMMENDATION, AgentCategory.INSIGHT)

    @property
    def description(self) -> str:
        """Human-readable description of category."""
        descriptions = {
            AgentCategory.CRITICAL: (
                "Regulatory/compliance calculations requiring zero hallucination guarantee. "
                "100% deterministic, full audit trail, no LLM reasoning."
            ),
            AgentCategory.RECOMMENDATION: (
                "Decision support agents using AI reasoning with RAG retrieval and "
                "multi-tool orchestration. Non-critical path, creative problem-solving."
            ),
            AgentCategory.INSIGHT: (
                "Hybrid agents with deterministic calculations enhanced by AI analysis. "
                "Numbers are deterministic, insights and narratives use LLM reasoning."
            ),
            AgentCategory.UTILITY: (
                "Framework code, base classes, and testing infrastructure. "
                "No specific agent pattern."
            )
        }
        return descriptions[self]


@dataclass
class AgentMetadata:
    """
    Metadata for agent categorization and compliance tracking.

    Attributes:
        name: Agent name
        category: Agent category (CRITICAL, RECOMMENDATION, INSIGHT, UTILITY)
        uses_chat_session: Whether agent uses ChatSession API
        uses_rag: Whether agent uses RAG for knowledge retrieval
        uses_tools: Whether agent uses ChatSession tools
        critical_for_compliance: Whether agent output is used in regulatory reporting
        audit_trail_required: Whether calculations must be fully auditable
        transformation_priority: Priority for AI transformation (HIGH, MEDIUM, LOW, N/A)
        description: Brief description of agent purpose
    """

    name: str
    category: AgentCategory
    uses_chat_session: bool = False
    uses_rag: bool = False
    uses_tools: bool = False
    critical_for_compliance: bool = False
    audit_trail_required: bool = False
    transformation_priority: Optional[str] = None
    description: str = ""

    def __post_init__(self):
        """Validate metadata consistency."""
        # Critical path agents cannot use ChatSession
        if self.category == AgentCategory.CRITICAL and self.uses_chat_session:
            raise ValueError(
                f"Agent {self.name}: CRITICAL category agents cannot use ChatSession"
            )

        # Critical path agents must have audit trails
        if self.category == AgentCategory.CRITICAL:
            self.audit_trail_required = True
            self.critical_for_compliance = True

        # RAG and tools require ChatSession
        if self.uses_rag or self.uses_tools:
            if not self.uses_chat_session:
                raise ValueError(
                    f"Agent {self.name}: RAG and tools require ChatSession"
                )

        # Validate category allows LLM usage
        if self.uses_chat_session and not self.category.allows_llm:
            raise ValueError(
                f"Agent {self.name}: Category {self.category} does not allow LLM usage"
            )

    @property
    def is_ai_enabled(self) -> bool:
        """Whether this agent uses AI capabilities."""
        return self.uses_chat_session or self.uses_rag or self.uses_tools

    @property
    def compliance_level(self) -> str:
        """Compliance criticality level."""
        if self.critical_for_compliance and self.audit_trail_required:
            return "REGULATORY CRITICAL"
        elif self.critical_for_compliance:
            return "COMPLIANCE IMPORTANT"
        else:
            return "NON-CRITICAL"

    def to_dict(self) -> dict:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "category": self.category.value,
            "uses_chat_session": self.uses_chat_session,
            "uses_rag": self.uses_rag,
            "uses_tools": self.uses_tools,
            "critical_for_compliance": self.critical_for_compliance,
            "audit_trail_required": self.audit_trail_required,
            "transformation_priority": self.transformation_priority,
            "description": self.description,
            "is_ai_enabled": self.is_ai_enabled,
            "compliance_level": self.compliance_level,
        }


# Example agent metadata instances for documentation
EXAMPLE_METADATA = {
    "calculator_agent": AgentMetadata(
        name="calculator_agent",
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        transformation_priority="N/A",
        description="ESRS metrics calculator with Zero Hallucination Guarantee"
    ),
    "recommendation_agent_ai": AgentMetadata(
        name="recommendation_agent_ai",
        category=AgentCategory.RECOMMENDATION,
        uses_chat_session=True,
        uses_rag=True,
        uses_tools=True,
        critical_for_compliance=False,
        transformation_priority="LOW (Already transformed)",
        description="AI-powered decarbonization recommendations using RAG and multi-tool reasoning"
    ),
    "benchmark_agent": AgentMetadata(
        name="benchmark_agent",
        category=AgentCategory.INSIGHT,
        uses_chat_session=False,  # Will be transformed
        uses_rag=False,  # Will be added
        critical_for_compliance=False,
        transformation_priority="HIGH",
        description="Peer benchmarking with AI-generated insights (calculations deterministic)"
    ),
}


def validate_agent_category(
    category: AgentCategory,
    uses_chat_session: bool,
    uses_rag: bool,
    uses_tools: bool,
    critical_for_compliance: bool
) -> tuple[bool, list[str]]:
    """
    Validate that agent configuration matches category requirements.

    Args:
        category: Agent category
        uses_chat_session: Whether agent uses ChatSession
        uses_rag: Whether agent uses RAG
        uses_tools: Whether agent uses tools
        critical_for_compliance: Whether agent is compliance-critical

    Returns:
        Tuple of (is_valid, list of violations)
    """
    violations = []

    # Rule 1: CRITICAL agents cannot use ChatSession
    if category == AgentCategory.CRITICAL and uses_chat_session:
        violations.append("CRITICAL agents must not use ChatSession (zero AI required)")

    # Rule 2: CRITICAL agents cannot use RAG
    if category == AgentCategory.CRITICAL and uses_rag:
        violations.append("CRITICAL agents must not use RAG (deterministic only)")

    # Rule 3: CRITICAL agents cannot use tools
    if category == AgentCategory.CRITICAL and uses_tools:
        violations.append("CRITICAL agents must not use ChatSession tools")

    # Rule 4: RAG/tools require ChatSession
    if (uses_rag or uses_tools) and not uses_chat_session:
        violations.append("RAG and tools require ChatSession")

    # Rule 5: RECOMMENDATION agents should use ChatSession
    if category == AgentCategory.RECOMMENDATION and not uses_chat_session:
        violations.append("RECOMMENDATION agents should use ChatSession for AI reasoning")

    # Rule 6: Compliance-critical agents should be CRITICAL category
    if critical_for_compliance and category != AgentCategory.CRITICAL:
        violations.append("Compliance-critical agents should be CRITICAL category")

    return len(violations) == 0, violations


if __name__ == "__main__":
    # Demonstrate category usage
    print("Agent Categories:")
    print("=" * 80)

    for category in AgentCategory:
        print(f"\n{category.name}:")
        print(f"  Value: {category.value}")
        print(f"  Allows LLM: {category.allows_llm}")
        print(f"  Requires Determinism: {category.requires_determinism}")
        print(f"  Requires Audit Trail: {category.requires_audit_trail}")
        print(f"  Allows RAG: {category.allows_rag}")
        print(f"  Allows Tools: {category.allows_tools}")
        print(f"  Description: {category.description}")

    print("\n" + "=" * 80)
    print("Example Agent Metadata:")
    print("=" * 80)

    for name, metadata in EXAMPLE_METADATA.items():
        print(f"\n{name}:")
        print(f"  Category: {metadata.category.name}")
        print(f"  AI Enabled: {metadata.is_ai_enabled}")
        print(f"  Compliance Level: {metadata.compliance_level}")
        print(f"  Priority: {metadata.transformation_priority}")
        print(f"  Description: {metadata.description}")
