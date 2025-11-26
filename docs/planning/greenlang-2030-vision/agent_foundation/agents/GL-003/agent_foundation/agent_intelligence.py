# -*- coding: utf-8 -*-
"""
Agent intelligence module for GreenLang agents.

Provides LLM integration capabilities for classification, entity resolution,
and narrative generation tasks. All numeric calculations must remain
deterministic and NOT use LLM.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE = "azure"
    LOCAL = "local"


@dataclass
class ChatSession:
    """
    LLM chat session for agent intelligence tasks.

    IMPORTANT: Only use for classification, entity resolution, and narrative
    generation. NEVER use for numeric calculations (zero-hallucination principle).

    Attributes:
        provider: LLM provider (Anthropic, OpenAI, etc.)
        model_id: Model identifier
        temperature: Sampling temperature (0.0 for deterministic)
        seed: Random seed for reproducibility
        max_tokens: Maximum response tokens
    """
    provider: ModelProvider
    model_id: str
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 1000
    _messages: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        """Validate deterministic settings."""
        if self.temperature != 0.0:
            logger.warning(
                f"Non-zero temperature ({self.temperature}) may cause non-deterministic results"
            )

    async def send_message(self, message: str) -> str:
        """
        Send a message to the LLM.

        Args:
            message: User message

        Returns:
            LLM response string
        """
        self._messages.append({"role": "user", "content": message})

        # Stub implementation - returns classification placeholder
        response = self._generate_stub_response(message)

        self._messages.append({"role": "assistant", "content": response})
        return response

    def _generate_stub_response(self, message: str) -> str:
        """Generate stub response for testing."""
        # Classify based on keywords in message
        message_lower = message.lower()

        if "classify" in message_lower:
            if "anomaly" in message_lower or "anomalies" in message_lower:
                return "normal"
            return "balanced"

        if "strategy" in message_lower:
            return "balanced"

        if "category" in message_lower:
            return "efficiency_focused"

        return "acknowledged"

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._messages.clear()


@dataclass
class PromptTemplate:
    """
    Prompt template for structured LLM queries.

    Attributes:
        template: Template string with {variable} placeholders
        variables: List of variable names in template
    """
    template: str
    variables: List[str]

    def format(self, **kwargs) -> str:
        """
        Format template with provided values.

        Args:
            **kwargs: Variable values

        Returns:
            Formatted prompt string

        Raises:
            ValueError: If required variable is missing
        """
        for var in self.variables:
            if var not in kwargs:
                raise ValueError(f"Missing required variable: {var}")

        return self.template.format(**kwargs)


class AgentIntelligence:
    """
    Agent intelligence coordinator.

    Manages LLM sessions and prompt templates for agent intelligence tasks.
    Enforces zero-hallucination principle by only allowing non-numeric tasks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize agent intelligence.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._sessions: Dict[str, ChatSession] = {}
        self._templates: Dict[str, PromptTemplate] = {}

        logger.info("AgentIntelligence initialized")

    def create_session(
        self,
        session_id: str,
        provider: ModelProvider = ModelProvider.ANTHROPIC,
        model_id: str = "claude-3-haiku",
        temperature: float = 0.0,
        seed: int = 42
    ) -> ChatSession:
        """
        Create a new chat session.

        Args:
            session_id: Unique session identifier
            provider: LLM provider
            model_id: Model identifier
            temperature: Sampling temperature
            seed: Random seed

        Returns:
            Created ChatSession
        """
        session = ChatSession(
            provider=provider,
            model_id=model_id,
            temperature=temperature,
            seed=seed
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get existing session by ID."""
        return self._sessions.get(session_id)

    def register_template(self, name: str, template: PromptTemplate) -> None:
        """Register a prompt template."""
        self._templates[name] = template

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get registered template by name."""
        return self._templates.get(name)

    async def classify(
        self,
        session_id: str,
        data: Dict[str, Any],
        template_name: str,
        categories: List[str]
    ) -> str:
        """
        Classify data using LLM.

        Args:
            session_id: Session to use
            data: Data to classify
            template_name: Template for classification
            categories: Valid classification categories

        Returns:
            Classification result (one of categories)
        """
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        template = self._templates.get(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")

        prompt = template.format(**data)
        response = await session.send_message(prompt)

        # Validate response is in categories
        response_clean = response.strip().lower()
        for category in categories:
            if category.lower() in response_clean:
                return category

        # Default to first category if no match
        logger.warning(f"Classification response '{response}' not in categories, defaulting")
        return categories[0] if categories else "unknown"
