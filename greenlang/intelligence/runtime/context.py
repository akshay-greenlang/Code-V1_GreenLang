# -*- coding: utf-8 -*-
"""
Context Window Management

Manages conversation history to prevent context overflow:
- Token counting per model
- Message truncation strategies
- Sliding window for long conversations
- Summarization for historical context

Prevents errors when conversation exceeds model context limits.

Architecture:
    Messages → count_tokens() → exceeds_limit? → truncate_messages() → Messages

Example:
    manager = ContextManager(model="gpt-4o", max_context=128000)

    # Long conversation
    messages = [...]  # 100K tokens

    # Truncate to fit
    truncated = manager.prepare_messages(
        messages=messages,
        max_completion_tokens=4096
    )
    # Returns messages that fit in (128K - 4K) = 124K token budget
"""

from __future__ import annotations
import logging
from typing import List, Optional
from greenlang.intelligence.schemas.messages import ChatMessage, Role

logger = logging.getLogger(__name__)


# Token limits for common models (context window - safety margin)
MODEL_CONTEXT_LIMITS = {
    # OpenAI models
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-3.5-turbo": 16_385,
    # Anthropic models
    "claude-3-opus-20240229": 200_000,
    "claude-3-sonnet-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
    "claude-2.1": 100_000,
    "claude-2": 100_000,
}


class ContextManager:
    """
    Manages conversation context to prevent overflow

    Strategies:
    1. Keep system message always
    2. Keep most recent messages (sliding window)
    3. Summarize old messages if needed
    4. Ensure total tokens < (context_limit - max_completion_tokens)

    Usage:
        manager = ContextManager(model="gpt-4o")

        truncated_messages = manager.prepare_messages(
            messages=long_conversation,
            max_completion_tokens=4096
        )

        # Use truncated messages in LLM call
        response = await provider.chat(messages=truncated_messages, ...)
    """

    def __init__(
        self,
        model: str,
        max_context_tokens: Optional[int] = None,
        safety_margin: int = 1000,
    ):
        """
        Initialize context manager

        Args:
            model: Model name for token limit lookup
            max_context_tokens: Override default context limit
            safety_margin: Safety margin tokens (default: 1000)
        """
        self.model = model
        self.safety_margin = safety_margin

        # Determine context limit
        if max_context_tokens:
            self.max_context_tokens = max_context_tokens
        else:
            # Look up from model table
            self.max_context_tokens = self._get_context_limit(model)

        logger.info(
            f"ContextManager initialized: model={model}, "
            f"max_context={self.max_context_tokens}, margin={safety_margin}"
        )

    def _get_context_limit(self, model: str) -> int:
        """
        Get context limit for model

        Args:
            model: Model name

        Returns:
            Context limit in tokens
        """
        # Try exact match
        if model in MODEL_CONTEXT_LIMITS:
            return MODEL_CONTEXT_LIMITS[model]

        # Try prefix match (e.g., "gpt-4o-2024-05-13" → "gpt-4o")
        for model_prefix, limit in MODEL_CONTEXT_LIMITS.items():
            if model.startswith(model_prefix):
                return limit

        # Default fallback (conservative)
        logger.warning(f"Unknown model {model}, using conservative limit: 8192")
        return 8_192

    def estimate_tokens(self, messages: List[ChatMessage]) -> int:
        """
        Estimate token count for messages

        Uses rough heuristic: 4 characters per token.
        For accurate counting, use tiktoken library (not included to avoid dependency).

        Args:
            messages: List of chat messages

        Returns:
            Estimated token count
        """
        total_chars = 0

        for msg in messages:
            # Count content
            if msg.content:
                total_chars += len(msg.content)

            # Add overhead for role, formatting, etc.
            total_chars += 20

            # Add overhead for tool calls if present
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                # Rough estimate: 50 chars per tool call
                total_chars += len(msg.tool_calls) * 50

        # Convert to tokens (rough: 4 chars per token)
        estimated_tokens = total_chars // 4

        return estimated_tokens

    def prepare_messages(
        self,
        messages: List[ChatMessage],
        max_completion_tokens: int = 4096,
        preserve_system: bool = True,
    ) -> List[ChatMessage]:
        """
        Prepare messages to fit within context window

        Workflow:
        1. Calculate available budget: context_limit - max_completion - safety_margin
        2. Count tokens in messages
        3. If exceeds budget, truncate using sliding window
        4. Always preserve system message if present

        Args:
            messages: Full conversation history
            max_completion_tokens: Tokens reserved for completion
            preserve_system: Always keep system message (default: True)

        Returns:
            Truncated message list that fits in context

        Example:
            >>> manager = ContextManager(model="gpt-4o")
            >>> messages = [ChatMessage(...) for _ in range(100)]  # Long conversation
            >>> truncated = manager.prepare_messages(messages, max_completion_tokens=4096)
            >>> len(truncated) < len(messages)
            True
        """
        # Calculate available budget
        available_tokens = (
            self.max_context_tokens - max_completion_tokens - self.safety_margin
        )

        # Estimate current token count
        current_tokens = self.estimate_tokens(messages)

        # If within budget, return as-is
        if current_tokens <= available_tokens:
            logger.debug(
                f"Messages fit in context: {current_tokens}/{available_tokens} tokens"
            )
            return messages

        # Need truncation
        logger.warning(
            f"Messages exceed context: {current_tokens}/{available_tokens} tokens. "
            f"Truncating..."
        )

        # Separate system message
        system_messages = [m for m in messages if m.role == Role.system]
        other_messages = [m for m in messages if m.role != Role.system]

        # Calculate system message tokens
        system_tokens = self.estimate_tokens(system_messages) if system_messages else 0

        # Available tokens for other messages
        available_for_others = available_tokens - system_tokens

        # Truncate from beginning (sliding window - keep most recent)
        truncated_others = self._truncate_sliding_window(
            other_messages, target_tokens=available_for_others
        )

        # Combine system + truncated messages
        if preserve_system and system_messages:
            result = system_messages + truncated_others
        else:
            result = truncated_others

        final_tokens = self.estimate_tokens(result)
        logger.info(
            f"Truncated messages: {len(messages)} → {len(result)} messages, "
            f"{current_tokens} → {final_tokens} tokens"
        )

        return result

    def _truncate_sliding_window(
        self,
        messages: List[ChatMessage],
        target_tokens: int,
    ) -> List[ChatMessage]:
        """
        Truncate messages using sliding window (keep most recent)

        Args:
            messages: Messages to truncate
            target_tokens: Target token count

        Returns:
            Truncated messages (most recent N that fit)
        """
        if not messages:
            return []

        # Start from end, work backwards
        result = []
        current_tokens = 0

        for msg in reversed(messages):
            msg_tokens = self.estimate_tokens([msg])

            if current_tokens + msg_tokens > target_tokens:
                # Would exceed budget, stop here
                break

            result.insert(0, msg)  # Insert at beginning to maintain order
            current_tokens += msg_tokens

        return result

    def check_overflow(
        self,
        messages: List[ChatMessage],
        max_completion_tokens: int = 4096,
    ) -> bool:
        """
        Check if messages would overflow context

        Args:
            messages: Messages to check
            max_completion_tokens: Tokens reserved for completion

        Returns:
            True if would overflow, False if fits

        Example:
            >>> manager = ContextManager(model="gpt-4o")
            >>> if manager.check_overflow(messages):
            ...     messages = manager.prepare_messages(messages)
        """
        available_tokens = (
            self.max_context_tokens - max_completion_tokens - self.safety_margin
        )

        current_tokens = self.estimate_tokens(messages)

        return current_tokens > available_tokens


# Global context manager cache
_context_managers = {}


def get_context_manager(model: str) -> ContextManager:
    """
    Get context manager for model (cached)

    Args:
        model: Model name

    Returns:
        ContextManager instance (cached)

    Example:
        manager = get_context_manager("gpt-4o")
        truncated = manager.prepare_messages(messages)
    """
    if model not in _context_managers:
        _context_managers[model] = ContextManager(model)
    return _context_managers[model]
