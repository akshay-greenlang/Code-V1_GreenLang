"""
GreenLang LLM Module
====================

Large Language Model integration for GreenLang.

This module provides client wrappers and configuration for integrating
LLMs into GreenLang applications for classification, entity resolution,
and narrative generation.

Example:
    >>> from greenlang.llm import LLMClient, LLMConfig
    >>> config = LLMConfig(model="gpt-4", temperature=0.7)
    >>> client = LLMClient(config)
    >>> response = client.generate("Classify this transaction")
"""

from greenlang.llm.client import (
    LLMClient,
    LLMResponse,
    LLMMessage,
    CompletionOptions,
)
from greenlang.llm.config import (
    LLMConfig,
    ModelProvider,
    ModelCapabilities,
)

__all__ = [
    # Client classes
    'LLMClient',
    'LLMResponse',
    'LLMMessage',
    'CompletionOptions',
    # Config classes
    'LLMConfig',
    'ModelProvider',
    'ModelCapabilities',
]

__version__ = '1.0.0'