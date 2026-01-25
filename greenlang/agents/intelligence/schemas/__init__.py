# -*- coding: utf-8 -*-
"""
Intelligence Layer Schemas

Strongly-typed data structures for:
- Chat messages (system, user, assistant, tool)
- Tool definitions & calls (JSON Schema based)
- LLM responses with usage metrics
- JSON schema validation helpers
"""

from greenlang.agents.intelligence.schemas.messages import ChatMessage, Role
from greenlang.agents.intelligence.schemas.tools import ToolDef, ToolCall, ToolChoice
from greenlang.agents.intelligence.schemas.responses import (
    ChatResponse,
    Usage,
    FinishReason,
    ProviderInfo,
)
from greenlang.agents.intelligence.schemas.jsonschema import JSONSchema

__all__ = [
    "ChatMessage",
    "Role",
    "ToolDef",
    "ToolCall",
    "ToolChoice",
    "ChatResponse",
    "Usage",
    "FinishReason",
    "ProviderInfo",
    "JSONSchema",
]
