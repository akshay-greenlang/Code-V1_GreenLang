"""
Chat Message Schemas

Strongly-typed messages for LLM conversations:
- System: Instructions and context
- User: User queries
- Assistant: LLM responses
- Tool: Tool execution results
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Role(str, Enum):
    """Message role in conversation"""

    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class ChatMessage(BaseModel):
    """
    Single message in LLM conversation

    Examples:
        System message:
            ChatMessage(role=Role.system, content="You are a climate analyst")

        User message:
            ChatMessage(role=Role.user, content="Calculate emissions for 100 gallons diesel")

        Assistant message with tool call:
            ChatMessage(role=Role.assistant, content=None, tool_calls=[...])

        Tool result:
            ChatMessage(
                role=Role.tool,
                content='{"emissions": 1021}',
                name="calculate_fuel_emissions",
                tool_call_id="call_abc123"
            )
    """

    role: Role = Field(description="Message role (system/user/assistant/tool)")
    content: Optional[str] = Field(
        default=None, description="Message text content (None for tool calls)"
    )
    name: Optional[str] = Field(
        default=None, description="Tool name (for tool role messages)"
    )
    tool_call_id: Optional[str] = Field(
        default=None, description="Tool call ID (for tool role messages)"
    )

    class Config:
        use_enum_values = True
        json_schema_extra = {
            "examples": [
                {
                    "role": "system",
                    "content": "You are a climate calculation assistant. Use tools for all numeric calculations.",
                },
                {"role": "user", "content": "What's the grid intensity in California?"},
                {
                    "role": "tool",
                    "name": "get_grid_intensity",
                    "content": '{"intensity": 0.233, "source": "EIA-923"}',
                    "tool_call_id": "call_001",
                },
            ]
        }
