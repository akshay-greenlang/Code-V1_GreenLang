"""
LLM Response Schemas

Normalized response format across providers:
- ChatResponse: Complete LLM response with text, tool calls, usage
- Usage: Token counts and cost tracking
- FinishReason: Why generation stopped
- ProviderInfo: Provider metadata for debugging
"""

from __future__ import annotations
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field


class FinishReason(str, Enum):
    """Why LLM stopped generating"""

    stop = "stop"  # Natural completion
    length = "length"  # Hit max tokens
    tool_calls = "tool_calls"  # Requested tool execution
    content_filter = "content_filter"  # Blocked by safety filter
    error = "error"  # Error occurred


class Usage(BaseModel):
    """
    Token usage and cost tracking

    Enables:
    - Budget enforcement (stop when cost cap reached)
    - Cost attribution (which agent/workflow spent what)
    - Optimization tracking (cache hit rate, prompt efficiency)

    Example:
        Usage(
            prompt_tokens=1234,
            completion_tokens=567,
            total_tokens=1801,
            cost_usd=0.0234
        )
    """

    prompt_tokens: int = Field(default=0, description="Input tokens consumed")
    completion_tokens: int = Field(default=0, description="Output tokens generated")
    total_tokens: int = Field(default=0, description="Total tokens (input + output)")
    cost_usd: float = Field(default=0.0, description="Total cost in USD")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "prompt_tokens": 1200,
                    "completion_tokens": 450,
                    "total_tokens": 1650,
                    "cost_usd": 0.0234,
                }
            ]
        }


class ProviderInfo(BaseModel):
    """
    Provider metadata for debugging and auditing

    Tracks:
    - Which provider/model generated the response
    - Request ID for provider support tickets
    - Useful for debugging provider-specific issues
    """

    provider: str = Field(description="Provider name (openai, anthropic, etc.)")
    model: str = Field(description="Model name (gpt-4, claude-2, etc.)")
    request_id: Optional[str] = Field(
        default=None, description="Provider's request ID (for support/debugging)"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "provider": "openai",
                    "model": "gpt-4-0613",
                    "request_id": "req_abc123xyz",
                }
            ]
        }


class ChatResponse(BaseModel):
    """
    Normalized LLM response

    Unifies responses from different providers (OpenAI, Anthropic, etc.) into
    a consistent format for downstream processing.

    Contains:
    - text: Generated text content (if any)
    - tool_calls: Requested tool executions (if any)
    - usage: Token counts and costs
    - finish_reason: Why generation stopped
    - provider_info: Provider metadata
    - raw: Original provider response (for debugging; NEVER log in production)

    Example:
        ChatResponse(
            text="I need to check the grid intensity first.",
            tool_calls=[
                {
                    "id": "call_001",
                    "name": "get_grid_intensity",
                    "arguments": {"region": "CA", "year": 2024}
                }
            ],
            usage=Usage(prompt_tokens=1200, completion_tokens=45, cost_usd=0.0123),
            finish_reason=FinishReason.tool_calls,
            provider_info=ProviderInfo(provider="openai", model="gpt-4-0613"),
            raw=None  # Redacted for security
        )
    """

    text: Optional[str] = Field(
        default=None, description="Generated text content (None if only tool calls)"
    )
    tool_calls: List[dict] = Field(
        default_factory=list,
        description="Requested tool calls (normalized: {id, name, arguments})",
    )
    usage: Usage = Field(description="Token usage and cost")
    finish_reason: FinishReason = Field(description="Why generation stopped")
    provider_info: ProviderInfo = Field(description="Provider metadata")
    raw: Optional[Any] = Field(
        default=None,
        description="Raw provider response (for debugging ONLY; never log/persist)",
    )

    class Config:
        use_enum_values = True
        json_schema_extra = {
            "examples": [
                {
                    "text": "Based on the calculation, the emissions are 1,021 kg CO2e.",
                    "tool_calls": [],
                    "usage": {
                        "prompt_tokens": 1200,
                        "completion_tokens": 450,
                        "total_tokens": 1650,
                        "cost_usd": 0.0234,
                    },
                    "finish_reason": "stop",
                    "provider_info": {
                        "provider": "openai",
                        "model": "gpt-4-0613",
                        "request_id": "req_abc123",
                    },
                    "raw": None,
                }
            ]
        }
