# -*- coding: utf-8 -*-
"""
LLM Provider Base Classes

Abstract base for LLM providers (OpenAI, Anthropic, Ollama, etc.):
- LLMProvider ABC: Unified chat() interface
- LLMProviderConfig: Provider configuration (model, API key, timeouts)
- LLMCapabilities: Provider capability metadata (function calling, JSON mode, etc.)

Contract enforcement:
- Budget caps (raise BudgetExceeded if would exceed)
- JSON schema validation (if json_schema provided)
- Function calling (if tools provided)
- Token usage tracking (cost attribution)

All providers MUST implement async chat() with consistent signature.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional
from pydantic import BaseModel, Field

from greenlang.agents.intelligence.schemas.messages import ChatMessage
from greenlang.agents.intelligence.schemas.tools import ToolDef
from greenlang.agents.intelligence.schemas.responses import ChatResponse, Usage
from greenlang.agents.intelligence.runtime.budget import Budget, BudgetExceeded
from greenlang.agents.intelligence.schemas.jsonschema import JSONSchema


class LLMCapabilities(BaseModel):
    """
    Provider capability metadata

    Describes what features a provider/model supports:
    - function_calling: Supports tool/function calling?
    - json_schema_mode: Supports JSON schema-constrained output?
    - max_output_tokens: Maximum tokens in single response
    - context_window_tokens: Total context window (input + output)

    Used for:
    - Provider selection (e.g., need function calling? filter providers)
    - Validation (error early if unsupported feature requested)
    - Optimization (e.g., batch requests to stay under context limit)

    Example:
        LLMCapabilities(
            function_calling=True,
            json_schema_mode=True,
            max_output_tokens=4096,
            context_window_tokens=128000
        )
    """

    function_calling: bool = Field(
        default=False,
        description="Supports tool/function calling?"
    )
    json_schema_mode: bool = Field(
        default=False,
        description="Supports JSON schema-constrained output?"
    )
    max_output_tokens: int = Field(
        default=4096,
        description="Maximum tokens in single response"
    )
    context_window_tokens: int = Field(
        default=8192,
        description="Total context window size (input + output)"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "function_calling": True,
                    "json_schema_mode": True,
                    "max_output_tokens": 4096,
                    "context_window_tokens": 128000
                }
            ]
        }


class LLMProviderConfig(BaseModel):
    """
    Provider configuration

    Settings for LLM provider initialization:
    - model: Model name (e.g., "gpt-4", "claude-3-opus")
    - api_key_env: Environment variable name for API key
    - timeout_s: Request timeout in seconds
    - max_retries: Max retry attempts on transient failures

    Security:
    - API keys MUST be loaded from environment variables
    - NEVER hardcode API keys in config
    - NEVER log API keys (use REDACTED in logs)

    Example:
        LLMProviderConfig(
            model="gpt-4-0613",
            api_key_env="OPENAI_API_KEY",
            timeout_s=60.0,
            max_retries=3
        )
    """

    model: str = Field(
        description="Model name (e.g., 'gpt-4', 'claude-3-opus')"
    )
    api_key_env: str = Field(
        description="Environment variable name for API key (e.g., 'OPENAI_API_KEY')"
    )
    timeout_s: float = Field(
        default=60.0,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Max retry attempts on transient failures (429, 503, etc.)"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "model": "gpt-4-0613",
                    "api_key_env": "OPENAI_API_KEY",
                    "timeout_s": 60.0,
                    "max_retries": 3
                }
            ]
        }


class LLMProvider(ABC):
    """
    Abstract base for LLM providers

    All providers (OpenAI, Anthropic, Ollama, etc.) must implement:
    - chat(): Async method for chat completions
    - capabilities: Property returning provider capabilities

    Contract enforcement:
    1. Budget: Must check budget BEFORE calling API (raise BudgetExceeded if would exceed)
    2. JSON schema: If json_schema provided, output MUST be valid JSON matching schema
    3. Tools: If tools provided, enable function calling (or error if not supported)
    4. Usage: Must return Usage with accurate token counts and cost

    Async-first API:
    - All I/O operations are async (network, file, etc.)
    - Use asyncio for concurrency (e.g., parallel provider calls)
    - Never block event loop with sync I/O

    Error handling:
    - Transient errors (429, 503): Retry with exponential backoff
    - Budget exceeded: Raise BudgetExceeded immediately
    - Invalid schema: Raise ValueError with clear message
    - API errors: Wrap in provider-specific exceptions

    Example implementation:
        class OpenAIProvider(LLMProvider):
            def __init__(self, config: LLMProviderConfig):
                self.config = config
                self.client = AsyncOpenAI(api_key=os.getenv(config.api_key_env))

            @property
            def capabilities(self) -> LLMCapabilities:
                return LLMCapabilities(
                    function_calling=True,
                    json_schema_mode=True,
                    max_output_tokens=4096,
                    context_window_tokens=128000
                )

            async def chat(
                self,
                messages: list[ChatMessage],
                *,
                tools: Optional[list[ToolDef]] = None,
                json_schema: Optional[JSONSchema] = None,
                budget: Budget,
                temperature: float = 0.0,
                top_p: float = 1.0,
                seed: int | None = None,
                tool_choice: str | None = None,
                metadata: Optional[Mapping[str, Any]] = None,
            ) -> ChatResponse:
                # 1. Validate capabilities
                if tools and not self.capabilities.function_calling:
                    raise ValueError("Provider does not support function calling")

                # 2. Estimate cost and check budget
                estimated_cost = self._estimate_cost(messages, tools)
                budget.check(add_usd=estimated_cost, add_tokens=0)

                # 3. Call provider API
                response = await self.client.chat.completions.create(...)

                # 4. Calculate actual usage
                usage = Usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                    cost_usd=self._calculate_cost(response.usage)
                )

                # 5. Add to budget
                budget.add(add_usd=usage.cost_usd, add_tokens=usage.total_tokens)

                # 6. Return normalized response
                return ChatResponse(...)
    """

    def __init__(self, config: LLMProviderConfig) -> None:
        """
        Initialize provider with configuration

        Args:
            config: Provider configuration (model, API key env var, timeouts, etc.)

        Raises:
            ValueError: If API key environment variable not set
            ValueError: If configuration invalid
        """
        self.config = config

    @property
    @abstractmethod
    def capabilities(self) -> LLMCapabilities:
        """
        Return provider capabilities

        Describes what features this provider/model supports.
        Used for validation and provider selection.

        Returns:
            Provider capabilities (function calling, JSON mode, token limits, etc.)

        Example:
            @property
            def capabilities(self) -> LLMCapabilities:
                return LLMCapabilities(
                    function_calling=True,
                    json_schema_mode=True,
                    max_output_tokens=4096,
                    context_window_tokens=128000
                )
        """
        pass

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: Optional[list[ToolDef]] = None,
        json_schema: Optional[JSONSchema] = None,
        budget: Budget,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        tool_choice: str | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ChatResponse:
        """
        Execute chat completion with budget enforcement

        Contract:
        1. MUST check budget BEFORE calling API (raise BudgetExceeded if would exceed)
        2. If json_schema provided, output MUST be valid JSON matching schema
        3. If tools provided, enable function calling (or error if not supported)
        4. MUST return Usage with accurate token counts and cost
        5. MUST add usage to budget after successful call

        Args:
            messages: Conversation history (system, user, assistant, tool messages)
            tools: Available tools for function calling (None = no tools)
            json_schema: JSON schema for response validation (None = no schema)
            budget: Budget tracker (enforces cost/token caps)
            temperature: Sampling temperature (0.0 = deterministic, 2.0 = creative)
            top_p: Nucleus sampling threshold (1.0 = disabled)
            seed: Random seed for reproducibility (None = random)
            tool_choice: Tool selection strategy ("auto"/"none"/"required"/tool_name)
            metadata: Provider-specific metadata (e.g., user ID for tracking)

        Returns:
            ChatResponse with text, tool calls, usage, and finish reason

        Raises:
            BudgetExceeded: If request would exceed budget cap
            ValueError: If unsupported feature requested (e.g., tools but no function calling)
            ValueError: If response doesn't match json_schema
            TimeoutError: If request exceeds timeout_s
            Exception: Provider-specific errors (wrapped with context)

        Example:
            # Simple text completion
            response = await provider.chat(
                messages=[
                    ChatMessage(role=Role.system, content="You are a climate analyst"),
                    ChatMessage(role=Role.user, content="What's the grid intensity in CA?")
                ],
                budget=Budget(max_usd=0.50)
            )
            print(response.text)

            # With function calling
            response = await provider.chat(
                messages=[...],
                tools=[
                    ToolDef(
                        name="get_grid_intensity",
                        description="Returns carbon intensity of electricity grid",
                        parameters={...}
                    )
                ],
                tool_choice="auto",
                budget=Budget(max_usd=0.50)
            )
            if response.tool_calls:
                print(f"LLM wants to call: {response.tool_calls[0]['name']}")

            # With JSON schema validation
            response = await provider.chat(
                messages=[...],
                json_schema={
                    "type": "object",
                    "properties": {
                        "intensity": {"type": "number"},
                        "source": {"type": "string"}
                    },
                    "required": ["intensity", "source"]
                },
                budget=Budget(max_usd=0.50)
            )
            data = json.loads(response.text)  # Guaranteed valid JSON
        """
        pass
