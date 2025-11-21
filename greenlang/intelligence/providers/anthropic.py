# -*- coding: utf-8 -*-
"""
Anthropic (Claude) Provider Implementation

Complete adapter for Anthropic's Claude models:
- Claude-3 family (Opus, Sonnet, Haiku)
- Claude-2
- Tool use support (Anthropic's native tool calling)
- Budget enforcement
- Error classification
- Async implementation

Cost table (Q4 2025):
- claude-3-opus-20240229: $15/1M input, $75/1M output
- claude-3-sonnet-20240229: $3/1M input, $15/1M output
- claude-3-haiku-20240307: $0.25/1M input, $1.25/1M output
- claude-2.1: $8/1M input, $24/1M output
- claude-2.0: $8/1M input, $24/1M output

Anthropic-specific features:
- Uses "tools" parameter (native tool calling)
- Tool choice: "auto", "any", {"type": "tool", "name": "tool_name"}
- Response format: tool_use blocks in content array
- System messages must be passed as separate parameter
- Supports temperature, top_p for sampling control

Example:
    provider = AnthropicProvider(
        config=LLMProviderConfig(
            model="claude-3-sonnet-20240229",
            api_key_env="ANTHROPIC_API_KEY",
            timeout_s=60.0,
            max_retries=3
        )
    )

    response = await provider.chat(
        messages=[
            ChatMessage(role=Role.system, content="You are a climate analyst"),
            ChatMessage(role=Role.user, content="Calculate emissions for 100 gallons diesel")
        ],
        tools=[
            ToolDef(
                name="calculate_fuel_emissions",
                description="Calculate CO2e from fuel combustion",
                parameters={...}
            )
        ],
        budget=Budget(max_usd=0.50),
        temperature=0.0
    )
"""

from __future__ import annotations
import os
import json
import time
import asyncio
import logging
from typing import Any, Dict, List, Mapping, Optional

try:
    import anthropic
    from anthropic import AsyncAnthropic, APIError, APIStatusError
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)

from greenlang.intelligence.providers.base import (
    LLMProvider,
    LLMProviderConfig,
    LLMCapabilities,
)
from greenlang.intelligence.schemas.messages import ChatMessage, Role
from greenlang.intelligence.schemas.tools import ToolDef, ToolCall, ToolChoice
from greenlang.intelligence.schemas.responses import (
    ChatResponse,
    Usage,
    FinishReason,
    ProviderInfo,
)
from greenlang.intelligence.schemas.jsonschema import JSONSchema
from greenlang.intelligence.runtime.budget import Budget, BudgetExceeded
from greenlang.intelligence.providers.errors import (
    ProviderAuthError,
    ProviderRateLimit,
    ProviderTimeout,
    ProviderServerError,
    ProviderBadRequest,
    ProviderContentFilter,
    classify_provider_error,
)
from greenlang.intelligence.runtime.json_validator import (
    parse_and_validate,
    get_repair_prompt,
    JSONRetryTracker,
    GLJsonParseError,
)


# Cost table (USD per 1M tokens) - Q4 2025
ANTHROPIC_COSTS = {
    # Claude 3 Opus - Most capable, highest cost
    "claude-3-opus-20240229": {
        "input": 15.0 / 1_000_000,
        "output": 75.0 / 1_000_000,
    },
    "claude-3-opus": {
        "input": 15.0 / 1_000_000,
        "output": 75.0 / 1_000_000,
    },
    # Claude 3 Sonnet - Balanced performance and cost
    "claude-3-sonnet-20240229": {
        "input": 3.0 / 1_000_000,
        "output": 15.0 / 1_000_000,
    },
    "claude-3-sonnet": {
        "input": 3.0 / 1_000_000,
        "output": 15.0 / 1_000_000,
    },
    # Claude 3 Haiku - Fastest, lowest cost
    "claude-3-haiku-20240307": {
        "input": 0.25 / 1_000_000,
        "output": 1.25 / 1_000_000,
    },
    "claude-3-haiku": {
        "input": 0.25 / 1_000_000,
        "output": 1.25 / 1_000_000,
    },
    # Claude 2.1
    "claude-2.1": {
        "input": 8.0 / 1_000_000,
        "output": 24.0 / 1_000_000,
    },
    # Claude 2.0
    "claude-2.0": {
        "input": 8.0 / 1_000_000,
        "output": 24.0 / 1_000_000,
    },
    "claude-2": {
        "input": 8.0 / 1_000_000,
        "output": 24.0 / 1_000_000,
    },
}


# Model capabilities mapping
MODEL_CAPABILITIES = {
    # Claude 3 models have tool calling and large context windows
    "claude-3-opus-20240229": LLMCapabilities(
        function_calling=True,
        json_schema_mode=False,  # Anthropic doesn't have native JSON schema mode
        max_output_tokens=4096,
        context_window_tokens=200000,
    ),
    "claude-3-opus": LLMCapabilities(
        function_calling=True,
        json_schema_mode=False,
        max_output_tokens=4096,
        context_window_tokens=200000,
    ),
    "claude-3-sonnet-20240229": LLMCapabilities(
        function_calling=True,
        json_schema_mode=False,
        max_output_tokens=4096,
        context_window_tokens=200000,
    ),
    "claude-3-sonnet": LLMCapabilities(
        function_calling=True,
        json_schema_mode=False,
        max_output_tokens=4096,
        context_window_tokens=200000,
    ),
    "claude-3-haiku-20240307": LLMCapabilities(
        function_calling=True,
        json_schema_mode=False,
        max_output_tokens=4096,
        context_window_tokens=200000,
    ),
    "claude-3-haiku": LLMCapabilities(
        function_calling=True,
        json_schema_mode=False,
        max_output_tokens=4096,
        context_window_tokens=200000,
    ),
    # Claude 2 models do NOT support tool calling
    "claude-2.1": LLMCapabilities(
        function_calling=False,
        json_schema_mode=False,
        max_output_tokens=4096,
        context_window_tokens=100000,
    ),
    "claude-2.0": LLMCapabilities(
        function_calling=False,
        json_schema_mode=False,
        max_output_tokens=4096,
        context_window_tokens=100000,
    ),
    "claude-2": LLMCapabilities(
        function_calling=False,
        json_schema_mode=False,
        max_output_tokens=4096,
        context_window_tokens=100000,
    ),
}


class AnthropicProvider(LLMProvider):
    """
    Anthropic (Claude) LLM Provider

    Implements LLMProvider interface for Anthropic's Claude models.
    Supports Claude-3 family (Opus, Sonnet, Haiku) and Claude-2.

    Features:
    - Native tool calling for Claude-3 models
    - Budget enforcement (pre-call estimation + post-call tracking)
    - Error classification and retry logic
    - Async implementation with timeout handling
    - Cost calculation per model

    Anthropic-specific behavior:
    - System messages extracted and passed as 'system' parameter
    - Tool calls returned in content blocks (normalized to ToolCall format)
    - Stop reason mapped to FinishReason enum
    - Tool choice normalized: "auto", "any", or {"type": "tool", "name": "..."}

    Example:
        provider = AnthropicProvider(
            config=LLMProviderConfig(
                model="claude-3-sonnet-20240229",
                api_key_env="ANTHROPIC_API_KEY",
                timeout_s=60.0,
                max_retries=3
            )
        )

        response = await provider.chat(
            messages=[
                ChatMessage(role=Role.user, content="What is 2+2?")
            ],
            budget=Budget(max_usd=0.10),
            temperature=0.0
        )

        print(response.text)
        print(f"Cost: ${response.usage.cost_usd:.4f}")
    """

    def __init__(self, config: LLMProviderConfig) -> None:
        """
        Initialize Anthropic provider

        Args:
            config: Provider configuration (model, API key env var, timeouts)

        Raises:
            ImportError: If anthropic SDK not installed
            ValueError: If API key environment variable not set
            ValueError: If model not supported
        """
        super().__init__(config)

        # Check if SDK available
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )

        # Get API key from environment
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found in environment variable: {config.api_key_env}"
            )

        # Validate model
        if config.model not in ANTHROPIC_COSTS:
            raise ValueError(
                f"Unsupported model: {config.model}. "
                f"Supported: {list(ANTHROPIC_COSTS.keys())}"
            )

        # Initialize client
        self.client = AsyncAnthropic(
            api_key=api_key,
            timeout=config.timeout_s,
            max_retries=0,  # We handle retries ourselves
        )

        logger.info(
            f"Initialized Anthropic provider: model={config.model}, "
            f"timeout={config.timeout_s}s, max_retries={config.max_retries}"
        )

    @property
    def capabilities(self) -> LLMCapabilities:
        """
        Return capabilities for configured model

        Claude-3 models support tool calling, Claude-2 does not.
        All models have large context windows (100K-200K tokens).

        Returns:
            Model capabilities
        """
        return MODEL_CAPABILITIES.get(
            self.config.model,
            # Default for unknown models
            LLMCapabilities(
                function_calling=False,
                json_schema_mode=False,
                max_output_tokens=4096,
                context_window_tokens=100000,
            )
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
        """
        Execute chat completion with budget enforcement

        Args:
            messages: Conversation history (system, user, assistant, tool)
            tools: Available tools for function calling (Claude-3 only)
            json_schema: JSON schema for response (not natively supported)
            budget: Budget tracker (enforces cost/token caps)
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling threshold
            seed: Random seed (not supported by Anthropic)
            tool_choice: Tool selection strategy ("auto"/"any"/tool_name)
            metadata: Provider-specific metadata

        Returns:
            ChatResponse with text, tool calls, usage, and finish reason

        Raises:
            BudgetExceeded: If request would exceed budget
            ValueError: If unsupported feature requested
            ProviderError: Provider-specific errors

        Example:
            response = await provider.chat(
                messages=[
                    ChatMessage(role=Role.user, content="Calculate 2+2")
                ],
                tools=[
                    ToolDef(
                        name="calculate",
                        description="Perform arithmetic",
                        parameters={"type": "object", ...}
                    )
                ],
                budget=Budget(max_usd=0.10),
                temperature=0.0,
                tool_choice="auto"
            )

            if response.tool_calls:
                print(f"Calling: {response.tool_calls[0]['name']}")
        """
        # 1. Validate capabilities
        if tools and not self.capabilities.function_calling:
            raise ValueError(
                f"Model {self.config.model} does not support tool calling. "
                "Use Claude-3 models (Opus, Sonnet, Haiku) for tool support."
            )

        if json_schema:
            # Anthropic doesn't have native JSON schema mode
            # We can add a system message to request JSON format
            pass  # Handle below

        # 2. Estimate cost and check budget
        estimated_cost = self._estimate_cost(messages, tools)
        budget.check(add_usd=estimated_cost, add_tokens=0)

        logger.debug(
            f"Estimated cost: ${estimated_cost:.4f} "
            f"(remaining budget: ${budget.remaining_usd:.4f})"
        )

        # 3. Convert messages to Anthropic format
        system_message, anthropic_messages = self._convert_messages(messages, json_schema)

        # 4. Prepare API call parameters
        api_params: Dict[str, Any] = {
            "model": self.config.model,
            "messages": anthropic_messages,
            "max_tokens": self.capabilities.max_output_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        # Add system message if present
        if system_message:
            api_params["system"] = system_message

        # Add tools if provided
        if tools:
            api_params["tools"] = self._convert_tools(tools)

            # Convert tool_choice to Anthropic format
            if tool_choice:
                api_params["tool_choice"] = self._convert_tool_choice(tool_choice)

        # Add metadata if provided
        if metadata:
            api_params["metadata"] = metadata

        # 5. Call API with JSON retry logic (CTO SPEC: >3 retries = fail)
        request_id = metadata.get("request_id", f"req_{int(time.time() * 1000)}") if metadata else f"req_{int(time.time() * 1000)}"
        json_tracker = JSONRetryTracker(request_id=request_id, max_attempts=3) if json_schema else None

        for attempt in range(4):  # 0, 1, 2, 3 = 4 attempts total
            # Call Anthropic API
            response = await self._call_with_retry(api_params)

            # Calculate usage for THIS attempt
            usage = self._calculate_usage(response)

            # CTO SPEC: Increment cost meter on EVERY attempt
            budget.add(add_usd=usage.cost_usd, add_tokens=usage.total_tokens)

            # Extract text and tool calls
            text_parts = []
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input
                    })

            text = "".join(text_parts) if text_parts else None

            # If tool calls, no JSON validation needed
            if tool_calls:
                break

            # If json_schema provided, validate response
            if json_schema and text:
                try:
                    # Validate JSON
                    validated_json = parse_and_validate(text, json_schema)
                    json_tracker.record_success(attempt, validated_json)
                    logger.info(f"JSON validation succeeded on attempt {attempt + 1}")
                    break
                except Exception as e:
                    json_tracker.record_failure(attempt, e)

                    if json_tracker.should_fail():
                        # CTO SPEC: Raise GLJsonParseError after >3 attempts
                        logger.error(
                            f"JSON parsing failed after {json_tracker.attempts} attempts "
                            f"(request_id={request_id})"
                        )
                        raise json_tracker.build_error()

                    # Generate repair prompt for next attempt
                    repair_prompt = get_repair_prompt(json_schema, attempt + 1)

                    # Add repair instructions to system message
                    if system_message:
                        api_params["system"] = system_message + "\n\n" + repair_prompt
                    else:
                        api_params["system"] = repair_prompt

                    logger.warning(
                        f"JSON validation failed on attempt {attempt + 1}, retrying with repair prompt"
                    )
                    continue
            else:
                # No JSON schema or no text - done
                break

        # 6. Normalize and return response
        # Create normalized response manually since we already extracted text/tool_calls
        finish_reason = self._map_finish_reason(response.stop_reason)

        provider_info = ProviderInfo(
            provider="anthropic",
            model=response.model,
            request_id=response.id
        )

        logger.info(
            f"Anthropic call complete: {usage.total_tokens} tokens, "
            f"${usage.cost_usd:.4f}, finish_reason={finish_reason.value}, "
            f"model={response.model}, request_id={response.id}, "
            f"attempts={attempt + 1 if json_schema else 1}"
        )

        return ChatResponse(
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
            provider_info=provider_info,
            raw=None
        )

    def _estimate_cost(
        self,
        messages: list[ChatMessage],
        tools: Optional[list[ToolDef]] = None
    ) -> float:
        """
        Estimate cost before API call

        Conservative estimate based on:
        - Input token count (rough approximation: 4 chars per token)
        - Tool definitions (if provided)
        - Expected output tokens (assume max for safety)

        Args:
            messages: Conversation messages
            tools: Tool definitions

        Returns:
            Estimated cost in USD
        """
        costs = ANTHROPIC_COSTS.get(self.config.model)
        if not costs:
            # Unknown model - use Claude-2 pricing as conservative estimate
            costs = ANTHROPIC_COSTS["claude-2"]

        # Estimate input tokens (rough: 4 chars per token)
        input_chars = sum(len(msg.content or "") for msg in messages)
        if tools:
            # Add tool definition size (rough estimate)
            input_chars += sum(
                len(json.dumps(tool.parameters)) + len(tool.name) + len(tool.description)
                for tool in tools
            )

        estimated_input_tokens = input_chars // 4

        # Assume full output token budget for safety
        estimated_output_tokens = self.capabilities.max_output_tokens

        # Calculate cost
        input_cost = estimated_input_tokens * costs["input"]
        output_cost = estimated_output_tokens * costs["output"]

        return input_cost + output_cost

    def _calculate_usage(self, response: Any) -> Usage:
        """
        Calculate actual usage from API response

        Args:
            response: Anthropic API response

        Returns:
            Usage with token counts and cost
        """
        # Extract token usage from response
        usage_data = response.usage
        input_tokens = usage_data.input_tokens
        output_tokens = usage_data.output_tokens
        total_tokens = input_tokens + output_tokens

        # Calculate cost
        costs = ANTHROPIC_COSTS.get(self.config.model)
        if not costs:
            costs = ANTHROPIC_COSTS["claude-2"]

        input_cost = input_tokens * costs["input"]
        output_cost = output_tokens * costs["output"]
        total_cost = input_cost + output_cost

        return Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=total_cost,
        )

    def _convert_messages(
        self,
        messages: list[ChatMessage],
        json_schema: Optional[JSONSchema] = None
    ) -> tuple[Optional[str], list[dict]]:
        """
        Convert ChatMessage list to Anthropic format

        Anthropic expects:
        - System message as separate 'system' parameter (not in messages array)
        - Messages array with role and content
        - Tool results in specific format

        Args:
            messages: GreenLang ChatMessage list
            json_schema: Optional JSON schema (adds to system message)

        Returns:
            Tuple of (system_message, anthropic_messages)
        """
        system_parts = []
        anthropic_messages = []

        for msg in messages:
            if msg.role == Role.system:
                # Extract system message
                if msg.content:
                    system_parts.append(msg.content)

            elif msg.role == Role.user:
                # User message
                anthropic_messages.append({
                    "role": "user",
                    "content": msg.content or ""
                })

            elif msg.role == Role.assistant:
                # Assistant message (may have tool calls)
                if msg.content:
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": msg.content
                    })
                # Note: Tool calls are in content blocks, handled by response parsing

            elif msg.role == Role.tool:
                # Tool result - convert to Anthropic format
                anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content or ""
                        }
                    ]
                })

        # Add JSON schema instruction to system message if provided
        if json_schema:
            system_parts.append(
                f"\nYou MUST respond with valid JSON matching this schema:\n"
                f"{json.dumps(json_schema, indent=2)}\n"
                f"Do not include any text outside the JSON object."
            )

        # Combine system message parts
        system_message = "\n".join(system_parts) if system_parts else None

        return system_message, anthropic_messages

    def _convert_tools(self, tools: list[ToolDef]) -> list[dict]:
        """
        Convert ToolDef list to Anthropic format

        Anthropic tool format:
        {
            "name": "tool_name",
            "description": "Tool description",
            "input_schema": {...}  # JSON Schema
        }

        Args:
            tools: GreenLang ToolDef list

        Returns:
            Anthropic tool list
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters
            }
            for tool in tools
        ]

    def _convert_tool_choice(self, tool_choice: str) -> dict:
        """
        Convert tool_choice to Anthropic format

        GreenLang format:
        - "auto" -> let model decide
        - "none" -> disable tools
        - "required" -> must use a tool
        - "tool_name" -> must use specific tool

        Anthropic format:
        - {"type": "auto"} -> let model decide
        - {"type": "any"} -> must use a tool
        - {"type": "tool", "name": "tool_name"} -> specific tool

        Args:
            tool_choice: GreenLang tool choice strategy

        Returns:
            Anthropic tool_choice dict
        """
        if tool_choice == ToolChoice.AUTO or tool_choice == "auto":
            return {"type": "auto"}
        elif tool_choice == ToolChoice.REQUIRED or tool_choice == "required":
            return {"type": "any"}
        elif tool_choice == ToolChoice.NONE or tool_choice == "none":
            # Anthropic doesn't have "none" - just don't pass tools
            return {"type": "auto"}
        else:
            # Assume it's a specific tool name
            return {"type": "tool", "name": tool_choice}

    def _normalize_response(self, response: Any, usage: Usage) -> ChatResponse:
        """
        Normalize Anthropic response to ChatResponse format

        Extracts:
        - Text content
        - Tool calls (from content blocks)
        - Finish reason (stop_reason -> FinishReason)

        Args:
            response: Anthropic API response
            usage: Calculated usage

        Returns:
            Normalized ChatResponse
        """
        # Extract text and tool calls from content blocks
        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                # Convert to normalized ToolCall format
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input
                })

        # Combine text parts
        text = "".join(text_parts) if text_parts else None

        # Map stop_reason to FinishReason
        finish_reason = self._map_finish_reason(response.stop_reason)

        # Create provider info
        provider_info = ProviderInfo(
            provider="anthropic",
            model=response.model,
            request_id=response.id
        )

        return ChatResponse(
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
            provider_info=provider_info,
            raw=None  # Don't store raw response for security
        )

    def _map_finish_reason(self, stop_reason: str) -> FinishReason:
        """
        Map Anthropic stop_reason to FinishReason enum

        Anthropic stop reasons:
        - "end_turn" -> natural completion
        - "max_tokens" -> hit token limit
        - "tool_use" -> requested tool execution
        - "stop_sequence" -> hit stop sequence

        Args:
            stop_reason: Anthropic stop_reason

        Returns:
            FinishReason enum value
        """
        mapping = {
            "end_turn": FinishReason.stop,
            "max_tokens": FinishReason.length,
            "tool_use": FinishReason.tool_calls,
            "stop_sequence": FinishReason.stop,
        }
        return mapping.get(stop_reason, FinishReason.stop)

    async def _call_with_retry(self, api_params: Dict[str, Any]) -> Any:
        """
        Call Anthropic API with exponential backoff retry

        Retries on:
        - Rate limits (429)
        - Server errors (5xx)
        - Timeouts

        Does NOT retry on:
        - Auth errors (401, 403)
        - Bad requests (400, 422)
        - Content filters

        Args:
            api_params: API call parameters

        Returns:
            Anthropic API response

        Raises:
            ProviderError: Classified provider error
        """
        max_retries = self.config.max_retries
        base_delay = 1.0  # seconds

        for attempt in range(max_retries + 1):
            try:
                # Call API
                response = await self.client.messages.create(**api_params)
                return response

            except APIStatusError as e:
                # Classify error
                error = classify_provider_error(
                    error=e,
                    provider="anthropic",
                    status_code=e.status_code,
                    error_message=str(e)
                )

                # Check if retryable
                is_retryable = isinstance(error, (
                    ProviderRateLimit,
                    ProviderTimeout,
                    ProviderServerError
                ))

                # If last attempt or not retryable, raise
                if attempt >= max_retries or not is_retryable:
                    raise error

                # Calculate backoff delay
                if isinstance(error, ProviderRateLimit) and error.retry_after:
                    delay = error.retry_after
                else:
                    delay = base_delay * (2 ** attempt)

                # Log retry
                logger.warning(
                    f"Retrying after error (attempt {attempt + 1}/{max_retries}), "
                    f"waiting {delay}s: {error}"
                )

                # Wait before retry
                await asyncio.sleep(delay)

            except Exception as e:
                # Unexpected error - classify and raise
                error = classify_provider_error(
                    error=e,
                    provider="anthropic",
                    error_message=str(e)
                )
                raise error

        # Should never reach here
        raise ProviderServerError(
            message="Max retries exceeded",
            provider="anthropic"
        )


# Factory function for convenience
def create_anthropic_provider(
    model: str = "claude-3-sonnet-20240229",
    api_key_env: str = "ANTHROPIC_API_KEY",
    timeout_s: float = 60.0,
    max_retries: int = 3
) -> AnthropicProvider:
    """
    Factory function to create AnthropicProvider with common defaults

    Args:
        model: Claude model name (default: claude-3-sonnet)
        api_key_env: Environment variable for API key
        timeout_s: Request timeout in seconds
        max_retries: Max retry attempts

    Returns:
        Configured AnthropicProvider

    Example:
        provider = create_anthropic_provider(
            model="claude-3-haiku-20240307",
            timeout_s=30.0
        )

        response = await provider.chat(
            messages=[...],
            budget=Budget(max_usd=0.10)
        )
    """
    config = LLMProviderConfig(
        model=model,
        api_key_env=api_key_env,
        timeout_s=timeout_s,
        max_retries=max_retries
    )
    return AnthropicProvider(config)
