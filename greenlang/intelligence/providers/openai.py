"""
OpenAI Provider Implementation

Complete OpenAI adapter for GreenLang Intelligence system:
- Supports GPT-4, GPT-4 Turbo, GPT-4o, GPT-3.5 Turbo
- Tool/function calling with automatic format conversion
- JSON schema mode (response_format parameter)
- Budget enforcement (estimate + check before call, add after)
- Comprehensive error classification and retry logic
- Async implementation with httpx for resilient HTTP calls

Cost Table (Q4 2025):
- gpt-4-turbo: $10/1M input, $30/1M output
- gpt-4: $30/1M input, $60/1M output
- gpt-4o: $5/1M input, $15/1M output
- gpt-3.5-turbo: $0.50/1M input, $1.50/1M output

Security:
- API keys loaded from environment (never hardcoded)
- API keys never logged (use REDACTED in logs)
- Raw responses redacted in production
"""

from __future__ import annotations
import os
import json
import asyncio
import logging
from typing import Any, Dict, List, Mapping, Optional

try:
    from openai import AsyncOpenAI
    from openai import APIError, APIConnectionError, RateLimitError, Timeout, AuthenticationError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from greenlang.intelligence.providers.base import (
    LLMProvider,
    LLMProviderConfig,
    LLMCapabilities,
)
from greenlang.intelligence.providers.errors import (
    ProviderError,
    ProviderAuthError,
    ProviderRateLimit,
    ProviderTimeout,
    ProviderServerError,
    ProviderBadRequest,
    ProviderContentFilter,
    classify_provider_error,
)
from greenlang.intelligence.schemas.messages import ChatMessage, Role
from greenlang.intelligence.schemas.tools import ToolDef, ToolCall
from greenlang.intelligence.schemas.responses import ChatResponse, Usage, FinishReason, ProviderInfo
from greenlang.intelligence.schemas.jsonschema import JSONSchema
from greenlang.intelligence.runtime.budget import Budget, BudgetExceeded

logger = logging.getLogger(__name__)


# Cost table for OpenAI models (USD per 1M tokens, Q4 2025)
MODEL_COSTS = {
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4-turbo-preview": {"input": 10.0, "output": 30.0},
    "gpt-4-turbo-2024-04-09": {"input": 10.0, "output": 30.0},
    "gpt-4-1106-preview": {"input": 10.0, "output": 30.0},
    "gpt-4-0125-preview": {"input": 10.0, "output": 30.0},
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-0613": {"input": 30.0, "output": 60.0},
    "gpt-4-0314": {"input": 30.0, "output": 60.0},
    "gpt-4-32k": {"input": 60.0, "output": 120.0},
    "gpt-4-32k-0613": {"input": 60.0, "output": 120.0},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-2024-05-13": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-1106": {"input": 1.00, "output": 2.00},
    "gpt-3.5-turbo-16k": {"input": 3.00, "output": 4.00},
}

# Model capabilities mapping
MODEL_CAPABILITIES = {
    "gpt-4-turbo": LLMCapabilities(
        function_calling=True,
        json_schema_mode=True,
        max_output_tokens=4096,
        context_window_tokens=128000,
    ),
    "gpt-4": LLMCapabilities(
        function_calling=True,
        json_schema_mode=True,
        max_output_tokens=8192,
        context_window_tokens=8192,
    ),
    "gpt-4-32k": LLMCapabilities(
        function_calling=True,
        json_schema_mode=True,
        max_output_tokens=32768,
        context_window_tokens=32768,
    ),
    "gpt-4o": LLMCapabilities(
        function_calling=True,
        json_schema_mode=True,
        max_output_tokens=16384,
        context_window_tokens=128000,
    ),
    "gpt-4o-mini": LLMCapabilities(
        function_calling=True,
        json_schema_mode=True,
        max_output_tokens=16384,
        context_window_tokens=128000,
    ),
    "gpt-3.5-turbo": LLMCapabilities(
        function_calling=True,
        json_schema_mode=True,
        max_output_tokens=4096,
        context_window_tokens=16385,
    ),
}


def _get_model_base_name(model: str) -> str:
    """
    Extract base model name from versioned model string

    Args:
        model: Full model name (e.g., "gpt-4-0613", "gpt-4o-2024-05-13")

    Returns:
        Base model name for capability/cost lookup

    Example:
        >>> _get_model_base_name("gpt-4-0613")
        "gpt-4"
        >>> _get_model_base_name("gpt-4o-mini-2024-07-18")
        "gpt-4o-mini"
    """
    # Try exact match first
    if model in MODEL_COSTS:
        return model

    # Try common prefixes
    for base in ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4-32k", "gpt-4", "gpt-3.5-turbo"]:
        if model.startswith(base):
            return base

    # Default fallback
    return "gpt-4-turbo"


class OpenAIProvider(LLMProvider):
    """
    OpenAI provider implementation

    Implements LLMProvider ABC with full support for:
    - Chat completions (GPT-4, GPT-4 Turbo, GPT-4o, GPT-3.5 Turbo)
    - Tool/function calling (automatic format conversion)
    - JSON schema mode (response_format parameter)
    - Budget enforcement (estimate cost, check, then add after call)
    - Error classification (map OpenAI errors to taxonomy)
    - Retry logic with exponential backoff

    Usage:
        # Initialize provider
        config = LLMProviderConfig(
            model="gpt-4-turbo",
            api_key_env="OPENAI_API_KEY",
            timeout_s=60.0,
            max_retries=3
        )
        provider = OpenAIProvider(config)

        # Simple chat
        response = await provider.chat(
            messages=[
                ChatMessage(role=Role.system, content="You are a helpful assistant"),
                ChatMessage(role=Role.user, content="Hello!")
            ],
            budget=Budget(max_usd=0.50)
        )
        print(response.text)

        # With function calling
        tools = [
            ToolDef(
                name="get_weather",
                description="Get weather for location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            )
        ]
        response = await provider.chat(
            messages=[ChatMessage(role=Role.user, content="What's the weather in SF?")],
            tools=tools,
            budget=Budget(max_usd=0.50)
        )
        if response.tool_calls:
            print(f"Tool call: {response.tool_calls[0]['name']}")

    Security:
        - API key loaded from environment variable (OPENAI_API_KEY or custom)
        - API key never logged (uses REDACTED in logs)
        - Raw responses not persisted in production
    """

    def __init__(self, config: LLMProviderConfig) -> None:
        """
        Initialize OpenAI provider

        Args:
            config: Provider configuration (model, API key env, timeouts)

        Raises:
            ValueError: If API key environment variable not set
            ImportError: If openai package not installed
        """
        super().__init__(config)

        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        # Load API key from environment
        api_key = os.getenv(config.api_key_env) or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                f"API key not found in environment. Set {config.api_key_env} or OPENAI_API_KEY"
            )

        # Initialize async client
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=config.timeout_s,
            max_retries=0,  # We handle retries ourselves
        )

        # Get model base name for cost/capability lookups
        self.model_base = _get_model_base_name(config.model)

        logger.info(
            f"Initialized OpenAI provider: model={config.model}, "
            f"timeout={config.timeout_s}s, max_retries={config.max_retries}"
        )

    @property
    def capabilities(self) -> LLMCapabilities:
        """
        Return provider capabilities

        Returns:
            Capabilities for configured model (function calling, JSON mode, token limits)
        """
        return MODEL_CAPABILITIES.get(
            self.model_base,
            LLMCapabilities(
                function_calling=True,
                json_schema_mode=True,
                max_output_tokens=4096,
                context_window_tokens=128000,
            ),
        )

    def _estimate_tokens(self, messages: List[ChatMessage], tools: Optional[List[ToolDef]] = None) -> int:
        """
        Estimate input token count

        Uses rough heuristic: 4 chars per token (OpenAI uses ~3-4 for English).
        For accurate counting, use tiktoken library (not included to avoid dependency).

        Args:
            messages: Chat messages
            tools: Tool definitions

        Returns:
            Estimated token count
        """
        # Count message content
        total_chars = 0
        for msg in messages:
            if msg.content:
                total_chars += len(msg.content)
            # Add overhead for role, formatting
            total_chars += 20

        # Count tool definitions (can be substantial)
        if tools:
            for tool in tools:
                total_chars += len(tool.name) + len(tool.description)
                total_chars += len(json.dumps(tool.parameters))

        # Convert chars to tokens (rough estimate)
        return total_chars // 4

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate cost in USD

        Args:
            prompt_tokens: Input token count
            completion_tokens: Output token count

        Returns:
            Total cost in USD
        """
        costs = MODEL_COSTS.get(
            self.model_base,
            {"input": 10.0, "output": 30.0},  # Default to gpt-4-turbo pricing
        )

        input_cost = (prompt_tokens / 1_000_000) * costs["input"]
        output_cost = (completion_tokens / 1_000_000) * costs["output"]

        return input_cost + output_cost

    def _convert_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """
        Convert ChatMessage to OpenAI format

        Args:
            messages: Normalized messages

        Returns:
            OpenAI-formatted messages
        """
        openai_messages = []
        for msg in messages:
            openai_msg: Dict[str, Any] = {"role": msg.role.value}

            if msg.content:
                openai_msg["content"] = msg.content

            # Tool messages need special handling
            if msg.role == Role.tool:
                if msg.name:
                    openai_msg["name"] = msg.name
                if msg.tool_call_id:
                    openai_msg["tool_call_id"] = msg.tool_call_id

            openai_messages.append(openai_msg)

        return openai_messages

    def _convert_tools(self, tools: List[ToolDef]) -> List[Dict[str, Any]]:
        """
        Convert ToolDef to OpenAI function format

        Args:
            tools: Tool definitions

        Returns:
            OpenAI-formatted tools
        """
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            })
        return openai_tools

    def _normalize_tool_calls(self, openai_tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """
        Normalize OpenAI tool calls to standard format

        Args:
            openai_tool_calls: OpenAI tool call objects

        Returns:
            Normalized tool calls: [{"id": str, "name": str, "arguments": dict}]
        """
        normalized = []
        for tool_call in openai_tool_calls:
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                # If arguments aren't valid JSON, wrap as string
                arguments = {"raw": tool_call.function.arguments}

            normalized.append({
                "id": tool_call.id,
                "name": tool_call.function.name,
                "arguments": arguments,
            })

        return normalized

    def _normalize_finish_reason(self, openai_finish_reason: str) -> FinishReason:
        """
        Map OpenAI finish_reason to normalized enum

        Args:
            openai_finish_reason: OpenAI's finish reason

        Returns:
            Normalized finish reason
        """
        mapping = {
            "stop": FinishReason.stop,
            "length": FinishReason.length,
            "tool_calls": FinishReason.tool_calls,
            "function_call": FinishReason.tool_calls,
            "content_filter": FinishReason.content_filter,
        }
        return mapping.get(openai_finish_reason, FinishReason.stop)

    async def _call_with_retry(
        self,
        openai_messages: List[Dict[str, Any]],
        openai_tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        tool_choice: Optional[str] = None,
    ) -> Any:
        """
        Call OpenAI API with exponential backoff retry

        Retries on:
        - Rate limits (429)
        - Server errors (5xx)
        - Timeouts

        Does NOT retry on:
        - Auth errors (401/403)
        - Bad requests (400/422)
        - Content filters

        Args:
            openai_messages: OpenAI-formatted messages
            openai_tools: OpenAI-formatted tools
            response_format: Response format spec (for JSON mode)
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            seed: Random seed
            tool_choice: Tool selection strategy

        Returns:
            OpenAI response object

        Raises:
            ProviderError: Classified error after all retries exhausted
        """
        max_retries = self.config.max_retries
        retry_count = 0
        base_delay = 1.0

        while True:
            try:
                # Build request parameters
                params: Dict[str, Any] = {
                    "model": self.config.model,
                    "messages": openai_messages,
                    "temperature": temperature,
                    "top_p": top_p,
                }

                if seed is not None:
                    params["seed"] = seed

                if openai_tools:
                    params["tools"] = openai_tools
                    if tool_choice:
                        # Map tool_choice to OpenAI format
                        if tool_choice == "auto":
                            params["tool_choice"] = "auto"
                        elif tool_choice == "none":
                            params["tool_choice"] = "none"
                        elif tool_choice == "required":
                            params["tool_choice"] = "required"
                        else:
                            # Specific tool name
                            params["tool_choice"] = {
                                "type": "function",
                                "function": {"name": tool_choice}
                            }

                if response_format:
                    params["response_format"] = response_format

                # Make API call
                response = await self.client.chat.completions.create(**params)
                return response

            except AuthenticationError as e:
                # Never retry auth errors
                raise ProviderAuthError(
                    message=str(e),
                    provider="openai",
                    status_code=401,
                    original_error=e,
                )

            except RateLimitError as e:
                # Retry rate limits with backoff
                if retry_count >= max_retries:
                    raise ProviderRateLimit(
                        message=f"Rate limit exceeded after {retry_count} retries: {e}",
                        provider="openai",
                        retry_after=None,
                        original_error=e,
                    )

                # Exponential backoff
                delay = base_delay * (2 ** retry_count)
                logger.warning(
                    f"Rate limit hit (retry {retry_count + 1}/{max_retries}), "
                    f"waiting {delay}s: {e}"
                )
                await asyncio.sleep(delay)
                retry_count += 1

            except Timeout as e:
                # Retry timeouts
                if retry_count >= max_retries:
                    raise ProviderTimeout(
                        message=f"Timeout after {retry_count} retries: {e}",
                        provider="openai",
                        timeout_seconds=self.config.timeout_s,
                        original_error=e,
                    )

                delay = base_delay * (2 ** retry_count)
                logger.warning(
                    f"Timeout (retry {retry_count + 1}/{max_retries}), "
                    f"waiting {delay}s: {e}"
                )
                await asyncio.sleep(delay)
                retry_count += 1

            except APIConnectionError as e:
                # Retry connection errors
                if retry_count >= max_retries:
                    raise ProviderServerError(
                        message=f"Connection error after {retry_count} retries: {e}",
                        provider="openai",
                        original_error=e,
                    )

                delay = base_delay * (2 ** retry_count)
                logger.warning(
                    f"Connection error (retry {retry_count + 1}/{max_retries}), "
                    f"waiting {delay}s: {e}"
                )
                await asyncio.sleep(delay)
                retry_count += 1

            except APIError as e:
                # Classify and potentially retry
                status_code = getattr(e, "status_code", None)

                # Server errors (5xx): retry
                if status_code and status_code >= 500:
                    if retry_count >= max_retries:
                        raise ProviderServerError(
                            message=f"Server error after {retry_count} retries: {e}",
                            provider="openai",
                            status_code=status_code,
                            original_error=e,
                        )

                    delay = base_delay * (2 ** retry_count)
                    logger.warning(
                        f"Server error {status_code} (retry {retry_count + 1}/{max_retries}), "
                        f"waiting {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                    retry_count += 1
                    continue

                # Client errors (4xx): don't retry, classify and raise
                error_msg = str(e)
                if "content" in error_msg.lower() or "filter" in error_msg.lower():
                    raise ProviderContentFilter(
                        message=error_msg,
                        provider="openai",
                        status_code=status_code,
                        original_error=e,
                    )

                raise ProviderBadRequest(
                    message=error_msg,
                    provider="openai",
                    status_code=status_code,
                    original_error=e,
                )

            except Exception as e:
                # Unexpected error: classify and raise
                raise classify_provider_error(
                    error=e,
                    provider="openai",
                    status_code=None,
                    error_message=str(e),
                )

    async def chat(
        self,
        messages: List[ChatMessage],
        *,
        tools: Optional[List[ToolDef]] = None,
        json_schema: Optional[JSONSchema] = None,
        budget: Budget,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        tool_choice: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ChatResponse:
        """
        Execute chat completion with budget enforcement

        Workflow:
        1. Validate capabilities (error if unsupported feature requested)
        2. Estimate cost and check budget (raise BudgetExceeded if would exceed)
        3. Call OpenAI API (with retry logic)
        4. Calculate actual usage
        5. Add to budget
        6. Return normalized response

        Args:
            messages: Conversation history
            tools: Available tools for function calling
            json_schema: JSON schema for response validation
            budget: Budget tracker (enforces cost/token caps)
            temperature: Sampling temperature (0.0 = deterministic, 2.0 = creative)
            top_p: Nucleus sampling threshold (1.0 = disabled)
            seed: Random seed for reproducibility
            tool_choice: Tool selection strategy ("auto"/"none"/"required"/tool_name)
            metadata: Provider-specific metadata (unused by OpenAI)

        Returns:
            ChatResponse with text, tool calls, usage, and finish reason

        Raises:
            BudgetExceeded: If request would exceed budget cap
            ValueError: If unsupported feature requested
            ProviderError: On API errors (auth, rate limit, timeout, etc.)
        """
        # 1. Validate capabilities
        if tools and not self.capabilities.function_calling:
            raise ValueError(
                f"Model {self.config.model} does not support function calling"
            )

        if json_schema and not self.capabilities.json_schema_mode:
            raise ValueError(
                f"Model {self.config.model} does not support JSON schema mode"
            )

        # 2. Estimate cost and check budget
        estimated_tokens = self._estimate_tokens(messages, tools)
        estimated_cost = self._calculate_cost(
            prompt_tokens=estimated_tokens,
            completion_tokens=min(1000, self.capabilities.max_output_tokens // 4),
        )

        # Check budget BEFORE calling API
        budget.check(add_usd=estimated_cost, add_tokens=estimated_tokens)

        logger.debug(
            f"Estimated: {estimated_tokens} tokens, ${estimated_cost:.4f} "
            f"(remaining budget: ${budget.remaining_usd:.4f})"
        )

        # 3. Convert to OpenAI format
        openai_messages = self._convert_messages(messages)
        openai_tools = self._convert_tools(tools) if tools else None

        # Handle JSON schema mode
        response_format = None
        if json_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": json_schema,
                }
            }

        # 4. Call OpenAI API with retry
        openai_response = await self._call_with_retry(
            openai_messages=openai_messages,
            openai_tools=openai_tools,
            response_format=response_format,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            tool_choice=tool_choice,
        )

        # 5. Extract response data
        choice = openai_response.choices[0]
        message = choice.message

        # Get text content
        text = message.content if message.content else None

        # Get tool calls
        tool_calls = []
        if message.tool_calls:
            tool_calls = self._normalize_tool_calls(message.tool_calls)

        # Calculate actual usage
        usage = Usage(
            prompt_tokens=openai_response.usage.prompt_tokens,
            completion_tokens=openai_response.usage.completion_tokens,
            total_tokens=openai_response.usage.total_tokens,
            cost_usd=self._calculate_cost(
                prompt_tokens=openai_response.usage.prompt_tokens,
                completion_tokens=openai_response.usage.completion_tokens,
            ),
        )

        # Normalize finish reason
        finish_reason = self._normalize_finish_reason(choice.finish_reason)

        # Provider info
        provider_info = ProviderInfo(
            provider="openai",
            model=self.config.model,
            request_id=getattr(openai_response, "id", None),
        )

        # 6. Add to budget
        budget.add(add_usd=usage.cost_usd, add_tokens=usage.total_tokens)

        logger.info(
            f"OpenAI call complete: {usage.total_tokens} tokens, "
            f"${usage.cost_usd:.4f}, finish_reason={finish_reason.value}"
        )

        # 7. Return normalized response
        return ChatResponse(
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
            provider_info=provider_info,
            raw=None,  # Don't include raw response (security)
        )


if __name__ == "__main__":
    """
    Usage examples and tests

    Set OPENAI_API_KEY environment variable before running:
        export OPENAI_API_KEY=sk-...
        python -m greenlang.intelligence.providers.openai
    """
    import asyncio

    async def example_basic_chat():
        """Example: Basic chat completion"""
        print("\n=== Example: Basic Chat ===")

        config = LLMProviderConfig(
            model="gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
            timeout_s=30.0,
            max_retries=3,
        )

        provider = OpenAIProvider(config)

        messages = [
            ChatMessage(role=Role.system, content="You are a helpful assistant."),
            ChatMessage(role=Role.user, content="What is 2+2?"),
        ]

        budget = Budget(max_usd=0.10)

        response = await provider.chat(messages=messages, budget=budget)

        print(f"Response: {response.text}")
        print(f"Cost: ${response.usage.cost_usd:.6f}")
        print(f"Tokens: {response.usage.total_tokens}")
        print(f"Budget remaining: ${budget.remaining_usd:.6f}")

    async def example_function_calling():
        """Example: Function calling"""
        print("\n=== Example: Function Calling ===")

        config = LLMProviderConfig(
            model="gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
        )

        provider = OpenAIProvider(config)

        tools = [
            ToolDef(
                name="calculate_emissions",
                description="Calculate CO2 emissions from fuel combustion",
                parameters={
                    "type": "object",
                    "properties": {
                        "fuel_type": {
                            "type": "string",
                            "enum": ["diesel", "gasoline", "natural_gas"],
                            "description": "Type of fuel",
                        },
                        "amount": {
                            "type": "number",
                            "minimum": 0,
                            "description": "Fuel amount in gallons",
                        },
                    },
                    "required": ["fuel_type", "amount"],
                },
            )
        ]

        messages = [
            ChatMessage(
                role=Role.system,
                content="You are a climate assistant. Use tools for calculations.",
            ),
            ChatMessage(
                role=Role.user,
                content="Calculate emissions for 100 gallons of diesel",
            ),
        ]

        budget = Budget(max_usd=0.10)

        response = await provider.chat(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            budget=budget,
        )

        if response.tool_calls:
            print(f"Tool call requested:")
            for tc in response.tool_calls:
                print(f"  - {tc['name']}: {tc['arguments']}")
        else:
            print(f"Text response: {response.text}")

        print(f"Cost: ${response.usage.cost_usd:.6f}")

    async def example_json_mode():
        """Example: JSON schema mode"""
        print("\n=== Example: JSON Schema Mode ===")

        config = LLMProviderConfig(
            model="gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
        )

        provider = OpenAIProvider(config)

        json_schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "number"},
                "explanation": {"type": "string"},
            },
            "required": ["answer", "explanation"],
            "additionalProperties": False,
        }

        messages = [
            ChatMessage(
                role=Role.user,
                content="What is 15 * 24? Return JSON with answer and explanation.",
            ),
        ]

        budget = Budget(max_usd=0.10)

        response = await provider.chat(
            messages=messages,
            json_schema=json_schema,
            budget=budget,
        )

        print(f"JSON response: {response.text}")

        # Validate JSON
        try:
            data = json.loads(response.text)
            print(f"Parsed: answer={data['answer']}, explanation={data['explanation']}")
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON: {e}")

    async def example_budget_enforcement():
        """Example: Budget cap enforcement"""
        print("\n=== Example: Budget Enforcement ===")

        config = LLMProviderConfig(
            model="gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
        )

        provider = OpenAIProvider(config)

        # Very low budget (will likely exceed)
        budget = Budget(max_usd=0.0001)

        messages = [
            ChatMessage(role=Role.user, content="Write a long essay about climate change."),
        ]

        try:
            response = await provider.chat(messages=messages, budget=budget)
            print(f"Response: {response.text[:100]}...")
        except BudgetExceeded as e:
            print(f"Budget exceeded: {e}")
            print(f"Spent: ${e.spent_usd:.6f} / ${e.max_usd:.6f}")

    async def main():
        """Run all examples"""
        print("OpenAI Provider Examples")
        print("=" * 50)

        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY environment variable not set")
            print("Set it with: export OPENAI_API_KEY=sk-...")
            return

        try:
            await example_basic_chat()
            await example_function_calling()
            await example_json_mode()
            await example_budget_enforcement()
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()

    # Run examples
    asyncio.run(main())
