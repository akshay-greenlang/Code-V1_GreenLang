"""
Fake LLM Provider for Unit Testing

This module provides FakeProvider and helper utilities for testing intelligence
components without making real network calls or requiring API keys.

Features:
- Configurable responses (set what it returns)
- Configurable usage (tokens, cost)
- Configurable errors (simulate failures)
- Call history tracking (record all calls for assertions)

Usage:
    # Basic usage - return pre-configured responses
    fake = FakeProvider([
        make_text_response("Grid intensity is 450 gCO2/kWh", tokens=100, cost_usd=0.01)
    ])
    response = await fake.chat(messages, budget=Budget(max_usd=0.50))
    assert response.text == "Grid intensity is 450 gCO2/kWh"

    # Simulate tool calls
    fake = FakeProvider([
        make_tool_call_response([
            {"name": "get_grid_intensity", "arguments": {"region": "CA"}}
        ])
    ])
    response = await fake.chat(messages, tools=[...], budget=budget)
    assert len(response.tool_calls) == 1

    # Simulate errors
    fake = FakeProvider(error_to_raise=ProviderRateLimit("Rate limit exceeded", "fake"))
    try:
        await fake.chat(messages, budget=budget)
    except ProviderRateLimit:
        # Expected

    # Assert call history
    fake = FakeProvider([make_text_response("result")])
    await fake.chat(messages, budget=budget, temperature=0.5)
    assert len(fake.call_history) == 1
    assert fake.call_history[0]["temperature"] == 0.5
"""

from __future__ import annotations
import json
from typing import Any, List, Mapping, Optional
from copy import deepcopy

from greenlang.intelligence.providers.base import (
    LLMProvider,
    LLMProviderConfig,
    LLMCapabilities,
)
from greenlang.intelligence.providers.errors import (
    ProviderError,
    ProviderRateLimit,
    ProviderTimeout,
    ProviderServerError,
    ProviderAuthError,
    ProviderBadRequest,
)
from greenlang.intelligence.schemas.messages import ChatMessage, Role
from greenlang.intelligence.schemas.tools import ToolDef
from greenlang.intelligence.schemas.responses import (
    ChatResponse,
    Usage,
    FinishReason,
    ProviderInfo,
)
from greenlang.intelligence.runtime.budget import Budget, BudgetExceeded
from greenlang.intelligence.schemas.jsonschema import JSONSchema


class FakeProvider(LLMProvider):
    """
    Fake LLM provider for unit testing

    Enables testing without:
    - Network calls
    - API keys
    - Rate limits
    - Cost (unless simulated)

    Features:
    1. Configurable responses - pre-set what to return
    2. Configurable usage - control tokens and cost
    3. Configurable errors - simulate failures
    4. Call history - record all calls for assertions
    5. Conditional responses - return different responses based on call count

    Example: Basic text response
        >>> fake = FakeProvider([
        ...     make_text_response("The answer is 42", tokens=100, cost_usd=0.01)
        ... ])
        >>> response = await fake.chat(messages, budget=Budget(max_usd=0.50))
        >>> assert response.text == "The answer is 42"
        >>> assert response.usage.total_tokens == 100
        >>> assert response.usage.cost_usd == 0.01

    Example: Tool call response
        >>> fake = FakeProvider([
        ...     make_tool_call_response([
        ...         {"name": "calculate", "arguments": {"x": 10, "y": 20}}
        ...     ])
        ... ])
        >>> response = await fake.chat(messages, tools=[...], budget=budget)
        >>> assert len(response.tool_calls) == 1
        >>> assert response.tool_calls[0]["name"] == "calculate"

    Example: Multiple responses (cycle through)
        >>> fake = FakeProvider([
        ...     make_text_response("First response"),
        ...     make_text_response("Second response"),
        ... ])
        >>> r1 = await fake.chat(messages, budget=budget)
        >>> assert r1.text == "First response"
        >>> r2 = await fake.chat(messages, budget=budget)
        >>> assert r2.text == "Second response"

    Example: Simulate errors
        >>> fake = FakeProvider(
        ...     error_to_raise=ProviderRateLimit("Rate limit", "fake", retry_after=60)
        ... )
        >>> try:
        ...     await fake.chat(messages, budget=budget)
        ... except ProviderRateLimit as e:
        ...     assert e.retry_after == 60

    Example: Errors before success
        >>> fake = FakeProvider(
        ...     responses=[make_text_response("Success!")],
        ...     errors_before_success=2,
        ...     error_to_raise=ProviderTimeout("Timeout", "fake")
        ... )
        >>> # First two calls raise timeout
        >>> try: await fake.chat(messages, budget=budget)
        >>> except ProviderTimeout: pass
        >>> try: await fake.chat(messages, budget=budget)
        >>> except ProviderTimeout: pass
        >>> # Third call succeeds
        >>> response = await fake.chat(messages, budget=budget)
        >>> assert response.text == "Success!"

    Example: Assert call history
        >>> fake = FakeProvider([make_text_response("result")])
        >>> await fake.chat(
        ...     messages=[ChatMessage(role=Role.user, content="Hello")],
        ...     budget=budget,
        ...     temperature=0.7,
        ...     tools=[ToolDef(name="tool1", description="...", parameters={})]
        ... )
        >>> assert len(fake.call_history) == 1
        >>> call = fake.call_history[0]
        >>> assert call["temperature"] == 0.7
        >>> assert len(call["messages"]) == 1
        >>> assert call["messages"][0].content == "Hello"
        >>> assert len(call["tools"]) == 1
        >>> assert call["tools"][0].name == "tool1"
    """

    def __init__(
        self,
        responses: Optional[List[ChatResponse]] = None,
        error_to_raise: Optional[Exception] = None,
        errors_before_success: int = 0,
        capabilities: Optional[LLMCapabilities] = None,
    ):
        """
        Initialize FakeProvider

        Args:
            responses: List of ChatResponse objects to return (cycles through them)
            error_to_raise: Exception to raise on chat() calls
            errors_before_success: Number of times to raise error before returning response
            capabilities: Provider capabilities (defaults to all features enabled)
        """
        # Initialize base with minimal config
        super().__init__(
            LLMProviderConfig(model="fake-model", api_key_env="FAKE_API_KEY")
        )

        # Responses to return
        self.responses = responses or [
            make_text_response("Fake response", tokens=100, cost_usd=0.01)
        ]
        self.response_index = 0

        # Error simulation
        self.error_to_raise = error_to_raise
        self.errors_before_success = errors_before_success
        self._error_count = 0

        # Call tracking
        self.call_history: List[dict] = []

        # Capabilities (default to all features)
        self._capabilities = capabilities or LLMCapabilities(
            function_calling=True,
            json_schema_mode=True,
            max_output_tokens=4096,
            context_window_tokens=128000,
        )

    @property
    def capabilities(self) -> LLMCapabilities:
        """Return provider capabilities"""
        return self._capabilities

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
        Execute fake chat completion

        Records call parameters and returns configured response or raises configured error.

        Args:
            messages: Conversation history
            tools: Available tools (validated against capabilities if provided)
            json_schema: JSON schema for response (validated against capabilities if provided)
            budget: Budget tracker (enforced)
            temperature: Sampling temperature
            top_p: Nucleus sampling
            seed: Random seed
            tool_choice: Tool selection strategy
            metadata: Additional metadata

        Returns:
            ChatResponse from configured responses list

        Raises:
            ValueError: If unsupported feature requested (e.g., tools but no function calling)
            BudgetExceeded: If response would exceed budget
            ProviderError: If error_to_raise is configured
        """
        # Record call
        self.call_history.append(
            {
                "messages": deepcopy(messages),
                "tools": deepcopy(tools) if tools else None,
                "json_schema": deepcopy(json_schema) if json_schema else None,
                "budget": Budget(
                    max_usd=budget.max_usd,
                    max_tokens=budget.max_tokens,
                    spent_usd=budget.spent_usd,
                    spent_tokens=budget.spent_tokens,
                ),
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
                "tool_choice": tool_choice,
                "metadata": metadata,
            }
        )

        # Validate capabilities
        if tools and not self.capabilities.function_calling:
            raise ValueError("FakeProvider does not support function calling (per configured capabilities)")

        if json_schema and not self.capabilities.json_schema_mode:
            raise ValueError("FakeProvider does not support JSON schema mode (per configured capabilities)")

        # Simulate errors if configured
        if self.error_to_raise:
            # If errors_before_success == 0, always raise error
            # If errors_before_success > 0, raise error N times then succeed
            if self.errors_before_success == 0:
                # Always raise error
                raise self.error_to_raise
            elif self._error_count < self.errors_before_success:
                # Raise error N times, then succeed
                self._error_count += 1
                raise self.error_to_raise
            # After errors_before_success errors, continue to success

        # Get next response (cycle through responses list)
        response = deepcopy(self.responses[self.response_index])
        self.response_index = (self.response_index + 1) % len(self.responses)

        # Check budget BEFORE returning response
        budget.check(
            add_usd=response.usage.cost_usd, add_tokens=response.usage.total_tokens
        )

        # Add usage to budget
        budget.add(
            add_usd=response.usage.cost_usd, add_tokens=response.usage.total_tokens
        )

        return response

    def reset(self) -> None:
        """
        Reset provider state

        Clears call history and resets response index.
        Useful for reusing provider across multiple tests.

        Example:
            >>> fake = FakeProvider([make_text_response("response")])
            >>> await fake.chat(messages, budget=budget)
            >>> assert len(fake.call_history) == 1
            >>> fake.reset()
            >>> assert len(fake.call_history) == 0
        """
        self.call_history = []
        self.response_index = 0
        self._error_count = 0

    def set_responses(self, responses: List[ChatResponse]) -> None:
        """
        Update configured responses

        Useful for changing responses mid-test without creating new provider.

        Args:
            responses: New list of responses to return

        Example:
            >>> fake = FakeProvider([make_text_response("first")])
            >>> fake.set_responses([make_text_response("second")])
            >>> response = await fake.chat(messages, budget=budget)
            >>> assert response.text == "second"
        """
        self.responses = responses
        self.response_index = 0

    def set_error(
        self, error: Optional[Exception], errors_before_success: int = 0
    ) -> None:
        """
        Update error configuration

        Args:
            error: Exception to raise (None to disable errors)
            errors_before_success: Number of times to raise error before succeeding

        Example:
            >>> fake = FakeProvider([make_text_response("success")])
            >>> fake.set_error(ProviderTimeout("timeout", "fake"), errors_before_success=1)
            >>> try: await fake.chat(messages, budget=budget)
            >>> except ProviderTimeout: pass
            >>> response = await fake.chat(messages, budget=budget)
            >>> assert response.text == "success"
        """
        self.error_to_raise = error
        self.errors_before_success = errors_before_success
        self._error_count = 0

    def get_call_count(self) -> int:
        """
        Get number of times chat() was called

        Returns:
            Number of calls

        Example:
            >>> fake = FakeProvider([make_text_response("response")])
            >>> await fake.chat(messages, budget=budget)
            >>> await fake.chat(messages, budget=budget)
            >>> assert fake.get_call_count() == 2
        """
        return len(self.call_history)

    def get_last_call(self) -> Optional[dict]:
        """
        Get parameters from last chat() call

        Returns:
            Dict with call parameters or None if no calls

        Example:
            >>> fake = FakeProvider([make_text_response("response")])
            >>> await fake.chat(messages, budget=budget, temperature=0.8)
            >>> last_call = fake.get_last_call()
            >>> assert last_call["temperature"] == 0.8
        """
        return self.call_history[-1] if self.call_history else None


# =============================================================================
# Helper Functions for Creating Test Responses
# =============================================================================


def make_text_response(
    text: str,
    tokens: int = 100,
    cost_usd: float = 0.01,
    finish_reason: FinishReason = FinishReason.stop,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
) -> ChatResponse:
    """
    Create a simple text response

    Args:
        text: Response text content
        tokens: Total token count (prompt + completion)
        cost_usd: Total cost in USD
        finish_reason: Why generation stopped
        prompt_tokens: Input tokens (defaults to tokens // 2)
        completion_tokens: Output tokens (defaults to tokens // 2)

    Returns:
        ChatResponse with text content

    Example:
        >>> response = make_text_response(
        ...     "Grid intensity is 450 gCO2/kWh",
        ...     tokens=150,
        ...     cost_usd=0.015
        ... )
        >>> assert response.text == "Grid intensity is 450 gCO2/kWh"
        >>> assert response.usage.total_tokens == 150
        >>> assert response.usage.cost_usd == 0.015
    """
    if prompt_tokens is None:
        prompt_tokens = tokens // 2
    if completion_tokens is None:
        completion_tokens = tokens - prompt_tokens

    return ChatResponse(
        text=text,
        tool_calls=[],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost_usd,
        ),
        finish_reason=finish_reason,
        provider_info=ProviderInfo(
            provider="fake", model="fake-model", request_id="fake-req-123"
        ),
        raw=None,
    )


def make_tool_call_response(
    tool_calls: List[dict],
    tokens: int = 100,
    cost_usd: float = 0.01,
    text: Optional[str] = None,
) -> ChatResponse:
    """
    Create a response with tool calls

    Args:
        tool_calls: List of tool call dicts with 'name' and 'arguments' keys
        tokens: Total token count
        cost_usd: Total cost in USD
        text: Optional text content alongside tool calls

    Returns:
        ChatResponse with tool calls

    Example:
        >>> response = make_tool_call_response([
        ...     {
        ...         "name": "get_grid_intensity",
        ...         "arguments": {"region": "CA", "year": 2024}
        ...     },
        ...     {
        ...         "name": "calculate_emissions",
        ...         "arguments": {"kwh": 1000, "intensity": 450}
        ...     }
        ... ])
        >>> assert len(response.tool_calls) == 2
        >>> assert response.tool_calls[0]["name"] == "get_grid_intensity"
        >>> assert response.finish_reason == FinishReason.tool_calls
    """
    # Normalize tool calls (add IDs if missing)
    normalized_calls = []
    for i, call in enumerate(tool_calls):
        normalized_call = {
            "id": call.get("id", f"call_{i:03d}"),
            "name": call["name"],
            "arguments": call.get("arguments", {}),
        }
        normalized_calls.append(normalized_call)

    prompt_tokens = tokens // 2
    completion_tokens = tokens - prompt_tokens

    return ChatResponse(
        text=text,
        tool_calls=normalized_calls,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=tokens,
            cost_usd=cost_usd,
        ),
        finish_reason=FinishReason.tool_calls,
        provider_info=ProviderInfo(
            provider="fake", model="fake-model", request_id="fake-req-123"
        ),
        raw=None,
    )


def make_json_response(
    data: dict,
    tokens: int = 100,
    cost_usd: float = 0.01,
    finish_reason: FinishReason = FinishReason.stop,
) -> ChatResponse:
    """
    Create a JSON response

    Args:
        data: Dictionary to serialize as JSON text
        tokens: Total token count
        cost_usd: Total cost in USD
        finish_reason: Why generation stopped

    Returns:
        ChatResponse with JSON text content

    Example:
        >>> response = make_json_response({
        ...     "intensity": 450.3,
        ...     "unit": "gCO2/kWh",
        ...     "source": "EIA-923"
        ... })
        >>> import json
        >>> result = json.loads(response.text)
        >>> assert result["intensity"] == 450.3
    """
    text = json.dumps(data, indent=2)
    return make_text_response(text, tokens, cost_usd, finish_reason)


def make_error_response(
    error_type: str,
    message: str,
    provider: str = "fake",
    **kwargs,
) -> Exception:
    """
    Create a provider error exception

    Args:
        error_type: Type of error (auth, rate_limit, timeout, server, bad_request)
        message: Error message
        provider: Provider name
        **kwargs: Additional error-specific parameters

    Returns:
        ProviderError subclass

    Example:
        >>> error = make_error_response(
        ...     "rate_limit",
        ...     "Rate limit exceeded",
        ...     retry_after=60
        ... )
        >>> assert isinstance(error, ProviderRateLimit)
        >>> assert error.retry_after == 60

        >>> error = make_error_response("timeout", "Request timed out")
        >>> assert isinstance(error, ProviderTimeout)

        >>> error = make_error_response("auth", "Invalid API key")
        >>> assert isinstance(error, ProviderAuthError)
    """
    error_map = {
        "auth": ProviderAuthError,
        "rate_limit": ProviderRateLimit,
        "timeout": ProviderTimeout,
        "server": ProviderServerError,
        "bad_request": ProviderBadRequest,
    }

    error_class = error_map.get(error_type)
    if not error_class:
        raise ValueError(
            f"Unknown error_type: {error_type}. Must be one of: {list(error_map.keys())}"
        )

    return error_class(message, provider, **kwargs)


# =============================================================================
# Common Test Fixtures
# =============================================================================


def fixture_simple_text_completion() -> FakeProvider:
    """
    Fixture: Simple text completion (no tools, no JSON schema)

    Example:
        >>> fake = fixture_simple_text_completion()
        >>> response = await fake.chat(
        ...     messages=[ChatMessage(role=Role.user, content="Hello")],
        ...     budget=Budget(max_usd=0.50)
        ... )
        >>> assert "climate calculation assistant" in response.text.lower()
    """
    return FakeProvider(
        [
            make_text_response(
                "I'm a climate calculation assistant. How can I help you calculate emissions today?",
                tokens=120,
                cost_usd=0.012,
            )
        ]
    )


def fixture_tool_calling() -> FakeProvider:
    """
    Fixture: Function/tool calling scenario

    Returns provider that requests a tool call, then returns text after tool result.

    Example:
        >>> fake = fixture_tool_calling()
        >>> # First call: LLM requests tool
        >>> response = await fake.chat(messages, tools=[...], budget=budget)
        >>> assert len(response.tool_calls) == 1
        >>> assert response.tool_calls[0]["name"] == "get_grid_intensity"
        >>>
        >>> # Second call: LLM responds with tool result
        >>> messages.append(ChatMessage(
        ...     role=Role.tool,
        ...     name="get_grid_intensity",
        ...     content='{"intensity": 450, "unit": "gCO2/kWh"}',
        ...     tool_call_id=response.tool_calls[0]["id"]
        ... ))
        >>> response = await fake.chat(messages, tools=[...], budget=budget)
        >>> assert "450 gCO2/kWh" in response.text
    """
    return FakeProvider(
        [
            # First response: request tool call
            make_tool_call_response(
                [{"name": "get_grid_intensity", "arguments": {"region": "CA"}}],
                tokens=80,
                cost_usd=0.008,
            ),
            # Second response: use tool result
            make_text_response(
                "Based on the data, California's grid intensity is 450 gCO2/kWh [tool:get_grid_intensity].",
                tokens=150,
                cost_usd=0.015,
            ),
        ]
    )


def fixture_json_schema_response() -> FakeProvider:
    """
    Fixture: JSON schema-constrained response

    Example:
        >>> fake = fixture_json_schema_response()
        >>> response = await fake.chat(
        ...     messages,
        ...     json_schema={"type": "object", "properties": {...}},
        ...     budget=budget
        ... )
        >>> import json
        >>> data = json.loads(response.text)
        >>> assert data["intensity"] == 450.3
        >>> assert data["unit"] == "gCO2/kWh"
    """
    return FakeProvider(
        [
            make_json_response(
                {
                    "intensity": 450.3,
                    "unit": "gCO2/kWh",
                    "source": "EIA-923",
                    "region": "CA",
                    "timestamp": "2024-10-01T12:00:00Z",
                },
                tokens=140,
                cost_usd=0.014,
            )
        ]
    )


def fixture_budget_exceeded() -> FakeProvider:
    """
    Fixture: Response that exceeds budget

    Returns provider with high-cost response to trigger BudgetExceeded.

    Example:
        >>> fake = fixture_budget_exceeded()
        >>> try:
        ...     await fake.chat(messages, budget=Budget(max_usd=0.10))
        ... except BudgetExceeded as e:
        ...     assert e.spent_usd > 0.10
    """
    return FakeProvider(
        [
            make_text_response(
                "This is an expensive response that will exceed budget.",
                tokens=10000,
                cost_usd=1.50,  # High cost
            )
        ]
    )


def fixture_rate_limit_error() -> FakeProvider:
    """
    Fixture: Rate limit error

    Returns provider that raises rate limit error.

    Example:
        >>> fake = fixture_rate_limit_error()
        >>> try:
        ...     await fake.chat(messages, budget=budget)
        ... except ProviderRateLimit as e:
        ...     assert e.retry_after == 60
        ...     assert e.status_code == 429
    """
    return FakeProvider(
        error_to_raise=make_error_response(
            "rate_limit",
            "Rate limit exceeded. Please try again in 60 seconds.",
            retry_after=60,
        )
    )


def fixture_timeout_error() -> FakeProvider:
    """
    Fixture: Timeout error

    Returns provider that raises timeout error.

    Example:
        >>> fake = fixture_timeout_error()
        >>> try:
        ...     await fake.chat(messages, budget=budget)
        ... except ProviderTimeout as e:
        ...     assert e.timeout_seconds == 60.0
    """
    return FakeProvider(
        error_to_raise=make_error_response(
            "timeout",
            "Request timed out after 60 seconds",
            timeout_seconds=60.0,
        )
    )


def fixture_retry_then_success() -> FakeProvider:
    """
    Fixture: Errors then success

    Returns provider that raises timeout twice, then succeeds on third try.

    Example:
        >>> fake = fixture_retry_then_success()
        >>> # First two calls fail
        >>> for _ in range(2):
        ...     try:
        ...         await fake.chat(messages, budget=budget)
        ...     except ProviderTimeout:
        ...         pass  # Expected
        >>> # Third call succeeds
        >>> response = await fake.chat(messages, budget=budget)
        >>> assert response.text == "Success after retries!"
    """
    return FakeProvider(
        responses=[make_text_response("Success after retries!", tokens=100, cost_usd=0.01)],
        error_to_raise=make_error_response("timeout", "Timeout"),
        errors_before_success=2,
    )


# =============================================================================
# Usage Examples
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("=== FakeProvider Examples ===\n")

    async def examples():
        # Example 1: Simple text response
        print("Example 1: Simple text response")
        fake = FakeProvider([make_text_response("Hello, world!", tokens=50, cost_usd=0.005)])
        budget = Budget(max_usd=0.50)
        response = await fake.chat(
            messages=[ChatMessage(role=Role.user, content="Say hello")],
            budget=budget,
        )
        print(f"Response: {response.text}")
        print(f"Cost: ${response.usage.cost_usd:.4f}")
        print(f"Tokens: {response.usage.total_tokens}")
        print(f"Calls made: {fake.get_call_count()}\n")

        # Example 2: Tool calling
        print("Example 2: Tool calling")
        fake = fixture_tool_calling()
        budget = Budget(max_usd=0.50)
        response = await fake.chat(
            messages=[ChatMessage(role=Role.user, content="What's CA grid intensity?")],
            tools=[
                ToolDef(
                    name="get_grid_intensity",
                    description="Get grid carbon intensity",
                    parameters={"type": "object", "properties": {"region": {"type": "string"}}},
                )
            ],
            budget=budget,
        )
        print(f"Tool calls: {len(response.tool_calls)}")
        print(f"Tool name: {response.tool_calls[0]['name']}")
        print(f"Arguments: {response.tool_calls[0]['arguments']}\n")

        # Example 3: JSON response
        print("Example 3: JSON response")
        fake = fixture_json_schema_response()
        budget = Budget(max_usd=0.50)
        response = await fake.chat(
            messages=[ChatMessage(role=Role.user, content="Get grid data")],
            json_schema={"type": "object", "properties": {}},
            budget=budget,
        )
        data = json.loads(response.text)
        print(f"JSON response: {data}")
        print(f"Intensity: {data['intensity']} {data['unit']}\n")

        # Example 4: Simulate rate limit error
        print("Example 4: Simulate rate limit error")
        fake = fixture_rate_limit_error()
        budget = Budget(max_usd=0.50)
        try:
            await fake.chat(
                messages=[ChatMessage(role=Role.user, content="Hello")],
                budget=budget,
            )
            print("ERROR: Should have raised ProviderRateLimit!\n")
        except ProviderRateLimit as e:
            print(f"Caught rate limit error: {e.message}")
            print(f"Retry after: {e.retry_after}s\n")

        # Example 5: Assert call history
        print("Example 5: Assert call history")
        fake = FakeProvider([make_text_response("response")])
        budget = Budget(max_usd=0.50)
        await fake.chat(
            messages=[
                ChatMessage(role=Role.system, content="You are an assistant"),
                ChatMessage(role=Role.user, content="Hello"),
            ],
            budget=budget,
            temperature=0.7,
            seed=42,
        )
        last_call = fake.get_last_call()
        print(f"Last call temperature: {last_call['temperature']}")
        print(f"Last call seed: {last_call['seed']}")
        print(f"Messages in last call: {len(last_call['messages'])}")
        print(f"Budget max_usd: ${last_call['budget'].max_usd:.2f}\n")

        # Example 6: Multiple responses (cycle through)
        print("Example 6: Multiple responses")
        fake = FakeProvider(
            [
                make_text_response("First response"),
                make_text_response("Second response"),
                make_text_response("Third response"),
            ]
        )
        budget = Budget(max_usd=0.50)
        for i in range(5):  # Call more times than responses
            response = await fake.chat(
                messages=[ChatMessage(role=Role.user, content=f"Request {i+1}")],
                budget=budget,
            )
            print(f"Call {i+1}: {response.text}")
        print()

        # Example 7: Retry scenario
        print("Example 7: Retry scenario (2 failures, then success)")
        fake = fixture_retry_then_success()
        budget = Budget(max_usd=0.50)
        for attempt in range(3):
            try:
                response = await fake.chat(
                    messages=[ChatMessage(role=Role.user, content="Retry me")],
                    budget=budget,
                )
                print(f"Attempt {attempt+1}: Success! Response: {response.text}")
                break
            except ProviderTimeout:
                print(f"Attempt {attempt+1}: Timeout (expected)")

    asyncio.run(examples())
    print("\n=== Examples Complete ===")
