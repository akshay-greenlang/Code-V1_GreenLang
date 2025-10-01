"""
FakeProvider - Demo Mode LLM Provider

Provides simulated LLM responses without requiring API keys.
Perfect for:
- Open source developers trying GreenLang
- Development/testing without API costs
- CI/CD pipelines
- Demonstrations and tutorials

Features:
- Pre-recorded climate responses
- Tool call demonstrations
- Realistic response times (100-500ms delay)
- Budget tracking (simulated)
- All responses clearly marked as "demo mode"

Usage:
    # FakeProvider is automatically used when no API keys are available
    from greenlang.intelligence import create_provider

    provider = create_provider()  # Auto-selects FakeProvider if no keys
    # OR explicitly request demo mode
    provider = create_provider(model="demo")
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any, List, Mapping, Optional

from greenlang.intelligence.providers.base import (
    LLMProvider,
    LLMProviderConfig,
    LLMCapabilities,
)
from greenlang.intelligence.schemas.messages import ChatMessage, Role
from greenlang.intelligence.schemas.tools import ToolDef
from greenlang.intelligence.schemas.responses import ChatResponse, Usage, FinishReason, ProviderInfo
from greenlang.intelligence.schemas.jsonschema import JSONSchema
from greenlang.intelligence.runtime.budget import Budget, BudgetExceeded
from greenlang.intelligence.demo_responses import get_demo_response, get_demo_tool_result

logger = logging.getLogger(__name__)


class FakeProvider(LLMProvider):
    """
    Fake LLM provider for demo mode

    Provides pre-recorded responses for common climate queries without requiring API keys.
    All responses are clearly marked as "demo mode" and encourage users to add API keys
    for production use.

    Capabilities:
    - Function calling (simulated)
    - JSON schema mode (basic support)
    - Budget tracking (simulated costs)
    - Realistic response times (100-500ms)

    Usage:
        config = LLMProviderConfig(
            model="demo",
            api_key_env="",  # Not required for FakeProvider
        )
        provider = FakeProvider(config)

        response = await provider.chat(
            messages=[
                ChatMessage(role=Role.user, content="What's the carbon intensity in CA?")
            ],
            budget=Budget(max_usd=0.50)
        )
        print(response.text)
        # Output includes: "...approximately 200-250 gCO2e/kWh (demo mode)..."

    Cost simulation:
    - Simulated cost: $0.00001 per request (negligible)
    - Token counting: ~4 chars per token (rough estimate)
    - Budget tracking works normally

    Security:
    - No API calls made
    - No data leaves your machine
    - No API keys required or used
    """

    def __init__(self, config: LLMProviderConfig) -> None:
        """
        Initialize FakeProvider

        Args:
            config: Provider configuration (api_key_env not required)
        """
        super().__init__(config)

        logger.info(
            "Initialized FakeProvider (demo mode) - "
            "All responses are simulated. Add API key for production use."
        )

    @property
    def capabilities(self) -> LLMCapabilities:
        """
        Return provider capabilities

        FakeProvider simulates a capable LLM with function calling and JSON mode.

        Returns:
            Capabilities (function calling and JSON mode enabled)
        """
        return LLMCapabilities(
            function_calling=True,
            json_schema_mode=True,
            max_output_tokens=4096,
            context_window_tokens=16000,
        )

    def _estimate_tokens(self, messages: List[ChatMessage], tools: Optional[List[ToolDef]] = None) -> int:
        """
        Estimate token count from messages

        Uses simple heuristic: 4 chars per token

        Args:
            messages: Chat messages
            tools: Tool definitions

        Returns:
            Estimated token count
        """
        total_chars = 0
        for msg in messages:
            if msg.content:
                total_chars += len(msg.content)
            total_chars += 20  # Overhead for role, formatting

        if tools:
            for tool in tools:
                total_chars += len(tool.name) + len(tool.description)
                total_chars += len(str(tool.parameters))

        return total_chars // 4

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
        Execute chat completion (simulated)

        Returns pre-recorded responses based on query content.
        Simulates realistic behavior including:
        - Response latency (100-500ms)
        - Token counting
        - Budget tracking
        - Tool calls (when appropriate)

        Args:
            messages: Conversation history
            tools: Available tools (may trigger tool call responses)
            json_schema: JSON schema (basic validation support)
            budget: Budget tracker (simulated cost ~$0.00001)
            temperature: Ignored in demo mode
            top_p: Ignored in demo mode
            seed: Ignored in demo mode
            tool_choice: Influences whether to return tool calls
            metadata: Ignored in demo mode

        Returns:
            ChatResponse with demo text or tool calls

        Raises:
            BudgetExceeded: If simulated cost would exceed budget
        """
        # 1. Estimate tokens (input)
        prompt_tokens = self._estimate_tokens(messages, tools)

        # 2. Simulate cost (very low for demo mode)
        # Use ~$0.00001 per request (negligible but non-zero for budget tracking)
        estimated_cost = 0.00001

        # 3. Check budget BEFORE generating response
        budget.check(add_usd=estimated_cost, add_tokens=prompt_tokens)

        logger.debug(
            f"FakeProvider (demo mode): {prompt_tokens} tokens, "
            f"${estimated_cost:.6f} simulated cost"
        )

        # 4. Extract user query from last message
        query = ""
        for msg in reversed(messages):
            if msg.role == Role.user and msg.content:
                query = msg.content
                break

        # 5. Simulate realistic response delay (100-500ms)
        delay = 0.1 + (abs(hash(query)) % 400) / 1000  # 100-500ms
        await asyncio.sleep(delay)

        # 6. Get demo response
        demo_result = get_demo_response(query, tools)

        # 7. Build response based on type
        text = None
        tool_calls = []
        finish_reason = FinishReason.stop

        if demo_result["type"] == "tool_call":
            # Tool call response
            tool_calls = demo_result["tool_calls"]
            finish_reason = FinishReason.tool_calls
            text = None  # No text when tool calls present
        else:
            # Text response
            text = demo_result["content"]

            # If JSON schema requested, try to format as JSON
            if json_schema:
                # Simple JSON wrapper for demo mode
                text = f'{{"response": "{text[:200]}...", "demo_mode": true}}'

        # 8. Calculate completion tokens
        completion_tokens = len(text) // 4 if text else 50  # Rough estimate

        # 9. Create usage info
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=estimated_cost,
        )

        # 10. Provider info
        provider_info = ProviderInfo(
            provider="fake",
            model="demo",
            request_id="demo_" + str(abs(hash(query)))[:10],
        )

        # 11. Add to budget
        budget.add(add_usd=usage.cost_usd, add_tokens=usage.total_tokens)

        logger.info(
            f"FakeProvider response: {usage.total_tokens} tokens, "
            f"${usage.cost_usd:.6f}, finish_reason={finish_reason.value} (DEMO MODE)"
        )

        # 12. Return response
        return ChatResponse(
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
            provider_info=provider_info,
            raw=None,
        )


if __name__ == "__main__":
    """
    Demo of FakeProvider - no API key required!

    Run with:
        python -m greenlang.intelligence.providers.fake
    """
    import asyncio

    async def demo_text_response():
        """Demo: Text response"""
        print("\n=== Demo: Text Response ===")

        config = LLMProviderConfig(
            model="demo",
            api_key_env="",
        )
        provider = FakeProvider(config)

        messages = [
            ChatMessage(
                role=Role.user,
                content="What's the carbon intensity of California's electricity grid?"
            )
        ]

        budget = Budget(max_usd=0.10)

        response = await provider.chat(messages=messages, budget=budget)

        print(f"Response:\n{response.text}\n")
        print(f"Cost: ${response.usage.cost_usd:.6f}")
        print(f"Tokens: {response.usage.total_tokens}")

    async def demo_tool_call():
        """Demo: Tool call"""
        print("\n=== Demo: Tool Call ===")

        config = LLMProviderConfig(
            model="demo",
            api_key_env="",
        )
        provider = FakeProvider(config)

        tools = [
            ToolDef(
                name="calculate_fuel_emissions",
                description="Calculate CO2 emissions from fuel combustion",
                parameters={
                    "type": "object",
                    "properties": {
                        "fuel_type": {"type": "string"},
                        "amount": {"type": "number"},
                    },
                    "required": ["fuel_type", "amount"],
                },
            )
        ]

        messages = [
            ChatMessage(
                role=Role.user,
                content="Calculate emissions for 100 gallons of diesel"
            )
        ]

        budget = Budget(max_usd=0.10)

        response = await provider.chat(
            messages=messages,
            tools=tools,
            budget=budget
        )

        if response.tool_calls:
            print("Tool call requested:")
            for tc in response.tool_calls:
                print(f"  Tool: {tc['name']}")
                print(f"  Arguments: {tc['arguments']}")

                # Simulate tool execution
                result = get_demo_tool_result(tc['name'], tc['arguments'])
                print(f"  Result:\n{result}")
        else:
            print(f"Text response: {response.text}")

    async def demo_fuel_emissions():
        """Demo: Fuel emissions query"""
        print("\n=== Demo: Fuel Emissions ===")

        config = LLMProviderConfig(
            model="demo",
            api_key_env="",
        )
        provider = FakeProvider(config)

        messages = [
            ChatMessage(
                role=Role.user,
                content="How much CO2 is produced by burning diesel fuel?"
            )
        ]

        budget = Budget(max_usd=0.10)

        response = await provider.chat(messages=messages, budget=budget)

        print(f"Response:\n{response.text}\n")

    async def main():
        """Run all demos"""
        print("FakeProvider Demo - No API Key Required!")
        print("=" * 60)

        await demo_text_response()
        await demo_tool_call()
        await demo_fuel_emissions()

        print("\n" + "=" * 60)
        print("Demo complete! To use real LLM providers:")
        print("  1. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        print("  2. Use create_provider() to auto-detect")

    asyncio.run(main())
