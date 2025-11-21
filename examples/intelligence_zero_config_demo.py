# -*- coding: utf-8 -*-
"""
GreenLang Intelligence Layer - Zero Config Demo

Demonstrates the Intelligence Layer working WITHOUT API keys!
Perfect for open source developers to try GreenLang immediately.

This example shows:
1. Auto-detection of API keys (falls back to demo mode if none)
2. Text responses from demo mode
3. Tool calling with demo mode
4. Easy upgrade path to production

Run with:
    # Without API keys (uses demo mode)
    python examples/intelligence_zero_config_demo.py

    # With API keys (uses real LLM)
    export OPENAI_API_KEY=sk-...
    python examples/intelligence_zero_config_demo.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from greenlang.intelligence import (
    create_provider,
    ChatSession,
    ChatMessage,
    Role,
    Budget,
    ToolDef,
    has_any_api_key,
)


async def demo_auto_detect():
    """Demo: Auto-detect provider (uses demo mode if no keys)"""
    print("\n" + "=" * 70)
    print("DEMO 1: Auto-Detect Provider")
    print("=" * 70)

    # Check if API keys are available
    if has_any_api_key():
        print("Status: API keys found! Using real LLM provider.")
    else:
        print("Status: No API keys found. Using demo mode.")
        print("        (Responses are pre-recorded and clearly marked)")

    # Create provider - works without API keys!
    provider = create_provider(model="auto")
    session = ChatSession(provider)

    print(f"\nProvider: {type(provider).__name__}")
    print(f"Model: {provider.config.model}")

    # Simple query
    print("\nQuery: What's the carbon intensity of California's electricity grid?")

    response = await session.chat(
        messages=[
            ChatMessage(
                role=Role.user,
                content="What's the carbon intensity of California's electricity grid?"
            )
        ],
        budget=Budget(max_usd=0.50)
    )

    print(f"\nResponse:")
    print("-" * 70)
    print(response.text)
    print("-" * 70)
    print(f"\nCost: ${response.usage.cost_usd:.6f}")
    print(f"Tokens: {response.usage.total_tokens}")


async def demo_fuel_emissions():
    """Demo: Fuel emissions calculation"""
    print("\n" + "=" * 70)
    print("DEMO 2: Fuel Emissions Query")
    print("=" * 70)

    provider = create_provider()
    session = ChatSession(provider)

    print("Query: How much CO2 is produced by burning 100 gallons of diesel?")

    response = await session.chat(
        messages=[
            ChatMessage(
                role=Role.user,
                content="How much CO2 is produced by burning 100 gallons of diesel?"
            )
        ],
        budget=Budget(max_usd=0.50)
    )

    print(f"\nResponse:")
    print("-" * 70)
    print(response.text)
    print("-" * 70)


async def demo_tool_calling():
    """Demo: Tool calling with demo mode"""
    print("\n" + "=" * 70)
    print("DEMO 3: Tool Calling")
    print("=" * 70)

    provider = create_provider()
    session = ChatSession(provider)

    # Define a tool
    tools = [
        ToolDef(
            name="calculate_fuel_emissions",
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
                        "description": "Amount of fuel",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["gallons", "liters"],
                        "description": "Unit of measurement",
                    },
                },
                "required": ["fuel_type", "amount"],
            },
        )
    ]

    print("Available tools:")
    print(f"  - {tools[0].name}: {tools[0].description}")

    print("\nQuery: Calculate emissions for 100 gallons of diesel")

    response = await session.chat(
        messages=[
            ChatMessage(
                role=Role.user,
                content="Calculate emissions for 100 gallons of diesel"
            )
        ],
        tools=tools,
        budget=Budget(max_usd=0.50)
    )

    if response.tool_calls:
        print("\nTool call requested:")
        print("-" * 70)
        for tc in response.tool_calls:
            print(f"Tool: {tc['name']}")
            print(f"Arguments: {tc['arguments']}")

            # In real usage, you'd execute the tool here
            # For demo, we'll show what would happen
            from greenlang.intelligence.demo_responses import get_demo_tool_result
            result = get_demo_tool_result(tc['name'], tc['arguments'])
            print(f"\nTool result:")
            print(result)
        print("-" * 70)
    else:
        print(f"\nText response:")
        print("-" * 70)
        print(response.text)
        print("-" * 70)


async def demo_explicit_demo_mode():
    """Demo: Explicitly request demo mode"""
    print("\n" + "=" * 70)
    print("DEMO 4: Explicit Demo Mode")
    print("=" * 70)

    print("Explicitly requesting demo mode (even if API keys present)")

    # Force demo mode
    provider = create_provider(model="demo")
    session = ChatSession(provider)

    print(f"Provider: {type(provider).__name__}")

    response = await session.chat(
        messages=[
            ChatMessage(
                role=Role.user,
                content="What are carbon offsets?"
            )
        ],
        budget=Budget(max_usd=0.50)
    )

    print(f"\nResponse:")
    print("-" * 70)
    print(response.text)
    print("-" * 70)


async def demo_multiple_queries():
    """Demo: Multiple queries to show consistency"""
    print("\n" + "=" * 70)
    print("DEMO 5: Multiple Queries")
    print("=" * 70)

    provider = create_provider()
    session = ChatSession(provider)
    budget = Budget(max_usd=1.00)

    queries = [
        "What's the grid intensity in Texas?",
        "Explain renewable energy",
        "What is a carbon footprint?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: {query}")

        response = await session.chat(
            messages=[ChatMessage(role=Role.user, content=query)],
            budget=budget
        )

        # Show first 150 chars of response
        preview = response.text[:150] + "..." if len(response.text) > 150 else response.text
        print(f"   Response: {preview}")
        print(f"   Cost: ${response.usage.cost_usd:.6f}")

    print(f"\nTotal budget used: ${budget.spent_usd:.6f} / ${budget.max_usd:.2f}")


async def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("GREENLANG INTELLIGENCE LAYER - ZERO CONFIG DEMO")
    print("=" * 70)
    print("\nThis demo works WITHOUT API keys!")
    print("Responses come from pre-recorded demo data.")
    print("\nTo upgrade to production:")
    print("  export OPENAI_API_KEY=sk-...")
    print("  or")
    print("  export ANTHROPIC_API_KEY=sk-...")
    print("=" * 70)

    try:
        await demo_auto_detect()
        await demo_fuel_emissions()
        await demo_tool_calling()
        await demo_explicit_demo_mode()
        await demo_multiple_queries()

        print("\n" + "=" * 70)
        print("ALL DEMOS COMPLETE!")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("1. Works immediately without API keys (demo mode)")
        print("2. Same API for demo and production modes")
        print("3. Clear warnings when in demo mode")
        print("4. Easy upgrade path (just add API key)")
        print("\nNext steps:")
        print("- Try with your own queries")
        print("- Add API key for production use")
        print("- Explore tool calling and JSON mode")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
