"""
INTL-103 Tool Runtime Demo

Demonstrates "no naked numbers" enforcement with a working example.

This shows how to:
1. Define a tool with Quantity outputs
2. Register tools
3. Run the tool runtime with a mock provider
4. Handle claims and {{claim:i}} macros

CTO Specification: This is the keystone for everything else.
"""

from greenlang.intelligence.runtime.tools import Tool, ToolRegistry, ToolRuntime
from greenlang.intelligence.runtime.schemas import Quantity
from unittest.mock import Mock


# ============================================================================
# STEP 1: DEFINE TOOLS
# ============================================================================

# Define energy intensity calculator tool
energy_intensity = Tool(
    name="energy_intensity",
    description="Compute kWh/m2 given annual kWh and floor area.",
    args_schema={
        "type": "object",
        "required": ["annual_kwh", "floor_m2"],
        "properties": {
            "annual_kwh": {"type": "number", "minimum": 0},
            "floor_m2": {"type": "number", "exclusiveMinimum": 0}
        }
    },
    result_schema={
        "type": "object",
        "required": ["intensity"],
        "properties": {
            "intensity": {"$ref": "greenlang://schemas/quantity.json"}
        }
    },
    fn=lambda annual_kwh, floor_m2: {
        "intensity": {"value": annual_kwh / floor_m2, "unit": "kWh/m2"}
    }
)

# ============================================================================
# STEP 2: CREATE MOCK PROVIDER
# ============================================================================

def create_mock_provider():
    """
    Create a mock provider that simulates LLM behavior

    The provider must implement:
    - init_chat(system_prompt, user_msg) -> state
    - chat_step(schema, tools, state) -> AssistantStep
    - inject_tool_result(result) -> new_state
    - inject_error(msg) -> new_state
    """
    provider = Mock()

    # Initialize chat
    provider.init_chat = Mock(return_value="state_0")

    # Simulate LLM behavior:
    # 1. First step: call tool
    # 2. Second step: finalize with claims
    provider.chat_step = Mock(side_effect=[
        # Step 1: Tool call
        {
            "kind": "tool_call",
            "tool_name": "energy_intensity",
            "arguments": {
                "annual_kwh": 12000,
                "floor_m2": 1000
            }
        },
        # Step 2: Final with claims
        {
            "kind": "final",
            "final": {
                "message": "The energy intensity for this building is {{claim:0}}.",
                "claims": [
                    {
                        "source_call_id": "tc_1",
                        "path": "$.intensity",
                        "quantity": {"value": 12.0, "unit": "kWh/m2"}
                    }
                ]
            }
        }
    ])

    provider.inject_tool_result = Mock(return_value="state_1")
    provider.inject_error = Mock(return_value="state_error")

    return provider


# ============================================================================
# STEP 3: REGISTER TOOLS & RUN
# ============================================================================

def main():
    """Main demo"""

    # 1. Create registry and register tools
    registry = ToolRegistry()
    registry.register(energy_intensity)

    # 2. Create mock provider
    provider = create_mock_provider()

    # 3. Create runtime (in Replay mode for testing)
    runtime = ToolRuntime(provider, registry, mode="Replay")

    # 4. Run the tool runtime
    result = runtime.run(
        system_prompt="You are a climate advisor.",
        user_msg="What's the energy intensity of the building?"
    )

    # 5. Display results
    print("=" * 60)
    print("INTL-103 Tool Runtime Demo")
    print("=" * 60)
    print()
    print("Final Message:")
    print(result["message"])
    print()
    print("Provenance:")
    for claim in result["provenance"]:
        print(f"  - Claim from {claim['source_call_id']} at {claim['path']}")
        print(f"    Quantity: {claim['quantity']}")
    print()

    # 6. Show metrics
    metrics = runtime.get_metrics()
    print("Metrics:")
    print(f"  Tool Use Rate: {metrics['tool_use_rate']:.1%}")
    print(f"  Total Tool Calls: {metrics['total_tool_calls']}")
    print(f"  Naked Number Rejections: {metrics['naked_number_rejections']}")
    print()

    # 7. Show provenance log
    print("Provenance Log:")
    for entry in runtime.provenance:
        print(f"  [{entry['id']}] {entry['tool_name']}")
        print(f"    Arguments: {entry['arguments']}")
        print(f"    Output: {entry['output']}")
        print(f"    Quantities: {list(entry['unit_index'].keys())}")
    print()

    print("✅ Demo completed successfully!")
    print()
    print("Key Takeaways:")
    print("1. Tool must return Quantity {value, unit} for ALL numerics")
    print("2. LLM must use {{claim:i}} macros, not raw numbers")
    print("3. Claims link macros to tool outputs via JSONPath")
    print("4. Runtime validates and renders the final message")


# ============================================================================
# ALTERNATIVE: EXAMPLE OF WHAT FAILS
# ============================================================================

def demo_failure():
    """Show what happens when LLM tries to emit naked numbers"""

    registry = ToolRegistry()
    registry.register(energy_intensity)

    # Provider that tries to emit naked number
    provider = Mock()
    provider.init_chat = Mock(return_value="state_0")
    provider.chat_step = Mock(return_value={
        "kind": "final",
        "final": {
            "message": "The energy intensity is 12.0 kWh/m2.",  # NAKED NUMBER!
            "claims": []
        }
    })

    runtime = ToolRuntime(provider, registry)

    try:
        result = runtime.run("You are a climate advisor.", "What's the intensity?")
    except Exception as e:
        print("=" * 60)
        print("Example: Naked Number Rejection")
        print("=" * 60)
        print()
        print(f"❌ Error: {e}")
        print()
        print("This is expected! The runtime blocked the naked number '12.0'.")
        print("The LLM must either:")
        print("1. Call a tool to get the value, OR")
        print("2. Use {{claim:i}} macros backed by claims[]")


if __name__ == "__main__":
    main()
    print()
    demo_failure()
