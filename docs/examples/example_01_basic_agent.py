# -*- coding: utf-8 -*-
"""
Example 01: Basic Agent

This example demonstrates the fundamental concepts of creating a simple
agent using the BaseAgent class. You'll learn:
- How to create a custom agent
- How to implement the execute method
- How to validate inputs
- How to run an agent and check results
"""

from greenlang.agents import BaseAgent, AgentConfig, AgentResult
from typing import Dict, Any


class HelloWorldAgent(BaseAgent):
    """A simple agent that greets users."""

    def __init__(self):
        config = AgentConfig(
            name="HelloWorldAgent",
            description="A simple greeting agent",
            version="1.0.0",
            enable_metrics=True
        )
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Generate a personalized greeting.

        Args:
            input_data: Must contain 'name' key

        Returns:
            AgentResult with greeting message
        """
        name = input_data.get('name', 'World')
        greeting = f"Hello, {name}! Welcome to GreenLang."

        return AgentResult(
            success=True,
            data={"greeting": greeting, "name": name},
            metadata={"message_length": len(greeting)}
        )

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate that input contains a name.

        Args:
            input_data: Input data to validate

        Returns:
            True if valid, False otherwise
        """
        if 'name' not in input_data:
            self.logger.error("Missing required field: name")
            return False

        if not isinstance(input_data['name'], str):
            self.logger.error("name must be a string")
            return False

        if len(input_data['name'].strip()) == 0:
            self.logger.error("name cannot be empty")
            return False

        return True


def main():
    """Run the example."""
    print("=" * 60)
    print("Example 01: Basic Agent")
    print("=" * 60)
    print()

    # Create an instance of the agent
    agent = HelloWorldAgent()

    # Example 1: Valid input
    print("Test 1: Valid input")
    print("-" * 40)
    result = agent.run({"name": "Alice"})

    if result.success:
        print(f"✓ Success!")
        print(f"  Greeting: {result.data['greeting']}")
        print(f"  Execution time: {result.metrics.execution_time_ms:.2f}ms")
    else:
        print(f"✗ Failed: {result.error}")
    print()

    # Example 2: Default name
    print("Test 2: No name provided (uses default)")
    print("-" * 40)
    result = agent.run({})

    if result.success:
        print(f"✓ Success!")
        print(f"  Greeting: {result.data['greeting']}")
    else:
        print(f"✗ Failed: {result.error}")
    print()

    # Example 3: Invalid input (will fail validation)
    print("Test 3: Invalid input (empty string)")
    print("-" * 40)
    result = agent.run({"name": ""})

    if result.success:
        print(f"✓ Success!")
        print(f"  Greeting: {result.data['greeting']}")
    else:
        print(f"✗ Failed (expected): {result.error}")
    print()

    # Example 4: Check agent statistics
    print("Agent Statistics:")
    print("-" * 40)
    stats = agent.get_stats()
    print(f"  Total executions: {stats['executions']}")
    print(f"  Success rate: {stats['success_rate']}%")
    print(f"  Average time: {stats['avg_time_ms']:.2f}ms")
    print()

    print("=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
