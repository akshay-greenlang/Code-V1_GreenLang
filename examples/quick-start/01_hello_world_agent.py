# -*- coding: utf-8 -*-
"""
Example 1: Hello World Agent
=============================

The simplest possible GreenLang agent.
Demonstrates basic agent creation and execution.
"""

import asyncio
from greenlang.agents.templates import BaseAgent


async def main():
    """Run hello world agent example."""
    # Create a simple agent
    agent = BaseAgent(name="HelloWorldAgent")

    # Execute agent
    result = await agent.execute({"message": "Hello, GreenLang!"})

    print(f"Agent: {agent.name}")
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
