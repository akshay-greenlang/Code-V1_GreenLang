"""
Example 2: Simple LLM Chatbot
==============================

Demonstrates ChatSession usage for LLM interactions.
"""

import asyncio
from greenlang.intelligence import ChatSession, ChatMessage, MessageRole
from greenlang.intelligence.providers import get_provider, ProviderType


async def main():
    """Run simple chatbot example."""
    # Initialize provider (requires OPENAI_API_KEY env variable)
    provider = get_provider(ProviderType.OPENAI)

    # Create chat session
    chat = ChatSession(
        provider=provider,
        model="gpt-3.5-turbo",
        temperature=0.7,
        system_message="You are a helpful sustainability expert."
    )

    # Send a message
    messages = [
        ChatMessage(role=MessageRole.USER, content="What is Scope 1 emissions?")
    ]

    response = await chat.chat(messages)

    print(f"User: {messages[0].content}")
    print(f"Assistant: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
