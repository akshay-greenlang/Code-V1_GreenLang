# -*- coding: utf-8 -*-
"""
Integration Example: ChatSession + SemanticCache
=================================================

Demonstrates how to integrate ChatSession with SemanticCache for cost savings.
"""

import asyncio
from datetime import datetime
from greenlang.intelligence import ChatSession, ChatMessage, MessageRole
from greenlang.intelligence.cache_warming import SemanticCache
from greenlang.intelligence.providers import get_provider, ProviderType
from greenlang.determinism import DeterministicClock


async def main():
    """Run ChatSession + SemanticCache integration."""
    # Initialize provider
    provider = get_provider(ProviderType.OPENAI)

    # Initialize semantic cache
    semantic_cache = SemanticCache(
        similarity_threshold=0.85,
        embedding_model="text-embedding-ada-002"
    )

    # Initialize chat session
    chat = ChatSession(
        provider=provider,
        model="gpt-3.5-turbo",
        system_message="You are a sustainability expert."
    )

    # Function to ask question with caching
    async def ask_with_cache(question: str) -> tuple[str, bool, float]:
        start = DeterministicClock.now()

        # Check cache first
        cached = await semantic_cache.get(question)
        if cached:
            duration = (DeterministicClock.now() - start).total_seconds()
            return cached["response"], True, duration

        # Cache miss - call LLM
        messages = [ChatMessage(role=MessageRole.USER, content=question)]
        response = await chat.chat(messages)

        # Store in cache
        await semantic_cache.set(question, {
            "response": response.content,
            "model": chat.model
        })

        duration = (DeterministicClock.now() - start).total_seconds()
        return response.content, False, duration

    # Test with similar questions
    questions = [
        "What is carbon footprint?",
        "What is a carbon footprint?",  # Very similar - should hit cache
        "Explain carbon footprint",      # Similar - should hit cache
        "What are greenhouse gases?"     # Different - cache miss
    ]

    print("\nChatSession + SemanticCache Integration")
    print("=" * 60)

    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        answer, cached, duration = await ask_with_cache(question)

        print(f"  Cached: {'YES' if cached else 'NO'}")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Answer: {answer[:100]}...")

    # Show cache statistics
    print(f"\nCache Statistics:")
    print(f"  Hit rate: {semantic_cache.get_hit_rate():.1f}%")
    print(f"  Cache size: {len(semantic_cache.cache)}")


if __name__ == "__main__":
    asyncio.run(main())
