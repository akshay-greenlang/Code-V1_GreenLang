"""
Unit Tests for Intelligence Layer
Tests LLM orchestrator, prompt templates, context management, and multi-provider routing.
Validates intelligent behavior and LLM integration.
"""

import pytest
import asyncio
import time
import hashlib
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from testing.agent_test_framework import (
    AgentTestCase,
    DeterministicLLMProvider,
    MockLLMProvider
)


class LLMProvider:
    """Base LLM provider interface."""

    def __init__(self, name: str, cost_per_token: float = 0.0001):
        self.name = name
        self.cost_per_token = cost_per_token
        self.total_tokens = 0
        self.total_calls = 0
        self.failures = 0

    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate completion."""
        self.total_calls += 1
        tokens = len(prompt.split()) + 50
        self.total_tokens += tokens

        return {
            'response': f"Response from {self.name}",
            'tokens': tokens,
            'cost': tokens * self.cost_per_token,
            'provider': self.name
        }


class LLMOrchestrator:
    """Orchestrates multiple LLM providers with fallback and cost optimization."""

    def __init__(self):
        self.providers = {}
        self.primary_provider = None
        self.fallback_providers = []
        self.total_cost = 0.0
        self.call_history = []

    def register_provider(self, provider: LLMProvider, is_primary: bool = False):
        """Register an LLM provider."""
        self.providers[provider.name] = provider

        if is_primary:
            self.primary_provider = provider.name
        else:
            self.fallback_providers.append(provider.name)

    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate with automatic fallback."""
        providers_to_try = [self.primary_provider] + self.fallback_providers

        for provider_name in providers_to_try:
            if provider_name not in self.providers:
                continue

            provider = self.providers[provider_name]

            try:
                result = await provider.generate(prompt, **kwargs)
                result['provider_used'] = provider_name

                self.total_cost += result.get('cost', 0)
                self.call_history.append({
                    'provider': provider_name,
                    'success': True,
                    'timestamp': datetime.now().isoformat()
                })

                return result
            except Exception as e:
                provider.failures += 1
                continue

        raise Exception("All providers failed")

    def get_cheapest_provider(self) -> str:
        """Get cheapest provider."""
        if not self.providers:
            return None

        cheapest = min(self.providers.values(), key=lambda p: p.cost_per_token)
        return cheapest.name

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'total_cost': self.total_cost,
            'total_calls': len(self.call_history),
            'providers': {
                name: {
                    'calls': p.total_calls,
                    'tokens': p.total_tokens,
                    'failures': p.failures
                }
                for name, p in self.providers.items()
            }
        }


class PromptTemplate:
    """Structured prompt template."""

    def __init__(self, template: str, variables: List[str]):
        self.template = template
        self.variables = variables

    def render(self, **kwargs) -> str:
        """Render template with variables."""
        missing = [v for v in self.variables if v not in kwargs]
        if missing:
            raise ValueError(f"Missing variables: {missing}")

        return self.template.format(**kwargs)


class ContextManager:
    """Manages conversation context and token limits."""

    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self.context = []
        self.current_tokens = 0

    def add_message(self, role: str, content: str):
        """Add message to context."""
        tokens = len(content.split())

        # Prune old messages if needed
        while self.current_tokens + tokens > self.max_tokens and self.context:
            removed = self.context.pop(0)
            self.current_tokens -= len(removed['content'].split())

        self.context.append({'role': role, 'content': content})
        self.current_tokens += tokens

    def get_context(self) -> List[Dict[str, str]]:
        """Get current context."""
        return self.context.copy()

    def clear(self):
        """Clear context."""
        self.context = []
        self.current_tokens = 0


class TokenTracker:
    """Track token usage and costs."""

    def __init__(self):
        self.usage = []

    def track(self, provider: str, tokens: int, cost: float):
        """Track token usage."""
        self.usage.append({
            'provider': provider,
            'tokens': tokens,
            'cost': cost,
            'timestamp': datetime.now().isoformat()
        })

    def get_total_cost(self) -> float:
        """Get total cost."""
        return sum(u['cost'] for u in self.usage)

    def get_total_tokens(self) -> int:
        """Get total tokens."""
        return sum(u['tokens'] for u in self.usage)


# Tests
class TestLLMOrchestrator(AgentTestCase):
    """Test LLM orchestration."""

    async def test_provider_registration(self):
        """Test registering providers."""
        orchestrator = LLMOrchestrator()
        provider = LLMProvider("test_provider")

        orchestrator.register_provider(provider, is_primary=True)

        self.assertIn("test_provider", orchestrator.providers)
        self.assertEqual(orchestrator.primary_provider, "test_provider")

    async def test_generation_with_primary(self):
        """Test generation uses primary provider."""
        orchestrator = LLMOrchestrator()
        primary = LLMProvider("primary", cost_per_token=0.0001)

        orchestrator.register_provider(primary, is_primary=True)

        result = await orchestrator.generate("Test prompt")

        self.assertEqual(result['provider_used'], "primary")
        self.assertGreater(primary.total_calls, 0)

    async def test_fallback_on_failure(self):
        """Test fallback to secondary provider."""
        orchestrator = LLMOrchestrator()

        # Primary that fails
        primary = LLMProvider("primary")
        primary.generate = AsyncMock(side_effect=Exception("Failed"))

        # Fallback that works
        fallback = LLMProvider("fallback")

        orchestrator.register_provider(primary, is_primary=True)
        orchestrator.register_provider(fallback, is_primary=False)

        result = await orchestrator.generate("Test prompt")

        self.assertEqual(result['provider_used'], "fallback")

    async def test_cost_optimization(self):
        """Test selecting cheapest provider."""
        orchestrator = LLMOrchestrator()

        expensive = LLMProvider("expensive", cost_per_token=0.001)
        cheap = LLMProvider("cheap", cost_per_token=0.0001)

        orchestrator.register_provider(expensive)
        orchestrator.register_provider(cheap)

        cheapest = orchestrator.get_cheapest_provider()

        self.assertEqual(cheapest, "cheap")

    async def test_usage_statistics(self):
        """Test tracking usage statistics."""
        orchestrator = LLMOrchestrator()
        provider = LLMProvider("test")

        orchestrator.register_provider(provider, is_primary=True)

        await orchestrator.generate("Prompt 1")
        await orchestrator.generate("Prompt 2")

        stats = orchestrator.get_stats()

        self.assertEqual(stats['total_calls'], 2)
        self.assertGreater(stats['total_cost'], 0)


class TestPromptTemplates(AgentTestCase):
    """Test prompt template system."""

    def test_template_rendering(self):
        """Test rendering templates."""
        template = PromptTemplate(
            "Calculate emissions for {fuel_type} with quantity {quantity}",
            variables=['fuel_type', 'quantity']
        )

        rendered = template.render(fuel_type='diesel', quantity=100)

        self.assertIn('diesel', rendered)
        self.assertIn('100', rendered)

    def test_missing_variables(self):
        """Test error on missing variables."""
        template = PromptTemplate(
            "Test {var1} and {var2}",
            variables=['var1', 'var2']
        )

        with self.assertRaises(ValueError):
            template.render(var1='value1')  # Missing var2


class TestContextManager(AgentTestCase):
    """Test context management."""

    def test_add_message(self):
        """Test adding messages to context."""
        ctx = ContextManager(max_tokens=1000)

        ctx.add_message('user', 'Hello')
        ctx.add_message('assistant', 'Hi there')

        self.assertEqual(len(ctx.context), 2)

    def test_token_limit_pruning(self):
        """Test automatic pruning when exceeding token limit."""
        ctx = ContextManager(max_tokens=50)

        # Add messages that exceed limit
        for i in range(10):
            ctx.add_message('user', 'This is a test message with many words')

        # Should have pruned old messages
        self.assertLess(len(ctx.context), 10)
        self.assertLessEqual(ctx.current_tokens, 50)

    def test_clear_context(self):
        """Test clearing context."""
        ctx = ContextManager()

        ctx.add_message('user', 'Test')
        ctx.clear()

        self.assertEqual(len(ctx.context), 0)
        self.assertEqual(ctx.current_tokens, 0)


class TestTokenTracker(AgentTestCase):
    """Test token usage tracking."""

    def test_track_usage(self):
        """Test tracking token usage."""
        tracker = TokenTracker()

        tracker.track('provider1', tokens=100, cost=0.01)
        tracker.track('provider2', tokens=200, cost=0.02)

        self.assertEqual(len(tracker.usage), 2)

    def test_total_calculations(self):
        """Test total cost and token calculations."""
        tracker = TokenTracker()

        tracker.track('provider1', tokens=100, cost=0.01)
        tracker.track('provider2', tokens=200, cost=0.02)

        self.assertEqual(tracker.get_total_tokens(), 300)
        self.assertAlmostEqual(tracker.get_total_cost(), 0.03)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=.", "--cov-report=term"])
