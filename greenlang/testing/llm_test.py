"""
LLM Testing Framework
====================

Test cases and utilities for testing LLM integrations.

This module provides specialized test cases for testing LLM interactions,
caching, token counting, cost tracking, and streaming responses.
"""

import unittest
from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager
import json
import time

from .mocks import MockChatSession
from .assertions import (
    assert_token_count,
    assert_cost_within_budget,
    assert_no_hallucination,
)


class LLMTestCase(unittest.TestCase):
    """
    Base test case for testing LLM integrations.

    Provides mocking for LLM responses, cache testing, token counting,
    cost tracking, and streaming response testing.

    Example:
    --------
    ```python
    class TestLLMIntegration(LLMTestCase):
        def test_with_mock_llm(self):
            with self.mock_llm_response("mocked response"):
                result = my_llm_function()
                self.assertEqual(result, "mocked response")

        def test_caching_works(self):
            result1 = my_llm_function()
            result2 = my_llm_function()
            self.assert_cache_hit(result2)
            self.assert_token_savings(result2)
    ```
    """

    def setUp(self):
        """Set up test fixtures and mock LLM."""
        self.mock_chat = MockChatSession()

        # Track LLM calls
        self.llm_calls = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_tokens = 0
        self.total_cost = 0.0

        # Set up test responses
        self.mock_responses = []
        self.response_index = 0

    def tearDown(self):
        """Clean up after tests."""
        self.mock_chat.reset()
        self.llm_calls.clear()

    @contextmanager
    def mock_llm_response(
        self,
        response: str,
        tokens: Optional[int] = None,
        cost: Optional[float] = None,
        cached: bool = False
    ):
        """
        Context manager for mocking a single LLM response.

        Args:
            response: The mocked response text
            tokens: Token count (auto-calculated if None)
            cost: Cost in dollars (auto-calculated if None)
            cached: Whether this is a cached response

        Example:
        --------
        ```python
        with self.mock_llm_response("Hello, world!", tokens=10, cost=0.001):
            result = my_llm_call()
            self.assertEqual(result, "Hello, world!")
        ```
        """
        # Auto-calculate tokens if not provided
        if tokens is None:
            tokens = len(response.split()) * 1.3  # Rough estimate

        # Auto-calculate cost if not provided
        if cost is None:
            cost = tokens * 0.00001  # Rough estimate

        # Create mock response
        mock_response = {
            'text': response,
            'tokens': tokens,
            'cost': cost,
            'cached': cached,
            'timestamp': time.time(),
        }

        # Patch ChatSession to return mock response
        original_send = self.mock_chat.send_message

        def mock_send(*args, **kwargs):
            self.llm_calls.append({
                'args': args,
                'kwargs': kwargs,
                'response': mock_response,
            })

            if cached:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

            self.total_tokens += tokens
            self.total_cost += cost

            return response

        self.mock_chat.send_message = mock_send

        try:
            yield mock_response
        finally:
            self.mock_chat.send_message = original_send

    @contextmanager
    def mock_llm_responses(self, responses: List[str]):
        """
        Context manager for mocking multiple LLM responses.

        Args:
            responses: List of response texts

        Example:
        --------
        ```python
        with self.mock_llm_responses(["response 1", "response 2"]):
            result1 = my_llm_call()
            result2 = my_llm_call()
        ```
        """
        self.mock_responses = responses
        self.response_index = 0

        original_send = self.mock_chat.send_message

        def mock_send(*args, **kwargs):
            if self.response_index >= len(self.mock_responses):
                raise IndexError("No more mock responses available")

            response = self.mock_responses[self.response_index]
            self.response_index += 1

            tokens = len(response.split()) * 1.3
            cost = tokens * 0.00001

            self.llm_calls.append({
                'args': args,
                'kwargs': kwargs,
                'response': {
                    'text': response,
                    'tokens': tokens,
                    'cost': cost,
                    'cached': False,
                },
            })

            self.cache_misses += 1
            self.total_tokens += tokens
            self.total_cost += cost

            return response

        self.mock_chat.send_message = mock_send

        try:
            yield
        finally:
            self.mock_chat.send_message = original_send

    @contextmanager
    def mock_streaming_response(self, chunks: List[str]):
        """
        Context manager for mocking streaming LLM responses.

        Args:
            chunks: List of response chunks

        Example:
        --------
        ```python
        with self.mock_streaming_response(["Hello", " ", "world"]):
            for chunk in my_streaming_call():
                print(chunk)
        ```
        """
        def chunk_generator():
            for chunk in chunks:
                yield chunk

        original_stream = self.mock_chat.stream_message
        self.mock_chat.stream_message = lambda *args, **kwargs: chunk_generator()

        try:
            yield
        finally:
            self.mock_chat.stream_message = original_stream

    def assert_llm_called(self, times: int = 1):
        """Assert that LLM was called a specific number of times."""
        self.assertEqual(
            len(self.llm_calls),
            times,
            f"Expected {times} LLM calls, got {len(self.llm_calls)}"
        )

    def assert_llm_called_with(self, message: str, **kwargs):
        """Assert that LLM was called with specific arguments."""
        for call in self.llm_calls:
            if message in str(call['args']) or message in str(call['kwargs']):
                return

        self.fail(f"LLM was not called with message: {message}")

    def assert_cache_hit(self, expected: bool = True):
        """Assert that the last LLM call was a cache hit."""
        if not self.llm_calls:
            self.fail("No LLM calls recorded")

        last_call = self.llm_calls[-1]
        cached = last_call['response'].get('cached', False)

        if expected:
            self.assertTrue(cached, "Expected cache hit but got cache miss")
        else:
            self.assertFalse(cached, "Expected cache miss but got cache hit")

    def assert_token_savings(self, min_savings: float = 0.5):
        """
        Assert that caching saved a minimum percentage of tokens.

        Args:
            min_savings: Minimum savings as a ratio (0.5 = 50%)
        """
        if self.cache_hits == 0:
            self.fail("No cache hits recorded")

        total_calls = self.cache_hits + self.cache_misses
        savings_ratio = self.cache_hits / total_calls

        self.assertGreaterEqual(
            savings_ratio,
            min_savings,
            f"Token savings {savings_ratio:.2%} below minimum {min_savings:.2%}"
        )

    def assert_total_tokens(self, max_tokens: int):
        """Assert that total tokens used is within limit."""
        self.assertLessEqual(
            self.total_tokens,
            max_tokens,
            f"Total tokens {self.total_tokens} exceeded limit {max_tokens}"
        )

    def assert_total_cost(self, max_cost: float):
        """Assert that total cost is within budget."""
        self.assertLessEqual(
            self.total_cost,
            max_cost,
            f"Total cost ${self.total_cost:.4f} exceeded budget ${max_cost:.4f}"
        )

    def assert_response_format(self, response: str, format_type: str):
        """
        Assert that response matches expected format.

        Args:
            response: The LLM response
            format_type: Expected format ('json', 'yaml', 'markdown', 'xml')
        """
        if format_type == 'json':
            try:
                json.loads(response)
            except json.JSONDecodeError:
                self.fail("Response is not valid JSON")

        elif format_type == 'yaml':
            try:
                import yaml
                yaml.safe_load(response)
            except yaml.YAMLError:
                self.fail("Response is not valid YAML")

        elif format_type == 'markdown':
            # Basic markdown validation
            self.assertTrue(
                any(marker in response for marker in ['#', '**', '-', '*']),
                "Response does not appear to be markdown"
            )

        elif format_type == 'xml':
            try:
                import xml.etree.ElementTree as ET
                ET.fromstring(response)
            except ET.ParseError:
                self.fail("Response is not valid XML")

    def get_llm_metrics(self) -> Dict[str, Any]:
        """Get aggregated LLM metrics from test execution."""
        return {
            'total_calls': len(self.llm_calls),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / max(len(self.llm_calls), 1),
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'avg_tokens_per_call': self.total_tokens / max(len(self.llm_calls), 1),
            'avg_cost_per_call': self.total_cost / max(len(self.llm_calls), 1),
        }
