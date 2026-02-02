# -*- coding: utf-8 -*-
"""
Example LLM Test
================

Demonstrates how to test LLM integrations.
"""

from greenlang.testing import LLMTestCase


# Example LLM function for demonstration
def analyze_emissions_with_llm(chat_session, data):
    """Analyze emissions data using LLM."""
    prompt = f"""
    Analyze the following emissions data:
    {data}

    Provide:
    1. Total emissions
    2. Main contributors
    3. Recommendations for reduction
    """

    response = chat_session.send_message(prompt)
    return response


class TestLLMIntegration(LLMTestCase):
    """Test suite for LLM integration."""

    def test_basic_llm_call(self):
        """Test basic LLM function call."""
        test_data = {
            'scope_1': 10000,
            'scope_2': 4000,
            'scope_3': 2860,
        }

        expected_response = """
        Total emissions: 16,860 kg CO2e

        Main contributors:
        1. Scope 1: 10,000 kg CO2e (59%)
        2. Scope 2: 4,000 kg CO2e (24%)
        3. Scope 3: 2,860 kg CO2e (17%)

        Recommendations:
        - Transition to renewable energy
        - Improve energy efficiency
        - Engage suppliers on emissions reduction
        """

        with self.mock_llm_response(expected_response, tokens=200, cost=0.002):
            result = analyze_emissions_with_llm(self.mock_chat, test_data)
            self.assertIn("16,860", result)
            self.assertIn("renewable energy", result.lower())

    def test_llm_called_with_correct_prompt(self):
        """Test that LLM is called with correct prompt."""
        test_data = {'scope_1': 10000}

        with self.mock_llm_response("Analysis complete"):
            result = analyze_emissions_with_llm(self.mock_chat, test_data)
            self.assert_llm_called(times=1)
            self.assert_llm_called_with("emissions data")

    def test_token_counting(self):
        """Test token usage tracking."""
        test_data = {'scope_1': 10000}

        with self.mock_llm_response("Response", tokens=500):
            result = analyze_emissions_with_llm(self.mock_chat, test_data)

        # Assert tokens within budget
        self.assert_total_tokens(max_tokens=1000)

    def test_cost_tracking(self):
        """Test cost tracking."""
        test_data = {'scope_1': 10000}

        with self.mock_llm_response("Response", tokens=1000, cost=0.01):
            result = analyze_emissions_with_llm(self.mock_chat, test_data)

        # Assert cost within budget
        self.assert_total_cost(max_cost=0.02)

    def test_multiple_llm_calls(self):
        """Test multiple LLM calls."""
        responses = [
            "First analysis",
            "Second analysis",
            "Third analysis"
        ]

        with self.mock_llm_responses(responses):
            r1 = self.mock_chat.send_message("Prompt 1")
            r2 = self.mock_chat.send_message("Prompt 2")
            r3 = self.mock_chat.send_message("Prompt 3")

            self.assertEqual(r1, "First analysis")
            self.assertEqual(r2, "Second analysis")
            self.assertEqual(r3, "Third analysis")

        # Assert correct number of calls
        self.assert_llm_called(times=3)

    def test_llm_metrics(self):
        """Test LLM metrics collection."""
        with self.mock_llm_response("Response", tokens=100, cost=0.001):
            self.mock_chat.send_message("Test")

        with self.mock_llm_response("Response", tokens=200, cost=0.002, cached=True):
            self.mock_chat.send_message("Test")

        metrics = self.get_llm_metrics()

        self.assertEqual(metrics['total_calls'], 2)
        self.assertEqual(metrics['total_tokens'], 300)
        self.assertEqual(metrics['total_cost'], 0.003)
        self.assertEqual(metrics['cache_hits'], 1)
        self.assertEqual(metrics['cache_misses'], 1)
        self.assertEqual(metrics['cache_hit_rate'], 0.5)


if __name__ == '__main__':
    import unittest
    unittest.main()
