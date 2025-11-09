"""
Agent Test Template
===================

Template for testing GreenLang agents.

Copy this template and customize for your specific agent.
"""

from greenlang.testing import AgentTestCase
# Import your agent here
# from your_module import YourAgent


class TestYourAgent(AgentTestCase):
    """Test suite for YourAgent."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Initialize your agent
        # self.agent = YourAgent()

        # Load test data
        self.test_input = {
            "field1": "value1",
            "field2": "value2",
        }

        self.expected_output = {
            "result": "expected_result",
        }

    def test_basic_execution(self):
        """Test basic agent execution."""
        # Run the agent
        result = self.run_agent(self.agent, self.test_input)

        # Assert success
        self.assert_success(result)

        # Verify output
        self.assertIsNotNone(result['result'])

    def test_output_schema(self):
        """Test that agent output matches expected schema."""
        # Define expected schema
        schema = {
            "type": "object",
            "properties": {
                "result": {"type": "string"}
            },
            "required": ["result"]
        }

        # Run and validate
        result = self.run_agent(self.agent, self.test_input)
        self.assert_output_schema(result, schema)

    def test_performance(self):
        """Test agent performance."""
        result = self.run_agent(self.agent, self.test_input)

        # Assert performance within bounds
        self.assert_performance(
            result,
            max_time=2.0,  # 2 seconds max
            max_memory=100 * 1024 * 1024  # 100 MB max
        )

    def test_with_mock_infrastructure(self):
        """Test agent with mocked infrastructure."""
        # Set up mock responses
        self.mock_chat.add_response("Mocked LLM response")

        # Run with mocked infrastructure
        with self.mock_infrastructure():
            result = self.run_agent(self.agent, self.test_input)
            self.assert_success(result)

    def test_batch_processing(self):
        """Test agent with batch input."""
        batch_input = [
            {"field1": "value1"},
            {"field1": "value2"},
            {"field1": "value3"},
        ]

        results = self.run_agent_batch(self.agent, batch_input)

        # Assert all succeeded
        for result in results:
            self.assert_success(result)

    def test_error_handling(self):
        """Test agent error handling."""
        invalid_input = {}

        with self.assertRaises(ValueError):
            self.run_agent(self.agent, invalid_input)

    def test_determinism(self):
        """Test that agent produces consistent results."""
        self.assert_deterministic(
            self.agent,
            self.test_input,
            runs=3
        )

    def test_with_fixture_data(self):
        """Test agent with fixture data."""
        # Load fixture
        fixture_data = self.load_fixture('sample_emissions_data.json')

        result = self.run_agent(self.agent, fixture_data)
        self.assert_success(result)


if __name__ == '__main__':
    import unittest
    unittest.main()
