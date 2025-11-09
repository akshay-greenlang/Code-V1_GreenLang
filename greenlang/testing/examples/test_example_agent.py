"""
Example Agent Test
==================

Demonstrates how to test a GreenLang agent.
"""

from greenlang.testing import AgentTestCase


# Example agent for demonstration
class EmissionsCalculatorAgent:
    """Simple emissions calculator agent."""

    def run(self, input_data):
        """Calculate emissions."""
        quantity = input_data.get('quantity', 0)
        emission_factor = input_data.get('emission_factor', 2.5)

        total_emissions = quantity * emission_factor

        return {
            'total_emissions': total_emissions,
            'quantity': quantity,
            'emission_factor': emission_factor,
            'unit': 'kg CO2e',
            'methodology': 'GHG Protocol',
        }


class TestEmissionsCalculatorAgent(AgentTestCase):
    """Test suite for EmissionsCalculatorAgent."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        self.agent = EmissionsCalculatorAgent()

        self.sample_input = {
            'quantity': 1000,
            'emission_factor': 2.5,
        }

    def test_basic_calculation(self):
        """Test basic emissions calculation."""
        result = self.run_agent(self.agent, self.sample_input)

        # Assert success
        self.assert_success(result)

        # Verify output
        self.assertEqual(result['result']['total_emissions'], 2500)
        self.assertEqual(result['result']['unit'], 'kg CO2e')

    def test_output_schema(self):
        """Test that output matches expected schema."""
        schema = {
            "type": "object",
            "properties": {
                "total_emissions": {"type": "number"},
                "quantity": {"type": "number"},
                "emission_factor": {"type": "number"},
                "unit": {"type": "string"},
                "methodology": {"type": "string"}
            },
            "required": ["total_emissions", "unit"]
        }

        result = self.run_agent(self.agent, self.sample_input)
        self.assert_output_schema(result, schema)

    def test_performance(self):
        """Test that agent performs within acceptable bounds."""
        result = self.run_agent(self.agent, self.sample_input)

        # Should complete in under 1 second
        self.assert_performance(result, max_time=1.0)

    def test_deterministic_results(self):
        """Test that agent produces consistent results."""
        self.assert_deterministic(
            self.agent,
            self.sample_input,
            runs=5
        )

    def test_batch_processing(self):
        """Test agent with batch input."""
        batch_input = [
            {'quantity': 100, 'emission_factor': 2.5},
            {'quantity': 500, 'emission_factor': 2.5},
            {'quantity': 1000, 'emission_factor': 2.5},
        ]

        results = self.run_agent_batch(self.agent, batch_input)

        # All should succeed
        for result in results:
            self.assert_success(result)

        # Verify calculations
        self.assertEqual(results[0]['result']['total_emissions'], 250)
        self.assertEqual(results[1]['result']['total_emissions'], 1250)
        self.assertEqual(results[2]['result']['total_emissions'], 2500)

    def test_zero_quantity(self):
        """Test agent with zero quantity."""
        zero_input = {'quantity': 0, 'emission_factor': 2.5}

        result = self.run_agent(self.agent, zero_input)

        self.assert_success(result)
        self.assertEqual(result['result']['total_emissions'], 0)

    def test_with_fixture_data(self):
        """Test agent with fixture data."""
        fixture = self.load_fixture('sample_emissions_data.json')

        # Use first emission record
        emission = fixture['emissions_data'][0]

        input_data = {
            'quantity': emission['quantity'],
            'emission_factor': emission['emission_factor'],
        }

        result = self.run_agent(self.agent, input_data)

        self.assert_success(result)
        self.assertEqual(
            result['result']['total_emissions'],
            emission['total_emissions']
        )


if __name__ == '__main__':
    import unittest
    unittest.main()
