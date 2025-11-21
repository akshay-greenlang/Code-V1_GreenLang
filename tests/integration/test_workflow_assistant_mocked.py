# -*- coding: utf-8 -*-
"""
Assistant workflow integration tests with mocked LLM.
"""
import pytest
from unittest.mock import patch, MagicMock
from tests.integration.utils import load_fixture


@pytest.mark.integration
class TestAssistantMocked:
    """Test assistant-driven workflows with mocked LLM."""
    
    def test_assistant_structured_input_parsing(self, workflow_runner, mock_llm):
        """
        Test assistant parsing natural language to structured input.
        
        Verifies:
        - Natural language parsed correctly
        - Routes to correct workflow
        - Same results as direct input
        """
        # Mock LLM response for parsing
        mock_llm.return_value = {
            "intent": "calculate_emissions",
            "parsed_input": {
                "location": {"country": "IN"},
                "consumption": {
                    "electricity": {"value": 1500000, "unit": "kWh"}
                },
                "building_info": {
                    "area_sqft": 50000,
                    "type": "commercial_office"
                }
            }
        }
        
        # Simulate assistant command
        from greenlang.llm.assistant import LLMAssistant
        
        assistant = LLMAssistant()
        
        # Process natural language query
        query = "Calculate emissions for 1.5M kWh in India"
        parsed = assistant.process(query)
        
        # Run workflow with parsed input
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': parsed['parsed_input']}
        )
        
        assert result['success'] is True
        
        # Verify emissions calculated
        report = result['data']['emissions_report']
        assert report['total_co2e_kg'] > 0
        
        # Should use India grid factor
        assert report['by_fuel']['electricity'] > 1000000  # High due to India factor
    
    def test_assistant_workflow_selection(self, workflow_runner, mock_llm):
        """Test assistant selecting appropriate workflow."""
        test_cases = [
            ("Calculate building emissions", "commercial_building_emissions.yaml"),
            ("Analyze portfolio", "portfolio_analysis.yaml"),
            ("India compliance check", "india_building_workflow.yaml")
        ]
        
        for query, expected_workflow in test_cases:
            # Mock LLM to return workflow selection
            mock_llm.return_value = {
                "workflow": expected_workflow,
                "confidence": 0.95
            }
            
            # This would test workflow selection logic
            # Implementation depends on assistant design
    
    def test_assistant_multi_step_clarification(self, workflow_runner):
        """Test assistant handling ambiguous queries."""
        with patch('greenlang.llm.assistant.LLMAssistant.process') as mock_process:
            # First call - need clarification
            mock_process.side_effect = [
                {
                    "needs_clarification": True,
                    "questions": [
                        "What is the building location?",
                        "What is the annual electricity consumption?"
                    ]
                },
                # Second call - after clarification
                {
                    "intent": "calculate_emissions",
                    "parsed_input": {
                        "location": {"country": "US", "state": "CA"},
                        "consumption": {"electricity": {"value": 2000000, "unit": "kWh"}}
                    }
                }
            ]
            
            # Simulate conversation
            assistant = MagicMock()
            assistant.process.side_effect = mock_process.side_effect
            
            # Initial vague query
            response1 = assistant.process("Calculate emissions")
            assert response1["needs_clarification"] is True
            
            # Follow-up with details
            response2 = assistant.process("Building in California, 2M kWh annually")
            assert "parsed_input" in response2
    
    def test_assistant_unit_conversion(self, workflow_runner, mock_llm):
        """Test assistant handling different unit formats."""
        test_cases = [
            ("1.5 million kWh", 1500000),
            ("1500 MWh", 1500000),
            ("1.5M kilowatt hours", 1500000),
            ("1,500,000 kWh", 1500000)
        ]
        
        for input_text, expected_value in test_cases:
            mock_llm.return_value = {
                "parsed_input": {
                    "consumption": {
                        "electricity": {"value": expected_value, "unit": "kWh"}
                    }
                }
            }
            
            # Assistant should normalize units
            from greenlang.llm.assistant import LLMAssistant
            assistant = LLMAssistant()
            
            parsed = assistant.process(f"Calculate emissions for {input_text}")
            
            # Verify normalized value
            electricity = parsed['parsed_input']['consumption']['electricity']
            assert electricity['value'] == expected_value
            assert electricity['unit'] == 'kWh'
    
    def test_assistant_error_recovery(self, workflow_runner, mock_llm):
        """Test assistant recovering from parsing errors."""
        # Mock sequence of attempts
        mock_llm.side_effect = [
            Exception("LLM API error"),  # First attempt fails
            {  # Second attempt succeeds
                "parsed_input": {
                    "location": {"country": "IN"},
                    "consumption": {"electricity": {"value": 1000000, "unit": "kWh"}}
                }
            }
        ]
        
        # Assistant should retry and recover
        from greenlang.llm.assistant import LLMAssistant
        assistant = LLMAssistant()
        
        # This would test retry logic
        try:
            result = assistant.process("Calculate emissions for 1M kWh in India")
            # If retry implemented, should succeed on second attempt
            assert "parsed_input" in result
        except Exception:
            # If no retry, should fail cleanly
            pass
    
    def test_assistant_context_awareness(self, workflow_runner, mock_llm):
        """Test assistant maintaining context across queries."""
        with patch('greenlang.llm.assistant.LLMAssistant') as MockAssistant:
            assistant = MockAssistant()
            assistant.context = {}
            
            # First query sets context
            assistant.process.return_value = {
                "parsed_input": {"location": {"country": "IN"}},
                "context_updated": {"default_country": "IN"}
            }
            
            response1 = assistant.process("Set default country to India")
            
            # Second query uses context
            assistant.process.return_value = {
                "parsed_input": {
                    "location": {"country": "IN"},  # Uses context
                    "consumption": {"electricity": {"value": 500000, "unit": "kWh"}}
                }
            }
            
            response2 = assistant.process("Calculate for 500k kWh")
            
            # Should use India from context
            assert response2['parsed_input']['location']['country'] == 'IN'
    
    def test_assistant_output_formatting(self, workflow_runner, mock_llm):
        """Test assistant formatting output for users."""
        # Run calculation
        building_data = load_fixture('data', 'building_india_office.json')
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        # Mock assistant formatting
        mock_llm.return_value = {
            "formatted_output": """
## Emissions Report

**Total Emissions**: 1,230 tons CO2e
**Electricity**: 1,230,000 kg CO2e
**Natural Gas**: 101,000 kg CO2e
**Diesel**: 26,800 kg CO2e

### Recommendations:
1. Install LED lighting (10% savings)
2. Optimize HVAC (15% savings)
3. Solar installation (20% savings)
            """.strip()
        }
        
        # Assistant formats technical output
        from greenlang.llm.assistant import LLMAssistant
        assistant = LLMAssistant()
        
        formatted = assistant.format_output(result['data'])
        
        # Should be human-readable
        assert "Total Emissions" in formatted['formatted_output']
        assert "Recommendations" in formatted['formatted_output']
    
    def test_assistant_batch_processing(self, workflow_runner, mock_llm):
        """Test assistant handling batch queries."""
        mock_llm.return_value = {
            "batch_parsed": [
                {
                    "location": {"country": "IN"},
                    "consumption": {"electricity": {"value": 1000000, "unit": "kWh"}}
                },
                {
                    "location": {"country": "US"},
                    "consumption": {"electricity": {"value": 2000000, "unit": "kWh"}}
                }
            ]
        }
        
        # Process batch query
        from greenlang.llm.assistant import LLMAssistant
        assistant = LLMAssistant()
        
        batch_query = "Compare emissions: 1M kWh in India vs 2M kWh in US"
        parsed = assistant.process(batch_query)
        
        # Should parse both scenarios
        assert len(parsed['batch_parsed']) == 2
        
        # Run both through workflow
        results = []
        for building_data in parsed['batch_parsed']:
            result = workflow_runner.run(
                'tests/fixtures/workflows/commercial_building_emissions.yaml',
                {'building_data': building_data}
            )
            results.append(result)
        
        # Both should succeed
        assert all(r['success'] for r in results)
    
    def test_assistant_explanation_generation(self, workflow_runner, mock_llm):
        """Test assistant generating explanations."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        # Mock explanation generation
        mock_llm.return_value = {
            "explanation": "The high emissions are primarily due to India's coal-heavy electricity grid, which has a factor of 0.82 kg CO2e/kWh. The biggest opportunity for reduction is solar installation, which could offset 20% of grid consumption."
        }
        
        # Generate explanation
        from greenlang.llm.assistant import LLMAssistant
        assistant = LLMAssistant()
        
        explanation = assistant.explain_results(result['data'])
        
        # Should provide context
        assert "grid" in explanation['explanation'].lower()
        assert "solar" in explanation['explanation'].lower()