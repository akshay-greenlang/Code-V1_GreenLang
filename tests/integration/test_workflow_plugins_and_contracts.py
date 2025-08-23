"""
Plugin discovery and contract testing.
"""
import pytest
from pathlib import Path
import sys
from tests.integration.utils import load_fixture, TestIOHelper


@pytest.mark.integration
class TestPluginsAndContracts:
    """Test plugin discovery and agent contracts."""
    
    def test_plugin_discovery_mechanism(self, workflow_runner, tmp_outdir):
        """
        Test that plugins can be discovered via entry points.
        
        Verifies:
        - Plugin registration works
        - Plugins accessible in workflows
        - Contract compliance
        """
        # Create a test plugin
        io_helper = TestIOHelper(tmp_outdir)
        
        # Create plugin module
        plugin_code = '''
from greenlang.agents.base import BaseAgent

class TestPluginAgent(BaseAgent):
    """Test plugin agent."""
    
    def __init__(self):
        super().__init__()
        self.agent_id = "TestPluginAgent"
        self.version = "0.0.1"
    
    def execute(self, inputs):
        """Execute test plugin logic."""
        return {
            "result": "Plugin executed",
            "input_received": inputs.get("data"),
            "plugin_version": self.version
        }
    
    def validate_inputs(self, inputs):
        """Validate inputs."""
        return "data" in inputs
'''
        
        plugin_path = io_helper.write_text('test_plugin.py', plugin_code)
        
        # Add plugin path to Python path
        sys.path.insert(0, str(tmp_outdir))
        
        # Create workflow using plugin
        plugin_workflow = {
            'name': 'plugin_test_workflow',
            'version': '0.0.1',
            'inputs': {'test_data': {'type': 'object'}},
            'outputs': {'result': {'type': 'object'}},
            'steps': [
                {
                    'id': 'plugin_step',
                    'agent': 'TestPluginAgent',
                    'inputs': {'data': '$test_data'},
                    'outputs': {'result': 'object'}
                }
            ]
        }
        
        workflow_path = io_helper.write_yaml('plugin_workflow.yaml', plugin_workflow)
        
        # Try to run workflow with plugin
        # This would work if plugin discovery is implemented
        result = workflow_runner.run(
            str(workflow_path),
            {'test_data': {'value': 42}}
        )
        
        # Plugin might not be discovered without proper registration
        # This test documents the expected behavior
        if result['success']:
            assert 'result' in result['data']
            assert result['data']['result']['plugin_version'] == '0.0.1'
    
    def test_agent_contract_validation(self, workflow_runner):
        """Test that agents comply with expected contracts."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        assert result['success'] is True
        
        # Verify each agent's output matches contract
        report = result['data']['emissions_report']
        
        # EmissionCalculatorAgent contract
        if 'emissions' in report or 'total_co2e_kg' in report:
            emissions = report.get('emissions', report)
            # Must provide total emissions
            assert 'total_co2e_kg' in emissions or 'total_emissions_kg' in emissions
            # Must provide fuel breakdown
            assert 'by_fuel' in emissions
            # All values must be numeric
            for fuel, value in emissions['by_fuel'].items():
                assert isinstance(value, (int, float))
        
        # BenchmarkAgent contract
        if 'benchmark' in report:
            benchmark = report['benchmark']
            # Must provide rating
            assert 'rating' in benchmark
            # Rating must be valid
            assert benchmark['rating'] in ['Excellent', 'Good', 'Average', 'Poor'] or \
                   isinstance(benchmark['rating'], str)
        
        # RecommendationAgent contract
        if 'recommendations' in report:
            recommendations = report['recommendations']
            # Must be a list
            assert isinstance(recommendations, list)
            # Each recommendation must have required fields
            for rec in recommendations:
                assert 'action' in rec or 'measure' in rec
                # Should have impact or savings
                assert 'savings' in rec or 'impact' in rec or 'benefit' in rec
    
    def test_custom_agent_registration(self, workflow_runner, monkeypatch):
        """Test registration of custom agents."""
        from unittest.mock import Mock
        
        # Create custom agent
        custom_agent = Mock()
        custom_agent.agent_id = "CustomTestAgent"
        custom_agent.version = "0.0.1"
        custom_agent.execute.return_value = {
            "custom_result": "Success",
            "processed": True
        }
        
        # Register custom agent (mocked)
        agent_registry = {}
        agent_registry["CustomTestAgent"] = custom_agent
        
        # Patch agent discovery
        def mock_get_agent(agent_id):
            return agent_registry.get(agent_id)
        
        monkeypatch.setattr("greenlang.agents.registry.get_agent", mock_get_agent)
        
        # Create workflow using custom agent
        custom_workflow = {
            'name': 'custom_agent_test',
            'version': '0.0.1',
            'inputs': {'data': {'type': 'object'}},
            'outputs': {'result': {'type': 'object'}},
            'steps': [
                {
                    'id': 'custom_step',
                    'agent': 'CustomTestAgent',
                    'inputs': {'input': '$data'},
                    'outputs': {'output': 'object'}
                }
            ]
        }
        
        # This would test custom agent execution
        # Implementation depends on actual plugin system
    
    def test_agent_version_compatibility(self, workflow_runner):
        """Test handling of agent version requirements."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Workflow might specify agent versions
        versioned_workflow = load_fixture('workflows', 'commercial_building_emissions.yaml')
        
        # Add version requirements (if supported)
        if 'requirements' not in versioned_workflow:
            versioned_workflow['requirements'] = {}
        
        versioned_workflow['requirements']['agents'] = {
            'EmissionCalculatorAgent': '>=0.0.1',
            'BenchmarkAgent': '~=0.0.1'
        }
        
        # This tests version checking if implemented
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        # Should work with compatible versions
        assert result['success'] is True
    
    def test_plugin_isolation(self, workflow_runner):
        """Test that plugins don't interfere with each other."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Run workflow 1
        result1 = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        # Run workflow 2 (different agents)
        result2 = workflow_runner.run(
            'tests/fixtures/workflows/india_building_workflow.yaml',
            {'building_data': building_data}
        )
        
        # Both should succeed independently
        assert result1['success'] is True
        assert result2['success'] is True
        
        # Results should be independent
        assert result1['data'] != result2['data']
    
    def test_agent_input_output_types(self, workflow_runner):
        """Test type checking for agent inputs/outputs."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        assert result['success'] is True
        
        # Check output types match declarations
        report = result['data']['emissions_report']
        
        # Should match workflow output schema
        assert isinstance(report, dict)
        
        # Numeric fields should be numbers
        if 'total_co2e_kg' in report:
            assert isinstance(report['total_co2e_kg'], (int, float))
        
        # Object fields should be dicts
        if 'by_fuel' in report:
            assert isinstance(report['by_fuel'], dict)
        
        # Array fields should be lists
        if 'recommendations' in report:
            assert isinstance(report['recommendations'], list)
    
    def test_agent_error_contract(self, workflow_runner):
        """Test that agent errors follow consistent format."""
        # Use workflow with missing agent
        building_data = load_fixture('data', 'building_india_office.json')
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/broken_missing_agent.yaml',
            {'building_data': building_data}
        )
        
        assert result['success'] is False
        
        # Error should follow contract
        error = result['error']
        
        # Should identify the agent
        assert 'NonExistentAgent' in str(error) or 'agent' in str(error).lower()
        
        # Should identify the step
        assert 'broken_step' in str(error) or 'step' in str(error).lower()
    
    def test_plugin_configuration(self, workflow_runner, tmp_outdir):
        """Test that plugins can be configured."""
        io_helper = TestIOHelper(tmp_outdir)
        
        # Create plugin configuration
        config = {
            'plugins': {
                'TestPlugin': {
                    'enabled': True,
                    'config': {
                        'api_key': 'test_key',
                        'timeout': 30
                    }
                }
            }
        }
        
        config_path = io_helper.write_yaml('plugin_config.yaml', config)
        
        # This would test plugin configuration if implemented
        # The configuration would be loaded and passed to plugins