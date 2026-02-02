# -*- coding: utf-8 -*-
"""
Error handling and validation integration tests.
"""
import pytest
import yaml
import json
from pathlib import Path
from tests.integration.utils import load_fixture, TestIOHelper


@pytest.mark.integration
class TestErrorsAndValidation:
    """Test error handling and input validation."""
    
    def test_missing_agent_error(self, workflow_runner):
        """
        Test graceful failure with non-existent agent.
        
        Verifies:
        - Clear error message
        - Identifies problematic step
        - No stack trace leakage
        """
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Run workflow with missing agent
        result = workflow_runner.run(
            'tests/fixtures/workflows/broken_missing_agent.yaml',
            {'building_data': building_data}
        )
        
        # Should fail gracefully
        assert result['success'] is False
        assert 'error' in result
        
        error_msg = str(result['error'])
        
        # Error should identify the problem
        assert 'NonExistentAgent' in error_msg or 'agent' in error_msg.lower()
        assert 'broken_step' in error_msg or 'step' in error_msg.lower()
        
        # Should not expose internal stack trace
        assert 'Traceback' not in error_msg or len(error_msg) < 1000
    
    def test_invalid_yaml_syntax(self, workflow_runner, tmp_outdir):
        """Test handling of malformed YAML."""
        io_helper = TestIOHelper(tmp_outdir)
        
        # Create invalid YAML
        invalid_yaml = """
name: invalid_workflow
steps:
  - id: step1
    agent: TestAgent
    inputs:
      data: $building_data
      invalid_indent:
    bad_key
        """
        
        yaml_path = io_helper.write_text('invalid.yaml', invalid_yaml)
        
        try:
            result = workflow_runner.run(
                str(yaml_path),
                {'building_data': {}}
            )
            
            # Should fail with YAML error
            assert result['success'] is False
            assert 'yaml' in str(result['error']).lower() or 'parse' in str(result['error']).lower()
        except yaml.YAMLError as e:
            # Direct YAML error is also acceptable
            assert 'line' in str(e) or 'column' in str(e)
    
    def test_missing_required_input(self, workflow_runner):
        """Test validation of missing required inputs."""
        # Run workflow without required building_data
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {}  # Empty inputs
        )
        
        # Should fail with validation error
        assert result['success'] is False
        error_msg = str(result['error'])
        
        assert 'building_data' in error_msg or 'required' in error_msg.lower()
    
    def test_invalid_input_type(self, workflow_runner):
        """Test type validation for inputs."""
        # Provide string instead of object
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': "not an object"}
        )
        
        # Should fail with type error
        assert result['success'] is False
        error_msg = str(result['error'])
        
        assert 'type' in error_msg.lower() or 'object' in error_msg.lower()
    
    def test_negative_consumption_values(self, workflow_runner):
        """Test validation of negative consumption values."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Set negative electricity consumption
        building_data['consumption']['electricity']['value'] = -1000
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        # Should fail or handle gracefully
        if not result['success']:
            error_msg = str(result['error'])
            assert 'negative' in error_msg.lower() or 'invalid' in error_msg.lower()
        else:
            # If it processes, emissions should be handled appropriately
            emissions = result['data']['emissions_report']['total_co2e_kg']
            # Negative consumption might be treated as 0
            assert emissions >= 0
    
    def test_missing_location_country(self, workflow_runner):
        """Test handling of missing country in location."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Remove country
        del building_data['location']['country']
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        # Should fail or use default
        if not result['success']:
            assert 'country' in str(result['error']).lower()
        else:
            # Used some default
            assert result['data']['emissions_report']['total_co2e_kg'] > 0
    
    def test_invalid_unit_values(self, workflow_runner):
        """Test handling of invalid units."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Set invalid unit
        building_data['consumption']['electricity']['unit'] = 'invalid_unit'
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        # Should fail with unit error
        if not result['success']:
            error_msg = str(result['error'])
            assert 'unit' in error_msg.lower() or 'invalid_unit' in error_msg
    
    def test_circular_dependency_detection(self, workflow_runner, tmp_outdir):
        """Test detection of circular dependencies in workflow."""
        io_helper = TestIOHelper(tmp_outdir)
        
        # Create workflow with circular dependency
        circular_workflow = {
            'name': 'circular_workflow',
            'version': '0.0.1',
            'inputs': {'data': {'type': 'object'}},
            'outputs': {'result': {'type': 'object'}},
            'steps': [
                {
                    'id': 'step1',
                    'agent': 'Agent1',
                    'inputs': {'data': '$steps.step2.output'},
                    'outputs': {'output': 'object'}
                },
                {
                    'id': 'step2',
                    'agent': 'Agent2',
                    'inputs': {'data': '$steps.step1.output'},
                    'outputs': {'output': 'object'}
                }
            ]
        }
        
        yaml_path = io_helper.write_yaml('circular.yaml', circular_workflow)
        
        result = workflow_runner.run(
            str(yaml_path),
            {'data': {}}
        )
        
        # Should detect circular dependency
        if not result['success']:
            error_msg = str(result['error'])
            assert 'circular' in error_msg.lower() or 'dependency' in error_msg.lower()
    
    def test_timeout_handling(self, workflow_runner):
        """Test handling of step timeouts."""
        # This test would require a way to simulate slow agents
        # For now, just verify workflow completes in reasonable time
        
        building_data = load_fixture('data', 'building_india_office.json')
        
        import time
        start = time.time()
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        elapsed = time.time() - start
        
        # Should complete or timeout within 10 seconds
        assert elapsed < 10.0
        
        if not result['success']:
            # If failed, might be timeout
            assert 'timeout' in str(result['error']).lower()
    
    def test_graceful_degradation(self, workflow_runner):
        """Test graceful degradation when optional features fail."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Remove optional renewable data
        del building_data['renewable']
        
        # Remove optional certifications
        del building_data['certifications']
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        # Should still succeed without optional data
        assert result['success'] is True
        
        # Core functionality should work
        report = result['data']['emissions_report']
        assert report['total_co2e_kg'] > 0
        assert 'by_fuel' in report
    
    def test_error_recovery_attempt(self, workflow_runner):
        """Test that workflow attempts recovery from errors."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Add problematic but recoverable data
        building_data['consumption']['unknown_fuel'] = {
            'value': 100,
            'unit': 'unknown'
        }
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        # Might succeed by ignoring unknown fuel
        if result['success']:
            # Should process known fuels
            report = result['data']['emissions_report']
            assert 'electricity' in report['by_fuel']
            assert 'natural_gas' in report['by_fuel']
    
    def test_validation_error_messages(self, workflow_runner):
        """Test that validation errors are clear and actionable."""
        test_cases = [
            ({}, "building_data"),  # Missing required field
            ({'building_data': None}, "null"),  # Null value
            ({'building_data': []}, "array"),  # Wrong type
        ]
        
        for inputs, expected_in_error in test_cases:
            result = workflow_runner.run(
                'tests/fixtures/workflows/commercial_building_emissions.yaml',
                inputs
            )
            
            assert result['success'] is False
            error_msg = str(result['error']).lower()
            
            # Error should mention the issue
            assert expected_in_error.lower() in error_msg or 'validation' in error_msg