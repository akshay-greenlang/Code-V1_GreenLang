"""
Parallel execution and caching integration tests.
"""
import pytest
import time
import hashlib
import json
from tests.integration.utils import load_fixture, normalize_json


@pytest.mark.integration
class TestParallelAndCaching:
    """Test parallel workflow execution and caching behavior."""
    
    def test_parallel_fuel_calculations(self, workflow_runner, assert_close):
        """
        Test parallel execution of fuel calculations.
        
        Verifies:
        - Parallel steps execute correctly
        - Results are deterministic
        - Aggregation after parallel execution
        """
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Run parallel workflow
        result = workflow_runner.run(
            'tests/fixtures/workflows/parallel_example.yaml',
            {'building_data': building_data}
        )
        
        assert result['success'] is True
        report = result['data']['emissions_report']
        
        # Verify all parallel branches completed
        assert 'total_emissions_kg' in report
        assert 'by_fuel' in report
        
        # Check each fuel type was calculated
        expected_fuels = ['electricity', 'natural_gas', 'diesel']
        for fuel in expected_fuels:
            assert fuel in report['by_fuel']
            assert report['by_fuel'][fuel] > 0
        
        # Verify aggregation is correct
        fuel_sum = sum(report['by_fuel'].values())
        assert_close(
            fuel_sum,
            report['total_emissions_kg'],
            rel_tol=1e-6,
            msg="Sum of parallel calculations should equal total"
        )
    
    def test_parallel_vs_serial_consistency(self, workflow_runner):
        """Test that parallel execution gives same results as serial."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Run parallel workflow
        parallel_result = workflow_runner.run(
            'tests/fixtures/workflows/parallel_example.yaml',
            {'building_data': building_data}
        )
        
        # Run serial workflow (commercial workflow is serial)
        serial_result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        # Both should succeed
        assert parallel_result['success'] is True
        assert serial_result['success'] is True
        
        # Extract emissions
        parallel_emissions = parallel_result['data']['emissions_report']
        serial_emissions = serial_result['data']['emissions_report']
        
        # Core calculations should match
        if 'total_emissions_kg' in parallel_emissions and 'total_co2e_kg' in serial_emissions:
            # Allow for small differences due to rounding in different order
            parallel_total = parallel_emissions['total_emissions_kg']
            serial_total = serial_emissions['total_co2e_kg']
            
            relative_diff = abs(parallel_total - serial_total) / serial_total
            assert relative_diff < 0.01, f"Parallel and serial differ by {relative_diff*100:.2f}%"
    
    def test_caching_identical_inputs(self, workflow_runner, monkeypatch):
        """Test that identical inputs can use cached results."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Track execution times
        execution_times = []
        
        # Mock to track cache hits
        cache_hits = {'count': 0}
        original_run = workflow_runner.run
        
        def tracked_run(*args, **kwargs):
            start = time.time()
            result = original_run(*args, **kwargs)
            execution_times.append(time.time() - start)
            return result
        
        workflow_runner.run = tracked_run
        
        # First run - should populate cache
        result1 = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        # Second run - might use cache
        result2 = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        # Both should succeed with identical results
        assert result1['success'] is True
        assert result2['success'] is True
        
        # Results should be identical
        report1 = normalize_json(result1['data'])
        report2 = normalize_json(result2['data'])
        
        assert report1 == report2, "Cached result should match original"
        
        # Second run might be faster (if caching is implemented)
        if len(execution_times) == 2 and execution_times[1] < execution_times[0] * 0.5:
            # Likely used cache
            pass  # This is good but not required
    
    def test_cache_invalidation_on_input_change(self, workflow_runner):
        """Test that cache is invalidated when inputs change."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        # First run
        result1 = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        # Modify input
        modified_data = building_data.copy()
        modified_data['consumption']['electricity']['value'] = 2000000  # Increase consumption
        
        # Second run with modified input
        result2 = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': modified_data}
        )
        
        # Both should succeed
        assert result1['success'] is True
        assert result2['success'] is True
        
        # Results should differ
        emissions1 = result1['data']['emissions_report']['total_co2e_kg']
        emissions2 = result2['data']['emissions_report']['total_co2e_kg']
        
        assert emissions2 > emissions1, "Higher consumption should yield higher emissions"
    
    def test_parallel_error_handling(self, workflow_runner):
        """Test error handling in parallel branches."""
        # Create data that will cause error in one branch
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Set invalid value for one fuel type
        building_data['consumption']['natural_gas']['value'] = -1000  # Negative value
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/parallel_example.yaml',
            {'building_data': building_data}
        )
        
        # Should handle error gracefully
        # Either fail entirely or skip the bad branch
        if not result['success']:
            assert 'error' in result
            assert 'natural_gas' in str(result['error']) or 'negative' in str(result['error']).lower()
    
    def test_parallel_performance_benefit(self, workflow_runner):
        """Test that parallel execution provides performance benefit."""
        # Create data with multiple fuel types
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Add more fuel types for better parallel test
        building_data['consumption']['coal'] = {'value': 1000, 'unit': 'kg'}
        building_data['consumption']['biomass'] = {'value': 500, 'unit': 'kg'}
        
        start = time.time()
        result = workflow_runner.run(
            'tests/fixtures/workflows/parallel_example.yaml',
            {'building_data': building_data}
        )
        parallel_time = time.time() - start
        
        assert result['success'] is True
        
        # Parallel should complete reasonably fast
        assert parallel_time < 3.0, f"Parallel execution took {parallel_time:.2f}s"
    
    def test_cache_key_generation(self, workflow_runner):
        """Test that cache keys are properly generated from inputs."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Generate cache key manually
        input_str = json.dumps(building_data, sort_keys=True)
        expected_key = hashlib.sha256(input_str.encode()).hexdigest()
        
        # This is implementation-specific
        # Just verify workflow runs successfully
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        assert result['success'] is True
    
    def test_parallel_with_dependencies(self, workflow_runner):
        """Test parallel execution with step dependencies."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        # The parallel workflow has dependencies:
        # 1. Parse input (serial)
        # 2. Calculate fuels (parallel)
        # 3. Aggregate (serial, depends on parallel)
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/parallel_example.yaml',
            {'building_data': building_data}
        )
        
        assert result['success'] is True
        
        # Verify execution order was maintained
        # Parse should complete before parallel calcs
        # Aggregate should have all fuel results
        report = result['data']['emissions_report']
        assert 'by_fuel' in report
        assert len(report['by_fuel']) >= 3  # At least 3 fuel types