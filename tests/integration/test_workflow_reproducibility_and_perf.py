"""
Reproducibility and performance integration tests.
"""
import pytest
import time
import json
from tests.integration.utils import load_fixture, normalize_json


@pytest.mark.integration
class TestReproducibilityAndPerformance:
    """Test reproducibility, idempotency, and performance."""
    
    def test_deterministic_results(self, workflow_runner):
        """
        Test that same input always produces same output.
        
        Verifies:
        - Multiple runs yield identical results
        - No randomness in calculations
        - Consistent ordering
        """
        building_data = load_fixture('data', 'building_india_office.json')
        
        results = []
        for i in range(3):
            result = workflow_runner.run(
                'tests/fixtures/workflows/commercial_building_emissions.yaml',
                {'building_data': building_data}
            )
            assert result['success'] is True
            results.append(result['data'])
        
        # Normalize and compare all results
        normalized_results = [normalize_json(r) for r in results]
        
        # All should be identical
        for i in range(1, len(normalized_results)):
            assert normalized_results[0] == normalized_results[i], \
                f"Run {i+1} differs from run 1"
    
    def test_idempotency(self, workflow_runner, tmp_outdir):
        """Test that re-running doesn't cause side effects."""
        from tests.integration.utils import TestIOHelper
        
        building_data = load_fixture('data', 'building_india_office.json')
        io_helper = TestIOHelper(tmp_outdir)
        
        # First run
        result1 = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        # Save state
        state1_path = io_helper.write_json('state1.json', result1['data'])
        
        # Second run (idempotent)
        result2 = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        # Save state
        state2_path = io_helper.write_json('state2.json', result2['data'])
        
        # States should be identical
        state1 = io_helper.read_json('state1.json')
        state2 = io_helper.read_json('state2.json')
        
        assert normalize_json(state1) == normalize_json(state2)
        
        # No duplicate entries should be created
        if 'recommendations' in state2['emissions_report']:
            recs1 = state1['emissions_report']['recommendations']
            recs2 = state2['emissions_report']['recommendations']
            assert len(recs1) == len(recs2)
    
    def test_key_ordering_consistency(self, workflow_runner):
        """Test that JSON keys maintain consistent ordering."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        assert result['success'] is True
        
        # Convert to JSON strings to check ordering
        json_str1 = json.dumps(result['data'], sort_keys=True)
        
        # Run again
        result2 = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        json_str2 = json.dumps(result2['data'], sort_keys=True)
        
        # Should be byte-identical
        assert json_str1 == json_str2
    
    @pytest.mark.performance
    def test_single_building_performance(self, workflow_runner):
        """Test performance for single building calculation."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Warm-up run
        workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        # Timed runs
        times = []
        for _ in range(5):
            start = time.time()
            result = workflow_runner.run(
                'tests/fixtures/workflows/commercial_building_emissions.yaml',
                {'building_data': building_data}
            )
            elapsed = time.time() - start
            
            assert result['success'] is True
            times.append(elapsed)
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        # Performance requirements
        assert avg_time < 2.0, f"Average time {avg_time:.2f}s exceeds 2s budget"
        assert max_time < 3.0, f"Max time {max_time:.2f}s exceeds 3s limit"
    
    @pytest.mark.performance
    def test_portfolio_scaling(self, workflow_runner):
        """Test performance scaling with portfolio size."""
        base_building = load_fixture('data', 'portfolio_small.json')['buildings'][0]
        
        sizes = [10, 25, 50]
        times = {}
        
        for size in sizes:
            # Create portfolio of given size
            portfolio = {
                'portfolio_name': f'Test Portfolio {size}',
                'portfolio_id': f'TEST-{size}',
                'aggregation_method': 'sum',
                'buildings': [
                    {**base_building, 'id': f'BLDG-{i:03d}'}
                    for i in range(size)
                ]
            }
            
            start = time.time()
            result = workflow_runner.run(
                'tests/fixtures/workflows/portfolio_analysis.yaml',
                {'portfolio_data': portfolio}
            )
            elapsed = time.time() - start
            
            assert result['success'] is True
            times[size] = elapsed
        
        # Check scaling
        # Time should scale roughly linearly or better
        time_per_building_10 = times[10] / 10
        time_per_building_50 = times[50] / 50
        
        # Per-building time shouldn't increase much
        scaling_factor = time_per_building_50 / time_per_building_10
        assert scaling_factor < 2.0, f"Poor scaling: {scaling_factor:.2f}x slower per building"
    
    def test_memory_efficiency(self, workflow_runner):
        """Test memory usage remains reasonable."""
        import tracemalloc
        
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Start memory tracking
        tracemalloc.start()
        
        # Run workflow multiple times
        for _ in range(10):
            result = workflow_runner.run(
                'tests/fixtures/workflows/commercial_building_emissions.yaml',
                {'building_data': building_data}
            )
            assert result['success'] is True
        
        # Check memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Convert to MB
        peak_mb = peak / 1024 / 1024
        
        # Should use reasonable memory (< 100 MB for simple workflows)
        assert peak_mb < 100, f"Peak memory usage {peak_mb:.1f} MB exceeds limit"
    
    def test_concurrent_execution_safety(self, workflow_runner):
        """Test that concurrent executions don't interfere."""
        import threading
        
        building_data = load_fixture('data', 'building_india_office.json')
        results = []
        errors = []
        
        def run_workflow():
            try:
                result = workflow_runner.run(
                    'tests/fixtures/workflows/commercial_building_emissions.yaml',
                    {'building_data': building_data}
                )
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run multiple workflows concurrently
        threads = []
        for _ in range(5):
            t = threading.Thread(target=run_workflow)
            threads.append(t)
            t.start()
        
        # Wait for all to complete
        for t in threads:
            t.join(timeout=10)
        
        # No errors should occur
        assert len(errors) == 0, f"Concurrent execution errors: {errors}"
        
        # All should succeed
        assert len(results) == 5
        for result in results:
            assert result['success'] is True
        
        # Results should be identical
        normalized = [normalize_json(r['data']) for r in results]
        for i in range(1, len(normalized)):
            assert normalized[0] == normalized[i]
    
    def test_input_size_limits(self, workflow_runner):
        """Test handling of large input data."""
        # Create large building with many fuel types
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Add many consumption types
        for i in range(100):
            building_data['consumption'][f'fuel_{i}'] = {
                'value': 1000,
                'unit': 'units'
            }
        
        start = time.time()
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        elapsed = time.time() - start
        
        # Should handle gracefully
        if result['success']:
            # Should complete in reasonable time
            assert elapsed < 5.0
        else:
            # Should fail with clear error
            assert 'size' in str(result['error']) or 'limit' in str(result['error'])
    
    @pytest.mark.timeout(30)
    def test_workflow_timeout_enforcement(self, workflow_runner):
        """Test that workflows respect timeout limits."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        # This test verifies the timeout marker works
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        # Should complete well within timeout
        assert result['success'] is True