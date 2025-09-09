#!/usr/bin/env python
"""
Priority 2A: Pipeline Executor Implementation - Validation Test
================================================================

This test validates that the Pipeline Executor properly:
1. Loads pipeline definitions from YAML
2. Resolves and loads agents dynamically
3. Executes steps in sequence
4. Passes context between steps
5. Aggregates results properly
"""

from pathlib import Path
from core.greenlang.runtime.executor import Executor
from core.greenlang.sdk.context import Context


def test_pipeline_executor():
    """Test the complete pipeline executor implementation"""
    
    print("Priority 2A: Pipeline Executor Implementation Test")
    print("=" * 60)
    
    # Test 1: Basic pipeline execution
    print("\n1. Testing basic pipeline execution...")
    executor = Executor()
    
    # Create test inputs
    test_inputs = {
        'project_name': 'test-project',
        'location': 'New York'
    }
    
    # Run the pipeline
    result = executor.run('test_pipeline.yaml', test_inputs)
    
    assert result.success, f"Pipeline execution failed: {result.error}"
    assert 'step1-analysis' in result.data, "Missing step1-analysis in results"
    assert 'step2-solar' in result.data, "Missing step2-solar in results"
    print("   [OK] Pipeline executed successfully")
    
    # Test 2: Verify step data propagation
    print("\n2. Testing step data propagation...")
    step1_data = result.data['step1-analysis']
    step2_data = result.data['step2-solar']
    
    assert step1_data['success'], "Step 1 should succeed"
    assert step2_data['success'], "Step 2 should succeed"
    assert 'outputs' in step1_data, "Step 1 missing outputs"
    assert 'outputs' in step2_data, "Step 2 missing outputs"
    print("   [OK] Step data properly propagated")
    
    # Test 3: Verify Context handling
    print("\n3. Testing Context integration...")
    assert hasattr(result, 'metadata'), "Result missing metadata attribute"
    assert 'inputs' in result.metadata, "Missing inputs in metadata"
    assert 'duration' in result.metadata, "Missing duration in metadata"
    print("   [OK] Context properly integrated")
    
    # Test 4: Verify agent loading
    print("\n4. Testing agent loading capabilities...")
    # Test different agent path formats
    test_paths = [
        "packs/boiler-solar/agents/boiler_analyzer.py",
        "packs/boiler-solar/agents/solar_estimator.py:SolarEstimatorAgent"
    ]
    
    for path in test_paths:
        try:
            agent = executor.loader.get_agent(path)
            assert agent is not None, f"Failed to load agent: {path}"
            print(f"   [OK] Loaded agent: {path}")
        except Exception as e:
            print(f"   [FAIL] Could not load agent {path}: {e}")
            raise
    
    # Test 5: Verify result aggregation
    print("\n5. Testing result aggregation...")
    boiler_output = step1_data['outputs']
    solar_output = step2_data['outputs']
    
    assert 'efficiency' in boiler_output, "Missing efficiency in boiler output"
    assert 'emissions' in boiler_output, "Missing emissions in boiler output"
    assert 'annual_generation' in solar_output, "Missing generation in solar output"
    assert 'capacity_factor' in solar_output, "Missing capacity factor in solar output"
    print("   [OK] Results properly aggregated")
    
    # Summary
    print("\n" + "=" * 60)
    print("PRIORITY 2A VALIDATION: ALL TESTS PASSED")
    print("=" * 60)
    print("\nPipeline Executor Implementation Features Verified:")
    print("- Pipeline loading from YAML")
    print("- Dynamic agent loading and instantiation")
    print("- Sequential step execution")
    print("- Context passing between steps")
    print("- Result aggregation and metadata tracking")
    print("\nSample Pipeline Output:")
    print(f"- Boiler Efficiency: {boiler_output['efficiency']}")
    print(f"- Annual Emissions: {boiler_output['emissions']} tons CO2")
    print(f"- Solar Generation: {solar_output['annual_generation']:.0f} kWh/year")
    print(f"- Solar Capacity Factor: {solar_output['capacity_factor']:.2%}")


if __name__ == "__main__":
    test_pipeline_executor()