#!/usr/bin/env python3
"""
Direct test of executor without pack loading dependencies
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Import executor and related classes
from greenlang.runtime.executor import Executor as PipelineExecutor


def test_direct_pipeline_execution():
    """Test pipeline execution directly"""
    # Use default test inputs
    inputs = {
        "building_size_sqft": 1000,
        "building_type": "residential",
        "location": "test_location",
        "capacity": 100
    }

    # Create a simple pipeline manually (not from YAML)
    pipeline = {
        'id': 'test_pipeline',
        'name': 'Test Pipeline',
        'steps': [
            {
                'name': 'test_step',
                'agent': 'mock',
                'code': """
# Test code for pipeline execution
outputs = {}
outputs['processed_inputs'] = inputs
outputs['calculated_value'] = inputs.get('capacity', 0) * 2
outputs['building_info'] = {
    'size': inputs.get('building_size_sqft', 0),
    'type': inputs.get('building_type', 'unknown'),
    'location': inputs.get('location', 'unknown')
}
outputs['status'] = 'success'
outputs['timestamp_test'] = 'fixed_timestamp_2024_01_01'  # Fixed to test determinism
"""
            }
        ]
    }

    # Create executor with deterministic settings
    executor = PipelineExecutor(
        backend="local",
        deterministic=True,
        config={
            'seed': 42,
            'freeze_env': True,
            'normalize_floats': True,
            'float_precision': 6
        }
    )

    # Test execution
    results = []

    for run_num in range(1, 3):  # 2 runs for determinism check
        try:
            # Execute pipeline directly
            result = executor.execute(pipeline, inputs)
            if result.get('success'):
                results.append(result.get('data'))
            else:
                assert False, f"Pipeline execution failed: {result.get('error')}"
        except Exception as e:
            assert False, f"Error during execution: {e}"

    # Check determinism
    if len(results) >= 2:
        assert results[0] == results[1], "Results are not deterministic"

    return True


def test_basic_execution():
    """Test basic execution flow"""
    executor = PipelineExecutor()

    # Simple test pipeline
    pipeline = {
        'id': 'simple_test',
        'name': 'Simple Test',
        'steps': [
            {
                'name': 'simple_step',
                'agent': 'test',
                'code': "outputs = {'result': 'success'}"
            }
        ]
    }

    result = executor.execute(pipeline, {})
    assert result is not None

    return True


if __name__ == '__main__':
    # Run tests
    test_direct_pipeline_execution()
    test_basic_execution()
    print("All tests passed!")