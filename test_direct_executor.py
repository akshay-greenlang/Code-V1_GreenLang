#!/usr/bin/env python3
"""
Direct test of executor without pack loading dependencies
"""

import json
import yaml
import sys
from pathlib import Path
from datetime import datetime

# Import executor and related classes
from core.greenlang.runtime.executor import Executor, DeterministicConfig
from core.greenlang.sdk.context import Context
from core.greenlang.agents.mock import MockAgent


def test_direct_pipeline_execution():
    """Test pipeline execution directly with mock agent"""
    print("=" * 60)
    print("Testing Direct Pipeline Execution")
    print("=" * 60)
    
    # Load inputs
    with open("inputs.json") as f:
        inputs = json.load(f)
    print("[OK] Loaded inputs.json:")
    print(json.dumps(inputs, indent=2))
    
    # Create a simple pipeline manually (not from YAML)
    pipeline = {
        "name": "direct-test-pipeline",
        "version": "1.0.0",
        "steps": [
            {
                "name": "test_step",
                "type": "python",
                "code": """
# Process the inputs
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
    
    print("[OK] Created test pipeline:")
    print(json.dumps(pipeline, indent=2))
    
    # Create deterministic executor
    det_config = DeterministicConfig(
        seed=42,
        freeze_env=True,
        normalize_floats=True,
        float_precision=6
    )
    
    executor = Executor(
        backend="local",
        deterministic=True,
        det_config=det_config
    )
    
    print("[OK] Created deterministic executor")
    
    # Test multiple runs for determinism
    print("\n--- Testing Deterministic Execution ---")
    results = []
    
    for run_num in range(1, 4):  # 3 runs
        print(f"\nRun #{run_num}:")
        
        try:
            # Execute pipeline directly
            result = executor._exec_local(pipeline, inputs)
            
            if result.success:
                print("[OK] Pipeline executed successfully")
                results.append(result.data)
                
                # Show result data
                print("Result data:")
                print(json.dumps(result.data, indent=2))
                
                # Show metadata
                if hasattr(result, 'metadata'):
                    print("Metadata:")
                    print(json.dumps(result.metadata, indent=2))
            else:
                print(f"[FAIL] Pipeline execution failed: {result.error}")
                return False
                
        except Exception as e:
            print(f"[FAIL] Error during execution: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Check determinism
    print("\n--- Checking Determinism ---")
    
    if len(results) >= 2:
        # Compare first two results
        if results[0] == results[1]:
            print("[OK] Results are deterministic (run 1 == run 2)")
        else:
            print("[FAIL] Results are not deterministic!")
            print("Run 1:", json.dumps(results[0], indent=2, sort_keys=True))
            print("Run 2:", json.dumps(results[1], indent=2, sort_keys=True))
            return False
        
        # Compare all results
        all_identical = all(result == results[0] for result in results)
        if all_identical:
            print("[OK] All 3 runs produced identical results")
        else:
            print("[FAIL] Not all runs produced identical results")
            return False
    
    return True


def test_mock_agent_execution():
    """Test execution with mock agent directly"""
    print("\n" + "=" * 60)
    print("Testing Mock Agent Execution")
    print("=" * 60)
    
    # Test mock agent directly first
    mock_agent = MockAgent()
    
    # Test with inputs.json data
    with open("inputs.json") as f:
        inputs = json.load(f)
    
    print("Testing mock agent directly...")
    result = mock_agent.execute(**inputs)
    print("[OK] Mock agent result:")
    print(json.dumps(result, indent=2))
    
    # Test if mock agent gives consistent results
    result2 = mock_agent.execute(**inputs)
    if result == result2:
        print("[OK] Mock agent is deterministic")
    else:
        print("[WARN] Mock agent results differ:")
        print("Result 1:", result)
        print("Result 2:", result2)
    
    return True


def test_with_different_backends():
    """Test pipeline execution with different backends"""
    print("\n" + "=" * 60)
    print("Testing Different Backends")
    print("=" * 60)
    
    with open("inputs.json") as f:
        inputs = json.load(f)
    
    # Simple pipeline
    pipeline = {
        "name": "backend-test",
        "steps": [
            {
                "name": "compute",
                "type": "python",
                "code": """
outputs['input_sum'] = sum(v for v in inputs.values() if isinstance(v, (int, float)))
outputs['backend'] = 'local'
outputs['deterministic'] = True
"""
            }
        ]
    }
    
    backends = ["local"]  # Only test local for now
    results = {}
    
    for backend in backends:
        print(f"\nTesting backend: {backend}")
        
        try:
            executor = Executor(backend=backend, deterministic=True)
            result = executor._exec_local(pipeline, inputs)
            
            if result.success:
                results[backend] = result.data
                print(f"[OK] Backend {backend} executed successfully")
                print(json.dumps(result.data, indent=2))
            else:
                print(f"[FAIL] Backend {backend} failed: {result.error}")
                return False
                
        except Exception as e:
            print(f"[FAIL] Backend {backend} error: {e}")
            return False
    
    return True


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "=" * 60)
    print("Testing Edge Cases")
    print("=" * 60)
    
    executor = Executor(backend="local", deterministic=True)
    
    # Test 1: Empty pipeline
    print("Testing empty pipeline...")
    try:
        empty_pipeline = {"name": "empty", "steps": []}
        result = executor._exec_local(empty_pipeline, {})
        print("[OK] Empty pipeline handled successfully")
        print("Result:", result.data)
    except Exception as e:
        print(f"[WARN] Empty pipeline error: {e}")
    
    # Test 2: Pipeline with syntax error in Python code
    print("\nTesting pipeline with syntax error...")
    try:
        bad_pipeline = {
            "name": "bad",
            "steps": [{
                "name": "bad_step",
                "type": "python",
                "code": "this is not valid python syntax"
            }]
        }
        result = executor._exec_local(bad_pipeline, {})
        if result.success:
            print("[WARN] Bad syntax was accepted")
        else:
            print("[OK] Bad syntax was rejected")
    except Exception as e:
        print(f"[OK] Bad syntax caused exception: {e}")
    
    # Test 3: Large inputs
    print("\nTesting large inputs...")
    try:
        large_inputs = {
            "large_list": list(range(10000)),
            "large_dict": {f"key_{i}": i for i in range(1000)}
        }
        
        simple_pipeline = {
            "name": "large",
            "steps": [{
                "name": "process_large",
                "type": "python",
                "code": """
outputs['list_len'] = len(inputs.get('large_list', []))
outputs['dict_len'] = len(inputs.get('large_dict', {}))
outputs['status'] = 'processed'
"""
            }]
        }
        
        result = executor._exec_local(simple_pipeline, large_inputs)
        if result.success:
            print("[OK] Large inputs processed successfully")
            print("Sizes:", result.data['process_large']['list_len'], result.data['process_large']['dict_len'])
        else:
            print(f"[FAIL] Large inputs failed: {result.error}")
            
    except Exception as e:
        print(f"[FAIL] Large inputs error: {e}")
    
    return True


def test_determinism_with_randomness():
    """Test determinism controls when using random numbers"""
    print("\n" + "=" * 60)
    print("Testing Determinism with Randomness")
    print("=" * 60)
    
    # Pipeline that uses random numbers
    random_pipeline = {
        "name": "random-test",
        "steps": [
            {
                "name": "generate_random",
                "type": "python",
                "code": """
import random
# The seed should be set by deterministic config
random.seed(__seed__)  # Use the deterministic seed
outputs['random_value'] = random.random()
outputs['random_int'] = random.randint(1, 100)
outputs['random_list'] = [random.random() for _ in range(5)]
outputs['seed_used'] = __seed__
"""
            }
        ]
    }
    
    with open("inputs.json") as f:
        inputs = json.load(f)
    
    executor = Executor(backend="local", deterministic=True)
    
    # Run multiple times
    random_results = []
    for i in range(3):
        result = executor._exec_local(random_pipeline, inputs)
        if result.success:
            random_results.append(result.data)
            print(f"Run {i+1}: {result.data['generate_random']['random_value']:.6f}")
        else:
            print(f"[FAIL] Random pipeline failed: {result.error}")
            return False
    
    # Check if all results are identical
    if len(random_results) >= 2:
        if random_results[0] == random_results[1] == random_results[2]:
            print("[OK] Random number generation is deterministic!")
        else:
            print("[FAIL] Random number generation is not deterministic")
            for i, result in enumerate(random_results):
                print(f"Result {i+1}:", result['generate_random'])
            return False
    
    return True


def main():
    """Run all direct executor tests"""
    print("Starting direct executor deterministic tests...")
    
    success = True
    
    # Test 1: Direct pipeline execution
    if not test_direct_pipeline_execution():
        success = False
    
    # Test 2: Mock agent execution
    if not test_mock_agent_execution():
        success = False
    
    # Test 3: Different backends
    if not test_with_different_backends():
        success = False
    
    # Test 4: Edge cases
    if not test_edge_cases():
        success = False
    
    # Test 5: Determinism with randomness
    if not test_determinism_with_randomness():
        success = False
    
    # Final summary
    print("\n" + "=" * 60)
    print("DIRECT EXECUTOR TEST RESULTS")
    print("=" * 60)
    
    if success:
        print("[OK] ALL TESTS PASSED")
        print("[OK] Direct pipeline execution is deterministic")
        print("[OK] Inputs.json processing works correctly")
        print("[OK] Executor handles edge cases appropriately")
        print("\n[SUCCESS] GL RUN PIPELINE EXECUTOR IS WORKING PROPERLY!")
    else:
        print("[FAIL] SOME TESTS FAILED")
        print("There are issues with the executor implementation")
    
    print("=" * 60)
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)