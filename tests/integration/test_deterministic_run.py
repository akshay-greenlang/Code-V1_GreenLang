#!/usr/bin/env python3
"""
Test script for testing deterministic pipeline execution with inputs.json
"""

import json
import yaml
import sys
from pathlib import Path
from datetime import datetime

# Import executor and related classes
from greenlang.runtime.executor import Executor, DeterministicConfig
from greenlang.sdk.context import Context

def load_inputs(inputs_file):
    """Load inputs from JSON or YAML file"""
    inputs_path = Path(inputs_file)
    if not inputs_path.exists():
        raise FileNotFoundError(f"Inputs file not found: {inputs_file}")
    
    if inputs_path.suffix == ".json":
        with open(inputs_path) as f:
            return json.load(f)
    elif inputs_path.suffix in [".yaml", ".yml"]:
        with open(inputs_path) as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported input format: {inputs_path.suffix}")

def load_pipeline(pipeline_file):
    """Load pipeline from YAML file"""
    pipeline_path = Path(pipeline_file)
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_file}")
    
    with open(pipeline_path) as f:
        return yaml.safe_load(f)

def test_pipeline_with_inputs():
    """Test pipeline execution with inputs.json"""
    print("=" * 60)
    print("Testing GL Run Pipeline with Deterministic Execution")
    print("=" * 60)
    
    # Load inputs
    try:
        inputs = load_inputs("inputs.json")
        print("[OK] Loaded inputs.json:")
        print(json.dumps(inputs, indent=2))
    except Exception as e:
        print(f"[FAIL] Failed to load inputs.json: {e}")
        return False
    
    # Load pipeline
    try:
        pipeline = load_pipeline("test_simple.yaml")
        print("[OK] Loaded test_simple.yaml:")
        print(yaml.dump(pipeline, indent=2))
    except Exception as e:
        print(f"[FAIL] Failed to load test_simple.yaml: {e}")
        return False
    
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
            # Execute pipeline
            result = executor.run("test_simple.yaml", inputs)
            
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
            return False
    
    # Check determinism
    print("\n--- Checking Determinism ---")
    
    if len(results) >= 2:
        # Compare first two results
        if results[0] == results[1]:
            print("[OK] Results are deterministic (run 1 == run 2)")
        else:
            print("[FAIL] Results are not deterministic!")
            print("Run 1:", json.dumps(results[0], indent=2))
            print("Run 2:", json.dumps(results[1], indent=2))
            return False
        
        # Compare all results
        all_identical = all(result == results[0] for result in results)
        if all_identical:
            print("[OK] All 3 runs produced identical results")
        else:
            print("[FAIL] Not all runs produced identical results")
            return False
    
    return True

def test_different_inputs():
    """Test with different input sets to ensure determinism per input"""
    print("\n" + "=" * 60)
    print("Testing Different Input Sets")
    print("=" * 60)
    
    # Create test inputs
    test_inputs = [
        {"building_size_sqft": 10000, "location": "New York", "building_type": "office"},
        {"building_size_sqft": 25000, "location": "California", "building_type": "retail"},
        {"building_size_sqft": 50000, "location": "Texas", "building_type": "commercial"}
    ]
    
    executor = Executor(backend="local", deterministic=True)
    
    for i, inputs in enumerate(test_inputs):
        print(f"\n--- Input Set {i+1} ---")
        print(json.dumps(inputs, indent=2))
        
        # Run twice with same inputs
        try:
            result1 = executor.run("test_simple.yaml", inputs)
            result2 = executor.run("test_simple.yaml", inputs)
            
            if result1.success and result2.success:
                if result1.data == result2.data:
                    print("[OK] Deterministic for this input set")
                else:
                    print("[FAIL] Non-deterministic for this input set")
                    return False
            else:
                print("[FAIL] One or both runs failed")
                return False
                
        except Exception as e:
            print(f"[FAIL] Error: {e}")
            return False
    
    return True

def test_artifacts_generation():
    """Test artifact generation and consistency"""
    print("\n" + "=" * 60)
    print("Testing Artifact Generation")
    print("=" * 60)
    
    artifacts_dir = Path("test_artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    inputs = load_inputs("inputs.json")
    executor = Executor(backend="local", deterministic=True)
    
    try:
        result = executor.run("test_simple.yaml", inputs, artifacts_dir=artifacts_dir)
        
        if result.success:
            print("[OK] Pipeline executed successfully")
            
            # Check for generated artifacts
            artifacts = list(artifacts_dir.glob("*"))
            if artifacts:
                print(f"[OK] Generated {len(artifacts)} artifacts:")
                for artifact in artifacts:
                    print(f"  - {artifact.name}")
            else:
                print("- No artifacts generated")
            
            return True
        else:
            print(f"[FAIL] Execution failed: {result.error}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False

def test_input_validation():
    """Test input validation and error handling"""
    print("\n" + "=" * 60)
    print("Testing Input Validation")
    print("=" * 60)
    
    executor = Executor(backend="local", deterministic=True)
    
    # Test with empty inputs
    print("Testing with empty inputs...")
    try:
        result = executor.run("test_simple.yaml", {})
        if result.success:
            print("[OK] Empty inputs handled successfully")
        else:
            print(f"- Empty inputs result: {result.error}")
    except Exception as e:
        print(f"- Empty inputs exception: {e}")
    
    # Test with malformed inputs
    print("\nTesting with malformed inputs...")
    try:
        result = executor.run("test_simple.yaml", {"invalid": None})
        if result.success:
            print("[OK] Malformed inputs handled successfully") 
        else:
            print(f"- Malformed inputs result: {result.error}")
    except Exception as e:
        print(f"- Malformed inputs exception: {e}")
    
    return True

def check_for_timestamps_and_randomness():
    """Check for non-deterministic elements like timestamps"""
    print("\n" + "=" * 60)
    print("Checking for Timestamps and Randomness")
    print("=" * 60)
    
    inputs = load_inputs("inputs.json")
    executor = Executor(backend="local", deterministic=True)
    
    # Run pipeline and check for timestamp patterns
    try:
        result = executor.run("test_simple.yaml", inputs)
        
        if result.success:
            result_str = json.dumps(result.data, sort_keys=True)
            
            # Check for common timestamp patterns
            timestamp_patterns = [
                "timestamp", "created_at", "updated_at", "datetime", 
                "T", "Z", "2023", "2024", "2025"
            ]
            
            found_timestamps = []
            for pattern in timestamp_patterns:
                if pattern in result_str:
                    found_timestamps.append(pattern)
            
            if found_timestamps:
                print(f"[WARN] Potential timestamp patterns found: {found_timestamps}")
                print("This might affect determinism")
            else:
                print("[OK] No obvious timestamp patterns found")
            
            # Check for UUID patterns (random IDs)
            import re
            uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
            if re.search(uuid_pattern, result_str, re.IGNORECASE):
                print("[WARN] UUID patterns found - might affect determinism")
            else:
                print("[OK] No UUID patterns found")
            
            return True
        else:
            print(f"[FAIL] Execution failed: {result.error}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False

def main():
    """Run all tests"""
    print("Starting comprehensive deterministic execution tests...")
    
    success = True
    
    # Test 1: Basic deterministic execution with inputs.json
    if not test_pipeline_with_inputs():
        success = False
    
    # Test 2: Different input sets
    if not test_different_inputs():
        success = False
    
    # Test 3: Artifact generation
    if not test_artifacts_generation():
        success = False
    
    # Test 4: Input validation
    if not test_input_validation():
        success = False
    
    # Test 5: Check for non-deterministic elements
    if not check_for_timestamps_and_randomness():
        success = False
    
    # Final summary
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TEST RESULTS")
    print("=" * 60)
    
    if success:
        print("[OK] ALL TESTS PASSED")
        print("[OK] Pipeline execution is deterministic")
        print("[OK] Inputs.json processing works correctly")
        print("[OK] No significant non-deterministic elements detected")
        print("\n[SUCCESS] GL RUN PIPELINE COMMAND IS WORKING PROPERLY!")
    else:
        print("[FAIL] SOME TESTS FAILED")
        print("There are issues with deterministic execution")
    
    print("=" * 60)
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)