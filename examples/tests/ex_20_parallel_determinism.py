"""Example 20: Parallel workflow determinism."""

import pytest
import json
import threading
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.core.orchestrator import run_workflow
except Exception:
    run_workflow = None

@pytest.mark.example
def test_parallel_workflow_determinism():
    """Parallel execution yields identical results."""
    if run_workflow is None:
        pytest.skip("Orchestrator not importable")
    
    with open("examples/fixtures/building_india_office.json", "r") as f:
        input_data = json.load(f)
    
    results = []
    threads = []
    
    def run_and_store(index):
        """Run workflow and store result."""
        result = run_workflow(
            "examples/fixtures/workflow_minimal.yaml",
            input_data
        )
        results.append((index, result))
    
    # Launch 5 parallel executions
    for i in range(5):
        thread = threading.Thread(target=run_and_store, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all to complete
    for thread in threads:
        thread.join()
    
    # Sort by index to maintain order
    results.sort(key=lambda x: x[0])
    
    # All results should be identical
    first_result = results[0][1]
    for index, result in results[1:]:
        assert result["success"] == first_result["success"]
        assert result["data"]["emissions"]["total_co2e_kg"] == \
               first_result["data"]["emissions"]["total_co2e_kg"]
        
        # Check all numeric values are identical
        if "intensity" in result["data"]:
            assert result["data"]["intensity"] == first_result["data"]["intensity"]