"""Example 25: Concurrency example (run same workflow N times; results identical)."""

import pytest
import json
import concurrent.futures
import hashlib
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.core.orchestrator import run_workflow
except Exception:
    run_workflow = None

@pytest.mark.example
def test_concurrent_workflow_determinism():
    """N concurrent runs produce identical results."""
    if run_workflow is None:
        pytest.skip("Orchestrator not importable")
    
    with open("examples/fixtures/building_india_office.json", "r") as f:
        input_data = json.load(f)
    
    def run_workflow_and_hash():
        """Run workflow and return hash of result."""
        result = run_workflow(
            "examples/fixtures/workflow_minimal.yaml",
            input_data
        )
        # Create deterministic hash of result
        result_json = json.dumps(result, sort_keys=True)
        return hashlib.sha256(result_json.encode()).hexdigest()
    
    # Run 10 workflows concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_workflow_and_hash) for _ in range(10)]
        hashes = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    # All hashes should be identical
    assert len(set(hashes)) == 1, f"Got different results: {set(hashes)}"

@pytest.mark.example
def test_concurrent_agent_calls():
    """Multiple agents called concurrently produce consistent results."""
    try:
        from greenlang.agents.fuel_agent import FuelAgent
    except Exception:
        pytest.skip("FuelAgent not importable")
    
    agent = FuelAgent()
    test_cases = [
        {"fuel_type": "electricity", "consumption": {"value": 1000, "unit": "kWh"}, "country": "US"},
        {"fuel_type": "natural_gas", "consumption": {"value": 100, "unit": "therms"}, "country": "US"},
        {"fuel_type": "diesel", "consumption": {"value": 500, "unit": "liters"}, "country": "EU"},
    ]
    
    def process_case(case):
        """Process a test case and return result."""
        result = agent.run(case)
        return (case["fuel_type"], result["data"]["co2e_emissions_kg"])
    
    # Run cases concurrently multiple times
    all_results = []
    for _ in range(3):  # Run 3 rounds
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_case, case) for case in test_cases]
            round_results = dict(f.result() for f in concurrent.futures.as_completed(futures))
            all_results.append(round_results)
    
    # All rounds should produce identical results
    first_round = all_results[0]
    for round_results in all_results[1:]:
        assert round_results == first_round, "Concurrent execution produced different results"