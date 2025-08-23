"""Example 10: Minimal YAML workflow with orchestrator."""

import json
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.core.orchestrator import run_workflow
except Exception:
    run_workflow = None

@pytest.mark.example
def test_workflow_minimal_end_to_end():
    """YAML → orchestrator → emissions/intensity/benchmark."""
    if run_workflow is None:
        pytest.skip("Orchestrator not importable")
    
    with open("examples/fixtures/building_india_office.json", "r") as f:
        input_data = json.load(f)
    
    result = run_workflow(
        "examples/fixtures/workflow_minimal.yaml",
        input_data
    )
    
    assert result["success"] is True
    d = result["data"]
    assert d["emissions"]["total_co2e_kg"] > 0
    assert "intensity" in d and "benchmark" in d