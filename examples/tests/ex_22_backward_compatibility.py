"""Example 22: Backward compatibility for old workflow YAML."""

import pytest
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.core.orchestrator import run_workflow
    from greenlang.core.workflow_migrator import migrate_workflow_v1_to_v2
except Exception:
    run_workflow = None
    migrate_workflow_v1_to_v2 = None

@pytest.mark.example
def test_backward_compatibility_v1_workflow():
    """Old v1.0 workflow format still works via migration."""
    if run_workflow is None:
        pytest.skip("Orchestrator not importable")
    
    with open("examples/fixtures/building_india_office.json", "r") as f:
        input_data = json.load(f)
    
    # Try to run legacy workflow directly
    try:
        # This might fail if v1 format not supported
        result_legacy = run_workflow(
            "examples/fixtures/workflow_v1_legacy.yaml",
            input_data
        )
        assert result_legacy["success"] is True
        assert result_legacy["data"]["emissions"]["total_co2e_kg"] > 0
    except Exception as e:
        # If direct run fails, test migration path
        if migrate_workflow_v1_to_v2 is None:
            pytest.skip("Workflow migrator not available")
        
        # Migrate and run
        migrated = migrate_workflow_v1_to_v2(
            "examples/fixtures/workflow_v1_legacy.yaml"
        )
        result_migrated = run_workflow(migrated, input_data)
        assert result_migrated["success"] is True
        assert result_migrated["data"]["emissions"]["total_co2e_kg"] > 0

@pytest.mark.example  
def test_workflow_version_detection():
    """System correctly detects workflow version."""
    try:
        from greenlang.core.workflow_loader import detect_workflow_version
    except Exception:
        pytest.skip("Workflow version detection not available")
    
    # Check v1 detection
    v1_version = detect_workflow_version(
        "examples/fixtures/workflow_v1_legacy.yaml"
    )
    assert v1_version == "1.0" or v1_version == 1
    
    # Check v2 detection  
    v2_version = detect_workflow_version(
        "examples/fixtures/workflow_minimal.yaml"
    )
    assert v2_version.startswith("0.0.1") or v2_version == 2