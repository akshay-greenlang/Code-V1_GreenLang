"""Example 30: Test for custom workflow with XLSX export tutorial."""

import pytest
import os
import tempfile
import yaml
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from examples.tutorials.custom_workflow_xlsx import (
        create_custom_workflow, export_to_xlsx
    )
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    create_custom_workflow = None
    export_to_xlsx = None
    HAS_PANDAS = False

@pytest.mark.example
def test_create_custom_workflow():
    """Test creating a custom workflow YAML."""
    if create_custom_workflow is None:
        pytest.skip("Custom workflow tutorial not available")
    
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
        temp_path = f.name
    
    try:
        # Create workflow
        workflow_path = create_custom_workflow(temp_path)
        assert os.path.exists(workflow_path)
        
        # Load and verify workflow
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        assert workflow["name"] == "custom_emissions_workflow"
        assert workflow["version"] == "1.0.0"
        assert len(workflow["steps"]) == 7
        assert workflow["export"]["format"] == "xlsx"
        
        # Check for custom water_usage agent
        agent_ids = [step["agent_id"] for step in workflow["steps"]]
        assert "water_usage" in agent_ids
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@pytest.mark.example
def test_export_to_xlsx():
    """Test exporting results to Excel format."""
    if not HAS_PANDAS or export_to_xlsx is None:
        pytest.skip("pandas or export_to_xlsx not available")
    
    # Sample results
    results = {
        "emissions": {
            "total_co2e_kg": 1091800,
            "by_fuel": {
                "electricity": 1065000,
                "diesel": 26800,
                "natural_gas": 0
            }
        },
        "intensity": {"co2e_per_sqft": 21.84},
        "benchmark": {"rating": "Good", "threshold": 25.0},
        "recommendations": [
            "Install solar panels",
            "Upgrade to LED lighting"
        ]
    }
    
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
        temp_path = f.name
    
    try:
        # Export to Excel
        output_path = export_to_xlsx(results, temp_path)
        assert os.path.exists(output_path)
        
        # Verify Excel file structure
        excel_file = pd.ExcelFile(output_path)
        sheet_names = excel_file.sheet_names
        
        assert "Summary" in sheet_names
        assert "Emissions" in sheet_names
        assert "Recommendations" in sheet_names
        
        # Check Summary sheet
        summary_df = pd.read_excel(output_path, sheet_name="Summary")
        assert "Total Emissions" in summary_df["Metric"].values
        assert 1091800 in summary_df["Value"].values
        
        # Check Emissions sheet
        emissions_df = pd.read_excel(output_path, sheet_name="Emissions")
        assert len(emissions_df) == 3  # Three fuel types
        assert "electricity" in emissions_df["Fuel Type"].values
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@pytest.mark.example
def test_load_custom_workflow_fixture():
    """Test loading the custom workflow fixture."""
    with open("examples/fixtures/custom_workflow.yaml", "r") as f:
        workflow = yaml.safe_load(f)
    
    assert workflow["name"] == "custom_emissions_with_xlsx_export"
    assert workflow["output_format"] == "xlsx"
    assert "water_usage" in [s["agent_id"] for s in workflow["steps"]]
    assert len(workflow["export"]["sheets"]) == 4