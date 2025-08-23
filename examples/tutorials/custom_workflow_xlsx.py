"""Tutorial: Build a custom workflow and export to XLSX.

This example shows how to create a custom workflow that includes
a custom agent and exports results to Excel format.
"""

import json
import yaml
try:
    import pandas as pd
    import openpyxl
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

def create_custom_workflow(workflow_path="custom_workflow.yaml"):
    """Create a custom workflow YAML file."""
    workflow = {
        "name": "custom_emissions_workflow",
        "version": "1.0.0",
        "steps": [
            {"name": "validate", "agent_id": "validator"},
            {"name": "fuel_emissions", "agent_id": "fuel"},
            {"name": "water_emissions", "agent_id": "water_usage"},
            {"name": "aggregate", "agent_id": "carbon"},
            {"name": "intensity", "agent_id": "intensity"},
            {"name": "benchmark", "agent_id": "benchmark"},
            {"name": "report", "agent_id": "report", "format": "xlsx"}
        ],
        "export": {
            "format": "xlsx",
            "path": "./emissions_report.xlsx"
        }
    }
    
    with open(workflow_path, 'w') as f:
        yaml.dump(workflow, f)
    
    return workflow_path

def export_to_xlsx(results, output_path="emissions_report.xlsx"):
    """Export workflow results to Excel file.
    
    Args:
        results: Dictionary of workflow results
        output_path: Path for the Excel file
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required for Excel export")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = {
            "Metric": ["Total Emissions", "Intensity", "Rating"],
            "Value": [
                results.get("emissions", {}).get("total_co2e_kg", 0),
                results.get("intensity", {}).get("co2e_per_sqft", 0),
                results.get("benchmark", {}).get("rating", "N/A")
            ]
        }
        pd.DataFrame(summary_data).to_excel(
            writer, sheet_name='Summary', index=False
        )
        
        # Emissions breakdown
        if "emissions" in results and "by_fuel" in results["emissions"]:
            emissions_df = pd.DataFrame(
                list(results["emissions"]["by_fuel"].items()),
                columns=["Fuel Type", "Emissions (kg CO2e)"]
            )
            emissions_df.to_excel(
                writer, sheet_name='Emissions', index=False
            )
        
        # Recommendations
        if "recommendations" in results:
            rec_df = pd.DataFrame(
                results["recommendations"], 
                columns=["Recommendation"]
            )
            rec_df.to_excel(
                writer, sheet_name='Recommendations', index=False
            )
    
    return output_path

# Example usage
if __name__ == "__main__":
    # Create workflow
    workflow_file = create_custom_workflow()
    print(f"Created workflow: {workflow_file}")
    
    # Example results
    sample_results = {
        "emissions": {
            "total_co2e_kg": 1091800,
            "by_fuel": {
                "electricity": 1065000,
                "diesel": 26800
            }
        },
        "intensity": {"co2e_per_sqft": 21.84},
        "benchmark": {"rating": "Good"},
        "recommendations": [
            "Install solar panels",
            "Upgrade to LED lighting",
            "Implement smart HVAC controls"
        ]
    }
    
    # Export to Excel
    if HAS_PANDAS:
        xlsx_file = export_to_xlsx(sample_results)
        print(f"Exported to: {xlsx_file}")