# -*- coding: utf-8 -*-
"""
Test script for Climatenza feasibility workflow
"""

import sys
import os
import json
from pathlib import Path

# Removed sys.path manipulation - using installed package

from greenlang.core.orchestrator import Orchestrator
from greenlang.core.workflow import Workflow
from greenlang.agents import (
    InputValidatorAgent, FuelAgent, CarbonAgent, ReportAgent,
    BenchmarkAgent, BoilerAgent, GridFactorAgent, BuildingProfileAgent,
    IntensityAgent, RecommendationAgent, SiteInputAgent, SolarResourceAgent,
    LoadProfileAgent, FieldLayoutAgent, EnergyBalanceAgent
)

def test_workflow():
    print("=== Testing Climatenza Feasibility Workflow ===\n")
    
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    # Register all agents
    print("Registering agents...")
    orchestrator.register_agent("validator", InputValidatorAgent())
    orchestrator.register_agent("fuel", FuelAgent())
    orchestrator.register_agent("carbon", CarbonAgent())
    orchestrator.register_agent("report", ReportAgent())
    orchestrator.register_agent("benchmark", BenchmarkAgent())
    orchestrator.register_agent("boiler", BoilerAgent())
    orchestrator.register_agent("grid_factor", GridFactorAgent())
    orchestrator.register_agent("building_profile", BuildingProfileAgent())
    orchestrator.register_agent("intensity", IntensityAgent())
    orchestrator.register_agent("recommendation", RecommendationAgent())
    
    # Register Climatenza agents
    orchestrator.register_agent("SiteInputAgent", SiteInputAgent())
    orchestrator.register_agent("SolarResourceAgent", SolarResourceAgent())
    orchestrator.register_agent("LoadProfileAgent", LoadProfileAgent())
    orchestrator.register_agent("FieldLayoutAgent", FieldLayoutAgent())
    orchestrator.register_agent("EnergyBalanceAgent", EnergyBalanceAgent())
    
    # Load workflow
    print("Loading workflow...")
    workflow_path = "gl_workflows/feasibility_base.yaml"
    workflow = Workflow.from_yaml(workflow_path)
    orchestrator.register_workflow("main", workflow)
    
    # Prepare input data with the site_file path (relative to current directory)
    input_data = {
        "inputs": {
            "site_file": "examples/dairy_hotwater_site.yaml"
        }
    }
    
    # Execute workflow
    print("Executing workflow...\n")
    result = orchestrator.execute_workflow("main", input_data)
    
    # Debug: Print context results
    print("\nDebug - Context Results:")
    if not result["success"] and "LoadSiteData" in str(result):
        # Try to get partial results for debugging
        pass
    
    # Display results
    if result["success"]:
        print("[OK] Workflow completed successfully!\n")
        print("=== CLIMATENZA AI FEASIBILITY RESULTS ===\n")
        if "data" in result and result["data"]:
            for key, value in result["data"].items():
                if isinstance(value, float):
                    if "gwh" in key.lower():
                        print(f"  {key}: {value:.3f} GWh")
                    elif "fraction" in key.lower():
                        print(f"  {key}: {value:.1%}")
                    elif "m2" in key.lower():
                        print(f"  {key}: {value:,.0f} m²")
                    else:
                        print(f"  {key}: {value:,.0f}")
                else:
                    print(f"  {key}: {value}")
        else:
            print("  No output data available")
    else:
        print("[ERROR] Workflow failed!\n")
        print("Errors:")
        for error in result["errors"]:
            print(f"  - {error['step']}: {error['error']}")
    
    # Debug: Print what was passed
    print("\nDebug Info:")
    print(f"Input data structure: {json.dumps(input_data, indent=2)}")
    
    return result

if __name__ == "__main__":
    test_workflow()