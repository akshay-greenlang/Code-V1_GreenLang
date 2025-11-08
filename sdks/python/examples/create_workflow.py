"""
Example: Create a workflow using GreenLang SDK

This example demonstrates how to create a workflow that analyzes
carbon footprint data.
"""

from greenlang_sdk import GreenLangClient

# Initialize client with your API key
client = GreenLangClient(api_key="gl_your_api_key_here")

# Define workflow
workflow_def = {
    "name": "Carbon Footprint Analysis",
    "description": "Comprehensive carbon footprint analysis from various data sources",
    "category": "carbon",
    "agents": [
        {
            "agent_id": "data_collector",
            "config": {
                "sources": ["energy", "transportation", "waste"]
            }
        },
        {
            "agent_id": "carbon_analyzer",
            "config": {
                "threshold": 100,
                "unit": "kg_co2"
            }
        },
        {
            "agent_id": "report_generator",
            "config": {
                "format": "comprehensive",
                "include_recommendations": True
            }
        }
    ],
    "config": {
        "parallel_execution": False,
        "cache_results": True,
        "timeout_seconds": 300
    },
    "is_public": False
}

try:
    # Create workflow
    workflow = client.create_workflow(workflow_def)

    print(f"Workflow created successfully!")
    print(f"ID: {workflow.id}")
    print(f"Name: {workflow.name}")
    print(f"Agents: {len(workflow.agents)}")
    print(f"Created at: {workflow.created_at}")

except Exception as e:
    print(f"Error creating workflow: {e}")
finally:
    client.close()
