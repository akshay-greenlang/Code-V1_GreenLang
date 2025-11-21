# -*- coding: utf-8 -*-
"""
Example: Execute an agent using GreenLang SDK

This example demonstrates how to execute a carbon analyzer agent directly.
"""

from greenlang_sdk import GreenLangClient

# Initialize client with your API key
client = GreenLangClient(api_key="gl_your_api_key_here")

# Input data for the agent
input_data = {
    "query": "Calculate carbon emissions for 1000 kWh of electricity usage",
    "location": "California, USA",
    "data_sources": ["energy_grid", "regional_factors"]
}

# Agent configuration
agent_config = {
    "precision": "high",
    "include_sources": True,
    "confidence_threshold": 0.8
}

try:
    # Execute agent
    print("Executing carbon analyzer agent...")
    result = client.execute_agent(
        agent_id="carbon_analyzer",
        input_data=input_data,
        config=agent_config
    )

    # Check result
    if result.is_successful:
        print(f"\nExecution successful!")
        print(f"Execution ID: {result.id}")
        print(f"Duration: {result.duration_ms}ms")
        print(f"\nResults:")
        print(f"  Answer: {result.data.get('answer')}")
        print(f"  Confidence: {result.data.get('confidence', 0):.2%}")

        # Display citations if available
        if result.citations:
            print(f"\nCitations ({len(result.citations)}):")
            for i, citation in enumerate(result.citations, 1):
                print(f"  {i}. {citation.source_title}")
                print(f"     URL: {citation.source_url}")
                print(f"     Relevance: {citation.relevance_score:.2%}")
    else:
        print(f"Execution failed: {result.error}")

except Exception as e:
    print(f"Error executing agent: {e}")
finally:
    client.close()
