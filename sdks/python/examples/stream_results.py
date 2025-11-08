"""
Example: Stream workflow results using GreenLang SDK

This example demonstrates how to stream results from a long-running workflow.
"""

from greenlang_sdk import GreenLangClient

# Initialize client
client = GreenLangClient(api_key="gl_your_api_key_here")

workflow_id = "wf_your_workflow_id"

input_data = {
    "query": "Perform comprehensive climate risk analysis",
    "scope": "global",
    "timeframe": "2025-2050"
}

try:
    print("Starting workflow execution with streaming...")
    print("=" * 60)

    # Stream results
    for chunk in client.stream_execution(workflow_id, input_data):
        # Handle different chunk types
        if chunk.get("type") == "progress":
            print(f"Progress: {chunk.get('percentage', 0):.0%} - {chunk.get('message', '')}")

        elif chunk.get("type") == "agent_result":
            agent_name = chunk.get("agent_name", "Unknown")
            print(f"\nAgent '{agent_name}' completed:")
            print(f"  Output: {chunk.get('output', {})}")

        elif chunk.get("type") == "citation":
            citation = chunk.get("citation", {})
            print(f"  Citation: {citation.get('source_title', 'Unknown')}")

        elif chunk.get("type") == "complete":
            print(f"\n{'=' * 60}")
            print("Workflow completed!")
            result = chunk.get("result", {})
            print(f"Final result: {result}")

        elif chunk.get("type") == "error":
            print(f"\nError: {chunk.get('error', 'Unknown error')}")
            break

except KeyboardInterrupt:
    print("\nStreaming interrupted by user")
except Exception as e:
    print(f"Error during streaming: {e}")
finally:
    client.close()
