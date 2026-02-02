# -*- coding: utf-8 -*-
"""
Example Usage of GreenLang GraphQL API
Demonstrates how to set up and use the GraphQL API
"""

import asyncio
from greenlang.core.orchestrator import Orchestrator
from greenlang.auth.rbac import RBACManager
from greenlang.auth.auth import AuthManager
from greenlang.agents.base import BaseAgent, AgentConfig, AgentResult
from greenlang.core.workflow import Workflow, WorkflowStep
from greenlang.api.graphql.server import create_graphql_app, run_dev_server


# ==============================================================================
# Example 1: Basic Setup
# ==============================================================================

def example_basic_setup():
    """Basic GraphQL API setup"""
    print("=" * 60)
    print("Example 1: Basic Setup")
    print("=" * 60)

    # Initialize core services
    orchestrator = Orchestrator()
    auth_manager = AuthManager()
    rbac_manager = RBACManager()

    # Create a test user
    user_id = auth_manager.create_user(
        tenant_id="demo-tenant",
        username="demo_user",
        email="demo@example.com",
        password="SecurePassword123!",
    )
    print(f"Created user: {user_id}")

    # Assign roles
    rbac_manager.assign_role(user_id, "developer")
    print(f"Assigned 'developer' role to user")

    # Create auth token
    token = auth_manager.create_token(
        tenant_id="demo-tenant",
        user_id=user_id,
        name="Demo Token",
    )
    print(f"Created auth token: {token.token_value[:20]}...")

    # Create GraphQL app
    app = create_graphql_app(
        orchestrator=orchestrator,
        auth_manager=auth_manager,
        rbac_manager=rbac_manager,
        enable_playground=True,
        enable_introspection=True,
        debug=True,
    )
    print("GraphQL app created successfully!")

    return app, orchestrator, auth_manager, rbac_manager, user_id, token


# ==============================================================================
# Example 2: Creating Agents
# ==============================================================================

def example_create_agents(orchestrator):
    """Create example agents"""
    print("\n" + "=" * 60)
    print("Example 2: Creating Agents")
    print("=" * 60)

    # Define custom agent
    class DataProcessorAgent(BaseAgent):
        """Agent that processes data"""

        def execute(self, input_data):
            data = input_data.get("data", [])
            processed = [x * 2 for x in data]

            return AgentResult(
                success=True,
                data={"processed": processed, "count": len(processed)},
                metadata={"agent": self.config.name},
            )

    # Create agent config
    config = AgentConfig(
        name="Data Processor",
        description="Processes numerical data",
        version="1.0.0",
        enabled=True,
        parameters={"multiplier": 2},
    )

    # Create and register agent
    agent = DataProcessorAgent(config)
    agent_id = "data-processor-1"
    orchestrator.register_agent(agent_id, agent)

    print(f"Registered agent: {agent_id}")
    print(f"Agent name: {agent.config.name}")
    print(f"Agent version: {agent.config.version}")

    # Create another agent
    class DataValidatorAgent(BaseAgent):
        """Agent that validates data"""

        def execute(self, input_data):
            data = input_data.get("data", [])
            valid = all(isinstance(x, (int, float)) for x in data)

            return AgentResult(
                success=valid,
                data={"valid": valid, "count": len(data)},
                error=None if valid else "Invalid data types",
            )

    validator_config = AgentConfig(
        name="Data Validator",
        description="Validates numerical data",
        version="1.0.0",
    )

    validator = DataValidatorAgent(validator_config)
    validator_id = "data-validator-1"
    orchestrator.register_agent(validator_id, validator)

    print(f"Registered agent: {validator_id}")

    return agent_id, validator_id


# ==============================================================================
# Example 3: Creating Workflows
# ==============================================================================

def example_create_workflow(orchestrator, agent_id, validator_id):
    """Create example workflow"""
    print("\n" + "=" * 60)
    print("Example 3: Creating Workflow")
    print("=" * 60)

    # Create workflow
    workflow = Workflow(
        name="Data Processing Pipeline",
        description="Validates and processes numerical data",
        version="1.0.0",
        steps=[
            WorkflowStep(
                name="Validate Input",
                agent_id=validator_id,
                description="Validate input data",
                on_failure="stop",
                retry_count=0,
            ),
            WorkflowStep(
                name="Process Data",
                agent_id=agent_id,
                description="Process validated data",
                condition="results['Validate Input']['success'] == True",
                input_mapping={"data": "input.data"},
                on_failure="stop",
                retry_count=3,
            ),
        ],
        output_mapping={"result": "results.Process Data.data"},
    )

    workflow_id = "data-pipeline-1"
    orchestrator.register_workflow(workflow_id, workflow)

    print(f"Registered workflow: {workflow_id}")
    print(f"Workflow name: {workflow.name}")
    print(f"Steps: {len(workflow.steps)}")

    for step in workflow.steps:
        print(f"  - {step.name} (agent: {step.agent_id})")

    return workflow_id


# ==============================================================================
# Example 4: GraphQL Queries
# ==============================================================================

def example_graphql_queries():
    """Example GraphQL queries"""
    print("\n" + "=" * 60)
    print("Example 4: Example GraphQL Queries")
    print("=" * 60)

    queries = {
        "List Agents": """
query {
  agents(pagination: { page: 1, pageSize: 10 }) {
    nodes {
      id
      name
      description
      version
      enabled
      stats {
        executions
        successRate
        avgTimeMs
      }
    }
    pageInfo {
      totalCount
      hasNextPage
    }
  }
}
        """,
        "Get Workflow": """
query {
  workflow(id: "data-pipeline-1") {
    id
    name
    description
    steps {
      name
      agentId
      description
      onFailure
      retryCount
    }
  }
}
        """,
        "Execute Workflow": """
mutation {
  executeWorkflow(input: {
    workflowId: "data-pipeline-1"
    inputData: {
      data: [1, 2, 3, 4, 5]
    }
  }) {
    success
    execution {
      id
      status
      outputData
      stepResults {
        stepName
        status
      }
    }
    errors
  }
}
        """,
        "Watch Execution": """
subscription {
  executionUpdated(executionId: "exec-123") {
    event
    execution {
      id
      status
      completedAt
    }
    timestamp
  }
}
        """,
    }

    for name, query in queries.items():
        print(f"\n{name}:")
        print("-" * 40)
        print(query.strip())

    return queries


# ==============================================================================
# Example 5: Python Client Usage
# ==============================================================================

def example_python_client(token_value):
    """Example Python client"""
    print("\n" + "=" * 60)
    print("Example 5: Python Client Usage")
    print("=" * 60)

    client_code = f'''
import requests

# GraphQL endpoint
url = "http://localhost:8000/graphql"

# Headers with auth token
headers = {{
    "Authorization": "Bearer {token_value[:20]}...",
    "Content-Type": "application/json",
}}

# Query
query = """
query {{
  agents {{
    nodes {{
      id
      name
      version
    }}
  }}
}}
"""

# Execute query
response = requests.post(
    url,
    json={{"query": query}},
    headers=headers,
)

# Process result
data = response.json()
if "data" in data:
    agents = data["data"]["agents"]["nodes"]
    for agent in agents:
        print(f"Agent: {{agent['name']}} ({{agent['id']}})")
else:
    print(f"Error: {{data.get('errors')}}")
    '''

    print(client_code)


# ==============================================================================
# Example 6: Running the Server
# ==============================================================================

def example_run_server():
    """Example of running the server"""
    print("\n" + "=" * 60)
    print("Example 6: Running the Server")
    print("=" * 60)

    server_code = '''
from greenlang.api.graphql.server import run_dev_server
from greenlang.core.orchestrator import Orchestrator
from greenlang.auth.rbac import RBACManager
from greenlang.auth.auth import AuthManager

# Initialize services
orchestrator = Orchestrator()
auth_manager = AuthManager()
rbac_manager = RBACManager()

# Run development server
run_dev_server(
    orchestrator=orchestrator,
    auth_manager=auth_manager,
    rbac_manager=rbac_manager,
    host="0.0.0.0",
    port=8000,
    reload=True,
)

# Server will start at:
# - GraphQL endpoint: http://localhost:8000/graphql
# - Playground: http://localhost:8000/playground
# - Health check: http://localhost:8000/health
# - Metrics: http://localhost:8000/metrics
    '''

    print(server_code)

    print("\nTo run the server:")
    print("1. Save the code above to a file (e.g., server.py)")
    print("2. Run: python server.py")
    print("3. Open http://localhost:8000/playground in your browser")


# ==============================================================================
# Example 7: Using Subscriptions
# ==============================================================================

def example_subscriptions():
    """Example WebSocket subscriptions"""
    print("\n" + "=" * 60)
    print("Example 7: WebSocket Subscriptions")
    print("=" * 60)

    subscription_code = '''
from greenlang.api.graphql.subscriptions import (
    notify_execution_created,
    notify_execution_progress,
    notify_execution_status_changed,
)
from greenlang.api.graphql.types import ExecutionStatus

# After creating an execution
execution_id = "exec-123"
execution_obj = ...  # Your execution object

# Notify subscribers of new execution
await notify_execution_created(execution_id, execution_obj)

# Update progress
await notify_execution_progress(
    execution_id=execution_id,
    current_step="Process Data",
    completed_steps=1,
    total_steps=2,
    estimated_time=30.0,
)

# Update status
await notify_execution_status_changed(
    execution_id=execution_id,
    old_status=ExecutionStatus.RUNNING,
    new_status=ExecutionStatus.COMPLETED,
)
    '''

    print(subscription_code)


# ==============================================================================
# Main Example Runner
# ==============================================================================

def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("GreenLang GraphQL API - Example Usage")
    print("=" * 60)

    # Example 1: Setup
    app, orchestrator, auth_manager, rbac_manager, user_id, token = example_basic_setup()

    # Example 2: Create agents
    agent_id, validator_id = example_create_agents(orchestrator)

    # Example 3: Create workflow
    workflow_id = example_create_workflow(orchestrator, agent_id, validator_id)

    # Example 4: Show example queries
    queries = example_graphql_queries()

    # Example 5: Python client
    example_python_client(token.token_value)

    # Example 6: Running server
    example_run_server()

    # Example 7: Subscriptions
    example_subscriptions()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Created {len(orchestrator.list_agents())} agents")
    print(f"Created {len(orchestrator.list_workflows())} workflows")
    print(f"Created 1 user with 'developer' role")
    print(f"Auth token: {token.token_value[:20]}...")
    print("\nNext steps:")
    print("1. Run the server using example_run_server()")
    print("2. Visit http://localhost:8000/playground")
    print("3. Use the auth token in the Authorization header")
    print("4. Try the example queries!")

    print("\n" + "=" * 60)
    print("For more information, see:")
    print("- GRAPHQL_API.md for comprehensive documentation")
    print("- README.md for overview and features")
    print("- schema.graphql for complete schema definition")
    print("=" * 60)


if __name__ == "__main__":
    main()
