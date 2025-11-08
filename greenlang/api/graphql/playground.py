"""
GraphQL Playground Integration
Interactive API explorer with documentation and examples
"""

from typing import Optional, Dict, Any, List
import json


# ==============================================================================
# Example Queries and Mutations
# ==============================================================================

EXAMPLE_QUERIES = {
    "List Agents": """
# List all agents with pagination
query ListAgents {
  agents(
    pagination: { page: 1, pageSize: 10 }
    sort: [{ field: "name", order: ASC }]
  ) {
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
      currentPage
    }
  }
}
""",
    "Get Agent Details": """
# Get detailed information about a specific agent
query GetAgent($agentId: ID!) {
  agent(id: $agentId) {
    id
    name
    description
    version
    enabled
    parameters
    resourcePaths
    logLevel
    tags
    stats {
      executions
      successes
      failures
      successRate
      totalTimeMs
      avgTimeMs
    }
    createdAt
    updatedAt
  }
}

# Variables:
# {
#   "agentId": "your-agent-id"
# }
""",
    "List Workflows": """
# List workflows with their steps
query ListWorkflows {
  workflows(
    pagination: { page: 1, pageSize: 10 }
  ) {
    nodes {
      id
      name
      description
      version
      steps {
        name
        agentId
        description
        onFailure
        retryCount
      }
      tags
      createdAt
    }
    pageInfo {
      totalCount
      hasNextPage
    }
  }
}
""",
    "Get Execution History": """
# Get execution history with filtering
query GetExecutions($status: ExecutionStatus) {
  executions(
    pagination: { page: 1, pageSize: 20 }
    filter: { status: $status }
    sort: [{ field: "startedAt", order: DESC }]
  ) {
    nodes {
      id
      executionId
      status
      startedAt
      completedAt
      totalDuration
      stepResults {
        stepName
        status
        duration
      }
    }
    pageInfo {
      totalCount
      currentPage
    }
  }
}

# Variables:
# {
#   "status": "COMPLETED"
# }
""",
    "Current User Info": """
# Get current authenticated user information
query CurrentUser {
  currentUser {
    id
    username
    email
    active
    roles {
      name
      description
      permissions {
        resource
        action
        scope
      }
    }
    createdAt
  }
}
""",
    "System Health": """
# Check system health status
query SystemHealth {
  systemHealth {
    status
    version
    uptime
    agentCount
    workflowCount
    executionCount
    timestamp
    checks {
      name
      status
      message
      latency
    }
  }
}
""",
    "Check Permission": """
# Check if user has specific permission
query CheckPermission($resource: String!, $action: String!) {
  checkPermission(resource: $resource, action: $action)
}

# Variables:
# {
#   "resource": "agent",
#   "action": "create"
# }
""",
}

EXAMPLE_MUTATIONS = {
    "Create Agent": """
# Create a new agent
mutation CreateAgent($input: CreateAgentInput!) {
  createAgent(input: $input) {
    id
    name
    description
    version
    enabled
    createdAt
  }
}

# Variables:
# {
#   "input": {
#     "name": "My Custom Agent",
#     "description": "Description of the agent",
#     "version": "1.0.0",
#     "enabled": true,
#     "parameters": {},
#     "tags": ["custom", "demo"]
#   }
# }
""",
    "Update Agent": """
# Update an existing agent
mutation UpdateAgent($id: ID!, $input: UpdateAgentInput!) {
  updateAgent(id: $id, input: $input) {
    id
    name
    description
    version
    enabled
    updatedAt
  }
}

# Variables:
# {
#   "id": "agent-id",
#   "input": {
#     "description": "Updated description",
#     "enabled": false
#   }
# }
""",
    "Create Workflow": """
# Create a new workflow
mutation CreateWorkflow($input: CreateWorkflowInput!) {
  createWorkflow(input: $input) {
    id
    name
    description
    version
    steps {
      name
      agentId
    }
    createdAt
  }
}

# Variables:
# {
#   "input": {
#     "name": "My Workflow",
#     "description": "Workflow description",
#     "version": "1.0.0",
#     "steps": [
#       {
#         "name": "Step 1",
#         "agentId": "agent-id-1",
#         "description": "First step"
#       },
#       {
#         "name": "Step 2",
#         "agentId": "agent-id-2",
#         "description": "Second step",
#         "onFailure": "skip"
#       }
#     ],
#     "tags": ["demo"]
#   }
# }
""",
    "Execute Workflow": """
# Execute a workflow
mutation ExecuteWorkflow($input: ExecuteWorkflowInput!) {
  executeWorkflow(input: $input) {
    success
    execution {
      id
      executionId
      status
      startedAt
      stepResults {
        stepName
        status
      }
    }
    errors
  }
}

# Variables:
# {
#   "input": {
#     "workflowId": "workflow-id",
#     "inputData": {
#       "key": "value"
#     },
#     "tags": ["manual-execution"]
#   }
# }
""",
    "Execute Single Agent": """
# Execute a single agent
mutation ExecuteSingleAgent($input: ExecuteSingleAgentInput!) {
  executeSingleAgent(input: $input) {
    success
    execution {
      id
      executionId
      status
      outputData
    }
    errors
  }
}

# Variables:
# {
#   "input": {
#     "agentId": "agent-id",
#     "inputData": {
#       "param1": "value1"
#     }
#   }
# }
""",
    "Create Role": """
# Create a new RBAC role
mutation CreateRole($input: CreateRoleInput!) {
  createRole(input: $input) {
    name
    description
    permissions {
      resource
      action
      scope
    }
    createdAt
  }
}

# Variables:
# {
#   "input": {
#     "name": "custom_role",
#     "description": "Custom role for specific use case",
#     "permissions": [
#       {
#         "resource": "agent",
#         "action": "read"
#       },
#       {
#         "resource": "workflow",
#         "action": "execute"
#       }
#     ]
#   }
# }
""",
    "Assign Role": """
# Assign roles to a user
mutation AssignRole($input: AssignRoleInput!) {
  assignRole(input: $input) {
    id
    username
    roles {
      name
      description
    }
  }
}

# Variables:
# {
#   "input": {
#     "userId": "user-id",
#     "roleNames": ["developer", "operator"]
#   }
# }
""",
    "Create API Key": """
# Create a new API key
mutation CreateAPIKey($input: CreateAPIKeyInput!) {
  createAPIKey(input: $input) {
    keyId
    name
    displayKey
    scopes
    createdAt
    expiresAt
  }
}

# Variables:
# {
#   "input": {
#     "name": "Production API Key",
#     "description": "Key for production environment",
#     "scopes": ["agent:read", "workflow:execute"],
#     "expiresIn": 31536000,
#     "rateLimit": 1000
#   }
# }
""",
    "Batch Create Agents": """
# Create multiple agents at once
mutation BatchCreateAgents($inputs: [CreateAgentInput!]!) {
  batchCreateAgents(inputs: $inputs) {
    id
    name
    version
  }
}

# Variables:
# {
#   "inputs": [
#     {
#       "name": "Agent 1",
#       "description": "First agent",
#       "version": "1.0.0"
#     },
#     {
#       "name": "Agent 2",
#       "description": "Second agent",
#       "version": "1.0.0"
#     }
#   ]
# }
""",
}

EXAMPLE_SUBSCRIPTIONS = {
    "Watch Execution": """
# Subscribe to execution updates
subscription WatchExecution($executionId: ID!) {
  executionUpdated(executionId: $executionId) {
    event
    execution {
      id
      status
      completedAt
    }
    timestamp
  }
}

# Variables:
# {
#   "executionId": "execution-id"
# }
""",
    "Watch Execution Progress": """
# Subscribe to execution progress
subscription WatchProgress($executionId: ID!) {
  executionProgress(executionId: $executionId) {
    executionId
    currentStep
    completedSteps
    totalSteps
    progress
    estimatedTimeRemaining
    timestamp
  }
}

# Variables:
# {
#   "executionId": "execution-id"
# }
""",
    "Watch All Executions": """
# Subscribe to all execution updates
subscription WatchAllExecutions {
  executionUpdated {
    event
    execution {
      id
      executionId
      status
      workflowId
    }
    timestamp
  }
}
""",
    "Watch Agent Updates": """
# Subscribe to agent updates
subscription WatchAgents($agentId: ID) {
  agentUpdated(agentId: $agentId) {
    event
    agent {
      id
      name
      enabled
    }
    timestamp
  }
}

# Variables (optional):
# {
#   "agentId": "agent-id"
# }
""",
    "System Metrics Stream": """
# Subscribe to real-time system metrics
subscription SystemMetricsStream($interval: Int) {
  systemMetrics(interval: $interval) {
    cpuUsage
    memoryUsage
    activeExecutions
    requestsPerSecond
    timestamp
  }
}

# Variables:
# {
#   "interval": 5000
# }
""",
}


# ==============================================================================
# Playground Configuration
# ==============================================================================

def get_playground_config(
    endpoint: str = "/graphql",
    subscription_endpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get GraphQL Playground configuration

    Args:
        endpoint: GraphQL HTTP endpoint
        subscription_endpoint: GraphQL WebSocket endpoint

    Returns:
        Playground configuration
    """
    config = {
        "endpoint": endpoint,
        "subscriptionEndpoint": subscription_endpoint or endpoint.replace("http", "ws"),
        "settings": {
            "editor.theme": "dark",
            "editor.fontSize": 14,
            "editor.fontFamily": "'Source Code Pro', monospace",
            "editor.reuseHeaders": True,
            "request.credentials": "include",
            "tracing.hideTracingResponse": False,
            "prettier.printWidth": 80,
            "prettier.tabWidth": 2,
            "prettier.useTabs": False,
        },
        "tabs": [
            {
                "name": "Welcome",
                "endpoint": endpoint,
                "query": _get_welcome_query(),
            },
            {
                "name": "List Agents",
                "endpoint": endpoint,
                "query": EXAMPLE_QUERIES["List Agents"],
            },
            {
                "name": "Create Workflow",
                "endpoint": endpoint,
                "query": EXAMPLE_MUTATIONS["Create Workflow"],
            },
            {
                "name": "Watch Execution",
                "endpoint": subscription_endpoint or endpoint,
                "query": EXAMPLE_SUBSCRIPTIONS["Watch Execution"],
            },
        ],
    }

    return config


def _get_welcome_query() -> str:
    """Get welcome query for playground"""
    return """# Welcome to GreenLang GraphQL API!
#
# This is an interactive GraphQL playground where you can:
# - Explore the API schema
# - Run queries and mutations
# - Test subscriptions
# - View documentation
#
# Quick Start:
# 1. Check out the example tabs above
# 2. Press Ctrl+Space to see available fields
# 3. Use the "Docs" panel on the right to explore the schema
#
# Example: Get system health

query SystemHealth {
  systemHealth {
    status
    version
    agentCount
    workflowCount
    executionCount
    checks {
      name
      status
      message
    }
  }
}

# Try this query by clicking the Play button!
"""


# ==============================================================================
# Playground HTML
# ==============================================================================

def get_playground_html(
    endpoint: str = "/graphql",
    subscription_endpoint: Optional[str] = None,
    title: str = "GreenLang GraphQL Playground",
) -> str:
    """
    Get HTML for GraphQL Playground

    Args:
        endpoint: GraphQL HTTP endpoint
        subscription_endpoint: GraphQL WebSocket endpoint
        title: Page title

    Returns:
        HTML string
    """
    config = get_playground_config(endpoint, subscription_endpoint)
    config_json = json.dumps(config, indent=2)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      padding: 0;
      font-family: 'Source Code Pro', monospace;
    }}
    #root {{
      height: 100vh;
    }}
  </style>
  <link
    rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/graphql-playground-react/build/static/css/index.css"
  />
  <script
    src="https://cdn.jsdelivr.net/npm/graphql-playground-react/build/static/js/middleware.js"
  ></script>
</head>
<body>
  <div id="root"></div>
  <script>
    window.addEventListener('load', function (event) {{
      const config = {config_json};

      GraphQLPlayground.init(document.getElementById('root'), config);
    }})
  </script>
</body>
</html>
"""
    return html


# ==============================================================================
# Example Collection
# ==============================================================================

def get_all_examples() -> Dict[str, Dict[str, str]]:
    """
    Get all example operations

    Returns:
        Dictionary of examples by category
    """
    return {
        "queries": EXAMPLE_QUERIES,
        "mutations": EXAMPLE_MUTATIONS,
        "subscriptions": EXAMPLE_SUBSCRIPTIONS,
    }


def get_example(operation_type: str, name: str) -> Optional[str]:
    """
    Get specific example by type and name

    Args:
        operation_type: "queries", "mutations", or "subscriptions"
        name: Example name

    Returns:
        Example GraphQL operation string
    """
    examples = {
        "queries": EXAMPLE_QUERIES,
        "mutations": EXAMPLE_MUTATIONS,
        "subscriptions": EXAMPLE_SUBSCRIPTIONS,
    }

    return examples.get(operation_type, {}).get(name)


def list_examples(operation_type: Optional[str] = None) -> List[str]:
    """
    List available examples

    Args:
        operation_type: Filter by type (optional)

    Returns:
        List of example names
    """
    if operation_type:
        examples = {
            "queries": EXAMPLE_QUERIES,
            "mutations": EXAMPLE_MUTATIONS,
            "subscriptions": EXAMPLE_SUBSCRIPTIONS,
        }
        return list(examples.get(operation_type, {}).keys())
    else:
        return (
            list(EXAMPLE_QUERIES.keys())
            + list(EXAMPLE_MUTATIONS.keys())
            + list(EXAMPLE_SUBSCRIPTIONS.keys())
        )
