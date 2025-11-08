# GreenLang GraphQL API Documentation

## Overview

The GreenLang GraphQL API provides a comprehensive, type-safe interface for interacting with the GreenLang system. This API supports:

- Agent management and execution
- Workflow definition and orchestration
- Real-time execution monitoring via subscriptions
- Role-based access control (RBAC)
- API key management
- System monitoring and health checks

## Quick Start

### Starting the Server

```python
from greenlang.core.orchestrator import Orchestrator
from greenlang.auth.rbac import RBACManager
from greenlang.auth.auth import AuthManager
from greenlang.api.graphql.server import run_dev_server

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
)
```

### Accessing the API

- **GraphQL Endpoint**: `http://localhost:8000/graphql`
- **Playground**: `http://localhost:8000/playground`
- **Health Check**: `http://localhost:8000/health`
- **Metrics**: `http://localhost:8000/metrics`

## Authentication

All GraphQL requests require authentication using Bearer tokens:

```http
Authorization: Bearer <your-token>
```

### Getting an Auth Token

```python
from greenlang.auth.auth import AuthManager

auth_manager = AuthManager()

# Create user
user_id = auth_manager.create_user(
    tenant_id="my-tenant",
    username="myuser",
    email="user@example.com",
    password="securepassword123",
)

# Create token
token = auth_manager.create_token(
    tenant_id="my-tenant",
    user_id=user_id,
    name="My API Token",
)

print(f"Token: {token.token_value}")
```

## Core Concepts

### Agents

Agents are executable units that perform specific tasks. Each agent has:
- **Configuration**: Name, version, parameters
- **Execution logic**: Defined in the `execute()` method
- **Statistics**: Success rate, execution time, etc.

### Workflows

Workflows orchestrate multiple agents in sequence or conditionally:
- **Steps**: Ordered list of agent executions
- **Input mapping**: Map data between steps
- **Error handling**: Configure retry and failure behavior
- **Output mapping**: Define final workflow output

### Executions

Executions track the runtime state of workflows or agents:
- **Status**: PENDING, RUNNING, COMPLETED, FAILED
- **Results**: Output data from each step
- **Metrics**: Performance and timing information
- **Real-time updates**: Via GraphQL subscriptions

## API Reference

### Queries

#### List Agents

```graphql
query ListAgents {
  agents(
    pagination: { page: 1, pageSize: 20 }
    sort: [{ field: "name", order: ASC }]
    filter: { status: ENABLED }
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
```

#### Get Agent Details

```graphql
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
      customCounters
      customTimers
    }
    createdAt
    updatedAt
  }
}
```

#### List Workflows

```graphql
query ListWorkflows {
  workflows(
    pagination: { page: 1, pageSize: 20 }
    sort: [{ field: "createdAt", order: DESC }]
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
        condition
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
```

#### Get Execution History

```graphql
query GetExecutions($status: ExecutionStatus) {
  executions(
    pagination: { page: 1, pageSize: 50 }
    filter: { status: $status }
    sort: [{ field: "startedAt", order: DESC }]
  ) {
    nodes {
      id
      executionId
      status
      workflowId
      agentId
      userId
      startedAt
      completedAt
      totalDuration
      stepResults {
        stepName
        status
        error
        duration
        attempts
      }
      errors
    }
    pageInfo {
      totalCount
      currentPage
      totalPages
    }
  }
}
```

#### Current User

```graphql
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
    permissions {
      resource
      action
      scope
    }
    createdAt
  }
}
```

#### System Health

```graphql
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
```

#### Check Permission

```graphql
query CheckPermission($resource: String!, $action: String!) {
  checkPermission(
    resource: $resource
    action: $action
    context: { tenant: "my-tenant" }
  )
}
```

### Mutations

#### Create Agent

```graphql
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
```

**Variables:**
```json
{
  "input": {
    "name": "My Custom Agent",
    "description": "Performs custom data processing",
    "version": "1.0.0",
    "enabled": true,
    "parameters": {
      "timeout": 30,
      "retries": 3
    },
    "resourcePaths": ["/path/to/config.yaml"],
    "logLevel": "INFO",
    "tags": ["custom", "processing"]
  }
}
```

#### Update Agent

```graphql
mutation UpdateAgent($id: ID!, $input: UpdateAgentInput!) {
  updateAgent(id: $id, input: $input) {
    id
    name
    description
    enabled
    updatedAt
  }
}
```

**Variables:**
```json
{
  "id": "agent-id",
  "input": {
    "description": "Updated description",
    "enabled": false,
    "parameters": {
      "timeout": 60
    }
  }
}
```

#### Create Workflow

```graphql
mutation CreateWorkflow($input: CreateWorkflowInput!) {
  createWorkflow(input: $input) {
    id
    name
    description
    version
    steps {
      name
      agentId
      description
    }
    createdAt
  }
}
```

**Variables:**
```json
{
  "input": {
    "name": "Data Processing Pipeline",
    "description": "Multi-step data processing workflow",
    "version": "1.0.0",
    "steps": [
      {
        "name": "Extract Data",
        "agentId": "extractor-agent-id",
        "description": "Extract data from source",
        "onFailure": "stop",
        "retryCount": 3
      },
      {
        "name": "Transform Data",
        "agentId": "transformer-agent-id",
        "description": "Transform extracted data",
        "condition": "results.Extract_Data.success == true",
        "inputMapping": {
          "input": "results.Extract_Data.data"
        }
      },
      {
        "name": "Load Data",
        "agentId": "loader-agent-id",
        "description": "Load transformed data",
        "inputMapping": {
          "input": "results.Transform_Data.data"
        }
      }
    ],
    "outputMapping": {
      "result": "results.Load_Data.data"
    },
    "tags": ["etl", "production"]
  }
}
```

#### Execute Workflow

```graphql
mutation ExecuteWorkflow($input: ExecuteWorkflowInput!) {
  executeWorkflow(input: $input) {
    success
    execution {
      id
      executionId
      status
      startedAt
      completedAt
      stepResults {
        stepName
        status
        result
        error
      }
      outputData
    }
    errors
  }
}
```

**Variables:**
```json
{
  "input": {
    "workflowId": "workflow-id",
    "inputData": {
      "source": "database",
      "query": "SELECT * FROM users"
    },
    "context": {
      "environment": "production",
      "tenant": "customer-1"
    },
    "tags": ["manual-execution"]
  }
}
```

#### Execute Single Agent

```graphql
mutation ExecuteSingleAgent($input: ExecuteSingleAgentInput!) {
  executeSingleAgent(input: $input) {
    success
    execution {
      id
      status
      outputData
      totalDuration
    }
    errors
  }
}
```

**Variables:**
```json
{
  "input": {
    "agentId": "agent-id",
    "inputData": {
      "param1": "value1",
      "param2": 123
    }
  }
}
```

#### Create RBAC Role

```graphql
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
```

**Variables:**
```json
{
  "input": {
    "name": "data_analyst",
    "description": "Data analyst with read access",
    "permissions": [
      {
        "resource": "agent",
        "action": "read"
      },
      {
        "resource": "workflow",
        "action": "read"
      },
      {
        "resource": "execution",
        "action": "read"
      },
      {
        "resource": "workflow",
        "action": "execute",
        "scope": "tenant:customer-1"
      }
    ],
    "parentRoles": ["viewer"]
  }
}
```

#### Assign Role to User

```graphql
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
```

**Variables:**
```json
{
  "input": {
    "userId": "user-id",
    "roleNames": ["data_analyst", "developer"]
  }
}
```

#### Create API Key

```graphql
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
```

**Variables:**
```json
{
  "input": {
    "name": "Production API Key",
    "description": "Key for production integration",
    "scopes": ["agent:read", "workflow:execute"],
    "expiresIn": 31536000,
    "allowedIps": ["192.168.1.0/24"],
    "rateLimit": 1000
  }
}
```

### Subscriptions

#### Watch Execution Updates

```graphql
subscription WatchExecution($executionId: ID!) {
  executionUpdated(executionId: $executionId) {
    event
    execution {
      id
      status
      completedAt
      stepResults {
        stepName
        status
      }
    }
    timestamp
  }
}
```

#### Watch Execution Progress

```graphql
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
```

#### Watch All Executions

```graphql
subscription WatchAllExecutions {
  executionUpdated {
    event
    execution {
      id
      executionId
      status
      workflowId
      agentId
    }
    timestamp
  }
}
```

#### Watch Agent Updates

```graphql
subscription WatchAgents($agentId: ID) {
  agentUpdated(agentId: $agentId) {
    event
    agent {
      id
      name
      enabled
      stats {
        executions
        successRate
      }
    }
    timestamp
  }
}
```

#### System Metrics Stream

```graphql
subscription SystemMetricsStream($interval: Int) {
  systemMetrics(interval: $interval) {
    cpuUsage
    memoryUsage
    activeExecutions
    requestsPerSecond
    timestamp
  }
}
```

**Variables:**
```json
{
  "interval": 5000
}
```

## Advanced Features

### Query Complexity Analysis

The API enforces query complexity limits to prevent expensive operations:

- **Max Depth**: 10 levels (configurable)
- **Max Complexity**: 1000 points (configurable)
- **Field Costs**: Different fields have different costs
- **List Multipliers**: List fields multiply complexity by size

**Example**: Complex nested query

```graphql
query {
  agents(pagination: { page: 1, pageSize: 100 }) {  # Cost: 100 * field_cost
    nodes {
      id
      workflows {  # Cost multiplied by agent count
        nodes {
          steps {  # Additional nesting adds complexity
            agent {
              workflows {  # Deep nesting increases cost
                nodes {
                  id
                }
              }
            }
          }
        }
      }
    }
  }
}
```

### DataLoader for N+1 Prevention

The API uses DataLoader for efficient data fetching:

```graphql
query {
  workflows {
    nodes {
      id
      steps {
        agentId
        agent {  # Batched loading prevents N+1
          name
          version
        }
      }
    }
  }
}
```

All `agent` fields are loaded in a single batch operation.

### Pagination

All list queries support cursor-based and offset-based pagination:

```graphql
query {
  agents(
    pagination: {
      page: 2          # Page number (offset-based)
      pageSize: 20     # Items per page
      offset: 40       # Alternative: direct offset
      limit: 20        # Alternative: direct limit
    }
  ) {
    nodes {
      id
    }
    pageInfo {
      totalCount
      totalPages
      currentPage
      hasNextPage
      hasPreviousPage
      startCursor
      endCursor
    }
  }
}
```

### Filtering and Sorting

Advanced filtering and sorting capabilities:

```graphql
query {
  executions(
    filter: {
      status: COMPLETED
      startedAfter: "2024-01-01T00:00:00Z"
      startedBefore: "2024-12-31T23:59:59Z"
      workflowId: "workflow-123"
    }
    sort: [
      { field: "startedAt", order: DESC }
      { field: "totalDuration", order: ASC }
    ]
  ) {
    nodes {
      id
      status
      totalDuration
    }
  }
}
```

## Error Handling

### Error Types

The API returns structured errors with the following codes:

- **FORBIDDEN**: Insufficient permissions
- **BAD_REQUEST**: Invalid input or validation error
- **NOT_FOUND**: Resource not found
- **INTERNAL_ERROR**: Server error

### Example Error Response

```json
{
  "errors": [
    {
      "message": "User lacks permission: agent:create",
      "extensions": {
        "code": "FORBIDDEN"
      }
    }
  ]
}
```

## Performance Best Practices

### 1. Use Field Selection

Only request fields you need:

```graphql
# Good: Minimal fields
query {
  agents {
    nodes {
      id
      name
    }
  }
}

# Bad: Requesting all fields
query {
  agents {
    nodes {
      id
      name
      description
      version
      enabled
      parameters
      resourcePaths
      logLevel
      tags
      metadata
      stats {
        # ... all stats
      }
    }
  }
}
```

### 2. Limit Pagination

Use appropriate page sizes:

```graphql
# Good: Reasonable page size
agents(pagination: { pageSize: 20 })

# Bad: Too large
agents(pagination: { pageSize: 1000 })
```

### 3. Avoid Deep Nesting

Limit query depth:

```graphql
# Good: Shallow query
query {
  workflow(id: "123") {
    steps {
      agentId
    }
  }
}

# Bad: Deep nesting
query {
  workflows {
    nodes {
      steps {
        agent {
          workflows {
            nodes {
              steps {
                agent {
                  # Too deep!
                }
              }
            }
          }
        }
      }
    }
  }
}
```

### 4. Use Subscriptions Wisely

Subscriptions maintain WebSocket connections. Use them only when needed:

```graphql
# Good: Specific execution
subscription {
  executionUpdated(executionId: "exec-123") {
    event
    execution {
      status
    }
  }
}

# Bad: All executions (high volume)
subscription {
  executionUpdated {
    # Receives updates for ALL executions
  }
}
```

## Security

### RBAC Integration

All operations respect role-based access control:

```graphql
# Only works if user has required permissions
mutation {
  createAgent(input: { name: "New Agent", ... }) {
    id
  }
}
# Requires: agent:create permission
```

### API Key Management

Create and manage API keys:

```graphql
mutation {
  createAPIKey(input: {
    name: "Service Account Key"
    scopes: ["workflow:execute"]
    allowedIps: ["10.0.0.0/8"]
    rateLimit: 100
  }) {
    keyId
    displayKey
  }
}
```

### Rate Limiting

API keys support rate limiting (requests per hour):

```graphql
mutation {
  createAPIKey(input: {
    name: "Limited Key"
    rateLimit: 1000  # 1000 requests per hour
  }) {
    keyId
  }
}
```

## Monitoring and Observability

### Health Checks

```bash
curl http://localhost:8000/health
```

### Prometheus Metrics

```bash
curl http://localhost:8000/metrics
```

Returns:
```json
{
  "agents_total": 42,
  "workflows_total": 15,
  "executions_total": 1234,
  "subscriptions_active": 5,
  "subscriptions_by_type": {
    "executions": 3,
    "agents": 1,
    "workflows": 1,
    "system": 0
  }
}
```

### Real-time Metrics

```graphql
subscription {
  systemMetrics(interval: 5000) {
    cpuUsage
    memoryUsage
    activeExecutions
    requestsPerSecond
    timestamp
  }
}
```

## Client Examples

### Python Client

```python
import requests

# GraphQL endpoint
url = "http://localhost:8000/graphql"

# Headers with auth token
headers = {
    "Authorization": "Bearer your-token-here",
    "Content-Type": "application/json",
}

# Query
query = """
query {
  agents {
    nodes {
      id
      name
    }
  }
}
"""

# Execute query
response = requests.post(
    url,
    json={"query": query},
    headers=headers,
)

# Process result
data = response.json()
agents = data["data"]["agents"]["nodes"]
for agent in agents:
    print(f"Agent: {agent['name']} ({agent['id']})")
```

### JavaScript Client

```javascript
// Using Apollo Client
import { ApolloClient, InMemoryCache, gql } from '@apollo/client';

const client = new ApolloClient({
  uri: 'http://localhost:8000/graphql',
  cache: new InMemoryCache(),
  headers: {
    Authorization: `Bearer ${token}`,
  },
});

// Execute query
const { data } = await client.query({
  query: gql`
    query {
      agents {
        nodes {
          id
          name
        }
      }
    }
  `,
});

console.log(data.agents.nodes);
```

### WebSocket Subscriptions (JavaScript)

```javascript
import { createClient } from 'graphql-ws';

const client = createClient({
  url: 'ws://localhost:8000/graphql',
  connectionParams: {
    authToken: token,
  },
});

// Subscribe to execution updates
client.subscribe(
  {
    query: `
      subscription {
        executionUpdated(executionId: "exec-123") {
          event
          execution {
            status
          }
        }
      }
    `,
  },
  {
    next: (data) => {
      console.log('Update:', data);
    },
    error: (error) => {
      console.error('Error:', error);
    },
    complete: () => {
      console.log('Subscription complete');
    },
  }
);
```

## Deployment

### Production Configuration

```python
from greenlang.api.graphql.server import create_production_app
from greenlang.api.graphql.complexity import ComplexityConfig

# Configure complexity limits
complexity_config = ComplexityConfig(
    max_depth=8,
    max_complexity=500,
    enable_introspection_limit=True,
)

# Create production app
app = create_production_app(
    orchestrator=orchestrator,
    auth_manager=auth_manager,
    rbac_manager=rbac_manager,
    complexity_config=complexity_config,
    cors_origins=["https://app.example.com"],
)
```

### Running with Uvicorn

```bash
uvicorn greenlang.api.graphql.server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "greenlang.api.graphql.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Troubleshooting

### Common Issues

**Issue**: Authentication errors
```
Error: No authentication token provided
```
**Solution**: Include `Authorization: Bearer <token>` header

**Issue**: Permission denied
```
Error: User lacks permission: agent:create
```
**Solution**: Assign appropriate role to user

**Issue**: Query too complex
```
Error: Query complexity 1500 exceeds maximum 1000
```
**Solution**: Reduce query depth or pagination size

**Issue**: WebSocket connection fails
```
Error: WebSocket connection failed
```
**Solution**: Ensure WebSocket endpoint is accessible and auth token is valid

## Support and Resources

- **GitHub**: [https://github.com/akshay-greenlang/Code-V1_GreenLang](https://github.com/akshay-greenlang/Code-V1_GreenLang)
- **GraphQL Playground**: `http://localhost:8000/playground`
- **Schema Documentation**: Available in Playground's "Docs" panel

## Changelog

### Version 1.0.0 (Phase 4)

- Initial GraphQL API release
- Comprehensive schema for agents, workflows, executions
- DataLoader integration for N+1 prevention
- Real-time subscriptions via WebSocket
- Query complexity analysis
- RBAC integration
- GraphQL Playground
- Full test coverage
