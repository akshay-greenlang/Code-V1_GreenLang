# GreenLang GraphQL Schema Documentation

**Version:** 1.0.0
**Last Updated:** 2025-11-21

## Overview

The GreenLang GraphQL API provides a comprehensive, type-safe interface for managing climate compliance agents, workflows, and executions. Built on GraphQL, it offers:

- **Strongly Typed**: Full TypeScript/GraphQL type safety
- **Flexible Queries**: Request exactly the data you need
- **Real-time Updates**: WebSocket subscriptions for live execution monitoring
- **Batch Operations**: Efficient bulk operations
- **RBAC Integration**: Role-based access control on all operations
- **Relay Pagination**: Cursor-based pagination for efficient data fetching

## GraphQL Endpoint

```
POST https://api.greenlang.io/graphql
```

### Authentication

Include JWT token in Authorization header:

```
Authorization: Bearer <jwt-token>
```

### GraphQL Playground

Interactive API explorer available at:

```
https://api.greenlang.io/graphql/playground
```

## Core Types

### Agent

Represents a climate compliance agent (calculator, validator, reporter, etc.)

```graphql
type Agent {
  id: ID!
  name: String!
  description: String!
  version: String!
  enabled: Boolean!
  parameters: JSON
  resourcePaths: [String!]!
  logLevel: String!
  tags: [String!]!
  metadata: JSON
  stats: AgentStats!
  createdAt: DateTime!
  updatedAt: DateTime!

  # Relationships
  workflows: [Workflow!]!
  executions(
    pagination: PaginationInput
    filter: ExecutionFilterInput
    sort: [SortInput!]
  ): ExecutionConnection!
}
```

**Example Query:**

```graphql
query GetAgent {
  agent(id: "agent_123") {
    id
    name
    version
    enabled
    stats {
      executions
      successRate
      avgTimeMs
    }
    executions(pagination: { limit: 10 }) {
      nodes {
        id
        status
        startedAt
      }
    }
  }
}
```

### Workflow

Multi-agent pipeline for complex compliance tasks

```graphql
type Workflow {
  id: ID!
  name: String!
  description: String!
  version: String!
  steps: [WorkflowStep!]!
  outputMapping: JSON
  metadata: JSON
  tags: [String!]!
  createdAt: DateTime!
  updatedAt: DateTime!

  # Relationships
  agents: [Agent!]!
  executions(
    pagination: PaginationInput
    filter: ExecutionFilterInput
    sort: [SortInput!]
  ): ExecutionConnection!
}
```

**Example Query:**

```graphql
query GetWorkflow {
  workflow(id: "workflow_456") {
    id
    name
    steps {
      name
      agentId
      description
      inputMapping
      outputKey
    }
    agents {
      id
      name
      version
    }
  }
}
```

### Execution

Agent or workflow execution instance

```graphql
type Execution {
  id: ID!
  executionId: String!
  workflowId: ID
  workflow: Workflow
  agentId: ID
  agent: Agent
  userId: ID
  user: User
  status: ExecutionStatus!
  inputData: JSON!
  outputData: JSON
  context: JSON
  errors: [JSON!]!
  tags: [String!]!

  # Step results
  stepResults: [ExecutionStepResult!]!

  # Metrics
  totalDuration: Float
  startedAt: DateTime!
  completedAt: DateTime

  # Metadata
  metadata: JSON
  createdAt: DateTime!
  updatedAt: DateTime!
}
```

**Example Query:**

```graphql
query GetExecution {
  execution(id: "exec_789") {
    id
    status
    startedAt
    completedAt
    totalDuration
    stepResults {
      stepName
      status
      result
      duration
    }
    workflow {
      name
    }
    user {
      email
    }
  }
}
```

## Queries

### Agent Queries

#### Get Single Agent

```graphql
query {
  agent(id: "agent_123") {
    id
    name
    version
    enabled
    stats {
      executions
      successRate
      avgTimeMs
    }
  }
}
```

#### List Agents with Filtering

```graphql
query {
  agents(
    pagination: { page: 1, pageSize: 20 }
    filter: {
      status: ENABLED
      tags: ["carbon", "calculator"]
      createdAfter: "2024-01-01T00:00:00Z"
    }
    sort: [
      { field: "name", order: ASC }
    ]
  ) {
    edges {
      node {
        id
        name
        version
        tags
      }
      cursor
    }
    pageInfo {
      hasNextPage
      hasPreviousPage
      totalCount
      totalPages
    }
  }
}
```

### Workflow Queries

#### Get Workflow with Executions

```graphql
query {
  workflow(id: "workflow_456") {
    id
    name
    version
    steps {
      name
      agentId
      agent {
        name
        version
      }
      inputMapping
      outputKey
    }
    executions(
      pagination: { limit: 10 }
      filter: { status: COMPLETED }
    ) {
      nodes {
        id
        status
        startedAt
        totalDuration
      }
      pageInfo {
        totalCount
      }
    }
  }
}
```

### Execution Queries

#### Get Execution with Details

```graphql
query {
  execution(id: "exec_789") {
    id
    executionId
    status
    startedAt
    completedAt
    totalDuration

    inputData
    outputData

    workflow {
      name
      version
    }

    stepResults {
      stepName
      agentId
      status
      result
      error
      metrics {
        executionTimeMs
        recordsProcessed
      }
      startedAt
      completedAt
      duration
    }

    errors
    tags
  }
}
```

#### List Executions with Advanced Filtering

```graphql
query {
  executions(
    pagination: { page: 1, pageSize: 50 }
    filter: {
      status: COMPLETED
      workflowId: "workflow_456"
      startedAfter: "2024-11-01T00:00:00Z"
      startedBefore: "2024-11-30T23:59:59Z"
    }
    sort: [
      { field: "startedAt", order: DESC }
    ]
  ) {
    edges {
      node {
        id
        status
        startedAt
        totalDuration
        workflow {
          name
        }
      }
    }
    pageInfo {
      totalCount
      currentPage
      totalPages
    }
  }
}
```

### RBAC Queries

#### Get User with Roles and Permissions

```graphql
query {
  currentUser {
    id
    username
    email
    roles {
      name
      description
      permissions {
        resource
        action
        scope
      }
    }
  }
}
```

#### Check Permission

```graphql
query {
  checkPermission(
    resource: "workflow"
    action: "EXECUTE"
    context: { workflowId: "workflow_456" }
  )
}
```

### System Queries

#### System Health

```graphql
query {
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

#### Metrics

```graphql
query {
  metrics(
    timeRange: {
      start: "2024-11-01T00:00:00Z"
      end: "2024-11-21T23:59:59Z"
    }
    aggregation: AVG
  ) {
    name
    value
    unit
    timestamp
    labels
  }
}
```

## Mutations

### Agent Mutations

#### Create Agent

```graphql
mutation {
  createAgent(
    input: {
      name: "CO2 Calculator"
      description: "Calculates carbon emissions"
      version: "1.0.0"
      enabled: true
      parameters: {
        gwp_set: "IPCC_AR6_100"
        precision: "high"
      }
      resourcePaths: ["/data/emission_factors"]
      logLevel: "INFO"
      tags: ["carbon", "calculator", "scope1"]
    }
  ) {
    id
    name
    version
    enabled
  }
}
```

#### Update Agent

```graphql
mutation {
  updateAgent(
    id: "agent_123"
    input: {
      enabled: false
      logLevel: "DEBUG"
      tags: ["carbon", "calculator", "scope1", "deprecated"]
    }
  ) {
    id
    enabled
    logLevel
    tags
  }
}
```

#### Delete Agent

```graphql
mutation {
  deleteAgent(id: "agent_123")
}
```

### Workflow Mutations

#### Create Workflow

```graphql
mutation {
  createWorkflow(
    input: {
      name: "CBAM Pipeline"
      description: "EU CBAM reporting workflow"
      version: "1.0.0"
      steps: [
        {
          name: "Intake"
          agentId: "agent_intake"
          description: "Validate shipment data"
          outputKey: "validated_data"
        }
        {
          name: "Calculate"
          agentId: "agent_calc"
          description: "Calculate emissions"
          inputMapping: {
            shipments: "$.validated_data"
          }
          outputKey: "emissions"
        }
        {
          name: "Report"
          agentId: "agent_report"
          description: "Generate CBAM report"
          inputMapping: {
            emissions: "$.emissions"
          }
          outputKey: "report"
        }
      ]
      outputMapping: {
        report: "$.report"
        total_emissions: "$.emissions.total"
      }
      tags: ["cbam", "eu", "reporting"]
    }
  ) {
    id
    name
    version
    steps {
      name
      agentId
    }
  }
}
```

#### Clone Workflow

```graphql
mutation {
  cloneWorkflow(
    id: "workflow_456"
    name: "CBAM Pipeline - Copy"
  ) {
    id
    name
    version
  }
}
```

### Execution Mutations

#### Execute Workflow

```graphql
mutation {
  executeWorkflow(
    input: {
      workflowId: "workflow_456"
      inputData: {
        shipments: [
          {
            product: "steel"
            quantity: 1000
            origin: "CN"
          }
        ]
      }
      context: {
        user_id: "user_123"
        tenant_id: "tenant_456"
      }
      tags: ["production", "q4-2024"]
    }
  ) {
    success
    execution {
      id
      executionId
      status
      startedAt
    }
    errors
  }
}
```

#### Execute Single Agent

```graphql
mutation {
  executeSingleAgent(
    input: {
      agentId: "agent_calc"
      inputData: {
        fuel_type: "diesel"
        amount: 100
        unit: "gallons"
      }
      context: {
        geography: "US"
      }
      tags: ["test", "standalone"]
    }
  ) {
    success
    execution {
      id
      status
      outputData
    }
    errors
  }
}
```

#### Cancel Execution

```graphql
mutation {
  cancelExecution(id: "exec_789") {
    id
    status
  }
}
```

#### Retry Execution

```graphql
mutation {
  retryExecution(id: "exec_789") {
    success
    execution {
      id
      status
      startedAt
    }
    errors
  }
}
```

### RBAC Mutations

#### Create Role

```graphql
mutation {
  createRole(
    input: {
      name: "workflow_executor"
      description: "Can execute workflows"
      permissions: [
        {
          resource: "workflow"
          action: "EXECUTE"
          scope: "tenant"
        }
        {
          resource: "execution"
          action: "READ"
          scope: "own"
        }
      ]
      parentRoles: ["authenticated_user"]
    }
  ) {
    name
    description
    permissions {
      resource
      action
      scope
    }
  }
}
```

#### Assign Role

```graphql
mutation {
  assignRole(
    input: {
      userId: "user_123"
      roleNames: ["workflow_executor", "data_viewer"]
    }
  ) {
    id
    username
    roles {
      name
    }
  }
}
```

### Batch Mutations

#### Batch Create Agents

```graphql
mutation {
  batchCreateAgents(
    inputs: [
      {
        name: "Scope 1 Calculator"
        description: "Direct emissions"
        version: "1.0.0"
        tags: ["carbon", "scope1"]
      }
      {
        name: "Scope 2 Calculator"
        description: "Electricity emissions"
        version: "1.0.0"
        tags: ["carbon", "scope2"]
      }
    ]
  ) {
    id
    name
    version
  }
}
```

#### Batch Delete Executions

```graphql
mutation {
  batchDeleteExecutions(
    ids: ["exec_001", "exec_002", "exec_003"]
  )
}
```

## Subscriptions

### Execution Updates

#### Subscribe to Execution Updates

```graphql
subscription {
  executionUpdated(workflowId: "workflow_456") {
    event
    execution {
      id
      status
      startedAt
      completedAt
    }
    timestamp
  }
}
```

#### Subscribe to Status Changes

```graphql
subscription {
  executionStatusChanged(executionId: "exec_789") {
    executionId
    oldStatus
    newStatus
    timestamp
  }
}
```

#### Subscribe to Execution Progress

```graphql
subscription {
  executionProgress(executionId: "exec_789") {
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

### Agent Updates

```graphql
subscription {
  agentUpdated(agentId: "agent_123") {
    event
    agent {
      id
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

### System Metrics

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

## Directives

### @auth

Requires specific permission to access field

```graphql
type Mutation {
  deleteWorkflow(id: ID!): Boolean! @auth(requires: DELETE)
}
```

### @rateLimit

Rate limit specific field

```graphql
type Mutation {
  executeWorkflow(input: ExecuteWorkflowInput!): ExecutionResult!
    @rateLimit(max: 100, window: 60)
}
```

### @complexity

Query complexity calculation

```graphql
type Agent {
  executions: ExecutionConnection!
    @complexity(value: 10, multipliers: ["first", "last"])
}
```

## Pagination

GreenLang GraphQL uses Relay-style cursor pagination:

### Connection Pattern

```graphql
query {
  agents(pagination: { page: 1, pageSize: 20 }) {
    edges {
      node {
        id
        name
      }
      cursor
    }
    pageInfo {
      hasNextPage
      hasPreviousPage
      startCursor
      endCursor
      totalCount
      totalPages
      currentPage
    }
  }
}
```

### Offset Pagination

```graphql
query {
  agents(pagination: { offset: 40, limit: 20 }) {
    nodes {
      id
      name
    }
    pageInfo {
      totalCount
    }
  }
}
```

## Error Handling

GraphQL errors follow this structure:

```json
{
  "errors": [
    {
      "message": "Agent not found",
      "locations": [{ "line": 2, "column": 3 }],
      "path": ["agent"],
      "extensions": {
        "code": "NOT_FOUND",
        "agentId": "agent_invalid"
      }
    }
  ],
  "data": {
    "agent": null
  }
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `UNAUTHENTICATED` | No valid authentication token |
| `FORBIDDEN` | Insufficient permissions |
| `NOT_FOUND` | Resource not found |
| `BAD_USER_INPUT` | Invalid input data |
| `INTERNAL_SERVER_ERROR` | Unexpected server error |
| `RATE_LIMITED` | Rate limit exceeded |
| `COMPLEXITY_TOO_HIGH` | Query complexity exceeds limit |

## Best Practices

### 1. Request Only Needed Fields

```graphql
# Good
query {
  agents {
    nodes {
      id
      name
    }
  }
}

# Avoid
query {
  agents {
    nodes {
      # All fields
    }
  }
}
```

### 2. Use Fragments for Reusability

```graphql
fragment AgentDetails on Agent {
  id
  name
  version
  enabled
  tags
}

query {
  agent(id: "agent_123") {
    ...AgentDetails
    stats {
      executions
      successRate
    }
  }
}
```

### 3. Implement Pagination

```graphql
query {
  agents(pagination: { pageSize: 50 }) {
    edges {
      node {
        id
        name
      }
    }
    pageInfo {
      hasNextPage
      totalPages
    }
  }
}
```

### 4. Use Variables

```graphql
query GetAgent($id: ID!) {
  agent(id: $id) {
    id
    name
    version
  }
}

# Variables
{
  "id": "agent_123"
}
```

### 5. Handle Errors Gracefully

```javascript
const result = await client.query({
  query: GET_AGENT,
  variables: { id: "agent_123" }
})

if (result.errors) {
  // Handle GraphQL errors
  console.error(result.errors)
}

if (result.data) {
  // Process successful response
  console.log(result.data.agent)
}
```

## Schema Introspection

Query the GraphQL schema itself:

```graphql
query {
  __schema {
    types {
      name
      description
    }
  }
}
```

```graphql
query {
  __type(name: "Agent") {
    name
    fields {
      name
      type {
        name
        kind
      }
    }
  }
}
```

## Rate Limits

| Operation Type | Rate Limit |
|----------------|-----------|
| Queries | 1000/minute |
| Mutations | 500/minute |
| Subscriptions | 100 concurrent |
| Query Complexity | Max 1000 points |

## Resources

- **GraphQL Playground**: https://api.greenlang.io/graphql/playground
- **Schema SDL**: https://api.greenlang.io/graphql/schema.graphql
- **Documentation**: https://docs.greenlang.io/api/graphql
- **Support**: support@greenlang.io
