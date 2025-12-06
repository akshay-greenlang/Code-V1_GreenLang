# Process Heat GraphQL Schema - Usage Guide

Complete guide for using the GraphQL schema and resolvers for Process Heat agents.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Schema Overview](#schema-overview)
3. [Queries](#queries)
4. [Mutations](#mutations)
5. [Subscriptions](#subscriptions)
6. [Integration Examples](#integration-examples)
7. [Error Handling](#error-handling)
8. [Performance Tips](#performance-tips)

## Quick Start

### 1. Installation

```bash
pip install strawberry-graphql[fastapi]
pip install fastapi uvicorn
```

### 2. FastAPI Setup

```python
from fastapi import FastAPI
from greenlang.infrastructure.api.graphql_integration import (
    setup_graphql, GraphQLConfig
)

app = FastAPI(title="Process Heat GraphQL API")

# Configure and setup GraphQL
config = GraphQLConfig(path="/graphql")
setup_graphql(app, config)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3. Access GraphQL Playground

Start the server and navigate to: http://localhost:8000/graphql

## Schema Overview

The GraphQL schema provides comprehensive type definitions and resolvers for:

- **Agents**: Process heat agents with status, metrics, and configuration
- **Emissions**: GHG emission calculation results with provenance tracking
- **Jobs**: Asynchronous calculation jobs with progress tracking
- **Reports**: Regulatory compliance reports with findings

### Core Types

```graphql
type ProcessHeatAgent {
    id: ID!
    name: String!
    agentType: String!
    status: String!
    enabled: Boolean!
    version: String!
    lastRun: DateTime
    nextRun: DateTime
    metrics: AgentMetricsType!
    errorMessage: String
    createdAt: DateTime!
    updatedAt: DateTime!
}

type EmissionResult {
    id: ID!
    facilityId: String!
    co2Tonnes: Float!
    ch4Tonnes: Float!
    n2oTonnes: Float!
    totalCo2eTonnes: Float!
    provenanceHash: String!
    calculationMethod: String!
    timestamp: DateTime!
    confidenceScore: Float!
}

type CalculationJob {
    id: ID!
    status: String!
    progressPercent: Int!
    agentId: String!
    inputSummary: String!
    results: [EmissionResult!]
    errorDetails: String
    executionTimeMs: Float!
    createdAt: DateTime!
    startedAt: DateTime
    completedAt: DateTime
}

type ComplianceReport {
    id: ID!
    reportType: String!
    status: String!
    periodStart: Date!
    periodEnd: Date!
    findings: [ComplianceFinding!]!
    summary: String!
    actionItemsCount: Int!
    generatedAt: DateTime!
}
```

## Queries

### 1. List All Agents

```graphql
query {
    agents {
        id
        name
        agentType
        status
        enabled
        version
        lastRun
        metrics {
            executionTimeMs
            memoryUsageMb
            recordsProcessed
            processingRate
            cacheHitRatio
            errorCount
        }
    }
}
```

**Response:**

```json
{
    "data": {
        "agents": [
            {
                "id": "agent-001",
                "name": "Thermal Command",
                "agentType": "GL-001",
                "status": "idle",
                "enabled": true,
                "version": "1.0.0",
                "lastRun": "2025-12-06T10:30:00Z",
                "metrics": {
                    "executionTimeMs": 1234.5,
                    "memoryUsageMb": 256.2,
                    "recordsProcessed": 15000,
                    "processingRate": 1250.0,
                    "cacheHitRatio": 0.85,
                    "errorCount": 0
                }
            }
        ]
    }
}
```

### 2. Get Agent by Status

```graphql
query {
    agents(status: "running") {
        id
        name
        status
        lastRun
    }
}
```

### 3. Get Single Agent

```graphql
query {
    agent(id: "agent-001") {
        id
        name
        agentType
        status
        enabled
        metrics {
            errorCount
            executionTimeMs
        }
        errorMessage
    }
}
```

### 4. Query Emissions

```graphql
query {
    emissions(
        facilityId: "facility-1"
        dateRange: {
            startDate: "2025-01-01"
            endDate: "2025-03-31"
        }
    ) {
        id
        facilityId
        co2Tonnes
        ch4Tonnes
        n2oTonnes
        totalCo2eTonnes
        provenanceHash
        calculationMethod
        timestamp
        confidenceScore
    }
}
```

### 5. Get Calculation Jobs

```graphql
query {
    jobs(status: "completed") {
        id
        status
        progressPercent
        agentId
        inputSummary
        results {
            id
            co2Tonnes
            totalCo2eTonnes
            provenanceHash
        }
        executionTimeMs
        createdAt
        completedAt
    }
}
```

### 6. Get Compliance Reports

```graphql
query {
    complianceReports(reportType: "ghg_emissions") {
        id
        reportType
        status
        periodStart
        periodEnd
        findings {
            id
            category
            severity
            description
            remediationAction
            deadline
        }
        summary
        actionItemsCount
        generatedAt
    }
}
```

## Mutations

### 1. Start a Calculation Job

```graphql
mutation {
    runCalculation(input: {
        agentId: "agent-001"
        facilityId: "facility-1"
        dateRange: {
            startDate: "2025-01-01"
            endDate: "2025-03-31"
        }
        priority: "high"
    }) {
        id
        status
        progressPercent
        agentId
        createdAt
    }
}
```

**Response:**

```json
{
    "data": {
        "runCalculation": {
            "id": "job-a1b2c3d4",
            "status": "pending",
            "progressPercent": 0,
            "agentId": "agent-001",
            "createdAt": "2025-12-06T10:35:00Z"
        }
    }
}
```

### 2. Update Agent Configuration

```graphql
mutation {
    updateAgentConfig(
        id: "agent-001"
        config: {
            enabled: true
            executionIntervalMinutes: 30
            parameters: "{\"threshold\": 100}"
        }
    ) {
        id
        name
        enabled
        version
        updatedAt
    }
}
```

### 3. Generate Compliance Report

```graphql
mutation {
    generateReport(
        reportType: "energy_audit"
        params: {
            facilityIds: ["facility-1", "facility-2"]
            dateRange: {
                startDate: "2025-01-01"
                endDate: "2025-03-31"
            }
            includeRecommendations: true
        }
    ) {
        id
        reportType
        status
        periodStart
        periodEnd
        summary
        actionItemsCount
        generatedAt
    }
}
```

## Subscriptions

### 1. Monitor Job Progress

```graphql
subscription {
    jobProgress(jobId: "job-a1b2c3d4") {
        jobId
        progressPercent
        status
        message
        timestamp
    }
}
```

**Event Stream:**

```json
{
    "data": {
        "jobProgress": {
            "jobId": "job-a1b2c3d4",
            "progressPercent": 25,
            "status": "running",
            "message": "Processing at 25%",
            "timestamp": "2025-12-06T10:40:00Z"
        }
    }
}
```

### 2. Subscribe to Agent Alerts

```graphql
subscription {
    agentAlerts(agentIds: ["agent-001", "agent-002"]) {
        agentId
        alertType
        message
        metricName
        metricValue
        timestamp
    }
}
```

## Integration Examples

### Example 1: Python Application Integration

```python
import asyncio
from fastapi import FastAPI
from greenlang.infrastructure.api.graphql_integration import (
    setup_graphql, query_agents, query_emissions
)

app = FastAPI()
setup_graphql(app)

async def fetch_agent_metrics():
    """Fetch and display agent metrics."""
    from greenlang.infrastructure.api.graphql_schema import create_process_heat_schema
    from greenlang.infrastructure.api.graphql_integration import QueryExecutor

    schema = create_process_heat_schema()
    executor = QueryExecutor(schema)

    # Query agents
    result = await query_agents(executor, status="idle")
    agents = result["data"]["agents"]

    for agent in agents:
        print(f"Agent: {agent['name']}")
        print(f"  Status: {agent['status']}")
        print(f"  Execution Time: {agent['metrics']['executionTimeMs']}ms")
        print(f"  Memory: {agent['metrics']['memoryUsageMb']}MB")

# Run the example
# asyncio.run(fetch_agent_metrics())
```

### Example 2: HTTP Client Integration

```bash
# Query with curl
curl -X POST http://localhost:8000/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "{ agents { id name status } }"
  }'

# Start a job
curl -X POST http://localhost:8000/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "mutation { runCalculation(input: {agentId: \"agent-001\", facilityId: \"fac-1\", dateRange: {startDate: \"2025-01-01\", endDate: \"2025-03-31\"}}) { id status } }"
  }'
```

### Example 3: JavaScript/TypeScript Client

```javascript
// Using apollo-client or similar
const query = gql`
    query GetAgents {
        agents {
            id
            name
            status
            metrics {
                executionTimeMs
                errorCount
            }
        }
    }
`;

const client = new ApolloClient({
    uri: 'http://localhost:8000/graphql',
    cache: new InMemoryCache(),
});

client.query({ query }).then(result => {
    console.log('Agents:', result.data.agents);
});
```

## Error Handling

### Validation Errors

```json
{
    "data": null,
    "errors": [
        {
            "message": "Field \"agents\" argument \"status\" expected value",
            "locations": [
                {
                    "line": 2,
                    "column": 5
                }
            ]
        }
    ]
}
```

### Query Execution Errors

```json
{
    "data": {
        "agent": null
    },
    "errors": [
        {
            "message": "Agent not found",
            "path": ["agent"]
        }
    ]
}
```

### Handling Errors in Python

```python
async def safe_query(executor, query, variables=None):
    """Execute query with error handling."""
    try:
        result = await executor.execute_query(query, variables)

        if "errors" in result:
            print(f"GraphQL Errors: {result['errors']}")
            return None

        return result["data"]

    except Exception as e:
        print(f"Execution Error: {e}")
        return None
```

## Performance Tips

### 1. Use Field Selection

**Good:**
```graphql
query {
    agents {
        id
        name
        status
    }
}
```

**Avoid:**
```graphql
query {
    agents {
        # Requesting all fields can be slower
        id
        name
        status
        lastRun
        nextRun
        version
        enabled
        metrics { ... }
        errorMessage
        createdAt
        updatedAt
    }
}
```

### 2. Use Pagination for Large Datasets

```graphql
query {
    jobs(status: "completed") {
        id
        status
        # ... only fetch what you need
    }
}
```

### 3. Batch Queries

```graphql
query {
    agents {
        id
        name
    }
    jobs {
        id
        status
    }
    reports: complianceReports {
        id
        reportType
    }
}
```

### 4. Implement Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
async def get_agent_cached(agent_id: str):
    """Get agent with caching."""
    return await query_agent(executor, agent_id)
```

## Schema Definition Language (SDL)

The complete schema in SDL format:

```graphql
type Query {
    agents(status: String): [ProcessHeatAgent!]!
    agent(id: ID!): ProcessHeatAgent
    emissions(facilityId: ID!, dateRange: DateRangeInput): [EmissionResult!]!
    jobs(status: String): [CalculationJob!]!
    complianceReports(reportType: String): [ComplianceReport!]!
}

type Mutation {
    runCalculation(input: CalculationInput!): CalculationJob!
    updateAgentConfig(id: ID!, config: AgentConfigInput!): ProcessHeatAgent!
    generateReport(reportType: String!, params: ReportParamsInput!): ComplianceReport!
}

type Subscription {
    jobProgress(jobId: ID!): JobProgressEvent!
    agentAlerts(agentIds: [ID!]): AlertEvent!
}
```

## Additional Resources

- [Strawberry GraphQL Documentation](https://strawberry.rocks/)
- [GraphQL Best Practices](https://graphql.org/learn/best-practices/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## Support

For issues or questions:
1. Check the test files in `tests/unit/test_graphql_schema.py`
2. Review the integration module in `greenlang/infrastructure/api/graphql_integration.py`
3. Consult the schema definition in `greenlang/infrastructure/api/graphql_schema.py`
