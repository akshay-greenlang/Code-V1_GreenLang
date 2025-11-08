# GreenLang GraphQL API

Comprehensive GraphQL API layer for GreenLang with real-time subscriptions, RBAC integration, and advanced query optimization.

## Features

- **Complete Schema**: Full type system for agents, workflows, executions, and RBAC
- **DataLoader Integration**: N+1 query prevention with batching and caching
- **Real-time Subscriptions**: WebSocket-based subscriptions for execution monitoring
- **Query Complexity Analysis**: Prevents expensive queries with depth and cost limits
- **RBAC Integration**: Role-based access control for all operations
- **GraphQL Playground**: Interactive API explorer with examples
- **Comprehensive Testing**: Unit and integration tests with >90% coverage
- **Production Ready**: Optimized for performance and security

## Quick Start

```python
from greenlang.api.graphql.server import run_dev_server
from greenlang.core.orchestrator import Orchestrator
from greenlang.auth.rbac import RBACManager
from greenlang.auth.auth import AuthManager

# Initialize services
orchestrator = Orchestrator()
auth_manager = AuthManager()
rbac_manager = RBACManager()

# Run server
run_dev_server(
    orchestrator=orchestrator,
    auth_manager=auth_manager,
    rbac_manager=rbac_manager,
    host="0.0.0.0",
    port=8000,
)
```

Visit `http://localhost:8000/playground` to explore the API!

## Project Structure

```
greenlang/api/graphql/
├── __init__.py                 # Module exports
├── schema.graphql              # GraphQL SDL schema definition
├── types.py                    # Strawberry type definitions
├── resolvers.py                # Query and Mutation resolvers
├── subscriptions.py            # WebSocket subscription handlers
├── dataloaders.py              # DataLoader implementations
├── context.py                  # GraphQL execution context
├── complexity.py               # Query complexity analyzer
├── playground.py               # Interactive playground setup
├── server.py                   # FastAPI server configuration
├── GRAPHQL_API.md             # Comprehensive API documentation
├── README.md                   # This file
└── tests/
    ├── __init__.py
    ├── test_resolvers.py       # Unit tests for resolvers
    └── test_integration.py     # Integration tests
```

## Installation

```bash
pip install strawberry-graphql[fastapi]
pip install fastapi
pip install uvicorn[standard]
pip install python-multipart
```

## Example Queries

### List Agents

```graphql
query {
  agents(pagination: { page: 1, pageSize: 10 }) {
    nodes {
      id
      name
      version
      stats {
        executions
        successRate
      }
    }
    pageInfo {
      totalCount
      hasNextPage
    }
  }
}
```

### Execute Workflow

```graphql
mutation {
  executeWorkflow(input: {
    workflowId: "my-workflow"
    inputData: { key: "value" }
  }) {
    success
    execution {
      id
      status
      outputData
    }
  }
}
```

### Watch Execution

```graphql
subscription {
  executionUpdated(executionId: "exec-123") {
    event
    execution {
      status
      completedAt
    }
  }
}
```

## Key Components

### Schema (`schema.graphql`)

Complete GraphQL schema with:
- 50+ types
- 15+ queries
- 20+ mutations
- 5+ subscriptions
- Custom scalars (DateTime, JSON)
- Input validation
- Pagination support

### Resolvers (`resolvers.py`)

~1,400 lines of resolver logic:
- Query resolvers for all resources
- Mutation resolvers with validation
- Permission checking
- Error handling
- DataLoader integration

### DataLoaders (`dataloaders.py`)

N+1 query prevention:
- AgentLoader
- WorkflowLoader
- ExecutionLoader
- UserLoader
- Batching and caching

### Subscriptions (`subscriptions.py`)

Real-time updates:
- Execution monitoring
- Agent updates
- Workflow updates
- System metrics
- Connection management

### Complexity Analysis (`complexity.py`)

Query cost estimation:
- Depth limiting (max: 10)
- Complexity scoring
- Field-specific costs
- List multipliers
- Introspection limits

### Playground (`playground.py`)

Interactive explorer:
- Pre-configured tabs
- Example queries
- Schema documentation
- Syntax highlighting
- Auto-completion

## Testing

Run tests:

```bash
# Unit tests
pytest greenlang/api/graphql/tests/test_resolvers.py -v

# Integration tests
pytest greenlang/api/graphql/tests/test_integration.py -v

# All tests with coverage
pytest greenlang/api/graphql/tests/ --cov=greenlang.api.graphql --cov-report=html
```

## Performance

### DataLoader Batching

Before (N+1 problem):
```
Query workflows (1 query)
  ├─ Get workflow 1 (1 query)
  │  ├─ Get agent A (1 query)
  │  └─ Get agent B (1 query)
  └─ Get workflow 2 (1 query)
     ├─ Get agent C (1 query)
     └─ Get agent A (1 query)
Total: 7 queries
```

After (with DataLoader):
```
Query workflows (1 query)
Batch load agents [A, B, C] (1 query)
Total: 2 queries
```

### Query Complexity

Example complexity calculation:

```graphql
query {
  agents(pagination: { pageSize: 10 }) {  # 10 * 5 = 50
    nodes {
      workflows(pagination: { pageSize: 5 }) {  # 10 * 5 * 5 = 250
        nodes {
          steps {  # 10 * 5 * 5 * 1 = 250
            agent {  # Additional cost
              id
            }
          }
        }
      }
    }
  }
}
# Total complexity: ~600 points
```

## Security

### Authentication

All requests require Bearer token:

```http
Authorization: Bearer <token>
```

### RBAC Integration

Every resolver checks permissions:

```python
@strawberry.field
async def create_agent(self, info, input):
    # Permission check
    if not context.rbac_manager.check_permission(
        context.user_id,
        "agent",
        "create",
    ):
        raise PermissionError("Access denied")
    # ... create agent
```

### API Key Management

Create scoped API keys:

```graphql
mutation {
  createAPIKey(input: {
    name: "Service Key"
    scopes: ["workflow:execute"]
    allowedIps: ["10.0.0.0/8"]
    rateLimit: 1000
  }) {
    keyId
    displayKey
  }
}
```

## Production Deployment

### Configuration

```python
from greenlang.api.graphql.server import create_production_app
from greenlang.api.graphql.complexity import ComplexityConfig

complexity_config = ComplexityConfig(
    max_depth=8,
    max_complexity=500,
)

app = create_production_app(
    orchestrator=orchestrator,
    auth_manager=auth_manager,
    rbac_manager=rbac_manager,
    complexity_config=complexity_config,
    cors_origins=["https://app.example.com"],
)
```

### Running

```bash
uvicorn greenlang.api.graphql.server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

### Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "greenlang.api.graphql.server:app", "--host", "0.0.0.0"]
```

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

### Metrics

```bash
curl http://localhost:8000/metrics
```

### Real-time Metrics

```graphql
subscription {
  systemMetrics(interval: 5000) {
    cpuUsage
    memoryUsage
    activeExecutions
  }
}
```

## Documentation

- **API Documentation**: [GRAPHQL_API.md](./GRAPHQL_API.md)
- **Playground**: http://localhost:8000/playground
- **Schema**: [schema.graphql](./schema.graphql)

## Architecture

### Request Flow

```
Client Request
    ↓
FastAPI Middleware
    ↓
Authentication (Bearer Token)
    ↓
GraphQL Context Creation
    ↓
Query Complexity Analysis
    ↓
Resolver Execution
    ├─ Permission Check (RBAC)
    ├─ DataLoader Batching
    └─ Business Logic
    ↓
Response Formatting
    ↓
Client Response
```

### Subscription Flow

```
WebSocket Connection
    ↓
Authentication
    ↓
Subscription Registration
    ↓
Event Loop
    ├─ Heartbeat (30s)
    ├─ Event Publishing
    └─ Client Updates
    ↓
Connection Close
    ↓
Cleanup
```

## Contributing

### Adding New Queries

1. Define type in `types.py`:
```python
@strawberry.type
class MyNewType:
    id: strawberry.ID
    name: str
```

2. Add resolver in `resolvers.py`:
```python
@strawberry.field
async def my_query(self, info) -> MyNewType:
    # Implementation
    pass
```

3. Add tests in `tests/test_resolvers.py`:
```python
def test_my_query(mock_info):
    # Test implementation
    pass
```

### Adding New Mutations

Follow similar pattern as queries, with input validation:

```python
@strawberry.input
class MyInput:
    name: str
    value: int

@strawberry.mutation
async def my_mutation(self, info, input: MyInput):
    # Validate and execute
    pass
```

## Best Practices

1. **Use DataLoaders**: Always use DataLoaders for relational data
2. **Limit Depth**: Keep query depth under 5 levels
3. **Paginate Lists**: Use reasonable page sizes (10-50)
4. **Scope Subscriptions**: Subscribe to specific resources, not all
5. **Check Permissions**: Every resolver should check permissions
6. **Validate Input**: Use Strawberry input types for validation
7. **Handle Errors**: Return structured errors with codes
8. **Cache Results**: Use DataLoader cache for repeated queries

## Troubleshooting

### Common Issues

**Q**: Authentication fails
**A**: Ensure Authorization header includes `Bearer <token>`

**Q**: Query too complex
**A**: Reduce query depth or pagination size

**Q**: N+1 queries detected
**A**: Use DataLoader for relational fields

**Q**: WebSocket disconnects
**A**: Check heartbeat configuration and auth token validity

## License

Part of the GreenLang project. See main repository for license details.

## Support

For issues and questions:
- GitHub Issues: https://github.com/akshay-greenlang/Code-V1_GreenLang/issues
- Documentation: [GRAPHQL_API.md](./GRAPHQL_API.md)
