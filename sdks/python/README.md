# GreenLang Python SDK

Official Python SDK for the GreenLang API.

## Installation

```bash
pip install greenlang-sdk
```

## Quick Start

```python
from greenlang_sdk import GreenLangClient

# Initialize client
client = GreenLangClient(api_key="gl_your_api_key_here")

# Execute a workflow
result = client.execute_workflow(
    workflow_id="wf_123",
    input_data={"query": "What is carbon footprint?"}
)

print(result.data)
```

## Features

- **Type-safe API** with Pydantic models
- **Automatic retry logic** with exponential backoff
- **Pagination support** for list operations
- **Streaming results** for long-running workflows
- **Comprehensive error handling**
- **Context manager support** for resource cleanup

## Authentication

The SDK requires an API key for authentication. You can obtain an API key from the [GreenLang Partner Portal](https://partners.greenlang.com).

```python
client = GreenLangClient(
    api_key="gl_your_api_key",
    base_url="https://api.greenlang.com",  # optional
    timeout=30,  # optional, in seconds
    max_retries=3  # optional
)
```

## Usage Examples

### Creating a Workflow

```python
from greenlang_sdk import GreenLangClient

client = GreenLangClient(api_key="gl_your_api_key")

# Define workflow
workflow_def = {
    "name": "Carbon Footprint Analysis",
    "description": "Analyze carbon emissions from various sources",
    "category": "carbon",
    "agents": [
        {
            "agent_id": "data_collector",
            "config": {"sources": ["energy", "transportation"]}
        },
        {
            "agent_id": "carbon_analyzer",
            "config": {"threshold": 100}
        }
    ]
}

# Create workflow
workflow = client.create_workflow(workflow_def)
print(f"Created workflow: {workflow.id}")
```

### Executing an Agent

```python
from greenlang_sdk import GreenLangClient

client = GreenLangClient(api_key="gl_your_api_key")

# Execute agent
result = client.execute_agent(
    agent_id="carbon_analyzer",
    input_data={
        "query": "Calculate emissions for 1000 kWh",
        "location": "California"
    },
    config={"precision": "high"}
)

if result.is_successful:
    print(f"Answer: {result.data['answer']}")
    print(f"Confidence: {result.data['confidence']}")

    # Display citations
    for citation in result.citations:
        print(f"- {citation.source_title}: {citation.source_url}")
```

### Listing Workflows

```python
from greenlang_sdk import GreenLangClient

client = GreenLangClient(api_key="gl_your_api_key")

# List workflows with pagination
workflows = client.list_workflows(limit=10, category="carbon")

for workflow in workflows:
    print(f"{workflow.name} ({workflow.id})")
```

### Streaming Results

```python
from greenlang_sdk import GreenLangClient

client = GreenLangClient(api_key="gl_your_api_key")

# Stream workflow execution
for chunk in client.stream_execution("wf_123", {"query": "test"}):
    if chunk.get("type") == "progress":
        print(f"Progress: {chunk['percentage']}%")
    elif chunk.get("type") == "complete":
        print(f"Result: {chunk['result']}")
```

### Pagination

```python
from greenlang_sdk import GreenLangClient

client = GreenLangClient(api_key="gl_your_api_key")

# Iterate over all workflows automatically
for workflow in client.list_workflows_iter(page_size=20):
    print(workflow.name)
```

### Context Manager

```python
from greenlang_sdk import GreenLangClient

# Automatically close session
with GreenLangClient(api_key="gl_your_api_key") as client:
    result = client.execute_workflow("wf_123", {"query": "test"})
    print(result.data)
```

## Error Handling

The SDK provides specific exception classes for different error types:

```python
from greenlang_sdk import (
    GreenLangClient,
    AuthenticationException,
    RateLimitException,
    NotFoundException,
    ValidationException,
    APIException
)

client = GreenLangClient(api_key="gl_your_api_key")

try:
    result = client.execute_workflow("wf_123", {"query": "test"})
except AuthenticationException:
    print("Invalid API key")
except RateLimitException as e:
    print(f"Rate limit exceeded. Retry after: {e.retry_after}s")
except NotFoundException:
    print("Workflow not found")
except ValidationException as e:
    print(f"Invalid request: {e.message}")
except APIException as e:
    print(f"API error: {e.message}")
```

## API Reference

### Client Methods

#### Workflows

- `create_workflow(workflow_def: Dict) -> Workflow`
- `get_workflow(workflow_id: str) -> Workflow`
- `list_workflows(limit: int = 20, offset: int = 0, category: Optional[str] = None) -> List[Workflow]`
- `list_workflows_iter(page_size: int = 20, category: Optional[str] = None) -> Iterator[Workflow]`
- `update_workflow(workflow_id: str, updates: Dict) -> Workflow`
- `delete_workflow(workflow_id: str) -> None`
- `execute_workflow(workflow_id: str, input_data: Dict, stream: bool = False) -> ExecutionResult`

#### Agents

- `get_agent(agent_id: str) -> Agent`
- `list_agents(category: Optional[str] = None, limit: int = 20, offset: int = 0) -> List[Agent]`
- `execute_agent(agent_id: str, input_data: Dict, config: Optional[Dict] = None) -> ExecutionResult`

#### Executions

- `get_execution(execution_id: str) -> ExecutionResult`
- `list_executions(workflow_id: Optional[str] = None, status: Optional[str] = None, limit: int = 20, offset: int = 0) -> List[ExecutionResult]`

#### Citations

- `get_citations(execution_id: str) -> List[Citation]`

#### Streaming

- `stream_execution(workflow_id: str, input_data: Dict) -> Iterator[Dict]`

#### Utility

- `health_check() -> Dict`

### Models

#### Workflow

```python
class Workflow:
    id: str
    name: str
    description: Optional[str]
    category: Optional[str]
    agents: List[Dict[str, Any]]
    config: Dict[str, Any]
    is_public: bool
    created_at: datetime
    updated_at: datetime
    version: str
```

#### Agent

```python
class Agent:
    id: str
    name: str
    description: str
    category: AgentCategory
    capabilities: List[str]
    config_schema: Dict[str, Any]
    is_public: bool
    version: str
    created_at: datetime
```

#### ExecutionResult

```python
class ExecutionResult:
    id: str
    workflow_id: Optional[str]
    agent_id: Optional[str]
    status: WorkflowStatus
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]]
    error: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]
    duration_ms: Optional[int]
    citations: List[Citation]
    metadata: Dict[str, Any]

    # Properties
    data: Optional[Dict[str, Any]]  # Alias for output_data
    is_complete: bool
    is_successful: bool
```

#### Citation

```python
class Citation:
    id: str
    execution_id: str
    source_type: CitationType
    source_url: Optional[HttpUrl]
    source_title: Optional[str]
    source_author: Optional[str]
    published_date: Optional[datetime]
    excerpt: Optional[str]
    relevance_score: float
    metadata: Dict[str, Any]
```

## Advanced Usage

### Custom Base URL

```python
client = GreenLangClient(
    api_key="gl_your_api_key",
    base_url="https://custom-api.example.com"
)
```

### Custom Timeout

```python
client = GreenLangClient(
    api_key="gl_your_api_key",
    timeout=60  # 60 seconds
)
```

### Custom Retry Configuration

```python
client = GreenLangClient(
    api_key="gl_your_api_key",
    max_retries=5  # Retry up to 5 times
)
```

## Development

### Running Tests

```bash
pytest tests/
```

### Building Documentation

```bash
cd docs
make html
```

## Support

- **Documentation**: [https://docs.greenlang.com](https://docs.greenlang.com)
- **Email**: support@greenlang.com
- **GitHub**: [https://github.com/greenlang/greenlang-sdk-python](https://github.com/greenlang/greenlang-sdk-python)

## License

MIT License. See LICENSE file for details.

## Changelog

### v1.0.0 (2025-01-08)

- Initial release
- Support for workflows, agents, and executions
- Streaming support
- Automatic pagination
- Comprehensive error handling
- Type-safe API with Pydantic models
