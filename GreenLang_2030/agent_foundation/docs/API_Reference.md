# API Reference

## GreenLang Agent Foundation API Documentation

Complete API reference for the GreenLang Agent Foundation, covering all classes, methods, and interfaces.

---

## Table of Contents

1. [Base Agent API](#base-agent-api)
2. [Memory API](#memory-api)
3. [Intelligence API](#intelligence-api)
4. [Orchestration API](#orchestration-api)
5. [Communication API](#communication-api)
6. [REST API Endpoints](#rest-api-endpoints)
7. [WebSocket API](#websocket-api)
8. [Event API](#event-api)

---

## Base Agent API

### BaseAgent Class

```python
class BaseAgent(ABC):
    """
    Abstract base class for all GreenLang agents.

    Attributes:
        config (AgentConfig): Agent configuration
        state (AgentState): Current agent state
        logger (Logger): Agent logger instance
        metrics (MetricsCollector): Metrics collector
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize base agent.

        Args:
            config: Agent configuration object

        Raises:
            ConfigurationError: If configuration is invalid
        """

    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method (must be implemented).

        Args:
            input_data: Input data dictionary

        Returns:
            Processed result dictionary

        Raises:
            ValidationError: If input validation fails
            ProcessingError: If processing fails
        """

    async def initialize(self) -> None:
        """
        Initialize agent resources.

        Raises:
            InitializationError: If initialization fails
        """

    async def start(self) -> None:
        """
        Start agent processing.

        Raises:
            RuntimeError: If agent not in READY state
        """

    async def stop(self) -> None:
        """
        Stop agent gracefully.
        """

    async def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status.

        Returns:
            Status dictionary containing:
            - state: Current state
            - uptime: Uptime in seconds
            - metrics: Performance metrics
            - health: Health check results
        """

    async def health_check(self) -> HealthStatus:
        """
        Perform health check.

        Returns:
            HealthStatus object with check results
        """
```

### AgentConfig Class

```python
@dataclass
class AgentConfig:
    """
    Agent configuration.

    Attributes:
        name (str): Agent name
        version (str): Agent version
        capabilities (List[str]): Agent capabilities
        timeout (int): Processing timeout in seconds
        retry_attempts (int): Number of retry attempts
        memory_enabled (bool): Enable memory systems
        llm_provider (str): LLM provider name
        observability (Dict): Observability settings
    """

    name: str
    version: str = "1.0.0"
    capabilities: List[str] = field(default_factory=list)
    timeout: int = 30
    retry_attempts: int = 3
    memory_enabled: bool = True
    llm_provider: str = "openai"
    observability: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate configuration.

        Raises:
            ValidationError: If configuration is invalid
        """

    @classmethod
    def from_yaml(cls, path: str) -> 'AgentConfig':
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            AgentConfig instance
        """

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            Configuration dictionary
        """
```

### AgentState Enum

```python
class AgentState(Enum):
    """Agent lifecycle states."""

    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PROCESSING = "processing"
    PAUSED = "paused"
    STOPPING = "stopping"
    TERMINATED = "terminated"
    ERROR = "error"
    RECOVERING = "recovering"
```

---

## Memory API

### MemoryManager Class

```python
class MemoryManager:
    """
    Unified memory management interface.

    Provides access to all memory systems.
    """

    def __init__(self, config: MemoryConfig):
        """
        Initialize memory manager.

        Args:
            config: Memory configuration
        """

    async def store(
        self,
        key: str,
        value: Any,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Store item in memory.

        Args:
            key: Memory key
            value: Value to store
            memory_type: Type of memory to use
            metadata: Optional metadata

        Returns:
            Storage key/ID

        Raises:
            MemoryError: If storage fails
        """

    async def retrieve(
        self,
        key: str,
        memory_type: MemoryType = None
    ) -> Optional[Any]:
        """
        Retrieve item from memory.

        Args:
            key: Memory key
            memory_type: Specific memory type (None = search all)

        Returns:
            Retrieved value or None
        """

    async def search(
        self,
        query: str,
        memory_types: List[MemoryType] = None,
        top_k: int = 10
    ) -> List[MemoryItem]:
        """
        Search memory by query.

        Args:
            query: Search query
            memory_types: Memory types to search
            top_k: Number of results

        Returns:
            List of memory items
        """

    async def consolidate(self) -> ConsolidationStats:
        """
        Consolidate memories across tiers.

        Returns:
            Consolidation statistics
        """

    async def forget(self, key: str) -> bool:
        """
        Remove item from memory.

        Args:
            key: Memory key

        Returns:
            True if removed successfully
        """
```

### MemoryType Enum

```python
class MemoryType(Enum):
    """Memory system types."""

    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
```

---

## Intelligence API

### LLMClient Class

```python
class LLMClient:
    """
    Multi-provider LLM client.

    Supports multiple LLM providers with fallback.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = None,
        api_key: str = None,
        **kwargs
    ):
        """
        Initialize LLM client.

        Args:
            provider: LLM provider name
            model: Model name
            api_key: API key (or from env)
            **kwargs: Provider-specific arguments
        """

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        system: str = None,
        **kwargs
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system: System prompt
            **kwargs: Provider-specific parameters

        Returns:
            Generated text

        Raises:
            LLMError: If generation fails
        """

    async def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> Message:
        """
        Chat completion.

        Args:
            messages: Conversation messages
            **kwargs: Additional parameters

        Returns:
            Assistant message
        """

    async def embed(
        self,
        text: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings.

        Args:
            text: Text or list of texts

        Returns:
            Embedding vector(s)
        """
```

### RAGSystem Class

```python
class RAGSystem:
    """
    Retrieval-Augmented Generation system.
    """

    def __init__(self, config: RAGConfig):
        """
        Initialize RAG system.

        Args:
            config: RAG configuration
        """

    async def add_document(
        self,
        document: Union[str, Path, Document],
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Add document to RAG system.

        Args:
            document: Document content, path, or Document object
            metadata: Document metadata

        Returns:
            Document ID
        """

    async def query(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict[str, Any] = None
    ) -> RAGResponse:
        """
        Query RAG system.

        Args:
            query: Query string
            top_k: Number of documents to retrieve
            filters: Metadata filters

        Returns:
            RAGResponse with answer and sources
        """

    async def update_index(self) -> None:
        """
        Update vector index.
        """

    async def clear(self) -> None:
        """
        Clear all documents.
        """
```

---

## Orchestration API

### AgentOrchestrator Class

```python
class AgentOrchestrator:
    """
    Multi-agent orchestration system.
    """

    def __init__(self, config: OrchestratorConfig = None):
        """
        Initialize orchestrator.

        Args:
            config: Orchestrator configuration
        """

    def register(
        self,
        agent: BaseAgent,
        name: str = None
    ) -> None:
        """
        Register agent with orchestrator.

        Args:
            agent: Agent instance
            name: Optional agent name
        """

    async def execute_workflow(
        self,
        workflow: Workflow,
        input_data: Dict[str, Any],
        timeout: int = 300
    ) -> WorkflowResult:
        """
        Execute multi-agent workflow.

        Args:
            workflow: Workflow definition
            input_data: Initial input data
            timeout: Workflow timeout in seconds

        Returns:
            WorkflowResult with outputs

        Raises:
            WorkflowError: If workflow fails
        """

    async def execute_parallel(
        self,
        agents: List[str],
        input_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute agents in parallel.

        Args:
            agents: List of agent names
            input_data: Input for all agents

        Returns:
            List of results
        """

    async def execute_pipeline(
        self,
        pipeline: List[str],
        input_data: Any
    ) -> Any:
        """
        Execute agent pipeline.

        Args:
            pipeline: Ordered list of agent names
            input_data: Initial input

        Returns:
            Final output
        """

    def get_agent(self, name: str) -> BaseAgent:
        """
        Get registered agent by name.

        Args:
            name: Agent name

        Returns:
            Agent instance

        Raises:
            KeyError: If agent not found
        """

    def list_agents(self) -> List[str]:
        """
        List registered agent names.

        Returns:
            List of agent names
        """
```

---

## Communication API

### MessageBus Class

```python
class MessageBus:
    """
    Inter-agent message bus.
    """

    def __init__(self, broker_url: str = None):
        """
        Initialize message bus.

        Args:
            broker_url: Message broker URL
        """

    async def send(
        self,
        message: Message,
        timeout: int = 30
    ) -> None:
        """
        Send message.

        Args:
            message: Message to send
            timeout: Send timeout

        Raises:
            MessageError: If send fails
        """

    async def receive(
        self,
        agent_id: str,
        timeout: int = None
    ) -> Optional[Message]:
        """
        Receive message for agent.

        Args:
            agent_id: Receiving agent ID
            timeout: Receive timeout (None = block)

        Returns:
            Message or None if timeout
        """

    async def subscribe(
        self,
        topic: str,
        handler: Callable
    ) -> str:
        """
        Subscribe to topic.

        Args:
            topic: Topic pattern
            handler: Message handler function

        Returns:
            Subscription ID
        """

    async def unsubscribe(self, subscription_id: str) -> None:
        """
        Unsubscribe from topic.

        Args:
            subscription_id: Subscription ID
        """

    async def broadcast(
        self,
        topic: str,
        message: Any
    ) -> int:
        """
        Broadcast message to topic.

        Args:
            topic: Topic name
            message: Message to broadcast

        Returns:
            Number of recipients
        """
```

### Message Class

```python
@dataclass
class Message:
    """
    Inter-agent message.
    """

    sender: str
    recipient: str
    type: MessageType
    payload: Any
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: Priority = Priority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## REST API Endpoints

### Agent Management

```yaml
/api/v1/agents:
  GET:
    description: List all agents
    responses:
      200:
        schema:
          type: array
          items:
            $ref: '#/definitions/Agent'

  POST:
    description: Create new agent
    parameters:
      - name: body
        schema:
          $ref: '#/definitions/AgentConfig'
    responses:
      201:
        schema:
          $ref: '#/definitions/Agent'

/api/v1/agents/{agent_id}:
  GET:
    description: Get agent details
    parameters:
      - name: agent_id
        type: string
    responses:
      200:
        schema:
          $ref: '#/definitions/Agent'

  DELETE:
    description: Delete agent
    parameters:
      - name: agent_id
        type: string
    responses:
      204:
        description: Agent deleted

/api/v1/agents/{agent_id}/process:
  POST:
    description: Process data with agent
    parameters:
      - name: agent_id
        type: string
      - name: body
        schema:
          type: object
    responses:
      200:
        schema:
          type: object

/api/v1/agents/{agent_id}/status:
  GET:
    description: Get agent status
    parameters:
      - name: agent_id
        type: string
    responses:
      200:
        schema:
          $ref: '#/definitions/AgentStatus'
```

### Memory Operations

```yaml
/api/v1/memory/store:
  POST:
    description: Store in memory
    parameters:
      - name: body
        schema:
          type: object
          properties:
            key:
              type: string
            value:
              type: object
            memory_type:
              type: string
    responses:
      201:
        schema:
          type: object
          properties:
            id:
              type: string

/api/v1/memory/retrieve/{key}:
  GET:
    description: Retrieve from memory
    parameters:
      - name: key
        type: string
    responses:
      200:
        schema:
          type: object

/api/v1/memory/search:
  POST:
    description: Search memory
    parameters:
      - name: body
        schema:
          type: object
          properties:
            query:
              type: string
            top_k:
              type: integer
    responses:
      200:
        schema:
          type: array
          items:
            $ref: '#/definitions/MemoryItem'
```

### Workflow Execution

```yaml
/api/v1/workflows:
  POST:
    description: Execute workflow
    parameters:
      - name: body
        schema:
          $ref: '#/definitions/Workflow'
    responses:
      202:
        schema:
          type: object
          properties:
            workflow_id:
              type: string
            status:
              type: string

/api/v1/workflows/{workflow_id}/status:
  GET:
    description: Get workflow status
    parameters:
      - name: workflow_id
        type: string
    responses:
      200:
        schema:
          $ref: '#/definitions/WorkflowStatus'
```

---

## WebSocket API

### Real-time Agent Communication

```javascript
// Connect to WebSocket
const ws = new WebSocket('wss://api.greenlang.ai/v1/ws');

// Authentication
ws.send(JSON.stringify({
  type: 'auth',
  token: 'your-jwt-token'
}));

// Subscribe to agent events
ws.send(JSON.stringify({
  type: 'subscribe',
  topics: ['agent.status', 'agent.messages'],
  agent_id: 'agent-123'
}));

// Send message to agent
ws.send(JSON.stringify({
  type: 'message',
  recipient: 'agent-123',
  payload: {
    task: 'process',
    data: {}
  }
}));

// Receive events
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch(data.type) {
    case 'agent.status':
      console.log('Agent status:', data.status);
      break;
    case 'agent.message':
      console.log('Agent message:', data.message);
      break;
    case 'error':
      console.error('Error:', data.error);
      break;
  }
};
```

---

## Event API

### Event Types

```python
class EventType(Enum):
    """System event types."""

    # Lifecycle events
    AGENT_CREATED = "agent.created"
    AGENT_STARTED = "agent.started"
    AGENT_STOPPED = "agent.stopped"
    AGENT_ERROR = "agent.error"

    # Task events
    TASK_ASSIGNED = "task.assigned"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"

    # Memory events
    MEMORY_STORED = "memory.stored"
    MEMORY_RETRIEVED = "memory.retrieved"
    MEMORY_CONSOLIDATED = "memory.consolidated"

    # System events
    RESOURCE_ALLOCATED = "resource.allocated"
    RESOURCE_RELEASED = "resource.released"
    SCALING_TRIGGERED = "scaling.triggered"
```

### Event Handler

```python
class EventHandler:
    """
    Event handling system.
    """

    async def on_event(
        self,
        event_type: EventType,
        handler: Callable
    ) -> str:
        """
        Register event handler.

        Args:
            event_type: Event type to handle
            handler: Handler function

        Returns:
            Handler ID
        """

    async def emit(
        self,
        event_type: EventType,
        data: Dict[str, Any]
    ) -> None:
        """
        Emit event.

        Args:
            event_type: Event type
            data: Event data
        """

    async def remove_handler(self, handler_id: str) -> None:
        """
        Remove event handler.

        Args:
            handler_id: Handler ID
        """
```

---

## Error Codes

### Standard Error Codes

| Code | Name | Description |
|------|------|-------------|
| 1001 | VALIDATION_ERROR | Input validation failed |
| 1002 | PROCESSING_ERROR | Processing failed |
| 1003 | TIMEOUT_ERROR | Operation timed out |
| 1004 | MEMORY_ERROR | Memory operation failed |
| 1005 | LLM_ERROR | LLM call failed |
| 1006 | COMMUNICATION_ERROR | Communication failed |
| 1007 | WORKFLOW_ERROR | Workflow execution failed |
| 1008 | RESOURCE_ERROR | Resource allocation failed |
| 1009 | AUTHENTICATION_ERROR | Authentication failed |
| 1010 | AUTHORIZATION_ERROR | Authorization failed |

---

## Rate Limits

### API Rate Limits

| Endpoint | Limit | Window |
|----------|-------|---------|
| /api/v1/agents/* | 1000 | 1 minute |
| /api/v1/memory/* | 500 | 1 minute |
| /api/v1/workflows/* | 100 | 1 minute |
| WebSocket connections | 10 | Per user |
| LLM calls | 100 | 1 minute |

---

## SDK Examples

### Python SDK

```python
from greenlang import GreenLangClient

# Initialize client
client = GreenLangClient(
    api_key="your-api-key",
    base_url="https://api.greenlang.ai"
)

# Create agent
agent = client.agents.create(
    name="my-agent",
    type="compliance",
    config={"timeout": 60}
)

# Process data
result = agent.process({
    "task": "analyze",
    "data": {}
})

# Search memory
memories = client.memory.search(
    query="carbon emissions",
    top_k=10
)
```

### JavaScript SDK

```javascript
import { GreenLangClient } from '@greenlang/sdk';

// Initialize client
const client = new GreenLangClient({
  apiKey: 'your-api-key',
  baseURL: 'https://api.greenlang.ai'
});

// Create agent
const agent = await client.agents.create({
  name: 'my-agent',
  type: 'compliance',
  config: { timeout: 60 }
});

// Process data
const result = await agent.process({
  task: 'analyze',
  data: {}
});

// Search memory
const memories = await client.memory.search({
  query: 'carbon emissions',
  topK: 10
});
```

---

**Last Updated**: November 2024
**Version**: 1.0.0
**Maintainer**: GreenLang API Team