# Getting Started with GreenLang Agent Foundation

## Quick Start (5 Minutes)

Get your first GreenLang agent running in under 5 minutes! This guide will walk you through installation, creating your first agent, and running it successfully.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Your First Agent](#your-first-agent)
4. [Basic Operations](#basic-operations)
5. [Configuration](#configuration)
6. [Running Examples](#running-examples)
7. [Next Steps](#next-steps)

---

## Prerequisites

### System Requirements

- **Python**: 3.11 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 10GB free space
- **OS**: Linux, macOS, or Windows with WSL2

### Required Software

```bash
# Check Python version
python --version  # Should be 3.11+

# Check pip
pip --version

# Check git
git --version
```

### Optional but Recommended

- **Docker**: For containerized deployment
- **Kubernetes**: For production deployment
- **Redis**: For caching and short-term memory
- **PostgreSQL**: For long-term storage

---

## Installation

### Method 1: Install from PyPI (Recommended)

```bash
# Create virtual environment
python -m venv greenlang-env
source greenlang-env/bin/activate  # On Windows: greenlang-env\Scripts\activate

# Install GreenLang
pip install greenlang-ai

# Verify installation
greenlang --version
```

### Method 2: Install from Source

```bash
# Clone repository
git clone https://github.com/greenlang/agent-foundation.git
cd agent-foundation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

### Method 3: Docker Installation

```bash
# Pull official image
docker pull greenlang/agent-foundation:latest

# Run container
docker run -d \
  --name greenlang-agent \
  -p 8000:8000 \
  -e API_KEY=your-api-key \
  greenlang/agent-foundation:latest
```

---

## Your First Agent

### Step 1: Create a Simple Agent

Create a file named `my_first_agent.py`:

```python
from greenlang import BaseAgent, AgentConfig
from typing import Dict, Any
import asyncio

class HelloWorldAgent(BaseAgent):
    """Your first GreenLang agent!"""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.logger.info(f"Agent {self.name} initialized!")

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return greeting."""

        # Extract name from input
        name = input_data.get('name', 'World')

        # Generate greeting
        greeting = f"Hello, {name}! I'm {self.name}."

        # Log the interaction
        self.logger.info(f"Processed greeting for {name}")

        return {
            'status': 'success',
            'message': greeting,
            'agent': self.name,
            'timestamp': self.current_timestamp()
        }

# Create and run the agent
async def main():
    # Configure the agent
    config = AgentConfig(
        name="hello-agent",
        version="1.0.0",
        description="A friendly greeting agent"
    )

    # Initialize agent
    agent = HelloWorldAgent(config)

    # Process some data
    result = await agent.process({'name': 'GreenLang Developer'})
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 2: Run Your Agent

```bash
python my_first_agent.py
```

Expected output:
```json
{
    "status": "success",
    "message": "Hello, GreenLang Developer! I'm hello-agent.",
    "agent": "hello-agent",
    "timestamp": "2024-11-14T10:30:45Z"
}
```

---

## Basic Operations

### Creating Agents with Memory

```python
from greenlang import BaseAgent, AgentConfig, MemoryManager

class MemoryAgent(BaseAgent):
    """Agent with memory capabilities."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        # Initialize memory
        self.memory = MemoryManager(
            short_term_capacity=100,
            long_term_enabled=True
        )

    async def remember(self, key: str, value: Any) -> None:
        """Store information in memory."""
        await self.memory.store(key, value)
        self.logger.info(f"Stored {key} in memory")

    async def recall(self, key: str) -> Any:
        """Retrieve information from memory."""
        value = await self.memory.retrieve(key)
        return value

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process with memory context."""

        # Check if we've seen this user before
        user = input_data.get('user')
        previous_interaction = await self.recall(f"user:{user}")

        if previous_interaction:
            response = f"Welcome back, {user}! Last seen: {previous_interaction}"
        else:
            response = f"Nice to meet you, {user}!"
            await self.remember(f"user:{user}", self.current_timestamp())

        return {'message': response}
```

### Using LLM Integration

```python
from greenlang import BaseAgent, AgentConfig, LLMClient

class IntelligentAgent(BaseAgent):
    """Agent with LLM capabilities."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        # Initialize LLM client
        self.llm = LLMClient(
            provider="openai",  # or "anthropic", "google", "local"
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY")
        )

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process using LLM."""

        question = input_data.get('question')

        # Generate response using LLM
        response = await self.llm.generate(
            prompt=f"Answer this question concisely: {question}",
            max_tokens=150,
            temperature=0.7
        )

        return {
            'question': question,
            'answer': response,
            'model': self.llm.model
        }
```

### Multi-Agent Coordination

```python
from greenlang import AgentOrchestrator, BaseAgent

# Create orchestrator
orchestrator = AgentOrchestrator()

# Register multiple agents
orchestrator.register(DataCollectorAgent(config1))
orchestrator.register(ProcessorAgent(config2))
orchestrator.register(ReporterAgent(config3))

# Define workflow
workflow = {
    'steps': [
        {'agent': 'data-collector', 'action': 'collect'},
        {'agent': 'processor', 'action': 'process'},
        {'agent': 'reporter', 'action': 'generate_report'}
    ]
}

# Execute workflow
result = await orchestrator.execute_workflow(
    workflow=workflow,
    input_data={'source': 'database'}
)
```

---

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Database
DATABASE_URL=postgresql://user:pass@localhost/greenlang
REDIS_URL=redis://localhost:6379

# Agent Configuration
AGENT_MAX_WORKERS=10
AGENT_TIMEOUT=30
AGENT_RETRY_ATTEMPTS=3

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Monitoring
PROMETHEUS_PORT=9090
JAEGER_ENDPOINT=http://localhost:14268
```

### Configuration File

Create `config.yaml`:

```yaml
agent:
  name: my-agent
  version: 1.0.0
  capabilities:
    - reasoning
    - memory
    - tool_use

llm:
  default_provider: openai
  providers:
    openai:
      model: gpt-4
      temperature: 0.7
      max_tokens: 2000
    anthropic:
      model: claude-3-opus
      temperature: 0.5

memory:
  short_term:
    type: redis
    capacity: 1000
    ttl: 3600
  long_term:
    type: postgresql
    retention_days: 90

observability:
  logging:
    level: INFO
    format: json
  metrics:
    enabled: true
    port: 9090
  tracing:
    enabled: true
    sample_rate: 0.1
```

### Loading Configuration

```python
from greenlang import AgentConfig, load_config

# Load from file
config = load_config('config.yaml')

# Or create programmatically
config = AgentConfig(
    name="my-agent",
    version="1.0.0",
    llm_provider="openai",
    memory_enabled=True,
    observability={
        'logging_level': 'INFO',
        'metrics_enabled': True
    }
)

# Create agent with config
agent = MyAgent(config)
```

---

## Running Examples

### Example 1: Carbon Calculator Agent

```python
from greenlang.agents import CarbonCalculatorAgent

# Initialize agent
agent = CarbonCalculatorAgent(
    calculation_method="GHG Protocol",
    emission_factors="EPA 2024"
)

# Calculate emissions
result = await agent.calculate({
    'activity': 'electricity',
    'amount': 1000,
    'unit': 'kWh',
    'location': 'US-CA'
})

print(f"CO2 emissions: {result['emissions_kg']} kg CO2e")
```

### Example 2: Compliance Checker Agent

```python
from greenlang.agents import ComplianceAgent

# Initialize CSRD compliance agent
agent = ComplianceAgent(
    regulation="CSRD",
    company_type="large_enterprise"
)

# Check compliance
report = await agent.check_compliance({
    'company_data': company_profile,
    'reporting_year': 2024
})

print(f"Compliance score: {report['score']}%")
print(f"Missing requirements: {report['gaps']}")
```

### Example 3: RAG-Powered Q&A Agent

```python
from greenlang.agents import RAGAgent

# Initialize RAG agent with documents
agent = RAGAgent()

# Load documents
await agent.load_documents([
    'sustainability_report.pdf',
    'esg_policy.docx',
    'carbon_data.csv'
])

# Ask questions
answer = await agent.query(
    "What are our Scope 3 emissions for 2023?"
)

print(f"Answer: {answer['response']}")
print(f"Sources: {answer['citations']}")
```

---

## Command Line Interface

GreenLang provides a CLI for common operations:

```bash
# Initialize new project
greenlang init my-project

# Create new agent from template
greenlang create agent --template compliance --name csrd-agent

# Run agent
greenlang run my_agent.py --config config.yaml

# Test agent
greenlang test my_agent.py

# Deploy agent
greenlang deploy my_agent.py --environment production

# Monitor agents
greenlang monitor --dashboard

# List running agents
greenlang list agents

# Stop agent
greenlang stop agent-id
```

---

## Testing Your Agent

### Unit Testing

```python
import pytest
from greenlang.testing import AgentTestCase

class TestMyAgent(AgentTestCase):
    """Test suite for MyAgent."""

    async def test_process_valid_input(self):
        """Test processing valid input."""
        agent = MyAgent(self.get_test_config())

        result = await agent.process({'name': 'Test'})

        assert result['status'] == 'success'
        assert 'Test' in result['message']

    async def test_error_handling(self):
        """Test error handling."""
        agent = MyAgent(self.get_test_config())

        with pytest.raises(ValidationError):
            await agent.process({'invalid': 'data'})

# Run tests
# pytest test_my_agent.py
```

### Integration Testing

```python
from greenlang.testing import IntegrationTest

class TestAgentIntegration(IntegrationTest):
    """Integration tests for agent system."""

    async def test_multi_agent_workflow(self):
        """Test complete workflow."""
        # Setup
        orchestrator = self.create_orchestrator()

        # Execute workflow
        result = await orchestrator.execute_workflow(
            self.test_workflow,
            self.test_data
        )

        # Verify
        assert result['status'] == 'completed'
        assert len(result['steps']) == 3
```

---

## Debugging

### Enable Debug Logging

```python
import logging

# Set debug level
logging.basicConfig(level=logging.DEBUG)

# Or for specific agent
agent.logger.setLevel(logging.DEBUG)
```

### Use Debug Mode

```python
# Initialize agent in debug mode
agent = MyAgent(config, debug=True)

# This enables:
# - Detailed logging
# - Step-by-step execution
# - Performance profiling
# - Memory tracking
```

### Interactive Debugging

```python
# Use Python debugger
import pdb

async def process(self, input_data):
    pdb.set_trace()  # Breakpoint here
    result = await self.analyze(input_data)
    return result
```

---

## Common Issues and Solutions

### Issue 1: Import Error

```bash
ImportError: No module named 'greenlang'
```

**Solution**: Ensure GreenLang is installed in your active environment:
```bash
pip install greenlang-ai
```

### Issue 2: API Key Error

```bash
Error: No API key provided for LLM
```

**Solution**: Set environment variables:
```bash
export OPENAI_API_KEY=your-key
# Or use .env file
```

### Issue 3: Memory Error

```bash
MemoryError: Redis connection failed
```

**Solution**: Ensure Redis is running:
```bash
# Start Redis
redis-server

# Or use Docker
docker run -d -p 6379:6379 redis
```

### Issue 4: Timeout Error

```bash
TimeoutError: Agent execution exceeded 30 seconds
```

**Solution**: Increase timeout in config:
```python
config = AgentConfig(
    timeout=60,  # Increase to 60 seconds
    retry_attempts=3
)
```

---

## Next Steps

### Learn More

1. **[Agent Development Guide](Agent_Development_Guide.md)** - Build complex agents
2. **[Memory Systems Guide](Memory_Systems_Guide.md)** - Implement sophisticated memory
3. **[Intelligence Layer Guide](Intelligence_Layer_Guide.md)** - Advanced LLM integration
4. **[Orchestration Guide](Orchestration_Guide.md)** - Multi-agent systems

### Explore Examples

Check out our example repository:
```bash
git clone https://github.com/greenlang/examples
cd examples

# Run example agents
python examples/carbon_calculator.py
python examples/compliance_checker.py
python examples/rag_agent.py
```

### Join the Community

- **Discord**: https://discord.gg/greenlang
- **GitHub Discussions**: https://github.com/greenlang/discussions
- **Stack Overflow**: Tag `greenlang`
- **Twitter**: @GreenLangAI

### Get Support

- **Documentation**: https://docs.greenlang.ai
- **API Reference**: https://api.greenlang.ai/docs
- **Email Support**: support@greenlang.ai
- **Enterprise Support**: enterprise@greenlang.ai

---

## Congratulations!

You've successfully:
- ✅ Installed GreenLang Agent Foundation
- ✅ Created your first agent
- ✅ Learned basic operations
- ✅ Understood configuration options
- ✅ Run example agents

You're now ready to build production-grade AI agents with GreenLang!

---

**Next**: [Agent Development Guide](Agent_Development_Guide.md) →