# Factory Guide

## Agent Factory Patterns

Guide to using the Agent Factory for rapid agent creation and deployment.

---

## Agent Factory

### Basic Usage

```python
from greenlang.factory import AgentFactory

# Create factory
factory = AgentFactory()

# Create agent from template
agent = factory.create(
    template='compliance',
    config={
        'name': 'csrd-agent',
        'regulation': 'CSRD',
        'company_type': 'large_enterprise'
    }
)

# Initialize and use
await agent.initialize()
result = await agent.process(data)
```

---

## Templates

### Available Templates

```python
templates = {
    'compliance': ComplianceAgentTemplate,
    'calculator': CalculatorAgentTemplate,
    'reporter': ReporterAgentTemplate,
    'analyzer': AnalyzerAgentTemplate,
    'integrator': IntegratorAgentTemplate
}
```

### Creating Custom Templates

```python
class CustomAgentTemplate:
    """Custom agent template."""

    @staticmethod
    def create(config: Dict[str, Any]) -> BaseAgent:
        """Create agent from template."""
        agent_config = AgentConfig(
            name=config['name'],
            capabilities=['reasoning', 'memory']
        )

        agent = CustomAgent(agent_config)

        # Configure agent
        agent.setup_memory()
        agent.setup_tools()
        agent.setup_llm()

        return agent
```

---

## Agent Configuration

### Configuration Schema

```python
from pydantic import BaseModel

class AgentFactoryConfig(BaseModel):
    """Factory configuration schema."""
    template: str
    name: str
    version: str = "1.0.0"
    capabilities: List[str]
    memory_config: Dict[str, Any]
    llm_config: Dict[str, Any]
    custom_tools: List[str] = []
```

---

## Batch Creation

### Create Multiple Agents

```python
async def create_agent_fleet(specs: List[Dict]) -> List[BaseAgent]:
    """Create multiple agents."""
    factory = AgentFactory()

    agents = []
    for spec in specs:
        agent = factory.create(
            template=spec['template'],
            config=spec['config']
        )
        await agent.initialize()
        agents.append(agent)

    return agents

# Usage
specs = [
    {'template': 'compliance', 'config': {'name': 'csrd'}},
    {'template': 'calculator', 'config': {'name': 'carbon'}},
    {'template': 'reporter', 'config': {'name': 'esg'}}
]

fleet = await create_agent_fleet(specs)
```

---

## Best Practices

1. **Use templates** for common agent types
2. **Validate configuration** before creation
3. **Initialize agents** after creation
4. **Register agents** with orchestrator
5. **Clean up** unused agents

---

**Last Updated**: November 2024
**Version**: 1.0.0