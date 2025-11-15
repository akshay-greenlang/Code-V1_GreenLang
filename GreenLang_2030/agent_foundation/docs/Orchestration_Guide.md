# Orchestration Guide

## Multi-Agent Coordination and Workflows

Guide to coordinating multiple agents for complex workflows.

---

## Agent Orchestration

### Basic Orchestration

```python
from greenlang import AgentOrchestrator

# Create orchestrator
orchestrator = AgentOrchestrator()

# Register agents
orchestrator.register(DataCollectorAgent(config1), name='collector')
orchestrator.register(ProcessorAgent(config2), name='processor')
orchestrator.register(ReporterAgent(config3), name='reporter')

# Execute workflow
workflow = {
    'steps': [
        {'agent': 'collector', 'action': 'collect_data'},
        {'agent': 'processor', 'action': 'process_data'},
        {'agent': 'reporter', 'action': 'generate_report'}
    ]
}

result = await orchestrator.execute_workflow(workflow, {'source': 'database'})
```

---

## Coordination Patterns

### Pipeline Pattern

```python
async def execute_pipeline(agents: List[BaseAgent], data: Any) -> Any:
    """Execute agents in sequence."""
    result = data
    for agent in agents:
        result = await agent.process(result)
    return result
```

### Parallel Pattern

```python
async def execute_parallel(agents: List[BaseAgent], data: Any) -> List[Any]:
    """Execute agents in parallel."""
    tasks = [agent.process(data) for agent in agents]
    return await asyncio.gather(*tasks)
```

### Scatter-Gather Pattern

```python
async def scatter_gather(
    agents: List[BaseAgent],
    data_chunks: List[Any]
) -> Any:
    """Distribute work and aggregate results."""
    # Scatter
    tasks = [
        agent.process(chunk)
        for agent, chunk in zip(agents, data_chunks)
    ]

    # Gather
    results = await asyncio.gather(*tasks)

    # Aggregate
    return aggregate_results(results)
```

---

## Communication

### Message Passing

```python
class MessageBus:
    """Inter-agent message bus."""

    async def send(self, sender: str, recipient: str, message: Dict):
        """Send message between agents."""
        await self.broker.publish(
            topic=f"agent.{recipient}",
            message={
                'sender': sender,
                'payload': message,
                'timestamp': datetime.utcnow()
            }
        )

    async def subscribe(self, agent_id: str, handler: Callable):
        """Subscribe to messages."""
        await self.broker.subscribe(
            topic=f"agent.{agent_id}",
            callback=handler
        )
```

---

**Last Updated**: November 2024
**Version**: 1.0.0