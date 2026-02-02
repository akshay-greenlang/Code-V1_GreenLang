# Multi-Agent Pipeline Application

Production-ready multi-agent pipeline with orchestration.

## Pipeline Stages

1. **Intake Agent**: Load and validate emissions data
2. **Calculator Agent**: Perform emissions calculations
3. **Reporting Agent**: Generate compliance reports

## Features

- Agent-to-agent communication
- Error handling and retries
- Complete observability
- Distributed processing support

## Quick Start

```python
from src.main import MultiAgentPipelineApplication

app = MultiAgentPipelineApplication()

result = await app.run_pipeline("data/emissions.csv")
print(f"Processed {result['result']['records']} records")
```
