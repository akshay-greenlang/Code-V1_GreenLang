# GL-Agent-Factory Developer Guide

This guide covers everything you need to know to develop, extend, and contribute to GL-Agent-Factory.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Development Setup](#development-setup)
3. [Creating New Agents](#creating-new-agents)
4. [Working with Emission Factors](#working-with-emission-factors)
5. [Testing Guidelines](#testing-guidelines)
6. [Code Quality Standards](#code-quality-standards)
7. [API Development](#api-development)
8. [Deployment](#deployment)

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Gateway                              │
│                    (FastAPI + Authentication)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Agents    │  │  Execution  │  │   Batch     │              │
│  │  Registry   │  │   Engine    │  │  Processor  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  Calculation Engines                      │    │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐   │    │
│  │  │  GHG    │  │  Scope 3 │  │  QUDT   │  │  Monte   │   │    │
│  │  │ Protocol│  │ Cat 8-15 │  │  Units  │  │  Carlo   │   │    │
│  │  └─────────┘  └──────────┘  └─────────┘  └──────────┘   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  Data Layer                               │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │    │
│  │  │  Emission    │  │   Feature    │  │    IoC       │   │    │
│  │  │  Factors     │  │    Flags     │  │  Container   │   │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
├─────────────────────────────────────────────────────────────────┤
│                    PostgreSQL + Redis                            │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| Agent Registry | Central registry for all 143+ agents | `backend/agents/registry.py` |
| Emission Factors | Repository for EPA, DEFRA, IEA data | `backend/data/` |
| Calculation Engines | Scope 3, Monte Carlo, QUDT | `backend/engines/` |
| IoC Container | Dependency injection | `backend/core/container.py` |
| Feature Flags | Controlled rollouts | `backend/services/feature_flags.py` |

---

## Development Setup

### Prerequisites

```bash
# Required
python >= 3.11
node >= 18 (for CLI tools)
docker >= 24.0

# Recommended
postgresql >= 15
redis >= 7
```

### Environment Setup

```bash
# Clone repository
git clone https://github.com/greenlang/GL-Agent-Factory.git
cd GL-Agent-Factory

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment file
cp .env.example .env

# Start services
docker-compose up -d postgres redis

# Run migrations
alembic upgrade head

# Start development server
uvicorn backend.app:app --reload
```

### IDE Configuration

#### VS Code

Recommended extensions:
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Ruff (charliermarsh.ruff)

Settings (`.vscode/settings.json`):
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.formatting.provider": "none",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff"
  }
}
```

---

## Creating New Agents

### 1. Generate Agent Scaffold

```bash
python -m cli.agent_generator create \
  --id GL-XXX \
  --name "MY-AGENT-NAME" \
  --category "Category" \
  --type "Calculator"
```

### 2. Implement Agent Class

```python
# backend/agents/gl_xxx_my_agent/agent.py

from dataclasses import dataclass
from typing import Any, Dict
from decimal import Decimal

from backend.engines.base_calculator import BaseCalculator


@dataclass
class MyAgentInput:
    """Input schema with validation."""
    quantity: float
    unit: str
    region: str = "global"


@dataclass
class MyAgentOutput:
    """Output schema with provenance."""
    result: Decimal
    unit: str
    methodology: str
    confidence: float


class MyAgent(BaseCalculator):
    """
    GL-XXX: My Agent Description

    Purpose:
        Clear description of what this agent calculates.

    Methodology:
        Reference to standards and formulas used.
    """

    AGENT_ID = "GL-XXX"
    AGENT_NAME = "MY-AGENT-NAME"
    VERSION = "1.0.0"
    STANDARDS = ["GHG Protocol", "ISO 14064-1"]

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self._load_emission_factors()

    def calculate(self, inputs: MyAgentInput) -> MyAgentOutput:
        """Execute deterministic calculation."""
        self._validate_inputs(inputs)

        # Your calculation logic here
        result = self._perform_calculation(inputs)

        return MyAgentOutput(
            result=Decimal(str(result)),
            unit="tCO2e",
            methodology="ISO 14064-1:2018",
            confidence=0.95
        )

    def _validate_inputs(self, inputs: MyAgentInput) -> None:
        """Validate all inputs."""
        if inputs.quantity <= 0:
            raise ValueError("Quantity must be positive")

    def _perform_calculation(self, inputs: MyAgentInput) -> float:
        """Core calculation logic."""
        # Implement your formula
        emission_factor = self._get_emission_factor(inputs.region)
        return inputs.quantity * emission_factor

    def _get_emission_factor(self, region: str) -> float:
        """Get emission factor for region."""
        # Use emission factor service
        return 0.5  # Placeholder
```

### 3. Register Agent

Add to `backend/agents/registry.py`:

```python
AgentInfo(
    "GL-XXX",
    "MY-AGENT-NAME",
    "gl_xxx_my_agent",
    "MyAgent",
    "Category",
    "Calculator",
    "Medium",
    "P2"
),
```

### 4. Add Tests

```python
# backend/agents/gl_xxx_my_agent/tests/test_agent.py

import pytest
from decimal import Decimal
from ..agent import MyAgent, MyAgentInput


class TestMyAgent:
    @pytest.fixture
    def agent(self):
        return MyAgent()

    def test_basic_calculation(self, agent):
        inputs = MyAgentInput(quantity=100.0, unit="kWh")
        result = agent.calculate(inputs)

        assert result.result > 0
        assert result.unit == "tCO2e"

    def test_determinism(self, agent):
        """Same inputs must produce same outputs."""
        inputs = MyAgentInput(quantity=100.0, unit="kWh")

        result1 = agent.calculate(inputs)
        result2 = agent.calculate(inputs)

        assert result1.result == result2.result

    def test_validation_error(self, agent):
        with pytest.raises(ValueError):
            agent.calculate(MyAgentInput(quantity=-100, unit="kWh"))
```

---

## Working with Emission Factors

### Using EmissionFactorRepository

```python
from backend.data.emission_factor_repository import get_repository, FactorQuery
from backend.data.models import EmissionFactorSource

# Get repository instance
repo = get_repository()

# Query by source
query = FactorQuery(source=EmissionFactorSource.EPA, year=2024)
result = repo.find(query)

# Get specific factor
factor = repo.get_by_id("ef://epa/stationary/natural_gas/2024")

# Get grid factor
grid_factor = repo.get_grid_factor("US-WECC", year=2024)
```

### Adding New Emission Factors

1. Create JSON file in `backend/data/emission_factors/{source}/`
2. Follow the schema:

```json
{
  "factors": [
    {
      "id": "ef://source/category/fuel/year",
      "value": 53.06,
      "unit": "kg CO2e/unit",
      "source_document": "EPA GHG Emission Factors Hub",
      "source_url": "https://...",
      "year": 2024,
      "region": "US"
    }
  ]
}
```

---

## Testing Guidelines

### Test Categories

| Category | Location | Markers | Purpose |
|----------|----------|---------|---------|
| Unit | `*/tests/test_*.py` | `@pytest.mark.unit` | Component isolation |
| Integration | `tests/integration/` | `@pytest.mark.integration` | Service interaction |
| E2E | `tests/e2e/` | `@pytest.mark.e2e` | Full workflow |
| Golden | `tests/golden/` | `@pytest.mark.golden` | Determinism verification |

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=backend --cov-report=html

# Specific category
pytest -m integration

# Parallel execution
pytest -n auto

# Single agent
pytest backend/agents/gl_001_carbon_emissions/tests/
```

### Writing Good Tests

```python
class TestEmissionCalculation:
    """Follow Arrange-Act-Assert pattern."""

    def test_electricity_emission(self, agent, mock_ef_service):
        # Arrange
        inputs = create_test_inputs(quantity=1000)
        mock_ef_service.get_factor.return_value = create_test_factor()

        # Act
        result = agent.calculate(inputs)

        # Assert
        assert result.value == pytest.approx(400, rel=0.01)
        assert result.unit == "kg CO2e"
```

---

## Code Quality Standards

### Style Guide

We use **Ruff** for linting and formatting:

```bash
# Format code
ruff format .

# Check linting
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

### Type Hints

All code must include type hints:

```python
from typing import Dict, List, Optional

def calculate_emissions(
    activity_data: Dict[str, float],
    emission_factors: List[float],
    region: Optional[str] = None,
) -> float:
    """Calculate emissions with full type hints."""
    ...
```

### Pre-commit Hooks

Install pre-commit hooks:

```bash
pre-commit install
```

Hooks run automatically on commit:
- Ruff linting and formatting
- MyPy type checking
- pytest (quick tests only)

---

## API Development

### Adding New Endpoints

1. Create router in `backend/app/routers/`:

```python
# backend/app/routers/my_feature.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel

router = APIRouter(prefix="/my-feature", tags=["My Feature"])


class MyRequest(BaseModel):
    field: str


class MyResponse(BaseModel):
    result: str


@router.post("/action", response_model=MyResponse)
async def my_action(request: MyRequest):
    """Endpoint description."""
    return MyResponse(result="done")
```

2. Register in `backend/app/main.py`:

```python
from backend.app.routers import my_feature

app.include_router(my_feature.router)
```

### Using Dependency Injection

```python
from backend.core.container import get_container, inject

container = get_container()
container.register_singleton(IDatabase, PostgresDatabase)


@router.get("/data")
@inject
async def get_data(db: IDatabase):
    return await db.fetch_all()
```

---

## Deployment

### Docker Build

```bash
# Build image
docker build -t gl-agent-factory:latest .

# Run container
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  -e REDIS_URL=redis://... \
  gl-agent-factory:latest
```

### Kubernetes

```bash
# Apply manifests
kubectl apply -f k8s/

# Check status
kubectl get pods -l app=gl-agent-factory
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `REDIS_URL` | Yes | Redis connection string |
| `SECRET_KEY` | Yes | JWT signing key (min 32 chars) |
| `DEBUG` | No | Enable debug mode (default: false) |
| `LOG_LEVEL` | No | Logging level (default: INFO) |

---

## Getting Help

- **Slack**: #gl-agent-factory channel
- **Documentation**: [docs.greenlang.io](https://docs.greenlang.io)
- **Issues**: [GitHub Issues](https://github.com/greenlang/GL-Agent-Factory/issues)

---

*Happy coding!*
