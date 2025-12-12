# Contributing to GL-Agent-Factory

Thank you for your interest in contributing to GL-Agent-Factory! This document provides guidelines and instructions for contributing to the GreenLang Agent Factory platform.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Architecture](#project-architecture)
- [Contributing Guidelines](#contributing-guidelines)
- [Agent Development](#agent-development)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Style Guide](#style-guide)

---

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

---

## Getting Started

### Prerequisites

- **Python**: 3.11+ (required)
- **Node.js**: 18+ (for frontend/CLI tools)
- **Docker**: 24.0+ (for containerized development)
- **PostgreSQL**: 15+ (or Docker container)
- **Redis**: 7+ (for caching and queues)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/greenlang/GL-Agent-Factory.git
cd GL-Agent-Factory

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment configuration
cp .env.example .env

# Start development services
docker-compose up -d postgres redis

# Run database migrations
alembic upgrade head

# Start the development server
uvicorn backend.app:app --reload --port 8000
```

---

## Development Setup

### Environment Variables

Create a `.env` file with the following required variables:

```env
# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/gl_agent_factory
DATABASE_POOL_SIZE=20

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-min-32-chars
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# API Configuration
API_VERSION=v1
DEBUG=true
LOG_LEVEL=DEBUG

# External Services (optional for local dev)
ANTHROPIC_API_KEY=sk-ant-xxx  # For LLM features
```

### IDE Setup

#### VS Code (Recommended)

Install the recommended extensions:

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "charliermarsh.ruff",
    "tamasfe.even-better-toml",
    "redhat.vscode-yaml"
  ]
}
```

#### PyCharm

1. Set Python interpreter to your virtual environment
2. Enable Ruff for linting
3. Configure pytest as the test runner

---

## Project Architecture

```
GL-Agent-Factory/
├── backend/
│   ├── agents/           # Agent implementations (GL-001 to GL-100+)
│   │   ├── gl_001_carbon_emissions/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py       # Agent class implementation
│   │   │   ├── models.py      # Pydantic models
│   │   │   └── tests/         # Agent-specific tests
│   │   └── registry.py        # Central agent registry
│   ├── app/              # FastAPI application
│   │   ├── routers/      # API endpoints
│   │   ├── middleware/   # Request/response middleware
│   │   ├── gateway/      # API gateway
│   │   └── docs/         # OpenAPI configuration
│   ├── engines/          # Calculation engines
│   ├── services/         # Business logic services
│   ├── models/           # SQLAlchemy ORM models
│   ├── db/               # Database utilities
│   └── tests/            # Test suites
├── cli/                  # Command-line interface
├── docs/                 # Documentation
│   ├── api/              # API documentation
│   ├── guides/           # User guides
│   └── runbooks/         # Operational runbooks
├── k8s/                  # Kubernetes manifests
├── monitoring/           # Grafana/Prometheus configs
└── scripts/              # Utility scripts
```

### Key Components

| Component | Description |
|-----------|-------------|
| `AgentRegistry` | Central registry for all calculation agents |
| `CalculationEngine` | Deterministic calculation execution |
| `ProvenanceTracker` | Audit trail and lineage tracking |
| `TenantService` | Multi-tenancy management |
| `EventBus` | Async event processing via Redis Streams |

---

## Contributing Guidelines

### Types of Contributions

We welcome the following types of contributions:

1. **Bug Fixes**: Fix issues with existing functionality
2. **New Agents**: Add new calculation agents (follow agent template)
3. **Features**: Implement new platform capabilities
4. **Documentation**: Improve guides, API docs, and examples
5. **Tests**: Increase test coverage
6. **Performance**: Optimize existing code

### What We're Looking For

- Climate/sustainability domain expertise
- Zero-hallucination calculation methodology
- Regulatory compliance knowledge (CSRD, CBAM, EUDR, etc.)
- Process heat and industrial energy optimization

---

## Agent Development

### Creating a New Agent

1. **Generate scaffold** using the CLI:

```bash
python -m cli.agent_generator create \
  --id GL-XXX \
  --name "MY-NEW-AGENT" \
  --category "Category" \
  --type "Optimizer"
```

2. **Implement the agent class**:

```python
# backend/agents/gl_xxx_my_agent/agent.py
from dataclasses import dataclass
from typing import Any, Dict
from backend.engines.base_calculator import BaseCalculator

@dataclass
class MyAgentInput:
    """Input schema with validation."""
    value: float
    unit: str

@dataclass
class MyAgentOutput:
    """Output schema with provenance."""
    result: float
    unit: str
    methodology: str
    confidence: float

class MyNewAgent(BaseCalculator):
    """
    GL-XXX: My New Agent

    Purpose:
        [Clear description of what this agent calculates]

    Methodology:
        [Reference to standards, formulas, or methodologies used]

    Inputs:
        - value: The input value
        - unit: Unit of measurement

    Outputs:
        - result: Calculated result
        - methodology: Calculation methodology reference
    """

    AGENT_ID = "GL-XXX"
    AGENT_NAME = "MY-NEW-AGENT"
    VERSION = "1.0.0"

    # Standards compliance
    STANDARDS = ["ISO 14064-1", "GHG Protocol"]

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self._load_emission_factors()

    def calculate(self, inputs: MyAgentInput) -> MyAgentOutput:
        """
        Execute deterministic calculation.

        IMPORTANT: This method must be:
        - Deterministic (same inputs = same outputs)
        - Traceable (log all calculation steps)
        - Validated (check all inputs before calculation)
        """
        self._validate_inputs(inputs)

        # Calculation logic here
        result = self._perform_calculation(inputs)

        return MyAgentOutput(
            result=result,
            unit="tCO2e",
            methodology="ISO 14064-1:2018",
            confidence=0.95
        )

    def _validate_inputs(self, inputs: MyAgentInput) -> None:
        """Validate all inputs before calculation."""
        if inputs.value < 0:
            raise ValueError("Value must be non-negative")
        if inputs.unit not in self.SUPPORTED_UNITS:
            raise ValueError(f"Unsupported unit: {inputs.unit}")
```

3. **Register the agent** in `backend/agents/registry.py`:

```python
AgentInfo(
    "GL-XXX",
    "MY-NEW-AGENT",
    "gl_xxx_my_agent",
    "MyNewAgent",
    "Category",
    "Type",
    "Medium",  # Complexity
    "P2"       # Priority
),
```

4. **Add tests** (minimum 80% coverage required):

```python
# backend/agents/gl_xxx_my_agent/tests/test_agent.py
import pytest
from ..agent import MyNewAgent, MyAgentInput

class TestMyNewAgent:
    @pytest.fixture
    def agent(self):
        return MyNewAgent()

    def test_calculate_basic(self, agent):
        inputs = MyAgentInput(value=100.0, unit="kWh")
        result = agent.calculate(inputs)

        assert result.result > 0
        assert result.unit == "tCO2e"
        assert result.methodology is not None

    def test_determinism(self, agent):
        """Verify same inputs produce same outputs."""
        inputs = MyAgentInput(value=100.0, unit="kWh")

        result1 = agent.calculate(inputs)
        result2 = agent.calculate(inputs)

        assert result1.result == result2.result

    def test_invalid_input_raises(self, agent):
        with pytest.raises(ValueError):
            agent.calculate(MyAgentInput(value=-100, unit="kWh"))
```

### Agent Quality Requirements

All agents must meet these requirements before merge:

| Requirement | Threshold |
|-------------|-----------|
| Test Coverage | >= 80% |
| Determinism Tests | Required |
| Input Validation | Complete |
| Documentation | Docstrings + README |
| Type Hints | 100% |
| Linting | Zero errors |

---

## Testing Requirements

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend --cov-report=html

# Run specific test file
pytest backend/agents/gl_001_carbon_emissions/tests/

# Run integration tests
pytest tests/integration/ -m integration

# Run with parallel execution
pytest -n auto
```

### Test Categories

| Category | Location | Markers |
|----------|----------|---------|
| Unit Tests | `*/tests/test_*.py` | `@pytest.mark.unit` |
| Integration | `tests/integration/` | `@pytest.mark.integration` |
| E2E | `tests/e2e/` | `@pytest.mark.e2e` |
| Performance | `tests/performance/` | `@pytest.mark.slow` |

### Writing Good Tests

```python
class TestAgentCalculation:
    """Test class naming: Test{ComponentName}"""

    def test_function_does_expected_thing(self):
        """Test method naming: test_{function}_{scenario}"""
        # Arrange
        agent = MyAgent()
        inputs = create_test_inputs()

        # Act
        result = agent.calculate(inputs)

        # Assert
        assert result.value == expected_value
```

---

## Pull Request Process

### Before Submitting

1. **Ensure tests pass**: `pytest`
2. **Run linting**: `ruff check . && ruff format .`
3. **Type checking**: `mypy backend/`
4. **Update documentation** if needed
5. **Add changelog entry** in your PR description

### PR Template

```markdown
## Description
[Clear description of changes]

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Agent Changes (if applicable)
- [ ] New agent added: GL-XXX
- [ ] Existing agent modified: GL-XXX
- [ ] Registry updated

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests passing
- [ ] Coverage >= 80%

## Checklist
- [ ] Code follows project style guide
- [ ] Self-reviewed my code
- [ ] Commented hard-to-understand areas
- [ ] Documentation updated
- [ ] No new warnings generated
```

### Review Process

1. **Automated Checks**: CI must pass (tests, linting, type checking)
2. **Code Review**: At least 1 maintainer approval required
3. **Domain Review**: For agents, a domain expert review may be required
4. **Merge**: Squash and merge to maintain clean history

---

## Style Guide

### Python Style

We use **Ruff** for linting and formatting:

```bash
# Format code
ruff format .

# Check linting
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

Key rules:
- Line length: 100 characters
- Imports: Sorted with isort rules
- Quotes: Double quotes for strings
- Docstrings: Google style

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

### Documentation

- **Docstrings**: Required for all public functions/classes
- **README**: Each agent directory needs a README.md
- **API Docs**: Update OpenAPI specs for endpoint changes

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code change that neither fixes bug nor adds feature
- `test`: Adding missing tests
- `chore`: Maintenance tasks

Examples:
```
feat(agents): add GL-050 VFD optimizer agent
fix(registry): resolve duplicate agent ID handling
docs(contributing): update agent development guide
```

---

## Getting Help

- **Documentation**: Check [docs/](docs/) directory
- **Issues**: Search existing issues or create new one
- **Discussions**: Use GitHub Discussions for questions
- **Slack**: Join #gl-agent-factory channel (internal)

---

## Recognition

Contributors are recognized in:
- [CHANGELOG.md](CHANGELOG.md)
- GitHub contributors page
- Release notes

Thank you for contributing to sustainable technology!
