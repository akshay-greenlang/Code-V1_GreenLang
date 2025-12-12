# GL-Agent-Factory Quick Start Guide

Get up and running with GL-Agent-Factory in 5 minutes.

## Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- PostgreSQL 15+ (or use Docker)
- Redis 7+ (for caching)

## Installation

### Option 1: pip install (Recommended)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install GL-Agent-Factory
pip install gl-agent-factory

# Or install from source
git clone https://github.com/greenlang/GL-Agent-Factory.git
cd GL-Agent-Factory
pip install -e ".[dev]"
```

### Option 2: Docker

```bash
# Pull the official image
docker pull greenlang/gl-agent-factory:latest

# Run with default settings
docker run -p 8000:8000 greenlang/gl-agent-factory
```

## Quick Configuration

Create a `.env` file in your project root:

```env
# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/gl_agent_factory

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-min-32-chars

# API
DEBUG=true
LOG_LEVEL=INFO
```

## Your First Calculation

### Using the Python SDK

```python
from gl_agent_factory import AgentRegistry, EmissionFactorService

# Initialize services
registry = AgentRegistry()
ef_service = EmissionFactorService()

# Get the Carbon Emissions calculator
agent = registry.get_agent("GL-001")

# Calculate emissions from electricity usage
result = agent.calculate({
    "activity_type": "electricity",
    "quantity": 1000,  # 1000 kWh
    "unit": "kWh",
    "region": "US-WECC",
    "year": 2024
})

print(f"Carbon Footprint: {result.carbon_footprint} tCO2e")
print(f"Methodology: {result.methodology}")
```

### Using the REST API

```bash
# Start the API server
uvicorn backend.app:app --reload --port 8000

# Make a calculation request
curl -X POST "http://localhost:8000/v1/agents/GL-001/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "activity_type": "electricity",
      "quantity": 1000,
      "unit": "kWh",
      "region": "US-WECC"
    }
  }'
```

## Available Agents

GL-Agent-Factory includes 143+ specialized agents:

### Climate & Compliance (GL-001 to GL-013)

| Agent | Name | Description |
|-------|------|-------------|
| GL-001 | Carbon Emissions | GHG Protocol Scope 1, 2, 3 calculations |
| GL-002 | CBAM Compliance | EU Carbon Border Adjustment Mechanism |
| GL-003 | CSRD Reporting | Corporate Sustainability Reporting Directive |
| GL-006 | Scope 3 Emissions | Full supply chain emissions |

### Process Heat (GL-020 to GL-100)

| Agent | Name | Description |
|-------|------|-------------|
| GL-025 | Cogeneration | Combined heat and power optimization |
| GL-031 | Furnace Guardian | Industrial furnace monitoring |
| GL-040 | Safety Monitor | Process safety compliance |

List all agents:

```python
from gl_agent_factory import AgentRegistry

registry = AgentRegistry()
stats = registry.get_statistics()
print(f"Total Agents: {stats['total_agents']}")

# Filter by category
emissions_agents = registry.list_agents(category="Emissions")
```

## Common Use Cases

### 1. Calculate Scope 1 Emissions

```python
result = registry.get_agent("GL-001").calculate({
    "activity_type": "stationary_combustion",
    "fuel_type": "natural_gas",
    "quantity": 1000,
    "unit": "therms",
    "region": "US"
})
```

### 2. Calculate Scope 2 Emissions

```python
result = registry.get_agent("GL-001").calculate({
    "activity_type": "electricity",
    "quantity": 50000,
    "unit": "kWh",
    "region": "GB",  # UK grid
    "method": "location_based"
})
```

### 3. CBAM Calculation

```python
result = registry.get_agent("GL-002").calculate({
    "imports": [
        {
            "cn_code": "7208",
            "product": "Steel plates",
            "quantity_tonnes": 100,
            "origin_country": "CN"
        }
    ],
    "reporting_period": "2024-Q1"
})
```

## Testing Your Setup

Run the test suite to verify installation:

```bash
# Run all tests
pytest

# Run specific category
pytest tests/integration/ -v

# Run with coverage
pytest --cov=backend --cov-report=html
```

## Next Steps

1. **Read the full documentation**: [docs.greenlang.io](https://docs.greenlang.io)
2. **Explore agent examples**: See `examples/` directory
3. **Join the community**: [Discord](https://discord.gg/greenlang)
4. **Report issues**: [GitHub Issues](https://github.com/greenlang/GL-Agent-Factory/issues)

## Getting Help

- **Documentation**: [https://docs.greenlang.io](https://docs.greenlang.io)
- **API Reference**: [https://api.greenlang.io/docs](https://api.greenlang.io/docs)
- **Examples**: See the `examples/` directory
- **Support**: api-support@greenlang.io

---

*Happy calculating!*
