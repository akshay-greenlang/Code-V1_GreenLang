# GreenLang Developer Training

**Document Version:** 1.0
**Last Updated:** December 2025
**Audience:** Software Developers, Integration Engineers
**Prerequisites:** Completed [01_getting_started.md](01_getting_started.md), Python proficiency

---

## Table of Contents

1. [Introduction](#introduction)
2. [Development Environment Setup](#development-environment-setup)
3. [Architecture Deep Dive](#architecture-deep-dive)
4. [API Integration](#api-integration)
5. [Custom Agent Development](#custom-agent-development)
6. [Pipeline Development](#pipeline-development)
7. [ML Model Integration](#ml-model-integration)
8. [Testing Strategies](#testing-strategies)
9. [Performance Optimization](#performance-optimization)
10. [Security Best Practices](#security-best-practices)
11. [Deployment Patterns](#deployment-patterns)
12. [Advanced Topics](#advanced-topics)

---

## Introduction

This training module covers GreenLang development from an integration and extension perspective. Upon completion, you will be able to:

- Build custom integrations using REST and gRPC APIs
- Develop custom agents for domain-specific calculations
- Create processing pipelines for complex workflows
- Integrate ML models with explainability
- Write comprehensive tests (unit, integration, performance)
- Deploy and scale GreenLang applications

---

## Development Environment Setup

### Prerequisites

```bash
# Required tools
python >= 3.10
pip >= 22.0
git >= 2.30
docker >= 20.0
docker-compose >= 2.0

# Recommended IDE
Visual Studio Code with Python extension
# OR
PyCharm Professional
```

### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/greenlang/greenlang.git
cd greenlang

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install

# Verify setup
pytest tests/unit -v --tb=short
```

### Project Structure

```
greenlang/
├── greenlang/
│   ├── agents/           # Agent implementations
│   ├── core/             # Core framework
│   ├── infrastructure/   # API, events, connectors
│   ├── ml/               # Machine learning modules
│   └── safety/           # Safety and compliance
├── tests/
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   ├── contract/         # Contract tests
│   ├── chaos/            # Chaos tests
│   └── security/         # Security tests
├── docs/                 # Documentation
├── examples/             # Example code
└── pyproject.toml        # Project configuration
```

### IDE Configuration

For VS Code, create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.mypyEnabled": true,
    "python.linting.pylintEnabled": false,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "editor.formatOnSave": true
}
```

---

## Architecture Deep Dive

### Core Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                           API Layer                                  │
│  ┌─────────────────────┐  ┌─────────────────────┐                   │
│  │    REST Router      │  │    gRPC Service     │                   │
│  └─────────────────────┘  └─────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                         Agent Pipeline                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Validate │→│  Lookup  │→│ Calculate │→│  Report  │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                        Core Services                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │  Provenance  │  │     ML       │  │    Safety    │               │
│  │   Tracker    │  │  Explainer   │  │   Manager    │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                       Infrastructure                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Database │  │  Events  │  │  Cache   │  │   ERP    │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

### Agent Base Class

All agents inherit from `BaseAgent`:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from greenlang.core import ProvenanceTracker, AgentConfig

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass
class AgentResult(Generic[OutputT]):
    """Standard agent result container."""
    output: OutputT
    provenance_hash: str
    processing_time_ms: float
    metadata: dict


class BaseAgent(ABC, Generic[InputT, OutputT]):
    """Base class for all GreenLang agents."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.provenance = ProvenanceTracker()

    @abstractmethod
    def validate_input(self, input_data: InputT) -> bool:
        """Validate input data before processing."""
        pass

    @abstractmethod
    def process(self, input_data: InputT) -> AgentResult[OutputT]:
        """Main processing logic."""
        pass

    def execute(self, input_data: InputT) -> AgentResult[OutputT]:
        """Execute with provenance tracking."""
        if not self.validate_input(input_data):
            raise ValidationError("Input validation failed")

        # Track input for provenance
        self.provenance.track_input(input_data)

        # Process
        result = self.process(input_data)

        # Finalize provenance
        result.provenance_hash = self.provenance.finalize()

        return result
```

### Provenance System

The provenance system ensures bit-perfect reproducibility:

```python
from greenlang.core.provenance import ProvenanceTracker

tracker = ProvenanceTracker()

# Track all inputs
tracker.track_input({
    "fuel_type": "diesel",
    "quantity": 1000,
    "region": "US"
})

# Track emission factor used
tracker.track_parameter("emission_factor", 2.68, source="EPA_2024")

# Track calculation
tracker.track_calculation(
    formula="emissions = quantity * emission_factor",
    inputs={"quantity": 1000, "emission_factor": 2.68},
    result=2680.0
)

# Generate provenance hash
hash_value = tracker.finalize()
print(f"Provenance: {hash_value}")
# sha256:a1b2c3d4e5f6...
```

---

## API Integration

### REST API

#### Authentication

```python
import httpx

# API key authentication
client = httpx.Client(
    base_url="https://api.greenlang.io",
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

# OAuth2 authentication
from greenlang.auth import OAuth2Client

oauth = OAuth2Client(
    client_id="your_client_id",
    client_secret="your_client_secret",
    token_url="https://auth.greenlang.io/token"
)

client = httpx.Client(
    base_url="https://api.greenlang.io",
    auth=oauth
)
```

#### Making Requests

```python
# Create calculation
response = client.post(
    "/api/v1/calculations",
    json={
        "fuel_type": "diesel",
        "quantity": 1000,
        "unit": "liters",
        "region": "US"
    }
)
result = response.json()
print(f"Calculation ID: {result['calculation_id']}")

# Get calculation result
response = client.get(f"/api/v1/calculations/{result['calculation_id']}")
calculation = response.json()
print(f"Emissions: {calculation['emissions_kg_co2e']} kg CO2e")

# List calculations with pagination
response = client.get(
    "/api/v1/calculations",
    params={"limit": 100, "offset": 0, "status": "completed"}
)
```

#### Webhooks

```python
from greenlang.webhooks import WebhookManager

wm = WebhookManager()

# Register webhook
webhook = wm.register(
    url="https://your-app.com/webhook",
    events=["calculation.completed", "alarm.triggered"],
    secret="your_webhook_secret"
)

# Webhook payload structure
"""
{
    "event": "calculation.completed",
    "timestamp": "2025-12-07T12:00:00Z",
    "data": {
        "calculation_id": "calc_abc123",
        "emissions_kg_co2e": 2680.0,
        "provenance_hash": "sha256:abc..."
    },
    "signature": "sha256=..."  # HMAC signature
}
"""
```

### gRPC API

```python
import grpc
from greenlang.proto import calculation_pb2, calculation_pb2_grpc

# Create channel with authentication
credentials = grpc.ssl_channel_credentials()
call_credentials = grpc.access_token_call_credentials("YOUR_TOKEN")
composite_credentials = grpc.composite_channel_credentials(
    credentials, call_credentials
)

channel = grpc.secure_channel(
    "grpc.greenlang.io:443",
    composite_credentials
)

# Create stub
stub = calculation_pb2_grpc.CalculationServiceStub(channel)

# Make request
request = calculation_pb2.CalculateRequest(
    fuel_type="diesel",
    quantity=1000.0,
    unit="liters",
    region="US"
)

response = stub.Calculate(request)
print(f"Emissions: {response.emissions_kg_co2e}")

# Streaming calculations
def generate_requests():
    for record in data_records:
        yield calculation_pb2.CalculateRequest(
            fuel_type=record["fuel_type"],
            quantity=record["quantity"],
            unit=record["unit"]
        )

responses = stub.StreamCalculations(generate_requests())
for response in responses:
    print(f"Result: {response.emissions_kg_co2e}")
```

---

## Custom Agent Development

### Creating a Custom Agent

```python
from dataclasses import dataclass
from typing import List

from greenlang.agents.base import BaseAgent, AgentResult
from greenlang.core import AgentConfig, ValidationError


@dataclass
class CustomInput:
    """Input for CustomAgent."""
    activity_type: str
    activity_value: float
    custom_factor: float = None


@dataclass
class CustomOutput:
    """Output from CustomAgent."""
    emissions_kg_co2e: float
    emission_factor_used: float
    methodology: str


class CustomEmissionAgent(BaseAgent[CustomInput, CustomOutput]):
    """
    Custom agent for domain-specific emission calculations.

    This agent demonstrates how to create a custom calculation agent
    that integrates with GreenLang's provenance and quality systems.
    """

    # Default emission factors (customize for your domain)
    DEFAULT_FACTORS = {
        "process_a": 2.5,
        "process_b": 3.2,
        "process_c": 1.8,
    }

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.custom_factors = config.get("custom_factors", {})

    def validate_input(self, input_data: CustomInput) -> bool:
        """Validate input data."""
        if input_data.activity_type not in self.DEFAULT_FACTORS:
            if input_data.activity_type not in self.custom_factors:
                if input_data.custom_factor is None:
                    raise ValidationError(
                        f"Unknown activity type '{input_data.activity_type}' "
                        "and no custom_factor provided"
                    )

        if input_data.activity_value < 0:
            raise ValidationError("activity_value must be non-negative")

        return True

    def _get_emission_factor(self, activity_type: str, custom_factor: float = None) -> float:
        """Get emission factor for activity type."""
        if custom_factor is not None:
            return custom_factor
        if activity_type in self.custom_factors:
            return self.custom_factors[activity_type]
        return self.DEFAULT_FACTORS[activity_type]

    def process(self, input_data: CustomInput) -> AgentResult[CustomOutput]:
        """Process calculation."""
        import time
        start_time = time.time()

        # Get emission factor
        ef = self._get_emission_factor(
            input_data.activity_type,
            input_data.custom_factor
        )

        # Track in provenance
        self.provenance.track_parameter("emission_factor", ef)

        # Calculate emissions
        emissions = input_data.activity_value * ef

        # Track calculation
        self.provenance.track_calculation(
            formula="emissions = activity_value * emission_factor",
            inputs={
                "activity_value": input_data.activity_value,
                "emission_factor": ef
            },
            result=emissions
        )

        # Create output
        output = CustomOutput(
            emissions_kg_co2e=emissions,
            emission_factor_used=ef,
            methodology="Custom Activity-Based Method"
        )

        processing_time = (time.time() - start_time) * 1000

        return AgentResult(
            output=output,
            provenance_hash=self.provenance.finalize(),
            processing_time_ms=processing_time,
            metadata={"version": self.config.version}
        )


# Usage
config = AgentConfig(
    name="custom_emission_agent",
    version="1.0.0",
    custom_factors={"new_process": 4.0}
)

agent = CustomEmissionAgent(config)
result = agent.execute(CustomInput(
    activity_type="process_a",
    activity_value=1000
))

print(f"Emissions: {result.output.emissions_kg_co2e} kg CO2e")
print(f"Provenance: {result.provenance_hash}")
```

### Agent Registration

Register custom agents for pipeline use:

```python
from greenlang.registry import AgentRegistry

registry = AgentRegistry()

# Register the custom agent
registry.register(
    name="custom_emission",
    agent_class=CustomEmissionAgent,
    version="1.0.0",
    description="Custom domain-specific emission calculations"
)

# Use in pipeline
agent = registry.get("custom_emission")
```

---

## Pipeline Development

### Creating Pipelines

```python
from greenlang.pipelines import Pipeline, PipelineBuilder
from greenlang.agents import (
    DataValidationAgent,
    EmissionFactorLookupAgent,
    CalculationAgent,
    QualityScoreAgent,
    ReportingAgent
)


# Method 1: Direct construction
pipeline = Pipeline(
    name="standard_emissions",
    agents=[
        DataValidationAgent(),
        EmissionFactorLookupAgent(),
        CalculationAgent(),
        QualityScoreAgent(),
        ReportingAgent()
    ]
)

result = pipeline.execute(input_data)


# Method 2: Using builder pattern
pipeline = (
    PipelineBuilder("advanced_emissions")
    .add(DataValidationAgent())
    .add(EmissionFactorLookupAgent())
    .add(CalculationAgent())
    .add_conditional(
        condition=lambda x: x.requires_audit,
        agent=AuditTrailAgent()
    )
    .add(QualityScoreAgent())
    .add(ReportingAgent())
    .with_error_handler(ErrorRecoveryAgent())
    .with_retry(max_retries=3, backoff=2.0)
    .build()
)
```

### Parallel Processing

```python
from greenlang.pipelines import ParallelStage

# Run multiple agents in parallel
pipeline = Pipeline(
    name="parallel_processing",
    agents=[
        DataValidationAgent(),
        ParallelStage([
            Scope1CalculationAgent(),
            Scope2CalculationAgent(),
            Scope3CalculationAgent()
        ]),
        AggregationAgent(),
        ReportingAgent()
    ]
)
```

### Error Handling in Pipelines

```python
from greenlang.pipelines import Pipeline, RetryPolicy, CircuitBreaker

pipeline = Pipeline(
    name="resilient_pipeline",
    agents=[...],
    retry_policy=RetryPolicy(
        max_retries=3,
        backoff_seconds=2.0,
        retry_on=[NetworkError, TimeoutError]
    ),
    circuit_breaker=CircuitBreaker(
        failure_threshold=5,
        reset_timeout=60
    ),
    error_handler=lambda error, context: log_and_notify(error)
)
```

---

## ML Model Integration

### Using Built-in Explainability

```python
from greenlang.ml.explainability import LIMEExplainer, SHAPExplainer

# LIME explanations
lime_explainer = LIMEExplainer(model)
explanation = lime_explainer.explain(
    instance=input_data,
    num_features=10
)

print("Feature contributions:")
for feature, contribution in explanation.feature_contributions.items():
    print(f"  {feature}: {contribution:+.3f}")

# SHAP explanations
shap_explainer = SHAPExplainer(model)
shap_values = shap_explainer.explain(input_data)
shap_explainer.plot_waterfall(shap_values)
```

### Causal Inference

```python
from greenlang.ml.explainability import CausalInferenceEngine

engine = CausalInferenceEngine(model, data)

# Analyze causal relationships
causal_effects = engine.estimate_causal_effect(
    treatment="energy_efficiency_upgrade",
    outcome="emissions_reduction",
    confounders=["facility_size", "production_volume"]
)

print(f"Average Treatment Effect: {causal_effects.ate}")
print(f"95% CI: {causal_effects.confidence_interval}")
```

### Champion/Challenger Deployment

```python
from greenlang.ml import ChampionChallengerDeployment

deployment = ChampionChallengerDeployment(
    champion_model=current_model,
    challenger_model=new_model,
    traffic_split=0.9,  # 90% to champion
    evaluation_metrics=["accuracy", "latency_p99"]
)

# Deploy and monitor
deployment.start()

# Check performance
metrics = deployment.get_metrics()
if metrics.challenger_better():
    deployment.promote_challenger()
```

---

## Testing Strategies

### Unit Testing

```python
import pytest
from unittest.mock import Mock, patch

from greenlang.agents import CalculationAgent
from greenlang.core import AgentConfig


class TestCalculationAgent:
    """Unit tests for CalculationAgent."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        config = AgentConfig(name="test", version="1.0.0")
        return CalculationAgent(config)

    @pytest.fixture
    def valid_input(self):
        """Valid input fixture."""
        return {
            "fuel_type": "diesel",
            "quantity": 1000,
            "unit": "liters"
        }

    def test_process_valid_input(self, agent, valid_input):
        """Test processing with valid input."""
        result = agent.execute(valid_input)

        assert result.output.emissions_kg_co2e > 0
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_provenance_reproducibility(self, agent, valid_input):
        """Test that provenance is deterministic."""
        result1 = agent.execute(valid_input)
        result2 = agent.execute(valid_input)

        assert result1.provenance_hash == result2.provenance_hash

    @pytest.mark.parametrize("fuel,quantity,expected", [
        ("diesel", 1000, 2680.0),
        ("natural_gas", 500, 965.0),
        ("coal", 100, 345.0),
    ])
    def test_calculation_accuracy(self, agent, fuel, quantity, expected):
        """Test calculation accuracy against known values."""
        result = agent.execute({
            "fuel_type": fuel,
            "quantity": quantity,
            "unit": "liters"
        })

        assert result.output.emissions_kg_co2e == pytest.approx(expected, rel=1e-6)

    def test_invalid_input_raises_error(self, agent):
        """Test that invalid input raises ValidationError."""
        with pytest.raises(ValidationError):
            agent.execute({"fuel_type": "invalid"})
```

### Integration Testing

```python
import pytest
from testcontainers.postgres import PostgresContainer


class TestDatabaseIntegration:
    """Integration tests with real database."""

    @pytest.fixture(scope="class")
    def postgres(self):
        """Start PostgreSQL container for tests."""
        with PostgresContainer("postgres:14") as postgres:
            yield postgres

    @pytest.fixture
    def db_client(self, postgres):
        """Create database client."""
        return DatabaseClient(postgres.get_connection_url())

    @pytest.mark.integration
    def test_calculation_persistence(self, db_client):
        """Test calculation results are persisted correctly."""
        # Run calculation
        result = calculate_emissions(input_data)

        # Verify persistence
        stored = db_client.get_calculation(result.calculation_id)
        assert stored is not None
        assert stored.emissions_kg_co2e == result.emissions_kg_co2e
```

### Performance Testing

```python
import pytest
import time


class TestPerformance:
    """Performance benchmark tests."""

    @pytest.mark.performance
    def test_calculation_latency(self, agent, benchmark):
        """Test calculation meets latency target (<5ms)."""
        result = benchmark(agent.execute, valid_input)
        assert result.processing_time_ms < 5.0

    @pytest.mark.performance
    def test_throughput(self, agent):
        """Test throughput meets target (1000 records/sec)."""
        records = [generate_input() for _ in range(10000)]

        start = time.time()
        results = agent.process_batch(records)
        duration = time.time() - start

        throughput = len(records) / duration
        assert throughput >= 1000
```

---

## Performance Optimization

### Profiling

```python
import cProfile
import pstats
from pstats import SortKey

# Profile execution
profiler = cProfile.Profile()
profiler.enable()

result = pipeline.execute(large_dataset)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats(SortKey.CUMULATIVE)
stats.print_stats(20)
```

### Caching Strategies

```python
from greenlang.cache import CacheManager

cache = CacheManager(
    backend="redis",
    url="redis://localhost:6379",
    ttl=3600  # 1 hour
)

# Cache emission factors
@cache.cached(key_prefix="ef")
def get_emission_factor(fuel_type: str, region: str) -> float:
    return database.query_emission_factor(fuel_type, region)

# Cache calculation results
@cache.cached(key_prefix="calc", ttl=300)
def calculate_cached(input_hash: str) -> dict:
    return expensive_calculation(input_hash)
```

### Async Processing

```python
import asyncio
from greenlang.async_agents import AsyncCalculationAgent

async def process_batch_async(records):
    agent = AsyncCalculationAgent(config)

    # Process concurrently
    tasks = [agent.execute_async(record) for record in records]
    results = await asyncio.gather(*tasks)

    return results

# Run
results = asyncio.run(process_batch_async(large_batch))
```

---

## Security Best Practices

### Input Validation

```python
from pydantic import BaseModel, validator, constr, confloat


class CalculationInput(BaseModel):
    """Validated calculation input."""

    fuel_type: constr(min_length=1, max_length=50, regex=r'^[a-z_]+$')
    quantity: confloat(ge=0, le=1e9)
    unit: str
    region: constr(regex=r'^[A-Z]{2}$')

    @validator('fuel_type')
    def validate_fuel_type(cls, v):
        allowed = ['diesel', 'natural_gas', 'coal', 'gasoline']
        if v not in allowed:
            raise ValueError(f"fuel_type must be one of {allowed}")
        return v

    @validator('unit')
    def validate_unit(cls, v, values):
        valid_units = {
            'diesel': ['liters', 'gallons'],
            'natural_gas': ['cubic_meters', 'therms'],
        }
        fuel = values.get('fuel_type')
        if fuel and v not in valid_units.get(fuel, []):
            raise ValueError(f"Invalid unit for {fuel}")
        return v
```

### Secrets Management

```python
from greenlang.security import SecretManager

# Never hardcode secrets
secrets = SecretManager()

# Load from secure store
db_password = secrets.get("DATABASE_PASSWORD")
api_key = secrets.get("EXTERNAL_API_KEY")

# Rotate secrets
secrets.rotate("API_KEY", new_value=generate_key())
```

### Audit Logging

```python
from greenlang.security import AuditLogger

audit = AuditLogger()

# Log sensitive operations
audit.log(
    action="calculation.created",
    actor="user_123",
    resource="calc_abc123",
    details={"input_hash": "sha256:..."}
)

# Query audit trail
events = audit.query(
    start_time="2025-12-01",
    end_time="2025-12-07",
    action="calculation.*"
)
```

---

## Deployment Patterns

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "greenlang.api:app", "-b", "0.0.0.0:8000", "-w", "4"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  greenlang:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/greenlang
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  db:
    image: postgres:14
    environment:
      - POSTGRES_PASSWORD=secret
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine

volumes:
  pgdata:
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: greenlang
spec:
  replicas: 3
  selector:
    matchLabels:
      app: greenlang
  template:
    metadata:
      labels:
        app: greenlang
    spec:
      containers:
      - name: greenlang
        image: greenlang/greenlang:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "256Mi"
            cpu: "200m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
```

---

## Advanced Topics

### Custom Emission Factors

```python
from greenlang.factors import EmissionFactorRegistry

registry = EmissionFactorRegistry()

# Register custom factors
registry.register(
    name="custom_process",
    factor=4.5,
    unit="kg CO2e per unit",
    source="Internal Study 2025",
    valid_from="2025-01-01",
    uncertainty=0.15
)

# Use in calculations
factor = registry.get("custom_process", region="US", date="2025-06-01")
```

### Event-Driven Architecture

```python
from greenlang.events import EventBus, Event

bus = EventBus()

# Subscribe to events
@bus.subscribe("calculation.completed")
async def on_calculation_completed(event: Event):
    await notify_downstream(event.data)
    await update_dashboard(event.data)

# Publish events
await bus.publish(Event(
    type="calculation.completed",
    data={"calculation_id": "calc_123", "emissions": 2680.0}
))
```

### Multi-tenancy

```python
from greenlang.tenancy import TenantContext

# Set tenant context
with TenantContext(tenant_id="org_123"):
    # All operations scoped to tenant
    result = agent.execute(input_data)
    # Data automatically isolated
```

---

## Certification Checklist

### Knowledge Assessment

```
[ ] Understand GreenLang architecture
[ ] Know API authentication methods
[ ] Understand agent development patterns
[ ] Know pipeline construction
[ ] Understand provenance system
[ ] Know testing strategies
```

### Practical Skills

```
[ ] Build custom agent
[ ] Create processing pipeline
[ ] Integrate with REST API
[ ] Write unit and integration tests
[ ] Deploy with Docker
[ ] Implement error handling
```

### Exercises Completed

```
[ ] Exercise: Custom Agent Development
[ ] Exercise: Pipeline with Error Handling
[ ] Exercise: API Integration
[ ] Exercise: Performance Optimization
```

---

**Training Complete!** You are now ready to develop with GreenLang.
