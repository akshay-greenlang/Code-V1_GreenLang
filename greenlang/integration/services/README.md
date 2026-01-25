# GreenLang Shared Services

Core infrastructure services for sustainability applications extracted from production systems.

## Overview

The GreenLang Shared Services module provides reusable, production-tested services for carbon accounting, supply chain sustainability, and regulatory compliance applications.

### Extracted Services

1. **Factor Broker** - Multi-source emission factor resolution
2. **Entity MDM** - ML-powered entity master data management
3. **Methodologies** - Uncertainty quantification and data quality assessment
4. **PCF Exchange** - Product Carbon Footprint exchange protocols

### Agent Templates

1. **IntakeAgent** - Multi-format data ingestion
2. **CalculatorAgent** - Zero-hallucination calculations
3. **ReportingAgent** - Multi-format export with compliance

---

## Factor Broker Service

Runtime emission factor resolution with multi-source cascading, license compliance, and caching.

### Features

- **Multi-source cascading**: ecoinvent → DESNZ UK → EPA US → Proxy
- **License compliance**: 24-hour TTL caching for ecoinvent compliance
- **Intelligent caching**: Redis-based with automatic invalidation
- **Provenance tracking**: Full audit trail for every factor
- **Performance**: P95 latency <50ms with 85%+ cache hit rate

### Usage

```python
from greenlang.services import FactorBroker, FactorRequest

# Initialize broker
broker = FactorBroker()

# Request emission factor
request = FactorRequest(
    product="Steel",
    region="US",
    gwp_standard="AR6",
    unit="kg"
)

# Resolve factor
response = await broker.resolve(request)

print(f"Factor: {response.value} {response.unit}")
print(f"Source: {response.metadata.source}")
print(f"Quality: {response.data_quality_score}/100")
```

### Data Sources

| Source | Coverage | Priority | License |
|--------|----------|----------|---------|
| ecoinvent | 18,000+ products, global | 1 | Commercial |
| DESNZ UK | UK/EU specific | 2 | Open Government |
| EPA US | US specific | 3 | Public Domain |
| Proxy | Calculated fallback | 4 | Internal |

### Configuration

Set environment variables:

```bash
ECOINVENT_API_ENDPOINT=https://...
ECOINVENT_API_KEY=your_key
REDIS_HOST=localhost
REDIS_PORT=6379
CACHE_TTL_SECONDS=86400  # 24 hours
```

---

## Entity MDM Service

ML-powered entity resolution with two-stage matching and human-in-the-loop review.

### Features

- **Two-stage resolution**:
  - Stage 1: Fast vector similarity search (sentence-transformers)
  - Stage 2: Precision BERT re-ranking
- **Human review queue**: Low-confidence matches routed for review
- **External integrations**: LEI, DUNS, OpenCorporates connectors
- **Configurable thresholds**: Auto-match vs. review thresholds

### Usage

```python
from greenlang.services import EntityResolver, SupplierEntity

# Initialize resolver
resolver = EntityResolver()

# Resolve entity
query = SupplierEntity(
    entity_id="temp_123",
    name="Apple Inc",
)

result = resolver.resolve(query)

if result.status == "auto_matched":
    print(f"Matched: {result.matched_entity_id}")
    print(f"Confidence: {result.confidence:.2%}")
elif result.status == "pending_review":
    print(f"Needs review: {result.candidates[0].entity.name}")
```

### Performance

- **Throughput**: 1,000+ resolutions/second
- **Auto-match rate**: 85%+ with default thresholds
- **Accuracy**: 98%+ on validated test set

---

## Methodologies Service

Uncertainty quantification and data quality assessment following ILCD, GHG Protocol, and ISO standards.

### Features

- **Pedigree Matrix**: ILCD data quality assessment (5 dimensions)
- **Monte Carlo**: 10,000 iterations in <1 second
- **DQI Calculator**: Data Quality Indicators per ecoinvent methodology
- **Uncertainty propagation**: Multiple distribution types supported

### Usage

```python
from greenlang.services import (
    PedigreeMatrixEvaluator,
    MonteCarloSimulator,
    PedigreeScore
)

# Assess data quality
evaluator = PedigreeMatrixEvaluator()

pedigree = PedigreeScore(
    reliability=5,      # Verified data
    completeness=5,     # Complete dataset
    temporal=5,         # <3 years old
    geographical=5,     # Exact match
    technological=5     # Exact match
)

uncertainty = evaluator.calculate_uncertainty(pedigree)
print(f"Uncertainty: ±{uncertainty:.1%}")

# Run Monte Carlo simulation
simulator = MonteCarloSimulator(seed=42)

result = simulator.simulate(
    value=100.0,
    uncertainty=0.10,
    iterations=10000
)

print(f"Mean: {result.mean:.2f}")
print(f"P95: {result.p95:.2f}")
```

### Supported Distributions

- Normal (Gaussian)
- Lognormal
- Uniform
- Triangular
- Beta
- Gamma

---

## PCF Exchange Service

Product Carbon Footprint exchange following PACT Pathfinder v2.0, Catena-X, and SAP SDX.

### Features

- **PACT Pathfinder v2.0**: Full implementation of technical specifications
- **Catena-X**: Automotive industry data exchange
- **Validation**: Comprehensive PCF data validation
- **Versioning**: PCF version management

### Usage

```python
from greenlang.services import PCFExchangeService, PCFExchangeRequest

# Initialize service
service = PCFExchangeService()

# Export PCF
request = PCFExchangeRequest(
    operation="export",
    target_system="pact",
    pcf_data=my_pcf_data
)

response = await service.exchange(request)

if response.success:
    print(f"PCF exported: {response.exchange_id}")
```

### Supported Protocols

- **PACT Pathfinder v2.0**: Partnership for Carbon Transparency
- **Catena-X**: Automotive supply chain network
- **SAP SDX**: SAP Sustainability Data Exchange (planned)

---

## Agent Templates

### IntakeAgent

Multi-format data ingestion with validation and entity resolution.

```python
from greenlang.agents.templates import IntakeAgent

agent = IntakeAgent(schema=my_schema)

result = await agent.ingest(
    file_path="data.csv",
    format="csv",
    validate=True,
    resolve_entities=True
)

print(f"Rows read: {result.rows_read}")
print(f"Rows valid: {result.rows_valid}")
```

**Supported Formats**: CSV, JSON, Excel, XML, PDF, Parquet

### CalculatorAgent

Zero-hallucination calculations with full provenance.

```python
from greenlang.agents.templates import CalculatorAgent

agent = CalculatorAgent()

# Register formula
agent.register_formula(
    name="scope3_category1",
    formula=lambda quantity, factor: quantity * factor,
    required_inputs=["quantity", "factor"]
)

# Calculate
result = await agent.calculate(
    formula_name="scope3_category1",
    inputs={"quantity": 1000, "factor": 1.85},
    with_uncertainty=True
)

print(f"Result: {result.value} ± {result.uncertainty:.1%}")
print(f"Hash: {result.provenance.hash}")
```

### ReportingAgent

Multi-format export with compliance checking.

```python
from greenlang.agents.templates import ReportingAgent, ComplianceFramework

agent = ReportingAgent()

result = await agent.generate_report(
    data=my_dataframe,
    format="excel",
    output_path="report.xlsx",
    check_compliance=[ComplianceFramework.GHG_PROTOCOL]
)

for check in result.compliance_checks:
    print(f"{check.framework}: {'PASS' if check.passed else 'FAIL'}")
```

**Supported Formats**: JSON, Excel, CSV, HTML, PDF, XBRL

---

## Integration Patterns

### Pattern 1: Full Stack Integration

```python
from greenlang.services import FactorBroker, EntityResolver
from greenlang.agents.templates import IntakeAgent, CalculatorAgent, ReportingAgent

# Initialize services
factor_broker = FactorBroker()
entity_resolver = EntityResolver()

# Initialize agents
intake = IntakeAgent(entity_resolver=entity_resolver)
calculator = CalculatorAgent(factor_broker=factor_broker)
reporter = ReportingAgent()

# Workflow
data = await intake.ingest(file_path="input.csv")
results = await calculator.batch_calculate("scope3", data.data)
report = await reporter.generate_report(results, format="excel")
```

### Pattern 2: Service Composition

```python
from greenlang.services import FactorBroker, MonteCarloSimulator

broker = FactorBroker()
monte_carlo = MonteCarloSimulator()

# Resolve factor
factor = await broker.resolve(request)

# Quantify uncertainty
uncertainty = monte_carlo.simulate(
    value=factor.value,
    uncertainty=factor.uncertainty,
    iterations=10000
)
```

---

## Performance Characteristics

| Service | Throughput | Latency (P95) | Cache Hit Rate |
|---------|-----------|---------------|----------------|
| Factor Broker | 5,000 req/s | <50ms | 85%+ |
| Entity MDM | 1,000 req/s | <100ms | N/A |
| Methodologies | 10,000 calc/s | <10ms | N/A |
| PCF Exchange | 500 req/s | <200ms | N/A |

---

## Extension Mechanisms

### Custom Factor Sources

```python
from greenlang.services.factor_broker.sources import FactorSource

class MyCustomSource(FactorSource):
    async def fetch_factor(self, request):
        # Implement custom logic
        pass

    async def health_check(self):
        return {"status": "healthy"}
```

### Custom Compliance Rules

```python
agent = ReportingAgent(
    compliance_rules={
        "my_framework": {
            "required_fields": ["scope1", "scope2", "scope3"],
            "thresholds": {"total_emissions": 1000000}
        }
    }
)
```

---

## Migration from App-Specific Services

### Before (VCCI App)

```python
from services.factor_broker import FactorBroker
from entity_mdm.ml.resolver import EntityResolver
```

### After (Shared Services)

```python
from greenlang.services import FactorBroker, EntityResolver
```

Same API, zero code changes required beyond imports!

---

## Testing

Each service includes comprehensive tests:

```bash
# Run all service tests
pytest greenlang/services/

# Run specific service tests
pytest greenlang/services/factor_broker/tests/
pytest greenlang/services/entity_mdm/tests/
pytest greenlang/services/methodologies/tests/
```

---

## License

Proprietary - GreenLang Platform

**Note on Data Sources**:
- ecoinvent data requires commercial license
- DESNZ and EPA data are publicly available
- Respect all license terms when using services

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/greenlang/platform
- Documentation: https://docs.greenlang.com
- Email: support@greenlang.com

---

## Version History

### v1.0.0 (2025-01-26)

- Initial extraction from production applications
- Factor Broker service (5,530 lines)
- Entity MDM service with ML resolution
- Methodologies service (7,007 lines)
- PCF Exchange service (PACT Pathfinder v2.0)
- Agent templates (Intake, Calculator, Reporting)
