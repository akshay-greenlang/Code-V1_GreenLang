# GL-010 EmissionsGuardian

**Enterprise-grade emissions compliance monitoring agent for industrial facilities**

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()
[![Standards](https://img.shields.io/badge/EPA-40%20CFR%2075-orange.svg)]()

## Overview

GL-010 EmissionsGuardian is a production-grade AI agent for continuous emissions monitoring and compliance management. It integrates CEMS data acquisition, regulatory compliance evaluation, RATA automation, fugitive emissions detection, and carbon trading recommendations.

### Key Capabilities

- **CEMS Integration**: Real-time data acquisition from Continuous Emissions Monitoring Systems
- **Compliance Engine**: EPA 40 CFR Part 75 compliant emission calculations
- **RATA Automation**: Relative Accuracy Test Audit support per Appendix A
- **Fugitive Detection**: ML-powered leak detection with SHAP/LIME explainability
- **Carbon Trading**: Offset tracking and trading recommendations
- **Audit Trails**: Complete provenance with SHA-256 hashing

### Zero-Hallucination Architecture

EmissionsGuardian follows strict zero-hallucination principles:

1. **Deterministic Calculations**: All compliance decisions use EPA-specified formulas
2. **ML for Detection Only**: Machine learning assists detection, never compliance decisions
3. **Human Approval Required**: Trading and reporting require explicit approval
4. **Complete Provenance**: Every calculation is traceable with cryptographic hashes

## Quick Start

### Installation

```bash
# Clone the repository
cd "GL Agents/GL-010_EmissionGuardian"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the agent
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080
```

### Docker

```bash
# Build the image
docker build -t gl-010-emissionguardian:1.0.0 .

# Run the container
docker run -d \
  --name emissionguardian \
  -p 8080:8080 \
  -e DATABASE_URL="postgresql://..." \
  -e KAFKA_BROKERS="kafka:9092" \
  gl-010-emissionguardian:1.0.0
```

### Configuration

Configure via environment variables or YAML:

```yaml
# config.yaml
agent:
  id: GL-010
  name: EmissionsGuardian
  mode: full  # monitoring, compliance, rata_support, fugitive_detection, trading, full

cems:
  polling_interval_seconds: 60
  buffer_size: 1000

compliance:
  enforce_read_only: true
  fail_closed: true

trading:
  auto_trade_enabled: false
  approval_threshold_usd: 10000

monitoring:
  prometheus_port: 9090
  health_check_interval: 30
```

## Architecture

```
GL-010_EmissionGuardian/
├── agent.py                 # Main agent orchestrator
├── api/                     # FastAPI REST endpoints
│   ├── main.py             # Application entry point
│   ├── routes_cems.py      # CEMS data endpoints
│   ├── routes_compliance.py # Compliance endpoints
│   ├── routes_rata.py      # RATA endpoints
│   ├── routes_fugitive.py  # Fugitive detection endpoints
│   ├── routes_trading.py   # Trading endpoints
│   └── routes_reports.py   # Reporting endpoints
├── calculators/            # EPA calculation engines
│   ├── rata_calculator.py  # RATA per Appendix A
│   ├── emission_rate.py    # Emission rate calculations
│   ├── mass_emissions.py   # Mass emissions per Part 75
│   ├── averaging.py        # Averaging period calculations
│   └── data_substitution.py # Missing data substitution
├── cems/                   # CEMS data handling
│   ├── data_acquisition.py # OPC-UA/Modbus connectors
│   ├── normalization.py    # Unit normalization
│   ├── quality_assurance.py # QA/QC checks
│   └── hourly_aggregation.py # Hour averaging
├── compliance/             # Regulatory compliance
│   ├── engine.py           # Compliance evaluation
│   ├── schemas.py          # Permit and rule schemas
│   └── rules_repository.py # Rule persistence
├── fugitive/               # Fugitive detection
│   ├── feature_engineering.py # Sensor feature extraction
│   ├── anomaly_detector.py # Anomaly detection models
│   ├── classifier.py       # Leak classification
│   └── explainability.py   # SHAP/LIME integration
├── trading/                # Carbon trading
│   ├── market_data.py      # ICE/CME market feeds
│   ├── recommendation_engine.py # Trading recommendations
│   ├── position_manager.py # Position tracking
│   ├── offset_tracker.py   # Offset management
│   └── risk_manager.py     # Trading risk controls
├── explainability/         # Decision explainability
│   ├── calculation_explainer.py # Calculation traces
│   ├── compliance_explainer.py  # Rule evaluation traces
│   └── ml_explainer.py     # SHAP/LIME wrappers
├── monitoring/             # Observability
│   ├── health.py           # Health checks
│   ├── metrics.py          # Prometheus metrics
│   ├── alerts.py           # Alert management
│   └── safety.py           # Safety controls
├── audit/                  # Audit logging
│   └── schemas.py          # Audit event schemas
└── tests/                  # Test suite
    ├── conftest.py         # Fixtures
    ├── test_unit/          # Unit tests
    └── test_integration/   # Integration tests
```

## API Reference

### CEMS Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/cems/readings` | Get current CEMS readings |
| POST | `/api/v1/cems/readings` | Submit CEMS readings |
| GET | `/api/v1/cems/hourly/{date}` | Get hourly averages |
| GET | `/api/v1/cems/quality/{reading_id}` | Get QA status |

### Compliance Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/compliance/status` | Current compliance status |
| GET | `/api/v1/compliance/exceedances` | List exceedance events |
| POST | `/api/v1/compliance/evaluate` | Evaluate reading against permit |
| GET | `/api/v1/compliance/report/{period}` | Generate compliance report |

### RATA Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/rata/perform` | Perform RATA calculation |
| GET | `/api/v1/rata/results/{test_id}` | Get RATA results |
| GET | `/api/v1/rata/history` | RATA test history |

### Fugitive Detection Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/fugitive/detect` | Analyze sensor readings |
| GET | `/api/v1/fugitive/alerts` | Active leak alerts |
| GET | `/api/v1/fugitive/explain/{alert_id}` | Get detection explanation |

### Trading Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/trading/market` | Current market prices |
| GET | `/api/v1/trading/positions` | Current positions |
| GET | `/api/v1/trading/recommendations` | Trading recommendations |
| POST | `/api/v1/trading/approve/{rec_id}` | Approve recommendation |

### Health & Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness probe |
| GET | `/health/ready` | Readiness probe |
| GET | `/metrics` | Prometheus metrics |

## Calculators

### RATA Calculator

Implements EPA 40 CFR Part 75 Appendix A Relative Accuracy Test Audit:

```python
from calculators.rata_calculator import perform_rata, RATAInput

# Input reference method and CEMS data
input_data = RATAInput(
    reference_method_values=[123.5, 125.2, ...],  # 9 runs minimum
    cems_values=[124.1, 124.8, ...],
    reference_method_uncertainty=2.0,  # %
)

result = perform_rata(input_data)
print(f"Relative Accuracy: {result.relative_accuracy_percent:.2f}%")
print(f"Bias: {result.bias_percent:.2f}%")
print(f"Pass: {result.passed}")
```

### Emission Rate Calculator

```python
from calculators.emission_rate import EmissionRateCalculator

calculator = EmissionRateCalculator()

# Calculate SO2 emission rate
result = calculator.calculate_so2_rate(
    so2_concentration_ppm=150.0,
    stack_flow_scfh=1_000_000.0,
    moisture_percent=8.0,
    o2_reference_percent=3.0,
)
print(f"SO2 Rate: {result.lb_per_hour:.2f} lb/hr")
```

## Compliance Engine

The compliance engine evaluates CEMS readings against permit limits:

```python
from compliance.engine import ComplianceEngine
from compliance.schemas import PermitRule, AveragingPeriod

engine = ComplianceEngine()

# Define permit rules
rule = PermitRule(
    pollutant="SO2",
    limit_value=200.0,
    limit_unit="ppm",
    averaging_period=AveragingPeriod.HOURLY,
    rolling_hours=1,
)

# Evaluate reading
status = engine.evaluate(reading, [rule])
if status.has_exceedance:
    print(f"Exceedance: {status.exceedance_events[0].description}")
```

## Fugitive Detection

ML-powered fugitive emissions detection with explainability:

```python
from fugitive import FeatureEngineer, AnomalyDetector, FugitiveClassifier

# Feature engineering
features = engineer.extract_features(sensor_readings, meteo_data)

# Anomaly detection
anomaly = detector.detect(features)
if anomaly.is_anomaly:
    # Classify the anomaly
    result = classifier.classify(features)

    # Get SHAP explanation
    explanation = explainer.explain(result)
    print(f"Top factors: {explanation.top_features}")
```

## Safety Controls

EmissionsGuardian implements fail-closed safety:

- **Read-Only OT Access**: Never writes to SCADA/DAHS systems
- **Approval Workflows**: Human approval for trading >$10k
- **Rate Limiting**: API rate limits prevent abuse
- **Circuit Breaker**: Automatic failsafe on component failures
- **Audit Logging**: All actions logged with provenance

## Standards Compliance

| Standard | Coverage |
|----------|----------|
| EPA 40 CFR Part 75 | CEMS, RATA, compliance calculations |
| EPA 40 CFR Part 60 | Emission standards (partial) |
| ISO 50001:2018 | Energy management integration |
| SOX Compliance | Financial trading audit trails |
| SOC 2 Type II | Security controls framework |

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_unit/test_calculators.py -v
```

### Code Quality

```bash
# Linting
ruff check .

# Type checking
mypy .

# Formatting
black .
```

## Deployment

### Kubernetes

```bash
# Apply manifests
kubectl apply -f deploy/kubernetes/

# Check status
kubectl get pods -l app=emissionguardian
```

### Helm

```bash
helm install emissionguardian deploy/helm/gl-010-emissionguardian \
  --set database.url="postgresql://..." \
  --set kafka.brokers="kafka:9092"
```

## Monitoring

### Prometheus Metrics

- `emissionguardian_readings_total` - Total CEMS readings processed
- `emissionguardian_exceedances_total` - Total exceedance events
- `emissionguardian_rata_tests_total` - RATA tests performed
- `emissionguardian_fugitive_alerts_total` - Fugitive alerts generated
- `emissionguardian_trading_recommendations_total` - Trading recommendations

### Grafana Dashboards

Import dashboards from `deploy/grafana/` for operational visibility.

## Support

- **Documentation**: See `/docs` directory
- **Issues**: Report issues on GitHub
- **Contact**: GreenLang Support Team

## License

MIT License - See LICENSE file for details.

---

**GL-010 EmissionsGuardian** - Zero-hallucination emissions compliance for industrial facilities.
