# GL-013 PREDICTMAINT - Quick Start Guide

```
================================================================================
                    QUICK START GUIDE - GL-013 PREDICTMAINT
                        Get Running in 5 Minutes
================================================================================
```

**Time to Complete:** 5-10 minutes
**Prerequisites:** Python 3.10+, Docker (optional)
**Difficulty:** Beginner

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Basic Configuration](#basic-configuration)
4. [First Prediction](#first-prediction)
5. [Core Operations](#core-operations)
6. [Next Steps](#next-steps)

---

## Prerequisites

Before starting, ensure you have:

- **Python 3.10 or higher** - Check with `python --version`
- **pip** - Python package manager
- **Git** - For cloning repository
- **Docker** (optional) - For containerized deployment

### Quick Prerequisite Check

```bash
# Check Python version
python --version
# Expected: Python 3.10.x or higher

# Check pip
pip --version
# Expected: pip 23.x or higher

# Check Git
git --version
# Expected: git version 2.x
```

---

## Installation

### Option 1: pip Install (Recommended)

```bash
# Install the GL-013 PREDICTMAINT package
pip install greenlang-gl013

# Verify installation
python -c "from gl_013 import PredictiveMaintenanceAgent; print('Installation successful!')"
```

### Option 2: From Source

```bash
# Clone repository
git clone https://github.com/greenlang/gl-013-predictmaint.git
cd gl-013-predictmaint

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from gl_013 import PredictiveMaintenanceAgent; print('Installation successful!')"
```

### Option 3: Docker

```bash
# Pull the official image
docker pull greenlang/gl-013:latest

# Run the container
docker run -d \
  --name gl-013-predictmaint \
  -p 8080:8080 \
  -p 9090:9090 \
  -e DETERMINISTIC_MODE=true \
  -e ZERO_HALLUCINATION=true \
  greenlang/gl-013:latest

# Verify container is running
docker ps | grep gl-013
```

---

## Basic Configuration

### Minimal Configuration

Create a configuration file `config.yaml`:

```yaml
# config.yaml - Minimal GL-013 Configuration
agent:
  id: GL-013
  codename: PREDICTMAINT
  version: "1.0.0"
  deterministic: true

runtime:
  timeout_seconds: 120
  cache_ttl_seconds: 300

compliance:
  zero_hallucination:
    enabled: true
```

### Environment Variables

Set required environment variables:

```bash
# Required
export GL013_LOG_LEVEL=INFO
export GL013_DETERMINISTIC_MODE=true
export GL013_ZERO_HALLUCINATION=true

# Optional
export GL013_CACHE_TTL=300
export GL013_METRICS_ENABLED=true
export GL013_METRICS_PORT=9090
```

---

## First Prediction

### Step 1: Import the Agent

```python
from gl_013 import PredictiveMaintenanceAgent, AgentConfig
from gl_013.calculators import RULCalculator, VibrationAnalyzer
from gl_013.calculators.constants import MachineClass
from decimal import Decimal
```

### Step 2: Initialize the Agent

```python
# Create configuration
config = AgentConfig(
    deterministic=True,
    seed=42,
    zero_hallucination=True
)

# Initialize agent
agent = PredictiveMaintenanceAgent(config)
print("Agent initialized successfully!")
```

### Step 3: Make Your First RUL Prediction

```python
# Initialize RUL Calculator
rul_calculator = RULCalculator(precision=6, store_provenance_records=True)

# Calculate RUL for a motor
result = rul_calculator.calculate_weibull_rul(
    equipment_type="motor_ac_induction_large",
    operating_hours=Decimal("50000"),
    current_health_score=Decimal("75.0"),
    confidence_level="95%"
)

# Print results
print(f"Equipment: Large AC Induction Motor")
print(f"Operating Hours: 50,000")
print(f"RUL: {result.rul_hours} hours ({result.rul_days} days)")
print(f"95% Confidence Interval: [{result.confidence_lower}, {result.confidence_upper}] hours")
print(f"Current Reliability: {result.current_reliability}")
print(f"Provenance Hash: {result.provenance_hash}")
```

**Expected Output:**

```
Equipment: Large AC Induction Motor
Operating Hours: 50,000
RUL: 28453.21 hours (1185.55 days)
95% Confidence Interval: [24185.23, 32721.19] hours
Current Reliability: 0.8523
Provenance Hash: sha256:abc123...
```

### Step 4: Analyze Vibration Data

```python
# Initialize Vibration Analyzer
vibration_analyzer = VibrationAnalyzer()

# Assess vibration severity per ISO 10816
result = vibration_analyzer.assess_severity(
    velocity_rms=Decimal("4.2"),  # mm/s RMS
    machine_class=MachineClass.CLASS_II
)

# Print results
print(f"Velocity RMS: 4.2 mm/s")
print(f"Machine Class: II (15-75 kW)")
print(f"ISO 10816 Zone: {result.zone.name}")
print(f"Alarm Level: {result.alarm_level.name}")
print(f"Assessment: {result.assessment}")
print(f"Recommendation: {result.recommendation}")
print(f"Margin to Zone C: {result.margin_to_next_zone} mm/s")
```

**Expected Output:**

```
Velocity RMS: 4.2 mm/s
Machine Class: II (15-75 kW)
ISO 10816 Zone: ZONE_B
Alarm Level: ALERT
Assessment: Acceptable for long-term operation
Recommendation: Continue monitoring, plan inspection
Margin to Zone C: 2.9 mm/s
```

---

## Core Operations

### Calculate Failure Probability

```python
from gl_013.calculators import FailureProbabilityCalculator

calculator = FailureProbabilityCalculator()

result = calculator.calculate(
    equipment_type="pump_centrifugal",
    operating_hours=Decimal("45000"),
    horizon_days=30
)

print(f"30-day Failure Probability: {result.probability_percent:.2f}%")
print(f"Risk Level: {result.risk_level}")
print(f"Recommended Action: {result.recommended_action}")
```

### Detect Anomalies

```python
from gl_013.calculators import AnomalyDetector

detector = AnomalyDetector()

result = detector.detect_zscore(
    current_value=Decimal("85.5"),
    historical_mean=Decimal("65.0"),
    historical_std=Decimal("5.0")
)

print(f"Z-Score: {result.z_score:.2f}")
print(f"Anomaly Detected: {result.is_anomaly}")
print(f"Severity: {result.severity}")
```

### Calculate Health Index

```python
from gl_013.calculators import HealthIndexCalculator

calculator = HealthIndexCalculator()

result = calculator.calculate(
    vibration_score=Decimal("78.5"),
    thermal_score=Decimal("82.0"),
    performance_score=Decimal("75.0"),
    weights={"vibration": 0.4, "thermal": 0.3, "performance": 0.3}
)

print(f"Overall Health Score: {result.overall_score:.1f}")
print(f"Health Status: {result.status}")
print(f"Trend: {result.trend}")
```

### Optimize Maintenance Schedule

```python
from gl_013.calculators import MaintenanceScheduler

scheduler = MaintenanceScheduler()

result = scheduler.optimize(
    equipment_id="PUMP-001",
    rul_hours=Decimal("2000"),
    failure_probability=Decimal("0.15"),
    preventive_cost=Decimal("5000"),
    corrective_cost=Decimal("50000")
)

print(f"Optimal Maintenance Date: {result.recommended_date}")
print(f"Cost Savings: ${result.cost_savings:,.2f}")
print(f"Priority: {result.priority}")
```

---

## Next Steps

### 1. Read the Full Documentation

- **README.md** - Complete feature overview
- **ARCHITECTURE.md** - System architecture and data flows
- **API Reference** - Full API documentation

### 2. Configure Integrations

Set up CMMS integration (SAP PM, Maximo, Oracle EAM):

```python
from gl_013.integrations import CMSSConnector

cmms = CMSSConnector(
    system_type="SAP_PM",
    base_url="https://sap.company.com/api",
    auth_type="oauth2",
    client_id="your_client_id",
    client_secret="your_client_secret"
)
```

### 3. Enable Monitoring

Configure Prometheus metrics:

```bash
# Start with metrics enabled
GL013_METRICS_ENABLED=true GL013_METRICS_PORT=9090 python -m gl_013

# Access metrics
curl http://localhost:9090/metrics
```

### 4. Deploy to Production

See deployment guide for Kubernetes deployment:

```bash
kubectl apply -f deployment/kubernetes/
```

### 5. Run Tests

Validate your setup with the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=gl_013 --cov-report=html
```

---

## Getting Help

- **Documentation:** https://docs.greenlang.io/gl-013
- **API Status:** https://status.greenlang.io
- **Support Email:** support@greenlang.io
- **Community Forum:** https://community.greenlang.io
- **GitHub Issues:** https://github.com/greenlang/gl-013-predictmaint/issues

---

## Quick Reference

### Common Commands

| Command | Description |
|---------|-------------|
| `pip install greenlang-gl013` | Install package |
| `python -m gl_013 --config config.yaml` | Start agent |
| `python -m gl_013.validate_config` | Validate configuration |
| `pytest tests/ -v` | Run tests |
| `curl localhost:9090/metrics` | Get Prometheus metrics |
| `curl localhost:8080/health` | Health check |

### Key Classes

| Class | Purpose |
|-------|---------|
| `PredictiveMaintenanceAgent` | Main agent class |
| `RULCalculator` | RUL prediction |
| `VibrationAnalyzer` | ISO 10816 analysis |
| `FailureProbabilityCalculator` | Failure prediction |
| `AnomalyDetector` | Anomaly detection |
| `HealthIndexCalculator` | Health scoring |
| `MaintenanceScheduler` | Schedule optimization |
| `CMSSConnector` | CMMS integration |

---

```
================================================================================
              You're Ready! Start predicting equipment failures today.
                         GL-013 PREDICTMAINT - GreenLang
================================================================================
```
