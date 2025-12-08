# GreenLang Getting Started Guide

**Document Version:** 1.0
**Last Updated:** December 2025
**Audience:** New Users, All Roles

---

## Table of Contents

1. [Introduction](#introduction)
2. [System Overview](#system-overview)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Core Concepts](#core-concepts)
7. [Your First Calculation](#your-first-calculation)
8. [Next Steps](#next-steps)

---

## Introduction

Welcome to GreenLang, an enterprise-grade platform for carbon emissions calculations, environmental compliance, and sustainability reporting. This guide will help you get up and running quickly with the platform.

### What is GreenLang?

GreenLang is a comprehensive framework designed for:

- **Carbon Emissions Calculations**: Accurate, auditable emissions calculations following GHG Protocol standards
- **Regulatory Compliance**: CBAM, EU ETS, and other environmental regulation support
- **Industrial Safety**: ISA-18.2 alarm management, NFPA 86 furnace compliance
- **ML-Powered Insights**: Explainable AI for emissions predictions and optimization
- **Enterprise Integration**: ERP connectors, webhooks, and API integrations

### Who Should Use This Guide?

This guide is for:

- **Operators** who will use GreenLang for daily calculations
- **Developers** who will integrate GreenLang into existing systems
- **Administrators** who will deploy and manage GreenLang installations
- **Analysts** who will use GreenLang for reporting and insights

---

## System Overview

### Architecture

GreenLang follows a modular, agent-based architecture:

```
+-------------------+     +-------------------+     +-------------------+
|    Data Sources   |     |   GreenLang Core  |     |     Outputs       |
+-------------------+     +-------------------+     +-------------------+
| - ERP Systems     |---->| - Agent Pipeline  |---->| - Reports         |
| - IoT Sensors     |     | - ML Models       |     | - Dashboards      |
| - Manual Input    |     | - Compliance      |     | - API Responses   |
| - CSV/Excel       |     | - Safety Systems  |     | - Webhooks        |
+-------------------+     +-------------------+     +-------------------+
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Agent Pipeline** | Modular processing units for calculations |
| **Provenance Tracker** | Cryptographic audit trail for all calculations |
| **ML Explainability** | LIME, SHAP, and causal inference for model transparency |
| **Safety Module** | ISA-18.2 alarms and NFPA 86 compliance |
| **API Layer** | REST and gRPC APIs for integration |

---

## Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.10+ | 3.11+ |
| **Memory** | 4 GB | 16 GB |
| **Storage** | 10 GB | 50 GB |
| **OS** | Linux/Windows/macOS | Linux (Ubuntu 22.04+) |

### Required Skills

- Basic Python knowledge
- Familiarity with REST APIs
- Understanding of environmental regulations (helpful but not required)

### Software Dependencies

```bash
# Core dependencies
python >= 3.10
pip >= 22.0
git >= 2.30

# Optional dependencies
docker >= 20.0    # For containerized deployment
postgresql >= 14  # For production database
redis >= 6.0      # For caching
```

---

## Installation

### Option 1: pip Install (Recommended for Development)

```bash
# Create virtual environment
python -m venv greenlang-env
source greenlang-env/bin/activate  # On Windows: greenlang-env\Scripts\activate

# Install GreenLang
pip install greenlang

# Verify installation
greenlang --version
```

### Option 2: From Source

```bash
# Clone repository
git clone https://github.com/greenlang/greenlang.git
cd greenlang

# Install dependencies
pip install -e ".[dev]"

# Run tests to verify
pytest tests/ -v
```

### Option 3: Docker

```bash
# Pull the official image
docker pull greenlang/greenlang:latest

# Run the container
docker run -d -p 8000:8000 --name greenlang greenlang/greenlang:latest

# Verify it's running
curl http://localhost:8000/api/v1/health
```

### Configuration

Create a configuration file at `~/.greenlang/config.yaml`:

```yaml
# GreenLang Configuration
environment: development

database:
  url: "sqlite:///greenlang.db"
  # For production: "postgresql://user:pass@host:5432/greenlang"

api:
  host: "0.0.0.0"
  port: 8000
  debug: false

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

emissions:
  default_region: "US"
  emission_factors_source: "EPA"
```

---

## Quick Start

### 1. Start the API Server

```bash
# Start the GreenLang API server
greenlang serve

# Output:
# INFO: GreenLang API server starting...
# INFO: Server running at http://localhost:8000
# INFO: API docs available at http://localhost:8000/docs
```

### 2. Verify the Installation

```bash
# Check health endpoint
curl http://localhost:8000/api/v1/health

# Expected response:
# {"status": "healthy", "version": "1.0.0", "timestamp": "2025-12-07T12:00:00Z"}
```

### 3. Run Your First Calculation

```bash
# Calculate emissions for diesel fuel consumption
curl -X POST http://localhost:8000/api/v1/calculations \
  -H "Content-Type: application/json" \
  -d '{
    "fuel_type": "diesel",
    "quantity": 1000,
    "unit": "liters",
    "region": "US",
    "sector": "stationary_combustion"
  }'

# Expected response:
# {
#   "calculation_id": "calc_abc123",
#   "emissions_kg_co2e": 2680.0,
#   "provenance_hash": "sha256:abc123...",
#   "methodology": "GHG Protocol Scope 1",
#   "timestamp": "2025-12-07T12:00:00Z"
# }
```

---

## Core Concepts

### 1. Agents

Agents are the fundamental processing units in GreenLang. Each agent performs a specific task:

```python
from greenlang.agents import CalculationAgent

# Create an agent
agent = CalculationAgent(config={
    "name": "diesel_calculator",
    "version": "1.0.0"
})

# Process data
result = agent.process({
    "fuel_type": "diesel",
    "quantity": 1000,
    "unit": "liters"
})

print(f"Emissions: {result.emissions_kg_co2e} kg CO2e")
```

### 2. Provenance Tracking

Every calculation in GreenLang includes cryptographic provenance for auditability:

```python
# Provenance hash ensures bit-perfect reproducibility
result = agent.process(input_data)

print(f"Provenance Hash: {result.provenance_hash}")
# sha256:3a7f8c2b...

# Re-running with same input produces identical hash
result2 = agent.process(input_data)
assert result.provenance_hash == result2.provenance_hash  # Always true
```

### 3. Pipelines

Combine multiple agents into processing pipelines:

```python
from greenlang.pipelines import Pipeline

# Create a pipeline
pipeline = Pipeline([
    DataValidationAgent(),
    EmissionFactorLookupAgent(),
    CalculationAgent(),
    ReportingAgent()
])

# Process through entire pipeline
final_result = pipeline.execute(raw_input_data)
```

### 4. Emission Factors

GreenLang uses standardized emission factors:

| Fuel Type | Region | Factor (kg CO2e/unit) | Source |
|-----------|--------|----------------------|--------|
| Diesel | US | 2.68/liter | EPA |
| Natural Gas | US | 1.93/liter | EPA |
| Coal | EU | 3.45/kg | EEA |
| Electricity | US | 0.42/kWh | EPA eGRID |

### 5. Regulatory Frameworks

GreenLang supports multiple regulatory frameworks:

- **GHG Protocol**: Scope 1, 2, 3 emissions
- **CBAM**: EU Carbon Border Adjustment Mechanism
- **EU ETS**: European Emissions Trading System
- **ISA-18.2**: Alarm management standard
- **NFPA 86**: Furnace safety standard

---

## Your First Calculation

Let's walk through a complete calculation example:

### Step 1: Prepare Input Data

```python
from greenlang import GreenLang

# Initialize the client
client = GreenLang()

# Prepare calculation input
calculation_input = {
    "facility_id": "FAC-001",
    "reporting_period": "2025-Q1",
    "fuel_consumption": [
        {
            "fuel_type": "diesel",
            "quantity": 5000,
            "unit": "liters",
            "source": "fleet_vehicles"
        },
        {
            "fuel_type": "natural_gas",
            "quantity": 10000,
            "unit": "cubic_meters",
            "source": "heating"
        }
    ],
    "electricity": {
        "consumption_kwh": 50000,
        "grid_region": "US-WECC"
    }
}
```

### Step 2: Run the Calculation

```python
# Execute calculation
result = client.calculate_emissions(calculation_input)

# Check results
print(f"Total Scope 1: {result.scope_1_emissions} kg CO2e")
print(f"Total Scope 2: {result.scope_2_emissions} kg CO2e")
print(f"Total Emissions: {result.total_emissions} kg CO2e")
print(f"Provenance: {result.provenance_hash}")
```

### Step 3: Verify and Export

```python
# Verify the calculation is reproducible
verification = client.verify_calculation(result.calculation_id)
print(f"Verified: {verification.is_valid}")

# Export to report format
report = client.export_report(
    calculation_id=result.calculation_id,
    format="pdf",
    template="ghg_protocol"
)

report.save("emissions_report_2025_Q1.pdf")
```

---

## Next Steps

### For Operators

- Continue to [02_operator_training.md](02_operator_training.md)
- Learn about daily calculation workflows
- Understand data validation and quality scoring

### For Developers

- Continue to [03_developer_training.md](03_developer_training.md)
- Learn about API integration patterns
- Explore custom agent development

### For Administrators

- Continue to [04_administrator_training.md](04_administrator_training.md)
- Learn about deployment and scaling
- Understand security configuration

### Hands-On Exercises

- Check out [training_exercises/](training_exercises/) for practical exercises
- Complete Exercise 01: Basic Calculations
- Complete Exercise 02: Pipeline Creation

### Additional Resources

| Resource | Description |
|----------|-------------|
| [API Documentation](../api/README.md) | Complete API reference |
| [Configuration Guide](../configuration/README.md) | Detailed configuration options |
| [Troubleshooting](05_troubleshooting_workshop.md) | Common issues and solutions |
| [FAQ](../faq.md) | Frequently asked questions |

---

## Getting Help

### Support Channels

- **Documentation**: https://docs.greenlang.io
- **GitHub Issues**: https://github.com/greenlang/greenlang/issues
- **Community Forum**: https://community.greenlang.io
- **Enterprise Support**: support@greenlang.io

### Reporting Issues

When reporting issues, please include:

1. GreenLang version (`greenlang --version`)
2. Python version (`python --version`)
3. Operating system
4. Steps to reproduce
5. Expected vs actual behavior
6. Relevant log output

---

**Congratulations!** You've completed the Getting Started guide. Continue to the next training module appropriate for your role.
