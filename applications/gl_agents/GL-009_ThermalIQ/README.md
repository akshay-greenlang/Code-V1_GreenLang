# GL-009 ThermalIQ - Industrial Thermal Efficiency & Exergy Analysis Agent

**Agent ID:** GL-009
**Codename:** THERMALIQ
**Type:** Calculator (Shared Calculation Library)
**Priority:** P1
**Version:** 2.0.0
**Status:** Production

---

## Overview

ThermalIQ is a **zero-hallucination** thermal fluid analysis agent providing deterministic First Law (energy) and Second Law (exergy) efficiency calculations for industrial thermal systems. Built for regulatory compliance and engineering precision, ThermalIQ guarantees reproducible, auditable results with complete SHA-256 provenance tracking.

### Core Value Proposition

Transform complex thermodynamic data into actionable insights with:
- Deterministic efficiency calculations (no LLM involvement in numeric computations)
- Interactive Sankey diagrams for energy/exergy flow visualization
- Intelligent fluid recommendations across 25+ heat transfer fluids
- SHAP/LIME explainability for ML-augmented predictions
- Full compliance with ASME PTC 4.1 and ISO 50001:2018 standards

---

## Key Features

### Thermodynamic Calculations

| Feature | Description | Standard |
|---------|-------------|----------|
| **First Law Efficiency** | Energy efficiency: `eta_I = Q_out / Q_in` | ASME PTC 4.1 |
| **Second Law Efficiency** | Exergy efficiency: `eta_II = Ex_out / Ex_in` | Bejan (2016) |
| **Carnot Factor Analysis** | Theoretical maximum: `theta = 1 - T_cold/T_hot` | Carnot (1824) |
| **Exergy Destruction** | Irreversibility quantification: `Ex_d = T0 * S_gen` | Kotas (1985) |
| **Heat Balance Closure** | Validate energy conservation within tolerance | ASME PTC 4.1 |
| **Uncertainty Quantification** | GUM-compliant propagation analysis | ISO/IEC Guide 98-3 |

### Thermal Fluid Library (25+ Fluids)

**Water/Steam**
- Water, Steam (IAPWS-IF97)

**Synthetic Heat Transfer Fluids**
- Therminol: 55, 59, 62, 66, VP-1
- Dowtherm: A, G, J, MX, Q, RP
- Syltherm: 800, XLT

**Glycol Solutions**
- Ethylene Glycol (20-60% concentrations)
- Propylene Glycol (20-60% concentrations)

**Molten Salts**
- Solar Salt (60% NaNO3, 40% KNO3)
- Hitec, Hitec XL

**Other**
- Mineral oils, Supercritical CO2, Refrigerants (R134a, R410A, R717, CO2)

### Visualization & Reporting

- **Sankey Diagrams**: Interactive energy/exergy flow visualization (Plotly-based)
- **Property Plots**: Temperature-dependent property charts
- **Efficiency Dashboards**: Real-time monitoring displays
- **PDF Report Generation**: Regulatory-compliant documentation

### ML Explainability

- **SHAP Analysis**: Feature importance and contribution visualization
- **LIME Explanations**: Local interpretable model-agnostic explanations
- **Engineering Rationale**: Context-aware recommendation narratives

### Integration Capabilities

- **REST API**: FastAPI-based endpoints with OpenAPI documentation
- **GraphQL**: Flexible query interface via Strawberry
- **Kafka Streaming**: Real-time event streaming (Avro schemas)
- **OPC-UA**: Industrial OT data ingestion
- **Process Historians**: OSIsoft PI, Honeywell PHD, AspenTech IP.21

---

## Directory Structure

```
GL-009_ThermalIQ/
|
|-- __init__.py                 # Agent metadata and exports
|-- pack.yaml                   # GreenLang Pack Manifest (v2.0)
|-- requirements.txt            # Python dependencies
|-- ARCHITECTURE.md             # Detailed architecture documentation
|-- README.md                   # This file
|
|-- core/                       # Core orchestration layer
|   |-- __init__.py
|   |-- orchestrator.py         # Main ThermalIQOrchestrator class
|   |-- config.py               # Configuration management
|   |-- schemas.py              # Pydantic data models
|   |-- handlers.py             # Request/response handlers
|
|-- calculators/                # Deterministic calculation engines
|   |-- __init__.py
|   |-- thermal_efficiency.py   # First Law efficiency calculations
|   |-- exergy_calculator.py    # Second Law (exergy) analysis
|   |-- heat_balance.py         # Heat balance closure verification
|   |-- uncertainty.py          # GUM-compliant uncertainty propagation
|
|-- fluids/                     # Thermal fluid property library
|   |-- __init__.py
|   |-- fluid_library.py        # 25+ fluid database with metadata
|   |-- property_correlations.py # Temperature-dependent correlations
|
|-- visualization/              # Visualization generators
|   |-- __init__.py
|   |-- sankey_generator.py     # Energy/exergy Sankey diagrams
|   |-- property_plots.py       # Fluid property visualization
|   |-- efficiency_dashboard.py # Real-time monitoring dashboards
|
|-- explainability/             # ML explainability modules
|   |-- __init__.py
|   |-- shap_explainer.py       # SHAP value computation
|   |-- lime_explainer.py       # LIME explanations
|   |-- engineering_rationale.py # Natural language rationale generation
|   |-- report_generator.py     # PDF/HTML report generation
|
|-- api/                        # API layer
|   |-- __init__.py
|   |-- rest_api.py             # FastAPI REST endpoints
|   |-- graphql_schema.py       # Strawberry GraphQL schema
|   |-- api_schemas.py          # API request/response models
|   |-- middleware.py           # Authentication, rate limiting, CORS
|
|-- streaming/                  # Event streaming
|   |-- __init__.py
|   |-- kafka_producer.py       # Kafka event publishing
|   |-- kafka_consumer.py       # Kafka event consumption
|   |-- event_schemas.py        # Avro event schemas
|
|-- deployment/                 # Deployment configurations
|   |-- deployment.yaml         # Kubernetes deployment manifest
|   |-- configmap.yaml          # Configuration ConfigMap
|
|-- tests/                      # Comprehensive test suite (85%+ coverage target)
    |-- __init__.py
    |-- conftest.py             # Pytest fixtures
    |-- test_thermal_efficiency.py
    |-- test_exergy_calculator.py
    |-- test_fluid_library.py
    |-- test_sankey_generator.py
    |-- test_explainability.py
    |-- test_orchestrator.py
    |-- test_api.py
    |-- test_golden_values.py   # Validated reference calculations
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/greenlang/agents.git
cd agents/gl-009-thermaliq

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### First Law Efficiency Calculation

```python
from calculators.thermal_efficiency import ThermalEfficiencyCalculator

# Initialize calculator
calc = ThermalEfficiencyCalculator(precision=3)

# Calculate First Law efficiency
result = calc.calculate_first_law_efficiency(
    heat_in=1000.0,   # kW - Total heat input
    heat_out=850.0    # kW - Useful heat output
)

print(f"First Law Efficiency: {result.efficiency_percent}%")
# Output: First Law Efficiency: 85.000%

print(f"Uncertainty: +/- {result.uncertainty_percent}%")
# Output: Uncertainty: +/- 1.803%

print(f"Provenance Hash: {result.provenance_hash[:16]}...")
# Output: Provenance Hash: a1b2c3d4e5f6g7h8...

# Access calculation steps for audit trail
for step in result.calculation_steps:
    print(f"Step {step.step_number}: {step.description}")
    print(f"  Formula: {step.formula}")
    print(f"  Result: {step.output_value} {step.output_unit}")
```

#### Second Law (Exergy) Efficiency Calculation

```python
from calculators.exergy_calculator import ExergyCalculator

# Initialize with dead state (reference environment)
calc = ExergyCalculator(
    T0=298.15,    # Dead state temperature (K) - 25C
    P0=101.325    # Dead state pressure (kPa) - 1 atm
)

# Calculate physical exergy
result = calc.calculate_physical_exergy(
    T=573.15,     # Stream temperature (K) - 300C
    P=500.0,      # Stream pressure (kPa)
    h=2961.0,     # Specific enthalpy (kJ/kg)
    s=7.271,      # Specific entropy (kJ/(kg*K))
    h0=104.9,     # Dead state enthalpy (kJ/kg)
    s0=0.367,     # Dead state entropy (kJ/(kg*K))
    mass_flow=2.5 # Mass flow rate (kg/s)
)

print(f"Physical Exergy: {result.exergy_kJ} kJ/kg")
print(f"Exergy Rate: {result.exergy_kW} kW")
print(f"Uncertainty: +/- {result.uncertainty_percent}%")
```

#### Exergy Destruction Analysis

```python
# Calculate exergy destruction (irreversibility)
destruction = calc.calculate_exergy_destruction(
    exergy_in=500.0,    # Exergy input (kW)
    exergy_out=350.0,   # Exergy output (kW)
    process_name="heat_exchanger"
)

print(f"Exergy Destroyed: {destruction.exergy_destroyed_kW} kW")
print(f"Destruction Ratio: {destruction.destruction_ratio}")
print(f"Entropy Generation: {destruction.entropy_generation_kW_K} kW/K")
print(f"Improvement Potential: {destruction.improvement_potential} kW")
```

#### Thermal Fluid Properties

```python
from fluids.fluid_library import ThermalFluidLibrary

# Initialize fluid library
library = ThermalFluidLibrary()

# Get properties for Therminol 66 at 300C
props = library.get_properties(
    fluid_name="therminol_66",
    T=573.15,       # Temperature (K)
    P=101.325       # Pressure (kPa)
)

print(f"Fluid: {props.fluid_name}")
print(f"Density: {props.density_kg_m3} kg/m3")
print(f"Specific Heat: {props.specific_heat_kJ_kg_K} kJ/(kg*K)")
print(f"Viscosity: {props.viscosity_mPa_s} mPa*s")
print(f"Thermal Conductivity: {props.conductivity_W_m_K} W/(m*K)")
print(f"Prandtl Number: {props.prandtl}")
print(f"Data Source: {props.data_source}")

# List available fluids
fluids = library.list_fluids()
print(f"Available fluids: {len(fluids)}")

# Get fluid recommendations for solar thermal application
recommendations = library.recommend_fluid(
    T_range=(473.15, 623.15),  # 200-350C
    application='solar_thermal'
)

for rec in recommendations[:3]:
    print(f"{rec.fluid_name}: Score {rec.score}")
    print(f"  Reasons: {rec.reasons}")
```

#### Heat Loss Breakdown

```python
from calculators.thermal_efficiency import ThermalEfficiencyCalculator

calc = ThermalEfficiencyCalculator()

# Define heat losses (per ASME PTC 4.1 categories)
losses = {
    "flue_gas_sensible": 50.0,    # kW
    "flue_gas_moisture": 15.0,    # kW
    "radiation": 20.0,            # kW
    "convection": 10.0,           # kW
    "unburned_carbon": 5.0,       # kW
}

breakdown = calc.calculate_heat_loss_breakdown(
    losses=losses,
    total_heat_input=1000.0  # kW
)

print(f"Total Loss: {breakdown.total_loss} kW")
print("\nLoss Percentages:")
for loss_type, pct in breakdown.loss_percentages.items():
    print(f"  {loss_type}: {pct}%")
print(f"\nProvenance: {breakdown.provenance_hash[:16]}...")
```

---

## API Reference

### REST API Endpoints

**Base URL:** `https://api.greenlang.io/v1/thermaliq`

#### Health Check

```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-09T10:30:00Z",
  "components": {
    "api": "ok",
    "fluid_database": "ok",
    "exergy_calculator": "ok",
    "sankey_generator": "ok"
  },
  "uptime_seconds": 86400.5
}
```

#### List Available Fluids

```http
GET /api/v1/fluids?category=thermal_oil
```

**Response:**
```json
{
  "fluids": [
    {
      "name": "Therminol66",
      "category": "thermal_oil",
      "molecular_weight_g_mol": 252.0,
      "critical_temperature_C": 600.0,
      "critical_pressure_kPa": 1500.0,
      "safety_class": "B1A"
    }
  ],
  "categories": ["thermal_oil", "refrigerant", "water_glycol"],
  "total_count": 8
}
```

#### Get Fluid Properties

```http
GET /api/v1/fluids/{fluid_name}/properties?temperature_C=300&pressure_kPa=500
```

**Response:**
```json
{
  "fluid_name": "Therminol66",
  "properties": {
    "temperature_C": 300.0,
    "pressure_kPa": 500.0,
    "phase": "liquid",
    "density_kg_m3": 850.5,
    "specific_heat_kJ_kgK": 2.35,
    "enthalpy_kJ_kg": 650.2,
    "entropy_kJ_kgK": 1.85,
    "viscosity_Pa_s": 0.0008,
    "thermal_conductivity_W_mK": 0.11,
    "prandtl_number": 17.2
  },
  "is_valid_state": true,
  "warnings": [],
  "data_source": "Eastman Technical Data",
  "computation_hash": "a1b2c3d4e5f6g7h8"
}
```

#### Calculate Thermal Efficiency

```http
POST /api/v1/efficiency
Content-Type: application/json
Authorization: Bearer {access_token}
```

**Request Body:**
```json
{
  "streams": [
    {
      "stream_id": "hot_oil_supply",
      "fluid_name": "Therminol66",
      "inlet_temperature_C": 350.0,
      "outlet_temperature_C": 250.0,
      "mass_flow_kg_s": 5.0,
      "specific_heat_kJ_kgK": 2.35
    },
    {
      "stream_id": "hot_oil_return",
      "fluid_name": "Therminol66",
      "inlet_temperature_C": 250.0,
      "outlet_temperature_C": 200.0,
      "mass_flow_kg_s": 5.0,
      "specific_heat_kJ_kgK": 2.30
    }
  ],
  "ambient_temperature_C": 25.0,
  "method": "combined"
}
```

**Response:**
```json
{
  "request_id": "req_abc123",
  "timestamp": "2025-01-09T10:35:00Z",
  "first_law_efficiency_percent": 87.5,
  "energy_input_kW": 1175.0,
  "energy_output_kW": 1028.1,
  "energy_loss_kW": 146.9,
  "second_law_efficiency_percent": 72.3,
  "exergy_input_kW": 485.2,
  "exergy_output_kW": 350.8,
  "exergy_destruction_kW": 134.4,
  "stream_efficiencies": [...],
  "computation_hash": "f8e7d6c5b4a39281",
  "method_used": "combined"
}
```

#### Perform Exergy Analysis

```http
POST /api/v1/exergy
Content-Type: application/json
Authorization: Bearer {access_token}
```

**Request Body:**
```json
{
  "streams": [...],
  "dead_state_temperature_C": 25.0,
  "dead_state_pressure_kPa": 101.325,
  "include_chemical_exergy": false,
  "include_kinetic_exergy": false,
  "include_potential_exergy": false
}
```

**Response:**
```json
{
  "request_id": "req_def456",
  "timestamp": "2025-01-09T10:40:00Z",
  "dead_state_temperature_C": 25.0,
  "dead_state_pressure_kPa": 101.325,
  "total_exergy_input_kW": 485.2,
  "total_exergy_output_kW": 350.8,
  "total_exergy_destruction_kW": 134.4,
  "exergy_efficiency_percent": 72.3,
  "physical_exergy_kW": 485.2,
  "components": [
    {
      "name": "hot_oil_supply",
      "exergy_input_kW": 320.5,
      "exergy_output_kW": 245.2,
      "exergy_destruction_kW": 75.3,
      "exergy_efficiency_percent": 76.5,
      "irreversibility_kW": 75.3
    }
  ],
  "improvement_potential_kW": 37.2,
  "computation_hash": "1a2b3c4d5e6f7890",
  "processing_time_ms": 45.2
}
```

#### Generate Sankey Diagram

```http
POST /api/v1/sankey
Content-Type: application/json
Authorization: Bearer {access_token}
```

**Request Body:**
```json
{
  "streams": [...],
  "diagram_type": "energy",
  "show_losses": true,
  "output_format": "plotly_json"
}
```

**Response:**
```json
{
  "request_id": "req_ghi789",
  "timestamp": "2025-01-09T10:45:00Z",
  "nodes": [
    {"id": "input", "name": "Energy Input", "value": 1175.0, "category": "input", "color": "#2196F3"},
    {"id": "output", "name": "Useful Output", "value": 1028.1, "category": "output", "color": "#8BC34A"},
    {"id": "losses", "name": "Total Losses", "value": 146.9, "category": "loss", "color": "#9E9E9E"}
  ],
  "links": [
    {"source": "input", "target": "process", "value": 1175.0, "label": "1175 kW"},
    {"source": "process", "target": "output", "value": 1028.1, "label": "1028 kW"},
    {"source": "process", "target": "losses", "value": 146.9, "label": "147 kW"}
  ],
  "total_input_kW": 1175.0,
  "total_output_kW": 1028.1,
  "total_losses_kW": 146.9,
  "diagram_type": "energy",
  "computation_hash": "abcdef1234567890"
}
```

#### Get Fluid Recommendations

```http
POST /api/v1/recommend-fluid
Content-Type: application/json
Authorization: Bearer {access_token}
```

**Request Body:**
```json
{
  "application": "solar_thermal",
  "min_temperature_C": 200.0,
  "max_temperature_C": 400.0,
  "max_gwp": 100,
  "require_non_flammable": false,
  "preferred_categories": ["thermal_oil"],
  "top_n": 5
}
```

**Response:**
```json
{
  "request_id": "req_jkl012",
  "timestamp": "2025-01-09T10:50:00Z",
  "application": "solar_thermal",
  "temperature_range_C": [200.0, 400.0],
  "recommendations": [
    {
      "fluid_name": "Therminol66",
      "category": "thermal_oil",
      "suitability_score": 92.5,
      "ranking": 1,
      "pros": ["Good temperature margin", "Low viscosity"],
      "cons": [],
      "notes": "Widely used in CSP applications"
    }
  ],
  "best_overall": "Therminol66",
  "best_environmental": "Water",
  "best_performance": "Therminol66"
}
```

---

## Regulatory Compliance

### Standards Compliance

| Standard | Description | Applicability |
|----------|-------------|---------------|
| **ASME PTC 4.1** | Steam Generating Units | Boiler efficiency calculations, heat loss methods |
| **ASME PTC 4** | Fired Steam Generators | Performance test codes |
| **ASME PTC 46** | Overall Plant Performance | Combined cycle efficiency |
| **ASME PTC 19.1** | Test Uncertainty | Uncertainty quantification methods |
| **ISO 50001:2018** | Energy Management Systems | Energy performance indicators (EnPIs) |
| **ISO 14414** | Pump System Energy Assessment | Auxiliary system efficiency |
| **API 560** | Fired Heaters | Refinery fired heater efficiency |
| **EN 12952** | Water-tube Boilers | European boiler efficiency standards |
| **EPA 40 CFR Part 60** | Emissions Monitoring | Flue gas analysis requirements |
| **IEC 61508** | Functional Safety | Safety-critical calculations |

### Thermodynamic References

| Reference | Application |
|-----------|-------------|
| Moran & Shapiro, *Fundamentals of Engineering Thermodynamics*, 9th Ed. | First Law efficiency |
| Bejan, *Advanced Engineering Thermodynamics*, 4th Ed. | Second Law (exergy) analysis |
| Kotas, *The Exergy Method of Thermal Plant Analysis* | Exergy destruction |
| Szargut, *Exergy Method: Technical and Ecological Applications* | Chemical exergy |
| IAPWS-IF97 | Water/steam properties |
| ASHRAE Handbook | Glycol solution properties |

---

## Zero-Hallucination Guarantees

ThermalIQ enforces strict zero-hallucination principles for all numerical calculations:

### Deterministic Calculation Path

```
+------------------+     +------------------+     +------------------+
|   User Input     | --> |  Deterministic   | --> |   Validated      |
|   (Validated)    |     |   Calculation    |     |   Output         |
+------------------+     +------------------+     +------------------+
                                |
                                v
                    +------------------------+
                    |   SHA-256 Provenance   |
                    |   Hash Generation      |
                    +------------------------+
```

### Guarantees

| Guarantee | Implementation |
|-----------|----------------|
| **DETERMINISTIC** | Same inputs always produce identical outputs |
| **REPRODUCIBLE** | Full provenance tracking with SHA-256 hashes |
| **AUDITABLE** | Complete calculation step trails |
| **STANDARDS-BASED** | All formulas from published, peer-reviewed sources |
| **NO LLM IN NUMERICS** | Zero hallucination risk in calculation path |

### Provenance Tracking

Every calculation includes a cryptographic hash for audit verification:

```python
# Example provenance verification
result = calc.calculate_first_law_efficiency(1000.0, 850.0)

# Provenance hash captures:
# - Formula version (first_law_efficiency_v1)
# - All input values
# - Calculation steps and intermediate results
# - Final output value
print(f"Provenance Hash: {result.provenance_hash}")
# Output: a1b2c3d4e5f6g7h8...

# Verify reproducibility
is_reproducible = calc.verify_reproducibility(result, 1000.0, 850.0)
assert is_reproducible == True  # Same hash guaranteed
```

### LLM Usage Constraints

LLMs are ONLY used for:
- Classification of efficiency improvement opportunities
- Natural language recommendation text generation
- Root cause classification for inefficiencies
- Report narrative synthesis

LLMs are NEVER used for:
- Numerical calculations
- Thermodynamic property lookups
- Efficiency calculations
- Exergy calculations

---

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Calculation Accuracy** | >= 99.5% | vs. validated test cases |
| **Property Lookup** | < 10 ms | Single property at T, P |
| **Efficiency Calculation** | < 100 ms | Full First/Second Law |
| **Exergy Analysis** | < 500 ms | Complete exergy breakdown |
| **Sankey Generation** | < 1 s | Energy/exergy diagram |
| **SHAP/LIME Explanation** | < 5 s | Feature importance analysis |
| **First Law Closure** | < 2% | Heat balance closure error |
| **Availability** | >= 99.9% | API uptime |
| **Error Rate** | < 0.1% | Calculation failures |
| **Test Coverage** | >= 85% | Code coverage target |

---

## Deployment

### Docker

```bash
# Build image
docker build -t greenlang/gl-009-thermaliq:2.0.0 .

# Run container
docker run -d \
  -p 8080:8080 \
  -p 9090:9090 \
  -e THERMALIQ_LOG_LEVEL=INFO \
  -e THERMALIQ_METRICS_PORT=9090 \
  greenlang/gl-009-thermaliq:2.0.0
```

### Kubernetes

```yaml
# Deploy using provided manifests
kubectl apply -f deployment/configmap.yaml
kubectl apply -f deployment/deployment.yaml
```

### Resource Requirements

| Resource | Minimum | Maximum |
|----------|---------|---------|
| CPU | 500m | 4000m |
| Memory | 512Mi | 4Gi |
| Replicas | 2 | 20 |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test modules
pytest tests/test_thermal_efficiency.py -v
pytest tests/test_exergy_calculator.py -v
pytest tests/test_fluid_library.py -v

# Run golden value validation tests
pytest tests/test_golden_values.py -v
```

---

## Dependencies

### Core Libraries

- **pydantic** >= 2.0.0 - Data validation
- **fastapi** >= 0.100.0 - REST API framework
- **numpy** >= 1.24.0 - Numerical computing
- **scipy** >= 1.10.0 - Scientific computing

### Thermodynamic Libraries

- **CoolProp** >= 6.4.0 - Fluid property calculations
- **iapws** >= 1.5.0 - IAPWS-IF97 water/steam

### Visualization

- **plotly** >= 5.15.0 - Interactive diagrams
- **matplotlib** >= 3.7.0 - Static plots
- **kaleido** >= 0.2.1 - Static image export

### ML/Explainability

- **shap** >= 0.42.0 - SHAP explanations
- **lime** >= 0.2.0 - LIME explanations
- **scikit-learn** >= 1.2.0 - ML utilities

---

## Support

- **Documentation:** https://docs.greenlang.io/agents/gl-009
- **API Status:** https://status.greenlang.io
- **Support:** process-heat@greenlang.ai
- **Repository:** https://github.com/greenlang/agents/gl-009-thermaliq

---

## License

Copyright (c) 2025 GreenLang. All rights reserved.

Licensed under the Apache License, Version 2.0. See LICENSE file for details.

---

## Changelog

### v2.0.0 (2025-Q4)
- Added 25+ thermal fluid library
- SHAP/LIME explainability integration
- Kafka streaming support
- GraphQL API endpoints
- Enhanced Sankey diagram generation
- Comprehensive uncertainty quantification

### v1.0.0 (2025-Q2)
- Initial release
- First Law efficiency calculations
- Basic exergy analysis
- REST API endpoints
