# GL-008 TRAPCATCHER Architecture

## Overview

GL-008 TRAPCATCHER is a production-grade steam trap monitoring and diagnostic agent that implements zero-hallucination principles for industrial energy management.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GL-008 TRAPCATCHER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Acoustic  │    │   Thermal   │    │  Contextual │    │   External  │  │
│  │   Sensors   │    │   Cameras   │    │    Data     │    │    CMMS     │  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  │
│         │                  │                  │                  │         │
│         ▼                  ▼                  ▼                  ▼         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Integration Layer                                │   │
│  │   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────┐ │   │
│  │   │  OPC-UA      │ │   Modbus     │ │    MQTT      │ │  REST API  │ │   │
│  │   │  Connector   │ │   Connector  │ │   Connector  │ │  Connector │ │   │
│  │   └──────────────┘ └──────────────┘ └──────────────┘ └────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                   │                                         │
│                                   ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Core Processing Engine                          │   │
│  │                                                                      │   │
│  │   ┌──────────────────────────────────────────────────────────────┐  │   │
│  │   │                   Bounds Validator                            │  │   │
│  │   │        (Pressure: 0-25 bar, Temp: 273-523 K)                 │  │   │
│  │   └──────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                       │   │
│  │   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │   │
│  │   │  Acoustic  │  │   Thermal  │  │  Energy    │  │  Failure   │   │   │
│  │   │ Calculator │  │ Calculator │  │   Loss     │  │ Predictor  │   │   │
│  │   │            │  │            │  │ Calculator │  │  (Weibull) │   │   │
│  │   └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘   │   │
│  │         │               │               │               │          │   │
│  │         └───────────────┴───────────────┴───────────────┘          │   │
│  │                              │                                       │   │
│  │                              ▼                                       │   │
│  │   ┌──────────────────────────────────────────────────────────────┐  │   │
│  │   │               Trap State Classifier                          │  │   │
│  │   │        (Multimodal Late Fusion: 40/40/20 weights)            │  │   │
│  │   │                                                              │  │   │
│  │   │   ┌────────────┐  ┌────────────┐  ┌────────────┐            │  │   │
│  │   │   │  Acoustic  │  │  Thermal   │  │  Context   │            │  │   │
│  │   │   │   Score    │  │   Score    │  │   Score    │            │  │   │
│  │   │   │   (40%)    │  │   (40%)    │  │   (20%)    │            │  │   │
│  │   │   └────────────┘  └────────────┘  └────────────┘            │  │   │
│  │   └──────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                       │   │
│  └──────────────────────────────┼───────────────────────────────────────┘   │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Explainability Engine                             │   │
│  │                                                                      │   │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │   │     SHAP     │  │   Evidence   │  │  Counterfact │              │   │
│  │   │  Attribution │  │    Chain     │  │   Analysis   │              │   │
│  │   └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Output Layer                                 │   │
│  │                                                                      │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │   │
│  │   │  REST API   │  │  Prometheus │  │   Climate   │  │   CMMS    │  │   │
│  │   │  /diagnose  │  │   Metrics   │  │  Reporter   │  │  Updates  │  │   │
│  │   └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Integration Layer

Handles data acquisition from multiple industrial protocols:

| Connector | Protocol | Purpose |
|-----------|----------|---------|
| `acoustic_sensor_connector.py` | OPC-UA | Ultrasonic sensor data |
| `thermal_camera_connector.py` | Modbus TCP | IR thermal imagery |
| `trap_monitor_connector.py` | OPC-UA/Modbus | Armstrong, Spirax Sarco |
| `historian_connector.py` | REST/OPC-UA | OSIsoft PI, Honeywell |
| `cmms_connector.py` | REST | Maximo, SAP PM |

### 2. Calculators

Deterministic calculation engines:

| Calculator | Function | Standard |
|------------|----------|----------|
| `steam_trap_energy_loss_calculator.py` | Napier equation for steam loss | ASME PTC 39 |
| `acoustic_diagnostic_calculator.py` | Frequency band analysis | ISO 7841 Annex B |
| `temperature_differential_calculator.py` | ΔT analysis for blockage | DOE Protocol |
| `trap_population_analyzer.py` | Fleet health scoring | ISO 7841 |

### 3. Core Classifier

**Trap State Classifier** (`core/trap_state_classifier.py`)

Multimodal late fusion architecture:
- Acoustic modality: 40% weight
- Thermal modality: 40% weight
- Contextual modality: 20% weight

Classification states:
- `NORMAL`: Operating correctly
- `LEAKING`: Steam loss detected
- `BLOCKED`: Condensate backup
- `BLOWTHROUGH`: Failed open (continuous steam)
- `FAILED_OPEN`: Stuck open
- `FAILED_CLOSED`: Stuck closed

### 4. Explainability

**Diagnostic Explainer** (`explainability/diagnostic_explainer.py`)

- SHAP-compatible feature attribution
- Evidence chain with 5 elements (step, type, observation, inference, confidence)
- Counterfactual analysis ("what would change the diagnosis?")
- Three explanation styles: Technical, Operator, Executive

### 5. Monitoring

**Prometheus Metrics** (`monitoring/metrics.py`)

```
trapcatcher_diagnoses_total{trap_type, condition, severity}
trapcatcher_failures_detected_total{trap_type, failure_mode}
trapcatcher_energy_loss_kw{facility, area}
trapcatcher_co2_emissions_kg{facility, area}
trapcatcher_diagnosis_duration_seconds{trap_type}
trapcatcher_fleet_health_score{facility}
```

### 6. Bounds Validation

**Physical Bounds** (`core/bounds_validator.py`)

| Parameter | Min | Max | Unit | Standard |
|-----------|-----|-----|------|----------|
| Pressure | 0 | 25 | bar | ASME PTC 39 |
| Temperature | 273 | 523 | K | ASME PTC 39 |
| Acoustic Level | 0 | 120 | dB | ISO 7841 |
| Acoustic Freq | 20 | 100 | kHz | ISO 7841 |
| Flow Rate | 0 | 500 | kg/hr | ASME PTC 39 |

## Data Flow

```
1. Sensor Data Ingestion
   ├── Acoustic: 20-100 kHz ultrasonic
   ├── Thermal: IR camera frames
   └── Context: Operating hours, pressure, trap type

2. Validation
   ├── Bounds checking (physical limits)
   └── Quality flags (sensor health)

3. Feature Extraction
   ├── Acoustic: RMS, peak freq, spectral entropy
   ├── Thermal: ΔT inlet/outlet, gradient
   └── Context: Age, maintenance history

4. Classification
   ├── Per-modality scores
   └── Weighted fusion (40/40/20)

5. Energy Loss Calculation
   ├── Napier equation for steam flow
   └── Enthalpy lookup (IAPWS-IF97)

6. Explainability
   ├── SHAP values for each feature
   └── Evidence chain construction

7. Output
   ├── Classification result with confidence
   ├── Energy loss in kW and $/year
   ├── CO2e emissions
   └── Maintenance priority ranking
```

## Zero-Hallucination Guarantees

1. **Deterministic Calculations**: All formulas from ASME PTC 39 / ISO 7841
2. **No AI Inference**: Classification uses explicit weighted scoring
3. **Reproducibility**: Same inputs → identical outputs (SHA-256 verified)
4. **Frozen Dataclasses**: 92 immutable data structures
5. **Provenance Tracking**: Every calculation traceable with hashes

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe |
| `/health/ready` | GET | Readiness probe |
| `/metrics` | GET | Prometheus metrics |
| `/diagnose` | POST | Single trap diagnosis |
| `/analyze/fleet` | POST | Fleet analysis |
| `/status` | GET | Agent status |

## Deployment

### Kubernetes Resources

- Deployment with 2+ replicas
- HorizontalPodAutoscaler (2-10 pods, 70% CPU)
- PodDisruptionBudget (minAvailable: 1)
- ServiceMonitor for Prometheus
- Network policies for isolation

### Resource Requirements

| Resource | Request | Limit |
|----------|---------|-------|
| CPU | 250m | 1000m |
| Memory | 512Mi | 2Gi |

## Standards Compliance

- **ASME PTC 39**: Steam Traps - Performance Test Codes
- **ISO 7841**: Automatic steam traps - Steam loss determination
- **DOE Steam System Assessment Protocol**: Survey methodology
- **OpenMetrics**: Prometheus metrics format
- **OpenTelemetry**: Distributed tracing
