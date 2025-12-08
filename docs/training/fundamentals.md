# GreenLang Fundamentals

## Module Overview

**Duration:** 4 hours
**Prerequisites:** None
**Level:** Beginner

This module provides a comprehensive introduction to the GreenLang Process Heat Agents platform, covering architecture, core concepts, process heat domain knowledge, and foundational ML/AI concepts.

---

## Part 1: GreenLang Architecture Overview

### 1.1 What is GreenLang?

GreenLang is an enterprise-grade AI platform designed specifically for process heat management in industrial settings. It combines advanced machine learning capabilities with deep domain expertise to optimize energy efficiency, ensure safety compliance, and reduce operational costs.

**Key Value Propositions:**

| Benefit | Description | Typical Impact |
|---------|-------------|----------------|
| Energy Efficiency | AI-optimized heat management | 15-30% energy reduction |
| Safety Compliance | Automated safety monitoring | 99.9% compliance rate |
| Predictive Maintenance | Failure prediction and prevention | 40% reduction in unplanned downtime |
| Operational Excellence | Intelligent process optimization | 20% productivity improvement |

### 1.2 System Architecture

```
+------------------------------------------------------------------+
|                        GreenLang Platform                         |
+------------------------------------------------------------------+
|                                                                    |
|  +------------------+  +------------------+  +------------------+  |
|  |   Agent Layer    |  |   Agent Layer    |  |   Agent Layer    |  |
|  |                  |  |                  |  |                  |  |
|  | Process Heat     |  | Safety           |  | Optimization     |  |
|  | Monitoring Agent |  | Compliance Agent |  | Agent            |  |
|  +--------+---------+  +--------+---------+  +--------+---------+  |
|           |                     |                     |            |
|  +--------v---------------------v---------------------v---------+  |
|  |                   Orchestration Layer                        |  |
|  |                                                               |  |
|  |  - Agent Coordination    - Workflow Management               |  |
|  |  - Event Processing      - State Management                  |  |
|  +---------------------------+----------------------------------+  |
|                              |                                     |
|  +---------------------------v----------------------------------+  |
|  |                      ML/AI Engine                            |  |
|  |                                                               |  |
|  |  - Predictive Models     - Anomaly Detection                 |  |
|  |  - Causal Inference      - Explainability                    |  |
|  +---------------------------+----------------------------------+  |
|                              |                                     |
|  +---------------------------v----------------------------------+  |
|  |                   Infrastructure Layer                       |  |
|  |                                                               |  |
|  |  - Data Ingestion        - API Gateway                       |  |
|  |  - Event Streaming       - Security & Auth                   |  |
|  +--------------------------------------------------------------+  |
|                                                                    |
+------------------------------------------------------------------+
```

### 1.3 Architecture Layers

#### Agent Layer

The Agent Layer contains specialized AI agents that perform specific functions:

| Agent Type | Function | Key Capabilities |
|------------|----------|------------------|
| Process Heat Monitoring | Real-time temperature and heat flow monitoring | Anomaly detection, trend analysis |
| Safety Compliance | Regulatory compliance monitoring | ISA 18.2, NFPA 86, OSHA PSM |
| Optimization | Process optimization recommendations | Energy efficiency, throughput |
| Predictive Maintenance | Equipment health monitoring | Failure prediction, maintenance scheduling |

#### Orchestration Layer

The Orchestration Layer coordinates agent activities:

- **Agent Coordination:** Manages communication between agents
- **Workflow Management:** Executes complex multi-agent workflows
- **Event Processing:** Handles real-time event streams
- **State Management:** Maintains consistent system state

#### ML/AI Engine

The ML/AI Engine provides intelligent capabilities:

- **Predictive Models:** Time-series forecasting, regression
- **Anomaly Detection:** Unsupervised detection of abnormal patterns
- **Causal Inference:** Root cause analysis
- **Explainability:** LIME, SHAP, attention visualization

#### Infrastructure Layer

The Infrastructure Layer provides foundational services:

- **Data Ingestion:** Multi-protocol data collection (OPC-UA, Modbus, MQTT)
- **API Gateway:** REST, GraphQL, WebSocket APIs
- **Event Streaming:** Real-time event distribution
- **Security:** Authentication, authorization, encryption

### 1.4 Deployment Models

GreenLang supports multiple deployment models:

| Model | Description | Best For |
|-------|-------------|----------|
| On-Premises | Deployed within customer data center | High security requirements, air-gapped networks |
| Cloud (SaaS) | Hosted on GreenLang cloud infrastructure | Rapid deployment, minimal IT overhead |
| Hybrid | Combination of on-premises and cloud | Gradual cloud migration, data sovereignty |
| Edge | Deployed on edge devices close to equipment | Low latency requirements, limited connectivity |

---

## Part 2: Core Concepts

### 2.1 Agents

Agents are the fundamental building blocks of GreenLang. Each agent is an autonomous unit that:

- **Monitors** specific aspects of the process
- **Analyzes** data using ML models
- **Decides** on appropriate actions
- **Acts** or recommends actions

#### Agent Lifecycle

```
    +----------+
    |  CREATED |
    +----+-----+
         |
         v
    +----+-----+
    |  INITIALIZED |
    +----+-----+
         |
         v
    +----+-----+      +-----------+
    |  RUNNING  +----->  PAUSED   |
    +----+-----+      +-----+-----+
         |                  |
         v                  |
    +----+-----+            |
    |  STOPPED |<-----------+
    +----------+
```

#### Agent Configuration

```python
from greenlang.agents import ProcessHeatAgent

# Create and configure an agent
agent = ProcessHeatAgent(
    name="furnace_zone_1",
    config={
        "monitoring_interval": 1000,  # milliseconds
        "temperature_range": {"min": 800, "max": 1200},
        "alarm_thresholds": {
            "warning": 0.8,
            "critical": 0.95
        },
        "ml_model": "gradient_boost_v2"
    }
)

# Start the agent
agent.start()
```

### 2.2 Modules

Modules are reusable components that provide specific functionality:

| Module Type | Purpose | Examples |
|-------------|---------|----------|
| Input Modules | Data acquisition | OPC-UA connector, Modbus reader |
| Processing Modules | Data transformation | Filtering, aggregation, normalization |
| ML Modules | Machine learning | Prediction, classification, anomaly detection |
| Output Modules | Results delivery | Alerts, reports, API responses |
| Safety Modules | Compliance monitoring | ISA 18.2, NFPA 86 |

#### Module Composition

```python
from greenlang.modules import (
    OPCUAInput,
    DataNormalizer,
    AnomalyDetector,
    AlertOutput
)

# Build a processing pipeline
pipeline = (
    OPCUAInput(server="opc.tcp://plc.local:4840")
    >> DataNormalizer(method="z-score")
    >> AnomalyDetector(model="isolation_forest")
    >> AlertOutput(channels=["email", "sms", "dashboard"])
)
```

### 2.3 Orchestration

Orchestration coordinates multiple agents and modules:

#### Workflow Definition

```yaml
# workflow.yaml
name: temperature_monitoring
version: 1.0
trigger:
  type: schedule
  interval: 60s

steps:
  - name: collect_data
    agent: data_collector
    action: collect
    params:
      sources: ["zone_1", "zone_2", "zone_3"]

  - name: analyze
    agent: ml_analyzer
    action: predict
    depends_on: collect_data

  - name: alert
    agent: alert_manager
    action: evaluate
    depends_on: analyze
    condition: "analyze.anomaly_score > 0.8"
```

#### Event-Driven Orchestration

```python
from greenlang.orchestration import Orchestrator, Event

orchestrator = Orchestrator()

@orchestrator.on(Event.TEMPERATURE_EXCEEDED)
async def handle_temperature_exceeded(event):
    """React to temperature threshold exceeded."""
    await orchestrator.dispatch(
        agent="safety_controller",
        action="reduce_fuel_input",
        params={"zone": event.zone, "reduction": 0.2}
    )
```

### 2.4 Data Flow

Understanding data flow is crucial for effective GreenLang operation:

```
+-------------+     +--------------+     +---------------+
|   Sensors   |     |   Edge       |     |   GreenLang   |
|   & PLCs    +---->|   Gateway    +---->|   Platform    |
+-------------+     +--------------+     +-------+-------+
                                                 |
    +--------------------------------------------+
    |
    v
+---+---+     +----------+     +------------+     +-----------+
| Ingest |---->| Process  |---->|  Analyze   |---->|  Action   |
+-------+     +----------+     +------------+     +-----------+
    |              |                 |                  |
    v              v                 v                  v
+-------+     +----------+     +------------+     +-----------+
| Store |     | Transform|     |  Predict   |     |  Alert    |
+-------+     +----------+     +------------+     +-----------+
```

---

## Part 3: Process Heat Domain Basics

### 3.1 Introduction to Process Heat

Process heat refers to thermal energy used in industrial processes. It accounts for approximately 70% of industrial energy consumption and represents a significant opportunity for efficiency improvements.

#### Process Heat Applications

| Industry | Application | Temperature Range |
|----------|-------------|-------------------|
| Steel | Furnaces, rolling mills | 800-1600C |
| Glass | Melting, annealing | 600-1500C |
| Cement | Kilns, preheaters | 800-1450C |
| Chemicals | Reactors, distillation | 100-800C |
| Food | Drying, sterilization | 50-200C |

### 3.2 Key Process Heat Metrics

#### Temperature Metrics

| Metric | Description | Typical Units |
|--------|-------------|---------------|
| Setpoint | Target temperature | C, F, K |
| Process Value (PV) | Actual measured temperature | C, F, K |
| Deviation | Difference from setpoint | C, F, K |
| Rate of Change | Temperature change rate | C/min, F/min |

#### Energy Metrics

| Metric | Description | Typical Units |
|--------|-------------|---------------|
| Heat Input | Energy supplied to process | kW, MW, BTU/hr |
| Heat Loss | Energy lost to environment | kW, MW, BTU/hr |
| Efficiency | Useful heat / Total heat input | % |
| Specific Energy | Energy per unit product | kWh/ton, BTU/lb |

### 3.3 Heat Transfer Fundamentals

Understanding heat transfer modes is essential for process optimization:

#### Conduction

Heat transfer through solid materials:

```
Q = k * A * (T1 - T2) / d

Where:
  Q = Heat transfer rate (W)
  k = Thermal conductivity (W/m*K)
  A = Cross-sectional area (m^2)
  T1, T2 = Temperatures (K)
  d = Thickness (m)
```

#### Convection

Heat transfer through fluid motion:

```
Q = h * A * (Ts - Tf)

Where:
  Q = Heat transfer rate (W)
  h = Convection coefficient (W/m^2*K)
  A = Surface area (m^2)
  Ts = Surface temperature (K)
  Tf = Fluid temperature (K)
```

#### Radiation

Heat transfer through electromagnetic waves:

```
Q = epsilon * sigma * A * (T1^4 - T2^4)

Where:
  Q = Heat transfer rate (W)
  epsilon = Emissivity (0-1)
  sigma = Stefan-Boltzmann constant (5.67e-8 W/m^2*K^4)
  A = Surface area (m^2)
  T1, T2 = Absolute temperatures (K)
```

### 3.4 Process Control Concepts

#### PID Control

Most process heat systems use PID (Proportional-Integral-Derivative) control:

```
Output = Kp*e(t) + Ki*integral(e(t)) + Kd*de(t)/dt

Where:
  Kp = Proportional gain
  Ki = Integral gain
  Kd = Derivative gain
  e(t) = Error (Setpoint - Process Value)
```

#### Control Loop Tuning

| Parameter | Too Low | Too High |
|-----------|---------|----------|
| Kp | Sluggish response | Oscillation |
| Ki | Steady-state error | Overshoot, instability |
| Kd | Slow response to disturbances | Noise amplification |

### 3.5 Safety Considerations

#### Critical Safety Parameters

| Parameter | Concern | Mitigation |
|-----------|---------|------------|
| Over-temperature | Equipment damage, fire | High-limit controllers, interlocks |
| Under-temperature | Product quality, freeze | Low-limit alarms, trace heating |
| Pressure | Explosion, leaks | Relief valves, pressure monitoring |
| Atmosphere | Explosion, corrosion | Gas analysis, purge systems |

#### Safety Standards

| Standard | Scope | Key Requirements |
|----------|-------|------------------|
| NFPA 86 | Furnace safety | Combustion safeguards, ventilation |
| ISA 18.2 | Alarm management | Rationalization, prioritization |
| OSHA PSM | Process safety | Hazard analysis, procedures |
| IEC 61511 | Functional safety | SIL requirements, validation |

---

## Part 4: ML/AI Foundations for Operators

### 4.1 Introduction to Machine Learning

Machine learning enables computers to learn from data without explicit programming. In process heat applications, ML helps:

- **Predict** future temperatures and equipment failures
- **Detect** anomalies and abnormal conditions
- **Optimize** process parameters
- **Classify** product quality

#### Types of Machine Learning

| Type | Description | Examples in GreenLang |
|------|-------------|----------------------|
| Supervised | Learn from labeled examples | Temperature prediction, quality classification |
| Unsupervised | Find patterns in unlabeled data | Anomaly detection, clustering |
| Reinforcement | Learn through trial and error | Process optimization |

### 4.2 Understanding ML Model Outputs

#### Predictions

Predictions are numerical forecasts:

```
Temperature Prediction:
  Current: 850C
  Predicted (1 hour): 862C
  Confidence Interval: [855C, 869C] (95%)
```

#### Classifications

Classifications assign categories:

```
Equipment Health Classification:
  Class: DEGRADED
  Confidence: 87%

  Class Probabilities:
    HEALTHY: 10%
    DEGRADED: 87%
    CRITICAL: 3%
```

#### Anomaly Scores

Anomaly scores indicate abnormality:

```
Anomaly Detection:
  Score: 0.73 (range: 0-1)
  Status: WARNING

  Interpretation:
    0.0-0.3: Normal
    0.3-0.7: Attention
    0.7-1.0: Anomalous
```

### 4.3 Explainability

GreenLang provides explanations for ML decisions:

#### Feature Importance

Shows which inputs influenced the prediction:

```
Temperature Prediction Explanation:

  Top Contributing Features:
  1. Fuel Flow Rate: +15C (45%)
  2. Inlet Air Temperature: +8C (24%)
  3. Product Throughput: -5C (15%)
  4. Ambient Temperature: +3C (9%)
  5. Furnace Age: +2C (7%)
```

#### Natural Language Explanations

GreenLang generates human-readable explanations:

```
"The temperature in Zone 1 is predicted to increase by 12C over
the next hour. This is primarily due to the 5% increase in fuel
flow rate at 14:30 and the recent reduction in cooling water flow.
Consider reducing fuel input or increasing cooling to maintain
target temperature."
```

### 4.4 Working with ML Predictions

#### Confidence Levels

Always consider prediction confidence:

| Confidence | Action |
|------------|--------|
| > 90% | High confidence, act on prediction |
| 70-90% | Moderate confidence, verify with other data |
| 50-70% | Low confidence, use as advisory only |
| < 50% | Very low confidence, investigate model performance |

#### Prediction Horizons

Accuracy decreases with prediction horizon:

| Horizon | Typical Accuracy | Use Case |
|---------|------------------|----------|
| 1 minute | 98-99% | Real-time control |
| 1 hour | 90-95% | Operational planning |
| 1 day | 80-90% | Production scheduling |
| 1 week | 60-80% | Maintenance planning |

### 4.5 Common ML Terms

| Term | Definition |
|------|------------|
| Model | Mathematical representation learned from data |
| Training | Process of learning from historical data |
| Inference | Using trained model to make predictions |
| Feature | Input variable used by the model |
| Label | Known outcome used for training |
| Overfitting | Model too specific to training data |
| Underfitting | Model too simple to capture patterns |

---

## Summary

In this module, you learned:

1. **Architecture:** GreenLang's layered architecture including agents, orchestration, ML engine, and infrastructure
2. **Core Concepts:** Agents, modules, and orchestration patterns
3. **Process Heat:** Fundamentals of process heat, heat transfer, and process control
4. **ML/AI:** Basic machine learning concepts and how to interpret ML outputs

---

## Knowledge Check

### Questions

1. What are the four main layers of the GreenLang architecture?
2. Explain the difference between agents and modules.
3. What are the three modes of heat transfer?
4. How should you interpret an anomaly score of 0.75?
5. What does a confidence interval tell you about a prediction?

### Practical Exercises

1. **Architecture Diagram:** Draw the GreenLang architecture from memory
2. **Agent Configuration:** Write a basic agent configuration for a furnace monitoring application
3. **Heat Transfer:** Calculate heat loss through conduction given thermal properties
4. **ML Interpretation:** Given a feature importance chart, explain the prediction

---

## Next Steps

After completing this module, proceed to:

- **Operators:** [Operator Training](./operator_training.md)
- **Administrators:** [Administrator Training](./administrator_training.md)
- **Quick Start:** [5-Minute Start](../quickstart/5_minute_start.md)

---

*Module Version: 1.0.0*
*Last Updated: December 2025*
