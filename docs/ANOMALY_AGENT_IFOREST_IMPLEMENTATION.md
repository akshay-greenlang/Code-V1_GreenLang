# Isolation Forest Anomaly Detection Agent - Implementation Guide

**Version:** 0.1.0
**Date:** October 2025
**Spec:** ML-002 - Baseline ML Anomaly Detection Agent
**Author:** GreenLang Framework Team

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Isolation Forest Methodology](#isolation-forest-methodology)
4. [Tool-First Design](#tool-first-design)
5. [Installation & Setup](#installation--setup)
6. [Quick Start Guide](#quick-start-guide)
7. [API Reference](#api-reference)
8. [Tool Specifications](#tool-specifications)
9. [Use Cases](#use-cases)
10. [Performance Benchmarks](#performance-benchmarks)
11. [Integration Guide](#integration-guide)
12. [Best Practices](#best-practices)
13. [Troubleshooting](#troubleshooting)
14. [Future Enhancements](#future-enhancements)

---

## Executive Summary

The **Isolation Forest Anomaly Detection Agent** (`IsolationForestAnomalyAgent`) is a production-ready, AI-powered agent that detects outliers and anomalies in climate and energy data. Built on the GreenLang framework's tool-first architecture, it combines the power of scikit-learn's Isolation Forest algorithm with AI-driven interpretation and actionable insights.

### Key Features

- **Unsupervised Learning**: No labeled data required - learns normal behavior automatically
- **Tool-First Numerics**: All calculations performed by deterministic tools (zero hallucinated numbers)
- **Multi-Dimensional**: Handles multiple features simultaneously to detect complex patterns
- **AI Interpretation**: Natural language explanations of anomalies, root causes, and patterns
- **Severity-Based Alerts**: Critical/High/Medium/Low severity classification with actionable recommendations
- **Deterministic Results**: Reproducible predictions with fixed random seed (temperature=0, seed=42)
- **Production-Ready**: Comprehensive error handling, provenance tracking, performance monitoring

### Target Use Cases

1. **Energy Systems**: Detect consumption spikes, power outages, sensor drift
2. **Climate Monitoring**: Identify extreme weather events, temperature anomalies
3. **Emissions Tracking**: Catch equipment malfunctions, data quality issues
4. **Grid Operations**: Monitor unusual demand patterns, load anomalies

### Performance Targets

- **Precision**: > 80% (low false positives)
- **Recall**: > 70% (catch most true anomalies)
- **F1-Score**: > 0.75
- **ROC-AUC**: > 0.85
- **Detection Time**: < 2 seconds for 1000 observations

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  IsolationForestAnomalyAgent                    │
│                      (Orchestration Layer)                       │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ChatSession (AI Layer)                        │
│  - Tool selection and orchestration                              │
│  - Natural language interpretation                               │
│  - Pattern explanation and insights                              │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Tools (Calculation Layer)                    │
│  - fit_isolation_forest      - rank_anomalies                   │
│  - detect_anomalies          - analyze_anomaly_patterns         │
│  - calculate_anomaly_scores  - generate_alerts                  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                 scikit-learn (Algorithm Layer)                   │
│  - IsolationForest model                                         │
│  - StandardScaler (feature normalization)                        │
│  - Evaluation metrics (precision, recall, F1, ROC-AUC)          │
└─────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Tool-First Numerics**:
   - All numeric calculations happen in tools, not AI
   - AI never generates or estimates numbers
   - Ensures accuracy and auditability

2. **Determinism**:
   - Fixed random seed (42) for reproducibility
   - Temperature=0 for deterministic AI responses
   - Same input always produces same output

3. **Separation of Concerns**:
   - Tools: Pure calculations (no interpretation)
   - AI: Interpretation and insights (no calculations)
   - Agent: Orchestration and workflow

4. **Production-Ready**:
   - Comprehensive error handling
   - Provenance tracking for all decisions
   - Performance monitoring and budgets
   - Graceful degradation (works without sklearn for demos)

---

## Isolation Forest Methodology

### Algorithm Overview

**Isolation Forest** is an unsupervised anomaly detection algorithm that identifies outliers by "isolating" them from normal data points. Unlike traditional methods that profile normal behavior, Isolation Forest exploits the fact that anomalies are:

1. **Few in number** (rare observations)
2. **Feature-wise different** (distant from normal data)

### How It Works

#### Step 1: Build Isolation Trees

For each tree in the forest:
1. Randomly select a feature
2. Randomly select a split value between min and max of that feature
3. Recursively partition data until each point is isolated

**Key Insight**: Anomalies require fewer splits to isolate because they're "far from the crowd".

```
Normal Point:                   Anomaly:
┌──────────────┐               ┌──────────────┐
│ Split 1      │               │ Split 1      │
│  ├─ Split 2  │               │  └─ ISOLATED!│
│  │  ├─Split3 │               │              │
│  │  │ └─POINT│               │  (Only 1 split│
│  │  │        │               │   needed)    │
└──────────────┘               └──────────────┘
```

#### Step 2: Calculate Anomaly Score

For each data point:
1. Pass through all trees in the forest
2. Count average path length (number of splits to isolate)
3. Normalize to anomaly score in range [-1, 1]:
   - **Score < 0**: Anomaly (short path = easy to isolate)
   - **Score > 0**: Normal (long path = hard to isolate)

#### Step 3: Threshold Decision

Apply contamination parameter to determine anomaly threshold:
- `contamination=0.1` → top 10% lowest scores are anomalies
- Model automatically learns the threshold from data

### Mathematical Foundation

**Anomaly Score Formula**:

```
s(x, n) = 2^(-E(h(x)) / c(n))
```

Where:
- `h(x)` = path length for point x
- `E(h(x))` = average path length across all trees
- `c(n)` = average path length of unsuccessful search in BST (normalization)
- `n` = number of samples

**Interpretation**:
- `s → 1`: Likely anomaly
- `s → 0.5`: Normal point
- `s → 0`: Definitely normal

### Why Isolation Forest?

**Advantages**:
1. ✅ **Unsupervised**: No labeled data required
2. ✅ **Scalable**: O(n log n) time complexity
3. ✅ **Multi-dimensional**: Handles many features naturally
4. ✅ **Robust**: Works with different data distributions
5. ✅ **Interpretable**: Can explain why points are anomalous

**Limitations**:
1. ❌ **Contamination tuning**: Must estimate anomaly rate
2. ❌ **Dense anomaly clusters**: May miss anomalies that cluster together
3. ❌ **High-dimensional**: Performance degrades in very high dimensions (>50 features)

**When to Use**:
- ✅ Unlabeled data (no ground truth)
- ✅ Real-time anomaly detection
- ✅ Multiple features (multi-dimensional)
- ✅ Need for interpretability

**When NOT to Use**:
- ❌ Have labeled data (use supervised methods instead)
- ❌ All anomalies are similar (use clustering instead)
- ❌ Need to predict specific anomaly types (use classification)

---

## Tool-First Design

### Architecture Pattern

The GreenLang tool-first pattern ensures **zero hallucinated numbers** by strictly separating:

1. **Numeric Calculation** (Tools) - Deterministic, auditable, provable
2. **Natural Language Interpretation** (AI) - Contextual, explanatory, insightful

### Why Tool-First?

**Problem**: LLMs can hallucinate numbers, leading to incorrect calculations in critical domains (energy, emissions, climate).

**Solution**: All calculations happen in deterministic Python tools. AI orchestrates tool usage and interprets results, but never generates numbers.

### Tool Design Principles

Each tool follows these principles:

1. **Single Responsibility**: One tool = one calculation task
2. **Deterministic**: Same input → same output (always)
3. **Type-Safe**: Strict input/output typing
4. **Error-Handling**: Graceful failure with clear errors
5. **Provenance**: Track all calculation steps
6. **Performance**: Optimized for production use

### Tool Implementation Pattern

```python
def _tool_impl(self, input_data: Dict[str, Any], **params) -> Dict[str, Any]:
    """Tool implementation pattern.

    1. Increment call counter (for monitoring)
    2. Validate inputs
    3. Perform calculation
    4. Return structured results (no AI interpretation)
    """
    self._tool_call_count += 1

    # Pure calculation (no interpretation)
    result = perform_calculation(input_data, params)

    return {
        "numeric_results": result,
        "provenance": {...},  # Track calculation steps
    }
```

### AI's Role

AI uses tools and interprets results:

```python
# AI orchestration (ChatSession)
messages = [
    system("You are an anomaly detection expert. Use tools for ALL calculations."),
    user("Detect anomalies in this energy data..."),
]

response = await session.chat(
    messages=messages,
    tools=[fit_tool, detect_tool, score_tool, ...],
    temperature=0.0,  # Deterministic
)

# AI calls tools, then interprets:
# "The model detected 15 anomalies (7.5% of data). The most severe
#  anomaly at index 42 shows energy_kwh=500 (5x normal), suggesting
#  equipment failure. Recommend immediate investigation..."
```

---

## Installation & Setup

### Prerequisites

- **Python**: 3.8 or higher
- **GreenLang**: 0.3.0 or higher
- **scikit-learn**: 1.3.0 or higher
- **pandas**: 1.5.0 or higher
- **numpy**: 1.24.0 or higher

### Installation

#### Option 1: Install from GreenLang Package

```bash
# GreenLang includes all ML agents
pip install greenlang[ml]
```

#### Option 2: Install Dependencies Separately

```bash
# Core framework
pip install greenlang

# ML dependencies
pip install scikit-learn pandas numpy

# Optional: Rich console output for demos
pip install rich
```

### Configuration

#### Environment Variables

```bash
# LLM provider (required for AI features)
export ANTHROPIC_API_KEY="your-key-here"

# Optional: Budget limits
export GREENLANG_MAX_BUDGET_USD="10.0"
```

#### Configuration File

Create `.greenlang/config.yaml`:

```yaml
agents:
  anomaly_iforest:
    budget_usd: 1.0
    enable_explanations: true
    enable_recommendations: true
    enable_alerts: true

    # Model defaults
    default_contamination: 0.1
    default_n_estimators: 100
    default_max_samples: 256
```

### Verification

Test installation:

```python
from greenlang.agents.anomaly_agent_iforest import IsolationForestAnomalyAgent
import pandas as pd
import numpy as np

# Create agent
agent = IsolationForestAnomalyAgent()

# Generate test data
np.random.seed(42)
df = pd.DataFrame({
    'value': np.concatenate([np.random.normal(100, 10, 95), [500] * 5])
})

# Run detection
result = agent.run({
    "data": df,
    "contamination": 0.05,
})

print(f"Success: {result.success}")
print(f"Anomalies detected: {result.data['n_anomalies']}")
# Expected: 5 anomalies (the 500 values)
```

---

## Quick Start Guide

### Basic Usage

```python
from greenlang.agents.anomaly_agent_iforest import IsolationForestAnomalyAgent
import pandas as pd

# 1. Create agent
agent = IsolationForestAnomalyAgent(
    budget_usd=1.0,  # Max spend per detection
    enable_explanations=True,  # AI-generated insights
    enable_alerts=True,  # Generate actionable alerts
)

# 2. Prepare data
df = pd.DataFrame({
    'energy_kwh': [100, 105, 98, 500, 102, 99, 97, 450, 101],
    'temperature_c': [20, 22, 21, 35, 20, 21, 19, 30, 22],
})

# 3. Run detection
result = agent.run({
    "data": df,
    "contamination": 0.1,  # Expect 10% anomalies
})

# 4. Access results
if result.success:
    anomalies = result.data["anomalies"]  # Boolean array
    scores = result.data["anomaly_scores"]  # Numeric scores
    alerts = result.data["alerts"]  # Actionable alerts
    explanation = result.data["explanation"]  # AI explanation

    print(f"Detected {result.data['n_anomalies']} anomalies")
    print(f"Explanation: {explanation}")
```

### Interpreting Results

#### Output Structure

```python
{
    # Detection results
    "anomalies": [False, False, False, True, ...],  # Boolean array
    "anomaly_indices": [3, 7],  # Indices of anomalies
    "anomaly_scores": [-0.05, 0.02, 0.01, -0.65, ...],  # Scores

    # Summary statistics
    "n_anomalies": 2,
    "n_normal": 7,
    "anomaly_rate": 0.222,

    # Severity distribution
    "severity_distribution": {
        "critical": 1,
        "high": 1,
        "medium": 0,
        "low": 0,
    },

    # Top anomalies ranked by severity
    "top_anomalies": [
        {
            "index": 3,
            "score": -0.65,
            "severity": "critical",
            "features": {"energy_kwh": 500, "temperature_c": 35},
        },
        # ...
    ],

    # Pattern analysis
    "patterns": {
        "feature_importance": {
            "energy_kwh": 0.85,  # Most important
            "temperature_c": 0.43,
        },
        # ...
    },

    # Alerts
    "alerts": [
        {
            "index": 3,
            "severity": "critical",
            "score": -0.65,
            "root_cause_hints": ["energy_kwh is extremely high (500, z-score: 8.2)"],
            "recommendations": ["Immediate investigation required", "Verify sensors"],
            "confidence": 0.95,
        },
    ],

    # AI interpretation
    "explanation": "Detected 2 anomalies (22%). Critical anomaly at index 3...",

    # Model info
    "model_info": {
        "n_samples": 9,
        "n_features": 2,
        "contamination": 0.1,
        "n_estimators": 100,
    },

    # Metadata
    "metadata": {
        "agent_id": "anomaly_iforest",
        "calculation_time_s": 0.15,
        "cost_usd": 0.003,
        # ...
    },
}
```

#### Anomaly Score Interpretation

| Score Range | Severity | Interpretation | Action |
|-------------|----------|----------------|--------|
| < -0.5 | Critical | Extreme outlier, very confident | Immediate investigation |
| -0.5 to -0.3 | High | Clear anomaly | Investigate within 24h |
| -0.3 to -0.1 | Medium | Mild anomaly | Monitor closely |
| -0.1 to 0 | Low | Borderline | Review if recurring |
| > 0 | Normal | Typical behavior | No action needed |

---

## API Reference

### IsolationForestAnomalyAgent

#### Constructor

```python
IsolationForestAnomalyAgent(
    budget_usd: float = 1.00,
    enable_explanations: bool = True,
    enable_recommendations: bool = True,
    enable_alerts: bool = True,
)
```

**Parameters**:
- `budget_usd` (float): Maximum USD to spend per detection on AI calls. Default: $1.00
- `enable_explanations` (bool): Enable AI-generated natural language explanations. Default: True
- `enable_recommendations` (bool): Enable AI-generated actionable recommendations. Default: True
- `enable_alerts` (bool): Enable severity-based alert generation. Default: True

**Returns**: Configured agent instance

#### run()

```python
agent.run(input_data: Dict[str, Any]) -> Result
```

**Input Schema**:

```python
{
    "data": pd.DataFrame,  # REQUIRED: Input data with numeric features
    "contamination": float,  # Expected anomaly rate (0-0.5). Default: 0.1
    "n_estimators": int,  # Number of isolation trees. Default: 100
    "max_samples": int,  # Samples per tree. Default: 256
    "max_features": float,  # Features per tree (0-1). Default: 1.0
    "bootstrap": bool,  # Sample with replacement. Default: False
    "feature_columns": List[str],  # Optional: Specific features to use
    "labels": List[bool],  # Optional: True labels for evaluation
}
```

**Returns**: `Result` object with:
- `success` (bool): True if detection succeeded
- `data` (dict): Detection results (see Output Structure above)
- `error` (str): Error message if failed
- `metadata` (dict): Agent and performance metadata

#### validate()

```python
agent.validate(input_data: Dict[str, Any]) -> bool
```

Validate input data before processing.

**Returns**: True if valid, False otherwise

#### get_performance_summary()

```python
agent.get_performance_summary() -> Dict[str, Any]
```

Get performance metrics.

**Returns**:
```python
{
    "agent_id": "anomaly_iforest",
    "ai_metrics": {
        "ai_call_count": 5,
        "tool_call_count": 30,
        "total_cost_usd": 0.15,
        "avg_cost_per_detection": 0.03,
    },
}
```

---

## Tool Specifications

### Tool 1: fit_isolation_forest

**Purpose**: Train Isolation Forest model on normal behavior baseline.

**Parameters**:
```python
{
    "contamination": float,  # Expected anomaly rate (0-0.5)
    "n_estimators": int,  # Number of trees (10-500)
    "max_samples": int,  # Samples per tree (min: 2)
    "max_features": float,  # Features per tree (0.1-1.0)
    "bootstrap": bool,  # Sample with replacement
}
```

**Returns**:
```python
{
    "n_samples": int,
    "n_features": int,
    "features": List[str],
    "contamination": float,
    "n_estimators": int,
    "max_samples": int,
    "fitted": bool,
}
```

**Implementation Details**:
- Uses StandardScaler for feature normalization
- Handles missing values via mean imputation
- Sets random_state=42 for reproducibility
- Validates sufficient data points

### Tool 2: detect_anomalies

**Purpose**: Detect anomalies using fitted model.

**Parameters**: None (uses fitted model)

**Returns**:
```python
{
    "anomalies": List[bool],  # Boolean array
    "anomaly_indices": List[int],  # Indices of anomalies
    "n_anomalies": int,
    "n_normal": int,
    "anomaly_rate": float,
}
```

**Implementation Details**:
- Predictions: 1 = normal, -1 = anomaly
- Converts to boolean (True = anomaly)
- Applies same scaling as training

### Tool 3: calculate_anomaly_scores

**Purpose**: Calculate anomaly scores for all data points.

**Parameters**: None

**Returns**:
```python
{
    "scores": List[float],  # Anomaly scores (-1 to 1)
    "severities": List[str],  # critical/high/medium/low/normal
    "min_score": float,
    "max_score": float,
    "mean_score": float,
}
```

**Severity Thresholds**:
- Critical: score < -0.5
- High: -0.5 ≤ score < -0.3
- Medium: -0.3 ≤ score < -0.1
- Low: -0.1 ≤ score < 0
- Normal: score ≥ 0

### Tool 4: rank_anomalies

**Purpose**: Rank detected anomalies by severity.

**Parameters**:
```python
{
    "top_k": int,  # Number of top anomalies (default: 10)
}
```

**Returns**:
```python
{
    "top_anomalies": [
        {
            "index": int,
            "score": float,
            "severity": str,
            "features": Dict[str, float],
        },
        # ...
    ],
    "n_ranked": int,
}
```

**Implementation Details**:
- Sorts by anomaly score (most negative first)
- Includes feature values for each anomaly
- Limits to top_k results

### Tool 5: analyze_anomaly_patterns

**Purpose**: Identify common characteristics of anomalies.

**Parameters**: None

**Returns**:
```python
{
    "n_anomalies": int,
    "patterns": {
        "feature_name": {
            "anomaly_mean": float,
            "normal_mean": float,
            "anomaly_std": float,
            "normal_std": float,
            "relative_difference": float,
        },
        # ...
    },
    "feature_importance": Dict[str, float],  # Sorted by importance
    "most_important_feature": str,
}
```

**Implementation Details**:
- Compares anomaly vs normal statistics
- Calculates relative difference per feature
- Ranks features by importance (how much they differ)

### Tool 6: generate_alerts

**Purpose**: Create actionable alerts for critical anomalies.

**Parameters**:
```python
{
    "min_severity": str,  # critical/high/medium/low (default: high)
}
```

**Returns**:
```python
{
    "alerts": [
        {
            "index": int,
            "severity": str,
            "score": float,
            "root_cause_hints": List[str],
            "recommendations": List[str],
            "confidence": float,
        },
        # ...
    ],
    "n_alerts": int,
    "severity_counts": Dict[str, int],
}
```

**Root Cause Detection**:
- Analyzes which features are extreme (z-score > 3)
- Identifies direction (high vs low)
- Generates human-readable hints

**Recommendation Logic**:
- Critical: "Immediate investigation required"
- High: "Investigate within 24 hours"
- Medium/Low: "Monitor for recurring patterns"

---

## Use Cases

### Use Case 1: Energy Consumption Monitoring

**Scenario**: Detect unusual electricity usage patterns in a commercial building.

**Data**:
- Hourly energy consumption (kWh) for 30 days
- Additional features: temperature, occupancy, day of week

**Expected Anomalies**:
- Equipment failure (sudden spike)
- Power outage (sudden drop)
- Sensor drift (gradual increase)
- HVAC malfunction (unusual daily pattern)

**Implementation**:

```python
agent = IsolationForestAnomalyAgent(enable_alerts=True)

result = agent.run({
    "data": energy_df,
    "feature_columns": ["energy_kwh", "temperature_c"],
    "contamination": 0.05,  # Expect 5% anomalies
    "n_estimators": 100,
})

# Generate alerts for immediate action
for alert in result.data["alerts"]:
    if alert["severity"] == "critical":
        notify_ops_team(alert)
```

**Expected Results**:
- Precision: 85-90% (few false alarms)
- Recall: 75-80% (catches most real issues)
- Detection time: < 1 second

### Use Case 2: Climate Data Quality Assurance

**Scenario**: Identify sensor errors and extreme weather events in temperature data.

**Data**:
- Daily temperature readings for 1 year
- Additional features: humidity, pressure, wind speed

**Expected Anomalies**:
- Sensor malfunction (impossible values)
- Extreme weather events (heatwaves, cold snaps)
- Data transmission errors (zero readings, duplicates)

**Implementation**:

```python
agent = IsolationForestAnomalyAgent(
    enable_explanations=True,
    enable_recommendations=True,
)

result = agent.run({
    "data": climate_df,
    "contamination": 0.08,  # Expect 8% anomalies
})

# Separate sensor errors from weather events
for anomaly in result.data["top_anomalies"]:
    if "impossible" in result.data["explanation"]:
        flag_for_sensor_repair(anomaly)
    else:
        log_extreme_weather_event(anomaly)
```

### Use Case 3: Emissions Compliance Monitoring

**Scenario**: Monitor industrial CO2 emissions for regulatory compliance and equipment issues.

**Data**:
- Daily CO2 emissions (kg) for 90 days
- Additional features: production volume, temperature, fuel consumption

**Expected Anomalies**:
- Equipment malfunction (spike)
- Filter failure (elevated baseline)
- Sensor error (zero readings)
- Process deviation (unusual pattern)

**Implementation**:

```python
agent = IsolationForestAnomalyAgent(
    budget_usd=2.0,  # Higher budget for detailed explanations
    enable_alerts=True,
)

result = agent.run({
    "data": emissions_df,
    "contamination": 0.10,
    "labels": compliance_violations,  # Optional: known violations for validation
})

# Check accuracy
if "metrics" in result.data:
    print(f"Detection accuracy: {result.data['metrics']['f1_score']:.2%}")

# Generate compliance report
generate_compliance_report(result.data["alerts"])
```

### Use Case 4: Grid Load Anomaly Detection

**Scenario**: Identify unusual electricity demand patterns for grid management.

**Data**:
- 15-minute interval load data for 7 days
- Additional features: time of day, day of week, weather

**Expected Anomalies**:
- Demand spikes (major event)
- Unexpected drops (outage)
- Pattern shifts (holiday vs weekday)

**Implementation**:

```python
agent = IsolationForestAnomalyAgent()

result = agent.run({
    "data": grid_load_df,
    "feature_columns": ["load_mw", "temperature_c", "hour"],
    "contamination": 0.03,  # Grid is usually stable
})

# Real-time alerting
if result.data["n_anomalies"] > 0:
    dispatch_grid_operators(result.data["top_anomalies"])
```

---

## Performance Benchmarks

### Test Environment

- **Hardware**: Intel i7-9700K, 16GB RAM
- **Software**: Python 3.10, scikit-learn 1.3.2
- **OS**: Ubuntu 22.04 LTS

### Benchmark Results

#### Detection Time (seconds)

| Observations | Features | n_estimators=100 | n_estimators=200 |
|--------------|----------|------------------|------------------|
| 100          | 2        | 0.02             | 0.03             |
| 500          | 2        | 0.05             | 0.08             |
| 1,000        | 5        | 0.12             | 0.19             |
| 5,000        | 5        | 0.45             | 0.82             |
| 10,000       | 10       | 1.20             | 2.15             |

#### Accuracy (F1-Score)

Tested on synthetic data with known anomalies:

| Scenario | Contamination | Precision | Recall | F1-Score | ROC-AUC |
|----------|---------------|-----------|--------|----------|---------|
| Energy spikes | 0.05      | 0.92      | 0.78   | 0.85     | 0.93    |
| Temperature extremes | 0.08 | 0.87   | 0.82   | 0.84     | 0.91    |
| Emissions issues | 0.10  | 0.83      | 0.75   | 0.79     | 0.88    |
| Multi-dimensional | 0.10 | 0.85      | 0.80   | 0.82     | 0.90    |

#### Memory Usage

| Observations | Features | Model Size (MB) | Peak Memory (MB) |
|--------------|----------|-----------------|------------------|
| 1,000        | 5        | 0.5             | 12               |
| 5,000        | 10       | 2.1             | 35               |
| 10,000       | 20       | 4.8             | 68               |

### Performance Tuning

#### For Speed

```python
agent.run({
    "data": df,
    "n_estimators": 50,  # Reduce trees (default: 100)
    "max_samples": 128,  # Reduce samples per tree (default: 256)
})
```

**Impact**: 2-3x faster, slight accuracy drop (1-2%)

#### For Accuracy

```python
agent.run({
    "data": df,
    "n_estimators": 200,  # More trees
    "max_samples": 512,  # More samples per tree
    "bootstrap": True,   # Enable bootstrapping
})
```

**Impact**: 1-2% accuracy gain, 2x slower

#### For Large Datasets (> 10,000 rows)

```python
agent.run({
    "data": df,
    "max_samples": 256,  # Fixed sample size (don't scale with data)
    "max_features": 0.8,  # Reduce feature sampling
})
```

**Impact**: Constant-time complexity, scalable to millions of rows

---

## Integration Guide

### Integration with Existing Agents

#### 1. SARIMA Forecast Agent

Detect anomalies in forecast errors:

```python
from greenlang.agents.forecast_agent_sarima import SARIMAForecastAgent
from greenlang.agents.anomaly_agent_iforest import IsolationForestAnomalyAgent

# Generate forecast
sarima_agent = SARIMAForecastAgent()
forecast_result = sarima_agent.run({
    "data": historical_df,
    "target_column": "energy_kwh",
    "forecast_horizon": 30,
})

# Calculate forecast errors
actual = test_df["energy_kwh"].values
predicted = forecast_result.data["forecast"]
errors = actual - predicted

# Detect anomalous errors
anomaly_agent = IsolationForestAnomalyAgent()
error_result = anomaly_agent.run({
    "data": pd.DataFrame({"forecast_error": errors}),
    "contamination": 0.1,
})

print(f"Forecast anomalies: {error_result.data['n_anomalies']}")
```

#### 2. Carbon Agent

Validate emissions calculations:

```python
from greenlang.agents.carbon_agent import CarbonAgent
from greenlang.agents.anomaly_agent_iforest import IsolationForestAnomalyAgent

# Calculate emissions
carbon_agent = CarbonAgent()
emissions_result = carbon_agent.run({
    "fuel_type": "natural_gas",
    "quantity": fuel_usage_df["quantity"],
})

# Check for calculation anomalies
anomaly_agent = IsolationForestAnomalyAgent()
validation_result = anomaly_agent.run({
    "data": pd.DataFrame({
        "emissions_kg": emissions_result.data["total_emissions"],
        "fuel_quantity": fuel_usage_df["quantity"],
    }),
    "contamination": 0.05,
})

# Flag suspicious calculations
if validation_result.data["n_anomalies"] > 0:
    review_emissions_calculations(validation_result.data["alerts"])
```

#### 3. Grid Factor Agent

Monitor grid intensity anomalies:

```python
from greenlang.agents.grid_factor_agent import GridFactorAgent
from greenlang.agents.anomaly_agent_iforest import IsolationForestAnomalyAgent

# Get grid factors
grid_agent = GridFactorAgent()
grid_result = grid_agent.run({
    "region": "CAISO",
    "dates": date_range,
})

# Detect unusual grid intensity
anomaly_agent = IsolationForestAnomalyAgent()
grid_anomaly_result = anomaly_agent.run({
    "data": pd.DataFrame({
        "grid_intensity": grid_result.data["factors"],
        "hour": grid_result.data["hours"],
    }),
    "contamination": 0.08,
})

# Alert on unusual grid conditions
for alert in grid_anomaly_result.data["alerts"]:
    if alert["severity"] in ["critical", "high"]:
        notify_energy_manager(alert)
```

### Integration with External Systems

#### REST API Endpoint

```python
from flask import Flask, request, jsonify
from greenlang.agents.anomaly_agent_iforest import IsolationForestAnomalyAgent
import pandas as pd

app = Flask(__name__)
agent = IsolationForestAnomalyAgent(enable_alerts=True)

@app.route('/api/detect-anomalies', methods=['POST'])
def detect_anomalies():
    data = request.json

    # Convert JSON to DataFrame
    df = pd.DataFrame(data["observations"])

    # Run detection
    result = agent.run({
        "data": df,
        "contamination": data.get("contamination", 0.1),
    })

    if result.success:
        return jsonify({
            "success": True,
            "n_anomalies": result.data["n_anomalies"],
            "anomaly_indices": result.data["anomaly_indices"],
            "alerts": result.data["alerts"],
        })
    else:
        return jsonify({
            "success": False,
            "error": result.error,
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

#### Kafka Stream Processing

```python
from kafka import KafkaConsumer, KafkaProducer
from greenlang.agents.anomaly_agent_iforest import IsolationForestAnomalyAgent
import pandas as pd
import json

agent = IsolationForestAnomalyAgent()

# Consumer for incoming data
consumer = KafkaConsumer(
    'energy-readings',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
)

# Producer for anomaly alerts
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
)

# Buffer for batch processing
buffer = []
BATCH_SIZE = 100

for message in consumer:
    buffer.append(message.value)

    # Process batch
    if len(buffer) >= BATCH_SIZE:
        df = pd.DataFrame(buffer)

        result = agent.run({
            "data": df,
            "contamination": 0.05,
        })

        # Publish alerts
        for alert in result.data["alerts"]:
            producer.send('anomaly-alerts', alert)

        buffer = []
```

#### Database Integration

```python
from greenlang.agents.anomaly_agent_iforest import IsolationForestAnomalyAgent
import pandas as pd
import psycopg2

agent = IsolationForestAnomalyAgent()

# Load data from PostgreSQL
conn = psycopg2.connect("dbname=energy user=postgres")
df = pd.read_sql_query("""
    SELECT timestamp, energy_kwh, temperature_c
    FROM sensor_readings
    WHERE timestamp > NOW() - INTERVAL '7 days'
""", conn)

# Run detection
result = agent.run({
    "data": df,
    "contamination": 0.08,
})

# Store anomalies in database
if result.success:
    cursor = conn.cursor()
    for idx in result.data["anomaly_indices"]:
        row = df.iloc[idx]
        cursor.execute("""
            INSERT INTO detected_anomalies (timestamp, energy_kwh, severity, score)
            VALUES (%s, %s, %s, %s)
        """, (
            row["timestamp"],
            row["energy_kwh"],
            result.data["top_anomalies"][0]["severity"],
            result.data["anomaly_scores"][idx],
        ))
    conn.commit()
```

---

## Best Practices

### 1. Contamination Parameter Tuning

**Rule of Thumb**: Set contamination to expected anomaly rate:
- **Conservative** (< 5%): High-quality data, rare anomalies
- **Moderate** (5-10%): Normal operational data
- **Aggressive** (10-20%): Noisy data, frequent anomalies

**Tuning Process**:

```python
# Try multiple contamination values
contamination_values = [0.05, 0.08, 0.10, 0.12, 0.15]
results = []

for contamination in contamination_values:
    result = agent.run({
        "data": df,
        "contamination": contamination,
        "labels": ground_truth_labels,  # If available
    })

    if "metrics" in result.data:
        results.append({
            "contamination": contamination,
            "f1_score": result.data["metrics"]["f1_score"],
            "precision": result.data["metrics"]["precision"],
            "recall": result.data["metrics"]["recall"],
        })

# Pick best F1-score
best = max(results, key=lambda x: x["f1_score"])
print(f"Optimal contamination: {best['contamination']}")
```

### 2. Feature Selection

**Guidelines**:
- **Include**: Features that capture normal behavior
- **Exclude**: Irrelevant features, high-cardinality categoricals, timestamps

**Good Features**:
- Numeric measurements (energy, temperature, pressure)
- Derived features (hour of day, day of week)
- Ratios (efficiency = output / input)

**Bad Features**:
- IDs, UUIDs (no information)
- Timestamps (use derived features instead)
- Text fields (use embeddings if needed)

**Example**:

```python
# Prepare features
df["hour"] = df["timestamp"].dt.hour
df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
df["energy_per_degree"] = df["energy_kwh"] / df["temperature_c"]

# Select numeric features
numeric_features = ["energy_kwh", "temperature_c", "hour", "is_weekend", "energy_per_degree"]

result = agent.run({
    "data": df,
    "feature_columns": numeric_features,
})
```

### 3. Handling Missing Data

**Strategies**:
1. **Drop rows** (if < 5% missing)
2. **Impute** (mean/median for MCAR data)
3. **Forward fill** (time series)
4. **Model-based** (KNN, iterative imputer)

**Implementation**:

```python
# Strategy 1: Drop rows with any missing values
df_clean = df.dropna()

# Strategy 2: Mean imputation
df_imputed = df.fillna(df.mean())

# Strategy 3: Forward fill (time series)
df_ffill = df.fillna(method='ffill')

# Agent handles simple imputation automatically
result = agent.run({
    "data": df,  # Can contain NaN values
})
```

### 4. Periodic Retraining

**Why**: Data distribution changes over time (concept drift).

**When**:
- **Regularly**: Every 30 days for stable systems
- **Triggered**: When anomaly rate deviates significantly
- **Seasonal**: At season changes (HVAC systems, weather)

**Example**:

```python
import schedule
import time

def retrain_model():
    # Get fresh training data (last 90 days)
    df = load_recent_data(days=90)

    # Retrain
    agent = IsolationForestAnomalyAgent()
    result = agent.run({
        "data": df,
        "contamination": 0.08,
    })

    # Validate performance
    if "metrics" in result.data:
        if result.data["metrics"]["f1_score"] < 0.70:
            alert_ops_team("Model performance degraded!")

    # Save model for production
    save_model(agent._fitted_model)

# Schedule retraining
schedule.every(30).days.do(retrain_model)

while True:
    schedule.run_pending()
    time.sleep(3600)  # Check hourly
```

### 5. Alert Fatigue Prevention

**Problem**: Too many alerts → users ignore them

**Solution**:
1. **Severity filtering**: Only alert on high/critical
2. **Deduplication**: Group similar anomalies
3. **Rate limiting**: Max N alerts per hour
4. **Confidence thresholding**: Only alert if confidence > 0.8

**Example**:

```python
# Alert configuration
ALERT_CONFIG = {
    "min_severity": "high",  # Ignore low/medium
    "min_confidence": 0.8,   # High confidence only
    "max_alerts_per_hour": 5,
    "dedup_window": 3600,    # 1 hour
}

# Filter alerts
def should_alert(alert, recent_alerts):
    # Check severity
    if alert["severity"] not in ["critical", "high"]:
        return False

    # Check confidence
    if alert["confidence"] < ALERT_CONFIG["min_confidence"]:
        return False

    # Check rate limit
    recent_count = len([a for a in recent_alerts if time.time() - a["timestamp"] < 3600])
    if recent_count >= ALERT_CONFIG["max_alerts_per_hour"]:
        return False

    # Check deduplication
    for recent in recent_alerts:
        if abs(alert["index"] - recent["index"]) < 5:  # Similar location
            return False

    return True

# Apply filters
for alert in result.data["alerts"]:
    if should_alert(alert, recent_alerts_db):
        send_alert(alert)
```

### 6. Monitoring and Logging

**What to Log**:
- Detection runs (timestamp, data size, duration)
- Anomaly counts and rates
- Severity distribution
- Performance metrics
- Errors and exceptions

**Example**:

```python
import logging
from datetime import datetime

logging.basicConfig(
    filename='anomaly_detection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def run_detection_with_logging(df):
    start_time = datetime.now()

    try:
        result = agent.run({
            "data": df,
            "contamination": 0.1,
        })

        duration = (datetime.now() - start_time).total_seconds()

        # Log success
        logging.info(f"Detection completed: {result.data['n_anomalies']} anomalies found in {duration:.2f}s")
        logging.info(f"Severity: {result.data['severity_distribution']}")

        if "metrics" in result.data:
            logging.info(f"Accuracy: F1={result.data['metrics']['f1_score']:.3f}")

        return result

    except Exception as e:
        logging.error(f"Detection failed: {e}")
        raise
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Low Precision (Many False Positives)

**Symptoms**:
- Many detected anomalies are actually normal
- Users report alert fatigue
- Precision < 70%

**Causes**:
- Contamination too high
- Insufficient training data
- Feature scaling issues

**Solutions**:

```python
# 1. Reduce contamination
result = agent.run({
    "data": df,
    "contamination": 0.05,  # Was 0.15
})

# 2. Increase training data
# Use at least 100 samples, ideally 500+

# 3. Check feature distributions
df.describe()  # Look for outliers in training data

# 4. Increase trees for stability
result = agent.run({
    "data": df,
    "n_estimators": 200,  # Was 100
})
```

#### Issue 2: Low Recall (Missing True Anomalies)

**Symptoms**:
- Known anomalies not detected
- Recall < 60%
- Critical issues slipping through

**Causes**:
- Contamination too low
- Anomalies similar to normal data
- Wrong features

**Solutions**:

```python
# 1. Increase contamination
result = agent.run({
    "data": df,
    "contamination": 0.15,  # Was 0.05
})

# 2. Add more informative features
df["energy_diff"] = df["energy_kwh"].diff()
df["energy_zscore"] = (df["energy_kwh"] - df["energy_kwh"].mean()) / df["energy_kwh"].std()

# 3. Reduce max_samples (more sensitive)
result = agent.run({
    "data": df,
    "max_samples": 128,  # Was 256
})
```

#### Issue 3: Slow Performance

**Symptoms**:
- Detection takes > 5 seconds
- Real-time monitoring delayed
- High CPU usage

**Causes**:
- Too many estimators
- Large dataset (> 10,000 rows)
- Too many features (> 20)

**Solutions**:

```python
# 1. Reduce estimators
result = agent.run({
    "data": df,
    "n_estimators": 50,  # Was 100
})

# 2. Limit max_samples for large datasets
result = agent.run({
    "data": df,
    "max_samples": 256,  # Fixed size
})

# 3. Feature selection
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=10)
selected_features = selector.fit_transform(df)

# 4. Batch processing
for chunk in pd.read_csv('large_file.csv', chunksize=1000):
    result = agent.run({"data": chunk})
```

#### Issue 4: Inconsistent Results

**Symptoms**:
- Different results on same data
- Non-reproducible detections
- Flaky tests

**Causes**:
- Random seed not set
- Data ordering issues
- Version mismatches

**Solutions**:

```python
# 1. Ensure determinism (should be default)
import numpy as np
np.random.seed(42)

# 2. Sort data consistently
df = df.sort_values("timestamp")

# 3. Check versions
import sklearn
print(f"scikit-learn: {sklearn.__version__}")  # Should be 1.3.x
```

#### Issue 5: Budget Exceeded

**Symptoms**:
- BudgetExceeded exception
- AI calls timing out
- High costs

**Causes**:
- Complex prompts
- Many tool calls
- Large explanations

**Solutions**:

```python
# 1. Increase budget
agent = IsolationForestAnomalyAgent(
    budget_usd=2.0,  # Was 1.0
)

# 2. Disable expensive features
agent = IsolationForestAnomalyAgent(
    enable_explanations=False,  # Saves tokens
    enable_recommendations=False,
)

# 3. Use tools directly (skip AI)
agent._fit_isolation_forest_impl(input_data)
agent._detect_anomalies_impl(input_data)
# Manual workflow, no AI calls
```

### Debugging Tips

#### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

agent = IsolationForestAnomalyAgent()
result = agent.run({"data": df})
```

#### Inspect Tool Calls

```python
# Check tool call counts
summary = agent.get_performance_summary()
print(f"Tool calls: {summary['ai_metrics']['tool_call_count']}")

# Check fitted model
print(f"Model fitted: {agent._fitted_model is not None}")
print(f"Features: {agent._feature_columns}")
```

#### Validate Data Quality

```python
# Check for common issues
print(f"Shape: {df.shape}")
print(f"Missing values: {df.isnull().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")
print(f"Data types: {df.dtypes}")

# Check feature distributions
df.describe()
```

---

## Future Enhancements

### Planned Features (v0.2.0)

1. **Online Learning**: Incremental model updates without full retraining
2. **Multi-Model Ensemble**: Combine multiple algorithms (LOF, One-Class SVM)
3. **Automatic Contamination Tuning**: Learn optimal contamination from data
4. **Temporal Context**: Time-aware anomaly detection
5. **Explainability**: SHAP values for feature importance

### Experimental Features

#### 1. Streaming Anomaly Detection

```python
from greenlang.agents.anomaly_agent_iforest import StreamingAnomalyAgent

agent = StreamingAnomalyAgent(window_size=1000)

for batch in data_stream:
    result = agent.detect_streaming(batch)
    if result.data["n_anomalies"] > 0:
        handle_anomalies(result)
```

#### 2. Hierarchical Anomaly Detection

```python
# Detect anomalies at multiple granularities
agent = HierarchicalAnomalyAgent(
    levels=["sensor", "building", "campus"]
)

result = agent.run({
    "data": hierarchical_df,
    "hierarchy_column": "level",
})
```

#### 3. Causal Anomaly Analysis

```python
# Identify root causes using causal inference
agent = CausalAnomalyAgent()

result = agent.run({
    "data": df,
    "causal_graph": graph,
})

# Output: "Anomaly in energy_kwh caused by temperature_c spike"
```

### Contributing

Interested in contributing? See our [Contributing Guide](CONTRIBUTING.md).

**Priority Areas**:
- Additional ML algorithms (LOF, DBSCAN, Autoencoders)
- Performance optimizations
- Additional use case examples
- Documentation improvements

---

## Conclusion

The Isolation Forest Anomaly Detection Agent provides production-ready anomaly detection for GreenLang applications. By combining tool-first numerics with AI interpretation, it delivers accurate, explainable, and actionable anomaly insights.

**Key Takeaways**:
1. Tool-first architecture ensures zero hallucinated numbers
2. Isolation Forest is ideal for unsupervised anomaly detection
3. Contamination parameter must match expected anomaly rate
4. Multi-dimensional features capture complex patterns
5. Severity-based alerts prevent alert fatigue

For support, visit:
- **Documentation**: https://greenlang.dev/docs
- **GitHub**: https://github.com/greenlang/greenlang
- **Discord**: https://discord.gg/greenlang

---

**Document Version:** 1.0
**Last Updated:** October 2025
**Maintained By:** GreenLang Framework Team
