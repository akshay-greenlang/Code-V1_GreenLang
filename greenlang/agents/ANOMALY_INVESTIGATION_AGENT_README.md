# Anomaly Investigation Agent

**Pattern:** InsightAgent (Hybrid Architecture)
**Category:** AgentCategory.INSIGHT
**Temperature:** 0.6 (analytical consistency)
**Version:** 1.0.0

## Overview

The Anomaly Investigation Agent combines deterministic anomaly detection with AI-powered root cause analysis. It follows the InsightAgent pattern with clear separation between calculations and insights.

## Architecture

### Deterministic Layer (calculate method)
- **Detection Engine:** Isolation Forest (from `anomaly_agent_iforest.py`)
- **Outputs:** Anomaly scores, severity classification, pattern identification
- **Characteristics:** 100% reproducible, full audit trail, no AI/LLM

### AI-Powered Layer (explain method)
- **Investigation:** Root cause analysis with evidence gathering
- **Tools:** 3 diagnostic tools for evidence collection
- **RAG:** 4 specialized knowledge collections
- **Output:** Comprehensive investigation report with recommendations

## Usage

### Basic Usage

```python
import pandas as pd
from greenlang.agents.anomaly_investigation_agent import AnomalyInvestigationAgent
from greenlang.intelligence import ChatSession, create_provider
from greenlang.intelligence.rag import RAGEngine

# Initialize agent
agent = AnomalyInvestigationAgent(
    enable_audit_trail=True,
    detection_budget_usd=1.00,
    investigation_budget_usd=2.00
)

# Prepare data
data = pd.DataFrame({
    "energy_kwh": [100, 105, 98, 500, 102, 99, ...],
    "temperature_c": [20, 22, 21, 35, 20, 21, ...],
    "humidity_pct": [60, 62, 58, 85, 61, 59, ...]
})

# Step 1: Deterministic anomaly detection
detection_result = agent.calculate({
    "data": data,
    "contamination": 0.05,  # Expected 5% anomalies
    "feature_columns": ["energy_kwh", "temperature_c"],
    "system_type": "HVAC",
    "system_id": "HVAC-001",
    "location": "Building A"
})

print(f"Anomalies detected: {detection_result['n_anomalies']}")
print(f"Severity distribution: {detection_result['severity_distribution']}")

# Step 2: AI-powered root cause investigation
chat_session = ChatSession(create_provider())
rag_engine = RAGEngine(...)  # Configure your RAG engine

investigation_report = await agent.explain(
    calculation_result=detection_result,
    context={
        "data": data,
        "system_type": "HVAC",
        "system_id": "HVAC-001",
        "location": "Building A",
        "investigation_depth": "comprehensive"
    },
    session=chat_session,
    rag_engine=rag_engine,
    temperature=0.6
)

print(investigation_report)
```

### Input Schema

#### calculate() method
```python
{
    "data": pd.DataFrame,              # Required: sensor/energy data
    "contamination": float,            # Optional: expected anomaly rate (0-0.5), default: 0.1
    "n_estimators": int,               # Optional: number of trees, default: 100
    "feature_columns": List[str],      # Optional: specific features to analyze
    "timestamp_column": str,           # Optional: for temporal analysis
    "location": str,                   # Optional: system location
    "system_id": str                   # Optional: system identifier
}
```

#### explain() method context
```python
{
    "data": pd.DataFrame,              # Required: original data
    "timestamp_column": str,           # Optional: timestamp column name
    "location": str,                   # Optional: system location
    "system_id": str,                  # Optional: system identifier
    "system_type": str,                # Optional: e.g., "HVAC", "solar", "grid"
    "recent_changes": List[str],       # Optional: recent system changes
    "investigation_depth": str         # Optional: "quick", "standard", "comprehensive"
}
```

## Investigation Tools

### 1. maintenance_log_tool
Query maintenance logs for events correlating with anomalies.

**Parameters:**
- `start_time`: Start timestamp (ISO format)
- `end_time`: End timestamp (ISO format)
- `system_id`: System identifier
- `event_types`: Event types to filter (e.g., 'maintenance', 'repair', 'calibration')

**Returns:**
- Maintenance events during anomaly periods
- Correlation strength
- Confidence level

### 2. sensor_diagnostic_tool
Check sensor health, calibration, and failure indicators.

**Parameters:**
- `sensor_ids`: List of sensor IDs
- `diagnostic_type`: "calibration", "health", "drift", or "comprehensive"
- `time_window`: Analysis window (e.g., "7d", "30d")

**Returns:**
- Sensor health status
- Calibration status
- Drift detection results
- Recommendations

### 3. weather_data_tool
Retrieve weather conditions during anomaly periods.

**Parameters:**
- `location`: Location identifier
- `start_time`: Start timestamp (ISO format)
- `end_time`: End timestamp (ISO format)
- `metrics`: Weather metrics to retrieve

**Returns:**
- Weather conditions (temperature, humidity, wind, etc.)
- Extreme weather events
- Correlation with anomalies

## RAG Collections

### anomaly_patterns
Historical anomaly cases and their resolutions.

**Content:**
- Past anomaly investigations
- Resolution strategies
- Success rates
- Lessons learned

### root_cause_database
Known failure modes and diagnostic patterns.

**Content:**
- Equipment failure signatures
- Common root causes by system type
- Diagnostic decision trees
- Failure probabilities

### sensor_specifications
Sensor behavior, failure modes, and calibration requirements.

**Content:**
- Sensor specifications
- Calibration schedules
- Known failure modes
- Drift patterns
- Maintenance requirements

### maintenance_procedures
Maintenance correlation patterns and preventive measures.

**Content:**
- Maintenance schedules
- Preventive procedures
- Maintenance impact patterns
- Post-maintenance signatures

## Output Schema

### Detection Result (calculate method)
```python
{
    "anomalies": List[bool],                    # Boolean array of anomaly flags
    "anomaly_scores": List[float],              # Anomaly scores (-1 to 1)
    "anomaly_indices": List[int],               # Indices of detected anomalies
    "severity_distribution": Dict[str, int],    # Count by severity level
    "top_anomalies": List[Dict],                # Top N anomalies with details
    "patterns": Dict,                           # Feature importance and patterns
    "n_anomalies": int,                         # Total anomalies detected
    "anomaly_rate": float,                      # Percentage of anomalies
    "model_info": Dict,                         # Model parameters
    "calculation_trace": List[str]              # Step-by-step audit trail
}
```

### Investigation Report (explain method)
Comprehensive markdown report including:
- Executive summary
- Root cause analysis
- Evidence from tools (maintenance, sensors, weather)
- Pattern analysis insights
- Confidence-scored recommendations
- Calculation audit trail
- Investigation metadata

## Key Features

### Deterministic Guarantees
- Same inputs always produce same anomaly detections
- Full audit trail for regulatory compliance
- Reproducible for compliance verification
- No AI/LLM in calculation path

### AI Value-Add
- Evidence-based root cause analysis
- Multi-source investigation (maintenance, sensors, environment)
- Historical knowledge integration via RAG
- Confidence-scored recommendations
- Actionable remediation strategies

### Compliance Ready
- Full audit trail for calculations
- Transparent AI reasoning
- Evidence citations
- Confidence levels for all conclusions
- Reproducibility verification

## Examples

### Example 1: Energy Spike Investigation

```python
# Detect energy consumption anomalies
result = agent.calculate({
    "data": energy_data,
    "contamination": 0.05,
    "feature_columns": ["energy_kwh", "demand_kw"],
    "system_type": "electrical_grid",
    "system_id": "GRID-EAST-01"
})

# Investigate root cause
report = await agent.explain(
    calculation_result=result,
    context={
        "data": energy_data,
        "system_type": "electrical_grid",
        "system_id": "GRID-EAST-01",
        "location": "East Coast",
        "investigation_depth": "comprehensive"
    },
    session=chat_session,
    rag_engine=rag_engine
)
```

### Example 2: HVAC Temperature Anomaly

```python
# Detect temperature anomalies
result = agent.calculate({
    "data": hvac_data,
    "contamination": 0.1,
    "feature_columns": ["temperature_c", "humidity_pct", "energy_kwh"],
    "system_type": "HVAC",
    "system_id": "HVAC-B2-F3"
})

# Quick investigation
report = await agent.explain(
    calculation_result=result,
    context={
        "data": hvac_data,
        "system_type": "HVAC",
        "system_id": "HVAC-B2-F3",
        "location": "Building 2, Floor 3",
        "recent_changes": ["Filter replaced 2025-11-01"],
        "investigation_depth": "quick"
    },
    session=chat_session,
    rag_engine=rag_engine,
    temperature=0.6
)
```

### Example 3: Solar Panel Performance

```python
# Detect solar generation anomalies
result = agent.calculate({
    "data": solar_data,
    "contamination": 0.08,
    "feature_columns": ["generation_kwh", "efficiency_pct", "panel_temp_c"],
    "system_type": "solar_pv",
    "system_id": "SOLAR-ARRAY-A"
})

# Standard investigation with weather correlation
report = await agent.explain(
    calculation_result=result,
    context={
        "data": solar_data,
        "system_type": "solar_pv",
        "system_id": "SOLAR-ARRAY-A",
        "location": "Rooftop Array A",
        "investigation_depth": "standard"
    },
    session=chat_session,
    rag_engine=rag_engine
)
```

## Performance Metrics

```python
# Get performance summary
summary = agent.get_performance_summary()

print(f"Total investigations: {summary['total_investigations']}")
print(f"Total cost: ${summary['total_cost_usd']:.2f}")
print(f"Avg cost per investigation: ${summary['avg_cost_per_investigation']:.2f}")
```

## Best Practices

### 1. Data Quality
- Ensure sufficient data points (minimum 100 samples)
- Handle missing values appropriately
- Validate timestamp alignment
- Check for data collection issues

### 2. Contamination Parameter
- Start with 0.05-0.1 (5-10% expected anomalies)
- Adjust based on system characteristics
- Too low: miss real anomalies
- Too high: false positives

### 3. Investigation Depth
- **Quick:** For routine checks, basic evidence gathering
- **Standard:** For normal investigations, balanced analysis
- **Comprehensive:** For critical systems, full evidence collection

### 4. Context Information
- Provide system_type for relevant RAG retrieval
- Include system_id for maintenance log queries
- Add location for weather correlation
- List recent_changes for post-change analysis

### 5. Cost Management
- Adjust budgets based on criticality
- Use "quick" depth for routine checks
- Reserve "comprehensive" for critical incidents
- Monitor costs via performance summary

## Integration with Existing Systems

### With Monitoring Systems
```python
# Continuous monitoring integration
while True:
    recent_data = get_recent_sensor_data(hours=24)

    result = agent.calculate({
        "data": recent_data,
        "contamination": 0.05,
        "system_type": "HVAC",
        "system_id": system_id
    })

    if result['n_anomalies'] > 0 and result['severity_distribution'].get('critical', 0) > 0:
        # Trigger investigation for critical anomalies
        report = await agent.explain(
            calculation_result=result,
            context={"data": recent_data, ...},
            session=chat_session,
            rag_engine=rag_engine
        )

        send_alert(report)

    time.sleep(3600)  # Check hourly
```

### With Maintenance Systems
```python
# Post-maintenance verification
maintenance_event = {
    "timestamp": "2025-11-06T10:00:00Z",
    "type": "sensor_calibration",
    "system_id": "HVAC-001"
}

# Collect data before and after
before_data = get_data(start=maintenance_event["timestamp"] - timedelta(days=7), ...)
after_data = get_data(start=maintenance_event["timestamp"], ...)

# Detect anomalies in post-maintenance data
result = agent.calculate({"data": after_data, ...})

# Investigate if anomalies appear after maintenance
if result['n_anomalies'] > expected_baseline:
    report = await agent.explain(
        calculation_result=result,
        context={
            "data": after_data,
            "recent_changes": [f"{maintenance_event['type']} at {maintenance_event['timestamp']}"],
            ...
        },
        session=chat_session,
        rag_engine=rag_engine
    )
```

## Troubleshooting

### No Anomalies Detected
- Check contamination parameter (may be too low)
- Verify data has sufficient variability
- Ensure numeric features are present
- Review feature scaling requirements

### Too Many False Positives
- Increase contamination parameter
- Review feature selection
- Check for seasonal patterns
- Validate sensor calibration

### Low Confidence in Investigation
- Verify RAG collections are populated
- Check tool connectivity (maintenance logs, weather API)
- Increase investigation_depth
- Provide more context information

### High Costs
- Reduce investigation_depth to "quick" or "standard"
- Lower investigation_budget_usd
- Batch investigations for multiple anomalies
- Cache RAG results for similar queries

## Version History

### v1.0.0 (2025-11-06)
- Initial release
- InsightAgent pattern implementation
- Integration with anomaly_agent_iforest
- 3 investigation tools
- 4 RAG collections
- Comprehensive investigation reports

## References

- **Base Pattern:** `greenlang/agents/base_agents.py` - InsightAgent class
- **Detection Engine:** `greenlang/agents/anomaly_agent_iforest.py`
- **Example Pattern:** `greenlang/agents/benchmark_agent_ai.py`
- **Categories:** `greenlang/agents/categories.py`

## Support

For issues, questions, or contributions, see the main GreenLang documentation or contact the development team.
