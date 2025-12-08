# Weights & Biases Integration Guide for GreenLang Process Heat Agents

## Overview

This guide covers the Weights & Biases (W&B) integration for GreenLang Process Heat agent ML experiments. The integration provides comprehensive experiment tracking, model versioning, hyperparameter sweeps, and team collaboration capabilities while maintaining GreenLang's zero-hallucination guarantee.

## Table of Contents

1. [Setup and Configuration](#setup-and-configuration)
2. [Basic Usage](#basic-usage)
3. [Agent-Specific Configuration](#agent-specific-configuration)
4. [Hyperparameter Sweeps](#hyperparameter-sweeps)
5. [Alerting and Notifications](#alerting-and-notifications)
6. [Report Generation](#report-generation)
7. [MLflow Integration Bridge](#mlflow-integration-bridge)
8. [Caching Strategy for Cost Reduction](#caching-strategy-for-cost-reduction)
9. [Best Practices](#best-practices)
10. [Dashboard Configuration](#dashboard-configuration)
11. [Team Collaboration](#team-collaboration)
12. [Troubleshooting](#troubleshooting)

---

## Setup and Configuration

### Installation

Install W&B and the GreenLang ML package:

```bash
pip install wandb>=0.15.0
pip install greenlang[ml]
```

### API Key Configuration

#### Option 1: Environment Variable (Recommended)

```bash
export WANDB_API_KEY=your_api_key_here
```

#### Option 2: Configuration File

Create a `.env` file:

```env
WANDB_API_KEY=your_api_key_here
WANDB_ENTITY=your-team-name
WANDB_PROJECT=greenlang-process-heat
```

#### Option 3: Programmatic Configuration

```python
from greenlang.ml.mlops.wandb_integration import WandBConfig, WandBExperimentTracker

config = WandBConfig(
    api_key="your_api_key_here",
    project="greenlang-process-heat",
    entity="your-team-name"
)

tracker = WandBExperimentTracker(config=config)
```

### Configuration Options

The `WandBConfig` class provides comprehensive configuration:

```python
from greenlang.ml.mlops.wandb_integration import WandBConfig

config = WandBConfig(
    # API Configuration
    api_key=None,  # Uses WANDB_API_KEY env var if not set
    api_key_env_var="WANDB_API_KEY",

    # Project Settings
    project="greenlang-process-heat",
    entity="your-team-name",

    # Run Settings
    run_name_prefix="greenlang",
    run_name_separator="_",

    # Default Tags
    default_tags=["process-heat", "greenlang", "zero-hallucination"],

    # Logging Settings
    log_frequency=100,
    log_code=True,
    log_git=True,

    # Storage Settings
    offline_mode=False,
    dir="./wandb_runs",

    # Caching Settings (for 66% cost reduction)
    enable_caching=True,
    cache_dir="./wandb_cache",
    cache_ttl_hours=24,

    # Provenance Settings
    enable_provenance=True,
)
```

---

## Basic Usage

### Quick Start

```python
from greenlang.ml.mlops.wandb_integration import (
    WandBExperimentTracker,
    AgentType,
    create_experiment_tracker
)

# Quick initialization
tracker = create_experiment_tracker(
    project="greenlang-process-heat",
    entity="your-team"
)

# Start a training run
with tracker.init_run(
    run_name="fuel_model_training",
    agent_type=AgentType.GL_008_FUEL
):
    # Log hyperparameters
    tracker.log_hyperparameters({
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 100,
        "n_estimators": 200
    })

    # Training loop
    for epoch in range(100):
        loss = train_epoch(model, data)
        accuracy = evaluate(model, val_data)

        # Log metrics
        tracker.log_metrics({
            "loss": loss,
            "accuracy": accuracy,
            "epoch": epoch
        }, step=epoch)

    # Log final model
    tracker.log_model(model, "fuel_emission_model")

    # Log training data artifact
    tracker.log_artifact(
        "./data/training_data.parquet",
        "training_data",
        artifact_type="dataset"
    )
```

### Logging Metrics

```python
# Single metric
tracker.log_metrics({"loss": 0.1})

# Multiple metrics
tracker.log_metrics({
    "loss": 0.1,
    "accuracy": 0.95,
    "rmse": 0.05,
    "r2_score": 0.92
})

# With step for time series
for step in range(100):
    tracker.log_metrics({"loss": compute_loss()}, step=step)
```

### Logging Hyperparameters

```python
tracker.log_hyperparameters({
    "learning_rate": 0.01,
    "batch_size": 32,
    "hidden_layers": [256, 128, 64],
    "dropout_rate": 0.3,
    "optimizer": "adam",
    "fuel_type_embeddings": True
})
```

### Logging Models

```python
# Basic model logging
tracker.log_model(trained_model, "fuel_emission_model")

# With metadata and aliases
tracker.log_model(
    trained_model,
    "fuel_emission_model",
    metadata={
        "framework": "sklearn",
        "training_samples": 10000,
        "features": ["temperature", "pressure", "flow_rate"]
    },
    aliases=["latest", "production-candidate"]
)
```

### Logging Artifacts

```python
# Dataset
tracker.log_artifact(
    "./data/train.csv",
    "training_data",
    artifact_type="dataset"
)

# Feature importance plot
tracker.log_artifact(
    "./plots/feature_importance.png",
    "feature_importance",
    artifact_type="visualization"
)

# Configuration file
tracker.log_artifact(
    "./config/model_config.yaml",
    "model_config",
    artifact_type="config"
)
```

### Logging Tables

```python
# From dictionary
tracker.log_table({
    "epoch": [1, 2, 3, 4, 5],
    "loss": [0.5, 0.3, 0.2, 0.15, 0.1],
    "accuracy": [0.8, 0.85, 0.9, 0.92, 0.95]
}, "training_progress")

# From pandas DataFrame
import pandas as pd
df = pd.DataFrame({
    "feature": ["temp", "pressure", "flow"],
    "importance": [0.4, 0.35, 0.25]
})
tracker.log_table(df, "feature_importance")
```

---

## Agent-Specific Configuration

### Supported Agent Types

The integration supports all 20 GreenLang Process Heat agents:

```python
from greenlang.ml.mlops.wandb_integration import AgentType

# Carbon & Regulatory Agents
AgentType.GL_001_CARBON      # Carbon footprint calculation
AgentType.GL_002_CBAM        # CBAM compliance
AgentType.GL_003_CSRD        # CSRD reporting
AgentType.GL_004_EUDR        # EU Deforestation Regulation
AgentType.GL_005_BUILDING    # Building energy
AgentType.GL_006_SCOPE3      # Scope 3 emissions
AgentType.GL_007_TAXONOMY    # EU Taxonomy alignment

# Process Heat Specific Agents
AgentType.GL_008_FUEL        # Fuel analysis
AgentType.GL_009_HEAT_RECOVERY  # Heat recovery optimization
AgentType.GL_010_COMBUSTION  # Combustion efficiency
AgentType.GL_011_STEAM       # Steam system optimization
AgentType.GL_012_THERMAL     # Thermal process modeling
AgentType.GL_013_PROCESS     # Process heat integration
AgentType.GL_014_EFFICIENCY  # Energy efficiency
AgentType.GL_015_EMISSIONS   # Emissions monitoring
AgentType.GL_016_OPTIMIZATION  # Process optimization
AgentType.GL_017_PREDICTION  # Demand prediction
AgentType.GL_018_MONITORING  # Real-time monitoring
AgentType.GL_019_CONTROL     # Control systems
AgentType.GL_020_SAFETY      # Safety compliance
```

### Default Hyperparameters by Agent

Each agent type has optimized default hyperparameters:

```python
from greenlang.ml.mlops.wandb_integration import ProcessHeatRunConfig, AgentType

# Get defaults for Fuel agent
fuel_params = ProcessHeatRunConfig.get_default_hyperparameters(AgentType.GL_008_FUEL)
# Returns: {"n_estimators": 200, "max_depth": 15, "learning_rate": 0.08, ...}

# Get defaults for Combustion agent (neural network)
combustion_params = ProcessHeatRunConfig.get_default_hyperparameters(AgentType.GL_010_COMBUSTION)
# Returns: {"hidden_layers": [256, 128, 64], "dropout_rate": 0.3, ...}

# Get defaults for Prediction agent (LSTM)
prediction_params = ProcessHeatRunConfig.get_default_hyperparameters(AgentType.GL_017_PREDICTION)
# Returns: {"sequence_length": 24, "lstm_units": 128, "attention_heads": 8, ...}
```

### Agent-Specific Run Configuration

```python
from greenlang.ml.mlops.wandb_integration import ProcessHeatRunConfig, AgentType, SweepMethod

config = ProcessHeatRunConfig(
    agent_type=AgentType.GL_008_FUEL,
    model_name="fuel_emission_predictor",
    agent_version="2.1.0",
    model_type="xgboost",

    # Sweep settings
    sweep_enabled=True,
    sweep_method=SweepMethod.BAYES,
    sweep_count=100,

    # Optimization settings
    primary_metric="rmse",
    metric_goal="minimize",

    # Data provenance
    data_hash="sha256:abc123..."
)
```

---

## Hyperparameter Sweeps

### Creating a Sweep

```python
from greenlang.ml.mlops.wandb_integration import (
    WandBSweepManager,
    AgentType,
    SweepMethod
)

sweep_manager = WandBSweepManager()

# Create sweep with default parameters for agent type
sweep_id = sweep_manager.create_sweep(
    agent_type=AgentType.GL_008_FUEL,
    method=SweepMethod.BAYES,
    metric="rmse",
    goal="minimize"
)

print(f"Created sweep: {sweep_id}")
```

### Custom Sweep Parameters

```python
sweep_id = sweep_manager.create_sweep(
    agent_type=AgentType.GL_010_COMBUSTION,
    method=SweepMethod.BAYES,
    metric="rmse",
    goal="minimize",
    parameters={
        "hidden_layers": {
            "values": [[128, 64], [256, 128], [256, 128, 64], [512, 256, 128]]
        },
        "dropout_rate": {
            "min": 0.1,
            "max": 0.5,
            "distribution": "uniform"
        },
        "learning_rate": {
            "min": 0.0001,
            "max": 0.01,
            "distribution": "log_uniform_values"
        },
        "batch_size": {
            "values": [32, 64, 128, 256]
        },
        "epochs": {
            "values": [50, 100, 150, 200]
        }
    }
)
```

### Running a Sweep

```python
import wandb

def train():
    """Training function for sweep."""
    # Get hyperparameters from W&B
    config = wandb.config

    # Build model with sweep parameters
    model = build_model(
        hidden_layers=config.hidden_layers,
        dropout_rate=config.dropout_rate,
        learning_rate=config.learning_rate
    )

    # Train model
    for epoch in range(config.epochs):
        loss = train_epoch(model, config.batch_size)
        wandb.log({"loss": loss, "epoch": epoch})

    # Evaluate and log final metrics
    rmse, r2 = evaluate(model)
    wandb.log({"rmse": rmse, "r2_score": r2})

# Run sweep
sweep_manager.run_sweep(
    sweep_id=sweep_id,
    train_function=train,
    count=100  # Number of runs
)
```

### Getting Best Run

```python
best_run = sweep_manager.get_best_run(sweep_id)

print(f"Best run: {best_run['run_id']}")
print(f"Best config: {best_run['config']}")
print(f"Best metrics: {best_run['metrics']}")
```

### Sweep Methods

- **GRID**: Exhaustive search over all parameter combinations
- **RANDOM**: Random sampling from parameter distributions
- **BAYES**: Bayesian optimization (recommended for most cases)

```python
from greenlang.ml.mlops.wandb_integration import SweepMethod

# Grid search (for small parameter spaces)
sweep_manager.create_sweep(method=SweepMethod.GRID, ...)

# Random search (for exploration)
sweep_manager.create_sweep(method=SweepMethod.RANDOM, ...)

# Bayesian optimization (for efficiency)
sweep_manager.create_sweep(method=SweepMethod.BAYES, ...)
```

---

## Alerting and Notifications

### Setting Up Alerts

```python
from greenlang.ml.mlops.wandb_integration import WandBAlerting, AlertLevel

alerting = WandBAlerting()

# Alert when loss exceeds threshold
alerting.add_alert(
    name="high_loss",
    metric="loss",
    condition="above",
    threshold=0.5,
    level=AlertLevel.WARNING,
    slack_webhook="https://hooks.slack.com/services/...",
    cooldown_minutes=30
)

# Alert when accuracy drops
alerting.add_alert(
    name="low_accuracy",
    metric="accuracy",
    condition="below",
    threshold=0.9,
    level=AlertLevel.ERROR,
    email_recipients=["ml-team@company.com"],
    cooldown_minutes=60
)

# Alert for critical failures
alerting.add_alert(
    name="training_failure",
    metric="error_rate",
    condition="above",
    threshold=0.1,
    level=AlertLevel.CRITICAL,
    slack_webhook="https://hooks.slack.com/services/...",
    email_recipients=["on-call@company.com"],
    cooldown_minutes=5
)
```

### Checking Alerts

```python
# During training
for epoch in range(100):
    metrics = train_epoch()
    tracker.log_metrics(metrics, step=epoch)

    # Check alerts
    triggered = alerting.check_alerts(run_id, metrics)
    if triggered:
        print(f"Alerts triggered: {triggered}")
```

### Alert Levels

- `INFO`: Informational, no action required
- `WARNING`: Something to monitor
- `ERROR`: Needs attention
- `CRITICAL`: Immediate action required

---

## Report Generation

### Cross-Run Comparison

```python
from greenlang.ml.mlops.wandb_integration import WandBReportGenerator

report_gen = WandBReportGenerator()

# Compare multiple runs
comparison = report_gen.compare_runs(
    run_ids=["run_1abc", "run_2def", "run_3ghi"],
    metrics=["rmse", "r2_score", "mae"]
)

for run_id, data in comparison.items():
    print(f"\n{run_id} ({data['name']}):")
    print(f"  State: {data['state']}")
    print(f"  Metrics: {data['metrics']}")
```

### Best Model Selection

```python
# Select best model by minimizing RMSE
best = report_gen.select_best_model(
    run_ids=["run_1", "run_2", "run_3"],
    metric="rmse",
    goal="minimize"
)

print(f"Best run: {best['run_id']}")
print(f"Best RMSE: {best['value']}")
print(f"Best config: {best['config']}")
```

### Generate Summary Report

```python
# Generate markdown report
report = report_gen.generate_summary_report(
    run_ids=["run_1", "run_2", "run_3"],
    output_path="./reports/experiment_summary.md"
)

# Create W&B report (interactive)
report_url = report_gen.create_wandb_report(
    title="Fuel Model Comparison Q4 2024",
    run_ids=["run_1", "run_2", "run_3"],
    description="Comparison of fuel emission models"
)
```

---

## MLflow Integration Bridge

For teams using both W&B and MLflow:

```python
from greenlang.ml.mlops.wandb_integration import WandBMLflowBridge, AgentType

bridge = WandBMLflowBridge(
    mlflow_tracking_uri="http://mlflow.company.com:5000"
)

# Log to both platforms simultaneously
with bridge.dual_run(
    run_name="fuel_training",
    agent_type=AgentType.GL_008_FUEL,
    config={"learning_rate": 0.01}
) as (wandb_run, mlflow_run):

    # Train model
    model = train_model()

    # Log metrics to both platforms
    bridge.log_metrics({"rmse": 0.05, "r2": 0.95})

    # Log model to both platforms
    wandb_artifact, mlflow_uri = bridge.log_model(
        model,
        "fuel_emission_model"
    )

    print(f"W&B artifact: {wandb_artifact}")
    print(f"MLflow URI: {mlflow_uri}")
```

---

## Caching Strategy for Cost Reduction

The integration implements caching for 66% cost reduction:

### How It Works

1. **Prompt Caching**: Repeated configurations are cached to avoid redundant API calls
2. **Result Memoization**: Expensive computations are cached with TTL
3. **Local-First Storage**: Results stored locally before syncing

### Configuration

```python
config = WandBConfig(
    enable_caching=True,
    cache_dir="./wandb_cache",
    cache_ttl_hours=24  # Cache lifetime
)
```

### Cache Statistics

```python
# Get cache stats
stats = tracker._cache.get_stats()
print(f"Memory entries: {stats['memory_entries']}")
print(f"Disk entries: {stats['disk_entries']}")
print(f"Disk size: {stats['disk_size_mb']:.2f} MB")
```

### Manual Cache Management

```python
# Clear expired entries
cleared = tracker._cache.clear_expired()
print(f"Cleared {cleared} expired entries")
```

---

## Best Practices

### 1. Consistent Run Naming

```python
# Use agent type in run names
with tracker.init_run(
    run_name="v2.1_improved_features",
    agent_type=AgentType.GL_008_FUEL
):
    ...

# Resulting name: greenlang_GL-008-Fuel_v2.1_improved_features_20241207_143052
```

### 2. Comprehensive Tagging

```python
with tracker.init_run(
    run_name="production_training",
    agent_type=AgentType.GL_008_FUEL,
    tags=[
        "production",
        "natural-gas",
        "xgboost",
        "q4-2024",
        "team-alpha"
    ]
):
    ...
```

### 3. Data Provenance

Always track data provenance for regulatory compliance:

```python
import hashlib

# Compute data hash
with open("training_data.parquet", "rb") as f:
    data_hash = hashlib.sha256(f.read()).hexdigest()

with tracker.init_run(...):
    tracker.log_hyperparameters({
        "data_hash": data_hash,
        "data_version": "v3.2.1",
        "training_samples": 50000
    })
```

### 4. Model Versioning

```python
# Use semantic versioning for models
tracker.log_model(
    model,
    "fuel_emission_model_v2.1.0",
    aliases=["v2.1", "latest", "production-candidate"]
)
```

### 5. Group Related Runs

```python
# Use groups for related experiments
with tracker.init_run(
    run_name="experiment_1",
    group="fuel_model_hyperparameter_search",
    job_type="training"
):
    ...
```

### 6. Regular Cleanup

```python
# Clear expired cache entries periodically
if tracker._cache:
    tracker._cache.clear_expired()
```

---

## Dashboard Configuration

### Recommended Panels

1. **Training Progress**
   - Loss curve (line chart)
   - Accuracy over epochs (line chart)
   - Learning rate schedule (line chart)

2. **Performance Metrics**
   - RMSE comparison (bar chart)
   - R-squared scores (bar chart)
   - MAE distribution (histogram)

3. **Hyperparameter Analysis**
   - Parallel coordinates plot
   - Hyperparameter importance
   - Correlation matrix

4. **Resource Usage**
   - GPU utilization
   - Memory usage
   - Training time

### Creating Custom Dashboards

```python
# Access W&B API for dashboard creation
api = wandb.Api()
project = api.project("greenlang-process-heat")

# Get runs for dashboard
runs = api.runs(f"{entity}/{project}")
```

---

## Team Collaboration

### Setting Up Teams

1. Create a W&B team at https://wandb.ai/authorize
2. Configure entity in config:

```python
config = WandBConfig(
    entity="greenlang-ml-team",
    project="process-heat-agents"
)
```

### Access Control

- **Admin**: Full access to project settings
- **Member**: Can create runs and artifacts
- **Viewer**: Read-only access

### Sharing Results

```python
# Generate shareable report
report_url = report_gen.create_wandb_report(
    title="Q4 2024 Model Performance",
    run_ids=["run_1", "run_2", "run_3"]
)
print(f"Share this link: {report_url}")
```

---

## Troubleshooting

### Common Issues

#### API Key Not Found

```bash
# Check if API key is set
echo $WANDB_API_KEY

# Login manually
wandb login
```

#### Offline Mode Issues

```python
# Enable offline mode explicitly
config = WandBConfig(offline_mode=True)

# Sync offline runs later
# wandb sync ./wandb/offline-run-*
```

#### Large Artifact Upload Failures

```python
# For large artifacts, use chunked upload
tracker.log_artifact(
    "./large_model/",
    "large_model",
    artifact_type="model"
)
```

#### Run Not Finishing

```python
# Always use context manager or call finish
try:
    with tracker.init_run("my_run"):
        train()
except Exception as e:
    tracker.finish_run(exit_code=1)
    raise
```

### Debug Mode

```python
import logging
logging.getLogger("wandb").setLevel(logging.DEBUG)
```

### Getting Help

- W&B Documentation: https://docs.wandb.ai
- GreenLang Issues: https://github.com/greenlang/greenlang/issues
- Slack: #ml-platform channel

---

## API Reference

### WandBExperimentTracker

| Method | Description |
|--------|-------------|
| `init_run()` | Initialize a new run |
| `log_metrics()` | Log training/validation metrics |
| `log_hyperparameters()` | Log hyperparameter configurations |
| `log_model()` | Save and version models |
| `log_artifact()` | Track datasets and artifacts |
| `log_table()` | Log tabular data for analysis |
| `finish_run()` | Clean up and sync |

### WandBSweepManager

| Method | Description |
|--------|-------------|
| `create_sweep()` | Define sweep configuration |
| `run_sweep()` | Execute sweep runs |
| `get_best_run()` | Get best performing run |
| `stop_sweep()` | Stop running sweep |
| `get_sweep_status()` | Get sweep statistics |

### WandBAlerting

| Method | Description |
|--------|-------------|
| `add_alert()` | Configure metric alert |
| `check_alerts()` | Check metrics against alerts |
| `remove_alert()` | Remove alert configuration |
| `list_alerts()` | List configured alerts |

### WandBReportGenerator

| Method | Description |
|--------|-------------|
| `compare_runs()` | Compare multiple runs |
| `select_best_model()` | Select best model by metric |
| `generate_summary_report()` | Generate markdown report |
| `create_wandb_report()` | Create interactive W&B report |

---

## Version History

- **1.0.0** (2024-12-07): Initial release with full W&B integration
  - Experiment tracking for all 20 Process Heat agents
  - Hyperparameter sweeps (grid, random, Bayesian)
  - MLflow integration bridge
  - Caching for 66% cost reduction
  - SHA-256 provenance tracking
