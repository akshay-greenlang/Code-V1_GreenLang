# Getting Started with GreenLang

This guide will help you get started with GreenLang, a secure pipeline orchestration platform with default-deny security and capability-based access control.

## Prerequisites

- GreenLang CLI installed ([Installation Guide](installation.md))
- Basic understanding of YAML
- Optional: Docker for containerized execution

## Quick Start

### 1. Initialize a New Project

```bash
# Create a new GreenLang project
gl init my-project

# Navigate to project directory
cd my-project
```

This creates:
- `pipeline.yaml` - Pipeline configuration
- `pack.yaml` - Pack manifest
- `.greenlang/` - Local configuration

### 2. Create Your First Pipeline

Create a file named `pipeline.yaml`:

```yaml
version: "1.0"
kind: Pipeline
metadata:
  name: hello-world
  description: My first GreenLang pipeline

# Security capabilities (default-deny)
capabilities:
  net:
    allow: false  # No network access
  fs:
    allow: true   # Limited filesystem access
    read_paths: ["./data"]
    write_paths: ["./output"]

# Pipeline steps
steps:
  - id: hello
    name: Say Hello
    agent: Echo
    with:
      message: "Hello, GreenLang!"
    outputs:
      result: hello.txt

# Pipeline outputs
outputs:
  greeting:
    type: file
    path: hello.txt
```

### 3. Run the Pipeline

```bash
# Run locally (pipeline only)
gl run pipeline.yaml

# Run with explicit input file and output directory
gl run pipeline.yaml input.json out

# Run CBAM MVP flow (from monorepo root)
gl run cbam cbam.yaml imports.csv out
```

### 4. View Results

```bash
# Check generated output files
ls out/

# CBAM artifacts are written under:
# out/cbam_report.xml
# out/report_summary.xlsx
# out/audit/
```

## Core Concepts

### Pipelines

Pipelines define workflows as a series of steps:

```yaml
steps:
  - id: fetch-data
    name: Fetch Data
    agent: HTTPFetcher
    with:
      url: https://api.example.com/data

  - id: process
    name: Process Data
    depends_on: [fetch-data]
    agent: DataProcessor
    with:
      input: ${{ steps.fetch-data.outputs.data }}
```

### Security Capabilities

GreenLang enforces default-deny security:

```yaml
capabilities:
  # Network access (default: false)
  net:
    allow: true
    egress_allowlist:
      - api.example.com
      - storage.googleapis.com

  # Filesystem access (default: false)
  fs:
    allow: true
    read_paths: ["./data", "/tmp/cache"]
    write_paths: ["./output"]

  # Subprocess execution (default: false)
  subprocess:
    allow: false

  # Clock/time operations (default: false)
  clock:
    allow: true
```

### Packs

Packs are reusable components:

```bash
# Install a pack
gl pack install greenlang/weather-forecast

# List installed packs
gl pack list

# Use in pipeline
steps:
  - id: forecast
    pack: greenlang/weather-forecast
    with:
      location: "San Francisco"
```

## Working with Data

### Input Parameters

```yaml
inputs:
  params:
    environment:
      type: string
      default: "production"
    max_records:
      type: integer
      default: 1000

steps:
  - id: process
    agent: DataProcessor
    with:
      env: ${{ inputs.params.environment }}
      limit: ${{ inputs.params.max_records }}
```

Run with an input file:

```bash
gl run pipeline.yaml input.json out
```

### File Handling

```yaml
steps:
  - id: read-csv
    agent: FileReader
    with:
      path: "./data/input.csv"
    outputs:
      data: parsed_data.json

  - id: transform
    depends_on: [read-csv]
    agent: DataTransformer
    with:
      input: ${{ steps.read-csv.outputs.data }}
    outputs:
      result: transformed.json
```

## Security Best Practices

### 1. Use Minimal Capabilities

Only request capabilities you need:

```yaml
# BAD: Too permissive
capabilities:
  net:
    allow: true  # Full network access
  fs:
    allow: true  # Full filesystem access

# GOOD: Minimal required
capabilities:
  net:
    allow: true
    egress_allowlist: ["api.myservice.com"]
  fs:
    allow: true
    read_paths: ["./config.json"]
    write_paths: ["./output"]
```

### 2. Verify Pack Signatures

```bash
# Verify before installing
gl pack verify greenlang/ml-pipeline

# Install only signed packs
gl pack install greenlang/ml-pipeline --require-signature
```

### 3. Use Secrets Safely

```yaml
# Reference secrets (never hardcode)
steps:
  - id: api-call
    agent: HTTPClient
    with:
      url: https://api.example.com
      auth:
        type: bearer
        token: ${{ secrets.API_TOKEN }}
```

Set secrets via environment:

```bash
export GL_SECRET_API_TOKEN="your-token-here"
gl run pipeline.yaml
```

## Advanced Features

### Conditional Execution

```yaml
steps:
  - id: check-condition
    agent: Evaluator
    with:
      expression: "${{ inputs.params.environment }} == 'production'"

  - id: production-only
    when: ${{ steps.check-condition.outputs.result }}
    agent: ProductionTask
```

### Parallel Execution

```yaml
steps:
  - id: task-a
    agent: TaskA

  - id: task-b
    agent: TaskB

  - id: task-c
    agent: TaskC

  - id: combine
    depends_on: [task-a, task-b, task-c]
    agent: Combiner
```

### Error Handling

```yaml
steps:
  - id: risky-operation
    agent: RiskyTask
    retry:
      max_attempts: 3
      backoff: exponential
    on_failure:
      - id: cleanup
        agent: CleanupTask
```

## Debugging

### Enable Debug Logging

```bash
# Run and capture console logs
gl run pipeline.yaml input.json out
```

### Inspect Execution

```bash
# Inspect generated artifacts
ls out/
ls out/audit/
```

## Common Patterns

### ETL Pipeline

```yaml
name: etl-pipeline
steps:
  - id: extract
    agent: DatabaseExtractor
    with:
      query: "SELECT * FROM users WHERE created > '2024-01-01'"

  - id: transform
    agent: DataTransformer
    with:
      input: ${{ steps.extract.outputs.data }}
      operations: ["normalize", "validate", "enrich"]

  - id: load
    agent: DataLoader
    with:
      data: ${{ steps.transform.outputs.result }}
      destination: "warehouse"
```

### ML Pipeline

```yaml
name: ml-training
steps:
  - id: prepare-data
    agent: DataPrep
    with:
      dataset: "./data/training.csv"

  - id: train-model
    agent: MLTrainer
    with:
      data: ${{ steps.prepare-data.outputs.processed }}
      algorithm: "random-forest"
      hyperparameters:
        n_estimators: 100
        max_depth: 10

  - id: evaluate
    agent: ModelEvaluator
    with:
      model: ${{ steps.train-model.outputs.model }}
      test_data: "./data/test.csv"
```

## Next Steps

- Explore [Pipeline Development](pipelines.md)
- Learn about [Pack System](packs.md)
- Review [Security Model](SECURITY_MODEL.md)
- Check out [Examples](../examples/README.md)

## Getting Help

```bash
# Built-in help
gl --help
gl run --help

# Check system status
gl doctor

# Community
# - GitHub: https://github.com/greenlang/greenlang
# - Docs: https://docs.greenlang.ai
```