# Complex Pipeline Examples

This directory contains advanced GreenLang pipeline examples that demonstrate sophisticated workflow patterns, error handling strategies, and parallel execution techniques.

## Examples Overview

### 1. **Main Complex Pipeline** (`gl.yaml`)
A comprehensive multi-facility carbon analysis pipeline showcasing advanced GreenLang features.

### 2. **Error Recovery Example** (`error-recovery-example.yaml`)
Demonstrates various error handling and recovery patterns.

### 3. **Parallel Execution Example** (`parallel-execution-example.yaml`)
Shows different parallelization strategies and coordination patterns.

---

## Main Complex Pipeline (`gl.yaml`)

### Features Demonstrated

#### **Multi-Step Workflow with Complex Dependencies**
```yaml
depends_on: [preprocess-facilities, load-regional-factors]
```

#### **Conditional Execution**
```yaml
conditional:
  condition: "${inputs.energy_consumption.renewable_kwh} > 0"
  on_skip:
    renewable_offset_tons: 0
    renewable_percentage: 0
```

#### **Parallel Processing**
```yaml
parallel: true
parallel_group: "facility_processing"
```

#### **Advanced Input Schema**
- Nested object validation
- Array input handling
- Enum constraints
- Default values and optional fields

#### **Comprehensive Error Handling**
- Multiple error strategies per step
- Fallback mechanisms
- Quality gates and validation

### Sample Usage

```bash
# Run with sample multi-facility data
gl pipeline run examples/pipelines/complex/gl.yaml \
  --input-file multi_facility_input.json \
  --config analysis_config.yaml
```

**Sample Input** (`multi_facility_input.json`):
```json
{
  "facilities": [
    {
      "id": "facility_001",
      "name": "Manufacturing Plant A",
      "location": "IN-North",
      "building_area_sqft": 250000,
      "facility_type": "manufacturing",
      "energy_data": {
        "electricity_kwh": 3500000,
        "natural_gas_mmbtu": 25000,
        "renewable_kwh": 750000
      }
    },
    {
      "id": "facility_002",
      "name": "Office Complex B",
      "location": "IN-South",
      "building_area_sqft": 100000,
      "facility_type": "office",
      "energy_data": {
        "electricity_kwh": 800000,
        "natural_gas_mmbtu": 2000,
        "renewable_kwh": 150000
      }
    }
  ],
  "analysis_config": {
    "include_scope3": true,
    "carbon_pricing": true,
    "sensitivity_analysis": true
  }
}
```

### Key Outputs

```yaml
# Portfolio-level results
total_portfolio_emissions: 2847.3  # tons CO2e
facility_count: 15
emissions_per_sqft: 0.0142

# Analysis features
renewable_offset_tons: 456.2
carbon_cost_current_usd: 241421.50
sensitivity_results: { ... }

# Quality metrics
data_quality_score: 0.94
qa_check_passed: true
```

---

## Error Recovery Example

### Error Handling Patterns Demonstrated

#### **1. Stop on Critical Failure**
```yaml
error_handling:
  on_failure: stop
  message: "Critical system requirements not met"
```

#### **2. Retry with Exponential Backoff**
```yaml
error_handling:
  on_failure: retry
  retry_count: 3
  retry_delay: 5
  backoff_strategy: "exponential"
  backoff_multiplier: 2
```

#### **3. Fallback to Alternative Data**
```yaml
error_handling:
  on_failure: use_fallback
  fallback_action: load_default_dataset
  fallback_inputs:
    dataset_name: "emission_factors_conservative"
```

#### **4. Continue with Default Outputs**
```yaml
error_handling:
  on_failure: continue
  default_outputs:
    enriched_data: "${steps.load-data.outputs.dataset}"
    enrichment_quality: "base_only"
```

#### **5. Circuit Breaker Pattern**
```yaml
circuit_breaker:
  failure_threshold: 5
  recovery_timeout: 300
  half_open_max_calls: 3
```

#### **6. Partial Success Handling**
```yaml
error_handling:
  on_failure: continue
  partial_success: true
  minimum_success_rate: 0.5
```

### Usage Example

```bash
# Test error handling with unreliable data source
gl pipeline run examples/pipelines/complex/error-recovery-example.yaml \
  --input unreliable_data_source="https://unstable-api.example.com/data" \
  --input enable_fallbacks=true
```

---

## Parallel Execution Example

### Parallelization Strategies

#### **1. Independent Parallel Tasks**
```yaml
# Multiple steps with same parallel_group execute simultaneously
parallel: true
parallel_group: "regional_factors"
```

#### **2. Data Partitioning for Parallel Processing**
```yaml
- name: process-partition-1
  parallel: true
  parallel_group: "facility_processing"
  resource_allocation:
    cpu_weight: 2
    memory_mb: 1024
```

#### **3. Fan-out/Fan-in Pattern**
```yaml
# Fan-out to multiple parallel branches
depends_on: [partition-facilities]
parallel: true

# Fan-in synchronization point
depends_on: [process-partition-1, process-partition-2, process-partition-3, process-partition-4]
```

#### **4. Resource-Aware Parallel Execution**
```yaml
execution:
  max_parallel_steps: 8
  parallel_execution_strategy: "optimal_resource_allocation"
  resource_management:
    cpu_allocation: "dynamic"
    memory_allocation: "shared_pool"
```

### Usage Example

```bash
# Run with large dataset for parallel processing demonstration
gl pipeline run examples/pipelines/complex/parallel-execution-example.yaml \
  --input-file large_facility_dataset.json \
  --config parallel_config.yaml \
  --parallel-workers 8
```

---

## Advanced Features Demonstrated

### 1. **Complex Data Flow Patterns**

#### Reference Usage
```yaml
# Cross-step references
total_emissions: "${steps.calculate-scope1.outputs.total_co2e_tons}"

# Array aggregation
partition_results:
  - "${steps.process-partition-1.outputs.scope1_data}"
  - "${steps.process-partition-2.outputs.scope1_data}"

# Conditional references
renewable_data: "${steps.renewable-analysis.outputs}"
```

#### System Variables
```yaml
calculation_timestamp: "${system.timestamp}"
execution_log: "${system.execution_log}"
pipeline_version: "${version}"
```

### 2. **Advanced Control Flow**

#### Conditional Steps
```yaml
conditional:
  condition: "${inputs.analysis_config.include_scope3} == true"
  on_skip:
    scope3_emissions_tons: 0
```

#### Dynamic Configuration
```yaml
# Configuration driven by input parameters
batch_size: "${inputs.processing_mode == 'fast' ? 100 : 25}"
timeout: "${inputs.processing_mode == 'thorough' ? '30m' : '10m'}"
```

### 3. **Quality Assurance Integration**

#### Multi-level Validation
```yaml
- name: preprocess-facilities     # Input validation
- name: validate-results         # Intermediate validation
- name: quality-assurance-check  # Final QA
```

#### Quality Gates
```yaml
quality_criteria:
  data_completeness_min: 0.95
  calculation_accuracy_threshold: 0.99
  outlier_count_max: 3
```

### 4. **Performance Optimization**

#### Caching
```yaml
cache_duration: "12h"
```

#### Resource Management
```yaml
resource_limits:
  memory_mb: 4096
  cpu_cores: 4
```

#### Timeouts and Checkpointing
```yaml
execution:
  timeout: "45m"
  checkpoint_interval: "5m"
```

---

## Best Practices Illustrated

### 1. **Error Resilience**
- Multiple fallback strategies
- Graceful degradation patterns
- Comprehensive error reporting
- Recovery mechanism integration

### 2. **Performance Optimization**
- Strategic parallelization
- Resource-aware execution
- Efficient data partitioning
- Caching for expensive operations

### 3. **Data Quality Management**
- Multi-stage validation
- Quality scoring and gates
- Outlier detection and handling
- Audit trail generation

### 4. **Maintainability**
- Clear step naming and documentation
- Modular design patterns
- Configuration externalization
- Comprehensive output metadata

---

## File Structure

```
examples/pipelines/complex/
├── gl.yaml                          # Main complex pipeline
├── error-recovery-example.yaml      # Error handling patterns
├── parallel-execution-example.yaml  # Parallel execution showcase
├── README.md                        # This documentation
├── sample-inputs/                   # Sample input files
│   ├── multi_facility_input.json
│   ├── large_facility_dataset.json
│   └── analysis_config.yaml
└── sample-outputs/                  # Expected output examples
    ├── portfolio_summary.json
    ├── executive_dashboard.html
    └── error_analysis_report.html
```

---

## Learning Objectives

After working with these examples, you'll understand:

1. **Complex Workflow Design**: Multi-step pipelines with sophisticated dependencies
2. **Error Handling Mastery**: Various failure recovery strategies and patterns
3. **Parallel Processing**: Efficient parallelization and resource management
4. **Data Flow Management**: Advanced reference patterns and data routing
5. **Quality Assurance**: Comprehensive validation and quality control
6. **Performance Optimization**: Caching, resource allocation, and execution tuning
7. **Enterprise Patterns**: Scalable architectures for production workflows

---

## Related Examples

- [Minimal Pipeline](../minimal/) - Basic pipeline structure
- [Advanced Pack](../../packs/advanced/) - Enterprise pack features
- [SDK Examples](../../sdk/) - Programmatic pipeline execution

---

## Troubleshooting

### Common Issues

1. **Memory Limitations**: Reduce `max_parallel_steps` or `batch_size`
2. **Timeout Errors**: Increase pipeline or step-level timeouts
3. **Data Quality Failures**: Check input validation and quality thresholds
4. **Parallel Execution Issues**: Verify resource allocation and dependencies

### Debug Commands

```bash
# Run with verbose logging
gl pipeline run gl.yaml --verbose --log-level debug

# Validate pipeline structure
gl pipeline validate gl.yaml --strict

# Check resource requirements
gl pipeline analyze gl.yaml --resource-estimation
```