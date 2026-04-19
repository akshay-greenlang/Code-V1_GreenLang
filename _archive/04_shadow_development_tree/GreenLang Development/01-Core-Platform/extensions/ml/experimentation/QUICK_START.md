# A/B Testing Framework - Quick Start Guide

## Installation

Already included in GreenLang. No additional dependencies needed (beyond numpy/scipy).

## Basic Usage (5 minutes)

### 1. Create Experiment

```python
from greenlang.ml.experimentation import ABTestManager

manager = ABTestManager()

exp_id = manager.create_experiment(
    name="boiler_model_v2",
    variants=["baseline", "optimized"],
    traffic_split={"baseline": 0.5, "optimized": 0.5}
)
```

### 2. Assign Variants & Record Metrics

```python
# For each observation
variant = manager.assign_variant(user_id, exp_id)
metric_value = compute_efficiency(variant)
manager.record_metric(exp_id, variant, "efficiency", metric_value)
```

### 3. Analyze Results

```python
result = manager.analyze_results(exp_id)

print(f"Winner: {result.winner}")
print(f"P-value: {result.p_value:.4f}")
print(f"Effect size: {result.effect_size:.4f}")
print(f"Significant: {result.is_significant}")
```

## API Reference

### ABTestManager

```python
# Create experiment
exp_id = manager.create_experiment(
    name: str,
    variants: List[str],
    traffic_split: Optional[Dict[str, float]] = None,
    metric_type: MetricType = MetricType.CONTINUOUS,
    test_type: TestType = TestType.WELCH_T
) -> str

# Assign variant (deterministic)
variant = manager.assign_variant(
    user_id: str,
    experiment_id: str
) -> str

# Record metric
manager.record_metric(
    experiment_id: str,
    variant: str,
    metric_name: str,
    value: float
)

# Record conversion
manager.record_conversion(
    experiment_id: str,
    variant: str,
    success: bool
)

# Get analysis results
result = manager.analyze_results(
    experiment_id: str
) -> ExperimentResult

# Get winning variant (if significant)
winner = manager.get_winner(
    experiment_id: str
) -> Optional[str]

# Get experiment status
status = manager.get_status(
    experiment_id: str
) -> Dict[str, Any]

# Export Prometheus metrics
metrics = manager.export_prometheus_metrics(
    experiment_id: str
) -> str
```

### StatisticalAnalyzer

```python
# T-test
t_stat, p_value = StatisticalAnalyzer.welch_t_test(
    group_a: List[float],
    group_b: List[float]
)

# Chi-squared test
chi2, p_value = StatisticalAnalyzer.chi_squared_test(
    group_a: Tuple[int, int],  # (successes, total)
    group_b: Tuple[int, int]
)

# Bayesian probability
prob_b_wins = StatisticalAnalyzer.bayesian_a_b(
    group_a: List[float],
    group_b: List[float]
) -> float  # 0.0 to 1.0

# Effect size
d = StatisticalAnalyzer.cohen_d(
    group_a: List[float],
    group_b: List[float]
) -> float

# Sample size calculator
n = StatisticalAnalyzer.calculate_sample_size(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> int
```

## Common Patterns

### Pattern 1: Compare Two Models

```python
# Setup
manager = ABTestManager()
exp_id = manager.create_experiment(
    name="model_comparison",
    variants=["model_v1", "model_v2"],
    metric_type=MetricType.CONTINUOUS,
    test_type=TestType.WELCH_T
)

# Run experiment
for i in range(200):
    variant = manager.assign_variant(f"sample_{i}", exp_id)
    prediction = model_v1(data) if variant == "model_v1" else model_v2(data)
    error = abs(prediction - ground_truth)
    manager.record_metric(exp_id, variant, "prediction_error", error)

# Decide
result = manager.analyze_results(exp_id)
if result.is_significant and result.winner == "model_v2":
    deploy_model_v2()
```

### Pattern 2: Compare Strategies (3+ variants)

```python
manager = ABTestManager()
exp_id = manager.create_experiment(
    name="strategy_comparison",
    variants=["strategy_a", "strategy_b", "strategy_c"],
    traffic_split={"strategy_a": 0.34, "strategy_b": 0.33, "strategy_c": 0.33}
)

# Run and analyze
for observation in observations:
    variant = manager.assign_variant(observation.id, exp_id)
    outcome = execute_strategy(variant, observation)
    manager.record_metric(exp_id, variant, "outcome", outcome)

result = manager.analyze_results(exp_id)
```

### Pattern 3: Conversion Metrics

```python
manager = ABTestManager()
exp_id = manager.create_experiment(
    name="conversion_test",
    variants=["control", "treatment"],
    metric_type=MetricType.CONVERSION,
    test_type=TestType.CHI_SQUARED
)

# Record conversions
for user in users:
    variant = manager.assign_variant(user.id, exp_id)
    success = check_conversion(user, variant)
    manager.record_conversion(exp_id, variant, success)

result = manager.analyze_results(exp_id)
```

## Interpretation Guide

### P-Value

- **p < 0.05**: Statistically significant (reject null hypothesis)
- **p >= 0.05**: Not significant (insufficient evidence)
- **p < 0.01**: Highly significant
- **p < 0.001**: Very highly significant

### Effect Size (Cohen's d)

- **0.2**: Small effect
- **0.5**: Medium effect
- **0.8**: Large effect
- **1.0+**: Very large effect

### Power

- **Power = 0.80**: 80% chance of detecting real effect (default)
- **Power < 0.80**: May miss real effects
- **More samples → Higher power**

### Sample Size Planning

```python
# How many samples do I need?
effect_size = 0.5  # Expected difference

n = StatisticalAnalyzer.calculate_sample_size(
    effect_size=effect_size,
    alpha=0.05,
    power=0.80
)

# Result for effect_size=0.5:
# n = 64 per variant
# Total = 128 observations
```

## REST API (FastAPI)

```bash
# Create experiment
curl -X POST http://localhost:8000/api/v1/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test",
    "variants": ["a", "b"],
    "metric_type": "continuous",
    "test_type": "welch_t"
  }'

# Get variant assignment
curl "http://localhost:8000/api/v1/experiments/abc123/assign?user_id=user_456"

# Record metric
curl -X POST http://localhost:8000/api/v1/experiments/abc123/metrics \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_id": "abc123",
    "variant": "a",
    "metric_name": "efficiency",
    "value": 0.92
  }'

# Get results
curl "http://localhost:8000/api/v1/experiments/abc123/results"

# Get status
curl "http://localhost:8000/api/v1/experiments/abc123/status"

# Get Prometheus metrics
curl "http://localhost:8000/api/v1/experiments/abc123/prometheus"
```

## Examples

Run comprehensive examples:

```bash
python -m greenlang.ml.experimentation.examples
```

Includes:
1. Boiler efficiency model comparison
2. Combustion diagnostics strategy comparison
3. Steam distribution optimization (3 variants)
4. Deterministic assignment verification
5. Sample size planning guide

## Monitoring with Prometheus

Export and visualize metrics:

```python
metrics = manager.export_prometheus_metrics(exp_id)
# Send to Prometheus scrape endpoint
```

Metrics include:
- `ab_test_p_value` - Statistical significance
- `ab_test_effect_size` - Cohen's d
- `ab_test_sample_count` - Total observations
- `ab_test_variant_mean` - Per-variant average
- `ab_test_variant_std` - Per-variant standard deviation

## Provenance & Audit Trail

All results include SHA-256 provenance hash:

```python
result = manager.analyze_results(exp_id)
print(result.provenance_hash)
# "a4c8e9f2b1d3c7a9e5f2b8d4c1a3e7f9..."

# Hash is deterministic - same inputs produce same hash
# Useful for audit trails and reproducibility
```

## Troubleshooting

### "Insufficient samples" error
- Need at least 2-10 samples per variant
- For more statistical power, aim for 50+ samples per variant
- Use `calculate_sample_size()` to determine duration

### Experiment not significant but effect looks large
- Small sample size reduces power
- Add more observations to test
- Check if true effect size is smaller than expected

### Different results on same data
- Assignment is deterministic (check user_ids match)
- Check if metrics are being recorded correctly
- Verify variant names match exactly

### Want to restart experiment
- Create new experiment with different name
- Assignments are cached per (user_id, exp_id) pair
- Same user_id in new exp_id will get different assignment

## Performance

For typical Process Heat applications:
- Variant assignment: < 1ms (O(1))
- Metric recording: < 1ms (O(1))
- Analysis: 10-100ms for 100-1000 samples
- Prometheus export: 5-10ms

## Next Steps

1. Run examples: `python -m greenlang.ml.experimentation.examples`
2. Read full documentation: `greenlang/ml/experimentation/README.md`
3. Check unit tests: `pytest tests/unit/test_ab_testing_experimentation.py -v`
4. Integrate with your Process Heat agents
5. Set up REST API endpoints with FastAPI
6. Configure Prometheus monitoring

## Support

For issues or questions:
- Check README.md for detailed documentation
- Review examples.py for usage patterns
- Check tests for expected behavior
- Refer to statistical references in README
