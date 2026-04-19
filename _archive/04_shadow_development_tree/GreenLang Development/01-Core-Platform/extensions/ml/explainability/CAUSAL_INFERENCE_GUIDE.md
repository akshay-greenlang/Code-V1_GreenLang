# Causal Inference Module for GreenLang

## Overview

The causal inference module (`causal_inference.py`) provides DoWhy-based causal effect estimation for GreenLang's Process Heat agents. It enables understanding of cause-effect relationships in emissions and energy efficiency beyond simple correlations.

## Key Capabilities

### 1. **Causal Effect Estimation**
- **Average Treatment Effect (ATE)**: Overall impact of treatment on outcome
- **Confidence Intervals**: 95% CI with bootstrap-based estimation
- **Statistical Significance**: P-values and standard errors
- **Robustness Testing**: Refutation checks for estimate validity

### 2. **Causal Graph Construction**
- DAG (Directed Acyclic Graph) specification
- Confounding variable identification
- Instrumental variable support
- Effect modifier detection

### 3. **Counterfactual Analysis**
- "What-if" scenario predictions
- Individual Treatment Effect (ITE) calculation
- Sensitivity analysis across parameter ranges

### 4. **Process Heat Causal Models**
Pre-built models for common scenarios:
- **Excess Air → Efficiency**: How oxygen percentage affects boiler efficiency
- **Maintenance → Failure**: Preventive maintenance impact on failure probability
- **Load Changes → Emissions**: Steam load variability driving CO2 emissions
- **Fouling → Heat Transfer**: Fouling degradation of heat exchanger performance
- **Temperature → Corrosion**: Operating temperature effects on corrosion rates

## Architecture

### Core Components

```
CausalInference (Main Engine)
├── CausalInferenceConfig (Configuration)
├── CausalEffectResult (Output Model)
├── CounterfactualResult (What-if Output)
└── ProcessHeatCausalModels (Factory Methods)
```

### Identification Methods
- **Backdoor**: Adjust for confounders to identify effect
- **Frontdoor**: Use mediators to identify effect
- **Instrumental Variables**: Use exogenous instruments
- **Regression Discontinuity**: Exploit threshold effects

### Estimation Methods
- **Linear Regression**: Simple parametric approach
- **Propensity Score Matching**: Match treatment/control units
- **Propensity Score Weighting**: Weight for covariate balance
- **Instrumental Variable**: 2SLS with exogenous variables
- **Double ML**: Debiased machine learning estimator
- **Causal Forest**: Heterogeneous treatment effect estimation

### Refutation Tests
- **Random Common Cause**: Add fake confounder, should not change estimate
- **Placebo Treatment**: Replace treatment with random variable
- **Data Subset**: Re-estimate on random data subset
- **Bootstrap**: Resample data, check stability

## Usage Examples

### Basic Usage: Estimate Excess Air Impact

```python
import pandas as pd
from greenlang.ml.explainability.causal_inference import (
    ProcessHeatCausalModels,
    CausalInferenceConfig,
    EstimationMethod
)

# Load your boiler operation data
data = pd.read_csv("boiler_operations.csv")

# Create causal model using factory method
model = ProcessHeatCausalModels.excess_air_efficiency_model(data)

# Estimate the causal effect
result = model.estimate_causal_effect()

# Interpret results
print(f"ATE: {result.average_treatment_effect:.4f}")
print(f"95% CI: {result.confidence_interval}")
print(f"Is Robust: {result.is_robust}")
print(f"Confounders: {model.get_confounders()}")
```

### What-If Analysis: Excess Air Reduction

```python
# Current operating state
current_state = {
    "excess_air_ratio": 0.25,  # 25% excess oxygen
    "efficiency": 0.78,
    "fuel_type": "gas",
    "burner_age": 5,
    "ambient_temp": 15
}

# What-if: Reduce to 15% excess air?
target_o2 = 0.15

# Counterfactual prediction
counterfactual = model.estimate_counterfactual(current_state, target_o2)

# Results
efficiency_gain = counterfactual.counterfactual_outcome - current_state["efficiency"]
print(f"Current Efficiency: {current_state['efficiency']:.2%}")
print(f"Predicted Efficiency: {counterfactual.counterfactual_outcome:.2%}")
print(f"Efficiency Gain: {efficiency_gain:.2%}")
print(f"95% CI: {counterfactual.confidence_interval}")
```

### Maintenance Analysis with Propensity Score Matching

```python
from greenlang.ml.explainability.causal_inference import (
    CausalInferenceConfig,
    EstimationMethod,
    IdentificationMethod
)

# Custom configuration for maintenance analysis
config = CausalInferenceConfig(
    treatment="maintenance_frequency",
    outcome="failure_probability",
    common_causes=["equipment_age", "utilization"],
    identification_method=IdentificationMethod.BACKDOOR,
    estimation_method=EstimationMethod.PROPENSITY_SCORE_MATCHING,
    confidence_level=0.99
)

model = CausalInference(data, config)
result = model.estimate_causal_effect()

# Business interpretation
visits_per_year = data["maintenance_frequency"].mean()
baseline_failure = data["failure_probability"].mean()

additional_visit_impact = result.average_treatment_effect
new_failure_rate = baseline_failure + additional_visit_impact

print(f"Baseline failure rate: {baseline_failure:.2%}")
print(f"Effect of one additional visit: {additional_visit_impact:.4f}")
print(f"Expected failure rate with +1 visit: {new_failure_rate:.2%}")
```

### Sensitivity Analysis: Load-Emissions Relationship

```python
# Analyze across different load change scenarios
sensitivity_analysis = {
    "load_changes": [-30, -20, -10, 0, 10, 20, 30],
    "predicted_emissions": []
}

base_instance = {
    "steam_load_change": 0,
    "co2_emissions": 250,
    "demand_pattern": "peak",
    "weather_temp": 15,
    "fuel_type": "gas",
    "boiler_efficiency": 0.82
}

model = ProcessHeatCausalModels.load_changes_emissions_model(data)

for load_change in sensitivity_analysis["load_changes"]:
    cf = model.estimate_counterfactual(base_instance, load_change)
    sensitivity_analysis["predicted_emissions"].append(cf.counterfactual_outcome)

# Plot sensitivity curve
import matplotlib.pyplot as plt
plt.plot(sensitivity_analysis["load_changes"],
         sensitivity_analysis["predicted_emissions"])
plt.xlabel("Steam Load Change (%)")
plt.ylabel("CO2 Emissions (kg/h)")
plt.title("Emissions Sensitivity to Load Changes")
plt.show()
```

### Fouling Detection and Maintenance Planning

```python
model = ProcessHeatCausalModels.fouling_heat_transfer_model(data)
result = model.estimate_causal_effect()

# Current heat exchanger condition
current_fouling = 2.5  # mm of fouling
current_u = 950  # W/m2K

# How much would cleaning improve performance?
clean_u_prediction = current_u + (-current_fouling * result.average_treatment_effect)

# Determine if cleaning is justified
u_threshold = 1100  # Minimum acceptable U
improvement_needed = u_threshold - current_u
fouling_reduction_needed = -improvement_needed / result.average_treatment_effect

print(f"Current U coefficient: {current_u:.0f} W/m2K")
print(f"Effect of fouling: {result.average_treatment_effect:.1f} W/m2K per mm")
print(f"To reach {u_threshold} W/m2K, need to remove {fouling_reduction_needed:.1f} mm of fouling")
print(f"Clean U prediction: {clean_u_prediction:.0f} W/m2K")
```

## Process Heat Causal Models

### 1. Excess Air → Efficiency

**Causal Graph:**
```
fuel_type → excess_air_ratio → efficiency ← fuel_type
burner_age → excess_air_ratio → efficiency ← burner_age
ambient_temp → efficiency
```

**Variables:**
- Treatment: `excess_air_ratio` (0.15 - 0.35)
- Outcome: `efficiency` (0.70 - 0.92)
- Confounders: `fuel_type`, `burner_age`, `ambient_temp`

**Interpretation:**
- Negative ATE (efficiency decreases with excess air)
- Each 0.1 increase in excess air → ~3% efficiency loss
- Backdoor adjustment sufficient for identification

### 2. Maintenance → Failure

**Causal Graph:**
```
equipment_age → maintenance_frequency → failure_probability ← equipment_age
utilization → maintenance_frequency → failure_probability ← utilization
```

**Variables:**
- Treatment: `maintenance_frequency` (1-8 visits/year)
- Outcome: `failure_probability` (0.01-0.25)
- Confounders: `equipment_age`, `utilization`, `maintenance_cost`

**Interpretation:**
- Negative ATE (more maintenance reduces failure)
- ~2-3% failure reduction per additional visit
- Propensity score matching accounts for selection bias

### 3. Load Changes → Emissions

**Causal Graph:**
```
demand_pattern → steam_load_change → co2_emissions ← demand_pattern
weather → steam_load_change → co2_emissions ← weather
fuel_type → co2_emissions
boiler_efficiency → co2_emissions
```

**Variables:**
- Treatment: `steam_load_change` (-30% to +30%)
- Outcome: `co2_emissions` (50-300 kg/h)
- Confounders: `demand_pattern`, `weather_temp`, `fuel_type`, `boiler_efficiency`

**Interpretation:**
- Positive ATE (emissions increase with load)
- ~1.5 kg CO2/h per 1% load change
- Demand-driven increases in emissions

### 4. Fouling → Heat Transfer

**Causal Graph:**
```
water_quality → fouling → heat_transfer ← water_quality
operating_hours → fouling → heat_transfer ← operating_hours
temperature → fouling → heat_transfer ← temperature
```

**Variables:**
- Treatment: `fouling_mm` (0.5-3.5 mm)
- Outcome: `heat_transfer_coef` (700-1800 W/m2K)
- Confounders: `water_quality`, `operating_hours`, `temperature`, `fluid_velocity`

**Interpretation:**
- Negative ATE (fouling reduces heat transfer)
- ~150-200 W/m2K reduction per mm of fouling
- Critical for maintenance planning

### 5. Temperature → Corrosion

**Causal Graph:**
```
pressure → temperature → corrosion_rate ← pressure
steam_quality → temperature → corrosion_rate ← steam_quality
water_ph → corrosion_rate
```

**Variables:**
- Treatment: `outlet_temp_c` (75-100°C)
- Outcome: `corrosion_rate` (mm/year)
- Confounders: `pressure_bar`, `steam_quality`, `water_ph`

**Interpretation:**
- Temperature effects on corrosion rates
- Critical for material selection and replacement planning

## Data Requirements

### Minimum Dataset Size
- **Minimum**: 50 samples
- **Recommended**: 200-500 samples
- **Ideal**: 1000+ samples for robust estimation

### Variable Requirements

**Required:**
- Treatment variable (numeric or categorical)
- Outcome variable (numeric, continuous preferred)

**Highly Recommended:**
- Confounder variables (causes of both treatment and outcome)
- At least 3-5 confounders for robust identification

**Optional:**
- Instrumental variables (affects treatment, not outcome)
- Effect modifiers (treatment effect varies by value)

### Data Quality Guidelines

```python
# Check data quality before analysis
def validate_causal_data(data, config):
    """Validate data meets causal inference requirements."""

    # 1. Check for missing values
    missing_pct = data.isnull().sum() / len(data) * 100
    if (missing_pct > 5).any():
        print("Warning: >5% missing values found")

    # 2. Check for low variance
    for col in data.columns:
        if data[col].nunique() < 5:
            print(f"Warning: Low variance in {col}")

    # 3. Check for balance
    treatment_var = config.treatment
    if data[treatment_var].nunique() == 2:  # Binary treatment
        balance = data[treatment_var].value_counts() / len(data)
        if balance.min() < 0.2:
            print("Warning: Imbalanced treatment groups")

    return True
```

## Interpretation Guide

### Causal Effect Result

```python
result = model.estimate_causal_effect()

# result.average_treatment_effect: Main causal effect
# - Magnitude: Size of effect
# - Sign: Direction (positive/negative)

# result.confidence_interval: Uncertainty bounds
# - (lower, upper) at specified confidence level
# - Effect should be "real" if CI doesn't cross zero

# result.standard_error: Estimation precision
# - Smaller = more precise estimate
# - Used to construct confidence intervals

# result.p_value: Statistical significance
# - p < 0.05: Effect is statistically significant
# - p > 0.05: Cannot reject null (no effect)

# result.is_robust: Refutation test results
# - True: Estimate passes robustness checks
# - False: Results may be sensitive to assumptions

# result.refutation_results: Details of robustness checks
# - Each refutation method shows whether estimate held
# - Passed if direction/magnitude stable under perturbations
```

### Counterfactual Result

```python
cf = model.estimate_counterfactual(instance, treatment_value=X)

# cf.original_outcome: Actual outcome for this instance
# cf.counterfactual_outcome: Predicted outcome if treatment = X
# cf.individual_treatment_effect: Difference (ITE)
# cf.confidence_interval: Prediction uncertainty

# Example interpretation:
if cf.counterfactual_outcome > cf.original_outcome:
    print("Increasing treatment to {X} would improve outcome")
```

## Best Practices

### 1. Confounder Selection

```python
# DO: Think carefully about causal relationships
common_causes = [
    "equipment_age",      # Affects both maintenance and failure
    "operating_hours",    # Affects both fouling and heat transfer
    "fuel_type"          # Affects both excess air and efficiency
]

# DON'T: Include post-treatment variables
# These are affected by treatment, creating bias
bad_causes = [
    "maintenance_cost",  # Caused by maintenance_frequency
    "efficiency_loss",   # Caused by fouling
]
```

### 2. Identification Strategy

```python
# Choose method matching your causal graph
config = CausalInferenceConfig(
    treatment="T",
    outcome="Y",
    # Backdoor: When you can measure all confounders
    common_causes=["X1", "X2"],
    identification_method=IdentificationMethod.BACKDOOR,

    # OR Instrumental Variables: When confounders unmeasured
    instruments=["Z"],
    identification_method=IdentificationMethod.IV,
)
```

### 3. Estimation Method Selection

```python
# Match estimation method to your data:

# Small samples, few confounders:
EstimationMethod.LINEAR_REGRESSION

# Treatment assignment not random (selection bias):
EstimationMethod.PROPENSITY_SCORE_MATCHING

# Many confounders, high-dimensional:
EstimationMethod.DOUBLE_ML

# Heterogeneous effects (effect varies):
EstimationMethod.CAUSAL_FOREST
```

### 4. Robustness Checking

```python
# Always run refutation tests
result = model.estimate_causal_effect()

for method, details in result.refutation_results.items():
    if details.get("passed"):
        print(f"✓ {method}: passed")
    else:
        print(f"✗ {method}: failed - estimate may be unreliable")
```

### 5. Sensitivity Analysis

```python
# Vary key parameters to understand robustness
for conf_level in [0.90, 0.95, 0.99]:
    config = CausalInferenceConfig(..., confidence_level=conf_level)
    model = CausalInference(data, config)
    result = model.estimate_causal_effect()
    print(f"CI at {conf_level}: {result.confidence_interval}")

# If results stable across confidence levels, more robust
```

## Advanced Topics

### Heterogeneous Treatment Effects

```python
# Does effect vary by equipment type?
for equipment_type in ["gas_fired", "oil_fired"]:
    subset = data[data["equipment_type"] == equipment_type]
    model = ProcessHeatCausalModels.excess_air_efficiency_model(subset)
    result = model.estimate_causal_effect()
    print(f"{equipment_type}: ATE = {result.average_treatment_effect:.4f}")
```

### Mediation Analysis

```python
# How much of maintenance effect goes through labor vs parts?
# Would need mediator variables in data
config = CausalInferenceConfig(
    treatment="maintenance_frequency",
    outcome="failure_probability",
    common_causes=[...],
    effect_modifiers=["labor_hours", "parts_cost"]
)
```

### Sensitivity to Unmeasured Confounding

```python
# How strong would unmeasured confounder need to be
# to change causal conclusion?

result = model.estimate_causal_effect()

# Use refutation_random_common_cause to test
# If estimate stable with fake confounders added,
# more likely real
```

## Troubleshooting

### Issue: ATE suspiciously large or small

```python
# Check 1: Confounding bias
confounders = model.get_confounders()
if len(confounders) < 3:
    print("Warning: Few confounders, may have bias")

# Check 2: Measurement scale
print(f"Treatment range: {data[config.treatment].min()} to {data[config.treatment].max()}")
print(f"Outcome range: {data[config.outcome].min()} to {data[config.outcome].max()}")

# Check 3: Causal direction
# Ensure treatment temporally precedes outcome
```

### Issue: Confidence interval very wide

```python
# Check 1: Sample size
print(f"N samples: {len(data)}")
# Increase sample size if possible

# Check 2: Treatment variance
print(f"Treatment std dev: {data[config.treatment].std()}")
# Need sufficient treatment variation

# Check 3: Outcome noise
print(f"Outcome std dev: {data[config.outcome].std()}")
# High noise requires larger sample
```

### Issue: Refutation tests failing

```python
# Likely means estimate not robust
result = model.estimate_causal_effect()

if not result.is_robust:
    print("Estimate failed robustness checks")
    print("Options:")
    print("1. Add more confounders to config")
    print("2. Check for measurement error")
    print("3. Try different estimation method")
    print("4. Increase sample size")
```

## Performance Characteristics

- **Initialization**: ~10-50ms
- **Causal graph construction**: ~1-5ms
- **Effect estimation**: 100-5000ms (depends on method)
- **Counterfactual prediction**: 10-50ms
- **Refutation tests**: 1-30 seconds (depends on n_bootstrap)

## Dependencies

```
dowhy >= 0.11.0          # Core causal inference library
numpy >= 1.19.0          # Numerical computations
pandas >= 1.0.0          # Data handling
scikit-learn >= 0.24.0   # ML algorithms for estimation
networkx >= 2.5          # Graph operations
```

## References

- Pearl, J. (2009). Causality: Models, Reasoning, and Inference (2nd ed.)
- Rotnitzky, A., et al. (2007). Inverse probability weighting
- Angrist, J. D., & Pischke, J.-S. (2008). Mostly Harmless Econometrics
- Athey, S., & Wager, S. (2019). Generalized Random Forests
- DoWhy documentation: https://microsoft.github.io/dowhy/

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review example code in `process_heat_causal_examples.py`
3. Run unit tests: `pytest tests/unit/test_causal_inference.py`
4. Consult DoWhy documentation for advanced topics
