# Methodologies and Uncertainty Catalog

**Version:** 1.0.0
**Date:** 2025-10-30
**Status:** Production Ready

## Overview

The Methodologies module provides scientific methodologies for emissions calculations including data quality assessment, uncertainty quantification, and Monte Carlo simulation. This is a core component of the GL-VCCI Scope 3 Carbon Platform (Phase 2, Week 3-4).

## Key Features

- **ILCD Pedigree Matrix**: Data quality assessment across 5 dimensions
- **Monte Carlo Simulation**: 10,000 iterations in <1 second with multiple distributions
- **DQI Calculator**: Composite Data Quality Index (0-100 scale)
- **Uncertainty Quantification**: Factor-specific and category-based uncertainties
- **Analytical Propagation**: Fast Taylor series approximation
- **GWP Constants**: IPCC AR5 and AR6 values

## Components

### 1. Pedigree Matrix (`pedigree_matrix.py`)

Implements the ILCD Pedigree Matrix for data quality assessment.

**5 Dimensions:**
- Reliability (1-5): Verification and source quality
- Completeness (1-5): Sample size and representativeness
- Temporal (1-5): Age of data
- Geographical (1-5): Geographic representativeness
- Technological (1-5): Technology representativeness

**Example:**
```python
from services.methodologies import PedigreeScore, PedigreeMatrixEvaluator

# Create pedigree score
pedigree = PedigreeScore(
    reliability=1,      # Verified measurements
    completeness=2,     # Good sample size
    temporal=1,         # <3 years old
    geographical=2,     # Similar region
    technological=1     # Same technology
)

# Calculate uncertainty
evaluator = PedigreeMatrixEvaluator()
uncertainty_result = evaluator.pedigree_to_uncertainty_result(
    mean=1000.0,
    pedigree=pedigree,
    base_uncertainty=0.1
)

print(f"Mean: {uncertainty_result.mean:.2f} kg CO2e")
print(f"Std Dev: {uncertainty_result.std_dev:.2f}")
print(f"95% CI: [{uncertainty_result.confidence_95_lower:.2f}, "
      f"{uncertainty_result.confidence_95_upper:.2f}]")
```

### 2. Monte Carlo Simulator (`monte_carlo.py`)

High-performance Monte Carlo engine for uncertainty propagation.

**Supported Distributions:**
- Normal
- Lognormal (recommended for emissions)
- Uniform
- Triangular

**Example:**
```python
from services.methodologies import MonteCarloSimulator

simulator = MonteCarloSimulator(seed=42)

# Simple emission calculation: E = Activity × Factor
result = simulator.simple_propagation(
    activity_data=1000.0,
    activity_uncertainty=0.1,       # 10% uncertainty
    emission_factor=2.5,
    factor_uncertainty=0.15,        # 15% uncertainty
    iterations=10000
)

print(f"Emission: {result.mean:.2f} ± {result.std_dev:.2f} kg CO2e")
print(f"p5: {result.p5:.2f}, p50: {result.p50:.2f}, p95: {result.p95:.2f}")
print(f"CV: {result.coefficient_of_variation:.2%}")
print(f"Computation time: {result.computation_time:.3f}s")

# Sensitivity analysis
print("\nTop contributors:")
for param in result.top_contributors:
    sensitivity = result.sensitivity_indices[param]
    print(f"  {param}: {sensitivity:.2%}")
```

### 3. DQI Calculator (`dqi_calculator.py`)

Calculates composite Data Quality Index (0-100 scale).

**Components:**
- Pedigree matrix scores (50% weight)
- Factor source quality (30% weight)
- Data tier (20% weight)

**Example:**
```python
from services.methodologies import DQICalculator, PedigreeScore

calculator = DQICalculator()

# Create pedigree score
pedigree = PedigreeScore(
    reliability=1, completeness=2, temporal=1,
    geographical=2, technological=1
)

# Calculate DQI
dqi = calculator.calculate_dqi(
    pedigree_score=pedigree,
    factor_source="ecoinvent",
    data_tier=1
)

print(f"DQI Score: {dqi.score:.2f}")
print(f"Quality Label: {dqi.quality_label}")
print(f"Components:")
print(f"  Pedigree: {dqi.pedigree_contribution:.2f}")
print(f"  Source: {dqi.source_contribution:.2f}")
print(f"  Tier: {dqi.tier_contribution:.2f}")

# Generate report with recommendations
report = calculator.generate_dqi_report(dqi)
print("\nRecommendations:")
for rec in report["recommendations"]:
    print(f"  - {rec}")
```

### 4. Uncertainty Quantifier (`uncertainty.py`)

Comprehensive uncertainty quantification and propagation.

**Methods:**
- Category-based defaults
- Tier multipliers
- Pedigree-based
- Custom uncertainties

**Example:**
```python
from services.methodologies import UncertaintyQuantifier

quantifier = UncertaintyQuantifier()

# Quantify uncertainty
result = quantifier.quantify_uncertainty(
    mean=1000.0,
    category="electricity",
    tier=2
)

print(f"Mean: {result.mean:.2f}")
print(f"Uncertainty: {result.relative_std_dev:.2%}")
print(f"95% CI: [{result.confidence_95_lower:.2f}, {result.confidence_95_upper:.2f}]")

# Propagate uncertainty through calculation
emission = quantifier.propagate_simple(
    activity_mean=1000.0,
    activity_uncertainty=0.1,
    factor_mean=2.5,
    factor_uncertainty=0.15,
    method="analytical"  # or "monte_carlo"
)

print(f"\nEmission: {emission.mean:.2f} ± {emission.std_dev:.2f}")
```

## Data Models

### PedigreeScore
```python
PedigreeScore(
    reliability: int (1-5),
    completeness: int (1-5),
    temporal: int (1-5),
    geographical: int (1-5),
    technological: int (1-5),
    reference_year: Optional[int],
    data_year: Optional[int],
    notes: Optional[str]
)
```

### DQIScore
```python
DQIScore(
    score: float (0-100),
    quality_label: str,  # "Excellent", "Good", "Fair", "Poor"
    pedigree_contribution: float,
    source_contribution: float,
    tier_contribution: float,
    pedigree_score: Optional[PedigreeScore],
    factor_source: Optional[str],
    data_tier: Optional[int]
)
```

### UncertaintyResult
```python
UncertaintyResult(
    mean: float,
    median: float,
    std_dev: float,
    relative_std_dev: float,
    confidence_95_lower: float,
    confidence_95_upper: float,
    distribution_type: DistributionType,
    uncertainty_sources: List[str],
    pedigree_score: Optional[PedigreeScore]
)
```

### MonteCarloResult
```python
MonteCarloResult(
    iterations: int,
    mean: float,
    median: float,
    std_dev: float,
    p5: float,
    p50: float,
    p95: float,
    min_value: float,
    max_value: float,
    sensitivity_indices: Dict[str, float],
    top_contributors: List[str],
    computation_time: float
)
```

## Constants

### GWP Values (AR5)
```python
GWP_AR5 = {
    "CO2": 1.0,
    "CH4": 28.0,
    "N2O": 265.0,
    "HFC-134a": 1300.0,
    "SF6": 23500.0,
    ...
}
```

### Default Uncertainties
```python
DEFAULT_UNCERTAINTIES = {
    "metals": 0.15,          # 15%
    "electricity": 0.10,     # 10%
    "plastics": 0.20,        # 20%
    "road_transport": 0.15,  # 15%
    "default": 0.50,         # 50%
    ...
}
```

## Configuration

Configuration can be set via environment variables:

```bash
# Monte Carlo configuration
METHODOLOGIES_MONTE_CARLO__DEFAULT_ITERATIONS=10000
METHODOLOGIES_MONTE_CARLO__DEFAULT_SEED=42

# Uncertainty configuration
METHODOLOGIES_UNCERTAINTY__DEFAULT_UNCERTAINTY=0.5
METHODOLOGIES_UNCERTAINTY__MIN_UNCERTAINTY=0.01
METHODOLOGIES_UNCERTAINTY__MAX_UNCERTAINTY=5.0

# DQI configuration
METHODOLOGIES_DQI__PEDIGREE_WEIGHT=0.5
METHODOLOGIES_DQI__SOURCE_WEIGHT=0.3
METHODOLOGIES_DQI__TIER_WEIGHT=0.2
```

Or programmatically:

```python
from services.methodologies import update_config

update_config(
    monte_carlo={"default_iterations": 20000},
    uncertainty={"default_uncertainty": 0.3}
)
```

## Performance Benchmarks

- **Monte Carlo (10,000 iterations)**: <1 second
- **Monte Carlo (100,000 iterations)**: <5 seconds
- **Analytical propagation**: <0.001 seconds
- **Pedigree assessment**: <0.001 seconds
- **DQI calculation**: <0.001 seconds

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/services/methodologies/ -v

# Run specific test file
pytest tests/services/methodologies/test_pedigree_matrix.py -v

# Run with coverage
pytest tests/services/methodologies/ --cov=services.methodologies --cov-report=html
```

## Scientific References

1. **ILCD Handbook (2010)**
   - European Commission - JRC
   - https://eplca.jrc.ec.europa.eu/ilcd.html

2. **GHG Protocol: Corporate Value Chain (Scope 3) Standard**
   - World Resources Institute & WBCSD
   - Chapter 7: Uncertainty Assessment

3. **IPCC Guidelines for National Greenhouse Gas Inventories**
   - Volume 1, Chapter 3: Uncertainties
   - https://www.ipcc-nggip.iges.or.jp/

4. **ISO 14044:2006**
   - Environmental management - Life cycle assessment

5. **Weidema et al. (2013)**
   - "Data quality management for LCA"
   - International Journal of LCA

## Integration Examples

### Example 1: Complete Emission Calculation with Uncertainty

```python
from services.methodologies import (
    PedigreeScore,
    DQICalculator,
    MonteCarloSimulator
)

# Step 1: Assess data quality
pedigree = PedigreeScore(
    reliability=1, completeness=2, temporal=1,
    geographical=2, technological=1
)

dqi_calc = DQICalculator()
dqi = dqi_calc.calculate_dqi(
    pedigree_score=pedigree,
    factor_source="ecoinvent",
    data_tier=1
)

# Step 2: Run Monte Carlo simulation
simulator = MonteCarloSimulator(seed=42)
result = simulator.simple_propagation(
    activity_data=1000.0,
    activity_uncertainty=0.1,
    emission_factor=2.5,
    factor_uncertainty=0.15,
    iterations=10000
)

# Step 3: Generate report
print(f"Emission Calculation Report")
print(f"=" * 50)
print(f"\nData Quality:")
print(f"  DQI Score: {dqi.score:.2f} ({dqi.quality_label})")
print(f"\nEmission Results:")
print(f"  Mean: {result.mean:.2f} kg CO2e")
print(f"  Std Dev: {result.std_dev:.2f} kg CO2e")
print(f"  CV: {result.coefficient_of_variation:.2%}")
print(f"\nConfidence Intervals:")
print(f"  90% CI: [{result.p5:.2f}, {result.p95:.2f}] kg CO2e")
print(f"\nSensitivity Analysis:")
for param, sensitivity in result.sensitivity_indices.items():
    print(f"  {param}: {abs(sensitivity):.2%}")
```

### Example 2: Batch Processing with Quality Tracking

```python
from services.methodologies import (
    DQICalculator,
    UncertaintyQuantifier
)

# Initialize
dqi_calc = DQICalculator()
quantifier = UncertaintyQuantifier()

# Process multiple emissions
emissions_data = [
    {"activity": 1000, "factor": 2.5, "category": "electricity", "tier": 1},
    {"activity": 500, "factor": 3.0, "category": "natural_gas", "tier": 2},
    {"activity": 200, "factor": 1.5, "category": "road_transport", "tier": 2},
]

results = []
for data in emissions_data:
    # Calculate emission with uncertainty
    emission = quantifier.propagate_simple(
        activity_mean=data["activity"],
        activity_uncertainty=quantifier.get_category_uncertainty(data["category"]),
        factor_mean=data["factor"],
        factor_uncertainty=0.15
    )

    # Assess quality
    dqi = dqi_calc.calculate_dqi(data_tier=data["tier"])

    results.append({
        "emission": emission.mean,
        "uncertainty": emission.std_dev,
        "dqi": dqi.score,
        "quality": dqi.quality_label
    })

# Summary
total_emission = sum(r["emission"] for r in results)
avg_quality = sum(r["dqi"] for r in results) / len(results)

print(f"Total Emission: {total_emission:.2f} kg CO2e")
print(f"Average DQI: {avg_quality:.2f}")
```

## Support

For questions or issues:
- Email: support@greenlang.ai
- Documentation: https://docs.greenlang.ai/methodologies
- GitHub Issues: https://github.com/greenlang/vcci-platform/issues

## License

Copyright © 2025 GreenLang AI. All rights reserved.
