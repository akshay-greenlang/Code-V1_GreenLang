# Calculation Application

Zero-hallucination emissions calculation application built with GreenLang CalculatorAgent.

## Features

- **Zero-Hallucination**: All calculations use registered, validated formulas
- **Parallel Processing**: Thread and process pool support for batch calculations
- **Uncertainty Quantification**: Calculate and track uncertainty in results
- **Formula Registry**: Extensible formula registration system
- **Provenance Tracking**: Full audit trail for all calculations
- **Performance Optimization**: Multi-tier caching for repeated calculations
- **100% Infrastructure**: Built entirely with GreenLang components

## Quick Start

```python
from src.main import CalculationApplication

# Initialize
app = CalculationApplication()

# Calculate Scope 1 emissions
result = await app.calculate(
    formula_name="scope1_emissions",
    inputs={"activity_data": 1000, "emission_factor": 2.5},
    calculate_uncertainty=True
)

print(f"Emissions: {result.value} ± {result.uncertainty} kg CO2e")
```

## Registered Formulas

1. **scope1_emissions** - Direct emissions from fuel combustion
2. **scope2_electricity** - Electricity emissions
3. **scope3_transportation** - Transportation emissions
4. **total_ghg_emissions** - Total GHG emissions
5. **emissions_intensity** - Emissions per unit of production
6. **calculate_uncertainty** - Uncertainty quantification

## Batch Processing

```python
calculations = [
    {"formula_name": "scope1_emissions", "inputs": {...}},
    {"formula_name": "scope2_electricity", "inputs": {...}}
]

results = await app.batch_calculate(calculations, parallel=True)
```

## Custom Formulas

```python
def custom_formula(param1: float, param2: float) -> float:
    return param1 * param2 + 10

app.register_custom_formula(
    name="custom",
    formula=custom_formula,
    required_inputs=["param1", "param2"],
    unit="kg CO2e"
)
```
