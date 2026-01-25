# Framework_GreenLang Quick Start Guide

## Overview

Framework_GreenLang is the standard development framework for all GreenLang AI agents. It provides:

- **Quality Standards**: 9-dimension scoring rubric aligned with ISO/IEC 42001, NIST AI RMF
- **Agent Templates**: Pre-built templates for calculators, optimizers, and more
- **Shared Utilities**: Unit conversion, provenance tracking, validation, constants
- **Development Tools**: Scaffolding generator, validator, scorer

## Installation

```python
# Add Framework_GreenLang to your Python path
import sys
sys.path.insert(0, "c:/Users/aksha/Code-V1_GreenLang/GL Agents")

from Framework_GreenLang import *
```

## Creating a New Agent

### Step 1: Define Configuration

```python
from Framework_GreenLang.templates import AgentConfig, AgentCategory, AgentType

config = AgentConfig(
    agent_id="GL-007",
    name="STEAMTRAP",
    full_name="Steam Trap Monitoring and Analysis Agent",
    description="Detects and quantifies steam trap failures for energy recovery",
    category=AgentCategory.STEAM,
    agent_type=AgentType.CALCULATOR,
    standards=["ISO 6552", "ASME PTC 39"],
    include_api=True,
    include_explainability=True,
)
```

### Step 2: Generate Scaffolding

```python
from Framework_GreenLang.tools import AgentScaffolder

scaffolder = AgentScaffolder(output_dir="c:/Users/aksha/Code-V1_GreenLang/GL Agents")
agent_path = scaffolder.create_agent(config)
print(f"Created agent at: {agent_path}")
```

### Step 3: Implement Calculators

Edit `calculators/` modules to add your deterministic calculations:

```python
from Framework_GreenLang.shared import DeterministicCalculator, CalculationResult

class SteamTrapCalculator(DeterministicCalculator):
    NAME = "SteamTrapLoss"
    VERSION = "1.0.0"

    def _validate_inputs(self, inputs):
        errors = []
        if inputs.pressure <= 0:
            errors.append("Pressure must be positive")
        return errors

    def _calculate(self, inputs):
        # Your deterministic calculation here
        return result
```

### Step 4: Validate Agent

```python
from Framework_GreenLang.tools import AgentValidator

validator = AgentValidator("c:/agents/GL-007_STEAMTRAP")
report = validator.validate()
print(validator.get_summary())
```

### Step 5: Score Agent

```python
from Framework_GreenLang.scoring import AgentScorer

scorer = AgentScorer()
report = scorer.score_agent("c:/agents/GL-007_STEAMTRAP")
print(f"Score: {report.total_score}/100")
```

## Using Shared Utilities

### Unit Conversion

```python
from Framework_GreenLang.shared import UnitConverter

conv = UnitConverter()

# Temperature
celsius = conv.convert_temperature(100, "F", "C")  # 37.78

# Energy
kWh = conv.convert_energy(3600000, "J", "kWh")  # 1.0

# Pressure
psi = conv.convert_pressure(101325, "Pa", "psi")  # 14.7

# Auto-detect type
result = conv.convert(500, "kPa", "bar")  # 5.0
```

### Provenance Tracking

```python
from Framework_GreenLang.shared import ProvenanceTracker

tracker = ProvenanceTracker(agent_id="GL-007", version="1.0.0")

# Track a calculation
with tracker.track("steam_loss", inputs) as ctx:
    result = calculate_steam_loss(inputs)
    ctx.set_output(result)

# Get provenance record
record = tracker.last_record
print(f"Computation hash: {record.computation_hash}")
print(f"Inputs hash: {record.inputs_hash}")
```

### Input Validation

```python
from Framework_GreenLang.shared import ValidationEngine

engine = ValidationEngine()
engine.add_field("temperature", required=True, min_value=-273.15)
engine.add_field("pressure", required=True, min_value=0)
engine.add_field("efficiency", required=True, min_value=0, max_value=1)

result = engine.validate({
    "temperature": 150,
    "pressure": 500000,
    "efficiency": 0.85,
})

if result.is_valid:
    print("Inputs valid!")
else:
    print(f"Errors: {result.errors}")
```

### Physical Constants

```python
from Framework_GreenLang.shared import PhysicalConstants, EmissionFactors

# Physical constants
water_cp = PhysicalConstants.WATER_SPECIFIC_HEAT.value  # 4186 J/(kg·K)
steam_latent = PhysicalConstants.WATER_LATENT_HEAT_VAPORIZATION.value

# Emission factors
ng_factor = EmissionFactors.NATURAL_GAS_KG_CO2_PER_KWH.value  # 0.18293
elec_factor = EmissionFactors.get_electricity_factor("UK").value  # 0.20705
```

## Quality Standards

Target scores by dimension:

| Dimension | Weight | Target |
|-----------|--------|--------|
| Mathematical Rigor | 20% | ≥18/20 |
| Determinism | 15% | 15/15 |
| Data Models | 15% | ≥13/15 |
| Explainability | 15% | ≥13/15 |
| Testing | 12% | ≥10/12 |
| API | 8% | ≥7/8 |
| Deployment | 8% | ≥7/8 |
| Documentation | 5% | ≥4/5 |
| Safety | 2% | 2/2 |
| **Total** | 100% | **≥90** |

## Best Practices

1. **Always use provenance tracking** - Every calculation must have SHA-256 hash
2. **Validate all inputs** - Use Pydantic v2 models with constraints
3. **Document formulas** - Include source references (e.g., "ASME PTC 4.1 Eq. 3.2")
4. **Write golden tests** - Known input/output pairs for determinism verification
5. **Target 85% coverage** - Comprehensive unit and integration tests
6. **Include explainability** - Human-readable audit trails for all outputs

## Support

- Framework documentation: `Framework_GreenLang/docs/`
- Quality standards: `Framework_GreenLang/standards/`
- Example agents: GL-001 through GL-006
