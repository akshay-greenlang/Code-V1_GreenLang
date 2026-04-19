# GL-080: OPEX Optimizer Agent (OPEXOPTIMIZER)

## Overview

The OpexOptimizerAgent analyzes and optimizes operational expenditures for energy systems and facilities, identifying cost reduction opportunities in energy, maintenance, and labor.

## Features

- Operating cost analysis by category
- Maintenance optimization recommendations
- Energy cost reduction strategies
- Labor efficiency analysis
- Cost driver identification
- Multi-year projections

## Quick Start

```python
from backend.agents.gl_080_opex_optimizer import (
    OpexOptimizerAgent,
    OpexOptimizerInput,
    EnergyCost,
    EnergyType,
)

agent = OpexOptimizerAgent()
input_data = OpexOptimizerInput(
    facility_name="Manufacturing Plant",
    energy_costs=[
        EnergyCost(
            energy_type=EnergyType.ELECTRICITY,
            annual_consumption=5000000,
            rate_per_unit=0.08,
        ),
    ],
)
result = agent.run(input_data)
print(f"Total OPEX: ${result.total_annual_opex_usd:,.2f}")
```

## License

Proprietary - GreenLang Platform
