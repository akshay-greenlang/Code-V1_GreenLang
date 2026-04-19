# GL-079: CAPEX Analyzer Agent (CAPEXANALYZER)

## Overview

The CapexAnalyzerAgent provides comprehensive capital expenditure analysis for energy efficiency and sustainability projects. It handles equipment costing, installation estimates, soft costs, contingency planning, and benchmarking.

## Features

- **Equipment Costing**: Detailed equipment cost breakdown
- **Installation Estimates**: Labor, material, and subcontractor costs
- **Soft Cost Analysis**: Engineering, permitting, project management
- **Contingency Planning**: Risk-based contingency calculation
- **Benchmarking**: Compare against industry cost benchmarks
- **Sensitivity Analysis**: Identify cost drivers
- **Provenance Tracking**: Complete SHA-256 audit trail

## Quick Start

```python
from backend.agents.gl_079_capex_analyzer import (
    CapexAnalyzerAgent,
    CapexAnalyzerInput,
    EquipmentCost,
    InstallationCost,
    SoftCost,
    EquipmentType,
    CostCategory,
)

agent = CapexAnalyzerAgent()

input_data = CapexAnalyzerInput(
    project_name="Solar + Storage Installation",
    equipment_costs=[
        EquipmentCost(
            equipment_type=EquipmentType.SOLAR_PV,
            description="Solar PV System",
            quantity=1,
            unit_cost_usd=75000,
            capacity_per_unit=50,
            capacity_unit="kW",
        ),
    ],
    installation_costs=[
        InstallationCost(
            category=CostCategory.ELECTRICAL,
            description="Electrical Installation",
            labor_hours=160,
            labor_rate_usd=75,
            material_cost_usd=5000,
        ),
    ],
    contingency_percent=10.0,
    project_size_capacity=50,
    capacity_unit="kW",
)

result = agent.run(input_data)

print(f"Total CAPEX: ${result.total_capex_usd:,.2f}")
print(f"Cost per kW: ${result.cost_per_capacity_unit:,.2f}")
```

## Calculation Methods

### Total CAPEX
```
Subtotal = Equipment + Installation + Soft_Costs
Contingency = Subtotal * Contingency%
Total_CAPEX = Subtotal + Contingency
```

### Contingency Guidelines (AACE 18R-97)
- Conceptual: 25-50%
- Budget: 15-30%
- Definitive: 5-15%

## Testing

```bash
pytest backend/agents/gl_079_capex_analyzer/test_agent.py -v
```

## License

Proprietary - GreenLang Platform
