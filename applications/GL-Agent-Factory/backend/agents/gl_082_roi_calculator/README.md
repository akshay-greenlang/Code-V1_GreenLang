# GL-082: ROI Calculator Agent (ROICALCULATOR)

## Overview

The ROICalculatorAgent provides comprehensive return on investment analysis for energy efficiency and sustainability projects, including NPV, IRR, MIRR, payback period, and sensitivity analysis.

## Features

- NPV (Net Present Value)
- IRR (Internal Rate of Return)
- MIRR (Modified IRR)
- Simple and discounted payback
- Profitability Index
- Sensitivity analysis
- Investment decision support

## Quick Start

```python
from backend.agents.gl_082_roi_calculator import (
    ROICalculatorAgent,
    ROICalculatorInput,
    CashFlow,
    InvestmentType,
    CashFlowType,
)

agent = ROICalculatorAgent()
input_data = ROICalculatorInput(
    project_name="Solar Installation",
    investment_type=InvestmentType.RENEWABLE,
    initial_investment_usd=500000,
    cash_flows=[
        CashFlow(
            year=1,
            amount_usd=80000,
            flow_type=CashFlowType.ENERGY_SAVINGS,
            is_recurring=True,
        ),
    ],
    analysis_period_years=20,
    discount_rate_percent=8,
)
result = agent.run(input_data)
print(f"NPV: ${result.roi_metrics.npv_usd:,.2f}")
print(f"IRR: {result.roi_metrics.irr_percent}%")
print(f"Decision: {result.investment_decision}")
```

## Formulas

### NPV
```
NPV = SUM(CF_t / (1 + r)^t) for t = 0 to N
```

### IRR
```
Solves: SUM(CF_t / (1 + IRR)^t) = 0
```

### MIRR
```
MIRR = (FV_positives / PV_negatives)^(1/n) - 1
```

## License

Proprietary - GreenLang Platform
