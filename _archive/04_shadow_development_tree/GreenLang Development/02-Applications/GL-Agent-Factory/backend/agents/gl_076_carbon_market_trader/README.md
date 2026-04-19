# GL-076: Carbon Market Trader Agent (CARBONTRADER)

## Overview

The CarbonMarketTraderAgent provides comprehensive carbon credit trading optimization and portfolio management capabilities for compliance with major carbon market frameworks including EU ETS, California Cap-and-Trade, and RGGI.

## Features

- **Portfolio Optimization**: Modern Portfolio Theory-based position management
- **Compliance Management**: Real-time tracking against surrender obligations
- **Risk Assessment**: VaR, CVaR, and comprehensive risk scoring
- **Trading Recommendations**: Rule-based recommendation engine with confidence scoring
- **Provenance Tracking**: Complete SHA-256 audit trail for regulatory compliance

## Standards Compliance

- EU ETS (European Union Emissions Trading System)
- California Cap-and-Trade Program
- RGGI (Regional Greenhouse Gas Initiative)
- ICAP (International Carbon Action Partnership)

## Installation

```python
from backend.agents.gl_076_carbon_market_trader import (
    CarbonMarketTraderAgent,
    CarbonMarketInput,
    EmissionAllowance,
    MarketPrice,
    ComplianceObligation,
    TradingLimits,
)
```

## Quick Start

```python
from datetime import datetime
from backend.agents.gl_076_carbon_market_trader import (
    CarbonMarketTraderAgent,
    CarbonMarketInput,
    EmissionAllowance,
    MarketPrice,
    ComplianceObligation,
    TradingLimits,
    AllowanceType,
)

# Initialize agent
agent = CarbonMarketTraderAgent()

# Prepare input data
input_data = CarbonMarketInput(
    emission_allowances=[
        EmissionAllowance(
            allowance_type=AllowanceType.EUA,
            vintage_year=2024,
            quantity_tonnes=10000,
            acquisition_price_eur=75.50,
        ),
    ],
    market_prices=[
        MarketPrice(
            allowance_type=AllowanceType.EUA,
            current_price_eur=82.30,
            volatility_30d_pct=25.0,
        ),
    ],
    compliance_obligations=ComplianceObligation(
        period_end=datetime(2024, 12, 31),
        required_surrenders_tonnes=8500,
        penalty_per_tonne_eur=100.0,
    ),
    trading_limits=TradingLimits(
        max_daily_volume_tonnes=5000,
        max_position_tonnes=50000,
    ),
)

# Run analysis
result = agent.run(input_data)

# Access results
print(f"Portfolio Value: EUR {result.total_portfolio_value_eur:,.2f}")
print(f"Compliance Status: {result.compliance_status.state.value}")
print(f"Risk Level: {result.risk_assessment.overall_risk_level.value}")
print(f"VaR 95%: EUR {result.risk_assessment.var_95_eur:,.2f}")

# Review recommendations
for rec in result.trading_recommendations:
    print(f"Action: {rec.action.value} {rec.quantity_tonnes} tonnes")
    print(f"Confidence: {rec.confidence_score:.1%}")
    print(f"Rationale: {rec.rationale}")
```

## Input Schema

### EmissionAllowance
| Field | Type | Description |
|-------|------|-------------|
| allowance_type | AllowanceType | Type of emission allowance (EUA, CCA, etc.) |
| vintage_year | int | Vintage year of allowance |
| quantity_tonnes | float | Quantity in tonnes CO2e |
| acquisition_price_eur | float | Acquisition price per tonne |

### MarketPrice
| Field | Type | Description |
|-------|------|-------------|
| allowance_type | AllowanceType | Type of emission allowance |
| current_price_eur | float | Current market price |
| volatility_30d_pct | float | 30-day volatility percentage |

### ComplianceObligation
| Field | Type | Description |
|-------|------|-------------|
| period_end | datetime | Compliance period end date |
| required_surrenders_tonnes | float | Required surrender amount |
| penalty_per_tonne_eur | float | Non-compliance penalty rate |

### TradingLimits
| Field | Type | Description |
|-------|------|-------------|
| max_daily_volume_tonnes | float | Maximum daily trading volume |
| max_position_tonnes | float | Maximum total position size |
| risk_tolerance | float | Risk tolerance (0-1) |

## Output Schema

### CarbonMarketOutput
| Field | Type | Description |
|-------|------|-------------|
| trading_recommendations | List[TradingRecommendation] | Actionable recommendations |
| portfolio_positions | List[PortfolioPosition] | Current position summary |
| risk_assessment | RiskAssessment | Complete risk metrics |
| compliance_status | ComplianceStatus | Compliance state and gap analysis |
| provenance_hash | str | SHA-256 audit trail hash |

## Calculation Methods

### Portfolio Valuation
```
Portfolio Value = SUM(quantity_i * current_price_i)
Unrealized P&L = Portfolio Value - Total Cost Basis
```

### Risk Metrics (Parametric VaR)
```
VaR_alpha = Value * z_alpha * sigma * sqrt(T/252)
Expected Shortfall = VaR * phi(z) / (1-alpha)
```

### Compliance Gap
```
Gap = (Holdings + Free Allocation) - Required Surrenders
Coverage Ratio = Available / Required
```

## Zero-Hallucination Guarantee

All calculations use deterministic formulas from established financial mathematics:
- No LLM inference in calculation path
- Standard VaR methodology (RiskMetrics)
- Modern Portfolio Theory for optimization
- Complete reproducibility for regulatory audits

## Testing

Run the test suite:
```bash
pytest backend/agents/gl_076_carbon_market_trader/test_agent.py -v
```

Test coverage target: 85%+

## API Reference

### CarbonMarketTraderAgent

#### Methods

**run(input_data: CarbonMarketInput) -> CarbonMarketOutput**

Execute comprehensive carbon market analysis.

**Arguments:**
- `input_data`: Complete input data including allowances, prices, and constraints

**Returns:**
- `CarbonMarketOutput`: Complete analysis results with recommendations

### Trading Actions

| Action | Description |
|--------|-------------|
| BUY | Acquire additional allowances |
| SELL | Sell surplus allowances |
| HOLD | Maintain current position |
| HEDGE | Execute hedging strategy |
| BANK | Bank allowances for future periods |

### Risk Levels

| Level | Description |
|-------|-------------|
| MINIMAL | < 25% average risk score |
| LOW | 25-40% average risk score |
| MODERATE | 40-60% average risk score |
| HIGH | 60-80% average risk score |
| CRITICAL | > 80% average risk score |

## Version History

- **1.0.0** (2024): Initial release with EU ETS and California C&T support

## License

Proprietary - GreenLang Platform
