# Scope3CalculatorAgent - Quick Start Guide

**Get up and running in 5 minutes**

---

## Installation

```bash
cd GL-VCCI-Carbon-APP/VCCI-Scope3-Platform
pip install -r requirements.txt
```

---

## Minimal Example

```python
import asyncio
from services.agents.calculator import Scope3CalculatorAgent, Category1Input
from services.factor_broker import FactorBroker

async def main():
    # Initialize
    factor_broker = FactorBroker()
    calculator = Scope3CalculatorAgent(factor_broker=factor_broker)

    # Calculate
    result = await calculator.calculate_category_1(
        Category1Input(
            product_name="Steel",
            quantity=1000,
            quantity_unit="kg",
            region="US",
            supplier_pcf=1.85  # Tier 1 data
        )
    )

    # Display results
    print(f"✅ Emissions: {result.emissions_tco2e:.3f} tCO2e")
    print(f"📊 Quality: {result.data_quality.rating} ({result.data_quality.dqi_score:.0f}/100)")
    print(f"🎯 Tier: {result.tier.value}")

asyncio.run(main())
```

**Output**:
```
✅ Emissions: 1.850 tCO2e
📊 Quality: excellent (90/100)
🎯 Tier: tier_1
```

---

## Category Examples

### Category 1: Purchased Goods

```python
# Tier 1: Supplier-specific PCF
result = await calculator.calculate_category_1(
    Category1Input(
        product_name="Steel",
        quantity=1000,
        quantity_unit="kg",
        region="US",
        supplier_pcf=1.85
    )
)

# Tier 2: Average product emission factor
result = await calculator.calculate_category_1(
    Category1Input(
        product_name="Aluminum",
        quantity=500,
        quantity_unit="kg",
        region="US",
        product_category="Metals"
    )
)

# Tier 3: Spend-based
result = await calculator.calculate_category_1(
    Category1Input(
        product_name="Office Supplies",
        region="US",
        spend_usd=10000,
        economic_sector="services"
    )
)
```

### Category 4: Logistics (ISO 14083)

```python
from services.agents.calculator import Category4Input, TransportMode

result = await calculator.calculate_category_4(
    Category4Input(
        transport_mode=TransportMode.ROAD_TRUCK_HEAVY,
        distance_km=500,
        weight_tonnes=25
    )
)

print(f"Logistics: {result.emissions_tco2e:.3f} tCO2e")
print(f"ISO 14083: {result.metadata['iso_14083_compliant']}")
```

### Category 6: Business Travel

```python
from services.agents.calculator import (
    Category6Input,
    Category6FlightInput,
    Category6HotelInput,
    CabinClass
)

result = await calculator.calculate_category_6(
    Category6Input(
        flights=[
            Category6FlightInput(
                distance_km=6000,
                cabin_class=CabinClass.ECONOMY,
                num_passengers=1
            )
        ],
        hotels=[
            Category6HotelInput(nights=3, region="GB")
        ]
    )
)

print(f"Travel: {result.emissions_tco2e:.3f} tCO2e")
```

---

## Batch Processing

```python
records = [
    {"product_name": "Steel", "quantity": 1000, "quantity_unit": "kg", "region": "US"},
    {"product_name": "Aluminum", "quantity": 500, "quantity_unit": "kg", "region": "US"},
]

batch = await calculator.calculate_batch(records, category=1)

print(f"✅ Success: {batch.successful_records}/{batch.total_records}")
print(f"📊 Total: {batch.total_emissions_tco2e:.3f} tCO2e")
```

---

## Configuration

```python
from services.agents.calculator import CalculatorConfig

config = CalculatorConfig(
    enable_monte_carlo=True,
    monte_carlo_iterations=10000,
    enable_provenance=True
)

calculator = Scope3CalculatorAgent(
    factor_broker=factor_broker,
    config=config
)
```

---

## Access Results

```python
result = await calculator.calculate_category_1(data)

# Core emissions
print(result.emissions_kgco2e)  # In kgCO2e
print(result.emissions_tco2e)   # In tCO2e

# Quality
print(result.data_quality.dqi_score)  # 0-100
print(result.data_quality.rating)      # excellent/good/fair/poor
print(result.tier)                     # tier_1/tier_2/tier_3

# Uncertainty (if enabled)
if result.uncertainty:
    print(result.uncertainty.mean)
    print(result.uncertainty.p5)
    print(result.uncertainty.p95)
    print(result.uncertainty.uncertainty_range)

# Provenance
print(result.provenance.calculation_id)
print(result.provenance.provenance_chain)
```

---

## Performance Stats

```python
stats = calculator.get_performance_stats()

print(f"Total: {stats['total_calculations']}")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Throughput: {stats['throughput_per_second']:.0f}/sec")
```

---

## Common Patterns

### With Uncertainty

```python
config = CalculatorConfig(
    enable_monte_carlo=True,
    monte_carlo_iterations=10000
)

result = await calculator.calculate_category_1(data)

print(f"Mean: {result.uncertainty.mean:.2f} kgCO2e")
print(f"95% CI: [{result.uncertainty.p5:.2f}, {result.uncertainty.p95:.2f}]")
```

### With Provenance

```python
config = CalculatorConfig(enable_provenance=True)

result = await calculator.calculate_category_1(data)

print(f"Calc ID: {result.provenance.calculation_id}")
print(f"Chain: {result.provenance.provenance_chain}")
```

### Batch with Error Handling

```python
try:
    batch = await calculator.calculate_batch(records, category=1)

    if batch.failed_records > 0:
        print("⚠️  Some records failed:")
        for error in batch.errors:
            print(f"  - Record {error['record_index']}: {error['error']}")

except BatchProcessingError as e:
    print(f"❌ Batch failed: {e.message}")
```

---

## Next Steps

1. **Read Full Documentation**: See [README.md](./README.md)
2. **Review Implementation**: See [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)
3. **Run Tests**: `pytest tests/agents/calculator/ -v`
4. **Check Examples**: See `examples/` directory
5. **Configure**: Set environment variables (see README)

---

**Need Help?**
- Full API docs: [README.md](./README.md)
- Implementation details: [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)
- GitHub: [GL-VCCI-Carbon-APP](https://github.com/akshay-greenlang/Code-V1_GreenLang)
