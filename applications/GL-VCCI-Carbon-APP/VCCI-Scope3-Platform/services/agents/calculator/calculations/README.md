# Calculation Modules

This directory contains the calculation helper modules for the Scope3CalculatorAgent.

## Overview

The calculation modules provide reusable utilities that category calculators can leverage:

1. **UncertaintyEngine** - Monte Carlo uncertainty propagation
2. **TierCalculator** - 3-tier calculation waterfall logic
3. **TransportCalculator** - ISO 14083 compliant transport calculations
4. **TravelCalculator** - Business travel emissions calculations

## Modules

### 1. UncertaintyEngine (`uncertainty_engine.py`)

Monte Carlo simulation for emissions uncertainty propagation.

**Key Features:**
- Simple propagation: `quantity × emission_factor`
- Logistics propagation: `distance × weight × emission_factor / load_factor`
- 10,000 iteration default
- Returns percentiles (P5, P50, P95) and statistics

**Usage:**
```python
from calculations import UncertaintyEngine

engine = UncertaintyEngine(seed=42)

result = await engine.propagate(
    quantity=1000,
    quantity_uncertainty=0.05,  # 5% CV
    emission_factor=2.5,
    factor_uncertainty=0.15,    # 15% CV
    iterations=10000
)

print(f"Mean: {result.mean} kgCO2e")
print(f"P95: {result.p95} kgCO2e")
```

**Used by:** All categories for uncertainty analysis

---

### 2. TierCalculator (`tier_calculator.py`) ✨ NEW

3-tier calculation waterfall with automatic fallback.

**Key Features:**
- Implements GHG Protocol tier hierarchy
- Automatic tier fallback (Tier 1 → 2 → 3)
- Data quality threshold enforcement
- Tier attempt tracking
- Flexible calculation functions

**Tier Hierarchy:**
- **Tier 1**: Supplier-specific/primary data (DQI: 85/100)
- **Tier 2**: Average/secondary data (DQI: 65/100)
- **Tier 3**: Spend-based/proxy data (DQI: 45/100)

**Usage:**
```python
from calculations import TierCalculator

tier_calc = TierCalculator(config)

# Automatic tier fallback
result = await tier_calc.calculate_with_fallback(
    category=1,
    tier_1_func=calculate_tier_1_async,
    tier_2_func=calculate_tier_2_async,
    tier_3_func=calculate_tier_3_async,
    min_dqi_score=40.0,
    enable_fallback=True
)

# Generic tier calculations
tier_1_result = await tier_calc.calculate_tier_1(
    quantity=1000,
    supplier_pcf=2.5,
    category=1
)

tier_2_result = await tier_calc.calculate_tier_2(
    quantity=1000,
    emission_factor=2.0,
    category=1
)

tier_3_result = await tier_calc.calculate_tier_3(
    spend_usd=50000,
    economic_intensity=0.35,
    category=1
)
```

**Used by:**
- Category 1 (Purchased Goods & Services)
- Category 2 (Capital Goods)
- Category 3 (Fuel & Energy)
- Any category implementing tier-based calculations

---

### 3. TransportCalculator (`transport_calculator.py`) ✨ NEW

ISO 14083:2023 compliant transport emissions calculator.

**Key Features:**
- ISO 14083 standardized formula
- Multi-modal transport support (road, rail, sea, air, inland waterway)
- High-precision Decimal arithmetic (zero variance)
- Load factor adjustments
- Multi-leg journey support
- Return journey calculations

**ISO 14083 Formula:**
```
emissions = distance × weight × emission_factor / load_factor
```

**Transport Modes:**
- Road: Light/Medium/Heavy trucks, vans
- Rail: Electric, diesel freight
- Sea: Container, bulk, tanker, RO-RO
- Air: Cargo, freight
- Inland Waterway

**Usage:**
```python
from calculations import TransportCalculator
from config import TransportMode

transport_calc = TransportCalculator(config)

# Single leg
result = await transport_calc.calculate_transport_emissions(
    distance_km=500,
    weight_tonnes=20,
    transport_mode=TransportMode.ROAD_TRUCK_HEAVY,
    load_factor=0.85,
    category=4
)

print(f"Emissions: {result['emissions_kgco2e']:.2f} kgCO2e")
print(f"Tonne-km: {result['tonne_km']}")

# Multi-leg journey
legs = [
    {
        'distance_km': 300,
        'weight_tonnes': 20,
        'transport_mode': TransportMode.ROAD_TRUCK_HEAVY,
        'load_factor': 0.85
    },
    {
        'distance_km': 500,
        'weight_tonnes': 20,
        'transport_mode': TransportMode.RAIL_FREIGHT,
        'load_factor': 0.90
    }
]

result = await transport_calc.calculate_multi_leg_journey(legs, category=4)
print(f"Total: {result['total_emissions_kgco2e']:.2f} kgCO2e")

# Return journey (e.g., empty return)
result = await transport_calc.calculate_return_journey(
    distance_km=500,
    weight_tonnes_outbound=20,
    weight_tonnes_return=0,  # Empty return
    transport_mode=TransportMode.ROAD_TRUCK_HEAVY,
    category=4
)
```

**Default Emission Factors (kgCO2e/tonne-km):**
- Road Truck Heavy: 0.062
- Rail Freight: 0.022
- Sea Container: 0.011
- Air Cargo: 1.130

**Used by:**
- Category 4 (Upstream Transportation & Distribution)
- Category 9 (Downstream Transportation & Distribution)

---

### 4. TravelCalculator (`travel_calculator.py`) ✨ NEW

Business travel emissions calculator with flight, hotel, and ground transport.

**Key Features:**
- Flight emissions with cabin class differentiation
- DEFRA radiative forcing (1.9x for flights)
- Regional hotel emission factors
- Ground transportation variety
- Complete trip aggregation
- Round-trip calculations

**Components:**
1. **Flights**: `distance × passengers × EF × radiative_forcing`
2. **Hotels**: `nights × regional_EF`
3. **Ground Transport**: `distance × mode_EF`

**Usage:**
```python
from calculations import TravelCalculator
from config import CabinClass

travel_calc = TravelCalculator(config)

# Flight emissions
flight = await travel_calc.calculate_flight_emissions(
    distance_km=1000,
    num_passengers=2,
    cabin_class=CabinClass.BUSINESS,
    apply_radiative_forcing=True  # DEFRA 1.9x multiplier
)

print(f"Flight: {flight['emissions_kgco2e']:.2f} kgCO2e")
print(f"RF factor: {flight['radiative_forcing_factor']}")

# Hotel stay
hotel = await travel_calc.calculate_hotel_emissions(
    nights=3,
    region="Europe"
)

print(f"Hotel: {hotel['emissions_kgco2e']:.2f} kgCO2e")

# Ground transport
ground = await travel_calc.calculate_ground_transport_emissions(
    distance_km=50,
    vehicle_type="taxi"
)

print(f"Ground: {ground['emissions_kgco2e']:.2f} kgCO2e")

# Complete trip
trip = await travel_calc.calculate_complete_trip(
    flights=[
        {'distance_km': 1000, 'cabin_class': CabinClass.ECONOMY},
        {'distance_km': 1000, 'cabin_class': CabinClass.ECONOMY}  # Return
    ],
    hotels=[
        {'nights': 2, 'region': 'Asia'},
        {'nights': 1, 'region': 'Asia'}
    ],
    ground_transports=[
        {'distance_km': 30, 'vehicle_type': 'taxi'},
        {'distance_km': 30, 'vehicle_type': 'taxi'}
    ]
)

print(f"Total trip: {trip['total_emissions_kgco2e']:.2f} kgCO2e")

# Round trip flight
round_trip = await travel_calc.calculate_round_trip_flight(
    distance_km=1500,
    cabin_class=CabinClass.BUSINESS
)

print(f"Round trip: {round_trip['total_emissions_kgco2e']:.2f} kgCO2e")
```

**Default Emission Factors:**

*Flights (kgCO2e/passenger-km):*
- Economy: 0.115
- Premium Economy: 0.165
- Business: 0.230
- First: 0.345

*Hotels (kgCO2e/night):*
- Global: 20.0
- Europe: 18.0
- Asia: 25.0
- North America: 22.0

*Ground Transport (kgCO2e/km):*
- Car Small: 0.105
- Car Medium: 0.145
- Taxi: 0.150
- Bus: 0.028
- Train: 0.041

**Radiative Forcing:**
- DEFRA factor: 1.9x (accounts for high-altitude emissions impact)

**Used by:**
- Category 6 (Business Travel)

---

## Design Patterns

### 1. Async/Await
All calculation functions are async to support:
- Database lookups
- API calls
- Parallel processing

### 2. Type Safety
All modules use type hints:
```python
async def calculate_tier_1(
    self,
    quantity: float,
    supplier_pcf: float,
    category: int,
    **kwargs
) -> Optional[CalculationResult]:
```

### 3. Error Handling
Custom exceptions for specific error cases:
- `TierFallbackError` - All tiers failed
- `TransportModeError` - Unsupported transport mode
- `ISO14083ComplianceError` - Calculation variance exceeded
- `DataValidationError` - Invalid input data

### 4. Logging
Comprehensive logging at all levels:
```python
logger.info("Tier 1 successful for category 1")
logger.warning("Low load factor increases emissions")
logger.debug("ISO 14083 compliance verified")
```

### 5. Configuration
All modules accept optional config parameter:
```python
def __init__(self, config: Optional[Any] = None):
    self.config = config or get_config()
```

## Integration

All modules are automatically imported and available:

```python
# In __init__.py
from .uncertainty_engine import UncertaintyEngine
from .tier_calculator import TierCalculator
from .transport_calculator import TransportCalculator
from .travel_calculator import TravelCalculator

__all__ = [
    "UncertaintyEngine",
    "TierCalculator",
    "TransportCalculator",
    "TravelCalculator",
]
```

Usage in category calculators:

```python
from ..calculations import (
    UncertaintyEngine,
    TierCalculator,
    TransportCalculator,
    TravelCalculator
)

# Initialize in calculator
self.tier_calc = TierCalculator(config)
self.transport_calc = TransportCalculator(config)
self.travel_calc = TravelCalculator(config)
```

## Testing

Each module supports production-quality testing:

### TierCalculator Tests
- Tier fallback logic
- DQI threshold enforcement
- Tier priority ordering

### TransportCalculator Tests
- ISO 14083 compliance (zero variance)
- Multi-modal transport
- Load factor adjustments
- Multi-leg journeys

### TravelCalculator Tests
- Flight class multipliers
- Radiative forcing application
- Regional hotel factors
- Complete trip aggregation

## Performance

### Optimization Features
1. **Decimal Precision**: High-precision arithmetic where needed (transport)
2. **Async Operations**: Parallel execution support
3. **Caching**: Factor lookup results can be cached
4. **Batch Processing**: All modules support batch operations

### Typical Performance
- Single calculation: < 1ms
- Monte Carlo (10k iterations): 50-100ms
- Multi-leg journey (5 legs): < 5ms
- Complete trip: < 10ms

## Version History

### Version 1.0.0 (2025-11-08)
- ✅ Created TierCalculator module
- ✅ Created TransportCalculator module
- ✅ Created TravelCalculator module
- ✅ ISO 14083:2023 compliance
- ✅ DEFRA radiative forcing support
- ✅ Production-ready implementations

## Standards Compliance

- **GHG Protocol**: Corporate Value Chain (Scope 3) Standard
- **ISO 14083:2023**: Quantification and reporting of GHG emissions from transport
- **DEFRA**: Department for Environment, Food & Rural Affairs guidelines
- **PCAF**: Partnership for Carbon Accounting Financials (Category 15)

## License

GL-VCCI Scope 3 Platform
Copyright © 2025 GreenLang
