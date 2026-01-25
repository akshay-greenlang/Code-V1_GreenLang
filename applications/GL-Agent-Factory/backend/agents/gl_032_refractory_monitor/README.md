# GL-032: Refractory Monitor Agent (REFRACTORY-MONITOR)

## Overview

The Refractory Monitor Agent assesses refractory health in industrial furnaces, heaters, and kilns using thermal imaging data, skin temperature measurements, and material age to predict remaining useful life.

## Features

- **Thermal Imaging Analysis**: Process thermal camera data to detect hotspots
- **Heat Loss Calculation**: Calculate heat loss through multi-layer refractory
- **Thermal Gradient Analysis**: Assess spalling risk from thermal gradients
- **Health Index**: Weighted scoring based on temperature, age, and hotspots
- **Remaining Life Prediction**: Linear regression on health history
- **Maintenance Priority**: Automatic priority determination

## Standards Compliance

| Standard | Description |
|----------|-------------|
| API 560 | Fired Heaters for General Refinery Service |
| ASTM C155 | Standard Classification of Insulating Firebrick |

## Quick Start

```python
from backend.agents.gl_032_refractory_monitor import *

agent = RefractoryMonitorAgent()

input_data = RefractoryMonitorInput(
    equipment_id="FH-001",
    skin_temps=[
        SkinTemperature(x_position=0, y_position=0, temp_celsius=85, zone=RefractoryZone.SIDEWALL)
    ],
    age_days=730,
    design_life_days=1825,
    material_type=RefractoryMaterial.CASTABLE,
    process_temp_celsius=900,
    design_skin_temp_celsius=80
)

result = agent.run(input_data)

print(f"Health Index: {result.health_index}")
print(f"Remaining Life: {result.remaining_life_days} days")
print(f"Priority: {result.maintenance_priority}")
```

## Health Index Calculation

| Component | Weight | Description |
|-----------|--------|-------------|
| Temperature | 40% | Skin temp vs design |
| Age | 40% | Age vs design life |
| Hotspots | -20% | Penalty for hotspots |

## Testing

```bash
pytest backend/agents/gl_032_refractory_monitor/test_agent.py -v
```

## Author

GreenLang Process Heat Reliability Team
