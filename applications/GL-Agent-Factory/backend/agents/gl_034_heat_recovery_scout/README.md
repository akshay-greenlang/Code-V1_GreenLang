# GL-034: Heat Recovery Scout Agent (HEAT-RECOVERY-SCOUT)

## Overview

Identifies waste heat recovery opportunities using pinch analysis principles.

## Features

- Match exhaust streams to heat demands
- NPV and payback calculations
- CO2 reduction estimation
- Technology recommendations
- Priority ranking

## Standards

- ISO 50001: Energy Management Systems
- DOE ITP: Industrial Technologies Program

## Quick Start

```python
from backend.agents.gl_034_heat_recovery_scout import *

agent = HeatRecoveryScoutAgent()
result = agent.run(HeatRecoveryScoutInput(
    facility_id="PLANT-001",
    exhaust_streams=[ExhaustStream(stream_id="EX-001", temperature_celsius=400, ...)],
    process_heat_demands=[HeatDemand(demand_id="HD-001", required_temp_celsius=150, ...)]
))
print(f"Total recoverable: {result.total_recoverable_heat_kw} kW")
```
