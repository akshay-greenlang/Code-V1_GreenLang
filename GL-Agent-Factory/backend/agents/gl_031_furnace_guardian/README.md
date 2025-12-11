# GL-031: Furnace Guardian Agent (FURNACE-GUARDIAN)

## Overview

The Furnace Guardian Agent provides comprehensive safety monitoring for industrial furnaces, ensuring compliance with NFPA 86, API 560, and EN 746 standards.

## Features

- **Interlock Validation**: Monitors all safety interlocks and detects bypasses
- **Purge Verification**: Validates purge cycle completion per NFPA 86 requirements
- **Flame Supervision**: Monitors flame detectors (UV/IR) and signal quality
- **Temperature Monitoring**: Checks against low, high, and high-high limits
- **Pressure Monitoring**: Validates pressures against safety limits
- **Safety Scoring**: Calculates weighted safety score (0-100)
- **Compliance Tracking**: Reports compliance status per applicable standards
- **Provenance**: Complete SHA-256 audit trail for regulatory compliance

## Standards Compliance

| Standard | Description |
|----------|-------------|
| NFPA 86 | Standard for Ovens and Furnaces |
| API 560 | Fired Heaters for General Refinery Service |
| EN 746 | Industrial Thermoprocessing Equipment |
| IEC 61511 | Safety Instrumented Systems |

## Installation

```python
from backend.agents.gl_031_furnace_guardian import (
    FurnaceGuardianAgent,
    FurnaceGuardianInput,
    TemperatureReading,
    PressureReading,
    FlameStatus,
    InterlockStatus,
    PurgeData,
)
```

## Quick Start

```python
from backend.agents.gl_031_furnace_guardian import *
from backend.agents.gl_031_furnace_guardian.models import *

# Create agent instance
agent = FurnaceGuardianAgent()

# Prepare input data
input_data = FurnaceGuardianInput(
    furnace_id="FRN-001",
    temps=[
        TemperatureReading(
            sensor_id="T1",
            value_celsius=650,
            low_limit=200,
            high_limit=800,
            high_high_limit=900
        )
    ],
    pressures=[
        PressureReading(
            sensor_id="P1",
            value_kpa=101.3,
            low_limit=90,
            high_limit=120
        )
    ],
    flame_status=FlameStatus(
        is_detected=True,
        signal_strength=85,
        noise_level=5,
        detector_type=FlameDetectorType.UV_IR_COMBINED
    ),
    interlocks=[
        InterlockStatus(
            interlock_type=InterlockType.FLAME_FAILURE,
            is_ok=True
        ),
        InterlockStatus(
            interlock_type=InterlockType.HIGH_TEMPERATURE,
            is_ok=True
        )
    ],
    purge_data=PurgeData(
        status=PurgeStatus.COMPLETE,
        airflow_cfm=5000,
        furnace_volume_cubic_feet=1000,
        purge_time_seconds=120,
        furnace_class="A"
    )
)

# Run analysis
result = agent.run(input_data)

# Access results
print(f"Safety Score: {result.safety_score}")
print(f"Risk Level: {result.risk_level}")
print(f"Violations: {len(result.violations)}")
print(f"Provenance Hash: {result.provenance_hash}")
```

## Safety Score Components

The overall safety score is calculated using weighted components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Interlocks | 30% | Percentage of interlocks in OK state |
| Purge | 20% | Whether purge verification passed |
| Flame | 25% | Flame detection and signal quality |
| Temperature | 15% | Temperatures within limits |
| Pressure | 10% | Pressures within limits |

## Risk Levels

| Score Range | Risk Level | Action Required |
|-------------|------------|-----------------|
| 95-100 | NONE | Normal operation |
| 85-94 | LOW | Monitor conditions |
| 70-84 | MODERATE | Investigate and correct |
| 50-69 | HIGH | Immediate attention required |
| 0-49 | CRITICAL | Emergency shutdown recommended |

## Purge Requirements (NFPA 86)

| Furnace Class | Required Volume Changes | Minimum Time |
|--------------|------------------------|--------------|
| Class A (<8000 Btu/ft3) | 4 | 30 seconds |
| Class B (>=8000 Btu/ft3) | 8 | 30 seconds |

## Violation Severities

| Severity | Description | Typical Response |
|----------|-------------|------------------|
| INFO | Informational notice | Log and monitor |
| WARNING | Condition requires attention | Schedule correction |
| ALARM | Safety limit exceeded | Investigate immediately |
| TRIP | Safety trip condition | Equipment stops |
| EMERGENCY | Critical safety violation | Immediate shutdown |

## API Reference

### FurnaceGuardianAgent

Main agent class for furnace safety monitoring.

**Methods:**
- `run(input_data: FurnaceGuardianInput) -> FurnaceGuardianOutput`: Execute safety analysis

### Input Models

- `FurnaceGuardianInput`: Main input container
- `TemperatureReading`: Temperature sensor data
- `PressureReading`: Pressure sensor data
- `FlameStatus`: Flame detector status
- `InterlockStatus`: Safety interlock status
- `PurgeData`: Purge cycle data

### Output Models

- `FurnaceGuardianOutput`: Main output container
- `SafetyViolation`: Detected safety violation
- `CorrectiveAction`: Recommended corrective action
- `ComplianceStatus`: Compliance status per standard

## Testing

Run the test suite:

```bash
pytest backend/agents/gl_031_furnace_guardian/test_agent.py -v
```

## Zero-Hallucination Guarantee

All calculations in this agent use deterministic formulas from published safety standards:
- No LLM inference in calculation path
- All formulas traced to specific standard sections
- Complete provenance chain with SHA-256 hashes

## Author

GreenLang Process Heat Safety Team

## Version History

- 1.0.0: Initial release with NFPA 86, API 560, EN 746 compliance
