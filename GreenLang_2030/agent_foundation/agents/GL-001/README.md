# GL-001 ProcessHeatOrchestrator

Master orchestrator for all process heat operations across industrial facilities.

## Overview

The ProcessHeatOrchestrator (GL-001) is the primary control agent for industrial process heat management. It coordinates thermal efficiency calculations, heat distribution optimization, emissions compliance, and multi-agent orchestration while maintaining zero-hallucination guarantees for all calculations.

## Features

### Core Capabilities
- **Thermal Efficiency Calculation**: Deterministic calculation of overall, Carnot, and heat recovery efficiencies
- **Heat Distribution Optimization**: Linear programming-based optimization for heat allocation
- **Energy Balance Validation**: Real-time validation of energy conservation principles
- **Emissions Compliance**: Automated compliance checking against regulatory limits
- **KPI Dashboard Generation**: Comprehensive operational, environmental, and financial KPIs
- **Multi-Agent Coordination**: Orchestration of GL-002 through GL-005 sub-agents
- **SCADA Integration**: Real-time data processing from industrial control systems
- **ERP Integration**: Business data synchronization for cost optimization

### Zero-Hallucination Guarantee
All calculations are performed using deterministic Python functions with no LLM involvement in numerical operations. LLM usage is restricted to:
- Classification tasks (temperature=0.0, seed=42)
- Alert categorization
- Narrative generation for reports

### Performance Targets
- Agent creation: <100ms
- Message processing: <10ms
- Calculation execution: <2s
- Dashboard generation: <5s
- Cache hit rate: >80%

## Installation

```bash
# Navigate to GL-001 directory
cd C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-001

# Install dependencies
pip install -r requirements.txt
```

## Configuration

```python
from config import ProcessHeatConfig, PlantConfiguration, SensorConfiguration

config = ProcessHeatConfig(
    agent_id="GL-001",
    agent_name="ProcessHeatOrchestrator",
    version="1.0.0",
    plants=[
        PlantConfiguration(
            plant_id="PLANT-001",
            plant_name="Chemical Plant Alpha",
            plant_type="chemical",
            location="Houston, TX",
            capacity_mw=500.0,
            max_temperature_c=850.0,
            min_temperature_c=150.0,
            nominal_pressure_bar=40.0,
            primary_fuel="natural_gas",
            renewable_percentage=15.0
        )
    ],
    sensors=[
        SensorConfiguration(
            sensor_id="TEMP-001",
            sensor_type="temperature",
            location="Reactor 1",
            unit="celsius",
            sampling_rate_hz=10.0,
            accuracy_percent=0.5
        )
    ]
)
```

## Usage

### Basic Example

```python
import asyncio
from process_heat_orchestrator import ProcessHeatOrchestrator
from config import ProcessHeatConfig

async def main():
    # Initialize orchestrator
    config = ProcessHeatConfig(...)
    orchestrator = ProcessHeatOrchestrator(config)

    # Prepare input data
    input_data = {
        'plant_data': {
            'inlet_temp_c': 500,
            'outlet_temp_c': 150,
            'ambient_temp_c': 25,
            'fuel_input_mw': 100,
            'useful_heat_mw': 85,
            'heat_recovery_mw': 5
        },
        'sensor_feeds': {
            'heat_demands': {
                'reactor_1': 30,
                'reactor_2': 25,
                'boiler_1': 20
            },
            'heat_sources': {
                'furnace_1': 50,
                'furnace_2': 40
            }
        },
        'constraints': {
            'max_temperature_variance_c': 5.0,
            'min_efficiency_percent': 85.0
        },
        'emissions_data': {
            'co2_kg_hr': 15000,
            'nox_kg_hr': 20,
            'heat_output_mw': 90
        }
    }

    # Execute orchestration
    result = await orchestrator.execute(input_data)

    # Access results
    print(f"Thermal Efficiency: {result['thermal_efficiency']['overall_efficiency']}%")
    print(f"Optimization Score: {result['heat_distribution']['optimization_score']}")
    print(f"Compliance Status: {result['emissions_compliance']['compliance_status']}")
    print(f"Execution Time: {result['execution_time_ms']}ms")

    # Shutdown gracefully
    await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### Multi-Agent Coordination

```python
# Coordinate multiple agents
input_data = {
    'coordinate_agents': True,
    'agent_ids': ['GL-002', 'GL-003', 'GL-004'],
    'agent_commands': {
        'optimize_boilers': {
            'priority': 'high',
            'target_efficiency': 0.90
        },
        'recover_waste_heat': {
            'priority': 'medium',
            'min_recovery_mw': 10
        }
    }
}

result = await orchestrator.execute(input_data)
print(f"Agents Coordinated: {result['coordination_result']['coordinated_agents']}")
```

### SCADA Integration

```python
# Process SCADA data feed
scada_feed = {
    'tags': {
        'TEMP_001': 523.5,
        'PRES_001': 42.3,
        'FLOW_001': 156.7
    },
    'quality': {
        'TEMP_001': 98,
        'PRES_001': 95,
        'FLOW_001': 92
    },
    'alarm_limits': {
        'TEMP_001': {'high': 550, 'high_high': 600},
        'PRES_001': {'high': 45, 'high_high': 50}
    }
}

processed_data = await orchestrator.integrate_scada(scada_feed)
print(f"Data Quality: {processed_data['quality_metrics']['data_availability']}%")
print(f"Active Alarms: {len(processed_data['alarms'])}")
```

## Tool Functions

### Deterministic Tools (Zero-Hallucination)

1. **calculate_thermal_efficiency**: Calculates overall, Carnot, and heat recovery efficiencies
2. **optimize_heat_distribution**: Optimizes heat allocation across process units
3. **validate_energy_balance**: Validates energy conservation principles
4. **check_emissions_compliance**: Checks against regulatory emission limits
5. **generate_kpi_dashboard**: Creates comprehensive performance dashboard
6. **coordinate_process_heat_agents**: Orchestrates sub-agent tasks
7. **integrate_scada_data**: Processes real-time SCADA feeds
8. **integrate_erp_data**: Synchronizes business data from ERP

## Performance Metrics

The agent tracks the following metrics:

```python
metrics = orchestrator.get_state()
print(f"Calculations Performed: {metrics['performance_metrics']['calculations_performed']}")
print(f"Average Calculation Time: {metrics['performance_metrics']['avg_calculation_time_ms']}ms")
print(f"Cache Hit Rate: {metrics['performance_metrics']['cache_hits'] / (metrics['performance_metrics']['cache_hits'] + metrics['performance_metrics']['cache_misses']) * 100}%")
print(f"Agents Coordinated: {metrics['performance_metrics']['agents_coordinated']}")
print(f"Errors Recovered: {metrics['performance_metrics']['errors_recovered']}")
```

## Provenance Tracking

All operations generate SHA-256 hashes for complete audit trails:

```python
result = await orchestrator.execute(input_data)
provenance_hash = result['provenance_hash']
print(f"Provenance Hash: {provenance_hash}")

# Each calculation also has individual provenance
thermal_hash = result['thermal_efficiency']['provenance_hash']
distribution_hash = result['heat_distribution']['provenance_hash']
```

## Error Handling

The orchestrator implements comprehensive error recovery:

```python
try:
    result = await orchestrator.execute(input_data)
except Exception as e:
    # Automatic recovery attempted
    # Partial results returned if possible
    if result['status'] == 'partial_success':
        print(f"Partial success: {result['error']}")
        print(f"Recovered data: {result['recovered_data']}")
```

## Architecture

```
ProcessHeatOrchestrator (GL-001)
├── BaseAgent (inheritance)
│   ├── Lifecycle Management
│   ├── State Tracking
│   └── Checkpointing
├── ProcessHeatTools
│   ├── Thermal Calculations
│   ├── Optimization Algorithms
│   └── Compliance Checking
├── AgentIntelligence
│   ├── ChatSession (deterministic)
│   └── Classification Only
├── Memory Systems
│   ├── ShortTermMemory
│   └── LongTermMemory
└── Message Bus
    └── Multi-Agent Coordination
```

## Testing

```bash
# Run unit tests
python -m pytest tests/test_process_heat_orchestrator.py -v

# Run integration tests
python -m pytest tests/test_integration.py -v

# Run performance benchmarks
python -m pytest tests/test_performance.py --benchmark
```

## Compliance

- **ISO 50001**: Energy Management Systems
- **EPA**: Emission regulations compliance
- **OSHA**: Process safety management
- **EU ETS**: Emissions trading system compliance

## Dependencies

- Python 3.9+
- agent_foundation (base infrastructure)
- numpy (numerical calculations)
- asyncio (async operations)
- hashlib (provenance tracking)
- logging (structured logging)

## Support

For issues or questions, contact the GreenLang development team.

## License

Proprietary - GreenLang 2030

## Version History

- **1.0.0** (2024-11-15): Initial release with full tool implementation
  - 8 deterministic tools implemented
  - Zero-hallucination guarantee
  - Full agent_foundation integration
  - SCADA/ERP integration support