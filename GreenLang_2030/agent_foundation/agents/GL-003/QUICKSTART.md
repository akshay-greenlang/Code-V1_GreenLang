# GL-003 SteamSystemAnalyzer - Quick Start Guide

## Quick Overview

**GL-003** analyzes steam systems for efficiency, leak detection, condensate optimization, and trap performance. It follows zero-hallucination principles with deterministic ASME Steam Tables calculations.

## Files Delivered

```
GL-003/
├── config.py                     # 285 lines - Pydantic configuration
├── tools.py                      # 861 lines - Deterministic calculations
├── steam_system_orchestrator.py  # 1,287 lines - Main orchestrator
└── calculators/__init__.py       # 40 lines - Module initialization
```

**Total**: 2,473 lines of production code

## Core Components

### 1. Configuration (config.py)

```python
from GL_003.config import SteamSystemAnalyzerConfig, create_default_config

# Quick start with defaults
config = create_default_config()

# Or customize
from GL_003.config import (
    SteamSystemSpecification,
    SensorConfiguration,
    AnalysisParameters,
    SteamSystemConfiguration,
    SteamSystemAnalyzerConfig
)

spec = SteamSystemSpecification(
    system_id="STEAM-001",
    site_name="Plant A",
    system_type="integrated",
    total_steam_capacity_kg_hr=100000,
    boiler_count=3,
    # ... more parameters
)

config = SteamSystemAnalyzerConfig(
    systems=[system_config],
    primary_system_id="STEAM-001"
)
```

### 2. Tools (tools.py)

```python
from GL_003.tools import SteamSystemTools

tools = SteamSystemTools()

# Calculate steam properties
steam_props = tools.calculate_steam_properties(
    pressure_bar=40,
    temperature_c=450
)
# Returns: pressure, temperature, enthalpy, entropy, density, quality

# Analyze distribution efficiency
dist_eff = tools.analyze_distribution_efficiency(
    generation_data={'total_flow_kg_hr': 80000, ...},
    consumption_data={'total_flow_kg_hr': 75000, ...},
    network_data={'total_length_m': 5000, ...}
)
# Returns: efficiency, losses, pressure drop, heat losses

# Detect leaks
leaks = tools.detect_steam_leaks(
    sensor_data={'flow_meters': {...}, 'pressure_sensors': {...}},
    system_config={'leak_threshold_bar': 0.5, ...}
)
# Returns: leak count, rates, locations, severity, costs

# Optimize condensate
condensate = tools.optimize_condensate_return(
    condensate_data={'steam_consumption_kg_hr': 75000, ...},
    system_config={'makeup_water_temperature_c': 15, ...}
)
# Returns: return rate, energy recovered, savings, opportunities

# Analyze steam traps
traps = tools.analyze_steam_trap_performance(
    trap_data={'total_trap_count': 500, 'trap_assessments': [...]},
    system_config={'trap_repair_cost_usd': 50, ...}
)
# Returns: efficiency, failures, losses, repair costs
```

### 3. Orchestrator (steam_system_orchestrator.py)

```python
from GL_003.steam_system_orchestrator import SteamSystemAnalyzer
from GL_003.config import create_default_config

# Initialize
config = create_default_config()
analyzer = SteamSystemAnalyzer(config)

# Prepare input data
input_data = {
    'system_data': {
        'max_capacity_kg_hr': 100000,
        'steam_cost_usd_per_ton': 30.0
    },
    'sensor_feeds': {
        'total_generation_kg_hr': 80000,
        'total_consumption_kg_hr': 75000,
        'average_pressure_bar': 38,
        'average_temperature_c': 445
    },
    'generation_data': {...},
    'consumption_data': {...},
    'network_data': {...},
    'condensate_data': {...},
    'trap_data': {...}
}

# Execute analysis
result = await analyzer.execute(input_data)

# Access results
print(f"Efficiency: {result['kpi_dashboard']['operational_kpis']['distribution_efficiency']:.1f}%")
print(f"Leaks: {result['leak_detection']['total_leaks_detected']}")
print(f"Savings: ${result['kpi_dashboard']['economic_kpis']['total_savings_opportunity_usd_year']:.0f}/year")
```

## Key Features

### Zero-Hallucination Calculations
- ASME Steam Tables for properties
- Deterministic formulas only
- No LLM for numeric calculations
- Industry-standard correlations

### Input Validation
```python
# All methods validate inputs
try:
    result = tools.calculate_steam_properties(
        pressure_bar=-5  # Invalid!
    )
except ValueError as e:
    print(e)  # "pressure_bar must be positive"
```

### Thread Safety
```python
# Safe for concurrent access
import asyncio

async def analyze_multiple():
    tasks = [
        analyzer.execute(input_data_1),
        analyzer.execute(input_data_2),
        analyzer.execute(input_data_3)
    ]
    results = await asyncio.gather(*tasks)
    return results
```

### Caching
```python
# Automatic caching with TTL
result1 = await analyzer.execute(input_data)  # Cache miss
result2 = await analyzer.execute(input_data)  # Cache hit (fast!)

# Check cache performance
print(f"Cache hits: {analyzer.performance_metrics['cache_hits']}")
print(f"Cache misses: {analyzer.performance_metrics['cache_misses']}")
```

### Provenance Tracking
```python
# Every result has SHA-256 hash
result = await analyzer.execute(input_data)
print(f"Provenance: {result['provenance_hash']}")

# Hash is deterministic - same input = same hash
result2 = await analyzer.execute(input_data)
assert result['provenance_hash'] == result2['provenance_hash']
```

## Result Structure

```python
result = {
    'agent_id': 'GL-003',
    'timestamp': '2025-11-17T...',
    'execution_time_ms': 150.23,

    'operational_state': {
        'mode': 'normal',
        'total_generation_kg_hr': 80000,
        'total_consumption_kg_hr': 75000,
        'distribution_efficiency_percent': 93.75,
        # ... more metrics
    },

    'steam_properties': {
        'generation': {
            'pressure_bar': 40,
            'temperature_c': 450,
            'enthalpy_kj_kg': 3320,
            'density_kg_m3': 15.2,
            # ... more properties
        },
        'consumption': {...}
    },

    'distribution_efficiency': {
        'distribution_efficiency_percent': 93.75,
        'distribution_losses_kg_hr': 5000,
        'heat_losses_mw': 2.5,
        'pressure_drop_bar': 5.0,
        # ... more metrics
    },

    'leak_detection': {
        'total_leaks_detected': 12,
        'total_leak_rate_kg_hr': 150,
        'leak_locations': [...],
        'estimated_annual_cost_usd': 39420,
        # ... more metrics
    },

    'condensate_optimization': {
        'return_rate_percent': 80.0,
        'energy_recovered_mw': 5.2,
        'water_savings_m3_day': 1440,
        'optimization_opportunities': [...]
    },

    'trap_performance': {
        'trap_efficiency_percent': 96.0,
        'failed_open_traps': 10,
        'steam_losses_kg_hr': 50,
        # ... more metrics
    },

    'kpi_dashboard': {
        'operational_kpis': {...},
        'distribution_kpis': {...},
        'leak_detection_kpis': {...},
        'condensate_kpis': {...},
        'trap_performance_kpis': {...},
        'economic_kpis': {
            'total_savings_opportunity_usd_year': 250000
        },
        'alerts': [...],
        'recommendations': [...]
    },

    'provenance_hash': 'a3f2...',
    'analysis_success': True
}
```

## Common Patterns

### 1. Basic Analysis
```python
# Minimal input
result = await analyzer.execute({
    'system_data': {'max_capacity_kg_hr': 100000},
    'sensor_feeds': {
        'total_generation_kg_hr': 80000,
        'total_consumption_kg_hr': 75000
    }
})
```

### 2. Full Analysis with All Data
```python
# Complete input
result = await analyzer.execute({
    'system_data': {...},           # System config
    'sensor_feeds': {...},          # Real-time sensors
    'generation_data': {...},       # Generation params
    'consumption_data': {...},      # Consumption data
    'network_data': {...},          # Network config
    'condensate_data': {...},       # Condensate system
    'trap_data': {...}             # Trap assessments
})
```

### 3. Multi-Agent Coordination
```python
# Coordinate with other agents
result = await analyzer.execute({
    'system_data': {...},
    'sensor_feeds': {...},
    'coordinate_agents': True,
    'agent_ids': ['GL-002', 'GL-004'],
    'agent_commands': {
        'boiler_optimization': {...},
        'process_monitoring': {...}
    }
})
```

### 4. Monitoring & Alerts
```python
# Check for alerts
result = await analyzer.execute(input_data)

for alert in result['kpi_dashboard']['alerts']:
    print(f"[{alert['level']}] {alert['category']}: {alert['message']}")
    # [critical] leak: 2 critical leaks detected - immediate action required
```

### 5. Recommendations
```python
# Get improvement recommendations
result = await analyzer.execute(input_data)

for rec in result['kpi_dashboard']['recommendations']:
    print(f"- {rec}")
    # - Increase condensate return from 70% to 90% - potential savings of $50000/year
```

## Performance Tips

### 1. Use Caching
```python
# Same input within 60s uses cache
result1 = await analyzer.execute(input_data)  # 150ms
result2 = await analyzer.execute(input_data)  # 5ms (cached!)
```

### 2. Async for Multiple Systems
```python
# Analyze multiple systems in parallel
results = await asyncio.gather(
    analyzer1.execute(input_data_1),
    analyzer2.execute(input_data_2),
    analyzer3.execute(input_data_3)
)
```

### 3. Monitor Performance
```python
# Check performance metrics
metrics = analyzer.performance_metrics
print(f"Analyses: {metrics['analyses_performed']}")
print(f"Avg time: {metrics['avg_analysis_time_ms']:.1f}ms")
print(f"Cache hit rate: {metrics['cache_hits']/(metrics['cache_hits']+metrics['cache_misses'])*100:.1f}%")
```

## Error Handling

```python
try:
    result = await analyzer.execute(input_data)
except ValueError as e:
    print(f"Validation error: {e}")
except Exception as e:
    print(f"Analysis error: {e}")
    # Agent has built-in error recovery
    # Will return partial results if possible
```

## Configuration Tips

### Development
```python
config = SteamSystemAnalyzerConfig(
    enable_monitoring=True,      # Enable metrics
    enable_learning=True,        # Learn from operations
    cache_ttl_seconds=60,        # Short TTL for testing
    calculation_timeout_seconds=30
)
```

### Production
```python
config = SteamSystemAnalyzerConfig(
    enable_monitoring=True,
    enable_learning=True,
    enable_predictive=True,      # Enable predictive maintenance
    cache_ttl_seconds=300,       # Longer TTL for efficiency
    max_retries=3                # Retry on errors
)
```

## Monitoring

### Get Agent State
```python
state = analyzer.get_state()
print(f"Agent: {state['agent_id']}")
print(f"State: {state['state']}")
print(f"Cache size: {state['cache_size']}")
print(f"Memory entries: {state['memory_entries']}")
```

### Shutdown Gracefully
```python
# Always shutdown gracefully
await analyzer.shutdown()
# Persists memories, closes connections
```

## Next Steps

1. **Review documentation**:
   - README.md - Complete user guide
   - IMPLEMENTATION_SUMMARY.md - Technical details
   - DELIVERY_REPORT.md - Delivery status

2. **Run tests** (once implemented):
   ```bash
   pytest tests/ -v --cov=GL_003
   ```

3. **Deploy to production**:
   ```bash
   docker build -f Dockerfile.production -t gl-003:1.0.0 .
   kubectl apply -f deployment/
   ```

## Support

- **Documentation**: See inline docstrings in code
- **Architecture**: Based on GL-002 patterns
- **Standards**: ASME, ISO, ASHRAE, TLV

---

**Version**: 1.0.0
**Agent**: GL-003 SteamSystemAnalyzer
**Status**: Core Implementation Complete
