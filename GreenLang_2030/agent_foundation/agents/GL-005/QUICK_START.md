# GL-005 CombustionControlAgent - Quick Start Guide

## Overview

GL-005 is a real-time combustion control agent that automatically manages fuel and air flows to maintain consistent heat output with optimal efficiency and emissions.

**Control Loop:** <100ms cycle time
**Update Rate:** 10 Hz (PID controllers)
**Primary Function:** Automated combustion control for heat output stability

---

## Architecture Quick View

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application (main.py)            │
│                    13 REST Endpoints                        │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│         CombustionControlOrchestrator                       │
│         (combustion_control_orchestrator.py)                │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Fuel Flow    │  │ Air Flow     │  │ O2 Trim      │    │
│  │ PID          │  │ PID          │  │ PID          │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Feedforward  │  │ Stability    │  │ Heat Output  │    │
│  │ Controller   │  │ Analyzer     │  │ Calculator   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                    Integrations                             │
│  ┌──────┐ ┌──────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐ │
│  │ DCS  │ │ PLC  │ │ Analyzers│ │ Sensors │ │  SCADA   │ │
│  └──────┘ └──────┘ └──────────┘ └─────────┘ └──────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
GL-005/agents/
├── combustion_control_orchestrator.py  # Main control logic (1,095 lines)
├── tools.py                            # 12 tool schemas (477 lines)
├── config.py                           # 95+ parameters (430 lines)
├── main.py                             # FastAPI app (455 lines)
└── __init__.py                         # Package exports (95 lines)
```

---

## Quick Start - Running the Agent

### 1. Install Dependencies

```bash
pip install fastapi uvicorn pydantic pydantic-settings prometheus-client
```

### 2. Configure Environment

Create `.env` file:

```bash
# Environment
GREENLANG_ENV=development
DEBUG=true
LOG_LEVEL=INFO

# Control Configuration
CONTROL_LOOP_INTERVAL_MS=100
CONTROL_AUTO_START=false
FUEL_TYPE=natural_gas
HEAT_OUTPUT_TARGET_KW=10000

# DCS Configuration
DCS_HOST=localhost
DCS_PORT=502
DCS_PROTOCOL=modbus_tcp

# PLC Configuration
PLC_HOST=localhost
PLC_PORT=502
PLC_MODBUS_ID=1

# Operating Limits
MIN_FUEL_FLOW=100.0
MAX_FUEL_FLOW=2000.0
MIN_AIR_FLOW=1000.0
MAX_AIR_FLOW=25000.0

# Control Targets
TARGET_O2_PERCENT=3.0
OPTIMAL_EXCESS_AIR_PERCENT=15.0
TARGET_EFFICIENCY_PERCENT=88.0

# PID Tuning - Fuel
FUEL_CONTROL_KP=2.0
FUEL_CONTROL_KI=0.5
FUEL_CONTROL_KD=0.1

# PID Tuning - Air
AIR_CONTROL_KP=1.5
AIR_CONTROL_KI=0.3
AIR_CONTROL_KD=0.08

# PID Tuning - O2 Trim
O2_TRIM_KP=100.0
O2_TRIM_KI=20.0
O2_TRIM_KD=5.0
O2_TRIM_ENABLED=true
```

### 3. Run the Application

```bash
cd C:/Users/aksha/Code-V1_GreenLang/GreenLang_2030/agent_foundation/agents/GL-005/agents
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access the API

- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health
- **Metrics:** http://localhost:8000/metrics

---

## API Endpoints Quick Reference

### Health & Status
```bash
GET  /                      # Agent info
GET  /health                # Liveness probe
GET  /readiness             # Readiness probe
GET  /status                # Detailed status
GET  /config                # Configuration
```

### Combustion Data
```bash
GET  /combustion/state      # Current combustion state
GET  /combustion/stability  # Stability metrics
GET  /state/history?limit=N # State history
```

### Control Operations
```bash
POST /control               # Trigger control cycle
POST /control/enable        # Enable/disable control
GET  /control/history       # Control action history
GET  /control/action/{id}   # Specific control action
```

### Safety & Performance
```bash
GET  /safety/interlocks     # Interlock status
GET  /performance/metrics   # Control performance
GET  /metrics               # Prometheus metrics
```

---

## Code Usage Examples

### 1. Initialize and Start Agent (Programmatic)

```python
import asyncio
from combustion_control_orchestrator import CombustionControlOrchestrator

async def main():
    # Create agent
    agent = CombustionControlOrchestrator()

    # Initialize integrations
    await agent.initialize_integrations()

    # Start control loop
    await agent.start()

asyncio.run(main())
```

### 2. Run Single Control Cycle

```python
async def run_cycle():
    agent = CombustionControlOrchestrator()
    await agent.initialize_integrations()

    # Run one control cycle
    result = await agent.run_control_cycle(heat_demand_kw=10000.0)

    print(f"Success: {result['success']}")
    print(f"Cycle time: {result['cycle_time_ms']}ms")
    print(f"Stability: {result['stability']['overall_stability_score']}")

asyncio.run(run_cycle())
```

### 3. Check Safety Interlocks

```python
async def check_safety():
    agent = CombustionControlOrchestrator()
    await agent.initialize_integrations()

    interlocks = await agent.check_safety_interlocks()

    if interlocks.all_safe():
        print("✓ All interlocks satisfied")
    else:
        failed = interlocks.get_failed_interlocks()
        print(f"✗ Failed interlocks: {failed}")

asyncio.run(check_safety())
```

### 4. Analyze Stability

```python
async def analyze():
    agent = CombustionControlOrchestrator()
    await agent.initialize_integrations()

    # Need some state history first
    for _ in range(60):
        await agent.read_combustion_state()
        await asyncio.sleep(0.1)

    # Analyze stability
    state = agent.current_state
    stability = await agent.analyze_stability(state)

    print(f"Stability Score: {stability.overall_stability_score:.1f}/100")
    print(f"Rating: {stability.stability_rating}")
    if stability.oscillation_detected:
        print(f"⚠ Oscillation detected: {stability.oscillation_frequency_hz}Hz")

asyncio.run(analyze())
```

### 5. Use API Client

```python
import httpx

# Get current state
response = httpx.get("http://localhost:8000/combustion/state")
state = response.json()
print(f"Fuel flow: {state['fuel_flow']} kg/hr")
print(f"Air flow: {state['air_flow']} m3/hr")
print(f"O2: {state['o2_percent']}%")

# Trigger control cycle
response = httpx.post("http://localhost:8000/control",
                     json={"heat_demand_kw": 12000})
result = response.json()
print(f"Control success: {result['success']}")

# Enable automatic control
response = httpx.post("http://localhost:8000/control/enable",
                     json={"enabled": true})
print(response.json()['message'])
```

---

## Configuration Quick Reference

### Critical Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `CONTROL_LOOP_INTERVAL_MS` | 100 | 10-10000 | Control cycle time (ms) |
| `HEAT_OUTPUT_TARGET_KW` | 10000 | 2000-20000 | Target heat output |
| `TARGET_O2_PERCENT` | 3.0 | 1.5-7.0 | Target O2 in flue gas |
| `OPTIMAL_EXCESS_AIR_PERCENT` | 15.0 | 5.0-35.0 | Target excess air |
| `FUEL_CONTROL_KP` | 2.0 | 0.1-10.0 | Fuel PID proportional |
| `AIR_CONTROL_KP` | 1.5 | 0.1-10.0 | Air PID proportional |
| `O2_TRIM_ENABLED` | true | true/false | Enable O2 trim |

### Operating Limits

| Parameter | Default | Safety Limit |
|-----------|---------|--------------|
| `MIN_FUEL_FLOW` | 100.0 | Must be > 0 |
| `MAX_FUEL_FLOW` | 2000.0 | Hardware limit |
| `MIN_AIR_FLOW` | 1000.0 | Flame stability |
| `MAX_AIR_FLOW` | 25000.0 | Blower capacity |
| `MAX_FLAME_TEMPERATURE_C` | 1800.0 | Material limit |
| `MAX_FURNACE_TEMPERATURE_C` | 1400.0 | Design limit |

---

## Control Flow Quick Reference

### Main Control Cycle (run_control_cycle)

```
1. Read Combustion State (<50ms)
   ├─ DCS/PLC data
   ├─ Sensor readings
   ├─ Analyzer measurements
   └─ Calculate derived values

2. Check Safety Interlocks
   ├─ Flame present
   ├─ Pressures OK
   ├─ Temperatures OK
   └─ Emergency stop clear

3. Analyze Stability
   ├─ Heat output variance
   ├─ Temperature stability
   ├─ O2 stability
   └─ Oscillation detection

4. Calculate Control Action
   ├─ Optimize fuel/air ratio
   ├─ PID feedback
   ├─ Feedforward compensation
   └─ O2 trim correction

5. Implement Control
   ├─ Validate setpoints
   ├─ Write to DCS
   ├─ Write to PLC (backup)
   └─ Publish to SCADA

6. Track Performance
   ├─ Cycle time
   ├─ Errors
   └─ Metrics
```

---

## Data Models Quick Reference

### CombustionState
Current process state
```python
{
    "timestamp": "2025-11-18T16:00:00Z",
    "fuel_flow": 1000.0,              # kg/hr
    "air_flow": 12500.0,              # m3/hr
    "air_fuel_ratio": 12.5,
    "furnace_temperature": 1200.0,    # °C
    "flue_gas_temperature": 350.0,    # °C
    "fuel_pressure": 500.0,           # kPa
    "air_pressure": 40.0,             # kPa
    "o2_percent": 3.2,                # %
    "co_ppm": 80.0,                   # ppm
    "heat_output_kw": 10500.0,        # kW
    "thermal_efficiency": 87.5        # %
}
```

### ControlAction
Control output
```python
{
    "action_id": "uuid-here",
    "fuel_flow_setpoint": 1050.0,
    "air_flow_setpoint": 13000.0,
    "fuel_valve_position": 52.5,      # %
    "air_damper_position": 52.0,      # %
    "interlock_satisfied": true,
    "hash": "sha256-hash-here"
}
```

### StabilityMetrics
Stability analysis
```python
{
    "overall_stability_score": 85.5,  # 0-100
    "stability_rating": "good",       # excellent/good/fair/poor/unstable
    "heat_output_stability_index": 0.85,
    "oscillation_detected": false,
    "oscillation_frequency_hz": null
}
```

---

## PID Tuning Quick Guide

### Fuel Flow PID
**Characteristics:** Fast response, moderate overshoot acceptable
```python
FUEL_CONTROL_KP = 2.0   # Aggressive response
FUEL_CONTROL_KI = 0.5   # Moderate integral
FUEL_CONTROL_KD = 0.1   # Light damping
```

### Air Flow PID
**Characteristics:** Slower than fuel, minimize overshoot
```python
AIR_CONTROL_KP = 1.5    # Moderate response
AIR_CONTROL_KI = 0.3    # Light integral
AIR_CONTROL_KD = 0.08   # Very light damping
```

### O2 Trim PID
**Characteristics:** Slow correction, prevent hunting
```python
O2_TRIM_KP = 100.0      # Direct correction (m3/hr per % O2)
O2_TRIM_KI = 20.0       # Very slow integral
O2_TRIM_KD = 5.0        # Light damping
```

### Tuning Tips
- **Increase Kp:** Faster response, more overshoot
- **Increase Ki:** Eliminate steady-state error, risk instability
- **Increase Kd:** Reduce overshoot, sensitive to noise
- **Rule of thumb:** Start conservative, tune up gradually

---

## Troubleshooting

### Agent Won't Start
```bash
# Check logs
tail -f logs/agent.log

# Verify integrations
curl http://localhost:8000/readiness

# Check config
python -c "from config import settings; print(settings.dict())"
```

### Control Cycle Too Slow
```bash
# Check performance
curl http://localhost:8000/performance/metrics

# Common causes:
# - DCS/PLC timeout too high
# - Too many sensors
# - Network latency
# - CPU overload
```

### Safety Interlocks Failing
```bash
# Check interlock status
curl http://localhost:8000/safety/interlocks

# Common failures:
# - Flame not detected
# - Fuel/air pressure low
# - Purge incomplete
# - Emergency stop active
```

### Poor Stability
```bash
# Check stability metrics
curl http://localhost:8000/combustion/stability

# Common causes:
# - PID gains too aggressive
# - Sensor noise
# - Fuel quality variation
# - Air flow oscillation
# - Control loop too fast/slow
```

---

## Performance Targets

| Metric | Target | Acceptable | Action if Exceeded |
|--------|--------|------------|-------------------|
| Control cycle time | <100ms | <150ms | Optimize I/O, reduce sensors |
| Data read time | <50ms | <75ms | Check network, use parallel reads |
| Stability score | >80 | >60 | Tune PID, check disturbances |
| Heat output variance | <2% | <5% | Improve control, check fuel quality |
| O2 stability | ±0.5% | ±1.0% | Tune O2 trim, check analyzer |

---

## Safety Checklist

Before enabling automatic control:

- [ ] All safety interlocks verified functional
- [ ] Flame detection working
- [ ] Fuel/air pressures within limits
- [ ] Temperature sensors calibrated
- [ ] Emergency stop tested
- [ ] Purge cycle verified
- [ ] DCS/PLC communication stable
- [ ] Operating limits configured correctly
- [ ] PID gains conservative for first run
- [ ] Manual override available
- [ ] Operators trained
- [ ] Runbooks available

---

## Monitoring Quick Start

### Prometheus Metrics
```bash
# Access metrics
curl http://localhost:8000/metrics

# Key metrics:
# - control_cycle_time_ms
# - stability_score
# - fuel_flow
# - air_flow
# - o2_level
# - heat_output
```

### Grafana Dashboard (Recommended Panels)
1. Control cycle time (line chart)
2. Stability score (gauge)
3. Fuel/air flows (dual-axis line chart)
4. O2 level (line chart with target band)
5. Heat output (line chart with target)
6. Safety interlock status (status panel)
7. Error rate (counter)

---

## Next Steps

1. **Review implementation:** Read `IMPLEMENTATION_SUMMARY.md`
2. **Configure environment:** Create `.env` file
3. **Implement calculators:** PID, stability, heat output modules
4. **Implement connectors:** DCS, PLC, analyzer integrations
5. **Write tests:** Unit and integration tests
6. **Deploy to dev:** Test in safe environment
7. **Performance tuning:** Optimize PID gains
8. **Production deployment:** After certification

---

## Support & Documentation

- **Implementation Summary:** `IMPLEMENTATION_SUMMARY.md`
- **Code Documentation:** Inline docstrings (Google style)
- **API Documentation:** http://localhost:8000/docs (Swagger UI)
- **Configuration Reference:** See `config.py` docstrings
- **Tool Schemas:** See `tools.py` TOOL_REGISTRY

---

**Quick Start Guide Version:** 1.0.0
**Agent Version:** 1.0.0
**Last Updated:** 2025-11-18
