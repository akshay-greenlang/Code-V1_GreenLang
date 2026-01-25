# ISA 18.2 Alarm Management System for Process Heat Agents

## Overview

This module implements a production-grade **ISA-18.2-2016 compliant alarm management system** specifically designed for GreenLang Process Heat agents. It provides complete alarm lifecycle management with advanced safety features.

**ISA Standard Reference:** ISA-18.2-2016 - Management of Alarms and Events for the Process Industries

## Key Features

### 1. Five-Level Priority System (ISA 18.2 Section 4.3)
- **Emergency** (1-2 sec response): Immediate action required
- **High** (5-10 sec response): Prompt action required
- **Medium** (30-60 sec response): Timely action required
- **Low** (no deadline): Awareness only
- **Diagnostic** (information only): Not for operator display

### 2. Alarm State Machine
```
NORMAL
  ↓
UNACKNOWLEDGED (alarm triggers)
  ↓ (operator acknowledges)
ACKNOWLEDGED
  ↓ (condition clears)
CLEARED

Alternative paths:
UNACKNOWLEDGED → SHELVED (temporary suppression)
Any state → STALE (standing >1 hour)
```

### 3. ISA 18.2 Metrics (Section 5.1)
- **Alarms per operator per 10 minutes** (target <10)
- **% acknowledged within 10 minutes** (target >90%)
- **Standing alarm count** (unacknowledged alarms, target 0)
- **Stale alarm count** (standing >1 hour, target 0)
- **Alarm flood detection** (>10 alarms/10 min, target 0)
- **Chattering detection** (fleeting alarms <1 sec, target 0)

### 4. Alarm Rationalization (Annex D)
Per ISA 18.2 Annex D, every alarm must be documented with:
- Consequence: What happens if not addressed?
- Response: What should the operator do?
- Response Time: Target action time in seconds

### 5. Advanced Safety Features
- **Deadband Hysteresis**: Prevents alarm chatter with configurable deadband
- **Alarm Flood Suppression**: Automatically detects and suppresses floods
- **Nuisance Alarm Suppression**: Suppress known false alarms
- **Chattering Detection**: Identifies fleeting alarms for investigation
- **Thread-Safe Operation**: Multi-operator environments
- **Audit Trail**: SHA-256 provenance hashing for compliance

## Installation

```python
from greenlang.safety.isa_18_2_alarms import (
    AlarmManager,
    AlarmPriority,
    AlarmState,
    AlarmType,
    AlarmMetrics,
)
```

## Usage Examples

### Basic Setup

```python
from greenlang.safety.isa_18_2_alarms import AlarmManager, AlarmPriority
from datetime import datetime

# Initialize manager
manager = AlarmManager(config={
    'operator_id': 'OP-001',
    'plant_id': 'PLANT-01'
})

# Configure high temperature alarm
manager.configure_alarm(
    tag='FURNACE_TEMP_01',
    description='Furnace Temperature High',
    priority=AlarmPriority.HIGH,
    setpoint=450.0,           # Alarm triggers at this value
    deadband=5.0,             # Clears when below (450-5)=445
    units='degC'
)
```

### Process Alarm Events

```python
# Process a measured value
result = manager.process_alarm(
    tag='FURNACE_TEMP_01',
    value=460.0,              # Value exceeds setpoint
    timestamp=datetime.now()
)

if result.alarm_triggered:
    print(f"NEW ALARM: {result.alarm_id}")

if result.alarm_cleared:
    print(f"Alarm cleared: {result.alarm_id}")

if result.flooded:
    print("WARNING: Alarm flood detected - operator overloaded")

if result.chattering:
    print("WARNING: Chattering alarm detected - needs investigation")
```

### Acknowledge Alarms

```python
# Operator acknowledges alarm
acknowledged = manager.acknowledge_alarm(
    alarm_id=alarm_result.alarm_id,
    operator_id='OP-001'
)

print(f"Acknowledged in {acknowledged.ack_time_sec:.1f} seconds")
```

### Shelve Nuisance Alarms

```python
# Temporarily suppress known nuisance alarm (max 24 hours)
manager.shelve_alarm(
    alarm_id=alarm_id,
    duration_hours=4,
    reason='Scheduled maintenance - burner cleaning'
)
```

### Rationalize Alarms

```python
# Document why this alarm is necessary (ISA 18.2 Annex D)
rationalization = manager.rationalize_alarm(
    tag='FURNACE_TEMP_01',
    consequence='Furnace temperature runaway leads to equipment damage and potential fire',
    response='Reduce fuel input rate and verify burner control system responsiveness',
    response_time_sec=30
)
```

### Query Standing Alarms

```python
# Get all unacknowledged alarms (sorted by priority)
standing = manager.get_standing_alarms()

for alarm in standing:
    print(f"{alarm.priority.value}: {alarm.tag} = {alarm.value}")
    print(f"  Since: {alarm.timestamp}")
    if alarm.ack_time_sec:
        print(f"  Ack time: {alarm.ack_time_sec:.1f}s")
```

### Get Metrics (ISA 18.2 Section 5.1)

```python
metrics = manager.get_alarm_metrics()

print(f"Alarms per 10 min: {metrics.alarms_per_10min:.1f} (target <10)")
print(f"Ack rate (10 min): {metrics.ack_rate_10min_pct:.1f}% (target >90%)")
print(f"Standing alarms: {metrics.standing_alarm_count} (target 0)")
print(f"Stale alarms: {metrics.stale_alarm_count} (target 0)")
print(f"Operator burden: {metrics.operator_burden}")
print(f"Rationalization: {metrics.rationalization_completeness_pct:.1f}%")

if metrics.chattering_alarms:
    print(f"Chattering alarms: {metrics.chattering_alarms}")
```

### Flood Detection

```python
# Check for alarm flood
is_flooded, counts = manager.check_alarm_flood(
    threshold=10,        # More than 10 alarms = flood
    window_minutes=10    # In this time window
)

if is_flooded:
    logger.critical("ALARM FLOOD: Operator overloaded, suppressing lower-priority alarms")
    # System should reduce operator burden by suppressing LOW/DIAGNOSTIC alarms
```

## Process Heat Integration Example

```python
from greenlang.safety.isa_18_2_alarms import AlarmManager, AlarmPriority, AlarmType

# Setup manager
alarm_mgr = AlarmManager(config={
    'operator_id': 'OP-HEAT-01',
    'plant_id': 'PLANT-HEAT-01'
})

# Configure typical process heat alarms
furnace_config = [
    ('FURNACE_TEMP', 'Furnace Temperature High', 450, 5, AlarmPriority.HIGH),
    ('STACK_TEMP', 'Stack Temperature High', 350, 10, AlarmPriority.MEDIUM),
    ('FUEL_PRESS', 'Fuel Pressure Low', 3.0, 0.5, AlarmPriority.MEDIUM),
    ('EXHAUST_FLOW', 'Exhaust Flow Low', 100, 15, AlarmPriority.HIGH),
]

for tag, desc, setpt, db, priority in furnace_config:
    alarm_mgr.configure_alarm(
        tag=tag,
        description=desc,
        priority=priority,
        setpoint=setpt,
        deadband=db
    )

# Rationalize all alarms
alarm_mgr.rationalize_alarm(
    tag='FURNACE_TEMP',
    consequence='Temperature runaway -> burner damage and safety risk',
    response='Reduce fuel supply immediately',
    response_time_sec=10
)

# Process sensor readings in your agent loop
def process_heat_sensors(sensor_data: Dict):
    """Process sensor data and trigger alarms as needed."""
    for tag, value in sensor_data.items():
        result = alarm_mgr.process_alarm(
            tag=tag,
            value=value
        )

        if result.alarm_triggered:
            print(f"NEW ALARM: {tag} = {value}")

        if result.flooded:
            # Handle operator overload
            pass

# Check operator metrics
metrics = alarm_mgr.get_alarm_metrics()
if metrics.operator_burden == 'CRITICAL':
    logger.critical(f"Operator burden critical: {metrics.alarms_per_10min} alarms/10min")
```

## Deadband Hysteresis Behavior

Deadband prevents alarm chatter by using hysteresis:

```
Setpoint = 450°C, Deadband = 5°C

Alarm TRIGGERS when:  value >= 450
Alarm CLEARS when:   value < 445  (450 - 5)

Scenario:
  T=0s:  value=440 → NORMAL
  T=5s:  value=455 → UNACKNOWLEDGED (triggers)
  T=10s: value=448 → ACKNOWLEDGED (still active)
  T=15s: value=446 → ACKNOWLEDGED (still active)
  T=20s: value=444 → CLEARED (below 445)
```

## Alarm Flood Behavior

When > 10 alarms occur in 10 minutes:

1. Flood is detected
2. Lower-priority alarms are suppressed
3. Only EMERGENCY and HIGH priority alarms remain visible
4. Log entry created for investigation

```python
# Typical flood scenario
is_flooded, counts = manager.check_alarm_flood()

# counts returns:
# {
#     AlarmPriority.EMERGENCY: 2,
#     AlarmPriority.HIGH: 5,
#     AlarmPriority.MEDIUM: 8,  # These would be suppressed
#     AlarmPriority.LOW: 4,      # These would be suppressed
#     AlarmPriority.DIAGNOSTIC: 0
# }
```

## Chattering Alarm Detection

Chattering = alarm triggers and clears in < 1 second:

```
T=0ms: value=455 → UNACKNOWLEDGED
T=500ms: value=440 → CLEARED
T=1500ms: value=455 → Marked as CHATTERING
```

Action: Investigate and adjust deadband or sensor damping

## Operator Performance Targets (ISA 18.2)

| Metric | Target | Threshold |
|--------|--------|-----------|
| Alarms per 10 min | < 10 | > 20 = CRITICAL |
| Acknowledgment rate | > 90% | < 50% = WARNING |
| Standing alarms | 0 | > 5 = WARNING |
| Stale alarms | 0 | > 0 = CRITICAL |
| Alarm floods | 0 | > 0 = CRITICAL |
| Chattering alarms | 0 | > 1 = WARNING |
| Rationalization | 100% | < 80% = WARNING |

## Thread Safety

AlarmManager is thread-safe for multi-operator environments:

```python
import threading

# Multiple operators can acknowledge simultaneously
threads = []
for operator_id in ['OP-001', 'OP-002', 'OP-003']:
    t = threading.Thread(
        target=manager.acknowledge_alarm,
        args=(alarm_id, operator_id)
    )
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```

## Configuration Best Practices

### Setpoint and Deadband Selection

```python
# Temperature sensor with ±1°C accuracy
manager.configure_alarm(
    tag='TEMP_01',
    priority=AlarmPriority.HIGH,
    setpoint=450.0,
    deadband=2.0  # 2x sensor accuracy to avoid chatter
)

# Pressure sensor with ±0.1 bar accuracy
manager.configure_alarm(
    tag='PRESS_01',
    priority=AlarmPriority.MEDIUM,
    setpoint=5.0,
    deadband=0.2  # 2x sensor accuracy
)
```

### Priority Assignment

```python
# EMERGENCY: Equipment damage, safety risk, immediate shutdown
# Example: Furnace temperature runaway
manager.configure_alarm(
    tag='FURNACE_TEMP_MAX',
    priority=AlarmPriority.EMERGENCY,
    setpoint=550.0
)

# HIGH: Process upset, manual intervention needed
# Example: Burner pressure low
manager.configure_alarm(
    tag='BURNER_PRESS_LOW',
    priority=AlarmPriority.HIGH,
    setpoint=3.0
)

# MEDIUM: Trend heading wrong direction
# Example: Gradual temperature rise
manager.configure_alarm(
    tag='FURNACE_TEMP_HIGH',
    priority=AlarmPriority.MEDIUM,
    setpoint=450.0
)

# LOW: Informational, no action needed
# Example: Equipment efficiency dropping
manager.configure_alarm(
    tag='EFFICIENCY_LOW',
    priority=AlarmPriority.LOW,
    setpoint=75.0
)

# DIAGNOSTIC: Debug information only
manager.configure_alarm(
    tag='SENSOR_READING',
    priority=AlarmPriority.DIAGNOSTIC,
    setpoint=0.0
)
```

## Integration with Process Heat Agents

```python
from greenlang.agents.process_heat.furnace_agent import FurnaceAgent
from greenlang.safety.isa_18_2_alarms import AlarmManager

class EnhancedFurnaceAgent(FurnaceAgent):
    def __init__(self, config):
        super().__init__(config)
        self.alarm_manager = AlarmManager(config={
            'operator_id': config.get('operator_id'),
            'plant_id': config.get('plant_id')
        })
        self._configure_alarms()

    def _configure_alarms(self):
        """Configure furnace-specific alarms."""
        self.alarm_manager.configure_alarm(
            tag='FURNACE_TEMP',
            description='Furnace Temperature High',
            priority=AlarmPriority.HIGH,
            setpoint=self.config['temp_setpoint'],
            deadband=self.config['temp_deadband']
        )

    def process_sensors(self, sensor_data):
        """Process sensor data with alarm management."""
        # Process furnace logic
        results = super().process_sensors(sensor_data)

        # Check alarms
        for tag, value in sensor_data.items():
            alarm_result = self.alarm_manager.process_alarm(tag, value)
            if alarm_result.alarm_triggered:
                self.logger.warning(f"Alarm triggered: {tag}")

        # Get metrics
        metrics = self.alarm_manager.get_alarm_metrics()
        results['alarm_metrics'] = metrics

        return results
```

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Configure alarm | <1ms | Stored in dict |
| Process alarm | <1ms | O(1) lookup, minimal computation |
| Acknowledge alarm | <1ms | Update state in place |
| Get metrics | <10ms | Iterates recent events (max 10k) |
| Check flood | <5ms | Scans time window |
| Thread-safe lock | <1us | RLock acquisition |

## Testing

37 unit tests covering:
- Configuration (5 tests)
- Processing (6 tests)
- Acknowledgment (3 tests)
- Shelving (3 tests)
- Rationalization (2 tests)
- Flood detection (3 tests)
- Chattering detection (1 test)
- Queries (3 tests)
- Metrics (7 tests)
- Thread safety (1 test)
- Integration (3 tests)

Run with:
```bash
pytest tests/unit/test_isa_18_2_alarms.py -v
```

## Reference Documents

- **ISA-18.2-2016**: Management of Alarms and Events for the Process Industries
  - Section 4.3: Alarm Priority Assignment
  - Section 5.1: Operator Performance Assessment
  - Annex C: Alarm Rationalization Process
  - Annex D: Operator Performance Metrics

- **IEC 61511-1:2016**: Functional Safety - Safety Instrumented Systems
  - Safety layer integration with alarm management

- **IEC 61508**: Functional Safety of Electrical/Electronic/Programmable Safety-Related Systems
  - Fault tolerance and diagnostic coverage calculations

## Common Issues and Solutions

### Alarm Chatter
**Problem:** Alarm triggers and clears repeatedly
**Solution:** Increase deadband or add sensor damping filter

### Operator Overload (Alarm Flood)
**Problem:** >10 alarms per 10 minutes
**Solution:** Rationalize alarms, adjust setpoints, suppress nuisance alarms

### Stale Alarms
**Problem:** Unacknowledged alarm standing >1 hour
**Solution:** Investigate root cause, adjust setpoint, or suppress if nuisance

### High False Alarm Rate
**Problem:** Many DIAGNOSTIC alarms not actionable
**Solution:** Remove non-critical alarms, reduce to DIAGNOSTIC priority

## Compliance

This implementation follows:
- ISA-18.2-2016 standard completely
- IEC 61511 functional safety guidelines
- EEMUA 191 alarm management best practices
- GreenLang safety architecture standards

## Author and License

**Author:** GreenLang Safety Engineering Team
**License:** Proprietary - GreenLang Software
**Version:** 1.0.0
**Last Updated:** December 2024
