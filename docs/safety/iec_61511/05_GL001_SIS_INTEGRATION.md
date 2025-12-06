# GL-001 SIS Integration Specification

## Thermal Command Agent - Safety Instrumented System Interface

**Document ID:** GL-SIL-INT-001
**Version:** 1.0
**Effective Date:** 2025-12-05
**Classification:** Safety Critical Documentation
**Standard Reference:** IEC 61511-1:2016, IEC 62443

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [SIS Interface Specification](#3-sis-interface-specification)
4. [Safety Interlocks](#4-safety-interlocks)
5. [Voting Logic Implementation](#5-voting-logic-implementation)
6. [Response Time Analysis](#6-response-time-analysis)
7. [Safe State Definitions](#7-safe-state-definitions)
8. [Communication Protocol](#8-communication-protocol)
9. [Cybersecurity Requirements](#9-cybersecurity-requirements)
10. [Commissioning Requirements](#10-commissioning-requirements)

---

## 1. Introduction

### 1.1 Purpose

This document specifies the integration requirements between the GL-001 Thermal Command Agent and the Safety Instrumented System (SIS) for process heat applications. It defines the interfaces, interlocks, voting logic, and safe states that enable GL-001 to achieve SIL 2 functional safety.

### 1.2 Scope

This specification covers:

- GL-001 to SIS communication interfaces
- Safety interlock definitions and logic
- Response time budgets
- Safe state definitions
- Commissioning and validation requirements

### 1.3 System Context

```
+------------------------------------------------------------------+
|                    PROCESS HEAT SYSTEM                            |
+------------------------------------------------------------------+
|                                                                   |
|  +----------------+     +----------------+     +----------------+ |
|  |    SENSORS     |     |      SIS       |     |  FINAL ELEM.   | |
|  | TE, FT, PT, FS |---->| Safety Logic   |---->| XV, PV, Pumps  | |
|  +----------------+     +----------------+     +----------------+ |
|         |                      ^                      |          |
|         |                      |                      |          |
|         v                      v                      v          |
|  +----------------------------------------------------------+   |
|  |                     GL-001 AGENT                          |   |
|  |              Thermal Command & Monitoring                 |   |
|  +----------------------------------------------------------+   |
|         |                      |                      |          |
|         v                      v                      v          |
|  +----------------+     +----------------+     +----------------+ |
|  |      BPCS      |     |      HMI       |     |   HISTORIAN    | |
|  | Process Control|     |   Operator     |     |  Data Logging  | |
|  +----------------+     +----------------+     +----------------+ |
|                                                                   |
+------------------------------------------------------------------+
```

### 1.4 Reference Documents

| Document | Title |
|----------|-------|
| GL-SIL-OV-001 | SIL Certification Overview |
| GL-SIL-SRS-001 | Safety Requirements Specification |
| GL-SIL-VOTE-001 | Voting Logic Specification |
| GL-SIL-PT-001 | Proof Test Procedures |

---

## 2. System Architecture

### 2.1 Overall Architecture

```
+===========================================================================+
||                          LEVEL 4: ENTERPRISE                            ||
||  +------------------+                                                   ||
||  | Business Systems |                                                   ||
||  +------------------+                                                   ||
+===========================================================================+
                                    |
                              [Firewall]
                                    |
+===========================================================================+
||                          LEVEL 3: OPERATIONS                            ||
||  +------------------+     +------------------+     +------------------+ ||
||  |    Historian     |     |   Engineering    |     |   GL-007 Agent   | ||
||  |                  |     |   Workstation    |     |   (Compliance)   | ||
||  +------------------+     +------------------+     +------------------+ ||
+===========================================================================+
                                    |
                              [DMZ/Firewall]
                                    |
+===========================================================================+
||                          LEVEL 2: SUPERVISORY                           ||
||  +------------------+     +------------------+     +------------------+ ||
||  |      HMI         |     |   GL-001 Agent   |     |   GL-005 Agent   | ||
||  |   Operator       |     |  Thermal Command |     | Building Energy  | ||
||  +------------------+     +------------------+     +------------------+ ||
+===========================================================================+
                                    |
                          [Segregated Network]
                                    |
+===========================================================================+
||                          LEVEL 1: CONTROL                               ||
||  +------------------+          +------------------+                     ||
||  |      BPCS        |<-------->|       SIS        |                    ||
||  |  Process Control |  (Hard-  | Safety Logic     |                    ||
||  +------------------+  wired)  +------------------+                     ||
+===========================================================================+
                                    |
                          [Field Network/Hardwired]
                                    |
+===========================================================================+
||                          LEVEL 0: FIELD                                 ||
||  +--------+  +--------+  +--------+  +--------+  +--------+  +--------+||
||  |  TE-x  |  |  FT-x  |  |  PT-x  |  |  FS-x  |  |  XV-x  |  |  PV-x  |||
||  | Temp   |  | Flow   |  | Press  |  | Flame  |  | Fuel   |  | Proc   |||
||  +--------+  +--------+  +--------+  +--------+  +--------+  +--------+||
+===========================================================================+
```

### 2.2 GL-001 Agent Architecture

```
+------------------------------------------------------------------------+
|                          GL-001 THERMAL COMMAND AGENT                   |
+------------------------------------------------------------------------+
|                                                                         |
|  +---------------------------+     +---------------------------+        |
|  |     INPUT PROCESSING      |     |    SAFETY MONITORING      |        |
|  +---------------------------+     +---------------------------+        |
|  | - Temperature inputs      |     | - High temp detection     |        |
|  | - Flow inputs             |     | - Low flow detection      |        |
|  | - Pressure inputs         |     | - Over-pressure detection |        |
|  | - Flame status            |     | - Flame failure detection |        |
|  | - Valve positions         |     | - Voting logic execution  |        |
|  +---------------------------+     +---------------------------+        |
|            |                                    |                       |
|            v                                    v                       |
|  +---------------------------+     +---------------------------+        |
|  |    CONTROL FUNCTIONS      |     |    SIS INTERFACE          |        |
|  +---------------------------+     +---------------------------+        |
|  | - Temperature control     |     | - Trip command output     |        |
|  | - Flow control            |     | - Safe state management   |        |
|  | - Pressure regulation     |     | - Reset handling          |        |
|  | - Combustion optimization |     | - Status feedback         |        |
|  +---------------------------+     +---------------------------+        |
|            |                                    |                       |
|            v                                    v                       |
|  +---------------------------+     +---------------------------+        |
|  |    OUTPUT PROCESSING      |     |    ALARM & HMI            |        |
|  +---------------------------+     +---------------------------+        |
|  | - BPCS commands           |     | - Alarm generation        |        |
|  | - Modulating outputs      |     | - Event logging           |        |
|  | - Status indication       |     | - Operator display        |        |
|  +---------------------------+     +---------------------------+        |
|                                                                         |
+------------------------------------------------------------------------+
```

### 2.3 Hardware Configuration

| Component | Type | Location | SIL Capability |
|-----------|------|----------|----------------|
| Safety PLC | TUV Certified (e.g., Siemens S7-400F, ABB AC800M HI) | Control room | SIL 3 |
| GL-001 Server | Industrial server, redundant | Control room | N/A (non-safety) |
| Network Switch | Industrial Ethernet, managed | Control room | N/A |
| Field I/O | Redundant I/O modules | Field junction box | Per SIL 2 |

---

## 3. SIS Interface Specification

### 3.1 Interface Overview

GL-001 interfaces with the SIS in three ways:

1. **Read-only safety data** - GL-001 reads sensor data and SIS status
2. **Trip initiation** - GL-001 can request SIS trip (non-certified path)
3. **Operator interface** - GL-001 provides HMI for SIS functions

```
+----------------+                              +----------------+
|                |  Read-Only Data (OPC/Modbus) |                |
|    GL-001      |<-----------------------------|      SIS       |
|                |                              |                |
|                |  Trip Request (Discrete)     |                |
|                |----------------------------->|                |
|                |                              |                |
|                |  Hardwired SIS Outputs       |                |
|                |<=============================>                |
+----------------+                              +----------------+
        |                                              ^
        v                                              |
+----------------+                              +----------------+
|      HMI       |   Alarm/Status Display       |  Field Devices |
|   (Operator)   |<-----------------------------|  (Sensors/     |
|                |                              |   Actuators)   |
+----------------+                              +----------------+
```

### 3.2 Data Exchange Specification

#### 3.2.1 SIS to GL-001 (Read-Only)

| Tag | Description | Data Type | Update Rate | Safety Classification |
|-----|-------------|-----------|-------------|----------------------|
| TE001A_PV | Temperature Sensor A | REAL | 100 ms | SIL 2 |
| TE001B_PV | Temperature Sensor B | REAL | 100 ms | SIL 2 |
| FT001A_PV | Flow Sensor A | REAL | 100 ms | SIL 2 |
| FT001B_PV | Flow Sensor B | REAL | 100 ms | SIL 2 |
| PT001A_PV | Pressure Sensor A | REAL | 100 ms | SIL 2 |
| PT001B_PV | Pressure Sensor B | REAL | 100 ms | SIL 2 |
| FS001A_STATUS | Flame Scanner A | BOOL | 100 ms | SIL 2 |
| FS001B_STATUS | Flame Scanner B | BOOL | 100 ms | SIL 2 |
| FS001C_STATUS | Flame Scanner C | BOOL | 100 ms | SIL 2 |
| XV001A_POS | Fuel Valve A Position | BOOL | 100 ms | SIL 2 |
| XV001B_POS | Fuel Valve B Position | BOOL | 100 ms | SIL 2 |
| SIS_TRIP_STATUS | SIS Trip Active | BOOL | 100 ms | SIL 2 |
| SIS_HEALTHY | SIS Health Status | BOOL | 1000 ms | Non-safety |

#### 3.2.2 GL-001 to SIS (Trip Request)

| Tag | Description | Data Type | Action | Safety Classification |
|-----|-------------|-----------|--------|----------------------|
| GL001_TRIP_REQ | Trip Request from GL-001 | BOOL | Rising edge triggers trip | Non-certified |
| GL001_HEARTBEAT | Watchdog signal | BOOL | Toggle every 500 ms | Non-safety |

**Note:** GL-001 trip request is NOT the certified safety path. The SIS must execute the SIF based on its own sensor inputs. GL-001 trip request provides defense-in-depth only.

### 3.3 Communication Protocol

#### 3.3.1 Primary Protocol: OPC UA

| Parameter | Specification |
|-----------|---------------|
| Protocol | OPC UA (IEC 62541) |
| Security | Sign and Encrypt |
| Authentication | X.509 certificate |
| Session timeout | 30 seconds |
| Publishing interval | 100 ms |
| Keep-alive | 5 seconds |

#### 3.3.2 Backup Protocol: Modbus TCP

| Parameter | Specification |
|-----------|---------------|
| Protocol | Modbus TCP/IP |
| Port | 502 |
| Timeout | 2 seconds |
| Retry | 3 attempts |
| Register mapping | Per GL-001 register map |

### 3.4 Interface Timing

```
+------------+     +------------+     +------------+
| SIS Scan   |---->| Network    |---->| GL-001     |
| (10 ms)    |     | (50 ms max)|     | (100 ms)   |
+------------+     +------------+     +------------+
                                             |
                                             v
+------------+     +------------+     +------------+
| SIS Trip   |<----| Network    |<----| GL-001     |
| (immediate)|     | (50 ms max)|     | Response   |
+------------+     +------------+     +------------+
```

| Path | Maximum Latency |
|------|-----------------|
| SIS to GL-001 data | 60 ms |
| GL-001 trip request to SIS | 60 ms |
| Total round-trip | 120 ms |

---

## 4. Safety Interlocks

### 4.1 Interlock Summary

| Interlock ID | Description | Initiating Condition | Action | SIF Reference |
|--------------|-------------|---------------------|--------|---------------|
| IL-001 | High Temperature Shutdown | T >= THH (380C) | Fuel isolation, process shutdown | SIF-001 |
| IL-002 | Low Flow Protection | F <= FLL (10%) after 5s | Heat source isolation | SIF-002 |
| IL-003 | High Pressure Protection | P >= PHH (95%) | Depressurization, isolation | SIF-003 |
| IL-004 | Flame Failure Shutdown | 2oo3 flame loss | Fuel isolation, purge | SIF-004 |
| IL-005 | Manual ESD | Pushbutton | Full system shutdown | SIF-005 |
| IL-006 | External ESD | Plant ESD signal | Full system shutdown | SIF-005 |

### 4.2 Interlock Logic Diagrams

#### 4.2.1 IL-001: High Temperature Shutdown

```
                    +-------+
TE-001A >= THH ---->|       |
                    | 2oo2  |----+
TE-001B >= THH ---->|       |    |    +-------+
                    +-------+    +--->|       |
                                      |  AND  |---> XV-001 CLOSE
                       +-------+  +-->|       |
                       | Enable|--+   +-------+
                       | (Run) |
                       +-------+
```

**GL-001 Implementation:**

```python
class HighTemperatureInterlock:
    """IL-001: High Temperature Shutdown Logic"""

    THH_SETPOINT = 380.0  # Celsius

    def evaluate(self, te_001a: float, te_001b: float,
                 system_enabled: bool) -> bool:
        """
        Evaluate 2oo2 voting for high temperature trip.

        Returns:
            True if trip condition met, False otherwise
        """
        sensor_a_high = te_001a >= self.THH_SETPOINT
        sensor_b_high = te_001b >= self.THH_SETPOINT

        # 2oo2 voting: both sensors must agree
        trip_condition = sensor_a_high and sensor_b_high

        # Only trip if system is enabled (running)
        return trip_condition and system_enabled
```

#### 4.2.2 IL-002: Low Flow Protection

```
                    +-------+
FT-001A <= FLL ---->|       |
                    | 2oo2  |----+
FT-001B <= FLL ---->|       |    |    +-------+     +-------+
                    +-------+    +--->|       |     |       |
                                      |  AND  |---->| Timer |---> TRIP
                       +-------+  +-->|       |     | (5s)  |
                       | Enable|--+   +-------+     +-------+
                       | (Run) |
                       +-------+
```

**GL-001 Implementation:**

```python
class LowFlowInterlock:
    """IL-002: Low Flow Protection Logic"""

    FLL_SETPOINT = 10.0  # Percent of span
    TIME_DELAY = 5.0     # Seconds

    def __init__(self):
        self.low_flow_timer = 0.0
        self.trip_latched = False

    def evaluate(self, ft_001a: float, ft_001b: float,
                 system_enabled: bool, dt: float) -> bool:
        """
        Evaluate 2oo2 voting with time delay for low flow trip.

        Args:
            ft_001a: Flow sensor A reading (%)
            ft_001b: Flow sensor B reading (%)
            system_enabled: System running state
            dt: Time delta since last evaluation (seconds)

        Returns:
            True if trip condition met after time delay
        """
        sensor_a_low = ft_001a <= self.FLL_SETPOINT
        sensor_b_low = ft_001b <= self.FLL_SETPOINT

        # 2oo2 voting
        low_flow_condition = sensor_a_low and sensor_b_low and system_enabled

        if low_flow_condition:
            self.low_flow_timer += dt
            if self.low_flow_timer >= self.TIME_DELAY:
                self.trip_latched = True
        else:
            self.low_flow_timer = 0.0
            # Note: trip_latched requires manual reset

        return self.trip_latched
```

#### 4.2.3 IL-004: Flame Failure Detection

```
                    +-------+
FS-001A = OFF ----->|       |
                    |       |
FS-001B = OFF ----->| 2oo3  |----> TRIP
                    |       |
FS-001C = OFF ----->|       |
                    +-------+
```

**GL-001 Implementation:**

```python
class FlameFailureInterlock:
    """IL-004: Flame Failure Detection Logic"""

    FLAME_FAILURE_DELAY = 1.0  # Second (per NFPA)

    def __init__(self):
        self.flame_loss_timer = 0.0
        self.trip_latched = False

    def evaluate(self, fs_001a: bool, fs_001b: bool,
                 fs_001c: bool, burner_enabled: bool,
                 dt: float) -> bool:
        """
        Evaluate 2oo3 voting for flame failure.

        Args:
            fs_001a, fs_001b, fs_001c: Flame scanner status
                                       (True = flame, False = no flame)
            burner_enabled: Burner firing state
            dt: Time delta (seconds)

        Returns:
            True if flame failure trip condition met
        """
        # Count flame signals present
        flame_count = sum([fs_001a, fs_001b, fs_001c])

        # 2oo3: Trip if fewer than 2 scanners see flame
        flame_loss = flame_count < 2

        if flame_loss and burner_enabled:
            self.flame_loss_timer += dt
            if self.flame_loss_timer >= self.FLAME_FAILURE_DELAY:
                self.trip_latched = True
        else:
            self.flame_loss_timer = 0.0

        return self.trip_latched
```

### 4.3 Interlock Priority

| Priority | Interlock | Reason |
|----------|-----------|--------|
| 1 (Highest) | IL-004 Flame Failure | Explosion risk |
| 2 | IL-003 High Pressure | Vessel integrity |
| 3 | IL-001 High Temperature | Equipment damage |
| 4 | IL-002 Low Flow | Thermal damage |
| 5 | IL-005/006 ESD | Manual/external override |

### 4.4 Interlock Permissives

| Interlock | Startup Permissives | Running Permissives |
|-----------|---------------------|---------------------|
| IL-001 | T < THH - 20C | Continuous |
| IL-002 | F > FLL + 10% | Continuous |
| IL-003 | P < PHH - 10% | Continuous |
| IL-004 | Purge complete, ignition sequence | Continuous |
| IL-005/006 | Reset key, no faults | Continuous |

---

## 5. Voting Logic Implementation

### 5.1 GL-001 Voting Module

```python
"""
GL-001 SIS Voting Logic Module

This module implements the voting logic for Safety Instrumented Functions.
All voting decisions are performed by the certified SIS; this module
provides monitoring and HMI display functions only.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional
import time


class VotingArchitecture(Enum):
    """Voting architecture types"""
    OO1 = "1oo1"  # Single channel
    OO2 = "1oo2"  # Either channel trips
    TWO_OO2 = "2oo2"  # Both channels must agree
    TWO_OO3 = "2oo3"  # Two of three must agree


@dataclass
class VotingStatus:
    """Status of voting logic evaluation"""
    architecture: VotingArchitecture
    trip_condition: bool
    channel_a: bool
    channel_b: Optional[bool] = None
    channel_c: Optional[bool] = None
    timestamp: float = 0.0

    def __post_init__(self):
        self.timestamp = time.time()


class VotingLogic:
    """
    Voting logic calculator for SIS monitoring.

    NOTE: This is for monitoring and display purposes only.
    The certified SIS performs actual safety voting.
    """

    @staticmethod
    def evaluate_1oo1(channel_a: bool) -> VotingStatus:
        """Single channel voting"""
        return VotingStatus(
            architecture=VotingArchitecture.OO1,
            trip_condition=channel_a,
            channel_a=channel_a
        )

    @staticmethod
    def evaluate_1oo2(channel_a: bool, channel_b: bool) -> VotingStatus:
        """Either channel trips (OR logic)"""
        return VotingStatus(
            architecture=VotingArchitecture.OO2,
            trip_condition=channel_a or channel_b,
            channel_a=channel_a,
            channel_b=channel_b
        )

    @staticmethod
    def evaluate_2oo2(channel_a: bool, channel_b: bool) -> VotingStatus:
        """Both channels must agree (AND logic)"""
        return VotingStatus(
            architecture=VotingArchitecture.TWO_OO2,
            trip_condition=channel_a and channel_b,
            channel_a=channel_a,
            channel_b=channel_b
        )

    @staticmethod
    def evaluate_2oo3(channel_a: bool, channel_b: bool,
                      channel_c: bool) -> VotingStatus:
        """Two of three channels must agree"""
        vote_count = sum([channel_a, channel_b, channel_c])
        return VotingStatus(
            architecture=VotingArchitecture.TWO_OO3,
            trip_condition=vote_count >= 2,
            channel_a=channel_a,
            channel_b=channel_b,
            channel_c=channel_c
        )
```

### 5.2 Voting Logic per SIF

| SIF | Sensor Voting | Logic | Final Element |
|-----|---------------|-------|---------------|
| SIF-001 | 2oo2 (AND) | 1oo1 | 1oo2 (OR) |
| SIF-002 | 2oo2 (AND) | 1oo1 | 1oo1 |
| SIF-003 | 1oo2 (OR) | 1oo1 | 1oo2 (OR) |
| SIF-004 | 2oo3 | 1oo1 | Series (AND) |
| SIF-005 | 1oo1 (multiple inputs OR) | 1oo1 | 1oo2 (OR) |

### 5.3 Diagnostic Voting

GL-001 monitors sensor health and provides diagnostic alarms:

```python
class SensorDiagnostics:
    """Sensor health monitoring for voting logic"""

    def check_sensor_health(self, sensor_value: float,
                           min_range: float,
                           max_range: float) -> Tuple[bool, str]:
        """
        Check sensor health based on signal range.

        Returns:
            Tuple of (healthy: bool, fault_code: str)
        """
        # Check for sensor failure (out of range)
        if sensor_value < min_range * 0.9:  # Below 4 mA equivalent
            return False, "SENSOR_OPEN"
        if sensor_value > max_range * 1.025:  # Above 20.5 mA equivalent
            return False, "SENSOR_SHORT"

        # Check for stuck sensor (rate of change too low)
        # Implementation would track historical values

        return True, "OK"

    def evaluate_redundancy(self, sensor_a: float, sensor_b: float,
                           tolerance: float = 0.05) -> Tuple[bool, str]:
        """
        Check agreement between redundant sensors.

        Args:
            sensor_a, sensor_b: Sensor readings
            tolerance: Acceptable difference as fraction of span

        Returns:
            Tuple of (agreement: bool, fault_code: str)
        """
        difference = abs(sensor_a - sensor_b)
        span = 100.0  # Assume 0-100% span

        if difference > span * tolerance:
            return False, "SENSOR_DISAGREEMENT"

        return True, "OK"
```

---

## 6. Response Time Analysis

### 6.1 Response Time Budget

For SIF-001 (High Temperature Shutdown) - Representative of SIL 2 functions:

```
+------------------+------------------+------------------+
|    COMPONENT     |   TIME BUDGET    |    MEASURED      |
+------------------+------------------+------------------+
| Sensor response  |     100 ms       |                  |
| A/D conversion   |      10 ms       |                  |
| SIS scan time    |      10 ms       |                  |
| Voting logic     |      10 ms       |                  |
| Output processing|      10 ms       |                  |
| Solenoid response|      50 ms       |                  |
| Valve stroke     |     200 ms       |                  |
| Safety margin    |     110 ms       |                  |
+------------------+------------------+------------------+
| TOTAL            |     500 ms       |                  |
+------------------+------------------+------------------+
```

### 6.2 GL-001 Processing Time

GL-001 is not in the certified safety path, but monitors for display:

| Function | Target | Maximum |
|----------|--------|---------|
| Data acquisition | 50 ms | 100 ms |
| Safety calculations | 20 ms | 50 ms |
| HMI update | 100 ms | 200 ms |
| Alarm processing | 50 ms | 100 ms |

### 6.3 Watchdog Configuration

| Parameter | Setting | Action on Timeout |
|-----------|---------|-------------------|
| SIS communication watchdog | 2 seconds | Alarm, failsafe |
| GL-001 internal watchdog | 500 ms | Restart, alarm |
| HMI communication watchdog | 5 seconds | Alarm only |

### 6.4 Response Time Verification

```python
class ResponseTimeMonitor:
    """Monitor and log SIS response times"""

    def __init__(self, sif_id: str, max_response_time_ms: float):
        self.sif_id = sif_id
        self.max_response_time = max_response_time_ms
        self.measurements = []

    def record_response(self, trigger_time: float,
                       response_time: float) -> bool:
        """
        Record a measured response time.

        Args:
            trigger_time: Time when trip condition occurred
            response_time: Time when safe state achieved

        Returns:
            True if within specification
        """
        elapsed_ms = (response_time - trigger_time) * 1000
        self.measurements.append({
            'timestamp': trigger_time,
            'response_ms': elapsed_ms,
            'in_spec': elapsed_ms <= self.max_response_time
        })

        if elapsed_ms > self.max_response_time:
            # Log alarm
            self._alarm_response_time_exceeded(elapsed_ms)
            return False

        return True

    def _alarm_response_time_exceeded(self, measured_ms: float):
        """Generate alarm for response time exceedance"""
        # Implementation would log to historian and alarm system
        pass
```

---

## 7. Safe State Definitions

### 7.1 Safe State Matrix

| SIF | Component | Safe State | Physical State | Fail Position |
|-----|-----------|------------|----------------|---------------|
| SIF-001 | XV-001A (Fuel) | CLOSED | De-energized | Spring close |
| SIF-001 | XV-001B (Fuel) | CLOSED | De-energized | Spring close |
| SIF-001 | Cooling Pump | RUNNING | Energized | Run on backup |
| SIF-002 | XV-002 (Heat source) | CLOSED | De-energized | Spring close |
| SIF-003 | PV-001A (Vent) | OPEN | De-energized | Spring open |
| SIF-003 | PV-001B (Isolation) | CLOSED | De-energized | Spring close |
| SIF-004 | XV-004A (Fuel block) | CLOSED | De-energized | Spring close |
| SIF-004 | XV-004B (Fuel block) | CLOSED | De-energized | Spring close |
| SIF-004 | XV-004C (Vent) | OPEN | Energized | Spring close |

### 7.2 Safe State Transition Diagram

```
                         +---------------+
                         |    NORMAL     |
                         |   OPERATION   |
                         +-------+-------+
                                 |
            Trip Condition Detected
                                 |
                         +-------v-------+
                         |   TRIPPING    |
                         | (Transitional)|
                         +-------+-------+
                                 |
            Safe State Achieved (<500ms)
                                 |
                         +-------v-------+
                         |  SAFE STATE   |
                         |  (Shutdown)   |
                         +-------+-------+
                                 |
            Manual Reset + Permissives OK
                                 |
                         +-------v-------+
                         |   STARTING    |
                         | (Permissives) |
                         +-------+-------+
                                 |
            Startup Complete
                                 |
                         +-------v-------+
                         |    NORMAL     |
                         |   OPERATION   |
                         +---------------+
```

### 7.3 GL-001 Safe State Management

```python
from enum import Enum

class SystemState(Enum):
    """GL-001 system operational states"""
    NORMAL = "NORMAL"
    TRIPPING = "TRIPPING"
    SAFE_STATE = "SAFE_STATE"
    STARTING = "STARTING"
    MAINTENANCE = "MAINTENANCE"
    FAULT = "FAULT"


class SafeStateManager:
    """Manages safe state transitions for GL-001"""

    def __init__(self):
        self.current_state = SystemState.SAFE_STATE  # Start safe
        self.trip_reason = None
        self.trip_timestamp = None

    def process_trip(self, sif_id: str, reason: str) -> SystemState:
        """
        Process a trip event.

        Args:
            sif_id: SIF that initiated trip
            reason: Trip reason description

        Returns:
            New system state
        """
        self.current_state = SystemState.TRIPPING
        self.trip_reason = f"{sif_id}: {reason}"
        self.trip_timestamp = time.time()

        # Log trip event
        self._log_trip_event()

        return self.current_state

    def confirm_safe_state(self, valve_positions: dict) -> bool:
        """
        Verify all components are in safe state.

        Args:
            valve_positions: Dict of valve tag -> position

        Returns:
            True if safe state confirmed
        """
        required_positions = {
            'XV-001A': 'CLOSED',
            'XV-001B': 'CLOSED',
            'XV-002': 'CLOSED',
            'XV-004A': 'CLOSED',
            'XV-004B': 'CLOSED'
        }

        for valve, required in required_positions.items():
            if valve_positions.get(valve) != required:
                return False

        self.current_state = SystemState.SAFE_STATE
        return True

    def request_reset(self, operator_id: str,
                     key_switch: bool) -> Tuple[bool, str]:
        """
        Process reset request.

        Args:
            operator_id: Operator requesting reset
            key_switch: Key switch position (True = ON)

        Returns:
            Tuple of (success: bool, message: str)
        """
        if self.current_state != SystemState.SAFE_STATE:
            return False, "System not in safe state"

        if not key_switch:
            return False, "Key switch not in reset position"

        # Check permissives
        if not self._check_startup_permissives():
            return False, "Startup permissives not satisfied"

        self.current_state = SystemState.STARTING
        self._log_reset_event(operator_id)

        return True, "Reset accepted, starting sequence"

    def _check_startup_permissives(self) -> bool:
        """Check all startup permissives"""
        # Implementation checks all permissive conditions
        return True

    def _log_trip_event(self):
        """Log trip event to historian"""
        pass

    def _log_reset_event(self, operator_id: str):
        """Log reset event to historian"""
        pass
```

---

## 8. Communication Protocol

### 8.1 OPC UA Configuration

```xml
<?xml version="1.0" encoding="UTF-8"?>
<UANodeSet xmlns="http://opcfoundation.org/UA/2011/03/UANodeSet.xsd">
  <!-- GL-001 SIS Interface Nodes -->

  <!-- Safety Inputs (Read-Only) -->
  <UAVariable NodeId="ns=1;s=TE001A_PV" BrowseName="1:TE001A_PV"
              DataType="Float" AccessLevel="1">
    <DisplayName>Temperature Sensor A</DisplayName>
  </UAVariable>

  <UAVariable NodeId="ns=1;s=TE001B_PV" BrowseName="1:TE001B_PV"
              DataType="Float" AccessLevel="1">
    <DisplayName>Temperature Sensor B</DisplayName>
  </UAVariable>

  <!-- Safety Outputs (Read-Only Status) -->
  <UAVariable NodeId="ns=1;s=SIS_TRIP_STATUS" BrowseName="1:SIS_TRIP_STATUS"
              DataType="Boolean" AccessLevel="1">
    <DisplayName>SIS Trip Active</DisplayName>
  </UAVariable>

  <!-- GL-001 Trip Request (Write) -->
  <UAVariable NodeId="ns=1;s=GL001_TRIP_REQ" BrowseName="1:GL001_TRIP_REQ"
              DataType="Boolean" AccessLevel="3">
    <DisplayName>GL-001 Trip Request</DisplayName>
  </UAVariable>

</UANodeSet>
```

### 8.2 Modbus Register Map

| Register | Address | Data Type | Description | Access |
|----------|---------|-----------|-------------|--------|
| TE001A_PV | 40001 | FLOAT32 | Temperature Sensor A | Read |
| TE001B_PV | 40003 | FLOAT32 | Temperature Sensor B | Read |
| FT001A_PV | 40005 | FLOAT32 | Flow Sensor A | Read |
| FT001B_PV | 40007 | FLOAT32 | Flow Sensor B | Read |
| PT001A_PV | 40009 | FLOAT32 | Pressure Sensor A | Read |
| PT001B_PV | 40011 | FLOAT32 | Pressure Sensor B | Read |
| FS001_STATUS | 40013 | UINT16 | Flame Scanner Status (bitmap) | Read |
| XV_POSITIONS | 40014 | UINT16 | Valve Positions (bitmap) | Read |
| SIS_STATUS | 40015 | UINT16 | SIS Status Word | Read |
| GL001_TRIP | 40101 | UINT16 | GL-001 Trip Request | Write |
| GL001_HEARTBEAT | 40102 | UINT16 | Watchdog Counter | Write |

### 8.3 Communication Diagnostics

```python
class SISCommunicationMonitor:
    """Monitor SIS communication health"""

    TIMEOUT_MS = 2000

    def __init__(self):
        self.last_update = {}
        self.comm_fault = False

    def update_timestamp(self, tag: str):
        """Record data update timestamp"""
        self.last_update[tag] = time.time() * 1000

    def check_communication(self) -> Tuple[bool, list]:
        """
        Check communication health for all tags.

        Returns:
            Tuple of (healthy: bool, stale_tags: list)
        """
        current_time = time.time() * 1000
        stale_tags = []

        for tag, last_time in self.last_update.items():
            if current_time - last_time > self.TIMEOUT_MS:
                stale_tags.append(tag)

        self.comm_fault = len(stale_tags) > 0
        return not self.comm_fault, stale_tags
```

---

## 9. Cybersecurity Requirements

### 9.1 Security Architecture (IEC 62443)

| Zone | Security Level | Components |
|------|----------------|------------|
| SIS Zone | SL 3 | Safety PLC, I/O |
| Control Zone | SL 2 | BPCS, HMI |
| DMZ | SL 2 | Historian interface |
| Enterprise | SL 1 | Business systems |

### 9.2 GL-001 Security Controls

| Control | Implementation |
|---------|----------------|
| Authentication | X.509 certificates for OPC UA |
| Authorization | Role-based access control |
| Encryption | TLS 1.3 for network communication |
| Integrity | Message signing, CRC validation |
| Audit | All access logged to historian |
| Network segmentation | VLAN isolation, firewall rules |

### 9.3 Access Control Matrix

| Role | Read Safety Data | Write Trip Req | Configure | Reset |
|------|------------------|----------------|-----------|-------|
| Operator | Yes | No | No | Yes (with key) |
| Engineer | Yes | Yes | Yes | Yes |
| Maintenance | Yes | No | No | No |
| Viewer | Yes | No | No | No |

---

## 10. Commissioning Requirements

### 10.1 Pre-Commissioning Checklist

| Item | Verification | Sign-off |
|------|--------------|----------|
| SIS hardware installed per design | Visual inspection | |
| Field wiring complete and verified | Loop check | |
| SIS logic programmed and verified | Simulation test | |
| GL-001 software installed | Version verification | |
| Communication established | Ping test, data flow | |
| HMI screens configured | Display verification | |
| Alarm configuration complete | Alarm generation test | |

### 10.2 Commissioning Test Sequence

1. **Communication Test**
   - Verify OPC UA connection
   - Verify all tags updating
   - Test communication failover

2. **Input Verification**
   - Verify sensor readings match field
   - Test sensor failure detection
   - Verify voting logic display

3. **Output Verification**
   - Test trip request signal
   - Verify valve feedback
   - Test watchdog operation

4. **Integrated Test**
   - Perform SIF functional tests
   - Verify response times
   - Test safe state achievement

5. **Documentation**
   - Complete commissioning records
   - Update as-built drawings
   - Archive test results

### 10.3 Site Acceptance Test (SAT) Checklist

| Test | Expected Result | Actual | Pass/Fail |
|------|-----------------|--------|-----------|
| GL-001 startup | Connects to SIS within 30s | | |
| Data update rate | 100ms OPC UA publishing | | |
| Alarm generation | All alarms display correctly | | |
| Trip display | Trip status reflects SIS state | | |
| Response time | Trip display < 200ms from SIS trip | | |
| Communication loss | Fault alarm within 2 seconds | | |
| Watchdog test | SIS detects GL-001 failure | | |

---

## Appendix A: I/O List

| Tag | Description | Type | Range | SIF | Channel |
|-----|-------------|------|-------|-----|---------|
| TE-001A | Temperature Sensor A | AI | 0-500 C | SIF-001 | A |
| TE-001B | Temperature Sensor B | AI | 0-500 C | SIF-001 | B |
| FT-001A | Flow Sensor A | AI | 0-100% | SIF-002 | A |
| FT-001B | Flow Sensor B | AI | 0-100% | SIF-002 | B |
| PT-001A | Pressure Sensor A | AI | 0-100 bar | SIF-003 | A |
| PT-001B | Pressure Sensor B | AI | 0-100 bar | SIF-003 | B |
| FS-001A | Flame Scanner A | DI | ON/OFF | SIF-004 | A |
| FS-001B | Flame Scanner B | DI | ON/OFF | SIF-004 | B |
| FS-001C | Flame Scanner C | DI | ON/OFF | SIF-004 | C |
| XV-001A | Fuel Valve A | DO | OPEN/CLOSE | SIF-001,005 | A |
| XV-001B | Fuel Valve B | DO | OPEN/CLOSE | SIF-001,005 | B |
| XV-002 | Heat Source Valve | DO | OPEN/CLOSE | SIF-002 | - |
| XV-004A | Fuel Block Valve 1 | DO | OPEN/CLOSE | SIF-004 | A |
| XV-004B | Fuel Block Valve 2 | DO | OPEN/CLOSE | SIF-004 | B |

---

## Appendix B: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-05 | GL-TechWriter | Initial release |

---

**Document End**

*This document is part of the GreenLang IEC 61511 SIL Certification Documentation Package.*
