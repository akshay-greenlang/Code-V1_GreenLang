# NFPA 85 Chapter 5 Compliance Documentation

## GL-002 FlameGuard - Boiler Efficiency Agent

**Standard:** NFPA 85-2023 Boiler and Combustion Systems Hazards Code
**Chapter:** 5 - Single Burner Boilers
**Agent:** GL-002_FlameGuard
**Last Updated:** December 2025
**Version:** 1.0.0

---

## 1. Scope and Application (NFPA 85 5.1)

### 1.1 Covered Equipment
This agent applies to single burner boilers with:
- Heat input >= 12.5 MMBtu/hr (3.66 MW)
- Steam pressure > 15 psig
- All fuel types (natural gas, fuel oil, coal)

### 1.2 Implementation Coverage

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| Purge Sequences | `safety/burner_management.py` | ✅ Complete |
| Flame Detection | `safety/flame_detector.py` | ✅ Complete |
| Safety Interlocks | `safety/safety_interlocks.py` | ✅ Complete |
| BMS State Machine | `safety/burner_management.py` | ✅ Complete |

---

## 2. Safety Interlocks and Trips (NFPA 85 5.3)

### 2.1 Required Safety Interlocks (5.3.1)

| Interlock ID | Description | NFPA Section | Implementation |
|--------------|-------------|--------------|----------------|
| IL-001 | Low Drum Level Cutoff | 5.3.1.4 | `interlock_matrix.py` |
| IL-002 | High Steam Pressure | 5.3.1.5 | `interlock_matrix.py` |
| IL-003 | Flame Failure | 5.3.5.2 | `burner_management.py` |
| IL-004 | Loss of Combustion Air | 5.3.1.1 | `interlock_matrix.py` |
| IL-005 | Low Fuel Pressure | 5.3.1.2 | `interlock_matrix.py` |
| IL-006 | High Fuel Pressure | 5.3.1.3 | `interlock_matrix.py` |
| IL-007 | Purge Not Complete | 5.6.4 | `burner_management.py` |
| IL-008 | Flame Scanner Failure | 5.3.3 | `interlock_matrix.py` |
| IL-009 | Emergency Stop | 5.3.4 | `interlock_matrix.py` |

### 2.2 Flame Detection Requirements (5.3.3)

**Implementation:** `safety/flame_detector.py`

| Requirement | Specification | Implementation Value |
|-------------|---------------|---------------------|
| Self-checking scanner | Continuous or periodic | Self-check every 60s |
| Flame signal threshold | Site-specific | 10% minimum |
| Scanner voting scheme | Per SIL requirement | 2oo3 for SIL-2 |
| Response time | < 1 second | 500ms typical |

```python
# Flame detection configuration
FLAME_THRESHOLD_PERCENT = 10.0
SCANNER_SELF_CHECK_INTERVAL_S = 60
VOTING_SCHEME = "2oo3"  # 2-out-of-3 for SIL-2
```

### 2.3 Emergency Shutdown (5.3.4)

**Implementation:** `safety/burner_management.py::emergency_stop()`

| Requirement | Implementation |
|-------------|----------------|
| E-stop accessible | Hardware requirement |
| Immediate fuel shutoff | `close_valve("main_fuel_valve")` |
| No auto-restart | State -> LOCKOUT, requires manual reset |

### 2.4 Flame Failure Response (5.3.5)

**Implementation:** `safety/burner_management.py`

```python
# NFPA 85 5.3.5.2 Timing Requirements
FLAME_FAILURE_RESPONSE_S = 4  # Maximum 4 seconds
```

| Event | Response Time | NFPA Limit | Compliant |
|-------|---------------|------------|-----------|
| Main flame loss | < 4 seconds | 4 seconds | ✅ |
| Pilot flame loss | < 10 seconds | 10 seconds | ✅ |

---

## 3. Purge Sequence Requirements (NFPA 85 5.6.4)

### 3.1 Pre-purge Timing

**Implementation:** `safety/burner_management.py`

```python
# NFPA 85 5.6.4.4 Requirements
PRE_PURGE_TIME_S = 300  # 5 minutes (minimum 4 volume changes)
MIN_AIRFLOW_PERCENT = 25  # Minimum 25% of full load airflow
```

### 3.2 Volume Change Calculation

**Implementation:** `safety/nfpa85_validator.py::validate_purge_volume_changes()`

```
Volume Changes = (Air Flow CFM × Purge Time min) / Furnace Volume CFT

Requirements:
- Minimum 4 volume changes
- Continuous airflow during purge
- All fuel valves closed
```

### 3.3 Post-purge Requirements (5.7.2)

```python
POST_PURGE_TIME_S = 60  # Minimum 1 minute
```

---

## 4. Trial for Ignition (NFPA 85 5.6.5)

### 4.1 Timing Requirements

**Implementation:** `safety/burner_management.py`

| Trial Type | Maximum Time | Implementation | Compliant |
|------------|--------------|----------------|-----------|
| Pilot Trial | 10 seconds | `PILOT_TRIAL_TIME_S = 10` | ✅ |
| Main Flame Trial | 10 seconds | `MAIN_FLAME_TRIAL_TIME_S = 10` | ✅ |

### 4.2 Trial Sequence

```
1. Verify purge complete
2. Verify all permissives satisfied
3. Open pilot fuel valve
4. Energize igniter
5. Start pilot trial timer (≤10s)
6. Verify pilot flame proven
7. Open main fuel valve
8. Start main flame trial timer (≤10s)
9. Verify main flame proven
10. De-energize igniter
11. Close pilot fuel valve (if applicable)
12. Enter FIRING state
```

---

## 5. Permissive Logic (NFPA 85 5.4)

### 5.1 Startup Permissives

**Implementation:** `safety/burner_management.py::_init_permissives()`

| Permissive | Required For | Description |
|------------|--------------|-------------|
| drum_level_ok | PRE_PURGE, FIRING | Drum level within limits |
| steam_pressure_ok | FIRING | Below high limit |
| fuel_pressure_ok | PILOT_TRIAL, FIRING | Within operating range |
| combustion_air_ok | PRE_PURGE, FIRING | FD fan running, air flow OK |
| purge_complete | PILOT_TRIAL | Pre-purge finished |
| flame_scanner_ok | PILOT_TRIAL, FIRING | Scanner self-test passed |
| no_lockout | PRE_PURGE | No active lockout |

### 5.2 Permissive State Machine

```
┌─────────────────────────────────────────────────────────────────┐
│                    BMS State Transitions                         │
├─────────────────────────────────────────────────────────────────┤
│ OFFLINE ─► PRE_PURGE ─► PILOT_TRIAL ─► PILOT_PROVEN ─►          │
│            │              │                │                     │
│            ▼              ▼                ▼                     │
│         [Purge         [Pilot           [Main                    │
│          Timer]         Timer]           Timer]                  │
│            │              │                │                     │
│            ▼              ▼                ▼                     │
│ ─► MAIN_FLAME_TRIAL ─► MAIN_FLAME_PROVEN ─► FIRING              │
│            │                                  │                  │
│            ▼                                  ▼                  │
│         LOCKOUT ◄──────── Trip ◄───────── POST_PURGE            │
│            │                                  │                  │
│            ▼                                  ▼                  │
│       Manual Reset ─────────────────────► OFFLINE               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Safety Integrity Level (SIL) Requirements

### 6.1 IEC 61511 Mapping

| Function | SIL Rating | Test Interval | Implementation |
|----------|------------|---------------|----------------|
| Low water cutoff | SIL-2 | Monthly | Hardware + Software |
| High pressure | SIL-2 | Monthly | Hardware + Software |
| Flame detection | SIL-2 | Continuous | 2oo3 voting |
| Fuel shutoff | SIL-2 | Annual | Double block & bleed |

### 6.2 Diagnostic Coverage

```python
# IEC 61511 Diagnostic Requirements
class SafetyIntegrityLevel(Enum):
    SIL_1 = "SIL-1"  # 90% DC
    SIL_2 = "SIL-2"  # 99% DC
    SIL_3 = "SIL-3"  # 99.9% DC
```

---

## 7. Testing and Validation

### 7.1 Test Files

| Test Category | File | Coverage |
|---------------|------|----------|
| NFPA 85 Interlocks | `tests/safety/test_nfpa85_interlocks.py` | ✅ |
| BMS Sequences | `tests/safety/test_bms_sequences.py` | ✅ |
| Flame Failure Response | `tests/safety/test_nfpa85_interlocks.py::TestFlameFailureResponse` | ✅ |
| Purge Timing | `tests/safety/test_nfpa85_interlocks.py::TestPurgeSequence` | ✅ |

### 7.2 Compliance Validation

**Implementation:** `safety/nfpa85_validator.py::NFPA85Validator`

```python
validator = NFPA85Validator(boiler_id="BOILER-001")
report = validator.generate_compliance_report(
    purge_params=purge_config,
    flame_params=flame_config,
    trial_params=trial_config,
    interlocks=interlock_status,
)
```

---

## 8. Audit Trail and Provenance

### 8.1 Trip History

All safety trips are logged with:
- Timestamp (UTC)
- From state
- Trip reason
- Operator actions

**Implementation:** `safety/burner_management.py::_trip_history`

### 8.2 Compliance Reports

Generated reports include:
- Report ID with timestamp
- All validation checks
- Critical findings
- SHA-256 provenance hash

---

## 9. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | Dec 2025 | GL-BackendDeveloper | Initial compliance documentation |

---

## 10. References

1. NFPA 85-2023 "Boiler and Combustion Systems Hazards Code"
2. IEC 61511-2016 "Functional Safety - Safety Instrumented Systems"
3. ASME CSD-1 "Controls and Safety Devices for Automatically Fired Boilers"
4. API 556 "Instrumentation, Control, and Protective Systems for Gas Fired Heaters"
