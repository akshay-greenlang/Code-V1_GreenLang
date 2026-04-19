# Runbook: Emergency Shutdown (ESD)

## Document Control
| Field | Value |
|-------|-------|
| Document ID | GL005-RB-004 |
| Version | 1.0.0 |
| Last Updated | 2025-12-22 |
| Owner | GreenLang Operations Team |
| Classification | CRITICAL - Safety Critical |
| Review Cycle | Monthly |

---

## Overview

This runbook defines Emergency Shutdown (ESD) procedures for the GL-005 CombustionControlAgent system. Emergency shutdowns are safety-critical events that immediately stop combustion operations to prevent injury, equipment damage, or environmental incidents.

**Reference Standards:**
- NFPA 85: Boiler and Combustion Systems Hazards Code
- IEC 61508: Functional Safety of Electrical/Electronic/Programmable Systems
- ISA-84: Functional Safety - Safety Instrumented Systems
- OSHA 29 CFR 1910.119: Process Safety Management

**Safety Integrity Level (SIL):** SIL 2 (Target PFD: 10^-2 to 10^-3)

---

## ESD Trigger Conditions

### Automatic ESD Triggers

| Trigger | Setpoint | Response Time | Priority |
|---------|----------|---------------|----------|
| Flame Failure (all burners) | Flame absent >200ms | <500ms | 1 |
| Combustion Temperature High-High | >1500C | <1s | 1 |
| Combustion Pressure High-High | >10,000 Pa | <1s | 1 |
| Fuel Pressure Low-Low | <50,000 Pa | <1s | 2 |
| CO Level Extremely High | >800 ppm | <2s | 2 |
| Fire Detected (fire suppression) | Fire alarm signal | <500ms | 1 |
| Gas Leak Detected | LEL >25% | <500ms | 1 |
| Operator Emergency Stop | E-Stop button | Immediate | 1 |

### Manual ESD Triggers

| Trigger | Method | Authorization |
|---------|--------|---------------|
| Operator E-Stop | Physical button at HMI | None (immediate) |
| Remote E-Stop | SCADA command | Operator credential |
| API E-Stop | REST endpoint | Safety Officer token |

### Detection Alerts

```yaml
# Critical alerts that may precede or accompany ESD
- alert: EmergencyShutdownTriggered
  expr: gl005_esd_active == 1
  labels:
    severity: critical
  annotations:
    summary: "EMERGENCY SHUTDOWN ACTIVE"
    runbook_url: "https://docs.greenlang.io/runbooks/emergency-shutdown"

- alert: SafetyInterlockTripped
  expr: gl005_safety_interlock_tripped == 1
  for: 0s
  labels:
    severity: critical
  annotations:
    summary: "Safety interlock {{ $labels.interlock_id }} tripped"
```

### Log Patterns

```
CRITICAL - SafetyValidator - EMERGENCY SHUTDOWN INITIATED
CRITICAL - SafetyValidator - Trigger: FLAME_FAILURE_ALL_BURNERS
CRITICAL - CombustionControlOrchestrator - Executing ESD sequence
INFO - CombustionControlOrchestrator - Step 1: Closing main fuel valve
INFO - CombustionControlOrchestrator - Step 2: Closing pilot fuel valve
INFO - CombustionControlOrchestrator - Step 3: Starting purge cycle
INFO - CombustionControlOrchestrator - ESD sequence complete - SAFE STATE
```

---

## Shutdown Sequence

### Automatic Shutdown Sequence

The ESD sequence executes automatically and cannot be interrupted once initiated.

```
ESD SEQUENCE (Total Time: <30 seconds)
=========================================
T+0.0s   ESD triggered
T+0.1s   Main fuel shutoff valve CLOSE command sent
T+0.2s   Pilot fuel shutoff valve CLOSE command sent
T+0.3s   Verify fuel valves closed (feedback check)
T+0.5s   ESD annunciation to SCADA/HMI
T+1.0s   Combustion air flow increased for purge
T+1.5s   De-energize ignition systems
T+2.0s   Post-purge timer started (4 air changes)
T+5-15s  Purge in progress (time varies by chamber volume)
T+15-30s Purge complete - verify air changes
T+30s    System in SAFE STATE - LOCKOUT active
```

### Shutdown Verification

```bash
# Verify ESD status
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/safety/esd/status | jq '.'

# Expected response during/after ESD:
# {
#   "esd_active": true,
#   "esd_trigger": "FLAME_FAILURE",
#   "esd_timestamp": "2025-12-22T10:30:00.000Z",
#   "sequence_step": "PURGE_IN_PROGRESS",
#   "main_fuel_valve": "CLOSED",
#   "pilot_fuel_valve": "CLOSED",
#   "purge_air_flow": "MAXIMUM",
#   "air_changes_completed": 2,
#   "air_changes_required": 4,
#   "safe_state_achieved": false,
#   "lockout_active": false
# }
```

---

## Safe State Verification

### Safe State Criteria

The system is in SAFE STATE when ALL of the following conditions are verified:

| Item | Verification Method | Expected State |
|------|-------------------|----------------|
| Main fuel valve | Position feedback | CLOSED |
| Pilot fuel valve | Position feedback | CLOSED |
| Fuel flow | Flow meter reading | 0 (zero) |
| Combustion chamber | Pressure/temp dropping | Decreasing |
| Purge | Air changes counter | >= 4 |
| Fire suppression | If triggered, active | Deployed |
| Personnel | Evacuation status | Clear of hazard zone |

### Verification Procedure

**Step 1: Fuel Isolation Verification**
```bash
# Verify all fuel valves closed
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/safety/fuel-isolation/status | jq '.'

# Check valve position feedback
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/valves/status | jq '.[] | select(.type == "fuel")'
```

**Step 2: Purge Verification**
```bash
# Verify purge completion
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/safety/purge/status | jq '.'

# Expected output:
# {
#   "purge_active": false,
#   "purge_complete": true,
#   "air_changes_completed": 5,
#   "purge_duration_seconds": 180,
#   "combustion_chamber_clear": true
# }
```

**Step 3: Atmospheric Verification**
```bash
# Verify no combustible gas accumulation
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/safety/atmosphere/status | jq '.'

# Check combustible gas detector readings
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8001/metrics | grep "combustible_gas"
```

**Step 4: Temperature Verification**
```bash
# Monitor combustion chamber cool-down
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/combustion/temperature | jq '.'

# Temperature should be decreasing toward ambient
```

### Safe State Checklist

**Immediate Verification (within 5 minutes):**
- [ ] Main fuel shutoff valve CLOSED (verified by position switch)
- [ ] Pilot fuel shutoff valve CLOSED (verified by position switch)
- [ ] Fuel flow meter reads ZERO
- [ ] Fuel supply block valves CLOSED (manual verification if required)
- [ ] Purge air flow ACTIVE
- [ ] ESD annunciation active on HMI
- [ ] Control room notified

**Complete Verification (within 30 minutes):**
- [ ] Purge cycle COMPLETE (minimum 4 air changes)
- [ ] Combustible gas level <5% LEL
- [ ] Combustion chamber temperature DECREASING
- [ ] Stack damper position OPEN (natural draft)
- [ ] Area cleared of personnel (if hazardous condition)
- [ ] Fire suppression status verified (if triggered)
- [ ] Incident logged in system

---

## Restart Authorization

### Authorization Requirements

| ESD Trigger | Cool-Down Time | Authorization Level | Additional Requirements |
|-------------|----------------|--------------------|-----------------------|
| Flame failure | 15 minutes | Shift Supervisor | Root cause identified |
| Temperature trip | 30 minutes | Plant Manager | Equipment inspection |
| Pressure trip | 30 minutes | Plant Manager | Equipment inspection |
| Fire detected | 4 hours | Safety Officer + Plant Manager | Fire investigation complete |
| Gas leak | 2 hours | Safety Officer | Leak repaired and tested |
| Manual E-Stop | 15 minutes | Shift Supervisor | Reason documented |

### Authorization Matrix

```
RESTART AUTHORIZATION HIERARCHY
===============================

Level 1 (Shift Supervisor):
- Flame failure (single burner, redundancy available)
- Manual E-Stop (operator-initiated, no actual hazard)
- Control system malfunction (software reset)

Level 2 (Plant Manager + Shift Supervisor):
- Flame failure (all burners)
- Temperature trip
- Pressure trip
- Fuel supply trip

Level 3 (Safety Officer + Plant Manager):
- Fire detected
- Gas leak detected
- Multiple simultaneous trips
- Any trip with equipment damage

Level 4 (Executive + Safety Officer + Plant Manager):
- Injury incident
- Environmental release
- Equipment catastrophic failure
- Regulatory authority involvement
```

### Authorization Procedure

**Step 1: Request Restart Authorization**
```bash
# Create restart authorization request
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/safety/restart/request \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -d '{
    "esd_event_id": "ESD_20251222_001",
    "root_cause": "Flame scanner false trip - scanner replaced",
    "corrective_actions": [
      "Replaced UV scanner SCANNER_01",
      "Verified flame detection with backup scanner",
      "Calibrated new scanner"
    ],
    "equipment_inspection": "Burner and fuel train inspected - no damage",
    "requestor_id": "OPERATOR_001"
  }'
```

**Step 2: Supervisor Review**
```bash
# Supervisor reviews and approves
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/safety/restart/approve \
  -H "Authorization: Bearer $SUPERVISOR_TOKEN" \
  -d '{
    "request_id": "RST_20251222_001",
    "approver_id": "SUPERVISOR_001",
    "approval_notes": "Root cause addressed, equipment verified safe"
  }'
```

**Step 3: Second Authorization (if required)**
```bash
# For Level 2+ restarts, additional approval required
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/safety/restart/second-approval \
  -H "Authorization: Bearer $PLANT_MANAGER_TOKEN" \
  -d '{
    "request_id": "RST_20251222_001",
    "approver_id": "PM_001",
    "approval_notes": "Equipment inspection reviewed, safe to restart"
  }'
```

**Step 4: Reset Safety System**
```bash
# Reset safety interlocks (requires all approvals)
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/safety/interlocks/reset \
  -H "Authorization: Bearer $SUPERVISOR_TOKEN" \
  -d '{
    "request_id": "RST_20251222_001",
    "physical_reset_confirmed": true
  }'

# Note: Physical reset may also be required at DCS/PLC
```

**Step 5: Initiate Startup Sequence**
```bash
# Begin controlled startup (see Flame Loss runbook for full sequence)
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/combustion/startup/begin \
  -H "Authorization: Bearer $SUPERVISOR_TOKEN" \
  -d '{
    "restart_authorization_id": "RST_20251222_001",
    "pre_purge_required": true
  }'
```

### Restart Checklist

**Pre-Restart Verification:**
- [ ] ESD root cause identified and documented
- [ ] Corrective actions completed
- [ ] Equipment inspection complete (if required)
- [ ] All approvals obtained
- [ ] Safety interlocks reset
- [ ] Control system ready
- [ ] Communication to DCS/PLC verified
- [ ] Operator stations manned

**Startup Sequence:**
- [ ] Pre-purge initiated (NFPA 85 requirement)
- [ ] Pre-purge complete (4 air changes minimum)
- [ ] Pilot ignition sequence
- [ ] Pilot flame verified
- [ ] Main burner ignition
- [ ] Main flame verified
- [ ] Stable operation achieved
- [ ] Control loops in AUTO

---

## Post-Incident Procedures

### Immediate Documentation (within 1 hour)

| Item | Responsible | Destination |
|------|-------------|-------------|
| ESD event timestamp | System (automatic) | Event log |
| Trigger condition | System (automatic) | Event log |
| Sequence completion | System (automatic) | Event log |
| Operator actions | Operator | Incident report |
| Equipment status | Operator | Incident report |
| Personnel status | Supervisor | Incident report |

### Incident Report Template

```markdown
## ESD Incident Report

**Event ID:** ESD_20251222_001
**Date/Time:** 2025-12-22 10:30:00 UTC
**Duration:** XX minutes
**Severity:** P0 / P1 / P2

### Trigger Information
- **Primary Trigger:** [FLAME_FAILURE / TEMP_HIGH / etc.]
- **Trigger Value:** [actual value that triggered ESD]
- **Setpoint:** [trip setpoint]

### Sequence Execution
- **ESD Initiated:** [timestamp]
- **Fuel Valves Closed:** [timestamp]
- **Purge Started:** [timestamp]
- **Purge Complete:** [timestamp]
- **Safe State Achieved:** [timestamp]

### Root Cause
[Description of root cause]

### Corrective Actions
1. [Action taken]
2. [Action taken]
3. [Action taken]

### Equipment Impact
- [Any equipment damage or wear noted]

### Personnel Impact
- [Any injuries or near-misses]

### Lessons Learned
- [What can be improved]

### Approval for Restart
- **Requestor:** [Name, ID]
- **Supervisor:** [Name, ID, timestamp]
- **Additional Approver:** [Name, ID, timestamp] (if required)
```

### Post-Incident Review (PIR)

**Timeline:**
- P0 ESD: PIR within 24 hours
- P1 ESD: PIR within 48 hours
- P2 ESD: PIR within 1 week

**PIR Attendees:**
- Shift Supervisor (on duty during event)
- Operations Manager
- Safety Officer
- Engineering representative
- Maintenance representative (if equipment involved)

**PIR Agenda:**
1. Timeline review (5 min)
2. Root cause analysis (15 min)
3. ESD system performance (10 min)
4. Response effectiveness (10 min)
5. Corrective actions (10 min)
6. Prevention measures (10 min)

---

## Prevention Measures

### Regular Testing

| Test | Frequency | Method | Owner |
|------|-----------|--------|-------|
| ESD logic test | Monthly | Simulated input | I&E Technician |
| Fuel valve stroke test | Quarterly | Partial stroke | Maintenance |
| Full ESD test | Annually | Process shutdown | Safety Officer |
| E-Stop button test | Weekly | Physical test | Operator |

### System Maintenance

| Component | Inspection Frequency | Replacement Interval |
|-----------|---------------------|---------------------|
| Fuel shutoff valves | Monthly | Per manufacturer |
| Position switches | Monthly | 5 years |
| Flame scanners | Weekly | 5 years |
| Pressure transmitters | Monthly | 10 years |
| E-Stop buttons | Weekly | 10 years |
| Logic solver (SIS) | Annual certification | Per manufacturer |

### Training Requirements

| Role | Training | Frequency |
|------|----------|-----------|
| Operator | ESD recognition and response | Annual |
| Supervisor | ESD authorization procedures | Annual |
| Maintenance | ESD testing procedures | Annual |
| All personnel | E-Stop location and use | Annual |

---

## Appendix

### A. ESD System Architecture

```
                    ┌─────────────────┐
                    │  Safety Sensors │
                    │  (Redundant)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Safety Logic   │
                    │  Solver (SIS)   │
                    │  SIL 2 Rated    │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
┌────────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐
│  Main Fuel SOV  │ │ Pilot Fuel SOV  │ │  Purge System   │
│  (Fail-Closed)  │ │  (Fail-Closed)  │ │  (Fail-Safe)    │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### B. Fuel Shutoff Valve Specifications

| Parameter | Main Fuel SOV | Pilot Fuel SOV |
|-----------|--------------|----------------|
| Closure time | <1 second | <1 second |
| Fail mode | Fail-closed | Fail-closed |
| Tight shutoff | Class VI | Class VI |
| Position feedback | Limit switches (2x) | Limit switches (2x) |
| Stroke test | Monthly partial | Monthly partial |

### C. Related Documentation

- [runbook-flame-loss.md](./runbook-flame-loss.md) - Flame failure procedures
- [runbook-sensor-failure.md](./runbook-sensor-failure.md) - Sensor-triggered ESD
- [ROLLBACK_PROCEDURE.md](../../runbooks/ROLLBACK_PROCEDURE.md) - Software rollback after ESD
- Safety Validator: `calculators/safety_validator.py`

### D. Emergency Contacts

| Role | Contact | Availability |
|------|---------|--------------|
| Safety Officer | (XXX) XXX-XXXX | 24/7 |
| Plant Manager | (XXX) XXX-XXXX | Business hours |
| Control Room | (XXX) XXX-XXXX | 24/7 |
| Fire Brigade | 911 / (XXX) XXX-XXXX | 24/7 |
| Environmental Response | (XXX) XXX-XXXX | 24/7 |

### E. Regulatory Notification Requirements

| Event | Notification Required | Timeline | Authority |
|-------|----------------------|----------|-----------|
| Fire | Yes | Immediate | Fire Marshal |
| Injury | Yes | Within 8 hours | OSHA |
| Environmental release | Yes | Within 24 hours | EPA/State DEQ |
| Equipment failure | Depends | Per permit | Varies |

### F. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-22 | GL-TechWriter | Initial version |
