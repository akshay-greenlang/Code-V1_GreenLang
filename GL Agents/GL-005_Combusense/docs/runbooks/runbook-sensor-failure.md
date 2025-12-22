# Runbook: Sensor Failure

## Document Control
| Field | Value |
|-------|-------|
| Document ID | GL005-RB-002 |
| Version | 1.0.0 |
| Last Updated | 2025-12-22 |
| Owner | GreenLang Operations Team |
| Classification | HIGH - Safety Related |
| Review Cycle | Monthly |

---

## Overview

This runbook provides procedures for detecting, responding to, and recovering from sensor failures in the GL-005 CombustionControlAgent system. Sensor failures can compromise control accuracy, safety interlock reliability, and regulatory compliance.

**Sensor Categories Covered:**
- Temperature sensors (thermocouples, RTDs)
- Pressure sensors (combustion chamber, fuel supply)
- Flow meters (fuel flow, air flow)
- Flame scanners (UV, IR, ionization)
- Combustion analyzers (O2, CO, NOx)

---

## Failure Detection Patterns

### Detection Methods

| Sensor Type | Failure Mode | Detection Method | Response Time |
|-------------|--------------|------------------|---------------|
| Temperature | Open circuit | Value = max range or NaN | <100ms |
| Temperature | Short circuit | Value = min range | <100ms |
| Temperature | Drift | Comparison with backup | 1 minute |
| Pressure | Out of range | Value outside 0-100% span | <100ms |
| Pressure | Frozen | No change over time | 5 minutes |
| Flow | Zero output | Value = 0 with process running | <500ms |
| Flow | Spike/noise | Excessive variance | 1 minute |
| Flame Scanner | No signal | Digital input = 0 | <50ms |
| Flame Scanner | Degraded | Signal quality <80% | 1 minute |
| Analyzer | Calibration drift | Deviation from expected | 10 minutes |
| Analyzer | Communication loss | No data received | <30 seconds |

### Prometheus Alerts

```yaml
# Temperature sensor failure
- alert: TemperatureSensorFailure
  expr: isnan(gl005_combustion_temperature_c) or gl005_combustion_temperature_c > 2000
  for: 10s
  labels:
    severity: critical
  annotations:
    summary: "Temperature sensor failure detected"

# Sensor data quality degraded
- alert: SensorDataQualityDegraded
  expr: gl005_sensor_quality{sensor_type=~".*"} < 80
  for: 1m
  labels:
    severity: warning
  annotations:
    summary: "Sensor {{ $labels.sensor_id }} quality degraded to {{ $value }}%"

# Redundant sensor mismatch
- alert: RedundantSensorMismatch
  expr: abs(gl005_temperature_primary - gl005_temperature_backup) > 10
  for: 30s
  labels:
    severity: high
  annotations:
    summary: "Temperature sensor mismatch detected"

# Frozen sensor value
- alert: SensorValueFrozen
  expr: changes(gl005_sensor_value[5m]) == 0 and gl005_process_running == 1
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Sensor {{ $labels.sensor_id }} value unchanged for 5 minutes"
```

### Log Patterns

```
ERROR - TemperatureSensorConnector - Sensor TEMP_01 read failure: timeout
WARNING - SafetyValidator - Redundant sensor mismatch: 10.5C difference
CRITICAL - FlameScannerConnector - Scanner SCANNER_01 health: FAILED
ERROR - CombustionAnalyzerConnector - O2 sensor communication lost
WARNING - FlowMeterConnector - Flow reading out of expected range
```

---

## Severity Classification

| Condition | Severity | Response Time |
|-----------|----------|---------------|
| Safety-critical sensor failed (no backup) | P0 - CRITICAL | Immediate |
| Safety-critical sensor failed (backup active) | P1 - HIGH | 15 minutes |
| Redundant sensors disagree | P1 - HIGH | 30 minutes |
| Control sensor failed (backup active) | P2 - MEDIUM | 1 hour |
| Monitoring-only sensor failed | P3 - LOW | Next business day |
| Sensor calibration drift detected | P2 - MEDIUM | 4 hours |

### Sensor Criticality Matrix

| Sensor | Function | Redundancy | Failure Impact |
|--------|----------|------------|----------------|
| Flame Scanner | Safety | 2-3x | CRITICAL - Shutdown |
| Furnace Temperature | Safety/Control | 2x | HIGH - Degraded operation |
| Fuel Flow | Safety/Control | 1x (calculated backup) | HIGH - Manual mode |
| Combustion Pressure | Safety | 2x | HIGH - Degraded operation |
| O2 Analyzer | Control/Compliance | 1x | MEDIUM - Open loop |
| Stack Temperature | Monitoring | 1x | LOW - No immediate impact |

---

## Fallback Procedures

### General Fallback Hierarchy

```
Level 1: Switch to redundant sensor (automatic)
Level 2: Use calculated/inferred value (automatic/manual)
Level 3: Reduce control authority (manual approval)
Level 4: Switch to manual mode (operator takeover)
Level 5: Safe shutdown (if no viable control)
```

### Temperature Sensor Fallback

**Automatic Fallback (Primary to Backup):**
```bash
# Check current temperature sensor configuration
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/sensors/temperature/status | jq '.'

# Response example:
# {
#   "primary_sensor": "TEMP_01",
#   "primary_status": "FAILED",
#   "backup_sensor": "TEMP_02",
#   "backup_status": "HEALTHY",
#   "active_sensor": "TEMP_02",
#   "fallback_mode": true
# }
```

**Manual Fallback (No Working Backup):**
```bash
# Enable calculated temperature mode (from heat balance)
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/sensors/temperature/fallback \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -d '{"mode": "calculated", "reason": "Both sensors failed"}'
```

### Pressure Sensor Fallback

**Automatic Fallback:**
```bash
# Check pressure sensor redundancy
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/sensors/pressure/status | jq '.'
```

**Manual Override:**
```bash
# Set pressure to fixed value (requires dual authorization)
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/sensors/pressure/override \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -d '{
    "sensor_id": "PRESS_01",
    "override_value": -500,
    "reason": "Sensor failure - using field gauge reading",
    "authorizer": "SUPERVISOR_ID",
    "duration_minutes": 60
  }'
```

### Flow Meter Fallback

**Calculated Flow (from valve position):**
```bash
# Enable calculated flow mode
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/sensors/flow/fallback \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -d '{
    "sensor_id": "FLOW_FUEL_01",
    "mode": "calculated",
    "calculation_method": "valve_cv",
    "valve_id": "FCV_001"
  }'
```

### Flame Scanner Fallback

**Multi-Scanner Voting Logic:**
```bash
# Check flame scanner voting status
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/flame/voting | jq '.'

# Expected response:
# {
#   "voting_logic": "2_of_3",
#   "scanner_1": {"status": "PRESENT", "health": "HEALTHY"},
#   "scanner_2": {"status": "PRESENT", "health": "FAILED"},
#   "scanner_3": {"status": "PRESENT", "health": "HEALTHY"},
#   "flame_detected": true,
#   "degraded_mode": true
# }
```

**Single Scanner Mode (Emergency Only):**
```bash
# WARNING: Reduced safety margin - requires Safety Officer approval
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/flame/single-scanner-mode \
  -H "Authorization: Bearer $SAFETY_OFFICER_TOKEN" \
  -d '{
    "active_scanner": "SCANNER_01",
    "reason": "Two scanners failed",
    "safety_officer_id": "SO_001",
    "max_duration_hours": 4
  }'
```

### Combustion Analyzer Fallback

**O2 Analyzer Failure:**
```bash
# Switch to open-loop control (fixed excess air)
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/control/o2-trim/disable \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -d '{
    "reason": "O2 analyzer failed",
    "fixed_excess_air_percent": 15,
    "notify_compliance": true
  }'
```

---

## Manual Override Steps

### Override Authorization Matrix

| Sensor Type | Duration | Approver 1 | Approver 2 |
|-------------|----------|------------|------------|
| Flame Scanner | 1 hour max | Safety Officer | - |
| Temperature (Safety) | 4 hours | Shift Supervisor | Safety Officer |
| Pressure (Safety) | 4 hours | Shift Supervisor | Safety Officer |
| Flow (Control) | 8 hours | Shift Supervisor | - |
| Analyzer | 24 hours | Shift Supervisor | - |

### Override Procedure

**Step 1: Document the Override Request**
```bash
# Create override request
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/overrides/request \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -d '{
    "sensor_id": "TEMP_01",
    "override_type": "fixed_value",
    "override_value": 850,
    "reason": "Thermocouple failed - using portable meter reading",
    "duration_hours": 4,
    "field_reading_source": "Fluke 52 II S/N 12345"
  }'
```

**Step 2: Obtain Authorization**
```bash
# Authorize override (requires second person)
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/overrides/authorize \
  -H "Authorization: Bearer $SUPERVISOR_TOKEN" \
  -d '{
    "override_id": "OVR_20251222_001",
    "authorizer_id": "SUPERVISOR_001",
    "authorization_code": "AUTH_CODE_FROM_SUPERVISOR"
  }'
```

**Step 3: Activate Override**
```bash
# Activate the authorized override
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/overrides/activate \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -d '{
    "override_id": "OVR_20251222_001"
  }'
```

**Step 4: Monitor Override Status**
```bash
# Check active overrides
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/overrides/active | jq '.'

# Set up override expiration alert
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/overrides/set-reminder \
  -d '{
    "override_id": "OVR_20251222_001",
    "remind_before_minutes": 30
  }'
```

**Step 5: Remove Override (When Sensor Repaired)**
```bash
# Deactivate override
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/overrides/deactivate \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -d '{
    "override_id": "OVR_20251222_001",
    "reason": "Sensor replaced and calibrated"
  }'
```

---

## Sensor Replacement Checklist

### Pre-Replacement

- [ ] Notify control room of planned sensor work
- [ ] Place control loop in MANUAL or CASCADE (as appropriate)
- [ ] Enable fallback/override for affected sensor
- [ ] Obtain replacement sensor (verify model, range, calibration)
- [ ] Lock-out/Tag-out (LOTO) as required
- [ ] Gather tools and calibration equipment

### Replacement Procedure

**For Temperature Sensors (Thermocouple):**
- [ ] Verify thermowell integrity
- [ ] Check thermocouple type matches specification (K, J, N, etc.)
- [ ] Verify lead wire polarity
- [ ] Insert sensor to proper depth
- [ ] Secure connections
- [ ] Verify signal at transmitter/DCS

**For Pressure Sensors:**
- [ ] Isolate process connection (close block valves)
- [ ] Bleed pressure from impulse line
- [ ] Verify mounting orientation
- [ ] Check zero and span with calibrator
- [ ] Open block valves slowly
- [ ] Verify stable reading

**For Flow Meters:**
- [ ] Verify straight run requirements met
- [ ] Check meter orientation (arrow direction)
- [ ] Verify electrical connections
- [ ] Configure transmitter parameters
- [ ] Verify zero flow reading

**For Flame Scanners:**
- [ ] Verify sight tube alignment
- [ ] Check cooling air supply (if required)
- [ ] Clean scanner lens
- [ ] Verify scanner type matches burner
- [ ] Test with burner firing (live flame test)

**For Combustion Analyzers:**
- [ ] Verify sample line integrity
- [ ] Check sample conditioning system
- [ ] Perform zero/span calibration with certified gases
- [ ] Verify response time
- [ ] Compare reading with portable analyzer

### Post-Replacement

- [ ] Remove override/fallback mode
- [ ] Return control loop to AUTO
- [ ] Verify sensor reading matches expected value
- [ ] Compare with redundant sensor (if available)
- [ ] Update maintenance records
- [ ] Document calibration data
- [ ] Clear LOTO
- [ ] Notify control room of completion

### Calibration Verification

```bash
# Verify sensor calibration in system
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/sensors/calibration \
  -d '{"sensor_id": "TEMP_01"}' | jq '.'

# Expected response:
# {
#   "sensor_id": "TEMP_01",
#   "last_calibration": "2025-12-22T10:30:00Z",
#   "next_calibration_due": "2026-03-22T10:30:00Z",
#   "calibration_offset": 0.5,
#   "calibration_technician": "TECH_001",
#   "calibration_certificate": "CAL_12345"
# }
```

---

## Resolution Verification

### Verification Steps

**Step 1: Sensor Health Check**
```bash
# Full sensor diagnostic
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/sensors/diagnostic \
  -d '{"sensor_id": "TEMP_01"}' | jq '.'
```

**Step 2: Redundancy Check**
```bash
# Compare with backup sensor
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/sensors/redundancy-check | jq '.'
```

**Step 3: Control Loop Response Test**
```bash
# Perform step test (small setpoint change)
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/control/step-test \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -d '{
    "loop_id": "TEMP_CONTROL_01",
    "step_magnitude_percent": 2,
    "verify_response": true
  }'
```

**Step 4: Safety Interlock Test**
```bash
# Verify safety interlocks using replaced sensor
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/safety/interlock-test \
  -H "Authorization: Bearer $SAFETY_OFFICER_TOKEN" \
  -d '{
    "interlock_id": "TEMP_HIGH_INTERLOCK",
    "test_mode": "simulate",
    "expected_action": "fuel_shutoff"
  }'
```

---

## Prevention Measures

### Proactive Monitoring

| Monitor | Threshold | Action |
|---------|-----------|--------|
| Sensor signal quality | <90% | Schedule inspection |
| Sensor drift from backup | >5% | Calibration check |
| Sensor age | >80% of life | Schedule replacement |
| Communication errors | >1/hour | Check wiring/network |

### Maintenance Schedule

| Sensor Type | Calibration Interval | Replacement Interval |
|-------------|---------------------|---------------------|
| Thermocouple | 6 months | 3-5 years |
| RTD | 12 months | 5-10 years |
| Pressure transmitter | 12 months | 10 years |
| Flow meter | 12 months | 15 years |
| Flame scanner | Weekly verification | 5 years |
| O2 analyzer | Weekly calibration | Cell replacement yearly |

### Spare Parts Inventory

Maintain minimum stock levels:
- [ ] 2x Temperature sensors per type
- [ ] 1x Pressure transmitter per type
- [ ] 1x Flow meter (critical positions)
- [ ] 2x Flame scanners per burner type
- [ ] O2 sensor cells (2 per analyzer)
- [ ] Calibration gases (certified, non-expired)

---

## Appendix

### A. Sensor Failure Decision Tree

```
Sensor Failure Detected
    |
    v
Is backup sensor available?
    |--- YES --> Automatic failover to backup
    |              |
    |              v
    |           Is backup healthy?
    |              |--- YES --> Continue operation (degraded)
    |              |              Schedule replacement
    |              |
    |              |--- NO --> Go to "NO backup"
    |
    |--- NO --> Is calculated value available?
                   |--- YES --> Enable calculated mode
                   |              Increase monitoring
                   |              Schedule emergency repair
                   |
                   |--- NO --> Is manual mode viable?
                                  |--- YES --> Switch to manual
                                  |              Operator takeover
                                  |
                                  |--- NO --> Initiate safe shutdown
```

### B. Related Documentation

- [runbook-flame-loss.md](./runbook-flame-loss.md) - Flame scanner failure escalation
- [runbook-emergency-shutdown.md](./runbook-emergency-shutdown.md) - Shutdown due to sensor failure
- [MAINTENANCE.md](../../runbooks/MAINTENANCE.md) - Calibration schedules

### C. Emergency Contacts

| Role | Contact |
|------|---------|
| Instrument Technician (On-Call) | @instrument-oncall |
| DCS Support | (XXX) XXX-XXXX |
| Sensor Vendor Support | (XXX) XXX-XXXX |
| Calibration Lab | (XXX) XXX-XXXX |

### D. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-22 | GL-TechWriter | Initial version |
