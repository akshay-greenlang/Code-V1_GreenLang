# Runbook: Flame Loss Event

## Document Control
| Field | Value |
|-------|-------|
| Document ID | GL005-RB-001 |
| Version | 1.0.0 |
| Last Updated | 2025-12-22 |
| Owner | GreenLang Operations Team |
| Classification | CRITICAL - Safety Critical |
| Review Cycle | Monthly |

---

## Overview

This runbook provides step-by-step procedures for responding to flame loss events in the GL-005 CombustionControlAgent system. Flame loss is a **safety-critical condition** that requires immediate action to prevent equipment damage, unsafe gas accumulation, and potential explosion hazards.

**Reference Standards:**
- NFPA 85: Boiler and Combustion Systems Hazards Code
- NFPA 86: Standard for Ovens and Furnaces
- API 556: Fired Heaters for General Refinery Service

---

## Detection Criteria

### Primary Detection Methods

| Detection Method | Response Time | Threshold | Source |
|-----------------|---------------|-----------|--------|
| UV Flame Scanner | <50ms | Signal loss >200ms | `flame_scanner_connector.py` |
| IR Flame Scanner | <50ms | Signal loss >200ms | `flame_scanner_connector.py` |
| Flame Rod (Ionization) | <30ms | Current <5uA | PLC Direct I/O |
| Flame Intensity Drop | <100ms | Intensity <10% of normal | SCADA Analog |

### Alert Triggers

```
Alert: gl005_flame_status{burner_id="*",status="absent"} == 0
Severity: CRITICAL
Source: Prometheus/Alertmanager

Metric: flame_scanner_status (Gauge)
Labels: scanner_id, burner_id
Values: 1=present, 0=absent
```

### Symptoms

**Immediate Indicators:**
- Flame scanner alarm activated (audible/visual)
- DCS/HMI displays "FLAME FAILURE" status
- Fuel safety shutoff valves close automatically
- Control system enters LOCKOUT state

**Secondary Indicators:**
- Combustion chamber temperature dropping rapidly
- Stack temperature decreasing
- O2 levels rising in flue gas
- Combustion pressure destabilizing

**Log Patterns:**
```
CRITICAL - FlameScannerConnector - Flame loss detected on BURNER_01
CRITICAL - SafetyValidator - FLAME FAILURE on BURNER_01
INFO - CombustionControlOrchestrator - Initiating emergency shutdown sequence
```

---

## Severity Classification

| Condition | Severity | Response Time |
|-----------|----------|---------------|
| Single burner flame loss (multi-burner system) | P1 - HIGH | 15 minutes |
| All burners flame loss | P0 - CRITICAL | Immediate |
| Flame loss with fuel still flowing | P0 - CRITICAL | Immediate |
| Flame loss during startup sequence | P1 - HIGH | 15 minutes |
| Intermittent flame detection (flicker) | P2 - MEDIUM | 30 minutes |

---

## Immediate Actions

### Phase 1: Emergency Response (0-5 minutes)

**Step 1: Verify Flame Loss**
```bash
# Check flame scanner status via API
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/combustion/flame-status | jq '.'

# Check Prometheus metrics
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8001/metrics | grep "flame_scanner_status"
```

**Step 2: Confirm Fuel Shutoff**
```bash
# Verify fuel valve closed
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/status | jq '.safety.fuel_valve_status'

# Check DCS fuel flow reading
kubectl logs -n greenlang deployment/gl-005-combustion-control --tail=20 | \
  grep -i "fuel_flow"
```

**Step 3: Initiate Purge Cycle (if not automatic)**
```bash
# Manual purge initiation (if required)
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/combustion/purge/start \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -d '{"purge_duration_minutes": 5, "air_changes": 4}'
```

**Step 4: Notify Operations**
- Page on-call engineer via Slack @oncall-gl005
- Update incident channel #gl005-incidents
- Contact control room operator by phone if no response in 5 minutes

### Phase 2: Safe State Verification (5-15 minutes)

**Checklist:**
- [ ] All fuel safety shutoff valves confirmed CLOSED
- [ ] Purge cycle in progress or completed (minimum 4 air changes)
- [ ] Combustion chamber pressure stable (draft maintained)
- [ ] No fuel accumulation detected (combustible gas monitor)
- [ ] Emergency ventilation active (if applicable)
- [ ] Personnel evacuated from hazardous areas

---

## Root Cause Investigation

### Investigation Checklist

**1. Flame Scanner Health Check**
```bash
# Check scanner diagnostics
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  python -c "
from integrations.flame_scanner_connector import FlameScannerConnector, FlameScannerConfig, ScannerType
import asyncio

async def check():
    config = FlameScannerConfig(
        scanner_id='SCANNER_BURNER_01',
        scanner_type=ScannerType.UV_DETECTOR,
        burner_id='BURNER_01'
    )
    connector = FlameScannerConnector(config)
    print('Health:', connector.scanner_health)
    print('Consecutive Failures:', connector.consecutive_failures)
    print('Signal Quality:', connector._calculate_signal_quality())

asyncio.run(check())
"

# Check scanner calibration status
kubectl logs -n greenlang deployment/gl-005-combustion-control --since=24h | \
  grep -i "calibration\|scanner_health"
```

**2. Fuel Supply Analysis**
```bash
# Check fuel pressure trend
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/combustion/fuel-supply | jq '.'

# Review fuel flow before flame loss
kubectl logs -n greenlang deployment/gl-005-combustion-control --since=30m | \
  grep "fuel_flow_rate" | tail -20
```

**3. Combustion Air Check**
```bash
# Check air flow readings
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/combustion/air-supply | jq '.'

# Verify damper positions
kubectl logs -n greenlang deployment/gl-005-combustion-control --since=30m | \
  grep "damper_position\|air_flow"
```

**4. Control System Analysis**
```bash
# Review control actions before flame loss
kubectl logs -n greenlang deployment/gl-005-combustion-control --since=30m | \
  grep "ControlAction\|PIDController" | tail -50

# Check for communication failures
kubectl logs -n greenlang deployment/gl-005-combustion-control --since=30m | \
  grep -i "connection\|timeout\|error"
```

### Common Root Causes

| Root Cause | Probability | Detection Method |
|------------|-------------|------------------|
| Fuel supply interruption | HIGH | Fuel pressure drop, flow = 0 |
| Flame scanner failure | MEDIUM | Scanner diagnostics, backup disagreement |
| Combustion air loss | MEDIUM | Air flow drop, damper position |
| Ignition system failure | MEDIUM | Igniter current, spark detection |
| Fuel quality issue | LOW | Fuel analysis, calorific value change |
| Control system malfunction | LOW | Control logs, setpoint tracking |
| Mechanical failure (burner) | LOW | Visual inspection, maintenance history |

---

## Recovery Procedures

### Pre-Restart Checklist

**Safety Verification:**
- [ ] Root cause identified and corrected
- [ ] All safety interlocks reset
- [ ] Purge cycle completed (verified 4 air changes minimum)
- [ ] Flame scanners confirmed operational
- [ ] Fuel supply pressure verified within limits
- [ ] Combustion air supply verified
- [ ] Operator authorization obtained

**System Readiness:**
- [ ] DCS/PLC communication verified
- [ ] All sensors responding
- [ ] Control loops in AUTO
- [ ] SCADA/HMI updated

### Restart Sequence

**Step 1: Pre-Purge**
```bash
# Initiate pre-purge (NFPA 85 requirement)
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/combustion/startup/pre-purge \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -d '{"duration_minutes": 5, "verify_air_changes": true}'
```

**Step 2: Verify Purge Completion**
```bash
# Wait for purge verification
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/combustion/startup/status | jq '.purge_status'
```

**Step 3: Ignition Sequence**
```bash
# Initiate pilot ignition (if applicable)
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/combustion/startup/ignite-pilot \
  -H "Authorization: Bearer $OPERATOR_TOKEN"

# Verify pilot flame
sleep 10
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/combustion/flame-status | jq '.pilot_flame'
```

**Step 4: Main Burner Ignition**
```bash
# Start main burner (after pilot verified)
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/combustion/startup/ignite-main \
  -H "Authorization: Bearer $OPERATOR_TOKEN"

# Monitor flame establishment
watch -n 2 "kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/combustion/flame-status"
```

**Step 5: Stabilization**
```bash
# Wait for stable operation (5 minutes minimum)
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/combustion/stability | jq '.'

# Verify emissions within limits
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/combustion/emissions | jq '.'
```

### Restart Authorization Matrix

| Condition | Authorization Required |
|-----------|----------------------|
| First restart attempt | Shift Supervisor |
| Second restart attempt | Plant Manager + Shift Supervisor |
| Third+ restart attempt | Safety Officer + Plant Manager |
| Restart after equipment repair | Maintenance Lead + Shift Supervisor |
| Restart after safety system modification | Safety Officer + Engineering Manager |

---

## Post-Incident Review

### Documentation Requirements

**Immediate (within 2 hours):**
- Incident report created in tracking system
- Timeline of events documented
- Immediate corrective actions recorded

**Within 24 hours:**
- Root cause analysis completed
- Equipment inspection reports attached
- Witness statements (if applicable)

**Within 48 hours:**
- Post-Incident Review (PIR) meeting scheduled
- Lessons learned documented
- Corrective action plan developed

### PIR Agenda Template

1. **Incident Summary** (5 min)
   - What happened
   - Timeline of events
   - Impact assessment

2. **Root Cause Analysis** (15 min)
   - Technical root cause
   - Contributing factors
   - Human factors

3. **Response Evaluation** (10 min)
   - What went well
   - What could be improved
   - Runbook effectiveness

4. **Corrective Actions** (10 min)
   - Immediate fixes implemented
   - Long-term improvements
   - Action item assignment

5. **Prevention Measures** (10 min)
   - Process improvements
   - System enhancements
   - Training needs

### Metrics to Track

| Metric | Target | Actual |
|--------|--------|--------|
| Time to detection | <50ms | |
| Time to fuel shutoff | <200ms | |
| Time to operator notification | <5 min | |
| Purge cycle completion | 5-15 min | |
| Total downtime | <2 hours | |
| Restart success rate | 100% first attempt | |

---

## Prevention Measures

### Regular Maintenance

| Task | Frequency | Responsible |
|------|-----------|-------------|
| Flame scanner calibration | Weekly | Instrument Technician |
| Flame scanner cleaning | Monthly | Maintenance |
| Ignition system inspection | Monthly | Maintenance |
| Fuel train inspection | Quarterly | Maintenance |
| Safety system testing | Annually | Safety Officer |

### Monitoring and Alerting

**Pre-emptive Alerts:**
```yaml
# Prometheus alert for flame instability
- alert: FlameInstabilityWarning
  expr: gl005_flame_stability_index < 70
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "Flame instability detected on {{ $labels.burner_id }}"
    description: "Stability index {{ $value }} below threshold"

# Alert for scanner degradation
- alert: FlameScannerDegraded
  expr: gl005_flame_scanner_quality < 80
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Flame scanner signal quality degraded"
```

### System Improvements

- [ ] Implement redundant flame scanners (voting logic 2-out-of-3)
- [ ] Add predictive flame stability analysis
- [ ] Enhance fuel supply monitoring (upstream pressure, quality)
- [ ] Improve ignition system diagnostics
- [ ] Add combustible gas detection at burner level

---

## Appendix

### A. Related Runbooks

- [runbook-sensor-failure.md](./runbook-sensor-failure.md) - Flame scanner failure procedures
- [runbook-emergency-shutdown.md](./runbook-emergency-shutdown.md) - Full ESD procedures
- [runbook-communication-loss.md](./runbook-communication-loss.md) - DCS/PLC failure recovery

### B. Reference Documentation

- NFPA 85 Chapter 6 - Flame Monitoring and Detection
- API 556 Section 8 - Safety Shutoff Systems
- GL-005 Safety Validation Logic: `calculators/safety_validator.py`
- Flame Scanner Connector: `integrations/flame_scanner_connector.py`

### C. Emergency Contacts

| Role | Contact |
|------|---------|
| On-Call Engineer | @oncall-gl005 (Slack) |
| Safety Officer | (XXX) XXX-XXXX (24/7) |
| Control Room | (XXX) XXX-XXXX |
| DCS Support | (XXX) XXX-XXXX |

### D. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-22 | GL-TechWriter | Initial version |
