# GL-005 Combusense Operational Runbooks Index

## Document Control
| Field | Value |
|-------|-------|
| Document ID | GL005-RB-INDEX |
| Version | 1.0.0 |
| Last Updated | 2025-12-22 |
| Owner | GreenLang Operations Team |
| Classification | OPERATIONAL |
| Review Cycle | Monthly |

---

## Overview

This index provides quick access to all operational runbooks for GL-005 CombustionControlAgent (Combusense). Use this document to rapidly identify the appropriate runbook during an incident.

---

## Quick Reference Table

| Incident Type | Severity | Runbook | Response Time |
|---------------|----------|---------|---------------|
| Flame loss (all burners) | P0 CRITICAL | [runbook-flame-loss.md](./runbook-flame-loss.md) | Immediate |
| Flame loss (single burner) | P1 HIGH | [runbook-flame-loss.md](./runbook-flame-loss.md) | 15 min |
| Flame instability | P2 MEDIUM | [runbook-flame-loss.md](./runbook-flame-loss.md) | 30 min |
| Safety sensor failed (no backup) | P0 CRITICAL | [runbook-sensor-failure.md](./runbook-sensor-failure.md) | Immediate |
| Safety sensor failed (backup active) | P1 HIGH | [runbook-sensor-failure.md](./runbook-sensor-failure.md) | 15 min |
| Control sensor failed | P2 MEDIUM | [runbook-sensor-failure.md](./runbook-sensor-failure.md) | 1 hour |
| DCS connection lost | P1 HIGH | [runbook-communication-loss.md](./runbook-communication-loss.md) | 15 min |
| PLC connection lost | P1 HIGH | [runbook-communication-loss.md](./runbook-communication-loss.md) | 15 min |
| SCADA timeout | P2 MEDIUM | [runbook-communication-loss.md](./runbook-communication-loss.md) | 30 min |
| All communications lost | P0 CRITICAL | [runbook-communication-loss.md](./runbook-communication-loss.md) | Immediate |
| Emergency shutdown triggered | P0 CRITICAL | [runbook-emergency-shutdown.md](./runbook-emergency-shutdown.md) | Immediate |
| Safety interlock trip | P0-P1 | [runbook-emergency-shutdown.md](./runbook-emergency-shutdown.md) | Immediate |
| Control cycle >100ms | P1 HIGH | [runbook-performance-degradation.md](./runbook-performance-degradation.md) | 30 min |
| Memory >90% | P1 HIGH | [runbook-performance-degradation.md](./runbook-performance-degradation.md) | 30 min |
| CPU >90% | P1 HIGH | [runbook-performance-degradation.md](./runbook-performance-degradation.md) | 30 min |
| Pod restarts | P2 MEDIUM | [runbook-performance-degradation.md](./runbook-performance-degradation.md) | 1 hour |

---

## Runbook Summary

### 1. Flame Loss Event
**File:** [runbook-flame-loss.md](./runbook-flame-loss.md)

**Covers:**
- Flame detection failure
- Flame scanner malfunction
- Fuel supply interruption
- Combustion air loss
- Ignition system failure

**Key Actions:**
1. Verify flame loss is genuine (not sensor fault)
2. Confirm fuel shutoff
3. Initiate purge cycle
4. Investigate root cause
5. Follow restart authorization procedure

---

### 2. Sensor Failure
**File:** [runbook-sensor-failure.md](./runbook-sensor-failure.md)

**Covers:**
- Temperature sensor failure
- Pressure sensor failure
- Flow meter failure
- Flame scanner failure
- Combustion analyzer failure

**Key Actions:**
1. Identify failed sensor
2. Activate fallback (automatic or manual)
3. Enable override if necessary
4. Schedule replacement
5. Verify system stability

---

### 3. Communication Loss
**File:** [runbook-communication-loss.md](./runbook-communication-loss.md)

**Covers:**
- DCS (OPC UA) connectivity loss
- PLC (Modbus TCP) connectivity loss
- SCADA timeout
- Network partition
- Manual mode operation

**Key Actions:**
1. Verify automatic failover
2. Check network connectivity
3. Enable manual mode if required
4. Diagnose root cause
5. Restore communications

---

### 4. Emergency Shutdown (ESD)
**File:** [runbook-emergency-shutdown.md](./runbook-emergency-shutdown.md)

**Covers:**
- Automatic ESD triggers
- Manual E-Stop activation
- Shutdown sequence verification
- Safe state confirmation
- Restart authorization

**Key Actions:**
1. Verify ESD sequence completed
2. Confirm safe state
3. Investigate trigger cause
4. Obtain restart authorization
5. Follow startup procedure

---

### 5. Performance Degradation
**File:** [runbook-performance-degradation.md](./runbook-performance-degradation.md)

**Covers:**
- Slow control response
- Resource exhaustion (CPU, memory)
- Database bottlenecks
- Integration latency
- Scaling procedures

**Key Actions:**
1. Identify performance bottleneck
2. Scale resources if needed
3. Apply immediate mitigations
4. Rollback if recent change caused issue
5. Plan long-term fix

---

## Escalation Matrix

### Severity Definitions

| Severity | Definition | Response Time | Resolution Target |
|----------|------------|---------------|-------------------|
| P0 CRITICAL | Safety risk, complete system failure, ESD | Immediate | 2 hours |
| P1 HIGH | Degraded operation, single backup remaining | 15 minutes | 4 hours |
| P2 MEDIUM | Partial degradation, monitoring-only impact | 1 hour | 8 hours |
| P3 LOW | Minor issue, no operational impact | Next business day | 1 week |

### Escalation Contacts

| Role | Contact Method | When to Contact |
|------|----------------|-----------------|
| **On-Call Engineer** | Slack: @oncall-gl005 | All P0/P1 incidents |
| **Engineering Manager** | Slack: @eng-manager-combustion | P0 incidents, P1 >2 hours |
| **Safety Officer** | Phone: (XXX) XXX-XXXX | Any safety-related P0 |
| **VP Engineering** | Phone: (XXX) XXX-XXXX | P0 >1 hour, P1 >4 hours |
| **Plant Manager** | Phone: (XXX) XXX-XXXX | Production impact |
| **Control Room** | Phone: (XXX) XXX-XXXX | All incidents (notify) |

### Escalation Flow

```
P0 CRITICAL:
  T+0:    On-Call Engineer paged
  T+15m:  Status update to #gl005-incidents
  T+30m:  Escalate to Engineering Manager + Safety Officer
  T+60m:  Escalate to VP Engineering
  T+90m:  Executive war room

P1 HIGH:
  T+0:    On-Call Engineer notified
  T+30m:  Status update to #gl005-incidents
  T+2h:   Escalate to Engineering Manager
  T+4h:   Escalate to VP Engineering

P2 MEDIUM:
  T+0:    On-Call Engineer notified
  T+2h:   Status update
  T+8h:   Escalate to Engineering Manager
```

---

## Contact Information Template

### Internal Contacts

| Role | Name | Phone | Email | Slack |
|------|------|-------|-------|-------|
| On-Call Engineer | Rotation | N/A | oncall-gl005@greenlang.io | @oncall-gl005 |
| Engineering Manager | [Name] | (XXX) XXX-XXXX | [email] | @[handle] |
| Safety Officer | [Name] | (XXX) XXX-XXXX | [email] | @[handle] |
| VP Engineering | [Name] | (XXX) XXX-XXXX | [email] | @[handle] |
| Plant Manager | [Name] | (XXX) XXX-XXXX | [email] | @[handle] |
| Control Room | N/A | (XXX) XXX-XXXX | control-room@[site].com | N/A |

### External Contacts

| Vendor/Service | Contact | Phone | Support Portal |
|----------------|---------|-------|----------------|
| DCS Vendor Support | [Vendor Name] | (XXX) XXX-XXXX | [URL] |
| PLC Vendor Support | [Vendor Name] | (XXX) XXX-XXXX | [URL] |
| SCADA Vendor Support | [Vendor Name] | (XXX) XXX-XXXX | [URL] |
| Network Operations | [Provider] | (XXX) XXX-XXXX | [URL] |
| Instrument Calibration | [Service] | (XXX) XXX-XXXX | [URL] |

### Emergency Services

| Service | Number |
|---------|--------|
| Fire Department | 911 / (XXX) XXX-XXXX |
| Ambulance | 911 |
| Poison Control | 1-800-222-1222 |
| Environmental Emergency | (XXX) XXX-XXXX |
| Security | (XXX) XXX-XXXX |

---

## Decision Trees

### Incident Classification Decision Tree

```
Is there an immediate safety risk?
├── YES → P0 CRITICAL
│   ├── Flame loss with fuel flowing → ESD Runbook
│   ├── Gas leak detected → ESD Runbook
│   ├── Fire detected → ESD Runbook
│   └── Safety interlock failed → ESD Runbook
│
└── NO → Is automated control functional?
    ├── NO → Is backup/manual control available?
    │   ├── YES → P1 HIGH (Communication Loss Runbook)
    │   └── NO → P0 CRITICAL (Communication Loss Runbook)
    │
    └── YES → Is performance degraded?
        ├── YES → Is degradation >20%?
        │   ├── YES → P1 HIGH (Performance Runbook)
        │   └── NO → P2 MEDIUM (Performance Runbook)
        │
        └── NO → Is this a sensor issue?
            ├── YES → Is it a safety sensor?
            │   ├── YES (no backup) → P0 CRITICAL (Sensor Runbook)
            │   ├── YES (backup active) → P1 HIGH (Sensor Runbook)
            │   └── NO → P2 MEDIUM (Sensor Runbook)
            │
            └── NO → P3 LOW (General Troubleshooting)
```

### Runbook Selection Matrix

| Symptom | Primary Runbook | Secondary Runbook |
|---------|-----------------|-------------------|
| "FLAME FAILURE" alarm | Flame Loss | Emergency Shutdown |
| Fuel valve closed unexpectedly | Emergency Shutdown | Flame Loss |
| "SENSOR FAILURE" alarm | Sensor Failure | Communication Loss |
| "DCS OFFLINE" alarm | Communication Loss | Sensor Failure |
| Control oscillating | Performance Degradation | Sensor Failure |
| Pod keeps restarting | Performance Degradation | Communication Loss |
| High latency alerts | Performance Degradation | Communication Loss |
| E-Stop activated | Emergency Shutdown | - |
| Temperature/Pressure trip | Emergency Shutdown | Sensor Failure |

---

## Incident Communication Templates

### P0 Incident Declaration (Slack)

```
:rotating_light: P0 INCIDENT - GL-005 CombustionControlAgent

**Status:** INVESTIGATING
**Started:** [TIMESTAMP UTC]
**Impact:** [Brief impact description]
**Incident Commander:** @[handle]

**Current Situation:**
[2-3 sentences describing current state]

**Immediate Actions:**
1. [Action being taken]
2. [Action being taken]

**Next Update:** 15 minutes

[Incident Ticket: INC-XXXXX]
```

### P1 Incident Declaration (Slack)

```
:warning: P1 INCIDENT - GL-005 CombustionControlAgent

**Status:** INVESTIGATING
**Started:** [TIMESTAMP UTC]
**Impact:** [Brief impact description]
**Owner:** @[handle]

**Current Situation:**
[2-3 sentences describing current state]

**Next Update:** 30 minutes

[Incident Ticket: INC-XXXXX]
```

### Incident Resolution (Slack)

```
:white_check_mark: RESOLVED - GL-005 CombustionControlAgent

**Incident:** [Brief description]
**Duration:** [X hours Y minutes]
**Root Cause:** [One sentence]
**Resolution:** [One sentence]

**Post-Incident Review:** Scheduled for [DATE TIME]

[Incident Ticket: INC-XXXXX]
```

---

## Related Documentation

### Operational Runbooks (This Directory)

| Document | Description |
|----------|-------------|
| [runbook-flame-loss.md](./runbook-flame-loss.md) | Flame loss detection and recovery |
| [runbook-sensor-failure.md](./runbook-sensor-failure.md) | Sensor failure handling |
| [runbook-communication-loss.md](./runbook-communication-loss.md) | DCS/PLC/SCADA connectivity |
| [runbook-emergency-shutdown.md](./runbook-emergency-shutdown.md) | ESD procedures |
| [runbook-performance-degradation.md](./runbook-performance-degradation.md) | Performance issues |

### General Runbooks (Parent Directory)

| Document | Description |
|----------|-------------|
| [INCIDENT_RESPONSE.md](../../runbooks/INCIDENT_RESPONSE.md) | General incident response |
| [TROUBLESHOOTING.md](../../runbooks/TROUBLESHOOTING.md) | General troubleshooting |
| [ROLLBACK_PROCEDURE.md](../../runbooks/ROLLBACK_PROCEDURE.md) | Deployment rollback |
| [SCALING_GUIDE.md](../../runbooks/SCALING_GUIDE.md) | Scaling procedures |
| [MAINTENANCE.md](../../runbooks/MAINTENANCE.md) | Scheduled maintenance |

### Technical Documentation

| Document | Location |
|----------|----------|
| System Architecture | `/docs/architecture/` |
| API Reference | `/docs/api/` |
| Safety Validator | `/calculators/safety_validator.py` |
| DCS Connector | `/integrations/dcs_connector.py` |
| Flame Scanner Connector | `/integrations/flame_scanner_connector.py` |

---

## Runbook Maintenance

### Review Schedule

| Review Type | Frequency | Participants |
|-------------|-----------|--------------|
| Content accuracy | Monthly | On-call engineers |
| Contact verification | Monthly | Operations team |
| Procedure drill | Quarterly | All operators |
| Full revision | Annually | Engineering + Safety |

### Update Process

1. Create branch from main
2. Update runbook content
3. Review with subject matter expert
4. Test commands/procedures
5. Submit pull request
6. Obtain approval from Safety Officer (for safety-critical changes)
7. Merge and deploy
8. Announce update to operations team

### Feedback Collection

Report runbook issues or improvements:
- Slack: #gl005-runbook-feedback
- Email: runbooks@greenlang.io
- Jira: Project GL-RUNBOOKS

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-22 | GL-TechWriter | Initial version |

---

## Appendix: Quick Commands Reference

### System Status

```bash
# Overall health
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/health | jq '.'

# Integration status
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/integrations/health | jq '.'

# Safety status
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/safety/status | jq '.'
```

### Logs

```bash
# Recent logs
kubectl logs -n greenlang deployment/gl-005-combustion-control --tail=100

# Error logs only
kubectl logs -n greenlang deployment/gl-005-combustion-control --tail=500 | grep -i error

# Follow logs
kubectl logs -n greenlang deployment/gl-005-combustion-control -f
```

### Metrics

```bash
# All metrics
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8001/metrics

# Specific metric
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8001/metrics | grep "gl005_control_cycle"
```

### Pod Management

```bash
# Pod status
kubectl get pods -n greenlang -l app=gl-005-combustion-control

# Pod resources
kubectl top pod -n greenlang -l app=gl-005-combustion-control

# Restart deployment
kubectl rollout restart deployment/gl-005-combustion-control -n greenlang
```
