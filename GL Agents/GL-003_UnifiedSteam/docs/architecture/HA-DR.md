# GL-003 UnifiedSteam - High Availability & Disaster Recovery Architecture

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | GL-003 |
| Agent Name | UnifiedSteam |
| Full Name | Steam System Optimizer |
| Version | 1.0.0 |
| Last Updated | 2025-12-22 |
| Owner | GreenLang Process Heat Team |

---

## 1. Executive Summary

GL-003 UnifiedSteam is the comprehensive steam system optimization agent providing IAPWS-IF97 thermodynamic calculations, steam trap diagnostics, desuperheater control, and condensate recovery optimization.

- **Availability Target**: 99.95% (4.38 hours downtime/year)
- **Recovery Point Objective (RPO)**: 5 minutes
- **Recovery Time Objective (RTO)**: 10 minutes

---

## 2. High Availability Architecture

### 2.1 Architecture Diagram

```
                              +------------------------+
                              |   Load Balancer        |
                              +------------+-----------+
                                           |
                   +-----------------------+------------------------+
                   |                                                |
       +-----------v-----------+                        +-----------v-----------+
       |   AZ-A                |                        |   AZ-B                |
       +-----------------------+                        +-----------------------+
       |                       |                        |                       |
       |  UnifiedSteam Pod x2  |<--- Kafka Sync ------->|  UnifiedSteam Pod x2  |
       |  (Active)             |                        |  (Active)             |
       |                       |                        |                       |
       |  +----------------+   |                        |  +----------------+   |
       |  | IAPWS-IF97     |   |                        |  | IAPWS-IF97     |   |
       |  | Calculator     |   |                        |  | Calculator     |   |
       |  +----------------+   |                        |  +----------------+   |
       |                       |                        |                       |
       |  +----------------+   |                        |  +----------------+   |
       |  | Trap Diagnostics|  |                        |  | Trap Diagnostics|  |
       |  | Engine         |   |                        |  | Engine         |   |
       |  +----------------+   |                        |  +----------------+   |
       |                       |                        |                       |
       |  Redis + PostgreSQL   |<--- Replication ------>|  Redis + PostgreSQL   |
       +-----------------------+                        +-----------------------+
                   |                                                |
       +-----------v-----------+                        +-----------v-----------+
       |  Steam Trap Sensors   |                        |  Acoustic Monitors    |
       |  (OPC-UA/Modbus)      |                        |  (Wireless/Wired)     |
       +-----------------------+                        +-----------------------+
```

### 2.2 Component Redundancy

| Component | AZ-A | AZ-B | Total |
|-----------|------|------|-------|
| Application Pods | 2 | 2 | 4 |
| Redis Nodes | 2 | 1 | 3 |
| PostgreSQL | 1 Primary | 1 Standby | 2 |
| InfluxDB (Telemetry) | 1 | 1 | 2 |

---

## 3. Failure Modes and Mitigation

### 3.1 Thermodynamic Calculation Failures

| Failure Mode | Detection | Mitigation | Recovery |
|--------------|-----------|------------|----------|
| IAPWS Region Error | Input validation | Boundary clamp | Immediate |
| Calculation Timeout | Watchdog | Cached result | < 1 second |
| Invalid Input | Schema validation | Reject with error | Immediate |

### 3.2 Steam Trap Monitoring Failures

| Failure Mode | Detection | Mitigation | Recovery |
|--------------|-----------|------------|----------|
| Sensor Offline | Heartbeat timeout | Mark degraded | Alert |
| Acoustic Signal Loss | Quality threshold | Temperature fallback | Immediate |
| Database Connection | Health check | Retry with backoff | < 30 seconds |

---

## 4. RTO/RPO Targets

| Component | RPO | RTO |
|-----------|-----|-----|
| Application State | 0 | 30 seconds |
| Trap Diagnostics Data | 5 minutes | 10 minutes |
| Thermodynamic Cache | 15 minutes | 2 minutes |
| Audit Logs | 0 | 15 minutes |

---

## 5. Failover Procedures

### 5.1 Automatic Failover

```yaml
livenessProbe:
  httpGet:
    path: /api/v1/health
    port: 8000
  periodSeconds: 10
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /api/v1/ready
    port: 8000
  periodSeconds: 5
```

### 5.2 Manual Failover

```bash
#!/bin/bash
# GL-003 UnifiedSteam Manual Failover

# Verify primary health
kubectl get pods -n greenlang -l app=gl-003-unifiedsteam

# Scale up secondary
kubectl scale deployment gl-003-unifiedsteam --replicas=4 -n greenlang

# Promote database
kubectl exec -it gl-003-postgres-0 -n greenlang -- patronictl failover
```

---

## 6. Monitoring

### 6.1 Key Metrics

| Metric | Threshold | Severity |
|--------|-----------|----------|
| `unifiedsteam_iapws_calculation_errors` | > 0 | Warning |
| `unifiedsteam_trap_failure_rate` | > 5% | Warning |
| `unifiedsteam_desuperheater_deviation` | > 5C | Warning |
| `unifiedsteam_condensate_return_rate` | < 80% | Warning |

---

## 7. Contact Information

| Role | Contact |
|------|---------|
| Primary On-Call | pager-sre@greenlang.ai |
| Steam System Expert | steam-team@greenlang.ai |

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-22 | Initial release |
