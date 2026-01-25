# GL-002 Flameguard - High Availability & Disaster Recovery Architecture

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | GL-002 |
| Agent Name | Flameguard |
| Full Name | Boiler Efficiency Optimizer |
| Version | 1.0.0 |
| Last Updated | 2025-12-22 |
| Classification | Internal - Operations |
| Owner | GreenLang Process Heat Team |

---

## 1. Executive Summary

This document defines the High Availability (HA) and Disaster Recovery (DR) architecture for GL-002 Flameguard, the Boiler Efficiency Optimization Agent. Given the safety-critical nature of boiler operations (NFPA 85, IEC 61511 SIL 2), the architecture is designed to achieve:

- **Availability Target**: 99.95% (4.38 hours downtime/year)
- **Recovery Point Objective (RPO)**: 5 minutes
- **Recovery Time Objective (RTO)**: 10 minutes
- **Safety Integrity Level**: SIL 2

---

## 2. High Availability Architecture

### 2.1 Architecture Diagram

```
                                    +------------------------+
                                    |   Regional Load        |
                                    |   Balancer (NLB)       |
                                    +------------+-----------+
                                                 |
                         +-----------------------+------------------------+
                         |                                                |
             +-----------v-----------+                        +-----------v-----------+
             |   Primary AZ (AZ-A)   |                        |   Secondary AZ (AZ-B) |
             +-----------+-----------+                        +-----------+-----------+
                         |                                                |
             +-----------v-----------+                        +-----------v-----------+
             |   Flameguard Pod x2   |                        |   Flameguard Pod x2   |
             |   (Active)            |<--- Kafka Sync ------->|   (Active)            |
             +----------+------------+                        +----------+------------+
                        |                                                 |
             +----------v------------+                        +----------v------------+
             |   Redis Sentinel      |<--- Replication ------>|   Redis Sentinel      |
             |   (Master)            |                        |   (Replica)           |
             +----------+------------+                        +----------+------------+
                        |                                                 |
             +----------v------------+                        +----------v------------+
             |   PostgreSQL          |<--- Streaming -------->|   PostgreSQL          |
             |   (Primary)           |   Replication          |   (Standby)           |
             +----------+------------+                        +----------+------------+
                        |                                                 |
             +----------v------------+                        +----------v------------+
             |   Kafka Broker        |<--- MirrorMaker ------>|   Kafka Broker        |
             +----------------------+                        +------------------------+
                        |
             +----------v------------+
             |   SCADA/DCS           |
             |   Integration         |
             |   (Modbus/OPC-UA)     |
             +----------------------+
```

### 2.2 Component Redundancy

| Component | AZ-A | AZ-B | Total | Notes |
|-----------|------|------|-------|-------|
| Application Pods | 2 | 2 | 4 | Active-Active |
| Redis Nodes | 2 (1M+1R) | 1 (R) | 3 | Sentinel failover |
| PostgreSQL | 1 (Primary) | 1 (Standby) | 2 | Streaming replication |
| Kafka Brokers | 2 | 1 | 3 | Replication factor 3 |
| Sentinel Nodes | 2 | 1 | 3 | Quorum of 2 |

### 2.3 Safety-Critical Considerations

Given SIL 2 requirements:

```yaml
safety_architecture:
  fail_safe_mode: true
  voting_logic: "2oo3"  # 2-out-of-3 for flame detection
  watchdog_timeout_ms: 5000
  emergency_shutdown:
    enabled: true
    bypass_allowed: false
  interlock_integration:
    steam_pressure_trip: true
    drum_level_trip: true
    fuel_pressure_trip: true
```

---

## 3. Failure Modes and Mitigation

### 3.1 Application Layer Failures

| Failure Mode | Detection | Mitigation | Recovery Time |
|--------------|-----------|------------|---------------|
| Pod Crash | Liveness probe | Auto-restart | < 30 seconds |
| Node Failure | Node heartbeat | Reschedule to healthy node | < 2 minutes |
| AZ Failure | Health checks | Traffic shift to surviving AZ | < 1 minute |
| Calculation Error | Validation | Fallback to conservative values | Immediate |

### 3.2 Safety System Failures

| Failure Mode | Detection | Mitigation | Recovery Time |
|--------------|-----------|------------|---------------|
| Flame Scanner Failure | Signal loss | 2oo3 voting, trip if < 2 | 4 seconds |
| Interlock Timeout | Watchdog | Safe state activation | < 5 seconds |
| Communication Loss | Heartbeat | Last known safe state | Immediate |
| Optimization Divergence | Bounds check | Clamp to safe limits | Immediate |

### 3.3 Database Layer Failures

| Failure Mode | Detection | Mitigation | Recovery Time |
|--------------|-----------|------------|---------------|
| PostgreSQL Primary Failure | Patroni | Automatic failover | < 30 seconds |
| Replication Lag | Lag monitor | Alert, promote if critical | < 1 minute |
| Connection Pool Exhaustion | Metrics | Auto-scaling pool | < 30 seconds |

---

## 4. RTO/RPO Targets by Component

### 4.1 Recovery Objectives Matrix

| Component | RPO | RTO | Backup Frequency | Retention |
|-----------|-----|-----|------------------|-----------|
| **Application State** | 0 | 30 seconds | N/A | N/A |
| **PostgreSQL Data** | 5 minutes | 10 minutes | Continuous (WAL) | 30 days |
| **Redis Cache** | 10 minutes | 2 minutes | RDB every 5 min | 7 days |
| **Efficiency Calculations** | 0 | Immediate | Event-sourced | 7 years |
| **Audit Logs** | 0 | 15 minutes | Continuous | 7 years |
| **Configuration** | 0 | 2 minutes | GitOps | Unlimited |

### 4.2 Data Classification

| Data Type | Criticality | Notes |
|-----------|-------------|-------|
| Safety Interlock State | Critical | SIL 2 compliant |
| Efficiency Calculations | High | ASME PTC 4.1 auditable |
| Emissions Data | Regulatory | EPA 40 CFR 60 |
| Optimization History | High | Compliance evidence |

---

## 5. Failover Procedures

### 5.1 Automatic Failover

#### Pod-Level Failover
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 5
  failureThreshold: 1

# PDB ensures minimum 2 pods always available
minAvailable: 2
```

#### Database Failover (Patroni)
```bash
# Automatic via Patroni - no manual intervention required
# Leader election completes in < 30 seconds
```

### 5.2 Manual Failover Procedure

```bash
#!/bin/bash
# Manual AZ failover for GL-002 Flameguard

# Step 1: Verify failure
kubectl get pods -n greenlang -l app=gl-002-flameguard -o wide

# Step 2: Scale up healthy AZ
kubectl scale deployment gl-002-flameguard --replicas=4 -n greenlang

# Step 3: Verify health
kubectl exec -it $(kubectl get pod -l app=gl-002-flameguard -o jsonpath='{.items[0].metadata.name}' -n greenlang) -n greenlang -- curl localhost:8080/health

# Step 4: Update traffic routing if needed
kubectl patch service gl-002-flameguard -n greenlang --type='json' -p='[{"op": "replace", "path": "/spec/selector/az", "value": "az-b"}]'
```

---

## 6. Health Checks and Monitoring

### 6.1 Health Check Endpoints

| Endpoint | Purpose | Interval | Timeout |
|----------|---------|----------|---------|
| `/health` | Basic liveness | 10s | 5s |
| `/health/deep` | All dependencies | 30s | 10s |
| `/health/safety` | Safety system status | 5s | 3s |
| `/metrics` | Prometheus metrics | 15s | 5s |

### 6.2 Key Metrics for HA

| Metric | Alert Threshold | Severity |
|--------|-----------------|----------|
| `flameguard_efficiency_calculation_errors` | > 0 | Critical |
| `flameguard_safety_interlock_status` | != normal | Critical |
| `flameguard_scada_connection_status` | disconnected | Critical |
| `flameguard_request_latency_p99` | > 500ms | Warning |
| `flameguard_pod_available` | < 2 | Critical |

---

## 7. SCADA/DCS Integration HA

### 7.1 Redundant Connections

```yaml
scada_integration:
  primary:
    protocol: modbus_tcp
    host: scada-primary.plant.local
    port: 502
    timeout_ms: 3000
  secondary:
    protocol: modbus_tcp
    host: scada-secondary.plant.local
    port: 502
    timeout_ms: 3000
  failover:
    enabled: true
    auto_reconnect: true
    reconnect_delay_ms: 5000
    max_retries: 3
```

### 7.2 OPC-UA HA Configuration

```yaml
opcua_integration:
  endpoints:
    - opc.tcp://opcua-server-1:4840
    - opc.tcp://opcua-server-2:4840
  session:
    timeout_ms: 30000
    keep_alive_ms: 5000
  subscription:
    publishing_interval_ms: 1000
    queue_size: 10
```

---

## 8. Testing and Validation

### 8.1 Chaos Engineering Tests

| Test | Frequency | Success Criteria |
|------|-----------|------------------|
| Pod Kill | Weekly | Recovery < 30s, no safety impact |
| AZ Failure | Monthly | Traffic shift < 1 min |
| Database Failover | Weekly | RPO/RTO met |
| SCADA Disconnect | Monthly | Graceful degradation |
| Network Partition | Quarterly | Split-brain prevention |

### 8.2 Safety Validation Tests

| Test | Frequency | Requirement |
|------|-----------|-------------|
| Flame Failure Response | Monthly | < 4 seconds to safe state |
| Interlock Test | Weekly | 100% trip success |
| Watchdog Test | Daily | Timeout activates safe state |
| Voting Logic | Monthly | 2oo3 correct behavior |

---

## 9. Appendices

### 9.1 Contact Information

| Role | Contact | Escalation |
|------|---------|------------|
| Primary On-Call | pager-sre@greenlang.ai | 15 min |
| Safety Engineer | safety@greenlang.ai | 30 min |
| Plant Operations | ops@plant.local | Immediate |

### 9.2 Related Documents

- [DISASTER-RECOVERY.md](./DISASTER-RECOVERY.md)
- [SLA.md](./SLA.md)
- [NFPA 85 Compliance](../compliance/NFPA85.md)
- [IEC 61511 SIL 2](../compliance/IEC61511.md)

### 9.3 Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-22 | GL DevOps | Initial release |
