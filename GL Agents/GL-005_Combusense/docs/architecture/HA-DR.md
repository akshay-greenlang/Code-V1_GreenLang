# GL-005 Combusense - High Availability & Disaster Recovery Architecture

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | GL-005 |
| Agent Name | Combusense |
| Full Name | Combustion Control & Sensing Agent |
| Version | 1.0.0 |
| Last Updated | 2025-12-22 |
| Owner | GreenLang Process Heat Team |

---

## 1. Executive Summary

GL-005 Combusense provides real-time combustion monitoring and control with advanced PID/feedforward control, stability analysis, emissions prediction, and comprehensive sensor integration. Given the safety-critical nature of combustion control, this architecture prioritizes:

- **Availability Target**: 99.99% (52.6 minutes downtime/year)
- **Recovery Point Objective (RPO)**: 1 minute
- **Recovery Time Objective (RTO)**: 5 minutes
- **Control Loop Latency**: < 100ms

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
       |  Combusense Pod x2    |<--- Real-time Sync --->|  Combusense Pod x2    |
       |  (Active Controller)  |                        |  (Hot Standby)        |
       |                       |                        |                       |
       |  +----------------+   |                        |  +----------------+   |
       |  | PID Controller |   |                        |  | PID Controller |   |
       |  | (Active)       |   |                        |  | (Tracking)     |   |
       |  +----------------+   |                        |  +----------------+   |
       |                       |                        |                       |
       |  +----------------+   |                        |  +----------------+   |
       |  | Stability      |   |                        |  | Stability      |   |
       |  | Analyzer       |   |                        |  | Analyzer       |   |
       |  +----------------+   |                        |  +----------------+   |
       |                       |                        |                       |
       |  +----------------+   |                        |  +----------------+   |
       |  | CQI Calculator |   |                        |  | CQI Calculator |   |
       |  +----------------+   |                        |  +----------------+   |
       |                       |                        |                       |
       |  Redis + PostgreSQL   |<--- Replication ------>|  Redis + PostgreSQL   |
       +-----------------------+                        +-----------------------+
                   |                                                |
       +-----------v-----------+                        +-----------v-----------+
       |  Sensor Array         |                        |  Sensor Array         |
       |  - Temperature        |                        |  - Temperature        |
       |  - Flame Scanner      |                        |  - Flame Scanner      |
       |  - O2/CO Analyzer     |                        |  - O2/CO Analyzer     |
       +-----------------------+                        +-----------------------+
                   |                                                |
       +-----------v-----------+                        +-----------v-----------+
       |  DCS/PLC              |<--- Redundant Link --->|  DCS/PLC              |
       |  (Primary)            |                        |  (Backup)             |
       +-----------------------+                        +-----------------------+
```

### 2.2 Component Redundancy

| Component | AZ-A | AZ-B | Total | Mode |
|-----------|------|------|-------|------|
| Application Pods | 2 | 2 | 4 | Active-Active |
| PID Controllers | 1 Active | 1 Tracking | 2 | Hot Standby |
| Redis Nodes | 2 | 1 | 3 | Sentinel |
| PostgreSQL | 1 Primary | 1 Standby | 2 | Streaming |
| InfluxDB | 1 | 1 | 2 | Replication |

### 2.3 Control System HA

```yaml
control_architecture:
  mode: active_standby
  bumpless_transfer: true
  tracking_enabled: true
  failover_time_ms: 50

  pid_controller:
    redundancy: dual
    anti_windup: true
    derivative_filter: true

  setpoint_manager:
    source_priority:
      - primary_controller
      - standby_controller
      - last_known_safe

  safety:
    watchdog_timeout_ms: 100
    flame_failure_response_ms: 50
    interlock_enabled: true
```

---

## 3. Failure Modes and Mitigation

### 3.1 Control System Failures

| Failure Mode | Detection | Mitigation | Recovery |
|--------------|-----------|------------|----------|
| PID Controller Crash | Watchdog | Bumpless transfer to standby | < 50ms |
| Setpoint Divergence | Limit check | Clamp to safe limits | Immediate |
| Control Output Failure | Actuator feedback | Hold last position | Alert |
| Communication Loss | Heartbeat | Fallback to safe state | Immediate |

### 3.2 Sensor Failures

| Failure Mode | Detection | Mitigation | Recovery |
|--------------|-----------|------------|----------|
| Temperature Sensor | Range/rate check | Use redundant sensor | Immediate |
| Flame Scanner | Signal quality | 2oo3 voting | 4 seconds |
| O2 Analyzer | Drift detection | Cross-check with CO | Calibration |
| Flow Meter | Consistency check | Calculated estimate | Alert |

### 3.3 Safety System Failures

| Failure Mode | Detection | Mitigation | Recovery |
|--------------|-----------|------------|----------|
| Flame Loss | 2oo3 voting | Emergency shutdown | 4 seconds |
| O2 Low | Threshold alarm | Reduce fuel | Immediate |
| CO High | Threshold alarm | Increase air | Immediate |
| Furnace Pressure | Interlock | Trip | Immediate |

---

## 4. RTO/RPO Targets

| Component | RPO | RTO | Notes |
|-----------|-----|-----|-------|
| Control State | 0 | 50ms | Hot standby |
| PID Tuning Parameters | 1 minute | 5 minutes | Replicated |
| Sensor Calibration | 5 minutes | 10 minutes | Database |
| CQI History | 5 minutes | 10 minutes | InfluxDB |
| Audit Logs | 0 | 15 minutes | Immutable |

---

## 5. Failover Procedures

### 5.1 Automatic Controller Failover

```yaml
controller_failover:
  trigger:
    - watchdog_timeout
    - health_check_failure
    - communication_loss

  procedure:
    1. detect_failure (< 10ms)
    2. verify_standby_ready (< 5ms)
    3. transfer_state (< 10ms)
    4. activate_standby (< 5ms)
    5. confirm_output_tracking (< 20ms)

  bumpless_transfer:
    enabled: true
    tracking_mode: integral_tracking
    ramp_rate_limit: true
```

### 5.2 Manual Failover

```bash
#!/bin/bash
# GL-005 Combusense Controller Failover

# 1. Verify standby is ready
kubectl exec -it gl-005-combusense-1 -n greenlang -- \
    curl localhost:8080/api/v1/controller/status

# 2. Initiate transfer
kubectl exec -it gl-005-combusense-0 -n greenlang -- \
    curl -X POST localhost:8080/api/v1/controller/transfer

# 3. Verify new active
kubectl exec -it gl-005-combusense-1 -n greenlang -- \
    curl localhost:8080/api/v1/controller/status
```

---

## 6. Monitoring

### 6.1 Control System Metrics

| Metric | Threshold | Severity |
|--------|-----------|----------|
| `combusense_pid_update_latency_ms` | > 10ms | Warning |
| `combusense_control_loop_latency_ms` | > 100ms | Critical |
| `combusense_setpoint_tracking_error` | > 2% | Warning |
| `combusense_cqi_score` | < 0.7 | Warning |
| `combusense_cqi_score` | < 0.5 | Critical |

### 6.2 Safety Metrics

| Metric | Threshold | Severity |
|--------|-----------|----------|
| `combusense_flame_signal` | < 20% | Critical |
| `combusense_o2_percent` | < 1% | Critical |
| `combusense_co_ppm` | > 200 | Critical |
| `combusense_furnace_pressure_mbar` | > 10 | Critical |

---

## 7. Real-time Communication HA

### 7.1 WebSocket HA

```yaml
websocket_ha:
  endpoints:
    - wss://gl-005-a.greenlang.io/ws
    - wss://gl-005-b.greenlang.io/ws

  failover:
    reconnect_delay_ms: 100
    max_retries: 5
    heartbeat_interval_ms: 1000

  message_ordering:
    enabled: true
    sequence_numbers: true
```

### 7.2 SSE Streaming HA

```yaml
sse_ha:
  endpoints:
    - https://gl-005-a.greenlang.io/sse
    - https://gl-005-b.greenlang.io/sse

  retry_after_ms: 3000
  last_event_id: true
```

---

## 8. Compliance Standards

| Standard | Requirement | Status |
|----------|-------------|--------|
| NFPA 86 | Ovens and Furnaces | Compliant |
| NFPA 85 | Boiler Safety | Compliant |
| IEC 61511 | SIS | SIL 2 |
| ISA-18.2 | Alarm Management | Compliant |
| EPA 40 CFR 60 | Emissions | Compliant |

---

## 9. Contact Information

| Role | Contact |
|------|---------|
| Primary On-Call | pager-sre@greenlang.ai |
| Safety Engineer | safety@greenlang.ai |
| Control Systems | controls@greenlang.ai |

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-22 | Initial release |
