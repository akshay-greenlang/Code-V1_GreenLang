# GL-005 Combusense - Service Level Agreement (SLA)

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | GL-005 |
| Agent Name | Combusense (Combustion Control & Sensing Agent) |
| Effective Date | 2025-01-01 |

---

## 1. Service Overview

GL-005 Combusense provides:
- Real-time PID control with feedforward compensation
- Combustion Quality Index (CQI) calculation
- Air-fuel ratio optimization
- Flame stability monitoring
- Emissions estimation (NOx, CO)
- WebSocket/SSE real-time streaming

---

## 2. Availability Targets

| Tier | Availability | Monthly Downtime |
|------|--------------|------------------|
| **Production** | 99.99% | 4.38 minutes |

### 2.1 Control System Availability

| Component | Availability | Notes |
|-----------|--------------|-------|
| PID Controller | 99.99% | Hot standby |
| Control Loop | 99.99% | < 100ms latency |
| Safety Interlocks | 100% | Hardware-backed |

---

## 3. Performance SLOs

### 3.1 Control System Performance

| Operation | Target | Maximum |
|-----------|--------|---------|
| PID Update Cycle | 1ms | 10ms |
| Control Loop Latency | 50ms | 100ms |
| Setpoint Tracking Error | < 2% | 5% |
| Disturbance Rejection | < 5s | 10s |

### 3.2 Calculation Performance

| Operation | P50 | P95 | P99 |
|-----------|-----|-----|-----|
| PID Calculation | 0.1ms | 0.5ms | 1ms |
| Stoichiometry | 0.5ms | 2ms | 5ms |
| Stability Analysis | 2ms | 10ms | 20ms |
| CQI Calculation | 1ms | 5ms | 10ms |
| Emissions Estimation | 2ms | 10ms | 20ms |

### 3.3 Throughput

| Metric | Target |
|--------|--------|
| Control cycles/second | > 100 |
| Sensor updates/second | > 100 |
| WebSocket messages/second | > 1000 |

---

## 4. Recovery Objectives

| Objective | Target |
|-----------|--------|
| RTO | 5 minutes |
| RPO | 1 minute |
| Controller Failover | 50ms |

---

## 5. Incident Response

| Severity | Acknowledgment | Resolution |
|----------|----------------|------------|
| P1 (Safety) | 5 minutes | 15 minutes |
| P2 (Critical) | 10 minutes | 30 minutes |
| P3 (Major) | 30 minutes | 2 hours |
| P4 (Minor) | 2 hours | 8 hours |

### 5.1 Safety Incidents

For any safety-related incident:
- Immediate acknowledgment (< 2 minutes)
- Safe state confirmed (< 5 minutes)
- Root cause analysis (< 24 hours)
- Corrective action plan (< 48 hours)

---

## 6. Compliance

| Standard | Requirement | Status |
|----------|-------------|--------|
| NFPA 86 | Oven safety | Compliant |
| NFPA 85 | Boiler safety | Compliant |
| IEC 61511 | SIS | SIL 2 |
| ISA-18.2 | Alarms | Compliant |
| EPA 40 CFR 60 | Emissions | Compliant |

---

## 7. Determinism Guarantee

| Metric | Target |
|--------|--------|
| Reproducibility | 100% |
| Hash verification | SHA-256 |
| Same input = Same output | Guaranteed |

All calculations are deterministic with no LLM in the control path.

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-22 | Initial release |
