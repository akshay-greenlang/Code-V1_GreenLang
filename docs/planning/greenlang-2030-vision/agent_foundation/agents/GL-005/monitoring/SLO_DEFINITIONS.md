# GL-005 CombustionControlAgent - SLI/SLO Definitions

## Overview

This document defines Service Level Indicators (SLIs) and Service Level Objectives (SLOs) for GL-005 CombustionControlAgent, aligned with SIL-2 safety requirements and operational excellence standards.

## SLI/SLO Framework

### 1. Control Loop Latency SLO

**Service Level Indicator (SLI):**
```promql
gl005:control_latency:sli = histogram_quantile(0.95,
  sum(rate(gl005_control_loop_duration_seconds_bucket[5m])) by (le, unit_id)
)
```

**Service Level Objective (SLO):**
- **Target:** P95 control loop latency < 100ms
- **Measurement Window:** Rolling 28 days
- **Error Budget:** 1% (allows 0.28 days of degraded performance per 28-day window)
- **Compliance Period:** Evaluated every 5 minutes

**Rationale:**
- Real-time control systems require sub-100ms response for stability
- SIL-2 safety systems mandate deterministic timing
- Industry standard for combustion control is 50-100ms

**Business Impact:**
- **Below SLO:** Unstable combustion, safety risk, production quality degradation
- **Above SLO:** Optimal control responsiveness, stable operations

**Alert Thresholds:**
- **Warning:** P95 > 80ms for 5 minutes
- **Critical:** P95 > 100ms for 2 minutes

---

### 2. Control Success Rate SLO

**Service Level Indicator (SLI):**
```promql
gl005:control_success_rate:sli =
  sum(rate(gl005_control_loop_executions_total{status="success"}[5m])) by (unit_id) /
  sum(rate(gl005_control_loop_executions_total[5m])) by (unit_id)
```

**Service Level Objective (SLO):**
- **Target:** 99.0% control loop execution success rate
- **Measurement Window:** Rolling 28 days
- **Error Budget:** 1% (allows ~403 minutes of failed executions per 28-day window)
- **Compliance Period:** Evaluated every 5 minutes

**Rationale:**
- High-availability control systems require >99% success rate
- Matches industry standards for critical process control
- Allows for occasional transient failures without system degradation

**Business Impact:**
- **Below SLO:** Frequent control failures, manual intervention required, safety risk
- **Above SLO:** Reliable automated control, reduced operator workload

**Alert Thresholds:**
- **Warning:** Success rate < 99.5% for 10 minutes
- **Critical:** Success rate < 99.0% for 5 minutes

**Error Budget Burn Rate Alerts:**
- **Fast burn (1h window):** Burn rate > 14x (consumes 1.4% of budget per hour)
- **Moderate burn (6h window):** Burn rate > 6x (consumes 6% of budget in 6 hours)

---

### 3. Safety Response Time SLO (SIL-2 Requirement)

**Service Level Indicator (SLI):**
```promql
gl005:safety_response_time:sli = histogram_quantile(0.95,
  sum(rate(gl005_safety_check_duration_seconds_bucket[5m])) by (le, unit_id)
)
```

**Service Level Objective (SLO):**
- **Target:** P95 safety check duration < 20ms
- **Measurement Window:** Rolling 7 days (stricter for safety-critical)
- **Error Budget:** 0.1% (allows 0.007 days of degraded performance per 7-day window)
- **Compliance Period:** Evaluated every 30 seconds
- **Regulatory Requirement:** IEC 61511 SIL-2

**Rationale:**
- SIL-2 safety systems require <50ms response time (we target 20ms for margin)
- Safety checks run every control cycle (100ms), must complete in <20% of cycle time
- Regulatory compliance for industrial safety instrumented systems

**Business Impact:**
- **Below SLO:** Non-compliance with SIL-2 safety standards, regulatory violation, potential plant shutdown
- **Above SLO:** Full safety compliance, reliable protection

**Alert Thresholds:**
- **Warning:** P95 > 15ms for 2 minutes
- **Critical:** P95 > 20ms for 1 minute (immediate investigation required)

---

### 4. System Availability SLO

**Service Level Indicator (SLI):**
```promql
gl005:availability:sli = avg_over_time(gl005_safety_status[5m])
```

**Service Level Objective (SLO):**
- **Target:** 99.9% uptime (safety status = SAFE)
- **Measurement Window:** Rolling 30 days
- **Error Budget:** 0.1% (allows ~43 minutes of downtime per 30-day window)
- **Compliance Period:** Evaluated every 5 minutes

**Rationale:**
- High-availability process control requires "three nines" reliability
- Downtime includes both planned and unplanned outages
- Aligns with plant-wide availability targets

**Business Impact:**
- **Below SLO:** Production losses, revenue impact, customer commitments at risk
- **Above SLO:** Reliable production, maximized throughput

**Exclusions:**
- Planned maintenance windows (pre-scheduled)
- External infrastructure failures (power, network outside GreenLang control)

**Alert Thresholds:**
- **Critical:** Safety status transitions to UNSAFE (immediate alert)
- **Warning:** Availability < 99.9% over rolling 24h window

---

### 5. Emissions Compliance SLO

**Service Level Indicator (SLI):**
```promql
gl005:emissions_compliance:sli =
  (gl005_emissions_nox_ppm <= 50) and (gl005_emissions_co_ppm <= 100)
```

**Service Level Objective (SLO):**
- **Target:** 100% compliance with EPA emissions limits (NOx ≤50ppm, CO ≤100ppm)
- **Measurement Window:** Rolling 30 days
- **Error Budget:** 0% (zero tolerance for regulatory violations)
- **Compliance Period:** Evaluated every 1 minute

**Rationale:**
- EPA Clean Air Act requires continuous emissions compliance
- Zero tolerance for regulatory violations (fines, legal liability)
- Environmental responsibility and ESG commitments

**Business Impact:**
- **Below SLO:** EPA fines ($25,000-$50,000 per day), permit revocation risk, reputation damage
- **Above SLO:** Full regulatory compliance, avoided penalties

**Alert Thresholds:**
- **Warning:** Emissions approaching limits (NOx >40ppm or CO >80ppm) for 5 minutes
- **Critical:** Emissions exceed limits for 10 minutes (immediate corrective action)

---

### 6. Agent Execution Time SLOs

**Service Level Indicators (SLIs):**
```promql
# Data Intake Agent
gl005:data_intake:sli = histogram_quantile(0.95,
  sum(rate(gl005_agent_duration_seconds_bucket{agent="data_intake"}[5m])) by (le)
)

# Combustion Analysis Agent
gl005:combustion_analysis:sli = histogram_quantile(0.95,
  sum(rate(gl005_agent_duration_seconds_bucket{agent="combustion_analysis"}[5m])) by (le)
)

# Control Optimizer Agent
gl005:control_optimizer:sli = histogram_quantile(0.95,
  sum(rate(gl005_agent_duration_seconds_bucket{agent="control_optimizer"}[5m])) by (le)
)

# Command Execution Agent
gl005:command_execution:sli = histogram_quantile(0.95,
  sum(rate(gl005_agent_duration_seconds_bucket{agent="command_execution"}[5m])) by (le)
)

# Audit & Safety Agent
gl005:audit_safety:sli = histogram_quantile(0.95,
  sum(rate(gl005_agent_duration_seconds_bucket{agent="audit_safety"}[5m])) by (le)
)
```

**Service Level Objectives (SLOs):**

| Agent | P95 Target | Error Budget | Window | Alert Threshold |
|-------|-----------|--------------|--------|-----------------|
| Data Intake | <50ms | 5% | 28 days | >60ms for 5min |
| Combustion Analysis | <30ms | 5% | 28 days | >40ms for 5min |
| Control Optimizer | <50ms | 5% | 28 days | >60ms for 5min |
| Command Execution | <20ms | 5% | 28 days | >30ms for 5min |
| Audit & Safety | <10ms | 2% | 7 days | >15ms for 2min |

**Rationale:**
- Agent execution times must sum to <100ms total control loop latency
- Safety-critical agents (Audit & Safety) have stricter SLOs
- Targets based on profiling of optimized implementations

**Business Impact:**
- **Below SLO:** Agent bottleneck contributing to overall control latency, degraded performance
- **Above SLO:** Optimal agent performance, control loop meets latency targets

---

### 7. Integration Latency SLO

**Service Level Indicators (SLIs):**
```promql
# DCS Read Latency
gl005:dcs_latency:sli = gl005_dcs_read_latency_ms

# PLC Read Latency
gl005:plc_latency:sli = gl005_plc_read_latency_ms
```

**Service Level Objective (SLO):**
- **Target:** P95 integration latency < 50ms (both DCS and PLC)
- **Measurement Window:** Rolling 28 days
- **Error Budget:** 5% (allows occasional network delays)
- **Compliance Period:** Evaluated every 5 minutes

**Rationale:**
- External integration latency is largest contributor to control loop delay
- 50ms target allows buffer within 100ms total control loop budget
- Network/infrastructure variability requires error budget

**Business Impact:**
- **Below SLO:** Integration delays causing control loop latency violations, unstable control
- **Above SLO:** Fast data acquisition, responsive control

**Alert Thresholds:**
- **Warning:** P95 > 75ms for 10 minutes
- **Critical:** P95 > 100ms for 5 minutes

---

## Error Budget Policy

### Error Budget Calculation
```
Error Budget Remaining = SLO Target - Actual SLI
```

Example for 99% SLO over 28 days:
- Total time: 40,320 minutes
- Allowed error: 403.2 minutes (1%)
- If 200 minutes consumed: 50% budget remaining

### Error Budget Burn Rate

**Fast Burn Alert (1h window):**
```promql
gl005:error_budget_burn_rate:1h =
  (1 - gl005:control_success_rate:sli) / (1 - 0.99)

# Alert if burn_rate > 14x
# (consumes 14% of monthly budget per hour)
```

**Moderate Burn Alert (6h window):**
```promql
gl005:error_budget_burn_rate:6h =
  (1 - avg_over_time(gl005:control_success_rate:sli[6h])) / (1 - 0.99)

# Alert if burn_rate > 6x
# (consumes 36% of monthly budget in 6 hours)
```

### Error Budget Actions

| Budget Remaining | Action |
|------------------|--------|
| **>75%** | Normal operations, continue feature development |
| **50-75%** | Monitor closely, defer non-critical changes |
| **25-50%** | Code freeze on risky changes, focus on reliability |
| **<25%** | Emergency response, halt all feature work, investigate root cause |
| **0%** | Post-mortem required, executive review, reliability sprint |

---

## SLO Review Cadence

### Weekly Review
- Error budget consumption trends
- Alert noise analysis (false positives/negatives)
- Near-miss incidents (SLO violations avoided)

### Monthly Review
- SLO achievement vs. targets
- Error budget allocation effectiveness
- SLO adjustments based on operational experience

### Quarterly Review
- SLO alignment with business objectives
- Regulatory compliance assessment
- Capacity planning based on SLI trends

---

## SLO Dashboard Panels

### Recommended Grafana Panels

**1. SLO Compliance Summary (Single Stat)**
```promql
# Green if all SLOs met, red if any violated
min(
  (gl005:control_latency:sli < 0.1) and
  (gl005:control_success_rate:sli >= 0.99) and
  (gl005:safety_response_time:sli < 0.02) and
  (gl005:availability:sli >= 0.999) and
  (gl005:emissions_compliance:sli == 1)
)
```

**2. Error Budget Burn Rate (Graph)**
```promql
gl005:error_budget_burn_rate:1h  # Fast burn
gl005:error_budget_burn_rate:6h  # Moderate burn
```

**3. SLI vs SLO Comparison (Gauge)**
```promql
# Control Latency: Gauge with threshold at 0.1
gl005:control_latency:sli

# Success Rate: Gauge with threshold at 0.99
gl005:control_success_rate:sli
```

**4. Error Budget Remaining (Bar Chart)**
```promql
# Calculate remaining budget percentage
(1 - (1 - gl005:control_success_rate:sli) / (1 - 0.99)) * 100
```

---

## Appendix: SLO Calculation Examples

### Example 1: Control Success Rate SLO

**Scenario:**
- SLO: 99% success rate over 28 days
- Actual performance: 98.5% over last 28 days

**Calculation:**
```
Error Budget = 1 - 0.99 = 1% (403.2 minutes per 28 days)
Actual Errors = 1 - 0.985 = 1.5% (604.8 minutes)
Budget Consumed = 604.8 / 403.2 = 150% (over budget!)
```

**Result:** SLO violated, error budget exceeded by 50%

**Action:** Emergency response, halt feature deployments, investigate root cause

---

### Example 2: Control Latency SLO

**Scenario:**
- SLO: P95 latency < 100ms
- Actual P95 latency: 85ms

**Calculation:**
```
Target = 100ms
Actual = 85ms
Margin = 100 - 85 = 15ms (15% headroom)
```

**Result:** SLO met with comfortable margin

**Action:** Continue normal operations, monitor trends

---

## References

- IEC 61511: Safety Instrumented Systems for the Process Industry Sector
- EPA Clean Air Act Emissions Standards (40 CFR Part 60)
- Google SRE Book - Service Level Objectives
- GreenLang Monitoring Standards v2.1
