# GL-008 SteamTrapInspector Service Level Objectives (SLOs)

**Version:** 1.0
**Last Updated:** 2025-11-26
**Owner:** GL-008 Platform Team
**Review Frequency:** Quarterly

---

## Overview

This document defines the Service Level Objectives (SLOs) for the GL-008 SteamTrapInspector agent. These SLOs represent our commitment to reliability, accuracy, and performance for steam trap inspection and failure detection.

**SLO Philosophy:**
- **User-Centric:** SLOs reflect what matters to maintenance teams and facility operators
- **Measurable:** All SLOs are backed by Prometheus metrics with clear measurement methodology
- **Achievable:** Targets are ambitious but realistic based on system capabilities
- **Business-Aligned:** SLOs support cost savings, safety, and sustainability goals

---

## SLO Summary Table

| SLO ID | Service Level Indicator (SLI) | Target | Measurement Window | Severity if Violated |
|--------|-------------------------------|--------|-------------------|---------------------|
| SLO-001 | Detection Accuracy | >95% | Rolling 24h | High |
| SLO-002 | Inspection Latency (P95) | <3 seconds | Rolling 1h | High |
| SLO-003 | System Availability | 99.9% | Rolling 30d | Critical |
| SLO-004 | Energy Loss Calculation Accuracy | ±5% | Per calculation | High |
| SLO-005 | False Positive Rate | <5% | Rolling 24h | High |
| SLO-006 | Fleet Coverage | >99% | Rolling 24h | Medium |

---

## SLO-001: Detection Accuracy

### Objective
Ensure steam trap failure detection is highly accurate to maximize cost savings and minimize false alarms.

### Service Level Indicator (SLI)
```
Detection Accuracy = (True Positives) / (True Positives + False Positives + False Negatives)
```

### Target
**>95%** over any rolling 24-hour period

### Measurement
**Prometheus Query:**
```promql
sum(gl008_true_positives_total) /
(sum(gl008_true_positives_total) +
 sum(gl008_false_positives_total) +
 sum(gl008_false_negatives_total))
```

**Data Source:**
- `gl008_true_positives_total` - Failures confirmed by technician inspection
- `gl008_false_positives_total` - False alarms (healthy traps flagged as failed)
- `gl008_false_negatives_total` - Missed failures (failed traps not detected)

**Measurement Window:** Rolling 24 hours
**Evaluation Frequency:** Every 5 minutes

### Rationale
- **95% target** balances detection sensitivity with operational efficiency
- Industry benchmark: Manual inspection accuracy typically 85-90%
- GL-008 provides 5-10% improvement over manual methods
- Higher accuracy = higher ROI from inspection program

### Consequences of Violation
- **Customer Impact:** Reduced trust in detection results
- **Business Impact:** Lower cost savings, wasted maintenance effort
- **Alert:** `GL008SLODetectionAccuracy` fires after 1 hour below target

### Improvement Actions
1. Analyze false positive/negative patterns by failure mode
2. Retrain ML models with recent labeled data
3. Adjust detection thresholds per trap type
4. Improve sensor calibration procedures
5. Review technician feedback loop

---

## SLO-002: Inspection Latency (P95)

### Objective
Ensure inspections complete quickly to enable real-time operational decisions and high throughput.

### Service Level Indicator (SLI)
```
P95 Inspection Latency = 95th percentile of end-to-end inspection time
```

### Target
**<3 seconds** (P95) over any rolling 1-hour period

### Measurement
**Prometheus Query:**
```promql
histogram_quantile(0.95,
  sum(rate(gl008_inspection_latency_seconds_bucket[1h])) by (le)
)
```

**Data Source:**
- `gl008_inspection_latency_seconds_bucket` - Histogram of inspection times from data acquisition through result output

**Measurement Window:** Rolling 1 hour
**Evaluation Frequency:** Every 1 minute

### Rationale
- **3-second target** enables real-time inspection during facility rounds
- Supports inspection of 20+ traps per hour per technician
- Allows for immediate decision-making on critical failures
- P95 (not average) ensures consistent experience even under load

### Consequences of Violation
- **Customer Impact:** Slower facility rounds, reduced productivity
- **Business Impact:** Lower inspection throughput, delayed failure detection
- **Alert:** `GL008SLOInspectionLatency` fires after 30 minutes above target

### Performance Breakdown
Typical latency budget (P95):
- Sensor data acquisition: 0.5s
- Acoustic analysis: 0.8s
- Thermal analysis: 0.6s
- Ultrasonic analysis: 0.5s
- Classification/inference: 0.3s
- Result persistence: 0.2s
- **Total:** ~2.9s (100ms buffer)

### Improvement Actions
1. Optimize analysis algorithms (vectorization, parallel processing)
2. Scale compute resources during peak hours
3. Implement edge processing for sensor data
4. Cache emission factors and trap specifications
5. Database query optimization

---

## SLO-003: System Availability

### Objective
Ensure GL-008 inspection system is operational and accessible 24/7 to support continuous monitoring.

### Service Level Indicator (SLI)
```
Availability = (Time System Healthy) / (Total Time)
```

### Target
**99.9%** over any rolling 30-day period

**Allowed Downtime:**
- Per month: 43.2 minutes
- Per week: 10.1 minutes
- Per day: 1.4 minutes

### Measurement
**Prometheus Query:**
```promql
avg_over_time(gl008_system_health{component="inspection_engine"}[30d])
```

**Data Source:**
- `gl008_system_health` - Health check status (1=healthy, 0=unhealthy)
- Checked every 30 seconds via `/health` endpoint

**Measurement Window:** Rolling 30 days
**Evaluation Frequency:** Every 1 minute

### Rationale
- **99.9% (three nines)** is standard for critical business systems
- Allows for planned maintenance windows (e.g., monthly updates)
- Higher than 99.9% would require costly redundancy
- Steam trap inspection is critical but not life-safety (vs 99.99% for medical)

### Health Check Criteria
System considered healthy when ALL components pass:
- Inspection engine: Responding to health checks
- Database: Connection pool responsive, query latency <100ms
- Sensor interface: Able to acquire test data
- API: Responding to requests with <5% error rate

### Consequences of Violation
- **Customer Impact:** Unable to perform inspections, delayed failure detection
- **Business Impact:** Lost inspection data, potential undetected failures escalating
- **Alert:** `GL008SLOAvailability` fires after 5 minutes below target

### Improvement Actions
1. Implement multi-region deployment for failover
2. Add circuit breakers for external dependencies
3. Improve graceful degradation (e.g., offline mode)
4. Automated recovery procedures
5. Reduce dependency on external services

---

## SLO-004: Energy Loss Calculation Accuracy

### Objective
Ensure energy loss calculations are accurate to support reliable ROI analysis and cost justification.

### Service Level Indicator (SLI)
```
Calculation Accuracy = |Calculated Energy Loss - Verified Energy Loss| / Verified Energy Loss
```

### Target
**±5%** error for 95% of calculations

### Measurement
**Prometheus Query:**
```promql
abs(gl008_energy_loss_kw - gl008_energy_loss_kw_verified) /
gl008_energy_loss_kw_verified
```

**Data Source:**
- `gl008_energy_loss_kw` - Calculated energy loss from agent
- `gl008_energy_loss_kw_verified` - Verified loss from manual calculation or flow meters

**Measurement Window:** Per calculation (verified sample: 10% of failures)
**Evaluation Frequency:** Weekly batch verification

### Rationale
- **±5% accuracy** is industry standard for energy audits
- Supports financial decisions (ROI calculations within acceptable margin)
- More accurate than traditional manual methods (±10-15% typical)
- Zero-hallucination approach ensures deterministic calculation

### Calculation Methodology
Energy loss calculated using:
```
Energy Loss (kW) = Steam Loss Rate (kg/hr) × Enthalpy Difference (kJ/kg) / 3600
```

Where:
- Steam loss rate: From acoustic/ultrasonic signature + trap orifice size
- Enthalpy difference: Steam tables lookup (pressure, temperature)
- All inputs verified against trap specifications and steam system data

### Consequences of Violation
- **Customer Impact:** Incorrect ROI calculations, poor repair prioritization
- **Business Impact:** Reduced credibility, financial reporting errors
- **Alert:** `GL008SLOEnergyCalculationAccuracy` fires after 1 hour of violations

### Improvement Actions
1. Validate steam property lookups against NIST data
2. Improve steam loss rate estimation algorithms
3. Calibrate acoustic/ultrasonic sensors more frequently
4. Cross-validate with flow meter data where available
5. Review calculation formulas with thermodynamics experts

---

## SLO-005: False Positive Rate

### Objective
Minimize false alarms to maintain technician trust and prevent wasted maintenance effort.

### Service Level Indicator (SLI)
```
False Positive Rate = (False Positives) / (True Positives + False Positives)
```

### Target
**<5%** over any rolling 24-hour period

### Measurement
**Prometheus Query:**
```promql
sum(increase(gl008_false_positives_total[24h])) /
(sum(increase(gl008_true_positives_total[24h])) +
 sum(increase(gl008_false_positives_total[24h])))
```

**Data Source:**
- `gl008_false_positives_total` - Failures flagged by system but confirmed healthy by technician
- `gl008_true_positives_total` - Failures confirmed by technician

**Measurement Window:** Rolling 24 hours
**Evaluation Frequency:** Every 15 minutes

### Rationale
- **<5% false positives** maintains technician confidence in alerts
- Every false positive wastes ~30 minutes technician time + trip cost ($50-100)
- Lower false positive rate increases, but risks missing real failures (trade-off with recall)
- 5% allows for edge cases (unusual operating conditions, sensor noise)

### Impact Analysis
Example facility with 1000 traps, 10% failure rate:
- 100 failures detected per inspection cycle
- At 5% false positive rate: 5 false alarms
- Cost: 5 × $75 = $375 wasted maintenance cost
- Acceptable trade-off for 95% detection accuracy

### Consequences of Violation
- **Customer Impact:** Technician alert fatigue, reduced trust in system
- **Business Impact:** Wasted maintenance resources, lower ROI
- **Alert:** `GL008SLOFalsePositiveRate` fires after 2 hours above target

### Improvement Actions
1. Increase detection confidence thresholds (may reduce recall)
2. Implement multi-sensor fusion (require agreement from 2+ sensors)
3. Add temporal filtering (confirm failure over multiple inspections)
4. Filter out known benign conditions (e.g., startup transients)
5. Retrain models with better labeled negative examples

---

## SLO-006: Fleet Coverage

### Objective
Ensure all traps in the fleet are inspected regularly to prevent undetected failures from escalating.

### Service Level Indicator (SLI)
```
Fleet Coverage = (Traps Inspected in Period) / (Total Traps in Fleet) × 100%
```

### Target
**>99%** of fleet inspected every 24 hours

### Measurement
**Prometheus Query:**
```promql
avg(gl008_fleet_coverage_percent{time_window="24h"})
```

**Data Source:**
- `gl008_fleet_coverage_percent` - Percentage of registered traps with inspection in window

**Measurement Window:** Rolling 24 hours
**Evaluation Frequency:** Every 1 hour

### Rationale
- **99% coverage** ensures comprehensive fleet monitoring
- Allows for 1% exceptions (e.g., traps in maintenance, decommissioned, inaccessible)
- Daily inspection frequency detects degradation early (before catastrophic failure)
- Aligns with industry best practices (monthly manual inspection → daily automated)

### Coverage Calculation
```
Coverage = COUNT(DISTINCT trap_id WHERE last_inspection < 24h ago) /
           COUNT(DISTINCT trap_id WHERE status != 'decommissioned')
```

### Consequences of Violation
- **Customer Impact:** Undetected failures may escalate, higher energy losses
- **Business Impact:** Reduced effectiveness of inspection program, missed savings
- **Alert:** `GL008SLOFleetCoverage` fires after 4 hours below target

### Improvement Actions
1. Investigate coverage gaps (trap locations, sensor connectivity)
2. Add redundant inspection paths (mobile sensors for low-coverage areas)
3. Prioritize high-value traps when full coverage not achievable
4. Automate trap registration/decommissioning workflows
5. Scale inspection capacity (more sensors, faster processing)

---

## SLO Monitoring & Reporting

### Dashboards
- **Grafana Dashboard:** `gl008_trap_performance.json`
  - Real-time SLO compliance status
  - Historical SLO trends (7d, 30d, 90d)
  - SLO burn rate (error budget consumption)

### Alerting
All SLO violations trigger Prometheus alerts:
- **Critical SLOs:** Availability → PagerDuty escalation
- **High SLOs:** Accuracy, Latency, False Positive Rate → Slack + email
- **Medium SLOs:** Fleet Coverage → Email to operations team

### Error Budgets
Each SLO has an error budget representing allowed non-compliance:

| SLO | Target | Error Budget (30d) | Budget Consumption Alert |
|-----|--------|-------------------|-------------------------|
| Detection Accuracy | >95% | 5% × 720h = 36h | >50% budget consumed |
| Inspection Latency | <3s P95 | 5% × requests | >50% budget consumed |
| Availability | 99.9% | 43.2 min | >50% budget consumed |
| Energy Accuracy | ±5% | 5% × calculations | >50% budget consumed |
| False Positive Rate | <5% | 5% × detections | >50% budget consumed |
| Fleet Coverage | >99% | 1% × fleet | >50% budget consumed |

### Review Process
**Monthly SLO Review:**
1. Analyze SLO compliance (% time in-SLO)
2. Review error budget consumption
3. Root cause analysis for violations
4. Prioritize improvement actions

**Quarterly SLO Calibration:**
1. Validate SLO targets against business needs
2. Adjust targets based on system improvements
3. Add/remove SLOs based on customer feedback
4. Update measurement methodology if needed

---

## Appendix: SLO Calculation Examples

### Example 1: Detection Accuracy
Over 24-hour period:
- True Positives: 95 failures detected and confirmed
- False Positives: 3 healthy traps flagged as failed
- False Negatives: 2 failed traps not detected

```
Accuracy = 95 / (95 + 3 + 2) = 95 / 100 = 95.0%
Result: IN SLO (target: >95%)
```

### Example 2: Inspection Latency
1000 inspections in 1-hour window:
- P50 latency: 1.8s
- P95 latency: 2.7s
- P99 latency: 4.2s

```
P95 Latency = 2.7s
Result: IN SLO (target: <3s)
```

### Example 3: Availability
30-day period:
- Total time: 30 × 24 × 60 = 43,200 minutes
- Downtime: 2 incidents × 15 min + 1 incident × 8 min = 38 minutes
- Uptime: 43,200 - 38 = 43,162 minutes

```
Availability = 43,162 / 43,200 = 99.91%
Result: IN SLO (target: >99.9%)
```

### Example 4: False Positive Rate
24-hour period:
- True Positives: 48
- False Positives: 3

```
False Positive Rate = 3 / (48 + 3) = 3 / 51 = 5.88%
Result: OUT OF SLO (target: <5%)
Action: GL008SLOFalsePositiveRate alert fires
```

---

## Document Control

**Approval:**
- Platform Lead: _____________________ Date: _______
- ML Ops Lead: _____________________ Date: _______
- Operations Lead: _____________________ Date: _______

**Change Log:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-26 | GL-BackendDeveloper | Initial SLO definitions |

**Next Review:** 2026-02-26
