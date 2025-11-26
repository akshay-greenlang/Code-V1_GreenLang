# GL-009 THERMALIQ Service Level Objectives (SLOs)

## Overview

This document defines the Service Level Objectives (SLOs) for GL-009 THERMALIQ ThermalEfficiencyCalculator agent. SLOs are measurable targets for service reliability, performance, and data quality that drive operational excellence and user satisfaction.

**Document Version:** 1.0.0
**Last Updated:** 2025-11-26
**Owner:** GreenLang Platform Engineering
**Reviewers:** Energy Analytics Team, Operations Team

---

## SLO Summary Table

| SLO Name | Target | Measurement Window | Error Budget | Status Dashboard |
|----------|--------|-------------------|--------------|------------------|
| Availability | 99.9% | 30 days | 43.2 min/month | [Link](#slo-1-availability) |
| Calculation Latency (P99) | <500ms | 7 days | 1% violations | [Link](#slo-2-calculation-latency-p99) |
| Sankey Generation Time (P99) | <2s | 7 days | 1% violations | [Link](#slo-3-sankey-diagram-generation-p99) |
| Cache Hit Rate | >85% | 24 hours | 15% misses | [Link](#slo-4-cache-hit-rate) |
| Data Quality Score | >99% | 24 hours | 1% degraded | [Link](#slo-5-data-quality-score) |
| Error Rate | <0.1% | 30 days | 10x budget | [Link](#slo-6-error-rate) |
| Integration Connector Health | >95% | 24 hours | 5% downtime | [Link](#slo-7-integration-connector-health) |
| Heat Balance Accuracy | <2% error | Per calculation | 2% threshold | [Link](#slo-8-heat-balance-accuracy) |

---

## SLO Definitions

### SLO-1: Availability

**Objective:** GL-009 agent shall be available for thermal efficiency calculations 99.9% of the time.

**Rationale:** Industrial energy management requires near-continuous monitoring. Downtime delays efficiency insights and optimization actions.

**Measurement:**
```promql
# Availability = Successful Calculations / Total Calculation Attempts
(
  sum(rate(gl009_calculations_total{status="success"}[30d])) /
  sum(rate(gl009_calculations_total[30d]))
) >= 0.999
```

**Target:** 99.9% availability over 30-day rolling window

**Error Budget:** 43.2 minutes of downtime per month
- Daily budget: ~1.44 minutes
- Weekly budget: ~10 minutes
- Monthly budget: 43.2 minutes

**Exclusions:**
- Scheduled maintenance windows (announced 48h in advance)
- Upstream dependency failures (ERP, historian, SCADA) if connector implements retry/fallback
- Force majeure events (datacenter outages, network failures beyond GreenLang control)

**Measurement Frequency:** Real-time with 5-minute aggregation

**Alert Thresholds:**
- **Critical:** <99.0% over 1 hour (60% of monthly budget burned)
- **Warning:** <99.5% over 6 hours (trending toward SLO miss)

**Stakeholders:**
- **Responsible:** Platform Engineering
- **Accountable:** Engineering Manager
- **Consulted:** Energy Analytics Team
- **Informed:** Operations Team, Customers

**Recovery Actions:**
- Automatic pod restart on health check failure
- Failover to standby instance in multi-region deployment
- Graceful degradation: serve cached results if fresh calculations unavailable

---

### SLO-2: Calculation Latency (P99)

**Objective:** 99th percentile of thermal efficiency calculation requests shall complete in under 500ms.

**Rationale:** Real-time dashboards and control system integration require sub-second response times. P99 ensures consistent performance even under load.

**Measurement:**
```promql
# P99 latency across all calculation types
histogram_quantile(0.99,
  rate(gl009_calculation_duration_seconds_bucket[7d])
) < 0.5
```

**Target:** P99 < 500ms over 7-day rolling window

**Breakdown by Calculation Type:**
- First Law efficiency: P99 < 300ms
- Second Law efficiency: P99 < 500ms (more complex)
- Loss breakdown: P99 < 200ms
- Multi-equipment aggregation: P99 < 800ms

**Error Budget:** 1% of requests may exceed 500ms
- Per 100,000 calculations: max 1,000 slow requests
- Per day (~10,000 calculations): max 100 slow requests

**Contributing Factors:**
- Formula evaluation time: <100ms
- Database lookups (benchmarks, factors): <50ms
- Connector data retrieval: <200ms
- Calculation logic: <100ms
- Response serialization: <50ms

**Measurement Frequency:** Real-time histogram buckets

**Alert Thresholds:**
- **Critical:** P99 > 1s for >5 minutes
- **Warning:** P99 > 500ms for >10 minutes

**Optimization Strategies:**
- Cache emission factors, benchmarks (66% cost reduction observed in GL-008)
- Pre-fetch reference data during startup
- Use async I/O for connector calls
- Optimize thermodynamic formula evaluation
- Horizontal scaling for parallelization

**Stakeholders:**
- **Responsible:** Backend Engineering
- **Accountable:** Tech Lead
- **Consulted:** Performance Engineering
- **Informed:** API consumers, Dashboard team

---

### SLO-3: Sankey Diagram Generation (P99)

**Objective:** 99th percentile of Sankey energy flow diagram generation requests shall complete in under 2 seconds.

**Rationale:** Sankey diagrams are visualization-heavy and computationally intensive. 2-second target balances UX with complexity.

**Measurement:**
```promql
# P99 Sankey generation time
histogram_quantile(0.99,
  rate(gl009_sankey_generation_duration_seconds_bucket[7d])
) < 2.0
```

**Target:** P99 < 2s over 7-day rolling window

**Complexity Tiers:**
- Simple (1-5 flows): P99 < 1s
- Medium (6-15 flows): P99 < 2s
- Complex (16+ flows): P99 < 5s

**Error Budget:** 1% of requests may exceed 2s

**Measurement Frequency:** Per generation request

**Alert Thresholds:**
- **Critical:** P99 > 5s for >10 minutes
- **Warning:** P99 > 2s for >15 minutes

**Optimization Strategies:**
- Pre-calculate flow data during efficiency calculation
- Cache Sankey diagram for stable operating conditions
- Use SVG streaming for large diagrams
- Implement progressive rendering for complex diagrams

**Stakeholders:**
- **Responsible:** Visualization Engineering
- **Accountable:** Frontend Tech Lead
- **Informed:** UX Team, Customers

---

### SLO-4: Cache Hit Rate

**Objective:** Cache hit rate for thermal calculations shall exceed 85%.

**Rationale:** Many industrial processes operate in steady-state. High cache hit rate reduces computation, latency, and cost.

**Measurement:**
```promql
# Cache hit rate
gl009_cache_hit_rate >= 0.85
```

**Target:** >85% hit rate over 24-hour rolling window

**Cache Categories:**
- Emission factors: >95% (rarely change)
- Benchmark data: >90% (updated quarterly)
- Equipment configuration: >80% (change during maintenance)
- Recent calculations: >70% (repetitive queries)

**Error Budget:** Up to 15% cache misses acceptable

**Measurement Frequency:** Real-time gauge updated every cache lookup

**Alert Thresholds:**
- **Critical:** <70% for >30 minutes (infrastructure issue likely)
- **Warning:** <85% for >1 hour (trending toward SLO miss)

**Cache Invalidation Strategy:**
- TTL-based: emission factors (24h), benchmarks (7d)
- Event-based: equipment configuration changes
- LRU eviction: when cache size exceeds limit

**Optimization Strategies:**
- Increase cache size (currently 1GB target)
- Implement multi-tier caching (L1: in-memory, L2: Redis)
- Pre-warm cache with frequently accessed data
- Analyze miss patterns to optimize caching strategy

**Stakeholders:**
- **Responsible:** Backend Engineering
- **Accountable:** Tech Lead
- **Consulted:** Performance Engineering

---

### SLO-5: Data Quality Score

**Objective:** Input data quality score shall exceed 99% for all thermal efficiency calculations.

**Rationale:** Garbage in, garbage out. High-quality input data is essential for accurate efficiency calculations and regulatory compliance.

**Measurement:**
```promql
# Average data quality score across all data sources
avg(gl009_data_quality_score) >= 0.99
```

**Target:** >99% average quality score over 24-hour rolling window

**Data Quality Dimensions:**
1. **Completeness:** All required fields present (weight: 30%)
2. **Accuracy:** Values within physical bounds (weight: 25%)
3. **Consistency:** Heat balance closure <2% error (weight: 20%)
4. **Timeliness:** Data timestamp <5 minutes old (weight: 15%)
5. **Validity:** Passes thermodynamic sanity checks (weight: 10%)

**Quality Scoring Formula:**
```
Quality Score =
  0.30 * Completeness +
  0.25 * Accuracy +
  0.20 * Consistency +
  0.15 * Timeliness +
  0.10 * Validity
```

**Error Budget:** Up to 1% of data may be degraded quality

**Measurement Frequency:** Per calculation input validation

**Alert Thresholds:**
- **Critical:** <95% average score for >15 minutes
- **Warning:** <99% average score for >1 hour

**Data Quality Checks:**
- **Completeness:** Check for null/missing required fields
- **Accuracy:** Temperature in range [0-2000°C], pressure >0, flow rate >0
- **Consistency:** Energy input = useful output + losses ±2%
- **Timeliness:** Reject data older than 5 minutes (stale)
- **Validity:** Efficiency cannot exceed 100%, Second Law eff < First Law eff

**Remediation Actions:**
- Flag low-quality calculations for manual review
- Reject calculations with quality score <90%
- Alert data source owner when quality degrades
- Implement automatic sensor recalibration triggers

**Stakeholders:**
- **Responsible:** Data Quality Engineering
- **Accountable:** Data Platform Manager
- **Consulted:** Operations Team, Sensor Vendors
- **Informed:** Energy Analytics Team

---

### SLO-6: Error Rate

**Objective:** Calculation error rate shall be less than 0.1% of all requests.

**Rationale:** High reliability is critical for regulatory compliance and business trust. Errors undermine confidence in efficiency recommendations.

**Measurement:**
```promql
# Error rate over 30 days
(
  sum(rate(gl009_calculation_errors_total[30d])) /
  sum(rate(gl009_calculations_total[30d]))
) < 0.001
```

**Target:** <0.1% error rate over 30-day rolling window

**Error Budget:** Up to 1 error per 1,000 calculations
- Per 100,000 monthly calculations: max 100 errors
- Per day (~3,300 calculations): max 3 errors

**Error Categories (tracked separately):**
1. **Input validation errors:** <0.05% (user fixable)
2. **Calculation errors:** <0.02% (formula/logic bugs)
3. **Integration errors:** <0.02% (connector failures)
4. **System errors:** <0.01% (infrastructure failures)

**Measurement Frequency:** Real-time counter with 5-minute rate calculation

**Alert Thresholds:**
- **Critical:** >0.5% error rate for >5 minutes (5x budget)
- **Warning:** >0.1% error rate for >15 minutes (trending toward miss)

**Error Handling Strategy:**
- Graceful degradation: return partial results if possible
- Automatic retry with exponential backoff for transient errors
- Circuit breaker pattern for failing connectors
- Fallback to cached/historical data when appropriate

**Error Budget Policy:**
- If error budget exhausted: freeze feature releases until reliability restored
- Focus engineering effort on top error sources
- Post-incident review for any critical error spike

**Stakeholders:**
- **Responsible:** Backend Engineering, SRE
- **Accountable:** Engineering Manager
- **Consulted:** QA Engineering
- **Informed:** Customers, Support Team

---

### SLO-7: Integration Connector Health

**Objective:** Integration connectors (energy meters, historians, SCADA, ERP) shall maintain 95%+ health over 24 hours.

**Rationale:** GL-009 depends on external data sources. Connector health directly impacts calculation availability and data freshness.

**Measurement:**
```promql
# Connector health percentage
avg(gl009_connector_health) >= 0.95
```

**Target:** >95% health over 24-hour rolling window

**Connector Types and Targets:**
- **Energy meters (Modbus/OPC-UA):** >98% (critical path)
- **Process historians:** >95% (required for calculations)
- **SCADA systems:** >90% (real-time monitoring)
- **ERP systems:** >85% (cost data, less critical)
- **Weather APIs:** >90% (ambient conditions)

**Health Definition:**
- Connector returns successful response within timeout
- Data passes validation checks
- No authentication/authorization errors
- Latency within acceptable range (<1s P99)

**Error Budget:** Up to 5% connector downtime acceptable
- Per 24 hours: max 1.2 hours downtime per connector
- Per connector-day: max 72 minutes unavailable

**Measurement Frequency:** Health check every 30 seconds per connector

**Alert Thresholds:**
- **Critical:** <80% health for critical connectors (energy meters, historians)
- **Warning:** <95% health for >1 hour

**Resilience Strategies:**
- Implement retry logic with exponential backoff
- Circuit breaker pattern: stop calling failing endpoint
- Fallback to cached/historical data
- Multi-source redundancy where possible
- Graceful degradation: mark calculations as "estimated" if using fallback data

**Connector Monitoring:**
- Health check endpoint response time
- Success/failure rate
- Data freshness (timestamp check)
- Authentication token expiry monitoring

**Stakeholders:**
- **Responsible:** Integration Engineering
- **Accountable:** Integration Tech Lead
- **Consulted:** Vendor support teams, Network Engineering
- **Informed:** Operations Team

---

### SLO-8: Heat Balance Accuracy

**Objective:** Heat balance closure error shall be less than 2% for all thermal efficiency calculations.

**Rationale:** Energy conservation (First Law of Thermodynamics) requires inputs = outputs + losses. Errors >2% indicate measurement issues or missing loss terms.

**Measurement:**
```promql
# Heat balance error percentage
histogram_quantile(0.95,
  rate(gl009_heat_balance_error_percent_bucket[24h])
) < 2.0
```

**Target:** P95 heat balance error <2% over 24-hour window

**Error Formula:**
```
Heat Balance Error (%) =
  |Energy Input - (Useful Output + Total Losses)| / Energy Input * 100
```

**Acceptable Error Ranges:**
- **Excellent:** <1% (high-quality instrumentation)
- **Good:** 1-2% (acceptable for most applications)
- **Marginal:** 2-5% (flag for review, still usable)
- **Unacceptable:** >5% (reject calculation, investigate)

**Error Budget:** 5% of calculations may have 2-5% error

**Measurement Frequency:** Every calculation

**Alert Thresholds:**
- **Critical:** >5% error for any equipment for >15 minutes
- **Warning:** >2% error for >30 minutes

**Common Causes of Heat Balance Error:**
1. **Missing loss terms:** Unaccounted radiation, convection
2. **Sensor calibration drift:** Temperature, flow, pressure sensors
3. **Timing mismatch:** Input and output measured at different times
4. **Phase change not accounted:** Condensate, steam quality
5. **Incomplete combustion:** Unburned fuel losses

**Error Investigation Process:**
1. Check sensor calibration dates (>90 days → recalibrate)
2. Verify all loss terms included in calculation
3. Review data timestamp alignment
4. Inspect physical equipment for insulation damage
5. Run parallel calculation with alternate sensors

**Remediation Actions:**
- Flag calculations with >2% error as "low confidence"
- Trigger sensor calibration workflow
- Alert maintenance team for physical inspection
- Increase measurement frequency to reduce timing errors
- Update loss models if systematic error detected

**Stakeholders:**
- **Responsible:** Energy Engineering, Instrumentation Team
- **Accountable:** Plant Engineer
- **Consulted:** Thermodynamics Subject Matter Experts
- **Informed:** Operations Team, Regulatory Affairs

---

## Error Budget Policy

### Budget Tracking
- **Real-time dashboard:** Grafana panel showing error budget consumption
- **Weekly review:** Platform Engineering reviews budget burn rate
- **Monthly report:** SLO compliance report to stakeholders

### Budget Exhaustion Response
When error budget is exhausted (SLO missed):

1. **Immediate Actions (0-24 hours):**
   - Incident declared
   - Root cause analysis initiated
   - Stop all feature development
   - All hands on reliability improvements

2. **Short-term Actions (1-7 days):**
   - Implement quick fixes for top issues
   - Increase monitoring and alerting
   - Daily standup on reliability
   - Document lessons learned

3. **Long-term Actions (1-4 weeks):**
   - Address systemic reliability issues
   - Improve testing coverage
   - Update runbooks
   - Resume feature development only after SLO restored for 7 consecutive days

### Budget Allocation Philosophy
- **70% for toil:** Routine operational work, maintenance
- **20% for feature development:** New functionality
- **10% for experimentation:** Innovation, optimization

---

## SLO Review and Adjustment

### Review Frequency
- **Weekly:** Tactical review of error budget consumption
- **Monthly:** Strategic review of SLO achievement, trends
- **Quarterly:** Comprehensive review including customer feedback
- **Annually:** Major SLO revision based on business needs

### Adjustment Criteria
SLOs may be adjusted if:
- Consistently meeting target with >90% error budget remaining (tighten)
- Consistently missing target despite best efforts (loosen or invest in reliability)
- Business requirements change (e.g., new regulatory requirements)
- Technology changes enable better performance

### Approval Process
- **Minor adjustments (<5%):** Tech Lead approval
- **Major adjustments (>5%):** Engineering Manager + Product Manager approval
- **Target removal/addition:** VP Engineering approval

---

## SLO Reporting

### Dashboards
- **Real-time:** Grafana dashboards for live SLO tracking
- **Daily:** Automated Slack summary to #gl009-slo channel
- **Weekly:** Email report to engineering team
- **Monthly:** Executive dashboard with trends, burn rate, incidents

### Metrics Included in Reports
- Current SLO achievement percentage
- Error budget consumed vs. remaining
- Trend over last 7/30/90 days
- Top contributors to SLO misses
- Incidents and their impact on SLO
- Planned vs. unplanned downtime

### Report Recipients
- Engineering Team (daily)
- Engineering Manager (weekly)
- Product Management (monthly)
- Executive Leadership (quarterly)
- Customers (on-demand via customer portal)

---

## Appendix: PromQL Queries

### Availability
```promql
# 30-day availability
(
  sum(rate(gl009_calculations_total{status="success"}[30d])) /
  sum(rate(gl009_calculations_total[30d]))
)

# Error budget remaining (minutes)
43.2 - (43.2 * (1 - gl009_availability_30d))
```

### Latency
```promql
# P50, P90, P95, P99 latency
histogram_quantile(0.50, rate(gl009_calculation_duration_seconds_bucket[7d]))
histogram_quantile(0.90, rate(gl009_calculation_duration_seconds_bucket[7d]))
histogram_quantile(0.95, rate(gl009_calculation_duration_seconds_bucket[7d]))
histogram_quantile(0.99, rate(gl009_calculation_duration_seconds_bucket[7d]))

# Latency error budget consumption
(
  count(gl009_calculation_duration_seconds > 0.5) /
  count(gl009_calculation_duration_seconds)
) / 0.01 * 100
```

### Error Rate
```promql
# 30-day error rate
(
  sum(rate(gl009_calculation_errors_total[30d])) /
  sum(rate(gl009_calculations_total[30d]))
)

# Error budget remaining
0.001 - (
  sum(rate(gl009_calculation_errors_total[30d])) /
  sum(rate(gl009_calculations_total[30d]))
)
```

### Data Quality
```promql
# Average quality score
avg(gl009_data_quality_score)

# Quality score by dimension
avg(gl009_data_quality_score{dimension="completeness"})
avg(gl009_data_quality_score{dimension="accuracy"})
avg(gl009_data_quality_score{dimension="consistency"})
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-26 | Platform Engineering | Initial SLO definitions for GL-009 |

---

**Next Review Date:** 2026-02-26 (Quarterly)
