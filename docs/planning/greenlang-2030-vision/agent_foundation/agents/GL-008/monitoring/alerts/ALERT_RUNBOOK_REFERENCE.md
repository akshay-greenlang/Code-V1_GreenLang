# GL-008 SteamTrapInspector Alert Runbook Reference

**Version:** 1.0
**Last Updated:** 2025-11-26
**On-Call Team:** GL-008 Platform & Maintenance Operations
**Escalation:** See "Escalation Paths" section below

---

## Overview

This runbook provides step-by-step response procedures for all GL-008 SteamTrapInspector alerts. Each alert includes:
- **Severity classification** (Critical, High, Medium, Low)
- **Immediate actions** to take within first 5 minutes
- **Investigation steps** to diagnose root cause
- **Resolution procedures** to fix the issue
- **Escalation criteria** for when to involve senior engineers

**Alert Response Time Targets:**
- **Critical:** 5 minutes to acknowledge, 30 minutes to triage
- **High:** 15 minutes to acknowledge, 2 hours to triage
- **Medium:** 1 hour to acknowledge, 8 hours to triage
- **Low:** 24 hours to acknowledge, 1 week to resolve

---

## Table of Contents

1. [Critical Alerts](#critical-alerts)
2. [High Severity Alerts](#high-severity-alerts)
3. [Medium Severity Alerts](#medium-severity-alerts)
4. [Low Severity Alerts](#low-severity-alerts)
5. [SLO Violation Alerts](#slo-violation-alerts)
6. [Escalation Paths](#escalation-paths)
7. [Common Investigation Commands](#common-investigation-commands)

---

## Critical Alerts

### GL008CriticalTrapFailure

**Alert:** Critical steam trap failure detected
**Severity:** Critical
**SLO Impact:** Potential safety risk, high energy loss

#### Symptoms
- Alert fires when `gl008_trap_status{status="critical"}` > 0
- Indicates trap in critical failure state requiring immediate attention
- Typical failure modes: Blow-through, severe leaking, mechanical damage

#### Immediate Actions (0-5 minutes)
1. **Acknowledge alert** in PagerDuty/Slack
2. **Check alert details** for:
   - Trap ID and location
   - Facility name
   - Failure mode
   - Energy loss estimate
3. **Notify facility maintenance team** immediately via phone/radio
4. **Check if safety risk** (steam leak in occupied area, scalding hazard)
   - If YES: Initiate facility safety protocol, evacuate if needed
   - If NO: Proceed with maintenance response

#### Investigation (5-30 minutes)
1. **Pull trap inspection history:**
   ```bash
   curl http://gl008-api/v1/traps/{trap_id}/history?limit=10
   ```
2. **Review sensor data:**
   - Acoustic signature (frequency spectrum, amplitude)
   - Thermal image (temperature differential)
   - Ultrasonic readings (decibel levels)
3. **Check for related failures:**
   ```promql
   gl008_failures_detected_total{facility="$facility", failure_mode="$mode"}[24h]
   ```
4. **Estimate impact:**
   - Energy loss: `gl008_energy_loss_kw{trap_id="$trap_id"}`
   - Cost impact: `gl008_cost_impact_usd_per_year{trap_id="$trap_id"}`
   - CO2 emissions: `gl008_co2_emissions_kg_per_hour{trap_id="$trap_id"}`

#### Resolution
1. **Dispatch technician** to trap location with replacement parts
2. **Isolate trap** if possible (close isolation valves)
3. **Perform hands-on inspection** to confirm failure mode
4. **Repair or replace** trap according to maintenance procedures
5. **Verify repair** by running GL-008 inspection post-repair
6. **Update trap status** in system:
   ```bash
   curl -X PATCH http://gl008-api/v1/traps/{trap_id} \
     -d '{"status": "healthy", "last_maintenance": "2025-11-26"}'
   ```
7. **Document resolution** in maintenance log and alert system

#### Escalation Criteria
- **Immediate escalation** if:
  - Safety risk to personnel
  - Multiple critical failures (>3) in same facility
  - Cannot isolate trap and energy loss >100 kW
  - Technician unable to access trap location
- **Escalate to:** Facility Manager → Plant Engineering → Safety Officer

#### Prevention
- Review trap maintenance schedule (preventive maintenance missed?)
- Check for systemic issues (poor steam quality, water hammer, etc.)
- Verify sensor calibration (false critical rating?)

---

### GL008MultipleCriticalFailures

**Alert:** Multiple critical trap failures detected
**Severity:** Critical
**SLO Impact:** Fleet-wide issue, major energy loss, potential safety

#### Symptoms
- Alert fires when 3+ critical failures in same facility
- May indicate systemic problem (steam quality, pressure surge, etc.)

#### Immediate Actions (0-5 minutes)
1. **Acknowledge alert** and notify facility manager
2. **Check facility status:**
   ```promql
   gl008_fleet_critical_count{facility="$facility"}
   ```
3. **Review failure timeline:** Were failures simultaneous or sequential?
4. **Check steam system status:**
   - Boiler pressure/temperature logs
   - Recent steam system changes
   - Water treatment issues

#### Investigation (5-30 minutes)
1. **Map failure locations** (clustered or distributed?)
2. **Analyze failure modes:**
   ```promql
   sum by (failure_mode) (gl008_failures_detected_total{facility="$facility", severity="critical"}[1h])
   ```
3. **Check for common root cause:**
   - Same trap type/model?
   - Same steam header?
   - Recent maintenance activity?
4. **Review steam system events:**
   - Pressure spikes (check pressure logs)
   - Water hammer events (check condensate return)
   - Steam quality issues (check boiler blowdown logs)

#### Resolution
1. **If systemic issue suspected:**
   - Isolate affected steam header if possible
   - Investigate steam system (boiler operator, water treatment)
   - Correct root cause before individual trap repairs
2. **If coincidental failures:**
   - Dispatch multiple technicians for parallel repairs
   - Prioritize by energy loss/cost impact
3. **Triage repairs** using:
   ```bash
   curl http://gl008-api/v1/facilities/{facility}/failures?sort=energy_loss_desc
   ```
4. **Monitor for additional failures** while repairs in progress

#### Escalation Criteria
- **Immediate escalation** if:
  - >5 critical failures
  - Steam system instability suspected
  - Total energy loss >500 kW
  - Safety shutdown of production area required
- **Escalate to:** Plant Engineering → Corporate Engineering → VP Operations

#### Prevention
- Implement steam quality monitoring
- Review preventive maintenance program
- Consider fleet-wide trap audits
- Analyze for equipment aging issues

---

### GL008HighEnergyLoss

**Alert:** High energy loss from failed trap
**Severity:** Critical
**SLO Impact:** Major cost impact, sustainability goals

#### Symptoms
- Alert fires when `gl008_energy_loss_kw` > 100 kW
- Typically blow-through failures on large traps
- Annual cost impact often exceeds $50,000

#### Immediate Actions (0-5 minutes)
1. **Acknowledge alert**
2. **Verify energy loss calculation:**
   ```bash
   curl http://gl008-api/v1/traps/{trap_id}/energy-analysis
   ```
3. **Check trap size and steam pressure** (high energy loss expected on large traps)
4. **Prioritize for immediate repair** (payback period often <1 month)

#### Investigation (5-30 minutes)
1. **Confirm failure mode:**
   - Blow-through (trap passing live steam)
   - Plugged open (trap stuck open)
   - Severe leaking (gasket failure, body crack)
2. **Calculate financial impact:**
   - Daily cost: Energy loss (kW) × 24 hr × $0.08/kWh = $___/day
   - Annual cost: Daily cost × 365 = $___/year
3. **Check if trap can be isolated** (minimize energy loss while awaiting repair)
4. **Review historical performance:**
   ```promql
   gl008_energy_loss_kw{trap_id="$trap_id"}[30d]
   ```

#### Resolution
1. **Expedite repair** (highest priority work order)
2. **If trap can be isolated:**
   - Close isolation valves
   - Monitor downstream equipment (ensure condensate can drain)
   - Arrange bypass if needed
3. **If trap cannot be isolated:**
   - Repair within same shift (emergency overtime if needed)
   - Calculate hourly cost of delay (justify overtime)
4. **Post-repair verification:**
   - Run GL-008 inspection
   - Verify energy loss reduced to near-zero
   - Calculate actual savings

#### Escalation Criteria
- **Escalate immediately** if:
  - Energy loss >500 kW
  - Trap cannot be isolated and repair delayed >8 hours
  - Failure impacts critical production equipment
- **Escalate to:** Maintenance Manager → Plant Manager → Energy Manager

#### Prevention
- High-value traps should be on accelerated inspection schedule
- Consider redundant traps for critical applications
- Implement automated isolation (smart traps with remote shutoff)

---

### GL008BlowThroughFailure

**Alert:** Blow-through failure - SAFETY RISK
**Severity:** Critical
**SLO Impact:** Personnel safety, energy loss

#### Symptoms
- Alert fires on blow-through failure mode with critical severity
- Live steam escaping through failed trap
- Potential for steam burns, scalding hazard

#### Immediate Actions (0-5 minutes)
1. **Acknowledge alert**
2. **Assess safety risk:**
   - Is trap in occupied area?
   - Is steam visible or audible?
   - Are personnel at risk?
3. **If safety risk present:**
   - **EVACUATE AREA IMMEDIATELY**
   - Post warning signs
   - Notify safety officer
   - Consider facility-wide steam shutdown if severe
4. **If no immediate risk:**
   - Proceed with maintenance response
   - Continue monitoring

#### Investigation (5-15 minutes)
1. **Technician visual inspection** (from safe distance)
2. **Review acoustic signature** (blow-through has distinct high-frequency signature)
3. **Check steam pressure** (higher pressure = higher risk)
4. **Determine if trap can be safely isolated**

#### Resolution
1. **Safety-first approach:**
   - Do NOT attempt repair if personnel risk
   - Isolate steam header if needed
   - Allow trap to cool before approaching
2. **Once safe:**
   - Close isolation valves
   - Relieve pressure
   - Replace trap (blow-through = catastrophic failure, not repairable)
3. **Root cause analysis:**
   - Water hammer damage?
   - Trap oversized for application?
   - Steam quality issue?
4. **Post-repair:**
   - Verify isolation valves functional
   - Test new trap operation
   - Document incident in safety log

#### Escalation Criteria
- **Immediate escalation** if:
  - Personnel injury or near-miss
  - Cannot isolate trap safely
  - Requires facility shutdown
- **Escalate to:** Safety Officer → Plant Manager → Corporate Safety

#### Prevention
- Install isolation valves on all critical traps
- Implement water hammer mitigation
- Regular safety training on steam hazards
- Consider automated shutdown on blow-through detection

---

### GL008SystemDown

**Alert:** GL-008 inspection system is DOWN
**Severity:** Critical
**SLO Impact:** Availability SLO violation, no inspections possible

#### Symptoms
- Alert fires when `gl008_system_health{component="inspection_engine"}` == 0
- All inspections failing
- API returning 503 errors

#### Immediate Actions (0-5 minutes)
1. **Acknowledge alert** and notify platform team
2. **Check system health dashboard:**
   ```bash
   curl http://gl008-api/health
   ```
3. **Review recent deployments/changes** (was system updated recently?)
4. **Check infrastructure status:**
   - Kubernetes pods: `kubectl get pods -n gl008`
   - Database: `pg_isready -h gl008-db`
   - Redis cache: `redis-cli ping`

#### Investigation (5-30 minutes)
1. **Check application logs:**
   ```bash
   kubectl logs -n gl008 -l app=gl008-inspection-engine --tail=100
   ```
2. **Common failure modes:**
   - Database connection pool exhausted
   - Out of memory (check pod memory usage)
   - Unhandled exception in critical path
   - External dependency failure (sensor API, ML model service)
3. **Check metrics:**
   ```promql
   gl008_system_health{component="inspection_engine"}[30m]
   ```
   - When did it go down?
   - Was it gradual degradation or sudden?

#### Resolution
1. **Quick fixes to try first:**
   - Restart inspection engine pods: `kubectl rollout restart deployment/gl008-inspection-engine`
   - Clear cache: `redis-cli FLUSHDB`
   - Restart database connection pool (may require app restart)
2. **If restart doesn't work:**
   - Rollback to previous version: `kubectl rollout undo deployment/gl008-inspection-engine`
   - Check for resource constraints: `kubectl top pods -n gl008`
   - Scale up if needed: `kubectl scale deployment/gl008-inspection-engine --replicas=5`
3. **Monitor recovery:**
   ```promql
   gl008_system_health{component="inspection_engine"}
   ```
4. **Once recovered:**
   - Post-mortem incident review
   - Identify root cause
   - Implement prevention measures

#### Escalation Criteria
- **Escalate after 15 minutes** if:
  - Cannot identify root cause
  - Restart/rollback doesn't restore service
  - Data corruption suspected
- **Escalate to:** Senior Platform Engineer → VP Engineering

#### Prevention
- Implement health check improvements (catch failures before total outage)
- Add circuit breakers for external dependencies
- Increase resource limits/requests
- Implement automated recovery procedures

---

## High Severity Alerts

### GL008HighFailureRate

**Alert:** High trap failure rate detected
**Severity:** High
**SLO Impact:** Fleet health degradation

#### Symptoms
- >10% of fleet in failed state
- Indicates maintenance backlog or systemic issues

#### Immediate Actions (0-15 minutes)
1. **Acknowledge alert**
2. **Check current failure rate:**
   ```promql
   sum(gl008_fleet_failed_count{facility="$facility"}) / sum(gl008_fleet_total_traps{facility="$facility"})
   ```
3. **Review maintenance backlog:**
   ```bash
   curl http://gl008-api/v1/facilities/{facility}/maintenance-backlog
   ```

#### Investigation
1. **Analyze failure trends:**
   - Is failure rate increasing or stable?
   - Which trap types are failing most?
   - Are failures clustered by location?
2. **Check maintenance capacity:**
   - Technician availability
   - Parts inventory
   - Budget constraints
3. **Identify root causes:**
   - Deferred maintenance?
   - Equipment aging?
   - Steam quality issues?

#### Resolution
1. **Triage failures by cost impact:**
   ```bash
   curl http://gl008-api/v1/facilities/{facility}/failures?sort=cost_impact_desc
   ```
2. **Allocate maintenance resources:**
   - Prioritize high-impact failures
   - Schedule overtime if needed
   - Order critical spare parts
3. **Set repair timeline:**
   - Critical: <24 hours
   - High impact: <1 week
   - Medium impact: <1 month
4. **Monitor progress daily**

#### Escalation Criteria
- Escalate if failure rate >20% or maintenance backlog growing
- **Escalate to:** Maintenance Manager → Plant Manager

---

### GL008LowDetectionAccuracy

**Alert:** Detection accuracy below SLO
**Severity:** High
**SLO Impact:** SLO-001 violation

#### Symptoms
- Detection accuracy <95% over 1-hour window
- May indicate model drift, sensor issues, or data quality problems

#### Immediate Actions (0-15 minutes)
1. **Acknowledge alert**
2. **Check current accuracy metrics:**
   ```promql
   avg(gl008_detection_accuracy_rate{time_window="1h"})
   ```
3. **Break down by component:**
   - Precision: `avg(gl008_detection_precision)`
   - Recall: `avg(gl008_detection_recall)`
   - F1 score: `avg(gl008_detection_f1_score)`

#### Investigation
1. **Determine if precision or recall issue:**
   - Low precision → too many false positives
   - Low recall → missing failures (false negatives)
2. **Check recent detections:**
   ```bash
   curl http://gl008-api/v1/detections/recent?status=false_positive&limit=20
   ```
3. **Analyze patterns:**
   - Specific trap types with low accuracy?
   - Specific facilities?
   - Time-of-day correlation?
4. **Review sensor data quality:**
   - Calibration dates
   - Signal-to-noise ratio
   - Data completeness

#### Resolution
1. **Short-term fixes:**
   - Adjust detection thresholds (if precision issue)
   - Improve sensor calibration
   - Filter known false positive conditions
2. **Long-term fixes:**
   - Retrain ML models with recent labeled data
   - Collect more training data for low-accuracy segments
   - Improve feature engineering
3. **Validation:**
   - A/B test new model against current
   - Monitor accuracy metrics post-deployment

#### Escalation Criteria
- Escalate if accuracy <90% or unable to identify root cause within 4 hours
- **Escalate to:** ML Ops Lead → Data Science Team

---

### GL008HighFalsePositiveRate

**Alert:** High false positive rate
**Severity:** High
**SLO Impact:** SLO-005 violation, wasted maintenance effort

#### Symptoms
- False positive rate >5%
- Technicians reporting frequent "healthy trap" findings

#### Immediate Actions (0-15 minutes)
1. **Acknowledge alert**
2. **Check false positive rate:**
   ```promql
   sum(increase(gl008_false_positives_total[1h])) /
   (sum(increase(gl008_true_positives_total[1h])) + sum(increase(gl008_false_positives_total[1h])))
   ```
3. **Review recent false positives:**
   ```bash
   curl http://gl008-api/v1/detections/recent?status=false_positive&limit=50
   ```

#### Investigation
1. **Identify patterns in false positives:**
   - Specific trap types? (e.g., thermostatic traps)
   - Specific failure modes over-detected? (e.g., "blow_through")
   - Specific facilities or sensors?
2. **Check detection confidence:**
   ```promql
   histogram_quantile(0.5, gl008_detection_confidence{trap_id=~"$false_positive_traps"})
   ```
   - Are false positives low-confidence detections?
3. **Review technician feedback:**
   - Are technicians properly classifying? (training issue?)
   - Are there edge cases not in training data?

#### Resolution
1. **Increase detection threshold** (reduces false positives but may reduce recall):
   ```python
   # In detection config
   detection_threshold = 0.85  # Increase from 0.75
   ```
2. **Implement multi-sensor fusion:**
   - Require 2+ sensors to agree before flagging failure
3. **Add temporal filtering:**
   - Require failure to persist across 2+ inspections
4. **Improve training data:**
   - Collect false positive examples
   - Retrain model with better negative examples
5. **Monitor impact:**
   ```promql
   gl008_false_positives_total[24h]
   ```

#### Escalation Criteria
- Escalate if false positive rate >10% or corrective actions ineffective
- **Escalate to:** ML Ops Lead → Engineering Manager

---

### GL008HighInspectionLatency

**Alert:** Inspection latency exceeds SLO
**Severity:** High
**SLO Impact:** SLO-002 violation, reduced throughput

#### Symptoms
- P95 inspection latency >3 seconds
- Inspections taking longer than expected

#### Immediate Actions (0-15 minutes)
1. **Acknowledge alert**
2. **Check current latency:**
   ```promql
   histogram_quantile(0.95, sum(rate(gl008_inspection_latency_seconds_bucket[5m])) by (le))
   ```
3. **Check system load:**
   ```bash
   kubectl top pods -n gl008
   ```

#### Investigation
1. **Break down latency by component:**
   - Acoustic analysis: `histogram_quantile(0.95, rate(gl008_acoustic_analysis_seconds_bucket[5m]))`
   - Thermal analysis: `histogram_quantile(0.95, rate(gl008_thermal_analysis_seconds_bucket[5m]))`
   - Ultrasonic analysis: `histogram_quantile(0.95, rate(gl008_ultrasonic_analysis_seconds_bucket[5m]))`
2. **Check for bottlenecks:**
   - CPU usage: `kubectl top pods`
   - Database query latency
   - External API calls (sensor interface)
3. **Review recent changes:**
   - New model deployment (slower inference)?
   - Configuration changes?
   - Traffic spike?

#### Resolution
1. **Quick wins:**
   - Scale up replicas: `kubectl scale deployment/gl008-inspection-engine --replicas=10`
   - Clear cache to free memory
   - Restart slow pods
2. **Optimization:**
   - Enable result caching for repeat inspections
   - Optimize database queries (add indexes)
   - Parallelize sensor data processing
3. **Infrastructure:**
   - Increase CPU/memory limits
   - Add read replicas for database
   - Implement edge processing for sensors

#### Escalation Criteria
- Escalate if P95 latency >10s or unable to reduce within 2 hours
- **Escalate to:** Senior Platform Engineer → VP Engineering

---

## Medium Severity Alerts

### GL008DegradedTrapCount

**Alert:** High number of degraded traps
**Severity:** Medium
**SLO Impact:** Fleet health, potential future failures

#### Investigation
1. Review degraded trap list
2. Analyze degradation patterns
3. Estimate time-to-failure

#### Resolution
1. Schedule preventive maintenance
2. Prioritize by degradation rate
3. Monitor for progression to failed state

---

### GL008SensorCalibrationNeeded

**Alert:** Sensor calibration required
**Severity:** Medium
**SLO Impact:** Detection confidence, accuracy

#### Investigation
1. Check sensor calibration dates
2. Review confidence score trends
3. Compare against reference sensors

#### Resolution
1. Schedule calibration for affected sensors
2. Validate calibration procedures
3. Update calibration schedule if needed

---

### GL008LowFleetCoverage

**Alert:** Fleet inspection coverage below target
**Severity:** Medium
**SLO Impact:** SLO-006 violation

#### Investigation
1. Identify uncovered traps
2. Check for sensor connectivity issues
3. Review inspection routes

#### Resolution
1. Prioritize uncovered high-value traps
2. Fix sensor connectivity
3. Adjust inspection scheduling

---

### GL008InspectionErrorRate

**Alert:** Elevated inspection error rate
**Severity:** Medium
**SLO Impact:** Reliability, availability

#### Investigation
1. Review error logs
2. Categorize error types
3. Identify common patterns

#### Resolution
1. Fix software bugs
2. Improve error handling
3. Add retry logic

---

## Low Severity Alerts

### GL008MaintenanceDue

**Alert:** Routine maintenance due
**Severity:** Low
**SLO Impact:** None (preventive)

#### Resolution
1. Schedule routine inspection
2. Add to maintenance calendar
3. No immediate action required

---

### GL008TrendingTowardsFailure

**Alert:** Trap trending towards failure
**Severity:** Low
**SLO Impact:** None (predictive)

#### Resolution
1. Add to watchlist
2. Increase inspection frequency
3. Plan preventive replacement

---

## Escalation Paths

### Platform/System Issues
1. **L1:** On-call Platform Engineer (15 min response)
2. **L2:** Senior Platform Engineer (1 hour response)
3. **L3:** VP Engineering (4 hour response)

### Maintenance/Operations Issues
1. **L1:** Facility Maintenance Technician (30 min response)
2. **L2:** Maintenance Manager (2 hour response)
3. **L3:** Plant Manager (8 hour response)

### Safety Issues
1. **L1:** Facility Safety Officer (IMMEDIATE)
2. **L2:** Corporate Safety Director (30 min response)
3. **L3:** VP Operations (1 hour response)

### ML/Detection Issues
1. **L1:** On-call ML Ops Engineer (1 hour response)
2. **L2:** Data Science Lead (4 hour response)
3. **L3:** VP Engineering (24 hour response)

---

## Common Investigation Commands

### Check System Health
```bash
# Overall health
curl http://gl008-api/health

# Component health
kubectl get pods -n gl008
kubectl top pods -n gl008

# Database health
psql -h gl008-db -U admin -c "SELECT pg_is_in_recovery(), pg_last_xlog_receive_location();"
```

### Query Metrics
```bash
# Detection accuracy
curl -G http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=avg(gl008_detection_accuracy_rate{time_window="1h"})'

# Inspection latency P95
curl -G http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.95, sum(rate(gl008_inspection_latency_seconds_bucket[5m])) by (le))'

# Active alerts
curl -G http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=sum(gl008_active_alerts) by (severity, alert_type)'
```

### Review Logs
```bash
# Application logs (last 100 lines)
kubectl logs -n gl008 -l app=gl008-inspection-engine --tail=100

# Error logs only
kubectl logs -n gl008 -l app=gl008-inspection-engine --tail=500 | grep ERROR

# Follow logs in real-time
kubectl logs -n gl008 -l app=gl008-inspection-engine -f
```

### Database Queries
```sql
-- Recent inspections
SELECT trap_id, status, confidence, created_at
FROM inspections
WHERE created_at > NOW() - INTERVAL '1 hour'
ORDER BY created_at DESC
LIMIT 50;

-- False positives today
SELECT trap_id, failure_mode, technician_notes
FROM detections
WHERE status = 'false_positive'
  AND created_at > CURRENT_DATE
ORDER BY created_at DESC;

-- Top energy losses
SELECT trap_id, energy_loss_kw, cost_impact_usd_yr
FROM trap_failures
WHERE status = 'active'
ORDER BY energy_loss_kw DESC
LIMIT 20;
```

---

## Post-Incident Review Template

After resolving any **Critical** or **High** severity alert:

1. **Incident Summary**
   - Alert name and timestamp
   - Severity and duration
   - Impact (users, SLOs, revenue)

2. **Timeline**
   - Detection time
   - Acknowledgment time
   - Triage time
   - Resolution time

3. **Root Cause**
   - What happened?
   - Why did it happen?
   - Contributing factors?

4. **Resolution**
   - What fixed it?
   - Temporary or permanent fix?

5. **Prevention**
   - Action items to prevent recurrence
   - Monitoring improvements
   - Documentation updates

6. **Follow-up**
   - Owner for each action item
   - Target completion dates
   - Review in next team meeting

---

**Document maintained by GL-008 Platform Team**
**For urgent issues:** Slack #gl008-alerts or page GL-008 on-call
