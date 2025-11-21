# GL-005 CombustionControlAgent - Incident Response Runbook

## Document Control
- **Version:** 1.0.0
- **Last Updated:** 2025-11-18
- **Owner:** GreenLang Operations Team
- **Review Cycle:** Quarterly

## Executive Summary

This runbook defines incident response procedures for GL-005 CombustionControlAgent, a real-time combustion control system managing fuel-air optimization, heat output regulation, and safety interlocks. Given the critical nature of combustion control (safety risks, production impact), this runbook establishes clear escalation paths and response procedures for all incident severities.

---

## Incident Severity Classification

### P0 - CRITICAL (Production System Down / Safety Risk)

**Definition:**
- Complete combustion control system failure preventing safe operation
- Safety interlock violations causing emergency shutdown
- Control calculations producing invalid/unsafe results (compliance risk)
- Flame loss or combustion instability threatening equipment damage
- Multiple equipment failures preventing automated control

**Response Time:** 15 minutes
**Resolution Target:** 2 hours
**Escalation:** Immediate to VP Engineering + Safety Officer

**Examples:**
- All DCS/PLC connections lost (no control capability)
- Safety interlock malfunction (false trips or failure to trip)
- PID controller instability causing equipment damage
- Database corruption preventing control state persistence
- Flame scanner failure with backup also failed

---

### P1 - HIGH (Degraded Performance / Compliance Risk)

**Definition:**
- Single integration failure with backup available
- Control performance degraded but within safety limits
- Emissions calculations producing incorrect results (regulatory risk)
- Reduced redundancy (single point of failure remaining)
- Performance degradation >20% from baseline

**Response Time:** 30 minutes
**Resolution Target:** 4 hours
**Escalation:** On-call engineer + Engineering manager

**Examples:**
- Primary DCS connection failed (using PLC backup)
- One combustion analyzer offline (O2 sensor failed)
- Control loop latency >100ms (target <100ms)
- Heat output calculation drift >5% from actual
- Fuel-air optimization not converging

---

### P2 - MEDIUM (Partial Degradation)

**Definition:**
- Single non-critical component failure
- Performance degradation <20%
- Non-critical monitoring alert
- Planned maintenance window approaching
- Resource utilization >80%

**Response Time:** 2 hours
**Resolution Target:** 8 hours (next business day)
**Escalation:** On-call engineer

**Examples:**
- SCADA publishing delayed (control still functional)
- Prometheus metrics scraping intermittent
- Temperature sensor array partially offline
- Log aggregation backlog
- Memory usage at 85% (HPA not scaling properly)

---

### P3 - LOW (Minor Issue)

**Definition:**
- Cosmetic issue with no operational impact
- Documentation inconsistency
- Enhancement request
- Non-production environment issue

**Response Time:** Next business day
**Resolution Target:** 1 week
**Escalation:** Standard support ticket

**Examples:**
- Grafana dashboard visualization issue
- API documentation outdated
- Dev environment configuration drift
- Test coverage gap (coverage still >85%)

---

### P4 - INFORMATIONAL (No Action Required)

**Definition:**
- Informational alert
- Planned event
- System operating normally but metric threshold crossed

**Response Time:** N/A
**Resolution Target:** N/A
**Escalation:** None

**Examples:**
- Scheduled maintenance notification
- Normal HPA scale-up event
- Control performance within acceptable variance
- Routine configuration change

---

## GL-005 Specific Incident Scenarios

### Scenario 1: Safety Interlock False Trip (P0 - CRITICAL)

**Symptoms:**
- Emergency shutdown triggered without valid safety condition
- Control system locked out despite safe operating parameters
- Production line stopped
- Alert: `gl005_safety_interlocks_triggered{reason="temperature_high"}` but actual temperature within limits

**Immediate Actions (5 minutes):**
1. Verify actual field conditions (manual readings from DCS/PLC)
2. Confirm false trip (compare sensor readings vs setpoints)
3. Check safety validator logs for root cause
4. If confirmed false trip, initiate bypass procedure (requires dual authorization)

**Root Cause Investigation (30 minutes):**
```bash
# Check safety validator logs
kubectl logs -n greenlang deployment/gl-005-combustion-control --since=1h | grep "SafetyValidator"

# Check sensor calibration status
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  python -c "from integrations import temperature_sensor_array_connector; \
             connector = temperature_sensor_array_connector.TemperatureSensorArrayConnector(); \
             print(connector.validate_sensor_health())"

# Review recent configuration changes
kubectl get configmap gl-005-config -n greenlang -o yaml | grep -A 5 "SAFETY_"
```

**Resolution Steps:**
1. Identify faulty sensor or threshold configuration
2. If sensor failure: Switch to redundant sensor, schedule calibration
3. If threshold misconfiguration: Update safety limits in ConfigMap
4. If software bug: Deploy hotfix with emergency approval
5. Validate fix with controlled test cycle
6. Document incident and update runbook

**Prevention:**
- Weekly sensor calibration checks
- Quarterly safety threshold review
- Dual-sensor validation before trip
- Automated sensor health monitoring

---

### Scenario 2: Control Loop Latency Exceeding 100ms (P1 - HIGH)

**Symptoms:**
- Control response sluggish
- Heat output oscillating
- Alert: `gl005_control_cycle_duration_seconds > 0.1`
- Prometheus: P95 latency >150ms

**Immediate Actions (15 minutes):**
1. Check system resource utilization
2. Verify integration connection health
3. Identify bottleneck (database, integration, calculation)

**Diagnostic Commands:**
```bash
# Check pod resource usage
kubectl top pod -n greenlang -l app=gl-005-combustion-control

# Check control cycle performance breakdown
kubectl logs -n greenlang deployment/gl-005-combustion-control --tail=100 | \
  grep "control_cycle_duration"

# Check integration latencies
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl http://localhost:8001/metrics | grep "integration_latency"

# Check database connection pool
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  python -c "from config import get_config; print(get_config().DB_POOL_SIZE)"
```

**Resolution Steps:**
1. **If CPU bound:** Scale up replicas with HPA override
   ```bash
   kubectl patch hpa gl-005-hpa -n greenlang -p '{"spec":{"minReplicas":5}}'
   ```

2. **If memory bound:** Increase memory limits
   ```bash
   kubectl patch deployment gl-005-combustion-control -n greenlang -p \
     '{"spec":{"template":{"spec":{"containers":[{"name":"gl-005","resources":{"limits":{"memory":"3Gi"}}}]}}}}'
   ```

3. **If database bottleneck:** Increase connection pool
   ```bash
   kubectl set env deployment/gl-005-combustion-control -n greenlang DB_POOL_SIZE=20
   ```

4. **If integration latency:** Check DCS/PLC network connectivity
   ```bash
   kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
     ping -c 5 <DCS_HOST>
   ```

**Prevention:**
- Monitor P95 latency continuously (target <80ms with 20ms buffer)
- Load test quarterly with peak load +20%
- Optimize slow queries (use database query analyzer)
- Cache frequently accessed data (Redis)

---

### Scenario 3: DCS Connection Lost (P1 - HIGH)

**Symptoms:**
- Primary control connection unavailable
- Alert: `gl005_dcs_connection_status{status="disconnected"}`
- System auto-failover to PLC backup
- Operator dashboard shows "DCS OFFLINE"

**Immediate Actions (10 minutes):**
1. Verify PLC backup is active and healthy
2. Confirm control still functional via PLC
3. Check DCS server health (ping, OPC UA server status)
4. Notify operations team of degraded redundancy

**Diagnostic Commands:**
```bash
# Check DCS connector status
kubectl logs -n greenlang deployment/gl-005-combustion-control --tail=50 | \
  grep "DCSConnector"

# Verify PLC backup active
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl http://localhost:8000/status | jq '.integrations.plc'

# Test DCS network connectivity
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  nc -zv <DCS_HOST> <DCS_PORT>

# Check OPC UA server health from DCS admin console
# (manual check required)
```

**Resolution Steps:**
1. **If network issue:** Work with network team to restore connectivity
   - Check firewall rules
   - Verify VLAN configuration
   - Test with tcpdump for packet loss

2. **If DCS server issue:** Work with DCS vendor support
   - Check OPC UA server logs
   - Restart OPC UA server if needed
   - Verify certificate validity (TLS)

3. **If GL-005 configuration issue:** Update DCS endpoint configuration
   ```bash
   kubectl set env deployment/gl-005-combustion-control -n greenlang \
     DCS_HOST=<new_host> DCS_PORT=<new_port>
   ```

4. **Verify restoration:**
   ```bash
   # Monitor reconnection attempts
   kubectl logs -n greenlang deployment/gl-005-combustion-control -f | \
     grep "DCS connection"

   # Confirm circuit breaker closed
   kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
     curl http://localhost:8001/metrics | grep "dcs_circuit_breaker_state"
   ```

**Prevention:**
- Weekly DCS connectivity test
- Redundant network paths to DCS
- Certificate expiration monitoring (90-day notice)
- Quarterly failover testing

---

### Scenario 4: Combustion Instability Detected (P1 - HIGH)

**Symptoms:**
- Flame scanner detecting oscillations
- Alert: `gl005_stability_index < 0.7`
- Heat output varying >10%
- Emissions (CO, NOx) trending up
- Operator reports visible flame pulsation

**Immediate Actions (5 minutes):**
1. Check current stability index and trending
2. Verify fuel-air ratio within acceptable range
3. Review recent control adjustments
4. Check for upstream fuel supply issues

**Diagnostic Commands:**
```bash
# Check stability metrics
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl http://localhost:8000/combustion/stability | jq '.'

# Review recent control actions
kubectl logs -n greenlang deployment/gl-005-combustion-control --tail=20 | \
  grep "ControlAction"

# Check fuel-air ratio trend
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl http://localhost:8001/metrics | grep "gl005_fuel_air_ratio"

# Check emissions trend
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl http://localhost:8001/metrics | grep "gl005_emissions"
```

**Resolution Steps:**
1. **If fuel supply fluctuation:** Contact fuel supplier
   - Check fuel pressure upstream
   - Verify fuel quality (recent delivery?)
   - Check fuel flow meter for accuracy

2. **If air supply issue:** Check combustion air dampers
   - Verify air flow meter readings
   - Check fan VFD operation
   - Inspect air filter differential pressure

3. **If PID tuning issue:** Adjust PID parameters
   ```bash
   # Reduce aggressive control
   kubectl set env deployment/gl-005-combustion-control -n greenlang \
     PID_FUEL_KP=1.5 PID_FUEL_KI=0.3 PID_FUEL_KD=0.05

   # Increase O2 trim damping
   kubectl set env deployment/gl-005-combustion-control -n greenlang \
     PID_O2_TRIM_KD=10
   ```

4. **If burner mechanical issue:** Schedule maintenance
   - Inspect burner nozzle for fouling
   - Check fuel atomization quality
   - Verify ignition system health

**Prevention:**
- Daily stability index monitoring
- Weekly fuel quality testing
- Monthly burner inspection
- Quarterly PID tuning review with process engineer

---

### Scenario 5: Emissions Limit Exceeded (P1 - HIGH)

**Symptoms:**
- Alert: `gl005_emissions_nox_ppm > 50` (EPA limit)
- Alert: `gl005_emissions_co_ppm > 100` (EPA limit)
- Regulatory compliance risk
- Potential permit violation

**Immediate Actions (10 minutes):**
1. Verify emission readings from CEMS (Continuous Emission Monitoring System)
2. Check if transient spike or sustained exceedance
3. Review fuel-air ratio optimization status
4. Notify environmental compliance officer if sustained >1 hour

**Diagnostic Commands:**
```bash
# Check current emissions
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl http://localhost:8000/combustion/emissions | jq '.'

# Review emissions trend (last hour)
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl 'http://prometheus:9090/api/v1/query_range?query=gl005_emissions_nox_ppm&start=-1h&step=60s'

# Check fuel-air ratio optimization
kubectl logs -n greenlang deployment/gl-005-combustion-control --tail=50 | \
  grep "FuelAirOptimizer"

# Check combustion analyzer calibration status
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  python -c "from integrations import combustion_analyzer_connector; \
             connector = combustion_analyzer_connector.CombustionAnalyzerConnector(); \
             print(connector.get_calibration_status())"
```

**Resolution Steps:**
1. **If excess air too low:** Increase air flow
   ```bash
   # Temporarily override target excess air
   kubectl set env deployment/gl-005-combustion-control -n greenlang \
     EXCESS_AIR_TARGET_PERCENT=12.0
   ```

2. **If fuel quality issue:** Check fuel analysis
   - High nitrogen content → higher NOx
   - Poor atomization → higher CO
   - Contact fuel supplier for certification

3. **If burner wear:** Schedule maintenance
   - Worn nozzles cause poor mixing
   - Flame impingement increases NOx
   - Arrange burner replacement during next outage

4. **If analyzer calibration drift:** Recalibrate CEMS
   ```bash
   kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
     python -c "from integrations import combustion_analyzer_connector; \
                connector = combustion_analyzer_connector.CombustionAnalyzerConnector(); \
                connector.calibrate_analyzer()"
   ```

**Regulatory Reporting:**
- Document exceedance event (start time, duration, magnitude, corrective action)
- Report to regulatory agency if required (check permit conditions)
- Update environmental management system records

**Prevention:**
- Real-time emissions monitoring with predictive alerts
- Weekly CEMS calibration verification
- Monthly fuel quality testing
- Quarterly burner maintenance

---

## Escalation Matrix

### Emergency Contacts

| Role | Contact Method | Escalation Trigger |
|------|----------------|-------------------|
| **On-Call Engineer** | Slack @oncall-gl005 (24/7) | All P0/P1 incidents |
| **Engineering Manager** | Slack @eng-manager-combustion | P0 incidents, P1 >2 hours |
| **VP Engineering** | Phone: XXX-XXX-XXXX | P0 only, or P1 unresolved >4 hours |
| **Safety Officer** | Phone: XXX-XXX-XXXX (24/7) | P0 safety-related incidents immediately |
| **Environmental Compliance** | Email: compliance@greenlang.io | Emissions exceedances >1 hour |
| **Operations Manager** | Slack @ops-manager | Production impact incidents |
| **DCS Vendor Support** | Phone: XXX-XXX-XXXX | DCS integration issues |
| **PLC Vendor Support** | Phone: XXX-XXX-XXXX | PLC integration issues |

### Escalation Timing

```
P0 Incident Flow:
T+0 min:  Incident detected → On-call engineer paged
T+15 min: Status update to Slack #gl005-incidents
T+30 min: If unresolved → Escalate to Engineering Manager + Safety Officer
T+60 min: If unresolved → Escalate to VP Engineering
T+90 min: If unresolved → Executive war room

P1 Incident Flow:
T+0 min:  Incident detected → On-call engineer notified
T+30 min: Status update to Slack #gl005-incidents
T+2 hrs:  If unresolved → Escalate to Engineering Manager
T+4 hrs:  If unresolved → Escalate to VP Engineering
```

---

## Communication Protocols

### Internal Communication (Slack)

**Channel:** #gl005-incidents

**P0 Template:**
```
🚨 P0 INCIDENT - GL-005 CombustionControlAgent

**Status:** Investigating / Mitigating / Resolved
**Started:** 2025-11-18 14:30 UTC
**Impact:** Complete loss of automated combustion control
**Current Action:** Switching to manual control mode
**ETA:** 2 hours
**Incident Commander:** @john.doe
**Updates:** Every 15 minutes

[Incident Link: https://greenlang.atlassian.net/INC-12345]
```

**P1 Template:**
```
⚠️ P1 INCIDENT - GL-005 CombustionControlAgent

**Status:** Investigating / Mitigating / Resolved
**Started:** 2025-11-18 14:30 UTC
**Impact:** DCS connection lost, using PLC backup
**Current Action:** Diagnosing DCS network connectivity
**ETA:** 4 hours
**Owner:** @jane.smith
**Updates:** Every 30 minutes

[Incident Link: https://greenlang.atlassian.net/INC-12346]
```

### External Communication (Customers)

**Trigger:** P0 incidents affecting production >30 minutes

**Status Page Update Template:**
```
Title: GL-005 Combustion Control Service Degradation

Status: Investigating / Identified / Monitoring / Resolved

Description:
We are currently investigating an issue with the GL-005 CombustionControlAgent
affecting automated combustion control. Manual control mode has been activated
to ensure safe operation.

Impact:
- Automated combustion optimization unavailable
- Manual operator intervention required
- Heat output stability may be reduced

Next Update: 15 minutes

[Posted: 2025-11-18 14:35 UTC]
[Updated: 2025-11-18 14:50 UTC]
```

---

## Post-Incident Review

### Conduct PIR within 48 hours of P0/P1 resolution

**PIR Agenda:**
1. Incident timeline (5 minutes)
2. Root cause analysis (15 minutes)
3. Resolution effectiveness (10 minutes)
4. What went well (5 minutes)
5. What could be improved (10 minutes)
6. Action items (5 minutes)

**PIR Template:** See `INCIDENT_PIR_TEMPLATE.md`

**Action Item Tracking:**
- Create Jira tickets for all action items
- Assign owners and due dates
- Review in weekly engineering standup
- Close loop in next PIR

---

## Incident Metrics & SLAs

### Target SLAs

| Severity | Detection Time | Response Time | Resolution Time | Availability Impact |
|----------|----------------|---------------|-----------------|---------------------|
| P0 | <5 minutes | <15 minutes | <2 hours | >0.1% |
| P1 | <10 minutes | <30 minutes | <4 hours | 0.01-0.1% |
| P2 | <30 minutes | <2 hours | <8 hours | <0.01% |
| P3 | <4 hours | <1 day | <1 week | None |

### Monitoring Dashboards

**Primary:** https://grafana.greenlang.io/d/gl005-incidents

**Key Metrics:**
- Incident count by severity (weekly)
- Mean Time To Detect (MTTD)
- Mean Time To Respond (MTTR)
- Mean Time To Resolve (MTTR)
- Repeat incident rate
- Post-incident action item completion rate

---

## Runbook Maintenance

**Review Cycle:** Quarterly
**Owner:** GL-005 Product Owner
**Contributors:** On-call engineers, Safety Officer, Operations

**Review Checklist:**
- [ ] Escalation contacts up to date
- [ ] Incident scenarios reflect recent incidents
- [ ] Commands tested and functional
- [ ] SLAs still appropriate
- [ ] Regulatory requirements current
- [ ] Integration endpoints accurate

**Version History:**
| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-18 | AI Agent | Initial version |

---

## Related Documents

- [GL-005 Troubleshooting Guide](./TROUBLESHOOTING.md)
- [GL-005 Rollback Procedures](./ROLLBACK_PROCEDURE.md)
- [GL-005 Maintenance Schedule](./MAINTENANCE.md)
- [GL-005 Scaling Guide](./SCALING_GUIDE.md)
- [GreenLang Incident Management Policy](../../policies/INCIDENT_MANAGEMENT.md)
