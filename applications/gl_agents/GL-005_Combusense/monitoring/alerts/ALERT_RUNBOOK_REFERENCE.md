# GL-005 Alert Runbook Quick Reference

## Critical Alerts (P0) - Immediate Response Required

### GL005SafetyInterlock
**Alert:** Safety interlock triggered, system in UNSAFE state
**Response Time:** Immediate (0-2 minutes)
**Severity:** Critical
**SIL-2:** Yes

**Immediate Actions:**
1. Acknowledge alert in Alertmanager/PagerDuty
2. Check safety dashboard: `https://grafana.greenlang.io/d/gl005-safety`
3. Identify interlock reason from alert labels: `{{ $labels.reason }}`
4. DO NOT attempt to override safety interlock
5. Notify on-call control engineer and operations supervisor

**Common Causes:**
- Flame loss detected
- Temperature exceeded safety limit
- Pressure exceeded safety limit
- Manual emergency stop activated
- Loss of critical instrumentation

**Resolution Steps:**
1. Verify burner shutdown completed successfully
2. Investigate root cause of interlock trigger
3. Resolve underlying issue (restore sensor, fix equipment, etc.)
4. Perform safety checklist before attempting restart
5. Execute purge cycle (5 minutes minimum per NFPA 86)
6. Obtain supervisor approval to reset interlock
7. Monitor closely during startup

**Escalation:**
- If interlock persists >15 minutes: Notify plant manager
- If safety equipment failure suspected: Contact safety instrumented systems (SIS) vendor
- If regulatory reportable: Notify EHS team immediately

**Detailed Runbook:** [INCIDENT_RESPONSE.md#scenario-1](../runbooks/INCIDENT_RESPONSE.md#scenario-1-safety-interlock-triggered)

---

### GL005EmergencyShutdown
**Alert:** Emergency shutdown event triggered
**Response Time:** Immediate (0-2 minutes)
**Severity:** Critical
**SIL-2:** Yes

**Immediate Actions:**
1. Confirm all burners have shut down (visual/DCS confirmation)
2. Check for personnel safety issues - evacuate if needed
3. Identify shutdown reason: `{{ $labels.shutdown_reason }}`
4. Secure fuel supply (verify valves closed)
5. Notify emergency response team and management

**Common Shutdown Reasons:**
- High-high temperature alarm
- High-high pressure alarm
- Explosive gas detection
- Multiple burner flame failures
- Loss of forced draft fan
- DCS/PLC communication failure

**Resolution Steps:**
1. Complete incident report form (mandatory for all ESD events)
2. Perform equipment inspection (check for damage)
3. Verify all safety systems functional
4. Address root cause
5. Execute full startup checklist (purge, leak test, ignition sequence)
6. Regulatory notification if required (EPA, OSHA)

**Post-Incident:**
- File incident report within 24 hours
- Conduct root cause analysis meeting within 48 hours
- Update maintenance schedule if equipment failure identified

**Detailed Runbook:** [INCIDENT_RESPONSE.md#scenario-2](../runbooks/INCIDENT_RESPONSE.md#scenario-2-emergency-shutdown)

---

### GL005ControlLoopDown
**Alert:** Control loop stopped executing
**Response Time:** Immediate (0-5 minutes)
**Severity:** Critical
**SIL-2:** No (but safety impact)

**Immediate Actions:**
1. Check if system has automatically switched to manual control
2. Verify operator is aware and ready to take manual control
3. Check GL-005 pod/container status: `kubectl get pods -n greenlang | grep gl-005`
4. Check application logs: `kubectl logs -n greenlang deployment/gl-005-combustion-control --tail=100`

**Common Causes:**
- Application crash (OOM, segfault, panic)
- Kubernetes pod eviction (node failure, resource limits)
- Configuration error after deployment
- Integration failure (DCS/PLC unreachable)

**Resolution Steps:**
1. If pod crashed, check for restart loop:
   ```bash
   kubectl describe pod -n greenlang <pod-name>
   kubectl logs -n greenlang <pod-name> --previous
   ```
2. Check resource utilization:
   ```bash
   kubectl top pod -n greenlang <pod-name>
   ```
3. If OOM kill, increase memory limits temporarily:
   ```bash
   kubectl set resources deployment gl-005-combustion-control --limits=memory=4Gi
   ```
4. If config error, rollback to previous version:
   ```bash
   kubectl rollout undo deployment/gl-005-combustion-control -n greenlang
   ```
5. Monitor control loop resumption:
   ```promql
   rate(gl005_control_loop_executions_total[1m])
   ```

**Escalation:**
- If loop doesn't recover in 5 minutes: Page backend on-call engineer
- If requires code fix: Initiate hotfix deployment process
- If plant safety at risk: Initiate manual shutdown

**Detailed Runbook:** [TROUBLESHOOTING.md#issue-11](../runbooks/TROUBLESHOOTING.md#issue-11-control-loop-stopped)

---

### GL005FlameFailure
**Alert:** Flame detection failure on burner
**Response Time:** Immediate (0-2 minutes)
**Severity:** Critical
**SIL-2:** Yes

**Immediate Actions:**
1. Verify automatic fuel shutoff occurred (should be <1 second)
2. Confirm no unburned fuel accumulation in furnace
3. Check flame scanner status for affected burner
4. DO NOT attempt reignition until root cause identified

**Common Causes:**
- Burner tip plugged/fouled
- Fuel supply interruption
- Flame scanner malfunction
- UV sensor failure
- Excessive draft/air flow

**Resolution Steps:**
1. Inspect flame scanner (clean lens if fouled)
2. Test flame scanner signal (should be >2μA with flame present)
3. Check fuel pressure/flow to burner
4. Inspect burner tip for obstructions
5. Verify pilot igniter functional
6. Perform purge cycle before reignition attempt
7. Monitor flame signal during startup

**Safety Notes:**
- Never bypass flame detection
- If multiple flame failures occur, investigate common cause
- Report recurring failures to maintenance for PM

**Detailed Runbook:** [INCIDENT_RESPONSE.md#scenario-6](../runbooks/INCIDENT_RESPONSE.md#scenario-6-flame-failure)

---

### GL005TemperatureExceeded
**Alert:** Furnace temperature exceeded safety limit
**Response Time:** Immediate (0-3 minutes)
**Severity:** Critical
**SIL-2:** Yes

**Immediate Actions:**
1. Reduce firing rate immediately (manual override if needed)
2. Verify temperature reading is accurate (check redundant sensors)
3. Increase cooling (if applicable - water sprays, air flow)
4. Monitor for emergency shutdown threshold

**Common Causes:**
- Control system malfunction (runaway fuel flow)
- Temperature sensor failure (reads low, causes overfire)
- Heat removal system failure (cooling water, process flow stopped)
- Fuel composition change (higher BTU content)

**Resolution Steps:**
1. Verify all temperature sensors reading consistently
2. Check fuel flow control valve position (should modulate down)
3. Inspect refractory for damage (hot spots indicate failure)
4. Review fuel analysis (BTU content, composition)
5. Calibrate temperature sensors
6. Test control loop response (setpoint step test)

**Emergency Shutdown Triggers:**
- Temperature >1450°C (material failure risk)
- Temperature rising >50°C/min (runaway condition)
- Refractory damage observed

**Detailed Runbook:** [INCIDENT_RESPONSE.md#scenario-7](../runbooks/INCIDENT_RESPONSE.md#scenario-7-temperature-runaway)

---

## High Priority Alerts (P1) - Urgent Response

### GL005ControlLatencyHigh
**Alert:** Control loop latency exceeds 100ms SLO
**Response Time:** 5-15 minutes
**Severity:** High

**Immediate Actions:**
1. Check performance dashboard: `https://grafana.greenlang.io/d/gl005-agent-performance`
2. Identify which agent(s) contributing to latency
3. Check resource utilization (CPU, memory)

**Quick Checks:**
```bash
# Check pod resource usage
kubectl top pod -n greenlang | grep gl-005

# Check agent-level latency breakdown
curl http://gl-005-service:8001/metrics | grep agent_duration

# Check integration latency
curl http://gl-005-service:8001/metrics | grep dcs_read_latency
curl http://gl-005-service:8001/metrics | grep plc_read_latency
```

**Common Causes:**
- DCS/PLC network latency spike
- CPU throttling (resource limits)
- Memory pressure (GC pauses)
- Database query slowdown
- Agent code inefficiency

**Resolution Steps:**
1. If DCS/PLC latency: Check network path, restart OPC UA client
2. If CPU throttling: Increase CPU limits temporarily
3. If memory pressure: Restart pod, investigate memory leak
4. If database slow: Check query performance, add indexes
5. Review recent code changes for performance regressions

**Detailed Runbook:** [TROUBLESHOOTING.md#issue-11](../runbooks/TROUBLESHOOTING.md#issue-11-control-loop-latency)

---

### GL005StabilityLow
**Alert:** Combustion stability index below threshold
**Response Time:** 5-15 minutes
**Severity:** High

**Immediate Actions:**
1. Check combustion dashboard: `https://grafana.greenlang.io/d/gl005-combustion`
2. Verify stability index calculation: oscillation amplitude, frequency
3. Check if manual operator adjustments needed

**Symptoms:**
- Oscillating fuel flow, air flow, or temperature
- Visible flame pulsation/flicker
- Increased emissions (NOx, CO)
- Audible combustion noise changes

**Common Causes:**
- PID tuning too aggressive (high gains)
- Fuel supply pressure fluctuations
- Air damper sticking/hunting
- Process load changes (feed rate)
- Sensor noise/failure

**Resolution Steps:**
1. Review PID controller gains (reduce if hunting)
2. Check fuel pressure regulator (should be stable)
3. Inspect air damper actuator (lubricate, calibrate)
4. Tune oscillation damping parameters
5. Filter noisy sensors (increase averaging window)

**Detailed Runbook:** [INCIDENT_RESPONSE.md#scenario-4](../runbooks/INCIDENT_RESPONSE.md#scenario-4-combustion-instability)

---

### GL005EmissionsExceeded
**Alert:** NOx or CO emissions exceed EPA limits
**Response Time:** 5-10 minutes
**Severity:** High
**Regulatory:** Yes (EPA violation)

**Immediate Actions:**
1. Document timestamp and emission values (legal requirement)
2. Adjust combustion parameters to reduce emissions:
   - **High NOx:** Reduce excess air, lower flame temperature
   - **High CO:** Increase excess air, improve mixing
3. Notify environmental compliance officer

**Emissions Limits:**
- **NOx:** 50 ppm @ 3% O₂ (EPA limit)
- **CO:** 100 ppm @ 3% O₂ (EPA limit)

**Quick Adjustments:**
```
High NOx → Reduce O₂ setpoint by 0.5%
         → Reduce firing rate if possible
         → Enable FGR (flue gas recirculation) if available

High CO → Increase O₂ setpoint by 0.5%
        → Check burner for fouling
        → Verify air registers open
```

**Common Causes:**
- Excess air too low (high CO) or too high (high NOx)
- Burner fouling/wear (poor mixing)
- Fuel quality change
- Temperature too high (high NOx)
- Incomplete combustion (high CO)

**Resolution Steps:**
1. Verify CEMS (continuous emissions monitoring) accuracy
2. Optimize air-fuel ratio
3. Inspect/clean burners
4. Review fuel analysis
5. Re-tune combustion controls

**Regulatory Requirements:**
- Report exceedances >10 minutes duration to EPA
- Maintain emissions records for 5 years
- Investigate root cause within 48 hours

**Detailed Runbook:** [INCIDENT_RESPONSE.md#scenario-5](../runbooks/INCIDENT_RESPONSE.md#scenario-5-emissions-violation)

---

### GL005DCSConnectionLost
**Alert:** DCS connection lost, no sensor data
**Response Time:** 2-5 minutes
**Severity:** High

**Immediate Actions:**
1. Verify operator has local control via DCS
2. Check if backup control mode activated
3. Test DCS connectivity from GL-005 pod

**Quick Diagnostics:**
```bash
# Test DCS network connectivity
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  ping -c 5 <dcs-ip-address>

# Check OPC UA client status
kubectl logs -n greenlang deployment/gl-005-combustion-control | grep "OPC UA"

# Restart OPC UA connection
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8001/admin/reconnect-dcs
```

**Common Causes:**
- DCS server reboot/maintenance
- Network switch failure
- Firewall rule change
- OPC UA certificate expired
- Client connection leak (too many open connections)

**Resolution Steps:**
1. Verify DCS server running and reachable
2. Check network path (switches, firewalls)
3. Renew OPC UA certificates if expired
4. Restart GL-005 application (reconnects OPC UA client)
5. Contact DCS vendor if server-side issue

**Detailed Runbook:** [INCIDENT_RESPONSE.md#scenario-3](../runbooks/INCIDENT_RESPONSE.md#scenario-3-dcs-connection-failure)

---

## Medium Priority Alerts (P2) - Investigation Needed

### GL005MemoryHigh / GL005CPUHigh
**Alert:** Resource usage approaching limits
**Response Time:** 15-30 minutes
**Severity:** Medium

**Quick Actions:**
```bash
# Check resource usage trends
kubectl top pod -n greenlang | grep gl-005

# Check for memory leaks
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl http://localhost:8001/debug/pprof/heap > heap.prof

# Increase limits temporarily (if needed)
kubectl set resources deployment gl-005-combustion-control \
  --limits=cpu=2000m,memory=4Gi
```

**Investigation:**
1. Check for memory/CPU usage trends (gradual increase = leak)
2. Review recent code changes
3. Check for database connection leaks
4. Profile application (heap, CPU)

**Detailed Runbook:** [TROUBLESHOOTING.md#issue-8](../runbooks/TROUBLESHOOTING.md#issue-8-high-memory)

---

### GL005HeatOutputDeviation
**Alert:** Heat output deviating >5% from target
**Response Time:** 15-30 minutes
**Severity:** Medium

**Immediate Actions:**
1. Check if production demand changed (expected deviation)
2. Verify control loop tracking setpoint
3. Review operator adjustments

**Investigation:**
1. Check fuel flow measurement accuracy
2. Verify heat content of fuel (BTU/kg)
3. Review efficiency calculations
4. Check process heat losses

**Detailed Runbook:** [TROUBLESHOOTING.md#issue-13](../runbooks/TROUBLESHOOTING.md#issue-13-heat-output-deviation)

---

## SIL-2 Compliance Alerts

### GL005SIL2ComplianceFailure
**Alert:** SIL-2 compliance check failed
**Response Time:** Immediate (0-5 minutes)
**Severity:** Critical
**Regulatory:** IEC 61511 violation

**Immediate Actions:**
1. Identify failed compliance check: `{{ $labels.failure_reason }}`
2. Verify safety system still providing protection (degraded mode acceptable temporarily)
3. Notify safety instrumented systems (SIS) engineer

**Common Failures:**
- Safety check duration exceeded 20ms (response time violation)
- Redundant sensor disagreement (voting failure)
- Watchdog timeout (proof test failure)
- Communication loss to safety PLC

**Resolution Steps:**
1. Review SIL-2 compliance criteria
2. Investigate failed checks
3. Perform proof test if required
4. Document deviation and corrective actions
5. Restore full SIL-2 compliance before resuming production

**Regulatory:**
- IEC 61511 requires documented response to SIL violations
- Notify regulatory authority if safety risk present
- Update safety case documentation

**Detailed Runbook:** [INCIDENT_RESPONSE.md#scenario-8](../runbooks/INCIDENT_RESPONSE.md#scenario-8-sil2-compliance-failure)

---

## Alert Routing Matrix

| Alert | PagerDuty Severity | Notify | Response Time |
|-------|-------------------|--------|---------------|
| GL005SafetyInterlock | P0 Critical | Ops + Control Eng + Manager | 0-2 min |
| GL005EmergencyShutdown | P0 Critical | Ops + Control Eng + Manager + EHS | 0-2 min |
| GL005ControlLoopDown | P0 Critical | Ops + Backend Eng | 0-5 min |
| GL005FlameFailure | P0 Critical | Ops + Control Eng | 0-2 min |
| GL005TemperatureExceeded | P0 Critical | Ops + Control Eng | 0-3 min |
| GL005ControlLatencyHigh | P1 High | Backend Eng | 5-15 min |
| GL005StabilityLow | P1 High | Ops + Control Eng | 5-15 min |
| GL005EmissionsExceeded | P1 High | Ops + EHS | 5-10 min |
| GL005DCSConnectionLost | P1 High | Ops + IT Network | 2-5 min |
| GL005MemoryHigh | P2 Medium | Backend Eng | 15-30 min |
| GL005HeatOutputDeviation | P2 Medium | Ops | 15-30 min |
| GL005SIL2ComplianceFailure | P0 Critical | Ops + SIS Eng + Manager | 0-5 min |

---

## Escalation Paths

### Level 1: On-Call Engineer (0-15 minutes)
- Acknowledge alert
- Perform initial investigation
- Execute runbook procedures
- Resolve if within capability

### Level 2: Technical Lead (15-30 minutes)
- Escalate if Level 1 cannot resolve
- Coordinate cross-functional response
- Approve emergency changes
- Engage vendor support if needed

### Level 3: Management (30-60 minutes)
- Escalate if plant safety at risk
- Escalate if regulatory violation
- Escalate if customer impact >1 hour
- Authorize emergency shutdowns

### Level 4: Executive (>60 minutes or major incident)
- Notify for extended outages
- Notify for regulatory reportable events
- Notify for safety incidents
- Approve major expenditures for emergency repairs

---

## Quick Reference Links

- **Grafana Dashboards:** https://grafana.greenlang.io
  - Agent Performance: `/d/gl005-agent-performance`
  - Combustion Metrics: `/d/gl005-combustion`
  - Safety Monitoring: `/d/gl005-safety`
- **Prometheus Alerts:** https://prometheus.greenlang.io/alerts (internal)
- **Alertmanager:** https://alertmanager.greenlang.io (internal)
- **PagerDuty:** https://greenlang.pagerduty.com
- **Runbooks:** `/docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-005/runbooks/`
- **SLO Definitions:** `/docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-005/monitoring/SLO_DEFINITIONS.md`

---

## Post-Incident Actions

After resolving any P0 or P1 alert:

1. **Document in incident log** (Jira, ServiceNow, or incident management system)
2. **Update runbook** if new information discovered
3. **Schedule post-mortem** (within 48 hours for P0, within 1 week for P1)
4. **Review error budget** impact
5. **Create follow-up tasks** for permanent fixes
6. **Update monitoring/alerts** if gaps identified

---

**Document Version:** 1.0
**Last Updated:** 2025-11-26
**Owner:** GL-005 Reliability Team
**Review Cadence:** Quarterly
