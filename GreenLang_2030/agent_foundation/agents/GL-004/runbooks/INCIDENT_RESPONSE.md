# GL-004 BurnerOptimizationAgent - Incident Response Runbook

## Overview

This runbook provides procedures for responding to incidents related to GL-004 BurnerOptimizationAgent.

## Incident Severity Levels

| Level | Description | Response Time | Escalation |
|-------|-------------|---------------|------------|
| P0 | Critical - Safety risk, production down | < 15 min | Immediate |
| P1 | High - Major functionality impaired | < 1 hour | 30 minutes |
| P2 | Medium - Degraded performance | < 4 hours | 2 hours |
| P3 | Low - Minor issues | < 24 hours | Next business day |
| P4 | Informational | Best effort | N/A |

## P0 Incidents

### High Emissions Detected

**Symptoms:**
- NOx > maximum limit (50 ppm)
- CO > maximum limit (100 ppm)
- Emissions compliance alerts

**Immediate Actions:**
1. Verify emissions readings from CEMS
2. Check burner load and operating conditions
3. Review recent optimization changes
4. If emissions persist > 5 minutes:
   - Revert to last known good settings
   - Reduce burner load
   - Increase excess air temporarily

**Root Cause Investigation:**
- Check O2 analyzer calibration
- Verify fuel quality
- Review air flow measurements
- Check for incomplete combustion (CO presence)

**Resolution:**
- Adjust air-fuel ratio gradually
- Re-run optimization with stricter emissions constraints
- Monitor emissions for 30 minutes post-adjustment

### Safety Interlock Triggered

**Symptoms:**
- Flame loss detected
- Fuel/air pressure out of range
- Temperature limits exceeded
- Emergency stop activated

**Immediate Actions:**
1. **DO NOT** attempt to restart without investigation
2. Check all safety interlock statuses
3. Verify flame scanner operation
4. Check fuel and air pressure readings
5. Review temperature sensors

**Safe Restart Procedure:**
1. Ensure all interlocks cleared
2. Perform pre-purge (minimum 5 minutes)
3. Verify ignition system operational
4. Light off at minimum fire
5. Gradually increase to operating load
6. Monitor all parameters closely

**Escalation:**
- If cannot identify root cause: Contact burner manufacturer
- If multiple trips: Shut down and inspect burner physically

### Agent Crash / Unavailable

**Symptoms:**
- Health check failures
- Pod restarts
- Connection timeouts
- No optimization cycles running

**Immediate Actions:**
1. Check Kubernetes pod status:
   ```bash
   kubectl get pods -n greenlang -l app=gl-004
   kubectl logs -n greenlang <pod-name> --tail=100
   ```

2. Review error logs
3. Check resource utilization (CPU, memory)
4. Verify database and Redis connectivity

**Recovery:**
1. If pod is CrashLoopBackOff:
   - Review application logs
   - Check configuration
   - Verify secrets/configmaps

2. If resource exhaustion:
   - Scale up resources temporarily
   - Investigate memory leaks

3. If database issues:
   - Verify DATABASE_URL
   - Check connection pool settings
   - Test database connectivity

## P1 Incidents

### Optimization Not Converging

**Symptoms:**
- Optimization iterations > 100
- Convergence status = "failed"
- No improvement in efficiency

**Actions:**
1. Review current operating conditions
2. Check constraints configuration
3. Verify sensor data quality
4. Adjust optimization parameters:
   - Reduce convergence tolerance
   - Increase max iterations
   - Modify objective weights

2. If still not converging:
   - Revert to manual control
   - Schedule maintenance window for tuning

### Integration Failures

**Symptoms:**
- Modbus/OPC UA connection errors
- Sensor timeouts
- Data quality degradation

**Actions:**
1. Check network connectivity
2. Verify device IP addresses and ports
3. Test protocol communication:
   ```bash
   # For Modbus
   modpoll -m tcp -a 1 -r 0 -c 10 <host> <port>

   # For OPC UA
   opcua-client --endpoint opc.tcp://<host>:<port>
   ```

4. Review integration error logs
5. Check for firmware updates
6. Restart integration services if needed

## P2 Incidents

### Performance Degradation

**Symptoms:**
- API latency > 500ms
- Optimization cycles > 90 seconds
- High CPU/memory usage

**Actions:**
1. Check Prometheus metrics
2. Review slow query logs
3. Analyze trace data
4. Check for database connection pool exhaustion
5. Scale horizontally if needed:
   ```bash
   kubectl scale deployment gl-004 --replicas=5 -n greenlang
   ```

### Data Quality Issues

**Symptoms:**
- Sensor readings out of expected range
- Validation failures
- Inconsistent measurements

**Actions:**
1. Check sensor calibration dates
2. Compare with backup sensors
3. Review data validation rules
4. Temporarily increase data quality thresholds
5. Schedule sensor calibration

## P3 Incidents

### Minor Configuration Issues

**Actions:**
1. Review configuration changes
2. Update ConfigMap:
   ```bash
   kubectl edit configmap gl-004-config -n greenlang
   ```
3. Restart pods to apply changes

### Grafana Dashboard Issues

**Actions:**
1. Verify Prometheus metrics availability
2. Check dashboard JSON configuration
3. Review panel queries
4. Import updated dashboard

## Communication Templates

### P0 Incident Alert
```
PRIORITY: P0 - CRITICAL
SYSTEM: GL-004 BurnerOptimizationAgent
ISSUE: [Brief description]
IMPACT: [Safety/Production impact]
STATUS: Investigating/Mitigated/Resolved
ACTIONS TAKEN: [List of actions]
NEXT STEPS: [Planned actions]
ETA: [Estimated resolution time]
```

### Incident Resolution
```
INCIDENT RESOLVED: GL-004-[ID]
DURATION: [Time from detection to resolution]
ROOT CAUSE: [Identified cause]
ACTIONS TAKEN: [Summary of resolution]
PREVENTIVE MEASURES: [Future prevention]
LESSONS LEARNED: [Key takeaways]
```

## Post-Incident Review

Within 48 hours of P0/P1 incidents:
1. Conduct blameless post-mortem
2. Document timeline of events
3. Identify contributing factors
4. Implement preventive measures
5. Update runbooks with learnings

## Escalation Contacts

- **P0 Incidents**: greenlang-oncall@example.com (24/7)
- **Burner Safety**: safety-team@example.com
- **Infrastructure**: devops-oncall@example.com
- **Management**: director-engineering@example.com

## Related Runbooks

- TROUBLESHOOTING.md - Common issues and solutions
- ROLLBACK_PROCEDURE.md - Safe rollback procedures
- SCALING_GUIDE.md - Horizontal and vertical scaling
- MAINTENANCE.md - Routine maintenance procedures
