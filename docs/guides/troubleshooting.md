# GreenLang Process Heat Platform - Troubleshooting Guide

**Document Version:** 1.0.0
**Last Updated:** 2025-12-06
**Classification:** Technical Support Documentation
**Intended Audience:** Operators, Engineers, Support Staff

---

## Table of Contents

1. [Common Error Messages and Solutions](#1-common-error-messages-and-solutions)
2. [Agent Health Diagnostics](#2-agent-health-diagnostics)
3. [Data Quality Issues](#3-data-quality-issues)
4. [Integration Problems](#4-integration-problems)
5. [Performance Issues](#5-performance-issues)
6. [Calculation Errors](#6-calculation-errors)
7. [Authentication and Access Issues](#7-authentication-and-access-issues)
8. [Escalation Procedures](#8-escalation-procedures)

---

## 1. Common Error Messages and Solutions

### 1.1 Error Message Reference

#### ERR-001: Agent Not Responding

**Message:** `Agent GL-XXX-XXXX is not responding (timeout after 5000ms)`

**Cause:** The agent has not sent a heartbeat within the configured timeout period.

**Solutions:**

1. **Check agent status:**
   ```
   Navigate to Agents > [Agent ID] > Status
   ```

2. **Verify network connectivity:**
   - Ensure the agent server is reachable
   - Check firewall rules

3. **Restart the agent:**
   ```
   Agent Detail > [Restart Agent] button
   ```

4. **Check system resources:**
   - Memory: Should be < 80%
   - CPU: Should be < 90%

5. **Review agent logs:**
   ```
   Agent Detail > [View Logs]
   ```

**If issue persists:** Escalate to Level 2 Support

---

#### ERR-002: Database Connection Failed

**Message:** `Unable to connect to database: Connection refused`

**Cause:** The database server is unreachable or not accepting connections.

**Solutions:**

1. **Verify database is running:**
   ```bash
   # For administrators
   sudo systemctl status postgresql
   ```

2. **Check connection settings:**
   - Host, port, database name correct?
   - Credentials valid?

3. **Test database connectivity:**
   ```bash
   greenlang-cli diagnose database
   ```

4. **Check connection pool:**
   - Maximum connections may be exhausted
   - Review connection pool metrics

**If issue persists:** Contact Database Administrator

---

#### ERR-003: Safety Interlock Triggered

**Message:** `Safety interlock [INTERLOCK_ID] triggered: [CONDITION]`

**Cause:** A safety condition has been detected that requires intervention.

**Solutions:**

1. **DO NOT IGNORE** - Safety interlocks exist for protection

2. **Identify the condition:**
   - Review the interlock message
   - Check related equipment status

3. **Address the root cause:**
   - High pressure: Check relief valves, reduce firing
   - Low water: Verify feedwater system
   - High temperature: Reduce load, check cooling

4. **Reset only when safe:**
   - Verify condition is resolved
   - Obtain supervisor authorization
   - Document the incident

**WARNING:** Never bypass safety interlocks without proper authorization and permit-to-work.

---

#### ERR-004: Calculation Failed

**Message:** `Calculation failed: [REASON]`

**Common Reasons and Solutions:**

| Reason | Solution |
|--------|----------|
| Invalid input data | Check sensor readings for reasonableness |
| Division by zero | Verify flow rates are > 0 |
| Out of range | Check parameter limits in configuration |
| Missing required field | Ensure all required sensors are reporting |
| Timeout | Check system load, try again |

**Diagnostic steps:**

1. Review input data:
   ```
   Agent Detail > Recent Calculations > [Failed] > View Inputs
   ```

2. Check sensor status:
   ```
   Trends > [Relevant sensors] > Verify readings
   ```

3. Retry calculation manually:
   ```
   Agent Detail > [Calculate Now]
   ```

---

#### ERR-005: Communication Timeout

**Message:** `Communication timeout with [SYSTEM]: No response after [X]ms`

**Systems affected:** OPC-UA, MQTT, Kafka, DCS

**Solutions:**

1. **Check network path:**
   ```bash
   ping [target-host]
   traceroute [target-host]
   ```

2. **Verify service is running:**
   - Check OPC-UA server status
   - Verify MQTT broker health
   - Test Kafka connectivity

3. **Check authentication:**
   - Certificates valid and not expired?
   - Credentials correct?

4. **Review firewall rules:**
   - Required ports open?
   - No recent firewall changes?

5. **Check for overload:**
   - Target system under high load?
   - Network congestion?

---

#### ERR-006: Permit Exceedance Detected

**Message:** `Permit limit exceeded: [POLLUTANT] at [VALUE] exceeds limit of [LIMIT]`

**Severity:** HIGH - Requires immediate action

**Solutions:**

1. **Immediately notify:**
   - Shift Supervisor
   - Environmental Coordinator

2. **Reduce emissions source:**
   - Lower firing rate if possible
   - Switch to cleaner fuel if available

3. **Document the incident:**
   - Record time, duration, values
   - Capture provenance hash

4. **Investigate root cause:**
   - Equipment malfunction?
   - Operating condition change?
   - Measurement error?

5. **File required reports:**
   - Internal incident report
   - Regulatory notification (if required)

---

#### ERR-007: Emergency Shutdown Activated

**Message:** `EMERGENCY SHUTDOWN activated by [SOURCE]: [REASON]`

**Severity:** CRITICAL

**Immediate Actions:**

1. **Assess safety:**
   - Is there immediate danger?
   - Are all personnel safe?

2. **Notify:**
   - Shift Supervisor (immediately)
   - Operations Manager
   - Safety Manager

3. **Do NOT attempt restart until:**
   - Root cause identified
   - Hazard eliminated
   - Authorization obtained

4. **Document everything:**
   - Time of ESD
   - Conditions before ESD
   - Actions taken

See [Emergency Procedures in Operator Manual](./operator_manual.md#8-emergency-procedures)

---

### 1.2 Error Code Quick Reference

| Code | Category | Severity | Description |
|------|----------|----------|-------------|
| ERR-001 | Agent | Medium | Agent not responding |
| ERR-002 | Database | High | Database connection failed |
| ERR-003 | Safety | Critical | Safety interlock triggered |
| ERR-004 | Calculation | Low | Calculation failed |
| ERR-005 | Communication | Medium | Communication timeout |
| ERR-006 | Compliance | High | Permit exceedance |
| ERR-007 | Safety | Critical | Emergency shutdown |
| ERR-008 | Authentication | Medium | Authentication failed |
| ERR-009 | Configuration | Medium | Invalid configuration |
| ERR-010 | Resource | Medium | Resource exhaustion |

---

## 2. Agent Health Diagnostics

### 2.1 Understanding Agent Health

Agent health is displayed as a percentage (0-100%):

| Score | Status | Meaning |
|-------|--------|---------|
| 95-100% | Healthy | Normal operation |
| 80-94% | Degraded | Minor issues, monitoring |
| 50-79% | Warning | Investigate required |
| 0-49% | Critical | Immediate action needed |

### 2.2 Health Score Components

The health score is calculated from:

| Component | Weight | Description |
|-----------|--------|-------------|
| Heartbeat | 25% | Regular communication with orchestrator |
| Processing | 25% | Successful calculation completion |
| Latency | 20% | Response time within limits |
| Errors | 20% | Error rate below threshold |
| Resources | 10% | Memory/CPU within limits |

### 2.3 Diagnosing Unhealthy Agents

**Step 1: Identify the Problem**

Navigate to: `Agents > [Agent ID] > Health Details`

```
+------------------------------------------------------------------+
|  AGENT HEALTH DETAILS - GL-002-B001                              |
+------------------------------------------------------------------+
|  Overall Health: 72%                                              |
+------------------------------------------------------------------+
|  Component        | Score | Status    | Details                  |
+------------------------------------------------------------------+
|  Heartbeat        | 100%  | Healthy   | Last: 2 seconds ago      |
|  Processing       | 45%   | Warning   | 12 of 20 calcs failed    |
|  Latency          | 85%   | Healthy   | Avg: 245ms               |
|  Errors           | 60%   | Degraded  | Error rate: 8.5%         |
|  Resources        | 90%   | Healthy   | Memory: 1.2GB / 2GB      |
+------------------------------------------------------------------+
```

**Step 2: Investigate Low Scores**

For **Processing** issues:
- Check recent calculation failures
- Review input data quality
- Verify sensor connectivity

For **Error** issues:
- Review error logs
- Check for recurring patterns
- Identify root cause

For **Latency** issues:
- Check network connectivity
- Review system load
- Check dependent services

For **Resource** issues:
- Monitor memory trends
- Check for memory leaks
- Review batch sizes

**Step 3: Take Corrective Action**

Common remediation steps:

| Issue | Action |
|-------|--------|
| High failure rate | Fix input data, restart agent |
| High error rate | Review logs, fix root cause |
| High latency | Check network, reduce load |
| High memory | Restart agent, adjust config |
| Missed heartbeats | Check connectivity, restart |

### 2.4 Agent Health Monitoring Dashboard

Create an agent health dashboard:

1. Navigate to `Trends > Create Dashboard`
2. Add widgets for each agent:
   - Health score gauge
   - Error rate chart
   - Processing time trend
3. Set refresh interval to 30 seconds
4. Configure alerts for health < 80%

---

## 3. Data Quality Issues

### 3.1 Common Data Quality Problems

#### Problem: Sensor Reading Stuck

**Symptoms:**
- Value hasn't changed in extended period
- Trend shows flat line
- Calculations using stale data

**Diagnosis:**
1. Navigate to `Trends > [Sensor]`
2. Check last update timestamp
3. Compare to related sensors

**Solutions:**
- Verify sensor is operational
- Check communication path
- Restart data collection service
- Contact instrumentation team

---

#### Problem: Sensor Reading Spikes

**Symptoms:**
- Sudden unrealistic value changes
- Triggered false alarms
- Calculation errors

**Diagnosis:**
1. Review trend data for spike pattern
2. Check related sensors at same time
3. Compare to expected physical range

**Solutions:**
- Enable spike filtering in configuration
- Check for electrical interference
- Verify sensor calibration
- Add rate-of-change limits

---

#### Problem: Sensor Out of Range

**Symptoms:**
- Value outside physical limits
- Validation errors in calculations
- Quality tags showing "BAD"

**Diagnosis:**
1. Check raw sensor value
2. Verify engineering unit conversion
3. Compare to reference instrument

**Solutions:**
- Recalibrate sensor
- Verify scaling factors
- Check for sensor failure
- Replace sensor if needed

---

### 3.2 Data Quality Indicators

| Quality Flag | Meaning | Action |
|--------------|---------|--------|
| GOOD | Valid data | None |
| UNCERTAIN | Data quality questionable | Monitor |
| BAD | Invalid data | Investigate |
| STALE | Data not updated | Check sensor |
| SUBSTITUTED | Manual/calculated value | Verify source |
| CLAMPED | Value at limit | Check sensor |

### 3.3 Data Validation Settings

Configure data validation in agent settings:

```yaml
# Agent configuration - data validation
validation:
  enabled: true

  # Range validation
  range_checks:
    - sensor: steam_pressure_psig
      min: 0
      max: 300
      action: reject

    - sensor: flue_gas_o2_pct
      min: 0
      max: 21
      action: clamp

  # Rate of change validation
  rate_checks:
    - sensor: steam_temperature_f
      max_change_per_second: 10
      action: filter

  # Staleness check
  staleness:
    max_age_seconds: 60
    action: substitute_last_good
```

### 3.4 Data Quality Troubleshooting Checklist

- [ ] Verify sensor is physically working
- [ ] Check wiring and connections
- [ ] Verify signal conditioning
- [ ] Check A/D conversion
- [ ] Verify engineering units
- [ ] Check communication path
- [ ] Verify polling/scan rate
- [ ] Review quality tags
- [ ] Check historical trend
- [ ] Compare to redundant sensors

---

## 4. Integration Problems

### 4.1 OPC-UA Issues

#### Problem: Cannot Connect to OPC-UA Server

**Error:** `OPC-UA connection failed: BadSecurityChecksFailed`

**Diagnosis:**
```bash
greenlang-cli diagnose opcua --server opc.tcp://[server]:4840
```

**Solutions:**

1. **Check certificate:**
   - Is client certificate trusted by server?
   - Is server certificate trusted by client?
   - Are certificates not expired?

2. **Verify security mode:**
   - Does client security mode match server?
   - Try `None` mode for testing (not production)

3. **Check credentials:**
   - Username/password correct?
   - User account enabled on server?

4. **Verify endpoint:**
   - Is endpoint URL correct?
   - Is server accepting connections?

---

#### Problem: OPC-UA Subscription Lost

**Error:** `Subscription [ID] terminated unexpectedly`

**Solutions:**

1. Check server status
2. Verify network stability
3. Review server logs for disconnect reason
4. Increase keep-alive interval
5. Restart subscription

---

### 4.2 MQTT Issues

#### Problem: Cannot Connect to MQTT Broker

**Error:** `MQTT connection refused`

**Diagnosis:**
```bash
greenlang-cli diagnose mqtt --broker mqtt://[broker]:1883
```

**Solutions:**

1. Verify broker is running
2. Check port accessibility
3. Verify credentials
4. Check SSL/TLS configuration
5. Review broker connection limits

---

#### Problem: Messages Not Received

**Symptoms:** Subscribed topics showing no data

**Solutions:**

1. Verify topic subscription is correct
2. Check QoS settings
3. Verify publisher is active
4. Check for topic wildcards
5. Review broker logs

---

### 4.3 DCS Integration Issues

#### Problem: DCS Tags Not Updating

**Symptoms:** Tag values stale or not changing

**Solutions:**

1. **Check DCS side:**
   - Is the interface card operational?
   - Are tags configured for external access?

2. **Check network:**
   - Is OPC server reachable?
   - Any network latency issues?

3. **Check GreenLang configuration:**
   - Tag names correct?
   - Polling rate appropriate?

4. **Verify permissions:**
   - Read access granted?
   - Security settings correct?

---

### 4.4 Integration Diagnostic Commands

```bash
# OPC-UA diagnostics
greenlang-cli diagnose opcua \
  --server opc.tcp://dcs:4840 \
  --browse \
  --test-read Temperature.PV

# MQTT diagnostics
greenlang-cli diagnose mqtt \
  --broker mqtt://broker:1883 \
  --topic greenlang/test \
  --publish-test

# Kafka diagnostics
greenlang-cli diagnose kafka \
  --bootstrap-servers kafka:9092 \
  --list-topics \
  --test-produce

# All integrations
greenlang-cli diagnose integrations --all
```

---

## 5. Performance Issues

### 5.1 Slow Response Times

#### Symptoms:
- Dashboard loading slowly
- API calls timing out
- Calculations taking longer than usual

#### Diagnosis:

1. **Check system metrics:**
   ```
   Settings > System Status > Performance
   ```

2. **Review key metrics:**
   - CPU usage: Should be < 75%
   - Memory usage: Should be < 80%
   - Disk I/O: No saturation
   - Network: No congestion

3. **Check database performance:**
   ```bash
   greenlang-cli diagnose database --performance
   ```

#### Solutions:

| Cause | Solution |
|-------|----------|
| High CPU | Reduce concurrent agents, scale horizontally |
| High memory | Restart services, increase memory |
| Slow queries | Add indexes, optimize queries |
| Network latency | Check network path, reduce payload |
| Too many connections | Increase pool size, add caching |

### 5.2 High Memory Usage

#### Symptoms:
- Memory usage > 80%
- OOM (Out of Memory) errors
- Service restarts

#### Solutions:

1. **Identify memory consumers:**
   ```bash
   greenlang-cli diagnose memory --top 10
   ```

2. **Common fixes:**
   - Reduce batch sizes
   - Decrease history retention
   - Add more memory
   - Restart services to clear leaks

3. **Configuration adjustments:**
   ```yaml
   performance:
     max_memory_mb: 4096
     gc_interval_s: 60
     history_buffer_size: 1000
   ```

### 5.3 Database Performance

#### Symptoms:
- Slow queries in logs
- Connection pool exhaustion
- Timeout errors

#### Diagnosis:
```bash
greenlang-cli diagnose database \
  --slow-queries \
  --table-sizes \
  --connections
```

#### Solutions:

1. **Add indexes for common queries:**
   ```sql
   CREATE INDEX idx_provenance_agent_time
   ON provenance_records (agent_id, timestamp DESC);
   ```

2. **Archive old data:**
   ```bash
   greenlang-cli database archive \
     --older-than 90d \
     --tables provenance_records,audit_log
   ```

3. **Optimize PostgreSQL:**
   - Run VACUUM ANALYZE regularly
   - Tune shared_buffers
   - Adjust work_mem

---

## 6. Calculation Errors

### 6.1 Efficiency Calculation Issues

#### Problem: Efficiency > 100% or < 0%

**Cause:** Invalid input data or calculation error

**Diagnosis:**
1. Review input values for reasonableness
2. Check for sensor errors
3. Verify unit conversions

**Solutions:**
- Correct sensor calibration
- Fix unit conversion errors
- Add validation bounds

---

#### Problem: Efficiency Doesn't Match Expected

**Cause:** Different calculation method or input assumptions

**Diagnosis:**
1. Verify calculation standard (ASME PTC 4.1)
2. Compare input values
3. Check loss factors

**Solutions:**
- Ensure same calculation basis
- Verify all inputs aligned
- Compare intermediate values

---

### 6.2 Emissions Calculation Issues

#### Problem: Emission Factor Not Found

**Error:** `Emission factor not found for fuel type: [TYPE]`

**Solutions:**
1. Verify fuel type spelling matches database
2. Add custom emission factor if needed
3. Use generic fuel type

---

#### Problem: Calculated vs Measured Discrepancy

**Cause:** Calculation assumptions don't match actual conditions

**Diagnosis:**
1. Compare calculation inputs to actual
2. Verify emission factors current
3. Check for process changes

**Solutions:**
- Calibrate with measured data
- Update emission factors
- Adjust calculation parameters

---

### 6.3 Calculation Validation

All calculations should pass these checks:

| Check | Valid Range | Action if Failed |
|-------|-------------|------------------|
| Mass balance | +/- 5% | Review inputs |
| Energy balance | +/- 3% | Check losses |
| Efficiency | 50-99% | Verify inputs |
| Emissions | Positive | Check calculations |

---

## 7. Authentication and Access Issues

### 7.1 Login Problems

#### Problem: "Invalid credentials"

**Solutions:**
1. Verify username (case-sensitive)
2. Reset password if forgotten
3. Check account not locked
4. Verify account is active

---

#### Problem: "Account locked"

**Cause:** Too many failed login attempts

**Solutions:**
1. Wait 15 minutes (auto-unlock)
2. Contact administrator for manual unlock
3. Reset password after unlock

---

#### Problem: MFA code rejected

**Solutions:**
1. Check time sync on device
2. Verify using correct account
3. Wait for new code (30 seconds)
4. Reset MFA if persistent

---

### 7.2 Permission Issues

#### Problem: "Access denied"

**Cause:** User lacks required permission

**Solutions:**
1. Verify role assignment
2. Request additional permissions
3. Contact administrator

---

#### Problem: Cannot see certain equipment

**Cause:** Plant/equipment access restrictions

**Solutions:**
1. Verify plant assignment
2. Request access to additional plants
3. Check equipment visibility settings

---

### 7.3 Session Issues

#### Problem: Unexpected logout

**Cause:** Session expired or terminated

**Solutions:**
1. Log in again
2. Check session timeout settings
3. Verify no concurrent session limit

---

## 8. Escalation Procedures

### 8.1 Support Tiers

| Tier | Handled By | Response Time | Issues |
|------|------------|---------------|--------|
| **Tier 1** | Operators | Immediate | Basic troubleshooting |
| **Tier 2** | Engineers/IT | < 1 hour | Complex issues |
| **Tier 3** | GreenLang Support | < 4 hours | Product issues |
| **Critical** | On-call team | < 15 min | Production down |

### 8.2 Escalation Criteria

**Escalate to Tier 2 when:**
- Issue persists after basic troubleshooting
- Root cause unclear
- Configuration change needed
- Multiple systems affected

**Escalate to Tier 3 when:**
- Suspected product bug
- Issue not in documentation
- Tier 2 unable to resolve
- Need vendor assistance

**Escalate to Critical when:**
- Production impact
- Safety concern
- Regulatory compliance at risk
- Data loss potential

### 8.3 Information to Include in Escalation

Always include:

```
+------------------------------------------------------------------+
|  ESCALATION REPORT                                                |
+------------------------------------------------------------------+
|  Date/Time: _______________                                       |
|  Reported By: _______________                                     |
|  System: GreenLang Process Heat                                  |
|                                                                    |
|  ISSUE DESCRIPTION:                                               |
|  _______________________________________________________________  |
|                                                                    |
|  ERROR MESSAGES:                                                  |
|  _______________________________________________________________  |
|                                                                    |
|  STEPS TO REPRODUCE:                                              |
|  1. _______________                                               |
|  2. _______________                                               |
|  3. _______________                                               |
|                                                                    |
|  TROUBLESHOOTING ATTEMPTED:                                       |
|  _______________________________________________________________  |
|                                                                    |
|  IMPACT:                                                          |
|  [ ] Production affected                                          |
|  [ ] Safety concern                                               |
|  [ ] Compliance risk                                              |
|  [ ] Multiple users affected                                      |
|                                                                    |
|  ATTACHMENTS:                                                     |
|  [ ] Screenshots                                                  |
|  [ ] Log files                                                    |
|  [ ] Configuration                                                |
+------------------------------------------------------------------+
```

### 8.4 Contact Information

| Contact | Method | Availability |
|---------|--------|--------------|
| Help Desk | helpdesk@company.com | 24/7 |
| IT Support | x1234 | Business hours |
| Engineering | engineering@company.com | Business hours |
| GreenLang Support | support@greenlang.io | 24/7 |
| Emergency Hotline | 1-800-XXX-XXXX | 24/7 |

### 8.5 Collecting Diagnostic Information

Before escalating, collect:

```bash
# Generate diagnostic bundle
greenlang-cli diagnose --all --output diagnostic-bundle.zip

# Contents:
# - System status
# - Agent health
# - Recent logs (last 24 hours)
# - Configuration (secrets redacted)
# - Performance metrics
# - Error summary
```

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-06 | GL-TechWriter | Initial release |

---

*For additional support, contact GreenLang Support at support@greenlang.io*
