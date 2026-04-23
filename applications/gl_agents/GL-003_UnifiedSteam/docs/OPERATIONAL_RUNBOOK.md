# GL-003 UnifiedSteam Operational Runbook

**Version:** 1.0.0
**Last Updated:** 2024-12-27
**Agent:** GL-003 UnifiedSteam (Steam Aggregator)

---

## Table of Contents

1. [Overview](#overview)
2. [Startup Procedures](#startup-procedures)
3. [Common Operational Scenarios](#common-operational-scenarios)
4. [Troubleshooting Guide](#troubleshooting-guide)
5. [Alert Response Procedures](#alert-response-procedures)
6. [Maintenance Procedures](#maintenance-procedures)
7. [Emergency Procedures](#emergency-procedures)

---

## Overview

GL-003 UnifiedSteam is the steam system aggregator agent responsible for:
- Steam property calculations (IAPWS-IF97 compliant)
- Mass and energy balance monitoring
- Desuperheater optimization
- Condensate recovery optimization
- Steam trap diagnostics
- Multi-header network optimization

### Agent Dependencies

| Dependency | Description | Health Check |
|------------|-------------|--------------|
| OPC-UA Server | Real-time data source | `/health/opcua` |
| Historian | Historical data storage | `/health/historian` |
| Database | Configuration and audit | `/health/db` |
| Redis | Caching layer | `/health/cache` |

---

## Startup Procedures

### Pre-flight Checks

Before starting GL-003, verify the following:

```bash
# 1. Check system health
gl doctor

# 2. Verify configuration
gl config validate --file config/production.yaml

# 3. Test database connectivity
gl db ping

# 4. Verify OPC-UA connection
gl integration test opcua

# 5. Check historian connectivity
gl integration test historian
```

### Configuration Verification

1. **Steam Header Configuration**
   ```yaml
   headers:
     HP:
       design_pressure_kpa: 4000
       normal_pressure_kpa: 3800
       low_alarm_kpa: 3500
       high_alarm_kpa: 4200
     MP:
       design_pressure_kpa: 1000
       normal_pressure_kpa: 950
       low_alarm_kpa: 800
       high_alarm_kpa: 1100
     LP:
       design_pressure_kpa: 400
       normal_pressure_kpa: 380
       low_alarm_kpa: 300
       high_alarm_kpa: 450
   ```

2. **Balance Tolerance Settings**
   ```yaml
   balance:
     mass_tolerance_percent: 2.0
     energy_tolerance_percent: 3.0
     update_interval_seconds: 60
   ```

### Startup Sequence

```bash
# Start in development/test mode
gl start --mode development

# Start in production mode
gl start --mode production --config config/production.yaml

# Verify startup
gl status
```

Expected output:
```
GL-003 UnifiedSteam Status: RUNNING
Version: 1.0.0
Uptime: 0h 0m 15s
Steam Balance: CLOSED (0.3% imbalance)
Active Headers: HP, MP, LP
Last Calculation: 2024-12-27T10:00:00Z
```

---

## Common Operational Scenarios

### Steam Balance Monitoring

**Scenario:** Monitor real-time steam balance across the facility.

```bash
# View current balance
gl balance show

# View historical balance (last 24 hours)
gl balance history --hours 24

# Export balance data
gl balance export --format csv --output balance_report.csv
```

**Interpretation:**
- Mass imbalance < 2%: Normal operation
- Mass imbalance 2-5%: Investigation recommended
- Mass imbalance > 5%: Immediate investigation required

### Desuperheater Optimization

**Scenario:** Optimize spray water flow for target outlet temperature.

```bash
# View current desuperheater status
gl desuper status DESUPER-001

# Calculate optimal spray rate
gl desuper optimize DESUPER-001 \
  --inlet-temp 450 \
  --target-temp 350 \
  --inlet-flow 10.0

# Apply recommendation
gl desuper apply DESUPER-001 --spray-rate 1.47
```

**Best Practices:**
- Maintain minimum 10°C superheat at outlet
- Never exceed maximum spray water flow
- Monitor for water hammer (acoustic alerts)

### Condensate Recovery Optimization

**Scenario:** Optimize condensate return for maximum energy recovery.

```bash
# View recovery status
gl condensate status

# Analyze optimization opportunities
gl condensate analyze \
  --current-recovery 65 \
  --target-recovery 85

# View flash steam recovery potential
gl condensate flash-analysis
```

**Key Metrics:**
- Recovery rate target: >85%
- Flash steam recovery target: >50%
- Condensate return temperature: >80°C

### Steam Trap Diagnostics

**Scenario:** Perform steam trap survey and prioritize repairs.

```bash
# Run diagnostic on specific trap
gl trap diagnose TRAP-001

# Batch survey all traps
gl trap survey --area "Building A"

# Generate repair priority list
gl trap priorities --min-loss-kw 10

# Export survey results
gl trap export --format pdf --output trap_survey.pdf
```

**Trap Condition Codes:**
| Code | Condition | Action |
|------|-----------|--------|
| GOOD | Operating normally | Continue monitoring |
| LEAKING | Passing steam | Repair within 2 weeks |
| BLOCKED | Not passing condensate | Repair within 1 week |
| BLOW_THROUGH | Fully open | Immediate repair |

---

## Troubleshooting Guide

### Balance Doesn't Close (Mass/Energy Imbalance)

**Symptoms:**
- Mass imbalance > 2%
- Energy imbalance > 3%
- Balance status shows "OPEN"

**Root Cause Analysis:**

1. **Check for metering errors**
   ```bash
   gl balance diagnose --check meters
   ```
   - Compare flowmeter readings to expected values
   - Check last calibration dates
   - Verify signal quality

2. **Check for steam leaks**
   ```bash
   gl leaks survey --infrared
   gl leaks estimate --from-balance
   ```

3. **Check for unmetered loads**
   ```bash
   gl balance unmetered --estimate
   ```

4. **Verify condensate return**
   ```bash
   gl condensate check-return
   ```

**Resolution Steps:**
1. Identify largest discrepancy source
2. Dispatch field verification
3. Update configuration if permanent change
4. Re-run balance calculation

### Steam Quality Issues (Wet Steam)

**Symptoms:**
- Quality reading < 95%
- Water hammer reports
- Process heat transfer degradation

**Diagnosis:**
```bash
# Check quality at each header
gl quality check --all-headers

# Analyze separator performance
gl separator status

# Check header drainage
gl drainage status
```

**Common Causes:**
1. Boiler carryover (check drum level)
2. Inadequate separation
3. Failed drip leg traps
4. Excessive pressure drops

### Trap Failure Detection Issues

**Symptoms:**
- False positive/negative trap diagnoses
- Inconsistent readings

**Diagnosis:**
```bash
# Check sensor status
gl trap sensor-check TRAP-001

# View raw sensor data
gl trap raw-data TRAP-001 --last 1h

# Recalibrate baseline
gl trap calibrate TRAP-001
```

### Integration Errors (OPC-UA/Historian)

**Symptoms:**
- Missing data points
- Stale timestamps
- Connection errors

**Diagnosis:**
```bash
# Check OPC-UA status
gl integration status opcua

# Test specific tags
gl integration test-tag "STEAM.HP.FLOW"

# View connection history
gl integration log opcua --last 1h
```

**Resolution:**
1. Check network connectivity
2. Verify tag paths in configuration
3. Restart OPC-UA adapter if needed
4. Check historian disk space

---

## Alert Response Procedures

### Critical Pressure Alerts

**Alert:** `PRESSURE_HIGH_HIGH` or `PRESSURE_LOW_LOW`

**Immediate Actions:**
1. Verify reading with field instrumentation
2. Check relief valve status
3. Notify operations supervisor
4. Document in incident log

```bash
# Acknowledge alert
gl alert ack ALERT-12345 --reason "Field verification in progress"

# View pressure history
gl header HP pressure-history --last 1h
```

### Energy Imbalance Alerts

**Alert:** `ENERGY_IMBALANCE_CRITICAL` (>10%)

**Response:**
1. Review current balance calculations
2. Check for major equipment trips
3. Verify boiler outputs
4. Check for large load changes

```bash
# View imbalance breakdown
gl balance breakdown

# Compare to previous hour
gl balance compare --hours 1
```

### Trap Failure Alerts

**Alert:** `TRAP_BLOW_THROUGH` or `TRAP_BLOCKED`

**Response:**
1. Document trap location and condition
2. Estimate steam/energy loss
3. Schedule repair based on severity
4. Bypass if safety concern

```bash
# View trap details
gl trap info TRAP-001

# Log work order
gl trap work-order TRAP-001 --priority HIGH
```

### Safety Interlock Activations

**Alert:** `SAFETY_INTERLOCK_TRIP`

**Response:**
1. DO NOT attempt to override
2. Identify trip cause from DCS
3. Notify safety coordinator
4. Document fully before reset

```bash
# View interlock status
gl safety interlock-status

# View trip history
gl safety trip-log --last 24h
```

---

## Maintenance Procedures

### Scheduled Calibration Checks

**Frequency:** Monthly

```bash
# Generate calibration schedule
gl maintenance calibration-due

# Record calibration results
gl maintenance calibration-record \
  --meter FLOW-001 \
  --as-found 100.2 \
  --as-left 100.0 \
  --date 2024-12-27
```

**Key Instruments:**
- Steam flowmeters: Monthly
- Pressure transmitters: Quarterly
- Temperature elements: Quarterly
- Level transmitters: Monthly

### Steam Trap Survey Procedures

**Frequency:** Quarterly (minimum)

**Survey Procedure:**
1. Download current trap list
   ```bash
   gl trap list --export trap_list.csv
   ```

2. Perform field survey
   - Ultrasonic testing
   - Temperature differential
   - Visual inspection

3. Upload results
   ```bash
   gl trap import-survey survey_results.csv
   ```

4. Generate repair priority
   ```bash
   gl trap priorities --generate-work-orders
   ```

### Performance Benchmark Verification

**Frequency:** Monthly

```bash
# Run benchmark tests
gl benchmark run --full

# Compare to baseline
gl benchmark compare --baseline 2024-01

# Generate performance report
gl benchmark report --output monthly_performance.pdf
```

**Key Benchmarks:**
- Steam property calculation time: <5ms
- Balance calculation time: <100ms
- API response time: <200ms
- Trap diagnosis accuracy: >95%

---

## Emergency Procedures

### Complete Data Source Failure

```bash
# Switch to backup data source
gl failover activate --source backup-historian

# Enable degraded mode (cached calculations)
gl mode degraded --enable

# Notify operations
gl notify operations "Data source failure - running degraded"
```

### Agent Process Failure

```bash
# Check process status
gl status --verbose

# Restart agent
gl restart --graceful

# If graceful fails
gl restart --force

# Check logs
gl logs --last 100 --level ERROR
```

### Database Corruption

```bash
# Stop agent
gl stop

# Backup current state
gl db backup --emergency

# Restore from last good backup
gl db restore --backup 2024-12-26

# Restart agent
gl start --mode recovery
```

---

## Contact Information

| Role | Contact | Escalation |
|------|---------|------------|
| Operations | ops@facility.com | Primary |
| Engineering | eng@facility.com | Secondary |
| IT Support | it@facility.com | Technical |
| Vendor Support | support@greenlang.io | External |

---

*This runbook is maintained by the GL-003 UnifiedSteam team. For updates, submit a pull request to the documentation repository.*
