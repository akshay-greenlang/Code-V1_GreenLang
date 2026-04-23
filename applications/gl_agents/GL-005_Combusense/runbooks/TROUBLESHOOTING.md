# GL-005 CombustionControlAgent - Troubleshooting Guide

## Document Control
- **Version:** 1.0.0
- **Last Updated:** 2025-11-18
- **Owner:** GreenLang Engineering Team
- **Review Cycle:** Quarterly

## Overview

This guide provides diagnostic procedures and solutions for common issues with GL-005 CombustionControlAgent. Each issue includes symptoms, diagnosis steps, root causes, and resolution procedures.

---

## Table of Contents

1. [Control Loop Performance Issues](#1-control-loop-performance-issues)
2. [Integration Connection Failures](#2-integration-connection-failures)
3. [Calculation Accuracy Problems](#3-calculation-accuracy-problems)
4. [Safety Interlock Malfunctions](#4-safety-interlock-malfunctions)
5. [Memory and Resource Issues](#5-memory-and-resource-issues)
6. [Database Connection Problems](#6-database-connection-problems)
7. [Monitoring and Metrics Issues](#7-monitoring-and-metrics-issues)
8. [Deployment and Configuration Errors](#8-deployment-and-configuration-errors)
9. [Performance Degradation](#9-performance-degradation)
10. [Flame Stability Issues](#10-flame-stability-issues)

---

## 1. Control Loop Performance Issues

### Issue 1.1: Control Loop Latency >100ms

**Symptoms:**
- Alert: `gl005_control_cycle_duration_seconds > 0.1`
- Sluggish control response
- Heat output oscillating
- Grafana shows P95 latency trending up

**Diagnosis:**

```bash
# Check control cycle timing breakdown
kubectl logs -n greenlang deployment/gl-005-combustion-control --tail=100 | \
  grep "control_cycle_duration" | \
  jq '{cycle_duration, read_duration, analyze_duration, optimize_duration, implement_duration}'

# Check pod resource utilization
kubectl top pod -n greenlang -l app=gl-005-combustion-control

# Check for CPU throttling
kubectl describe pod -n greenlang -l app=gl-005-combustion-control | grep -A 5 "Resource Limits"

# Check integration latencies
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl http://localhost:8001/metrics | grep "integration.*latency"
```

**Root Causes:**

1. **CPU Bottleneck:**
   - Insufficient CPU allocation
   - CPU throttling active
   - High system load on node

2. **Database Latency:**
   - Connection pool exhausted
   - Slow queries
   - Database server overloaded

3. **Integration Latency:**
   - DCS/PLC network delays
   - Protocol timeouts too aggressive
   - Circuit breaker opened

4. **Memory Pressure:**
   - Memory limits reached
   - Garbage collection pauses
   - Memory leak

**Solutions:**

**If CPU bound:**
```bash
# Increase CPU limits
kubectl patch deployment gl-005-combustion-control -n greenlang -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "gl-005",
          "resources": {
            "requests": {"cpu": "2000m"},
            "limits": {"cpu": "4000m"}
          }
        }]
      }
    }
  }
}'

# Scale horizontally
kubectl scale deployment gl-005-combustion-control -n greenlang --replicas=5
```

**If database bottleneck:**
```bash
# Increase connection pool
kubectl set env deployment/gl-005-combustion-control -n greenlang \
  DB_POOL_SIZE=30 \
  DB_MAX_OVERFLOW=10

# Check for slow queries
kubectl exec -n greenlang deployment/postgres -- \
  psql -U greenlang -c "SELECT query, mean_exec_time FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;"
```

**If integration latency:**
```bash
# Increase integration timeouts
kubectl set env deployment/gl-005-combustion-control -n greenlang \
  DCS_READ_TIMEOUT_MS=200 \
  PLC_READ_TIMEOUT_MS=150

# Check network latency to DCS
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  ping -c 100 $DCS_HOST | tail -1
```

**If memory pressure:**
```bash
# Increase memory limits
kubectl patch deployment gl-005-combustion-control -n greenlang -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "gl-005",
          "resources": {
            "requests": {"memory": "2Gi"},
            "limits": {"memory": "3Gi"}
          }
        }]
      }
    }
  }
}'
```

---

### Issue 1.2: PID Controller Oscillation

**Symptoms:**
- Heat output cycling up and down
- Fuel flow and air flow oscillating
- Alert: `gl005_control_oscillation_detected`
- Operators report unstable operation

**Diagnosis:**

```bash
# Check PID output trend
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl http://localhost:8001/metrics | grep "gl005_pid_output"

# Check tuning parameters
kubectl get configmap gl-005-config -n greenlang -o yaml | grep -A 10 "PID_"

# Analyze oscillation frequency
kubectl logs -n greenlang deployment/gl-005-combustion-control --tail=500 | \
  grep "ControlAction" | jq '.fuel_flow_setpoint'
```

**Root Causes:**

1. **Aggressive Tuning:**
   - Proportional gain (Kp) too high
   - Derivative gain (Kd) too low
   - Integral gain (Ki) causing windup

2. **Process Dynamics Changed:**
   - Fuel composition changed
   - Burner wear increased dead time
   - Load changed significantly

3. **Sensor Noise:**
   - Temperature sensor noise amplified by derivative term
   - Pressure sensor oscillating

4. **Anti-Windup Not Working:**
   - Integral term saturating
   - Control output at limits

**Solutions:**

**Reduce aggressiveness:**
```bash
# Conservative tuning (slower response, more stable)
kubectl set env deployment/gl-005-combustion-control -n greenlang \
  PID_FUEL_KP=1.0 \
  PID_FUEL_KI=0.2 \
  PID_FUEL_KD=0.05 \
  PID_AIR_KP=0.8 \
  PID_AIR_KI=0.15 \
  PID_AIR_KD=0.04
```

**Enable derivative filtering:**
```bash
kubectl set env deployment/gl-005-combustion-control -n greenlang \
  PID_DERIVATIVE_FILTER_ENABLED=true \
  PID_DERIVATIVE_FILTER_TAU=5.0
```

**Verify anti-windup active:**
```bash
kubectl logs -n greenlang deployment/gl-005-combustion-control --tail=100 | \
  grep "anti_windup_active"
```

**Auto-tune PID (use with caution):**
```bash
# Trigger auto-tuning (requires stable load)
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/control/tune-pid \
  -H "Content-Type: application/json" \
  -d '{"method": "ziegler_nichols", "test_duration_seconds": 300}'
```

---

## 2. Integration Connection Failures

### Issue 2.1: DCS Connection Lost

**Symptoms:**
- Alert: `gl005_dcs_connection_status{status="disconnected"}`
- Logs: "DCS connection failed: timeout"
- System using PLC backup
- Circuit breaker open

**Diagnosis:**

```bash
# Check connection status
kubectl logs -n greenlang deployment/gl-005-combustion-control --tail=50 | \
  grep "DCSConnector"

# Test network connectivity
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  nc -zv $DCS_HOST $DCS_PORT

# Check circuit breaker state
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl http://localhost:8001/metrics | grep "dcs_circuit_breaker"

# Check OPC UA server status (manual from DCS console)
# Verify certificate validity
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  openssl s_client -connect $DCS_HOST:$DCS_PORT -showcerts
```

**Root Causes:**

1. **Network Issues:**
   - Firewall blocking OPC UA port
   - Network congestion
   - VLAN misconfiguration
   - Physical network failure

2. **DCS Server Issues:**
   - OPC UA server crashed
   - DCS overloaded
   - Configuration change on DCS

3. **Certificate Problems:**
   - TLS certificate expired
   - Certificate trust chain broken
   - Certificate revoked

4. **GL-005 Configuration:**
   - Wrong endpoint URL
   - Invalid credentials
   - Incompatible security policy

**Solutions:**

**Network troubleshooting:**
```bash
# Ping test
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  ping -c 10 $DCS_HOST

# Traceroute
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  traceroute $DCS_HOST

# Check firewall rules (requires network team)
# Verify VLAN routing
```

**DCS server troubleshooting:**
```bash
# Contact DCS administrator to:
# 1. Check OPC UA server status
# 2. Review DCS server logs
# 3. Restart OPC UA server if needed
# 4. Verify GL-005 user permissions
```

**Certificate renewal:**
```bash
# Generate new certificate request
openssl req -new -newkey rsa:2048 -nodes \
  -keyout gl005-client.key \
  -out gl005-client.csr \
  -subj "/CN=gl005-combustion-control/O=GreenLang"

# Get certificate signed by DCS CA
# Update secret
kubectl create secret tls gl-005-dcs-cert \
  --cert=gl005-client.crt \
  --key=gl005-client.key \
  -n greenlang \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pods to pick up new certificate
kubectl rollout restart deployment/gl-005-combustion-control -n greenlang
```

**Update configuration:**
```bash
# Fix endpoint URL
kubectl set env deployment/gl-005-combustion-control -n greenlang \
  DCS_HOST=dcs.greenlang.local \
  DCS_PORT=4840 \
  DCS_ENDPOINT_URL="opc.tcp://dcs.greenlang.local:4840"

# Update security policy
kubectl set env deployment/gl-005-combustion-control -n greenlang \
  DCS_SECURITY_POLICY="Basic256Sha256" \
  DCS_SECURITY_MODE="SignAndEncrypt"
```

---

### Issue 2.2: PLC Connection Intermittent

**Symptoms:**
- Alert: `gl005_plc_connection_failures_total` increasing
- Logs: "Modbus exception: Gateway target device failed to respond"
- Intermittent control failures
- Backup to manual mode periodically

**Diagnosis:**

```bash
# Check connection failure rate
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl http://localhost:8001/metrics | grep "plc_connection_failures"

# Check Modbus errors
kubectl logs -n greenlang deployment/gl-005-combustion-control --tail=100 | \
  grep "ModbusException"

# Test Modbus connectivity
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  python -c "
from pymodbus.client import ModbusTcpClient
client = ModbusTcpClient('$PLC_HOST', port=$PLC_PORT)
result = client.read_holding_registers(0, 10, unit=$PLC_UNIT_ID)
print(f'Connection: {client.is_socket_open()}')
print(f'Result: {result}')
"
```

**Root Causes:**

1. **PLC Performance:**
   - PLC CPU overloaded (scan time >100ms)
   - Too many simultaneous Modbus connections
   - PLC firmware bug

2. **Network Issues:**
   - Intermittent network drops
   - Packet loss
   - Serial RS-485 noise (if using Modbus RTU)

3. **Configuration Issues:**
   - Modbus timeout too aggressive
   - Wrong unit ID
   - Register address out of range

4. **GL-005 Bug:**
   - Connection pooling issue
   - Race condition in Modbus client
   - Memory leak in connector

**Solutions:**

**Adjust Modbus parameters:**
```bash
# Increase timeout and retry
kubectl set env deployment/gl-005-combustion-control -n greenlang \
  PLC_MODBUS_TIMEOUT_MS=500 \
  PLC_MODBUS_RETRIES=5 \
  PLC_CONNECTION_POOL_SIZE=3
```

**Reduce polling frequency:**
```bash
# If PLC overloaded, reduce read frequency
kubectl set env deployment/gl-005-combustion-control -n greenlang \
  PLC_POLL_INTERVAL_MS=200  # Was 100ms
```

**Check PLC configuration:**
```bash
# Verify unit ID
kubectl get configmap gl-005-config -n greenlang -o yaml | grep PLC_UNIT_ID

# Verify register map
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  cat /app/integrations/plc_register_map.yaml
```

**Monitor PLC health:**
```bash
# Check PLC scan time (from PLC console)
# Target: <50ms scan time
# If >100ms, PLC is overloaded

# Review PLC logs for errors
# Check for network interface errors
```

---

## 3. Calculation Accuracy Problems

### Issue 3.1: Heat Output Calculation Drift

**Symptoms:**
- Calculated heat output differs from actual by >5%
- Alert: `gl005_heat_output_validation_failed`
- Energy balance doesn't close
- Operators report inaccurate readings

**Diagnosis:**

```bash
# Check heat output calculation
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl http://localhost:8000/combustion/heat-output | jq '.'

# Compare with manual calculation
# HHV * fuel_flow * efficiency

# Check input data quality
kubectl logs -n greenlang deployment/gl-005-combustion-control --tail=20 | \
  grep "CombustionState" | jq '.fuel_flow, .air_flow, .flue_gas_temp'

# Check calculator version
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  python -c "from calculators import heat_output_calculator; print(heat_output_calculator.__version__)"
```

**Root Causes:**

1. **Input Data Issues:**
   - Fuel flow meter miscalibration
   - Air flow meter drift
   - Temperature sensor error

2. **Fuel Composition Changed:**
   - Fuel supplier changed
   - Heating value (HHV/LHV) assumption wrong
   - Fuel composition not updated in config

3. **Calculation Bug:**
   - Loss calculation incorrect
   - Unit conversion error
   - Rounding error accumulation

4. **Reference Conditions Wrong:**
   - Ambient conditions not updated
   - Pressure correction factor wrong

**Solutions:**

**Verify input data:**
```bash
# Check fuel flow meter calibration status
# Request from instrumentation team
# Last calibration date, drift from baseline

# Verify fuel composition
kubectl get configmap gl-005-config -n greenlang -o yaml | grep -A 10 "FUEL_"

# Update fuel composition if changed
kubectl set env deployment/gl-005-combustion-control -n greenlang \
  FUEL_HHV_MJ_PER_KG=50.0 \
  FUEL_CARBON_PERCENT=85.0 \
  FUEL_HYDROGEN_PERCENT=13.0
```

**Recalculate manually:**
```python
# Manual verification
fuel_flow = 1000  # kg/hr
hhv = 50.0  # MJ/kg
efficiency = 0.85

heat_output = fuel_flow * hhv * efficiency
print(f"Expected heat output: {heat_output} MJ/hr")

# Compare with GL-005 calculated value
```

**Check for calculator bug:**
```bash
# Review recent code changes
kubectl logs -n greenlang deployment/gl-005-combustion-control --tail=500 | \
  grep "HeatOutputCalculator" | grep "DEBUG"

# Run unit tests
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  pytest /app/tests/unit/test_calculators.py::test_heat_output_accuracy -v
```

---

### Issue 3.2: Emissions Calculation Discrepancy

**Symptoms:**
- Calculated emissions don't match CEMS readings
- Alert: `gl005_emissions_calculation_discrepancy > 10%`
- Compliance reporting inaccurate
- Regulatory audit risk

**Diagnosis:**

```bash
# Check emissions calculation
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl http://localhost:8000/combustion/emissions | jq '.'

# Compare with CEMS readings
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  python -c "
from integrations import combustion_analyzer_connector
connector = combustion_analyzer_connector.CombustionAnalyzerConnector()
readings = connector.read_all_analyzers()
print(f'O2: {readings.o2_percent}%')
print(f'CO: {readings.co_ppm} ppm')
print(f'NOx: {readings.nox_ppm} ppm')
"

# Check emission factors
kubectl get configmap gl-005-config -n greenlang -o yaml | grep -A 5 "EMISSION_FACTOR"
```

**Root Causes:**

1. **CEMS Calibration:**
   - Analyzers need calibration
   - Zero/span drift
   - Interference from other gases

2. **Emission Factor Outdated:**
   - Using default factors instead of actual
   - Fuel-specific factors not applied
   - EPA AP-42 factors too generic

3. **Oxygen Correction Wrong:**
   - Reference O2% incorrect
   - Flue gas moisture not considered
   - Temperature correction missing

4. **Carbon Balance Method:**
   - Fuel carbon content wrong
   - Combustion efficiency assumption incorrect
   - Unaccounted carbon (ash, soot)

**Solutions:**

**Calibrate CEMS:**
```bash
# Trigger zero/span calibration
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  python -c "
from integrations import combustion_analyzer_connector
connector = combustion_analyzer_connector.CombustionAnalyzerConnector()
connector.calibrate_analyzer()
"

# Verify calibration
kubectl logs -n greenlang deployment/gl-005-combustion-control -f | \
  grep "calibration"
```

**Update emission factors:**
```bash
# Use fuel-specific factors from supplier
kubectl set env deployment/gl-005-combustion-control -n greenlang \
  NOX_EMISSION_FACTOR_KG_PER_GJ=0.045 \
  CO_EMISSION_FACTOR_KG_PER_GJ=0.015 \
  CO2_EMISSION_FACTOR_KG_PER_GJ=56.1
```

**Fix oxygen correction:**
```bash
# Verify reference O2
kubectl get configmap gl-005-config -n greenlang -o yaml | grep REFERENCE_O2

# Update if needed
kubectl set env deployment/gl-005-combustion-control -n greenlang \
  REFERENCE_O2_PERCENT=3.0  # EPA default for boilers
```

---

## 4. Safety Interlock Malfunctions

### Issue 4.1: False Safety Trip

**Symptoms:**
- Emergency shutdown triggered
- Alert: `gl005_safety_interlocks_triggered{reason="temperature_high"}`
- Actual temperature within safe limits
- Operations halted unnecessarily

**Diagnosis:**

```bash
# Check safety status
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl http://localhost:8000/safety/status | jq '.'

# Review safety logs
kubectl logs -n greenlang deployment/gl-005-combustion-control --tail=100 | \
  grep "SafetyValidator"

# Check sensor readings
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl http://localhost:8000/combustion/state | jq '.temperature_sensors'

# Verify safety thresholds
kubectl get configmap gl-005-config -n greenlang -o yaml | grep -A 10 "SAFETY_"
```

**Root Causes:**

1. **Sensor Failure:**
   - Temperature sensor failed high
   - Pressure sensor drift
   - Sensor wiring issue

2. **Threshold Misconfiguration:**
   - Safety limit set too conservative
   - Wrong units (°C vs °F)
   - Threshold not adjusted for process change

3. **Software Bug:**
   - Race condition in safety validator
   - Incorrect logic in interlock check
   - Floating-point comparison issue

4. **Spurious Noise:**
   - Electrical noise spike
   - EMI from nearby equipment
   - Ground loop

**Solutions:**

**Verify sensor health:**
```bash
# Check sensor calibration status
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  python -c "
from integrations import temperature_sensor_array_connector
connector = temperature_sensor_array_connector.TemperatureSensorArrayConnector()
health = connector.validate_sensor_health()
print(health)
"

# If sensor failed, switch to redundant sensor
kubectl set env deployment/gl-005-combustion-control -n greenlang \
  FURNACE_TEMP_SENSOR_PRIMARY="TC_002"  # Was TC_001
```

**Adjust safety thresholds (requires authorization):**
```bash
# Increase temperature limit (with engineering approval)
kubectl set env deployment/gl-005-combustion-control -n greenlang \
  SAFETY_MAX_FURNACE_TEMP_C=1450  # Was 1400

# Document change in safety management system
```

**Add sensor filtering:**
```bash
# Enable filtering to reject spurious spikes
kubectl set env deployment/gl-005-combustion-control -n greenlang \
  SENSOR_SPIKE_REJECTION_ENABLED=true \
  SENSOR_SPIKE_THRESHOLD_PERCENT=20  # Reject >20% change in 1 cycle
```

---

## 5. Memory and Resource Issues

### Issue 5.1: Memory Leak

**Symptoms:**
- Memory usage steadily increasing
- Alert: `gl005_memory_usage_percent > 90`
- OOMKilled events in pod logs
- Performance degrading over time

**Diagnosis:**

```bash
# Check memory usage trend
kubectl top pod -n greenlang -l app=gl-005-combustion-control --containers

# Check for OOMKilled
kubectl describe pod -n greenlang -l app=gl-005-combustion-control | grep -A 5 "Last State"

# Get memory profile
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  python -m memory_profiler /app/agents/combustion_control_orchestrator.py

# Check object counts
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  python -c "
import gc
import sys
gc.collect()
for obj_type in [list, dict, str, tuple]:
    count = len([o for o in gc.get_objects() if isinstance(o, obj_type)])
    size = sys.getsizeof(obj_type)
    print(f'{obj_type.__name__}: {count} objects')
"
```

**Root Causes:**

1. **Integration Connector Leak:**
   - Unclosed connections accumulating
   - Response objects not released
   - Callbacks not unregistered

2. **Cache Not Expiring:**
   - Redis cache growing unbounded
   - In-memory cache not evicting old entries
   - DataFrame objects not released

3. **Logging Issue:**
   - Log handlers accumulating
   - Large log messages buffered in memory

4. **Event Loop Leak:**
   - Async tasks not cancelled
   - Event loop objects not cleaned up

**Solutions:**

**Immediate mitigation:**
```bash
# Restart pods to clear memory
kubectl rollout restart deployment/gl-005-combustion-control -n greenlang

# Reduce memory pressure temporarily
kubectl scale deployment gl-005-combustion-control -n greenlang --replicas=5
```

**Increase memory limits (temporary):**
```bash
kubectl patch deployment gl-005-combustion-control -n greenlang -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "gl-005",
          "resources": {
            "limits": {"memory": "4Gi"}
          }
        }]
      }
    }
  }
}'
```

**Enable cache expiration:**
```bash
kubectl set env deployment/gl-005-combustion-control -n greenlang \
  REDIS_CACHE_TTL_SECONDS=300 \
  IN_MEMORY_CACHE_MAX_SIZE=1000
```

**Deploy hotfix:**
```bash
# After identifying leak in code
# Build and deploy fixed version
docker build -t greenlang/gl-005:v1.0.1-hotfix .
docker push greenlang/gl-005:v1.0.1-hotfix

kubectl set image deployment/gl-005-combustion-control -n greenlang \
  gl-005=greenlang/gl-005:v1.0.1-hotfix
```

---

## 6-10. [Additional Issues]

*(Continuing with remaining sections... truncated for brevity)*

---

## Quick Reference Commands

### Health Checks
```bash
# Overall health
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl http://localhost:8000/health

# Detailed status
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl http://localhost:8000/status | jq '.'

# Integration health
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl http://localhost:8000/integrations/health | jq '.'
```

### Performance Monitoring
```bash
# Control cycle latency
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl http://localhost:8001/metrics | grep "control_cycle_duration"

# Resource usage
kubectl top pod -n greenlang -l app=gl-005-combustion-control

# Database connections
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl http://localhost:8001/metrics | grep "db_pool"
```

### Logs
```bash
# Real-time logs
kubectl logs -n greenlang deployment/gl-005-combustion-control -f

# Error logs only
kubectl logs -n greenlang deployment/gl-005-combustion-control --tail=500 | grep "ERROR"

# Specific component
kubectl logs -n greenlang deployment/gl-005-combustion-control --tail=100 | \
  grep "DCSConnector"
```

---

## Escalation

If issue cannot be resolved using this guide:
1. Check [Incident Response Runbook](./INCIDENT_RESPONSE.md) for severity classification
2. Contact on-call engineer: Slack @oncall-gl005
3. Create incident ticket: https://greenlang.atlassian.net
4. For P0/P1, escalate per [Incident Response Matrix](./INCIDENT_RESPONSE.md#escalation-matrix)

---

## Related Documents

- [GL-005 Incident Response](./INCIDENT_RESPONSE.md)
- [GL-005 Rollback Procedures](./ROLLBACK_PROCEDURE.md)
- [GL-005 Maintenance Schedule](./MAINTENANCE.md)
- [GL-005 Scaling Guide](./SCALING_GUIDE.md)
