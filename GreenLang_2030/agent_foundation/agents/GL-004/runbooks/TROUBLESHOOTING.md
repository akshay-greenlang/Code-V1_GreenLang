# GL-004 BurnerOptimizationAgent - Troubleshooting Guide

## Overview
This guide provides solutions for common issues with GL-004 BurnerOptimizationAgent.

## Common Issues

### 1. Optimization Not Converging

**Symptoms:**
- Iterations exceed 100
- No efficiency improvement
- Oscillating setpoints
- Convergence status = "failed"

**Root Causes:**
- Conflicting objectives (efficiency vs emissions)
- Constraints too tight
- Noisy sensor data
- Rapidly changing load

**Solutions:**
```bash
# Check optimization logs
kubectl logs -n greenlang <pod> | grep "optimization"

# Adjust objective weights in ConfigMap
kubectl edit configmap gl-004-config -n greenlang

# Increase convergence tolerance
CONVERGENCE_TOLERANCE=0.01  # from 0.001
```

**Tuning Steps:**
1. Reduce objective weight conflicts
2. Relax excess air constraints (5-30% → 3-35%)
3. Increase sensor data averaging
4. Reduce optimization frequency during transients

### 2. High Emissions Persisting

**Symptoms:**
- NOx >50 ppm sustained
- CO >100 ppm sustained
- Compliance alerts

**Diagnostics:**
```bash
# Check CEMS status
curl http://gl-004:8000/sensors/emissions

# Verify O2 levels
curl http://gl-004:8000/burner/state | jq '.o2_level'
```

**Solutions:**
- Increase excess air (+2-3%)
- Reduce burner load temporarily
- Check fuel quality
- Verify CEMS calibration
- Inspect burner nozzles

### 3. O2 Reading Unstable

**Symptoms:**
- O2 fluctuating >0.5%
- Data quality score <0.8
- Frequent optimizer corrections

**Root Causes:**
- Analyzer drift
- Sample line leaks
- Probe contamination
- Electrical interference

**Solutions:**
1. Calibrate O2 analyzer
2. Inspect sample system
3. Clean analyzer probe
4. Check signal cables
5. Increase averaging time

### 4. Flame Scanner Issues

**Symptoms:**
- Intermittent flame loss signals
- False flame detection
- Safety interlocks triggering

**Diagnostics:**
- Check scanner signal strength
- Verify optical path clarity
- Test flame rod (if applicable)
- Review scanner logs

**Solutions:**
- Clean scanner lens/probe
- Adjust scanner positioning
- Replace if signal <50%
- Check UV/IR sensor health

### 5. Control Loop Oscillation

**Symptoms:**
- Fuel/air flow cycling
- Temperature oscillating
- Burner load unstable

**PID Tuning:**
```python
# Reduce proportional gain
FUEL_CONTROL_KP=0.5  # from 1.0

# Increase integral time
FUEL_CONTROL_KI=0.05  # from 0.1

# Add derivative carefully
FUEL_CONTROL_KD=0.02  # from 0.05
```

### 6. Integration Failures

**Symptoms:**
- Modbus connection errors
- OPC UA timeouts
- Sensor read failures

**Network Diagnostics:**
```bash
# Test Modbus connectivity
modpoll -m tcp -a 1 -r 0 -c 10 <host> 502

# Check OPC UA endpoint
opcua-client --endpoint opc.tcp://<host>:4840

# Verify network
ping <burner-controller-host>
```

**Solutions:**
- Restart integration services
- Check firewall rules
- Verify IP addresses
- Update device firmware
- Increase timeout values

### 7. Database Connection Issues

**Symptoms:**
- "Connection pool exhausted"
- Slow API responses
- Transaction timeouts

**Solutions:**
```bash
# Check pool status
kubectl exec -n greenlang <pod> -- curl localhost:8000/status | jq '.database'

# Increase pool size
DB_POOL_SIZE=20  # from 10
DB_MAX_OVERFLOW=40  # from 20

# Restart pods
kubectl rollout restart deployment/gl-004 -n greenlang
```

### 8. Memory Leaks

**Symptoms:**
- Memory usage growing
- OOMKilled pods
- Slow performance over time

**Investigation:**
```bash
# Check memory usage
kubectl top pods -n greenlang -l app=gl-004

# Review memory metrics
curl http://gl-004:8001/metrics | grep memory
```

**Solutions:**
- Reduce cache TTL
- Clear optimization history
- Increase memory limits
- Investigate with profiler

### 9. Performance Degradation

**Symptoms:**
- API latency >500ms
- Optimization cycles >90s
- High CPU usage

**Diagnostics:**
```bash
# Check Prometheus metrics
curl http://gl-004:8001/metrics | grep latency

# Review slow queries
kubectl logs -n greenlang <pod> | grep "slow query"
```

**Solutions:**
- Scale horizontally (add replicas)
- Optimize database queries
- Add indexes
- Review algorithm efficiency
- Check for infinite loops

### 10. Data Quality Issues

**Symptoms:**
- Validation failures
- Sensor readings out of range
- Inconsistent measurements

**Solutions:**
1. Compare with backup sensors
2. Review calibration dates
3. Check sensor wiring
4. Verify power supply
5. Temporarily disable outlier rejection

## Emergency Procedures

### Emergency Stop
```bash
# Stop optimization immediately
kubectl exec -n greenlang <pod> -- curl -X POST http://localhost:8000/stop

# Revert to manual control
kubectl scale deployment/gl-004 --replicas=0 -n greenlang
```

### Safe Mode Operation
```bash
# Disable automatic optimization
OPTIMIZATION_INTERVAL_SECONDS=0

# Enable manual approval mode
MANUAL_APPROVAL_REQUIRED=true
```

## Escalation

If issues persist after troubleshooting:
1. **P0/P1**: Contact greenlang-oncall@example.com
2. **Burner Safety**: safety-team@example.com
3. **Infrastructure**: devops-oncall@example.com
4. **Vendor Support**: burner-manufacturer-support@

## Related Runbooks
- INCIDENT_RESPONSE.md
- ROLLBACK_PROCEDURE.md
- MAINTENANCE.md
