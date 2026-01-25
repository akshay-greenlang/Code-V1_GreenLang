# GL-013 PREDICTMAINT - Troubleshooting Guide

```
================================================================================
                    TROUBLESHOOTING GUIDE - GL-013 PREDICTMAINT
                         Diagnostic and Resolution Procedures
================================================================================
```

**Version:** 1.0.0
**Last Updated:** 2024-12-01
**Owner:** Site Reliability Engineering (SRE) Team
**On-Call Use:** Primary

---

## Table of Contents

1. [Overview](#overview)
2. [Diagnostic Commands](#diagnostic-commands)
3. [Common Issues](#common-issues)
4. [Calculator Issues](#calculator-issues)
5. [Integration Issues](#integration-issues)
6. [Performance Issues](#performance-issues)
7. [Data Issues](#data-issues)

---

## Overview

This guide provides troubleshooting procedures for common issues with GL-013 PREDICTMAINT. Use this guide alongside the Incident Response runbook.

### Before You Start

1. Check the [GL-013 Status Page](https://status.greenlang.io/gl-013)
2. Review recent [deployments](https://argocd.greenlang.io/gl-013)
3. Check for [known issues](https://jira.greenlang.io/issues/?jql=project=GL013+AND+type=Bug+AND+status=Open)

---

## Diagnostic Commands

### Health Check Commands

```bash
# Basic health check
curl -s http://localhost:8000/health | jq .

# Detailed health check
curl -s http://localhost:8000/health/detailed | jq .

# Readiness check
curl -s http://localhost:8000/health/ready | jq .

# Expected healthy response:
# {
#   "status": "healthy",
#   "version": "1.0.0",
#   "components": {
#     "database": "healthy",
#     "cache": "healthy",
#     "calculators": "healthy"
#   }
# }
```

### Kubernetes Commands

```bash
# Pod status
kubectl get pods -n greenlang -l app=gl-013-predictmaint -o wide

# Pod details
kubectl describe pod -n greenlang -l app=gl-013-predictmaint

# Pod logs (last 100 lines)
kubectl logs -n greenlang -l app=gl-013-predictmaint --tail=100

# Pod logs (follow)
kubectl logs -n greenlang -l app=gl-013-predictmaint -f

# Pod resource usage
kubectl top pods -n greenlang -l app=gl-013-predictmaint

# Recent events
kubectl get events -n greenlang --sort-by='.lastTimestamp' | grep gl-013
```

### Metrics Commands

```bash
# Prometheus metrics
curl -s http://localhost:9090/metrics | head -100

# Prediction metrics
curl -s http://localhost:9090/metrics | grep "predictmaint_predictions"

# Error metrics
curl -s http://localhost:9090/metrics | grep "predictmaint_errors"

# Latency metrics
curl -s http://localhost:9090/metrics | grep "predictmaint_latency"

# Cache metrics
curl -s http://localhost:9090/metrics | grep "predictmaint_cache"
```

### Database Commands

```bash
# Test PostgreSQL connection
psql -h postgres.greenlang.io -U readonly -d predictmaint -c "SELECT 1;"

# Check active connections
psql -h postgres.greenlang.io -U readonly -d predictmaint -c "SELECT count(*) FROM pg_stat_activity;"

# Check table sizes
psql -h postgres.greenlang.io -U readonly -d predictmaint -c "SELECT relname, pg_size_pretty(pg_total_relation_size(relid)) FROM pg_catalog.pg_statio_user_tables ORDER BY pg_total_relation_size(relid) DESC LIMIT 10;"
```

### Redis Commands

```bash
# Test Redis connection
redis-cli -h redis.greenlang.io ping

# Check memory usage
redis-cli -h redis.greenlang.io info memory

# Check key count
redis-cli -h redis.greenlang.io dbsize

# Check slow log
redis-cli -h redis.greenlang.io slowlog get 10
```

---

## Common Issues

### Issue: Pods Not Starting

**Symptoms:**
- Pods in Pending or CrashLoopBackOff state
- Health checks failing

**Diagnosis:**

```bash
# Check pod status
kubectl get pods -n greenlang -l app=gl-013-predictmaint

# Check pod events
kubectl describe pod -n greenlang -l app=gl-013-predictmaint | grep -A 20 Events

# Check init container logs
kubectl logs -n greenlang -l app=gl-013-predictmaint -c wait-for-db
kubectl logs -n greenlang -l app=gl-013-predictmaint -c wait-for-redis
```

**Common Causes and Fixes:**

| Cause | Solution |
|-------|----------|
| Image pull failure | Check image name, registry credentials |
| Resource limits exceeded | Increase limits or scale down |
| Database not ready | Check PostgreSQL connectivity |
| Redis not ready | Check Redis connectivity |
| ConfigMap missing | Apply ConfigMap manifest |
| Secret missing | Apply Secret manifest |

**Fix: Restart Deployment**
```bash
kubectl rollout restart deployment/gl-013-predictmaint -n greenlang
kubectl rollout status deployment/gl-013-predictmaint -n greenlang
```

### Issue: API Returning 5xx Errors

**Symptoms:**
- HTTP 500/502/503 responses
- Error rate spike in metrics

**Diagnosis:**

```bash
# Check error logs
kubectl logs -n greenlang -l app=gl-013-predictmaint --tail=200 | grep -i error

# Check error metrics
curl -s http://localhost:9090/metrics | grep "predictmaint_errors_total"

# Test specific endpoint
curl -v http://localhost:8000/api/v1/health
```

**Common Causes and Fixes:**

| Cause | Solution |
|-------|----------|
| Database connection failed | Check DB credentials, connectivity |
| Out of memory | Increase memory limits, scale out |
| Unhandled exception | Check logs, report bug |
| Configuration error | Verify ConfigMap values |

### Issue: Authentication Failures

**Symptoms:**
- HTTP 401/403 responses
- "Token expired" or "Invalid token" errors

**Diagnosis:**

```bash
# Check auth logs
kubectl logs -n greenlang -l app=gl-013-predictmaint --tail=100 | grep -i "auth\|token"

# Verify Keycloak connectivity
curl -v https://keycloak.greenlang.io/auth/realms/greenlang/.well-known/openid-configuration

# Check JWT secret
kubectl get secret gl-013-predictmaint-secrets -n greenlang -o jsonpath='{.data.JWT_SECRET}' | base64 -d
```

**Common Causes and Fixes:**

| Cause | Solution |
|-------|----------|
| Keycloak down | Check Keycloak service |
| Clock skew | Sync NTP on nodes |
| JWT secret mismatch | Rotate and redeploy secrets |
| Token expired | Request new token |

---

## Calculator Issues

### Issue: RUL Predictions Failing

**Symptoms:**
- RUL endpoint returning errors
- "Invalid equipment type" errors

**Diagnosis:**

```bash
# Check calculator logs
kubectl logs -n greenlang -l app=gl-013-predictmaint --tail=100 | grep -i "rul\|weibull"

# Test RUL calculation
curl -X POST http://localhost:8000/api/v1/calculate/rul \
  -H "Content-Type: application/json" \
  -d '{"equipment_type": "motor_ac_induction_large", "operating_hours": 50000}'
```

**Common Causes and Fixes:**

| Cause | Solution |
|-------|----------|
| Invalid equipment type | Use supported equipment types |
| Missing Weibull parameters | Add parameters to constants.py |
| Negative operating hours | Validate input data |
| Decimal precision error | Check precision settings |

### Issue: Vibration Analysis Errors

**Symptoms:**
- ISO 10816 zone returning "UNKNOWN"
- FFT analysis failing

**Diagnosis:**

```bash
# Check vibration logs
kubectl logs -n greenlang -l app=gl-013-predictmaint --tail=100 | grep -i "vibration\|iso"

# Test vibration analysis
curl -X POST http://localhost:8000/api/v1/analyze/vibration \
  -H "Content-Type: application/json" \
  -d '{"velocity_rms_mm_s": 4.2, "machine_class": "CLASS_II"}'
```

**Common Causes and Fixes:**

| Cause | Solution |
|-------|----------|
| Invalid machine class | Use CLASS_I, CLASS_II, CLASS_III, or CLASS_IV |
| Velocity out of range | Check sensor data validity |
| Missing spectrum data | Provide frequency/amplitude arrays |

### Issue: Anomaly Detection Not Working

**Symptoms:**
- Anomalies not detected
- False positive alerts

**Diagnosis:**

```bash
# Check anomaly detector logs
kubectl logs -n greenlang -l app=gl-013-predictmaint --tail=100 | grep -i "anomaly"

# Check detector configuration
kubectl get configmap gl-013-predictmaint-config -n greenlang -o yaml | grep -A 10 anomaly
```

**Common Causes and Fixes:**

| Cause | Solution |
|-------|----------|
| Threshold too high | Adjust z-score threshold (default: 3) |
| Insufficient historical data | Accumulate more baseline data |
| Configuration mismatch | Verify detector settings |

---

## Integration Issues

### Issue: SAP PM Connection Failed

**Symptoms:**
- Work orders not syncing
- "Connection refused" or "401 Unauthorized" errors

**Diagnosis:**

```bash
# Check integration logs
kubectl logs -n greenlang -l app=gl-013-predictmaint --tail=100 | grep -i sap

# Test SAP connectivity
curl -v https://sap.company.com/api/health

# Check credentials
kubectl get secret gl-013-predictmaint-secrets -n greenlang -o yaml | grep SAP
```

**Common Causes and Fixes:**

| Cause | Solution |
|-------|----------|
| Credentials expired | Rotate OAuth tokens |
| Network blocked | Check firewall rules |
| SAP service down | Contact SAP admin |
| URL changed | Update ConfigMap |

### Issue: IBM Maximo Integration Failing

**Symptoms:**
- Asset sync failing
- API key errors

**Diagnosis:**

```bash
# Check Maximo logs
kubectl logs -n greenlang -l app=gl-013-predictmaint --tail=100 | grep -i maximo

# Test Maximo API
curl -H "apikey: $MAXIMO_API_KEY" https://maximo.company.com/api/os/mxasset?_lid=maxadmin

# Verify API key
kubectl get secret gl-013-predictmaint-secrets -n greenlang -o jsonpath='{.data.MAXIMO_API_KEY}' | base64 -d
```

---

## Performance Issues

### Issue: High Prediction Latency

**Symptoms:**
- P99 latency >2000ms
- Users reporting slow responses

**Diagnosis:**

```bash
# Check latency metrics
curl -s http://localhost:9090/metrics | grep "predictmaint_prediction_latency"

# Check cache hit ratio
curl -s http://localhost:9090/metrics | grep "predictmaint_cache_hit"

# Check CPU/memory
kubectl top pods -n greenlang -l app=gl-013-predictmaint
```

**Common Causes and Fixes:**

| Cause | Solution |
|-------|----------|
| Cache miss rate high | Increase cache TTL |
| Database slow queries | Optimize queries, add indexes |
| Insufficient replicas | Scale out deployment |
| High CPU usage | Increase CPU limits |

**Fix: Scale Out**
```bash
kubectl scale deployment/gl-013-predictmaint -n greenlang --replicas=5
```

### Issue: Memory Pressure

**Symptoms:**
- OOMKilled pods
- Memory usage >90%

**Diagnosis:**

```bash
# Check memory usage
kubectl top pods -n greenlang -l app=gl-013-predictmaint

# Check for OOMKilled
kubectl get pods -n greenlang -l app=gl-013-predictmaint -o jsonpath='{.items[*].status.containerStatuses[*].lastState.terminated.reason}'
```

**Fix: Increase Memory Limits**
```bash
kubectl patch deployment gl-013-predictmaint -n greenlang -p '{"spec":{"template":{"spec":{"containers":[{"name":"gl-013-predictmaint","resources":{"limits":{"memory":"2Gi"}}}]}}}}'
```

---

## Data Issues

### Issue: Data Quality Alerts

**Symptoms:**
- Low data quality scores
- Missing sensor data

**Diagnosis:**

```bash
# Check data quality metrics
curl -s http://localhost:9090/metrics | grep "predictmaint_data_quality"

# Query recent data
psql -h postgres.greenlang.io -U readonly -d predictmaint -c "SELECT equipment_id, count(*), max(timestamp) FROM sensor_data WHERE timestamp > now() - interval '1 hour' GROUP BY equipment_id;"
```

**Common Causes and Fixes:**

| Cause | Solution |
|-------|----------|
| Sensor offline | Check IoT gateway |
| Network issues | Verify MQTT/OPC-UA connection |
| Data validation failures | Check input validation logs |

### Issue: Prediction Accuracy Degraded

**Symptoms:**
- Model accuracy metrics declining
- User-reported incorrect predictions

**Diagnosis:**

```bash
# Check model metrics
curl -s http://localhost:9090/metrics | grep "predictmaint_model_accuracy"

# Review recent predictions
psql -h postgres.greenlang.io -U readonly -d predictmaint -c "SELECT * FROM predictions WHERE created_at > now() - interval '24 hours' ORDER BY created_at DESC LIMIT 20;"
```

**Common Causes and Fixes:**

| Cause | Solution |
|-------|----------|
| Data drift | Retrain models |
| Configuration change | Review recent changes |
| Equipment changes | Update equipment parameters |

---

## Quick Reference

### Useful Log Patterns

```bash
# Error logs
kubectl logs -n greenlang -l app=gl-013-predictmaint | grep -i error

# Warning logs
kubectl logs -n greenlang -l app=gl-013-predictmaint | grep -i warn

# Database logs
kubectl logs -n greenlang -l app=gl-013-predictmaint | grep -i "postgres\|database"

# Cache logs
kubectl logs -n greenlang -l app=gl-013-predictmaint | grep -i "redis\|cache"

# Integration logs
kubectl logs -n greenlang -l app=gl-013-predictmaint | grep -i "cmms\|sap\|maximo"
```

### Common Fixes Summary

| Issue | Quick Fix |
|-------|-----------|
| Pods not starting | `kubectl rollout restart deployment/gl-013-predictmaint -n greenlang` |
| High latency | `kubectl scale deployment/gl-013-predictmaint -n greenlang --replicas=5` |
| Cache issues | `redis-cli -h redis.greenlang.io flushdb` |
| Config issues | `kubectl rollout restart deployment/gl-013-predictmaint -n greenlang` |

---

```
================================================================================
                    GL-013 PREDICTMAINT - Troubleshooting Guide
                         GreenLang Inc. - SRE Team
================================================================================
```
