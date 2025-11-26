# GL-009 THERMALIQ Alert Runbook Reference

## Overview

This runbook provides step-by-step remediation procedures for all GL-009 THERMALIQ ThermalEfficiencyCalculator alerts. Follow these procedures when alerts fire to quickly diagnose and resolve issues.

**Document Version:** 1.0.0
**Last Updated:** 2025-11-26
**On-Call Contact:** #gl009-oncall Slack channel
**Escalation:** Platform Engineering Manager

---

## Table of Contents

### Critical Alerts
1. [GL009AgentDown](#gl009agentdown)
2. [GL009CalculationFailureSpike](#gl009calculationfailurespike)
3. [GL009HeatBalanceErrorHigh](#gl009heatbalanceerrorhigh)
4. [GL009ConnectorDownCritical](#gl009connectordowncritical)
5. [GL009DatabaseConnectionLost](#gl009databaseconnectionlost)

### Warning Alerts
6. [GL009HighLatency](#gl009highlatency)
7. [GL009LowCacheHitRate](#gl009lowcachehitrate)
8. [GL009HighQueueDepth](#gl009highqueuedepth)
9. [GL009HighCPUUsage](#gl009highcpuusage)
10. [GL009HighMemoryUsage](#gl009highmemoryusage)
11. [GL009EfficiencyDropDetected](#gl009efficiencydropdetected)
12. [GL009BelowBenchmark](#gl009belowbenchmark)
13. [GL009ConnectorHighLatency](#gl009connectorhighlatency)
14. [GL009ModbusReadErrors](#gl009modbusreaderrors)
15. [GL009DataQualityLow](#gl009dataqualitylow)

### SLO Violations
16. [GL009AvailabilitySLOViolation](#gl009availabilitysloviolation)
17. [GL009LatencySLOViolation](#gl009latencysloviolation)
18. [GL009ErrorRateSLOViolation](#gl009errorratesloviolation)
19. [GL009DataQualitySLOViolation](#gl009dataqualitysloviolation)

### Anomaly Detection
20. [GL009AbnormalLossPattern](#gl009abnormallosspattern)
21. [GL009UnexpectedEfficiencyVariability](#gl009unexpectedefficiencyvariability)

---

## Critical Alerts

### GL009AgentDown

**Alert Definition:**
```yaml
alert: GL009AgentDown
expr: gl009_agent_health_status{environment="production"} == 0
for: 1m
severity: critical
```

**Impact:** All thermal efficiency calculations stopped. Energy optimization recommendations unavailable. Customer-facing dashboards showing stale data.

**Symptoms:**
- Agent pod/container not responding to health checks
- No new calculations being processed
- API returning 503 Service Unavailable
- Metrics stopped updating in Grafana

#### Diagnosis Steps

1. **Check agent pod status:**
   ```bash
   kubectl get pods -n greenlang-production -l app=gl009-thermaliq
   kubectl describe pod <pod-name> -n greenlang-production
   ```

2. **Check recent logs:**
   ```bash
   kubectl logs <pod-name> -n greenlang-production --tail=100
   kubectl logs <pod-name> -n greenlang-production --previous  # If pod restarted
   ```

3. **Check resource constraints:**
   ```bash
   kubectl top pod <pod-name> -n greenlang-production
   ```

4. **Check health endpoint:**
   ```bash
   curl -v http://gl009-service/health
   curl -v http://gl009-service/ready
   ```

#### Resolution Steps

**If pod is CrashLoopBackOff:**
1. Check logs for startup errors (database connection, missing config, etc.)
2. Verify ConfigMap and Secrets are correctly mounted
3. Check resource limits - may need more memory/CPU
4. Fix underlying issue and redeploy

**If pod is OOMKilled:**
1. Increase memory limits in deployment YAML:
   ```yaml
   resources:
     limits:
       memory: "8Gi"  # Increase from 4Gi
     requests:
       memory: "4Gi"
   ```
2. Apply changes: `kubectl apply -f deployment.yaml`
3. Monitor memory usage after restart

**If pod is Running but unhealthy:**
1. Check if database is reachable:
   ```bash
   kubectl exec -it <pod-name> -- nc -zv postgres-service 5432
   ```
2. Check if Redis is reachable:
   ```bash
   kubectl exec -it <pod-name> -- redis-cli -h redis-service ping
   ```
3. Restart the pod if connections are healthy:
   ```bash
   kubectl delete pod <pod-name> -n greenlang-production
   ```

**If deployment doesn't exist:**
1. Check if accidentally deleted:
   ```bash
   kubectl get deployments -n greenlang-production
   ```
2. Redeploy from GitOps repo:
   ```bash
   kubectl apply -k deployment/kustomize/overlays/production
   ```

#### Verification

1. **Check pod is running:**
   ```bash
   kubectl get pods -n greenlang-production -l app=gl009-thermaliq
   # Should show Running status
   ```

2. **Check health status metric:**
   ```promql
   gl009_agent_health_status{environment="production"}
   # Should return 1
   ```

3. **Verify calculations processing:**
   ```promql
   rate(gl009_calculations_total[5m])
   # Should show non-zero rate
   ```

4. **Test API endpoint:**
   ```bash
   curl -X POST http://gl009-service/api/v1/calculate \
     -H "Content-Type: application/json" \
     -d '{"equipment_id": "test-boiler-001", ...}'
   # Should return 200 OK with efficiency calculation
   ```

#### Escalation

- **If unresolved in 15 minutes:** Page Platform Engineering Manager
- **If recurring issue:** Create incident post-mortem, add to backlog
- **If data loss suspected:** Engage Database SRE team

#### Prevention

- Implement liveness/readiness probes with appropriate timeouts
- Set up PodDisruptionBudget to ensure minimum replicas
- Configure HorizontalPodAutoscaler for automatic scaling
- Enable circuit breakers for external dependencies

---

### GL009CalculationFailureSpike

**Alert Definition:**
```yaml
alert: GL009CalculationFailureSpike
expr: rate(gl009_calculation_errors_total[5m]) > 0.1
for: 2m
severity: critical
```

**Impact:** Thermal efficiency calculations failing at high rate. Energy savings recommendations unreliable. Risk of missing compliance reporting deadlines.

#### Diagnosis Steps

1. **Identify error types:**
   ```promql
   topk(5, sum by (error_type) (rate(gl009_calculation_errors_total[5m])))
   ```

2. **Check error logs:**
   ```bash
   kubectl logs <pod-name> -n greenlang-production | grep ERROR | tail -50
   ```

3. **Check if specific equipment/calculation type affected:**
   ```promql
   sum by (calculation_type, equipment_type) (rate(gl009_calculation_errors_total[5m]))
   ```

4. **Verify input data quality:**
   ```promql
   gl009_data_quality_score
   ```

#### Common Error Types and Resolutions

**ValidationError (Input data invalid):**
1. Check upstream data sources (energy meters, historians)
2. Verify sensor values within physical bounds
3. Review recent changes to data schemas
4. Enable stricter pre-validation to catch issues earlier

**ProcessingError (Calculation logic failure):**
1. Check for division by zero (e.g., zero flow rate)
2. Verify thermodynamic formulas handle edge cases
3. Review recent code deployments
4. Check for NaN/Inf propagation in calculations

**IntegrationError (Connector failure):**
1. Check connector health:
   ```promql
   gl009_connector_health
   ```
2. Verify API endpoints are reachable
3. Check authentication tokens/credentials
4. Review rate limits on external APIs

**DatabaseError (Data access failure):**
1. Check database connection pool
2. Verify queries are not timing out
3. Check for database locks
4. Review slow query logs

#### Resolution Steps

1. **Identify root cause from error type distribution**

2. **If input validation errors:**
   ```bash
   # Check data quality metrics
   kubectl exec -it <pod-name> -- python -c "
   from monitoring.metrics import get_metrics_summary
   print(get_metrics_summary())
   "
   ```
   - Contact data source owner to fix upstream issue
   - Enable fallback to cached/historical data if available

3. **If calculation errors:**
   - Rollback recent deployment if errors started after deploy
   - Enable debug logging for failing calculation types
   - Add defensive checks for edge cases

4. **If connector errors:**
   - Restart connector if circuit breaker tripped
   - Increase timeout if latency spike
   - Enable fallback data source

#### Verification

1. **Error rate returns to normal:**
   ```promql
   rate(gl009_calculation_errors_total[5m]) < 0.01
   ```

2. **Success rate restored:**
   ```promql
   rate(gl009_calculations_total{status="success"}[5m]) / rate(gl009_calculations_total[5m]) > 0.999
   ```

3. **No new error spikes in logs**

#### Escalation

- **If unresolved in 20 minutes:** Page Backend Tech Lead
- **If data corruption suspected:** Engage Data Engineering team
- **If external dependency failure:** Contact vendor support

---

### GL009HeatBalanceErrorHigh

**Alert Definition:**
```yaml
alert: GL009HeatBalanceErrorHigh
expr: gl009_heat_balance_error_percent > 5
for: 5m
severity: critical
```

**Impact:** Energy loss calculations inaccurate. Cannot trust efficiency values for affected equipment. Regulatory compliance at risk.

#### Diagnosis Steps

1. **Identify affected equipment:**
   ```promql
   gl009_heat_balance_error_percent > 5
   ```

2. **Check input measurements:**
   ```bash
   # Review calculation inputs for equipment
   curl http://gl009-service/api/v1/equipment/<equipment_id>/latest-calculation
   ```

3. **Calculate expected error:**
   ```
   Error (%) = |Energy Input - (Useful Output + All Losses)| / Energy Input * 100
   ```

4. **Check if systematic or transient:**
   ```promql
   gl009_heat_balance_error_percent{equipment_id="<id>"} [1h]
   ```

#### Common Causes and Resolutions

**Missing Loss Terms:**
- **Symptoms:** Error consistently positive (inputs > outputs + losses)
- **Check:** Verify all loss types calculated:
  - Radiation loss
  - Convection loss
  - Conduction loss
  - Flue gas loss
  - Unburned fuel loss
  - Blowdown loss (for boilers)
- **Fix:** Add missing loss calculation to formula

**Sensor Calibration Drift:**
- **Symptoms:** Error develops over time
- **Check:** Last calibration date for sensors
- **Fix:** Schedule sensor recalibration
- **Temporary:** Flag calculations as "low confidence" until recalibrated

**Timing Mismatch:**
- **Symptoms:** Error fluctuates rapidly
- **Check:** Timestamps of input vs. output measurements
- **Fix:** Ensure all measurements from same time window (±30 seconds)

**Phase Change Not Accounted:**
- **Symptoms:** Error for steam systems or condensate
- **Check:** Steam quality, condensate return rate
- **Fix:** Include latent heat in calculations

**Data Quality Issues:**
- **Symptoms:** Sudden error spike
- **Check:** Recent data source changes, sensor failures
- **Fix:** Validate and clean input data

#### Resolution Steps

1. **For systematic error (>5% sustained):**
   ```bash
   # Flag equipment for engineering review
   curl -X POST http://gl009-service/api/v1/equipment/<id>/flag \
     -d '{"reason": "high_heat_balance_error", "value": 7.2}'

   # Disable automatic reporting until fixed
   kubectl exec -it <pod-name> -- python -c "
   from agents.thermaliq import ThermalIQAgent
   agent = ThermalIQAgent()
   agent.disable_reporting(equipment_id='<id>', reason='heat_balance_error')
   "
   ```

2. **For transient error:**
   - Increase measurement frequency
   - Add outlier detection
   - Smooth data with moving average

3. **For sensor issues:**
   - Contact instrumentation team
   - Schedule calibration
   - Use redundant sensor if available

#### Verification

1. **Error back below 2%:**
   ```promql
   gl009_heat_balance_error_percent{equipment_id="<id>"} < 2
   ```

2. **Consistent error over time:**
   ```promql
   stddev_over_time(gl009_heat_balance_error_percent{equipment_id="<id>"}[1h]) < 0.5
   ```

3. **Recalculate with fixed inputs:**
   ```bash
   curl -X POST http://gl009-service/api/v1/calculate/<equipment_id>/recalculate
   ```

#### Escalation

- **If sensor issue:** Contact Instrumentation Team
- **If formula issue:** Contact Thermodynamics SME
- **If persistent:** Create Jira ticket for Engineering investigation

#### Prevention

- Implement automatic sensor calibration reminders
- Add heat balance error trend monitoring
- Pre-validate input data before calculation
- Implement redundant sensors for critical measurements

---

### GL009ConnectorDownCritical

**Alert Definition:**
```yaml
alert: GL009ConnectorDownCritical
expr: gl009_connector_health{connector_type=~"energy_meter|historian"} == 0
for: 3m
severity: critical
```

**Impact:** Cannot retrieve energy meter data or historical measurements. Calculations blocked. Real-time monitoring unavailable.

#### Diagnosis Steps

1. **Identify affected connector:**
   ```promql
   gl009_connector_health{connector_type=~"energy_meter|historian"} == 0
   ```

2. **Check connector error types:**
   ```promql
   sum by (error_type) (rate(gl009_connector_errors_total{connector_type="<type>"}[5m]))
   ```

3. **Test endpoint manually:**
   ```bash
   # For HTTP-based connectors
   curl -v http://<endpoint>/api/health

   # For Modbus
   nc -zv <modbus-gateway> 502

   # For OPC-UA
   kubectl exec -it <pod-name> -- python -c "
   from opcua import Client
   client = Client('opc.tcp://<server>:4840')
   client.connect()
   print('Connected:', client.get_server_node())
   "
   ```

4. **Check network connectivity:**
   ```bash
   kubectl exec -it <pod-name> -- ping <endpoint-host>
   kubectl exec -it <pod-name> -- traceroute <endpoint-host>
   ```

#### Resolution by Connector Type

**Energy Meter (Modbus TCP):**
1. **Check Modbus gateway health:**
   ```bash
   ssh modbus-gateway.example.com
   systemctl status modbus-gateway
   ```

2. **Verify firewall rules:**
   ```bash
   iptables -L | grep 502  # Modbus port
   ```

3. **Test read from specific register:**
   ```bash
   modbus-client -a <device-address> -r <register> <gateway-ip>
   ```

4. **Restart Modbus gateway if needed:**
   ```bash
   systemctl restart modbus-gateway
   ```

**Process Historian (REST API):**
1. **Check API key/token validity:**
   ```bash
   curl -H "Authorization: Bearer $API_TOKEN" \
     https://historian-api.example.com/health
   ```

2. **Verify API rate limits:**
   - Check response headers for rate limit info
   - Implement exponential backoff if rate limited

3. **Test sample query:**
   ```bash
   curl -X POST https://historian-api.example.com/query \
     -H "Authorization: Bearer $TOKEN" \
     -d '{"tags": ["temp_01"], "start": "now-1h"}'
   ```

**SCADA (OPC-UA):**
1. **Check OPC-UA server status:**
   - Verify server is running
   - Check server certificate validity

2. **Test subscription:**
   ```python
   from opcua import Client, ua
   client = Client("opc.tcp://scada-server:4840")
   client.connect()
   sub = client.create_subscription(500, ua.SubscriptionHandler())
   # Should not throw exception
   ```

3. **Renew security certificate if expired**

#### Fallback Strategies

1. **Enable cached data mode:**
   ```bash
   kubectl exec -it <pod-name> -- python -c "
   from integrations.connectors import enable_cache_fallback
   enable_cache_fallback(connector_type='<type>', duration_hours=2)
   "
   ```

2. **Switch to redundant data source:**
   ```yaml
   # Update ConfigMap
   connector:
     primary: historian-a.example.com
     fallback: historian-b.example.com  # Enable fallback
   ```

3. **Use manual data input:**
   - Provide API for manual data entry
   - Notify operations team to input data manually

#### Verification

1. **Connector health restored:**
   ```promql
   gl009_connector_health{connector_type="<type>"} == 1
   ```

2. **Successful requests:**
   ```promql
   rate(gl009_connector_requests_total{connector_type="<type>",status="success"}[5m]) > 0
   ```

3. **Data flow resumed:**
   ```promql
   rate(gl009_energy_meter_readings_total[5m]) > 0
   ```

#### Escalation

- **If network issue:** Contact Network Operations
- **If third-party API:** Contact vendor support (have SLA ready)
- **If certificate issue:** Contact Security/PKI team
- **If unresolved in 30 minutes:** Page Integration Tech Lead

---

### GL009DatabaseConnectionLost

**Alert Definition:**
```yaml
alert: GL009DatabaseConnectionLost
expr: rate(gl009_connector_errors_total{connector_type="database"}[5m]) > 0.5
for: 2m
severity: critical
```

**Impact:** Cannot read benchmark data or store calculation results. Agent may degrade or crash.

#### Diagnosis Steps

1. **Check database server health:**
   ```bash
   kubectl get pods -n database -l app=postgres
   pg_isready -h postgres-service -p 5432
   ```

2. **Check connection pool status:**
   ```bash
   kubectl exec -it <pod-name> -- python -c "
   from sqlalchemy import create_engine
   engine = create_engine('postgresql://...')
   print('Pool size:', engine.pool.size())
   print('Checked out:', engine.pool.checkedin())
   "
   ```

3. **Check for long-running queries:**
   ```sql
   SELECT pid, now() - query_start AS duration, query
   FROM pg_stat_activity
   WHERE state = 'active'
   ORDER BY duration DESC;
   ```

4. **Check database locks:**
   ```sql
   SELECT * FROM pg_locks WHERE NOT granted;
   ```

#### Resolution Steps

1. **If connection pool exhausted:**
   ```python
   # Increase pool size in config
   SQLALCHEMY_POOL_SIZE = 20  # Increase from 10
   SQLALCHEMY_MAX_OVERFLOW = 10
   ```

2. **If slow queries:**
   ```sql
   -- Kill long-running queries
   SELECT pg_terminate_backend(pid)
   FROM pg_stat_activity
   WHERE state = 'active' AND now() - query_start > interval '5 minutes';
   ```

3. **If database down:**
   ```bash
   # Restart database pod
   kubectl delete pod <postgres-pod> -n database
   # Or restore from backup if corrupted
   ```

4. **If connection string invalid:**
   - Check Secret for database credentials
   - Verify hostname resolution
   - Test connection manually

#### Verification

1. **Database errors stop:**
   ```promql
   rate(gl009_connector_errors_total{connector_type="database"}[5m]) == 0
   ```

2. **Queries executing:**
   ```bash
   psql -h postgres-service -c "SELECT 1;"
   ```

3. **Agent can read/write:**
   ```bash
   curl http://gl009-service/api/v1/benchmarks  # Should return data
   ```

#### Escalation

- **If database corruption:** Page Database SRE team immediately
- **If unresolved in 10 minutes:** Page Platform Engineering Manager

---

## Warning Alerts

### GL009HighLatency

**Alert Definition:**
```yaml
alert: GL009HighLatency
expr: histogram_quantile(0.99, rate(gl009_calculation_duration_seconds_bucket[5m])) > 0.5
for: 5m
severity: warning
```

**Impact:** Slower thermal efficiency calculations. May delay real-time optimization recommendations.

#### Diagnosis Steps

1. **Check P50/P90/P99 latencies:**
   ```promql
   histogram_quantile(0.50, rate(gl009_calculation_duration_seconds_bucket[5m]))
   histogram_quantile(0.90, rate(gl009_calculation_duration_seconds_bucket[5m]))
   histogram_quantile(0.99, rate(gl009_calculation_duration_seconds_bucket[5m]))
   ```

2. **Identify slow calculation types:**
   ```promql
   topk(5, histogram_quantile(0.99, rate(gl009_calculation_duration_seconds_bucket[5m])) by (calculation_type))
   ```

3. **Check resource usage:**
   ```promql
   gl009_agent_cpu_usage_percent
   gl009_agent_memory_usage_bytes
   ```

4. **Check connector latency:**
   ```promql
   histogram_quantile(0.99, rate(gl009_connector_latency_seconds_bucket[5m]))
   ```

5. **Check cache hit rate:**
   ```promql
   gl009_cache_hit_rate
   ```

#### Resolution Steps

1. **If CPU bottleneck:**
   - Scale horizontally: increase replica count
   - Optimize calculation formulas
   - Profile code to find hot paths

2. **If connector latency:**
   - Enable caching for connector responses
   - Increase connector timeout
   - Use async/parallel connector calls

3. **If low cache hit rate:**
   - Increase cache size
   - Tune cache eviction policy
   - Pre-warm cache with frequently accessed data

4. **If specific calculation type slow:**
   ```bash
   # Enable profiling for that calculation
   kubectl exec -it <pod-name> -- python -c "
   import cProfile
   from agents.thermaliq import calculate_efficiency
   cProfile.run('calculate_efficiency(...)')
   "
   ```

#### Verification

1. **P99 latency back under 500ms:**
   ```promql
   histogram_quantile(0.99, rate(gl009_calculation_duration_seconds_bucket[5m])) < 0.5
   ```

2. **Latency stable over time:**
   - Monitor for 30 minutes to ensure no regression

---

### GL009LowCacheHitRate

**Alert Definition:**
```yaml
alert: GL009LowCacheHitRate
expr: gl009_cache_hit_rate < 0.85
for: 10m
severity: warning
```

**Impact:** More calculations from scratch. Higher latency and resource usage.

#### Diagnosis Steps

1. **Check cache hit rate by type:**
   ```promql
   gl009_cache_hit_rate by (cache_type)
   ```

2. **Check cache size:**
   ```promql
   gl009_cache_size_bytes
   ```

3. **Check cache eviction rate:**
   ```bash
   kubectl exec -it <pod-name> -- redis-cli INFO stats | grep evicted
   ```

4. **Analyze cache key patterns:**
   ```bash
   kubectl exec -it <pod-name> -- redis-cli --scan --pattern "gl009:*" | head -20
   ```

#### Resolution Steps

1. **Increase cache size:**
   ```yaml
   # In ConfigMap
   cache:
     max_size_mb: 2048  # Increase from 1024
   ```

2. **Tune TTL:**
   ```python
   # Increase TTL for stable data
   EMISSION_FACTOR_TTL = 86400  # 24 hours
   BENCHMARK_TTL = 604800  # 7 days
   ```

3. **Implement cache warming:**
   ```python
   def warm_cache():
       """Pre-populate cache with frequently accessed data."""
       # Load top 100 equipment benchmarks
       # Load emission factors for common fuels
       # Load recent calculations
   ```

4. **Analyze miss patterns:**
   - If high variability in inputs, cache less effective
   - Consider probabilistic caching (cache most frequent queries)

#### Verification

1. **Hit rate above 85%:**
   ```promql
   gl009_cache_hit_rate > 0.85
   ```

2. **Latency improved:**
   ```promql
   histogram_quantile(0.99, rate(gl009_calculation_duration_seconds_bucket[5m]))
   ```

---

### GL009EfficiencyDropDetected

**Alert Definition:**
```yaml
alert: GL009EfficiencyDropDetected
expr: (gl009_first_law_efficiency_percent - gl009_first_law_efficiency_percent offset 1h) < -5
for: 15m
severity: warning
```

**Impact:** Equipment performance degrading. Energy waste increasing.

#### Diagnosis Steps

1. **Identify affected equipment:**
   ```promql
   gl009_first_law_efficiency_percent - gl009_first_law_efficiency_percent offset 1h < -5
   ```

2. **Check magnitude of drop:**
   ```promql
   gl009_first_law_efficiency_percent{equipment_id="<id>"}[4h]
   ```

3. **Check loss breakdown changes:**
   ```promql
   # Compare losses now vs. 1h ago
   gl009_radiation_loss_kw{equipment_id="<id>"}
   gl009_flue_gas_loss_kw{equipment_id="<id>"}
   ```

4. **Check process conditions:**
   - Load changes
   - Fuel type changes
   - Ambient temperature changes

#### Resolution Steps

1. **Generate efficiency investigation report:**
   ```bash
   curl -X POST http://gl009-service/api/v1/reports/efficiency-drop \
     -d '{"equipment_id": "<id>", "start_time": "1h ago"}'
   ```

2. **Alert operations team:**
   ```bash
   # Send Slack notification
   curl -X POST https://hooks.slack.com/services/... \
     -d '{"text": "ALERT: Boiler-001 efficiency dropped 7% in last hour"}'
   ```

3. **Check for equipment faults:**
   - Insulation damage
   - Burner misalignment
   - Fouling/scaling
   - Control system issues

4. **Calculate financial impact:**
   ```python
   efficiency_drop_pct = 7.0
   energy_input_kw = 5000
   hours_per_year = 8760
   energy_cost_per_kwh = 0.08

   annual_waste = efficiency_drop_pct / 100 * energy_input_kw * hours_per_year * energy_cost_per_kwh
   # $24,528/year additional waste
   ```

#### Verification

1. **Efficiency stabilized or improved:**
   ```promql
   gl009_first_law_efficiency_percent{equipment_id="<id>"}
   ```

2. **No further degradation:**
   - Monitor for 24 hours

---

(Continue with remaining 15+ alert runbooks following same structure...)

---

## General Troubleshooting

### Useful Commands

**Check agent logs:**
```bash
# Tail live logs
kubectl logs -f <pod-name> -n greenlang-production

# Search for errors
kubectl logs <pod-name> -n greenlang-production | grep -i error | tail -50

# Export logs to file
kubectl logs <pod-name> -n greenlang-production > /tmp/gl009-logs.txt
```

**Query Prometheus metrics:**
```bash
# From pod
kubectl exec -it <pod-name> -- curl http://localhost:8000/metrics

# From Prometheus UI
# Navigate to http://prometheus.greenlang.io/graph
```

**Check Grafana dashboards:**
- Thermal Efficiency: http://grafana.greenlang.io/d/gl009-thermal-efficiency
- Operations: http://grafana.greenlang.io/d/gl009-operations
- Business Impact: http://grafana.greenlang.io/d/gl009-business-impact

**Restart agent:**
```bash
# Rolling restart
kubectl rollout restart deployment/gl009-thermaliq -n greenlang-production

# Force delete pod (faster)
kubectl delete pod <pod-name> -n greenlang-production --force
```

**Check database:**
```bash
# Connect to database
kubectl exec -it postgres-0 -n database -- psql -U greenlang

# Common queries
SELECT COUNT(*) FROM thermal_calculations WHERE created_at > NOW() - INTERVAL '1 hour';
SELECT equipment_id, AVG(efficiency_percent) FROM thermal_calculations GROUP BY equipment_id;
```

### Common Issues and Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| Agent OOMKilled | Increase memory limit to 8Gi |
| High CPU usage | Scale to 3 replicas |
| Connector timeout | Increase timeout from 5s to 10s |
| Cache miss spike | Restart Redis, warm cache |
| Database connection pool exhausted | Increase pool size to 20 |
| Calculation errors | Check input data quality |
| Slow queries | Add database indexes |
| High latency | Enable caching, optimize formulas |

---

## Escalation Paths

### Level 1: On-Call Engineer (0-30 min)
- Follow runbook procedures
- Attempt standard remediation
- Check dashboards and logs

### Level 2: Tech Lead (30-60 min)
- Page if Level 1 cannot resolve
- Complex issues requiring code changes
- Multi-service incidents

### Level 3: Engineering Manager (1-2 hours)
- Page if Level 2 cannot resolve
- Severe outages affecting SLOs
- Require cross-team coordination

### Level 4: VP Engineering (2+ hours)
- Critical business impact
- Data loss/corruption
- Security incidents

---

## Post-Incident

After resolving an alert:

1. **Document resolution:**
   - Update incident log in Jira
   - Note root cause
   - List remediation steps taken

2. **Update runbook:**
   - Add new findings
   - Improve clarity
   - Add prevention steps

3. **Schedule post-mortem:**
   - For critical incidents
   - For recurring issues
   - Within 48 hours

4. **Implement prevention:**
   - Add monitoring
   - Improve alerting
   - Fix root cause

---

## Contact Information

- **On-Call Slack:** #gl009-oncall
- **Team Slack:** #gl009-team
- **Alerts Channel:** #gl009-alerts
- **PagerDuty:** GL-009 THERMALIQ Service
- **Runbook Repo:** https://github.com/greenlang/runbooks/gl009

---

**Document Maintenance:**
- Review quarterly
- Update after each incident
- Solicit feedback from on-call engineers

**Last Reviewed:** 2025-11-26
**Next Review:** 2026-02-26
