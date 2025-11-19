# GL-007 FurnacePerformanceMonitor - Alert Runbook

## Overview

This runbook provides detailed response procedures for all GL-007 alerts. Follow these steps when alerts fire to diagnose and resolve issues quickly.

## General Response Workflow

1. **Acknowledge Alert** - Acknowledge in PagerDuty/alerting system
2. **Assess Severity** - Review alert severity and impact
3. **Gather Context** - Check dashboards, logs, traces
4. **Execute Runbook** - Follow specific procedures below
5. **Document** - Record actions and findings
6. **Resolve** - Fix root cause and clear alert
7. **Post-Mortem** - For critical issues, conduct post-mortem

---

## Critical Alerts

### GL007AgentUnavailable

**Severity**: Critical
**Trigger**: Agent unavailable for >1 minute
**Impact**: Complete loss of furnace monitoring

#### Symptoms
- Health check endpoints return 503
- Prometheus shows `up{job="gl-007-furnace-monitor"} == 0`
- No recent metrics in Grafana

#### Investigation Steps
1. Check pod status:
   ```bash
   kubectl get pods -n greenlang | grep gl-007
   kubectl describe pod <pod-name> -n greenlang
   ```

2. Review pod logs:
   ```bash
   kubectl logs <pod-name> -n greenlang --tail=100
   ```

3. Check recent events:
   ```bash
   kubectl get events -n greenlang --sort-by='.lastTimestamp' | head -20
   ```

#### Resolution
- **If pod is CrashLooping**: Check logs for errors, review recent deployments
- **If pod is OOMKilled**: Increase memory limits in deployment
- **If pod is Pending**: Check resource availability, node affinity
- **If pod is Running but unhealthy**: Exec into pod and check application logs

#### Escalation
If issue persists >15 minutes, escalate to GL-007 on-call engineer.

---

### GL007HighErrorRate

**Severity**: Critical
**Trigger**: Error rate >5% over 5 minutes
**Impact**: Significant service degradation

#### Symptoms
- Increased 5xx responses
- Failed calculations
- User reports of failures

#### Investigation Steps
1. Check error breakdown:
   ```promql
   sum(rate(gl_007_http_requests_total{status="error"}[5m])) by (endpoint, method)
   ```

2. Review error logs:
   ```bash
   kubectl logs <pod-name> -n greenlang | grep -i error | tail -50
   ```

3. Check recent deployments:
   ```bash
   kubectl rollout history deployment/gl-007-furnace-monitor -n greenlang
   ```

#### Resolution
- **If caused by recent deployment**: Rollback immediately
  ```bash
  kubectl rollout undo deployment/gl-007-furnace-monitor -n greenlang
  ```
- **If database-related**: Check database connectivity and query performance
- **If SCADA-related**: Verify SCADA connection status
- **If calculation errors**: Review input data quality and validation

#### Prevention
- Add integration tests covering failure scenarios
- Implement gradual rollout (canary deployment)
- Add input validation and error handling

---

### GL007SCADAConnectionDown

**Severity**: Critical
**Trigger**: SCADA connection lost for >2 minutes
**Impact**: Loss of real-time furnace data

#### Symptoms
- `gl_007_scada_connection_status == 0`
- No new SCADA data points received
- Stale furnace metrics

#### Investigation Steps
1. Check SCADA connection status:
   ```promql
   gl_007_scada_connection_status{scada_system=~".*"}
   ```

2. Test network connectivity:
   ```bash
   kubectl exec -it <pod-name> -n greenlang -- ping <scada-host>
   kubectl exec -it <pod-name> -n greenlang -- telnet <scada-host> <port>
   ```

3. Review SCADA polling errors:
   ```promql
   sum(rate(gl_007_scada_polling_errors_total[5m])) by (scada_system, error_type)
   ```

#### Resolution
- **If network issue**: Check firewall rules, VPN connection
- **If SCADA system down**: Contact SCADA team, check system status
- **If authentication failure**: Verify credentials, check certificate expiry
- **If timeout**: Adjust timeout configuration, check SCADA load

#### Workaround
- Agent continues with cached data for up to 5 minutes
- Manual data entry possible via API for critical operations

---

### GL007FurnaceTemperatureAnomaly

**Severity**: Critical
**Trigger**: Temperature deviation >15% from 1-hour average
**Impact**: Potential safety issue or equipment damage

#### Symptoms
- Sudden temperature spike or drop
- Abnormal temperature patterns
- Production quality issues

#### Investigation Steps
1. Check temperature trends:
   ```promql
   gl_007_furnace_temperature_celsius{furnace_id="<furnace>"}
   ```

2. Review related metrics:
   - Fuel consumption: `gl_007_furnace_fuel_consumption_kg_hr`
   - Production rate: `gl_007_furnace_production_rate_tons_hr`
   - Burner status: `gl_007_burner_performance_index`

3. Check sensor health:
   ```promql
   gl_007_scada_tag_quality{tag_name=~".*temperature.*"}
   ```

#### Resolution
- **If sensor malfunction**: Verify with backup sensors, dispatch maintenance
- **If operational change**: Review recent setpoint changes, production schedule
- **If fuel quality issue**: Check fuel analysis, switch fuel source if needed
- **If burner issue**: Inspect burner operation, check air-fuel ratio

#### Immediate Actions
1. Alert operations team
2. Verify actual furnace temperature with backup measurement
3. If confirmed anomaly, adjust operations per safety procedures
4. Document incident for analysis

---

### GL007LowThermalEfficiency

**Severity**: Critical
**Trigger**: Efficiency <70% for >10 minutes
**Impact**: Significant energy waste, production impact

#### Symptoms
- Thermal efficiency below critical threshold
- High fuel consumption relative to production
- Increased operating costs

#### Investigation Steps
1. Check efficiency trend:
   ```promql
   gl_007_furnace_thermal_efficiency_percent{furnace_id="<furnace>"}
   ```

2. Analyze contributing factors:
   - Heat losses: `gl_007_heat_loss_rate_kw`
   - Combustion efficiency: `gl_007_combustion_efficiency_percent`
   - Excess air: `gl_007_excess_air_percent`
   - Flue gas temperature: `gl_007_flue_gas_temperature_celsius`

3. Check operational context:
   - Production rate changes
   - Fuel type or quality changes
   - Recent maintenance activities

#### Resolution
- **High heat loss**: Check refractory condition, insulation integrity
- **Poor combustion**: Optimize air-fuel ratio, check burner condition
- **High flue gas temp**: Check heat recovery system, clean heat exchangers
- **Operational**: Review load profile, optimize production scheduling

#### Long-term Actions
- Schedule maintenance inspection
- Optimize combustion controls
- Consider heat recovery upgrades
- Review operational procedures

---

### GL007RefractoryDegradationCritical

**Severity**: Critical
**Trigger**: Degradation rate >5mm/day
**Impact**: Imminent refractory failure risk

#### Symptoms
- Rapid refractory wear
- Increasing wall temperature
- Potential structural integrity issues

#### Investigation Steps
1. Review degradation trends:
   ```promql
   gl_007_refractory_degradation_rate_mm_day{furnace_id="<furnace>"}
   ```

2. Check temperature profile:
   ```promql
   gl_007_refractory_temperature_celsius{furnace_id="<furnace>"}
   ```

3. Review operational history:
   - Temperature excursions
   - Load cycling frequency
   - Chemical exposure

#### Resolution
1. **Immediate**: Schedule emergency inspection
2. **Short-term**:
   - Reduce operating temperature if possible
   - Minimize thermal cycling
   - Monitor continuously
3. **Long-term**: Plan refractory replacement during next shutdown

#### Safety Considerations
- Risk of refractory failure
- Potential for molten material breach
- Follow plant safety procedures strictly

---

## Warning Alerts

### GL007PerformanceDegradation

**Severity**: Warning
**Trigger**: Latency increase >15% vs 1-hour baseline
**Impact**: User experience degradation

#### Investigation Steps
1. Identify slow endpoints:
   ```promql
   histogram_quantile(0.95, sum(rate(gl_007_http_request_duration_seconds_bucket[5m])) by (le, endpoint))
   ```

2. Check system resources:
   ```promql
   gl_007_system_cpu_usage_percent
   gl_007_system_memory_usage_bytes
   ```

3. Review database latency:
   ```promql
   histogram_quantile(0.95, sum(rate(gl_007_db_query_duration_seconds_bucket[5m])) by (le))
   ```

#### Resolution
- **High CPU**: Scale horizontally, optimize calculations
- **High memory**: Check for memory leaks, increase limits
- **Database slow**: Optimize queries, add indexes, scale database
- **External API slow**: Implement caching, add timeouts

---

### GL007LowCacheHitRate

**Severity**: Warning
**Trigger**: Cache hit rate <75%
**Impact**: Increased latency, database load

#### Investigation Steps
1. Check cache hit rate by pattern:
   ```promql
   100 * sum(rate(gl_007_cache_hits_total[5m])) by (cache_key_pattern)
   / (sum(rate(gl_007_cache_hits_total[5m])) + sum(rate(gl_007_cache_misses_total[5m])))
   ```

2. Review cache evictions:
   ```promql
   sum(rate(gl_007_cache_evictions_total[5m])) by (cache_key_pattern)
   ```

3. Check cache size and usage:
   - Redis memory usage
   - Key count
   - Eviction policy

#### Resolution
- **High eviction rate**: Increase cache size
- **TTL too short**: Review and adjust TTL values
- **Cache warming needed**: Implement cache pre-population
- **Access pattern changed**: Review caching strategy

---

### GL007EfficiencyBelowTarget

**Severity**: Warning
**Trigger**: Efficiency <80% for >15 minutes
**Impact**: Suboptimal performance, higher costs

#### Investigation Steps
1. Compare to baseline:
   ```promql
   avg_over_time(gl_007_furnace_thermal_efficiency_percent[24h])
   ```

2. Check correlation metrics:
   ```promql
   gl_007_energy_production_correlation
   gl_007_efficiency_temperature_correlation
   ```

3. Review operational parameters:
   - Load factor
   - Fuel quality
   - Ambient conditions

#### Resolution
- **Normal variation**: Document, continue monitoring
- **Operational**: Adjust setpoints, optimize load profile
- **Equipment**: Schedule maintenance, clean heat transfer surfaces
- **Fuel quality**: Switch fuel source, adjust combustion settings

---

### GL007HighFuelConsumption

**Severity**: Warning
**Trigger**: Fuel consumption +20% vs 6-hour average
**Impact**: Increased costs, potential efficiency issue

#### Investigation Steps
1. Check fuel consumption trend:
   ```promql
   gl_007_furnace_fuel_consumption_kg_hr{furnace_id="<furnace>"}
   ```

2. Compare to production:
   ```promql
   gl_007_furnace_specific_energy_consumption_kwh_ton
   ```

3. Review potential causes:
   - Production rate changes
   - Fuel quality changes
   - Equipment degradation

#### Resolution
- **Production increase**: Normal, verify SEC is stable
- **Fuel quality**: Check fuel analysis, adjust combustion
- **Air leakage**: Inspect furnace, repair seals
- **Equipment issue**: Inspect burners, controls, heat recovery

---

### GL007PredictionAccuracyDrop

**Severity**: Warning
**Trigger**: Model accuracy <85%
**Impact**: Reduced prediction reliability

#### Investigation Steps
1. Check model performance:
   ```promql
   avg(gl_007_prediction_accuracy) by (model_type)
   ```

2. Review prediction errors:
   ```promql
   gl_007_model_prediction_error{model_type="<model>"}
   ```

3. Check data quality:
   - Sensor health
   - Missing data
   - Outliers

#### Resolution
- **Data drift**: Retrain model with recent data
- **Sensor issues**: Fix sensors, exclude bad data
- **Model staleness**: Schedule model update
- **Operational changes**: Update model features, retrain

---

### GL007SensorMalfunction

**Severity**: Warning
**Trigger**: Sensor reading unchanged for 15 minutes during production
**Impact**: Inaccurate monitoring data

#### Investigation Steps
1. Identify stuck sensors:
   ```promql
   changes(gl_007_furnace_temperature_celsius[15m]) == 0
   and gl_007_furnace_production_rate_tons_hr > 0
   ```

2. Check sensor quality:
   ```promql
   gl_007_scada_tag_quality{tag_name="<sensor>"}
   ```

3. Compare with nearby sensors

#### Resolution
1. Verify malfunction (not actual steady state)
2. Switch to backup sensor if available
3. Dispatch maintenance for inspection/replacement
4. Exclude sensor from calculations until fixed
5. Document for trending analysis

---

## Business Alerts

### GL007LowEnergySavings

**Severity**: Info
**Trigger**: Annual savings <$100k
**Impact**: ROI below target

#### Investigation Steps
1. Review savings trend:
   ```promql
   sum(gl_007_annual_energy_savings_usd)
   ```

2. Analyze by furnace:
   ```promql
   gl_007_energy_cost_savings_usd_hr{furnace_id=~".*"}
   ```

3. Check efficiency performance:
   - Average efficiency vs target
   - Optimization opportunities identified
   - Recommendations implemented

#### Resolution
- **Low efficiency**: Focus on optimization opportunities
- **Limited scope**: Expand monitoring to more furnaces
- **Market factors**: Review energy prices, update calculations
- **Operational**: Increase optimization engagement

---

## SLO Alerts

### GL007SLOAvailabilityViolation

**Severity**: Critical
**Trigger**: 30-day availability <99.9%
**Impact**: SLA breach risk

#### Investigation Steps
1. Check uptime history:
   ```promql
   avg_over_time(up{job="gl-007-furnace-monitor"}[30d])
   ```

2. Identify outage causes:
   ```bash
   kubectl get events -n greenlang --sort-by='.lastTimestamp' | grep gl-007
   ```

3. Review incident log

#### Resolution
1. Address root causes from incidents
2. Implement redundancy improvements
3. Review deployment processes
4. Update runbooks based on learnings

---

## Quality Alerts

### GL007DeterminismFailure

**Severity**: Critical
**Trigger**: Determinism score <100%
**Impact**: Zero-hallucination guarantee violated

#### Investigation Steps
1. Check determinism score by component:
   ```promql
   gl_007_determinism_score_percent{component=~".*"}
   ```

2. Review violations:
   ```promql
   sum(rate(gl_007_determinism_verification_failures_total[5m])) by (violation_type)
   ```

3. Check logs for non-deterministic operations

#### Resolution
1. **Immediate**: Stop affected operations
2. **Investigation**: Identify source of non-determinism
3. **Fix**: Remove random operations, fix timestamp usage
4. **Verify**: Run determinism tests
5. **Deploy**: Roll out fix
6. **Monitor**: Verify score returns to 100%

#### Common Causes
- Unseeded random number generation
- Timestamp-based calculations
- External API call ordering
- Concurrent operation race conditions

---

## Escalation Matrix

| Severity | Response Time | Escalation Path |
|----------|--------------|-----------------|
| Critical | 5 minutes | On-call engineer → Team lead → Director |
| Warning | 30 minutes | Assigned engineer → On-call if unresolved in 2h |
| Info | Next business day | Team backlog |

## Contact Information

- **On-call Engineer**: PagerDuty rotation
- **Team Lead**: greenlang-ops-lead@greenlang.io
- **SCADA Team**: scada-support@plant.io
- **Database Team**: dba@greenlang.io
- **Slack Channels**:
  - #gl-007-alerts (all alerts)
  - #gl-007-ops (operations)
  - #greenlang-incidents (critical incidents)

## Tools & Resources

- **Grafana**: https://grafana.greenlang.io/d/gl-007-ops
- **Prometheus**: https://prometheus.greenlang.io
- **Logs**: https://kibana.greenlang.io (or Loki)
- **Tracing**: https://jaeger.greenlang.io
- **Documentation**: https://docs.greenlang.io/gl-007
- **Runbooks**: https://github.com/greenlang/runbooks/gl-007

## Post-Incident Actions

After resolving any critical incident:

1. **Document** incident in incident tracker
2. **Timeline** create detailed timeline of events
3. **RCA** conduct root cause analysis
4. **Action Items** identify improvements
5. **Runbook Update** update this runbook with learnings
6. **Communication** share post-mortem with stakeholders
7. **Follow-up** track action items to completion

---

**Last Updated**: 2025-01-19
**Version**: 1.0.0
**Maintained by**: GreenLang Operations Team
