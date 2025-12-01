# GL-011 FUELCRAFT Incident Response Runbook

**Agent**: GL-011 FUELCRAFT FuelManagementOrchestrator
**Version**: 1.0.0
**Last Updated**: 2025-12-01
**Owner**: GreenLang SRE Team
**On-Call Rotation**: PagerDuty Schedule "GL-011-Primary"

---

## Table of Contents

1. [Incident Severity Levels](#incident-severity-levels)
2. [Incident Response Process](#incident-response-process)
3. [Incident Scenarios](#incident-scenarios)
   - [SEV1: Critical Incidents](#sev1-critical-incidents)
   - [SEV2: High Severity Incidents](#sev2-high-severity-incidents)
   - [SEV3: Medium Severity Incidents](#sev3-medium-severity-incidents)
   - [SEV4: Low Severity Incidents](#sev4-low-severity-incidents)
4. [Escalation Procedures](#escalation-procedures)
5. [Communication Templates](#communication-templates)
6. [Post-Incident Review](#post-incident-review)
7. [Contact Information](#contact-information)
8. [Appendices](#appendices)

---

## Incident Severity Levels

### SEV1: Critical - Fuel Optimization Failure Causing Plant Shutdown

**Definition**: Complete fuel optimization failure preventing plant operations, data corruption affecting fuel calculations, security breach in fuel management systems, or safety-critical fuel blend miscalculation.

**Response Time**: Immediate (< 5 minutes)

**Indicators**:
- Agent not responding to any optimization requests
- All fuel optimization calculations failing (error rate > 95%)
- Data corruption detected in fuel inventory database
- Security breach confirmed in fuel management systems
- Complete loss of fuel price market data feeds
- Provenance integrity compromised for fuel calculations
- Safety-critical fuel blend calculations returning invalid results
- Plant forced to emergency shutdown due to fuel optimization failure
- Carbon emissions calculations completely unavailable
- Real-time pricing integration total failure

**Required Actions**:
- Page on-call engineer immediately (PagerDuty high-urgency)
- Create war room (Slack: #incident-sev1-gl011)
- Notify VP Engineering within 15 minutes
- Notify Plant Operations Manager immediately
- Update status page immediately
- Start incident timeline documentation
- Engage fuel procurement team for manual backup procedures
- Contact downstream plant control systems team

**Business Impact**:
- No fuel optimization calculations possible
- Plant operations at risk of shutdown
- Regulatory reporting blocked for emissions
- Financial impact > $100,000/hour (plant downtime)
- Potential safety compliance violations
- Manual fuel blending required (inefficient and risky)

**SLA Requirements**:
- First response: < 5 minutes
- War room established: < 10 minutes
- Root cause identified: < 30 minutes
- Mitigation deployed: < 1 hour
- Full resolution: < 4 hours

---

### SEV2: High - Incorrect Fuel Blend Causing Emissions Violation

**Definition**: Significant functionality degraded but workarounds exist; fuel blend recommendations incorrect leading to emissions threshold exceedance; major integration failures affecting multiple connectors.

**Response Time**: 15 minutes

**Indicators**:
- Fuel optimization success rate < 90%
- Fuel blend recommendations causing emissions threshold violations
- Carbon footprint calculations off by > 10%
- Key optimizations timing out (> 60 seconds)
- Fuel price API connectivity issues affecting > 25% of markets
- Historical fuel consumption queries failing
- Procurement optimization recommendations unavailable
- Inventory management calculations producing outliers
- Multiple ERP connector failures simultaneously
- Cost optimization producing negative savings values

**Required Actions**:
- Page on-call engineer
- Create incident channel (Slack: #incident-sev2-gl011)
- Notify Engineering Manager within 30 minutes
- Notify Plant Operations within 1 hour
- Update status page
- Document workarounds for users
- Enable fallback optimization algorithms
- Alert emissions compliance team

**Business Impact**:
- Degraded optimization performance
- Some plants unable to generate fuel reports
- Emissions compliance at risk
- Manual workarounds required
- Financial impact $10,000-$100,000/hour
- Potential regulatory fines for emissions violations

**SLA Requirements**:
- First response: < 15 minutes
- Root cause identified: < 1 hour
- Mitigation deployed: < 2 hours
- Full resolution: < 8 hours

---

### SEV3: Medium - Integration Connector Failure

**Definition**: Minor issues causing inconvenience but not blocking core functionality; single connector failures; performance degradation affecting optimization speed.

**Response Time**: 1 hour

**Indicators**:
- Fuel optimization success rate 90-95%
- Performance degradation (latency +50%)
- Single ERP connector failure (SAP, Oracle, or Workday)
- Cache hit rate < 70% for fuel price data
- Individual fuel market price feed failures
- Non-critical procurement connector errors
- Elevated error logs for specific fuel types
- Partial inventory synchronization failures
- Single region fuel pricing unavailable

**Required Actions**:
- Notify on-call engineer (no page)
- Create tracking ticket in JIRA
- Update internal status dashboard
- Monitor for escalation
- Enable connector circuit breakers
- Use cached pricing data as fallback

**Business Impact**:
- Minor user inconvenience
- Reduced optimization accuracy
- Some manual data entry required
- Financial impact < $10,000/hour
- Limited regional impact

**SLA Requirements**:
- First response: < 1 hour
- Root cause identified: < 4 hours
- Mitigation deployed: < 8 hours
- Full resolution: < 24 hours

---

### SEV4: Low - Performance Degradation

**Definition**: Cosmetic issues, minor bugs, proactive alerts, or optimization enhancements needed; no immediate operational impact.

**Response Time**: Next business day

**Indicators**:
- Low-priority alerts for optimization performance
- Documentation issues in fuel management guides
- UI/UX improvements needed in dashboards
- Proactive capacity alerts for fuel processing
- Minor calculation precision improvements needed
- Enhancement requests for fuel blend algorithms
- Non-critical logging improvements
- Dashboard rendering delays

**Required Actions**:
- Create ticket in backlog
- Schedule for next sprint
- No immediate response required
- Document for future optimization

**Business Impact**:
- Minimal to no operational impact
- User experience improvements deferred
- Technical debt accumulation

**SLA Requirements**:
- First response: < 24 hours
- Resolution: Next sprint planning

---

## Incident Response Process

### Phase 1: Detection (0-5 minutes)

**Automated Detection**:
```bash
# Check Prometheus alerts for GL-011 FUELCRAFT
kubectl get prometheusrules -n gl-011-production | grep FIRING

# View active alerts
curl -s http://prometheus:9090/api/v1/alerts | jq '.data.alerts[] | select(.labels.service=="fuelcraft") | select(.state=="firing")'

# Check PagerDuty incidents
pd incident list --status triggered --service GL-011-FUELCRAFT

# Check Grafana alert status
curl -s http://grafana:3000/api/alerts | jq '.[] | select(.name | contains("fuelcraft"))'
```

**Manual Detection**:
```bash
# Check agent health
curl -s https://api.greenlang.io/v1/fuelcraft/health | jq .

# Check fuel optimization endpoint
curl -X POST https://api.greenlang.io/v1/fuelcraft/optimize \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @test_optimization.json

# Check recent errors
kubectl logs -n gl-011-production -l app=fuelcraft --tail=100 | grep ERROR

# Check fuel price API status
curl -s https://api.greenlang.io/v1/fuelcraft/prices/status | jq .

# Check ERP connector status
curl -s https://api.greenlang.io/v1/fuelcraft/connectors/status | jq .
```

**Initial Assessment Checklist**:
1. Verify alert is legitimate (not false positive)
2. Check status page for related incidents
3. Determine severity level using criteria above
4. Check if incident is already being handled
5. Verify impact scope (single plant, region, or global)
6. Check for upstream dependency failures

---

### Phase 2: Response (5-15 minutes)

**Incident Commander Actions**:

1. **Create Incident Channel**:
```bash
# Slack command
/incident create sev1 "GL-011: Fuel optimization failure - [PLANT_ID]"

# Set topic
/topic GL-011 incident - [Incident Commander: @username] - [Status: Investigating] - [Impact: Plant operations]

# Pin important messages
/pin This is a SEV1 incident. Please keep this channel focused on resolution.
```

2. **Gather Initial Information**:
```bash
# Service status
kubectl get pods -n gl-011-production -o wide

# Recent deployments
kubectl rollout history deployment/fuelcraft -n gl-011-production

# Error rate (last 5 minutes)
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=rate(fuelcraft_optimization_errors_total[5m])/rate(fuelcraft_optimization_requests_total[5m])' | \
  jq '.data.result[0].value[1]'

# Active optimizations
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=fuelcraft_active_optimizations' | \
  jq '.data.result[0].value[1]'

# Resource usage
kubectl top pods -n gl-011-production

# Database connections
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -t -c \
  "SELECT count(*) FROM pg_stat_activity WHERE datname='fuelcraft_production';"

# Cache status
kubectl exec -it redis-0 -n gl-011-production -- redis-cli INFO memory
```

3. **Start Incident Timeline**:
```markdown
## Incident Timeline - GL-011-FUELCRAFT

**Incident ID**: INC-2025-12-001
**Severity**: SEV1
**Status**: Investigating

### Timeline

- **15:23 UTC**: Alert triggered: High fuel optimization error rate
- **15:24 UTC**: Incident Commander assigned: @alice
- **15:25 UTC**: War room created: #incident-sev1-gl011-dec01
- **15:27 UTC**: Initial investigation started
- **15:30 UTC**: Root cause identified: [TBD]
- **15:XX UTC**: Mitigation applied: [TBD]
- **15:XX UTC**: Service restored: [TBD]

### Impact Assessment
- Affected Plants: [List]
- Affected Fuel Types: [List]
- Financial Impact: Estimated $[X]/hour

### Current Status
[Investigation underway / Mitigation in progress / Monitoring]
```

---

### Phase 3: Investigation (15-60 minutes)

**Investigation Checklist**:

- [ ] Check recent deployments (last 4 hours)
- [ ] Review recent configuration changes
- [ ] Check infrastructure status (database, cache, message queue)
- [ ] Review application logs
- [ ] Check external dependencies (fuel price APIs, ERP connectors)
- [ ] Analyze error patterns
- [ ] Check resource utilization (CPU, memory, disk, network)
- [ ] Review metrics and dashboards
- [ ] Check fuel inventory data integrity
- [ ] Verify market price feed status
- [ ] Review optimization algorithm logs
- [ ] Check provenance chain integrity

**Diagnostic Commands**:

```bash
# === DEPLOYMENT CHECKS ===

# Recent deployments
kubectl rollout history deployment/fuelcraft -n gl-011-production | tail -10

# Check ConfigMaps for recent changes
kubectl get configmap fuelcraft-config -n gl-011-production -o yaml | \
  head -50

# Check deployment events
kubectl describe deployment fuelcraft -n gl-011-production | \
  grep -A 20 "Events:"

# === DATABASE CHECKS ===

# Database connectivity
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -c "SELECT 1 AS connection_test;"

# Check for long-running queries
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -c \
  "SELECT pid, now() - pg_stat_activity.query_start AS duration, query, state
   FROM pg_stat_activity
   WHERE (now() - pg_stat_activity.query_start) > interval '1 minute'
   ORDER BY duration DESC;"

# Check database locks
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -c \
  "SELECT blocked_locks.pid AS blocked_pid,
          blocking_locks.pid AS blocking_pid,
          blocked_activity.query AS blocked_query,
          blocking_activity.query AS blocking_query
   FROM pg_catalog.pg_locks blocked_locks
   JOIN pg_catalog.pg_stat_activity blocked_activity
     ON blocked_activity.pid = blocked_locks.pid
   JOIN pg_catalog.pg_locks blocking_locks
     ON blocking_locks.locktype = blocked_locks.locktype
   JOIN pg_catalog.pg_stat_activity blocking_activity
     ON blocking_activity.pid = blocking_locks.pid
   WHERE NOT blocked_locks.granted
   LIMIT 10;"

# === CACHE CHECKS ===

# Cache connectivity (Redis)
kubectl exec -it redis-0 -n gl-011-production -- \
  redis-cli PING

# Cache memory usage
kubectl exec -it redis-0 -n gl-011-production -- \
  redis-cli INFO memory | grep used_memory_human

# Cache key count
kubectl exec -it redis-0 -n gl-011-production -- \
  redis-cli DBSIZE

# === EXTERNAL DEPENDENCY CHECKS ===

# Check fuel price API connections
kubectl logs -n gl-011-production -l app=fuelcraft --tail=500 | \
  grep "FuelPriceConnector"

# Check ERP connector status
kubectl logs -n gl-011-production -l app=fuelcraft --tail=500 | \
  grep "ERPConnector\|SAPConnector\|OracleConnector\|WorkdayConnector"

# Check market data feeds
kubectl exec -it deployment/fuelcraft -n gl-011-production -- \
  curl -v https://fuel-prices.example.com/api/health

# === NETWORK CHECKS ===

# Network connectivity to external services
kubectl exec -it deployment/fuelcraft -n gl-011-production -- \
  curl -v https://erp-gateway.example.com/api/health

# DNS resolution
kubectl exec -it deployment/fuelcraft -n gl-011-production -- \
  nslookup fuel-prices.example.com

# === APPLICATION LOGS ===

# Recent application errors
kubectl logs -n gl-011-production -l app=fuelcraft --tail=500 | \
  grep -E "ERROR|CRITICAL|FATAL" | \
  jq -r '{time: .timestamp, error: .error_message, plant: .plant_id}' 2>/dev/null | \
  tail -50

# Optimization failures
kubectl logs -n gl-011-production -l app=fuelcraft --tail=1000 | \
  grep "OptimizationFailed\|CalculationError" | \
  tail -20

# Fuel blend calculation errors
kubectl logs -n gl-011-production -l app=fuelcraft --tail=1000 | \
  grep "BlendCalculation\|EmissionsViolation" | \
  tail -20

# === METRICS CHECKS ===

# Check optimization latency
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.95, fuelcraft_optimization_duration_seconds_bucket[5m])' | \
  jq '.data.result[0].value[1]'

# Check error rate by type
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=sum by (error_type) (rate(fuelcraft_errors_total[5m]))' | \
  jq '.data.result[] | {error_type: .metric.error_type, rate: .value[1]}'

# Check fuel type processing rates
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=sum by (fuel_type) (rate(fuelcraft_optimizations_total[5m]))' | \
  jq '.data.result[]'
```

---

### Phase 4: Mitigation (Varies)

**Quick Mitigation Actions**:

1. **Scale Up Resources** (if performance issue):
```bash
# Increase replicas
kubectl scale deployment/fuelcraft -n gl-011-production --replicas=8

# Increase resource limits
kubectl set resources deployment/fuelcraft -n gl-011-production \
  --limits=cpu=4000m,memory=8Gi \
  --requests=cpu=2000m,memory=4Gi

# Verify scaling
kubectl get pods -n gl-011-production -l app=fuelcraft --watch
```

2. **Rollback Deployment** (if bad release):
```bash
# Check current revision
kubectl rollout history deployment/fuelcraft -n gl-011-production

# Rollback to previous version
kubectl rollout undo deployment/fuelcraft -n gl-011-production

# Verify rollback
kubectl rollout status deployment/fuelcraft -n gl-011-production

# Confirm version
kubectl get deployment fuelcraft -n gl-011-production \
  -o jsonpath='{.spec.template.spec.containers[0].image}'
```

3. **Clear Cache** (if cache corruption):
```bash
# Flush Redis cache for fuel prices
kubectl exec -it redis-0 -n gl-011-production -- \
  redis-cli KEYS "fuel_price:*" | xargs redis-cli DEL

# Flush optimization cache
kubectl exec -it redis-0 -n gl-011-production -- \
  redis-cli KEYS "optimization:*" | xargs redis-cli DEL

# Full cache flush (if necessary)
kubectl exec -it redis-0 -n gl-011-production -- redis-cli FLUSHDB

# Restart pods to force cache refresh
kubectl rollout restart deployment/fuelcraft -n gl-011-production
```

4. **Enable Fallback Mode** (if external service failing):
```bash
# Enable cached pricing fallback
kubectl set env deployment/fuelcraft -n gl-011-production \
  FUEL_PRICE_FALLBACK_ENABLED=true \
  FUEL_PRICE_CACHE_MAX_AGE_HOURS=24

# Enable default fuel blend ratios
kubectl set env deployment/fuelcraft -n gl-011-production \
  USE_DEFAULT_FUEL_BLENDS=true

# Disable failing ERP connector
kubectl set env deployment/fuelcraft -n gl-011-production \
  SAP_CONNECTOR_ENABLED=false

# Apply changes
kubectl rollout restart deployment/fuelcraft -n gl-011-production
```

5. **Kill Stuck Processes**:
```bash
# Find stuck pods
kubectl get pods -n gl-011-production -l app=fuelcraft | grep -v Running

# Force delete stuck pods
kubectl delete pod fuelcraft-abc123 -n gl-011-production --grace-period=0 --force

# Check for stuck optimization jobs
kubectl exec -it deployment/fuelcraft -n gl-011-production -- \
  python -c "from fuelcraft.jobs import cleanup_stuck_jobs; cleanup_stuck_jobs()"
```

6. **Database Recovery** (if database issue):
```bash
# Kill long-running queries
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -c \
  "SELECT pg_terminate_backend(pid)
   FROM pg_stat_activity
   WHERE (now() - pg_stat_activity.query_start) > interval '10 minutes'
   AND state != 'idle'
   AND pid != pg_backend_pid();"

# Reset database connections
kubectl rollout restart deployment/fuelcraft -n gl-011-production

# Verify connection pool
kubectl exec -it deployment/fuelcraft -n gl-011-production -- \
  python -c "from fuelcraft.db import pool; print(f'Active: {pool.checkedout()}, Available: {pool.checkedin()}')"
```

---

### Phase 5: Resolution (Varies)

**Verification Steps**:

1. **Test Core Functionality**:
```bash
# Submit test fuel optimization
curl -X POST https://api.greenlang.io/v1/fuelcraft/optimize \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "site_id": "SITE-TEST-001",
    "plant_id": "PLANT-TEST-001",
    "request_type": "multi_fuel",
    "energy_demand_mw": 100,
    "available_fuels": ["natural_gas", "coal", "biomass"],
    "optimization_objective": "minimize_cost",
    "time_horizon_hours": 24
  }'

# Verify optimization completed
curl -s https://api.greenlang.io/v1/fuelcraft/jobs/$JOB_ID | jq .

# Test fuel blend calculation
curl -X POST https://api.greenlang.io/v1/fuelcraft/blend \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "fuel_types": ["natural_gas", "biomass"],
    "target_emissions_kg_mwh": 400,
    "energy_output_mw": 50
  }'

# Test carbon footprint calculation
curl -X POST https://api.greenlang.io/v1/fuelcraft/carbon \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "site_id": "SITE-TEST-001",
    "time_period": {
      "start": "2025-12-01T00:00:00Z",
      "end": "2025-12-01T23:59:59Z"
    }
  }'
```

2. **Monitor Error Rates**:
```bash
# Check error rate over last 15 minutes
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=rate(fuelcraft_optimization_errors_total[15m])/rate(fuelcraft_optimization_requests_total[15m])' | \
  jq '.data.result[0].value[1]'

# Should be < 0.01 (1%)

# Watch error rate in real-time
watch -n 30 'curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode "query=rate(fuelcraft_optimization_errors_total[5m])" | \
  jq ".data.result[0].value[1]"'
```

3. **Check Key Metrics**:
```bash
# Optimization latency (p95)
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.95, fuelcraft_optimization_duration_seconds_bucket[5m])' | \
  jq '.data.result[0].value[1]'
# Should be < 30 seconds

# Cache hit rate
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=fuelcraft_cache_hits_total/(fuelcraft_cache_hits_total+fuelcraft_cache_misses_total)' | \
  jq '.data.result[0].value[1]'
# Should be > 0.80

# Active database connections
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -t -c \
  "SELECT count(*) FROM pg_stat_activity WHERE datname='fuelcraft_production';"
# Should be < 80% of pool size

# Fuel price API response time
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.95, fuelcraft_fuel_price_api_duration_seconds_bucket[5m])' | \
  jq '.data.result[0].value[1]'
# Should be < 5 seconds
```

4. **Plant Operations Validation**:
```bash
# Request validation from plant operations team
# Test with real plant accounts and fuel inventories

# Verify fuel blend recommendations are within safe ranges
curl -s https://api.greenlang.io/v1/fuelcraft/validate \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "site_id": "SITE-PROD-001",
    "validation_type": "blend_safety"
  }' | jq .

# Verify emissions calculations are within thresholds
curl -s https://api.greenlang.io/v1/fuelcraft/validate \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "site_id": "SITE-PROD-001",
    "validation_type": "emissions_compliance"
  }' | jq .
```

---

### Phase 6: Communication (Ongoing)

**Status Updates** (Every 30 minutes until resolved):

```markdown
## Status Update - 2025-12-01T16:00:00Z

**Incident**: GL-011-FUELCRAFT Optimization Failure
**Current Status**: Mitigating

**Impact**:
- 3 plants unable to receive fuel optimization recommendations
- Using manual backup procedures for fuel blending
- Estimated financial impact: $50,000/hour

**Actions Taken**:
- Identified root cause: Database connection pool exhaustion
- Scaled database connection pool from 100 to 200
- Restarted application pods to reset connections

**Next Steps**:
- Monitor error rates for 30 minutes
- Verify all plants receiving optimizations
- Conduct full functionality validation

**ETA**: Full resolution expected by 16:30 UTC
```

**Stakeholder Notification**:

```bash
# Post to status page
curl -X POST https://api.statuspage.io/v1/pages/$PAGE_ID/incidents \
  -H "Authorization: OAuth $STATUSPAGE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "incident": {
      "name": "GL-011 FUELCRAFT: Fuel Optimization Service Degradation",
      "status": "investigating",
      "impact_override": "major",
      "body": "We are investigating issues with fuel optimization calculations. Some plants may experience delays in receiving optimization recommendations. Manual backup procedures are available.",
      "component_ids": ["component-fuelcraft"],
      "metadata": {
        "affected_regions": ["us-east-1", "eu-west-1"],
        "workaround_available": true
      }
    }
  }'

# Update status page with progress
curl -X PATCH https://api.statuspage.io/v1/pages/$PAGE_ID/incidents/$INCIDENT_ID \
  -H "Authorization: OAuth $STATUSPAGE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "incident": {
      "status": "identified",
      "body": "Root cause identified as database connection pool exhaustion. Mitigation in progress."
    }
  }'
```

---

### Phase 7: Closure

**Incident Closure Checklist**:

- [ ] All systems operational
- [ ] Error rates back to normal (< 1%)
- [ ] Performance metrics within SLA
- [ ] No active alerts
- [ ] Status page updated to "Resolved"
- [ ] All affected plants notified of resolution
- [ ] Plant operations team confirmed normal operation
- [ ] Emissions compliance verified
- [ ] Stakeholders notified
- [ ] Incident timeline complete
- [ ] Post-incident review scheduled (within 2 business days)

**Closure Communication**:

```markdown
## Incident Resolved - 2025-12-01T16:45:00Z

**Incident ID**: INC-2025-12-001
**Incident Duration**: 1 hour 22 minutes

**Root Cause**:
Database connection pool exhausted due to slow queries from a new optimization algorithm deployment. Queries were not using indexes correctly, causing connection pile-up.

**Resolution**:
- Rolled back problematic optimization algorithm
- Added missing database indexes
- Increased connection pool size as temporary measure
- Implemented query timeout limits

**Prevention**:
- Adding database query performance tests to CI/CD pipeline
- Implementing connection pool monitoring alerts
- Creating runbook for connection pool exhaustion scenarios

**Post-Incident Review**: Scheduled for 2025-12-03 14:00 UTC
**Attendees**: SRE Team, Database Team, Optimization Algorithm Team
```

---

## Incident Scenarios

### SEV1: Critical Incidents

#### Scenario S1.1: Agent Not Responding

**Symptoms**:
- All API requests timing out
- Health check endpoint returning 503
- No logs being generated
- Kubernetes pods in CrashLoopBackOff
- All plants reporting no optimization data

**Severity**: SEV1

**Detection**:
```bash
# Check pod status
kubectl get pods -n gl-011-production -l app=fuelcraft

# Expected: Pods in Running state with READY 1/1
# Actual: CrashLoopBackOff, Error, or ImagePullBackOff

# Check pod events
kubectl describe pod fuelcraft-abc123 -n gl-011-production

# Check container logs from crashed container
kubectl logs fuelcraft-abc123 -n gl-011-production --previous
```

**Immediate Actions**:

1. **Assess Pod Status**:
```bash
# Get pod details
kubectl get pods -n gl-011-production -l app=fuelcraft -o wide

# If CrashLoopBackOff: Check logs for startup errors
kubectl logs -n gl-011-production -l app=fuelcraft --previous --tail=200

# If ImagePullBackOff: Check image availability
kubectl describe pod -n gl-011-production -l app=fuelcraft | grep -A 5 "Image:"

# If Pending: Check node resources
kubectl describe pod -n gl-011-production -l app=fuelcraft | grep -A 10 "Events:"
```

2. **Check Recent Changes**:
```bash
# Recent deployments
kubectl rollout history deployment/fuelcraft -n gl-011-production

# ConfigMap changes
kubectl get configmap fuelcraft-config -n gl-011-production -o yaml | \
  grep -A 5 "metadata:"

# Secret changes (check lastModified)
kubectl get secret fuelcraft-secrets -n gl-011-production -o yaml | \
  grep -A 5 "metadata:"
```

3. **Attempt Recovery**:
```bash
# If recent deployment: Rollback
kubectl rollout undo deployment/fuelcraft -n gl-011-production

# If configuration issue: Reset to known-good config
kubectl apply -f /backups/config/fuelcraft-config-last-good.yaml

# If resource exhaustion: Delete and recreate pods
kubectl delete pods -n gl-011-production -l app=fuelcraft

# Wait for recovery
kubectl rollout status deployment/fuelcraft -n gl-011-production --timeout=300s
```

**Duration**: 5-30 minutes
**Downtime**: Complete until resolved

---

#### Scenario S1.2: Complete Fuel Price API Failure

**Symptoms**:
- All fuel price queries returning errors
- Optimization calculations using stale cached prices
- Multiple market data feeds showing disconnected
- Real-time pricing dashboard showing "No Data"

**Severity**: SEV1

**Detection**:
```bash
# Check fuel price API status
curl -s https://api.greenlang.io/v1/fuelcraft/prices/status | jq .

# Check individual market feeds
kubectl logs -n gl-011-production -l app=fuelcraft --tail=500 | \
  grep "FuelPriceAPI\|MarketData"

# Check connection status to price providers
kubectl exec -it deployment/fuelcraft -n gl-011-production -- \
  python -c "from fuelcraft.prices import check_all_feeds; print(check_all_feeds())"
```

**Immediate Actions**:

1. **Enable Fallback Pricing**:
```bash
# Enable cached price fallback
kubectl set env deployment/fuelcraft -n gl-011-production \
  FUEL_PRICE_FALLBACK_ENABLED=true \
  FUEL_PRICE_CACHE_MAX_AGE_HOURS=48 \
  FUEL_PRICE_FALLBACK_SOURCE=historical_average

# Restart to apply
kubectl rollout restart deployment/fuelcraft -n gl-011-production
```

2. **Notify Price Providers**:
```bash
# Check provider status pages
# - Natural Gas: https://status.ngprice.example.com
# - Coal: https://status.coalprices.example.com
# - Biomass: https://status.biomassprices.example.com

# Log support ticket with providers
echo "Support ticket created for fuel price API outage" >> /var/log/incidents/price_api_outage.log
```

3. **Manual Price Entry** (if extended outage):
```bash
# Enable manual price override mode
kubectl set env deployment/fuelcraft -n gl-011-production \
  MANUAL_PRICE_OVERRIDE_ENABLED=true

# Enter manual prices via API
curl -X POST https://api.greenlang.io/v1/fuelcraft/prices/manual \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prices": [
      {"fuel_type": "natural_gas", "price": 3.50, "unit": "USD/MMBTU"},
      {"fuel_type": "coal", "price": 120.00, "unit": "USD/ton"},
      {"fuel_type": "biomass", "price": 80.00, "unit": "USD/ton"}
    ],
    "valid_until": "2025-12-02T00:00:00Z"
  }'
```

**Duration**: 15 minutes (fallback) to hours (provider recovery)
**Downtime**: None with fallback enabled

---

#### Scenario S1.3: Data Corruption in Fuel Inventory

**Symptoms**:
- Negative fuel inventory values
- Inventory totals not matching physical counts
- Optimization recommending impossible fuel quantities
- Database constraint violations on inventory table

**Severity**: SEV1

**Detection**:
```bash
# Check for data integrity issues
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -c \
  "SELECT site_id, fuel_type, current_quantity_kg
   FROM fuel_inventory
   WHERE current_quantity_kg < 0
      OR current_quantity_kg > max_capacity_kg;"

# Check for recent inventory anomalies
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -c \
  "SELECT * FROM fuel_inventory_log
   WHERE created_at >= NOW() - INTERVAL '1 hour'
     AND (quantity_change_kg > 100000 OR quantity_change_kg < -100000)
   ORDER BY created_at DESC
   LIMIT 20;"
```

**Immediate Actions**:

1. **Stop Writes Immediately**:
```bash
# Scale down to prevent further corruption
kubectl scale deployment/fuelcraft -n gl-011-production --replicas=0

# Or enable read-only mode
kubectl set env deployment/fuelcraft -n gl-011-production \
  DATABASE_READ_ONLY=true
```

2. **Assess Corruption Scope**:
```bash
# Count corrupted records
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -c \
  "SELECT COUNT(*) AS corrupted_inventory_records
   FROM fuel_inventory
   WHERE current_quantity_kg < 0
      OR current_quantity_kg > max_capacity_kg
      OR updated_at > NOW();"

# Check if corruption is spreading
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -c \
  "SELECT table_name, COUNT(*) AS error_count
   FROM information_schema.tables t
   JOIN (SELECT relname, n_dead_tup
         FROM pg_stat_user_tables
         WHERE n_dead_tup > 1000) s
   ON t.table_name = s.relname
   WHERE table_schema = 'public';"
```

3. **Restore from Backup**:
```bash
# Find latest clean backup
aws s3 ls s3://greenlang-backups/fuelcraft/inventory/ --recursive | \
  sort | tail -5

# Restore to point in time before corruption
kubectl exec -it postgres-0 -n gl-011-production -- \
  pg_restore -U fuelcraft -d fuelcraft_production \
    --table=fuel_inventory \
    --clean \
    /backups/fuelcraft_inventory_20251201_120000.dump

# Or use point-in-time recovery
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -c \
  "SELECT pg_wal_replay_resume();"
```

4. **Validate and Resume**:
```bash
# Validate restored data
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -c \
  "SELECT COUNT(*) AS valid_inventory
   FROM fuel_inventory
   WHERE current_quantity_kg >= 0
     AND current_quantity_kg <= max_capacity_kg;"

# Re-enable writes
kubectl set env deployment/fuelcraft -n gl-011-production \
  DATABASE_READ_ONLY=false

# Scale back up
kubectl scale deployment/fuelcraft -n gl-011-production --replicas=4
```

**Duration**: 30-120 minutes
**Downtime**: Yes (during restore)

---

#### Scenario S1.4: Security Breach in Fuel Management System

**Symptoms**:
- Unauthorized API access detected
- Unusual data access patterns
- Credentials potentially compromised
- Suspicious fuel optimization requests

**Severity**: SEV1

**Detection**:
```bash
# Check for unusual API access
kubectl logs -n gl-011-production -l app=fuelcraft --tail=10000 | \
  grep -E "401|403|Unauthorized" | \
  jq -r '{time: .timestamp, ip: .client_ip, endpoint: .path, user: .user_id}' | \
  sort | uniq -c | sort -rn | head -20

# Check for unusual data access
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -c \
  "SELECT usename, client_addr, query, query_start
   FROM pg_stat_activity
   WHERE query NOT LIKE '%pg_stat%'
   ORDER BY query_start DESC
   LIMIT 50;"
```

**Immediate Actions**:

1. **Isolate Affected Systems**:
```bash
# Block external access (keep internal only)
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: fuelcraft-emergency-isolation
  namespace: gl-011-production
spec:
  podSelector:
    matchLabels:
      app: fuelcraft
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: internal
EOF

# Or scale down completely
kubectl scale deployment/fuelcraft -n gl-011-production --replicas=0
```

2. **Rotate All Credentials**:
```bash
# Rotate database password
NEW_DB_PASSWORD=$(openssl rand -base64 32)
kubectl create secret generic fuelcraft-db-credentials \
  -n gl-011-production \
  --from-literal=password=$NEW_DB_PASSWORD \
  --dry-run=client -o yaml | kubectl apply -f -

# Rotate API keys
kubectl create secret generic fuelcraft-api-keys \
  -n gl-011-production \
  --from-literal=price-api-key=$(openssl rand -hex 32) \
  --from-literal=erp-api-key=$(openssl rand -hex 32) \
  --dry-run=client -o yaml | kubectl apply -f -

# Invalidate all JWT tokens
kubectl set env deployment/fuelcraft -n gl-011-production \
  JWT_SECRET=$(openssl rand -hex 64) \
  JWT_ISSUER_VERSION=v2
```

3. **Preserve Evidence**:
```bash
# Export logs for forensics
kubectl logs -n gl-011-production -l app=fuelcraft --tail=100000 > \
  /evidence/fuelcraft_logs_$(date +%Y%m%d_%H%M%S).log

# Export database audit logs
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -c \
  "COPY (SELECT * FROM audit_log WHERE created_at >= NOW() - INTERVAL '24 hours')
   TO '/tmp/audit_export.csv' WITH CSV HEADER;"

kubectl cp gl-011-production/postgres-0:/tmp/audit_export.csv \
  /evidence/audit_$(date +%Y%m%d_%H%M%S).csv

# Notify security team
slack-notify "#security-incidents" \
  "SECURITY: GL-011 FUELCRAFT potential breach. Evidence preserved. Investigation required."
```

4. **Contact Security Team**:
```bash
# Page security on-call
pd trigger --service security-oncall \
  --description "GL-011 FUELCRAFT potential security breach"

# Create security incident ticket
jira create --project SEC --type Incident \
  --summary "GL-011 FUELCRAFT Security Breach Investigation" \
  --priority Highest
```

**Duration**: Hours to days (depends on breach scope)
**Downtime**: Yes (until security clearance)

---

#### Scenario S1.5: Safety-Critical Fuel Blend Miscalculation

**Symptoms**:
- Fuel blend recommendations exceeding safety limits
- Emissions calculations showing impossible values
- Plant safety systems triggering alerts
- Combustion efficiency outside normal range

**Severity**: SEV1

**Detection**:
```bash
# Check for out-of-range blend recommendations
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -c \
  "SELECT optimization_id, site_id, fuel_type, blend_percentage
   FROM optimization_results
   WHERE created_at >= NOW() - INTERVAL '1 hour'
     AND (blend_percentage < 0 OR blend_percentage > 100
          OR blend_percentage IS NULL);"

# Check emissions calculations
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -c \
  "SELECT calculation_id, site_id, co2_emissions_kg_mwh
   FROM emissions_calculations
   WHERE created_at >= NOW() - INTERVAL '1 hour'
     AND (co2_emissions_kg_mwh < 0 OR co2_emissions_kg_mwh > 2000);"
```

**Immediate Actions**:

1. **Block Unsafe Recommendations**:
```bash
# Enable safety-critical mode (rejects out-of-range values)
kubectl set env deployment/fuelcraft -n gl-011-production \
  SAFETY_CRITICAL_MODE=true \
  BLEND_PERCENTAGE_MIN=0 \
  BLEND_PERCENTAGE_MAX=100 \
  EMISSIONS_MAX_KG_MWH=1500

# Restart to apply
kubectl rollout restart deployment/fuelcraft -n gl-011-production
```

2. **Notify Plant Safety Teams**:
```bash
# Send urgent notification
cat > safety_alert.json <<EOF
{
  "alert_type": "FUEL_BLEND_SAFETY",
  "severity": "CRITICAL",
  "message": "GL-011 FUELCRAFT detected potential unsafe fuel blend recommendations. Manual verification required for all blend changes in the last hour.",
  "action_required": "VERIFY_MANUAL",
  "affected_period": "$(date -u -d '-1 hour' +%Y-%m-%dT%H:%M:%SZ) to $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

# Send to all plant safety endpoints
for plant in $(kubectl exec -it deployment/fuelcraft -n gl-011-production -- \
  python -c "from fuelcraft.config import get_plants; print(' '.join(get_plants()))"); do
  curl -X POST https://safety-api.example.com/plants/$plant/alerts \
    -H "Authorization: Bearer $SAFETY_TOKEN" \
    -d @safety_alert.json
done
```

3. **Roll Back Optimization Algorithm**:
```bash
# Identify problematic algorithm version
kubectl logs -n gl-011-production -l app=fuelcraft --tail=1000 | \
  grep "BlendOptimizer" | head -5

# Rollback to previous version
kubectl set env deployment/fuelcraft -n gl-011-production \
  BLEND_OPTIMIZER_VERSION=v1.2.0

# Or full deployment rollback
kubectl rollout undo deployment/fuelcraft -n gl-011-production
```

4. **Validate All Recent Calculations**:
```bash
# Mark affected calculations for review
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -c \
  "UPDATE optimization_results
   SET status = 'PENDING_SAFETY_REVIEW',
       review_reason = 'Safety critical miscalculation incident'
   WHERE created_at >= NOW() - INTERVAL '2 hours'
     AND status = 'COMPLETED';"

# Generate validation report
kubectl exec -it deployment/fuelcraft -n gl-011-production -- \
  python /app/scripts/validate_safety_calculations.py \
    --since "2 hours ago" \
    --output /tmp/safety_validation_report.json

kubectl cp gl-011-production/fuelcraft-abc123:/tmp/safety_validation_report.json \
  ./safety_validation_$(date +%Y%m%d_%H%M%S).json
```

**Duration**: 1-4 hours
**Downtime**: Possible (if algorithm requires rollback)

---

### SEV2: High Severity Incidents

#### Scenario S2.1: Incorrect Fuel Blend Causing Emissions Violation

**Symptoms**:
- Plant emissions exceeding regulatory thresholds
- Blend recommendations not achieving target emissions
- Carbon intensity calculations higher than expected
- Compliance reports showing violations

**Severity**: SEV2

**Detection**:
```bash
# Check recent emissions violations
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -c \
  "SELECT site_id, plant_id, actual_emissions_kg_mwh,
          target_emissions_kg_mwh, violation_percentage
   FROM emissions_compliance
   WHERE created_at >= NOW() - INTERVAL '24 hours'
     AND actual_emissions_kg_mwh > target_emissions_kg_mwh
   ORDER BY violation_percentage DESC;"

# Check blend effectiveness
kubectl logs -n gl-011-production -l app=fuelcraft --tail=2000 | \
  grep "EmissionsTarget\|BlendEffectiveness" | \
  jq 'select(.actual > .target)'
```

**Immediate Actions**:

1. **Adjust Emissions Targets**:
```bash
# Apply conservative safety margin
kubectl set env deployment/fuelcraft -n gl-011-production \
  EMISSIONS_SAFETY_MARGIN_PERCENT=15 \
  EMISSIONS_TARGET_BUFFER_KG_MWH=50
```

2. **Update Emission Factors**:
```bash
# Verify emission factor database is current
kubectl exec -it deployment/fuelcraft -n gl-011-production -- \
  python /app/scripts/update_emission_factors.py \
    --source ipcc_2025 \
    --verify

# Force recalculation with updated factors
kubectl exec -it deployment/fuelcraft -n gl-011-production -- \
  python /app/scripts/recalculate_emissions.py \
    --since "24 hours ago" \
    --force
```

3. **Notify Compliance Team**:
```bash
# Generate compliance report
curl -X POST https://api.greenlang.io/v1/fuelcraft/compliance/report \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "report_type": "emissions_violation",
    "period": "last_24_hours",
    "include_recommendations": true
  }' > compliance_report_$(date +%Y%m%d).json

# Send to compliance team
slack-notify "#compliance" \
  "GL-011 FUELCRAFT: Emissions violation detected. See attached report."
```

**Duration**: 2-8 hours
**Downtime**: None (workaround available)

---

#### Scenario S2.2: Multiple ERP Connector Failures

**Symptoms**:
- SAP, Oracle, and/or Workday connectors failing simultaneously
- Fuel inventory not synchronizing
- Procurement orders not being processed
- Historical consumption data unavailable

**Severity**: SEV2

**Detection**:
```bash
# Check all connector statuses
curl -s https://api.greenlang.io/v1/fuelcraft/connectors/status | jq .

# Check connector error rates
kubectl logs -n gl-011-production -l app=fuelcraft --tail=2000 | \
  grep "ERPConnector" | \
  jq -r '{connector: .connector_type, status: .status, error: .error_message}' | \
  sort | uniq -c

# Check connection pool for each connector
kubectl exec -it deployment/fuelcraft -n gl-011-production -- \
  python -c "
from fuelcraft.connectors import get_all_connectors
for c in get_all_connectors():
    print(f'{c.name}: pool_size={c.pool_size}, active={c.active_connections}, available={c.available_connections}')
"
```

**Immediate Actions**:

1. **Enable Connector Fallbacks**:
```bash
# Enable cached data fallback for inventory
kubectl set env deployment/fuelcraft -n gl-011-production \
  ERP_INVENTORY_FALLBACK_ENABLED=true \
  ERP_INVENTORY_CACHE_MAX_AGE_HOURS=4

# Enable manual override for procurement
kubectl set env deployment/fuelcraft -n gl-011-production \
  ERP_PROCUREMENT_MANUAL_MODE=true
```

2. **Isolate Failing Connectors**:
```bash
# Disable specific failing connector
kubectl set env deployment/fuelcraft -n gl-011-production \
  SAP_CONNECTOR_ENABLED=false \
  ORACLE_CONNECTOR_ENABLED=false

# Keep working connectors active
kubectl set env deployment/fuelcraft -n gl-011-production \
  WORKDAY_CONNECTOR_ENABLED=true
```

3. **Contact ERP Teams**:
```bash
# Check ERP system status
curl -s https://sap-status.example.com/api/status | jq .
curl -s https://oracle-status.example.com/api/status | jq .
curl -s https://workday-status.example.com/api/status | jq .

# Create support tickets
jira create --project SUPPORT --type Incident \
  --summary "GL-011 FUELCRAFT: Multiple ERP connector failures" \
  --description "Connectors failing: SAP, Oracle. Workday still operational."
```

**Duration**: 1-4 hours
**Downtime**: None (with fallbacks)

---

#### Scenario S2.3: Procurement Optimization Returning Invalid Results

**Symptoms**:
- Procurement recommendations with negative quantities
- Cost optimizations showing negative savings
- Vendor recommendations not matching contract terms
- Delivery schedules impossible to meet

**Severity**: SEV2

**Detection**:
```bash
# Check for invalid procurement results
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -c \
  "SELECT procurement_id, site_id, fuel_type,
          recommended_quantity_kg, estimated_savings_usd
   FROM procurement_recommendations
   WHERE created_at >= NOW() - INTERVAL '6 hours'
     AND (recommended_quantity_kg < 0 OR estimated_savings_usd < 0);"

# Check optimization algorithm logs
kubectl logs -n gl-011-production -l app=fuelcraft --tail=2000 | \
  grep "ProcurementOptimizer\|OptimizationError" | \
  tail -50
```

**Immediate Actions**:

1. **Enable Conservative Mode**:
```bash
# Switch to conservative procurement optimizer
kubectl set env deployment/fuelcraft -n gl-011-production \
  PROCUREMENT_OPTIMIZER_MODE=conservative \
  PROCUREMENT_MIN_QUANTITY_KG=0 \
  PROCUREMENT_MAX_LEAD_TIME_DAYS=30
```

2. **Validate Procurement Data**:
```bash
# Run procurement data validation
kubectl exec -it deployment/fuelcraft -n gl-011-production -- \
  python /app/scripts/validate_procurement_data.py \
    --check-contracts \
    --check-vendors \
    --check-pricing

# Fix data inconsistencies
kubectl exec -it deployment/fuelcraft -n gl-011-production -- \
  python /app/scripts/fix_procurement_data.py \
    --dry-run
```

3. **Recalculate Affected Recommendations**:
```bash
# Mark affected recommendations
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -c \
  "UPDATE procurement_recommendations
   SET status = 'INVALIDATED',
       invalidation_reason = 'Algorithm error - SEV2 incident'
   WHERE created_at >= NOW() - INTERVAL '6 hours'
     AND status = 'ACTIVE';"

# Trigger recalculation
kubectl exec -it deployment/fuelcraft -n gl-011-production -- \
  python /app/scripts/recalculate_procurement.py \
    --invalidated-only
```

**Duration**: 2-4 hours
**Downtime**: None

---

#### Scenario S2.4: Carbon Footprint Calculation Errors

**Symptoms**:
- Carbon footprint values off by > 10%
- Scope 1/2/3 emissions not balancing
- Carbon intensity higher than industry benchmarks
- Regulatory reports showing inconsistencies

**Severity**: SEV2

**Detection**:
```bash
# Check for calculation anomalies
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -c \
  "SELECT site_id, calculation_date,
          total_carbon_kg,
          scope1_kg + scope2_kg + scope3_kg AS sum_scopes,
          ABS(total_carbon_kg - (scope1_kg + scope2_kg + scope3_kg)) AS discrepancy
   FROM carbon_footprint_calculations
   WHERE created_at >= NOW() - INTERVAL '24 hours'
     AND ABS(total_carbon_kg - (scope1_kg + scope2_kg + scope3_kg)) > 100
   ORDER BY discrepancy DESC
   LIMIT 20;"

# Check emission factor consistency
kubectl exec -it deployment/fuelcraft -n gl-011-production -- \
  python /app/scripts/check_emission_factors.py
```

**Immediate Actions**:

1. **Verify Emission Factor Database**:
```bash
# Compare with official IPCC/GHG Protocol values
kubectl exec -it deployment/fuelcraft -n gl-011-production -- \
  python /app/scripts/verify_emission_factors.py \
    --source ipcc_2025 \
    --source ghg_protocol \
    --output /tmp/factor_comparison.json

# Update if discrepancies found
kubectl exec -it deployment/fuelcraft -n gl-011-production -- \
  python /app/scripts/update_emission_factors.py \
    --source ipcc_2025 \
    --force
```

2. **Recalculate Carbon Footprint**:
```bash
# Recalculate affected period
kubectl exec -it deployment/fuelcraft -n gl-011-production -- \
  python /app/scripts/recalculate_carbon.py \
    --since "24 hours ago" \
    --verify \
    --output /tmp/carbon_recalc.json
```

3. **Notify Regulatory Compliance**:
```bash
# Generate discrepancy report
curl -X POST https://api.greenlang.io/v1/fuelcraft/carbon/report \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "report_type": "discrepancy_analysis",
    "period": "last_24_hours"
  }' > carbon_discrepancy_$(date +%Y%m%d).json

# Send to regulatory compliance team
slack-notify "#regulatory-compliance" \
  "GL-011 FUELCRAFT: Carbon calculation discrepancies detected. Investigation in progress."
```

**Duration**: 2-6 hours
**Downtime**: None

---

### SEV3: Medium Severity Incidents

#### Scenario S3.1: Single ERP Connector Failure (SAP)

**Symptoms**:
- SAP connector returning timeout errors
- SAP-sourced inventory data stale
- SAP procurement integration failing
- Other connectors working normally

**Severity**: SEV3

**Detection**:
```bash
# Check SAP connector status
curl -s https://api.greenlang.io/v1/fuelcraft/connectors/sap/status | jq .

# Check SAP error logs
kubectl logs -n gl-011-production -l app=fuelcraft --tail=1000 | \
  grep "SAPConnector" | tail -20

# Check SAP connection pool
kubectl exec -it deployment/fuelcraft -n gl-011-production -- \
  python -c "
from fuelcraft.connectors.sap import SAPConnector
c = SAPConnector.get_instance()
print(f'Status: {c.status}, Last Success: {c.last_success}, Errors: {c.error_count}')
"
```

**Immediate Actions**:

1. **Enable SAP Fallback**:
```bash
# Enable cached data for SAP-sourced data
kubectl set env deployment/fuelcraft -n gl-011-production \
  SAP_FALLBACK_TO_CACHE=true \
  SAP_CACHE_MAX_AGE_HOURS=8
```

2. **Reset SAP Connection**:
```bash
# Reset SAP connection pool
kubectl exec -it deployment/fuelcraft -n gl-011-production -- \
  python -c "
from fuelcraft.connectors.sap import SAPConnector
c = SAPConnector.get_instance()
c.reset_connection_pool()
print(f'Connection pool reset. New status: {c.status}')
"

# If reset fails, restart pods
kubectl rollout restart deployment/fuelcraft -n gl-011-production
```

3. **Contact SAP Support**:
```bash
# Check SAP system status
curl -s https://sap-status.internal.example.com/api/health | jq .

# Create support ticket if SAP side issue
jira create --project SAP-SUPPORT \
  --summary "GL-011 FUELCRAFT: SAP connector timeout issues" \
  --priority Medium
```

**Duration**: 1-2 hours
**Downtime**: None

---

#### Scenario S3.2: Fuel Price API Partial Failure

**Symptoms**:
- One or more fuel type prices unavailable
- Natural gas prices updating but coal prices stale
- Price confidence scores below threshold
- Regional price variations not updating

**Severity**: SEV3

**Detection**:
```bash
# Check individual fuel price feeds
for fuel in natural_gas coal biomass fuel_oil; do
  echo "=== $fuel ==="
  curl -s "https://api.greenlang.io/v1/fuelcraft/prices/$fuel/status" | jq .
done

# Check price freshness
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -c \
  "SELECT fuel_type, MAX(updated_at) AS last_update,
          EXTRACT(EPOCH FROM NOW() - MAX(updated_at))/3600 AS hours_stale
   FROM fuel_prices
   GROUP BY fuel_type
   ORDER BY hours_stale DESC;"
```

**Immediate Actions**:

1. **Enable Alternative Price Sources**:
```bash
# Enable secondary price provider
kubectl set env deployment/fuelcraft -n gl-011-production \
  ENABLE_SECONDARY_PRICE_PROVIDER=true \
  SECONDARY_PRICE_PROVIDER=backup_prices_api
```

2. **Use Historical Averages for Missing Fuels**:
```bash
# Enable historical fallback for specific fuels
kubectl set env deployment/fuelcraft -n gl-011-production \
  COAL_PRICE_FALLBACK_MODE=historical_average \
  COAL_PRICE_HISTORICAL_DAYS=7
```

3. **Monitor Price Data Quality**:
```bash
# Set up monitoring for price staleness
kubectl exec -it deployment/fuelcraft -n gl-011-production -- \
  python -c "
from fuelcraft.prices import PriceMonitor
m = PriceMonitor()
print(m.get_freshness_report())
"
```

**Duration**: 30 minutes - 2 hours
**Downtime**: None

---

#### Scenario S3.3: Cache Hit Rate Below Threshold

**Symptoms**:
- Cache hit rate < 70%
- Increased database load
- Higher API latency
- More external API calls than expected

**Severity**: SEV3

**Detection**:
```bash
# Check cache hit rate
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=fuelcraft_cache_hits_total/(fuelcraft_cache_hits_total+fuelcraft_cache_misses_total)' | \
  jq '.data.result[0].value[1]'

# Check Redis memory and evictions
kubectl exec -it redis-0 -n gl-011-production -- redis-cli INFO stats | \
  grep -E "keyspace_hits|keyspace_misses|evicted_keys"

# Check cache key distribution
kubectl exec -it redis-0 -n gl-011-production -- redis-cli INFO keyspace
```

**Immediate Actions**:

1. **Increase Cache TTL**:
```bash
# Increase TTL for frequently accessed data
kubectl set env deployment/fuelcraft -n gl-011-production \
  FUEL_PRICE_CACHE_TTL=7200 \
  OPTIMIZATION_RESULT_CACHE_TTL=3600 \
  INVENTORY_CACHE_TTL=1800
```

2. **Increase Redis Memory**:
```bash
# Check current memory
kubectl exec -it redis-0 -n gl-011-production -- redis-cli INFO memory

# Increase memory limit
kubectl exec -it redis-0 -n gl-011-production -- \
  redis-cli CONFIG SET maxmemory 8gb
```

3. **Warm Critical Caches**:
```bash
# Warm fuel price cache
kubectl exec -it deployment/fuelcraft -n gl-011-production -- \
  python /app/scripts/warm_cache.py \
    --type fuel_prices \
    --all-fuel-types

# Warm frequently accessed optimization results
kubectl exec -it deployment/fuelcraft -n gl-011-production -- \
  python /app/scripts/warm_cache.py \
    --type optimization_results \
    --recent-hours 24
```

**Duration**: 30 minutes - 1 hour
**Downtime**: None

---

#### Scenario S3.4: Performance Degradation (Latency +50%)

**Symptoms**:
- Optimization requests taking 50% longer than baseline
- p95 latency > 45 seconds (baseline: 30 seconds)
- User complaints about slow performance
- No errors but reduced throughput

**Severity**: SEV3

**Detection**:
```bash
# Check current latency
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.95, fuelcraft_optimization_duration_seconds_bucket[5m])' | \
  jq '.data.result[0].value[1]'

# Check resource utilization
kubectl top pods -n gl-011-production -l app=fuelcraft

# Check database query performance
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -c \
  "SELECT query, calls, mean_exec_time, max_exec_time
   FROM pg_stat_statements
   WHERE query NOT LIKE '%pg_stat%'
   ORDER BY mean_exec_time DESC
   LIMIT 10;"
```

**Immediate Actions**:

1. **Scale Up Resources**:
```bash
# Increase pod replicas
kubectl scale deployment/fuelcraft -n gl-011-production --replicas=6

# Increase resource limits
kubectl set resources deployment/fuelcraft -n gl-011-production \
  --limits=cpu=4000m,memory=8Gi
```

2. **Optimize Slow Queries**:
```bash
# Identify and add missing indexes
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -c \
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_optimizations_site_created
   ON optimizations(site_id, created_at DESC);"

# Update query statistics
kubectl exec -it postgres-0 -n gl-011-production -- \
  psql -U fuelcraft -d fuelcraft_production -c "ANALYZE;"
```

3. **Enable Performance Optimizations**:
```bash
# Enable aggressive caching
kubectl set env deployment/fuelcraft -n gl-011-production \
  ENABLE_AGGRESSIVE_CACHING=true \
  QUERY_RESULT_CACHE_ENABLED=true
```

**Duration**: 30 minutes - 2 hours
**Downtime**: None

---

### SEV4: Low Severity Incidents

#### Scenario S4.1: Dashboard Rendering Delays

**Symptoms**:
- KPI dashboard loading slowly
- Charts taking > 5 seconds to render
- Some visualizations timing out
- No impact on core optimization functionality

**Severity**: SEV4

**Detection**:
```bash
# Check dashboard API response times
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.95, fuelcraft_dashboard_api_duration_seconds_bucket[5m])' | \
  jq '.data.result[0].value[1]'

# Check frontend error logs
kubectl logs -n gl-011-production -l app=fuelcraft-frontend --tail=100 | \
  grep "timeout\|error"
```

**Actions**:
- Create ticket in backlog for dashboard optimization
- Schedule for next sprint
- No immediate action required

**Duration**: Next sprint
**Downtime**: None

---

#### Scenario S4.2: Documentation Outdated

**Symptoms**:
- User guide references deprecated features
- API documentation missing new endpoints
- Screenshots showing old UI
- Calculation methodology documentation outdated

**Severity**: SEV4

**Actions**:
- Create documentation update ticket
- Assign to technical writing team
- Schedule for documentation sprint

**Duration**: 1-2 weeks
**Downtime**: None

---

#### Scenario S4.3: Minor Calculation Precision Issues

**Symptoms**:
- Fuel blend percentages showing 4 decimal places instead of 2
- Cost calculations showing minor rounding differences
- No impact on optimization accuracy

**Severity**: SEV4

**Actions**:
- Create precision improvement ticket
- Add to technical debt backlog
- Review during next code review cycle

**Duration**: Next sprint
**Downtime**: None

---

## Escalation Procedures

### Escalation Matrix

| Severity | Initial Response | 30 Minutes | 1 Hour | 2 Hours | 4 Hours |
|----------|------------------|------------|--------|---------|---------|
| SEV1 | On-Call Engineer | Team Lead + VP Engineering | CTO + Customer Success | CEO Briefing | Executive War Room |
| SEV2 | On-Call Engineer | Team Lead | Engineering Manager | VP Engineering | CTO |
| SEV3 | On-Call Engineer | Team Lead (if needed) | Engineering Manager (if needed) | - | - |
| SEV4 | Ticket Creation | Sprint Planning | - | - | - |

### Escalation Triggers

**Escalate Immediately If**:
- Multiple SEV1 incidents occur simultaneously
- Data corruption affects multiple sites
- Security breach confirmed
- Safety-critical system affected
- Resolution not progressing after 30 minutes

**Escalate to Management If**:
- Customer-facing impact > 1 hour
- Financial impact > $100,000
- Regulatory compliance at risk
- Media attention likely

### Escalation Contacts

```yaml
escalation_chain:
  level_1:
    name: "On-Call Engineer"
    pagerduty: "GL-011-Primary"
    slack: "#gl-011-oncall"
    response_time: "5 minutes"

  level_2:
    name: "Team Lead - Fuel Management"
    pagerduty: "GL-011-Secondary"
    email: "fuel-team-lead@greenlang.io"
    phone: "+1-555-FUEL-001"
    response_time: "15 minutes"

  level_3:
    name: "Engineering Manager"
    pagerduty: "Engineering-Management"
    email: "eng-manager@greenlang.io"
    phone: "+1-555-ENG-MGMT"
    response_time: "30 minutes"

  level_4:
    name: "VP Engineering"
    pagerduty: "VP-Engineering"
    email: "vp-engineering@greenlang.io"
    phone: "+1-555-VP-ENG"
    response_time: "1 hour"

  level_5:
    name: "CTO"
    pagerduty: "CTO"
    email: "cto@greenlang.io"
    phone: "+1-555-CTO"
    response_time: "2 hours"
```

### How to Escalate

```bash
# PagerDuty escalation
pd incident escalate --incident-id $INCIDENT_ID --escalation-policy GL-011-Escalation

# Slack escalation
/incident escalate $INCIDENT_ID "Root cause not identified after 30 minutes"

# Manual escalation
# 1. Call the next level in the escalation chain
# 2. Brief them on: Current status, Impact, Actions taken, Help needed
# 3. Update incident channel with escalation notification
```

---

## Communication Templates

### SEV1 Initial Notification

```markdown
## URGENT: GL-011 FUELCRAFT Critical Incident

**Incident ID**: INC-YYYY-MM-XXX
**Severity**: SEV1 - Critical
**Status**: Investigating
**Incident Commander**: @username

### Impact
- Fuel optimization service is DOWN
- All plants unable to receive optimization recommendations
- Manual backup procedures REQUIRED

### Affected Systems
- GL-011 FUELCRAFT Optimization API
- Fuel Blend Calculation Service
- Carbon Footprint Calculator

### Actions in Progress
- On-call engineer investigating
- War room created: #incident-sev1-gl011-YYYY-MM-DD

### User Action Required
- Use manual fuel blending procedures until service restored
- Contact plant operations for backup processes

### Updates
- Next update in 15 minutes or upon significant change

### Contact
- Incident Channel: #incident-sev1-gl011-YYYY-MM-DD
- On-Call: @oncall-engineer
```

### SEV2 Initial Notification

```markdown
## GL-011 FUELCRAFT Service Degradation

**Incident ID**: INC-YYYY-MM-XXX
**Severity**: SEV2 - High
**Status**: Investigating

### Impact
- Fuel optimization service experiencing errors
- Some optimization requests failing
- Workarounds available

### Affected Systems
- [List affected components]

### Workaround
[Describe workaround if available]

### Actions in Progress
- On-call engineer investigating
- Root cause analysis underway

### Updates
- Next update in 30 minutes or upon significant change

### Contact
- Incident Channel: #incident-sev2-gl011
```

### Status Update Template

```markdown
## Status Update - [TIMESTAMP]

**Incident**: GL-011 FUELCRAFT [Brief Description]
**Status**: [Investigating / Identified / Monitoring / Resolved]

### Current State
[Brief description of current state]

### Actions Since Last Update
- [Action 1]
- [Action 2]

### Next Steps
- [Next action 1]
- [Next action 2]

### ETA
[Estimated time to resolution if known]

### Metrics
- Error Rate: X%
- Latency p95: Xs
- Affected Plants: X

### Next Update
[Time of next update]
```

### Resolution Notification

```markdown
## RESOLVED: GL-011 FUELCRAFT [Brief Description]

**Incident ID**: INC-YYYY-MM-XXX
**Duration**: X hours Y minutes
**Resolution Time**: [TIMESTAMP]

### Summary
[Brief summary of what happened]

### Root Cause
[Root cause description]

### Resolution
[How it was fixed]

### Impact
- Total affected plants: X
- Duration of impact: X hours
- No data loss occurred / Data recovery completed

### Preventive Measures
- [Measure 1]
- [Measure 2]

### Post-Incident Review
Scheduled for [DATE TIME]

### Contact
Questions: #gl-011-support
```

---

## Post-Incident Review

### Post-Incident Review Process

1. **Schedule Review** (within 2 business days):
```bash
# Create calendar invite
# Include: Incident Commander, On-Call Engineer, Team Lead, Affected Team Members

# Create PIR document
cp templates/pir_template.md pir/PIR-YYYY-MM-XXX.md
```

2. **Prepare Timeline**:
```bash
# Export incident timeline from Slack
/incident export $INCIDENT_ID --format markdown

# Export metrics during incident
curl -s http://prometheus:9090/api/v1/query_range \
  --data-urlencode 'query=fuelcraft_optimization_errors_total' \
  --data-urlencode "start=$INCIDENT_START_TIME" \
  --data-urlencode "end=$INCIDENT_END_TIME" \
  --data-urlencode 'step=60' > incident_metrics.json
```

3. **Conduct Review** (blameless):
- What happened? (timeline)
- What was the impact?
- What was the root cause?
- What went well?
- What could be improved?
- Action items and owners

### Post-Incident Review Template

```markdown
# Post-Incident Review: GL-011 FUELCRAFT [Brief Description]

**Incident ID**: INC-YYYY-MM-XXX
**Date**: YYYY-MM-DD
**Duration**: X hours Y minutes
**Severity**: SEVX
**Author**: @username
**Review Date**: YYYY-MM-DD

---

## Executive Summary

[2-3 sentence summary of the incident, impact, and resolution]

---

## Timeline

| Time (UTC) | Event |
|------------|-------|
| HH:MM | Alert triggered |
| HH:MM | On-call engineer paged |
| HH:MM | Investigation started |
| HH:MM | Root cause identified |
| HH:MM | Mitigation applied |
| HH:MM | Service restored |
| HH:MM | Incident closed |

---

## Impact

### User Impact
- Number of affected users/plants: X
- Duration of impact: X hours
- User-facing error rate: X%

### Business Impact
- Financial impact: $X
- SLA violations: Yes/No
- Regulatory implications: None/[Describe]

### Data Impact
- Data loss: None / [Describe]
- Data corruption: None / [Describe]

---

## Root Cause

### Primary Cause
[Detailed description of the root cause]

### Contributing Factors
1. [Factor 1]
2. [Factor 2]

### Root Cause Analysis
[5 Whys or other RCA technique]

---

## Detection

### How was the incident detected?
- [ ] Automated alerting
- [ ] User report
- [ ] Internal testing
- [ ] Other: [Describe]

### Detection Time
Time from incident start to detection: X minutes

### Detection Gaps
[What could have detected this sooner?]

---

## Response

### What went well?
1. [Positive 1]
2. [Positive 2]

### What could be improved?
1. [Improvement 1]
2. [Improvement 2]

### Response Time Analysis
- Time to acknowledge: X minutes
- Time to diagnose: X minutes
- Time to mitigate: X minutes
- Time to resolve: X minutes

---

## Action Items

| ID | Action | Owner | Priority | Due Date | Status |
|----|--------|-------|----------|----------|--------|
| 1 | [Action description] | @username | High | YYYY-MM-DD | Open |
| 2 | [Action description] | @username | Medium | YYYY-MM-DD | Open |

### Categories
- **Prevent**: Actions to prevent this from happening again
- **Detect**: Actions to detect this faster
- **Mitigate**: Actions to reduce impact when it happens
- **Process**: Process improvements

---

## Lessons Learned

1. [Lesson 1]
2. [Lesson 2]

---

## Appendices

### A. Relevant Logs
[Link to log exports]

### B. Metrics During Incident
[Link to dashboard/metrics]

### C. Communication Archives
[Link to Slack exports]

### D. Related Documentation
[Links to relevant runbooks, architecture docs]

---

## Sign-Off

- [ ] Incident Commander: @username
- [ ] Team Lead: @username
- [ ] Engineering Manager: @username
```

---

## Contact Information

### On-Call Rotations

```yaml
on_call_schedules:
  primary:
    name: "GL-011 Primary On-Call"
    pagerduty_schedule_id: "P123456"
    rotation: "weekly"
    escalation_after: "15 minutes"

  secondary:
    name: "GL-011 Secondary On-Call"
    pagerduty_schedule_id: "P789012"
    rotation: "weekly"
    escalation_after: "30 minutes"

  management:
    name: "Engineering Management"
    pagerduty_schedule_id: "P345678"
    rotation: "weekly"
    escalation_after: "1 hour"
```

### Key Contacts

| Role | Name | Email | Phone | Slack |
|------|------|-------|-------|-------|
| Team Lead | [Name] | fuel-lead@greenlang.io | +1-555-FUEL-001 | @fuel-lead |
| Engineering Manager | [Name] | eng-mgr@greenlang.io | +1-555-ENG-001 | @eng-mgr |
| Database Admin | [Name] | dba@greenlang.io | +1-555-DBA-001 | @dba |
| Security | [Name] | security@greenlang.io | +1-555-SEC-001 | @security |
| Customer Success | [Name] | cs@greenlang.io | +1-555-CS-001 | @customer-success |

### External Contacts

| Vendor | Service | Support Email | Support Phone | Status Page |
|--------|---------|---------------|---------------|-------------|
| AWS | Infrastructure | aws-support@amazon.com | - | status.aws.amazon.com |
| Fuel Price API | Market Data | support@fuelprices.com | +1-555-PRICE-1 | status.fuelprices.com |
| SAP | ERP Integration | sap-support@sap.com | +1-555-SAP-001 | status.sap.com |
| Oracle | ERP Integration | oracle-support@oracle.com | +1-555-ORA-001 | status.oracle.com |

### Communication Channels

| Channel | Purpose | Members |
|---------|---------|---------|
| #gl-011-team | Team communication | All GL-011 team members |
| #gl-011-oncall | On-call coordination | On-call engineers |
| #gl-011-alerts | Automated alerts | All + bots |
| #incidents | Cross-team incidents | SRE + Engineering |
| #incident-sev1-gl011 | SEV1 war room (created per incident) | As needed |

---

## Appendices

### Appendix A: Quick Reference Commands

```bash
# === HEALTH CHECKS ===

# Full health check
curl -s https://api.greenlang.io/v1/fuelcraft/health | jq .

# Check specific component
curl -s https://api.greenlang.io/v1/fuelcraft/health/database | jq .
curl -s https://api.greenlang.io/v1/fuelcraft/health/cache | jq .
curl -s https://api.greenlang.io/v1/fuelcraft/health/connectors | jq .

# === POD MANAGEMENT ===

# List pods
kubectl get pods -n gl-011-production -l app=fuelcraft

# Describe pod
kubectl describe pod fuelcraft-xxx -n gl-011-production

# Get logs
kubectl logs -n gl-011-production -l app=fuelcraft --tail=100

# Get previous container logs
kubectl logs fuelcraft-xxx -n gl-011-production --previous

# === DEPLOYMENT ===

# Check deployment status
kubectl rollout status deployment/fuelcraft -n gl-011-production

# Rollback
kubectl rollout undo deployment/fuelcraft -n gl-011-production

# Scale
kubectl scale deployment/fuelcraft -n gl-011-production --replicas=8

# === DATABASE ===

# Connect to database
kubectl exec -it postgres-0 -n gl-011-production -- psql -U fuelcraft -d fuelcraft_production

# Check connections
kubectl exec -it postgres-0 -n gl-011-production -- psql -U fuelcraft -d fuelcraft_production -c "SELECT count(*) FROM pg_stat_activity;"

# === CACHE ===

# Redis PING
kubectl exec -it redis-0 -n gl-011-production -- redis-cli PING

# Flush cache
kubectl exec -it redis-0 -n gl-011-production -- redis-cli FLUSHDB

# === METRICS ===

# Error rate
curl -s http://prometheus:9090/api/v1/query --data-urlencode 'query=rate(fuelcraft_optimization_errors_total[5m])'

# Latency p95
curl -s http://prometheus:9090/api/v1/query --data-urlencode 'query=histogram_quantile(0.95, fuelcraft_optimization_duration_seconds_bucket[5m])'
```

### Appendix B: Runbook Cross-References

| Situation | Related Runbook |
|-----------|-----------------|
| Performance issues | [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) |
| Need to rollback | [ROLLBACK_PROCEDURE.md](./ROLLBACK_PROCEDURE.md) |
| Need to scale | [SCALING_GUIDE.md](./SCALING_GUIDE.md) |
| Scheduled maintenance | [MAINTENANCE.md](./MAINTENANCE.md) |

### Appendix C: Alert Reference

| Alert Name | Severity | Threshold | Runbook Section |
|------------|----------|-----------|-----------------|
| FuelcraftHighErrorRate | SEV1 | > 10% for 5m | [SEV1 Incidents](#sev1-critical-incidents) |
| FuelcraftLatencyHigh | SEV2 | p95 > 60s for 5m | [Performance Issues](#scenario-s34-performance-degradation-latency-50) |
| FuelcraftPodCrashLoop | SEV1 | Any pod CrashLoopBackOff | [Agent Not Responding](#scenario-s11-agent-not-responding) |
| FuelcraftDatabaseDown | SEV1 | Database unreachable | [Database Issues](#database-recovery-if-database-issue) |
| FuelcraftCacheDown | SEV2 | Redis unreachable | [Cache Issues](#clear-cache-if-cache-corruption) |
| FuelcraftConnectorFailure | SEV3 | Any connector failing | [Connector Failures](#scenario-s31-single-erp-connector-failure-sap) |

### Appendix D: Recovery Time Objectives

| Component | RTO | RPO | Notes |
|-----------|-----|-----|-------|
| Optimization API | 15 minutes | 0 (stateless) | Scale up or rollback |
| Database | 30 minutes | 5 minutes | Point-in-time recovery |
| Cache | 5 minutes | N/A (can rebuild) | Warm after recovery |
| ERP Connectors | 1 hour | N/A | Fallback to cache |
| Price Feeds | 30 minutes | N/A | Fallback to cached/manual |

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-01 | GreenLang SRE Team | Initial release |

---

**End of Document**

*For questions or updates to this runbook, contact: sre-team@greenlang.io*
