# GL-009 THERMALIQ Incident Response Runbook

**Agent**: GL-009 THERMALIQ ThermalEfficiencyCalculator
**Version**: 1.0.0
**Last Updated**: 2025-11-26
**Owner**: GreenLang SRE Team
**On-Call Rotation**: PagerDuty Schedule "GL-009-Primary"

---

## Table of Contents

1. [Incident Severity Levels](#incident-severity-levels)
2. [Incident Response Process](#incident-response-process)
3. [Incident Scenarios](#incident-scenarios)
4. [Escalation Procedures](#escalation-procedures)
5. [Communication Templates](#communication-templates)
6. [Post-Incident Review](#post-incident-review)

---

## Incident Severity Levels

### SEV1: Critical - Complete Service Outage

**Definition**: Complete calculation failure, data corruption, or security breach affecting all users.

**Response Time**: Immediate (< 5 minutes)

**Indicators**:
- Agent not responding to any requests
- All calculation requests failing (error rate > 95%)
- Data corruption detected in production database
- Security breach confirmed
- Complete loss of energy metering data
- Provenance integrity compromised

**Required Actions**:
- Page on-call engineer immediately
- Create war room (Slack: #incident-sev1-gl009)
- Notify VP Engineering within 15 minutes
- Update status page immediately
- Start incident timeline documentation

**Business Impact**:
- No thermal efficiency calculations possible
- Regulatory reporting blocked
- Financial impact > $10,000/hour

---

### SEV2: Major - Partial Service Degradation

**Definition**: Significant functionality degraded but workarounds exist.

**Response Time**: 15 minutes

**Indicators**:
- Calculation success rate < 90%
- Key calculations timing out (> 30 seconds)
- Energy meter connectivity issues affecting > 25% of facilities
- Historical data queries failing
- Sankey diagram generation failing
- Benchmark comparisons unavailable

**Required Actions**:
- Page on-call engineer
- Create incident channel (Slack: #incident-sev2-gl009)
- Notify Engineering Manager within 30 minutes
- Update status page
- Document workarounds for users

**Business Impact**:
- Degraded calculation performance
- Some facilities unable to generate reports
- Manual workarounds required
- Financial impact $1,000-$10,000/hour

---

### SEV3: Minor - Service Issues

**Definition**: Minor issues causing inconvenience but not blocking core functionality.

**Response Time**: 1 hour

**Indicators**:
- Calculation success rate 90-95%
- Performance degradation (latency +50%)
- Cache hit rate < 70%
- Individual energy meter failures
- Non-critical connector errors
- Elevated error logs

**Required Actions**:
- Notify on-call engineer (no page)
- Create tracking ticket
- Update internal status
- Monitor for escalation

**Business Impact**:
- Minor user inconvenience
- Reduced performance
- Financial impact < $1,000/hour

---

### SEV4: Informational

**Definition**: Cosmetic issues, minor bugs, or proactive alerts.

**Response Time**: Next business day

**Indicators**:
- Low-priority alerts
- Documentation issues
- UI/UX improvements needed
- Proactive capacity alerts

**Required Actions**:
- Create ticket in backlog
- Schedule for next sprint
- No immediate response required

**Business Impact**:
- Minimal to no impact

---

## Incident Response Process

### Phase 1: Detection (0-5 minutes)

**Automated Detection**:
```bash
# Check Prometheus alerts
kubectl get prometheusrules -n gl-009-production | grep FIRING

# View active alerts
curl -s http://prometheus:9090/api/v1/alerts | jq '.data.alerts[] | select(.state=="firing")'

# Check PagerDuty incidents
pd incident list --status triggered --service GL-009
```

**Manual Detection**:
```bash
# Check agent health
curl https://api.greenlang.io/v1/thermaliq/health

# Check calculation endpoint
curl -X POST https://api.greenlang.io/v1/thermaliq/calculate \
  -H "Authorization: Bearer $TOKEN" \
  -d @test_calculation.json

# Check recent errors
kubectl logs -n gl-009-production -l app=thermaliq --tail=100 | grep ERROR
```

**Initial Assessment**:
1. Verify alert is legitimate (not false positive)
2. Check status page for related incidents
3. Determine severity level
4. Check if incident is already being handled

---

### Phase 2: Response (5-15 minutes)

**Incident Commander Actions**:

1. **Create Incident Channel**:
```bash
# Slack command
/incident create sev1 "GL-009: Complete calculation failure"

# Set topic
/topic GL-009 incident - [Incident Commander: @username] - [Status: Investigating]
```

2. **Gather Initial Information**:
```bash
# Service status
kubectl get pods -n gl-009-production -o wide

# Recent deployments
kubectl rollout history deployment/thermaliq -n gl-009-production

# Error rate
rate(thermaliq_calculation_errors_total[5m])

# Active calculations
thermaliq_active_calculations

# Resource usage
kubectl top pods -n gl-009-production
```

3. **Start Incident Timeline**:
```markdown
## Incident Timeline

- **15:23 UTC**: Alert triggered: High calculation error rate
- **15:24 UTC**: Incident Commander assigned: @alice
- **15:25 UTC**: Initial investigation started
- **15:27 UTC**: Root cause identified: Database connection pool exhausted
```

---

### Phase 3: Investigation (15-60 minutes)

**Investigation Checklist**:

- [ ] Check recent deployments (last 4 hours)
- [ ] Review recent configuration changes
- [ ] Check infrastructure status (database, cache, message queue)
- [ ] Review application logs
- [ ] Check external dependencies (energy meters, historians)
- [ ] Analyze error patterns
- [ ] Check resource utilization (CPU, memory, disk, network)
- [ ] Review metrics and dashboards

**Diagnostic Commands**:

```bash
# Recent deployments
kubectl rollout history deployment/thermaliq -n gl-009-production | tail -5

# Check ConfigMaps
kubectl get configmap thermaliq-config -n gl-009-production -o yaml

# Database connectivity
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  psql $DATABASE_URL -c "SELECT 1"

# Cache connectivity (Redis)
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  redis-cli -h redis-service PING

# Check energy meter connections
kubectl logs -n gl-009-production -l app=thermaliq --tail=500 | \
  grep "EnergyMeterConnector"

# Memory analysis
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python -c "import sys; print(sys.getsizeof(...))"

# Network connectivity
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  curl -v https://historian.example.com/api/health
```

---

### Phase 4: Mitigation (Varies)

**Quick Mitigation Actions**:

1. **Scale Up Resources** (if performance issue):
```bash
# Increase replicas
kubectl scale deployment/thermaliq -n gl-009-production --replicas=6

# Increase resource limits
kubectl set resources deployment/thermaliq -n gl-009-production \
  --limits=cpu=4000m,memory=8Gi
```

2. **Rollback Deployment** (if bad release):
```bash
# Rollback to previous version
kubectl rollout undo deployment/thermaliq -n gl-009-production

# Verify rollback
kubectl rollout status deployment/thermaliq -n gl-009-production
```

3. **Clear Cache** (if cache corruption):
```bash
# Flush Redis cache
kubectl exec -it redis-0 -n gl-009-production -- redis-cli FLUSHDB

# Restart pods to force cache refresh
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

4. **Kill Stuck Processes**:
```bash
# Find stuck pods
kubectl get pods -n gl-009-production | grep NotReady

# Force delete stuck pods
kubectl delete pod thermaliq-abc123 -n gl-009-production --grace-period=0 --force
```

5. **Circuit Breaker Activation** (if external service failing):
```bash
# Update ConfigMap to disable failing connector
kubectl edit configmap thermaliq-config -n gl-009-production

# Set: connectors.historian.enabled: false

# Reload configuration
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

---

### Phase 5: Resolution (Varies)

**Verification Steps**:

1. **Test Core Functionality**:
```bash
# Submit test calculation
curl -X POST https://api.greenlang.io/v1/thermaliq/calculate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "facility_id": "FAC-TEST-001",
    "time_period": {
      "start": "2025-11-01T00:00:00Z",
      "end": "2025-11-01T01:00:00Z"
    }
  }'

# Verify calculation completed
curl https://api.greenlang.io/v1/thermaliq/calculations/$CALC_ID
```

2. **Monitor Error Rates**:
```bash
# Check error rate over last 15 minutes
rate(thermaliq_calculation_errors_total[15m])

# Should be < 0.01 (1%)
```

3. **Check Key Metrics**:
```bash
# Calculation latency (p95)
histogram_quantile(0.95, thermaliq_calculation_duration_seconds_bucket)

# Should be < 10 seconds

# Cache hit rate
thermaliq_cache_hits_total / (thermaliq_cache_hits_total + thermaliq_cache_misses_total)

# Should be > 0.80

# Active connections
thermaliq_database_connections_active

# Should be < 80% of pool size
```

4. **User Acceptance Testing**:
```bash
# Request UAT from customer success team
# Test with real user accounts and facilities
```

---

### Phase 6: Communication (Ongoing)

**Status Updates** (Every 30 minutes until resolved):

```markdown
## Status Update - [Timestamp]

**Current Status**: [Investigating / Mitigating / Resolved]

**Impact**: [Brief description of user impact]

**Actions Taken**:
- [Action 1]
- [Action 2]

**Next Steps**:
- [Next step 1]
- [Next step 2]

**ETA**: [Expected resolution time]
```

**Stakeholder Notification**:

```bash
# Post to status page
curl -X POST https://api.statuspage.io/v1/incidents \
  -H "Authorization: OAuth $STATUSPAGE_TOKEN" \
  -d '{
    "incident": {
      "name": "GL-009 Calculation Service Degradation",
      "status": "investigating",
      "impact_override": "major",
      "body": "We are investigating issues with thermal efficiency calculations..."
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
- [ ] Stakeholders notified
- [ ] Incident timeline complete
- [ ] Post-incident review scheduled (within 2 business days)

**Closure Communication**:

```markdown
## Incident Resolved - [Timestamp]

**Incident Duration**: [X hours Y minutes]

**Root Cause**: [Brief summary]

**Resolution**: [What was done to fix]

**Prevention**: [What will be done to prevent recurrence]

**Post-Incident Review**: Scheduled for [Date/Time]
```

---

## Incident Scenarios

### Scenario 1: Agent Not Responding

**Symptoms**:
- All API requests timing out
- Health check endpoint returning 503
- No logs being generated
- Kubernetes pods in CrashLoopBackOff

**Severity**: SEV1

**Detection**:
```bash
# Check pod status
kubectl get pods -n gl-009-production -l app=thermaliq

# Check pod events
kubectl describe pod thermaliq-abc123 -n gl-009-production

# Check container logs
kubectl logs thermaliq-abc123 -n gl-009-production --previous
```

**Immediate Actions**:

1. **Check Pod Status**:
```bash
# Get pod details
kubectl get pods -n gl-009-production -l app=thermaliq -o wide

# Expected: Pods should be Running with READY 1/1
# If: CrashLoopBackOff, Error, ImagePullBackOff -> Follow specific sub-procedure
```

2. **Review Recent Changes**:
```bash
# Check recent deployments
kubectl rollout history deployment/thermaliq -n gl-009-production

# Check ConfigMap changes
kubectl get configmap thermaliq-config -n gl-009-production -o yaml | \
  grep -A 5 "metadata:"
```

3. **Check Resource Availability**:
```bash
# Node resources
kubectl top nodes

# Check if nodes are schedulable
kubectl get nodes

# Check for pod evictions
kubectl get events -n gl-009-production --sort-by='.lastTimestamp' | \
  grep Evicted
```

**Investigation Steps**:

1. **Analyze Container Logs**:
```bash
# Get logs from crashed container
kubectl logs thermaliq-abc123 -n gl-009-production --previous --tail=200

# Common issues to look for:
# - "Connection refused" -> Database/Redis unavailable
# - "Out of memory" -> Memory limit too low
# - "Cannot bind to port" -> Port already in use
# - "Config validation failed" -> Invalid configuration
# - "Import error" -> Missing dependencies
```

2. **Check Dependencies**:
```bash
# Database connectivity
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  nc -zv postgres-service 5432

# Redis connectivity
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  nc -zv redis-service 6379

# External historian
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  curl -v --max-time 5 https://historian.example.com/health
```

3. **Check Startup Probe Configuration**:
```bash
# View probe configuration
kubectl get deployment thermaliq -n gl-009-production -o yaml | \
  grep -A 10 startupProbe

# If failureThreshold too low, pods may be killed prematurely
```

**Resolution Procedures**:

**Sub-Procedure A: Database Connection Failed**
```bash
# Check database service
kubectl get service postgres-service -n gl-009-production

# Check database pod
kubectl get pods -n gl-009-production -l app=postgres

# Test database connectivity
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c "SELECT 1"

# If database is down, restart it
kubectl rollout restart statefulset/postgres -n gl-009-production

# Wait for database to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n gl-009-production --timeout=300s

# Restart GL-009 pods
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Sub-Procedure B: Out of Memory**
```bash
# Check memory usage before crash
kubectl describe pod thermaliq-abc123 -n gl-009-production | grep -A 5 "Last State"

# Increase memory limits
kubectl set resources deployment/thermaliq -n gl-009-production \
  --limits=memory=8Gi --requests=memory=4Gi

# Wait for rollout
kubectl rollout status deployment/thermaliq -n gl-009-production

# Monitor memory usage
kubectl top pods -n gl-009-production -l app=thermaliq --watch
```

**Sub-Procedure C: Configuration Error**
```bash
# Get current configuration
kubectl get configmap thermaliq-config -n gl-009-production -o yaml > config_backup.yaml

# Validate configuration
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python -c "import yaml; yaml.safe_load(open('/etc/config/config.yaml'))"

# If invalid, restore from backup
kubectl apply -f config_backup_previous.yaml

# Reload configuration
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Sub-Procedure D: Image Pull Failed**
```bash
# Check image pull error
kubectl describe pod thermaliq-abc123 -n gl-009-production | \
  grep -A 10 "Failed to pull image"

# Common causes:
# - Image doesn't exist
# - Registry authentication failed
# - Network connectivity to registry

# Verify image exists
docker pull ghcr.io/greenlang/thermaliq:v1.2.3

# Check image pull secret
kubectl get secret regcred -n gl-009-production -o yaml

# If authentication issue, recreate secret
kubectl delete secret regcred -n gl-009-production
kubectl create secret docker-registry regcred -n gl-009-production \
  --docker-server=ghcr.io \
  --docker-username=$GITHUB_USER \
  --docker-password=$GITHUB_TOKEN
```

**Post-Resolution**:
```bash
# Verify all pods running
kubectl get pods -n gl-009-production -l app=thermaliq

# Check health endpoint
curl https://api.greenlang.io/v1/thermaliq/health

# Run smoke test
./scripts/smoke_test.sh production

# Monitor for 30 minutes
watch -n 30 'kubectl get pods -n gl-009-production -l app=thermaliq'
```

---

### Scenario 2: Calculation Timeout Failures

**Symptoms**:
- Calculations timing out after 30 seconds
- Increased p95/p99 latency
- Queue depth increasing
- Users reporting slow responses

**Severity**: SEV2

**Detection**:
```bash
# Check timeout rate
rate(thermaliq_calculation_timeouts_total[5m])

# Check latency percentiles
histogram_quantile(0.95, thermaliq_calculation_duration_seconds_bucket)
histogram_quantile(0.99, thermaliq_calculation_duration_seconds_bucket)

# Check active calculations
thermaliq_active_calculations

# Check queue depth
thermaliq_calculation_queue_depth
```

**Immediate Actions**:

1. **Identify Slow Calculations**:
```bash
# Get recent timeout logs
kubectl logs -n gl-009-production -l app=thermaliq --tail=500 | \
  grep "Calculation timeout" | \
  jq -r '.facility_id, .calculation_id, .duration'

# Check for patterns:
# - Specific facilities timing out?
# - Specific time ranges?
# - Specific calculation types?
```

2. **Check Resource Utilization**:
```bash
# CPU usage
kubectl top pods -n gl-009-production -l app=thermaliq

# If CPU > 80%, scale up
kubectl scale deployment/thermaliq -n gl-009-production --replicas=8
```

3. **Check External Dependencies**:
```bash
# Historian response time
curl -w "@curl-format.txt" -o /dev/null -s \
  https://historian.example.com/api/query?facility=FAC001

# Energy meter connectivity
kubectl logs -n gl-009-production -l app=thermaliq --tail=200 | \
  grep "EnergyMeter.*timeout"
```

**Investigation Steps**:

1. **Analyze Slow Query Logs**:
```bash
# Check database slow query log
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT query, calls, mean_exec_time, max_exec_time
   FROM pg_stat_statements
   ORDER BY mean_exec_time DESC
   LIMIT 20;"
```

2. **Check Historian Performance**:
```bash
# Test historian query
time curl -X POST https://historian.example.com/api/query \
  -H "Authorization: Bearer $HISTORIAN_TOKEN" \
  -d '{
    "facility_id": "FAC-001",
    "start_time": "2025-11-01T00:00:00Z",
    "end_time": "2025-11-01T01:00:00Z",
    "tags": ["temperature", "flow_rate", "pressure"]
  }'

# Should complete in < 5 seconds
```

3. **Analyze Calculation Complexity**:
```bash
# Check data point count per calculation
kubectl logs -n gl-009-production -l app=thermaliq --tail=500 | \
  grep "Data points retrieved" | \
  jq -r '.data_point_count' | \
  sort -n | tail -20

# Calculations with >100,000 data points may timeout
```

**Resolution Procedures**:

**Sub-Procedure A: Historian Slow/Unavailable**
```bash
# Enable historian cache
kubectl set env deployment/thermaliq -n gl-009-production \
  HISTORIAN_CACHE_TTL=3600

# Increase historian timeout
kubectl set env deployment/thermaliq -n gl-009-production \
  HISTORIAN_TIMEOUT=60

# Enable circuit breaker
kubectl set env deployment/thermaliq -n gl-009-production \
  HISTORIAN_CIRCUIT_BREAKER_ENABLED=true

# Restart to apply changes
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Sub-Procedure B: Database Query Slow**
```bash
# Add missing index
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "CREATE INDEX CONCURRENTLY idx_energy_readings_facility_time
   ON energy_readings(facility_id, timestamp DESC);"

# Update statistics
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "ANALYZE energy_readings;"

# Test query performance
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "EXPLAIN ANALYZE
   SELECT * FROM energy_readings
   WHERE facility_id = 'FAC-001'
     AND timestamp >= '2025-11-01'
     AND timestamp < '2025-11-02';"
```

**Sub-Procedure C: Calculation Too Complex**
```bash
# Enable calculation chunking
kubectl set env deployment/thermaliq -n gl-009-production \
  MAX_DATA_POINTS_PER_CALCULATION=50000 \
  ENABLE_CALCULATION_CHUNKING=true

# Increase timeout for large calculations
kubectl set env deployment/thermaliq -n gl-009-production \
  CALCULATION_TIMEOUT=120

# Restart to apply changes
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Sub-Procedure D: Insufficient Resources**
```bash
# Increase CPU limits
kubectl set resources deployment/thermaliq -n gl-009-production \
  --limits=cpu=4000m --requests=cpu=2000m

# Scale horizontally
kubectl scale deployment/thermaliq -n gl-009-production --replicas=8

# Enable HPA
kubectl autoscale deployment thermaliq -n gl-009-production \
  --cpu-percent=70 --min=4 --max=12
```

**Post-Resolution**:
```bash
# Monitor timeout rate (should be < 0.1%)
rate(thermaliq_calculation_timeouts_total[15m])

# Check p95 latency (should be < 15 seconds)
histogram_quantile(0.95, thermaliq_calculation_duration_seconds_bucket)

# Verify queue draining
thermaliq_calculation_queue_depth
```

---

### Scenario 3: Energy Meter Connection Lost

**Symptoms**:
- Energy meter connector reporting "Connection refused"
- Missing data points in calculations
- Energy balance not closing
- Specific facilities unable to calculate

**Severity**: SEV2 (if multiple facilities) / SEV3 (if single facility)

**Detection**:
```bash
# Check energy meter connection status
kubectl logs -n gl-009-production -l app=thermaliq --tail=200 | \
  grep "EnergyMeterConnector" | \
  grep -i "error\|failed\|timeout"

# Check metric
rate(thermaliq_energy_meter_connection_errors_total[5m])

# Check data completeness
thermaliq_calculation_data_completeness < 0.95
```

**Immediate Actions**:

1. **Identify Affected Facilities**:
```bash
# List facilities with connection errors
kubectl logs -n gl-009-production -l app=thermaliq --tail=500 | \
  grep "EnergyMeter.*error" | \
  jq -r '.facility_id' | \
  sort -u

# Check energy meter health
for facility in $(cat affected_facilities.txt); do
  curl https://meter-api.example.com/facilities/$facility/health
done
```

2. **Check Network Connectivity**:
```bash
# Test connectivity from pod
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  curl -v --max-time 10 https://meter-api.example.com/health

# Check DNS resolution
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  nslookup meter-api.example.com

# Test TCP connection
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  nc -zv meter-api.example.com 443
```

3. **Verify Credentials**:
```bash
# Check secret exists
kubectl get secret energy-meter-credentials -n gl-009-production

# Verify credentials work
curl -X POST https://meter-api.example.com/auth/token \
  -d "client_id=$CLIENT_ID" \
  -d "client_secret=$CLIENT_SECRET"
```

**Investigation Steps**:

1. **Check Energy Meter API Status**:
```bash
# Call vendor status page
curl https://status.energy-meter-vendor.com/api/v2/status.json

# Check for incidents
curl https://status.energy-meter-vendor.com/api/v2/incidents.json
```

2. **Review Connection Logs**:
```bash
# Get detailed connection logs
kubectl logs -n gl-009-production -l app=thermaliq --tail=1000 | \
  grep "EnergyMeterConnector" | \
  jq 'select(.level=="ERROR" or .level=="WARN")'

# Common errors:
# - "401 Unauthorized" -> Credentials invalid/expired
# - "429 Too Many Requests" -> Rate limit hit
# - "503 Service Unavailable" -> Vendor API down
# - "timeout" -> Network/firewall issue
```

3. **Check Rate Limiting**:
```bash
# Check request rate
rate(thermaliq_energy_meter_requests_total[1m])

# Check rate limit headers from last request
kubectl logs -n gl-009-production -l app=thermaliq --tail=100 | \
  grep "X-RateLimit"
```

**Resolution Procedures**:

**Sub-Procedure A: Authentication Failed**
```bash
# Rotate credentials
# 1. Generate new credentials in vendor portal
# 2. Update Kubernetes secret
kubectl create secret generic energy-meter-credentials \
  -n gl-009-production \
  --from-literal=client-id=$NEW_CLIENT_ID \
  --from-literal=client-secret=$NEW_CLIENT_SECRET \
  --dry-run=client -o yaml | kubectl apply -f -

# 3. Restart pods to pick up new credentials
kubectl rollout restart deployment/thermaliq -n gl-009-production

# 4. Verify connection
kubectl logs -n gl-009-production -l app=thermaliq --tail=50 | \
  grep "EnergyMeter.*authenticated"
```

**Sub-Procedure B: Rate Limit Exceeded**
```bash
# Reduce request rate
kubectl set env deployment/thermaliq -n gl-009-production \
  ENERGY_METER_MAX_REQUESTS_PER_MINUTE=50 \
  ENERGY_METER_BACKOFF_ENABLED=true

# Enable request batching
kubectl set env deployment/thermaliq -n gl-009-production \
  ENERGY_METER_BATCH_REQUESTS=true \
  ENERGY_METER_BATCH_SIZE=100

# Increase cache TTL to reduce requests
kubectl set env deployment/thermaliq -n gl-009-production \
  ENERGY_METER_CACHE_TTL=7200

# Restart to apply changes
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Sub-Procedure C: Vendor API Down**
```bash
# Enable fallback to cached data
kubectl set env deployment/thermaliq -n gl-009-production \
  ENERGY_METER_ENABLE_FALLBACK=true \
  ENERGY_METER_FALLBACK_CACHE_MAX_AGE=86400

# Notify users of degraded service
# Post to status page about using cached data

# Contact vendor support
echo "Priority support ticket created: TICKET-12345"

# Set up monitoring for vendor recovery
watch -n 60 'curl -s https://meter-api.example.com/health | jq .status'
```

**Sub-Procedure D: Network Connectivity Issue**
```bash
# Check firewall rules
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  traceroute meter-api.example.com

# Check egress network policy
kubectl get networkpolicy -n gl-009-production

# If blocked, update network policy
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: thermaliq-egress
  namespace: gl-009-production
spec:
  podSelector:
    matchLabels:
      app: thermaliq
  policyTypes:
  - Egress
  egress:
  - to:
    - podSelector: {}
  - to:
    - namespaceSelector: {}
  - ports:
    - protocol: TCP
      port: 443
    to:
    - podSelector: {}
EOF

# Test connectivity again
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  curl -v https://meter-api.example.com/health
```

**Post-Resolution**:
```bash
# Verify connection restored
rate(thermaliq_energy_meter_connection_errors_total[5m]) == 0

# Check data completeness
thermaliq_calculation_data_completeness > 0.95

# Run test calculation for affected facilities
./scripts/test_calculation.sh --facility FAC-001
```

---

### Scenario 4: Historian Query Failures

**Symptoms**:
- Historical data queries timing out or failing
- Calculations missing historical context
- Benchmark comparisons unavailable
- Trend analysis failing

**Severity**: SEV2

**Detection**:
```bash
# Check historian error rate
rate(thermaliq_historian_query_errors_total[5m])

# Check query latency
histogram_quantile(0.95, thermaliq_historian_query_duration_seconds_bucket)

# Check historian connectivity
thermaliq_historian_connection_status == 0
```

**Immediate Actions**:

1. **Verify Historian Availability**:
```bash
# Check historian health
curl -v https://historian.example.com/api/health

# Check authentication
curl -X POST https://historian.example.com/api/auth \
  -H "Content-Type: application/json" \
  -d '{"username": "thermaliq", "password": "***"}'
```

2. **Enable Cache Fallback**:
```bash
# Use cached historical data temporarily
kubectl set env deployment/thermaliq -n gl-009-production \
  HISTORIAN_ENABLE_CACHE_FALLBACK=true

kubectl rollout restart deployment/thermaliq -n gl-009-production
```

3. **Check Query Patterns**:
```bash
# Analyze recent queries
kubectl logs -n gl-009-production -l app=thermaliq --tail=500 | \
  grep "HistorianQuery" | \
  jq '{facility: .facility_id, start: .query_start, end: .query_end, tags: .tags, duration: .duration}'
```

**Investigation Steps**:

1. **Test Sample Queries**:
```bash
# Small query (should succeed)
curl -X POST https://historian.example.com/api/query \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "facility_id": "FAC-001",
    "start_time": "2025-11-01T00:00:00Z",
    "end_time": "2025-11-01T01:00:00Z",
    "tags": ["temperature"]
  }'

# Large query (may fail)
curl -X POST https://historian.example.com/api/query \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "facility_id": "FAC-001",
    "start_time": "2025-01-01T00:00:00Z",
    "end_time": "2025-12-31T23:59:59Z",
    "tags": ["*"]
  }'
```

2. **Check Historian Capacity**:
```bash
# Check historian metrics
curl https://historian.example.com/metrics | grep query

# Check for resource constraints
# - CPU usage
# - Memory usage
# - Disk I/O
# - Network bandwidth
```

3. **Analyze Query Performance**:
```bash
# Get query statistics
kubectl logs -n gl-009-production -l app=thermaliq --tail=1000 | \
  grep "HistorianQuery" | \
  jq -s 'group_by(.facility_id) | map({facility: .[0].facility_id, avg_duration: (map(.duration) | add / length), count: length})'
```

**Resolution Procedures**:

**Sub-Procedure A: Historian Overloaded**
```bash
# Reduce query rate
kubectl set env deployment/thermaliq -n gl-009-production \
  HISTORIAN_MAX_CONCURRENT_QUERIES=5 \
  HISTORIAN_QUERY_RATE_LIMIT=10

# Enable query queuing
kubectl set env deployment/thermaliq -n gl-009-production \
  HISTORIAN_ENABLE_QUERY_QUEUE=true \
  HISTORIAN_QUEUE_SIZE=100

# Restart to apply changes
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Sub-Procedure B: Queries Too Large**
```bash
# Enable query chunking
kubectl set env deployment/thermaliq -n gl-009-production \
  HISTORIAN_ENABLE_QUERY_CHUNKING=true \
  HISTORIAN_MAX_QUERY_RANGE_HOURS=168 \
  HISTORIAN_CHUNK_SIZE_HOURS=24

# Restart to apply changes
kubectl rollout restart deployment/thermaliq -n gl-009-production

# Monitor chunked queries
kubectl logs -n gl-009-production -l app=thermaliq -f | \
  grep "HistorianQuery.*chunked"
```

**Sub-Procedure C: Authentication Issues**
```bash
# Refresh historian credentials
kubectl create secret generic historian-credentials \
  -n gl-009-production \
  --from-literal=username=$HISTORIAN_USER \
  --from-literal=password=$HISTORIAN_PASS \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pods
kubectl rollout restart deployment/thermaliq -n gl-009-production

# Verify authentication
kubectl logs -n gl-009-production -l app=thermaliq --tail=20 | \
  grep "Historian.*authenticated"
```

**Sub-Procedure D: Network Timeout**
```bash
# Increase timeout
kubectl set env deployment/thermaliq -n gl-009-production \
  HISTORIAN_QUERY_TIMEOUT=120 \
  HISTORIAN_CONNECTION_TIMEOUT=30

# Enable retry with backoff
kubectl set env deployment/thermaliq -n gl-009-production \
  HISTORIAN_RETRY_ENABLED=true \
  HISTORIAN_MAX_RETRIES=3 \
  HISTORIAN_RETRY_BACKOFF=exponential

# Restart to apply changes
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Post-Resolution**:
```bash
# Verify queries succeeding
rate(thermaliq_historian_query_errors_total[10m]) < 0.01

# Check query latency
histogram_quantile(0.95, thermaliq_historian_query_duration_seconds_bucket) < 30

# Run test calculation with historical data
./scripts/test_calculation.sh --with-history --facility FAC-001
```

---

### Scenario 5: High Error Rate (>1%)

**Symptoms**:
- Calculation error rate > 1%
- Intermittent calculation failures
- Increased user complaints
- Error logs growing rapidly

**Severity**: SEV2

**Detection**:
```bash
# Check error rate
rate(thermaliq_calculation_errors_total[5m]) / rate(thermaliq_calculation_requests_total[5m]) > 0.01

# Check error types
kubectl logs -n gl-009-production -l app=thermaliq --tail=500 | \
  grep "ERROR" | \
  jq -r '.error_type' | \
  sort | uniq -c | sort -rn
```

**Immediate Actions**:

1. **Categorize Errors**:
```bash
# Group errors by type
kubectl logs -n gl-009-production -l app=thermaliq --tail=1000 | \
  grep "ERROR" | \
  jq -r '.error_type' | \
  sort | uniq -c | sort -rn | head -10

# Common error types:
# - ValidationError
# - DatabaseError
# - HistorianError
# - CalculationError
# - TimeoutError
```

2. **Check for Patterns**:
```bash
# Errors by facility
kubectl logs -n gl-009-production -l app=thermaliq --tail=1000 | \
  grep "ERROR" | \
  jq -r '.facility_id' | \
  sort | uniq -c | sort -rn | head -10

# Errors by time
kubectl logs -n gl-009-production -l app=thermaliq --tail=1000 | \
  grep "ERROR" | \
  jq -r '.timestamp' | \
  cut -d'T' -f1 | \
  sort | uniq -c
```

3. **Sample Error Messages**:
```bash
# Get recent error details
kubectl logs -n gl-009-production -l app=thermaliq --tail=500 | \
  grep "ERROR" | \
  jq '{time: .timestamp, facility: .facility_id, error: .error_message, stack: .stack_trace}' | \
  head -10
```

**Investigation Steps**:

1. **Analyze Error Distribution**:
```bash
# Errors by pod
for pod in $(kubectl get pods -n gl-009-production -l app=thermaliq -o name); do
  echo "=== $pod ==="
  kubectl logs $pod -n gl-009-production --tail=200 | \
    grep "ERROR" | wc -l
done

# If errors concentrated in specific pods -> Pod-specific issue
# If errors distributed evenly -> Systemic issue
```

2. **Check Recent Changes**:
```bash
# Recent deployments
kubectl rollout history deployment/thermaliq -n gl-009-production | tail -3

# Recent config changes
kubectl get configmap thermaliq-config -n gl-009-production -o yaml | \
  grep -A 2 "metadata:"

# Recent scaling events
kubectl get events -n gl-009-production --sort-by='.lastTimestamp' | \
  grep -i "scaled\|rolling"
```

3. **Reproduce Error**:
```bash
# Extract failing request from logs
kubectl logs -n gl-009-production -l app=thermaliq --tail=500 | \
  grep "ERROR" | \
  jq -r '.request_payload' | \
  head -1 > failing_request.json

# Attempt to reproduce
curl -X POST https://api.greenlang.io/v1/thermaliq/calculate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @failing_request.json
```

**Resolution Procedures**:

**Sub-Procedure A: Validation Errors**
```bash
# Check validation rules
kubectl get configmap thermaliq-config -n gl-009-production -o yaml | \
  grep -A 50 "validation:"

# Common validation issues:
# - Missing required fields
# - Invalid date formats
# - Out-of-range values

# If validation too strict, relax rules temporarily
kubectl edit configmap thermaliq-config -n gl-009-production

# Reload configuration
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Sub-Procedure B: Database Errors**
```bash
# Check database connection pool
kubectl logs -n gl-009-production -l app=thermaliq --tail=200 | \
  grep "DatabaseError" | \
  jq -r '.error_message'

# Common issues:
# - Connection pool exhausted
# - Deadlocks
# - Constraint violations

# Increase connection pool size
kubectl set env deployment/thermaliq -n gl-009-production \
  DATABASE_POOL_SIZE=50 \
  DATABASE_MAX_OVERFLOW=20

# Restart to apply changes
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Sub-Procedure C: Calculation Errors**
```bash
# Analyze calculation failures
kubectl logs -n gl-009-production -l app=thermaliq --tail=500 | \
  grep "CalculationError" | \
  jq '{facility: .facility_id, error: .error_message, data_points: .data_point_count}'

# Common issues:
# - Division by zero (missing denominator data)
# - Energy balance not closing (>5% discrepancy)
# - Negative efficiency values

# Enable lenient calculation mode temporarily
kubectl set env deployment/thermaliq -n gl-009-production \
  CALCULATION_MODE=lenient \
  ALLOW_NEGATIVE_EFFICIENCY=true \
  ENERGY_BALANCE_THRESHOLD=0.10

# Restart to apply changes
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Sub-Procedure D: Recent Deployment Introduced Bug**
```bash
# Rollback to previous version
kubectl rollout undo deployment/thermaliq -n gl-009-production

# Wait for rollout
kubectl rollout status deployment/thermaliq -n gl-009-production

# Verify error rate decreased
rate(thermaliq_calculation_errors_total[5m]) / rate(thermaliq_calculation_requests_total[5m])

# File bug report with details
echo "Bug filed: BUG-12345 - High error rate in v1.2.3"
```

**Post-Resolution**:
```bash
# Monitor error rate for 1 hour
watch -n 60 'rate(thermaliq_calculation_errors_total[5m]) / rate(thermaliq_calculation_requests_total[5m])'

# Should be < 0.01 (1%)

# Review error logs to ensure no new error types
kubectl logs -n gl-009-production -l app=thermaliq --tail=500 | \
  grep "ERROR" | \
  jq -r '.error_type' | \
  sort | uniq -c
```

---

### Scenario 6: Cache Exhaustion

**Symptoms**:
- Redis memory usage at 100%
- Cache evictions occurring
- Degraded calculation performance
- Increased database load

**Severity**: SEV3

**Detection**:
```bash
# Check Redis memory usage
kubectl exec -it redis-0 -n gl-009-production -- redis-cli INFO memory | grep used_memory_human

# Check eviction count
kubectl exec -it redis-0 -n gl-009-production -- redis-cli INFO stats | grep evicted_keys

# Check cache hit rate
thermaliq_cache_hits_total / (thermaliq_cache_hits_total + thermaliq_cache_misses_total) < 0.70
```

**Immediate Actions**:

1. **Check Cache Size**:
```bash
# Get cache statistics
kubectl exec -it redis-0 -n gl-009-production -- redis-cli INFO memory

# Key metrics:
# - used_memory_human: Current memory usage
# - maxmemory_human: Memory limit
# - mem_fragmentation_ratio: Fragmentation
```

2. **Identify Large Keys**:
```bash
# Find largest keys
kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli --bigkeys

# Get key sizes
kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli --memkeys
```

3. **Check Key Patterns**:
```bash
# Count keys by pattern
kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli KEYS "calculation:*" | wc -l

kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli KEYS "historian:*" | wc -l

kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli KEYS "energy_meter:*" | wc -l
```

**Investigation Steps**:

1. **Analyze Key TTLs**:
```bash
# Check if keys have TTLs
kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli SCAN 0 COUNT 100 | \
  grep -v "^[0-9]" | \
  while read key; do
    ttl=$(redis-cli TTL "$key")
    echo "$key: $ttl"
  done | head -20

# Keys with TTL=-1 never expire
```

2. **Check Cache Growth Rate**:
```bash
# Sample memory usage over time
for i in {1..10}; do
  kubectl exec -it redis-0 -n gl-009-production -- \
    redis-cli INFO memory | grep used_memory_human
  sleep 60
done
```

3. **Review Eviction Policy**:
```bash
# Check eviction policy
kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli CONFIG GET maxmemory-policy

# Policies:
# - noeviction: Don't evict, return errors
# - allkeys-lru: Evict least recently used
# - volatile-lru: Evict LRU with TTL set
# - allkeys-random: Evict random keys
```

**Resolution Procedures**:

**Sub-Procedure A: Increase Cache Size**
```bash
# Increase Redis memory limit
kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli CONFIG SET maxmemory 8gb

# Make change persistent
kubectl edit statefulset redis -n gl-009-production
# Update memory limit in container resources

# Restart Redis
kubectl rollout restart statefulset redis -n gl-009-production
```

**Sub-Procedure B: Adjust TTLs**
```bash
# Reduce TTL for calculation cache
kubectl set env deployment/thermaliq -n gl-009-production \
  CALCULATION_CACHE_TTL=3600

# Reduce TTL for historian cache
kubectl set env deployment/thermaliq -n gl-009-production \
  HISTORIAN_CACHE_TTL=7200

# Restart to apply changes
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Sub-Procedure C: Clear Stale Keys**
```bash
# Clear keys older than 7 days without TTL
kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli SCAN 0 COUNT 1000 | \
  grep -v "^[0-9]" | \
  while read key; do
    ttl=$(redis-cli TTL "$key")
    if [ "$ttl" -eq "-1" ]; then
      redis-cli DEL "$key"
    fi
  done

# Force cache cleanup
kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli MEMORY PURGE
```

**Sub-Procedure D: Optimize Cache Usage**
```bash
# Enable compression for cached values
kubectl set env deployment/thermaliq -n gl-009-production \
  CACHE_COMPRESSION_ENABLED=true

# Reduce cached data size
kubectl set env deployment/thermaliq -n gl-009-production \
  CACHE_INCLUDE_RAW_DATA=false

# Implement cache sharding
# Update application config to use multiple Redis instances

# Restart to apply changes
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Post-Resolution**:
```bash
# Monitor memory usage
watch -n 30 'kubectl exec -it redis-0 -n gl-009-production -- redis-cli INFO memory | grep used_memory'

# Check cache hit rate improved
thermaliq_cache_hits_total / (thermaliq_cache_hits_total + thermaliq_cache_misses_total) > 0.80

# Verify evictions reduced
kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli INFO stats | grep evicted_keys
```

---

### Scenario 7: Memory Exhaustion

**Symptoms**:
- Pods being OOMKilled
- Calculation failures with "Out of memory" errors
- Increased garbage collection time
- Degraded performance

**Severity**: SEV2

**Detection**:
```bash
# Check pod memory usage
kubectl top pods -n gl-009-production -l app=thermaliq

# Check OOMKilled events
kubectl get events -n gl-009-production | grep OOMKilled

# Check memory limit
kubectl describe pod thermaliq-abc123 -n gl-009-production | \
  grep -A 5 "Limits:"
```

**Immediate Actions**:

1. **Increase Memory Limits**:
```bash
# Scale up memory
kubectl set resources deployment/thermaliq -n gl-009-production \
  --limits=memory=8Gi --requests=memory=4Gi

# Wait for rollout
kubectl rollout status deployment/thermaliq -n gl-009-production
```

2. **Check for Memory Leaks**:
```bash
# Get memory profile before crash
kubectl logs thermaliq-abc123 -n gl-009-production --previous | \
  grep "MemoryError\|OutOfMemory"

# Check pod uptime (frequent restarts indicate leak)
kubectl get pods -n gl-009-production -l app=thermaliq -o wide
```

3. **Monitor Memory Growth**:
```bash
# Watch memory usage in real-time
watch -n 5 'kubectl top pods -n gl-009-production -l app=thermaliq'
```

**Investigation Steps**:

1. **Analyze Memory Usage Patterns**:
```bash
# Get memory metrics
curl -s http://prometheus:9090/api/v1/query?query=container_memory_usage_bytes{pod=~"thermaliq.*"} | jq

# Plot memory over time
# Look for:
# - Steady linear growth (memory leak)
# - Sudden spikes (large calculations)
# - Sawtooth pattern (normal GC behavior)
```

2. **Check Large Calculations**:
```bash
# Find calculations with high data point counts
kubectl logs -n gl-009-production -l app=thermaliq --tail=1000 | \
  grep "Calculation.*data_points" | \
  jq '{calc_id: .calculation_id, facility: .facility_id, data_points: .data_point_count, memory: .peak_memory_mb}' | \
  sort -k4 -rn | head -20
```

3. **Profile Memory Usage**:
```bash
# Enable memory profiling
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python -m memory_profiler calculation_service.py

# Generate heap dump
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python -c "import objgraph; objgraph.show_growth()"
```

**Resolution Procedures**:

**Sub-Procedure A: Memory Leak Detected**
```bash
# Identify leaking code path
kubectl logs -n gl-009-production -l app=thermaliq --tail=5000 | \
  grep "Memory.*growing" | \
  jq -r '.function_name' | \
  sort | uniq -c | sort -rn

# Implement temporary mitigation: periodic restart
kubectl set env deployment/thermaliq -n gl-009-production \
  RESTART_AFTER_CALCULATIONS=100

# File bug for code fix
echo "Memory leak bug filed: BUG-12346"

# Deploy hotfix when available
kubectl set image deployment/thermaliq -n gl-009-production \
  thermaliq=ghcr.io/greenlang/thermaliq:v1.2.4-hotfix
```

**Sub-Procedure B: Large Calculations Consuming Memory**
```bash
# Limit calculation size
kubectl set env deployment/thermaliq -n gl-009-production \
  MAX_DATA_POINTS_PER_CALCULATION=100000 \
  ENABLE_STREAMING_CALCULATION=true

# Process large calculations in chunks
kubectl set env deployment/thermaliq -n gl-009-production \
  CALCULATION_CHUNK_SIZE=10000

# Restart to apply changes
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Sub-Procedure C: Insufficient Garbage Collection**
```bash
# Tune garbage collection (Python)
kubectl set env deployment/thermaliq -n gl-009-production \
  PYTHONHASHSEED=0 \
  PYTHONMALLOC=malloc

# Force aggressive GC
kubectl set env deployment/thermaliq -n gl-009-production \
  GC_AGGRESSIVE=true \
  GC_THRESHOLD_0=500 \
  GC_THRESHOLD_1=5 \
  GC_THRESHOLD_2=5

# Restart to apply changes
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Sub-Procedure D: External Library Memory Leak**
```bash
# Identify leaking library
kubectl logs -n gl-009-production -l app=thermaliq --tail=1000 | \
  grep "import" | grep -v "^#"

# Update dependencies
# Edit requirements.txt to upgrade leaking library
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  pip list --outdated

# Build and deploy new image with updated dependencies
docker build -t ghcr.io/greenlang/thermaliq:v1.2.4-deps .
docker push ghcr.io/greenlang/thermaliq:v1.2.4-deps
kubectl set image deployment/thermaliq -n gl-009-production \
  thermaliq=ghcr.io/greenlang/thermaliq:v1.2.4-deps
```

**Post-Resolution**:
```bash
# Monitor memory usage for 4 hours
watch -n 300 'kubectl top pods -n gl-009-production -l app=thermaliq'

# Verify no OOMKilled events
kubectl get events -n gl-009-production | grep OOMKilled

# Check memory growth rate
# Should be flat or sawtooth (GC), not linear
```

---

### Scenario 8: Database Connection Failures

**Symptoms**:
- "Connection refused" errors
- "Too many connections" errors
- Calculation failures
- Slow query performance

**Severity**: SEV1 (if all connections fail) / SEV2 (if intermittent)

**Detection**:
```bash
# Check database connectivity
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  psql $DATABASE_URL -c "SELECT 1"

# Check connection count
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT count(*) FROM pg_stat_activity WHERE datname='thermaliq_production';"

# Check for errors
kubectl logs -n gl-009-production -l app=thermaliq --tail=200 | \
  grep "DatabaseError\|Connection.*failed"
```

**Immediate Actions**:

1. **Check Database Status**:
```bash
# Check database pod
kubectl get pods -n gl-009-production -l app=postgres

# Check database service
kubectl get service postgres-service -n gl-009-production

# Check database logs
kubectl logs postgres-0 -n gl-009-production --tail=100
```

2. **Check Connection Pool**:
```bash
# View active connections
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT pid, usename, application_name, client_addr, state, query_start, state_change, query
   FROM pg_stat_activity
   WHERE datname='thermaliq_production'
   ORDER BY state_change DESC
   LIMIT 20;"

# Check max connections
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SHOW max_connections;"
```

3. **Kill Idle Connections** (if pool exhausted):
```bash
# Kill idle connections
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT pg_terminate_backend(pid)
   FROM pg_stat_activity
   WHERE datname='thermaliq_production'
     AND state='idle'
     AND state_change < NOW() - INTERVAL '10 minutes';"
```

**Investigation Steps**:

1. **Analyze Connection Patterns**:
```bash
# Check application connection pool settings
kubectl get configmap thermaliq-config -n gl-009-production -o yaml | \
  grep -A 10 "database:"

# Check for connection leaks
kubectl logs -n gl-009-production -l app=thermaliq --tail=1000 | \
  grep "Connection.*opened\|Connection.*closed" | \
  grep -c "opened"
kubectl logs -n gl-009-production -l app=thermaliq --tail=1000 | \
  grep "Connection.*opened\|Connection.*closed" | \
  grep -c "closed"

# If opened >> closed, connection leak exists
```

2. **Check Long-Running Queries**:
```bash
# Find long-running queries
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT pid, now() - query_start AS duration, state, query
   FROM pg_stat_activity
   WHERE state != 'idle'
     AND query NOT LIKE '%pg_stat_activity%'
   ORDER BY duration DESC
   LIMIT 10;"
```

3. **Check Database Locks**:
```bash
# Find blocking queries
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT blocked_locks.pid AS blocked_pid,
          blocked_activity.usename AS blocked_user,
          blocking_locks.pid AS blocking_pid,
          blocking_activity.usename AS blocking_user,
          blocked_activity.query AS blocked_statement,
          blocking_activity.query AS blocking_statement
   FROM pg_catalog.pg_locks blocked_locks
   JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
   JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
   JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
   WHERE NOT blocked_locks.granted;"
```

**Resolution Procedures**:

**Sub-Procedure A: Connection Pool Exhausted**
```bash
# Increase application connection pool
kubectl set env deployment/thermaliq -n gl-009-production \
  DATABASE_POOL_SIZE=100 \
  DATABASE_MAX_OVERFLOW=50 \
  DATABASE_POOL_RECYCLE=3600

# Increase database max_connections
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U postgres -c "ALTER SYSTEM SET max_connections = 500;"

# Restart database for change to take effect
kubectl rollout restart statefulset postgres -n gl-009-production

# Wait for database to be ready
kubectl wait --for=condition=ready pod/postgres-0 -n gl-009-production --timeout=300s

# Restart application
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Sub-Procedure B: Connection Leak**
```bash
# Enable connection leak detection
kubectl set env deployment/thermaliq -n gl-009-production \
  SQLALCHEMY_POOL_PRE_PING=true \
  SQLALCHEMY_POOL_RECYCLE=300 \
  SQLALCHEMY_ECHO_POOL=true

# Force connection cleanup
kubectl set env deployment/thermaliq -n gl-009-production \
  DATABASE_POOL_TIMEOUT=10

# Restart to apply changes
kubectl rollout restart deployment/thermaliq -n gl-009-production

# Monitor connection count
watch -n 10 'kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT count(*) FROM pg_stat_activity WHERE datname=\"thermaliq_production\";"'
```

**Sub-Procedure C: Database Unresponsive**
```bash
# Check database process
kubectl exec -it postgres-0 -n gl-009-production -- ps aux | grep postgres

# Check database resource usage
kubectl top pod postgres-0 -n gl-009-production

# Check for deadlocks
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT * FROM pg_stat_activity WHERE wait_event_type = 'Lock';"

# If necessary, restart database
kubectl delete pod postgres-0 -n gl-009-production
kubectl wait --for=condition=ready pod/postgres-0 -n gl-009-production --timeout=300s
```

**Sub-Procedure D: Network Partition**
```bash
# Check network connectivity
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  nc -zv postgres-service 5432

# Check DNS resolution
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  nslookup postgres-service

# Check network policy
kubectl get networkpolicy -n gl-009-production

# Test from different pod
kubectl run test-connectivity --image=postgres:15 -n gl-009-production -it --rm -- \
  psql -h postgres-service -U thermaliq -d thermaliq_production -c "SELECT 1"
```

**Post-Resolution**:
```bash
# Verify database connectivity
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  psql $DATABASE_URL -c "SELECT 1"

# Check connection count stable
watch -n 30 'kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT count(*) FROM pg_stat_activity WHERE datname=\"thermaliq_production\";"'

# Run test calculations
./scripts/test_calculation.sh --count 10
```

---

### Scenario 9: Incorrect Efficiency Calculations

**Symptoms**:
- Efficiency values outside expected range (0-100%)
- Negative efficiency values
- Efficiency values not matching manual calculations
- User reports of incorrect results

**Severity**: SEV1 (data integrity issue)

**Detection**:
```bash
# Check for anomalous efficiency values
kubectl logs -n gl-009-production -l app=thermaliq --tail=1000 | \
  grep "Efficiency calculated" | \
  jq 'select(.efficiency < 0 or .efficiency > 100)'

# Check data quality metric
thermaliq_calculation_data_quality_score < 90

# User reports
# Check support tickets
```

**Immediate Actions**:

1. **Halt New Calculations** (if widespread issue):
```bash
# Scale down to prevent more incorrect calculations
kubectl scale deployment/thermaliq -n gl-009-production --replicas=0

# Post incident notice
curl -X POST https://api.statuspage.io/v1/incidents \
  -H "Authorization: OAuth $STATUSPAGE_TOKEN" \
  -d '{
    "incident": {
      "name": "GL-009: Calculation Accuracy Issue - Service Paused",
      "status": "investigating",
      "impact_override": "critical"
    }
  }'
```

2. **Identify Affected Calculations**:
```bash
# Query for suspicious results
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT calculation_id, facility_id, efficiency, created_at
   FROM calculations
   WHERE efficiency < 0 OR efficiency > 100
     OR created_at > NOW() - INTERVAL '24 hours'
   ORDER BY created_at DESC;"
```

3. **Sample Manual Verification**:
```bash
# Get calculation details
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT * FROM calculations WHERE calculation_id='CALC-123';"

# Extract input data
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT * FROM energy_readings
   WHERE facility_id='FAC-001'
     AND timestamp >= '2025-11-01'
     AND timestamp < '2025-11-02';"

# Manually calculate efficiency
# Compare with stored result
```

**Investigation Steps**:

1. **Analyze Calculation Logic**:
```bash
# Review calculation code
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  cat /app/calculation_engine.py | grep -A 50 "def calculate_efficiency"

# Check for recent changes
git log --oneline --graph -- calculation_engine.py | head -10
git diff HEAD~5 HEAD -- calculation_engine.py
```

2. **Check Input Data Quality**:
```bash
# Verify energy meter data
kubectl logs -n gl-009-production -l app=thermaliq --tail=500 | \
  grep "Energy readings" | \
  jq '{facility: .facility_id, input_energy: .input_energy_kwh, output_energy: .output_energy_kwh, losses: .energy_losses_kwh}'

# Check for:
# - Missing data points
# - Zero values where non-zero expected
# - Outliers/anomalies
```

3. **Reproduce Issue**:
```bash
# Create test case with known values
cat > test_calculation.json <<EOF
{
  "facility_id": "FAC-TEST-001",
  "time_period": {
    "start": "2025-11-01T00:00:00Z",
    "end": "2025-11-01T01:00:00Z"
  },
  "energy_input": {
    "fuel": 1000.0,
    "electricity": 50.0
  },
  "energy_output": {
    "steam": 750.0,
    "electricity": 200.0
  }
}
EOF

# Expected efficiency: (750 + 200) / (1000 + 50) = 90.48%

# Run calculation
curl -X POST https://api.greenlang.io/v1/thermaliq/calculate \
  -H "Authorization: Bearer $TOKEN" \
  -d @test_calculation.json

# Compare result with expected
```

**Resolution Procedures**:

**Sub-Procedure A: Calculation Formula Error**
```bash
# Identify incorrect formula
# Example: Using wrong energy conversion factors

# Hotfix code
git checkout -b hotfix/calculation-formula
# Edit calculation_engine.py with correct formula
git commit -m "fix: correct efficiency calculation formula"
git push origin hotfix/calculation-formula

# Build and deploy hotfix
docker build -t ghcr.io/greenlang/thermaliq:v1.2.4-hotfix .
docker push ghcr.io/greenlang/thermaliq:v1.2.4-hotfix

# Deploy
kubectl set image deployment/thermaliq -n gl-009-production \
  thermaliq=ghcr.io/greenlang/thermaliq:v1.2.4-hotfix

# Verify fix
./scripts/test_calculation.sh --verify-accuracy
```

**Sub-Procedure B: Data Conversion Error**
```bash
# Check unit conversions
kubectl logs -n gl-009-production -l app=thermaliq --tail=500 | \
  grep "Unit conversion" | \
  jq '{from_unit: .from_unit, to_unit: .to_unit, factor: .conversion_factor}'

# Common issues:
# - MWh to kWh: factor should be 1000
# - BTU to kWh: factor should be 0.000293071
# - GJ to kWh: factor should be 277.778

# Fix unit conversion config
kubectl edit configmap thermaliq-config -n gl-009-production

# Update conversion factors
# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Sub-Procedure C: Missing Energy Components**
```bash
# Check if energy losses being omitted
kubectl logs -n gl-009-production -l app=thermaliq --tail=500 | \
  grep "Energy balance" | \
  jq '{input: .total_input, output: .total_output, losses: .total_losses, balance: .energy_balance_percent}'

# If losses not included in efficiency calculation:
# Efficiency = (Useful Output) / (Total Input)
# Losses should be: Input - Output = Losses

# Enable comprehensive energy accounting
kubectl set env deployment/thermaliq -n gl-009-production \
  INCLUDE_ENERGY_LOSSES=true \
  ENERGY_BALANCE_CHECK_ENABLED=true

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Sub-Procedure D: Recalculate Affected Results**
```bash
# Get list of affected calculations
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT calculation_id, facility_id, created_at
   FROM calculations
   WHERE created_at >= '2025-11-01'
     AND created_at < '2025-11-26'
   ORDER BY created_at;" \
  --csv > affected_calculations.csv

# Mark old calculations as invalid
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "UPDATE calculations
   SET status='invalid',
       invalid_reason='Calculation formula error - see incident INC-12345'
   WHERE created_at >= '2025-11-01'
     AND created_at < '2025-11-26';"

# Trigger recalculation
cat affected_calculations.csv | while IFS=, read calc_id facility_id created_at; do
  echo "Recalculating $calc_id for $facility_id"
  curl -X POST https://api.greenlang.io/v1/thermaliq/recalculate \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"original_calculation_id\": \"$calc_id\"}"
done

# Notify affected users
./scripts/notify_recalculation.sh affected_calculations.csv
```

**Post-Resolution**:
```bash
# Scale service back up
kubectl scale deployment/thermaliq -n gl-009-production --replicas=4

# Run accuracy validation
./scripts/validate_accuracy.sh --sample-size 100

# All results should be within tolerance (±0.1%)

# Update status page
curl -X PATCH https://api.statuspage.io/v1/incidents/$INCIDENT_ID \
  -H "Authorization: OAuth $STATUSPAGE_TOKEN" \
  -d '{"incident": {"status": "resolved"}}'

# Document root cause and fix in post-incident review
```

---

### Scenario 10: Sankey Diagram Generation Failures

**Symptoms**:
- Sankey diagram API returning errors
- Diagram rendering timeouts
- Blank/incomplete diagrams
- Visualization data missing

**Severity**: SEV3

**Detection**:
```bash
# Check Sankey generation errors
rate(thermaliq_sankey_generation_errors_total[5m]) > 0

# Check generation latency
histogram_quantile(0.95, thermaliq_sankey_generation_duration_seconds_bucket) > 10

# Check user reports
kubectl logs -n gl-009-production -l app=thermaliq --tail=200 | \
  grep "SankeyGenerator" | \
  grep "ERROR\|WARN"
```

**Immediate Actions**:

1. **Test Sankey Generation**:
```bash
# Request sample diagram
curl -X GET "https://api.greenlang.io/v1/thermaliq/calculations/CALC-123/sankey" \
  -H "Authorization: Bearer $TOKEN"

# Check response
# - 200 OK with SVG/JSON data -> Success
# - 500 Error -> Generation failure
# - 504 Timeout -> Performance issue
```

2. **Check Visualization Library**:
```bash
# Check library version
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  pip show plotly

# Check for known issues
# https://github.com/plotly/plotly.py/issues
```

3. **Review Recent Changes**:
```bash
# Check recent commits to visualization code
git log --oneline --graph -- sankey_generator.py | head -5
```

**Investigation Steps**:

1. **Analyze Failing Cases**:
```bash
# Get calculations with Sankey failures
kubectl logs -n gl-009-production -l app=thermaliq --tail=1000 | \
  grep "SankeyGenerator.*ERROR" | \
  jq '{calc_id: .calculation_id, facility: .facility_id, error: .error_message, node_count: .node_count, link_count: .link_count}'

# Look for patterns:
# - Too many nodes (>100)?
# - Circular flows?
# - Invalid flow values?
```

2. **Test Diagram Locally**:
```bash
# Extract failing calculation data
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT energy_flows FROM calculations WHERE calculation_id='CALC-123';" \
  > energy_flows.json

# Generate diagram locally
python -c "
from sankey_generator import SankeyGenerator
import json

with open('energy_flows.json') as f:
    flows = json.load(f)

generator = SankeyGenerator()
diagram = generator.generate(flows)
print(diagram)
"
```

3. **Check Resource Usage**:
```bash
# Sankey generation is CPU-intensive
kubectl top pods -n gl-009-production -l app=thermaliq

# Check memory during generation
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python -m memory_profiler sankey_generator.py
```

**Resolution Procedures**:

**Sub-Procedure A: Too Many Nodes/Links**
```bash
# Limit diagram complexity
kubectl set env deployment/thermaliq -n gl-009-production \
  SANKEY_MAX_NODES=50 \
  SANKEY_MAX_LINKS=100 \
  SANKEY_AGGREGATE_SMALL_FLOWS=true \
  SANKEY_SMALL_FLOW_THRESHOLD=0.01

# Restart to apply changes
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Sub-Procedure B: Circular Flow Detected**
```bash
# Enable flow validation
kubectl set env deployment/thermaliq -n gl-009-production \
  SANKEY_VALIDATE_FLOWS=true \
  SANKEY_ALLOW_CIRCULAR_FLOWS=false

# Add flow validation logic
# Edit sankey_generator.py to detect and handle circular flows

# Redeploy
kubectl set image deployment/thermaliq -n gl-009-production \
  thermaliq=ghcr.io/greenlang/thermaliq:v1.2.5
```

**Sub-Procedure C: Rendering Timeout**
```bash
# Increase generation timeout
kubectl set env deployment/thermaliq -n gl-009-production \
  SANKEY_GENERATION_TIMEOUT=60

# Enable async generation for large diagrams
kubectl set env deployment/thermaliq -n gl-009-production \
  SANKEY_ASYNC_GENERATION=true

# Restart to apply changes
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Sub-Procedure D: Library Bug**
```bash
# Downgrade to last known good version
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  pip install plotly==5.14.1

# Or upgrade to patched version
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  pip install --upgrade plotly

# Rebuild image
docker build -t ghcr.io/greenlang/thermaliq:v1.2.5-plotly .
docker push ghcr.io/greenlang/thermaliq:v1.2.5-plotly

# Deploy
kubectl set image deployment/thermaliq -n gl-009-production \
  thermaliq=ghcr.io/greenlang/thermaliq:v1.2.5-plotly
```

**Post-Resolution**:
```bash
# Test Sankey generation
for calc_id in $(cat sample_calculations.txt); do
  echo "Testing $calc_id"
  curl -X GET "https://api.greenlang.io/v1/thermaliq/calculations/$calc_id/sankey" \
    -H "Authorization: Bearer $TOKEN" \
    -o "sankey_$calc_id.svg"
done

# Verify error rate reduced
rate(thermaliq_sankey_generation_errors_total[15m]) < 0.01

# Check generation latency
histogram_quantile(0.95, thermaliq_sankey_generation_duration_seconds_bucket) < 10
```

---

## Escalation Procedures

### On-Call Rotation

**Primary On-Call**: PagerDuty schedule "GL-009-Primary"
- Responsible for initial response
- Available 24/7
- Response time: 5 minutes (SEV1), 15 minutes (SEV2)

**Secondary On-Call**: PagerDuty schedule "GL-009-Secondary"
- Escalation after 15 minutes (SEV1) or 30 minutes (SEV2)
- Provides additional support

**Engineering Manager**: Manual escalation
- SEV1: Notify within 15 minutes
- SEV2: Notify within 30 minutes

**VP Engineering**: Manual escalation
- SEV1: Notify within 30 minutes
- SEV2: Notify within 2 hours

### Escalation Triggers

**Automatic Escalation**:
- SEV1 unresolved after 30 minutes
- SEV2 unresolved after 2 hours
- Multiple SEV2+ incidents within 24 hours
- Incident impacting > 50% of users

**Manual Escalation**:
- Need for additional expertise
- External vendor involvement required
- Legal/compliance implications
- Regulatory reporting required

### Escalation Contacts

```yaml
roles:
  primary_oncall:
    pagerduty: GL-009-Primary
    slack: "#oncall-gl009"

  secondary_oncall:
    pagerduty: GL-009-Secondary

  engineering_manager:
    name: "Jane Smith"
    email: "jane.smith@greenlang.io"
    phone: "+1-555-0101"
    slack: "@jane.smith"

  vp_engineering:
    name: "John Doe"
    email: "john.doe@greenlang.io"
    phone: "+1-555-0100"
    slack: "@john.doe"

  database_specialist:
    name: "Alice Johnson"
    email: "alice.johnson@greenlang.io"
    slack: "@alice.johnson"

  security_team:
    email: "security@greenlang.io"
    slack: "#security-incidents"

  compliance_team:
    email: "compliance@greenlang.io"
    slack: "#compliance"
```

---

## Communication Templates

### Initial Incident Notification

**Subject**: [SEV{1-4}] GL-009 THERMALIQ - {Brief Description}

```
INCIDENT SUMMARY
Severity: SEV{1-4}
Service: GL-009 THERMALIQ ThermalEfficiencyCalculator
Started: {timestamp}
Status: Investigating
Impact: {brief impact description}

CURRENT STATUS
{1-2 sentences on current situation}

USER IMPACT
{description of user-facing impact}

ACTIONS TAKEN
- {action 1}
- {action 2}

NEXT STEPS
- {next step 1}
- {next step 2}

ETA TO RESOLUTION
{best estimate or "Unknown - investigating"}

INCIDENT COMMANDER
{name} (@{slack_handle})

INCIDENT CHANNEL
#incident-sev{1-4}-gl009-{date}
```

### Status Update Template

**Subject**: [UPDATE] [SEV{1-4}] GL-009 THERMALIQ - {Brief Description}

```
INCIDENT UPDATE - {timestamp}

Status: {Investigating / Mitigating / Monitoring / Resolved}

UPDATE SUMMARY
{2-3 sentences on progress}

ACTIONS TAKEN SINCE LAST UPDATE
- {action 1}
- {action 2}
- {action 3}

CURRENT SITUATION
{detailed status of affected systems}

NEXT STEPS
- {next step 1}
- {next step 2}

REVISED ETA
{updated estimate}

Questions? Post in #incident-sev{1-4}-gl009-{date}
```

### Resolution Notification

**Subject**: [RESOLVED] [SEV{1-4}] GL-009 THERMALIQ - {Brief Description}

```
INCIDENT RESOLVED - {timestamp}

The incident affecting GL-009 THERMALIQ has been resolved.

INCIDENT SUMMARY
Started: {start_timestamp}
Resolved: {end_timestamp}
Duration: {X hours Y minutes}
Severity: SEV{1-4}

ROOT CAUSE
{1-2 paragraphs explaining what happened}

RESOLUTION
{1-2 paragraphs explaining how it was fixed}

USER IMPACT
{summary of impact to users}

PREVENTION
{what will be done to prevent recurrence}

POST-INCIDENT REVIEW
Scheduled for: {date/time}
Attendees: {list}

We apologize for the inconvenience this caused.

POST-INCIDENT REVIEW ACTIONS
- [ ] Root cause analysis document
- [ ] Update runbooks
- [ ] Implement monitoring improvements
- [ ] Code fixes
```

### Customer Communication Template

**Subject**: Service Disruption - GL-009 THERMALIQ Thermal Efficiency Calculator

```
Dear {Customer Name},

We experienced a service disruption affecting the THERMALIQ Thermal Efficiency Calculator on {date} from {start_time} to {end_time} UTC.

WHAT HAPPENED
{2-3 sentences in customer-friendly language}

IMPACT TO YOUR ACCOUNT
{specific impact to this customer}
- {detail 1}
- {detail 2}

ACTIONS REQUIRED (if any)
{any actions customer needs to take, or "No action required"}

WHAT WE'RE DOING TO PREVENT THIS
{1-2 sentences on preventive measures}

We sincerely apologize for any inconvenience this caused. If you have questions or concerns, please contact support@greenlang.io.

Thank you for your patience.

The GreenLang Team
```

---

## Post-Incident Review

### PIR Template

**Title**: Post-Incident Review: {Incident Description}

**Date**: {PIR Date}
**Incident Date**: {Incident Date}
**Severity**: SEV{1-4}
**Duration**: {X hours Y minutes}
**Incident Commander**: {Name}

---

#### Executive Summary

{2-3 paragraph summary suitable for executives, covering:
- What happened
- Impact
- Root cause
- Resolution
- Prevention}

---

#### Timeline

| Time (UTC) | Event |
|------------|-------|
| {timestamp} | Alert triggered: {alert name} |
| {timestamp} | Incident Commander assigned |
| {timestamp} | Root cause identified |
| {timestamp} | Mitigation started |
| {timestamp} | Service restored |
| {timestamp} | Incident closed |

---

#### Impact Analysis

**User Impact**:
- {X} customers affected
- {Y} calculations failed
- {Z} reports delayed

**Business Impact**:
- Revenue impact: ${amount}
- SLA credits: ${amount}
- Support tickets: {count}

**Technical Impact**:
- {service} unavailable for {duration}
- {X}% error rate
- {Y} data points lost

---

#### Root Cause Analysis

**What Happened**:
{Detailed technical explanation}

**Why It Happened**:
{Contributing factors, including:
- Technical causes
- Process gaps
- Human factors}

**Why It Wasn't Caught Earlier**:
{Gaps in monitoring, testing, or processes}

---

#### What Went Well

- {positive aspect 1}
- {positive aspect 2}
- {positive aspect 3}

---

#### What Didn't Go Well

- {improvement area 1}
- {improvement area 2}
- {improvement area 3}

---

#### Action Items

| Action | Owner | Due Date | Priority |
|--------|-------|----------|----------|
| {Fix root cause} | {name} | {date} | P0 |
| {Improve monitoring} | {name} | {date} | P1 |
| {Update runbook} | {name} | {date} | P2 |
| {Training} | {name} | {date} | P2 |

---

#### Lessons Learned

{3-5 key takeaways}

---

### PIR Meeting Agenda

**Duration**: 60 minutes

**Attendees**:
- Incident Commander
- Engineering team members involved
- Engineering Manager
- SRE team representative
- Customer Success (if customer-impacting)

**Agenda**:

1. **Review Timeline** (10 min)
   - Walk through incident chronologically
   - Identify key decision points

2. **Root Cause Analysis** (15 min)
   - Technical deep dive
   - Contributing factors
   - Why not caught earlier

3. **What Went Well** (10 min)
   - Effective responses
   - Good processes
   - Helpful tools

4. **What Didn't Go Well** (15 min)
   - Improvement areas
   - Process gaps
   - Tool limitations

5. **Action Items** (10 min)
   - Preventive measures
   - Process improvements
   - Tool/monitoring enhancements
   - Assign owners and due dates

---

### Follow-up Actions

**Immediate** (Within 1 week):
- [ ] Fix critical bugs
- [ ] Update monitoring
- [ ] Patch security vulnerabilities

**Short-term** (Within 1 month):
- [ ] Implement preventive measures
- [ ] Update documentation
- [ ] Add/improve alerting
- [ ] Conduct training

**Long-term** (Within 1 quarter):
- [ ] Architectural improvements
- [ ] Process enhancements
- [ ] Tool upgrades

---

## Appendix

### Useful Commands Reference

```bash
# Pod management
kubectl get pods -n gl-009-production -l app=thermaliq
kubectl describe pod <pod-name> -n gl-009-production
kubectl logs <pod-name> -n gl-009-production --tail=200
kubectl logs <pod-name> -n gl-009-production --previous
kubectl exec -it <pod-name> -n gl-009-production -- bash

# Deployment management
kubectl rollout status deployment/thermaliq -n gl-009-production
kubectl rollout history deployment/thermaliq -n gl-009-production
kubectl rollout undo deployment/thermaliq -n gl-009-production
kubectl scale deployment/thermaliq -n gl-009-production --replicas=6

# Resource management
kubectl top pods -n gl-009-production
kubectl top nodes
kubectl set resources deployment/thermaliq -n gl-009-production --limits=cpu=4,memory=8Gi

# Configuration
kubectl get configmap thermaliq-config -n gl-009-production -o yaml
kubectl edit configmap thermaliq-config -n gl-009-production
kubectl set env deployment/thermaliq -n gl-009-production KEY=value

# Database
kubectl exec -it postgres-0 -n gl-009-production -- psql -U thermaliq -d thermaliq_production
kubectl exec -it postgres-0 -n gl-009-production -- pg_dump thermaliq_production > backup.sql

# Redis
kubectl exec -it redis-0 -n gl-009-production -- redis-cli
kubectl exec -it redis-0 -n gl-009-production -- redis-cli INFO
kubectl exec -it redis-0 -n gl-009-production -- redis-cli FLUSHDB

# Metrics
curl http://prometheus:9090/api/v1/query?query=<metric>
curl http://prometheus:9090/api/v1/alerts

# PagerDuty
pd incident list --status triggered
pd incident show <incident-id>
pd incident resolve <incident-id>
```

### Incident Severity Decision Tree

```
Is the service completely unavailable?
├─ YES → SEV1
└─ NO
   └─ Is core functionality degraded?
      ├─ YES
      │  └─ Are users blocked?
      │     ├─ YES → SEV2
      │     └─ NO → SEV3
      └─ NO → SEV4
```

### Emergency Contacts

**24/7 Support**:
- PagerDuty: https://greenlang.pagerduty.com
- Slack: #incidents
- Email: oncall@greenlang.io

**Vendor Support**:
- AWS Support: +1-877-860-2677
- Database Vendor: support@database-vendor.com
- Energy Meter Vendor: +1-555-METER-01

**Internal Escalation**:
- Engineering: #engineering
- Security: #security-incidents
- Executive: #exec-incidents

---

**Document Version**: 1.0.0
**Last Updated**: 2025-11-26
**Next Review**: 2025-12-26
**Owner**: GreenLang SRE Team
