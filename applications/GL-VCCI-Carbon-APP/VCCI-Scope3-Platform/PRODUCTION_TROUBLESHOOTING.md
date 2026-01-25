# GL-VCCI Production Troubleshooting Guide
## Scope 3 Carbon Intelligence Platform v2.0

**Version:** 2.0.0
**Date:** November 8, 2025
**Audience:** DevOps, SRE, Support Engineers

---

## Table of Contents

1. [Quick Diagnostic Tools](#quick-diagnostic-tools)
2. [Common Issues](#common-issues)
3. [API Issues](#api-issues)
4. [Database Issues](#database-issues)
5. [Worker/Queue Issues](#worker-queue-issues)
6. [Performance Issues](#performance-issues)
7. [Data Quality Issues](#data-quality-issues)
8. [Integration Issues](#integration-issues)
9. [Security Issues](#security-issues)
10. [Escalation Procedures](#escalation-procedures)

---

## Quick Diagnostic Tools

### Health Check Commands

```bash
# 1. Overall system health
curl https://api.vcci.greenlang.io/health/ready

# 2. Component health
kubectl get pods -n vcci-production
kubectl top pods -n vcci-production
kubectl top nodes

# 3. Database health
psql $DATABASE_URL -c "SELECT version();"
psql $DATABASE_URL -c "SELECT count(*) FROM pg_stat_activity;"

# 4. Redis health
redis-cli -u $REDIS_URL ping
redis-cli -u $REDIS_URL info stats

# 5. Worker health
celery -A worker inspect active
celery -A worker inspect stats

# 6. Recent errors
kubectl logs -n vcci-production deployment/backend-api --tail=100 | grep ERROR
kubectl logs -n vcci-production deployment/worker --tail=100 | grep ERROR
```

### Monitoring Dashboard

**Grafana:** https://grafana.vcci.greenlang.io/d/vcci-production

**Key Metrics to Check:**
- API request rate (should be > 0)
- API error rate (should be < 1%)
- API latency p95 (should be < 500ms)
- Database CPU (should be < 70%)
- Database connections (should be < 80% of max)
- Worker queue depth (should be < 1000)
- Pod CPU/Memory usage (should be < 80%)

### Log Aggregation

**CloudWatch/ELK:** Search for errors in last 1 hour

```bash
# Kubernetes logs
kubectl logs -n vcci-production -l app=backend-api --since=1h | grep -i error

# Application logs (if using ELK)
curl -XGET "http://elasticsearch:9200/vcci-logs-*/_search" -H 'Content-Type: application/json' -d'
{
  "query": {
    "bool": {
      "must": [
        {"range": {"@timestamp": {"gte": "now-1h"}}},
        {"match": {"level": "ERROR"}}
      ]
    }
  },
  "size": 100
}'
```

---

## Common Issues

### Issue 1: "Service Unavailable" (503 Error)

**Symptoms:**
- API returns 503 errors
- Users cannot access the platform
- Health checks failing

**Possible Causes:**

1. **No healthy pods**
   ```bash
   # Check pod status
   kubectl get pods -n vcci-production

   # If pods are CrashLoopBackOff or Error:
   kubectl describe pod <pod-name> -n vcci-production
   kubectl logs <pod-name> -n vcci-production --previous
   ```

   **Fix:**
   ```bash
   # Restart deployment
   kubectl rollout restart deployment/backend-api -n vcci-production

   # If still failing, check for resource constraints
   kubectl describe nodes | grep -A 5 "Allocated resources"
   ```

2. **Database connection failure**
   ```bash
   # Test database connectivity
   psql $DATABASE_URL -c "SELECT 1;"

   # Check database status (AWS RDS)
   aws rds describe-db-instances --db-instance-identifier vcci-prod
   ```

   **Fix:**
   ```bash
   # Verify credentials
   kubectl get secret vcci-database-credentials -n vcci-production -o json

   # Check security groups (ensure backend can reach DB)
   # Update DATABASE_URL if endpoint changed
   kubectl set env deployment/backend-api DATABASE_URL=$NEW_URL -n vcci-production
   ```

3. **Out of memory**
   ```bash
   # Check pod memory usage
   kubectl top pods -n vcci-production

   # Check for OOMKilled pods
   kubectl get pods -n vcci-production | grep OOMKilled
   ```

   **Fix:**
   ```bash
   # Increase memory limits
   kubectl set resources deployment/backend-api -n vcci-production \
     --limits=memory=4Gi \
     --requests=memory=2Gi
   ```

### Issue 2: Slow API Response Times

**Symptoms:**
- API latency p95 > 2 seconds
- Timeouts reported by users
- Grafana shows latency spike

**Diagnostic Steps:**

1. **Check database slow queries**
   ```sql
   -- Top 10 slowest queries
   SELECT
     query,
     calls,
     mean_exec_time,
     max_exec_time
   FROM pg_stat_statements
   ORDER BY mean_exec_time DESC
   LIMIT 10;
   ```

   **Fix:**
   ```sql
   -- Identify missing indexes
   SELECT
     schemaname,
     tablename,
     attname,
     n_distinct,
     correlation
   FROM pg_stats
   WHERE schemaname = 'public'
   AND correlation < 0.1;

   -- Add missing index
   CREATE INDEX CONCURRENTLY idx_suppliers_carbon_intensity
   ON suppliers(carbon_intensity)
   WHERE carbon_intensity > 0;
   ```

2. **Check for N+1 queries**
   ```bash
   # Enable query logging temporarily
   kubectl set env deployment/backend-api SQLALCHEMY_ECHO=true -n vcci-production

   # Review logs for duplicate queries
   kubectl logs -n vcci-production deployment/backend-api | grep SELECT | sort | uniq -c | sort -rn
   ```

   **Fix:** Update application code to use `joinedload` or `selectinload`

3. **Check Redis cache hit rate**
   ```bash
   redis-cli INFO stats | grep hit_rate
   ```

   If hit rate < 90%, increase cache TTL or cache more aggressively.

### Issue 3: Background Jobs Not Processing

**Symptoms:**
- Worker queue depth increasing
- Calculations not completing
- Users not receiving email notifications

**Diagnostic Steps:**

1. **Check worker status**
   ```bash
   # List active workers
   celery -A worker inspect active

   # Check worker stats
   celery -A worker inspect stats

   # Check queue depth
   redis-cli LLEN celery:intake
   redis-cli LLEN celery:calculator
   ```

2. **Check for stuck tasks**
   ```bash
   # List reserved tasks
   celery -A worker inspect reserved

   # List scheduled tasks
   celery -A worker inspect scheduled
   ```

   **Fix:**
   ```bash
   # Purge stuck tasks (CAUTION: data loss)
   celery -A worker purge

   # Restart workers
   kubectl rollout restart deployment/worker -n vcci-production
   ```

3. **Check for worker errors**
   ```bash
   kubectl logs -n vcci-production deployment/worker --tail=500 | grep ERROR
   ```

   Common errors:
   - `SoftTimeLimitExceeded`: Task taking too long (increase time limit)
   - `MemoryError`: Worker out of memory (increase worker memory)
   - `ConnectionError`: Cannot connect to database/Redis (check connectivity)

---

## API Issues

### 401 Unauthorized

**Cause:** Invalid or expired JWT token

**Fix:**
```bash
# Check JWT configuration
kubectl get configmap vcci-api-config -n vcci-production -o yaml | grep JWT

# Regenerate JWT secret (if compromised)
kubectl create secret generic vcci-jwt-keys \
  --from-literal=JWT_SECRET_KEY=$(openssl rand -hex 32) \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart API to pick up new secret
kubectl rollout restart deployment/backend-api -n vcci-production
```

### 422 Validation Error

**Cause:** Invalid request payload

**Fix:**
```bash
# Check OpenAPI spec
curl https://api.vcci.greenlang.io/openapi.json | jq '.paths."/api/v1/suppliers".post.requestBody'

# Validate payload against schema
echo '{"name": "Test Supplier"}' | jq '.name'
```

### 429 Rate Limit Exceeded

**Cause:** Too many requests from single IP/tenant

**Fix:**
```bash
# Check rate limit configuration
kubectl get configmap vcci-api-config -n vcci-production -o yaml | grep RATE_LIMIT

# Temporarily increase rate limit for VIP tenant
kubectl set env deployment/backend-api RATE_LIMIT_TIER_1=10000 -n vcci-production

# Or whitelist IP
kubectl annotate ingress vcci-ingress nginx.ingress.kubernetes.io/limit-whitelist="1.2.3.4" -n vcci-production
```

### 500 Internal Server Error

**Cause:** Unhandled exception in application code

**Fix:**
```bash
# Find error in logs
kubectl logs -n vcci-production deployment/backend-api --tail=1000 | grep -A 20 "500"

# Common fixes:
# 1. Database connection pool exhausted
kubectl set env deployment/backend-api SQLALCHEMY_POOL_SIZE=50 -n vcci-production

# 2. Unhandled null value
# Update application code to handle null

# 3. Timeout calling external API
# Increase timeout or add retry logic
```

---

## Database Issues

### High CPU Usage (> 80%)

**Diagnostic:**
```sql
-- Active queries
SELECT
  pid,
  now() - query_start AS duration,
  state,
  query
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY duration DESC;

-- Long-running queries
SELECT
  pid,
  now() - query_start AS duration,
  query
FROM pg_stat_activity
WHERE now() - query_start > interval '5 minutes';
```

**Fix:**
```sql
-- Kill long-running query (CAUTION)
SELECT pg_terminate_backend(12345);  -- Replace with actual PID

-- Analyze and vacuum tables
VACUUM ANALYZE suppliers;
VACUUM ANALYZE emissions;

-- Update statistics
ANALYZE;
```

### Connection Pool Exhausted

**Symptoms:**
- `FATAL: remaining connection slots are reserved`
- API returns database connection errors

**Diagnostic:**
```sql
-- Check connection count
SELECT count(*) FROM pg_stat_activity;

-- Check max connections
SHOW max_connections;

-- Connections by state
SELECT state, count(*) FROM pg_stat_activity GROUP BY state;
```

**Fix:**
```bash
# Increase max_connections (requires reboot)
aws rds modify-db-instance \
  --db-instance-identifier vcci-prod \
  --db-parameter-group-name vcci-prod-params \
  --apply-immediately

# Or reduce pool size in application
kubectl set env deployment/backend-api SQLALCHEMY_POOL_SIZE=10 -n vcci-production

# Or add connection pooler (PgBouncer)
kubectl apply -f infrastructure/kubernetes/pgbouncer.yaml
```

### Replication Lag

**Diagnostic:**
```sql
-- Check replication lag (on primary)
SELECT
  client_addr,
  state,
  sync_state,
  replay_lag
FROM pg_stat_replication;
```

**Fix:**
```bash
# Increase wal_sender_timeout
aws rds modify-db-parameter-group \
  --db-parameter-group-name vcci-prod-params \
  --parameters "ParameterName=wal_sender_timeout,ParameterValue=120s"

# Check network connectivity to replica
ping <replica-endpoint>

# If lag persists, consider promoting replica and creating new one
```

### Disk Space Full

**Symptoms:**
- Database writes failing
- `No space left on device` errors

**Diagnostic:**
```sql
-- Check database size
SELECT pg_size_pretty(pg_database_size('vcci_production'));

-- Check table sizes
SELECT
  schemaname,
  tablename,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 10;
```

**Fix:**
```bash
# Immediate: Increase disk size (AWS RDS)
aws rds modify-db-instance \
  --db-instance-identifier vcci-prod \
  --allocated-storage 1000 \
  --apply-immediately

# Long-term: Archive old data
psql $DATABASE_URL -c "DELETE FROM audit_logs WHERE created_at < NOW() - INTERVAL '1 year';"
psql $DATABASE_URL -c "VACUUM FULL audit_logs;"
```

---

## Worker/Queue Issues

### Queue Depth Growing Unbounded

**Symptoms:**
- Redis memory increasing
- Tasks not completing
- Users seeing "calculation pending" for hours

**Diagnostic:**
```bash
# Check queue depths
for queue in intake calculator hotspot engagement reporting; do
  echo "$queue: $(redis-cli LLEN celery:$queue)"
done

# Check active workers
celery -A worker inspect active_queues

# Check worker concurrency
celery -A worker inspect stats | grep pool.max-concurrency
```

**Fix:**
```bash
# Scale up workers
kubectl scale deployment/worker --replicas=10 -n vcci-production

# Or increase concurrency
kubectl set env deployment/worker CELERYD_CONCURRENCY=8 -n vcci-production

# Or purge old tasks (CAUTION: data loss)
redis-cli DEL celery:intake
```

### Worker Memory Leak

**Symptoms:**
- Worker pods getting OOMKilled
- Memory usage growing over time
- Workers restarting frequently

**Diagnostic:**
```bash
# Check worker memory usage
kubectl top pods -n vcci-production | grep worker

# Check for memory-intensive tasks
celery -A worker inspect stats | grep "pool.max-memory-per-child"
```

**Fix:**
```bash
# Set max tasks per worker (forces restart)
kubectl set env deployment/worker CELERYD_MAX_TASKS_PER_CHILD=100 -n vcci-production

# Increase worker memory limit
kubectl set resources deployment/worker \
  --limits=memory=8Gi \
  --requests=memory=4Gi \
  -n vcci-production

# Enable memory profiling
kubectl set env deployment/worker CELERYD_ENABLE_MEMORY_PROFILE=true -n vcci-production
```

### Task Failures

**Common Task Errors:**

1. **SoftTimeLimitExceeded**
   ```bash
   # Increase task time limit
   kubectl set env deployment/worker CELERYD_TASK_SOFT_TIME_LIMIT=3600 -n vcci-production
   ```

2. **Retry limit exceeded**
   ```python
   # Check task retry configuration
   @app.task(bind=True, max_retries=5, default_retry_delay=300)
   def calculate_emissions(self, supplier_id):
       try:
           # Task logic
       except Exception as exc:
           raise self.retry(exc=exc)
   ```

3. **External API timeout**
   ```bash
   # Increase request timeout
   kubectl set env deployment/worker EXTERNAL_API_TIMEOUT=60 -n vcci-production
   ```

---

## Performance Issues

### Slow Dashboard Load

**Diagnostic:**
```bash
# Check frontend bundle size
curl -s https://vcci.greenlang.io | grep "<script"

# Check API response times
curl -w "@curl-format.txt" -o /dev/null -s https://api.vcci.greenlang.io/api/v1/dashboard

# curl-format.txt:
# time_namelookup: %{time_namelookup}\n
# time_connect: %{time_connect}\n
# time_starttransfer: %{time_starttransfer}\n
# time_total: %{time_total}\n
```

**Fix:**
```bash
# Enable CDN caching
kubectl annotate ingress vcci-ingress \
  nginx.ingress.kubernetes.io/cache-control="public, max-age=3600" \
  -n vcci-production

# Enable gzip compression (should already be enabled)
kubectl get configmap vcci-nginx-config -n vcci-production -o yaml | grep gzip
```

### High Database Load

**Diagnostic:**
```sql
-- Check for missing indexes
SELECT
  schemaname,
  tablename,
  attname
FROM pg_stats
WHERE schemaname = 'public'
AND n_distinct > 100
AND NOT EXISTS (
  SELECT 1
  FROM pg_indexes
  WHERE schemaname = pg_stats.schemaname
  AND tablename = pg_stats.tablename
  AND indexdef LIKE '%' || attname || '%'
);

-- Check for sequential scans on large tables
SELECT
  schemaname,
  tablename,
  seq_scan,
  seq_tup_read,
  idx_scan
FROM pg_stat_user_tables
WHERE seq_scan > 1000
AND schemaname = 'public'
ORDER BY seq_scan DESC;
```

**Fix:**
```sql
-- Add missing index
CREATE INDEX CONCURRENTLY idx_emissions_reporting_period
ON emissions(reporting_period)
WHERE reporting_period >= '2024-01-01';

-- Increase work_mem for complex queries
SET work_mem = '256MB';
```

### Redis Cache Misses

**Diagnostic:**
```bash
# Check cache stats
redis-cli INFO stats

# Check hit rate
redis-cli INFO stats | grep keyspace_hits
redis-cli INFO stats | grep keyspace_misses

# Calculate hit rate
# hit_rate = hits / (hits + misses)
```

**Fix:**
```bash
# Increase cache TTL
kubectl set env deployment/backend-api CACHE_TTL=3600 -n vcci-production

# Increase Redis memory
kubectl set resources deployment/redis \
  --limits=memory=8Gi \
  --requests=memory=4Gi \
  -n vcci-production

# Or change eviction policy
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

---

## Data Quality Issues

### Duplicate Suppliers

**Diagnostic:**
```sql
-- Find duplicate suppliers
SELECT
  name,
  COUNT(*)
FROM suppliers
GROUP BY name
HAVING COUNT(*) > 1;
```

**Fix:**
```sql
-- Merge duplicates (manual process)
-- 1. Identify canonical record
-- 2. Update foreign keys to point to canonical
UPDATE emissions SET supplier_id = 'canonical-uuid'
WHERE supplier_id = 'duplicate-uuid';

-- 3. Delete duplicate
DELETE FROM suppliers WHERE id = 'duplicate-uuid';
```

### Missing Emission Factors

**Symptoms:**
- Calculations failing with "emission factor not found"
- Default proxy factors being used

**Diagnostic:**
```sql
-- Check emission factor coverage
SELECT
  category,
  COUNT(DISTINCT industry_code) AS industries_covered
FROM emission_factors
GROUP BY category;
```

**Fix:**
```bash
# Sync emission factors from external sources
python scripts/sync_emission_factors.py --source ecoinvent --category all

# Or manually upload missing factors
psql $DATABASE_URL -f data/emission_factors_supplement.sql
```

### Data Validation Errors

**Diagnostic:**
```sql
-- Check for invalid data
SELECT * FROM suppliers WHERE name IS NULL OR name = '';
SELECT * FROM emissions WHERE total_emissions < 0;
SELECT * FROM emissions WHERE spend_amount <= 0;
```

**Fix:**
```sql
-- Add data validation constraints
ALTER TABLE emissions
ADD CONSTRAINT chk_emissions_positive
CHECK (total_emissions >= 0);

-- Clean existing data
UPDATE emissions SET total_emissions = 0 WHERE total_emissions < 0;
```

---

## Integration Issues

### ERP Connector Failing

**Symptoms:**
- Supplier data not syncing
- "Connection refused" errors
- Authentication failures

**Diagnostic:**
```bash
# Test SAP connector
python -m connectors.sap.test_connection --env production

# Check connector logs
kubectl logs -n vcci-production deployment/backend-api | grep "SAP"

# Verify credentials
kubectl get secret vcci-sap-credentials -n vcci-production -o json | jq -r '.data.password' | base64 -d
```

**Fix:**
```bash
# Rotate credentials
kubectl create secret generic vcci-sap-credentials \
  --from-literal=username=$SAP_USER \
  --from-literal=password=$SAP_PASS \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart to pick up new credentials
kubectl rollout restart deployment/backend-api -n vcci-production

# Or whitelist IP (if firewall issue)
# Contact SAP admin to whitelist outbound IP
```

### Email Delivery Failures

**Symptoms:**
- Users not receiving notifications
- Supplier invitations not sent
- "SMTP error" in logs

**Diagnostic:**
```bash
# Check email configuration
kubectl get configmap vcci-api-config -n vcci-production -o yaml | grep SMTP

# Test email sending
python scripts/test_email.py --to test@example.com

# Check SPF/DKIM records
dig TXT vcci.greenlang.io
```

**Fix:**
```bash
# Update SMTP credentials
kubectl set env deployment/backend-api \
  SMTP_PASSWORD=$NEW_PASSWORD \
  -n vcci-production

# Or switch to SES
kubectl set env deployment/backend-api \
  EMAIL_BACKEND=ses \
  AWS_REGION=us-east-1 \
  -n vcci-production
```

---

## Security Issues

### Suspected Intrusion

**Immediate Actions:**

1. **Isolate affected systems**
   ```bash
   # Deny all ingress traffic
   kubectl apply -f - <<EOF
   apiVersion: networking.k8s.io/v1
   kind: NetworkPolicy
   metadata:
     name: deny-all-ingress
     namespace: vcci-production
   spec:
     podSelector: {}
     policyTypes:
     - Ingress
   EOF
   ```

2. **Preserve evidence**
   ```bash
   # Snapshot all pods
   kubectl get pods -n vcci-production -o yaml > pods-snapshot.yaml

   # Dump logs
   for pod in $(kubectl get pods -n vcci-production -o name); do
     kubectl logs $pod -n vcci-production > logs-$(basename $pod).log
   done

   # Database snapshot
   pg_dump -Fc $DATABASE_URL > db-snapshot-$(date +%Y%m%d_%H%M%S).dump
   ```

3. **Rotate all credentials**
   ```bash
   # Database
   psql $DATABASE_URL -c "ALTER USER vcci_admin WITH PASSWORD '$NEW_DB_PASS';"

   # JWT
   kubectl create secret generic vcci-jwt-keys \
     --from-literal=JWT_SECRET_KEY=$(openssl rand -hex 32) \
     --dry-run=client -o yaml | kubectl apply -f -

   # API keys
   python scripts/rotate_api_keys.py --all
   ```

4. **Escalate to security team**
   ```bash
   # Email: security@greenlang.io
   # Slack: @security-oncall
   # Phone: +1-XXX-XXX-XXXX
   ```

### Brute Force Attack

**Symptoms:**
- Many failed login attempts from same IP
- 401 errors spiking

**Fix:**
```bash
# Block IP at load balancer
kubectl annotate ingress vcci-ingress \
  nginx.ingress.kubernetes.io/limit-req-zone="10r/m" \
  -n vcci-production

# Or block at WAF (CloudFlare/AWS WAF)
aws wafv2 create-ip-set \
  --name BlockedIPs \
  --scope REGIONAL \
  --ip-address-version IPV4 \
  --addresses 1.2.3.4/32
```

---

## Escalation Procedures

### Escalation Matrix

| Severity | Response Time | Escalation Path |
|----------|--------------|-----------------|
| **P0 - Critical** | 15 minutes | On-Call → Incident Commander → CTO |
| **P1 - High** | 1 hour | On-Call → Team Lead |
| **P2 - Medium** | 4 hours | On-Call → Team Lead |
| **P3 - Low** | 24 hours | Support → Engineering |

### When to Escalate

**Escalate to Incident Commander if:**
- RTO at risk (downtime > 30 minutes)
- Data loss detected
- Security breach suspected
- Multiple components failing
- Customer-facing impact affecting > 10 tenants

**Escalate to CTO if:**
- Downtime > 1 hour
- Data corruption
- Confirmed security breach
- Regulatory/compliance violation

### Incident Response

1. **Declare incident**
   ```bash
   # Create incident channel
   /incident create "API downtime affecting all tenants"
   ```

2. **Assemble response team**
   - Incident Commander
   - Technical Lead
   - Communications Lead

3. **Update status page**
   ```bash
   # Statuspage.io API
   curl -X POST https://api.statuspage.io/v1/pages/PAGE_ID/incidents \
     -H "Authorization: OAuth $TOKEN" \
     -d "name=API Degradation" \
     -d "status=investigating"
   ```

4. **Communicate with customers**
   - Post to status page every 30 minutes
   - Email to affected tenants
   - In-app notification

5. **Post-incident review**
   - Root cause analysis (RCA)
   - Action items to prevent recurrence
   - Update runbooks

---

## Quick Reference Card

**Print this and keep at your desk!**

```
┌─────────────────────────────────────────────────────────────┐
│             GL-VCCI PRODUCTION TROUBLESHOOTING              │
├─────────────────────────────────────────────────────────────┤
│ HEALTH CHECKS                                               │
│ curl https://api.vcci.greenlang.io/health/ready            │
│ kubectl get pods -n vcci-production                         │
│                                                             │
│ COMMON FIXES                                                │
│ • Restart API: kubectl rollout restart deployment/backend-api│
│ • Scale workers: kubectl scale deployment/worker --replicas=5│
│ • View logs: kubectl logs -f deployment/backend-api         │
│                                                             │
│ EMERGENCY CONTACTS                                          │
│ • DevOps On-Call: oncall-devops@greenlang.io               │
│ • Security: security@greenlang.io                           │
│ • CTO: +1-XXX-XXX-XXXX                                      │
│                                                             │
│ MONITORING                                                  │
│ • Grafana: https://grafana.vcci.greenlang.io               │
│ • Logs: https://logs.vcci.greenlang.io                     │
│ • Status: https://status.vcci.greenlang.io                 │
└─────────────────────────────────────────────────────────────┘
```

---

**Document Version:** 2.0.0
**Last Updated:** November 8, 2025
**Next Review:** Monthly
**Feedback:** devops@greenlang.io
