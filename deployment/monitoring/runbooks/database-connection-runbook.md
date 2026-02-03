# Database Connection Issues Runbook

## Alert Names
- `RDSHighConnectionCount`
- `DatabaseConnectionPoolExhausted`
- `DatabaseConnectionPoolCritical`
- `HighDatabaseQueryErrorRate`

## Overview

This runbook provides guidance for diagnosing and resolving database connection issues in GreenLang. Database connection problems can manifest as connection pool exhaustion, connection timeouts, or connection refused errors.

---

## Quick Reference

| Severity | Condition | Response Time | Escalation |
|----------|-----------|---------------|------------|
| Warning  | Pool >80% utilized | 15 minutes | On-call engineer |
| Critical | Pool >95% utilized | 5 minutes | On-call + DBA |
| Critical | Connection failures | 5 minutes | Incident commander |

---

## Common Symptoms

1. **Application Errors**
   - "Connection refused" errors
   - "Connection timeout" errors
   - "Too many connections" errors
   - Slow query responses

2. **Metrics Indicators**
   - High connection count on RDS
   - Connection pool utilization >80%
   - Increased query error rate
   - Rising database latency

---

## Diagnosis Steps

### 1. Check RDS Connection Status

```bash
# Get RDS metrics via AWS CLI
aws cloudwatch get-metric-statistics \
  --namespace AWS/RDS \
  --metric-name DatabaseConnections \
  --dimensions Name=DBInstanceIdentifier,Value=greenlang-db \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
  --period 60 \
  --statistics Average

# Check RDS instance status
aws rds describe-db-instances \
  --db-instance-identifier greenlang-db \
  --query 'DBInstances[0].{Status:DBInstanceStatus,Connections:Endpoint}'
```

### 2. Check Application Connection Pool

```bash
# Check application metrics
kubectl exec -it <app-pod> -n greenlang -- curl http://localhost:8000/metrics | grep db_connection

# Check application logs for connection errors
kubectl logs -l app=greenlang-app -n greenlang --tail=100 | grep -i "connection\|database\|pool"
```

### 3. Connect to Database and Diagnose

```bash
# Connect to the database
kubectl exec -it <app-pod> -n greenlang -- psql $DATABASE_URL

# Or use port-forwarding
kubectl port-forward svc/database 5432:5432 -n greenlang
psql -h localhost -U greenlang_admin -d greenlang
```

```sql
-- Check current connections
SELECT count(*) as total_connections,
       count(*) FILTER (WHERE state = 'active') as active,
       count(*) FILTER (WHERE state = 'idle') as idle,
       count(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction
FROM pg_stat_activity
WHERE datname = 'greenlang';

-- Check connection limit
SHOW max_connections;

-- Check connections by application/user
SELECT usename, application_name, client_addr, count(*)
FROM pg_stat_activity
WHERE datname = 'greenlang'
GROUP BY usename, application_name, client_addr
ORDER BY count(*) DESC;

-- Find long-running queries
SELECT pid, now() - pg_stat_activity.query_start AS duration, query, state
FROM pg_stat_activity
WHERE datname = 'greenlang'
  AND state != 'idle'
ORDER BY duration DESC
LIMIT 10;

-- Find idle connections holding transactions
SELECT pid, now() - pg_stat_activity.xact_start AS duration, query, state
FROM pg_stat_activity
WHERE datname = 'greenlang'
  AND state = 'idle in transaction'
ORDER BY duration DESC;
```

### 4. Check Network Connectivity

```bash
# Test connectivity from application pod
kubectl exec -it <app-pod> -n greenlang -- nc -zv <rds-endpoint> 5432

# Check DNS resolution
kubectl exec -it <app-pod> -n greenlang -- nslookup <rds-endpoint>

# Check security group allows connection (AWS CLI)
aws ec2 describe-security-groups --group-ids <rds-security-group-id>
```

### 5. Check for Connection Leaks

```bash
# Check application logs for unclosed connections
kubectl logs -l app=greenlang-app -n greenlang --since=1h | grep -E "connection leak|unclosed|not returned to pool"

# Monitor connection count over time
watch -n 5 'kubectl exec -it <app-pod> -n greenlang -- curl -s http://localhost:8000/metrics | grep db_connections_active'
```

---

## Remediation Steps

### Immediate Actions

#### 1. Kill Idle Connections (If Pool is Exhausted)

```sql
-- Kill connections idle for more than 5 minutes
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'greenlang'
  AND state = 'idle'
  AND query_start < now() - interval '5 minutes';

-- Kill idle in transaction connections (careful!)
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'greenlang'
  AND state = 'idle in transaction'
  AND xact_start < now() - interval '10 minutes';
```

#### 2. Restart Application Pods (Clear Local Connection Pools)

```bash
# Rolling restart of deployment
kubectl rollout restart deployment greenlang-app -n greenlang

# Watch the rollout
kubectl rollout status deployment greenlang-app -n greenlang
```

#### 3. Increase RDS Max Connections (Temporary)

```bash
# Modify parameter group
aws rds modify-db-parameter-group \
  --db-parameter-group-name greenlang-params \
  --parameters "ParameterName=max_connections,ParameterValue=200,ApplyMethod=immediate"

# Note: Some parameters require reboot
aws rds reboot-db-instance --db-instance-identifier greenlang-db
```

#### 4. Scale Down Non-Critical Services

```bash
# Temporarily scale down less critical consumers
kubectl scale deployment greenlang-workers -n greenlang --replicas=1
```

### Application-Level Fixes

#### 1. Adjust Connection Pool Settings

```python
# Example SQLAlchemy configuration
engine = create_engine(
    DATABASE_URL,
    pool_size=10,          # Base pool size
    max_overflow=20,       # Additional connections allowed
    pool_timeout=30,       # Wait time for connection
    pool_recycle=1800,     # Recycle connections after 30 min
    pool_pre_ping=True,    # Test connections before use
)
```

#### 2. Implement Connection Pooling Middleware

```yaml
# Deploy PgBouncer as a sidecar or separate service
apiVersion: v1
kind: ConfigMap
metadata:
  name: pgbouncer-config
  namespace: greenlang
data:
  pgbouncer.ini: |
    [databases]
    greenlang = host=<rds-endpoint> port=5432 dbname=greenlang

    [pgbouncer]
    listen_addr = 0.0.0.0
    listen_port = 6432
    auth_type = md5
    auth_file = /etc/pgbouncer/userlist.txt
    pool_mode = transaction
    max_client_conn = 500
    default_pool_size = 25
    min_pool_size = 5
    reserve_pool_size = 5
    server_lifetime = 3600
    server_idle_timeout = 600
```

#### 3. Fix Connection Leaks

```python
# Always use context managers
from contextlib import contextmanager

@contextmanager
def get_db_connection():
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)  # Always return to pool

# Usage
with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
```

### Infrastructure-Level Fixes

#### 1. Scale RDS Instance

```bash
# Scale to larger instance class
aws rds modify-db-instance \
  --db-instance-identifier greenlang-db \
  --db-instance-class db.r5.large \
  --apply-immediately

# Monitor modification status
aws rds describe-db-instances \
  --db-instance-identifier greenlang-db \
  --query 'DBInstances[0].DBInstanceStatus'
```

#### 2. Add Read Replicas

```bash
# Create read replica for read-heavy workloads
aws rds create-db-instance-read-replica \
  --db-instance-identifier greenlang-db-replica \
  --source-db-instance-identifier greenlang-db \
  --db-instance-class db.r5.large
```

#### 3. Enable RDS Proxy

```bash
# Create RDS Proxy for connection pooling
aws rds create-db-proxy \
  --db-proxy-name greenlang-proxy \
  --engine-family POSTGRESQL \
  --auth '{"AuthScheme":"SECRETS","SecretArn":"arn:aws:secretsmanager:...","IAMAuth":"DISABLED"}' \
  --role-arn arn:aws:iam::...:role/rds-proxy-role \
  --vpc-subnet-ids subnet-123 subnet-456 \
  --vpc-security-group-ids sg-789
```

---

## Prevention Checklist

### Application Best Practices

1. **Connection Pooling**
   - Use connection pools (SQLAlchemy, HikariCP, etc.)
   - Set appropriate pool sizes (start small, scale up)
   - Configure connection timeouts

2. **Connection Management**
   - Always close connections (use context managers)
   - Set query timeouts
   - Handle connection errors gracefully

3. **Query Optimization**
   - Use prepared statements
   - Avoid long-running transactions
   - Implement proper indexing

### Infrastructure Best Practices

1. **Monitoring**
   - Alert on connection pool utilization >70%
   - Monitor RDS connection count
   - Track query latencies

2. **Scaling**
   - Right-size RDS instance
   - Use read replicas for read-heavy workloads
   - Consider RDS Proxy for serverless/Lambda

3. **Configuration**
   - Set appropriate `max_connections`
   - Configure `idle_in_transaction_session_timeout`
   - Enable `log_connections` for debugging

---

## Verification

After remediation:

```bash
# Check connection count is stable
kubectl exec -it <app-pod> -n greenlang -- curl http://localhost:8000/metrics | grep db_connections

# Verify application is healthy
kubectl exec -it <app-pod> -n greenlang -- curl http://localhost:8000/health

# Check RDS connections
aws cloudwatch get-metric-statistics \
  --namespace AWS/RDS \
  --metric-name DatabaseConnections \
  --dimensions Name=DBInstanceIdentifier,Value=greenlang-db \
  --start-time $(date -u -d '10 minutes ago' +%Y-%m-%dT%H:%M:%SZ) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
  --period 60 \
  --statistics Average
```

---

## Escalation

1. **Warning Level (Pool >80%)**
   - On-call engineer investigates
   - Slack: #greenlang-oncall

2. **Critical Level (Pool >95% or Connection Failures)**
   - Page DBA on-call
   - Page incident commander if user-facing impact
   - Slack: #greenlang-incidents

3. **If RDS is Unresponsive**
   - Contact AWS Support
   - Consider failover to standby (Multi-AZ)
   - Activate disaster recovery plan

---

## Related Runbooks

- [High CPU Usage Runbook](./high-cpu-runbook.md)
- [Pod Crash Loop Runbook](./pod-crash-loop-runbook.md)
- [RDS Performance Runbook](./rds-performance-runbook.md)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-01-15 | Platform Team | Initial version |
| 1.1 | 2024-02-01 | DBA Team | Added PgBouncer section |
| 1.2 | 2024-02-15 | SRE Team | Added RDS Proxy guidance |
