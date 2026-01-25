# Database Runbook

This runbook covers alerts related to PostgreSQL and Redis database systems in the GreenLang platform.

---

## Table of Contents

- [PostgreSQLConnectionPoolHigh](#postgresqlconnectionpoolhigh)
- [PostgreSQLSlowQueries](#postgresqlslowqueries)
- [RedisMemoryHigh](#redismemoryhigh)
- [RedisConnectionsHigh](#redisconnectionshigh)

---

## PostgreSQLConnectionPoolHigh

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | PostgreSQLConnectionPoolHigh |
| **Severity** | Warning |
| **Team** | Platform |
| **Evaluation Interval** | 60s |
| **For Duration** | 5m |
| **Threshold** | 80% of max_connections |

**PromQL Expression:**

```promql
pg_stat_activity_count / pg_settings_max_connections > 0.8
```

### Description

This alert fires when PostgreSQL connection usage exceeds 80% of the configured maximum connections. Connection exhaustion will prevent new queries and cause application failures.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | High | New requests may fail to get DB connections |
| **Data Impact** | Medium | Queries may timeout or fail |
| **SLA Impact** | High | Can cause cascading failures |
| **Revenue Impact** | High | Service degradation |

### Diagnostic Steps

1. **Check current connection count**

   ```bash
   # Total connections
   kubectl exec -n greenlang deploy/postgres -- \
     psql -c "SELECT count(*) FROM pg_stat_activity;"

   # Max connections setting
   kubectl exec -n greenlang deploy/postgres -- \
     psql -c "SHOW max_connections;"
   ```

2. **Identify connection sources**

   ```bash
   # Connections by application
   kubectl exec -n greenlang deploy/postgres -- \
     psql -c "SELECT application_name, count(*)
              FROM pg_stat_activity
              GROUP BY application_name
              ORDER BY count(*) DESC;"

   # Connections by state
   kubectl exec -n greenlang deploy/postgres -- \
     psql -c "SELECT state, count(*)
              FROM pg_stat_activity
              GROUP BY state
              ORDER BY count(*) DESC;"

   # Connections by client address
   kubectl exec -n greenlang deploy/postgres -- \
     psql -c "SELECT client_addr, count(*)
              FROM pg_stat_activity
              WHERE client_addr IS NOT NULL
              GROUP BY client_addr
              ORDER BY count(*) DESC;"
   ```

3. **Check for idle connections**

   ```bash
   # Long-idle connections
   kubectl exec -n greenlang deploy/postgres -- \
     psql -c "SELECT pid, usename, application_name, state,
                     query_start, state_change,
                     now() - state_change as idle_duration
              FROM pg_stat_activity
              WHERE state = 'idle'
              ORDER BY idle_duration DESC
              LIMIT 20;"
   ```

4. **Check for blocked queries**

   ```bash
   # Blocked queries waiting for locks
   kubectl exec -n greenlang deploy/postgres -- \
     psql -c "SELECT blocked_locks.pid AS blocked_pid,
                     blocked_activity.usename AS blocked_user,
                     blocking_locks.pid AS blocking_pid,
                     blocking_activity.usename AS blocking_user,
                     blocked_activity.query AS blocked_statement
              FROM pg_catalog.pg_locks blocked_locks
              JOIN pg_catalog.pg_stat_activity blocked_activity
                ON blocked_activity.pid = blocked_locks.pid
              JOIN pg_catalog.pg_locks blocking_locks
                ON blocking_locks.locktype = blocked_locks.locktype
                AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
                AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
                AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
                AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
                AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
                AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
                AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
                AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
                AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
                AND blocking_locks.pid != blocked_locks.pid
              JOIN pg_catalog.pg_stat_activity blocking_activity
                ON blocking_activity.pid = blocking_locks.pid
              WHERE NOT blocked_locks.granted;"
   ```

5. **Check application connection pool settings**

   ```bash
   # Check application pool configuration
   kubectl get configmap -n greenlang app-config -o yaml | grep -i pool
   ```

### Resolution Steps

#### Scenario 1: Connection leak in application

```bash
# 1. Identify leaking application
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "SELECT application_name, count(*),
                  sum(CASE WHEN state = 'idle' THEN 1 ELSE 0 END) as idle
           FROM pg_stat_activity
           WHERE application_name != ''
           GROUP BY application_name
           ORDER BY count(*) DESC;"

# 2. Restart the leaking application
kubectl rollout restart deployment -n greenlang <leaking-app>

# 3. Fix connection pool configuration
kubectl set env deployment/<leaking-app> -n greenlang \
  DB_POOL_SIZE=10 \
  DB_POOL_MAX_OVERFLOW=5 \
  DB_POOL_TIMEOUT=30 \
  DB_POOL_RECYCLE=1800
```

#### Scenario 2: Long-running idle connections

```bash
# 1. Terminate idle connections older than 1 hour
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "SELECT pg_terminate_backend(pid)
           FROM pg_stat_activity
           WHERE state = 'idle'
             AND state_change < now() - interval '1 hour'
             AND pid != pg_backend_pid();"

# 2. Configure idle connection timeout
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "ALTER SYSTEM SET idle_in_transaction_session_timeout = '5min';"

kubectl exec -n greenlang deploy/postgres -- \
  psql -c "SELECT pg_reload_conf();"
```

#### Scenario 3: Increase max_connections (temporary)

```bash
# 1. Check current setting
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "SHOW max_connections;"

# 2. Increase max_connections (requires restart)
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "ALTER SYSTEM SET max_connections = 200;"

# 3. Restart PostgreSQL
kubectl rollout restart statefulset -n greenlang postgres

# WARNING: Increasing max_connections uses more memory
# Each connection uses ~10MB of RAM
```

#### Scenario 4: Use connection pooler (PgBouncer)

```bash
# 1. Deploy PgBouncer if not already present
kubectl apply -f k8s/pgbouncer/deployment.yaml

# 2. Update applications to connect via PgBouncer
kubectl set env deployment/<app> -n greenlang \
  DATABASE_HOST=pgbouncer.greenlang.svc.cluster.local

# 3. Configure PgBouncer pool mode
# transaction mode is recommended for most apps
```

### Post-Resolution

1. **Verify connection count normalized**

   ```bash
   watch "kubectl exec -n greenlang deploy/postgres -- psql -c 'SELECT count(*) FROM pg_stat_activity;'"
   ```

2. **Set up connection monitoring**

3. **Review application connection pooling** best practices

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Initial response |
| L2 | Database Team | #database-oncall Slack | If >90% connections |
| L3 | Backend Team | #backend-oncall Slack | If connection leak |

---

## PostgreSQLSlowQueries

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | PostgreSQLSlowQueries |
| **Severity** | Warning |
| **Team** | Backend |
| **Evaluation Interval** | 60s |
| **For Duration** | 10m |
| **Threshold** | Mean query time > 1s |

**PromQL Expression:**

```promql
rate(pg_stat_statements_mean_time_seconds[5m]) > 1
```

### Description

This alert fires when the average query execution time exceeds 1 second. Slow queries impact application performance and can cause connection pool exhaustion.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | Medium | Slower response times |
| **Data Impact** | Low | Data integrity not affected |
| **SLA Impact** | Medium | Contributes to latency SLA |
| **Revenue Impact** | Medium | Degraded user experience |

### Diagnostic Steps

1. **Identify slow queries**

   ```bash
   # Top queries by mean execution time
   kubectl exec -n greenlang deploy/postgres -- \
     psql -c "SELECT substring(query, 1, 100) as query,
                     calls,
                     mean_time::numeric(10,2) as mean_ms,
                     total_time::numeric(10,2) as total_ms
              FROM pg_stat_statements
              ORDER BY mean_time DESC
              LIMIT 20;"
   ```

2. **Identify queries by total time**

   ```bash
   # Queries consuming most total time
   kubectl exec -n greenlang deploy/postgres -- \
     psql -c "SELECT substring(query, 1, 100) as query,
                     calls,
                     total_time::numeric(10,2) as total_ms,
                     (total_time/calls)::numeric(10,2) as avg_ms
              FROM pg_stat_statements
              ORDER BY total_time DESC
              LIMIT 20;"
   ```

3. **Check currently running queries**

   ```bash
   # Long-running active queries
   kubectl exec -n greenlang deploy/postgres -- \
     psql -c "SELECT pid, now() - query_start as duration,
                     state, substring(query, 1, 100) as query
              FROM pg_stat_activity
              WHERE state = 'active'
                AND query_start < now() - interval '10 seconds'
              ORDER BY duration DESC;"
   ```

4. **Analyze specific query**

   ```bash
   # Get query execution plan
   kubectl exec -n greenlang deploy/postgres -- \
     psql -c "EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
              SELECT * FROM your_slow_query_here;"
   ```

5. **Check index usage**

   ```bash
   # Tables with low index usage
   kubectl exec -n greenlang deploy/postgres -- \
     psql -c "SELECT schemaname, relname,
                     seq_scan, idx_scan,
                     CASE WHEN seq_scan + idx_scan = 0 THEN 0
                          ELSE round(100.0 * idx_scan / (seq_scan + idx_scan), 2)
                     END as idx_scan_pct
              FROM pg_stat_user_tables
              WHERE seq_scan > 1000
              ORDER BY seq_scan DESC
              LIMIT 20;"
   ```

6. **Check table statistics**

   ```bash
   # Tables needing ANALYZE
   kubectl exec -n greenlang deploy/postgres -- \
     psql -c "SELECT schemaname, relname,
                     last_vacuum, last_autovacuum,
                     last_analyze, last_autoanalyze,
                     n_dead_tup
              FROM pg_stat_user_tables
              ORDER BY n_dead_tup DESC
              LIMIT 20;"
   ```

### Resolution Steps

#### Scenario 1: Missing index

```bash
# 1. Identify query needing index
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "EXPLAIN ANALYZE SELECT * FROM emissions WHERE company_id = 'abc';"

# 2. Create index (use CONCURRENTLY to avoid locking)
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "CREATE INDEX CONCURRENTLY idx_emissions_company_id
           ON emissions(company_id);"

# 3. Verify index is being used
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "EXPLAIN ANALYZE SELECT * FROM emissions WHERE company_id = 'abc';"
```

#### Scenario 2: Outdated statistics

```bash
# 1. Run ANALYZE on affected tables
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "ANALYZE emissions;"

# 2. Or analyze entire database
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "ANALYZE;"

# 3. Check autovacuum settings
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "SELECT name, setting FROM pg_settings
           WHERE name LIKE 'autovacuum%';"
```

#### Scenario 3: Table bloat

```bash
# 1. Check table bloat
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "SELECT schemaname, relname, n_live_tup, n_dead_tup,
                  round(100.0 * n_dead_tup / NULLIF(n_live_tup + n_dead_tup, 0), 2) as dead_pct
           FROM pg_stat_user_tables
           WHERE n_dead_tup > 10000
           ORDER BY n_dead_tup DESC;"

# 2. Run VACUUM on bloated tables
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "VACUUM ANALYZE emissions;"

# 3. For severe bloat, consider VACUUM FULL (locks table)
# Schedule during maintenance window
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "VACUUM FULL emissions;"
```

#### Scenario 4: Query optimization needed

```bash
# 1. Analyze query plan
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "EXPLAIN (ANALYZE, BUFFERS) <slow_query>;"

# Look for:
# - Sequential scans on large tables
# - Nested loops with high row counts
# - Hash joins spilling to disk
# - Sort operations on large datasets

# 2. Common optimizations:
# - Add WHERE clause indexes
# - Add covering indexes for frequently accessed columns
# - Rewrite correlated subqueries as JOINs
# - Use LIMIT for large result sets
# - Consider materialized views for complex aggregations
```

#### Scenario 5: Kill long-running query (emergency)

```bash
# 1. Identify the query PID
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "SELECT pid, query FROM pg_stat_activity
           WHERE state = 'active'
             AND query_start < now() - interval '5 minutes';"

# 2. Cancel the query (graceful)
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "SELECT pg_cancel_backend(<pid>);"

# 3. If cancel doesn't work, terminate (forceful)
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "SELECT pg_terminate_backend(<pid>);"
```

### Post-Resolution

1. **Reset pg_stat_statements** to track improvement

   ```bash
   kubectl exec -n greenlang deploy/postgres -- \
     psql -c "SELECT pg_stat_statements_reset();"
   ```

2. **Monitor query performance** for 24 hours

3. **Document index additions** in schema migrations

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Initial response |
| L2 | Database Team | #database-oncall Slack | If complex query issue |
| L3 | Backend Team | #backend-oncall Slack | If code changes needed |

---

## RedisMemoryHigh

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | RedisMemoryHigh |
| **Severity** | Warning |
| **Team** | Platform |
| **Evaluation Interval** | 60s |
| **For Duration** | 10m |
| **Threshold** | 80% of maxmemory |

**PromQL Expression:**

```promql
redis_memory_used_bytes / redis_memory_max_bytes > 0.8
```

### Description

This alert fires when Redis memory usage exceeds 80% of the configured maximum. High memory usage can trigger eviction policies or cause Redis to reject writes.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | Medium | Cache misses increase, slower responses |
| **Data Impact** | Medium | Keys may be evicted |
| **SLA Impact** | Medium | Impacts calculation latency |
| **Revenue Impact** | Medium | Degraded performance |

### Diagnostic Steps

1. **Check memory usage**

   ```bash
   # Memory info
   kubectl exec -n greenlang deploy/redis -- redis-cli info memory

   # Key metrics:
   # - used_memory_human
   # - maxmemory_human
   # - maxmemory_policy
   ```

2. **Check memory by key pattern**

   ```bash
   # Memory usage by key prefix (sample-based)
   kubectl exec -n greenlang deploy/redis -- redis-cli --bigkeys

   # More detailed memory analysis
   kubectl exec -n greenlang deploy/redis -- redis-cli memory doctor
   ```

3. **Check key count and TTLs**

   ```bash
   # Total key count
   kubectl exec -n greenlang deploy/redis -- redis-cli dbsize

   # Keys by pattern
   kubectl exec -n greenlang deploy/redis -- redis-cli --scan --pattern 'ef:*' | wc -l
   kubectl exec -n greenlang deploy/redis -- redis-cli --scan --pattern 'session:*' | wc -l
   kubectl exec -n greenlang deploy/redis -- redis-cli --scan --pattern 'cache:*' | wc -l

   # Keys without TTL
   kubectl exec -n greenlang deploy/redis -- \
     redis-cli --scan | while read key; do
       ttl=$(redis-cli ttl "$key");
       [ "$ttl" = "-1" ] && echo "$key";
     done | wc -l
   ```

4. **Check eviction policy and stats**

   ```bash
   # Current eviction policy
   kubectl exec -n greenlang deploy/redis -- redis-cli config get maxmemory-policy

   # Eviction stats
   kubectl exec -n greenlang deploy/redis -- redis-cli info stats | grep evicted
   ```

### Resolution Steps

#### Scenario 1: Keys without TTL consuming memory

```bash
# 1. Find keys without TTL
kubectl exec -n greenlang deploy/redis -- \
  redis-cli --scan --pattern '*' | head -1000 | while read key; do
    ttl=$(redis-cli ttl "$key")
    [ "$ttl" = "-1" ] && echo "$key"
  done > /tmp/no_ttl_keys.txt

# 2. Set TTL on keys that should expire
kubectl exec -n greenlang deploy/redis -- \
  redis-cli --scan --pattern 'cache:*' | while read key; do
    redis-cli expire "$key" 3600
  done

# 3. Delete stale keys
kubectl exec -n greenlang deploy/redis -- \
  redis-cli --scan --pattern 'old:*' | xargs redis-cli del
```

#### Scenario 2: Large keys consuming memory

```bash
# 1. Identify large keys
kubectl exec -n greenlang deploy/redis -- redis-cli --bigkeys

# 2. Check specific key size
kubectl exec -n greenlang deploy/redis -- redis-cli memory usage <key>

# 3. If key is a hash/list/set, consider restructuring
# 4. Delete if no longer needed
kubectl exec -n greenlang deploy/redis -- redis-cli del <large_key>
```

#### Scenario 3: Configure eviction policy

```bash
# 1. Check current policy
kubectl exec -n greenlang deploy/redis -- redis-cli config get maxmemory-policy

# 2. Set appropriate policy
# For cache: allkeys-lru (evict least recently used)
kubectl exec -n greenlang deploy/redis -- \
  redis-cli config set maxmemory-policy allkeys-lru

# For sessions: volatile-lru (only evict keys with TTL)
kubectl exec -n greenlang deploy/redis -- \
  redis-cli config set maxmemory-policy volatile-lru

# 3. Persist configuration
kubectl exec -n greenlang deploy/redis -- redis-cli config rewrite
```

#### Scenario 4: Increase Redis memory (if appropriate)

```bash
# 1. Increase maxmemory
kubectl exec -n greenlang deploy/redis -- \
  redis-cli config set maxmemory 4gb

# 2. Persist configuration
kubectl exec -n greenlang deploy/redis -- redis-cli config rewrite

# 3. Update resource limits in deployment
kubectl patch deployment -n greenlang redis \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"redis","resources":{"limits":{"memory":"5Gi"}}}]}}}}'
```

#### Scenario 5: Flush specific prefixes

```bash
# 1. Count keys to be deleted
kubectl exec -n greenlang deploy/redis -- \
  redis-cli --scan --pattern 'temp:*' | wc -l

# 2. Delete keys by pattern (careful - this is destructive)
kubectl exec -n greenlang deploy/redis -- \
  redis-cli --scan --pattern 'temp:*' | xargs redis-cli del

# 3. For large deletions, use UNLINK (async delete)
kubectl exec -n greenlang deploy/redis -- \
  redis-cli --scan --pattern 'temp:*' | xargs redis-cli unlink
```

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Initial response |
| L2 | Platform Team | #platform-oncall Slack | If >90% usage |
| L3 | Backend Team | #backend-oncall Slack | If cache strategy issue |

---

## RedisConnectionsHigh

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | RedisConnectionsHigh |
| **Severity** | Warning |
| **Team** | Platform |
| **Evaluation Interval** | 60s |
| **For Duration** | 5m |
| **Threshold** | 500 connections |

**PromQL Expression:**

```promql
redis_connected_clients > 500
```

### Description

This alert fires when Redis has more than 500 connected clients. High connection counts can exhaust file descriptors and impact Redis performance.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | Medium | Connection timeouts may occur |
| **Data Impact** | Low | Connections rejected |
| **SLA Impact** | Medium | Cache access failures |
| **Revenue Impact** | Medium | Degraded application performance |

### Diagnostic Steps

1. **Check current connections**

   ```bash
   # Connection count
   kubectl exec -n greenlang deploy/redis -- redis-cli info clients

   # Key metrics:
   # - connected_clients
   # - blocked_clients
   # - maxclients
   ```

2. **Identify connection sources**

   ```bash
   # List client connections
   kubectl exec -n greenlang deploy/redis -- redis-cli client list | head -50

   # Count by IP
   kubectl exec -n greenlang deploy/redis -- redis-cli client list | \
     awk -F'addr=' '{print $2}' | awk -F':' '{print $1}' | sort | uniq -c | sort -rn
   ```

3. **Check connection limits**

   ```bash
   # Max clients setting
   kubectl exec -n greenlang deploy/redis -- redis-cli config get maxclients

   # File descriptor limit
   kubectl exec -n greenlang deploy/redis -- cat /proc/$(pgrep redis)/limits | grep "open files"
   ```

4. **Check for connection leaks**

   ```bash
   # Connection age distribution
   kubectl exec -n greenlang deploy/redis -- redis-cli client list | \
     awk -F'age=' '{print $2}' | awk '{print $1}' | sort -n | tail -20
   ```

### Resolution Steps

#### Scenario 1: Connection leak in application

```bash
# 1. Identify leaking application by IP
kubectl exec -n greenlang deploy/redis -- redis-cli client list | \
  awk -F'addr=' '{print $2}' | awk -F':' '{print $1}' | sort | uniq -c | sort -rn | head -10

# 2. Map IP to pod
kubectl get pods -n greenlang -o wide | grep <ip>

# 3. Restart the leaking application
kubectl rollout restart deployment -n greenlang <leaking-app>

# 4. Fix connection pool in application
kubectl set env deployment/<app> -n greenlang \
  REDIS_POOL_SIZE=10 \
  REDIS_POOL_TIMEOUT=20
```

#### Scenario 2: Kill idle connections

```bash
# 1. Find old idle connections
kubectl exec -n greenlang deploy/redis -- redis-cli client list | \
  grep "idle=[0-9]\{4,\}" | head -20

# 2. Kill specific client
kubectl exec -n greenlang deploy/redis -- redis-cli client kill <ip:port>

# 3. Kill all clients from specific IP
kubectl exec -n greenlang deploy/redis -- redis-cli client kill addr <ip>

# 4. Set client timeout
kubectl exec -n greenlang deploy/redis -- redis-cli config set timeout 300
kubectl exec -n greenlang deploy/redis -- redis-cli config rewrite
```

#### Scenario 3: Scale applications with connection pools

```bash
# 1. Check application replica count
kubectl get deployment -n greenlang <app> -o jsonpath='{.spec.replicas}'

# 2. If many replicas, each with its own pool
# Consider using a connection proxy like Twemproxy or Redis Cluster

# 3. Reduce pool size per application
kubectl set env deployment/<app> -n greenlang REDIS_POOL_SIZE=5
```

#### Scenario 4: Increase maxclients

```bash
# 1. Check current limit
kubectl exec -n greenlang deploy/redis -- redis-cli config get maxclients

# 2. Increase maxclients
kubectl exec -n greenlang deploy/redis -- redis-cli config set maxclients 10000

# 3. Also need to increase file descriptor limit
# Edit redis deployment to add:
# securityContext:
#   sysctls:
#   - name: net.core.somaxconn
#     value: "10000"
```

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Initial response |
| L2 | Platform Team | #platform-oncall Slack | If approaching maxclients |
| L3 | Backend Team | #backend-oncall Slack | If connection leak |

---

## Quick Reference Card

| Alert | Severity | Threshold | First Check | Quick Fix |
|-------|----------|-----------|-------------|-----------|
| PostgreSQLConnectionPoolHigh | Warning | >80% | `pg_stat_activity` | Kill idle connections |
| PostgreSQLSlowQueries | Warning | >1s mean | `pg_stat_statements` | Add indexes |
| RedisMemoryHigh | Warning | >80% | `info memory` | Set eviction policy |
| RedisConnectionsHigh | Warning | >500 | `client list` | Kill idle clients |

## Database Health Checklist

### PostgreSQL Daily Checks

```bash
# Connection count
psql -c "SELECT count(*) FROM pg_stat_activity;"

# Long-running queries
psql -c "SELECT pid, now() - query_start as duration, query
         FROM pg_stat_activity WHERE state = 'active'
         AND query_start < now() - interval '1 minute';"

# Table bloat
psql -c "SELECT relname, n_dead_tup FROM pg_stat_user_tables
         WHERE n_dead_tup > 10000 ORDER BY n_dead_tup DESC LIMIT 10;"

# Index usage
psql -c "SELECT relname, idx_scan, seq_scan FROM pg_stat_user_tables
         WHERE seq_scan > idx_scan ORDER BY seq_scan DESC LIMIT 10;"
```

### Redis Daily Checks

```bash
# Memory usage
redis-cli info memory | grep used_memory_human

# Key count
redis-cli dbsize

# Connection count
redis-cli info clients | grep connected_clients

# Evictions
redis-cli info stats | grep evicted_keys
```
