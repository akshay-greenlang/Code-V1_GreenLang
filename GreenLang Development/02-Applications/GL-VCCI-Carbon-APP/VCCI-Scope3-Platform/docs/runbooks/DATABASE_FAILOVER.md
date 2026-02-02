# Database Failover Runbook

**Scenario**: Amazon RDS PostgreSQL Multi-AZ automatic or manual failover required due to primary instance failure, maintenance, or degraded performance.

**Severity**: P1 (Critical - Service Impact)

**RTO/RPO**:
- RTO: 2-5 minutes (automatic failover)
- RPO: Near-zero (synchronous replication)

**Owner**: Platform Team / Database Team

## Prerequisites

- AWS Console access with RDS permissions
- kubectl access to EKS cluster
- Database connection credentials
- Monitoring dashboard access (CloudWatch, Grafana)
- Incident communication channel (Slack, PagerDuty)

## Detection

### Automatic Failover Indicators

1. **CloudWatch Alarms**:
   - `DatabaseConnectionErrors` > 50 over 2 minutes
   - `FreeableMemory` < 512 MB
   - `CPUUtilization` > 90% for 10 minutes
   - `ReplicationLag` > 60 seconds

2. **Application Symptoms**:
   - Database connection timeouts
   - 500 errors from API endpoints
   - Increased error rates in application logs
   - Circuit breaker trips

3. **RDS Console Events**:
   - "Multi-AZ failover started" event
   - "DB instance restarting" notification
   - "Recovery of the DB instance" message

### Check Current Database Status

```bash
# Check RDS instance status
aws rds describe-db-instances \
  --db-instance-identifier vcci-scope3-prod-postgres \
  --query 'DBInstances[0].{Status:DBInstanceStatus,MultiAZ:MultiAZ,AvailabilityZone:AvailabilityZone,SecondaryAZ:SecondaryAvailabilityZone}' \
  --output table

# Check for recent events
aws rds describe-events \
  --source-type db-instance \
  --source-identifier vcci-scope3-prod-postgres \
  --duration 60 \
  --output table
```

**Expected Output**:
```
---------------------------------------------------------------------
|                      DescribeDBInstances                          |
+---------------------+---------------+------------+-----------------+
| AvailabilityZone    | MultiAZ       | SecondaryAZ| Status          |
+---------------------+---------------+------------+-----------------+
| us-west-2a          | True          | us-west-2b | available       |
+---------------------+---------------+------------+-----------------+
```

## Step-by-Step Procedure

### Phase 1: Assessment and Monitoring

#### Step 1: Verify Automatic Failover in Progress

```bash
# Monitor RDS events in real-time
aws rds describe-events \
  --source-type db-instance \
  --source-identifier vcci-scope3-prod-postgres \
  --duration 10 \
  --output json | jq -r '.Events[] | "\(.Date) - \(.Message)"'

# Check instance status continuously
watch -n 5 'aws rds describe-db-instances \
  --db-instance-identifier vcci-scope3-prod-postgres \
  --query "DBInstances[0].DBInstanceStatus" \
  --output text'
```

**Expected Timeline**:
- **T+0s**: Failover initiated
- **T+30s**: DNS propagation begins
- **T+60-120s**: New primary promoted
- **T+120-300s**: Connections re-established

#### Step 2: Notify Stakeholders

```bash
# Post to incident channel
curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "🚨 RDS Failover Detected - vcci-scope3-prod-postgres",
    "blocks": [
      {
        "type": "section",
        "text": {
          "type": "mrkdwn",
          "text": "*RDS Multi-AZ Failover in Progress*\n*Instance*: vcci-scope3-prod-postgres\n*Status*: Monitoring\n*ETA*: 2-5 minutes"
        }
      }
    ]
  }'
```

#### Step 3: Monitor Replication Lag

```bash
# Check replication lag from CloudWatch
aws cloudwatch get-metric-statistics \
  --namespace AWS/RDS \
  --metric-name ReplicaLag \
  --dimensions Name=DBInstanceIdentifier,Value=vcci-scope3-prod-postgres \
  --start-time $(date -u -d '10 minutes ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 60 \
  --statistics Average \
  --output table
```

**Healthy Replication Lag**: < 5 seconds

### Phase 2: Manual Failover (If Required)

#### Step 4: Trigger Manual Failover

**When to Trigger**:
- Planned maintenance window
- Primary instance degraded but not failing automatically
- Testing failover procedures

```bash
# Initiate manual failover
aws rds reboot-db-instance \
  --db-instance-identifier vcci-scope3-prod-postgres \
  --force-failover \
  --output json

# Response will include:
# {
#   "DBInstance": {
#     "DBInstanceStatus": "rebooting",
#     ...
#   }
# }
```

**CAUTION**: This will cause 2-5 minutes of downtime. Coordinate with team.

#### Step 5: Monitor Failover Progress

```bash
# Create monitoring script
cat > /tmp/monitor_failover.sh << 'EOF'
#!/bin/bash
START_TIME=$(date +%s)
echo "Failover initiated at $(date)"
echo "================================================"

while true; do
  CURRENT_TIME=$(date +%s)
  ELAPSED=$((CURRENT_TIME - START_TIME))

  STATUS=$(aws rds describe-db-instances \
    --db-instance-identifier vcci-scope3-prod-postgres \
    --query 'DBInstances[0].DBInstanceStatus' \
    --output text)

  AZ=$(aws rds describe-db-instances \
    --db-instance-identifier vcci-scope3-prod-postgres \
    --query 'DBInstances[0].AvailabilityZone' \
    --output text)

  echo "[${ELAPSED}s] Status: $STATUS | AZ: $AZ"

  if [ "$STATUS" = "available" ]; then
    echo "================================================"
    echo "Failover completed in ${ELAPSED} seconds"
    break
  fi

  sleep 5
done
EOF

chmod +x /tmp/monitor_failover.sh
/tmp/monitor_failover.sh
```

### Phase 3: Application Recovery

#### Step 6: Verify DNS Propagation

```bash
# Get current endpoint
ENDPOINT=$(aws rds describe-db-instances \
  --db-instance-identifier vcci-scope3-prod-postgres \
  --query 'DBInstances[0].Endpoint.Address' \
  --output text)

echo "Current Endpoint: $ENDPOINT"

# Test DNS resolution
nslookup $ENDPOINT

# Verify connectivity
nc -zv $ENDPOINT 5432

# Test from application perspective
kubectl run -it --rm debug-db-connection \
  --image=postgres:14 \
  --restart=Never \
  -- psql -h $ENDPOINT -U vcci_admin -d scope3_platform -c "SELECT version();"
```

**Expected Output**:
```
PostgreSQL 14.7 on x86_64-pc-linux-gnu, compiled by gcc (GCC) 7.3.1 20180712 (Red Hat 7.3.1-12), 64-bit
```

#### Step 7: Restart Application Pods

```bash
# Check current pod connection status
kubectl get pods -n vcci-scope3 -l app=api-gateway -o wide

# Check for connection errors in logs
kubectl logs -n vcci-scope3 -l app=api-gateway --tail=50 | grep -i "database\|connection\|error"

# Perform rolling restart to refresh connections
kubectl rollout restart deployment/api-gateway -n vcci-scope3
kubectl rollout restart deployment/data-ingestion -n vcci-scope3
kubectl rollout restart deployment/calculation-engine -n vcci-scope3
kubectl rollout restart deployment/reporting-service -n vcci-scope3

# Monitor restart progress
kubectl rollout status deployment/api-gateway -n vcci-scope3 --timeout=5m
```

#### Step 8: Clear Connection Pool

```bash
# Execute connection pool flush if application supports it
kubectl exec -n vcci-scope3 deployment/api-gateway -- \
  curl -X POST http://localhost:8080/admin/connections/reset

# Or restart PgBouncer if used
kubectl rollout restart deployment/pgbouncer -n vcci-scope3
```

### Phase 4: Validation

#### Step 9: Database Connectivity Tests

```bash
# Test read operations
kubectl run -it --rm db-test-read \
  --image=postgres:14 \
  --restart=Never \
  -- psql -h $ENDPOINT -U vcci_admin -d scope3_platform -c "\
    SELECT
      COUNT(*) as total_entities,
      MAX(updated_at) as last_update
    FROM entity_master
    WHERE status = 'active';"

# Test write operations
kubectl run -it --rm db-test-write \
  --image=postgres:14 \
  --restart=Never \
  -- psql -h $ENDPOINT -U vcci_admin -d scope3_platform -c "\
    CREATE TABLE IF NOT EXISTS health_check_$(date +%Y%m%d_%H%M%S) (
      id SERIAL PRIMARY KEY,
      check_time TIMESTAMP DEFAULT NOW()
    );
    INSERT INTO health_check_$(date +%Y%m%d_%H%M%S) DEFAULT VALUES;
    DROP TABLE health_check_$(date +%Y%m%d_%H%M%S);"

# Verify replication status
psql -h $ENDPOINT -U vcci_admin -d scope3_platform << EOF
SELECT
  client_addr,
  state,
  sync_state,
  replay_lag
FROM pg_stat_replication;
EOF
```

#### Step 10: Application Health Checks

```bash
# Test API endpoints
curl -f https://api.vcci-scope3.com/health/database

# Expected response:
# {
#   "status": "healthy",
#   "database": {
#     "connected": true,
#     "latency_ms": 12,
#     "active_connections": 45,
#     "max_connections": 200
#   }
# }

# Check calculation engine
curl -f https://api.vcci-scope3.com/health/calculations

# Verify data ingestion
kubectl logs -n vcci-scope3 deployment/data-ingestion --tail=20 | grep "successfully processed"
```

#### Step 11: Monitor Error Rates

```bash
# Check error rate in last 5 minutes
kubectl logs -n vcci-scope3 -l app=api-gateway --since=5m | \
  grep -E "ERROR|FATAL" | wc -l

# Query Prometheus for error rate
curl -s 'http://prometheus:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])' | \
  jq '.data.result'

# Check database connection metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/RDS \
  --metric-name DatabaseConnections \
  --dimensions Name=DBInstanceIdentifier,Value=vcci-scope3-prod-postgres \
  --start-time $(date -u -d '5 minutes ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 60 \
  --statistics Average,Maximum \
  --output table
```

### Phase 5: Post-Failover Actions

#### Step 12: Verify Multi-AZ Configuration

```bash
# Confirm Multi-AZ is still enabled
aws rds describe-db-instances \
  --db-instance-identifier vcci-scope3-prod-postgres \
  --query 'DBInstances[0].{MultiAZ:MultiAZ,PrimaryAZ:AvailabilityZone,SecondaryAZ:SecondaryAvailabilityZone,Status:DBInstanceStatus}' \
  --output table

# Check for any pending modifications
aws rds describe-db-instances \
  --db-instance-identifier vcci-scope3-prod-postgres \
  --query 'DBInstances[0].PendingModifiedValues' \
  --output json
```

#### Step 13: Review Performance Metrics

```bash
# Check CPU utilization
aws cloudwatch get-metric-statistics \
  --namespace AWS/RDS \
  --metric-name CPUUtilization \
  --dimensions Name=DBInstanceIdentifier,Value=vcci-scope3-prod-postgres \
  --start-time $(date -u -d '30 minutes ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average,Maximum \
  --output table

# Check IOPS
aws cloudwatch get-metric-statistics \
  --namespace AWS/RDS \
  --metric-name ReadIOPS \
  --dimensions Name=DBInstanceIdentifier,Value=vcci-scope3-prod-postgres \
  --start-time $(date -u -d '30 minutes ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average,Maximum \
  --output table
```

#### Step 14: Document Incident

```bash
# Create incident report
cat > /tmp/failover_report_$(date +%Y%m%d_%H%M%S).md << EOF
# RDS Failover Incident Report

**Date**: $(date)
**Instance**: vcci-scope3-prod-postgres
**Trigger**: [Automatic/Manual]
**Duration**: [X minutes]

## Timeline
- **T+0**: Failover detected/initiated
- **T+X**: DNS propagation completed
- **T+X**: Application pods restarted
- **T+X**: Service fully restored

## Availability Zone Change
- **Before**: [AZ]
- **After**: [AZ]

## Impact
- Affected Services: [List]
- User Impact: [Description]
- Data Loss: None (RPO: 0)

## Root Cause
[Description]

## Action Items
- [ ] Review CloudWatch alarms
- [ ] Update runbook if needed
- [ ] Schedule post-incident review
EOF

cat /tmp/failover_report_$(date +%Y%m%d_%H%M%S).md
```

#### Step 15: Close Incident

```bash
# Update status page
curl -X POST https://api.statuspage.io/v1/incidents/[INCIDENT_ID] \
  -H "Authorization: Bearer $STATUSPAGE_API_KEY" \
  -d "status=resolved" \
  -d "body=Database failover completed successfully. All services restored."

# Notify stakeholders
curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "✅ RDS Failover Completed",
    "blocks": [
      {
        "type": "section",
        "text": {
          "type": "mrkdwn",
          "text": "*RDS Multi-AZ Failover Resolved*\n*Duration*: X minutes\n*Status*: All services operational\n*Report*: [Link to incident report]"
        }
      }
    ]
  }'
```

## Validation Checklist

- [ ] RDS instance status is "available"
- [ ] Multi-AZ configuration is enabled
- [ ] Replication lag < 5 seconds
- [ ] Database connections < 80% of max
- [ ] Application pods all in "Running" state
- [ ] API health checks returning 200 OK
- [ ] Error rate back to baseline (< 0.1%)
- [ ] No connection timeout errors in logs
- [ ] CloudWatch alarms cleared
- [ ] Status page updated
- [ ] Incident documented

## Troubleshooting

### Issue 1: Failover Taking Longer Than 5 Minutes

**Symptoms**: RDS instance stuck in "rebooting" or "modifying" state

**Possible Causes**:
- Large transaction rollback in progress
- Network connectivity issues
- AWS service degradation

**Resolution**:
```bash
# Check AWS service health
curl -s https://status.aws.amazon.com/ | grep -i rds

# Review RDS logs
aws rds download-db-log-file-portion \
  --db-instance-identifier vcci-scope3-prod-postgres \
  --log-file-name error/postgresql.log.2024-01-15-12 \
  --output text

# Contact AWS Support if stuck > 10 minutes
aws support create-case \
  --subject "RDS Failover Delayed - vcci-scope3-prod-postgres" \
  --service-code "amazon-rds" \
  --severity-code "urgent" \
  --category-code "other" \
  --communication-body "Multi-AZ failover initiated at [TIME] still in progress after [X] minutes"
```

### Issue 2: Applications Not Reconnecting

**Symptoms**: Persistent connection errors after failover completes

**Possible Causes**:
- Stale DNS cache
- Connection pool not refreshed
- PgBouncer holding old connections

**Resolution**:
```bash
# Force DNS refresh in pods
kubectl exec -n vcci-scope3 deployment/api-gateway -- sh -c "nscd -i hosts"

# Restart all application deployments
for deployment in api-gateway data-ingestion calculation-engine reporting-service; do
  kubectl rollout restart deployment/$deployment -n vcci-scope3
done

# Check for PgBouncer issues
kubectl logs -n vcci-scope3 deployment/pgbouncer --tail=100 | grep -i "error\|fail"

# Restart PgBouncer
kubectl delete pods -n vcci-scope3 -l app=pgbouncer
```

### Issue 3: High Replication Lag After Failover

**Symptoms**: Replication lag > 60 seconds, not decreasing

**Possible Causes**:
- Large write workload
- Network issues between AZs
- Standby instance undersized

**Resolution**:
```bash
# Check current lag
aws cloudwatch get-metric-statistics \
  --namespace AWS/RDS \
  --metric-name ReplicaLag \
  --dimensions Name=DBInstanceIdentifier,Value=vcci-scope3-prod-postgres \
  --start-time $(date -u -d '10 minutes ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 60 \
  --statistics Average,Maximum \
  --output table

# Identify slow queries
kubectl run -it --rm db-slow-queries \
  --image=postgres:14 \
  --restart=Never \
  -- psql -h $ENDPOINT -U vcci_admin -d scope3_platform -c "\
    SELECT
      pid,
      now() - pg_stat_activity.query_start AS duration,
      query,
      state
    FROM pg_stat_activity
    WHERE state != 'idle'
      AND now() - pg_stat_activity.query_start > interval '5 seconds'
    ORDER BY duration DESC
    LIMIT 10;"

# Consider temporary rate limiting
# Update application config to reduce write load if necessary
```

### Issue 4: Connection Pool Exhaustion

**Symptoms**: "too many connections" errors

**Possible Causes**:
- Application not releasing connections
- Connection pool misconfigured
- Spike in traffic during recovery

**Resolution**:
```bash
# Check current connections
kubectl run -it --rm db-connections \
  --image=postgres:14 \
  --restart=Never \
  -- psql -h $ENDPOINT -U vcci_admin -d scope3_platform -c "\
    SELECT
      COUNT(*) as total_connections,
      state,
      application_name
    FROM pg_stat_activity
    GROUP BY state, application_name
    ORDER BY total_connections DESC;"

# Terminate idle connections older than 5 minutes
psql -h $ENDPOINT -U vcci_admin -d scope3_platform << EOF
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'idle'
  AND state_change < NOW() - INTERVAL '5 minutes'
  AND pid <> pg_backend_pid();
EOF

# Temporarily increase max_connections if needed (requires restart)
aws rds modify-db-parameter-group \
  --db-parameter-group-name vcci-scope3-prod-params \
  --parameters "ParameterName=max_connections,ParameterValue=300,ApplyMethod=pending-reboot"
```

## Related Documentation

- [Incident Response Runbook](./INCIDENT_RESPONSE.md)
- [Data Recovery Runbook](./DATA_RECOVERY.md)
- [Performance Tuning Runbook](./PERFORMANCE_TUNING.md)
- [AWS RDS Multi-AZ Documentation](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Concepts.MultiAZ.html)
- [PostgreSQL High Availability](https://www.postgresql.org/docs/current/high-availability.html)
- [Connection Pooling Best Practices](../guides/database-connection-pooling.md)

## Appendix: Monitoring Queries

### CloudWatch Dashboard

Key metrics to monitor:
- `DatabaseConnections` - Active connections
- `CPUUtilization` - CPU usage percentage
- `FreeableMemory` - Available memory
- `ReadIOPS` / `WriteIOPS` - I/O operations
- `ReplicaLag` - Replication delay
- `NetworkReceiveThroughput` - Network in
- `NetworkTransmitThroughput` - Network out

### Automated Failover Test Schedule

```bash
# Schedule quarterly failover tests (off-hours)
# Add to maintenance calendar:
# - Q1: First Sunday of March, 2:00 AM UTC
# - Q2: First Sunday of June, 2:00 AM UTC
# - Q3: First Sunday of September, 2:00 AM UTC
# - Q4: First Sunday of December, 2:00 AM UTC
```

## Contact Information

- **On-Call Engineer**: PagerDuty escalation
- **Database Team**: db-team@company.com
- **AWS Support**: Premium Support case
- **Incident Commander**: Defined in PagerDuty rotation
