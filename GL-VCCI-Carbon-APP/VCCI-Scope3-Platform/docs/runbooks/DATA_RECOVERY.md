# Data Recovery Runbook

**Scenario**: Restore data from backups following data loss, corruption, accidental deletion, or ransomware incident using RDS snapshots, S3 versioning, and point-in-time recovery.

**Severity**: P0 (Complete data loss) / P1 (Partial data loss) / P2 (Single record recovery)

**RTO/RPO**:
- RTO: 30 minutes - 4 hours (depending on data size)
- RPO: 5 minutes (automated backups) to 24 hours (daily snapshots)

**Owner**: Database Team / Platform Team

## Prerequisites

- AWS Console access with RDS/S3 permissions
- kubectl access to EKS cluster
- Database credentials and connection strings
- Backup retention policy understanding
- Incident communication established

## Detection

### Data Loss Indicators

1. **Application Reports**:
   - Missing records in database queries
   - Null values where data should exist
   - Historical data unavailable
   - Integrity constraint violations

2. **Database Symptoms**:
   - Table truncated unexpectedly
   - Corrupted database files
   - Replication lag suddenly dropping to zero (after deletion)
   - Foreign key constraint errors

3. **User Reports**:
   - "My data is missing"
   - "Wrong calculation results"
   - "Can't find historical records"

### Identify Recovery Scope

```bash
# Check recent database operations (if audit logging enabled)
kubectl run -it --rm db-audit-check \
  --image=postgres:14 \
  --restart=Never \
  -- psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform -c "\
    SELECT
      schemaname,
      tablename,
      last_vacuum,
      last_autovacuum,
      last_analyze,
      n_tup_ins AS inserts,
      n_tup_upd AS updates,
      n_tup_del AS deletes
    FROM pg_stat_user_tables
    WHERE n_tup_del > 0
    ORDER BY n_tup_del DESC
    LIMIT 20;"

# Check table sizes for anomalies
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
SELECT
  schemaname,
  tablename,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
  pg_total_relation_size(schemaname||'.'||tablename) AS bytes
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY bytes DESC
LIMIT 20;
EOF

# Check for dropped tables
# Review PostgreSQL logs
aws rds download-db-log-file-portion \
  --db-instance-identifier vcci-scope3-prod-postgres \
  --log-file-name error/postgresql.log.$(date +%Y-%m-%d-%H) \
  --output text | grep -i "drop\|truncate\|delete"
```

## Step-by-Step Procedure

### Part 1: Assessment and Planning

#### Step 1: Determine Recovery Point Objective

```bash
# List available RDS snapshots
aws rds describe-db-snapshots \
  --db-instance-identifier vcci-scope3-prod-postgres \
  --snapshot-type automated \
  --query 'DBSnapshots[*].[DBSnapshotIdentifier,SnapshotCreateTime,Status]' \
  --output table | sort -r

# List manual snapshots
aws rds describe-db-snapshots \
  --db-instance-identifier vcci-scope3-prod-postgres \
  --snapshot-type manual \
  --query 'DBSnapshots[*].[DBSnapshotIdentifier,SnapshotCreateTime,Status]' \
  --output table

# Check point-in-time recovery window
aws rds describe-db-instances \
  --db-instance-identifier vcci-scope3-prod-postgres \
  --query 'DBInstances[0].{LatestRestorableTime:LatestRestorableTime,BackupRetentionPeriod:BackupRetentionPeriod}' \
  --output table
```

**Expected Output**:
```
---------------------------------------------------------------------
| DescribeDBInstances                                               |
+-----------------------------+------------------------------------+
| BackupRetentionPeriod       | 35                                 |
| LatestRestorableTime        | 2024-01-15T14:35:00.000Z          |
+-----------------------------+------------------------------------+
```

#### Step 2: Identify Exact Recovery Time

```bash
# Interview stakeholders to determine last known good state
# Ask: "When did you last verify the data was correct?"
# Check application logs for last successful operation

# Review audit logs if available
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
SELECT
  event_time,
  user_name,
  database_name,
  query_text,
  client_addr
FROM audit_log
WHERE table_name = 'entity_master'
  AND event_time > NOW() - INTERVAL '24 hours'
ORDER BY event_time DESC
LIMIT 50;
EOF

# Check S3 object versions for file-based data
aws s3api list-object-versions \
  --bucket vcci-scope3-data-prod \
  --prefix emissions-data/ \
  --max-items 20
```

#### Step 3: Communicate Recovery Plan

```bash
# Notify stakeholders
curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "🔄 Data Recovery Initiated",
    "blocks": [
      {
        "type": "section",
        "text": {
          "type": "mrkdwn",
          "text": "*Data Recovery Operation*\n*Scope*: [Description]\n*Recovery Point*: [Timestamp]\n*ETA*: 30-60 minutes\n*Impact*: Service will be in read-only mode"
        }
      }
    ]
  }'

# Set maintenance mode if needed
kubectl scale deployment api-gateway -n vcci-scope3 --replicas=0
kubectl apply -f maintenance-page.yaml
```

### Part 2: RDS Snapshot Recovery

#### Step 4: Restore from Automated Snapshot

**Scenario**: Complete database recovery

```bash
# Select appropriate snapshot
SNAPSHOT_ID="rds:vcci-scope3-prod-postgres-2024-01-15-06-00"

# Restore to new instance (safer than in-place restore)
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier vcci-scope3-prod-postgres-recovered \
  --db-snapshot-identifier $SNAPSHOT_ID \
  --db-instance-class db.r6g.2xlarge \
  --db-subnet-group-name vcci-scope3-db-subnet-group \
  --vpc-security-group-ids sg-0abcd1234efgh5678 \
  --publicly-accessible false \
  --multi-az \
  --tags Key=Environment,Value=production Key=Purpose,Value=recovery

# Monitor restoration progress (typically 15-45 minutes)
watch -n 30 'aws rds describe-db-instances \
  --db-instance-identifier vcci-scope3-prod-postgres-recovered \
  --query "DBInstances[0].DBInstanceStatus" \
  --output text'

# Wait for instance to be available
aws rds wait db-instance-available \
  --db-instance-identifier vcci-scope3-prod-postgres-recovered

# Get new endpoint
RECOVERY_ENDPOINT=$(aws rds describe-db-instances \
  --db-instance-identifier vcci-scope3-prod-postgres-recovered \
  --query 'DBInstances[0].Endpoint.Address' \
  --output text)

echo "Recovery Instance Endpoint: $RECOVERY_ENDPOINT"
```

#### Step 5: Validate Restored Database

```bash
# Test connection
kubectl run -it --rm db-recovery-test \
  --image=postgres:14 \
  --restart=Never \
  -- psql -h $RECOVERY_ENDPOINT -U vcci_admin -d scope3_platform -c "SELECT version();"

# Verify data integrity
psql -h $RECOVERY_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
-- Check table counts
SELECT
  'entity_master' AS table_name,
  COUNT(*) AS row_count
FROM entity_master
UNION ALL
SELECT
  'emission_factors' AS table_name,
  COUNT(*) AS row_count
FROM emission_factors
UNION ALL
SELECT
  'calculation_results' AS table_name,
  COUNT(*) AS row_count
FROM calculation_results;

-- Verify critical records exist
SELECT
  entity_id,
  entity_name,
  status,
  updated_at
FROM entity_master
WHERE entity_id = '[KNOWN_ENTITY_ID]';
EOF

# Compare record counts with production
PROD_COUNT=$(psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform -tAc "SELECT COUNT(*) FROM entity_master")
RECOVERY_COUNT=$(psql -h $RECOVERY_ENDPOINT -U vcci_admin -d scope3_platform -tAc "SELECT COUNT(*) FROM entity_master")

echo "Production Count: $PROD_COUNT"
echo "Recovery Count: $RECOVERY_COUNT"
echo "Difference: $(($PROD_COUNT - $RECOVERY_COUNT))"
```

#### Step 6: Extract Specific Data (Surgical Recovery)

**Scenario**: Only specific records/tables need recovery

```bash
# Export specific table from recovered instance
pg_dump -h $RECOVERY_ENDPOINT \
  -U vcci_admin \
  -d scope3_platform \
  -t entity_master \
  --data-only \
  --column-inserts \
  -f /tmp/entity_master_recovery.sql

# Filter specific records if needed
psql -h $RECOVERY_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF' > /tmp/specific_records.sql
COPY (
  SELECT * FROM calculation_results
  WHERE calculation_date BETWEEN '2024-01-01' AND '2024-01-15'
    AND entity_id IN ('ENTITY001', 'ENTITY002')
) TO STDOUT WITH CSV HEADER;
EOF

# Review data before importing
head -50 /tmp/entity_master_recovery.sql

# Import into production (in transaction for safety)
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
BEGIN;

-- Disable triggers temporarily to avoid cascading issues
ALTER TABLE entity_master DISABLE TRIGGER ALL;

-- Import recovered data
\i /tmp/entity_master_recovery.sql

-- Re-enable triggers
ALTER TABLE entity_master ENABLE TRIGGER ALL;

-- Verify import
SELECT COUNT(*) FROM entity_master WHERE updated_at > NOW() - INTERVAL '1 minute';

-- If everything looks good, commit
COMMIT;
-- Otherwise: ROLLBACK;
EOF
```

### Part 3: Point-in-Time Recovery (PITR)

#### Step 7: Restore to Specific Timestamp

**Scenario**: Need data as it was at exact time (e.g., before accidental deletion)

```bash
# Determine target time (must be within backup retention window)
TARGET_TIME="2024-01-15T10:30:00Z"

# Perform PITR to new instance
aws rds restore-db-instance-to-point-in-time \
  --source-db-instance-identifier vcci-scope3-prod-postgres \
  --target-db-instance-identifier vcci-scope3-prod-pitr-recovery \
  --restore-time $TARGET_TIME \
  --db-instance-class db.r6g.2xlarge \
  --db-subnet-group-name vcci-scope3-db-subnet-group \
  --vpc-security-group-ids sg-0abcd1234efgh5678 \
  --publicly-accessible false \
  --multi-az \
  --tags Key=Environment,Value=production Key=Purpose,Value=pitr-recovery

# Monitor restoration (typically 30-60 minutes for large databases)
while true; do
  STATUS=$(aws rds describe-db-instances \
    --db-instance-identifier vcci-scope3-prod-pitr-recovery \
    --query 'DBInstances[0].DBInstanceStatus' \
    --output text)
  echo "$(date): Status = $STATUS"
  if [ "$STATUS" = "available" ]; then
    break
  fi
  sleep 60
done

# Get PITR endpoint
PITR_ENDPOINT=$(aws rds describe-db-instances \
  --db-instance-identifier vcci-scope3-prod-pitr-recovery \
  --query 'DBInstances[0].Endpoint.Address' \
  --output text)

echo "PITR Instance Endpoint: $PITR_ENDPOINT"
```

#### Step 8: Compare and Merge Data

```bash
# Compare deleted records
psql -h $PITR_ENDPOINT -U vcci_admin -d scope3_platform -c "\
  SELECT entity_id, entity_name, created_at
  FROM entity_master
  WHERE entity_id NOT IN (
    SELECT entity_id FROM dblink(
      'host=$DB_ENDPOINT dbname=scope3_platform user=vcci_admin password=$DB_PASSWORD',
      'SELECT entity_id FROM entity_master'
    ) AS t(entity_id VARCHAR)
  )
  LIMIT 100;"

# Export missing records
psql -h $PITR_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF' > /tmp/missing_records.sql
COPY (
  SELECT * FROM entity_master
  WHERE entity_id NOT IN (
    -- List of entity_ids that exist in production
    'ENTITY123', 'ENTITY124'  -- etc.
  )
) TO STDOUT WITH CSV HEADER;
EOF

# Import missing records
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
BEGIN;
\copy entity_master FROM '/tmp/missing_records.sql' WITH CSV HEADER;
COMMIT;
EOF
```

### Part 4: S3 Data Recovery

#### Step 9: Recover S3 Objects (Versioned Buckets)

```bash
# List versions of deleted object
aws s3api list-object-versions \
  --bucket vcci-scope3-data-prod \
  --prefix "emissions-data/2024/01/entity_001_emissions.json"

# Identify version to restore
VERSION_ID="Rb2J8KqL9X.YGJmFHNpXCEw8gvDhZ4Qp"

# Restore specific version
aws s3api get-object \
  --bucket vcci-scope3-data-prod \
  --key "emissions-data/2024/01/entity_001_emissions.json" \
  --version-id $VERSION_ID \
  /tmp/recovered_emissions.json

# Verify recovered file
cat /tmp/recovered_emissions.json | jq '.'

# Re-upload as latest version
aws s3 cp /tmp/recovered_emissions.json \
  s3://vcci-scope3-data-prod/emissions-data/2024/01/entity_001_emissions.json

# Or restore by removing delete marker
aws s3api delete-object \
  --bucket vcci-scope3-data-prod \
  --key "emissions-data/2024/01/entity_001_emissions.json" \
  --version-id $DELETE_MARKER_VERSION_ID
```

#### Step 10: Bulk S3 Recovery

```bash
# List all deleted objects in timeframe
aws s3api list-object-versions \
  --bucket vcci-scope3-data-prod \
  --prefix "emissions-data/2024/01/" \
  --query 'DeleteMarkers[?LastModified>=`2024-01-15T00:00:00.000Z`].[Key,VersionId]' \
  --output text > /tmp/deleted_objects.txt

# Create recovery script
cat > /tmp/recover_s3_objects.sh << 'EOF'
#!/bin/bash
while IFS=$'\t' read -r key version_id; do
  echo "Recovering: $key (version: $version_id)"
  aws s3api delete-object \
    --bucket vcci-scope3-data-prod \
    --key "$key" \
    --version-id "$version_id"
done < /tmp/deleted_objects.txt
EOF

chmod +x /tmp/recover_s3_objects.sh

# Review before executing
head -20 /tmp/deleted_objects.txt

# Execute recovery
/tmp/recover_s3_objects.sh

# Verify recovery
aws s3 ls s3://vcci-scope3-data-prod/emissions-data/2024/01/ --recursive | wc -l
```

### Part 5: Application-Level Recovery

#### Step 11: Trigger Data Re-ingestion

**Scenario**: Source data intact, need to rebuild derived data

```bash
# Re-run data ingestion jobs
kubectl create job data-reingestion-$(date +%Y%m%d-%H%M%S) \
  --from=cronjob/daily-emissions-ingestion \
  -n vcci-scope3

# Monitor job progress
kubectl logs -n vcci-scope3 -l job-name=data-reingestion-* --follow

# Re-calculate emissions from source data
kubectl run -it --rm recalculation \
  --image=vcci-scope3-calculation-engine:latest \
  --restart=Never \
  -n vcci-scope3 \
  -- python -m calculation_engine.cli recalculate \
    --start-date 2024-01-01 \
    --end-date 2024-01-15 \
    --entity-id all
```

#### Step 12: Restore from Application Backups

```bash
# List application-level backups
aws s3 ls s3://vcci-scope3-backups/application-data/

# Download backup
aws s3 cp s3://vcci-scope3-backups/application-data/backup-20240115.tar.gz /tmp/

# Extract
tar -xzf /tmp/backup-20240115.tar.gz -C /tmp/recovery/

# Review contents
ls -lh /tmp/recovery/

# Import data via API
for file in /tmp/recovery/entities/*.json; do
  curl -X POST https://api.vcci-scope3.com/api/v1/entities/import \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_TOKEN" \
    -d @$file
  echo "Imported: $file"
done
```

### Part 6: Cutover and Validation

#### Step 13: Switch to Recovered Database

**For complete database recovery**:

```bash
# Create final snapshot of current production (just in case)
aws rds create-db-snapshot \
  --db-instance-identifier vcci-scope3-prod-postgres \
  --db-snapshot-identifier pre-recovery-snapshot-$(date +%Y%m%d-%H%M%S)

# Rename current production instance
aws rds modify-db-instance \
  --db-instance-identifier vcci-scope3-prod-postgres \
  --new-db-instance-identifier vcci-scope3-prod-postgres-old \
  --apply-immediately

# Rename recovered instance to production name
aws rds modify-db-instance \
  --db-instance-identifier vcci-scope3-prod-postgres-recovered \
  --new-db-instance-identifier vcci-scope3-prod-postgres \
  --apply-immediately

# Wait for modifications
aws rds wait db-instance-available --db-instance-identifier vcci-scope3-prod-postgres

# Update DNS if using custom CNAME
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890ABC \
  --change-batch '{
    "Changes": [{
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "db.vcci-scope3.com",
        "Type": "CNAME",
        "TTL": 300,
        "ResourceRecords": [{"Value": "'$NEW_ENDPOINT'"}]
      }
    }]
  }'
```

#### Step 14: Restart Application Services

```bash
# Update database connection strings if needed
kubectl patch secret database-credentials -n vcci-scope3 --patch "
data:
  DB_HOST: $(echo -n $NEW_ENDPOINT | base64)
"

# Restart all application pods
kubectl rollout restart deployment/api-gateway -n vcci-scope3
kubectl rollout restart deployment/data-ingestion -n vcci-scope3
kubectl rollout restart deployment/calculation-engine -n vcci-scope3
kubectl rollout restart deployment/reporting-service -n vcci-scope3

# Monitor restart
kubectl get pods -n vcci-scope3 -w

# Verify all pods are running
kubectl get pods -n vcci-scope3 -o wide
```

#### Step 15: Comprehensive Validation

```bash
# 1. Database connectivity
kubectl run -it --rm db-final-check \
  --image=postgres:14 \
  --restart=Never \
  -- psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform -c "\
    SELECT 'Database accessible' AS status;"

# 2. Record counts validation
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
SELECT
  'entity_master' AS table_name,
  COUNT(*) AS expected_count,
  12567 AS baseline_count,  -- Update with known good count
  COUNT(*) - 12567 AS difference
FROM entity_master;
EOF

# 3. Data integrity checks
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
-- Check for orphaned records
SELECT COUNT(*) FROM calculation_results
WHERE entity_id NOT IN (SELECT entity_id FROM entity_master);

-- Check for null critical fields
SELECT COUNT(*) FROM entity_master WHERE entity_id IS NULL OR entity_name IS NULL;

-- Verify date ranges
SELECT MIN(created_at), MAX(created_at) FROM calculation_results;
EOF

# 4. Application functionality tests
# Test API endpoints
curl -f https://api.vcci-scope3.com/api/v1/entities | jq '.items | length'

# Test calculation
curl -X POST https://api.vcci-scope3.com/api/v1/emissions/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "entity_id": "ENTITY001",
    "period": "2024-01"
  }' | jq .

# 5. User acceptance testing
# Have users verify their critical data

# 6. Monitor error rates
kubectl logs -n vcci-scope3 -l app=api-gateway --since=10m | grep -c ERROR
```

### Part 7: Cleanup and Documentation

#### Step 16: Clean Up Recovery Resources

```bash
# After confirming recovery success (wait 24-48 hours)

# Delete old production instance (if renamed)
aws rds delete-db-instance \
  --db-instance-identifier vcci-scope3-prod-postgres-old \
  --skip-final-snapshot  # Or create final snapshot if desired

# Delete PITR recovery instance
aws rds delete-db-instance \
  --db-instance-identifier vcci-scope3-prod-pitr-recovery \
  --skip-final-snapshot

# Clean up temporary files
rm -rf /tmp/recovery/
rm /tmp/*_recovery.sql

# Archive recovery scripts
tar -czf recovery-$(date +%Y%m%d).tar.gz /tmp/recover_*.sh
aws s3 cp recovery-$(date +%Y%m%d).tar.gz s3://vcci-scope3-backups/recovery-procedures/
```

#### Step 17: Document Recovery

```bash
# Create incident report
cat > /tmp/data_recovery_report_$(date +%Y%m%d).md << EOF
# Data Recovery Incident Report

**Date**: $(date)
**Recovery Point**: $TARGET_TIME
**RTO Achieved**: [X minutes]
**RPO Achieved**: [X minutes]

## Incident Summary
[Description of data loss]

## Root Cause
[What caused the data loss]

## Recovery Actions Taken
1. Identified recovery point: $TARGET_TIME
2. Restored from snapshot/PITR: $SNAPSHOT_ID
3. Validated data integrity
4. Switched to recovered database
5. Validated application functionality

## Data Impact
- Records Lost: [Number]
- Records Recovered: [Number]
- Time Period Affected: [Range]

## Lessons Learned
- [Lesson 1]
- [Lesson 2]

## Action Items
- [ ] Implement additional safeguards
- [ ] Update backup procedures
- [ ] Review access controls
- [ ] Schedule backup restore testing

## Timeline
- **T+0**: Data loss detected
- **T+15m**: Recovery initiated
- **T+45m**: Database restored
- **T+60m**: Validation completed
- **T+75m**: Service restored

EOF

cat /tmp/data_recovery_report_$(date +%Y%m%d).md
```

## Validation Checklist

- [ ] All expected records present in database
- [ ] No orphaned or referential integrity issues
- [ ] Application services running without errors
- [ ] API endpoints responding correctly
- [ ] Users can access their data
- [ ] No performance degradation
- [ ] Monitoring shows normal metrics
- [ ] Backup jobs running successfully
- [ ] Recovery procedures documented
- [ ] Stakeholders notified of completion

## Troubleshooting

### Issue 1: Restored Database Performance Issues

**Symptoms**: Slow queries after restoration

**Resolution**:
```bash
# Analyze and vacuum tables
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
VACUUM ANALYZE;
EOF

# Rebuild indexes
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
REINDEX DATABASE scope3_platform;
EOF

# Update statistics
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
ANALYZE;
EOF
```

### Issue 2: Missing Recent Transactions

**Symptoms**: Data between backup time and current time is missing

**Resolution**:
```bash
# Review application logs for recent transactions
kubectl logs -n vcci-scope3 deployment/api-gateway --since=24h | grep "POST\|PUT\|DELETE"

# Check S3 for file-based submissions
aws s3 ls s3://vcci-scope3-data-prod/incoming/ --recursive

# Re-process recent submissions
# Manually import from source systems
```

### Issue 3: Foreign Key Constraint Violations

**Symptoms**: Restore fails with FK constraint errors

**Resolution**:
```bash
# Disable constraints temporarily
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
ALTER TABLE calculation_results DISABLE TRIGGER ALL;
ALTER TABLE entity_relationships DISABLE TRIGGER ALL;

-- Import data
\i /tmp/recovery_data.sql

-- Re-enable constraints
ALTER TABLE calculation_results ENABLE TRIGGER ALL;
ALTER TABLE entity_relationships ENABLE TRIGGER ALL;

-- Validate constraints
SELECT COUNT(*) FROM calculation_results WHERE entity_id NOT IN (SELECT entity_id FROM entity_master);
EOF
```

## Related Documentation

- [Database Failover Runbook](./DATABASE_FAILOVER.md)
- [Incident Response Runbook](./INCIDENT_RESPONSE.md)
- [Security Incident Runbook](./SECURITY_INCIDENT.md)
- [Backup Strategy Documentation](../architecture/backup-strategy.md)
- [AWS RDS Backup and Restore](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/CHAP_CommonTasks.BackupRestore.html)

## Contact Information

- **Database Team**: db-team@company.com
- **Platform Team**: platform-team@company.com
- **On-Call Engineer**: PagerDuty escalation
- **Data Protection Officer**: dpo@company.com (for GDPR incidents)
