# Deployment Rollback Runbook

**Scenario**: Roll back failed application deployments, database migrations, or infrastructure changes to restore service functionality after deployment issues.

**Severity**: P0 (Production down) / P1 (Degraded service) / P2 (Non-critical issues)

**RTO/RPO**: 5-15 minutes (application) / 15-30 minutes (with database)

**Owner**: Platform Team / DevOps

## Prerequisites

- kubectl access to EKS cluster
- Access to CI/CD system (GitHub Actions, ArgoCD)
- Database migration tools
- Deployment history knowledge
- Communication channels ready

## Detection

### Deployment Failure Indicators

1. **Automated Alerts**:
   - Deployment status failed
   - Health check failures
   - Increased error rate (> 5%)
   - P95 latency degradation (> 2x baseline)

2. **Application Symptoms**:
   - 500 errors in API responses
   - Pods in CrashLoopBackOff state
   - ReadinessProbe failures
   - Database connection errors

3. **Post-Deployment Issues**:
   - New functionality not working
   - Performance degradation
   - Data inconsistencies
   - Third-party integration failures

### Check Deployment Status

```bash
# Check recent deployments
kubectl rollout history deployment/api-gateway -n vcci-scope3

# Check current deployment status
kubectl rollout status deployment/api-gateway -n vcci-scope3

# Check pod status
kubectl get pods -n vcci-scope3 -l app=api-gateway -o wide

# Check events for errors
kubectl get events -n vcci-scope3 --sort-by='.lastTimestamp' | tail -50

# Check pod logs for errors
kubectl logs -n vcci-scope3 -l app=api-gateway --tail=100 | grep -i "error\|exception\|fatal"
```

**Expected Output for Healthy Deployment**:
```
NAME                           READY   STATUS    RESTARTS   AGE
api-gateway-7d9f8b6c5d-abc12   1/1     Running   0          5m
api-gateway-7d9f8b6c5d-def34   1/1     Running   0          5m
api-gateway-7d9f8b6c5d-ghi56   1/1     Running   0          5m
```

## Step-by-Step Procedure

### Part 1: Assessment and Decision

#### Step 1: Identify Problematic Deployment

```bash
# Check deployment revision history
kubectl rollout history deployment/api-gateway -n vcci-scope3

# Expected output:
# REVISION  CHANGE-CAUSE
# 1         Initial deployment
# 2         Update to v1.2.3
# 3         Update to v1.2.4 (current, failing)

# Get current revision number
CURRENT_REVISION=$(kubectl get deployment api-gateway -n vcci-scope3 -o jsonpath='{.metadata.annotations.deployment\.kubernetes\.io/revision}')
echo "Current Revision: $CURRENT_REVISION"

# Get previous stable revision
PREVIOUS_REVISION=$((CURRENT_REVISION - 1))
echo "Previous Stable Revision: $PREVIOUS_REVISION"

# Compare current vs previous configuration
kubectl rollout history deployment/api-gateway -n vcci-scope3 --revision=$CURRENT_REVISION > /tmp/current.yaml
kubectl rollout history deployment/api-gateway -n vcci-scope3 --revision=$PREVIOUS_REVISION > /tmp/previous.yaml
diff /tmp/previous.yaml /tmp/current.yaml
```

#### Step 2: Assess Impact and Decide on Rollback

```bash
# Check error rate
kubectl logs -n vcci-scope3 -l app=api-gateway --since=15m | \
  grep -c "ERROR" | \
  awk '{print "Error count:", $1}'

# Check failed requests
curl -s http://prometheus:9090/api/v1/query?query='rate(http_requests_total{status=~"5.."}[5m])' | \
  jq '.data.result[0].value[1]'

# Decision criteria:
# - Error rate > 5%: Immediate rollback
# - New pods not becoming ready: Immediate rollback
# - Performance degradation > 50%: Immediate rollback
# - Minor issues with workaround available: Consider hotfix instead

# Document decision
cat > /tmp/rollback_decision_$(date +%Y%m%d_%H%M%S).txt << EOF
Deployment Rollback Decision
============================
Time: $(date)
Deployment: api-gateway
Current Version: v1.2.4
Current Revision: $CURRENT_REVISION

Issue: [Description of problem]
Impact: [User impact description]
Error Rate: [Percentage]

Decision: ROLLBACK to revision $PREVIOUS_REVISION
Reason: [Justification]

Authorized by: [Name]
EOF
```

#### Step 3: Notify Stakeholders

```bash
# Send notification
curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "⚠️ Deployment Rollback Initiated",
    "blocks": [
      {
        "type": "section",
        "text": {
          "type": "mrkdwn",
          "text": "*Deployment Rollback*\n*Service*: api-gateway\n*From*: v1.2.4 (rev '$CURRENT_REVISION')\n*To*: v1.2.3 (rev '$PREVIOUS_REVISION')\n*Reason*: [Brief reason]\n*ETA*: 5-10 minutes"
        }
      }
    ]
  }'

# Update status page if applicable
curl -X POST https://api.statuspage.io/v1/incidents \
  -H "Authorization: Bearer $STATUSPAGE_API_KEY" \
  -d "name=API Gateway Deployment Issue" \
  -d "status=investigating" \
  -d "body=We are rolling back a recent deployment due to errors. Service may be degraded."
```

### Part 2: Application Rollback

#### Step 4: Rollback Kubernetes Deployment

```bash
# Quick rollback to previous revision
kubectl rollout undo deployment/api-gateway -n vcci-scope3

# Or rollback to specific revision
kubectl rollout undo deployment/api-gateway -n vcci-scope3 --to-revision=$PREVIOUS_REVISION

# Monitor rollback progress
kubectl rollout status deployment/api-gateway -n vcci-scope3 --watch

# Expected output:
# Waiting for deployment "api-gateway" rollout to finish: 1 old replicas are pending termination...
# Waiting for deployment "api-gateway" rollout to finish: 1 old replicas are pending termination...
# deployment "api-gateway" successfully rolled out

# Verify new pods are running
kubectl get pods -n vcci-scope3 -l app=api-gateway -o wide
```

**Timeline**:
- **T+0s**: Rollback initiated
- **T+10-30s**: New pods starting
- **T+30-60s**: Old pods terminating
- **T+60-120s**: Rollback complete

#### Step 5: Rollback Multiple Services (If Needed)

```bash
# If deployment involved multiple services
SERVICES="api-gateway data-ingestion calculation-engine reporting-service"

for service in $SERVICES; do
  echo "Rolling back $service..."
  kubectl rollout undo deployment/$service -n vcci-scope3

  # Wait for each rollback to complete
  kubectl rollout status deployment/$service -n vcci-scope3 --timeout=5m

  if [ $? -eq 0 ]; then
    echo "✓ $service rolled back successfully"
  else
    echo "✗ $service rollback failed"
  fi
done

# Verify all pods running
kubectl get pods -n vcci-scope3 -o wide
```

#### Step 6: Rollback ConfigMaps and Secrets (If Changed)

```bash
# Check if ConfigMap was updated in problematic deployment
kubectl get configmap app-config -n vcci-scope3 -o yaml > /tmp/current_configmap.yaml

# If ConfigMap needs rollback, restore from version control
git checkout HEAD~1 k8s/configmaps/app-config.yaml
kubectl apply -f k8s/configmaps/app-config.yaml

# Or restore from backup
kubectl apply -f /backup/configmaps/app-config-$(date -d yesterday +%Y%m%d).yaml

# Restart pods to pick up ConfigMap changes
kubectl rollout restart deployment/api-gateway -n vcci-scope3

# If Secret changed
kubectl create secret generic api-keys \
  --from-file=api-key.txt=/backup/secrets/api-key-previous.txt \
  -n vcci-scope3 \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl rollout restart deployment/api-gateway -n vcci-scope3
```

### Part 3: Database Migration Rollback

#### Step 7: Assess Database Changes

```bash
# Check if migration ran as part of deployment
kubectl logs -n vcci-scope3 -l app=migration-job --tail=100

# Check migration status (example using Alembic)
kubectl exec -n vcci-scope3 deployment/api-gateway -- \
  python -m alembic current

# Output shows current migration version
# Expected: revision abc123def456

# Check migration history
kubectl exec -n vcci-scope3 deployment/api-gateway -- \
  python -m alembic history

# Identify which migration to rollback to
CURRENT_MIGRATION="def456ghi789"
PREVIOUS_MIGRATION="abc123def456"

echo "Current Migration: $CURRENT_MIGRATION"
echo "Target Migration: $PREVIOUS_MIGRATION"
```

#### Step 8: Backup Database Before Migration Rollback

```bash
# Create snapshot before rolling back migration
aws rds create-db-snapshot \
  --db-instance-identifier vcci-scope3-prod-postgres \
  --db-snapshot-identifier pre-migration-rollback-$(date +%Y%m%d-%H%M%S)

# Wait for snapshot to complete (typically 5-15 minutes)
aws rds wait db-snapshot-available \
  --db-snapshot-identifier pre-migration-rollback-$(date +%Y%m%d-%H%M%S)

echo "Backup snapshot created successfully"
```

#### Step 9: Execute Migration Rollback

```bash
# Scale down application pods to prevent writes during migration
kubectl scale deployment/api-gateway -n vcci-scope3 --replicas=0
kubectl scale deployment/data-ingestion -n vcci-scope3 --replicas=0
kubectl scale deployment/calculation-engine -n vcci-scope3 --replicas=0

# Wait for pods to terminate
sleep 30

# Run migration rollback (example using Alembic)
kubectl run -it --rm migration-rollback \
  --image=vcci-scope3-api-gateway:v1.2.3 \
  --restart=Never \
  -n vcci-scope3 \
  -- python -m alembic downgrade $PREVIOUS_MIGRATION

# Or for Flyway
kubectl run -it --rm migration-rollback \
  --image=flyway/flyway:latest \
  --restart=Never \
  -n vcci-scope3 \
  -- flyway -url=jdbc:postgresql://$DB_ENDPOINT:5432/scope3_platform \
    -user=$DB_USER -password=$DB_PASSWORD \
    undo

# For Django migrations
kubectl run -it --rm migration-rollback \
  --image=vcci-scope3-api-gateway:v1.2.3 \
  --restart=Never \
  -n vcci-scope3 \
  -- python manage.py migrate app_name $PREVIOUS_MIGRATION

# Verify migration rollback
kubectl exec -n vcci-scope3 deployment/api-gateway -- \
  python -m alembic current

# Scale application back up
kubectl scale deployment/api-gateway -n vcci-scope3 --replicas=3
kubectl scale deployment/data-ingestion -n vcci-scope3 --replicas=2
kubectl scale deployment/calculation-engine -n vcci-scope3 --replicas=4
```

#### Step 10: Handle Data Migration Rollback

```bash
# If migration included data changes that can't be automatically rolled back

# Option A: Restore specific table from snapshot
# Create temporary instance from snapshot
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier vcci-scope3-temp-restore \
  --db-snapshot-identifier pre-migration-rollback-$(date +%Y%m%d-%H%M%S) \
  --db-instance-class db.r6g.large

# Wait for instance
aws rds wait db-instance-available --db-instance-identifier vcci-scope3-temp-restore

# Export affected table
pg_dump -h $TEMP_ENDPOINT -U vcci_admin -d scope3_platform -t affected_table \
  --data-only --column-inserts > /tmp/table_backup.sql

# Restore to production
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
BEGIN;
TRUNCATE TABLE affected_table CASCADE;
\i /tmp/table_backup.sql
COMMIT;
EOF

# Delete temporary instance
aws rds delete-db-instance \
  --db-instance-identifier vcci-scope3-temp-restore \
  --skip-final-snapshot

# Option B: Manual data fix script
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
BEGIN;

-- Reverse data changes made by failed migration
UPDATE entity_master
SET new_column = NULL
WHERE new_column IS NOT NULL;

-- Or restore deleted records from archive table
INSERT INTO entity_master
SELECT * FROM entity_master_archive
WHERE archived_at > '2024-01-15 10:00:00';

COMMIT;
EOF
```

### Part 4: Infrastructure Rollback

#### Step 11: Rollback Helm Chart Deployment

```bash
# If deployed via Helm
helm list -n vcci-scope3

# Check revision history
helm history vcci-scope3-platform -n vcci-scope3

# Rollback to previous version
PREVIOUS_HELM_REVISION=3
helm rollback vcci-scope3-platform $PREVIOUS_HELM_REVISION -n vcci-scope3

# Monitor rollback
kubectl get pods -n vcci-scope3 -w

# Verify rollback
helm status vcci-scope3-platform -n vcci-scope3
```

#### Step 12: Rollback Terraform/Infrastructure Changes

```bash
# If infrastructure change caused issue

# Navigate to terraform directory
cd infrastructure/terraform/prod

# Check current state
terraform state list

# Identify resources that changed
terraform show

# Option A: Revert terraform code
git log --oneline terraform/ | head -10
git revert <commit-hash>

# Apply reverted configuration
terraform plan -out=rollback.tfplan
terraform apply rollback.tfplan

# Option B: Import and modify if resources were deleted
# terraform import aws_security_group.example sg-0123456789abcdef

# Verify infrastructure state
terraform plan  # Should show no changes
```

### Part 5: Validation

#### Step 13: Verify Application Health

```bash
# Check all pods are running
kubectl get pods -n vcci-scope3 -o wide | grep -v "Running.*1/1\|Running.*2/2"

# If output is empty, all pods are healthy

# Check deployment status
kubectl get deployments -n vcci-scope3

# All should show READY matching DESIRED
# NAME                    READY   UP-TO-DATE   AVAILABLE   AGE
# api-gateway            3/3     3            3           2d
# calculation-engine     4/4     4            4           2d
# data-ingestion         2/2     2            2           2d

# Test readiness probes
kubectl describe pod -n vcci-scope3 -l app=api-gateway | grep -A 5 "Readiness"

# Check for recent errors
kubectl logs -n vcci-scope3 -l app=api-gateway --since=5m | grep -c "ERROR"
# Expected: 0 or very low number
```

#### Step 14: Test Application Functionality

```bash
# Test API endpoints
curl -f https://api.vcci-scope3.com/health
# Expected: {"status": "healthy"}

# Test critical business functionality
curl -X POST https://api.vcci-scope3.com/api/v1/emissions/calculate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_TOKEN" \
  -d '{
    "entity_id": "TEST001",
    "period": "2024-01"
  }' | jq .

# Expected: Successful calculation response

# Test database connectivity
kubectl run -it --rm db-test \
  --image=postgres:14 \
  --restart=Never \
  -- psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform -c "SELECT COUNT(*) FROM entity_master;"

# Run automated smoke tests
kubectl run -it --rm smoke-tests \
  --image=vcci-scope3-tests:latest \
  --restart=Never \
  -n vcci-scope3 \
  -- pytest tests/smoke/ -v

# Expected: All tests passing
```

#### Step 15: Monitor Metrics

```bash
# Check error rate
curl -s 'http://prometheus:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])' | \
  jq '.data.result[0].value[1]'
# Expected: "0" or very low value

# Check response time
curl -s 'http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))' | \
  jq '.data.result[0].value[1]'
# Expected: < 2.0 (2 seconds)

# Check database query performance
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
SELECT
  query,
  calls,
  mean_exec_time as mean_ms
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat_statements%'
ORDER BY calls DESC
LIMIT 10;
EOF

# Check resource utilization
kubectl top pods -n vcci-scope3
# Expected: CPU and memory within normal ranges
```

### Part 6: Post-Rollback Actions

#### Step 16: Communicate Resolution

```bash
# Update stakeholders
curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "✅ Deployment Rollback Complete",
    "blocks": [
      {
        "type": "section",
        "text": {
          "type": "mrkdwn",
          "text": "*Rollback Completed*\n*Service*: api-gateway\n*Rolled back to*: v1.2.3 (rev '$PREVIOUS_REVISION')\n*Status*: All systems operational\n*Duration*: 12 minutes"
        }
      }
    ]
  }'

# Update status page
curl -X PATCH https://api.statuspage.io/v1/incidents/[INCIDENT_ID] \
  -H "Authorization: Bearer $STATUSPAGE_API_KEY" \
  -d "status=resolved" \
  -d "body=Deployment issue resolved via rollback. All services operational."
```

#### Step 17: Document Rollback

```bash
# Create rollback report
cat > /tmp/rollback_report_$(date +%Y%m%d_%H%M%S).md << EOF
# Deployment Rollback Report

**Date**: $(date)
**Service**: api-gateway
**Rolled Back From**: v1.2.4 (revision $CURRENT_REVISION)
**Rolled Back To**: v1.2.3 (revision $PREVIOUS_REVISION)

## Issue Description
[Detailed description of the problem]

## Detection
- **Time**: $(date -d '15 minutes ago')
- **Method**: [Automated alert / User report]
- **Symptoms**: [List of symptoms]

## Rollback Actions
1. Rolled back Kubernetes deployment
2. Rolled back database migration from $CURRENT_MIGRATION to $PREVIOUS_MIGRATION
3. Verified application functionality
4. Monitored metrics for 15 minutes

## Timeline
- **T+0**: Issue detected
- **T+5**: Rollback decision made
- **T+7**: Kubernetes rollback initiated
- **T+10**: Database migration rolled back
- **T+12**: Validation completed
- **T+15**: Service fully restored

## Impact
- **Duration**: 15 minutes
- **Users Affected**: ~1,000 concurrent users
- **Error Rate**: 12% during incident
- **Requests Failed**: ~450 requests

## Root Cause
[Initial assessment - to be followed by detailed RCA]

## Preventive Measures
1. [Measure 1]
2. [Measure 2]

## Follow-up Actions
- [ ] Investigate root cause of v1.2.4 failure
- [ ] Fix issues in v1.2.4
- [ ] Update deployment checklist
- [ ] Review rollback procedures
- [ ] Schedule post-incident review

**Prepared by**: [Name]
**Reviewed by**: [Name]
EOF

cat /tmp/rollback_report_$(date +%Y%m%d_%H%M%S).md
```

#### Step 18: Plan Next Deployment

```bash
# Tag failed version for investigation
git tag -a v1.2.4-failed -m "Deployment rolled back due to [issue]"
git push origin v1.2.4-failed

# Create hotfix branch if quick fix needed
git checkout -b hotfix/v1.2.4-fix v1.2.4-failed

# Or wait for proper fix in next release
# Document required fixes for v1.2.5

# Update deployment checklist based on lessons learned
cat >> deployment_checklist.md << EOF

## Additional Checks (Added after rollback incident)
- [ ] Verify migration rollback script exists
- [ ] Test deployment in staging with production-like data
- [ ] Verify backwards compatibility with previous version
- [ ] Ensure graceful degradation if new feature fails
- [ ] Check database migration can be rolled back
EOF
```

## Validation Checklist

- [ ] All pods in Running state with correct replica count
- [ ] No errors in pod logs
- [ ] Health checks passing
- [ ] API endpoints responding correctly
- [ ] Error rate back to baseline (< 0.1%)
- [ ] Response times normal (P95 < 2s)
- [ ] Database queries performing normally
- [ ] Database migration at correct version
- [ ] No data corruption or loss
- [ ] Monitoring metrics normal
- [ ] Stakeholders notified
- [ ] Rollback documented
- [ ] Post-incident review scheduled

## Troubleshooting

### Issue 1: Rollback Pod Not Starting

**Symptoms**: New pods stuck in Pending or CrashLoopBackOff

**Diagnosis**:
```bash
kubectl describe pod -n vcci-scope3 -l app=api-gateway | tail -50
kubectl logs -n vcci-scope3 -l app=api-gateway --previous
```

**Resolution**:
- Check for resource constraints
- Verify image tag exists
- Check ConfigMap/Secret references
- Review readiness probe configuration

### Issue 2: Database Migration Rollback Fails

**Symptoms**: Migration downgrade errors

**Resolution**:
```bash
# Manual intervention required
# Restore from snapshot instead
# See DATA_RECOVERY.md
```

### Issue 3: Partial Rollback State

**Symptoms**: Some services rolled back, others still on new version

**Resolution**:
```bash
# Identify services in inconsistent state
kubectl get deployments -n vcci-scope3 -o custom-columns=NAME:.metadata.name,IMAGE:.spec.template.spec.containers[0].image

# Roll back all to same version
for deployment in api-gateway data-ingestion calculation-engine; do
  kubectl set image deployment/$deployment -n vcci-scope3 \
    $deployment=vcci-scope3-$deployment:v1.2.3
done
```

### Issue 4: Configuration Drift After Rollback

**Symptoms**: Application behavior different from before failed deployment

**Resolution**:
```bash
# Verify all ConfigMaps and Secrets
kubectl get configmap -n vcci-scope3 -o yaml > /tmp/current_config.yaml
diff /backup/configmaps/known-good.yaml /tmp/current_config.yaml

# Restore from version control
git checkout HEAD~5 k8s/
kubectl apply -f k8s/
```

## Related Documentation

- [Incident Response Runbook](./INCIDENT_RESPONSE.md)
- [Data Recovery Runbook](./DATA_RECOVERY.md)
- [Database Failover Runbook](./DATABASE_FAILOVER.md)
- [Kubernetes Deployment Strategies](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [Database Migration Best Practices](../guides/database-migrations.md)

## Appendix: Rollback Decision Matrix

| Symptom | Severity | Action | Rollback? |
|---------|----------|--------|-----------|
| Error rate > 10% | P0 | Immediate rollback | Yes |
| Pods not starting | P0 | Immediate rollback | Yes |
| Performance degradation > 50% | P1 | Rollback | Yes |
| Database migration failed | P1 | Rollback both app and DB | Yes |
| Minor UI bug | P3 | Hotfix in next release | No |
| Feature flag not working | P2 | Disable feature, keep deployment | No |
| Single endpoint error | P2 | Assess impact, consider hotfix | Maybe |

## Contact Information

- **Platform Team**: platform-team@company.com
- **On-Call Engineer**: PagerDuty escalation
- **DevOps**: devops@company.com
- **Database Team**: db-team@company.com
