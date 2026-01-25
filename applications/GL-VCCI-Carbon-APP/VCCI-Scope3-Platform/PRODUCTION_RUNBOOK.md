# GL-VCCI Production Runbook
## Operational Procedures for Scope 3 Carbon Intelligence Platform v2.0

**Version:** 2.0.0
**Date:** November 8, 2025
**Audience:** DevOps, SRE, Operations Team

---

## Daily Operations

### Morning Health Check (9:00 AM UTC)

```bash
#!/bin/bash
# Daily health check script

echo "=== GL-VCCI Daily Health Check ==="
echo "Date: $(date)"
echo

# 1. System Health
echo "1. System Health"
kubectl get pods -n vcci-production | grep -v Running
curl -s https://api.vcci.greenlang.io/health/ready | jq '.status'

# 2. Key Metrics (last 24h)
echo "2. Key Metrics"
echo "- API Requests: $(curl -s 'http://prometheus:9090/api/v1/query?query=sum(rate(http_requests_total[24h]))' | jq -r '.data.result[0].value[1]')"
echo "- Error Rate: $(curl -s 'http://prometheus:9090/api/v1/query?query=sum(rate(http_requests_total{status=~"5.."}[24h]))/sum(rate(http_requests_total[24h]))' | jq -r '.data.result[0].value[1]')"
echo "- Active Tenants: $(psql $DATABASE_URL -tAc "SELECT count(DISTINCT tenant_id) FROM sessions WHERE last_active > NOW() - INTERVAL '24 hours'")"

# 3. Database Health
echo "3. Database"
psql $DATABASE_URL -c "SELECT 'Connections', count(*) FROM pg_stat_activity UNION ALL SELECT 'Size (GB)', pg_database_size('vcci_production')::bigint/1024/1024/1024"

# 4. Worker Queues
echo "4. Worker Queues"
for queue in intake calculator hotspot engagement reporting; do
  depth=$(redis-cli LLEN celery:$queue)
  echo "- $queue: $depth tasks"
done

# 5. Recent Errors
echo "5. Recent Errors (last hour)"
kubectl logs -n vcci-production deployment/backend-api --since=1h | grep ERROR | wc -l

# 6. Backup Status
echo "6. Last Backup"
aws s3 ls s3://vcci-backups/daily/ | tail -1

echo
echo "=== Health Check Complete ==="
```

### Weekly Tasks (Every Monday)

1. **Review Performance Metrics**
   - Check Grafana dashboard for trends
   - Review slow query log
   - Analyze API latency p95/p99

2. **Capacity Planning**
   ```bash
   # Check resource utilization trends
   kubectl top nodes
   kubectl top pods -n vcci-production --sort-by=memory

   # Database size trend
   psql $DATABASE_URL -c "SELECT pg_size_pretty(pg_database_size('vcci_production'))"
   ```

3. **Certificate Expiry Check**
   ```bash
   # Check SSL certificate expiry
   echo | openssl s_client -servername api.vcci.greenlang.io -connect api.vcci.greenlang.io:443 2>/dev/null | openssl x509 -noout -dates
   ```

4. **Security Updates**
   ```bash
   # Check for security updates
   kubectl get pods -n vcci-production -o jsonpath="{.items[*].spec.containers[*].image}" | tr ' ' '\n' | sort -u | xargs -I {} trivy image {}
   ```

### Monthly Tasks

1. **Disaster Recovery Test**
   - Restore latest backup to staging
   - Verify data integrity
   - Document any issues

2. **Access Review**
   - Review user access permissions
   - Remove inactive users
   - Audit admin accounts

3. **Compliance Check**
   - Review audit logs
   - Check SOC 2 controls
   - Update compliance register

---

## Operational Procedures

### Deploying a New Release

**Pre-Deployment:**
```bash
# 1. Review change log
git log --oneline v2.0.0..v2.1.0

# 2. Run validation script
python scripts/validate_production_env.py --env production

# 3. Notify team
/announce "Deploying v2.1.0 to production at $(date)"
```

**Deployment:**
```bash
# 4. Tag release
git tag -a v2.1.0 -m "Release v2.1.0"
git push origin v2.1.0

# 5. Build and push images
./scripts/build.sh v2.1.0
./scripts/push.sh v2.1.0

# 6. Update manifests
kubectl set image deployment/backend-api backend-api=vcci/backend-api:v2.1.0 -n vcci-production

# 7. Watch rollout
kubectl rollout status deployment/backend-api -n vcci-production

# 8. Verify health
curl https://api.vcci.greenlang.io/health/ready
```

**Post-Deployment:**
```bash
# 9. Run smoke tests
pytest tests/smoke/ -v

# 10. Monitor for 30 minutes
watch -n 30 'kubectl get pods -n vcci-production && kubectl top pods -n vcci-production'

# 11. Update status page
curl -X POST statuspage.io/api/... -d "Deployment complete"
```

### Scaling Operations

**Scale Up (Anticipated High Load):**
```bash
# Before major event (e.g., reporting deadline)

# 1. Scale API servers
kubectl scale deployment/backend-api --replicas=10 -n vcci-production

# 2. Scale workers
kubectl scale deployment/worker --replicas=20 -n vcci-production

# 3. Increase database connections
# (Ensure max_connections can handle increased load)

# 4. Monitor closely
watch -n 10 'kubectl top pods -n vcci-production'
```

**Scale Down (After High Load Period):**
```bash
# Return to normal capacity

kubectl scale deployment/backend-api --replicas=3 -n vcci-production
kubectl scale deployment/worker --replicas=3 -n vcci-production
```

### Adding a New Tenant

```bash
# 1. Create tenant in database
psql $DATABASE_URL <<EOF
INSERT INTO tenants (name, slug, subscription_tier, created_at)
VALUES ('ACME Corp', 'acme-corp', 'enterprise', NOW())
RETURNING id;
EOF

# 2. Create admin user
python scripts/create_tenant_admin.py \
  --tenant-slug acme-corp \
  --email admin@acme.com \
  --name "John Doe"

# 3. Set up initial configuration
python scripts/configure_tenant.py \
  --tenant-slug acme-corp \
  --emission-factors ecoinvent \
  --reporting-standard ghg-protocol

# 4. Send welcome email
python scripts/send_welcome_email.py --tenant-slug acme-corp

# 5. Add to monitoring
# (Auto-discovered via tenant_id label)
```

### Removing/Suspending a Tenant

```bash
# Suspend (temporary)
psql $DATABASE_URL -c "UPDATE tenants SET status = 'suspended' WHERE slug = 'tenant-slug'"

# Delete (permanent - CAUTION!)
# 1. Export data first
pg_dump $DATABASE_URL --table=suppliers --table=emissions \
  -T '*' --where="tenant_id = 'TENANT_UUID'" > tenant_backup.sql

# 2. Delete all data
psql $DATABASE_URL -c "DELETE FROM suppliers WHERE tenant_id = 'TENANT_UUID'"
psql $DATABASE_URL -c "DELETE FROM users WHERE tenant_id = 'TENANT_UUID'"
psql $DATABASE_URL -c "DELETE FROM tenants WHERE id = 'TENANT_UUID'"
```

### Certificate Renewal

**Let's Encrypt (Automated via cert-manager):**
```bash
# Verify cert-manager is running
kubectl get pods -n cert-manager

# Check certificate status
kubectl describe certificate vcci-tls -n vcci-production

# Force renewal (if needed)
kubectl delete secret vcci-tls -n vcci-production
kubectl delete certificate vcci-tls -n vcci-production
kubectl apply -f infrastructure/kubernetes/certificates.yaml
```

**Manual Certificate Update:**
```bash
# 1. Obtain new certificate
# (From CA or generate)

# 2. Create secret
kubectl create secret tls vcci-tls \
  --cert=path/to/cert.pem \
  --key=path/to/key.pem \
  -n vcci-production \
  --dry-run=client -o yaml | kubectl apply -f -

# 3. Verify
kubectl describe secret vcci-tls -n vcci-production

# 4. No restart needed (Nginx auto-reloads)
```

### Database Maintenance

**Vacuum (Weekly):**
```bash
# Automated vacuum (should run automatically)
psql $DATABASE_URL -c "SELECT schemaname, tablename, last_autovacuum FROM pg_stat_user_tables WHERE schemaname = 'public' ORDER BY last_autovacuum DESC NULLS LAST"

# Manual vacuum (if needed)
psql $DATABASE_URL -c "VACUUM ANALYZE emissions"
```

**Reindex (Monthly):**
```bash
# Reindex large tables
psql $DATABASE_URL -c "REINDEX TABLE CONCURRENTLY emissions"
psql $DATABASE_URL -c "REINDEX TABLE CONCURRENTLY suppliers"
```

**Update Statistics:**
```bash
psql $DATABASE_URL -c "ANALYZE"
```

---

## Incident Response

### Severity Definitions

| Severity | Definition | Response Time | Examples |
|----------|-----------|---------------|----------|
| **P0** | Complete outage | 15 minutes | API down, database unreachable |
| **P1** | Major degradation | 1 hour | High error rate, slow responses |
| **P2** | Minor degradation | 4 hours | Single feature broken |
| **P3** | Cosmetic issue | 24 hours | UI glitch, documentation error |

### Incident Response Checklist

**P0/P1 Incidents:**

```markdown
- [ ] 1. Acknowledge incident (< 5 minutes)
- [ ] 2. Create incident channel (#incident-YYYY-MM-DD)
- [ ] 3. Assign Incident Commander
- [ ] 4. Update status page ("Investigating")
- [ ] 5. Assemble response team
- [ ] 6. Identify root cause
- [ ] 7. Implement fix or workaround
- [ ] 8. Verify resolution
- [ ] 9. Update status page ("Resolved")
- [ ] 10. Post-incident review (within 48h)
```

**Communication Template:**

```
INCIDENT SUMMARY
Incident ID: INC-2025-11-08-001
Severity: P1
Start Time: 2025-11-08 14:30 UTC
Status: Investigating

IMPACT:
- API response times elevated (2-5 seconds vs. normal <500ms)
- Affecting ~30% of requests
- All tenants impacted

ROOT CAUSE:
- Database connection pool exhausted
- Slow query causing lock contention

MITIGATION:
- Killed slow query (PID 12345)
- Increased connection pool size
- Monitoring recovery

NEXT UPDATE: 15 minutes
```

### Rollback Procedure

```bash
# If new deployment causing issues

# 1. Identify previous version
kubectl rollout history deployment/backend-api -n vcci-production

# 2. Rollback
kubectl rollout undo deployment/backend-api -n vcci-production

# 3. Verify
kubectl rollout status deployment/backend-api -n vcci-production

# 4. Check health
curl https://api.vcci.greenlang.io/health/ready

# 5. Rollback database (if needed)
alembic downgrade -1
```

---

## Monitoring & Alerts

### Critical Alerts

**Immediate Response Required:**

1. **APIDown**
   - Trigger: Health check failing for 2 minutes
   - Response: Check pod status, restart if needed

2. **HighErrorRate**
   - Trigger: 5xx error rate > 1% for 5 minutes
   - Response: Check logs, identify root cause

3. **DatabaseDown**
   - Trigger: Database unreachable for 1 minute
   - Response: Check RDS status, failover if needed

4. **DiskSpaceCritical**
   - Trigger: Disk > 90% full
   - Response: Clean up old data, expand disk

**Monitoring Checklist:**

```bash
# Every hour (automated)
- Check API response times
- Check error rates
- Check queue depths
- Check database connection count

# Every 4 hours
- Review Grafana dashboards
- Check for alert fatigue (too many alerts)
- Verify backups completing

# Daily
- Review slow query log
- Check for security anomalies
- Verify certificate expiry dates
```

---

## Maintenance Windows

### Scheduling Maintenance

**Preferred Windows:**
- Primary: Sundays 2:00-6:00 AM UTC (Sat night US)
- Secondary: Wednesdays 2:00-6:00 AM UTC

**Notification Timeline:**
- 7 days before: Email to all tenants
- 3 days before: In-app banner
- 1 day before: Email reminder
- 1 hour before: Status page update

**Maintenance Procedure:**

```bash
# 1. Enable maintenance mode
kubectl apply -f infrastructure/kubernetes/maintenance-mode.yaml

# 2. Scale down to 1 replica
kubectl scale deployment/backend-api --replicas=1 -n vcci-production

# 3. Perform maintenance
# (Database migration, infrastructure updates, etc.)

# 4. Verify health
pytest tests/smoke/

# 5. Scale back up
kubectl scale deployment/backend-api --replicas=3 -n vcci-production

# 6. Disable maintenance mode
kubectl delete -f infrastructure/kubernetes/maintenance-mode.yaml

# 7. Update status page
curl -X PATCH statuspage.io/api/... -d "Maintenance complete"
```

---

## Common Operations Reference

### Quick Commands

```bash
# Get pod logs
kubectl logs -f deployment/backend-api -n vcci-production

# Execute command in pod
kubectl exec -it deployment/backend-api -n vcci-production -- bash

# Port forward to database
kubectl port-forward svc/postgresql 5432:5432 -n vcci-production

# Run migration
kubectl exec deployment/backend-api -n vcci-production -- alembic upgrade head

# Clear Redis cache
redis-cli -u $REDIS_URL FLUSHDB

# Restart deployment (zero downtime)
kubectl rollout restart deployment/backend-api -n vcci-production

# Check resource usage
kubectl top nodes
kubectl top pods -n vcci-production
```

### Database Queries

```sql
-- Active sessions
SELECT count(*) FROM sessions WHERE last_active > NOW() - INTERVAL '15 minutes';

-- Recent calculations
SELECT count(*), status FROM calculations WHERE created_at > NOW() - INTERVAL '1 hour' GROUP BY status;

-- Largest tenants
SELECT t.name, COUNT(s.id) AS supplier_count, SUM(pg_column_size(s.*))::bigint/1024/1024 AS size_mb
FROM tenants t LEFT JOIN suppliers s ON t.id = s.tenant_id GROUP BY t.id ORDER BY supplier_count DESC LIMIT 10;

-- Slow queries
SELECT query, calls, mean_exec_time FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;
```

---

## Contacts & Escalation

### On-Call Schedule

| Week | Primary | Secondary |
|------|---------|-----------|
| Week 1 | DevOps Engineer A | DevOps Engineer B |
| Week 2 | DevOps Engineer B | DevOps Engineer C |
| Week 3 | DevOps Engineer C | DevOps Engineer A |

### Contact Information

- **DevOps On-Call:** oncall-devops@greenlang.io / PagerDuty
- **Database On-Call:** oncall-dba@greenlang.io
- **Security On-Call:** security@greenlang.io
- **Incident Commander:** VP Engineering (+1-XXX-XXX-XXXX)

### Vendor Support

- **AWS Support:** Enterprise Support (< 15 min response)
- **Redis Labs:** Email support@redislabs.com
- **Sentry:** support@sentry.io

---

**Document Version:** 2.0.0
**Last Updated:** November 8, 2025
**Next Review:** Monthly
**Owner:** DevOps Team
