# GL-006 HeatRecoveryMaximizer Maintenance Guide

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | GL-006 |
| Codename | HEATRECLAIM |
| Version | 1.0.0 |
| Last Updated | 2024-11-26 |

---

## 1. Overview

This guide provides procedures for routine maintenance, upgrades, and operational tasks for the GL-006 HeatRecoveryMaximizer agent. Following these procedures ensures reliable operation and minimizes downtime.

---

## 2. Maintenance Schedule

### 2.1 Regular Maintenance Windows

| Task | Frequency | Duration | Window |
|------|-----------|----------|--------|
| Security patches | Weekly | 30 min | Sunday 02:00-02:30 UTC |
| Minor updates | Bi-weekly | 1 hour | Sunday 02:00-03:00 UTC |
| Major updates | Monthly | 2 hours | First Sunday 02:00-04:00 UTC |
| Database maintenance | Weekly | 30 min | Sunday 03:00-03:30 UTC |
| Log rotation | Daily | Automated | 00:00 UTC |
| Cache cleanup | Daily | Automated | 04:00 UTC |

### 2.2 Maintenance Calendar

```
Week 1: Security patches + Database maintenance
Week 2: Minor updates (if available)
Week 3: Security patches + Database maintenance
Week 4: Major updates (if scheduled) + Full system review
```

---

## 3. Pre-Maintenance Checklist

Before any maintenance:

- [ ] Notify stakeholders 24 hours in advance
- [ ] Update status page with maintenance window
- [ ] Verify backup systems are operational
- [ ] Ensure rollback plan is ready
- [ ] Confirm on-call engineer is available
- [ ] Check cluster health
- [ ] Document current state

### 3.1 Pre-Maintenance Commands

```bash
# Document current state
echo "=== PRE-MAINTENANCE STATE ===" > maintenance-$(date +%Y%m%d).log
kubectl get pods -n greenlang -l app=gl-006-heatreclaim -o wide >> maintenance-$(date +%Y%m%d).log
kubectl get deployment gl-006-heatreclaim -n greenlang -o yaml >> maintenance-$(date +%Y%m%d).log
kubectl get configmap gl-006-heatreclaim-config -n greenlang -o yaml >> maintenance-$(date +%Y%m%d).log

# Verify health
kubectl exec -it $(kubectl get pods -n greenlang -l app=gl-006-heatreclaim -o jsonpath='{.items[0].metadata.name}') -n greenlang -- curl -s localhost:8000/health

# Check error rate
curl -s localhost:9090/metrics | grep gl006_errors_total
```

---

## 4. Application Updates

### 4.1 Minor Version Updates

```bash
# 1. Update image tag in kustomization
cd deployment/kustomize/overlays/production
kustomize edit set image gcr.io/greenlang/gl-006-heatreclaim:v1.0.2

# 2. Preview changes
kubectl diff -k .

# 3. Apply update
kubectl apply -k .

# 4. Monitor rollout
kubectl rollout status deployment gl-006-heatreclaim -n greenlang

# 5. Verify health
kubectl exec -it $(kubectl get pods -n greenlang -l app=gl-006-heatreclaim -o jsonpath='{.items[0].metadata.name}') -n greenlang -- curl -s localhost:8000/health
```

### 4.2 Major Version Updates

```bash
# 1. Create backup
kubectl get deployment gl-006-heatreclaim -n greenlang -o yaml > deployment-backup.yaml
kubectl get configmap gl-006-heatreclaim-config -n greenlang -o yaml > configmap-backup.yaml

# 2. Run database migrations (if needed)
kubectl exec -it $(kubectl get pods -n greenlang -l app=gl-006-heatreclaim -o jsonpath='{.items[0].metadata.name}') -n greenlang -- alembic upgrade head

# 3. Update image
kustomize edit set image gcr.io/greenlang/gl-006-heatreclaim:v2.0.0
kubectl apply -k .

# 4. Monitor rollout
kubectl rollout status deployment gl-006-heatreclaim -n greenlang -w

# 5. Run smoke tests
./scripts/smoke-test.sh production

# 6. Verify all functionality
kubectl logs -n greenlang -l app=gl-006-heatreclaim --since=5m | grep -i error
```

---

## 5. Configuration Updates

### 5.1 ConfigMap Updates

```bash
# 1. Edit ConfigMap
kubectl edit configmap gl-006-heatreclaim-config -n greenlang

# 2. Or apply from file
kubectl apply -f configmap-updated.yaml -n greenlang

# 3. Restart to pick up changes
kubectl rollout restart deployment gl-006-heatreclaim -n greenlang

# 4. Verify restart
kubectl rollout status deployment gl-006-heatreclaim -n greenlang
```

### 5.2 Secret Updates

```bash
# 1. Update secret (base64 encoded)
kubectl edit secret gl-006-heatreclaim-secrets -n greenlang

# 2. Or create from literal
kubectl create secret generic gl-006-heatreclaim-secrets \
  --from-literal=database-url='postgresql://user:newpass@host/db' \
  --dry-run=client -o yaml | kubectl apply -f -

# 3. Restart to pick up changes
kubectl rollout restart deployment gl-006-heatreclaim -n greenlang
```

---

## 6. Database Maintenance

### 6.1 PostgreSQL Vacuum

```bash
# Run vacuum analyze
kubectl exec -it postgresql-0 -n greenlang -- psql -U greenlang -d gl006_heatreclaim -c "VACUUM ANALYZE;"

# Check table sizes
kubectl exec -it postgresql-0 -n greenlang -- psql -U greenlang -d gl006_heatreclaim -c "
SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC
LIMIT 10;"
```

### 6.2 Database Backup

```bash
# Create backup
kubectl exec -it postgresql-0 -n greenlang -- pg_dump -U greenlang gl006_heatreclaim | gzip > backup-$(date +%Y%m%d).sql.gz

# Verify backup
gunzip -c backup-$(date +%Y%m%d).sql.gz | head -20
```

### 6.3 Index Maintenance

```bash
# Reindex database
kubectl exec -it postgresql-0 -n greenlang -- psql -U greenlang -d gl006_heatreclaim -c "REINDEX DATABASE gl006_heatreclaim;"

# Check index usage
kubectl exec -it postgresql-0 -n greenlang -- psql -U greenlang -d gl006_heatreclaim -c "
SELECT indexrelname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC
LIMIT 10;"
```

---

## 7. Cache Maintenance

### 7.1 Redis Maintenance

```bash
# Check memory usage
kubectl exec -it redis-0 -n greenlang -- redis-cli INFO memory

# Check key count
kubectl exec -it redis-0 -n greenlang -- redis-cli DBSIZE

# Clear expired keys
kubectl exec -it redis-0 -n greenlang -- redis-cli BGSAVE

# Memory optimization
kubectl exec -it redis-0 -n greenlang -- redis-cli MEMORY PURGE
```

### 7.2 Cache Invalidation

```bash
# Clear specific pattern
kubectl exec -it redis-0 -n greenlang -- redis-cli KEYS "gl006:cache:*" | xargs redis-cli DEL

# Clear all cache (use with caution)
kubectl exec -it redis-0 -n greenlang -- redis-cli FLUSHDB
```

---

## 8. Log Management

### 8.1 Log Rotation

Logs are automatically rotated, but manual cleanup can be done:

```bash
# Check log volume usage
kubectl exec -it $(kubectl get pods -n greenlang -l app=gl-006-heatreclaim -o jsonpath='{.items[0].metadata.name}') -n greenlang -- du -sh /var/log/greenlang

# Manual log cleanup (if needed)
kubectl exec -it $(kubectl get pods -n greenlang -l app=gl-006-heatreclaim -o jsonpath='{.items[0].metadata.name}') -n greenlang -- find /var/log/greenlang -name "*.log" -mtime +7 -delete
```

### 8.2 Log Export

```bash
# Export logs for analysis
kubectl logs -n greenlang -l app=gl-006-heatreclaim --since=24h > logs-$(date +%Y%m%d).log

# Export structured logs
kubectl logs -n greenlang -l app=gl-006-heatreclaim --since=24h | jq '.' > logs-$(date +%Y%m%d).json
```

---

## 9. Certificate Management

### 9.1 Check Certificate Expiry

```bash
# Check TLS certificate expiry
kubectl get secret gl-006-heatreclaim-tls -n greenlang -o jsonpath='{.data.tls\.crt}' | base64 -d | openssl x509 -noout -enddate

# Check all certificates
kubectl get certificates -n greenlang
```

### 9.2 Certificate Renewal

If using cert-manager:

```bash
# Force renewal
kubectl delete certificate gl-006-heatreclaim-tls -n greenlang
kubectl apply -f certificate.yaml

# Verify new certificate
kubectl get certificate gl-006-heatreclaim-tls -n greenlang -w
```

---

## 10. Security Maintenance

### 10.1 Security Scanning

```bash
# Scan container image
trivy image gcr.io/greenlang/gl-006-heatreclaim:v1.0.0

# Scan Kubernetes manifests
kubectl get deployment gl-006-heatreclaim -n greenlang -o yaml | trivy config -
```

### 10.2 Secret Rotation

```bash
# 1. Generate new credentials
NEW_DB_PASSWORD=$(openssl rand -base64 32)

# 2. Update database password
kubectl exec -it postgresql-0 -n greenlang -- psql -U postgres -c "ALTER USER greenlang PASSWORD '${NEW_DB_PASSWORD}';"

# 3. Update secret
kubectl create secret generic gl-006-heatreclaim-secrets \
  --from-literal=database-url="postgresql://greenlang:${NEW_DB_PASSWORD}@postgresql:5432/gl006_heatreclaim" \
  --dry-run=client -o yaml | kubectl apply -f -

# 4. Restart application
kubectl rollout restart deployment gl-006-heatreclaim -n greenlang
```

### 10.3 Access Audit

```bash
# Review RBAC permissions
kubectl get rolebinding -n greenlang
kubectl describe rolebinding gl-006-heatreclaim-rolebinding -n greenlang

# Check service account
kubectl get serviceaccount gl-006-heatreclaim-sa -n greenlang -o yaml
```

---

## 11. Performance Maintenance

### 11.1 Resource Optimization

```bash
# Check resource utilization
kubectl top pods -n greenlang -l app=gl-006-heatreclaim

# Review and adjust resources if needed
kubectl edit deployment gl-006-heatreclaim -n greenlang
```

### 11.2 HPA Tuning

```bash
# Review HPA metrics
kubectl get hpa gl-006-heatreclaim-hpa -n greenlang -o yaml

# Adjust targets if needed
kubectl patch hpa gl-006-heatreclaim-hpa -n greenlang -p '{"spec":{"metrics":[{"type":"Resource","resource":{"name":"cpu","target":{"type":"Utilization","averageUtilization":60}}}]}}'
```

---

## 12. Disaster Recovery

### 12.1 Backup Verification

```bash
# Verify database backup
pg_restore --list backup-$(date +%Y%m%d).sql.gz

# Test restore to temporary database
kubectl exec -it postgresql-0 -n greenlang -- createdb -U postgres gl006_test
gunzip -c backup-$(date +%Y%m%d).sql.gz | kubectl exec -i postgresql-0 -n greenlang -- psql -U greenlang gl006_test
kubectl exec -it postgresql-0 -n greenlang -- dropdb -U postgres gl006_test
```

### 12.2 Recovery Drill

Monthly recovery drill checklist:

- [ ] Restore database from backup
- [ ] Verify data integrity
- [ ] Test application connectivity
- [ ] Validate core functionality
- [ ] Document recovery time
- [ ] Update recovery procedures if needed

---

## 13. Post-Maintenance Verification

After any maintenance:

```bash
# 1. Check pod health
kubectl get pods -n greenlang -l app=gl-006-heatreclaim -o wide

# 2. Verify endpoints
kubectl get endpoints gl-006-heatreclaim -n greenlang

# 3. Test health endpoint
curl http://gl-006.api.greenlang.io/health

# 4. Check error rate
curl -s localhost:9090/metrics | grep gl006_errors_total

# 5. Run smoke tests
./scripts/smoke-test.sh production

# 6. Update maintenance log
echo "Maintenance completed at $(date)" >> maintenance-$(date +%Y%m%d).log
```

---

## 14. Maintenance Communication Templates

### 14.1 Scheduled Maintenance Notice

```
SCHEDULED MAINTENANCE - GL-006 HeatRecoveryMaximizer

Date: [DATE]
Time: [START TIME] - [END TIME] UTC
Duration: [DURATION]

Description:
[Description of maintenance activities]

Impact:
- [Expected impact on users]
- [Affected functionality]

Contact:
For questions, contact platform-team@greenlang.io
```

### 14.2 Maintenance Complete Notice

```
MAINTENANCE COMPLETE - GL-006 HeatRecoveryMaximizer

Completed: [TIMESTAMP]
Duration: [ACTUAL DURATION]

Summary:
- [Completed task 1]
- [Completed task 2]

Status: All systems operational

Contact:
For issues, contact platform-oncall@greenlang.io
```

---

## 15. Related Documents

- [INCIDENT_RESPONSE.md](./INCIDENT_RESPONSE.md)
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
- [ROLLBACK_PROCEDURE.md](./ROLLBACK_PROCEDURE.md)
- [SCALING_GUIDE.md](./SCALING_GUIDE.md)

---

*This guide is maintained by the Platform Team. For updates, contact platform-team@greenlang.io*
