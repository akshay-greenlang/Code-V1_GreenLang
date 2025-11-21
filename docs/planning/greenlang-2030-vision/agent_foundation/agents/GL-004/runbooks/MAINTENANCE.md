# GL-004 - Maintenance Guide

## Daily Maintenance
- Review error logs
- Check optimization success rate (target >95%)
- Monitor emissions compliance
- Verify sensor data quality scores

## Weekly Maintenance
- Review performance trends in Grafana
- Check database size and growth rate
- Verify backup completion
- Update alert thresholds if needed

## Monthly Maintenance
- Apply security patches
- Update dependencies (pip install --upgrade)
- Calibrate O2 analyzers and CEMS
- Review capacity planning
- Performance optimization review

## Database Maintenance
\`\`\`bash
# Backup
pg_dump greenlang > backup_$(date +%Y%m%d).sql

# Vacuum and analyze
psql greenlang -c "VACUUM ANALYZE optimizations;"

# Archive old data (>90 days)
psql greenlang -c "DELETE FROM optimizations WHERE timestamp < NOW() - INTERVAL '90 days';"
\`\`\`

## Certificate Renewal
\`\`\`bash
# Check expiration
openssl x509 -in cert.pem -noout -dates

# Renew
kubectl cert-manager renew gl-004-tls -n greenlang
\`\`\`

## Disaster Recovery
\`\`\`bash
# Full backup
kubectl get all -n greenlang -l app=gl-004 -o yaml > backup/gl-004-full.yaml
pg_dump greenlang > backup/database.sql

# Restore
kubectl apply -f backup/gl-004-full.yaml
psql greenlang < backup/database.sql
\`\`\`
