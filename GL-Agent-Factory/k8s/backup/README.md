# GreenLang PostgreSQL Backup and Disaster Recovery

This directory contains Kubernetes manifests for automated PostgreSQL backup and disaster recovery for the GreenLang platform.

## Overview

The backup solution provides:

- **Daily automated backups** at 2:00 AM UTC via CronJob
- **S3 storage** with configurable retention (default: 30 days)
- **Gzip compression** for efficient storage
- **SHA256 checksums** for integrity verification
- **Point-in-time recovery** from any backup
- **Pre-restore safety backups** to prevent data loss
- **Slack notifications** for backup status alerts

## Architecture

```
+------------------+     +-------------------+     +-------------+
|  Kubernetes      |     |   PostgreSQL      |     |   AWS S3    |
|  CronJob         |---->|   (pg_dump)       |---->|   Bucket    |
|  (2 AM daily)    |     |                   |     |             |
+------------------+     +-------------------+     +-------------+
                                                         |
                                                         v
                                                  +-------------+
                                                  |  Retention  |
                                                  |  (30 days)  |
                                                  +-------------+
```

## Files

| File | Description |
|------|-------------|
| `postgres-backup-cronjob.yaml` | Main CronJob, RBAC, ConfigMap, NetworkPolicy |
| `postgres-backup-secret.yaml` | S3 credentials template (DO NOT commit actual secrets) |
| `postgres-restore-job.yaml` | Point-in-time recovery job |
| `kustomization.yaml` | Kustomize configuration for deployment |

## Quick Start

### 1. Configure S3 Credentials

**Option A: Using IRSA (Recommended for EKS)**

```yaml
# Annotate the service account in postgres-backup-cronjob.yaml
annotations:
  eks.amazonaws.com/role-arn: "arn:aws:iam::ACCOUNT_ID:role/GreenLangBackupRole"
```

**Option B: Using Static Credentials**

```bash
# Create the secret (DO NOT commit this file)
kubectl create secret generic postgres-backup-s3-credentials \
  --from-literal=AWS_ACCESS_KEY_ID='your-access-key' \
  --from-literal=AWS_SECRET_ACCESS_KEY='your-secret-key' \
  -n greenlang-production
```

### 2. Create S3 Bucket

```bash
aws s3 mb s3://greenlang-backups-production --region us-east-1

# Enable versioning (recommended)
aws s3api put-bucket-versioning \
  --bucket greenlang-backups-production \
  --versioning-configuration Status=Enabled

# Enable encryption
aws s3api put-bucket-encryption \
  --bucket greenlang-backups-production \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'
```

### 3. Deploy

```bash
# Deploy using kustomize
kubectl apply -k k8s/backup/

# Or apply individual files
kubectl apply -f k8s/backup/postgres-backup-cronjob.yaml
```

### 4. Trigger Manual Backup

```bash
# Using the helper script
./scripts/backup-restore.sh backup

# Or using kubectl
kubectl create job postgres-backup-manual-$(date +%s) \
  --from=cronjob/postgres-backup \
  -n greenlang-production
```

## Disaster Recovery Procedures

### Restore from Latest Backup

```bash
# Using the helper script
./scripts/backup-restore.sh restore --latest --confirm

# Or manually
kubectl patch configmap postgres-restore-config \
  -n greenlang-production \
  --type merge \
  -p '{"data":{"RESTORE_MODE":"latest","CONFIRM_RESTORE":"true"}}'

kubectl apply -f k8s/backup/postgres-restore-job.yaml
kubectl logs -f job/postgres-restore -n greenlang-production
```

### Restore to Specific Point in Time

```bash
# List available backups
./scripts/backup-restore.sh list

# Restore to specific timestamp
./scripts/backup-restore.sh restore --timestamp 20241209_020000 --confirm
```

### Restore from Specific File

```bash
./scripts/backup-restore.sh restore --file greenlang_greenlang_20241209_020000.sql.gz --confirm
```

## Monitoring and Alerting

### Check Backup Status

```bash
# View CronJob status
kubectl get cronjob postgres-backup -n greenlang-production

# View recent jobs
kubectl get jobs -n greenlang-production -l app.kubernetes.io/component=backup

# View backup logs
kubectl logs -l app.kubernetes.io/component=backup -n greenlang-production --tail=100
```

### Prometheus Metrics

The backup job exposes the following metrics (when configured):

- `greenlang_backup_last_success_timestamp` - Timestamp of last successful backup
- `greenlang_backup_last_duration_seconds` - Duration of last backup
- `greenlang_backup_last_size_bytes` - Size of last backup

### Slack Notifications

Configure Slack notifications in the ConfigMap:

```yaml
data:
  ENABLE_NOTIFICATIONS: "true"
  SLACK_CHANNEL: "#greenlang-alerts"
```

Add webhook URL to secrets:

```bash
kubectl patch secret postgres-backup-s3-credentials \
  -n greenlang-production \
  --type merge \
  -p '{"stringData":{"SLACK_WEBHOOK_URL":"https://hooks.slack.com/..."}}'
```

## Configuration Reference

### ConfigMap: postgres-backup-config

| Key | Default | Description |
|-----|---------|-------------|
| `BACKUP_RETENTION_DAYS` | `30` | Days to keep backups |
| `BACKUP_COMPRESSION` | `gzip` | Compression algorithm |
| `S3_BUCKET` | `greenlang-backups-production` | S3 bucket name |
| `S3_REGION` | `us-east-1` | AWS region |
| `S3_PREFIX` | `postgres/daily` | S3 key prefix |
| `PGHOST` | `postgresql-service` | PostgreSQL host |
| `PGPORT` | `5432` | PostgreSQL port |
| `PGDATABASE` | `greenlang` | Database name |
| `ENABLE_NOTIFICATIONS` | `false` | Enable Slack alerts |
| `SLACK_CHANNEL` | `#greenlang-alerts` | Slack channel |

### CronJob Schedule

The default schedule is `0 2 * * *` (daily at 2:00 AM UTC).

Common schedule patterns:

| Schedule | Description |
|----------|-------------|
| `0 2 * * *` | Daily at 2 AM |
| `0 */6 * * *` | Every 6 hours |
| `0 2 * * 0` | Weekly on Sunday |
| `0 2 1 * *` | Monthly on the 1st |

## Security Considerations

1. **Secrets Management**
   - Use IRSA or External Secrets Operator in production
   - Never commit actual credentials to git
   - Rotate S3 credentials regularly

2. **Network Security**
   - NetworkPolicy restricts egress to PostgreSQL and S3 only
   - Backup pods run as non-root user
   - Read-only root filesystem where possible

3. **S3 Bucket Security**
   - Enable bucket versioning
   - Enable server-side encryption
   - Use bucket policies to restrict access
   - Enable access logging

4. **Backup Encryption**
   - Consider enabling client-side encryption for sensitive data
   - Use KMS for S3 server-side encryption

## Troubleshooting

### Backup Job Fails

```bash
# Check job status
kubectl describe job -l app.kubernetes.io/component=backup -n greenlang-production

# Check pod logs
kubectl logs -l app.kubernetes.io/component=backup -n greenlang-production --all-containers

# Check events
kubectl get events -n greenlang-production --sort-by='.lastTimestamp' | grep backup
```

### Cannot Connect to PostgreSQL

```bash
# Verify PostgreSQL is running
kubectl get pods -l app=postgresql -n greenlang-production

# Test connectivity from backup pod
kubectl run -it --rm debug --image=postgres:15-alpine --restart=Never -- \
  pg_isready -h postgresql-service -p 5432
```

### S3 Upload Fails

```bash
# Verify AWS credentials
kubectl exec -it <backup-pod> -- env | grep AWS

# Test S3 access
kubectl run -it --rm aws-debug --image=amazon/aws-cli --restart=Never -- \
  s3 ls s3://greenlang-backups-production/
```

### Restore Job Hangs

```bash
# Check if CONFIRM_RESTORE is set
kubectl get configmap postgres-restore-config -n greenlang-production -o yaml | grep CONFIRM

# Check active connections that might block restore
kubectl exec -it <postgres-pod> -- psql -c "SELECT * FROM pg_stat_activity WHERE datname='greenlang';"
```

## IAM Policy for S3 Access

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::greenlang-backups-production",
        "arn:aws:s3:::greenlang-backups-production/*"
      ]
    }
  ]
}
```

## Related Documentation

- [Helm Chart Configuration](../../helm/greenlang/README.md)
- [Helper Script Usage](../../scripts/backup-restore.sh)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/15/backup.html)
- [AWS S3 Best Practices](https://docs.aws.amazon.com/AmazonS3/latest/userguide/security-best-practices.html)
