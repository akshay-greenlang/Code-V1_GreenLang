# Redis Backup Terraform Module

This Terraform module creates AWS infrastructure for Redis backup and recovery automation.

## Features

- **S3 Bucket**: Secure backup storage with lifecycle rules
- **KMS Encryption**: Server-side encryption for backups at rest
- **IAM Role**: Least-privilege access for backup jobs
- **IRSA Support**: IAM Roles for Kubernetes Service Accounts
- **Lifecycle Rules**: Automatic transition to Glacier and expiration
- **CloudWatch**: Logging and alerting for backup operations

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Redis Backup Infrastructure                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   CronJob    │    │   CronJob    │    │    Job       │       │
│  │  (RDB 6hr)   │    │  (AOF 1hr)   │    │  (Restore)   │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         └───────────────────┼───────────────────┘                │
│                             │                                    │
│                             ▼                                    │
│                    ┌────────────────┐                            │
│                    │  IAM Role      │                            │
│                    │  (IRSA)        │                            │
│                    └────────┬───────┘                            │
│                             │                                    │
│         ┌───────────────────┼───────────────────┐                │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐         │
│  │   S3 Bucket  │   │   KMS Key    │   │  CloudWatch  │         │
│  │  (Backups)   │   │ (Encryption) │   │   (Logs)     │         │
│  └──────────────┘   └──────────────┘   └──────────────┘         │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────┐       │
│  │                 Lifecycle Rules                       │       │
│  ├──────────────────────────────────────────────────────┤       │
│  │  RDB: Standard → Intelligent-Tiering → Glacier       │       │
│  │  AOF: Standard → Delete (1 day)                      │       │
│  │  Archive: Deep Archive (7 years)                     │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Usage

```hcl
module "redis_backup" {
  source = "./modules/redis-backup"

  project_name = "greenlang"
  environment  = "production"

  tags = {
    Team = "Platform"
  }
}
```

### With EKS IRSA

```hcl
module "redis_backup" {
  source = "./modules/redis-backup"

  project_name = "greenlang"
  environment  = "production"

  # EKS IRSA configuration
  eks_oidc_provider_arn = module.eks.oidc_provider_arn
  kubernetes_namespace  = "greenlang"

  # Retention settings
  rdb_retention_days = 90
  aof_retention_days = 7

  # Monitoring
  enable_cloudwatch_alarms = true
  alarm_sns_topic_arns     = [aws_sns_topic.alerts.arn]

  tags = {
    Team = "Platform"
  }
}
```

### With Custom Retention

```hcl
module "redis_backup" {
  source = "./modules/redis-backup"

  project_name = "greenlang"
  environment  = "production"

  # Extended retention for compliance
  rdb_retention_days          = 365
  rdb_glacier_transition_days = 30
  aof_retention_days          = 7
  archive_retention_days      = 2555  # 7 years

  # Enable archive tier
  enable_archive_tier = true

  tags = {
    Compliance = "SOC2"
  }
}
```

## Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|:--------:|
| project_name | Name of the project | `string` | `"greenlang"` | no |
| environment | Environment name | `string` | `"production"` | no |
| rdb_retention_days | RDB backup retention | `number` | `90` | no |
| aof_retention_days | AOF backup retention | `number` | `7` | no |
| enable_kms_encryption | Enable KMS encryption | `bool` | `true` | no |
| eks_oidc_provider_arn | EKS OIDC provider ARN | `string` | `""` | no |
| enable_cloudwatch_alarms | Enable CloudWatch alarms | `bool` | `true` | no |

See [variables.tf](variables.tf) for complete list.

## Outputs

| Name | Description |
|------|-------------|
| bucket_name | S3 bucket name |
| bucket_arn | S3 bucket ARN |
| iam_role_arn | IAM role ARN for backup jobs |
| kms_key_arn | KMS key ARN |
| backup_configuration | Configuration for Kubernetes ConfigMap |

See [outputs.tf](outputs.tf) for complete list.

## Kubernetes Integration

After deploying this module, update your Kubernetes ServiceAccount:

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: redis-backup-sa
  namespace: greenlang
  annotations:
    eks.amazonaws.com/role-arn: <iam_role_arn from outputs>
```

And update the ConfigMap with bucket details:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-backup-config
  namespace: greenlang
data:
  S3_BUCKET: <bucket_name from outputs>
  S3_REGION: <bucket_region from outputs>
  S3_PREFIX: "redis-backups"
```

## Lifecycle Rules

| Backup Type | Standard | Intelligent-Tiering | Glacier | Delete |
|-------------|----------|---------------------|---------|--------|
| RDB | 0-7 days | 7-30 days | 30-90 days | 90 days |
| AOF | 0-7 days | - | - | 7 days |
| Archive | 0-1 day | - | Deep Archive | 7 years |

## Security

- All backups encrypted with KMS (SSE-KMS)
- Public access blocked on S3 bucket
- TLS enforced for all S3 operations
- IAM role follows least-privilege principle
- CloudTrail logging enabled

## Cost Optimization

- Intelligent-Tiering for frequently accessed backups
- Glacier for older backups
- Deep Archive for compliance archives
- Automatic cleanup of old backups and multipart uploads

## Monitoring

When `enable_cloudwatch_alarms = true`, the following alarms are created:

- **BackupFailed**: Triggered when a backup job fails
- **BackupSizeAnomaly**: Triggered when backup size is unusually small

## Requirements

| Name | Version |
|------|---------|
| terraform | >= 1.0 |
| aws | >= 5.0 |

## License

Copyright 2024 GreenLang. All rights reserved.
