# GreenLang KMS Module (SEC-003)

Centralized AWS Key Management Service (KMS) module for GreenLang encryption at rest.

## Overview

This module creates and manages Customer Master Keys (CMKs) for all GreenLang services, implementing SEC-003 encryption at rest requirements. It provides dedicated keys for each service category with least-privilege access policies.

## Key Hierarchy

```
greenlang-{env}-master-cmk         (Master key, CloudWatch audit logs)
  |
  +-- greenlang-{env}-database-cmk     (Aurora PostgreSQL, RDS)
  +-- greenlang-{env}-storage-cmk      (S3, EFS)
  +-- greenlang-{env}-cache-cmk        (ElastiCache Redis)
  +-- greenlang-{env}-secrets-cmk      (Secrets Manager, Parameter Store)
  +-- greenlang-{env}-application-cmk  (Application DEKs - envelope encryption)
  +-- greenlang-{env}-eks-cmk          (EKS Kubernetes secrets)
  +-- greenlang-{env}-backup-cmk       (pgBackRest, Redis RDB, AWS Backup)
```

## Features

- **Dedicated Keys per Service**: Separate CMKs for database, storage, cache, secrets, application, EKS, and backup
- **Automatic Key Rotation**: AWS-managed annual rotation (enabled by default)
- **Multi-Region Support**: Optional multi-region keys for disaster recovery
- **Least-Privilege Policies**: Service-specific IAM policies with minimal required permissions
- **Audit Logging**: CloudWatch log group for KMS operation auditing
- **Deletion Protection**: Configurable waiting period (7-30 days) before key deletion

## Usage

### Basic Usage (All Keys)

```hcl
module "kms" {
  source = "../../modules/kms"

  project_name = "greenlang"
  environment  = "prod"

  # Enable all keys for production
  create_database_key    = true
  create_storage_key     = true
  create_cache_key       = true
  create_secrets_key     = true
  create_application_key = true
  create_eks_key         = true
  create_backup_key      = true

  # Security settings
  enable_key_rotation    = true
  deletion_window_days   = 30
  enable_multi_region    = true  # For DR

  # Access control
  key_administrators = [
    "arn:aws:iam::123456789012:role/SecurityAdmin"
  ]

  tags = {
    Team        = "platform"
    CostCenter  = "infrastructure"
  }
}
```

### With Service Role Access

```hcl
module "kms" {
  source = "../../modules/kms"

  project_name = "greenlang"
  environment  = "prod"

  # Key administrators (can manage key policies)
  key_administrators = [
    "arn:aws:iam::123456789012:role/SecurityAdmin",
    "arn:aws:iam::123456789012:role/PlatformAdmin"
  ]

  # Application roles that need to use envelope encryption
  application_service_roles = [
    "arn:aws:iam::123456789012:role/greenlang-api-role",
    "arn:aws:iam::123456789012:role/greenlang-worker-role"
  ]

  # Database roles (Aurora/RDS access)
  database_service_roles = [
    "arn:aws:iam::123456789012:role/greenlang-api-role"
  ]

  # Storage roles (S3/EFS access)
  storage_service_roles = [
    "arn:aws:iam::123456789012:role/greenlang-data-pipeline-role"
  ]

  # Backup roles (pgBackRest, Redis backup)
  backup_service_roles = [
    "arn:aws:iam::123456789012:role/greenlang-backup-role",
    "arn:aws:iam::123456789012:role/greenlang-restore-role"
  ]

  # EKS roles
  eks_service_roles = [
    "arn:aws:iam::123456789012:role/greenlang-eks-cluster-role",
    "arn:aws:iam::123456789012:role/greenlang-eks-node-role"
  ]

  tags = {
    Team = "platform"
  }
}
```

### Minimal Usage (Specific Keys Only)

```hcl
module "kms" {
  source = "../../modules/kms"

  project_name = "greenlang"
  environment  = "dev"

  # Only create keys needed for dev
  create_database_key    = true
  create_storage_key     = true
  create_cache_key       = false  # Use AWS-managed key in dev
  create_secrets_key     = true
  create_application_key = false  # No envelope encryption in dev
  create_eks_key         = false  # Minikube in dev
  create_backup_key      = false  # No backups in dev

  # Shorter deletion window for dev
  deletion_window_days = 7

  # No multi-region for dev
  enable_multi_region = false

  tags = {
    Team = "platform"
  }
}
```

## Integration with Other Modules

### Aurora PostgreSQL

```hcl
module "aurora" {
  source = "../aurora-postgresql"

  # Use centralized database key
  create_kms_key = false  # Don't create module-specific key
  kms_key_arn    = module.kms.database_key_arn
}
```

### S3 Buckets

```hcl
module "s3" {
  source = "../s3"

  # Use centralized storage key
  kms_key_arn = module.kms.storage_key_arn
}
```

### ElastiCache Redis

```hcl
module "elasticache" {
  source = "../elasticache"

  # Use centralized cache key
  at_rest_encryption_enabled = true
  kms_key_arn               = module.kms.cache_key_arn
}
```

### EKS Cluster

```hcl
module "eks" {
  source = "../eks"

  # Use centralized EKS key for secrets encryption
  cluster_encryption_config = {
    provider_key_arn = module.kms.eks_key_arn
    resources        = ["secrets"]
  }
}
```

### pgBackRest Backups

```hcl
module "pgbackrest" {
  source = "../pgbackrest"

  # Use centralized backup key
  kms_key_arn = module.kms.backup_key_arn
}
```

## Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|:--------:|
| project_name | Project name for resource naming | `string` | `"greenlang"` | no |
| environment | Environment (dev, staging, prod) | `string` | - | yes |
| create_database_key | Create CMK for Aurora/RDS | `bool` | `true` | no |
| create_storage_key | Create CMK for S3/EFS | `bool` | `true` | no |
| create_cache_key | Create CMK for ElastiCache | `bool` | `true` | no |
| create_secrets_key | Create CMK for Secrets Manager | `bool` | `true` | no |
| create_application_key | Create CMK for envelope encryption | `bool` | `true` | no |
| create_eks_key | Create CMK for EKS secrets | `bool` | `true` | no |
| create_backup_key | Create CMK for backups | `bool` | `true` | no |
| enable_key_rotation | Enable annual auto-rotation | `bool` | `true` | no |
| deletion_window_days | Days before key deletion (7-30) | `number` | `30` | no |
| enable_multi_region | Create multi-region keys | `bool` | `false` | no |
| key_administrators | IAM ARNs for key administration | `list(string)` | `[]` | no |
| *_service_roles | IAM roles allowed to use specific keys | `list(string)` | `[]` | no |
| enable_cloudwatch_logging | Enable CloudWatch audit logging | `bool` | `true` | no |
| log_retention_days | CloudWatch log retention | `number` | `365` | no |
| tags | Additional tags | `map(string)` | `{}` | no |

## Outputs

| Name | Description |
|------|-------------|
| master_key_arn | ARN of the master CMK |
| master_key_id | ID of the master CMK |
| master_key_alias | Alias of the master CMK |
| database_key_arn | ARN of the database CMK |
| storage_key_arn | ARN of the storage CMK |
| cache_key_arn | ARN of the cache CMK |
| secrets_key_arn | ARN of the secrets CMK |
| application_key_arn | ARN of the application CMK |
| eks_key_arn | ARN of the EKS CMK |
| backup_key_arn | ARN of the backup CMK |
| all_keys | Map of all key ARNs by type |
| all_key_ids | Map of all key IDs by type |
| all_key_aliases | Map of all key aliases by type |
| key_summary | Summary of created keys and configuration |

## Security Considerations

### Key Rotation
- Automatic annual rotation is enabled by default
- AWS manages the rotation process transparently
- Previous key versions are retained for decryption

### Deletion Protection
- 30-day waiting period by default (configurable 7-30 days)
- Key deletion can be cancelled during waiting period
- All encrypted data becomes inaccessible after deletion

### Least Privilege
- Separate keys per service category
- Service-specific IAM policies
- Dynamic role assignment through variables

### Multi-Region Keys
- Optional for disaster recovery
- Required for cross-region replication
- Same key ID across regions

### Audit Logging
- CloudWatch log group for KMS operations
- Encrypted with the master CMK
- 365-day retention by default

## Compliance

This module supports the following compliance requirements:

| Framework | Controls |
|-----------|----------|
| SOC 2 | CC6.6 (Encryption), CC6.7 (Key Management) |
| ISO 27001 | A.10.1 (Cryptographic Controls) |
| GDPR | Article 32 (Security of Processing) |
| PCI DSS | Requirement 3 (Protect Stored Data) |
| HIPAA | 164.312(a)(2)(iv) (Encryption) |

## Cost Considerations

- Each CMK costs $1/month
- API requests: $0.03 per 10,000 requests
- Multi-region keys have additional replication costs
- Estimate: ~$8-10/month for full key set per environment

## Troubleshooting

### Key Not Found
```
Error: KMS key not found
```
Ensure the key was created and check the `create_*_key` variables.

### Access Denied
```
Error: User is not authorized to perform kms:Decrypt
```
Add the IAM role to the appropriate `*_service_roles` variable.

### Key Pending Deletion
```
Error: Key is pending deletion
```
Use `aws kms cancel-key-deletion` to restore the key, or wait for a new key.

## Related Documentation

- [SEC-003 PRD](../../docs/PRD-SEC-003-Encryption-at-Rest.md)
- [AWS KMS Best Practices](https://docs.aws.amazon.com/kms/latest/developerguide/best-practices.html)
- [Envelope Encryption](https://docs.aws.amazon.com/kms/latest/developerguide/concepts.html#enveloping)
