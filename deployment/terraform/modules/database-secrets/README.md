# GreenLang Database Secrets Module

This Terraform module manages database credentials in AWS Secrets Manager with automatic rotation for GreenLang's PostgreSQL/TimescaleDB cluster.

## Features

- AWS Secrets Manager secrets for all database credentials
- 90-day automatic credential rotation via Lambda
- KMS encryption for all secrets
- IAM roles for EKS pod access (IRSA)
- External Secrets Operator integration
- CloudWatch monitoring and alerting
- SNS notifications for rotation events

## Usage

```hcl
module "database_secrets" {
  source = "./modules/database-secrets"

  name_prefix        = "greenlang"
  secret_name_prefix = "greenlang"

  # Database configuration
  database_host      = module.aurora.cluster_endpoint
  database_port      = 5432
  database_name      = "greenlang"
  cluster_identifier = "greenlang-postgresql"

  # User configuration
  master_username      = "postgres"
  app_username         = "greenlang_app"
  replication_username = "replicator"

  # Rotation settings
  enable_rotation = true
  rotation_days   = 90

  # pgBackRest configuration
  pgbackrest_s3_bucket = "greenlang-backups"
  pgbackrest_s3_region = "us-east-1"

  # EKS integration
  eks_cluster_oidc_issuer_url = module.eks.oidc_provider_url
  eks_namespace               = "greenlang"
  eks_service_account_name    = "greenlang-db-secrets"

  # VPC for Lambda
  lambda_subnet_ids         = module.vpc.private_subnets
  lambda_security_group_ids = [aws_security_group.lambda.id]

  tags = {
    Environment = "production"
    Project     = "GreenLang"
  }
}
```

## Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|:--------:|
| name_prefix | Prefix for naming resources | `string` | `"greenlang"` | no |
| secret_name_prefix | Prefix for secret names | `string` | `"greenlang"` | no |
| database_host | PostgreSQL host | `string` | n/a | yes |
| database_port | PostgreSQL port | `number` | `5432` | no |
| database_name | Database name | `string` | `"greenlang"` | no |
| enable_rotation | Enable automatic rotation | `bool` | `true` | no |
| rotation_days | Days between rotations | `number` | `90` | no |
| eks_cluster_oidc_issuer_url | OIDC issuer URL for IRSA | `string` | `""` | no |
| lambda_subnet_ids | Subnets for Lambda VPC | `list(string)` | `[]` | no |
| lambda_security_group_ids | Security groups for Lambda | `list(string)` | `[]` | no |

## Outputs

| Name | Description |
|------|-------------|
| master_credentials_secret_arn | ARN of master credentials secret |
| app_credentials_secret_arn | ARN of application credentials secret |
| replication_credentials_secret_arn | ARN of replication credentials secret |
| pgbouncer_credentials_secret_arn | ARN of PgBouncer credentials secret |
| pgbackrest_encryption_secret_arn | ARN of pgBackRest encryption secret |
| all_secret_arns | List of all secret ARNs |
| eks_secrets_role_arn | IAM role ARN for EKS pods |
| rotation_lambda_arn | ARN of rotation Lambda function |

## Secrets Created

| Secret | Purpose |
|--------|---------|
| `{prefix}/database/master` | PostgreSQL superuser credentials |
| `{prefix}/database/application` | Application user credentials |
| `{prefix}/database/replication` | Streaming replication user |
| `{prefix}/database/pgbouncer` | PgBouncer authentication |
| `{prefix}/database/pgbackrest` | Backup encryption keys |

## Rotation

The module deploys a Lambda function that handles credential rotation:

1. **createSecret**: Generates new password
2. **setSecret**: Updates password in PostgreSQL
3. **testSecret**: Verifies new credentials work
4. **finishSecret**: Promotes new version to AWSCURRENT

## Kubernetes Integration

Use with External Secrets Operator:

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: postgresql-app-credentials
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: greenlang-database-secrets
    kind: SecretStore
  target:
    name: postgresql-app-credentials
  data:
    - secretKey: DATABASE_URL
      remoteRef:
        key: greenlang/database/application
        property: connectionString
```

## License

Copyright GreenLang. All rights reserved.
