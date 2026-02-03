# Aurora PostgreSQL Module with TimescaleDB Support

This Terraform module creates a production-ready Amazon Aurora PostgreSQL cluster with TimescaleDB extension support.

## Features

- **PostgreSQL 15.x Compatible**: Uses Aurora PostgreSQL 15.x engine
- **Multi-AZ Deployment**: Automatic failover across availability zones
- **Configurable Read Replicas**: Scale read traffic with up to 15 replicas
- **TimescaleDB Support**: Pre-configured parameter groups for TimescaleDB extension
- **KMS Encryption**: Storage encryption with customer-managed or auto-created KMS keys
- **IAM Authentication**: Optional IAM database authentication
- **Performance Insights**: Deep performance monitoring and analysis
- **Enhanced Monitoring**: OS-level metrics at configurable intervals
- **Secrets Manager Integration**: Automatic credential management with optional rotation
- **CloudWatch Alarms**: Pre-configured alarms for critical metrics
- **S3 Export**: Export data directly to S3 buckets

## Architecture

```
                                  +------------------+
                                  |   Application    |
                                  +--------+---------+
                                           |
                                           v
+------------------------------------------+------------------------------------------+
|                                    VPC                                              |
|  +----------------+    +----------------+    +----------------+                     |
|  |  AZ-1          |    |  AZ-2          |    |  AZ-3          |                     |
|  |                |    |                |    |                |                     |
|  |  +---------+   |    |  +---------+   |    |  +---------+   |                     |
|  |  | Writer  |   |    |  | Reader  |   |    |  | Reader  |   |                     |
|  |  | Instance|   |    |  | Instance|   |    |  | Instance|   |                     |
|  |  +---------+   |    |  +---------+   |    |  +---------+   |                     |
|  +----------------+    +----------------+    +----------------+                     |
|           |                    |                    |                               |
|           +--------------------+--------------------+                               |
|                                |                                                    |
|                    +-----------+-----------+                                        |
|                    | Aurora Storage Volume |                                        |
|                    | (Encrypted with KMS)  |                                        |
|                    +-----------------------+                                        |
+-------------------------------------------------------------------------------------+
```

## Usage

### Basic Usage

```hcl
module "aurora_postgresql" {
  source = "../../modules/aurora-postgresql"

  project_name = "myapp"
  environment  = "prod"

  # Network
  vpc_id     = "vpc-12345678"
  subnet_ids = ["subnet-1", "subnet-2", "subnet-3"]

  # Database
  database_name  = "myapp_prod"
  engine_version = "15.4"

  # Instance sizing
  instance_class = "db.r6g.large"
  replica_count  = 2

  # Security
  allowed_security_groups = ["sg-app-servers"]
}
```

### With TimescaleDB Configuration

```hcl
module "aurora_postgresql" {
  source = "../../modules/aurora-postgresql"

  project_name = "timeseries-app"
  environment  = "prod"

  vpc_id     = "vpc-12345678"
  subnet_ids = ["subnet-1", "subnet-2", "subnet-3"]

  database_name  = "timeseries_prod"
  engine_version = "15.4"

  instance_class = "db.r6g.xlarge"
  replica_count  = 2

  # TimescaleDB specific settings
  timescaledb_max_background_workers = 16
  timescaledb_telemetry_level        = "off"

  # Additional parameters for time-series workloads
  cluster_parameters = [
    {
      name         = "work_mem"
      value        = "524288"  # 512 MB
      apply_method = "immediate"
    },
    {
      name         = "maintenance_work_mem"
      value        = "1048576"  # 1 GB
      apply_method = "immediate"
    }
  ]

  allowed_security_groups = ["sg-app-servers"]
}
```

### With Serverless v2

```hcl
module "aurora_postgresql" {
  source = "../../modules/aurora-postgresql"

  project_name = "serverless-app"
  environment  = "dev"

  vpc_id     = "vpc-12345678"
  subnet_ids = ["subnet-1", "subnet-2"]

  database_name  = "app_dev"
  engine_version = "15.4"

  # Serverless configuration
  instance_class          = "db.serverless"
  serverless_min_capacity = 0.5
  serverless_max_capacity = 16

  replica_count = 1

  allowed_security_groups = ["sg-app-servers"]
}
```

## TimescaleDB Setup

After the cluster is created, you need to create the TimescaleDB extension in your database:

```sql
-- Connect to your database
\c your_database

-- Create the TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Verify installation
SELECT extname, extversion FROM pg_extension WHERE extname = 'timescaledb';

-- Create a hypertable example
CREATE TABLE sensor_data (
    time TIMESTAMPTZ NOT NULL,
    sensor_id INTEGER,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION
);

SELECT create_hypertable('sensor_data', 'time');
```

## Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|:--------:|
| project_name | Name of the project | string | - | yes |
| environment | Environment name | string | - | yes |
| vpc_id | VPC ID | string | - | yes |
| subnet_ids | List of subnet IDs (minimum 2) | list(string) | - | yes |
| database_name | Name of the default database | string | "greenlang" | no |
| engine_version | PostgreSQL engine version (15.x) | string | "15.4" | no |
| instance_class | Instance class | string | "db.r6g.large" | no |
| replica_count | Number of read replicas | number | 2 | no |
| backup_retention_period | Backup retention in days | number | 35 | no |
| monitoring_interval | Enhanced monitoring interval | number | 60 | no |
| performance_insights_enabled | Enable Performance Insights | bool | true | no |
| iam_database_authentication_enabled | Enable IAM auth | bool | true | no |

See `variables.tf` for the complete list of inputs.

## Outputs

| Name | Description |
|------|-------------|
| cluster_endpoint | Writer endpoint |
| cluster_reader_endpoint | Reader endpoint |
| security_group_id | Security group ID |
| master_credentials_secret_arn | Master credentials secret ARN |
| application_credentials_secret_arn | Application credentials secret ARN |
| kms_key_arn | KMS key ARN |
| enhanced_monitoring_role_arn | Enhanced monitoring IAM role ARN |

See `outputs.tf` for the complete list of outputs.

## Retrieving Credentials

Use AWS CLI or SDK to retrieve database credentials:

```bash
# Get master credentials
aws secretsmanager get-secret-value \
    --secret-id greenlang-prod-aurora-master-credentials \
    --query 'SecretString' \
    --output text | jq

# Get application credentials
aws secretsmanager get-secret-value \
    --secret-id greenlang-prod-aurora-app-credentials \
    --query 'SecretString' \
    --output text | jq
```

## Connecting with IAM Authentication

```python
import boto3
import psycopg2

# Generate auth token
client = boto3.client('rds')
token = client.generate_db_auth_token(
    DBHostname='your-cluster-endpoint',
    Port=5432,
    DBUsername='your_username',
    Region='us-east-1'
)

# Connect using the token as password
conn = psycopg2.connect(
    host='your-cluster-endpoint',
    port=5432,
    database='your_database',
    user='your_username',
    password=token,
    sslmode='require'
)
```

## Security Considerations

1. **Network Isolation**: Deploy in private subnets without public access
2. **Encryption**: All data encrypted at rest with KMS
3. **SSL/TLS**: Force SSL connections via parameter group
4. **IAM Authentication**: Use IAM roles instead of passwords where possible
5. **Secrets Rotation**: Enable automatic credential rotation
6. **Security Groups**: Restrict access to specific application security groups

## Monitoring

The module creates:
- CloudWatch alarms for CPU, memory, connections, replication lag, storage, and IOPS
- CloudWatch dashboard for visual monitoring
- Enhanced monitoring metrics
- Performance Insights for query analysis

## Backup and Recovery

- Automated backups with configurable retention (up to 35 days)
- Point-in-time recovery within the retention window
- Final snapshot on deletion (configurable)
- Cross-region snapshot copy (configure separately)

## Maintenance

- Auto minor version upgrades enabled by default
- Configurable maintenance window
- Rolling updates for reader instances

## Cost Optimization

1. Use Serverless v2 for variable workloads
2. Right-size instances based on Performance Insights data
3. Use smaller reader instances if appropriate
4. Consider Reserved Instances for production workloads

## Troubleshooting

### High CPU Utilization
- Check Performance Insights for slow queries
- Review `pg_stat_statements` for query patterns
- Consider read replicas for read-heavy workloads

### Connection Issues
- Verify security group rules
- Check max_connections parameter
- Monitor connection count alarms

### Replication Lag
- Check network throughput
- Review write patterns
- Consider larger reader instances

## License

Apache 2.0
