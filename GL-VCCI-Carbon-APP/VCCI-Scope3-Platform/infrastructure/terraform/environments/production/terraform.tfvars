# Production Environment Variables

project_name = "vcci-scope3"
environment  = "production"
aws_region   = "us-west-2"
aws_region_secondary = "eu-central-1"
availability_zones = ["us-west-2a", "us-west-2b", "us-west-2c"]

# VPC
vpc_cidr = "10.0.0.0/16"
enable_nat_gateway = true
single_nat_gateway = false
enable_vpc_endpoints = true

# EKS
eks_cluster_version = "1.27"
compute_node_group_instance_types = ["t3.xlarge"]
compute_node_group_desired_size = 12
compute_node_group_min_size = 3
compute_node_group_max_size = 20
memory_node_group_instance_types = ["r6g.2xlarge"]
memory_node_group_desired_size = 5
memory_node_group_min_size = 2
memory_node_group_max_size = 10
gpu_node_group_instance_types = ["g4dn.xlarge"]
gpu_node_group_desired_size = 2
gpu_node_group_min_size = 1
gpu_node_group_max_size = 5

# RDS
rds_engine_version = "15.3"
rds_instance_class = "db.r6g.2xlarge"
rds_allocated_storage = 500
rds_max_allocated_storage = 1000
rds_multi_az = true
rds_read_replica_count = 2
rds_backup_retention_period = 7
rds_deletion_protection = true

# ElastiCache
elasticache_engine_version = "7.0"
elasticache_node_type = "cache.r6g.large"
elasticache_num_node_groups = 3
elasticache_replicas_per_node_group = 2

# S3
s3_enable_replication = true

# Monitoring
enable_cloudwatch_logs = true
enable_cloudwatch_alarms = true

# Backup
enable_aws_backup = true
backup_retention_days = 30
