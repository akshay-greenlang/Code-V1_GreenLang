# Development Environment Variables

project_name = "vcci-scope3"
environment  = "dev"
aws_region   = "us-west-2"
aws_region_secondary = "eu-central-1"
availability_zones = ["us-west-2a"]

# VPC - Single AZ for cost savings
vpc_cidr = "10.1.0.0/16"
enable_nat_gateway = true
single_nat_gateway = true
enable_vpc_endpoints = false

# EKS - Minimal configuration
eks_cluster_version = "1.27"
compute_node_group_instance_types = ["t3.medium"]
compute_node_group_desired_size = 2
compute_node_group_min_size = 1
compute_node_group_max_size = 5
memory_node_group_instance_types = ["t3.large"]
memory_node_group_desired_size = 1
memory_node_group_min_size = 1
memory_node_group_max_size = 3
gpu_node_group_instance_types = ["g4dn.xlarge"]
gpu_node_group_desired_size = 0
gpu_node_group_min_size = 0
gpu_node_group_max_size = 1

# RDS - Single AZ, smaller instance
rds_engine_version = "15.3"
rds_instance_class = "db.t3.large"
rds_allocated_storage = 100
rds_max_allocated_storage = 200
rds_multi_az = false
rds_read_replica_count = 0
rds_backup_retention_period = 1
rds_deletion_protection = false

# ElastiCache - Minimal configuration
elasticache_engine_version = "7.0"
elasticache_node_type = "cache.t3.medium"
elasticache_num_node_groups = 1
elasticache_replicas_per_node_group = 1

# S3 - No replication
s3_enable_replication = false

# Monitoring - Basic
enable_cloudwatch_logs = true
enable_cloudwatch_alarms = false

# Backup - Minimal retention
enable_aws_backup = false
backup_retention_days = 7
