# Staging Environment Variables

project_name = "vcci-scope3"
environment  = "staging"
aws_region   = "us-west-2"
aws_region_secondary = "eu-central-1"
availability_zones = ["us-west-2a", "us-west-2b"]

# VPC - 2 AZs
vpc_cidr = "10.2.0.0/16"
enable_nat_gateway = true
single_nat_gateway = false
enable_vpc_endpoints = true

# EKS - Mid-size configuration
eks_cluster_version = "1.27"
compute_node_group_instance_types = ["t3.large"]
compute_node_group_desired_size = 6
compute_node_group_min_size = 2
compute_node_group_max_size = 10
memory_node_group_instance_types = ["r6g.xlarge"]
memory_node_group_desired_size = 2
memory_node_group_min_size = 1
memory_node_group_max_size = 5
gpu_node_group_instance_types = ["g4dn.xlarge"]
gpu_node_group_desired_size = 1
gpu_node_group_min_size = 0
gpu_node_group_max_size = 3

# RDS - Multi-AZ, mid-size
rds_engine_version = "15.3"
rds_instance_class = "db.r6g.xlarge"
rds_allocated_storage = 250
rds_max_allocated_storage = 500
rds_multi_az = true
rds_read_replica_count = 1
rds_backup_retention_period = 3
rds_deletion_protection = true

# ElastiCache - Mid-size
elasticache_engine_version = "7.0"
elasticache_node_type = "cache.r6g.large"
elasticache_num_node_groups = 2
elasticache_replicas_per_node_group = 1

# S3 - With replication
s3_enable_replication = true

# Monitoring
enable_cloudwatch_logs = true
enable_cloudwatch_alarms = true

# Backup
enable_aws_backup = true
backup_retention_days = 14
