# Production Environment - Main Configuration

terraform {
  required_version = ">= 1.5.0"

  backend "s3" {
    bucket         = "vcci-scope3-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-west-2"
    dynamodb_table = "vcci-scope3-terraform-locks"
    encrypt        = true
  }
}

module "infrastructure" {
  source = "../../"

  # Pass all variables from terraform.tfvars
  project_name         = var.project_name
  environment          = var.environment
  aws_region           = var.aws_region
  aws_region_secondary = var.aws_region_secondary
  availability_zones   = var.availability_zones

  # VPC
  vpc_cidr             = var.vpc_cidr
  enable_nat_gateway   = var.enable_nat_gateway
  single_nat_gateway   = var.single_nat_gateway
  enable_vpc_endpoints = var.enable_vpc_endpoints

  # EKS
  eks_cluster_version                      = var.eks_cluster_version
  compute_node_group_instance_types        = var.compute_node_group_instance_types
  compute_node_group_desired_size          = var.compute_node_group_desired_size
  compute_node_group_min_size              = var.compute_node_group_min_size
  compute_node_group_max_size              = var.compute_node_group_max_size
  memory_node_group_instance_types         = var.memory_node_group_instance_types
  memory_node_group_desired_size           = var.memory_node_group_desired_size
  memory_node_group_min_size               = var.memory_node_group_min_size
  memory_node_group_max_size               = var.memory_node_group_max_size
  gpu_node_group_instance_types            = var.gpu_node_group_instance_types
  gpu_node_group_desired_size              = var.gpu_node_group_desired_size
  gpu_node_group_min_size                  = var.gpu_node_group_min_size
  gpu_node_group_max_size                  = var.gpu_node_group_max_size

  # RDS
  rds_engine_version          = var.rds_engine_version
  rds_instance_class          = var.rds_instance_class
  rds_allocated_storage       = var.rds_allocated_storage
  rds_max_allocated_storage   = var.rds_max_allocated_storage
  rds_multi_az                = var.rds_multi_az
  rds_read_replica_count      = var.rds_read_replica_count
  rds_backup_retention_period = var.rds_backup_retention_period
  rds_deletion_protection     = var.rds_deletion_protection

  # ElastiCache
  elasticache_engine_version          = var.elasticache_engine_version
  elasticache_node_type               = var.elasticache_node_type
  elasticache_num_node_groups         = var.elasticache_num_node_groups
  elasticache_replicas_per_node_group = var.elasticache_replicas_per_node_group

  # S3
  s3_enable_replication = var.s3_enable_replication

  # Monitoring
  enable_cloudwatch_logs   = var.enable_cloudwatch_logs
  enable_cloudwatch_alarms = var.enable_cloudwatch_alarms

  # Backup
  enable_aws_backup     = var.enable_aws_backup
  backup_retention_days = var.backup_retention_days
}

# Use same variables.tf as root module
variable "project_name" { type = string }
variable "environment" { type = string }
variable "aws_region" { type = string }
variable "aws_region_secondary" { type = string }
variable "availability_zones" { type = list(string) }
variable "vpc_cidr" { type = string }
variable "enable_nat_gateway" { type = bool }
variable "single_nat_gateway" { type = bool }
variable "enable_vpc_endpoints" { type = bool }
variable "eks_cluster_version" { type = string }
variable "compute_node_group_instance_types" { type = list(string) }
variable "compute_node_group_desired_size" { type = number }
variable "compute_node_group_min_size" { type = number }
variable "compute_node_group_max_size" { type = number }
variable "memory_node_group_instance_types" { type = list(string) }
variable "memory_node_group_desired_size" { type = number }
variable "memory_node_group_min_size" { type = number }
variable "memory_node_group_max_size" { type = number }
variable "gpu_node_group_instance_types" { type = list(string) }
variable "gpu_node_group_desired_size" { type = number }
variable "gpu_node_group_min_size" { type = number }
variable "gpu_node_group_max_size" { type = number }
variable "rds_engine_version" { type = string }
variable "rds_instance_class" { type = string }
variable "rds_allocated_storage" { type = number }
variable "rds_max_allocated_storage" { type = number }
variable "rds_multi_az" { type = bool }
variable "rds_read_replica_count" { type = number }
variable "rds_backup_retention_period" { type = number }
variable "rds_deletion_protection" { type = bool }
variable "elasticache_engine_version" { type = string }
variable "elasticache_node_type" { type = string }
variable "elasticache_num_node_groups" { type = number }
variable "elasticache_replicas_per_node_group" { type = number }
variable "s3_enable_replication" { type = bool }
variable "enable_cloudwatch_logs" { type = bool }
variable "enable_cloudwatch_alarms" { type = bool }
variable "enable_aws_backup" { type = bool }
variable "backup_retention_days" { type = number }
