# terraform.tfvars - Development Environment Variable Values
# Component: GL-FOUND-X-003 - GreenLang Unit & Reference Normalizer
# Environment: Development
#
# IMPORTANT: This file should NOT be committed with sensitive values.
# Use environment variables or Terraform Cloud for secrets.

aws_region   = "us-east-1"
project_name = "gl-normalizer"
environment  = "dev"

# VPC Configuration (smaller for dev)
vpc_cidr = "10.10.0.0/16"
private_subnet_cidrs = ["10.10.1.0/24", "10.10.2.0/24", "10.10.3.0/24"]
public_subnet_cidrs  = ["10.10.101.0/24", "10.10.102.0/24", "10.10.103.0/24"]

# EKS Configuration
eks_cluster_version = "1.29"
eks_public_access_cidrs = ["0.0.0.0/0"]  # Restrict in actual dev environment

# Kafka password - PLACEHOLDER, set via environment variable:
# export TF_VAR_kafka_password="your-secure-password"
# kafka_password = "PLACEHOLDER"

# Common Tags
tags = {
  Project     = "GreenLang"
  Component   = "GL-FOUND-X-003"
  Environment = "dev"
  ManagedBy   = "terraform"
  Owner       = "platform-team"
  CostCenter  = "engineering"
}
