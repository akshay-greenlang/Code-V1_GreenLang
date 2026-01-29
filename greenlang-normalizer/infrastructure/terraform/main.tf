# GreenLang Normalizer - Terraform Infrastructure
# This configuration deploys the normalizer service to AWS EKS

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.24"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12"
    }
  }

  backend "s3" {
    bucket         = "greenlang-terraform-state"
    key            = "normalizer/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}

# Variables
variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "greenlang-prod"
}

variable "normalizer_replicas" {
  description = "Number of normalizer replicas"
  type        = number
  default     = 3
}

# Provider configuration
provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project     = "GreenLang"
      Component   = "Normalizer"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

data "aws_eks_cluster" "cluster" {
  name = var.cluster_name
}

data "aws_eks_cluster_auth" "cluster" {
  name = var.cluster_name
}

provider "kubernetes" {
  host                   = data.aws_eks_cluster.cluster.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority[0].data)
  token                  = data.aws_eks_cluster_auth.cluster.token
}

provider "helm" {
  kubernetes {
    host                   = data.aws_eks_cluster.cluster.endpoint
    cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority[0].data)
    token                  = data.aws_eks_cluster_auth.cluster.token
  }
}

# Namespace
resource "kubernetes_namespace" "greenlang" {
  metadata {
    name = "greenlang"
    labels = {
      name        = "greenlang"
      environment = var.environment
    }
  }
}

# Redis for caching
resource "helm_release" "redis" {
  name       = "normalizer-redis"
  repository = "https://charts.bitnami.com/bitnami"
  chart      = "redis"
  version    = "18.6.1"
  namespace  = kubernetes_namespace.greenlang.metadata[0].name

  values = [
    <<-EOT
    architecture: standalone
    auth:
      enabled: true
      password: "${random_password.redis_password.result}"
    master:
      persistence:
        size: 10Gi
      resources:
        requests:
          memory: 256Mi
          cpu: 100m
        limits:
          memory: 1Gi
          cpu: 500m
    metrics:
      enabled: true
    EOT
  ]
}

resource "random_password" "redis_password" {
  length  = 32
  special = false
}

# PostgreSQL for persistence
resource "helm_release" "postgresql" {
  name       = "normalizer-postgresql"
  repository = "https://charts.bitnami.com/bitnami"
  chart      = "postgresql"
  version    = "13.2.24"
  namespace  = kubernetes_namespace.greenlang.metadata[0].name

  values = [
    <<-EOT
    auth:
      username: normalizer
      password: "${random_password.db_password.result}"
      database: normalizer
    primary:
      persistence:
        size: 50Gi
      resources:
        requests:
          memory: 512Mi
          cpu: 250m
        limits:
          memory: 2Gi
          cpu: 1000m
    metrics:
      enabled: true
    EOT
  ]
}

resource "random_password" "db_password" {
  length  = 32
  special = false
}

# Secrets
resource "kubernetes_secret" "normalizer_secrets" {
  metadata {
    name      = "gl-normalizer-secrets"
    namespace = kubernetes_namespace.greenlang.metadata[0].name
  }

  data = {
    "database-url" = "postgresql://normalizer:${random_password.db_password.result}@normalizer-postgresql:5432/normalizer"
    "redis-url"    = "redis://:${random_password.redis_password.result}@normalizer-redis-master:6379/0"
    "jwt-secret"   = random_password.jwt_secret.result
  }
}

resource "random_password" "jwt_secret" {
  length  = 64
  special = false
}

# Outputs
output "namespace" {
  description = "Kubernetes namespace"
  value       = kubernetes_namespace.greenlang.metadata[0].name
}

output "redis_host" {
  description = "Redis host"
  value       = "normalizer-redis-master"
}

output "postgresql_host" {
  description = "PostgreSQL host"
  value       = "normalizer-postgresql"
}
