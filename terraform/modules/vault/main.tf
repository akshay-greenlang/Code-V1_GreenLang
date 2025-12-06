# GreenLang Vault Module
# TASK-155: Implement API Key Management (Vault)
# Provides secret management and API key storage

terraform {
  required_version = ">= 1.0"
  required_providers {
    helm = {
      source  = "hashicorp/helm"
      version = ">= 2.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.0"
    }
    vault = {
      source  = "hashicorp/vault"
      version = ">= 3.0"
    }
  }
}

# Variables
variable "namespace" {
  description = "Kubernetes namespace for Vault"
  type        = string
  default     = "vault"
}

variable "domain" {
  description = "Vault domain"
  type        = string
  default     = "vault.greenlang.io"
}

variable "environment" {
  description = "Environment (dev, staging, production)"
  type        = string
  default     = "production"
}

variable "eks_cluster_name" {
  description = "EKS cluster name"
  type        = string
}

variable "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  type        = string
}

variable "eks_cluster_ca_cert" {
  description = "EKS cluster CA certificate"
  type        = string
}

variable "kms_key_id" {
  description = "KMS key ID for auto-unseal"
  type        = string
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Namespace
resource "kubernetes_namespace" "vault" {
  metadata {
    name = var.namespace
    labels = {
      "app.kubernetes.io/name"    = "vault"
      "app.kubernetes.io/part-of" = "greenlang"
      "environment"               = var.environment
    }
  }
}

# Vault Helm Release
resource "helm_release" "vault" {
  name       = "vault"
  repository = "https://helm.releases.hashicorp.com"
  chart      = "vault"
  version    = "0.27.0"
  namespace  = kubernetes_namespace.vault.metadata[0].name

  values = [
    yamlencode({
      global = {
        enabled = true
      }

      injector = {
        enabled = true
        replicas = var.environment == "production" ? 2 : 1
        resources = {
          requests = {
            memory = "64Mi"
            cpu    = "50m"
          }
          limits = {
            memory = "128Mi"
            cpu    = "100m"
          }
        }
        agentDefaults = {
          cpuLimit   = "500m"
          cpuRequest = "250m"
          memLimit   = "128Mi"
          memRequest = "64Mi"
        }
      }

      server = {
        ha = {
          enabled  = var.environment == "production"
          replicas = var.environment == "production" ? 3 : 1
          raft = {
            enabled = true
            setNodeId = true
            config = <<-EOF
              ui = true

              listener "tcp" {
                tls_disable = 1
                address = "[::]:8200"
                cluster_address = "[::]:8201"
                telemetry {
                  unauthenticated_metrics_access = true
                }
              }

              storage "raft" {
                path = "/vault/data"
                retry_join {
                  leader_api_addr = "http://vault-0.vault-internal:8200"
                }
                retry_join {
                  leader_api_addr = "http://vault-1.vault-internal:8200"
                }
                retry_join {
                  leader_api_addr = "http://vault-2.vault-internal:8200"
                }
              }

              seal "awskms" {
                region     = "us-east-1"
                kms_key_id = "${var.kms_key_id}"
              }

              service_registration "kubernetes" {}

              telemetry {
                prometheus_retention_time = "30s"
                disable_hostname = true
              }
            EOF
          }
        }

        resources = {
          requests = {
            memory = "256Mi"
            cpu    = "250m"
          }
          limits = {
            memory = "512Mi"
            cpu    = "500m"
          }
        }

        serviceAccount = {
          create = true
          name   = "vault"
          annotations = {
            "eks.amazonaws.com/role-arn" = aws_iam_role.vault.arn
          }
        }

        dataStorage = {
          enabled      = true
          size         = "10Gi"
          storageClass = "gp3"
        }

        auditStorage = {
          enabled      = true
          size         = "10Gi"
          storageClass = "gp3"
        }

        ingress = {
          enabled = true
          annotations = {
            "kubernetes.io/ingress.class"              = "nginx"
            "cert-manager.io/cluster-issuer"           = "letsencrypt-prod"
            "nginx.ingress.kubernetes.io/ssl-redirect" = "true"
          }
          hosts = [{
            host  = var.domain
            paths = ["/"]
          }]
          tls = [{
            secretName = "vault-tls"
            hosts      = [var.domain]
          }]
        }
      }

      ui = {
        enabled = true
      }

      csi = {
        enabled = true
        resources = {
          requests = {
            memory = "64Mi"
            cpu    = "50m"
          }
          limits = {
            memory = "128Mi"
            cpu    = "100m"
          }
        }
      }
    })
  ]
}

# IAM Role for Vault (IRSA)
data "aws_caller_identity" "current" {}

data "aws_eks_cluster" "cluster" {
  name = var.eks_cluster_name
}

resource "aws_iam_role" "vault" {
  name = "${var.eks_cluster_name}-vault"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/${replace(data.aws_eks_cluster.cluster.identity[0].oidc[0].issuer, "https://", "")}"
        }
        Condition = {
          StringEquals = {
            "${replace(data.aws_eks_cluster.cluster.identity[0].oidc[0].issuer, "https://", "")}:sub" = "system:serviceaccount:${var.namespace}:vault"
            "${replace(data.aws_eks_cluster.cluster.identity[0].oidc[0].issuer, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy" "vault_kms" {
  name = "${var.eks_cluster_name}-vault-kms"
  role = aws_iam_role.vault.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = "arn:aws:kms:*:${data.aws_caller_identity.current.account_id}:key/${var.kms_key_id}"
      }
    ]
  })
}

# Kubernetes Auth Backend Configuration
resource "vault_auth_backend" "kubernetes" {
  type = "kubernetes"
  path = "kubernetes"

  depends_on = [helm_release.vault]
}

resource "vault_kubernetes_auth_backend_config" "config" {
  backend                = vault_auth_backend.kubernetes.path
  kubernetes_host        = var.eks_cluster_endpoint
  kubernetes_ca_cert     = base64decode(var.eks_cluster_ca_cert)
  issuer                 = "https://kubernetes.default.svc.cluster.local"
  disable_iss_validation = true

  depends_on = [vault_auth_backend.kubernetes]
}

# KV Secrets Engine for API Keys
resource "vault_mount" "api_keys" {
  path        = "greenlang/api-keys"
  type        = "kv"
  options     = { version = "2" }
  description = "GreenLang API Keys"

  depends_on = [helm_release.vault]
}

# KV Secrets Engine for Database Credentials
resource "vault_mount" "database" {
  path        = "greenlang/database"
  type        = "kv"
  options     = { version = "2" }
  description = "GreenLang Database Credentials"

  depends_on = [helm_release.vault]
}

# KV Secrets Engine for Service Secrets
resource "vault_mount" "services" {
  path        = "greenlang/services"
  type        = "kv"
  options     = { version = "2" }
  description = "GreenLang Service Secrets"

  depends_on = [helm_release.vault]
}

# Policy for GreenLang Agents (read-only)
resource "vault_policy" "greenlang_agents_read" {
  name = "greenlang-agents-read"

  policy = <<-EOT
    # Read API keys
    path "greenlang/api-keys/data/*" {
      capabilities = ["read"]
    }

    path "greenlang/api-keys/metadata/*" {
      capabilities = ["read", "list"]
    }

    # Read database credentials
    path "greenlang/database/data/*" {
      capabilities = ["read"]
    }

    # Read service secrets
    path "greenlang/services/data/*" {
      capabilities = ["read"]
    }
  EOT

  depends_on = [helm_release.vault]
}

# Policy for GreenLang API (read/write API keys)
resource "vault_policy" "greenlang_api" {
  name = "greenlang-api"

  policy = <<-EOT
    # Manage API keys
    path "greenlang/api-keys/data/*" {
      capabilities = ["create", "read", "update", "delete"]
    }

    path "greenlang/api-keys/metadata/*" {
      capabilities = ["read", "list", "delete"]
    }

    # Read database credentials
    path "greenlang/database/data/*" {
      capabilities = ["read"]
    }

    # Read service secrets
    path "greenlang/services/data/*" {
      capabilities = ["read"]
    }
  EOT

  depends_on = [helm_release.vault]
}

# Policy for Secrets Rotation
resource "vault_policy" "greenlang_rotation" {
  name = "greenlang-rotation"

  policy = <<-EOT
    # Full access to all secrets for rotation
    path "greenlang/api-keys/*" {
      capabilities = ["create", "read", "update", "delete", "list"]
    }

    path "greenlang/database/*" {
      capabilities = ["create", "read", "update", "delete", "list"]
    }

    path "greenlang/services/*" {
      capabilities = ["create", "read", "update", "delete", "list"]
    }
  EOT

  depends_on = [helm_release.vault]
}

# Kubernetes Role for GreenLang Agents
resource "vault_kubernetes_auth_backend_role" "greenlang_agents" {
  backend                          = vault_auth_backend.kubernetes.path
  role_name                        = "greenlang-agents"
  bound_service_account_names      = ["greenlang-agents-sa", "default"]
  bound_service_account_namespaces = ["greenlang-agents"]
  token_ttl                        = 3600
  token_max_ttl                    = 86400
  token_policies                   = ["greenlang-agents-read"]

  depends_on = [vault_policy.greenlang_agents_read]
}

# Kubernetes Role for GreenLang API
resource "vault_kubernetes_auth_backend_role" "greenlang_api" {
  backend                          = vault_auth_backend.kubernetes.path
  role_name                        = "greenlang-api"
  bound_service_account_names      = ["greenlang-api-sa"]
  bound_service_account_namespaces = ["greenlang-agents"]
  token_ttl                        = 3600
  token_max_ttl                    = 86400
  token_policies                   = ["greenlang-api"]

  depends_on = [vault_policy.greenlang_api]
}

# Kubernetes Role for Secrets Rotation
resource "vault_kubernetes_auth_backend_role" "greenlang_rotation" {
  backend                          = vault_auth_backend.kubernetes.path
  role_name                        = "greenlang-rotation"
  bound_service_account_names      = ["greenlang-rotation-sa"]
  bound_service_account_namespaces = ["greenlang-agents"]
  token_ttl                        = 3600
  token_max_ttl                    = 7200
  token_policies                   = ["greenlang-rotation"]

  depends_on = [vault_policy.greenlang_rotation]
}

# Outputs
output "vault_url" {
  description = "Vault URL"
  value       = "https://${var.domain}"
}

output "vault_role_arn" {
  description = "Vault IAM role ARN"
  value       = aws_iam_role.vault.arn
}

output "api_keys_path" {
  description = "Path to API keys secrets"
  value       = vault_mount.api_keys.path
}

output "database_path" {
  description = "Path to database secrets"
  value       = vault_mount.database.path
}
