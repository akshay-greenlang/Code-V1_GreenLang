# GreenLang Vault Module
# TASK-155: Implement API Key Management (Vault)
# Provides secret management and API key storage for EKS deployment with AWS KMS integration
#
# This module deploys:
# - HashiCorp Vault via Helm chart with HA configuration
# - AWS IAM role for KMS auto-unseal (IRSA)
# - Kubernetes authentication backend
# - KV v2 secrets engines for API keys, database credentials, and service secrets
# - Vault policies and Kubernetes auth roles for GreenLang services

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
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
    random = {
      source  = "hashicorp/random"
      version = ">= 3.0"
    }
  }
}

# Local values for computed configuration
locals {
  # Determine HA settings based on environment or explicit configuration
  ha_enabled  = var.ha_enabled != null ? var.ha_enabled : var.environment == "production"
  ha_replicas = local.ha_enabled ? var.ha_replicas : 1

  # Injector replicas based on environment
  injector_replicas = var.environment == "production" ? var.injector_replicas : 1

  # Common labels for all resources
  common_labels = merge(
    {
      "app.kubernetes.io/name"       = "vault"
      "app.kubernetes.io/part-of"    = "greenlang"
      "app.kubernetes.io/managed-by" = "terraform"
      "environment"                  = var.environment
    },
    var.labels
  )

  # AWS region from KMS configuration
  aws_region = var.aws_region
}

# Namespace
resource "kubernetes_namespace" "vault" {
  metadata {
    name = var.namespace
    labels = local.common_labels
  }
}

# Vault Helm Release
resource "helm_release" "vault" {
  name       = "vault"
  repository = "https://helm.releases.hashicorp.com"
  chart      = "vault"
  version    = var.vault_chart_version
  namespace  = kubernetes_namespace.vault.metadata[0].name

  values = [
    yamlencode({
      global = {
        enabled = true
      }

      injector = {
        enabled  = true
        replicas = local.injector_replicas
        resources = {
          requests = {
            memory = var.injector_resources.requests.memory
            cpu    = var.injector_resources.requests.cpu
          }
          limits = {
            memory = var.injector_resources.limits.memory
            cpu    = var.injector_resources.limits.cpu
          }
        }
        agentDefaults = {
          cpuLimit   = var.agent_resources.cpu_limit
          cpuRequest = var.agent_resources.cpu_request
          memLimit   = var.agent_resources.memory_limit
          memRequest = var.agent_resources.memory_request
        }
      }

      server = {
        image = var.vault_image_tag != "" ? {
          tag = var.vault_image_tag
        } : null

        ha = {
          enabled  = local.ha_enabled
          replicas = local.ha_replicas
          raft = {
            enabled   = true
            setNodeId = true
            config    = <<-EOF
              ui = ${var.ui_enabled}

              listener "tcp" {
                tls_disable = 1
                address = "[::]:8200"
                cluster_address = "[::]:8201"
                telemetry {
                  unauthenticated_metrics_access = ${var.enable_unauthenticated_metrics}
                }
              }

              storage "raft" {
                path = "/vault/data"
                %{for i in range(local.ha_replicas)~}
                retry_join {
                  leader_api_addr = "http://vault-${i}.vault-internal:8200"
                }
                %{endfor~}
              }

              seal "awskms" {
                region     = "${local.aws_region}"
                kms_key_id = "${var.kms_key_id}"
              }

              service_registration "kubernetes" {}

              telemetry {
                prometheus_retention_time = "${var.prometheus_retention_time}"
                disable_hostname = true
              }
            EOF
          }
        }

        resources = {
          requests = {
            memory = var.server_resources.requests.memory
            cpu    = var.server_resources.requests.cpu
          }
          limits = {
            memory = var.server_resources.limits.memory
            cpu    = var.server_resources.limits.cpu
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
          size         = var.data_storage_size
          storageClass = var.storage_class
        }

        auditStorage = {
          enabled      = true
          size         = var.audit_storage_size
          storageClass = var.storage_class
        }

        ingress = {
          enabled = var.ingress_enabled
          annotations = {
            "kubernetes.io/ingress.class"              = var.ingress_class
            "cert-manager.io/cluster-issuer"           = var.cluster_issuer
            "nginx.ingress.kubernetes.io/ssl-redirect" = tostring(var.enable_ssl_redirect)
          }
          hosts = [{
            host  = var.domain
            paths = ["/"]
          }]
          tls = [{
            secretName = var.tls_secret_name
            hosts      = [var.domain]
          }]
        }

        # Pod disruption budget for HA
        disruptionBudget = var.pod_disruption_budget_enabled ? {
          enabled      = true
          minAvailable = var.pod_disruption_budget_min_available
        } : {
          enabled = false
        }

        # Pod anti-affinity for spreading across nodes
        affinity = var.affinity_enabled ? {
          podAntiAffinity = {
            preferredDuringSchedulingIgnoredDuringExecution = [{
              weight = 100
              podAffinityTerm = {
                labelSelector = {
                  matchLabels = {
                    "app.kubernetes.io/name"     = "vault"
                    "app.kubernetes.io/instance" = "vault"
                  }
                }
                topologyKey = "kubernetes.io/hostname"
              }
            }]
          }
        } : null

        # Priority class
        priorityClassName = var.priority_class_name != "" ? var.priority_class_name : null
      }

      ui = {
        enabled = var.ui_enabled
      }

      csi = {
        enabled = var.csi_enabled
        resources = {
          requests = {
            memory = var.csi_resources.requests.memory
            cpu    = var.csi_resources.requests.cpu
          }
          limits = {
            memory = var.csi_resources.limits.memory
            cpu    = var.csi_resources.limits.cpu
          }
        }
      }
    })
  ]

  # Allow extra values to override defaults
  dynamic "set" {
    for_each = var.extra_helm_values != "" ? [1] : []
    content {
      name  = "extraValues"
      value = var.extra_helm_values
    }
  }
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
        Resource = "arn:aws:kms:${local.aws_region}:${data.aws_caller_identity.current.account_id}:key/${var.kms_key_id}"
      }
    ]
  })
}

# Additional IAM policy for Secrets Manager (if root token storage is enabled)
resource "aws_iam_role_policy" "vault_secrets_manager" {
  count = var.store_root_token_in_secrets_manager ? 1 : 0

  name = "${var.eks_cluster_name}-vault-secrets-manager"
  role = aws_iam_role.vault.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:PutSecretValue",
          "secretsmanager:UpdateSecret"
        ]
        Resource = aws_secretsmanager_secret.root_token[0].arn
      }
    ]
  })
}

# Kubernetes Auth Backend Configuration
resource "vault_auth_backend" "kubernetes" {
  type = "kubernetes"
  path = var.kubernetes_auth_path

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
  path        = var.api_keys_path
  type        = "kv"
  options     = { version = "2" }
  description = "GreenLang API Keys"

  depends_on = [helm_release.vault]
}

# KV Secrets Engine for Database Credentials
resource "vault_mount" "database" {
  path        = var.database_path
  type        = "kv"
  options     = { version = "2" }
  description = "GreenLang Database Credentials"

  depends_on = [helm_release.vault]
}

# KV Secrets Engine for Service Secrets
resource "vault_mount" "services" {
  path        = var.services_path
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
    path "${var.api_keys_path}/data/*" {
      capabilities = ["read"]
    }

    path "${var.api_keys_path}/metadata/*" {
      capabilities = ["read", "list"]
    }

    # Read database credentials
    path "${var.database_path}/data/*" {
      capabilities = ["read"]
    }

    # Read service secrets
    path "${var.services_path}/data/*" {
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
    path "${var.api_keys_path}/data/*" {
      capabilities = ["create", "read", "update", "delete"]
    }

    path "${var.api_keys_path}/metadata/*" {
      capabilities = ["read", "list", "delete"]
    }

    # Read database credentials
    path "${var.database_path}/data/*" {
      capabilities = ["read"]
    }

    # Read service secrets
    path "${var.services_path}/data/*" {
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
    path "${var.api_keys_path}/*" {
      capabilities = ["create", "read", "update", "delete", "list"]
    }

    path "${var.database_path}/*" {
      capabilities = ["create", "read", "update", "delete", "list"]
    }

    path "${var.services_path}/*" {
      capabilities = ["create", "read", "update", "delete", "list"]
    }
  EOT

  depends_on = [helm_release.vault]
}

# Kubernetes Role for GreenLang Agents
resource "vault_kubernetes_auth_backend_role" "greenlang_agents" {
  backend                          = vault_auth_backend.kubernetes.path
  role_name                        = "greenlang-agents"
  bound_service_account_names      = var.greenlang_agents_service_accounts
  bound_service_account_namespaces = [var.greenlang_agents_namespace]
  token_ttl                        = var.token_ttl
  token_max_ttl                    = var.token_max_ttl
  token_policies                   = [vault_policy.greenlang_agents_read.name]

  depends_on = [vault_policy.greenlang_agents_read]
}

# Kubernetes Role for GreenLang API
resource "vault_kubernetes_auth_backend_role" "greenlang_api" {
  backend                          = vault_auth_backend.kubernetes.path
  role_name                        = "greenlang-api"
  bound_service_account_names      = var.greenlang_api_service_accounts
  bound_service_account_namespaces = [var.greenlang_agents_namespace]
  token_ttl                        = var.token_ttl
  token_max_ttl                    = var.token_max_ttl
  token_policies                   = [vault_policy.greenlang_api.name]

  depends_on = [vault_policy.greenlang_api]
}

# Kubernetes Role for Secrets Rotation
resource "vault_kubernetes_auth_backend_role" "greenlang_rotation" {
  backend                          = vault_auth_backend.kubernetes.path
  role_name                        = "greenlang-rotation"
  bound_service_account_names      = var.greenlang_rotation_service_accounts
  bound_service_account_namespaces = [var.greenlang_agents_namespace]
  token_ttl                        = var.token_ttl
  token_max_ttl                    = min(var.token_max_ttl, 7200)  # Rotation role has shorter max TTL for security
  token_policies                   = [vault_policy.greenlang_rotation.name]

  depends_on = [vault_policy.greenlang_rotation]
}

# =============================================================================
# AWS Secrets Manager for Root Token (Optional)
# =============================================================================
# WARNING: The root token should only be stored temporarily during initial setup.
# It should be revoked after creating appropriate policies and auth methods.

resource "aws_secretsmanager_secret" "root_token" {
  count = var.store_root_token_in_secrets_manager ? 1 : 0

  name        = "${var.eks_cluster_name}-${var.root_token_secret_name}"
  description = "Vault root token for ${var.eks_cluster_name} - REVOKE AFTER INITIAL SETUP"

  tags = merge(var.tags, {
    Name        = "${var.eks_cluster_name}-${var.root_token_secret_name}"
    Environment = var.environment
    Warning     = "REVOKE_AFTER_INITIAL_SETUP"
  })
}

# Note: The actual root token value should be stored manually after Vault initialization
# This resource only creates the secret container

# =============================================================================
# ServiceMonitor for Prometheus Operator (Optional)
# =============================================================================

resource "kubernetes_manifest" "vault_service_monitor" {
  count = var.enable_unauthenticated_metrics ? 1 : 0

  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "ServiceMonitor"
    metadata = {
      name      = "vault"
      namespace = kubernetes_namespace.vault.metadata[0].name
      labels    = local.common_labels
    }
    spec = {
      selector = {
        matchLabels = {
          "app.kubernetes.io/name"     = "vault"
          "app.kubernetes.io/instance" = "vault"
        }
      }
      endpoints = [{
        port     = "http"
        path     = "/v1/sys/metrics"
        interval = "30s"
        params = {
          format = ["prometheus"]
        }
      }]
    }
  }

  depends_on = [helm_release.vault]
}

# =============================================================================
# NetworkPolicy for Vault (Security Hardening)
# =============================================================================

resource "kubernetes_network_policy" "vault" {
  metadata {
    name      = "vault-network-policy"
    namespace = kubernetes_namespace.vault.metadata[0].name
    labels    = local.common_labels
  }

  spec {
    pod_selector {
      match_labels = {
        "app.kubernetes.io/name" = "vault"
      }
    }

    # Allow ingress from any pod that needs to access Vault
    ingress {
      from {
        namespace_selector {}
      }
      ports {
        port     = 8200
        protocol = "TCP"
      }
    }

    # Allow cluster communication between Vault replicas
    ingress {
      from {
        pod_selector {
          match_labels = {
            "app.kubernetes.io/name" = "vault"
          }
        }
      }
      ports {
        port     = 8201
        protocol = "TCP"
      }
    }

    # Allow egress to AWS services (KMS) and DNS
    egress {
      to {
        ip_block {
          cidr = "0.0.0.0/0"
        }
      }
      ports {
        port     = 443
        protocol = "TCP"
      }
    }

    egress {
      to {
        namespace_selector {}
        pod_selector {
          match_labels = {
            "k8s-app" = "kube-dns"
          }
        }
      }
      ports {
        port     = 53
        protocol = "UDP"
      }
    }

    policy_types = ["Ingress", "Egress"]
  }
}
