# GreenLang Keycloak Module
# TASK-152: Implement OAuth2/OIDC (Keycloak)
# Provides enterprise identity and access management

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
    keycloak = {
      source  = "mrparkers/keycloak"
      version = ">= 4.0"
    }
  }
}

# Variables
variable "namespace" {
  description = "Kubernetes namespace for Keycloak"
  type        = string
  default     = "keycloak"
}

variable "keycloak_admin_password" {
  description = "Keycloak admin password"
  type        = string
  sensitive   = true
}

variable "database_url" {
  description = "PostgreSQL database URL"
  type        = string
  sensitive   = true
}

variable "domain" {
  description = "Keycloak domain"
  type        = string
  default     = "auth.greenlang.io"
}

variable "environment" {
  description = "Environment (dev, staging, production)"
  type        = string
  default     = "production"
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Namespace
resource "kubernetes_namespace" "keycloak" {
  metadata {
    name = var.namespace
    labels = {
      "app.kubernetes.io/name"       = "keycloak"
      "app.kubernetes.io/part-of"    = "greenlang"
      "environment"                  = var.environment
      "istio-injection"              = "enabled"
    }
  }
}

# Admin Secret
resource "kubernetes_secret" "keycloak_admin" {
  metadata {
    name      = "keycloak-admin-secret"
    namespace = kubernetes_namespace.keycloak.metadata[0].name
  }

  data = {
    password = var.keycloak_admin_password
  }

  type = "Opaque"
}

# Keycloak Helm Release
resource "helm_release" "keycloak" {
  name       = "keycloak"
  repository = "https://codecentric.github.io/helm-charts"
  chart      = "keycloakx"
  version    = "2.3.0"
  namespace  = kubernetes_namespace.keycloak.metadata[0].name

  values = [
    yamlencode({
      replicas = var.environment == "production" ? 3 : 1

      image = {
        repository = "quay.io/keycloak/keycloak"
        tag        = "23.0"
      }

      command = [
        "/opt/keycloak/bin/kc.sh",
        "start",
        "--hostname=${var.domain}",
        "--hostname-strict=false",
        "--http-enabled=true",
        "--proxy=edge"
      ]

      extraEnv = [
        {
          name  = "KEYCLOAK_ADMIN"
          value = "admin"
        },
        {
          name = "KEYCLOAK_ADMIN_PASSWORD"
          valueFrom = {
            secretKeyRef = {
              name = kubernetes_secret.keycloak_admin.metadata[0].name
              key  = "password"
            }
          }
        },
        {
          name  = "KC_DB"
          value = "postgres"
        },
        {
          name  = "KC_DB_URL"
          value = var.database_url
        },
        {
          name  = "KC_HEALTH_ENABLED"
          value = "true"
        },
        {
          name  = "KC_METRICS_ENABLED"
          value = "true"
        },
        {
          name  = "KC_LOG_LEVEL"
          value = var.environment == "production" ? "INFO" : "DEBUG"
        }
      ]

      resources = {
        requests = {
          memory = "512Mi"
          cpu    = "250m"
        }
        limits = {
          memory = "1Gi"
          cpu    = "1000m"
        }
      }

      serviceAccount = {
        create = true
        name   = "keycloak"
      }

      podSecurityContext = {
        runAsNonRoot = true
        runAsUser    = 1000
        fsGroup      = 1000
      }

      securityContext = {
        allowPrivilegeEscalation = false
        readOnlyRootFilesystem   = false  # Keycloak needs write access
        capabilities = {
          drop = ["ALL"]
        }
      }

      livenessProbe = {
        httpGet = {
          path = "/health/live"
          port = "http"
        }
        initialDelaySeconds = 60
        periodSeconds       = 10
        failureThreshold    = 3
      }

      readinessProbe = {
        httpGet = {
          path = "/health/ready"
          port = "http"
        }
        initialDelaySeconds = 30
        periodSeconds       = 10
        failureThreshold    = 3
      }

      service = {
        type = "ClusterIP"
        ports = {
          http = {
            port       = 8080
            protocol   = "TCP"
            targetPort = "http"
          }
        }
      }

      ingress = {
        enabled = true
        annotations = {
          "kubernetes.io/ingress.class"              = "nginx"
          "cert-manager.io/cluster-issuer"           = "letsencrypt-prod"
          "nginx.ingress.kubernetes.io/ssl-redirect" = "true"
          "nginx.ingress.kubernetes.io/proxy-buffer-size" = "128k"
        }
        rules = [{
          host = var.domain
          paths = [{
            path     = "/"
            pathType = "Prefix"
          }]
        }]
        tls = [{
          secretName = "keycloak-tls"
          hosts      = [var.domain]
        }]
      }

      podDisruptionBudget = {
        enabled      = var.environment == "production"
        minAvailable = 2
      }

      topologySpreadConstraints = var.environment == "production" ? [
        {
          maxSkew           = 1
          topologyKey       = "topology.kubernetes.io/zone"
          whenUnsatisfiable = "DoNotSchedule"
          labelSelector = {
            matchLabels = {
              "app.kubernetes.io/name" = "keycloakx"
            }
          }
        }
      ] : []
    })
  ]

  depends_on = [kubernetes_secret.keycloak_admin]
}

# Keycloak Realm Configuration
resource "keycloak_realm" "greenlang" {
  realm   = "greenlang"
  enabled = true

  display_name = "GreenLang"

  # Login settings
  login_theme         = "keycloak"
  registration_allowed = false
  reset_password_allowed = true
  remember_me         = true
  verify_email        = true
  login_with_email_allowed = true
  duplicate_emails_allowed = false

  # Token settings
  access_token_lifespan                 = "5m"
  access_token_lifespan_for_implicit_flow = "5m"
  sso_session_idle_timeout              = "30m"
  sso_session_max_lifespan              = "10h"
  offline_session_idle_timeout          = "720h"
  offline_session_max_lifespan_enabled  = true
  offline_session_max_lifespan          = "720h"

  # Password policy
  password_policy = "length(12) and upperCase(1) and lowerCase(1) and specialChars(1) and digits(1) and notUsername"

  # Security defenses
  security_defenses {
    brute_force_detection {
      permanent_lockout                = false
      max_login_failures               = 5
      wait_increment_seconds           = 60
      quick_login_check_milli_seconds  = 1000
      minimum_quick_login_wait_seconds = 60
      max_failure_wait_seconds         = 900
      failure_reset_time_seconds       = 43200
    }
    headers {
      x_frame_options                     = "SAMEORIGIN"
      content_security_policy             = "frame-src 'self'; frame-ancestors 'self'; object-src 'none';"
      content_security_policy_report_only = ""
      x_content_type_options              = "nosniff"
      x_robots_tag                        = "none"
      x_xss_protection                    = "1; mode=block"
      strict_transport_security           = "max-age=31536000; includeSubDomains"
    }
  }

  # OTP policy
  otp_policy {
    type              = "totp"
    algorithm         = "HmacSHA1"
    digits            = 6
    initial_counter   = 0
    look_ahead_window = 1
    period            = 30
  }

  depends_on = [helm_release.keycloak]
}

# GreenLang API Client
resource "keycloak_openid_client" "greenlang_api" {
  realm_id                     = keycloak_realm.greenlang.id
  client_id                    = "greenlang-api"
  name                         = "GreenLang API"
  enabled                      = true
  access_type                  = "CONFIDENTIAL"
  standard_flow_enabled        = true
  direct_access_grants_enabled = true
  service_accounts_enabled     = true

  valid_redirect_uris = [
    "https://api.greenlang.io/*",
    "https://app.greenlang.io/*",
    var.environment != "production" ? "http://localhost:*/*" : null,
  ]

  web_origins = [
    "https://api.greenlang.io",
    "https://app.greenlang.io",
    var.environment != "production" ? "http://localhost:3000" : null,
  ]

  login_theme = "keycloak"

  depends_on = [keycloak_realm.greenlang]
}

# GreenLang Web App Client
resource "keycloak_openid_client" "greenlang_webapp" {
  realm_id                     = keycloak_realm.greenlang.id
  client_id                    = "greenlang-webapp"
  name                         = "GreenLang Web Application"
  enabled                      = true
  access_type                  = "PUBLIC"
  standard_flow_enabled        = true
  direct_access_grants_enabled = false

  valid_redirect_uris = [
    "https://app.greenlang.io/*",
    var.environment != "production" ? "http://localhost:3000/*" : null,
  ]

  web_origins = [
    "https://app.greenlang.io",
    var.environment != "production" ? "http://localhost:3000" : null,
  ]

  pkce_code_challenge_method = "S256"

  depends_on = [keycloak_realm.greenlang]
}

# Roles
resource "keycloak_role" "viewer" {
  realm_id    = keycloak_realm.greenlang.id
  name        = "viewer"
  description = "Can view data and reports"
}

resource "keycloak_role" "operator" {
  realm_id    = keycloak_realm.greenlang.id
  name        = "operator"
  description = "Can execute agents and write data"
  composite_roles = [keycloak_role.viewer.id]
}

resource "keycloak_role" "analyst" {
  realm_id    = keycloak_realm.greenlang.id
  name        = "analyst"
  description = "Can analyze data and generate reports"
  composite_roles = [keycloak_role.operator.id]
}

resource "keycloak_role" "manager" {
  realm_id    = keycloak_realm.greenlang.id
  name        = "manager"
  description = "Can manage agents and approve submissions"
  composite_roles = [keycloak_role.analyst.id]
}

resource "keycloak_role" "admin" {
  realm_id    = keycloak_realm.greenlang.id
  name        = "admin"
  description = "Can manage users and system configuration"
  composite_roles = [keycloak_role.manager.id]
}

resource "keycloak_role" "super_admin" {
  realm_id    = keycloak_realm.greenlang.id
  name        = "super_admin"
  description = "Full system access including tenant management"
  composite_roles = [keycloak_role.admin.id]
}

# Client Scopes
resource "keycloak_openid_client_scope" "greenlang_emissions" {
  realm_id    = keycloak_realm.greenlang.id
  name        = "emissions"
  description = "Access to emission data"
}

resource "keycloak_openid_client_scope" "greenlang_reports" {
  realm_id    = keycloak_realm.greenlang.id
  name        = "reports"
  description = "Access to reports"
}

resource "keycloak_openid_client_scope" "greenlang_agents" {
  realm_id    = keycloak_realm.greenlang.id
  name        = "agents"
  description = "Access to run agents"
}

# Outputs
output "keycloak_url" {
  description = "Keycloak URL"
  value       = "https://${var.domain}"
}

output "realm_name" {
  description = "Keycloak realm name"
  value       = keycloak_realm.greenlang.realm
}

output "api_client_id" {
  description = "API client ID"
  value       = keycloak_openid_client.greenlang_api.client_id
}

output "webapp_client_id" {
  description = "Web app client ID"
  value       = keycloak_openid_client.greenlang_webapp.client_id
}
