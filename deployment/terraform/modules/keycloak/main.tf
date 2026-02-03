# GreenLang Keycloak Module
# TASK-152: Implement OAuth2/OIDC (Keycloak)
# Provides enterprise identity and access management for EKS deployment

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

# ==============================================================================
# LOCAL VALUES
# ==============================================================================

locals {
  # Environment-based defaults
  replicas = var.replicas != null ? var.replicas : (var.environment == "production" ? 3 : 1)

  log_level = var.log_level != null ? var.log_level : (var.environment == "production" ? "INFO" : "DEBUG")

  enable_pdb = var.enable_pod_disruption_budget != null ? var.enable_pod_disruption_budget : (var.environment == "production")

  enable_topology = var.enable_topology_spread != null ? var.enable_topology_spread : (var.environment == "production")

  enable_localhost = var.enable_localhost_redirect != null ? var.enable_localhost_redirect : (var.environment != "production")

  # Localhost redirect URIs for development
  localhost_redirect_uris = local.enable_localhost ? ["http://localhost:${var.localhost_port}/*"] : []
  localhost_origins       = local.enable_localhost ? ["http://localhost:${var.localhost_port}"] : []

  # Combined redirect URIs
  api_redirect_uris    = concat(var.api_client_redirect_uris, local.localhost_redirect_uris)
  api_web_origins      = concat(var.api_client_web_origins, local.localhost_origins)
  webapp_redirect_uris = concat(var.webapp_client_redirect_uris, local.localhost_redirect_uris)
  webapp_web_origins   = concat(var.webapp_client_web_origins, local.localhost_origins)

  # Standard labels
  common_labels = merge({
    "app.kubernetes.io/name"       = "keycloak"
    "app.kubernetes.io/part-of"    = "greenlang"
    "app.kubernetes.io/version"    = var.keycloak_version
    "app.kubernetes.io/managed-by" = "terraform"
    "environment"                  = var.environment
  }, var.labels)

  # Namespace reference
  namespace = var.create_namespace ? kubernetes_namespace.keycloak[0].metadata[0].name : var.namespace
}

# ==============================================================================
# KUBERNETES NAMESPACE
# ==============================================================================

resource "kubernetes_namespace" "keycloak" {
  count = var.create_namespace ? 1 : 0

  metadata {
    name = var.namespace
    labels = merge(local.common_labels, {
      "istio-injection" = var.enable_istio_injection ? "enabled" : "disabled"
    })
    annotations = var.annotations
  }
}

# ==============================================================================
# KUBERNETES SECRETS
# ==============================================================================

resource "kubernetes_secret" "keycloak_admin" {
  metadata {
    name      = "keycloak-admin-secret"
    namespace = local.namespace
    labels    = local.common_labels
  }

  data = {
    username = var.admin_username
    password = var.keycloak_admin_password
  }

  type = "Opaque"

  depends_on = [kubernetes_namespace.keycloak]
}

# ==============================================================================
# KEYCLOAK HELM RELEASE
# ==============================================================================

resource "helm_release" "keycloak" {
  name       = "keycloak"
  repository = "https://codecentric.github.io/helm-charts"
  chart      = "keycloakx"
  version    = var.helm_chart_version
  namespace  = local.namespace

  values = [
    yamlencode({
      replicas = local.replicas

      image = {
        repository = "quay.io/keycloak/keycloak"
        tag        = var.keycloak_version
      }

      command = [
        "/opt/keycloak/bin/kc.sh",
        "start",
        "--hostname=${var.domain}",
        "--hostname-strict=false",
        "--http-enabled=true",
        "--proxy=edge"
      ]

      extraEnv = concat([
        {
          name  = "KEYCLOAK_ADMIN"
          value = var.admin_username
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
          value = var.database_vendor
        },
        {
          name  = "KC_DB_URL"
          value = var.database_url
        },
        {
          name  = "KC_DB_SCHEMA"
          value = var.database_schema
        },
        {
          name  = "KC_HEALTH_ENABLED"
          value = tostring(var.enable_health_endpoints)
        },
        {
          name  = "KC_METRICS_ENABLED"
          value = tostring(var.enable_metrics)
        },
        {
          name  = "KC_LOG_LEVEL"
          value = local.log_level
        }
      ], var.extra_env)

      resources = var.resources

      serviceAccount = {
        create      = true
        name        = var.service_account_name
        annotations = var.enable_irsa ? { "eks.amazonaws.com/role-arn" = var.irsa_role_arn } : {}
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
        initialDelaySeconds = var.liveness_probe.initial_delay_seconds
        periodSeconds       = var.liveness_probe.period_seconds
        timeoutSeconds      = var.liveness_probe.timeout_seconds
        failureThreshold    = var.liveness_probe.failure_threshold
      }

      readinessProbe = {
        httpGet = {
          path = "/health/ready"
          port = "http"
        }
        initialDelaySeconds = var.readiness_probe.initial_delay_seconds
        periodSeconds       = var.readiness_probe.period_seconds
        timeoutSeconds      = var.readiness_probe.timeout_seconds
        failureThreshold    = var.readiness_probe.failure_threshold
        successThreshold    = var.readiness_probe.success_threshold
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
        annotations = merge({
          "kubernetes.io/ingress.class"                    = var.ingress_class
          "cert-manager.io/cluster-issuer"                 = var.cluster_issuer
          "nginx.ingress.kubernetes.io/ssl-redirect"       = tostring(var.enable_ssl_redirect)
          "nginx.ingress.kubernetes.io/proxy-buffer-size"  = "128k"
          "nginx.ingress.kubernetes.io/proxy-buffers"      = "4 256k"
        }, var.annotations)
        rules = [{
          host = var.domain
          paths = [{
            path     = "/"
            pathType = "Prefix"
          }]
        }]
        tls = [{
          secretName = var.tls_secret_name
          hosts      = [var.domain]
        }]
      }

      podDisruptionBudget = {
        enabled      = local.enable_pdb
        minAvailable = var.pdb_min_available
      }

      topologySpreadConstraints = local.enable_topology ? [
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

      affinity = var.enable_pod_anti_affinity ? {
        podAntiAffinity = {
          preferredDuringSchedulingIgnoredDuringExecution = [{
            weight = 100
            podAffinityTerm = {
              labelSelector = {
                matchLabels = {
                  "app.kubernetes.io/name" = "keycloakx"
                }
              }
              topologyKey = "kubernetes.io/hostname"
            }
          }]
        }
      } : {}

      nodeSelector     = var.node_selector
      tolerations      = var.tolerations
      priorityClassName = var.priority_class_name != "" ? var.priority_class_name : null
    })
  ]

  depends_on = [kubernetes_secret.keycloak_admin]
}

# ==============================================================================
# KEYCLOAK REALM CONFIGURATION
# ==============================================================================

resource "keycloak_realm" "greenlang" {
  count = var.create_realm ? 1 : 0

  realm   = var.realm_name
  enabled = true

  display_name = var.realm_display_name

  # Login settings
  login_theme              = "keycloak"
  registration_allowed     = var.registration_allowed
  reset_password_allowed   = true
  remember_me              = true
  verify_email             = true
  login_with_email_allowed = true
  duplicate_emails_allowed = false

  # Token settings
  access_token_lifespan                  = var.token_settings.access_token_lifespan
  access_token_lifespan_for_implicit_flow = var.token_settings.access_token_lifespan
  sso_session_idle_timeout               = var.token_settings.sso_session_idle_timeout
  sso_session_max_lifespan               = var.token_settings.sso_session_max_lifespan
  offline_session_idle_timeout           = var.token_settings.offline_session_idle_timeout
  offline_session_max_lifespan_enabled   = true
  offline_session_max_lifespan           = var.token_settings.offline_session_max_lifespan

  # Password policy
  password_policy = var.password_policy

  # Security defenses
  security_defenses {
    brute_force_detection {
      permanent_lockout                = var.brute_force_protection.permanent_lockout
      max_login_failures               = var.brute_force_protection.max_login_failures
      wait_increment_seconds           = var.brute_force_protection.wait_increment_seconds
      quick_login_check_milli_seconds  = 1000
      minimum_quick_login_wait_seconds = 60
      max_failure_wait_seconds         = var.brute_force_protection.max_failure_wait_seconds
      failure_reset_time_seconds       = var.brute_force_protection.failure_reset_time
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

# ==============================================================================
# KEYCLOAK CLIENTS
# ==============================================================================

# GreenLang API Client (Confidential)
resource "keycloak_openid_client" "greenlang_api" {
  count = var.create_clients ? 1 : 0

  realm_id                     = keycloak_realm.greenlang[0].id
  client_id                    = "greenlang-api"
  name                         = "GreenLang API"
  enabled                      = true
  access_type                  = "CONFIDENTIAL"
  standard_flow_enabled        = true
  direct_access_grants_enabled = true
  service_accounts_enabled     = true

  valid_redirect_uris = [for uri in local.api_redirect_uris : uri if uri != null]
  web_origins         = [for origin in local.api_web_origins : origin if origin != null]

  login_theme = "keycloak"

  depends_on = [keycloak_realm.greenlang]
}

# GreenLang Web App Client (Public with PKCE)
resource "keycloak_openid_client" "greenlang_webapp" {
  count = var.create_clients ? 1 : 0

  realm_id                     = keycloak_realm.greenlang[0].id
  client_id                    = "greenlang-webapp"
  name                         = "GreenLang Web Application"
  enabled                      = true
  access_type                  = "PUBLIC"
  standard_flow_enabled        = true
  direct_access_grants_enabled = false

  valid_redirect_uris = [for uri in local.webapp_redirect_uris : uri if uri != null]
  web_origins         = [for origin in local.webapp_web_origins : origin if origin != null]

  pkce_code_challenge_method = "S256"

  depends_on = [keycloak_realm.greenlang]
}

# ==============================================================================
# KEYCLOAK ROLES
# ==============================================================================

resource "keycloak_role" "viewer" {
  count = var.create_roles ? 1 : 0

  realm_id    = keycloak_realm.greenlang[0].id
  name        = "viewer"
  description = "Can view data and reports"
}

resource "keycloak_role" "operator" {
  count = var.create_roles ? 1 : 0

  realm_id        = keycloak_realm.greenlang[0].id
  name            = "operator"
  description     = "Can execute agents and write data"
  composite_roles = [keycloak_role.viewer[0].id]
}

resource "keycloak_role" "analyst" {
  count = var.create_roles ? 1 : 0

  realm_id        = keycloak_realm.greenlang[0].id
  name            = "analyst"
  description     = "Can analyze data and generate reports"
  composite_roles = [keycloak_role.operator[0].id]
}

resource "keycloak_role" "manager" {
  count = var.create_roles ? 1 : 0

  realm_id        = keycloak_realm.greenlang[0].id
  name            = "manager"
  description     = "Can manage agents and approve submissions"
  composite_roles = [keycloak_role.analyst[0].id]
}

resource "keycloak_role" "admin" {
  count = var.create_roles ? 1 : 0

  realm_id        = keycloak_realm.greenlang[0].id
  name            = "admin"
  description     = "Can manage users and system configuration"
  composite_roles = [keycloak_role.manager[0].id]
}

resource "keycloak_role" "super_admin" {
  count = var.create_roles ? 1 : 0

  realm_id        = keycloak_realm.greenlang[0].id
  name            = "super_admin"
  description     = "Full system access including tenant management"
  composite_roles = [keycloak_role.admin[0].id]
}

# ==============================================================================
# KEYCLOAK CLIENT SCOPES
# ==============================================================================

resource "keycloak_openid_client_scope" "greenlang_emissions" {
  count = var.create_client_scopes ? 1 : 0

  realm_id    = keycloak_realm.greenlang[0].id
  name        = "emissions"
  description = "Access to emission data"
}

resource "keycloak_openid_client_scope" "greenlang_reports" {
  count = var.create_client_scopes ? 1 : 0

  realm_id    = keycloak_realm.greenlang[0].id
  name        = "reports"
  description = "Access to reports"
}

resource "keycloak_openid_client_scope" "greenlang_agents" {
  count = var.create_client_scopes ? 1 : 0

  realm_id    = keycloak_realm.greenlang[0].id
  name        = "agents"
  description = "Access to run agents"
}
