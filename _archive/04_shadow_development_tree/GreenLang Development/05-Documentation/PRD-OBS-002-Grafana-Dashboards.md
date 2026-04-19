# PRD-OBS-002: Configure Grafana Dashboards

**Status:** Approved
**Version:** 1.0
**Created:** 2026-02-07
**Author:** GreenLang Platform Team
**Priority:** P0 - Critical
**Dependencies:** OBS-001 (Prometheus), INFRA-001 (K8s), INFRA-009 (Loki), SEC-001 (JWT Auth)

---

## 1. Executive Summary

This PRD defines the production-grade Grafana visualization and dashboarding platform for GreenLang Climate OS. While 50+ dashboard JSON files exist across the codebase and a basic single-replica Grafana deployment is in place, the platform lacks enterprise-grade deployment infrastructure, centralized provisioning, HA, authentication integration, multi-tenancy, dashboard-as-code pipeline, and a programmatic SDK. This PRD delivers a complete, production-ready Grafana platform.

### Current State Analysis

| Component | Status | Gaps |
|-----------|--------|------|
| Dashboard JSONs | 50+ exist | Scattered, no folder organization, no provisioning |
| Grafana Deployment | Basic K8s manifest | Single replica, SQLite DB, Grafana 10.2.0 (EOL) |
| Datasources | 4 configured | Missing Thanos, CloudWatch, PostgreSQL datasources |
| Authentication | Basic auth only | No OAuth2/OIDC, no SSO, no RBAC integration |
| HA/Persistence | None | SQLite, single replica, no session sharing |
| Helm Chart | None | No Helm-based deployment |
| Terraform | None | No AWS resources for Grafana backend |
| Dashboard Provisioning | Basic ConfigMap | No sidecar, no folder hierarchy, no versioning |
| Alerting | Basic | No unified alerting config, no notification channels |
| SDK | None | No programmatic dashboard management |
| CI/CD | None | No dashboard linting, validation, or deployment pipeline |
| Monitoring | None | No self-monitoring dashboards |

### Goals

1. **Production Deployment**: Grafana 11.4 with HA (2+ replicas), PostgreSQL backend, Redis session cache
2. **Helm Chart**: Complete Helm chart with environment overlays (dev/staging/prod)
3. **Terraform Module**: AWS resources (RDS PostgreSQL, S3 image storage, IAM roles)
4. **Dashboard Organization**: Folder hierarchy with 50+ dashboards organized by domain
5. **Data Sources**: 8+ datasources (Thanos, Prometheus, Loki, Jaeger, Alertmanager, PostgreSQL, CloudWatch, Tempo)
6. **Authentication**: OAuth2/OIDC with Keycloak/Cognito, role mapping, team sync
7. **Dashboard-as-Code**: Sidecar provisioning, CI/CD validation, Grafonnet library
8. **Python SDK**: Programmatic dashboard builder, folder manager, API client
9. **Alerting**: Unified alerting with Alertmanager, notification channels (Slack, PagerDuty, Email)
10. **Self-Monitoring**: Grafana health dashboard, performance alerts, usage analytics

---

## 2. Architecture

### 2.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        GreenLang Grafana Platform                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                       Grafana Application Layer                        │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │  Grafana 0   │  │  Grafana 1   │  │  Grafana 2   │  (HA Replicas) │  │
│  │  │  (Leader)    │  │  (Follower)  │  │  (Follower)  │                │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                │  │
│  │         │                 │                 │                         │  │
│  └─────────┼─────────────────┼─────────────────┼─────────────────────────┘  │
│            │                 │                 │                              │
│  ┌─────────┴─────────────────┴─────────────────┴─────────────────────────┐  │
│  │                       Backend Services                                 │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │  PostgreSQL  │  │  Redis       │  │  S3          │                │  │
│  │  │  (Grafana DB)│  │  (Sessions)  │  │  (Images)    │                │  │
│  │  │  Aurora RDS  │  │  ElastiCache │  │  Rendering   │                │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     Dashboard Provisioning                             │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │  Sidecar     │  │  ConfigMaps  │  │  CI/CD       │                │  │
│  │  │  (k8s-sidecar│  │  (Dashboard  │  │  (Lint +     │                │  │
│  │  │   container) │  │   JSONs)     │  │   Validate)  │                │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        Data Sources                                    │  │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐  │  │
│  │  │ Thanos │ │ Loki   │ │ Jaeger │ │ Alert- │ │ Postgre│ │ Cloud- │  │  │
│  │  │ Query  │ │        │ │ /Tempo │ │ manager│ │ SQL    │ │ Watch  │  │  │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     Authentication & Authorization                     │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │  OAuth2/OIDC │  │  Team Sync   │  │  Folder      │                │  │
│  │  │  (Cognito/   │  │  (LDAP/SCIM) │  │  Permissions │                │  │
│  │  │   Keycloak)  │  │              │  │              │                │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Overview

| Component | Version | Replicas | Purpose |
|-----------|---------|----------|---------|
| Grafana | 11.4.x | 2-3 (HA) | Visualization, dashboarding, alerting |
| k8s-sidecar | 1.27.x | 1 per pod | Auto-reload dashboards from ConfigMaps |
| Grafana Image Renderer | 3.11.x | 1-2 | PDF/PNG rendering for reports and alerts |
| PostgreSQL (Aurora) | 15.x | 2 (writer+reader) | Grafana metadata, sessions, preferences |
| Redis (ElastiCache) | 7.x | Existing cluster | Unified alerting HA, session sharing |
| S3 | - | - | Rendered image storage, plugin cache |

### 2.3 Dashboard Folder Hierarchy

```
GreenLang/
├── 00-Executive/
│   ├── platform-overview.json          (NEW - executive summary)
│   └── business-kpis.json             (NEW - SLA/uptime/cost)
├── 01-Infrastructure/
│   ├── kubernetes-cluster.json         (EXISTING)
│   ├── infrastructure-overview.json    (EXISTING)
│   ├── kong-gateway.json               (EXISTING)
│   ├── cicd-pipeline.json              (EXISTING)
│   ├── feature-flags.json              (EXISTING)
│   └── agent-factory-v1.json           (EXISTING)
├── 02-Data-Stores/
│   ├── postgresql-overview.json        (EXISTING)
│   ├── postgresql-replication.json     (EXISTING)
│   ├── timescaledb-hypertables.json    (EXISTING)
│   ├── pgbouncer-metrics.json          (EXISTING)
│   ├── redis-overview.json             (EXISTING)
│   ├── redis-replication.json          (EXISTING)
│   ├── redis-sentinel.json             (EXISTING)
│   ├── redis-streams.json              (EXISTING)
│   ├── redis-cluster.json              (EXISTING)
│   ├── s3-overview.json                (EXISTING)
│   ├── s3-data-lake.json              (EXISTING)
│   ├── s3-storage.json                (EXISTING)
│   └── s3-costs.json                  (EXISTING)
├── 03-Observability/
│   ├── prometheus-health.json          (EXISTING)
│   ├── thanos-health.json              (EXISTING)
│   ├── alertmanager-health.json        (EXISTING)
│   ├── log-exploration.json            (EXISTING)
│   ├── loki-operations.json            (EXISTING)
│   ├── grafana-health.json             (NEW - self-monitoring)
│   └── grafana-usage.json              (NEW - usage analytics)
├── 04-Security/
│   ├── auth-service.json               (EXISTING)
│   ├── rbac-service.json               (EXISTING)
│   ├── encryption-service.json         (EXISTING)
│   ├── tls-security.json               (EXISTING)
│   ├── audit-service.json              (EXISTING)
│   ├── secrets-service.json            (EXISTING)
│   ├── security-scanning.json          (EXISTING)
│   ├── security-operations.json        (EXISTING)
│   ├── soc2-compliance.json            (EXISTING)
│   ├── pii-service.json                (EXISTING)
│   └── security-posture.json           (NEW - unified security view)
├── 05-Applications/
│   ├── greenlang-agents.json           (EXISTING)
│   ├── api-performance.json            (EXISTING)
│   ├── dr-status.json                  (EXISTING)
│   └── application-health.json         (NEW - app overview)
└── 06-Alerts/
    └── active-alerts.json              (NEW - alert summary)
```

### 2.4 Data Flow

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Git Repository  │────▶│  GitHub Actions   │────▶│  K8s ConfigMaps  │
│  (dashboard JSON)│     │  (lint+validate)  │     │  (labeled)       │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                          │
┌──────────────────┐     ┌──────────────────┐             │
│  Python SDK      │────▶│  Grafana API     │             │
│  (programmatic)  │     │  (CRUD)          │             ▼
└──────────────────┘     └──────────────────┘     ┌──────────────────┐
                                                  │  k8s-sidecar    │
┌──────────────────┐     ┌──────────────────┐     │  (auto-reload)  │
│  Grafana UI      │────▶│  PostgreSQL      │     └────────┬─────────┘
│  (manual edits)  │     │  (persistence)   │              │
└──────────────────┘     └──────────────────┘              ▼
                                                  ┌──────────────────┐
                                                  │  Grafana Server  │
                                                  │  (render)        │
                                                  └──────────────────┘
```

---

## 3. Technical Specifications

### 3.1 Grafana Server Configuration

```ini
# grafana.ini - Production Configuration
[server]
protocol = http
http_addr = 0.0.0.0
http_port = 3000
domain = grafana.greenlang.io
root_url = https://%(domain)s/
serve_from_sub_path = false
enable_gzip = true
min_tls_version = TLS1.3

[database]
type = postgres
host = ${GF_DATABASE_HOST}:5432
name = grafana
user = ${GF_DATABASE_USER}
password = ${GF_DATABASE_PASSWORD}
ssl_mode = verify-full
ca_cert_path = /etc/ssl/certs/rds-ca-bundle.pem
max_open_conn = 50
max_idle_conn = 25
conn_max_lifetime = 14400
log_queries = false

[remote_cache]
type = redis
connstr = addr=${GF_REDIS_HOST}:6379,pool_size=100,db=2,ssl=true

[security]
admin_user = admin
admin_password = ${GF_SECURITY_ADMIN_PASSWORD}
secret_key = ${GF_SECURITY_SECRET_KEY}
disable_gravatar = true
cookie_secure = true
cookie_samesite = strict
allow_embedding = false
strict_transport_security = true
strict_transport_security_max_age_seconds = 63072000
strict_transport_security_preload = true
strict_transport_security_subdomains = true
x_content_type_options = true
x_xss_protection = true
content_security_policy = true
angular_support_enabled = false

[users]
allow_sign_up = false
allow_org_create = false
auto_assign_org = true
auto_assign_org_role = Viewer
default_theme = dark
viewers_can_edit = false

[auth]
login_cookie_name = grafana_session
login_maximum_inactive_lifetime_duration = 30m
login_maximum_lifetime_duration = 8h
disable_login_form = false
oauth_auto_login = false
oauth_allow_insecure_email_lookup = false

[auth.anonymous]
enabled = false

[auth.basic]
enabled = true

[auth.generic_oauth]
enabled = true
name = GreenLang SSO
allow_sign_up = true
auto_login = false
client_id = ${GF_AUTH_GENERIC_OAUTH_CLIENT_ID}
client_secret = ${GF_AUTH_GENERIC_OAUTH_CLIENT_SECRET}
scopes = openid profile email groups
auth_url = ${GF_AUTH_GENERIC_OAUTH_AUTH_URL}
token_url = ${GF_AUTH_GENERIC_OAUTH_TOKEN_URL}
api_url = ${GF_AUTH_GENERIC_OAUTH_API_URL}
role_attribute_path = contains(groups[*], 'platform-admins') && 'Admin' || contains(groups[*], 'platform-editors') && 'Editor' || 'Viewer'
groups_attribute_path = groups
allowed_groups = platform-admins platform-editors platform-viewers
use_pkce = true
use_refresh_token = true

[unified_alerting]
enabled = true
execute_alerts = true
evaluation_timeout = 30s
max_attempts = 3
min_interval = 10s
ha_listen_address = ${POD_IP}:9094
ha_peers = grafana-headless.monitoring.svc:9094
ha_advertise_address = ${POD_IP}:9094
ha_peer_timeout = 15s
ha_gossip_interval = 200ms

[alerting]
enabled = false

[smtp]
enabled = true
host = ${GF_SMTP_HOST}:587
user = ${GF_SMTP_USER}
password = ${GF_SMTP_PASSWORD}
from_address = grafana@greenlang.io
from_name = GreenLang Observability
startTLS_policy = MandatoryStartTLS

[log]
mode = console
level = info
filters = rendering:debug

[log.console]
level = info
format = json

[metrics]
enabled = true
interval_seconds = 10
disable_total_stats = false
basic_auth_username = ${GF_METRICS_BASIC_AUTH_USERNAME}
basic_auth_password = ${GF_METRICS_BASIC_AUTH_PASSWORD}

[analytics]
reporting_enabled = false
check_for_updates = false
check_for_plugin_updates = false

[dashboards]
versions_to_keep = 20
min_refresh_interval = 10s
default_home_dashboard_path =

[explore]
enabled = true

[plugins]
enable_alpha = false
plugin_admin_enabled = true
allow_loading_unsigned_plugins =

[feature_toggles]
enable = publicDashboards correlations traceToMetrics newTraceViewHeader traceqlSearch metricsSummary lokiQuerySplitting nestedFolders dashboardScene

[rendering]
server_url = http://grafana-image-renderer:8081/render
callback_url = http://grafana:3000/
concurrent_render_request_limit = 10

[date_formats]
full_date = YYYY-MM-DD HH:mm:ss
interval_second = HH:mm:ss
interval_minute = HH:mm
interval_hour = MM/DD HH:mm
interval_day = MM/DD
interval_month = YYYY-MM
interval_year = YYYY

[annotations]
cleanupjob_batchsize = 100

[live]
max_connections = 100
allowed_origins = https://grafana.greenlang.io
```

### 3.2 Data Sources Configuration

```yaml
apiVersion: 1

datasources:
  # Primary Metrics (via Thanos for long-term + HA)
  - name: Thanos
    type: prometheus
    uid: thanos
    access: proxy
    url: http://thanos-query.monitoring.svc:9090
    isDefault: true
    editable: false
    jsonData:
      timeInterval: "15s"
      queryTimeout: "60s"
      httpMethod: POST
      manageAlerts: true
      prometheusType: Thanos
      cacheLevel: High
      incrementalQuerying: true
      incrementalQueryOverlapWindow: 10m
      exemplarTraceIdDestinations:
        - name: traceID
          datasourceUid: jaeger

  # Direct Prometheus (for real-time, <7d queries)
  - name: Prometheus
    type: prometheus
    uid: prometheus
    access: proxy
    url: http://gl-prometheus-server.monitoring.svc:9090
    editable: false
    jsonData:
      timeInterval: "15s"
      queryTimeout: "30s"
      httpMethod: POST

  # Log Aggregation
  - name: Loki
    type: loki
    uid: loki
    access: proxy
    url: http://loki-read.monitoring.svc:3100
    editable: false
    jsonData:
      maxLines: 5000
      derivedFields:
        - datasourceUid: jaeger
          matcherRegex: '"trace_id":"(\w+)"'
          name: TraceID
          url: "$${__value.raw}"

  # Distributed Tracing
  - name: Jaeger
    type: jaeger
    uid: jaeger
    access: proxy
    url: http://jaeger-query.monitoring.svc:16686
    editable: false
    jsonData:
      tracesToLogsV2:
        datasourceUid: loki
        spanStartTimeShift: "-1h"
        spanEndTimeShift: "1h"
        tags:
          - key: service.name
            value: service_name
        filterByTraceID: true
        filterBySpanID: true
      tracesToMetrics:
        datasourceUid: thanos
        spanStartTimeShift: "-1h"
        spanEndTimeShift: "1h"
        tags:
          - key: service.name
            value: service
        queries:
          - name: Request Rate
            query: sum(rate(http_server_requests_total{service="$${__tags.service}"}[5m]))
          - name: Error Rate
            query: sum(rate(http_server_requests_total{service="$${__tags.service}",status_code=~"5.."}[5m]))

  # Alert Manager
  - name: Alertmanager
    type: alertmanager
    uid: alertmanager
    access: proxy
    url: http://gl-prometheus-alertmanager.monitoring.svc:9093
    editable: false
    jsonData:
      implementation: prometheus
      handleGrafanaManagedAlerts: true

  # Application Database (read replica)
  - name: PostgreSQL
    type: postgres
    uid: postgresql
    access: proxy
    url: ${POSTGRES_READ_HOST}:5432
    editable: false
    secureJsonData:
      password: ${POSTGRES_READ_PASSWORD}
    jsonData:
      database: greenlang
      user: grafana_reader
      sslmode: verify-full
      maxOpenConns: 10
      maxIdleConns: 5
      connMaxLifetime: 14400
      postgresVersion: 1500
      timescaledb: true

  # AWS CloudWatch
  - name: CloudWatch
    type: cloudwatch
    uid: cloudwatch
    access: proxy
    editable: false
    jsonData:
      authType: default
      defaultRegion: eu-west-1
      assumeRoleArn: ${CLOUDWATCH_ROLE_ARN}
```

### 3.3 Notification Channels

```yaml
apiVersion: 1

contactPoints:
  - orgId: 1
    name: platform-critical
    receivers:
      - uid: pagerduty-critical
        type: pagerduty
        settings:
          integrationKey: ${PAGERDUTY_INTEGRATION_KEY}
          severity: critical
          class: platform
          group: greenlang
      - uid: slack-critical
        type: slack
        settings:
          url: ${SLACK_WEBHOOK_CRITICAL}
          recipient: "#platform-alerts-critical"
          username: GreenLang Grafana
          icon_emoji: ":rotating_light:"
          mentionChannel: here
          text: |
            {{ template "slack.critical.text" . }}
          title: |
            {{ template "slack.critical.title" . }}

  - orgId: 1
    name: platform-warning
    receivers:
      - uid: slack-warning
        type: slack
        settings:
          url: ${SLACK_WEBHOOK_WARNING}
          recipient: "#platform-alerts"
          username: GreenLang Grafana
          icon_emoji: ":warning:"

  - orgId: 1
    name: platform-info
    receivers:
      - uid: email-info
        type: email
        settings:
          addresses: platform-team@greenlang.io
          singleEmail: true

policies:
  - orgId: 1
    receiver: platform-warning
    group_by:
      - grafana_folder
      - alertname
    group_wait: 30s
    group_interval: 5m
    repeat_interval: 4h
    routes:
      - receiver: platform-critical
        matchers:
          - severity = critical
        group_wait: 10s
        group_interval: 1m
        repeat_interval: 1h
        mute_time_intervals:
          - maintenance-window
      - receiver: platform-warning
        matchers:
          - severity = warning
        group_wait: 30s
        group_interval: 5m
        repeat_interval: 4h
      - receiver: platform-info
        matchers:
          - severity = info
        group_wait: 1m
        group_interval: 10m
        repeat_interval: 12h

muteTimes:
  - orgId: 1
    name: maintenance-window
    time_intervals:
      - times:
          - start_time: "02:00"
            end_time: "04:00"
        weekdays:
          - sunday
        location: UTC
```

### 3.4 Grafana Image Renderer

```yaml
# Dedicated rendering service for PDF/PNG export
renderer:
  enabled: true
  replicas: 2
  image:
    repository: grafana/grafana-image-renderer
    tag: 3.11.3
  env:
    HTTP_HOST: 0.0.0.0
    HTTP_PORT: "8081"
    ENABLE_METRICS: "true"
    RENDERING_MODE: clustered
    RENDERING_CLUSTERING_MODE: browser
    RENDERING_CLUSTERING_MAX_CONCURRENCY: "5"
    RENDERING_CLUSTERING_TIMEOUT: "60"
    LOG_LEVEL: info
  resources:
    requests:
      cpu: 100m
      memory: 256Mi
    limits:
      cpu: 500m
      memory: 1Gi
  service:
    type: ClusterIP
    port: 8081
```

---

## 4. Helm Chart Specification

### 4.1 Chart Structure

```
deployment/helm/grafana/
├── Chart.yaml
├── values.yaml
├── values-dev.yaml
├── values-staging.yaml
├── values-prod.yaml
└── templates/
    ├── _helpers.tpl
    ├── configmap-grafana-ini.yaml
    ├── configmap-datasources.yaml
    ├── configmap-dashboard-providers.yaml
    ├── configmap-notifiers.yaml
    ├── deployment.yaml
    ├── headless-service.yaml
    ├── service.yaml
    ├── ingress.yaml
    ├── hpa.yaml
    ├── pdb.yaml
    ├── networkpolicy.yaml
    ├── serviceaccount.yaml
    ├── servicemonitor.yaml
    ├── secret.yaml
    ├── pvc.yaml
    ├── deployment-renderer.yaml
    ├── service-renderer.yaml
    └── NOTES.txt
```

### 4.2 Resource Sizing

| Environment | Replicas | CPU Req | CPU Lim | Mem Req | Mem Lim | Storage |
|-------------|----------|---------|---------|---------|---------|---------|
| Dev | 1 | 100m | 500m | 256Mi | 512Mi | 5Gi |
| Staging | 2 | 250m | 1000m | 512Mi | 1Gi | 10Gi |
| Production | 3 | 500m | 2000m | 1Gi | 2Gi | 20Gi |

---

## 5. Terraform Module Specification

### 5.1 Module Structure

```
deployment/terraform/modules/grafana/
├── main.tf           # RDS PostgreSQL for Grafana backend
├── variables.tf      # Input variables
├── outputs.tf        # Output values
├── iam.tf            # IAM roles for CloudWatch, S3
├── s3.tf             # S3 bucket for image storage
└── security-groups.tf # Network security
```

### 5.2 AWS Resources

| Resource | Purpose | Configuration |
|----------|---------|---------------|
| RDS PostgreSQL | Grafana metadata DB | db.t3.medium, Multi-AZ, encrypted, 7d backup |
| S3 Bucket | Image rendering storage | Private, lifecycle 30d, encrypted |
| IAM Role | CloudWatch data source | Read-only CloudWatch access via IRSA |
| Security Group | RDS access | Ingress from EKS nodes only |

---

## 6. Python SDK Specification

### 6.1 Module Structure

```
greenlang/monitoring/grafana/
├── __init__.py
├── client.py           # Grafana HTTP API client
├── dashboard_builder.py # Programmatic dashboard construction
├── panel_builder.py     # Panel/visualization builder
├── folder_manager.py    # Folder CRUD and permissions
├── datasource_manager.py # Datasource management
├── alert_manager.py     # Alert rule management
├── provisioning.py      # Dashboard provisioning utilities
└── models.py            # Pydantic models for Grafana objects
```

### 6.2 Key Classes

```python
class GrafanaClient:
    """HTTP client for Grafana API with retry and auth."""
    async def get_dashboard(uid: str) -> Dashboard
    async def create_dashboard(dashboard: Dashboard) -> str
    async def update_dashboard(dashboard: Dashboard) -> str
    async def delete_dashboard(uid: str) -> None
    async def search_dashboards(query: str, folder_id: int) -> list[Dashboard]
    async def get_health() -> HealthStatus

class DashboardBuilder:
    """Fluent API for building Grafana dashboards."""
    def with_title(title: str) -> DashboardBuilder
    def with_tags(tags: list[str]) -> DashboardBuilder
    def with_time_range(from_: str, to: str) -> DashboardBuilder
    def with_refresh(interval: str) -> DashboardBuilder
    def add_row(title: str) -> RowBuilder
    def add_panel(panel: Panel) -> DashboardBuilder
    def add_variable(variable: Variable) -> DashboardBuilder
    def add_annotation(annotation: Annotation) -> DashboardBuilder
    def build() -> dict

class PanelBuilder:
    """Fluent API for building Grafana panels."""
    def stat(title: str) -> PanelBuilder
    def gauge(title: str) -> PanelBuilder
    def timeseries(title: str) -> PanelBuilder
    def table(title: str) -> PanelBuilder
    def heatmap(title: str) -> PanelBuilder
    def logs(title: str) -> PanelBuilder
    def with_target(datasource: str, expr: str) -> PanelBuilder
    def with_threshold(value: float, color: str) -> PanelBuilder
    def with_override(matcher: dict, properties: list) -> PanelBuilder
    def build() -> dict

class FolderManager:
    """Manage Grafana folders and permissions."""
    async def create_folder(title: str, uid: str) -> Folder
    async def set_permissions(folder_uid: str, permissions: list) -> None
    async def sync_folders(folder_hierarchy: dict) -> None
```

---

## 7. Kubernetes Manifests

### 7.1 Structure

```
deployment/kubernetes/grafana/
├── namespace.yaml
├── configmap-dashboards-infrastructure.yaml
├── configmap-dashboards-datastores.yaml
├── configmap-dashboards-observability.yaml
├── configmap-dashboards-security.yaml
├── configmap-dashboards-applications.yaml
├── configmap-dashboards-executive.yaml
├── configmap-dashboards-alerts.yaml
├── networkpolicy.yaml
├── servicemonitor.yaml
├── poddisruptionbudget.yaml
└── kustomization.yaml
```

---

## 8. New Dashboards

### 8.1 Platform Overview (Executive)

| Panel | Type | Metric/Query |
|-------|------|-------------|
| Platform Uptime | Stat | `avg_over_time(up{job=~".*greenlang.*"}[24h]) * 100` |
| Active Agents | Stat | `gl_agent_factory_agents_total{state="running"}` |
| API Request Rate | Timeseries | `sum(rate(http_server_requests_total[5m]))` |
| Error Rate | Gauge | `sum(rate(http_server_requests_total{status_code=~"5.."}[5m])) / sum(rate(http_server_requests_total[5m])) * 100` |
| P99 Latency | Timeseries | `histogram_quantile(0.99, sum(rate(http_server_request_duration_seconds_bucket[5m])) by (le))` |
| Active Alerts | Stat | `count(ALERTS{alertstate="firing"})` |
| Storage Usage | Bar Gauge | `sum(kubelet_volume_stats_used_bytes) by (persistentvolumeclaim)` |
| Cost Estimate | Stat | Custom CloudWatch query |

### 8.2 Grafana Health (Self-Monitoring)

| Panel | Type | Metric/Query |
|-------|------|-------------|
| Grafana Uptime | Stat | `grafana_build_info` |
| Active Users | Timeseries | `grafana_stat_active_users` |
| Dashboard Load Time | Histogram | `grafana_api_dashboard_get_milliseconds` |
| API Request Rate | Timeseries | `rate(grafana_http_request_total[5m])` |
| API Errors | Timeseries | `rate(grafana_http_request_total{status_code=~"5.."}[5m])` |
| DB Connection Pool | Gauge | `grafana_database_conn_open` |
| Alerting Queue | Timeseries | `grafana_alerting_queue_capacity` |
| Rendering Duration | Histogram | `grafana_rendering_request_total` |
| Data Source Health | Table | `grafana_datasource_request_total` |
| Plugin Status | Table | `grafana_plugin_build_info` |
| Memory Usage | Timeseries | `process_resident_memory_bytes{job="grafana"}` |
| CPU Usage | Timeseries | `rate(process_cpu_seconds_total{job="grafana"}[5m])` |
| Cache Hit Rate | Gauge | Remote cache metrics |
| Session Count | Stat | `grafana_stat_active_sessions` |
| Dashboard Count | Stat | `grafana_stat_total_dashboards` |
| Alert Rule Count | Stat | `grafana_stat_total_alert_rules` |

### 8.3 Grafana Usage Analytics

| Panel | Type | Metric/Query |
|-------|------|-------------|
| Daily Active Users | Timeseries | `grafana_stat_active_users` |
| Dashboard Views by Folder | Bar Chart | `sum(grafana_api_dashboard_get_milliseconds_count) by (folder)` |
| Top 10 Dashboards | Table | Dashboard request counts |
| Data Source Query Rate | Timeseries | `rate(grafana_datasource_request_total[5m])` by datasource |
| Slow Queries | Table | `grafana_datasource_request_duration_seconds > 5` |
| Login Activity | Timeseries | `rate(grafana_api_login_post_total[1h])` |
| Alert Notification Rate | Timeseries | `rate(grafana_alerting_notification_sent_total[1h])` |

### 8.4 Security Posture (Unified)

| Panel | Type | Metric/Query |
|-------|------|-------------|
| Security Score | Gauge | Composite of all security metrics |
| Auth Success/Failure | Timeseries | `rate(gl_auth_total[5m])` by status |
| RBAC Denials | Timeseries | `rate(gl_rbac_authorization_total{decision="deny"}[5m])` |
| Active Vulnerabilities | Stat | `gl_security_vulnerabilities_total` by severity |
| Encryption Operations | Timeseries | `rate(gl_encryption_operations_total[5m])` |
| TLS Certificate Expiry | Table | `gl_tls_certificate_expiry_seconds` |
| Audit Events | Timeseries | `rate(gl_audit_events_total[5m])` |
| PII Detections | Timeseries | `rate(gl_pii_detections_total[5m])` |
| SOC 2 Compliance | Gauge | `gl_soc2_compliance_score` |

### 8.5 Active Alerts Summary

| Panel | Type | Metric/Query |
|-------|------|-------------|
| Firing Alerts | Stat | `count(ALERTS{alertstate="firing"})` |
| Pending Alerts | Stat | `count(ALERTS{alertstate="pending"})` |
| Alerts by Severity | Pie Chart | `count(ALERTS{alertstate="firing"}) by (severity)` |
| Alert Timeline | Timeseries | `changes(ALERTS[1h])` |
| Alert History | Table | Alertmanager API data |
| Silenced Alerts | Stat | Alertmanager silences |
| Top Firing Alert Rules | Table | Most frequently firing alerts |

---

## 9. Alert Rules

### 9.1 Grafana Health Alerts

```yaml
groups:
  - name: grafana-health
    interval: 30s
    rules:
      - alert: GrafanaDown
        expr: up{job="grafana"} == 0
        for: 2m
        labels:
          severity: critical
          service: grafana
        annotations:
          summary: "Grafana instance is down"
          runbook_url: "https://docs.greenlang.io/runbooks/grafana-down"

      - alert: GrafanaHighMemory
        expr: process_resident_memory_bytes{job="grafana"} > 1.5e9
        for: 10m
        labels:
          severity: warning
          service: grafana
        annotations:
          summary: "Grafana using >1.5GB memory"

      - alert: GrafanaDashboardLoadSlow
        expr: histogram_quantile(0.95, rate(grafana_api_dashboard_get_milliseconds_bucket[5m])) > 3000
        for: 5m
        labels:
          severity: warning
          service: grafana
        annotations:
          summary: "P95 dashboard load time >3s"

      - alert: GrafanaAPIErrors
        expr: rate(grafana_http_request_total{status_code=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
          service: grafana
        annotations:
          summary: "Grafana API error rate elevated"

      - alert: GrafanaDBConnectionPoolExhausted
        expr: grafana_database_conn_open / grafana_database_conn_max > 0.8
        for: 5m
        labels:
          severity: warning
          service: grafana
        annotations:
          summary: "Grafana DB connection pool >80% utilized"

      - alert: GrafanaAlertingQueueFull
        expr: grafana_alerting_queue_capacity > 0.9
        for: 2m
        labels:
          severity: critical
          service: grafana
        annotations:
          summary: "Grafana alerting queue near capacity"

      - alert: GrafanaDataSourceUnreachable
        expr: grafana_datasource_request_total{status="error"} > 0
        for: 5m
        labels:
          severity: warning
          service: grafana
        annotations:
          summary: "Grafana cannot reach data source {{ $labels.datasource }}"

      - alert: GrafanaRenderingFailed
        expr: rate(grafana_rendering_request_total{status="failure"}[5m]) > 0
        for: 5m
        labels:
          severity: warning
          service: grafana
        annotations:
          summary: "Grafana image rendering failures detected"

      - alert: GrafanaHighLoginFailureRate
        expr: rate(grafana_api_login_post_total{status_code!="200"}[5m]) / rate(grafana_api_login_post_total[5m]) > 0.3
        for: 5m
        labels:
          severity: warning
          service: grafana
        annotations:
          summary: "High login failure rate (>30%)"

      - alert: GrafanaCacheHitRateLow
        expr: grafana_remote_cache_hit_total / (grafana_remote_cache_hit_total + grafana_remote_cache_miss_total) < 0.5
        for: 15m
        labels:
          severity: info
          service: grafana
        annotations:
          summary: "Grafana cache hit rate below 50%"

      - alert: GrafanaBackendDBDown
        expr: grafana_database_conn_open == 0
        for: 1m
        labels:
          severity: critical
          service: grafana
        annotations:
          summary: "Grafana cannot connect to PostgreSQL backend"

      - alert: GrafanaTooManyDashboards
        expr: grafana_stat_total_dashboards > 200
        for: 1h
        labels:
          severity: info
          service: grafana
        annotations:
          summary: "Dashboard sprawl detected (>200 dashboards)"
```

---

## 10. CI/CD Pipeline

### 10.1 Dashboard Validation Workflow

```yaml
name: grafana-dashboard-ci
on:
  pull_request:
    paths:
      - 'deployment/monitoring/dashboards/**'
      - 'deployment/kubernetes/grafana/**'
      - 'deployment/helm/grafana/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - name: Lint Dashboard JSON
        # Validate JSON syntax, required fields, datasource references
      - name: Validate Panel Queries
        # Check PromQL/LogQL syntax
      - name: Check Dashboard UIDs
        # Ensure unique UIDs across all dashboards
      - name: Validate Folder Structure
        # Ensure dashboards are in correct folders
      - name: Schema Validation
        # Validate against Grafana dashboard schema
```

---

## 11. Testing Requirements

| Test Type | Count | Coverage Target |
|-----------|-------|----------------|
| Unit Tests (SDK) | 80+ | 85% |
| Integration Tests (API) | 30+ | Core flows |
| Dashboard Validation | 50+ | All dashboards |
| Load Tests | 10+ | Concurrent users |

---

## 12. Success Criteria

| Metric | Target |
|--------|--------|
| Grafana Uptime | 99.9% |
| Dashboard Load Time (P95) | < 3s |
| Data Source Query (P99) | < 10s |
| Alert Delivery Time | < 60s |
| Dashboard Count | 50+ organized |
| Concurrent Users | 50+ supported |
| Image Rendering | < 30s per render |
| Cache Hit Rate | > 70% |

---

## 13. Deliverables Summary

| # | Deliverable | Files |
|---|-------------|-------|
| 1 | Helm Chart (Grafana) | ~18 files |
| 2 | Terraform Module | ~5 files |
| 3 | K8s Manifests (dashboards, provisioning) | ~12 files |
| 4 | New Dashboards (5 new JSONs) | ~5 files |
| 5 | Python SDK | ~9 files |
| 6 | Monitoring (alerts + dashboards) | ~3 files |
| 7 | CI/CD Workflow | ~1 file |
| 8 | Environment Configs (Terraform) | ~3 files |
| 9 | Runbooks | ~3 files |
| 10 | Tests (unit + integration + load) | ~8 files |
| **Total** | | **~67 files** |
