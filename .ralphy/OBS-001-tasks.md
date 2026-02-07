# OBS-001: Deploy Prometheus Metrics Collection - Development Tasks

**Status:** COMPLETE
**Created:** 2026-02-06
**Completed:** 2026-02-06
**Priority:** P0 - CRITICAL
**Depends On:** INFRA-001 (K8s), INFRA-002 (PostgreSQL), INFRA-003 (Redis)
**PRD:** `GreenLang Development/05-Documentation/PRD-OBS-001-Prometheus-Metrics-Collection.md`
**Existing Code:** prometheus.yml (326 lines), alert_rules.yml (600+ lines), 35+ Grafana dashboards, 73+ Python metrics

---

## Phase 1: Terraform Module for Prometheus Stack (P0)

### 1.1 Module Structure
- [ ] Create `deployment/terraform/modules/prometheus-stack/versions.tf`:
  - Terraform >= 1.5.0
  - AWS provider >= 5.0
  - Helm provider >= 2.12
  - Kubernetes provider >= 2.25

### 1.2 Variables
- [ ] Create `deployment/terraform/modules/prometheus-stack/variables.tf`:
  - `environment` (string, required)
  - `cluster_name` (string, required)
  - `prometheus_replica_count` (number, default: 2)
  - `prometheus_retention_days` (number, default: 7)
  - `prometheus_storage_size` (string, default: "50Gi")
  - `thanos_retention_raw` (string, default: "30d")
  - `thanos_retention_5m` (string, default: "120d")
  - `thanos_retention_1h` (string, default: "730d")
  - `alertmanager_slack_webhook` (string, sensitive)
  - `alertmanager_pagerduty_key` (string, sensitive)
  - `enable_pushgateway` (bool, default: true)
  - `enable_thanos` (bool, default: true)
  - `additional_scrape_configs` (list)
  - `tags` (map)

### 1.3 S3 Storage for Thanos
- [ ] Create `deployment/terraform/modules/prometheus-stack/s3.tf`:
  - `aws_s3_bucket.thanos_metrics` with versioning
  - `aws_s3_bucket_lifecycle_configuration` (Intelligent Tiering at 30d, delete at 730d)
  - `aws_s3_bucket_server_side_encryption_configuration` (AES256)
  - `aws_s3_bucket_public_access_block` (all blocked)
  - `aws_s3_bucket_policy` for Thanos IAM access

### 1.4 IAM for IRSA
- [ ] Create `deployment/terraform/modules/prometheus-stack/iam.tf`:
  - `aws_iam_role.prometheus_thanos` for S3 access
  - `aws_iam_policy.thanos_s3_access` (s3:PutObject, GetObject, ListBucket, DeleteObject)
  - OIDC provider trust relationship
  - ServiceAccount annotation output

### 1.5 Prometheus Helm Release
- [ ] Create `deployment/terraform/modules/prometheus-stack/prometheus.tf`:
  - `helm_release.kube_prometheus_stack`
  - kube-prometheus-stack chart v56.x
  - Values from template with Thanos sidecar
  - ServiceMonitor for self-monitoring
  - Namespace creation

### 1.6 Thanos Helm Release
- [ ] Create `deployment/terraform/modules/prometheus-stack/thanos.tf`:
  - `helm_release.thanos` (bitnami/thanos)
  - Query, Store Gateway, Compactor, Ruler components
  - S3 objstore config from secret
  - Query Frontend for caching

### 1.7 Alertmanager Configuration
- [ ] Create `deployment/terraform/modules/prometheus-stack/alertmanager.tf`:
  - `kubernetes_secret.alertmanager_secrets` for Slack/PagerDuty
  - Alertmanager config template
  - Route configuration (critical → PagerDuty, warning → Slack)

### 1.8 PushGateway
- [ ] Create `deployment/terraform/modules/prometheus-stack/pushgateway.tf`:
  - `helm_release.pushgateway` conditional on enable_pushgateway
  - HA deployment (2 replicas)
  - Persistence enabled
  - ServiceMonitor for self-monitoring

### 1.9 ServiceMonitors
- [ ] Create `deployment/terraform/modules/prometheus-stack/servicemonitors.tf`:
  - ServiceMonitor CRDs for GreenLang services
  - PodMonitor for annotation-based discovery
  - Additional scrape configs integration

### 1.10 Module Main and Outputs
- [ ] Create `deployment/terraform/modules/prometheus-stack/main.tf`:
  - Local variables
  - Data sources (EKS cluster, OIDC provider)
  - Kubernetes namespace
- [ ] Create `deployment/terraform/modules/prometheus-stack/outputs.tf`:
  - `prometheus_endpoint`
  - `thanos_query_endpoint`
  - `alertmanager_endpoint`
  - `pushgateway_endpoint`
  - `thanos_bucket_name`
  - `thanos_bucket_arn`
  - `prometheus_service_account`

### 1.11 Module Documentation
- [ ] Create `deployment/terraform/modules/prometheus-stack/README.md`:
  - Module description
  - Requirements
  - Input variables table
  - Output values table
  - Usage examples
  - Architecture diagram

---

## Phase 2: Helm Values Enhancement (P0)

### 2.1 kube-prometheus-stack Values
- [ ] Create `deployment/helm/prometheus-stack/Chart.yaml`:
  - Name: greenlang-prometheus-stack
  - Version: 1.0.0
  - Dependencies: kube-prometheus-stack, thanos

### 2.2 Base Values
- [ ] Create `deployment/helm/prometheus-stack/values.yaml`:
  - Prometheus HA config (2 replicas, anti-affinity)
  - Thanos sidecar configuration
  - External labels (cluster, region, replica)
  - 7-day local retention, 50Gi storage
  - ServiceMonitor and PodMonitor selectors
  - Alertmanager HA config
  - Grafana datasources (Prometheus, Thanos, Alertmanager)

### 2.3 Environment Overlays
- [ ] Create `deployment/helm/prometheus-stack/values-dev.yaml`:
  - 1 replica (no HA in dev)
  - 2-day retention
  - 10Gi storage
  - Thanos disabled

- [ ] Create `deployment/helm/prometheus-stack/values-staging.yaml`:
  - 2 replicas
  - 7-day retention
  - 25Gi storage
  - Thanos enabled (30d S3 retention)

- [ ] Create `deployment/helm/prometheus-stack/values-prod.yaml`:
  - 2 replicas
  - 7-day local retention
  - 50Gi storage
  - Thanos enabled (2-year S3 retention)
  - PagerDuty integration

### 2.4 Thanos Values
- [ ] Create `deployment/helm/thanos/Chart.yaml`:
  - Name: greenlang-thanos
  - Version: 1.0.0
  - Dependency: bitnami/thanos

- [ ] Create `deployment/helm/thanos/values.yaml`:
  - Query (2 replicas, DNS discovery)
  - Query Frontend (caching enabled)
  - Store Gateway (2 replicas, 50Gi)
  - Compactor (1 replica, 100Gi)
  - Ruler (2 replicas)
  - S3 objstore configuration

- [ ] Create `deployment/helm/thanos/values-prod.yaml`:
  - Production-specific overrides
  - Retention: 30d raw, 120d 5m, 730d 1h

---

## Phase 3: Python PushGateway SDK (P1)

### 3.1 BatchJobMetrics Class
- [ ] Create `greenlang/monitoring/pushgateway.py`:
  - `PushGatewayConfig` dataclass (url, job_name, grouping_key)
  - `BatchJobMetrics` class:
    - `__init__(job_name, pushgateway_url, grouping_key)`
    - Standard metrics: duration, last_success, records_processed, errors
    - `push()` method to send metrics
    - `delete()` method to clear metrics
    - `@track_duration` context manager
    - `record_success()` helper
    - `record_error(error_type)` helper
    - `record_records(count, record_type)` helper
  - `get_pushgateway_client()` singleton factory
  - Retry logic with exponential backoff
  - Timeout handling

### 3.2 Integration with Agent Factory
- [ ] Modify `greenlang/infrastructure/agent_factory/lifecycle/executor.py`:
  - Import BatchJobMetrics
  - Add PushGateway metrics for agent execution jobs
  - Track pack build duration
  - Track deployment duration

### 3.3 Integration with Remediation Jobs
- [ ] Modify `greenlang/infrastructure/pii_service/remediation/jobs.py`:
  - Import BatchJobMetrics
  - Push remediation job metrics
  - Track items processed

---

## Phase 4: ServiceMonitors Consolidation (P1)

### 4.1 Core ServiceMonitors
- [ ] Create `deployment/kubernetes/monitoring/servicemonitors/greenlang-api.yaml`:
  - ServiceMonitor for greenlang-api
  - 15s scrape interval
  - Honor labels
  - Relabel configs for service/pod labels

- [ ] Create `deployment/kubernetes/monitoring/servicemonitors/agent-factory.yaml`:
  - ServiceMonitor for agent-factory
  - Metric relabeling to keep gl_agent_* metrics

- [ ] Create `deployment/kubernetes/monitoring/servicemonitors/auth-service.yaml`:
  - ServiceMonitor for auth endpoints
  - Include /metrics endpoint

### 4.2 Infrastructure ServiceMonitors
- [ ] Create `deployment/kubernetes/monitoring/servicemonitors/postgresql.yaml`:
  - ServiceMonitor for postgres_exporter
  - Include replication metrics

- [ ] Create `deployment/kubernetes/monitoring/servicemonitors/redis.yaml`:
  - ServiceMonitor for redis_exporter
  - Include cluster metrics

- [ ] Create `deployment/kubernetes/monitoring/servicemonitors/kong.yaml`:
  - ServiceMonitor for Kong Prometheus plugin

### 4.3 PodMonitor for All Agents
- [ ] Create `deployment/kubernetes/monitoring/podmonitors/greenlang-pods.yaml`:
  - PodMonitor for prometheus.io/scrape annotation
  - All GreenLang namespaces
  - Honor labels for multi-tenant

### 4.4 Kustomization
- [ ] Create `deployment/kubernetes/monitoring/servicemonitors/kustomization.yaml`:
  - Include all ServiceMonitors
  - Include all PodMonitors
  - Common labels

---

## Phase 5: Alert Rules Consolidation (P1)

### 5.1 Prometheus Health Alerts
- [ ] Create `deployment/monitoring/alerts/prometheus-health-alerts.yaml`:
  - PrometheusTargetMissing (critical)
  - PrometheusConfigReloadFailed (critical)
  - PrometheusTSDBCompactionsFailed (warning)
  - PrometheusRuleEvaluationFailures (warning)
  - PrometheusStorageAlmostFull (warning)
  - PrometheusHighMemoryUsage (warning)
  - PrometheusHighCardinality (warning)
  - PrometheusSlowQueries (warning)

### 5.2 Thanos Health Alerts
- [ ] Create `deployment/monitoring/alerts/thanos-health-alerts.yaml`:
  - ThanosCompactorMultipleRunning (critical)
  - ThanosQueryHighDNSFailures (warning)
  - ThanosStoreGatewayBucketOperationsFailed (warning)
  - ThanosCompactorHalted (critical)
  - ThanosSidecarPrometheusDown (critical)
  - ThanosQueryHighLatency (warning)
  - ThanosRulerHighFailureRate (warning)

### 5.3 Alertmanager Health Alerts
- [ ] Create `deployment/monitoring/alerts/alertmanager-health-alerts.yaml`:
  - AlertmanagerClusterDown (critical)
  - AlertmanagerConfigInconsistent (warning)
  - AlertmanagerNotificationsFailing (critical)
  - AlertmanagerSilenceExpiring (info)

### 5.4 PushGateway Alerts
- [ ] Create `deployment/monitoring/alerts/pushgateway-alerts.yaml`:
  - PushGatewayDown (warning)
  - PushGatewayHighMetricAge (warning)
  - BatchJobFailed (warning)
  - BatchJobStale (warning)

---

## Phase 6: Grafana Dashboards (P1)

### 6.1 Prometheus Health Dashboard
- [ ] Create `deployment/monitoring/dashboards/prometheus-health.json`:
  - Row: Overview
    - Prometheus uptime stat
    - Active targets stat
    - Scrape success rate stat
    - Config reload status stat
  - Row: Performance
    - Scrape duration timeseries (P50/P95/P99)
    - Rule evaluation duration timeseries
    - Query duration timeseries
  - Row: Storage
    - Samples ingested rate
    - TSDB storage size
    - Head series count
    - Compaction status
  - Row: Targets
    - Target status table
    - Scrape errors by job
  - Variables: datasource, cluster

### 6.2 Thanos Health Dashboard
- [ ] Create `deployment/monitoring/dashboards/thanos-health.json`:
  - Row: Overview
    - Query uptime stat
    - Store Gateway uptime stat
    - Compactor status stat
  - Row: Query
    - Query latency (P50/P95/P99)
    - Concurrent queries
    - Store API calls
  - Row: Store Gateway
    - Block operations
    - S3 operations
    - Cache hit ratio
  - Row: Compactor
    - Blocks compacted
    - Blocks deleted
    - Compaction duration
  - Row: S3 Storage
    - Bucket size
    - Upload rate
    - Download rate
  - Variables: datasource, cluster

### 6.3 Alertmanager Dashboard
- [ ] Create `deployment/monitoring/dashboards/alertmanager-health.json`:
  - Active alerts count
  - Alerts by severity
  - Notification success rate
  - Silences active
  - Alert group latency

---

## Phase 7: Operational Runbooks (P2)

### 7.1 Prometheus Runbooks
- [ ] Create `docs/runbooks/prometheus-high-memory.md`:
  - Symptoms
  - Investigation steps (cardinality check, sample rate)
  - Remediation (relabel configs, memory increase, recording rules)

- [ ] Create `docs/runbooks/prometheus-target-down.md`:
  - Symptoms
  - Investigation (target status, network policies, pod health)
  - Remediation

- [ ] Create `docs/runbooks/prometheus-slow-queries.md`:
  - Symptoms
  - Investigation (query complexity, recording rules needed)
  - Remediation

### 7.2 Thanos Runbooks
- [ ] Create `docs/runbooks/thanos-compactor-halted.md`:
  - Symptoms
  - Investigation (logs, S3 permissions, overlapping blocks)
  - Remediation

- [ ] Create `docs/runbooks/thanos-store-gateway-issues.md`:
  - Symptoms
  - Investigation (S3 access, cache, memory)
  - Remediation

### 7.3 General Runbooks
- [ ] Create `docs/runbooks/alertmanager-notifications-failing.md`:
  - Symptoms
  - Investigation (webhook status, config validation)
  - Remediation

- [ ] Create `docs/runbooks/batch-job-metrics-stale.md`:
  - Symptoms
  - Investigation (job status, pushgateway health)
  - Remediation

---

## Phase 8: Environment Integration (P0)

### 8.1 Dev Environment
- [ ] Create `deployment/terraform/environments/dev/prometheus.tf`:
  - Module instantiation with dev values
  - Thanos disabled
  - 1 replica

### 8.2 Staging Environment
- [ ] Create `deployment/terraform/environments/staging/prometheus.tf`:
  - Module instantiation with staging values
  - Thanos enabled
  - 2 replicas

### 8.3 Prod Environment
- [ ] Create `deployment/terraform/environments/prod/prometheus.tf`:
  - Module instantiation with prod values
  - Full Thanos stack
  - 2 replicas
  - PagerDuty integration

---

## Phase 9: Testing (P2)

### 9.1 Unit Tests
- [ ] Create `tests/unit/monitoring/__init__.py`
- [ ] Create `tests/unit/monitoring/conftest.py`:
  - Mock PushGateway fixtures
  - Mock Prometheus client fixtures

- [ ] Create `tests/unit/monitoring/test_pushgateway.py`:
  - Test BatchJobMetrics initialization
  - Test push() method
  - Test delete() method
  - Test track_duration context manager
  - Test retry logic
  - Test error handling

### 9.2 Integration Tests
- [ ] Create `tests/integration/monitoring/__init__.py`
- [ ] Create `tests/integration/monitoring/test_prometheus_scrape.py`:
  - Test ServiceMonitor targets are scraped
  - Test metrics endpoint accessibility
  - Test metric format validation

- [ ] Create `tests/integration/monitoring/test_thanos_upload.py`:
  - Test blocks upload to S3
  - Test Store Gateway queries
  - Test Query federation

- [ ] Create `tests/integration/monitoring/test_alertmanager.py`:
  - Test alert delivery to Slack
  - Test silence API
  - Test inhibition rules

### 9.3 Load Tests
- [ ] Create `tests/load/monitoring/__init__.py`
- [ ] Create `tests/load/monitoring/test_prometheus_load.py`:
  - Test high cardinality (1M series)
  - Test concurrent queries (100 parallel)
  - Test ingest rate (100K samples/sec)

---

## Phase 10: Documentation (P2)

### 10.1 Architecture Documentation
- [ ] Create `docs/architecture/prometheus-stack.md`:
  - Architecture overview
  - Component descriptions
  - Data flow diagrams
  - Retention policies

### 10.2 Operations Guide
- [ ] Create `docs/operations/prometheus-operations.md`:
  - Day-to-day operations
  - Scaling guidance
  - Backup/restore procedures
  - Capacity planning

### 10.3 Developer Guide
- [ ] Create `docs/development/metrics-guide.md`:
  - How to add metrics to services
  - Metric naming conventions
  - ServiceMonitor creation
  - PushGateway SDK usage

---

## Summary

| Phase | Tasks | Priority | Status |
|-------|-------|----------|--------|
| Phase 1: Terraform Module | 11/11 | P0 | **COMPLETE** |
| Phase 2: Helm Values | 8/8 | P0 | **COMPLETE** |
| Phase 3: PushGateway SDK | 3/3 | P1 | **COMPLETE** |
| Phase 4: ServiceMonitors | 8/8 | P1 | **COMPLETE** |
| Phase 5: Alert Rules | 4/4 | P1 | **COMPLETE** |
| Phase 6: Grafana Dashboards | 3/3 | P1 | **COMPLETE** |
| Phase 7: Operational Runbooks | 7/7 | P2 | **COMPLETE** |
| Phase 8: Environment Integration | 3/3 | P0 | **COMPLETE** |
| Phase 9: Testing | 8/8 | P2 | **COMPLETE** |
| Phase 10: Documentation | 3/3 | P2 | **COMPLETE** |
| **TOTAL** | **58/58** | - | **COMPLETE** |

---

## Final Output

| Category | Files | Lines |
|----------|-------|-------|
| Terraform Module | 12 | 2,720 |
| Helm Charts | 8 | 1,575 |
| Python SDK | 2 | 450 |
| K8s Manifests (ServiceMonitors/PodMonitors) | 11 | 800 |
| Alert Rules | 4 | 600 |
| Dashboards | 3 | 3,232 |
| Runbooks | 8 | 1,200 |
| Tests | 8 | 1,500 |
| Documentation | 3 | 900 |
| Modified Files | 3 | +200 |
| **Total** | **~62 new + 3 modified** | **~13,177** |
