# PRD-OBS-001: Prometheus Metrics Collection

**Status:** Approved
**Version:** 1.0
**Created:** 2026-02-06
**Author:** GreenLang Platform Team
**Priority:** P0 - Critical
**Dependencies:** INFRA-001 (K8s), INFRA-002 (PostgreSQL), INFRA-003 (Redis)

---

## 1. Executive Summary

This PRD formalizes and completes the Prometheus metrics collection infrastructure for GreenLang Climate OS. While substantial Prometheus infrastructure exists (~87% complete), this PRD addresses remaining gaps: Thanos for long-term storage and HA, Terraform-based deployment, federation for multi-cluster, PushGateway for batch jobs, and operational tooling.

### Current State Analysis

| Component | Status | Gaps |
|-----------|--------|------|
| Prometheus Server | Deployed | No HA, no long-term storage |
| ServiceMonitors | 15+ defined | Missing some new services |
| PrometheusRules | 100+ alerts | Need consolidation |
| Python Instrumentation | 73+ metrics | Well instrumented |
| Grafana Dashboards | 35+ dashboards | Need Prometheus health dashboard |
| Alertmanager | Configured | Missing advanced routing |
| Long-term Storage | Not deployed | Thanos needed |
| Federation | Not deployed | Multi-cluster support needed |

### Goals

1. **High Availability**: Deploy Prometheus with Thanos for HA and long-term storage
2. **Infrastructure as Code**: Terraform module for Prometheus stack deployment
3. **Federation Ready**: Support multi-cluster metric aggregation
4. **Batch Job Metrics**: PushGateway for short-lived jobs
5. **Operational Excellence**: Health dashboards, runbooks, capacity planning

---

## 2. Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GreenLang Prometheus Stack                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        Prometheus Layer                                │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ Prometheus  │  │ Prometheus  │  │ Thanos      │  │ Thanos      │  │   │
│  │  │ Server 0    │  │ Server 1    │  │ Sidecar 0   │  │ Sidecar 1   │  │   │
│  │  │ (HA Pair)   │  │ (HA Pair)   │  │             │  │             │  │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │   │
│  │         │                │                │                │         │   │
│  └─────────┼────────────────┼────────────────┼────────────────┼─────────┘   │
│            │                │                │                │             │
│  ┌─────────┴────────────────┴────────────────┴────────────────┴─────────┐   │
│  │                         Thanos Layer                                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ Thanos      │  │ Thanos      │  │ Thanos      │  │ Thanos      │  │   │
│  │  │ Query       │  │ Store GW    │  │ Compactor   │  │ Ruler       │  │   │
│  │  └──────┬──────┘  └──────┬──────┘  └─────────────┘  └─────────────┘  │   │
│  │         │                │                                           │   │
│  └─────────┼────────────────┼───────────────────────────────────────────┘   │
│            │                │                                               │
│  ┌─────────┴────────────────┴───────────────────────────────────────────┐   │
│  │                       Storage Layer (S3)                               │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐│   │
│  │  │  gl-thanos-metrics-{env}  (Intelligent Tiering, 2-year retention)││   │
│  │  └──────────────────────────────────────────────────────────────────┘│   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │                      Supporting Services                                │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ Alertmanager│  │ PushGateway │  │ kube-state  │  │ node-       │  │   │
│  │  │ (HA x2)     │  │ (HA x2)     │  │ -metrics    │  │ exporter    │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Overview

| Component | Version | Replicas | Purpose |
|-----------|---------|----------|---------|
| Prometheus | 2.50+ | 2 (HA) | Metrics collection and short-term storage |
| Thanos Sidecar | 0.34+ | 2 | Upload blocks to S3, provide Store API |
| Thanos Query | 0.34+ | 2 | Unified query across stores |
| Thanos Store Gateway | 0.34+ | 2 | Query historical data from S3 |
| Thanos Compactor | 0.34+ | 1 | Compact and downsample blocks |
| Thanos Ruler | 0.34+ | 2 | Evaluate recording/alerting rules |
| Alertmanager | 0.27+ | 2 (HA) | Alert routing and deduplication |
| PushGateway | 1.7+ | 2 (HA) | Metrics from batch jobs |
| kube-state-metrics | 2.10+ | 1 | Kubernetes state metrics |
| node-exporter | 1.7+ | DaemonSet | Node-level metrics |

### 2.3 Data Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Application   │────▶│   Prometheus    │────▶│  Thanos Sidecar │
│   /metrics      │     │   (scrape)      │     │  (upload to S3) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
┌─────────────────┐     ┌─────────────────┐             │
│   Batch Jobs    │────▶│   PushGateway   │             │
│   (push)        │     │                 │             ▼
└─────────────────┘     └─────────────────┘     ┌─────────────────┐
                                                │   S3 Bucket     │
┌─────────────────┐     ┌─────────────────┐     │   (2-year)      │
│   Grafana       │────▶│   Thanos Query  │────▶└─────────────────┘
│   (visualize)   │     │   (unified)     │             ▲
└─────────────────┘     └─────────────────┘             │
                                │               ┌─────────────────┐
                                └──────────────▶│  Thanos Store   │
                                                │  Gateway        │
                                                └─────────────────┘
```

---

## 3. Technical Specifications

### 3.1 Prometheus Server Configuration

```yaml
# High-level configuration
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: greenlang-${environment}
    region: ${aws_region}
    replica: $(POD_NAME)

# Thanos sidecar integration
thanos:
  enabled: true
  objectStorageConfig:
    type: S3
    bucket: gl-thanos-metrics-${environment}
    endpoint: s3.${aws_region}.amazonaws.com

# Retention for local storage (Thanos handles long-term)
retention:
  time: 7d          # Keep 7 days locally
  size: 50GB        # Max local storage

# Resource limits
resources:
  requests:
    cpu: 500m
    memory: 2Gi
  limits:
    cpu: 2000m
    memory: 8Gi
```

### 3.2 Thanos Configuration

```yaml
# Thanos Query
query:
  replicaCount: 2
  dnsDiscovery:
    enabled: true
    sidecarsService: prometheus-thanos-sidecar
    sidecarsNamespace: monitoring
  stores:
    - dnssrv+_grpc._tcp.thanos-store-gateway.monitoring.svc

# Thanos Store Gateway
storegateway:
  replicaCount: 2
  persistence:
    enabled: true
    size: 50Gi
  resources:
    requests:
      cpu: 250m
      memory: 1Gi
    limits:
      cpu: 1000m
      memory: 4Gi

# Thanos Compactor
compactor:
  replicaCount: 1
  retentionResolutionRaw: 30d
  retentionResolution5m: 120d
  retentionResolution1h: 730d  # 2 years
  persistence:
    enabled: true
    size: 100Gi

# Thanos Ruler
ruler:
  replicaCount: 2
  alertmanagers:
    - http://alertmanager.monitoring.svc:9093
```

### 3.3 S3 Storage Configuration

```hcl
# Thanos metrics bucket
resource "aws_s3_bucket" "thanos_metrics" {
  bucket = "gl-thanos-metrics-${var.environment}"

  tags = {
    Name        = "GreenLang Thanos Metrics"
    Environment = var.environment
    Component   = "observability"
    DataClass   = "operational"
    Retention   = "2-years"
  }
}

# Lifecycle rules for cost optimization
resource "aws_s3_bucket_lifecycle_configuration" "thanos_lifecycle" {
  bucket = aws_s3_bucket.thanos_metrics.id

  rule {
    id     = "intelligent-tiering"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "INTELLIGENT_TIERING"
    }
  }

  rule {
    id     = "delete-old-blocks"
    status = "Enabled"

    expiration {
      days = 730  # 2 years
    }
  }
}
```

### 3.4 PushGateway for Batch Jobs

```yaml
# PushGateway configuration
pushgateway:
  enabled: true
  replicaCount: 2

  persistence:
    enabled: true
    size: 2Gi

  service:
    type: ClusterIP
    port: 9091

  # Security
  podSecurityContext:
    runAsNonRoot: true
    runAsUser: 65534
    fsGroup: 65534
```

### 3.5 Python SDK for Batch Jobs

```python
# greenlang/monitoring/pushgateway.py
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram
from prometheus_client import push_to_gateway, delete_from_gateway

class BatchJobMetrics:
    """Push metrics from batch jobs to PushGateway."""

    def __init__(
        self,
        job_name: str,
        pushgateway_url: str = "http://pushgateway.monitoring.svc:9091",
        grouping_key: dict[str, str] | None = None,
    ):
        self.job_name = job_name
        self.pushgateway_url = pushgateway_url
        self.grouping_key = grouping_key or {}
        self.registry = CollectorRegistry()

        # Standard batch job metrics
        self.job_duration = Gauge(
            "gl_batch_job_duration_seconds",
            "Duration of batch job execution",
            ["job_name", "status"],
            registry=self.registry,
        )
        self.job_last_success = Gauge(
            "gl_batch_job_last_success_timestamp",
            "Timestamp of last successful job run",
            ["job_name"],
            registry=self.registry,
        )
        self.job_records_processed = Counter(
            "gl_batch_job_records_processed_total",
            "Total records processed by batch job",
            ["job_name", "record_type"],
            registry=self.registry,
        )
        self.job_errors = Counter(
            "gl_batch_job_errors_total",
            "Total errors in batch job",
            ["job_name", "error_type"],
            registry=self.registry,
        )

    def push(self) -> None:
        """Push metrics to PushGateway."""
        push_to_gateway(
            self.pushgateway_url,
            job=self.job_name,
            registry=self.registry,
            grouping_key=self.grouping_key,
        )

    def delete(self) -> None:
        """Delete metrics from PushGateway (on job completion)."""
        delete_from_gateway(
            self.pushgateway_url,
            job=self.job_name,
            grouping_key=self.grouping_key,
        )

    @contextmanager
    def track_duration(self, status: str = "success"):
        """Context manager to track job duration."""
        start = time.time()
        try:
            yield
            self.job_duration.labels(job_name=self.job_name, status=status).set(
                time.time() - start
            )
            self.job_last_success.labels(job_name=self.job_name).set_to_current_time()
        except Exception as e:
            self.job_duration.labels(job_name=self.job_name, status="error").set(
                time.time() - start
            )
            self.job_errors.labels(
                job_name=self.job_name, error_type=type(e).__name__
            ).inc()
            raise
        finally:
            self.push()
```

---

## 4. Functional Requirements

### 4.1 Prometheus Deployment (P0)

| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| FR-1.1 | Deploy HA Prometheus | 2 replicas with pod anti-affinity |
| FR-1.2 | External labels | Cluster, region, replica labels set |
| FR-1.3 | Service discovery | All GreenLang namespaces discovered |
| FR-1.4 | Scrape targets | 20+ job configurations |
| FR-1.5 | Local retention | 7 days / 50GB |

### 4.2 Thanos Long-term Storage (P0)

| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| FR-2.1 | Thanos Sidecar | Uploads blocks to S3 every 2 hours |
| FR-2.2 | Thanos Query | Unified queries across all stores |
| FR-2.3 | Store Gateway | Queries historical data from S3 |
| FR-2.4 | Compactor | Downsamples to 5m/1h resolutions |
| FR-2.5 | Retention | 2-year retention in S3 |

### 4.3 Alertmanager (P0)

| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| FR-3.1 | HA deployment | 2 replicas with mesh |
| FR-3.2 | Slack integration | Critical alerts to #greenlang-alerts |
| FR-3.3 | PagerDuty integration | P1 alerts trigger pages |
| FR-3.4 | Email integration | Weekly summary reports |
| FR-3.5 | Silences | API for silence management |

### 4.4 PushGateway (P1)

| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| FR-4.1 | HA deployment | 2 replicas |
| FR-4.2 | Python SDK | BatchJobMetrics class |
| FR-4.3 | Job isolation | Grouping keys supported |
| FR-4.4 | Stale cleanup | Metrics expire after 1 hour |

### 4.5 ServiceMonitors (P1)

| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| FR-5.1 | Auto-discovery | All pods with prometheus.io annotations |
| FR-5.2 | ServiceMonitors | All services have dedicated monitors |
| FR-5.3 | PodMonitors | Agent pods monitored |
| FR-5.4 | Custom metrics | GreenLang-specific exporters |

### 4.6 Prometheus Health (P1)

| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| FR-6.1 | Self-monitoring | Prometheus scrapes itself |
| FR-6.2 | Health dashboard | Prometheus health in Grafana |
| FR-6.3 | Capacity alerts | Storage/memory alerts |
| FR-6.4 | Target alerts | Scrape failure alerts |

---

## 5. Non-Functional Requirements

### 5.1 Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| Scrape latency | < 5s P99 | prometheus_target_scrape_pool_sync_total |
| Query latency | < 30s P99 | prometheus_engine_query_duration_seconds |
| Ingestion rate | > 100K samples/s | prometheus_tsdb_head_samples_appended_total |
| Storage growth | < 10GB/day | prometheus_tsdb_storage_blocks_bytes |

### 5.2 Availability

| Component | SLO | Measurement |
|-----------|-----|-------------|
| Prometheus | 99.9% | up{job="prometheus"} |
| Thanos Query | 99.9% | up{job="thanos-query"} |
| Alertmanager | 99.9% | up{job="alertmanager"} |
| PushGateway | 99.5% | up{job="pushgateway"} |

### 5.3 Data Retention

| Resolution | Retention | Storage |
|------------|-----------|---------|
| Raw (15s) | 30 days | S3 |
| 5-minute | 120 days | S3 |
| 1-hour | 2 years | S3 |
| Local | 7 days | PVC |

### 5.4 Security

- **Network Policies**: Only monitoring namespace can access Prometheus
- **RBAC**: Dedicated ServiceAccounts with minimal permissions
- **TLS**: All inter-component communication encrypted
- **Authentication**: Grafana auth for UI access
- **Secrets**: S3 credentials via IRSA (no static keys)

---

## 6. Terraform Module Specification

### 6.1 Module Structure

```
deployment/terraform/modules/prometheus-stack/
├── main.tf                 # Main resources
├── variables.tf            # Input variables
├── outputs.tf              # Output values
├── versions.tf             # Provider versions
├── prometheus.tf           # Prometheus Helm release
├── thanos.tf               # Thanos components
├── alertmanager.tf         # Alertmanager configuration
├── pushgateway.tf          # PushGateway deployment
├── s3.tf                   # S3 bucket for Thanos
├── iam.tf                  # IAM roles for IRSA
├── servicemonitors.tf      # ServiceMonitor CRDs
└── README.md               # Documentation
```

### 6.2 Input Variables

```hcl
variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
}

variable "prometheus_replica_count" {
  description = "Number of Prometheus replicas"
  type        = number
  default     = 2
}

variable "prometheus_retention_days" {
  description = "Local retention in days"
  type        = number
  default     = 7
}

variable "prometheus_storage_size" {
  description = "PVC size for Prometheus"
  type        = string
  default     = "50Gi"
}

variable "thanos_retention_raw" {
  description = "Raw metrics retention in S3"
  type        = string
  default     = "30d"
}

variable "thanos_retention_5m" {
  description = "5m downsampled retention"
  type        = string
  default     = "120d"
}

variable "thanos_retention_1h" {
  description = "1h downsampled retention"
  type        = string
  default     = "730d"
}

variable "alertmanager_slack_webhook" {
  description = "Slack webhook URL for alerts"
  type        = string
  sensitive   = true
}

variable "alertmanager_pagerduty_key" {
  description = "PagerDuty integration key"
  type        = string
  sensitive   = true
}

variable "enable_pushgateway" {
  description = "Enable PushGateway for batch jobs"
  type        = bool
  default     = true
}

variable "enable_thanos" {
  description = "Enable Thanos for long-term storage"
  type        = bool
  default     = true
}

variable "additional_scrape_configs" {
  description = "Additional Prometheus scrape configurations"
  type        = list(any)
  default     = []
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
```

### 6.3 Output Values

```hcl
output "prometheus_endpoint" {
  description = "Prometheus server endpoint"
  value       = "http://prometheus-server.monitoring.svc:9090"
}

output "thanos_query_endpoint" {
  description = "Thanos Query endpoint"
  value       = var.enable_thanos ? "http://thanos-query.monitoring.svc:9090" : null
}

output "alertmanager_endpoint" {
  description = "Alertmanager endpoint"
  value       = "http://alertmanager.monitoring.svc:9093"
}

output "pushgateway_endpoint" {
  description = "PushGateway endpoint"
  value       = var.enable_pushgateway ? "http://pushgateway.monitoring.svc:9091" : null
}

output "thanos_bucket_name" {
  description = "S3 bucket for Thanos metrics"
  value       = var.enable_thanos ? aws_s3_bucket.thanos_metrics[0].id : null
}

output "thanos_bucket_arn" {
  description = "S3 bucket ARN for Thanos metrics"
  value       = var.enable_thanos ? aws_s3_bucket.thanos_metrics[0].arn : null
}
```

---

## 7. Helm Values Specification

### 7.1 kube-prometheus-stack values

```yaml
# deployment/helm/prometheus-stack/values.yaml

## Prometheus
prometheus:
  enabled: true

  prometheusSpec:
    replicas: 2

    # Thanos sidecar
    thanos:
      image: quay.io/thanos/thanos:v0.34.1
      objectStorageConfig:
        existingSecret:
          name: thanos-objstore-secret
          key: objstore.yml

    # External labels
    externalLabels:
      cluster: greenlang-${ENVIRONMENT}
      region: ${AWS_REGION}

    # Retention
    retention: 7d
    retentionSize: 50GB

    # Storage
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: gp3
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 50Gi

    # Resources
    resources:
      requests:
        cpu: 500m
        memory: 2Gi
      limits:
        cpu: 2000m
        memory: 8Gi

    # Pod anti-affinity for HA
    affinity:
      podAntiAffinity:
        requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchLabels:
                app.kubernetes.io/name: prometheus
            topologyKey: kubernetes.io/hostname

    # Service discovery
    serviceMonitorSelectorNilUsesHelmValues: false
    podMonitorSelectorNilUsesHelmValues: false
    ruleSelectorNilUsesHelmValues: false

    # Additional scrape configs
    additionalScrapeConfigs:
      - job_name: 'greenlang-agents'
        kubernetes_sd_configs:
          - role: pod
            namespaces:
              names:
                - greenlang
                - gl-agents
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true

## Alertmanager
alertmanager:
  enabled: true

  alertmanagerSpec:
    replicas: 2

    storage:
      volumeClaimTemplate:
        spec:
          storageClassName: gp3
          resources:
            requests:
              storage: 10Gi

    resources:
      requests:
        cpu: 100m
        memory: 256Mi
      limits:
        cpu: 500m
        memory: 512Mi

  config:
    global:
      resolve_timeout: 5m
      slack_api_url_file: /etc/alertmanager/secrets/slack-webhook

    route:
      group_by: ['alertname', 'cluster', 'service']
      group_wait: 30s
      group_interval: 5m
      repeat_interval: 4h
      receiver: 'default'
      routes:
        - match:
            severity: critical
          receiver: 'pagerduty-critical'
        - match:
            severity: warning
          receiver: 'slack-warnings'

    receivers:
      - name: 'default'
        slack_configs:
          - channel: '#greenlang-alerts'
            send_resolved: true

      - name: 'pagerduty-critical'
        pagerduty_configs:
          - service_key_file: /etc/alertmanager/secrets/pagerduty-key
            severity: critical

      - name: 'slack-warnings'
        slack_configs:
          - channel: '#greenlang-alerts-warning'
            send_resolved: true

## Grafana
grafana:
  enabled: true

  adminPassword: # Set via secret

  persistence:
    enabled: true
    size: 10Gi

  datasources:
    datasources.yaml:
      apiVersion: 1
      datasources:
        - name: Prometheus
          type: prometheus
          url: http://prometheus-server.monitoring.svc:9090
          isDefault: true
        - name: Thanos
          type: prometheus
          url: http://thanos-query.monitoring.svc:9090
        - name: Alertmanager
          type: alertmanager
          url: http://alertmanager.monitoring.svc:9093

## kube-state-metrics
kubeStateMetrics:
  enabled: true

## node-exporter
nodeExporter:
  enabled: true

## Prometheus Operator
prometheusOperator:
  enabled: true

  admissionWebhooks:
    enabled: true
    patch:
      enabled: true
```

### 7.2 Thanos values

```yaml
# deployment/helm/thanos/values.yaml

## Global
image:
  registry: quay.io
  repository: thanos/thanos
  tag: v0.34.1

objstoreConfig: |-
  type: S3
  config:
    bucket: gl-thanos-metrics-${ENVIRONMENT}
    endpoint: s3.${AWS_REGION}.amazonaws.com
    region: ${AWS_REGION}

## Query
query:
  enabled: true
  replicaCount: 2

  dnsDiscovery:
    enabled: true
    sidecarsService: prometheus-thanos-discovery
    sidecarsNamespace: monitoring

  stores:
    - dnssrv+_grpc._tcp.thanos-storegateway.monitoring.svc.cluster.local

  resources:
    requests:
      cpu: 250m
      memory: 512Mi
    limits:
      cpu: 1000m
      memory: 2Gi

## Query Frontend (optional caching layer)
queryFrontend:
  enabled: true
  replicaCount: 2

  config: |-
    type: IN-MEMORY
    config:
      max_size: 512MB
      max_size_items: 1000
      validity: 5m

## Store Gateway
storegateway:
  enabled: true
  replicaCount: 2

  persistence:
    enabled: true
    size: 50Gi
    storageClass: gp3

  resources:
    requests:
      cpu: 250m
      memory: 1Gi
    limits:
      cpu: 1000m
      memory: 4Gi

## Compactor
compactor:
  enabled: true

  retentionResolutionRaw: 30d
  retentionResolution5m: 120d
  retentionResolution1h: 730d

  persistence:
    enabled: true
    size: 100Gi
    storageClass: gp3

  resources:
    requests:
      cpu: 250m
      memory: 1Gi
    limits:
      cpu: 1000m
      memory: 4Gi

## Ruler
ruler:
  enabled: true
  replicaCount: 2

  alertmanagers:
    - http://alertmanager.monitoring.svc:9093

  config: |-
    groups: []  # Rules loaded from ConfigMap

  persistence:
    enabled: true
    size: 10Gi
```

---

## 8. ServiceMonitor Specifications

### 8.1 GreenLang API ServiceMonitor

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: greenlang-api
  namespace: monitoring
  labels:
    release: prometheus
spec:
  selector:
    matchLabels:
      app: greenlang-api
  namespaceSelector:
    matchNames:
      - greenlang
  endpoints:
    - port: metrics
      interval: 15s
      path: /metrics
      scheme: http
      honorLabels: true
      relabelings:
        - sourceLabels: [__meta_kubernetes_pod_label_app]
          targetLabel: service
        - sourceLabels: [__meta_kubernetes_pod_name]
          targetLabel: pod
```

### 8.2 Agent Factory ServiceMonitor

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: agent-factory
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app.kubernetes.io/component: agent-factory
  namespaceSelector:
    matchNames:
      - greenlang
  endpoints:
    - port: metrics
      interval: 15s
      path: /metrics
      metricRelabelings:
        - sourceLabels: [__name__]
          regex: 'gl_agent_.*'
          action: keep
```

### 8.3 PodMonitor for All Agents

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: greenlang-pods
  namespace: monitoring
spec:
  selector:
    matchLabels:
      prometheus.io/scrape: "true"
  namespaceSelector:
    matchNames:
      - greenlang
      - gl-agents
      - gl-fuel
      - gl-cbam
      - gl-building
  podMetricsEndpoints:
    - port: metrics
      interval: 15s
      path: /metrics
```

---

## 9. Alert Rules Consolidation

### 9.1 Prometheus Health Alerts

```yaml
groups:
  - name: prometheus.rules
    rules:
      - alert: PrometheusTargetMissing
        expr: up == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Prometheus target missing (instance {{ $labels.instance }})"
          description: "A Prometheus target is down for more than 5 minutes."

      - alert: PrometheusConfigReloadFailed
        expr: prometheus_config_last_reload_successful != 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Prometheus config reload failed"

      - alert: PrometheusTSDBCompactionsFailed
        expr: increase(prometheus_tsdb_compactions_failed_total[1h]) > 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Prometheus TSDB compactions failing"

      - alert: PrometheusRuleEvaluationFailures
        expr: increase(prometheus_rule_evaluation_failures_total[5m]) > 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Prometheus rule evaluation failures"

      - alert: PrometheusStorageAlmostFull
        expr: |
          (prometheus_tsdb_storage_blocks_bytes /
           (prometheus_tsdb_storage_blocks_bytes + prometheus_tsdb_head_chunks_storage_size_bytes)) > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Prometheus storage almost full (>80%)"

      - alert: PrometheusHighMemoryUsage
        expr: |
          process_resident_memory_bytes{job="prometheus"} /
          container_spec_memory_limit_bytes{container="prometheus"} > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Prometheus memory usage >80%"
```

### 9.2 Thanos Health Alerts

```yaml
groups:
  - name: thanos.rules
    rules:
      - alert: ThanosCompactorMultipleRunning
        expr: sum(up{job=~".*thanos-compactor.*"}) > 1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Multiple Thanos Compactors running"

      - alert: ThanosQueryHighDNSFailures
        expr: |
          rate(thanos_query_store_apis_dns_failures_total[5m]) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Thanos Query DNS failures"

      - alert: ThanosStoreGatewayBucketOperationsFailed
        expr: |
          rate(thanos_objstore_bucket_operation_failures_total[5m]) > 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Thanos Store Gateway bucket operations failing"

      - alert: ThanosCompactorHalted
        expr: thanos_compact_halted == 1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Thanos Compactor has halted"

      - alert: ThanosSidecarPrometheusDown
        expr: thanos_sidecar_prometheus_up != 1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Thanos Sidecar cannot reach Prometheus"
```

---

## 10. Grafana Dashboard Specifications

### 10.1 Prometheus Health Dashboard

```json
{
  "title": "Prometheus & Thanos Health",
  "uid": "prometheus-health",
  "panels": [
    {
      "title": "Prometheus Uptime",
      "type": "stat",
      "targets": [
        { "expr": "up{job=\"prometheus\"}" }
      ]
    },
    {
      "title": "Active Targets",
      "type": "stat",
      "targets": [
        { "expr": "count(up)" }
      ]
    },
    {
      "title": "Scrape Duration",
      "type": "timeseries",
      "targets": [
        { "expr": "histogram_quantile(0.99, rate(prometheus_target_scrape_pool_sync_total[5m]))" }
      ]
    },
    {
      "title": "Samples Ingested",
      "type": "timeseries",
      "targets": [
        { "expr": "rate(prometheus_tsdb_head_samples_appended_total[5m])" }
      ]
    },
    {
      "title": "Storage Size",
      "type": "timeseries",
      "targets": [
        { "expr": "prometheus_tsdb_storage_blocks_bytes" }
      ]
    },
    {
      "title": "Thanos Query Latency",
      "type": "timeseries",
      "targets": [
        { "expr": "histogram_quantile(0.99, rate(thanos_query_duration_seconds_bucket[5m]))" }
      ]
    },
    {
      "title": "Thanos Store Gateway",
      "type": "stat",
      "targets": [
        { "expr": "up{job=\"thanos-storegateway\"}" }
      ]
    },
    {
      "title": "S3 Upload Rate",
      "type": "timeseries",
      "targets": [
        { "expr": "rate(thanos_shipper_uploads_total[5m])" }
      ]
    }
  ]
}
```

---

## 11. Operational Runbooks

### 11.1 Prometheus High Memory

**Symptoms**: PrometheusHighMemoryUsage alert firing

**Investigation**:
1. Check cardinality: `prometheus_tsdb_head_series`
2. Check sample rate: `rate(prometheus_tsdb_head_samples_appended_total[5m])`
3. Identify high-cardinality metrics: `topk(10, count by (__name__)({__name__=~".+"}))`

**Remediation**:
1. Drop high-cardinality labels with relabel_configs
2. Increase memory limits
3. Reduce retention
4. Add recording rules to pre-aggregate

### 11.2 Thanos Compactor Halted

**Symptoms**: ThanosCompactorHalted alert firing

**Investigation**:
1. Check compactor logs: `kubectl logs -n monitoring deployment/thanos-compactor`
2. Check S3 bucket permissions
3. Check for overlapping blocks in S3

**Remediation**:
1. Fix S3 permissions if needed
2. Delete overlapping blocks manually
3. Restart compactor with `--wait` flag first run

### 11.3 Target Down

**Symptoms**: PrometheusTargetMissing alert firing

**Investigation**:
1. Check target status in Prometheus UI
2. Check network policies
3. Check service/pod health

**Remediation**:
1. Fix network policy if blocking
2. Restart unhealthy pods
3. Update ServiceMonitor if endpoint changed

---

## 12. Testing Requirements

### 12.1 Unit Tests

| Test | Description |
|------|-------------|
| PushGateway SDK | Test BatchJobMetrics class methods |
| Metric registry | Test custom collector registration |
| Alert rules | Test PromQL expressions |

### 12.2 Integration Tests

| Test | Description |
|------|-------------|
| Scrape targets | All ServiceMonitors return metrics |
| Thanos upload | Blocks uploaded to S3 within 2h |
| Alert delivery | Test alerts reach Slack/PagerDuty |
| Query federation | Thanos Query returns unified results |

### 12.3 Load Tests

| Test | Description |
|------|-------------|
| High cardinality | 1M series sustained |
| Query load | 100 concurrent queries |
| Ingest rate | 100K samples/sec |

---

## 13. Migration Plan

### 13.1 Phase 1: Thanos Deployment (Week 1)

1. Deploy Thanos Sidecar alongside existing Prometheus
2. Create S3 bucket with lifecycle rules
3. Configure IRSA for S3 access
4. Verify blocks uploading to S3

### 13.2 Phase 2: Thanos Components (Week 2)

1. Deploy Thanos Query
2. Deploy Thanos Store Gateway
3. Deploy Thanos Compactor
4. Update Grafana datasources

### 13.3 Phase 3: PushGateway & SDK (Week 3)

1. Deploy PushGateway
2. Release Python SDK
3. Migrate batch jobs to use SDK

### 13.4 Phase 4: Consolidation (Week 4)

1. Consolidate alert rules
2. Deploy health dashboards
3. Create operational runbooks
4. Training for operations team

---

## 14. Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Prometheus uptime | 99.9% | 99.5% |
| Query latency P99 | < 30s | ~45s |
| Data retention | 2 years | 30 days |
| Active targets | 100% scraped | 95% |
| Alert delivery | < 5min | ~10min |
| Storage cost | -30% | baseline |

---

## 15. Appendix

### A. Existing Files to Modify

| File | Changes |
|------|---------|
| `deployment/kubernetes/monitoring/prometheus-values.yaml` | Add Thanos sidecar config |
| `deployment/infrastructure/monitoring/prometheus/prometheus.yml` | Update for HA |
| `deployment/infrastructure/monitoring/helm/Chart.yaml` | Add Thanos dependency |

### B. New Files to Create

| File | Purpose |
|------|---------|
| `deployment/terraform/modules/prometheus-stack/` | Terraform module |
| `deployment/helm/thanos/` | Thanos Helm values |
| `greenlang/monitoring/pushgateway.py` | Python SDK |
| `deployment/monitoring/dashboards/prometheus-health.json` | Health dashboard |
| `deployment/monitoring/alerts/prometheus-alerts.yaml` | Consolidated alerts |
| `docs/runbooks/prometheus-*.md` | Operational runbooks |

### C. References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Thanos Documentation](https://thanos.io/tip/thanos/getting-started.md/)
- [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack)
- [Thanos Helm Chart](https://github.com/bitnami/charts/tree/main/bitnami/thanos)
