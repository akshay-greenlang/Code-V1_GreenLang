# GreenLang Monitoring Stack Setup Guide

**Version:** 1.0
**Last Updated:** 2025-11-07
**Document Classification:** Operations Reference
**Review Cycle:** Quarterly

---

## Executive Summary

This guide provides step-by-step procedures for setting up the complete monitoring stack for GreenLang production environment. The stack includes Prometheus (metrics), Grafana (dashboards), Loki (logs), Jaeger (tracing), and Alertmanager (alerting).

**Monitoring Stack Components:**
- **Prometheus:** Metrics collection and storage
- **Grafana:** Visualization and dashboards
- **Loki:** Log aggregation and query
- **Jaeger:** Distributed tracing
- **Alertmanager:** Alert routing and notification
- **Node Exporter:** System metrics
- **Blackbox Exporter:** Endpoint monitoring

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                    Monitoring Stack                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │  GreenLang   │─────▶│ Prometheus   │                    │
│  │  Application │      │ (Metrics)    │                    │
│  └──────────────┘      └──────┬───────┘                    │
│                                │                             │
│                         ┌──────▼───────┐                    │
│                         │   Grafana    │                    │
│                         │ (Dashboards) │                    │
│                         └──────────────┘                    │
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │  GreenLang   │─────▶│    Loki      │                    │
│  │  Application │      │   (Logs)     │                    │
│  └──────────────┘      └──────────────┘                    │
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │  GreenLang   │─────▶│   Jaeger     │                    │
│  │  Application │      │  (Tracing)   │                    │
│  └──────────────┘      └──────────────┘                    │
│                                                              │
│                        ┌──────────────┐                     │
│                        │AlertManager  │                     │
│                        │(Alerts)      │                     │
│                        └──────────────┘                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Prometheus Setup](#prometheus-setup)
3. [Grafana Setup](#grafana-setup)
4. [Loki Setup](#loki-setup)
5. [Jaeger Setup](#jaeger-setup)
6. [Alertmanager Setup](#alertmanager-setup)
7. [Dashboard Configuration](#dashboard-configuration)
8. [Alert Rules](#alert-rules)

---

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores
- RAM: 8 GB
- Disk: 100 GB SSD
- Network: 1 Gbps

**Recommended for Production:**
- CPU: 8 cores
- RAM: 16 GB
- Disk: 500 GB SSD (with growth capacity)
- Network: 10 Gbps

### Software Requirements

```bash
# Kubernetes cluster (version 1.24+)
kubectl version

# Helm 3
helm version

# Storage provisioner configured
kubectl get storageclass
```

### Network Requirements

**Ports to Open:**
- 9090: Prometheus
- 3000: Grafana
- 3100: Loki
- 16686: Jaeger UI
- 14250: Jaeger gRPC
- 9093: Alertmanager

---

## Prometheus Setup

### Step 1: Install Prometheus via Helm

```bash
# Add Prometheus community Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Create namespace
kubectl create namespace monitoring

# Install Prometheus
helm install prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --values prometheus-values.yaml
```

### Step 2: Configure Prometheus

Create `prometheus-values.yaml`:

```yaml
# prometheus-values.yaml
server:
  global:
    scrape_interval: 30s
    scrape_timeout: 10s
    evaluation_interval: 30s

  retention: "30d"

  persistentVolume:
    enabled: true
    size: 100Gi
    storageClass: "gp3-ssd"

  resources:
    requests:
      cpu: 500m
      memory: 2Gi
    limits:
      cpu: 2000m
      memory: 4Gi

  # Prometheus configuration
  serverFiles:
    prometheus.yml:
      scrape_configs:
        # GreenLang API
        - job_name: 'greenlang-api'
          kubernetes_sd_configs:
            - role: pod
              namespaces:
                names:
                  - greenlang
          relabel_configs:
            - source_labels: [__meta_kubernetes_pod_label_app]
              action: keep
              regex: greenlang-api
            - source_labels: [__meta_kubernetes_pod_ip]
              target_label: __address__
              replacement: $1:8000

        # GreenLang Workers
        - job_name: 'greenlang-worker'
          kubernetes_sd_configs:
            - role: pod
              namespaces:
                names:
                  - greenlang
          relabel_configs:
            - source_labels: [__meta_kubernetes_pod_label_app]
              action: keep
              regex: greenlang-worker

        # Node Exporter
        - job_name: 'node-exporter'
          kubernetes_sd_configs:
            - role: node
          relabel_configs:
            - source_labels: [__address__]
              regex: '(.*):10250'
              replacement: '${1}:9100'
              target_label: __address__

        # PostgreSQL
        - job_name: 'postgresql'
          static_configs:
            - targets: ['postgres-exporter:9187']

        # Redis
        - job_name: 'redis'
          static_configs:
            - targets: ['redis-exporter:9121']

alertmanager:
  enabled: true
  persistentVolume:
    enabled: true
    size: 10Gi

nodeExporter:
  enabled: true

pushgateway:
  enabled: false
```

### Step 3: Verify Prometheus Installation

```bash
# Check Prometheus pods
kubectl get pods -n monitoring | grep prometheus

# Port-forward to access Prometheus UI
kubectl port-forward -n monitoring svc/prometheus-server 9090:80

# Open browser: http://localhost:9090

# Verify targets are being scraped
# Navigate to Status > Targets
# All targets should show as "UP"
```

### Step 4: Install Exporters

**PostgreSQL Exporter:**
```bash
# Create deployment
kubectl apply -f - << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres-exporter
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres-exporter
  template:
    metadata:
      labels:
        app: postgres-exporter
    spec:
      containers:
      - name: postgres-exporter
        image: prometheuscommunity/postgres-exporter:v0.12.0
        env:
        - name: DATA_SOURCE_NAME
          value: "postgresql://exporter:password@db.greenlang.io:5432/greenlang?sslmode=disable"
        ports:
        - containerPort: 9187
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-exporter
  namespace: monitoring
spec:
  selector:
    app: postgres-exporter
  ports:
  - port: 9187
    targetPort: 9187
EOF
```

**Redis Exporter:**
```bash
kubectl apply -f - << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis-exporter
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis-exporter
  template:
    metadata:
      labels:
        app: redis-exporter
    spec:
      containers:
      - name: redis-exporter
        image: oliver006/redis_exporter:v1.45.0
        env:
        - name: REDIS_ADDR
          value: "redis.greenlang.svc.cluster.local:6379"
        ports:
        - containerPort: 9121
---
apiVersion: v1
kind: Service
metadata:
  name: redis-exporter
  namespace: monitoring
spec:
  selector:
    app: redis-exporter
  ports:
  - port: 9121
    targetPort: 9121
EOF
```

---

## Grafana Setup

### Step 1: Install Grafana via Helm

```bash
# Add Grafana Helm repository
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Install Grafana
helm install grafana grafana/grafana \
  --namespace monitoring \
  --values grafana-values.yaml
```

### Step 2: Configure Grafana

Create `grafana-values.yaml`:

```yaml
# grafana-values.yaml
persistence:
  enabled: true
  size: 20Gi
  storageClassName: "gp3-ssd"

resources:
  requests:
    cpu: 250m
    memory: 512Mi
  limits:
    cpu: 1000m
    memory: 1Gi

adminPassword: "ChangeMe123!"  # Change this!

datasources:
  datasources.yaml:
    apiVersion: 1
    datasources:
      # Prometheus
      - name: Prometheus
        type: prometheus
        access: proxy
        url: http://prometheus-server
        isDefault: true
        editable: false

      # Loki
      - name: Loki
        type: loki
        access: proxy
        url: http://loki:3100
        editable: false

      # Jaeger
      - name: Jaeger
        type: jaeger
        access: proxy
        url: http://jaeger-query:16686
        editable: false

dashboardProviders:
  dashboardproviders.yaml:
    apiVersion: 1
    providers:
      - name: 'default'
        orgId: 1
        folder: ''
        type: file
        disableDeletion: false
        editable: true
        options:
          path: /var/lib/grafana/dashboards/default

ingress:
  enabled: true
  hosts:
    - grafana.greenlang.io
  tls:
    - secretName: grafana-tls
      hosts:
        - grafana.greenlang.io
```

### Step 3: Access Grafana

```bash
# Get Grafana admin password
kubectl get secret --namespace monitoring grafana -o jsonpath="{.data.admin-password}" | base64 --decode

# Port-forward to access Grafana
kubectl port-forward -n monitoring svc/grafana 3000:80

# Open browser: http://localhost:3000
# Login: admin / <password from above>
```

### Step 4: Configure Data Sources

1. Log in to Grafana
2. Go to Configuration > Data Sources
3. Verify Prometheus, Loki, and Jaeger are configured
4. Click "Test" on each data source
5. All should show "Data source is working"

---

## Loki Setup

### Step 1: Install Loki via Helm

```bash
# Install Loki
helm install loki grafana/loki-stack \
  --namespace monitoring \
  --values loki-values.yaml
```

### Step 2: Configure Loki

Create `loki-values.yaml`:

```yaml
# loki-values.yaml
loki:
  enabled: true
  persistence:
    enabled: true
    size: 50Gi
    storageClassName: "gp3-ssd"

  config:
    auth_enabled: false

    ingester:
      chunk_idle_period: 3m
      chunk_block_size: 262144
      chunk_retain_period: 1m
      max_transfer_retries: 0
      lifecycler:
        ring:
          kvstore:
            store: inmemory
          replication_factor: 1

    limits_config:
      enforce_metric_name: false
      reject_old_samples: true
      reject_old_samples_max_age: 168h
      max_entries_limit_per_query: 5000
      max_query_length: 0h

    schema_config:
      configs:
        - from: 2023-01-01
          store: boltdb-shipper
          object_store: filesystem
          schema: v11
          index:
            prefix: index_
            period: 24h

    server:
      http_listen_port: 3100

    storage_config:
      boltdb_shipper:
        active_index_directory: /data/loki/boltdb-shipper-active
        cache_location: /data/loki/boltdb-shipper-cache
        cache_ttl: 24h
        shared_store: filesystem
      filesystem:
        directory: /data/loki/chunks

    chunk_store_config:
      max_look_back_period: 0s

    table_manager:
      retention_deletes_enabled: true
      retention_period: 720h  # 30 days

promtail:
  enabled: true
  config:
    clients:
      - url: http://loki:3100/loki/api/v1/push

    scrape_configs:
      # Pods
      - job_name: kubernetes-pods
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels:
              - __meta_kubernetes_pod_label_app
            target_label: app
          - source_labels:
              - __meta_kubernetes_pod_name
            target_label: pod
          - source_labels:
              - __meta_kubernetes_namespace
            target_label: namespace
        pipeline_stages:
          - json:
              expressions:
                level: level
                message: message
                timestamp: timestamp
          - timestamp:
              source: timestamp
              format: RFC3339
```

### Step 3: Verify Loki Installation

```bash
# Check Loki pods
kubectl get pods -n monitoring | grep loki

# Test Loki API
kubectl port-forward -n monitoring svc/loki 3100:3100

# Test query
curl http://localhost:3100/ready

# Should return "ready"

# Test LogQL query
curl -G -s "http://localhost:3100/loki/api/v1/query" \
  --data-urlencode 'query={job="greenlang-api"}' | jq
```

---

## Jaeger Setup

### Step 1: Install Jaeger

```bash
# Install Jaeger operator
kubectl create namespace observability
kubectl apply -f https://github.com/jaegertracing/jaeger-operator/releases/download/v1.42.0/jaeger-operator.yaml -n observability

# Wait for operator to be ready
kubectl wait --for=condition=available deployment/jaeger-operator -n observability --timeout=300s
```

### Step 2: Deploy Jaeger Instance

```bash
kubectl apply -f - << EOF
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: jaeger
  namespace: monitoring
spec:
  strategy: production

  storage:
    type: elasticsearch
    options:
      es:
        server-urls: http://elasticsearch:9200
        index-prefix: jaeger

  collector:
    maxReplicas: 5
    resources:
      requests:
        cpu: 100m
        memory: 256Mi
      limits:
        cpu: 1000m
        memory: 512Mi

  query:
    replicas: 2
    resources:
      requests:
        cpu: 100m
        memory: 256Mi
      limits:
        cpu: 500m
        memory: 512Mi

  ingress:
    enabled: true
    hosts:
      - jaeger.greenlang.io
EOF
```

### Step 3: Verify Jaeger Installation

```bash
# Check Jaeger pods
kubectl get pods -n monitoring | grep jaeger

# Port-forward to access Jaeger UI
kubectl port-forward -n monitoring svc/jaeger-query 16686:16686

# Open browser: http://localhost:16686

# You should see Jaeger UI
```

### Step 4: Configure Application to Send Traces

```python
# In your GreenLang application
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger-agent.monitoring.svc.cluster.local",
    agent_port=6831,
)

# Set up tracer provider
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Get tracer
tracer = trace.get_tracer(__name__)

# Use in code
with tracer.start_as_current_span("agent_execution"):
    result = agent.execute(input_data)
```

---

## Alertmanager Setup

### Step 1: Configure Alertmanager

Alertmanager is installed with Prometheus. Configure it:

```bash
kubectl apply -f - << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-config
  namespace: monitoring
data:
  alertmanager.yml: |
    global:
      resolve_timeout: 5m
      slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'

    route:
      group_by: ['alertname', 'severity']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 12h
      receiver: 'default'
      routes:
        # Critical alerts to PagerDuty
        - match:
            severity: critical
          receiver: 'pagerduty'
          continue: true

        # Warnings to Slack
        - match:
            severity: warning
          receiver: 'slack'

    receivers:
      - name: 'default'
        slack_configs:
          - channel: '#alerts'
            title: '[{{ .Status | toUpper }}] {{ .GroupLabels.alertname }}'
            text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

      - name: 'pagerduty'
        pagerduty_configs:
          - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
            description: '{{ .GroupLabels.alertname }}'

      - name: 'slack'
        slack_configs:
          - channel: '#warnings'
            title: '[{{ .Status | toUpper }}] {{ .GroupLabels.alertname }}'
            text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

    inhibit_rules:
      - source_match:
          severity: 'critical'
        target_match:
          severity: 'warning'
        equal: ['alertname']
EOF
```

### Step 2: Restart Alertmanager

```bash
kubectl rollout restart deployment -n monitoring alertmanager
```

### Step 3: Verify Alertmanager

```bash
# Port-forward to access Alertmanager UI
kubectl port-forward -n monitoring svc/prometheus-alertmanager 9093:80

# Open browser: http://localhost:9093

# You should see Alertmanager UI
```

---

## Dashboard Configuration

### Import Pre-Built Dashboards

```bash
# Create directory for dashboards
mkdir -p grafana-dashboards

# Download GreenLang system dashboard
cat > grafana-dashboards/greenlang-system.json << 'EOF'
{
  "dashboard": {
    "title": "GreenLang System Overview",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(gl_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(gl_errors_total[5m]) / rate(gl_requests_total[5m])"
          }
        ]
      },
      {
        "title": "p95 Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(gl_api_latency_seconds_bucket[5m]))"
          }
        ]
      }
    ]
  }
}
EOF

# Import dashboard into Grafana
curl -X POST http://admin:$GRAFANA_PASSWORD@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @grafana-dashboards/greenlang-system.json
```

### Essential Dashboards to Create

1. **System Overview**
   - Request rate
   - Error rate
   - Latency percentiles (p50, p95, p99)
   - CPU usage
   - Memory usage

2. **API Performance**
   - Endpoint latency
   - Endpoint error rates
   - Request counts by endpoint
   - Response status codes

3. **Database Performance**
   - Connection pool usage
   - Query duration
   - Slow queries
   - Deadlocks

4. **Agent Performance**
   - Agent execution rate
   - Agent success/failure rate
   - Agent execution duration
   - Concurrent executions

5. **Infrastructure**
   - Node CPU/Memory
   - Disk I/O
   - Network I/O
   - Pod status

---

## Alert Rules

### Configure Alert Rules

```bash
kubectl apply -f - << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-alerts
  namespace: monitoring
data:
  alerts.yml: |
    groups:
      - name: greenlang_alerts
        interval: 30s
        rules:
          # High error rate
          - alert: HighErrorRate
            expr: |
              rate(gl_errors_total[5m]) / rate(gl_requests_total[5m]) > 0.05
            for: 5m
            labels:
              severity: critical
            annotations:
              summary: "High error rate detected"
              description: "Error rate is {{ \$value | humanizePercentage }} (threshold: 5%)"

          # High latency
          - alert: HighLatency
            expr: |
              histogram_quantile(0.95, rate(gl_api_latency_seconds_bucket[5m])) > 1
            for: 10m
            labels:
              severity: warning
            annotations:
              summary: "High API latency detected"
              description: "p95 latency is {{ \$value | humanizeDuration }} (threshold: 1s)"

          # Service down
          - alert: ServiceDown
            expr: |
              up{job="greenlang-api"} == 0
            for: 2m
            labels:
              severity: critical
            annotations:
              summary: "GreenLang API is down"
              description: "GreenLang API has been down for more than 2 minutes"

          # High CPU usage
          - alert: HighCPUUsage
            expr: |
              rate(container_cpu_usage_seconds_total{pod=~"greenlang-.*"}[5m]) * 100 > 80
            for: 10m
            labels:
              severity: warning
            annotations:
              summary: "High CPU usage detected"
              description: "CPU usage is {{ \$value }}% (threshold: 80%)"

          # High memory usage
          - alert: HighMemoryUsage
            expr: |
              container_memory_usage_bytes{pod=~"greenlang-.*"} / container_spec_memory_limit_bytes{pod=~"greenlang-.*"} * 100 > 85
            for: 10m
            labels:
              severity: warning
            annotations:
              summary: "High memory usage detected"
              description: "Memory usage is {{ \$value }}% (threshold: 85%)"

          # Database connection pool exhausted
          - alert: DatabaseConnectionPoolExhausted
            expr: |
              pg_stat_database_numbackends / pg_settings_max_connections * 100 > 90
            for: 5m
            labels:
              severity: critical
            annotations:
              summary: "Database connection pool nearly exhausted"
              description: "Connection pool usage is {{ \$value }}% (threshold: 90%)"

          # Disk space low
          - alert: DiskSpaceLow
            expr: |
              (node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100 < 20
            for: 10m
            labels:
              severity: warning
            annotations:
              summary: "Disk space low"
              description: "Available disk space is {{ \$value }}% (threshold: 20%)"
EOF
```

---

## Notification Channels

### Slack Integration

```bash
# In Grafana UI:
# 1. Go to Alerting > Notification channels
# 2. Click "New channel"
# 3. Select "Slack"
# 4. Enter webhook URL
# 5. Test and save
```

### PagerDuty Integration

```bash
# In Grafana UI:
# 1. Go to Alerting > Notification channels
# 2. Click "New channel"
# 3. Select "PagerDuty"
# 4. Enter integration key
# 5. Set severity mapping
# 6. Test and save
```

### Email Integration

```bash
# Configure SMTP in Grafana
kubectl edit configmap grafana -n monitoring

# Add SMTP configuration:
[smtp]
enabled = true
host = smtp.example.com:587
user = alerts@greenlang.io
password = ********
skip_verify = false
from_address = alerts@greenlang.io
from_name = GreenLang Alerts
```

---

## Validation Checklist

### Monitoring Stack Health Check

```bash
# Check all monitoring pods are running
kubectl get pods -n monitoring

# Expected output (all Running):
# - prometheus-server
# - prometheus-alertmanager
# - grafana
# - loki
# - promtail (DaemonSet - one per node)
# - jaeger-query
# - jaeger-collector
# - jaeger-agent (DaemonSet)

# Verify metrics collection
kubectl port-forward -n monitoring svc/prometheus-server 9090:80
curl http://localhost:9090/api/v1/targets

# All targets should show "up": true

# Verify Grafana access
kubectl port-forward -n monitoring svc/grafana 3000:80
curl http://localhost:3000/api/health

# Should return {"database":"ok","version":"..."}

# Test log aggregation
kubectl port-forward -n monitoring svc/loki 3100:3100
curl http://localhost:3100/ready

# Should return "ready"

# Test tracing
kubectl port-forward -n monitoring svc/jaeger-query 16686:16686
curl http://localhost:16686/api/services

# Should return list of services
```

---

## Troubleshooting

### Issue: Prometheus not scraping targets

**Solution:**
```bash
# Check service discovery
kubectl logs -n monitoring deployment/prometheus-server | grep "discovery"

# Verify target endpoints exist
kubectl get endpoints -n greenlang

# Check network policies
kubectl get networkpolicies -A
```

### Issue: Grafana can't connect to Prometheus

**Solution:**
```bash
# Verify Prometheus service
kubectl get svc -n monitoring prometheus-server

# Test connectivity from Grafana pod
kubectl exec -it -n monitoring deployment/grafana -- curl http://prometheus-server/api/v1/status/config
```

### Issue: Logs not appearing in Loki

**Solution:**
```bash
# Check Promtail logs
kubectl logs -n monitoring daemonset/loki-promtail

# Verify Promtail can reach Loki
kubectl exec -it -n monitoring daemonset/loki-promtail -- wget -O- http://loki:3100/ready
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-07 | Operations Team | Initial comprehensive setup guide |

**Next Review Date:** 2026-02-07
**Approved By:** [CTO], [Operations Lead]

---

**Monitor everything, alert on what matters!**
