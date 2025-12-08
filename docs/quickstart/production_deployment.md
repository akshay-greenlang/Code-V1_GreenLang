# GreenLang Production Deployment Guide

## Overview

This guide provides a comprehensive checklist and procedures for deploying GreenLang in production environments. Follow this guide to ensure a secure, reliable, and performant deployment.

**Target Environments:**
- Enterprise data centers
- Private cloud deployments
- Hybrid cloud architectures
- Air-gapped industrial networks

**Estimated Time:** 4-8 hours (depending on complexity)

---

## Pre-Deployment Checklist

### Infrastructure Requirements

| Requirement | Specification | Verified |
|-------------|---------------|----------|
| **Compute** | | |
| API Servers | 3x (8 CPU, 16 GB RAM) | [ ] |
| Agent Workers | 2x (4 CPU, 8 GB RAM) per 100 agents | [ ] |
| ML Engine | 2x (8 CPU, 32 GB RAM, GPU optional) | [ ] |
| **Storage** | | |
| Database | 500 GB+ NVMe SSD, RAID 10 | [ ] |
| Time-Series | 1 TB+ NVMe SSD | [ ] |
| Object Storage | 500 GB+ for backups/models | [ ] |
| **Network** | | |
| Internal Network | 1 Gbps minimum, 10 Gbps recommended | [ ] |
| Load Balancer | Layer 7, SSL termination | [ ] |
| Firewall | Properly configured | [ ] |
| **DNS** | | |
| API Endpoint | api.greenlang.yourcompany.com | [ ] |
| Dashboard | dashboard.greenlang.yourcompany.com | [ ] |

### Software Prerequisites

| Software | Version | Verified |
|----------|---------|----------|
| Kubernetes | 1.25+ | [ ] |
| Helm | 3.10+ | [ ] |
| PostgreSQL | 14+ | [ ] |
| TimescaleDB | 2.10+ | [ ] |
| Redis | 7.0+ | [ ] |
| cert-manager | 1.10+ | [ ] |
| Ingress Controller | nginx 1.5+ or similar | [ ] |

### Security Prerequisites

| Requirement | Details | Verified |
|-------------|---------|----------|
| TLS Certificates | Valid certificates for all endpoints | [ ] |
| Secrets Management | Vault, AWS Secrets Manager, or equivalent | [ ] |
| LDAP/SSO | Connection details and service account | [ ] |
| Network Policies | Defined and tested | [ ] |
| Audit Requirements | Logging destination configured | [ ] |

---

## Section 1: Production-Ready Deployment Checklist

### 1.1 Namespace and RBAC Setup

```bash
# Create namespace
kubectl create namespace greenlang-prod

# Apply RBAC configuration
kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: greenlang-sa
  namespace: greenlang-prod
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: greenlang-role
  namespace: greenlang-prod
rules:
  - apiGroups: [""]
    resources: ["pods", "services", "configmaps", "secrets"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["apps"]
    resources: ["deployments", "statefulsets"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: greenlang-rolebinding
  namespace: greenlang-prod
subjects:
  - kind: ServiceAccount
    name: greenlang-sa
    namespace: greenlang-prod
roleRef:
  kind: Role
  name: greenlang-role
  apiGroup: rbac.authorization.k8s.io
EOF
```

### 1.2 Secrets Configuration

```bash
# Create database credentials
kubectl create secret generic greenlang-db-credentials \
  --namespace greenlang-prod \
  --from-literal=username=greenlang \
  --from-literal=password="$(openssl rand -base64 32)"

# Create Redis credentials
kubectl create secret generic greenlang-redis-credentials \
  --namespace greenlang-prod \
  --from-literal=password="$(openssl rand -base64 32)"

# Create JWT signing keys
openssl genrsa -out jwt-private.pem 4096
openssl rsa -in jwt-private.pem -pubout -out jwt-public.pem

kubectl create secret generic greenlang-jwt-keys \
  --namespace greenlang-prod \
  --from-file=private.pem=jwt-private.pem \
  --from-file=public.pem=jwt-public.pem

# Clean up local key files
rm jwt-private.pem jwt-public.pem

# Create encryption key for data at rest
kubectl create secret generic greenlang-encryption-key \
  --namespace greenlang-prod \
  --from-literal=key="$(openssl rand -base64 32)"
```

### 1.3 Storage Configuration

```yaml
# storage.yaml
---
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: greenlang-fast
provisioner: kubernetes.io/aws-ebs  # Adjust for your provider
parameters:
  type: gp3
  iopsPerGB: "50"
  throughput: "250"
  encrypted: "true"
reclaimPolicy: Retain
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: greenlang-db-pvc
  namespace: greenlang-prod
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: greenlang-fast
  resources:
    requests:
      storage: 500Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: greenlang-timeseries-pvc
  namespace: greenlang-prod
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: greenlang-fast
  resources:
    requests:
      storage: 1Ti
```

```bash
kubectl apply -f storage.yaml
```

---

## Section 2: High Availability Configuration

### 2.1 Database High Availability

```yaml
# values-postgresql.yaml
postgresql:
  enabled: true
  architecture: replication

  primary:
    persistence:
      enabled: true
      storageClass: greenlang-fast
      size: 500Gi
    resources:
      requests:
        cpu: 2000m
        memory: 8Gi
      limits:
        cpu: 4000m
        memory: 16Gi
    extendedConfiguration: |
      shared_buffers = 4GB
      effective_cache_size = 12GB
      work_mem = 256MB
      maintenance_work_mem = 1GB
      max_connections = 200
      wal_buffers = 64MB
      checkpoint_completion_target = 0.9
      max_wal_size = 4GB
      random_page_cost = 1.1
      effective_io_concurrency = 200

  readReplicas:
    replicaCount: 2
    persistence:
      enabled: true
      storageClass: greenlang-fast
      size: 500Gi
    resources:
      requests:
        cpu: 1000m
        memory: 4Gi
      limits:
        cpu: 2000m
        memory: 8Gi

  metrics:
    enabled: true
    serviceMonitor:
      enabled: true
```

### 2.2 Redis Cluster Configuration

```yaml
# values-redis.yaml
redis:
  enabled: true
  architecture: replication

  master:
    persistence:
      enabled: true
      storageClass: greenlang-fast
      size: 50Gi
    resources:
      requests:
        cpu: 500m
        memory: 2Gi
      limits:
        cpu: 1000m
        memory: 4Gi

  replica:
    replicaCount: 2
    persistence:
      enabled: true
      storageClass: greenlang-fast
      size: 50Gi
    resources:
      requests:
        cpu: 250m
        memory: 1Gi
      limits:
        cpu: 500m
        memory: 2Gi

  sentinel:
    enabled: true
    quorum: 2

  metrics:
    enabled: true
    serviceMonitor:
      enabled: true
```

### 2.3 Application High Availability

```yaml
# values-greenlang.yaml
global:
  environment: production
  imageRegistry: registry.greenlang.io
  imagePullSecrets:
    - name: greenlang-registry

api:
  replicas: 3

  resources:
    requests:
      cpu: 1000m
      memory: 2Gi
    limits:
      cpu: 2000m
      memory: 4Gi

  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: 80

  podDisruptionBudget:
    enabled: true
    minAvailable: 2

  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        - labelSelector:
            matchLabels:
              app: greenlang-api
          topologyKey: kubernetes.io/hostname

  topologySpreadConstraints:
    - maxSkew: 1
      topologyKey: topology.kubernetes.io/zone
      whenUnsatisfiable: ScheduleAnyway
      labelSelector:
        matchLabels:
          app: greenlang-api

agent:
  replicas: 2

  resources:
    requests:
      cpu: 2000m
      memory: 4Gi
    limits:
      cpu: 4000m
      memory: 8Gi

  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 20
    targetCPUUtilizationPercentage: 70

mlEngine:
  replicas: 2

  resources:
    requests:
      cpu: 4000m
      memory: 16Gi
    limits:
      cpu: 8000m
      memory: 32Gi

  gpu:
    enabled: true
    count: 1
    type: nvidia-tesla-t4
```

### 2.4 Deploy with Helm

```bash
# Add GreenLang Helm repository
helm repo add greenlang https://charts.greenlang.io
helm repo update

# Deploy GreenLang with production values
helm upgrade --install greenlang greenlang/greenlang \
  --namespace greenlang-prod \
  --values values-postgresql.yaml \
  --values values-redis.yaml \
  --values values-greenlang.yaml \
  --wait \
  --timeout 30m

# Verify deployment
kubectl get pods -n greenlang-prod
kubectl get svc -n greenlang-prod
```

---

## Section 3: Security Hardening

### 3.1 Network Policies

```yaml
# network-policies.yaml
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: greenlang-prod
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-api-ingress
  namespace: greenlang-prod
spec:
  podSelector:
    matchLabels:
      app: greenlang-api
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8000
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-api-to-db
  namespace: greenlang-prod
spec:
  podSelector:
    matchLabels:
      app: greenlang-api
  policyTypes:
    - Egress
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: postgresql
      ports:
        - protocol: TCP
          port: 5432
    - to:
        - podSelector:
            matchLabels:
              app: redis
      ports:
        - protocol: TCP
          port: 6379
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-db-internal
  namespace: greenlang-prod
spec:
  podSelector:
    matchLabels:
      app: postgresql
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: greenlang-api
        - podSelector:
            matchLabels:
              app: greenlang-agent
        - podSelector:
            matchLabels:
              app: greenlang-ml
      ports:
        - protocol: TCP
          port: 5432
```

```bash
kubectl apply -f network-policies.yaml
```

### 3.2 Pod Security Standards

```yaml
# pod-security.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: greenlang-prod
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

### 3.3 TLS Configuration

```yaml
# tls-config.yaml
---
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: greenlang-tls
  namespace: greenlang-prod
spec:
  secretName: greenlang-tls-secret
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
    - api.greenlang.yourcompany.com
    - dashboard.greenlang.yourcompany.com
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: greenlang-ingress
  namespace: greenlang-prod
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts:
        - api.greenlang.yourcompany.com
        - dashboard.greenlang.yourcompany.com
      secretName: greenlang-tls-secret
  rules:
    - host: api.greenlang.yourcompany.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: greenlang-api
                port:
                  number: 8000
    - host: dashboard.greenlang.yourcompany.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: greenlang-dashboard
                port:
                  number: 8080
```

### 3.4 Security Configuration

```yaml
# security-config.yaml (ConfigMap)
apiVersion: v1
kind: ConfigMap
metadata:
  name: greenlang-security-config
  namespace: greenlang-prod
data:
  security.yaml: |
    security:
      # Authentication
      auth:
        type: jwt
        issuer: greenlang-prod
        audience: greenlang-api
        access_token_expire: 3600
        refresh_token_expire: 86400

        password_policy:
          min_length: 14
          require_uppercase: true
          require_lowercase: true
          require_numbers: true
          require_special: true
          max_age_days: 90
          history_count: 24

        mfa:
          enabled: true
          required_for_admins: true
          methods:
            - totp
            - webauthn

        lockout:
          enabled: true
          max_attempts: 5
          lockout_duration: 1800

      # Session Management
      session:
        max_concurrent: 3
        idle_timeout: 1800
        absolute_timeout: 28800

      # TLS Settings
      tls:
        min_version: TLS1.2
        cipher_suites:
          - TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
          - TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
          - TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384

      # CORS Settings
      cors:
        enabled: true
        origins:
          - https://dashboard.greenlang.yourcompany.com
        methods:
          - GET
          - POST
          - PUT
          - DELETE
        credentials: true

      # Rate Limiting
      rate_limiting:
        enabled: true
        requests_per_minute: 1000
        burst: 100

      # Audit Logging
      audit:
        enabled: true
        events:
          - authentication
          - authorization
          - configuration_change
          - data_access
          - alarm_response
        output:
          - type: elasticsearch
            host: elasticsearch.logging:9200
            index: greenlang-audit
```

---

## Section 4: Monitoring Setup

### 4.1 Prometheus Configuration

```yaml
# monitoring/prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: greenlang-prod
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s

    alerting:
      alertmanagers:
        - static_configs:
            - targets:
                - alertmanager:9093

    rule_files:
      - /etc/prometheus/rules/*.yaml

    scrape_configs:
      - job_name: 'greenlang-api'
        kubernetes_sd_configs:
          - role: pod
            namespaces:
              names:
                - greenlang-prod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app]
            regex: greenlang-api
            action: keep
          - source_labels: [__meta_kubernetes_pod_container_port_number]
            regex: "9090"
            action: keep

      - job_name: 'greenlang-agent'
        kubernetes_sd_configs:
          - role: pod
            namespaces:
              names:
                - greenlang-prod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app]
            regex: greenlang-agent
            action: keep

      - job_name: 'greenlang-ml'
        kubernetes_sd_configs:
          - role: pod
            namespaces:
              names:
                - greenlang-prod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app]
            regex: greenlang-ml
            action: keep
```

### 4.2 Alert Rules

```yaml
# monitoring/alert-rules.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-rules
  namespace: greenlang-prod
data:
  greenlang-alerts.yaml: |
    groups:
      - name: greenlang-critical
        rules:
          - alert: GreenLangAPIDown
            expr: up{job="greenlang-api"} == 0
            for: 1m
            labels:
              severity: critical
            annotations:
              summary: "GreenLang API is down"
              description: "GreenLang API has been down for more than 1 minute."

          - alert: GreenLangHighErrorRate
            expr: rate(http_requests_total{job="greenlang-api",status=~"5.."}[5m]) / rate(http_requests_total{job="greenlang-api"}[5m]) > 0.05
            for: 5m
            labels:
              severity: critical
            annotations:
              summary: "High error rate on GreenLang API"
              description: "Error rate is above 5% for more than 5 minutes."

          - alert: GreenLangDatabaseDown
            expr: pg_up == 0
            for: 1m
            labels:
              severity: critical
            annotations:
              summary: "PostgreSQL database is down"

          - alert: GreenLangAgentProcessingLag
            expr: greenlang_agent_processing_lag_seconds > 60
            for: 5m
            labels:
              severity: critical
            annotations:
              summary: "Agent processing lag exceeds 60 seconds"

      - name: greenlang-warning
        rules:
          - alert: GreenLangHighLatency
            expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="greenlang-api"}[5m])) > 1
            for: 10m
            labels:
              severity: warning
            annotations:
              summary: "High API latency (P95 > 1s)"

          - alert: GreenLangHighMemoryUsage
            expr: container_memory_usage_bytes{pod=~"greenlang-.*"} / container_spec_memory_limit_bytes{pod=~"greenlang-.*"} > 0.85
            for: 10m
            labels:
              severity: warning
            annotations:
              summary: "High memory usage on GreenLang pods"

          - alert: GreenLangDiskSpaceLow
            expr: (node_filesystem_avail_bytes{mountpoint="/data"} / node_filesystem_size_bytes{mountpoint="/data"}) < 0.15
            for: 30m
            labels:
              severity: warning
            annotations:
              summary: "Disk space below 15%"
```

### 4.3 Grafana Dashboards

```yaml
# monitoring/grafana-dashboards.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: greenlang-dashboards
  namespace: greenlang-prod
  labels:
    grafana_dashboard: "true"
data:
  greenlang-overview.json: |
    {
      "dashboard": {
        "title": "GreenLang Overview",
        "panels": [
          {
            "title": "API Request Rate",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(http_requests_total{job=\"greenlang-api\"}[5m])"
              }
            ]
          },
          {
            "title": "API Latency P95",
            "type": "gauge",
            "targets": [
              {
                "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job=\"greenlang-api\"}[5m]))"
              }
            ]
          },
          {
            "title": "Active Agents",
            "type": "stat",
            "targets": [
              {
                "expr": "greenlang_active_agents_total"
              }
            ]
          },
          {
            "title": "ML Predictions/sec",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(greenlang_ml_predictions_total[5m])"
              }
            ]
          },
          {
            "title": "Active Alarms",
            "type": "stat",
            "targets": [
              {
                "expr": "greenlang_active_alarms_total"
              }
            ]
          }
        ]
      }
    }
```

---

## Section 5: Post-Deployment Verification

### 5.1 Health Checks

```bash
# Verify all pods are running
kubectl get pods -n greenlang-prod -o wide

# Check API health
curl -k https://api.greenlang.yourcompany.com/health

# Check database connectivity
kubectl exec -it -n greenlang-prod deploy/greenlang-api -- \
  greenlang-cli db check

# Check Redis connectivity
kubectl exec -it -n greenlang-prod deploy/greenlang-api -- \
  greenlang-cli cache check

# Verify ML engine
kubectl exec -it -n greenlang-prod deploy/greenlang-api -- \
  greenlang-cli ml status
```

### 5.2 Functional Tests

```bash
# Run integration tests
kubectl run greenlang-test \
  --namespace greenlang-prod \
  --image greenlang/integration-tests:latest \
  --restart=Never \
  --env="API_URL=http://greenlang-api:8000" \
  -- pytest /tests/integration --junitxml=/results/report.xml

# View test results
kubectl logs -n greenlang-prod greenlang-test
```

### 5.3 Performance Tests

```bash
# Run load test
kubectl run greenlang-loadtest \
  --namespace greenlang-prod \
  --image greenlang/loadtest:latest \
  --restart=Never \
  --env="API_URL=http://greenlang-api:8000" \
  --env="USERS=100" \
  --env="DURATION=300"

# Monitor during load test
kubectl top pods -n greenlang-prod
```

### 5.4 Security Validation

```bash
# Run security scan
kubectl run greenlang-security-scan \
  --namespace greenlang-prod \
  --image greenlang/security-scan:latest \
  --restart=Never \
  --env="TARGET=https://api.greenlang.yourcompany.com"

# Check TLS configuration
openssl s_client -connect api.greenlang.yourcompany.com:443 -tls1_2

# Verify network policies
kubectl exec -it -n greenlang-prod deploy/greenlang-api -- \
  curl -s --connect-timeout 5 http://google.com || echo "Egress blocked (expected)"
```

---

## Section 6: Operational Procedures

### 6.1 Backup Schedule

```yaml
# backups/backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: greenlang-backup
  namespace: greenlang-prod
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: backup
              image: greenlang/backup:latest
              env:
                - name: BACKUP_TYPE
                  value: full
                - name: S3_BUCKET
                  value: greenlang-backups-prod
                - name: RETENTION_DAYS
                  value: "30"
              envFrom:
                - secretRef:
                    name: greenlang-backup-credentials
          restartPolicy: OnFailure
```

### 6.2 Maintenance Windows

```bash
# Pre-maintenance checklist
greenlang-cli maintenance prepare \
  --window-start "2025-12-07T02:00:00Z" \
  --window-end "2025-12-07T04:00:00Z" \
  --notify-users

# During maintenance
kubectl scale deployment greenlang-api --replicas=1 -n greenlang-prod

# Post-maintenance
kubectl scale deployment greenlang-api --replicas=3 -n greenlang-prod
greenlang-cli maintenance complete
```

### 6.3 Upgrade Procedure

```bash
# 1. Backup before upgrade
greenlang-cli backup create --name pre-upgrade

# 2. Review release notes
greenlang-cli releases show v1.1.0

# 3. Upgrade with Helm
helm upgrade greenlang greenlang/greenlang \
  --namespace greenlang-prod \
  --values values-production.yaml \
  --version 1.1.0 \
  --wait

# 4. Verify upgrade
greenlang-cli version
kubectl get pods -n greenlang-prod

# 5. Run smoke tests
greenlang-cli test smoke
```

---

## Deployment Completion Checklist

### Pre-Production

| Task | Status |
|------|--------|
| Infrastructure provisioned | [ ] |
| Network configured | [ ] |
| SSL certificates installed | [ ] |
| Secrets configured | [ ] |
| LDAP/SSO integrated | [ ] |

### Deployment

| Task | Status |
|------|--------|
| Kubernetes resources deployed | [ ] |
| Database initialized | [ ] |
| Initial admin user created | [ ] |
| Agents configured | [ ] |
| Integrations connected | [ ] |

### Verification

| Task | Status |
|------|--------|
| Health checks passing | [ ] |
| Functional tests passing | [ ] |
| Performance acceptable | [ ] |
| Security scan clean | [ ] |
| Monitoring operational | [ ] |

### Documentation

| Task | Status |
|------|--------|
| Runbook created | [ ] |
| On-call procedures documented | [ ] |
| Backup/recovery tested | [ ] |
| Team trained | [ ] |
| Go-live approval obtained | [ ] |

---

## Support

- **Documentation:** https://docs.greenlang.io
- **Enterprise Support:** enterprise-support@greenlang.io
- **Emergency Hotline:** +1-800-GREENLANG (24/7 for enterprise customers)

---

*Production Deployment Guide Version: 1.0.0*
*Last Updated: December 2025*
