# GL-FOUND-X-001 GreenLang Orchestrator - Operational Runbook

**Version:** 2.0.0
**Last Updated:** 2026-01-27
**Maintainer:** GreenLang Platform Team

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Deployment](#2-deployment)
3. [Monitoring & Alerting](#3-monitoring--alerting)
4. [Common Operations](#4-common-operations)
5. [Troubleshooting](#5-troubleshooting)
6. [Error Codes Reference](#6-error-codes-reference)
7. [Emergency Procedures](#7-emergency-procedures)

---

## 1. System Overview

### 1.1 Architecture Diagram

```
                                    +------------------+
                                    |   Load Balancer  |
                                    |   (Ingress/ALB)  |
                                    +--------+---------+
                                             |
                                             v
+------------------------------------------------------------------------------------+
|                              CONTROL PLANE                                         |
|  +------------------+    +------------------+    +------------------+              |
|  |  Orchestrator    |    |  Orchestrator    |    |  Orchestrator    |   (HPA)     |
|  |  Pod (Leader)    |<-->|  Pod (Replica)   |<-->|  Pod (Replica)   |              |
|  +--------+---------+    +--------+---------+    +--------+---------+              |
|           |                       |                       |                        |
|           +-------------------+---+-------------------+---+                        |
|                               |                                                    |
|                               v                                                    |
|  +------------------+    +------------------+    +------------------+              |
|  |   PostgreSQL     |    |     Redis        |    |  Policy Engine   |              |
|  |   (Audit Store)  |    |  (Queue/Cache)   |    |     (OPA)        |              |
|  +------------------+    +------------------+    +------------------+              |
+------------------------------------------------------------------------------------+
                               |
                               v
+------------------------------------------------------------------------------------+
|                              DATA PLANE (Per Tenant Namespace)                     |
|                                                                                    |
|  +------------------+    +------------------+    +------------------+              |
|  |   K8s Job        |    |   K8s Job        |    |   K8s Job        |              |
|  |  (Agent Pod)     |    |  (Agent Pod)     |    |  (Agent Pod)     |              |
|  |                  |    |                  |    |                  |              |
|  | GL_INPUT_URI  <--|----|--> S3 Artifact   |    |                  |              |
|  | GL_OUTPUT_URI -->|----|--> Store         |    |                  |              |
|  +------------------+    +------------------+    +------------------+              |
|                                                                                    |
+------------------------------------------------------------------------------------+
                               |
                               v
                    +------------------+
                    |   S3 / MinIO     |
                    |  (Artifact Store)|
                    |                  |
                    | {tenant}/runs/   |
                    |   {run_id}/      |
                    |     steps/       |
                    |       {step_id}/ |
                    |         input.json
                    |         result.json
                    |         artifacts/
                    +------------------+
```

### 1.2 Key Components

| Component | Description | Technology | Criticality |
|-----------|-------------|------------|-------------|
| **Orchestrator** | Core execution engine managing pipeline DAGs | Python/FastAPI | Critical |
| **K8s Executor** | Executes agent containers as K8s Jobs | kubernetes_asyncio | Critical |
| **S3 Artifact Store** | Stores input/output artifacts with checksums | aioboto3 (S3/MinIO) | Critical |
| **Policy Engine** | OPA + YAML hybrid policy enforcement | OPA Rego + Python | High |
| **Audit Event Store** | Hash-chained immutable audit trail | PostgreSQL + asyncpg | High |
| **Redis Queue** | Job queue and distributed locking | Redis | Medium |

### 1.3 Dependencies

| Dependency | Version | Purpose | Required |
|------------|---------|---------|----------|
| PostgreSQL | 14+ | Audit event store, run state | Yes |
| Redis | 7+ | Job queue, distributed locking, caching | Yes |
| S3 / MinIO | - | Artifact storage | Yes |
| Kubernetes | 1.25+ | Agent execution backend | Yes |
| OPA | 0.55+ | Policy evaluation (optional) | No |

### 1.4 Data Flow

1. **Run Submission**: Client submits pipeline YAML via REST API
2. **Plan Compilation**: Orchestrator compiles DAG, generates idempotency keys
3. **Policy Check**: Pre-run policies evaluated (OPA + YAML rules)
4. **Step Scheduling**: Ready steps queued based on dependency graph
5. **K8s Execution**: Each step runs as isolated K8s Job with:
   - `GL_INPUT_URI`: S3 path to `input.json` (RunContext)
   - `GL_OUTPUT_URI`: S3 prefix for outputs
6. **Artifact Storage**: Agent writes `result.json` and artifacts to S3
7. **Audit Events**: Hash-chained events emitted at each transition

---

## 2. Deployment

### 2.1 Prerequisites

Before deploying the orchestrator, ensure the following are available:

| Prerequisite | Description | Verification Command |
|--------------|-------------|----------------------|
| K8s Cluster | v1.25+ with RBAC enabled | `kubectl version` |
| S3 Bucket | For artifact storage | `aws s3 ls s3://{bucket}` |
| PostgreSQL | For audit store | `psql -h {host} -U {user} -c '\l'` |
| Redis | For queue/caching | `redis-cli -h {host} ping` |
| Helm | v3.10+ | `helm version` |

### 2.2 Environment Variables Reference

```bash
# ============================================================================
# CORE CONFIGURATION
# ============================================================================

# Service Identity
GL_SERVICE_NAME=greenlang-orchestrator
GL_ENVIRONMENT=production           # production | staging | development
GL_LOG_LEVEL=INFO                   # DEBUG | INFO | WARNING | ERROR

# ============================================================================
# KUBERNETES EXECUTOR
# ============================================================================

GL_K8S_NAMESPACE=greenlang          # Default namespace for jobs
GL_K8S_SERVICE_ACCOUNT=greenlang-runner
GL_K8S_IMAGE_PULL_SECRETS=ghcr-secret,ecr-secret
GL_K8S_JOB_TTL_SECONDS=3600         # Cleanup completed jobs after 1 hour
GL_K8S_ACTIVE_DEADLINE_SECONDS=3600 # Max job duration
GL_K8S_BACKOFF_LIMIT=0              # K8s-level retries (0 = orchestrator handles)
GL_K8S_IN_CLUSTER=true              # true if running in K8s

# ============================================================================
# S3 ARTIFACT STORE
# ============================================================================

GL_S3_BUCKET=greenlang-artifacts
GL_S3_REGION=us-east-1
GL_S3_ENDPOINT=                     # Leave empty for AWS, set for MinIO
GL_S3_ACCESS_KEY_ID=                # Optional if using IAM roles
GL_S3_SECRET_ACCESS_KEY=            # Optional if using IAM roles
GL_S3_USE_SSL=true
GL_S3_VERIFY_SSL=true

# ============================================================================
# POSTGRESQL (Audit Store)
# ============================================================================

GL_POSTGRES_HOST=postgres.greenlang.svc.cluster.local
GL_POSTGRES_PORT=5432
GL_POSTGRES_USER=greenlang
GL_POSTGRES_PASSWORD=               # Use secret reference
GL_POSTGRES_DATABASE=greenlang_audit
GL_POSTGRES_POOL_SIZE=10
GL_POSTGRES_MAX_OVERFLOW=5

# Full URL alternative:
# GL_AUDIT_DB_URL=postgresql+asyncpg://user:pass@host:5432/db

# ============================================================================
# REDIS (Queue/Cache)
# ============================================================================

GL_REDIS_HOST=redis.greenlang.svc.cluster.local
GL_REDIS_PORT=6379
GL_REDIS_PASSWORD=                  # Use secret reference
GL_REDIS_DB=0
GL_REDIS_SSL=false

# ============================================================================
# POLICY ENGINE
# ============================================================================

GL_OPA_ENABLED=true
GL_OPA_URL=http://opa.greenlang.svc.cluster.local:8181
GL_OPA_TIMEOUT_SECONDS=5
GL_OPA_RETRY_COUNT=2

GL_YAML_RULES_ENABLED=true
GL_YAML_RULES_PATH=/etc/greenlang/policies/rules.yaml

GL_POLICY_DEFAULT_ACTION=deny       # deny | allow
GL_POLICY_STRICT_MODE=true          # Deny on policy evaluation errors
GL_POLICY_CACHE_TTL_SECONDS=300

# ============================================================================
# EXECUTION SETTINGS
# ============================================================================

GL_MAX_PARALLEL_STEPS=10            # Max concurrent step executions
GL_DEFAULT_TIMEOUT_SECONDS=3600     # Default step timeout
GL_DEFAULT_CPU_LIMIT=1              # Default CPU limit per step
GL_DEFAULT_MEMORY_LIMIT=2Gi         # Default memory limit per step
GL_ENABLE_LEGACY_FALLBACK=true      # Fallback to HTTP adapter

# ============================================================================
# OBSERVABILITY
# ============================================================================

GL_METRICS_PORT=9090
GL_HEALTH_PORT=8080
GL_TRACE_EXPORTER=otlp              # otlp | jaeger | none
GL_TRACE_ENDPOINT=http://otel-collector:4317
GL_TRACE_SAMPLE_RATE=0.1            # 10% sampling in production
```

### 2.3 Helm Chart Values

```yaml
# values.yaml for greenlang-orchestrator Helm chart
# Version: 2.0.0

replicaCount: 3

image:
  repository: ghcr.io/greenlang/orchestrator
  tag: "2.0.0"
  pullPolicy: IfNotPresent

imagePullSecrets:
  - name: ghcr-secret

serviceAccount:
  create: true
  name: greenlang-orchestrator
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789:role/greenlang-orchestrator

podAnnotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "9090"
  prometheus.io/path: "/metrics"

resources:
  requests:
    cpu: "500m"
    memory: "1Gi"
  limits:
    cpu: "2"
    memory: "4Gi"

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

# Pod disruption budget
podDisruptionBudget:
  enabled: true
  minAvailable: 2

# Liveness and readiness probes
livenessProbe:
  httpGet:
    path: /health/live
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3

# Environment configuration
env:
  GL_ENVIRONMENT: production
  GL_LOG_LEVEL: INFO
  GL_K8S_NAMESPACE: greenlang
  GL_S3_BUCKET: greenlang-artifacts-prod
  GL_S3_REGION: us-east-1
  GL_OPA_ENABLED: "true"
  GL_MAX_PARALLEL_STEPS: "10"

# Secrets (reference from Kubernetes secrets)
envFrom:
  - secretRef:
      name: greenlang-orchestrator-secrets

# ConfigMap for policy rules
configMaps:
  policies:
    mountPath: /etc/greenlang/policies
    data:
      rules.yaml: |
        name: production-policies
        version: "1.0.0"
        rules:
          - name: require_approval_for_production
            condition: namespace == 'production'
            action: require_approval
            approval_type: manager

# Service configuration
service:
  type: ClusterIP
  port: 8080
  metricsPort: 9090

# Ingress configuration
ingress:
  enabled: true
  className: nginx
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: orchestrator.greenlang.io
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: orchestrator-tls
      hosts:
        - orchestrator.greenlang.io

# PostgreSQL dependency (using subchart or external)
postgresql:
  enabled: false  # Use external PostgreSQL
  external:
    host: postgres.greenlang.svc.cluster.local
    port: 5432
    database: greenlang_audit
    existingSecret: postgres-credentials

# Redis dependency
redis:
  enabled: false  # Use external Redis
  external:
    host: redis.greenlang.svc.cluster.local
    port: 6379
    existingSecret: redis-credentials

# OPA dependency
opa:
  enabled: true
  image:
    repository: openpolicyagent/opa
    tag: "0.55.0"
  resources:
    requests:
      cpu: "100m"
      memory: "256Mi"
    limits:
      cpu: "500m"
      memory: "512Mi"

# Network policies
networkPolicy:
  enabled: true
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
    - from:
        - podSelector:
            matchLabels:
              app: greenlang-api

# RBAC for K8s executor
rbac:
  create: true
  rules:
    - apiGroups: ["batch"]
      resources: ["jobs"]
      verbs: ["create", "delete", "get", "list", "watch"]
    - apiGroups: [""]
      resources: ["pods", "pods/log"]
      verbs: ["get", "list", "watch"]
```

### 2.4 Rolling Update Procedure

**Pre-Update Checklist:**

```bash
# 1. Check current deployment status
kubectl get deployment greenlang-orchestrator -n greenlang -o wide
kubectl get pods -l app=greenlang-orchestrator -n greenlang

# 2. Verify no stuck jobs
kubectl get jobs -l app.kubernetes.io/name=greenlang-agent -n greenlang | grep -v Completed

# 3. Check current metrics
curl -s http://orchestrator:9090/metrics | grep gl_active_runs

# 4. Backup current configmaps/secrets
kubectl get configmap greenlang-policies -n greenlang -o yaml > policies-backup.yaml
```

**Rolling Update Steps:**

```bash
# 1. Update Helm values (change image tag)
vim values.yaml  # Update image.tag to new version

# 2. Dry-run to verify changes
helm diff upgrade greenlang-orchestrator ./greenlang-orchestrator \
  -f values.yaml \
  -n greenlang

# 3. Apply the update
helm upgrade greenlang-orchestrator ./greenlang-orchestrator \
  -f values.yaml \
  -n greenlang \
  --wait \
  --timeout 10m

# 4. Monitor rollout
kubectl rollout status deployment/greenlang-orchestrator -n greenlang

# 5. Verify new pods are healthy
kubectl get pods -l app=greenlang-orchestrator -n greenlang
kubectl logs -l app=greenlang-orchestrator -n greenlang --tail=50
```

**Rollback Procedure:**

```bash
# Immediate rollback
helm rollback greenlang-orchestrator -n greenlang

# Or rollback to specific revision
helm history greenlang-orchestrator -n greenlang
helm rollback greenlang-orchestrator 5 -n greenlang
```

---

## 3. Monitoring & Alerting

### 3.1 Key Metrics to Monitor

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| `gl_run_success_rate` | Percentage of successful runs | > 95% |
| `gl_step_latency_seconds` | Step execution latency | p99 < 300s |
| `gl_queue_depth` | Pending jobs in queue | < 100 |
| `gl_active_runs` | Currently executing runs | < max_parallel |
| `gl_policy_evaluation_ms` | Policy check latency | p99 < 100ms |
| `gl_audit_chain_valid` | Audit chain integrity | = 1 (always) |
| `gl_k8s_job_failures` | K8s job failure count | < 5/min |
| `gl_s3_error_rate` | S3 operation error rate | < 1% |

### 3.2 Prometheus Queries for Common Scenarios

**Run Success Rate (Last Hour):**

```promql
# Overall success rate
sum(rate(gl_run_completed_total{status="success"}[1h])) /
sum(rate(gl_run_completed_total[1h])) * 100

# Success rate by tenant
sum by (tenant_id) (rate(gl_run_completed_total{status="success"}[1h])) /
sum by (tenant_id) (rate(gl_run_completed_total[1h])) * 100
```

**Step Latency Distribution:**

```promql
# P50, P90, P99 step latency
histogram_quantile(0.50, sum(rate(gl_step_duration_seconds_bucket[5m])) by (le))
histogram_quantile(0.90, sum(rate(gl_step_duration_seconds_bucket[5m])) by (le))
histogram_quantile(0.99, sum(rate(gl_step_duration_seconds_bucket[5m])) by (le))

# Latency by agent type
histogram_quantile(0.99, sum(rate(gl_step_duration_seconds_bucket[5m])) by (le, agent_id))
```

**Queue Depth and Throughput:**

```promql
# Current queue depth
gl_queue_depth

# Throughput (runs completed per minute)
sum(rate(gl_run_completed_total[5m])) * 60

# Steps executed per minute
sum(rate(gl_step_completed_total[5m])) * 60
```

**Error Rate by Type:**

```promql
# Error rate by error class
sum by (error_class) (rate(gl_errors_total[5m]))

# Top 5 error codes
topk(5, sum by (error_code) (rate(gl_errors_total[5m])))

# Policy violation rate
sum(rate(gl_policy_violations_total[5m]))
```

**Resource Utilization:**

```promql
# Memory usage per orchestrator pod
container_memory_usage_bytes{pod=~"greenlang-orchestrator.*"}

# CPU usage
rate(container_cpu_usage_seconds_total{pod=~"greenlang-orchestrator.*"}[5m])

# K8s job resource consumption by tenant
sum by (tenant_id) (kube_pod_container_resource_requests{pod=~"gl-.*", resource="memory"})
```

### 3.3 Alert Thresholds and Escalation

| Alert | Condition | Severity | Escalation |
|-------|-----------|----------|------------|
| `OrchestratorDown` | `up{job="greenlang-orchestrator"} == 0` | Critical | PagerDuty (immediate) |
| `HighRunFailureRate` | Success rate < 90% for 15m | Critical | PagerDuty (5m) |
| `HighStepLatency` | p99 > 600s for 10m | Warning | Slack #platform-alerts |
| `QueueBacklogHigh` | Queue depth > 500 for 5m | Warning | Slack #platform-alerts |
| `PolicyEngineDown` | OPA health check failing | High | PagerDuty (15m) |
| `AuditChainBroken` | `gl_audit_chain_valid == 0` | Critical | PagerDuty (immediate) |
| `S3ErrorRateHigh` | S3 error rate > 5% for 5m | High | Slack #platform-alerts |
| `PostgresConnectionPool` | Available connections < 2 | Warning | Slack #platform-alerts |

**Sample Alert Rules (Prometheus):**

```yaml
groups:
  - name: greenlang-orchestrator
    rules:
      - alert: OrchestratorDown
        expr: up{job="greenlang-orchestrator"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "GreenLang Orchestrator is down"
          description: "{{ $labels.instance }} has been down for more than 1 minute"

      - alert: HighRunFailureRate
        expr: |
          (sum(rate(gl_run_completed_total{status="failed"}[15m])) /
           sum(rate(gl_run_completed_total[15m]))) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High pipeline run failure rate"
          description: "Run failure rate is {{ $value | humanizePercentage }}"

      - alert: AuditChainIntegrityFailure
        expr: gl_audit_chain_valid == 0
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "Audit chain integrity compromised"
          description: "Hash chain verification failed for run {{ $labels.run_id }}"

      - alert: StepLatencyHigh
        expr: histogram_quantile(0.99, sum(rate(gl_step_duration_seconds_bucket[5m])) by (le)) > 600
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High step execution latency"
          description: "p99 step latency is {{ $value | humanizeDuration }}"
```

---

## 4. Common Operations

### 4.1 Scaling the Orchestrator

**Horizontal Scaling:**

```bash
# Manual scaling
kubectl scale deployment greenlang-orchestrator -n greenlang --replicas=5

# Verify HPA is working
kubectl get hpa greenlang-orchestrator -n greenlang
kubectl describe hpa greenlang-orchestrator -n greenlang

# Adjust HPA limits
kubectl patch hpa greenlang-orchestrator -n greenlang \
  --type='json' \
  -p='[{"op": "replace", "path": "/spec/maxReplicas", "value": 15}]'
```

**Vertical Scaling (Resource Adjustment):**

```bash
# Update resource requests/limits
kubectl set resources deployment greenlang-orchestrator -n greenlang \
  --requests=cpu=1,memory=2Gi \
  --limits=cpu=4,memory=8Gi
```

**Scaling Worker Capacity:**

```bash
# Increase max parallel steps (requires restart)
kubectl set env deployment/greenlang-orchestrator -n greenlang \
  GL_MAX_PARALLEL_STEPS=20

# Scale underlying K8s node pool if needed
# (Cloud-provider specific)
```

### 4.2 Draining Jobs Before Maintenance

**Graceful Drain Procedure:**

```bash
# 1. Stop accepting new runs (set to maintenance mode)
kubectl annotate deployment greenlang-orchestrator -n greenlang \
  greenlang.io/maintenance-mode="true"

# 2. Wait for active runs to complete (monitor)
watch 'kubectl exec -n greenlang deploy/greenlang-orchestrator -- \
  curl -s localhost:8080/api/v1/admin/active-runs | jq ".count"'

# 3. Check no pending jobs
kubectl get jobs -l app.kubernetes.io/name=greenlang-agent -n greenlang \
  --field-selector=status.successful!=1

# 4. Once drained, proceed with maintenance
kubectl get pods -l app=greenlang-orchestrator -n greenlang

# 5. After maintenance, remove maintenance mode
kubectl annotate deployment greenlang-orchestrator -n greenlang \
  greenlang.io/maintenance-mode-
```

**Force Drain (Emergency):**

```bash
# Cancel all active runs
kubectl exec -n greenlang deploy/greenlang-orchestrator -- \
  curl -X POST localhost:8080/api/v1/admin/cancel-all

# Delete all pending K8s jobs
kubectl delete jobs -l app.kubernetes.io/name=greenlang-agent -n greenlang
```

### 4.3 Rotating Credentials

**S3 Credentials Rotation:**

```bash
# 1. Create new secret with rotated credentials
kubectl create secret generic greenlang-s3-credentials-new -n greenlang \
  --from-literal=AWS_ACCESS_KEY_ID=<new-key> \
  --from-literal=AWS_SECRET_ACCESS_KEY=<new-secret>

# 2. Update deployment to use new secret
kubectl set env deployment/greenlang-orchestrator -n greenlang \
  --from=secret/greenlang-s3-credentials-new

# 3. Verify new credentials work
kubectl exec -n greenlang deploy/greenlang-orchestrator -- \
  curl -s localhost:8080/health/ready | jq '.s3'

# 4. Delete old secret
kubectl delete secret greenlang-s3-credentials-old -n greenlang
```

**PostgreSQL Password Rotation:**

```bash
# 1. Update password in PostgreSQL
psql -h $PG_HOST -U postgres -c "ALTER USER greenlang PASSWORD 'new-password';"

# 2. Update K8s secret
kubectl create secret generic postgres-credentials -n greenlang \
  --from-literal=password='new-password' \
  --dry-run=client -o yaml | kubectl apply -f -

# 3. Restart orchestrator pods to pick up new credentials
kubectl rollout restart deployment/greenlang-orchestrator -n greenlang
```

**Redis Password Rotation:**

```bash
# 1. Update Redis password (requires Redis restart)
redis-cli -h $REDIS_HOST CONFIG SET requirepass "new-password"

# 2. Update K8s secret
kubectl create secret generic redis-credentials -n greenlang \
  --from-literal=password='new-password' \
  --dry-run=client -o yaml | kubectl apply -f -

# 3. Restart orchestrator
kubectl rollout restart deployment/greenlang-orchestrator -n greenlang
```

### 4.4 Backup and Restore Procedures

**Audit Store Backup:**

```bash
# Daily backup (should be automated via CronJob)
pg_dump -h $PG_HOST -U greenlang -d greenlang_audit \
  --format=custom \
  --file=/backups/audit-$(date +%Y%m%d).dump

# Upload to S3
aws s3 cp /backups/audit-$(date +%Y%m%d).dump \
  s3://greenlang-backups/audit/audit-$(date +%Y%m%d).dump

# Verify backup integrity
pg_restore --list /backups/audit-$(date +%Y%m%d).dump | head
```

**Audit Store Restore:**

```bash
# 1. Stop orchestrator
kubectl scale deployment greenlang-orchestrator -n greenlang --replicas=0

# 2. Download backup
aws s3 cp s3://greenlang-backups/audit/audit-20260127.dump /tmp/

# 3. Restore database
pg_restore -h $PG_HOST -U greenlang -d greenlang_audit \
  --clean --if-exists \
  /tmp/audit-20260127.dump

# 4. Verify chain integrity
psql -h $PG_HOST -U greenlang -d greenlang_audit \
  -c "SELECT run_id, COUNT(*) FROM run_events GROUP BY run_id ORDER BY COUNT(*) DESC LIMIT 10;"

# 5. Restart orchestrator
kubectl scale deployment greenlang-orchestrator -n greenlang --replicas=3
```

**S3 Artifact Backup:**

```bash
# Cross-region replication should be enabled for critical buckets
# Manual backup for specific tenant
aws s3 sync s3://greenlang-artifacts/tenant-123/ \
  s3://greenlang-artifacts-backup/tenant-123/ \
  --source-region us-east-1 \
  --region us-west-2
```

---

## 5. Troubleshooting

### 5.1 Run Stuck in PENDING

**Symptoms:**
- Run status shows `PENDING` for extended period
- No K8s jobs created
- Queue depth increasing

**Diagnosis:**

```bash
# 1. Check orchestrator logs
kubectl logs -l app=greenlang-orchestrator -n greenlang --tail=100 | grep -i error

# 2. Check run details via API
curl -s "http://orchestrator:8080/api/v1/runs/{run_id}" | jq

# 3. Check Redis queue
redis-cli -h $REDIS_HOST LLEN greenlang:run_queue

# 4. Check for policy blocks
curl -s "http://orchestrator:8080/api/v1/runs/{run_id}/events" | jq '.[] | select(.event_type == "POLICY_EVALUATED")'
```

**Common Causes and Fixes:**

| Cause | Fix |
|-------|-----|
| Policy pre-run check blocked | Review policy decision, request approval if needed |
| Redis connection issues | Check Redis connectivity, restart orchestrator |
| K8s namespace quota exceeded | Request quota increase or free resources |
| Orchestrator overloaded | Scale up replicas, check HPA |
| Invalid pipeline YAML | Fix YAML errors (check events for details) |

### 5.2 Step Timeout Issues

**Symptoms:**
- Steps failing with `GL-E-K8S-JOB-TIMEOUT`
- K8s jobs reaching active deadline

**Diagnosis:**

```bash
# 1. Check step duration distribution
kubectl exec -n greenlang deploy/greenlang-orchestrator -- \
  curl -s localhost:9090/metrics | grep gl_step_duration

# 2. Get job details
kubectl describe job gl-{step_id}-* -n greenlang

# 3. Check pod logs
kubectl logs -l greenlang.io/step-id={step_id} -n greenlang --tail=500

# 4. Check resource utilization
kubectl top pod -l greenlang.io/step-id={step_id} -n greenlang
```

**Fixes:**

```yaml
# Increase timeout in pipeline YAML
steps:
  - name: slow_step
    agent: GL-MRV-X-001
    timeout_seconds: 7200  # 2 hours
    resources:
      cpu: "2"
      memory: "8Gi"
```

### 5.3 Policy Violation Errors

**Symptoms:**
- Runs failing with `GL-E-OPA-DENY` or `GL-E-YAML-POLICY`
- Steps blocked at pre-step

**Diagnosis:**

```bash
# 1. Get policy decision details
curl -s "http://orchestrator:8080/api/v1/runs/{run_id}/events" | \
  jq '.[] | select(.event_type == "POLICY_EVALUATED")'

# 2. Check OPA directly
curl -X POST http://opa:8181/v1/data/greenlang/policies/pre_run \
  -H "Content-Type: application/json" \
  -d '{"input": {...}}'

# 3. View active YAML rules
kubectl get configmap greenlang-policies -n greenlang -o yaml
```

**Fixes:**

1. **Modify pipeline to comply**: Update YAML to meet policy requirements
2. **Request policy exception**: Submit approval request through workflow
3. **Update policy rules**: If rule is incorrect, update YAML rules ConfigMap

### 5.4 Audit Chain Integrity Failure

**Symptoms:**
- Alert `AuditChainIntegrityFailure` fired
- `gl_audit_chain_valid == 0`
- Error code `GL-E-AUDIT-CHAIN-BROKEN`

**Diagnosis:**

```bash
# 1. Identify broken chain
psql -h $PG_HOST -U greenlang -d greenlang_audit -c "
SELECT run_id, event_id, event_type, prev_event_hash, event_hash
FROM run_events
WHERE run_id = '{run_id}'
ORDER BY timestamp;"

# 2. Verify hash computation
kubectl exec -n greenlang deploy/greenlang-orchestrator -- \
  python -c "
from greenlang.orchestrator.audit.event_store import RunEvent
# Verify hash computation for specific event
"
```

**CRITICAL: This indicates potential tampering or data corruption.**

**Immediate Actions:**

1. **Preserve evidence**: Do not modify audit data
2. **Export affected run**: `GET /api/v1/runs/{run_id}/audit-package`
3. **Notify security team**: Potential compliance incident
4. **Restore from backup**: If corruption (not tampering) is confirmed

### 5.5 K8s Job Failures

**Symptoms:**
- Steps failing with K8s-related error codes
- Pods in CrashLoopBackOff or ImagePullBackOff

**Diagnosis:**

```bash
# 1. List failed jobs
kubectl get jobs -l app.kubernetes.io/name=greenlang-agent -n greenlang \
  --field-selector=status.successful!=1

# 2. Describe failed job
kubectl describe job gl-{step_id}-* -n greenlang

# 3. Get pod status
kubectl get pods -l job-name=gl-{step_id}-* -n greenlang -o wide

# 4. Check events
kubectl get events -n greenlang --sort-by='.lastTimestamp' | tail -20
```

**Common Issues:**

| Error | Cause | Fix |
|-------|-------|-----|
| `ImagePullBackOff` | Image not found or auth failed | Check image name, pull secrets |
| `OOMKilled` (exit 137) | Insufficient memory | Increase memory limit |
| `CrashLoopBackOff` | Agent crash | Check agent logs, fix code |
| `Pending` | Insufficient resources | Free resources or scale cluster |
| `Evicted` | Node pressure | Use QoS Guaranteed, scale cluster |

---

## 6. Error Codes Reference

### 6.1 GL-E-1xxx: Input Errors (USER_CONFIG)

| Code | Description | Fix |
|------|-------------|-----|
| `GL-E-YAML-INVALID` | YAML syntax error | Validate YAML with `greenlang pipeline validate` |
| `GL-E-YAML-SCHEMA` | Schema validation failed | Check pipeline schema documentation |
| `GL-E-YAML-VERSION` | Unsupported pipeline version | Update to supported version |
| `GL-E-DAG-CYCLE` | Circular dependency detected | Remove cycle from step dependencies |
| `GL-E-DAG-ORPHAN` | Step not connected to output | Connect step or remove it |
| `GL-E-DAG-MISSING-DEP` | Missing dependency reference | Add missing step or fix reference |
| `GL-E-PARAM-MISSING` | Required parameter missing | Add parameter to pipeline |
| `GL-E-PARAM-TYPE` | Parameter type mismatch | Fix parameter type |
| `GL-E-PARAM-RANGE` | Parameter out of range | Adjust value to valid range |
| `GL-E-DATA-NOT-FOUND` | Input dataset not found | Verify dataset path |
| `GL-E-DATA-SCHEMA` | Input data schema mismatch | Fix input data format |

### 6.2 GL-E-2xxx: Execution Errors (AGENT_BUG / TRANSIENT)

| Code | Description | Retry | Fix |
|------|-------------|-------|-----|
| `GL-E-AGENT-CRASH` | Agent crashed | No | Check agent logs, fix code |
| `GL-E-AGENT-TIMEOUT` | Agent timed out | Yes (bounded) | Increase timeout, optimize agent |
| `GL-E-AGENT-OUTPUT-INVALID` | Invalid agent output | No | Fix agent output schema |
| `GL-E-NETWORK-TIMEOUT` | Network timeout | Yes (exponential) | Transient, retry |
| `GL-E-NETWORK-REFUSED` | Connection refused | Yes (exponential) | Check service health |
| `GL-E-API-RATE-LIMIT` | Rate limit exceeded | Yes (exponential) | Backoff, request limit increase |
| `GL-E-API-5XX` | Server error | Yes (exponential) | Transient, retry |

### 6.3 GL-E-3xxx: Policy Errors (POLICY_DENIED)

| Code | Description | Fix |
|------|-------------|-----|
| `GL-E-OPA-DENY` | OPA policy denied | Review policy, request exception |
| `GL-E-YAML-POLICY` | YAML rule violated | Modify pipeline to comply |
| `GL-E-RBAC-DENIED` | RBAC permission denied | Request required permission |
| `GL-E-TENANT-ISOLATION` | Cross-tenant access | Use resources in your tenant |
| `GL-E-DATA-CLASSIFICATION` | Data classification violation | Use appropriate data classification |

### 6.4 GL-E-4xxx: Infrastructure Errors (INFRASTRUCTURE / RESOURCE)

| Code | Description | Retry | Fix |
|------|-------------|-------|-----|
| `GL-E-K8S-JOB-OOM` | Out of memory (137) | Yes (bounded) | Increase memory limit |
| `GL-E-K8S-JOB-TIMEOUT` | Job deadline exceeded | Yes (bounded) | Increase timeout |
| `GL-E-K8S-IMAGEPULL` | Image pull failed | Yes (exponential) | Check image, credentials |
| `GL-E-K8S-EVICTION` | Pod evicted | Yes | Scale cluster, use QoS |
| `GL-E-K8S-QUOTA` | Quota exceeded | Yes (bounded) | Request quota increase |
| `GL-E-K8S-SCHEDULING` | Cannot schedule | Yes (exponential) | Reduce resources, scale cluster |
| `GL-E-S3-ACCESS-DENIED` | S3 permission denied | No | Check IAM permissions |
| `GL-E-S3-NOT-FOUND` | S3 object not found | No | Verify path |
| `GL-E-S3-TIMEOUT` | S3 timeout | Yes (exponential) | Transient |
| `GL-E-DB-CONNECTION` | Database connection failed | Yes (exponential) | Check DB health |
| `GL-E-DB-TIMEOUT` | Database timeout | Yes (exponential) | Optimize query, check DB |

### 6.5 GL-E-5xxx: Internal Errors (AGENT_BUG)

| Code | Description | Fix |
|------|-------------|-----|
| `GL-E-INTERNAL` | Internal orchestrator error | Contact support |
| `GL-E-ASSERTION` | Assertion failed | Report bug |
| `GL-E-UNEXPECTED` | Unexpected error | Retry, then contact support |
| `GL-E-PROVENANCE-MISSING` | Missing provenance data | Fix agent to include provenance |
| `GL-E-PROVENANCE-MISMATCH` | Hash mismatch | Check data integrity |
| `GL-E-AUDIT-CHAIN-BROKEN` | Audit chain compromised | CRITICAL: Contact security |

---

## 7. Emergency Procedures

### 7.1 Kill All Running Jobs

**Use when:** System overload, runaway jobs, security incident

```bash
#!/bin/bash
# emergency-kill-all.sh

NAMESPACE=${1:-greenlang}

echo "WARNING: This will kill ALL running GreenLang jobs in namespace: $NAMESPACE"
read -p "Type 'CONFIRM' to proceed: " confirmation

if [ "$confirmation" != "CONFIRM" ]; then
    echo "Aborted."
    exit 1
fi

# 1. Stop orchestrator from starting new jobs
kubectl scale deployment greenlang-orchestrator -n $NAMESPACE --replicas=0

# 2. Cancel all runs via API (if orchestrator still responsive)
kubectl exec -n $NAMESPACE deploy/greenlang-orchestrator -- \
    curl -X POST localhost:8080/api/v1/admin/emergency-stop 2>/dev/null || true

# 3. Delete all agent jobs
kubectl delete jobs -l app.kubernetes.io/name=greenlang-agent -n $NAMESPACE --force --grace-period=0

# 4. Kill any remaining pods
kubectl delete pods -l app.kubernetes.io/name=greenlang-agent -n $NAMESPACE --force --grace-period=0

# 5. Clear Redis queue
kubectl exec -n $NAMESPACE deploy/redis -- redis-cli FLUSHDB

echo "Emergency stop complete. All jobs killed."
echo "Orchestrator is scaled to 0. Scale up manually when ready."
```

### 7.2 Rollback to Previous Version

```bash
#!/bin/bash
# emergency-rollback.sh

NAMESPACE=${1:-greenlang}
REVISION=${2:-""}  # Leave empty for previous revision

echo "Rolling back greenlang-orchestrator..."

if [ -z "$REVISION" ]; then
    # Rollback to previous
    helm rollback greenlang-orchestrator -n $NAMESPACE
else
    # Rollback to specific revision
    helm rollback greenlang-orchestrator $REVISION -n $NAMESPACE
fi

# Wait for rollout
kubectl rollout status deployment/greenlang-orchestrator -n $NAMESPACE --timeout=5m

# Verify health
kubectl exec -n $NAMESPACE deploy/greenlang-orchestrator -- \
    curl -s localhost:8080/health/ready | jq

echo "Rollback complete."
```

### 7.3 Disable Policy Enforcement

**Use when:** Policy engine causing false positives, emergency bypass needed

**WARNING:** This bypasses security controls. Use only in emergencies.

```bash
#!/bin/bash
# emergency-disable-policies.sh

NAMESPACE=${1:-greenlang}

echo "WARNING: Disabling policy enforcement. This bypasses security controls."
read -p "Type 'BYPASS-SECURITY' to proceed: " confirmation

if [ "$confirmation" != "BYPASS-SECURITY" ]; then
    echo "Aborted."
    exit 1
fi

# Create audit record
kubectl exec -n $NAMESPACE deploy/greenlang-orchestrator -- \
    curl -X POST localhost:8080/api/v1/admin/audit-log \
    -H "Content-Type: application/json" \
    -d '{"action": "POLICY_BYPASS_ENABLED", "user": "'$(whoami)'", "reason": "emergency"}'

# Disable policy enforcement
kubectl set env deployment/greenlang-orchestrator -n $NAMESPACE \
    GL_OPA_ENABLED=false \
    GL_YAML_RULES_ENABLED=false \
    GL_POLICY_DEFAULT_ACTION=allow

# Restart to apply
kubectl rollout restart deployment/greenlang-orchestrator -n $NAMESPACE
kubectl rollout status deployment/greenlang-orchestrator -n $NAMESPACE

echo "Policy enforcement DISABLED."
echo "Remember to re-enable after emergency is resolved!"
echo ""
echo "To re-enable:"
echo "  kubectl set env deployment/greenlang-orchestrator -n $NAMESPACE \\"
echo "      GL_OPA_ENABLED=true GL_YAML_RULES_ENABLED=true GL_POLICY_DEFAULT_ACTION=deny"
```

### 7.4 Emergency Tenant Isolation

**Use when:** Tenant compromise, data breach, abuse detection

```bash
#!/bin/bash
# emergency-isolate-tenant.sh

TENANT_ID=${1:?Usage: $0 <tenant_id>}
NAMESPACE=${2:-greenlang}

echo "Isolating tenant: $TENANT_ID"

# 1. Kill all tenant's running jobs
kubectl delete jobs -l greenlang.io/tenant-id=$TENANT_ID -n $NAMESPACE --force

# 2. Block tenant at network policy level
cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: isolate-tenant-$TENANT_ID
  namespace: $NAMESPACE
spec:
  podSelector:
    matchLabels:
      greenlang.io/tenant-id: $TENANT_ID
  policyTypes:
    - Ingress
    - Egress
  # Empty ingress/egress = block all traffic
EOF

# 3. Add tenant to blocklist
kubectl exec -n $NAMESPACE deploy/greenlang-orchestrator -- \
    curl -X POST localhost:8080/api/v1/admin/tenant-blocklist \
    -H "Content-Type: application/json" \
    -d '{"tenant_id": "'$TENANT_ID'", "reason": "emergency_isolation", "blocked_by": "'$(whoami)'"}'

# 4. Revoke tenant's S3 access (if using tenant-specific roles)
# aws iam put-user-policy ... (depends on IAM setup)

# 5. Create incident record
kubectl exec -n $NAMESPACE deploy/greenlang-orchestrator -- \
    curl -X POST localhost:8080/api/v1/admin/audit-log \
    -H "Content-Type: application/json" \
    -d '{"action": "TENANT_ISOLATED", "tenant_id": "'$TENANT_ID'", "user": "'$(whoami)'"}'

echo "Tenant $TENANT_ID has been isolated."
echo ""
echo "To remove isolation:"
echo "  kubectl delete networkpolicy isolate-tenant-$TENANT_ID -n $NAMESPACE"
echo "  curl -X DELETE localhost:8080/api/v1/admin/tenant-blocklist/$TENANT_ID"
```

---

## Appendix A: Quick Reference Commands

```bash
# Check orchestrator status
kubectl get pods -l app=greenlang-orchestrator -n greenlang

# View orchestrator logs
kubectl logs -l app=greenlang-orchestrator -n greenlang --tail=100 -f

# Check active runs
curl -s http://orchestrator:8080/api/v1/runs?status=running | jq

# Get run details
curl -s http://orchestrator:8080/api/v1/runs/{run_id} | jq

# Cancel a run
curl -X POST http://orchestrator:8080/api/v1/runs/{run_id}:cancel

# Check queue depth
redis-cli -h redis LLEN greenlang:run_queue

# Verify audit chain
curl -s http://orchestrator:8080/api/v1/runs/{run_id}/audit-package | jq '.chain_valid'

# Health check
curl -s http://orchestrator:8080/health/ready | jq

# Prometheus metrics
curl -s http://orchestrator:9090/metrics | grep gl_
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 2.0.0 | 2026-01-27 | GreenLang Platform Team | Initial GLIP v1 runbook |

---

**For additional support:**
- Slack: #greenlang-platform
- PagerDuty: GreenLang Platform On-Call
- Email: platform-support@greenlang.io
