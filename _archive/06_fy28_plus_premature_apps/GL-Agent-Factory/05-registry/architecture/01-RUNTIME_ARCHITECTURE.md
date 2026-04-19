# Agent Runtime Architecture

**Version:** 1.0.0
**Status:** PRODUCTION
**Owner:** GL-DevOpsEngineer
**Last Updated:** 2025-12-03

---

## Executive Summary

The Agent Runtime Environment is the production infrastructure that executes agents with proper isolation, monitoring, resource management, and SLO enforcement. It integrates deeply with Kubernetes for orchestration, Prometheus/Grafana for observability, and the Registry for agent lifecycle management.

**Key Capabilities:**
- Kubernetes-native agent deployment and orchestration
- Multi-tenant isolation with namespace-based segregation
- Automatic resource scaling (HPA/VPA)
- Comprehensive observability and monitoring
- SLO-driven auto-scaling and auto-remediation
- Zero-downtime deployments with canary and blue-green strategies
- Integration with Registry for version management

---

## Runtime Architecture Overview

```
┌───────────────────────────────────────────────────────────────┐
│                     Ingress Layer                              │
│  (NGINX Ingress Controller + TLS Termination + Rate Limiting) │
└────────────────────────┬──────────────────────────────────────┘
                         │
┌────────────────────────┴──────────────────────────────────────┐
│                   API Gateway Layer                            │
│  (Authentication, Authorization, Request Routing, Telemetry)   │
└────────────────────────┬──────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼──────┐  ┌──────▼──────┐  ┌─────▼──────┐
│   Tenant A   │  │   Tenant B   │  │  Tenant C  │
│  Namespace   │  │  Namespace   │  │ Namespace  │
├──────────────┤  ├─────────────┤  ├────────────┤
│ Agent Pods   │  │ Agent Pods  │  │ Agent Pods │
│ - CBAM Calc  │  │ - CSRD App  │  │ - Custom   │
│ - VCCI App   │  │ - Custom    │  │            │
└──────┬───────┘  └──────┬──────┘  └──────┬─────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
┌────────────────────────▼──────────────────────────────────────┐
│                   Shared Services Layer                        │
├───────────────────────────────────────────────────────────────┤
│  • PostgreSQL (RDS Multi-AZ)                                  │
│  • Redis Cluster (ElastiCache)                                │
│  • S3 (Data Storage)                                          │
│  • Secrets Manager (Credentials)                              │
│  • CloudWatch / Prometheus (Monitoring)                       │
└───────────────────────────────────────────────────────────────┘
```

---

## Kubernetes Integration

### Namespace Strategy

Each tenant gets dedicated namespaces for environment isolation:

```yaml
namespace_structure:
  tenant_a:
    development: "tenant-a-dev"
    staging: "tenant-a-staging"
    production: "tenant-a-prod"

  tenant_b:
    development: "tenant-b-dev"
    staging: "tenant-b-staging"
    production: "tenant-b-prod"

namespace_labels:
  tenant-a-prod:
    tenant-id: "customer-abc-123"
    environment: "production"
    sla-tier: "premium"
    data-residency: "eu-central-1"
```

### Agent Deployment Manifest

```yaml
# deployment.yaml - Agent Deployment Template
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-cbam-calculator-v2
  namespace: tenant-a-prod
  labels:
    app: gl-cbam-calculator-v2
    version: "2.3.1"
    agent-id: "gl-cbam-calculator-v2"
    lifecycle-state: "certified"
    tenant-id: "customer-abc-123"
  annotations:
    greenlang.io/agent-id: "gl-cbam-calculator-v2"
    greenlang.io/version: "2.3.1"
    greenlang.io/registry-url: "https://registry.greenlang.ai/agents/gl-cbam-calculator-v2/2.3.1"
    greenlang.io/promoted-at: "2025-11-15T12:00:00Z"

spec:
  replicas: 3
  revisionHistoryLimit: 10

  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0  # Zero downtime

  selector:
    matchLabels:
      app: gl-cbam-calculator-v2
      version: "2.3.1"

  template:
    metadata:
      labels:
        app: gl-cbam-calculator-v2
        version: "2.3.1"
        agent-id: "gl-cbam-calculator-v2"
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"

    spec:
      serviceAccountName: gl-agent-service-account

      # Security Context
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault

      # Init Container - Pre-flight checks
      initContainers:
      - name: init-check-dependencies
        image: gcr.io/greenlang/init-checker:1.0.0
        command: ["/bin/sh", "-c"]
        args:
          - |
            echo "Checking database connectivity..."
            until pg_isready -h $DATABASE_HOST -p 5432; do
              echo "Waiting for database..."
              sleep 2
            done
            echo "Checking Redis connectivity..."
            until redis-cli -h $REDIS_HOST ping; do
              echo "Waiting for Redis..."
              sleep 2
            done
            echo "Dependencies ready!"
        env:
        - name: DATABASE_HOST
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: database-host
        - name: REDIS_HOST
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: redis-host

      # Main Container
      containers:
      - name: agent
        image: gcr.io/greenlang/cbam-calculator:2.3.1
        imagePullPolicy: IfNotPresent

        ports:
        - name: http
          containerPort: 8000
          protocol: TCP
        - name: metrics
          containerPort: 9090
          protocol: TCP

        # Environment Variables
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: AGENT_ID
          value: "gl-cbam-calculator-v2"
        - name: AGENT_VERSION
          value: "2.3.1"
        - name: TENANT_ID
          value: "customer-abc-123"

        # Secrets
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: redis-url
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: anthropic-api-key

        # Resource Limits
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"

        # Health Checks
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
          successThreshold: 1

        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
          successThreshold: 1

        # Startup Probe (for slow-starting apps)
        startupProbe:
          httpGet:
            path: /health/startup
            port: 8000
          initialDelaySeconds: 0
          periodSeconds: 10
          timeoutSeconds: 3
          failureThreshold: 30  # 5 minutes max startup time

        # Volume Mounts
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
        - name: cache
          mountPath: /app/cache

      # Volumes
      volumes:
      - name: config
        configMap:
          name: agent-config
      - name: cache
        emptyDir:
          sizeLimit: 1Gi

      # Node Selection
      nodeSelector:
        workload-type: "agents"
        environment: "production"

      # Affinity - Spread across availability zones
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - gl-cbam-calculator-v2
              topologyKey: topology.kubernetes.io/zone

      # Tolerations
      tolerations:
      - key: "workload-type"
        operator: "Equal"
        value: "agents"
        effect: "NoSchedule"

---
# service.yaml - Agent Service
apiVersion: v1
kind: Service
metadata:
  name: gl-cbam-calculator-v2
  namespace: tenant-a-prod
  labels:
    app: gl-cbam-calculator-v2
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"

spec:
  type: ClusterIP
  selector:
    app: gl-cbam-calculator-v2
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: 9090
    protocol: TCP

  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 3600
```

---

## Auto-Scaling

### Horizontal Pod Autoscaler (HPA)

```yaml
# hpa.yaml - Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gl-cbam-calculator-v2-hpa
  namespace: tenant-a-prod

spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gl-cbam-calculator-v2

  minReplicas: 3
  maxReplicas: 50

  metrics:
  # CPU-based scaling
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70

  # Memory-based scaling
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80

  # Custom metric: Request rate
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"

  # Custom metric: Queue depth
  - type: Pods
    pods:
      metric:
        name: queue_depth
      target:
        type: AverageValue
        averageValue: "100"

  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # 5 minutes
      policies:
      - type: Percent
        value: 50  # Scale down max 50% at a time
        periodSeconds: 60
      - type: Pods
        value: 5  # Scale down max 5 pods at a time
        periodSeconds: 60
      selectPolicy: Min  # Use most conservative policy

    scaleUp:
      stabilizationWindowSeconds: 0  # Scale up immediately
      policies:
      - type: Percent
        value: 100  # Scale up max 100% at a time
        periodSeconds: 30
      - type: Pods
        value: 10  # Scale up max 10 pods at a time
        periodSeconds: 30
      selectPolicy: Max  # Use most aggressive policy
```

### Vertical Pod Autoscaler (VPA)

```yaml
# vpa.yaml - Vertical Pod Autoscaler (recommendations only)
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: gl-cbam-calculator-v2-vpa
  namespace: tenant-a-prod

spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gl-cbam-calculator-v2

  updatePolicy:
    updateMode: "Off"  # Recommendation mode only

  resourcePolicy:
    containerPolicies:
    - containerName: agent
      minAllowed:
        cpu: "250m"
        memory: "256Mi"
      maxAllowed:
        cpu: "4000m"
        memory: "8Gi"
      controlledResources:
      - cpu
      - memory
```

---

## Resource Isolation and Quotas

### Namespace Resource Quotas

```yaml
# resource-quota.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: tenant-a-prod-quota
  namespace: tenant-a-prod

spec:
  hard:
    # Compute resources
    requests.cpu: "100"
    requests.memory: "200Gi"
    limits.cpu: "200"
    limits.memory: "400Gi"

    # Object counts
    pods: "500"
    services: "50"
    configmaps: "100"
    secrets: "100"
    persistentvolumeclaims: "50"

    # Storage
    requests.storage: "1Ti"
```

### Limit Ranges

```yaml
# limit-range.yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: tenant-a-prod-limits
  namespace: tenant-a-prod

spec:
  limits:
  # Default limits for containers
  - type: Container
    default:
      cpu: "2000m"
      memory: "2Gi"
    defaultRequest:
      cpu: "500m"
      memory: "512Mi"
    max:
      cpu: "8000m"
      memory: "16Gi"
    min:
      cpu: "100m"
      memory: "128Mi"

  # Default limits for pods
  - type: Pod
    max:
      cpu: "16000m"
      memory: "32Gi"
    min:
      cpu: "100m"
      memory: "128Mi"

  # Persistent volume claims
  - type: PersistentVolumeClaim
    max:
      storage: "100Gi"
    min:
      storage: "1Gi"
```

### Network Policies

```yaml
# network-policy.yaml - Tenant Isolation
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: tenant-a-prod-network-policy
  namespace: tenant-a-prod

spec:
  podSelector: {}  # Apply to all pods in namespace

  policyTypes:
  - Ingress
  - Egress

  ingress:
  # Allow from ingress controller
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000

  # Allow from same namespace (pod-to-pod)
  - from:
    - podSelector: {}
    ports:
    - protocol: TCP
      port: 8000
    - protocol: TCP
      port: 9090

  egress:
  # Allow to PostgreSQL
  - to:
    - namespaceSelector:
        matchLabels:
          name: database
    ports:
    - protocol: TCP
      port: 5432

  # Allow to Redis
  - to:
    - namespaceSelector:
        matchLabels:
          name: cache
    ports:
    - protocol: TCP
      port: 6379

  # Allow to external LLM providers
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 443
    # Add egress rules for api.anthropic.com, api.openai.com

  # DNS resolution
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53
```

---

## Monitoring and Observability

### Prometheus Metrics

```python
# agent_metrics.py - Agent Metrics Export
from prometheus_client import Counter, Histogram, Gauge, Info

# Request metrics
REQUEST_COUNT = Counter(
    'agent_requests_total',
    'Total number of agent requests',
    ['agent_id', 'version', 'tenant_id', 'capability', 'status']
)

REQUEST_DURATION = Histogram(
    'agent_request_duration_seconds',
    'Agent request duration in seconds',
    ['agent_id', 'version', 'tenant_id', 'capability'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Error metrics
ERROR_COUNT = Counter(
    'agent_errors_total',
    'Total number of agent errors',
    ['agent_id', 'version', 'tenant_id', 'error_type']
)

# Performance metrics
LATENCY_P50 = Gauge(
    'agent_latency_p50_seconds',
    'Agent P50 latency',
    ['agent_id', 'version', 'tenant_id']
)

LATENCY_P95 = Gauge(
    'agent_latency_p95_seconds',
    'Agent P95 latency',
    ['agent_id', 'version', 'tenant_id']
)

LATENCY_P99 = Gauge(
    'agent_latency_p99_seconds',
    'Agent P99 latency',
    ['agent_id', 'version', 'tenant_id']
)

# Resource metrics
MEMORY_USAGE = Gauge(
    'agent_memory_usage_bytes',
    'Agent memory usage in bytes',
    ['agent_id', 'version', 'tenant_id']
)

CPU_USAGE = Gauge(
    'agent_cpu_usage_percent',
    'Agent CPU usage percentage',
    ['agent_id', 'version', 'tenant_id']
)

# LLM metrics
LLM_TOKENS_USED = Counter(
    'agent_llm_tokens_total',
    'Total LLM tokens used',
    ['agent_id', 'version', 'tenant_id', 'provider', 'model']
)

LLM_COST = Counter(
    'agent_llm_cost_usd',
    'LLM cost in USD',
    ['agent_id', 'version', 'tenant_id', 'provider', 'model']
)

# Agent metadata
AGENT_INFO = Info(
    'agent_info',
    'Agent metadata',
    ['agent_id', 'version', 'lifecycle_state', 'tenant_id']
)
```

### Prometheus ServiceMonitor

```yaml
# servicemonitor.yaml - Prometheus Scrape Config
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: gl-cbam-calculator-v2
  namespace: tenant-a-prod
  labels:
    prometheus: kube-prometheus

spec:
  selector:
    matchLabels:
      app: gl-cbam-calculator-v2

  endpoints:
  - port: metrics
    interval: 15s
    path: /metrics
    scheme: http

    relabelings:
    - sourceLabels: [__meta_kubernetes_pod_label_agent_id]
      targetLabel: agent_id
    - sourceLabels: [__meta_kubernetes_pod_label_version]
      targetLabel: version
    - sourceLabels: [__meta_kubernetes_pod_label_tenant_id]
      targetLabel: tenant_id
    - sourceLabels: [__meta_kubernetes_namespace]
      targetLabel: namespace
```

### Grafana Dashboards

```yaml
grafana_dashboards:
  agent_performance_dashboard:
    panels:
      - title: "Request Rate"
        query: 'rate(agent_requests_total{agent_id="gl-cbam-calculator-v2"}[5m])'
        type: "graph"

      - title: "P95 Latency"
        query: 'histogram_quantile(0.95, agent_request_duration_seconds_bucket{agent_id="gl-cbam-calculator-v2"})'
        type: "graph"
        slo_threshold: 0.5  # 500ms

      - title: "Error Rate"
        query: 'rate(agent_errors_total{agent_id="gl-cbam-calculator-v2"}[5m]) / rate(agent_requests_total{agent_id="gl-cbam-calculator-v2"}[5m])'
        type: "graph"
        slo_threshold: 0.01  # 1%

      - title: "CPU Usage"
        query: 'sum(rate(container_cpu_usage_seconds_total{pod=~"gl-cbam-calculator.*"}[5m])) by (pod)'
        type: "graph"

      - title: "Memory Usage"
        query: 'sum(container_memory_usage_bytes{pod=~"gl-cbam-calculator.*"}) by (pod)'
        type: "graph"

      - title: "Pod Count"
        query: 'count(kube_pod_info{pod=~"gl-cbam-calculator.*"})'
        type: "stat"

      - title: "LLM Token Usage (30d)"
        query: 'sum(increase(agent_llm_tokens_total{agent_id="gl-cbam-calculator-v2"}[30d]))'
        type: "stat"

      - title: "LLM Cost (30d)"
        query: 'sum(increase(agent_llm_cost_usd{agent_id="gl-cbam-calculator-v2"}[30d]))'
        type: "stat"
```

---

## SLO Enforcement

### SLO Definitions (from PERFORMANCE_SLOS.md)

```yaml
agent_runtime_slos:
  availability:
    certified_agents:
      target: "99.99%"
      measurement_window: "30 days"
      error_budget: "4.32 minutes/month"

  latency:
    p50_target: "< 100ms"
    p95_target: "< 500ms"
    p99_target: "< 2000ms"
    measurement_window: "5 minutes"

  error_rate:
    target: "< 0.5%"
    measurement_window: "5 minutes"

  throughput:
    min_throughput: "> 1000 req/sec"
    measurement_window: "1 minute"
```

### PrometheusRules for SLO Alerting

```yaml
# prometheus-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: agent-slo-alerts
  namespace: monitoring

spec:
  groups:
  - name: agent_slo_alerts
    interval: 30s
    rules:

    # High P95 Latency
    - alert: AgentHighP95Latency
      expr: |
        histogram_quantile(0.95,
          rate(agent_request_duration_seconds_bucket[5m])
        ) > 0.5
      for: 5m
      labels:
        severity: warning
        slo: latency
      annotations:
        summary: "Agent {{ $labels.agent_id }} P95 latency exceeds SLO"
        description: "P95 latency is {{ $value }}s (SLO: <0.5s)"

    # High Error Rate
    - alert: AgentHighErrorRate
      expr: |
        (
          rate(agent_errors_total[5m])
          /
          rate(agent_requests_total[5m])
        ) > 0.005
      for: 5m
      labels:
        severity: critical
        slo: error_rate
      annotations:
        summary: "Agent {{ $labels.agent_id }} error rate exceeds SLO"
        description: "Error rate is {{ $value | humanizePercentage }} (SLO: <0.5%)"

    # Low Availability
    - alert: AgentLowAvailability
      expr: |
        (
          sum(rate(agent_requests_total{status="success"}[30d]))
          /
          sum(rate(agent_requests_total[30d]))
        ) < 0.9999
      for: 5m
      labels:
        severity: critical
        slo: availability
      annotations:
        summary: "Agent {{ $labels.agent_id }} availability below SLO"
        description: "Availability is {{ $value | humanizePercentage }} (SLO: >99.99%)"

    # Pod Crash Loop
    - alert: AgentPodCrashLoop
      expr: |
        rate(kube_pod_container_status_restarts_total{pod=~"gl-.*"}[15m]) > 0
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Agent pod {{ $labels.pod }} is crash looping"
        description: "Pod has restarted {{ $value }} times in 15 minutes"

    # Memory Pressure
    - alert: AgentMemoryPressure
      expr: |
        (
          container_memory_usage_bytes{pod=~"gl-.*"}
          /
          container_spec_memory_limit_bytes{pod=~"gl-.*"}
        ) > 0.90
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "Agent pod {{ $labels.pod }} memory pressure"
        description: "Memory usage is {{ $value | humanizePercentage }} of limit"

    # CPU Throttling
    - alert: AgentCPUThrottling
      expr: |
        rate(container_cpu_cfs_throttled_seconds_total{pod=~"gl-.*"}[5m])
        /
        rate(container_cpu_cfs_periods_total{pod=~"gl-.*"}[5m])
        > 0.25
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "Agent pod {{ $labels.pod }} experiencing CPU throttling"
        description: "{{ $value | humanizePercentage }} of CPU time is throttled"
```

### Auto-Remediation

```yaml
auto_remediation:
  high_latency:
    trigger: "p95_latency > 2 × SLO for 5 minutes"
    actions:
      - "Scale up pods (HPA)"
      - "Alert on-call engineer"
      - "Check for external dependencies (DB, Redis, LLM)"

  high_error_rate:
    trigger: "error_rate > 5% for 2 minutes"
    actions:
      - "Auto-rollback to previous version"
      - "Page on-call engineer"
      - "Create incident in PagerDuty"

  pod_crash_loop:
    trigger: "Pod restarts > 3 in 10 minutes"
    actions:
      - "Capture pod logs and heap dump"
      - "Mark pod unhealthy"
      - "Page on-call engineer"
      - "Create postmortem task"

  resource_exhaustion:
    trigger: "memory > 95% or CPU throttling > 50%"
    actions:
      - "Scale up pods (HPA)"
      - "Create VPA recommendation"
      - "Alert resource optimization team"
```

---

## Deployment Strategies

### Rolling Update (Default)

```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 1  # Add 1 pod before removing old pods
    maxUnavailable: 0  # Zero downtime

workflow:
  1. Create new pod with new version
  2. Wait for new pod to be ready (readiness probe)
  3. Add new pod to service endpoints
  4. Remove one old pod
  5. Repeat until all pods updated
```

### Blue-Green Deployment

```yaml
blue_green_deployment:
  workflow:
    1. Deploy new version (green) alongside old version (blue)
    2. Test green deployment (smoke tests)
    3. Switch traffic from blue to green (update service selector)
    4. Monitor green deployment (5-10 minutes)
    5. If successful, delete blue deployment
    6. If issues detected, rollback to blue

  example:
    # Blue (current production)
    - deployment: gl-cbam-calculator-v2-blue
      version: "2.3.0"
      replicas: 10
      service_selector: "version: 2.3.0"

    # Green (new version)
    - deployment: gl-cbam-calculator-v2-green
      version: "2.3.1"
      replicas: 10
      service_selector: "version: 2.3.1"

    # Traffic switch
    - update_service_selector: "version: 2.3.1"
```

### Canary Deployment

```yaml
canary_deployment:
  workflow:
    1. Deploy canary version (5% of pods)
    2. Route 5% of traffic to canary
    3. Monitor canary metrics (error rate, latency)
    4. If metrics good, increase to 25%
    5. If metrics good, increase to 50%
    6. If metrics good, increase to 100%
    7. Delete old version

  example:
    # Stable (95% of traffic)
    - deployment: gl-cbam-calculator-v2-stable
      version: "2.3.0"
      replicas: 19
      weight: 95

    # Canary (5% of traffic)
    - deployment: gl-cbam-calculator-v2-canary
      version: "2.3.1"
      replicas: 1
      weight: 5

  progression:
    - stage: 1
      canary_weight: 5
      duration: 10m
      success_criteria:
        - error_rate_canary < error_rate_stable * 1.1
        - latency_p95_canary < latency_p95_stable * 1.2

    - stage: 2
      canary_weight: 25
      duration: 20m
      success_criteria: [same as stage 1]

    - stage: 3
      canary_weight: 50
      duration: 30m
      success_criteria: [same as stage 1]

    - stage: 4
      canary_weight: 100
      duration: null
      success_criteria: null
```

### Deployment Automation with Argo CD

```yaml
# argocd-application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: gl-cbam-calculator-v2
  namespace: argocd

spec:
  project: default

  source:
    repoURL: https://github.com/greenlang/agents
    targetRevision: HEAD
    path: agents/gl-cbam-calculator-v2/manifests

  destination:
    server: https://kubernetes.default.svc
    namespace: tenant-a-prod

  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    syncOptions:
    - CreateNamespace=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m

  # Health assessment
  ignoreDifferences:
  - group: apps
    kind: Deployment
    jsonPointers:
    - /spec/replicas  # Ignore replica count (managed by HPA)
```

---

## Integration with Registry

### Runtime-Registry Communication

```python
class RuntimeOrchestrator:
    """Orchestrate agent deployment from Registry to Runtime"""

    def deploy_agent(
        self,
        agent_id: str,
        version: str,
        environment: str,
        tenant_id: str
    ) -> DeploymentResult:
        # 1. Fetch agent metadata from Registry
        agent_metadata = registry_client.get_agent_version(agent_id, version)

        # 2. Validate governance policies
        policy_check = governance_engine.evaluate_policy(
            agent_id, version, environment, tenant_id, "deploy"
        )
        if not policy_check.allowed:
            raise PolicyViolationError(policy_check.failed_rules)

        # 3. Generate Kubernetes manifests
        manifests = self.generate_manifests(
            agent_metadata, environment, tenant_id
        )

        # 4. Apply manifests to cluster
        kubernetes_client.apply(manifests)

        # 5. Wait for deployment to be ready
        deployment_ready = self.wait_for_deployment(
            agent_id, version, tenant_id, timeout=300
        )

        # 6. Register deployment in Registry
        registry_client.register_deployment(
            agent_id, version, environment, tenant_id
        )

        # 7. Enable monitoring
        monitoring_client.enable_monitoring(agent_id, version, tenant_id)

        return DeploymentResult(
            success=deployment_ready,
            agent_id=agent_id,
            version=version,
            environment=environment,
            tenant_id=tenant_id
        )
```

---

## Cost Tracking and Optimization

### Cost Metrics

```yaml
cost_tracking:
  compute_costs:
    metric: 'sum(rate(container_cpu_usage_seconds_total[1h])) * cpu_cost_per_hour'
    aggregation: "by tenant_id"

  memory_costs:
    metric: 'sum(container_memory_usage_bytes[1h]) * memory_cost_per_gb_hour'
    aggregation: "by tenant_id"

  llm_costs:
    metric: 'sum(agent_llm_cost_usd)'
    aggregation: "by tenant_id, agent_id"

  storage_costs:
    metric: 'sum(kubelet_volume_stats_used_bytes) * storage_cost_per_gb_month'
    aggregation: "by tenant_id"

cost_optimization:
  strategies:
    - "Right-size pod resources based on VPA recommendations"
    - "Use spot instances for non-production workloads"
    - "Scale down non-production environments outside business hours"
    - "Optimize LLM token usage with caching and prompt engineering"
    - "Use reserved instances for baseline production capacity"
```

---

## Disaster Recovery

### Backup Strategy

```yaml
backup_strategy:
  database:
    method: "RDS automated backups"
    frequency: "Daily at 2am UTC"
    retention: "30 days"
    point_in_time_recovery: "5 minute granularity"

  redis:
    method: "ElastiCache snapshots"
    frequency: "Daily at 3am UTC"
    retention: "7 days"

  kubernetes_state:
    method: "Velero snapshots"
    frequency: "Every 6 hours"
    retention: "30 days"
    includes:
      - "All namespaces"
      - "PersistentVolumes"
      - "ConfigMaps and Secrets"

  s3_data:
    method: "S3 versioning + cross-region replication"
    retention: "Indefinite"
    replication_target: "us-west-2"
```

### Disaster Recovery Plan

```yaml
disaster_recovery:
  rto: "1 hour"  # Recovery Time Objective
  rpo: "5 minutes"  # Recovery Point Objective

  scenarios:
    region_failure:
      trigger: "Primary region (eu-central-1) unavailable"
      procedure:
        1. "Detect region failure (automated monitoring)"
        2. "Fail over DNS to secondary region (eu-west-1)"
        3. "Restore database from latest backup"
        4. "Restore Kubernetes state from Velero"
        5. "Verify all services operational"
        6. "Notify stakeholders"
      expected_duration: "45 minutes"

    database_failure:
      trigger: "RDS primary instance failure"
      procedure:
        1. "RDS automatic failover to standby (1-2 minutes)"
        2. "Verify application connectivity"
        3. "Monitor for replication lag"
      expected_duration: "2 minutes"

    complete_cluster_failure:
      trigger: "Kubernetes cluster unavailable"
      procedure:
        1. "Provision new EKS cluster"
        2. "Restore cluster state from Velero"
        3. "Restore database and Redis from backups"
        4. "Redeploy applications"
        5. "Update DNS records"
      expected_duration: "60 minutes"
```

---

## Best Practices

### For Agent Developers

1. **Implement Health Checks** - Provide `/health/live`, `/health/ready`, `/health/startup`
2. **Export Prometheus Metrics** - Expose `/metrics` endpoint
3. **Graceful Shutdown** - Handle SIGTERM signal properly
4. **Resource Requests** - Specify accurate CPU/memory requests
5. **Stateless Design** - Design agents to be stateless for easy scaling

### For Platform Operators

1. **Monitor SLOs** - Track SLOs for all certified agents
2. **Automate Deployment** - Use GitOps (Argo CD) for deployments
3. **Cost Optimization** - Regular review of resource utilization
4. **Disaster Recovery Testing** - Quarterly DR drills
5. **Security Scanning** - Continuous container image scanning

---

## Related Documentation

- [Registry Overview](00-REGISTRY_OVERVIEW.md)
- [Registry API Specification](../api-specs/00-REGISTRY_API.md)
- [Agent Lifecycle Management](../lifecycle/00-AGENT_LIFECYCLE.md)
- [Governance Controls](../governance/00-GOVERNANCE_CONTROLS.md)
- [Performance SLOs](C:\Users\aksha\Code-V1_GreenLang\slo\PERFORMANCE_SLOS.md)

---

**Questions or feedback?**
- Slack: #runtime-infrastructure
- Email: runtime@greenlang.ai
- Wiki: https://wiki.greenlang.ai/runtime
