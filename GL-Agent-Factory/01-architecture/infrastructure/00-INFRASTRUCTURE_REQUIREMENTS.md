# GreenLang Agent Factory: Infrastructure Requirements

**Version:** 1.0.0
**Date:** December 3, 2025
**Status:** ARCHITECTURE SPECIFICATION
**Classification:** Technical Architecture Document

---

## Overview

This document defines the infrastructure requirements for the GreenLang Agent Factory, including compute, storage, networking, and integration with existing Kubernetes, monitoring, and metrics infrastructure.

---

## 1. Compute Requirements

### 1.1 Kubernetes Cluster Specifications

```
+===========================================================================+
|                    Kubernetes Cluster Requirements                         |
+===========================================================================+

Production Cluster (Primary):
+------------------+-------------------+-------------------+
| Node Pool        | Instance Type     | Count             |
+------------------+-------------------+-------------------+
| System           | m6i.xlarge        | 3 (HA)            |
| API Gateway      | c6i.2xlarge       | 3-10 (HPA)        |
| Agent Factory    | m6i.2xlarge       | 3-20 (HPA)        |
| Agent Runtime    | c6i.xlarge        | 10-100 (HPA)      |
| Worker           | m6i.xlarge        | 5-50 (HPA)        |
| GPU (Optional)   | g4dn.xlarge       | 0-5 (On-demand)   |
+------------------+-------------------+-------------------+

Total Capacity:
- Minimum: 24 nodes, 96 vCPU, 384GB RAM
- Maximum: 188 nodes, 752 vCPU, 3008GB RAM
- Target Utilization: 70% average
```

### 1.2 Node Specifications

```yaml
# Node Pool Configurations
node_pools:
  system:
    purpose: "Kubernetes system components"
    instance_type: "m6i.xlarge"  # 4 vCPU, 16GB RAM
    min_size: 3
    max_size: 5
    disk_size_gb: 100
    taints:
      - key: "CriticalAddonsOnly"
        value: "true"
        effect: "NoSchedule"

  api_gateway:
    purpose: "API Gateway and ingress"
    instance_type: "c6i.2xlarge"  # 8 vCPU, 16GB RAM
    min_size: 3
    max_size: 10
    disk_size_gb: 50
    labels:
      workload-type: "api"
    node_affinity:
      - "topology.kubernetes.io/zone"

  agent_factory:
    purpose: "Agent Factory services"
    instance_type: "m6i.2xlarge"  # 8 vCPU, 32GB RAM
    min_size: 3
    max_size: 20
    disk_size_gb: 200
    labels:
      workload-type: "factory"

  agent_runtime:
    purpose: "Agent execution runtime"
    instance_type: "c6i.xlarge"  # 4 vCPU, 8GB RAM
    min_size: 10
    max_size: 100
    disk_size_gb: 50
    labels:
      workload-type: "runtime"
    spot_instances: true
    spot_percentage: 60

  worker:
    purpose: "Background workers and batch jobs"
    instance_type: "m6i.xlarge"  # 4 vCPU, 16GB RAM
    min_size: 5
    max_size: 50
    disk_size_gb: 100
    labels:
      workload-type: "worker"
    spot_instances: true
    spot_percentage: 80

  gpu:
    purpose: "ML inference and embedding generation"
    instance_type: "g4dn.xlarge"  # 4 vCPU, 16GB RAM, 1 T4 GPU
    min_size: 0
    max_size: 5
    disk_size_gb: 200
    labels:
      workload-type: "gpu"
    taints:
      - key: "nvidia.com/gpu"
        value: "true"
        effect: "NoSchedule"
```

### 1.3 Resource Requests and Limits

```yaml
# Service Resource Configurations
services:
  api_gateway:
    requests:
      cpu: "500m"
      memory: "512Mi"
    limits:
      cpu: "2000m"
      memory: "2Gi"
    replicas:
      min: 3
      max: 20
      target_cpu: 70

  factory_service:
    requests:
      cpu: "1000m"
      memory: "1Gi"
    limits:
      cpu: "4000m"
      memory: "4Gi"
    replicas:
      min: 3
      max: 50
      target_cpu: 70

  runtime_service:
    requests:
      cpu: "500m"
      memory: "512Mi"
    limits:
      cpu: "2000m"
      memory: "2Gi"
    replicas:
      min: 10
      max: 200
      target_cpu: 70

  registry_service:
    requests:
      cpu: "250m"
      memory: "256Mi"
    limits:
      cpu: "1000m"
      memory: "1Gi"
    replicas:
      min: 3
      max: 10
      target_cpu: 60

  worker_service:
    requests:
      cpu: "500m"
      memory: "512Mi"
    limits:
      cpu: "2000m"
      memory: "2Gi"
    replicas:
      min: 5
      max: 100
      target_cpu: 80
```

### 1.4 Horizontal Pod Autoscaler Configuration

```yaml
# HPA Configurations
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-runtime-hpa
  namespace: greenlang
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-runtime
  minReplicas: 10
  maxReplicas: 200
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
    - type: External
      external:
        metric:
          name: agent_queue_depth
        target:
          type: AverageValue
          averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Pods
          value: 10
          periodSeconds: 60
        - type: Percent
          value: 100
          periodSeconds: 60
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Pods
          value: 5
          periodSeconds: 60
        - type: Percent
          value: 10
          periodSeconds: 60
      selectPolicy: Min
```

---

## 2. Storage Requirements

### 2.1 Storage Architecture

```
+===========================================================================+
|                    Storage Architecture                                    |
+===========================================================================+

                        +------------------+
                        |   Applications   |
                        +--------+---------+
                                 |
         +-----------------------+-----------------------+
         |                       |                       |
+--------v--------+    +--------v--------+    +--------v--------+
|   PostgreSQL    |    |     Redis       |    |   Object Store  |
|   (Relational)  |    |    (Cache)      |    |     (S3)        |
+-----------------+    +-----------------+    +-----------------+
| - Tenant DBs    |    | - Session Cache |    | - Agent Packs   |
| - Agent Meta    |    | - EF Cache      |    | - Audit Logs    |
| - Audit Logs    |    | - Rate Limits   |    | - Reports       |
| - Config        |    | - Pub/Sub       |    | - Backups       |
+-----------------+    +-----------------+    +-----------------+

         +-----------------------+-----------------------+
         |                       |                       |
+--------v--------+    +--------v--------+    +--------v--------+
|   Vector Store  |    |     Kafka       |    |   TimeSeries    |
|  (Embeddings)   |    |   (Events)      |    | (Prometheus)    |
+-----------------+    +-----------------+    +-----------------+
| - RAG Vectors   |    | - Agent Events  |    | - Metrics       |
| - EF Embeddings |    | - Lifecycle     |    | - SLOs          |
| - Doc Chunks    |    | - Telemetry     |    | - Alerts        |
+-----------------+    +-----------------+    +-----------------+
```

### 2.2 PostgreSQL Requirements

```yaml
# PostgreSQL Configuration
postgresql:
  deployment_type: "managed"  # AWS RDS, Azure PostgreSQL, or GCP Cloud SQL
  version: "15"

  primary:
    instance_class: "db.r6g.2xlarge"  # 8 vCPU, 64GB RAM
    storage:
      type: "io2"
      size_gb: 1000
      iops: 10000
      encrypted: true
    multi_az: true
    backup_retention_days: 30
    maintenance_window: "Sun:03:00-Sun:04:00"

  read_replicas:
    count: 3
    instance_class: "db.r6g.xlarge"  # 4 vCPU, 32GB RAM
    regions:
      - "us-east-1"
      - "eu-west-1"
      - "ap-southeast-1"

  connection_pooling:
    provider: "pgbouncer"
    pool_mode: "transaction"
    max_client_connections: 10000
    default_pool_size: 100
    reserve_pool_size: 50

  performance_settings:
    max_connections: 500
    shared_buffers: "16GB"
    effective_cache_size: "48GB"
    work_mem: "256MB"
    maintenance_work_mem: "2GB"
    checkpoint_completion_target: 0.9
    wal_buffers: "64MB"
    max_wal_size: "4GB"
    random_page_cost: 1.1
    effective_io_concurrency: 200

  # Tenant Isolation
  tenant_databases:
    naming_pattern: "greenlang_tenant_{uuid}"
    per_tenant_storage_limit_gb: 100
    per_tenant_connection_limit: 50
```

### 2.3 Redis Requirements

```yaml
# Redis Configuration
redis:
  deployment_type: "cluster"  # Redis Cluster for HA
  version: "7.0"

  cluster:
    node_count: 6  # 3 primaries + 3 replicas
    node_type: "cache.r6g.xlarge"  # 4 vCPU, 26GB RAM
    shards: 3

  memory:
    maxmemory: "20gb"
    maxmemory_policy: "allkeys-lru"

  persistence:
    aof_enabled: true
    aof_fsync: "everysec"
    rdb_enabled: true
    rdb_save_intervals:
      - "900 1"   # 15 min if 1 change
      - "300 10"  # 5 min if 10 changes
      - "60 10000"  # 1 min if 10000 changes

  sentinel:
    enabled: true
    quorum: 2
    down_after_milliseconds: 5000
    failover_timeout: 60000

  connection:
    max_clients: 10000
    timeout: 0
    tcp_keepalive: 300

  cache_tiers:
    l1_memory:
      ttl_seconds: 300
      max_size_mb: 100
    l2_redis:
      ttl_seconds: 3600
      max_size_gb: 10
```

### 2.4 Object Storage Requirements

```yaml
# S3/Object Storage Configuration
object_storage:
  provider: "s3"  # Or MinIO for on-premise

  buckets:
    agent_packs:
      name: "greenlang-agent-packs-{env}"
      versioning: true
      lifecycle_rules:
        - transition_to_ia_days: 90
        - transition_to_glacier_days: 365
        - expiration_days: 2555  # 7 years for compliance

    audit_logs:
      name: "greenlang-audit-logs-{env}"
      versioning: true
      object_lock: true  # WORM for compliance
      retention_days: 2555  # 7 years

    reports:
      name: "greenlang-reports-{env}"
      versioning: true
      lifecycle_rules:
        - transition_to_ia_days: 30

    backups:
      name: "greenlang-backups-{env}"
      versioning: true
      cross_region_replication: true
      lifecycle_rules:
        - transition_to_glacier_days: 30
        - expiration_days: 365

  encryption:
    type: "aws:kms"
    key_rotation: true

  access:
    block_public_access: true
    require_ssl: true
```

### 2.5 Vector Store Requirements

```yaml
# Vector Store Configuration
vector_store:
  development:
    provider: "chromadb"
    persistence: "./chroma_data"
    collection_settings:
      hnsw_space: "cosine"
      hnsw_construction_ef: 100
      hnsw_m: 16

  production:
    provider: "pinecone"
    environment: "us-east-1-aws"
    index_settings:
      metric: "cosine"
      pods: 2
      pod_type: "p2.x1"
      replicas: 2

    namespaces:
      - "emission_factors"
      - "regulatory_docs"
      - "agent_metadata"

    dimensions: 384  # all-MiniLM-L6-v2

  storage_estimates:
    vectors_per_agent: 1000
    bytes_per_vector: 1536  # 384 * 4 bytes
    total_agents: 10000
    total_storage_gb: 15  # Plus metadata overhead
```

### 2.6 Kafka Requirements

```yaml
# Kafka Configuration
kafka:
  deployment_type: "managed"  # AWS MSK, Confluent Cloud
  version: "3.5"

  cluster:
    broker_count: 6
    broker_type: "kafka.m5.2xlarge"  # 8 vCPU, 32GB RAM
    storage_per_broker_gb: 1000

  topics:
    agent_lifecycle:
      partitions: 32
      replication_factor: 3
      retention_ms: 604800000  # 7 days
      cleanup_policy: "delete"

    agent_execution:
      partitions: 100
      replication_factor: 3
      retention_ms: 86400000  # 1 day
      cleanup_policy: "delete"

    agent_metrics:
      partitions: 50
      replication_factor: 3
      retention_ms: 3600000  # 1 hour
      cleanup_policy: "delete"

    audit_log:
      partitions: 20
      replication_factor: 3
      retention_ms: -1  # Infinite (use compaction)
      cleanup_policy: "compact"

  performance:
    num_partitions: 100
    default_replication_factor: 3
    min_insync_replicas: 2
    auto_create_topics: false

  security:
    encryption_at_rest: true
    encryption_in_transit: true
    authentication: "SASL_SSL"
```

---

## 3. Networking Requirements

### 3.1 Network Architecture

```
+===========================================================================+
|                    Network Architecture                                    |
+===========================================================================+

                         Internet
                            |
                     +------v------+
                     |   WAF/DDoS  |
                     |  Protection |
                     +------+------+
                            |
                     +------v------+
                     | Global LB   |
                     | (Route 53)  |
                     +------+------+
                            |
          +-----------------+-----------------+
          |                 |                 |
    +-----v-----+     +-----v-----+     +-----v-----+
    | US-EAST-1 |     | EU-WEST-1 |     | AP-SOUTH-1|
    +-----+-----+     +-----+-----+     +-----+-----+
          |                 |                 |
    +-----v-----+     +-----v-----+     +-----v-----+
    | VPC       |     | VPC       |     | VPC       |
    | 10.0.0.0  |     | 10.1.0.0  |     | 10.2.0.0  |
    +-----+-----+     +-----+-----+     +-----+-----+
          |                 |                 |
    +-----v-----+     +-----v-----+     +-----v-----+
    | Subnets   |     | Subnets   |     | Subnets   |
    | Public    |     | Public    |     | Public    |
    | Private   |     | Private   |     | Private   |
    | Database  |     | Database  |     | Database  |
    +-----------+     +-----------+     +-----------+
```

### 3.2 VPC Configuration

```yaml
# VPC Configuration
vpc:
  cidr_block: "10.0.0.0/16"  # 65,536 IPs per region

  subnets:
    public:
      count: 3  # One per AZ
      cidr_blocks:
        - "10.0.0.0/20"   # 4,096 IPs
        - "10.0.16.0/20"
        - "10.0.32.0/20"
      purpose: "Load balancers, NAT gateways"

    private_app:
      count: 3
      cidr_blocks:
        - "10.0.64.0/19"  # 8,192 IPs
        - "10.0.96.0/19"
        - "10.0.128.0/19"
      purpose: "Application workloads"

    private_data:
      count: 3
      cidr_blocks:
        - "10.0.160.0/20"  # 4,096 IPs
        - "10.0.176.0/20"
        - "10.0.192.0/20"
      purpose: "Databases, caches"

  nat_gateways:
    count: 3  # One per AZ for HA
    elastic_ips: true

  vpc_endpoints:
    - "s3"
    - "ecr.api"
    - "ecr.dkr"
    - "logs"
    - "monitoring"
    - "secretsmanager"
    - "kms"
```

### 3.3 Network Policies

```yaml
# Kubernetes Network Policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: agent-factory-policy
  namespace: greenlang
spec:
  podSelector:
    matchLabels:
      app: agent-factory
  policyTypes:
    - Ingress
    - Egress

  ingress:
    # Allow from API Gateway
    - from:
        - podSelector:
            matchLabels:
              app: api-gateway
      ports:
        - protocol: TCP
          port: 8080

    # Allow from monitoring
    - from:
        - namespaceSelector:
            matchLabels:
              name: monitoring
      ports:
        - protocol: TCP
          port: 9090

  egress:
    # Allow DNS
    - to:
        - namespaceSelector:
            matchLabels:
              name: kube-system
      ports:
        - protocol: UDP
          port: 53

    # Allow to databases
    - to:
        - podSelector:
            matchLabels:
              app: postgresql
      ports:
        - protocol: TCP
          port: 5432

    # Allow to Redis
    - to:
        - podSelector:
            matchLabels:
              app: redis
      ports:
        - protocol: TCP
          port: 6379

    # Allow to external APIs (HTTPS)
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - protocol: TCP
          port: 443
```

### 3.4 Load Balancer Configuration

```yaml
# Ingress Configuration
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: greenlang-ingress
  namespace: greenlang
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/rate-limit: "1000"
    nginx.ingress.kubernetes.io/rate-limit-burst-multiplier: "5"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
    - hosts:
        - api.greenlang.ai
        - factory.greenlang.ai
      secretName: greenlang-tls
  rules:
    - host: api.greenlang.ai
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: api-gateway
                port:
                  number: 8080
```

### 3.5 Service Mesh Configuration

```yaml
# Istio Service Mesh Configuration
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: agent-factory-vs
  namespace: greenlang
spec:
  hosts:
    - agent-factory
  http:
    - match:
        - uri:
            prefix: "/api/v1/factory"
      route:
        - destination:
            host: agent-factory
            port:
              number: 8080
      retries:
        attempts: 3
        perTryTimeout: 10s
        retryOn: "5xx,reset,connect-failure"
      timeout: 30s

---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: agent-factory-dr
  namespace: greenlang
spec:
  host: agent-factory
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 1000
      http:
        h2UpgradePolicy: UPGRADE
        http1MaxPendingRequests: 1000
        http2MaxRequests: 1000
    loadBalancer:
      simple: LEAST_CONN
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
```

---

## 4. Integration with Existing Infrastructure

### 4.1 Kubernetes Integration

```
+===========================================================================+
|              Integration with Existing kubernetes/ Structure               |
+===========================================================================+

Existing Structure (kubernetes/manifests/):
+-- namespace.yaml          <- Reuse greenlang namespace
+-- rbac.yaml               <- Extend RBAC for factory roles
+-- networkpolicy.yaml      <- Add factory network policies
+-- configmap.yaml          <- Extend for factory config
+-- executor-deployment.yaml <- Reference pattern for agents
+-- runner-deployment.yaml   <- Reference pattern for workers
+-- worker-hpa.yaml          <- Reference HPA pattern
+-- servicemonitor.yaml      <- Extend for factory metrics

New Factory Manifests:
+-- factory-deployment.yaml
+-- factory-service.yaml
+-- factory-hpa.yaml
+-- factory-configmap.yaml
+-- registry-deployment.yaml
+-- registry-service.yaml
+-- runtime-deployment.yaml
+-- runtime-service.yaml
+-- runtime-hpa.yaml
```

### 4.2 Monitoring Integration

```yaml
# Integration with existing monitoring/
monitoring_integration:
  prometheus:
    existing_config: "monitoring/prometheus/"
    new_scrape_configs:
      - job_name: "agent-factory"
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app]
            regex: "agent-factory"
            action: keep

      - job_name: "agent-runtime"
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app]
            regex: "agent-runtime"
            action: keep

      - job_name: "agent-registry"
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app]
            regex: "agent-registry"
            action: keep

  grafana:
    existing_dashboards: "monitoring/grafana/dashboards/"
    new_dashboards:
      - "agent-factory-overview.json"
      - "agent-runtime-performance.json"
      - "agent-registry-health.json"
      - "agent-quality-metrics.json"

  alertmanager:
    existing_config: "monitoring/alertmanager/"
    new_alerts:
      - "agent-factory-alerts.yaml"
      - "agent-runtime-alerts.yaml"
```

### 4.3 Metrics Integration

```yaml
# Integration with existing metrics/
metrics_integration:
  existing_collectors:
    - "metrics/collectors/system.py"
    - "metrics/collectors/application.py"

  new_collectors:
    - "metrics/collectors/agent_factory.py"
    - "metrics/collectors/agent_runtime.py"
    - "metrics/collectors/agent_quality.py"

  prometheus_rules:
    # Extend existing prometheus-alerts.yaml
    additional_rules:
      - alert: AgentFactoryHighLatency
        expr: histogram_quantile(0.95, agent_factory_request_duration_seconds_bucket) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Agent Factory P95 latency above 100ms"

      - alert: AgentRuntimeHighErrorRate
        expr: rate(agent_runtime_errors_total[5m]) / rate(agent_runtime_requests_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Agent Runtime error rate above 1%"

      - alert: AgentRegistryUnhealthy
        expr: agent_registry_health_score < 0.8
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "Agent Registry health score below 80%"
```

---

## 5. High Availability Requirements

### 5.1 Multi-AZ Deployment

```
+===========================================================================+
|                    Multi-AZ High Availability                              |
+===========================================================================+

                     Region: us-east-1
+------------------+------------------+------------------+
|       AZ-A       |       AZ-B       |       AZ-C       |
+------------------+------------------+------------------+
|                  |                  |                  |
| +-------------+  | +-------------+  | +-------------+  |
| | API GW (2)  |  | | API GW (2)  |  | | API GW (2)  |  |
| +-------------+  | +-------------+  | +-------------+  |
|                  |                  |                  |
| +-------------+  | +-------------+  | +-------------+  |
| | Factory (3) |  | | Factory (3) |  | | Factory (3) |  |
| +-------------+  | +-------------+  | +-------------+  |
|                  |                  |                  |
| +-------------+  | +-------------+  | +-------------+  |
| | Runtime (10)|  | | Runtime (10)|  | | Runtime (10)|  |
| +-------------+  | +-------------+  | +-------------+  |
|                  |                  |                  |
| +-------------+  | +-------------+  | +-------------+  |
| | PostgreSQL  |  | | PostgreSQL  |  | | PostgreSQL  |  |
| | Primary     |  | | Standby     |  | | Read Replica|  |
| +-------------+  | +-------------+  | +-------------+  |
|                  |                  |                  |
| +-------------+  | +-------------+  | +-------------+  |
| | Redis (2)   |  | | Redis (2)   |  | | Redis (2)   |  |
| +-------------+  | +-------------+  | +-------------+  |
|                  |                  |                  |
+------------------+------------------+------------------+
```

### 5.2 Failover Configuration

```yaml
# Failover Configuration
high_availability:
  database:
    primary_region: "us-east-1"
    failover_regions:
      - "us-west-2"
    automatic_failover: true
    failover_timeout_seconds: 60
    rpo_minutes: 1
    rto_minutes: 15

  redis:
    sentinel_quorum: 2
    failover_timeout_ms: 60000
    min_replicas_to_write: 1
    automatic_failover: true

  kafka:
    min_insync_replicas: 2
    unclean_leader_election: false
    auto_create_topics: false

  kubernetes:
    pod_disruption_budget:
      min_available: 2
      max_unavailable: 1
    topology_spread_constraints:
      - maxSkew: 1
        topologyKey: "topology.kubernetes.io/zone"
        whenUnsatisfiable: "DoNotSchedule"
```

### 5.3 Disaster Recovery

```yaml
# Disaster Recovery Configuration
disaster_recovery:
  backup_schedule:
    postgresql:
      full_backup: "0 2 * * *"  # Daily at 2 AM
      incremental: "0 * * * *"  # Hourly
      retention_days: 30

    redis:
      rdb_snapshot: "0 */6 * * *"  # Every 6 hours
      retention_days: 7

    object_storage:
      cross_region_replication: true
      replication_regions:
        - "us-west-2"
        - "eu-west-1"

  recovery_targets:
    rto: 1 hour
    rpo: 15 minutes

  dr_runbook:
    - step: "Promote standby database"
      automation: "terraform apply -target=aws_db_instance.primary"
    - step: "Update DNS"
      automation: "aws route53 update-record"
    - step: "Scale up services"
      automation: "kubectl scale deployment --replicas=10"
    - step: "Verify health"
      automation: "scripts/health_check.sh"
```

---

## 6. Cost Optimization

### 6.1 Cost Breakdown

```
+===========================================================================+
|                    Monthly Cost Estimate (Production)                      |
+===========================================================================+

Component                    Instances    Monthly Cost    Optimization
-------------------------------------------------------------------------------
Kubernetes Cluster
  - System Nodes             3 x m6i.xl   $450           Reserved instances
  - API Gateway Nodes        5 x c6i.2xl  $1,200         HPA + Spot (40%)
  - Factory Nodes            5 x m6i.2xl  $1,500         Reserved instances
  - Runtime Nodes            30 x c6i.xl  $3,600         Spot instances (60%)
  - Worker Nodes             10 x m6i.xl  $900           Spot instances (80%)

PostgreSQL (RDS)
  - Primary                  1 x r6g.2xl  $1,200         Reserved (1-year)
  - Read Replicas            3 x r6g.xl   $900           Reserved (1-year)

Redis (ElastiCache)
  - Cluster                  6 x r6g.xl   $1,800         Reserved (1-year)

Kafka (MSK)
  - Brokers                  6 x m5.2xl   $3,000         Reserved (1-year)

Storage
  - S3                       5 TB         $125           Lifecycle policies
  - EBS (GP3)                10 TB        $800           GP3 optimization

Network
  - Data Transfer            10 TB        $900           VPC endpoints
  - NAT Gateway              3 x          $300           Single AZ option
  - Load Balancer            3 x          $200           ALB sharing

LLM APIs
  - Anthropic/OpenAI         Variable     $5,000         Caching (66% reduction)
  - Embeddings               Variable     $500           Batch processing

-------------------------------------------------------------------------------
TOTAL (before optimization)               $22,375/month
TOTAL (with optimization)                 $15,663/month  (30% savings)
-------------------------------------------------------------------------------
```

### 6.2 Cost Optimization Strategies

```yaml
cost_optimization:
  reserved_instances:
    coverage_target: 70%
    term: "1-year"
    payment: "partial-upfront"
    services:
      - "postgresql"
      - "redis"
      - "kafka"
      - "system-nodes"

  spot_instances:
    enabled: true
    workloads:
      - name: "agent-runtime"
        spot_percentage: 60
        fallback_to_on_demand: true
      - name: "workers"
        spot_percentage: 80
        fallback_to_on_demand: true

  auto_scaling:
    scale_down_delay: 300  # 5 minutes
    scale_up_threshold: 70
    scale_down_threshold: 30
    night_schedule:
      enabled: true
      min_replicas: 3
      hours: "22:00-06:00"

  storage_tiering:
    s3_lifecycle:
      - transition_to_ia: 30
      - transition_to_glacier: 90
    ebs_optimization:
      use_gp3: true
      optimize_iops: true

  caching:
    llm_response_cache: true
    cache_hit_target: 66%
    ef_cache_ttl: 3600
```

---

## 7. Capacity Planning

### 7.1 Growth Projections

| Metric | Current | 6 Months | 12 Months | 24 Months |
|--------|---------|----------|-----------|-----------|
| Agents | 500 | 2,000 | 5,000 | 10,000 |
| Requests/day | 100K | 500K | 1.5M | 5M |
| Data Storage | 100GB | 500GB | 2TB | 10TB |
| Compute Nodes | 24 | 50 | 100 | 200 |
| Monthly Cost | $15K | $30K | $60K | $120K |

### 7.2 Scaling Triggers

```yaml
scaling_triggers:
  compute:
    - metric: "cpu_utilization"
      threshold: 70
      action: "scale_up"
      cooldown: 300

    - metric: "memory_utilization"
      threshold: 80
      action: "scale_up"
      cooldown: 300

    - metric: "queue_depth"
      threshold: 1000
      action: "scale_up"
      cooldown: 60

  storage:
    - metric: "disk_utilization"
      threshold: 80
      action: "expand_volume"
      increment: "20%"

    - metric: "iops_utilization"
      threshold: 90
      action: "increase_iops"
      increment: "50%"

  database:
    - metric: "connection_count"
      threshold: 400
      action: "add_read_replica"
      cooldown: 3600
```

---

## 8. Security Requirements

### 8.1 Infrastructure Security

```yaml
infrastructure_security:
  encryption:
    at_rest:
      databases: "AES-256"
      volumes: "AES-256"
      s3: "aws:kms"
    in_transit:
      external: "TLS 1.3"
      internal: "mTLS"

  network:
    vpc_flow_logs: true
    waf_enabled: true
    ddos_protection: "aws-shield-advanced"
    security_groups:
      - "deny-all-default"
      - "allow-specific-ports"

  access:
    iam_roles: "least-privilege"
    service_accounts: "per-workload"
    secrets_management: "aws-secrets-manager"
    key_rotation: "automatic-90-days"

  compliance:
    cis_benchmark: true
    security_hub: true
    guardduty: true
    inspector: true
```

---

## Related Documents

| Document | Location | Description |
|----------|----------|-------------|
| Architecture Overview | `../system-design/00-ARCHITECTURE_OVERVIEW.md` | High-level system view |
| Layer Architecture | `../system-design/01-LAYER_ARCHITECTURE.md` | Layer specifications |
| Data Flow Patterns | `../data-flows/00-DATA_FLOW_PATTERNS.md` | Data flow documentation |
| Security Architecture | `../security/00-SECURITY_ARCHITECTURE.md` | Security design |

---

**Document Owner:** GL-AppArchitect
**Last Updated:** December 3, 2025
**Review Cycle:** Quarterly
