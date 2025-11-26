# GL-010 EMISSIONWATCH Scaling Guide

## Document Control

| Property | Value |
|----------|-------|
| Document ID | GL-010-RUNBOOK-SC-001 |
| Version | 1.0.0 |
| Last Updated | 2025-11-26 |
| Owner | GL-010 Operations Team |
| Classification | Internal |
| Review Cycle | Quarterly |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Capacity Planning](#2-capacity-planning)
3. [Resource Requirements](#3-resource-requirements)
4. [Horizontal Scaling](#4-horizontal-scaling)
5. [Vertical Scaling](#5-vertical-scaling)
6. [Multi-Facility Scaling](#6-multi-facility-scaling)
7. [Database Scaling](#7-database-scaling)
8. [Performance Optimization](#8-performance-optimization)
9. [Monitoring and Alerts](#9-monitoring-and-alerts)
10. [Scaling Procedures](#10-scaling-procedures)
11. [Cost Optimization](#11-cost-optimization)
12. [Appendices](#12-appendices)

---

## 1. Overview

### 1.1 Purpose

This Scaling Guide provides comprehensive guidance for scaling GL-010 EMISSIONWATCH to handle varying workloads, from single-facility deployments to enterprise-scale multi-facility operations with thousands of emission sources.

### 1.2 Scope

This guide covers:
- Capacity planning and resource estimation
- Horizontal and vertical scaling strategies
- Multi-facility and multi-region architectures
- Database scaling approaches
- Performance optimization techniques
- Cost-effective scaling decisions

### 1.3 Scaling Principles for Emissions Monitoring

**Key Requirements:**
- **Data Continuity**: Scaling operations must not cause data gaps
- **Regulatory Compliance**: Must maintain data quality thresholds
- **Real-Time Processing**: CEMS data must be processed within regulatory timeframes
- **Audit Trail**: All scaling operations must be logged

### 1.4 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        GL-010 EMISSIONWATCH ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                  │
│  │   CEMS Data     │    │   Processing    │    │   Storage &     │                  │
│  │   Ingestion     │───►│   Workers       │───►│   Reporting     │                  │
│  │                 │    │                 │    │                 │                  │
│  │  - Connectors   │    │  - Calculators  │    │  - Database     │                  │
│  │  - Collectors   │    │  - Compliance   │    │  - Cache        │                  │
│  │  - Buffers      │    │  - Validators   │    │  - Archive      │                  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                  │
│         │                       │                       │                           │
│         └───────────────────────┼───────────────────────┘                           │
│                                 ▼                                                    │
│                    ┌─────────────────────────┐                                      │
│                    │    Message Queue        │                                      │
│                    │    (Kafka/RabbitMQ)     │                                      │
│                    └─────────────────────────┘                                      │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Capacity Planning

### 2.1 Workload Estimation

**CEMS Data Points Per Second:**

| Facility Size | Sources | Parameters | Sample Rate | Data Points/Second |
|---------------|---------|------------|-------------|-------------------|
| Small | 1-5 | 10 | 1/min | 0.8 - 4 |
| Medium | 5-20 | 10 | 1/min | 4 - 16 |
| Large | 20-50 | 15 | 1/min | 20 - 50 |
| Enterprise | 50-200 | 15 | 1/min | 50 - 200 |

**Calculation Requirements:**

| Operation | CPU Cost | Memory Cost | Frequency |
|-----------|----------|-------------|-----------|
| Data validation | Low | Low | Per data point |
| Unit conversion | Low | Low | Per data point |
| Emissions calculation | Medium | Low | Per minute |
| Rolling average | Medium | Medium | Per minute |
| Compliance check | Medium | Medium | Per minute |
| Report generation | High | High | Daily/Quarterly |

### 2.2 Baseline Resource Calculator

```bash
# Calculate baseline resource requirements
greenlang capacity calculate \
  --facilities 10 \
  --sources-per-facility 15 \
  --pollutants 5 \
  --sample-rate "1/min" \
  --retention-years 7

# Example output:
# Estimated Resources:
#   CPU: 4 cores (peak: 8 cores)
#   Memory: 8 GB (peak: 16 GB)
#   Storage: 500 GB/year
#   Network: 10 Mbps sustained
#   Database connections: 50
```

### 2.3 Capacity Planning Formula

**Data Volume Estimation:**

```
Daily Data Volume (GB) =
  Facilities ×
  Sources_per_Facility ×
  Pollutants ×
  Samples_per_Day ×
  Bytes_per_Sample /
  1,000,000,000

Where:
  Samples_per_Day = 1440 (for 1-minute samples)
  Bytes_per_Sample = ~500 (including metadata)
```

**Example Calculation:**

```
For 50 facilities, 20 sources each, 5 pollutants:
Daily Volume = 50 × 20 × 5 × 1440 × 500 / 1,000,000,000
             = 3.6 GB/day
             = 1.3 TB/year (raw data)
             = 9.1 TB for 7-year retention
```

### 2.4 Growth Planning

**Typical Growth Factors:**

| Scenario | Annual Growth | Planning Horizon |
|----------|---------------|------------------|
| Organic growth | 10-20% | 3 years |
| Acquisition | 50-100% step | 1 year |
| Regulatory expansion | 20-30% | 2 years |
| New pollutant monitoring | 10-20% | 1 year |

```bash
# Project capacity needs
greenlang capacity forecast \
  --current-facilities 50 \
  --growth-rate 0.20 \
  --years 3 \
  --include-new-regulations

# Example output:
# Year 1: 60 facilities, 6 cores, 12 GB memory
# Year 2: 72 facilities, 8 cores, 16 GB memory
# Year 3: 86 facilities, 10 cores, 20 GB memory
```

---

## 3. Resource Requirements

### 3.1 Component Resource Matrix

| Component | Min CPU | Recommended CPU | Min Memory | Recommended Memory |
|-----------|---------|-----------------|------------|-------------------|
| CEMS Collector | 0.25 cores | 0.5 cores | 256 MB | 512 MB |
| Data Validator | 0.5 cores | 1 core | 512 MB | 1 GB |
| Emissions Calculator | 1 core | 2 cores | 1 GB | 2 GB |
| Compliance Engine | 0.5 cores | 1 core | 512 MB | 1 GB |
| Report Generator | 1 core | 2 cores | 2 GB | 4 GB |
| API Server | 0.5 cores | 1 core | 512 MB | 1 GB |
| Cache Server | 0.25 cores | 0.5 cores | 1 GB | 2 GB |

### 3.2 Resource Requirements by Deployment Size

**Small Deployment (1-10 facilities):**

```yaml
# Kubernetes resource requirements - Small
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-010-emissionwatch
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: emissionwatch
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2000m"
              memory: "4Gi"
```

**Medium Deployment (10-50 facilities):**

```yaml
# Kubernetes resource requirements - Medium
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-010-emissionwatch
spec:
  replicas: 4
  template:
    spec:
      containers:
        - name: emissionwatch
          resources:
            requests:
              cpu: "1000m"
              memory: "2Gi"
            limits:
              cpu: "4000m"
              memory: "8Gi"
```

**Large Deployment (50-200 facilities):**

```yaml
# Kubernetes resource requirements - Large
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-010-emissionwatch
spec:
  replicas: 8
  template:
    spec:
      containers:
        - name: emissionwatch
          resources:
            requests:
              cpu: "2000m"
              memory: "4Gi"
            limits:
              cpu: "8000m"
              memory: "16Gi"
```

### 3.3 Storage Requirements

| Data Type | Size per Facility/Year | Retention | Total per Facility |
|-----------|------------------------|-----------|-------------------|
| Raw CEMS data | 20 GB | 7 years | 140 GB |
| Calculated emissions | 5 GB | 7 years | 35 GB |
| Compliance records | 1 GB | 7 years | 7 GB |
| Reports | 2 GB | 7 years | 14 GB |
| Audit logs | 1 GB | 7 years | 7 GB |
| **Total** | **29 GB/year** | **7 years** | **203 GB** |

### 3.4 Network Requirements

| Traffic Type | Bandwidth (per facility) | Latency Requirement |
|--------------|--------------------------|---------------------|
| CEMS data ingest | 100 Kbps sustained | < 500ms |
| API requests | 10 Kbps average | < 200ms |
| Report generation | 1 Mbps burst | N/A |
| Database replication | 50 Kbps sustained | < 100ms |

---

## 4. Horizontal Scaling

### 4.1 Auto-Scaling Configuration

**Horizontal Pod Autoscaler (HPA):**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gl-010-emissionwatch-hpa
  namespace: gl-agents
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gl-010-emissionwatch
  minReplicas: 2
  maxReplicas: 20
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
    - type: Pods
      pods:
        metric:
          name: cems_data_points_per_second
        target:
          type: AverageValue
          averageValue: "1000"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
        - type: Pods
          value: 4
          periodSeconds: 15
      selectPolicy: Max
```

### 4.2 Component-Specific Scaling

**CEMS Collector Scaling:**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gl-010-cems-collector-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gl-010-cems-collector
  minReplicas: 1
  maxReplicas: 10
  metrics:
    - type: Pods
      pods:
        metric:
          name: cems_connections_active
        target:
          type: AverageValue
          averageValue: "50"
```

**Emissions Calculator Scaling:**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gl-010-calculator-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gl-010-calculator
  minReplicas: 2
  maxReplicas: 16
  metrics:
    - type: Pods
      pods:
        metric:
          name: calculation_queue_depth
        target:
          type: AverageValue
          averageValue: "100"
```

### 4.3 Load Balancing Configuration

**Service Load Balancing:**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: gl-010-emissionwatch
  namespace: gl-agents
  annotations:
    service.kubernetes.io/topology-aware-hints: auto
spec:
  type: ClusterIP
  selector:
    app: gl-010-emissionwatch
  ports:
    - port: 8080
      targetPort: 8080
      protocol: TCP
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800  # 3 hours for long-running reports
```

### 4.4 Queue-Based Scaling

**Configure Kafka Consumer Scaling:**

```bash
# Scale calculator workers based on queue lag
greenlang scaling configure kafka-consumer \
  --consumer-group gl-010-calculators \
  --topic emissions-calculation-requests \
  --min-consumers 2 \
  --max-consumers 16 \
  --lag-threshold 1000 \
  --scale-up-rate 2 \
  --scale-down-rate 1
```

**KEDA Scaler for Kafka:**

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: gl-010-calculator-scaler
  namespace: gl-agents
spec:
  scaleTargetRef:
    name: gl-010-calculator
  minReplicaCount: 2
  maxReplicaCount: 16
  triggers:
    - type: kafka
      metadata:
        bootstrapServers: kafka.gl-platform:9092
        consumerGroup: gl-010-calculators
        topic: emissions-calculation-requests
        lagThreshold: "1000"
```

### 4.5 Manual Horizontal Scaling

```bash
# Scale deployment manually
kubectl scale deployment gl-010-emissionwatch -n gl-agents --replicas=6

# Scale specific component
kubectl scale deployment gl-010-calculator -n gl-agents --replicas=8

# Verify scaling
kubectl get pods -n gl-agents -l app=gl-010-emissionwatch

# Check HPA status
kubectl get hpa -n gl-agents
```

---

## 5. Vertical Scaling

### 5.1 When to Scale Vertically

| Symptom | Vertical Scale | Horizontal Scale |
|---------|----------------|------------------|
| Single calculation taking too long | Yes | No |
| Memory pressure during report generation | Yes | No |
| High CPU across all pods | Maybe | Yes |
| Queue depth increasing | No | Yes |
| Database connection exhaustion | No | Yes (with pooling) |

### 5.2 Memory Optimization

**JVM Memory Configuration:**

```yaml
# For Java-based components
env:
  - name: JAVA_OPTS
    value: "-Xms2g -Xmx6g -XX:+UseG1GC -XX:MaxGCPauseMillis=200"
  - name: JAVA_TOOL_OPTIONS
    value: "-XX:+ExitOnOutOfMemoryError"
```

**Python Memory Configuration:**

```yaml
# For Python-based components
env:
  - name: PYTHONMALLOC
    value: "pymalloc"
  - name: MALLOC_ARENA_MAX
    value: "2"
```

### 5.3 CPU Optimization

**Scientific Library Tuning (NumPy/SciPy):**

```bash
# Configure OpenBLAS threads
export OPENBLAS_NUM_THREADS=4

# Configure MKL threads (if using Intel MKL)
export MKL_NUM_THREADS=4

# Configure OMP threads
export OMP_NUM_THREADS=4
```

**Kubernetes CPU Configuration:**

```yaml
# For CPU-intensive calculations
spec:
  containers:
    - name: calculator
      resources:
        requests:
          cpu: "2000m"  # Request 2 cores
        limits:
          cpu: "4000m"  # Allow burst to 4 cores
```

### 5.4 Vertical Pod Autoscaler (VPA)

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: gl-010-emissionwatch-vpa
  namespace: gl-agents
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gl-010-emissionwatch
  updatePolicy:
    updateMode: "Auto"  # or "Off" for recommendations only
  resourcePolicy:
    containerPolicies:
      - containerName: emissionwatch
        minAllowed:
          cpu: "500m"
          memory: "1Gi"
        maxAllowed:
          cpu: "8000m"
          memory: "16Gi"
        controlledResources: ["cpu", "memory"]
        controlledValues: RequestsAndLimits
```

### 5.5 Vertical Scaling Procedure

```bash
# Step 1: Check current resource usage
kubectl top pod -n gl-agents -l app=gl-010-emissionwatch

# Step 2: Get VPA recommendations
kubectl get vpa gl-010-emissionwatch-vpa -n gl-agents -o yaml

# Step 3: Apply new resource limits
kubectl patch deployment gl-010-emissionwatch -n gl-agents --type=merge -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "emissionwatch",
          "resources": {
            "requests": {
              "cpu": "2000m",
              "memory": "4Gi"
            },
            "limits": {
              "cpu": "8000m",
              "memory": "16Gi"
            }
          }
        }]
      }
    }
  }
}'

# Step 4: Monitor rollout
kubectl rollout status deployment/gl-010-emissionwatch -n gl-agents

# Step 5: Verify new resource allocation
kubectl describe pod -n gl-agents -l app=gl-010-emissionwatch | grep -A5 "Limits:\|Requests:"
```

---

## 6. Multi-Facility Scaling

### 6.1 Multi-Site Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         MULTI-FACILITY ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Region: US-East                    Region: US-West                                 │
│  ┌───────────────────┐              ┌───────────────────┐                           │
│  │  Edge Collector   │              │  Edge Collector   │                           │
│  │  - Facility 1-25  │              │  - Facility 26-50 │                           │
│  │  - Local buffer   │              │  - Local buffer   │                           │
│  └─────────┬─────────┘              └─────────┬─────────┘                           │
│            │                                  │                                      │
│            └──────────────┬───────────────────┘                                      │
│                           ▼                                                          │
│               ┌─────────────────────────┐                                           │
│               │   Central Processing    │                                           │
│               │   - Aggregation         │                                           │
│               │   - Compliance          │                                           │
│               │   - Reporting           │                                           │
│               └─────────────────────────┘                                           │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Facility Sharding Strategy

**Shard by Facility ID:**

```yaml
# ConfigMap for facility sharding
apiVersion: v1
kind: ConfigMap
metadata:
  name: gl-010-facility-shards
  namespace: gl-agents
data:
  shard-config.yaml: |
    shards:
      - id: shard-1
        facilities: ["facility-001", "facility-002", "facility-003"]
        processor: gl-010-processor-1
      - id: shard-2
        facilities: ["facility-004", "facility-005", "facility-006"]
        processor: gl-010-processor-2
      - id: shard-3
        facilities: ["facility-007", "facility-008", "facility-009"]
        processor: gl-010-processor-3
```

**Dynamic Shard Assignment:**

```bash
# Assign facilities to shards based on load
greenlang scaling assign-shards \
  --agent GL-010 \
  --strategy "load-balanced" \
  --target-load-per-shard 20 \
  --rebalance-threshold 0.2
```

### 6.3 Regional Deployment

**Multi-Region Kubernetes Configuration:**

```yaml
# Regional deployment - US East
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-010-emissionwatch-us-east
  namespace: gl-agents
  labels:
    region: us-east
spec:
  replicas: 4
  selector:
    matchLabels:
      app: gl-010-emissionwatch
      region: us-east
  template:
    metadata:
      labels:
        app: gl-010-emissionwatch
        region: us-east
    spec:
      nodeSelector:
        topology.kubernetes.io/region: us-east-1
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchLabels:
                    app: gl-010-emissionwatch
                topologyKey: topology.kubernetes.io/zone
```

### 6.4 Data Aggregation

**Configure Hierarchical Aggregation:**

```bash
# Configure facility-level aggregation
greenlang aggregation configure \
  --level facility \
  --aggregation-interval 1m \
  --metrics "all-pollutants,data-quality"

# Configure regional aggregation
greenlang aggregation configure \
  --level region \
  --aggregation-interval 5m \
  --roll-up-from facility

# Configure corporate aggregation
greenlang aggregation configure \
  --level corporate \
  --aggregation-interval 1h \
  --roll-up-from region
```

### 6.5 Edge Processing

**Edge Collector Deployment:**

```yaml
# Edge collector for remote facilities
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-010-edge-collector
  namespace: gl-agents
spec:
  replicas: 1
  template:
    spec:
      containers:
        - name: edge-collector
          image: greenlang/gl-010-edge-collector:v2.4.0
          env:
            - name: FACILITY_IDS
              value: "facility-001,facility-002"
            - name: LOCAL_BUFFER_SIZE
              value: "1GB"
            - name: UPLOAD_INTERVAL
              value: "60s"
            - name: OFFLINE_MODE_DURATION
              value: "24h"
          resources:
            requests:
              cpu: "250m"
              memory: "512Mi"
            limits:
              cpu: "1000m"
              memory: "2Gi"
          volumeMounts:
            - name: local-buffer
              mountPath: /data/buffer
      volumes:
        - name: local-buffer
          persistentVolumeClaim:
            claimName: edge-collector-buffer
```

---

## 7. Database Scaling

### 7.1 Database Connection Pooling

**PgBouncer Configuration:**

```ini
# pgbouncer.ini
[databases]
gl010_emissions = host=postgres-primary port=5432 dbname=gl010_emissions

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt

# Pool settings
pool_mode = transaction
default_pool_size = 50
min_pool_size = 10
reserve_pool_size = 10
max_client_conn = 500
max_db_connections = 100

# Timeouts
server_idle_timeout = 600
query_timeout = 120
client_idle_timeout = 0
```

**Application Connection Pool Configuration:**

```yaml
# Application database configuration
database:
  pool:
    min_connections: 10
    max_connections: 50
    connection_timeout: 10s
    idle_timeout: 300s
    max_lifetime: 3600s
  read_replicas:
    - host: postgres-replica-1
      weight: 50
    - host: postgres-replica-2
      weight: 50
```

### 7.2 Read Replica Configuration

```bash
# Configure read replicas for reporting queries
greenlang db configure-replicas \
  --database gl010-emissions-db \
  --primary postgres-primary:5432 \
  --replicas "postgres-replica-1:5432,postgres-replica-2:5432" \
  --read-routing "reporting-queries,historical-queries,analytics"

# Verify replication lag
greenlang db replication-status \
  --database gl010-emissions-db
```

### 7.3 Table Partitioning

**Time-Based Partitioning for Emissions Data:**

```sql
-- Create partitioned emissions table
CREATE TABLE emissions_data (
    id BIGSERIAL,
    facility_id VARCHAR(50) NOT NULL,
    pollutant VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    value NUMERIC(15,6) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    data_quality_flag INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions
CREATE TABLE emissions_data_2025_11 PARTITION OF emissions_data
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

CREATE TABLE emissions_data_2025_12 PARTITION OF emissions_data
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

-- Create indexes on partitions
CREATE INDEX idx_emissions_facility_time_2025_11
    ON emissions_data_2025_11 (facility_id, timestamp);
```

**Automated Partition Management:**

```bash
# Configure automatic partition creation
greenlang db partitions configure \
  --database gl010-emissions-db \
  --table emissions_data \
  --partition-by "month" \
  --create-ahead 3 \
  --retain-months 84  # 7 years

# Check partition status
greenlang db partitions status \
  --database gl010-emissions-db \
  --table emissions_data
```

### 7.4 Database Caching

**Redis Cache Configuration:**

```yaml
# Redis cache for emissions data
apiVersion: v1
kind: ConfigMap
metadata:
  name: gl-010-cache-config
  namespace: gl-agents
data:
  redis.conf: |
    maxmemory 4gb
    maxmemory-policy allkeys-lru

    # Cache TTLs (in seconds)
    # Real-time emissions: 60 seconds
    # Calculated values: 300 seconds
    # Compliance status: 60 seconds
    # Emission factors: 86400 seconds
    # Regulatory limits: 86400 seconds
```

**Cache Strategy:**

```bash
# Configure cache strategy
greenlang cache configure \
  --agent GL-010 \
  --cache-type redis \
  --endpoints "redis-cluster:6379" \
  --strategies "
    emission_factors:write_through:ttl=86400,
    regulatory_limits:write_through:ttl=86400,
    real_time_emissions:write_back:ttl=60,
    compliance_status:write_back:ttl=60
  "
```

### 7.5 Database Scaling Metrics

```bash
# Monitor database performance
greenlang db metrics \
  --database gl010-emissions-db \
  --metrics "connections,queries_per_second,cache_hit_ratio,replication_lag"

# Alert thresholds
greenlang db alerts configure \
  --database gl010-emissions-db \
  --alerts "
    connections_percent:warning=70,critical=90,
    queries_per_second:warning=1000,critical=2000,
    cache_hit_ratio:warning=90,critical=80,
    replication_lag_seconds:warning=10,critical=60
  "
```

---

## 8. Performance Optimization

### 8.1 Calculation Caching Strategies

**Cache Frequently Used Values:**

```bash
# Configure calculation cache
greenlang cache configure-calculations \
  --agent GL-010 \
  --cache-items "
    f_factors:permanent,
    conversion_factors:permanent,
    emission_factors:daily_refresh,
    rolling_averages:5min_ttl,
    compliance_limits:daily_refresh
  "

# Warm cache on startup
greenlang cache warm \
  --agent GL-010 \
  --items "f_factors,conversion_factors,emission_factors,compliance_limits"
```

### 8.2 Database Query Optimization

**Index Strategy:**

```sql
-- Core indexes for emissions queries
CREATE INDEX CONCURRENTLY idx_emissions_facility_pollutant_time
    ON emissions_data (facility_id, pollutant, timestamp DESC);

CREATE INDEX CONCURRENTLY idx_emissions_time_facility
    ON emissions_data (timestamp, facility_id);

CREATE INDEX CONCURRENTLY idx_compliance_facility_status
    ON compliance_records (facility_id, status, check_time DESC);

-- Partial indexes for common queries
CREATE INDEX CONCURRENTLY idx_emissions_violations
    ON compliance_records (facility_id, check_time DESC)
    WHERE status = 'violation';

-- Covering indexes for dashboard queries
CREATE INDEX CONCURRENTLY idx_emissions_summary
    ON emissions_data (facility_id, timestamp)
    INCLUDE (pollutant, value, data_quality_flag);
```

**Query Optimization:**

```bash
# Analyze slow queries
greenlang db analyze-queries \
  --database gl010-emissions-db \
  --threshold-ms 100 \
  --period "last-24h"

# Get index recommendations
greenlang db index-recommendations \
  --database gl010-emissions-db \
  --based-on "slow_queries,missing_indexes"
```

### 8.3 Report Generation Batching

**Configure Report Batching:**

```bash
# Configure batch report generation
greenlang reporting configure-batching \
  --agent GL-010 \
  --batch-size 10 \
  --concurrent-batches 4 \
  --priority-order "deadline,facility_size" \
  --off-peak-window "02:00-06:00"
```

**Batch Processing Configuration:**

```yaml
# Report generation job configuration
apiVersion: batch/v1
kind: Job
metadata:
  name: gl-010-quarterly-reports
spec:
  parallelism: 4
  completions: 50  # Number of facilities
  backoffLimit: 3
  template:
    spec:
      containers:
        - name: report-generator
          image: greenlang/gl-010-report-generator:v2.4.0
          resources:
            requests:
              cpu: "2000m"
              memory: "4Gi"
            limits:
              cpu: "4000m"
              memory: "8Gi"
```

### 8.4 Data Pipeline Optimization

**Stream Processing Configuration:**

```yaml
# Kafka Streams configuration
kafka:
  streams:
    application.id: gl-010-emissions-processor
    num.stream.threads: 4
    cache.max.bytes.buffering: 104857600  # 100MB
    commit.interval.ms: 1000

    # Producer settings
    producer:
      batch.size: 65536
      linger.ms: 10
      compression.type: lz4

    # Consumer settings
    consumer:
      max.poll.records: 1000
      fetch.min.bytes: 65536
```

### 8.5 Memory Optimization

**Efficient Data Structures:**

```python
# Use memory-efficient data structures
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass(slots=True)  # Reduce memory overhead
class EmissionsRecord:
    facility_id: str
    pollutant: str
    timestamp: int  # Unix timestamp
    value: float
    quality_flag: int

# Use numpy arrays for bulk calculations
emissions_array = np.zeros(
    (num_facilities, num_pollutants, samples_per_day),
    dtype=np.float32  # Use float32 instead of float64
)
```

---

## 9. Monitoring and Alerts

### 9.1 Scaling Metrics

**Key Metrics to Monitor:**

| Metric | Warning Threshold | Critical Threshold | Action |
|--------|-------------------|-------------------|--------|
| CPU Utilization | 70% | 85% | Scale up |
| Memory Utilization | 75% | 90% | Scale up |
| Queue Depth | 500 | 1000 | Scale workers |
| Request Latency (p99) | 1s | 2s | Investigate |
| Error Rate | 0.1% | 1% | Investigate |
| Data Processing Lag | 1 min | 5 min | Scale up |

### 9.2 Prometheus Metrics

```yaml
# ServiceMonitor for GL-010
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: gl-010-emissionwatch
  namespace: gl-agents
spec:
  selector:
    matchLabels:
      app: gl-010-emissionwatch
  endpoints:
    - port: metrics
      interval: 15s
      path: /metrics
```

**Custom Metrics:**

```python
# Custom Prometheus metrics
from prometheus_client import Counter, Gauge, Histogram

# Processing metrics
emissions_processed = Counter(
    'gl010_emissions_processed_total',
    'Total emissions data points processed',
    ['facility_id', 'pollutant']
)

calculation_duration = Histogram(
    'gl010_calculation_duration_seconds',
    'Emissions calculation duration',
    ['calculation_type'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

queue_depth = Gauge(
    'gl010_queue_depth',
    'Current queue depth',
    ['queue_name']
)

active_facilities = Gauge(
    'gl010_active_facilities',
    'Number of active facilities'
)
```

### 9.3 Scaling Alerts

```yaml
# Prometheus alerts for scaling
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: gl-010-scaling-alerts
  namespace: gl-agents
spec:
  groups:
    - name: gl010-scaling
      rules:
        - alert: GL010HighCPU
          expr: |
            avg(rate(container_cpu_usage_seconds_total{
              pod=~"gl-010-emissionwatch.*"
            }[5m])) by (pod) > 0.85
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "High CPU usage on GL-010"
            description: "Pod {{ $labels.pod }} CPU > 85% for 5 minutes"

        - alert: GL010HighMemory
          expr: |
            container_memory_working_set_bytes{
              pod=~"gl-010-emissionwatch.*"
            } / container_spec_memory_limit_bytes > 0.9
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "High memory usage on GL-010"

        - alert: GL010HighQueueDepth
          expr: gl010_queue_depth > 1000
          for: 2m
          labels:
            severity: warning
          annotations:
            summary: "High queue depth on GL-010"

        - alert: GL010ProcessingLag
          expr: gl010_processing_lag_seconds > 300
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "Data processing lag > 5 minutes"
```

### 9.4 Capacity Dashboard

```bash
# Create capacity dashboard
greenlang dashboard create capacity \
  --agent GL-010 \
  --panels "
    cpu_utilization:by_pod,
    memory_utilization:by_pod,
    queue_depth:by_queue,
    processing_rate:total,
    active_facilities:gauge,
    data_points_per_second:rate,
    hpa_status:by_deployment,
    scaling_events:timeline
  "
```

---

## 10. Scaling Procedures

### 10.1 Scale-Up Procedure

```bash
# Step 1: Verify current capacity
greenlang capacity current --agent GL-010

# Step 2: Calculate required capacity
greenlang capacity calculate \
  --facilities 75 \
  --growth-buffer 20%

# Step 3: Update HPA limits if needed
kubectl patch hpa gl-010-emissionwatch-hpa -n gl-agents --type=merge -p '
{
  "spec": {
    "maxReplicas": 25
  }
}'

# Step 4: Manually scale if immediate capacity needed
kubectl scale deployment gl-010-emissionwatch -n gl-agents --replicas=10

# Step 5: Verify scale-up
kubectl get pods -n gl-agents -l app=gl-010-emissionwatch
greenlang health --agent GL-010 --full

# Step 6: Monitor performance
greenlang metrics watch --agent GL-010 --duration 30m
```

### 10.2 Scale-Down Procedure

```bash
# Step 1: Verify current load
greenlang capacity current --agent GL-010

# Step 2: Check for in-flight operations
greenlang operations pending --agent GL-010

# Step 3: Wait for stable period (no pending reports, off-peak)
greenlang operations wait-stable --agent GL-010 --timeout 30m

# Step 4: Reduce HPA max if permanently reducing
kubectl patch hpa gl-010-emissionwatch-hpa -n gl-agents --type=merge -p '
{
  "spec": {
    "maxReplicas": 15
  }
}'

# Step 5: Scale down
kubectl scale deployment gl-010-emissionwatch -n gl-agents --replicas=6

# Step 6: Verify data continuity
greenlang data verify-continuity --agent GL-010 --time-range "last-30m"
```

### 10.3 Emergency Scaling

```bash
# EMERGENCY SCALE-UP
# Use when experiencing data loss or severe performance degradation

# Step 1: Immediate scale to maximum
kubectl scale deployment gl-010-emissionwatch -n gl-agents --replicas=20

# Step 2: Scale database connections
greenlang db scale-connections \
  --database gl010-emissions-db \
  --max-connections 200

# Step 3: Enable emergency caching
greenlang cache enable-emergency-mode --agent GL-010

# Step 4: Notify stakeholders
greenlang notify emergency-scale \
  --agent GL-010 \
  --reason "Performance degradation" \
  --action "Emergency scale-up executed"
```

---

## 11. Cost Optimization

### 11.1 Right-Sizing Recommendations

```bash
# Get right-sizing recommendations
greenlang capacity right-size \
  --agent GL-010 \
  --analysis-period "30d" \
  --target-utilization 70

# Example output:
# Component               Current    Recommended  Savings
# gl-010-emissionwatch    8 x 4GB    6 x 3GB      25%
# gl-010-calculator       4 x 2GB    3 x 2GB      25%
# gl-010-collector        4 x 1GB    2 x 1GB      50%
```

### 11.2 Spot Instance Usage

```yaml
# Node pool for spot instances (non-critical workloads)
apiVersion: v1
kind: ConfigMap
metadata:
  name: gl-010-spot-config
data:
  spot-workloads: |
    - report-generator  # Can tolerate interruption
    - data-archiver     # Can tolerate interruption
    - analytics         # Can tolerate interruption

  non-spot-workloads: |
    - cems-collector    # Critical - no interruption
    - calculator        # Critical - no interruption
    - compliance-engine # Critical - no interruption
```

### 11.3 Auto-Scaling Schedule

```yaml
# CronJob for scheduled scaling
apiVersion: batch/v1
kind: CronJob
metadata:
  name: gl-010-schedule-scale-down
  namespace: gl-agents
spec:
  schedule: "0 22 * * *"  # 10 PM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: scaler
              image: bitnami/kubectl:latest
              command:
                - /bin/sh
                - -c
                - |
                  kubectl scale deployment gl-010-report-generator \
                    -n gl-agents --replicas=1
          restartPolicy: OnFailure

---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: gl-010-schedule-scale-up
  namespace: gl-agents
spec:
  schedule: "0 6 * * *"  # 6 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: scaler
              image: bitnami/kubectl:latest
              command:
                - /bin/sh
                - -c
                - |
                  kubectl scale deployment gl-010-report-generator \
                    -n gl-agents --replicas=4
          restartPolicy: OnFailure
```

### 11.4 Cost Monitoring

```bash
# Get cost breakdown
greenlang cost report \
  --agent GL-010 \
  --period "last-30d" \
  --breakdown-by "component,resource"

# Set cost alerts
greenlang cost alert \
  --agent GL-010 \
  --monthly-budget 5000 \
  --warning-threshold 80 \
  --critical-threshold 100
```

---

## 12. Appendices

### Appendix A: Quick Reference

| Action | Command |
|--------|---------|
| Check capacity | `greenlang capacity current --agent GL-010` |
| Scale up | `kubectl scale deployment gl-010-emissionwatch -n gl-agents --replicas=N` |
| Scale down | `kubectl scale deployment gl-010-emissionwatch -n gl-agents --replicas=N` |
| Check HPA | `kubectl get hpa -n gl-agents` |
| Check VPA | `kubectl get vpa -n gl-agents` |
| View metrics | `greenlang metrics summary --agent GL-010` |

### Appendix B: Sizing Reference Table

| Facilities | CPU (cores) | Memory (GB) | Storage (TB) | Replicas |
|------------|-------------|-------------|--------------|----------|
| 1-10 | 2-4 | 4-8 | 0.5-2 | 2-3 |
| 10-25 | 4-8 | 8-16 | 2-5 | 3-5 |
| 25-50 | 8-16 | 16-32 | 5-10 | 5-8 |
| 50-100 | 16-32 | 32-64 | 10-20 | 8-12 |
| 100-200 | 32-64 | 64-128 | 20-40 | 12-20 |

### Appendix C: Related Documentation

| Document | Location |
|----------|----------|
| Incident Response | ./INCIDENT_RESPONSE.md |
| Troubleshooting | ./TROUBLESHOOTING.md |
| Rollback Procedure | ./ROLLBACK_PROCEDURE.md |
| Maintenance Guide | ./MAINTENANCE.md |

### Appendix D: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-26 | GL-TechWriter | Initial release |

---

**Document Classification:** Internal Use Only

**Next Review Date:** 2026-02-26

**Feedback:** Submit feedback to docs@greenlang.io with subject "GL-010 Scaling Guide Feedback"
