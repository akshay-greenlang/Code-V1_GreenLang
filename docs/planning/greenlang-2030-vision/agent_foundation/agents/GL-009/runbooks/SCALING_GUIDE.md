# GL-009 THERMALIQ Scaling Guide

**Agent**: GL-009 THERMALIQ ThermalEfficiencyCalculator
**Version**: 1.0.0
**Last Updated**: 2025-11-26
**Owner**: GreenLang SRE Team

---

## Table of Contents

1. [Overview](#overview)
2. [Capacity Planning](#capacity-planning)
3. [Horizontal Scaling](#horizontal-scaling)
4. [Vertical Scaling](#vertical-scaling)
5. [Auto-Scaling](#auto-scaling)
6. [Database Scaling](#database-scaling)
7. [Cache Scaling](#cache-scaling)
8. [Load Balancing](#load-balancing)
9. [Multi-Region Deployment](#multi-region-deployment)
10. [Performance Optimization](#performance-optimization)

---

## Overview

This guide provides procedures for scaling GL-009 THERMALIQ to handle increased load, improve performance, and ensure high availability across multiple regions.

### Current Production Capacity

**As of 2025-11-26**:

| Resource | Current | Max Capacity | Utilization |
|----------|---------|--------------|-------------|
| Pods | 4 | 12 (HPA) | 45% |
| CPU per Pod | 2 cores | 4 cores | 60% |
| Memory per Pod | 4 GB | 8 GB | 55% |
| Database Connections | 80 | 200 | 40% |
| Redis Memory | 4 GB | 16 GB | 30% |
| Request Rate | 50 req/sec | 200 req/sec | 25% |

### Scaling Triggers

**Scale Up When**:
- CPU usage > 70% for 5 minutes
- Memory usage > 80% for 5 minutes
- Request queue depth > 50
- Response latency p95 > 20 seconds
- Error rate > 2%

**Scale Down When**:
- CPU usage < 30% for 15 minutes
- Memory usage < 40% for 15 minutes
- Request queue depth < 5
- Response latency p95 < 5 seconds
- Stable for > 30 minutes

---

## Capacity Planning

### Capacity Calculation

**Per-Pod Capacity**:
```
Calculations per second = 1000 / avg_calculation_duration_ms
Example: 1000 / 10000 = 0.1 calculations/sec/pod
```

**Cluster Capacity**:
```
Total capacity = pods × calculations_per_sec_per_pod
Example: 4 pods × 0.1 = 0.4 calculations/sec = 24 calculations/min
```

**Required Capacity Planning**:
```bash
# Step 1: Determine peak load requirements
# From historical data or projected growth

PEAK_CALCULATIONS_PER_HOUR=1000
PEAK_CALCULATIONS_PER_SECOND=$(echo "$PEAK_CALCULATIONS_PER_HOUR / 3600" | bc -l)

# Step 2: Calculate required pods
AVG_CALCULATION_DURATION_SECONDS=10
CALCULATIONS_PER_POD_PER_SECOND=$(echo "1 / $AVG_CALCULATION_DURATION_SECONDS" | bc -l)

REQUIRED_PODS=$(echo "$PEAK_CALCULATIONS_PER_SECOND / $CALCULATIONS_PER_POD_PER_SECOND" | bc)

# Add 20% buffer for spikes
REQUIRED_PODS_WITH_BUFFER=$(echo "$REQUIRED_PODS * 1.2" | bc | cut -d. -f1)

echo "Required pods: $REQUIRED_PODS_WITH_BUFFER"
```

### Resource Requirements Per Pod

**CPU**:
```
Base CPU: 0.5 cores (idle)
Per calculation: 0.1 cores × calculation_duration_seconds
Peak CPU = base + (concurrent_calculations × 0.1 × duration)

Example:
- 5 concurrent calculations
- 10 second duration each
- Peak CPU = 0.5 + (5 × 0.1 × 10) = 5.5 cores

Recommended limit: 4 cores per pod
Recommended request: 2 cores per pod
```

**Memory**:
```
Base memory: 512 MB (application overhead)
Per calculation: data_points × 100 bytes

Example:
- 50,000 data points per calculation
- 5 concurrent calculations
- Memory = 512 MB + (5 × 50,000 × 100 / 1024 / 1024) = ~750 MB

Recommended limit: 8 GB per pod
Recommended request: 4 GB per pod
```

**Storage**:
```
Logs: 10 GB per pod (rotated daily)
Temp files: 5 GB per pod
Recommended: 20 GB ephemeral storage per pod
```

### Growth Projections

**Monthly Growth Rate**: 15%
**Current Load**: 50 calculations/hour
**Projected Load**:

| Month | Calculations/Hour | Required Pods | Notes |
|-------|-------------------|---------------|-------|
| Current | 50 | 4 | Baseline |
| Month 1 | 58 | 4 | Within capacity |
| Month 2 | 66 | 5 | Scale up needed |
| Month 3 | 76 | 6 | |
| Month 6 | 116 | 8 | Consider optimization |
| Month 12 | 267 | 12 | At max HPA limit |

**Action Items**:
- Month 2: Increase HPA max to 8 pods
- Month 6: Implement calculation optimization
- Month 12: Increase HPA max to 20 pods, consider sharding

---

## Horizontal Scaling

### Manual Scaling

**Scale Up**:
```bash
# Current replica count
kubectl get deployment thermaliq -n gl-009-production

# Scale to 8 replicas
kubectl scale deployment/thermaliq -n gl-009-production --replicas=8

# Wait for all pods to be ready
kubectl rollout status deployment/thermaliq -n gl-009-production

# Verify all pods running
kubectl get pods -n gl-009-production -l app=thermaliq

# Check resource usage
kubectl top pods -n gl-009-production -l app=thermaliq
```

**Scale Down**:
```bash
# Drain connections gracefully before scaling down
kubectl set env deployment/thermaliq -n gl-009-production \
  SHUTDOWN_GRACE_PERIOD=300

# Scale down
kubectl scale deployment/thermaliq -n gl-009-production --replicas=4

# Monitor pod termination
kubectl get pods -n gl-009-production -l app=thermaliq --watch

# Verify no dropped requests
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=rate(thermaliq_calculation_errors_total[5m])' | \
  jq '.data.result[0].value[1]'
```

### Replica Distribution

**Pod Anti-Affinity** (spread across nodes):
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: thermaliq
  namespace: gl-009-production
spec:
  replicas: 8
  selector:
    matchLabels:
      app: thermaliq
  template:
    metadata:
      labels:
        app: thermaliq
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchLabels:
                  app: thermaliq
              topologyKey: kubernetes.io/hostname
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: node-role.kubernetes.io/worker
                operator: In
                values:
                - "true"
      containers:
      - name: thermaliq
        image: ghcr.io/greenlang/thermaliq:latest
        resources:
          requests:
            cpu: 2000m
            memory: 4Gi
          limits:
            cpu: 4000m
            memory: 8Gi
```

Apply:
```bash
kubectl apply -f deployment_scaled.yaml
```

### Pod Disruption Budget

Prevent too many pods from being down simultaneously:

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: thermaliq-pdb
  namespace: gl-009-production
spec:
  minAvailable: 2  # Always keep at least 2 pods running
  selector:
    matchLabels:
      app: thermaliq
```

Apply:
```bash
kubectl apply -f pdb.yaml

# Verify PDB
kubectl get pdb -n gl-009-production
```

---

## Vertical Scaling

### Increase Resource Limits

**Increase CPU**:
```bash
# Current resources
kubectl get deployment thermaliq -n gl-009-production -o yaml | \
  grep -A 10 "resources:"

# Increase CPU limits
kubectl set resources deployment/thermaliq -n gl-009-production \
  --limits=cpu=6000m \
  --requests=cpu=3000m

# Wait for rollout
kubectl rollout status deployment/thermaliq -n gl-009-production

# Verify new limits
kubectl describe pod -n gl-009-production -l app=thermaliq | \
  grep -A 5 "Limits:"
```

**Increase Memory**:
```bash
# Increase memory limits
kubectl set resources deployment/thermaliq -n gl-009-production \
  --limits=memory=12Gi \
  --requests=memory=6Gi

# Wait for rollout
kubectl rollout status deployment/thermaliq -n gl-009-production

# Monitor memory usage
watch -n 30 'kubectl top pods -n gl-009-production -l app=thermaliq'
```

### Vertical Pod Autoscaler (VPA)

Install VPA (if not already installed):
```bash
kubectl apply -f https://github.com/kubernetes/autoscaler/releases/download/vertical-pod-autoscaler-0.13.0/vpa-v0.13.0.yaml
```

Create VPA for THERMALIQ:
```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: thermaliq-vpa
  namespace: gl-009-production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: thermaliq
  updatePolicy:
    updateMode: "Auto"  # Options: Off, Initial, Recreate, Auto
  resourcePolicy:
    containerPolicies:
    - containerName: thermaliq
      minAllowed:
        cpu: 1000m
        memory: 2Gi
      maxAllowed:
        cpu: 8000m
        memory: 16Gi
      controlledResources:
      - cpu
      - memory
```

Apply and monitor:
```bash
kubectl apply -f vpa.yaml

# Check VPA recommendations
kubectl describe vpa thermaliq-vpa -n gl-009-production

# Monitor VPA adjustments
kubectl get vpa thermaliq-vpa -n gl-009-production --watch
```

---

## Auto-Scaling

### Horizontal Pod Autoscaler (HPA)

**CPU-Based HPA**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: thermaliq-hpa
  namespace: gl-009-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: thermaliq
  minReplicas: 4
  maxReplicas: 12
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 25
        periodSeconds: 60
      selectPolicy: Min
```

**Memory-Based HPA**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: thermaliq-hpa-memory
  namespace: gl-009-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: thermaliq
  minReplicas: 4
  maxReplicas: 12
  metrics:
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**Custom Metrics HPA** (queue depth):
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: thermaliq-hpa-custom
  namespace: gl-009-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: thermaliq
  minReplicas: 4
  maxReplicas: 16
  metrics:
  - type: Pods
    pods:
      metric:
        name: thermaliq_calculation_queue_depth
      target:
        type: AverageValue
        averageValue: "10"
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

Apply and monitor:
```bash
# Apply HPA
kubectl apply -f hpa.yaml

# Check HPA status
kubectl get hpa thermaliq-hpa -n gl-009-production

# Watch HPA in action
kubectl get hpa thermaliq-hpa -n gl-009-production --watch

# Describe HPA for details
kubectl describe hpa thermaliq-hpa -n gl-009-production
```

### Custom Metrics Server

Set up Prometheus Adapter for custom metrics:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: adapter-config
  namespace: monitoring
data:
  config.yaml: |
    rules:
    - seriesQuery: 'thermaliq_calculation_queue_depth'
      resources:
        overrides:
          namespace: {resource: "namespace"}
          pod: {resource: "pod"}
      name:
        matches: "^thermaliq_calculation_queue_depth"
        as: "thermaliq_calculation_queue_depth"
      metricsQuery: 'avg_over_time(thermaliq_calculation_queue_depth[2m])'
```

Deploy Prometheus Adapter:
```bash
helm install prometheus-adapter prometheus-community/prometheus-adapter \
  -n monitoring \
  -f adapter-values.yaml

# Verify custom metrics available
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1 | jq .
```

---

## Database Scaling

### Read Replicas

Create read replica for query offloading:

```yaml
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: thermaliq-postgres
  namespace: gl-009-production
spec:
  instances: 3  # 1 primary + 2 replicas
  primaryUpdateStrategy: unsupervised
  postgresql:
    parameters:
      max_connections: "500"
      shared_buffers: "2GB"
      effective_cache_size: "6GB"
      maintenance_work_mem: "512MB"
      checkpoint_completion_target: "0.9"
      wal_buffers: "16MB"
      default_statistics_target: "100"
      random_page_cost: "1.1"
      effective_io_concurrency: "200"
      work_mem: "5242kB"
  storage:
    size: 100Gi
    storageClass: fast-ssd
  monitoring:
    enabled: true
```

Configure application to use read replicas:

```python
# config.py
DATABASE_URLS = {
    'primary': 'postgresql://user:pass@postgres-primary:5432/thermaliq',
    'replica': 'postgresql://user:pass@postgres-replica:5432/thermaliq'
}

# Use primary for writes
def get_write_connection():
    return create_engine(DATABASE_URLS['primary'])

# Use replica for reads
def get_read_connection():
    return create_engine(DATABASE_URLS['replica'])
```

### Connection Pooling with PgBouncer

Deploy PgBouncer:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pgbouncer
  namespace: gl-009-production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pgbouncer
  template:
    metadata:
      labels:
        app: pgbouncer
    spec:
      containers:
      - name: pgbouncer
        image: edoburu/pgbouncer:1.20.0
        env:
        - name: DATABASE_URL
          value: "postgres://thermaliq:password@postgres-service:5432/thermaliq_production"
        - name: POOL_MODE
          value: "transaction"
        - name: MAX_CLIENT_CONN
          value: "2000"
        - name: DEFAULT_POOL_SIZE
          value: "50"
        - name: MIN_POOL_SIZE
          value: "10"
        - name: RESERVE_POOL_SIZE
          value: "10"
        - name: RESERVE_POOL_TIMEOUT
          value: "3"
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 1Gi
        ports:
        - containerPort: 5432
---
apiVersion: v1
kind: Service
metadata:
  name: pgbouncer
  namespace: gl-009-production
spec:
  selector:
    app: pgbouncer
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
```

Update application to use PgBouncer:
```bash
kubectl set env deployment/thermaliq -n gl-009-production \
  DATABASE_HOST=pgbouncer \
  DATABASE_PORT=5432
```

### Database Sharding

For very large scale, implement sharding:

```python
# Shard by facility_id
def get_shard_for_facility(facility_id: str) -> str:
    shard_count = 4
    shard_num = hash(facility_id) % shard_count
    return f"postgresql://user:pass@postgres-shard-{shard_num}:5432/thermaliq"

# Usage
facility_id = "FAC-001"
db_url = get_shard_for_facility(facility_id)
engine = create_engine(db_url)
```

Create sharded databases:
```bash
for i in {0..3}; do
  kubectl apply -f - <<EOF
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: thermaliq-postgres-shard-$i
  namespace: gl-009-production
spec:
  instances: 2
  storage:
    size: 50Gi
EOF
done
```

---

## Cache Scaling

### Redis Cluster

Deploy Redis Cluster for horizontal scaling:

```yaml
apiVersion: redis.redis.opstreelabs.in/v1beta1
kind: RedisCluster
metadata:
  name: thermaliq-redis
  namespace: gl-009-production
spec:
  clusterSize: 6  # 3 masters + 3 replicas
  kubernetesConfig:
    image: redis:7.0
    imagePullPolicy: IfNotPresent
  redisExporter:
    enabled: true
    image: oliver006/redis_exporter:latest
  storage:
    volumeClaimTemplate:
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 20Gi
        storageClassName: fast-ssd
  resources:
    requests:
      cpu: 1000m
      memory: 4Gi
    limits:
      cpu: 2000m
      memory: 8Gi
```

Apply and configure:
```bash
kubectl apply -f redis-cluster.yaml

# Wait for cluster to be ready
kubectl wait --for=condition=ready rediscluster/thermaliq-redis -n gl-009-production --timeout=600s

# Update application to use Redis Cluster
kubectl set env deployment/thermaliq -n gl-009-production \
  REDIS_MODE=cluster \
  REDIS_CLUSTER_NODES=thermaliq-redis-0:6379,thermaliq-redis-1:6379,thermaliq-redis-2:6379
```

### Redis Sentinel (High Availability)

For HA with automatic failover:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-sentinel-config
  namespace: gl-009-production
data:
  sentinel.conf: |
    sentinel monitor thermaliq-master redis-master 6379 2
    sentinel down-after-milliseconds thermaliq-master 5000
    sentinel parallel-syncs thermaliq-master 1
    sentinel failover-timeout thermaliq-master 10000
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-sentinel
  namespace: gl-009-production
spec:
  serviceName: redis-sentinel
  replicas: 3
  selector:
    matchLabels:
      app: redis-sentinel
  template:
    metadata:
      labels:
        app: redis-sentinel
    spec:
      containers:
      - name: sentinel
        image: redis:7.0
        command:
        - redis-sentinel
        - /etc/redis/sentinel.conf
        volumeMounts:
        - name: config
          mountPath: /etc/redis
        ports:
        - containerPort: 26379
      volumes:
      - name: config
        configMap:
          name: redis-sentinel-config
```

### Multi-Tier Caching

Implement L1 (in-memory) + L2 (Redis) caching:

```python
# cache.py
from functools import wraps
import redis
import json

# L1 Cache (in-memory, per-pod)
l1_cache = {}
L1_MAX_SIZE = 1000
L1_TTL = 300  # 5 minutes

# L2 Cache (Redis, shared)
redis_client = redis.Redis(host='redis-service', port=6379)
L2_TTL = 3600  # 1 hour

def two_tier_cache(ttl_l1=L1_TTL, ttl_l2=L2_TTL):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = f"{func.__name__}:{args}:{kwargs}"

            # Check L1 cache
            if key in l1_cache:
                return l1_cache[key]

            # Check L2 cache
            l2_value = redis_client.get(key)
            if l2_value:
                value = json.loads(l2_value)
                # Populate L1
                if len(l1_cache) < L1_MAX_SIZE:
                    l1_cache[key] = value
                return value

            # Cache miss - compute value
            value = func(*args, **kwargs)

            # Store in L2
            redis_client.setex(key, ttl_l2, json.dumps(value))

            # Store in L1
            if len(l1_cache) < L1_MAX_SIZE:
                l1_cache[key] = value

            return value
        return wrapper
    return decorator

# Usage
@two_tier_cache(ttl_l1=300, ttl_l2=3600)
def get_calculation_result(calculation_id):
    return database.query(f"SELECT * FROM calculations WHERE id='{calculation_id}'")
```

---

## Load Balancing

### Ingress Configuration

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: thermaliq-ingress
  namespace: gl-009-production
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/limit-connections: "50"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.greenlang.io
    secretName: greenlang-tls
  rules:
  - host: api.greenlang.io
    http:
      paths:
      - path: /v1/thermaliq
        pathType: Prefix
        backend:
          service:
            name: thermaliq-service
            port:
              number: 8080
```

### Service Configuration

```yaml
apiVersion: v1
kind: Service
metadata:
  name: thermaliq-service
  namespace: gl-009-production
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
spec:
  type: LoadBalancer
  sessionAffinity: ClientIP  # Sticky sessions
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 3600
  selector:
    app: thermaliq
  ports:
  - name: http
    port: 8080
    targetPort: 8080
    protocol: TCP
```

### Load Balancing Strategies

**Round Robin** (default):
```yaml
spec:
  sessionAffinity: None
```

**Sticky Sessions** (for stateful operations):
```yaml
spec:
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800  # 3 hours
```

**Weighted Load Balancing** (using Istio):
```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: thermaliq-destination
  namespace: gl-009-production
spec:
  host: thermaliq-service
  trafficPolicy:
    loadBalancer:
      consistentHash:
        httpHeaderName: "x-facility-id"
  subsets:
  - name: v1
    labels:
      version: v1
  - name: v2
    labels:
      version: v2
---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: thermaliq-routing
  namespace: gl-009-production
spec:
  hosts:
  - thermaliq-service
  http:
  - match:
    - headers:
        x-beta-user:
          exact: "true"
    route:
    - destination:
        host: thermaliq-service
        subset: v2
      weight: 100
  - route:
    - destination:
        host: thermaliq-service
        subset: v1
      weight: 90
    - destination:
        host: thermaliq-service
        subset: v2
      weight: 10
```

---

## Multi-Region Deployment

### Global Load Balancing

Use cloud provider global load balancer or Cloudflare:

**AWS Global Accelerator**:
```bash
# Create accelerator
aws globalaccelerator create-accelerator \
  --name thermaliq-global \
  --ip-address-type IPV4 \
  --enabled

# Add endpoint groups for each region
aws globalaccelerator create-endpoint-group \
  --listener-arn $LISTENER_ARN \
  --endpoint-group-region us-east-1 \
  --traffic-dial-percentage 50

aws globalaccelerator create-endpoint-group \
  --listener-arn $LISTENER_ARN \
  --endpoint-group-region eu-west-1 \
  --traffic-dial-percentage 50
```

**Cloudflare Load Balancer**:
```bash
# Configure via Cloudflare dashboard
# Add origins:
# - us-east-1: api-us.greenlang.io
# - eu-west-1: api-eu.greenlang.io
# - ap-southeast-1: api-ap.greenlang.io

# Load balancing strategy: Geo-steering
# Health checks: /v1/thermaliq/health every 30s
```

### Regional Deployment

Deploy to multiple regions:

```bash
# Region 1: US East (Primary)
kubectl config use-context us-east-1
kubectl create namespace gl-009-production
kubectl apply -k deployment/overlays/us-east-1/

# Region 2: EU West (Secondary)
kubectl config use-context eu-west-1
kubectl create namespace gl-009-production
kubectl apply -k deployment/overlays/eu-west-1/

# Region 3: AP Southeast (Tertiary)
kubectl config use-context ap-southeast-1
kubectl create namespace gl-009-production
kubectl apply -k deployment/overlays/ap-southeast-1/
```

### Data Replication

**Database Replication** (PostgreSQL):
```bash
# Primary in us-east-1
# Replicas in eu-west-1 and ap-southeast-1

# On primary
kubectl exec -it postgres-primary-0 -n gl-009-production -- \
  psql -U postgres -c "SELECT pg_create_physical_replication_slot('replica_eu');"

kubectl exec -it postgres-primary-0 -n gl-009-production -- \
  psql -U postgres -c "SELECT pg_create_physical_replication_slot('replica_ap');"

# On replicas
# Update postgresql.conf:
# primary_conninfo = 'host=postgres-primary-us port=5432 user=replicator password=xxx'
# primary_slot_name = 'replica_eu'
```

**Redis Replication**:
```yaml
# Use Redis Cluster with nodes in multiple regions
# Or use Redis Enterprise for active-active replication
```

### Failover Strategy

**Active-Active**:
- All regions serve traffic
- Global load balancer distributes based on geography
- Eventual consistency for data

**Active-Passive**:
- Primary region serves all traffic
- Secondary regions standby
- Failover on primary failure

**Failover Procedure**:
```bash
# Detect primary region failure
# Update DNS to point to secondary region
# Or update load balancer to remove failed region

# AWS Route53 health check failover
aws route53 change-resource-record-sets \
  --hosted-zone-id $ZONE_ID \
  --change-batch '{
    "Changes": [{
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "api.greenlang.io",
        "Type": "A",
        "SetIdentifier": "Primary",
        "Failover": "PRIMARY",
        "AliasTarget": {
          "HostedZoneId": "$ELB_ZONE_ID",
          "DNSName": "api-us.greenlang.io",
          "EvaluateTargetHealth": true
        }
      }
    }]
  }'
```

---

## Performance Optimization

### Application Optimization

**1. Enable Calculation Parallelization**:
```python
# Before (sequential)
results = []
for facility in facilities:
    result = calculate_efficiency(facility)
    results.append(result)

# After (parallel)
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(calculate_efficiency, facilities))
```

**2. Optimize Database Queries**:
```sql
-- Before (N+1 queries)
SELECT * FROM facilities;
-- For each facility:
--   SELECT * FROM energy_readings WHERE facility_id = ?;

-- After (single query with join)
SELECT f.*, er.*
FROM facilities f
LEFT JOIN energy_readings er ON er.facility_id = f.id
WHERE f.status = 'active';
```

**3. Implement Query Batching**:
```python
# Batch historian queries
def fetch_energy_data_batch(facility_ids, start_time, end_time):
    # Single API call for multiple facilities
    response = historian_client.query_batch(
        facility_ids=facility_ids,
        start=start_time,
        end=end_time
    )
    return response

# Instead of individual queries per facility
```

**4. Use Async I/O**:
```python
import asyncio
import aiohttp

async def fetch_energy_meter_data_async(facility_id):
    async with aiohttp.ClientSession() as session:
        async with session.get(f'https://meter-api/facilities/{facility_id}') as response:
            return await response.json()

async def process_facilities_async(facility_ids):
    tasks = [fetch_energy_meter_data_async(fid) for fid in facility_ids]
    results = await asyncio.gather(*tasks)
    return results
```

### Database Optimization

**1. Add Indexes**:
```sql
-- Index for facility + time range queries
CREATE INDEX CONCURRENTLY idx_energy_readings_facility_time
ON energy_readings(facility_id, timestamp DESC);

-- Index for calculation lookups
CREATE INDEX CONCURRENTLY idx_calculations_facility_created
ON calculations(facility_id, created_at DESC);

-- Partial index for active calculations
CREATE INDEX CONCURRENTLY idx_calculations_active
ON calculations(status, created_at)
WHERE status IN ('pending', 'processing');

-- Covering index (includes all needed columns)
CREATE INDEX CONCURRENTLY idx_energy_readings_covering
ON energy_readings(facility_id, timestamp DESC)
INCLUDE (value, unit, quality);
```

**2. Partition Large Tables**:
```sql
-- Partition by month
CREATE TABLE energy_readings_2025_11 PARTITION OF energy_readings
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

CREATE TABLE energy_readings_2025_12 PARTITION OF energy_readings
FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

-- Auto-create partitions
CREATE EXTENSION pg_partman;
SELECT create_parent('public.energy_readings', 'timestamp', 'native', 'monthly');
```

**3. Optimize Queries**:
```sql
-- Before
SELECT * FROM energy_readings
WHERE facility_id = 'FAC-001'
  AND timestamp >= NOW() - INTERVAL '7 days'
ORDER BY timestamp DESC;

-- After (with limit, only needed columns)
SELECT id, timestamp, value, unit
FROM energy_readings
WHERE facility_id = 'FAC-001'
  AND timestamp >= NOW() - INTERVAL '7 days'
ORDER BY timestamp DESC
LIMIT 10000;
```

**4. Use Connection Pooling**:
```python
# SQLAlchemy with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

### Caching Optimization

**1. Cache Warm-up**:
```bash
# Warm cache on deployment
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: cache-warmup
  namespace: gl-009-production
spec:
  template:
    spec:
      containers:
      - name: warmup
        image: ghcr.io/greenlang/thermaliq:latest
        command:
        - python
        - /app/scripts/warm_cache.py
        - --top-facilities=100
        - --days=30
      restartPolicy: Never
EOF
```

**2. Optimize Cache Keys**:
```python
# Before (too specific, low hit rate)
cache_key = f"calculation:{facility_id}:{start_time}:{end_time}:{precision}"

# After (hierarchical, better hit rate)
cache_key = f"calculation:{facility_id}:{start_time.date()}:hourly"
```

**3. Use Cache Compression**:
```python
import zlib
import json

def cache_set_compressed(key, value, ttl=3600):
    json_data = json.dumps(value)
    compressed = zlib.compress(json_data.encode())
    redis_client.setex(key, ttl, compressed)

def cache_get_compressed(key):
    compressed = redis_client.get(key)
    if compressed:
        json_data = zlib.decompress(compressed).decode()
        return json.loads(json_data)
    return None
```

### Network Optimization

**1. Enable HTTP/2**:
```yaml
apiVersion: v1
kind: Service
metadata:
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-backend-protocol: "http2"
```

**2. Use gRPC for Internal Communication**:
```python
# Instead of REST APIs between services
# Use gRPC for better performance

# service.proto
service ThermalIQService {
  rpc CalculateEfficiency (CalculationRequest) returns (CalculationResponse);
}
```

**3. Enable Response Compression**:
```python
from flask import Flask
from flask_compress import Compress

app = Flask(__name__)
Compress(app)

# Responses automatically compressed with gzip
```

---

## Scaling Checklist

### Before Scaling Up

- [ ] Verify current resource utilization
- [ ] Check available node capacity
- [ ] Review cost implications
- [ ] Test with synthetic load
- [ ] Backup current configuration
- [ ] Notify stakeholders
- [ ] Schedule maintenance window (if needed)

### During Scaling

- [ ] Monitor pod rollout
- [ ] Check health endpoints
- [ ] Verify load distribution
- [ ] Monitor error rates
- [ ] Check resource usage
- [ ] Verify database connections
- [ ] Test end-to-end functionality

### After Scaling

- [ ] Verify all pods healthy
- [ ] Check performance metrics
- [ ] Load test new capacity
- [ ] Update documentation
- [ ] Update capacity planning
- [ ] Review cost vs performance
- [ ] Schedule follow-up review

---

**Document Version**: 1.0.0
**Last Updated**: 2025-11-26
**Next Review**: 2026-01-26
**Owner**: GreenLang SRE Team
