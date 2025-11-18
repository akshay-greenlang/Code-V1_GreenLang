# GL-002 BoilerEfficiencyOptimizer - Scaling Guide

Comprehensive scaling operations guide for GL-002 BoilerEfficiencyOptimizer production deployments. This runbook covers horizontal scaling, vertical scaling, and performance optimization.

## Table of Contents

1. [Scaling Overview](#scaling-overview)
2. [When to Scale](#when-to-scale)
3. [Horizontal Scaling (HPA)](#horizontal-scaling-hpa)
4. [Vertical Scaling (Resources)](#vertical-scaling-resources)
5. [Database Scaling](#database-scaling)
6. [Cache Scaling (Redis)](#cache-scaling-redis)
7. [Multi-Region Deployment](#multi-region-deployment)
8. [Performance Testing](#performance-testing)
9. [Capacity Planning](#capacity-planning)
10. [Cost Optimization](#cost-optimization)

---

## Scaling Overview

### Current Architecture

```
Production Baseline (3 pods):
- CPU: 500m request, 1000m limit per pod
- Memory: 512Mi request, 1024Mi limit per pod
- Total capacity: 1500m CPU, 1.5Gi Memory

Auto-Scaling (HPA):
- Min replicas: 3
- Max replicas: 10
- Scale up trigger: 70% CPU or 80% Memory
- Scale down trigger: <50% CPU and <60% Memory
- Scale down stabilization: 5 minutes
```

### Scaling Dimensions

**Horizontal Scaling** (Add more pods)
- Best for: Increased request throughput
- Pros: Better availability, distributes load
- Cons: More complex state management
- Cost: Linear increase

**Vertical Scaling** (Bigger pods)
- Best for: Memory-intensive operations, large calculations
- Pros: Simpler, fewer moving parts
- Cons: Limited by node size, single point of failure
- Cost: Can be more expensive per unit

**Database Scaling**
- Read replicas for read-heavy workloads
- Connection pooling optimization
- Query optimization

**Cache Scaling**
- Redis Cluster for distributed caching
- Increased memory allocation
- Improved hit rates

---

## When to Scale

### Scaling Triggers

#### Immediate Scaling Required (P1)

Execute scaling within 15 minutes if:

```bash
# Check CPU usage
kubectl top pods -n greenlang -l app=gl-002-boiler-efficiency

# Trigger: Average CPU >85% across all pods for >5 minutes
# Action: Scale up immediately

# Check memory usage
kubectl top pods -n greenlang -l app=gl-002-boiler-efficiency

# Trigger: Any pod >90% memory usage
# Action: Scale up or increase limits immediately

# Check error rate
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep error_rate

# Trigger: Error rate >5% due to resource constraints
# Action: Scale up immediately

# Check response time
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep duration

# Trigger: p95 response time >5 seconds
# Action: Investigate and scale if needed
```

#### Planned Scaling (P2)

Schedule scaling for:

- **Predictable Load Increases**: Marketing campaigns, end-of-month reporting
- **Seasonal Patterns**: Higher usage during winter (heating season)
- **Growth Projections**: Onboarding new customers
- **Performance Optimization**: Proactive scaling before issues occur

#### Monitoring Metrics

```bash
# CPU utilization
kubectl top pods -n greenlang -l app=gl-002-boiler-efficiency | awk '{sum+=$2; count++} END {print "Average CPU:", sum/count "m"}'

# Memory utilization
kubectl top pods -n greenlang -l app=gl-002-boiler-efficiency | awk '{sum+=$3; count++} END {print "Average Memory:", sum/count "Mi"}'

# Request rate (from Prometheus)
curl -s "http://prometheus:9090/api/v1/query?query=rate(gl_002_http_requests_total[5m])" | jq

# Pod count
kubectl get deployment gl-002-boiler-efficiency -n greenlang -o jsonpath='{.status.replicas}'

# Queue depth (if applicable)
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep queue_depth
```

---

## Horizontal Scaling (HPA)

### Manual Horizontal Scaling

#### Quick Scale Up

```bash
# Scale to specific replica count
kubectl scale deployment gl-002-boiler-efficiency --replicas=5 -n greenlang

# Watch scaling progress
kubectl get pods -n greenlang -l app=gl-002-boiler-efficiency -w

# Verify all pods are ready
kubectl get deployment gl-002-boiler-efficiency -n greenlang

# Expected output:
# NAME                       READY   UP-TO-DATE   AVAILABLE   AGE
# gl-002-boiler-efficiency   5/5     5            5           10d
```

**Scaling Timeline**:
- 0:00 - Scale command executed
- 0:30 - New pods created, pulling image (if not cached)
- 1:00 - Pods starting, running init containers
- 1:30 - Application starting, health checks beginning
- 2:00 - Health checks passing, pods added to service
- 2:30 - All 5 pods ready and serving traffic

#### Quick Scale Down

```bash
# Scale down to reduce costs during low usage
kubectl scale deployment gl-002-boiler-efficiency --replicas=3 -n greenlang

# Watch scale down
kubectl get pods -n greenlang -l app=gl-002-boiler-efficiency -w

# Verify graceful termination
kubectl logs -n greenlang <terminating-pod-name> --previous
```

**Important**: Scale down gradually during business hours to avoid service disruption.

### Automatic Horizontal Scaling (HPA)

#### Enable/Update HPA

```bash
# Apply HPA configuration
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gl-002-boiler-efficiency-hpa
  namespace: greenlang
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gl-002-boiler-efficiency
  minReplicas: 3
  maxReplicas: 10
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
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scaling down
      policies:
      - type: Percent
        value: 50  # Scale down max 50% at a time
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0  # Scale up immediately
      policies:
      - type: Percent
        value: 100  # Can double pods at once
        periodSeconds: 60
      - type: Pods
        value: 2  # Or add max 2 pods at once
        periodSeconds: 60
      selectPolicy: Max  # Use the policy that scales most
EOF

# Verify HPA is active
kubectl get hpa gl-002-boiler-efficiency-hpa -n greenlang

# Expected output:
# NAME                           REFERENCE                             TARGETS                        MINPODS   MAXPODS   REPLICAS   AGE
# gl-002-boiler-efficiency-hpa   Deployment/gl-002-boiler-efficiency   45%/70%, 55%/80%               3         10        3          10d
#                                                                       CPU      Memory
```

#### Monitor HPA Activity

```bash
# Watch HPA in real-time
kubectl get hpa gl-002-boiler-efficiency-hpa -n greenlang -w

# View HPA events
kubectl describe hpa gl-002-boiler-efficiency-hpa -n greenlang | tail -20

# Check HPA decision logs
kubectl get hpa gl-002-boiler-efficiency-hpa -n greenlang -o yaml | grep -A 10 conditions

# View scaling events
kubectl get events -n greenlang --sort-by='.lastTimestamp' | grep -i scale
```

#### Adjust HPA Thresholds

```bash
# More aggressive scaling (scale up earlier)
kubectl patch hpa gl-002-boiler-efficiency-hpa -n greenlang --type='json' -p='[
  {"op": "replace", "path": "/spec/metrics/0/resource/target/averageUtilization", "value": 60}
]'

# More conservative scaling (allow higher utilization)
kubectl patch hpa gl-002-boiler-efficiency-hpa -n greenlang --type='json' -p='[
  {"op": "replace", "path": "/spec/metrics/0/resource/target/averageUtilization", "value": 80}
]'

# Increase max replicas for higher capacity
kubectl patch hpa gl-002-boiler-efficiency-hpa -n greenlang --type='json' -p='[
  {"op": "replace", "path": "/spec/maxReplicas", "value": 15}
]'

# Verify changes
kubectl get hpa gl-002-boiler-efficiency-hpa -n greenlang -o yaml
```

#### Custom Metrics HPA (Advanced)

Scale based on custom application metrics:

```bash
# Scale based on optimization queue depth
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gl-002-boiler-efficiency-hpa-custom
  namespace: greenlang
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gl-002-boiler-efficiency
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: gl_002_optimization_queue_depth
      target:
        type: AverageValue
        averageValue: "50"  # Scale when avg queue depth >50 per pod
  - type: Pods
    pods:
      metric:
        name: gl_002_http_request_duration_seconds
      target:
        type: AverageValue
        averageValue: "2"  # Scale when avg response time >2s
EOF
```

---

## Vertical Scaling (Resources)

### Increase Resource Limits

#### Immediate Resource Increase (Emergency)

```bash
# Increase CPU limits (handles CPU throttling)
kubectl set resources deployment gl-002-boiler-efficiency -n greenlang \
  --limits=cpu=2000m \
  --requests=cpu=1000m

# Increase memory limits (handles OOMKilled)
kubectl set resources deployment gl-002-boiler-efficiency -n greenlang \
  --limits=memory=2Gi \
  --requests=memory=1Gi

# Increase both (comprehensive)
kubectl set resources deployment gl-002-boiler-efficiency -n greenlang \
  --limits=cpu=2000m,memory=2Gi \
  --requests=cpu=1000m,memory=1Gi

# Verify changes applied
kubectl get deployment gl-002-boiler-efficiency -n greenlang -o yaml | grep -A 6 resources:

# Monitor rollout
kubectl rollout status deployment/gl-002-boiler-efficiency -n greenlang

# Check if pods are running with new limits
kubectl describe pod -n greenlang <pod-name> | grep -A 10 "Limits:"
```

**Resource Increase Timeline**:
- 0:00 - Command executed
- 0:10 - Deployment updated, rolling update triggered
- 0:30 - First new pod created with higher limits
- 1:00 - First new pod ready, old pod terminating
- 2:00 - Second pod updated
- 3:00 - Third pod updated
- 3:30 - All pods running with new limits

#### Gradual Resource Adjustment

```bash
# Step 1: Test with single pod
kubectl patch deployment gl-002-boiler-efficiency -n greenlang --type='json' -p='[
  {"op": "add", "path": "/spec/template/metadata/labels/test-resources", "value": "true"}
]'

# Deploy test pod with higher resources
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-002-boiler-efficiency-test
  namespace: greenlang
spec:
  replicas: 1
  selector:
    matchLabels:
      app: gl-002-boiler-efficiency-test
  template:
    metadata:
      labels:
        app: gl-002-boiler-efficiency-test
    spec:
      containers:
      - name: boiler-optimizer
        image: ghcr.io/greenlang/gl-002:latest
        resources:
          requests:
            cpu: 2000m  # Double current
            memory: 2Gi
          limits:
            cpu: 4000m
            memory: 4Gi
EOF

# Monitor test pod performance
kubectl top pod -n greenlang -l app=gl-002-boiler-efficiency-test

# If successful, apply to main deployment
kubectl set resources deployment gl-002-boiler-efficiency -n greenlang \
  --limits=cpu=4000m,memory=4Gi \
  --requests=cpu=2000m,memory=2Gi

# Cleanup test deployment
kubectl delete deployment gl-002-boiler-efficiency-test -n greenlang
```

### Resource Right-Sizing

#### Analyze Current Usage

```bash
# Check average resource usage over time
kubectl top pods -n greenlang -l app=gl-002-boiler-efficiency

# Get resource usage from Prometheus (last 24 hours)
curl -s "http://prometheus:9090/api/v1/query?query=avg_over_time(container_cpu_usage_seconds_total{pod=~'gl-002.*'}[24h])" | jq

curl -s "http://prometheus:9090/api/v1/query?query=avg_over_time(container_memory_working_set_bytes{pod=~'gl-002.*'}[24h])" | jq

# Calculate recommendations
kubectl describe node <node-name> | grep -A 20 "Allocated resources"
```

**Right-Sizing Formula**:
```
Request = P95 usage + 20% buffer
Limit = Request × 2
```

**Example**:
```
P95 CPU usage: 600m
Request: 600m × 1.2 = 720m → Round to 750m
Limit: 750m × 2 = 1500m

P95 Memory usage: 800Mi
Request: 800Mi × 1.2 = 960Mi → Round to 1Gi
Limit: 1Gi × 2 = 2Gi
```

---

## Database Scaling

### Connection Pool Scaling

```bash
# Check current connection pool settings
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- env | grep DATABASE

# Check connection pool metrics
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep db_pool

# Increase connection pool size
kubectl edit configmap gl-002-config -n greenlang

# Modify:
#   DATABASE_POOL_SIZE: "40"  # from 20
#   DATABASE_MAX_OVERFLOW: "20"  # from 10
#   DATABASE_POOL_TIMEOUT: "30"
#   DATABASE_POOL_RECYCLE: "3600"

# Restart deployment to apply
kubectl rollout restart deployment/gl-002-boiler-efficiency -n greenlang

# Verify new settings
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep -E '(db_pool_size|db_connections_active)'
```

### Read Replica Setup (AWS RDS)

```bash
# Create read replica
aws rds create-db-instance-read-replica \
  --db-instance-identifier gl-002-prod-read-replica \
  --source-db-instance-identifier gl-002-prod \
  --db-instance-class db.r5.xlarge \
  --availability-zone us-east-1b

# Wait for replica to be available (10-20 minutes)
aws rds wait db-instance-available --db-instance-identifier gl-002-prod-read-replica

# Get replica endpoint
aws rds describe-db-instances \
  --db-instance-identifier gl-002-prod-read-replica \
  --query 'DBInstances[0].Endpoint.Address' \
  --output text

# Update application to use read replica
kubectl create secret generic gl-002-secrets \
  --from-literal=DATABASE_READ_URL='postgresql://user:pass@read-replica-endpoint:5432/boiler' \
  --dry-run=client -o yaml | kubectl apply -n greenlang -f -

# Enable read replica in application
kubectl edit configmap gl-002-config -n greenlang

# Add:
#   DATABASE_READ_REPLICA_ENABLED: "true"
#   DATABASE_READ_REPLICA_RATIO: "0.7"  # 70% reads go to replica

# Restart
kubectl rollout restart deployment/gl-002-boiler-efficiency -n greenlang
```

### Database Query Optimization

```bash
# Identify slow queries
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  psql $DATABASE_URL -c "
    SELECT query, mean_exec_time, calls
    FROM pg_stat_statements
    ORDER BY mean_exec_time DESC
    LIMIT 10;
  "

# Add missing indexes
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  psql $DATABASE_URL -c "
    CREATE INDEX CONCURRENTLY idx_boiler_id_timestamp
    ON optimization_results(boiler_id, created_at DESC);
  "

# Vacuum and analyze
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  psql $DATABASE_URL -c "VACUUM ANALYZE;"

# Update table statistics
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  psql $DATABASE_URL -c "ANALYZE VERBOSE;"
```

---

## Cache Scaling (Redis)

### Increase Redis Memory

```bash
# Check current Redis memory usage
kubectl exec -n greenlang deployment/redis -- redis-cli INFO memory

# Increase Redis memory limits
kubectl set resources deployment redis -n greenlang \
  --limits=memory=4Gi \
  --requests=memory=2Gi

# Configure Redis max memory
kubectl exec -n greenlang deployment/redis -- \
  redis-cli CONFIG SET maxmemory 3gb

kubectl exec -n greenlang deployment/redis -- \
  redis-cli CONFIG SET maxmemory-policy allkeys-lru

# Make changes persistent
kubectl exec -n greenlang deployment/redis -- \
  redis-cli CONFIG REWRITE

# Verify
kubectl exec -n greenlang deployment/redis -- \
  redis-cli CONFIG GET maxmemory
```

### Redis Cluster (High Availability)

```bash
# Deploy Redis Cluster (6 nodes: 3 masters + 3 replicas)
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-cluster
  namespace: greenlang
spec:
  serviceName: redis-cluster
  replicas: 6
  selector:
    matchLabels:
      app: redis-cluster
  template:
    metadata:
      labels:
        app: redis-cluster
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command:
        - redis-server
        - --cluster-enabled
        - "yes"
        - --cluster-config-file
        - /data/nodes.conf
        - --cluster-node-timeout
        - "5000"
        - --appendonly
        - "yes"
        - --maxmemory
        - "2gb"
        - --maxmemory-policy
        - allkeys-lru
        ports:
        - containerPort: 6379
          name: client
        - containerPort: 16379
          name: gossip
        volumeMounts:
        - name: data
          mountPath: /data
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 1000m
            memory: 2Gi
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
EOF

# Create Redis Cluster headless service
kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: redis-cluster
  namespace: greenlang
spec:
  clusterIP: None
  ports:
  - port: 6379
    targetPort: 6379
    name: client
  - port: 16379
    targetPort: 16379
    name: gossip
  selector:
    app: redis-cluster
EOF

# Initialize cluster (after all pods are ready)
kubectl exec -n greenlang redis-cluster-0 -- redis-cli --cluster create \
  $(kubectl get pods -n greenlang -l app=redis-cluster -o jsonpath='{range.items[*]}{.status.podIP}:6379 ') \
  --cluster-replicas 1

# Update application to use Redis Cluster
kubectl edit configmap gl-002-config -n greenlang

# Modify:
#   REDIS_MODE: "cluster"
#   REDIS_CLUSTER_NODES: "redis-cluster-0.redis-cluster:6379,redis-cluster-1.redis-cluster:6379,redis-cluster-2.redis-cluster:6379"

# Restart application
kubectl rollout restart deployment/gl-002-boiler-efficiency -n greenlang
```

### Optimize Cache Hit Rate

```bash
# Check cache hit rate
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep cache_hit_rate

# Target: >80% hit rate

# Increase cache TTL for stable data
kubectl edit configmap gl-002-config -n greenlang

# Modify:
#   CACHE_TTL_SECONDS: "7200"  # 2 hours (from 3600)
#   CACHE_OPTIMIZATION_RESULTS_TTL: "14400"  # 4 hours
#   CACHE_BOILER_CONFIG_TTL: "86400"  # 24 hours (rarely changes)

# Warm up cache on startup
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -X POST http://localhost:8000/api/v1/cache/warmup

# Monitor hit rate improvement
watch -n 60 'kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep cache_hit_rate'
```

---

## Multi-Region Deployment

### Active-Active Multi-Region

```bash
# Deploy to US-EAST-1 (primary)
export AWS_REGION=us-east-1
kubectl config use-context eks-us-east-1

kubectl apply -f deployment/deployment.yaml -n greenlang
kubectl apply -f deployment/service.yaml -n greenlang
kubectl apply -f deployment/ingress.yaml -n greenlang

# Deploy to EU-WEST-1 (secondary)
export AWS_REGION=eu-west-1
kubectl config use-context eks-eu-west-1

kubectl apply -f deployment/deployment.yaml -n greenlang
kubectl apply -f deployment/service.yaml -n greenlang
kubectl apply -f deployment/ingress.yaml -n greenlang

# Configure Global Load Balancer (AWS Route53)
aws route53 change-resource-record-sets --hosted-zone-id Z1234567890ABC --change-batch '{
  "Changes": [
    {
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "api.boiler.greenlang.io",
        "Type": "A",
        "SetIdentifier": "US-EAST-1",
        "Region": "us-east-1",
        "AliasTarget": {
          "HostedZoneId": "Z1234567890ABC",
          "DNSName": "us-east-1-lb.elb.amazonaws.com",
          "EvaluateTargetHealth": true
        }
      }
    },
    {
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "api.boiler.greenlang.io",
        "Type": "A",
        "SetIdentifier": "EU-WEST-1",
        "Region": "eu-west-1",
        "AliasTarget": {
          "HostedZoneId": "Z0987654321XYZ",
          "DNSName": "eu-west-1-lb.elb.amazonaws.com",
          "EvaluateTargetHealth": true
        }
      }
    }
  ]
}'

# Configure database replication between regions
# (AWS RDS cross-region read replica)
aws rds create-db-instance-read-replica \
  --db-instance-identifier gl-002-eu-west-1-replica \
  --source-db-instance-identifier arn:aws:rds:us-east-1:123456789012:db:gl-002-prod \
  --db-instance-class db.r5.large \
  --region eu-west-1
```

---

## Performance Testing

### Load Testing

```bash
# Install k6 load testing tool
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: k6-script
  namespace: greenlang
data:
  load-test.js: |
    import http from 'k6/http';
    import { check, sleep } from 'k6';

    export let options = {
      stages: [
        { duration: '2m', target: 50 },   // Ramp up to 50 users
        { duration: '5m', target: 50 },   // Stay at 50 users
        { duration: '2m', target: 100 },  // Ramp up to 100 users
        { duration: '5m', target: 100 },  // Stay at 100 users
        { duration: '2m', target: 0 },    // Ramp down
      ],
      thresholds: {
        http_req_duration: ['p(95)<2000'],  // 95% of requests <2s
        http_req_failed: ['rate<0.01'],     // <1% error rate
      },
    };

    export default function() {
      let response = http.post('http://gl-002-boiler-efficiency.greenlang.svc.cluster.local/api/v1/boiler/optimize',
        JSON.stringify({
          boiler_id: 'test-boiler-001',
          test_mode: true
        }),
        { headers: { 'Content-Type': 'application/json' } }
      );

      check(response, {
        'status is 200': (r) => r.status === 200,
        'response time <2s': (r) => r.timings.duration < 2000,
      });

      sleep(1);
    }
EOF

# Run load test
kubectl run k6 --image=grafana/k6 --restart=Never -n greenlang \
  --command -- k6 run --vus 100 --duration 15m /scripts/load-test.js \
  --volume-mount configmap/k6-script:/scripts

# Monitor during load test
kubectl top pods -n greenlang -l app=gl-002-boiler-efficiency
kubectl get hpa -n greenlang -w
```

### Stress Testing

```bash
# Stress test to find breaking point
kubectl run k6-stress --image=grafana/k6 --restart=Never -n greenlang \
  --command -- k6 run --vus 500 --duration 10m /scripts/stress-test.js

# Monitor for failures
kubectl logs -f -n greenlang deployment/gl-002-boiler-efficiency | grep -i error

# Check when system starts degrading
# - Response time >5s
# - Error rate >5%
# - OOMKilled events
# - CPU throttling
```

---

## Capacity Planning

### Estimate Required Resources

```bash
# Calculate based on expected load

# Baseline capacity per pod:
# - 100 requests/minute at <1s response time
# - 50m CPU usage per 10 req/min
# - 100Mi memory usage base + 10Mi per concurrent request

# Example: Need to handle 1000 req/min
# Required pods: 1000 / 100 = 10 pods
# CPU per pod: 50m × (1000/10)/10 = 500m
# Memory per pod: 100Mi + 10Mi × 10 = 200Mi

# Apply capacity
kubectl scale deployment gl-002-boiler-efficiency --replicas=10 -n greenlang

kubectl set resources deployment gl-002-boiler-efficiency -n greenlang \
  --requests=cpu=500m,memory=256Mi \
  --limits=cpu=1000m,memory=512Mi
```

### Growth Planning

```
Month 1:  100 boilers,  1K req/min  → 3 pods
Month 3:  250 boilers,  2.5K req/min → 5 pods
Month 6:  500 boilers,  5K req/min  → 10 pods
Month 12: 1000 boilers, 10K req/min → 20 pods
```

---

## Cost Optimization

### Reduce Costs During Off-Peak

```bash
# Schedule scale down during off-peak (example: nights)
# Use CronJob to scale

kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: gl-002-scale-down-night
  namespace: greenlang
spec:
  schedule: "0 20 * * *"  # 8 PM UTC
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: scaler
          containers:
          - name: kubectl
            image: bitnami/kubectl
            command:
            - /bin/sh
            - -c
            - kubectl scale deployment gl-002-boiler-efficiency --replicas=1 -n greenlang
          restartPolicy: OnFailure
---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: gl-002-scale-up-morning
  namespace: greenlang
spec:
  schedule: "0 6 * * *"  # 6 AM UTC
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: scaler
          containers:
          - name: kubectl
            image: bitnami/kubectl
            command:
            - /bin/sh
            - -c
            - kubectl scale deployment gl-002-boiler-efficiency --replicas=3 -n greenlang
          restartPolicy: OnFailure
EOF
```

### Spot Instances (AWS)

```bash
# Use spot instances for non-critical pods
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-002-boiler-efficiency-spot
  namespace: greenlang
spec:
  replicas: 5
  selector:
    matchLabels:
      app: gl-002-boiler-efficiency-spot
  template:
    metadata:
      labels:
        app: gl-002-boiler-efficiency-spot
    spec:
      nodeSelector:
        node.kubernetes.io/instance-type: t3.large
        eks.amazonaws.com/capacityType: SPOT
      tolerations:
      - key: "spot"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
      containers:
      - name: boiler-optimizer
        image: ghcr.io/greenlang/gl-002:latest
        # ... rest of spec
EOF
```

---

## Scaling Checklist

### Pre-Scaling

- [ ] Check current resource utilization
- [ ] Identify bottleneck (CPU, memory, database, cache)
- [ ] Review recent scaling history
- [ ] Estimate required capacity
- [ ] Plan rollback strategy

### During Scaling

- [ ] Monitor pod rollout
- [ ] Check health endpoints
- [ ] Monitor error rates
- [ ] Watch resource usage
- [ ] Verify HPA (if enabled)

### Post-Scaling

- [ ] Verify all pods healthy
- [ ] Run smoke tests
- [ ] Monitor for 30 minutes
- [ ] Check cost impact
- [ ] Document scaling event

---

## Additional Resources

- **Troubleshooting Guide**: `TROUBLESHOOTING.md`
- **Incident Response**: `INCIDENT_RESPONSE.md`
- **Rollback Procedure**: `ROLLBACK_PROCEDURE.md`
- **Deployment Guide**: `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-002\deployment\README.md`
- **HPA Documentation**: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
- **Resource Management**: https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/
- **Grafana Dashboards**: https://grafana.greenlang.io/d/gl-002-scaling
